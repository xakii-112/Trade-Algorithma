"""BULGU-33-1/2/3 (Tur-33) — DEJENERE girdide yön UYDURULMAMALI.

Tur-28 dersi ("bir göstergeyi/skoru dejenere girdilerle SINA") o turda YALNIZ
`strategy.rsi` + `strategy._bias_and_confidence` üzerinde uygulanmıştı. Aynı desenin
KARDEŞ ÇAĞIRANLARI — `indicators.indicator_panel` ve `indicators_ext.extended_rows` —
dokunulmadan kalmıştı ve toplam 51 karşılaştırma `1 if x > y else -1` biçimindeydi.

ÖLÇÜLEN (düzeltme öncesi, tamamen düz 120 barlık pencere = SIFIR bilgi):
    bull=11 · bear=46 · consensus=**−0,542** → etiket **"GÜÇLÜ SAT"**
Yani halt olmuş / illikit / seans-dışı bir isimde pano, hiçbir bilgi yokken operatöre
güçlü bir SATIŞ görüşü gösteriyordu. Üç ayrı kök neden birbirini besliyordu:

  1. eşitlik ve NaN → −1 (ayı)          → `strategy.vote` (üç durumlu, tek kaynak)
  2. `mfi` 0/0 → 100 ("azami aşırı alım") → `rsi`in Tur-28 düzeltmesinin birebir kardeşi
  3. `spearman` tam-bağlı pencerede +50   → sıra varyansı yokken katsayı tanımsızdır
     ayrıca `supertrend` ATR=0'da −1, `%B` span=0'da 0.5 uyduruyordu.

Bu testler ölçümü KİLİTLER: düz seride yön oyu veren gösterge sayısı SIFIR olmalı ve
normal seride panel düzeltme öncesiyle BİT-DENK kalmalı (fix yalnız dejenere yolu kapatır).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.signals import indicators as ind
from nihai.signals import indicators_ext as ext
from nihai.signals.strategy import note3, vote


def _flat_df(n: int = 120, price: float = 100.0, volume: float = 1000.0) -> pd.DataFrame:
    """Tamamen DÜZ OHLCV — halt / illikit isim / seans-dışı barın uç hâli."""
    col = np.full(n, price, dtype=float)
    return pd.DataFrame(
        {"open": col, "high": col, "low": col, "close": col, "volume": np.full(n, volume)},
        index=pd.date_range("2026-01-01 09:30", periods=n, freq="1min"),
    )


def _normal_df(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) * 1.0005,
         "low": np.minimum(open_, close) * 0.9995, "close": close,
         "volume": rng.uniform(800, 1200, n)},
        index=pd.date_range("2026-01-01 09:30", periods=n, freq="1min"),
    )


# --- 1) Üç durumlu oy kuralının kendisi ------------------------------------------
@pytest.mark.parametrize(("x", "y", "beklenen"), [
    (1.0, 0.0, 1), (-1.0, 0.0, -1),
    (0.0, 0.0, 0),                       # EŞİTLİK → yön yok (eskiden −1)
    (5.0, 5.0, 0), (-3.0, -3.0, 0),
    (float("nan"), 0.0, 0),              # ölçülemedi → yön yok (eskiden −1)
    (0.0, float("nan"), 0),
    (float("inf"), 0.0, 0), (1.0, float("-inf"), 0),
    (None, 0.0, 0), ("x", 0.0, 0),       # dönüştürülemez → istisna DEĞİL, 0
])
def test_vote_esitlik_ve_olcumsuz_sifirdir(x, y, beklenen):
    assert vote(x, y) == beklenen


def test_note3_oyla_tutarli():
    """Oy 0 iken not "ayı" diyemez — pano kendi kendisiyle çelişmemeli."""
    assert note3(1, "yukarı", "aşağı") == "yukarı"
    assert note3(-1, "yukarı", "aşağı") == "aşağı"
    assert note3(0, "yukarı", "aşağı") not in ("yukarı", "aşağı")


# --- 2) Panelin bütünü: sıfır bilgi → sıfır görüş ---------------------------------
def test_duz_seride_hicbir_gosterge_yon_oyu_vermez():
    """BULGU-33-1'in kilidi. Düzeltme öncesi: bull=11 / bear=46 / skor −0,542."""
    pan = ind.indicator_panel(_flat_df())
    yonlu = [r for r in pan["rows"] if r["signal"] != 0]
    assert yonlu == [], f"düz seride yön oyu veren gösterge(ler): {[r['name'] for r in yonlu]}"
    assert pan["bull"] == 0 and pan["bear"] == 0
    assert pan["consensus_score"] == 0.0
    assert pan["consensus"] == "YETERSİZ ÖLÇÜM"
    assert pan["directional"] == 0


def test_az_sayida_oydan_guclu_hukum_cikmaz():
    """Tur-23 dersi ("kapı PASS dese bile n_kept oku") panoda: 2 oydan 0 skoru +1,000
    eder ve etiket "GÜÇLÜ AL" olurdu. Oran, paydası küçükken okunamaz."""
    df = _flat_df()
    df.iloc[-1, df.columns.get_loc("close")] = 101.0        # tek bar kıpırdasın
    df.iloc[-1, df.columns.get_loc("high")] = 101.0
    pan = ind.indicator_panel(df)
    if pan["directional"] < 10:
        assert pan["consensus"] == "YETERSİZ ÖLÇÜM", (
            f"{pan['directional']} yönlü oydan '{pan['consensus']}' hükmü çıktı")


def test_duz_seride_extended_rows_da_notr():
    rows = ext.extended_rows(_flat_df(), price=100.0)
    assert rows, "genişletilmiş satırlar hiç üretilmedi (try/except her şeyi yutuyor olabilir)"
    assert all(r["signal"] == 0 for r in rows)


def test_normal_seride_panel_hala_yon_uretiyor():
    """Düzeltme dejenere yolu kapatır; BİLGİ VARKEN paneli susturmamalı."""
    pan = ind.indicator_panel(_normal_df())
    assert pan["bull"] > 0 and pan["bear"] > 0
    assert pan["bull"] + pan["bear"] >= 40


def test_sifir_hacimde_de_yon_uydurulmaz():
    """Hacim 0: MFI/CMF/OBV/VWAP payda dejenerasyonu — fiyat düzken yön çıkmamalı."""
    pan = ind.indicator_panel(_flat_df(volume=0.0))
    assert all(r["signal"] == 0 for r in pan["rows"])


# --- 3) Tekil kök nedenler --------------------------------------------------------
def test_mfi_duz_seride_notr_50(_=None):
    """BULGU-33-2 — `rsi`in Tur-28'de kapatılan 0/0 kusurunun kardeşi.
    pos==0 VE neg==0 → 100 değil 50. (pos>0, neg==0 hâlâ 100 = klasik konvansiyon.)"""
    duz = ind.mfi(_flat_df())
    assert duz.dropna().eq(50.0).all()

    # Kontrol: gerçekten hiç kayıp yoksa 100 KORUNMALI (düzeltme fazla ileri gitmesin)
    n = 60
    artan = np.arange(100.0, 100.0 + n)
    df = pd.DataFrame({"open": artan, "high": artan + 1, "low": artan - 1, "close": artan,
                       "volume": np.full(n, 1000.0)})
    assert ind.mfi(df).dropna().iloc[-1] == pytest.approx(100.0)


def test_spearman_tam_bagli_pencerede_tanimsiz():
    """BULGU-33-3 — düz pencerede sıra varyansı yok; formül +50 üretiyordu."""
    duz = ext.spearman(pd.Series(np.full(40, 7.0)))
    assert duz.dropna().empty, "düz seride Spearman bir SAYI döndürüyor (uydurma)"

    artan = ext.spearman(pd.Series(np.arange(40, dtype=float)))
    assert artan.dropna().iloc[-1] == pytest.approx(100.0)


def test_supertrend_atr_sifirken_yon_yok():
    """ATR=0 → bantlar çakışır, "fiyat hangi bandın dışında" TANIMSIZ (eskiden −1)."""
    _line, yon = ind.supertrend(_flat_df())
    assert yon.dropna().empty


def test_supertrend_isinma_penceresinde_bos():
    """Isınma barları eskiden 1.0 (=YUKARI TREND) ile doldurulmuş bırakılıyordu."""
    # period=10 → ATR ilk kez 9. barda (0-tabanlı) doğar; öncesi ölçümsüzdür.
    _line, yon = ind.supertrend(_normal_df(n=40))
    assert yon.iloc[:9].isna().all()
    assert yon.iloc[9:].notna().all()                 # normal veride yön yine üretiliyor


def test_supertrend_yukselen_seride_hala_pozitif():
    """Düzeltme normal davranışı bozmamalı (mevcut testin ikizi, bilerek)."""
    n = 80
    artan = np.arange(n, dtype=float)
    df = pd.DataFrame({"open": artan, "high": artan + 1.0, "low": artan - 1.0,
                       "close": artan + 0.5, "volume": np.full(n, 1000.0)})
    _line, yon = ind.supertrend(df)
    assert yon.dropna().iloc[-1] > 0


def test_bollinger_yuzde_b_span_sifirken_uydurmuyor():
    """span==0 → %B tanımsız; eskiden 0.5 uydurulup `>=0.5` ile BOĞA oyuna çevriliyordu."""
    rows = {r["name"]: r for r in ext.extended_rows(_flat_df(), price=100.0)}
    assert rows["%B (Bollinger)"]["signal"] == 0
