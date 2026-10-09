"""research.edge_lab.build_overnight_dataset testleri — TAMAMEN ÇEVRİMDIŞI (ağ monkeypatch'li).

Odak: leak-free'lik (özellik kolonları t+1 açılışını GÖRMEZ, yalnız label görür), maliyet
eşiği (5 bps) etiket kuralı, şema (ts,symbol,f_*,label), eksik open/close → sembol atlanır,
alpaca->yfinance fallback zinciri. Gerçek piyasa verisine/ağa ÇIKILMAZ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.edge_lab import build_overnight_dataset as bod


# ---------------------------------------------------------------------------
# Sentetik OHLCV üretici (deterministik)
# ---------------------------------------------------------------------------
def _synthetic_ohlcv(n: int = 40, seed: int = 0, start: str = "2022-01-03") -> pd.DataFrame:
    """n iş günü, rastgele-yürüyüş kapanış + tutarlı open/high/low/volume."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = 100.0 + np.cumsum(rng.normal(scale=1.0, size=n))
    # açılış = önceki kapanışa küçük gece sıçraması (ilk gün kendi başına)
    open_ = close.copy()
    open_[1:] = close[:-1] * (1 + rng.normal(scale=0.003, size=n - 1))
    open_[0] = close[0] * 0.999
    high = np.maximum(open_, close) + rng.uniform(0, 0.5, size=n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.5, size=n)
    volume = rng.uniform(1e6, 5e6, size=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                        "volume": volume}, index=idx)


# ---------------------------------------------------------------------------
# Şema
# ---------------------------------------------------------------------------
def test_sema_kolonlari_harness_ile_birebir():
    df = _synthetic_ohlcv(n=40, seed=1)
    feat = bod.build_features_for_symbol(df, "TEST")
    assert feat is not None
    assert list(feat.columns) == bod.SCHEMA_COLS
    assert set(feat.columns) == {"ts", "symbol", *bod.FEATURE_COLS, "label"}
    assert feat["label"].isin([0, 1]).all()
    assert (feat["symbol"] == "TEST").all()


def test_son_satir_gelecek_verisi_yoksa_sahte_label_sizmaz():
    """Regresyon: pandas'ta `NaN > eşik` False döner (NaN değil) — son satırda t+1 verisi
    yokken bunu doğrudan etikete çevirirsek sahte label=0 dropna'dan KAÇAR. Son satırın ts'i
    kesinlikle çıktıda YER ALMAMALI."""
    df = _synthetic_ohlcv(n=40, seed=42)
    feat = bod.build_features_for_symbol(df, "LASTROW")
    assert feat is not None
    assert df.index[-1] not in set(feat["ts"])


def test_yetersiz_gecmis_none_doner():
    # MIN_HISTORY_DAYS'ten (20) az gün + son satır t+1 gerektirir → hiç geçerli satır kalmaz
    df = _synthetic_ohlcv(n=5, seed=2)
    feat = bod.build_features_for_symbol(df, "SHORT")
    assert feat is None


# ---------------------------------------------------------------------------
# Etiket = maliyet-eşiği (5 bps) kuralı
# ---------------------------------------------------------------------------
def test_etiket_maliyet_esigi_sinir_degerleri():
    # eşik = 5 bps = 0.0005 net; close[0]=100 -> open[1]=100.5 -> overnight=+0.005 (50bps),
    # eşiğin açıkça üstünde — maliyet-eşiği kuralının doğru sabiti kullandığını doğrular.
    assert bod.COST_THRESHOLD == pytest.approx(0.0005)
    on_ret_t1 = (100.5 - 100.0) / 100.0
    assert on_ret_t1 > bod.COST_THRESHOLD


def test_etiket_esik_altinda_negatif_etiket_sifir():
    rng = np.random.default_rng(7)
    n = 30
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = np.full(n, 100.0)
    open_ = close.copy()
    # Her gece getirisi tam olarak eşiğin YARISI (pozitif ama eşiği geçmiyor) -> label hep 0
    for i in range(1, n):
        open_[i] = close[i - 1] * (1 + bod.COST_THRESHOLD / 2)
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    # hacim sabit OLMAMALI (rolling std=0 -> f_volume_z NaN -> tüm satırlar düşer); hafif gürültü ver
    volume = 1e6 + rng.normal(scale=1e4, size=n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": volume}, index=idx)
    feat = bod.build_features_for_symbol(df, "HALFTHR")
    assert feat is not None
    assert (feat["label"] == 0).all()


def test_etiket_esik_ustunde_pozitif_etiket_bir():
    rng = np.random.default_rng(8)
    n = 30
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = np.full(n, 100.0)
    open_ = close.copy()
    for i in range(1, n):
        open_[i] = close[i - 1] * (1 + bod.COST_THRESHOLD * 3)   # eşiğin 3 katı -> hep pozitif
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    volume = 1e6 + rng.normal(scale=1e4, size=n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": volume}, index=idx)
    feat = bod.build_features_for_symbol(df, "OVERTHR")
    assert feat is not None
    assert (feat["label"] == 1).all()


# ---------------------------------------------------------------------------
# Leak-free: t+1 açılışı ETİKETTE var, ÖZELLİKLERDE yok
# ---------------------------------------------------------------------------
def test_leak_free_ozellikler_gelecek_bara_bagli_degil():
    """t satırının f_* kolonları yalnız t VE ÖNCESİ barlardan hesaplanır: t+1 (ve ötesi)
    barlarının open/close/volume değerlerini değiştirmek t satırının ÖZELLİKLERİNİ
    ETKİLEMEMELİ — ama t satırının label'ı (t+1 overnight getirisi) DEĞİŞMELİ."""
    df = _synthetic_ohlcv(n=40, seed=3)
    feat_before = bod.build_features_for_symbol(df, "LEAK")
    assert feat_before is not None

    df_mut = df.copy()
    # t+1'den itibaren TÜM barları büyük ölçüde boz (gelecek tamamen farklı bir dünya olsun)
    mutate_from = 25
    df_mut.iloc[mutate_from:, df_mut.columns.get_indexer(["open", "close", "high", "low"])] *= 3.0
    df_mut.iloc[mutate_from:, df_mut.columns.get_loc("volume")] *= 10.0
    feat_after = bod.build_features_for_symbol(df_mut, "LEAK")
    assert feat_after is not None

    # ts=idx[mutate_from - 1]: bu satırın ÖZELLİKLERİ yalnız kendisine kadarki (mutasyon
    # ÖNCESİ) barlardan gelir — AYNI kalmalı. Ama LABEL'i t+1 = idx[mutate_from] (mutasyonun
    # İLK barı) overnight-getirisine bakar — bu DEĞİŞMELİ (t+1 bilgisi doğru şekilde label'a
    # yansıyor mu testi).
    decision_ts = df.index[mutate_from - 1]
    row_before = feat_before[feat_before["ts"] == decision_ts]
    row_after = feat_after[feat_after["ts"] == decision_ts]
    assert len(row_before) == 1 and len(row_after) == 1
    for col in bod.FEATURE_COLS:
        assert row_before[col].iloc[0] == pytest.approx(row_after[col].iloc[0], abs=1e-9), (
            f"{col} gelecek-bar mutasyonundan ETKİLENMEMELİYDİ (leak-free ihlali)"
        )

    # decision_ts + 1 iş günü mutasyondan sonraki bar OLDUĞUNDAN label DEĞİŞMELİ (aksi
    # halde t+1 bilgisi zaten label'a doğru şekilde YANSIMIYOR demektir).
    assert row_before["label"].iloc[0] != row_after["label"].iloc[0]


def test_leak_free_ozellik_kolonlari_gelecek_isimli_kolon_icermez():
    """Kolon-adı düzeyinde de doğrulama: FEATURE_COLS içinde 'next'/'lead'/'future' gibi
    ileri-bakan bir kolon YOK; SADECE 'label' geleceğe (t+1) bakar."""
    forbidden_markers = ("next", "lead", "future", "t1", "tomorrow")
    for col in bod.FEATURE_COLS:
        low = col.lower()
        assert not any(m in low for m in forbidden_markers), f"{col} ileri-bakan görünüyor"
    assert bod.SCHEMA_COLS[-1] == "label"


# ---------------------------------------------------------------------------
# Eksik open/close → sembol atlanır (koşu çökmez)
# ---------------------------------------------------------------------------
def test_eksik_close_kolonu_sembol_atlanir():
    df = _synthetic_ohlcv(n=40, seed=4).drop(columns=["close"])
    feat = bod.build_features_for_symbol(df, "NOCLOSE")
    assert feat is None


def test_bos_df_sembol_atlanir():
    feat = bod.build_features_for_symbol(pd.DataFrame(), "EMPTY")
    assert feat is None


def test_build_dataset_eksik_kolonlu_sembolu_atlar_digerini_isler(monkeypatch):
    good = _synthetic_ohlcv(n=40, seed=5)
    bad = _synthetic_ohlcv(n=40, seed=6).drop(columns=["open"])

    def fake_fetch(symbol: str, years: int):
        return {"GOOD": good, "BAD": bad}.get(symbol)

    out = bod.build_dataset(["GOOD", "BAD"], fetch_fn=fake_fetch)
    assert set(out["symbol"].unique()) == {"GOOD"}
    assert len(out) > 0


def test_build_dataset_veri_gelmeyen_sembolu_atlar_cokmez(monkeypatch):
    def fake_fetch(symbol: str, years: int):
        return None   # hiçbir sembol için veri yok (ağ tamamen kapalı senaryosu)

    out = bod.build_dataset(["A", "B", "C"], fetch_fn=fake_fetch)
    assert out.empty
    assert list(out.columns) == bod.SCHEMA_COLS


# ---------------------------------------------------------------------------
# alpaca -> yfinance fallback zinciri (ağa ÇIKMADAN, iç fonksiyonlar monkeypatch'li)
# ---------------------------------------------------------------------------
def test_fetch_daily_bars_alpaca_basarisizsa_yfinancee_duser(monkeypatch):
    calls = {"alpaca": 0, "yfinance": 0}

    def fake_alpaca(symbol, years):
        calls["alpaca"] += 1
        return None   # alpaca boş/hata döndü

    def fake_yf(symbol, years):
        calls["yfinance"] += 1
        return _synthetic_ohlcv(n=25, seed=9)

    monkeypatch.setattr(bod, "_fetch_alpaca", fake_alpaca)
    monkeypatch.setattr(bod, "_fetch_yfinance", fake_yf)

    df = bod.fetch_daily_bars("ANY", years=5)
    assert df is not None and not df.empty
    assert calls == {"alpaca": 1, "yfinance": 1}


def test_fetch_daily_bars_alpaca_basariliysa_yfinance_cagrilmaz(monkeypatch):
    calls = {"alpaca": 0, "yfinance": 0}

    def fake_alpaca(symbol, years):
        calls["alpaca"] += 1
        return _synthetic_ohlcv(n=25, seed=10)

    def fake_yf(symbol, years):
        calls["yfinance"] += 1
        return _synthetic_ohlcv(n=25, seed=11)

    monkeypatch.setattr(bod, "_fetch_alpaca", fake_alpaca)
    monkeypatch.setattr(bod, "_fetch_yfinance", fake_yf)

    df = bod.fetch_daily_bars("ANY", years=5)
    assert df is not None and not df.empty
    assert calls == {"alpaca": 1, "yfinance": 0}
