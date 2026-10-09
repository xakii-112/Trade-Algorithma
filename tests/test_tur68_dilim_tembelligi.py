"""Tur-68 — backtest bar döngüsündeki dilim, KULLANILDIĞI dalda kurulur (katalog #34).

**Bulgu.** `backtest_symbol`ün bar döngüsü her barın başında `df_slice = df.iloc[: i + 1]`
kuruyordu, ama `df_slice` YALNIZ *"açık pozisyon yok ve kill-switch kapalı"* dalında
okunuyordu (`generate_signal` + `recorder.on_open`). Yani dilim, pozisyon açıkken de
kuruluyor ve ATILIYORDU. Ölçüldü (2000 barlık koşu): **1939 dilimin 1418'i (%73) boşa**.

`df.iloc[:n]` ucuz bir görünüm DEĞİLDİR — yeni bir `BlockManager`, indeks dilimi ve
`__finalize__` demektir. Profil bunu `backtest_symbol`ün TEK EN BÜYÜK kalemi olarak
gösterdi (`.iloc.__getitem__` çağrılarının çağıranı: 1939 çağrı / 0,087 sn kümülatif).

**Onarım.** Satır, kullanıldığı `if`in içine taşındı. Tek satır yer değiştirdi; ifade,
sınır (`i + 1`), `shared` argümanı ve çağrı sırası AYNI.

**ÖLÇÜLDÜ** (6 sentetik senaryo, ısındırma sonrası p50, `mc_iters=0`):

| senaryo | bar | işlem | ÖNCE | SONRA | oran |
|---|---|---|---|---|---|
| S0 | 2000 | 70 | 163,84 ms | 119,76 ms | 1,37× |
| S1 | 1500 | 65 | 118,76 ms | 92,47 ms | 1,28× |
| S2 | 800 | 51 | 107,16 ms | 88,24 ms | 1,21× |
| S3 | 2500 | 30 | 101,20 ms | 59,25 ms | **1,71×** |
| S4 | 400 | 30 | 52,20 ms | 46,99 ms | 1,11× |
| **TOPLAM** | | | **696,44 ms** | **525,62 ms** | **1,32×** |

Kazanç, pozisyonun AÇIK kaldığı süreyle orantılıdır (S3 az işlem/uzun tutuş = en çok
atlanan dilim = en büyük kazanç). `.iloc` çağrısı 1939 → **478**'e indi.

**TAM `BacktestResult` BİT-DENKTİR** — altı senaryonun tamamında `dataclasses.asdict`
eşitliği (işlem sayısı, WR, final equity, `trade_pnls`, `equity_curve`, tüm metrikler).

**Bu turun asıl riski hız değil LOOK-AHEAD'dir.** Bir backtest döngüsünde kod taşımak,
sinyalin GÖRDÜĞÜ veriyi kazara değiştirebilir ve bu sessizce "harika" bir backtest üretir.
Bu yüzden buradaki testlerin ağırlığı hızda değil, **dilim sınırında**dır: `generate_signal`
bar `i`nin kapanışına kadar (dahil) veriden FAZLASINI asla görmemelidir.

**Kapsam notu.** Kardeş döngüler (`backtest.py:821` portföy, `:941` sentetik opsiyon,
`research/feed_ab_test.py`) zaten dilimi çağrı yerinde kuruyor — desen arandı, yalnız
`backtest_symbol` bu israfı taşıyordu (Tur-35 dersi: önce DESENİ ara).
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

from nihai.execution import backtest as bt

KOK = pathlib.Path(__file__).resolve().parents[1]
KAYNAK = KOK / "nihai" / "execution" / "backtest.py"


def _barlar(n: int = 600, seed: int = 3, vol: float = 0.012) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03 14:30", periods=n, freq="1h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, vol, n)))
    hi = c * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = c * (1 - np.abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {"open": np.r_[c[0], c[:-1]], "high": np.maximum(hi, c),
         "low": np.minimum(lo, c), "close": c,
         "volume": rng.integers(1_000_000, 9_000_000, n).astype(float)}, index=idx)


def _backtest_symbol_dugumu() -> ast.FunctionDef:
    tree = ast.parse(KAYNAK.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "backtest_symbol":
            return n
    raise AssertionError("backtest_symbol bulunamadı")


# --- 1. YAPISAL: dilim, döngünün tepesinde DEĞİL, kullanıldığı dalda kurulur --------
def test_dilim_kosullu_dalda_kuruluyor():
    """`df_slice` ataması, `generate_signal`ı da içeren bir `If` gövdesinde olmalı.

    Tepeye geri taşınırsa bar başına bir DataFrame daha kurulur ve %73'ü atılır."""
    fn = _backtest_symbol_dugumu()
    atamalar = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "df_slice" for t in n.targets)]
    assert len(atamalar) == 1, f"df_slice ataması {len(atamalar)} yerde — tek olmalı"
    hedef = atamalar[0]

    saran_if = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and any(hedef is alt for alt in ast.walk(node))
        and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "generate_signal"
                for c in ast.walk(node))
    ]
    assert saran_if, (
        "df_slice bir `if` içinde ve generate_signal ile AYNI dalda kurulmuyor — "
        "döngü tepesine geri taşınmış olabilir (Tur-68 israfı geri gelir)"
    )


# --- 2. DAVRANIŞ: boşa kurulan dilim kalmadı ---------------------------------------
def test_dilim_sayisi_sinyal_sayisina_esit(monkeypatch):
    """Kurulan her dilim KULLANILMALI: dilim sayısı == generate_signal çağrı sayısı.

    Bu testin ölçtüğü şey hız değil İSRAFTIR — israf geri gelirse sayılar ayrışır."""
    sayac = {"sinyal": 0}
    gercek = bt.generate_signal

    def sayan(symbol, df_slice, **kw):
        sayac["sinyal"] += 1
        sayan.son_uzunluk = len(df_slice)      # type: ignore[attr-defined]
        return gercek(symbol, df_slice, **kw)

    monkeypatch.setattr(bt, "generate_signal", sayan)

    # `df.iloc` bir property'dir; sayaç indeksleyici SINIFINA takılır. Bu pandas'ın
    # özel API'sidir — sürüm değiştirip kaybolursa test SESSİZ GEÇMESİN, atlansın.
    indeksleyici = pytest.importorskip("pandas.core.indexing")
    if not hasattr(indeksleyici, "_iLocIndexer"):
        pytest.skip("pandas._iLocIndexer yok — sayaç takılamıyor (sürüm değişmiş)")

    dilimler = {"n": 0}
    orijinal_iloc = indeksleyici._iLocIndexer.__getitem__

    df = _barlar(600, seed=3)

    def sayan_iloc(self, key):
        # Yalnız BU çerçeve üzerindeki `df.iloc[:n]` PREFIX dilimleri sayılır.
        # `self.obj is df` şartı olmazsa takvim/strateji modüllerinin kendi dilimleri
        # de sayaca girer (ilk yazımda öyle oldu: 420 ↔ 320).
        if (self.obj is df and isinstance(key, slice)
                and key.start is None and key.stop is not None):
            dilimler["n"] += 1
        return orijinal_iloc(self, key)

    monkeypatch.setattr(indeksleyici._iLocIndexer, "__getitem__", sayan_iloc)
    bt.backtest_symbol("T", df=df, mc_iters=0)

    assert sayac["sinyal"] > 0, "senaryo hiç sinyal üretmedi — test bir şey ölçmüyor"
    assert dilimler["n"] == sayac["sinyal"], (
        f"prefix dilim {dilimler['n']} kez kuruldu ama generate_signal "
        f"{sayac['sinyal']} kez çağrıldı → {dilimler['n'] - sayac['sinyal']} dilim BOŞA"
    )


# --- 3. DOĞRULUK: look-ahead YOK (bu turun asıl riski) -----------------------------
def test_sinyal_karar_barindan_ilerisini_GORMUYOR(monkeypatch):
    """`generate_signal`e giden dilim, kendi son barından ötesini ASLA taşımamalı.

    Bir backtest döngüsünde kod taşımanın gerçek tehlikesi hız değil, sinyalin gördüğü
    pencerenin kazara genişlemesidir — bu sessizce "mükemmel" bir backtest üretir.
    Burada her çağrıda dilimin ana çerçeveye göre bittiği yer kaydedilir ve dilimin
    İÇERİĞİ ana çerçevenin aynı önekiyle karşılaştırılır."""
    df = _barlar(600, seed=3)
    ihlaller: list[str] = []
    gercek = bt.generate_signal

    def denetleyen(symbol, df_slice, **kw):
        n = len(df_slice)
        if n > len(df):
            ihlaller.append(f"dilim ana çerçeveden uzun: {n} > {len(df)}")
        elif not df_slice.index.equals(df.index[:n]):
            ihlaller.append(f"dilim ÖNEK değil (n={n})")
        elif not np.array_equal(df_slice["close"].to_numpy(), df["close"].to_numpy()[:n]):
            ihlaller.append(f"dilim içeriği ana çerçeveyle uyuşmuyor (n={n})")
        return gercek(symbol, df_slice, **kw)

    monkeypatch.setattr(bt, "generate_signal", denetleyen)
    bt.backtest_symbol("T", df=df, mc_iters=0)
    assert not ihlaller, "LOOK-AHEAD ihlali:\n  " + "\n  ".join(ihlaller[:5])


def test_dilim_sinirinin_infazdan_BIR_bar_gerisinde_oldugu(monkeypatch):
    """Sözleşme: karar bar `i`nin KAPANIŞINDA, infaz bar `i+1`in AÇILIŞINDA.

    Dilim `df.iloc[: i + 1]` olduğu için uzunluğu `i + 1`dir; infaz fiyatı `open[i+1]`
    diliminin DIŞINDADIR. Dilim bir bar uzasaydı sinyal, üzerinde işlem yapacağı barın
    kapanışını görürdü."""
    df = _barlar(400, seed=11)
    uzunluklar: list[int] = []
    gercek = bt.generate_signal

    def kaydeden(symbol, df_slice, **kw):
        uzunluklar.append(len(df_slice))
        return gercek(symbol, df_slice, **kw)

    monkeypatch.setattr(bt, "generate_signal", kaydeden)
    bt.backtest_symbol("T", df=df, mc_iters=0)
    assert uzunluklar, "hiç sinyal üretilmedi"
    # infaz i+1'de olduğu için son karar barı en fazla len(df)-2 olabilir → dilim ≤ len-1
    assert max(uzunluklar) <= len(df) - 1, (
        f"en uzun dilim {max(uzunluklar)}, ana çerçeve {len(df)} — son bar infaz için "
        "ayrılmış olmalı (dilim onu görmemeli)"
    )


# --- 4. DETERMİNİZM: aynı girdi → aynı sonuç ---------------------------------------
@pytest.mark.parametrize(("n", "seed", "vol"), [(600, 3, 0.012), (400, 11, 0.02)])
def test_ayni_girdi_ayni_sonuc(n, seed, vol):
    """Yeniden yapılandırma sonrası motor hâlâ deterministik (gizli durum sızmadı)."""
    from dataclasses import asdict
    df = _barlar(n, seed, vol)
    assert asdict(bt.backtest_symbol("T", df=df, mc_iters=200)) == \
        asdict(bt.backtest_symbol("T", df=df, mc_iters=200))
