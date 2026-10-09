"""BULGU B-10 — `nihai/signals/indicators.py` kapsama kör noktası.

Ölçülen: 296 ifade, **%0 kapsama**. Modül ölü DEĞİL — canlı pano `/api/analyze`
ucunda her sembol analizinde çağrılıyor (`web_dashboard.py` içinde `ichimoku,
indicator_panel` geç-import edilir). 296 satırlık gösterge matematiği üretimde
koşuyordu ve tek bir test onu görmüyordu.

NEDEN GÖZDEN KAÇTI: `tests/test_indicators.py` ismi bu modülü test ediyor gibi
duruyor ama içeriği `nihai.signals.strategy`'nin RSI/ATR/ADX'ini doğruluyor.
İsim benzerliği, kapsama boşluğunu "zaten testli" sanısıyla örtmüş.

Testlerin duruşu: bunlar KENAR iddiası değil, panoda gösterilen ÖLÇÜMLERdir.
O yüzden "kârlı mı" denmez; doğruluk + dayanıklılık + look-ahead yokluğu denir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.signals import indicators as ind


def _df(n: int = 60, *, sabit: float | None = None, seed: int = 7) -> pd.DataFrame:
    """Sentetik ama tutarlı OHLCV (h >= max(o,c), l <= min(o,c))."""
    if sabit is not None:
        close = np.full(n, sabit, dtype=float)
    else:
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.linspace(1000, 2000, n)},
        index=pd.date_range("2026-01-01", periods=n, freq="h"),
    )


def _trend_df(n: int = 80) -> pd.DataFrame:
    return pd.DataFrame({
        "open": np.arange(n, dtype=float), "high": np.arange(n, dtype=float) + 1.0,
        "low": np.arange(n, dtype=float) - 1.0, "close": np.arange(n, dtype=float) + 0.5,
        "volume": np.full(n, 1000.0),
    })


# --- Bilinen cevap --------------------------------------------------------------


def test_obv_yon_mantigi():
    """OBV: kapanış yükselirse +hacim, düşerse -hacim."""
    df = pd.DataFrame({"open": [1.0, 1, 1], "high": [2.0, 2, 2], "low": [0.0, 0, 0],
                       "close": [10.0, 11.0, 10.5], "volume": [100.0, 200.0, 300.0]})
    o = ind.obv(df)
    assert o.iloc[1] - o.iloc[0] == pytest.approx(200.0)
    assert o.iloc[2] - o.iloc[1] == pytest.approx(-300.0)


def test_roc_yuzde_degisim():
    s = pd.Series([100.0, 110.0, 121.0])
    assert ind.roc(s, period=1).iloc[1] == pytest.approx(10.0)
    assert ind.roc(s, period=2).iloc[2] == pytest.approx(21.0)


def test_williams_r_araligi():
    w = ind.williams_r(_df(80), period=14).dropna()
    assert not w.empty and w.between(-100, 0).all()


def test_stochastic_araligi():
    k, d = ind.stochastic(_df(80))
    assert k.dropna().between(0, 100).all()
    assert d.dropna().between(0, 100).all()


def test_mfi_araligi():
    assert ind.mfi(_df(80)).dropna().between(0, 100).all()


def test_pivot_points_klasik_formul():
    df = _df(30)
    p = ind.pivot_points(df)
    h, low, c = df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
    assert p["pivot"] == pytest.approx((h + low + c) / 3, abs=0.01)   # kaynak 2 hane yuvarlar
    assert p["r3"] > p["r1"] > p["pivot"] > p["s1"] > p["s3"]


def test_pivot_points_tek_barda_bos_doner():
    """Sözleşme: <2 bar → {} (pano 'seviye yok' gösterir, patlamaz)."""
    assert ind.pivot_points(_df(1)) == {}


def test_heikin_ashi_kapanis_ortalamasi():
    """ha_close = (O+H+L+C)/4 — kolon adları ha_* (pano bunlara bakar)."""
    df = _df(30)
    ha = ind.heikin_ashi(df)
    assert list(ha.columns) == ["ha_open", "ha_high", "ha_low", "ha_close"]
    beklenen = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    assert ha["ha_close"].iloc[-1] == pytest.approx(beklenen.iloc[-1], rel=1e-9)


def test_heikin_ashi_ilk_bar_seed_ve_rekursiyon():
    """ha_open[0]=(O+C)/2, sonrası (önceki ha_open + önceki ha_close)/2 — look-ahead yok."""
    df = _df(10)
    ha = ind.heikin_ashi(df)
    assert ha["ha_open"].iloc[0] == pytest.approx(
        (df["open"].iloc[0] + df["close"].iloc[0]) / 2)
    assert ha["ha_open"].iloc[1] == pytest.approx(
        (ha["ha_open"].iloc[0] + ha["ha_close"].iloc[0]) / 2)


def test_supertrend_yukselen_seride_pozitif_yon():
    _st, yon = ind.supertrend(_trend_df())
    assert yon.iloc[-1] == 1


def test_parabolic_sar_yukseliste_fiyatin_altinda():
    df = _trend_df()
    assert ind.parabolic_sar(df).iloc[-1] < df["close"].iloc[-1]


def test_ichimoku_anahtarlari():
    """/api/analyze ichimoku'yu doğrudan JSON'a koyuyor — anahtarlar sabit kalmalı."""
    ich = ind.ichimoku(_df(120))
    assert {"tenkan", "kijun"} <= set(ich)


# --- Yozlaşma / sınır -----------------------------------------------------------


def test_sabit_fiyat_sifira_bolme_uretmez():
    """Tamamen yatay seri birçok göstergede 0'a bölme adayı (range=0, TR=0).
    NaN kabul; sonsuz DEĞİL — inf pano JSON'unu da bozar."""
    df = _df(60, sabit=100.0)
    for fn in (ind.mfi, ind.cmf, ind.cci, ind.williams_r, ind.obv, ind.vwap,
               ind.awesome_oscillator, ind.chaikin_oscillator):
        arr = np.asarray(fn(df), dtype=float)
        assert not np.isinf(arr).any(), f"{fn.__name__} inf üretti"


def test_kisa_seri_cokmez():
    """Pano yeni/ince sembolde birkaç barla çağırabilir — çökmek yerine NaN dönmeli."""
    df = _df(5)
    for fn in (ind.mfi, ind.cci, ind.williams_r, ind.cmf, ind.awesome_oscillator,
               ind.obv, ind.vwap):
        fn(df)


def test_tek_barlik_veri_cokmez():
    df = _df(1)
    ind.obv(df)
    ind.pivot_points(df)
    ind.heikin_ashi(df)


# --- Look-ahead yok (Madde 3 disiplini) -----------------------------------------


@pytest.mark.parametrize("fn", [
    ind.mfi, ind.cci, ind.williams_r, ind.obv,
    lambda d: ind.stochastic(d)[0],
    lambda d: ind.dmi(d)[0],
])
def test_son_bar_gecmisi_degistirmez(fn):
    """Son barı değiştirmek ÖNCEKİ barların değerini değiştirmemeli; değiştiriyorsa
    gösterge geleceğe bakıyordur (merkezli pencere / ters doldurma vb.)."""
    df = _df(80)
    bozuk = df.copy()
    bozuk.iloc[-1, bozuk.columns.get_loc("close")] *= 1.25
    bozuk.iloc[-1, bozuk.columns.get_loc("high")] *= 1.30

    a = pd.Series(fn(df)).iloc[:-1].to_numpy(dtype=float)
    b = pd.Series(fn(bozuk)).iloc[:-1].to_numpy(dtype=float)
    ayni = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-9, atol=1e-9)
    assert ayni.all(), "son bar geçmiş değerleri değiştirdi = look-ahead"


# --- Panonun gerçekten tükettiği yüzey ------------------------------------------


def test_indicator_panel_pano_sozlesmesi():
    """/api/analyze bu sözlüğü doğrudan JSON'a koyar: her satırda name/value/
    signal/note; signal ∈ {-1,0,1} (pano rengi buna bakar)."""
    panel = ind.indicator_panel(_df(120), live_price=105.0)
    assert isinstance(panel, dict) and "rows" in panel
    rows = panel["rows"]
    assert len(rows) >= 10
    for r in rows:
        assert {"name", "value", "signal", "note"} <= set(r)
        assert r["signal"] in (-1, 0, 1)
        assert isinstance(r["name"], str) and r["name"]


def test_indicator_panel_nan_uretmez():
    """JSON'a NaN yazılırsa tarayıcının JSON.parse'ı patlar → detay sekmesi ölür."""
    panel = ind.indicator_panel(_df(120), live_price=105.0)
    for r in panel["rows"]:
        if isinstance(r["value"], float):
            assert not np.isnan(r["value"]), f"{r['name']} NaN döndürdü"


def test_indicator_panel_kisa_veride_cokmez():
    """220 bar her zaman gelmez; pano 500 yerine az satırla yaşamalı."""
    panel = ind.indicator_panel(_df(8), live_price=None)
    assert isinstance(panel, dict) and "rows" in panel


def test_indicator_panel_canli_fiyatsiz_calisir():
    """live_price=None yolu (kotasyon yoksa) — B-4 elemesinden sonra bu yol sıklaşır."""
    panel = ind.indicator_panel(_df(120), live_price=None)
    assert panel["rows"]
