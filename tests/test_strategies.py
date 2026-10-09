"""Strateji testleri — MACD, Donchian, Squeeze + dispatcher/ensemble."""
import numpy as np
import pandas as pd

import nihai.signals.strategy as S
from nihai.signals.strategy import SignalAction


def _ohlc(closes, highs=None, lows=None):
    c = pd.Series(closes, dtype="float64")
    h = pd.Series(highs, dtype="float64") if highs is not None else c + 0.5
    l = pd.Series(lows, dtype="float64") if lows is not None else c - 0.5
    return pd.DataFrame({"open": c, "high": h, "low": l, "close": c, "volume": 1e6})


def _fires(detect_fn, df, want, start=35):
    """Olay-bazlı stratejiler yalnızca kırılım barında tetiklenir; motorun bar-bar
    tükettiği gibi dilimleri tarayıp beklenen sinyalin bir kez çıktığını doğrula."""
    for i in range(start, len(df) + 1):
        a, r = detect_fn(df.iloc[:i])
        if a == want:
            return True, r
    return False, []


# --- MACD ---
def test_macd_bullish_cross():
    # Önce düşüş (hist negatif), sonra güçlü dönüş → bir noktada bullish cross
    df = _ohlc(list(np.linspace(100, 80, 40)) + list(np.linspace(80, 120, 20)))
    ok, r = _fires(S.detect_macd, df, SignalAction.LONG)
    assert ok and any("MACD_BULL_CROSS" in x for x in r)


def test_macd_bearish_cross():
    df = _ohlc(list(np.linspace(80, 110, 40)) + list(np.linspace(110, 70, 20)))
    ok, r = _fires(S.detect_macd, df, SignalAction.SHORT)
    assert ok and any("MACD_BEAR_CROSS" in x for x in r)


def test_macd_insufficient_bars():
    a, r = S.detect_macd(_ohlc(list(range(10))))
    assert a == SignalAction.WAIT and "INSUFFICIENT_BARS" in r


# --- Donchian ---
def test_donchian_breakout_long_in_trend():
    # Güçlü yukarı rampa → ADX yüksek + son bar yeni zirve
    closes = list(np.linspace(100, 200, 60))
    a, r = S.detect_donchian(_ohlc(closes))
    assert a == SignalAction.LONG
    assert any("DONCHIAN_BREAK_HIGH" in x for x in r)


def test_donchian_skips_weak_trend():
    # Yatay/gürültülü → ADX zayıf → işlem yok
    np.random.seed(7)
    closes = list(100 + np.random.normal(0, 0.3, 60))
    a, r = S.detect_donchian(_ohlc(closes))
    assert a == SignalAction.WAIT


# --- Squeeze ---
def test_squeeze_breakout_long():
    # Volatilite rejim değişimi: yüksek-vol → sıkışma (göreli düşük) → yukarı patlama.
    # (Squeeze GÖRELI bir ölçü; baseline yüksek-vol olmadan sıkışma tanımlanamaz.)
    np.random.seed(1)
    high_vol = list(100 + np.cumsum(np.random.normal(0, 2.0, 45)))
    base = high_vol[-1]
    quiet = list(base + np.random.normal(0, 0.08, 35))   # sıkışma (göreli sakin)
    burst = list(np.linspace(base + 1, base + 18, 8))     # patlama
    df = _ohlc(high_vol + quiet + burst)
    ok, r = _fires(S.detect_squeeze, df, SignalAction.LONG, start=80)
    assert ok and any("SQUEEZE_BREAK_UP" in x for x in r)


def test_squeeze_no_signal_without_squeeze():
    # Sürekli yüksek volatilite → göreli sıkışma yok → sinyal yok
    np.random.seed(2)
    closes = list(100 + np.cumsum(np.random.normal(0, 3, 80)))
    a, r = S.detect_squeeze(df=_ohlc(closes))
    assert a == SignalAction.WAIT


# --- Dispatcher / ensemble ---
def test_dispatcher_routes_new_modes():
    df = _ohlc(list(np.linspace(100, 200, 60)))
    for mode, label in [("macd", "MACD"), ("donchian", "Donchian"), ("squeeze", "Squeeze")]:
        a, r, lbl = S.detect_setup(df, mode)
        assert lbl == label

def test_strategy_modes_constant_complete():
    for m in ("event_breakout", "mean_reversion", "rsi_divergence",
              "macd", "donchian", "squeeze", "ensemble"):
        assert m in S.STRATEGY_MODES


_VALID_ROUTING = ("event_breakout", "macd", "donchian", "squeeze",
                  "mean_reversion", "rsi_divergence")


def test_ensemble_first_trigger_path():
    # Eski (varsayılan) yol: güçlü trend rampasında ilk-tetik kazanır → VIA_ etiketi.
    df = _ohlc(list(np.linspace(100, 200, 80)))
    a, r, lbl = S._detect_ensemble_first_trigger(df)
    assert a in (SignalAction.LONG, SignalAction.SHORT)
    assert any(x.startswith("VIA_") for x in r)
    assert lbl in _VALID_ROUTING


def test_ensemble_confluence_path():
    # Confluence oylaması: en az bir detektör etiketi + CONFLUENCE_N + geçerli routing.
    df = _ohlc(list(np.linspace(100, 200, 80)))
    a, r, lbl = S._detect_ensemble(df)
    assert a in (SignalAction.LONG, SignalAction.SHORT)
    assert any(x.startswith("CONFLUENCE_") for x in r)
    assert lbl in _VALID_ROUTING


# --- Katı confluence (§16.2 mutabakat) modu ---
def test_base_strategy_modes_excludes_meta_and_external():
    # Walk-forward/oto-tarama aday seti yalnız fiyat-only baz modlar olmalı.
    for m in S.BASE_STRATEGY_MODES:
        assert m in S.STRATEGY_MODES
    for bad in ("pead", "insider", "sentiment", "ensemble", "confluence"):
        assert bad not in S.BASE_STRATEGY_MODES, f"{bad} BASE'de olmamalı"
    assert "confluence" in S.STRATEGY_MODES  # tam listede var (config doğrulama)


def test_confluence_min_votes_gates_single_detector():
    # min_votes ulaşılamaz → tek/birkaç oy elenir (WAIT + CONFLUENCE_BELOW_MIN).
    df = _ohlc(list(np.linspace(100, 200, 80)))
    a, r, _ = S._detect_ensemble(df, min_votes=99)
    assert a == SignalAction.WAIT
    assert any(x.startswith("CONFLUENCE_BELOW_MIN") for x in r)


def test_confluence_mode_routes_and_is_strict():
    # confluence modu min_votes>=2 uygular; ensemble (min_votes=1) en az onun kadar gevşek.
    df = _ohlc(list(np.linspace(100, 200, 80)))
    a_loose, _, _ = S._detect_ensemble(df, min_votes=1)
    a_strict, r_strict, lbl = S.detect_setup(df, "confluence", symbol="TEST")
    # Gevşek tetiklerken katı ya aynı yönde ya da (yetersiz mutabakat) WAIT döner.
    assert a_strict in (a_loose, SignalAction.WAIT)
    if a_strict in (SignalAction.LONG, SignalAction.SHORT):
        assert lbl in _VALID_ROUTING


# --- Kalite skoru hacim-sütunu dayanıklılığı (KeyError regresyonu) ---
def test_quality_score_without_volume_column():
    # detect_vol_breakout 'volume' yoksa savunmacı; signal_quality_score de KeyError
    # atmamalı (bonusu atlar). Hacimsiz df ile çağrı sorunsuz skor döndürmeli.
    closes = list(np.linspace(100, 130, 60))
    df = pd.DataFrame({"open": closes, "high": [c + 0.5 for c in closes],
                       "low": [c - 0.5 for c in closes], "close": closes})
    assert "volume" not in df.columns
    score, breakdown = S.signal_quality_score(
        df, SignalAction.LONG, htf_trend_aligned=True,
        iv_rank_val=None, chain_health_ok=True,
    )
    assert isinstance(score, float)
    assert "volume" not in breakdown  # hacim yok → bonus yazılmadı
