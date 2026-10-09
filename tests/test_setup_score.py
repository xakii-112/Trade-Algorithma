"""B7/B8 — kurulum kalitesi skoru + RS-vs-benchmark + hibrit sıralama testleri."""
import numpy as np
import pandas as pd

import nihai.execution.auto_screener as a
from nihai.signals.setup_score import benchmark_returns, compute_setup_score


def _df(closes):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "volume": np.full(len(closes), 1_000_000.0),
    })


def test_insufficient_data_scores_zero():
    ss = compute_setup_score(_df([100, 101, 102]))
    assert ss.score == 0.0


def test_uptrend_positive_trend_component():
    closes = [100 + i * 0.5 for i in range(260)]   # istikrarlı yükseliş
    ss = compute_setup_score(_df(closes))
    assert ss.components["trend"] == 20.0           # fiyat > SMA200
    assert ss.score > 0


def test_downtrend_negative_trend_component():
    closes = [300 - i * 0.5 for i in range(260)]
    ss = compute_setup_score(_df(closes))
    assert ss.components["trend"] == -20.0          # fiyat < SMA200


def test_relative_strength_vs_benchmark():
    # Sembol son 20 barda benchmark'tan güçlü → rs_short pozitif
    closes = [100 + i * 0.5 for i in range(260)]
    bench = {20: 1.0, 60: 2.0}
    ss = compute_setup_score(_df(closes), bench)
    assert ss.rs_short is not None and ss.rs_long is not None
    assert "rs_short" in ss.components
    # sembol getirisi > benchmark → RS bileşeni pozitif
    assert ss.rs_short > 0 and ss.components["rs_short"] > 0


def test_benchmark_returns_helper():
    closes = [100 * (1.01 ** i) for i in range(100)]
    br = benchmark_returns(_df(closes), horizons=(20, 60))
    assert 20 in br and 60 in br and br[60] > br[20] > 0


def test_benchmark_returns_empty_on_none():
    assert benchmark_returns(None) == {}


def test_rs_uses_long_horizon_separate_from_runup():
    # RS uzun ufku (rs_horizons) runup'tan AYRI: özel ufukla RS hesaplanır
    closes = [100 + i * 0.5 for i in range(260)]
    bench = {30: 3.0, 90: 5.0}
    ss = compute_setup_score(_df(closes), bench, runup_bars=5, rs_horizons=(30, 90))
    assert ss.rs_short is not None and ss.rs_long is not None
    assert "rs_short" in ss.components and "rs_long" in ss.components


def test_screener_days_to_bars_conversion(monkeypatch):
    # B8 — işlem-günü → bar dönüşümü interval'e göre
    monkeypatch.setattr(a, "AUTO_SCREEN_INTERVAL", "1d")
    assert a._rs_horizons_bars() == (a.SCREEN_RS_SHORT_DAYS, a.SCREEN_RS_LONG_DAYS)
    monkeypatch.setattr(a, "AUTO_SCREEN_INTERVAL", "1h")
    short, long = a._rs_horizons_bars()
    assert short == a.SCREEN_RS_SHORT_DAYS * 7 and long == a.SCREEN_RS_LONG_DAYS * 7


# Hibrit sıralama (_rank) ---------------------------------------------------------
def test_hybrid_rank_combines_backtest_and_setup(monkeypatch):
    monkeypatch.setattr(a, "SCREEN_HYBRID_ENABLED", True)
    monkeypatch.setattr(a, "SCREEN_HYBRID_SETUP_WEIGHT", 1.0)
    monkeypatch.setattr(a, "AUTO_SCREEN_SORT", "expectancy")
    monkeypatch.setattr(a, "AUTO_SCREEN_MIN_TRADES", 0)
    monkeypatch.setattr(a, "AUTO_SCREEN_MIN_EXPECTANCY", -1e9)
    monkeypatch.setattr(a, "AUTO_SCREEN_MAX_RUIN", 0)
    # A: orta backtest ama güçlü kurulum; B: güçlü backtest zayıf kurulum; C: zayıf/zayıf
    rows = [
        {"symbol": "A", "trades": 10, "expectancy": 5.0, "setup_score": 80.0, "ruin_prob": None},
        {"symbol": "B", "trades": 10, "expectancy": 6.0, "setup_score": -40.0, "ruin_prob": None},
        {"symbol": "C", "trades": 10, "expectancy": -2.0, "setup_score": -60.0, "ruin_prob": None},
    ]
    ranked = a._rank(rows)
    assert [r["symbol"] for r in ranked][0] == "A"   # güçlü kurulum öne taşır
    assert ranked[-1]["symbol"] == "C"
    assert all("combined_score" in r for r in ranked)


def test_rank_backtest_only_when_hybrid_off(monkeypatch):
    monkeypatch.setattr(a, "SCREEN_HYBRID_ENABLED", False)
    monkeypatch.setattr(a, "AUTO_SCREEN_SORT", "expectancy")
    monkeypatch.setattr(a, "AUTO_SCREEN_MIN_TRADES", 0)
    monkeypatch.setattr(a, "AUTO_SCREEN_MIN_EXPECTANCY", -1e9)
    monkeypatch.setattr(a, "AUTO_SCREEN_MAX_RUIN", 0)
    rows = [
        {"symbol": "A", "trades": 10, "expectancy": 5.0, "setup_score": 80.0, "ruin_prob": None},
        {"symbol": "B", "trades": 10, "expectancy": 6.0, "setup_score": -40.0, "ruin_prob": None},
    ]
    ranked = a._rank(rows)
    assert ranked[0]["symbol"] == "B"   # yalnızca backtest expectancy → B önce
