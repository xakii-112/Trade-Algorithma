"""A3 — portföy VaR/CVaR testleri (saf fonksiyonlar + sahte-provider orkestrasyon)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from nihai.risk.safety import (
    _position_exposure,
    compute_portfolio_var,
    historical_var,
    monte_carlo_var,
)


def test_historical_var_basic():
    # -100..100 simetrik dağılım → sol kuyruk pozitif VaR
    samples = list(range(-100, 101, 10))
    res = historical_var(samples, levels=(0.95, 0.99))
    assert res.method == "historical"
    assert res.n_samples == len(samples)
    assert res.var["95"] > 0
    # %99 daha derin kuyruk → en az %95 kadar büyük; CVaR ≥ VaR
    assert res.var["99"] >= res.var["95"]
    assert res.cvar["95"] >= res.var["95"]


def test_historical_var_insufficient_samples():
    res = historical_var([42.0], levels=(0.95,))
    assert res.var == {} and "yetersiz" in res.note


def test_historical_var_horizon_scaling():
    samples = list(range(-100, 101, 10))
    one = historical_var(samples, horizon_days=1).var["95"]
    four = historical_var(samples, horizon_days=4).var["95"]
    # sqrt(4)=2 → 4-günlük VaR ≈ 2× 1-günlük
    assert abs(four - 2 * one) < 1e-6


def test_monte_carlo_var_deterministic_and_ordered():
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.02, size=(250, 2))   # 2 varlık, %2 günlük vol
    exposures = [10000.0, -5000.0]
    a = monte_carlo_var(R, exposures, seed=123, n_sims=5000)
    b = monte_carlo_var(R, exposures, seed=123, n_sims=5000)
    assert a.var["95"] == b.var["95"]        # aynı seed → deterministik
    assert a.var["99"] >= a.var["95"] > 0
    assert a.cvar["99"] >= a.var["99"]


def test_monte_carlo_var_invalid_input():
    res = monte_carlo_var([[0.01]], [1.0])   # T=1 < 2 → geçersiz
    assert res.n_samples == 0 and "geçersiz" in res.note


def test_position_exposure_stock_and_option():
    long_stock = SimpleNamespace(side="LONG", quantity=100, asset_type="stock")
    short_stock = SimpleNamespace(side="SHORT", quantity=100, asset_type="stock")
    assert _position_exposure(long_stock, 50.0) == 5000.0
    assert _position_exposure(short_stock, 50.0) == -5000.0

    opt = SimpleNamespace(side="LONG", quantity=2, asset_type="option",
                          greeks=SimpleNamespace(delta=0.4))
    assert _position_exposure(opt, 100.0) == 0.4 * 100 * 2 * 100


class _FakeProvider:
    """get_bars → sabit, hafif rastgele yürüyüşlü OHLCV DataFrame."""
    def __init__(self, n=120):
        self.n = n

    def get_bars(self, symbol, interval, lookback):
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        rets = rng.normal(0.0005, 0.02, size=self.n)
        close = 100 * np.cumprod(1 + rets)
        idx = pd.date_range("2025-01-01", periods=self.n, freq="D")
        return pd.DataFrame({"open": close, "high": close, "low": close,
                             "close": close, "volume": 1000}, index=idx)


def test_compute_portfolio_var_end_to_end():
    positions = [
        SimpleNamespace(symbol="AAA", side="LONG", quantity=100, asset_type="stock", greeks=None),
        SimpleNamespace(symbol="BBB", side="SHORT", quantity=50, asset_type="stock", greeks=None),
    ]
    out = compute_portfolio_var(positions, _FakeProvider(), n_sims=2000, seed=7)
    assert out["ok"] is True
    assert out["n_positions"] == 2
    assert out["portfolio_value"] > 0
    assert out["historical"]["var"]["95"] > 0
    assert out["monte_carlo"]["var"]["95"] > 0
    assert len(out["exposures"]) == 2


def test_compute_portfolio_var_empty():
    out = compute_portfolio_var([], _FakeProvider())
    assert out["ok"] is False and "açık pozisyon yok" in out["note"]
