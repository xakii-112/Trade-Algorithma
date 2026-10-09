"""Vol konisi: analitik Parkinson, skaler Yang–Zhang, sahte sağlayıcılı uç."""
import datetime

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nihai.ai import quant_math as qm
from nihai.web import api_quant as api


def _ohlc(n=400):
    rng = np.random.default_rng(17)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    op = close * np.exp(rng.normal(0, 0.004, n))
    return op, np.maximum(op, close) * 1.01, np.minimum(op, close) * 0.99, close


def test_parkinson_closed_form_and_percentile_ties():
    x = np.full(300, 100.0)
    expected = np.log(110 / 90) * np.sqrt(252 / (4 * np.log(2)))
    result = qm.volatility_cone(x, x * 1.1, x * .9, x, estimator="parkinson", iv_current=3)
    assert result["windows"] == [10, 20, 30, 60, 90, 180, 252]
    assert result["p50"] == pytest.approx([expected] * 7)
    assert result["iv_percentile"] == [100] * 7
    result = qm.volatility_cone(x, x * 1.1, x * .9, x, estimator="parkinson", iv_current=.1)
    assert result["iv_percentile"] == [0] * 7


@pytest.mark.parametrize("estimator", ["yang_zhang", "parkinson"])
def test_cone_matches_existing_scalar_estimators_and_is_deterministic(estimator):
    o, h, l, c = _ohlc()
    result = qm.volatility_cone(o, h, l, c, windows=(20,), estimator=estimator, iv_current=.2)
    assert result == qm.volatility_cone(o, h, l, c, windows=(20,), estimator=estimator, iv_current=.2)
    if estimator == "yang_zhang":
        history = [qm.vol_yang_zhang(o[i-20:i+1], h[i-20:i+1], l[i-20:i+1], c[i-20:i+1]) * np.sqrt(252)
                   for i in range(20, len(c))]
    else:
        history = [qm.vol_parkinson(h[i-19:i+1], l[i-19:i+1]) * np.sqrt(252)
                   for i in range(19, len(c))]
    assert result["current"][0] == pytest.approx(history[-1], rel=1e-12)
    assert result["p25"][0] == pytest.approx(np.percentile(history, 25), rel=1e-12)
    assert result["max"][0] == pytest.approx(max(history), rel=1e-12)
    assert result["n_history"][0] == len(history)


def test_cone_defensive_inputs_and_missing_iv():
    for bad in ([], None, [1], [np.nan] * 400, [np.inf] * 400):
        assert qm.volatility_cone(bad, bad, bad, bad) is None
    x = np.ones(400)
    assert qm.volatility_cone(x, x, x, x)["current"] == [0] * 7
    assert qm.volatility_cone(x, x, x, x, iv_current=np.inf)["iv_percentile"] == [None] * 7
    assert qm.volatility_cone(x, x * .5, x, x) is None
    assert qm.volatility_cone(x, x, x, x, windows=(1,)) is None
    assert qm.volatility_cone(x, x, x, x, ppy=np.nan) is None
    assert qm.volatility_cone(x, x, x, x, min_history=np.nan) is None
    assert qm.volatility_cone(x, x, x, x, estimator="made_up") is None


def test_vol_cone_api_uses_daily_bars_and_observed_atm_iv(monkeypatch):
    class Provider:
        def get_bars(self, symbol, interval, lookback):
            assert interval == "1d"
            return pd.DataFrame(dict(zip(("open", "high", "low", "close"), _ohlc(), strict=True)),
                                index=pd.date_range("2024-01-01", periods=400))

        def get_option_chain(self, *args):
            side = pd.DataFrame({"strike": [90, 100, 110], "impliedVolatility": [.8, .3, .7]})
            return {"calls": side, "puts": side, "underlying_price": 100,
                    "expiry": str(datetime.date.today() + datetime.timedelta(days=30))}

    monkeypatch.setattr(api, "_provider", lambda: Provider())
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        result = client.get("/api/quant/vol_cone?symbol=AAA").json()
        assert result["ok"] and result["cone"]["iv_current"] == .3
        assert result["cone"]["estimator"] == "yang_zhang"
        assert len(result["cone"]["iv_percentile"]) == 7
        assert not client.get("/api/quant/vol_cone?estimator=bad").json()["ok"]
