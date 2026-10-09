"""GEX kökü, birimler ve saatlik delta hedge ayrıştırması."""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from nihai.ai import option_flow as flow
from nihai.ai.quant_math import bs_greeks
from nihai.data import market_calendar
from nihai.web import api_quant


def _side(strike, oi=100):
    return {"strike": [strike], "openInterest": [oi], "impliedVolatility": [.3]}


def test_gamma_flip_is_repriced_brent_root_even_on_coarse_grid():
    calls, puts = _side(110), _side(90)
    result = flow.dealer_gamma_exposure(calls, puts, 100, .1, .02, grid=11, span=.3)
    assert result["gamma_flip"] is not None
    at_root = flow.dealer_gamma_exposure(calls, puts, result["gamma_flip"], .1, .02)
    assert abs(at_root["net_gex"]) < 1e-6
    assert result["net_gex_full_move"] == pytest.approx(result["net_gex"] * 100)
    assert flow.dealer_gamma_exposure(calls, calls, 100, .1)["regime"] == "notr_gamma"


def test_clock_hedge_sign_and_scale_match_delta_finite_difference():
    calls = _side(105, oi=1)
    spot, t, rate, change = 100.0, .2, .03, -.00001
    clock = flow.dealer_hedging_clock(calls, None, spot, t, rate, hours=1,
                                     iv_change_per_hour=change)
    row = clock["rows"][0]
    dt = 1 / (365 * 24)
    d0 = bs_greeks(spot, 105, t, rate, .3)["delta"]
    d1 = bs_greeks(spot, 105, t - dt, rate, .3 + change)["delta"]
    assert row["net_usd"] == pytest.approx(-(d1 - d0) * 100 * spot, rel=.002)
    assert row["net_usd"] == row["vanna_usd"] + row["charm_usd"]
    assert clock == flow.dealer_hedging_clock(calls, None, spot, t, rate, hours=1,
                                             iv_change_per_hour=change)


def test_clock_stops_at_zero_dte_and_rejects_unmeasured_inputs():
    calls = _side(100)
    clock = flow.dealer_hedging_clock(calls, None, 100, 2.5 / (365 * 24))
    assert [row["hour"] for row in clock["rows"]] == [1, 2, 2.5]
    assert flow.dealer_hedging_clock(calls, None, 100, 0) is None
    assert flow.dealer_hedging_clock(calls, None, 100, .1, hours=np.nan) is None
    assert flow.dealer_hedging_clock(calls, None, 100, .1, iv_change_per_hour=-1) is None
    assert flow.dealer_hedging_clock({"strike": [100]}, None, 100, .1) is None
    assert flow.dealer_hedging_clock(None, None, 100, .1) is None


def test_expiry_uses_actual_early_close_and_not_one_day_floor(monkeypatch):
    now = datetime(2025, 11, 28, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    monkeypatch.setattr(market_calendar, "now_et", lambda: now)
    assert api_quant._expiry_years("2025-11-28") == pytest.approx(1 / (365 * 24))
    assert api_quant._expiry_years("2025-11-29") is None
    assert api_quant._expiry_years("2025-11-26") == 0
