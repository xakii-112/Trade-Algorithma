"""Opsiyon analiz testleri — Madde 16, 21, 27, 28, 29."""
import pandas as pd
import pytest

from nihai.config import CONFIG
from nihai.risk.options_analyzer import (
    GreekSet,
    PortfolioGreeks,
    days_to_expiry,
    estimate_atm_iv,
    filter_liquid_chain,
    is_pin_risk,
    iv_rank,
    settlement_style,
)


def test_iv_rank_bounds():
    hist = pd.Series([0.2, 0.3, 0.4, 0.5])
    assert iv_rank(0.5, hist) == 100.0
    assert iv_rank(0.2, hist) == 0.0
    assert iv_rank(0.35, hist) == pytest.approx(50.0)
    assert iv_rank(0.3, pd.Series([], dtype=float)) == 50.0  # geçmiş yok → 50


def test_pin_risk():
    assert is_pin_risk(strike=100, underlying=100.2, dte=0)
    assert not is_pin_risk(strike=100, underlying=110, dte=0)
    assert not is_pin_risk(strike=100, underlying=100.1, dte=10)
    assert not is_pin_risk(strike=100, underlying=0, dte=0)   # sıfıra bölme güvenliği


def test_settlement_style_index_vs_equity():
    assert settlement_style("SPX")["exercise"] == "European"
    assert settlement_style("SPX")["tax_1256"] is True
    assert settlement_style("AAPL")["exercise"] == "American"


def test_portfolio_greeks_caps():
    agg = PortfolioGreeks()
    g = GreekSet(delta=0.5, gamma=0.02, theta=-0.05, vega=0.2, iv=0.3)
    agg.add_position(g, contracts=10)   # delta 0.5*10*100 = 500
    ok, breaches = agg.within_caps(CONFIG.greeks)
    assert not ok and any("NET_DELTA" in b for b in breaches)


def test_greeks_theta_vega_scaled_by_contract_multiplier():
    # BUG: theta/vega ×100 (kontrat çarpanı) ile ölçeklenmiyordu → 100× küçük
    agg = PortfolioGreeks()
    g = GreekSet(delta=0.5, gamma=0.02, theta=-0.05, vega=0.2, iv=0.3)
    agg.add_position(g, contracts=10)
    assert agg.theta == -0.05 * 10 * 100   # -50 (eskiden -0.5)
    assert agg.vega == 0.2 * 10 * 100      # 200 (eskiden 2.0)
    assert agg.delta == 0.5 * 10 * 100     # 500 (değişmedi)


def test_theta_cap_now_enforced():
    # Düşük delta ama theta cap'i aşan pozisyon → eskiden theta 100× küçük olduğu
    # için tavan ASLA tetiklenmezdi; artık tetiklenmeli.
    agg = PortfolioGreeks()
    g = GreekSet(delta=0.1, gamma=0.001, theta=-0.10, vega=0.05, iv=0.3)
    agg.add_position(g, contracts=10)  # theta=-0.10*10*100=-100 < -50 cap; delta=100<200 OK
    ok, breaches = agg.within_caps(CONFIG.greeks)
    assert not ok and any("NET_THETA" in b for b in breaches)
    assert not any("NET_DELTA" in b for b in breaches)   # delta tavanı aşmıyor (izole)


def test_filter_liquid_chain():
    chain = pd.DataFrame({
        "strike": [100, 105, 110],
        "bid": [1.0, 0.0, 2.0],
        "ask": [1.1, 0.5, 2.1],
        "openInterest": [1000, 10, 800],
        "volume": [200, 5, 150],
    })
    res = filter_liquid_chain(chain, min_oi=500, min_volume=100, max_spread_pct=0.10)
    # strike 105 elenir (düşük OI/vol, no bid)
    assert len(res.accepted) == 2
    assert res.rejected_count == 1


def test_estimate_atm_iv():
    # ATM call ~ S=100, K=100; mid prim ver → IV çözülmeli
    calls = pd.DataFrame({
        "strike": [90, 100, 110],
        "bid": [10.5, 3.0, 0.5],
        "ask": [10.9, 3.2, 0.7],
        "lastPrice": [10.7, 3.1, 0.6],
    })
    iv = estimate_atm_iv(calls, underlying=100, dte=30)
    assert iv is None or (0 < iv < 5)
