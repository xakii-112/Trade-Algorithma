"""Greek QP, kovaryans tutarlılığı, stres ve taban koruması; ağ yok."""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nihai.ai import advanced_math as am
from nihai.ai import portfolio as pf
from nihai.ai import quant_math as qm
from nihai.web import api_quant as aq


def test_greek_qp_closed_form_bounds_redundancy_and_invalid_results(monkeypatch):
    raw = qm.greek_neutral_hedge([[1, 1]], [4])
    np.testing.assert_allclose(raw["weights"], [-2, -2], atol=1e-8)
    np.testing.assert_array_equal(raw["weights"], qm.greek_neutral_hedge([[1, 1]], [4])["weights"])
    assert raw["objective"] == pytest.approx(8)
    bounded = qm.greek_neutral_hedge([[1, 1]], [4], [-1, -10], [10, 10])
    np.testing.assert_allclose(bounded["weights"], [-1, -3], atol=1e-8)
    redundant = qm.greek_neutral_hedge([[1, 1], [2, 2], [0, 0]], [4, 8, 0])
    assert redundant["rank"] == 1
    np.testing.assert_allclose(redundant["weights"], raw["weights"], atol=1e-8)
    scaled = qm.greek_neutral_hedge([[1e-8, 0], [0, 1e8]], [2e-8, 3e8])
    np.testing.assert_allclose(scaled["weights"], [-2, -3], atol=1e-8)
    for a, g, lo, hi in (([[1, 1]], [4], 0, 1), ([[1, 1], [2, 2]], [4, 9], None, None),
                         ([[0, 0]], [1], None, None), ([[1]], [1], 2, 1),
                         ([[1]], [1], np.nan, 1), ([[np.inf]], [1], None, None),
                         ([[1]], [np.nan], None, None), ([[1, 2]], [1, 2], None, None),
                         ([], [], None, None), (None, None, None, None)):
        assert qm.greek_neutral_hedge(a, g, lo, hi) is None
    monkeypatch.setattr(am, "qp_solve", lambda *a, **k: np.array([5., 5.]))
    assert qm.greek_neutral_hedge([[1, 1]], [4]) is None


def test_greek_api_schema_selected_risks_and_unhedged_exposure():
    app = FastAPI()
    app.include_router(aq.router)
    client = TestClient(app)
    payload = dict(current=dict(delta=200, gamma=8, vega=30), neutralize=["delta"], instruments=[
        dict(name="Underlying", delta=1, gamma=0, vega=0, lower=-500, upper=500)])
    result = client.post("/api/quant/greek_hedge", json=payload).json()
    assert result["ok"] and result["continuous"]
    assert result["instruments"][0]["quantity"] == pytest.approx(-200)
    assert result["after"] == dict(delta=0, gamma=8, vega=30)
    payload["neutralize"] = ["gamma"]
    assert not client.post("/api/quant/greek_hedge", json=payload).json()["ok"]
    payload["current"]["gamma"] = "NaN"
    assert client.post("/api/quant/greek_hedge", json=payload).status_code == 422
    payload["instruments"] *= 33
    assert client.post("/api/quant/greek_hedge", json=payload).status_code == 422


def test_rmt_covariance_reaches_weights_risk_and_api(monkeypatch, tmp_path):
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0003, .01, (100, 3)) * [1, 2, 3]
    prices = 100 * np.exp(np.vstack([np.zeros(3), np.cumsum(returns, axis=0)]))
    panel = pd.DataFrame(prices, columns=["AAA", "BBB", "CCC"], index=pd.bdate_range("2025-01-02", periods=101))
    expected = qm.denoise_corr_mp(returns)["cov_denoised"]
    monkeypatch.setattr(pf, "PORTFOLIO_PROPOSALS_FILE", tmp_path / "proposals.jsonl")
    candidates = [{"symbol": name} for name in panel.columns]
    report = pf.propose(candidates, panel, {"covariance_model": "rmt", "max_weight": 1})
    assert report["cov_source"] == "rmt"
    np.testing.assert_allclose(report["covariance"], expected, atol=1e-12)
    hrp = np.array(list(report["methods"]["hrp"]["weights"].values()))
    np.testing.assert_allclose(hrp, pf.hrp_weights(expected), atol=1e-8)
    assert report["methods"]["hrp"]["vol_per_bar"] == pytest.approx(np.sqrt(hrp @ expected @ hrp))
    got = dict(panel=panel, names=list(panel.columns), missing=[], rets=returns, ppy=1638)
    monkeypatch.setattr(aq, "_symbol_panel", lambda *a, **k: (got, None))
    monkeypatch.setattr(aq, "_faktor_bloku", lambda *a: None)
    api = aq.quant_portfolio(covariance_model="rmt", max_weight=1)
    assert api["ok"] and api["covariance_model"] == "rmt"
    out = api["methods"]["hrp"]
    np.testing.assert_allclose(out["weights"], hrp, atol=5e-5)
    np.testing.assert_allclose(out["risk_contrib"], hrp * (expected @ hrp) / (hrp @ expected @ hrp), atol=5e-5)
    assert out["vol_annual"] == pytest.approx(np.sqrt(hrp @ expected @ hrp * 1638), abs=5e-5)
    monkeypatch.setattr(qm, "denoise_corr_mp", lambda *a: None)
    assert not pf.propose(candidates, panel, {"covariance_model": "rmt"})["methods"]
    assert not aq.quant_portfolio(covariance_model="rmt")["ok"]
    assert not pf.propose(candidates, panel, {"covariance_model": "invalid"})["methods"]


def test_tims_nets_each_scenario_and_reprices_put_call_parity():
    long = dict(kind="stock", spot=100., quantity=100., multiplier=1.)
    short = {**long, "quantity": -50.}
    stress = pf.tims_stress([long, short], available_margin=200)
    assert stress == pf.tims_stress([long, short], available_margin=200)
    assert len(stress["scenarios"]) == 10
    assert stress["stress_requirement"] == pytest.approx(600)
    assert stress["margin_shortfall"] == pytest.approx(400)
    assert stress["scenarios"][0]["leg_pnl"] == pytest.approx([-1200, 600])
    assert pf.tims_stress([long])["margin_shortfall"] is None
    assert pf.tims_stress([long, short], -100)["margin_shortfall"] == pytest.approx(700)
    option = dict(spot=100., strike=110., t=.5, sigma=.25, r=.04, multiplier=100.)
    # S − C + P = PV(K); ortak spot/IV şoklarında net değer sabit kalır.
    parity = pf.tims_stress([long, {**option, "kind": "c", "quantity": -1.},
                             {**option, "kind": "p", "quantity": 1.}])
    assert parity["baseline_value"] == pytest.approx(11000 * np.exp(-.04 * .5))
    assert [s["pnl"] for s in parity["scenarios"]] == pytest.approx([0.] * 10, abs=1e-8)
    # Temettülü call−put farkı e^(-qT) S − e^(-rT) K.
    dividend = pf.tims_stress([{**option, "q": .03, "kind": "c", "quantity": 1.},
                               {**option, "q": .03, "kind": "p", "quantity": -1.}])
    assert dividend["baseline_value"] == pytest.approx(100 * (100 * np.exp(-.03 * .5) - 110 * np.exp(-.04 * .5)))
    for bad in ([], [dict(kind="c")], [{**long, "spot": np.nan}], [{**long, "multiplier": -1}],
                [{**option, "kind": "p", "quantity": -1, "t": 0}]):
        assert pf.tims_stress(bad) is None
    assert pf.tims_stress([long], np.inf) is None
    app = FastAPI()
    app.include_router(aq.router)
    assert TestClient(app).post("/api/quant/tims", json={"positions": [long, short]}).json()["ok"]
    api = aq.quant_options()
    greeks = qm.bs_greeks(100, 100, .25, .04, .25)
    assert api["now"]["greeks"]["theta"] == pytest.approx(greeks["theta"] / 365, abs=5e-7)
    assert api["now"]["greeks"]["vega"] == pytest.approx(greeks["vega"] / 100, abs=5e-7)
    assert len(api["tims"]["scenarios"]) == 10


def test_tipp_locks_peak_floor_and_state_reader_does_not_initialize(monkeypatch, tmp_path):
    from nihai import config
    equity = np.array([100., 120., 110., 100., 130.])
    peaks = np.maximum.accumulate(equity)
    path = [pf.tipp_allocation(v, peak, 90) for v, peak in zip(equity, peaks, strict=True)]
    assert [row["taban"] for row in path] == pytest.approx([90, 108, 108, 108, 117])
    assert [row["riskli_tutar"] for row in path] == pytest.approx([30, 36, 6, 0, 39])
    assert path[3]["floor_shortfall"] == 8
    assert path[3] == pf.tipp_allocation(100, 120, 90)
    assert pf.tipp_allocation(130, 120, 90)["high_water"] == 130
    for args in ((0, 100, 90), (100, 100, None), (100, np.inf, 90), (100, 100, -1), (100, 100, 90, 1)):
        assert pf.tipp_allocation(*args) is None
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(config, "STATE_FILE", state_path)
    assert not aq.quant_insurance(90)["ok"] and not state_path.exists()
    state_path.write_text(json.dumps({"portfolio": {"equity": 100., "ath_equity": 120.}, "saved_at": "2025-01-01T12:00:00"}))
    saved = state_path.read_bytes()
    report = aq.quant_insurance(90)
    assert report["ok"] and report["tipp"]["taban"] == 108
    assert report["cppi"]["taban"] == 90
    assert report["asof"] == "2025-01-01T12:00:00"
    assert state_path.read_bytes() == saved


def test_herc_preserves_local_erc_and_true_tree_branches():
    sigma = np.array([1., 2., 3., 4.])
    corr = np.array([[1., .9, .1, .1], [.9, 1., .1, .1], [.1, .1, 1., .9], [.1, .1, .9, 1.]])
    cov = corr * np.outer(sigma, sigma)
    w = pf.herc_weights(cov, n_clusters=2)
    np.testing.assert_array_equal(w, pf.herc_weights(cov, n_clusters=2))
    inner_a, inner_b = np.array([2/3, 1/3]), np.array([4/7, 3/7])
    vol_a = np.sqrt(inner_a @ cov[:2, :2] @ inner_a)
    vol_b = np.sqrt(inner_b @ cov[2:, 2:] @ inner_b)
    share_a = vol_b / (vol_a + vol_b)
    np.testing.assert_allclose(w, np.r_[share_a * inner_a, (1-share_a) * inner_b], atol=1e-6)
    for ids in ([0, 1], [2, 3]):
        sub = cov[np.ix_(ids, ids)]
        np.testing.assert_allclose(w[ids] * (sub @ w[ids]) / (w[ids] @ sub @ w[ids]), [.5, .5], atol=1e-6)
    global_contrib = w * (cov @ w) / (w @ cov @ w)
    assert global_contrib[:2].sum() == pytest.approx(.5, abs=1e-6)
    np.testing.assert_allclose(pf.herc_weights(cov * 1e-12, 2), w, atol=1e-6)
    assert pf.herc_weights(cov, 2, bounds=(np.zeros(4), np.full(4, .1))) is None
    for bad in (None, [[1, 2], [2, 1]], [[np.nan]], np.zeros((2, 2))):
        assert pf.herc_weights(bad) is None
    assert pf.herc_weights(cov, 5) is None


def test_cvar_parity_exact_tail_mass_closed_form_and_report_consumer(monkeypatch, tmp_path):
    np.testing.assert_allclose(pf._cvar_tail_weights(np.array([3., 2., 1., 0.]), .375), [2/3, 1/3, 0, 0])
    np.testing.assert_allclose(pf._cvar_tail_weights(np.array([3., 3., 1.]), .5), [.5, .5, 0])
    returns = np.linspace(-.05, .03, 100)[:, None] * [1, 2, 4]
    result = pf.cvar_risk_parity(returns, budget=.6)
    np.testing.assert_allclose(result["weights"], .6 * np.array([4/7, 2/7, 1/7]), atol=1e-6)
    np.testing.assert_allclose(result["shares"], np.full(3, 1/3), atol=1e-6)
    assert sum(result["contributions"]) == pytest.approx(result["es"])
    constant_loss = pf.cvar_risk_parity(np.tile([-.01, -.02], (30, 1)))
    np.testing.assert_allclose(constant_loss["weights"], [2/3, 1/3], atol=1e-6)
    assert pf.cvar_risk_parity(returns, bounds=(np.zeros(3), [.4, .8, .8])) is None
    for bad in (None, np.zeros((30, 3)), np.full((30, 3), .01), np.full((30, 3), np.nan), [[1, 2]]):
        assert pf.cvar_risk_parity(bad) is None
    assert pf.cvar_risk_parity(returns, alpha=1) is None
    assert pf.cvar_risk_parity(returns, budget=0) is None

    rng = np.random.default_rng(42)
    rets = rng.normal(0, .02, (250, 4)) * [1, 2, 3, 4]
    measured = pf.cvar_risk_parity(rets)
    np.testing.assert_array_equal(measured["weights"], pf.cvar_risk_parity(rets)["weights"])
    assert measured and measured["max_error"] < 1e-5
    np.testing.assert_allclose(pf.cvar_risk_parity(rets * 7)["weights"], measured["weights"], atol=1e-5)
    order = [2, 0, 3, 1]
    np.testing.assert_allclose(pf.cvar_risk_parity(rets[:, order])["weights"], measured["weights"][order], atol=1e-5)
    prices = 100 * np.exp(np.vstack([np.zeros(4), rets.cumsum(axis=0)]))
    panel = pd.DataFrame(prices, columns=list("ABCD"), index=pd.bdate_range("2025-01-01", periods=251))
    monkeypatch.setattr(pf, "PORTFOLIO_PROPOSALS_FILE", tmp_path / "rp_proposals.jsonl")
    got = dict(panel=panel, names=list(panel.columns), missing=[], rets=rets, ppy=252)
    monkeypatch.setattr(aq, "_symbol_panel", lambda *a, **k: (got, None))
    monkeypatch.setattr(aq, "_faktor_bloku", lambda *a: None)
    api = aq.quant_portfolio(max_weight=1, herc_clusters=2)
    assert "herc" in api["methods"] and "cvar_rp" in api["methods"]
    tail = api["methods"]["cvar_rp"]["tail_risk"]
    assert tail["es"] > 0 and tail["max_error"] < 1e-5
    np.testing.assert_allclose(tail["shares"], [.25] * 4, atol=1e-5)
