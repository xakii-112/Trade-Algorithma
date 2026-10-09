"""Takvim eşitliği, merkezî farkın gizlediği düşüş ve pano ısı haritası."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nihai.ai import options_math as om
from nihai.web import api_quant as api


def slices(variances):
    return [dict(t=.1 * (i + 1), params=dict(a=w, b=0., rho=0., m=0., sigma=.1))
            for i, w in enumerate(variances)]


def test_calendar_equality_is_not_arbitrage_and_adjacent_drop_cannot_hide():
    flat = om.dupire_local_vol_svi(slices([.04, .04]), n_k=7)
    assert flat["n_calendar_ihlal"] == 0
    assert np.all(flat["calendar_slope"] == 0) and np.all(flat["durrleman_g"] == 1)
    assert np.isnan(flat["local_vol"]).all()  # sıfır türev ayrı bir tanımsız yerel-vol sınırı
    drop = om.dupire_local_vol_svi(slices([.04, .02, .06]), n_k=7)
    assert drop["n_calendar_ihlal"] == 7
    np.testing.assert_allclose(drop["calendar_slope"][0], -.2)
    np.testing.assert_allclose(drop["calendar_slope"][1], .4)
    assert np.isnan(drop["local_vol"][:2]).all()
    np.testing.assert_allclose(drop["local_vol"][2], np.sqrt(.4))
    again = om.dupire_local_vol_svi(slices([.04, .02, .06]), n_k=7)
    np.testing.assert_equal(drop["local_vol"], again["local_vol"])


def test_butterfly_heatmap_agrees_with_closed_form_and_defensive_inputs():
    params = dict(a=.0001, b=2.2, rho=.8, m=0., sigma=.05)
    k = np.array([-.2, 0., .2])
    result = om.dupire_local_vol_svi([dict(t=.1, params=params),
        dict(t=.2, params={**params, "a": .01})], k_grid=k)
    w = params["a"] + params["b"] * (params["rho"] * k + np.sqrt(k*k + params["sigma"]**2))
    first = params["b"] * (params["rho"] + k / np.sqrt(k*k + params["sigma"]**2))
    second = params["b"] * params["sigma"]**2 / (k*k + params["sigma"]**2)**1.5
    expected = (1-k*first/(2*w))**2 - first**2/4*(1/w+.25) + second/2
    np.testing.assert_allclose(result["durrleman_g"][0], expected)
    assert expected[-1] < 0 and result["n_butterfly_ihlal"] > 0
    for bad in (None, [], [dict(t=.1, params=params)], slices([np.nan, np.inf])):
        assert om.dupire_local_vol_svi(bad) is None
    assert om.dupire_local_vol_svi(slices([.02, .04]), n_k=np.inf) is None


def test_smile_api_exposes_heatmap_from_same_observed_chains(monkeypatch):
    expiries = [(date.today() + timedelta(days=d)).isoformat() for d in (30, 60, 90)]
    class Provider:
        def get_option_chain(self, symbol, expiry=None):
            side = pd.DataFrame(dict(strike=np.linspace(70, 130, 13), impliedVolatility=.25,
                                     bid=2., ask=2.1, openInterest=100, volume=10))
            return dict(calls=side, puts=side, underlying_price=100., expiry=expiry or expiries[0],
                        all_expiries=expiries)
    monkeypatch.setattr(api, "_provider", lambda: Provider())
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        data = client.get("/api/quant/smile?symbol=AAA").json()
    assert data["ok"]
    surface = data["dupire"]
    heatmap = surface["arbitrage_heatmap"]
    assert len(heatmap["butterfly_g"]) == 3 and len(heatmap["calendar_slope"]) == 2
    assert all(len(row) == len(surface["k"]) for row in heatmap["butterfly_g"])
    assert heatmap["scope"] == "sampled_k_grid_not_global_arbitrage_proof"
    assert surface["n_calendar_ihlal"] == 0
