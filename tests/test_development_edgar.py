"""Toplu EDGAR: gerçek istemci kuyruğu, PIT önbellek ve analitik G bileşenleri."""
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

from nihai.ai import fundamentals as fu
from nihai.data import edgar_facts as ef
from nihai.data import insider
from nihai.web import api_quant as api


def _history(scale=1.0):
    return [{"_donem_sonu": f"{year}-12-31", "varliklar": 100.0,
             "net_kar": 10.0 * scale, "faaliyet_nakit": 15.0 * scale,
             "arge": 3.0 * scale, "capex": 5.0 * scale, "reklam": 2.0 * scale,
             "hasilat": 100.0 * 2 ** (year - 2020)} for year in range(2020, 2026)]


def test_medians_closed_form_missing_and_deterministic():
    histories = {str(n): _history(n) for n in (1, 2, 3)}
    result = fu.universe_medians(histories)
    assert result == fu.universe_medians(histories)
    assert result["medyanlar"]["roa"] == pytest.approx(0.2)
    assert result["medyanlar"]["cfo_varlik"] == pytest.approx(0.3)
    assert result["medyanlar"]["buyume_varyans"] == 0
    assert fu.universe_medians({"A": _history()})["medyanlar"]["roa"] is None
    histories["2"][-1]["faaliyet_nakit"] = np.nan
    assert fu.universe_medians(histories)["medyanlar"]["cfo_varlik"] is None
    assert all(v is None for v in fu.mohanram_metrics(None).values())


def test_all_eight_signals_have_measurements_but_no_imputed_inputs():
    history = _history()
    medians = {k: 0.001 if k.endswith("varyans") else 0.0
               for k in fu.mohanram_metrics(history)}
    result = fu.mohanram_g_score(history[-1], history[-2], medians, history=history)
    assert result["g_score"] == result["n_sinyal"] == 8
    assert result["eksik"] == []
    result = fu.mohanram_g_score({**history[-1], "reklam": None, "sga": 9999}, {}, medians)
    assert result["sinyaller"]["reklam_medyan_ustu"] is None
    assert result["sinyaller"]["kazanc_oynakligi_dusuk"] is None
    assert fu.mohanram_g_score(history[-1], {}, {"roa": np.nan})["sinyaller"]["roa_medyan_ustu"] is None
    assert fu.mohanram_metrics(history[:2])["roa_varyans"] is None
    assert fu.mohanram_metrics(history[:2] + history[3:])["roa_varyans"] is None
    history[2]["net_kar"] = np.inf
    assert fu.mohanram_metrics(history)["roa_varyans"] is None


def test_shared_sec_queue_limits_concurrent_dispatch_and_respects_429(monkeypatch):
    clock = [0.0]
    dispatch = []
    monkeypatch.setattr(insider, "_THROTTLE", 0.0)
    monkeypatch.setattr(insider, "_retry_after", 0.0)
    monkeypatch.setattr(insider, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda delay: clock.__setitem__(0, clock[0] + delay)))
    monkeypatch.setattr(insider, "_headers", lambda: {})

    def get(*args, **kwargs):
        dispatch.append(clock[0])
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {})

    monkeypatch.setattr(requests, "get", get)
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(insider._get, ["sec-test"] * 25)) == [{}] * 25
    assert min(np.diff(dispatch)) >= 0.11 - 1e-12
    assert max(sum(start <= t < start + 1 for t in dispatch) for start in dispatch) <= 10

    def limited(*args, **kwargs):
        return SimpleNamespace(status_code=429, headers={"Retry-After": "60"},
                               raise_for_status=lambda: (_ for _ in ()).throw(requests.HTTPError("429")))

    monkeypatch.setattr(requests, "get", limited)
    assert insider._get("sec-test") is None
    limited_at = clock[0]
    monkeypatch.setattr(requests, "get", get)
    insider._get("sec-test")
    assert dispatch[-1] - limited_at >= 60


def test_warm_queue_pit_snapshot_and_one_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(ef, "CACHE_DIR", tmp_path)
    calls = []

    def facts(symbol, **kwargs):
        calls.append(symbol)
        if symbol == "BAD":
            raise ValueError("bad issuer")
        return pd.DataFrame({"symbol": [symbol]})

    def annual(d, **kwargs):
        assert kwargs["as_of"] == "2026-06-30"
        return pd.DataFrame(_history()).set_index(pd.date_range("2020-12-31", periods=6, freq="YE"))

    monkeypatch.setattr(ef, "facts", facts)
    monkeypatch.setattr(ef, "annual_frame", annual)
    result = ef.warm(["AAA", "BBB", "CCC", "BAD", "aaa"], as_of="2026-06-30")
    assert result["istenen"] == 4 and result["olculen"] == 3
    assert result["eksik"] == ["BAD"]
    assert sorted(calls) == ["AAA", "BAD", "BBB", "CCC"]
    assert ef.median_snapshot(as_of="2026-06-29") is None
    assert ef.median_snapshot(as_of="2026-06-30")["medyanlar"]["roa"] == 0.1
    assert list(tmp_path.glob("*.csv")) == []
    json.loads(next(tmp_path.glob("*.json")).read_text("utf-8"))


def test_cache_read_never_fetches_even_when_refresh_requested(monkeypatch, tmp_path):
    monkeypatch.setattr(ef, "CACHE_DIR", tmp_path / "missing")
    calls = []
    monkeypatch.setattr(ef, "ticker_to_cik", lambda *a: calls.append(a))
    assert ef.facts("AAA", fetch=False, refresh=True).empty
    assert ef.facts("../escape", fetch=True).empty
    assert calls == [] and not (tmp_path / "missing").exists()


def test_historical_market_cap_excludes_future_prices(monkeypatch):
    bars = pd.DataFrame({"close": [10, 1000]}, index=pd.to_datetime(["2025-12-30", "2026-01-02"]))
    monkeypatch.setattr(api, "_bars_raw", lambda *a: bars)
    assert api._market_cap("AAA", 10, "2025-12-31")[0] == 100
    assert api._market_cap("AAA", 10, "2025-01-01")[0] is None


def test_distill_ignores_nonfinite_missing_and_unknown_tags():
    rows = [{"filed": "2026-01-01", "end": "2025-12-31", "val": val}
            for val in (None, "bad", np.nan, np.inf, 100)]
    result = ef.distill({"facts": {"us-gaap": {"Assets": {"units": {"USD": rows}},
                                               "Unknown": {"units": {"USD": rows}}}}})
    assert result["val"].tolist() == [100]
    assert ef.distill({}).empty
