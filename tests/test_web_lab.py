"""Yeni panelin gerçek iş akışı, güven sınırları ve ölçülmedi/hata durumları."""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nihai.web import api_lab as lab
from nihai.web import api_operations as ops
from nihai.web import lab_jobs as jobs
from nihai.web import web_dashboard as web

client = TestClient(web.app)


def test_panel_and_new_post_boundaries(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "WEB_TOKEN", "")
    monkeypatch.setattr(lab, "DATA_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(jobs, "start", lambda kind, params: calls.append((kind, params)) or {"job_id": "a" * 32})
    assert 'id="lab-backtest"' in client.get("/").text
    assert "function labEdgeRun" in client.get("/lab.js").text
    assert client.post("/api/backtest/run", json={"symbols": ["AAPL"]},
                       headers={"Origin": "https://foreign.invalid"}).status_code == 403
    assert client.post("/api/backtest/run", content='{"symbols":["AAPL"]}').status_code == 415
    for body in ({"symbols": ["--help"]}, {"symbols": ["AAPL"], "mc_runs": 5001},
                 {"symbols": ["AAPL"], "impact_coef": -1},
                 {"symbols": ["AAPL"], "impact_coef": 6},
                 {"symbols": ["AAPL"], "from_date": "2025-02-01", "to_date": "2024-01-01"},
                 {"symbols": ["AAPL"], "shell": "echo unsafe"}):
        assert client.post("/api/backtest/run", json=body).status_code == 422
    for params in ({"csv": 12}, {"csv": "../outside.csv"}, {"csv": "x.csv", "command": "bad"}):
        assert client.post("/api/research/run", json={"tool": "train_meta", "parameters": params}).status_code == 422
    assert not calls
    assert client.post("/api/backtest/run", json={"symbols": ["aapl", "AAPL"]}).status_code == 202
    assert calls[0][0] == "backtest" and calls[0][1]["symbols"] == ["AAPL"]
    monkeypatch.setattr(web, "WEB_TOKEN", "lab-test-secret")
    assert client.post("/api/backtest/run", json={"symbols": ["SPY"]}).status_code == 401


def test_real_worker_dsr_result_and_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "DATA_DIR", tmp_path)
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "web_jobs")
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    rng = np.random.default_rng(34)
    columns = list("abcde")
    pd.DataFrame(rng.normal(0.001, 0.01, (96, 5)), columns=columns).to_csv(tmp_path / "returns.csv", index=False)
    started = jobs.start("dsr", {"dataset": "returns.csv", "columns": columns, "n_trials": 10})
    job_id = started["job_id"]
    with pytest.raises(HTTPException) as duplicate:
        jobs.start("dsr", {})
    assert duplicate.value.status_code == 409
    deadline = time.monotonic() + 40
    while jobs.status(job_id)["status"] in ("running", "unobserved") and time.monotonic() < deadline:
        time.sleep(0.1)
    result = jobs.report(job_id)
    assert result["status"] == "completed", result
    assert result["result"]["n_trials"] == 10
    assert [c["column"] for c in result["result"]["columns"]] == columns
    assert all(0 <= c["dsr"] <= 1 for c in result["result"]["columns"])
    assert result["result"]["pbo"] is not None
    assert "NaN" not in json.dumps(result, allow_nan=False)
    monkeypatch.setattr(lab, "DATA_DIR", tmp_path)
    assert lab.execute("dsr", {"dataset": "returns.csv", "columns": ["a", "b"]}, tmp_path)["pbo"] is None
    # Yeniden açılmış panel eski işi sonuç dosyasından gösterebilir.
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    assert jobs.status(job_id)["status"] == "completed"
    failed = jobs.start("dsr", {"dataset": "missing.csv", "columns": ["x"]})["job_id"]
    deadline = time.monotonic() + 40
    while jobs.status(failed)["status"] in ("running", "unobserved") and time.monotonic() < deadline:
        time.sleep(0.1)
    assert jobs.report(failed)["status"] == "failed"
    assert "CSV" in jobs.status(failed)["error"]


def test_job_cancel_owns_process_and_redacts_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    job_id = "a" * 32
    path = tmp_path / job_id
    path.mkdir()
    jobs.write_json(path / "request.json", {"kind": "dsr", "created_at": time.time()})
    monkeypatch.setenv("EXAMPLE_API_KEY", "secret-for-lab-test")
    (path / "output.log").write_text("secret-for-lab-test https://api.telegram.org/bot123:token_value", encoding="utf-8")
    proc = SimpleNamespace(poll=lambda: None)
    stopped = []
    monkeypatch.setattr(jobs, "_PROCESSES", {job_id: proc})
    monkeypatch.setattr(jobs, "_stop", lambda target: stopped.append(target))
    assert jobs.cancel(job_id)["status"] == "cancelled"
    assert stopped == [proc]
    assert "secret-for-lab-test" not in jobs.status(job_id)["log"]
    assert "token_value" not in jobs.status(job_id)["log"]
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    with pytest.raises(HTTPException):
        jobs.cancel(job_id)
    with pytest.raises(HTTPException):
        jobs.report("../not-a-job")


def test_maintenance_exact_preview_and_archive_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "LOG_DIR", tmp_path)
    old = tmp_path / f"decision_log_{date.today()-timedelta(days=40):%Y_%m_%d}.txt.gz"
    warm = tmp_path / f"decision_log_{date.today()-timedelta(days=14):%Y_%m_%d}.txt"
    today = tmp_path / f"decision_log_{date.today():%Y_%m_%d}.txt"
    for path in (old, warm, today):
        path.write_text("measured evidence", encoding="utf-8")
    plan = lab.maintenance_plan(lab.MaintenanceInput())
    assert old.name in [p["file"] for p in plan["delete"]]
    assert old.exists() and warm.exists()
    params = {"apply": True, "confirmed": True, "plan_token": plan["plan_token"]}
    warm.write_text("changed evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="önizlemeden"):
        lab.execute("maintenance", params, tmp_path)
    assert old.exists()
    params["plan_token"] = lab.maintenance_plan(lab.MaintenanceInput())["plan_token"]
    lab.execute("maintenance", params, tmp_path)
    assert not old.exists() and not warm.exists()
    assert warm.with_suffix(".txt.gz").exists() and today.exists()


def test_tax_year_option_multiplier_and_bad_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "DATA_DIR", tmp_path)
    base = dict(symbol="AAPL", asset_type="stock", side="LONG", entry_time="2025-01-01",
                exit_time="2026-02-02", quantity=2, entry_price=10, exit_price=12, realized_pnl=4)
    pd.DataFrame([base, {**base, "symbol": "SPX", "asset_type": "option", "entry_price": 3, "exit_price": 4},
                  {**base, "exit_time": "bad", "realized_pnl": -10},
                  {**base, "entry_time": "2024-01-01", "exit_time": "2024-02-01"}]).to_csv(tmp_path / "trades.csv", index=False)
    report = lab.tax_report(2026)
    assert report["n"] == 1 and report["n_1256"] == 1
    assert report["long_term_total"] == 4 and report["total_1256"] == 200
    normal = lab.tax_download(2026, "8949").body.decode("utf-8-sig")
    special = lab.tax_download(2026, "6781").body.decode("utf-8-sig")
    assert "AAPL" in normal and "SPX" not in normal and "SPX" in special


def test_operations_do_not_claim_missing_measurements_or_send(monkeypatch, tmp_path):
    from nihai import audit_chain
    from nihai.notifier import deadman, telegram_bot
    monkeypatch.setattr(audit_chain, "CHAIN_PATH", tmp_path / "chain.jsonl")
    assert lab.audit_chain()["measured"] is False
    audit_chain.CHAIN_PATH.write_text("null\n", encoding="utf-8")
    assert lab.audit_chain()["ok"] is False
    assert lab.audit_chain()["merkle_root"] is None
    monkeypatch.setattr(ops, "DATA_DIR", tmp_path)
    monkeypatch.setattr(deadman, "check", lambda **kw: {"verdict": "SAGLIKLI", "files": {"log": {"age_min": None}}})
    monkeypatch.setattr(deadman, "_snitch", lambda url: pytest.fail("Kanıt yokken dış ping atılamaz"))
    assert ops.deadman_ping(ops.ConfirmInput(confirmed=True))["sent"] is False
    monkeypatch.setattr(telegram_bot, "DATA_DIR", tmp_path)
    response = SimpleNamespace(status_code=429, json=lambda: {"ok": False})
    monkeypatch.setattr(telegram_bot, "_http", lambda: SimpleNamespace(post=lambda *a, **k: response))
    assert telegram_bot._post("test") is False
    with (tmp_path / "telegram_status.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("null\ninvalid-json\n")
    assert telegram_bot.status()["last_send"]["ok"] is False
    monkeypatch.setattr(ops, "HUB_URL", "https://remote.invalid")
    with pytest.raises(ValueError, match="yerel"):
        ops._start_hub()
    for health in ({"pid": 123}, {"pid": True, "service_root": str(ops.ROOT)},
                   {"pid": 123, "service_root": str(tmp_path)}):
        with pytest.raises(ValueError):
            ops._hub_pid(health)


def test_runtime_selection_applies_only_when_not_explicit(tmp_path, monkeypatch):
    from nihai import config
    from nihai import runtime_strategy as runtime
    from nihai.signals import strategy
    monkeypatch.setattr(runtime, "PATH", tmp_path / "runtime.json")
    monkeypatch.setattr(runtime, "_overrides", {})
    monkeypatch.setattr(runtime, "_stamp", None)
    prior = config.CONFIG.strategy
    monkeypatch.setattr(config, "CONFLUENCE_MIN_VOTES", config.CONFLUENCE_MIN_VOTES)
    selected = {"strategy": "confluence", "min_votes": 3, "stop_atr": 2.2, "target_atr": 4.4}
    try:
        runtime.save(selected)
        assert runtime.apply() == "confluence"
        assert config.CONFIG.strategy.trend_stop_atr == 2.2
        assert config.CONFLUENCE_MIN_VOTES == 3
        selected.update(strategy="ensemble", min_votes=4)
        runtime.PATH.write_text(json.dumps(selected), encoding="utf-8")
        assert runtime.apply() == "ensemble"
        before = runtime.PATH.read_text(encoding="utf-8")
        with pytest.raises(ValueError):
            runtime.save({**selected, "stop_atr": float("nan")})
        assert runtime.PATH.read_text(encoding="utf-8") == before
        monkeypatch.setattr(runtime, "apply", lambda: pytest.fail("Açık strateji seçimi geçersiz kılınamaz"))
        strategy.generate_signal("TEST", pd.DataFrame(), strategy_mode="macd")
    finally:
        object.__setattr__(config.CONFIG, "strategy", prior)


def test_edge_data_leakage_rejected_and_gates_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "DATA_DIR", tmp_path)
    dates = pd.bdate_range("2025-01-01", periods=100)
    pd.DataFrame({"ts": dates, "symbol": ["AAPL"] * 100, "x": np.arange(100),
                  "label": np.arange(100) % 2}).to_csv(tmp_path / "edge.csv", index=False)
    params = dict(name="web_test_family", dataset="edge.csv", feature_cols=["x"],
                  expected_start="2025-01-01", expected_end=dates[-1].date().isoformat())
    with pytest.raises(ValueError, match="özellik olamaz"):
        lab.EdgeInput(**{**params, "feature_cols": ["label"]})
    report = lab.execute("edge_lab", params, tmp_path)
    assert report["candidate"] == params["name"] and report["gates"]
    assert not all(g["passed"] for g in report["gates"])  # tek sembol: dayanıklı terfi değil
    assert report["verdict"]


@pytest.mark.parametrize("mode", ["single", "portfolio", "options", "walk_forward"])
def test_backtest_modes_use_real_engine_with_local_bars(mode, monkeypatch, tmp_path):
    from dataclasses import replace

    from nihai.config import CONFIG
    from nihai.execution import backtest as bt
    price = 100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0.001, 0.015, 160)))
    bars = pd.DataFrame({"open": price, "close": price, "high": price * 1.02,
                         "low": price * .98, "volume": 1_000_000}, index=pd.bdate_range("2024-01-01", periods=160))
    monkeypatch.setattr(bt, "load_bars", lambda *a, **kw: bars.copy())
    if mode == "options":
        from nihai.data import dolt_options
        history = pd.DataFrame([dict(date=day, act_symbol="AAPL", expiration=day + pd.Timedelta(days=35),
            strike=100., call_put="Call", bid=2., ask=2.2, vol=.3, delta=.5, gamma=.02, theta=-.03, vega=.1)
            for day in bars.index])
        monkeypatch.setattr(dolt_options, "local_option_history",
                            lambda *a, **kw: dolt_options.prepare_option_history(history, "AAPL"))
    prior = CONFIG.strategy
    monkeypatch.setitem(vars(CONFIG.backtest), "impact_coef", CONFIG.backtest.impact_coef)
    object.__setattr__(CONFIG, "strategy", replace(prior))
    try:
        result = lab.execute("backtest", {"symbols": ["AAPL"], "mode": mode, "strategy": "macd",
                                          "impact_coef": .4, "mc_runs": 10, "is_bars": 70, "oos_bars": 40}, tmp_path)
        assert bt._bt_slippage().impact_coef == .4
        assert result["mode"] == mode
        assert len(result["results"]) == 1
        assert result["results"][0]["initial_equity"] > 0
        assert result["results"][0].get("bars_total", len(result["results"][0].get("folds", []))) > 0
    finally:
        object.__setattr__(CONFIG, "strategy", prior)


def test_mc_bands_include_path_ruin_without_changing_legacy_metrics():
    from nihai.execution.backtest import monte_carlo_paths
    base = monte_carlo_paths([-120, 200], 100, 100)
    expanded = monte_carlo_paths([-120, 200], 100, 100, include_bands=True)
    assert all(expanded[key] == value for key, value in base.items())
    assert expanded["ruin_during_path_probability"] > 0
    for curve in expanded["bands"].values():
        assert curve[0] == 100 and curve[-1] == 180


def test_stale_dolt_and_failed_ping_are_not_success(tmp_path, monkeypatch):
    import os

    import requests

    from nihai.data import dolt_options
    from nihai.notifier import deadman
    cached = tmp_path / "cache.csv"
    cached.write_text("stale", encoding="utf-8")
    old = time.time() - 10 * 86400
    os.utime(cached, (old, old))
    monkeypatch.setattr(dolt_options, "bulk_csv", lambda **kw: cached)
    with pytest.raises(ValueError, match="yenilenemedi"):
        lab.execute("tool", {"tool": "dolt", "parameters": {"table": "volatility_history"}}, tmp_path)
    assert cached.read_text(encoding="utf-8") == "stale"
    def reject():
        raise requests.HTTPError("503")
    monkeypatch.setattr(requests, "get", lambda *a, **kw: SimpleNamespace(raise_for_status=reject))
    assert deadman._snitch("https://example.invalid/mock") is False
