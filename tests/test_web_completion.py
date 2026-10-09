"""Kalan arayüz akışlarının gerçek veri, terfi ve gönderim sınırları."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nihai.web import api_operations as ops
from nihai.web import lab_jobs as jobs
from nihai.web.web_dashboard import app


def test_tick_latency_uses_event_time_and_preserves_unknown_and_clock_skew(monkeypatch):
    from nihai.data import alpaca_stream_provider as stream
    p = stream.AlpacaStreamProvider(rest=SimpleNamespace())
    stamp = datetime(2026, 9, 8, tzinfo=UTC)
    monkeypatch.setattr(stream.time, "time", lambda: stamp.timestamp() + .25)
    asyncio.run(p._on_trade(SimpleNamespace(symbol="AAPL", price=100, timestamp=stamp)))
    asyncio.run(p._on_quote(SimpleNamespace(symbol="AAPL", bid_price=99, ask_price=101)))
    asyncio.run(p._on_quote(SimpleNamespace(symbol="AAPL", bid_price=99, ask_price=101,
                                           timestamp=stamp.replace(tzinfo=None))))
    status = p.stream_status()["tick_latency"]
    assert status["trades"]["last"]["delay_ms"] == pytest.approx(250)
    assert status["quotes"]["p50_ms"] is None and status["quotes"]["missing_timestamps"] == 2
    asyncio.run(p._on_trade(SimpleNamespace(symbol="AAPL", price=100, timestamp=stamp + timedelta(seconds=1))))
    assert p.stream_status()["tick_latency"]["trades"]["last"]["delay_ms"] == -750
    assert p.stream_status()["tick_latency"]["trades"]["negative_samples"] == 1
    for _ in range(520):
        p._measure_tick("trades", "AAPL", stamp, stamp.timestamp() + .5)
    assert p.stream_status()["tick_latency"]["trades"]["samples"] == 512


def test_cash_calendar_matches_state_wash_window_and_exemptions(tmp_path, monkeypatch):
    from nihai.config import CONFIG
    from nihai.execution.state import StateStore
    object_days = CONFIG.regulation.wash_sale_days
    store = StateStore(tmp_path / "state.json")
    today = date(2026, 9, 8)
    loss_date = today - timedelta(days=object_days)
    store.loss_history = [{"symbol": "AAPL", "date": loss_date.isoformat(), "loss": -20},
                          {"symbol": "SPX", "date": loss_date.isoformat(), "loss": -30}]
    store.unsettled_lots = [{"settle_date": today.isoformat(), "amount": 50},
                            {"settle_date": (today + timedelta(days=1)).isoformat(), "amount": 100}]
    report = store.cash_timeline(today)
    assert report["settlements"][0]["matured"] and report["settlements"][1]["days_left"] == 1
    assert [r["symbol"] for r in report["quarantine"]] == ["AAPL"]
    assert report["quarantine"][0]["days_left"] == 1
    assert store.cash_timeline(today + timedelta(days=1))["quarantine"] == []
    assert len(store.unsettled_lots) == 2  # Görüntüleme kayıt silmez.
    from nihai.web import web_dashboard as web
    store.path.write_text(json.dumps({"loss_history": store.loss_history,
        "unsettled_lots": store.unsettled_lots, "saved_at": "known"}), encoding="utf-8")
    monkeypatch.setattr(web, "StateStore", lambda: StateStore(store.path))
    monkeypatch.setattr(web, "now_et_naive", lambda: datetime.combine(today, datetime.min.time()))
    result = TestClient(app).get("/api/portfolio/timeline").json()
    assert result["quarantine"][0]["release_date"] == "2026-09-09"
    assert result["saved_at"] == "known" and result["measured"]


def test_promotion_binds_evidence_to_model_and_reloads_without_restart(tmp_path, monkeypatch):
    from nihai.execution import prediction_tracker as tracker
    from nihai.runtime_strategy import settings
    from nihai.signals import meta_gate as mg
    from nihai.signals import model_promotion as promotion
    from nihai.signals.meta_features import FEATURE_NAMES

    monkeypatch.setattr(mg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mg, "_GATE", None)
    monkeypatch.setattr(promotion, "DATA_DIR", tmp_path)
    monkeypatch.setattr(promotion, "PATH", tmp_path / "meta_activation.json")
    n = len(FEATURE_NAMES)
    model = {"w": [0] * (n + 1), "mu": [0] * n, "sd": [1] * n,
             "features": list(FEATURE_NAMES), "gate_threshold": .5}
    model_path = tmp_path / "meta_model.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    first = mg.get_gate()
    assert first.model_id and mg.get_gate() is first
    strategy = settings()["strategy"]
    rng = np.random.default_rng(7)
    rows = []
    for i in range(1200):
        high = i % 2 == 0
        ret = (1 if high else -1) + rng.normal(0, .25)
        rows.append({"symbol": f"S{i % 11}", "logged_at": (datetime(2026, 7, 1) + timedelta(hours=i)).isoformat(),
            "model": None, "meta_model_id": first.model_id, "meta_strategy": strategy, "interval": "1m",
            "pwin": .8 if high else .2, "pwin_side": "LONG", "hit": "1" if ret > 0 else "0",
            "direction": "+", "fwd_ret_pct": ret, "price": 100, "bar_close": 100, "fwd_price": 100 + ret})
    monkeypatch.setattr(tracker, "_read_log_rows", lambda: [{**r, "meta_model_id": "old"} for r in rows])
    assert not promotion.review("1m")["eligible"]
    monkeypatch.setattr(tracker, "_read_log_rows", lambda: rows)
    approved = promotion.review("1m")
    assert approved["eligible"], approved["checks"]
    assert approved["matched_rows"] == len(rows)
    assert not promotion.review("1h")["eligible"]
    result = promotion.activate(approved)
    assert result["activated"]
    assert promotion.live_policy("off", "1m", strategy) == ("enforce", approved["threshold"])
    assert promotion.live_policy("off", "1h", strategy) == ("off", None)
    assert promotion.status()["applied"][0]["mode"] == "enforce"
    assert promotion.demote()["success"]
    assert promotion.live_policy("off", "1m", strategy) == ("shadow", approved["threshold"])
    promotion.activate(approved)
    model["gate_threshold"] = .6
    model_path.write_text(json.dumps(model), encoding="utf-8")
    assert mg.get_gate().model_id != first.model_id
    assert promotion.live_policy("off", "1m", strategy) == ("shadow", None)
    with pytest.raises(ValueError):
        promotion.activate(approved)
    with pytest.raises(ValueError):
        promotion.activate({**approved, "reviewed_at": time.time() - 1801})


def test_chart_preview_and_exact_image_send_with_confirmation(tmp_path, monkeypatch):
    from nihai.notifier import chart_render, telegram_bot
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    job_id = "d" * 32
    path = tmp_path / job_id
    path.mkdir()
    params = {"symbol": "AAPL", "interval": "1h", "lookback": 80}
    jobs.write_json(path / "request.json", {"kind": "telegram_chart", "parameters": params, "created_at": time.time()})
    image = b"\x89PNG\r\n\x1a\nchecked-image"
    monkeypatch.setattr(chart_render, "render_price_png", lambda **kw: image if kw == params else None)
    result = ops.execute_operation("telegram_chart", params, path)
    jobs.write_json(path / "result.json", {"status": "completed", "result": result})
    monkeypatch.setattr(telegram_bot, "is_enabled", lambda: True)
    sent = []
    monkeypatch.setattr(telegram_bot, "_post_photo", lambda data, caption: sent.append((data, caption)) or True)
    client = TestClient(app)
    assert client.get(f"/api/telegram/chart/{job_id}.png").content == image
    url = f"/api/telegram/chart/{job_id}/send"
    assert client.post(url, json={}).status_code == 422 and not sent
    assert client.post(url, json={"confirmed": True}).json()["success"]
    assert sent == [(image, "AAPL · 1h")]
    monkeypatch.setattr(telegram_bot, "_post_photo", lambda *a: False)
    assert client.post(url, json={"confirmed": True}).json()["success"] is False
    assert client.post("/api/telegram/chart", json={"symbol": "../../bad"}).status_code == 422
    assert client.post("/api/ml/activate-shadow", json={"review_job": job_id}).status_code == 422
    assert client.post("/api/ml/activate-shadow", json={"review_job": job_id, "confirmed": True}).status_code == 409
