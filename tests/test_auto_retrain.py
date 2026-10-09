"""Otomatik eğitim: gerçek pozitif kontrol, tekrar/yarım koşu ve üretim izolasyonu."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from nihai.ai import learning
from nihai.signals.meta_features import FEATURE_NAMES
from research import auto_retrain as auto
from research import train_meta as tm


def _data(n=600):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = (x[:, 0] + .3 * rng.normal(size=n) > 0).astype(int)
    df = pd.DataFrame(x, columns=FEATURE_NAMES)
    df["label"], df["pnl"] = y, np.where(y == 1, 1., -1.)
    df["symbol"] = [f"S{i % 20}" for i in range(n)]
    df["entry_time"] = pd.date_range("2025-01-01", periods=n, freq="4h").astype(str)
    df.attrs["cohorts"] = [("1m", "5")]
    return df


def test_cycle_persists_tested_candidate_and_resumes_without_live_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(auto, "RUNS", tmp_path / "retraining")
    monkeypatch.setattr(auto, "KINDS", ("logistic",))
    monkeypatch.setattr(tm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(learning, "DATA_DIR", tmp_path)
    live = tmp_path / "meta_model.json"
    live.write_text("production", encoding="utf-8")
    data = [_data()]
    monkeypatch.setattr(auto.build_live_dataset, "build", lambda: data[0])
    calls = []
    assess = auto.live_retrain.assess

    def checked(*args, **kwargs):
        result = assess(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(auto.live_retrain, "assess", checked)
    result = auto.run()
    assert result["verdict"] == "DURABLE" and result["promoted"] is False
    root = auto.RUNS / result["run_id"]
    candidate = json.loads((root / "meta_model.json").read_text())
    assert candidate["w"] == calls[0]["results"]["logistic"]["splits"]["time"]["model"].w.tolist()
    assert candidate["gate_threshold"] == result["threshold"]
    assert candidate["provenance"]["dataset_sha256"] == result["dataset_sha256"]
    assert hashlib.sha256((root / "dataset.csv").read_bytes()).hexdigest() == result["dataset_sha256"]
    assert "DURABLE" in learning.learning_report({})
    latest = auto.RUNS / "latest.json"
    latest.unlink()  # report tamamlandıktan sonraki latest yazımı kesilmiş olsun
    assert auto.run()["status"] == "UNCHANGED" and len(calls) == 1
    assert json.loads(latest.read_text())["run_id"] == result["run_id"]
    data[0] = _data(700)
    assert auto.run()["status"] == "WAITING_DATA" and len(calls) == 1
    data[0] = _data(900).iloc[600:].copy()  # dosya kısalsa da 300 YENİ satır birikti
    assert auto.run()["status"] == "COMPLETED" and len(calls) == 2
    before = latest.read_bytes()
    data[0] = _data(1200)

    def fail(*args, **kwargs):
        raise RuntimeError("training interrupted")

    monkeypatch.setattr(auto.live_retrain, "assess", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        auto.run()
    assert latest.read_bytes() == before
    monkeypatch.setattr(auto.live_retrain, "assess", checked)
    assert auto.run()["status"] == "COMPLETED" and len(calls) == 3
    before = latest.read_bytes()
    data[0].loc[0, "entry_time"] = "invalid"
    with pytest.raises(ValueError, match="tarihler"):
        auto.run()
    assert latest.read_bytes() == before
    assert live.read_text() == "production"


def test_process_lock_is_live_not_a_stale_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(auto, "RUNS", tmp_path)
    seen = []

    def nested():
        seen.append(auto.run()["status"])
        return pd.DataFrame()

    monkeypatch.setattr(auto.build_live_dataset, "build", nested)
    assert auto.run()["status"] == "INSUFFICIENT"
    assert auto.run()["status"] == "INSUFFICIENT"
    assert seen == ["BUSY", "BUSY"]  # kapatılan tanıtıcının kilidi ikinci dış koşuyu engellemez


def test_mixed_dates_are_all_sorted_in_time_order():
    df = pd.DataFrame({"entry_time": ["2026-01-02T10:00:00-05:00", "2026-01-01 14:00:00",
                                     "2026-01-02T14:00:00+00:00"], "id": [2, 0, 1]})
    assert tm._chrono_key(df.entry_time).notna().all()
    tr, te = tm.split_time(df, 2 / 3)
    assert tr.id.tolist() == [0, 1] and te.id.tolist() == [2]
