"""KALEM 3 — save_model provenance izi (code_sha, dataset_sha256, ECE, konfig sayısı, ...).

provenance dict save_model()'e verilirse meta['provenance'] olarak dosyalara gömülür:
  - lojistik → meta_model.json
  - gbm/xgb  → meta_model.pkl (dict içinde) + meta_model_info.json

meta_gate._load() yalnız bilinen anahtarları (w/mu/sd veya clf, features, gate_threshold, kind)
okur → fazladan 'provenance' anahtarı SESSİZCE yok sayılır, yükleme sözleşmesi bozulmaz. Bu
dosya meta_gate.py'yi YALNIZ İTHAL EDİP KULLANIR — değiştirmez (görev sınırı; nihai/** başka
ajanın alanı). İzolasyon: mevcut test_meta_gate.py'nin kurduğu konvansiyon aynen izlenir —
`monkeypatch.setattr(<modül>, "DATA_DIR", tmp_path)` — conftest'in PAYLAŞILAN oturum-genelindeki
NIHAI_DATA_DIR tmp'sine YAZMAYIZ (başka test dosyalarını kirletmemek için)."""
from __future__ import annotations

import hashlib
import json
import pickle
import re

import numpy as np
import pytest

from nihai.signals.meta_features import FEATURE_NAMES
from research.train_meta import (
    LogisticModel,
    _code_sha,
    _dataset_sha256,
    _provenance,
    save_model,
)


# ---------------------------------------------------------------------------
# _provenance() / _dataset_sha256() / _code_sha() — saf yardımcılar
# ---------------------------------------------------------------------------
def test_dataset_sha256_matches_hashlib_for_real_file(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes(b"hello-world-content")
    assert _dataset_sha256(str(p)) == hashlib.sha256(b"hello-world-content").hexdigest()


def test_dataset_sha256_none_for_missing_or_nonfile_path(tmp_path):
    assert _dataset_sha256(str(tmp_path / "missing.csv")) is None
    assert _dataset_sha256("canlı loglar (build_live_dataset)") is None


def test_code_sha_returns_str_or_none():
    sha = _code_sha()
    assert sha is None or (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{4,16}", sha))


def test_provenance_fields_present_and_correct(tmp_path):
    csv = tmp_path / "ds.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    prov = _provenance(dataset_path=str(csv), n_rows=123, train_frac=0.6, min_gate_keep=50,
                       n_configs_tried=16, calibrate=None, ece_test=0.0456)
    assert prov["dataset_sha256"] == hashlib.sha256(csv.read_bytes()).hexdigest()
    assert prov["n_rows"] == 123
    assert prov["n_features"] == len(FEATURE_NAMES)
    assert prov["seed"] == 42
    assert prov["train_frac"] == 0.6
    assert prov["min_gate_keep"] == 50
    assert prov["threshold_selection"] == "train"
    assert prov["n_configs_tried"] == 16
    assert prov["calibrate"] is None
    assert prov["ece_test"] == 0.0456
    assert isinstance(prov["saved_at"], str) and prov["saved_at"]
    assert "code_sha" in prov


def test_provenance_ece_none_when_not_provided():
    prov = _provenance(dataset_path="x.csv", n_rows=1, train_frac=0.6, min_gate_keep=50,
                       n_configs_tried=8, calibrate="isotonic", ece_test=None)
    assert prov["ece_test"] is None
    assert prov["calibrate"] == "isotonic"


# ---------------------------------------------------------------------------
# save_model(provenance=...) — dosyalara gömülme (her iki dal: json / pkl+info)
# ---------------------------------------------------------------------------
def _fit_logistic():
    n = len(FEATURE_NAMES)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, n))
    y = (X[:, 0] > 0).astype(float)
    return LogisticModel.fit(X, y)


class _FakeGBM:
    """save_model yalnız .kind ve (gbm/xgb dalında) .clf'e dokunur — gerçek sklearn fit
    gerektirmeden aynı kod yolunu (pickle.dump({'clf':..., **meta}, f)) egzersiz eder."""
    kind = "gbm"

    def __init__(self):
        self.clf = {"stub": True}


def test_save_model_embeds_provenance_in_json_for_logistic(tmp_path, monkeypatch):
    import research.train_meta as tm

    monkeypatch.setattr(tm, "DATA_DIR", tmp_path)
    prov = _provenance(dataset_path="x.csv", n_rows=60, train_frac=0.6, min_gate_keep=50,
                       n_configs_tried=16, calibrate=None, ece_test=0.05)
    path = tm.save_model(_fit_logistic(), trained_on="x.csv", oos_auc=0.55,
                         gate_threshold=0.55, provenance=prov)
    assert path == tmp_path / "meta_model.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["provenance"]["n_rows"] == 60
    assert data["provenance"]["threshold_selection"] == "train"
    assert data["provenance"]["n_configs_tried"] == 16


@pytest.mark.parametrize("candidate", [False, True])
def test_save_model_embeds_provenance_in_pkl_and_info_json_for_gbm(tmp_path, monkeypatch, candidate):
    import research.train_meta as tm

    monkeypatch.setattr(tm, "DATA_DIR", tmp_path)
    target = tmp_path / "candidate" if candidate else tmp_path
    live = tmp_path / "meta_model.pkl"
    if candidate:
        live.write_bytes(b"production")
    prov = _provenance(dataset_path="y.csv", n_rows=50, train_frac=0.6, min_gate_keep=50,
                       n_configs_tried=8, calibrate="sigmoid", ece_test=0.02)
    path = tm.save_model(_FakeGBM(), trained_on="y.csv", oos_auc=0.58,
                         gate_threshold=0.5, provenance=prov,
                         output_dir=target if candidate else None)
    assert path == target / "meta_model.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f)
    assert d["provenance"]["calibrate"] == "sigmoid"
    info = json.loads((target / "meta_model_info.json").read_text(encoding="utf-8"))
    assert info["provenance"]["n_rows"] == 50
    if candidate:
        assert live.read_bytes() == b"production"
        assert not (tmp_path / "meta_model_info.json").exists()


def test_save_model_without_provenance_omits_key(tmp_path, monkeypatch):
    """provenance verilmezse (None, varsayılan) 'provenance' anahtarı dosyada HİÇ olmamalı —
    geriye-dönük uyumluluk (eski çağıranlar/yüklemeler etkilenmez)."""
    import research.train_meta as tm

    monkeypatch.setattr(tm, "DATA_DIR", tmp_path)
    path = tm.save_model(_fit_logistic(), trained_on="x.csv", oos_auc=0.55, gate_threshold=0.55)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "provenance" not in data


# ---------------------------------------------------------------------------
# meta_gate sözleşmesi kırılmıyor — İTHAL EDİP KULLAN, DEĞİŞTİRME.
# ---------------------------------------------------------------------------
def test_meta_gate_loads_model_with_provenance_present(tmp_path, monkeypatch):
    import nihai.signals.meta_gate as mg
    import research.train_meta as tm

    monkeypatch.setattr(tm, "DATA_DIR", tmp_path)
    prov = _provenance(dataset_path="z.csv", n_rows=60, train_frac=0.6, min_gate_keep=50,
                       n_configs_tried=16, calibrate=None, ece_test=None)
    tm.save_model(_fit_logistic(), trained_on="z.csv", oos_auc=0.55,
                 gate_threshold=0.62, provenance=prov)

    monkeypatch.setattr(mg, "DATA_DIR", tmp_path)
    gate = mg._MetaGate()   # doğrudan örnekle (get_gate() singleton'ını başka testlerle paylaşma)
    assert gate.active is True
    assert gate.kind == "logistic"
    assert gate.threshold == pytest.approx(0.62)
    assert isinstance(gate._log[0], np.ndarray)   # w/mu/sd normal şekilde parse edildi


def test_meta_gate_ignores_provenance_key_gracefully(tmp_path, monkeypatch):
    """'provenance' anahtarı JSON'da OLSA BİLE meta_gate._load onu okumaz/kullanmaz — fazladan
    anahtar sessizce yok sayılır (sözleşme: yalnız w/mu/sd/features/gate_threshold okunur)."""
    import nihai.signals.meta_gate as mg

    n = len(FEATURE_NAMES)
    model = {
        "w": [0.0] + [0.1] * n, "mu": [0.0] * n, "sd": [1.0] * n,
        "features": list(FEATURE_NAMES), "gate_threshold": 0.7,
        "provenance": {"code_sha": "deadbeef", "n_rows": 999, "extra_unknown_field": "x"},
    }
    (tmp_path / "meta_model.json").write_text(json.dumps(model), encoding="utf-8")
    monkeypatch.setattr(mg, "DATA_DIR", tmp_path)
    gate = mg._MetaGate()
    assert gate.active is True
    assert gate.kind == "logistic"
    assert gate.threshold == pytest.approx(0.7)
