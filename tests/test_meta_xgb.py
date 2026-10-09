"""XGBoost meta-model: eğitim + serileştirme + canlı kapı serving round-trip.

Kapsanan değişmezler:
- XGBModel.fit ham VE kalibre modunda [0,1] olasılık üretir.
- train_meta'nın kaydettiği pkl formatı (`{"clf":..., "features":..., "gate_threshold":...}`)
  _MetaGate tarafından yüklenir → active, predict_proba serving yolu çalışır.
- Kalibrasyon serving'i bozmaz (CalibratedClassifierCV de predict_proba sunar).

xgboost kurulu değilse testler atlanır (üretimde de meta_gate try/except ile pasife düşer)."""
import pickle

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")  # kurulu değilse tüm modülü atla

import nihai.signals.meta_gate as mg  # noqa: E402
from nihai.signals.meta_features import FEATURE_NAMES  # noqa: E402
from research.train_meta import XGBModel  # noqa: E402


def _synth(n=400, seed=0):
    """Ayrılabilir sentetik veri: feature 0 etiketle korele → AUC > 0.5 beklenir."""
    rng = np.random.default_rng(seed)
    d = len(FEATURE_NAMES)
    X = rng.normal(size=(n, d))
    logit = 1.5 * X[:, 0] - 0.8 * X[:, 1]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float)
    return X, y


def test_xgb_fit_raw_probs_in_range():
    X, y = _synth()
    m = XGBModel.fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X),)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0
    # ayrılabilir veride eğitim AUC anlamlı şekilde 0.5 üstü olmalı (model gerçekten öğreniyor)
    from research.train_meta import _auc
    assert _auc(y, p) > 0.7


def test_xgb_fit_calibrated_probs_in_range():
    X, y = _synth(seed=1)
    m = XGBModel.fit(X, y, calibrate="isotonic")
    p = m.predict_proba(X)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


def test_xgb_pkl_round_trip_through_gate(tmp_path, monkeypatch):
    X, y = _synth(seed=2)
    m = XGBModel.fit(X, y)
    # train_meta.save formatını bire bir taklit et
    meta = {"features": list(FEATURE_NAMES), "kind": "xgb", "gate_threshold": 0.55}
    with open(tmp_path / "meta_model.pkl", "wb") as f:
        pickle.dump({"clf": m.clf, **meta}, f)

    monkeypatch.setattr(mg, "DATA_DIR", tmp_path)
    # serving yolunu izole et: extract_features yerine sabit feats sözlüğü
    # Tur-43: imza `shared=` kwarg'ı ile genişledi → stub da kabul etmeli.
    monkeypatch.setattr(mg, "extract_features",
                        lambda df, sig, **_: {n: 0.0 for n in FEATURE_NAMES})
    g = mg._MetaGate()
    assert g.active is True
    assert g.threshold == 0.55
    prob = g.probability(df_slice="dummy", signal="dummy")
    assert prob is not None
    assert 0.0 <= prob <= 1.0


def test_gate_passive_when_xgb_clf_unpicklable(tmp_path, monkeypatch):
    # Bozuk pkl → kapı pasif (fail-safe), çökme yok
    (tmp_path / "meta_model.pkl").write_bytes(b"not a real pickle")
    monkeypatch.setattr(mg, "DATA_DIR", tmp_path)
    g = mg._MetaGate()
    assert g.active is False and g.kind is None
