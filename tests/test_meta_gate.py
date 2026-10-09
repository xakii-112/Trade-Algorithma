"""meta_gate yükleme/fail-safe davranışı testleri (opt-in disiplini).

Kritik değişmez: model yoksa VEYA bozuksa kapı PASİF olmalı (probability→None, hiçbir
sinyal elenmez) — doğrulanmamış kaldıraç sessizce sızmasın. DATA_DIR tmp_path'e
yönlendirilir; gerçek model dosyaları yazılır, ağ/global cache yok."""
import json

import numpy as np

import nihai.signals.meta_gate as mg
from nihai.signals.meta_features import FEATURE_NAMES


def _gate(tmp_path, monkeypatch):
    monkeypatch.setattr(mg, "DATA_DIR", tmp_path)
    return mg._MetaGate()


def test_no_model_is_passive(tmp_path, monkeypatch):
    g = _gate(tmp_path, monkeypatch)
    assert g.active is False
    assert g.kind is None
    assert g.probability(None, None) is None      # pasifken arg'a dokunmadan None
    assert g.explain(None, None) is None


def test_corrupt_json_stays_passive(tmp_path, monkeypatch):
    (tmp_path / "meta_model.json").write_text("{bozuk json", encoding="utf-8")
    g = _gate(tmp_path, monkeypatch)
    assert g.active is False and g.kind is None   # çökme yok, fail-safe pasif


def test_valid_logistic_model_loads_active(tmp_path, monkeypatch):
    n = len(FEATURE_NAMES)
    model = {
        "w": [0.0] + [0.1] * n,          # bias + ağırlıklar
        "mu": [0.0] * n,
        "sd": [1.0] * n,
        "features": list(FEATURE_NAMES),
        "gate_threshold": 0.62,
    }
    (tmp_path / "meta_model.json").write_text(json.dumps(model), encoding="utf-8")
    g = _gate(tmp_path, monkeypatch)
    assert g.active is True
    assert g.kind == "logistic"
    assert g.threshold == 0.62
    assert isinstance(g._log[0], np.ndarray)


def test_default_threshold_when_absent(tmp_path, monkeypatch):
    n = len(FEATURE_NAMES)
    model = {"w": [0.0] + [0.1] * n, "mu": [0.0] * n, "sd": [1.0] * n}
    (tmp_path / "meta_model.json").write_text(json.dumps(model), encoding="utf-8")
    g = _gate(tmp_path, monkeypatch)
    assert g.threshold == 0.5


def test_get_gate_caches_and_reload_resets(tmp_path, monkeypatch):
    monkeypatch.setattr(mg, "DATA_DIR", tmp_path)
    mg.reload_gate()
    first = mg.get_gate()
    assert mg.get_gate() is first      # aynı örnek (cache)
    mg.reload_gate()
    assert mg.get_gate() is not first  # reload yeni örnek kurar
    mg.reload_gate()                   # test izolasyonu: global'i temizle


# ---------------------------------------------------------------------------
# B3 — save_model model-seçimi: bayat .pkl yeni lojistiği gölgelememeli
# ---------------------------------------------------------------------------
def _fit_logistic():
    from research.train_meta import LogisticModel
    n = len(FEATURE_NAMES)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, n))
    y = (X[:, 0] > 0).astype(float)
    return LogisticModel.fit(X, y)


def test_logistic_save_archives_stale_pkl_and_gate_selects_logistic(tmp_path, monkeypatch):
    """B3: eski GBM .pkl dururken lojistik kaydedilince .pkl ARŞİVLENMELİ. _load .pkl'i
    .json'dan ÖNCE okuduğundan, arşivlenmezse yeni lojistik eski GBM'le sessizce gölgelenir.
    Arşivlenince kapı lojistiği seçer."""
    import pickle

    import research.train_meta as tm

    # Bayat GBM .pkl (clf yükte çağrılmaz → placeholder yeterli).
    with (tmp_path / "meta_model.pkl").open("wb") as f:
        pickle.dump({"clf": [1, 2, 3], "features": list(FEATURE_NAMES),
                     "gate_threshold": 0.9, "kind": "gbm"}, f)

    monkeypatch.setattr(tm, "DATA_DIR", tmp_path)
    path = tm.save_model(_fit_logistic(), trained_on="test", oos_auc=0.55, gate_threshold=0.62)
    assert path == tmp_path / "meta_model.json"
    assert not (tmp_path / "meta_model.pkl").exists()          # bayat pkl arşivlendi
    assert list(tmp_path.glob("meta_model.pkl.bak-*"))         # arşiv dosyası var
    assert (tmp_path / "meta_model.json").exists()

    monkeypatch.setattr(mg, "DATA_DIR", tmp_path)
    g = mg._MetaGate()
    assert g.kind == "logistic"                                # gölgeleme yok
    assert g.threshold == 0.62


def test_gbm_save_archives_stale_json(tmp_path, monkeypatch):
    """B3 simetrik: gbm/xgb kaydında bayat .json arşivlenir (denetim/geri-dönüş netliği)."""
    import json

    import research.train_meta as tm
    from research.train_meta import GBMModel

    (tmp_path / "meta_model.json").write_text(json.dumps({"kind": "logistic"}), encoding="utf-8")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, len(FEATURE_NAMES)))
    y = (X[:, 0] > 0).astype(float)
    monkeypatch.setattr(tm, "DATA_DIR", tmp_path)
    path = tm.save_model(GBMModel.fit(X, y), trained_on="t", oos_auc=0.5, gate_threshold=0.5)
    assert path == tmp_path / "meta_model.pkl"
    assert not (tmp_path / "meta_model.json").exists()
    assert list(tmp_path.glob("meta_model.json.bak-*"))
