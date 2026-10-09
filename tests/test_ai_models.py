"""nihai.ai.models testleri — 4 katmanlı kapasite merdiveni (fit/predict/param_count/save/load).

Ağ yok; save/load YALNIZ tmp_path'e yazar (üretim data/'sına asla dokunulmaz — models.py'da
zaten DATA_DIR referansı YOK, testler ayrıca path_base'i tmp_path altında verir).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from nihai.ai import models as md


@pytest.fixture()
def ayrilabilir_veri():
    """AUC>0.9 garanti edecek kadar ayrılabilir küçük sentetik veri (hızlı fit için küçük)."""
    X, y = make_classification(
        n_samples=400, n_features=6, n_informative=4, n_redundant=0,
        n_clusters_per_class=1, class_sep=2.0, random_state=42)
    return X, y


# ---------------------------------------------------------------------------
# Temel sözleşme / hata yolları
# ---------------------------------------------------------------------------
def test_param_tiers_sabiti():
    assert md.PARAM_TIERS == ["logistic", "elasticnet", "gbm_small", "mlp"]


def test_gecersiz_tier_value_error():
    with pytest.raises(ValueError):
        md.CapacityModel(tier="bilinmeyen", feature_names=["f0"])


def test_fit_oncesi_predict_proba_runtime_error():
    m = md.CapacityModel(tier="logistic", feature_names=["f0", "f1"])
    with pytest.raises(RuntimeError):
        m.predict_proba([[0.0, 0.0]])


def test_fit_oncesi_param_count_runtime_error():
    m = md.CapacityModel(tier="logistic", feature_names=["f0"])
    with pytest.raises(RuntimeError):
        m.param_count()


def test_fit_oncesi_meta_param_count_none():
    m = md.CapacityModel(tier="logistic", feature_names=["f0"])
    meta = m.meta()
    assert meta["param_count"] is None
    assert meta["trained_at"] is None
    assert meta["tier"] == "logistic"


def test_fit_uyumsuz_boyut_value_error():
    m = md.CapacityModel(tier="logistic", feature_names=["f0", "f1"])
    with pytest.raises(ValueError):
        m.fit([[1, 2]], [1])   # tek satır (<2)


def test_fit_oncesi_save_runtime_error(tmp_path):
    m = md.CapacityModel(tier="logistic", feature_names=["f0"])
    with pytest.raises(RuntimeError):
        m.save(tmp_path / "model")


# ---------------------------------------------------------------------------
# 4 katman: fit + predict_proba + AUC>0.9 + param_count monotonikliği
# ---------------------------------------------------------------------------
def test_dort_katman_fit_predict_auc(ayrilabilir_veri):
    X, y = ayrilabilir_veri
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    param_counts: dict[str, int] = {}
    for tier in md.PARAM_TIERS:
        m = md.CapacityModel(tier=tier, feature_names=feature_names, seed=42)
        fitted = m.fit(X, y)
        assert fitted is m   # zincirlenebilir (spec: fit -> CapacityModel)
        proba = m.predict_proba(X)
        assert proba.shape == (X.shape[0],)
        assert np.all((proba >= 0.0) & (proba <= 1.0))
        auc = roc_auc_score(y, proba)
        assert auc > 0.9, f"{tier} AUC={auc} beklenenden düşük"
        pc = m.param_count()
        assert pc > 0
        param_counts[tier] = pc
    assert param_counts["logistic"] < param_counts["elasticnet"] < param_counts["mlp"]


def test_param_count_logistic_formul():
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_redundant=0,
                                n_clusters_per_class=1, class_sep=2.0, random_state=1)
    m = md.CapacityModel(tier="logistic", feature_names=[f"f{i}" for i in range(5)], seed=1)
    m.fit(X, y)
    assert m.param_count() == 5 + 1   # f+1 (spec formülü)


def test_param_count_elasticnet_formul():
    X, y = make_classification(n_samples=300, n_features=5, n_informative=3, n_redundant=0,
                                n_clusters_per_class=1, class_sep=2.0, random_state=1)
    m = md.CapacityModel(tier="elasticnet", feature_names=[f"f{i}" for i in range(5)], seed=1)
    m.fit(X, y)
    import math
    assert m.param_count() == math.comb(5, 2) + 5 + 1   # C(f,2)+f+1 (spec formülü)


def test_determinizm_ayni_seed_ayni_param_count(ayrilabilir_veri):
    X, y = ayrilabilir_veri
    fn = [f"f{i}" for i in range(X.shape[1])]
    m1 = md.CapacityModel(tier="gbm_small", feature_names=fn, seed=7).fit(X, y)
    m2 = md.CapacityModel(tier="gbm_small", feature_names=fn, seed=7).fit(X, y)
    assert m1.param_count() == m2.param_count()
    assert np.array_equal(m1.predict_proba(X), m2.predict_proba(X))


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------
def test_save_load_gidis_donus(tmp_path, ayrilabilir_veri):
    X, y = ayrilabilir_veri
    fn = [f"f{i}" for i in range(X.shape[1])]
    m = md.CapacityModel(tier="logistic", feature_names=fn, seed=42).fit(X, y)
    base = tmp_path / "modelim"
    m.save(base)

    assert (tmp_path / "modelim.json").exists()
    assert (tmp_path / "modelim.pkl").exists()

    loaded = md.CapacityModel.load(base)
    assert loaded is not None
    assert loaded.tier == m.tier
    assert loaded.feature_names == m.feature_names
    assert loaded.seed == m.seed
    assert loaded.n_features_ == m.n_features_
    assert loaded.param_count() == m.param_count()
    assert np.allclose(loaded.predict_proba(X), m.predict_proba(X))


def test_save_yalniz_verilen_path_base_yazar(tmp_path, ayrilabilir_veri):
    """save() üretim data/ dizinine YAZMAMALI — yalnız verilen path_base'e (+.json/.pkl)."""
    X, y = ayrilabilir_veri
    fn = [f"f{i}" for i in range(X.shape[1])]
    m = md.CapacityModel(tier="logistic", feature_names=fn, seed=42).fit(X, y)
    alt_dir = tmp_path / "alt_klasor"
    base = alt_dir / "meta_model"
    m.save(base)
    yazilanlar = sorted(p.name for p in alt_dir.iterdir())
    assert yazilanlar == ["meta_model.json", "meta_model.pkl"]


def test_save_meta_json_alanlari(tmp_path, ayrilabilir_veri):
    X, y = ayrilabilir_veri
    fn = [f"f{i}" for i in range(X.shape[1])]
    m = md.CapacityModel(tier="mlp", feature_names=fn, seed=3).fit(X, y)
    base = tmp_path / "m"
    m.save(base)
    payload = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    for key in ("tier", "n_features", "param_count", "seed", "trained_at", "feature_names"):
        assert key in payload
    assert payload["tier"] == "mlp"
    assert payload["feature_names"] == fn


def test_load_eksik_dosya_none(tmp_path):
    assert md.CapacityModel.load(tmp_path / "yok") is None


def test_load_bozuk_json_none(tmp_path):
    base = tmp_path / "bozuk"
    (tmp_path / "bozuk.json").write_text("{not valid json", encoding="utf-8")
    (tmp_path / "bozuk.pkl").write_bytes(b"\x00\x01")
    assert md.CapacityModel.load(base) is None


def test_load_pkl_eksik_none(tmp_path, ayrilabilir_veri):
    X, y = ayrilabilir_veri
    fn = [f"f{i}" for i in range(X.shape[1])]
    m = md.CapacityModel(tier="logistic", feature_names=fn, seed=1).fit(X, y)
    base = tmp_path / "yarim"
    m.save(base)
    (tmp_path / "yarim.pkl").unlink()
    assert md.CapacityModel.load(base) is None


# ---------------------------------------------------------------------------
# recommend_tier / param_ladder_report
# ---------------------------------------------------------------------------
def test_recommend_tier_sinir_degerleri():
    f = 25
    assert md.recommend_tier(50 * f - 1, f) == "logistic"
    assert md.recommend_tier(50 * f, f) == "elasticnet"
    assert md.recommend_tier(400 * f - 1, f) == "elasticnet"
    assert md.recommend_tier(400 * f, f) == "gbm_small"
    assert md.recommend_tier(2000 * f - 1, f) == "gbm_small"
    assert md.recommend_tier(2000 * f, f) == "mlp"


def test_recommend_tier_gecersiz_n_features():
    assert md.recommend_tier(1000, 0) == "logistic"
    assert md.recommend_tier(1000, -5) == "logistic"


def test_param_ladder_report_yapisi_ve_degerleri():
    import math
    report = md.param_ladder_report(n_features=25)
    tiers = [r["tier"] for r in report]
    assert tiers == md.PARAM_TIERS
    by_tier = {r["tier"]: r["param_count"] for r in report}
    assert by_tier["logistic"] == 26                       # 25+1
    assert by_tier["elasticnet"] == math.comb(25, 2) + 25 + 1   # 326
    assert by_tier["mlp"] == 64 * 25 + 64 + 64 * 32 + 32 + 32 + 1   # 3777
    # gbm_small kapalı-form yok; None ya da pozitif int olabilir (veri-bağımlı temsili fit)
    assert by_tier["gbm_small"] is None or by_tier["gbm_small"] > 0


def test_param_ladder_report_varsayilan_calisir():
    report = md.param_ladder_report()
    assert len(report) == 4
    assert all("tier" in r and "param_count" in r for r in report)


# ---------------------------------------------------------------------------
# v3 (Tur-16): neural katmanlar — fail-soft + facade + merdiven/öneri genişlemesi
# ---------------------------------------------------------------------------
from nihai.ai import neural as nrl


def test_neural_tiers_literal_kopya_eslesir():
    """models.NEURAL_TIERS neural.py'nin LİTERAL kopyasıdır (import maliyeti gerekçesi
    models.py docstring'inde) — tek-kaynak sözleşmesi bu testle garanti."""
    assert md.NEURAL_TIERS == nrl.NEURAL_TIERS
    assert md.ALL_TIERS == md.PARAM_TIERS + md.NEURAL_TIERS


def test_recommend_tier_varsayilan_v2_bit_denk():
    """allow_neural verilmedikçe v2 davranışı — 75k örnek bile mlp'de durur."""
    assert md.recommend_tier(75_000, 25) == "mlp"
    assert md.recommend_tier(2000 * 25, 25) == "mlp"


def test_recommend_tier_torchsuz_mlp_de_durur(monkeypatch):
    monkeypatch.setattr(nrl, "TORCH_AVAILABLE", False)
    assert md.recommend_tier(100_000, 25, allow_neural=True) == "mlp"


def test_param_ladder_report_torchsuz_degrade_satiri(monkeypatch):
    monkeypatch.setattr(nrl, "TORCH_AVAILABLE", False)
    report = md.param_ladder_report(25, include_neural=True)
    assert [r["tier"] for r in report] == md.ALL_TIERS
    for r in report[4:]:
        assert r["param_count"] is None
        assert "torch" in r["note"]


def test_capacity_model_neural_torchsuz_runtime_error(monkeypatch):
    """torch yokken neural tier kurucusu RuntimeError (ortam eksikliği; ValueError =
    programcı hatası ayrımı korunur)."""
    monkeypatch.setattr(nrl, "TORCH_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        md.CapacityModel(tier="deep_mlp", feature_names=["f0"])


def test_theoretical_param_count_torchsuz_none(monkeypatch):
    monkeypatch.setattr(nrl, "TORCH_AVAILABLE", False)
    assert nrl.theoretical_param_count("deep_mlp", 25) is None


@pytest.mark.skipif(not nrl.TORCH_AVAILABLE, reason="torch yok")
def test_recommend_tier_allow_neural_esikleri():
    """Plan BÖLÜM D eşikleri: 75k → transformer; 8.000·f/25 → deep_mlp; canlı-log
    senaryosu (~12k örnek, v3 özellik seti ~60) DÜRÜSTÇE neural önermez."""
    assert md.recommend_tier(75_000, 25, allow_neural=True) == "quant_transformer"
    assert md.recommend_tier(19_200, 60, allow_neural=True) == "deep_mlp"
    assert md.recommend_tier(12_000, 60, allow_neural=True) == "elasticnet"


@pytest.mark.skipif(not nrl.TORCH_AVAILABLE, reason="torch yok")
def test_param_ladder_report_include_neural_gercek_sayim():
    report = md.param_ladder_report(70, include_neural=True)
    assert [r["tier"] for r in report] == md.ALL_TIERS
    by_tier = {r["tier"]: r["param_count"] for r in report}
    assert by_tier["quant_transformer"] >= 3_777_000   # ×1000 sözleşmesi (plan BÖLÜM D)
    assert by_tier["deep_mlp"] > 100_000


@pytest.mark.skipif(not nrl.TORCH_AVAILABLE, reason="torch yok")
def test_theoretical_param_count_sozlesme():
    assert nrl.theoretical_param_count("quant_transformer", 70) >= 3_777_000
    dm = nrl.theoretical_param_count("deep_mlp", 25)   # f=25 → ~187k; ~380k rakamı f≈60 içindir
    assert dm is not None and 150_000 < dm < 1_000_000
    with pytest.raises(ValueError):
        nrl.theoretical_param_count("mlp", 25)


@pytest.mark.skipif(not nrl.TORCH_AVAILABLE, reason="torch yok")
def test_capacity_model_neural_facade_gidis_donus(tmp_path):
    """deep_mlp FACADE sözleşmesi: fit → predict → param_count → meta → save(.json/.pt)
    → load → aynı tahmin (NeuralCapacity delegasyonu uçtan uca)."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] + 0.1 * rng.normal(size=60) > 0).astype(int)
    fn = [f"f{i}" for i in range(5)]
    m = md.CapacityModel(tier="deep_mlp", feature_names=fn, seed=11).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (60,)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert m.param_count() > 100_000
    assert m.meta()["tier"] == "deep_mlp"
    base = tmp_path / "npn"
    m.save(base)
    assert (tmp_path / "npn.json").exists()
    assert (tmp_path / "npn.pt").exists()
    loaded = md.CapacityModel.load(base)
    assert loaded is not None
    assert loaded.tier == "deep_mlp"
    assert loaded.param_count() == m.param_count()
    assert np.allclose(loaded.predict_proba(X), proba, atol=1e-6)
