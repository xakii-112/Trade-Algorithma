"""nihai.ai.neural testleri — sinir ağı çekirdeği (WP-N, Tur-16).

Ağır testler küçük tutulur (mini n, max_epochs<=2). Param-sözleşmesi testi fit GEREKTİRMEZ:
QuantTransformer/DeepMLP DOĞRUDAN kurulur, numel sayılır (models.py testlerindeki
gbm_small'daki gibi "gerçek mimariden ölç" deseni). Torch'lu testler
`@pytest.mark.skipif(not neural.TORCH_AVAILABLE, ...)` ile (importorskip DEĞİL — modül
fail-soft; SPEC §4). save/load YALNIZ tmp_path'e yazar.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nihai.ai import neural

_REPO_ROOT = Path(__file__).resolve().parents[1]

TORCH = pytest.mark.skipif(not neural.TORCH_AVAILABLE, reason="torch yok")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def feat_plain() -> list[str]:
    return [f"f{i}" for i in range(6)]


@pytest.fixture()
def feat_multiscale() -> list[str]:
    return ([f"h_{i}" for i in range(3)] + [f"d_{i}" for i in range(3)]
             + [f"w_{i}" for i in range(2)] + ["plain0"])


@pytest.fixture()
def ayrilabilir_veri(feat_plain):
    rng = np.random.default_rng(42)
    n = 40
    X = rng.normal(size=(n, len(feat_plain)))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture()
def aux_hedefler():
    rng = np.random.default_rng(1)
    n = 40
    return {
        "magnitude": rng.integers(0, neural.MAGNITUDE_BUCKETS, size=n),
        "vol": rng.random(n),
        "regime": rng.integers(0, neural.REGIME_CLASSES, size=n),
    }


# ---------------------------------------------------------------------------
# A. Sabitler
# ---------------------------------------------------------------------------
def test_neural_tiers_sabiti():
    # v4 (Tur-22): merdivene ≈30M parametreli quant_transformer_xl eklendi. Sıra ANLAMLIDIR
    # (küçükten büyüğe) — models.recommend_tier eşikleri bu sırayı varsayar.
    # v5 (Tur-32): ≈93M parametreli SEYREK (MoE) basamak — sıra kuralı korunur.
    # v6 (Tur-34): ≈284M parametreli İNCE-TANELİ + PAYLAŞIMLI MoE basamağı.
    # v7 (Tur-35): ≈772M parametreli ULTRA basamağı (40 uzman + dağılımsal `vol_q` başlığı).
    # v8 (Tur-36): ≈2,01 MİLYAR parametreli MEGA basamağı (80 uzman + kanıt-temelli yön).
    assert neural.NEURAL_TIERS == ["deep_mlp", "quant_transformer", "quant_transformer_xl",
                                   "quant_transformer_moe", "quant_transformer_moe_xl",
                                   "quant_transformer_ultra", "quant_transformer_mega"]
    # `deep_mlp` dışındakiler token tabanlıdır (maskeli ön-eğitim + attention rollout orada).
    assert neural.TOKENIZED_TIERS == neural.NEURAL_TIERS[1:]


def test_torch_available_bool_tipinde():
    assert isinstance(neural.TORCH_AVAILABLE, bool)


def test_magnitude_regime_sabitleri():
    assert neural.MAGNITUDE_BUCKETS == 5
    assert neural.REGIME_CLASSES == 6


def test_feature_scale_id_onekler():
    assert neural._feature_scale_id("h_ret5") == 1
    assert neural._feature_scale_id("d_rsi") == 2
    assert neural._feature_scale_id("w_trend") == 3
    assert neural._feature_scale_id("gap_pct") == 0    # öneksiz → ölçeksiz


# ---------------------------------------------------------------------------
# B. Kurucu / hata yolları (CapacityModel paritesi)
# ---------------------------------------------------------------------------
@TORCH
def test_gecersiz_tier_value_error(feat_plain):
    with pytest.raises(ValueError):
        neural.NeuralCapacity(tier="bilinmeyen", feature_names=feat_plain)


@TORCH
def test_fit_oncesi_predict_proba_runtime_error(feat_plain):
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain)
    with pytest.raises(RuntimeError):
        m.predict_proba(np.zeros((1, len(feat_plain))))


@TORCH
def test_fit_oncesi_predict_with_uncertainty_runtime_error(feat_plain):
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain)
    with pytest.raises(RuntimeError):
        m.predict_with_uncertainty(np.zeros((1, len(feat_plain))))


@TORCH
def test_fit_oncesi_param_count_runtime_error(feat_plain):
    m = neural.NeuralCapacity("quant_transformer", feature_names=feat_plain)
    with pytest.raises(RuntimeError):
        m.param_count()


@TORCH
def test_fit_oncesi_save_runtime_error(tmp_path, feat_plain):
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain)
    with pytest.raises(RuntimeError):
        m.save(tmp_path / "model")


@TORCH
def test_fit_oncesi_meta_param_count_none(feat_plain):
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain)
    meta = m.meta()
    assert meta["param_count"] is None
    assert meta["trained_at"] is None
    assert meta["tier"] == "deep_mlp"
    assert meta["n_features"] is None


@TORCH
def test_is_fitted_once_true_after_fit(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1)
    assert m.is_fitted is False
    m.fit(X, y, max_epochs=2)
    assert m.is_fitted is True


# ---------------------------------------------------------------------------
# C. Fail-soft (torch yok senaryosu)
# ---------------------------------------------------------------------------
def test_failsoft_monkeypatch_kurucu_runtime_error(monkeypatch, feat_plain):
    monkeypatch.setattr(neural, "TORCH_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        neural.NeuralCapacity("deep_mlp", feature_names=feat_plain)


def test_failsoft_monkeypatch_load_none(monkeypatch, tmp_path):
    monkeypatch.setattr(neural, "TORCH_AVAILABLE", False)
    assert neural.NeuralCapacity.load(tmp_path / "yok") is None


def test_failsoft_monkeypatch_build_model_runtime_error(monkeypatch, feat_plain):
    monkeypatch.setattr(neural, "TORCH_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        neural._build_torch_model("deep_mlp", feat_plain, seed=1)


def test_failsoft_gercek_import_torch_bloklu(tmp_path):
    """torch paketi GERÇEKTEN erişilemezken (import-engelleyici meta-path) modül importu
    patlamaz: TORCH_AVAILABLE False olur, DeepMLP/QuantTransformer None kalır, kurucu
    RuntimeError fırlatır, load() None döner (anayasa madde 8 — importorskip DEĞİL,
    gerçek fail-soft davranışı ayrı bir süreçte doğrulanır)."""
    script = """
import sys


class _Blocker:
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(name + " bloklandi (test)")
        return None


sys.meta_path.insert(0, _Blocker())

from nihai.ai import neural

assert neural.TORCH_AVAILABLE is False
assert neural.DeepMLP is None
assert neural.QuantTransformer is None
try:
    neural.NeuralCapacity("deep_mlp", feature_names=["f0"])
    raise SystemExit("FAIL: RuntimeError beklenirken firlatilmadi")
except RuntimeError:
    pass
assert neural.NeuralCapacity.load("hicbir_yer") is None
print("OK")
"""
    env = dict(os.environ)
    env["NIHAI_DATA_DIR"] = str(tmp_path / "data")
    env["NIHAI_LOG_DIR"] = str(tmp_path / "logs")
    env["NIHAI_DISABLE_FILE_LOG"] = "1"
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(_REPO_ROOT),
                          capture_output=True, text=True, timeout=60, env=env)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# D. fit girdi doğrulama
# ---------------------------------------------------------------------------
@TORCH
def test_fit_uyumsuz_satir_sayisi_value_error(feat_plain):
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain)
    with pytest.raises(ValueError):
        m.fit(np.zeros((5, len(feat_plain))), np.zeros(3))


@TORCH
def test_fit_tek_satir_value_error(feat_plain):
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain)
    with pytest.raises(ValueError):
        m.fit(np.zeros((1, len(feat_plain))), np.zeros(1))


@TORCH
def test_fit_yanlis_sutun_sayisi_value_error(feat_plain):
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain)
    with pytest.raises(ValueError):
        m.fit(np.zeros((10, len(feat_plain) + 1)), np.zeros(10))


@TORCH
def test_fit_aux_yanlis_uzunluk_value_error(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain)
    with pytest.raises(ValueError):
        m.fit(X, y, aux={"vol": np.zeros(3)}, max_epochs=1)


@TORCH
def test_predict_proba_yanlis_sekil_value_error(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    with pytest.raises(ValueError):
        m.predict_proba(np.zeros((3, len(feat_plain) + 2)))


# ---------------------------------------------------------------------------
# E. fit + predict temel davranış
# ---------------------------------------------------------------------------
@TORCH
def test_deep_mlp_fit_predict_proba_sekil_ve_aralik(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1)
    fitted = m.fit(X, y, max_epochs=2, batch_size=16)
    assert fitted is m   # zincirlenebilir (spec: fit -> NeuralCapacity)
    proba = m.predict_proba(X)
    assert proba.shape == (X.shape[0],)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


@TORCH
def test_quant_transformer_fit_predict_proba_aux_ile(ayrilabilir_veri, feat_plain, aux_hedefler):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("quant_transformer", feature_names=feat_plain, seed=3)
    m.fit(X, y, aux=aux_hedefler, max_epochs=2, batch_size=16)
    proba = m.predict_proba(X)
    assert proba.shape == (X.shape[0],)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert sorted(m.meta()["aux_heads_trained"]) == ["magnitude", "regime", "vol"]


@TORCH
def test_aux_kismi_verilince_yalniz_verilenler_kaydedilir(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=2)
    m.fit(X, y, aux={"vol": np.random.default_rng(0).random(len(y))}, max_epochs=1)
    assert m.meta()["aux_heads_trained"] == ["vol"]


@TORCH
def test_fit_x_y_tek_basina_calisir_aux_yok(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=5)
    m.fit(X, y, max_epochs=1)   # aux verilmedi — spec: "fit(X,y) tek başına çalışmalı"
    assert m.meta()["aux_heads_trained"] == []


# ---------------------------------------------------------------------------
# F. Belirsizlik (MC-dropout)
# ---------------------------------------------------------------------------
@TORCH
def test_predict_with_uncertainty_sekiller(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    mean_p, std_p = m.predict_with_uncertainty(X, T=5)
    assert mean_p.shape == (X.shape[0],)
    assert std_p.shape == (X.shape[0],)


@TORCH
def test_predict_with_uncertainty_std_negatif_degil(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("quant_transformer", feature_names=feat_plain, seed=1).fit(
        X, y, max_epochs=1, batch_size=32)
    mean_p, std_p = m.predict_with_uncertainty(X, T=8)
    assert np.all(std_p >= 0.0)
    assert np.all((mean_p >= 0.0) & (mean_p <= 1.0))


@TORCH
def test_predict_with_uncertainty_varsayilan_t20(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    mean_p, _ = m.predict_with_uncertainty(X[:3])   # T varsayılan=20, yalnız şekil kontrolü
    assert mean_p.shape == (3,)


# ---------------------------------------------------------------------------
# G. Parametre sözleşmesi — KRİTİK, fit GEREKTİRMEZ (mimariyi kur, numel say)
# ---------------------------------------------------------------------------
@TORCH
def test_quant_transformer_param_count_esik_3_777_000(feat_multiscale):
    """TEST SÖZLEŞMESİ (spec §3.1/BÖLÜM I): quant_transformer >= 3.777.000 parametre.
    fit çağrılmaz — mimari doğrudan kurulup gerçek numel toplanır (ağır-test disiplini)."""
    net = neural.QuantTransformer(feat_multiscale)
    total = sum(p.numel() for p in net.parameters())
    assert total >= 3_777_000


@TORCH
def test_quant_transformer_param_count_hedef_civari():
    """Plan BÖLÜM D hedefi ≈5.5M (±%10 serbest — spec 3.1). Gerçek üretim özellik sayısına
    (~96) yakın bir listeyle ölçüm; yalnız BİLGİ amaçlı geniş bir bant (taban testi ayrı)."""
    feat = ([f"h_{i}" for i in range(30)] + [f"d_{i}" for i in range(20)]
            + [f"w_{i}" for i in range(10)] + [f"f{i}" for i in range(10)])
    net = neural.QuantTransformer(feat)
    total = sum(p.numel() for p in net.parameters())
    assert 3_777_000 <= total <= 8_000_000


@TORCH
def test_quant_transformer_param_count_n_features_bagimsiz_esik_ustunde():
    """Blok/füzyon/başlık parametreleri token sayısından bağımsızdır (self-attention'da
    token-başı ağırlık yok) — çok küçük bir özellik listesiyle bile eşik geçilir."""
    net_kucuk = neural.QuantTransformer(["tek_ozellik"])
    total_kucuk = sum(p.numel() for p in net_kucuk.parameters())
    assert total_kucuk >= 3_777_000


@TORCH
def test_deep_mlp_param_count_pozitif_ve_transformerdan_kucuk(feat_multiscale):
    mlp = neural.DeepMLP(feat_multiscale)
    qt = neural.QuantTransformer(feat_multiscale)
    total_mlp = sum(p.numel() for p in mlp.parameters())
    total_qt = sum(p.numel() for p in qt.parameters())
    assert total_mlp > 0
    assert total_mlp < total_qt


@TORCH
def test_param_count_fit_sonrasi_neuralcapacity_uzerinden(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("quant_transformer", feature_names=feat_plain, seed=1)
    m.fit(X, y, max_epochs=1, batch_size=32)
    pc = m.param_count()
    assert pc >= 3_777_000
    assert pc == sum(p.numel() for p in m._model.parameters())


@TORCH
def test_feature_names_bos_deger_hatasi():
    with pytest.raises(ValueError):
        neural.QuantTransformer([])
    with pytest.raises(ValueError):
        neural.DeepMLP([])


# ---------------------------------------------------------------------------
# H. Determinizm (anayasa madde 6)
# ---------------------------------------------------------------------------
@TORCH
def test_determinizm_ayni_seed_ayni_agirliklar(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m1 = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=7).fit(
        X, y, max_epochs=2, batch_size=16)
    m2 = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=7).fit(
        X, y, max_epochs=2, batch_size=16)
    sd1, sd2 = m1._model.state_dict(), m2._model.state_dict()
    assert set(sd1.keys()) == set(sd2.keys())
    for k in sd1:
        assert neural.torch.equal(sd1[k], sd2[k]), f"ağırlık farklı: {k}"


@TORCH
def test_determinizm_ayni_seed_ayni_predict_proba(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m1 = neural.NeuralCapacity("quant_transformer", feature_names=feat_plain, seed=11).fit(
        X, y, max_epochs=2, batch_size=16)
    m2 = neural.NeuralCapacity("quant_transformer", feature_names=feat_plain, seed=11).fit(
        X, y, max_epochs=2, batch_size=16)
    assert np.array_equal(m1.predict_proba(X), m2.predict_proba(X))
    assert m1.param_count() == m2.param_count()


@TORCH
def test_determinizm_farkli_seed_farkli_agirlik(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m1 = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    m2 = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=2).fit(X, y, max_epochs=1)
    p1 = list(m1._model.parameters())[0]
    p2 = list(m2._model.parameters())[0]
    assert not neural.torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# I. Erken durma
# ---------------------------------------------------------------------------
@TORCH
def test_erken_durma_devreye_girer(ayrilabilir_veri, feat_plain):
    """Aşırı yüksek lr → val kaybı iyileşmeyi bırakır → patience tükenince max_epochs'tan
    ÖNCE durur (ölçüldü: lr=5.0/patience=2 → tipik 3 epoch'ta durur; taban n_epochs_trained
    < max_epochs koşulu deterministik-yakın davranışla test edilir)."""
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=42)
    m.fit(X, y, max_epochs=30, patience=2, lr=5.0, batch_size=64)
    meta = m.meta()
    assert meta["has_val_split"] is True
    assert meta["n_epochs_trained"] < 30
    assert meta["early_stopped"] is True


@TORCH
def test_kucuk_n_val_bolunmez_erken_durma_pasif(feat_plain):
    # n=3, val_frac=0.15 varsayılanıyla round(3*0.15)=0 → n_val<1 → val bölünmez (bkz. fit
    # docstring'i: "n çok küçükse val bölünmez, erken-durma devre dışı kalır").
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, len(feat_plain)))
    y = np.array([0, 1, 0])
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1)
    m.fit(X, y, max_epochs=2)
    meta = m.meta()
    assert meta["has_val_split"] is False
    assert meta["early_stopped"] is False
    assert meta["n_epochs_trained"] == 2


# ---------------------------------------------------------------------------
# J. save / load
# ---------------------------------------------------------------------------
@TORCH
def test_save_load_gidis_donus_predict_proba_ayni(tmp_path, ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=42).fit(
        X, y, max_epochs=2, batch_size=16)
    base = tmp_path / "modelim"
    m.save(base)

    assert (tmp_path / "modelim.json").exists()
    assert (tmp_path / "modelim.pt").exists()

    loaded = neural.NeuralCapacity.load(base)
    assert loaded is not None
    assert loaded.tier == m.tier
    assert loaded.feature_names == m.feature_names
    assert loaded.seed == m.seed
    assert loaded.n_features_ == m.n_features_
    assert loaded.param_count() == m.param_count()
    assert np.allclose(loaded.predict_proba(X), m.predict_proba(X))


@TORCH
def test_save_yalniz_verilen_path_base_yazar(tmp_path, ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    alt_dir = tmp_path / "alt_klasor"
    base = alt_dir / "meta_model"
    m.save(base)
    yazilanlar = sorted(p.name for p in alt_dir.iterdir())
    assert yazilanlar == ["meta_model.json", "meta_model.pt"]


@TORCH
def test_save_meta_json_alanlari(tmp_path, ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=3).fit(X, y, max_epochs=1)
    base = tmp_path / "m"
    m.save(base)
    payload = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    for key in ("tier", "n_features", "param_count", "seed", "trained_at", "feature_names",
                "feat_mu", "feat_sigma"):
        assert key in payload
    assert payload["tier"] == "deep_mlp"
    assert payload["feature_names"] == feat_plain
    assert len(payload["feat_mu"]) == len(feat_plain)
    assert len(payload["feat_sigma"]) == len(feat_plain)


@TORCH
def test_load_eksik_dosya_none(tmp_path):
    assert neural.NeuralCapacity.load(tmp_path / "yok") is None


@TORCH
def test_load_bozuk_json_none(tmp_path):
    base = tmp_path / "bozuk"
    (tmp_path / "bozuk.json").write_text("{not valid json", encoding="utf-8")
    (tmp_path / "bozuk.pt").write_bytes(b"\x00\x01")
    assert neural.NeuralCapacity.load(base) is None


@TORCH
def test_load_pt_eksik_none(tmp_path, ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    base = tmp_path / "yarim"
    m.save(base)
    (tmp_path / "yarim.pt").unlink()
    assert neural.NeuralCapacity.load(base) is None


@TORCH
def test_load_feature_names_liste_degil_none(tmp_path):
    base = tmp_path / "kotu"
    (tmp_path / "kotu.json").write_text(
        json.dumps({"tier": "deep_mlp", "feature_names": "not-a-list"}), encoding="utf-8")
    (tmp_path / "kotu.pt").write_bytes(b"\x00")
    assert neural.NeuralCapacity.load(base) is None


@TORCH
def test_load_bilinmeyen_tier_none(tmp_path):
    base = tmp_path / "kotutier"
    (tmp_path / "kotutier.json").write_text(
        json.dumps({"tier": "bilinmeyen_tier", "feature_names": ["f0"]}), encoding="utf-8")
    (tmp_path / "kotutier.pt").write_bytes(b"\x00")
    assert neural.NeuralCapacity.load(base) is None


# ---------------------------------------------------------------------------
# K. Çoklu zaman-ölçeği (SCALE önek konvansiyonu)
# ---------------------------------------------------------------------------
@TORCH
def test_quant_transformer_karisik_onekli_feature_names_calisir(feat_multiscale):
    net = neural.QuantTransformer(feat_multiscale)
    x = neural.torch.zeros(2, len(feat_multiscale))
    out = net(x)
    assert out["direction"].shape == (2, 2)
    assert out["magnitude"].shape == (2, neural.MAGNITUDE_BUCKETS)
    assert out["vol"].shape == (2, 1)
    assert out["regime"].shape == (2, neural.REGIME_CLASSES)


@TORCH
def test_quant_transformer_oneksiz_tek_olcek_calisir():
    """Önek hiç yoksa tek ölçek (savunmacı) — spec §3.1: yine çalışır, çökme yok."""
    feat = [f"plain{i}" for i in range(10)]
    net = neural.QuantTransformer(feat)
    x = neural.torch.zeros(3, len(feat))
    out = net(x)
    assert out["direction"].shape == (3, 2)


@TORCH
def test_deep_mlp_forward_sekiller(feat_plain):
    net = neural.DeepMLP(feat_plain)
    x = neural.torch.zeros(4, len(feat_plain))
    out = net(x)
    assert out["direction"].shape == (4, 2)
    assert out["magnitude"].shape == (4, neural.MAGNITUDE_BUCKETS)
    assert out["vol"].shape == (4, 1)
    assert out["regime"].shape == (4, neural.REGIME_CLASSES)


# ---------------------------------------------------------------------------
# L. CapacityModel API parite
# ---------------------------------------------------------------------------
@TORCH
def test_meta_anahtarlari_capacitymodel_ile_birebir(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    meta = m.meta()
    for key in ("tier", "n_features", "param_count", "seed", "trained_at"):
        assert key in meta


@TORCH
def test_n_features_ve_trained_at_fit_sonrasi_dogru(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1)
    m.fit(X, y, max_epochs=1)
    assert m.n_features_ == len(feat_plain)
    assert m.trained_at is not None
    from datetime import datetime
    datetime.fromisoformat(m.trained_at)   # parse edilebilir olmalı (ValueError atmamalı)


@TORCH
def test_kurucu_imza_capacitymodel_paritesi(feat_plain):
    m = neural.NeuralCapacity(tier="deep_mlp", feature_names=feat_plain, seed=99)
    assert m.tier == "deep_mlp"
    assert m.feature_names == feat_plain
    assert m.seed == 99
    assert m.n_features_ is None
    assert m.trained_at is None


# ===========================================================================
# M. v4 (Tur-22) — ölçüm yardımcıları (saf numpy, torch GEREKTİRMEZ)
# ===========================================================================
def test_auc_np_mukemmel_ayrim():
    y = np.array([0, 0, 1, 1])
    assert neural._auc_np(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert neural._auc_np(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_auc_np_beraberlikte_ortalama_sira():
    """Tüm tahminler eşitse AUC tam 0.5 olmalı (beraberlik = ortalama sıra kuralı).
    Naif sıralama bunu 0.0 veya 1.0 gösterirdi — sessiz yanlış-pozitif kaynağı."""
    y = np.array([0, 1, 0, 1])
    assert neural._auc_np(y, np.full(4, 0.5)) == 0.5


def test_auc_np_tek_sinif_none():
    assert neural._auc_np(np.ones(5), np.linspace(0, 1, 5)) is None
    assert neural._auc_np(np.zeros(5), np.linspace(0, 1, 5)) is None


def test_brier_ve_ece_bilinen_degerler():
    y = np.array([1.0, 0.0])
    assert neural._brier_np(y, np.array([1.0, 0.0])) == 0.0
    assert neural._brier_np(y, np.array([0.0, 1.0])) == 1.0
    # Kusursuz kalibre: 0.5 diyen kovada gerçekleşen oran da 0.5.
    assert neural._ece_np(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == pytest.approx(0.0)
    # Tam ters: hep 1.0 diyor, hiç gerçekleşmiyor → ECE 1.0.
    assert neural._ece_np(np.zeros(4), np.ones(4)) == pytest.approx(1.0)


def test_evaluation_metrics_anahtarlari_ve_bos_girdi():
    res = neural.evaluation_metrics([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7])
    for key in ("n", "auc", "brier", "ece", "accuracy", "pos_rate", "pred_mean"):
        assert key in res
    assert res["n"] == 4
    assert res["accuracy"] == 1.0
    assert res["pos_rate"] == 0.5
    bos = neural.evaluation_metrics([], [])
    assert bos["n"] == 0 and bos["auc"] is None


# ===========================================================================
# N. v4 — quant_transformer_xl mimarisi (parametre sözleşmesi, fit GEREKTİRMEZ)
# ===========================================================================
@TORCH
def test_xl_param_count_esik_25m(feat_multiscale):
    """v4 TEST SÖZLEŞMESİ: quant_transformer_xl >= 25.000.000 parametre (v3'ün ≈5,5×'i)."""
    net = neural.QuantTransformerXL(feat_multiscale)
    assert sum(p.numel() for p in net.parameters()) >= 25_000_000


@TORCH
def test_xl_param_count_n_features_bagimsiz_esik_ustunde():
    """Blok yığını (12×) tek başına eşiği geçer — tek özellikli listeyle de sözleşme tutar."""
    net = neural.QuantTransformerXL(["tek_ozellik"])
    assert sum(p.numel() for p in net.parameters()) >= 25_000_000


@TORCH
def test_xl_v3_transformerdan_buyuk_ve_bantta(feat_multiscale):
    xl = sum(p.numel() for p in neural.QuantTransformerXL(feat_multiscale).parameters())
    v3 = sum(p.numel() for p in neural.QuantTransformer(feat_multiscale).parameters())
    assert xl > 4 * v3            # kapasite sıçraması gerçek (≈5,5×)
    assert 25_000_000 <= xl <= 45_000_000   # bilgi amaçlı geniş bant


@TORCH
def test_xl_forward_sekiller_ve_probe_basliklari(feat_multiscale):
    net = neural.QuantTransformerXL(feat_multiscale)
    out = net(neural.torch.zeros(2, len(feat_multiscale)))
    assert out["direction"].shape == (2, 2)
    assert out["magnitude"].shape == (2, neural.MAGNITUDE_BUCKETS)
    assert out["vol"].shape == (2, 1)
    assert out["regime"].shape == (2, neural.REGIME_CLASSES)
    assert out["probe1"].shape == (2, 2) and out["probe2"].shape == (2, 2)


@TORCH
def test_xl_dizi_duzeni_explain_sozlesmesi(feat_plain):
    """`explain.attention_rollout` [CLS] + 3×[SCALE] + n_features düzenini VARSAYAR —
    XL bu düzeni v3 ile aynı tutmalı (pos_embed uzunluğu bunun tek kaynağı)."""
    net = neural.QuantTransformerXL(feat_plain)
    assert net.pos_embed.shape == (1, len(feat_plain) + 4, neural._XL_D_MODEL)


@TORCH
def test_xl_dikkat_modulu_multiheadattention(feat_plain):
    """Açıklanabilirlik kilidi: rollout hook'ları `nn.MultiheadAttention` tipine bakar;
    özel bir dikkat implementasyonu bu yeteneği SESSİZCE öldürürdü."""
    net = neural.QuantTransformerXL(feat_plain)
    assert any(isinstance(m, neural.nn.MultiheadAttention) for m in net.modules())


@TORCH
def test_xl_bos_feature_names_value_error():
    with pytest.raises(ValueError):
        neural.QuantTransformerXL([])


@TORCH
def test_theoretical_param_count_xl_ve_bilinmeyen_tier():
    assert neural.theoretical_param_count("quant_transformer_xl", 8) >= 25_000_000
    with pytest.raises(ValueError):
        neural.theoretical_param_count("yok_boyle_tier", 8)


@TORCH
def test_xl_fit_predict_ve_determinizm(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m1 = neural.NeuralCapacity("quant_transformer_xl", feature_names=feat_plain, seed=5).fit(
        X, y, max_epochs=1, batch_size=32)
    p1 = m1.predict_proba(X)
    assert p1.shape == (X.shape[0],) and np.all((p1 >= 0) & (p1 <= 1))
    assert m1.param_count() >= 25_000_000
    m2 = neural.NeuralCapacity("quant_transformer_xl", feature_names=feat_plain, seed=5).fit(
        X, y, max_epochs=1, batch_size=32)
    assert np.array_equal(p1, m2.predict_proba(X))


# ===========================================================================
# O. v4 — yapı taşları
# ===========================================================================
@TORCH
def test_rmsnorm_olcek_normalize_eder():
    ln = neural._RMSNorm(4)
    out = ln(neural.torch.tensor([[3.0, 3.0, 3.0, 3.0]]))
    # γ=1 iken RMS(x)=3 → çıktı birim ölçekte (ortalama ÇIKARILMAZ; LayerNorm olsaydı 0 olurdu)
    assert neural.torch.allclose(out, neural.torch.ones(1, 4), atol=1e-4)


@TORCH
def test_drop_path_eval_kimlik_train_stokastik():
    dp = neural._DropPath(0.5)
    x = neural.torch.ones(64, 3)
    dp.eval()
    assert neural.torch.equal(dp(x), x)          # değerlendirmede KİMLİK
    dp.train()
    neural.torch.manual_seed(0)
    out = dp(x)
    assert (out == 0).any()                       # eğitimde bazı örnekler tamamen düşer
    assert out.shape == x.shape


@TORCH
def test_drop_path_sifir_olasilik_kimlik():
    dp = neural._DropPath(0.0)
    dp.train()
    x = neural.torch.ones(8, 3)
    assert neural.torch.equal(dp(x), x)


@TORCH
def test_layerscale_kimlik_yakini_baslar():
    ls = neural._LayerScale(8)
    assert float(ls.gamma.detach().abs().max()) == pytest.approx(neural._XL_LAYERSCALE_INIT)


@TORCH
def test_periodic_tokenizer_sekil_ve_olcek_gommesi(feat_multiscale):
    tok = neural._PeriodicTokenizer(len(feat_multiscale), 16,
                                     [neural._feature_scale_id(n) for n in feat_multiscale])
    out = tok(neural.torch.zeros(3, len(feat_multiscale)))
    assert out.shape == (3, len(feat_multiscale), 16)


@TORCH
def test_xl_stokastik_derinlik_dogrusal_artar(feat_plain):
    net = neural.QuantTransformerXL(feat_plain)
    ps = [b.dp1.p for b in net.blocks]
    assert ps[0] == 0.0
    assert ps[-1] == pytest.approx(neural._XL_DROP_PATH_MAX)
    assert all(ps[i] <= ps[i + 1] for i in range(len(ps) - 1))


# ===========================================================================
# P. v4 — eğitim reçetesi ekleri (opt-in; varsayılan v3 ile bit-denk)
# ===========================================================================
@TORCH
def test_varsayilan_recete_v3_ile_notr(ayrilabilir_veri, feat_plain):
    """Hiçbir v4 argümanı verilmezse: sıcaklık 1.0 (kalibrasyon YOK), scheduler yok,
    EMA yok, sınıf ağırlığı yok, label smoothing 0 → v3 davranışı."""
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    r = m.meta()["recipe"]
    assert m.temperature == 1.0
    assert r["scheduler"] == "none" and r["ema_decay"] is None
    assert r["class_weight"] is None and r["label_smoothing"] == 0.0
    assert r["calibrated"] is False and r["early_stop_metric"] == "val_loss"


@TORCH
@pytest.mark.parametrize("kwargs", [
    {"scheduler": "yok"}, {"class_weight": "yok"}, {"early_stop_metric": "yok"}])
def test_gecersiz_recete_argumani_value_error(ayrilabilir_veri, feat_plain, kwargs):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1)
    with pytest.raises(ValueError):
        m.fit(X, y, max_epochs=1, **kwargs)


@TORCH
def test_cosine_scheduler_ciktiyi_degistirir(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    duz = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=3).fit(
        X, y, max_epochs=3, batch_size=16)
    cos = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=3).fit(
        X, y, max_epochs=3, batch_size=16, scheduler="cosine")
    assert cos.meta()["recipe"]["scheduler"] == "cosine"
    assert not np.array_equal(duz.predict_proba(X), cos.predict_proba(X))


def test_cosine_lr_isinma_ve_azalma():
    """Isınmada LR artar, sonda tabana yaklaşır (saf fonksiyon — torch gerektirmez)."""
    lrs = [neural._cosine_lr(s, 100, 1.0, 0.1, 0.01) for s in range(100)]
    assert lrs[0] < lrs[9] <= 1.0            # ısınma dilimi artan
    assert lrs[-1] == pytest.approx(0.01, abs=1e-3)   # kosinüs tabanı
    assert max(lrs) == pytest.approx(1.0, rel=1e-6)


@TORCH
def test_ema_agirliklari_degistirir(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    ham = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=4).fit(
        X, y, max_epochs=2, batch_size=16)
    ema = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=4).fit(
        X, y, max_epochs=2, batch_size=16, ema_decay=0.9)
    assert ema.meta()["recipe"]["ema_decay"] == 0.9
    assert not np.array_equal(ham.predict_proba(X), ema.predict_proba(X))


@TORCH
def test_class_weight_balanced_agirliklari_egitimden_hesaplar(feat_plain):
    """Dengesiz y (≈%25 pozitif) → azınlık sınıfının ağırlığı BÜYÜK olmalı. Ağırlıklar
    YALNIZ eğitim diliminden hesaplanır (val'e bakmak sızıntı olurdu)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, len(feat_plain)))
    y = np.array([1 if i % 4 == 0 else 0 for i in range(40)], dtype=float)
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(
        X, y, max_epochs=1, class_weight="balanced")
    w = m.meta()["recipe"]["class_weights"]
    assert w is not None and len(w) == 2
    assert w[1] > w[0]


@TORCH
def test_class_weight_tek_sinifli_egitimde_sessizce_pasif(feat_plain):
    """Eğitim dilimi tek sınıflıysa 1/(2·0) tanımsızdır → ağırlık UYGULANMAZ (çökme yok)."""
    X = np.random.default_rng(1).normal(size=(20, len(feat_plain)))
    y = np.zeros(20)
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(
        X, y, max_epochs=1, class_weight="balanced")
    assert m.meta()["recipe"]["class_weights"] is None


@TORCH
def test_label_smoothing_ciktiyi_yumusatir(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=6).fit(
        X, y, max_epochs=2, label_smoothing=0.1)
    assert m.meta()["recipe"]["label_smoothing"] == 0.1


@TORCH
def test_early_stop_metric_val_auc(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=7).fit(
        X, y, max_epochs=3, early_stop_metric="val_auc")
    meta = m.meta()
    assert meta["recipe"]["early_stop_metric"] == "val_auc"
    assert meta["val_loss"] is None        # val_loss YALNIZ 'val_loss' ölçütünde anlamlı
    assert meta["best_score"] is not None


@TORCH
def test_kalibrasyon_sirayi_korur_olasiligi_degistirir(ayrilabilir_veri, feat_plain):
    """Sıcaklık ölçekleme MONOTONDUR: AUC (sıra tabanlı) DEĞİŞMEZ, olasılıklar değişir.
    Kalibrasyonun "kenar üretmediğinin", yalnız dürüstleştirdiğinin kanıtı."""
    X, y = ayrilabilir_veri
    ham = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=8).fit(
        X, y, max_epochs=2)
    kal = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=8).fit(
        X, y, max_epochs=2, calibrate=True)
    assert kal.meta()["recipe"]["calibrated"] is True
    assert kal.temperature > 0.0
    p_ham, p_kal = ham.predict_proba(X), kal.predict_proba(X)
    assert neural._auc_np(y, p_ham) == pytest.approx(neural._auc_np(y, p_kal))


@TORCH
def test_val_metrics_fit_sonrasi_dolu(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=9).fit(X, y, max_epochs=1)
    vm = m.meta()["val_metrics"]
    assert vm is not None and vm["n"] > 0
    assert vm["brier"] is not None and vm["ece"] is not None
    assert m.meta()["train_seconds"] >= 0.0
    assert 0.0 <= m.meta()["train_pos_rate"] <= 1.0


@TORCH
def test_val_bolunmezse_val_metrics_none(feat_plain):
    X = np.random.default_rng(0).normal(size=(3, len(feat_plain)))
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(
        X, np.array([0, 1, 0]), max_epochs=1)
    assert m.meta()["val_metrics"] is None


@TORCH
def test_device_auto_cpu_olmayan_ortamda_dusmez(ayrilabilir_veri, feat_plain):
    """CUDA yoksa 'auto' SESSİZCE cpu'ya düşer (ortam eksikliği hata değildir)."""
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(
        X, y, max_epochs=1, device="auto")
    assert m.meta()["recipe"]["device"] in ("cpu", "cuda")
    assert m.predict_proba(X).shape == (len(y),)


def test_resolve_device_bilinmeyen_ad_cpu():
    if not neural.TORCH_AVAILABLE:
        pytest.skip("torch yok")
    assert str(neural._resolve_device("hicboyleaygityok")) == "cpu"
    assert str(neural._resolve_device("cpu")) == "cpu"


# ===========================================================================
# R. v4 — OOD ölçüsü ve evaluate()
# ===========================================================================
@TORCH
def test_ood_zmax_sekil_ve_uc_deger(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    z = m.ood_zmax(X)
    assert z.shape == (X.shape[0],) and np.all(z >= 0)
    uzak = m.ood_zmax(X[:1] + 100.0)
    assert uzak[0] > z.max()      # dağılımdan uzaklaşan girdi daha büyük |z| verir


@TORCH
def test_ood_zmax_sabit_kolonu_dislar(feat_plain):
    """σ≈0 kolon (ör. `rr_ratio`) z hesabından DIŞLANIR — 1e-15'lik bölen sahte 10¹⁵
    mertebesinde z üretirdi (canlı köprüde yanlış susma nedeni)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, len(feat_plain)))
    X[:, 0] = 7.0        # tamamen sabit kolon
    y = (X[:, 1] > 0).astype(float)
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    probe = X[:1].copy()
    probe[0, 0] = 7.5    # YALNIZ sabit kolonu oynat
    assert m.ood_zmax(probe)[0] == pytest.approx(m.ood_zmax(X[:1])[0])


@TORCH
def test_train_zmax_q99_kaydedilir_ve_yuklenir(tmp_path, ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    assert m.train_zmax_q99 is not None and m.train_zmax_q99 > 0
    m.save(tmp_path / "z")
    loaded = neural.NeuralCapacity.load(tmp_path / "z")
    assert loaded is not None
    assert loaded.train_zmax_q99 == pytest.approx(round(m.train_zmax_q99, 4))


@TORCH
def test_ood_zmax_fit_oncesi_runtime_error(feat_plain):
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain)
    with pytest.raises(RuntimeError):
        m.ood_zmax(np.zeros((1, len(feat_plain))))


@TORCH
def test_evaluate_dogruluk_seti(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    res = m.evaluate(X, y)
    assert res["n"] == len(y)
    assert 0.0 <= res["brier"] <= 1.0


@TORCH
def test_kalibrasyon_save_load_gidis_donus(tmp_path, ayrilabilir_veri, feat_plain):
    """Sıcaklık artefaktla birlikte taşınmalı — aksi hâlde yeniden yüklenen model
    SESSİZCE kalibrasyonsuz tahmin ederdi."""
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=2).fit(
        X, y, max_epochs=2, calibrate=True)
    m.save(tmp_path / "kal")
    loaded = neural.NeuralCapacity.load(tmp_path / "kal")
    assert loaded is not None
    assert loaded.temperature == pytest.approx(round(m.temperature, 4), rel=1e-3)
    assert np.allclose(loaded.predict_proba(X), m.predict_proba(X), atol=1e-4)


@TORCH
def test_v3_artefakti_temperature_alanisiz_yuklenir(tmp_path, ayrilabilir_veri, feat_plain):
    """Geriye dönük uyum: `temperature`/`train_zmax_q99` alanı OLMAYAN eski JSON nötr
    varsayılanlarla okunur (kayıtlı canlı model artefaktları kırılmaz)."""
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    base = tmp_path / "eski"
    m.save(base)
    payload = json.loads((tmp_path / "eski.json").read_text(encoding="utf-8"))
    for k in ("temperature", "train_zmax_q99", "recipe", "val_metrics"):
        payload.pop(k, None)
    (tmp_path / "eski.json").write_text(json.dumps(payload), encoding="utf-8")
    loaded = neural.NeuralCapacity.load(base)
    assert loaded is not None
    assert loaded.temperature == 1.0
    assert loaded.train_zmax_q99 is None
    assert np.allclose(loaded.predict_proba(X), m.predict_proba(X))


# ===========================================================================
# S. v4 — NeuralEnsemble (derin topluluk)
# ===========================================================================
@TORCH
def test_ensemble_param_count_uye_toplami(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    ens = neural.NeuralEnsemble("deep_mlp", feature_names=feat_plain, seed=1, n_members=3).fit(
        X, y, max_epochs=1)
    tek = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    assert ens.param_count() == 3 * tek.param_count()
    assert ens.meta()["ensemble"] == 3
    assert ens.meta()["member_param_count"] == tek.param_count()


@TORCH
def test_ensemble_uyeler_farkli_seed_farkli_tahmin(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    ens = neural.NeuralEnsemble("deep_mlp", feature_names=feat_plain, seed=1, n_members=3).fit(
        X, y, max_epochs=1)
    seeds = [m.seed for m in ens.members]
    assert seeds == [1, 2, 3]
    spread = ens.member_spread(X)
    assert spread.shape == (len(y),) and float(spread.max()) > 0.0   # çeşitlilik GERÇEK


@TORCH
def test_ensemble_predict_proba_uye_ortalamasi(ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    ens = neural.NeuralEnsemble("deep_mlp", feature_names=feat_plain, seed=1, n_members=2).fit(
        X, y, max_epochs=1)
    beklenen = np.mean([m.predict_proba(X) for m in ens.members], axis=0)
    assert np.allclose(ens.predict_proba(X), beklenen)


@TORCH
def test_ensemble_belirsizlik_toplam_varyans_yasasi(ayrilabilir_veri, feat_plain):
    """Toplam std² = üye-içi MC varyansının ortalaması + üye ortalamalarının varyansı."""
    X, y = ayrilabilir_veri
    ens = neural.NeuralEnsemble("deep_mlp", feature_names=feat_plain, seed=1, n_members=2).fit(
        X, y, max_epochs=1)
    mean_p, std_p = ens.predict_with_uncertainty(X, T=4)
    ic = [m.predict_with_uncertainty(X, T=4) for m in ens.members]
    beklenen_var = (np.mean([sd ** 2 for _, sd in ic], axis=0)
                    + np.stack([mu for mu, _ in ic], axis=0).var(axis=0))
    assert np.allclose(std_p ** 2, beklenen_var, atol=1e-8)
    assert np.all((mean_p >= 0) & (mean_p <= 1))


@TORCH
def test_ensemble_save_load_gidis_donus(tmp_path, ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    ens = neural.NeuralEnsemble("deep_mlp", feature_names=feat_plain, seed=1, n_members=2).fit(
        X, y, max_epochs=1)
    base = tmp_path / "topluluk"
    ens.save(base)
    assert (tmp_path / "topluluk.json").exists()
    assert (tmp_path / "topluluk.m0.pt").exists() and (tmp_path / "topluluk.m1.pt").exists()
    loaded = neural.NeuralEnsemble.load(base)
    assert loaded is not None
    assert loaded.n_members == 2
    assert loaded.param_count() == ens.param_count()
    assert np.allclose(loaded.predict_proba(X), ens.predict_proba(X))


@TORCH
def test_ensemble_eksik_uye_none(tmp_path, ayrilabilir_veri, feat_plain):
    """Yarım topluluk sessizce BAŞKA bir modeldir — kabul edilmez (None)."""
    X, y = ayrilabilir_veri
    ens = neural.NeuralEnsemble("deep_mlp", feature_names=feat_plain, seed=1, n_members=2).fit(
        X, y, max_epochs=1)
    base = tmp_path / "yarim"
    ens.save(base)
    (tmp_path / "yarim.m1.pt").unlink()
    assert neural.NeuralEnsemble.load(base) is None


@TORCH
def test_ensemble_hata_yollari(feat_plain):
    with pytest.raises(ValueError):
        neural.NeuralEnsemble("bilinmeyen", feature_names=feat_plain)
    with pytest.raises(ValueError):
        neural.NeuralEnsemble("deep_mlp", feature_names=feat_plain, n_members=0)
    e = neural.NeuralEnsemble("deep_mlp", feature_names=feat_plain)
    assert e.is_fitted is False
    for call in (lambda: e.predict_proba(np.zeros((1, len(feat_plain)))),
                 lambda: e.param_count(),
                 lambda: e.member_spread(np.zeros((1, len(feat_plain)))),
                 lambda: e.ood_zmax(np.zeros((1, len(feat_plain))))):
        with pytest.raises(RuntimeError):
            call()
    assert e.meta()["param_count"] is None


@TORCH
def test_load_any_tip_ayrimi(tmp_path, ayrilabilir_veri, feat_plain):
    X, y = ayrilabilir_veri
    tek = neural.NeuralCapacity("deep_mlp", feature_names=feat_plain, seed=1).fit(X, y, max_epochs=1)
    tek.save(tmp_path / "tek")
    ens = neural.NeuralEnsemble("deep_mlp", feature_names=feat_plain, seed=1, n_members=2).fit(
        X, y, max_epochs=1)
    ens.save(tmp_path / "cok")
    assert isinstance(neural.load_any(tmp_path / "tek"), neural.NeuralCapacity)
    assert isinstance(neural.load_any(tmp_path / "cok"), neural.NeuralEnsemble)
    assert neural.load_any(tmp_path / "hic_yok") is None


def test_load_any_failsoft_torch_yok(monkeypatch, tmp_path):
    monkeypatch.setattr(neural, "TORCH_AVAILABLE", False)
    assert neural.load_any(tmp_path / "x") is None
    assert neural.NeuralEnsemble.load(tmp_path / "x") is None
    with pytest.raises(RuntimeError):
        neural.NeuralEnsemble("deep_mlp", feature_names=["f0"])


# ===========================================================================
# P. v5 (Tur-32) — SEYREK UZMAN KARIŞIMI (MoE) basamağı
# ===========================================================================
@TORCH
def test_moe_parametre_sozlesmesi_ve_xl_ustunluğu():
    """v5 SÖZLEŞMESİ: quant_transformer_moe >= 85M parametre ve XL'den KESİN büyük.
    Taban `n_features`'tan bağımsızdır (blok yığını tek başına ≈90M)."""
    p_moe = neural.theoretical_param_count("quant_transformer_moe", 8)
    p_xl = neural.theoretical_param_count("quant_transformer_xl", 8)
    assert p_moe >= 85_000_000
    assert p_moe > 2 * p_xl          # seyrek genişletmenin ANLAMI: ≈3× kapasite


@TORCH
def test_moe_forward_cikti_sozlugu_ve_denge_kaybi(feat_plain):
    """Çıktı XL'in 6 anahtarını + `moe_balance`'ı taşır. Denge kaybı [1, n_experts]
    aralığındadır: 1 = kusursuz dengeli yönlendirme, n_experts = tek uzmana çöküş."""
    model = neural._build_torch_model("quant_transformer_moe", feat_plain, seed=3)
    out = model(neural.torch.randn(16, len(feat_plain)))
    assert {"direction", "magnitude", "vol", "regime", "probe1", "probe2",
            "moe_balance"} == set(out.keys())
    bal = float(out["moe_balance"].detach())
    assert 1.0 - 1e-6 <= bal <= neural._MOE_EXPERTS + 1e-6


@TORCH
def test_moe_yonlendirme_deterministik(feat_plain):
    """Gürültülü kapı YOK → aynı girdi + eval modu = bit-denk çıktı (anayasa madde 6)."""
    model = neural._build_torch_model("quant_transformer_moe", feat_plain, seed=11).eval()
    x = neural.torch.randn(8, len(feat_plain))
    with neural.torch.no_grad():
        a, b = model(x)["direction"], model(x)["direction"]
    assert neural.torch.equal(a, b)


@TORCH
def test_moe_denge_kaybi_kayba_giriyor(feat_plain):
    """`_compute_loss` `moe_balance` anahtarını GÖRÜRSE kayba ekler; diğer tier'larda
    anahtar yoktur → v4 davranışı bit-denk kalır (regresyon kilidi)."""
    y = neural.torch.zeros(4, dtype=neural.torch.long)
    out_moe = {"direction": neural.torch.zeros(4, 2),
               "moe_balance": neural.torch.tensor(4.0)}
    out_dense = {"direction": neural.torch.zeros(4, 2)}
    l_moe = float(neural._compute_loss(out_moe, y, {}))
    l_dense = float(neural._compute_loss(out_dense, y, {}))
    assert l_moe == pytest.approx(l_dense + neural._MOE_BALANCE_WEIGHT * 4.0)


@TORCH
def test_moe_fit_predict_ve_kayit_yukle(ayrilabilir_veri, feat_plain, tmp_path):
    """Kapasite basamağı UÇTAN UCA çalışmalı: fit → predict → save → load_any → aynı
    olasılıklar. (Merdivenin üst basamağı fit edilemiyorsa kapasite kâğıt üstündedir.)"""
    X, y = ayrilabilir_veri
    m = neural.NeuralCapacity("quant_transformer_moe", feature_names=feat_plain, seed=5)
    m.fit(X, y, max_epochs=1, batch_size=16)
    p = m.predict_proba(X)
    assert p.shape == (X.shape[0],) and np.all((p >= 0) & (p <= 1))
    assert m.param_count() >= 85_000_000
    m.save(tmp_path / "moe")
    loaded = neural.load_any(tmp_path / "moe")
    assert isinstance(loaded, neural.NeuralCapacity)
    assert np.allclose(loaded.predict_proba(X), p)


# ===========================================================================
# Q. v6 (Tur-34) — İNCE-TANELİ + PAYLAŞIMLI MoE basamağı
# ===========================================================================
@TORCH
def test_moexl_parametre_sozlesmesi():
    """v6 SÖZLEŞMESİ: quant_transformer_moe_xl >= 270M ve MoE'nin ≈3 katı. Taban
    `n_features`'tan bağımsızdır (blok yığını tek başına ≈278M)."""
    p_moexl = neural.theoretical_param_count("quant_transformer_moe_xl", 8)
    p_moe = neural.theoretical_param_count("quant_transformer_moe", 8)
    assert p_moexl >= 270_000_000
    assert p_moexl > 2.5 * p_moe


@TORCH
def test_moexl_merdiven_monoton():
    """Merdiven sırası ANLAMLIDIR: her basamak bir öncekinden KESİN büyük olmalı
    (models.recommend_tier eşikleri bu sırayı varsayar)."""
    counts = [neural.theoretical_param_count(t, 25) for t in neural.NEURAL_TIERS]
    assert all(a is not None and b is not None and a < b
               for a, b in zip(counts, counts[1:], strict=False))


@TORCH
def test_moexl_forward_cikti_sozlugu(feat_plain):
    """Çıktı MoE'nin 7 anahtarını + `moe_zloss`'u taşır. Denge kaybı [1, n_routed]
    aralığında, z-kaybı POZİTİF (logsumexp² her zaman >= 0)."""
    model = neural._build_torch_model("quant_transformer_moe_xl", feat_plain, seed=3)
    out = model(neural.torch.randn(8, len(feat_plain)))
    assert {"direction", "magnitude", "vol", "regime", "probe1", "probe2",
            "moe_balance", "moe_zloss"} == set(out.keys())
    bal = float(out["moe_balance"].detach())
    assert 1.0 - 1e-6 <= bal <= neural._MOEXL_ROUTED_EXPERTS + 1e-6
    assert float(out["moe_zloss"].detach()) >= 0.0


@TORCH
def test_moexl_zloss_kayba_giriyor():
    """`_compute_loss` `moe_zloss`'u YALNIZ varsa ekler → MoE/XL/v4 bit-denk kalır."""
    y = neural.torch.zeros(4, dtype=neural.torch.long)
    base = {"direction": neural.torch.zeros(4, 2)}
    with_z = {"direction": neural.torch.zeros(4, 2), "moe_zloss": neural.torch.tensor(9.0)}
    assert float(neural._compute_loss(with_z, y, {})) == pytest.approx(
        float(neural._compute_loss(base, y, {})) + neural._MOE_ZLOSS_WEIGHT * 9.0)


@TORCH
def test_moexl_paylasimli_uzman_her_zaman_calisir(feat_plain):
    """PAYLAŞIMLI uzman yönlendirmeden BAĞIMSIZDIR: ağırlıklarını sıfırlamak çıktıyı
    DEĞİŞTİRMELİ (aksi hâlde dal ölü demektir — 'eklendi' iddiası ölçülmemiş kalırdı)."""
    model = neural._build_torch_model("quant_transformer_moe_xl", feat_plain, seed=5).eval()
    x = neural.torch.randn(4, len(feat_plain))
    with neural.torch.no_grad():
        before = model(x)["direction"].clone()
        for blk in model.blocks:
            blk.ffn.shared.down.weight.zero_()
            blk.ffn.shared.down.bias.zero_()
        after = model(x)["direction"]
    assert not neural.torch.allclose(before, after)


@TORCH
def test_moexl_yonlendirme_deterministik(feat_plain):
    """Gürültülü kapı YOK → aynı girdi + eval modu = bit-denk çıktı (anayasa madde 6)."""
    model = neural._build_torch_model("quant_transformer_moe_xl", feat_plain, seed=11).eval()
    x = neural.torch.randn(4, len(feat_plain))
    with neural.torch.no_grad():
        a, b = model(x)["direction"], model(x)["direction"]
    assert neural.torch.equal(a, b)


# ---------------------------------------------------------------------------
# Q2. Yönlendirme teşhisi — moe_routing_stats
# ---------------------------------------------------------------------------
@TORCH
def test_routing_stats_moe_olmayan_tier_none(feat_plain):
    """MoE olmayan basamakta dürüstçe None — 'uzmanlaşma yok' DEĞİL, 'ölçülecek
    yönlendirici yok' (ölçüm yokluğu ölçüm gibi sunulmaz)."""
    model = neural._build_torch_model("quant_transformer_xl", feat_plain, seed=2)
    assert neural.moe_routing_stats(model, np.zeros((4, len(feat_plain)))) is None


@TORCH
def test_routing_stats_yapisi_ve_paylar(feat_plain):
    """Blok başına kullanım payları toplamı 1, entropi [0,1], uzman sayısı doğru."""
    model = neural._build_torch_model("quant_transformer_moe", feat_plain, seed=4)
    rng = np.random.default_rng(0)
    st = neural.moe_routing_stats(model, rng.normal(size=(32, len(feat_plain))))
    assert st is not None
    assert st["n_blocks"] == neural._MOE_BLOCKS
    assert st["n_experts"] == neural._MOE_EXPERTS
    assert st["n_samples"] == 32 and st["specialization"] is None
    for blk in st["blocks"]:
        assert sum(blk["usage"]) == pytest.approx(1.0)
        assert 0.0 <= blk["entropy_norm"] <= 1.0
        assert blk["collapsed"] is (blk["max_share"] > 0.5)


@TORCH
def test_routing_stats_bayragi_geri_aliniyor(feat_plain):
    """Teşhis bayrağı ölçüm SONUNDA kapatılmalı ve tampon boşaltılmalı — açık kalırsa
    her canlı çıkarım blok başına int64 tamponu biriktirirdi."""
    model = neural._build_torch_model("quant_transformer_moe", feat_plain, seed=6)
    neural.moe_routing_stats(model, np.zeros((4, len(feat_plain))))
    ffns = neural._moe_ffn_modules(model)
    assert ffns and all((not f.record_routing) and f.last_top1 is None for f in ffns)


@TORCH
def test_routing_stats_uzmanlasma_testi(feat_plain):
    """`groups` verilince grup×uzman tablosu + ki-kare testi üretilir; ölçülemeyen
    blokta `measured=False` + gerekçe döner (uydurma p yok)."""
    model = neural._build_torch_model("quant_transformer_moe", feat_plain, seed=8)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(64, len(feat_plain)))
    g = np.where(x[:, 0] > 0, "yuksek_vol", "dusuk_vol")
    st = neural.moe_routing_stats(model, x, groups=g)
    assert st is not None and st["groups"] == ["dusuk_vol", "yuksek_vol"]
    assert len(st["specialization"]) == st["n_blocks"]
    for s in st["specialization"]:
        assert sum(sum(r) for r in s["table"]) == 64
        if s["measured"]:
            assert s["test"]["cramer_v"] >= 0.0 and s["reason"] is None
        else:
            assert s["test"] is None and s["reason"] in {"CLS_TEK_UZMAN", "TEK_GRUP_VEYA_BOS"}


@TORCH
def test_routing_stats_uyumsuz_grup_uzunlugu_none(feat_plain):
    model = neural._build_torch_model("quant_transformer_moe", feat_plain, seed=9)
    assert neural.moe_routing_stats(model, np.zeros((4, len(feat_plain))),
                                    groups=np.array(["a", "b"])) is None


def test_chi2_bagimsizlik_guvenilirlik_kapisi():
    """Beklenen sayımların >%20'si 5'in altındaysa p DÖNDÜRÜLMEZ (ki-kare yaklaşımı
    geçersiz) — anlamlılık uydurulmaz."""
    seyrek = np.array([[1, 1], [1, 1]], dtype=np.int64)
    res = neural._chi2_independence(seyrek)
    assert res is not None and res["reliable"] is False and res["p_value"] is None
    yogun = np.array([[50, 10], [10, 50]], dtype=np.int64)
    res2 = neural._chi2_independence(yogun)
    assert res2["reliable"] is True and res2["p_value"] < 0.001
    assert 0.0 <= res2["cramer_v"] <= 1.0


def test_chi2_dejenere_tablo_none():
    """Tek sütun/satır → karşılaştırılacak ikinci hücre yok → None (0 değil)."""
    assert neural._chi2_independence(np.array([[5, 0], [7, 0]], dtype=np.int64)) is None
    assert neural._chi2_independence(np.array([[5, 3]], dtype=np.int64)) is None
