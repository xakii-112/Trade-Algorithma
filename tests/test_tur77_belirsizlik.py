"""Tur-77 · Madde 5.1 (aleatorik/epistemik ayrışma) + 5.2 (Dirichlet inanç kütleleri).

Üç soru:
  1. **Matematik doğru mu?** Kapalı formlar: tüm örnekler aynıysa epistemik = 0; örnekler
     {0,1} yarı yarıya ise aleatorik = 0 ve karşılıklı bilgi = ln 2; varyans toplamı
     özdeşlik gereği p̄(1−p̄). İnançta Σb + u = 1.
  2. **Mevcut yol bozuldu mu?** `predict_with_uncertainty` ortak örneklem kaynağına taşındı →
     ayrışmanın `p`/`sqrt(epistemik_var)`'ı onunla BİREBİR aynı olmalı; toplulukta epistemik
     varyans = üye-içi + üye-arası (toplam varyans yasası, eşit T).
  3. **Uydurma yok mu?** Referans azsa eşik YOK; başlık yoksa inanç YOK.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nihai.ai import neural

TORCH = pytest.mark.skipif(not neural.TORCH_AVAILABLE, reason="torch yok")


# ---------------------------------------------------------------------------
# 1 — kapalı formlar (torch gerekmez)
# ---------------------------------------------------------------------------
def test_ozdes_ornekler_epistemik_sifir():
    d = neural.decompose_uncertainty(np.full((5, 3), 0.3))
    assert d is not None
    assert np.allclose(d["epistemik_var"], 0.0, atol=1e-15)
    assert np.allclose(d["karsilikli_bilgi"], 0.0, atol=1e-12)
    assert np.allclose(d["aleatorik_var"], 0.21, atol=1e-15)
    h = -(0.3 * math.log(0.3) + 0.7 * math.log(0.7))
    assert np.allclose(d["toplam_entropi"], h, atol=1e-12)
    assert np.allclose(d["aleatorik_entropi"], h, atol=1e-12)
    assert d["n_ornek"] == 5


def test_zit_kesin_ornekler_saf_epistemik():
    d = neural.decompose_uncertainty(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert d is not None
    assert np.allclose(d["p"], 0.5)
    assert np.allclose(d["aleatorik_var"], 0.0, atol=1e-15)
    assert np.allclose(d["epistemik_var"], 0.25, atol=1e-15)
    assert np.allclose(d["karsilikli_bilgi"], math.log(2.0), atol=1e-9)
    assert np.allclose(d["aleatorik_entropi"], 0.0, atol=1e-9)


def test_varyans_ozdesligi_ve_bilgi_negatif_olmaz():
    rng = np.random.default_rng(7)
    arr = rng.uniform(0.0, 1.0, size=(40, 200))
    d = neural.decompose_uncertainty(arr)
    assert d is not None
    p = d["p"]
    assert np.allclose(d["toplam_var"], p * (1.0 - p), atol=1e-15)
    assert np.all(d["karsilikli_bilgi"] >= 0.0)
    assert np.allclose(d["toplam_entropi"], d["aleatorik_entropi"] + d["karsilikli_bilgi"], atol=1e-12)


@pytest.mark.parametrize("girdi", [
    np.array([0.2, 0.3]),                    # 1B
    np.zeros((0, 3)),                        # boş
    np.array([[0.2, np.nan]]),               # NaN
    np.array([[0.2, 1.2]]),                  # [0,1] dışı
    np.array([[-0.1, 0.5]]),
    "abc",
])
def test_gecersiz_girdi_none(girdi):
    assert neural.decompose_uncertainty(girdi) is None


def test_inanc_kapali_formu():
    b = neural.evidential_belief(np.array([[1.0, 1.0], [3.0, 5.0]]))
    assert b is not None
    assert np.allclose(b["inanc"][0], [0.0, 0.0]) and b["bosluk"][0] == pytest.approx(1.0)
    assert np.allclose(b["inanc"][1], [2 / 8, 4 / 8]) and b["bosluk"][1] == pytest.approx(2 / 8)
    assert np.allclose(b["inanc"].sum(axis=1) + b["bosluk"], 1.0, atol=1e-15)
    assert np.allclose(b["p"][1], [3 / 8, 5 / 8])
    ucu = neural.evidential_belief(np.array([[2.0, 3.0, 5.0]]))
    assert ucu is not None and ucu["bosluk"][0] == pytest.approx(0.3)


@pytest.mark.parametrize("alpha", [np.array([[0.5, 2.0]]), np.array([[2.0]]),
                                   np.array([[np.inf, 2.0]]), np.array([1.0, 2.0])])
def test_inanc_gecersiz_none(alpha):
    assert neural.evidential_belief(alpha) is None


def test_bilinmeyen_bayragi_esigi_referanstan_turer_ve_uydurmaz():
    ref = np.arange(100, dtype=float)
    f = neural.epistemic_flag([5.0, 95.0, 89.0], ref, q=0.9)
    assert f is not None and f["esik"] == pytest.approx(np.quantile(ref, 0.9))
    assert f["bayrak"].tolist() == [False, True, False] and f["n_ref"] == 100
    assert neural.epistemic_flag([1.0], np.arange(29, dtype=float)) is None      # az referans
    assert neural.epistemic_flag([1.0], ref, q=1.0) is None
    assert neural.epistemic_flag([np.nan], ref) is None
    g = neural.epistemic_flag([5.0, 95.0, 89.0], ref, q=0.9)                    # determinizm
    assert g is not None and g["bayrak"].tolist() == f["bayrak"].tolist() and g["esik"] == f["esik"]


# ---------------------------------------------------------------------------
# 2 — model yolları: mevcut çıktı bit-denk + toplam varyans yasası
# ---------------------------------------------------------------------------
def _veri(n: int = 240, f: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    return X, y, [f"f{i}" for i in range(f)]


@TORCH
def test_tek_model_ayrisma_mevcut_belirsizlikle_birebir():
    X, y, names = _veri()
    m = neural.NeuralCapacity("deep_mlp", feature_names=names, seed=3).fit(X, y, max_epochs=2)
    mu, sd = m.predict_with_uncertainty(X[:50], T=8)
    d = m.predict_uncertainty_decomposition(X[:50], T=8)
    assert d is not None and d["n_ornek"] == 8
    # Aynı örneklem; fark yalnız float32 ortalamanın float64'te alınması (ölçüldü ~1e-8).
    assert np.allclose(d["p"], mu, atol=1e-6, rtol=0)
    assert np.allclose(np.sqrt(d["epistemik_var"]), sd, atol=1e-6, rtol=0)
    mu2, sd2 = m.predict_with_uncertainty(X[:50], T=8)                 # tohum → tekrarlanabilir
    assert np.array_equal(mu, mu2) and np.array_equal(sd, sd2)


@TORCH
def test_toplulukta_epistemik_uye_ici_arti_uye_arasi():
    X, y, names = _veri()
    ens = neural.NeuralEnsemble("deep_mlp", feature_names=names, seed=11, n_members=3)
    ens.fit(X, y, max_epochs=2)
    d = ens.predict_uncertainty_decomposition(X[:40], T=6)
    assert d is not None and d["n_ornek"] == 18
    assert np.allclose(d["epistemik_var"], d["uye_ici_mc_var"] + d["uye_arasi_var"], atol=1e-12)
    mu, sd = ens.predict_with_uncertainty(X[:40], T=6)
    assert np.allclose(d["p"], mu, atol=1e-6, rtol=0)
    assert np.allclose(np.sqrt(d["epistemik_var"]), sd, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# 3 — inanç: başlık yoksa None, varsa Σb + u = 1 ve predict_evidential ile tutarlı
# ---------------------------------------------------------------------------
@TORCH
def test_basliksiz_modelde_inanc_yok():
    X, y, names = _veri()
    m = neural.NeuralCapacity("deep_mlp", feature_names=names, seed=1).fit(X, y, max_epochs=1)
    assert m.predict_belief(X[:10]) is None
    ens = neural.NeuralEnsemble("deep_mlp", feature_names=names, seed=1, n_members=2)
    assert ens.fit(X, y, max_epochs=1).predict_belief(X[:10]) is None


@TORCH
def test_kanit_basliginda_inanc_tutarli():
    X, y, names = _veri()
    m = neural.NeuralCapacity("deep_mlp", feature_names=names, seed=2,
                              extra_heads=("dir_evid",)).fit(X, y, max_epochs=2)
    b = m.predict_belief(X[:30])
    ev = m.predict_evidential(X[:30])
    assert b is not None and ev is not None
    assert np.allclose(b["inanc"].sum(axis=1) + b["bosluk"], 1.0, atol=1e-12)
    assert np.all(b["inanc"] >= -1e-12)
    assert np.allclose(b["p"][:, 1], ev[0], atol=1e-12)
    assert np.allclose(b["bosluk"], ev[1], atol=1e-12)
    ens = neural.NeuralEnsemble("deep_mlp", feature_names=names, seed=5, n_members=2,
                                extra_heads=("dir_evid",)).fit(X, y, max_epochs=1)
    be = ens.predict_belief(X[:30])
    eve = ens.predict_evidential(X[:30])
    assert be is not None and eve is not None
    assert np.allclose(be["p"][:, 1], eve[0], atol=1e-9)
    assert np.allclose(be["bosluk"], eve[1], atol=1e-9)
