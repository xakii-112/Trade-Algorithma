"""Tur-93 — canlı gölge sinir ağı 282M `quant_transformer_moe_xl` ile değişirken çıkarım yolu.

Üç soru:
  * TOPLU MC DOĞRU MU: T kopya tek ileri geçişte örneklenir; T=1'de döngüyle BİT-DENK, T>1'de
    her kopya bağımsız maske alır ve satır eşlemesi korunur.
  * CANLI SEÇİM DOĞRU MU: deep_mlp CPU'da ve döngüde kalır (geçmişle bit-denk); büyük basamak
    CUDA'ya (auto) ve toplu MC'ye geçer; aygıt hatası gölgeyi düşürmez.
  * EĞİTİM BAYRAĞI: `--batch-size` verilmezse reçeteye girmez (eski davranış).
"""
from __future__ import annotations

import numpy as np
import pytest

import nihai.execution.neural_shadow as ns
from nihai.ai import neural
from nihai.signals.meta_features import FEATURE_NAMES

pytestmark = pytest.mark.skipif(not neural.TORCH_AVAILABLE, reason="torch yok")


@pytest.fixture(scope="module")
def kucuk_model():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 6))
    y = (X[:, 0] > 0).astype(int)   # güçlü sinyal: satırlar ayırt edilebilsin (eşleme testi)
    m = neural.NeuralCapacity("deep_mlp", [f"f{i}" for i in range(6)], seed=7)
    m.fit(X, y, max_epochs=15)
    return m, X


def test_toplu_T1_dongu_ile_bit_denk(kucuk_model):
    m, X = kucuk_model
    a = m._mc_samples(X[:5], 1)
    b = m._mc_samples(X[:5], 1, batched=True)
    assert a.shape == b.shape == (1, 5)
    assert np.array_equal(a, b)


def test_toplu_kopyalar_bagimsiz_ve_satir_eslemesi_korunur(kucuk_model):
    m, X = kucuk_model
    Xs = X[[0, 50, 100]] * 0.05   # sıfıra yakın → model kararsız, dropout yayılımı görünür
    s = m._mc_samples(Xs, 40, batched=True)
    assert s.shape == (40, 3)
    # Her kopya ayrı dropout maskesi aldıysa kopyalar arası yayılım sıfır olamaz.
    assert float(s.std(axis=0).min()) > 0.0
    # Satır i'nin toplu ortalaması, döngünün AYNI satırına yakın (başka satırın değerine değil).
    d = m._mc_samples(Xs, 40)
    mb, md = s.mean(axis=0), d.mean(axis=0)
    assert np.max(np.abs(mb - md)) < 0.05
    # Aynı tohum → aynı örneklem (deterministik).
    assert np.array_equal(s, m._mc_samples(Xs, 40, batched=True))


def test_toplu_ayirt_edilen_satirlar_yerinde_kalir(kucuk_model):
    m, _ = kucuk_model
    Xs = np.zeros((2, 6))
    Xs[0, 0], Xs[1, 0] = 3.0, -3.0
    mb = m._mc_samples(Xs, 40, batched=True).mean(axis=0)
    md = m._mc_samples(Xs, 40).mean(axis=0)
    assert md[0] > 0.75 and md[1] < 0.25          # model satırları gerçekten ayırıyor
    assert mb[0] > 0.75 and mb[1] < 0.25          # toplu yol her satırı kendi yerinde tutar


def test_toplu_yol_tek_ileri_gecis(kucuk_model):
    m, X = kucuk_model
    sayac = []
    kanca = m._model.register_forward_hook(lambda *a: sayac.append(1))
    try:
        m._mc_samples(X[:3], 20, batched=True)
        toplu = len(sayac)
        m._mc_samples(X[:3], 20)
    finally:
        kanca.remove()
    assert toplu == 1 and len(sayac) - toplu == 20


def test_topluluk_batched_argumanini_uyelere_iletir(kucuk_model):
    m, X = kucuk_model
    ens = neural.NeuralEnsemble("deep_mlp", list(m.feature_names), n_members=2)
    ens.members = [m, m]
    pe, se = ens.predict_with_uncertainty(X[:3], T=5, batched=True)
    pb, sb = m.predict_with_uncertainty(X[:3], T=5, batched=True)
    pd_, _ = m.predict_with_uncertainty(X[:3], T=5)
    assert np.allclose(pe, pb) and np.allclose(se, sb)   # özdeş iki üye → üye sonucunun aynısı
    assert not np.allclose(pe, pd_)


def test_predict_with_uncertainty_toplu_sekil(kucuk_model):
    m, X = kucuk_model
    p, sd = m.predict_with_uncertainty(X[:4] * 0.05, T=10, batched=True)
    assert p.shape == sd.shape == (4,)
    assert np.all((p > 0) & (p < 1)) and np.all(sd >= 0)


def test_to_device_cpu_ve_kurulamayan_cuda_cpuya_duser(kucuk_model, monkeypatch):
    m, _ = kucuk_model
    m._device = "eski"   # çıkarım tensörleri bu alandan kurulur; taşıma onu güncellemeli
    assert str(m.to_device("cpu")) == "cpu" and str(m._device) == "cpu"
    monkeypatch.setattr(neural.torch.cuda, "is_available", lambda: False)
    assert str(m.to_device("cuda")) == "cpu"
    assert str(m._device) == "cpu"


# ------------------------------------------------------------------ canlı seçim
class _Sahte:
    def __init__(self, tier, patla=False):
        self.tier = tier
        self.feature_names = list(FEATURE_NAMES)
        self.n_members = 1
        self.istenen: list[str] = []
        self.kw: dict = {}
        self._patla = patla

    def to_device(self, dev):
        self.istenen.append(dev)
        if self._patla and dev != "cpu":
            raise RuntimeError("CUDA out of memory")
        return dev

    def predict_with_uncertainty(self, X, T=20, **kw):
        self.kw = kw
        return [0.55], [0.03]


def _kur(monkeypatch, model):
    monkeypatch.delenv("NEURAL_SHADOW", raising=False)
    monkeypatch.setattr(ns._NeuralShadow, "_pick", lambda self: ("x", None))
    monkeypatch.setattr(neural, "load_any", lambda base: model)
    return ns._NeuralShadow()


def _feats():
    return {n: 0.0 for n in FEATURE_NAMES}


def test_deep_mlp_cpu_ve_dongu_kalir(monkeypatch):
    monkeypatch.delenv("NEURAL_SHADOW_DEVICE", raising=False)
    m = _Sahte("deep_mlp")
    sh = _kur(monkeypatch, m)
    assert m.istenen == ["cpu"] and sh.status()["device"] == "cpu"
    assert sh.pwin(_feats(), "LONG") is not None
    assert m.kw == {}   # batched argümanı HİÇ geçmez → geçmiş çıkarım yolu aynı


def test_buyuk_basamak_auto_ile_cuda_ve_toplu(monkeypatch):
    monkeypatch.delenv("NEURAL_SHADOW_DEVICE", raising=False)
    m = _Sahte("quant_transformer_moe_xl")
    sh = _kur(monkeypatch, m)
    assert m.istenen == ["cuda"] and sh.status()["device"] == "cuda"
    assert sh.pwin(_feats(), "SHORT") == (0.55, 0.03)
    assert m.kw == {"batched": True}


def test_env_cpu_buyuk_basamagi_cpuda_tutar(monkeypatch):
    monkeypatch.setenv("NEURAL_SHADOW_DEVICE", "cpu")
    m = _Sahte("quant_transformer_moe_xl")
    sh = _kur(monkeypatch, m)
    assert m.istenen == ["cpu"] and sh.device == "cpu"
    sh.pwin(_feats(), "LONG")
    assert m.kw == {"batched": True}


def test_aygit_hatasi_golgeyi_dusurmez(monkeypatch):
    monkeypatch.delenv("NEURAL_SHADOW_DEVICE", raising=False)
    m = _Sahte("quant_transformer_moe_xl", patla=True)
    sh = _kur(monkeypatch, m)
    assert m.istenen == ["cuda", "cpu"] and sh.device == "cpu"
    assert sh.active and sh.pwin(_feats(), "LONG") is not None


# ------------------------------------------------------------------ eğitim bayrağı
def _yakala(monkeypatch):
    import research.train_neural as tn
    alinan: dict = {}

    def sahte_cv(X, y, names, **kw):
        alinan.update(kw["recipe"])
        return {"auc_mean": None, "dsr_report": {"dsr": None, "n": 0}}

    monkeypatch.setattr(tn, "run_cv", sahte_cv)
    return tn, alinan


def test_batch_size_verilmezse_recetede_yok(monkeypatch):
    tn, alinan = _yakala(monkeypatch)
    assert tn.main(["--synthetic", "60", "--tier", "deep_mlp", "--folds", "2"]) == 0
    assert "batch_size" not in alinan


def test_batch_size_recete_ile_fite_gider(monkeypatch):
    tn, alinan = _yakala(monkeypatch)
    assert tn.main(["--synthetic", "60", "--tier", "deep_mlp", "--folds", "2",
                    "--batch-size", "128"]) == 0
    assert alinan["batch_size"] == 128
