"""`NEURAL_DEVICE` / `NEURAL_AMP` opt-in kablosu — varsayılan DEĞİŞMEMELİ.

Tur-71: RTX 4070 boştaydı çünkü CUDA eksik değil, `fit(device="cpu", amp=False)`
VARSAYILAN argümanı öyleydi ve hiçbir çağıran başka bir şey geçmiyordu. Ölçüldü:
`quant_transformer` (5,47M) cpu 305,25 s → gpu 6,21 s (49,1×) → gpu+amp 3,36 s (90,9×);
`deep_mlp` (0,22M) gpu+amp 0,45 s = 0,48× (YAVAŞLADI).

Bu dosya iki şeyi kilitler:
  1. env YOKKEN üretim yolu hâlâ `device="cpu", amp=False` geçer (bit-denklik korunur),
  2. amp `deep_mlp`e env'le bile AÇILAMAZ (ölçülen 0,48× gerilemesi).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from nihai import config as cfg
from nihai.ai import models


class _SpyNeural:
    """`NeuralCapacity` yerine geçen casus — fit'e HANGİ argümanların geldiğini yakalar."""

    def __init__(self, tier, feature_names, seed=42):
        self.tier, self.seen = tier, None

    def fit(self, X, y, **kw):
        self.seen = kw
        return self


@pytest.fixture
def spy(monkeypatch):
    box = {}

    def _factory(tier, feature_names, seed=42):
        box["obj"] = _SpyNeural(tier, feature_names, seed)
        return box["obj"]

    monkeypatch.setattr(models, "_neural_mod",
                        lambda: SimpleNamespace(NeuralCapacity=_factory, TORCH_AVAILABLE=True))
    return box


def _fit(tier, spy_box):
    X = np.random.default_rng(0).normal(0, 1, (8, 3))
    y = np.array([0, 1] * 4)
    models.CapacityModel(tier, ["a", "b", "c"]).fit(X, y)
    return spy_box["obj"].seen


@pytest.mark.parametrize("var", ["NEURAL_DEVICE", "NEURAL_AMP"])
def test_env_yokken_varsayilan_degismez(monkeypatch, spy, var):
    """Anahtarlar KAPALIYKEN üretim yolu eski davranışın aynısını geçmeli."""
    monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("NEURAL_DEVICE", raising=False)
    monkeypatch.delenv("NEURAL_AMP", raising=False)
    assert _fit("quant_transformer", spy) == {"device": "cpu", "amp": False}


def test_auto_gecer(monkeypatch, spy):
    monkeypatch.setenv("NEURAL_DEVICE", "auto")
    assert _fit("quant_transformer", spy)["device"] == "auto"


def test_amp_deep_mlp_te_ACILAMAZ(monkeypatch, spy):
    """Ölçülen 0,48× gerilemesi: env açık olsa BİLE küçük tier'a amp verilmez."""
    monkeypatch.setenv("NEURAL_DEVICE", "auto")
    monkeypatch.setenv("NEURAL_AMP", "1")
    assert _fit("deep_mlp", spy) == {"device": "auto", "amp": False}
    assert _fit("quant_transformer", spy) == {"device": "auto", "amp": True}


def test_amp_cpu_da_acilmaz(monkeypatch, spy):
    """`amp` yalnız CUDA'da anlamlı; cihaz cpu iken env açık olsa da False kalır."""
    monkeypatch.setenv("NEURAL_DEVICE", "cpu")
    monkeypatch.setenv("NEURAL_AMP", "1")
    assert _fit("quant_transformer", spy)["amp"] is False


@pytest.mark.parametrize("raw,beklenen", [
    ("auto", "auto"), ("AUTO", "auto"), ("  cuda  ", "cuda"), ("cuda:1", "cuda:1"),
    ("cpu", "cpu"), ("", "cpu"), ("gpu", "cpu"), ("cuda:x", "cpu"), ("çöp", "cpu"),
])
def test_neural_device_fail_soft(monkeypatch, raw, beklenen):
    """Yanlış yazılmış env eğitimi ÇÖKERTMEZ, sessizce cpu'da kalır."""
    monkeypatch.setenv("NEURAL_DEVICE", raw)
    assert cfg.neural_device() == beklenen


@pytest.mark.parametrize("raw,beklenen", [("1", True), ("true", True), ("on", True),
                                          ("0", False), ("", False), ("saçma", False)])
def test_neural_amp_env(monkeypatch, raw, beklenen):
    monkeypatch.setenv("NEURAL_AMP", raw)
    assert cfg.neural_amp() is beklenen
