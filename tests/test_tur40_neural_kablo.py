"""Gölge sinir ağı KÖPRÜSÜ (kablo) — shadow_runner → brain.think(neural_model=...).

BULGU: `brain.think()` üretimde HER ZAMAN `neural_model=None` ile çağrılıyordu; diskteki
eğitilmiş artefaktı canlı akışa yükleyen tek satır kod yoktu. ÖNGÖR adımı böylece hiç
çalışmadı. Bu dosya kablonun TAKILI olduğunu ve yanlış şemalı artefaktın ATLANDIĞINI
çapalar. Ağ çağrısı yok: provider sahte, VIX ısıtması kapalı.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.ai import shadow_runner


class _FakeProvider:
    def get_bars(self, symbol: str, interval: str, lookback_bars: int = 200):
        idx = pd.date_range("2026-01-01", periods=90, freq="D")
        closes = 100.0 + np.arange(90, dtype=float) * 0.1
        return pd.DataFrame({"open": closes - 0.05, "high": closes + 0.2,
                             "low": closes - 0.2, "close": closes,
                             "volume": np.full(90, 1e6)}, index=idx)


@pytest.fixture(autouse=True)
def _izole(monkeypatch, tmp_path):
    monkeypatch.setattr(shadow_runner, "WARM_VIX", False, raising=False)
    monkeypatch.setattr(shadow_runner, "SHADOW_LOG_FILE", tmp_path / "run.jsonl")
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider())
    monkeypatch.setattr(shadow_runner, "_shadow_portfolio_view", lambda *a, **k: None)


def _reset_cache(monkeypatch, model):
    monkeypatch.setattr(shadow_runner, "_NEURAL_LOADED", True, raising=False)
    monkeypatch.setattr(shadow_runner, "_NEURAL_MODEL", model, raising=False)


def test_think_neural_model_ile_cagrilir(monkeypatch):
    """KABLO: run_once, brain.think'e neural_model'i GEÇER (eskiden hiç geçmiyordu)."""
    gorulen: list = []
    sentinel = object()
    _reset_cache(monkeypatch, sentinel)

    gercek_think = shadow_runner.brain.think

    def _sahte_think(symbol, strategy, **kw):
        gorulen.append(kw.get("neural_model"))
        return gercek_think(symbol, strategy, features=kw.get("features"),
                            closes=kw.get("closes"))

    monkeypatch.setattr(shadow_runner.brain, "think", _sahte_think)
    out = shadow_runner.run_once(symbols=["AAPL", "MSFT"], max_symbols=2)
    assert out["n_ok"] == 2
    assert gorulen == [sentinel, sentinel]


def test_ozet_n_neural_tasir(monkeypatch):
    """Kablonun ÖLÇÜSÜ özette görünür: ÖNGÖR üreten sembol sayısı."""
    _reset_cache(monkeypatch, None)
    out = shadow_runner.run_once(symbols=["AAPL"], max_symbols=1)
    assert out["n_neural"] == 0          # model yok → adım pasif, sessizce değil GÖRÜNÜR


def test_uyumsuz_sema_atlanir(monkeypatch, tmp_path):
    """ŞEMA KAPISI: feature_names canlı şemayla eşleşmeyen artefakt YÜKLENMEZ.
    Eşleşmeseydi brain eksik özellikleri 0.0 doldurup gürültüyü p_win diye taşırdı."""
    neural = pytest.importorskip("nihai.ai.neural")
    if not neural.TORCH_AVAILABLE:
        pytest.skip("torch yok")

    class _Sahte:
        is_fitted = True
        feature_names = ["f0", "f1"]      # canlı şema 25 ad taşır

    monkeypatch.setattr(shadow_runner, "_NEURAL_LOADED", False, raising=False)
    monkeypatch.setattr(shadow_runner, "_NEURAL_MODEL", None, raising=False)
    monkeypatch.setattr(shadow_runner, "_neural_bases", lambda: [tmp_path / "x"])
    monkeypatch.setattr(neural, "load_any", lambda base: _Sahte())
    assert shadow_runner._load_neural_model() is None


def test_diskteki_artefakt_semayla_uyumlu():
    """Diskteki canlı artefakt gerçekten yüklenebiliyor ve şeması TUTUYOR (yoksa atla)."""
    neural = pytest.importorskip("nihai.ai.neural")
    if not neural.TORCH_AVAILABLE:
        pytest.skip("torch yok")
    from nihai.signals.meta_features import FEATURE_NAMES
    bases = shadow_runner._neural_bases()
    if not bases:
        pytest.skip("diskte neural artefakt yok")
    m = neural.load_any(bases[0])
    if m is None:
        pytest.skip("artefakt okunamadı")
    assert list(m.feature_names) == list(FEATURE_NAMES)
    p, sd = m.predict_with_uncertainty(np.zeros((1, len(FEATURE_NAMES))))
    assert 0.0 <= float(np.ravel(p)[0]) <= 1.0 and float(np.ravel(sd)[0]) >= 0.0
