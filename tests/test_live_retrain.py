"""live_retrain.assess — kapalı eğitim döngüsünün verdikt disiplini.

Sentetik veriyle üç sözleşme:
1. Guardrail: küçük örneklem → INSUFFICIENT (asla kenar iddiası yok).
2. Öğrenilebilir sinyal (bir özellik etiketi belirliyor) → DURABLE + eşik döner.
3. Saf gürültü etiketleri → NO_EDGE (DURABLE asla dönmemeli — yanlış terfi koruması).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.signals.meta_features import FEATURE_NAMES
from research.live_retrain import assess


def _make_df(n: int, *, informative: bool, n_symbols: int = 20, seed: int = 7) -> pd.DataFrame:
    """entry_time artan, n_symbols sembollü sentetik dataset. informative=True ise
    ilk özellik etiketi (gürültülü) belirler ve pnl etiketle hizalıdır."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    if informative:
        margin = X[:, 0] + 0.3 * rng.normal(size=n)
        label = (margin > 0).astype(int)
    else:
        label = rng.integers(0, 2, size=n)
    pnl = np.where(label == 1, 1.0, -1.0) + 0.1 * rng.normal(size=n)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = label
    df["pnl"] = pnl
    df["symbol"] = [f"SYM{i % n_symbols}" for i in range(n)]
    df["period"] = "live"
    df["entry_time"] = pd.date_range("2026-01-01", periods=n, freq="4h").astype(str)
    return df


def test_insufficient_sample_refuses_verdict():
    df = _make_df(50, informative=True)
    res = assess(df, kinds=["logistic"], verbose=False)
    assert res["verdict"] == "INSUFFICIENT"
    assert res["durable_kind"] is None


def test_informative_signal_is_durable():
    df = _make_df(600, informative=True)
    res = assess(df, kinds=["logistic"], verbose=False)
    assert res["verdict"] == "DURABLE"
    assert res["durable_kind"] == "logistic"
    assert res["threshold"] is not None
    cv = res["results"]["logistic"]["cv"]
    assert cv["mean_auc"] > 0.53 and cv["durable"]


def test_noise_labels_never_durable():
    df = _make_df(600, informative=False)
    res = assess(df, kinds=["logistic"], verbose=False)
    assert res["verdict"] == "NO_EDGE"
    assert res["durable_kind"] is None


def test_span_guardrail():
    """Satır/sembol yeterli ama süre < min gün → INSUFFICIENT (tek günlük patlama
    verisinden model eğitme tuzağı)."""
    df = _make_df(600, informative=True)
    df["entry_time"] = pd.date_range("2026-01-01", periods=len(df), freq="1min").astype(str)
    res = assess(df, kinds=["logistic"], verbose=False)
    assert res["verdict"] == "INSUFFICIENT"


@pytest.mark.parametrize("bad_col", ["label", "pnl"])
def test_required_columns(bad_col):
    """Eksik zorunlu kolon KeyError yerine sessiz yanlış sonuç ÜRETMEMELİ."""
    df = _make_df(600, informative=True).drop(columns=[bad_col])
    with pytest.raises(KeyError):
        assess(df, kinds=["logistic"], verbose=False)


def test_min_gate_keep_guard_blocks_thin_gate():
    """B1: kapı kenarı yeterli örnekleme (>= min_gate_keep tutulan işlem) dayanmıyorsa
    DAYANIKLI SAYILMAZ. Canlı --promote'un avuç-dolusu işlemle sahte kenarı üretime
    yazmasına karşı guard (train_meta --save ile aynı disiplin). Aynı informatif veri
    default(50)'de DAYANIKLI iken yüksek eşikte NO_EDGE'e döner."""
    df = _make_df(600, informative=True)
    assert assess(df, kinds=["logistic"], verbose=False)["verdict"] == "DURABLE"
    res = assess(df, kinds=["logistic"], min_gate_keep=200, verbose=False)
    assert res["verdict"] == "NO_EDGE"
    assert res["durable_kind"] is None


def test_promoted_model_and_threshold_are_consistent():
    """B5: terfi için sunulan model = OOS-değerlendirilen TIME alt-modeli; verdikt eşiği de
    o alt-modelin best_gate eşiğinden gelir. Model+eşik AYNI dağılımdan olduğundan, o eşikte
    time-test'te tutulan işlem sayısı best_gate[4] ile BİRE BİR eşleşmeli. (Eskiden burada
    tam-veride TAZE model eğitilip alt-model eşiğiyle kaydediliyordu → olasılık ölçeği kayar,
    tutulan sayı best_gate ile tutmazdı = sessiz model↔eşik uyuşmazlığı.)"""
    from research.train_meta import split_time

    df = _make_df(600, informative=True)
    res = assess(df, kinds=["logistic"], verbose=False)
    assert res["verdict"] == "DURABLE"
    kind = res["durable_kind"]
    tsplit = res["results"][kind]["splits"]["time"]
    model = tsplit["model"]
    thr, _pf, _e, _wr, kept = tsplit["best_gate"]
    assert res["threshold"] == thr           # verdikt eşiği = time alt-model eşiği
    _, te = split_time(df, 0.6)
    pte = model.predict_proba(te[FEATURE_NAMES].to_numpy(float))
    assert int((pte >= thr).sum()) == kept   # deploy modeli, deploy eşiğinde gate'i üretir
