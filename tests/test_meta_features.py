"""Meta-labeling özellik vektörü testleri — train/serve hiza güvencesi."""
import numpy as np
import pandas as pd
import pytest

from nihai.signals.meta_features import (
    FEATURE_NAMES,
    ensure_feature_columns,
    extract_features,
    features_to_vector,
    neutral_default,
)


def test_features_to_vector_default_order():
    # names verilmezse güncel FEATURE_NAMES sırası + uzunluğu.
    feats = {n: float(i) for i, n in enumerate(FEATURE_NAMES)}
    v = features_to_vector(feats)
    assert len(v) == len(FEATURE_NAMES)
    assert v == [float(i) for i in range(len(FEATURE_NAMES))]


def test_features_to_vector_respects_custom_names():
    # Modelle kaydedilmiş eğitim-anı sırası geçilince vektör O sırada kurulur
    # (aynı-uzunlukta yeniden-sıralama sessiz skew'i: bu fix onu engeller).
    feats = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert features_to_vector(feats, ["c", "a", "b"]) == [3.0, 1.0, 2.0]
    assert features_to_vector(feats, ["b", "c"]) == [2.0, 3.0]


def test_features_to_vector_missing_feature_is_zero():
    # Modelde olup sözlükte olmayan özellik → 0.0 (NaN/eksik nötr), patlamaz.
    assert features_to_vector({"a": 5.0}, ["a", "missing"]) == [5.0, 0.0]


# --------------------------------------------------- 2026-07-03 genişletmesi (19→25)

def _bars(closes, *, start="2026-06-01 10:00", freq="1h"):
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": c * 0.999, "high": c * 1.01, "low": c * 0.99,
        "close": c, "volume": np.full(len(c), 1_000_000.0),
    }, index=idx)


class _Sig:
    action = None
    rr_ratio = 2.0
    quality_score = 70.0
    reasons: list = []


def test_extract_features_covers_all_names_and_new_micro_features():
    """Her FEATURE_NAMES anahtarı üretilir; yeni mikro-yapı özellikleri sınır içinde."""
    df = _bars(np.linspace(100, 110, 260))
    feats = extract_features(df, _Sig())
    assert set(feats) == set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in feats.values())
    assert 0.0 <= feats["close_pos"] <= 1.0
    assert feats["range_pct"] > 0.0
    assert feats["hour"] == float(df.index[-1].hour)
    # Monoton artan seri: streak pozitif ve ±10'a kırpılı; rsi_slope sonlu.
    assert 1 <= feats["streak"] <= 10


def test_extract_features_streak_direction_and_gap():
    """streak yön işaretini taşır; gap_pct = (open - önceki kapanış)/önceki kapanış."""
    closes = [100, 101, 102, 101, 100, 99]       # son 3 bar düşüş → streak = -3
    df = _bars(closes)
    feats = extract_features(df, _Sig())
    assert feats["streak"] == -3.0
    exp_gap = (df["open"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]
    assert feats["gap_pct"] == pytest.approx(exp_gap)


def test_extract_features_empty_df_neutral_defaults():
    """Boş df guard'ı yeni özellikler dahil NÖTR tabloyu kullanır (close_pos=0.5)."""
    feats = extract_features(pd.DataFrame(), _Sig())
    assert set(feats) == set(FEATURE_NAMES)
    assert feats["close_pos"] == 0.5 and feats["rsi14"] == 50.0 and feats["rvol"] == 1.0
    assert feats["gap_pct"] == 0.0 and feats["hour"] == 0.0


def test_ensure_feature_columns_upgrades_old_dataset():
    """Eski-şema CSV (eksik yeni kolonlar + NaN hücre) → nötr-doldurulur, KeyError yok."""
    old = pd.DataFrame({"adx14": [20.0, np.nan], "rsi14": [55.0, np.nan],
                        "pnl": [1.0, -1.0], "label": [1, 0]})
    df, added = ensure_feature_columns(old)
    assert set(added) == set(FEATURE_NAMES) - {"adx14", "rsi14"}
    X = df[FEATURE_NAMES].to_numpy(float)      # train_meta'nın yaptığı çağrı
    assert np.isfinite(X).all()
    assert df["rsi14"].iloc[1] == neutral_default("rsi14")   # NaN → 50
    assert df["close_pos"].iloc[0] == 0.5
