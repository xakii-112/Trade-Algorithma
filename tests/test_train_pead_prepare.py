"""B7 — train_pead_ml.prepare() NaN/şema güvenliği.

Değişmez: prepare() sonrası modele giren HİÇBİR özellik (generic + PEAD ekstraları) NaN
olmamalı ve eski-şema (eksik generic kolon) KeyError yerine nötr-dolgulu çalışmalı.
NaN'ın modele sızması sessiz bozulmadır (CLAUDE.md: NaN asla modele ulaşmaz)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nihai.signals.meta_features import FEATURE_NAMES
from research.train_pead_ml import PEAD_FEATURES, prepare


def test_prepare_no_nan_when_surprise_sue_volume_missing_or_nan():
    """Tümü-NaN surprise/sue + tümü-0 dollar_vol + bir NaN generic → hepsi finite döner."""
    n = 10
    df = pd.DataFrame({f: [0.0] * n for f in FEATURE_NAMES})
    df.loc[0, FEATURE_NAMES[0]] = np.nan          # generic NaN (winsorize öncesi sızıntı riski)
    df["surprise"] = [np.nan] * n                  # eşleşmeyen earnings → NaN payda
    df["sue"] = [np.nan] * n
    df["dollar_vol_med"] = [0.0] * n               # hepsi 0 → rank NaN riski
    df["label"] = [0, 1] * (n // 2)
    df["pnl"] = [1.0, -1.0] * (n // 2)
    out = prepare(df)
    X = out[PEAD_FEATURES].to_numpy(float)
    assert np.isfinite(X).all()                    # NaN/inf modele ULAŞMAZ


def test_prepare_upgrades_old_schema_missing_generic_columns():
    """Eski-şema dataset (generic kolonlar yok) → ensure_feature_columns ile nötr eklenir,
    KeyError atmaz; tüm PEAD_FEATURES mevcut ve finite."""
    df = pd.DataFrame({
        "surprise": [1.0, 2.0, -1.0, 3.0], "sue": [0.1, 0.2, -0.1, 0.3],
        "dollar_vol_med": [1e6, 2e6, 3e6, 4e6],
        "label": [1, 0, 1, 0], "pnl": [1.0, -1.0, 1.0, -1.0],
    })
    out = prepare(df)
    assert all(f in out.columns for f in PEAD_FEATURES)
    assert np.isfinite(out[PEAD_FEATURES].to_numpy(float)).all()


def test_prepare_winsorizes_surprise_extremes():
    """surprise aşırı-uçları robust aralığa kırpılır (surprise_w ham surprise'tan dar)."""
    vals = [-500.0] + [0.5] * 98 + [500.0]         # iki aykırı + normal kütle
    df = pd.DataFrame({f: [0.0] * 100 for f in FEATURE_NAMES})
    df["surprise"] = vals
    df["sue"] = [0.0] * 100
    df["dollar_vol_med"] = list(range(1, 101))
    df["label"] = [0, 1] * 50
    df["pnl"] = [1.0, -1.0] * 50
    out = prepare(df)
    assert out["surprise_w"].max() < 500.0 and out["surprise_w"].min() > -500.0
    assert np.isfinite(out[PEAD_FEATURES].to_numpy(float)).all()
