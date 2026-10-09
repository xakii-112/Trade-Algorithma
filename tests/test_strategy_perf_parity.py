"""Tur-6 perf optimizasyonu parite guard'ları.

Optimizasyonlar DAVRANIŞ DEĞİŞTİRMEZ sözü verdi; bu testler sözü kilitler:
1. numpy `_true_range` == eski pd.concat(...).max(axis=1) implementasyonu (bit-bit).
2. Önceden-hesaplanmış parametre (atr_series/atr_val/adx_val/rsi_val) geçilen filtre/
   skor fonksiyonları, parametresiz (kendisi hesaplayan) halleriyle AYNI sonucu verir.
3. generate_signal: optimize akış, tüm baz stratejilerde aynı sinyali üretir
   (aksiyon/stop/target/skor/RR) — sentetik veri, ağsız.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.signals.strategy import (
    SignalAction,
    _SharedIndicators,
    _true_range,
    adx,
    atr,
    bollinger_bands,
    compute_stops_targets,
    detect_donchian,
    detect_event,
    detect_macd,
    detect_mean_reversion,
    detect_rsi_divergence,
    detect_squeeze,
    detect_trend_pullback,
    ema,
    generate_signal,
    macd,
    regime_filter,
    rsi,
    signal_quality_score,
    volatility_explosion_filter,
)


def _true_range_ref(df: pd.DataFrame) -> pd.Series:
    """Eski (Tur-6 öncesi) implementasyon — referans."""
    h, l, c = df["high"], df["low"], df["close"]
    return pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)


def _make_df(n=300, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = rng.integers(1e5, 5e6, n).astype(float)
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="h")
    return pd.DataFrame({"open": open_, "high": np.maximum(high, close),
                         "low": np.minimum(low, close), "close": close,
                         "volume": vol}, index=idx)


# ---------------------------------------------------------------------------
# 1) _true_range: numpy == eski concat/max
# ---------------------------------------------------------------------------
def test_true_range_matches_reference():
    df = _make_df(400)
    new = _true_range(df)
    ref = _true_range_ref(df)
    pd.testing.assert_series_equal(new, ref, check_names=False)


def test_true_range_single_row_and_empty():
    df = _make_df(1)
    # tek bar: shift-NaN → TR = high-low (eski skipna davranışı)
    new = _true_range(df)
    assert float(new.iloc[0]) == pytest.approx(float(df["high"].iloc[0] - df["low"].iloc[0]))
    empty = df.iloc[0:0]
    assert _true_range(empty).empty


def test_true_range_with_nan_rows():
    df = _make_df(50)
    df.loc[df.index[10], ["high", "low", "close"]] = np.nan
    new = _true_range(df)
    ref = _true_range_ref(df)
    pd.testing.assert_series_equal(new, ref, check_names=False)


# ---------------------------------------------------------------------------
# 2) Önceden-hesaplanmış parametre pariteleri
# ---------------------------------------------------------------------------
def test_vol_filter_precomputed_parity():
    df = _make_df()
    assert volatility_explosion_filter(df) == \
        volatility_explosion_filter(df, atr_series=atr(df, 14))


def test_regime_filter_precomputed_parity():
    df = _make_df()
    assert regime_filter(df) == regime_filter(df, adx_val=adx(df).iloc[-1])


def test_stops_targets_precomputed_parity():
    df = _make_df()
    a = float(atr(df, 14).iloc[-1])
    for action in (SignalAction.LONG, SignalAction.SHORT):
        for mode in ("event_breakout", "mean_reversion"):
            assert compute_stops_targets(df, action, mode) == \
                compute_stops_targets(df, action, mode, atr_val=a)


def test_quality_score_precomputed_parity():
    df = _make_df()
    kw = dict(htf_trend_aligned=True, iv_rank_val=40.0, chain_health_ok=True, confluence=2)
    base = signal_quality_score(df, SignalAction.LONG, **kw)
    pre = signal_quality_score(df, SignalAction.LONG, **kw,
                               adx_val=adx(df).iloc[-1], rsi_val=rsi(df["close"]).iloc[-1])
    assert base == pre


# ---------------------------------------------------------------------------
# 3) generate_signal: optimize akış tüm baz modlarda aynı sinyali üretir
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 4) Tur-7 — NEDENSELLİK/PREFIX değişmezi: tam df üzerinde hesaplanıp ilk n bara
#    kırpılan gösterge, df.iloc[:n] üzerinde sıfırdan hesaplananla BİT-BİT aynı.
#    _SharedIndicators.upto() mimarisinin (O(n²)→O(n)) tek dayanağı budur; kırılırsa
#    backtest sinyalleri canlıdan sapar → check_exact=True ile kilitlenir.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [61, 150, 299])
def test_indicator_prefix_identity(n):
    df = _make_df(300, seed=5)
    head = df.iloc[:n]
    c_full, c_head = df["close"], head["close"]

    def eq(a, b):
        pd.testing.assert_series_equal(a.iloc[:n], b, check_names=False, check_exact=True)

    eq(ema(c_full, 20), ema(c_head, 20))
    eq(ema(c_full, 50), ema(c_head, 50))
    eq(rsi(c_full, 14), rsi(c_head, 14))
    eq(atr(df, 14), atr(head, 14))
    eq(adx(df), adx(head))
    for s_full, s_head in zip(bollinger_bands(c_full), bollinger_bands(c_head), strict=True):
        eq(s_full, s_head)
    for s_full, s_head in zip(macd(c_full), macd(c_head), strict=True):
        eq(s_full, s_head)


@pytest.mark.parametrize("n", [80, 160, 299])
def test_detectors_shared_upto_parity(n):
    """Her detektör: shared'sız == çağrı-içi shared == tam-df upto(n) görünümü."""
    df = _make_df(300, seed=13)
    head = df.iloc[:n]
    full_cache = _SharedIndicators(df)
    detectors = [detect_event, detect_macd, detect_donchian, detect_squeeze,
                 detect_mean_reversion, detect_rsi_divergence, detect_trend_pullback]
    for fn in detectors:
        plain = fn(head)
        incall = fn(head, shared=_SharedIndicators(head))
        upto = fn(head, shared=full_cache.upto(n))
        assert plain == incall == upto, fn.__name__


@pytest.mark.parametrize("mode", ["event_breakout", "mean_reversion", "rsi_divergence",
                                  "macd", "donchian", "squeeze", "trend_pullback",
                                  "ensemble", "confluence"])
def test_generate_signal_shared_upto_parity(mode):
    """generate_signal: backtest'in kullandığı upto() yolu, shared'sız yol ile
    tüm alanlarda birebir aynı sinyali üretir."""
    df = _make_df(320, seed=17)
    full_cache = _SharedIndicators(df)
    for i in (240, 280, 319):
        s1 = generate_signal("SYN", df.iloc[:i + 1], strategy_mode=mode)
        s2 = generate_signal("SYN", df.iloc[:i + 1], strategy_mode=mode,
                             shared=full_cache.upto(i + 1))
        assert (s1.action, s1.price, s1.stop, s1.target, s1.quality_score,
                s1.rr_ratio, s1.reasons) == \
               (s2.action, s2.price, s2.stop, s2.target, s2.quality_score,
                s2.rr_ratio, s2.reasons), mode


@pytest.mark.parametrize("mode", ["event_breakout", "mean_reversion", "rsi_divergence",
                                  "macd", "donchian", "squeeze", "trend_pullback",
                                  "vol_breakout", "ensemble"])
def test_generate_signal_stable_across_slices(mode):
    """Sinyal alanları deterministik ve dilim-tekrarında aynı (memo sızıntısı /
    durum kirliliği yok). Aynı dilime iki çağrı → birebir aynı çıktı."""
    df = _make_df(320, seed=11)
    for i in (250, 280, 319):
        s1 = generate_signal("SYN", df.iloc[:i + 1], strategy_mode=mode)
        s2 = generate_signal("SYN", df.iloc[:i + 1], strategy_mode=mode)
        assert (s1.action, s1.price, s1.stop, s1.target, s1.quality_score,
                s1.rr_ratio, s1.reasons) == \
               (s2.action, s2.price, s2.stop, s2.target, s2.quality_score,
                s2.rr_ratio, s2.reasons)
