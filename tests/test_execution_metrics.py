"""test_execution_metrics.py — backtest metrik matematiği (max_drawdown / sharpe /
monte_carlo / _period_to_bars) + live_predict_service yön/rejim yardımcıları.

Bu karar-yolları test kapsamı dışındaydı; saf-fonksiyon oldukları için sağlayıcı/ağ
gerektirmeden bilinen girdi→çıktı eşlemeleriyle kilitlenir."""
from __future__ import annotations

import pandas as pd
import pytest

import nihai.execution.backtest as bt
from nihai.execution.live_predict_service import (
    _bias_and_confidence,
    _htf_direction,
    _regime_tilt,
)


# --------------------------------------------------------------------- max_drawdown
def test_max_drawdown_known_series():
    # tepe 120'de, dip 90 → (120-90)/120 = %25
    assert bt.max_drawdown([100, 120, 90, 130]) == pytest.approx(25.0)


def test_max_drawdown_empty_and_monotonic():
    assert bt.max_drawdown([]) == 0.0
    assert bt.max_drawdown([100, 110, 120]) == 0.0     # hiç düşüş yok


# --------------------------------------------------------------------- sharpe_annualized
def test_sharpe_known_value():
    # mean=0.15, pstdev=0.05 → 3.0 × sqrt(252)
    assert bt.sharpe_annualized([0.1, 0.2], 252) == pytest.approx(3.0 * 252 ** 0.5)


def test_sharpe_degenerate_returns_zero():
    assert bt.sharpe_annualized([0.05], 252) == 0.0        # len < 2
    assert bt.sharpe_annualized([0.05, 0.05], 252) == 0.0  # sd == 0


# --------------------------------------------------------------------- monte_carlo_paths
def test_monte_carlo_empty_is_empty():
    assert bt.monte_carlo_paths([], 1000.0, n_iters=10) == {}


def test_monte_carlo_deterministic_seed_and_ruin_bounds():
    pnls = [40.0, -60.0, 25.0, -50.0, 30.0]
    a = bt.monte_carlo_paths(pnls, 100.0, n_iters=500, seed=42)
    b = bt.monte_carlo_paths(pnls, 100.0, n_iters=500, seed=42)
    assert a == b                                     # seed=42 → birebir tekrarlanır
    assert 0.0 <= a["ruin_probability"] <= 1.0
    # final equity sıra-bağımsız (PnL toplamı sabit) → tüm persentiller = initial + sum
    assert a["final_equity_p50"] == pytest.approx(100.0 + sum(pnls))
    assert a["max_drawdown_p05"] <= a["max_drawdown_p50"] <= a["max_drawdown_p95"]


# --------------------------------------------------------------------- _period_to_bars
def test_period_to_bars_defaults():
    assert bt._period_to_bars(None, "1h") == 2000
    assert bt._period_to_bars("garbage", "1h") == 2000


def test_period_to_bars_bounds():
    assert bt._period_to_bars("1d", "1d") == 50          # alt sınır
    assert bt._period_to_bars("100y", "1d") == 10000     # üst sınır
    for p in ("6mo", "1y", "2y", "5y"):
        assert 50 <= bt._period_to_bars(p, "1d") <= 10000


def test_period_to_bars_monotonic_in_period():
    assert (bt._period_to_bars("6mo", "1d") <= bt._period_to_bars("1y", "1d")
            <= bt._period_to_bars("2y", "1d"))


# ----------------------------------------------- live_predict_service._htf_direction
def _close_df(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="h")
    return pd.DataFrame({"close": closes}, index=idx)


def test_htf_direction_insufficient_or_missing():
    assert _htf_direction(None) == 0
    assert _htf_direction(pd.DataFrame()) == 0
    assert _htf_direction(_close_df(list(range(10)))) == 0   # < 50 bar


def test_htf_direction_up_and_down():
    up = _close_df([100 + i for i in range(60)])             # EMA20 > EMA50
    down = _close_df([160 - i for i in range(60)])           # EMA20 < EMA50
    assert _htf_direction(up) == 1
    assert _htf_direction(down) == -1


def test_htf_direction_duz_pencerede_yon_YOK():
    """BULGU-61-1: tamamen DÜZ HTF penceresi 'ayı onayı' DEĞİL 'bilgi yok'tur.

    Eski kod `1 if e20 > e50 else -1` yazıyordu → e20 == e50 tam eşitliği sessizce −1
    sayılıyordu ve fonksiyonun kendi docstring'i ("belirsiz → 0") ile çelişiyordu.
    Zarar dolaylıdır ama ölçülebilir: htf_dir −1 iken `_bias_and_confidence`
    `conf *= 0.65` uygular, yani SIFIR bilgi %35 güven cezasına dönüşür."""
    duz = _close_df([100.0] * 80)
    assert _htf_direction(duz) == 0

    _d, _b, c_notr = _bias_and_confidence(105.0, 104.0, 102.0, 62.0, 28.0, htf_dir=0)
    _d, _b, c_ayi = _bias_and_confidence(105.0, 104.0, 102.0, 62.0, 28.0, htf_dir=-1)
    assert c_ayi < c_notr                    # uydurma "ayı onayı"nın bedeli ölçülüyor


def test_bias_ve_confidence_NaN_girdiyi_SHORT_saymaz():
    """Tur-61: `_yon` kopyası `vote`a inince NaN fiyat da 'bilgi yok' oldu.

    Eskiden `_yon(nan, e20)` → −1.0 (tam SHORT): ölçülemeyen bir girdi yönlü oy
    üretiyordu (Tur-31 dersi: ölçüm yokluğu ödüllendirilmez — burada CEZALANDIRILIYORDU).
    Sonlu girdilerde davranış değişmedi; testin ikinci yarısı bunu kilitler."""
    import numpy as _np
    _d, bias_nan, _c = _bias_and_confidence(float("nan"), 104.0, 102.0, 50.0, 20.0)
    # bileşenler: vote(e20,e50)=+1, rsi=0.0, vote(nan,e20)=0  →  ortalama +1/3
    assert bias_nan == pytest.approx(1.0 / 3.0)
    _d, bias_ok, _c = _bias_and_confidence(105.0, 104.0, 102.0, 50.0, 20.0)
    assert bias_ok == pytest.approx(2.0 / 3.0)   # sonlu girdide eski davranış: +1,0,+1
    assert not _np.isnan(bias_nan)


# ----------------------------------------------- live_predict_service._regime_tilt
def test_regime_tilt_mappings():
    assert _regime_tilt(None, "+") == 1.0
    assert _regime_tilt("BULLISH", "+") == 1.10
    assert _regime_tilt("BULLISH", "-") == 0.85
    assert _regime_tilt("DEFENSIVE", "+") == 0.80
    assert _regime_tilt("DEFENSIVE", "-") == 1.10
    assert _regime_tilt("NEUTRAL", "+") == 1.0       # bilinmeyen rejim → nötr
