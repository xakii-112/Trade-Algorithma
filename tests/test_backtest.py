"""Backtest motoru testleri — offline (uydurma bar), Madde 4/15/52."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import nihai.execution.backtest as bt
from nihai.config import CONFIG
from nihai.signals.strategy import Signal, SignalAction


@pytest.fixture
def synth_df():
    np.random.seed(7)
    idx = pd.date_range("2024-01-01", periods=300, freq="D")
    base = 100 + np.cumsum(np.random.normal(0.12, 1.0, 300))
    close = np.maximum(base, 5)
    high = close + np.abs(np.random.normal(0.5, 0.4, 300))
    low = close - np.abs(np.random.normal(0.5, 0.4, 300))
    op = close + np.random.normal(0, 0.3, 300)
    vol = np.random.randint(1_000_000, 5_000_000, 300).astype(float)
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_stock_backtest_runs(synth_df):
    r = bt.backtest_symbol("AAA", interval="1d", df=synth_df, strategy_mode="event_breakout")
    assert r.bars_total == 300
    assert r.trades_count >= 0
    assert len(r.equity_curve) > 1


def test_option_synth_backtest_runs(synth_df):
    r = bt.backtest_option_synth("AAA", interval="1d", df=synth_df, strategy_mode="event_breakout")
    assert r.symbol == "OPT:AAA"
    assert r.final_equity > 0


def test_realized_iv_disabled_returns_constant(synth_df):
    bc = dataclasses.replace(CONFIG.backtest, option_bt_dynamic_iv=False, option_bt_iv=0.27)
    assert bt._realized_iv(synth_df, 100, bc, 252) == 0.27


def test_realized_iv_clamped_and_tracks_vol():
    # Düşük-vol vs yüksek-vol pencere: IV yüksek-volda daha büyük, sınırlar içinde
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    calm = pd.DataFrame({"close": 100 + np.linspace(0, 1, 60)}, index=idx)      # neredeyse düz
    np.random.seed(1)
    wild = pd.DataFrame({"close": 100 + np.cumsum(np.random.normal(0, 4, 60))}, index=idx)
    bc = CONFIG.backtest
    iv_calm = bt._realized_iv(calm, 59, bc, 252)
    iv_wild = bt._realized_iv(wild, 59, bc, 252)
    assert bc.option_bt_iv_floor <= iv_calm <= iv_wild <= bc.option_bt_iv_ceiling
    assert iv_wild > iv_calm


def test_realized_iv_ceiling_clamps_extreme_vol():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    np.random.seed(2)
    crazy = pd.DataFrame({"close": np.maximum(100 + np.cumsum(np.random.normal(0, 30, 40)), 5)}, index=idx)
    bc = CONFIG.backtest
    assert bt._realized_iv(crazy, 39, bc, 252) <= bc.option_bt_iv_ceiling


def test_option_synth_dynamic_vs_constant_iv_differ(synth_df):
    # Dinamik IV ve sabit IV farklı equity üretmeli (vega P&L kanalı aktif)
    r_dyn = bt.backtest_option_synth("AAA", interval="1d", df=synth_df, strategy_mode="event_breakout")
    import nihai.config as cfg
    orig = cfg.CONFIG
    try:
        bc_const = dataclasses.replace(CONFIG.backtest, option_bt_dynamic_iv=False)
        cfg.CONFIG = dataclasses.replace(CONFIG, backtest=bc_const)
        bt.CONFIG = cfg.CONFIG
        r_const = bt.backtest_option_synth("AAA", interval="1d", df=synth_df, strategy_mode="event_breakout")
    finally:
        cfg.CONFIG = orig
        bt.CONFIG = orig
    # En azından biri işlem yaptıysa sonuçlar ayrışmalı (ikisi de 0 işlemse atla)
    if r_dyn.trades_count or r_const.trades_count:
        assert r_dyn.final_equity != r_const.final_equity


def test_portfolio_backtest_shared_equity(synth_df):
    data = {"AAA": synth_df, "BBB": synth_df.copy()}
    r = bt.backtest_portfolio(["AAA", "BBB"], interval="1d", data=data, strategy_mode="ensemble")
    assert r.symbol.startswith("PORTFOLIO")
    assert len(r.equity_curve) > 1


def test_monte_carlo_reports_ruin(synth_df):
    r = bt.backtest_symbol("AAA", interval="1d", df=synth_df, mc_iters=200)
    if r.trades_count > 0:
        assert r.monte_carlo is not None
        assert 0.0 <= r.monte_carlo["ruin_probability"] <= 1.0


def test_empty_df_safe():
    r = bt.backtest_symbol("ZZZ", interval="1d", df=pd.DataFrame())
    assert r.trades_count == 0 and r.final_equity == r.initial_equity


def test_walk_forward_runs_and_picks_per_fold(synth_df):
    r = bt.walk_forward("AAA", interval="1d", df=synth_df, is_bars=120, oos_bars=40,
                        warmup_bars=60, candidates=["event_breakout", "mean_reversion", "donchian"],
                        metric="expectancy")
    assert r.symbol == "AAA"
    assert len(r.folds) >= 1
    # her fold seçilen aday listede ve OOS penceresi IS'ten SONRA (look-ahead-safe)
    for f in r.folds:
        assert f.chosen in r.candidates
        assert f.oos_start >= f.is_start + 120
    assert sum(r.strategy_picks.values()) == len(r.folds)
    assert r.final_equity > 0
    assert r.oos_max_drawdown_pct >= 0


def test_walk_forward_efficiency_finite(synth_df):
    r = bt.walk_forward("AAA", interval="1d", df=synth_df, is_bars=120, oos_bars=40,
                        warmup_bars=60, candidates=["event_breakout", "donchian"], metric="sharpe")
    assert np.isfinite(r.wf_efficiency)


def test_walk_forward_rejects_short_is_window(synth_df):
    with pytest.raises(ValueError):
        bt.walk_forward("AAA", interval="1d", df=synth_df, is_bars=50, warmup_bars=60)


# A4 — rejim çarpanı arama tz/normalize-güvenliği (sessiz no-op bug regresyonu) -----
def test_regime_mult_at_tz_safe_lookup():
    # Benchmark serisi GÜNLÜK tz-naive; sorgu tz-AWARE intraday timestamp (yfinance 1h gibi).
    # Karıştırma asof'u patlatmamalı; 1-gün-önceki değeri döndürmeli (1.0 fallback DEĞİL).
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    series = pd.Series([0.5] * 10, index=idx)
    ts_aware = pd.Timestamp("2024-01-05 14:30", tz="America/New_York")
    assert bt._regime_mult_at(series, ts_aware) == 0.5


def test_regime_mult_at_before_series_is_neutral():
    idx = pd.date_range("2024-03-01", periods=5, freq="D")
    series = pd.Series([0.5] * 5, index=idx)
    # seri başlamadan önce → neutral_mult (1.0 fallback değil)
    assert bt._regime_mult_at(series, pd.Timestamp("2024-01-01")) == CONFIG.regime.neutral_mult


def test_regime_mult_at_lagged_one_day():
    # 02..06 günleri farklı çarpan; 1-gün gecikme → sorgu gününün BİR ÖNCESİ kullanılır
    idx = pd.date_range("2024-01-02", periods=5, freq="D")
    series = pd.Series([0.5, 0.8, 1.0, 0.8, 0.5], index=idx)
    # 2024-01-04 sorgusu → 2024-01-03'ün değeri (0.8)
    assert bt._regime_mult_at(series, pd.Timestamp("2024-01-04")) == 0.8


# C3 — SONRAKİ-BAR KONVANSİYONU kilidi (golden-reference parite; DEĞİŞTİRME) -----------
def test_entry_bar_skipped_from_exit_checks():
    """Giriş barında (i+1) delinen stop KASITLI olarak kontrol edilmez: pozisyon o barda
    açılır ama ilk çıkış-kontrolü bir SONRAKİ barda yapılır. Sadece giriş barının low'u
    stop'u delen sentetik senaryoda pozisyon HAYATTA kalıp BT_END ile kapanmalı (STOP değil)."""
    n, warmup, entry_len = 70, 60, 63          # karar i=62 (len 63) → giriş bar 63 açılışı
    op = [100.0] * n
    high = [101.0] * n
    low = [99.0] * n
    low[entry_len] = 90.0                       # SADECE giriş barı (63) stop 95'i deler
    close = [100.0] * n
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({"open": op, "high": high, "low": low, "close": close,
                       "volume": [1e6] * n}, index=idx)

    def _fake_signal(symbol, df_slice, strategy_mode=None, shared=None, **kw):
        if len(df_slice) == entry_len:          # tek giriş: karar barı i=62
            return Signal(SignalAction.LONG, symbol, 100.0, 95.0, 200.0,
                          reasons=["synthetic"], quality_score=1.0, rr_ratio=3.0)
        return Signal(SignalAction.WAIT, symbol, 100.0, 0.0, 0.0)

    class _Cap:
        def __init__(self):
            self.opens: list = []
            self.closes: list = []

        def on_open(self, pos, df_slice, sig):
            self.opens.append(pos)

        def on_close(self, ce):
            self.closes.append(ce)

    cap = _Cap()
    import unittest.mock as _mock
    with _mock.patch.object(bt, "generate_signal", _fake_signal):
        r = bt.backtest_symbol("AAA", interval="1d", df=df, warmup_bars=warmup, recorder=cap)

    assert len(cap.opens) == 1                  # tam bir giriş
    assert r.trades_count == 1
    assert cap.closes[-1].exit_trigger == "BT_END"   # giriş-barı stop delinmesi ATLANDI
    # (aksi konvansiyonda bar 63'te STOP tetiklenir; parite kilidi budur)


# D6 — sağlayıcı önbelleği: selection başına tek kurulum (httpx client yeniden kurulmasın) --
def test_bt_provider_cached_per_selection(monkeypatch):
    calls: list[str] = []

    def _fake_get_provider(sel, **kw):
        calls.append(sel)
        return SimpleNamespace(name=sel, kw=kw)

    monkeypatch.setattr(bt, "get_provider", _fake_get_provider)
    bt._BT_PROVIDER_CACHE.clear()
    try:
        p1 = bt._bt_provider("hub")
        p2 = bt._bt_provider("hub")
        assert p1 is p2 and calls == ["hub"]          # aynı selection → tek get_provider
        p3 = bt._bt_provider("alpaca")
        assert p3 is not p1 and calls == ["hub", "alpaca"]
    finally:
        bt._BT_PROVIDER_CACHE.clear()
