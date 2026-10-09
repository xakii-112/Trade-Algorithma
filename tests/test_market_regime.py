"""A4 — piyasa rejimi → risk çarpanı testleri (offline, sentetik benchmark)."""
import pandas as pd

from nihai.config import CONFIG
from nihai.risk.risk_manager import PortfolioState, RiskManager
from nihai.signals.market_regime import compute_regime, regime_multiplier_series


def _df(closes):
    return pd.DataFrame({"close": closes})


def _set_regime(**kw):
    old = {k: getattr(CONFIG.regime, k) for k in kw}
    for k, v in kw.items():
        object.__setattr__(CONFIG.regime, k, v)
    return old


def _restore_regime(old):
    for k, v in old.items():
        object.__setattr__(CONFIG.regime, k, v)


def test_uptrend_is_bullish():
    closes = [100 + i for i in range(260)]   # düzgün yükseliş
    r = compute_regime(_df(closes))
    assert r.regime == "BULLISH"
    assert r.multiplier == CONFIG.regime.bullish_mult


def test_downtrend_is_defensive():
    closes = [400 - i for i in range(260)]   # düzgün düşüş → fiyat < SMA200
    r = compute_regime(_df(closes))
    assert r.regime == "DEFENSIVE"
    assert r.multiplier == CONFIG.regime.defensive_mult


def test_insufficient_data_is_neutral():
    r = compute_regime(_df([100, 101, 102]))
    assert r.regime == "NEUTRAL"


def test_none_df_is_neutral():
    assert compute_regime(None).regime == "NEUTRAL"


def test_high_vix_forces_defensive():
    closes = [100 + i for i in range(260)]   # aksi halde bullish
    r = compute_regime(_df(closes), vix=35.0)
    assert r.regime == "DEFENSIVE"


def test_regime_multiplier_stacks_in_risk_pct():
    old = _set_regime(enable_regime_risk=True)
    try:
        ps = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
        mgr = RiskManager(ps)
        mgr.regime_multiplier = 0.5
        assert abs(mgr._current_risk_pct(0.02) - 0.01) < 1e-9   # 0.02 × 0.5
    finally:
        _restore_regime(old)


def test_regime_multiplier_ignored_when_disabled():
    # enable_regime_risk varsayılan kapalı → çarpan yok sayılır
    ps = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    mgr = RiskManager(ps)
    mgr.regime_multiplier = 0.5
    assert abs(mgr._current_risk_pct(0.02) - 0.02) < 1e-9


# A4 backtest serisi (vektörel, look-ahead-güvenli) ------------------------------
def test_regime_series_matches_scalar_uptrend():
    closes = [100 + i * 0.5 for i in range(260)]
    s = regime_multiplier_series(_df(closes))
    assert len(s) == 260
    assert s.iloc[-1] == CONFIG.regime.bullish_mult          # vektörel == compute_regime
    assert compute_regime(_df(closes)).multiplier == s.iloc[-1]


def test_regime_series_downtrend_defensive():
    closes = [400 - i * 0.5 for i in range(260)]
    s = regime_multiplier_series(_df(closes))
    assert s.iloc[-1] == CONFIG.regime.defensive_mult


def test_regime_series_neutral_before_enough_data():
    closes = [100 + i * 0.5 for i in range(260)]
    s = regime_multiplier_series(_df(closes))
    assert s.iloc[0] == CONFIG.regime.neutral_mult           # SMA200 henüz NaN → nötr


def test_regime_series_empty_on_none():
    assert regime_multiplier_series(None).empty
