"""indicators_ext osilatörleri (demarker/dpo/ppo/tsi) yön/sınır testleri.

Bunlar konsensüse oy veren osilatörler; sınır ([0,1] veya [-100,100]) ve monoton-seri
yönü yanlışsa yanlış oy üretirler. Sabit seri → trend osilatörü 0; monoton artış/azalış
uçlara gider."""
import numpy as np
import pandas as pd
import pytest

from nihai.signals.indicators_ext import demarker, dpo, ppo, tsi


def _ohlc_from_close(close):
    s = pd.Series(close, dtype=float)
    return pd.DataFrame({"open": s, "high": s + 0.5, "low": s - 0.5, "close": s})


def test_demarker_bounded_unit_interval():
    df = _ohlc_from_close(np.random.default_rng(0).normal(100, 2, 80).cumsum())
    val = demarker(df, 14).dropna()
    assert ((val >= 0.0) & (val <= 1.0)).all()


def test_demarker_one_on_monotone_up():
    df = _ohlc_from_close(np.arange(1.0, 40.0))
    assert demarker(df, 14).dropna().iloc[-1] == pytest.approx(1.0)


def test_demarker_zero_on_monotone_down():
    df = _ohlc_from_close(np.arange(40.0, 1.0, -1.0))
    assert demarker(df, 14).dropna().iloc[-1] == pytest.approx(0.0)


def test_dpo_zero_on_constant_series():
    c = pd.Series([5.0] * 60)
    assert dpo(c, 20).dropna().iloc[-1] == pytest.approx(0.0)


def test_ppo_positive_in_uptrend_negative_in_downtrend():
    up = pd.Series(np.arange(1.0, 80.0))
    down = pd.Series(np.arange(80.0, 1.0, -1.0))
    assert ppo(up, 12, 26).iloc[-1] > 0
    assert ppo(down, 12, 26).iloc[-1] < 0


def test_ppo_zero_on_constant():
    c = pd.Series([10.0] * 60)
    assert ppo(c, 12, 26).iloc[-1] == pytest.approx(0.0)


def test_tsi_bounded_and_signed_by_trend():
    up = pd.Series(np.arange(1.0, 120.0))
    down = pd.Series(np.arange(120.0, 1.0, -1.0))
    u = tsi(up).dropna().iloc[-1]
    d = tsi(down).dropna().iloc[-1]
    assert -100.0 <= u <= 100.0 and -100.0 <= d <= 100.0
    assert u == pytest.approx(100.0)     # tüm momentum pozitif
    assert d == pytest.approx(-100.0)
