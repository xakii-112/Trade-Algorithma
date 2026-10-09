"""indicators_ext.py hareketli-ortalama ailesi + osilatör testleri.

Bu göstergeler konsensüse oy veriyor; formül hatası yön oyununu bozar. İki sınıf test:
(a) bilinen girdide KESİN değer (sma/wma), (b) sabit/monoton seride değişmez
(tüm MA'lar sabite yakınsar; cmo monoton-artışta +100). Repaint/look-ahead yok —
hepsi nedensel rolling."""
import numpy as np
import pandas as pd
import pytest

from nihai.signals.indicators_ext import (
    cmo,
    dema,
    hma,
    sma,
    smma,
    tema,
    tma,
    wma,
)


def test_sma_exact_on_ramp():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, 3)
    assert np.isnan(out.iloc[1])           # warmup
    assert out.iloc[2] == 2.0              # (1+2+3)/3
    assert out.iloc[4] == 4.0              # (3+4+5)/3


def test_wma_weights_recent_more():
    s = pd.Series([1.0, 2.0, 3.0])
    # ağırlıklar (1,2,3): (1*1 + 2*2 + 3*3)/6 = 14/6
    assert wma(s, 3).iloc[2] == pytest.approx((1 * 1 + 2 * 2 + 3 * 3) / 6)
    # artan seride WMA, eşit-ağırlıklı SMA'dan büyük olmalı (son değere yakın)
    assert wma(s, 3).iloc[2] > sma(s, 3).iloc[2]


def test_tma_double_smoothing_warms_up_slower():
    s = pd.Series(np.arange(1.0, 11.0))
    # tma = sma(sma) → iki kat ısınma; geçerli değer sayısı tek sma'dan az
    assert tma(s, 3).notna().sum() < sma(s, 3).notna().sum()


def _const(n=60, v=7.0):
    return pd.Series([v] * n, dtype=float)


def test_mas_converge_to_constant_series():
    c = _const()
    for fn in (sma, wma, smma, dema, tema, tma, hma):
        out = fn(c, 10)
        assert out.dropna().iloc[-1] == pytest.approx(7.0), fn.__name__


def test_dema_tema_more_responsive_than_smma():
    # Adım fonksiyonu: DEMA/TEMA, SMMA'dan daha hızlı yeni seviyeye yaklaşır
    s = pd.Series([0.0] * 30 + [10.0] * 30)
    assert dema(s, 10).iloc[-1] >= smma(s, 10).iloc[-1]
    assert tema(s, 10).iloc[-1] >= smma(s, 10).iloc[-1]


def test_cmo_bounded_and_plus_100_on_monotone_up():
    up = pd.Series(np.arange(1.0, 40.0))
    val = cmo(up, 14).dropna().iloc[-1]
    assert -100.0 <= val <= 100.0
    assert val == pytest.approx(100.0)   # tüm değişimler pozitif → +100


def test_cmo_minus_100_on_monotone_down():
    down = pd.Series(np.arange(40.0, 1.0, -1.0))
    val = cmo(down, 14).dropna().iloc[-1]
    assert val == pytest.approx(-100.0)
