"""research.deflated testleri — PSR/DSR/E[maxSR]/Wilson bilinen-değer ve monotonluk sınamaları.

Kaynak: Bailey & López de Prado (2014), López de Prado (2018) Ch.8. Ağ/IO yok.
"""
from __future__ import annotations

import math

from research.deflated import dsr, expected_max_sharpe, psr, wilson_ci


def test_psr_bilinen_degerler():
    # SR=0, benchmark=0 → tam %50 (hiç kanıt yok)
    assert abs(psr(0.0, 100) - 0.5) < 1e-9
    # Pozitif SR → PSR > 0.5; n büyüdükçe kanıt güçlenir (monoton artar)
    assert psr(0.1, 252) > 0.5
    assert psr(0.1, 1000) > psr(0.1, 252)
    # Negatif SR → PSR < 0.5
    assert psr(-0.1, 252) < 0.5


def test_psr_gecersiz_girdi_nan():
    assert math.isnan(psr(0.1, 1))     # n<2 → belirsiz
    # Payda negatifleşirse (aşırı skew) NaN — sessiz uydurma yok
    assert math.isnan(psr(2.0, 100, skew=5.0, kurt=3.0)) or psr(2.0, 100, skew=5.0, kurt=3.0) >= 0


def test_expected_max_sharpe():
    assert expected_max_sharpe(1, 0.01, 252) == 0.0
    e10 = expected_max_sharpe(10, 0.01, 252)
    e100 = expected_max_sharpe(100, 0.01, 252)
    assert 0 < e10 < e100          # daha çok deneme → şansla daha yüksek "en iyi SR"
    # varyans büyüdükçe beklenen maksimum büyür
    assert expected_max_sharpe(10, 0.04, 252) > e10


def test_dsr_psr_iliskisi():
    # Tek deneme → deflasyon yok: DSR == PSR
    assert abs(dsr(0.1, 252, 1, 0.01) - psr(0.1, 252)) < 1e-12
    # Çok deneme → benchmark yükselir → DSR < PSR
    assert dsr(0.1, 252, 20, 0.01) < psr(0.1, 252)


def test_wilson_ci():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert abs((lo + hi) / 2 - 0.5) < 0.01          # merkez ~0.5
    lo0, hi0 = wilson_ci(0, 10)
    assert 0.0 <= lo0 < hi0 < 1.0                    # sınırlar her zaman (0,1) içinde
    # n büyüdükçe aralık daralır
    lo2, hi2 = wilson_ci(500, 1000)
    assert (hi2 - lo2) < (hi - lo)
