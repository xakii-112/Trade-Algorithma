"""test_greeks_parity.py — py_vollib → vollib migrasyon parite testi (P0-1).

nihai/risk/options_analyzer.py ve nihai/execution/backtest.py artık `vollib` paketinden
import ediyor (py_vollib deprecated; API birebir aynı: delta/gamma/theta/vega(flag,S,K,t,r,sigma),
implied_volatility(price,S,K,t,r,flag)). Bu dosya GOLDEN SABİT kopyalamaz — bunun yerine
herhangi doğru bir Black-Scholes çözücüsünün sağlaması gereken, kütüphaneden BAĞIMSIZ
DEĞİŞMEZLERİ doğrular:

- IV round-trip: bilinen sigma → BS fiyatı → implied_volatility aynı sigma'yı ~1e-6
  toleransla geri vermeli (Madde 16 — IV Rank/Percentile'ın girdisi budur).
- Put-call parite: delta_c - delta_p == 1 (temettüsüz BS), gamma_c == gamma_p,
  vega_c == vega_p — Madde 27 (portföy Greek limitleri) bu özdeşliklere dayanır.
- ATM call theta < 0 makullük kontrolü (zaman değeri vade yaklaştıkça erir).
- options_analyzer.compute_greeks (public sarmalayıcı) gerçekçi senaryoda sonlu/beklenen
  aralıkta değer üretmeli — migrasyon sonrası HAS_VOLLIB=True dalının hâlâ çalıştığını kilitler.
"""
import math

import pytest
from vollib.black_scholes import black_scholes
from vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
from vollib.black_scholes.implied_volatility import implied_volatility

from nihai.risk.options_analyzer import HAS_VOLLIB, compute_greeks

S = 100.0    # dayanak fiyatı — tüm senaryolarda sabit referans
R = 0.03     # risksiz oran
IV_TOL = 1e-6

# (K, sigma) noktaları: ITM/ATM/OTM x düşük/yüksek IV. Call için K=85 ITM/K=115 OTM,
# put için tam tersi — aynı liste her iki flag ile de kullanılır.
_MONEYNESS_POINTS = [
    (85.0, 0.20),
    (100.0, 0.20),
    (115.0, 0.20),
    (85.0, 0.45),
    (100.0, 0.45),
    (115.0, 0.45),
]
_MATURITIES = [30 / 365, 180 / 365]   # kısa (30 gün) ve orta (180 gün) vade


def _iv_round_trip_cases():
    return [(T, K, sigma, flag)
            for T in _MATURITIES
            for K, sigma in _MONEYNESS_POINTS
            for flag in ("c", "p")]


def _parity_cases():
    return [(T, K, sigma) for T in _MATURITIES for K, sigma in _MONEYNESS_POINTS]


# ---------------------------------------------------------------------------
# IV round-trip — Madde 16
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T,K,sigma,flag", _iv_round_trip_cases())
def test_iv_round_trip_recovers_known_sigma(T, K, sigma, flag):
    """Bilinen sigma ile BS fiyatı üret → implied_volatility aynı sigma'yı geri versin.
    Golden sabit YOK: bu round-trip herhangi doğru bir çözücüde ~1e-6 altında tutar."""
    price = black_scholes(flag, S, K, T, R, sigma)
    recovered = implied_volatility(price, S, K, T, R, flag)
    assert recovered == pytest.approx(sigma, abs=IV_TOL)


# ---------------------------------------------------------------------------
# Put-call parite — Madde 27 (portföy Greek limitleri buna dayanır)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T,K,sigma", _parity_cases())
def test_put_call_parity_delta(T, K, sigma):
    """Temettüsüz Black-Scholes'ta delta_call - delta_put == 1 (kesin cebirsel özdeşlik)."""
    d_c = delta("c", S, K, T, R, sigma)
    d_p = delta("p", S, K, T, R, sigma)
    assert (d_c - d_p) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("T,K,sigma", _parity_cases())
def test_gamma_and_vega_equal_for_call_and_put(T, K, sigma):
    """Gamma ve vega BS formülünde option_type'a bağlı değildir → call/put için ÖZDEŞ
    olmalı (sıkı tolerans: yalnız kayan nokta artığı kadar fark beklenir)."""
    assert gamma("c", S, K, T, R, sigma) == pytest.approx(gamma("p", S, K, T, R, sigma), abs=1e-12)
    assert vega("c", S, K, T, R, sigma) == pytest.approx(vega("p", S, K, T, R, sigma), abs=1e-12)


# ---------------------------------------------------------------------------
# Theta makullük — ATM call
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T,sigma", [(T, s) for T in _MATURITIES for s in (0.20, 0.45)])
def test_atm_call_theta_is_negative(T, sigma):
    """ATM call için zaman değeri vade yaklaştıkça erir → theta < 0."""
    assert theta("c", S, S, T, R, sigma) < 0.0


# ---------------------------------------------------------------------------
# options_analyzer.compute_greeks — public sarmalayıcı, migrasyon sonrası entegrasyon
# ---------------------------------------------------------------------------
def test_compute_greeks_realistic_scenario_sigma_given():
    """test_greeks.py'deki kurulum desenine benzer gerçekçi ATM senaryo (sigma verilen
    yol). HAS_VOLLIB=True olmalı (venv'e vollib kurulu) — aksi halde bu testler kaba
    yaklaşıma düşer ve aşağıdaki sıkı parite iddiaları anlamsızlaşırdı."""
    assert HAS_VOLLIB is True

    call = compute_greeks(option_type="c", S=100.0, K=100.0, T=0.25, r=0.03, sigma=0.30)
    put = compute_greeks(option_type="p", S=100.0, K=100.0, T=0.25, r=0.03, sigma=0.30)
    assert call is not None and put is not None

    for g in (call, put):
        assert math.isfinite(float(g.delta))
        assert math.isfinite(float(g.gamma))
        assert math.isfinite(float(g.theta))
        assert math.isfinite(float(g.vega))
        assert math.isfinite(float(g.iv))

    assert 0.0 < call.delta < 1.0
    assert -1.0 < put.delta < 0.0
    assert call.gamma > 0.0
    assert call.theta < 0.0
    assert call.vega > 0.0
    # public sarmalayıcı üzerinden de put-call parite tutmalı
    assert (call.delta - put.delta) == pytest.approx(1.0, abs=1e-6)
    assert call.gamma == pytest.approx(put.gamma, abs=1e-9)
    assert call.vega == pytest.approx(put.vega, abs=1e-9)


def test_compute_greeks_market_price_solves_iv():
    """market_price verilen yol: implied_volatility içeriden çözülür (estimate_atm_iv'nin
    kullandığı dal — Madde 16). Gerçekçi ATM prim → makul IV + sonlu Greekler."""
    g = compute_greeks(option_type="c", S=100.0, K=100.0, T=0.25, r=0.03, market_price=6.5)
    assert g is not None
    assert math.isfinite(float(g.iv)) and 0.0 < g.iv < 5.0
    assert math.isfinite(float(g.delta)) and 0.0 < g.delta < 1.0
