"""PortfolioGreeks toplama + cap kontrolü (Madde 27) — ölçek-bug regresyon guard'ı.

Dokümante edilmiş bug: py_vollib theta/vega de hisse-başı döner; pozisyon değeri için
×(contracts*multiplier) gerekir. Eski kod theta/vega'yı ölçeklemiyordu → cap'ler asla
tetiklenmiyordu. Bu testler DÖRT Greek'in de aynı ölçekle toplandığını kilitler.
Yerel Caps namespace ile CONFIG değerlerinden bağımsız test edilir."""
from dataclasses import dataclass

from nihai.risk.options_analyzer import GreekSet, PortfolioGreeks


@dataclass
class _Caps:
    max_net_delta: float = 200.0
    max_net_theta: float = -50.0     # daha negatif = ihlal
    max_net_vega: float = 300.0
    max_net_gamma: float = 50.0


def _greek(delta=0.5, gamma=0.02, theta=-0.10, vega=0.10):
    return GreekSet(delta=delta, gamma=gamma, theta=theta, vega=vega, iv=0.3)


def test_all_four_greeks_scale_by_contracts_times_multiplier():
    agg = PortfolioGreeks()
    agg.add_position(_greek(), contracts=10, multiplier=100)
    scale = 10 * 100
    assert agg.delta == 0.5 * scale
    assert agg.gamma == 0.02 * scale
    assert agg.theta == -0.10 * scale   # ÖLÇEK-BUG guard: ×100 uygulanmalı
    assert agg.vega == 0.10 * scale     # ÖLÇEK-BUG guard


def test_positions_accumulate():
    agg = PortfolioGreeks()
    agg.add_position(_greek(delta=0.5), contracts=1, multiplier=100)
    agg.add_position(_greek(delta=0.3), contracts=1, multiplier=100)
    assert agg.delta == (0.5 + 0.3) * 100


def test_within_caps_all_clear():
    agg = PortfolioGreeks()
    agg.add_position(_greek(), contracts=1, multiplier=100)  # delta 50, theta -10
    ok, breaches = agg.within_caps(_Caps())
    assert ok is True and breaches == []


def test_delta_cap_breach_reported():
    agg = PortfolioGreeks()
    agg.add_position(_greek(delta=0.5), contracts=10, multiplier=100)  # delta 500 > 200
    ok, breaches = agg.within_caps(_Caps())
    assert ok is False and any("NET_DELTA" in b for b in breaches)


def test_theta_cap_breach_when_too_negative():
    agg = PortfolioGreeks()
    agg.add_position(_greek(theta=-0.10), contracts=10, multiplier=100)  # theta -100 < -50
    ok, breaches = agg.within_caps(_Caps())
    assert ok is False and any("NET_THETA" in b for b in breaches)


def test_gamma_cap_breach_reported():
    agg = PortfolioGreeks()
    agg.add_position(_greek(gamma=0.10), contracts=10, multiplier=100)  # gamma 100 > 50
    ok, breaches = agg.within_caps(_Caps())
    assert ok is False and any("NET_GAMMA" in b for b in breaches)


def test_vega_cap_breach_reported():
    agg = PortfolioGreeks()
    agg.add_position(_greek(vega=0.50), contracts=10, multiplier=100)  # vega 500 > 300
    ok, breaches = agg.within_caps(_Caps())
    assert ok is False and any("NET_VEGA" in b for b in breaches)
