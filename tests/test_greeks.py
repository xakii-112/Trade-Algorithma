"""compute_greeks + assignment_risk_short_call testleri (opsiyon matematiği çekirdeği).

Greekler pozisyon boyutu/cap kontrolünü besliyor; işaret hatası (call delta negatif)
yön/hedge'i ters çevirir. Testler py_vollib varlığından BAĞIMSIZ tutuldu — yalnız her
iki dalda da geçerli değişmezler (işaret, sınır, iv geçişi) doğrulanır."""
from nihai.risk.options_analyzer import assignment_risk_short_call, compute_greeks


def test_invalid_inputs_return_none():
    assert compute_greeks(option_type="c", S=0.0, K=100, T=0.1, sigma=0.3) is None
    assert compute_greeks(option_type="c", S=100, K=0.0, T=0.1, sigma=0.3) is None
    assert compute_greeks(option_type="c", S=100, K=100, T=0.0, sigma=0.3) is None
    assert compute_greeks(option_type="c", S=100, K=100, T=-1, sigma=0.3) is None


def test_call_delta_positive_put_delta_negative():
    call = compute_greeks(option_type="c", S=100, K=100, T=0.25, sigma=0.3)
    put = compute_greeks(option_type="p", S=100, K=100, T=0.25, sigma=0.3)
    assert call is not None and put is not None
    assert call.delta > 0
    assert put.delta < 0


def test_atm_call_delta_in_unit_interval():
    g = compute_greeks(option_type="c", S=100, K=100, T=0.25, sigma=0.3)
    assert 0.0 < g.delta < 1.0


def test_sigma_propagates_to_iv():
    g = compute_greeks(option_type="c", S=100, K=100, T=0.25, sigma=0.42)
    assert g.iv == 0.42


# ---------------- assignment_risk_short_call ----------------

def test_assignment_none_exdiv_is_safe():
    assert assignment_risk_short_call(strike=100, underlying=110, days_to_ex_div=None) is False


def test_assignment_itm_near_exdiv_is_risky():
    # underlying > strike (ITM call) ve ex-div 0-1 gün → erken atama riski
    assert assignment_risk_short_call(strike=100, underlying=110, days_to_ex_div=0) is True
    assert assignment_risk_short_call(strike=100, underlying=110, days_to_ex_div=1) is True


def test_assignment_otm_not_risky():
    assert assignment_risk_short_call(strike=120, underlying=110, days_to_ex_div=0) is False


def test_assignment_exdiv_too_far_not_risky():
    assert assignment_risk_short_call(strike=100, underlying=110, days_to_ex_div=3) is False
