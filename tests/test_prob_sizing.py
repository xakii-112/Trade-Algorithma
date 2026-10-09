"""prob_sizing.prob_to_size_mult — saf eşleme davranışı testleri.

Değişmezler: ramp monoton ve sınırlandırılmış; None nötr (cap); olasılık [0,1] dışına
taşsa bile çıktı [floor,cap] içinde; dejenere aralık çökmesin."""
import pytest

from nihai.risk.prob_sizing import prob_to_size_mult


def test_below_p_lo_returns_floor():
    assert prob_to_size_mult(0.30, p_lo=0.45, p_hi=0.60, floor=0.25) == 0.25
    assert prob_to_size_mult(0.45, p_lo=0.45, p_hi=0.60, floor=0.25) == 0.25  # sınır dahil


def test_above_p_hi_returns_cap():
    assert prob_to_size_mult(0.75, p_lo=0.45, p_hi=0.60, cap=1.0) == 1.0
    assert prob_to_size_mult(0.60, p_lo=0.45, p_hi=0.60, cap=1.0) == 1.0  # sınır dahil


def test_midpoint_is_linear():
    # p_lo=0.45, p_hi=0.60 ortası 0.525 → floor + 0.5*(cap-floor)
    m = prob_to_size_mult(0.525, p_lo=0.45, p_hi=0.60, floor=0.20, cap=1.0)
    assert m == pytest.approx(0.20 + 0.5 * 0.80)


def test_monotonic_nondecreasing():
    prev = -1.0
    for i in range(0, 101):
        p = i / 100.0
        m = prob_to_size_mult(p, p_lo=0.4, p_hi=0.7, floor=0.1)
        assert m >= prev - 1e-12
        assert 0.1 - 1e-9 <= m <= 1.0 + 1e-9
        prev = m


def test_none_is_neutral_cap():
    # Model pasif/hata → boyutlama nötr kalmalı (mevcut davranışı bozmaz)
    assert prob_to_size_mult(None) == 1.0
    assert prob_to_size_mult(None, cap=0.9) == 0.9


def test_out_of_range_prob_clipped():
    assert prob_to_size_mult(-0.5, p_lo=0.45, p_hi=0.60, floor=0.25) == 0.25
    assert prob_to_size_mult(1.5, p_lo=0.45, p_hi=0.60, cap=1.0) == 1.0


def test_degenerate_range_no_crash():
    # p_lo >= p_hi → çökme/sıfıra-bölme yok; deterministik eşik davranışı.
    # p<=p_lo önce kontrol edilir: 0.5<=0.6 → floor; 0.7>=p_hi(0.4) → cap.
    assert prob_to_size_mult(0.5, p_lo=0.6, p_hi=0.4, floor=0.3, cap=1.0) == 0.3
    assert prob_to_size_mult(0.7, p_lo=0.6, p_hi=0.4, floor=0.3, cap=1.0) == 1.0
    assert prob_to_size_mult(0.3, p_lo=0.6, p_hi=0.6, floor=0.3, cap=1.0) == 0.3


def test_custom_floor_respected():
    assert prob_to_size_mult(0.0, floor=0.4) == 0.4
    assert prob_to_size_mult(0.1, p_lo=0.45, floor=0.4) == 0.4
