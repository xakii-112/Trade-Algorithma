"""Halt/LULD (Madde 24) ve SSR (Madde 30) testleri."""
import pandas as pd

from nihai.data.market_structure import halt_or_luld_guard, ssr_active, ssr_short_guard


def test_ssr_active_threshold():
    assert ssr_active(100, 89)        # %11 düşüş → aktif
    assert not ssr_active(100, 95)    # %5 → değil
    assert not ssr_active(0, 50)      # geçersiz prev


def test_ssr_short_guard_message():
    on, why = ssr_short_guard(100, 88)
    assert on and "SSR_ACTIVE" in why
    assert ssr_short_guard(100, 99) == (False, "CLEAR")


def test_halt_frozen_detected():
    df = pd.DataFrame({"close": [10, 10, 10], "volume": [0, 0, 0]})
    on, why = halt_or_luld_guard(df)
    assert on and "HALT" in why


def test_luld_anomaly_detected():
    df = pd.DataFrame({"close": [10, 10, 12], "volume": [5, 5, 5]})
    on, why = halt_or_luld_guard(df)
    assert on and "LULD" in why


def test_normal_bars_clear():
    df = pd.DataFrame({"close": [10, 10.1, 10.2], "volume": [5, 5, 5]})
    assert halt_or_luld_guard(df) == (False, "CLEAR")
