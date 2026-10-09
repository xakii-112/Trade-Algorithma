"""event_calendar guard fonksiyonları (Madde 17/25/26) testleri.

Bu guard'lar bilanço/temettü/split yakınında yeni giriş veya long-opsiyonu BLOKE eder;
tampon penceresi (0..buffer, dahil) yanlışsa ya gerçek olayda girer ya gereksiz bloke
eder. get_company_events monkeypatch'lenir → ağ yok, tarihler 'bugün'e göre relatif."""
from datetime import date, timedelta

import nihai.data.event_calendar as ec
from nihai.data.event_calendar import (
    CompanyEvents,
    corporate_action_guard,
    event_guard,
    iv_crush_guard_for_option,
)


def _patch_events(monkeypatch, **kw):
    ev = CompanyEvents(symbol="TEST", **kw)
    monkeypatch.setattr(ec, "get_company_events", lambda *a, **k: ev)


def test_event_guard_clear_when_no_events(monkeypatch):
    _patch_events(monkeypatch)
    assert event_guard("TEST") == (False, "CLEAR")


def test_event_guard_blocks_imminent_earnings(monkeypatch):
    _patch_events(monkeypatch, next_earnings=date.today() + timedelta(days=1))
    blocked, reason = event_guard("TEST", earnings_buffer_days=2)
    assert blocked is True and "EARNINGS_IN_1_DAYS" in reason


def test_event_guard_earnings_outside_buffer_clear(monkeypatch):
    _patch_events(monkeypatch, next_earnings=date.today() + timedelta(days=5))
    assert event_guard("TEST", earnings_buffer_days=2)[0] is False


def test_event_guard_buffer_boundary_inclusive(monkeypatch):
    # delta == buffer → hâlâ bloke (<=)
    _patch_events(monkeypatch, next_earnings=date.today() + timedelta(days=2))
    assert event_guard("TEST", earnings_buffer_days=2)[0] is True


def test_event_guard_past_earnings_not_blocked(monkeypatch):
    # Negatif delta (geçmiş) → bloke etme (0 <= delta koşulu)
    _patch_events(monkeypatch, next_earnings=date.today() - timedelta(days=1))
    assert event_guard("TEST")[0] is False


def test_event_guard_blocks_ex_dividend(monkeypatch):
    _patch_events(monkeypatch, next_ex_dividend=date.today() + timedelta(days=1))
    blocked, reason = event_guard("TEST", ex_div_buffer_days=1)
    assert blocked is True and "EX_DIV_IN_1_DAYS" in reason


def test_corporate_action_guard_recent_split_blocks(monkeypatch):
    _patch_events(monkeypatch, last_split=date.today() - timedelta(days=2))
    blocked, reason = corporate_action_guard("TEST", lookback_days=5)
    assert blocked is True and "RECENT_SPLIT_2_DAYS_AGO" in reason


def test_corporate_action_guard_old_split_clear(monkeypatch):
    _patch_events(monkeypatch, last_split=date.today() - timedelta(days=30))
    assert corporate_action_guard("TEST", lookback_days=5) == (False, "CLEAR")


def test_iv_crush_guard_blocks_within_proximity(monkeypatch):
    _patch_events(monkeypatch, next_earnings=date.today() + timedelta(days=3))
    blocked, reason = iv_crush_guard_for_option("TEST", earnings_proximity_days=7)
    assert blocked is True and "IV_CRUSH_RISK_EARNINGS_IN_3_DAYS" in reason


def test_iv_crush_guard_clear_when_far(monkeypatch):
    _patch_events(monkeypatch, next_earnings=date.today() + timedelta(days=20))
    assert iv_crush_guard_for_option("TEST", earnings_proximity_days=7)[0] is False
