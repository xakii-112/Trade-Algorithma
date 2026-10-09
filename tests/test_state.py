"""State testleri — Madde 7 (cooldown), 20 (kill-switch), 23 (T+1), 53 (wash-sale)."""
from datetime import date, datetime, timedelta

from nihai.config import CONFIG
from nihai.execution.state import StateStore
from nihai.risk.risk_manager import PortfolioState


def make_store(tmp_path):
    s = StateStore(path=tmp_path / "state.json")
    s.load_or_init(25_000)
    return s


def test_open_position_survives_save_load(tmp_path):
    # BUG: VirtualPosition(**pos) tip dönüşümü yapmıyordu → entry_time str, greeks dict
    from nihai.risk.options_analyzer import GreekSet
    from nihai.risk.risk_manager import VirtualPosition
    s = make_store(tmp_path)
    s.portfolio.open_positions.append(VirtualPosition(
        symbol="AAPL", asset_type="option", side="LONG", quantity=2,
        entry_price=3.5, entry_time=datetime(2026, 6, 1, 10, 30), stop=3.0, target=5.0,
        option_type="c", strike=200, expiry="2026-06-19",
        greeks=GreekSet(delta=0.5, gamma=0.02, theta=-0.05, vega=0.2, iv=0.3)))
    s.save()
    s2 = StateStore(path=tmp_path / "state.json")
    s2.load_or_init(25_000)
    pos = s2.portfolio.open_positions[0]
    assert isinstance(pos.entry_time, datetime)       # str DEĞİL
    assert isinstance(pos.greeks, GreekSet)           # dict DEĞİL
    assert pos.greeks.delta == 0.5
    assert pos.entry_time == datetime(2026, 6, 1, 10, 30)   # gün-içi saat korundu


def test_reloaded_position_closes_without_crash(tmp_path):
    # BUG: taşınan pozisyon kapanınca post-mortem 'exit_time - entry_time' TypeError veriyordu
    from nihai.risk.position_manager import PositionManager, SlippageModel
    from nihai.risk.risk_manager import VirtualPosition
    s = make_store(tmp_path)
    s.portfolio.open_positions.append(VirtualPosition(
        symbol="X", asset_type="stock", side="LONG", quantity=10,
        entry_price=100, entry_time=datetime(2026, 6, 1, 10, 30), stop=95, target=110))
    s.save()
    s2 = StateStore(path=tmp_path / "state.json")
    s2.load_or_init(25_000)
    pmgr = PositionManager(s2.portfolio, SlippageModel())
    assert pmgr.rehydrate_trackers() == 1
    pos = s2.portfolio.open_positions[0]
    ce = pmgr.force_close(pos, 95.0, "TEST_CLOSE", datetime(2026, 6, 1, 14, 0))   # crash ETMEMELİ
    assert ce.exit_trigger == "TEST_CLOSE"
    assert len(s2.portfolio.open_positions) == 0


def test_wash_sale_window(tmp_path):
    s = make_store(tmp_path)
    now = datetime(2026, 6, 1, 10, 0)
    s.register_close_for_wash_sale("AAPL", -120.0, now)
    hit, why = s.wash_sale_check("AAPL", now)
    assert hit and "WASH_SALE" in why
    # 30 günden sonra temizlenmeli
    later = now + timedelta(days=CONFIG.regulation.wash_sale_days + 1)
    assert s.wash_sale_check("AAPL", later) == (False, "CLEAR")


def test_wash_sale_only_losses(tmp_path):
    s = make_store(tmp_path)
    now = datetime(2026, 6, 1)
    s.register_close_for_wash_sale("MSFT", +50.0, now)  # kâr → kayda girmez
    assert s.wash_sale_check("MSFT", now) == (False, "CLEAR")


def test_kill_switch_on_negative_expectancy(tmp_path, monkeypatch):
    # Kill-switch env-kapılı (varsayılan kapalı); test için aç.
    import nihai.execution.state as state_mod
    monkeypatch.setattr(state_mod, "ENABLE_KILL_SWITCH", True)
    s = make_store(tmp_path)
    for _ in range(30):
        s.record_trade_pnl(-10.0)
    assert s.kill_switch_active


def test_kill_switch_auto_resets_on_recovery(tmp_path, monkeypatch):
    # A3 — rolling expectancy toparlayınca kilit kendiliğinden kalkar.
    import nihai.execution.state as state_mod
    monkeypatch.setattr(state_mod, "ENABLE_KILL_SWITCH", True)
    monkeypatch.setattr(state_mod, "KILL_SWITCH_RESET_EXPECTANCY", 0.0)
    s = make_store(tmp_path)
    for _ in range(30):
        s.record_trade_pnl(-10.0)
    assert s.kill_switch_active
    # Pencere maxlen=50; güçlü kazançlar rolling ortalamayı ≥0'a çeker → reset
    for _ in range(50):
        s.record_trade_pnl(+100.0)
    assert not s.kill_switch_active


def test_kill_switch_stays_off_when_disabled(tmp_path):
    # Varsayılan (ENABLE_KILL_SWITCH kapalı) → 30 zarar bile kilitlemez
    s = make_store(tmp_path)
    for _ in range(30):
        s.record_trade_pnl(-10.0)
    assert not s.kill_switch_active


def test_cooldown_uses_bar_minutes(tmp_path):
    s = make_store(tmp_path)
    now = datetime(2026, 6, 1, 10, 0, 0)
    s.set_cooldown("AAA", now)
    assert s.cooldown_active("AAA", now + timedelta(minutes=1))      # hâlâ aktif
    # 2 bar * 60dk = 120dk sonra bitmeli
    assert not s.cooldown_active("AAA", now + timedelta(minutes=200))


def test_t1_settlement(tmp_path):
    s = make_store(tmp_path)
    sell = date(2026, 6, 1)
    s.add_unsettled_proceeds(5_000, sell)
    s.settle_matured(sell)  # aynı gün → henüz takas olmadı
    assert s.portfolio.unsettled_cash == 5_000
    s.settle_matured(sell + timedelta(days=CONFIG.regulation.settlement_days + 1))
    assert s.portfolio.unsettled_cash == 0


def test_persistence_roundtrip(tmp_path):
    s = make_store(tmp_path)
    s.recent_signals = ["a", "b"]
    s.register_close_for_wash_sale("AAPL", -10, datetime(2026, 6, 1))
    s.save()
    s2 = StateStore(path=tmp_path / "state.json")
    s2.load_or_init(25_000)
    assert s2.recent_signals == ["a", "b"]
    assert any(r["symbol"] == "AAPL" for r in s2.loss_history)
