"""Risk yöneticisi testleri — Madde 2, 9, 10, 12, 14, 27 (offline)."""
import dataclasses
from datetime import date

import pytest

import nihai.risk.risk_manager as rm
from nihai.config import CONFIG
from nihai.risk.options_analyzer import GreekSet
from nihai.risk.risk_manager import PortfolioState, RiskManager, VirtualPosition


def _patch_portfolio(monkeypatch, **changes):
    """Frozen CONFIG'i replace ile değiştirip risk_manager modül global'ini patch'le."""
    new_caps = dataclasses.replace(CONFIG.portfolio, **changes)
    monkeypatch.setattr(rm, "CONFIG", dataclasses.replace(rm.CONFIG, portfolio=new_caps))


def _patch_risk(monkeypatch, **changes):
    """Frozen CONFIG'i replace ile değiştirip risk_manager modül global'indeki risk ayarlarını patch'le."""
    new_risk = dataclasses.replace(CONFIG.risk, **changes)
    monkeypatch.setattr(rm, "CONFIG", dataclasses.replace(rm.CONFIG, risk=new_risk))


def fresh(equity=200_000):
    return PortfolioState(equity=equity, ath_equity=equity, cash_balance=equity)


def add_stock(ps, sym, sector, side="LONG", entry=100, stop=97.75, qty=50):
    ps.open_positions.append(VirtualPosition(
        symbol=sym, asset_type="stock", side=side, quantity=qty,
        entry_price=entry, entry_time=date.today(), stop=stop, sector=sector,
        last_mark=entry))


def test_risk_dollars_short_not_zero():
    # BUG: SHORT'ta stop>entry → eski max(entry-stop,0)=0 → sıfır risk gösteriyordu
    short = VirtualPosition(symbol="X", asset_type="stock", side="SHORT", quantity=10,
                            entry_price=100, entry_time=date.today(), stop=105)
    assert short.risk_dollars == 50.0       # |100-105|*10, eskiden 0.0
    long = VirtualPosition(symbol="Y", asset_type="stock", side="LONG", quantity=10,
                           entry_price=100, entry_time=date.today(), stop=95)
    assert long.risk_dollars == 50.0        # LONG davranışı korundu


def test_roll_session_resets_daily_counters():
    ps = PortfolioState(equity=25_000, ath_equity=25_000, cash_balance=25_000)
    ps.daily_pnl = -500.0
    ps.consecutive_stops = 5
    ps.last_session_date = date(2026, 6, 2)
    assert ps.roll_session_if_new_day(date(2026, 6, 2)) is False   # aynı gün → reset yok
    assert ps.consecutive_stops == 5
    assert ps.roll_session_if_new_day(date(2026, 6, 3)) is True     # yeni gün → reset
    assert ps.daily_pnl == 0.0 and ps.consecutive_stops == 0
    assert ps.last_session_date == date(2026, 6, 3)


def test_consecutive_stops_deadlock_broken_by_new_day(monkeypatch):
    # BUG deadlock: cooldown aktifken giriş yok → kapanış yok → asla sıfırlanmaz.
    # Yeni seans günü reset'i bu kilidi kırar.
    _patch_risk(monkeypatch, consecutive_stops_cooldown=3)
    ps = PortfolioState(equity=25_000, ath_equity=25_000, cash_balance=25_000)
    ps.consecutive_stops = rm.CONFIG.risk.consecutive_stops_cooldown + 2
    ps.last_session_date = date(2026, 6, 2)
    mgr = RiskManager(ps)
    assert mgr.consecutive_stops_cooldown()           # kilitli — tüm sinyaller reddedilir
    ps.roll_session_if_new_day(date(2026, 6, 3))
    assert not mgr.consecutive_stops_cooldown()        # yeni gün → kilit açıldı


def test_gross_risk_cap_counts_shorts():
    # Short pozisyon artık brüt riske katkı veriyor (eskiden 0 olduğu için sayılmıyordu)
    ps = fresh(10_000)
    ps.open_positions.append(VirtualPosition(
        symbol="S", asset_type="stock", side="SHORT", quantity=100,
        entry_price=100, entry_time=date.today(), stop=130, sector="Z"))  # risk=30*100=3000
    gross = sum(p.risk_dollars for p in ps.open_positions)
    assert gross == 3000.0


def test_per_symbol_cluster_cap():
    ps = fresh()
    mgr = RiskManager(ps)
    cap = CONFIG.safety.max_signals_per_symbol_open
    for _ in range(cap):
        rc = mgr.size_stock(entry=10, stop=9, sector="Z", symbol="AAA", side="LONG")
        assert rc.approved
        add_stock(ps, "AAA", "Z", entry=10, stop=9, qty=rc.sizing["shares"])
    rc = mgr.size_stock(entry=10, stop=9, sector="Z", symbol="AAA", side="LONG")
    assert not rc.approved and "SYMBOL_CLUSTER_CAP" in rc.reason


def _net_signed(ps):
    return sum(p.signed_notional for p in ps.open_positions)


def test_net_directional_cap_downsizes_not_rejects(monkeypatch):
    # Cap-aware boyutlama (varsayılan): ikinci long net yönseli AŞMAK yerine kalan
    # headroom'a sığacak şekilde KÜÇÜLÜR — reddedilmez.
    # Bağlayıcı cap pinle (varsayılan 2.0 gevşek → senaryo tetiklenmez; test tunable
    # default'a bağlı kalmamalı). 1.0 ile: ilk poz sığar, ikinci downsize olur.
    _patch_portfolio(monkeypatch, max_net_directional_pct=1.0)
    ps = PortfolioState(equity=25_000, ath_equity=25_000, cash_balance=25_000)
    mgr = RiskManager(ps)
    rc1 = mgr.size_stock(entry=100, stop=97.75, sector="S1", symbol="A", side="LONG")
    assert rc1.approved and not rc1.sizing["downsized"]
    add_stock(ps, "A", "S1", "LONG", 100, 97.75, rc1.sizing["shares"])

    rc2 = mgr.size_stock(entry=100, stop=97.75, sector="S2", symbol="B", side="LONG")
    assert rc2.approved and rc2.sizing["downsized"]
    assert rc2.sizing["shares"] < rc2.sizing["target_shares"]
    add_stock(ps, "B", "S2", "LONG", 100, 97.75, rc2.sizing["shares"])
    # net yönsel tavanı ASLA aşılmadı (sığdırıldı) — uygulanan (patch'li) cap'e göre
    cap = ps.equity * rm.CONFIG.portfolio.max_net_directional_pct
    assert abs(_net_signed(ps)) <= cap + 1e-6

    # short tarafta bol headroom → tam boyut açılır
    rc3 = mgr.size_stock(entry=100, stop=102.25, sector="S3", symbol="C", side="SHORT")
    assert rc3.approved and not rc3.sizing["downsized"]


def test_cap_aware_disabled_falls_back_to_reject(monkeypatch):
    # cap_aware_sizing=False → eski davranış: küçültme yerine reddet.
    # Bağlayıcı cap (1.0) pinle ki ikinci long gerçekten net-yönseli aşsın.
    _patch_portfolio(monkeypatch, cap_aware_sizing=False, max_net_directional_pct=1.0)
    ps = PortfolioState(equity=25_000, ath_equity=25_000, cash_balance=25_000)
    mgr = RiskManager(ps)
    rc1 = mgr.size_stock(entry=100, stop=97.75, sector="S1", symbol="A", side="LONG")
    assert rc1.approved
    add_stock(ps, "A", "S1", "LONG", 100, 97.75, rc1.sizing["shares"])
    rc2 = mgr.size_stock(entry=100, stop=97.75, sector="S2", symbol="B", side="LONG")
    assert not rc2.approved and "NET_DIRECTIONAL" in rc2.reason


def test_high_priced_symbol_small_account_still_trades():
    # Orijinal hata senaryosu: $25k hesap + yüksek-fiyatlı sembol + dar stop →
    # risk-bazlı nominal net-yönsel cap'i tek pozisyonda doyuruyordu. Cap-aware ile
    # pozisyon küçülür ama AÇILIR (eskiden hiç açılmıyordu).
    ps = PortfolioState(equity=25_000, ath_equity=25_000, cash_balance=25_000)
    mgr = RiskManager(ps)
    rc = mgr.size_stock(entry=900, stop=890, sector="Tech", symbol="HIGH", side="LONG")
    assert rc.approved and rc.sizing["shares"] >= 1
    # nominal net-yönsel tavanı içinde
    assert rc.sizing["notional"] <= ps.equity * CONFIG.portfolio.max_net_directional_pct + 1e-6


def test_min_position_notional_skips_dust(monkeypatch):
    _patch_portfolio(monkeypatch, min_position_notional=1_000_000.0)
    ps = PortfolioState(equity=25_000, ath_equity=25_000, cash_balance=25_000)
    mgr = RiskManager(ps)
    rc = mgr.size_stock(entry=100, stop=99, sector="S", symbol="A", side="LONG")
    assert not rc.approved and "BELOW_MIN_POSITION" in rc.reason


def test_daily_loss_breach_blocks(monkeypatch):
    _patch_risk(monkeypatch, daily_max_loss_pct=0.10)  # fren .env'de kapalı; mantığı için aç
    ps = fresh(25_000)
    ps.daily_pnl = -ps.equity * rm.CONFIG.risk.daily_max_loss_pct - 1
    mgr = RiskManager(ps)
    assert mgr.daily_loss_breach()
    rc = mgr.size_stock(entry=100, stop=99, symbol="X")
    assert not rc.approved and rc.reason == "DAILY_MAX_LOSS_HIT"


def test_daily_loss_breach_disabled_when_pct_zero(monkeypatch):
    # DAILY_MAX_LOSS_PCT=0 → kasa yarıya inse bile fren devreye girmez.
    _patch_risk(monkeypatch, daily_max_loss_pct=0.0)
    ps = fresh(25_000)
    ps.daily_pnl = -ps.equity * 0.5
    mgr = RiskManager(ps)
    assert not mgr.daily_loss_breach()
    rc = mgr.size_stock(entry=100, stop=99, symbol="X")
    assert rc.reason != "DAILY_MAX_LOSS_HIT"


def test_consecutive_stops_cooldown(monkeypatch):
    _patch_risk(monkeypatch, consecutive_stops_cooldown=3)
    ps = fresh(25_000)
    ps.consecutive_stops = rm.CONFIG.risk.consecutive_stops_cooldown
    mgr = RiskManager(ps)
    rc = mgr.size_stock(entry=100, stop=99, symbol="X")
    assert not rc.approved and rc.reason == "CONSECUTIVE_STOPS_COOLDOWN"


def _enable_pdt(monkeypatch):
    """PDT'yi testte açıkça aç (varsayılan da açık; env ile kapatılmış olabilir)."""
    new_reg = dataclasses.replace(CONFIG.regulation, pdt_enabled=True)
    monkeypatch.setattr(rm, "CONFIG", dataclasses.replace(CONFIG, regulation=new_reg))


def test_pdt_blocks_under_threshold(monkeypatch):
    _enable_pdt(monkeypatch)
    ps = fresh(10_000)  # < 25k
    ps.day_trades_count = CONFIG.regulation.pdt_max_day_trades_under_threshold
    mgr = RiskManager(ps)
    assert mgr.pdt_blocks_day_trade()


def test_pdt_disabled_allows_day_trade(monkeypatch):
    # PDT_ENABLED=0 → eşik altı + sayaç dolu olsa bile engellemez.
    new_reg = dataclasses.replace(CONFIG.regulation, pdt_enabled=False)
    monkeypatch.setattr(rm, "CONFIG", dataclasses.replace(CONFIG, regulation=new_reg))
    ps = fresh(10_000)  # < 25k
    ps.day_trades_count = CONFIG.regulation.pdt_max_day_trades_under_threshold + 5
    mgr = RiskManager(ps)
    assert not mgr.pdt_blocks_day_trade()


def test_adaptive_risk_halves_in_drawdown():
    ps = PortfolioState(equity=80_000, ath_equity=100_000, cash_balance=80_000)  # %20 DD
    mgr = RiskManager(ps)
    base = CONFIG.risk.per_trade_risk_pct_directional
    assert mgr._current_risk_pct(base) == pytest.approx(base * CONFIG.risk.drawdown_risk_multiplier)


def test_greek_cap_blocks_excess_delta():
    ps = fresh(1_000_000)
    mgr = RiskManager(ps)
    # büyük delta'lı çok kontrat → net delta tavanı aşılmalı
    g = GreekSet(delta=0.9, gamma=0.05, theta=-0.05, vega=0.2, iv=0.3)
    rc = mgr.size_long_option(0.10, g, symbol="AAA", option_type="c")
    assert not rc.approved and "GREEK_CAP" in rc.reason


# A5 — düşük-güven kademeli boyutlama
def test_confidence_multiplier_scales_size():
    ps = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    mgr = RiskManager(ps)
    full = mgr.size_stock(entry=100, stop=95, symbol="A", side="LONG", confidence_multiplier=1.0)
    half = mgr.size_stock(entry=100, stop=95, symbol="A", side="LONG", confidence_multiplier=0.5)
    assert full.approved and half.approved
    assert half.sizing["shares"] == full.sizing["shares"] // 2
    assert half.sizing["confidence_mult"] == 0.5


def test_confidence_multiplier_clamped_to_one():
    # >1 çarpan riski artırmamalı (1.0'a kırpılır)
    ps = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    mgr = RiskManager(ps)
    base = mgr.size_stock(entry=100, stop=95, symbol="A", side="LONG", confidence_multiplier=1.0)
    over = mgr.size_stock(entry=100, stop=95, symbol="A", side="LONG", confidence_multiplier=5.0)
    assert over.sizing["shares"] == base.sizing["shares"]
