"""Pozisyon yöneticisi testleri — Madde 4 (gap), 6, 11, 28, 32, 41, 50 (offline)."""
from datetime import date, datetime

from nihai.config import CONFIG
from nihai.risk.options_analyzer import GreekSet, days_to_expiry
from nihai.risk.position_manager import PositionManager, SlippageModel
from nihai.risk.risk_manager import PortfolioState, VirtualPosition
from nihai.signals.strategy import Signal, SignalAction


def make_pmgr(commission=0.0):
    ps = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    slip = SlippageModel(stock_slippage_pct=0.0, option_slippage_pct=0.0,
                         stock_commission_per_trade=commission)
    return ps, PositionManager(ps, slip)


def open_stock(pmgr, *, side="LONG", entry=100, stop=95, target=110, qty=100):
    sig = Signal(action=SignalAction.LONG if side == "LONG" else SignalAction.SHORT,
                 symbol="AAA", price=entry, stop=stop, target=target, rr_ratio=2.0)
    return pmgr.open_from_signal(sig, {"shares": qty, "risk_dollars": qty * 5},
                                 now=datetime(2024, 1, 1), asset_type="stock")


def test_gap_stop_exits_at_open_below_stop():
    ps, pmgr = make_pmgr()
    open_stock(pmgr, side="LONG", entry=100, stop=95)
    # bar açılışı stop'un altında → GAP_STOP, açılış fiyatından
    closed = pmgr.update_all_with_bars({"AAA": (90.0, 92.0, 89.0, 91.0)}, now=datetime(2024, 1, 2))
    assert len(closed) == 1
    assert closed[0].exit_trigger == "GAP_STOP"


def test_target_hit():
    ps, pmgr = make_pmgr()
    open_stock(pmgr, side="LONG", entry=100, stop=95, target=110)
    closed = pmgr.update_all_with_bars({"AAA": (101.0, 112.0, 100.0, 111.0)}, now=datetime(2024, 1, 2))
    assert len(closed) == 1 and closed[0].exit_trigger == "TARGET"


def test_stock_commission_applied():
    ps, pmgr = make_pmgr(commission=1.0)
    open_stock(pmgr, side="LONG", entry=100, stop=95, target=110, qty=10)
    closed = pmgr.update_all_with_bars({"AAA": (101.0, 112.0, 100.0, 111.0)}, now=datetime(2024, 1, 2))
    # hedef 110, giriş 100, 10 adet → +100; komisyon 2 (giriş+çıkış) düşülmeli
    assert closed[0].realized_pnl == 100.0 - 2.0


def test_unrealized_pnl_property():
    ps, pmgr = make_pmgr()
    pos = open_stock(pmgr, side="LONG", entry=100, qty=10)
    pos.last_mark = 105
    assert pos.unrealized_pnl == 50.0


def test_rehydrate_monitors_loaded_position():
    # state.json'dan yüklenmiş gibi açık pozisyon (tracker yok)
    ps = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    ps.open_positions.append(VirtualPosition(
        symbol="AAA", asset_type="stock", side="LONG", quantity=10,
        entry_price=100, entry_time=date.today(), stop=95, target=110, last_mark=100))
    pmgr = PositionManager(ps, SlippageModel(stock_slippage_pct=0, option_slippage_pct=0))
    # rehydrate olmadan: tracker yok → izlenmez (eski bug davranışı)
    assert pmgr.update_all_with_bars({"AAA": (101, 112, 100, 111)}, now=datetime(2024, 1, 2)) == []
    # rehydrate sonrası: izlenir ve TARGET ile kapanır
    assert pmgr.rehydrate_trackers() == 1
    closed = pmgr.update_all_with_bars({"AAA": (101, 112, 100, 111)}, now=datetime(2024, 1, 2))
    assert len(closed) == 1 and closed[0].exit_trigger == "TARGET"


def _set_risk(**kw):
    old = {k: getattr(CONFIG.risk, k) for k in kw}
    for k, v in kw.items():
        object.__setattr__(CONFIG.risk, k, v)
    return old


def _restore_risk(old):
    for k, v in old.items():
        object.__setattr__(CONFIG.risk, k, v)


# A1 — Kademeli trailing + kâr-tabanı kilidi -------------------------------------
def test_tiered_trailing_tightens_with_profit():
    old = _set_risk(enable_trailing=True, trail_drop_target=True, trail_tiered=True,
                    trail_tiers=((5.0, 1.0), (3.0, 1.5), (1.5, 2.0)),
                    trail_lock_trigger_r=3.0, trail_lock_floor_r=1.0, time_stop_bars=999)
    try:
        ps, pmgr = make_pmgr()
        pos = open_stock(pmgr, side="LONG", entry=100, stop=95, target=110)  # risk = 5
        # Lehte +6R hareket (high 130) → en sıkı katman (1.0R) → stop = 130 − 1.0×5 = 125
        closed = pmgr.update_all_with_bars({"AAA": (100.0, 130.0, 99.0, 128.0)},
                                           now=datetime(2024, 1, 2))
        assert closed == []
        assert abs(pos.stop - 125.0) < 1e-6
        # Sonraki bar stop'a değer → STOP @125 (kademeli sıkışma kazancı kilitledi)
        closed = pmgr.update_all_with_bars({"AAA": (128.0, 129.0, 124.0, 124.0)},
                                           now=datetime(2024, 1, 3))
        assert len(closed) == 1 and closed[0].exit_trigger == "STOP"
    finally:
        _restore_risk(old)


def test_trailing_profit_floor_lock_binds():
    # Geniş trail katmanı (10R) → ham trail stop'u entry'nin ALTINA iter; taban kilidi
    # devreye girip stop'u entry + 1R'ye (=105) yükseltmeli.
    old = _set_risk(enable_trailing=True, trail_drop_target=True, trail_tiered=True,
                    trail_tiers=((3.0, 10.0),), trail_lock_trigger_r=3.0,
                    trail_lock_floor_r=1.0, time_stop_bars=999)
    try:
        ps, pmgr = make_pmgr()
        pos = open_stock(pmgr, side="LONG", entry=100, stop=95, target=110)  # risk = 5
        closed = pmgr.update_all_with_bars({"AAA": (100.0, 130.0, 99.0, 128.0)},
                                           now=datetime(2024, 1, 2))
        assert closed == []
        assert abs(pos.stop - 105.0) < 1e-6   # entry + trail_lock_floor_r×risk
    finally:
        _restore_risk(old)


def test_trailing_byte_identical_when_tiered_off():
    # trail_tiered KAPALI → eski düz trail_r davranışı (kilit yok)
    old = _set_risk(enable_trailing=True, trail_drop_target=True, trail_tiered=False,
                    trail_r=2.0, trail_lock_trigger_r=3.0, time_stop_bars=999)
    try:
        ps, pmgr = make_pmgr()
        pos = open_stock(pmgr, side="LONG", entry=100, stop=95, target=110)  # risk = 5
        pmgr.update_all_with_bars({"AAA": (100.0, 130.0, 99.0, 128.0)}, now=datetime(2024, 1, 2))
        assert abs(pos.stop - (130.0 - 2.0 * 5)) < 1e-6   # düz: high_water − trail_r×risk = 120
    finally:
        _restore_risk(old)


# A2 — MFE give-back koruması (R-tabanlı) ----------------------------------------
def test_mfe_giveback_exits_after_peak_retrace():
    # entry=100, stop=80 → risk=20 (1R). arm_r=1.5 → MFE≥30 (fiyat≥130) armlar.
    # giveback_r=1.0 → tepeden 20 (1R) geri verilince çık.
    old = _set_risk(enable_mfe_giveback=True, mfe_giveback_arm_r=1.5,
                    mfe_giveback_r=1.0, enable_trailing=False, time_stop_bars=999)
    try:
        ps, pmgr = make_pmgr()
        open_stock(pmgr, side="LONG", entry=100, stop=80, target=200)  # geniş → stop/target tetiklenmez
        # bar1: tepe 135 (MFE=1.75R ≥ arm), kapanış 132 → giveback 0.15R < 1.0 → çıkış yok
        closed = pmgr.update_all_with_bars({"AAA": (100.0, 135.0, 99.0, 132.0)},
                                           now=datetime(2024, 1, 2))
        assert closed == []
        # bar2: tepe hâlâ 135; kapanış 113 → giveback (135−113)/20 = 1.1R ≥ 1.0 → MFE_GIVEBACK @113
        closed = pmgr.update_all_with_bars({"AAA": (132.0, 134.0, 112.0, 113.0)},
                                           now=datetime(2024, 1, 3))
        assert len(closed) == 1 and closed[0].exit_trigger == "MFE_GIVEBACK"
        assert abs(closed[0].exit_price - 113.0) < 1e-6
    finally:
        _restore_risk(old)


def test_mfe_giveback_not_armed_below_threshold():
    # MFE arm_r'ye ulaşmazsa büyük geri-veriş bile tetiklemez
    old = _set_risk(enable_mfe_giveback=True, mfe_giveback_arm_r=1.5,
                    mfe_giveback_r=1.0, enable_trailing=False, time_stop_bars=999)
    try:
        ps, pmgr = make_pmgr()
        open_stock(pmgr, side="LONG", entry=100, stop=80, target=200)  # risk=20
        # tepe 120 → MFE=1.0R < arm 1.5R → asla armlamaz, geri-veriş yok sayılır
        closed = pmgr.update_all_with_bars({"AAA": (100.0, 120.0, 99.0, 101.0)},
                                           now=datetime(2024, 1, 2))
        assert closed == []
    finally:
        _restore_risk(old)


def test_mfe_giveback_disabled_by_default():
    # Varsayılan (kapalı) → MFE_GIVEBACK asla tetiklenmez
    ps, pmgr = make_pmgr()
    open_stock(pmgr, side="LONG", entry=100, stop=80, target=200)
    closed = pmgr.update_all_with_bars({"AAA": (100.0, 150.0, 99.0, 110.0)}, now=datetime(2024, 1, 2))
    assert closed == []   # büyük geri-çekilme ama özellik kapalı


def test_option_dte_guard_closes_near_expiry():
    ps, pmgr = make_pmgr()
    # vadeye çok yakın opsiyon → DTE_GUARD
    near = (datetime.now()).date()
    sig = Signal(action=SignalAction.LONG, symbol="AAA", price=2.0, stop=1.0, target=4.0)
    g = GreekSet(0.5, 0.02, -0.05, 0.1, 0.3)
    pmgr.open_from_signal(sig, {"contracts": 1, "risk_dollars": 200}, now=datetime.now(),
                          asset_type="option", option_type="c", strike=100,
                          expiry=near.isoformat(), greeks=g)
    closed = pmgr.update_all_with_quotes_stub = None  # noqa
    # check_exit doğrudan: DTE <= eşik
    from nihai.risk.position_manager import PositionTracker
    tr = list(pmgr._trackers.values())[0]
    res = pmgr.check_exit(tr, bar_high=2.0, bar_low=2.0, bar_open=2.0, bar_close=2.0,
                          now_dt=datetime.now())
    assert res is not None and res[0] == "DTE_GUARD"
