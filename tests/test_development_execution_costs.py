"""Dört kalem muhasebe, ölçüm kapsamı ve ortak sanal dolum yolu (offline)."""
import csv
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nihai.config import CONFIG
from nihai.execution import cost_attribution as ca
from nihai.execution.state import _deserialize_position, _serialize
from nihai.risk.position_manager import PositionManager, SlippageModel, liquidity_inputs, quote_spread_pct
from nihai.risk.risk_manager import PortfolioState
from nihai.signals.strategy import Signal, SignalAction


def test_shortfall_closed_form_partial_fills_and_signed_opportunity():
    args = dict(side="BUY", quantity=100, filled_quantity=60, signal_price=10,
                decision_price=11, fill_price=12, fees=3, benchmark_price=14)
    cost = ca.implementation_shortfall(**args)
    assert [cost[k] for k in ca.COST_FIELDS] == [60, 60, 3, 160, 283]
    assert cost["paper_pnl"] == 400 and cost["actual_pnl"] == 117
    assert cost["paper_pnl"] - cost["actual_pnl"] == cost["total_shortfall"]
    assert cost == ca.implementation_shortfall(**args)  # bit-bit deterministik
    short = ca.implementation_shortfall(**{**args, "side": "SELL"})
    assert [short[k] for k in ca.COST_FIELDS] == [-60, -60, 3, -160, -277]
    assert short["paper_pnl"] - short["actual_pnl"] == short["total_shortfall"]
    missed = ca.implementation_shortfall(**{**args, "filled_quantity": 0, "decision_price": None,
                                           "fill_price": None, "benchmark_price": 8, "fees": 0})
    assert [missed[k] for k in ca.COST_FIELDS] == [0, 0, 0, -200, -200]
    option = ca.implementation_shortfall(**{**args, "multiplier": 100})
    assert option["total_shortfall"] == 28003  # ücret zaten dolar, 100 ile tekrar çarpılmaz
    flat = ca.implementation_shortfall(**{**args, "decision_price": 10, "fill_price": 10,
                                         "benchmark_price": 10, "fees": 0})
    assert all(flat[k] == 0 for k in ca.COST_FIELDS)


def test_shortfall_missing_measurements_and_input_guards():
    args = dict(side="BUY", quantity=2, filled_quantity=2, signal_price=10,
                decision_price=None, fill_price=11, fees=1)
    cost = ca.implementation_shortfall(**args)
    assert cost["delay_cost"] is None and cost["market_impact"] is None
    assert cost["total_shortfall"] == 3 and cost["opportunity_cost"] == 0
    assert ca.implementation_shortfall(**{**args, "filled_quantity": 1})["total_shortfall"] is None
    assert ca.implementation_shortfall(**{**args, "fees": None})["total_shortfall"] is None
    # Tur-78 (R3-B1): savunmacı sözleşme — geçersiz girdi None, istisna YOK.
    for invalid in ({"side": "LONG"}, {"quantity": 0}, {"quantity": 1}, {"filled_quantity": -1},
                    {"multiplier": 0}, {"fees": float("inf")}, {"fill_price": -1},
                    {"signal_price": float("nan")}, {"quantity": float("inf")}, {"quantity": None},
                    {"fees": "x"}, {"quantity": 1e308, "filled_quantity": 1e308, "multiplier": 1e308}):
        assert ca.implementation_shortfall(**{**args, **invalid}) is None


def test_cost_summary_non_finite_is_unmeasured_not_exception():
    base = ca.cost_summary([{"fees": 2.0, "delay_cost": 1.0}])
    out = ca.cost_summary([{"fees": float("nan"), "delay_cost": float("inf")}, {"fees": "x"},
                           {"fees": 1e308}, {"fees": 1e308}], base)
    assert out["records"] == 5 and base["records"] == 1  # önceki sözlük değişmez
    assert out["fees"] == 2.0 + 1e308 and out["measured"]["fees"] == 2  # taşan ikinci 1e308 sayılmaz
    assert out["delay_cost"] == 1.0 and out["measured"]["delay_cost"] == 1


def test_unmeasurable_cost_leg_does_not_block_close_bookkeeping():
    portfolio = PortfolioState(equity=10000, cash_balance=10000, ath_equity=10000)
    manager = PositionManager(portfolio, SlippageModel(impact_coef=0), record_logs=False)
    pos = manager.open_from_signal(Signal(SignalAction.LONG, "TEST", 100, 50, 150), {"shares": 10},
                                   now=datetime(2024, 1, 2), signal_price=98)
    pos.entry_signal_price = float("nan")  # bozuk/eski durum dosyasından gelmiş gibi
    event = manager.force_close(pos, 110, "TEST", datetime(2024, 1, 3))
    assert pos not in portfolio.open_positions and manager._key(pos) not in manager._trackers
    assert event.realized_pnl == pytest.approx((event.exit_price - pos.entry_price) * 10)  # komisyon 0
    assert all(event.cost_attribution[k] is None for k in ca.COST_FIELDS)
    assert manager.execution_costs["measured"]["total_shortfall"] == 0


def test_cost_summary_and_csv_preserve_unknown_coverage(tmp_path):
    path = tmp_path / "orders.csv"
    path.write_text("side,quantity,filled_quantity,signal_price,decision_price,fill_price,fees,benchmark_price\n"
                    "BUY,100,60,10,11,12,3,14\nBUY,100,0,10,,,0,8\n", encoding="utf-8")
    result = ca.read_costs(path, orders=True)
    assert result["total_shortfall"] == 83
    assert all(n == 2 for n in result["measured"].values())
    combined = ca.cost_summary([dict.fromkeys(ca.COST_FIELDS)], result)
    assert combined["records"] == 3 and result["records"] == 2
    assert combined["measured"] == result["measured"]
    path.write_text("symbol,realized_pnl\nOLD,22\n", encoding="utf-8")
    old = ca.read_costs(path)
    assert old["records"] == 1 and old["fees"] is None and old["measured"]["fees"] == 0
    path.write_text("side,quantity,filled_quantity\nBUY,bad,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="satır 2"):
        ca.read_costs(path, orders=True)


@pytest.mark.parametrize("side,reference,exit_price", [("LONG", 98, 110), ("SHORT", 102, 90)])
def test_position_round_trip_restart_and_csv_identity(side, reference, exit_price, monkeypatch, tmp_path):
    from nihai.signals import signal_engine as se

    portfolio = PortfolioState(equity=10000, cash_balance=10000, ath_equity=10000)
    model = SlippageModel(stock_commission_per_trade=2, impact_coef=0)
    manager = PositionManager(portfolio, model, record_logs=False)
    sig = Signal(SignalAction[side], "TEST", 100, 50, 150)
    pos = manager.open_from_signal(sig, {"shares": 10}, now=datetime(2024, 1, 2), signal_price=reference)
    portfolio.open_positions[:] = [_deserialize_position(_serialize(pos))]
    manager = PositionManager(portfolio, model)
    manager.rehydrate_trackers()
    monkeypatch.setattr(se, "TRADES_CSV", tmp_path / "trades.csv")
    monkeypatch.setattr("nihai.risk.position_manager.log_trade_postmortem", lambda *args: None)
    monkeypatch.setattr("nihai.risk.position_manager.log_decision", lambda *args: None)
    event = manager.force_close(portfolio.open_positions[0], exit_price, "TEST", datetime(2024, 1, 3))
    costs = event.cost_attribution
    assert costs["delay_cost"] == 20 and costs["fees"] == 4 and costs["opportunity_cost"] == 0
    paper = (exit_price - reference) * 10 * (1 if side == "LONG" else -1)
    assert paper - event.realized_pnl == pytest.approx(costs["total_shortfall"])
    assert sum(costs[k] for k in ca.COST_FIELDS[:-1]) == pytest.approx(costs["total_shortfall"])
    assert manager.execution_costs["measured"]["total_shortfall"] == 1
    with se.TRADES_CSV.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["total_shortfall"]) == pytest.approx(costs["total_shortfall"])


def test_legacy_position_unknown_and_failed_exit_keeps_tracker():
    portfolio = PortfolioState(equity=10000, cash_balance=10000, ath_equity=10000)
    manager = PositionManager(portfolio, SlippageModel(impact_coef=0), record_logs=False)
    sig = Signal(SignalAction.LONG, "TEST", 100, 50, 150)
    pos = manager.open_from_signal(sig, {"shares": 10}, now=datetime(2024, 1, 2))
    pos.entry_signal_price = pos.entry_decision_price = None
    for price in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            manager.force_close(pos, price, "TEST", datetime(2024, 1, 3))
        assert pos in portfolio.open_positions and manager._key(pos) in manager._trackers
    event = manager.force_close(pos, 110, "TEST", datetime(2024, 1, 3))
    assert event.cost_attribution["total_shortfall"] is None
    assert event.cost_attribution["fees"] == 0
    assert manager.execution_costs["measured"]["total_shortfall"] == 0


def test_dynamic_spread_impact_inputs_units_and_failures(monkeypatch):
    model = SlippageModel(impact_coef=1)
    args = dict(qty=100, adv_shares=10000, sigma=.02, spread_pct=.004)
    assert model.slip_pct("stock", **args) == .004  # yarım makas .002 + etki .002
    assert model.slip_pct("stock", **args) == model.slip_pct("stock", **args)
    assert model.apply_entry(100, "LONG", "stock", **args) == 100.4
    assert model.apply_exit(100, "SHORT", "stock", **args) == 100.4
    assert SlippageModel(impact_coef=0).slip_pct("stock", spread_pct=0) == 0
    for missing in (float("inf"), float("nan"), None, -1):
        assert model.slip_pct("stock", **{**args, "sigma": missing}) == .002
    with pytest.raises(ValueError):
        model.slip_pct("stock", **{**args, "qty": 1e12})
    with pytest.raises(ValueError):
        SlippageModel(impact_coef=float("inf"))
    with pytest.raises(ValueError):
        model.apply_entry(1.7e308, "LONG", "stock", spread_pct=1.5)
    monkeypatch.setitem(vars(CONFIG.backtest), "impact_coef", .4)
    assert SlippageModel().impact_coef == .4  # sanal döngü de mevcut ayarı okur
    assert quote_spread_pct(SimpleNamespace(bid=99.5, ask=100.5, spread_measured=True), 100) == .01
    # Tur-89: tavanı aşan makas (IEX ince defteri) ölçüm sayılmaz
    assert quote_spread_pct(SimpleNamespace(bid=99, ask=101, spread_measured=True), 100) is None
    assert quote_spread_pct(SimpleNamespace(bid=99, ask=101), 100) is None
    assert quote_spread_pct(SimpleNamespace(bid=102, ask=101), 100) is None
    df = pd.DataFrame(dict(high=[101.] * 25, low=[99.] * 25, close=[100.] * 25, volume=[10000.] * 25))
    inputs = liquidity_inputs(df)
    assert inputs == {"adv_shares": 10000, "bar_sigma": .02}
    assert inputs == liquidity_inputs(df)
    assert liquidity_inputs(df.iloc[:2])["bar_sigma"] is None
    df.loc[24, "volume"] = np.nan
    assert liquidity_inputs(df)["adv_shares"] is None


def test_backtest_keeps_signal_reference_and_causal_exit_liquidity(monkeypatch):
    from nihai.execution import backtest as bt

    monkeypatch.setitem(vars(CONFIG.backtest), "impact_coef", .5)
    monkeypatch.setattr(bt, "get_sector", lambda s: "UNKNOWN")
    monkeypatch.setattr(bt.RiskManager, "size_stock", lambda *a, **k: SimpleNamespace(approved=True, sizing={"shares": 10}))
    df = pd.DataFrame(dict(open=[102.] * 30, high=[103.] * 30, low=[99.] * 30,
                           close=[100.] * 30, volume=[10000.] * 30),
                      index=pd.date_range("2024-01-02 09:30", periods=30, freq="h"))
    df.iloc[-1, df.columns.get_loc("volume")] = 1e20
    seen = []
    original = bt.PositionManager.update_all_with_bars

    def observe(self, bars, **kwargs):
        seen.append(kwargs.get("liquidity_by_symbol"))
        return original(self, bars, **kwargs)

    monkeypatch.setattr(bt.PositionManager, "update_all_with_bars", observe)
    calls = []

    def signal(symbol, bars, **kwargs):
        calls.append(len(bars))
        return Signal(SignalAction.LONG if len(calls) == 1 else SignalAction.WAIT, symbol, 100, 50, 200)

    monkeypatch.setattr(bt, "generate_signal", signal)
    result = bt.backtest_symbol("TEST", df=df, warmup_bars=20)
    assert result.execution_costs["records"] == 1
    assert result.execution_costs["delay_cost"] == 20
    assert all(item["TEST"]["adv_shares"] == 10000 for item in seen)
    # İnfaz barı hacmi, önceki karar barının maliyetine sızmaz.
    v = np.array([3., 5., 7., 1e20])
    assert bt._causal_adv(v, 3)[:3].tolist() == [3, 4, 5]
    v[2] = np.nan
    assert np.isnan(bt._causal_adv(v, 2)[2:]).all()


def test_virtual_intrabar_exit_uses_previous_completed_liquidity():
    portfolio = PortfolioState(equity=100000, cash_balance=100000, ath_equity=100000)
    manager = PositionManager(portfolio, SlippageModel(impact_coef=1), record_logs=False)
    sig = Signal(SignalAction.LONG, "TEST", 100, 95, 120)
    manager.open_from_signal(sig, {"shares": 100}, now=datetime(2024, 1, 2, 9, 30))
    df = pd.DataFrame(dict(open=[100.] * 40, high=[101.] * 40, low=[99.] * 40,
                           close=[100.] * 40, volume=[10000.] * 40))
    df.iloc[-1] = [94, 95, 93, 94, 1e20]
    seen = []

    def bars(symbol, interval, lookback_bars):
        seen.append(lookback_bars)
        return df.tail(lookback_bars)

    event, = manager.update_all_with_1m_bars(SimpleNamespace(get_bars=bars), datetime(2024, 1, 2, 10, 30))
    assert seen[0] > 20
    assert event.exit_price == pytest.approx(94 * (1 - .001 - .02 * .1))


def test_cost_cli_restores_configuration_after_failure(monkeypatch):
    before = CONFIG.backtest.impact_coef

    def fail(args):
        ca.set_costs(0, 0, 0, 5)
        raise RuntimeError("test")

    monkeypatch.setattr(ca, "_compare_costs", fail)
    monkeypatch.setattr("sys.argv", ["cost_attribution"])
    with pytest.raises(RuntimeError, match="test"):
        ca.main()
    assert CONFIG.backtest.impact_coef == before
