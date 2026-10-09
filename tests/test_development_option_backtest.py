"""Gerçek zincir: sonraki bar, gerçek makas/ücret, veri boşluğu ve tüketici bağlantısı."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nihai.data import dolt_options as dolt
from nihai.execution import backtest as bt
from nihai.risk.risk_manager import OPTION_FEES
from nihai.signals.strategy import SignalAction


def inputs():
    days = pd.bdate_range("2024-02-05", periods=5)
    bars = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=1e6), index=days)
    chain = pd.DataFrame([dict(date=day, act_symbol="AAA", expiration="2024-03-15",
        strike=strike, call_put="Call", bid=bid, ask=ask, vol=.3,
        delta=.5 if strike == 100 else .8, gamma=.02, theta=-.03, vega=.1)
        for day, bid, ask in zip(days, [1.9, 2.2, 3.5, 3., 2.7], [2., 2.5, 3.7, 3.2, 2.9], strict=True)
        for strike in (100., 105.)])
    return bars, chain


@pytest.fixture
def engine(monkeypatch):
    decisions = []
    def signal(symbol, data, **kwargs):
        decisions.append(data.index[-1])
        return SimpleNamespace(action=SignalAction.LONG, is_actionable=lambda: len(data) == 1)
    monkeypatch.setattr(bt, "generate_signal", signal)
    monkeypatch.setattr(bt.RiskManager, "size_long_option",
                        lambda *a, **kw: SimpleNamespace(approved=True, sizing={"contracts": 2}))
    monkeypatch.setattr(bt, "CONFIG", replace(bt.CONFIG, option_strategy=replace(
        bt.CONFIG.option_strategy, premium_target_gain_pct=.25, premium_stop_loss_pct=.5)))
    return decisions


def run(bars, chain):
    return bt.backtest_option_chain("AAA", df=bars, chain_df=chain, warmup_bars=0, initial_equity=10000)


def test_next_bar_quotes_and_exact_costs(engine):
    bars, chain = inputs()
    result = run(bars, chain)
    buy, sell = result.option_data["fills"]
    assert buy["decision_date"] == "2024-02-05" and buy["date"] == "2024-02-06"
    assert buy["price"] == 2.5 and buy["strike"] == 100
    # Hedef 7 Şubat kapanışında görüldü; aynı gün 3.5 ile kurgusal çıkış yok.
    assert sell["date"] == "2024-02-08" and sell["price"] == 3 and sell["reason"] == "PREMIUM_TARGET"
    fee = sum(OPTION_FEES.values())
    pnl = (3 - 2.5) * 200 - 4 * fee
    assert result.trade_pnls == pytest.approx([pnl])
    assert result.final_equity == pytest.approx(10000 + pnl)
    assert result.equity_curve[2] == pytest.approx(10000 - .3 * 200 - 2 * fee)
    assert result.option_data["spread_cost"] == pytest.approx(50)
    assert result.option_data["fees"] == pytest.approx(4 * fee)
    assert result.option_data["complete"] and result.option_data["exec_lag_bars"] == 1
    assert engine[0] == bars.index[0]
    assert run(bars, chain).trade_pnls == result.trade_pnls


def test_future_chain_cannot_reselect_contract(engine):
    bars, chain = inputs()
    baseline = run(bars, chain)
    future = chain.date > bars.index[0]
    chain.loc[future & (chain.strike == 100), "delta"] = .1
    chain.loc[future & (chain.strike == 105), "delta"] = .5
    result = run(bars, chain)
    assert result.option_data["fills"][0]["strike"] == 100
    assert result.trade_pnls == baseline.trade_pnls
    history = dolt.prepare_option_history(chain, "AAA")
    snapshot = dolt.option_snapshot(history, as_of="2024-02-05")
    assert snapshot.date.max() == bars.index[0] and snapshot.delta.tolist() == [.5, .8]
    assert dolt.option_snapshot(history, as_of="2024-02-04").empty
    assert dolt.option_snapshot(history, as_of="2024-02-05T12:00:00").empty


def test_missing_quote_never_becomes_synthetic_or_stale_fill(engine, monkeypatch, tmp_path):
    bars, chain = inputs()
    chain["bid"], chain["ask"] = 2.2, 2.5
    chain = chain[chain.date != bars.index[-1]]
    result = run(bars, chain)
    assert result.trades_count == 0 and len(result.option_data["fills"]) == 1
    assert result.final_equity is None and result.total_return_pct is None
    assert result.max_drawdown_pct is None and result.sharpe_annualized is None
    assert result.equity_curve[-1] is None
    assert result.option_data["missing_mark_bars"] == 1 and not result.option_data["complete"]
    monkeypatch.setattr(bt, "LOG_DIR", tmp_path)
    report = bt.save_report(result)
    assert "OPT_AAA" in report.name  # Windows dosya adında iki nokta yok
    assert '"final_equity": null' in report.read_text(encoding="utf-8")
    bt.print_report(result)  # eksik metrik konsol biçimlendirmesini düşürmez
    assert "N/A" in result.short_summary()


@pytest.mark.parametrize("problem", ["missing_quote", "crossed", "missing_greeks"])
def test_unfillable_entry_is_cancelled(engine, problem):
    bars, chain = inputs()
    entry = chain.date == bars.index[1]
    if problem == "missing_quote":
        chain = chain[~entry]
    elif problem == "crossed":
        chain.loc[entry, "ask"] = .1
    else:
        chain.loc[entry, "gamma"] = np.nan
    result = run(bars, chain)
    assert result.option_data["fills"] == [] and result.final_equity == 10000
    assert result.option_data["unfilled_entries"] + result.option_data["rejected_entries"] == 1


def test_final_liquidation_zero_bid_and_expiry_guard(engine):
    bars, chain = inputs()
    chain["bid"], chain["ask"] = 2.2, 2.5
    result = run(bars, chain)
    assert result.option_data["fills"][-1]["reason"] == "BT_END"
    chain.loc[chain.date == bars.index[-1], "bid"] = 0.
    result = run(bars, chain)
    assert result.trades_count == 0 and result.option_data["open_position"] is not None
    assert result.final_equity == pytest.approx(10000 - 500 - 2 * sum(OPTION_FEES.values()))
    assert not result.option_data["complete"]  # sıfır bid ölçüm; dolum kanıtı değil
    chain["expiration"] = "2024-02-13"
    chain["bid"] = 2.2
    result = run(bars, chain)
    assert result.option_data["fills"][-1]["reason"] == "DTE_GUARD"


def test_local_reader_filters_without_network_or_bulk_copy(monkeypatch, tmp_path):
    _, chain = inputs()
    csv = tmp_path / "option_chain.csv"
    chain.to_csv(csv, index=False)
    monkeypatch.setattr(dolt, "_build_sidecar", lambda *a: pytest.fail("Tam tablo kopyası kurulamaz"))
    history = dolt.local_option_history("aaa", path=csv, start="2024-02-06", end="2024-02-07")
    assert len(history) == 4 and history.date.min() == pd.Timestamp("2024-02-06")
    assert history.date.max() == pd.Timestamp("2024-02-07")
    assert not csv.with_suffix(".parquet").exists()
    assert dolt.local_option_history("AAA", path=tmp_path / "missing.csv").empty
    damaged = pd.concat([chain, chain.iloc[:1].assign(ask=3)], ignore_index=True)
    damaged.loc[2, "bid"] = np.inf
    cleaned = dolt.prepare_option_history(damaged, "AAA")
    assert len(cleaned) == len(chain) - 2 and cleaned.attrs["rejected_rows"] == 3


def test_empty_data_daily_boundary_and_cli_routing(engine, monkeypatch):
    bars, chain = inputs()
    with pytest.raises(ValueError, match="sentetik"):
        run(bars, pd.DataFrame())
    with pytest.raises(ValueError, match="interval=1d"):
        bt.backtest_option_chain("AAA", interval="1h", df=bars, chain_df=chain)
    result = run(bars, chain)
    calls = []
    monkeypatch.setattr(bt, "backtest_option_chain", lambda symbol, **kw: calls.append((symbol, kw)) or result)
    monkeypatch.setattr(bt.sys, "argv", ["backtest", "--options", "--symbol", "AAA"])
    bt.main()
    assert calls[0][1]["interval"] == "1d"
    from nihai.web.api_lab import BacktestInput
    with pytest.raises(ValueError, match="1d"):
        BacktestInput(symbols=["AAA"], mode="options", interval="1h")
