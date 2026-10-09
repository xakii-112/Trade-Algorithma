"""
stats.py — Ship edilen default config için ayrıntılı kâr/zarar metrikleri.
Win rate, payoff (ort.kazanç/ort.kayıp), profit factor, expectancy — iki periyotta.
"""
from __future__ import annotations

import argparse
import statistics
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.execution.backtest import backtest_symbol, load_bars
from nihai.universe import BASKET


def analyze(bars, interval, mode):
    pnls = []
    for s, df in bars.items():
        r = backtest_symbol(s, interval=interval, df=df, strategy_mode=mode)
        pnls += r.trade_pnls
    if not pnls:
        return None
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = abs(statistics.mean(losses)) if losses else 0.0
    return {
        "trades": len(pnls),
        "win_rate_pct": round(100 * len(wins) / len(pnls), 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff (kar/zarar)": round(avg_win / avg_loss, 2) if avg_loss else None,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
        "expectancy": round(statistics.mean(pnls), 2),
        "total_pnl": round(sum(pnls), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="event_breakout")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--periods", default="1y,2y")
    args = ap.parse_args()
    periods = [p.strip() for p in args.periods.split(",")]

    print(f"=== {args.mode} | {args.interval} | ship-default config ===\n")
    for per in periods:
        bars = {}
        for s in BASKET:
            df = load_bars(s, args.interval, period=per)
            if not df.empty:
                bars[s] = df
        m = analyze(bars, args.interval, args.mode)
        print(f"--- {per} ({len(bars)} sembol) ---")
        if m:
            for k, v in m.items():
                print(f"  {k:<22} {v}")
        print()


if __name__ == "__main__":
    main()
