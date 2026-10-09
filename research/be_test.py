"""
be_test.py — Breakeven stop'un WR/PF/expectancy üzerindeki etkisi.
Baseline (BE kapalı) vs çeşitli BE tetik seviyeleri — event_breakout, 1y+2y.
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

from nihai.config import CONFIG
from nihai.execution.backtest import backtest_symbol, load_bars
from nihai.universe import BASKET


def metrics(bars, interval, mode):
    pnls = []
    for s, df in bars.items():
        r = backtest_symbol(s, interval=interval, df=df, strategy_mode=mode)
        pnls += r.trade_pnls
    if not pnls:
        return None
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    aw = statistics.mean(wins) if wins else 0.0
    al = abs(statistics.mean(losses)) if losses else 0.0
    gl = abs(sum(losses))
    return {
        "n": len(pnls),
        "WR": round(100 * len(wins) / len(pnls), 1),
        "payoff": round(aw / al, 2) if al else None,
        "PF": round(sum(wins) / gl, 2) if gl else None,
        "E": round(statistics.mean(pnls), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="event_breakout")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--periods", default="1y,2y")
    ap.add_argument("--long-only", action="store_true")
    args = ap.parse_args()
    periods = [p.strip() for p in args.periods.split(",")]

    object.__setattr__(CONFIG.strategy, "enable_long_only", args.long_only)

    bars_by = {}
    for per in periods:
        b = {}
        for s in BASKET:
            df = load_bars(s, args.interval, period=per)
            if not df.empty:
                b[s] = df
        bars_by[per] = b

    # (etiket, enable_be, trigger_r)
    configs = [
        ("baseline (BE off)", False, 0.0),
        ("BE @ 0.5R", True, 0.5),
        ("BE @ 1.0R", True, 1.0),
        ("BE @ 1.5R", True, 1.5),
    ]
    print(f"=== BREAKEVEN | {args.mode} {args.interval} long_only={args.long_only} "
          f"periyotlar={periods} ===\n")
    hdr = f"{'config':<20}"
    for per in periods:
        hdr += f"{per+'_WR%':>9}{per+'_PF':>8}{per+'_E':>9}"
    print(hdr)
    print("-" * len(hdr))
    for label, en, tr in configs:
        object.__setattr__(CONFIG.risk, "enable_breakeven", en)
        object.__setattr__(CONFIG.risk, "be_trigger_r", tr)
        line = f"{label:<20}"
        for per in periods:
            m = metrics(bars_by[per], args.interval, args.mode)
            line += f"{m['WR']:>9}{str(m['PF']):>8}{m['E']:>9}"
        print(line)
    print("\nHedef: WR ↑ VE PF her iki periyotta ↑ (özellikle PF>baseline).")


if __name__ == "__main__":
    main()
