"""
trail_test.py — Trailing stop'un WR'ı sabit tutup payoff/PF'i yükseltip yükseltmediğini ölçer.
Baseline (sabit hedef) vs trailing (hedef kaldırılmış, çeşitli trail mesafeleri).
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
        "WR": round(100 * len(wins) / len(pnls), 1),
        "payoff": round(aw / al, 2) if al else None,
        "PF": round(sum(wins) / gl, 2) if gl else None,
        "E": round(statistics.mean(pnls), 1),
        "n": len(pnls),
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
    object.__setattr__(CONFIG.risk, "enable_breakeven", False)
    object.__setattr__(CONFIG.risk, "trail_drop_target", True)

    bars_by = {}
    for per in periods:
        b = {}
        for s in BASKET:
            df = load_bars(s, args.interval, period=per)
            if not df.empty:
                b[s] = df
        bars_by[per] = b

    # (etiket, enable_trailing, trail_r)
    configs = [
        ("baseline (fixed tgt)", False, 0.0),
        ("trail 1.5R", True, 1.5),
        ("trail 2.0R", True, 2.0),
        ("trail 2.5R", True, 2.5),
        ("trail 3.0R", True, 3.0),
    ]
    print(f"=== TRAILING | {args.mode} {args.interval} long_only={args.long_only} "
          f"periyotlar={periods} ===\n")
    hdr = f"{'config':<22}"
    for per in periods:
        hdr += f"{per+'_WR%':>9}{per+'_payoff':>11}{per+'_PF':>8}{per+'_E':>9}"
    print(hdr)
    print("-" * len(hdr))
    for label, en, tr in configs:
        object.__setattr__(CONFIG.risk, "enable_trailing", en)
        object.__setattr__(CONFIG.risk, "trail_r", tr)
        line = f"{label:<22}"
        for per in periods:
            m = metrics(bars_by[per], args.interval, args.mode)
            line += f"{m['WR']:>9}{str(m['payoff']):>11}{str(m['PF']):>8}{m['E']:>9}"
        print(line)
    print("\nHedef: WR ≈ baseline (sabit) AMA payoff/PF ve E ↑ (her iki periyotta).")


if __name__ == "__main__":
    main()
