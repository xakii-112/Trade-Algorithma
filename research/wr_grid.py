"""
wr_grid.py — Başarı oranı (WR) ↔ kâr/zarar (payoff) ödünleşimini ölçer.

Dar hedef WR'ı yükseltir ama payoff'u düşürür. Bu betik (stop, target) ızgarasını
tarar ve her hücrede WR + payoff + profit_factor + expectancy raporlar — böylece
"WR'ı yükselten bir config aynı zamanda kârlı mı?" sorusu doğrudan görülür.
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

import nihai.config as config
from nihai.config import CONFIG
from nihai.execution.backtest import backtest_symbol, load_bars
from nihai.universe import BASKET

# WR'ı yükseltmeye odaklı: dar hedef (target ≤ stop) → daha sık küçük kazanç.
GRID = [
    (1.5, 3.0),   # baseline (geniş hedef, düşük WR)
    (1.5, 1.5),   # 1:1
    (2.0, 1.5),   # hedef < stop
    (2.0, 1.0),   # çok dar hedef (yüksek WR beklenir)
    (3.0, 1.5),   # geniş stop + dar hedef (en yüksek WR)
    (3.0, 1.0),
    (1.5, 1.0),
]


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

    config.ENSEMBLE_MODE = "first_trigger"
    object.__setattr__(CONFIG.strategy, "enable_trend_alignment_gate", False)
    object.__setattr__(CONFIG.strategy, "enable_adx_floor_gate", False)
    object.__setattr__(CONFIG.strategy, "min_signal_quality_score", 1.0)
    object.__setattr__(CONFIG.strategy, "enable_long_only", args.long_only)

    print(f"=== WR IZGARASI | mod={args.mode} {args.interval} long_only={args.long_only} "
          f"periyotlar={periods} ===\n")
    bars_by = {}
    for per in periods:
        b = {}
        for s in BASKET:
            df = load_bars(s, args.interval, period=per)
            if not df.empty:
                b[s] = df
        bars_by[per] = b

    hdr = f"{'stop/target':<13}"
    for per in periods:
        hdr += f"{per+'_WR%':>9}{per+'_payoff':>11}{per+'_PF':>8}{per+'_E':>9}"
    print(hdr)
    print("-" * len(hdr))
    for stop_m, tgt_m in GRID:
        object.__setattr__(CONFIG.strategy, "trend_stop_atr", stop_m)
        object.__setattr__(CONFIG.strategy, "trend_target_atr", tgt_m)
        line = f"{stop_m}/{tgt_m:<9}"
        for per in periods:
            m = metrics(bars_by[per], args.interval, args.mode)
            if m:
                line += f"{m['WR']:>9}{str(m['payoff']):>11}{str(m['PF']):>8}{m['E']:>9}"
            else:
                line += f"{'—':>9}{'—':>11}{'—':>8}{'—':>9}"
        print(line)
    print("\nNot: WR yükseldikçe payoff düşer. Kârlılık = PF>1 (her iki periyotta).")


if __name__ == "__main__":
    main()
