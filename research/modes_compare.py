"""
modes_compare.py — Hangi strateji ailesi en az kayıpla/en çok kârla çalışıyor?

Ablasyon trend-takip kapılarının ZARAR verdiğini gösterdi → hipotez: mean-reversion
aileleri bu sepette (intraday mega-cap) daha iyi. Veriye soralım.
Barları bir kez indirir, her strategy_mode'u aynı df'lere karşı koşar.
"""
from __future__ import annotations

import statistics
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import argparse

import nihai.config as config
from nihai.config import CONFIG
from nihai.execution.backtest import backtest_symbol, load_bars
from nihai.universe import BASKET

ALL_MODES = ["event_breakout", "macd", "donchian", "squeeze",
             "mean_reversion", "rsi_divergence", "ensemble"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="1y")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    ap.add_argument("--long-only", action="store_true")
    args = ap.parse_args()
    PERIOD = args.period
    INTERVAL = args.interval
    MODES = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"--- interval={INTERVAL} periyot={PERIOD} long_only={args.long_only} modlar={MODES} ---")
    # Tüm edge kapılarını KAPAT (saf strateji davranışını ölç), legacy ensemble.
    config.ENSEMBLE_MODE = "first_trigger"
    object.__setattr__(CONFIG.strategy, "enable_trend_alignment_gate", False)
    object.__setattr__(CONFIG.strategy, "enable_adx_floor_gate", False)
    object.__setattr__(CONFIG.strategy, "min_signal_quality_score", 1.0)
    object.__setattr__(CONFIG.strategy, "enable_long_only", args.long_only)

    print(f"Barlar indiriliyor ({len(BASKET)} sembol, {INTERVAL}/{PERIOD})...")
    bars = {}
    for s in BASKET:
        df = load_bars(s, INTERVAL, period=PERIOD)
        if not df.empty:
            bars[s] = df
    print(f"  {len(bars)} sembol hazır.\n")

    for mode in MODES:
        pnls_all, per_sym = [], []
        for s, df in bars.items():
            r = backtest_symbol(s, interval=INTERVAL, df=df, strategy_mode=mode)
            if r.trade_pnls:
                pnls_all += r.trade_pnls
                per_sym.append(statistics.mean(r.trade_pnls))
        if not pnls_all:
            print(f"{mode:<18} (işlem yok)")
            continue
        wins = sum(1 for p in pnls_all if p > 0)
        prof = sum(1 for e in per_sym if e > 0)
        print(f"{mode:<18} trades={len(pnls_all):>4}  weighted_E={statistics.mean(pnls_all):>9.2f}  "
              f"mean_sym_E={statistics.mean(per_sym):>9.2f}  "
              f"pct_profit={100*prof/len(per_sym):>5.1f}  WR={100*wins/len(pnls_all):>5.1f}")


if __name__ == "__main__":
    main()
