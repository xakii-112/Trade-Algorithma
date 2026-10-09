"""
wf_pullback.py — trend_pullback için DÜRÜST out-of-sample testi (walk-forward).

optimize_pullback.py in-sample optimize etti (aynı 1y/2y üzerinde). Bu script asıl
kanıtı arar: optimize edilmiş çıkışla (stop=1.0, target=5.0 ATR) trend_pullback'i her
sembolde yuvarlanan IS/OOS pencerelerinde test eder. Yalnızca OOS (görülmemiş) sonuçlar
toplanır. Pozitif OOS expectancy = gerçek kenar; negatif = in-sample optimizasyon overfit'ti.

    python wf_pullback.py --interval 1d --period 2y
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
from nihai.execution.backtest import load_bars, walk_forward
from nihai.universe import BASKET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="trend_pullback")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--stop", type=float, default=None, help="ATR stop çarpanı (yoksa config varsayılanı)")
    ap.add_argument("--target", type=float, default=None, help="ATR target çarpanı (yoksa config varsayılanı)")
    ap.add_argument("--is-bars", dest="is_bars", type=int, default=180)
    ap.add_argument("--oos-bars", dest="oos_bars", type=int, default=60)
    args = ap.parse_args()

    if args.stop is not None:
        object.__setattr__(CONFIG.strategy, "trend_stop_atr", args.stop)
    if args.target is not None:
        object.__setattr__(CONFIG.strategy, "trend_target_atr", args.target)
    object.__setattr__(CONFIG.strategy, "enable_trend_alignment_gate", False)
    object.__setattr__(CONFIG.strategy, "enable_adx_floor_gate", False)
    object.__setattr__(CONFIG.strategy, "min_signal_quality_score", 1.0)

    print(f"=== WALK-FORWARD · {args.mode} · {args.interval}/{args.period} · "
          f"stop={args.stop or CONFIG.strategy.trend_stop_atr} "
          f"target={args.target or CONFIG.strategy.trend_target_atr} ATR ===")
    print(f"(IS={args.is_bars} / OOS={args.oos_bars} bar; yalnızca OOS sonuçlar)\n")

    oos_pnls_all: list[float] = []
    effs: list[float] = []
    per_sym_e: list[float] = []
    print(f"{'sembol':>7} {'OOS_n':>6} {'OOS_E':>9} {'OOS_ret%':>9} {'WF_eff':>7}")
    print("-" * 46)
    for s in BASKET:
        df = load_bars(s, args.interval, period=args.period)
        if df.empty:
            continue
        r = walk_forward(s, interval=args.interval, df=df, candidates=[args.mode],
                         is_bars=args.is_bars, oos_bars=args.oos_bars, metric="expectancy")
        # OOS işlem PnL'lerini stitched curve'den türetemeyiz; expectancy×n yaklaşık değil,
        # walk_forward zaten OOS expectancy/return veriyor → onları topla.
        n = r.oos_trades_count
        if n > 0:
            oos_pnls_all.append(r.oos_expectancy * n)  # toplam OOS PnL ~ E*n
            per_sym_e.append(r.oos_expectancy)
        effs.append(r.wf_efficiency)
        print(f"{s:>7} {n:>6} {r.oos_expectancy:>9.2f} {r.oos_total_return_pct:>+9.2f} "
              f"{r.wf_efficiency:>7.2f}")

    print("-" * 46)
    if per_sym_e:
        print(f"Sembol-ortalama OOS expectancy : {statistics.mean(per_sym_e):+.2f}")
        print(f"Pozitif-OOS sembol oranı       : "
              f"{100*sum(1 for e in per_sym_e if e>0)/len(per_sym_e):.1f}%  "
              f"({sum(1 for e in per_sym_e if e>0)}/{len(per_sym_e)})")
    if effs:
        print(f"Ortalama WF verimliliği        : {statistics.mean(effs):+.2f}  "
              f"(>0 → OOS'ta da kâr yönü korunuyor)")
    print(f"İşlem açan sembol sayısı       : {len(per_sym_e)}/{len(BASKET)}")


if __name__ == "__main__":
    main()
