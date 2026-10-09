"""
vix_test.py — Çapraz-varlık VIX rejim kapısının A/B ölçümü.

Tez: VIX > eşik (yüksek-korku) iken yeni giriş bastırılırsa kırılım/trend sinyalleri
iyileşir mi? Gate OFF (baseline) vs birkaç eşik, sabit sepette İKİ periyotta. Dayanıklılık
gözü: bir eşiğin skoru = iki periyodun EN KÖTÜSÜ (worst-case). Tek-periyot şansı ödüllenmez.

    python vix_test.py --mode event_breakout --interval 1h --periods 1y,2y
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

THRESHOLDS = [20.0, 22.0, 25.0, 28.0, 32.0]


def weighted_e(bars: dict, mode: str) -> tuple[float, int]:
    pnls: list[float] = []
    for s, df in bars.items():
        r = backtest_symbol(s, df=df, strategy_mode=mode)
        if r.trade_pnls:
            pnls += r.trade_pnls
    if not pnls:
        return 0.0, 0
    return statistics.mean(pnls), len(pnls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="event_breakout")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--periods", default="1y,2y")
    args = ap.parse_args()
    periods = [p.strip() for p in args.periods.split(",") if p.strip()]

    # Diğer kapıları kapat (saf VIX etkisi).
    object.__setattr__(CONFIG.strategy, "enable_trend_alignment_gate", False)
    object.__setattr__(CONFIG.strategy, "enable_adx_floor_gate", False)
    object.__setattr__(CONFIG.strategy, "min_signal_quality_score", 1.0)

    bars_by_period: dict[str, dict] = {}
    for per in periods:
        print(f"Barlar indiriliyor ({len(BASKET)} sembol, {args.interval}/{per})...")
        b = {}
        for s in BASKET:
            df = load_bars(s, args.interval, period=per)
            if not df.empty:
                b[s] = df
        bars_by_period[per] = b
        print(f"  {len(b)} sembol hazır.")

    print(f"\n=== VIX rejim kapısı · {args.mode} · {args.interval} · periyotlar={periods} ===")
    print("(skor = periyotların EN KÖTÜSÜ; baseline = kapı kapalı)\n")

    rows = []
    # Baseline: gate OFF
    object.__setattr__(CONFIG.strategy, "enable_vix_regime_gate", False)
    base = {}
    for per in periods:
        e, n = weighted_e(bars_by_period[per], args.mode)
        base[per] = (e, n)
    base_worst = min(base[p][0] for p in periods)

    # Eşik taraması: gate ON
    object.__setattr__(CONFIG.strategy, "enable_vix_regime_gate", True)
    for thr in THRESHOLDS:
        object.__setattr__(CONFIG.strategy, "vix_high_threshold", thr)
        pr = {}
        for per in periods:
            e, n = weighted_e(bars_by_period[per], args.mode)
            pr[per] = (e, n)
        worst = min(pr[p][0] for p in periods)
        rows.append((worst, thr, pr))

    rows.sort(key=lambda x: x[0], reverse=True)

    hdr = f"{'config':>16} {'worst_E':>9}"
    for per in periods:
        hdr += f" | {per+'_E':>9} {per+'_n':>6}"
    print(hdr)
    print("-" * len(hdr))
    # baseline satırı
    line = f"{'baseline(OFF)':>16} {base_worst:>9.2f}"
    for per in periods:
        e, n = base[per]
        line += f" | {e:>9.2f} {n:>6}"
    print(line)
    print("-" * len(hdr))
    for worst, thr, pr in rows:
        line = f"{'VIX>'+str(thr):>16} {worst:>9.2f}"
        for per in periods:
            e, n = pr[per]
            line += f" | {e:>9.2f} {n:>6}"
        print(line)

    best = rows[0]
    print(f"\n→ En iyi eşik (worst-case): VIX>{best[1]} · {best[0]:+.2f}  "
          f"| baseline (kapalı): {base_worst:+.2f}")
    if best[0] > base_worst:
        print(f"  VIX kapısı worst-case'i {best[0]-base_worst:+.2f} İYİLEŞTİRDİ.")
    else:
        print("  VIX kapısı worst-case'i İYİLEŞTİRMEDİ (baseline daha iyi/eşit).")


if __name__ == "__main__":
    main()
