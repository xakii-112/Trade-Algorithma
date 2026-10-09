"""
optimize_pullback.py — trend_pullback çıkış (stop/target) grid optimizasyonu.

Çıkış parametreleri (ATR çarpanı) expectancy'yi girişten çok belirler. Bu script
trend_stop_atr × trend_target_atr gridini sabit sepette İKİ periyotta tarar ve
DAYANIKLILIK gözüyle sıralar: bir kombinasyonun skoru = iki periyodun EN KÖTÜSÜ
(worst-case). Böylece tek-periyot şansı (squeeze/mean_reversion tuzağı) ödüllendirilmez.

    python optimize_pullback.py --interval 1h
    python optimize_pullback.py --interval 1d --periods 1y,2y
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

MODE = "trend_pullback"
STOPS = [1.0, 1.5, 2.0, 2.5]
TARGETS = [2.0, 3.0, 4.0, 5.0, 6.0]


def weighted_e(bars: dict, stop: float, target: float) -> tuple[float, int, float]:
    """Sepet geneli işlem-ağırlıklı expectancy + işlem sayısı + WR."""
    object.__setattr__(CONFIG.strategy, "trend_stop_atr", stop)
    object.__setattr__(CONFIG.strategy, "trend_target_atr", target)
    pnls: list[float] = []
    for s, df in bars.items():
        r = backtest_symbol(s, df=df, strategy_mode=MODE)
        if r.trade_pnls:
            pnls += r.trade_pnls
    if not pnls:
        return 0.0, 0, 0.0
    wr = 100 * sum(1 for p in pnls if p > 0) / len(pnls)
    return statistics.mean(pnls), len(pnls), wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--periods", default="1y,2y")
    args = ap.parse_args()
    periods = [p.strip() for p in args.periods.split(",") if p.strip()]

    # Saf strateji davranışı: kapılar kapalı (modes_compare ile tutarlı).
    object.__setattr__(CONFIG.strategy, "enable_trend_alignment_gate", False)
    object.__setattr__(CONFIG.strategy, "enable_adx_floor_gate", False)
    object.__setattr__(CONFIG.strategy, "min_signal_quality_score", 1.0)

    # Barları periyot başına bir kez indir.
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

    print(f"\n=== trend_pullback çıkış gridi · {args.interval} · periyotlar={periods} ===")
    print("(skor = periyotların EN KÖTÜSÜ; dayanıklılık için worst-case)\n")

    rows = []
    for stop in STOPS:
        for target in TARGETS:
            rr = target / stop
            if rr < CONFIG.strategy.min_rr_ratio:
                continue  # RR kapısı zaten reddeder → işlem üretmez
            per_results = {}
            for per in periods:
                e, n, wr = weighted_e(bars_by_period[per], stop, target)
                per_results[per] = (e, n, wr)
            worst = min(per_results[p][0] for p in periods)
            rows.append((worst, stop, target, rr, per_results))

    rows.sort(key=lambda x: x[0], reverse=True)
    hdr = f"{'stop':>5} {'tgt':>5} {'RR':>5} {'worst_E':>9}"
    for per in periods:
        hdr += f" | {per+'_E':>9} {per+'_n':>6} {per+'_WR':>6}"
    print(hdr)
    print("-" * len(hdr))
    for worst, stop, target, rr, pr in rows:
        line = f"{stop:>5.1f} {target:>5.1f} {rr:>5.2f} {worst:>9.2f}"
        for per in periods:
            e, n, wr = pr[per]
            line += f" | {e:>9.2f} {n:>6} {wr:>6.1f}"
        print(line)

    if rows:
        best = rows[0]
        print(f"\n→ En dayanıklı çıkış: stop={best[1]} ATR, target={best[2]} ATR "
              f"(RR={best[3]:.2f}) · worst-case E={best[0]:+.2f}")


if __name__ == "__main__":
    main()
