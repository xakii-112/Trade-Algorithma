"""
exit_grid.py — Çıkış (stop/target ATR çarpanı) ızgara taraması.

Entry + zaman dilimi + yön (long-only) denendi; son kaldıraç ÇIKIŞ. Trend-takipte
sabit hedef kazananı sınırlar. Bu betik (stop_atr, target_atr) ızgarasını sabit bir
mod + long-only ile tarar. Aynı df'ler iki periyotta (1y, 2y) → dayanıklılık.

    python exit_grid.py --mode donchian
    python exit_grid.py --mode event_breakout --interval 1h
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

# (stop_atr, target_atr) adayları. Genişleyen hedef = kazananı bırak-koş; trailing
# yerine geniş sabit hedef yaklaşık eder. Asimetri (geniş hedef/dar stop) trend tezi.
GRID = [
    (1.5, 3.0),   # mevcut varsayılan
    (1.5, 2.0),   # daha yakın hedef (daha yüksek isabet)
    (1.0, 3.0),   # dar stop
    (2.0, 4.0),   # geniş stop+hedef
    (1.5, 4.5),   # kazananı daha çok bırak
    (1.5, 6.0),   # çok geniş hedef (fat-tail yakala)
    (1.0, 4.0),   # dar stop + geniş hedef (en asimetrik)
    (2.0, 6.0),
]

# A6 — TRAILING varyantları (--trailing). Stop/target sabit (config varsayılanı) tutulup
# trailing kipi taranır → A1 (kademeli trail + kâr-tabanı kilidi) OOS'ta doğrulanır.
# Her varyant: (etiket, config-override-dict). trail_drop_target=True → kazananı koştur.
_DEF_TIERS = ((5.0, 1.0), (3.0, 1.5), (1.5, 2.0))
TRAIL_VARIANTS: list[tuple[str, dict]] = [
    ("trailsiz",       {"enable_trailing": False}),
    ("duz_1.5R",       {"enable_trailing": True, "trail_tiered": False, "trail_r": 1.5}),
    ("duz_2.0R",       {"enable_trailing": True, "trail_tiered": False, "trail_r": 2.0}),
    ("duz_3.0R",       {"enable_trailing": True, "trail_tiered": False, "trail_r": 3.0}),
    ("kademeli",       {"enable_trailing": True, "trail_tiered": True, "trail_tiers": _DEF_TIERS,
                        "trail_lock_trigger_r": 0.0}),
    ("kademeli+kilit", {"enable_trailing": True, "trail_tiered": True, "trail_tiers": _DEF_TIERS,
                        "trail_lock_trigger_r": 3.0, "trail_lock_floor_r": 1.0}),
]


def sweep_trailing(bars_by_period, interval, mode, periods):
    """A6 — trailing varyantlarını iki periyotta tara (stop/target config varsayılanında)."""
    object.__setattr__(CONFIG.risk, "trail_drop_target", True)   # kazananı koştur (sabit hedef kalk)
    hdr = f"{'trailing kipi':<16}"
    for per in periods:
        hdr += f"{per+'_E':>10}{per+'_prof%':>11}"
    print(hdr)
    print("-" * len(hdr))
    robust = []
    for label, overrides in TRAIL_VARIANTS:
        for k, v in overrides.items():
            object.__setattr__(CONFIG.risk, k, v)
        line = f"{label:<16}"
        es = []
        for per in periods:
            m = measure(bars_by_period[per], interval, mode)
            if m:
                line += f"{m['weighted_E']:>10}{m['pct_profit']:>11}"
                es.append(m["weighted_E"])
            else:
                line += f"{'—':>10}{'—':>11}"
        print(line)
        if es and all(e > 0 for e in es):
            robust.append((label, min(es)))
    print("\n" + "=" * 60)
    if robust:
        robust.sort(key=lambda x: -x[1])
        print("TÜM PERİYOTLARDA POZİTİF (dayanıklı):")
        for label, mn in robust:
            print(f"  {label}  (min weighted_E={mn})")
    else:
        print("Hiçbir trailing kipi tüm periyotlarda pozitif DEĞİL.")
    print("=" * 60)


def measure(bars, interval, mode):
    pnls_all, per_sym = [], []
    for s, df in bars.items():
        r = backtest_symbol(s, interval=interval, df=df, strategy_mode=mode)
        if r.trade_pnls:
            pnls_all += r.trade_pnls
            per_sym.append(statistics.mean(r.trade_pnls))
    if not pnls_all:
        return None
    wins = sum(1 for p in pnls_all if p > 0)
    prof = sum(1 for e in per_sym if e > 0)
    return {
        "trades": len(pnls_all),
        "weighted_E": round(statistics.mean(pnls_all), 2),
        "pct_profit": round(100 * prof / len(per_sym), 1),
        "WR": round(100 * wins / len(pnls_all), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="donchian")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--periods", default="1y,2y")
    ap.add_argument("--trailing", action="store_true",
                    help="A6: stop/target ızgarası yerine TRAILING varyantlarını tara")
    args = ap.parse_args()
    periods = [p.strip() for p in args.periods.split(",")]

    # Sabitler: long-only AÇIK, kapılar kapalı, legacy ensemble.
    config.ENSEMBLE_MODE = "first_trigger"
    object.__setattr__(CONFIG.strategy, "enable_trend_alignment_gate", False)
    object.__setattr__(CONFIG.strategy, "enable_adx_floor_gate", False)
    object.__setattr__(CONFIG.strategy, "min_signal_quality_score", 1.0)
    object.__setattr__(CONFIG.strategy, "enable_long_only", True)

    sweep_label = "TRAILING VARYANTLARI" if args.trailing else "ÇIKIŞ IZGARASI"
    print(f"=== {sweep_label} | mod={args.mode} interval={args.interval} "
          f"long_only=True periyotlar={periods} ===\n")
    bars_by_period = {}
    for per in periods:
        print(f"Barlar indiriliyor ({len(BASKET)} sembol, {args.interval}/{per})...")
        b = {}
        for s in BASKET:
            df = load_bars(s, args.interval, period=per)
            if not df.empty:
                b[s] = df
        bars_by_period[per] = b
    print()

    if args.trailing:
        sweep_trailing(bars_by_period, args.interval, args.mode, periods)
        return

    hdr = f"{'stop/target':<14}"
    for per in periods:
        hdr += f"{per+'_E':>10}{per+'_prof%':>11}"
    print(hdr)
    print("-" * len(hdr))
    robust = []
    for stop_m, tgt_m in GRID:
        object.__setattr__(CONFIG.strategy, "trend_stop_atr", stop_m)
        object.__setattr__(CONFIG.strategy, "trend_target_atr", tgt_m)
        line = f"{stop_m}/{tgt_m:<10}"
        es = []
        for per in periods:
            m = measure(bars_by_period[per], args.interval, args.mode)
            if m:
                line += f"{m['weighted_E']:>10}{m['pct_profit']:>11}"
                es.append(m["weighted_E"])
            else:
                line += f"{'—':>10}{'—':>11}"
        print(line)
        if es and all(e > 0 for e in es):
            robust.append((stop_m, tgt_m, min(es)))

    print("\n" + "=" * 60)
    if robust:
        robust.sort(key=lambda x: -x[2])
        print("TÜM PERİYOTLARDA POZİTİF (dayanıklı):")
        for s, t, mn in robust:
            print(f"  stop={s} target={t}  (min weighted_E={mn})")
    else:
        print("Hiçbir (stop,target) tüm periyotlarda pozitif DEĞİL.")
    print("=" * 60)


if __name__ == "__main__":
    main()
