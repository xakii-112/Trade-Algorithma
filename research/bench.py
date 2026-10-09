"""
bench.py — Strateji değişikliklerini ölçmek için sabit kıyas (benchmark) harness'i.

Aynı sembol sepetini, aynı periyotta backtest eder ve toplulaştırılmış metrikleri
yazar. Değişiklik ÖNCESİ ve SONRASI çalıştırılıp karşılaştırılır (compound: ölç →
değiştir → tekrar ölç).

    python bench.py                      # varsayılan sepet, 1h, 2y, ensemble
    python bench.py --interval 1d --period 3y
    python bench.py --strategy ensemble --tag after
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.config import LOG_DIR
from nihai.execution.backtest import backtest_symbol
from nihai.universe import BASKET  # tek kaynak


def run(symbols, *, interval, period, strategy_mode, mc):
    rows = []
    for i, s in enumerate(symbols, 1):
        try:
            r = backtest_symbol(s, interval=interval, period=period,
                                strategy_mode=strategy_mode, mc_iters=mc)
            if r.bars_total == 0:
                print(f"[{i}/{len(symbols)}] {s}: veri yok, atlandı")
                continue
            rows.append({
                "symbol": s, "trades": r.trades_count,
                "win_rate": round(r.win_rate, 4),
                "expectancy": round(r.expectancy, 2),
                "total_return_pct": round(r.total_return_pct, 2),
                "max_drawdown_pct": round(r.max_drawdown_pct, 2),
                "sharpe": round(r.sharpe_annualized, 2),
                "profit_factor": (round(r.profit_factor, 2)
                                  if getattr(r, "profit_factor", None) else None),
            })
            print(f"[{i}/{len(symbols)}] {s}: trades={r.trades_count} "
                  f"WR={r.win_rate*100:.0f}% E={r.expectancy:+.1f} "
                  f"ret={r.total_return_pct:+.1f}% sharpe={r.sharpe_annualized:.2f}")
        except Exception as e:
            print(f"[{i}/{len(symbols)}] {s}: hata {e}")
    return rows


def aggregate(rows):
    if not rows:
        return {}
    total_trades = sum(r["trades"] for r in rows)
    # İşlem-ağırlıklı expectancy (toplam PnL ~ Σ trades*E)
    weighted_e = (sum(r["expectancy"] * r["trades"] for r in rows) / total_trades
                  if total_trades else 0.0)
    profitable = [r for r in rows if r["expectancy"] > 0]
    rets = [r["total_return_pct"] for r in rows]
    sharpes = [r["sharpe"] for r in rows]
    wrs = [r["win_rate"] for r in rows]
    return {
        "symbols_tested": len(rows),
        "total_trades": total_trades,
        "avg_trades_per_symbol": round(total_trades / len(rows), 1),
        "weighted_expectancy": round(weighted_e, 2),
        "mean_symbol_expectancy": round(statistics.mean(r["expectancy"] for r in rows), 2),
        "pct_symbols_profitable": round(100 * len(profitable) / len(rows), 1),
        "mean_win_rate_pct": round(100 * statistics.mean(wrs), 1),
        "mean_total_return_pct": round(statistics.mean(rets), 2),
        "median_total_return_pct": round(statistics.median(rets), 2),
        "mean_sharpe": round(statistics.mean(sharpes), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--strategy", default="ensemble")
    ap.add_argument("--mc", type=int, default=0)
    ap.add_argument("--tag", default="bench")
    args = ap.parse_args()

    print(f"=== BENCH [{args.tag}] {args.interval}/{args.period} strat={args.strategy} "
          f"| {len(BASKET)} sembol ===")
    rows = run(BASKET, interval=args.interval, period=args.period,
               strategy_mode=args.strategy, mc=args.mc)
    agg = aggregate(rows)

    print("\n" + "=" * 60)
    print(f" TOPLULAŞTIRILMIŞ — [{args.tag}]")
    print("=" * 60)
    for k, v in agg.items():
        print(f"  {k:<28} {v}")
    print("=" * 60)

    out = LOG_DIR / f"bench_{args.tag}_{args.interval}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"args": vars(args), "aggregate": agg, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
