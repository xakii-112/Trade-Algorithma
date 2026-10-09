"""
pead_pit_test.py — PEAD'i POINT-IN-TIME (survivorship-düzeltilmiş) evrende doğrula.

Evren = `as-of` tarihindeki S&P 500 üyeliği (pit_universe.sp500_as_of), yani sonradan
çıkarılan/iflas/satın-alınan isimleri DE içerir. Test penceresi as-of → bugün.

Kritik kıyas: PEAD metriklerini (a) TÜM PIT evreni, (b) "bugün artık YOK" (removed-since)
alt-kümesi, (c) hayatta-kalan alt-küme üzerinde ayrı raporlar. Kenar yalnızca hayatta
kalanlardan geliyorsa survivorship-yanlı; çıkarılanlarda da varsa kenar gerçek.

SINIR: çıkarılan tickerların yfinance EARNINGS verisi çoğu zaman boş (ölü ticker) →
o isimlerden az/sıfır PEAD işlemi gelebilir; bu durum da raporlanır (coverage).

    python pead_pit_test.py --as-of 2022-01-01
    python pead_pit_test.py --as-of 2020-01-01 --limit 0     # tüm evren (yavaş, ~30 dk)
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.data.pit_universe import _current_and_changes, _norm, sp500_as_of
from nihai.execution.backtest import backtest_symbol, load_bars


def _agg(pnls: list[float], label: str) -> None:
    if not pnls:
        print(f"  {label:<22} (işlem yok)")
        return
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    pf = round(gw / gl, 2) if gl > 0 else None
    print(f"  {label:<22} işlem={len(pnls):<4} WR={100*len(wins)/len(pnls):>5.1f}% "
          f"PF={pf} E={statistics.mean(pnls):>+10.2f}")


def main():
    ap = argparse.ArgumentParser(description="PEAD point-in-time (survivorship-clean) doğrulama")
    ap.add_argument("--as-of", default="2022-01-01", help="evren üyelik tarihi = test başlangıcı")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (varsayılan: bugün)")
    ap.add_argument("--limit", type=int, default=0, help="ilk N sembol (deneme; 0=hepsi)")
    ap.add_argument("--throttle", type=float, default=0.2)
    args = ap.parse_args()

    universe = sp500_as_of(args.as_of)
    cur, _ = _current_and_changes()
    removed = set(universe) - set(_norm(s) for s in cur)   # bugün artık yok
    if args.limit > 0:
        universe = universe[: args.limit]
    n = len(universe)
    print(f"=== PEAD point-in-time | S&P500 @ {args.as_of} | {n} sembol "
          f"({len(removed)} bugün YOK) | test {args.as_of}→{args.end or 'bugün'} ===\n")

    all_pnl: list[float] = []
    removed_pnl: list[float] = []
    survivor_pnl: list[float] = []
    bh: list[float] = []
    n_with_data = n_with_trades = n_removed_with_data = 0

    for i, s in enumerate(universe, 1):
        try:
            df = load_bars(s, "1d", start=args.as_of, end=args.end)
            if df.empty or len(df) < 60:
                continue
            n_with_data += 1
            if s in removed:
                n_removed_with_data += 1
            first, last = float(df["close"].iloc[0]), float(df["close"].iloc[-1])
            if first > 0:
                bh.append((last / first - 1) * 100)
            r = backtest_symbol(s, interval="1d", df=df, strategy_mode="pead")
            if r.trade_pnls:
                n_with_trades += 1
                all_pnl.extend(r.trade_pnls)
                (removed_pnl if s in removed else survivor_pnl).extend(r.trade_pnls)
        except Exception:
            pass
        if i % 25 == 0:
            print(f"  [{i}/{n}] veri={n_with_data} işlemli={n_with_trades} "
                  f"PEAD-işlem={len(all_pnl)}", flush=True)
        if args.throttle > 0:
            time.sleep(args.throttle)

    print(f"\n{'='*64}")
    print(f"COVERAGE: {n_with_data}/{n} sembolde veri | {n_with_trades} sembolde PEAD işlemi")
    print(f"  removed-since: {n_removed_with_data}/{len(removed)} veri bulundu "
          f"(ölü ticker earnings'i çoğu zaman boş)")
    print(f"  ortalama al-tut (beta kontrolü): {statistics.mean(bh):+.1f}%" if bh else "")
    print(f"{'='*64}")
    _agg(all_pnl, "TÜM PIT evreni")
    _agg(survivor_pnl, "hayatta-kalanlar")
    _agg(removed_pnl, "removed-since")
    print(f"{'='*64}")
    print("Yorum: TÜM PIT evreni PF>1 + E>0 ise PEAD survivorship'e dayanıklı.")
    print("removed-since vs hayatta-kalan benzer ise kenar gerçek; sadece hayatta")
    print("kalanlarda ise survivorship-yanlı (ama removed earnings coverage'ına dikkat).")


if __name__ == "__main__":
    main()
