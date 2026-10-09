"""
confluence_test.py — Katı `confluence` (mutabakat) modunu İZOLE ölçer.

`confluence` modu (strategy.py): bir sinyal için ≥CONFLUENCE_MIN_VOTES bağımsız
detektörün HEMFİKİR olmasını ister (§16.2). Bu harness onun gevşek `ensemble`'a ve
tek-detektör `event_breakout`'a kıyasla precision'ı (WR/PF) gerçekten keskinleştirip
keskinleştirmediğini ölçer — VE bunun survivorship artefaktı olup olmadığını test eder.

KENAR İDDİASI YOK. EDGE_FINDINGS §13.5/§16 disiplini: parlak in-sample sayılar
(PF 1.68→2.75) survivorship-temiz PIT evrende deflate oldu. confluence fiyat-göstergesi
ailesinden türediği için (§6: o ailenin OOS kenarı yok) muhtemelen aynı sonuç çıkar;
bu harness onu DOĞRULAR, varsaymaz.

    python -m research.confluence_test                         # BASKET, 1d, 2y+5y
    python -m research.confluence_test --min-votes 3           # daha katı mutabakat
    python -m research.confluence_test --interval 1h --periods 1y,2y
    python -m research.confluence_test --pit --as-of 2022-01-01  # survivorship-temiz
    python -m research.confluence_test --pit --as-of 2022-01-01 --limit 0  # tüm evren (yavaş)
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

import nihai.config as cfg
from nihai.execution.backtest import backtest_symbol, load_bars
from nihai.universe import BASKET

# confluence modu dispatcher'da `from nihai.config import CONFLUENCE_MIN_VOTES`
# çağrı-anında okur → buradan modül attribute'unu set etmek eşiği etkiler.


def _set_min_votes(n: int) -> None:
    cfg.CONFLUENCE_MIN_VOTES = max(2, n)


def _agg(pnls: list[float]) -> dict | None:
    if not pnls:
        return None
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = abs(statistics.mean(losses)) if losses else 0.0
    return {
        "trades": len(pnls),
        "win_rate_pct": round(100 * len(wins) / len(pnls), 1),
        "payoff": round(avg_win / avg_loss, 2) if avg_loss else None,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
        "expectancy": round(statistics.mean(pnls), 2),
        "total_pnl": round(sum(pnls), 2),
    }


def _run_basket(period: str, interval: str, mode: str) -> tuple[dict | None, float, int]:
    """Sepeti backtest et; toplu PnL metriği + ortalama al-tut (beta kontrolü) döndür."""
    pnls: list[float] = []
    buyhold: list[float] = []
    n_sym = 0
    for s in BASKET:
        df = load_bars(s, interval, period=period)
        if df.empty or len(df) < 60:
            continue
        n_sym += 1
        r = backtest_symbol(s, interval=interval, df=df, strategy_mode=mode)
        pnls += r.trade_pnls
        first, last = float(df["close"].iloc[0]), float(df["close"].iloc[-1])
        if first > 0:
            buyhold.append((last / first - 1) * 100)
    bh = statistics.mean(buyhold) if buyhold else 0.0
    return _agg(pnls), bh, n_sym


def _print_metrics(label: str, m: dict | None) -> None:
    print(f"  {label}:")
    if not m:
        print("    (işlem yok)")
        return
    for k, v in m.items():
        print(f"    {k:<16} {v}")
    print()


def run_basket(args) -> None:
    periods = [p.strip() for p in args.periods.split(",")]
    modes = ["confluence", "ensemble", args.compare]
    print(f"=== confluence izole ölçüm | {args.interval} | sepet={len(BASKET)} sembol "
          f"| min_votes={cfg.CONFLUENCE_MIN_VOTES} ===")
    print("Hedef: confluence, ensemble/baz'a göre WR↑ ve PF↑ (precision keskinleşmeli, §16.2).")
    print("Beta kontrolü: toplam getiri ort. al-tut'un altındaysa = sadece piyasa betası.")
    print("DİKKAT: bu in-sample/BASKET (survivorship-yanlı) — kenar için --pit şart.\n")

    for per in periods:
        print(f"{'='*64}\n--- PERİYOT: {per} ---\n{'='*64}")
        results = []
        bh = 0.0
        n_sym = 0
        for mode in modes:
            m, bh_m, n = _run_basket(per, args.interval, mode)
            results.append((mode, m))
            bh, n_sym = bh_m, n
        print(f"  [{n_sym} sembol]  ortalama AL-TUT getirisi: {bh:+.1f}%\n")
        for mode, m in results:
            _print_metrics(mode, m)


def _agg_line(pnls: list[float], label: str) -> None:
    if not pnls:
        print(f"  {label:<22} (işlem yok)")
        return
    wins = [p for p in pnls if p > 0]
    gw = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = round(gw / gl, 2) if gl > 0 else None
    print(f"  {label:<22} işlem={len(pnls):<4} WR={100*len(wins)/len(pnls):>5.1f}% "
          f"PF={pf} E={statistics.mean(pnls):>+10.2f}")


def run_pit(args) -> None:
    """POINT-IN-TIME (survivorship-temiz) ölçüm: as-of tarihindeki S&P500 üyeliği,
    çıkarılanlar dahil. confluence metriklerini TÜM/hayatta-kalan/removed-since böler."""
    from nihai.data.pit_universe import _current_and_changes, _norm, sp500_as_of

    universe = sp500_as_of(args.as_of)
    cur, _ = _current_and_changes()
    removed = set(universe) - {_norm(s) for s in cur}
    if args.limit > 0:
        universe = universe[: args.limit]
    n = len(universe)
    print(f"=== confluence point-in-time | S&P500 @ {args.as_of} | {n} sembol "
          f"({len(removed)} bugün YOK) | min_votes={cfg.CONFLUENCE_MIN_VOTES} "
          f"| test {args.as_of}→{args.end or 'bugün'} ===\n")

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
            r = backtest_symbol(s, interval="1d", df=df, strategy_mode="confluence")
            if r.trade_pnls:
                n_with_trades += 1
                all_pnl.extend(r.trade_pnls)
                (removed_pnl if s in removed else survivor_pnl).extend(r.trade_pnls)
        except Exception:
            pass
        if i % 25 == 0:
            print(f"  [{i}/{n}] veri={n_with_data} işlemli={n_with_trades} "
                  f"confluence-işlem={len(all_pnl)}", flush=True)
        if args.throttle > 0:
            time.sleep(args.throttle)

    print(f"\n{'='*64}")
    print(f"COVERAGE: {n_with_data}/{n} sembolde veri | {n_with_trades} sembolde işlem")
    print(f"  removed-since: {n_removed_with_data}/{len(removed)} veri bulundu")
    if bh:
        print(f"  ortalama al-tut (beta kontrolü): {statistics.mean(bh):+.1f}%")
    print(f"{'='*64}")
    _agg_line(all_pnl, "TÜM PIT evreni")
    _agg_line(survivor_pnl, "hayatta-kalanlar")
    _agg_line(removed_pnl, "removed-since")
    print(f"{'='*64}")
    print("Yorum: TÜM PIT evreni PF>1 + E>0 ise confluence survivorship'e dayanıklı.")
    print("Yalnız hayatta-kalanlarda kenar varsa = survivorship-yanlı (beklenen sonuç, §16).")


def main():
    ap = argparse.ArgumentParser(description="Katı confluence (mutabakat) izole ölçüm")
    ap.add_argument("--periods", default="2y,5y", help="virgülle: 1y,2y,5y (BASKET modu)")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--compare", default="event_breakout",
                    help="tek-detektör kıyas stratejisi")
    ap.add_argument("--min-votes", type=int, default=cfg.CONFLUENCE_MIN_VOTES,
                    help="confluence için gereken min hemfikir detektör (>=2)")
    ap.add_argument("--pit", action="store_true",
                    help="survivorship-temiz POINT-IN-TIME modu (S&P500 @ --as-of)")
    ap.add_argument("--as-of", default="2022-01-01", help="--pit: evren üyelik tarihi = başlangıç")
    ap.add_argument("--end", default=None, help="--pit: YYYY-MM-DD (varsayılan bugün)")
    ap.add_argument("--limit", type=int, default=0, help="--pit: ilk N sembol (0=hepsi)")
    ap.add_argument("--throttle", type=float, default=0.2, help="--pit: sembol başı bekleme (s)")
    args = ap.parse_args()

    _set_min_votes(args.min_votes)
    if args.pit:
        run_pit(args)
    else:
        run_basket(args)


if __name__ == "__main__":
    main()
