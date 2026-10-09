"""
insider_test.py — SEC EDGAR insider (Form 4 açık-piyasa alım) sinyalini İZOLE ölç (C1).

EDGE_FINDINGS metodolojisi (docs/EDGE_FINDINGS.md §13.5/§16): önce ölç → iki bağımsız
periyotta doğrula → al-tut (beta) ile kıyasla → POINT-IN-TIME (survivorship-temiz) evrende
yeniden ölç. PEAD bu testte survivorship'in büyük kısmının artefakt olduğunu gösterdi;
insider için de AYNI deflasyon beklenir. KENAR İDDİASI YOK; bu harness "gerçek kenar mı,
boğa-betası/survivorship mi?" sorusunu ayırır.

İki mod:
  1) BASKET (varsayılan): nihai.universe.BASKET üzerinde 2 periyot + al-tut beta + kıyas.
       python insider_test.py
       python insider_test.py --periods 1y,2y,5y
  2) PIT (--pit): pit_universe.sp500_as_of ile survivorship-temiz S&P500 evreni; metrikleri
     TÜM evren / hayatta-kalan / removed-since alt-kümeleri için ayrı raporlar.
       python insider_test.py --pit --as-of 2022-01-01 --limit 60

NOT: ilk çalıştırma EDGAR'dan sembol başına TÜM Form 4 XML'lerini çeker ve
data/insider_cache/<SYM>.json'a yazar → İLK koşu YAVAŞTIR (sembol başına dakikalar),
sonraki koşular cache'ten hızlıdır. Kibarlık için SEC_THROTTLE (varsayılan 0.2s) uygulanır.
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

from nihai.execution.backtest import backtest_symbol, load_bars
from nihai.universe import BASKET


def _agg(pnls: list[float]) -> dict | None:
    if not pnls:
        return None
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win, gross_loss = sum(wins), abs(sum(losses))
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


def _agg_line(pnls: list[float], label: str) -> None:
    if not pnls:
        print(f"  {label:<22} (işlem yok)")
        return
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    pf = round(gw / gl, 2) if gl > 0 else None
    print(f"  {label:<22} işlem={len(pnls):<4} WR={100*len(wins)/len(pnls):>5.1f}% "
          f"PF={pf} E={statistics.mean(pnls):>+10.2f}")


# ---------------------------------------------------------------------------
# Mod 1: BASKET — 2 periyot + al-tut beta + kıyas stratejisi
# ---------------------------------------------------------------------------
def run_period(period: str, interval: str, mode: str) -> tuple[dict | None, float, int]:
    """Sepeti tek tek backtest et; toplu PnL metriği + ortalama al-tut getirisi döndür.
    al-tut = (son/ilk − 1)·100, sembol başına; insider getirisini piyasa betasından ayırır."""
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


def run_basket(args) -> None:
    periods = [p.strip() for p in args.periods.split(",")]
    print(f"=== INSIDER izole ölçüm | {args.interval} | sepet={len(BASKET)} sembol ===")
    print("Hedef: iki periyotta da insider profit_factor>1 VE expectancy>0 (dayanıklı aday).")
    print("Beta kontrolü: insider toplam getirisi ort. al-tut'un altındaysa = sadece piyasa betası.")
    print("UYARI: kenar görünse bile --pit ile survivorship-temiz tekrar ölç (PEAD §16 dersi).\n")
    for per in periods:
        print(f"{'='*64}\n--- PERİYOT: {per} ---\n{'='*64}")
        ins_m, bh, n_sym = run_period(per, args.interval, "insider")
        cmp_m, _, _ = run_period(per, args.interval, args.compare)
        print(f"  [{n_sym} sembol]  ortalama AL-TUT getirisi: {bh:+.1f}%\n")
        for label, m in (("INSIDER", ins_m), (args.compare, cmp_m)):
            print(f"  {label}:")
            if not m:
                print("    (işlem yok)")
                continue
            for k, v in m.items():
                print(f"    {k:<16} {v}")
            print()


# ---------------------------------------------------------------------------
# Mod 2: PIT — survivorship-temiz S&P500 evreni (pead_pit_test.py deseni)
# ---------------------------------------------------------------------------
def run_pit(args) -> None:
    from nihai.data.pit_universe import _current_and_changes, _norm, sp500_as_of
    universe = sp500_as_of(args.as_of)
    cur, _ = _current_and_changes()
    removed = set(universe) - {_norm(s) for s in cur}
    if args.limit > 0:
        universe = universe[: args.limit]
    n = len(universe)
    print(f"=== INSIDER point-in-time | S&P500 @ {args.as_of} | {n} sembol "
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
            r = backtest_symbol(s, interval="1d", df=df, strategy_mode="insider")
            if r.trade_pnls:
                n_with_trades += 1
                all_pnl.extend(r.trade_pnls)
                (removed_pnl if s in removed else survivor_pnl).extend(r.trade_pnls)
        except Exception:
            pass
        if i % 25 == 0:
            print(f"  [{i}/{n}] veri={n_with_data} işlemli={n_with_trades} "
                  f"insider-işlem={len(all_pnl)}", flush=True)
        if args.throttle > 0:
            time.sleep(args.throttle)

    print(f"\n{'='*64}")
    print(f"COVERAGE: {n_with_data}/{n} sembolde veri | {n_with_trades} sembolde insider işlemi")
    print(f"  removed-since: {n_removed_with_data}/{len(removed)} veri bulundu")
    print(f"  ortalama al-tut (beta kontrolü): {statistics.mean(bh):+.1f}%" if bh else "")
    print(f"{'='*64}")
    _agg_line(all_pnl, "TÜM PIT evreni")
    _agg_line(survivor_pnl, "hayatta-kalanlar")
    _agg_line(removed_pnl, "removed-since")
    print(f"{'='*64}")
    print("Yorum: TÜM PIT evreni PF>1 + E>0 ise insider survivorship'e dayanıklı.")
    print("removed-since ≈ hayatta-kalan ise kenar gerçek; sadece hayatta kalanlarda ise")
    print("survivorship-yanlı (PEAD bu testte deflate oldu — §16; aynısı beklenir).")


def main():
    ap = argparse.ArgumentParser(description="Insider (EDGAR Form 4) izole ölçüm + PIT survivorship")
    ap.add_argument("--pit", action="store_true", help="survivorship-temiz PIT evreni modu")
    # BASKET modu
    ap.add_argument("--periods", default="2y,5y", help="virgülle: 1y,2y,5y")
    ap.add_argument("--interval", default="1d", help="insider yalnızca 1d'de anlamlı")
    ap.add_argument("--compare", default="event_breakout",
                    help="kıyas stratejisi (fiyat-göstergesi referansı)")
    # PIT modu
    ap.add_argument("--as-of", default="2022-01-01", help="(--pit) evren üyelik tarihi = test başı")
    ap.add_argument("--end", default=None, help="(--pit) YYYY-MM-DD (varsayılan: bugün)")
    ap.add_argument("--limit", type=int, default=60, help="(--pit) ilk N sembol (0=hepsi, yavaş)")
    ap.add_argument("--throttle", type=float, default=0.0, help="(--pit) sembol arası ek bekleme")
    args = ap.parse_args()
    if args.pit:
        run_pit(args)
    else:
        run_basket(args)


if __name__ == "__main__":
    main()
