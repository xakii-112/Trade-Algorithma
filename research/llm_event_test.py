"""
llm_event_test.py — LLM (Claude) 8-K olay-çıkarımı ADAY sinyalini İZOLE ölç (C3).

EDGE_FINDINGS metodolojisi (docs/EDGE_FINDINGS.md §13.5/§16): önce ölç → iki periyot →
al-tut beta → POINT-IN-TIME survivorship. KENAR İDDİASI YOK.

insider_test.py / sentiment_test.py / pead_pit_test.py ile AYNI iskelet. Reddit sentiment'ten
FARKI: 8-K full-text TARİHSEL ve point-in-time'dır (filing tarihi = halka açıklık) → soğuk
başlangıçta bile geçmişe gidilir, forward-accumulation gerekmez. Yani bu harness ilk çağrıda da
gerçek işlem üretebilir (ANTHROPIC_API_KEY + SEC_USER_AGENT varsa).

MALİYET UYARISI: her yeni 8-K = 1 LLM çağrısı (accession başına disk-cache'li → bir kez).
PIT modunda yüzlerce sembol × yıllar = pahalı olabilir; --limit ve --throttle ile sınırla,
maliyet-duyarlıysa .env'de LLM_EVENT_MODEL=claude-haiku-4-5. Tur-11 saha notları:
  - Gemini ÜCRETSİZ katman günde ~20 istek (429 RESOURCE_EXHAUSTED) → PIT ölçümü için
    yetersiz; Anthropic (Haiku) anahtarı veya ücretli katman gerekir.
  - Fiyat tarafında toplu yfinance 5y çekimi bot-korumasına takılır → PIT koşusunu
    DATA_PROVIDER=alpaca_hist ile çalıştır (SIP, delist'ler dahil, disk-cache'li).
  - (Ledger Tur-11) filings-başına üst sınır bayrağı henüz YOK; eklenene dek maliyet
    kontrolü --limit (sembol) + LLM günlük kotasıyla sınırlıdır.

  python -m research.llm_event_test                       # BASKET, 2 periyot + al-tut beta
  python -m research.llm_event_test --periods 1y,2y
  DATA_PROVIDER=alpaca_hist python -m research.llm_event_test --pit --as-of 2022-01-01 --limit 30
  python -m research.llm_event_test --coverage            # birikmiş çıkarım kapsamasını raporla
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
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = round(sum(wins) / gl, 2) if gl > 0 else None
    print(f"  {label:<22} işlem={len(pnls):<4} WR={100*len(wins)/len(pnls):>5.1f}% "
          f"PF={pf} E={statistics.mean(pnls):>+10.2f}")


def report_coverage() -> None:
    """Birikmiş LLM-çıkarım kapsamasını raporla (kaç sembol, kaç materyal olay)."""
    from nihai.data.llm_event import llm_events
    have = total = material = 0
    for s in BASKET:
        ev = llm_events(s)
        if ev is not None and not ev.empty:
            have += 1
            total += len(ev)
            mat = int((ev["score"].abs() >= 0.4).sum())
            material += mat
            print(f"  {s:<6} {len(ev):>4} olay  {mat:>3} materyal  (son: {ev.index[-1]})")
    print(f"\nLLM olayı olan sembol: {have}/{len(BASKET)} | toplam gün: {total} | materyal: {material}")
    if have == 0:
        print("Henüz çıkarım YOK — `python -m nihai.data.llm_event SYM` ile biriktir")
        print("(ANTHROPIC_API_KEY + SEC_USER_AGENT .env'de olmalı).")


# ---------------------------------------------------------------------------
# Mod 1: BASKET — 2 periyot + al-tut beta + kıyas
# ---------------------------------------------------------------------------
def run_period(period: str, interval: str, mode: str) -> tuple[dict | None, float, int]:
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
    print(f"=== LLM-EVENT izole ölçüm | {args.interval} | sepet={len(BASKET)} sembol ===")
    print("Hedef: iki periyotta da PF>1 VE E>0 (dayanıklı aday) + al-tut'u aşmalı.")
    print("‼ ANTHROPIC_API_KEY yoksa çıkarım devre-dışı → işlem=0 (beklenen).\n")
    for per in periods:
        print(f"{'='*64}\n--- PERİYOT: {per} ---\n{'='*64}")
        m, bh, n_sym = run_period(per, args.interval, "llm_event")
        cmp_m, _, _ = run_period(per, args.interval, args.compare)
        print(f"  [{n_sym} sembol]  ortalama AL-TUT getirisi: {bh:+.1f}%\n")
        for label, mm in (("LLM_EVENT", m), (args.compare, cmp_m)):
            print(f"  {label}:")
            if not mm:
                print("    (işlem yok — çıkarım birikmemiş ya da API anahtarı yok)")
                continue
            for k, v in mm.items():
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
    print(f"=== LLM-EVENT point-in-time | S&P500 @ {args.as_of} | {n} sembol "
          f"({len(removed)} bugün YOK) | test {args.as_of}→{args.end or 'bugün'} ===")
    print("‼ ANTHROPIC_API_KEY yoksa çıkarım devre-dışı → işlem=0 (beklenen).")
    print("‼ MALİYET: her yeni 8-K = 1 Claude çağrısı (cache'li). --max-filings ile sınırla.\n")

    all_pnl: list[float] = []
    removed_pnl: list[float] = []
    survivor_pnl: list[float] = []
    bh: list[float] = []
    n_with_data = n_with_trades = 0

    for i, s in enumerate(universe, 1):
        try:
            df = load_bars(s, "1d", start=args.as_of, end=args.end)
            if df.empty or len(df) < 60:
                continue
            n_with_data += 1
            first, last = float(df["close"].iloc[0]), float(df["close"].iloc[-1])
            if first > 0:
                bh.append((last / first - 1) * 100)
            r = backtest_symbol(s, interval="1d", df=df, strategy_mode="llm_event")
            if r.trade_pnls:
                n_with_trades += 1
                all_pnl.extend(r.trade_pnls)
                (removed_pnl if s in removed else survivor_pnl).extend(r.trade_pnls)
        except Exception:
            pass
        if i % 25 == 0:
            print(f"  [{i}/{n}] veri={n_with_data} işlemli={n_with_trades} "
                  f"llm-işlem={len(all_pnl)}", flush=True)
        if args.throttle > 0:
            time.sleep(args.throttle)

    print(f"\n{'='*64}")
    print(f"COVERAGE: {n_with_data}/{n} sembolde fiyat | {n_with_trades} sembolde LLM işlemi")
    print(f"  ortalama al-tut (beta kontrolü): {statistics.mean(bh):+.1f}%" if bh else "")
    print(f"{'='*64}")
    _agg_line(all_pnl, "TÜM PIT evreni")
    _agg_line(survivor_pnl, "hayatta-kalanlar")
    _agg_line(removed_pnl, "removed-since")
    print(f"{'='*64}")
    print("Yorum: TÜM PIT evreni PF>1+E>0 + al-tut'u aşma → dayanıklı aday. 'removed-since' ile")
    print("'hayatta-kalanlar' BENZER ise survivorship'ten arınık; ayrışıyorsa kenar artefakt.")


def main():
    ap = argparse.ArgumentParser(description="LLM (Claude) 8-K olay izole ölçüm + PIT survivorship")
    ap.add_argument("--pit", action="store_true", help="survivorship-temiz PIT evreni modu")
    ap.add_argument("--coverage", action="store_true", help="yalnız birikmiş çıkarım kapsamasını raporla")
    # BASKET modu
    ap.add_argument("--periods", default="2y,5y", help="virgülle: 1y,2y,5y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--compare", default="event_breakout", help="kıyas (fiyat-göstergesi referansı)")
    # PIT modu
    ap.add_argument("--as-of", default="2022-01-01", help="(--pit) evren üyelik tarihi = test başı")
    ap.add_argument("--end", default=None, help="(--pit) YYYY-MM-DD (varsayılan: bugün)")
    ap.add_argument("--limit", type=int, default=30, help="(--pit) ilk N sembol (0=hepsi)")
    ap.add_argument("--throttle", type=float, default=0.0)
    args = ap.parse_args()
    if args.coverage:
        report_coverage()
    elif args.pit:
        run_pit(args)
    else:
        run_basket(args)


if __name__ == "__main__":
    main()
