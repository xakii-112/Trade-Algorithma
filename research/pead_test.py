"""
pead_test.py — PEAD (Post-Earnings Announcement Drift) sinyalini İZOLE ölçer.

EDGE_FINDINGS metodolojisi: önce ölç → iki bağımsız periyotta doğrula → al-tut
(beta) kontrolüyle kıyasla. PEAD fiyat-göstergesi DEĞİL, temel/olay-bazlı bir
anomali (detect_pead); bu yüzden fiyat-göstergesi ailesinden bağımsız bir kenar
kaynağı olabilir. Bu harness onun gerçekten kenarı var mı, yoksa boğa-betası mı
olduğunu ayırır.

Önemli: PEAD yalnızca GÜNLÜK (1d) barda anlamlı (drift günler-haftalar sürer) ve
yfinance Ticker.earnings_dates + Surprise% gerektirir (sembol-başı cache'li).

    python pead_test.py                       # BASKET, 1d, 2y+5y, vs al-tut + event_breakout
    python pead_test.py --periods 1y,2y,5y
    python pead_test.py --surprise-min 3.0    # daha seçici sürpriz eşiği
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

from nihai.execution.backtest import backtest_symbol, load_bars
from nihai.universe import BASKET


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


def run_period(period: str, interval: str, mode: str) -> tuple[dict | None, float, int]:
    """Sepeti tek tek backtest et; toplu PnL metriği + ortalama al-tut getirisi döndür.

    al-tut = (son_close/ilk_close − 1)·100, sembol başına; PEAD getirisini piyasa
    betasından ayırmak için kontrol. PEAD ancak al-tut'u aşan bir şey üretirse
    'kenar' sayılır (yoksa sadece boğa rejimini biniyordur)."""
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


def main():
    ap = argparse.ArgumentParser(description="PEAD izole ölçüm (vs al-tut + event_breakout)")
    ap.add_argument("--periods", default="2y,5y", help="virgülle: 1y,2y,5y")
    ap.add_argument("--interval", default="1d", help="PEAD yalnızca 1d'de anlamlı")
    ap.add_argument("--compare", default="event_breakout",
                    help="kıyas stratejisi (fiyat-göstergesi referansı)")
    args = ap.parse_args()
    periods = [p.strip() for p in args.periods.split(",")]

    print(f"=== PEAD izole ölçüm | {args.interval} | sepet={len(BASKET)} sembol ===")
    print("Hedef: iki periyotta da PEAD profit_factor>1 VE expectancy>0 (dayanıklı kenar).")
    print("Beta kontrolü: PEAD toplam getirisi ort. al-tut'un altındaysa = sadece piyasa betası.\n")

    for per in periods:
        print(f"{'='*64}\n--- PERİYOT: {per} ---\n{'='*64}")
        pead_m, bh, n_sym = run_period(per, args.interval, "pead")
        cmp_m, _, _ = run_period(per, args.interval, args.compare)

        print(f"  [{n_sym} sembol]  ortalama AL-TUT getirisi: {bh:+.1f}%\n")
        for label, m in (("PEAD", pead_m), (args.compare, cmp_m)):
            print(f"  {label}:")
            if not m:
                print("    (işlem yok)")
                continue
            for k, v in m.items():
                print(f"    {k:<16} {v}")
            print()


if __name__ == "__main__":
    main()
