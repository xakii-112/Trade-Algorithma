"""
build_meta_dataset.py — Meta-labeling eğitim verisini üretir.

Birincil strateji (varsayılan event_breakout) sepeti backtest eder; her giriş için
look-ahead-güvenli özellik vektörü + nihai kâr/zarar etiketi toplanıp CSV'ye yazılır.
Bu CSV'yi train_meta.py tüketir (zamana göre out-of-sample bölüp modeli doğrular).

    python build_meta_dataset.py                         # BASKET, 1h, 2y, event_breakout
    python build_meta_dataset.py --interval 1h --period 2y --mode event_breakout
    python build_meta_dataset.py --universe nasdaq100    # daha çok veri (yavaş, ~100 sembol)
"""
from __future__ import annotations

import argparse
import csv
import sys

import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.config import DATA_DIR
from nihai.execution.backtest import backtest_symbol, load_bars
from nihai.signals.meta_features import FEATURE_NAMES, TradeRecorder
from nihai.universe import BASKET

EXTRA_COLS = ["pnl", "label", "symbol", "period", "entry_time", "bars_held", "mae_pct", "mfe_pct", "removed_since"]


def _year_windows(df, warmup_bars: int = 60):
    """df'i takvim yıllarına böl; her pencereye önceki ~warmup_bars barı ekle (gösterge
    ısınması). Neden: çok-yıllık TEK bileşik kasada kaybeden strateji kasayı eritip
    girişleri süresiz durduruyor (gözlem: 10y 1h PIT'te satırların %95'i 2016; AAL 9
    işlemde öldü) → örnekleme kayıp-rejimi-sansürlü olur. Her yıl taze kasa = her yıl
    örneklenir; pencere içi risk gerçekçiliği korunur. Yield: (yıl, sub_df)."""
    years = sorted(df.index.year.unique())
    for y in years:
        end_i = df.index.searchsorted(pd.Timestamp(f"{y + 1}-01-01"))
        start_i = df.index.searchsorted(pd.Timestamp(f"{y}-01-01"))
        sub = df.iloc[max(start_i - warmup_bars, 0):end_i]
        if len(sub) >= 80:
            yield y, sub


def build(symbols: list[str], *, interval: str, period: str, mode: str,
          out: str, throttle: float = 0.3, start: str | None = None,
          removed: set[str] | None = None, provider: str | None = None,
          kill_switch: bool = False, yearly_windows: bool = False) -> int:
    """Sembolleri tara, her sembolden SONRA CSV'ye artımlı ekle (kill'e dayanıklı,
    ilerleme izlenebilir). Toplam satır sayısını döndürür.

    start verilirse (PIT modu) period yerine start→bugün penceresi yüklenir; removed
    kümesindeki semboller satıra `removed_since=1` etiketlenir (survivorship analizi).
    provider verilirse load_bars o sağlayıcıyı kullanır (örn. alpaca_hist: SIP feed,
    delist edilmiş semboller dahil → PIT evrende yfinance'ten daha eksiksiz)."""
    import time
    removed = removed or set()
    cols = FEATURE_NAMES + EXTRA_COLS
    total = 0
    n = len(symbols)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for i, s in enumerate(symbols, 1):
            try:
                df = (load_bars(s, interval, start=start, provider_name=provider) if start
                      else load_bars(s, interval, period=period, provider_name=provider))
                if df.empty or len(df) < 80:
                    print(f"[{i}/{n}] {s}: veri yetersiz, atlandı", flush=True)
                    continue
                # kill_switch=False (varsayılan): Madde 20 kilidi çok-yıllık pencerede
                # kalıcı tetiklenip örneklemeyi kayıp rejimlerinde sansürlüyordu
                # (10y 1h PIT: satırların %95'i 2016'ya sıkıştı). Eğitim seti işlem-bazlı
                # etiket ister → dataset üretiminde kapalı; --kill-switch ile eski davranış.
                if yearly_windows:
                    # _year_windows: her yıl taze kasa (bileşik-iflas sansürünü kırar).
                    pieces = [(f"{y}", sub) for y, sub in _year_windows(df)]
                else:
                    pieces = [(period, df)]
                rows: list[dict] = []
                trades = 0
                for plabel, sub in pieces:
                    rec = TradeRecorder(symbol=s, period=plabel)
                    r = backtest_symbol(s, interval=interval, df=sub, strategy_mode=mode,
                                        recorder=rec, disable_kill_switch=not kill_switch)
                    trades += r.trades_count
                    rows.extend(rec.rows)
                rs = 1 if s in removed else 0
                for row in rows:
                    row["removed_since"] = rs
                w.writerows(rows)
                f.flush()
                total += len(rows)
                print(f"[{i}/{n}] {s}: {trades} işlem, {len(rows)} satır "
                      f"(toplam {total})", flush=True)
            except Exception as e:
                print(f"[{i}/{n}] {s}: hata {e}", flush=True)
            if throttle > 0:
                time.sleep(throttle)
    return total


def main():
    ap = argparse.ArgumentParser(description="Meta-labeling eğitim verisi üret")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--mode", default="event_breakout", help="birincil strateji (sinyal kaynağı)")
    ap.add_argument("--universe", default="basket", choices=["basket", "nasdaq100", "sp500", "pit_sp500"])
    ap.add_argument("--as-of", default=None, help="PIT modu: bu tarihteki S&P500 üyeliği (çıkarılanlar dahil), start=as-of")
    ap.add_argument("--provider", default=None,
                    help="veri sağlayıcı override (örn. alpaca_hist: SIP + delist'ler dahil; "
                         "boş → env DATA_PROVIDER/yfinance)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--kill-switch", action="store_true",
                    help="Madde 20 kill-switch'i AÇIK tut (eski davranış; çok-yıllık "
                         "pencerede örneklemi kayıp rejimlerinde sansürler)")
    ap.add_argument("--yearly-windows", action="store_true",
                    help="her sembolü bağımsız 1'er yıllık pencerelerde koş (taze kasa; "
                         "çok-yıllık bileşik-iflas sansürünü kırar — PIT 10y için önerilir)")
    args = ap.parse_args()

    start = None
    removed: set[str] = set()
    if args.universe == "pit_sp500" or args.as_of:
        # Survivorship-temiz evren (§13.5): as-of tarihindeki üyelik, çıkarılanlar dahil.
        # yfinance 1h ≤730g → PIT (>2y geçmiş) yalnızca 1d. 1h istenirse uyar.
        as_of = args.as_of or "2022-01-01"
        # 1d zorlaması yalnız yfinance kısıtı (1h ≤730g). alpaca_hist SIP start/end
        # native destekler ve 2016'ya kadar 1h verir → intraday PIT mümkün (sonda:
        # data/alpaca_coverage_probe.csv, removed-since %100 kapsama).
        if args.interval != "1d" and args.provider not in ("alpaca_hist", "alpaca_history"):
            print(f"UYARI: PIT modu yfinance kısıtıyla 1d gerektirir (1h ≤730g). "
                  f"interval={args.interval} → 1d'ye çevriliyor "
                  f"(intraday PIT için --provider alpaca_hist kullan).")
            args.interval = "1d"
        from nihai.data.pit_universe import _current_and_changes, _norm, sp500_as_of
        symbols = sp500_as_of(as_of)
        cur, _ = _current_and_changes()
        removed = set(symbols) - {_norm(s) for s in cur}
        start = as_of
        args.universe = "pit_sp500"
        uni_label = f"pit_sp500@{as_of} ({len(removed)} removed-since)"
    elif args.universe == "basket":
        symbols = BASKET
        uni_label = "basket"
    else:
        from nihai.execution.screen import nasdaq100_tickers, sp500_tickers
        symbols = nasdaq100_tickers() if args.universe == "nasdaq100" else sp500_tickers()
        uni_label = args.universe

    tag = f"{args.as_of}" if start else args.period
    ptag = f"_{args.provider}" if args.provider else ""
    out = args.out or str(DATA_DIR / f"meta_dataset_{args.universe}_{args.mode}_{args.interval}_{tag}{ptag}.csv")
    print(f"=== Meta-dataset üretimi | {uni_label} ({len(symbols)}) | "
          f"{args.interval}/{tag} | birincil={args.mode}"
          f"{' | sağlayıcı=' + args.provider if args.provider else ''} ===")
    print(f"→ {out} (artımlı yazılıyor)\n")
    total = build(symbols, interval=args.interval, period=args.period, mode=args.mode,
                  out=out, start=start, removed=removed, provider=args.provider,
                  kill_switch=args.kill_switch, yearly_windows=args.yearly_windows)

    print(f"\n{'='*56}")
    print(f"Toplam {total} işlem satırı yazıldı")
    print(f"→ {out}")
    print(f"{'='*56}")


if __name__ == "__main__":
    main()
