"""
screen.py — Toplu (universe) backtest tarayıcı.

Bir hisse evrenini (Nasdaq-100 / S&P 500 / dosya) sırayla backtest eder ve
sonuçları expectancy / Sharpe / drawdown'a göre sıralı bir CSV'ye yazar.

Önemli:
- Veri yfinance'ten gelir → her sembol ayrı HTTP isteği. Ban'a karşı --throttle ile
  araya bekleme konur (Madde 48 farkındalığı). Büyük evrenler dakikalar sürer.
- Derinlik/order-book verisi YOKTUR; yalnızca OHLCV bar.
- 1h geçmişi yfinance'te ~730 günle sınırlıdır; 6 ay sorun değildir.

Kullanım:
    python screen.py --universe nasdaq100 --interval 1h --period 6mo
    python screen.py --universe sp500 --interval 1d --period 1y --top 30
    python screen.py --symbols-file watch.txt --interval 1h --period 6mo --mc 200
    python screen.py --universe nasdaq100 --limit 5            # hızlı deneme
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import pandas as pd
from loguru import logger

from nihai import net
from nihai.config import INITIAL_EQUITY, LOG_DIR
from nihai.execution.backtest import backtest_symbol
from nihai.signals.strategy import STRATEGY_MODES

# --- Nasdaq-100 yedek liste (Wikipedia çekilemezse; zamanla bayatlayabilir) ---
# Güncellenme: 2026-06-08 (Wikipedia Nasdaq-100, 101 sembol). Delisted isimler
# (ör. ANSS→Synopsys, WBA→özelleştirme) bilinçli olarak listede DEĞİL — yedek
# liste devreye girdiğinde ölü ticker taranıp failover'ı boşuna tetiklemesin.
_NASDAQ100_FALLBACK = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "ALNY", "AMAT", "AMD",
    "AMGN", "AMZN", "APP", "ARM", "ASML", "AVGO", "AXON", "BKNG", "BKR", "CCEP",
    "CDNS", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSX", "CTAS",
    "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG", "FAST", "FER", "FTNT",
    "GEHC", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INSM", "INTC", "INTU", "ISRG",
    "KDP", "KHC", "KLAC", "LIN", "LITE", "LRCX", "MAR", "MCHP", "MDLZ", "MELI",
    "META", "MNST", "MPWR", "MRVL", "MSFT", "MSTR", "MU", "NFLX", "NVDA", "NXPI",
    "ODFL", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PLTR", "PYPL", "QCOM",
    "REGN", "ROP", "ROST", "SBUX", "SHOP", "SNDK", "SNPS", "STX", "TMUS", "TRI",
    "TSLA", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "WDC", "WMT", "XEL",
    "ZS",
]


def _read_html_tables(url: str) -> list[pd.DataFrame]:
    """Wikipedia'dan tabloları çek, sonra parse et.

    Wikimedia'nın robot politikası (w.wiki/4wJS) jenerik/eksik User-Agent'lı
    istekleri 403 ile reddeder ve 141 baytlık "respect our robot policy" sayfası
    döner — bu durumda read_html "No tables found" hatası verir. Politika
    iletişim bilgisi içeren bir UA ister; aşağıdaki UA bunu sağlar (200 döner).
    """
    headers = {
        "User-Agent": net.contact_ua("NihaiAlgoBacktest/1.0"),
        "Accept": "text/html,application/xhtml+xml",
    }
    resp = net.shared_client().get(url, headers=headers, timeout=20,
                                   follow_redirects=True)
    resp.raise_for_status()  # 403/5xx'i sessiz "No tables" yerine açık hata olarak yüzeye çıkar
    # flavor="lxml": html5lib bağımlılığı olmadan parse (lxml zaten requirements'ta)
    return pd.read_html(StringIO(resp.text), flavor="lxml")


def _extract_ticker_col(tables: list[pd.DataFrame]) -> list[str]:
    for t in tables:
        for col in t.columns:
            name = str(col).lower()
            if "ticker" in name or "symbol" in name:
                vals = [str(s).strip().upper().replace(".", "-") for s in t[col].tolist()]
                vals = [v for v in vals if v and v != "NAN" and len(v) <= 6]
                if len(vals) >= 50:
                    return vals
    return []


# BULGU-69-1 kilidi (Tur-70) — İKİ liste de SESSİZCE degrade olur: nasdaq100 bayat yedeğe
# düşer (yalnız WARNING), sp500 boş döner (ERROR). CLAUDE.md kuralı: durum yüzeyi yazan tur
# onu OKUYAN yüzeyi de AYNI turda bağlamalı. O log satırı 7/24 tarayan bir sürecin
# system.log'unda pratikte görünmez — kalıcı yüzey burasıdır:
# /api/system/health → `universe` → pano "Evren Kaynakları" kartı.
# Kilit: tests/test_tur70_evren_yuzeyi.py. Yazan ile guard AYNI fonksiyondadır, çünkü
# dört çağıran (build_universe, load_symbols, research/build_meta_dataset, research/xsect)
# aynı boşluğu paylaşır; çağırana yazılsaydı üçü açıkta kalırdı (Tur-65 dersi).
_EVREN_DURUM: dict[str, dict[str, object]] = {}


def _durum_yaz(ad: str, n: int, *, canli: bool) -> None:
    _EVREN_DURUM[ad] = {"n": n, "canli": canli, "at": time.time()}


def evren_durum() -> dict[str, dict[str, object]]:
    """Son evren çekiminin sonucu: ``{ad: {n, canli, at}}``.

    ``canli=False`` = DEGRADE: nasdaq100 bayat yedek listesine düştü ya da sp500 boş
    döndü. Boş sözlük = bu süreçte henüz evren kurulmadı (çekim başarısız DEĞİL).

    SÜREÇ-İÇİ: tarayan döngü ile pano AYNI süreçte koşar (``--serve-web`` web
    sunucusunu daemon thread'te başlatıp aynı süreçte ``run_loop``a girer), bu yüzden
    sayılar GERÇEKTİR. Ayrı bir süreçten okunsaydı daima boş görünürdü — yüzeyin
    burada olmasının nedeni budur (Tur-61 ``bridges`` deseninin aynısı)."""
    return {k: dict(v) for k, v in _EVREN_DURUM.items()}


def nasdaq100_tickers() -> list[str]:
    try:
        tickers = _extract_ticker_col(_read_html_tables("https://en.wikipedia.org/wiki/Nasdaq-100"))
        if tickers:
            out = sorted(set(tickers))
            logger.info(f"[screen] Nasdaq-100: {len(out)} sembol (Wikipedia)")
            _durum_yaz("nasdaq100", len(out), canli=True)
            return out
    except Exception as e:
        logger.warning(f"[screen] Wikipedia Nasdaq-100 çekilemedi ({e}); yedek liste kullanılıyor")
    # 200 dönüp tablo <50 satır parse edilen dal da BURAYA düşer (except'e değil) —
    # `canli=False` ikisini de kapsar.
    out = sorted(set(_NASDAQ100_FALLBACK))
    _durum_yaz("nasdaq100", len(out), canli=False)
    return out


def sp500_tickers() -> list[str]:
    try:
        tickers = _extract_ticker_col(
            _read_html_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"))
        if tickers:
            out = sorted(set(tickers))
            logger.info(f"[screen] S&P 500: {len(out)} sembol (Wikipedia)")
            _durum_yaz("sp500", len(out), canli=True)
            return out
    except Exception as e:
        logger.warning(f"[screen] Wikipedia S&P 500 çekilemedi ({e})")
    # ASİMETRİ: nasdaq100_tickers() çekemezse bayat yedek listeye düşer, bu ise [] döner —
    # 'both' evreninde union ~600 yerine 101 isme iner, hiçbir yerde "evren yarılandı"
    # demeden. Guard ÇAĞIRANDA DEĞİL BURADA, çünkü dört çağıran aynı boşluğu paylaşır
    # (auto_screener.build_universe, load_symbols, research/build_meta_dataset, research/
    # xsect) ve ayrıca 200 dönüp tablo <50 satır parse edilen dal buradan başka HİÇBİR
    # yerde loglanmıyordu. Kök neden genellikle iletişim bilgisi taşımayan UA'dır
    # (Wikimedia 403) — bkz. nihai.net.contact_ua.
    logger.error("[screen] S&P 500 evreni BOŞ (Wikipedia 403 / SEC_USER_AGENT?) — "
                 "bu evreni kullanan tarama/veri seti EKSİK kurulur")
    _durum_yaz("sp500", 0, canli=False)
    return []


def load_symbols(args) -> list[str]:
    if args.symbols_file:
        p = Path(args.symbols_file)
        raw = p.read_text(encoding="utf-8").replace(",", "\n").splitlines()
        return sorted({s.strip().upper().replace(".", "-") for s in raw if s.strip()})
    if args.universe == "sp500":
        return sp500_tickers()
    return nasdaq100_tickers()


# Paralel tarama için worker-process config'i. Her sembolde AYNI olan parametreleri
# (interval/period/...) tekrar tekrar pickle'lamamak için initializer ile worker
# global'ine BİR kez taşınır (auto_screener._BENCH_RETURNS ile aynı desen).
_SCREEN_CFG: dict = {}


def _set_screen_cfg(cfg: dict) -> None:
    """Tarama parametrelerini bu process'in global'ine kur (seri yol; BLAS'a dokunmaz —
    seri modda numpy tüm BLAS thread'lerini kullanmalı)."""
    global _SCREEN_CFG
    _SCREEN_CFG = cfg or {}


def _screen_worker_init(cfg: dict) -> None:
    """Process worker başlatıcısı: BLAS/numpy thread havuzunu 1'e sabitle + tarama
    parametrelerini worker global'ine yerleştir.

    Her worker zaten ayrı bir çekirdek kullanır; numpy'nin process-içi çoklu thread
    açması over-subscription yaratır (8 process × 8 BLAS thread → çekişme). Tek-thread'e
    sabitleyince worker'lar çekirdekleri temiz paylaşır (auto_screener._pin_worker_threads
    ile aynı gerekçe)."""
    import os as _os
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS"):
        _os.environ[_v] = "1"
    _set_screen_cfg(cfg)


def _screen_one(sym: str) -> dict | None:
    """Tek sembolü backtest'le → satır dict'i (veri yok/hata → None). Modül-seviye +
    picklable → ProcessPoolExecutor worker olarak güvenli. backtest_symbol durumsuzdur
    (yfinance modül-seviye; hub/alpaca/tv her çağrıda izole sağlayıcı kurar).

    Paralel modda her worker, sembolü bitirdikten sonra throttle kadar bekler → yfinance
    ban'ı önleme (Madde 48): efektif istek hızı backtest CPU süresine göre zaten düşüktür."""
    cfg = _SCREEN_CFG
    try:
        r = backtest_symbol(sym, interval=cfg["interval"], period=cfg["period"],
                            initial_equity=cfg["equity"], mc_iters=cfg["mc"],
                            strategy_mode=cfg["strategy_mode"],
                            provider_name=cfg["provider_name"])
        if r.trades_count == 0 and r.bars_total == 0:
            return None
        if r.total_return_pct is None or r.max_drawdown_pct is None or r.sharpe_annualized is None:
            return None
        ruin = (r.monte_carlo or {}).get("ruin_probability")
        return {
            "symbol": sym, "trades": r.trades_count, "win_rate_pct": round(r.win_rate * 100, 1),
            "expectancy": round(r.expectancy, 2), "total_return_pct": round(r.total_return_pct, 2),
            "max_drawdown_pct": round(r.max_drawdown_pct, 2),
            "sharpe": round(r.sharpe_annualized, 2),
            "ruin_prob": None if ruin is None else round(ruin, 4),
        }
    except Exception as e:
        logger.debug(f"[screen] {sym}: hata {e}")
        return None
    finally:
        thr = cfg.get("throttle", 0.0)
        if thr and thr > 0:
            time.sleep(thr)


def run_screen(symbols, *, interval, period, strategy_mode, equity, mc, throttle,
               provider_name, workers=1, executor="process"):
    """Sembol evrenini backtest'le → satır listesi.

    workers>1 ise semboller PARALEL taranır. backtest CPU-AĞIRLIKLI olduğundan
    (~1.2 sn ağ + ~7.5 sn hesap/sembol) gerçek hızlanma çoklu ÇEKİRDEK (executor=
    'process') ile gelir; 'thread' GIL'e takılır. workers<=1 eski seri davranıştır.
    ~600 sembol: tek çekirdek ~75 dk, 8 çekirdek ~10 dk."""
    n = len(symbols)
    cfg = {"interval": interval, "period": period, "strategy_mode": strategy_mode,
           "equity": equity, "mc": mc, "provider_name": provider_name, "throttle": throttle}

    def _summary(row: dict) -> str:
        return (f"işlem={row['trades']} WR={row['win_rate_pct']}% "
                f"exp={row['expectancy']} getiri={row['total_return_pct']}%")

    # --- Seri yol (workers<=1): eski davranış birebir korunur ---
    if workers <= 1:
        _set_screen_cfg(cfg)   # BLAS'a dokunma; seri modda tam BLAS thread'i istenir
        rows = []
        for i, sym in enumerate(symbols, 1):
            row = _screen_one(sym)   # throttle bekleme _screen_one finally'sinde
            if row is not None:
                rows.append(row)
                logger.info(f"[{i}/{n}] {sym}: {_summary(row)}")
            else:
                logger.warning(f"[{i}/{n}] {sym}: veri yok/atlandı")
        return rows

    # --- Paralel yol (workers>1): process (gerçek çok-çekirdek) veya thread ---
    from concurrent.futures import (
        ProcessPoolExecutor,
        ThreadPoolExecutor,
        as_completed,
    )
    use_process = executor == "process"
    Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    ex_kwargs: dict = {"max_workers": workers}
    if use_process:
        ex_kwargs["initializer"] = _screen_worker_init   # BLAS pin + cfg taşı
        ex_kwargs["initargs"] = (cfg,)
    else:
        # Thread yolu: worker'lar parent process'i paylaşır → cfg'yi burada kur.
        _set_screen_cfg(cfg)
        ex_kwargs["thread_name_prefix"] = "screen"

    rows = []
    done = 0
    try:
        with Executor(**ex_kwargs) as ex:
            futures = {ex.submit(_screen_one, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                done += 1
                sym = futures[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    logger.warning(f"[{done}/{n}] {sym}: worker hatası {e}")
                    continue
                if row is not None:
                    rows.append(row)
                    logger.info(f"[{done}/{n}] {sym}: {_summary(row)}")
                else:
                    logger.warning(f"[{done}/{n}] {sym}: veri yok/atlandı")
    except Exception as e:
        # Process havuzu kurulamazsa (spawn/pickle sorunu) seri yola düş → tarama
        # tümden çökmesin (auto_screener ile aynı dayanıklılık).
        logger.warning(f"[screen] paralel havuz hatası ({e}) — seri yola düşülüyor")
        return run_screen(symbols, interval=interval, period=period,
                          strategy_mode=strategy_mode, equity=equity, mc=mc,
                          throttle=throttle, provider_name=provider_name, workers=1)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Toplu backtest tarayıcı (universe screen)")
    ap.add_argument("--universe", default="nasdaq100", choices=["nasdaq100", "sp500"])
    ap.add_argument("--symbols-file", default=None, help="Her satırda bir ticker (universe yerine)")
    ap.add_argument("--interval", default="1h", choices=["1m", "5m", "15m", "1h", "1d"])
    ap.add_argument("--period", default="6mo", help="örn. 6mo, 1y, 2y")
    # Tur-106 (BULGU-106-12): seçenekler elle yazılmış 7'lik bayat bir alt kümeydi (STRATEGY_MODES 15 kip; quant_trend,
    # trend_pullback, vol_breakout … seçilemiyordu) ve varsayılan (canlı `quant_trend`) 1h'de her sembolde 0 işlem verip
    # sessizce boş rapor basıyordu.
    ap.add_argument("--strategy", default=None, choices=list(STRATEGY_MODES))
    ap.add_argument("--equity", type=float, default=INITIAL_EQUITY)
    ap.add_argument("--mc", type=int, default=0, help="Sembol başına Monte Carlo iter (0=kapalı, yavaşlatır)")
    ap.add_argument("--provider", default="yfinance",
                    choices=["yfinance", "tradingview", "hub", "alpaca", "alpaca_stream"],
                    help="Veri sağlayıcı. Toplu tarama için yfinance önerilir; tradingview "
                         "her sembolde yeniden bağlanır (~10 sn/sembol → çok yavaş). "
                         "hub/alpaca eşzamanlı isteğe dayanıklı → paralel taramada 429 riski yok "
                         "(load_bars hepsini destekler).")
    ap.add_argument("--throttle", type=float, default=0.5, help="Sembol başına bekleme (sn) — ban önleme")
    ap.add_argument("--workers", type=int,
                    default=max(1, min(16, (os.cpu_count() or 4))),
                    help="Paralel tarama worker sayısı (çekirdek). 1=seri. Varsayılan: "
                         "mantıksal çekirdek sayısı (en fazla 16).")
    ap.add_argument("--executor", default="process", choices=["process", "thread"],
                    help="Paralellik motoru. 'process'=gerçek çoklu-çekirdek (CPU-ağırlıklı "
                         "backtest için doğru); 'thread' GIL'e takılır (hızlandırmaz).")
    ap.add_argument("--limit", type=int, default=0, help="İlk N sembolle sınırla (deneme)")
    ap.add_argument("--top", type=int, default=25, help="Konsola yazılacak en iyi N")
    ap.add_argument("--sort", default="expectancy",
                    choices=["expectancy", "sharpe", "total_return_pct", "win_rate_pct"])
    args = ap.parse_args()
    from nihai.runtime_strategy import (
        settings as _strateji,  # generate_signal'ın varsayılanı (pano seçimi > .env)
    )
    if (args.strategy or _strateji()["strategy"]) == "quant_trend" and args.interval != "1d":
        ap.error("quant_trend günlük bar ister (≥253 bar): --interval 1d --period 2y gibi kullanın "
                 "(1h barda her sembol 0 işlem verir).")

    symbols = load_symbols(args)
    if args.limit > 0:
        symbols = symbols[: args.limit]
    if not symbols:
        logger.error("Sembol listesi boş.")
        return

    workers = min(args.workers, len(symbols)) if symbols else 1
    logger.info(f"[screen] {len(symbols)} sembol | {args.interval} | {args.period} | "
                f"strateji={args.strategy or 'config'} | throttle={args.throttle}s | "
                f"worker={workers}×{args.executor} ({args.provider})")
    if workers > 1 and args.provider == "yfinance":
        logger.info("[screen] not: paralel yfinance rate-limit'e girebilir; eşzamanlı "
                    "isteklere dayanıklı bir sağlayıcı için --provider hub/alpaca düşünün.")
    rows = run_screen(symbols, interval=args.interval, period=args.period,
                      strategy_mode=args.strategy, equity=args.equity, mc=args.mc,
                      throttle=args.throttle, provider_name=args.provider,
                      workers=workers, executor=args.executor)

    if not rows:
        logger.error("Hiç sonuç yok (veri çekilemedi olabilir).")
        return

    rows.sort(key=lambda r: (r.get(args.sort) is None, -(r.get(args.sort) or 0)))

    out = LOG_DIR / f"screen_{args.universe}_{args.interval}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 92)
    print(f" EN İYİ {min(args.top, len(rows))} ({args.sort}'e göre) — {len(rows)} sembol tarandı")
    print("=" * 92)
    print(f"{'#':>3} {'SEMBOL':<8} {'İŞLEM':>6} {'WR%':>6} {'EXPECT':>9} {'GETİRİ%':>9} {'MDD%':>7} {'SHARPE':>7}")
    print("-" * 92)
    for i, r in enumerate(rows[: args.top], 1):
        print(f"{i:>3} {r['symbol']:<8} {r['trades']:>6} {r['win_rate_pct']:>6.1f} "
              f"{r['expectancy']:>9.2f} {r['total_return_pct']:>9.2f} "
              f"{r['max_drawdown_pct']:>7.2f} {r['sharpe']:>7.2f}")
    print("=" * 92)
    print(f"\n→ Tam sonuç CSV: {out}\n")


if __name__ == "__main__":
    main()
