"""
auto_screener.py — Dinamik watchlist için otomatik evren tarayıcı.

Sistem, bir hisse evrenini (Nasdaq-100 + S&P 500 birleşik) periyodik olarak
backtest'ler, en iyi N skoru seçer ve canlı watchlist'i (watchlist.py) yeniler.
"Tamamen dinamik top-N" mod: her tarama listeyi taze skorlarla değiştirir.

main döngüsünden arka plan thread'i olarak başlatılır (tarama uzun sürer →
60 sn'lik sinyal döngüsünü bloke etmemeli). screen.py'deki universe çekme ve
backtest_symbol mantığı yeniden kullanılır (tek kaynak).

Kullanım (bağımsız):
    python auto_screener.py            # tek tarama yap, watchlist'i güncelle, çık
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from loguru import logger

import nihai.execution.watchlist as watchlist
from nihai.config import (
    AUTO_SCREEN_EXECUTOR,
    AUTO_SCREEN_INTERVAL,
    AUTO_SCREEN_INTERVAL_HOURS,
    AUTO_SCREEN_MAX_RUIN,
    AUTO_SCREEN_MC,
    AUTO_SCREEN_MIN_EXPECTANCY,
    AUTO_SCREEN_MIN_TRADES,
    AUTO_SCREEN_PERIOD,
    AUTO_SCREEN_PROVIDER,
    AUTO_SCREEN_SORT,
    AUTO_SCREEN_THROTTLE,
    AUTO_SCREEN_TOP_N,
    AUTO_SCREEN_UNIVERSE,
    AUTO_SCREEN_WORKERS,
    AUTO_SCREEN_WRITE_CSV,
    CONFIDENCE_COMFORT_TRADES,
    CONFIDENCE_LOW_MULT,
    DUAL_TIER_ENABLED,
    FOCUSED_TOP_N,
    INITIAL_EQUITY,
    LOG_DIR,
    SCREEN_HYBRID_ENABLED,
    SCREEN_HYBRID_SETUP_WEIGHT,
    SCREEN_RS_BENCHMARK,
    SCREEN_RS_LONG_DAYS,
    SCREEN_RS_SHORT_DAYS,
    SCREEN_RUNUP_DAYS,
)
from nihai.execution.backtest import backtest_symbol, load_bars
from nihai.execution.screen import nasdaq100_tickers, sp500_tickers

# B7/B8 — hibrit skorlama için benchmark (SPY) ufuk→getiri sözlüğü. Tarama başına BİR
# kez hesaplanır; process worker'larına initializer ile, seri/thread yolunda doğrudan
# bu modül global'i üzerinden ulaşılır (df pickle'lamaktan kaçınmak için sadece float'lar).
_BENCH_RETURNS: dict[int, float] = {}

# AUTO_SCREEN_INTERVAL için yaklaşık işlem-günü başına bar sayısı (RTH). RS/runup ufukları
# işlem-günü cinsinden tanımlı; burada bar'a çevrilir → interval değişse de anlam korunur.
_BARS_PER_DAY = {"1m": 390, "2m": 195, "5m": 78, "15m": 26, "30m": 13,
                 "60m": 7, "1h": 7, "90m": 5, "1d": 1, "5d": 0.2, "1wk": 0.2}


def _bars(days: int, *, floor: int) -> int:
    """İşlem-günü → bar (AUTO_SCREEN_INTERVAL'e göre), en az `floor`."""
    bpd = _BARS_PER_DAY.get(AUTO_SCREEN_INTERVAL, 7)
    return max(floor, int(round(days * bpd)))


def _rs_horizons_bars() -> tuple[int, int]:
    """B8 — görel-güç ufukları (kısa, uzun) bar cinsinden. Parent ve worker AYNI config'ten
    hesapladığı için bench_returns anahtarlarıyla setup-skoru ufukları uyumlu kalır."""
    return (_bars(SCREEN_RS_SHORT_DAYS, floor=2), _bars(SCREEN_RS_LONG_DAYS, floor=3))


def _confidence_risk_mult(trades: int) -> float:
    """A5 — işlem sayısı 'rahat' eşiğin altındaysa düşük-güven çarpanı, değilse 1.0.
    (Boyutlamaya UYGULANMASI CONFIDENCE_SIZING_ENABLED ile kapılıdır — burada yalnızca
    bilgi amaçlı meta üretilir.)"""
    return CONFIDENCE_LOW_MULT if trades < CONFIDENCE_COMFORT_TRADES else 1.0


# D7 — tur-içi memo. scan_and_update bir turda build_universe'ü İKİ kez çağırıyordu
# (dual-tier: scan_universe içinde + 'rest' hesabında) → tur başına 4 Wikipedia fetch.
# Memo turda bir-kez fetch'e indirir; scan_and_update her turun BAŞINDA memoyu temizler
# → yeni tur taze liste çeker (index üyelik değişiklikleri yakalanır).
_UNIVERSE_MEMO: dict[str, list[str]] = {}


def build_universe(name: str) -> list[str]:
    """nasdaq100 | sp500 | both → sembol listesi (tekilleştirilmiş; D7 tur-içi memolu)."""
    name = (name or "both").lower()
    cached = _UNIVERSE_MEMO.get(name)
    if cached is not None:
        return cached
    if name == "nasdaq100":
        result = nasdaq100_tickers()
    elif name == "sp500":
        result = sp500_tickers()
    else:  # both — boş S&P 500 dönüşünü sp500_tickers() KENDİ İÇİNDE ERROR'lar (screen.py);
        # guard orada çünkü aynı boşluğu dört çağıran birden paylaşıyor.
        result = sorted(set(nasdaq100_tickers()) | set(sp500_tickers()))
    _UNIVERSE_MEMO[name] = result
    return result


def olculemeyen_strateji() -> str | None:
    """Aktif strateji tarayıcının bar aralığında İŞLEM ÜRETEBİLİR mi? Üretemiyorsa nedeni.

    Tur-106 (BULGU-106-4): tarama `backtest_symbol`'ü aktif stratejiyle koşar; `quant_trend` günlük
    bar + ≥253 bar ister (`detect_quant_trend`) ve 1h/6 ay barda her sembolde 0 işlem verir.
    Ana süreç quant koduyla yeniden başladığından (2026-09-27) beri her tarama "513 sonuç, 0
    filtreyi geçti" yazdı: 516 backtest Alpaca bütçesini ~1 dk tüketip canlı sembolleri 429'a
    (yfinance yedeğine) düşürüyor, sonuç atılıyordu; boş sonuç tarama anını da işaretlemediği için
    Tur-91'in kapalı-piyasa ertelemesi hiç tutmuyor, tarama 4 saatte bir yineleniyordu."""
    from nihai.runtime_strategy import (
        settings,  # backtest_symbol → generate_signal ile AYNI kaynak (pano > .env)
    )
    if settings()["strategy"] == "quant_trend" and AUTO_SCREEN_INTERVAL != "1d":
        return (f"aktif strateji quant_trend günlük bar ister, {AUTO_SCREEN_INTERVAL}/"
                f"{AUTO_SCREEN_PERIOD} taramada her sembol 0 işlem verir")
    return None


def _passes_filters(r: dict) -> bool:
    """Eleme: yeterli işlem + min expectancy + max ruin (MC açıksa)."""
    if r["trades"] < AUTO_SCREEN_MIN_TRADES:
        return False
    if r["expectancy"] < AUTO_SCREEN_MIN_EXPECTANCY:
        return False
    if AUTO_SCREEN_MAX_RUIN > 0:
        ruin = r.get("ruin_prob")
        if ruin is not None and ruin > AUTO_SCREEN_MAX_RUIN:
            return False
    return True


def _pin_worker_threads() -> None:
    """Process worker başlatıcısı: BLAS/numpy thread havuzunu 1'e sabitler.
    Her worker zaten ayrı bir çekirdek kullanır; ek olarak numpy'nin process-içi
    çoklu thread açması over-subscription yaratır (8 process × 8 BLAS thread = 64
    thread / 8 çekirdek → çekişme). Tek-thread'e sabitleyince worker'lar çekirdekleri
    temiz paylaşır. Ölçüldü: küçük ama tutarlı kazanç + 16-worker'da thread patlamasını
    önler. Not: BLAS import'tan ÖNCE etkili olması için env burada set edilir; numpy
    bazı platformlarda import sonrası okumayabilir, bu yüzden best-effort."""
    import os as _os
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS"):
        _os.environ[_v] = "1"


def _worker_init(bench_returns: dict) -> None:
    """Process worker başlatıcısı (hibrit modda): BLAS thread'lerini sabitle + benchmark
    getirilerini worker global'ine yerleştir (B7/B8). Her worker ayrı process olduğundan
    _BENCH_RETURNS modül global'i parent'tan miras kalmaz; initializer ile taşınır."""
    _pin_worker_threads()
    global _BENCH_RETURNS
    _BENCH_RETURNS = bench_returns or {}


def _scan_one(sym: str) -> dict | None:
    """Tek sembolü backtest'le → satır dict'i (veya atlanırsa None). Thread-güvenli:
    backtest_symbol→load_bars durumsuzdur (yfinance modül-seviye; hub/alpaca/tv her
    çağrıda kendi izole sağlayıcısını kurar), paylaşılan mutable durum yoktur."""
    try:
        # B7 — hibritte barı BİR kez yükle, hem backtest'e (df=) hem setup-skoruna ver
        # (çift fetch yok). Hibrit kapalıyken eski yol: backtest kendi barını çeker.
        df = None
        if SCREEN_HYBRID_ENABLED:
            df = load_bars(sym, AUTO_SCREEN_INTERVAL, period=AUTO_SCREEN_PERIOD,
                           provider_name=AUTO_SCREEN_PROVIDER)
            if df is None or df.empty:
                return None
        r = backtest_symbol(sym, interval=AUTO_SCREEN_INTERVAL, period=AUTO_SCREEN_PERIOD,
                            initial_equity=INITIAL_EQUITY, mc_iters=AUTO_SCREEN_MC,
                            provider_name=AUTO_SCREEN_PROVIDER, df=df)
        if r.trades_count == 0 and r.bars_total == 0:
            return None
        if r.total_return_pct is None or r.max_drawdown_pct is None or r.sharpe_annualized is None:
            return None
        ruin = (r.monte_carlo or {}).get("ruin_probability")
        row = {
            "symbol": sym, "trades": r.trades_count,
            "win_rate_pct": round(r.win_rate * 100, 1),
            "expectancy": round(r.expectancy, 2),
            "total_return_pct": round(r.total_return_pct, 2),
            "max_drawdown_pct": round(r.max_drawdown_pct, 2),
            "sharpe": round(r.sharpe_annualized, 2),
            "ruin_prob": None if ruin is None else round(ruin, 4),
        }
        if SCREEN_HYBRID_ENABLED:
            from nihai.signals.setup_score import compute_setup_score
            ss = compute_setup_score(df, _BENCH_RETURNS,
                                     runup_bars=_bars(SCREEN_RUNUP_DAYS, floor=3),
                                     rs_horizons=_rs_horizons_bars())
            row["setup_score"] = ss.score
            row["rs_short"] = None if ss.rs_short is None else round(ss.rs_short, 2)
            row["rs_long"] = None if ss.rs_long is None else round(ss.rs_long, 2)
        return row
    except Exception as e:
        logger.debug(f"[auto-screen] {sym}: hata {e}")
        return None


def scan_universe(*, stop_event: threading.Event | None = None,
                  progress: Callable[[int, int, str], None] | None = None) -> list[dict]:
    """
    Evreni tara, her sembolü backtest'le, filtreden geçenleri sıralı döndür.
    stop_event set edilirse erken çıkar (graceful shutdown).

    AUTO_SCREEN_WORKERS>1 ise semboller PARALEL taranır. backtest CPU-ağırlıklı
    olduğundan varsayılan motor "process" (gerçek çoklu-çekirdek); "thread" GIL'e
    takıldığı için CPU-bound işte hızlandırmaz. ~600 sembol: tek çekirdek ~75 dk,
    8 çekirdek ~10 dk.
    """
    from concurrent.futures import (
        ProcessPoolExecutor,
        ThreadPoolExecutor,
        as_completed,
    )

    symbols = build_universe(AUTO_SCREEN_UNIVERSE)
    n = len(symbols)
    workers = min(AUTO_SCREEN_WORKERS, n) if n else 1
    logger.info(f"[auto-screen] {n} sembol taranıyor | evren={AUTO_SCREEN_UNIVERSE} | "
                f"{AUTO_SCREEN_INTERVAL}/{AUTO_SCREEN_PERIOD} | sıralama={AUTO_SCREEN_SORT} | "
                f"worker={workers}×{AUTO_SCREEN_EXECUTOR} ({AUTO_SCREEN_PROVIDER})")
    rows: list[dict] = []

    # B7/B8 — hibritte benchmark (SPY) getirilerini BİR kez hesapla. Seri/thread yolunda
    # modül global'i doğrudan kullanılır; process yolunda initializer ile taşınır.
    global _BENCH_RETURNS
    _BENCH_RETURNS = {}
    if SCREEN_HYBRID_ENABLED:
        try:
            from nihai.signals.setup_score import benchmark_returns
            bench_df = load_bars(SCREEN_RS_BENCHMARK, AUTO_SCREEN_INTERVAL,
                                 period=AUTO_SCREEN_PERIOD, provider_name=AUTO_SCREEN_PROVIDER)
            _BENCH_RETURNS = benchmark_returns(bench_df, _rs_horizons_bars())
            logger.info(f"[auto-screen] HİBRİT: {SCREEN_RS_BENCHMARK} görel-güç referansı "
                        f"hazır (ufuk={_rs_horizons_bars()} bar → {_BENCH_RETURNS})")
        except Exception as e:
            logger.warning(f"[auto-screen] benchmark getirisi alınamadı ({e}) — RS bileşeni boş")

    # --- Seri yol (worker=1): eski davranış, throttle korunur ---
    if workers <= 1:
        for i, sym in enumerate(symbols, 1):
            if stop_event is not None and stop_event.is_set():
                logger.info("[auto-screen] durdurma sinyali — tarama yarıda kesildi")
                break
            row = _scan_one(sym)
            if row is not None:
                rows.append(row)
                if progress:
                    progress(i, n, sym)
            if AUTO_SCREEN_THROTTLE > 0:
                time.sleep(AUTO_SCREEN_THROTTLE)
        # seri yol burada biter; sıralamaya düş
        return _rank(rows)

    # --- Paralel yol (worker>1): process (gerçek çok-çekirdek) veya thread ---
    use_process = AUTO_SCREEN_EXECUTOR == "process"
    Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    ex_kwargs: dict[str, Any] = {"max_workers": workers}
    if use_process:
        # Hibritte benchmark getirilerini de worker'lara taşı (B7/B8); değilse sadece BLAS pin.
        if SCREEN_HYBRID_ENABLED:
            ex_kwargs["initializer"] = _worker_init
            ex_kwargs["initargs"] = (_BENCH_RETURNS,)
        else:
            ex_kwargs["initializer"] = _pin_worker_threads   # BLAS over-subscription'ı önle
    else:
        ex_kwargs["thread_name_prefix"] = "screen"

    done = 0
    try:
        with Executor(**ex_kwargs) as ex:
            # _scan_one modül-seviye (picklable) → process worker olarak güvenli.
            futures = {ex.submit(_scan_one, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                done += 1
                if stop_event is not None and stop_event.is_set():
                    logger.info("[auto-screen] durdurma sinyali — bekleyen işler iptal ediliyor")
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    row = fut.result()
                except Exception as e:
                    logger.debug(f"[auto-screen] {futures[fut]}: worker hatası {e}")
                    continue
                if row is not None:
                    rows.append(row)
                    if progress:
                        progress(done, n, futures[fut])
    except Exception as e:
        # Process havuzu kurulamazsa (örn. spawn/pickle sorunu) seri yola düş — tarama
        # asla tümden çökmesin.
        logger.warning(f"[auto-screen] paralel havuz hatası ({e}) — seri yola düşülüyor")
        rows = [r for r in (_scan_one(s) for s in symbols) if r is not None]

    return _rank(rows)


def _zscores(vals: list[float]) -> list[float]:
    """Basit z-normalizasyon; std≈0 ise hepsi 0 (nötr)."""
    n = len(vals)
    if n == 0:
        return []
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    std = var ** 0.5
    if std < 1e-9:
        return [0.0] * n
    return [(v - mean) / std for v in vals]


def _rank(rows: list[dict]) -> list[dict]:
    """Satırları filtrele + sırala. Hibrit (SCREEN_HYBRID) açıksa sıralama
    z(backtest_metrik) + w·z(setup_score); değilse yalnızca backtest metriği."""
    # sıralama metriği (CSV başlığıyla uyum: 'sharpe' kolonu ≈ 'sharpe_annualized')
    sort_key = "sharpe" if AUTO_SCREEN_SORT == "sharpe" else AUTO_SCREEN_SORT
    kept = [r for r in rows if _passes_filters(r)]

    if SCREEN_HYBRID_ENABLED and kept and any("setup_score" in r for r in kept):
        bt_z = _zscores([float(r.get(sort_key) or 0.0) for r in kept])
        su_z = _zscores([float(r.get("setup_score") or 0.0) for r in kept])
        for r, b, s in zip(kept, bt_z, su_z, strict=False):
            r["combined_score"] = round(b + SCREEN_HYBRID_SETUP_WEIGHT * s, 4)
        kept.sort(key=lambda r: -r.get("combined_score", 0.0))
        logger.info(f"[auto-screen] tarama bitti (HİBRİT): {len(rows)} sonuç, {len(kept)} "
                    f"filtreyi geçti — sıralama z({sort_key})+{SCREEN_HYBRID_SETUP_WEIGHT}·z(setup)")
    else:
        kept.sort(key=lambda r: (r.get(sort_key) is None, -(r.get(sort_key) or 0)))
        logger.info(f"[auto-screen] tarama bitti: {len(rows)} sonuç, {len(kept)} filtreyi geçti")
    return kept


def _write_ranking_csv(rows: list[dict]) -> str | None:
    """Sıralanmış tarama sonucunu logs/auto_screen_<ts>.csv'ye yaz (operatör görünürlüğü)."""
    if not (AUTO_SCREEN_WRITE_CSV and rows):
        return None
    import csv
    from datetime import datetime
    out = LOG_DIR / f"auto_screen_{datetime.now():%Y%m%d_%H%M%S}.csv"
    try:
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info(f"[auto-screen] sıralama CSV: {out}")
        return str(out)
    except Exception as e:
        logger.debug(f"[auto-screen] CSV yazılamadı: {e}")
        return None


def scan_and_update(*, stop_event: threading.Event | None = None) -> list[str]:
    """Bir tam tarama yap ve watchlist'i güncelle. Yeni listeyi döndürür.

    İki-katman (DUAL_TIER_ENABLED) → tüm evren backtest'lenip SIRALANIR; en iyi
    FOCUSED_TOP_N öne (odak/TradingView katmanı), kalan tüm evren arkaya (geniş/
    yfinance katmanı) konur. Partisyon örtüşmesiz, tüm evreni kapsar.

    AUTO_SCREEN_TOP_N <= 0 (dual kapalı) → "HEPSİ" modu: backtest sıralaması yapılmaz,
    watchlist doğrudan tüm evren olur (en iyi N seçimi yok, herkes taranır).
    """
    _UNIVERSE_MEMO.clear()   # D7: yeni tur → taze evren fetch'i (tur-içi çağrılar memodan)
    neden = olculemeyen_strateji()
    if neden and (DUAL_TIER_ENABLED or AUTO_SCREEN_TOP_N > 0):
        logger.warning(f"[auto-screen] tarama ATLANDI — {neden}; watchlist ve katman sınırı korundu")
        return watchlist.current()
    if DUAL_TIER_ENABLED:
        ranked = scan_universe(stop_event=stop_event)
        if stop_event is not None and stop_event.is_set():
            return watchlist.current()
        if not ranked:
            logger.warning("[auto-screen] sıralanmış veri yok — watchlist ve katman sınırı korundu")
            return watchlist.current()
        _write_ranking_csv(ranked)
        metric = "sharpe" if AUTO_SCREEN_SORT == "sharpe" else AUTO_SCREEN_SORT
        focused = [r["symbol"] for r in ranked[:FOCUSED_TOP_N]]
        fset = set(focused)
        rest = [s for s in build_universe(AUTO_SCREEN_UNIVERSE) if s not in fset]
        meta_by_sym = {r["symbol"]: {"score": r.get(metric), "metric": metric,
                                     "risk_mult": _confidence_risk_mult(r.get("trades", CONFIDENCE_COMFORT_TRADES))}
                       for r in ranked[:FOCUSED_TOP_N]}
        new_list = watchlist.replace(focused + rest, meta_by_sym=meta_by_sym,
                                     source="dual", mark_scan=True, focused_n=FOCUSED_TOP_N)
        logger.info(f"[auto-screen] iki-katman: odak {len(focused)} (sıralı/TV) + "
                    f"geniş {len(rest)} (yf) = {len(new_list)} sembol")
        return new_list

    if AUTO_SCREEN_TOP_N <= 0:
        universe = build_universe(AUTO_SCREEN_UNIVERSE)
        if stop_event is not None and stop_event.is_set():
            return watchlist.current()
        new_list = watchlist.replace(universe, source="universe", mark_scan=True)
        logger.info(f"[auto-screen] HEPSİ modu: watchlist = tüm evren "
                    f"({len(new_list)} sembol, backtest atlandı)")
        return new_list

    ranked = scan_universe(stop_event=stop_event)
    if stop_event is not None and stop_event.is_set():
        return watchlist.current()
    if not ranked:
        logger.warning("[auto-screen] filtreyi geçen sembol yok — watchlist değiştirilmedi")
        with_scan = watchlist.replace(watchlist.current(), source="auto", mark_scan=True)
        return with_scan

    _write_ranking_csv(ranked)
    top = ranked[: AUTO_SCREEN_TOP_N]
    metric = "sharpe" if AUTO_SCREEN_SORT == "sharpe" else AUTO_SCREEN_SORT
    meta_by_sym = {r["symbol"]: {"score": r.get(metric), "metric": metric,
                                 "risk_mult": _confidence_risk_mult(r.get("trades", CONFIDENCE_COMFORT_TRADES))}
                   for r in top}
    new_list = watchlist.replace([r["symbol"] for r in top],
                                 meta_by_sym=meta_by_sym, source="auto", mark_scan=True)
    logger.info(f"[auto-screen] watchlist güncellendi → {len(new_list)} sembol: "
                f"{', '.join(new_list[:10])}{' …' if len(new_list) > 10 else ''}")
    return new_list


def girdi_tazelenmedi(now: datetime | None = None) -> bool:
    """Piyasa kapalı ve son tarama son seans kapanışından SONRA mı? Öyleyse barlar aynıdır.

    Tur-91 (BULGU-91-4): kapalı piyasada 4 saatte bir ve her yeniden başlatmada aynı kapanmış barlar
    yeniden taranıyordu; tek tarama (516 sembol × 1h) ~1,5 dk Alpaca bütçesini tüketip hub'ı 429'a
    sokuyor (2026-09-23 seans dışında 1.982 yfinance yedeği), sağlayıcı karışımı yüzünden aynı veriden
    63/64/63 farklı liste üretiyordu. Ölçemezse False (tarama yapılır — bugünkü davranış)."""
    from nihai.data.market_calendar import NYSE, is_market_open
    now = now or datetime.now().astimezone()
    son = watchlist.last_scan()
    if not son or is_market_open(now):
        return False
    try:
        import pandas as pd
        kapanis = NYSE.previous_close(pd.Timestamp(now).tz_convert("UTC").floor("min"))
        return datetime.fromisoformat(son).astimezone() >= kapanis
    except Exception as e:  # noqa: BLE001 — zamanlama kararı taramayı düşüremez
        logger.debug(f"[auto-screen] tazelik ölçülemedi: {e}")
        return False


class AutoScreener:
    """Arka plan thread'i: ilk taramayı yapar, sonra her N saatte bir yineler.

    Vadesi gelen tarama, piyasa kapalıyken son taramadan beri yeni bar yoksa seans açılana dek ertelenir
    (`girdi_tazelenmedi`). ponytail: açılışın ilk yoklaması dünün barlarıyla bir kez gereksiz tarar;
    engellemek ilk HTF barının kapanışını beklemeyi ister."""

    def __init__(self, *, interval_hours: float = AUTO_SCREEN_INTERVAL_HOURS,
                 run_immediately: bool = True,
                 on_update: Callable[[list[str]], None] | None = None):
        self.interval_sec = max(60.0, interval_hours * 3600.0)
        self.run_immediately = run_immediately
        self.on_update = on_update
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_run: datetime | None = None

    def _loop(self) -> None:
        # İlk tarama hemen (mevcut watchlist tohumla çalışırken arka planda taze liste hazırlanır)
        next_due = 0.0 if self.run_immediately else self.interval_sec
        waited = next_due
        ertelendi = False
        while not self._stop.is_set():
            if waited >= next_due and girdi_tazelenmedi():
                if not ertelendi:
                    logger.info("[auto-screen] piyasa kapalı, son taramadan beri yeni bar yok — "
                                "tarama seans açılışına ertelendi")
                    ertelendi = True
            elif waited >= next_due:
                ertelendi = False
                try:
                    new_list = scan_and_update(stop_event=self._stop)
                    self.last_run = datetime.now()
                    if self.on_update and not self._stop.is_set():
                        try:
                            self.on_update(new_list)
                        except Exception as e:
                            logger.debug(f"[auto-screen] on_update hatası: {e}")
                except Exception as e:
                    logger.exception(f"[auto-screen] tarama döngüsü hatası: {e}")
                waited = 0.0
                next_due = self.interval_sec
            # 5 sn'lik dilimlerle bekle ki durdurmaya hızlı tepki ver
            self._stop.wait(5.0)
            waited += 5.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="auto-screener", daemon=True)
        self._thread.start()
        logger.info(f"[auto-screen] thread başladı | her {self.interval_sec/3600:.1f} saatte bir")

    def stop(self) -> None:
        self._stop.set()


def main():
    """Bağımsız: tek tarama yap, watchlist'i güncelle, çık."""
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    new_list = scan_and_update()
    print("\n=== Güncel dinamik watchlist ===")
    for i, s in enumerate(new_list, 1):
        m = watchlist.meta().get(s, {})
        print(f"{i:>3} {s:<8} {m.get('metric','')}={m.get('score')}")
    print(f"\nToplam {len(new_list)} sembol → data/watchlist.json")


if __name__ == "__main__":
    main()
