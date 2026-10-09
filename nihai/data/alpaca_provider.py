"""
alpaca_provider.py — Alpaca Market Data veri sağlayıcısı (gerçek-zamanlı).

PDF Referansları:
- Madde 34 : Veri sağlığı kontrolü (DataProvider base'inden)
- Madde 45 : Veri yedekliliği — opsiyon zinciri için yfinance fallback (FailoverProvider)
- Madde 48 : Düşük gecikme (Alpaca IEX feed ücretsiz gerçek-zamanlı; SIP paylı)

yfinance ~15 dk gecikmeliyken Alpaca IEX feed GERÇEK-ZAMANLI (ücretsiz). Resmi API.
Opsiyon verisi ayrı (paylı) feed olduğundan get_option_chain {} döner → FailoverProvider
yfinance'e geçip opsiyonu oradan alır.

Kurulum:
    pip install alpaca-py
.env:
    ALPACA_API_KEY=...        ALPACA_SECRET_KEY=...
    ALPACA_FEED=iex           # iex (ücretsiz) | sip (paylı)

Tasarım: alpaca-py LAZY import edilir (SDK kurulu değilse modül yine yüklenir; sağlayıcı
boş/None döner → FailoverProvider yfinance'e geçer). Normalize fonksiyonları SAF ve
enjekte edilebilir client ile SDK'sız test edilebilir.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime
from typing import ClassVar

import pandas as pd
from loguru import logger

from nihai.config import INTERVAL_MINUTES
from nihai.data.data_provider import (
    DataProvider,
    Interval,
    Quote,
    YFinanceProvider,
    resample_bars,
    rth_mask,
)

# Interval → (amount, unit) — TimeFrame alpaca'dan LAZY kurulur (import güvenliği)
_TF_SPEC: dict[str, tuple[int, str]] = {
    "1m": (1, "Minute"), "5m": (5, "Minute"), "15m": (15, "Minute"),
    "1h": (1, "Hour"), "1d": (1, "Day"),
}
# Tur-90 (BULGU-90-2): Alpaca saat barı SAATE hizalıdır; 09:00 barı 09:00–09:30 seans-öncesini
# taşır (16 sembol × 31 günde 197 günde vardı, 56'sında barın tepe/dibini o 30 dk belirledi).
# Canlı 1h bu yüzden 30 dk alt barlardan kurulur: seans-dışı alt bar düşer, kalanlar toplanır.
# Tur-91 (BULGU-91-1): toplama SEANS açılışına çapalıdır (09:30, 10:30 … 15:30 = 15:30–16:00),
# yfinance yedeğinin ızgarası; saat ızgarası hub 429'da seriyi iki ızgara arasında bölüyordu.
_RTH_SUB_MINUTES: dict[str, int] = {"1h": 30}

# --- Kurumsal işlem düzeltmesi (BULGU-19 → Tur-90 BULGU-90-1: canlı yol da) ----------------
# API varsayılanı 'raw' = bölünme/temettü DÜZELTİLMEMİŞ. BULGU-19 bunu yalnız araştırma yolunda
# (`alpaca_history`) kapattı; canlı `get_bars` hâlâ ham barla koşuyordu. Ölçüldü (2026-09-23):
# APH 1h penceresinde 2026-09-03 barı −%49,8 (2:1 bölünme), CRWD 1d'de −%74,9, KLAC −%89,4.
# TEK kaynak burasıdır; `alpaca_history` bunu içe aktarır (env adı geriye uyum için korunur).
_VALID_ADJUSTMENT = ("raw", "split", "dividend", "all")


def _adjustment() -> str:
    """Geçerli düzeltme kipi (env ALPACA_HIST_ADJUSTMENT; varsayılan 'all'), çağrı anında okunur."""
    adj = (os.getenv("ALPACA_HIST_ADJUSTMENT") or "all").lower()
    if adj not in _VALID_ADJUSTMENT:
        logger.warning(f"[alpaca] geçersiz ALPACA_HIST_ADJUSTMENT={adj!r} → 'all'")
        return "all"
    return adj


def _canli_duzeltme(interval: str) -> str:
    """Canlı yolun düzeltme kipi: günlükte `_adjustment()`, GÜN İÇİNDE temettü bileşeni çıkarılmış hâli.

    Tur-91 (BULGU-91-2): yfinance yedeğinin gün-içi barı yalnız BÖLÜNME düzeltmelidir (günlüğü
    temettü de düzeltir). 'all' ile her temettü tarihinden geriye Alpaca/yfinance oranı bir temettü
    verimi kadar basamaklanıyordu (SPY 0,9975, XOM 0,9936); yedeğe düşüşte geçmiş fiyat düzeyi atlıyor,
    eyleme dönük 1h kararların %3,9'u değişiyordu. Araştırma yolu `_adjustment()`ı değişmeden okur."""
    adj = _adjustment()
    return adj if interval == "1d" else {"all": "split", "dividend": "raw"}.get(adj, adj)

# --- BULGU B-4: kotasyon akıl-sağlığı kapısı -----------------------------------
# ÜCRETSİZ IEX feed'inin `latest_quote`'u, IEX defteri o sembolde pasifken BAYAT
# ya da piyasa-dışı bir bid/ask döndürebilir. 2026-07-25 ölçümü (piyasa kapalı):
#
#     sembol   Alpaca bar   Alpaca quote   yfinance bar   yfinance quote
#     AAPL        333.48        312.08          333.07          333.02   → %-6.4
#     NVDA        206.87        197.07          206.96          206.84   → %-4.7
#     SPY         738.29        738.63          738.86          738.93   → %+0.05
#
# İki bağımsız kaynağın BARLARI birbirini doğruluyor; sapan tek şey Alpaca'nın
# kotasyonu. Kotasyon fiyatı sinyalde giriş/stop/hedef mesafesine giriyor → %6'lık
# bir sapma sessizce yanlış işlem planı üretir. Bu yüzden kotasyon, sağlayıcının
# ZATEN elindeki son bar kapanışına karşı çapraz doğrulanır; saparsa None döner ve
# FailoverProvider yfinance'e geçer (Madde 45 deseni; EK REST çağrısı YAPILMAZ).
QUOTE_SANITY_PCT_DEFAULT = 3.0
# Referans bar kapanışı bu süreden eskiyse kapı UYGULANMAZ (fail-open): bayat bir
# çapanın sağlam kotasyonu elemesi, kapıyı hiç koymamaktan daha kötü olurdu.
QUOTE_SANITY_REF_TTL_SEC = 900.0


def quote_sanity_pct() -> float:
    """İzin verilen azami |quote − son bar kapanışı| yüzdesi. <=0 → kapı KAPALI."""
    try:
        return float(os.getenv("QUOTE_SANITY_PCT", QUOTE_SANITY_PCT_DEFAULT))
    except ValueError:
        return QUOTE_SANITY_PCT_DEFAULT


def quote_diverges(last: float, ref_close: float | None, max_pct: float | None = None) -> bool:
    """SAF: kotasyon fiyatı referans bar kapanışından kabul edilemez ölçüde sapmış mı?

    ref_close yok/0 veya kapı kapalıysa (max_pct <= 0) DAİMA False — yani bilgi
    yoksa kotasyon elenmez (fail-open)."""
    if not ref_close or ref_close <= 0 or not last or last <= 0:
        return False
    limit = quote_sanity_pct() if max_pct is None else max_pct
    if limit <= 0:
        return False
    return abs(last - ref_close) / ref_close * 100.0 > limit


def _to_alpaca_symbol(symbol: str) -> str:
    """Çift-sınıf ticker biçimini Alpaca'ya çevir (Madde 45 — gereksiz failover'ı önle).

    Alpaca çift-sınıf hisselerde NOKTA kullanır (BRK.B, BF.B); watchlist'ler ise
    S&P/yfinance kaynaklı olduğu için TİRE içerir (BRK-B, BF-B). Çevirmezsek Alpaca
    'invalid symbol' döner → bu isimler HER çağrıda yfinance'e (15 dk gecikmeli)
    düşer. Sadece Alpaca API çağrısında kullanılır; dönen Quote/df sembolü çağıranın
    verdiği orijinal biçimde kalır (cache anahtarları/watchlist eşleşmesi bozulmasın)."""
    return symbol.upper().replace("-", ".")


def _start_for_lookback(interval: Interval, lookback_bars: int) -> datetime:
    """`lookback_bars` kadar barı kapsayacak geçmiş başlangıç zamanı (UTC).

    Alpaca `start` verilmezse yalnızca bugünün barlarını döner → strateji için
    yetersiz. İşlem günü başına bar sayısından geriye gidip hafta sonu/tatil tamponu
    ekleriz (intraday yalnızca 6.5 saat/gün işler)."""
    from datetime import timedelta
    bars_per_day = {"1m": 390, "5m": 78, "15m": 26, "1h": 7, "1d": 1}.get(interval, 7)
    trading_days = lookback_bars / bars_per_day
    cal_days = int(trading_days * 1.7) + 5          # ~5/7 işlem günü + tatil tamponu
    return datetime.now(UTC) - timedelta(days=cal_days)


def _normalize_alpaca_bars(raw: pd.DataFrame, symbol: str, interval: Interval,
                           lookback_bars: int, *, rth: bool = False,
                           src_minutes: int | None = None) -> pd.DataFrame:
    """Alpaca bar DataFrame'ini sistem formatına çevir (SAF — SDK'sız test edilebilir).

    Alpaca formatı: MultiIndex (symbol, timestamp) VEYA tek timestamp index;
    kolonlar open/high/low/close/volume (+ trade_count/vwap). Çıktı: küçük-harf OHLCV,
    tz-naive datetime index, Madde 3 oluşmakta olan son bar dışlanır, son N bar.

    rth=True (canlı yol, Tur-90): gün-içi barlardan seansla örtüşmeyenler düşer (Madde 22);
    `src_minutes` ham barın süresidir (interval'dan kısaysa süzülüp interval'a toplanır).
    Araştırma yolu (`alpaca_history`) False geçer: diskteki önbellekler süzülmemiş barlarla
    kuruldu ve yeniden kurulum ayrı, bilinçli bir adımdır.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw
    # MultiIndex (symbol, timestamp) → sembolü ayıkla
    if isinstance(df.index, pd.MultiIndex):
        if symbol in df.index.get_level_values(0):
            df = df.xs(symbol, level=0)
        else:
            df = df.droplevel(0)
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    if len(keep) < 4:
        return pd.DataFrame()
    # Tur-57: Alpaca barları `trade_count` ve `vwap` DA taşır; bu satır onları yıllarca
    # düşürüyordu. İkisi de ölçülemez sanılan üç metriğin girdisidir (ortalama işlem
    # büyüklüğü = hacim/işlem sayısı, konsolide VWAP sapması, balina imzası —
    # microstructure.trade_size_profile / vwap_deviation). Kolonlar ADLA okunduğu için
    # (tüm tüketiciler df["close"] gibi erişir) eklemek geriye dönük GÜVENLİDİR.
    keep += [c for c in ("trade_count", "vwap") if c in df.columns]
    df = df[keep]

    # Index'i tz-aware UTC datetime'a getir (oluşan bar kontrolü için), sonra naive yap
    idx = pd.to_datetime(df.index, utc=True)
    df.index = idx

    if rth and interval != "1d":
        bar_dk = INTERVAL_MINUTES.get(interval, 60)
        kaynak_dk = src_minutes or bar_dk
        df = df[rth_mask(df.index, kaynak_dk)]
        if kaynak_dk < bar_dk and not df.empty:
            df = resample_bars(df, bar_dk)
        if df.empty:
            return pd.DataFrame()

    # Madde 3 — oluşmakta olan son barı dışla
    if interval != "1d" and len(df) > 1:
        now_utc = datetime.now(UTC)
        bar_delta = pd.Timedelta(minutes=INTERVAL_MINUTES.get(interval, 60))
        if now_utc - df.index[-1].to_pydatetime() < bar_delta:
            df = df.iloc[:-1]

    # Sistem geneli naive datetime kullanır (tz-aware çıkarma hatalarını önle)
    df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df.tail(lookback_bars)


def _quote_from_alpaca(aq, symbol: str) -> Quote | None:
    """Alpaca latest-quote nesnesini Quote'a çevir (SAF). bid/ask 0 ise mid'den türet."""
    if aq is None:
        return None
    bid = float(getattr(aq, "bid_price", 0) or 0)
    ask = float(getattr(aq, "ask_price", 0) or 0)
    measured = 0 < bid <= ask < float("inf") and getattr(aq, "timestamp", None) is not None
    last = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask)
    if bid == 0 or ask == 0:
        # tek taraf varsa sentetik makas üret (yfinance sağlayıcısıyla aynı mantık)
        ref = last or bid or ask
        if ref > 0:
            bid, ask = ref * 0.999, ref * 1.001
            last = ref
    ts = getattr(aq, "timestamp", None) or datetime.now()
    return Quote(symbol=symbol, bid=bid, ask=ask, last=last, timestamp=ts, spread_measured=measured)


def _alpaca_options_enabled() -> bool:
    """ALPACA_OPTIONS=1/true/on ise OPRA zenginleştirmesi açık (varsayılan KAPALI)."""
    return (os.getenv("ALPACA_OPTIONS") or "").strip().lower() in ("1", "true", "on", "yes")


def _snap_get(obj: object, *names: str) -> object:
    """Snapshot alanını hem nesne (alpaca modeli) hem sözlük (test/JSON) üstünden oku."""
    for n in names:
        v = getattr(obj, n, None)
        if v is None and isinstance(obj, dict):
            v = obj.get(n)
        if v is not None:
            return v
    return None


def _pos_float(v: object) -> float | None:
    """Sonlu ve POZİTİF float ise değeri, değilse None — "yazılsın mı" kararının tek kapısı."""
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")) or f <= 0:
        return None
    return f


def enrich_chain_side(df: pd.DataFrame, snapshots: dict) -> tuple[pd.DataFrame, int]:
    """yfinance zincir tarafını Alpaca OPRA anlık verisiyle zenginleştir (SAF fonksiyon).

    Eşleme OCC kontrat sembolü üstünden yapılır (`contractSymbol` iki kaynakta da OCC).
    YALNIZ sonlu-pozitif Alpaca değerleri yazılır: bid/ask ve implied vol; greekler YENİ
    kolon olarak eklenir (delta/gamma/theta/vega — `option_flow` bunları doğrudan kullanır,
    IV'den yeniden türetmek zorunda kalmaz).

    Dönüş: (zenginleştirilmiş KOPYA, güncellenen satır sayısı). Eşleşme yoksa df aynen
    döner; eksik alan UYDURULMAZ."""
    if df is None or getattr(df, "empty", True) or not snapshots:
        return df, 0
    if "contractSymbol" not in df.columns:
        return df, 0
    out = df.copy()
    updated = 0
    for idx, occ in out["contractSymbol"].items():
        snap = snapshots.get(str(occ))
        if snap is None:
            continue
        touched = False
        q = _snap_get(snap, "latest_quote", "latestQuote")
        if q is not None:
            bid = _pos_float(_snap_get(q, "bid_price", "bp", "bid"))
            ask = _pos_float(_snap_get(q, "ask_price", "ap", "ask"))
            if bid is not None and ask is not None and ask >= bid:
                out.at[idx, "bid"], out.at[idx, "ask"] = bid, ask
                touched = True
        iv = _pos_float(_snap_get(snap, "implied_volatility", "impliedVolatility"))
        if iv is not None:
            out.at[idx, "impliedVolatility"] = iv
            touched = True
        g = _snap_get(snap, "greeks")
        if g is not None:
            for name in ("delta", "gamma", "theta", "vega"):
                val = _snap_get(g, name)
                if val is None:
                    continue
                try:
                    out.at[idx, name] = float(val)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                touched = True
        updated += int(touched)
    return out, updated


class AlpacaProvider(DataProvider):
    name = "alpaca"

    # Sınıf-düzeyi çağrı sayacı: TÜM AlpacaProvider örneklerinin (odak katman +
    # ana döngü + failover sarmaları, hepsi ayrı instance) Alpaca'ya yaptığı GERÇEK
    # REST çağrılarını toplar. get_option_chain Alpaca'ya GİTMEZ (yfinance fallback)
    # → sayılmaz. Rate-limit gözlemlenebilirliği için (Madde 48). Thread-safe.
    _calls_lock = threading.Lock()
    _calls: ClassVar[dict[str, int]] = {"get_bars": 0, "get_quote": 0,
                                    "get_option_chain": 0, "asset_flags": 0}

    @classmethod
    def _tick(cls, method: str) -> None:
        with cls._calls_lock:
            cls._calls[method] = cls._calls.get(method, 0) + 1

    @classmethod
    def call_counts(cls) -> dict[str, int]:
        """Kümülatif çağrı sayaçlarının kopyası (get_bars/get_quote + total)."""
        with cls._calls_lock:
            snap = dict(cls._calls)
        snap["total"] = sum(snap.values())
        return snap

    def __init__(self, api_key: str | None = None, secret_key: str | None = None,
                 feed: str | None = None, *, data_client=None):
        self._api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self._secret = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self._feed = (feed or os.getenv("ALPACA_FEED") or "iex").lower()
        # Tur-57 (ÖLÇÜLDÜ 2026-08-29): ÜCRETSİZ anahtarla SIP *tarihsel* barlar zaten
        # yetkili — yalnız son 15 dakika kapalı ("subscription does not permit querying
        # recent SIP data"). IEX barları konsolide hacmin %2,2–3,4'ünü görüyor (AAPL/SPY/
        # NVDA, 3 gün). Yani hacim tabanlı her ölçü IEX'te 30 kat KÜÇÜK ölçülüyor.
        # AMA SIP barları uzatılmış seansı da içerir (5dk: 225 bar vs IEX 81) → bar KÜMESİ
        # değişir, göstergeler değişir. Bu yüzden opt-in: ALPACA_BARS_FEED=sip.
        self._bars_feed = (os.getenv("ALPACA_BARS_FEED") or self._feed).lower()
        self._client = data_client            # test için enjekte edilebilir
        self._client_tried = data_client is not None
        self._trading_client = None
        self._trading_tried = False
        self._asset_cache: dict[str, tuple[str, dict]] = {}   # sembol → (gün, bayraklar)
        self._yf_fallback = YFinanceProvider()  # yalnızca opsiyon zinciri için
        # B-4 kapısı için ÜCRETSİZ çapa: get_bars zaten çekiyor, kapanışı hatırla.
        # sembol → (kapanış, monotonic zaman). Ek REST çağrısı YOK.
        self._last_close: dict[str, tuple[float, float]] = {}

    # -----------------------------------------------------------------
    def _get_client(self):
        """alpaca-py StockHistoricalDataClient'i lazy kur. SDK/anahtar yoksa None."""
        if self._client is not None:
            return self._client
        if self._client_tried:
            return None
        self._client_tried = True
        if not (self._api_key and self._secret):
            logger.warning("[alpaca] ALPACA_API_KEY/SECRET yok — sağlayıcı pasif, "
                           "FailoverProvider yfinance'e geçecek")
            return None
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            self._client = StockHistoricalDataClient(self._api_key, self._secret)
            # Tur-82: alpaca-py 429'da 3 sn uyuyup 3 kez yeniden dener (kurucu bunu dışarı
            # açmaz). Hub'da bu, isteği istemcinin 6 sn zaman aşımının ÖTESİNE taşır: yanıtı
            # kimse okumaz ama bütçe harcanır ve hub thread'i bloklanır. ÖLÇÜLDÜ (2026-09-18,
            # açılış taraması 16 süreç × 516 sembol): 2 dk'da 162 adet 429 ve istemcide 171
            # zaman aşımı; tarama 53-77 sn yerine 4+ dk sürdü. Yeniden deneme yok → 429 anında
            # hata → FailoverProvider'ın yfinance yedeği (Madde 45) aynı istekte cevap verir.
            if hasattr(self._client, "_retry"):
                self._client._retry = 0
            logger.info(f"[alpaca] bağlandı (feed={self._feed})")
        except Exception as e:
            logger.warning(f"[alpaca] SDK/bağlantı yok ({e}) — yfinance fallback devrede")
            self._client = None
        return self._client

    def _timeframe(self, interval: Interval):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        amount, unit = _TF_SPEC.get(interval, (1, "Hour"))
        return TimeFrame(amount, getattr(TimeFrameUnit, unit))

    def _feed_enum(self, feed: str | None = None):
        """feed adı (str) → DataFeed enum. Pydantic string'i de kabul eder ama
        tip-doğru olsun diye enum'a çeviririz (geçersizse IEX)."""
        from alpaca.data.enums import DataFeed
        try:
            return DataFeed(feed or self._feed)
        except ValueError:
            return DataFeed.IEX

    # -----------------------------------------------------------------
    def get_bars(self, symbol: str, interval: Interval, lookback_bars: int = 200) -> pd.DataFrame:
        client = self._get_client()
        if client is None:
            return pd.DataFrame()
        self._tick("get_bars")  # client mevcut → bu satırdan sonra gerçek REST isteği
        asym = _to_alpaca_symbol(symbol)   # BRK-B → BRK.B (yoksa 'invalid symbol' → failover)
        alt_dk = _RTH_SUB_MINUTES.get(interval)
        try:
            from alpaca.data.enums import Adjustment
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            # NOT: `limit` + `start` birlikte, start'tan en ESKİ N barı döner (son barları
            # keser). En GÜNCEL barlar gerektiğinden limit YOK — start penceresi çek,
            # _normalize_alpaca_bars tail(lookback_bars) ile en yeni N'i alır. Pencere RTH bar
            # sayısıyla kurulur (`_start_for_lookback`), süzme sonrası sayı bu yüzden yeter.
            req = StockBarsRequest(
                symbol_or_symbols=asym,
                timeframe=(TimeFrame(alt_dk, TimeFrameUnit.Minute) if alt_dk
                           else self._timeframe(interval)),
                start=_start_for_lookback(interval, lookback_bars + 10),
                feed=self._feed_enum(self._bars_feed),
                adjustment=Adjustment(_canli_duzeltme(interval)),
            )
            raw = client.get_stock_bars(req).df
        except Exception as e:
            # Abonelik yetmiyorsa (ALPACA_BARS_FEED=sip + ücretsiz plan + taze pencere)
            # feed'i KALICI olarak düşür ve bir kez uyar: her barda tekrar denemek
            # tarama döngüsünü sessizce yavaşlatırdı.
            if self._bars_feed != self._feed and "subscription" in str(e).lower():
                logger.warning(f"[alpaca] {self._bars_feed} bar yetkisi yok ({e}) — "
                               f"bu oturum için {self._feed} feed'ine düşülüyor")
                self._bars_feed = self._feed
                return self.get_bars(symbol, interval, lookback_bars)
            logger.warning(f"[alpaca] {symbol} {interval} bar çekme hatası: {e}")
            return pd.DataFrame()
        # MultiIndex level'ı Alpaca'nın döndürdüğü (nokta) sembolle eşleşir → asym ver.
        df = _normalize_alpaca_bars(raw, asym, interval, lookback_bars, rth=True, src_minutes=alt_dk)
        self._remember_close(symbol, df)
        return df

    # -----------------------------------------------------------------
    # B-4 kotasyon akıl-sağlığı kapısı
    # -----------------------------------------------------------------
    def _remember_close(self, symbol: str, df: pd.DataFrame) -> None:
        """get_bars sonucunun son kapanışını çapa olarak sakla (SAF yan-etki, IO yok)."""
        if df is None or df.empty or "close" not in df.columns:
            return
        try:
            close = float(df["close"].iloc[-1])
        except (TypeError, ValueError, IndexError):
            return
        if close > 0:
            self._last_close[symbol.upper()] = (close, time.monotonic())

    def _ref_close(self, symbol: str) -> float | None:
        """Kapı için taze referans kapanış; yoksa/bayatsa None (kapı uygulanmaz)."""
        hit = self._last_close.get(symbol.upper())
        if not hit:
            return None
        close, seen = hit
        return close if (time.monotonic() - seen) <= QUOTE_SANITY_REF_TTL_SEC else None

    def get_quote(self, symbol: str) -> Quote | None:
        client = self._get_client()
        if client is None:
            return None
        self._tick("get_quote")  # client mevcut → bu satırdan sonra gerçek REST isteği
        asym = _to_alpaca_symbol(symbol)   # BRK-B → BRK.B (yoksa 'invalid symbol' → failover)
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            req = StockLatestQuoteRequest(symbol_or_symbols=asym, feed=self._feed_enum())
            res = client.get_stock_latest_quote(req)
            aq = res.get(asym) if isinstance(res, dict) else res
        except Exception as e:
            logger.warning(f"[alpaca] {symbol} quote hatası: {e}")
            return None
        # Quote.symbol çağıranın orijinal biçiminde kalır (cache/watchlist eşleşmesi).
        q = _quote_from_alpaca(aq, symbol.upper())
        if q is None:
            return None
        ref = self._ref_close(symbol)
        if ref is not None and quote_diverges(q.last, ref):
            logger.warning(
                f"[alpaca] {symbol} kotasyonu ELENDİ (B-4): last={q.last:.4f} vs "
                f"son bar kapanışı={ref:.4f} — sapma "
                f"%{abs(q.last - ref) / ref * 100:.2f} > "
                f"%{quote_sanity_pct():.2f}; yfinance'e failover"
            )
            return None
        return q

    # -----------------------------------------------------------------
    # Opsiyon zinciri — yfinance TABAN + Alpaca OPRA ZENGİNLEŞTİRME (Tur-57)
    # -----------------------------------------------------------------
    def _option_client(self):
        """OptionHistoricalDataClient'i lazy kur (SDK/anahtar yoksa None)."""
        if getattr(self, "_opt_client", None) is not None:
            return self._opt_client
        if getattr(self, "_opt_tried", False) or not (self._api_key and self._secret):
            return None
        self._opt_tried = True
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            self._opt_client = OptionHistoricalDataClient(self._api_key, self._secret)
        except Exception as e:
            logger.warning(f"[alpaca] opsiyon istemcisi kurulamadı ({e}) — zincir yalnız yfinance")
            self._opt_client = None
        return self._opt_client

    def _option_snapshots(self, symbol: str, expiry: str | None) -> dict:
        """OPRA anlık zinciri: kontrat sembolü → snapshot (IV + greekler + NBBO)."""
        client = self._option_client()
        if client is None or not expiry:
            return {}
        try:
            from alpaca.data.requests import OptionChainRequest
            self._tick("get_option_chain")
            res = client.get_option_chain(OptionChainRequest(
                underlying_symbol=_to_alpaca_symbol(symbol), expiration_date=expiry))
            return dict(res) if res else {}
        except Exception as e:
            logger.warning(f"[alpaca] {symbol} {expiry} OPRA zinciri alınamadı: {e}")
            return {}

    def get_option_chain(self, symbol: str, expiry: str | None = None) -> dict:
        """Zincirin TABANI yfinance'tir (hacim + OI + tüm vadeler yalnız orada var).

        `ALPACA_OPTIONS=1` ise üstüne Alpaca OPRA anlık verisi İŞLENİR: canlı NBBO
        (bid/ask), gerçek IV ve greekler. Yalnız SONLU ve POZİTİF Alpaca değerleri
        yazılır — yani zenginleştirme veriyi yalnız İYİLEŞTİREBİLİR, bozamaz. Herhangi
        bir hata → düz yfinance zinciri (Madde 45 fail-soft deseni).

        Varsayılan KAPALI: ek çağrı tarama döngüsüne gecikme ekler ve bu bir DAVRANIŞ
        değişikliğidir — proje disiplini önce ÖLÇ, sonra aç (docs/TUR57_RAPORU)."""
        chain = self._yf_fallback.get_option_chain(symbol, expiry)
        if not chain or not _alpaca_options_enabled():
            return chain
        snaps = self._option_snapshots(symbol, str(chain.get("expiry") or ""))
        if not snaps:
            return chain
        total = 0
        for side in ("calls", "puts"):
            df = chain.get(side)
            if df is None or getattr(df, "empty", True):
                continue
            chain[side], n = enrich_chain_side(df, snaps)
            total += n
        chain["option_source"] = "yfinance+alpaca_opra" if total else "yfinance"
        chain["enriched_rows"] = total
        return chain

    # -----------------------------------------------------------------
    # Açığa satış uygunluğu — Alpaca TRADING API (ÜCRETSİZ; ölçüldü 2026-08-29)
    # -----------------------------------------------------------------
    def _get_trading_client(self):
        if self._trading_client is not None or self._trading_tried:
            return self._trading_client
        self._trading_tried = True
        if not (self._api_key and self._secret):
            return None
        try:
            from alpaca.trading.client import TradingClient
            self._trading_client = TradingClient(self._api_key, self._secret, paper=True)
        except Exception as e:
            logger.warning(f"[alpaca] trading istemcisi kurulamadı ({e}) — borç bayrakları yok")
            self._trading_client = None
        return self._trading_client

    def asset_flags(self, symbol: str) -> dict | None:
        """`shortable` / `easy_to_borrow` / `marginable` — SHORT sinyalinin ön koşulu.

        PDF Madde 30'un eksik yarısı: SSR zaten kodluydu ama hissenin O GÜN açığa
        satılabilir OLUP OLMADIĞI hiç sorulmuyordu. Bayraklar gün içinde değişebilir →
        önbellek GÜN bazlıdır. Anahtar/SDK yoksa None (bilinmiyor ≠ satılabilir)."""
        sym = symbol.upper().strip()
        today = datetime.now(UTC).date().isoformat()
        hit = self._asset_cache.get(sym)
        if hit and hit[0] == today:
            return hit[1]
        client = self._get_trading_client()
        if client is None:
            return None
        try:
            self._tick("asset_flags")
            a = client.get_asset(_to_alpaca_symbol(sym))
        except Exception as e:
            logger.debug(f"[alpaca] {sym} varlık bayrakları alınamadı: {e}")
            return None
        flags = {"symbol": sym,
                 "shortable": bool(getattr(a, "shortable", False)),
                 "easy_to_borrow": bool(getattr(a, "easy_to_borrow", False)),
                 "marginable": bool(getattr(a, "marginable", False)),
                 "tradable": bool(getattr(a, "tradable", False)),
                 "as_of": today}
        self._asset_cache[sym] = (today, flags)
        return flags
