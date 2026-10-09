"""
data_provider.py — Soyut veri sağlayıcı + yfinance implementasyonu.

PDF Referansları:
- Madde 3  : Look-ahead koruması (oluşmakta olan bar dışlanır)
- Madde 22 : Pre-market/after-hours sığ verisinden sinyal üretilmez
- Madde 34 : Veri sağlığı kontrolü (bid>ask, sıfır hacim, absürt spread)
- Madde 45 : Veri beslemesi yedekliliği (abstract sayesinde swap kolay)
- Madde 48 : İleride WebSocket / Polygon eklenebilir

NOT: yfinance gecikmeli (~15 dk) veri verir. Canlı düşük gecikme için ileride
Polygon/Alpaca WebSocket implementasyonu eklenir (aynı interface).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

Interval = Literal["1m", "5m", "15m", "1h", "1d"]


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    last: float
    timestamp: datetime
    spread_measured: bool = False  # sentetik/eskiden kaydedilmiş bid-ask ölçüm sayılmaz

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2 if self.bid > 0 and self.ask > 0 else self.last

    @property
    def spread(self) -> float:
        return max(self.ask - self.bid, 0.0)

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return self.spread / m if m > 0 else float("inf")


# ---------------------------------------------------------------------------
# Soyut interface (Madde 45)
# ---------------------------------------------------------------------------
class DataProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def get_bars(self, symbol: str, interval: Interval, lookback_bars: int = 200) -> pd.DataFrame:
        """OHLCV barları döndürür. Madde 3 — oluşmakta olan son bar mutlaka dışlanır."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote | None:
        """Anlık (NBBO benzeri) kotasyon. Bayat/çapraz tespiti için bid/ask döner."""

    @abstractmethod
    def get_option_chain(self, symbol: str, expiry: str | None = None) -> dict:
        """{'calls': DataFrame, 'puts': DataFrame, 'expiry': str, 'underlying_price': float}"""

    # Madde 34 — Veri sağlığı kontrolü
    def is_quote_healthy(self, q: Quote) -> tuple[bool, str]:
        if q.bid <= 0 or q.ask <= 0:
            return False, "ZERO_OR_NEGATIVE_QUOTE"
        if q.bid > q.ask:
            return False, "CROSSED_MARKET"
        if q.spread_pct > 0.5:
            return False, "ABSURD_SPREAD"
        if q.last <= 0:
            return False, "STALE_LAST"
        return True, "OK"


# ---------------------------------------------------------------------------
# Madde 34'ün BAR tarafı — `is_quote_healthy` kotasyonu denetler, barları HİÇBİR ŞEY
# denetlemiyordu. `nihai/ai/data_health.py` o denetimi (NaN/OHLC geçerliliği/eksik-bar/
# donmuş fiyat/sıçrama-outlier/hacim anomalisi → 0..100 skor) yazılı olarak taşıyordu
# ama kendi docstring'inin dediği gibi "bağlama KÖPRÜSÜ Dalga-2 entegrasyonunda kurulur"
# — kurulmamıştı, modülün üretimde çağıranı yoktu. Köprü budur.
#
# KARAR-DIŞI ve ÇAĞRI-BAŞINA DEĞİL: bu fonksiyon `get_bars` içinden ÇAĞRILMAZ (her
# çağrıda sıçrama tespiti koşturmak ölçülebilir bir israftır ve sağlayıcıyı AI katmanına
# bağımlı kılardı). Çağıran `nihai/ai/observatory.bar_shadow`'dur: bar başına BİR kez,
# `_bar_context` cache'i altında. Skor hiçbir barı REDDETMEZ, hiçbir sinyali bastırmaz —
# yalnız gölge kolona ve operatör raporuna gider.
# ---------------------------------------------------------------------------
def bars_health(df: pd.DataFrame | None, symbol: str | None = None) -> dict | None:
    """OHLCV bar df'inin veri-sağlığı karnesi → {'score','issues','quarantine',...}.

    AI katmanı yoksa/hata verirse None (sağlayıcı yolu hiçbir koşulda düşmez)."""
    if df is None or getattr(df, "empty", True):
        return None
    try:
        from nihai.ai.data_health import health_check
        return health_check(df, symbol)
    except Exception as e:      # noqa: BLE001 — veri yolu AI katmanına bağımlı olamaz
        logger.debug(f"[data_health] {symbol}: ölçülemedi ({e})")
        return None


# ---------------------------------------------------------------------------
# Madde 22 — NORMAL SEANS süzgeci (Tur-90, BULGU-90-2). Bu dosyanın başlığı "seans-öncesi/
# sonrası sığ verisinden sinyal üretilmez" der; yfinance yolu bunu `prepost=False` ile yapar,
# Alpaca yolu hiç yapmıyordu. Alpaca IEX barları uzatılmış seansı taşır (1h: SPY'de 200 barın
# 41'i, trade_count 6–11) ve saat barı SAATE hizalıdır (09:00 barı 09:00–09:30 seans-öncesini
# içerir). Takvim XNYS'ten gelir: tatil ve yarım gün (13:00 kapanış) kendiliğinden doğrudur.
# ---------------------------------------------------------------------------
_NYSE_CAL = None


def rth_mask(index: pd.DatetimeIndex, bar_minutes: int) -> np.ndarray:
    """Bar [başlangıç, başlangıç+bar_minutes) NYSE normal seansıyla ÖRTÜŞÜYOR mu?

    naive index = UTC (sistem sözleşmesi). Seansla hiç örtüşmeyen bar (08:00, 16:00+ ET,
    tatil) False; alt barları süzüp sonra toplamak çağıranın işidir (`resample_bars`)."""
    global _NYSE_CAL
    if len(index) == 0:
        return np.zeros(0, dtype=bool)
    if _NYSE_CAL is None:
        import exchange_calendars as xcals
        _NYSE_CAL = xcals.get_calendar("XNYS")
    s = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    gun = s.tz_convert("America/New_York").tz_localize(None).normalize()
    sch = _NYSE_CAL.schedule
    acilis = pd.to_datetime(sch["open"].reindex(gun).to_numpy(), utc=True)
    kapanis = pd.to_datetime(sch["close"].reindex(gun).to_numpy(), utc=True)
    bitis = s + pd.Timedelta(minutes=bar_minutes)
    return np.asarray((s < kapanis) & (bitis > acilis), dtype=bool)   # NaT (seans yok) → False


_SEANS_ACILIS_DK = 9 * 60 + 30     # 09:30 ET, gün içi dakika


def resample_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Alt barları SEANS AÇILIŞINA (09:30 ET) çapalı üst bara topla (label/closed='left'); boş kova düşer.

    Tur-91 (BULGU-91-1): kovalar eskiden SAATE hizalıydı (1h: 09:00, 10:00…); yfinance yedeği ve
    backtest'in yfinance yolu 09:30, 10:30… verir. Hub 429 alınca aynı hissenin 1h serisi seans
    içinde iki ızgara arasında gidip geliyordu. UTC ofseti tam saat olduğu için (570 mod dk) çapası
    60'ı bölen her süre için UTC'de de aynıdır; 5m/15m/30m'de ofset 0'dır (davranış değişmez).
    `trade_count` toplanır, `vwap` hacim ağırlıklı yeniden kurulur (Tur-57 kolonları)."""
    kova = {"rule": f"{minutes}min", "label": "left", "closed": "left",
            "offset": pd.Timedelta(minutes=_SEANS_ACILIS_DK % minutes)}
    kural = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "trade_count" in df.columns:
        kural["trade_count"] = "sum"
    out = df.resample(**kova).agg(kural)
    if "vwap" in df.columns and "volume" in df.columns:
        pv = (df["vwap"] * df["volume"]).resample(**kova).sum()
        out["vwap"] = pv / out["volume"].where(out["volume"] > 0)
    return out.dropna(subset=["open", "high", "low", "close"])


# ---------------------------------------------------------------------------
# yfinance implementasyonu
# ---------------------------------------------------------------------------
def _build_yf_session():
    """Yahoo'nun bot tespitini (User-Agent / TLS parmak izi) aşmak için tek bir
    tarayıcı-kimlikli oturum üretir — canlı modda 60 sn'de bir atılan isteklerde
    HTTP 429 (rate-limit / geçici IP engeli) riskini düşürür.

    yfinance >= 1.x curl_cffi ile gelir; ``new_session()`` 'chrome' impersonation
    yapan bir oturum döndürür (TLS parmak izi + gerçekçi User-Agent — en sağlam yol).
    curl_cffi yoksa ya da eski sürümlerde ``new_session`` tanımlı değilse, özel bir
    User-Agent başlıklı düz ``requests.Session`` 'a düşeriz. Hiçbiri kurulamazsa
    ``None`` döner; yfinance kendi varsayılan oturumunu üretir (davranış bozulmaz).
    """
    try:
        from yfinance._http import new_session
        return new_session()
    except Exception:
        try:
            import requests
            s = requests.Session()
            s.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            })
            return s
        except Exception as e:  # requests bile yoksa — sessiz geç, yfinance kendi halleder
            logger.debug(f"[data] custom yf session olusturulamadi: {e}")
            return None


class YFinanceProvider(DataProvider):
    name = "yfinance"

    _yf_period: ClassVar[dict[str, str]] = {"1m": "5d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "5y"}

    def __init__(self) -> None:
        import threading
        # Madde 4 (Yahoo engeli) — tüm isteklerde paylaşılan tarayıcı-kimlikli oturum.
        self._session = _build_yf_session()
        # Madde 1 — Ticker nesnelerini sembol bazında cache'le. Her get_quote/
        # get_option_chain çağrısında yeni yf.Ticker üretmek yfinance tarafında
        # gereksiz kaynak/cookie kurulumu yaratır. Tarayıcılar arası (scanner)
        # thread'lerden erişildiği için kilitle.
        self._tickers: dict[str, yf.Ticker] = {}
        self._tickers_lock = threading.Lock()

    def _ticker(self, symbol: str) -> yf.Ticker:
        """Sembol için cache'lenmiş Ticker (paylaşılan oturuma bağlı) döndürür."""
        with self._tickers_lock:
            tkr = self._tickers.get(symbol)
            if tkr is None:
                tkr = yf.Ticker(symbol, session=self._session)
                self._tickers[symbol] = tkr
            return tkr

    def get_bars(self, symbol: str, interval: Interval, lookback_bars: int = 200) -> pd.DataFrame:
        period = self._yf_period.get(interval, "60d")
        df = yf.download(
            symbol, period=period, interval=interval,
            auto_adjust=True, progress=False, prepost=False,  # Madde 22 — sadece RTH
            threads=False,   # tek sembol; TierScanner thread'leriyle çakışmasın (kontrolsüz pool)
            session=self._session,  # Madde 4 — paylaşılan tarayıcı-kimlikli oturum
        )
        if df.empty:
            logger.warning(f"[data] empty bars: {symbol} {interval}")
            return df

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        # Madde 3 — Look-ahead: oluşmakta olan son barı dışla.
        # yfinance intraday verisinde sonuncusu yarı-açık olabilir; güvenli yol son barı atmak.
        if interval != "1d" and len(df) > 1:
            from nihai.config import INTERVAL_MINUTES
            now_local = datetime.now(tz=df.index.tz) if df.index.tz else datetime.now()
            bar_delta = pd.Timedelta(minutes=INTERVAL_MINUTES.get(interval, 60))
            if now_local - df.index[-1].to_pydatetime() < bar_delta:
                df = df.iloc[:-1]

        # Tur-90 (BULGU-90-5): sistem sözleşmesi "naive = UTC"dir (Alpaca yolu, `_et_gunu`,
        # `_canon_ts`). yfinance gün-içini ET-tz'li, günlüğü naive TARİH döndürüyordu → yedeğe
        # düşülen her çağrıda aynı barın zamanı değişiyor, `hour` özelliği UTC ↔ ET atlıyordu.
        idx = df.index
        if idx.tz is None:        # günlük: naive tarih = ET gece yarısı (Alpaca 1d ile aynı etiket)
            idx = idx.tz_localize("America/New_York")
        df.index = idx.tz_convert("UTC").tz_localize(None)
        # Tur-91 (BULGU-91-3): seans sonrası son GÜNLÜK satır OHLC'siz gelir (hacim dolu; 2026-09-23'te
        # 25/25 sembol). Alpaca yolu ve backtest bu satırı atar; yedek atmıyordu → son kapanış NaN.
        return df.dropna(subset=["open", "high", "low", "close"]).tail(lookback_bars)

    def get_quote(self, symbol: str) -> Quote | None:
        try:
            # Madde 1 — Ticker cache'li; ama fast_info bid/ask/last'ı INSTANCE üzerinde
            # kalıcı cache'ler. Cache'lenmiş Ticker'ı doğrudan kullanırsak canlı kotasyon
            # ilk değerde DONAR. Bu yüzden Ticker'ı (oturum/cookie) tekrar kullanıp her
            # çağrıda TAZE bir FastInfo kuruyoruz — hem #1 kazancı hem güncel fiyat.
            tkr = self._ticker(symbol)
            try:
                from yfinance.scrapers.quote import FastInfo
                info = FastInfo(tkr)
            except Exception:
                tkr._fast_info = None  # son çare: instance cache'ini sıfırla, taze çek
                info = tkr.fast_info
            bid = float(getattr(info, "bid", 0) or 0)
            ask = float(getattr(info, "ask", 0) or 0)
            last = float(getattr(info, "last_price", 0) or 0)
            measured = 0 < bid <= ask < float("inf")
            # yfinance bid/ask sık 0 döner — last'tan sentetik üret.
            if bid == 0 or ask == 0:
                if last > 0:
                    bid = last * 0.999
                    ask = last * 1.001
            return Quote(symbol=symbol, bid=bid, ask=ask, last=last, timestamp=datetime.now(),
                         spread_measured=measured)
        except Exception as e:
            # Kurtarılabilir (None döner; failover/health-check ele alır). Çoğu zaman
            # geçici ağ/yfinance iç hatası (örn. 'exchangeTimezoneName' eksik alanı).
            logger.warning(f"[data] quote error {symbol}: {e}")
            return None

    def get_option_chain(self, symbol: str, expiry: str | None = None) -> dict:
        # Madde 1 — cache'li Ticker. option_chain()/options her çağrıda Yahoo'dan
        # taze çeker (sonucu instance'ta cache'lemez), bu yüzden tekrar kullanımı güvenli.
        tkr = self._ticker(symbol)
        try:
            expiries = list(tkr.options or [])
        except Exception as e:
            logger.warning(f"[data] options expiries fail {symbol}: {e}")   # kurtarılabilir → {} döner
            return {}

        if not expiries:
            return {}

        expiry = expiry or expiries[0]
        try:
            chain = tkr.option_chain(expiry)
        except Exception as e:
            logger.warning(f"[data] option_chain fail {symbol} {expiry}: {e}")  # kurtarılabilir → {} döner
            return {}

        underlying = self.get_quote(symbol)
        return {
            "expiry": expiry,
            "all_expiries": expiries,
            "calls": chain.calls,
            "puts": chain.puts,
            "underlying_price": underlying.last if underlying else 0.0,
        }


# ---------------------------------------------------------------------------
# FailoverProvider — Madde 45: birincil kopunca otomatik yedeğe geç
# ---------------------------------------------------------------------------
class FailoverProvider(DataProvider):
    """İki provider sarmalar — primary boş/hata dönerse otomatik secondary'ye geç."""

    def __init__(self, primary: DataProvider, secondary: DataProvider):
        self.primary = primary
        self.secondary = secondary
        self.name = f"{primary.name}+{secondary.name}_failover"
        # Hangi sağlayıcının kaç kez servis ettiği (gözlemlenebilirlik — "Alpaca'dan mı geliyor?")
        self.primary_hits = 0
        self.secondary_hits = 0
        self.last_source: str | None = None
        # Hangi sembolün kaç kez yedeğe düştüğü (kalıcı failover'ları teşhis için —
        # ör. Alpaca'da olmayan/yeniden adlandırılmış ticker hep buraya birikir).
        self.secondary_by_symbol: dict[str, int] = {}

    def source_stats(self) -> dict:
        """Aktif veri kaynağı + isabet sayaçları (pano/health için)."""
        top = sorted(self.secondary_by_symbol.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "active": self.last_source,
            "primary": self.primary.name, "primary_hits": self.primary_hits,
            "secondary": self.secondary.name, "secondary_hits": self.secondary_hits,
            # En çok yedeğe düşen ilk 20 sembol (kronik failover'ı görünür kılar)
            "secondary_by_symbol": dict(top[:20]),
        }

    def _try(self, method_name: str, *args, **kwargs):
        sym = str(args[0]).upper() if args else "?"
        for prov, label in ((self.primary, "primary"), (self.secondary, "secondary")):
            try:
                result = getattr(prov, method_name)(*args, **kwargs)
                # DataFrame veya dict için boş kontrolü
                if hasattr(result, "empty") and result.empty:
                    logger.debug(f"[failover] {label} {prov.name}.{method_name} bos")
                    continue
                if isinstance(result, dict) and not result:
                    continue
                if result is None:
                    continue
                # Başarılı servis — kaynak sayacını güncelle
                if label == "primary":
                    self.primary_hits += 1
                else:
                    self.secondary_hits += 1
                    self.secondary_by_symbol[sym] = self.secondary_by_symbol.get(sym, 0) + 1
                    logger.info(f"[failover] secondary ({prov.name}) kullanildi: {sym}.{method_name}")
                self.last_source = prov.name
                return result
            except Exception as e:
                logger.warning(f"[failover] {label} {prov.name}.{method_name} fail: {e}")
        # İkisi de fail olduysa son denenenin (boş) sonucu döner
        if method_name == "get_bars":
            return pd.DataFrame()
        return None if method_name == "get_quote" else {}

    def get_bars(self, symbol, interval, lookback_bars=200):
        return self._try("get_bars", symbol, interval, lookback_bars)

    def get_quote(self, symbol):
        return self._try("get_quote", symbol)

    def get_option_chain(self, symbol, expiry=None):
        return self._try("get_option_chain", symbol, expiry)


# ---------------------------------------------------------------------------
# CachingProvider — Madde 48: kısa TTL ile REST çağrılarını azalt (rate-limit)
# ---------------------------------------------------------------------------
class CachingProvider(DataProvider):
    """Herhangi bir provider'ı sarmalar; get_bars/get_quote/get_option_chain
    sonuçlarını (method, args) anahtarıyla TTL süresince cache'ler. Thread-safe
    (web thread + ana döngü aynı provider'ı çağırır)."""

    def __init__(self, inner: DataProvider, ttl_sec: float):
        import os
        import threading
        self.inner = inner
        self.ttl = ttl_sec
        # Quote'lar AYRI (kısa) TTL — canlı fiyat ~1 dk bayatlamasın. Stream quote'u
        # bellekte (bedava) olduğundan kısa TTL rate-limit'e dokunmaz; bar/opsiyon
        # uzun TTL'de kalır (REST rate-limit koruması). QUOTE_CACHE_TTL_SEC ile ayarlı.
        self.quote_ttl = max(0.0, float(os.getenv("QUOTE_CACHE_TTL_SEC", "1.5")))
        self.name = f"{inner.name}+cache{int(ttl_sec)}s"
        self._cache: dict = {}
        self._lock = threading.Lock()

    @staticmethod
    def _is_cacheable(val) -> bool:
        """Boş/başarısız sonuçları cache'leme — yoksa geçici hata TTL boyunca kilitlenir."""
        if val is None:
            return False
        if hasattr(val, "empty"):       # DataFrame
            return not val.empty
        if isinstance(val, dict):       # option chain
            return bool(val)
        return True

    def _get(self, key, fn, ttl=None):
        import time as _t
        eff_ttl = self.ttl if ttl is None else ttl
        now = _t.monotonic()
        if eff_ttl > 0:
            with self._lock:
                hit = self._cache.get(key)
                if hit and (now - hit[0]) < eff_ttl:
                    return hit[1]
        val = fn()  # ağ çağrısı kilit dışında
        if eff_ttl > 0 and self._is_cacheable(val):
            with self._lock:
                self._cache[key] = (now, val)
        return val

    def get_bars(self, symbol, interval, lookback_bars=200):
        """ÜST-KÜME yeniden kullanımı: anahtar (sembol, interval); girdide kaç bar İSTENDİĞİ
        saklanır ve istenen ≥ yeni istek ise `tail(n)` ile servis edilir.

        Tur-81 (BULGU-81-2): anahtar `lookback_bars`i İÇERİYORDU. Aynı süreçte aynı
        (sembol, "1m") için üç ayrı çağıran var — tahmin servisi 220, korelasyon 60,
        `prediction_tracker.resolve` 400 — ve hiçbiri ötekinin girdisini kullanamıyordu;
        400'lük çerçeve ötekilerin ÜST KÜMESİ olmasına rağmen. ÖLÇÜLDÜ (sayaçlı stub,
        canlı karışım [220,60,400,220,60,220,400,220,60,400]): TTL penceresi başına
        **3 upstream çağrı**; üst-küme ile **2**. Üç lookback'in altı sıralanışında
        toplam 18/18 → 11/18. Bu sarmalayıcı hem tarayıcı sürecinde hem HUB'da kuruludur,
        yani kazanç iki yerde birden alınır.

        Bayatlama sözleşmesi DEĞİŞMEZ: dönen çerçeve, girdi anındaki "en yeni n bar"dır —
        aynı TTL penceresinde taze bir çağrının döndüreceğiyle aynı. Daha AZ bar istenmesi
        girdiyi DARALTMAZ (yarışta küçük istek büyüğün üstüne yazamaz)."""
        import time as _t
        if self.ttl <= 0:
            return self.inner.get_bars(symbol, interval, lookback_bars)
        key = ("bars", symbol, interval)
        now = _t.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit and (now - hit[0]) < self.ttl and hit[2] >= lookback_bars:
                val = hit[1]
                return val.tail(lookback_bars) if len(val) > lookback_bars else val
        val = self.inner.get_bars(symbol, interval, lookback_bars)
        if self._is_cacheable(val):
            with self._lock:
                prev = self._cache.get(key)
                if not (prev and (now - prev[0]) < self.ttl and prev[2] > lookback_bars):
                    self._cache[key] = (now, val, lookback_bars)
        return val

    def get_quote(self, symbol):
        return self._get(("quote", symbol), lambda: self.inner.get_quote(symbol),
                         ttl=self.quote_ttl)

    def get_option_chain(self, symbol, expiry=None):
        return self._get(("chain", symbol, expiry),
                         lambda: self.inner.get_option_chain(symbol, expiry))

    def is_quote_healthy(self, q: Quote) -> tuple[bool, str]:
        return self.inner.is_quote_healthy(q)

    def source_stats(self) -> dict:
        """İçteki FailoverProvider'ın kaynak istatistiklerini iletir (varsa)."""
        fn = getattr(self.inner, "source_stats", None)
        return fn() if fn else {}


# ---------------------------------------------------------------------------
# HubProvider — process-dışı veri hub'ından (hub_server.py) HTTP ile okur
# ---------------------------------------------------------------------------
class HubProvider(DataProvider):
    """Alpaca'ya doğrudan bağlanmak yerine yerel hub_server.py'ye HTTP ile sorar.

    Birden çok proje/varyant aynı hub'ı paylaşır → tek upstream bağlantı, özdeş veri.
    Hub erişilemezse boş/None döner → dıştaki FailoverProvider yfinance'e geçer (Madde 45).
    """
    name = "hub"

    # --- DEVRE KESİCİ (Ek Tur 21, ÖLÇÜM: hub KAPALIYKEN her çağrı 2,03 sn) -------------
    # Ölçüm: hub kapalıyken 10 ardışık get_bars = 2,02–2,06 sn/çağrı (ilk-çağrı artefaktı
    # DEĞİL, her seferinde). Hub'ın kapalı olması istisna değil: watchdog yeniden başlatana
    # kadar ~30 sn, reboot sonrası saatler (2026-07-18'de 10+ saat). O sürede taramanın her
    # sembolü ve panonun her isteği bu bedeli ayrı ayrı ödüyordu — sonunda zaten yfinance'e
    # düşerek. Kesici açıkken hub ATLANIR, failover ANINDA yedeğe geçer (Madde 45 korunur).
    #
    # YALNIZ BAĞLANTI hatası kesiciyi açar; HTTP durum hatası (404/500) AÇMAZ — tek bir
    # desteklenmeyen uç yüzünden ayakta olan hub'ın tamamı 30 sn devre dışı kalmamalı.
    # `HUB_BREAKER_SEC=0` kesiciyi kapatır (eski davranış).
    _down_until: ClassVar[dict[str, float]] = {}   # base_url → ne zamana kadar kapalı

    @staticmethod
    def _breaker_sec() -> float:
        import os
        try:
            return float(os.getenv("HUB_BREAKER_SEC", "30"))
        except ValueError:
            return 30.0

    def _breaker_open(self) -> bool:
        import time
        return time.monotonic() < self._down_until.get(self.base_url, 0.0)

    def _breaker_trip(self) -> None:
        import time
        sec = self._breaker_sec()
        if sec > 0:
            HubProvider._down_until[self.base_url] = time.monotonic() + sec

    def __init__(self, base_url: str | None = None, timeout: float = 6.0):
        import os
        self.base_url = (base_url or os.getenv("HUB_URL") or "http://127.0.0.1:8799").rstrip("/")
        self.timeout = timeout
        import httpx
        # Tur-106 (BULGU-106-1): boşta bağlantı ömrü SUNUCUNUNKİNDEN KISA olmalı. İkisi de 5 sn iken
        # (httpx varsayılanı = uvicorn `timeout_keep_alive`) ~5 sn ritimli döngü, sunucunun tam o an
        # kapattığı bağlantıyı yeniden kullanıyordu → "Server disconnected" / WinError 10053-10054,
        # canlıda saatte 50-98 uyarı ve her biri yfinance yedeğine düşüş. Ayrı süreçli yük
        # deneyinde 1020 istekte 10-12 hata ↔ 3 sn ile 0/990.
        self._client = httpx.Client(timeout=timeout, limits=httpx.Limits(keepalive_expiry=3.0))

    def _get(self, path: str, params: dict):
        if self._breaker_open():
            logger.debug(f"[hub] devre kesici açık — {path} atlandı (yedeğe düşülüyor)")
            return None
        try:
            r = self._client.get(self.base_url + path, params=params)
            r.raise_for_status()
            HubProvider._down_until.pop(self.base_url, None)   # hub ayakta → kesici sıfırla
            return r
        except Exception as e:
            import httpx
            # Tur-82 (BULGU-82-2): eskiden her `TransportError` (ReadTimeout DAHİL) kesiciyi
            # açıyordu — docstring "YALNIZ BAĞLANTI hatası" demesine rağmen. Ayakta ama YAVAŞ
            # bir hub'ın tek geç yanıtı böylece sürecin TÜM çağrılarını 30 sn yfinance'e
            # yolluyordu: 17 Eylül seansında 4546 zaman aşımı → 9344 atlanan çağrı → 13856
            # yfinance yedeği. Artık yalnız hub KAPALIYKEN (bağlantı kurulamıyor) açılır;
            # geç yanıt yalnız O çağrıyı yedeğe düşürür.
            if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError)):
                self._breaker_trip()
            logger.warning(f"[hub] {path} {params} erişilemedi: {e}")
            return None

    def get_bars(self, symbol: str, interval: Interval, lookback_bars: int = 200) -> pd.DataFrame:
        r = self._get("/bars", {"symbol": symbol.upper(), "interval": interval, "lookback": lookback_bars})
        if r is None:
            return pd.DataFrame()
        try:
            from io import StringIO
            df = pd.read_json(StringIO(r.text), orient="split")
            if df.empty:
                return df
            df.index = pd.to_datetime(df.index)
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            logger.warning(f"[hub] bars ayrıştırma hatası ({symbol}): {e}")
            return pd.DataFrame()

    def get_quote(self, symbol: str) -> Quote | None:
        r = self._get("/quote", {"symbol": symbol.upper()})
        if r is None:
            return None
        try:
            d = r.json()
            if not d:
                return None
            ts = d.get("timestamp")
            tsv = datetime.fromisoformat(ts) if ts else datetime.now()
            return Quote(symbol=d["symbol"], bid=float(d["bid"]), ask=float(d["ask"]),
                         last=float(d["last"]), timestamp=tsv,
                         spread_measured=d.get("spread_measured") is True)
        except Exception as e:
            logger.warning(f"[hub] quote ayrıştırma hatası ({symbol}): {e}")
            return None

    def get_option_chain(self, symbol: str, expiry: str | None = None) -> dict:
        params = {"symbol": symbol.upper()}
        if expiry:
            params["expiry"] = expiry
        r = self._get("/options", params)
        if r is None:
            return {}
        try:
            from io import StringIO
            d = r.json()
            if not d:
                return {}
            calls = pd.read_json(StringIO(d["calls"]), orient="split") if d.get("calls") else pd.DataFrame()
            puts = pd.read_json(StringIO(d["puts"]), orient="split") if d.get("puts") else pd.DataFrame()
            return {"calls": calls, "puts": puts, "expiry": d.get("expiry"),
                    "underlying_price": d.get("underlying_price"),
                    "all_expiries": d.get("all_expiries")}
        except Exception as e:
            logger.warning(f"[hub] options ayrıştırma hatası ({symbol}): {e}")
            return {}

    def asset_flags(self, symbol: str) -> dict | None:
        """Madde 30 borç bayraklarını HUB'dan oku (Tur-58).

        Bu metot olmadan `asset_flags()` zincir gezintisi hub düğümünde biter ve
        modül-düzeyi yedek YOLU kendi `AlpacaProvider()`'ını kurar: hub'ın tek-upstream
        amacı delinir, her tüketici süreç ayrı bir trading istemcisi açar. Hub cevap
        vermezse (kesici/uç yok/boş gövde) **None** döner → çağıran modül-yedeğine
        düşer; "bilinmiyor" asla "satılabilir"e çevrilmez."""
        r = self._get("/asset_flags", {"symbol": symbol.upper()})
        if r is None:
            return None
        try:
            d = r.json()
        except Exception as e:  # noqa: BLE001 — teşhis; çağıranı düşürmez
            logger.debug(f"[hub] asset_flags ayrıştırılamadı ({symbol}): {e}")
            return None
        return d or None

    def stream_status(self) -> dict | None:
        """WebSocket akış karnesini HUB'dan oku (Tur-89).

        Akış hub SÜRECİNDE koşar; `DATA_PROVIDER=hub` iken tüketici sürecin zincirinde akış
        düğümü yoktu ve pano "akış sağlayıcısı kayıtlı değil" basıyordu (akış bağlıyken)."""
        r = self._get("/streams", {})
        if r is None:
            return None
        try:
            d = r.json()
        except Exception as e:  # noqa: BLE001 — teşhis; çağıranı düşürmez
            logger.debug(f"[hub] streams ayrıştırılamadı: {e}")
            return None
        return d.get("streams") if isinstance(d, dict) else None


# ---------------------------------------------------------------------------
# Açığa satış bayrakları — TEK okuyucu (Tur-57)
# ---------------------------------------------------------------------------
_FLAG_PROVIDER = None


def asset_flags(symbol: str, provider: DataProvider | None = None) -> dict | None:
    """`shortable` / `easy_to_borrow` / `marginable` — sarmalayıcı zincirini gezerek bul.

    Sağlayıcı zinciri (`CachingProvider.inner`, `FailoverProvider.primary/secondary`)
    içinde `asset_flags` taşıyan ilk sağlayıcıya sorar; hiçbiri yoksa Alpaca'nın ÜCRETSİZ
    trading API'sine düşer (bayraklar veri planına değil hesaba bağlıdır — 2026-08-29'da
    ölçüldü). Anahtar/SDK yoksa **None** döner: "bilinmiyor" ile "satılabilir" AYNI ŞEY
    DEĞİLDİR ve çağıran ikisini ayırt edebilmelidir.

    İKİ çağıranı var (`execution/main.py` SHORT yolu ve `web/api_quant` akış paneli);
    ikinci bir kopya yazmak aynı zinciri iki farklı biçimde gezme riskini doğururdu."""
    node = provider
    for _ in range(4):
        if node is None:
            break
        fn = getattr(node, "asset_flags", None)
        if callable(fn):
            try:
                got = fn(symbol)
            except Exception as e:  # noqa: BLE001 — teşhis katmanı, çağıranı düşürmez
                # ARIZA ile ÖLÇÜM YOKLUĞU ayrı şeylerdir. Patlayan bir sağlayıcı bozuktur;
                # onu atlayıp doğrudan Alpaca'ya gitmek arızayı GİZLER ve her tüketici
                # süreçte yeni bir istemci açar. Arızada zincir BİTER (Tur-57 sözleşmesi).
                logger.debug(f"[flags] {symbol} sağlayıcıdan okunamadı: {e}")
                return None
            # Tur-58: BOŞ cevap zinciri bitirmez. HubProvider bu metodu kazandığında
            # (hub kapalıyken None döner) eski "ilk taşıyanın cevabını döndür" kuralı
            # modül-düzeyi Alpaca yedeğini erişilemez kılıyordu — yani bir uç EKLEMEK
            # mevcut bir yolu sessizce kapatırdı. Ölçüm YOKSA arama devam eder.
            if got:
                return got
        node = getattr(node, "inner", None) or getattr(node, "primary", None)
    global _FLAG_PROVIDER
    if _FLAG_PROVIDER is None:
        try:
            from nihai.data.alpaca_provider import AlpacaProvider
            _FLAG_PROVIDER = AlpacaProvider()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[flags] Alpaca sağlayıcısı kurulamadı: {e}")
            return None
    try:
        return _FLAG_PROVIDER.asset_flags(symbol)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[flags] {symbol} bayrakları okunamadı: {e}")
        return None


# ---------------------------------------------------------------------------
# Akış sağlığı — TEK okuyucu (Tur-58; `asset_flags`in kardeşi)
# ---------------------------------------------------------------------------
def stream_status(provider: DataProvider | None) -> dict | None:
    """WebSocket akış karnesi — sarmalayıcı zincirini gezerek bul (yoksa None).

    `live_predict_service._find_stream_status` bunun tek kopyasıydı ve karne SADECE
    o servisin `snapshot()`'ında yüzeye çıkıyordu; pano ve `/metrics` aynı bilgiyi
    okumak isteyince ikinci bir kopya doğacaktı. Tur-35 dersi (aynı desen üç modülde
    kopyalanmıştı) burada ÖNCEDEN uygulanır: gezinti tek yerde durur.

    `asset_flags`ten farkı, burada ARIZA ile ölçüm yokluğunun ayrılmasına gerek
    olmamasıdır: akış karnesi bir TEŞHİSTİR, karar yolunda değildir."""
    seen: set[int] = set()
    stack = [provider]
    while stack:
        p = stack.pop()
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        fn = getattr(p, "stream_status", None)
        if callable(fn):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 — teşhis katmanı, çağıranı düşürmez
                logger.debug(f"[stream] karne okunamadı: {e}")
                return None
        stack.extend(getattr(p, attr) for attr in ("inner", "primary", "secondary")
                     if hasattr(p, attr))
    return None


# ---------------------------------------------------------------------------
# Factory — DATA_PROVIDER env'inden seçim
# ---------------------------------------------------------------------------
def _make_single(selection: str) -> DataProvider:
    s = selection.lower()
    if s in ("yfinance", "yf"):
        return YFinanceProvider()
    if s in ("tradingview", "tv"):
        from nihai.data.tv_provider import TradingViewProvider
        return TradingViewProvider()
    if s == "alpaca":
        from nihai.data.alpaca_provider import AlpacaProvider
        return AlpacaProvider()
    if s in ("alpaca_stream", "alpaca_ws"):
        from nihai.data.alpaca_stream_provider import AlpacaStreamProvider
        prov = AlpacaStreamProvider()
        prov.start()   # WS thread'ini ayağa kaldır (anahtar yoksa pasif kalır, REST'e düşer)
        return prov
    if s == "hub":
        return HubProvider()   # yerel hub_server.py'den HTTP ile okur (tek upstream bağlantı)
    raise ValueError(f"Bilinmeyen DATA_PROVIDER: {selection!r}")


def get_provider(name: str | None = None, *, with_failover: bool = True,
                 with_cache: bool = True) -> DataProvider:
    """
    Madde 45 — provider seçim + otomatik failover (+ Madde 48 cache).

    Default with_failover=True: primary fail/empty olursa diğerine düşer.
    with_cache=True: sonuçlar CONFIG'deki kısa TTL ile cache'lenir (rate-limit koruması).
    """
    import os
    primary_name = (name or os.getenv("DATA_PROVIDER") or "yfinance").lower()
    primary = _make_single(primary_name)

    if with_failover:
        # yfinance her zaman güvenli yedek (anahtarsız, opsiyon zinciri verir).
        # Primary yfinance ise yedek tradingview.
        secondary_name = "tradingview" if primary_name in ("yfinance", "yf") else "yfinance"
        try:
            secondary = _make_single(secondary_name)
            provider: DataProvider = FailoverProvider(primary, secondary)
        except Exception as e:
            logger.warning(f"[provider] secondary ({secondary_name}) yuklenmedi, "
                           f"failover devre disi: {e}")
            provider = primary
    else:
        provider = primary

    if with_cache:
        from nihai.config import DATA_CACHE_TTL_SEC
        if DATA_CACHE_TTL_SEC > 0:
            provider = CachingProvider(provider, DATA_CACHE_TTL_SEC)
    return provider
