"""
alpaca_history.py — Alpaca'dan ARAŞTIRMA-kalitesi tarihsel bar çekimi (SIP, disk-cache'li).

Canlı AlpacaProvider'dan (alpaca_provider.py) farkı:
- Feed varsayılanı **SIP** (konsolide tüm-borsa): ücretsiz planda tarihsel SIP verisi
  son ~15 dk hariç serbesttir → `end` şimdi-16dk'ya KIRPILIR. IEX feed'i yalnız IEX
  hacmini görür (~%2-3) — eğitim setinde hacim özellikleri için SIP şart.
- `start`/`end` tarih aralığı destekler (canlı sağlayıcı n-bar tabanlıdır).
- Delist edilmiş semboller de döner (Alpaca inaktif varlık geçmişi tutar) →
  point-in-time (survivorship-temiz) evrenlerle kullanılabilir (ACI DERS #1).
- Disk cache: DATA_DIR/alpaca_hist/{SYM}_{interval}[_{adj}].csv — dataset yeniden
  koşularında Alpaca'ya tekrar gidilmez (200 istek/dk limiti + nezaket).
- **Bölünme/temettü düzeltmesi (BULGU-19):** `adjustment` artık isteğe yazılır
  (env `ALPACA_HIST_ADJUSTMENT`, varsayılan `all`). Öncesinde parametre hiç
  gönderilmiyordu → API varsayılanı `raw` = düzeltmesiz barlar. Düzeltme kipi cache
  dosya adına girer; `raw` eski adı korur (eski dosyalar bozulmadan kalır ama artık
  okunmaz → bir kez yeniden çekim gerekir).

Normalize, canlı sağlayıcıyla AYNI koddan geçer (_normalize_alpaca_bars) → format
tutarlılığı (küçük-harf OHLCV, tz-naive index). KENAR İDDİASI YOK — bu modül yalnız
veri taşır.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from nihai.config import DATA_DIR
from nihai.data.alpaca_provider import (  # noqa: F401 — _VALID_ADJUSTMENT dışa açık (testler)
    _VALID_ADJUSTMENT,
    _adjustment,
    _normalize_alpaca_bars,
    _to_alpaca_symbol,
)
from nihai.data.data_provider import Interval

HIST_DIR = Path(os.getenv("ALPACA_HIST_DIR", str(DATA_DIR / "alpaca_hist")))

# SIP kuralı: ücretsiz planda "son 15 dk" sorgulanamaz → güvenli kırpma payı.
_SIP_SAFETY = timedelta(minutes=16)

# --- Kurumsal işlem düzeltmesi (BULGU-19, 2026-08-01) -----------------------
# Alpaca SDK'da StockBarsRequest.adjustment varsayılanı None → API varsayılanı
# 'raw' = DÜZELTİLMEMİŞ. Kanıt: GE_1d.csv 2021-07-30 12.95 → 2021-08-02 100.60
# (1:8 ters bölünme). PIT S&P500 evreninde 43/505 sembol (%8,5) etkileniyordu —
# NVDA/AMZN/GOOGL/AVGO/NFLX/CMG/TSLA/WMT... Backtest'te GE tek başına -%981
# getiri üretti (özsermaye negatife düştü); canlıda da bölünme sonrası ~200 barlık
# lookback penceresi haftalarca sahte -%90 boşluk taşır (tüm göstergeler bozulur).
# Varsayılan 'all' — backtest.py'nin yfinance dalı zaten auto_adjust=True kullanıyor,
# bu iki yolu hizalar. Tur-90: "canlıda da" cümlesi 7 hafta yalnız YAZILI kaldı — canlı
# `AlpacaProvider.get_bars` ham barla koşuyordu. `_adjustment` artık `alpaca_provider`'da
# TEK kaynaktır (yukarıdaki içe aktarma); iki yol aynı kipi okur. Modül düzeyinde çözülür —
# `_fetch` imzası testlerin monkeypatch sözleşmesidir (5 pozisyonel argüman).

_client = None
_client_tried = False


def _get_client():
    """Lazy StockHistoricalDataClient. SDK/anahtar yoksa None (sessiz devre-dışı)."""
    global _client, _client_tried
    if _client is not None or _client_tried:
        return _client
    _client_tried = True
    key, sec = os.getenv("ALPACA_API_KEY", ""), os.getenv("ALPACA_SECRET_KEY", "")
    if not (key and sec):
        logger.warning("[alpaca_hist] ALPACA_API_KEY/SECRET yok — tarihsel çekim devre-dışı")
        return None
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        _client = StockHistoricalDataClient(key, sec)
    except Exception as e:
        logger.warning(f"[alpaca_hist] SDK yok/başlatılamadı: {e}")
        _client = None
    return _client


def _clamp_end(end: datetime | None) -> datetime:
    """SIP-güvenli bitiş: verilmemişse/geleceğe taşıyorsa şimdi-16dk."""
    lim = datetime.now(UTC) - _SIP_SAFETY
    if end is None:
        return lim
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return min(end, lim)


def _cache_path(symbol: str, interval: str, feed: str = "sip") -> Path:
    """Cache yolu — düzeltme kipi VE feed dosya adına GİRER (BULGU-19, Tur-84).

    'raw' eski (düzeltmesiz) formattır ve ESKİ adı korur: diskteki 2483 eski dosya
    bozulmaz/silinmez, yalnız artık okunmaz. Düzeltilmiş veri AYRI dosyaya yazılır →
    ham ve düzeltilmiş barlar aynı CSV'de ASLA karışmaz (karışsaydı sessizce
    bozuk fiyat serisi servis edilirdi).

    Tur-84: feed de aynı sınıftır. Anahtar feed'i taşımıyordu → `feed="iex"` isteği diskteki
    SIP dosyasını okuyabiliyor, tazelenirken de ÜSTÜNE IEX yazıyordu (IEX konsolide hacmin
    ~%3'ü; LMT'de dönemin 1m barlarının yarısı yok). Varsayılan 'sip' eski adı korur."""
    safe = symbol.upper().replace("/", "_").replace(".", "-")
    adj = _adjustment()
    suffix = "" if adj == "raw" else f"_{adj}"
    if feed.lower() != "sip":
        suffix += f"_{feed.lower()}"
    return HIST_DIR / f"{safe}_{interval}{suffix}.csv"


def _empty_meta_path(symbol: str, interval: str, feed: str = "sip") -> Path:
    """Boş (negatif) cache'in yanına yazılan kapsam metası: {"start": iso}."""
    return _cache_path(symbol, interval, feed).with_suffix(".empty.json")


def _read_cache(symbol: str, interval: str, start: datetime,
                feed: str = "sip") -> pd.DataFrame | None:
    """Cache'i oku; start'ı kapsıyor ve yeterince tazeyse kullan.

    Tazelik: ALPACA_HIST_CACHE_MAX_AGE_DAYS (varsayılan 1 = 'bugün yazılmış', eski
    davranış). Çok-günlük dataset koşularında büyük değer ver — yoksa gece yarısı
    tarih değişince TÜM cache bayat sayılır, her sembol ağdan yeniden çekilir
    (gözlem: v2 koşusu 27dk/sembole düştü, 429 fırtınasında semboller atlandı)."""
    p = _cache_path(symbol, interval, feed)
    if not p.exists():
        return None
    try:
        max_age = int(os.getenv("ALPACA_HIST_CACHE_MAX_AGE_DAYS", "1"))
        age = (datetime.now().date() - datetime.fromtimestamp(p.stat().st_mtime).date()).days
        if age >= max_age:
            return None                       # bayat cache → tazele
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if df.empty:
            # Boş-negatif cache yalnız KAPSADIĞI pencere için geçerlidir: start=S0 ile
            # yapılan boş çekim "S0'dan itibaren veri yok" der; daha ERKEN start isteyen
            # koşuya servis edilirse sembol (tipik olarak delist edilmiş = removed-since)
            # PIT dataset'inden sessizce düşer → survivorship yönlü sapma. Meta yoksa
            # (eski format) kapsama kanıtlanamaz → None (bir kez yeniden çekilir, meta yazılır).
            meta_p = _empty_meta_path(symbol, interval, feed)
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                cached_start = pd.Timestamp(meta["start"])
                if pd.Timestamp(start.replace(tzinfo=None)) >= cached_start:
                    return df                 # istenen pencere ⊆ boş çıkan pencere
            except Exception:
                pass
            return None
        # start'tan ≤7 takvim günü sonrasında başlıyorsa kapsıyor say (tatil payı)
        if df.index[0] <= pd.Timestamp(start.replace(tzinfo=None)) + pd.Timedelta(days=7):
            return df
    except Exception as e:
        logger.debug(f"[alpaca_hist] cache okunamadı {p.name}: {e}")
    return None


def _fetch(symbol: str, interval: Interval, start: datetime, end: datetime,
           feed: str) -> pd.DataFrame | None:
    """Alpaca REST çekimi → normalize df. Veri yok → boş df; HATA/istemci yok → None.

    Ayrım önemli (BULGU-14): None cache'LENMEZ — geçici bir ağ/rate-limit hatası
    'veri yok' diye negatif-cache'e yapışırsa sembol o günkü tüm koşulardan düşer."""
    client = _get_client()
    if client is None:
        return None
    asym = _to_alpaca_symbol(symbol)
    try:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        tf = {"1d": TimeFrame.Day, "1h": TimeFrame.Hour,
              "15m": TimeFrame(15, TimeFrameUnit.Minute),
              "5m": TimeFrame(5, TimeFrameUnit.Minute),
              "1m": TimeFrame.Minute}.get(interval, TimeFrame.Day)
        try:
            dfeed = DataFeed(feed.lower())
        except ValueError:
            dfeed = DataFeed.SIP
        from alpaca.data.enums import Adjustment
        req = StockBarsRequest(symbol_or_symbols=asym, timeframe=tf,
                               start=start, end=end, feed=dfeed,
                               adjustment=Adjustment(_adjustment()))
        raw = client.get_stock_bars(req).df
    except Exception as e:
        logger.warning(f"[alpaca_hist] {symbol} {interval} çekim hatası: {e}")
        return None
    # lookback = tümü (tarih aralığı zaten start/end ile sınırlı)
    n = len(raw) if raw is not None else 0
    return _normalize_alpaca_bars(raw, asym, interval, max(n, 1))


def get_history(symbol: str, interval: Interval = "1d", *,
                start: str | datetime, end: str | datetime | None = None,
                feed: str | None = None, use_cache: bool = True, fetch: bool = True) -> pd.DataFrame:
    """Sembolün tarihsel barları (küçük-harf OHLCV, tz-naive index).

    start/end str ('2021-07-01') veya datetime; end SIP-güvenli kırpılır.
    feed None → env ALPACA_HIST_FEED (varsayılan 'sip'). Boş df = veri yok/hata.
    fetch=False: yalnız diskteki kapsamı oku (bayat/kısmi olabilir); ağ/yazma yok.
    """
    if isinstance(start, str):
        start = datetime.fromisoformat(start).replace(tzinfo=UTC)
    elif start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if isinstance(end, str):
        end = datetime.fromisoformat(end).replace(tzinfo=UTC)
    end = _clamp_end(end)
    feed = (feed or os.getenv("ALPACA_HIST_FEED") or "sip").lower()

    if not fetch:
        try:
            df = pd.read_csv(_cache_path(symbol, interval, feed), index_col=0, parse_dates=True)
            if not isinstance(df.index, pd.DatetimeIndex):
                return pd.DataFrame()
        except (OSError, ValueError):
            return pd.DataFrame()
    elif use_cache:
        cached = _read_cache(symbol, interval, start, feed)
        if cached is not None:
            df = cached
        else:
            fetched = _fetch(symbol, interval, start, end, feed)
            if fetched is None:
                return pd.DataFrame()          # HATA → cache YAZMA (sonraki koşu yeniden dener)
            df = fetched
            try:
                HIST_DIR.mkdir(parents=True, exist_ok=True)
                df.to_csv(_cache_path(symbol, interval, feed))   # gerçek boş da yazılır (negatif cache)
                meta_p = _empty_meta_path(symbol, interval, feed)
                if df.empty:
                    # Kapsam metası: bu boşluk yalnız start'tan-itibaren pencereler için geçerli.
                    meta_p.write_text(json.dumps(
                        {"start": start.replace(tzinfo=None).isoformat()}), encoding="utf-8")
                elif meta_p.exists():
                    meta_p.unlink()            # dolu veri geldi → bayat boşluk metası temizlenir
            except Exception as e:
                logger.debug(f"[alpaca_hist] cache yazılamadı {symbol}: {e}")
    else:
        fetched = _fetch(symbol, interval, start, end, feed)
        if fetched is None:
            return pd.DataFrame()
        df = fetched

    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        return pd.DataFrame()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    # cache start'tan geniş olabilir → istenen pencereye kırp (tz-naive karşılaştırma)
    s_naive = pd.Timestamp(start.replace(tzinfo=None))
    e_naive = pd.Timestamp(end.replace(tzinfo=None))
    return df[(df.index >= s_naive) & (df.index <= e_naive)]
