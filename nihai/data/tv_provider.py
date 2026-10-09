"""
tv_provider.py — TradingView veri sağlayıcısı (tvDatafeed üzerinden).

UYARI: tvDatafeed RESMİ DEĞİL — TradingView'in iç WebSocket protokolüne
reverse-engineering ile bağlanır. TV protokolü değişirse kütüphanenin
güncellenmesi gerekir. Kişisel/araştırma kullanımı yaygın; ToS riskini
kullanıcı üstlenir.

PDF Referansları:
- Madde 34 : Veri sağlığı kontrolü (DataProvider base'inden gelir)
- Madde 45 : Veri yedekliliği — opsiyon zinciri için yfinance fallback'i kullanılır
- Madde 48 : WebSocket gecikmeyi düşürür (TV WS doğal olarak gerçek-zamanlı*)

(*) Login olmadan ~delayed; login ile real-time (TV plan'ına göre).

Kurulum:
    pip install tvdatafeed
    # PyPI sürümü güncel değilse:
    pip install --upgrade git+https://github.com/StreamAlpha/tvdatafeed.git
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd
from loguru import logger

from nihai.data.data_provider import DataProvider, Interval, Quote, YFinanceProvider

# tvDatafeed kütüphanesi, bizim failover'la sessizce ele aldığımız bağlantı/boş-veri
# durumlarını kendi stdlib logger'ından stderr'e ERROR olarak basar (örn.
# "Connection timed out", "no data..."). Gerçek aksiyonu biz alıyoruz → gürültüyü kıs.
for _name in ("tvDatafeed", "tvDatafeed.main"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Sembol → Exchange eşlemesi
# ---------------------------------------------------------------------------
# TV'de "AAPL" yetmez, "NASDAQ:AAPL" gerek. Çoğu popüler hisse NASDAQ veya NYSE.
DEFAULT_EXCHANGE_MAP: dict[str, str] = {
    # NASDAQ Tech
    "AAPL": "NASDAQ", "MSFT": "NASDAQ", "GOOGL": "NASDAQ", "GOOG": "NASDAQ",
    "AMZN": "NASDAQ", "META": "NASDAQ", "NVDA": "NASDAQ", "TSLA": "NASDAQ",
    "AVGO": "NASDAQ", "AMD": "NASDAQ", "INTC": "NASDAQ", "NFLX": "NASDAQ",
    "ADBE": "NASDAQ", "CSCO": "NASDAQ", "PEP": "NASDAQ", "COST": "NASDAQ",
    "QQQ": "NASDAQ",
    # NYSE
    "JPM": "NYSE", "BAC": "NYSE", "WMT": "NYSE", "V": "NYSE", "MA": "NYSE",
    "DIS": "NYSE", "PFE": "NYSE", "KO": "NYSE", "XOM": "NYSE", "CVX": "NYSE",
    # AMEX (ETF)
    "SPY": "AMEX", "DIA": "AMEX", "IWM": "AMEX", "VTI": "AMEX", "GLD": "AMEX",
    "SLV": "AMEX", "USO": "AMEX", "TLT": "NASDAQ", "VXX": "BATS",
    # Endeksler (cash settled — Madde 29)
    "SPX": "SP", "NDX": "NASDAQ", "RUT": "TVC", "VIX": "TVC",
    # Forex / Crypto (TV güçlü tarafları)
    "BTCUSD": "BINANCE", "ETHUSD": "BINANCE", "EURUSD": "FX",
}


def resolve_exchange(symbol: str) -> str:
    """Bilinen sembol için exchange döner. Bilinmiyorsa NASDAQ varsayar (en geniş kapsam)."""
    sym = symbol.upper()
    # Önce env'den özel mapping kontrolü (TV_EXCHANGE_MAP=AAPL:NASDAQ,SPY:AMEX,...)
    custom = os.getenv("TV_EXCHANGE_MAP", "")
    if custom:
        for entry in custom.split(","):
            if ":" in entry:
                k, v = entry.split(":", 1)
                if k.strip().upper() == sym:
                    return v.strip().upper()
    return DEFAULT_EXCHANGE_MAP.get(sym, "NASDAQ")


# ---------------------------------------------------------------------------
# Interval mapping
# ---------------------------------------------------------------------------
def _tv_interval(interval: Interval):
    """Kendi 1m/1h/1d notasyonunu tvDatafeed Interval enum'una çevir."""
    from tvDatafeed import Interval as TVI  # local import — kütüphane yoksa hata burada
    mapping = {
        "1m": TVI.in_1_minute,
        "5m": TVI.in_5_minute,
        "15m": TVI.in_15_minute,
        "1h": TVI.in_1_hour,
        "1d": TVI.in_daily,
    }
    if interval not in mapping:
        raise ValueError(f"Desteklenmeyen interval: {interval}")
    return mapping[interval]


# ---------------------------------------------------------------------------
# TradingViewProvider
# ---------------------------------------------------------------------------
class TradingViewProvider(DataProvider):
    name = "tradingview"

    def __init__(self, username: str | None = None, password: str | None = None,
                 yf_fallback: bool = True):
        """
        username/password verilirse real-time veri; yoksa delayed.
        yf_fallback=True → opsiyon zinciri yfinance'tan alınır (TV opsiyon vermiyor).
        """
        try:
            from tvDatafeed import TvDatafeed
        except ImportError as e:
            raise ImportError(
                "tvDatafeed kurulu değil. Şununla kur:\n"
                "  pip install tvdatafeed\n"
                "  # ya da:\n"
                "  pip install git+https://github.com/StreamAlpha/tvdatafeed.git"
            ) from e

        username = username or os.getenv("TV_USERNAME") or None
        password = password or os.getenv("TV_PASSWORD") or None
        # tvDatafeed'in TvDatafeed sınıfı username/password verilmezse anonim çalışır
        try:
            self._tv = TvDatafeed(username=username, password=password)
            mode = "AUTHENTICATED (real-time)" if username else "ANONYMOUS (delayed)"
            logger.info(f"[tv] TvDatafeed başlatıldı — {mode}")
        except Exception as e:
            logger.warning(f"[tv] login başarısız, anonim devam: {e}")
            self._tv = TvDatafeed()

        self._yf_fallback = YFinanceProvider() if yf_fallback else None

    # -----------------------------------------------------------------
    def get_bars(self, symbol: str, interval: Interval, lookback_bars: int = 200) -> pd.DataFrame:
        exch = resolve_exchange(symbol)
        try:
            df = self._tv.get_hist(
                symbol=symbol.upper(),
                exchange=exch,
                interval=_tv_interval(interval),
                n_bars=max(lookback_bars + 10, 50),   # buffer
            )
        except Exception as e:
            # Ağ/TV hatası kurtarılabilir: bars failover'ı dıştaki FailoverProvider yapar
            # (get_provider TV'yi her zaman yfinance ile sarar). Burada İÇ fallback YAPMA —
            # yoksa yfinance iki kez çağrılır (çift istek + çift "empty bars" logu). Boş dön.
            logger.warning(f"[tv] {exch}:{symbol} {interval} bar çekme hatası: {e}")
            return pd.DataFrame()

        if df is None or df.empty:
            logger.warning(f"[tv] {exch}:{symbol} {interval} boş döndü")
            return pd.DataFrame()

        # tvDatafeed çıktısı: index=datetime, columns=[symbol, open, high, low, close, volume]
        df = df.rename(columns=str.lower)
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])

        # Madde 3 — look-ahead guard: TV'nin canlı oluşmakta olan barı varsa at
        if interval != "1d" and len(df) > 1:
            now = datetime.now(tz=df.index.tz) if df.index.tz else datetime.now()
            unit_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1H", "1d": "1D"}
            try:
                bar_delta = pd.Timedelta(unit_map[interval])
                if now - df.index[-1].to_pydatetime() < bar_delta:
                    df = df.iloc[:-1]
            except Exception:
                pass

        return df.tail(lookback_bars)

    # -----------------------------------------------------------------
    def get_quote(self, symbol: str) -> Quote | None:
        """TV'de quote endpoint yok — son 1m barın close'unu proxy olarak kullan."""
        try:
            df = self.get_bars(symbol, "1m", lookback_bars=2)
            if df.empty:
                return None
            last = df.iloc[-1]
            close = float(last["close"])
            # Sentetik bid/ask (TV makas vermez)
            return Quote(
                symbol=symbol, bid=close * 0.999, ask=close * 1.001,
                last=close, timestamp=df.index[-1].to_pydatetime()
                if hasattr(df.index[-1], "to_pydatetime") else datetime.now(),
            )
        except Exception as e:
            logger.error(f"[tv] quote proxy hatası {symbol}: {e}")
            return None

    # -----------------------------------------------------------------
    def get_option_chain(self, symbol: str, expiry: str | None = None) -> dict:
        """TV opsiyon zinciri vermiyor — yfinance fallback."""
        if self._yf_fallback:
            return self._yf_fallback.get_option_chain(symbol, expiry)
        logger.warning(f"[tv] opsiyon zinciri yok ve yf_fallback kapalı: {symbol}")
        return {}
