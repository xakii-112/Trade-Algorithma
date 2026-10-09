"""
alpaca_stream_provider.py — Alpaca WebSocket (gerçek-zamanlı stream) veri sağlayıcısı.

REST polling yerine PUSH: arka plan thread'inde Alpaca StockDataStream'e (IEX dakika-bar
+ quote) abone olur, sembol-başı canlı tampon tutar. get_bars seed (REST, saatte ~1
çağrı) + canlı dakika-agregasyonu birleşimiyle servis eder → her taramada sembol-başı
REST çağrısı YOK (rate-limit çöker). get_quote canlı stream quote'unu döndürür.

ÜCRETSIZ IEX feed sınırı: ~30 sembol aboneliği + 1 eşzamanlı bağlantı. Bu yüzden EL ile
seçilen ≤30 sembol (STREAM_SYMBOLS) için tasarlandı; dinamik watchlist (100-500) sığmaz.

Zarif degrade: alpaca-py/anahtar yoksa VEYA piyasa kapalıysa (bar gelmez) → composed
REST AlpacaProvider'a (geçmiş bar) düşer. Sistem hiçbir koşulda çökmez.

.env:
    DATA_PROVIDER=alpaca_stream
    ALPACA_API_KEY=...  ALPACA_SECRET_KEY=...  ALPACA_FEED=iex
    STREAM_SYMBOLS=AAPL,MSFT,NVDA,...   # ≤30; boşsa erişilen ilk semboller (cap'li) abone edilir
    STREAM_MAX_SYMBOLS=30
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from loguru import logger

from nihai.config import INTERVAL_MINUTES
from nihai.data.alpaca_provider import AlpacaProvider, quote_diverges
from nihai.data.data_provider import DataProvider, Interval, Quote, resample_bars, rth_mask

_OHLCV = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def _aggregate_minutes(minutes: pd.DataFrame, interval: Interval) -> pd.DataFrame:
    """1m bar tamponunu hedef interval'a topla (OHLCV). Bar başlangıç-zamanı indeksli
    (yfinance/Alpaca konvansiyonu: label/closed='left').

    1d (Tur-90, BULGU-90-3): gün ET işlem gününe göre kurulur ve REST ile AYNI etiketi taşır
    (ET gece yarısı → naive UTC, 04:00/05:00). Eskiden `resample("1D")` UTC gece yarısı
    etiketliyordu; REST tohumu 04:00'te olduğu için birleşik seride her gün İKİ KEZ vardı."""
    if minutes.empty or interval == "1m":
        return minutes
    if interval == "1d":
        gun = minutes.index.tz_localize("UTC").tz_convert("America/New_York").normalize()
        agg = minutes.groupby(gun).agg(_OHLCV)
        agg.index = pd.DatetimeIndex(agg.index).tz_convert("UTC").tz_localize(None)
        return agg.dropna(subset=["open", "high", "low", "close"])
    # Tur-91 (BULGU-91-1): gün içi kova REST tohumu ve yfinance yedeğiyle AYNI (seans açılışına
    # çapalı) ızgarada; eskiden saate hizalıydı ve birleşik seride iki ızgara karışıyordu.
    return resample_bars(minutes, INTERVAL_MINUTES.get(interval, 60))


# alpaca-py DataStream._run_forever monkeypatch — handler henüz abone değilken (dinamik
# sembol modu) veya DNS/bağlantı hatasında alpaca-py iç döngüsü `await asyncio.sleep(0)`
# ile %100 CPU spin + log flood yapar. Yamalı sürüm üstel backoff ile yeniden bağlanır.
# (MainHub kopyasıyla ÖZDEŞ tutulur — entegrasyon ıraksamasını önlemek için.)
try:
    import asyncio
    import logging

    import alpaca.data.live.websocket

    async def _patched_run_forever(self) -> None:
        self._loop = asyncio.get_running_loop()
        while not any(
            v
            for k, v in self._handlers.items()
            if k not in ("cancelErrors", "corrections")
        ):
            if not self._stop_stream_queue.empty():
                self._stop_stream_queue.get(timeout=1)
                return
            await asyncio.sleep(0)

        log = logging.getLogger("alpaca")
        log.info(f"started {self._name} stream (monkeypatched)")
        self._should_run = True
        self._running = False
        backoff = 1.0

        while True:
            try:
                if not self._should_run:
                    log.info(f"{self._name} stream stopped")
                    return
                if not self._running:
                    log.info(f"starting {self._name} websocket connection")
                    await self._start_ws()
                    await self._send_subscribe_msg()
                    self._running = True
                    backoff = 1.0
                await self._consume()
            except Exception as e:
                await self.close()
                self._running = False

                import socket
                is_conn_error = isinstance(e, (socket.gaierror, ConnectionRefusedError, OSError))
                if is_conn_error:
                    log.warning(f"Connection/DNS error during {self._name} WS: {e}. Retrying in {backoff:.0f}s...")
                else:
                    log.warning(f"Error during websocket communication ({type(e).__name__}): {e}. Retrying in {backoff:.0f}s...")

                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)
            finally:
                await asyncio.sleep(0.1)

    alpaca.data.live.websocket.DataStream._run_forever = _patched_run_forever  # type: ignore[method-assign]
    logger.info("[alpaca_stream] DataStream._run_forever monkeypatched successfully")

    _orig_dispatch = alpaca.data.live.websocket.DataStream._dispatch

    async def _patched_dispatch(self, msg) -> None:
        # Tur-82 (BULGU-82-1): sunucunun abonelik ONAYI ve hata mesajları (ör. 405 symbol limit
        # exceeded) alpaca-py'de yalnız stdlib `logging`e düşer → hub.log'da HİÇ görünmedi ve
        # akış aylarca "bağlı ama boş" kaldı. Sağlayıcı kurduysa denetim mesajı ona da iletilir.
        cb = getattr(self, "_nihai_control", None)
        if cb is not None and msg.get("T") in ("subscription", "error"):
            cb(msg)
        await _orig_dispatch(self, msg)

    alpaca.data.live.websocket.DataStream._dispatch = _patched_dispatch  # type: ignore[method-assign]
except Exception as patch_err:
    logger.warning(f"[alpaca_stream] DataStream monkeypatch basarisiz: {patch_err}")


def _channel_plan(symbols: list[str], budget: int) -> tuple[list[str], list[str], list[str]]:
    """(bars, quotes, trades) — IEX Basic planının abonelik sınırına göre.

    ÖLÇÜLDÜ (Tur-82, 2026-09-18, canlı hesap): sınır trades+quotes TOPLAMINA uygulanır (30 sembol-
    kanal); bars ve updatedBars SAYILMAZ (150'de kabul edildi). 30 sembol × (bars+quotes+trades)
    = 60 kanal → sunucu isteği ATOMİK reddeder (405) ve HİÇBİR abonelik kurulmaz. Kotasyon önce
    gelir: `get_quote` akıştan yalnız bid/ask varken servis eder; trades yalnız `last`'ı rafine
    eder, o yüzden bütçeden artan olursa verilir."""
    quotes = symbols[:budget]
    return symbols, quotes, symbols[:max(0, budget - len(quotes))]


class AlpacaStreamProvider(DataProvider):
    name = "alpaca_stream"

    def __init__(self, symbols: list[str] | None = None, feed: str | None = None,
                 *, max_symbols: int | None = None, rest: AlpacaProvider | None = None):
        self._api_key = os.getenv("ALPACA_API_KEY", "")
        self._secret = os.getenv("ALPACA_SECRET_KEY", "")
        self._feed = (feed or os.getenv("ALPACA_FEED") or "iex").lower()
        self._max_symbols = max_symbols or int(os.getenv("STREAM_MAX_SYMBOLS", "30"))
        env_syms = [s.strip().upper() for s in os.getenv("STREAM_SYMBOLS", "").split(",") if s.strip()]
        self._wanted: list[str] = (symbols or env_syms)[: self._max_symbols]

        self._rest = rest or AlpacaProvider()       # seed + fallback (yfinance opsiyonu içinde)
        self._minutes: dict[str, deque] = {}        # symbol -> deque[(ts, o,h,l,c,v)]
        self._quotes: dict[str, Quote] = {}
        self._trades: dict[str, tuple[float, Any]] = {}  # symbol -> (son işlem fiyatı, ts)
        self._seed: dict[tuple[str, str], tuple[float, pd.DataFrame, int]] = {}  # (sym,interval)->(t, df, lookback)
        self._subscribed: set[str] = set()
        self._trade_syms: set[str] = set()          # trades kanalı da olanlar (bütçe sayımı)
        self._server_subs: dict[str, int] | None = None   # sunucunun ONAYLADIĞI (kanal → sayı)
        self._lock = threading.RLock()
        self._stream: Any = None  # StockDataStream (lazy import) | None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stream_ok = False
        self._connection_attempts = 0
        self._last_message_at: float | None = None
        self._last_stream_error: str | None = None
        self._message_counts = {"bars": 0, "quotes": 0, "trades": 0}
        self._tick_delays: dict[str, deque] = {k: deque(maxlen=512) for k in ("quotes", "trades")}
        self._tick_missing = {"quotes": 0, "trades": 0}

    # -----------------------------------------------------------------
    # Stream yaşam döngüsü
    # -----------------------------------------------------------------
    def start(self) -> None:
        """WS thread'ini başlat (idempotent). SDK/anahtar yoksa pasif kalır (REST'e düşer)."""
        if self._thread and self._thread.is_alive():
            return
        if not (self._api_key and self._secret):
            logger.warning("[alpaca_stream] ALPACA anahtarı yok → stream pasif, REST fallback")
            return
        # Alpaca alt-modüllerini ANA thread'de ön-yükle: WS thread'i (live) ile REST seed
        # (historical) eşzamanlı İLK import yaparsa Python import kilidi deadlock atar.
        # Burada sıralı ısıtınca thread artık cache'li modül bulur (race yok).
        try:
            import alpaca.data.enums  # noqa: F401
            import alpaca.data.historical  # noqa: F401
            import alpaca.data.live  # noqa: F401
        except Exception as e:
            logger.warning(f"[alpaca_stream] alpaca-py yok ({e}) → stream pasif, REST fallback")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="alpaca-stream", daemon=True)
        self._thread.start()
        logger.info(f"[alpaca_stream] thread başladı (feed={self._feed}, "
                    f"semboller={self._wanted or 'dinamik'})")

    def _run_loop(self) -> None:
        """WS bağlantısını çalıştır; koparsa backoff ile yeniden bağlan (reconnect)."""
        backoff = 2.0
        while not self._stop.is_set():
            try:
                from alpaca.data.enums import DataFeed
                from alpaca.data.live import StockDataStream
                try:
                    feed = DataFeed(self._feed)
                except ValueError:
                    feed = DataFeed.IEX
                self._stream = StockDataStream(self._api_key, self._secret, feed=feed)
                self._stream._nihai_control = self._on_control
                with self._lock:
                    self._server_subs = None     # yeni bağlantının onayı yok (eski onay yanıltır)
                self._connection_attempts += 1
                wanted = sorted(set(self._wanted) | self._subscribed)
                if wanted:
                    bars, quotes, trades = _channel_plan(wanted, self._max_symbols)
                    self._stream.subscribe_bars(self._on_bar, *bars)
                    if quotes:
                        self._stream.subscribe_quotes(self._on_quote, *quotes)
                    if trades:
                        self._stream.subscribe_trades(self._on_trade, *trades)
                    self._subscribed.update(wanted)
                    self._trade_syms = set(trades)
                self._stream_ok = True
                logger.info(f"[alpaca_stream] bağlandı, {len(self._subscribed)} sembol abone")
                self._stream.run()   # bloklar (kendi asyncio döngüsü)
            except Exception as e:
                self._stream_ok = False
                self._last_stream_error = type(e).__name__
                if self._stop.is_set():
                    break
                logger.warning(f"[alpaca_stream] bağlantı koptu ({e}); {backoff:.0f}s sonra yeniden")
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
            else:
                self._stream_ok = False
                backoff = 2.0

    async def _on_bar(self, bar) -> None:
        sym = str(getattr(bar, "symbol", "")).upper()
        ts = getattr(bar, "timestamp", None)
        if not sym or ts is None:
            return
        row = (pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts, tz="UTC"),
               float(bar.open), float(bar.high), float(bar.low), float(bar.close), float(bar.volume or 0))
        with self._lock:
            buf = self._minutes.setdefault(sym, deque(maxlen=2000))
            buf.append(row)
            self._message_counts["bars"] += 1
            self._last_message_at = time.time()

    async def _on_quote(self, q) -> None:
        received = time.time()
        sym = str(getattr(q, "symbol", "")).upper()
        if not sym:
            return
        bid = float(getattr(q, "bid_price", 0) or 0)
        ask = float(getattr(q, "ask_price", 0) or 0)
        last = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask)
        ts = getattr(q, "timestamp", None) or datetime.now(UTC)
        with self._lock:
            self._quotes[sym] = Quote(symbol=sym, bid=bid, ask=ask, last=last, timestamp=ts,
                                      spread_measured=(0 < bid <= ask < float("inf")
                                                       and getattr(q, "timestamp", None) is not None))
            self._message_counts["quotes"] += 1
            self._last_message_at = time.time()
            self._measure_tick("quotes", sym, getattr(q, "timestamp", None), received)

    async def _on_trade(self, t) -> None:
        """Gerçek son işlem fiyatı (Midas'ın gösterdiği gibi) — bid/ask orta noktası değil."""
        received = time.time()
        sym = str(getattr(t, "symbol", "")).upper()
        price = float(getattr(t, "price", 0) or 0)
        if not sym or price <= 0:
            return
        ts = getattr(t, "timestamp", None) or datetime.now(UTC)
        with self._lock:
            self._trades[sym] = (price, ts)
            self._message_counts["trades"] += 1
            self._last_message_at = time.time()
            self._measure_tick("trades", sym, getattr(t, "timestamp", None), received)

    def _on_control(self, msg: dict) -> None:
        """Sunucu denetim mesajı: abonelik onayı ya da hata (Tur-82; bkz. `_patched_dispatch`)."""
        if msg.get("T") == "subscription":
            subs = {k: len(msg.get(k) or []) for k in ("bars", "quotes", "trades")}
            with self._lock:
                self._server_subs = subs
            logger.info(f"[alpaca_stream] sunucu aboneliği onayladı: {subs}")
            return
        err = f"{msg.get('msg')} ({msg.get('code')})"
        with self._lock:
            self._last_stream_error = err
        logger.error(f"[alpaca_stream] sunucu hatası: {err} — istenen abonelik UYGULANMADI")

    def _measure_tick(self, kind: str, symbol: str, timestamp, received: float) -> None:
        # Çağıran kilidi tutar. Eksik/naive damgaya alım saatini atamak sahte 0 ms üretir.
        try:
            ts = pd.Timestamp(timestamp)
            if pd.isna(ts) or ts.tzinfo is None:
                raise ValueError("Saat dilimli olay damgası gerekli")
            delay = (received - ts.timestamp()) * 1000
            if not math.isfinite(delay):
                raise ValueError("Geçersiz damga")
        except (ValueError, TypeError, OverflowError):
            self._tick_missing[kind] += 1
            return
        self._tick_delays[kind].append({"symbol": symbol, "received_at": received,
                                       "event_at": ts.timestamp(), "delay_ms": round(delay, 3)})

    def _tick_status(self) -> dict:
        result = {}
        for kind, samples in self._tick_delays.items():
            values = sorted(s["delay_ms"] for s in samples)
            result[kind] = {"samples": len(values), "missing_timestamps": self._tick_missing[kind],
                "last": samples[-1] if samples else None,
                "p50_ms": values[math.ceil(len(values) * .5) - 1] if values else None,
                "p95_ms": values[math.ceil(len(values) * .95) - 1] if values else None,
                "max_ms": values[-1] if values else None,
                "negative_samples": sum(v < 0 for v in values)}
        return result

    def _ensure_subscribed(self, symbol: str) -> None:
        """STREAM_SYMBOLS boşsa, erişilen sembolleri dinamik abone et (cap'li)."""
        sym = symbol.upper()
        if sym in self._subscribed or not self._stream_ok or self._stream is None:
            return
        # Bütçe KANAL sayısıdır (trades+quotes, bkz. `_channel_plan`); dinamik ekleme trades açmaz.
        if len(self._subscribed) + len(self._trade_syms) >= self._max_symbols:
            return  # ücretsiz IEX limiti — sessizce REST'e bırak
        try:
            self._stream.subscribe_bars(self._on_bar, sym)
            self._stream.subscribe_quotes(self._on_quote, sym)
            self._subscribed.add(sym)
            logger.info(f"[alpaca_stream] +{sym} abone ({len(self._subscribed)}/{self._max_symbols})")
        except Exception as e:
            logger.debug(f"[alpaca_stream] {sym} abone hatası: {e}")

    def stop(self) -> None:
        self._stop.set()
        self._stream_ok = False
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass

    # -----------------------------------------------------------------
    # DataProvider arayüzü
    # -----------------------------------------------------------------
    def _minutes_df(self, symbol: str) -> pd.DataFrame:
        with self._lock:
            buf = list(self._minutes.get(symbol.upper(), ()))
        if not buf:
            return pd.DataFrame()
        df = pd.DataFrame(buf, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        return df[~df.index.duplicated(keep="last")].sort_index()

    def _get_seed(self, symbol: str, interval: Interval, lookback_bars: int) -> pd.DataFrame:
        """REST seed (geçmiş bar). interval süresi kadar TTL ile yeniden seed'lenir
        (1h için saatte ~1 REST çağrısı → rate-limit dostu)."""
        key = (symbol.upper(), interval)
        ttl = max(INTERVAL_MINUTES.get(interval, 60) * 60, 60)
        now = time.monotonic()
        with self._lock:
            hit = self._seed.get(key)
        # TTL içinde VE önceki seed istenen lookback'i kapsıyorsa cache'i kullan.
        # Daha büyük lookback istenirse (Madde 1: analiz pencere genişletme) yeniden seed'le.
        if hit and (now - hit[0]) < ttl and hit[2] >= lookback_bars:
            return hit[1]
        fresh = self._rest.get_bars(symbol, interval, lookback_bars + 5)
        if fresh is not None and not fresh.empty:
            with self._lock:
                self._seed[key] = (now, fresh, lookback_bars)
            return fresh
        return hit[1] if hit else pd.DataFrame()

    def get_bars(self, symbol: str, interval: Interval, lookback_bars: int = 200) -> pd.DataFrame:
        self._ensure_subscribed(symbol)
        seed = self._get_seed(symbol, interval, lookback_bars)
        dk = self._minutes_df(symbol)
        # Tur-90 (BULGU-90-2, Madde 22): tampon uzatılmış seans dakikalarını da tutar; REST tohumu
        # artık yalnız normal seansı taşıdığı için canlı toplam da aynı süzgeçten geçer.
        if not dk.empty:
            dk = dk[rth_mask(dk.index, 1)]
        live = _aggregate_minutes(dk, interval)
        if seed.empty and len(live) < lookback_bars:
            # Tur-106 (BULGU-106-5): REST tohumu düşünce (429) tampondan kurulan 1-8 günlük bar
            # EKSİKSİZ seri gibi dönüyordu; boş olmadığı için hub'ın da istemcinin de yedeği devreye
            # girmiyordu ve canlı quant (≥253 günlük bar) o sembolde günü `QUANT_YETERSIZ_GECMIS` ile
            # geçirdi (28 Eyl 6 sembol, 1 Eki 5 sembol). İsteği karşılayamayan kaynak cevap vermez.
            return pd.DataFrame()
        if interval != "1d" and not seed.empty and not live.empty:
            # Tur-82: seed'in gün-içi barları KAPANMIŞTIR (REST'te de Madde 3) ve REST'in tam
            # kaydıdır → canlı toplam yalnız seed'in son barından SONRAKİNİ ekler. Eskiden
            # çakışan barda canlı kazanıyordu: akış kopunca o saat EKSİK dakikalardan kurulup
            # tamponda (2000 dk) durduğu sürece — yeniden seed'lendikten sonra bile — REST'in
            # tam barını eziyordu. Sağlıklı akışta iki kaynak aynı IEX işlemlerini toplar (eşit).
            live = live[live.index > seed.index[-1]]
        elif interval == "1d" and not seed.empty and not live.empty:
            # Tur-90 (BULGU-90-3): kapanmış günler REST'in tam kaydıdır; tampon (2000 dk) onları
            # eksik tutabilir. Tohumun SON günü (REST çekim anındaki bugün, TTL 24 sa) canlıyla
            # BİRLEŞİR: açılış REST'ten, tepe/dip ikisinden, kapanış canlıdan, hacim büyüğünden.
            son = seed.index[-1]
            if son in live.index:
                s, c = seed.loc[son], live.loc[son]
                seed = seed.copy()
                seed.loc[son, ["high", "low", "close", "volume"]] = [
                    max(s["high"], c["high"]), min(s["low"], c["low"]), c["close"],
                    max(s["volume"], c["volume"])]
            live = live[live.index > son]
        if live.empty:
            return seed.tail(lookback_bars) if not seed.empty else seed
        # Birleştir: seed + canlı (gün-içinde çakışma yukarıda ayıklandı; 1d'de canlı = bugünün taze barı).
        merged = pd.concat([seed, live])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        # Madde 3 — oluşmakta olan son (yarı) barı dışla.
        if interval != "1d" and len(merged) > 1:
            bar_delta = pd.Timedelta(minutes=INTERVAL_MINUTES.get(interval, 60))
            last_ts = merged.index[-1]
            if (datetime.now(UTC).replace(tzinfo=None) - last_ts.to_pydatetime()) < bar_delta:
                merged = merged.iloc[:-1]
        return merged.tail(lookback_bars)

    def _ref_close(self, symbol: str) -> float | None:
        """B-4 kapısı için ÜCRETSİZ referans kapanış: önce canlı dakika tamponu,
        yoksa en son REST seed'i. Yeni veri çekmez (rate-limit dostu)."""
        sym = symbol.upper()
        live = self._minutes_df(sym)
        if not live.empty:
            try:
                close = float(live["close"].iloc[-1])
                if close > 0:
                    return close
            except (TypeError, ValueError, IndexError):
                pass
        with self._lock:
            seeds = [v for (s, _iv), v in self._seed.items() if s == sym]
        best = max(seeds, key=lambda v: v[0], default=None)   # en son seed'lenen
        if best is None or best[1] is None or best[1].empty:
            return None
        try:
            close = float(best[1]["close"].iloc[-1])
        except (TypeError, ValueError, IndexError):
            return None
        return close if close > 0 else None

    def get_quote(self, symbol: str) -> Quote | None:
        sym = symbol.upper()
        self._ensure_subscribed(symbol)
        with self._lock:
            q = self._quotes.get(sym)
            tr = self._trades.get(sym)
        if q is not None and q.bid > 0 and q.ask > 0:
            # last = GERÇEK son işlem fiyatı (Midas gibi); yoksa bid/ask orta noktası.
            if tr is not None and tr[0] > 0:
                cand = Quote(symbol=sym, bid=q.bid, ask=q.ask, last=tr[0], timestamp=tr[1],
                             spread_measured=q.spread_measured and self.is_quote_healthy(q)[0])
            else:
                cand = q
            # B-4: IEX defteri pasifken stream'in son bid/ask'i BAYAT kalabilir
            # (ölçüm: AAPL %-6.4). Elimizdeki bar kapanışına karşı çapraz doğrula.
            ref = self._ref_close(sym)
            if not quote_diverges(cand.last, ref):
                return cand
            logger.warning(
                f"[alpaca_stream] {sym} önbellek kotasyonu ELENDİ (B-4): "
                f"last={cand.last:.4f} vs bar kapanışı={ref:.4f} — REST'e düşülüyor"
            )
        return self._rest.get_quote(symbol)   # stream yok/bayat → REST (kendi kapısı var)

    def get_option_chain(self, symbol: str, expiry: str | None = None) -> dict:
        return self._rest.get_option_chain(symbol, expiry)   # AlpacaProvider zaten yf-fallback

    def stream_status(self) -> dict:
        """Pano/health için: bağlantı + abonelik + canlı bar/quote sayaçları."""
        with self._lock:
            return {
                "connected": self._stream_ok,
                "subscribed": sorted(self._subscribed),
                "symbols_with_minutes": sorted(self._minutes.keys()),
                "symbols_with_quotes": sorted(self._quotes.keys()),
                "symbols_with_trades": sorted(self._trades.keys()),
                "feed": self._feed,
                "max_symbols": self._max_symbols,
                "message_counts": dict(self._message_counts),
                "tick_latency": self._tick_status(),
                "latency_note": "Son 512 tick: olay damgası → yerel callback alımı. Saat kaymasını da içerir; negatif değerler korunur. Dakika barları dahil değildir.",
                "last_message_at": self._last_message_at,
                "connection_attempts": self._connection_attempts,
                "reconnects": max(0, self._connection_attempts - 1),
                "last_error": self._last_stream_error,
                "server_subscriptions": self._server_subs,   # None = sunucu henüz ONAYLAMADI
                "buffers": {s: {"used": len(b), "capacity": b.maxlen} for s, b in self._minutes.items()},
                "connection_note": "connected SDK döngüsünü, last_message_at gerçek alınan veriyi gösterir.",
            }
