"""AlpacaStreamProvider SAF-mantık testleri (ağ/WS gerekmez) — BULGU-1 birleştirme
regresyon koruması: lookback-duyarlı seed cache + gerçek-son-işlem fiyatı + stream_status.

Constructor `rest=` ile enjekte edilebilir; thread başlatılmaz (start() çağrılmaz)."""
import asyncio
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from nihai.data.alpaca_stream_provider import AlpacaStreamProvider, _aggregate_minutes
from nihai.data.data_provider import Quote


class FakeRest:
    """REST seed/fallback yerine — kaç kez get_bars çağrıldığını sayar."""
    name = "fake_rest"

    def __init__(self):
        self.bars_calls = 0
        self.quote_calls = 0

    def get_bars(self, symbol, interval, lookback_bars=200):
        self.bars_calls += 1
        idx = pd.date_range("2026-01-01", periods=lookback_bars, freq="h")
        return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                             "volume": 100.0}, index=idx)

    def get_quote(self, symbol):
        self.quote_calls += 1
        return Quote(symbol=symbol, bid=5.0, ask=5.2, last=5.1, timestamp=datetime.now())


def _provider(rest=None):
    return AlpacaStreamProvider(symbols=["AAPL"], rest=rest or FakeRest())


def test_stream_metrics_count_received_messages_and_buffer():
    p = _provider()
    now = datetime.now()
    asyncio.run(p._on_bar(SimpleNamespace(symbol="AAPL", timestamp=now, open=10, high=11, low=9, close=10, volume=100)))
    asyncio.run(p._on_quote(SimpleNamespace(symbol="AAPL", timestamp=now, bid_price=10, ask_price=11)))
    asyncio.run(p._on_trade(SimpleNamespace(symbol="AAPL", timestamp=now, price=10.5)))
    status = p.stream_status()
    assert status["message_counts"] == {"bars": 1, "quotes": 1, "trades": 1}
    assert status["last_message_at"] is not None and status["connected"] is False
    assert status["buffers"]["AAPL"] == {"used": 1, "capacity": 2000}


# --- _aggregate_minutes ---------------------------------------------------
def test_aggregate_minutes_rolls_up_to_hourly():
    idx = pd.date_range("2026-01-02 14:30", periods=120, freq="1min")  # seans açılışı (09:30 ET, UTC)
    minutes = pd.DataFrame({"open": range(120), "high": range(1, 121),
                            "low": range(120), "close": range(120),
                            "volume": [10] * 120}, index=idx).astype(float)
    out = _aggregate_minutes(minutes, "1h")
    assert not out.empty
    assert out.iloc[0]["volume"] == 600    # tam 60 dakika × 10
    assert out.iloc[0]["open"] == 0.0       # ilk dakikanın open'ı


def test_aggregate_minutes_passthrough_for_1m():
    idx = pd.date_range("2026-01-01", periods=3, freq="1min")
    minutes = pd.DataFrame({"open": [1, 2, 3]}, index=idx).astype(float)
    assert _aggregate_minutes(minutes, "1m").equals(minutes)


# --- lookback-duyarlı seed cache (BULGU-1) --------------------------------
def test_seed_cached_within_ttl_for_same_lookback():
    rest = FakeRest()
    p = _provider(rest)
    p._get_seed("AAPL", "1h", 100)
    p._get_seed("AAPL", "1h", 100)
    assert rest.bars_calls == 1   # TTL içinde, aynı lookback → cache


def test_seed_reseeds_when_larger_lookback_requested():
    """Daha geniş pencere istenince cache yetmez → yeniden seed (kısa-geçmiş hatası önlenir)."""
    rest = FakeRest()
    p = _provider(rest)
    p._get_seed("AAPL", "1h", 100)
    p._get_seed("AAPL", "1h", 300)   # daha büyük → yeniden seed
    assert rest.bars_calls == 2


def test_seed_smaller_lookback_uses_cache():
    rest = FakeRest()
    p = _provider(rest)
    p._get_seed("AAPL", "1h", 300)
    p._get_seed("AAPL", "1h", 100)   # daha küçük → cache yeterli
    assert rest.bars_calls == 1


# --- gerçek-son-işlem fiyatı (BULGU-1) ------------------------------------
def test_get_quote_prefers_real_trade_price():
    p = _provider()
    now = datetime.now()
    with p._lock:
        p._quotes["AAPL"] = Quote("AAPL", bid=10.0, ask=10.2, last=10.1, timestamp=now)
        p._trades["AAPL"] = (10.17, now)   # gerçek son işlem
    q = p.get_quote("AAPL")
    assert q.last == 10.17        # bid/ask orta noktası (10.1) DEĞİL, gerçek işlem
    assert q.bid == 10.0 and q.ask == 10.2


def test_get_quote_without_trade_returns_midpoint_quote():
    p = _provider()
    now = datetime.now()
    with p._lock:
        p._quotes["AAPL"] = Quote("AAPL", bid=10.0, ask=10.2, last=10.1, timestamp=now)
    q = p.get_quote("AAPL")
    assert q.last == 10.1         # işlem yok → mevcut quote


def test_get_quote_falls_back_to_rest_when_no_live():
    rest = FakeRest()
    p = _provider(rest)
    q = p.get_quote("AAPL")       # canlı quote yok → REST
    assert q.last == 5.1 and rest.quote_calls == 1


def test_on_trade_records_price():
    p = _provider()
    trade = SimpleNamespace(symbol="aapl", price=42.5, timestamp=datetime.now())
    asyncio.run(p._on_trade(trade))
    assert p._trades["AAPL"][0] == 42.5


def test_on_trade_ignores_nonpositive_price():
    p = _provider()
    asyncio.run(p._on_trade(SimpleNamespace(symbol="AAPL", price=0, timestamp=datetime.now())))
    assert "AAPL" not in p._trades


# --- stream_status --------------------------------------------------------
def test_stream_status_includes_trades_key():
    p = _provider()
    with p._lock:
        p._trades["AAPL"] = (1.0, datetime.now())
    st = p.stream_status()
    assert "symbols_with_trades" in st
    assert st["symbols_with_trades"] == ["AAPL"]
