"""BULGU B-4 — kotasyon akıl-sağlığı kapısı.

Gerçek olay (2026-07-25, piyasa kapalı): Alpaca IEX `latest_quote` AAPL için
bid 311.77 / ask 312.39 (mid 312.08) döndürdü; AYNI sağlayıcının barları ve
bağımsız yfinance ise 333.07–333.48 diyordu. Kotasyon fiyatı sinyalin giriş/stop/
hedef mesafesine girdiğinden bu sessizce yanlış işlem planı üretir.

Kapı: kotasyon, sağlayıcının ZATEN elindeki son bar kapanışına karşı doğrulanır;
saparsa None döner → FailoverProvider yfinance'e geçer. Ek REST çağrısı yapılmaz.
"""
from __future__ import annotations

import time

import pandas as pd
import pytest

from nihai.data.alpaca_provider import (
    QUOTE_SANITY_REF_TTL_SEC,
    AlpacaProvider,
    quote_diverges,
    quote_sanity_pct,
)

# --- Sahte Alpaca SDK nesneleri -------------------------------------------------


class _FakeQuote:
    def __init__(self, bid: float, ask: float):
        self.bid_price = bid
        self.ask_price = ask
        self.timestamp = None


class _FakeClient:
    """StockHistoricalDataClient yerine: sabit bar df + sabit latest-quote."""

    def __init__(self, bars: pd.DataFrame, quote: _FakeQuote):
        self._bars = bars
        self._quote = quote
        self.quote_calls = 0
        self.bar_calls = 0

    def get_stock_bars(self, req):           # noqa: ARG002
        self.bar_calls += 1
        return type("R", (), {"df": self._bars})()

    def get_stock_latest_quote(self, req):   # noqa: ARG002
        self.quote_calls += 1
        return self._quote


def _bars_df(symbol: str, close: float) -> pd.DataFrame:
    """_normalize_alpaca_bars'ın beklediği MultiIndex (symbol, timestamp) biçimi."""
    idx = pd.MultiIndex.from_tuples(
        [(symbol, pd.Timestamp("2026-07-24 19:00", tz="UTC")),
         (symbol, pd.Timestamp("2026-07-24 20:00", tz="UTC"))],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {"open": [close, close], "high": [close, close], "low": [close, close],
         "close": [close, close], "volume": [1000, 1000]},
        index=idx,
    )


def _provider(bar_close: float, bid: float, ask: float) -> tuple[AlpacaProvider, _FakeClient]:
    client = _FakeClient(_bars_df("AAPL", bar_close), _FakeQuote(bid, ask))
    return AlpacaProvider(api_key="k", secret_key="s", data_client=client), client


# --- Saf fonksiyon --------------------------------------------------------------


def test_saf_referans_yoksa_elenmez():
    """Çapa bilinmiyorsa kapı DAİMA açık (fail-open) — bilgi yokken eleme yapılmaz."""
    assert quote_diverges(312.08, None) is False
    assert quote_diverges(312.08, 0) is False


def test_saf_sapma_esigi():
    assert quote_diverges(312.08, 333.48, max_pct=3.0) is True      # %-6.4
    assert quote_diverges(333.02, 333.48, max_pct=3.0) is False     # %-0.14
    assert quote_diverges(320.0, 333.48, max_pct=3.0) is True       # %-4.0
    assert quote_diverges(330.0, 333.48, max_pct=3.0) is False      # %-1.04


def test_saf_kapi_kapatilabilir():
    """QUOTE_SANITY_PCT<=0 → kapı tamamen devre dışı (acil kaçış kapağı)."""
    assert quote_diverges(312.08, 333.48, max_pct=0) is False
    assert quote_diverges(312.08, 333.48, max_pct=-1) is False


def test_esik_env_ile_okunur(monkeypatch):
    monkeypatch.setenv("QUOTE_SANITY_PCT", "10")
    assert quote_sanity_pct() == 10.0
    assert quote_diverges(312.08, 333.48) is False   # %6.4 < %10 → geçer
    monkeypatch.setenv("QUOTE_SANITY_PCT", "saçma")
    assert quote_sanity_pct() == 3.0                 # bozuk değer → varsayılan


# --- Sağlayıcı davranışı --------------------------------------------------------


def test_gercek_olay_sapan_kotasyon_elenir():
    """2026-07-25 AAPL vakası: bar 333.48, quote mid 312.08 → None (failover)."""
    prov, client = _provider(bar_close=333.48, bid=311.77, ask=312.39)
    prov.get_bars("AAPL", "1h", 2)          # çapayı doldurur
    assert prov.get_quote("AAPL") is None
    assert client.quote_calls == 1           # REST yine çağrıldı, sonucu elendi


def test_saglikli_kotasyon_gecer():
    prov, _ = _provider(bar_close=333.48, bid=333.30, ask=333.50)
    prov.get_bars("AAPL", "1h", 2)
    q = prov.get_quote("AAPL")
    assert q is not None
    assert q.last == pytest.approx(333.40, abs=0.01)


def test_capa_yoksa_kotasyon_gecer():
    """get_bars hiç çağrılmadıysa referans yok → kapı uygulanmaz (fail-open)."""
    prov, _ = _provider(bar_close=333.48, bid=311.77, ask=312.39)
    q = prov.get_quote("AAPL")               # önce get_bars YOK
    assert q is not None
    assert q.last == pytest.approx(312.08, abs=0.01)


def test_bayat_capa_kapiyi_uygulamaz(monkeypatch):
    """TTL dolmuş çapa sağlam kotasyonu elememeli — bayat çapa > kapısızlık DEĞİL."""
    prov, _ = _provider(bar_close=333.48, bid=311.77, ask=312.39)
    prov.get_bars("AAPL", "1h", 2)
    ileri = time.monotonic() + QUOTE_SANITY_REF_TTL_SEC + 1
    monkeypatch.setattr(time, "monotonic", lambda: ileri)
    assert prov.get_quote("AAPL") is not None


def test_kapi_ek_rest_cagrisi_yapmaz():
    """Kapı ÜCRETSİZ olmalı: get_bars'ın zaten çektiği kapanışı kullanır (B-3 oran limiti)."""
    prov, client = _provider(bar_close=333.48, bid=311.77, ask=312.39)
    prov.get_bars("AAPL", "1h", 2)
    bar_cagri_once = client.bar_calls
    prov.get_quote("AAPL")
    assert client.bar_calls == bar_cagri_once   # kapı yeni bar çekmedi


def test_sayac_elenen_kotasyonu_da_sayar():
    """Elenen kotasyon da GERÇEK bir REST çağrısıydı — Madde 48 sayacı düşmemeli."""
    prov, _ = _provider(bar_close=333.48, bid=311.77, ask=312.39)
    prov.get_bars("AAPL", "1h", 2)
    once = AlpacaProvider.call_counts()["get_quote"]
    prov.get_quote("AAPL")
    assert AlpacaProvider.call_counts()["get_quote"] == once + 1


# --- Stream sağlayıcısı ---------------------------------------------------------


def test_stream_bayat_onbellek_kotasyonu_reste_duser(monkeypatch):
    """AlpacaStreamProvider'ın bellek-içi quote/trade önbelleği de kapıdan geçer."""
    from nihai.data.alpaca_stream_provider import AlpacaStreamProvider
    from nihai.data.data_provider import Quote

    monkeypatch.setattr(AlpacaStreamProvider, "_start_stream", lambda self: None, raising=False)
    sp = AlpacaStreamProvider.__new__(AlpacaStreamProvider)
    sp._lock = __import__("threading").RLock()
    sp._quotes = {"AAPL": Quote(symbol="AAPL", bid=311.77, ask=312.39, last=312.08,
                                timestamp=pd.Timestamp("2026-07-24 20:00", tz="UTC"))}
    sp._trades = {}
    sp._minutes = {}
    sp._seed = {("AAPL", "1h"): (time.monotonic(), _flat_df(333.48), 200)}
    sp._subscribed = {"AAPL"}
    sp._stream_ok = False
    sp._stream = None
    sp._max_symbols = 30

    saglam = Quote(symbol="AAPL", bid=333.30, ask=333.50, last=333.40,
                   timestamp=pd.Timestamp("2026-07-24 20:00", tz="UTC"))
    sp._rest = type("R", (), {"get_quote": staticmethod(lambda s: saglam)})()

    q = sp.get_quote("AAPL")
    assert q is not None
    assert q.last == pytest.approx(333.40)   # bayat 312.08 DEĞİL


def _flat_df(close: float) -> pd.DataFrame:
    idx = pd.to_datetime(["2026-07-24 19:00", "2026-07-24 20:00"])
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 100},
        index=idx,
    )
