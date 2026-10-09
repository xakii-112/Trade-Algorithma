"""Alpaca sağlayıcı testleri — normalize (SAF) + kimlik-yok davranışı (alpaca-py gerekmez)."""
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

import nihai.data.alpaca_provider as ap
from nihai.data.data_provider import Quote


def _alpaca_bars_df(symbol, timestamps):
    """Alpaca formatı: MultiIndex (symbol, timestamp), küçük-harf kolonlar + ekstra."""
    n = len(timestamps)
    idx = pd.MultiIndex.from_product([[symbol], timestamps], names=["symbol", "timestamp"])
    return pd.DataFrame({
        "open": np.linspace(100, 110, n), "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n), "close": np.linspace(100, 110, n),
        "volume": np.full(n, 1e6), "trade_count": np.full(n, 100), "vwap": np.linspace(100, 110, n),
    }, index=idx)


def test_normalize_multiindex_to_system_format():
    ts = pd.date_range("2026-06-01 09:30", periods=50, freq="h", tz="UTC")
    raw = _alpaca_bars_df("AAPL", ts)
    out = ap._normalize_alpaca_bars(raw, "AAPL", "1d", 200)   # 1d → oluşan bar dışlama yok
    # Tur-57: OHLCV sırası KORUNUR, `trade_count`/`vwap` artık DÜŞÜRÜLMEZ. Bu satır
    # önceden tam da düşürmeyi kilitliyordu — yani kusuru test koruyordu: ortalama işlem
    # büyüklüğü ve konsolide VWAP sapması "ölçülemez" sanılıyordu, oysa veri geliyordu.
    assert list(out.columns) == ["open", "high", "low", "close", "volume",
                                 "trade_count", "vwap"]
    assert out.index.tz is None                  # tz-naive
    assert len(out) == 50
    assert out["close"].iloc[-1] == 110.0


def test_normalize_excludes_forming_bar():
    # Son bar "şimdi" → 1h aralığında oluşmakta → dışlanmalı
    now = pd.Timestamp.now(tz="UTC").floor("h")
    ts = pd.date_range(end=now, periods=10, freq="h", tz="UTC")
    raw = _alpaca_bars_df("AAPL", ts)
    out = ap._normalize_alpaca_bars(raw, "AAPL", "1h", 200)
    assert len(out) == 9                          # son (oluşan) bar atıldı


def test_normalize_keeps_closed_last_bar():
    # Son bar 5 saat önce kapanmış → 1h aralığında tam → korunur
    old = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=5)
    ts = pd.date_range(end=old, periods=10, freq="h", tz="UTC")
    raw = _alpaca_bars_df("AAPL", ts)
    out = ap._normalize_alpaca_bars(raw, "AAPL", "1h", 200)
    assert len(out) == 10


def test_normalize_tail_limit():
    ts = pd.date_range("2026-01-01", periods=300, freq="D", tz="UTC")
    raw = _alpaca_bars_df("AAPL", ts)
    out = ap._normalize_alpaca_bars(raw, "AAPL", "1d", 50)
    assert len(out) == 50


def test_normalize_empty():
    assert ap._normalize_alpaca_bars(pd.DataFrame(), "AAPL", "1h", 200).empty


def test_start_for_lookback_covers_enough_history():
    # BUG: start verilmezse Alpaca yalnızca bugünü döner; start lookback'i kapsamalı
    now = datetime.now(UTC)
    # 1d: 200 bar → en az ~200 işlem günü geriye (takvim günü daha fazla)
    start_1d = ap._start_for_lookback("1d", 200)
    assert start_1d.tzinfo is not None
    assert (now - start_1d).days >= 200          # 200 bar için en az 200 gün
    # 1h: 200 bar → ~30 işlem günü; intraday günde ~7 bar
    start_1h = ap._start_for_lookback("1h", 200)
    assert 30 <= (now - start_1h).days <= 90      # makul pencere (ne çok dar ne çok geniş)
    # daha çok bar → daha geriye
    assert ap._start_for_lookback("1d", 400) < ap._start_for_lookback("1d", 100)


def test_quote_from_alpaca_normal():
    aq = SimpleNamespace(bid_price=100.0, ask_price=100.2, timestamp=datetime(2026, 6, 1))
    q = ap._quote_from_alpaca(aq, "AAPL")
    assert isinstance(q, Quote)
    assert q.bid == 100.0 and q.ask == 100.2
    assert abs(q.last - 100.1) < 1e-9
    assert q.spread_measured


def test_quote_from_alpaca_zero_side_synthetic():
    # bid/ask 0, sadece bir taraf → sentetik makas üretilir
    aq = SimpleNamespace(bid_price=0.0, ask_price=50.0, timestamp=None)
    q = ap._quote_from_alpaca(aq, "X")
    assert q.bid > 0 and q.ask > 0 and q.bid < q.ask
    assert abs(q.last - 50.0) < 1e-6
    assert not q.spread_measured


def test_quote_from_alpaca_none():
    assert ap._quote_from_alpaca(None, "X") is None


def test_provider_without_credentials_is_passive(monkeypatch):
    # Anahtar yok → client None → boş/None döner (FailoverProvider devralır).
    # Env'i de boşla: __init__ 'api_key or os.getenv()' ile gerçek .env anahtarına düşmesin.
    monkeypatch.setenv("ALPACA_API_KEY", "")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "")
    prov = ap.AlpacaProvider(api_key="", secret_key="")
    assert prov.get_bars("AAPL", "1h", 200).empty
    assert prov.get_quote("AAPL") is None


def test_get_provider_alpaca_failover_to_yfinance():
    # DATA_PROVIDER=alpaca → FailoverProvider(alpaca, yfinance) kurulmalı (ağ yok)
    from nihai.data.data_provider import CachingProvider, FailoverProvider, get_provider
    prov = get_provider("alpaca", with_cache=False)
    assert isinstance(prov, FailoverProvider)
    assert prov.primary.name == "alpaca"
    assert prov.secondary.name == "yfinance"
