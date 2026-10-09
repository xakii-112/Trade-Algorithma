"""Sağlayıcı failover testleri — TV çift-fetch bug'ı + FailoverProvider tekilliği."""
import pandas as pd

import nihai.data.data_provider as dp


class _FakeTv:
    """tvDatafeed yerine: get_hist çağrılınca exception fırlat."""
    def get_hist(self, **kwargs):
        raise OSError("[Errno 11001] getaddrinfo failed")


class _SpyYf:
    """get_bars çağrılırsa sayacı arttır (çağrılmamalı)."""
    name = "yfinance"
    def __init__(self):
        self.bars_calls = 0
    def get_bars(self, symbol, interval, lookback_bars=200):
        self.bars_calls += 1
        return pd.DataFrame()


def _make_tv_with_fakes():
    from nihai.data.tv_provider import TradingViewProvider
    # __init__ ağ kurar (TvDatafeed) → atla: örneği elle kur
    tv = object.__new__(TradingViewProvider)
    tv.name = "tradingview"
    tv._tv = _FakeTv()
    tv._yf_fallback = _SpyYf()
    return tv


def test_tv_get_bars_no_internal_fallback_on_error():
    # TV exception verince İÇ yfinance fallback ÇAĞRILMAMALI (çift-fetch bug'ı düzeltildi)
    tv = _make_tv_with_fakes()
    out = tv.get_bars("AAPL", "1h", 200)
    assert out.empty
    assert tv._yf_fallback.bars_calls == 0   # iç fallback tetiklenmedi → tek otorite FailoverProvider


def test_tv_error_logged_as_warning_not_error():
    # loguru kullanılıyor → caplog değil, kendi sink'imizle yakala
    from loguru import logger
    records = []
    sink = logger.add(lambda m: records.append(m.record), level="DEBUG")
    try:
        tv = _make_tv_with_fakes()
        tv.get_bars("AAPL", "1h", 200)
    finally:
        logger.remove(sink)
    tv_levels = [r["level"].name for r in records if "[tv]" in r["message"]]
    assert "ERROR" not in tv_levels      # kurtarılabilir → ERROR değil
    assert "WARNING" in tv_levels


def test_failover_uses_secondary_once_on_primary_empty():
    # Primary boş → secondary tam BİR kez kullanılır (çift değil)
    class _EmptyPrimary:
        name = "primary"
        def get_bars(self, *a, **k):
            return pd.DataFrame()
    spy = _SpyYf()
    fo = dp.FailoverProvider(_EmptyPrimary(), spy)
    out = fo.get_bars("AAPL", "1h", 200)
    assert out.empty
    assert spy.bars_calls == 1   # secondary tam bir kez


def test_source_stats_counts_primary_and_active():
    # Kaynak gözlemlenebilirliği: primary servis edince sayaç + active güncellenir
    class _P:
        name = "alpaca"
        def get_bars(self, *a, **k):
            return pd.DataFrame({"close": [1.0]})
    class _S:
        name = "yfinance"
        def get_bars(self, *a, **k):
            return pd.DataFrame({"close": [2.0]})
    fo = dp.FailoverProvider(_P(), _S())
    fo.get_bars("X", "1h", 5)
    fo.get_bars("X", "1h", 5)
    st = fo.source_stats()
    assert st["active"] == "alpaca"
    assert st["primary_hits"] == 2 and st["secondary_hits"] == 0


def test_source_stats_counts_secondary_on_primary_empty():
    class _EmptyP:
        name = "alpaca"
        def get_bars(self, *a, **k):
            return pd.DataFrame()
    class _S:
        name = "yfinance"
        def get_bars(self, *a, **k):
            return pd.DataFrame({"close": [2.0]})
    fo = dp.FailoverProvider(_EmptyP(), _S())
    fo.get_bars("X", "1h", 5)
    st = fo.source_stats()
    assert st["active"] == "yfinance"
    assert st["primary_hits"] == 0 and st["secondary_hits"] == 1
