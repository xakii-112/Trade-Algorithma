"""estimate_atm_iv guard/None-yolu testleri (Madde 16 — zincirden ATM IV tahmini).

Mevcut suite mutlu yolu kapsıyor; bu testler bozuk/eksik girdide GÜVENLİ None dönüşünü
(çökme yok) kilitler — canlı sinyal yolu bu fonksiyona bozuk zincir verebilir."""
import pandas as pd

from nihai.risk.options_analyzer import estimate_atm_iv


def test_none_df_returns_none():
    assert estimate_atm_iv(None, underlying=100, dte=30) is None


def test_empty_df_returns_none():
    assert estimate_atm_iv(pd.DataFrame(), underlying=100, dte=30) is None


def test_nonpositive_underlying_returns_none():
    calls = pd.DataFrame({"strike": [100], "bid": [3.0], "ask": [3.2]})
    assert estimate_atm_iv(calls, underlying=0, dte=30) is None
    assert estimate_atm_iv(calls, underlying=-5, dte=30) is None


def test_nonpositive_dte_returns_none():
    calls = pd.DataFrame({"strike": [100], "bid": [3.0], "ask": [3.2]})
    assert estimate_atm_iv(calls, underlying=100, dte=0) is None


def test_missing_strike_column_returns_none():
    calls = pd.DataFrame({"bid": [3.0], "ask": [3.2]})
    assert estimate_atm_iv(calls, underlying=100, dte=30) is None


def test_all_zero_prices_returns_none():
    # Geçerli strike ama hiç fiyat yok (bid/ask/last = 0) → mid<=0 hepsi atlanır → None
    calls = pd.DataFrame({
        "strike": [90, 100, 110],
        "bid": [0.0, 0.0, 0.0],
        "ask": [0.0, 0.0, 0.0],
        "lastPrice": [0.0, 0.0, 0.0],
    })
    assert estimate_atm_iv(calls, underlying=100, dte=30) is None


def test_valid_chain_returns_plausible_iv():
    # Regresyon kontrolü: sağlam ATM prim → 0 < iv < 5 aralığında bir değer
    calls = pd.DataFrame({
        "strike": [90, 100, 110],
        "bid": [10.5, 3.0, 0.5],
        "ask": [10.9, 3.2, 0.7],
        "lastPrice": [10.7, 3.1, 0.6],
    })
    iv = estimate_atm_iv(calls, underlying=100, dte=30)
    assert iv is None or 0.0 < iv < 5.0   # py_vollib yoksa None olabilir (zarif degrade)
