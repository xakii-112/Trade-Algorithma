"""chart_analyst saf-fonksiyon testleri — ağ YOK, LLM çağrısı YOK.

parse_read (savunmacı ayrıştırma), numeric_context (bağlam metni), format_telegram
(caption limiti + zorunlu feragat satırı) doğrulanır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.notifier.chart_analyst import (
    ChartRead,
    format_telegram,
    numeric_context,
    parse_read,
)


def _bars(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    idx = pd.date_range("2026-01-05 10:00", periods=n, freq="h")
    return pd.DataFrame({
        "open": close - 0.1, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": rng.integers(1_000, 50_000, n).astype(float),
    }, index=idx)


# ---------------------------------------------------------------- parse_read
def test_parse_read_tam_veri():
    r = parse_read({
        "trend": "yukari", "momentum": "alici",
        "destek": [101.234, 99.9, 95.0],          # 3 verildi → 2'ye kırpılır
        "direnc": [110.5],
        "hacim_yorumu": "x" * 200,                 # 140'a kırpılır
        "formasyon": "yükselen üçgen",
        "risk_notu": "EMA50 altı kapanış okumayı bozar",
        "ozet": "Kısa özet.",
    })
    assert r is not None
    assert r.trend == "yukari" and r.momentum == "alici"
    assert r.destek == [101.23, 99.9] and r.direnc == [110.5]
    assert len(r.hacim_yorumu) == 140
    assert r.formasyon == "yükselen üçgen"


def test_parse_read_savunmaci():
    # Bilinmeyen enum → güvenli varsayılan; bozuk sayılar filtrelenir; None → None
    assert parse_read(None) is None
    assert parse_read("metin") is None
    r = parse_read({"trend": "moon", "momentum": "yolo",
                    "destek": ["abc", 55.0, None], "direnc": "hepsi"})
    assert r is not None
    assert r.trend == "belirsiz" and r.momentum == "notr"
    assert r.destek == [55.0] and r.direnc == []
    assert r.formasyon == "yok"


def test_parse_read_bos_dict():
    r = parse_read({})
    assert r is not None
    assert r.trend == "belirsiz" and r.destek == [] and r.ozet == ""


# ---------------------------------------------------------- numeric_context
def test_numeric_context_icerik():
    df = _bars()
    ctx = numeric_context(df, "AAPL", "1h")
    last = float(df["close"].iloc[-1])
    assert "AAPL" in ctx and "1h" in ctx and f"{last:.2f}" in ctx
    assert "EMA20" in ctx and "RSI14" in ctx
    # Seviye çapası: modelin uydurma aralığı yerine gerçek pencere sınırları verilmeli
    assert f"{float(df['low'].min()):.2f}" in ctx


def test_numeric_context_hacimsiz():
    df = _bars().drop(columns=["volume"])
    ctx = numeric_context(df, "MSFT", "1d")
    assert "MSFT" in ctx            # hacim yokken de çalışır (KeyError yok)


# ---------------------------------------------------------- format_telegram
def test_format_telegram_feragat_ve_limit():
    r = ChartRead(trend="yukari", momentum="guclu_alici", destek=[100.0, 98.5],
                  direnc=[105.0], hacim_yorumu="hacim artıyor",
                  formasyon="bayrak", risk_notu="98.5 altı bozar", ozet="ö" * 400)
    text = format_telegram(r, "NVDA", "1h")
    assert len(text) <= 1000                         # Telegram caption güvenli sınırı
    assert "NVDA" in text and "Yukarı" in text
    assert "sinyal/tavsiye değildir" in text         # feragat zorunlu
    assert "`100.00`" in text and "`105.00`" in text


def test_format_telegram_bos_okuma():
    text = format_telegram(ChartRead(), "AAPL", "1h")
    assert "Belirsiz" in text and "sinyal/tavsiye değildir" in text
    assert "Destek" not in text and "Direnç" not in text   # boş liste → satır yok


# formasyon 'yok' ise satır basılmaz
@pytest.mark.parametrize("formasyon,gorunur", [("yok", False), ("Yok", False), ("çift dip", True)])
def test_format_telegram_formasyon(formasyon, gorunur):
    text = format_telegram(ChartRead(formasyon=formasyon), "T", "1h")
    assert ("Formasyon" in text) is gorunur
