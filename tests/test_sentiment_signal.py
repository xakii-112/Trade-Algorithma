"""sentiment_signal saf karar mantığı testleri.

Önceki bar bir sentiment zirvesi miydi? Düşük-hacim filtresi (gürültü reddi), momentum
vs contrarian yön, ve eksik-veri WAIT yolları davranış-kritik. Sentetik history DataFrame
ile ağsız test."""
import pandas as pd

from nihai.data.sentiment import sentiment_signal


def _hist(date, score, n_posts):
    return pd.DataFrame({"score": [score], "n_posts": [n_posts]}, index=[date])


def test_none_history_waits():
    assert sentiment_signal(None, "2026-01-01")[0] == "WAIT"


def test_empty_history_waits():
    assert sentiment_signal(pd.DataFrame(), "2026-01-01")[0] == "WAIT"


def test_missing_prev_date_waits():
    h = _hist("2026-01-01", 0.8, 10)
    action, reasons = sentiment_signal(h, "2025-12-31")
    assert action == "WAIT" and "NO_SENTIMENT_EVENT" in reasons


def test_low_volume_waits():
    h = _hist("2026-01-01", 0.9, 2)        # n_posts 2 < min_posts 5
    action, reasons = sentiment_signal(h, "2026-01-01", min_posts=5)
    assert action == "WAIT" and any("LOW_VOLUME" in r for r in reasons)


def test_bullish_momentum_long():
    h = _hist("2026-01-01", 0.8, 20)
    assert sentiment_signal(h, "2026-01-01", score_min=0.5)[0] == "LONG"


def test_bearish_momentum_short():
    h = _hist("2026-01-01", -0.8, 20)
    assert sentiment_signal(h, "2026-01-01", score_min=0.5)[0] == "SHORT"


def test_neutral_waits():
    h = _hist("2026-01-01", 0.1, 20)       # |score| < score_min
    action, reasons = sentiment_signal(h, "2026-01-01", score_min=0.5)
    assert action == "WAIT" and any("NEUTRAL" in r for r in reasons)


def test_contrarian_fades_bull_to_short():
    h = _hist("2026-01-01", 0.8, 20)
    assert sentiment_signal(h, "2026-01-01", allow_contrarian=True)[0] == "SHORT"


def test_contrarian_fades_bear_to_long():
    h = _hist("2026-01-01", -0.8, 20)
    assert sentiment_signal(h, "2026-01-01", allow_contrarian=True)[0] == "LONG"


def test_score_min_boundary_inclusive():
    # score == score_min → bullish (>=)
    h = _hist("2026-01-01", 0.5, 20)
    assert sentiment_signal(h, "2026-01-01", score_min=0.5)[0] == "LONG"
