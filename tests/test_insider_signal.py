"""insider_signal saf karar mantığı testleri (SEC Form 4 küme-alım).

Önceki bar bir insider küme-ALIM günü müydü? Küme eşiği (min_buyers), alım>satım
baskınlığı, ve satış-sinyalinin varsayılan KAPALI (gürültülü) olması davranış-kritik.
Sentetik events DataFrame ile ağsız test."""
import pandas as pd

from nihai.data.insider import insider_signal


def _events(date, *, n_buy=0, n_sell=0, buy_shares=0.0, sell_shares=0.0):
    return pd.DataFrame({
        "n_buy_filings": [n_buy], "n_sell_filings": [n_sell],
        "buy_shares": [buy_shares], "sell_shares": [sell_shares],
    }, index=[date])


def test_none_events_waits():
    assert insider_signal(None, "2026-01-01")[0] == "WAIT"


def test_missing_prev_date_waits():
    ev = _events("2026-01-01", n_buy=2, buy_shares=1000)
    action, reasons = insider_signal(ev, "2025-12-31")
    assert action == "WAIT" and "NO_INSIDER_EVENT" in reasons


def test_buy_cluster_goes_long():
    ev = _events("2026-01-01", n_buy=2, buy_shares=5000, sell_shares=100)
    action, reasons = insider_signal(ev, "2026-01-01", min_buyers=1)
    assert action == "LONG" and any("BUY_cluster" in r for r in reasons)


def test_buy_below_cluster_threshold_waits():
    ev = _events("2026-01-01", n_buy=1, buy_shares=5000)
    assert insider_signal(ev, "2026-01-01", min_buyers=2)[0] == "WAIT"


def test_buy_not_dominant_waits():
    # n_buy yeterli ama satım hisseleri daha fazla → küme değil
    ev = _events("2026-01-01", n_buy=2, buy_shares=100, sell_shares=9000)
    assert insider_signal(ev, "2026-01-01", min_buyers=1)[0] == "WAIT"


def test_sell_cluster_ignored_by_default():
    ev = _events("2026-01-01", n_sell=3, sell_shares=9000, buy_shares=10)
    assert insider_signal(ev, "2026-01-01", min_buyers=1)[0] == "WAIT"


def test_sell_cluster_short_when_allowed():
    ev = _events("2026-01-01", n_sell=3, sell_shares=9000, buy_shares=10)
    action, reasons = insider_signal(ev, "2026-01-01", min_buyers=1, allow_short=True)
    assert action == "SHORT" and any("SELL_cluster" in r for r in reasons)
