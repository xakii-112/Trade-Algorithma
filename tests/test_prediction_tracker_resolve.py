"""test_prediction_tracker_resolve.py — verdict() hüküm eşikleri + C1 tz-dayanıklı resolve.

verdict(): report() özetinden makine-okunur DÜRÜST hüküm (|isabet-0.5|<0.03 → kenar yok).
C1: tz-aware KAYIT vs tz-naive ÇÖZÜM aynı barı eşlemeli (ham str eşleşmesi her satırı
kalıcı 'pending' bırakıyordu → forward-kanıt döngüsü sessizce ölüyordu, ACI DERS #5)."""
from __future__ import annotations

import pandas as pd
import pytest

import nihai.execution.prediction_tracker as pt


@pytest.fixture()
def tmp_log(tmp_path, monkeypatch):
    """PRED_LOG/FEATS_LOG'u geçici dosyaya yönlendir (üretim data/ dokunulmaz)."""
    monkeypatch.setattr(pt, "PRED_LOG", tmp_path / "predictions_log.csv")
    monkeypatch.setattr(pt, "FEATS_LOG", tmp_path / "live_features_log.csv")
    return pt.PRED_LOG


class _FakeProvider:
    def __init__(self, df):
        self._df = df

    def get_bars(self, symbol, interval, lookback_bars=400):
        return self._df


def _ohlc(prices):
    return {"open": prices, "high": prices, "low": prices, "close": prices}


def _row(symbol, bar_ts, direction="+", price=100.0):
    return {"symbol": symbol, "action": "WAIT", "direction": direction,
            "bar_ts": bar_ts, "confidence": 0.6, "prob_cal": 0.52, "price": price}


# --------------------------------------------------------------------- verdict()
def test_verdict_no_data():
    v = pt.verdict({"resolved": 0})
    assert v["label"] == "NO_DATA"
    assert v["hit_rate"] is None


def test_verdict_no_edge_near_coinflip():
    v = pt.verdict({"resolved": 200, "hit_rate": 0.51})
    assert v["label"] == "NO_EDGE"          # edge 0.01 < 0.03
    assert v["edge_pp"] == 1.0


def test_verdict_weak_positive():
    v = pt.verdict({"resolved": 200, "hit_rate": 0.60})
    assert v["label"] == "WEAK_POSITIVE"
    assert v["edge_pp"] == 10.0
    assert v["hit_rate"] == 0.60


def test_verdict_inverted():
    v = pt.verdict({"resolved": 200, "hit_rate": 0.40})
    assert v["label"] == "INVERTED"         # isabet <%50 → ters yön
    assert v["edge_pp"] == 10.0


def test_verdict_edge_boundary_is_strict():
    # edge tam 0.03 → NO_EDGE DEĞİL (eşik katı '<'): 0.53 → WEAK_POSITIVE, 0.47 → INVERTED,
    # 0.52 (edge 0.02) → NO_EDGE.
    assert pt.verdict({"resolved": 100, "hit_rate": 0.53})["label"] == "WEAK_POSITIVE"
    assert pt.verdict({"resolved": 100, "hit_rate": 0.47})["label"] == "INVERTED"
    assert pt.verdict({"resolved": 100, "hit_rate": 0.52})["label"] == "NO_EDGE"


# --------------------------------------------------- C1: tz-aware kayıt / tz-naive çözüm
def test_canon_ts_normalizes_and_is_idempotent():
    # tz-aware ET 10:00-04:00 == 14:00 UTC; naive 14:00 aynı kanonik anahtara inmeli.
    assert pt._canon_ts("2026-06-19 10:00:00-04:00") == pt._canon_ts("2026-06-19 14:00:00")
    # Aynı naive gösterimde idempotent (hâlihazırda eşleşen kayıtları bozmaz).
    assert pt._canon_ts("2026-06-19 14:00:00") == pt._canon_ts("2026-06-19 14:00:00")
    # Ayrıştırılamayan → ham str (geriye-uyumlu).
    assert pt._canon_ts("not-a-timestamp") == "not-a-timestamp"


def test_resolve_matches_tzaware_record_against_naive_df(tmp_log):
    # Kayıt tz-AWARE (canlı hub); çözüm df'i tz-NAIVE (stripped) → aynı barı eşleyip çözmeli.
    aware_ts = "2026-06-19 10:00:00-04:00"       # = 2026-06-19 14:00 UTC
    assert pt.record([_row("AAPL", aware_ts, direction="+", price=100.0)],
                     "1h", horizon=3) == 1
    idx = pd.date_range("2026-06-19 14:00:00", periods=6, freq="h")   # naive UTC
    df = pd.DataFrame(_ohlc([100, 101, 102, 103, 104, 105]), index=idx)
    res = pt.resolve(provider=_FakeProvider(df))
    assert res["resolved"] == 1 and res["pending"] == 0
    rep = pt.report()
    assert rep["hit_rate"] == 1.0                # 103 > 100, yön + → isabet
    assert rep["avg_signed_ret_pct"] == pytest.approx(3.0, abs=1e-6)


def test_resolve_both_tzaware_still_matches(tmp_log):
    # İki taraf da tz-aware (farklı ofset temsili): eşleşme kanonik-UTC üzerinden korunur.
    idx = pd.date_range("2026-06-19 14:00:00", periods=5, freq="h", tz="UTC")
    bts = "2026-06-19 10:00:00-04:00"            # 14:00 UTC = idx[0]
    assert pt.record([_row("MSFT", bts, direction="+", price=50.0)], "1h", horizon=2) == 1
    df = pd.DataFrame(_ohlc([50, 51, 52, 53, 54]), index=idx)
    res = pt.resolve(provider=_FakeProvider(df))
    assert res["resolved"] == 1 and res["pending"] == 0
