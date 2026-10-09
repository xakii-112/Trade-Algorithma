"""test_build_live_dataset.py — canlı log → eğitim CSV join'i (AI eğitim hattı)."""
from __future__ import annotations

import csv

import pytest

from nihai.execution.prediction_tracker import FEAT_FIELDS, FIELDS
from nihai.signals.meta_features import FEATURE_NAMES
from research.build_live_dataset import EXTRA_COLS, build


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _pred(symbol, bar_ts, *, direction="+", fwd="2.0", side="LONG",
          resolved="2026-07-02T10:00:00", pwin="0.6"):
    r = dict.fromkeys(FIELDS, "")
    r.update({"logged_at": "t", "symbol": symbol, "interval": "1h", "bar_ts": bar_ts,
              "direction": direction, "action": "LONG", "confidence": "0.6",
              "pwin": pwin, "pwin_side": side, "price": "100", "horizon_bars": "5",
              "resolved_at": resolved, "fwd_price": "102",
              "hit": "1" if float(fwd) > 0 else "0", "fwd_ret_pct": fwd})
    return r


def _feat(symbol, bar_ts, adx=25.0):
    r = {n: "0.5" for n in FEAT_FIELDS}
    r.update({"logged_at": "t", "symbol": symbol, "interval": "1h",
              "bar_ts": bar_ts, "adx14": str(adx)})
    return r


def test_build_joins_and_labels(tmp_path):
    """Çözülmüş+özellikli satırlar join'lenir; etiket TARAF-işaretli getiriye göre."""
    pred_p = tmp_path / "predictions_log.csv"
    feat_p = tmp_path / "live_features_log.csv"
    _write_csv(pred_p, FIELDS, [
        # LONG taraf, yön '+', ham +2% → kazanç (label 1)
        _pred("AAPL", "b1", direction="+", fwd="2.0", side="LONG"),
        # SHORT taraf (kapı-engelli aday), yön '+', ham +2% → SHORT kaybeder (label 0)
        _pred("MSFT", "b2", direction="+", fwd="2.0", side="SHORT"),
        # Çözülmemiş → hariç
        _pred("TSLA", "b3", resolved=""),
        # pwin_side yok (eski şema) → hariç
        _pred("NVDA", "b4", side=""),
        # Özellik satırı yok → hariç
        _pred("AMD", "b5"),
    ])
    _write_csv(feat_p, FEAT_FIELDS, [
        _feat("AAPL", "b1", adx=33.0), _feat("MSFT", "b2"),
        _feat("TSLA", "b3"), _feat("NVDA", "b4"),
    ])
    df = build(pred_path=pred_p, feats_path=feat_p)
    assert list(df.columns) == [*FEATURE_NAMES, *EXTRA_COLS]
    assert len(df) == 2
    a = df[df["symbol"] == "AAPL"].iloc[0]
    m = df[df["symbol"] == "MSFT"].iloc[0]
    assert a["label"] == 1 and a["pnl"] == pytest.approx(2.0)
    assert a["adx14"] == pytest.approx(33.0)          # özellikler join'den geliyor
    assert m["label"] == 0 and m["pnl"] == pytest.approx(-2.0)  # taraf-işaretli
    assert (df["period"] == "live").all()


def test_build_empty_when_logs_missing(tmp_path):
    """Log dosyaları yoksa boş ama DOĞRU kolonlu DataFrame (çökme yok)."""
    df = build(pred_path=tmp_path / "yok1.csv", feats_path=tmp_path / "yok2.csv")
    assert df.empty
    assert list(df.columns) == [*FEATURE_NAMES, *EXTRA_COLS]


def test_same_symbol_time_uses_matching_interval(tmp_path):
    pred, feats = tmp_path / "p.csv", tmp_path / "f.csv"
    _write_csv(pred, FIELDS, [_pred("AAPL", "b1")])
    _write_csv(feats, FEAT_FIELDS, [_feat("AAPL", "b1", 33),
                                   {**_feat("AAPL", "b1", 99), "interval": "1m"}])
    df = build(pred_path=pred, feats_path=feats)
    assert len(df) == 1 and df.iloc[0].adx14 == 33
    assert df.attrs["cohorts"] == [("1h", "5")]


def test_numeric_bar_ts_joins_not_silently_dropped(tmp_path):
    """B9: bar_ts sayısal görünüp feats'te bir boş hücre kolonu float'a çevirince,
    dtype=str olmadan '1000' (preds) vs '1000.0' (feats) uyuşmazlığı satırı SESSİZCE
    düşürürdü. Anahtar kolonları str okumak iki tarafı hizalar → satır join'lenir."""
    pred_p = tmp_path / "predictions_log.csv"
    feat_p = tmp_path / "live_features_log.csv"
    _write_csv(pred_p, FIELDS, [_pred("AAA", "1000")])
    # 2. feats satırının boş bar_ts'i, dtype çıkarımıyla kolonu float yapar (1000 → 1000.0).
    _write_csv(feat_p, FEAT_FIELDS, [_feat("AAA", "1000"), _feat("BBB", "")])
    df = build(pred_path=pred_p, feats_path=feat_p)
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "AAA"
    assert df.iloc[0]["label"] == 1        # LONG, +2% → kazanç


def test_build_nan_features_become_neutral_not_nan(tmp_path):
    """BULGU-11 regresyonu: başlık-yükseltmeli eski satırın '' hücresi NaN olur;
    `x or 0.0` NaN'ı GEÇİRİRDİ (NaN truthy) → dataset'e NaN sızar, model sessizce
    bozulurdu. Artık nötr varsayılana düşer (rsi14→50, rvol→1, diğerleri→0 vb.)."""
    import math

    from nihai.signals.meta_features import neutral_default

    pred_p = tmp_path / "predictions_log.csv"
    feat_p = tmp_path / "live_features_log.csv"
    _write_csv(pred_p, FIELDS, [_pred("AAPL", "b1")])
    r = _feat("AAPL", "b1")
    r["rsi14"] = ""      # eski-şema boş hücre → pandas NaN
    r["rvol"] = ""
    r["ret_5"] = ""
    _write_csv(feat_p, FEAT_FIELDS, [r])
    df = build(pred_path=pred_p, feats_path=feat_p)
    assert len(df) == 1
    row = df.iloc[0]
    assert not any(isinstance(row[n], float) and math.isnan(row[n]) for n in FEATURE_NAMES)
    assert row["rsi14"] == neutral_default("rsi14") == 50.0
    assert row["rvol"] == neutral_default("rvol") == 1.0
    assert row["ret_5"] == 0.0
