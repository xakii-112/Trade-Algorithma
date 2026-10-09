"""Tur-10 perf/sağlamlık kalemleri (D2/D3/D4/D8/D9 + B7) — bit-exact parite guard'ları.

Desen (test_strategy_perf_parity): ESKİ (Tur-10 öncesi) implementasyon REFERANS olarak
test içinde tutulur ve YENİ (vektörize) üretim koduyla BİREBİR (==) karşılaştırılır —
"hızlandırırken davranış değişmedi" garantisi kalıcı teste döner.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nihai.signals.meta_features import FEATURE_NAMES
from nihai.web.web_dashboard import _bars_payload, _hist_payload, _series_payload
from research.agreement import _map_mom_pct
from research.build_pead_dataset import _enrich_row
from research.forward_harness import _auc as fh_auc
from research.purged_cv import permutation_importance
from research.train_meta import LogisticModel
from research.train_meta import _auc as tm_auc


# ---------------------------------------------------------------------------
# D2 — _auc vektörizasyonu (train_meta + forward_harness)
# ---------------------------------------------------------------------------
def _auc_ref(y: np.ndarray, p: np.ndarray):
    """Tur-10 öncesi saf-Python rank döngüsü (referans). Tek sınıf → None."""
    pos_n = int((y == 1).sum())
    neg_n = int((y == 0).sum())
    if pos_n == 0 or neg_n == 0:
        return None
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    avg_rank = np.zeros(len(counts))
    cum = 0
    for i, c in enumerate(counts):
        avg_rank[i] = cum + (c + 1) / 2.0
        cum += c
    ranks = avg_rank[inv]
    sum_pos = ranks[y == 1].sum()
    return (sum_pos - pos_n * (pos_n + 1) / 2.0) / (pos_n * neg_n)


def test_auc_parity_random_ve_tieli():
    rng = np.random.default_rng(42)
    for n in (2, 3, 7, 50, 500):
        for _ in range(10):
            y = (rng.random(n) < 0.5).astype(float)
            p_cont = rng.random(n)                    # sürekli (tie nadir)
            p_tie = rng.integers(0, 4, n) / 3.0       # ağır tie (rank ortalaması kritik)
            for p in (p_cont, p_tie):
                ref = _auc_ref(y, p)
                if ref is None:
                    assert tm_auc(y, p) == 0.5
                    assert fh_auc(y, p) is None
                else:
                    assert tm_auc(y, p) == ref        # bit-exact
                    assert fh_auc(y, p) == float(ref)


def test_auc_tek_sinif():
    p = np.linspace(0, 1, 5)
    assert tm_auc(np.ones(5), p) == 0.5
    assert fh_auc(np.ones(5), p) is None
    assert tm_auc(np.zeros(5), p) == 0.5
    assert fh_auc(np.zeros(5), p) is None


# ---------------------------------------------------------------------------
# D3 — permutation_importance yerinde permütasyon (purged_cv)
# ---------------------------------------------------------------------------
def _perm_ref(model, X: np.ndarray, y: np.ndarray, *, base_auc: float, seed: int = 0):
    """Tur-10 öncesi referans: her özellik için tam X.copy()."""
    rng = np.random.default_rng(seed)
    drops: dict[str, float] = {}
    for j, name in enumerate(FEATURE_NAMES):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        drops[name] = base_auc - tm_auc(y, model.predict_proba(Xp))
    return drops


def test_permutation_importance_parity_ve_restore():
    rng = np.random.default_rng(7)
    n, d = 80, len(FEATURE_NAMES)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(float)
    model = LogisticModel.fit(X, y, l2=1.0)
    base = tm_auc(y, model.predict_proba(X))
    X_before = X.copy()
    yeni = permutation_importance(model, X, y, base_auc=base, seed=3)
    assert np.array_equal(X, X_before)      # yerinde permütasyon GERİ yüklendi (çağıranın X'i bozulmaz)
    ref = _perm_ref(model, X, y, base_auc=base, seed=3)
    assert yeni == ref                      # bit-exact (rng tüketimi dahil aynı)


# ---------------------------------------------------------------------------
# D4 — prediction_tracker.report/coverage 'rows' paylaşımı (tek okuma)
# ---------------------------------------------------------------------------
def test_report_coverage_rows_param(tmp_path, monkeypatch):
    from nihai.execution import prediction_tracker as pt
    monkeypatch.setenv("META_GATE_THRESHOLD", "0.5")  # gate_sim eşiği deterministik olsun
    log = tmp_path / "predictions_log.csv"
    log.write_text(
        "logged_at,symbol,interval,hit,fwd_ret_pct,confidence,pwin,pwin_side,action\n"
        "2026-07-01T10:00:00,AAPL,1h,1,0.5,0.7,0.6,LONG,ENTER\n"
        "2026-07-01T11:00:00,MSFT,1h,0,-0.3,0.4,0.35,SHORT,WAIT\n"
        "2026-07-02T12:00:00,NVDA,1h,,,0.5,,,\n",
        encoding="utf-8")
    monkeypatch.setattr(pt, "PRED_LOG", log)
    rows = pt._read_log_rows()
    assert len(rows) == 3
    assert pt.report(rows) == pt.report()          # paylaşılan-okuma == kendi-okuması
    assert pt.coverage(rows) == pt.coverage()
    assert pt.coverage(rows)["resolved"] == 2
    # dosya yok → None yolu eski davranışla birebir
    monkeypatch.setattr(pt, "PRED_LOG", tmp_path / "yok.csv")
    assert pt._read_log_rows() == []
    assert pt.report() == {"resolved": 0, "msg": "henüz kayıt yok"}
    assert pt.coverage()["tracking_file"] is False


# ---------------------------------------------------------------------------
# D8 — agreement momentum asof eşlemesi (vektörize)
# ---------------------------------------------------------------------------
def test_map_mom_pct_parity():
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    rng = np.random.default_rng(5)
    pct = pd.DataFrame({"AAA": rng.random(10), "BBB": rng.random(10)}, index=dates)
    df = pd.DataFrame({
        "symbol": ["AAA", "BBB", "YOK", "AAA", "BBB"],
        "entry_dt": [
            dates[3] + pd.Timedelta(hours=5),   # gün içi → o günün değeri (asof)
            dates[0] - pd.Timedelta(days=1),    # ilk tarihten ÖNCE → NaN
            dates[4],                           # sembol panelde yok → NaN
            dates[-1] + pd.Timedelta(days=3),   # son tarihten sonra → SON değer (ffill)
            dates[2],                           # tam eşleşme (side=right → o gün dahil)
        ],
    })
    ref = []                                    # Tur-10 öncesi satır-satır referans döngü
    for _, row in df.iterrows():
        s = row["symbol"]
        if s not in pct.columns:
            ref.append(np.nan)
            continue
        col = pct[s]
        idx = col.index.searchsorted(row["entry_dt"], side="right") - 1
        ref.append(float(col.iloc[idx]) if idx >= 0 else np.nan)
    yeni = _map_mom_pct(df, pct)
    assert np.array_equal(np.asarray(ref, float), yeni, equal_nan=True)
    assert np.array_equal(_map_mom_pct(df.iloc[0:0], pct), np.array([]))  # boş df güvenli


# ---------------------------------------------------------------------------
# D9 — web payload vektörizasyonu (bars / series / hist)
# ---------------------------------------------------------------------------
def _bars_ref(df) -> list[dict]:
    out = []
    for idx, row in df.iterrows():
        ts = pd.Timestamp(idx)
        epoch = int(ts.tz_localize("UTC").timestamp() if ts.tzinfo is None else ts.timestamp())
        out.append({
            "time": epoch,
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": float(row.get("volume", 0) or 0),
        })
    return out


def _series_ref(s) -> list[dict]:
    out = []
    for idx, val in s.items():
        if val is None or (isinstance(val, float) and val != val):
            continue
        ts = pd.Timestamp(idx)
        epoch = int(ts.tz_localize("UTC").timestamp() if ts.tzinfo is None else ts.timestamp())
        out.append({"time": epoch, "value": round(float(val), 4)})
    return out


def _hist_ref(s, up: str = "#22c55e", dn: str = "#ef4444") -> list[dict]:
    out = []
    for idx, val in s.items():
        if val is None or (isinstance(val, float) and val != val):
            continue
        ts = pd.Timestamp(idx)
        epoch = int(ts.tz_localize("UTC").timestamp() if ts.tzinfo is None else ts.timestamp())
        fv = float(val)
        out.append({"time": epoch, "value": round(fv, 4), "color": up if fv >= 0 else dn})
    return out


def test_bars_payload_parity():
    naive = pd.date_range("2026-03-01 14:30", periods=6, freq="h")
    aware = pd.date_range("2026-03-01 14:30", periods=6, freq="h", tz="UTC")
    for idx in (naive, aware):
        df = pd.DataFrame({
            "open": [1.23456, 2.0, 3.0, 4.0, 5.55555, 6.0],
            "high": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            "volume": [100.0, 0.0, 3e6, 42.5, 7.0, 1.0],   # 0 → or-0 kuralı
        }, index=idx)
        assert _bars_ref(df) == _bars_payload(df)
        df2 = df.drop(columns=["volume"])                   # kolon yok → 0.0
        assert _bars_ref(df2) == _bars_payload(df2)
        assert _bars_payload(df.iloc[0:0]) == []
    # NaN hacim: iki yolda da NaN KORUNUR (or-0 kuralı NaN'ı 0 yapmaz — birebir)
    dfn = pd.DataFrame({"open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0],
                        "close": [1.0, 2.0], "volume": [5.0, np.nan]}, index=naive[:2])
    r, y = _bars_ref(dfn), _bars_payload(dfn)
    assert math.isnan(r[1]["volume"]) and math.isnan(y[1]["volume"])
    for a, b in zip(r, y, strict=True):
        assert {k: v for k, v in a.items() if k != "volume"} == \
               {k: v for k, v in b.items() if k != "volume"}
    assert r[0]["volume"] == y[0]["volume"] == 5.0


def test_series_ve_hist_payload_parity():
    naive = pd.date_range("2026-03-01", periods=5, freq="D")
    aware = naive.tz_localize("UTC")
    for idx in (naive, aware):
        s = pd.Series([1.11115, np.nan, -2.5, 0.0, 3.3], index=idx)
        assert _series_ref(s) == _series_payload(s)
        assert _hist_ref(s) == _hist_payload(s)             # 0.0 → up rengi (fv >= 0)
        s2 = s.astype(object)
        s2.iloc[3] = None                                   # None da atlanır (birebir)
        assert _series_ref(s2) == _series_payload(s2)
        assert _hist_ref(s2) == _hist_payload(s2)
    assert _series_payload(pd.Series([], dtype=float)) == []


# ---------------------------------------------------------------------------
# B7 — build_pead_dataset._enrich_row kaynak finite-guard
# ---------------------------------------------------------------------------
def test_enrich_row_finite_guard():
    bars = pd.DataFrame(
        {"close": np.linspace(10, 20, 80), "volume": np.full(80, 1e6)},
        index=pd.date_range("2026-01-01", periods=80, freq="D"))
    entry = "2026-03-10"
    for bad in (np.nan, np.inf, -np.inf):
        earn = pd.DataFrame({"surprise": [bad]}, index=[pd.Timestamp("2026-03-08")])
        assert _enrich_row({"entry_time": entry}, bars, earn) is False
    earn_ok = pd.DataFrame({"surprise": [3.2]}, index=[pd.Timestamp("2026-03-08")])
    row = {"entry_time": entry}
    assert _enrich_row(row, bars, earn_ok) is True
    assert row["surprise"] == 3.2
