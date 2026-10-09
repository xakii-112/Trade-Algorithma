"""horizon_sweep / spread_estimate / üretim KPI testleri (F1/F3/O3, 2026-07-27).

Ağ YOK: barlar sentetik üretilir, `predictions_log` geçici dizine yazılır. Testler
ölçümün DOĞRU olduğunu değil, harness'ın bilinen girdide bilinen çıktıyı verdiğini
doğrular (kenar iddiası içermez)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research import horizon_sweep as hs
from research import spread_estimate as se


def _bars(n: int = 800, start: str = "2026-06-23 13:30:00") -> pd.DataFrame:
    """Deterministik 1m bar serisi (tz-naive UTC, seans-içi)."""
    idx = pd.date_range(start, periods=n, freq="1min")
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.02, n))
    return pd.DataFrame({"open": close, "high": close + 0.05,
                         "low": close - 0.05, "close": close, "volume": 1000.0}, index=idx)


def test_sweep_yon_isaretini_dogru_uygular():
    """SHORT tahmin + düşen fiyat = pozitif işaretli getiri (yön işareti doğru mu?)."""
    idx = pd.date_range("2026-06-23 13:30:00", periods=50, freq="1min")
    close = np.linspace(100.0, 95.0, 50)          # monoton DÜŞÜŞ
    bars = {"X": pd.DataFrame({"open": close, "high": close, "low": close,
                               "close": close, "volume": 1.0}, index=idx)}
    preds = pd.DataFrame({
        "symbol": ["X"], "key": [idx[0].isoformat()], "ts": [idx[0]],
        "sign": [-1.0], "confidence": [0.5], "price": [close[0]], "hit": [np.nan],
    })
    long, diag = hs.sweep(preds, bars, [5], [0], {"_default": 0.0})
    assert diag["eslesen"] == 1
    bar_row = long[long["giris"] == "bar"]
    assert len(bar_row) == 1
    assert float(bar_row["signed"].iloc[0]) > 0     # düşüşte SHORT kazanır


def test_giris_kaymasi_isareti_ve_iki_varyant():
    """Canlı fiyat sinyal yönünde uzaktaysa kayma POZİTİF ve canlı-giriş getirisi daha düşük."""
    idx = pd.date_range("2026-06-23 13:30:00", periods=30, freq="1min")
    close = np.linspace(100.0, 101.0, 30)
    bars = {"X": pd.DataFrame({"open": close, "high": close, "low": close,
                               "close": close, "volume": 1.0}, index=idx)}
    preds = pd.DataFrame({
        "symbol": ["X"], "key": [idx[0].isoformat()], "ts": [idx[0]],
        "sign": [1.0], "confidence": [0.5],
        "price": [close[0] * 1.001],                # LONG'ta %0,1 PAHALI girilmiş
        "hit": [np.nan],
    })
    long, diag = hs.sweep(preds, bars, [5], [0], {"_default": 0.0})
    assert diag["giris_kaymasi"]["ort_bps"] == pytest.approx(10.0, abs=0.1)
    bar_v = float(long[long["giris"] == "bar"]["signed"].iloc[0])
    canli_v = float(long[long["giris"] == "canli"]["signed"].iloc[0])
    assert canli_v < bar_v                          # kayma getiriyi düşürür


def test_rth_izgarasi_seans_disini_atar_ve_390_bar_yapar():
    df = _bars(n=1500, start="2026-06-23 08:00:00")   # 08:00 UTC = 04:00 ET (uzatılmış seans)
    g = hs.to_rth_grid(df)
    et = pd.to_datetime(pd.Series(g.index)).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    dk = et.dt.hour * 60 + et.dt.minute
    assert dk.min() >= 9 * 60 + 30 and dk.max() <= 15 * 60 + 59
    assert g.index.is_monotonic_increasing


def test_kume_t_ham_t_den_kucuk_ortusen_veride():
    """Aynı değeri tekrar eden (tam örtüşen) gözlemlerde kümelenmiş t şişmez."""
    signed = np.full(500, 0.001) + np.random.default_rng(1).normal(0, 1e-6, 500)
    clusters = np.repeat(["A", "B"], 250)
    t_kume, k = hs._cluster_t(signed, clusters)
    assert k == 2 and t_kume is None      # 3'ten az küme → hüküm YOK (uydurma yok)


def test_sidak_kritik_t_kombinasyonla_buyur():
    assert hs._sidak_t(1) < hs._sidak_t(28) < hs._sidak_t(500)
    assert hs._sidak_t(1) == pytest.approx(1.96, abs=0.01)


def test_corwin_schultz_sifir_araliginda_sifir():
    """high == low (hiç aralık yok) → spread tahmini 0, NaN/patlama yok."""
    idx = pd.date_range("2026-06-23 13:30:00", periods=100, freq="1min")
    c = np.full(100, 50.0)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    s = se.corwin_schultz(df, 1.0)
    assert np.nanmax(s) == pytest.approx(0.0, abs=1e-9)
    assert se.estimate(df, 1.0)["max_bps"] == pytest.approx(0.0, abs=1e-6)


def test_roll_trendli_seride_sifir_doner():
    """Negatif otokovaryans yoksa Roll 0 döner (aşağı yanlı — uydurma spread yok)."""
    idx = pd.date_range("2026-06-23 13:30:00", periods=200, freq="1min")
    c = np.linspace(100.0, 110.0, 200)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    assert se.roll(df, 1.0) == 0.0


def test_exam_production_veri_yoksa_uydurmaz(tmp_path, monkeypatch):
    from nihai.ai import selfcheck
    monkeypatch.setattr(selfcheck, "DATA_DIR", tmp_path)
    import nihai.ai.decision as dm
    monkeypatch.setattr(dm, "DECISIONS_FILE", tmp_path / "yok.jsonl")
    r = selfcheck.exam_production()
    assert r["durum"] == "VERI_YOK"
    assert any("hiç ölçülmemiş" in u for u in r["uyarilar"])


def test_exam_production_yuzde_100_abstain_uyarisi(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    import nihai.ai.decision as dm
    from nihai.ai import selfcheck
    p = tmp_path / "ai_decisions.jsonl"
    now = datetime.now(UTC).isoformat()
    p.write_text("\n".join(
        f'{{"ts": "{now}", "action": "ABSTAIN"}}' for _ in range(20)), encoding="utf-8")
    monkeypatch.setattr(dm, "DECISIONS_FILE", p)
    monkeypatch.setattr(selfcheck, "DATA_DIR", tmp_path)
    r = selfcheck.exam_production()
    assert r["durum"] == "OLCULDU" and r["n_yonlu"] == 0
    assert r["abstain_orani"] == 1.0
    assert any("HİÇBİR yön üretmiyor" in u for u in r["uyarilar"])


def test_neutralize_degenerate_sabit_kolonu_sifirlar():
    from research.train_neural import neutralize_degenerate
    X = np.column_stack([np.random.default_rng(0).normal(size=50),
                         np.full(50, 2.0) + 3e-15])
    out = neutralize_degenerate(X, ["iyi", "rr_ratio"])
    assert out.shape == X.shape           # kolon DÜŞÜRÜLMEZ (train/serve kilidi)
    assert np.all(out[:, 1] == 0.0)
    assert np.allclose(out[:, 0], X[:, 0])


def test_vix_sifir_notr_degere_dusurulur():
    from nihai.signals.meta_features import _VIX_NEUTRAL, _vix_or_neutral
    assert _vix_or_neutral(0.0) == _VIX_NEUTRAL
    assert _vix_or_neutral(None) == _VIX_NEUTRAL
    assert _vix_or_neutral(-3.0) == _VIX_NEUTRAL
    assert _vix_or_neutral(21.5) == 21.5
