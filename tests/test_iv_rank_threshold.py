"""iv_rank_threshold_test testleri — ÇEVRİMDIŞI.

En kritik test `test_rank_hesabi_uretimle_BIREBIR_AYNI`: ölçüm scripti üretimdeki
`get_iv_rank`'tan FARKLI bir rank hesaplarsa, ölçtüğü şey kodun kullandığı şey
olmaz ve verdikt sessizce anlamsızlaşır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.data.iv_history import ROLLING_WINDOW_DAYS, get_iv_rank
from research import iv_rank_threshold_test as rt


def _rolling_rank(iv: pd.Series, min_days: int = 1) -> pd.Series:
    """Script'in ürettiği rank formülünün tek-sembollük hâli."""
    lo = iv.rolling(ROLLING_WINDOW_DAYS, min_periods=min_days).min()
    hi = iv.rolling(ROLLING_WINDOW_DAYS, min_periods=min_days).max()
    span = hi - lo
    r = np.where(span > 1e-9, (iv - lo) / span * 100.0, np.nan)
    return pd.Series(r, index=iv.index).clip(0, 100)


def test_rank_hesabi_uretimle_BIREBIR_AYNI(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    iv = pd.Series(np.abs(rng.normal(0.30, 0.08, 400)).clip(0.02, 2.0))

    monkeypatch.setattr("nihai.data.iv_history.IV_HIST_DIR", tmp_path)
    monkeypatch.setattr("nihai.data.iv_history._path", lambda s: tmp_path / f"{s}.csv")

    benim = _rolling_rank(iv)
    for t in (300, 350, 399):                       # tam 252'lik pencere olan noktalar
        pencere = iv.iloc[: t + 1]
        pd.DataFrame({"date": [f"2020-01-{i%28+1:02d}" for i in range(len(pencere))],
                      "iv": pencere.to_numpy()}).to_csv(tmp_path / "X.csv", index=False)
        uretim = get_iv_rank("X", float(iv.iloc[t]))
        assert uretim.confidence == "OK"
        assert uretim.iv_rank == pytest.approx(float(benim.iloc[t]), abs=1e-9), (
            f"t={t}: üretim {uretim.iv_rank} ≠ ölçüm {benim.iloc[t]}"
        )


def test_welch_t_isareti_ve_yetersiz_ornek():
    yuksek = np.array([1.0, 1.1, 0.9, 1.2, 1.0])
    dusuk = np.array([0.0, 0.1, -0.1, 0.2, 0.0])
    assert rt._welch_t(yuksek, dusuk) > 2
    assert rt._welch_t(dusuk, yuksek) < -2
    assert np.isnan(rt._welch_t(np.array([1.0]), dusuk))       # n<2
    assert np.isnan(rt._welch_t(np.array([1.0, 1.0]), np.array([1.0, 1.0])))  # sıfır varyans


def _sahte_veri(n=400, seed=5):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "act_symbol": "X",
        "date": pd.bdate_range("2020-01-01", periods=n),
        "iv_current": np.abs(rng.normal(0.30, 0.07, n)).clip(0.02, 2.0),
        "hv_current": np.abs(rng.normal(0.30, 0.07, n)).clip(0.02, 2.0),
        "hv_month_ago": np.abs(rng.normal(0.30, 0.07, n)).clip(0.02, 2.0),
    })


def _calistir(monkeypatch, df, **kw):
    # Tur-64: toplu okuma `dolt_options.read_bulk`a taşındı (parquet yan dosyası) →
    # sahte veri artık o dikişten verilir; `veri_hazirla`'nın ölçtüğü matematik AYNI.
    monkeypatch.setattr(rt.dolt_options, "bulk_csv", lambda *a, **k: "sahte")
    monkeypatch.setattr(rt.dolt_options, "read_bulk",
                        lambda *a, cols=None, **k: df.copy()[list(cols)] if cols else df.copy())
    return rt.veri_hazirla(None, kw.pop("min_days", 30), kw.pop("exec_lag", 0), **kw)


def test_ileri_pencere_kenari_dogru_kaydiriyor(monkeypatch):
    """kenar(t) = hv_current(t+W) − iv_current(t). Kayma yanlışsa TÜM ölçüm bozulur."""
    df = _sahte_veri()
    W = 21
    out = _calistir(monkeypatch, df, hv_window=W)
    ilk = out.iloc[0]
    kaynak = df[df["date"] == ilk["date"]].index[0]
    beklenen = df["hv_current"].iloc[kaynak + W] - df["iv_current"].iloc[kaynak]
    assert ilk["kenar"] == pytest.approx(beklenen)
    # ileri verisi olmayan son W bar düşmüş olmalı
    assert out["date"].max() <= df["date"].iloc[-W - 1]


def test_non_overlap_striding_W_kadar(monkeypatch):
    """Örtüşme kaldırma sembol başına her W. gözlemi almalı (t şişmesini keser)."""
    df = _sahte_veri(n=500)
    W = 21
    tam = _calistir(monkeypatch, df, hv_window=W)
    seyrek = _calistir(monkeypatch, df, hv_window=W, non_overlap=True)
    assert len(seyrek) == len(range(0, len(tam), W))
    assert seyrek["date"].iloc[1] == tam["date"].iloc[W]


def test_exec_lag_rank_i_geriye_kaydirir(monkeypatch):
    """ACI DERS #7: kararın rank'i geçmişten okunmalı, aynı bardan değil."""
    # DİKKAT: monoton artan seri KULLANILAMAZ — her bar rolling-max olur, rank sabit
    # 100 çıkar ve kayma görünmez (dejenere girdi; Tur-28 dersi bunu tam olarak söylüyor).
    df = _sahte_veri(n=300, seed=11)
    lagsiz = _calistir(monkeypatch, df, hv_window=21, exec_lag=0)
    lagli = _calistir(monkeypatch, df, hv_window=21, exec_lag=1)

    ortak = lagsiz.merge(lagli, on="date", suffixes=("_0", "_1"))
    assert len(ortak) > 50
    # lag'li rank = lag'sizin BİR ÖNCEKİ barının rank'i olmalı
    assert not np.allclose(ortak["iv_rank_0"], ortak["iv_rank_1"]), "exec-lag etkisiz"
    beklenen = lagsiz.set_index("date")["iv_rank"].shift(1).reindex(ortak["date"])
    assert np.allclose(ortak["iv_rank_1"].to_numpy(), beklenen.to_numpy(), equal_nan=True)
