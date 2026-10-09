"""data_health.py testleri — veri sağlığı kapısı (Tur-16 WP-O)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.ai import data_health, quant_math


def _clean_ohlcv(n=100, seed=1):
    idx = pd.bdate_range("2022-01-03", periods=n)
    paths = quant_math.mc_gbm_paths(100.0, 0.05, 0.15, n - 1, 1, seed)
    close = paths[0]
    rng = np.random.default_rng(seed)
    open_ = close * (1.0 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.002, n)))
    volume = rng.integers(100_000, 500_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                        "volume": volume}, index=idx)


def test_temiz_veri_yuksek_skor_karantina_yok():
    df = _clean_ohlcv()
    r = data_health.health_check(df, symbol="AAA")
    assert r["score"] >= 80.0
    assert r["quarantine"] is False
    assert r["symbol"] == "AAA"
    assert r["n_bars"] == 100


def test_bos_dataframe():
    r = data_health.health_check(pd.DataFrame())
    assert r["score"] == 0.0 and r["quarantine"] is True
    assert r["issues"][0]["type"] == "veri_yok"


def test_dataframe_olmayan_girdi():
    r = data_health.health_check([1, 2, 3])
    assert r["score"] == 0.0 and r["quarantine"] is True


def test_close_kolonu_yok():
    df = pd.DataFrame({"open": [1.0, 2.0, 3.0, 4.0, 5.0]})
    r = data_health.health_check(df)
    assert r["score"] == 0.0 and r["quarantine"] is True
    assert r["issues"][0]["type"] == "close_yok"


def test_yetersiz_bar_sayisi_karantina():
    df = _clean_ohlcv(n=3, seed=2)
    r = data_health.health_check(df)
    assert r["quarantine"] is True
    assert any(i["type"] == "yetersiz_veri" for i in r["issues"])


def test_nan_degerler_skoru_dusurur():
    df = _clean_ohlcv(n=60, seed=3)
    df.loc[df.index[10:15], "close"] = np.nan
    r = data_health.health_check(df)
    assert any(i["type"] == "nan_or_type" for i in r["issues"])
    assert r["score"] < 100.0


def test_negatif_fiyat_ohlc_invalid():
    df = _clean_ohlcv(n=60, seed=4)
    df.loc[df.index[20], "close"] = -5.0
    r = data_health.health_check(df)
    assert any(i["type"] == "ohlc_invalid" for i in r["issues"])


def test_high_low_ihlali_ohlc_invalid():
    df = _clean_ohlcv(n=60, seed=5)
    df.loc[df.index[30], "high"] = 1.0
    df.loc[df.index[30], "low"] = 500.0
    r = data_health.health_check(df)
    assert any(i["type"] == "ohlc_invalid" for i in r["issues"])


def test_eksik_bar_buyuk_bosluk():
    df = _clean_ohlcv(n=100, seed=6)
    df2 = pd.concat([df.iloc[:20], df.iloc[60:]])   # ortadan 40 bar kopar (büyük tarih boşluğu)
    r = data_health.health_check(df2)
    assert any(i["type"] == "missing_bars" for i in r["issues"])


def test_donmus_fiyat_esik_ustu_yakalanir():
    df = _clean_ohlcv(n=60, seed=7)
    df.iloc[10:16, df.columns.get_loc("close")] = df["close"].iloc[10]   # 6 ardışık aynı (eşik=5)
    r = data_health.health_check(df)
    assert any(i["type"] == "frozen_price" for i in r["issues"])


def test_donmus_fiyat_esik_alti_yakalanmaz():
    df = _clean_ohlcv(n=60, seed=8)
    df.iloc[10:13, df.columns.get_loc("close")] = df["close"].iloc[10]   # 3 ardışık (eşik=5 altı)
    r = data_health.health_check(df)
    assert not any(i["type"] == "frozen_price" for i in r["issues"])


def test_aciklanamayan_sicrama_jump_outlier():
    df = _clean_ohlcv(n=80, seed=9)
    close = df["close"].to_numpy().copy()
    close[50] *= 1.37   # split-oranlarından hiçbirine yakın değil (V-şok)
    df["close"] = close
    r = data_health.health_check(df)
    assert any(i["type"] == "jump_outlier" for i in r["issues"])


def test_split_adayi_jump_degil_cezasiz():
    df_jump = _clean_ohlcv(n=80, seed=10)
    close_j = df_jump["close"].to_numpy().copy()
    close_j[50] *= 1.37
    df_jump["close"] = close_j
    r_jump = data_health.health_check(df_jump)

    df_split = _clean_ohlcv(n=80, seed=10)
    close_s = df_split["close"].to_numpy().copy()
    close_s[50:] *= 0.5   # kalıcı yarıya-iniş = klasik 2:1 split
    df_split["close"] = close_s
    r_split = data_health.health_check(df_split)

    assert any(i["type"] == "split_candidate" for i in r_split["issues"])
    assert not any(i["type"] == "jump_outlier" for i in r_split["issues"])
    # split cezasız, jump cezalı -> split skoru jump skorundan (aynı senaryoda) daha yüksek olmalı
    assert r_split["score"] > r_jump["score"]


def test_hacim_anomalisi_yakalanir():
    df = _clean_ohlcv(n=60, seed=11)
    vol = df["volume"].to_numpy().copy()
    vol[40] = vol.mean() + 50.0 * vol.std()
    df["volume"] = vol
    r = data_health.health_check(df)
    assert any(i["type"] == "volume_anomaly" for i in r["issues"])


def test_hacim_kolonu_yoksa_atlanir_cokmez():
    df = _clean_ohlcv(n=60, seed=12).drop(columns=["volume"])
    r = data_health.health_check(df)
    assert not any(i["type"] == "volume_anomaly" for i in r["issues"])
    assert r["score"] >= 0.0


def test_agir_bozuk_veri_karantina():
    n = 40
    df = _clean_ohlcv(n=n, seed=42)
    close = df["close"].to_numpy().copy()
    close[0:5] = np.nan               # NaN bloğu
    close[5:10] = -1.0                # negatif bloğu
    close[10:18] = close[10]          # donmuş bloğu (8 bar >= eşik)
    df["close"] = close
    # birkaç DAĞINIK tarih-boşluğu (her biri ayrı gap-olayı — missing_bars'ı da tetikler;
    # tek-yığın kopmadan farklı olarak birden çok küçük boşluk oran-tabanlı cezayı büyütür)
    df = df.drop(df.index[[20, 24, 28, 32, 36]])
    r = data_health.health_check(df)
    assert r["score"] < 50.0, f"beklenenden yüksek skor: {r}"
    assert r["quarantine"] is True


def test_skor_daima_0_100_araliginda():
    df = _clean_ohlcv(n=40, seed=13)
    close = df["close"].to_numpy().copy()
    close[::3] = np.nan
    df["close"] = close
    r = data_health.health_check(df)
    assert 0.0 <= r["score"] <= 100.0


def test_determinizm_ayni_girdi_ayni_sonuc():
    df = _clean_ohlcv(n=60, seed=14)
    r1 = data_health.health_check(df, symbol="X")
    r2 = data_health.health_check(df, symbol="X")
    assert r1 == r2


def test_kucuk_harf_buyuk_harf_kolon_esleme():
    df = _clean_ohlcv(n=40, seed=15)
    df.columns = [c.upper() for c in df.columns]
    r = data_health.health_check(df)
    assert r["n_bars"] == 40
    assert r["score"] > 0.0


# ---------------------------------------------------------------------------
# Alt-seviye yardımcılar
# ---------------------------------------------------------------------------
def test_missing_bar_count_integer_index_gap():
    idx = np.concatenate([np.arange(0, 11), np.arange(50, 61)])   # 11..49 arası büyük atlama
    n_gaps, detail = data_health._missing_bar_count(idx)
    assert n_gaps >= 1
    assert "bar-aralığı" in detail


def test_missing_bar_count_duzgun_index_gap_yok():
    idx = pd.RangeIndex(0, 30)
    n_gaps, _ = data_health._missing_bar_count(idx)
    assert n_gaps == 0


def test_frozen_price_runs_tumu_ayni():
    close = np.full(10, 42.0)
    runs = data_health._frozen_price_runs(close)
    assert runs == [10]


def test_frozen_price_runs_kisa_veri():
    assert data_health._frozen_price_runs(np.array([1.0, 2.0])) == []


def test_lazy_jump_detect_modul_yokken_none(monkeypatch):
    # advanced_math artık MEVCUT (Dalga-2) — yokluk senaryosu sys.modules blokajıyla
    # simüle edilir (None girdisi import'u ImportError'a düşürür → lazy getter None döner).
    import sys

    import nihai.ai
    monkeypatch.delattr(nihai.ai, "advanced_math", raising=False)
    monkeypatch.setitem(sys.modules, "nihai.ai.advanced_math", None)
    assert data_health._lazy_jump_detect() is None


def test_lazy_jump_detect_modul_varken_callable():
    fn = data_health._lazy_jump_detect()
    assert fn is not None and callable(fn)
