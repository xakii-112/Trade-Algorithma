"""
test_tur45_atr_kirpmasiz.py — Tur-43'ün son açık kalemi: vol filtresinin ATR kırpması.

ÖLÇÜLEN KUSUR. `generate_signal` her barda `shared.atr14()` çağırıyordu ve bu
`_SharedSlice` içinde `.iloc[:n]` ile YENİ bir Series + yeni bir DatetimeIndex
kuruyordu — **bar başına O(n)**. Oysa o seriden okunan yalnız ÜÇ şeydi:
son değer · son 20 barın ortalaması · "hepsi NaN mi?". Profilde
`volatility_explosion_filter` %19, kırpma makinesi %13 idi.

ÇÖZÜM: `atr14_np()` ebeveynde BİR kez numpy dizisi kurar; alt görünüm `[:n]` ile
**sıfır-kopya numpy görünümü** alır. Nedensellik burada YAPISALDIR: dilim geleceği
İÇERMEZ ve geçilecek bir `n` parametresi yoktur (yanlış geçirilebilecek bir sayı
yoksa look-ahead de yoktur — Tur-43'ün `nbars` kapısının bir adım ötesi).

Bu dosya üç şeyi kilitler: (a) numpy görünümü kırpılmış Series ile BİT-DENK,
(b) görünüm geleceği göremez, (c) filtre verdikti iki arayüzde de aynı.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.signals import strategy as S


def _df(n: int = 300, *, seed: int = 45, kind: str = "normal") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if kind == "duz":
        c = np.full(n, 100.0)
    elif kind == "nanli":
        c = 100 + np.cumsum(rng.normal(0, 1, n))
        c[rng.choice(n, max(1, n // 10), replace=False)] = np.nan
    elif kind == "patlama":
        c = 100 + np.cumsum(rng.normal(0, 1, n))
        c[-1] = c[-2] + 80.0
    else:
        c = 100 + np.cumsum(rng.normal(0, 1, n))
    hi = c.copy() if kind == "duz" else c + np.abs(rng.normal(0, 0.5, n))
    lo = c.copy() if kind == "duz" else c - np.abs(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {"open": c, "high": hi, "low": lo, "close": c,
         "volume": rng.integers(1e4, 1e6, n).astype(float)},
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )


def _esit(a, b) -> bool:
    av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.isnan(av), np.isnan(bv)
    return bool(np.array_equal(na, nb) and (av[~na] == bv[~nb]).all())


# ---------------------------------------------------------------------------
# (a) numpy görünümü ↔ kırpılmış Series: BİT-DENK
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["normal", "duz", "nanli"])
def test_atr14_np_kirpilmis_seri_ile_bit_denk(kind):
    df = _df(300, kind=kind)
    tam = S._SharedIndicators(df)
    assert _esit(tam.atr14(), tam.atr14_np())
    for n in (30, 120, 299, 300):
        assert _esit(tam.upto(n).atr14(), tam.upto(n).atr14_np()), f"n={n}"


def test_atr14_np_sifir_kopya_gorunumdur():
    """Kırpma yerine görünüm: yeni bir tampon AYRILMAMALI (yoksa kazanç yok)."""
    tam = S._SharedIndicators(_df(200))
    taban = tam.atr14_np()
    dilim = tam.upto(50).atr14_np()
    assert dilim.base is taban or dilim.base is taban.base, "dilim kopya çıkardı"
    assert dilim.size == 50


# ---------------------------------------------------------------------------
# (b) NEDENSELLİK: görünüm geleceği göremez
# ---------------------------------------------------------------------------
def test_gorunum_gelecegi_icermez():
    """Asıl güvence: `[:n]` dilimi n'den sonrasını YAPISAL olarak taşımaz —
    yanlış geçirilebilecek bir indeks parametresi yoktur."""
    df = _df(400)
    tam = S._SharedIndicators(df)
    for n in (60, 200, 399):
        v = tam.upto(n).atr14_np()
        assert v.size == n
        # aynı dilim, SIFIRDAN hesaplanmış ATR ile de bit-denk (Tur-7 argümanı)
        assert _esit(v, S.atr(df.iloc[:n], 14))


# ---------------------------------------------------------------------------
# (c) Filtre: iki arayüz aynı verdikti verir
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["normal", "duz", "nanli", "patlama"])
def test_filtre_seri_ve_numpy_ayni_verdikt(kind):
    df = _df(200, kind=kind)
    a = S.atr(df, 14)
    assert S.volatility_explosion_filter(df, atr_series=a) == \
           S.volatility_explosion_filter(df, atr_series=a.to_numpy(dtype=float))


def test_filtre_kirpilmis_gorunumle_bar_bar_ayni():
    """Backtest yolu: her barda numpy görünümü ile kırpılmış Series aynı verdikt."""
    df = _df(300)
    tam = S._SharedIndicators(df)
    for n in range(60, 300, 7):
        view = tam.upto(n)
        assert S.volatility_explosion_filter(df.iloc[:n], atr_series=view.atr14()) == \
               S.volatility_explosion_filter(df.iloc[:n], atr_series=view.atr14_np()), f"n={n}"


@pytest.mark.parametrize("dizi", [
    np.full(50, np.nan),
    pd.Series([np.nan] * 50),
    np.array([], dtype=float),
    pd.Series([], dtype=float),
])
def test_uc_durumlar_ATR_NA(dizi):
    """Tam-NaN ve boş seri hâlâ ATR_NA vermeli (Tur-43 sözleşmesi korunur)."""
    assert S.volatility_explosion_filter(_df(80), atr_series=dizi) == (True, "ATR_NA")


def test_son_bar_nan_ama_seri_dolu_ATR_NA_DEGIL():
    """Kısa devre yanlış dallanmamalı: son bar NaN + öncesi geçerli → ATR_NA DEĞİL."""
    a = S.atr(_df(120), 14).to_numpy(dtype=float).copy()
    a[-1] = np.nan
    assert S.volatility_explosion_filter(_df(120), atr_series=a)[1] != "ATR_NA"


def test_patlama_dali_tetiklenir_ve_orani_yazar():
    """`ATR_EXPLOSION_<oran>x` dalı numpy gövdesinde de çalışmalı (ölü dal olmasın)."""
    a = np.full(40, 1.0)
    a[-1] = 100.0
    ok, sebep = S.volatility_explosion_filter(_df(60), atr_series=a)
    assert ok is False
    assert sebep.startswith("ATR_EXPLOSION_") and sebep.endswith("x")


def test_ortalama_son_20_bari_okur():
    """`avg` son 20 bardan gelir — 21. bar sonucu DEĞİŞTİRMEMELİ."""
    a = np.concatenate([np.full(50, 999.0), np.full(20, 1.0)])
    b = np.concatenate([np.full(50, 1.0), np.full(20, 1.0)])
    assert S.volatility_explosion_filter(_df(80), atr_series=a) == \
           S.volatility_explosion_filter(_df(80), atr_series=b)


def test_generate_signal_atr_metadatasi_son_deger():
    """`generate_signal` ATR'yi artık diziden okuyor — metadata son bar değeri olmalı."""
    df = _df(300, seed=7)
    tam = S._SharedIndicators(df)
    sig = S.generate_signal("AAA", df, shared=tam)
    if "atr" in sig.metadata:
        assert sig.metadata["atr"] == pytest.approx(float(tam.atr14_np()[-1]))
