"""Tur-96 kilidi — AI'lar ↔ algoritmalar AYRIMI ve çoklu piyasa koşusunun ölçüm sözleşmeleri.

Tur iki şey kurdu ve ikisi de sessizce bozulabilir:

  * **AYRIM** (`research.coklu_piyasa`): her strateji kipi ya ALGORİTMA ya AI; nihai/ai + nihai/signals
    altındaki her modül ai | algoritma | araç. Yeni bir modül/kip sınıfsız kalırsa ayrım bayatlar —
    kilit dizinle EŞİTLİĞİ sorar (Tur-76 BULGU-76-5: elle tutulan liste dizinden kopar).
  * **ÖLÇÜM** (`research.piyasalar` + `coklu_piyasa`): veri temizliği getirileri değiştirmemeli, ölçekleme
    getirileri korumalı, çevrimiçi öğrenenler etiketi BİLİNMEDEN güncellenmemeli (karar i kapanışında,
    etiket y_i open[i+2]'de → güncelleme yalnız j ≤ i−2), yürüyen eğitim Aralık sonu kararlarının Ocak'ta
    doğan etiketini eğitime sokmamalı, oynaklık öngörüsü geleceği görmemeli.
"""
from __future__ import annotations

import importlib
import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import coklu_piyasa as K
from research import piyasalar as P

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ AYRIM
def _nesne(yol: str):
    parca = yol.split(".")
    for k in range(len(parca), 0, -1):
        try:
            obj = importlib.import_module(".".join(parca[:k]))
        except ModuleNotFoundError:
            continue
        for ad in parca[k:]:
            obj = getattr(obj, ad)
        return obj
    raise ModuleNotFoundError(yol)


def test_her_modul_siniflandirildi_ve_dizinle_esit():
    dizin = {f"nihai.{p.parent.name}.{p.stem}"
             for klasor in ("ai", "signals") for p in (ROOT / "nihai" / klasor).glob("*.py")
             if p.stem != "__init__"}
    assert set(K.MODUL_SINIFI) == dizin, (set(K.MODUL_SINIFI) ^ dizin)
    assert {s for s, _ in K.MODUL_SINIFI.values()} <= {"ai", "algoritma", "arac"}
    assert all(g.strip() for _, g in K.MODUL_SINIFI.values())


def test_her_strateji_kipi_tam_bir_kez_siniflandirildi():
    from nihai.signals.strategy import STRATEGY_MODES
    assert not set(K.ALGORITMALAR) & set(K.AILER)
    # llm_event bir DİL MODELİ sinyalidir → AI tarafında (dil_modeli) durur, algoritma değildir
    assert set(STRATEGY_MODES) - set(K.ALGORITMALAR) == {"llm_event"}
    assert "llm_event" in K.AILER["dil_modeli"].aciklama
    assert set(K.MODLAR_ALGO) <= {k for k, v in K.ALGORITMALAR.items() if v.kosulur}


@pytest.mark.parametrize("y", list(K.ALGORITMALAR.values()) + list(K.AILER.values()), ids=lambda y: y.ad)
def test_katalog_kaynagi_gercek_ve_neden_tutarli(y):
    assert y.tur == ("algoritma" if y.ad in K.ALGORITMALAR else "ai")
    assert _nesne(y.kaynak) is not None
    assert bool(y.neden) is (not y.kosulur), "koşulmayanın NEDENİ yazılı olmalı, koşulanın olmamalı"


# ------------------------------------------------------------------ piyasalar
def test_piyasa_katalogu_tutarli():
    semboller = [s for _, s in P.tum_semboller()]
    assert len(semboller) == len(set(semboller)), "bir sembol iki piyasada"
    assert len(P.PIYASALAR) >= 12
    for ad, v in P.PIYASALAR.items():
        assert v["semboller"] and 240 <= v["gun"] <= 366, ad
    for k in P.VERI_KAYNAKLARI:
        assert {"ad", "kapsam", "anahtar", "durum"} <= set(k) and k["durum"]


def test_temizle_zarfi_duzeltir_bozuk_satiri_atar():
    ix = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03", "2020-01-06"],
                          tz="Europe/Berlin")
    ham = pd.DataFrame({"Open": [10, 11, 11.5, np.nan, 12], "High": [9, 12, 12, 13, 12.5],
                        "Low": [9.5, 10, 10, 11, 0], "Close": [10.5, 11.5, 11.8, 12, 12.2],
                        "Volume": [np.nan, 5, 5, 5, 5]}, index=ix)
    df = P.temizle(ham)
    assert df.index.tz is None and df.index.is_unique and df.index.is_monotonic_increasing
    assert len(df) == 2                      # NaN açılış + sıfır dip atıldı, çift gün tekillendi
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert df["volume"].iloc[0] == 0.0 and df["close"].iloc[1] == 11.8   # son kopya kalır


def test_olcekle_getirileri_korur():
    df = pd.DataFrame({"open": [70000.0, 71000, 69000], "high": [71000.0, 72000, 70000],
                       "low": [69000.0, 70000, 68000], "close": [70500.0, 71500, 68500],
                       "volume": [1.0, 2, 3]}, index=pd.date_range("2020-01-01", periods=3))
    s = P.olcekle(df)
    assert s["close"].iloc[0] == pytest.approx(100.0)
    for k in ("open", "high", "low"):                         # OHLC aynı SABİT çarpanla (zarf korunur)
        np.testing.assert_allclose(s[k] / df[k], 100.0 / 70500.0)
    np.testing.assert_allclose(s["close"].pct_change().to_numpy()[1:], df["close"].pct_change().to_numpy()[1:],
                               rtol=1e-12)
    assert (s["volume"] == df["volume"]).all()


def test_aralik_onari_tek_bar_yanlis_baskiyi_zarfa_indirir():
    # SHEL.L 2019-11-07 deseni: fiyat ~2.330, low 3,04
    df = pd.DataFrame({"open": [2330.0, 2320.0, 2310.0], "high": [2349.0, 2340.0, 2330.0],
                       "low": [2320.0, 3.04, 2300.0], "close": [2335.0, 2329.0, 2305.0], "volume": 1.0},
                      index=pd.bdate_range("2019-11-06", periods=3))
    o, n = P.aralik_onar(df)
    assert n == 1
    assert o["low"].iloc[1] == 2320.0 and o["high"].iloc[1] == 2329.0      # açılış-kapanış zarfı
    pd.testing.assert_frame_equal(o.drop(index=df.index[1]), df.drop(index=df.index[1]))
    assert P.aralik_onar(df.drop(index=df.index[1]))[1] == 0
    kaynak = inspect.getsource(K.hazirla)
    assert "aralik_onar" in kaynak and '"Kripto"' in kaynak                 # kriptoda onarım YOK


def _seri(n: int, sicra_gun: int | None = None, bas: str = "2008-01-01") -> pd.DataFrame:
    r = np.random.default_rng(0).normal(0, 0.01, n)
    if sicra_gun is not None:
        r[sicra_gun] = 0.8
    c = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0},
                        index=pd.bdate_range(bas, periods=n))


def test_kusur_kurali_kripto_disini_disler_kriptoyu_disleMEZ():
    temiz, kirli = _seri(3000), _seri(3000, sicra_gun=2800)
    assert K.kusur_nedeni("ABD", temiz) is None
    assert "log getiri" in K.kusur_nedeni("ABD", kirli)
    assert K.kusur_nedeni("Kripto", kirli) is None           # kriptoda %80'lik gün gerçektir
    assert "yetersiz" in K.kusur_nedeni("ABD", _seri(300))


# ------------------------------------------------------------------ ölçüm yardımcıları
def test_pozisyon_getirisi_maliyeti_yalniz_degisimde_oder():
    r = K.pozisyon_getirisi(np.array([1, 1, -1, 0.0]), np.array([0.01, 0.02, 0.01, 0.05]), 0.001)
    np.testing.assert_allclose(r, [0.01 - 0.001, 0.02, -0.01 - 0.002, -0.001])


def test_auc_ve_nw_yardimcilari():
    assert K.auc(np.ones(100, dtype=int), np.linspace(0, 1, 100)) is None   # tek sınıf → tanımsız
    y = np.r_[np.zeros(60, dtype=int), np.ones(60, dtype=int)]
    assert K.auc(y, y.astype(float)) == 1.0
    x = np.random.default_rng(1).normal(0, 0.01, 500)
    a = K.nw_alfa(2.0 * x, x, 252)
    assert a["beta"] == pytest.approx(2.0) and abs(a["alfa_yillik_pct"]) < 1e-9


# ------------------------------------------------------------------ look-ahead sözleşmeleri
def _xy(n: int = 420, f: int = 5, seed: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float32)
    y = (rng.random(n) > 0.5).astype(np.int8)
    y[-2:] = -1
    return X, y


def test_ftrl_etiketi_iki_bar_gecikmeli_ogrenir():
    X, y = _xy()
    p0 = K._ftrl_kos(X, y)
    k = 330
    y2 = y.copy()
    y2[k] = 1 - y2[k]
    p1 = K._ftrl_kos(X, y2)
    np.testing.assert_array_equal(p0[: k + 2], p1[: k + 2])  # y_k bar k+2 kararından önce KULLANILMAZ
    assert not np.allclose(p0[k + 2:], p1[k + 2:])           # ve sonra gerçekten kullanılır
    # özellik ölçeklemesi de nedensel: m. satırın özelliği m'den önceki tahmini değiştirmez
    X2 = X.copy()
    X2[k] += 50.0
    np.testing.assert_array_equal(p0[:k], K._ftrl_kos(X2, y)[:k])


def test_hedge_etiketi_iki_bar_gecikmeli_ogrenir_ve_agirlik_toplami_bir():
    rng = np.random.default_rng(5)
    E = rng.integers(-1, 2, size=(420, 8)).astype(np.int8)
    _, y = _xy()
    p0, w = K._hedge_kos(E, y)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3) and "taban_orani" in w   # 4 ondalık yuvarlama
    k = 300
    y2 = y.copy()
    y2[k] = 1 - y2[k]
    p1, _ = K._hedge_kos(E, y2)
    np.testing.assert_array_equal(p0[: k + 2], p1[: k + 2])
    assert not np.allclose(p0[k + 2: -2], p1[k + 2: -2])


def test_hmm_rejim_olasiligi_nedenseldir():
    n = 1400
    r = np.random.default_rng(7).normal(0.0003, 0.01, n)
    tarih = pd.bdate_range("2013-01-01", periods=n)
    t0 = int(np.searchsorted(tarih.values, np.datetime64("2017-01-01")))
    p0 = K._hmm_kos(r, tarih, t0)
    m = t0 + 40
    r2 = r.copy()
    r2[m] += 0.05
    p1 = K._hmm_kos(r2, tarih, t0)
    assert np.isfinite(p0[t0:m]).all()
    np.testing.assert_allclose(p0[:m], p1[:m])               # m. barın şoku öncesini değiştirmez


def _garch_seri(n: int, bas: str = "2014-01-01") -> pd.DataFrame:
    """GARCH(1,1) üretimi (ω=2e-6, α=0,12, β=0,85): oynaklık KÜMELENMESİ taşır. İlk sürüm i.i.d. seri
    kullanıyordu → GARCH α sınırda 0 çıkıyor, bugünkü getiri öngörüyü hiç etkilemiyordu ve ileri-bakış
    mutasyonu testi GEÇİYORDU (dejenere kurgu, Tur-73)."""
    rng = np.random.default_rng(21)
    r = np.zeros(n)
    s2 = 2e-6 / (1 - 0.97)
    for t in range(1, n):
        s2 = 2e-6 + 0.12 * r[t - 1] ** 2 + 0.85 * s2
        r[t] = math.sqrt(s2) * rng.standard_normal()
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996,
                         "close": c, "volume": 1.0}, index=pd.bdate_range(bas, periods=n))


def test_oynaklik_ongorusu_gelecegi_gormez():
    df = _garch_seri(900)
    t0 = 700
    v0 = K._vol_kos(df, t0)
    m = 760
    df2 = df.copy()
    df2.iloc[m, df2.columns.get_loc("close")] *= 1.05
    v1 = K._vol_kos(df2, t0)
    for ad in ("ewma", "garch", "gjr", "har", "har_olcekli"):
        # j. kapanışta verilen öngörü (lr[j+1] için) kapanış m'den önce, yani j ≤ m−1'de DEĞİŞMEMELİ
        np.testing.assert_allclose(v0["tahmin"][ad][: m - t0], v1["tahmin"][ad][: m - t0], err_msg=ad)
        assert np.isfinite(v0["tahmin"][ad]).mean() > 0.9, ad
        assert not np.allclose(v0["tahmin"][ad][m - t0 + 1:], v1["tahmin"][ad][m - t0 + 1:], equal_nan=True), ad


def test_basarisiz_yeniden_fit_onceki_parametrelerle_surer(monkeypatch):
    """İlk sürüm başarısız GJR yeniden-fit'inde o çeyreği NaN bırakıyordu (sentetik seride %63)."""
    import nihai.ai.quant_math as qm
    asil, say = qm.gjr_garch_fit, {"n": 0}

    def ikincide_bos(x):
        say["n"] += 1
        return None if say["n"] == 2 else asil(x)
    monkeypatch.setattr(qm, "gjr_garch_fit", ikincide_bos)
    v = K._vol_kos(_garch_seri(900), 700)
    assert say["n"] >= 3 and np.isfinite(v["tahmin"]["gjr"]).all()


def test_yuruyen_egitim_aralik_sonu_etiketini_egitime_sokmaz():
    P_ = pd.DataFrame({"karar": pd.to_datetime(["2017-12-27", "2017-12-28", "2017-12-29", "2018-01-02"]),
                       "bilinir": pd.to_datetime(["2017-12-29", "2018-01-02", "2018-01-03", "2018-01-04"])})
    tr, te = K.egitim_test_bolmesi(P_, 2018)
    assert tr.tolist() == [True, False, False, False]       # 28-29 Aralık kararının etiketi Ocak'ta doğar
    assert te.tolist() == [False, False, False, True]
    kaynak = inspect.getsource(K.piyasa_ml_kos)
    assert 'P["karar"] < ilk' not in kaynak and "egitim_test_bolmesi(P, yil)" in kaynak


def test_ai_satiri_yonu_dogru_isaretler():
    n, t0 = 400, 100
    rng = np.random.default_rng(11)
    ileri = rng.normal(0, 0.01, n)
    ileri[-2:] = np.nan
    y = np.where(ileri > 0, 1, 0).astype(np.int8)
    y[-2:] = -1
    tarih = pd.bdate_range("2019-01-01", periods=n).to_numpy()
    E = {"S": {"t0": t0, "n": n, "ileri": ileri, "y": y, "tarih": tarih}}
    bh = pd.Series(ileri[t0: n - 2], index=pd.DatetimeIndex(tarih[t0 + 1: n - 1]))
    kahin = np.where(y == 1, 0.9, 0.1).astype(float)
    d = K._ai_satiri(E, {"S": kahin}, bh, 252, 0.0)
    assert d["auc"] == 1.0 and d["sharpe"] > 5 and d["sharpe_brut"] == d["sharpe"]
    ters = K._ai_satiri(E, {"S": 1.0 - kahin}, bh, 252, 0.0)
    assert ters["sharpe"] < -5
    duz = K._ai_satiri(E, {"S": kahin}, bh, 252, 0.0, uzun_duz=True)
    assert duz["sharpe"] > 0 and duz["gunluk_devir"] < d["gunluk_devir"]


def test_algoritma_kosusu_tarih_hizali_ve_kapi_kipini_geri_yukler():
    from nihai.config import CONFIG
    df = P.olcekle(_seri(420, bas="2016-06-01"))
    asil = CONFIG.strategy.meta_gate_mode
    object.__setattr__(CONFIG.strategy, "meta_gate_mode", "shadow")
    try:
        out = K._algoritma_kos("SENTETIK", df, 300, "donchian", kapi="enforce")
        assert CONFIG.strategy.meta_gate_mode == "shadow"   # "off"a değil ÖNCEKİ kipe döner
    finally:
        object.__setattr__(CONFIG.strategy, "meta_gate_mode", asil)
    assert out["r"].size == len(df) - 1 - 300 == out["tarih"].size
    assert pd.Timestamp(out["tarih"][0]) == df.index[301]
    # işlemsiz koşu (profit_factor None) çökmemeli — ilk sürüm FX'te (hacim yok → 0 işlem) düşerdi
    duz = df.copy()
    duz[["open", "high", "low", "close"]] = 100.0
    bos = K._algoritma_kos("SENTETIK", duz, 300, "donchian")
    assert bos["islem"] == 0 and bos["pf"] is None and not np.any(bos["r"])
