"""Tur-100 — ML oynaklık modelini derinleştirme düzeneği (research/quant_ml_derin.py).

Kilitlenen: yeni bilgi özelliklerinin KESİN ÖNCEKİ tarihten gelmesi (saat dilimi sızıntısı yok), kilitli ml21
tarifinin Tur-98 ve ileri sınama ile birebir kalması, zamansal erken durdurmanın doğrulamayı eğitimden SONRA ve
arınmalı kurması, yürüyen eğitimin geleceği / izleme evrenini / pencere dışını görmemesi, adaptasyon kollarının
(çeyreklik, yuvarlanan, donmuş, havuz payı) tanımı, ön-kayıtlı G1/G2 kapıları ve hüküm tablosu.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from research import ileri_sigma_ml21 as ISM
from research import quant_kayma as QK
from research import quant_ml as QM
from research import quant_ml_derin as D
from research import quant_ml_motor as QMM

PEN = next(iter(QK.SIGMA_PENCERE))


# ---------------------------------------------------------------------------------------------------
# D3 — yeni bilgi: yalnız kesin önceki tarih
# ---------------------------------------------------------------------------------------------------
def test_once_ayni_gunu_ve_gelecegi_kullanmaz():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    t = pd.to_datetime(["2019-12-31", "2020-01-01", "2020-01-02", "2020-01-05"])
    np.testing.assert_array_equal(D.once(s, t), [np.nan, np.nan, 1.0, 3.0])
    assert np.isnan(D.once(pd.Series(dtype=float), t)).all()
    assert 'side="left") - 1' in inspect.getsource(D.once)       # regresyon kapısı: aynı gün yasak


def _bilgi_paneli() -> pd.DataFrame:
    gun = pd.bdate_range("2021-03-01", periods=6)
    satir = []
    for pz, semboller in (("A", ("a1", "a2")), ("B", ("b1",)), (QK.IZLEME, ("i1",))):
        for k, s in enumerate(semboller):
            for j, g in enumerate(gun):
                satir.append({"t": g, "piyasa": pz, "sembol": s, "ln_rv5": -9.0 + j + k + (pz == "B"),
                              "ln_rv22": -8.0 + 2 * j + k + 5 * (pz == QK.IZLEME)})
    return pd.DataFrame(satir).set_index("t")


def test_bilgi_ozellikleri_kesin_onceki_gunden_ve_sizintisiz():
    P = _bilgi_paneli()
    gun = P.index.unique().sort_values()
    vix = pd.Series(np.arange(10.0, 16.0), index=gun)
    vix3m = pd.Series(np.arange(12.0, 18.0), index=gun)
    B = D.bilgi_ekle(P, vix, vix3m)
    g0, g1 = gun[0], gun[1]
    a = B[(B.index == g1) & (B.piyasa == "A")]
    a0 = P[(P.index == g0) & (P.piyasa == "A")]
    assert (a["pz_rv22"] == a0["ln_rv22"].mean()).all() and (a["pz_rv5"] == a0["ln_rv5"].mean()).all()
    assert B.loc[B.index == g0, ["pz_rv5", "pz_rv22", "gl_rv22", "ln_vix", "vix_ts"]].isna().all().all()
    h0 = P[(P.index == g0) & (P.piyasa != QK.IZLEME)]
    gl = h0.groupby("piyasa")["ln_rv22"].mean().mean()           # piyasa ortalamalarının ortalaması, izleme HARİÇ
    assert np.allclose(B.loc[B.index == g1, "gl_rv22"], gl)
    assert np.allclose(B.loc[B.index == g1, "ln_vix"], np.log((10.0 / 100) ** 2 / 252))
    assert np.allclose(B.loc[B.index == g1, "vix_ts"], np.log(10.0 / 12.0))
    # g1 ve sonrasını boz → g1 satırlarının özellikleri DEĞİŞMEZ (aynı gün ve gelecek görülmez)
    P2 = P.copy()
    P2.loc[P2.index >= g1, ["ln_rv5", "ln_rv22"]] = 99.0
    v2, v3 = vix.copy(), vix3m.copy()
    v2[v2.index >= g1] = 80.0
    v3[v3.index >= g1] = 1.0
    B2 = D.bilgi_ekle(P2, v2, v3)
    for k in D.YENI_OZ:
        np.testing.assert_array_equal(B.loc[B.index == g1, k].to_numpy(), B2.loc[B2.index == g1, k].to_numpy())


# ---------------------------------------------------------------------------------------------------
# Tarifler
# ---------------------------------------------------------------------------------------------------
def test_ml21_tarifi_tur98_ve_ileri_sinama_ile_birebir():
    p = QMM._gbm().get_params()
    assert {k: p[k] for k in D.GBM} == D.GBM
    assert D.GBM == ISM.GBM
    m = D._model("ml21")
    assert m.get_params()["early_stopping"] == "auto"            # kilitli tarif olduğu gibi (D2 teşhisi dahil)
    k1, k2 = D._model("k1", 37), D._model("k2", 37)
    assert k1.get_params()["early_stopping"] is False and k1.get_params()["max_iter"] == 37
    assert k2.get_params()["loss"] == "gamma" and k1.get_params()["loss"] == "squared_error"
    assert '"early_stopping": False' in inspect.getsource(D._model)


def test_on_kayitli_kollar_ve_ozellikler():
    assert D.ONGORU_KOLLARI == {"ml21": (QM.B_OZ, "ml21"), "k1": (QM.B_OZ, "k1"), "k2": (QM.B_OZ, "k2"),
                                "k3": (QM.B_OZ + D.YENI_OZ, "k1")}   # aday ln hedefini korur (D1 bilgi kolu)
    assert D.YENI_OZ == ["pz_rv5", "pz_rv22", "gl_rv22", "ln_vix", "vix_ts"]
    assert D.ES == {"dogrulama_gun": 365, "arinma_gun": 31, "max_iter": 600, "c_orneklem": 50_000}
    assert D.KAPI == {"min_piyasa": 10, "z": 1.645}
    assert D.MOTOR_KOLLARI == ("ewma", "ml21", "k1", "k3")
    assert set(D.ADAPTASYON) == {"ceyrek", "yuv2", "yuv5", "donmus11", "donmus18", "havuz25", "havuz50"}


def test_sinir_maskesi_yil_basinda_kilitli_formulle_ayni():
    t = pd.date_range("2008-01-01", "2026-12-31", freq="D")
    for y in (2011, 2018, 2026):
        np.testing.assert_array_equal(D.sinir_maskesi(t, pd.Timestamp(f"{y}-01-01"), 22), QM.egitim_maskesi(t, y, 22))


def test_zamansal_dogrulama_egitimden_sonra_ve_arinmali():
    t = pd.bdate_range("2015-01-01", "2019-12-31")
    tr, va = D.es_bolme(t)
    assert tr.any() and va.any() and not (tr & va).any()
    assert t[va].min() - t[tr].max() > pd.Timedelta(days=D.ES["arinma_gun"])   # 21 günlük hedef taşmaz
    assert t[va].max() == t.max() and (t.max() - t[va].min()).days <= D.ES["dogrulama_gun"]


def test_zaman_es_dogrulamada_en_iyi_iterasyonu_secer(monkeypatch):
    monkeypatch.setitem(D.ES, "max_iter", 60)
    rng = np.random.default_rng(0)
    t = pd.bdate_range("2012-01-01", periods=3000)
    X = rng.normal(size=(t.size, 3))
    rv = np.exp(0.8 * X[:, 0] + rng.normal(scale=0.3, size=t.size)) * 1e-4
    n, kayip = D.zaman_es(X, rv, t, "k1")
    assert len(kayip) == 60 and n == int(np.argmin(kayip)) + 1
    tr, va = D.es_bolme(t)                                        # son aşamanın kaybı elle: c eğitim alt örnekleminden
    m60 = D._model("k1", 60).fit(X[tr], np.log(rv[tr] + QM.EPS))
    alt = np.random.default_rng(0).choice(np.flatnonzero(tr), size=min(D.ES["c_orneklem"], int(tr.sum())), replace=False)
    f = np.exp(m60.predict(X[va])) * (rv[alt].mean() / np.exp(m60.predict(X[alt])).mean())
    assert kayip[-1] == pytest.approx(float(np.mean(np.log(f) + rv[va] / f)), rel=1e-9)
    m, c, n2 = D.egit(X, rv, t, "k1")                            # yeniden eğitim tam o sayıyla, erken durmasız
    assert n2 == n and m.get_params()["early_stopping"] is False
    gurultu = np.exp(rng.normal(size=t.size)) * 1e-4              # bilgisiz hedef: en iyi nokta İÇERİDE (tavan değil)
    ng, _ = D.zaman_es(X, gurultu, t, "k1")
    assert ng < D.ES["max_iter"] and D.egit(X, gurultu, t, "k1")[2] == ng
    assert c == pytest.approx(np.mean(rv) / np.mean(np.exp(m.predict(X))))
    m2, c2, _ = D.egit(X, rv, t, "k2")
    assert c2 == 1.0                                             # gamma: ortalama tutarlı, düzeltme yok
    np.testing.assert_array_equal(D.tahmin(m2, c2, X, "k2"), m2.predict(X))   # gamma çıktısı zaten ortalama


# ---------------------------------------------------------------------------------------------------
# Yürüyen eğitim ve adaptasyon kolları
# ---------------------------------------------------------------------------------------------------
def _sentetik(n_gun=1300, n_sym=9, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parca = []
    for k in range(n_sym):
        idx = pd.bdate_range("2008-01-01", periods=n_gun)
        d = pd.DataFrame(rng.normal(size=(n_gun, len(QM.B_OZ))), columns=QM.B_OZ, index=idx)
        d["rv21"] = np.exp(0.5 * d["ln_rv22"] + rng.normal(scale=0.3, size=n_gun)) * 1e-4
        d["sembol"], d["piyasa"] = f"S{k}", (QK.IZLEME if k == n_sym - 1 else "X")
        parca.append(d)
    return pd.concat(parca)


@pytest.fixture
def hizli(monkeypatch):
    monkeypatch.setattr(D, "GBM", {**D.GBM, "max_iter": 15})
    monkeypatch.setitem(D.ES, "max_iter", 15)
    monkeypatch.setattr(QM, "TEST_YILLARI", (2011, 2012))


@pytest.mark.parametrize("tarif", ["ml21", "k1"])
def test_yuruyen_gelecegi_ve_izlemeyi_gormez(hizli, tarif):
    P0 = _sentetik()
    havuz = (P0["piyasa"] != QK.IZLEME).to_numpy()
    a, bilgi = D.yuruyen(P0, havuz, QM.B_OZ, tarif)
    P1 = P0.copy()
    P1.loc[~QM.egitim_maskesi(P1.index, 2011, QM.ILERI + 1), "rv21"] *= 50   # etiketi 2011'den önce bilinmeyen satırlar
    P1.loc[~havuz, "rv21"] = 1.0                                  # izleme etiketleri
    b, _ = D.yuruyen(P1, havuz, QM.B_OZ, tarif)
    y11 = np.asarray(P0.index.year == 2011)
    assert np.isfinite(a[y11]).sum() > 1000 and set(bilgi) == {"2011-01-01", "2012-01-01"}
    np.testing.assert_array_equal(a[y11], b[y11])
    assert np.isfinite(a[y11 & ~havuz]).all()                    # izleme TAHMİN alır, eğitime girmez


def test_yuvarlanan_pencere_eski_veriyi_gormez(hizli):
    P0 = _sentetik()
    havuz = (P0["piyasa"] != QK.IZLEME).to_numpy()
    a, bilgi = D.yuruyen(P0, havuz, QM.B_OZ, "ml21", geriye_yil=3)
    tam, bt = D.yuruyen(P0, havuz, QM.B_OZ, "ml21")
    assert bilgi["2012-01-01"]["n_egitim"] < bt["2012-01-01"]["n_egitim"]
    P1 = P0.copy()
    P1.loc[np.asarray(P1.index < pd.Timestamp("2009-01-01")), "rv21"] *= 50
    b, _ = D.yuruyen(P1, havuz, QM.B_OZ, "ml21", geriye_yil=3)
    y12 = np.asarray(P0.index.year == 2012)
    np.testing.assert_array_equal(a[y12], b[y12])


def test_donmus_model_tek_egitimle_tum_yillari_ongorur(hizli):
    P0 = _sentetik()
    havuz = (P0["piyasa"] != QK.IZLEME).to_numpy()
    d, bilgi = D.yuruyen(P0, havuz, QM.B_OZ, "ml21", donem=("donmus", 2011))
    yil, _ = D.yuruyen(P0, havuz, QM.B_OZ, "ml21")
    assert list(bilgi) == ["2011-01-01"]
    y11, y12 = np.asarray(P0.index.year == 2011), np.asarray(P0.index.year == 2012)
    np.testing.assert_array_equal(d[y11], yil[y11])               # 2011: aynı model
    assert np.isfinite(d[y12]).all() and not np.array_equal(d[y12], yil[y12])


def test_donemler_ceyrek_ve_donmus():
    q = D.donemler("ceyrek")
    assert q[0] == (pd.Timestamp("2011-01-01"), pd.Timestamp("2011-04-01")) and len(q) == 64
    assert all(b == c for (_, b), (c, _) in zip(q, q[1:], strict=False))
    assert D.donemler(("donmus", 2018)) == [(pd.Timestamp("2018-01-01"), pd.Timestamp("2027-01-01"))]


def test_havuz_alt_kumesi_enstruman_duzeyinde_ve_belirlenimci():
    P = _sentetik(n_gun=20, n_sym=9)
    havuz = (P["piyasa"] != QK.IZLEME).to_numpy()
    m = D.havuz_alt_kumesi(P, havuz, 0.5)
    np.testing.assert_array_equal(m, D.havuz_alt_kumesi(P, havuz, 0.5))
    secilen = set(P.loc[m, "sembol"])
    assert len(secilen) == 4 and not (m & ~havuz).any()           # 8 havuz enstrümanının yarısı, izleme yok
    assert all(m[(P["sembol"] == s).to_numpy()].all() for s in secilen)   # enstrüman ya tüm satırıyla ya hiç


def test_havuz_payi_yalniz_secilen_enstrumanlarla_egitir(hizli):
    P0 = _sentetik(n_sym=17)
    havuz = (P0["piyasa"] != QK.IZLEME).to_numpy()
    a, bilgi = D.yuruyen(P0, havuz, QM.B_OZ, "ml21", havuz_payi=0.5)
    _, tam = D.yuruyen(P0, havuz, QM.B_OZ, "ml21")
    oran = bilgi["2012-01-01"]["n_egitim"] / tam["2012-01-01"]["n_egitim"]
    assert oran == pytest.approx(0.5, abs=0.01)
    P1 = P0.copy()
    P1.loc[havuz & ~D.havuz_alt_kumesi(P0, havuz, 0.5), "rv21"] = 1.0   # seçilmeyenlerin etiketi görülmez
    b, _ = D.yuruyen(P1, havuz, QM.B_OZ, "ml21", havuz_payi=0.5)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a[np.asarray(P0.index.year == 2012)]).all()      # öngörü TÜM satırlara


def test_permutasyon_onemi_kullanilan_ozelligi_bulur():
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 2))
    rv = np.exp(X[:, 0]) * 1e-4
    m = HistGradientBoostingRegressor(max_iter=50, random_state=0).fit(X, np.log(rv))
    o = D.permutasyon_onemi(m, 1.0, X, rv, "k1", ["a", "b"])
    assert o["a"] > 0.1 and abs(o["b"]) < 0.01


# ---------------------------------------------------------------------------------------------------
# Kapılar ve hüküm
# ---------------------------------------------------------------------------------------------------
def _ongoru_paneli(izleme_iyi: bool = True, n_pz: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=400)
    parca = []
    for k in range(n_pz + 1):
        pz = QK.IZLEME if k == n_pz else f"P{k}"
        rv = np.exp(rng.normal(size=idx.size)) * 1e-4
        iyi = rv * np.exp(rng.normal(scale=0.1, size=idx.size))
        kotu = 2 * rv * np.exp(rng.normal(scale=0.1, size=idx.size))
        a, b = (iyi, kotu) if (pz != QK.IZLEME or izleme_iyi) else (kotu, iyi)
        parca.append(pd.DataFrame({"piyasa": pz, "rv21": rv, "f_a": a, "f_b": b}, index=idx))
    return pd.concat(parca)


def test_g1_ongoru_kapisi():
    P = _ongoru_paneli()
    m = D.ongoru_farki(P, "f_a", "f_b")
    assert m["olumlu"] == 12 and m["izleme_t"] < -1.645 and m["gecti"] and m["ort_d"] < 0 and m["n_sonlu_disi"] == 0
    P.iloc[:3, P.columns.get_loc("f_a")] = [np.inf, 0.0, np.nan]   # taşma ve ≤0 sayılır; NaN (kol yok) sayılmaz
    assert D.ongoru_farki(P, "f_a", "f_b")["n_sonlu_disi"] == 2
    m = D.ongoru_farki(_ongoru_paneli(izleme_iyi=False), "f_a", "f_b")
    assert m["izleme_t"] > 0 and not m["gecti"]                   # izleme tersse geçmez
    m = D.ongoru_farki(_ongoru_paneli(n_pz=8), "f_a", "f_b")
    assert m["olumlu"] == 9 and not m["gecti"]                   # 10 piyasa şartı


def _motor(fark: float, izleme_fark: float, n_pz: int = 12, n_olumlu: int | None = None) -> dict:
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2011-01-03", periods=1500)
    ew = {}
    for k in range(n_pz):
        pz = QK.IZLEME if k == n_pz - 1 else f"P{k}"
        b = pd.Series(rng.normal(0.0003, 0.01, idx.size), index=idx)
        d = izleme_fark if pz == QK.IZLEME else (fark if n_olumlu is None or k < n_olumlu else -0.00005)
        ew[pz] = {f"ewma_{PEN}": b, f"k3_{PEN}": b + rng.normal(d, 0.001, idx.size)}
    return ew


def test_g2_karar_kapisi():
    m = D.karar_olcumu(_motor(0.0003, 0.0003), "k3", "ewma", PEN)
    assert m["z"] > 1.645 and m["olumlu"] == 12 and m["izleme_ds"] > 0 and m["gecti"] and m["mde"] > 0
    m = D.karar_olcumu(_motor(0.0003, -0.0003), "k3", "ewma", PEN)
    assert m["izleme_ds"] < 0 and not m["gecti"]                  # izleme ΔS ≤ 0 → geçmez
    m = D.karar_olcumu(_motor(0.0, 0.0), "k3", "ewma", PEN)
    assert not m["gecti"]
    m = D.karar_olcumu(_motor(0.0006, 0.0006, n_olumlu=6), "k3", "ewma", PEN)
    assert m["z"] > 1.645 and m["izleme_ds"] > 0 and m["olumlu"] == 7 and not m["gecti"]   # 10 piyasa şartı


def _karne(g1: bool, g2: bool, adapt: bool = False) -> dict:
    og = {"gecti": g1}
    return {"ongoru": {p: {"k3_ml21": og, **{f"{a}_ml21": {"gecti": adapt} for a in D.ADAPTASYON}}
                       for p in QK.SIGMA_PENCERE},
            "karar": {p: {"k3_ewma": {"gecti": g2}} for p in QK.SIGMA_PENCERE}}


def test_hukum_tablosu():
    assert D.hukum(_karne(True, True))["sonuc"].startswith("k3 İLERİ SINAMA ADAYI")
    assert D.hukum(_karne(True, False))["sonuc"] == "öngörü iyileşti, karar değeri kanıtlanmadı"
    assert D.hukum(_karne(False, True))["sonuc"] == "tarif değişmez"          # G2 tek başına yetmez
    assert D.hukum(_karne(False, False, adapt=True))["adaptasyon_gecen"] == list(D.ADAPTASYON)
    k = _karne(True, True)
    k["ongoru"][PEN]["k3_ml21"] = {"gecti": False}                 # İKİ pencerede de şart
    assert not D.hukum(k)["G1"]


def test_teshis_kollari_sonradan_kapisiz_ve_sabit_tarif():
    assert "TEŞHİS (SONRADAN, kapısız" in (D.__doc__ or "")
    assert D.TESHIS_KOLLARI == {"t_tam300": {"oz": QM.B_OZ, "tarif": "sabit", "n_sabit": 300},
                                "t_n100": {"oz": QM.B_OZ, "tarif": "sabit", "n_sabit": 100},
                                "t_n600": {"oz": QM.B_OZ, "tarif": "sabit", "n_sabit": 600},
                                "t_ml21_bilgi": {"oz": QM.B_OZ + D.YENI_OZ, "tarif": "ml21"}}
    kaynak = inspect.getsource(D.hukum)
    assert "TESHIS" not in kaynak and "t_" not in kaynak          # teşhis kolu hükme girmez
    rng = np.random.default_rng(0)
    t = pd.bdate_range("2012-01-01", periods=2000)
    X = rng.normal(size=(t.size, 2))
    rv = np.exp(X[:, 0]) * 1e-4
    m, c, n = D.egit(X, rv, t, "sabit", 25)
    assert n == 25 and m.get_params()["early_stopping"] is False and m.get_params()["loss"] == "squared_error"
    assert c == pytest.approx(np.mean(rv) / np.mean(np.exp(m.predict(X))))


def test_kilitli_ileri_sinama_tarifine_dokunulmadi():
    kaynak = inspect.getsource(D)
    for yasak in ("ISM.", "ileri_sigma_ml21", "model_kaydet", "defter_"):
        assert yasak not in kaynak                                # bu modül ileri sınamayı OKUMAZ/YAZMAZ
