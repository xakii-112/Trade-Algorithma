"""σ_ML21 ileri sınaması (`research.ileri_sigma_ml21`) — ön-kayıt kilidinin sözleşmeleri.

Sınama ancak şunlar tutarsa anlamlıdır: model tarifi Tur-98'inkidir ve yalnız GEÇMİŞ etiketle eğitilir; her yılın
öngörüsü o yılın dondurulmuş modelinden gelir (eksik model iki kolu sessizce aynılaştırmaz); kollar tek değişkende
(σ kaynağı) ayrışır ve geleceği görmez; deftere yalnız kayıt gününden SONRAKİ günler girer; kapılar ön-kayıttaki
gibidir ve karar günlerinden önce hüküm verilmez. Ağ gerektirmez.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nihai.signals import quant as Q
from research import ileri_sigma_ml21 as Y
from research import quant_kayma as QK
from research import quant_ml as QM
from research.ileri_kripto_yaris import parmak_izi

GUN = pd.Timedelta(days=1)


def _ohlc(close, bas="2024-06-03"):
    c = np.asarray(close, float)
    o = np.r_[c[0], c[:-1]] * 1.001
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99, "close": c},
                        index=pd.bdate_range(bas, periods=c.size))


def _yol(n=520, seed=3):
    rng = np.random.default_rng(seed)
    lr = rng.normal(0.0006, 0.013, n) + 0.003 * np.sin(np.arange(n) / 45)
    return _ohlc(100 * np.exp(np.cumsum(lr)))


# ── model ──────────────────────────────────────────────────────────────────────────────────────────
def test_gbm_tarifi_tur98_ile_ayni():
    from sklearn.ensemble import HistGradientBoostingRegressor

    from research import quant_ml_motor as QMM
    assert HistGradientBoostingRegressor(**Y.GBM).get_params() == QMM._gbm().get_params()


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-02", "2026-06-30")
    par = []
    for pz, s in (("ABD", "A"), ("ABD", "B"), ("Avrupa", "C"), (QK.IZLEME, "A")):
        g = pd.DataFrame(rng.normal(size=(len(idx), len(QM.B_OZ))), index=idx, columns=QM.B_OZ)
        g["rv21"] = np.exp(rng.normal(-9, 0.5, len(idx)))
        par.append(g.assign(piyasa=pz, sembol=s))
    return pd.concat(par)


def test_model_yalniz_havuzda_ve_yil_basindan_once_bilinen_etiketle_egitilir():
    P = _panel()
    d = Y.model_egit(P, 2026)
    beklenen = ((P["piyasa"] != QK.IZLEME) & QM.egitim_maskesi(P.index, 2026, QM.ILERI + 1)).sum()
    assert d["n"] == beklenen and d["yil"] == 2026 and d["c"] > 0
    m = ((P["piyasa"] != QK.IZLEME) & QM.egitim_maskesi(P.index, 2026, QM.ILERI + 1)).to_numpy()
    duz = np.exp(d["model"].predict(P[QM.B_OZ].to_numpy()[m])) * d["c"]
    np.testing.assert_allclose(duz.mean(), P["rv21"].to_numpy()[m].mean(), rtol=1e-9)   # yanlılık düzeltmesi
    P.loc[P["piyasa"] == QK.IZLEME, "rv21"] *= 1e6                 # izleme hedefi eğitime girmez
    P.loc[~QM.egitim_maskesi(P.index, 2026, QM.ILERI + 1), "rv21"] *= 1e6   # etiketi yıl başından sonra bilinen de
    d2 = Y.model_egit(P, 2026)
    X = P[QM.B_OZ].to_numpy()[:50]
    np.testing.assert_array_equal(d["model"].predict(X), d2["model"].predict(X))
    assert d2["c"] == d["c"]


class _Sabit:
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(X), self.v)


def test_ongoru_yilin_modelini_kullanir_ve_eksik_yilda_durur():
    idx = pd.bdate_range("2025-12-29", "2026-12-31")
    P = pd.DataFrame(np.zeros((len(idx), len(QM.B_OZ))), index=idx, columns=QM.B_OZ)
    f = Y.ongoru(P, {2026: {"model": _Sabit(np.log(4e-4)), "c": 1.5}})
    assert np.isnan(f[idx < Y.MOTOR_BAS]).all()                  # motor öncesi yıl öngörülmez
    np.testing.assert_allclose(f[idx >= Y.MOTOR_BAS], 6e-4)
    P2 = pd.concat([P, P.iloc[:1].set_axis([pd.Timestamp("2027-01-04")])])
    f2 = Y.ongoru(P2, {2026: {"model": _Sabit(np.log(4e-4)), "c": 1.5}, 2027: {"model": _Sabit(np.log(1e-4)), "c": 1.0}})
    np.testing.assert_allclose(f2[-1], 1e-4)                    # 2027 satırı 2027 modelinden
    with pytest.raises(RuntimeError, match="2027"):
        Y.ongoru(P2, {2026: {"model": _Sabit(0.0), "c": 1.0}})


# ── kollar: gerçek motor ───────────────────────────────────────────────────────────────────────────
def test_kollar_tek_degiskende_ayrisir():
    df = _yol()
    _, sig = Q.seriler(df["close"].to_numpy(float))
    ayni = Y.kos("SYN", df, pd.Series(sig ** 2, df.index))
    np.testing.assert_allclose(ayni["ml21"].to_numpy(), ayni["ewma"].to_numpy(), atol=1e-7)
    genis = Y.kos("SYN", df, pd.Series(4 * sig ** 2, df.index))   # σ iki kat → stop/adet değişir
    pd.testing.assert_series_equal(genis["ewma"], ayni["ewma"])
    assert np.abs(genis["ml21"].to_numpy() - genis["ewma"].to_numpy()).max() > 1e-4
    assert genis["ewma"].index[0] > Y.MOTOR_BAS


def test_kollar_gelecegi_gormez():
    df = _yol()
    _, sig = Q.seriler(df["close"].to_numpy(float))
    f = pd.Series(2 * sig ** 2, df.index)
    k = int(np.searchsorted(df.index, pd.Timestamp("2026-03-02")))
    oyna = df.copy()
    oyna.iloc[k + 1:] *= 1.3
    a, b = Y.kos("SYN", df, f), Y.kos("SYN", oyna, f)
    for kol in Y.KOLLAR:
        pd.testing.assert_series_equal(a[kol].loc[: df.index[k]], b[kol].loc[: df.index[k]])


# ── defter ─────────────────────────────────────────────────────────────────────────────────────────
def test_defter_yalniz_kayittan_ve_anahtarin_son_gununden_sonrasini_alir():
    gunler = pd.date_range(Y.KAYIT_GUNU - 2 * GUN, periods=6)                          # 09-25 … 09-30
    R = pd.DataFrame([{"tarih": t, "piyasa": p, "sembol": s, "kol": k, "getiri": 0.01}
                      for t in gunler for p, s in (("ABD", "SPY"), ("Kripto", "BTC")) for k in Y.KOLLAR])
    anahtar = ["piyasa", "sembol", "kol"]
    y = Y.yeni_satirlar(R, R.iloc[:0], anahtar)
    assert y.tarih.min() == Y.KAYIT_GUNU + GUN and len(y) == 3 * 4
    defter = y[(y.sembol == "SPY") & (y.kol == "ewma") & (y.tarih <= Y.KAYIT_GUNU + 2 * GUN)]
    y2 = Y.yeni_satirlar(R, defter, anahtar)
    spy_ewma = y2[(y2.sembol == "SPY") & (y2.kol == "ewma")]
    assert list(spy_ewma.tarih) == [Y.KAYIT_GUNU + 3 * GUN] and len(y2) == 12 - 2


def test_uyusmayan_revizyonu_sayar_nan_i_esit_sayar():
    d = pd.DataFrame({"tarih": [Y.KAYIT_GUNU + GUN] * 2, "piyasa": "ABD", "sembol": ["A", "B"],
                      "ewma_f": [1e-4, 2e-4], "f_ml21": [np.nan, 3e-4]})
    assert Y.uyusmayan(d.copy(), d, ["piyasa", "sembol"], ["ewma_f", "f_ml21"]) == 0
    R = pd.DataFrame({"tarih": [Y.KAYIT_GUNU + GUN], "piyasa": "ABD", "sembol": "A", "kol": "ewma", "getiri": [0.01]})
    assert Y.uyusmayan(R.assign(getiri=0.010001), R, ["piyasa", "sembol", "kol"], ["getiri"]) == 1


# ── kilit ──────────────────────────────────────────────────────────────────────────────────────────
def test_iz_farki_kodu_yakalar():
    s = pd.Series(np.random.default_rng(2).normal(0, 0.01, 180), pd.bdate_range("2026-01-02", periods=180))
    eski = {"ABD/SPY": {"ewma": parmak_izi(s), "ml21": parmak_izi(s)}}
    assert Y.iz_farki(eski, json.loads(json.dumps(eski))) == []
    yuvarlama = {"ABD/SPY": {k: {**v, "sha": "x"} for k, v in eski["ABD/SPY"].items()}}
    assert Y.iz_farki(eski, yuvarlama) == []                   # 10. hane gürültüsü: yıllık toplamlar hakem
    bozuk = s.copy()
    bozuk.iloc[10] += 1e-6
    sorun = Y.iz_farki(eski, {"ABD/SPY": {"ewma": parmak_izi(s), "ml21": parmak_izi(bozuk)}})
    assert len(sorun) == 1 and "ml21" in sorun[0]
    assert Y.iz_farki(eski, {}) and len(Y.iz_farki(eski, {})) == 2


# ── kapılar ────────────────────────────────────────────────────────────────────────────────────────
def _R(etki, n=300, seed=4):
    rng = np.random.default_rng(seed)
    gun = pd.bdate_range(Y.KAYIT_GUNU + GUN, periods=n)
    satir = []
    for i, pz in enumerate([f"P{k}" for k in range(14)] + [QK.IZLEME]):
        e = rng.normal(0.0004, 0.01, n)
        satir += [pd.DataFrame({"tarih": gun, "piyasa": pz, "sembol": f"S{i}", "kol": "ewma", "getiri": e}),
                  pd.DataFrame({"tarih": gun, "piyasa": pz, "sembol": f"S{i}", "kol": "ml21",
                                "getiri": e + (etki[i] if np.ndim(etki) else etki) + rng.normal(0, 0.0005, n)})]
    return pd.concat(satir, ignore_index=True)


def test_h2_kapisi_etkide_gecer_sifirda_gecmez():
    iyi = Y.h2_olcum(_R(0.0004))
    assert iyi["gecti"] and iyi["olumlu"] == 15 and iyi["z"] > Y.KAPI["z"] and iyi["mde"] > 0
    assert not Y.h2_olcum(_R(0.0))["gecti"]
    izl_ters = Y.h2_olcum(_R([0.0004] * 14 + [-0.0004]))
    assert izl_ters["z"] > Y.KAPI["z"] and izl_ters["olumlu"] == 14 and not izl_ters["gecti"]
    az = Y.h2_olcum(_R([0.002] * 4 + [-0.0002] * 10 + [0.002]))
    assert az["z"] > Y.KAPI["z"] and az["olumlu"] == 5 and az["izleme_ds"] > 0 and not az["gecti"]
    assert Y.h2_olcum(_R(0.0004), son=Y.KAYIT_GUNU)["n_piyasa"] == 0              # karar gününden sonrası sayılmaz


def _T(iyi: bool, n=120, seed=5, izleme_ters=False):
    rng = np.random.default_rng(seed)
    gun = pd.bdate_range(Y.KAYIT_GUNU + GUN, periods=n)
    T, rv = [], []
    for i, pz in enumerate([f"P{k}" for k in range(14)] + [QK.IZLEME]):
        r = np.exp(rng.normal(-9, 0.4, n))
        gurultu = r * np.exp(rng.normal(0, 0.8, n))
        ok = iyi != (izleme_ters and pz == QK.IZLEME)
        T.append(pd.DataFrame({"tarih": gun, "piyasa": pz, "sembol": f"S{i}",
                               "ewma_f": gurultu if ok else r, "f_ml21": r if ok else gurultu}))
        rv.append(pd.DataFrame({"tarih": gun, "piyasa": pz, "sembol": f"S{i}", "rv21": r}))
    return pd.concat(T, ignore_index=True), pd.concat(rv, ignore_index=True)


def test_h1_kapisi_iyi_ongoruyu_gecirir_kotuyu_gecirmez():
    iyi = Y.h1_olcum(*_T(True))
    assert iyi["gecti"] and iyi["olumlu"] == 15 and iyi["izleme_t"] < -Y.KAPI["z"]
    assert not Y.h1_olcum(*_T(False))["gecti"]
    izl = Y.h1_olcum(*_T(True, izleme_ters=True))
    assert izl["olumlu"] == 14 and izl["izleme_t"] > 0 and not izl["gecti"]


def test_hukum_karar_gunlerinden_once_vermez_h1_basarisizsa_aday_duser():
    h1_ok, h1_x, h2_ok = {"gecti": True}, {"gecti": False}, {"gecti": True}
    assert "ARA RAPOR" in Y.hukum(h1_ok, h2_ok, Y.H1_KARAR - GUN)[0]
    assert "DÜŞER" in Y.hukum(h1_x, h2_ok, Y.H1_KARAR)[0]
    assert any("H2 ARA" in x for x in Y.hukum(h1_ok, h2_ok, Y.H2_KARAR - GUN))
    assert "ADAY" in Y.hukum(h1_ok, h2_ok, Y.H2_KARAR)[-1]
    assert "GEÇMEDİ" in Y.hukum(h1_ok, {"gecti": False}, Y.H2_KARAR)[-1]


def test_onkayit_belgesi_kilitle_ayni():
    """Ön-kayıt belgesi kilitten sonra düzenlenirse test paketi kırmızıya döner (ek belge yazılmalı)."""
    kilit = json.loads(Y.KILIT.read_text(encoding="utf-8"))
    assert kilit["kayit_gunu"] == str(Y.KAYIT_GUNU.date())
    assert Y._sha(Y.ONKAYIT) == kilit["onkayit_sha256"]
    assert len(kilit["evren"]) > 100
    assert {pz for pz, _ in kilit["kanarya"]} == {pz for pz, _ in kilit["evren"]}
    assert set(kilit["kanarya_izi"]) == {f"{pz}/{s}" for pz, s in kilit["kanarya"]}
