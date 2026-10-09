"""Tur-97 — makine öğreniminin quant çekirdeğe hizmeti: ölçüm düzeneği sözleşmeleri + regresyon kapıları.

Kilitlenen: bayat (tatil) bar süzgeci (BULGU-97-5), yürüyen eğitimin nedenselliği, sabit-uzun taban ve
tek yönlü (sürüklenme öğrenen) model bayrağı (BULGU-97-6), σ kaynağı ön-kayıt kararları (vektörel +
motor), motor σ yamasının zincirlenmemesi ve eksik günde EWMA'ya düşmesi.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from research import coklu_piyasa as C
from research import piyasalar as P
from research import quant_kayma as QK
from research import quant_ml as QM


def _ohlc(close, high=None, low=None, bas="2020-01-01"):
    c = np.asarray(close, float)
    return pd.DataFrame({"open": c, "high": c * 1.01 if high is None else high,
                         "low": c * 0.99 if low is None else low, "close": c},
                        index=pd.bdate_range(bas, periods=c.size))


# ---------------------------------------------------------------------------------------------------
# BULGU-97-5 — bayat bar
# ---------------------------------------------------------------------------------------------------
def test_bayat_bar_yalniz_islem_olmayan_gunu_isaretler():
    c = np.array([100.0, 101.0, 101.0, 101.0, 102.0])
    hi = np.array([101.0, 102.0, 101.0, 101.5, 103.0])
    lo = np.array([99.0, 100.0, 101.0, 100.5, 101.0])
    m = P.bayat_bar_maskesi(_ohlc(c, hi, lo))
    # satır 2: aralık 0 + kapanış aynı → bayat; satır 3: kapanış aynı ama aralık var → İŞLEM GÜNÜ
    assert m.tolist() == [False, False, True, False, False]
    # aralık sıfır ama fiyat değişmiş (tek fiyattan işlem) → bayat DEĞİL; ilk satır asla bayat değil
    assert not P.bayat_bar_maskesi(_ohlc([100.0, 101.0], [100.0, 101.0], [100.0, 101.0])).any()
    assert P.bayat_bar_maskesi(pd.DataFrame(columns=["high", "low", "close"])).size == 0


def test_hazirla_bayat_bari_atar_ve_quant_ml_ikinci_kopya_tutmaz():
    assert "bayat_bar_maskesi" in inspect.getsource(C.hazirla)
    assert not hasattr(QM, "bayat_bar_maskesi")               # tek kaynak: piyasalar


# ---------------------------------------------------------------------------------------------------
# Yürüyen eğitim nedenselliği
# ---------------------------------------------------------------------------------------------------
def test_egitim_maskesi_etiketi_yil_basindan_sonra_bilinen_satiri_disari_atar():
    tarih = pd.DatetimeIndex(["2017-11-20", "2017-11-30", "2017-12-01", "2017-12-29"])
    m = QM.egitim_maskesi(tarih, 2018, 22)                      # 22 iş günü ≈ 30 takvim + 3 tampon
    assert m.tolist() == [True, False, False, False]
    assert QM.egitim_maskesi(pd.DatetimeIndex(["2017-12-27", "2017-12-28"]), 2018, 1).tolist() == [True, False]


def _sentetik_panel(n_gun=1800, n_sym=24, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parca = []
    for k in range(n_sym):
        idx = pd.bdate_range("2008-01-01", periods=n_gun)
        X = rng.normal(size=(n_gun, len(QM.A_OZ)))
        d = pd.DataFrame(X, columns=QM.A_OZ, index=idx)
        d["yA"] = np.clip(0.3 * X[:, 0] + rng.normal(size=n_gun), -3, 3)
        d["sembol"], d["piyasa"] = f"S{k}", "X"
        parca.append(d)
    return pd.concat(parca)


def test_trend_tahmini_gelecegin_etiketini_gormez():
    P0 = _sentetik_panel()
    a = QM.trend_tahmin(P0)
    P1 = P0.copy()
    P1.loc[P1.index.year >= 2013, "yA"] = -P1.loc[P1.index.year >= 2013, "yA"]   # geleceği boz
    b = QM.trend_tahmin(P1)
    yil12 = np.asarray(a.index.year == 2012)
    for k in ("p_lojistik", "p_zamanlama", "yhat_gbm"):
        assert np.isfinite(a.loc[yil12, k]).all()
        np.testing.assert_array_equal(a.loc[yil12, k].to_numpy(), b.loc[yil12, k].to_numpy())


def test_suruklenmeli_hedefte_lojistik_hep_uzun_zamanlama_dengeli():
    """BULGU-97-6'nın kendisi: pozitif sürüklenmeli hedefte sign(y) lojistiği sürüklenmeyi öğrenip her
    gün 'uzun' der; eğitim medyanına göre kurulan zamanlama hedefi bunu yapamaz."""
    P0 = _sentetik_panel(n_gun=1300, n_sym=16, seed=4)
    P0["yA"] = np.clip(P0["yA"] + 0.8, -3, 3)
    a = QM.trend_tahmin(P0)
    t = a[a["p_lojistik"].notna()]
    assert (t["p_lojistik"] > 0.5).mean() > 0.95
    assert 0.2 < (t["p_zamanlama"] > 0.5).mean() < 0.8


def test_ml_f_uygula_satirin_yilinin_modelini_kullanir():
    class Sabit:
        def __init__(self, v):
            self.v = v

        def predict(self, X):
            return np.full(len(X), self.v)

    idx = pd.DatetimeIndex(["2016-06-01", "2017-06-01", "2018-06-01"])
    Pn = pd.DataFrame(np.zeros((3, len(QM.B_OZ))), columns=QM.B_OZ, index=idx)
    f = QM.ml_f_uygula(Pn, {2016: (Sabit(0.0), 2.0), 2017: (Sabit(np.log(3.0)), 1.0)})
    assert f[0] == pytest.approx(2.0) and f[1] == pytest.approx(3.0) and np.isnan(f[2])


# ---------------------------------------------------------------------------------------------------
# BULGU-97-6 — sabit-uzun taban ve tek yönlü model
# ---------------------------------------------------------------------------------------------------
def test_trend_karnesi_sabit_tabani_ve_uzun_payini_tasir():
    assert "sabit" in QM.POLITIKALAR and "lojistik_zamanlama" in QM.POLITIKALAR
    g = pd.DataFrame({"kural_skor": [1.0, -1 / 3, 1 / 3], "p_lojistik": [0.6, 0.4, 0.51],
                      "p_zamanlama": [0.4, 0.6, 0.5], "yhat_gbm": [0.1, -0.2, 0.0]})
    assert QM._uzun(g, "sabit").tolist() == [True, True, True]
    assert QM._uzun(g, "kural").tolist() == [True, False, True]
    assert QM._uzun(g, "lojistik_zamanlama").tolist() == [False, True, False]
    src = inspect.getsource(QM.trend_degerlendir)
    assert '"fark_sabit"' in src and '"uzun_payi"' in src


def _ens(n=400, t0=50, seed=1):
    rng = np.random.default_rng(seed)
    return {"t0": t0, "n": n, "tarih": pd.bdate_range("2019-01-01", periods=n).to_numpy(),
            "ileri": rng.normal(0, 0.01, n), "y": (rng.random(n) > 0.5).astype(np.int8)}


def test_tek_yonlu_model_bayraklanir_ve_en_iyi_secimden_cikar():
    E = {"A": _ens(seed=1), "B": _ens(seed=2)}
    bh = pd.Series(np.zeros(400), index=pd.bdate_range("2019-01-01", periods=400))
    hep_uzun = {s: np.full(400, 0.55) + 0.01 * np.sin(np.arange(400)) for s in E}   # sabit DEĞİL
    karisik = {s: 0.5 + 0.1 * np.sign(np.sin(np.arange(400) / 7.0)) for s in E}
    d1 = C._ai_satiri(E, hep_uzun, bh, 252, 0.001)
    d2 = C._ai_satiri(E, karisik, bh, 252, 0.001)
    assert d1["tek_yonlu"] and d1["uzun_payi"] == 1.0
    assert not d2["tek_yonlu"] and 0.3 < d2["uzun_payi"] < 0.7
    assert 'not d.get("tek_yonlu")' in inspect.getsource(C.ozet_md)


# ---------------------------------------------------------------------------------------------------
# σ kaynağı — ön-kayıtlı kararlar
# ---------------------------------------------------------------------------------------------------
def _sigma_karne(t_kural, t_sabit, n=14):
    blok = {f"P{i}": {"kural_fark": {"t": t_kural}, "sabit_fark": {"t": t_sabit}} for i in range(n - 1)}
    blok["ABD"] = {"kural_fark": {"t": t_kural}, "sabit_fark": {"t": t_sabit}}
    return {"sigma": {p: blok for p in QM.PENCERELER}}


def test_vektorel_sigma_karari():
    assert QM.sigma_hukmu(_sigma_karne(2.0, 2.0))["baglanir"] is True
    assert QM.sigma_hukmu(_sigma_karne(2.0, -0.1))["baglanir"] is False        # sabit politikada kaybediyor
    k = _sigma_karne(2.0, 2.0)
    k["sigma"]["oos_2011_17"] = {**k["sigma"]["oos_2011_17"],
                                 "ABD": {"kural_fark": {"t": 1.0}, "sabit_fark": {"t": 2.0}}}
    assert QM.sigma_hukmu(k)["baglanir"] is False                              # ABD eşiği (1,645)
    k = _sigma_karne(-1.0, -1.0)
    for p in QM.PENCERELER:
        k["sigma"][p] = {**k["sigma"][p], "ABD": {"kural_fark": {"t": 3.0}, "sabit_fark": {"t": 3.0}}}
    h = QM.sigma_hukmu(k)
    assert h["baglanir"] is False and h["kosullar"]["test_2018_26/kural"]["olumlu_piyasa"] == 1


def test_motor_sigma_yamasi_zincirlenmez_eksik_gunde_ewmaya_duser():
    from nihai.signals import quant as Q
    from nihai.signals import strategy as S
    orj = S._SharedIndicators.quant_np
    QK._ORIJINAL.pop("quant_np", None)
    c = 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0.0005, 0.01, 400)))
    df = _ohlc(c)
    ml = pd.Series(0.05, index=df.index[:200])                  # son 200 gün ML yok
    try:
        for _ in range(3):                                      # BULGU-96-5: ardışık görevler
            with QK.ml_sigma(ml):
                skor, sig = S._SharedIndicators(df).quant_np()
            assert S._SharedIndicators.quant_np is orj
        s0, g0 = Q.seriler(c)
        np.testing.assert_array_equal(skor, s0)
        assert (sig[:200] == 0.05).all()
        np.testing.assert_array_equal(sig[200:], g0[200:])
    finally:
        S._SharedIndicators.quant_np = orj
        QK._ORIJINAL.pop("quant_np", None)


def test_motor_sigma_rapor_karari():
    src = inspect.getsource(QK.sigma_rapor)
    assert 'SIGMA_KARAR["min_piyasa"]' in src and 'SIGMA_KARAR["izleme_t"]' in src
    assert QK.SIGMA_KARAR == {"min_piyasa": 10, "izleme_t": 1.645}
    assert set(QK.SIGMA_PENCERE) == set(QM.PENCERELER)
