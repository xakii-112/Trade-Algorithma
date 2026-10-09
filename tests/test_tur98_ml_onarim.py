"""Tur-98 — ML onarımı: σ ufuk eşleme (H1) + meta-etiketleme kapısı (H2) düzeneği ve ölçüm hijyeni.

Kilitlenen: sonuç dosyasının piyasa + sembolle adlanması (BULGU-98-1: 7 çakışan sembolde TA serisi çift
sayılıyordu), 21 günlük RV hedefi, rejim özelliklerinin aynı piyasa + aynı gün kesiti olması, yürüyen
eğitimin geleceği ve izleme evrenini görmemesi, meta kapının yalnız kuralın girişlerini elemesi, motor
yamasının zincirlenmemesi, Memmel z + örtüşen hedef NW t'si, ön-kayıtlı kapı + yozlaşma bayrağı ve
iki adayın canlıya BAĞLANMAMIŞ olması.
"""
from __future__ import annotations

import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import coklu_piyasa as C
from research import quant_kayma as QK
from research import quant_ml as QM
from research import quant_ml_motor as QMM


def _ohlc(close, bas="2020-01-01"):
    c = np.asarray(close, float)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c},
                        index=pd.bdate_range(bas, periods=c.size))


# ---------------------------------------------------------------------------------------------------
# BULGU-98-1 — sonuç dosyası piyasa + sembolle adlanır
# ---------------------------------------------------------------------------------------------------
def test_sonuc_dosyasi_piyasa_ve_sembolle_adlanir():
    assert QK.dosya("ABD", "SPY") != QK.dosya(QK.IZLEME, "SPY")
    assert QK.dosya("ABD", "SPY", "sigma_") != QK.dosya("ABD", "SPY")
    assert QK.dosya("ABD", "SPY").parent == QK.SONUC_DIR
    kaynak = inspect.getsource(QK)
    for eski in ('f"{C._guvenli(sym)}.pkl"', 'f"sigma_{C._guvenli(sym)}.pkl"', 'f"sigma_{C._guvenli(s)}.pkl"'):
        assert eski not in kaynak                                # regresyon kapısı: yalnız sembolle adlandırma
    assert QMM._m98("ABD", "SPY") != QMM._m98(QK.IZLEME, "SPY")


def test_rapor_cakisan_sembolde_ta_serisini_bir_kez_sayar(tmp_path, monkeypatch):
    """Aynı sembol ABD havuzunda VE izleme listesinde: ABD grubu event_breakout'u yalnız Tur-96
    önbelleğinden okumalı; izleme koşusunun kendi event_breakout'u ABD'ye karışmamalı."""
    monkeypatch.setattr(QK, "SONUC_DIR", tmp_path)
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-02", periods=300)

    def s():
        return pd.Series(rng.normal(0.0003, 0.01, idx.size), index=idx)

    def kos(islem):
        return {"s": s(), "islem": islem, "kazanma": 0.5, "pf": 1.0}

    for pz, kollar in (("ABD", {"quant_trend": kos(3)}),
                       (QK.IZLEME, {"quant_trend": kos(4), "event_breakout": kos(7)})):
        with open(QK.dosya(pz, "SPY"), "wb") as f:
            pickle.dump({"piyasa": pz, "sembol": "SPY", "kos": kollar, "bh": {"test": s()}}, f)
    ens = {"algo": {"event_breakout": {"r": s().to_numpy(), "tarih": idx.to_numpy(), "islem": 5}}}
    monkeypatch.setattr(C, "ens_oku", lambda sym: ens)
    karne = QK.rapor([("ABD", "SPY"), (QK.IZLEME, "SPY")], [])
    assert karne["piyasalar"]["ABD"]["test"]["event_breakout"]["islem"] == 5           # 5+7 değil
    assert karne["piyasalar"]["ABD"]["test"]["quant_trend"]["islem"] == 3
    assert karne["piyasalar"][QK.IZLEME]["test"]["event_breakout"]["islem"] == 7
    assert karne["piyasalar"][QK.IZLEME]["test"]["quant_trend"]["islem"] == 4


# ---------------------------------------------------------------------------------------------------
# H1 — 21 günlük RV hedefi
# ---------------------------------------------------------------------------------------------------
def test_rv21_hedefi_sonraki_21_gunun_ortalama_karesidir(monkeypatch):
    idx = pd.bdate_range("2020-01-01", periods=60)
    r2 = pd.Series(np.arange(60, dtype=float) ** 2, index=idx)
    r2.iloc[-1] = np.nan                                         # son satırın ertesi günü yok
    monkeypatch.setattr(QM, "panel", lambda sym, pz: pd.DataFrame({"r2_ileri": r2}, index=idx))
    p = QMM._panel("X", "Y")
    for t in (0, 10, 38):
        assert p["rv21"].iloc[t] == pytest.approx(r2.iloc[t: t + QMM.ILERI].mean())
    assert p["rv21"].iloc[-QMM.ILERI:].isna().all()             # 21 günü dolmayan satır etiketsiz
    assert np.isfinite(p["rv21"].iloc[: 60 - QMM.ILERI]).all()


# ---------------------------------------------------------------------------------------------------
# H2 — rejim özellikleri: aynı piyasa, aynı gün
# ---------------------------------------------------------------------------------------------------
def test_rejim_ozellikleri_ayni_piyasa_ayni_gun_kesitidir():
    t = pd.DatetimeIndex(["2020-01-02"] * 5 + ["2020-01-03"] * 2)
    P0 = pd.DataFrame({"piyasa": ["A", "A", "A", "B", "B", "A", "A"],
                       "kural_skor": [1.0, -1.0, np.nan, 1.0, 1.0, 1 / 3, 1 / 3],
                       "m63": [1.0, 3.0, 5.0, 10.0, 20.0, 0.0, 2.0],
                       "sig_rejim": [1.0, 1.0, 1.0, 2.0, 2.0, 0.5, 1.5]}, index=t)
    R = QMM.rejim_ekle(P0)
    assert R["gen_pay"].tolist()[:5] == [0.5, 0.5, 0.5, 1.0, 1.0]   # NaN skor paydaya girmez; B karışmaz
    assert R["gen_pay"].iloc[5:].tolist() == [1.0, 1.0]
    assert R["gen_m63"].tolist() == [3.0, 3.0, 3.0, 15.0, 15.0, 1.0, 1.0]
    assert R["gen_sig"].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]
    assert "_t" not in R and "_uzun" not in R


# ---------------------------------------------------------------------------------------------------
# Yürüyen eğitim: gelecek ve izleme evreni eğitime girmez
# ---------------------------------------------------------------------------------------------------
def _sentetik(n_gun=1500, n_sym=14, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parca = []
    for k in range(n_sym):
        idx = pd.bdate_range("2008-01-01", periods=n_gun)
        d = pd.DataFrame(rng.normal(size=(n_gun, len(QM.B_OZ))), columns=QM.B_OZ, index=idx)
        for c in QM.A_OZ + QMM.REJIM_OZ:
            d[c] = rng.normal(size=n_gun)
        d["kural_skor"] = rng.choice([-1.0, -1 / 3, 1 / 3, 1.0], size=n_gun)
        d["rv21"] = np.exp(0.5 * d["ln_rv22"] + rng.normal(scale=0.3, size=n_gun)) * 1e-4
        d["yA"] = np.clip(0.5 * d["m21"] + rng.normal(size=n_gun), -3, 3)
        d["sembol"], d["piyasa"] = f"S{k}", ("IZ" if k == n_sym - 1 else "X")
        parca.append(d)
    return pd.concat(parca)


@pytest.fixture
def hizli_model(monkeypatch):
    from sklearn.ensemble import HistGradientBoostingRegressor
    monkeypatch.setattr(QMM, "_gbm", lambda: HistGradientBoostingRegressor(max_iter=15, random_state=0))
    monkeypatch.setattr(QM, "TEST_YILLARI", (2012, 2013))


def test_sigma21_ve_meta_gelecegi_ve_izlemeyi_gormez(hizli_model):
    P0 = _sentetik()
    havuz = (P0["piyasa"] != "IZ").to_numpy()
    a = QMM.meta_tahmin(QMM.sigma21_tahmin(P0, havuz), havuz)
    P1 = P0.copy()
    gelecek = np.asarray(P1.index.year >= 2012)
    P1.loc[gelecek, "rv21"] = P1.loc[gelecek, "rv21"] * 50       # 2012 ve sonrası etiketleri boz
    P1.loc[gelecek, "yA"] = -P1.loc[gelecek, "yA"]
    izl = (P1["piyasa"] == "IZ").to_numpy()
    P1.loc[izl, "rv21"] = 1.0                                   # izleme etiketleri tamamen boz
    P1.loc[izl, "yA"] = 3.0
    b = QMM.meta_tahmin(QMM.sigma21_tahmin(P1, havuz), havuz)
    y12 = np.asarray(a.index.year == 2012)
    for k in ("f_ml21", "f_har21", "meta_yhat"):
        dolu = y12 & np.isfinite(a[k].to_numpy())
        assert dolu.sum() > 100
        np.testing.assert_array_equal(a.loc[dolu, k].to_numpy(), b.loc[dolu, k].to_numpy())
    assert np.isfinite(a.loc[izl & y12, "f_ml21"]).all()        # izleme TAHMİN alır, eğitime girmez


def test_meta_yalniz_kuralin_girislerini_eler(hizli_model):
    P0 = _sentetik()
    a = QMM.meta_tahmin(P0, np.ones(len(P0), dtype=bool))
    uzun = (a["kural_skor"] > 0).to_numpy()
    dolu = a["meta_ok"].notna().to_numpy()
    assert not (dolu & ~uzun).any()                              # skor ≤ 0 günü karar almaz
    assert set(a.loc[dolu, "meta_ok"].unique()) == {0.0, 1.0}
    np.testing.assert_array_equal(a.loc[dolu, "meta_ok"].to_numpy(),
                                  (a.loc[dolu, "meta_yhat"] > 0).astype(float).to_numpy())


# ---------------------------------------------------------------------------------------------------
# Motor yaması
# ---------------------------------------------------------------------------------------------------
def test_yama_skor_ve_sigmayi_degistirir_ve_zincirlenmez():
    from nihai.signals import quant as Q
    from nihai.signals import strategy as S
    orj = S._SharedIndicators.quant_np
    QK._ORIJINAL.pop("quant_np", None)
    c = 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0.0005, 0.012, 600)))
    df = _ohlc(c)
    s0, g0 = Q.seriler(c)
    sig = pd.Series(0.05, index=df.index[:300])                  # son 300 gün σ yok → EWMA
    meta = pd.Series(np.where(np.arange(600) % 2 == 0, 0.0, 1.0), index=df.index)
    try:
        for _ in range(3):                                       # BULGU-96-5: ardışık görevler
            with QMM.yama(sigma=sig, meta=meta):
                skor, g = S._SharedIndicators(df).quant_np()
            assert S._SharedIndicators.quant_np is orj
            with QK.ml_sigma(sig):                               # Tur-97 yaması aynı orijinali paylaşır
                pass
            assert S._SharedIndicators.quant_np is orj
        assert (g[:300] == 0.05).all()
        np.testing.assert_array_equal(g[300:], g0[300:])
        elenen = (meta.to_numpy() == 0) & (s0 > 0)
        assert elenen.sum() > 50
        assert (skor[elenen] == 0).all()
        np.testing.assert_array_equal(skor[~elenen], s0[~elenen])   # negatif/NaN skor ve onaylı gün aynı
    finally:
        S._SharedIndicators.quant_np = orj
        QK._ORIJINAL.pop("quant_np", None)


def test_kollar_on_kayitli_bes_kol():
    assert QMM.KOLLAR == ("ewma", "ml21", "har21", "ml1", "meta")
    idx = pd.bdate_range("2020-01-01", periods=5)
    s = pd.DataFrame({"f_ml21": 4e-4, "f_har21": 1e-4, "f_ml1": 9e-4, "meta_ok": 1.0}, index=idx)
    k = QMM.kol_argumanlari(s)
    assert k["ewma"] == {}
    assert k["ml21"]["sigma"].iloc[0] == pytest.approx(0.02)     # √(ortalama günlük varyans)
    assert k["har21"]["sigma"].iloc[0] == pytest.approx(0.01)
    assert set(k["meta"]) == {"meta"}


# ---------------------------------------------------------------------------------------------------
# Çıkarım
# ---------------------------------------------------------------------------------------------------
def test_memmel_z_isaret_simetri_ve_ozdes_seri():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=2000)
    b = pd.Series(rng.normal(0.0002, 0.01, idx.size), index=idx)
    a = b + pd.Series(rng.normal(0.0004, 0.002, idx.size), index=idx)
    zab, zba = QMM.memmel_z(a, b)["z"], QMM.memmel_z(b, a)["z"]
    assert zab > 1.645 and zab == pytest.approx(-zba, abs=1e-3)
    assert QMM.memmel_z(b, b)["z"] is None                       # ρ=1, V=0 → tanımsız, 0 uydurulmaz
    assert QMM.memmel_z(a.iloc[:30], b.iloc[:30])["z"] is None


def test_nw_t_ortusen_hedefte_gecikme_tabani():
    rng = np.random.default_rng(7)
    e = rng.normal(size=3000)
    d = 0.05 + np.convolve(e, np.ones(21) / 21, mode="valid")   # 21 günlük örtüşen hedef
    t_kisa, t_uzun = QMM.nw_t(d, 0), QMM.nw_t(d, QMM.ILERI)
    assert t_uzun < t_kisa                                       # varsayılan L örtüşmeyi eksik sayar
    assert QMM.nw_t(d[:50], 21) is None


# ---------------------------------------------------------------------------------------------------
# Ön-kayıtlı kapı + yozlaşma
# ---------------------------------------------------------------------------------------------------
def _karne(olumlu: int, izleme_z: float, n: int = 15) -> dict:
    piy = {}
    for k in range(n):
        ad = QK.IZLEME if k == n - 1 else f"P{k}"
        blok = {}
        for pen in QK.SIGMA_PENCERE:
            ds = 0.1 if k < olumlu else -0.1
            m = {"d_sharpe": ds, "memmel": {"z": izleme_z if ad == QK.IZLEME else 0.0}}
            blok[pen] = {"ewma": {}, "ml21": dict(m), "meta": dict(m), "har21": dict(m), "ml1": dict(m)}
        piy[ad] = blok
    return {"piyasalar": piy}


def test_hukum_on_kayitli_kapi():
    assert QMM.KARAR == {"min_piyasa": 10, "izleme_z": 1.645, "min_eleme": 0.05}
    h = QMM.hukum(_karne(15, 1.70))
    assert h["ml21"]["gecti"] and h["meta"]["gecti"] and h["ml21"]["kapili"] and not h["har21"]["kapili"]
    assert not QMM.hukum(_karne(15, 1.60))["ml21"]["gecti"]      # izleme anlamlılığı şart
    assert QMM.hukum(_karne(10, 1.70))["ml21"]["gecti"]          # sınır: 10 piyasa yeter
    assert not QMM.hukum(_karne(9, 1.70))["ml21"]["gecti"]


def test_meta_yozlasma_bayragi():
    k = {"hukum": QMM.hukum(_karne(15, 1.7)),
         "meta_eleme": {"oos_2011_17": {"_hepsi": 0.2}, "test_2018_26": {"_hepsi": 0.04}}}
    QMM.meta_yozlasma(k)
    assert k["hukum"]["meta"]["yozlasma"] is True and k["hukum"]["meta"]["gecti"] is False
    k2 = {"hukum": QMM.hukum(_karne(15, 1.7)),
          "meta_eleme": {"oos_2011_17": {"_hepsi": 0.2}, "test_2018_26": {"_hepsi": 0.15}}}
    QMM.meta_yozlasma(k2)
    assert k2["hukum"]["meta"]["yozlasma"] is False and k2["hukum"]["meta"]["gecti"] is True


def test_on_kayit_belgesi_ve_pencereler():
    doc = QMM.__doc__ or ""
    for ifade in ("ÖN-KAYIT", "H1", "H2", "Memmel", "Yozlaşma", "BİLGİ KOLLARI"):
        assert ifade in doc
    assert QMM.META_OZ == QM.A_OZ + ["kural_skor"] + QMM.REJIM_OZ


def test_tur98_adaylari_canliya_baglanmadi():
    """Ön-kayıtlı kapı iki adayı da geçirmedi (karne 2026-09-26): üretim kodu onları okumaz."""
    kok = Path(QMM.ROOT) / "nihai"
    for yol in kok.rglob("*.py"):
        metin = yol.read_text(encoding="utf-8", errors="ignore")
        assert "quant_ml_motor" not in metin and "f_ml21" not in metin, yol
    if QMM.KARNE.exists():
        h = json.loads(QMM.KARNE.read_text(encoding="utf-8"))["hukum"]
        assert h["ml21"]["gecti"] is False and h["meta"]["gecti"] is False


def test_teshis_kolu_yalniz_genisligi_olcekler():
    """Sonradan eklenen teşhis kolu (EWMA × sabit) kapısızdır ve yalnız σ düzeyini değiştirir."""
    from nihai.signals import quant as Q
    from nihai.signals import strategy as S
    orj = S._SharedIndicators.quant_np
    QK._ORIJINAL.pop("quant_np", None)
    c = 100 * np.exp(np.cumsum(np.random.default_rng(4).normal(0.0003, 0.01, 400)))
    s0, g0 = Q.seriler(c)
    try:
        with QMM.yama(**QMM.kol_argumanlari(pd.DataFrame(
                {"f_ml21": 1e-4, "f_har21": 1e-4, "f_ml1": 1e-4, "meta_ok": 1.0},
                index=_ohlc(c).index))["ewma_k"]):
            skor, g = S._SharedIndicators(_ohlc(c)).quant_np()
        assert S._SharedIndicators.quant_np is orj
    finally:
        S._SharedIndicators.quant_np = orj
        QK._ORIJINAL.pop("quant_np", None)
    np.testing.assert_array_equal(skor, s0)
    np.testing.assert_allclose(g, g0 * QMM.TESHIS_OLCEK)
    assert QMM.TESHIS_KOLLARI == ("ewma_k",) and "ewma_k" not in QMM.KOLLAR
    assert QMM.hukum(_karne(15, 1.7))["ewma_k"]["kapili"] is False


def test_izleme_indirmesi_havuz_onbellegine_dokunmaz(monkeypatch):
    """BULGU-98-6: izleme listesi indirmesi Tur-96 havuzundaki sembolü yeniden ÇEKMEZ (dondurulmuş girdi)."""
    from research import piyasalar as P
    cagri = []
    monkeypatch.setattr(QK, "izleme_listesi", lambda: ["SPY", "NVDA", "AAPL", "ONDS"])
    monkeypatch.setattr(P, "tum_semboller", lambda: [("ABD", "SPY"), ("ABD", "AAPL"), ("Kripto", "BTC-USD")])
    monkeypatch.setattr(P, "yukle", lambda s, fetch=False: cagri.append((s, fetch)) or _ohlc([1.0, 2.0]))
    assert QK.indir() == ["NVDA", "ONDS"]
    assert cagri == [("NVDA", True), ("ONDS", True)]
