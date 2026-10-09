"""
Tur-76 — Ken French faktör köprüsü + faktör ayrıştırması (Madde 224/221).

Bu ailenin sorusu diğerlerinden FARKLIDIR: *"bu getirinin ne kadarı zaten BİLİNEN bir
risk primidir?"* Bu yüzden testler iki şeyi ayrı ayrı kilitler — AYRIŞTIRMANIN kendisi
(bilinen bir DGP geri kazanılıyor mu, sıfır alfa doğru okunuyor mu) ve ÖLÇÜM
KOŞULLARI (yüzde↔oran birimi, HAC gecikmesi, yetersiz kesişimde hüküm vermeme).

Ağ YOK: faktör serisi sentetik üretilir; ayrıştırıcı ona karşı sınanır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.data import factors as fx


def _faktor_df(n: int = 1500, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "mkt_rf": rng.normal(0.0003, 0.010, n),
        "smb": rng.normal(0.0000, 0.005, n),
        "hml": rng.normal(0.0000, 0.005, n),
        "rf": np.full(n, 0.00008),
    }, index=idx)


# ---------------------------------------------------------------------------
# 1) AYRIŞTIRMA — bilinen DGP geri kazanılmalı
# ---------------------------------------------------------------------------
def test_bilinen_DGP_alfa_ve_betalari_geri_kazaniyor():
    f = _faktor_df()
    rng = np.random.default_rng(3)
    alpha, b_mkt, b_smb = 0.0005, 1.30, 0.40
    y = (alpha + b_mkt * f["mkt_rf"] + b_smb * f["smb"] + f["rf"]
         + rng.normal(0, 0.002, len(f)))
    d = fx.factor_attribution(y, f)
    assert d is not None
    assert d["alpha"] == pytest.approx(alpha, abs=1e-4)
    assert d["yuklemeler"]["mkt_rf"] == pytest.approx(b_mkt, abs=0.02)
    assert d["yuklemeler"]["smb"] == pytest.approx(b_smb, abs=0.05)
    assert d["alpha_t"] > 2.0


def test_SIFIR_alfa_anlamsiz_t_veriyor():
    """Yalnız yakalayan değil, YAKALAMAYAN da olmalı — pozitif kontrolün karşılığı."""
    f = _faktor_df()
    rng = np.random.default_rng(5)
    y = 1.0 * f["mkt_rf"] + f["rf"] + rng.normal(0, 0.002, len(f))
    d = fx.factor_attribution(y, f)
    assert d is not None and abs(d["alpha_t"]) < 2.0


def test_KENDINE_regresyon_kimligi():
    """Piyasa fazlalığının kendisi regres edilirse β=1, α=0, R²=1 çıkmalı."""
    f = _faktor_df()
    d = fx.factor_attribution(f["mkt_rf"] + f["rf"], f)
    assert d is not None
    assert d["yuklemeler"]["mkt_rf"] == pytest.approx(1.0, abs=1e-9)
    assert abs(d["alpha"]) < 1e-12
    assert d["r2"] == pytest.approx(1.0, abs=1e-9)


def test_fazlalik_KAPALIYKEN_alfa_risksiz_faiz_kadar_kayiyor():
    """Ham getiriyi FAZLALIK faktörlere regres etmek sabiti risksiz faizle kirletir."""
    f = _faktor_df()
    y = 1.0 * f["mkt_rf"] + f["rf"]
    ile = fx.factor_attribution(y, f, fazlalik=True)
    olmadan = fx.factor_attribution(y, f, fazlalik=False)
    assert ile is not None and olmadan is not None
    assert olmadan["alpha"] - ile["alpha"] == pytest.approx(0.00008, abs=1e-6)


# ---------------------------------------------------------------------------
# 2) ÖLÇÜM KOŞULLARI
# ---------------------------------------------------------------------------
def test_YETERSIZ_kesisimde_HUKUM_VERMEZ():
    """Alfanın standart hatası 60 günün altında okunacak bir sayı değildir."""
    f = _faktor_df()
    kisa = pd.Series(np.zeros(30), index=f.index[:30])
    assert fx.factor_attribution(kisa, f) is None


def test_ORTAK_OLMAYAN_tarihler_sessizce_hizalanmaz():
    """Getiri serisi faktör penceresinin DIŞINDAYSA kesişim boştur → hüküm YOK."""
    f = _faktor_df(n=300)
    disarida = pd.Series(np.zeros(300),
                         index=pd.bdate_range("2035-01-01", periods=300))
    assert fx.factor_attribution(disarida, f) is None


def test_HAC_gecikmesi_CIKTIDA_tasiniyor():
    """Bir t değeri, hangi otokorelasyon düzeltmesiyle hesaplandığı yazılmadan okunamaz."""
    f = _faktor_df()
    d = fx.factor_attribution(f["mkt_rf"] + f["rf"], f)
    assert d is not None and d["lag"] == int(4 * (d["n"] / 100.0) ** (2.0 / 9.0))
    sabit = fx.factor_attribution(f["mkt_rf"] + f["rf"], f, lag=0)
    assert sabit is not None and sabit["lag"] == 0


def test_HAC_standart_hatasi_EKK_dan_FARKLI():
    """Otokorelasyonlu artıkta HAC ile sıradan EKK t'si AYRIŞMALIDIR.

    Ayrışmıyorsa `_hac_se` gerçekte Bartlett ağırlıklarını uygulamıyordur (ölü kod)."""
    f = _faktor_df(n=800)
    rng = np.random.default_rng(9)
    gurultu = np.zeros(len(f))
    for i in range(1, len(f)):                       # AR(1) artık
        gurultu[i] = 0.6 * gurultu[i - 1] + rng.normal(0, 0.002)
    y = 0.0002 + 1.0 * f["mkt_rf"] + f["rf"] + gurultu
    hac = fx.factor_attribution(y, f)
    ekk = fx.factor_attribution(y, f, lag=0)
    assert hac is not None and ekk is not None
    assert abs(hac["alpha_t"]) < abs(ekk["alpha_t"]), \
        "HAC otokorelasyonda t'yi KÜÇÜLTMELİ (EKK onu şişirir)"


def test_TEKIL_tasarim_matrisinde_None(monkeypatch):
    f = _faktor_df(n=200)
    f["kopya"] = f["mkt_rf"]                          # tam eşdoğrusal kolon
    y = f["mkt_rf"] + f["rf"]
    assert fx.factor_attribution(y, f) is None


# ---------------------------------------------------------------------------
# 3) AYRIŞTIRICI — yüzde ↔ oran ve dosya yapısı
# ---------------------------------------------------------------------------
def _zip_bytes(icerik: str) -> bytes:
    import io
    import zipfile
    tampon = io.BytesIO()
    with zipfile.ZipFile(tampon, "w") as z:
        z.writestr("F-F_Research_Data_Factors_daily.csv", icerik)
    return tampon.getvalue()


CSV = (
    "This file was created by using the 202607 CRSP database.\r\n"
    "\r\n"
    ",Mkt-RF,SMB,HML,RF\r\n"
    "20260101,   -0.67,    0.10,   -0.33,    0.01\r\n"
    "20260102,    1.25,   -0.20,    0.44,    0.01\r\n"
    "20260103,  -99.99,    0.00,    0.00,    0.01\r\n"
    "\r\n"
    "Copyright 2026 Eugene F. Fama and Kenneth R. French\r\n"
)


def test_ayristirici_YUZDEYI_ORANA_ceviriyor():
    """Ken French dosyası YÜZDEDİR; bölmeden regresyona sokmak alfayı 100 kat şişirir."""
    d = fx._ayristir(_zip_bytes(CSV), ("mkt_rf", "smb", "hml", "rf"))
    assert float(d["mkt_rf"].iloc[0]) == pytest.approx(-0.0067)
    assert float(d["hml"].iloc[1]) == pytest.approx(0.0044)


def test_ayristirici_EKSIK_isaretini_NaN_yapiyor():
    """−99.99 bir GÖZLEM değil EKSİK işaretidir; oran olarak −0,9999 saymak felakettir."""
    d = fx._ayristir(_zip_bytes(CSV), ("mkt_rf", "smb", "hml", "rf"))
    assert pd.isna(d["mkt_rf"].iloc[2])


def test_ayristirici_TELIF_ve_BASLIK_satirlarini_atiyor():
    d = fx._ayristir(_zip_bytes(CSV), ("mkt_rf", "smb", "hml", "rf"))
    assert len(d) == 3 and str(d.index[0].date()) == "2026-01-01"


def test_bilinmeyen_dataset_SESSIZ_gecmez():
    with pytest.raises(ValueError):
        fx.french_factors("yok_boyle_bir_set")


def test_french_factors_fetch_False_iken_AG_DENENMEZ(monkeypatch, tmp_path):
    denendi: list[str] = []
    monkeypatch.setattr(fx, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fx, "_indir", lambda dosya: denendi.append(dosya))
    d = fx.french_factors(fetch=False)
    assert denendi == [] and d.empty


# ---------------------------------------------------------------------------
# 4) ANAYASA — faktör köprüsü karar yoluna bağlı DEĞİL
# ---------------------------------------------------------------------------
def test_faktor_koprusu_KARAR_YOLUNA_bagli_DEGIL():
    import pathlib

    kok = pathlib.Path(__file__).resolve().parents[1] / "nihai"
    for alt in ("signals", "risk", "execution"):
        for p in (kok / alt).rglob("*.py"):
            assert "data.factors" not in p.read_text(encoding="utf-8"), \
                f"{p}: faktör köprüsü karar yoluna sızmış"


def test_portfoy_ucu_faktor_blogunu_ONBELLEK_YALNIZ_okuyor():
    """Uç Dartmouth'a çıkmamalı — `fetch=False` kaynakta DOĞRULANIR."""
    import inspect

    from nihai.web import api_quant
    kaynak = inspect.getsource(api_quant._faktor_bloku)
    assert "fetch=False" in kaynak
    assert "FREKANS_UYUSMUYOR" in kaynak, "günlük olmayan barda hüküm verilmemeli"
