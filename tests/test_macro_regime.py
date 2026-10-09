"""nihai.data.macro_regime testleri — TAMAMI ÇEVRİMDIŞI (`fred_data.get` monkeypatch).

İki test bu dosyanın varlık sebebidir:

* `test_net_likidite_birim_tuzagi_kapali` — RRPONTSYD MİLYAR, WALCL/WTREGEN MİLYON
  dolardır. Çevirmeyi unutan bir uygulama 2022 sonunda net likiditeyi 2,55 TRİLYON
  dolar fazla gösterirdi (%45,8 hata). Test kapalı formla kilitler.
* `test_yayin_gecikmesi_indeksi_ileri_kaydirir` — FRED'in `observation_date`'i verinin
  AİT OLDUĞU gündür, BİLİNDİĞİ gün değildir. Gecikmesiz kullanım ACI DERS #7'nin makro
  versiyonudur (model geleceği okur).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.data import macro_regime as mr

BUGUN = pd.Timestamp("2026-09-07")


def _seri(tarihler: list[str], degerler: list[float], ad: str) -> pd.Series:
    return pd.Series(degerler, index=pd.DatetimeIndex(tarihler), name=ad, dtype=float)


@pytest.fixture()
def sahte_fred(monkeypatch):
    """`fred_data.get`'i sözlükten besleyen sahte; kayıtsız seri → boş Series."""
    depo: dict[str, pd.Series] = {}

    def _get(series: str, **kwargs) -> pd.Series:
        return depo.get(series, pd.Series(dtype=float))

    monkeypatch.setattr(mr.fred_data, "get", _get)
    monkeypatch.setattr("nihai.data.alpaca_history.get_history", lambda *a, **k: pd.DataFrame())
    return depo


# ---------------------------------------------------------------------------
# 1) Madde 151 — birim tuzağı
# ---------------------------------------------------------------------------
def test_net_likidite_birim_tuzagi_kapali(sahte_fred):
    """RRP milyar → milyon çevrilmezse net likidite 1000 kat yanlış çıkarılır."""
    g = ["2026-09-01", "2026-09-02"]
    sahte_fred["WALCL"] = _seri(g, [6_000_000.0, 6_000_000.0], "WALCL")
    sahte_fred["WTREGEN"] = _seri(g, [800_000.0, 800_000.0], "WTREGEN")
    sahte_fred["RRPONTSYD"] = _seri(g, [2_500.0, 2_500.0], "RRPONTSYD")   # 2,5 TRİLYON $
    d = mr.net_liquidity(lag=False, today=BUGUN)
    # 6.000.000 − 800.000 − 2.500×1000 = 2.700.000 (milyon $)
    assert d["net_liquidity"].iloc[0] == pytest.approx(2_700_000.0)
    # Çevirmeyen naif hesap 5.197.500 verirdi — arada 2,4975 trilyon $ var.
    naif = 6_000_000.0 - 800_000.0 - 2_500.0
    assert abs(naif - d["net_liquidity"].iloc[0]) == pytest.approx(2_497_500.0)


def test_net_likidite_walcl_yoksa_bos_doner(sahte_fred):
    """Fail-soft: eksik girdide SIFIR değil BOŞ döner (sıfır bir ölçümdür)."""
    assert mr.net_liquidity(lag=False, today=BUGUN).empty


# ---------------------------------------------------------------------------
# 2) Yayın gecikmesi ve gelecek satırları
# ---------------------------------------------------------------------------
def test_yayin_gecikmesi_indeksi_ileri_kaydirir(sahte_fred):
    """SAHMREALTIME 35 gün gecikmelidir: 2026-08-01 gözlemi 2026-09-05'te bilinir."""
    sahte_fred["SAHMREALTIME"] = _seri(["2026-07-01", "2026-08-01"], [0.03, -0.07],
                                       "SAHMREALTIME")
    gecikmeli = mr.sahm_gap(lag=True, today=BUGUN)
    ham = mr.sahm_gap(lag=False, today=BUGUN)
    assert str(ham.index[-1].date()) == "2026-08-01"
    assert str(gecikmeli.index[-1].date()) == "2026-09-05"
    assert gecikmeli["level"].iloc[-1] == pytest.approx(-0.07)   # değer aynı, tarih farklı


def test_gelecek_satirlari_dusurulur(sahte_fred):
    """CBO projeksiyonları 2036'ya kadar satır taşır; bugünden sonrası panele girmez."""
    sahte_fred["NFCI"] = _seri(["2026-09-01", "2030-01-01"], [-0.5, -0.9], "NFCI")
    d = mr.financial_conditions(lag=False, today=BUGUN)
    assert len(d) == 1
    assert str(d.index[-1].date()) == "2026-09-01"


def test_lag_bayragi_panelin_attrs_inde_gorunur(sahte_fred):
    sahte_fred["NFCI"] = _seri(["2026-09-01"], [-0.5], "NFCI")
    assert mr.panel(lag=False, today=BUGUN).attrs["lag"] is False


# ---------------------------------------------------------------------------
# 3) Sınırlı ffill — aylık seri günlük gibi gösterilmez
# ---------------------------------------------------------------------------
def test_ffill_siniri_takvim_gunudur_satir_degil(sahte_fred):
    """Haftalık NFCI (bütçe 10 GÜN) 45 gün sonraki ızgara gününü DOLDURMAZ.

    `ffill(limit=N)` SATIR sayar: seyrek ızgarada tek bir temmuz gözlemi ağustos
    satırını doldurur ve sınır hiç ısırmaz. Bu test o davranışı kilitler.
    """
    sahte_fred["NFCI"] = _seri(["2026-07-01"], [-0.5], "NFCI")
    grid = pd.DatetimeIndex(["2026-07-05", "2026-08-15"])
    d = mr.panel(lag=False, today=BUGUN, grid=grid)
    assert not np.isnan(d["nfci"].iloc[0]), "5 gün sonrası doldurulmalıydı"
    assert np.isnan(d["nfci"].iloc[1]), "45 gün sonrası SINIRSIZ doldurulmuş"


# ---------------------------------------------------------------------------
# 4) Madde 169 — un-inversion bir OLAYdır, seviye değil
# ---------------------------------------------------------------------------
def test_uninversion_yalniz_gecis_barinda_yanar(sahte_fred):
    sahte_fred["T10Y2Y"] = _seri(
        ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"],
        [-0.20, -0.05, 0.10, 0.30, 0.40], "T10Y2Y")
    d = mr.curve_slope(lag=False, today=BUGUN)
    assert list(d["uninversion"]) == [0, 0, 1, 0, 0], "olay seviyeye dönüşmüş"


# ---------------------------------------------------------------------------
# 5) Madde 161 — Taylor kuralı kapalı formla
# ---------------------------------------------------------------------------
def test_taylor_kurali_kapali_form(sahte_fred):
    """π=3, u=4,0, u*=4,5 → i* = 2 + 3 + 0,5·(3−2) + 0,5·(2·(4,5−4,0)) = 6,0."""
    aylar = pd.date_range("2024-09-01", periods=13, freq="MS")
    # 12 ay önce 100, bugün 103 → 12 aylık enflasyon tam %3.
    pce = pd.Series([100.0] * 12 + [103.0], index=aylar, name="PCEPILFE")
    sahte_fred["PCEPILFE"] = pce
    sahte_fred["UNRATE"] = _seri(["2025-09-01"], [4.0], "UNRATE")
    sahte_fred["NROU"] = _seri(["2025-07-01"], [4.5], "NROU")
    sahte_fred["DFF"] = _seri(["2025-09-02"], [5.0], "DFF")
    d = mr.taylor_rule_gap(lag=False, today=BUGUN)
    assert d["inflation"].iloc[-1] == pytest.approx(3.0)
    assert d["taylor_rate"].iloc[-1] == pytest.approx(6.0)
    assert d["gap"].iloc[-1] == pytest.approx(-1.0)          # 5,0 − 6,0 = SIKI DEĞİL, gevşek


# ---------------------------------------------------------------------------
# 6) Makaslar — yarım makas ölçüm değildir
# ---------------------------------------------------------------------------
def test_makas_tek_bacakla_bos_doner(sahte_fred):
    sahte_fred["SOFR"] = _seri(["2026-09-01"], [4.3], "SOFR")
    assert mr.sofr_effr_basis(lag=False, today=BUGUN).empty


def test_makas_ortak_izgarada_hesaplanir(sahte_fred):
    g = ["2026-09-01", "2026-09-02"]
    sahte_fred["SOFR"] = _seri(g, [4.30, 4.35], "SOFR")
    sahte_fred["EFFR"] = _seri(g, [4.33, 4.33], "EFFR")
    d = mr.sofr_effr_basis(lag=False, today=BUGUN)
    assert list(np.round(d["spread"], 4)) == [-0.03, 0.02]


# ---------------------------------------------------------------------------
# 7) Panel — eksik gösterge kolonu düşürür, paneli düşürmez
# ---------------------------------------------------------------------------
def test_panel_eksik_gostergeyi_raporlar_ama_dusmez(sahte_fred):
    sahte_fred["NFCI"] = _seri(["2026-09-01", "2026-09-02"], [-0.5, -0.4], "NFCI")
    d = mr.panel(lag=False, today=BUGUN)
    assert "nfci" in d.columns and len(d) == 2
    assert "net_liq" in d.attrs["eksik"] and "sofr_effr" in d.attrs["eksik"]


def test_panel_hicbir_gosterge_yoksa_bos_ve_eksik_dolu(sahte_fred):
    d = mr.panel(lag=False, today=BUGUN)
    assert d.empty and len(d.attrs["eksik"]) == len(mr.PANEL_BUILDERS)


def test_summary_ok_false_veri_yokken(sahte_fred):
    out = mr.summary()
    assert out["ok"] is False and out["n"] == 0


def test_summary_son_degeri_tasir(sahte_fred):
    sahte_fred["NFCI"] = _seri(["2026-09-01", "2026-09-02"], [-0.5, -0.4], "NFCI")
    out = mr.summary()                       # summary() varsayılanı lag=True
    assert out["ok"] is True
    assert out["degerler"]["nfci"] == pytest.approx(-0.4)
    # NFCI yayın gecikmesi 5 gün: 2026-09-02 gözlemi 2026-09-07'de bilinir.
    assert out["as_of"] == "2026-09-07"


# ---------------------------------------------------------------------------
# 8) z-skoru geleceği sızdırmaz (projenin 1 numaralı kuralı)
# ---------------------------------------------------------------------------
def test_zscore_pencere_dolmadan_deger_uretmez():
    s = pd.Series(np.arange(10.0))
    z = mr._zscore(s, 5)
    assert z.iloc[:4].isna().all(), "pencere dolmadan z üretilmiş"
    assert not np.isnan(z.iloc[4])


def test_zscore_yuruyendir_tum_orneklem_degil():
    """z YALNIZ kendinden önceki pencereden hesaplanmalı — tüm örneklemden DEĞİL."""
    rng = np.random.default_rng(3)
    s = pd.Series(np.r_[rng.normal(0, 1, 40), rng.normal(0, 8, 40)])
    z = mr._zscore(s, 20)
    # 30. bar sakin rejimin içinde: yürüyen z yalnız 11-30 penceresini görür.
    pencere = s.iloc[11:31]
    beklenen = (s.iloc[30] - pencere.mean()) / pencere.std(ddof=1)
    assert z.iloc[30] == pytest.approx(beklenen)
    # Tüm örneklem std'si (fırtınayı da içerir) belirgin biçimde FARKLI bir z verirdi.
    tum = (s.iloc[30] - s.mean()) / s.std(ddof=1)
    assert abs(z.iloc[30] - tum) > 0.5


def test_uninversion_panelde_ileri_doldurulmaz(sahte_fred):
    """Olay kolonunun ffill bütçesi 0 GÜNdür; doldurulsa 200 gün süren seviyeye dönerdi."""
    sahte_fred["T10Y2Y"] = _seri(["2026-01-05", "2026-01-06"], [-0.05, 0.10], "T10Y2Y")
    grid = pd.DatetimeIndex(["2026-01-06", "2026-01-07", "2026-01-20"])
    d = mr.panel(lag=False, today=BUGUN, grid=grid)
    assert d["uninversion"].iloc[0] == 1
    assert np.isnan(d["uninversion"].iloc[1]) and np.isnan(d["uninversion"].iloc[2])
    # Seviye kolonu ise 5 günlük bütçeyle taşınır — olayla AYNI kural uygulanmıyor.
    assert d["curve_2s10s"].iloc[1] == pytest.approx(0.10)
    assert np.isnan(d["curve_2s10s"].iloc[2])
