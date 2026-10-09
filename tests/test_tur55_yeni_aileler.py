"""
test_tur55_yeni_aileler.py — BACKLOG #10 (BAB × makro) ve #7 (mevsimsellik) kilitleri.

Bu iki ailede sessizce kenar uydurmanın yolları:

BAB × MAKRO
* **Ölçüt seçimi.** Düz fazlalık (`fwd_ret − mkt_ret`) BAB'ı yapısal olarak cezalandırır:
  betası 0,7 olan hisse yükselen piyasada tanımı gereği geride kalır. Literatürün iddiası
  ALFA'dır → `excess` BETA-AYARLI olmalı, ve düz sürüm de şemada durmalı.
* **Yön kuralı.** Kesitsel medyanın ÜSTÜ = yüksek beta = SHORT.
* **Makro doldurma SINIRLI.** NFCI haftalıktır; sınırsız ffill bir aylık eski değeri
  "bugünkü finansal koşullar" diye sunardı (Tur-51).

MEVSİMSELLİK
* **LOOK-AHEAD.** `f_seas` yalnız GEÇMİŞ yılların aynı takvim ayından hesaplanmalı; hedef
  ayın kendi getirisi girerse aile mükemmel görünür ve tamamı artefakttır.
* **feature-lag 1.** Karar günü ayın son işlem günü; giriş yeni ayın İLK barı.
* **Kesitsel yüzdelik** aynı karar gününün kesitinden alınmalı (gelecekten değil).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.edge_lab import build_babmacro_dataset as BM
from research.edge_lab import build_seasonality_dataset as SZ


def _px(n: int = 900, start: str = "2019-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    return pd.DataFrame({"date": idx, "close": close,
                         "volume": 1e6 * np.exp(rng.normal(0, 0.2, n))})


# --------------------------------------------------------------------------- BAB × makro
def test_puanlama_BETA_AYARLI_fazlaligi_kullanir():
    """Düz fazlalık negatifken beta-ayarlı pozitifse, puan beta-ayarlıyı görmeli."""
    df = pd.DataFrame({"f_is_short": [0.0], "excess": [0.010], "excess_duz": [-0.020]})
    out = BM.puanla(df.copy(), horizon=21, cost_pct=0.0, borrow_bps_yr=0.0)
    assert out["pnl"].iloc[0] == pytest.approx(0.010)
    assert out["label"].iloc[0] == 1


def test_semada_HEM_beta_ayarli_HEM_duz_fazlalik_var():
    """Ölçüt seçimi gizlenemez: iki kolon da şemada durmalı."""
    assert "excess" in BM.SCHEMA_COLS and "excess_duz" in BM.SCHEMA_COLS


def test_makro_doldurma_SINIRLI(monkeypatch):
    """Haftalık seri sınırsız ileri doldurulamaz (Tur-51)."""
    gunler = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=60))
    seyrek = pd.DataFrame({"T10Y2Y": [1.0], "NFCI": [-0.5]},
                          index=[pd.Timestamp("2021-01-04")])
    monkeypatch.setattr(BM.fred_data, "get_all", lambda *a, **k: seyrek)
    d = BM.makro_gunluk(gunler, "2021-01-01")
    dolu = int(d["NFCI"].notna().sum())
    assert dolu == BM.MACRO_FFILL_LIMIT + 1, f"sınırsız doldurma: {dolu} gün doldu"
    assert d["NFCI"].iloc[-1] != d["NFCI"].iloc[-1] or np.isnan(d["NFCI"].iloc[-1])


def test_yon_kurali_YUKSEK_beta_SHORT():
    px = _px()
    idx = pd.DatetimeIndex(px["date"])
    mkt = pd.Series(np.random.default_rng(1).normal(0.0004, 0.01, len(px)), index=idx)
    makro = pd.DataFrame({"T10Y2Y": 1.0, "T10Y2Y_chg": 0.1,
                          "NFCI": -0.5, "NFCI_chg": 0.01}, index=idx)
    for pct, beklenen in ((0.9, 1.0), (0.1, 0.0)):
        bp = pd.DataFrame({"T": pct}, index=idx)
        rows = BM.rows_for("T", px, mkt, makro, bp, horizon=21)
        assert rows and all(r["f_is_short"] == beklenen for r in rows)


# ------------------------------------------------------------------------- mevsimsellik
def _karar_gunleri(px: pd.DataFrame) -> pd.DatetimeIndex:
    return SZ._ay_sonu_gunleri(pd.DatetimeIndex(px["date"]))


def test_seas_YALNIZ_gecmis_yillarin_ayni_ayindan():
    """Hedef ayın KENDİ getirisi özelliğe giremez — girerse aile artefakt olur.

    Kanıt: hedef aydan sonraki tüm fiyatlar bozulur; `f_seas` DEĞİŞMEMELİ.
    """
    px = _px()
    karar = _karar_gunleri(px)
    a = SZ.sembol_satirlari("T", px, karar, horizon=21)
    assert a, "satır üretilmedi"
    r = a[len(a) // 2]
    kirli = px.copy()
    m = pd.DatetimeIndex(kirli["date"]) >= r["ts"]
    kirli.loc[m, "close"] = kirli.loc[m, "close"] * 5.0      # hedef ay ve sonrası BOZUK
    b = SZ.sembol_satirlari("T", kirli, karar, horizon=21)
    esle = [x for x in b if x["ts"] == r["ts"]]
    assert esle, "aynı karar günü ikinci koşuda yok"
    assert esle[0]["f_seas"] == pytest.approx(r["f_seas"]), "f_seas GELECEĞİ görüyor"


def test_giris_ay_sonundan_SONRAKI_bardir():
    px = _px()
    karar = _karar_gunleri(px)
    rows = SZ.sembol_satirlari("T", px, karar, horizon=21)
    idx = pd.DatetimeIndex(px["date"])
    for r in rows[:5]:
        i = int(idx.get_loc(r["karar_gunu"]))
        assert r["ts"] == idx[i + 1], "giriş ayın son gününe yapılmış"


def test_gecmis_gozlem_sayisi_alt_sinirin_altinda_SATIR_YOK():
    """En az MIN_GOZLEM yıl geçmiş yoksa ölçüm yapılmaz (uydurma yok)."""
    kisa = _px(n=300, start="2024-01-02")            # yalnız ~1 yıl geçmiş
    rows = SZ.sembol_satirlari("T", kisa, _karar_gunleri(kisa), horizon=21)
    assert rows == []


def test_kesitsel_yuzdelik_AYNI_karar_gununden():
    """Yüzdelik başka bir günün kesitinden alınırsa yön kuralı bulanır."""
    d = pd.DataFrame({
        "karar_gunu": ["2024-01-31"] * 3 + ["2024-02-29"] * 3,
        "f_seas": [0.01, 0.02, 0.03, 0.30, 0.20, 0.10],
    })
    pct = d.groupby("karar_gunu")["f_seas"].rank(pct=True)
    assert list(pct[:3]) == [pytest.approx(1 / 3), pytest.approx(2 / 3), pytest.approx(1.0)]
    assert list(pct[3:]) == [pytest.approx(1.0), pytest.approx(2 / 3), pytest.approx(1 / 3)]
