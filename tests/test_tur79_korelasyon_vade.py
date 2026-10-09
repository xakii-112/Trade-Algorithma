"""Tur-79 — yol haritası §7.5: örtük korelasyon SERİSİ + ileri varyans eğrisi + SPDR sepetleri.

Üç soru:
1. **MATEMATİK TEK KAYNAKTAN MI?** Günlük seri (`implied_correlation_series`) Tur-73'ün tek-gün
   kimliğiyle (`implied_correlation`) her gün BİT-DÜZEYİNDE aynı ρ'yu vermeli; ileri varyans
   teleskop özdeşliğini tutmalı.
2. **ÖLÇÜM YOKLUĞU ÖLÇÜM GİBİ SUNULUYOR MU?** Kapsama kapısı (eksik ağır bileşen) ρ ÜRETMEMELİ;
   negatif ileri varyans KIRPILMAMALI; bir ucu yayımlanmamış vade yapısı satırı "contango" DEMEMELİ.
3. **OKUYUCU AĞA ÇIKIYOR MU?** `fetch=False` sepet ve CBOE okumaları ağa dokunmamalı.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from nihai.ai import options_math as om
from nihai.data import cboe_data, etf_holdings


# ---------------------------------------------------------------------------
# 1) implied_correlation_series ≡ implied_correlation (gün gün)
# ---------------------------------------------------------------------------
def _panel(seed: int = 3, T: int = 80, N: int = 12, nan_orani: float = 0.08):
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.2, 3.0, N)
    comp = rng.uniform(0.12, 0.65, (T, N))
    comp[rng.random((T, N)) < nan_orani] = np.nan
    idx = rng.uniform(0.10, 0.35, T)
    return idx, comp, w


def test_seri_TEK_GUN_kimligiyle_gun_gun_ayni():
    idx, comp, w = _panel()
    s = om.implied_correlation_series(idx, comp, w, min_kapsama=0.0)
    assert s is not None
    n_karsilastirilan = 0
    for t in range(idx.size):
        m = np.isfinite(comp[t])
        tek = om.implied_correlation(idx[t], w[m], comp[t, m])
        if tek is None:
            assert not s["olculdu"][t]
            continue
        assert s["olculdu"][t]
        assert s["rho_imp"][t] == pytest.approx(tek["rho_imp"], rel=1e-12, abs=1e-14)
        assert s["gecerli"][t] == tek["gecerli"]
        n_karsilastirilan += 1
    assert n_karsilastirilan >= 70


def test_bilinen_rho_GERI_KAZANILIR():
    rng = np.random.default_rng(11)
    T, N, rho = 30, 20, 0.35
    w = rng.uniform(0.5, 2.0, N)
    comp = rng.uniform(0.15, 0.6, (T, N))
    wn = w / w.sum()
    ws = comp * wn
    idx = np.sqrt(np.sum(ws**2, axis=1) + rho * (np.sum(ws, axis=1) ** 2 - np.sum(ws**2, axis=1)))
    s = om.implied_correlation_series(idx, comp, w)
    assert np.all(s["gecerli"])
    assert np.max(np.abs(s["rho_imp"] - rho)) < 1e-12


def test_kapsama_kapisi_AGIR_bileseni_eksik_gunde_rho_URETMEZ():
    idx, comp, w = _panel(nan_orani=0.0)
    agir = int(np.argmax(w))
    comp[5, agir] = np.nan
    pay = 1.0 - w[agir] / w.sum()
    s_gevsek = om.implied_correlation_series(idx, comp, w, min_kapsama=pay - 1e-9)
    s_sert = om.implied_correlation_series(idx, comp, w, min_kapsama=pay + 1e-9)
    assert s_gevsek["olculdu"][5] and np.isfinite(s_gevsek["rho_imp"][5])
    assert not s_sert["olculdu"][5] and np.isnan(s_sert["rho_imp"][5]) and not s_sert["gecerli"][5]
    assert s_sert["kapsama"][5] == pytest.approx(pay, rel=1e-12)
    assert s_sert["olculdu"][4]                      # komşu gün etkilenmez


@pytest.mark.parametrize("bozuk", [
    dict(index_vols=[0.2, 0.2], component_vols=np.ones((3, 2)), weights=[1, 1]),   # T uyuşmaz
    dict(index_vols=[0.2], component_vols=np.ones((1, 3)), weights=[1, 1]),         # N uyuşmaz
    dict(index_vols=[0.2], component_vols=np.ones((1, 1)), weights=[1]),            # N<2
    dict(index_vols=[0.2], component_vols=np.ones((1, 2)), weights=[1, -1]),        # negatif ağırlık
    dict(index_vols=[0.2], component_vols=np.ones((1, 2)), weights=[0, 0]),         # sıfır ağırlık
    dict(index_vols=[0.2], component_vols=np.ones(2), weights=[1, 1]),              # 1B bileşen
    dict(index_vols="x", component_vols=np.ones((1, 2)), weights=[1, 1]),
])
def test_seri_bozuk_girdide_None_istisna_yok(bozuk):
    assert om.implied_correlation_series(**bozuk) is None


def test_endeks_vol_yok_gunu_olculmez():
    idx, comp, w = _panel(nan_orani=0.0)
    idx[7] = np.nan
    idx[8] = 0.0
    s = om.implied_correlation_series(idx, comp, w)
    assert not s["olculdu"][7] and not s["olculdu"][8]


# ---------------------------------------------------------------------------
# 2) forward_variance_curve
# ---------------------------------------------------------------------------
def test_ileri_varyans_TELESKOP_ozdesligi():
    rng = np.random.default_rng(5)
    tn = np.array([9, 30, 93, 184, 366]) / 365.0
    vols = rng.uniform(0.12, 0.40, (50, 5))
    c = om.forward_variance_curve(tn, vols)
    dt = np.diff(tn)
    toplam = np.sum(c["fwd_var"] * dt[None, :], axis=1)
    assert np.max(np.abs(toplam - (c["toplam_varyans"][:, -1] - c["toplam_varyans"][:, 0]))) < 1e-14


def test_duz_vade_yapisinda_ileri_vol_SPOTA_esit():
    tn = [0.1, 0.25, 0.5, 1.0]
    c = om.forward_variance_curve(tn, [0.2, 0.2, 0.2, 0.2])
    assert np.allclose(c["fwd_vol"], 0.2, atol=1e-14)
    assert not c["negatif"].any()


def test_negatif_ileri_varyans_KIRPILMAZ_sayilir():
    # w(0.25)=0.3²·0.25=0.0225 > w(0.5)=0.2²·0.5=0.02 → takvim arbitrajı
    c = om.forward_variance_curve([0.25, 0.5], [0.30, 0.20])
    assert c["fwd_var"][0, 0] == pytest.approx((0.02 - 0.0225) / 0.25)
    assert c["negatif"][0, 0] and np.isnan(c["fwd_vol"][0, 0])


def test_nan_vol_yalniz_KOMSU_segmentleri_bozar():
    c = om.forward_variance_curve([0.1, 0.2, 0.3, 0.4], [[0.2, np.nan, 0.21, 0.22]])
    assert np.isnan(c["fwd_var"][0, 0]) and np.isnan(c["fwd_var"][0, 1])
    assert np.isfinite(c["fwd_var"][0, 2])
    assert not c["negatif"][0, 0]                    # ölçülmemiş segment NEGATİF SAYILMAZ


@pytest.mark.parametrize("tn,v", [([0.5, 0.25], [0.2, 0.2]), ([0.1], [0.2]), ([0.0, 0.1], [0.2, 0.2]),
                                  ([0.1, 0.2], [0.2, 0.2, 0.2]), ([0.1, np.nan], [0.2, 0.2])])
def test_ileri_varyans_gecersiz_vade_None(tn, v):
    assert om.forward_variance_curve(tn, v) is None


# ---------------------------------------------------------------------------
# 3) cboe_data — yeni seriler, vade günleri, ileri varyans, BULGU-79-2
# ---------------------------------------------------------------------------
def test_vade_gunleri_CBOE_METODOLOJISI():
    assert cboe_data.VADE_GUN == {"VIX9D": 9, "VIX": 30, "VIX3M": 93, "VIX6M": 184, "VIX1Y": 366}
    assert "VIX1D" not in cboe_data.VADE_GUN          # vadesi işlem dakikası, takvim günü değil
    for ad in ("VIX1D", "VIX1Y", "COR1M", "COR3M", "COR6M", "COR9M", "COR1Y", "DSPX"):
        assert ad in cboe_data.SERIES


def _yaz_onbellek(cdir, seriler: dict[str, pd.Series]):
    cdir.mkdir(parents=True, exist_ok=True)
    for ad, s in seriler.items():
        pd.DataFrame({"close": s}).rename_axis("date").to_csv(cdir / f"{ad}.csv")


def test_forward_variance_onbellekten_ELLE_hesapla_ayni_ve_FFILL_YOK(tmp_path, monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=6)
    vix = pd.Series([15.0, 16, 17, 18, 19, 20], index=idx)
    v3m = pd.Series([18.0, 18, 18, 18, 18, 18], index=idx)
    v6m = pd.Series([19.0, np.nan, 19, 19, 19, 19], index=idx)       # 2. gün yayımlanmamış
    _yaz_onbellek(tmp_path / "cboe", {"VIX": vix, "VIX3M": v3m, "VIX6M": v6m})
    monkeypatch.setattr(cboe_data, "CBOE_CACHE_DIR", tmp_path / "cboe")
    denendi: list = []
    gercek_oku = cboe_data.pd.read_csv

    def _kaydet(yol, *a, **k):
        if str(yol).startswith("http"):
            denendi.append(yol)
            raise OSError("ağ yok")
        return gercek_oku(yol, *a, **k)

    monkeypatch.setattr(cboe_data.pd, "read_csv", _kaydet)
    fv = cboe_data.forward_variance(fetch=False)
    assert not denendi
    assert fv.attrs["vade_gun"] == {"VIX": 30, "VIX3M": 93, "VIX6M": 184}
    t1, t2 = 30 / 365, 93 / 365
    beklenen = np.sqrt((0.18**2 * t2 - 0.15**2 * t1) / (t2 - t1))
    assert fv["fwd_VIX_VIX3M"].iloc[0] == pytest.approx(beklenen, rel=1e-12)
    assert np.isnan(fv["fwd_VIX3M_VIX6M"].iloc[1])                   # ffill YOK
    assert np.isnan(fv["neg_VIX3M_VIX6M"].iloc[1])                   # ölçülmedi ≠ negatif değil
    assert fv["neg_VIX_VIX3M"].iloc[0] == 0.0


def test_BULGU_79_2_bir_ucu_YAYIMLANMAMIS_gun_contango_DEMEZ(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=3)
    # VVIX o gün VAR (gerçek 2026-09-15 satırının şekli): satır düşmez, bayrak okunur.
    seriler = {"VIX": pd.DataFrame({"close": [15.0, 16.0, np.nan]}, index=idx),
               "VIX3M": pd.DataFrame({"close": [18.0, 14.0, 18.0]}, index=idx),
               "VVIX": pd.DataFrame({"close": [90.0, 91.0, 92.0]}, index=idx)}
    monkeypatch.setattr(cboe_data, "get_index", lambda n, **k: seriler.get(n, pd.DataFrame()))
    ts = cboe_data.term_structure()
    assert ts["backwardation"].iloc[0] == 0 and ts["backwardation"].iloc[1] == 1
    assert np.isnan(ts["backwardation"].iloc[2])


# ---------------------------------------------------------------------------
# 4) etf_holdings — xlsx ayrıştırma (stdlib), önbellek-yalnız, örtük korelasyon çerçevesi
# ---------------------------------------------------------------------------
def _xlsx(satirlar: list[list[str | float | None]]) -> bytes:
    """Asgari xlsx: tüm metinler paylaşılan dizge, sayılar hücre değeri."""
    paylasilan: list[str] = []
    xml_satirlar = []
    for i, s in enumerate(satirlar, start=1):
        hucreler = []
        for j, v in enumerate(s):
            ref = f"{chr(65 + j)}{i}"
            if v is None:
                continue
            if isinstance(v, str):
                paylasilan.append(v)
                hucreler.append(f'<c r="{ref}" t="s"><v>{len(paylasilan) - 1}</v></c>')
            else:
                hucreler.append(f'<c r="{ref}"><v>{v}</v></c>')
        xml_satirlar.append(f'<row r="{i}">{"".join(hucreler)}</row>')
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    sst = "".join(f"<si><t>{s}</t></si>" for s in paylasilan)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/sharedStrings.xml", f'<sst {ns} count="{len(paylasilan)}">{sst}</sst>')
        z.writestr("xl/worksheets/sheet1.xml", f'<worksheet {ns}><sheetData>{"".join(xml_satirlar)}</sheetData></worksheet>')
    return buf.getvalue()


_BASLIK = ["Name", "Ticker", "Identifier", "SEDOL", "Weight", "Sector", "Shares Held", "Local Currency"]
_ORNEK = [["Fund Name:", "SPDR Deneme"], ["Ticker Symbol:", "SPY"], ["Holdings:", "As of 14-Sep-2026"], [],
          _BASLIK,
          ["APPLE INC", "AAPL", "037833100", "2046251", 7.4, "-", 1.7e8, "USD"],
          ["NVIDIA CORP", "NVDA", "67066G104", "2379504", 7.8, "-", 2.9e8, "USD"],
          ["BERKSHIRE HATHAWAY INC CL B", "BRK.B", "084670702", "2073390", 1.4, "-", 3.0e7, "USD"],
          ["NAKIT", "-", None, None, 0.1, "-", None, "USD"]]


def test_parse_xlsx_agirliklar_ONDALIK_sirali_tarih_YEREL_AYARDAN_bagimsiz():
    d = etf_holdings.parse_xlsx(_xlsx(_ORNEK))
    assert d is not None
    assert list(d["ticker"]) == ["NVDA", "AAPL", "BRK.B"]            # ağırlığa göre azalan; '-' atlanır
    assert d["weight"].tolist() == pytest.approx([0.078, 0.074, 0.014])
    assert d.attrs["as_of"] == "2026-09-14"
    assert etf_holdings._as_of("As of 3-Mar-2025") == "2025-03-03"
    assert etf_holdings._as_of("As of 31-Foo-2025") is None


@pytest.mark.parametrize("raw", [b"", b"PK\x03\x04bozuk", _xlsx([["baslik", "yok"], ["a", "b"]])])
def test_parse_xlsx_bozuk_dosya_None(raw):
    assert etf_holdings.parse_xlsx(raw) is None


def test_holdings_fetch_False_AGA_CIKMAZ(tmp_path, monkeypatch):
    monkeypatch.setattr(etf_holdings, "CACHE_DIR", tmp_path / "yok")
    import nihai.net as net

    def _patla():
        raise AssertionError("fetch=False iken ağ istemcisi İSTENDİ")

    monkeypatch.setattr(net, "shared_client", _patla)
    assert etf_holdings.holdings("SPY", fetch=False).empty


def test_holdings_indirme_BASARISIZSA_bayat_onbellege_duser_ve_SOYLER(tmp_path, monkeypatch):
    monkeypatch.setattr(etf_holdings, "CACHE_DIR", tmp_path / "c")
    import nihai.net as net

    class _Yanit:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

    class _Istemci:
        def __init__(self, content):
            self.content = content

        def get(self, *a, **k):
            if self.content is None:
                raise OSError("ağ yok")
            return _Yanit(self.content)

    monkeypatch.setattr(net, "shared_client", lambda: _Istemci(_xlsx(_ORNEK)))
    ilk = etf_holdings.holdings("SPY", fetch=True, max_age_days=0)
    assert ilk.attrs["kaynak"] == "indirildi" and len(ilk) == 3
    monkeypatch.setattr(net, "shared_client", lambda: _Istemci(None))
    ikinci = etf_holdings.holdings("SPY", fetch=True, max_age_days=0)
    assert ikinci.attrs["kaynak"] == "bayat_onbellek" and ikinci.attrs["as_of"] == "2026-09-14"
    assert ikinci["ticker"].tolist() == ilk["ticker"].tolist()


def test_implied_correlation_frame_dolt_kopyasindan_ELLE_hesapla_ayni(tmp_path):
    rng = np.random.default_rng(2)
    gunler = pd.bdate_range("2025-01-01", periods=15)
    semboller = ["AAA", "BBB", "CCC", "DDD"]
    satirlar = []
    for g in gunler:
        satirlar.append({"date": g.date().isoformat(), "act_symbol": "SPY", "iv_current": 0.18})
        for s in semboller:
            satirlar.append({"date": g.date().isoformat(), "act_symbol": s,
                             "iv_current": float(rng.uniform(0.2, 0.5))})
    yol = tmp_path / "volatility_history__master.csv"
    pd.DataFrame(satirlar).to_csv(yol, index=False)
    h = pd.DataFrame({"ticker": semboller + ["ZZZ"], "weight": [0.4, 0.3, 0.2, 0.05, 0.05]})
    h.attrs["as_of"] = "2026-09-14"
    f = etf_holdings.implied_correlation_frame("SPY", n_top=5, iv_path=yol, holdings_df=h, min_kapsama=0.5)
    assert len(f) == 15 and f.attrs["eksik_semboller"] == ["ZZZ"] and f.attrs["agirlik_sabit"] is True
    tablo = pd.DataFrame(satirlar)
    g0 = tablo[tablo["date"] == gunler[0].date().isoformat()].set_index("act_symbol")["iv_current"]
    tek = om.implied_correlation(0.18, [0.4, 0.3, 0.2, 0.05], g0.reindex(semboller).to_numpy())
    assert f["rho_imp"].iloc[0] == pytest.approx(tek["rho_imp"], rel=1e-12)
    assert f["kapsama"].iloc[0] == pytest.approx(0.95 / 1.0)


def test_implied_correlation_frame_ETF_vol_yoksa_BOS(tmp_path):
    yol = tmp_path / "v.csv"
    pd.DataFrame([{"date": "2025-01-02", "act_symbol": "AAA", "iv_current": 0.3}]).to_csv(yol, index=False)
    h = pd.DataFrame({"ticker": ["AAA", "BBB"], "weight": [0.5, 0.5]})
    assert etf_holdings.implied_correlation_frame("SPY", iv_path=yol, holdings_df=h).empty
