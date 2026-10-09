"""
Tur-76 — SEC EDGAR köprüsü + temel skorlar. Üç soruyu birden sorar:

  * **ÖLÇÜM DOĞRU MU** — PIT kapısı (`filed <= as_of`) geçmişe sızdırmıyor mu, yeniden
    beyan doğru sürüme çözülüyor mu, etiket zinciri dönem başına doğru dolduruyor mu?
  * **EKSİK GİRDİ UYDURULUYOR MU** — ölçülemeyen sinyal 0 sayılıyor mu, eksik endeksle
    skor yine de basılıyor mu, medyansız kesitsel sinyal "geçti" sayılıyor mu?
  * **BAĞLANDI MI** — uçlar fail-soft mu, `fetch=False` gerçekten ağa çıkmıyor mu?

Ağ YOK: tüm girdiler sentetik `companyfacts` sözlükleridir (`distill` ile damıtılır).
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nihai.ai import fundamentals as fu
from nihai.data import edgar_facts as ef
from nihai.web.api_quant import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ---------------------------------------------------------------------------
# Sentetik companyfacts kurucusu
# ---------------------------------------------------------------------------
def _fact(end: str, val: float, filed: str, *, start: str | None = None,
          form: str = "10-K", fp: str = "FY") -> dict:
    d = {"end": end, "val": val, "filed": filed, "form": form, "fp": fp, "fy": end[:4]}
    if start:
        d["start"] = start
    return d


def _raw(us_gaap: dict) -> dict:
    return {"cik": 1, "entityName": "TEST", "facts": {"us-gaap": us_gaap}}


def _yillik(etiket: str, yillar: dict[str, float], *, filed_gun: str = "-11-15",
            donemsel: bool = True) -> dict:
    """`{yil_sonu: deger}` → tek etiketlik us-gaap bloğu (10-K, yıllık dönem)."""
    kayitlar = []
    for son, v in yillar.items():
        yil = int(son[:4])
        kayitlar.append(_fact(son, v, f"{yil}{filed_gun}",
                              start=(f"{yil - 1}-10-01" if donemsel else None)))
    return {etiket: {"units": {"USD": kayitlar}}}


# ---------------------------------------------------------------------------
# 1) PIT — `filed` BİLİNME tarihidir
# ---------------------------------------------------------------------------
def test_as_of_tarihinden_SONRA_dosyalanan_rakam_GORUNMEZ():
    """ACI DERS #7'nin şirket-temeli hâli: `end`e göre okuyan bir panel geleceği okur."""
    raw = _raw(_yillik("Assets", {"2024-09-30": 100.0, "2025-09-30": 200.0},
                       donemsel=False))
    d = ef.distill(raw)
    # 2025 kapanışı 2025-11-15'te dosyalanıyor; 2025-10-01'de HENÜZ BİLİNMİYOR.
    erken = ef.annual_frame(d, as_of="2025-10-01")
    assert list(erken.index.astype(str)) == ["2024-09-30"]
    gec = ef.annual_frame(d, as_of="2025-12-01")
    assert len(gec) == 2 and float(gec["varliklar"].iloc[-1]) == 200.0


def test_YENIDEN_BEYAN_as_of_gununde_BILINEN_surume_cozulur():
    """Aynı dönem sonu iki kez beyan edilirse `as_of`ta bilinen SON beyan kazanır."""
    raw = _raw({"Assets": {"units": {"USD": [
        _fact("2024-09-30", 100.0, "2024-11-15"),
        _fact("2024-09-30", 90.0, "2025-03-01"),        # düzeltme
    ]}}})
    d = ef.distill(raw)
    assert float(ef.annual_frame(d, as_of="2025-01-01")["varliklar"].iloc[-1]) == 100.0
    assert float(ef.annual_frame(d, as_of="2025-06-01")["varliklar"].iloc[-1]) == 90.0


# ---------------------------------------------------------------------------
# 2) ETİKET ZİNCİRİ — dönem başına doldurur (ölçülmüş AAPL temettü kusuru)
# ---------------------------------------------------------------------------
def test_zincir_DONEM_BASINA_doldurur_onde_olan_etiket_erken_bitse_bile():
    """`PaymentsOfDividendsCommonStock` 2017'de biter, `PaymentsOfDividends` sürer.

    Kalem başına TEK etiket seçen kural 2018 sonrasını NaN yapıyordu — sayı üretmemek
    de yanlış sayı üretmenin bir biçimidir."""
    onde = _yillik("PaymentsOfDividendsCommonStock", {"2016-09-30": 10.0, "2017-09-30": 11.0})
    arka = _yillik("PaymentsOfDividends", {"2018-09-30": 12.0, "2019-09-30": 13.0})
    d = ef.distill(_raw({**onde, **arka}))
    f = ef.annual_frame(d)
    assert len(f) == 4, "dört mali yılın dördü de dolmalı"
    assert f["temettu"].notna().all()
    assert float(f["temettu"].iloc[-1]) == 13.0
    # Karışım GİZLENMEZ: hangi etiketin kaç dönem doldurduğu taşınır.
    assert set(f.attrs["etiket"]["temettu"]) == {"PaymentsOfDividendsCommonStock",
                                                 "PaymentsOfDividends"}
    assert "temettu" in f.attrs["karisik"]


def test_KAPAK_TARIHLI_dei_kalemi_zincire_ALINMAZ():
    """`dei:EntityCommonStockSharesOutstanding` KAPAK tarihlidir, mali dönem sonu değil.

    Zincire konsaydı dönem-başına doldurma onu boş bir 'mali yıl' satırı yapardı
    (ölçüldü: AAPL 20 → 37 satır, 17'si tamamen NaN) ve `son_iki_yil` cari yıl diye o
    boş satırı döndürürdü."""
    assert "EntityCommonStockSharesOutstanding" not in ef.CONCEPTS["hisse_adedi"]
    raw = _raw(_yillik("Assets", {"2025-09-27": 100.0}, donemsel=False))
    raw["facts"]["dei"] = {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        _fact("2025-10-17", 15e9, "2025-10-31")]}}}
    f = ef.annual_frame(ef.distill(raw))
    assert list(f.index.astype(str)) == ["2025-09-27"], "kapak tarihi satır AÇMAMALI"


def test_10Q_satiri_yillik_cerceveye_GIRMEZ():
    raw = _raw({"Assets": {"units": {"USD": [
        _fact("2025-09-30", 100.0, "2025-11-15"),
        _fact("2025-06-30", 95.0, "2025-07-31", form="10-Q", fp="Q3"),
    ]}}})
    f = ef.annual_frame(ef.distill(raw))
    assert list(f.index.astype(str)) == ["2025-09-30"]


def test_yillik_OLMAYAN_donem_uzunlugu_elenir():
    """10-K içindeki ÇEYREK satırı (90 gün) yıllık kaleme karışmamalı."""
    raw = _raw({"Revenues": {"units": {"USD": [
        _fact("2025-09-30", 400.0, "2025-11-15", start="2024-10-01"),   # 364 gün
        _fact("2025-09-30", 100.0, "2025-11-15", start="2025-07-01"),   # 91 gün
    ]}}})
    f = ef.annual_frame(ef.distill(raw))
    assert float(f["hasilat"].iloc[-1]) == 400.0


# ---------------------------------------------------------------------------
# 3) SKORLAR — eksik girdi uydurulmaz
# ---------------------------------------------------------------------------
def _cari(**kw) -> dict:
    taban = {"_donem_sonu": "2025-09-30", "varliklar": 1000.0, "donen_varliklar": 400.0,
             "kisa_vadeli_yukumluluk": 200.0, "yukumlulukler": 500.0, "ozkaynak": 500.0,
             "hasilat": 900.0, "satis_maliyeti": 500.0, "net_kar": 120.0,
             "faaliyet_nakit": 150.0, "uzun_vadeli_borc": 300.0,
             "dagitilmamis_kar": 250.0, "faaliyet_kari": 160.0, "hisse_adedi": 100.0,
             "alacak": 80.0, "stok": 60.0, "borc_satici": 70.0, "duran_varlik": 300.0,
             "amortisman": 50.0, "sga": 100.0, "capex": 40.0, "temettu": 20.0,
             "geri_alim": 30.0}
    taban.update(kw)
    return taban


def _onceki(**kw) -> dict:
    o = _cari(_donem_sonu="2024-09-30", varliklar=900.0, net_kar=90.0, hasilat=800.0,
              faaliyet_nakit=110.0, uzun_vadeli_borc=320.0, donen_varliklar=340.0,
              kisa_vadeli_yukumluluk=190.0, hisse_adedi=105.0, satis_maliyeti=470.0,
              yukumlulukler=480.0, alacak=70.0, stok=55.0, borc_satici=60.0,
              duran_varlik=280.0, amortisman=45.0, sga=95.0)
    o.update(kw)
    return o


def test_piotroski_OLCULEMEYEN_sinyali_SIFIR_saymaz():
    """Eksik girdi 'başarısız sinyal' değildir — elenir ve `n_sinyal` düşer."""
    tam = fu.piotroski_f_score(_cari(), _onceki())
    assert tam["n_sinyal"] == 9 and tam["eksik"] == []
    eksik = fu.piotroski_f_score(_cari(faaliyet_nakit=None), _onceki())
    assert eksik["n_sinyal"] < 9
    assert "cfo_pozitif" in eksik["eksik"] and "tahakkuk_saglikli" in eksik["eksik"]
    # Eksik sinyal skoru ARTIRMAZ ama "0" olarak da SAYILMAZ.
    assert eksik["f_score"] <= tam["f_score"]
    assert eksik["sinyaller"]["cfo_pozitif"] is None


def test_piotroski_hesabi_ELLE_dogrulanan_sinyalleri_veriyor():
    p = fu.piotroski_f_score(_cari(), _onceki())
    s = p["sinyaller"]
    assert s["roa_pozitif"] is True                      # 120/1000 > 0
    assert s["cfo_pozitif"] is True
    assert s["roa_artti"] is True                        # 0,120 > 0,100
    assert s["tahakkuk_saglikli"] is True                # 150 > 120
    assert s["kaldirac_dustu"] is True                   # 0,30 <= 0,356
    assert s["seyrelme_yok"] is True                     # 100 <= 105
    assert s["devir_hizi_artti"] is True                 # 0,90 > 0,889


def test_altman_VARYANTI_ve_ESIGI_birlikte_tasir():
    """Klasik Z (piyasa değerli) ile Z' (defter değerli) EŞİKLERİ farklıdır."""
    ozel = fu.altman_z_score(_cari())
    klasik = fu.altman_z_score(_cari(), market_cap=5000.0)
    assert ozel["varyant"] == "ozel_1983" and klasik["varyant"] == "klasik_1968"
    assert klasik["z"] != ozel["z"], "piyasa değeri X4'ü değiştirmeli"
    assert klasik["bolge"] in ("GUVENLI", "GRI", "TEHLIKE")
    # Kapalı form doğrulaması (klasik katsayılar, elle).
    p = klasik["parcalar"]
    beklenen = (1.2 * p["x1_isletme_sermayesi"] + 1.4 * p["x2_dagitilmamis_kar"]
                + 3.3 * p["x3_ebit"] + 0.6 * p["x4_ozkaynak_yukumluluk"]
                + 0.999 * p["x5_devir"])
    assert klasik["z"] == pytest.approx(beklenen, rel=1e-12)


def test_altman_eksik_kalemde_z_URETMEZ():
    d = fu.altman_z_score(_cari(dagitilmamis_kar=None))
    assert d["z"] is None and "x2_dagitilmamis_kar" in d["eksik"]
    assert d["bolge"] is None


def test_beneish_EKSIK_endeksle_M_URETMEZ():
    """−1,78 eşiği TAM SEKİZ endeksle kalibre edilmiştir; altısıyla aynı eşik okunamaz."""
    tam = fu.beneish_m_score(_cari(), _onceki())
    assert tam["m_score"] is not None and tam["eksik"] == []
    eksik = fu.beneish_m_score(_cari(duran_varlik=None), _onceki())
    assert eksik["m_score"] is None and eksik["bayrak"] is None
    assert "aqi" in eksik["eksik"] and "depi" in eksik["eksik"]


def test_beneish_endeksleri_TERS_yonu_dogru_okuyor():
    """GMI ve DEPI ÖNCEKİ/CARİ oranıdır: bozulma > 1 vermelidir."""
    # brüt marj DÜŞÜYOR (cari maliyet artıyor) → GMI > 1
    d = fu.beneish_m_score(_cari(satis_maliyeti=700.0), _onceki())
    assert d["endeksler"]["gmi"] > 1.0
    # amortisman oranı DÜŞÜYOR (varlık ömrü uzatılmış) → DEPI > 1
    d2 = fu.beneish_m_score(_cari(amortisman=10.0), _onceki())
    assert d2["endeksler"]["depi"] > 1.0


def test_mohanram_MEDYANSIZ_kesitsel_sinyali_GECMIS_saymaz():
    """Medyanı 0 kabul etmek her kâr eden şirkete 'medyan üstü' derdi."""
    yok = fu.mohanram_g_score(_cari(), _onceki())
    assert yok["sinyaller"]["roa_medyan_ustu"] is None
    assert "roa_medyan_ustu" in yok["eksik"] and yok["medyan_verildi"] == []
    var = fu.mohanram_g_score(_cari(), _onceki(), {"roa": 0.05})
    assert var["sinyaller"]["roa_medyan_ustu"] is True   # 0,12 > 0,05
    assert var["n_sinyal"] > yok["n_sinyal"]


def test_ccc_DPO_paydasi_HASILAT_degil_satis_maliyeti():
    """Hasılatla hesaplamak DPO'yu brüt marj kadar KÜÇÜLTÜR — yaygın ve sessiz hata."""
    d = fu.cash_conversion_cycle(_cari(), _onceki())
    ort_borc = (70.0 + 60.0) / 2
    ort_smm = 500.0                                       # cari dönem satış maliyeti
    assert d["dpo"] == pytest.approx(365.0 * ort_borc / ort_smm, rel=1e-12)
    hasilatla = 365.0 * ort_borc / 900.0
    assert d["dpo"] > hasilatla, "payda satış maliyeti olmalı (daha büyük DPO)"


def test_dupont_carpimi_ROE_ile_TUTARLI():
    d = fu.dupont(_cari(), _onceki())
    assert d["tutarli"] is True
    assert d["carpim"] == pytest.approx(d["roe"], rel=1e-12)


def test_nakit_omru_POZITIF_nakit_akisinda_SONSUZ_sayi_basmaz():
    d = fu.net_cash_runway(_cari())                       # CFO 150 − capex 40 = +110
    assert d["yakim_yok"] is True and d["ay"] is None
    yakan = fu.net_cash_runway(_cari(faaliyet_nakit=-60.0, nakit=500.0,
                                     uzun_vadeli_borc=100.0))
    assert yakan["yakim_yok"] is False and yakan["ay"] == pytest.approx(12 * 400 / 100.0)


def test_emeklilik_acigi_EKSIK_bacakla_hesaplanmaz():
    """Eksik plan varlığını 0 saymak, fonlanmış planı fonlanmamış gibi gösterirdi."""
    d = fu.pension_gap(_cari(emeklilik_yukumlulugu=100.0))
    assert d["acik"] is None and d["fonlanma_orani"] is None


def test_serefiye_bozulma_YOKLUGU_ile_KALEM_yoklugu_ayri():
    assert fu.goodwill_risk(_cari())["bozulma_yazildi"] is None
    assert fu.goodwill_risk(_cari(serefiye_bozulmasi=0.0))["bozulma_yazildi"] is False
    assert fu.goodwill_risk(_cari(serefiye_bozulmasi=5.0))["bozulma_yazildi"] is True


def test_hissedar_getirisi_market_cap_YOKSA_oran_URETMEZ():
    d = fu.shareholder_yield(_cari())
    assert d["getiri"] is None and d["net_dagitim"] == 50.0   # 20 + 30 − 0


def test_rapor_TUM_skorlari_tek_sozlukte_topluyor():
    r = fu.rapor(_cari(), _onceki(), market_cap=5000.0, reel_faiz=0.02)
    for k in ("piotroski", "altman", "beneish", "mohanram", "tahakkuk", "dupont",
              "nakit_dongusu", "serefiye", "emeklilik", "arge", "nakit_omru",
              "hissedar_getirisi", "fcf", "kiralama", "kazanc_getirisi"):
        assert k in r, f"{k} raporda yok"
    assert r["kazanc_getirisi"]["risk_primi"] == pytest.approx(120.0 / 5000.0 - 0.02)


# ---------------------------------------------------------------------------
# 4) 8-K MADDE KODU süzgeci
# ---------------------------------------------------------------------------
def test_madde_kodu_TAM_PARCA_esleser():
    d = pd.DataFrame({"form": ["8-K"] * 4, "items": ["4.02", "4.021", "1.04,8.01",
                                                     "2.02,4.02"],
                      "filed": pd.to_datetime(["2026-01-01", "2026-01-02",
                                               "2026-01-03", "2026-01-04"])})
    v = ef.item_filings(d, "4.02")
    assert list(v.index) == [0, 3], "4.021 ve 1.04 eşleşmemeli"


# ---------------------------------------------------------------------------
# 5) UÇLAR — fail-soft ve ağa çıkmayan okuma
# ---------------------------------------------------------------------------
def test_fundamentals_ucu_onbelleksiz_sembolde_500_ATMAZ():
    r = client.get("/api/quant/fundamentals?symbol=ZZZZTEST")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is False and d["reason"] == "EDGAR_ONBELLEK_YOK"
    assert "ipucu" in d


def test_facts_fetch_False_iken_AG_DENENMEZ(monkeypatch, tmp_path):
    """Gözlem YAN ETKİYLE yapılır, istisnayla DEĞİL (Tur-74 dersi).

    Fail-soft bir yolda istisna fırlatan sahte, o istisna yutulduğu için hiçbir şey
    kanıtlamaz — guard'ı TAMAMEN silen bir mutasyon yine yeşil kalırdı."""
    denendi: list[str] = []
    monkeypatch.setattr(ef, "_edgar_get", lambda *a, **k: denendi.append("ag"))
    monkeypatch.setattr(ef, "CACHE_DIR", tmp_path)
    d = ef.facts("ZZZZ", fetch=False)
    assert denendi == [], "fetch=False iken ağ çağrısı YAPILMAMALI"
    assert d.empty and d.attrs["kaynak"] == "onbellek-yok"


def test_macro_ucu_fail_soft(monkeypatch):
    import nihai.data.macro_regime as mr
    monkeypatch.setattr(mr, "panel", lambda **kw: pd.DataFrame())
    d = client.get("/api/quant/macro").json()
    assert d["ok"] is False and d["reason"] == "MAKRO_ONBELLEK_YOK"


def test_macro_ucu_ONBELLEK_YALNIZ_okur(monkeypatch):
    """Uç 20 FRED serisini indirmemeli — `fetch=False` geçirdiği ÖLÇÜLÜR."""
    gecen: list[bool] = []

    def sahte_panel(**kw):
        gecen.append(kw.get("fetch"))
        return pd.DataFrame()

    import nihai.data.macro_regime as mr
    monkeypatch.setattr(mr, "panel", sahte_panel)
    client.get("/api/quant/macro")
    assert gecen == [False]


def test_fred_get_fetch_False_iken_AG_DENENMEZ(monkeypatch, tmp_path):
    from nihai.data import fred_data as fd
    denendi: list[str] = []
    monkeypatch.setattr(fd, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fd.requests, "get",
                        lambda *a, **k: denendi.append("ag"))
    s = fd.get("YOKSERI", fetch=False)
    assert denendi == [] and len(s) == 0
