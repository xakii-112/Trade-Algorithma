"""
fundamentals.py — şirket TEMEL analizi skorları (KARAR-DIŞI). Girdi: `data/edgar_facts`.

Listenin BÖLÜM 9'u (176-200) tek bir veri köprüsüne bağlıydı; köprü `edgar_facts`te
kuruldu, skorlar burada. **Hiçbiri bir kenar iddiası taşımaz** ve hiçbiri
`signals`/`risk`/`execution`'a bağlı DEĞİLDİR: listenin önerdiği eşikler ("F ≥ 7 ise
al", "M > −1,78 ise vetola") KODA YAZILMADI. Bir eşik bir kenar iddiasıdır ve
`research/edge_lab` kanıt kapısından geçmeden karara giremez (ACI DERS #3/#5).
Buradaki her fonksiyon SAYIYI üretir; hükmü operatör/araştırma verir.

## Üç sözleşme

**1. EKSİK GİRDİ UYDURULMAZ.** Her skor `eksik` listesi taşır; zorunlu bir kalem
yoksa skor `None`'dur, 0 değil. Bir F-Score'un 9 yerine 6 sinyalden hesaplanması
"zayıf şirket" demek değildir ve iki sayı aynı sütunda karşılaştırılamaz — bu yüzden
her skor `n_sinyal`/`n_olcum` ile birlikte döner (Tur-31: ölçüm yokluğu ölçüm gibi
raporlanamaz; Tur-33: bir oran, paydası okunmadan okunamaz).

**2. HER SKOR KENDİ VARYANTINI SÖYLER.** Altman Z'nin klasik biçimi piyasa değeri
ister; `market_cap` verilmezse Z' (Altman 1983, defter değerli özel şirket biçimi)
hesaplanır ve `varyant` alanı hangisinin koştuğunu YAZAR. İki varyantın eşikleri
FARKLIDIR; hangisinin çalıştığını söylemeden basılan bir Z, yanlış eşikle okunur.

**3. KESİTSEL ÖLÇÜT ÇAĞIRANDAN GELİR.** Mohanram G-Score'un beş sinyali "sektör
medyanının üstünde mi?" diye sorar. Medyanı bu modül UYDURMAZ: `medyanlar` verilmezse
o sinyaller ölçülmez (`None`) ve skor kalan sinyallerden kurulur. Sıfır kabul etmek
(yaygın kısayol) her zarar eden şirkete "medyan üstü" derdi.
"""
from __future__ import annotations

import math
from datetime import date
from itertools import pairwise
from statistics import median, variance
from typing import Any

# Beneish (1999) M-Score katsayıları — ön-kayıtlı, TARANMADI.
BENEISH_SABIT = -4.84
BENEISH_KATSAYI = {
    "dsri": 0.920, "gmi": 0.528, "aqi": 0.404, "sgi": 0.892,
    "depi": 0.115, "sgai": -0.172, "tata": 4.679, "lvgi": -0.327,
}
# Altman (1968) klasik ve (1983) Z' katsayıları.
ALTMAN_Z = (1.2, 1.4, 3.3, 0.6, 0.999)
ALTMAN_Z_OZEL = (0.717, 0.847, 3.107, 0.420, 0.998)


def _f(d: dict | None, ad: str) -> float | None:
    """Sözlükten sonlu float oku; yoksa/NaN ise None (0 ile ikame ETMEZ)."""
    if not d:
        return None
    v = d.get(ad)
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def _bol(pay: float | None, payda: float | None) -> float | None:
    """Payda yoksa/sıfırsa ya da oran TAŞIYORSA None — sıfıra bölmeyi büyük bir sayıyla
    ikame ETMEZ. Tur-78 (R3-B6): minik paydada iki oran da inf olup `inf > inf` yanlış
    sinyal üretiyordu."""
    if pay is None or payda is None or payda == 0:
        return None
    oran = pay / payda
    return oran if math.isfinite(oran) else None


def _ort(a: float | None, b: float | None) -> float | None:
    """İki dönemin ortalaması; biri yoksa VAR OLAN döner (tek dönemlik ortalama)."""
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2.0


def piotroski_f_score(cari: dict, onceki: dict) -> dict:
    """Madde 178 — Piotroski (2000) F-Score: 9 ikili sinyal (0-9).

    Sinyaller üç başlıkta: kârlılık (ROA>0 · CFO>0 · ΔROA>0 · CFO>net kâr) ·
    kaldıraç/likidite (Δkaldıraç<0 · Δcari oran>0 · yeni hisse ihracı YOK) ·
    verimlilik (Δbrüt marj>0 · Δvarlık devir hızı>0).

    Ölçülemeyen sinyal 0 SAYILMAZ, ELENİR: `f_score`/`n_sinyal` birlikte okunur.
    """
    va_c, va_o = _f(cari, "varliklar"), _f(onceki, "varliklar")
    ni_c, ni_o = _f(cari, "net_kar"), _f(onceki, "net_kar")
    cfo_c = _f(cari, "faaliyet_nakit")
    roa_c, roa_o = _bol(ni_c, va_c), _bol(ni_o, va_o)

    sinyaller: dict[str, bool | None] = {
        "roa_pozitif": None if roa_c is None else roa_c > 0,
        "cfo_pozitif": None if cfo_c is None else cfo_c > 0,
        "roa_artti": None if (roa_c is None or roa_o is None) else roa_c > roa_o,
        "tahakkuk_saglikli": None if (cfo_c is None or ni_c is None) else cfo_c > ni_c,
    }
    kald_c = _bol(_f(cari, "uzun_vadeli_borc"), va_c)
    kald_o = _bol(_f(onceki, "uzun_vadeli_borc"), va_o)
    sinyaller["kaldirac_dustu"] = (None if (kald_c is None or kald_o is None)
                                   else kald_c <= kald_o)
    cari_o_c = _bol(_f(cari, "donen_varliklar"), _f(cari, "kisa_vadeli_yukumluluk"))
    cari_o_o = _bol(_f(onceki, "donen_varliklar"), _f(onceki, "kisa_vadeli_yukumluluk"))
    sinyaller["cari_oran_artti"] = (None if (cari_o_c is None or cari_o_o is None)
                                    else cari_o_c > cari_o_o)
    adet_c, adet_o = _f(cari, "hisse_adedi"), _f(onceki, "hisse_adedi")
    sinyaller["seyrelme_yok"] = (None if (adet_c is None or adet_o is None)
                                 else adet_c <= adet_o)

    marj_c = _bol(_marj_pay(cari), _f(cari, "hasilat"))
    marj_o = _bol(_marj_pay(onceki), _f(onceki, "hasilat"))
    sinyaller["brut_marj_artti"] = (None if (marj_c is None or marj_o is None)
                                    else marj_c > marj_o)
    dev_c, dev_o = _bol(_f(cari, "hasilat"), va_c), _bol(_f(onceki, "hasilat"), va_o)
    sinyaller["devir_hizi_artti"] = (None if (dev_c is None or dev_o is None)
                                     else dev_c > dev_o)

    olculen = {k: v for k, v in sinyaller.items() if v is not None}
    return {
        "f_score": (sum(1 for v in olculen.values() if v) if olculen else None),
        "n_sinyal": len(olculen),
        "n_toplam": len(sinyaller),
        "sinyaller": sinyaller,
        "eksik": sorted(k for k, v in sinyaller.items() if v is None),
        "roa": roa_c,
    }


def _marj_pay(d: dict) -> float | None:
    """Brüt kâr: doğrudan varsa o, yoksa hasılat − satış maliyeti (ikisi de varsa)."""
    bk = _f(d, "brut_kar")
    if bk is not None:
        return bk
    h, sm = _f(d, "hasilat"), _f(d, "satis_maliyeti")
    return None if (h is None or sm is None) else h - sm


def altman_z_score(cari: dict, *, market_cap: float | None = None) -> dict:
    """Madde 186 — Altman Z (1968) / Z' (1983): iflas mesafesi.

    `market_cap` verilirse KLASİK Z (X4 = piyasa değeri / toplam yükümlülük), yoksa
    Z' (X4 = DEFTER özkaynağı / toplam yükümlülük). Eşikler farklıdır ve `varyant`
    alanı hangisinin koştuğunu söyler: klasik Z'de güvenli >2,99 / gri 1,81-2,99 /
    tehlike <1,81; Z''de güvenli >2,90 / gri 1,23-2,90 / tehlike <1,23.
    """
    ta = _f(cari, "varliklar")
    x1 = _bol(_fark(_f(cari, "donen_varliklar"), _f(cari, "kisa_vadeli_yukumluluk")), ta)
    x2 = _bol(_f(cari, "dagitilmamis_kar"), ta)
    x3 = _bol(_ebit(cari), ta)
    yuk = _f(cari, "yukumlulukler")
    klasik = market_cap is not None
    x4 = _bol(market_cap if klasik else _f(cari, "ozkaynak"), yuk)
    x5 = _bol(_f(cari, "hasilat"), ta)

    parcalar = {"x1_isletme_sermayesi": x1, "x2_dagitilmamis_kar": x2,
                "x3_ebit": x3, "x4_ozkaynak_yukumluluk": x4, "x5_devir": x5}
    eksik = sorted(k for k, v in parcalar.items() if v is None)
    kats = ALTMAN_Z if klasik else ALTMAN_Z_OZEL
    z: float | None = None
    if not eksik:                      # `eksik` boşsa parçaların HEPSİ float'tur
        z = sum(k * float(v) for k, v in zip(kats, parcalar.values(), strict=True)
                if v is not None)
    if z is None:
        bolge = None
    elif klasik:
        bolge = "GUVENLI" if z > 2.99 else ("GRI" if z >= 1.81 else "TEHLIKE")
    else:
        bolge = "GUVENLI" if z > 2.90 else ("GRI" if z >= 1.23 else "TEHLIKE")
    return {"z": z, "varyant": "klasik_1968" if klasik else "ozel_1983",
            "bolge": bolge, "parcalar": parcalar, "eksik": eksik}


def _fark(a: float | None, b: float | None) -> float | None:
    return None if (a is None or b is None) else a - b


def _ebit(d: dict) -> float | None:
    """FVÖK: faaliyet kârı varsa o; yoksa vergi öncesi kâr + faiz gideri."""
    fk = _f(d, "faaliyet_kari")
    if fk is not None:
        return fk
    vok, faiz = _f(d, "vergi_oncesi_kar"), _f(d, "faiz_gideri")
    return None if vok is None else vok + (faiz or 0.0)


def beneish_m_score(cari: dict, onceki: dict) -> dict:
    """Madde 177 — Beneish (1999) M-Score: kazanç manipülasyonu olasılık göstergesi.

    Sekiz endeks (DSRI · GMI · AQI · SGI · DEPI · SGAI · LVGI · TATA) ve ön-kayıtlı
    katsayılar. **Sekizinin HEPSİ ölçülemezse skor None** — Beneish'in eşiği (−1,78)
    tam sekiz endeksle kalibre edilmiştir; altı endeksten kurulan bir M ile aynı eşiği
    kullanmak kalibrasyonu sessizce bozardı.
    """
    e: dict[str, float | None] = {}
    e["dsri"] = _oran_orani(_f(cari, "alacak"), _f(cari, "hasilat"),
                            _f(onceki, "alacak"), _f(onceki, "hasilat"))
    mc = _bol(_marj_pay(cari), _f(cari, "hasilat"))
    mo = _bol(_marj_pay(onceki), _f(onceki, "hasilat"))
    e["gmi"] = _bol(mo, mc)                       # GMI: ÖNCEKİ / CARİ (bozulma > 1)
    e["aqi"] = _oran_orani(_yumusak_varlik(cari), _f(cari, "varliklar"),
                           _yumusak_varlik(onceki), _f(onceki, "varliklar"))
    e["sgi"] = _bol(_f(cari, "hasilat"), _f(onceki, "hasilat"))
    dep_c = _amortisman_orani(cari)
    dep_o = _amortisman_orani(onceki)
    e["depi"] = _bol(dep_o, dep_c)                # DEPI: ÖNCEKİ / CARİ
    e["sgai"] = _oran_orani(_f(cari, "sga"), _f(cari, "hasilat"),
                            _f(onceki, "sga"), _f(onceki, "hasilat"))
    e["lvgi"] = _oran_orani(_f(cari, "yukumlulukler"), _f(cari, "varliklar"),
                            _f(onceki, "yukumlulukler"), _f(onceki, "varliklar"))
    e["tata"] = _bol(_fark(_f(cari, "net_kar"), _f(cari, "faaliyet_nakit")),
                     _f(cari, "varliklar"))

    eksik = sorted(k for k, v in e.items() if v is None)
    m: float | None = None
    if not eksik:                      # sekiz endeksin HEPSİ ölçüldüyse
        m = BENEISH_SABIT + sum(BENEISH_KATSAYI[k] * float(v)
                                for k, v in e.items() if v is not None)
    return {"m_score": m, "endeksler": e, "eksik": eksik,
            "esik": -1.78, "bayrak": (None if m is None else m > -1.78)}


def _oran_orani(a1: float | None, b1: float | None,
                a0: float | None, b0: float | None) -> float | None:
    """(a1/b1) / (a0/b0) — Beneish endekslerinin ortak biçimi."""
    return _bol(_bol(a1, b1), _bol(a0, b0))


def _yumusak_varlik(d: dict) -> float | None:
    """AQI payı: toplam varlık − dönen varlık − maddi duran varlık."""
    ta, dv, ddv = _f(d, "varliklar"), _f(d, "donen_varliklar"), _f(d, "duran_varlik")
    if ta is None or dv is None or ddv is None:
        return None
    return ta - dv - ddv


def _amortisman_orani(d: dict) -> float | None:
    """Amortisman / (amortisman + net maddi duran varlık) — DEPI'nin iki bacağı."""
    am, ddv = _f(d, "amortisman"), _f(d, "duran_varlik")
    if am is None or ddv is None or (am + ddv) == 0:
        return None
    return am / (am + ddv)


def mohanram_metrics(history: list[dict] | None) -> dict[str, float | None]:
    """Yıllık G bileşenleri; en çok 5 yıl, varyans için en az 3 ölçüm.

    Mohanram'ın çeyreklik/sektörel deneyinin YILLIK vekilidir. Atlanan mali yıl
    veya eksik kalem varyans ölçümünü durdurur; SG&A reklam yerine geçirilmez.
    """
    fields = {"roa": "net_kar", "cfo_varlik": "faaliyet_nakit", "arge_varlik": "arge",
              "capex_varlik": "capex", "reklam_varlik": "reklam"}
    out: dict[str, float | None] = dict.fromkeys((*fields, "roa_varyans", "buyume_varyans"))
    if not isinstance(history, list) or not history or not all(isinstance(d, dict) for d in history):
        return out
    current = history[-1]
    assets = _f(current, "varliklar")
    if assets is not None and assets > 0:
        for key, field in fields.items():
            value = _bol(_f(current, field), assets)
            out[key] = value if value is not None and math.isfinite(value) else None
    rows = history[-6:]
    try:
        dates = [date.fromisoformat(d["_donem_sonu"]) for d in rows]
        if any(not 300 <= (b - a).days <= 400 for a, b in pairwise(dates)):
            return out
    except (KeyError, TypeError, ValueError):
        return out
    roa = [_bol(_f(d, "net_kar"), _f(d, "varliklar"))
           if (_f(d, "varliklar") or 0) > 0 else None for d in rows[-5:]]
    sales = [_f(d, "hasilat") for d in rows]
    growth = [None if a is None or a <= 0 or b is None else b / a - 1
              for a, b in pairwise(sales)]
    for key, values in (("roa_varyans", roa), ("buyume_varyans", growth)):
        valid = [v for v in values if v is not None and math.isfinite(v)]
        if len(valid) >= 3 and len(valid) == len(values):
            value = variance(valid)
            out[key] = value if math.isfinite(value) else None
    return out


def universe_medians(histories: dict[str, list[dict]] | None) -> dict:
    """Saf kesitsel özet; her kalem için en az 3 şirket, aksi halde None.

    Tarih/PIT süzmesini veri köprüsü yapar. Bu evren medyanıdır; sektör medyanı
    veya düşük-defter/piyasa-değeri filtresi uygulandığı iddia edilmez.
    """
    metrics = [mohanram_metrics(rows) for rows in (histories or {}).values()]
    values = {key: [value for m in metrics if (value := m[key]) is not None]
              for key in mohanram_metrics(None)}
    return {"medyanlar": {key: median(v) if len(v) >= 3 else None for key, v in values.items()},
            "n_kalem": {key: len(v) for key, v in values.items()}}


def mohanram_g_score(cari: dict, onceki: dict,
                     medyanlar: dict[str, float | None] | None = None, *,
                     history: list[dict] | None = None) -> dict:
    """Madde 188 — Mohanram (2005) G-Score: BÜYÜME/düşük-defter şirketleri için 8 sinyal.

    Beş sinyal SEKTÖR MEDYANINA görelidir (`medyanlar`: `roa` · `cfo_varlik` ·
    `arge_varlik` · `capex_varlik` · `reklam_varlik`). Medyan verilmezse o sinyaller
    ÖLÇÜLMEZ — sıfırla kıyaslamak (yaygın kısayol) her kâr eden şirkete "medyan üstü"
    derdi ve skoru kesitsel bilgi taşımadan şişirirdi.
    """
    med = medyanlar or {}
    va = _f(cari, "varliklar")
    ni, cfo = _f(cari, "net_kar"), _f(cari, "faaliyet_nakit")
    roa = _bol(ni, va)
    cfo_a = _bol(cfo, va)
    s: dict[str, bool | None] = {
        "roa_medyan_ustu": _ustunde(roa, _f(med, "roa")),
        "cfo_medyan_ustu": _ustunde(cfo_a, _f(med, "cfo_varlik")),
        # CFO/VA > NI/VA ⟺ CFO > NI (VA > 0): oranları kıyaslamak taşmaya ve negatif paydada
        # işaret dönmesine açıktı (Tur-78, R3-B6); ölçü doğrudan tutarlardan.
        "tahakkuk_saglikli": (None if cfo is None or ni is None or va is None or va <= 0 else cfo > ni),
        "arge_medyan_ustu": _ustunde(_bol(_f(cari, "arge"), va), _f(med, "arge_varlik")),
        "capex_medyan_ustu": _ustunde(_bol(_f(cari, "capex"), va), _f(med, "capex_varlik")),
        "reklam_medyan_ustu": _ustunde(_bol(_f(cari, "reklam"), va), _f(med, "reklam_varlik")),
    }
    metrics = mohanram_metrics(history)
    s["kazanc_oynakligi_dusuk"] = _ustunde(_f(med, "roa_varyans"), metrics["roa_varyans"])
    s["hasilat_oynakligi_dusuk"] = _ustunde(_f(med, "buyume_varyans"), metrics["buyume_varyans"])
    olculen = {k: v for k, v in s.items() if v is not None}
    return {"g_score": (sum(1 for v in olculen.values() if v) if olculen else None),
            "n_sinyal": len(olculen), "n_toplam": len(s), "sinyaller": s,
            "eksik": sorted(k for k, v in s.items() if v is None),
            "medyan_verildi": sorted(k for k in med if _f(med, k) is not None),
            "varyant": "yillik_evren_vekili", "oynaklik": metrics}


def _ustunde(x: float | None, medyan: float | None) -> bool | None:
    return (None if x is None or medyan is None or not math.isfinite(x) or not math.isfinite(medyan)
            else x > medyan)


def accruals_ratio(cari: dict, onceki: dict) -> dict:
    """Madde 187 — Sloan (1996) tahakkuk anomalisi: (net kâr − faaliyet nakdi) / ort. varlık.

    Yüksek pozitif tahakkuk = kârın nakde dönmeyen kısmı büyük. Bu bir SKORDUR,
    olasılık değildir (Tur-47: bir skora Brier uygulanmaz).
    """
    ort_va = _ort(_f(cari, "varliklar"), _f(onceki, "varliklar"))
    ni, cfo = _f(cari, "net_kar"), _f(cari, "faaliyet_nakit")
    return {"tahakkuk": _bol(_fark(ni, cfo), ort_va),
            "net_kar": ni, "faaliyet_nakit": cfo, "ortalama_varlik": ort_va,
            "eksik": [k for k, v in (("net_kar", ni), ("faaliyet_nakit", cfo),
                                     ("varliklar", ort_va)) if v is None]}


def dupont(cari: dict, onceki: dict | None = None) -> dict:
    """Madde 196 — DuPont ayrıştırması: ROE = net marj × varlık devri × kaldıraç.

    Özkaynak ve varlık ORTALAMA alınır (dönem sonu tek noktası devri şişirir).
    Çarpımın ROE ile tutarlılığı `tutarli` alanında ÖLÇÜLÜR, varsayılmaz.
    """
    ort_va = _ort(_f(cari, "varliklar"), _f(onceki, "varliklar") if onceki else None)
    ort_oz = _ort(_f(cari, "ozkaynak"), _f(onceki, "ozkaynak") if onceki else None)
    ni, hs = _f(cari, "net_kar"), _f(cari, "hasilat")
    marj, devir, kald = _bol(ni, hs), _bol(hs, ort_va), _bol(ort_va, ort_oz)
    roe = _bol(ni, ort_oz)
    carpim = (None if (marj is None or devir is None or kald is None)
              else marj * devir * kald)
    tutarli = (None if (carpim is None or roe is None)
               else abs(carpim - roe) <= 1e-9 * max(1.0, abs(roe)))
    return {"roe": roe, "net_marj": marj, "varlik_devri": devir, "kaldirac": kald,
            "carpim": carpim, "tutarli": tutarli,
            "eksik": [k for k, v in (("net_marj", marj), ("varlik_devri", devir),
                                     ("kaldirac", kald)) if v is None]}


def cash_conversion_cycle(cari: dict, onceki: dict | None = None,
                          *, gun: int = 365) -> dict:
    """Madde 195 — Nakit dönüşüm süresi: DSO + DIO − DPO (gün).

    Bilanço kalemleri ORTALAMA alınır; DPO'nun paydası satış maliyetidir (hasılat
    değil) — hasılatla hesaplamak yaygın ve sessiz bir hatadır ve DPO'yu brüt marj
    kadar küçültür.
    """
    hs, sm = _f(cari, "hasilat"), _f(cari, "satis_maliyeti")

    def ort(ad: str) -> float | None:
        return _ort(_f(cari, ad), _f(onceki, ad) if onceki else None)

    dso = _bol(ort("alacak"), hs)
    dio = _bol(ort("stok"), sm)
    dpo = _bol(ort("borc_satici"), sm)
    dso, dio, dpo = (None if v is None else v * gun for v in (dso, dio, dpo))
    ccc = (None if (dso is None or dio is None or dpo is None) else dso + dio - dpo)
    return {"ccc_gun": ccc, "dso": dso, "dio": dio, "dpo": dpo,
            "eksik": [k for k, v in (("dso", dso), ("dio", dio), ("dpo", dpo))
                      if v is None]}


def goodwill_risk(cari: dict) -> dict:
    """Madde 197 — Şerefiye bozulma maruziyeti: şerefiye / varlık ve / özkaynak.

    `bozulma_yazildi` yalnız kalem RAPORLANMIŞSA True/False'tur; kalem hiç yoksa
    None — "bozulma yazılmadı" ile "bozulma kalemi yok" aynı şey değildir.
    """
    sf = _f(cari, "serefiye")
    boz = _f(cari, "serefiye_bozulmasi")
    return {"serefiye": sf,
            "serefiye_varlik_orani": _bol(sf, _f(cari, "varliklar")),
            "serefiye_ozkaynak_orani": _bol(sf, _f(cari, "ozkaynak")),
            "bozulma": boz,
            "bozulma_yazildi": (None if boz is None else boz > 0)}


def pension_gap(cari: dict) -> dict:
    """Madde 198 — Emeklilik açığı: yükümlülük − plan varlıkları (ve özkaynağa oranı).

    İki kalemden biri yoksa açık `None`dur: eksik plan varlığını 0 saymak, fonlanmış
    bir planı tamamen fonlanmamış gibi gösterirdi.
    """
    yuk, var = _f(cari, "emeklilik_yukumlulugu"), _f(cari, "emeklilik_varligi")
    acik = _fark(yuk, var)
    return {"acik": acik, "yukumluluk": yuk, "plan_varligi": var,
            "ozkaynak_orani": _bol(acik, _f(cari, "ozkaynak")),
            "fonlanma_orani": _bol(var, yuk)}


def rd_intensity(cari: dict, *, yil: int = 5) -> dict:
    """Madde 199 — Ar-Ge yoğunluğu ve AKTİFLEŞTİRME etkisinin büyüklüğü.

    US GAAP Ar-Ge'yi GİDER yazar; Lev-Sougiannis (1996) çizgisindeki düzeltme onu
    `yil` yıla yayar. Burada TAM bir sermayeleştirme yapılmaz (geçmiş Ar-Ge serisi
    gerekir); yapılan şey ETKİNİN ÜST SINIRINI ölçmektir: tek yılın Ar-Ge'sinin
    özkaynağa oranı. Bu bir düzeltilmiş ROE İDDİASI DEĞİLDİR.
    """
    arge, hs = _f(cari, "arge"), _f(cari, "hasilat")
    return {"arge": arge, "arge_hasilat": _bol(arge, hs),
            "arge_ozkaynak": _bol(arge, _f(cari, "ozkaynak")),
            "yayma_yili": yil,
            "yillik_itfa_vekili": (None if arge is None else arge / yil)}


def net_cash_runway(cari: dict, onceki: dict | None = None) -> dict:
    """Madde 179 — Nakit ömrü (biyotek): net nakit / yıllık faaliyet nakit YAKIMI.

    Yakım yalnız faaliyet nakdi NEGATİFKEN tanımlıdır; pozitif nakit üreten şirkette
    `ay` **None**'dur ve `yakim_yok=True` yazılır — sonsuz bir ömrü büyük bir sayıyla
    göstermek, ölçümü sessizce uydurmaktır.
    """
    nakit = _f(cari, "nakit")
    borc = _f(cari, "uzun_vadeli_borc") or 0.0
    cfo = _f(cari, "faaliyet_nakit")
    capex = _f(cari, "capex") or 0.0
    serbest = None if cfo is None else cfo - capex
    net_nakit = None if nakit is None else nakit - borc
    yakim_yok = None if serbest is None else serbest >= 0
    ay = (None if (net_nakit is None or serbest is None or serbest >= 0)
          else 12.0 * net_nakit / abs(serbest))
    return {"ay": ay, "net_nakit": net_nakit, "serbest_nakit_akisi": serbest,
            "yakim_yok": yakim_yok,
            "onceki_donem_nakit": (_f(onceki, "nakit") if onceki else None)}


def shareholder_yield(cari: dict, *, market_cap: float | None = None) -> dict:
    """Madde 190 — Hissedar getirisi: (temettü + geri alım − ihraç) / piyasa değeri.

    `market_cap` yoksa tutarlar ham dolar olarak döner ve `getiri` None olur —
    özkaynak defter değerini piyasa değeri yerine koymak farklı bir orandır.
    """
    tem = _f(cari, "temettu") or 0.0
    geri = _f(cari, "geri_alim") or 0.0
    ihrac = _f(cari, "hisse_ihraci") or 0.0
    net = tem + geri - ihrac
    return {"getiri": _bol(net, market_cap), "net_dagitim": net,
            "temettu": _f(cari, "temettu"), "geri_alim": _f(cari, "geri_alim"),
            "ihrac": _f(cari, "hisse_ihraci"),
            "temettu_getirisi": _bol(tem, market_cap),
            "geri_alim_getirisi": _bol(geri, market_cap)}


def fcf_vs_dividend(cari: dict, *, market_cap: float | None = None) -> dict:
    """Madde 189 — Serbest nakit akışı getirisi ↔ temettü getirisi ve KARŞILAMA oranı.

    `karsilama = FCF / temettü`: 1'in altı, temettünün nakit üretiminden değil
    bilançodan/borçtan ödendiğini söyler.
    """
    cfo, capex = _f(cari, "faaliyet_nakit"), _f(cari, "capex")
    fcf = None if cfo is None else cfo - (capex or 0.0)
    tem = _f(cari, "temettu")
    return {"fcf": fcf, "fcf_getirisi": _bol(fcf, market_cap),
            "temettu_getirisi": _bol(tem, market_cap),
            "karsilama": _bol(fcf, tem),
            "eksik": [k for k, v in (("faaliyet_nakit", cfo), ("temettu", tem))
                      if v is None]}


def lease_burden(cari: dict) -> dict:
    """Madde 183 — Kiralama yükü: ASC 842 sonrası bilanço İÇİNDEDİR, ama ayrı okunmalı.

    "Bilanço dışı kiralama" 2019 sonrası ABD raporlamasında YOKTUR; ölçülebilen şey
    faaliyet kiralaması yükümlülüğünün borç ve varlığa oranıdır. Maddenin adı eski
    rejimden gelir; ölçüm YENİ rejime göre yapılır ve bu gizlenmez.
    """
    kira = _f(cari, "kiralama_yukumlulugu")
    borc = _f(cari, "uzun_vadeli_borc")
    toplam = (None if (kira is None and borc is None)
              else (kira or 0.0) + (borc or 0.0))
    return {"kiralama_yukumlulugu": kira, "uzun_vadeli_borc": borc,
            "kiralama_varlik_orani": _bol(kira, _f(cari, "varliklar")),
            "kiralama_borc_payi": _bol(kira, toplam),
            "rejim": "ASC842_bilanco_ici"}


def earnings_yield(cari: dict, *, market_cap: float | None = None,
                   reel_faiz: float | None = None) -> dict:
    """Madde 152 — Kazanç getirisi (E/P) ve HİSSE RİSK PRİMİ vekili (E/P − reel faiz).

    `reel_faiz` YÜZDE DEĞİL ORAN bekler (%2,35 → 0,0235); `macro_regime`in
    `breakeven_inflation`/`curve_slope` çıktıları puan cinsindendir, çağıran 100'e
    bölmelidir. Birim karışırsa prim 100 kat yanlış çıkar ve HATA VERMEZ.
    """
    ni = _f(cari, "net_kar")
    ep = _bol(ni, market_cap)
    return {"kazanc_getirisi": ep, "net_kar": ni, "market_cap": market_cap,
            "reel_faiz": reel_faiz,
            "risk_primi": (None if (ep is None or reel_faiz is None)
                           else ep - reel_faiz)}


def rapor(cari: dict, onceki: dict, *, market_cap: float | None = None,
          medyanlar: dict[str, float | None] | None = None,
          history: list[dict] | None = None,
          reel_faiz: float | None = None) -> dict[str, Any]:
    """Tüm skorları TEK sözlükte topla (pano/uç bunu okur).

    Bir alt hesap patlarsa tüm rapor DÜŞMEZ; o anahtar `None` olur (`api_quant._guard`
    deseni). Rapor bir HÜKÜM taşımaz — eşik/kural yoktur.
    """
    return {
        "donem": cari.get("_donem_sonu"),
        "onceki_donem": onceki.get("_donem_sonu"),
        "piotroski": piotroski_f_score(cari, onceki),
        "altman": altman_z_score(cari, market_cap=market_cap),
        "beneish": beneish_m_score(cari, onceki),
        "mohanram": mohanram_g_score(cari, onceki, medyanlar, history=history),
        "tahakkuk": accruals_ratio(cari, onceki),
        "dupont": dupont(cari, onceki),
        "nakit_dongusu": cash_conversion_cycle(cari, onceki),
        "serefiye": goodwill_risk(cari),
        "emeklilik": pension_gap(cari),
        "arge": rd_intensity(cari),
        "nakit_omru": net_cash_runway(cari, onceki),
        "hissedar_getirisi": shareholder_yield(cari, market_cap=market_cap),
        "fcf": fcf_vs_dividend(cari, market_cap=market_cap),
        "kiralama": lease_burden(cari),
        "kazanc_getirisi": earnings_yield(cari, market_cap=market_cap,
                                          reel_faiz=reel_faiz),
    }
