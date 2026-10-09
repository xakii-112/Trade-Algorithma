"""Tur-47 — YÖN GÜVENİ bir SKORDUR: hükmü verilmeli, "yüksek = iyi" diye BOYANMAMALI.

Kusur (Tur-36 §9 açık kalem #3, 10 tur açık kaldı): `report()` `buckets` tablosunu
(güven → isabet) HÜKÜMSÜZ basıyordu. Kardeş tablo `pwin_calibration` hükmünü denetim
B-4'te almıştı; aynı kusur bu tabloda kapanmadı. Terslik `config.py` YORUMUNDA yazılıydı
("canlı veride confidence şu an isabetle TERS korele") — bir kod yorumu çalışma-zamanı
hükmü değildir.

Canlı defterde ÖLÇÜLDÜ (63.237 çözülmüş tahmin): ρ = **−0,889**; panonun KENDİ renk
sınıflarıyla YEŞİL (conf≥0,66) isabet %43,95 ↔ GRİ (<0,40) %50,94 → **+6,98pp, z=+14,4**,
hem zaman hem sembol bölmesinde ayrı ayrı. Yani pano ölçülen EN KÖTÜ alt kümeyi "en iyi"
diye boyuyordu.

⛔ Hiçbir test "ters ⇒ tersine çevir" davranışı beklemez (ACI DERS #3/#7). Beklenen düzeltme
SIRALAMAYI İDDİA ETMEMEKTİR: renk nötrleşir, varsayılan sıralama anahtarı iddiasız olur.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nihai.execution.prediction_tracker import _calibration_verdict

_NIHAI = Path(__file__).resolve().parents[1] / "nihai"
_PANO = (_NIHAI / "web/web_dashboard.html").read_text(encoding="utf-8")


def _kovalar(spec: dict[str, tuple[int, float]]) -> dict[str, list[int]]:
    """{'0.2-0.3': (n, isabet_orani)} → kova → 0/1 listesi."""
    return {k: [1] * round(n * r) + [0] * (n - round(n * r)) for k, (n, r) in spec.items()}


# Canlı defterin şekli (yukarıdaki ölçüm): skor yükseldikçe isabet düşüyor.
_TERS = _kovalar({"0.1-0.2": (9425, 0.5223), "0.3-0.4": (7389, 0.4784),
                  "0.6-0.7": (6309, 0.4367), "0.9-1.0": (5658, 0.4470)})


# ---------------------------------------------------------------------------
# 1. SKOR kipi — Brier YOKTUR
# ---------------------------------------------------------------------------
def test_skor_kipinde_brier_hesaplanmaz():
    """Bir SKORA Brier uygulamak, onu olasılık sanmanın ta kendisidir.

    Tur-36 §5 bunu ölçtü: `confidence` uzmanının log-kaybı 1,282 ↔ taban 0,694 (1,85 katı).
    Alan yokluğu bir eksiklik değil, hükmün kendisidir."""
    v = _calibration_verdict([(0.6, 1.0)] * 500, _TERS, konu="skor")
    assert v["konu"] == "skor"
    for alan in ("brier", "brier_taban_orani", "brier_bilgisiz"):
        assert alan not in v, f"skor kipinde {alan} bulunmamalı (olasılık muamelesi)"


def test_olasilik_kipi_varsayilan_ve_brier_tasir():
    """Varsayılan DEĞİŞMEDİ: çağıran `konu` vermezse eski (olasılık) davranış sürer."""
    ciftler = [(0.6, 1.0)] * 300 + [(0.3, -1.0)] * 300
    v = _calibration_verdict(ciftler, _TERS)
    assert v["konu"] == "olasilik"
    assert "brier" in v and v["brier_bilgisiz"] == 0.25


# ---------------------------------------------------------------------------
# 2. Etiketler AYRIDIR — "SIRALIYOR" ≠ "TUTARLI"
# ---------------------------------------------------------------------------
def test_ters_skor_kendi_etiketini_alir():
    v = _calibration_verdict([], _TERS, konu="skor")
    assert v["label"] == "TERS_SIRALAMA"          # "TERS_KALIBRASYON" DEĞİL
    assert v["korelasyon"] < -0.3
    assert "TERS SIRALAMA" in v["text"]


def test_ters_skor_tersine_cevirmeyi_ONERMEZ():
    """ACI DERS #3/#7: aleyhte kanıttan kenar türetilmez. Çare İDDİA ETMEMEKTİR."""
    metin = _calibration_verdict([], _TERS, konu="skor")["text"]
    assert "kenar DEĞİLDİR" in metin
    assert "İDDİA ETMEMEKTİR" in metin
    for yasak in ("tersine çevir ", "ters işlem", "ters yönde gir"):
        assert yasak not in metin


def test_siralayan_skor_kalibre_IDDIA_ETMEZ():
    """Doğru sıralayan bir skor bile olasılık DEĞİLDİR; etiket bunu söylemelidir."""
    artan = _kovalar({"0.2-0.3": (300, 0.40), "0.4-0.5": (300, 0.48),
                      "0.6-0.7": (300, 0.55), "0.8-0.9": (300, 0.62)})
    v = _calibration_verdict([], artan, konu="skor")
    assert v["label"] == "SIRALIYOR"              # "TUTARLI" DEĞİL
    assert "OLASILIK olduğu anlamına GELMEZ" in v["text"]


def test_duz_skor_siralamiyor():
    duz = _kovalar({"0.2-0.3": (300, 0.48), "0.4-0.5": (300, 0.50),
                    "0.6-0.7": (300, 0.49), "0.8-0.9": (300, 0.50)})
    v = _calibration_verdict([], duz, konu="skor")
    assert v["label"] == "SIRALAMIYOR"
    assert v["yayilim"] < 0.05


def test_skor_kipinde_de_ornek_kapisi_isler():
    """Az veriden 'ters skor' demek yeni bir uydurma olurdu (ACI DERS #3)."""
    v = _calibration_verdict([], _kovalar({"0.2-0.3": (10, 0.6)}), konu="skor")
    assert v["label"] == "YETERSIZ_ORNEK"
    assert "korelasyon" not in v


# ---------------------------------------------------------------------------
# 3. OLASILIK kipi REGRESYONA karşı kilitli (B-4 sözleşmesi bit-denk kalmalı)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec,beklenen", [
    ({"0.2-0.3": (400, 0.56), "0.3-0.4": (400, 0.50),
      "0.4-0.5": (400, 0.44), "0.5-0.6": (400, 0.38)}, "TERS_KALIBRASYON"),
    ({"0.2-0.3": (300, 0.25), "0.3-0.4": (300, 0.35),
      "0.4-0.5": (300, 0.45), "0.6-0.7": (300, 0.65)}, "TUTARLI"),
    ({"0.2-0.3": (300, 0.48), "0.3-0.4": (300, 0.50),
      "0.4-0.5": (300, 0.49), "0.5-0.6": (300, 0.50)}, "ZAYIF"),
])
def test_olasilik_etiketleri_degismedi(spec, beklenen):
    kovalar = _kovalar(spec)
    ciftler = [(sum(float(x) for x in k.split("-")) / 2, 1.0 if y else -1.0)
               for k, ys in kovalar.items() for y in ys]
    assert _calibration_verdict(ciftler, kovalar)["label"] == beklenen


# ---------------------------------------------------------------------------
# 4. report() sözleşmesi — hüküm HER ZAMAN alanda
# ---------------------------------------------------------------------------
def test_report_confidence_verdict_tasir():
    from nihai.execution import prediction_tracker as pt
    rows = [{"hit": "1", "fwd_ret_pct": "0.1", "confidence": "0.55", "direction": "+"},
            {"hit": "0", "fwd_ret_pct": "-0.1", "confidence": "0.45", "direction": "-"}]
    rep = pt.report(rows)
    assert rep["confidence_verdict"]["label"] == "YETERSIZ_ORNEK"   # 2 satır → hüküm yok
    assert rep["confidence_verdict"]["konu"] == "skor"


def test_report_ters_defterde_TERS_SIRALAMA_der():
    """Canlı defterin şeklini taşıyan satırlarda hüküm sessiz KALMAMALI."""
    from nihai.execution import prediction_tracker as pt
    rows = [{"hit": str(h), "fwd_ret_pct": "0.1" if h else "-0.1",
             "confidence": k.split("-")[0], "direction": "+"}
            for k, ys in _TERS.items() for h in ys]
    assert pt.report(rows)["confidence_verdict"]["label"] == "TERS_SIRALAMA"


def test_confidence_sutunu_olmayan_satir_report_u_COKERTMEZ():
    """`report()` çağırana ait satır alabilir; eksik kolon KeyError vermemeli."""
    from nihai.execution import prediction_tracker as pt
    rep = pt.report([{"hit": "1", "fwd_ret_pct": "0.1", "direction": "+"}])
    assert rep["buckets"] == {}


# ---------------------------------------------------------------------------
# 5. PANO — ham güven artık "yüksek = iyi" diye BOYANMIYOR
# ---------------------------------------------------------------------------
def test_pano_ham_gUveni_trafik_isigiyla_boyamiyor():
    """REGRESYON KAPISI: kusurun kendisi tam olarak bu ifadeydi.

    `conf >= 0.66 ? yeşil : conf >= 0.40 ? turuncu : gri` — ölçülen en kötü alt kümeyi
    en iyi diye gösteriyordu."""
    assert "conf >= 0.66 ? '#22c55e'" not in _PANO
    assert "const confColor = '#94a3b8';" in _PANO


def test_pano_ham_gUveni_YUZDE_olarak_sunmuyor():
    """'%' birimi bir OLASILIK iddiasıdır; ham güven skordur."""
    assert "'Yön güveni: %'" not in _PANO
    assert "Yön güveni (skor): " in _PANO


def test_pano_confidence_verdict_i_basiyor():
    """Backend hüküm üretip pano basmazsa kusur kapanmamış olur (B-4'ün aynı hatası)."""
    assert "d.confidence_verdict" in _PANO
    assert "confLine" in _PANO


def test_pano_kalibre_olasilik_sutunu_KORUNDU():
    """Nötrleştirme gerçek olasılığı da soldurmamalı — `prob_cal` renkli KALIR."""
    assert "pcColor" in _PANO and "pc >= 0.55 ? '#22c55e'" in _PANO


# ---------------------------------------------------------------------------
# 6. Varsayılan SIRALAMA artık iddia taşımıyor
# ---------------------------------------------------------------------------
def test_varsayilan_siralama_gUvene_bagli_degil():
    """Sıralama da bir yüzeydir: 'yüksek güven önce' bir İDDİADIR ve ölçüm onu çürüttü."""
    src = (_NIHAI / "execution/live_predict_service.py").read_text(encoding="utf-8")
    blok = "\n".join(src[src.index("rows.sort(key="):].splitlines()[:3])
    assert "confidence" not in blok, f"sıralama hâlâ güvene bakıyor:\n{blok}"
    assert 'r["symbol"]' in blok, "eşitlikte deterministik anahtar kalmalı"


# ---------------------------------------------------------------------------
# 7. Taramanın ÜÇÜNCÜ bulgusu — kalibrasyonu FIT EDEN script de skora Brier diyordu
# ---------------------------------------------------------------------------
def test_kalibrasyon_fitinde_skora_brier_denmiyor():
    """`calibrate_forecast` ham güvenin MSE'sini `brier` diye kaydediyordu.

    Brier bir OLASILIK tahminini puanlar; `confs` kalibrasyonun GİRDİSİDİR (skor).
    Adı "brier" kaldıkça 0,3261 gibi bir sayı "kötü kalibre olasılık" diye okunur —
    doğru okuma "bu zaten olasılık değil"dir."""
    src = (Path(__file__).resolve().parents[1]
           / "research/calibrate_forecast.py").read_text(encoding="utf-8")
    assert '"brier":' not in src, "skorun MSE'si hâlâ 'brier' ANAHTARIYLA kaydediliyor"
    assert "m['brier']" not in src, "CLI hâlâ 'Brier' diye basıyor"
    assert '"skor_mse"' in src
    # Referanssız bir sayı iyi/kötü diye okunamaz (ACI DERS #9'un ölçüm tarafı).
    assert '"mse_bilgisiz"' in src and '"mse_taban_orani"' in src


# ---------------------------------------------------------------------------
# 8. Bozuk kova anahtarı hükmü DÜŞÜRMEZ ama uca 500 de attırmaz
# ---------------------------------------------------------------------------
def test_negatif_kova_anahtari_cokertmiyor():
    """`float(k.split("-")[0])` "-0.3--0.2" anahtarında `float('')` ile ÇÖKÜYORDU.

    Bozuk tek bir CSV satırı `/api/predictions/accuracy` ucunu 500'e düşürebilirdi.
    Beklenen: o kova hükme GİRMEZ, kalan kovalar hükmü verir."""
    kovalar = dict(_TERS)
    kovalar["-0.3--0.2"] = [1] * 50 + [0] * 50           # dejenere satırlardan gelmiş kova
    v = _calibration_verdict([], kovalar, konu="skor")
    assert v["label"] == "TERS_SIRALAMA"                  # kalan dört kova hükmü verdi
    assert v["n_kova"] == 4                               # bozuk kova sayılmadı


def test_yalniz_bozuk_kovalar_varsa_hukum_verilmez():
    kovalar = {"-0.3--0.2": [1] * 60, "-0.5--0.4": [0] * 60, "-0.7--0.6": [1] * 60}
    v = _calibration_verdict([], kovalar, konu="skor")
    assert v["label"] == "YETERSIZ_ORNEK"
    assert v["n_kova"] == 0 and v["n"] == 0


def test_bucket_key_ve_mid_ters_fonksiyondur():
    from nihai.execution.prediction_tracker import _bucket_key, _bucket_mid
    for i in range(10):
        x = i / 10 + 0.05
        assert _bucket_mid(_bucket_key(x)) == pytest.approx(x)
    assert _bucket_mid("uydurma") is None


# ---------------------------------------------------------------------------
# 9. İKİ hüküm fonksiyonu ÇELİŞMEZ — ACI DERS #9'un kendi ifadesi
# ---------------------------------------------------------------------------
def test_verdict_INVERTED_tersine_cevirmeyi_ONERMEZ():
    """`verdict()` "sinyali tersine çevirmek bile düşünülebilir" diyordu.

    AYNI MODÜLDEKİ `_calibration_verdict` tam bu çıkarımı AÇIKÇA YASAKLIYOR. İki hüküm
    aynı veride çelişince yanlış olanı okuyan operatör tuzağa düşer (ACI DERS #9)."""
    from nihai.execution.prediction_tracker import verdict
    v = verdict({"resolved": 5000, "hit_rate": 0.44})
    assert v["label"] == "INVERTED"                     # ÖLÇÜM etiketi korunur
    assert "tersine çevirmek bile düşünülebilir" not in v["text"]
    assert "DEĞİLDİR" in v["text"] and "#3/#7" in v["text"]


def test_verdict_diger_dallari_degismedi():
    from nihai.execution.prediction_tracker import verdict
    assert verdict({"resolved": 0})["label"] == "NO_DATA"
    assert verdict({"resolved": 5000, "hit_rate": 0.49})["label"] == "NO_EDGE"
    assert verdict({"resolved": 5000, "hit_rate": 0.56})["label"] == "WEAK_POSITIVE"
