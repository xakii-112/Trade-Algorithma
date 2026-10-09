"""prediction_tracker._calibration_verdict testleri — denetim bulgusu B-4 (2026-07-26).

Bulgu: `pwin_calibration` tablosu ham kova sayılarını basıyordu ama TERSLİĞİ
isimlendirmiyordu (Brier 0.2853 > bilgisiz taban 0.25 ve kovalar ters yönlü olduğu hâlde
rapor sessizdi). Bu testler hükmün üç durumunu ve örneklem kapısını sabitler.

⛔ Hiçbir test "ters ⇒ tersine çevir" davranışı beklemez: bu hüküm ölçüm hijyenidir,
işlem önerisi değildir (ACI DERS #3/#7).
"""
from __future__ import annotations

import pytest

from nihai.execution.prediction_tracker import _calibration_verdict


def _kovalar(spec: dict[str, tuple[int, float]]) -> dict[str, list[int]]:
    """{'0.2-0.3': (n, kazanma_orani)} → kova → 0/1 listesi."""
    out: dict[str, list[int]] = {}
    for key, (n, rate) in spec.items():
        kazanan = round(n * rate)
        out[key] = [1] * kazanan + [0] * (n - kazanan)
    return out


def _ciftler(kovalar: dict[str, list[int]]) -> list[tuple[float, float]]:
    """Kova sözlüğünden (p, getiri) çiftleri — p kova orta noktası, getiri işareti sonuç."""
    pairs: list[tuple[float, float]] = []
    for key, ys in kovalar.items():
        lo, hi = (float(x) for x in key.split("-"))
        p = (lo + hi) / 2.0
        pairs.extend((p, 1.0 if y else -1.0) for y in ys)
    return pairs


def test_ters_kalibrasyon_isimlendirilir():
    """Model yüksek p dedikçe gerçekleşen oran DÜŞÜYORSA açık uyarı üretilmeli."""
    kovalar = _kovalar({"0.2-0.3": (400, 0.56), "0.3-0.4": (400, 0.50),
                        "0.4-0.5": (400, 0.44), "0.5-0.6": (400, 0.38)})
    v = _calibration_verdict(_ciftler(kovalar), kovalar)
    assert v["label"] == "TERS_KALIBRASYON"
    assert v["korelasyon"] < 0
    assert "TERS KALİBRASYON" in v["text"]
    # kenar iddiası/öneri sızmamalı — yasak açıkça yazılı olmalı
    assert "kenar DEĞİLDİR" in v["text"]


def test_tutarli_kalibrasyon():
    kovalar = _kovalar({"0.2-0.3": (300, 0.25), "0.3-0.4": (300, 0.35),
                        "0.4-0.5": (300, 0.45), "0.6-0.7": (300, 0.65)})
    v = _calibration_verdict(_ciftler(kovalar), kovalar)
    assert v["label"] == "TUTARLI"
    assert v["korelasyon"] > 0.9


def test_zayif_kalibrasyon_duz_egri():
    """Kovalar arası gerçekleşen oran neredeyse sabitse sıralama bilgisi yoktur.

    REGRESYON KAPISI: çıplak Pearson ölçek-bağımsızdır — 0.48/0.50/0.49/0.50 gibi DÜZ bir
    eğri bile ρ≈+0.9 verip 'TUTARLI' görünüyordu. Yayılım kapısı bunu eler."""
    kovalar = _kovalar({"0.2-0.3": (300, 0.48), "0.3-0.4": (300, 0.50),
                        "0.4-0.5": (300, 0.49), "0.5-0.6": (300, 0.50)})
    v = _calibration_verdict(_ciftler(kovalar), kovalar)
    assert v["label"] == "ZAYIF"
    assert v["yayilim"] < 0.05
    assert "DÜZ" in v["text"]


def test_duz_egri_negatif_egimde_ters_sayilmaz():
    """Yayılım kapısı yalnız ρ > -0.3 iken devreye girer; gerçek terslik gizlenmemeli."""
    kovalar = _kovalar({"0.2-0.3": (300, 0.52), "0.3-0.4": (300, 0.50),
                        "0.4-0.5": (300, 0.49), "0.5-0.6": (300, 0.48)})
    v = _calibration_verdict(_ciftler(kovalar), kovalar)
    assert v["label"] == "TERS_KALIBRASYON"


def test_yetersiz_ornek_hukum_vermez():
    """Az veriden 'ters kalibre' demek yeni bir uydurma olurdu (ACI DERS #3)."""
    kovalar = _kovalar({"0.2-0.3": (10, 0.60), "0.3-0.4": (10, 0.40)})
    v = _calibration_verdict(_ciftler(kovalar), kovalar)
    assert v["label"] == "YETERSIZ_ORNEK"
    assert "korelasyon" not in v
    assert "Kalibrasyon hükmü YOK" in v["text"]


def test_kucuk_kovalar_hukme_girmez():
    """n < min_bucket_n olan kovalar gürültüdür — sayılmamalı."""
    kovalar = _kovalar({"0.2-0.3": (400, 0.56), "0.3-0.4": (400, 0.50),
                        "0.4-0.5": (400, 0.44), "0.9-1.0": (3, 1.0)})
    v = _calibration_verdict(_ciftler(kovalar), kovalar)
    assert v["n_kova"] == 3            # 3 kovalık eşiği geçen
    assert v["n"] == 1200              # tek-hanelik kova toplamın dışında


def test_brier_alanlari_ve_bilgisiz_taban():
    kovalar = _kovalar({"0.2-0.3": (200, 0.50), "0.3-0.4": (200, 0.50),
                        "0.4-0.5": (200, 0.50)})
    v = _calibration_verdict(_ciftler(kovalar), kovalar)
    assert v["brier_bilgisiz"] == 0.25
    assert 0.0 <= v["brier"] <= 1.0
    # taban-oranı kuralının Brier'i p(1-p); %50 tabanda 0.25
    assert v["brier_taban_orani"] == pytest.approx(0.25, abs=0.01)


def test_bos_girdi_cokmez():
    v = _calibration_verdict([], {})
    assert v["label"] == "YETERSIZ_ORNEK"
    assert v["n"] == 0


def test_report_ciktisinda_pwin_verdict_var(tmp_path, monkeypatch):
    """report() sözleşmesi: hüküm alanı her zaman bulunmalı (pano/CLI ona bakıyor)."""
    from nihai.execution import prediction_tracker as pt

    rows = [{"hit": "1", "fwd_ret_pct": "0.10", "confidence": "0.55",
             "pwin": "0.30", "pwin_side": "LONG", "direction": "+", "action": "BUY"},
            {"hit": "0", "fwd_ret_pct": "-0.10", "confidence": "0.45",
             "pwin": "0.40", "pwin_side": "SHORT", "direction": "-", "action": "SELL"}]
    rep = pt.report(rows)
    assert "pwin_verdict" in rep
    assert rep["pwin_verdict"]["label"] == "YETERSIZ_ORNEK"   # 2 satır → hüküm yok
