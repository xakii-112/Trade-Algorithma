"""
calibrate_forecast.py — Canlı tahminin YAYINLANAN güvenini GERÇEKLEŞEN isabete kalibre et.

Çıktı `data/forecast_calibration.json` ({"points": [[güven, olasılık], ...], "meta": {...}});
tahmin servisi ve hisse-detay ucu onu "Olasılık (kal.)" için okur.

Tur-83 (BULGU-83-3) YÖNTEMİ DEĞİŞTİRDİ. Eski sürüm tarihsel barlarda `_bias_and_confidence`ı
HTF/rejim HARİÇ koşturup 0,05'lik ham kovalarla ÖRNEKLEM-İÇİ bir eğri yazıyordu. O eğri
23 Haziran'dan beri hiç tazelenmedi ve canlı defterde SABİT TABAN ORANDAN KÖTÜ çıktı
(93.825 çözülmüş tahmin: Brier 0,25002 ↔ nedensel taban 0,24940, gün-kümeli t=+4,38).
Üç ayrı kusuru vardı: HEDEF farklıydı (bar→bar, beraberlik hariç ↔ canlı `hit`: kotasyon→bar,
beraberlik = ıska), GİRDİ farklıydı (yayınlanan güven HTF/rejim DAHİLDİR) ve fit örneklem-
İÇİYDİ (19 kovada 0,4775 → 0,5772 gibi gürültü sıçramaları doğrudan eğriye giriyordu).

Yeni yöntem:
  1. VERİ: `predictions_log.csv`in çözülmüş satırları. Yayınlanan kipin varyantı okunur
     (`_yon_kipi`: bar → `dir_bar`/`conf_bar`; canli → `dir_live`/`conf_live`, kolonun boş
     olduğu eski satırlarda `direction`/`confidence`). İsabet tracker'ın `hit` tanımıyla ve
     CANLI referanstan hesaplanır (tahminin kullanılabildiği fiyat; beraberlik = 0).
  2. YÜRÜYEN DOĞRULAMA: her gün için platt YALNIZ ÖNCEKİ günlerden fit edilir ve o gün
     puanlanır. Puanlanan şey YAYINLANACAK temsilin ta kendisidir (21 noktalı ızgara +
     servisin `_apply_calibration`ı). Rakip, nedensel koşan TABAN ORANDIR (ACI DERS #9).
  3. KAPI: eşleştirilmiş Brier farkının GÜN-kümeli t'si < −1,645 ve ≥5 test günü. Satırlar
     gün içinde bağımlıdır (30 sembol aynı piyasayı görür) → satır-t sahte anlamlılık üretir
     (BULGU-53-1). Kapı geçilirse eğri TÜM veriden platt'tır; geçilmezse eğri DÜZ = taban
     oran. "Öngörü gücü yok"un dürüst olasılığı budur — gürültülü bir eğri değil.
  4. META: girdi/yön/aralık/ufuk. `_load_calibration` bunlar eşleşmeyen eğriyi REDDEDER.

Gece görevi (`forward-harness.ps1`) her gün koşturur → eğri bir daha aylarca bayatlamaz.

Tur-84 — ARALIK (`data/forecast_band.json`). Pano "Öngörü" sütunu `fiyat ± ATR`yi yön renginde
gösteriyordu; yön yazı-tura olduğu için bu nokta tahmini "fiyat değişmez"den ölçülerek kötüydü
(BULGU-84-2). Yerini ufuk sonundaki fiyatın ~%80 ARALIĞI aldı: çapa karar barının KAPANIŞI (kotasyon
çapası aynı satırlarda ölçülerek kötüydü — BULGU-84-4), ölçek o barın ATR14'ü (%), kuantiller
z = getiri / ATR% dağılımından (sembolün kendi ≥300 satırı varsa o, yoksa havuz).
Veri defterin `atr_pct` kolonudur (Tur-84'te doğdu; kolonu taşıyan her satır ZAMAN-tabanlı ufukla
çözülür). Kapı: gün-gün yürüyen değerlendirmede yayınlanacak aralık (`_apply_band`), oynaklığı hiç
okumayan KOŞULSUZ aralığı (geçmiş getirilerin kendi %10/%90'ı) aralık skorunda (Gneiting-Raftery)
gün-kümeli t < −1,645 ile yenmeli. Geçmezse dosya durum=KAPI_GECILEMEDI yazar (pano "—");
yeterli test günü yoksa dosyaya DOKUNULMAZ (hüküm yok). `--bant-barlardan`: kolonu olmayan eski
satırlar için ölçeği ve zaman-tabanlı ileri fiyatı Alpaca IEX barlarından yeniden kuran TEK
SEFERLİK önyükleme (defter değiştirilmez).

Tur-85 — ÖLÇEK ÜSSÜ (BULGU-85-1). Ham ATR14 aralığının toplam kapsaması %79'du ama KOŞULLU değildi:
sembol-içi ATR ondalığında %70,6 (en sakin) → %86,4 (en oynak). 14 barlık ATR anlık oynaklığın gürültülü
tahminidir; sapması 5 dakikada kısmen geri döner. Ölçek artık ATR14^us (sembol kuantilleriyle bu, ölçeği
sembolün kendi düzeyine büzmekle özdeştir). `us` BANT_US_IZGARA'dan örneklem-içi aralık skoruyla seçilir;
yürüyen sınavda her gün YALNIZ geçmişten seçilen üs, sabit ATR'yi (us=1) AYNI satırlarda gün-kümeli
t < −1,645 ile yenmezse us=1 yayınlanır (Tur-83 platt↔taban kapısıyla aynı ilke: karmaşık olan ancak
basiti yürüyen sınavda yenerse). Ölçülen: 20 aday ölçeğin (gerçekleşen varyans 10–120 dk, zaman-normalize
TR², EWMA, önceki gün) HİÇBİRİ ATR14'ü yenmedi; üs −0,63 bps (t=−3,51, 32 günün 29'u) ve bulguların hiç
bakılmadığı Mart–Mayıs 2026'da (686.864 satır, 60 test günü) −0,53 bps, t=−8,46, 60/60 gün us=0,8.
Son yarım saat için ek bir KAPANIŞ ÇARPANI Haziran–Eylül'de kapıyı geçti (t=−2,75) ama örneklem-dışı
dönemde bir kez bile seçilmedi → eklenmedi (her gece yeniden sınanan boş bir özellik, örtüşen veride er
geç sahte bir "geçti" üretir).

    python -m research.calibrate_forecast
    python -m research.calibrate_forecast --kip canli --out C:/tmp/fc.json   # karşı varyant
    python -m research.calibrate_forecast --bant-barlardan                   # aralık önyüklemesi
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

T_ESIK = 1.645          # tek-yönlü ~%95 — adaptive_calibration / harness ile AYNI eşik
ISINMA_GUN = 3          # yalnız fit; asla puanlanmaz
MIN_TEST_GUN = 5        # bundan az test gününde kümeli t'ye güvenilmez → kapı geçilmez
IZGARA = tuple(i / 20 for i in range(21))   # yayınlanan eğrinin güven ızgarası (0..1)
BANT_KUANTIL = (0.10, 0.90)   # ~%80 merkezi aralık
BANT_MIN_SEMBOL = 300         # sembolün KENDİ z-kuantili için asgari satır (altında havuz)
BANT_US_IZGARA = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)   # ölçek üssü adayları; eşitlikte önce gelen (1,0) kazanır

Satir = tuple[str, str, float, float]   # aralık satırı: (gün, sembol, getiri %, ATR %)


def canli_ciftler(rows: list[dict], kip: str, interval: str,
                  horizon: int) -> list[tuple[str, float, int]]:
    """Çözülmüş satırlardan (gün, yayınlanan güven, isabet 0/1) — `kip` varyantında.

    İsabet CANLI referanstan (`price` → `fwd_price`) ve tracker'ın `hit` tanımıyla
    (beraberlik = 0). Varyantı olmayan/bozuk satır ATLANIR (uydurma yok)."""
    out: list[tuple[str, float, int]] = []
    for r in rows:
        if r.get("hit") not in ("0", "1") or r.get("interval") != interval:
            continue
        if str(r.get("horizon_bars") or "") != str(horizon):
            continue
        if kip == "bar":
            d, c = r.get("dir_bar"), r.get("conf_bar")
        elif r.get("dir_live"):
            d, c = r.get("dir_live"), r.get("conf_live")
        else:                              # Tur-83 öncesi satır: yayınlanan yön canlıydı
            d, c = r.get("direction"), r.get("confidence")
        if d not in ("+", "-"):
            continue
        try:
            conf, px, fwd = float(c), float(r["price"]), float(r["fwd_price"])  # type: ignore[arg-type]
        except (TypeError, ValueError, KeyError):
            continue
        if not (math.isfinite(conf) and px > 0 and fwd > 0):
            continue
        s = 1.0 if d == "+" else -1.0
        out.append((str(r.get("logged_at") or "")[:10], conf, 1 if s * (fwd - px) > 0 else 0))
    return out


def _gun_kumeli_t(fark: np.ndarray, kume: np.ndarray) -> float | None:
    """Ortalamanın GÜN-kümeli t'si: CR0 + küçük-küme düzeltmesi G/(G−1) (Cameron-Miller).

    Satır AĞIRLIKLIDIR: 1 satırlık gün 9.000 satırlık gün kadar oy almaz. (`horizon_sweep.
    _cluster_t` gün ORTALAMALARINI eşit tartar — bu defterde servisin kısa koştuğu 1-30
    satırlık günler var ve eşit tartım onların gürültüsünü kapıya taşır: canlı varyantta
    aynı veri t=−1,40 ↔ CR0 −2,93.) <2 küme veya sıfır varyans → None (hüküm yok)."""
    n = fark.size
    if n < 2:
        return None
    m = float(fark.mean())
    _, inv = np.unique(kume, return_inverse=True)
    g = int(inv.max()) + 1
    if g < 2:
        return None
    u = np.bincount(inv, weights=fark - m, minlength=g)
    var = float((u ** 2).sum()) / (n * n) * g / (g - 1)
    return m / math.sqrt(var) if var > 0 else None


def _egri(pairs: list[tuple[float, int]]) -> list[list[float]] | None:
    """Platt (lojistik) fit → yayınlanacak 21 noktalı eğri. Fit edilemezse None."""
    from nihai.ai import learning
    calib = learning.fit_calibrator(pairs, "platt")
    if calib is None:
        return None
    return [[x, round(learning.apply_calibration(x, calib), 4)] for x in IZGARA]


def yuruyen_kapi(ciftler: list[tuple[str, float, int]], *, isinma_gun: int = ISINMA_GUN,
                 min_test_gun: int = MIN_TEST_GUN) -> dict[str, Any]:
    """Gün-gün yürüyen değerlendirme: eğri ↔ nedensel taban, AYNI satırlarda.

    Kesin nedensellik: `g` gününü puanlayan eğri ve taban YALNIZ `g`'den önceki günlerden
    kurulur. Dönüş: gecti (bool), test_gun, n_test, brier_model, brier_taban, fark, gun_t."""
    from nihai.execution.live_predict_service import _apply_calibration

    gunler = sorted({g for g, _, _ in ciftler})
    gunluk: dict[str, list[tuple[float, int]]] = {g: [] for g in gunler}
    for g, c, y in ciftler:
        gunluk[g].append((c, y))
    gecmis: list[tuple[float, int]] = []
    fark: list[float] = []
    kume: list[str] = []
    bm = bt = 0.0
    for i, g in enumerate(gunler):
        if i >= isinma_gun and gecmis:
            taban = sum(y for _, y in gecmis) / len(gecmis)
            pts = _egri(gecmis)
            for c, y in gunluk[g]:
                pc = _apply_calibration(pts, c) if pts else None
                p = taban if pc is None else pc
                dm, dt = (p - y) ** 2, (taban - y) ** 2
                bm += dm
                bt += dt
                fark.append(dm - dt)
                kume.append(g)
        gecmis.extend(gunluk[g])
    n = len(fark)
    test_gun = len(set(kume))
    t = _gun_kumeli_t(np.asarray(fark, dtype=float), np.asarray(kume)) if n else None
    gecti = bool(test_gun >= min_test_gun and t is not None and t < -T_ESIK)
    return {"gecti": gecti, "test_gun": test_gun, "n_test": n,
            "brier_model": round(bm / n, 5) if n else None,
            "brier_taban": round(bt / n, 5) if n else None,
            "fark": round((bm - bt) / n, 6) if n else None,
            "gun_t": round(t, 2) if t is not None else None, "esik": -T_ESIK}


def kalibre_et(rows: list[dict], *, kip: str, interval: str, horizon: int) -> dict[str, Any]:
    """Kapıdan geçen eğri (platt) ya da DÜZ taban-oran eğrisi + meta. Veri yoksa ValueError."""
    ciftler = canli_ciftler(rows, kip, interval, horizon)
    if not ciftler:
        raise ValueError(f"çözülmüş '{kip}' varyantı satırı yok (interval={interval}, "
                         f"ufuk={horizon}) — önce `prediction_tracker --resolve`")
    ys = np.array([y for _, _, y in ciftler], dtype=float)
    cs = np.array([c for _, c, _ in ciftler], dtype=float)
    taban = float(ys.mean())
    kapi = yuruyen_kapi(ciftler)
    egri = _egri([(c, y) for _, c, y in ciftler]) if kapi["gecti"] else None
    if egri is None:
        egri = [[0.0, round(taban, 4)], [1.0, round(taban, 4)]]
    return {
        "points": egri,
        "meta": {
            "kaynak": "canli_defter", "girdi": "yayinlanan_guven", "yon": kip,
            "interval": interval, "horizon": horizon,
            "yontem": "platt" if kapi["gecti"] else "taban",
            "durum": "BILGI_VAR" if kapi["gecti"] else "TABAN",
            "samples": len(ciftler), "gun": len({g for g, _, _ in ciftler}),
            "overall_hit_rate": round(taban, 4),
            # Ham SKORUN (olasılık DEĞİL) MSE'si — girdiyi puanlar, Brier diye OKUNAMAZ (Tur-47).
            "skor_mse": round(float(np.mean((cs - ys) ** 2)), 4),
            "mse_taban_orani": round(taban * (1.0 - taban), 4),
            "mse_bilgisiz": 0.25,
            "yuruyen": kapi,
            "uretildi": datetime.now().isoformat(timespec="seconds"),
            "note": "canlı defter; yayınlanan güven (HTF/rejim DAHİL) → CANLI referanstan isabet "
                    "(beraberlik=0); gün-gün yürüyen platt ↔ nedensel taban, gün-kümeli t<−1,645 "
                    "değilse eğri DÜZ = taban oran (öngörü gücü yok).",
        },
    }


def bant_satirlari(rows: list[dict], interval: str, horizon: int) -> list[Satir]:
    """(gün, sembol, getiri %, ATR %) — defterde `atr_pct` taşıyan çözülmüş satırlar.

    Getiri aralığın ÇAPASINDAN, yani karar barının KAPANIŞINDAN (`bar_close` → `fwd_price`) ölçülür
    ve YÖNSÜZDÜR: aralık yön iddia etmez. Kotasyon çapası (`price`) aynı satırlarda ölçülerek kötüydü
    (BULGU-84-4). Tur-84 öncesi satırlarda `atr_pct` boştur → ATLANIR; kolonu taşıyan her satır
    zaman-tabanlı ufukla çözülmüştür (`prediction_tracker._forward_price`)."""
    out: list[Satir] = []
    for r in rows:
        if r.get("hit") not in ("0", "1") or r.get("interval") != interval:
            continue
        if str(r.get("horizon_bars") or "") != str(horizon):
            continue
        try:
            bc, fwd = float(r.get("bar_close") or "nan"), float(r["fwd_price"])
            a = float(r.get("atr_pct") or "nan")
        except (TypeError, ValueError, KeyError):
            continue
        if not (bc > 0 and fwd > 0 and math.isfinite(a) and a > 0):
            continue
        out.append((str(r.get("logged_at") or "")[:10], str(r["symbol"]),
                    (fwd - bc) / bc * 100.0, a))
    return out


def _z_ozet(z: np.ndarray) -> dict[str, Any]:
    lo, hi = np.quantile(z, BANT_KUANTIL)
    return {"z_lo": round(float(lo), 4), "z_hi": round(float(hi), 4),
            "z_med_abs": round(float(np.median(np.abs(z))), 4), "n": int(z.size)}


def _bant_diziler(satirlar: list[Satir]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(getiri, ATR%, sembol kodu, sembol adları) — sembol eşleşmesi BİR kez tamsayı koda iner."""
    n = len(satirlar)
    u, kod = np.unique(np.array([x[1] for x in satirlar], dtype=str), return_inverse=True)
    return (np.fromiter((x[2] for x in satirlar), float, n),
            np.fromiter((x[3] for x in satirlar), float, n), kod, u)


def _modeli_dizi(r: np.ndarray, a: np.ndarray, kod: np.ndarray, u: np.ndarray,
                 us: float) -> dict[str, Any]:
    z = r / a ** us                              # us=1,0: a**1.0 == a (IEEE) → Tur-84 ile BİREBİR
    say = np.bincount(kod, minlength=u.size)
    sembol = {str(u[k]): _z_ozet(z[kod == k]) for k in range(u.size) if say[k] >= BANT_MIN_SEMBOL}
    return {"havuz": _z_ozet(z), "sembol": sembol, "us": us}


def bant_modeli(satirlar: list[Satir], us: float = 1.0) -> dict[str, Any]:
    """{havuz, sembol: {SYM: {z_lo, z_hi, z_med_abs, n}}, us} — z = getiri / ATR%^us.

    Sembolün kendi kuantili yalnız ≥ BANT_MIN_SEMBOL satırda yayınlanır. Havuz kuantilleri
    sembolden sembole kapsamayı %66 (V) ile %90 (SPY) arasına dağıtıyordu; sembol kuantilleri
    %73–%82'ye topladı (Tur-84 ölçümü, aynı yürüyen satırlar)."""
    return _modeli_dizi(*_bant_diziler(satirlar), us)


def _aralik_skoru(lo, hi, y):
    """Gneiting-Raftery (2007) aralık skoru: genişlik + dışarıda kalan mesafenin 2/α katı.
    Küçük = iyi. Genişliği tek başına ödüllendirmez, kapsamayı tek başına ödüllendirmez.
    Skaler ya da dizi (üs seçimi dizide puanlar — tek formül)."""
    alfa = 2.0 * BANT_KUANTIL[0]
    return (hi - lo) + (2.0 / alfa) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))


def _ic_skor(r: np.ndarray, a: np.ndarray, kod: np.ndarray, u: np.ndarray, us: float) -> float:
    """`us` üssüyle kurulan modelin AYNI satırlardaki (örneklem-içi) ortalama aralık skoru."""
    m = _modeli_dizi(r, a, kod, u, us)
    q = [m["sembol"].get(str(k)) or m["havuz"] for k in u]
    s = a ** us
    lo = np.array([x["z_lo"] for x in q])[kod] * s
    hi = np.array([x["z_hi"] for x in q])[kod] * s
    return float(np.mean(_aralik_skoru(lo, hi, r)))


def bant_us_sec(satirlar: list[Satir]) -> float:
    """BANT_US_IZGARA'dan örneklem-İÇİ aralık skorunu en küçükleyen ölçek üssü (eşitlikte 1,0).

    Tek parametre, satır sayısı onbinler → örneklem-içi seçim ucuz ve kararlı; DÜRÜSTLÜĞÜ yürüyen
    sınav sağlar (her test günü yalnız geçmişten seçer — `bant_yuruyen`)."""
    d = _bant_diziler(satirlar)
    return min(BANT_US_IZGARA, key=lambda us: _ic_skor(*d, us))


def _yuruyen(satirlar: list[Satir], isinma_gun: int, us_sec: bool) -> dict[str, list]:
    """Satır başına yürüyen puanlar. `g` gününü puanlayan kuantiller (ve `us_sec` ise üs) YALNIZ
    `g`'den önceki günlerden kurulur; aralık servisin kendi `_apply_band`ıyla uygulanır (puanlanan
    şey yayınlanan temsilin ta kendisidir). Rakip, oynaklığı hiç okumayan taban: geçmiş getirilerin
    kendi %10/%90 kuantilleri.

    Maliyet ~ satır × gün × (1 + ızgara): her test günü büyüyen geçmişte model + seçim kurulur.
    Ölçüldü: 686.864 satır / 60 test günü ≈ 48 sn (canlı defterin ~2 aylık birikimi). Tavan budur;
    defter aylarca büyüyünce yükseltme yolu, kuantil ve seçimi son N günle sınırlamaktır (Tur-84
    ölçümü: kayan 20 gün ≈ genişleyen pencere)."""
    from nihai.execution.live_predict_service import _apply_band

    gunler = sorted({x[0] for x in satirlar})
    gunluk: dict[str, list[Satir]] = {g: [] for g in gunler}
    for s in satirlar:
        gunluk[s[0]].append(s)
    gecmis: list[Satir] = []
    out: dict[str, list] = {k: [] for k in ("kume", "sym", "sm", "su", "icinde", "alti", "us")}
    for i, g in enumerate(gunler):
        if i >= isinma_gun and gecmis:
            us = bant_us_sec(gecmis) if us_sec else 1.0
            model = bant_modeli(gecmis, us)
            ulo, uhi = (float(v) for v in np.quantile([x[2] for x in gecmis], BANT_KUANTIL))
            out["us"].append(us)
            for _, sym, r, a in gunluk[g]:
                b = _apply_band(model, sym, a)
                if b is None:
                    continue
                lo, hi, tipik = b
                out["kume"].append(g)
                out["sym"].append(sym)
                out["sm"].append(float(_aralik_skoru(lo, hi, r)))
                out["su"].append(float(_aralik_skoru(ulo, uhi, r)))
                out["icinde"].append(1.0 if lo <= r <= hi else 0.0)
                out["alti"].append(1.0 if abs(r) <= tipik else 0.0)
        gecmis.extend(gunluk[g])
    return out


def _yuruyen_ozeti(y: dict[str, list], min_test_gun: int) -> dict[str, Any]:
    fark = np.asarray(y["sm"], dtype=float) - np.asarray(y["su"], dtype=float)
    n = fark.size
    test_gun = len(set(y["kume"]))
    t = _gun_kumeli_t(fark, np.asarray(y["kume"])) if n else None
    gecti = bool(test_gun >= min_test_gun and t is not None and t < -T_ESIK)
    sembol_kaps: dict[str, list[float]] = defaultdict(list)
    for s, k in zip(y["sym"], y["icinde"], strict=True):
        sembol_kaps[s].append(k)
    sk = [float(np.mean(v)) for v in sembol_kaps.values()]
    return {"gecti": gecti, "test_gun": test_gun, "n_test": n,
            "kapsama": round(float(np.mean(y["icinde"])), 4) if n else None,
            "sembol_kapsama_min_max": [round(min(sk), 3), round(max(sk), 3)] if sk else None,
            "tipik_alti_pay": round(float(np.mean(y["alti"])), 4) if n else None,
            "skor_model_bps": round(float(np.sum(y["sm"])) / n * 100, 2) if n else None,
            "skor_kosulsuz_bps": round(float(np.sum(y["su"])) / n * 100, 2) if n else None,
            "gun_t": round(t, 2) if t is not None else None, "esik": -T_ESIK}


def _eslestir(taban: dict[str, list], aday: dict[str, list], min_test_gun: int,
              secim: list) -> dict[str, Any]:
    """Aday − taban aralık skoru, AYNI satırlarda; gün-kümeli t < −T_ESIK → aday kazanır."""
    if taban["kume"] != aday["kume"] or taban["sym"] != aday["sym"]:
        # `_apply_band` yalnız ölçek/model geçersizse satır atlar; üs ya da çarpan bunu değiştirmez.
        # Ayrışma eşleştirmeyi anlamsız kılardı → sessizce yanlış t üretmek yerine dur.
        raise RuntimeError("yürüyen adaylar farklı satırları puanladı — eşleştirme kurulamaz")
    fark = np.asarray(aday["sm"], dtype=float) - np.asarray(taban["sm"], dtype=float)
    test_gun = len(set(taban["kume"]))
    t = _gun_kumeli_t(fark, np.asarray(taban["kume"])) if fark.size else None
    return {"gecti": bool(test_gun >= min_test_gun and t is not None and t < -T_ESIK),
            "fark_bps": round(float(fark.mean()) * 100, 3) if fark.size else None,
            "gun_t": round(t, 2) if t is not None else None, "esik": -T_ESIK,
            "gunluk_secim": dict(Counter(str(v) for v in secim))}


def bant_yuruyen(satirlar: list[Satir], *, isinma_gun: int = ISINMA_GUN,
                 min_test_gun: int = MIN_TEST_GUN) -> dict[str, Any]:
    """Gün-gün yürüyen değerlendirme, İKİ aday AYNI satırlarda: sabit ATR (us=1) ve her gün YALNIZ
    geçmişten seçilen üs. Seçilen üs sabite karşı gün-kümeli t < −T_ESIK ile kazanırsa yayınlanan odur,
    değilse us=1 (Tur-85; Tur-83 platt↔taban kapısıyla aynı ilke). Özet YAYINLANAN adayındır (koşulsuz
    aralığa karşı kapı + kapsama); `us_kiyas` eşleştirilmiş farkı ve günlük üs seçimlerini taşır.

    Özetin sabit ATR'ye karşı kazancı hafifçe İYİMSERDİR: üs, kapı tüm dönemi gördükten sonra açılır
    (parametre seçimi nedensel, yapı kararı değil). Tam nedensel prosedür (kapı kararı da her gün yalnız
    geçmişten) ölçüldü: −0,43 ↔ −0,78 bps, ikisi de anlamlı (Tur-85 raporu §8)."""
    bir = _yuruyen(satirlar, isinma_gun, us_sec=False)
    usr = _yuruyen(satirlar, isinma_gun, us_sec=True)
    us_k = _eslestir(bir, usr, min_test_gun, usr["us"])
    ozet = _yuruyen_ozeti(usr if us_k["gecti"] else bir, min_test_gun)
    ozet["us_kiyas"] = us_k
    return ozet


def bant_kalibre_et(satirlar: list[Satir], *, kaynak: str,
                    interval: str, horizon: int) -> dict[str, Any] | None:
    """Kapıdan geçen kuantiller ya da durum=KAPI_GECILEMEDI (kuantilsiz → pano "—").

    Yeterli test günü yoksa None: HÜKÜM YOK → çağıran dosyaya DOKUNMAZ (önyükleme korunur)."""
    if not satirlar:
        return None
    kapi = bant_yuruyen(satirlar)
    if kapi["test_gun"] < MIN_TEST_GUN:
        return None
    # Üs ancak yürüyen sınavda sabit ATR'yi yendiyse 1'den farklıdır; seçim TÜM satırlardan (platt gibi).
    us = bant_us_sec(satirlar) if kapi["us_kiyas"]["gecti"] else 1.0
    out: dict[str, Any] = {"meta": {
        "kaynak": kaynak, "olcek": "atr14_yuzde_us", "referans": "bar_kapanisi", "ufuk": "zaman",
        "interval": interval, "horizon": horizon, "kuantil": list(BANT_KUANTIL),
        "durum": "GECTI" if kapi["gecti"] else "KAPI_GECILEMEDI",
        "samples": len(satirlar), "gun": len({x[0] for x in satirlar}),
        "yuruyen": kapi, "uretildi": datetime.now().isoformat(timespec="seconds"),
        "note": "ufuk sonundaki fiyatın ~%80 aralığı = son bar kapanışı × (1 + z·ATR14%^us); z kuantilleri "
                "sembolün kendi (≥300 satır) ya da havuz dağılımından; us yalnız yürüyen sınavda sabit "
                "ATR'yi (us=1) gün-kümeli t<−1,645 ile yenerse 1'den farklı; aralık KOŞULSUZ aralığı "
                "gün-kümeli t<−1,645 ile yenmezse kuantil yayınlanmaz.",
    }}
    if kapi["gecti"]:
        out.update(bant_modeli(satirlar, us))
    return out


def barlardan_bant_satirlari(rows: list[dict], interval: str, horizon: int, *,
                             get_history=None) -> list[Satir]:
    """TEK SEFERLİK ÖNYÜKLEME — `atr_pct`ı olmayan (Tur-84 öncesi) çözülmüş satırlar için ölçeği
    ve ZAMAN-tabanlı ileri fiyatı Alpaca IEX 1m barlarından yeniden kurar: canlı hub'ın gördüğü
    seri (feed=iex, DÜZELTMESİZ — defter ham fiyat taşır; temettü düzeltmesi 5 dakikalık getiriyi
    yüzde onlarca bps kaydırırdı). ATR servisle AYNI fonksiyondan (`strategy.atr`), ileri fiyat
    çözücüyle AYNI kuraldan (`prediction_tracker._forward_price`). Defter DEĞİŞTİRİLMEZ; diske
    önbellek yazılmaz (bir kerelik)."""
    from nihai.data.alpaca_history import get_history as _alpaca_gecmis
    from nihai.execution.prediction_tracker import _canon_ts, _forward_price, _pos_map
    from nihai.signals.strategy import atr
    cek: Any = get_history or _alpaca_gecmis
    by_sym: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if (r.get("hit") in ("0", "1") and r.get("interval") == interval
                and str(r.get("horizon_bars") or "") == str(horizon)):
            by_sym[str(r["symbol"])].append(r)
    out: list[Satir] = []
    eski = os.environ.get("ALPACA_HIST_ADJUSTMENT")
    os.environ["ALPACA_HIST_ADJUSTMENT"] = "raw"
    try:
        for sym, rs in sorted(by_sym.items()):
            gunler = sorted(str(r.get("logged_at") or "")[:10] for r in rs)
            try:
                bas = (date.fromisoformat(gunler[0]) - timedelta(days=5)).isoformat()
                son = (date.fromisoformat(gunler[-1]) + timedelta(days=4)).isoformat()
                df = cek(sym, interval, start=bas, end=son, feed="iex", use_cache=False)
            except Exception:
                continue
            if df is None or getattr(df, "empty", True):
                continue
            pmap = _pos_map(df)
            av = atr(df, 14).to_numpy(dtype=float)
            kapanis = df["close"].to_numpy(dtype=float)
            for r in rs:
                pos = pmap.get(_canon_ts(r["bar_ts"]))
                if pos is None:
                    continue
                bc = float(kapanis[pos])           # çapa: karar barının kapanışı (servisle aynı)
                if not (bc > 0 and np.isfinite(av[pos]) and av[pos] > 0):
                    continue
                fwd = _forward_price(df, pmap, r["bar_ts"], horizon, interval)
                if fwd is None or not fwd > 0:
                    continue
                out.append((str(r.get("logged_at") or "")[:10], sym,
                            (fwd - bc) / bc * 100.0, float(av[pos]) / bc * 100.0))
    finally:
        if eski is None:
            os.environ.pop("ALPACA_HIST_ADJUSTMENT", None)
        else:
            os.environ["ALPACA_HIST_ADJUSTMENT"] = eski
    return out


def main(argv: list[str] | None = None) -> int:
    from nihai.config import DATA_DIR, HTF_INTERVAL
    from nihai.execution import prediction_tracker as pt
    from nihai.execution.live_predict_service import _yon_kipi

    ap = argparse.ArgumentParser(description="Canlı tahmin güven kalibrasyonu (yürüyen, kapılı)")
    ap.add_argument("--kip", choices=("bar", "canli"), default=None,
                    help="yön varyantı (vars. servisin kipi: SIGNAL_ON_BAR_CLOSE)")
    ap.add_argument("--interval", default=os.getenv("PREDICT_INTERVAL") or HTF_INTERVAL)
    ap.add_argument("--horizon", type=int, default=pt.HORIZON_BARS)
    ap.add_argument("--log", default=str(pt.PRED_LOG), help="girdi defteri (vars. canlı)")
    ap.add_argument("--out", default=str(DATA_DIR / "forecast_calibration.json"))
    ap.add_argument("--bant-out", default=str(DATA_DIR / "forecast_band.json"),
                    help="aralık kuantilleri (Tur-84)")
    ap.add_argument("--bant-barlardan", action="store_true",
                    help="atr_pct'siz eski satırları Alpaca IEX barlarından yeniden kur (önyükleme)")
    args = ap.parse_args(argv)

    import csv
    with open(args.log, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    kip = args.kip or _yon_kipi()
    try:
        sonuc = kalibre_et(rows, kip=kip, interval=args.interval, horizon=args.horizon)
    except ValueError as e:
        print(f"[kalibrasyon] YAZILMADI: {e}")
        return 1
    from nihai.atomic_io import atomic_write_text
    atomic_write_text(Path(args.out), json.dumps(sonuc, ensure_ascii=False, indent=2))

    m, k = sonuc["meta"], sonuc["meta"]["yuruyen"]
    print(f"Kalibrasyon: kip={kip} interval={m['interval']} ufuk={m['horizon']} · "
          f"{m['samples']} örnek / {m['gun']} gün · taban isabet %{m['overall_hit_rate'] * 100:.2f}")
    print(f"Yürüyen: {k['test_gun']} test günü, n={k['n_test']} · Brier eğri {k['brier_model']} ↔ "
          f"taban {k['brier_taban']} · fark {k['fark']} · gün-kümeli t={k['gun_t']} (eşik {k['esik']})")
    print(f"Ham SKOR MSE {m['skor_mse']} (taban-oranı kuralı {m['mse_taban_orani']} · bilgisiz "
          f"{m['mse_bilgisiz']}) — girdi bir olasılık değil skordur, bu sayı Brier DEĞİLDİR")
    if m["durum"] == "TABAN":
        print(f"→ TABAN: güven taban oranın ötesinde bilgi taşımıyor; olasılık DÜZ "
              f"%{m['overall_hit_rate'] * 100:.2f} (dürüst 'öngörü gücü yok').")
    else:
        print("→ BİLGİ VAR: platt eğrisi yürüyen doğrulamada sabit tabanı yendi.")
    print(f"Kaydedildi: {args.out}")

    # Tur-84 — ARALIK: defterin `atr_pct` satırları (+ istenirse eski satırların IEX yeniden kurulumu).
    satirlar = bant_satirlari(rows, args.interval, args.horizon)
    kaynak = "canli_defter"
    if args.bant_barlardan:
        ek = barlardan_bant_satirlari([r for r in rows if not (r.get("atr_pct") or "").strip()],
                                      args.interval, args.horizon)
        kaynak = "canli_defter+iex_yeniden_kurulum" if satirlar else "iex_yeniden_kurulum"
        satirlar = satirlar + ek
    bant = bant_kalibre_et(satirlar, kaynak=kaynak, interval=args.interval, horizon=args.horizon)
    if bant is None:
        print(f"Aralık: YAZILMADI — {len(satirlar)} satır, yürüyen değerlendirme için < {MIN_TEST_GUN} "
              f"test günü (hüküm yok; mevcut {Path(args.bant_out).name} KORUNDU)")
        return 0
    atomic_write_text(Path(args.bant_out), json.dumps(bant, ensure_ascii=False, indent=2))
    bm, by = bant["meta"], bant["meta"]["yuruyen"]
    print(f"Aralık ({bm['kaynak']}): {bm['samples']} satır / {bm['gun']} gün · yürüyen {by['test_gun']} "
          f"test günü, n={by['n_test']} · kapsama %{by['kapsama'] * 100:.1f} (hedef %80; sembol "
          f"{by['sembol_kapsama_min_max']}) · tipik altı %{by['tipik_alti_pay'] * 100:.1f} (hedef %50)")
    print(f"Aralık skoru {by['skor_model_bps']} ↔ koşulsuz {by['skor_kosulsuz_bps']} bps · gün-kümeli "
          f"t={by['gun_t']} (eşik {by['esik']}) → {bm['durum']}")
    uk = by["us_kiyas"]
    print(f"Ölçek üssü: us={bant.get('us', '—')} · yürüyen seçim ↔ sabit ATR {uk['fark_bps']} bps, "
          f"gün-kümeli t={uk['gun_t']} → {'SEÇİLEN ÜS' if uk['gecti'] else 'SABİT ATR (us=1)'} · "
          f"günlük seçimler {uk['gunluk_secim']}")
    print(f"Kaydedildi: {args.bant_out}")
    return 0


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    raise SystemExit(main())
