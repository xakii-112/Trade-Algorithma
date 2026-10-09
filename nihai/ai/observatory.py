"""
observatory.py — ÜRETİM KÖPRÜSÜ: yazılmış ama HİÇ ÇAĞRILMAYAN ölçüm modüllerini canlı
hatta bağlayan TEK gölge geçidi (Tur-37). KARAR-DIŞI.

NEDEN BU MODÜL VAR
------------------
Denetim şunu ölçtü: `nihai/ai` paketi üretimde YALNIZ üç yerden import ediliyordu
(`execution/neural_shadow`, `notifier/daily_summary`, `web/api_extra`). `signals/*`,
`risk/*`, `execution/*` (neural_shadow hariç), `data/*` fiyat sağlayıcıları,
`telegram_bot`, `terminal_dashboard`, `web/tearsheet` — hiçbirinde tek satır
`nihai.ai` importu yoktu. Aynı denetim beş modülün ÜRETİLDİĞİNİ ama HİÇ
ÇAĞRILMADIĞINI gösterdi: `microstructure`, `regime_hmm`, `options_math` (yalnız
research), `adaptive_calibration`, `calibration_loop`.

Bunun bir kısmı TESADÜF DEĞİLDİR ve öyle kalmalıdır: kanıt kapısını geçmemiş bir
MODEL işlem hattına giremez (ACI DERS #3/#5). Ama "karara giremez" ile "hiç
çağrılmaz" AYNI ŞEY DEĞİLDİR — ve ikincisi bir erdem değil, ölçüm boşluğudur:
çağrılmayan modül ölçülmez, ölçülmeyen modül kanıt kapısına HİÇ gelemez. Kapının
önünde sıra bile yoktu.

Bu modül o sırayı kurar: ölçümler CANLI hatta üretilir, gölge kolonlara/raporlara
YAZILIR, ve hiçbirine karar verdirilmez. `execution/neural_shadow.py`'nin `_neural`
deseninin birebir aynısı — orada nöral p_win için kurulan köprü, burada mikroyapı /
rejim-HMM / veri-sağlığı için kurulur.

ANAYASA (bozulmaz)
------------------
Buradaki hiçbir çıktı sinyale, `pwin`'e, META_GATE'e, boyutlandırmaya, rejim
çarpanına veya emre dokunmaz. Çıktı üç yere gider ve YALNIZ oraya:
  1. `predictions_log.csv` gölge kolonları (kayıttan sonra satırdan düşürülür),
  2. operatör raporları (pano şeridi, Telegram `/beyin`, tear-sheet eki),
  3. `nihai/ai` kendi artefaktları.
Bir mikroyapı ölçüsünün "iyi" görünmesi kenar demek DEĞİLDİR (ACI DERS #3).

FAIL-SOFT: her fonksiyon istisna yerine `None`/boş sözlük döner. Gölge katman canlı
tahmin satırından DAHA AZ önemlidir; hiçbir koşulda onu düşüremez.

NEDENSELLİK: yalnız `hmm_filter` tabanlı (nedensel) yol kullanılır. `hmm_smooth`/
`hmm_viterbi` look-ahead içerir ve buradan ASLA çağrılmaz.

HMM YENİDEN-FIT DİSİPLİNİ: `regime_hmm.gaussian_hmm_fit` parametreleri TÜM örneklemden
tahmin eder — her barda yeniden fit etmek hem 65 ms/sembol maliyet demektir hem de
modülün kendi sözleşmesini ("eğitim penceresinde fit et, `hmm_filter` ile İLERİ
uygula") çiğner. Bu yüzden model sembol başına CACHE'lenir ve yalnız
`AI_OBS_HMM_REFIT_BARS` (varsayılan 60) barda bir yeniden fit edilir; aradaki her bar
tek ileri geçişle (≈1,7 ms) puanlanır.

RV SERİSİ İNŞASI — bilinçli sınır: HAR-RV literatürü GÜN-içi getirilerden GÜNLÜK RV
kurar. Canlı hatta elimizde tek bir bar df'i vardır, bu yüzden RV serisi KAYAN
pencereyle (`AI_OBS_RV_WINDOW`, varsayılan 20 bar) üretilir. Pencereler ÖRTÜŞÜR →
ardışık RV gözlemleri otokorelasyonludur; HAR katsayılarının standart hataları bu
yüzden okunmaz ve okunmuyor (yalnız nokta öngörüsü raporlanır). Sınır burada
yazılıdır ki sonraki operatör t-istatistiği aramasın.

.env:
    AI_OBSERVATORY=off          # tüm gölge ölçümü kapat (varsayılan: açık)
    AI_OBS_RV_WINDOW=20         # kayan gerçekleşen-varyans penceresi (bar)
    AI_OBS_HMM_STATES=2         # HMM durum sayısı (2..6)
    AI_OBS_HMM_REFIT_BARS=60    # kaç barda bir yeniden fit
"""
from __future__ import annotations

import math
import os
import threading
from typing import Any

import numpy as np

from nihai.console import ensure_utf8_stdout

_OFF_VALUES = ("0", "off", "false", "no")

# Sembol × durum × bar ADIMI başına HMM modeli: {"SYM|n|adım_ns": (fit bar_ts, model, fit_len, son görülen
# bar epoch-ns, fitten beri yeni bar)}. Yalnız bu modül yazar; `_LOCK` altında — tarama ve tahmin thread'leri
# eşzamanlı gelir.
_HMM_CACHE: dict[str, tuple[str, dict[str, Any], int, int | None, int]] = {}
_LOCK = threading.Lock()


def enabled() -> bool:
    """Gölge ölçüm açık mı? (`AI_OBSERVATORY=off` kapatır)."""
    return os.getenv("AI_OBSERVATORY", "on").strip().lower() not in _OFF_VALUES


def _int_env(key: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.getenv(key, str(default)))))
    except (TypeError, ValueError):
        return default


def _log_returns(close: np.ndarray) -> np.ndarray | None:
    """Ardışık LOG getiriler. Fiyatta sıfır/negatif/sonlu-değil varsa None (uydurma yok)."""
    if close.size < 2 or not np.all(np.isfinite(close)) or np.any(close <= 0):
        return None
    return np.diff(np.log(close))


def _rv_series(logret: np.ndarray, window: int) -> np.ndarray | None:
    """Kayan pencereli gerçekleşen varyans serisi — `microstructure.realized_variance`
    ile AYNI tanım (Σr², ORTALAMA DEĞİL). Aynı birimde olmaları şart: aksi halde `rv`
    ile `har_rv_fc` iki farklı ölçekte raporlanır ve öngörü "20 kat düşük" görünürdü.

    Modül docstring'indeki "RV SERİSİ İNŞASI" sınırı geçerlidir: pencereler örtüşür.
    HAR tasarımı için en az `HAR_MONTHLY+2` gözlem gerekir; yetmezse None."""
    n = logret.size - window + 1
    if window < 2 or n < 24:      # microstructure.HAR_MONTHLY + 2
        return None
    # Kümülatif kareler toplamı ile O(n) kayan pencere (her pencerede yeniden toplamak
    # 220 bar için de ölçülebilir bir israftır ve bu fonksiyon her barda çağrılır).
    sq = np.concatenate(([0.0], np.cumsum(logret ** 2)))
    rv = sq[window:] - sq[:-window]
    return rv if np.all(np.isfinite(rv)) and np.all(rv >= 0) else None


def _bar_zamanlari(index: Any) -> tuple[np.ndarray, int] | None:
    """(epoch-ns dizisi, son barın epoch-ns'i); DatetimeIndex değilse None. `asi8` tz'li dizinde
    UTC anıdır → aynı an farklı saat dilimi GÖSTERİMİYLE gelse de aynı sayıdır (naive = UTC sözleşmesi)."""
    asi8 = getattr(index, "asi8", None)
    if asi8 is None or not len(asi8):
        return None
    arr = np.asarray(asi8)
    return arr, int(arr[-1])


def _hmm_model(symbol: str, bar_ts: str, logret: np.ndarray,
               index: Any = None) -> dict[str, Any] | None:
    """Sembolün HMM'i — cache'ten ya da yeniden fit. Fit başarısızsa None.

    Yeniden fit koşulu: cache boş VEYA son fitten bu yana `AI_OBS_HMM_REFIT_BARS` kadar
    YENİ bar geldi. `bar_ts` fit anını işaretler (aynı barda ikinci çağrı fit etmez).

    BULGU-80-2: yeni bar sayısı eskiden `logret.size - fit_len` idi. Canlı servis SABİT
    uzunlukta pencere çeker (`lookback_bars=220`) → fark hep 0 → model süreç ömrü boyunca
    BİR KEZ fit ediliyordu (ölçüldü: 300 yeni barda 1 fit; büyüyen pencerede 5). Sayım artık
    her çağrıda SON GÖRÜLEN bardan bu yana gelen barları (zaman anına göre) BİRİKTİRİR: pencere
    uzunluğundan büyük eşikte de doğru sayar ve saat dilimi gösterimi değişince erken fit etmez
    (Codex incelemesi ikisini de ölçtü). DatetimeIndex yoksa eski uzunluk kuralı geçerlidir."""
    from nihai.ai.regime_hmm import gaussian_hmm_fit

    refit_bars = _int_env("AI_OBS_HMM_REFIT_BARS", 60, 5, 5000)
    n_states = _int_env("AI_OBS_HMM_STATES", 2, 2, 6)
    zaman = _bar_zamanlari(index)
    # BULGU-80-4: anahtar eskiden `sembol|durum`du; aynı sembol tahmin servisinden 1m
    # (`PREDICT_INTERVAL`) ve tarama köprüsünden 1h (`HTF_INTERVAL`) barla geliyor → bir aralığın
    # getirilerine fit edilen HMM ötekinin rejim olasılığında kullanılıyordu. Bar ADIMI anahtara girer.
    adim = int(np.median(np.diff(zaman[0]))) if zaman is not None and zaman[0].size >= 2 else None
    key = f"{symbol}|{n_states}|{adim}"
    with _LOCK:   # oku-say-yaz tek adımda: tarama ve tahmin thread'i aynı sembolü eşzamanlı sayabilir
        hit = _HMM_CACHE.get(key)
        if hit is not None:
            fit_ts, model, fit_len, son_ns, yeni = hit
            if zaman is None or son_ns is None:
                if fit_ts == bar_ts or (logret.size - fit_len) < refit_bars:
                    return model
            else:
                arr, ns = zaman
                if ns == son_ns:
                    return model                          # aynı bar (gösterimden bağımsız)
                yeni += int((arr > son_ns).sum())         # son görülen bardan bu yana gelenler
                if yeni < refit_bars:
                    _HMM_CACHE[key] = (fit_ts, model, fit_len, max(ns, son_ns), yeni)
                    return model
    fitted = gaussian_hmm_fit(logret, n_states=n_states)
    if fitted is None:
        return None
    with _LOCK:
        _HMM_CACHE[key] = (bar_ts, fitted, int(logret.size), zaman[1] if zaman else None, 0)
    return fitted


def _bps(x: float | None) -> float | None:
    """Oran → baz puan (0.0001 = 1 bps). None/sonlu-değil → None."""
    if x is None or not math.isfinite(float(x)):
        return None
    return round(float(x) * 10_000.0, 2)


def bar_shadow(df: Any, symbol: str | None = None) -> dict[str, Any] | None:
    """OHLCV bar df → GÖLGE ölçüleri (düz skalerler). KARAR-DIŞI.

    Kaynaklar ve her birinin ne söylediği:
      · `health`      — `ai/data_health.health_check` sağlık skoru (0..100). Bu köprü
                        kurulana kadar modül "YALNIZ ölçer, bağlama Dalga-2'de" diyordu;
                        Dalga-2 burasıdır. `quarantine` bayrağı da taşınır ama HİÇBİR
                        sinyali bastırmaz — yalnız loglanır.
      · `rv`          — son penceredeki gerçekleşen varyans (bar-başına).
      · `har_rv_fc`   — HAR-RV (Corsi 2009) 1 adım ileri RV ÖNGÖRÜSÜ. Projede oynaklık
                        hep ÖLÇÜLÜYOR, hiç TAHMİN EDİLMİYORDU.
      · `jump_share`  — bipower ayrıştırmasıyla sıçrama bileşeninin RV içindeki payı.
      · `spread_bps`  — Corwin-Schultz etkin makas (bps). Tahminci yansız değilse
                        (`reliable=False`) None kalır — ACI DERS: ölçüm YOKLUĞU bir
                        ölçüm gibi raporlanmaz. ACI DERS #7'nin teşhis aracı.
      · `hmm_*`       — NEDENSEL rejim okuması: bir sonraki barın en olası durumu,
                        olasılığı, rejim DEĞİŞİM olasılığı ve durumun okunabilir adı.
                        `knowledge.py`'nin kendi kataloğunda "geçiş olasılığı mevcut
                        modülde yoktur" diye yazılı boşluk buydu.

    `mode_prob` okunmadan `mode`/`label` okunamaz (regime_hmm sözleşmesi) — bu yüzden
    üçü de aynı sözlükte döner ve rapor katmanı ikisini birlikte basar.

    Herhangi bir aşama başarısız olursa o ANAHTAR yok olur; sözlük yine döner. Hiçbir
    girdi istisna fırlatmaz. Kapalıysa/df kullanılamazsa None."""
    if not enabled():
        return None
    try:
        close = np.asarray(df["close"], dtype=float)
        high = np.asarray(df["high"], dtype=float)
        low = np.asarray(df["low"], dtype=float)
    except Exception:
        return None
    logret = _log_returns(close)
    if logret is None or logret.size < 30:
        return None

    out: dict[str, Any] = {}

    # 1) Veri sağlığı — sağlayıcı katmanının köprüsü üzerinden (data/* ↔ nihai.ai bağı).
    try:
        from nihai.data.data_provider import bars_health
        health = bars_health(df, symbol)
        if health:
            out["health"] = round(float(health.get("score", 0.0)), 1)
            out["quarantine"] = bool(health.get("quarantine", False))
    except Exception:
        pass

    # 2) Mikroyapı — gerçekleşen oynaklık ÖNGÖRÜSÜ + etkin makas.
    try:
        from nihai.ai import microstructure as ms

        window = _int_env("AI_OBS_RV_WINDOW", 20, 2, 500)
        rv_now = ms.realized_variance(logret[-window:])
        if rv_now is not None:
            out["rv"] = float(rv_now)
        rv_hist = _rv_series(logret, window)
        if rv_hist is not None:
            model = ms.har_rv_fit(rv_hist)
            if model is not None:
                fc = ms.har_rv_forecast(model, rv_hist)
                if fc is not None and math.isfinite(fc) and fc >= 0:
                    out["har_rv_fc"] = float(fc)
                    out["har_persistence"] = round(float(model.get("persistence", 0.0)), 4)
        split = ms.jump_split(logret[-max(window, 30):])
        if split is not None:
            out["jump_share"] = round(float(split["jump_share"]), 4)
        cs = ms.corwin_schultz_spread(high, low)
        # `spread` yalnız yansız tahmin pozitifse dolu gelir; None ise makas ÖLÇÜLEMEDİ —
        # kırpılmış değeri yazmak Tur-31'in "ölçüm yokluğu ödüllendirilmez" dersini bozardı.
        if cs is not None and cs.get("spread") is not None:
            out["spread_bps"] = _bps(cs["spread"])
    except Exception:
        pass

    # 2b) HACİM PROFİLİ + YAPISAL KIRILMA (Tur-76). Tur-75 üç modül yazdı
    # (`volume_profile`/`econophysics`/`ml_hygiene`) ve üçünün de üretim çağıranı YOKTU;
    # çağrılmayan modül ölçülmez, ölçülmeyen modül kanıt kapısına HİÇ gelemez (Tur-37).
    # Seçim UCUZ ve NEDENSEL olanlarla sınırlı: hacmin fiyat eksenindeki dağılımı
    # (POC/değer alanı), net alıcı baskısı (CVD) ve CUSUM olay bayrağı. `ml_hygiene`
    # BİLEREK bağlanmadı — o bir ARAŞTIRMA aracıdır (etiketleme/CV), bar başına
    # çalıştırılacak bir ölçüm değil.
    try:
        from nihai.ai import volume_profile as vp

        pencere = _int_env("AI_OBS_VP_WINDOW", 120, 20, 2000)
        vol = np.asarray(df["volume"], dtype=float)
        prof = vp.volume_profile(high[-pencere:], low[-pencere:], vol[-pencere:])
        if prof is not None and prof.get("poc") is not None:
            son = float(close[-1])
            out["poc"] = round(float(prof["poc"]), 4)
            # Fiyatın değer alanına göre YERİ: 1 = VAH üstü, −1 = VAL altı, 0 = içeride.
            # Bir SKORDUR, olasılık değildir (Tur-47: bir skora Brier uygulanmaz).
            vah, val = float(prof["vah"]), float(prof["val"])
            out["value_area_pos"] = 1 if son > vah else (-1 if son < val else 0)
            out["poc_dist_bps"] = _bps((son - float(prof["poc"])) / son) if son else None
        cvd = vp.cumulative_volume_delta(close[-pencere:], vol[-pencere:])
        if cvd is not None and cvd.get("last") is not None:
            out["cvd_last"] = round(float(cvd["last"]), 2)
            if cvd.get("divergence") is not None:
                out["cvd_divergence"] = int(cvd["divergence"])
    except Exception:
        pass

    try:
        from nihai.ai.econophysics import cusum_filter

        cus = cusum_filter(logret)
        if cus is not None and cus.get("n"):
            # Son barın kendisi bir CUSUM olayı mı? (yapısal kırılma örneklemesi)
            # `or []` KULLANILMAZ: `events` bir numpy dizisidir ve `dizi or []`
            # `bool(dizi)` çağırıp "truth value of an array is ambiguous" ile patlar —
            # üstelik burada geniş `except` onu YUTAR ve iki ölçü sessizce kaybolurdu
            # (Tur-74: fail-soft bir yolda istisna hiçbir şey kanıtlamaz).
            olaylar = list(cus.get("events", []))
            out["cusum_event"] = bool(olaylar and int(olaylar[-1]) >= len(logret) - 1)
            out["cusum_rate"] = round(float(cus.get("event_rate", 0.0)), 4)
    except Exception:
        pass

    # 3) Rejim HMM — NEDENSEL (hmm_filter tabanlı); smooth/viterbi ASLA kullanılmaz.
    try:
        from nihai.ai.regime_hmm import hmm_next_state_probs, hmm_state_labels

        bar_ts = str(getattr(df, "index", [""])[-1])
        model = _hmm_model(symbol or "?", bar_ts, logret, getattr(df, "index", None))
        if model is not None:
            nxt = hmm_next_state_probs(logret, model, horizon=1)
            if nxt is not None:
                out["hmm_mode_prob"] = round(float(nxt["mode_prob"]), 4)
                out["hmm_switch_prob"] = round(float(nxt["switch_prob"]), 4)
                labels = hmm_state_labels(model)
                if labels and 0 <= int(nxt["mode"]) < len(labels):
                    out["hmm_label"] = labels[int(nxt["mode"])]
    except Exception:
        pass

    return out or None


# ---------------------------------------------------------------------------
# Kalibrasyon köprüsü — `calibration_loop` + `adaptive_calibration` üretim çağrısı.
# İkisi de yazılmıştı, artefakt üretiyordu, ama üretimde HİÇBİR çağıranı yoktu: tazeleme
# yalnız elle CLI koşulunca oluyordu. `daily_summary` gece koşusu artık `refresh_calibration`
# çağırır; okuma tarafını (`calibration_line`) pano/Telegram/tear-sheet paylaşır.
# ---------------------------------------------------------------------------
def refresh_calibration() -> dict[str, Any]:
    """Her iki kalibrasyon döngüsünü de yeniden ölçer (gece görevi). KARAR-DIŞI.

    `calibration_loop.refresh` = tek zaman-bölmeli OOS kapısı (ham'ı yenmeli).
    `adaptive_calibration.refresh` = yürüyen/prequential kapı (ham'ı VE SABİT TABANI
    yenmeli — ACI DERS #9 kapının içinde). İkisi AYRI artefakta yazar ve ayrı sorulara
    cevap verir; biri diğerinin yerine geçmez.

    Dönüş: {'loop': {...}|None, 'adaptive': {...}|None} — hata alan taraf None kalır
    (biri çökerse diğeri yine tazelenir; günlük özet hiçbir koşulda düşmez)."""
    out: dict[str, Any] = {"loop": None, "adaptive": None}
    for key, mod_name in (("loop", "calibration_loop"), ("adaptive", "adaptive_calibration")):
        try:
            import importlib
            out[key] = importlib.import_module(f"nihai.ai.{mod_name}").refresh()
        except Exception:
            out[key] = None
    return out


def calibration_line() -> str:
    """Kalibrasyonun TEK SATIRLIK dürüstlük şeridi — SALT OKUNUR (yeniden fit ETMEZ).

    Pano/Telegram/tear-sheet aynı satırı paylaşır ki operatör üç yerde üç farklı sayı
    görmesin. Artefakt yoksa "ölçülmedi" der; ASLA metrik uydurmaz.

    Biçim: `kalibrasyon: pwin=KAYITLI(B 0.2490) · neural=KAPI_GECILEMEDI | yürüyen: pwin=…`
    """
    parts: list[str] = []
    for label, mod_name, loader in (
        ("kalibrasyon", "calibration_loop", "load_calibrators"),
        ("yürüyen", "adaptive_calibration", "load_artifact"),
    ):
        try:
            import importlib
            art = getattr(importlib.import_module(f"nihai.ai.{mod_name}"), loader)()
        except Exception:
            art = {}
        if not isinstance(art, dict) or not art:
            parts.append(f"{label}: ölçülmedi")
            continue
        items = []
        for source in sorted(art):
            entry = art[source]
            if not isinstance(entry, dict):
                continue
            durum = str(entry.get("durum") or "?")
            brier = None
            for holder in ("adaptif", "kalibre", "oos"):
                blk = entry.get(holder)
                if isinstance(blk, dict) and isinstance(blk.get("brier"), (int, float)):
                    brier = float(blk["brier"])
                    break
            items.append(f"{source}={durum}" + (f"(B {brier:.4f})" if brier is not None else ""))
        parts.append(f"{label}: " + (" · ".join(items) if items else "ölçülmedi"))
    return " | ".join(parts)


def ai_status_text() -> str:
    """Operatöre gösterilecek çok satırlı AI DÜRÜSTLÜK KARNESİ (Telegram `/beyin`, pano).

    Üç kaynak: öğrenme karnesi (kaynak-başı Brier/BSS), son öz-değerlendirme kartı,
    kalibrasyon şeridi. Hepsi SALT OKUNUR — hiçbir sınav yeniden koşulmaz, hiçbir model
    yeniden fit edilmez. Her parça bağımsız fail-soft: biri yoksa diğerleri yine basılır.

    KARAR-DIŞI: burada görünen hiçbir sayı bir işlem kararına girmez; bu metin sistemin
    KENDİ kalibrasyonu hakkındaki dürüstlük şerididir (kenar iddiası DEĞİLDİR)."""
    lines = ["🧠 *AI durum* (KARAR-DIŞI — gölge ölçüm, kenar iddiası değil)"]
    try:
        from nihai.ai.learning import learning_report
        rep = (learning_report() or "").strip()
        if rep:
            lines.append(rep)
    except Exception:
        pass
    try:
        from nihai.ai.selfcheck import last_card_line
        card = (last_card_line() or "").strip()
        if card:
            lines.append(card)
    except Exception:
        pass
    lines.append(calibration_line())
    return "\n".join(lines)


def demo() -> None:
    """Kendi kendini sınayan çalıştırılabilir kontrol: `python -m nihai.ai.observatory`."""
    import pandas as pd

    rng = np.random.default_rng(11)
    n = 260
    # İki rejimli sentetik seri: HMM'in ayırt edecek bir şeyi olsun.
    vol = np.where(np.arange(n) < n // 2, 0.004, 0.012)
    ret = rng.normal(0.0, vol)
    close = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    df = pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998,
                       "close": close, "volume": 1e6}, index=idx)

    out = bar_shadow(df, "DEMO")
    assert out is not None, "gölge ölçüm üretilemedi"
    assert "har_rv_fc" in out and out["har_rv_fc"] > 0, "HAR-RV öngörüsü yok"
    assert 0.0 <= out["hmm_mode_prob"] <= 1.0, "HMM olasılığı bantta değil"
    assert 0.0 <= out["hmm_switch_prob"] <= 1.0, "geçiş olasılığı bantta değil"
    # Determinizm: aynı girdi → aynı çıktı (anayasa madde 6; HMM cache'i de bozmamalı).
    assert bar_shadow(df, "DEMO") == out, "aynı girdi farklı çıktı verdi"
    # Bozuk girdi çökmemeli.
    assert bar_shadow(df.iloc[:5], "DEMO") is None, "kısa df None dönmeliydi"
    assert bar_shadow(None, "DEMO") is None, "df=None None dönmeliydi"
    assert isinstance(calibration_line(), str)
    print("observatory demo OK:", {k: out[k] for k in sorted(out)})


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    demo()
