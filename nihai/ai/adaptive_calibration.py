"""
adaptive_calibration.py — AI olasılık kaynakları için YÜRÜYEN (prequential / walk-forward)
ÇEVRİMİÇİ kalibrasyon + ZORUNLU taban-oran kontrolü (Tur-27). KARAR-DIŞI / GÖLGE.

ANAYASA: Bu paket sinyal/eşik/META_GATE/execution'a ASLA bağlanmaz (CLAUDE.md "AI katmanı").
Yalnız ARTEFAKT ÜRETİR ve RAPOR YAZAR. `calibrate()` bile yalnız rapor/gölge içindir.

NEDEN (iki gerçek boşluk — ikisi de yerinde ölçüldü):

  1. **Hiçbir kalibrasyon ADAPTE OLMUYOR.** Sistemdeki her kalibratör TEK, STATİK bir fit:
     `calibration_loop` kronolojik %60/%40 böler, bir kez fit eder, biter.
     `data/forecast_calibration.json` ise `live_predict_service.__init__`'te BİR KEZ okunur
     (`self._calib`) — 7/24 çalışan bir serviste eğri tazelense bile süreç onu görmez.
     Rejim değişince eşleme bayatlar ve kimse elle CLI koşana kadar öyle kalır.
     Bu modül eşlemeyi ZAMAN İÇİNDE yeniden fit eder: her `refit_every` örnekte kalibratör
     YALNIZ GEÇMİŞTEN yeniden öğrenilir ve BİR SONRAKİ bloğa uygulanır (kesin nedensel —
     hiçbir fit'e gelecek veri girmez; sızıntısızlık kanıtı testte).

  2. **ACI DERS #9 kâğıtta kaldı, MAKİNEDE yok.** Tur-25'te `calibration_loop` meta `pwin`
     için ECE 0,1325→0,0045 ölçtü ve artefakta `durum="KAYITLI"` yazdı. Sonra ELLE yapılan
     bir kontrol kazancın neredeyse tamamının **taban orana büzülmeden** geldiğini gösterdi
     (sabit tahminci Brier 0,2494 · kalibre 0,2490 → fark 0,0004). Ama `refresh()` kapısı bu
     kontrolü YAPMAZ; artefakt bugün hâlâ "KAYITLI" diyor. Bir sonraki operatör aynı tuzağa
     düşer. Burada **sabit taban-oran tahmincisi ÜÇÜNCÜ bir yarışmacı olarak kapının içine
     gömüldü**: adaptif sürüm HEM ham'ı HEM tabanı yenmedikçe "işe yarıyor" DENMEZ.
        · ham'a karşı  → Brier VE ECE düzelmeli (kalibrasyon hijyeni; `calibration_loop` kapısı)
        · TABAN'a karşı → EŞLEŞTİRİLMİŞ Brier farkı, t < −1,645 (bilgi testi)
     Tabana karşı ECE ARANMAZ, bilinçli: sabit tahminci ECE'de yapısı gereği ~0'dır (hep taban
     oranı söyleyen tahminci kusursuz kalibredir), onu ECE'de yenmesini istemek hiçbir bilgili
     modelin geçemeyeceği bir KATEGORİ HATASI olurdu. Gerekçe: `_paired_brier_t` docstring'i.
     Ders artık prose değil, kod.

Yöntem (prequential / "interleaved test-then-train", Dawid 1984):
    üçlüler zamana göre sıralanır → ilk `warmup` yalnız FİT için (asla puanlanmaz) →
    kalan veri `refit_every`'lik bloklara bölünür → her blok için kalibratör O ANA KADARKİ
    geçmişten fit edilir (`window=None` genişleyen, `window=N` kayan pencere = rejim
    adaptasyonu knob'u) → blok bu kalibratörle puanlanır → blok geçmişe eklenir.
    ÜÇ yarışmacı TAM AYNI örneklerde puanlanır: `ham` · `adaptif` · `taban` (geçmişin
    koşan taban oranı = sabit tahminci).

`p_std` neden raporlanıyor: ACI DERS #9'un teşhis izi. Kalibre olasılıkların std'si çökmüşse
(Tur-25'te 0,0686→0,0156) model bilgi bulmuyor, sadece taban orana büzülüyordur. Metrik
iyileşmesi bunu GİZLER; std göstergesi ele verir.

BİLİNEN SINIR: `neural` kaynağı bugün ölçülemez — canlı köprü servisler kapanmadan önce
yalnız 6 satır üretti (yerinde ölçüldü, 2026-08-11). Modül bunu dürüstçe `YETERSIZ` der.

Kullanım:
    python -m nihai.ai.adaptive_calibration --eval
    python -m nihai.ai.adaptive_calibration --eval --window 1000 --refit-every 50
    python -m nihai.ai.adaptive_calibration --report --json
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
from loguru import logger

from nihai.ai import calibration_loop, learning
from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR

# Modül-düzeyi yol: testler monkeypatch'ler. Girdi logu `calibration_loop.PRED_LOG`'dur
# (etiketleme mantığının TEK doğruluk kaynağı orası — testler ONU patch'lemeli).
ARTIFACT_FILE = DATA_DIR / "ai_adaptive_calibration.json"

DEFAULT_SOURCES: tuple[str, ...] = ("pwin", "neural")
# Tur-77: "venn_abers" varsayılan kümeye girdi — günlük `--calib` görevi koşmazsa aralık
# teşhisi (nötr/taban kapsama, kova kapsaması) hiçbir rapora ulaşmaz (Tur-37: çağrılmayan
# ölçüm ölçülmez). Maliyeti ölçüldü: canlı pwin'de 0,9 sn (platt 3,0 sn).
DEFAULT_METHODS: tuple[str, ...] = ("platt", "isotonic", "venn_abers")

# `calibration_loop.DEFAULT_TAIL_N` (5000 HAM satır) bu logda yalnız 576 kullanılabilir üçlü
# veriyor (ölçüldü) — yürüyen değerlendirme için fazla dar. 60k ham satır tüm logu kapsar.
DEFAULT_TAIL_N = 60_000
DEFAULT_WARMUP = 1_000      # yalnız fit; ASLA puanlanmaz
DEFAULT_REFIT_EVERY = 100   # kaç örnekte bir kalibratör yeniden öğrenilir
DEFAULT_WINDOW: int | None = None   # None = genişleyen geçmiş; N = kayan pencere
DEFAULT_MIN_EVAL = 500      # puanlanan örnek bu sayının altındaysa verdikt verilmez
T_ESIK = 1.645              # tek-yönlü ~%95 — Tur-24 `harness` Welch eşiğiyle AYNI (bkz. _paired_brier_t)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _round(x: float | None, nd: int = 4) -> float | None:
    return round(x, nd) if isinstance(x, int | float) else None


def _fmt(x: float | None, spec: str = ".4f") -> str:
    return format(x, spec) if isinstance(x, int | float) else "?"


def _p_std(pairs: list[tuple[float, int]]) -> float | None:
    """Tahmin edilen olasılıkların std'si — ACI DERS #9 teşhis izi (büzülme ele verir).
    <2 örnek → None (numpy'siz; bu modülde ağır bağımlılık gereksiz)."""
    ps = [p for p, _ in pairs]
    n = len(ps)
    if n < 2:
        return None
    mean = sum(ps) / n
    return (sum((p - mean) ** 2 for p in ps) / n) ** 0.5


# Eşleştirilmiş Brier t-testi TEK kaynaktan gelir (`learning`): Tur-28'de `calibration_loop`
# kapısına da aynı taban-oran kontrolü konunca iki kopya olurdu. Eşik `T_ESIK=1.645`
# (tek-yönlü ~%95) BİLEREK Tur-24 `harness` Welch eşiğiyle aynı — yeni gelenek icat edilmedi.
_paired_brier_t = learning.paired_brier_t


def _score(pairs: list[tuple[float, int]]) -> dict[str, float | None]:
    """Bir yarışmacının karnesi: Brier + ECE (+ ortalama/std teşhis). Boş → hepsi None."""
    return {
        "brier": _round(learning.brier_score(pairs)),
        "ece": _round(learning.expected_calibration_error(pairs)),
        "p_mean": _round(sum(p for p, _ in pairs) / len(pairs)) if pairs else None,
        "p_std": _round(_p_std(pairs)),
    }


# ---------------------------------------------------------------------------
# VENN–ABERS ARALIK KALİBRASYONU (Tur-77, Madde 5.3)
#
# Platt/izotonik TEK bir nokta olasılık verir. Tümevarımsal Venn–Abers (IVAP; Vovk ve Petej
# 2014) aynı kalibrasyon kümesinden İKİ olasılık üretir: test skoru s, kalibrasyon kümesine
# önce y=0 sonra y=1 etiketiyle eklenir, her seferinde izotonik regresyon yeniden uydurulur
# ve s'deki değerler okunur → [p0, p1]. Aralığın genişliği kalibrasyon kümesinin o skor
# bölgesinde ne kadar SEYREK olduğunu söyler; aralık 0,5'i kapsıyorsa kalibrasyon kümesi
# yönü ayırt edemiyor demektir ("nötr").
#
# GARANTİNİN SINIRI (sayıyı okuyanın ne SANABİLECEĞİ): Venn–Abers'in "p0/p1'den biri
# kusursuz kalibredir" garantisi DEĞİŞTİRİLEBİLİRLİK (exchangeability) varsayar. Zaman
# serisi bu varsayımı SAĞLAMAZ — bu yüzden aralık burada bir "matematiksel garanti" diye
# sunulmaz; geçerliliği yürüyen değerlendirmede KOVA KAPSAMASIYLA ölçülür
# (`_venn_abers_teshis`). "nötr" bayrağı bir RAPOR alanıdır; hiçbir karar yoluna bağlanmaz.
#
# "0,5'İ KAPSIYORSA NÖTR" KURALI BİR BİLGİ DEDEKTÖRÜ DEĞİLDİR (Tur-77'de ölçüldü). Aralık
# genişliği kaynağın bilgisini değil kalibrasyon kümesinin YOĞUNLUĞUNU ölçer: aynı dağılımdan
# n=100 → ort. genişlik ≈0,10, n=3000 → ≈0,009 (üç tohumda 11–13×). Büyük kümede aralık o kadar
# daralır ki bilgisiz bir kaynak (taban 0,47) bile aralıkların yalnız %2–3'ünde 0,5'i, %3–16'sında
# koşan taban oranı kapsar; bilgili kaynakta %0–7. Yani iki kapsama payı da bilgisiz ile bilgiliyi
# güvenilir biçimde AYIRMAZ. Bilgi sorusunun cevabı yine eşleştirilmiş Brier farkıdır (tabana
# karşı t < −1,645) — bu modülün kapısı. İki pay da raporda TEŞHİS olarak basılır, hüküm değil.
#
# Hesap: izotonik uyum eşit skorları ağırlıklı ortalamayla birleştirir (ikincil bağ
# yaklaşımı — sklearn `IsotonicRegression` ile aynı). Bir test skorunun sonucu yalnız
# kalibrasyon skorları arasındaki KONUMUNA bağlıdır (k benzersiz skor → k "eşit" + k+1
# "arada" konum); konum başına iki uyum yapılır ve aynı bloktaki tekrarlar önbellekten okunur.
# ---------------------------------------------------------------------------
VA_METHOD = "venn_abers"


def venn_abers_fit(pairs: list[tuple[float, int]]) -> dict | None:
    """(p_ham, y) kalibrasyon çiftlerinden IVAP TABLOSU: benzersiz skorlar + skor başına
    gözlem sayısı + pozitif sayısı. İzotonik uyum burada DEĞİL tahmin anında yapılır
    (yalnız sorulan konumlar için — tüm konumları önceden hesaplamak k² iş demektir).

    <2 geçerli çift veya tek sınıf → None (`learning.fit_calibrator` ile aynı sözleşme:
    çağıran ham p'yi AYNEN kullanır)."""
    ps: list[float] = []
    ys: list[float] = []
    for p, y in pairs:
        if isinstance(p, bool) or not isinstance(p, int | float) or not math.isfinite(p):
            continue
        if y not in (0, 1):
            continue
        ps.append(float(p))
        ys.append(float(y))
    if len(ps) < 2 or len(set(ys)) < 2:
        return None
    s = np.asarray(ps, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    scores, inv = np.unique(s, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    positives = np.bincount(inv, weights=y_arr)
    return {"method": VA_METHOD, "n": len(ps), "scores": scores.tolist(),
            "counts": counts.tolist(), "positives": positives.tolist()}


def venn_abers_predict(table: dict | None, ps: list[float]) -> list[tuple[float, float] | None]:
    """Her p için (p0, p1). Tablo geçersizse veya p sonlu değilse o eleman None.

    Eşit skor (s = u_j): u_j'deki kovanın ortalaması (pozitif+etiket)/(sayı+1) olur.
    Arada skor (u_{j-1} < s < u_j): j konumuna ağırlığı 1 olan yeni nokta eklenir.
    İzotonik regresyon etikete göre monoton olduğundan p0 ≤ p1 yapısı gereği sağlanır."""
    out: list[tuple[float, float] | None] = [None] * len(ps)
    if not isinstance(table, dict) or table.get("method") != VA_METHOD:
        return out
    try:
        scores = np.asarray(table["scores"], dtype=float)
        counts = np.asarray(table["counts"], dtype=float)
        positives = np.asarray(table["positives"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return out
    if scores.size == 0 or not (scores.size == counts.size == positives.size):
        return out
    if np.any(counts <= 0) or not np.all(np.isfinite(scores)):
        return out
    from scipy.optimize import isotonic_regression

    base = positives / counts
    cache: dict[tuple[bool, int], tuple[float, float]] = {}
    for k, p in enumerate(ps):
        if isinstance(p, bool) or not isinstance(p, int | float) or not math.isfinite(p):
            continue
        j = int(np.searchsorted(scores, float(p)))
        esit = bool(j < scores.size and scores[j] == float(p))
        key = (esit, j)
        if key not in cache:
            vals: list[float] = []
            for etiket in (0.0, 1.0):
                if esit:
                    yy = base.copy()
                    ww = counts.copy()
                    yy[j] = (positives[j] + etiket) / (counts[j] + 1.0)
                    ww[j] = counts[j] + 1.0
                else:
                    yy = np.insert(base, j, etiket)
                    ww = np.insert(counts, j, 1.0)
                vals.append(float(isotonic_regression(yy, weights=ww, increasing=True).x[j]))
            cache[key] = (vals[0], vals[1])
        out[k] = cache[key]
    return out


def venn_abers_merge(p0: float, p1: float) -> float:
    """Aralığı tek olasılığa indirir: p = p1 / (1 − p0 + p1) — log-kayıpta minimaks birleştirme
    (Vovk ve Petej 2014). p0 ≤ p1 iken payda ≥ 1'dir ve sonuç daima [p0, p1] içindedir."""
    return float(p1) / (1.0 - float(p0) + float(p1))


def venn_abers_interval(pairs: list[tuple[float, int]], p: float) -> dict | None:
    """Tek çağrılık kolaylık: kalibrasyon çiftleri + ham p → {p0, p1, p, genislik, notr}.
    `notr` = aralık 0,5'i kapsıyor (RAPOR alanı; karar yoluna bağlanmaz). Ölçülemezse None."""
    iv = venn_abers_predict(venn_abers_fit(pairs), [p])[0]
    if iv is None:
        return None
    p0, p1 = iv
    return {"p0": p0, "p1": p1, "p": venn_abers_merge(p0, p1), "genislik": p1 - p0,
            "notr": bool(p0 <= 0.5 <= p1)}


def _venn_abers_teshis(rows: list[tuple[float, float, int, float]], n_kova: int = 10) -> dict | None:
    """Yürüyen değerlendirmede toplanan (p0, p1, y, koşan_taban) dörtlülerinin karnesi.

    Değiştirilebilirlik zaman serisinde sağlanmadığı için geçerlilik VARSAYILMAZ, ölçülür:
    örnekler birleşik p'ye göre eşit sayılı kovalara ayrılır; her kovada gözlenen oran,
    kovanın [ort p0, ort p1] aralığına (±1,96 binom SE toleransla) düşüyor mu? `kapsama_payi`
    kovaların örnek-ağırlıklı kapsanma payıdır. <`n_kova`·10 örnekte kova tablosu kurulmaz.

    `notr_payi` (aralık 0,5'i kapsıyor) ve `taban_kapsayan_payi` (aralık o anki koşan taban
    oranı kapsıyor) TEŞHİSTİR — bölüm başındaki ölçüm notuna bak: ikisi de bilgi dedektörü değil."""
    if not rows:
        return None
    arr = np.asarray(rows, dtype=float)
    p0, p1, y, taban = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    genislik = p1 - p0
    out: dict[str, Any] = {
        "n_aralik": int(arr.shape[0]),
        "genislik_ort": _round(float(np.mean(genislik))),
        "genislik_medyan": _round(float(np.median(genislik))),
        "notr_payi": _round(float(np.mean((p0 <= 0.5) & (0.5 <= p1)))),
        "taban_kapsayan_payi": _round(float(np.mean((p0 <= taban) & (taban <= p1)))),
        "kova": None, "kapsama_payi": None,
    }
    if arr.shape[0] < n_kova * 10:
        return out
    merged = p1 / (1.0 - p0 + p1)
    sira = np.argsort(merged, kind="mergesort")
    kovalar = []
    kapsanan = 0
    for idx in np.array_split(sira, n_kova):
        n = int(idx.size)
        oran = float(np.mean(y[idx]))
        alt, ust = float(np.mean(p0[idx])), float(np.mean(p1[idx]))
        se = math.sqrt(max(oran * (1.0 - oran), 0.0) / n)
        kapsadi = bool(alt - 1.96 * se <= oran <= ust + 1.96 * se)
        kapsanan += n if kapsadi else 0
        kovalar.append({"n": n, "oran": _round(oran), "p0_ort": _round(alt),
                        "p1_ort": _round(ust), "kapsadi": kapsadi})
    out["kova"] = kovalar
    out["kapsama_payi"] = _round(kapsanan / arr.shape[0])
    return out


def _fit_window(history: list[tuple[float, int]], window: int | None,
                method: str) -> tuple[dict | None, float]:
    """Geçmişten (yalnız GEÇMİŞ) kalibratör + taban oran. `window` verilirse son `window`
    örnek kullanılır (kayan pencere = rejim adaptasyonu), yoksa tüm geçmiş (genişleyen).

    Kalibratör fit edilemezse (tek sınıf / <2 örnek) `None` döner → `apply_calibration`
    geçiş-şeffaftır, yani o blokta adaptif sürüm HAM p'ye eşittir (sessiz degrade, çökme yok).
    Taban oran boş geçmişte 0.5'tir (maksimum belirsizlik). `venn_abers` için dönen şey
    IVAP tablosudur (`venn_abers_fit`)."""
    rows = history[-window:] if window else history
    base = (sum(y for _, y in rows) / len(rows)) if rows else 0.5
    if method == VA_METHOD:
        return venn_abers_fit(rows), base
    return learning.fit_calibrator(rows, method=method), base


def prequential_eval(source: str, method: str = "platt", *, warmup: int = DEFAULT_WARMUP,
                     refit_every: int = DEFAULT_REFIT_EVERY, window: int | None = DEFAULT_WINDOW,
                     tail_n: int = DEFAULT_TAIL_N,
                     min_eval: int = DEFAULT_MIN_EVAL) -> dict[str, Any]:
    """Kaynağı YÜRÜYEN (walk-forward) kalibrasyonla değerlendir; ham/adaptif/TABAN üçlüsünü
    TAM AYNI örneklerde puanla.

    Kesin nedensellik: `i`. bloğu puanlayan kalibratör YALNIZ `triples[:i]`'den fit edilir
    (kayan pencerede `triples[i-window:i]`). Hiçbir blok kendi fit'ine katılmaz.

    Yeterlilik kapısı: puanlanan örnek `min_eval`'in altındaysa `durum="YETERSIZ"` — hiçbir
    verdikt/kalibratör üretilmez (ör. `neural` bugün n=6).

    Dönen dict (`durum="OLCULDU"`): kaynak, method, window, refit_every, n_toplam, n_warmup,
    n_eval, n_refit, eval_range, ham/adaptif/taban karneleri, kazandi (bool), taban_yener
    (bool — ACI DERS #9 kontrolü), verdict (Türkçe), calibrator (TÜM veriden fit, kalıcılık
    için — walk-forward puanına DAHİL DEĞİL)."""
    if refit_every < 1:
        raise ValueError(f"refit_every >= 1 olmalı (verilen: {refit_every})")
    if window is not None and window < 2:
        raise ValueError(f"window None ya da >= 2 olmalı (verilen: {window})")

    triples = calibration_loop._load_pairs(source, tail_n)
    triples.sort(key=lambda t: t[0])     # ZAMANA GÖRE — zaman serisi, rastgele bölme YASAK
    n_total = len(triples)

    base_out: dict[str, Any] = {
        "kaynak": source, "method": method, "window": window, "refit_every": refit_every,
        "warmup": warmup, "n_toplam": n_total,
    }

    ham: list[tuple[float, int]] = []
    adaptif: list[tuple[float, int]] = []
    taban: list[tuple[float, int]] = []
    eval_rows: list[datetime] = []
    va_rows: list[tuple[float, float, int, float]] = []   # yalnız venn_abers: (p0, p1, y, taban)
    n_refit = 0

    history = [(p, y) for _, p, y in triples[:warmup]]
    i = warmup
    while i < n_total:
        calib, base_rate = _fit_window(history, window, method)
        n_refit += 1
        block = triples[i:i + refit_every]
        ivs = (venn_abers_predict(calib, [p for _, p, _ in block])
               if method == VA_METHOD else None)
        for k, (ts, p, y) in enumerate(block):
            ham.append((p, y))
            if ivs is None:
                adaptif.append((learning.apply_calibration(p, calib), y))
            elif ivs[k] is None:
                adaptif.append((p, y))            # tablo kurulamadı → geçiş-şeffaf (ham p)
            else:
                p0, p1 = ivs[k]                   # type: ignore[misc]
                adaptif.append((venn_abers_merge(p0, p1), y))
                va_rows.append((p0, p1, y, base_rate))
            taban.append((base_rate, y))
            eval_rows.append(ts)
        history.extend((p, y) for _, p, y in block)
        i += refit_every

    n_eval = len(ham)
    out = {**base_out, "n_warmup": min(warmup, n_total), "n_eval": n_eval, "n_refit": n_refit}

    if n_eval < min_eval:
        gerekce = f"n_eval={n_eval} < min_eval={min_eval}"
        return {**out, "durum": "YETERSIZ", "gerekce": gerekce,
                "verdict": f"[{source}/{method}] YETERSİZ örneklem ({gerekce}) — yürüyen "
                           "kalibrasyon ölçülmedi, hiçbir iyileşme iddiası yok."}

    s_ham, s_ad, s_tb = _score(ham), _score(adaptif), _score(taban)

    def _better(a: float | None, b: float | None) -> bool:
        return a is not None and b is not None and a < b

    # Ham'a karşı: kalibrasyon HİJYENİ → hem Brier hem ECE düzelmeli (calibration_loop kapısı).
    ham_yener = _better(s_ad["brier"], s_ham["brier"]) and _better(s_ad["ece"], s_ham["ece"])
    # Tabana karşı: BİLGİ var mı → eşleştirilmiş Brier farkı ANLAMLI ölçüde negatif olmalı.
    taban_fark, taban_t = _paired_brier_t(adaptif, taban)
    taban_yener = bool(taban_t is not None and taban_t < -T_ESIK)
    kazandi = bool(ham_yener and taban_yener)

    if kazandi:
        verdict = (f"[{source}/{method}] ADAPTİF KAZANDI: Brier {_fmt(s_ham['brier'])}(ham)/"
                   f"{_fmt(s_tb['brier'])}(TABAN)→{_fmt(s_ad['brier'])} · ECE "
                   f"{_fmt(s_ham['ece'])}→{_fmt(s_ad['ece'])} · sabit tabana karşı eşleştirilmiş "
                   f"Brier farkı {_fmt(taban_fark, '+.5f')} (t={_fmt(taban_t, '.2f')}) "
                   f"— n_eval={n_eval}, {n_refit} yeniden-fit. SABİT TABANI DA yendi.")
    elif ham_yener:
        verdict = (f"[{source}/{method}] ham'ı yeniyor ama SABİT TABANI YENMİYOR: eşleştirilmiş "
                   f"Brier farkı {_fmt(taban_fark, '+.5f')} (t={_fmt(taban_t, '.2f')}, eşik "
                   f"−{T_ESIK}) — ACI DERS #9: kazanç bilgi değil, taban orana BÜZÜLME "
                   f"(p_std {_fmt(s_ham['p_std'])}→{_fmt(s_ad['p_std'])}). Kalibratör önerilmez.")
    else:
        verdict = (f"[{source}/{method}] yürüyen kalibrasyon iyileşme sağlamadı: Brier "
                   f"{_fmt(s_ham['brier'])}→{_fmt(s_ad['brier'])}, ECE {_fmt(s_ham['ece'])}→"
                   f"{_fmt(s_ad['ece'])} (taban {_fmt(s_tb['brier'])}/{_fmt(s_tb['ece'])}). "
                   "Kalibratör önerilmez.")

    tum = [(p, y) for _, p, y in triples]
    sonuc: dict[str, Any] = {
        **out, "durum": "OLCULDU",
        "eval_range": [eval_rows[0].isoformat(), eval_rows[-1].isoformat()],
        "ham": s_ham, "adaptif": s_ad, "taban": s_tb,
        "taban_brier_fark": _round(taban_fark, 5), "taban_t": _round(taban_t, 2),
        "ham_yener": ham_yener, "taban_yener": taban_yener, "kazandi": kazandi,
        "verdict": verdict,
        # Kalıcılık adayı: TÜM veriden fit (bugünden sonrası için "güncel" eşleme).
        # Walk-forward puanına DAHİL DEĞİL — yalnız kapı geçilirse kaydedilir.
        "calibrator": (venn_abers_fit(tum) if method == VA_METHOD
                       else learning.fit_calibrator(tum, method=method))}
    if method == VA_METHOD:
        teshis = _venn_abers_teshis(va_rows)
        sonuc["venn_abers"] = teshis
        if teshis is not None:
            sonuc["verdict"] = (
                f"{verdict} · VA aralığı: ort. genişlik {_fmt(teshis['genislik_ort'])}, "
                f"0,5'i kapsayan (nötr) pay {_fmt(teshis['notr_payi'], '.1%')}, tabanı kapsayan pay "
                f"{_fmt(teshis['taban_kapsayan_payi'], '.1%')}, kova kapsaması "
                f"{_fmt(teshis['kapsama_payi'], '.1%')} (değiştirilebilirlik varsayımı zaman "
                "serisinde SAĞLANMAZ — kapsama ölçümdür, garanti değil).")
    return sonuc


def _save_artifact(data: dict[str, Any]) -> None:
    """Artefaktı atomik yaz (OneDrive kilidi — learning._save deseni). Hata → sessiz-degrade."""
    try:
        atomic_write_text(ARTIFACT_FILE, json.dumps(data, ensure_ascii=False, indent=1))
    except OSError as e:
        logger.warning(f"[ai.adaptive_calibration] artefakt yazılamadı ({ARTIFACT_FILE.name}): {e}")


def load_artifact() -> dict[str, Any]:
    """Kayıtlı artefakt; dosya yok/bozuk → `{}` (ASLA çökme, learning.load_state deseni)."""
    try:
        raw = json.loads(ARTIFACT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def refresh(sources: tuple[str, ...] = DEFAULT_SOURCES, methods: tuple[str, ...] = DEFAULT_METHODS,
            *, warmup: int = DEFAULT_WARMUP, refit_every: int = DEFAULT_REFIT_EVERY,
            window: int | None = DEFAULT_WINDOW, tail_n: int = DEFAULT_TAIL_N,
            min_eval: int = DEFAULT_MIN_EVAL) -> dict[str, Any]:
    """Her kaynak × method'u yürüyen değerlendirmeden geçir; SADECE `kazandi` (ham'ı Brier+ECE'de
    VE tabanı anlamlı Brier farkıyla yenen) kaydedilir. Birden fazla method geçerse en düşük
    adaptif Brier seçilir.

    ⚠️ Bu kapı `calibration_loop.refresh`'ten TEK farkla ayrılır ve o fark bilerek konmuştur:
    **sabit taban-oran tahmincisi de yenilmek zorundadır.** Tur-25'te bu kontrol elle yapıldı
    ve `pwin`'in "iyileşmesinin" neredeyse tamamının büzülme olduğunu gösterdi; kapıda
    olmadığı için artefakt yine de "KAYITLI" yazdı. Burada kapının parçası.
    Yine de bu bir KENAR kanıtı DEĞİLDİR (ACI DERS #3) — yalnız kalibrasyon hijyeni."""
    artifact = load_artifact()
    for source in sources:
        results = [prequential_eval(source, m, warmup=warmup, refit_every=refit_every,
                                    window=window, tail_n=tail_n, min_eval=min_eval)
                   for m in methods]
        measured = [r for r in results if r["durum"] == "OLCULDU"]
        passed = [r for r in measured if r["kazandi"]]
        updated_at = _utc_now_iso()
        if not measured:
            entry: dict[str, Any] = {
                "kaynak": source, "durum": "YETERSIZ", "updated_at": updated_at,
                "denemeler": results,
                "not": results[0].get("gerekce") if results else "predictions_log yok/boş"}
        elif not passed:
            entry = {"kaynak": source, "durum": "KAPI_GECILEMEDI", "updated_at": updated_at,
                     "denemeler": [{k: v for k, v in r.items() if k != "calibrator"}
                                   for r in measured],
                     "not": "hiçbir method HEM ham'ı HEM SABİT TABANI (Brier ve ECE) yenmedi "
                            "— kalibratör kaydedilmedi (ACI DERS #9 kontrolü kapıda)."}
        else:
            best = min(passed, key=lambda r: r["adaptif"]["brier"])
            entry = {**{k: v for k, v in best.items() if k != "durum"},
                     "durum": "KAYITLI", "updated_at": updated_at,
                     "denemeler": [{k: v for k, v in r.items() if k != "calibrator"}
                                   for r in measured]}
        artifact[source] = entry
    _save_artifact(artifact)
    return artifact


def calibrate(source: str, p: float) -> float:
    """KARAR-DIŞI: hiçbir CANLI karar yolundan çağrılmaz (anayasa) — yalnız rapor/gölge.
    Kayıtlı (`durum='KAYITLI'`) kalibratörü uygular; kayıt yoksa/kapıyı geçmemişse `p` AYNEN
    döner (`learning.apply_calibration`'ın geçiş-şeffaf sözleşmesi)."""
    entry = load_artifact().get(source)
    calib = entry.get("calibrator") if isinstance(entry, dict) and entry.get("durum") == "KAYITLI" else None
    if isinstance(calib, dict) and calib.get("method") == VA_METHOD:
        iv = venn_abers_predict(calib, [p])[0]
        return venn_abers_merge(*iv) if iv is not None else learning.apply_calibration(p, None)
    return learning.apply_calibration(p, calib)


def calibrate_interval(source: str, p: float) -> dict | None:
    """KARAR-DIŞI: kayıtlı Venn–Abers kalibratörü varsa [p0, p1] + birleşik p + `notr`
    (aralık 0,5'i kapsıyor). Kayıt yoksa, kapıyı geçmemişse ya da kayıtlı yöntem VA değilse
    None — nokta kalibratörden uydurma bir aralık TÜRETİLMEZ."""
    entry = load_artifact().get(source)
    if not (isinstance(entry, dict) and entry.get("durum") == "KAYITLI"):
        return None
    calib = entry.get("calibrator")
    if not (isinstance(calib, dict) and calib.get("method") == VA_METHOD):
        return None
    iv = venn_abers_predict(calib, [p])[0]
    if iv is None:
        return None
    p0, p1 = iv
    return {"p0": p0, "p1": p1, "p": venn_abers_merge(p0, p1), "genislik": p1 - p0,
            "notr": bool(p0 <= 0.5 <= p1)}


def report() -> str:
    """Her kaynak için tek blok: ham vs adaptif vs TABAN. Artefakt yoksa dürüstçe söyler."""
    lines = ["[Yürüyen kalibrasyon] (KARAR-DIŞI — yalnız rapor; canlı karara bağlı DEĞİL)"]
    art = load_artifact()
    if not art:
        lines.append("  ölçülemedi — data/ai_adaptive_calibration.json yok/boş (--eval koşun).")
        return "\n".join(lines)
    for source in sorted(art):
        e = art[source]
        if not isinstance(e, dict):
            continue
        durum = e.get("durum", "?")
        if durum in ("KAYITLI", "KAPI_GECILEMEDI"):
            best = e if durum == "KAYITLI" else (e.get("denemeler") or [{}])[0]
            h, a, t = best.get("ham", {}), best.get("adaptif", {}), best.get("taban", {})
            # Tur-106 (BULGU-106-13): ölçülen dönem yazılır — kaynak kesilince (quant geçişi) gece
            # koşusu 60k satırlık kuyruktaki ESKİ dilimi her gün "güncel" gibi yeniden raporluyordu.
            ar = best.get("eval_range") or [None, None]
            lines.append(f"  {source}: {durum} method={best.get('method')} "
                         f"n_eval={best.get('n_eval')} pencere={best.get('window') or 'genişleyen'} "
                         f"dönem={str(ar[0])[:10]}→{str(ar[-1])[:10]}")
            lines.append(f"      Brier  ham {_fmt(h.get('brier'))} · TABAN {_fmt(t.get('brier'))} "
                         f"· adaptif {_fmt(a.get('brier'))}")
            lines.append(f"      ECE    ham {_fmt(h.get('ece'))} · TABAN {_fmt(t.get('ece'))} "
                         f"· adaptif {_fmt(a.get('ece'))}")
            lines.append(f"      p_std  ham {_fmt(h.get('p_std'))} → adaptif "
                         f"{_fmt(a.get('p_std'))}  (çökmüşse = taban orana büzülme)")
            lines.append(f"      TABAN'a karşı eşleştirilmiş Brier farkı "
                         f"{_fmt(best.get('taban_brier_fark'), '+.5f')} "
                         f"(t={_fmt(best.get('taban_t'), '.2f')}, geçmek için t<−{T_ESIK})")
            lines.append(f"      {best.get('verdict', '')}")
            for d in e.get("denemeler") or []:
                va = d.get("venn_abers") if isinstance(d, dict) else None
                if isinstance(va, dict):
                    lines.append(f"      Venn–Abers: ort. aralık genişliği {_fmt(va.get('genislik_ort'))} · "
                                 f"0,5'i kapsayan (nötr) pay {_fmt(va.get('notr_payi'), '.1%')} · "
                                 f"kova kapsaması {_fmt(va.get('kapsama_payi'), '.1%')}")
        elif durum == "YETERSIZ":
            lines.append(f"  {source}: YETERSİZ örneklem — ölçülemedi ({e.get('not', '?')}).")
        else:
            lines.append(f"  {source}: {durum}")
    lines.append("  Not: bu kalibrasyon HİJYENİDİR; kenar kanıtı DEĞİLDİR (ACI DERS #3).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI: `--eval` (yürüyen değerlendirme + kapıyı geçeni kaydet) · `--report` · `--json` ·
    `--source`/`--method` · `--window` (kayan pencere; yoksa genişleyen) · `--warmup` ·
    `--refit-every` · `--tail` · `--min-eval`."""
    import argparse
    ap = argparse.ArgumentParser(
        description="Yürüyen (walk-forward) AI kalibrasyonu — taban-oran kontrolü kapıda (KARAR-DIŞI)")
    ap.add_argument("--eval", action="store_true", help="yürüyen değerlendirme koş + kapıyı geçeni kaydet")
    ap.add_argument("--report", action="store_true", help="kayıtlı sonuçların Türkçe özeti")
    ap.add_argument("--source", action="append", choices=sorted(DEFAULT_SOURCES))
    ap.add_argument("--method", action="append", choices=sorted(DEFAULT_METHODS))
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help="kayan pencere boyu (örnek); verilmezse genişleyen geçmiş")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--refit-every", type=int, default=DEFAULT_REFIT_EVERY, dest="refit_every")
    ap.add_argument("--tail", type=int, default=DEFAULT_TAIL_N, help="predictions_log kuyruğu (ham satır)")
    ap.add_argument("--min-eval", type=int, default=DEFAULT_MIN_EVAL, dest="min_eval")
    ap.add_argument("--json", action="store_true", help="çıktıyı JSON bas")
    args = ap.parse_args(argv)

    if not (args.eval or args.report):
        ap.print_help()
        return 0

    if args.eval:
        refresh(sources=tuple(args.source) if args.source else DEFAULT_SOURCES,
                methods=tuple(args.method) if args.method else DEFAULT_METHODS,
                warmup=args.warmup, refit_every=args.refit_every, window=args.window,
                tail_n=args.tail, min_eval=args.min_eval)
    if args.json:
        print(json.dumps(load_artifact(), ensure_ascii=False, indent=2, default=str))
    else:
        print(report())
    return 0


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    raise SystemExit(main())
