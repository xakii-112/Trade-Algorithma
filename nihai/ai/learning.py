"""
learning.py — ÖĞRENME döngüsü (FAZ AI v2). KARAR-DIŞI / gölge.

Çözülen canlı tahminlerden (predictions_log.csv, OKUMA-YALNIZ) sürekli öğrenir:
  - Kalibrasyon: Brier skoru + güvenilirlik kovaları (meta p_win → gerçekleşen taraf-kazanma).
  - Prior'lar: Beta-Binomial posterior (uniform prior; %95 CI) — kanıt azken 0.5'e küçülür,
    kenar iddiasını asla şişirmez.
  - Drift: yuvarlanan isabet serisinde iki-yönlü Page-Hinkley (rejim/veri kayması uyarısı).

ŞEMA GERÇEĞİ (OP-6): predictions_log'da 'strategy' kolonu YOK; kayıtlar yön (direction)
düzeyindedir. Bu yüzden prior anahtarları "GLOBAL" / "LONG" / "SHORT"tur; decision katmanı
strateji-adı eşleşmesi bulamazsa GLOBAL'e, o da yoksa nötr prior'a düşer.

Çıktı `data/ai_learning_state.json`'a ATOMİK yazılır; hiçbir üretim eşiği/sinyali değişmez.
Rapor metinleri İZLEME dilindedir — kenar kanıtı iması taşımaz (ACI DERS #3/#4).

v3 EKLERİ (Tur-16, WP-L — geriye-uyumlu, mevcut API/davranış bozulmaz):
  - `state["sources"]`: decision v3'ün hiyerarşik birleşimi için kaynak-başına Brier
    (prior/pwin/neural) — bkz. `update_from_predictions` alt-bölümü. Alan yoksa (eski
    kayıtlı state.json) okuyanlar degrade etmelidir (bu modül zorlamaz).
  - `fit_calibrator`/`apply_calibration`: Platt/izotonik kalibrasyon, YALNIZ çağıranın
    verdiği (OOS) çiftlerde fit edilir — bu modül kendi training/OOS ayrımını YAPMAZ.
  - `update_from_predictions` dönüşüne `retrain_candidate: bool` — Page-Hinkley
    tetiklendiyse True; otomatik yeniden-eğitim TETİKLEMEZ, yalnız bayrak+rapor satırı.
  - Bu modül `research/`'e import VEREMEZ (üretim→research bağımlılığı yasak); replay
    buffer/live-log join'i `research/train_neural.py`'dedir (SPEC §3.3).

A1 EKİ (Tur-25/WP-A, 2026-08-10 — geriye-uyumlu EK, `state["strategies"]` DOKUNULMADAN
kalır): ÖLÇÜLMÜŞ PROBLEM — `update_from_predictions`'ın sayımı zaman- AĞIRLIKSIZDIR: 5000
satırlık kuyrukta HAFTALAR öncesi bir kayıt DÜNKÜ kayıtla eşit ağırlıklıdır. `docs/
AI_DENETIM_RAPORU_2026-07-26.md` §E.4 bunun CANLI ZARARA yol açtığını ölçtü: karar motoru
bayat bir SHORT üstünlüğünü (geçmiş DOWN rejiminin artefaktı) prior alıp yükselen piyasada
SHORT girip kaybetti (net −%0.50, taban-göre −1.75 puan) — "ACI DERS #1'in doğrudan ampirik
doğrulaması". `page_hinkley` drift'i tespit ediyordu ama hiçbir tüketici yoktu. Bu ek:
  - `state["strategies_decayed"]`: `strategies` ile AYNI şema + `n_ham`/`half_life_days`/
    `drift_shrink_uygulandi`; üstel-unutmalı (yarılanma ömrü `DECAY_HALF_LIFE_DAYS`, env ile
    taşınır) ETKİN (float) sayımlardan üretilir. Referans zaman kuyruktaki EN SON parse
    edilebilen damgadır ("bugün" DEĞİL) → aynı log HER ZAMAN aynı posterior'u verir
    (determinizm). Drift tetiklendiyse etkin sayımlar `DRIFT_SHRINK`'e bölünür (posterior
    HER ZAMAN 0.5'e doğru genişler/güven düşer — asla tersi).
  - `state["n_unparsed"]`: zaman damgası (bar_ts→logged_at→resolved_at hiçbiri) ayrıştırıla-
    mayan satır sayısı — bu satırlar ağırlıklı istatistiğe girmez, sessizce yutulur.
  - `beta_posterior` artık REEL (float) `wins`/`losses` kabul eder (mevcut int çağrılar
    bit-denk kalır) — Beta dağılımı α,β için reel değer matematiksel olarak geçerlidir.
  - Tüketici: `decision._prior_from_state`, YALNIZ `AI_PRIOR_DECAY` env'i açıkken okur;
    kapalıyken (varsayılan) davranış BİT-DENKTİR.
"""
from __future__ import annotations

import csv
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger

from nihai.atomic_io import atomic_write_text, tail_lines
from nihai.config import DATA_DIR

# Modül-düzeyi yollar: testler monkeypatch'ler (reasoner/memory deseni).
LEARNING_STATE = DATA_DIR / "ai_learning_state.json"
PRED_LOG = DATA_DIR / "predictions_log.csv"          # OKUMA-YALNIZ

# Drift tespiti: ham 0/1 isabetleri bu pencerede yuvarlanan-ortalamaya çevirip PH uygula
# (ham ikili seride ±0.5'lik tek sapma lam eşiğini anlamsız kılar; ortalama seri gerekir).
ROLL_WINDOW = 50

# --- A1 (Tur-25/WP-A): zaman-ağırlıklı (üstel unutmalı) posterior sabitleri ----------------
# Yarılanma ömrü (gün): bir kaydın ağırlığı bu kadar gün sonra YARIYA iner. ~1 işlem ayı;
# rejim kalıcılığının (ACI DERS #1/#7'deki "rejim artefaktı" gözlemleri) tipik mertebesi
# budur. Sabit KEYFİDİR — bu yüzden AI_PRIOR_HALF_LIFE_DAYS env'iyle ÇAĞRI-ANINDA (import-
# anında DEĞİL; bkz. `_resolve_half_life_days`, decision.py `_resolve_utility_lambda`
# deseniyle aynı) geçersiz kılınabilir: testler/araştırma canlı koda dokunmadan farklı
# ömürleri dener.
AI_PRIOR_HALF_LIFE_DAYS_ENV = "AI_PRIOR_HALF_LIFE_DAYS"
DECAY_HALF_LIFE_DAYS: float = 30.0

# Page-Hinkley drift tetiklendiğinde ağırlıklı win/loss sayımları bu değere BÖLÜNÜR.
# İLKE: drift = geçmişin daha az bilgilendirici olduğu anlamına gelir; tepki HER ZAMAN
# DAHA AZ güven olmalı (posterior 0.5'e doğru genişler, CI açılır) — asla daha fazla
# güven/daha dar CI değil.
DRIFT_SHRINK: float = 2.0

# --- Tur-35: KAYNAK FÜZYONU (Tur-32 açık kalem #4 — iki tur ertelendi) --------------------
# Sistemin iki bağımsız olasılık kaynağı var (`pwin` meta-model, `neural` gölge ağı) ve
# `decision.py` onları Brier'den türeyen SABİT ağırlıklarla harmanlıyor. Sabit ağırlık iki
# şeyi birden varsayar: (1) kaynakların göreli kalitesi ZAMANLA değişmez, (2) kalite
# ölçümü tüm geçmişe eşit bakabilir. İkisi de rejimli bir piyasada şüphelidir.
# `online.hedge_combine` bunu VERİYE sorar: ağırlıklar her adımda o adıma KADARKİ kayıptan
# gelir (prequential — sızıntı yok) ve fixed-share rejim değişiminde ölü uzmanı diriltir.
# Ölçüm KARAR-DIŞIDIR: hiçbir ağırlık üretime bağlanmaz, yalnız rapora yazılır.
FUSION_ETA: float = 1.0
FUSION_SHARE: float = 0.02      # rejim değişiminde ağırlığın %2'si her adımda geri dağıtılır
FUSION_MIN_N: int = 200         # altında dürüstçe "YETERSİZ" (uydurma verdikt yok)
FUSION_T_THRESHOLD: float = -1.645   # Tur-24/27/28 ile AYNI eşik (tek-yönlü %5)


def _resolve_half_life_days() -> float:
    """AI_PRIOR_HALF_LIFE_DAYS env'ini ÇAĞRI ANINDA oku (decision.py `_resolve_utility_lambda`
    deseniyle aynı felsefe: import-anında DEĞİL — testler monkeypatch.setenv ile güvenle
    değiştirebilir). Boş/bozuk/NaN/≤0 → varsayılan DECAY_HALF_LIFE_DAYS (savunmacı; asla
    çökme/istisna)."""
    raw = os.environ.get(AI_PRIOR_HALF_LIFE_DAYS_ENV)
    if raw is None or not raw.strip():
        return DECAY_HALF_LIFE_DAYS
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return DECAY_HALF_LIFE_DAYS
    return v if (v == v and v > 0) else DECAY_HALF_LIFE_DAYS   # NaN + pozitiflik guard'ı


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _f(x: object) -> float | None:
    """CSV alanı → float; boş/bozuk/NaN → None (reasoner deseni)."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _norm_dir(s: str) -> str:
    """Yön alanını kanonikleştir: canlı logda direction '+'/'-' yazılır (pwin_side ise
    LONG/SHORT). Karşılaştırma/gruplama ÖNCE bu kanonik forma çevrilir; bilinmeyen → ''."""
    u = s.strip().upper()
    if u in ("+", "LONG", "UP", "BUY"):
        return "LONG"
    if u in ("-", "SHORT", "DOWN", "SELL"):
        return "SHORT"
    return ""


# ---------------------------------------------------------------------------
# A1: zaman-ağırlıklı posterior — tz-kanonikleştirme + yaş-ağırlığı yardımcıları.
# Modül-düzeyinde (nested DEĞİL) tanımlanır ki doğrudan test edilebilsinler.
# ---------------------------------------------------------------------------
def _parse_ts_naive_utc(s: str) -> datetime | None:
    """ISO zaman damgasını naive-UTC'ye kanonikleştirir: tz-aware ise UTC'ye çevirip tzinfo
    düşürülür; tz'siz ise OLDUĞU GİBİ bırakılır (proje sözleşmesi: naive = zaten UTC — bkz.
    `nihai/execution/prediction_tracker.py:_canon_ts`, `episodic.py` OP-1 dersi). Canlı log
    `bar_ts`'i KARIŞIK biçimde gelir (bazı satırlar "-04:00" gibi bir ofsetle, bazıları
    tz'siz) — bu fonksiyondan SONRA hepsi AYNI (naive-UTC) formda olur; BULGU-16'nın
    naive-LOCAL vs naive-UTC karışıklığı burada TEKRARLANMAZ çünkü iki ayrıştırılmış
    datetime ASLA kanonikleştirilmeden karşılaştırılmaz. Boş/ayrıştırılamaz → None
    (sessiz-yutma; çağıran `n_unparsed` sayacını artırır)."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.strip())
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _resolve_row_ts(bar_ts: str, logged_at: str, resolved_at: str) -> datetime | None:
    """Satırın "karar zamanı" damgası: sırayla `bar_ts` (karar barı — DOĞRU semantik) →
    `logged_at` → `resolved_at` dener; ilk ayrıştırılabilen kazanır. Hiçbiri ayrıştırılamazsa
    None (satır ağırlıklı istatistiğe GİRMEZ)."""
    for raw in (bar_ts, logged_at, resolved_at):
        dt = _parse_ts_naive_utc(raw)
        if dt is not None:
            return dt
    return None


def _age_weight(t_i: datetime, ref: datetime, half_life_days: float) -> float:
    """Üstel-unutma ağırlığı: w = 0.5 ** (yaş_gün / yarılanma_ömrü). `ref` kuyruktaki EN SON
    parse edilebilen damga OLMALIDIR (şimdiki saat DEĞİL — bkz. modül docstring'i/A1); bu
    sözleşme altında normal akışta hiçbir t_i ref'i geçemez, ama yine de savunmacı olarak
    negatif yaş (gelecek damgası) 0'a kırpılır (ağırlık ≤ 1.0 üst sınırı korunur)."""
    age_days = max(0.0, (ref - t_i).total_seconds() / 86400.0)
    return 0.5 ** (age_days / half_life_days)


# ---------------------------------------------------------------------------
# Kalibrasyon
# ---------------------------------------------------------------------------
def brier_score(pairs: list[tuple[float, int]]) -> float | None:
    """Ortalama (p - y)^2. Geçersiz (sonlu olmayan p / y∉{0,1}) çiftler atlanır; boş → None."""
    acc = 0.0
    n = 0
    for p, y in pairs:
        pf = _f(p)
        if pf is None or y not in (0, 1):
            continue
        acc += (pf - float(y)) ** 2
        n += 1
    return (acc / n) if n else None


def paired_brier_t(a: list[tuple[float, int]], b: list[tuple[float, int]]
                   ) -> tuple[float | None, float | None]:
    """EŞLEŞTİRİLMİŞ Brier farkı (a − b) ve t istatistiği. Negatif t = `a` daha iyi.

    NEDEN EŞLEŞTİRİLMİŞ ve NEDEN Brier: iki yarışmacı TAM AYNI örnekleri puanlar, bu yüzden
    örnek-başı fark alınabilir — bu, iki bağımsız ortalamayı kıyaslamaktan çok daha güçlüdür
    (piyasa gürültüsü ikisinde de aynıdır, farkta sadeleşir). Metrik Brier'dir çünkü Brier
    kalibrasyon + AYIRT ETME taşır; ECE YALNIZ kalibrasyonu ölçer ve sabit tahminci ECE'de
    YAPISI GEREĞİ ~0'dır (hep taban oranı söyleyen bir tahminci kusursuz kalibredir) — yani
    "tabanı ECE'de yen" demek, hiçbir bilgili modelin geçemeyeceği bir kategori hatasıdır.
    "Bilgi var mı?" sorusunun doğru metriği Brier'dir.

    ACI DERS #9'un MAKİNE hâli: bir kalibrasyon iyileşmesi raporlanmadan önce SABİT-TABAN
    tahmincisiyle kıyaslanmalıdır (Tur-25'te bu elle yapıldı ve "iyileşmenin" neredeyse
    tamamının taban orana BÜZÜLME olduğunu gösterdi). `calibration_loop` ve
    `adaptive_calibration` kapıları bunu buradan çağırır — tek doğruluk kaynağı.

    <2 geçerli çift veya sıfır varyans → (fark, None): anlamlılık İDDİA EDİLMEZ."""
    d = [((pa - ya) ** 2 - (pb - yb) ** 2) for (pa, ya), (pb, yb) in zip(a, b, strict=True)]
    n = len(d)
    if n < 2:
        return (None, None)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    if var <= 0.0:
        return (mean, None)
    return (mean, mean / ((var / n) ** 0.5))


def expected_calibration_error(pairs: list[tuple[float, int]], n_bins: int = 10) -> float | None:
    """ECE (Naeini ve ark. 2015): `reliability_bins` kovalarında |p_mean − y_rate|'in
    örneklem-ağırlıklı ortalaması. Boş kovalar katkı vermez; hiç geçerli çift yoksa None.

    NEDEN BRIER'İN YANINDA: Brier = kalibrasyon + AYIRT ETME (refinement) karışımıdır;
    ayırt etmesi sıfır ama kusursuz kalibre bir model (hep taban oranını söyleyen) düşük
    Brier alabilir, ayırt eden ama aşırı-güvenli bir model yüksek. ECE YALNIZ kalibrasyonu
    ölçer — bu projede kritik, çünkü ölçülen OOS-AUC ~0,50 iken (ayırt etme yok) tek dürüst
    ilerleme ekseni kalibrasyondur (Tur-20 dersi: kazanç kalibrasyon hizasındaydı).

    `reliability_bins` ile AYNI kova tanımını kullanır (tek doğruluk kaynağı) — panodaki
    eğri ile tek-sayılı özet birbirini tutar."""
    bins = reliability_bins(pairs, n_bins=n_bins)
    total_n = sum(int(b["n"]) for b in bins)
    if total_n <= 0:
        return None
    acc = 0.0
    for b in bins:
        n = int(b["n"])
        if n <= 0 or b["p_mean"] is None or b["y_rate"] is None:
            continue
        acc += (n / total_n) * abs(float(b["p_mean"]) - float(b["y_rate"]))
    return acc


def reliability_bins(pairs: list[tuple[float, int]], n_bins: int = 10) -> list[dict]:
    """Eşit-genişlik güvenilirlik kovaları: [{'lo','hi','n','p_mean','y_rate'}]. Boş kova
    n=0, p_mean/y_rate=None ile döner (kalibrasyon eğrisi çizimi için sabit ızgara)."""
    n_bins = max(1, int(n_bins))
    sums = [[0.0, 0, 0] for _ in range(n_bins)]      # [p_toplam, y_toplam, n]
    for p, y in pairs:
        pf = _f(p)
        if pf is None or y not in (0, 1):
            continue
        idx = min(n_bins - 1, max(0, int(pf * n_bins)))
        sums[idx][0] += pf
        sums[idx][1] += int(y)
        sums[idx][2] += 1
    out: list[dict] = []
    for i, (ps, ys, n) in enumerate(sums):
        out.append({
            "lo": round(i / n_bins, 3), "hi": round((i + 1) / n_bins, 3), "n": n,
            "p_mean": (round(ps / n, 4) if n else None),
            "y_rate": (round(ys / n, 4) if n else None),
        })
    return out


def log_loss_score(pairs: list[tuple[float, int]]) -> float | None:
    """Ortalama log-kayıp (çapraz entropi) = −ort(y·ln p + (1−y)·ln(1−p)), p ∈ [1e-6, 1−1e-6].

    Brier'in yanında NEDEN: log-kayıp da uygun (proper) bir skordur ama HATALI KESİNLİĞİ çok
    daha sert cezalandırır (p→0 iken y=1 gelirse ceza sınırsıza gider, Brier'de en fazla 1).
    Aşırı-güvenli bir modelin Brier'i idare edilebilir görünürken log-kaybı patlar — bu
    projede pwin'in "ağır fazla-güvenli" olduğu Tur-25'te tespit edilmişti, log-kayıp bunu
    tek sayıda gösterir. Bilgisiz taban: ln2 ≈ 0,693. Boş → None."""
    acc = 0.0
    n = 0
    for p, y in pairs:
        pf = _f(p)
        if pf is None or y not in (0, 1):
            continue
        pc = min(max(pf, 1e-6), 1.0 - 1e-6)
        acc += -(float(y) * math.log(pc) + (1.0 - float(y)) * math.log(1.0 - pc))
        n += 1
    return (acc / n) if n else None


def brier_skill_score(pairs: list[tuple[float, int]], base_rate: float | None = None) -> float | None:
    """BSS = 1 − Brier(model)/Brier(taban) — SABİT taban-oran tahmincisine göre BECERİ.

    ACI DERS #9'un tek-sayılık özeti: BSS ≤ 0 ise model, "hep taban oranını söyle" kuralından
    daha iyi DEĞİLDİR — Brier'in ne kadar düştüğü önemsizdir. Bu proje bugüne kadar bu kıyası
    kapılarda (paired_brier_t) yapıyordu ama RAPORLANAN tek bir beceri sayısı yoktu; "Brier
    0,25'ten 0,249'a düştü" satırı okuyucuyu yanıltmaya devam ediyordu.

    `base_rate` verilmezse ÖRNEKLEMİN KENDİ taban oranı kullanılır (in-sample taban; sızıntı
    değildir çünkü sabit tahminci hiçbir şey öğrenmez ve tahminleri değiştirmez — Tur-31'in
    dsr kapısında verilen aynı gerekçe). Taban Brier'i 0 ise (tek sınıflı örneklem) → None.
    Boş → None."""
    valid = [(pf, int(y)) for p, y in pairs
             if (pf := _f(p)) is not None and y in (0, 1)]
    if not valid:
        return None
    if base_rate is None:
        base = sum(y for _, y in valid) / len(valid)
    else:
        bf = _f(base_rate)
        if bf is None:
            return None
        base = bf
    model_b = brier_score(valid)
    base_b = brier_score([(base, y) for _, y in valid])
    if model_b is None or base_b is None or base_b <= 0:
        return None
    return 1.0 - model_b / base_b


def murphy_decomposition(pairs: list[tuple[float, int]], n_bins: int = 10) -> dict | None:
    """Murphy (1973) Brier ayrışması: BS ≈ GÜVENİLİRLİK − ÇÖZÜNÜRLÜK + BELİRSİZLİK.

      reliability (REL) = Σ_k (n_k/N)(p̄_k − ȳ_k)²   → KALİBRASYON hatası (küçük = iyi)
      resolution  (RES) = Σ_k (n_k/N)(ȳ_k − ȳ)²     → BİLGİ (büyük = iyi)
      uncertainty (UNC) = ȳ(1−ȳ)                    → problemin kendi zorluğu (modelden bağımsız)

    Bu, projenin en çok ihtiyaç duyduğu ayrımdır: `RES ≈ 0` demek "model taban oranın ötesinde
    HİÇBİR bilgi taşımıyor" demektir ve bunu Brier tek başına SÖYLEMEZ (iyi kalibre + sıfır
    bilgili bir tahminci düşük Brier alır — Tur-25/27/28'de tam olarak bu görüldü). REL ve RES
    yan yana okunduğunda "kalibrasyon kazancı" ile "bilgi kazancı" birbirine karışamaz.

    Ayrışma KOVA bazlıdır; kimlik ancak kova-içi tahminler sabitse TAM tutar. Kalan
    (`residual` = BS − (REL − RES + UNC)) kova-içi dağılımın payıdır ve DÜRÜSTÇE raporlanır
    (sıfır varsayılmaz). Boş/tek sınıflı → None."""
    valid = [(pf, int(y)) for p, y in pairs
             if (pf := _f(p)) is not None and y in (0, 1)]
    if not valid:
        return None
    total = len(valid)
    y_bar = sum(y for _, y in valid) / total
    bins = reliability_bins(valid, n_bins=n_bins)
    rel = res = 0.0
    for b in bins:
        n = int(b["n"])
        if n <= 0 or b["p_mean"] is None or b["y_rate"] is None:
            continue
        w = n / total
        rel += w * (float(b["p_mean"]) - float(b["y_rate"])) ** 2
        res += w * (float(b["y_rate"]) - y_bar) ** 2
    unc = y_bar * (1.0 - y_bar)
    bs = brier_score(valid)
    return {
        "brier": bs, "reliability": rel, "resolution": res, "uncertainty": unc,
        "residual": (None if bs is None else bs - (rel - res + unc)),
        "n": total, "base_rate": y_bar,
    }


def source_fusion(p_pwin: list[float], p_neural: list[float], outcomes: list[int], *,
                  eta: float = FUSION_ETA, share: float = FUSION_SHARE,
                  extra: dict[str, list[float]] | None = None) -> dict | None:
    """ÇEVRİMİÇİ kaynak füzyonu — Tur-32 açık kalem #4'ün kapanışı. KARAR-DIŞI.

    Üç dizi AYNI satırlara ait olmalıdır (hizalanmış üçlü) ve olay TEK'tir: *"canlı sinyalin
    YÖNÜ kazandı mı"*. Her kaynağın olasılığı bu ortak olaya çevrilmiş olarak verilmelidir
    (kaynak karşı tarafı söylüyorsa 1−p) — aksi hâlde uzmanlar farklı sorulara cevap verir
    ve birleşim anlamsızlaşır. Çeviri çağıranın işidir; bu fonksiyon hizalamayı VARSAYAR.

    `online.hedge_combine(include_base=True)` koşan taban oranını ÜÇÜNCÜ uzman yapar. Kapı
    iki katmanlıdır ve ikisi de geçilmelidir:
      1. `vs_base > 0`  — birleşimin log-kaybı sabit tabanınkinden düşük mü (yön kontrolü),
      2. `paired_brier_t(birleşim, taban) < FUSION_T_THRESHOLD` — fark GÜRÜLTÜ MÜ
         (Tur-24/27/28 ile AYNI t eşiği; tek doğruluk kaynağı `paired_brier_t`).

    `regret = loss_birleşim − loss_en_iyi_uzman` ayrıca raporlanır: pozitifse birleştirme
    en iyi TEK uzmandan geride kalmıştır — Hedge'in teorik garantisi "en iyi uzmana yakın
    kalmak"tır, "onu geçmek" DEĞİLDİR; bu satır o farkı okuyana hatırlatır.

    n < FUSION_MIN_N → `durum="YETERSIZ"` (uydurma verdikt yok). Geçersiz/uyumsuz girdi →
    None (quant_math deseni)."""
    # Lazy import: `online` bu modülü değil, bu modül onu kullanır — modül düzeyinde bağlamak
    # gereksiz bir yükleme zinciri kurardı (Tur-26 "import ucuz kalmalı" disiplini).
    from nihai.ai.online import hedge_combine, running_base_rate

    if not (len(p_pwin) == len(p_neural) == len(outcomes)) or not outcomes:
        return None
    # Tur-36: K uzman. `extra` sözlüğünün SIRASI korunur (py3.7+ dict) → `experts`/
    # `loss_experts`/`final_weights` listeleri aynı indekste AYNI uzmanı gösterir.
    # Hizalama sözleşmesi genişleyen her uzman için AYNEN geçerlidir: verilen olasılık
    # "canlı sinyalin YÖNÜ kazandı mı" olayına ÇEVRİLMİŞ olmalıdır (çeviri çağıranın işi).
    cols = [np.asarray(p_pwin, dtype=float), np.asarray(p_neural, dtype=float)]
    names = ["pwin", "neural"]
    for name, vals in (extra or {}).items():
        if len(vals) != len(outcomes):
            return None      # hizasız uzman = anlamsız birleşim (sessizce kırpma YOK)
        cols.append(np.asarray(vals, dtype=float))
        names.append(str(name))
    names.append("base")     # `include_base=True` koşan taban oranını SON uzman yapar
    preds = np.column_stack(cols)
    y = np.asarray(outcomes, dtype=float)
    res = hedge_combine(preds, y, eta=eta, share=share, include_base=True)
    if res is None:
        return None
    base = running_base_rate(y)
    if base is None:
        return None
    combined = np.asarray(res["combined"], dtype=float)
    y_int = [int(v) for v in np.round(y)]
    diff, t_stat = paired_brier_t(
        list(zip(combined.tolist(), y_int, strict=True)),
        list(zip(np.asarray(base, dtype=float).tolist(), y_int, strict=True)))
    n = int(res["n"])
    if n < FUSION_MIN_N:
        durum = "YETERSIZ"
    elif (res["vs_base"] or 0.0) > 0.0 and t_stat is not None and t_stat < FUSION_T_THRESHOLD:
        durum = "TABANI_GECTI"
    else:
        durum = "TABANI_GECILEMEDI"
    loss_exp = [round(float(v), 6) for v in res["loss_experts"]]
    return {
        "n": n,
        "experts": list(names),
        "eta": float(eta), "share": float(share),
        "loss_combined": round(float(res["loss_combined"]), 6),
        "loss_experts": loss_exp,
        "best_expert": names[int(res["best_expert"])],
        "regret": round(float(res["regret"]), 6),
        "vs_base": (None if res["vs_base"] is None else round(float(res["vs_base"]), 6)),
        "final_weights": [round(float(v), 4) for v in res["final_weights"]],
        "paired_brier_t": (None if t_stat is None else round(float(t_stat), 3)),
        "paired_brier_diff": (None if diff is None else round(float(diff), 6)),
        "durum": durum,
    }


def _source_card(pairs: list[tuple[float, int]]) -> dict:
    """Bir olasılık kaynağının KARNESİ — üç kapı tek yerden doldurulur (pwin/neural/prior).
    Brier (kalibrasyon+ayırt etme) · ECE (yalnız kalibrasyon) · BSS (sabit tabana karşı
    beceri) · resolution (bilgi). Son ikisi Tur-32'de eklendi: ilk ikisi ne kadar iyi
    görünürse görünsün, BSS ≤ 0 ve RES ≈ 0 ise kaynak bilgi TAŞIMIYOR demektir."""
    br = brier_score(pairs)
    ece = expected_calibration_error(pairs, n_bins=10)
    bss = brier_skill_score(pairs)
    murphy = murphy_decomposition(pairs, n_bins=10)
    return {
        "brier": (round(br, 4) if br is not None else None),
        "ece": (round(ece, 4) if ece is not None else None),
        "bss": (round(bss, 4) if bss is not None else None),
        "resolution": (round(murphy["resolution"], 5) if murphy is not None else None),
        "n": len(pairs),
    }


# ---------------------------------------------------------------------------
# v3: Kalibratör fit/uygula (Platt/izotonik) — YALNIZ çağıranın verdiği (OOS) çiftlerde.
# Bu fonksiyonlar train/test ayrımı YAPMAZ; sızıntı-güvenliği çağıranın sorumluluğudur
# (spec §3.2: "YALNIZ verilen (OOS) çiftlerde fit").
# ---------------------------------------------------------------------------
def fit_calibrator(pairs: list[tuple[float, int]], method: str = "platt") -> dict | None:
    """(p_ham, y) OOS çiftlerinden kalibratör fit eder. `method`: 'platt' (lojistik
    ölçekleme) | 'isotonic' (izotonik regresyon) — sklearn (mevcut bağımlılık, yeni
    paket YOK). Bilinmeyen `method` → ValueError (API ihlali — programcı hatası).

    Geçersiz/yetersiz VERİ (boş liste, <2 geçerli çift, tek sınıf) → None (çökme yok —
    quant_math deseni). Dönüş: {'method','n','brier_once','brier_sonra', ...katsayılar}
    — 'brier_once' ham p'nin, 'brier_sonra' kalibre edilmiş p'nin AYNI (OOS) çiftlerdeki
    Brier'i (kalibrasyonun işe yarayıp yaramadığının doğrudan kanıtı)."""
    if method not in ("platt", "isotonic"):
        raise ValueError(f"bilinmeyen method: {method!r} (geçerli: 'platt'|'isotonic')")
    valid = [(pf, int(y)) for p, y in pairs
             if (pf := _f(p)) is not None and y in (0, 1)]
    if len(valid) < 2 or len({y for _, y in valid}) < 2:
        return None
    ps = np.array([p for p, _ in valid], dtype=float)
    ys = np.array([y for _, y in valid], dtype=int)
    brier_once = brier_score(valid)
    if method == "platt":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        clf.fit(ps.reshape(-1, 1), ys)
        calibrated = clf.predict_proba(ps.reshape(-1, 1))[:, 1]
        brier_sonra = brier_score(list(zip(calibrated.tolist(), ys.tolist(), strict=True)))
        return {"method": "platt", "n": len(valid),
                "coef": float(clf.coef_[0][0]), "intercept": float(clf.intercept_[0]),
                "brier_once": (round(brier_once, 4) if brier_once is not None else None),
                "brier_sonra": (round(brier_sonra, 4) if brier_sonra is not None else None)}
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(ps, ys)
    calibrated = iso.predict(ps)
    brier_sonra = brier_score(list(zip(calibrated.tolist(), ys.tolist(), strict=True)))
    return {"method": "isotonic", "n": len(valid),
            "x_thresholds": iso.X_thresholds_.tolist(), "y_thresholds": iso.y_thresholds_.tolist(),
            "brier_once": (round(brier_once, 4) if brier_once is not None else None),
            "brier_sonra": (round(brier_sonra, 4) if brier_sonra is not None else None)}


def apply_calibration(p: float, calib: dict | None) -> float:
    """Tek bir ham olasılığa `fit_calibrator` çıktısını uygula. `calib=None` veya
    tanınmayan/eksik şema → p AYNEN döner (geçiş-şeffaf — decision.py kalibrasyon
    yokken mevcut davranışa SAYISAL DENK degrade edebilsin diye, spec §3.2/WP-D).
    `p` sayısal değilse ValueError (API ihlali — dönüş tipi non-optional float)."""
    pf = _f(p)
    if pf is None:
        raise ValueError("p sayısal (sonlu) olmalı")
    if not isinstance(calib, dict):
        return pf
    method = calib.get("method")
    if method == "platt":
        coef, intercept = calib.get("coef"), calib.get("intercept")
        if coef is None or intercept is None:
            return pf
        z = float(coef) * pf + float(intercept)
        return float(1.0 / (1.0 + np.exp(-z)))
    if method == "isotonic":
        xt, yt = calib.get("x_thresholds"), calib.get("y_thresholds")
        if not xt or not yt:
            return pf
        return float(np.interp(pf, xt, yt))
    return pf


# ---------------------------------------------------------------------------
# Beta-Binomial posterior (kanıt azken 0.5'e küçülür — dürüst prior)
# ---------------------------------------------------------------------------
def beta_posterior(wins: float, losses: float, prior_a: float = 1.0,
                   prior_b: float = 1.0) -> tuple[float, float, float]:
    """Uniform(1,1) prior'lu Beta posterior → (ortalama, lo95, hi95). `wins`/`losses` REEL
    (float) sayım kabul eder — A1'in zaman-ağırlıklı ETKİN sayımları tam sayı DEĞİLDİR; Beta
    dağılımı α,β parametreleri için reel değer matematiksel olarak geçerlidir. Negatif
    sayımlar 0'a kırpılır. Mevcut TAM SAYI çağrıları (ağırlıksız kullanım) BİT-DENK sonuç
    verir — `int`→`float` dönüşümü küçük tam sayılarda kayıpsızdır (bkz.
    test_ai_prior_decay.py::test_beta_posterior_int_cagri_bit_denk). CI
    scipy.stats.beta.ppf(0.025/0.975) iledir."""
    from scipy.stats import beta as beta_dist
    a = prior_a + max(0.0, float(wins))
    b = prior_b + max(0.0, float(losses))
    mean = a / (a + b)
    lo = float(beta_dist.ppf(0.025, a, b))
    hi = float(beta_dist.ppf(0.975, a, b))
    return (float(mean), lo, hi)


# ---------------------------------------------------------------------------
# Drift — iki yönlü Page-Hinkley (yuvarlanan-ortalama serisi üstünde)
# ---------------------------------------------------------------------------
def page_hinkley(values: list[float], delta: float = 0.005, lam: float = 0.05) -> dict:
    """Klasik PH: m_t = Σ(x_i - x̄_i ∓ delta); |m_t - ekstremum| > lam → drift. İki yön de
    izlenir (artış ve düşüş). Dönen: {'drift': bool, 'index': int|None} (ilk tetik indeksi).
    NOT: ham 0/1 ikili seriye değil, yuvarlanan-ortalama serisine uygulanır (ROLL_WINDOW)."""
    xs = [v for v in (_f(x) for x in values) if v is not None]
    if len(xs) < 3:
        return {"drift": False, "index": None}
    mean = 0.0
    m_inc = 0.0
    min_inc = 0.0    # artış dedektörü: m_inc - min_inc > lam
    m_dec = 0.0
    min_dec = 0.0    # düşüş dedektörü: m_dec - min_dec > lam
    for i, x in enumerate(xs):
        mean += (x - mean) / (i + 1)
        m_inc += (x - mean - delta)
        m_dec += (mean - x - delta)
        if (m_inc - min_inc) > lam or (m_dec - min_dec) > lam:
            return {"drift": True, "index": i}
        min_inc = min(min_inc, m_inc)
        min_dec = min(min_dec, m_dec)
    return {"drift": False, "index": None}


def _rolling_mean(xs: list[float], window: int) -> list[float]:
    """Basit yuvarlanan ortalama (tam pencereler); len(xs) < window → []."""
    if window <= 0 or len(xs) < window:
        return []
    out: list[float] = []
    acc = sum(xs[:window])
    out.append(acc / window)
    for i in range(window, len(xs)):
        acc += xs[i] - xs[i - window]
        out.append(acc / window)
    return out


# ---------------------------------------------------------------------------
# UCUZ CSV okuyucular (reasoner._tail_lines/_read_header deseninin kopyası —
# 2M+ satırlık predictions_log ASLA tam parse edilmez)
# ---------------------------------------------------------------------------
def _read_header(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8", newline="") as f:
            first = f.readline().strip()
    except OSError:
        return []
    return next(csv.reader([first])) if first else []


def _tail_lines(path: Path, n: int) -> list[str]:
    """`atomic_io.tail_lines`e ince sarmalayıcı (Tur-35: üç birebir kopya tek kaynağa indi).
    Ad testler/çağıranlar için korundu."""
    return tail_lines(path, n)


# ---------------------------------------------------------------------------
# Öğrenme güncellemesi
# ---------------------------------------------------------------------------
def update_from_predictions(tail_n: int = 5000, *, persist: bool = True) -> dict:
    """predictions_log'un SON tail_n satırındaki ÇÖZÜLMÜŞ kayıtlardan öğrenme durumunu üret:
    GLOBAL/LONG/SHORT Beta posterior'ları, pwin kalibrasyonu (Brier + kovalar), yuvarlanan
    isabet serisinde drift, kaynak-başına Brier (`sources`) ve `retrain_candidate` bayrağı.
    Sonuç LEARNING_STATE'e atomik yazılır ve döndürülür. `persist=False` yalnız önizler;
    pano örneklem seçimi karar hafızasını değiştiremez. Dönüşteki `saved`, bu çağrının
    yazma sonucudur (diskteki öğrenme şemasına eklenmez).
    Dosya yok/boş/şema eksik → boş-ama-geçerli durum (çökme yok).

    A1 EKİ: yukarıdaki AĞIRLIKSIZ sayıma (`state["strategies"]`) DOKUNULMADAN, aynı satır
    kümesinden üstel-unutmalı EK bir posterior daha üretilir → `state["strategies_decayed"]`
    (bkz. modül docstring'i). Zaman damgası ayrıştırılamayan satırlar bu EK'e girmez;
    sayıları `state["n_unparsed"]`'dadır (ana sayıma etkisi YOK)."""
    state: dict = {"updated_at": _utc_now_iso(), "tail_n": int(tail_n),
                   "strategies": {}, "strategies_decayed": {}, "n_unparsed": 0,
                   "pwin_calibration": None, "drift": None, "note": None,
                   # Kaynak-başına kalibrasyon karnesi: Brier (kalibrasyon+ayırt etme) YANINDA
                   # ECE (yalnız kalibrasyon) — bkz. expected_calibration_error docstring'i.
                   "sources": {"prior": _source_card([]),
                               "pwin": _source_card([]),
                               "neural": _source_card([])},
                   # Tur-35: hizalanmış kaynak füzyonu (KARAR-DIŞI ölçüm; bkz. source_fusion).
                   "source_fusion": None,
                   "retrain_candidate": False}
    header = _read_header(PRED_LOG)
    if not header or "resolved_at" not in header:
        state["note"] = "predictions_log yok/başlıksız — öğrenilecek çözülmüş kayıt bulunamadı"
        state["saved"] = _save(state) if persist else False
        return state
    idx = {name: header.index(name) for name in header}

    # Tur-36 optimizasyonu — ÇIKTI-DENK. Eski `cell(row, name)` kapanışı satır başına 11 kez
    # çağrılıyordu (60k satırda **664.524 çağrı**, profilin %28'i): her çağrıda bir sözlük
    # arama + sınır kontrolü + `.strip()`. Üçü de gereksizdi:
    #   · sütun indeksleri DÖNGÜ DIŞINDA sabittir → bir kez çözülür (yerel int),
    #   · satır başlık uzunluğuna BİR KEZ doldurulur (+1 fazladan boş hücre) → OLMAYAN
    #     sütunun indeksi o hücreyi gösterir; "" döner, dal YOK,
    #   · tüketicilerin ikisi de zaten kırpar (`_f` → float(" x ") aynı sonuç / ValueError;
    #     `_norm_dir` gövdesinde `.strip().upper()` var) → yalnız literal karşılaştırılan
    #     `resolved_at` kırpılır.
    n_col = len(header)
    blank = n_col                      # olmayan sütunlar bu (her zaman boş) hücreyi gösterir
    i_resolved = idx["resolved_at"]    # varlığı yukarıda kontrol edildi
    i_hit = idx.get("hit", blank)
    i_dir = idx.get("direction", blank)
    i_bar = idx.get("bar_ts", blank)
    i_log = idx.get("logged_at", blank)
    i_fwd = idx.get("fwd_ret_pct", blank)
    i_pw = idx.get("pwin", blank)
    i_side = idx.get("pwin_side", blank)
    i_npw = idx.get("neural_pwin", blank)
    i_nside = idx.get("neural_pwin_side", blank)

    # neural gölge p_win logu (Dalga-2 köprüsü) — kolonlar YOKSA sources.neural None kalır
    # (spec §3.2: "neural gölge logu yoksa None"). Mevcut pwin/pwin_side ile AYNI şema:
    # neural_pwin (float) + neural_pwin_side (LONG/SHORT/+/-).
    has_neural_cols = "neural_pwin" in idx and "neural_pwin_side" in idx

    hits_seq: list[int] = []                       # kronolojik 0/1 isabet (drift için)
    counts: dict[str, list[int]] = {"GLOBAL": [0, 0], "LONG": [0, 0], "SHORT": [0, 0]}
    cal_pairs: list[tuple[float, int]] = []        # (pwin, taraf-kazandı mı)
    neural_pairs: list[tuple[float, int]] = []     # (neural_pwin, taraf-kazandı mı) — varsa
    # A1: (t_i, direction, h) — zaman-ağırlıklı EK posterior için. `counts` (yukarı/aşağı)
    # bu listeden TAMAMEN bağımsız güncellenir; bit-denklik bu ayrışmadan gelir.
    timed_records: list[tuple[datetime, str, int]] = []
    # Tur-35 füzyonu: HİZALI üçlü — yalnız İKİ kaynağın da olasılık ürettiği satırlar.
    # Ortak olay "canlı sinyalin YÖNÜ kazandı mı" (= `hit`); kaynak karşı tarafı diyorsa
    # olasılığı 1−p ile çevrilir, yoksa uzmanlar farklı sorulara cevap verir.
    fus_pwin: list[float] = []
    fus_neural: list[float] = []
    fus_y: list[int] = []
    # Tur-36: ek uzman adayları. `prob_cal` ve `confidence` zaten `direction`'a HİZALIDIR
    # (ikisi de "bu yön tutar mı" sorusunu yanıtlar) → çeviri gerekmez. None = o satırda
    # ölçüm yok; bir uzmanın TEK bir None'u varsa o uzman TAMAMEN düşer (satır kırpmak
    # diğer uzmanların örneklemini değiştirir ve karşılaştırmayı bozardı).
    fus_extra: dict[str, list[float | None]] = {"prob_cal": [], "confidence": []}
    i_extra = {k: idx.get(k, blank) for k in fus_extra}
    n_unparsed = 0
    lines = _tail_lines(PRED_LOG, tail_n)
    pad = [""] * (n_col + 1)
    for row in csv.reader(lines):
        # Satır TAM `n_col + 1` hücreye normalize edilir: eksikler "" ile dolar, başlıktan
        # UZUN satırın fazlası atılır (eski `cell` de onları hiç okumuyordu) ve son hücre
        # (`blank`) HER ZAMAN boştur → olmayan sütun "" verir.
        row = (row + pad)[:n_col + 1]
        resolved = row[i_resolved].strip()
        if not resolved or resolved == "UNRESOLVABLE" or resolved == "resolved_at":
            continue
        hit = _f(row[i_hit])
        if hit is None:
            continue
        h = 1 if hit >= 0.5 else 0
        hits_seq.append(h)
        counts["GLOBAL"][0 if h else 1] += 1
        direction = _norm_dir(row[i_dir])
        if direction:
            counts[direction][0 if h else 1] += 1
        ts_i = _resolve_row_ts(row[i_bar], row[i_log], resolved)
        if ts_i is not None:
            timed_records.append((ts_i, direction, h))
        else:
            n_unparsed += 1
        fwd = _f(row[i_fwd])
        pw = _f(row[i_pw])
        side = _norm_dir(row[i_side])
        pw_dir: float | None = None      # P(canlı yön kazanır) — füzyonun ortak olayı
        npw_dir: float | None = None
        if pw is not None and fwd is not None and side and direction:
            side_ret = fwd if side == direction else -fwd
            cal_pairs.append((pw, 1 if side_ret > 0 else 0))
        if pw is not None and side and direction and 0.0 <= pw <= 1.0:
            pw_dir = pw if side == direction else 1.0 - pw
        if has_neural_cols and direction:
            npw = _f(row[i_npw])
            nside = _norm_dir(row[i_nside])
            if npw is not None and nside:
                if fwd is not None:
                    n_side_ret = fwd if nside == direction else -fwd
                    neural_pairs.append((npw, 1 if n_side_ret > 0 else 0))
                if 0.0 <= npw <= 1.0:
                    npw_dir = npw if nside == direction else 1.0 - npw
        if pw_dir is not None and npw_dir is not None:
            fus_pwin.append(pw_dir)
            fus_neural.append(npw_dir)
            fus_y.append(h)
            for key, col in i_extra.items():
                v = _f(row[col])
                fus_extra[key].append(v if v is not None and 0.0 <= v <= 1.0 else None)

    for key, (w, l) in counts.items():
        if w + l == 0:
            continue
        mean, lo, hi = beta_posterior(w, l)
        state["strategies"][key] = {"n": w + l, "win": w, "loss": l,
                                    "post_mean": round(mean, 4),
                                    "lo95": round(lo, 4), "hi95": round(hi, 4)}
    if cal_pairs:
        br = brier_score(cal_pairs)
        ece = expected_calibration_error(cal_pairs, n_bins=10)
        state["pwin_calibration"] = {"n": len(cal_pairs),
                                     "brier": (round(br, 4) if br is not None else None),
                                     "ece": (round(ece, 4) if ece is not None else None),
                                     "bins": reliability_bins(cal_pairs, n_bins=10)}
        state["sources"]["pwin"] = _source_card(cal_pairs)
    if neural_pairs:
        state["sources"]["neural"] = _source_card(neural_pairs)
    if fus_y:
        # `if v is not None` çalışma-zamanında ETKİSİZDİR (dıştaki `all(...)` zaten
        # garanti eder); tip denetleyicisi iki yan tümce arasında daraltma yapamadığı
        # için burada duruyor — `type: ignore` yerine gerçek daraltma tercih edildi.
        kept = {k: [float(v) for v in vals if v is not None]
                for k, vals in fus_extra.items()
                if len(vals) == len(fus_y) and all(v is not None for v in vals)}
        state["source_fusion"] = source_fusion(fus_pwin, fus_neural, fus_y, extra=kept)
    global_s = state["strategies"].get("GLOBAL")
    if global_s is not None and hits_seq:
        # "prior" kaynağı: GLOBAL posterior-ortalamasını SABİT tahmin olarak kullansaydık
        # Brier ne olurdu — decision v3'ün Beta-prior kaynağının kalibrasyon proxy'si.
        prior_pairs = [(global_s["post_mean"], h) for h in hits_seq]
        state["sources"]["prior"] = _source_card(prior_pairs)
    roll = _rolling_mean([float(h) for h in hits_seq], ROLL_WINDOW)
    if len(roll) >= 10:
        state["drift"] = page_hinkley(roll)
        state["drift"]["window"] = ROLL_WINDOW
        state["drift"]["n_resolved"] = len(hits_seq)
    state["retrain_candidate"] = bool(state["drift"] and state["drift"].get("drift"))

    # --- A1: zaman-ağırlıklı EK posterior (`state["strategies"]` yukarıda AYNEN kaldı) ---
    state["n_unparsed"] = n_unparsed
    half_life_days = _resolve_half_life_days()
    drift_shrink_uygulandi = bool(state["drift"] and state["drift"].get("drift"))
    ref_ts = max((t for t, _, _ in timed_records), default=None)
    if ref_ts is not None:
        w_counts: dict[str, list[float]] = {"GLOBAL": [0.0, 0.0], "LONG": [0.0, 0.0], "SHORT": [0.0, 0.0]}
        ham_counts: dict[str, int] = {"GLOBAL": 0, "LONG": 0, "SHORT": 0}
        for t_i, direction, h in timed_records:
            wgt = _age_weight(t_i, ref_ts, half_life_days)
            w_counts["GLOBAL"][0 if h else 1] += wgt
            ham_counts["GLOBAL"] += 1
            if direction:
                w_counts[direction][0 if h else 1] += wgt
                ham_counts[direction] += 1
        for key, (w_win, w_loss) in w_counts.items():
            n_ham = ham_counts[key]
            if n_ham == 0:
                continue
            # Drift = geçmiş daha az bilgilendirici → tepki HER ZAMAN daha AZ güven
            # (posterior 0.5'e doğru genişler). Asla tersi (bkz. DRIFT_SHRINK docstring'i).
            eff_w, eff_l = ((w_win / DRIFT_SHRINK, w_loss / DRIFT_SHRINK) if drift_shrink_uygulandi
                           else (w_win, w_loss))
            mean, lo, hi = beta_posterior(eff_w, eff_l)
            state["strategies_decayed"][key] = {
                "n": round(eff_w + eff_l, 4), "win": round(eff_w, 4), "loss": round(eff_l, 4),
                "post_mean": round(mean, 4), "lo95": round(lo, 4), "hi95": round(hi, 4),
                "n_ham": n_ham, "half_life_days": half_life_days,
                "drift_shrink_uygulandi": drift_shrink_uygulandi,
            }

    if not state["strategies"]:
        state["note"] = "çözülmüş kayıt yok — prior'lar nötr (0.5) kalır"
    state["saved"] = _save(state) if persist else False
    return state


def _save(state: dict) -> bool:
    """Durumu atomik yaz (OP-2: OneDrive kilidi). Hata → sessiz-degrade + log."""
    try:
        atomic_write_text(LEARNING_STATE, json.dumps(state, ensure_ascii=False, indent=1))
    except OSError as e:
        logger.warning(f"[ai.learning] durum yazılamadı ({LEARNING_STATE.name}): {e}")
        return False
    return True


def load_state() -> dict:
    """Kayıtlı öğrenme durumu; yok/bozuk → boş varsayılan (çökme yok)."""
    try:
        raw = json.loads(LEARNING_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"strategies": {}, "updated_at": None}
    if not isinstance(raw, dict) or not isinstance(raw.get("strategies"), dict):
        return {"strategies": {}, "updated_at": None}
    return raw


def learning_report(state: dict | None = None) -> str:
    """Türkçe öğrenme raporu (İZLEME dili; kenar iddiası İÇERMEZ)."""
    st = state if state is not None else load_state()
    lines = ["[Öğrenme]"]
    strats = st.get("strategies") or {}
    if not strats:
        lines.append("  çözülmüş kayıt yok — prior'lar nötr (0.5, geniş CI).")
    for key in ("GLOBAL", "LONG", "SHORT"):
        s = strats.get(key)
        if not isinstance(s, dict):
            continue
        lines.append(f"  {key}: n={s.get('n')} isabet-posterior {s.get('post_mean')}"
                     f" (%95 CI {s.get('lo95')}–{s.get('hi95')})")
    # A1: ağırlıklı posterior'lar AĞIRLIKSIZIN HEMEN YANINDA — ikisi tek bakışta
    # karşılaştırılabilsin diye (biri diğerini "doğru" ilan etmez, ikisi de izlenir).
    decayed = st.get("strategies_decayed")
    if isinstance(decayed, dict) and decayed:
        hl = next((v.get("half_life_days") for v in decayed.values() if isinstance(v, dict)), None)
        lines.append(f"  [Ağırlıklı — yarılanma ömrü {hl} gün]")
        for key in ("GLOBAL", "LONG", "SHORT"):
            s = decayed.get(key)
            if not isinstance(s, dict):
                continue
            shrink_txt = "  [drift-shrink uygulandı]" if s.get("drift_shrink_uygulandi") else ""
            lines.append(f"    {key}: n_ham={s.get('n_ham')} etkin-n={s.get('n')}"
                         f" isabet-posterior {s.get('post_mean')}"
                         f" (%95 CI {s.get('lo95')}–{s.get('hi95')}){shrink_txt}")
    cal = st.get("pwin_calibration")
    if isinstance(cal, dict) and cal.get("n"):
        lines.append(f"  pwin kalibrasyonu: n={cal['n']} Brier={cal.get('brier')}"
                     f" ECE={cal.get('ece')} (Brier 0.25=bilgisiz; ECE 0=kusursuz kalibre)")
    src = st.get("sources")
    if isinstance(src, dict):
        # Kaynakları YAN YANA göstermek şart: "hangi olasılık kaynağı daha dürüst?" sorusu
        # tek kaynağa bakarak cevaplanamaz (Brier'ler ancak birbirine göre anlam taşır).
        satir = [f"{k}(n={v.get('n')} B={v.get('brier')} E={v.get('ece')}"
                 f" BSS={v.get('bss')} RES={v.get('resolution')})"
                 for k, v in src.items() if isinstance(v, dict) and v.get("n")]
        if satir:
            lines.append("  kaynak karnesi: " + " · ".join(satir))
            lines.append("    (BSS ≤ 0 = sabit taban-orandan iyi DEĞİL · RES ≈ 0 = taban "
                         "ötesi bilgi YOK — ACI DERS #9)")
    fus = st.get("source_fusion")
    if isinstance(fus, dict) and fus.get("n"):
        w = fus.get("final_weights") or []
        w_txt = " ".join(f"{ad}={val}" for ad, val in zip(fus.get("experts") or [], w, strict=False))
        uz = "⊕".join(fus.get("experts") or [])
        lines.append(f"  çevrimiçi füzyon ({uz}): n={fus['n']}"
                     f" logkayıp={fus.get('loss_combined')} ↔ uzmanlar={fus.get('loss_experts')}"
                     f" · en iyi tek uzman={fus.get('best_expert')} · pişmanlık={fus.get('regret')}")
        lines.append(f"    tabana karşı eşleştirilmiş Brier farkı={fus.get('paired_brier_diff')}"
                     f" (t={fus.get('paired_brier_t')}, eşik {FUSION_T_THRESHOLD})"
                     f" → {fus.get('durum')} · son ağırlıklar: {w_txt}")
        lines.append("    (pişmanlık > 0 = birleşim en iyi TEK uzmanın gerisinde; Hedge'in "
                     "garantisi 'yakın kalmak'tır, 'geçmek' değil)")
    dr = st.get("drift")
    if isinstance(dr, dict):
        if dr.get("drift"):
            lines.append(f"  DRIFT UYARISI: yuvarlanan isabet serisinde kayma (pencere {dr.get('window')}).")
        else:
            lines.append("  drift: belirgin kayma yok.")
    if st.get("retrain_candidate"):
        lines.append("  RETRAIN ADAYI: drift tetiklendi — yeni aday model eğitimi "
                     "günlük döngüde yeni veri kapılarıyla değerlendirilir.")
    try:
        automatic = json.loads((DATA_DIR / "retraining" / "latest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        automatic = {}
    except (OSError, ValueError):
        automatic = {"verdict": "OKUNAMADI"}
    if isinstance(automatic, dict) and automatic:
        lines.append(f"  Otomatik aday eğitimi: {automatic.get('verdict')} · "
                     f"n={automatic.get('rows')} · sınav #{automatic.get('sinav_no')} · "
                     f"{automatic.get('finished_at')} (canlı modele terfi değildir; "
                     "örtüşen veride tekrarlar bağımsız kanıt değildir).")
    lines.append("  Not: bu ölçümler izleme amaçlıdır; kenar kanıtı değildir.")
    return "\n".join(lines)
