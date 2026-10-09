"""
calibration_loop.py — AI olasılık KAYNAKLARI için ZAMAN-BÖLMELİ, YALNIZ-OOS kalibrasyon
döngüsü (Tur-25). KARAR-DIŞI / GÖLGE.

ANAYASA: Bu paket sinyal/eşik/META_GATE/execution'a ASLA bağlanmaz (CLAUDE.md "AI katmanı"
bölümü). Yalnız ARTEFAKT ÜRETİR ve RAPOR YAZAR; hiçbir canlı kararı DEĞİŞTİRMEZ. `calibrate()`
bile yalnız rapor/gölge kullanımı içindir — hiçbir üretim karar yolu bu modülü ÇAĞIRMAZ.

NEDEN: `nihai/ai/learning.py`'deki `fit_calibrator`/`apply_calibration` (Platt + izotonik)
yazılmış ve test edilmiş ama üretimde hiçbir çağıranı yok — AI katmanı kalibrasyonu ÖLÇÜYOR
(`sources.*.brier` / `sources.*.ece`) ama hiçbir zaman DÜZELTMİYOR. Bu modül o boşluğu kapatır:
`predictions_log.csv`'den kaynak başına (pwin/neural) kalibratör fit eder, YALNIZ görülmemiş
(OOS) veride ölçer, ve yalnız GERÇEKTEN iyileşiyorsa kalıcılaştırır.

Metodoloji (learning.py'nin KENDİ sözleşmesi gereği — "bu modül train/OOS ayrımını YAPMAZ,
çağıranın sorumluluğudur"):
  1. `predictions_log.csv`'den kaynak-başına (ts, p, y) üçlüleri çıkar (etiketleme mantığı
     `learning.update_from_predictions`'daki İLE AYNI — bilerek KOPYALANDI, import EDİLMEDİ;
     learning.py'ye dokunulmuyor/bağlanmıyor).
  2. Üçlüleri ZAMANA GÖRE sırala (rastgele bölme YASAK — zaman serisi). İlk `split` oranı FIT,
     kalanı OOS.
  3. `learning.fit_calibrator` ile kalibratörü YALNIZ fit diliminde öğren.
  4. OOS diliminde `learning.apply_calibration` ile ham→kalibre dönüşümü yap, ham VE kalibre
     için Brier + ECE ölç (`learning.brier_score` / `expected_calibration_error`).
  5. Yeterlilik kapısı: n_oos < min_n veya fit dilimi tek-sınıf → "YETERSIZ", kalibratör ASLA
     üretilmez, "iyileşti" ASLA denmez.
  6. Kalıcılık kapısı (`refresh`): yalnız Brier VE ECE'nin İKİSİ de OOS'ta iyileşirse kaydedilir.

ZAMAN DAMGASI KANONİKLEŞTİRME (BULGU-16, Tur-9 dersi): `predictions_log.csv`'de ts adayı üç
kolon — `bar_ts` (veri sağlayıcı katmanından; tz-aware ise ET/başka bir tz olabilir [bkz.
`nihai/execution/prediction_tracker.py` `_canon_ts` docstring'i], naive ise KANITLANMIŞ ki
naive-UTC'dir [naive `bar_ts` + `_canon_ts`'in hub↔yfinance dedup'ı doğru çalışması bunu
gerektirir]) ve `logged_at`/`resolved_at` (prediction_tracker.record()/resolve() içinde ham
`datetime.now()` ile YAZILIR — bu makinede ÖLÇÜLDÜ: naive-YEREL, UTC+3). BULGU-16 tam olarak bu
tür bir naive-LOCAL/naive-UTC karışımının forming-bar sızıntısı yarattığı olaydı; burada AYNI
sınıf hatayı önlemek için `_parse_ts_utc` `nihai/ai/memory.py`/`nihai/ai/decision.py`'deki
`_parse_ts` ile AYNI sözleşmeyi kullanır (tz-aware → UTC'ye çevrilip tz düşülür; tz-naive →
ZATEN UTC varsayılır) — SONUÇ HER ZAMAN naive döner ki karşılaştırmalar aware/naive karışımı
TypeError'a asla düşmesin. BİLİNEN SINIR (bilinçli, prediction_tracker.resolve() docstring'indeki
"BİLİNEN SINIR" deseniyle aynı dürüstlükle belirtilir): `bar_ts` gerçek veride SATIRLARIN
TAMAMINDA doludur (kayıt şartı — `_is_directional` bunu zorunlu kılar), bu yüzden
`logged_at`/`resolved_at`'e düşen geri-dönüş yolu pratikte ÖLÜ KODDUR; o yol tetiklenirse
(bozuk/eksik `bar_ts`) naive-YEREL değer sistemin yerel-UTC farkı kadar (bu makinede ~3 saat)
kayabilir. Makine-yereline bağlı bir düzeltme (ör. `astimezone()`) YAPILMADI — bu, testlerin
HANGİ makinede/saat diliminde koştuğuna göre sonucun değişmesi (yeniden-üretilemezlik) anlamına
gelirdi, ki bu proje için makine-bağlı-olmayan determinizm (selfcheck.py'nin sabit-seed ilkesiyle
aynı ruh) yerel-hassasiyetten daha değerlidir.

Kullanım:
    python -m nihai.ai.calibration_loop --report
    python -m nihai.ai.calibration_loop --refresh
    python -m nihai.ai.calibration_loop --refresh --source pwin --method platt --json
"""
from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nihai.ai import learning
from nihai.atomic_io import atomic_write_text, tail_lines
from nihai.config import DATA_DIR

# Modül-düzeyi yollar: testler monkeypatch'ler (learning.py/reasoner/memory deseni).
PRED_LOG = DATA_DIR / "predictions_log.csv"              # OKUMA-YALNIZ
CALIBRATORS_FILE = DATA_DIR / "ai_calibrators.json"       # bu modülün TEK yazdığı artefakt

DEFAULT_SOURCES: tuple[str, ...] = ("pwin", "neural")
DEFAULT_METHODS: tuple[str, ...] = ("platt", "isotonic")
DEFAULT_SPLIT = 0.6            # zaman-sıralı fit oranı (kalan = OOS)
DEFAULT_TAIL_N = 5000          # predictions_log kuyruğu (ham satır; learning.py ile AYNI varsayılan)
DEFAULT_MIN_N = 200            # OOS yeterlilik eşiği
T_ESIK = 1.645                 # tek-yönlü ~%95 — Tur-24 `harness` Welch eşiğiyle AYNI
                               # (taban-oran kontrolü; bkz. learning.paired_brier_t)

# Kaynak adı → (olasılık kolonu, taraf kolonu). learning.update_from_predictions ile AYNI şema.
_SOURCE_FIELDS: dict[str, tuple[str, str]] = {
    "pwin": ("pwin", "pwin_side"),
    "neural": ("neural_pwin", "neural_pwin_side"),
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# learning.py'den KOPYALANMIŞ yardımcılar (import EDİLMEDİ — learning.py'ye dokunulmuyor).
# Etiketleme/okuma mantığının tek doğruluk kaynağı hâlâ learning.update_from_predictions'dır;
# burası onun AYNISINI (bilerek) tekrarlar.
# ---------------------------------------------------------------------------
def _f(x: object) -> float | None:
    """CSV alanı → float; boş/bozuk/NaN → None (learning._f ile birebir aynı)."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _norm_dir(s: str) -> str:
    """Yön alanını kanonikleştir (learning._norm_dir ile birebir aynı): canlı logda direction
    '+'/'-' yazılır (pwin_side ise LONG/SHORT). Bilinmeyen → ''."""
    u = s.strip().upper()
    if u in ("+", "LONG", "UP", "BUY"):
        return "LONG"
    if u in ("-", "SHORT", "DOWN", "SELL"):
        return "SHORT"
    return ""


def _read_header(path: Path) -> list[str]:
    """learning._read_header ile birebir aynı (ucuz CSV okuyucu — tam parse YOK)."""
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
# Zaman damgası kanonikleştirme (BULGU-16) — modül docstring'inde gerekçe.
# ---------------------------------------------------------------------------
def _parse_ts_utc(s: str) -> datetime | None:
    """ISO zaman damgasını naive-UTC datetime'a kanonikleştir.

    `nihai/ai/memory.py`/`nihai/ai/decision.py`'deki `_parse_ts` ile AYNI sözleşme (tz-aware →
    UTC'ye çevrilip tzinfo düşülür; tz-naive → ZATEN UTC varsayılır) — burada da KOPYALANDI,
    import EDİLMEDİ. Fark: oradaki tz-AWARE (UTC ekler), burası tz-NAIVE döner — bu modülde ts
    yalnız SIRALAMA/BÖLME için kullanılır, aware/naive karışımı TypeError riskini modül-içi
    tutarlılıkla (hepsi naive) baştan eler. Ayrıştırılamaz/boş girdi → None (learning._f deseni:
    çökme yok)."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _row_ts(row: list[str], idx: dict[str, int]) -> datetime | None:
    """Satırın kanonik ts'i: `bar_ts` → `logged_at` → `resolved_at` (ilk ayrıştırılabilen).
    Öncelik sırası spec gereği — `bar_ts` tercih edilir çünkü gerçek veride HER ZAMAN doludur
    ve doğrudan karar/gerçekleşme anını temsil eder; diğer ikisi yalnız savunmacı geri-dönüştür
    (bkz. modül docstring'i "BİLİNEN SINIR")."""
    for field in ("bar_ts", "logged_at", "resolved_at"):
        i = idx.get(field)
        if i is None or len(row) <= i:
            continue
        dt = _parse_ts_utc(row[i].strip())
        if dt is not None:
            return dt
    return None


# ---------------------------------------------------------------------------
# B1 — çekirdek: OOS kalibrasyon değerlendirmesi
# ---------------------------------------------------------------------------
def _load_pairs(source: str, tail_n: int = DEFAULT_TAIL_N) -> list[tuple[datetime, float, int]]:
    """predictions_log.csv'nin son `tail_n` (ham) satırından (ts, p, y) üçlüleri.

    `source`: 'pwin' | 'neural'. Etiketleme mantığı `learning.update_from_predictions`'daki İLE
    AYNIDIR (KOPYALANDI, import EDİLMEDİ):
        side_ret = fwd_ret_pct if pwin_side == direction else -fwd_ret_pct
        y = 1 if side_ret > 0 else 0
    Yalnız `resolved_at` dolu VE != 'UNRESOLVABLE' satırlar sayılır. Kolonlar (örn. neural_pwin)
    şemada yoksa boş liste döner (çökme yok). Sıralama BURADA YAPILMAZ — `evaluate_source`'ın
    sorumluluğudur (zaman-sıralı bölme tek bir yerde, tekrar riski olmadan)."""
    if source not in _SOURCE_FIELDS:
        raise ValueError(f"bilinmeyen source: {source!r} (geçerli: {sorted(_SOURCE_FIELDS)})")
    p_field, side_field = _SOURCE_FIELDS[source]
    header = _read_header(PRED_LOG)
    if not header or "resolved_at" not in header or p_field not in header or side_field not in header:
        return []
    idx = {name: header.index(name) for name in header}

    def cell(row: list[str], name: str) -> str:
        i = idx.get(name)
        return row[i].strip() if (i is not None and len(row) > i) else ""

    out: list[tuple[datetime, float, int]] = []
    lines = _tail_lines(PRED_LOG, tail_n)
    for row in csv.reader(lines):
        resolved = cell(row, "resolved_at")
        if not resolved or resolved in ("UNRESOLVABLE", "resolved_at"):
            continue
        direction = _norm_dir(cell(row, "direction"))
        fwd = _f(cell(row, "fwd_ret_pct"))
        p = _f(cell(row, p_field))
        side = _norm_dir(cell(row, side_field))
        if p is None or fwd is None or not side or not direction:
            continue
        ts = _row_ts(row, idx)
        if ts is None:
            continue
        side_ret = fwd if side == direction else -fwd
        out.append((ts, p, 1 if side_ret > 0 else 0))
    return out


def _ts_range(rows: list[tuple[datetime, float, int]]) -> list[str | None]:
    """[ilk, son] ISO ts (kronolojik varsayılır — çağıran sıralamış olmalı). Boş → [None, None]."""
    if not rows:
        return [None, None]
    return [rows[0][0].isoformat(), rows[-1][0].isoformat()]


def _round(x: float | None, nd: int = 4) -> float | None:
    return round(x, nd) if isinstance(x, int | float) else None


def _fmt(x: float | None, spec: str = "+.4f") -> str:
    return format(x, spec) if isinstance(x, int | float) else "?"


def evaluate_source(source: str, method: str = "platt", *, split: float = DEFAULT_SPLIT,
                    tail_n: int = DEFAULT_TAIL_N, min_n: int = DEFAULT_MIN_N) -> dict[str, Any]:
    """Kaynağı ZAMAN-BÖLMELİ, YALNIZ-OOS değerlendir. Rastgele bölme YOK — üçlüler zamana göre
    sıralanır, ilk `split` oranı FIT (kalibratör YALNIZ burada öğrenilir), kalan OOS (ham vs
    kalibre Brier/ECE/kova BURADA ölçülür; fit'e ASLA karışmaz — sızıntısızlık kanıtı testte).

    Yeterlilik kapısı: `n_oos < min_n` VEYA fit dilimi tek-sınıf (veya boş) → `{"durum":
    "YETERSIZ", ...}`; kalibratör ASLA üretilmez, "iyileşti" ASLA denmez (ör. neural kaynağı
    n=6 iken — bkz. testler).

    Dönen dict (`durum="OLCULDU"` iken): kaynak, method, n_fit, n_oos, fit_range, oos_range,
    brier_raw, brier_cal, ece_raw, ece_cal, delta_brier, delta_ece, improved (bool),
    verdict (Türkçe cümle), calibrator (learning.fit_calibrator çıktısı — kalıcılık için),
    bins_raw/bins_cal (OOS güvenilirlik kovaları, ham/kalibre)."""
    triples = _load_pairs(source, tail_n)
    triples.sort(key=lambda t: t[0])   # ZAMANA GÖRE sırala — zaman serisi, rastgele bölme YASAK
    n = len(triples)
    cut = int(n * split)
    fit_rows = triples[:cut]
    oos_rows = triples[cut:]
    n_fit, n_oos = len(fit_rows), len(oos_rows)
    fit_pairs = [(p, y) for _, p, y in fit_rows]
    oos_pairs = [(p, y) for _, p, y in oos_rows]

    base: dict[str, Any] = {
        "kaynak": source, "method": method, "n_fit": n_fit, "n_oos": n_oos,
        "fit_range": _ts_range(fit_rows), "oos_range": _ts_range(oos_rows),
    }

    reasons: list[str] = []
    if n_oos < min_n:
        reasons.append(f"n_oos={n_oos} < min_n={min_n}")
    fit_classes = {y for _, y in fit_pairs}
    if len(fit_classes) < 2:
        reasons.append("fit dilimi boş" if not fit_pairs else "fit dilimi tek-sınıf")
    if reasons:
        gerekce = " ve ".join(reasons)
        return {**base, "durum": "YETERSIZ", "gerekce": gerekce,
                "verdict": f"[{source}/{method}] YETERSİZ örneklem ({gerekce}) — kalibratör "
                          "üretilmedi, 'iyileşti' iddiası yok."}

    calib = learning.fit_calibrator(fit_pairs, method=method)
    if calib is None:
        return {**base, "durum": "YETERSIZ", "gerekce": "fit_calibrator None döndü",
                "verdict": f"[{source}/{method}] fit başarısız (fit_calibrator None) — "
                          "kalibratör üretilmedi."}

    cal_pairs = [(learning.apply_calibration(p, calib), y) for p, y in oos_pairs]
    # ÜÇÜNCÜ yarışmacı (Tur-28): SABİT taban-oran tahmincisi — fit diliminin taban oranını
    # her OOS örneği için söyler. ACI DERS #9: bir kalibrasyon "iyileşmesi" büyük ölçüde
    # taban orana BÜZÜLMEden geliyor olabilir; ham'a karşı Brier+ECE bunu GÖREMEZ.
    base_rate = sum(y for _, y in fit_pairs) / len(fit_pairs)
    base_pairs = [(base_rate, y) for _, y in oos_pairs]

    brier_raw = learning.brier_score(oos_pairs)
    brier_cal = learning.brier_score(cal_pairs)
    brier_base = learning.brier_score(base_pairs)
    ece_raw = learning.expected_calibration_error(oos_pairs)
    ece_cal = learning.expected_calibration_error(cal_pairs)

    delta_brier = (brier_cal - brier_raw) if (brier_raw is not None and brier_cal is not None) else None
    delta_ece = (ece_cal - ece_raw) if (ece_raw is not None and ece_cal is not None) else None
    # Ham'a karşı: kalibrasyon HİJYENİ (gürültü freni, iki bağımsız metrik aynı yönde).
    ham_yener = bool(delta_brier is not None and delta_ece is not None
                     and delta_brier < 0 and delta_ece < 0)
    # Tabana karşı: BİLGİ testi — eşleştirilmiş Brier farkı ANLAMLI ölçüde negatif olmalı.
    # Tabana karşı ECE ARANMAZ (sabit tahminci ECE'de yapısı gereği ~0; kategori hatası
    # olurdu — gerekçe: learning.paired_brier_t docstring'i).
    taban_fark, taban_t = learning.paired_brier_t(cal_pairs, base_pairs)
    taban_yener = bool(taban_t is not None and taban_t < -T_ESIK)
    improved = bool(ham_yener and taban_yener)

    if improved:
        verdict = (f"[{source}/{method}] OOS'ta HEM Brier ({_fmt(brier_raw, '.4f')}→"
                  f"{_fmt(brier_cal, '.4f')}, Δ{_fmt(delta_brier)}) HEM ECE "
                  f"({_fmt(ece_raw, '.4f')}→{_fmt(ece_cal, '.4f')}, Δ{_fmt(delta_ece)}) iyileşti "
                  f"VE sabit tabanı ({_fmt(brier_base, '.4f')}) eşleştirilmiş Brier farkı "
                  f"{_fmt(taban_fark, '+.5f')} (t={_fmt(taban_t, '.2f')}) ile yendi "
                  f"(n_fit={n_fit}, n_oos={n_oos}).")
    elif ham_yener:
        verdict = (f"[{source}/{method}] ham'ı yeniyor ama SABİT TABANI YENMİYOR: Brier "
                  f"{_fmt(brier_raw, '.4f')}→{_fmt(brier_cal, '.4f')} vs TABAN "
                  f"{_fmt(brier_base, '.4f')}; eşleştirilmiş fark {_fmt(taban_fark, '+.5f')} "
                  f"(t={_fmt(taban_t, '.2f')}, eşik −{T_ESIK}) — ACI DERS #9: kazanç bilgi "
                  "değil, taban orana BÜZÜLME. Kalibratör önerilmez.")
    else:
        verdict = (f"[{source}/{method}] OOS'ta iyileşme YOK/karışık: Brier "
                  f"{_fmt(brier_raw, '.4f')}→{_fmt(brier_cal, '.4f')} (Δ{_fmt(delta_brier)}), "
                  f"ECE {_fmt(ece_raw, '.4f')}→{_fmt(ece_cal, '.4f')} (Δ{_fmt(delta_ece)}), "
                  f"taban {_fmt(brier_base, '.4f')} — kalibratör önerilmez.")

    return {**base, "durum": "OLCULDU", "calibrator": calib,
            "brier_raw": _round(brier_raw), "brier_cal": _round(brier_cal),
            "brier_base": _round(brier_base), "base_rate": _round(base_rate),
            "ece_raw": _round(ece_raw), "ece_cal": _round(ece_cal),
            "delta_brier": _round(delta_brier), "delta_ece": _round(delta_ece),
            "taban_brier_fark": _round(taban_fark, 5), "taban_t": _round(taban_t, 2),
            "ham_yener": ham_yener, "taban_yener": taban_yener,
            "improved": improved, "verdict": verdict,
            "bins_raw": learning.reliability_bins(oos_pairs),
            "bins_cal": learning.reliability_bins(cal_pairs)}


# ---------------------------------------------------------------------------
# B2 — kalıcılık (SADECE gerçekten iyileşiyorsa)
# ---------------------------------------------------------------------------
def _save_artifact(data: dict[str, Any]) -> None:
    """Artefaktı atomik yaz (OneDrive kilidi — learning._save deseni). Hata → sessiz-degrade+log."""
    try:
        atomic_write_text(CALIBRATORS_FILE, json.dumps(data, ensure_ascii=False, indent=1))
    except OSError as e:
        logger.warning(f"[ai.calibration_loop] artefakt yazılamadı ({CALIBRATORS_FILE.name}): {e}")


def load_calibrators() -> dict[str, Any]:
    """Kayıtlı kalibratör artefaktı; dosya yok/bozuk/şema eksik → `{}` (ASLA çökme,
    learning.load_state deseni)."""
    try:
        raw = json.loads(CALIBRATORS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def refresh(sources: tuple[str, ...] = DEFAULT_SOURCES, methods: tuple[str, ...] = DEFAULT_METHODS,
           *, split: float = DEFAULT_SPLIT, tail_n: int = DEFAULT_TAIL_N,
           min_n: int = DEFAULT_MIN_N) -> dict[str, Any]:
    """Her `source` için TÜM `methods`'ı dener ve KALICILAŞTIRMA KAPISINI uygular: bir method
    hem HAM'ı (`brier_cal < brier_raw` VE `ece_cal < ece_raw`) hem SABİT TABAN'ı (eşleştirilmiş
    Brier farkı `t < −T_ESIK`) yenmelidir. Birden fazla method geçerse en düşük `brier_cal`'lı
    seçilir (Brier kalibrasyon+ayırt etmeyi birlikte taşır).

    TABAN-ORAN KONTROLÜ (Tur-28 — ACI DERS #9 artık MAKİNEDE). Bu kapı eskiden yalnız ham'a
    bakıyordu ve tam da dersin uyardığı tuzağa düştü: Tur-25'te `pwin` için ECE 0,1325→0,0045
    ölçüldü, artefakta `KAYITLI` yazıldı; SONRA elle yapılan kontrol kazancın neredeyse
    tamamının taban orana BÜZÜLMEden geldiğini gösterdi (sabit tahminci Brier 0,2494 ↔ kalibre
    0,2490 = 0,0004 fark). Kapıda kontrol olmadığı için artefakt aylarca yanıltıcı kaldı.
    Tur-27 aynı kontrolü `adaptive_calibration`'a koymuştu ama BU kapı hâlâ kördü — ikisi aynı
    veri üstünde çelişik verdikt veriyordu. Artık ikisi de `learning.paired_brier_t` kullanır.

    ⚠️ Bu kapı bir KANIT KAPISI DEĞİLDİR (ACI DERS #3 anlamında "kenar" iddia etmez) — yalnız
    ucuz bir çoklu-doğrulamadır. Tek bir metriğin (yalnız Brier VEYA yalnız ECE) iyileşmesini
    yeterli saymak, o TEK metriğe göre seçim yapmak (post-hoc metric shopping) anlamına gelir;
    iki BAĞIMSIZ metriğin AYNI ANDA iyileşmesini istemek, gürültünün ikisini birden şans eseri
    aynı yönde hareket ettirme olasılığını düşürür. Taban-oran testi buna ayrı bir soru ekler:
    "kalibre olasılık, hep taban oranı söyleyen bir tahminciden GERÇEKTEN daha mı bilgili?"

    Kapıyı geçemeyen kaynak artefaktta `{"durum": "KAPI_GECILEMEDI", ...}` olarak GÖRÜNÜR kalır
    (sessizce yok olmaz). Var olan artefaktla BİRLEŞTİRİLİR (yalnız `sources`'taki anahtarlar
    güncellenir) — kısmi refresh diğer kaynakların önceki kaydını silmez."""
    artifact = load_calibrators()
    for source in sources:
        results = [evaluate_source(source, method, split=split, tail_n=tail_n, min_n=min_n)
                   for method in methods]
        measured = [r for r in results if r["durum"] == "OLCULDU"]
        passed = [r for r in measured if r["improved"]]
        updated_at = _utc_now_iso()
        entry: dict[str, Any]
        if not measured:
            worst_reason = results[0].get("gerekce") if results else "predictions_log yok/boş"
            entry = {"kaynak": source, "durum": "YETERSIZ", "updated_at": updated_at,
                     "denemeler": results, "not": worst_reason}
        elif not passed:
            entry = {"kaynak": source, "durum": "KAPI_GECILEMEDI", "updated_at": updated_at,
                     "denemeler": measured,
                     "not": "hiçbir method HEM ham'ı (Brier ve ECE) HEM SABİT TABANI "
                            "(eşleştirilmiş Brier farkı) yendi — kalibratör kaydedilmedi "
                            "(ACI DERS #9 kontrolü kapıda; gürültü freni, kanıt kapısı değil)."}
        else:
            best = min(passed, key=lambda r: r["brier_cal"])
            entry = {"kaynak": source, "durum": "KAYITLI", "method": best["method"],
                     "calibrator": best["calibrator"], "n_fit": best["n_fit"],
                     "n_oos": best["n_oos"], "fit_range": best["fit_range"],
                     "oos_range": best["oos_range"], "brier_raw": best["brier_raw"],
                     "brier_cal": best["brier_cal"], "ece_raw": best["ece_raw"],
                     "ece_cal": best["ece_cal"], "delta_brier": best["delta_brier"],
                     "delta_ece": best["delta_ece"], "updated_at": updated_at,
                     "denemeler": measured}
        artifact[source] = entry
    _save_artifact(artifact)
    return artifact


def calibrate(source: str, p: float) -> float:
    """KARAR-DIŞI: bu fonksiyon hiçbir CANLI karar yolundan çağrılmaz — yalnız rapor/gölge
    kullanımı içindir (bkz. modül docstring'i ANAYASA). Kayıtlı (durum='KAYITLI') kalibratörü
    uygular; kayıt yoksa/kapıyı geçmemişse `p` AYNEN döner (`learning.apply_calibration`'ın
    geçiş-şeffaf sözleşmesiyle birebir — p sayısal değilse ValueError, davranış oradan miras)."""
    entry = load_calibrators().get(source)
    calib = entry.get("calibrator") if isinstance(entry, dict) and entry.get("durum") == "KAYITLI" else None
    return learning.apply_calibration(p, calib)


# ---------------------------------------------------------------------------
# B3 — rapor + CLI
# ---------------------------------------------------------------------------
def calibration_report() -> str:
    """Her kayıtlı kaynak için tek satır (n, önce/sonra Brier ve ECE, verdikt). Artefakt yok/boş
    → 'ölçülemedi' der; ASLA boş metrik uydurmaz."""
    lines = ["[Kalibrasyon döngüsü] (KARAR-DIŞI — yalnız rapor; hiçbir canlı karara bağlı değil)"]
    calibrators = load_calibrators()
    if not calibrators:
        lines.append("  ölçülemedi — data/ai_calibrators.json yok/boş (önce --refresh çalıştırın).")
        lines.append("  Not: bu ölçümler kalibrasyon HİJYENİDİR; kenar kanıtı değildir.")
        return "\n".join(lines)
    for source in sorted(calibrators):
        e = calibrators[source]
        if not isinstance(e, dict):
            continue
        durum = e.get("durum", "?")
        if durum == "KAYITLI":
            lines.append(
                f"  {source}: KAYITLI method={e.get('method')} n_fit={e.get('n_fit')} "
                f"n_oos={e.get('n_oos')} Brier {_fmt(e.get('brier_raw'), '.4f')}→"
                f"{_fmt(e.get('brier_cal'), '.4f')} (Δ{_fmt(e.get('delta_brier'))}) "
                f"ECE {_fmt(e.get('ece_raw'), '.4f')}→{_fmt(e.get('ece_cal'), '.4f')} "
                f"(Δ{_fmt(e.get('delta_ece'))})")
        elif durum == "KAPI_GECILEMEDI":
            # Hangi ayakta düştüğünü SÖYLE: "iyileşmedi" ile "iyileşti ama tabanı geçemedi"
            # taban tabana farklı okumalardır (ACI DERS #9). En iyi denemenin sayılarını bas.
            denemeler = [d for d in (e.get("denemeler") or []) if isinstance(d, dict)]
            en_iyi = min(denemeler, key=lambda d: d.get("brier_cal") or 9.9) if denemeler else {}
            if en_iyi.get("ham_yener"):
                lines.append(
                    f"  {source}: KAPI_GECILEMEDI — kalibrasyon ham'ı düzeltiyor (Brier "
                    f"{_fmt(en_iyi.get('brier_raw'), '.4f')}→{_fmt(en_iyi.get('brier_cal'), '.4f')}, "
                    f"ECE {_fmt(en_iyi.get('ece_raw'), '.4f')}→{_fmt(en_iyi.get('ece_cal'), '.4f')}) "
                    f"ama SABİT TABANI ({_fmt(en_iyi.get('brier_base'), '.4f')}) yenmiyor: "
                    f"t={_fmt(en_iyi.get('taban_t'), '.2f')} > −{T_ESIK}. ACI DERS #9 — kazanç "
                    "bilgi değil, taban orana büzülme; kalibratör kaydedilmedi.")
            else:
                lines.append(f"  {source}: KAPI_GECILEMEDI — hiçbir method HEM Brier HEM ECE'de "
                            "OOS'ta iyileşmedi; kalibratör kaydedilmedi.")
        elif durum == "YETERSIZ":
            lines.append(f"  {source}: YETERSIZ örneklem — OOS'ta ölçülemedi "
                        f"({e.get('not', '?')}).")
        else:
            lines.append(f"  {source}: {durum}")
    lines.append("  Not: bu ölçümler kalibrasyon HİJYENİDİR; kenar kanıtı değildir.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI: `--refresh` (kaynakları yeniden değerlendir + kapıyı geçeni kaydet) · `--report`
    (kayıtlı kalibratörlerin özeti) · `--source`/`--method` (tekrarlanabilir filtre) ·
    `--split`/`--tail`/`--min-n` (evaluate_source parametreleri) · `--json` (makine-okunur)."""
    import argparse
    ap = argparse.ArgumentParser(
        description="AI kaynak-kalibrasyon döngüsü (KARAR-DIŞI; zaman-bölmeli yalnız-OOS)")
    ap.add_argument("--refresh", action="store_true",
                    help="kaynakları yeniden değerlendir; HEM Brier HEM ECE OOS'ta iyileşirse kaydet")
    ap.add_argument("--report", action="store_true", help="kayıtlı kalibratörlerin Türkçe özeti")
    ap.add_argument("--source", action="append", choices=sorted(DEFAULT_SOURCES),
                    help="yalnız bu kaynak (tekrarlanabilir); varsayılan hepsi")
    ap.add_argument("--method", action="append", choices=sorted(DEFAULT_METHODS),
                    help="yalnız bu method (tekrarlanabilir); varsayılan hepsi")
    ap.add_argument("--split", type=float, default=DEFAULT_SPLIT, help="zaman-sıralı fit oranı")
    ap.add_argument("--tail", type=int, default=DEFAULT_TAIL_N, help="predictions_log kuyruğu (ham satır)")
    ap.add_argument("--min-n", type=int, default=DEFAULT_MIN_N, dest="min_n",
                    help="OOS yeterlilik eşiği")
    ap.add_argument("--json", action="store_true", help="çıktıyı JSON bas")
    args = ap.parse_args(argv)

    if not (args.refresh or args.report):
        ap.print_help()
        return 0

    sources = tuple(args.source) if args.source else DEFAULT_SOURCES
    methods = tuple(args.method) if args.method else DEFAULT_METHODS

    if args.refresh:
        refresh(sources=sources, methods=methods, split=args.split, tail_n=args.tail,
               min_n=args.min_n)
    if args.json:
        print(json.dumps(load_calibrators(), ensure_ascii=False, indent=2, default=str))
    else:
        print(calibration_report())
    return 0


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    raise SystemExit(main())
