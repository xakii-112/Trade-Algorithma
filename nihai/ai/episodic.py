"""
episodic.py — EPİZODİK + SEMANTİK hafıza (FAZ AI v2). KARAR-DIŞI.

Epizodik katman: "benzer durumu daha önce gördük mü?" — her epizot bir gözlem/gölge-karar
anıdır ({id, ts, kind, symbol, features, summary, tags, outcome}). JSONL append-only;
sonradan gelen sonuç AYRI bir çözüm satırı ({resolution_of, ts, outcome}) olarak eklenir,
okuyucular birleştirir (en son çözüm kazanır). Kilit/trim/bozuk-sayaç davranışı memory.py
deseninin aynısıdır.

Semantik katman: LESSONS — projenin en pahalı öğrendiği derslerin (CLAUDE.md "ACI DERSLER"
+ operasyonel dersler) makine-okur kataloğu. `check_against_lessons` bir öneri/etiket
metnini derslerle çarpıştırır; çıktı UYARIDIR, karar değildir (ACI DERS #3/#4).

--- v3 GENİŞLEMESİ (WP-H, docs/AI_V3_DERIN_BEYIN_SPEC.md §3.7) ---
- `similar_episodes(..., metric=...)`: "rms" (VARSAYILAN, v2 ile BİT-DENK) yanında "cosine"
  seçeneği — bkz. `_cosine_distance`/`_regime_one_hot`/`_outcome_vector`.
- `_maybe_trim` artık kronolojik DEĞİL, `_forget_score`'a göre düşük-skorludan kırpar
  (yaş + örüntü-sıklığı + bilgi-değeri; deterministik, ağırlıklar belgeli).
- `propose_lesson_candidate`: otomatik ders-ADAYI (insan onayı olmadan LESSONS'a terfi YOK).
"""
from __future__ import annotations

import json
import math
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from loguru import logger

from nihai.atomic_io import atomic_write_text, tail_lines
from nihai.config import DATA_DIR

# Modül-düzeyi yol: testler monkeypatch'ler (memory.py deseni).
EPISODES_FILE = DATA_DIR / "ai_episodes.jsonl"
MAX_BYTES = int(os.getenv("AI_EPISODES_MAX_BYTES", str(5 * 1024 * 1024)))   # ~5MB
KEEP_LINES = int(os.getenv("AI_EPISODES_KEEP_LINES", "2000"))
_LOCK = threading.Lock()

# similar_episodes: en az bu kadar ORTAK sayısal anahtar yoksa epizot kıyaslanmaz.
MIN_SHARED_KEYS = 3

# v3 — ders-adayı dosyası (spec §3.7). KATALOĞA (LESSONS) OTOMATİK TERFİ YOK — yalnız
# insan onayıyla elle eklenir (dürüstlük guard'ı).
LESSON_CANDIDATES_FILE = DATA_DIR / "ai_lesson_candidates.jsonl"


@dataclass
class Episode:
    """Tek bir epizot (gözlem/gölge-karar anısı)."""

    id: str
    ts: str
    kind: str
    symbol: str | None = None
    features: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    outcome: dict | None = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


_DEDUPE_TAIL = 2000   # aynı-gün kopya aramasında okunan son satır (tüm dosya DEĞİL — Tur-35/76)


def _ayni_gun_kopyasi(ep: Episode) -> Episode | None:
    """Kuyrukta AYNI UTC gününde aynı (kind, symbol, summary, tags, features) epizot var mı?"""
    gun = ep.ts[:10]
    for raw in reversed(tail_lines(EPISODES_FILE, _DEDUPE_TAIL)):
        try:
            r = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(r, dict) or "resolution_of" in r:
            continue
        ts = str(r.get("ts") or "")
        if ts[:10] < gun:
            break                                  # kronolojik defter: önceki güne inildi
        if (ts[:10] == gun and r.get("kind") == ep.kind and r.get("symbol") == ep.symbol
                and r.get("summary") == ep.summary and list(r.get("tags") or []) == ep.tags
                and r.get("features") == ep.features):
            return Episode(id=str(r.get("id")), ts=ts, kind=ep.kind, symbol=ep.symbol,
                           features=dict(ep.features), summary=ep.summary, tags=list(ep.tags),
                           outcome=r.get("outcome"))
    return None


def append_episode(kind: str, symbol: str | None = None, features: dict | None = None,
                   summary: str = "", tags: list[str] | None = None,
                   outcome: dict | None = None, *, ayni_gun_tekil: bool = False) -> Episode:
    """Yeni epizotu JSONL'e ekle (thread-safe append; ardından boyut koruması).

    `ayni_gun_tekil=True` (Tur-77): aynı UTC gününde aynı sembol+özellik vektörü+karar zaten
    yazıldıysa YENİ satır eklenmez, var olan epizot döner. Gölge koşucu aynı GÜNLÜK barı gün içinde
    defalarca değerlendiriyor ve her koşu aynı anıyı yeniden yazıyordu — ölçüldü 2026-09-14:
    1.508 epizodun %44,4'ü kopya, 1.000 çözümde yalnız 210 benzersiz (sembol, karar barı), tek
    karar 19 kez. Varsayılan False → mevcut çağıranlar BİT-DENK."""
    num_feats: dict[str, float] = {}
    for k, v in (features or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
            num_feats[str(k)] = float(v)
    ep = Episode(id=uuid.uuid4().hex[:12], ts=_utc_now_iso(), kind=str(kind),
                 symbol=symbol, features=num_feats, summary=str(summary),
                 tags=[str(t) for t in (tags or [])],
                 outcome=(dict(outcome) if outcome else None))
    line = json.dumps({"id": ep.id, "ts": ep.ts, "kind": ep.kind, "symbol": ep.symbol,
                       "features": ep.features, "summary": ep.summary, "tags": ep.tags,
                       "outcome": ep.outcome}, ensure_ascii=False)
    with _LOCK:
        if ayni_gun_tekil and EPISODES_FILE.exists():
            onceki = _ayni_gun_kopyasi(ep)
            if onceki is not None:
                return onceki
        EPISODES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with EPISODES_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        _maybe_trim()
    return ep


def resolve_episode(episode_id: str, outcome: dict) -> bool:
    """Var olan epizota sonucu bağla: AYRI çözüm satırı eklenir (append-only; okuyucular
    birleştirir, en son çözüm kazanır). Epizot yoksa False (satır yazılmaz)."""
    if not any(r.get("id") == episode_id for r in _iter_raw()):
        return False
    line = json.dumps({"resolution_of": str(episode_id), "ts": _utc_now_iso(),
                       "outcome": dict(outcome)}, ensure_ascii=False)
    with _LOCK:
        with EPISODES_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        _maybe_trim()
    return True


# --- v3 unutma eğrisi (spec §3.7 WP-H): skor = ağırlıklı(yaş, örüntü-sıklığı, bilgi-değeri) ---
_FORGET_WEIGHT_AGE = 0.4
_FORGET_WEIGHT_FREQ = 0.3
_FORGET_WEIGHT_VALUE = 0.3
_FORGET_AGE_HALFLIFE_DAYS = 30.0   # yaş skoru bu kadar günde yarıya iner (üstel çürüme)
_FORGET_VALUE_RESOLVED = 1.0       # outcome bilinen (çözülmüş) epizot
_FORGET_VALUE_UNRESOLVED = 0.3     # outcome yok — değersiz değil ama daha düşük öncelik


def _forget_score(ep: Episode, tag_freq: dict[tuple[str, tuple[str, ...]], int], now: datetime) -> float:
    """Unutma-kırpma skoru (spec §3.7): YÜKSEK=tut, DÜŞÜK=önce kırp. Üç ağırlıklı bileşen
    (`_FORGET_WEIGHT_*`, toplamı 1.0, her bileşen [0,1]'e normalize — deterministik, RNG yok):

      - yaş     : üstel çürüme `exp(-ln2 · yaş_gün / _FORGET_AGE_HALFLIFE_DAYS)` (yeni→1, eski→0).
                  ts ayrıştırılamazsa yaş=0 varsayılır (en-taze — silinmeye karşı savunmacı).
      - örüntü  : bu epizotun (kind, sıralı-tags) İMZASININ havuzdaki GÖRECELİ sıklığı — gerçek
                  bir erişim/tıklama sayacı YOK (Episode şemasında yok, eklemek yeni mutable
                  durum gerektirir); bu, "gelecekte tekrar sorgulanma/eşleşme olasılığı" için
                  deterministik bir VEKİL (sık görülen durum kalıpları daha 'erişilir' kabul edilir).
      - bilgi   : çözülmüş (outcome bilinen) epizot ham gözlemden daha değerlidir
                  (`_FORGET_VALUE_RESOLVED` vs `_FORGET_VALUE_UNRESOLVED`)."""
    ts = _parse_ts(ep.ts)
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0) if ts is not None else 0.0
    age_score = math.exp(-math.log(2) * age_days / _FORGET_AGE_HALFLIFE_DAYS)

    sig = (ep.kind, tuple(sorted(ep.tags)))
    freq = tag_freq.get(sig, 1)
    max_freq = max(tag_freq.values()) if tag_freq else 1
    freq_score = freq / max_freq if max_freq > 0 else 0.0

    value_score = _FORGET_VALUE_RESOLVED if isinstance(ep.outcome, dict) else _FORGET_VALUE_UNRESOLVED

    return (_FORGET_WEIGHT_AGE * age_score + _FORGET_WEIGHT_FREQ * freq_score
            + _FORGET_WEIGHT_VALUE * value_score)


def _maybe_trim() -> None:
    """MAX_BYTES aşımında `_forget_score`'a göre EN DÜŞÜK-skorludan kırpar (v3 — artık salt
    kronolojik DEĞİL; bkz. spec §3.7 "unutma eğrisi"). 5MB/2000-satır koruması AYNEN: tetik
    eşiği (MAX_BYTES) ve sonuç satır tavanı (KEEP_LINES) değişmedi — yalnız HANGİ satırların
    tutulacağı değişti. Bir satırın skoru, ait olduğu epizotun skorudur (id ile eşleşen
    "epizot" satırı VEYA resolution_of ile eşleşen "çözüm" satırı) → bir epizot ve onun
    sonradan-gelen çözüm satırı BİRLİKTE tutulur/kırpılır (aynı skoru paylaşırlar). Bozuk/
    boş/eşleşmeyen satırlar en düşük öncelikle (−1.0) işaretlenir — önce onlar gider.
    Kalan satırlar ORİJİNAL dosya sırasıyla yazılır (kronoloji/"son çözüm kazanır" sözleşmesi
    bozulmaz). Çağıran _LOCK tutmalı; hata → sessiz-degrade."""
    try:
        if EPISODES_FILE.stat().st_size <= MAX_BYTES:
            return
        with EPISODES_FILE.open(encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    if len(lines) <= KEEP_LINES:
        return

    now = datetime.now(UTC)
    episodes = _merged_episodes()
    tag_freq: dict[tuple[str, tuple[str, ...]], int] = {}
    for ep in episodes:
        sig = (ep.kind, tuple(sorted(ep.tags)))
        tag_freq[sig] = tag_freq.get(sig, 0) + 1
    scores: dict[str, float] = {ep.id: _forget_score(ep, tag_freq, now) for ep in episodes}

    def _line_score(raw: str) -> float:
        s = raw.strip()
        if not s:
            return -1.0
        try:
            obj = json.loads(s)
        except ValueError:
            return -1.0
        if not isinstance(obj, dict):
            return -1.0
        eid = obj.get("id") if isinstance(obj.get("id"), str) else obj.get("resolution_of")
        return scores.get(eid, 0.0) if isinstance(eid, str) else -1.0

    # skor DESC (en değerli önce), eş-skorda YENİ (büyük orijinal-index) önce — deterministik
    # kararlı sıralama (ikinci anahtar da sabit/tekrarlanabilir).
    order = sorted(range(len(lines)), key=lambda i: (-_line_score(lines[i]), -i))
    kept_idx = set(order[:KEEP_LINES])
    kept_lines = [ln for i, ln in enumerate(lines) if i in kept_idx]   # orijinal dosya sırası
    atomic_write_text(EPISODES_FILE, "".join(kept_lines))


def _iter_raw() -> list[dict]:
    """Tüm geçerli JSON satırları (epizot + çözüm). Bozuk satır atlanır ama SAYISI loglanır."""
    if not EPISODES_FILE.exists():
        return []
    out: list[dict] = []
    corrupt = 0
    try:
        with EPISODES_FILE.open(encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except ValueError:
                    corrupt += 1
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
                else:
                    corrupt += 1
    except OSError as e:
        logger.warning(f"[ai.episodic] okunamadı ({EPISODES_FILE.name}): {e}")
        return out
    if corrupt:
        logger.warning(f"[ai.episodic] {corrupt} bozuk satır atlandı ({EPISODES_FILE.name})")
    return out


def _parse_ts(ts: object) -> datetime | None:
    if not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _merged_episodes() -> list[Episode]:
    """Epizotları çözüm satırlarıyla birleştirerek kronolojik döndür (en son çözüm kazanır)."""
    episodes: list[Episode] = []
    resolutions: dict[str, dict] = {}
    for r in _iter_raw():
        rid = r.get("resolution_of")
        if isinstance(rid, str):
            out = r.get("outcome")
            if isinstance(out, dict):
                resolutions[rid] = out      # dosya sırası = zaman sırası; sonraki ezer
            continue
        if not isinstance(r.get("id"), str):
            continue
        feats = r.get("features")
        episodes.append(Episode(
            id=r["id"], ts=str(r.get("ts") or ""), kind=str(r.get("kind") or ""),
            symbol=(r.get("symbol") if isinstance(r.get("symbol"), str) else None),
            features={k: float(v) for k, v in (feats or {}).items()
                      if isinstance(v, (int, float))} if isinstance(feats, dict) else {},
            summary=str(r.get("summary") or ""),
            tags=[str(t) for t in r.get("tags") or []],
            outcome=(r.get("outcome") if isinstance(r.get("outcome"), dict) else None)))
    for ep in episodes:
        if ep.id in resolutions:
            ep.outcome = resolutions[ep.id]
    return episodes


def recent_episodes(kind: str | None = None, days: int = 30, limit: int = 500) -> list[Episode]:
    """Son `days` gün içindeki epizotlar (kronolojik; en fazla son `limit`). ts'siz kayıt
    zaman filtresine takılmaz (memory.recent davranışıyla tutarlı)."""
    cutoff = datetime.now(UTC) - timedelta(days=days)
    out: list[Episode] = []
    for ep in _merged_episodes():
        if kind is not None and ep.kind != kind:
            continue
        ts = _parse_ts(ep.ts)
        if ts is not None and ts < cutoff:
            continue
        out.append(ep)
    return out[-limit:]


# --- v3 kosinüs metriği yardımcıları (spec §3.7 WP-H) ---
# Rejim etiketi one-hot sözlüğü: quant_math.classify_regime'in ürettiği label formatı
# (trend × vol_bucket = 3×3) + "unknown". BUGÜN episodic.append_episode çağrıları rejim
# etiketini tags'e EKLEMİYOR (decision.py yalnız strateji+aksiyon tag'ler) — bu segment o
# yüzden pratikte çoğunlukla sıfır-vektördür; gelecekte bir çağıran regime label'ını tags'e
# eklerse otomatik aktifleşecek bir kanca (Dalga-2 entegrasyon notu, dönüş raporunda da var).
_REGIME_LABELS: tuple[str, ...] = tuple(
    f"{trend}_{vb}" for trend in ("up", "down", "flat") for vb in ("low", "mid", "high")
) + ("unknown",)


def _regime_one_hot(tags: list[str]) -> tuple[float, ...]:
    """`tags` içinde `_REGIME_LABELS`'tan biri varsa o slot 1.0, aksi hepsi 0.0 (bilgi yok —
    bir kategoriye ZORLANMAZ, nötr sıfır-vektör döner)."""
    tagset = set(tags or [])
    return tuple(1.0 if lbl in tagset else 0.0 for lbl in _REGIME_LABELS)


def _outcome_vector(outcome: dict | None) -> tuple[float, float]:
    """Sonuç (outcome) 2-boyutlu vektör: (bilgi-var-mı, kazandı-mı∈{-1,0,+1}). Bilinmeyen/
    eksik/bool-olmayan 'win' → (0.0, 0.0) NÖTR."""
    if not isinstance(outcome, dict) or not isinstance(outcome.get("win"), bool):
        return (0.0, 0.0)
    return (1.0, 1.0 if outcome["win"] else -1.0)


def _cosine_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """1 − kosinüs-benzerliği (KÜÇÜK=yakın; rms'in "küçük=yakın" sıralama sözleşmesiyle
    TUTARLI — çağıran her iki metrikte de aynı yönde `scored.sort(key=lambda t: t[1])`
    kullanabilir). Sıfır-norm (yön tanımsız) → maksimum uzaklık 1.0 (savunmacı)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 1.0
    cos_sim = max(-1.0, min(1.0, dot / (na * nb)))
    return 1.0 - cos_sim


def similar_episodes(features: dict[str, float], k: int = 5, kind: str | None = None,
                     days: int = 365, metric: str = "rms") -> list[tuple[Episode, float]]:
    """Sayısal özellik uzayında en yakın k epizot (spec §3.7 WP-H). `metric`:
      - "rms" (VARSAYILAN — v2 ile davranışı BİT-DENK korunur): popülasyon istatistikleriyle
        (epizot setinden mean/std) z-normalize edilmiş RMS uzaklık, YALNIZ ortak anahtarlar
        üstünde; ortak anahtar sayısı < MIN_SHARED_KEYS olan epizot atlanır.
      - "cosine" (v3, YENİ): epizot vektörü = [ortak-anahtarların z-normu] ⊕
        [rejim one-hot, `_regime_one_hot`] ⊕ [sonuç vektörü, `_outcome_vector`]; sorgu
        (query) tarafında rejim/sonuç BİLİNMEDİĞİNDEN o iki segment SIFIR (nötr) — yalnız
        z-norm özellik segmenti her iki tarafta da doludur. Dönüş `_cosine_distance` ile
        yine UZAKLIK yönlüdür (küçük=yakın), rms ile sıralama sözleşmesi TUTARLI kalır.
    Bilinmeyen `metric` → ValueError (programcı hatası — API ihlali, savunmacı-veri değil).
    Hiç aday/ortak-anahtar yoksa []."""
    if metric not in ("rms", "cosine"):
        raise ValueError(f"bilinmeyen metric={metric!r} — 'rms' veya 'cosine' olmalı")
    query = {str(kk): float(vv) for kk, vv in (features or {}).items()
             if isinstance(vv, (int, float)) and not isinstance(vv, bool) and vv == vv}
    if not query:
        return []
    pool = [ep for ep in recent_episodes(kind=kind, days=days, limit=10_000) if ep.features]
    if not pool:
        return []
    # Popülasyon istatistikleri (anahtar başına mean/std) — z-normalizasyon tabanı (HER İKİ
    # metrik de aynı tabanı paylaşır; cosine'in z-norm segmenti de bundan türetilir).
    stats: dict[str, tuple[float, float]] = {}
    values: dict[str, list[float]] = {}
    for ep in pool:
        for kk, vv in ep.features.items():
            values.setdefault(kk, []).append(vv)
    for kk, vals in values.items():
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        stats[kk] = (m, math.sqrt(var))
    scored: list[tuple[Episode, float]] = []
    for ep in pool:
        shared = [kk for kk in query if kk in ep.features]
        if len(shared) < MIN_SHARED_KEYS:
            continue
        if metric == "rms":
            acc = 0.0
            for kk in shared:
                m, sd = stats.get(kk, (0.0, 0.0))
                sd = sd if sd > 1e-12 else 1.0     # sabit kolon → ham fark
                dq = (query[kk] - m) / sd
                de = (ep.features[kk] - m) / sd
                acc += (dq - de) ** 2
            scored.append((ep, math.sqrt(acc / len(shared))))
        else:   # "cosine"
            q_vec: list[float] = []
            e_vec: list[float] = []
            for kk in shared:
                m, sd = stats.get(kk, (0.0, 0.0))
                sd = sd if sd > 1e-12 else 1.0
                q_vec.append((query[kk] - m) / sd)
                e_vec.append((ep.features[kk] - m) / sd)
            q_vec.extend((0.0,) * len(_REGIME_LABELS))
            e_vec.extend(_regime_one_hot(ep.tags))
            q_vec.extend((0.0, 0.0))
            e_vec.extend(_outcome_vector(ep.outcome))
            scored.append((ep, _cosine_distance(tuple(q_vec), tuple(e_vec))))
    scored.sort(key=lambda t: t[1])
    return scored[:k]


def _fwd_ret(ep: Episode) -> float | None:
    """Çözüm satırındaki ileri getiri (%); yoksa/sonlu değilse None."""
    v = ep.outcome.get("fwd_ret_pct") if isinstance(ep.outcome, dict) else None
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        return None
    return float(v)


def analog_ozeti(features: dict[str, float], k: int = 5, kind: str | None = "ai_decision",
                 days: int = 365, haric_id: str | None = None,
                 haric_sembol: str | None = None, min_taban: int = 20) -> dict:
    """K-NN TARİHSEL ANI raporu (Tur-77, Madde 5.4) — KARAR-DIŞI, operatör içindir.

    Sorgu vektörüne kosinüs metriğinde en yakın `k` ÇÖZÜLMÜŞ epizot + o anların ileri getirisi,
    ve AYNI havuzun koşulsuz ileri getirisi (taban). Taban olmadan bir komşu ortalaması
    okunamaz (ACI DERS #9): "benzer günler +%1 getirdi" ancak havuzun kendisi +%1 değilse bilgidir.

    Havuz yalnız çözülmüş epizotlardır: `similar_episodes(metric="cosine")` sonuç vektörünü de
    vektöre kattığından, kazan/kaybet bilgisi TAŞIYAN bir epizot sorguya daha uzak görünür —
    karışık havuzda sıralama çözülmemişlere kayardı; tümü çözülmüş havuzda sonuç segmentinin
    normu herkes için aynıdır ve sıralama adil kalır.

    ÖLÇÜLEN SINIRLAR (2026-09-14, `data/ai_episodes.jsonl`): epizot vektörü 25 boyutludur
    (`FEATURE_NAMES`, listedeki "96" nöral özellik sayısıdır, epizoda yazılmaz); defter
    2026-08-22'de başlar (ör. "2020 Mart" anısı YOKTUR); karar motoru her epizotta ABSTAIN
    dediği için çözümlerde `win` HEP boştur → ileri getiri TARAF-İŞARETSİZ sembol getirisidir.
    `k` komşunun ortalaması istatistiksel çıkarım değildir; yalnız bağlamdır."""
    out: dict = {"komsular": [], "n_havuz": 0, "n_taban": 0, "n_kopya_elenen": 0,
                 "taban_fwd_ret_ort": None,
                 "taban_pozitif_payi": None, "komsu_fwd_ret_ort": None,
                 "komsu_pozitif_payi": None, "yeterli": False, "reason": None,
                 "metric": "cosine", "not": "ileri getiri taraf-işaretsizdir; komşu ortalaması "
                                             "çıkarım değil bağlamdır"}
    k = max(1, int(k))
    try:
        scored = similar_episodes(features, k=10_000, kind=kind, days=days, metric="cosine")
    except ValueError as e:
        out["reason"] = f"SORGU_GECERSIZ:{e}"
        return out
    # KOPYA ANI tekilleştirmesi (Tur-77 ölçümü: 1.000 çözümde 210 benzersiz karar barı). Aynı
    # (sembol, karar barı) tek anıdır; en yakın temsilci tutulur (skor sıralı listede ilk görülen).
    goruldu: set[tuple] = set()
    havuz: list[tuple[Episode, float, float]] = []
    kopya = 0
    for ep, d in scored:
        r = _fwd_ret(ep)
        if r is None or ep.id == haric_id or (haric_sembol and ep.symbol == haric_sembol):
            continue
        bar = (ep.outcome or {}).get("decision_bar")
        anahtar = ((ep.symbol, bar) if bar
                   else (ep.symbol, ep.ts[:10], tuple(sorted(ep.features.items()))))
        if anahtar in goruldu:
            kopya += 1
            continue
        goruldu.add(anahtar)
        havuz.append((ep, d, r))
    out["n_havuz"] = len(scored)
    out["n_taban"] = len(havuz)
    out["n_kopya_elenen"] = kopya
    if not havuz:
        out["reason"] = "COZULMUS_KOMSU_YOK"
        return out
    getiriler = [r for _, _, r in havuz]
    out["taban_fwd_ret_ort"] = round(sum(getiriler) / len(getiriler), 4)
    out["taban_pozitif_payi"] = round(sum(1 for r in getiriler if r > 0) / len(getiriler), 4)
    secilen = havuz[:k]
    out["komsular"] = [{"id": ep.id, "ts": ep.ts, "symbol": ep.symbol,
                        "benzerlik": round(1.0 - d, 4), "fwd_ret_pct": r,
                        "horizon_bars": (ep.outcome or {}).get("horizon_bars"),
                        "karar": ep.summary, "tags": list(ep.tags)}
                       for ep, d, r in secilen]
    kr = [r for _, _, r in secilen]
    out["komsu_fwd_ret_ort"] = round(sum(kr) / len(kr), 4)
    out["komsu_pozitif_payi"] = round(sum(1 for r in kr if r > 0) / len(kr), 4)
    out["yeterli"] = len(secilen) >= k and len(havuz) >= int(min_taban)
    if not out["yeterli"]:
        out["reason"] = f"AZ_ORNEK:komsu={len(secilen)}/{k},taban={len(havuz)}/{min_taban}"
    return out


def stats_by_tag(tag: str, days: int = 365) -> dict:
    """Etiketli epizotların sonuç sayımı: outcome={'win': bool} kayıtlarından
    {'n','win','loss','win_rate'} (win_rate: sonuçsuz n=0 → None). İZLEME amaçlıdır,
    kenar kanıtı değildir."""
    n = win = loss = 0
    for ep in recent_episodes(days=days, limit=10_000):
        if tag not in ep.tags:
            continue
        n += 1
        if isinstance(ep.outcome, dict) and isinstance(ep.outcome.get("win"), bool):
            if ep.outcome["win"]:
                win += 1
            else:
                loss += 1
    denom = win + loss
    return {"n": n, "win": win, "loss": loss,
            "win_rate": (win / denom if denom else None)}


# ---------------------------------------------------------------------------
# SEMANTİK katman — kurumsal ders kataloğu (CLAUDE.md "ACI DERSLER" + operasyonel)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Lesson:
    """Tek bir kurumsal ders: iddia (ne öğrenildi) + guard (ne yapılmalı) + kaynak."""

    id: str
    claim: str
    guard: str
    source: str


LESSONS: list[Lesson] = [
    Lesson("ACI-1", "Survivorship bias kenar uydurur: PEAD nasdaq100 PF 1.57 → PIT S&P500 PF 1.05.",
           "Her kenar iddiasını point-in-time (survivorship-temiz) evrende yeniden ölç.",
           "CLAUDE.md ACI DERSLER #1"),
    Lesson("ACI-2", "Meta-labeling PIT'te kenarsız: OOS-AUC 0.51-0.52 (yazı-tura), hiçbir eşik kâra çevirmedi.",
           "nasdaq100 PF'lerini kanıt sayma; PIT + kapı simülasyonu iste.",
           "CLAUDE.md ACI DERSLER #2"),
    Lesson("ACI-3", "AUC değil KAPI SİMÜLASYONU karar verir; tek bölmede iyi görünen overfit olabilir.",
           "Hem time hem symbol bölmesinde expectancy/PF BASE üstüne çıkmalı (dayanıklılık).",
           "CLAUDE.md ACI DERSLER #3"),
    Lesson("ACI-4", "OOS-kenarlı baz sinyal yokken kapı yalnız gürültü ekler.",
           "META_GATE varsayılan KAPALI kalır; gölge mod kanıt döngüsünü yaşatır.",
           "CLAUDE.md ACI DERSLER #4"),
    Lesson("ACI-5", "Gerçek darboğaz veri ve OOS-kenarlı baz sinyal yokluğu; model mimarisi değil.",
           "Yeni sınıflandırıcı eklemek çözmez: Sharadar/CRSP PIT verisi veya canlı paper kanıtı gerekir.",
           "CLAUDE.md ACI DERSLER #5"),
    Lesson("ACI-6", "PEAD kesit-ML de kenarsız (PIT 3228 olay, OOS-AUC ~0.50); model surprise'a değil vix/ret'e bakıyordu.",
           "Tek-bölme PF 4.40 gibi değerleri şans/survivorship olarak sorgula; özellik-önemine bak.",
           "CLAUDE.md ACI DERSLER #6"),
    Lesson("ACI-7", "Tüm metrikler yeşilken bile kenar zamanlama artefaktı olabilir: pairs PF 2.08 → exec-lag 1 barla PF 1.03.",
           "Yeni aday sinyalde --exec-lag 1 duyarlılığı kanıt kapısının parçasıdır; DSR/PBO bunu yakalamaz.",
           "CLAUDE.md ACI DERSLER #7"),
    # Tur-48: #8 ve #9 CLAUDE.md'ye 2026-07/08'de eklenmişti ama BU KATALOĞA HİÇ GİRMEDİ.
    # Sonuç: `check_against_lessons` (agent/brain/decision'ın okuduğu tek ders kapısı)
    # projenin en çok atıf alan iki dersi için ASLA ateşlenemiyordu — üstelik ikisi de tam
    # olarak KANIT KAPILARININ bozulmasını anlatıyor. Tur-35 bu boşluğun yalnız BİÇİM
    # tarafını kapatmıştı (`test_ders_ref_format_aci_n` desenini 1-7 → 1-9 genişleterek);
    # desen genişledi, katalog genişlemedi → `ders_ref=["ACI-9"]` SESSİZCE boşa işaret etti.
    Lesson("ACI-8", "Sınıf dengesizliği DSR'yi şişirir: PIT S&P500'de (taban %35,7) OOS-AUC 0.500 iken DSR 1.0 çıktı.",
           "Sözde-getiri TRİVİYAL kurala göre FAZLALIK ölçülür (sıfıra karşı DEĞİL): "
           "excess_pseudo_returns / kapı-sonrası fazlalık. Kapı raporun geri kalanıyla "
           "çelişiyorsa (AUC~0.50 + DSR~1.0) bozuk olan KAPIDIR, sonuç değil.",
           "CLAUDE.md ACI DERSLER #8"),
    Lesson("ACI-9", "Kalibrasyon kazancı 'bilgi bulundu' demek değildir: Brier 0.2723→0.2490 iken SABİT taban-oran tahmincisi 0.2494 veriyordu.",
           "Bir kalibrasyon iyileşmesi raporlanmadan önce sabit taban-oran tahmincisiyle "
           "EŞLEŞTİRİLMİŞ Brier farkı (learning.paired_brier_t, t<-1.645) ile kıyaslanır; "
           "kural TEK kapıya değil kuralı kullanan HER kapıya konur.",
           "CLAUDE.md ACI DERSLER #9"),
    Lesson("OP-1", "Naive-LOCAL vs naive-UTC karışımı forming-bar sızıntısı yarattı (UTC+3'te yarı-bar atılmıyordu).",
           "Tüm zaman damgaları tz-canonik (UTC, tz-aware) işlenir; naive karşılaştırma yasak.",
           "Tur-9 BULGU-16"),
    Lesson("OP-2", "OneDrive/AV kilidi sıcak döngüde düz yazımı bozar (predictions_log çöküşü).",
           "Durum dosyaları YALNIZ nihai.atomic_io ile yazılır (yeniden-denemeli atomik replace).",
           "Tur-12 atomic_io"),
    Lesson("OP-3", "SYSTEM hesabıyla kayıtlı watchdog OneDrive profiline erişemez ve görünmez-zombi olur.",
           "Zamanlanmış görevler interactive-user olarak kaydedilir; baseline YERİNDE ölçülür.",
           "Tur-12/13 servis devri"),
    Lesson("OP-4", "Purge yalnız sol taraflıydı; sağ-purge eksikliği CV sızıntısı bırakır.",
           "Purged CV her iki yönde simetrik purge + embargo uygular.",
           "Tur-12 purged_cv"),
    Lesson("OP-5", "Look-ahead en sinsi hata sınıfı: özellik/sinyal yalnız karar barına kadarki veriden.",
           "df.iloc[:i+1] dilimi kuraldır; forming-bar gerekirse kasıtlı atılır (Madde 3).",
           "CLAUDE.md çalışma ilkeleri"),
    Lesson("OP-6", "Canlı log şemasında 'strategy' kolonu yok; tahminler yön (direction) düzeyindedir.",
           "Öğrenme prior'ları GLOBAL/LONG/SHORT anahtarlarıyla tutulur; strateji-adı eşleşmesi bulunamazsa nötr prior kullanılır.",
           "AI v2 entegrasyonu 2026-07-19"),
]

# Ders-id → arama anahtar kelimeleri (küçük harf; check_against_lessons eşleşmesi).
_LESSON_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ACI-1": ("survivorship", "pit", "point-in-time", "evren", "delist"),
    "ACI-2": ("meta-labeling", "meta labeling", "auc", "yazı-tura", "esik", "eşik"),
    "ACI-3": ("tek bölme", "tek bolme", "pf", "profit factor", "kapı simülasyon", "kapi simulasyon", "auc"),
    "ACI-4": ("meta_gate", "kapı", "kapi", "gate", "gürültü", "gurultu"),
    "ACI-5": ("darboğaz", "darbogaz", "veri", "sharadar", "crsp", "paper"),
    "ACI-6": ("pead", "kesit", "surprise", "özellik önem", "ozellik onem"),
    "ACI-7": ("exec-lag", "exec lag", "gecikme", "pairs", "kointegrasyon", "zamanlama"),
    # Anahtarlar AYIRT EDİCİ seçildi: kısa/gömülü parçalar (ör. çıplak "ece") Türkçe
    # gövdelerin İÇİNDE eşleşir ("gelecek" → "ece") ve kapıyı gürültüye boğardı.
    "ACI-8": ("dsr", "deflated sharpe", "sözde-getiri", "sozde-getiri", "sınıf dengesiz",
              "sinif dengesiz", "çoğunluk", "cogunluk", "triviyal"),
    "ACI-9": ("kalibrasyon", "brier", "sabit taban", "taban oran", "platt", "izotonik",
              "büzülme", "buzulme"),
    "OP-1": ("naive", "utc", "forming", "yarım bar", "yarim bar", "tz"),
    "OP-2": ("onedrive", "kilit", "atomik", "atomic"),
    "OP-3": ("watchdog", "system", "zamanlanmış görev", "zamanlanmis gorev", "servis"),
    "OP-4": ("purge", "embargo", "cv sızıntı", "cv sizinti"),
    "OP-5": ("look-ahead", "lookahead", "sızıntı", "sizinti", "karar barı", "karar bari", "leak"),
    "OP-6": ("strategy kolonu", "direction", "prior", "global"),
}


def check_against_lessons(text_or_tags: str | list[str]) -> list[Lesson]:
    """Metni/etiketleri ders anahtar-kelimeleriyle çarpıştır; eşleşen dersleri döndür
    (id sırasıyla, tekrarsız). Çıktı yalnız UYARI/şeffaflık içindir — karar değildir."""
    if isinstance(text_or_tags, str):
        hay = text_or_tags.lower()
    else:
        hay = " ".join(str(t) for t in text_or_tags).lower()
    if not hay.strip():
        return []
    hits: list[Lesson] = []
    for les in LESSONS:
        kws = _LESSON_KEYWORDS.get(les.id, ())
        if any(kw in hay for kw in kws):
            hits.append(les)
    return hits


# ---------------------------------------------------------------------------
# v3 — Ders-adayı üretici (spec §3.7 WP-H)
# ---------------------------------------------------------------------------
def propose_lesson_candidate(claim: str, evidence: str, source: str) -> dict:
    """Otomatik ders-ADAYI üretir (ör. drift/kalibrasyon-bozulma olayından): tek satır
    ATOMİK olarak `LESSON_CANDIDATES_FILE`'a eklenir (`status="candidate"`).

    KATALOĞA (`LESSONS`) OTOMATİK TERFİ YOKTUR — bu fonksiyon `LESSONS` listesini asla
    değiştirmez; yeni bir ders yalnız İNSAN ONAYIYLA elle koda eklenir (dürüstlük guard'ı,
    ACI DERS #3/#4 disiplini: "yeni ders/kenar bulundu" iması otomatikleşemez).

    Dosya küçük/düşük-frekanslı olduğundan (episodic.jsonl'in aksine) "atomik ekleme" burada
    mevcut içeriği okuyup + yeni satırı ekleyip `atomic_write_text` ile TAMAMINI yeniden
    yazmak olarak uygulanır (OP-2/atomic_io dersi: kısmi/yarım yazım riski yok).

    Dönüş: yazılan kaydın dict'i (id/ts/claim/evidence/source/status)."""
    rec = {
        "id": uuid.uuid4().hex[:12],
        "ts": _utc_now_iso(),
        "claim": str(claim),
        "evidence": str(evidence),
        "source": str(source),
        "status": "candidate",
    }
    line = json.dumps(rec, ensure_ascii=False)
    with _LOCK:
        LESSON_CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
        prev = (LESSON_CANDIDATES_FILE.read_text(encoding="utf-8")
                if LESSON_CANDIDATES_FILE.exists() else "")
        atomic_write_text(LESSON_CANDIDATES_FILE, prev + line + "\n")
    return rec
