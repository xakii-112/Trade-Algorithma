"""
consolidate.py — SEMANTİK bellek konsolidasyonu (FAZ AI v3 "Derin Quant Beyin", WP-H).
KARAR-DIŞI/GÖLGE: bu modül hiçbir üretim sinyaline/eşiğine/META_GATE'e bağlanmaz; salt
gözlem/özet üretir (docs/AI_V3_DERIN_BEYIN_SPEC.md §0 anayasası — bkz. nihai/ai/episodic.py
modül docstring'i, aynı ilkeler burada da geçerlidir).

Ne yapar: episodic.py'nin HAM (epizodik) belleğini periyodik olarak "gece görevi" gibi
tarayıp benzer epizotları DETERMİNİSTİK kümelere indirger ve her kümenin özetini
(`{"pattern","n","isabet",...}`) `data/ai_semantic_memory.jsonl`'e atomik ekler — beyin'in
HATIRLA adımı böylece önce "bu TÜR durumda genelde ne oluyor" (anlamsal özet, ucuz/hızlı),
sonra gerekirse ham epizotlara (episodic.similar_episodes, pahalı/detaylı) bakabilir.

Kümeleme yöntemi (rastgelelik YOK, tamamen deterministik):
  1) epizotlar ÖNCE (kind, sıralı-tags) imzasına göre kabaca KOVALANIR (deterministik grup-anahtarı).
  2) her kova İÇİNDE, epizotlar dosya/kronolojik sırayla tek-geçişli ("leader") kümelenir:
     bir epizot, MEVCUT kümelerden birinin merkezine (centroid) z-normalize RMS uzaklığı
     `_CLUSTER_DISTANCE_THRESHOLD` altındaysa o kümeye katılır (merkez güncellenir); değilse
     YENİ bir küme açar. Aynı girdi HER ZAMAN aynı kümeleri üretir (sıra/eşik sabit).

İdempotens: her küme özeti (pattern, n, isabet) üçlüsünden bir anahtar türetilir; bu anahtar
zaten dosyada VARSA yeni satır YAZILMAZ (aynı girdiyle ikinci koşu dosyayı büyütmez). Epizot
kümesi DEĞİŞİRSE (yeni epizot eklendi → n/isabet değişti) YENİ bir özet satırı eklenir — bu,
"eskiyi sil" değil "yeni bilgiyi ekle" (append-only) felsefesine uyar (episodic.py'nin
resolution-satırı deseniyle aynı ruh).
"""
from __future__ import annotations

import json
import math
import threading
from datetime import UTC, datetime

from loguru import logger

from nihai.ai import episodic
from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR

# Modül-düzeyi yol: testler monkeypatch'ler (episodic.py deseni; env yönlendirmesine uyar —
# kök conftest.py NIHAI_DATA_DIR'ı tmp'ye yönlendirdiğinden DATA_DIR zaten test-güvenlidir).
SEMANTIC_MEMORY_FILE = DATA_DIR / "ai_semantic_memory.jsonl"
_LOCK = threading.Lock()

# Tek-geçişli kümeleme eşiği: z-normalize RMS uzaklığı bu değerin ALTINDAKİ epizotlar aynı
# kümeye katılır (episodic.similar_episodes'daki z-norm RMS ile AYNI ölçek/matematik —
# "1.0" kabaca "1 standart sapma" mesafesine karşılık gelir; belgeli sabit).
_CLUSTER_DISTANCE_THRESHOLD = 1.0


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _std(vals: list[float]) -> float:
    """Popülasyon standart sapması (ddof=0; episodic.py'deki stats hesabıyla aynı formül)."""
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return math.sqrt(var)


def _centroid(eps: list[episodic.Episode]) -> dict[str, float]:
    """Bir epizot grubunun özellik-merkezi (anahtar başına ORTALAMA; yalnızca o epizotlarda
    GEÇEN anahtarlar üstünde — eksik anahtar 0 SAYILMAZ, ortalamaya hiç girmez)."""
    acc: dict[str, list[float]] = {}
    for ep in eps:
        for kk, vv in ep.features.items():
            acc.setdefault(kk, []).append(vv)
    return {kk: sum(v) / len(v) for kk, v in acc.items()}


def _z_rms_distance(a: dict[str, float], b: dict[str, float],
                    stats: dict[str, tuple[float, float]]) -> float:
    """İki özellik-sözlüğü arası z-normalize RMS uzaklığı (episodic.similar_episodes'daki
    "rms" metriğiyle AYNI matematik), YALNIZ ortak anahtarlar üstünde. Ortak anahtar yoksa
    sonsuz (kıyaslanamaz — asla aynı kümeye zorlanmaz)."""
    shared = [kk for kk in a if kk in b]
    if not shared:
        return math.inf
    acc = 0.0
    for kk in shared:
        m, sd = stats.get(kk, (0.0, 0.0))
        sd = sd if sd > 1e-12 else 1.0
        acc += ((a[kk] - m) / sd - (b[kk] - m) / sd) ** 2
    return math.sqrt(acc / len(shared))


def _cluster_episodes(episodes: list[episodic.Episode]) -> list[list[episodic.Episode]]:
    """Deterministik tek-geçişli ("leader") kümeleme (modül docstring'i). RNG YOK — aynı
    `episodes` listesi (aynı sırada) HER ZAMAN aynı kümeleri üretir."""
    buckets: dict[tuple[str, tuple[str, ...]], list[episodic.Episode]] = {}
    for ep in episodes:
        sig = (ep.kind, tuple(sorted(ep.tags)))
        buckets.setdefault(sig, []).append(ep)

    clusters: list[list[episodic.Episode]] = []
    for bucket in buckets.values():   # dict insertion sırası = ilk-görülme sırası (deterministik)
        values: dict[str, list[float]] = {}
        for ep in bucket:
            for kk, vv in ep.features.items():
                values.setdefault(kk, []).append(vv)
        stats = {kk: (sum(v) / len(v), _std(v)) for kk, v in values.items()}

        local_clusters: list[list[episodic.Episode]] = []
        centroids: list[dict[str, float]] = []
        for ep in bucket:
            best_idx, best_dist = -1, math.inf
            for idx, centroid in enumerate(centroids):
                d = _z_rms_distance(ep.features, centroid, stats)
                if d < best_dist:
                    best_idx, best_dist = idx, d
            if best_idx >= 0 and best_dist <= _CLUSTER_DISTANCE_THRESHOLD:
                local_clusters[best_idx].append(ep)
                centroids[best_idx] = _centroid(local_clusters[best_idx])
            else:
                local_clusters.append([ep])
                centroids.append(dict(ep.features))
        clusters.extend(local_clusters)
    return clusters


def _summarize_cluster(cluster: list[episodic.Episode], ts: str) -> dict:
    """Bir kümeyi tek özet satırına indirger: pattern (kind + tags), n, isabet (çözülmüş
    epizotlardaki kazanma oranı; hiç çözülmemişse None — stats_by_tag'in win_rate deseniyle
    AYNI savunmacı sözleşme), n_resolved, feature_centroid (semantic_recall için)."""
    kind = cluster[0].kind
    tags = sorted(cluster[0].tags)
    pattern = kind + (f" ({', '.join(tags)})" if tags else "")
    wins: list[bool] = []
    for ep in cluster:
        if isinstance(ep.outcome, dict):
            win = ep.outcome.get("win")
            if isinstance(win, bool):
                wins.append(win)
    isabet = (sum(wins) / len(wins)) if wins else None
    return {
        "pattern": pattern,
        "n": len(cluster),
        "isabet": (round(isabet, 4) if isabet is not None else None),
        "n_resolved": len(wins),
        "feature_centroid": {k: round(v, 6) for k, v in _centroid(cluster).items()},
        "ts": ts,
    }


def _idempotency_key(rec: dict) -> tuple:
    """İdempotens anahtarı: (pattern, n, isabet). Aynı üçlü zaten dosyadaysa yeni satır
    EKLENMEZ (spec §3.7: "aynı girdiyle 2. koşu satır eklemez")."""
    return (rec.get("pattern"), rec.get("n"), rec.get("isabet"))


def _read_all_patterns() -> list[dict]:
    """SEMANTIC_MEMORY_FILE'daki geçerli kayıtlar (bozuk satır sessizce atlanır, sayısı loglanır)."""
    if not SEMANTIC_MEMORY_FILE.exists():
        return []
    out: list[dict] = []
    corrupt = 0
    try:
        with SEMANTIC_MEMORY_FILE.open(encoding="utf-8") as f:
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
        logger.warning(f"[ai.consolidate] okunamadı ({SEMANTIC_MEMORY_FILE.name}): {e}")
        return out
    if corrupt:
        logger.warning(f"[ai.consolidate] {corrupt} bozuk satır atlandı ({SEMANTIC_MEMORY_FILE.name})")
    return out


def _append_atomic(new_lines: list[str]) -> None:
    """Dosya küçük/düşük-frekanslı olduğundan (episodic.jsonl'in aksine) "atomik ekleme"
    burada mevcut içeriği okuyup + yeni satırları ekleyip TAMAMINI `atomic_write_text` ile
    yeniden yazmak olarak uygulanır (OP-2 dersi: kısmi/yarım yazım riski yok)."""
    with _LOCK:
        SEMANTIC_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        prev = SEMANTIC_MEMORY_FILE.read_text(encoding="utf-8") if SEMANTIC_MEMORY_FILE.exists() else ""
        atomic_write_text(SEMANTIC_MEMORY_FILE, prev + "".join(ln + "\n" for ln in new_lines))


def consolidate(days: int = 90, min_cluster: int = 5) -> dict:
    """Son `days` gündeki (özellik taşıyan) epizotları deterministik kümeler (modül
    docstring'i); `min_cluster` eşiğini geçen her küme için bir özet satırı
    `SEMANTIC_MEMORY_FILE`'e ATOMİK EKLENİR (zaten aynı özetle bir satır varsa TEKRAR
    YAZILMAZ — idempotent, bkz. `_idempotency_key`). Boş epizot havuzu → hiçbir şey yazmaz,
    sıfır-sayaçlı rapor döner (çökme yok).

    Dönüş: {'n_episodes','clusters_found','clusters_qualifying','clusters_written',
    'clusters_skipped_existing'} — KARAR-DIŞI: bir kenar/performans iddiası içermez,
    yalnız hafıza konsolidasyonunun hacmini raporlar (ACI DERS #3/#4 disiplini)."""
    episodes = [ep for ep in episodic.recent_episodes(days=days, limit=10_000) if ep.features]
    clusters = _cluster_episodes(episodes)
    min_cluster = max(1, int(min_cluster))
    qualifying = [c for c in clusters if len(c) >= min_cluster]

    existing_keys = {_idempotency_key(r) for r in _read_all_patterns()}
    ts = _utc_now_iso()
    new_lines: list[str] = []
    written = 0
    skipped = 0
    for cluster in qualifying:
        rec = _summarize_cluster(cluster, ts)
        key = _idempotency_key(rec)
        if key in existing_keys:
            skipped += 1
            continue
        new_lines.append(json.dumps(rec, ensure_ascii=False))
        existing_keys.add(key)
        written += 1

    if new_lines:
        try:
            _append_atomic(new_lines)
        except OSError as e:
            logger.warning(f"[ai.consolidate] {SEMANTIC_MEMORY_FILE.name} yazılamadı: {e}")
            written, skipped = 0, skipped + len(new_lines)

    return {
        "n_episodes": len(episodes),
        "clusters_found": len(clusters),
        "clusters_qualifying": len(qualifying),
        "clusters_written": written,
        "clusters_skipped_existing": skipped,
    }


def semantic_recall(features: dict, k: int = 3) -> list[dict]:
    """Brain HATIRLA adımı için: sorgu özelliklerine en yakın k SEMANTİK (konsolide edilmiş)
    örüntü (spec §3.7). `SEMANTIC_MEMORY_FILE`'daki her kaydın `feature_centroid`'i ile
    z-normalize RMS uzaklığı (episodic.py'nin "rms" metriğiyle AYNI matematik) — en yakından
    en uzağa sıralı döner. Sorgu boş/sayısal değilse veya dosya boşsa/yoksa []."""
    query = {str(kk): float(vv) for kk, vv in (features or {}).items()
             if isinstance(vv, (int, float)) and not isinstance(vv, bool) and vv == vv}
    if not query:
        return []
    patterns = _read_all_patterns()
    if not patterns:
        return []

    values: dict[str, list[float]] = {}
    for rec in patterns:
        centroid = rec.get("feature_centroid")
        if not isinstance(centroid, dict):
            continue
        for kk, vv in centroid.items():
            if isinstance(vv, (int, float)):
                values.setdefault(kk, []).append(float(vv))
    stats = {kk: (sum(v) / len(v), _std(v)) for kk, v in values.items()}

    scored: list[tuple[dict, float]] = []
    for rec in patterns:
        centroid = rec.get("feature_centroid")
        if not isinstance(centroid, dict):
            continue
        shared = [kk for kk in query if kk in centroid]
        if not shared:
            continue
        acc = 0.0
        for kk in shared:
            m, sd = stats.get(kk, (0.0, 0.0))
            sd = sd if sd > 1e-12 else 1.0
            cv = float(centroid[kk])
            acc += ((query[kk] - m) / sd - (cv - m) / sd) ** 2
        scored.append((rec, math.sqrt(acc / len(shared))))
    scored.sort(key=lambda t: t[1])
    return [rec for rec, _ in scored[:k]]
