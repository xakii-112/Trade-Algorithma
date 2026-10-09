"""
memory.py — Kalıcı AI operatör HAFIZASI (JSONL). KARAR-DIŞI.

Operatör katmanının "dünü hatırlaması" için basit, append-güvenli bir kayıt: her satır bir
JSON nesnesi ({ts, kind, payload}). Sinyal/karar verisi DEĞİL — günlük digest, gözlem notu
gibi rapor-yan ürünleri. Data-dir çözümlemesine saygılıdır (test'te conftest geçici dizine
yönlendirir → üretim data/'sı kirlenmez).

API:
  - append(kind, payload)         → UTC ts ekler, satırı atomik/append-güvenli yazar
  - recent(kind=None, days=7, ...) → son penceredeki kayıtlar (kind ile filtreli)
  - last(kind)                    → o türün EN SON kaydı (yoksa None)

Dayanıklılık:
  - Kilit: prediction_tracker deseni (threading.Lock; tek-satır append atomiktir).
  - Boyut koruması: dosya ~5MB'ı aşınca son ~2000 satıra atomik kırpılır (atomic_io).
  - Bozuk (yarım JSON) satır SESSİZCE ATLANMAZ: atlanır AMA sayısı logger.warning'e yazılır
    (sessiz veri kaybı yasak).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime, timedelta

from loguru import logger

from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR

# Modül-düzeyi yol: testler monkeypatch'ler (daily_summary/prediction_tracker deseni).
MEM_FILE = DATA_DIR / "ai_memory.jsonl"
MAX_BYTES = int(os.getenv("AI_MEMORY_MAX_BYTES", str(5 * 1024 * 1024)))   # ~5MB
KEEP_LINES = int(os.getenv("AI_MEMORY_KEEP_LINES", "2000"))
_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    """UTC ts (tz-aware, saniye çözünürlüğü) → '...+00:00'. Yerel-saat karışıklığı yok."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def append(kind: str, payload: dict) -> None:
    """{ts(UTC), kind, payload} kaydını JSONL'e ekle (thread-safe; tek-satır append atomiktir).

    payload kopyalanır (çağıranın dict'i sonradan değişse de kayıt sabit). Yazımdan sonra
    boyut koruması tetiklenir."""
    rec = {"ts": _utc_now_iso(), "kind": str(kind), "payload": dict(payload)}
    line = json.dumps(rec, ensure_ascii=False)
    with _LOCK:
        MEM_FILE.parent.mkdir(parents=True, exist_ok=True)
        with MEM_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        _maybe_trim()


def _maybe_trim() -> None:
    """Dosya MAX_BYTES'ı aşarsa son KEEP_LINES satıra atomik kırp (eski kayıtları düşür).
    Çağıran _LOCK tutmalı. Hata → sessiz-degrade (kırpma opsiyonel, veri kaybı yaratmaz)."""
    try:
        if MEM_FILE.stat().st_size <= MAX_BYTES:
            return
        with MEM_FILE.open(encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    if len(lines) <= KEEP_LINES:
        return
    atomic_write_text(MEM_FILE, "".join(lines[-KEEP_LINES:]))


def _iter_records() -> list[dict]:
    """Tüm geçerli kayıtları oku. Bozuk/yarım JSON satırları ATLANIR ama sayısı loglanır."""
    if not MEM_FILE.exists():
        return []
    records: list[dict] = []
    corrupt = 0
    try:
        with MEM_FILE.open(encoding="utf-8") as f:
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
                    records.append(obj)
                else:
                    corrupt += 1
    except OSError as e:
        logger.warning(f"[ai.memory] okunamadı ({MEM_FILE.name}): {e}")
        return records
    if corrupt:
        logger.warning(f"[ai.memory] {corrupt} bozuk satır atlandı ({MEM_FILE.name})")
    return records


def _parse_ts(ts: object) -> datetime | None:
    """ISO ts → tz-aware datetime (naive ise UTC varsay). Ayrıştırılamazsa None."""
    if not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def recent(kind: str | None = None, days: int = 7, limit: int = 200) -> list[dict]:
    """Son `days` gün içindeki kayıtlar (kronolojik; en fazla son `limit`). kind verilirse
    yalnız o tür. ts'siz/ayrıştırılamayan kayıt zaman-filtresine takılmaz (dahil edilir)."""
    cutoff = datetime.now(UTC) - timedelta(days=days)
    out: list[dict] = []
    for rec in _iter_records():
        if kind is not None and rec.get("kind") != kind:
            continue
        ts = _parse_ts(rec.get("ts"))
        if ts is not None and ts < cutoff:
            continue
        out.append(rec)
    return out[-limit:]


def last(kind: str) -> dict | None:
    """Verilen türün EN SON kaydı (yoksa None)."""
    for rec in reversed(_iter_records()):
        if rec.get("kind") == kind:
            return rec
    return None
