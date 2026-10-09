"""Kanıtı model kimliğine bağlanan, operatör onaylı meta-kapı terfisi."""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from datetime import date

from loguru import logger

from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR

PATH = DATA_DIR / "meta_activation.json"
INTERVALS = ("1m", "5m", "15m", "1h", "1d")
_LOCK = threading.RLock()
_stamp = None
_cached: dict | None = None
_ack_at: dict = {}


def _read() -> dict | None:
    global _stamp, _cached
    with _LOCK:
        try:
            stat = PATH.stat()
        except FileNotFoundError:
            _stamp, _cached = None, None
            return None
        stamp = (str(PATH), stat.st_mtime_ns, stat.st_size)
        if stamp != _stamp:
            p = json.loads(PATH.read_text(encoding="utf-8"))
            if (p["mode"] not in ("enforce", "shadow") or p["interval"] not in INTERVALS
                    or not re.fullmatch(r"[0-9a-f]{64}", p["model_id"])
                    or not re.fullmatch(r"[0-9a-f]{32}", p["revision"])
                    or not isinstance(p["strategy"], str) or not p["strategy"]
                    or not math.isfinite(p["threshold"]) or not 0 < p["threshold"] < 1):
                raise ValueError("Geçersiz terfi kaydı")
            _stamp, _cached = stamp, p
        return _cached


def policy_revision() -> str | None:
    try:
        p = _read()
        return p["revision"] if p else None
    except (OSError, ValueError, TypeError, KeyError):
        return "unreadable"


def live_policy(default_mode: str, interval: str, strategy: str) -> tuple[str, float | None]:
    """Yalnız açıkça canlı çağrılarda; farklı bar/stratejiye kanıt taşınmaz."""
    try:
        p = _read()
    except (OSError, ValueError, TypeError, KeyError):
        return "shadow", None
    if not p or (p["interval"], p["strategy"]) != (interval, strategy):
        return default_mode, None
    from nihai.signals.meta_gate import get_gate
    gate = get_gate()
    matches = gate.active and gate.model_id == p["model_id"]
    mode = p["mode"] if matches else "shadow"
    key = (p["revision"], interval, strategy, gate.model_id, mode)
    now = time.time()
    with _LOCK:
        if now - _ack_at.get(key, 0) >= 30:
            try:
                atomic_write_text(DATA_DIR / f"meta_applied_{os.getpid()}.json", json.dumps({
                    "pid": os.getpid(), "revision": p["revision"], "model_id": gate.model_id,
                    "mode": mode, "interval": interval, "strategy": strategy, "observed_at": now}))
                _ack_at.clear()
                _ack_at[key] = now
            except OSError as exc:
                logger.warning(f"[meta] terfi uygulama kaydı yazılamadı: {type(exc).__name__}")
    return mode, p["threshold"] if matches else None


def status() -> dict:
    from nihai.signals.meta_gate import get_gate
    gate = get_gate()
    try:
        policy, error = _read(), None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        policy, error = None, type(exc).__name__
    applied = []
    for path in DATA_DIR.glob("meta_applied_*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if policy and row.get("revision") == policy["revision"]:
                applied.append(row)
        except (OSError, ValueError, TypeError):
            continue
    return {"model_id": gate.model_id, "kind": gate.kind, "policy": policy,
            "applied": applied, "error": error, "intervals": list(INTERVALS)}


def review(interval: str) -> dict:
    from nihai.config import CONFIG
    from nihai.execution.prediction_tracker import _read_log_rows, _side_return
    from nihai.runtime_strategy import settings
    from nihai.signals.meta_gate import get_gate
    from research import shadow_promote as sp
    from research.live_retrain import MIN_SPAN_DAYS

    if interval not in INTERVALS:
        raise ValueError("Geçersiz canlı bar aralığı")
    gate = get_gate()
    strategy = settings()["strategy"]
    rows = [r for r in _read_log_rows() if gate.model_id and r.get("meta_model_id") == gate.model_id
            and r.get("meta_strategy") == strategy and r.get("interval") == interval
            and str(r.get("hit")) in ("0", "1") and _side_return(r) is not None]
    report = sp.run(rows)
    threshold = float(CONFIG.strategy.meta_gate_threshold or gate.threshold)
    spec = next(s for s in sp.SURFACES if s["key"] == "meta_pwin")
    rule = {"thr": threshold, "keep": "high"}
    _, oos_rows = sp._split_by_time(rows, .4)
    fixed = sp._apply_rule(rows, spec, rule, _side_return, min_side=sp.MIN_SIDE_DEFAULT)
    oos = sp._apply_rule(oos_rows, spec, rule, _side_return, min_side=sp.MIN_SIDE_DEFAULT)

    def common_price(r):
        # Aynı bar kapanışını referans al; yön tahmini ile işlem tarafını karıştırma.
        side = r.get("pwin_side")
        if side not in ("LONG", "SHORT"):
            return None
        return sp._bar_return({**r, "direction": "+" if side == "LONG" else "-"})

    slip = sp._apply_rule(rows, spec, rule, common_price, min_side=sp.MIN_SIDE_DEFAULT)
    days = sorted({str(r.get("logged_at", ""))[:10] for r in rows})
    try:
        span = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days if days else 0
    except ValueError:
        span = 0
    checks = {
        "Yüklü model kimliği": bool(gate.active and gate.model_id),
        "Canlı kanıt süresi": span >= MIN_SPAN_DAYS,
        "Gölge terfi kapıları": report["surfaces"]["meta_pwin"]["label"] == "PROMOTE",
        "Sabit eşikte ayrışma": fixed["label"] == "CONFIRMED" and fixed.get("p") is not None and fixed["p"] < sp.PROMOTE_P,
        "Sabit eşikte OOS onayı": oos["label"] == "CONFIRMED",
        "Ortak fiyat referansı": slip["label"] == "CONFIRMED",
    }
    return {"reviewed_at": time.time(), "model_id": gate.model_id, "kind": gate.kind,
            "interval": interval, "strategy": strategy, "threshold": threshold, "matched_rows": len(rows),
            "eligible": all(checks.values()), "checks": checks, "shadow_report": report,
            "fixed_threshold": fixed, "oos": oos, "slip_control": slip,
            "note": "Yalnız bu model SHA-256 kimliği, strateji ve bar aralığıyla kaydedilmiş canlı kanıt kullanılır. Eski kimliksiz satırlar terfi kanıtı sayılmaz."}


def activate(approved: dict) -> dict:
    """İncelenen model değiştiyse veya kanıt gerilediyse terfi etme."""
    if time.time() - approved["reviewed_at"] > 1800 or not approved["eligible"]:
        raise ValueError("Uygun ve son 30 dakika içinde hazırlanmış terfi incelemesi gerekli.")
    current = review(approved["interval"])
    if not current["eligible"]:
        failed = [k for k, passed in current["checks"].items() if not passed]
        raise ValueError("Terfi reddedildi: " + ", ".join(failed))
    if any(current[k] != approved[k] for k in ("model_id", "strategy", "interval", "threshold")):
        raise ValueError("Model veya parametreler incelemeden sonra değişti; tekrar inceleyin.")
    policy = {k: current[k] for k in ("model_id", "strategy", "interval", "threshold")}
    policy.update(mode="enforce", revision=uuid.uuid4().hex, approved_at=time.time())
    with _LOCK:
        previous = _read()
        if previous:
            atomic_write_text(PATH.with_suffix(".previous.json"), json.dumps(previous))
        atomic_write_text(PATH, json.dumps(policy))
    return {"activated": True, "policy": policy,
            "note": "Terfi kaydedildi. Canlı süreçler sonraki uygun sinyalde uygular; uygulama kaydı durum kartında görünür."}


def demote() -> dict:
    with _LOCK:
        p = _read()
        if not p:
            raise ValueError("Etkin bir web terfi kaydı yok.")
        policy = {**p, "mode": "shadow", "revision": uuid.uuid4().hex, "approved_at": time.time()}
        atomic_write_text(PATH, json.dumps(policy))
    return {"success": True, "policy": policy, "message": "Gölge kip kaydedildi; sonraki uygun sinyalde uygulanır."}
