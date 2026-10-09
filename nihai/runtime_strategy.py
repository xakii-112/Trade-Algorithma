"""Süreçler arasında kalıcı operatör seçimi; yeni sinyalde okunur."""
from __future__ import annotations

import json
import math
import threading
from dataclasses import replace

from loguru import logger

from nihai import config
from nihai.atomic_io import atomic_write_text

PATH = config.DATA_DIR / "runtime_strategy.json"
_LOCK = threading.RLock()
_stamp = None
_overrides: dict = {}


def validate(value: dict) -> dict:
    from nihai.signals.strategy import STRATEGY_MODES
    if set(value) != {"strategy", "min_votes", "stop_atr", "target_atr"}:
        raise ValueError("Strateji, asgari oy, stop ATR ve hedef ATR gerekli.")
    if value["strategy"] not in STRATEGY_MODES:
        raise ValueError("Geçersiz strateji.")
    if type(value["min_votes"]) is not int or not 2 <= value["min_votes"] <= 7:
        raise ValueError("Asgari oy 2–7 arasında tamsayı olmalı.")
    for key, upper in (("stop_atr", 20), ("target_atr", 100)):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 < number <= upper:
            raise ValueError(f"{key}: 0'dan büyük, en fazla {upper} olmalı.")
    return dict(value)


def settings() -> dict:
    global _stamp, _overrides
    with _LOCK:
        try:
            stat = PATH.stat()
            stamp = (str(PATH), stat.st_mtime_ns, stat.st_size)
            if stamp != _stamp:
                _overrides = validate(json.loads(PATH.read_text(encoding="utf-8")))
                _stamp = stamp
        except FileNotFoundError:
            _overrides, _stamp = {}, None
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(f"[strateji] operatör seçimi okunamadı: {type(exc).__name__}")
        return {"strategy": config.ACTIVE_STRATEGY, "min_votes": config.CONFLUENCE_MIN_VOTES,
                "stop_atr": config.CONFIG.strategy.trend_stop_atr,
                "target_atr": config.CONFIG.strategy.trend_target_atr, **_overrides}


def apply() -> str:
    with _LOCK:
        selected = settings()
        if _overrides:
            # Stop/hedef ATR çifti tek nesne değişiminde yayınlanır.
            object.__setattr__(config.CONFIG, "strategy", replace(config.CONFIG.strategy,
                trend_stop_atr=selected["stop_atr"], trend_target_atr=selected["target_atr"]))
            config.CONFLUENCE_MIN_VOTES = selected["min_votes"]
        return selected["strategy"]


def save(value: dict) -> dict:
    selected = validate(value)
    with _LOCK:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(PATH, json.dumps(selected, ensure_ascii=False))
    return {"success": True, **selected, "message": "Kaydedildi; yeni sinyallerde uygulanır."}
