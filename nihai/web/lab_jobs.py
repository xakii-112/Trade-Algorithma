"""Panel işleri: sınırlı sayıda ayrı süreç, kalıcı sonuç ve sonlu log kuyruğu."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import HTTPException

from nihai.atomic_io import atomic_write_text, tail_lines
from nihai.config import DATA_DIR

ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = Path(DATA_DIR) / "web_jobs"
_LOCK = threading.Lock()
_PROCESSES: dict[str, subprocess.Popen] = {}
MAX_RUNNING = 2
TIMEOUT_SECONDS = 4 * 3600


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items() if k != "model"}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    atomic_write_text(path, json.dumps(clean(value), ensure_ascii=False, allow_nan=False))


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(404, "İş bulunamadı.")
    path = JOBS_DIR / job_id
    if not (path / "request.json").is_file():
        raise HTTPException(404, "İş bulunamadı.")
    return path


def redact(text: str) -> str:
    for name, value in os.environ.items():
        if len(value) >= 8 and any(k in name.upper() for k in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "SNITCH")):
            text = text.replace(value, "[gizli]")
    return re.sub(r"bot\d+:[A-Za-z0-9_-]+", "bot[gizli]", text)


def status(job_id: str) -> dict:
    path = job_dir(job_id)
    meta = json.loads((path / "request.json").read_text(encoding="utf-8"))
    final = path / "result.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8"))
        meta.update({k: result[k] for k in ("status", "error", "finished_at") if k in result})
    else:
        with _LOCK:
            proc = _PROCESSES.get(job_id)
        # ponytail: süreç tutamaçları tek pano örneğine ait; yeniden başlatmada
        # bitiş dosyası gelene kadar canlılık iddia edilmez.
        meta["status"] = "running" if proc is not None and proc.poll() is None else "unobserved"
    meta["log"] = redact("\n".join(tail_lines(path / "output.log", 100))) if (path / "output.log").exists() else ""
    meta["elapsed_seconds"] = round(meta.get("finished_at", time.time()) - meta["created_at"], 1)
    return meta


def report(job_id: str) -> dict:
    path = job_dir(job_id) / "result.json"
    if not path.exists():
        raise HTTPException(409, "İş henüz sonuçlanmadı.")
    return json.loads(path.read_text(encoding="utf-8"))


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=20,
                       creationflags=subprocess.CREATE_NO_WINDOW, check=False)
    else:
        import signal
        os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=20)


def _monitor(job_id: str, proc: subprocess.Popen) -> None:
    path = job_dir(job_id)
    error = None
    try:
        code = proc.wait(timeout=TIMEOUT_SECONDS)
        if code:
            error = f"İşlem çıkış kodu: {code}. Logu inceleyin."
    except subprocess.TimeoutExpired:
        _stop(proc)
        error = "İş dört saat sınırında durduruldu."
    finally:
        with _LOCK:
            if not (path / "result.json").exists():
                write_json(path / "result.json", {"status": "failed", "error": error or "Sonuç üretilemedi.",
                                                 "finished_at": time.time()})
            _PROCESSES.pop(job_id, None)


def start(kind: str, parameters: dict) -> dict:
    with _LOCK:
        if sum(p.poll() is None for p in _PROCESSES.values()) >= MAX_RUNNING:
            raise HTTPException(409, "İki iş zaten çalışıyor; bitmesini bekleyin.")
        # ponytail: aynı türden tek iş; ortak dosya yazan araçlar için yeterli sınır.
        for active_id, proc in _PROCESSES.items():
            request = JOBS_DIR / active_id / "request.json"
            if proc.poll() is None and json.loads(request.read_text(encoding="utf-8"))["kind"] == kind:
                raise HTTPException(409, "Bu türden bir iş zaten çalışıyor.")
        job_id = uuid.uuid4().hex
        path = JOBS_DIR / job_id
        path.mkdir(parents=True)
        meta = {"job_id": job_id, "kind": kind, "parameters": parameters, "created_at": time.time()}
        write_json(path / "request.json", meta)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   NIHAI_DATA_DIR=str(DATA_DIR))
        try:
            with (path / "output.log").open("wb") as output:
                proc = subprocess.Popen([sys.executable, "-u", "-m", "nihai.web.lab_jobs", job_id],
                    cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    start_new_session=os.name != "nt")
        except OSError as exc:
            write_json(path / "result.json", {"status": "failed", "error": redact(str(exc)),
                                               "finished_at": time.time()})
            raise HTTPException(503, "İş süreci başlatılamadı.") from exc
        _PROCESSES[job_id] = proc
    threading.Thread(target=_monitor, args=(job_id, proc), daemon=True).start()
    return {"job_id": job_id, "status": "running"}


def cancel(job_id: str) -> dict:
    path = job_dir(job_id)
    with _LOCK:
        proc = _PROCESSES.get(job_id)
        if proc is None or proc.poll() is not None:
            raise HTTPException(409, "Bu pano örneğine ait çalışan iş bulunamadı.")
        _stop(proc)
        final = path / "result.json"
        if final.exists():
            return {"job_id": job_id, "status": json.loads(final.read_text(encoding="utf-8"))["status"]}
        write_json(final, {"status": "cancelled", "finished_at": time.time()})
        return {"job_id": job_id, "status": "cancelled"}


def main() -> None:
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    path = job_dir(sys.argv[1])
    request = json.loads((path / "request.json").read_text(encoding="utf-8"))
    try:
        from nihai import runtime_strategy
        from nihai.web.api_lab import execute
        runtime_strategy.PATH = path / "runtime_strategy.json"
        result = execute(request["kind"], request["parameters"], path)
        output = {"status": "completed", "result": result}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        output = {"status": "failed", "error": redact(f"{type(exc).__name__}: {exc}")}
    output["finished_at"] = time.time()
    write_json(path / "result.json", output)


if __name__ == "__main__":
    main()
