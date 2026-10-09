"""Yerel servis gözlemi ve mevcut bakım araçları; emir yolu içermez."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import Field, field_validator

from nihai.atomic_io import atomic_write_text, tail_lines
from nihai.config import DATA_DIR, HUB_URL, LOG_DIR
from nihai.web import lab_jobs as jobs
from nihai.web.api_lab import Input, normalize_symbols, operator_request

router = APIRouter(dependencies=[Depends(operator_request)])
ROOT = Path(__file__).resolve().parents[2]
MAINHUB = ROOT.parent / "MainHub"


def _hub_address():
    parsed = urlparse(HUB_URL)
    if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Süreç yönetimi yalnız yerel hub için kullanılabilir.")
    return parsed


def hub_request(route: str) -> dict:
    from nihai.net import shared_client
    _hub_address()
    response = shared_client().get(HUB_URL.rstrip("/") + route, timeout=3)
    response.raise_for_status()
    return response.json()


@router.get("/api/hub/streams")
def hub_streams():
    try:
        return {"ok": True, **hub_request("/streams")}
    except Exception as exc:
        return {"ok": False, "reason": jobs.redact(str(exc)), "streams": None}


class ConfirmInput(Input):
    confirmed: bool = False


@router.post("/api/ml/demote-shadow")
def demote_shadow(req: ConfirmInput):
    if not req.confirmed:
        raise HTTPException(422, "Gölge kipe dönüş için onay gerekli.")
    from nihai.signals.model_promotion import demote
    try:
        return demote()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/api/hub/{action}", status_code=202)
def hub_control(action: str, req: ConfirmInput):
    if action not in ("start", "stop", "restart", "sync"):
        raise HTTPException(404, "Bilinmeyen hub işlemi.")
    if not req.confirmed:
        raise HTTPException(422, "Servis işlemi için onay gerekli.")
    return jobs.start("hub", {"action": action})


def _powershell(script: str) -> str:
    if os.name != "nt":
        raise RuntimeError("Windows Görev Zamanlayıcı bu işletim sisteminde bulunmuyor.")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, encoding="utf-8", errors="replace", timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW, check=False)
    if result.returncode:
        raise RuntimeError(jobs.redact(result.stderr.strip())[:600])
    return result.stdout.strip()


@router.get("/api/system/watchdogs")
def watchdogs():
    result = {"deadman": deadman_status(), "tasks": [], "logs": {}}
    try:
        raw = _powershell("""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$tasks = @(Get-ScheduledTask | Where-Object { $_.TaskName -like 'NihaiAlgo-*' } | ForEach-Object {
    $info = Get-ScheduledTaskInfo -InputObject $_
    [pscustomobject]@{name=$_.TaskName; state=[string]$_.State; last_run=$info.LastRunTime.ToString('o');
        next_run=$info.NextRunTime.ToString('o'); last_result=$info.LastTaskResult; missed=$info.NumberOfMissedRuns}
})
ConvertTo-Json -InputObject $tasks -Compress
""")
        result["tasks"] = json.loads(raw or "[]")
        result["measured"] = True
    except Exception as exc:
        result.update(measured=False, reason=jobs.redact(str(exc)))
    for name in ("deadman.log", "predict_watchdog.log", "hub_watchdog.log", "forward-harness.log"):
        path = LOG_DIR / name
        result["logs"][name] = {"modified_at": path.stat().st_mtime,
                                "tail": jobs.redact("\n".join(tail_lines(path, 30)))} if path.exists() else None
    ping = DATA_DIR / "deadman_web_ping.json"
    result["last_manual_ping"] = json.loads(ping.read_text(encoding="utf-8")) if ping.exists() else None
    return result


@router.get("/api/system/deadman")
def deadman_status():
    from nihai.notifier.deadman import check
    grace = float(os.getenv("DEADMAN_GRACE_MIN", "5"))
    result = check(grace_min=grace, data_dir=DATA_DIR, log_dir=LOG_DIR)
    return {**result, "checked_at": time.time(), "grace_min": grace}


@router.post("/api/system/deadman/ping")
def deadman_ping(req: ConfirmInput):
    if not req.confirmed:
        raise HTTPException(422, "Dış ping için onay gerekli.")
    from nihai.notifier.deadman import _snitch, check
    result = check(data_dir=DATA_DIR, log_dir=LOG_DIR)
    result.update(sent=False, checked_at=time.time())
    url = os.getenv("DEADMAN_SNITCH_URL", "").strip()
    if result["verdict"] != "SAGLIKLI" or any(v["age_min"] is None for v in result["files"].values()):
        result["reason"] = "Kanıt akışı doğrulanmadı; ping gönderilmedi."
    elif not url:
        result["reason"] = "Dış deadman adresi yapılandırılmamış."
    else:
        result["sent"] = _snitch(url)
        result["reason"] = "Ping gönderildi." if result["sent"] else "Ping gönderilemedi."
    jobs.write_json(DATA_DIR / "deadman_web_ping.json", result)
    return result


@router.get("/api/telegram/status")
def telegram_status(probe: bool = False):
    from nihai.notifier import telegram_bot
    return telegram_bot.status(probe=probe)


@router.post("/api/telegram/send-test")
def telegram_test(req: ConfirmInput):
    if not req.confirmed:
        raise HTTPException(422, "Test mesajı için onay gerekli.")
    from nihai.notifier import telegram_bot
    if not telegram_bot.is_enabled():
        raise HTTPException(409, "Telegram yapılandırılmamış.")
    sent = telegram_bot._post("Nihai Algoritma web paneli: test bildirimi.")
    return {"success": sent, "message": "Telegram kabul etti." if sent else "Gönderim başarısız; durum kartını inceleyin."}


class ChartInput(Input):
    symbol: str
    interval: Literal["1m", "5m", "15m", "1h", "1d"] = "1h"
    lookback: int = Field(default=120, ge=40, le=400)

    @field_validator("symbol")
    @classmethod
    def symbol_value(cls, value):
        return normalize_symbols([value])[0]


@router.post("/api/telegram/chart", status_code=202)
def telegram_chart(req: ChartInput):
    return jobs.start("telegram_chart", req.model_dump())


def _chart_job(job_id: str):
    path = jobs.job_dir(job_id)
    request = json.loads((path / "request.json").read_text(encoding="utf-8"))
    if (request["kind"] != "telegram_chart" or jobs.status(job_id)["status"] != "completed"
            or not (path / "chart.png").is_file()):
        raise HTTPException(404, "Tamamlanmış grafik bulunamadı.")
    return path, ChartInput(**request["parameters"])


@router.get("/api/telegram/chart/{job_id}.png")
def telegram_chart_png(job_id: str):
    path, _ = _chart_job(job_id)
    return FileResponse(path / "chart.png", media_type="image/png", filename="chart.png")


@router.post("/api/telegram/chart/{job_id}/send")
def telegram_chart_send(job_id: str, req: ConfirmInput):
    if not req.confirmed:
        raise HTTPException(422, "Grafik gönderimi için onay gerekli.")
    path, chart = _chart_job(job_id)
    from nihai.notifier import telegram_bot
    if not telegram_bot.is_enabled():
        raise HTTPException(409, "Telegram yapılandırılmamış.")
    sent = telegram_bot._post_photo((path / "chart.png").read_bytes(), f"{chart.symbol} · {chart.interval}")
    return {"success": sent, "message": "Grafik Telegram'a gönderildi." if sent else "Grafik gönderilemedi."}


@router.get("/api/strategy/status")
def strategy_status():
    from nihai.runtime_strategy import settings
    from nihai.signals.strategy import STRATEGY_MODES
    return {**settings(), "strategies": list(STRATEGY_MODES)}


@router.post("/api/strategy/set-active")
def strategy_set(req: dict):
    from nihai.runtime_strategy import save
    try:
        return save(req)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/api/guards/blocked-signals")
def blocked_signals():
    from collections import Counter, deque

    from nihai.signals.signal_engine import _today_log
    # Tur-77: dosya adı YAZICININ kuralıyla (yerel tarih) kurulur. ET tarihi TR 00:00–07:00
    # arasında bir gün geride kalır ve okuyucu dünkü dosyayı "bugün" diye gösterirdi.
    path = _today_log()
    rows, counts = deque(maxlen=200), Counter()
    if path.exists():
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                for code in ("HALT_LULD", "SSR_NOTICE"):
                    if code in line:
                        counts[code] += 1
                        rows.append({"code": code, "blocked": code == "HALT_LULD", "detail": line.strip()})
                        break
    return {"rows": list(rows), "counts": dict(counts), "scope": "Bugünün tamamı; son 200 kayıt",
            "file": path.name, "note": "SSR bu sistemde uyarıdır; HALT/LULD engeldir."}


@router.get("/api/decision/funnel")
def decision_funnel(tarih: str | None = None):
    """Tur-77 · Madde 6.4b — günün KARAR HUNİSİ (hangi guard kaç denemeyi eledi). KARAR-DIŞI.
    `tarih` = YYYY-MM-DD (yazıcının yerel tarih adlandırması); verilmezse bugünün dosyası."""
    import re

    from nihai.signals.signal_engine import _today_log, karar_hunisi
    if tarih is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", tarih):
        raise HTTPException(422, "tarih YYYY-MM-DD olmalı")
    path = _today_log() if tarih is None else LOG_DIR / f"decision_log_{tarih.replace('-', '_')}.txt"
    sonuc = karar_hunisi(path)
    if sonuc is None:
        return {"ok": False, "reason": "LOG_YOK", "dosya": path.name}
    return {"ok": True, **sonuc,
            "not": "birim tarama denemesidir (aynı sembol her turda yeniden taranır); "
                   "sembol_en_ileri her sembolün gün içinde ulaştığı en ileri aşamayı sayar"}


@router.get("/api/system/tiers")
def tier_status():
    import nihai.config as config
    from nihai.execution import watchlist
    rows = []
    for tier in ("odak", "genis"):
        path = DATA_DIR / f"tier_{tier}.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"tier": tier, "measured": False})
    return {"enabled": config.DUAL_TIER_ENABLED, "tiers": rows, **watchlist.partition(config.FOCUSED_TOP_N)}


def _hub_pid(health: dict) -> int:
    root = Path(health.get("service_root", "")).resolve()
    if not health.get("service_root") or root not in (ROOT.resolve(), MAINHUB.resolve()) or type(health.get("pid")) is not int:
        raise ValueError("Hub süreç kimliği doğrulanamadı. Güncel hub sürümünü yerel başlatıcıyla açın.")
    pid = health["pid"]
    if pid <= 0 or pid == os.getpid():
        raise ValueError("Geçersiz hub süreç kimliği.")
    return pid


def _stop_hub():
    pid = _hub_pid(hub_request("/health"))
    for log_dir in (LOG_DIR, MAINHUB / "logs"):
        log_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(log_dir / "hub.web-stopped", "Panelden durduruldu\n")
        (log_dir / "hub.should-run").unlink(missing_ok=True)
    if os.name == "nt":
        result = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                                timeout=20, creationflags=subprocess.CREATE_NO_WINDOW, check=False)
        if result.returncode:
            raise RuntimeError("Hub durdurulamadı; süreç yetkisini kontrol edin.")
    else:
        import signal
        os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        try:
            hub_request("/health")
        except Exception:
            return
        time.sleep(0.2)
    raise RuntimeError("Hub durma isteğinden sonra hâlâ yanıt veriyor.")


def _start_hub():
    """Kanonik veri hub'ını başlatır: `python -m nihai.data.hub_server` (bu depo) — `proje-ac.ps1` ve hub gözcüsüyle
    AYNI süreç. Tur-106 (BULGU-106-23): eskiden `MainHub/hub_server.py`'yi MainHub'ın kendi venv'i ve `.env`'iyle
    açıyordu; o kopya türetilmiş bir artefakttır (Tur-58), `/quote` yanıtında `spread_measured` taşımaz (kayma
    modeli her kotasyonu ölçülmemiş sayar), Tur-106 UTF-8 düzeltmesi yoktur ve gözcü/`proje-kapat` onu komut
    satırından (`nihai.data.hub_server`) tanımaz."""
    parsed = _hub_address()
    try:
        health = hub_request("/health")
    except Exception:
        health = None
    if health:
        _hub_pid(health)
        return {"already_running": True, "health": health}
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise ValueError("Proje Python ortamı (.venv) bulunamadı.")
    for log_dir in (LOG_DIR, MAINHUB / "logs"):
        (log_dir / "hub.web-stopped").unlink(missing_ok=True)
    log = LOG_DIR / "hub_web_start.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HUB_HOST=parsed.hostname, HUB_PORT=str(parsed.port or 80))
    env.pop("__PYVENV_LAUNCHER__", None)
    with log.open("ab") as stream:
        proc = subprocess.Popen([str(python), "-m", "nihai.data.hub_server"], cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Hub başlatılırken kapandı; logs/hub_web_start.log dosyasını inceleyin.")
        try:
            health = hub_request("/health")
            _hub_pid(health)
            return {"health": health}
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Hub başladı ancak 20 saniye içinde sağlık yanıtı vermedi.")


def execute_operation(kind: str, params: dict, path: Path):
    if kind == "telegram_chart":
        from nihai.notifier.chart_render import render_price_png
        chart = ChartInput(**params)
        png = render_price_png(**chart.model_dump())
        if not png:
            raise ValueError("Grafik üretilemedi; yeterli bar ve çizim bağımlılıklarını kontrol edin.")
        (path / "chart.png").write_bytes(png)
        return {"chart_job": path.name, "symbol": chart.symbol, "interval": chart.interval}
    if kind != "hub" or params.get("action") not in ("start", "stop", "restart", "sync"):
        raise ValueError("Bilinmeyen işlem.")
    action = params["action"]
    if action == "sync":
        from tools.sync_mainhub import generate
        return {"updated": generate(apply=True), "note": "Yenilenen dosyalar hub yeniden başladığında kullanılır."}
    if action in ("stop", "restart"):
        _stop_hub()
    return _start_hub() if action in ("start", "restart") else {"stopped": True}
