<#
predict-watchdog.ps1 - keeps the prediction RECORDER (main --serve-web) alive.

WHY THIS EXISTS (gap):
  hub-watchdog.ps1 keeps hub_server.py (the DATA server) alive, but recording of
  predictions happens in the LivePredictService, which only runs inside
  `main --serve-web` (web_dashboard.start_web_server -> get_service().start()).
  Without this process, hub serves real-time data but NOTHING writes to
  data/predictions_log.csv -> no out-of-sample evidence accumulates.

  This watchdog mirrors hub-watchdog.ps1: it checks the web port (8765) and, if the
  recorder is down, restarts it with the same single-process .venv mechanism. It also
  periodically hits /api/predictions/accuracy?resolve=true so matured predictions get
  LABELLED continuously (turns raw predictions into scored evidence + live verdict).

  Pair it with hub-watchdog.ps1 (start hub FIRST so the recorder reads real-time data;
  if hub is down the recorder silently degrades to delayed yfinance).

Usage:
    .\predict-watchdog.ps1                      # check every 30s, resolve every 5 min
    .\predict-watchdog.ps1 -IntervalSec 15
    .\predict-watchdog.ps1 -Port 8765 -ResolveEverySec 300

Stop: Ctrl+C in the window.
To keep it running across reboots/logins, register it in Task Scheduler with an
"at startup" trigger (see register-predict-watchdog.ps1).
#>
param(
    [int]$Port = 8765,
    [int]$IntervalSec = 30,
    [int]$ResolveEverySec = 300,
    [int]$MaxLogMB = 50
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$cfg    = Join-Path $PSScriptRoot ".venv\pyvenv.cfg"
if (-not (Test-Path $cfg)) { throw "pyvenv.cfg not found - is .venv installed?" }
$basePy = (Select-String -Path $cfg -Pattern '^\s*executable\s*=\s*(.+?)\s*$').Matches.Groups[1].Value
if (-not (Test-Path $basePy)) { throw "base python not found: $basePy" }

$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$wdLog  = Join-Path $logDir "predict_watchdog.log"
$svcOut = Join-Path $logDir "predict_serve.out.log"
$svcErr = Join-Path $logDir "predict_serve.err.log"
# KASITLI kapatma bayragi (proje-kapat.ps1 koyar, proje-ac.ps1 siler) - hub-watchdog.ps1 ile ayni.
$stopFlag = Join-Path $logDir "project.stopped"

function Write-WdLog([string]$msg) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line  = "$stamp | $msg"
    Write-Output $line
    Add-Content -Path $wdLog -Value $line -Encoding utf8
}

# Redirect logs grow unbounded on long healthy runs; truncate over the cap each cycle
# (inode preserved; Start-Process redirect handle stays valid).
function Limit-LogSize([string]$path, [int]$maxMB) {
    if ($maxMB -le 0) { return }
    try {
        if ((Test-Path $path) -and ((Get-Item $path).Length -gt ($maxMB * 1MB))) {
            Clear-Content -Path $path -Force -ErrorAction Stop
            Write-WdLog "log truncated (>$maxMB MB): $(Split-Path $path -Leaf)"
        }
    } catch {
        # Denetim Tur-11: sessiz catch truncate'in hic calismadigini gizliyordu.
        Write-WdLog ("WARN: truncate FAILED ({0}): {1}" -f (Split-Path $path -Leaf), $_.Exception.Message)
    }
}

# Fast port check (raw TcpClient with 1s timeout)
function Test-PortUp([int]$p) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $client.BeginConnect("127.0.0.1", $p, $null, $null)
        $ok  = $iar.AsyncWaitHandle.WaitOne(1000, $false)
        if ($ok -and $client.Connected) { return $true }
        return $false
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Start-Recorder {
    # Denetim Tur-11: port kapali ama surec YASIYOR olabilir (agir acilis/asili kalma).
    # Oldurmeden yeni surec baslatmak her dongude bir python biriktirir - once temizle.
    # NOT: yalniz --serve-web'li main eslesir; tarayici-only main surecine DOKUNULMAZ.
    $alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*nihai.execution.main*' -and $_.CommandLine -like '*--serve-web*' }
    if ($alive) {
        foreach ($p in $alive) {
            Write-WdLog ("WARN: recorder process alive (PID {0}) but port {1} closed - killing stale process" -f $p.ProcessId, $Port)
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Seconds 2
    }
    Write-WdLog "recorder DOWN - restarting (main --serve-web)..."
    $env:__PYVENV_LAUNCHER__ = $venvPy
    # Tur-106 (BULGU-106-19): Start-Process yonlendirmesi err.log'u SIFIRLAR; asili/coken surecin
    # stderr'i (uvicorn kapanisi, Traceback) tek kanittir. 2026-10-02 23:39 asili hub'in nedeni boyle silindi.
    if ((Test-Path $svcErr) -and (Get-Item $svcErr).Length -gt 0) { Move-Item -LiteralPath $svcErr -Destination ($svcErr -replace '\.log$', '.prev.log') -Force -ErrorAction SilentlyContinue }
    $proc = Start-Process -FilePath $basePy `
        -ArgumentList "-m","nihai.execution.main","--serve-web" `
        -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $svcOut -RedirectStandardError $svcErr
    Write-WdLog "recorder started (PID $($proc.Id)) - waiting for web port $Port..."
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if (Test-PortUp $Port) { Write-WdLog "recorder UP (port $Port) OK"; return }
    }
    Write-WdLog "WARN: recorder did not open port $Port within 30s - will retry next cycle"
}

# Hit the accuracy endpoint so matured predictions get resolved/labelled (look-ahead-safe).
function Invoke-Resolve([int]$p) {
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$p/api/predictions/accuracy?resolve=true" `
            -TimeoutSec 60 -ErrorAction Stop
        $hr = if ($null -ne $r.hit_rate) { "{0:P1}" -f $r.hit_rate } else { "n/a" }
        $rv = if ($r.resolve) { $r.resolve } else { "" }
        Write-WdLog "resolve OK | resolved_total=$($r.resolved) hit_rate=$hr $rv"
    } catch {
        Write-WdLog "resolve skipped: $($_.Exception.Message)"
    }
}

Write-WdLog "predict-watchdog started | port=$Port | check every $IntervalSec s | resolve every $ResolveEverySec s"
$wasUp = $true
$lastResolve = [DateTime]::MinValue
$stopped = $false
while ($true) {
    # STOP bayragi: kullanici projeyi KASITLI kapatti -> ne dirilt ne resolve et.
    if (Test-Path $stopFlag) {
        if (-not $stopped) { Write-WdLog "STOP bayragi bulundu (logs\project.stopped) - yeniden baslatma DURDURULDU" }
        $stopped = $true
        Start-Sleep -Seconds $IntervalSec
        continue
    }
    if ($stopped) {
        Write-WdLog "STOP bayragi kalkti - izleme yeniden AKTIF"
        $stopped = $false
        $wasUp = $true
    }
    if (Test-PortUp $Port) {
        if (-not $wasUp) { Write-WdLog "recorder back UP OK" }
        $wasUp = $true
        if (((Get-Date) - $lastResolve).TotalSeconds -ge $ResolveEverySec) {
            Invoke-Resolve $Port
            $lastResolve = Get-Date
        }
    } else {
        $wasUp = $false
        Start-Recorder
    }
    Limit-LogSize $svcErr $MaxLogMB
    Limit-LogSize $svcOut $MaxLogMB
    Start-Sleep -Seconds $IntervalSec
}
