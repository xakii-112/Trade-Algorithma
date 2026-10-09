<#
hub-watchdog.ps1 - keeps hub_server.py alive.

The live system reads from hub_server via DATA_PROVIDER=hub. If the hub goes down,
the system silently falls back to yfinance (slow, rate-limited) and the live scan
degrades. This watchdog checks the hub periodically and, if it is down, restarts it
using the same single-process .venv mechanism as run-hub.ps1.

Usage:
    .\hub-watchdog.ps1                     # check every 30s
    .\hub-watchdog.ps1 -IntervalSec 15     # check more often
    .\hub-watchdog.ps1 -Port 8799

Stop: Ctrl+C in the window.
To keep it running across reboots/logins, register it in Task Scheduler with an
"at log on" / "at startup" trigger.
#>
param(
    [int]$Port = 8799,
    [int]$IntervalSec = 30,
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
$wdLog  = Join-Path $logDir "hub_watchdog.log"
$hubOut = Join-Path $logDir "hub_server.out.log"
$hubErr = Join-Path $logDir "hub_server.err.log"
# KASITLI kapatma bayragi (proje-kapat.ps1 koyar, proje-ac.ps1 siler).
# Gorevin kendisi yukseltisiz devre disi birakilamadigi icin (0x80070005) durdurmanin
# kalici olmasinin TEK yolu budur; yoksa watchdog 10 dk icinde her seyi geri diriltir.
$stopFlag = Join-Path $logDir "project.stopped"

function Write-WdLog([string]$msg) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line  = "$stamp | $msg"
    Write-Output $line
    Add-Content -Path $wdLog -Value $line -Encoding utf8
}

# Redirect logları (hub_server.{out,err}.log) rotasyonsuz buyur; uzun saglikli kosuda
# stderr spam'i ile GB'lara cikabilir (gozlem: 500MB). Her dongude boyut tavanini asani
# truncate et (inode korunur; Start-Process redirect handle'i gecerli kalir).
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

# Fast port check (Test-NetConnection is slow; use raw TcpClient with 1s timeout)
function Test-HubUp([int]$p) {
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

function Start-Hub {
    # Denetim Tur-11: port kapali ama surec YASIYOR olabilir (asili/yavas acilis).
    # Oldurmeden yeni surec baslatmak her dongude bir python biriktirir - once temizle.
    $alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*nihai.data.hub_server*' }
    if ($alive) {
        foreach ($p in $alive) {
            Write-WdLog ("WARN: hub process alive (PID {0}) but port {1} closed - killing stale process" -f $p.ProcessId, $Port)
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Seconds 2
    }
    Write-WdLog "hub DOWN - restarting..."
    $env:__PYVENV_LAUNCHER__ = $venvPy
    # Tur-106 (BULGU-106-19): Start-Process yonlendirmesi err.log'u SIFIRLAR; asili/coken surecin
    # stderr'i (uvicorn kapanisi, Traceback) tek kanittir. 2026-10-02 23:39 asili hub'in nedeni boyle silindi.
    if ((Test-Path $hubErr) -and (Get-Item $hubErr).Length -gt 0) { Move-Item -LiteralPath $hubErr -Destination ($hubErr -replace '\.log$', '.prev.log') -Force -ErrorAction SilentlyContinue }
    $proc = Start-Process -FilePath $basePy -ArgumentList "-m","nihai.data.hub_server" -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden -RedirectStandardOutput $hubOut -RedirectStandardError $hubErr
    Write-WdLog "hub started (PID $($proc.Id)) - waiting for it to come up..."
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Seconds 1
        if (Test-HubUp $Port) { Write-WdLog "hub UP (port $Port) OK"; return }
    }
    Write-WdLog "WARN: hub did not come up within 20s - will retry next cycle"
}

Write-WdLog "watchdog started | port=$Port | check every $IntervalSec s"
$wasUp = $true
$stopped = $false
while ($true) {
    # STOP bayragi: kullanici projeyi KASITLI kapatti -> diriltme. Bayrak kalkinca kendiliginden
    # devam eder (surec olmeden), boylece proje-ac.ps1 sonrasi keep-alive aninda geri gelir.
    if ((Test-Path $stopFlag) -or (Test-Path -LiteralPath (Join-Path $logDir 'hub.web-stopped'))) {
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
    if (Test-HubUp $Port) {
        if (-not $wasUp) { Write-WdLog "hub back UP OK" }
        $wasUp = $true
    } else {
        $wasUp = $false
        Start-Hub
    }
    Limit-LogSize $hubErr $MaxLogMB
    Limit-LogSize $hubOut $MaxLogMB
    Start-Sleep -Seconds $IntervalSec
}
