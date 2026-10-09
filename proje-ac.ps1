<#
proje-ac.ps1 - Nihai Algoritma'nin TAMAMINI acar (hub + kayitci/web panel).

proje-kapat.ps1'in tersi: iki servisi baslatir, portlarin gercekten acildigini DOGRULAR,
SONRA logs\project.stopped bayragini kaldirir ve paneli tarayicida acar.

Watchdog gorevleri zaten her 10 dk tetikleniyor; bayrak kalktigi an keep-alive kendiliginden
devreye girer. Bayrak servisler baslatilirken kaldirilmaz: yoksa watchdog ayni anda ikinci bir
surec baslatabilir (bkz. adim 3 ustundeki not).

Python baslatma mekanizmasi hub-watchdog.ps1 ile AYNI (tek-surec .venv deseni:
base python + __PYVENV_LAUNCHER__), boylece watchdog'un baslattigiyla birebir ayni surec olur.

Kullanim:
    .\proje-ac.ps1
    .\proje-ac.ps1 -NoBrowser -NoPause
#>
param([switch]$NoBrowser, [switch]$NoPause)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$cfg    = Join-Path $PSScriptRoot ".venv\pyvenv.cfg"
if (-not (Test-Path $cfg)) { throw "pyvenv.cfg bulunamadi - .venv kurulu mu? ($cfg)" }
$basePy = (Select-String -Path $cfg -Pattern '^\s*executable\s*=\s*(.+?)\s*$').Matches.Groups[1].Value
if (-not (Test-Path $basePy)) { throw "base python bulunamadi: $basePy" }

$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$stopFlag = Join-Path $logDir "project.stopped"

function Test-PortUp([int]$p) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $client.BeginConnect("127.0.0.1", $p, $null, $null)
        $ok  = $iar.AsyncWaitHandle.WaitOne(1000, $false)
        return ($ok -and $client.Connected)
    } catch { return $false } finally { $client.Close() }
}

function Start-Svc([string]$label, [string[]]$moduleArgs, [int]$port, [string]$outLog, [string]$errLog, [int]$waitSec) {
    if (Test-PortUp $port) { Write-Host ("      {0}: zaten ayakta (port {1})" -f $label, $port) -ForegroundColor DarkGray; return $true }
    $env:__PYVENV_LAUNCHER__ = $venvPy
    # Tur-106 (BULGU-106-19): Start-Process yonlendirmesi err.log'u SIFIRLAR; asili/coken surecin
    # stderr'i (uvicorn kapanisi, Traceback) tek kanittir. 2026-10-02 23:39 asili hub'in nedeni boyle silindi.
    if ((Test-Path $errLog) -and (Get-Item $errLog).Length -gt 0) { Move-Item -LiteralPath $errLog -Destination ($errLog -replace '\.log$', '.prev.log') -Force -ErrorAction SilentlyContinue }
    $proc = Start-Process -FilePath $basePy -ArgumentList $moduleArgs `
        -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    Write-Host ("      {0}: baslatildi (PID {1}), port {2} bekleniyor..." -f $label, $proc.Id, $port) -ForegroundColor DarkYellow
    for ($i = 0; $i -lt $waitSec; $i++) {
        Start-Sleep -Seconds 1
        if (Test-PortUp $port) { return $true }
    }
    return $false
}

Write-Host ""
Write-Host "=== NIHAI ALGORITMA - ACILIYOR ===" -ForegroundColor Cyan
Write-Host ""

# Bayrak servisler ayaga kalkana kadar YERINDE kalir: bayrak varken watchdog'lar ne baslatir ne
# oldurur. Eskiden once bayrak kalkiyordu; 2026-09-17'de predict-watchdog 2 sn icinde kayitciyi
# baslatti, bu betigin port kontrolu import asamasindaki sureci goremedi ve IKINCI kayitci acildi
# (iki surec ayni acik pozisyonlari ayri ayri kapatip trades.csv'ye cift satir yazdi).

# (1) Hub (veri saglayici, 8799). Once bu: kayitci DATA_PROVIDER=hub ile buradan okuyor.
Write-Host "[1/3] Veri hub'i (8799)..." -ForegroundColor Yellow
$hubOk = Start-Svc "hub" @("-m", "nihai.data.hub_server") 8799 `
    (Join-Path $logDir "hub_server.out.log") (Join-Path $logDir "hub_server.err.log") 25

# (2) Kayitci + web panel (8765)
Write-Host "[2/3] Kayitci + web panel (8765)..." -ForegroundColor Yellow
$webOk = Start-Svc "kayitci" @("-m", "nihai.execution.main", "--serve-web") 8765 `
    (Join-Path $logDir "predict_serve.out.log") (Join-Path $logDir "predict_serve.err.log") 40

# (3) Bayragi kaldir -> watchdog'lar ayakta olan portlari gorur, keep-alive yeniden aktif.
if (Test-Path $stopFlag) {
    Remove-Item $stopFlag -Force
    Write-Host "[3/3] STOP bayragi kaldirildi -> keep-alive yeniden aktif." -ForegroundColor Yellow
} else {
    Write-Host "[3/3] STOP bayragi zaten yoktu." -ForegroundColor DarkYellow
}

# Tur-106 (BULGU-106-23): Laboratuvar "Durdur" logs\hub.web-stopped birakir ve hub gozcusu bu bayrak varken
# diriltmez. proje-ac her seyi actigina gore bu bayrak da kalkar; yoksa hub'in bir sonraki cokusu kalici olur.
Remove-Item -LiteralPath (Join-Path $logDir 'hub.web-stopped') -Force -ErrorAction SilentlyContinue

# Watchdog gorevlerini hemen tetiklemeyi dene (izin yoksa sorun degil: 10 dk'da bir zaten tetikleniyor).
foreach ($t in "NihaiAlgo-HubWatchdog-SYSTEM", "NihaiAlgo-PredictWatchdog-SYSTEM") {
    try { Start-ScheduledTask -TaskName $t -ErrorAction Stop } catch { }
}

Write-Host ""
if ($hubOk -and $webOk) {
    Write-Host "PROJE ACIK." -ForegroundColor Green
    Write-Host "  Panel : http://127.0.0.1:8765" -ForegroundColor Gray
    Write-Host "  Hub   : http://127.0.0.1:8799" -ForegroundColor Gray
    if (-not $NoBrowser) { Start-Process "http://127.0.0.1:8765" }
} else {
    if (-not $hubOk) { Write-Host "HATA: hub 8799 portunu acmadi. Bak: logs\hub_server.err.log" -ForegroundColor Red }
    if (-not $webOk) { Write-Host "HATA: kayitci 8765 portunu acmadi. Bak: logs\predict_serve.err.log" -ForegroundColor Red }
    Write-Host "(Watchdog 10 dk icinde tekrar deneyecek.)" -ForegroundColor DarkYellow
}
Write-Host ""

if (-not $NoPause) {
    Write-Host "Kapatmak icin bir tusa basin..." -ForegroundColor DarkGray
    try { $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown') } catch { Start-Sleep -Seconds 5 }
}
