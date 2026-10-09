<#
start-hub.ps1 — NihaiAlgoritma veri hub'ini (nihai.data.hub_server) DENETIMLI baslatir.

DAVRANIS (istenen):
  * Sen baslatinca (bu script) -> hub calisir ve COKERSE otomatik yeniden baslar.
  * Sen kapatinca (stop-hub.ps1, Ctrl+C ya da pencereyi kapat) -> hub durur ve
    BIR DAHA otomatik baslamaz. Bilgisayar acilisinda da kendiliginden BASLAMAZ.

Nasil calisir: bir "bayrak" dosyasi (logs\hub.should-run) varligi "hub ayakta
olmali" demektir. Bu script hub'i on planda denetler; hub kendiliginden olurse
(crash) ve bayrak hala duruyorsa 5 sn sonra yeniden baslatir. stop-hub.ps1 bayragi
silince denetim dongusu temiz cikar (restart YOK).

Kullanim:
    .\start-hub.ps1
    .\start-hub.ps1 -RestartDelaySec 10
Durdurma:
    Bu pencerede Ctrl+C  YA DA  baska pencerede  .\stop-hub.ps1
#>
param(
    [int]$RestartDelaySec = 5
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$cfg    = Join-Path $PSScriptRoot ".venv\pyvenv.cfg"
if (-not (Test-Path $cfg)) { throw "pyvenv.cfg bulunamadi - .venv kurulu mu?" }
$basePy = (Select-String -Path $cfg -Pattern '^\s*executable\s*=\s*(.+?)\s*$').Matches.Groups[1].Value
if (-not (Test-Path $basePy)) { throw "Base python bulunamadi: $basePy" }

$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$flag = Join-Path $logDir "hub.should-run"
$log  = Join-Path $logDir "hub_supervisor.log"

function Write-Log([string]$msg) {
    $line = "{0} | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

# Bayragi olustur = "hub ayakta olmali"
Remove-Item -LiteralPath (Join-Path $logDir 'hub.web-stopped') -Force -ErrorAction SilentlyContinue
Set-Content -Path $flag -Value (Get-Date -Format o) -Encoding utf8
$env:__PYVENV_LAUNCHER__ = $venvPy

Write-Log "start-hub: denetim basladi (auto-restart AKTIF). Durdurmak icin Ctrl+C ya da stop-hub.ps1"
try {
    while ((Test-Path $flag) -and -not (Test-Path -LiteralPath (Join-Path $logDir 'hub.web-stopped'))) {
        & $basePy -m nihai.data.hub_server @args
        $code = $LASTEXITCODE
        if (-not (Test-Path $flag)) {
            Write-Log "stop istegi alindi (bayrak silindi) - yeniden baslatilmiyor."
            break
        }
        Write-Log "hub beklenmedik sekilde durdu (Exit $code) - $RestartDelaySec sn sonra yeniden basliyor..."
        Start-Sleep -Seconds $RestartDelaySec
    }
}
finally {
    # Ctrl+C / pencere kapatma da buraya duser -> bayragi temizle, restart olmasin
    if (Test-Path $flag) { Remove-Item $flag -Force -ErrorAction SilentlyContinue }
    Write-Log "start-hub: denetim sonlandi (auto-restart DEVRE DISI)."
}
