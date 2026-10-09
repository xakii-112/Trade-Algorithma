<#
deadman.ps1 - OLU ADAM ANAHTARI'ni TEK SEFER calistirir ve ciktiyi
logs/deadman.log'a ekler.

NEDEN BU SCRIPT VAR (Tur-61, BULGU-61-7): nihai/notifier/deadman.py Tur-59'da tam da
"servisler sessizce oldu" durumunu yakalamak icin yazildi, ama onu CALISTIRAN hicbir
sarmalayici ve hicbir gorev YOKTU -> modul uretimde HIC kosmadi. Olculdu (2026-08-31):
uc servis portu da KAPALI, predictions_log 35 saat, live_features_log 87 saat bayat ve
hicbir uyari cikmadi. Yazilip KOSULMAYAN bir izleyici, izleme degildir.

Modulun kendi sozlesmesi korunur: SEANS KAPALIYKEN uyarmaz (yanlis alarm uretmez) ve
hicbir karar vermez - yalnizca olcer ve gerekiyorsa haber verir.

Elle:
    .\deadman.ps1                 # denetle + logla
    .\deadman.ps1 -Send           # bayatsa Telegram uyarisi
    .\deadman.ps1 -Send -Ping     # + sagliklyysa dis snitch ping'i
    .\deadman.ps1 -MaxLogMB 5
#>
param(
    [switch]$Send,
    [switch]$Ping,
    [int]$MaxLogMB = 5
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# Tur-24: cikti dosyaya yonlendirildiginde stdout cp1254'e duser ve ok/emoji karakterinde
# surec COKER. Cocuk python'un da UTF-8 yazmasi icin PYTHONIOENCODING sart.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$env:PYTHONIOENCODING = "utf-8"

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "venv python bulunamadi: $venvPy (.venv kurulu mu?)" }

$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$log = Join-Path $logDir "deadman.log"

# 15 dakikada bir kosar -> log sinirsiz buyumemeli.
if ((Test-Path $log) -and ((Get-Item $log).Length -gt ($MaxLogMB * 1MB))) {
    Clear-Content -Path $log
}

$pyArgs = @("-m", "nihai.notifier.deadman")
if ($Send) { $pyArgs += "--send" }
if ($Ping) { $pyArgs += "--ping" }

# PS 5.1 tuzagi (forward-harness.ps1 ile ayni): ErrorActionPreference=Stop altinda native
# komutun 2>&1 yonlendirmesi, python stderr'e tek satir yazsa bile NativeCommandError
# firlatir -> cagri suresince Continue'ya dus, cikis kodunu kendimiz oku.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$out = & $venvPy @pyArgs 2>&1 | ForEach-Object { $_.ToString() } | Out-String
$code = $LASTEXITCODE
$ErrorActionPreference = $prevEAP

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $log -Value "[$stamp] $($out.Trim())" -Encoding utf8
Write-Output $out
if ($code -ne 0) {
    Add-Content -Path $log -Value "[$stamp] [HATA] deadman cikis kodu: $code" -Encoding utf8
}
