<#
register-forward-harness.ps1 - forward-harness.ps1'i Gorev Zamanlayici'ya GUNLUK gorev
olarak kaydeder. predict-watchdog (SYSTEM/AtStartup, surekli) kayit BIRIKTIRIR; bu gorev
ise gunde bir kez DEGERLENDIRIR + history.csv'ye snapshot ekler (zaman serisi).

predict-watchdog'dan farkli olarak MEVCUT KULLANICI hesabiyla, "yalnizca oturum acikken"
(Interactive) calisir -> YONETICI/UAC GEREKMEZ, OneDrive yolu + ag (yfinance) erisimi tam.

Kullanim:
    .\register-forward-harness.ps1                 # her gun 23:30 + 12:00 (yedek)
    .\register-forward-harness.ps1 -At "22:10"
    .\register-forward-harness.ps1 -WakeToRun      # uyuyan makineyi tetik icin uyandir
    .\register-forward-harness.ps1 -Unregister     # gorevi kaldir
#>
param(
    [string]$At = "23:30",
    [string]$NoonAt = "12:00",
    [switch]$Unregister,
    [switch]$WakeToRun
)
$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$wrapper   = Join-Path $scriptDir "forward-harness.ps1"
$taskName  = "NihaiAlgo-ForwardHarness-Daily"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "GOREV KALDIRILDI: $taskName"
    return
}

if (-not (Test-Path $wrapper)) { throw "forward-harness.ps1 bulunamadi: $wrapper" }

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument ('-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $wrapper)
# Tur-13: iki gunluk tetik. 23:30'da makine uyursa StartWhenAvailable uykuda tetiklemez
# (gozlem: 2026-07-14/15 history bosluklari) -> ogle yedegi gunun kosusunu garanti eder.
# Iki satir/gun zararsiz: history.csv append-only zaman serisi (07-05'te de cift satir vardi).
$trigger  = New-ScheduledTaskTrigger -Daily -At $At
$trigger2 = New-ScheduledTaskTrigger -Daily -At $NoonAt
# Mevcut kullanici, yalnizca oturum acikken (Interactive) -> admin/sifre gerekmez.
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
# Tur-13: 30 dk limiti 2026-07-16 19:53 kosusunu birikmis resolve backlog'unda SESSIZCE
# oldurdu (banner var, history satiri yok) -> 4 saate cikarildi.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 4)
# Tur-88: 2026-09-21 22:51 -> 09-22 14:24 uykusu IKI tetigi birden kacirdi (Missed=2) ve
# StartWhenAvailable uyanista da kosmadi -> ogrenme hafizasi ~28 saat bayat kaldi. Opt-in, cunku
# dizustunu gece uyandirmak istenmeyebilir (guc planinda uyandirma zamanlayicilari acik olmali).
if ($WakeToRun) { $settings.WakeToRun = $true }

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger @($trigger, $trigger2) `
    -Principal $principal -Settings $settings `
    -Description "Forward harness: gunluk resolve+snapshot+report -> history.csv zaman serisi" -Force | Out-Null

Write-Host "GOREV KAYITLI: $taskName (gunluk $At, mevcut kullanici / oturum acikken)"
Write-Host "Elle test:  Start-ScheduledTask -TaskName '$taskName'"
Write-Host "Kaldirma:   .\register-forward-harness.ps1 -Unregister"
