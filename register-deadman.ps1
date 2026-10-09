<#
register-deadman.ps1 - deadman.ps1'i Gorev Zamanlayici'ya 15 DAKIKALIK gorev olarak
kaydeder.

NEDEN INTERACTIVE, SYSTEM DEGIL (Tur-12 dersi): SYSTEM hesabi kullanicinin OneDrive
profiline erisemez; bu depo OneDrive altindadir. register-forward-harness.ps1 ile AYNI
model kullanilir -> YONETICI/UAC GEREKMEZ.

NEDEN 15 DAKIKA: deadman'in kendi bayatlik sinirlari 30/60 dk'dir; daha seyrek bir tetik
uyariyi sinirin cok otesine kaydirirdi, daha sik olmasi ise bir sey kazandirmaz.

Kullanim:
    .\register-deadman.ps1                    # her 15 dk (yalniz oturum acikken)
    .\register-deadman.ps1 -Minutes 30
    .\register-deadman.ps1 -Send              # bayatsa Telegram uyarisi da gonder
    .\register-deadman.ps1 -Unregister        # gorevi kaldir
#>
param(
    [int]$Minutes = 15,
    [switch]$Send,
    [switch]$Ping,
    [switch]$Unregister
)
$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$wrapper   = Join-Path $scriptDir "deadman.ps1"
$taskName  = "NihaiAlgo-Deadman-15dk"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "GOREV KALDIRILDI: $taskName"
    return
}

if (-not (Test-Path $wrapper)) { throw "deadman.ps1 bulunamadi: $wrapper" }

$extra = ""
if ($Send) { $extra += " -Send" }
if ($Ping) { $extra += " -Ping" }

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument ('-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"{1}' -f $wrapper, $extra)

# Her $Minutes dakikada bir tekrarlayan tetik + oturum acilisinda telafi.
# DIKKAT (bu makinede OLCULDU, register-predict-watchdog.ps1 ve register-watchdog-system.ps1
# ayni notu tasiyor): -RepetitionDuration ([TimeSpan]::MaxValue) GECERSIZ XML uretir ->
# "Gorev XML'si ... aralik disi bir deger iceriyor (P99999999DT23H59M59S)".
# Calisan yol: SONLU bir Duration ile olustur, sonra alani BOS birak = suresiz.
$repeat = New-ScheduledTaskTrigger -Once -At "00:03" `
    -RepetitionInterval (New-TimeSpan -Minutes $Minutes) `
    -RepetitionDuration (New-TimeSpan -Days 1)
$repeat.Repetition.Duration = ""
$trigger = @(
    $repeat,
    (New-ScheduledTaskTrigger -AtLogOn -User ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name))
)

# Mevcut kullanici, yalnizca oturum acikken -> admin/sifre gerekmez (Tur-12).
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
# Denetim saniyeler surer; 5 dk'lik limit takilan bir kosuyu tetiklerin onunu tikamadan keser.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "Olu adam anahtari: canli kanit akisi durduysa haber ver (seans kapaliyken susar)" -Force | Out-Null

Write-Host "GOREV KAYITLI: $taskName (her $Minutes dk, mevcut kullanici / oturum acikken)"
Write-Host "Elle test:  Start-ScheduledTask -TaskName '$taskName'"
Write-Host "Kaldirma:   .\register-deadman.ps1 -Unregister"
