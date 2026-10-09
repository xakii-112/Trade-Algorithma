<#
tur17-shadow-gorev.ps1 - Quant Beyin GOLGE-KOSUCU (nihai.ai.shadow_runner) icin Gorev
Zamanlayici gorevini TANIMLAR/KAYDEDER. Piyasa saatlerinde saatte 1 kez
".venv\Scripts\python.exe -m nihai.ai.shadow_runner --once" calistirir (calisma dizini
repo koku). Bu script yalnizca gorevi TANIMLAR/KAYDEDER - shadow_runner.py'nin KENDISI
sinyal/esik/META_GATE/execution'a DOKUNMAZ (bkz. nihai/ai/shadow_runner.py modul
docstring'i, ANAYASA bolumu); yalniz fiyat OKUR ve brain.think() cagirir.

ONEMLI - BU SCRIPT'I YONETICI (Administrator) PowerShell'de CALISTIR:
  tur13-servis-devri.ps1 / register-predict-watchdog.ps1 ile AYNI bilinen kisit (Tur-13,
  2026-07-17): bu makinede YENI Gorev Zamanlayici gorevi olusturma yukseltisiz reddediliyor
  (0x80070005). ILK kayit icin Administrator PowerShell sart; SONRAKI guncellemeler
  (script'i tekrar calistirmak) yukseltisiz de calisir. Gorevin KENDISI SYSTEM DEGIL,
  INTERACTIVE (mevcut kullanici, oturum acikken) prensibiyle kaydedilir - CLAUDE.md/Tur-13
  dersinin aynisi: SYSTEM hesabi OneDrive profiline erisemez (repo OneDrive altinda),
  interactive-user hem OneDrive'a hem .venv'e sorunsuz erisir.

Kullanim (Administrator PowerShell, repo kokunde):
    .\tur17-shadow-gorev.ps1                          # ABD seansindan turetilen saat+sure, Pzt-Cum, saatte 1
    .\tur17-shadow-gorev.ps1 -At "09:30" -DurationHours 7
    .\tur17-shadow-gorev.ps1 -Unregister              # gorevi kaldir
#>
param(
    [string]$At = "",            # bos = ABD seansindan TURETILIR (Tur-58, Madde 43)
    [double]$DurationHours = 0,  # 0 = seans uzunlugundan TURETILIR
    [switch]$Unregister
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$taskName = "NihaiAlgo-ShadowRunner-Hourly"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "GOREV KALDIRILDI: $taskName"
    return
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "python.exe bulunamadi (venv kurulu mu?): $python" }

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "UYARI: Bu script Administrator PowerShell ister (ILK kayitta, bkz. Tur-13 0x80070005 kisiti)." -ForegroundColor Yellow
    Write-Host "Baslat menusu -> 'PowerShell' -> sag tik -> Yonetici olarak calistir -> bu scripti yeniden calistir." -ForegroundColor Yellow
    Write-Host "(Gorev SYSTEM DEGIL, interactive current-user olarak kaydedilecek - yalnizca KAYIT ADIMI yukseltme ister.)" -ForegroundColor Yellow
}

# main gorev icerigi: repo kokunde "python -m nihai.ai.shadow_runner --once" (sonsuz dongu
# YOK - her tetikte YENI/temiz bir Python sureci; shadow_runner.py docstring'inin gerekcesi).
$action = New-ScheduledTaskAction -Execute $python `
    -Argument "-m nihai.ai.shadow_runner --once" `
    -WorkingDirectory $PSScriptRoot

# Gunluk tetik + saatlik tekrar (piyasa saatleri penceresi).
#
# TUR-58 DUZELTMESI (Madde 43). Eski varsayilan "09:30 + 7 saat" idi ve yorumu
# "kabaca ABD piyasa saatlerinin yerel karsiligi" diyordu. OLCULDU ve YANLISTI:
# ABD seansi TR saatiyle YAZ 16:30-23:00, KIS 17:30-00:00'dir. Yani 09:30-16:30
# penceresi seans ACILMADAN ONCE bitiyordu - golge kosucu her gun KAPALI piyasada
# kosuyordu. Ayrica ABD ve TR yaz saatine farkli tarihlerde gectigi icin sabit bir
# saat yilda iki kez ayrica 1 saat kayar (raporun G maddesi).
#
# Cozum tahmin degil OLCUM: tetik saati ve pencere uzunlugu, borsanin O GUNKU
# takviminden turetilir (`market_calendar.local_session_window`). Kullanici yine
# -At/-DurationHours ile elle ezebilir.
if (-not $At) {
    $At = & $python -m nihai.data.market_calendar --at 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $At) {
        throw "Seans penceresi olculemedi (bugun piyasa kapali olabilir). Hafta ici bir gunde calistir ya da -At ver."
    }
    $At = $At.Trim()
    Write-Host "Tetik saati ABD seansindan turetildi: $At (yerel)" -ForegroundColor Cyan
}
if ($DurationHours -le 0) {
    $d = & $python -m nihai.data.market_calendar --duration 2>$null
    $DurationHours = if ($LASTEXITCODE -eq 0 -and $d) { [double]$d.Trim() } else { 6.5 }
}
# Tur-106 (BULGU-106-22): -Daily hafta sonu da kosuyordu; tur20'nin B-6 dersi (Cumartesi turlari kapali
# piyasada ayni gunluk bari yeniden yazar) bu betige hic tasinmamisti -> Pzt-Cum.
# PS5.1 kisiti: taze tetikte $trigger.Repetition.Interval DOGRUDAN atanamaz
# (PropertyNotFound). Bilinen desen: Once tetigini -RepetitionInterval ile kur, Repetition
# NESNESINI haftalik tetige kopyala.
$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $At
# Tur-106 (BULGU-106-37): `New-TimeSpan -Hours` Int32 alir -> 6.5 saat SESSIZCE 6'ya (yarim gun 3.5 -> 4'e)
# yuvarlaniyordu: betik "6.5 saat" yazip 6 saat kaydediyor, son seans saatinin kosusu dusuyordu. Dakikayla kur.
$repSource = New-ScheduledTaskTrigger -Once -At $At `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Minutes ([int][math]::Round($DurationHours * 60)))
$trigger.Repetition = $repSource.Repetition

# Interactive current-user (SYSTEM DEGIL) - forward-harness/predict-watchdog deseniyle ayni
# gerekce: OneDrive altindaki repo + .venv'e SYSTEM baglaminda erisilemez.
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 20)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "Quant Beyin GOLGE-KOSUCU: saatlik shadow_runner --once (KARAR-DISI, yalniz fiyat okur)" `
    -Force | Out-Null

$t = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -eq $t) { throw "KAYIT BASARISIZ: $taskName olusmadi (politika/izin engeli?)" }

Write-Host "GOREV KAYITLI: $taskName (interactive current-user, Pzt-Cum $At'ten itibaren $DurationHours saat, saatte 1 kez)" -ForegroundColor Green
Get-ScheduledTask -TaskName $taskName | Format-List TaskName, State
Write-Host "Elle test (tek tur, hemen):  Start-ScheduledTask -TaskName '$taskName'"
Write-Host "Koşu ozeti:  data\ai_shadow_runner_log.jsonl  (JSONL, en son satir en son kosu)"
Write-Host "Kaldirma:    .\tur17-shadow-gorev.ps1 -Unregister"
Write-Host "NOT: AI_DECISION_MODE .env'de 'shadow' degilse gorev calisir ama golge birikim OLMAZ (ozette 'note' alani uyarir)."
