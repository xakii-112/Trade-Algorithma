<#
ESKI (Tur-106): kanonik betik tur17-shadow-gorev.ps1 - Tur-58'den beri saati ABD seansindan TURETIR
(DST'yi kendisi izler) ve Tur-106'dan beri Pzt-Cum kurar. Bunu degil onu calistir.

tur20-shadow-gorev.ps1 - Quant Beyin GOLGE-KOSUCU gorevini ABD SEANSINA hizalar ve HAFTA
SONUNU kapatir. tur17-shadow-gorev.ps1'in yerine gecer (ayni gorev adi, -Force ile uzerine
yazar).

NEDEN (denetim 2026-07-26, bulgu B-6 / plan maddesi O5):
  Eski tetik gunluk 09:30 TR + 1s araliklarla 7 saat = 06:30-13:30 UTC. ABD acilisi 13:30 UTC
  -> AI piyasa ACILMADAN dusunuyordu. Gunluk barlar seans disinda DEGISMEDIGI icin 8 turun
  6'si BIREBIR ayni sonucu uretti (kararlarin %75.8'i tekrar; HRP 8 turda yalnizca 2 farkli
  cikti). Ustelik tetik gunlukdu -> CUMARTESI de kostu (248 golge kararin tamami 2026-07-25
  Cumartesi'ne aitti).

  Seansa hizalanan turlarda o gunun OLUSMAKTA olan gunluk bari her saat guncellenir; boylece
  her tur GERCEKTEN farkli bir girdi gorur. Tekrar orani bu yuzden duser - tur sayisini
  azaltarak degil, turlari ANLAMLI kilarak.

  KABUL KRITERI (plan O5): ai_decisions.jsonl'de ayni gun+sembol+posterior tekrari %10 altina
  insin. Bu ancak gorev BIRKAC GUN kostuktan sonra olculebilir (asagidaki "Dogrulama" komutu).

SAAT DILIMI:
  Gorev YEREL saatle kurulur (TR = UTC+3, yaz/kis degismez). ABD seansi:
    - YAZ  (EDT, mart-kasim): 13:30-20:00 UTC -> 16:30-23:00 TR   <-- VARSAYILAN
    - KIS  (EST, kasim-mart): 14:30-21:00 UTC -> 17:30-24:00 TR
  Kis saatine gecince -At "17:30" ile bir kez yeniden calistir. (Otomatik DST takibi bilerek
  YOK: Gorev Zamanlayici yerel saati izler, ABD DST'sini degil; sessiz yanlis hizalama yerine
  ACIK bir parametre tercih edildi.)

ONEMLI - BU SCRIPT'I YONETICI (Administrator) PowerShell'de CALISTIR:
  tur17-shadow-gorev.ps1 ile AYNI bilinen kisit (Tur-13, 2026-07-17): bu makinede YENI gorev
  olusturma yukseltisiz reddediliyor (0x80070005). Gorevin KENDISI SYSTEM DEGIL, INTERACTIVE
  (mevcut kullanici) prensibiyle kaydedilir - SYSTEM hesabi OneDrive profiline erisemez.

Kullanim (Administrator PowerShell, repo kokunde):
    .\tur20-shadow-gorev.ps1                     # yaz saati: 16:30 TR'den 6.5 saat, Pzt-Cum
    .\tur20-shadow-gorev.ps1 -At "17:30"         # kis saati
    .\tur20-shadow-gorev.ps1 -DurationHours 6.5
    .\tur20-shadow-gorev.ps1 -Unregister
#>
param(
    [string]$At = "16:30",
    [double]$DurationHours = 6.5,
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
    Write-Host "UYARI: Bu script Administrator PowerShell ister (bkz. Tur-13 0x80070005 kisiti)." -ForegroundColor Yellow
    Write-Host "Baslat menusu -> 'PowerShell' -> sag tik -> Yonetici olarak calistir -> bu scripti yeniden calistir." -ForegroundColor Yellow
}

$action = New-ScheduledTaskAction -Execute $python `
    -Argument "-m nihai.ai.shadow_runner --once" `
    -WorkingDirectory $PSScriptRoot

# HAFTALIK tetik (Pzt-Cum) + saatlik tekrar. Gunluk tetik CUMARTESI de kosuyordu (B-6).
# PS5.1 kisiti (tur17 dersi): taze bir tetikte $trigger.Repetition.Interval DOGRUDAN
# atanamaz (PropertyNotFound) -> Repetition NESNESI bir -Once tetiginden KOPYALANIR.
$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $At
$repSource = New-ScheduledTaskTrigger -Once -At $At `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Minutes ([int][math]::Round($DurationHours * 60)))   # Tur-106 (BULGU-106-37): -Hours Int32 alir, 6.5 -> 6
$trigger.Repetition = $repSource.Repetition

# Interactive current-user (SYSTEM DEGIL) - OneDrive altindaki repo + .venv erisimi icin sart.
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 20)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "Quant Beyin GOLGE-KOSUCU: ABD seansinda saatlik shadow_runner --once (Pzt-Cum, KARAR-DISI)" `
    -Force | Out-Null

$t = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -eq $t) { throw "KAYIT BASARISIZ: $taskName olusmadi (politika/izin engeli?)" }

Write-Host "GOREV KAYITLI: $taskName" -ForegroundColor Green
Write-Host "  tetik : Pzt-Cum $At'ten itibaren $DurationHours saat, saatte 1 kez (ABD seansi)"
Write-Host "  kimlik: interactive current-user (SYSTEM DEGIL)"
Get-ScheduledTask -TaskName $taskName | Format-List TaskName, State

Write-Host ""
Write-Host "Elle test (tek tur, hemen):  Start-ScheduledTask -TaskName '$taskName'"
Write-Host "Kosu ozeti:  data\ai_shadow_runner_log.jsonl"
Write-Host "Dogrulama (birkac gun sonra, O5 kabul kriteri - tekrar orani %10 altinda mi):"
Write-Host "  .\.venv\Scripts\python.exe -m nihai.ai.shadow_runner --repeat-report"
Write-Host "Kaldirma:    .\tur20-shadow-gorev.ps1 -Unregister"
Write-Host ""
Write-Host "NOT: Gorev su an DISABLED durumda olabilir (30 Temmuz 2026 itibariyle oyleydi)." -ForegroundColor Yellow
Write-Host "     Etkinlestirme:  Enable-ScheduledTask -TaskName '$taskName'" -ForegroundColor Yellow
Write-Host "NOT: AI_DECISION_MODE .env'de 'shadow' degilse gorev calisir ama golge birikim OLMAZ." -ForegroundColor Yellow
