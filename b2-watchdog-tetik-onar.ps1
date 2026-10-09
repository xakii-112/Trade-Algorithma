<#
b2-watchdog-tetik-onar.ps1 - BULGU B-2 tek-komut onarimi (YONETICI olarak calistir).

NEDEN GEREKLI (olculen kanit, 2026-07-25):
  * Canli gorevlerin tetigi YALNIZ "Daily 00:05" idi. register-*.ps1'in Tur-18'de
    ekledigi AtLogOn tetigi goreve HIC islenmemisti (script yonetici koşusu bekliyordu).
  * predict-watchdog sureci 07:53'te oldu (LastTaskResult 0xC000013A = CTRL+BREAK ile
    sonlandirma). Bir sonraki tetik ertesi gun 00:05 oldugundan 10 SAAT gozetimsiz kaldi;
    kayitci da o pencerede dustu. Cumartesi oldugu icin veri kaybi olmadi - is gunu olsa
    seansin buyuk bolumu kaydedilmezdi.
  * Set-ScheduledTask yukseltisiz "Erisim engellendi" veriyor -> bu script yonetici ister.

NE YAPAR: iki watchdog gorevine UC katmanli tetik yazar
  1. her 10 dk SURESIZ tekrar  -> watchdog neden olurse olsun en gec 10 dk'da geri gelir
  2. AtLogOn                   -> reboot/oturum acilisinda hemen ayaga kalkar
  3. (mevcut) StartWhenAvailable -> kacirilan tetigi telafi eder
MultipleInstances=IgnoreNew oldugundan zaten calisirken gelen tetik sessizce yok sayilir.

Kullanim (yonetici PowerShell):
    .\b2-watchdog-tetik-onar.ps1
    .\b2-watchdog-tetik-onar.ps1 -WhatIf   # yalniz ne yapacagini goster
#>
param([switch]$WhatIf)
$ErrorActionPreference = "Stop"

$gorevler = @("NihaiAlgo-PredictWatchdog-SYSTEM", "NihaiAlgo-HubWatchdog-SYSTEM")
$me = [System.Security.Principal.WindowsIdentity]::GetCurrent()
$yonetici = (New-Object System.Security.Principal.WindowsPrincipal $me).IsInRole(
    [System.Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $yonetici) {
    Write-Warning "Bu script YONETICI PowerShell ister (Set-ScheduledTask aksi halde 'Erisim engellendi')."
    Write-Host    "Yonetici PowerShell ac ve tekrar calistir:"
    Write-Host    "    cd '$PSScriptRoot'; .\b2-watchdog-tetik-onar.ps1"
    return
}

function New-SuresizTekrarTetik {
    # DIKKAT: -RepetitionDuration ([TimeSpan]::MaxValue) bu makinede GECERSIZ XML uretir
    # (P99999999DT23H59M59S). Calisan yol: sonlu Duration ver, sonra alani BOS birak.
    $t = New-ScheduledTaskTrigger -Once -At "00:05" `
        -RepetitionInterval (New-TimeSpan -Minutes 10) `
        -RepetitionDuration (New-TimeSpan -Days 1)
    $t.Repetition.Duration = ""
    return $t
}

foreach ($n in $gorevler) {
    $mevcut = Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue
    if ($null -eq $mevcut) { Write-Warning "$n bulunamadi - once register-*.ps1 calistir."; continue }

    $tetikler = @(
        (New-SuresizTekrarTetik),
        (New-ScheduledTaskTrigger -AtLogOn -User $me.Name)
    )

    if ($WhatIf) { Write-Host "[WhatIf] $n -> her 10 dk suresiz tekrar + AtLogOn"; continue }

    Set-ScheduledTask -TaskName $n -Trigger $tetikler | Out-Null

    $t = Get-ScheduledTask -TaskName $n
    $ozet = $t.Triggers | ForEach-Object {
        $c = $_.CimClass.CimClassName -replace 'MSFT_Task', ''
        if ($_.Repetition.Interval) {
            $sure = if ($_.Repetition.Duration) { $_.Repetition.Duration } else { "SURESIZ" }
            "$c(her $($_.Repetition.Interval) / $sure)"
        } else { $c }
    }
    Write-Host ("OK  {0,-34} -> {1}" -f $n, ($ozet -join " + "))

    # Gorev su an calismiyorsa hemen ayaga kaldir (dormant kalmasin).
    if ((Get-ScheduledTask -TaskName $n).State -ne "Running") { Start-ScheduledTask -TaskName $n }
}

if (-not $WhatIf) {
    Write-Host ""
    Write-Host "DOGRULAMA - 10 dk sonra watchdog loglarinda yeni satir gormelisin:"
    Write-Host "    Get-Content logs\predict_watchdog.log -Tail 3"
    Write-Host "Gercek sinav: watchdog surecini elle oldur, 10 dk icinde kendiliginden donmeli."
}
