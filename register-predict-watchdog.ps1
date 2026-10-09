<#
register-predict-watchdog.ps1 - predict-watchdog.ps1'i (tahmin KAYDEDICI canli tutucu)
Gorev Zamanlayici'ya kaydeder. register-watchdog-system.ps1'in (hub icin) esi; ikisi BIRLIKTE
calismali: hub-watchdog veriyi, predict-watchdog tahmin kaydini canli tutar.

Tur-12 DUZELTMESI (denetim bulgusu HIGH + MED): eskiden SYSTEM/AtStartup + UAC-self-elevate idi
ve kayittan sonra Start-ScheduledTask CAGRILMIYORDU. Uc sorun:
  1. SYSTEM kullanicinin OneDrive profiline erisemez -> watchdog coker (keep-alive yok).
  2. UAC-self-elevate edip 'return' -> UAC reddi/hata sessizce yutuluyordu (gorev olusmuyordu).
  3. Start yoktu -> kayit olsa bile gorev bir sonraki ACILISA kadar "Ready" bekliyordu (dormant).
Cozum: ForwardHarness deseni -> MEVCUT KULLANICI, Interactive (oturum acikken), AtLogOn,
admin/UAC gerekmez; kayittan sonra HEMEN baslat + DOGRULA.
NOT: gorev adi tarihsel "-SYSTEM" kaldi (referanslar bozulmasin) ama principal INTERACTIVE'dir.

Kullanim:
    .\register-predict-watchdog.ps1              # kaydet + hemen baslat + dogrula
    .\register-predict-watchdog.ps1 -Unregister  # gorevi kaldir
#>
param([switch]$Unregister)
$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$watchdog  = Join-Path $scriptDir "predict-watchdog.ps1"
$taskName  = "NihaiAlgo-PredictWatchdog-SYSTEM"

if ($Unregister) {
    # BULGU (2026-07-26): bkz. register-watchdog-system.ps1 icindeki ayni duzeltme.
    # SilentlyContinue + kosulsuz basari mesaji = sessiz hata; kaldirma artik DOGRULANIR.
    $err = $null
    try   { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop }
    catch { $err = $_.Exception.Message }

    if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
        throw ("KALDIRMA BASARISIZ: '{0}' HALA KAYITLI. Sebep: {1}`n" -f $taskName, $err) +
              ("Yukseltisiz oturum bu gorevi degistiremez. YONETICI PowerShell'de:`n" ) +
              ("    Unregister-ScheduledTask -TaskName {0} -Confirm:`$false" -f $taskName)
    }
    if ($err) { Write-Host "GOREV ZATEN YOKTU: $taskName" -ForegroundColor DarkYellow }
    else      { Write-Host "GOREV KALDIRILDI (dogrulandi): $taskName" -ForegroundColor Green }
    return
}

if (-not (Test-Path $watchdog)) { throw "predict-watchdog.ps1 bulunamadi: $watchdog" }

$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument ('-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $watchdog)
# Tur-13: bu makinede YENI gorev olusturma yukseltisiz reddediliyor (0x80070005) ->
# ILK kayit yonetici ister (tur13-servis-devri.ps1); guncelleme istemez.
# Tur-18: tetik Daily 00:05 + AtLogOn. "StartWhenAvailable = AtLogOn esdegeri" varsayimi
# YANLISTI (yalniz kacirilan tetigi telafi eder, gun ici reboot'u DEGIL) - tam gerekce ve
# olculen kanit: register-watchdog-system.ps1.
#
# BULGU B-2 (2026-07-25) - OLCULEN KANIT: canli gorevde AtLogOn tetigi YOKTU (yalniz
# Daily), yani yukaridaki Tur-18 karari yazildi ama HIC UYGULANMADI. Sonuc: watchdog
# sureci 07:53'te oldu (LastTaskResult 0xC000013A = CTRL+BREAK ile sonlandirma; oturum
# bitisi/uyku) ve bir sonraki 00:05'e kadar OLU kaldi -> 10 SAATLIK gozetimsiz pencere.
# Kayitci da o sirada dusmustu; Cumartesi oldugu icin veri kaybi olmadi, is gunu olsa
# seansin yarisi kaydedilmezdi.
#
# Kalici cozum tek bir tetik degil, UC katman:
#   1. Her 10 dk TEKRARLAYAN tetik (asil self-healing): watchdog NEDEN olurse olsun
#      en gec 10 dk icinde geri gelir. MultipleInstances=IgnoreNew oldugundan zaten
#      calisiyorken tetiklenen kopya sessizce yok sayilir (guvenli).
#   2. AtLogOn: oturum acilisinda hemen ayaga kalk (reboot sonrasi bekleme yok).
#   3. StartWhenAvailable: kacirilan tetigi (makine kapaliydi) telafi et.
# Tekrarlama -Once + SURESIZ Duration ile kalici olur; -Daily'de tekrarlama gunluk
# pencereyle sinirli kalirdi (ShadowRunner'da oldugu gibi: PT1H/PT7H).
# DIKKAT (olculdu): -RepetitionDuration ([TimeSpan]::MaxValue) bu makinede GECERSIZ XML
# uretiyor -> "Gorev XML'si ... aralik disi bir deger iceriyor (P99999999DT23H59M59S)".
# Calisan yol: sonlu bir Duration ile olustur, sonra alani BOS birak = suresiz.
$repeat = New-ScheduledTaskTrigger -Once -At "00:05" `
    -RepetitionInterval (New-TimeSpan -Minutes 10) `
    -RepetitionDuration (New-TimeSpan -Days 1)
$repeat.Repetition.Duration = ""
$trigger = @(
    $repeat,
    (New-ScheduledTaskTrigger -AtLogOn -User ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name))
)
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "main --serve-web tahmin kaydedici canli tutucu (interactive current-user, AtLogOn)" -Force | Out-Null

# Hemen baslat + DOGRULA (MED bulgusu: eskiden Start yoktu -> gorev dormant kaliyordu).
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 2
$t = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -eq $t) { throw "KAYIT BASARISIZ: $taskName olusmadi (politika/izin engeli?)" }
Write-Host "GOREV KAYITLI + BASLATILDI: $taskName (interactive current-user / AtLogOn)"
Get-ScheduledTask -TaskName $taskName | Format-List TaskName, State
Write-Host "Birlikte calismasi gereken hub gorevi: NihaiAlgo-HubWatchdog-SYSTEM"
Write-Host "Kaldirma: .\register-predict-watchdog.ps1 -Unregister"
