<#
register-watchdog-system.ps1 - hub-watchdog.ps1'i Gorev Zamanlayici'ya kaydeder.

Tur-12 DUZELTMESI (denetim bulgusu HIGH): eskiden SYSTEM/AtStartup + UAC-self-elevate idi.
Iki kritik sorun vardi:
  1. SYSTEM hesabi kullanicinin OneDrive profiline (proje + .venv orada) ERISEMEZ ->
     watchdog baslangicta coker, hub'i asla yeniden baslatamaz (keep-alive FIILEN YOK).
  2. Elevated degilse UAC ile kendini yeniden baslatip 'return' ediyordu; UAC reddi VEYA
     elevated-cocugun hatasi SESSIZCE yutuluyordu (gorev hic olusmadan cikis 0) -> pratikte
     yalnizca ForwardHarness gorevi kayitli kaldi (gozlenen durum).
Cozum: ForwardHarness ile AYNI kanitlanmis desen -> MEVCUT KULLANICI, Interactive (yalnizca
oturum acikken), admin/UAC GEREKMEZ, OneDrive + ag erisimi TAM. Tetik AtLogOn (surekli watchdog).
NOT: gorev adi tarihsel nedenle "-SYSTEM" kaldi (stop-hub.ps1 vb. bu adi referans aliyor) ama
principal artik INTERACTIVE current-user'dir; ad yalniz tanimlayici.

Kullanim:
    .\register-watchdog-system.ps1              # kaydet + hemen baslat + dogrula
    .\register-watchdog-system.ps1 -Unregister  # gorevi kaldir
#>
param([switch]$Unregister)
$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$watchdog  = Join-Path $scriptDir "hub-watchdog.ps1"
$taskName  = "NihaiAlgo-HubWatchdog-SYSTEM"

if ($Unregister) {
    # BULGU (2026-07-26): burada -ErrorAction SilentlyContinue + KOSULSUZ "GOREV KALDIRILDI"
    # vardi. Yukseltisiz oturumda Unregister 0x80070005 (Erisim engellendi) ile SESSIZCE
    # basarisiz olurken betik yine BASARILI raporluyordu. Bir AI ajani bu cikitya guvenip
    # gorevin kalktigini sandi, gorev tetiklenmeye devam edince de watchdog .ps1'lerini
    # "exit 0" ile bosaltarak "cozdu" (256 satir kayip). Artik gercek hata yakalanir ve
    # kaldirma DOGRULANIR; gorev duruyorsa LOUD hata (kayit yolundaki desenin esi, bkz. asagi).
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

if (-not (Test-Path $watchdog)) { throw "hub-watchdog.ps1 bulunamadi: $watchdog" }

$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument ('-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $watchdog)
# Tur-13 OLCUM: bu makinede YENI gorev OLUSTURMA yukseltisiz reddediliyor (0x80070005;
# Daily tetikle de denendi) - mevcut gorevi GUNCELLEME ise calisiyor. Yani ILK kayit
# yonetici PowerShell ister (bkz. tur13-servis-devri.ps1); sonraki guncellemeler istemez.
# TETIK: Daily 00:05 + AtLogOn (Tur-18 DUZELTMESI).
# Eskiden yalniz Daily+StartWhenAvailable idi ve yorum "AtLogOn ile ayni pratik kapsama"
# diyordu - YANLIS. StartWhenAvailable yalnizca KACIRILAN tetigi telafi eder (makine o
# saatte kapaliysa). Gun ici reboot'ta 00:05 tetigi O GUN ZATEN KOSMUS sayilir -> kacirilmis
# degildir -> telafi calismaz. Olculen sonuc: 2026-07-18 03:00 reboot'unda iki servis de
# 13:46'ya kadar (10+ saat) OLU kaldi. AtLogOn bu bosluğu kapatir.
# MultipleInstances varsayilani IgnoreNew -> calisan watchdog varken ikinci ornek baslamaz.
#
# BULGU B-2 (2026-07-25): yukaridaki Tur-18 karari CANLI GOREVE HIC ISLENMEMISTI -
# olcum aninda gorevin tek tetigi Daily idi. Ustelik AtLogOn tek basina yetmez: 07:53'te
# olen watchdog, oturum acik kaldigi icin AtLogOn de gormedi ve 00:05'e kadar olu kalirdi.
# Bu yuzden asil self-healing katmani her 10 dk TEKRARLAYAN tetiktir (-Once + suresiz
# RepetitionDuration); AtLogOn reboot'u, StartWhenAvailable kacirilan tetigi kapatir.
# DIKKAT (olculdu): -RepetitionDuration ([TimeSpan]::MaxValue) bu makinede GECERSIZ XML
# uretiyor (P99999999DT23H59M59S). Calisan yol: sonlu Duration + alani BOS birak = suresiz.
$repeat = New-ScheduledTaskTrigger -Once -At "00:05" `
    -RepetitionInterval (New-TimeSpan -Minutes 10) `
    -RepetitionDuration (New-TimeSpan -Days 1)
$repeat.Repetition.Duration = ""
$trigger = @(
    $repeat,
    (New-ScheduledTaskTrigger -AtLogOn -User ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name))
)
# Mevcut kullanici, yalnizca oturum acikken (Interactive) -> admin/UAC gerekmez, OneDrive erisir.
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
# ExecutionTimeLimit 0 = sonsuz (watchdog surekli doner); cokerse 1 dk'da bir yeniden dene.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "hub_server.py canli tutucu (interactive current-user, AtLogOn)" -Force | Out-Null

# Cift watchdog olmasin: eski kullanici-seviyesi Startup kisayolunu kaldir (varsa).
$lnk = Join-Path ([Environment]::GetFolderPath('Startup')) 'NihaiAlgo-HubWatchdog.lnk'
if (Test-Path $lnk) { Remove-Item $lnk -Force; Write-Host "Eski Startup kisayolu kaldirildi." }

# Hemen baslat + DOGRULA -> sessiz basarisizlik yok: gorev olusmadiysa LOUD hata firlat.
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 2
$t = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -eq $t) { throw "KAYIT BASARISIZ: $taskName olusmadi (politika/izin engeli?)" }
Write-Host "GOREV KAYITLI + BASLATILDI: $taskName (interactive current-user / AtLogOn)"
Get-ScheduledTask -TaskName $taskName | Format-List TaskName, State
Write-Host "Kaldirma: .\register-watchdog-system.ps1 -Unregister"
