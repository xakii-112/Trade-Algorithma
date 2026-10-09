<#
saat-onar.ps1 - SISTEM SAATININ kaymasini onarir (Tur-75 BULGU / Tur-76 acik kalemi).

NEDEN BU SCRIPT VAR: Tur-75 bu makinenin saatini olctu ve ~650 ms GERIDE buldu; Tur-76
yeniden olctu, kayma -965 ms'ye CIKMISTI (yani sorun kendiliginden duzelmiyor, BUYUYOR).
Teshis Tur-76'da yapildi ve kok neden "bir kez resync unutulmus" DEGIL:

    Get-Service W32Time  ->  Status: Stopped / StartType: Manual

Windows Saat hizmeti KAPALI ve elle-baslatmaya ayarli. Yani makineyi saati tutan hicbir
sey yok; her yeniden baslatmada kayma yeniden birikmeye basliyor. Bu yuzden script
yalnizca `w32tm /resync` CALISTIRMAZ, hizmeti KALICI olarak (Automatic) da acar - aksi
halde onarim bir sonraki yeniden baslatmada geri alinirdi.

NEDEN ONEMLI: bu depoda karar loglari, bar zaman damgalari ve tahmin cozumleri UTC'ye
gore kiyaslanir. 1 saniyeye yaklasan bir kayma; "bar kapandi mi", "sinyal hangi barda
uretildi", "tahmin hangi saniyede cozuldu" sorularinin hepsini ayni yonde kaydirir.

KULLANIM (YONETICI PowerShell gerekir):
    .\saat-onar.ps1              # onar + dogrula
    .\saat-onar.ps1 -OlcumSadece # HICBIR SEY DEGISTIRME, yalnizca olc ve raporla

IDDIA SINIRI: script "saat duzeldi" DEMEZ - onarim sonrasi kaymayi projenin KENDI
olcum araciyla (python -m nihai.audit_chain --clock) yeniden olcer ve SAYIYI basar.
Olcum yapilamazsa bunu acikca soyler; "0 ms kayma" ile "olcemedim" ayni sey degildir.
#>
param(
    [switch]$OlcumSadece,
    [int]$EsikMs = 200
)

$ErrorActionPreference = "Stop"
$kok = Split-Path -Parent $MyInvocation.MyCommand.Path

function Olc {
    param([string]$Etiket)
    Push-Location $kok
    try {
        $cikti = & python -m nihai.audit_chain --clock 2>&1 | Select-Object -Last 1
    } catch {
        $cikti = "OLCULEMEDI: $($_.Exception.Message)"
    } finally {
        Pop-Location
    }
    Write-Host ("  {0,-10} {1}" -f $Etiket, $cikti)
    return "$cikti"
}

Write-Host "=== SISTEM SAATI ONARIMI ===" -ForegroundColor Cyan

# 1) MEVCUT DURUM - once olc, sonra dokun.
$svc = Get-Service W32Time -ErrorAction SilentlyContinue
if ($null -eq $svc) {
    Write-Host "W32Time hizmeti bu makinede YOK - onarim yolu farklidir, durduruldu." -ForegroundColor Red
    exit 2
}
Write-Host ("Hizmet   : {0} / baslangic {1}" -f $svc.Status, $svc.StartType)
$onceki = Olc "ONCE"

if ($OlcumSadece) {
    Write-Host "Yalnizca olcum istendi; hicbir sey degistirilmedi." -ForegroundColor Yellow
    exit 0
}

# 2) YETKI - yukseltisiz gidilecek yol yok, ve bunu SESSIZCE denemek yaniltici olurdu.
$kimlik = [Security.Principal.WindowsIdentity]::GetCurrent()
$rol = New-Object Security.Principal.WindowsPrincipal($kimlik)
if (-not $rol.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host ""
    Write-Host "YONETICI YETKISI GEREKIYOR." -ForegroundColor Red
    Write-Host "Baslat menusunde 'PowerShell' -> sag tik -> 'Yonetici olarak calistir',"
    Write-Host "sonra:"
    Write-Host ("    cd '{0}'" -f $kok)
    Write-Host "    .\saat-onar.ps1"
    exit 1
}

# 3) KALICI ONARIM - hizmeti Automatic yap ve baslat.
#    Sirasi onemli: StartType once ayarlanir ki bir sonraki acilista da kosmasi
#    garanti olsun; sadece Start-Service demek bu oturumu duzeltir, sorunu degil.
Write-Host "Hizmet Automatic yapiliyor ve baslatiliyor..."
Set-Service W32Time -StartupType Automatic
if ((Get-Service W32Time).Status -ne "Running") { Start-Service W32Time }
Start-Sleep -Seconds 2
Write-Host ("Hizmet   : {0} / baslangic {1}" -f (Get-Service W32Time).Status,
                                                (Get-Service W32Time).StartType)

# 4) ZAMAN KAYNAGI - yapilandirma bossa NTP havuzu yazilir, DOLUYSA DOKUNULMAZ.
#    Calisan bir domain/kurum yapilandirmasini ezmek bu scriptin isi degildir.
$kaynak = (w32tm /query /source) 2>&1
Write-Host ("Kaynak   : {0}" -f $kaynak)
if ("$kaynak" -match "Local CMOS Clock|Free-running") {
    Write-Host "Zaman kaynagi yerel saat; NTP havuzu ayarlaniyor..." -ForegroundColor Yellow
    w32tm /config /manualpeerlist:"time.windows.com,0x9 pool.ntp.org,0x9" `
          /syncfromflags:manual /reliable:no /update | Out-Null
    Restart-Service W32Time
    Start-Sleep -Seconds 2
}

# 5) SENKRON
Write-Host "w32tm /resync /force ..."
$resync = (w32tm /resync /force) 2>&1
Write-Host ("  {0}" -f ($resync -join " "))

# 6) DOGRULAMA - IDDIA DEGIL OLCUM.
Write-Host ""
Write-Host "=== DOGRULAMA ===" -ForegroundColor Cyan
w32tm /query /status 2>&1 | Select-Object -First 8 | ForEach-Object { Write-Host "  $_" }
$sonraki = Olc "SONRA"

if ($sonraki -match "(-?\d+[.,]?\d*)\s*ms") {
    $ms = [double]($Matches[1] -replace ",", ".")
    if ([math]::Abs($ms) -le $EsikMs) {
        Write-Host ("SONUC: kayma {0:N1} ms, esik {1} ms icinde." -f $ms, $EsikMs) -ForegroundColor Green
        exit 0
    }
    Write-Host ("SONUC: kayma HALA {0:N1} ms (esik {1} ms)." -f $ms, $EsikMs) -ForegroundColor Yellow
    Write-Host "NTP paketi guvenlik duvarindan (UDP 123) cikamiyor olabilir; ONCE/SONRA"
    Write-Host "satirlarini karsilastirin - degismediyse sorun ag tarafindadir."
    exit 3
}
Write-Host "SONUC: kayma OLCULEMEDI (yukaridaki SONRA satirina bakin)." -ForegroundColor Yellow
Write-Host "Onceki olcum: $onceki"
exit 4
