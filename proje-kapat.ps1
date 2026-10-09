<#
proje-kapat.ps1 - Nihai Algoritma'nin TAMAMINI durdurur ve KAPALI KALMASINI saglar.

Neden ayri bir betik: stop-hub.ps1 yalniz hub'i durdurur ve kapatmayi KALICI kilamaz.
Watchdog gorevleri (NihaiAlgo-HubWatchdog-SYSTEM / -PredictWatchdog-SYSTEM) her 10 dk
tetikleniyor ve bu makinede yukseltisiz devre disi birakilamiyor (0x80070005 Erisim
engellendi). Yani "sadece surecleri oldur" yaklasimi 10 dk sonra bozulur - 2026-07-26'da
tam olarak bu yasandi.

Kapsam (Tur-106): servisler (hub + kayitci/pano + varsa ayri golge kasa kosucusu) ve iki watchdog. Zamanlanmis GOREVLER bayragi
TANIMAZ: NihaiAlgo-ForwardHarness-Daily (12:00/23:30) ve NihaiAlgo-ShadowRunner-Hourly kapaliyken de
kosar (golge kosucu hub yokken yfinance'e duser); deadman PROJE_KAPALI raporlar.

Cozum: logs\project.stopped bayragi. Watchdog'lar her dongude bu bayragi okur ve varsa
hicbir seyi diriltmez. Yonetici yetkisi GEREKMEZ, hicbir dosya bozulmaz.

Sirala onemli: (1) bayrak, (2) watchdog surecleri, (3) servisler.
Bayrak once konur ki bir sonraki tetiklemede acilan TAZE watchdog dogrudan stand-by'a gecsin.

Kullanim:
    .\proje-kapat.ps1
    .\proje-kapat.ps1 -NoPause    # kisayol/otomasyon disi kullanim
#>
param([switch]$NoPause)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$stopFlag = Join-Path $logDir "project.stopped"

Write-Host ""
Write-Host "=== NIHAI ALGORITMA - KAPATILIYOR ===" -ForegroundColor Cyan
Write-Host ""

# (1) Bayrak
Set-Content -Path $stopFlag -Value ("KAPATILDI: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -Encoding utf8
Write-Host "[1/4] STOP bayragi kondu -> watchdog artik diriltmeyecek." -ForegroundColor Yellow

# (2) Watchdog surecleri. Calisan ornekler betigi BELLEKTE tutuyor; bayragi gormeleri icin
#     olmeleri gerekir. Gorev 10 dk icinde taze ornek acar, o da bayragi okuyup bekler.
#     DIKKAT: "-like '*hub-watchdog.ps1*'" YETMEZ - adi yalnizca METINDE gecen alakasiz bir
#     PowerShell oturumu (or. bu betikleri inceleyen bir konsol) da eslesir ve zorla oldurulur.
#     Yalniz GERCEKTEN "-File ...watchdog.ps1" ile acilmis surecleri hedefle.
$wd = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -match '-File\s+"?[^"]*(hub|predict)-watchdog\.ps1' }
if ($wd) {
    foreach ($p in $wd) {
        Write-Host ("      watchdog sonlandiriliyor (PID {0})" -f $p.ProcessId) -ForegroundColor DarkYellow
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "      calisan watchdog yok." -ForegroundColor DarkYellow
}
Write-Host "[2/4] Watchdog surecleri durduruldu." -ForegroundColor Yellow

# (3) Servisler: hub (8799) + kayitci/web (8765) + varsa tarayici-only main + ayri golge kasa kosucusu.
#     Tur-106 (BULGU-106-29): `golge_kasalar --kos` kendi tahmin servisini acan AYRI bir surectir (Tur-102);
#     eslesmedigi icin proje "kapali"yken kosmaya, kasa defterine yazmaya ve motor.lock'u tutmaya devam ediyordu
#     (sonraki proje-ac'ta ana surecin kasa motoru kilidi alamaz). Yalniz --kos'lu kosucu hedeflenir.
$svc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*nihai.data.hub_server*' -or $_.CommandLine -like '*nihai.execution.main*' -or
                   $_.CommandLine -like '*nihai.execution.golge_kasalar*--kos*' }
if ($svc) {
    foreach ($p in $svc) {
        Write-Host ("      python sonlandiriliyor (PID {0})" -f $p.ProcessId) -ForegroundColor DarkYellow
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "      calisan servis yok." -ForegroundColor DarkYellow
}
Write-Host "[3/4] Servisler durduruldu." -ForegroundColor Yellow

# (4) YETIM multiprocessing isci surecleri.
#     Windows'ta havuz ebeveyni oldurulunce 'spawn' ile acilmis cocuklar HAYATTA KALIR:
#     olu bir boruyu bekleyerek bellegi tutmaya devam ederler. Olcum (2026-07-31):
#     kapatmadan 45 dk sonra 16 yetim isci 1109 MB RAM tutuyordu. Komut satirlarinda
#     'nihai...' GECMEDIGI icin adim (3) bunlari hic gormez.
#     Guvenlik: yalniz multiprocessing.spawn iscileri VE yalniz ebeveyni OLMEYENLER
#     hedeflenir - baska projelerin python'u (or. graphify-mcp) ya da canli bir havuz
#     ETKILENMEZ.
$workers = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*multiprocessing.spawn*' }
$orphans = @()
foreach ($w in $workers) {
    $pp = [int]$w.ParentProcessId
    $parent = $null
    try { $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$pp" -ErrorAction Stop } catch { $parent = $null }
    if (-not $parent) { $orphans += $w }
}
if ($orphans) {
    $mb = [math]::Round((($orphans | Measure-Object -Property WorkingSetSize -Sum).Sum / 1MB), 0)
    Write-Host ("      {0} yetim isci sureci ({1} MB) temizleniyor" -f $orphans.Count, $mb) -ForegroundColor DarkYellow
    foreach ($o in $orphans) { Stop-Process -Id $o.ProcessId -Force -ErrorAction SilentlyContinue }
} else {
    Write-Host "      yetim isci sureci yok." -ForegroundColor DarkYellow
}
Write-Host "[4/4] Yetim isciler temizlendi." -ForegroundColor Yellow

# DOGRULA - sessiz basarisizlik yok: port hala aciksa LOUD uyari.
Start-Sleep -Seconds 3
$open = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -in 8799, 8765 }
Write-Host ""
if ($open) {
    Write-Host "UYARI: bu portlar HALA ACIK:" -ForegroundColor Red
    $open | Format-Table LocalPort, OwningProcess -AutoSize | Out-String | Write-Host
    Write-Host "Tekrar calistir; duzelmezse Gorev Yoneticisi'nden python.exe'leri kapat." -ForegroundColor Red
} else {
    Write-Host "PROJE KAPALI. 8799 ve 8765 kapali, bayrak duruyor." -ForegroundColor Green
    Write-Host "Tekrar acmak icin: proje-ac.ps1 (veya masaustundeki 'Projeyi Ac' kisayolu)" -ForegroundColor Gray
}
Write-Host ""

if (-not $NoPause) {
    Write-Host "Kapatmak icin bir tusa basin..." -ForegroundColor DarkGray
    try { $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown') } catch { Start-Sleep -Seconds 5 }
}
