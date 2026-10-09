<#
tur18-servis-tetik.ps1 - Tur-18'in YUKSELTI GEREKTIREN tek adimi.

NE YAPAR
  Iki watchdog gorevine AtLogOn tetigi ekler (mevcut Daily 00:05 tetigi KORUNUR).

NEDEN
  Kayit scriptlerindeki eski yorum "Daily + StartWhenAvailable = AtLogOn esdegeri kapsama"
  diyordu; bu YANLIS. StartWhenAvailable yalnizca KACIRILAN tetigi telafi eder (makine o
  saatte kapaliysa). Gun ici reboot'ta 00:05 tetigi o gun zaten kosmus sayilir -> telafi
  DEVREYE GIRMEZ. Olculen sonuc: 2026-07-18'de 03:00 reboot'undan sonra hub + kayitci
  13:46'ya kadar (10+ saat) OLU kaldi ve forward-test kaydi o sure boyunca hic birikmedi.

NEDEN YONETICI
  Bu makinede gorev GUNCELLEME de yukselti istiyor (Set-ScheduledTask -> "Erisim engellendi";
  Tur-13'te olculen "guncelleme yukselti istemez" davranisi bu tetik alani icin gecerli degil).

KULLANIM (YONETICI PowerShell)
    cd "C:\Users\Kagan\OneDrive\Desktop\NihaiAlgoritma toplu\NihaiAlgoritma"
    .\tur18-servis-tetik.ps1

Idempotent: AtLogOn zaten varsa dokunmaz. Geri alma: Set-ScheduledTask ile yalniz Daily
tetigini birakmak, veya .\register-watchdog-system.ps1 / .\register-predict-watchdog.ps1
dosyalarini eski haline dondurup yeniden kaydetmek.
#>
$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "HATA: bu script YONETICI PowerShell ister (Set-ScheduledTask erisim engelli)." -ForegroundColor Red
    Write-Host "Baslat -> 'PowerShell' -> sag tik -> 'Yonetici olarak calistir' -> tekrar dene."
    exit 1
}

$me    = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$tasks = @("NihaiAlgo-HubWatchdog-SYSTEM", "NihaiAlgo-PredictWatchdog-SYSTEM")
$degisen = 0

foreach ($name in $tasks) {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($null -eq $task) { Write-Host "ATLANDI (gorev yok): $name" -ForegroundColor Yellow; continue }
    $hasLogon = $task.Triggers | Where-Object { $_.CimClass.CimClassName -eq "MSFT_TaskLogonTrigger" }
    if ($hasLogon) { Write-Host "ZATEN VAR : $name (AtLogOn)"; continue }
    $triggers = @($task.Triggers) + @(New-ScheduledTaskTrigger -AtLogOn -User $me)
    Set-ScheduledTask -TaskName $name -Trigger $triggers | Out-Null
    Write-Host "EKLENDI   : $name (AtLogOn)" -ForegroundColor Green
    $degisen++
}

Write-Host "`n=== DOGRULAMA ==="
foreach ($name in $tasks) {
    $t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($null -eq $t) { continue }
    $trg = ($t.Triggers | ForEach-Object { $_.CimClass.CimClassName }) -join " + "
    "{0,-34} {1}" -f $name, $trg
}
Write-Host "`nDegisen gorev: $degisen. Artik oturum acilisinda (reboot sonrasi) watchdog'lar kendiliginden kalkar."
