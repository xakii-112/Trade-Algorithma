<#
logon-watchdog-tetik.ps1 - AtLogOn tetiginin YUKSELTI GEREKTIRMEYEN esdegeri (Tur-18).

SORUN
  Watchdog gorevlerinin tetigi yalniz "Daily 00:05". Gun ici reboot'ta o tetik zaten
  kosmus sayilir -> StartWhenAvailable telafi ETMEZ -> servisler ertesi 00:05'e kadar
  olu kalir (olculen: 2026-07-18, 10+ saat). Dogru cozum goreve AtLogOn tetigi eklemek
  ama bu makinede Set-ScheduledTask YUKSELTI istiyor ("Erisim engellendi").

BU SCRIPT
  Ayni etkiyi yukseltisiz saglar: oturum acilisinda MEVCUT gorevleri tetikler.
  Start-ScheduledTask kullanicinin kendi gorevi icin yukselti ISTEMEZ.
  Yeni watchdog dongusu YARATMAZ; gorev zaten calisiyorsa MultipleInstances=IgnoreNew
  nedeniyle ikinci ornek baslamaz -> cift-watchdog riski YOK.

KURULUM (normal PowerShell, bir kez)
    .\logon-watchdog-tetik.ps1 -Install     # Baslangic klasorune kisayol koy
    .\logon-watchdog-tetik.ps1 -Uninstall   # kisayolu kaldir
    .\logon-watchdog-tetik.ps1              # simdi bir kez tetikle (kurulum yapmaz)

KALICI COZUM
  Bir yonetici penceren oldugunda .\tur18-servis-tetik.ps1 calistir; gercek AtLogOn
  tetigi eklenince bu kisayol gereksizlesir (-Uninstall ile kaldirilabilir, zararsizdir).
#>
param([switch]$Install, [switch]$Uninstall)
$ErrorActionPreference = "Stop"

$tasks    = @("NihaiAlgo-HubWatchdog-SYSTEM", "NihaiAlgo-PredictWatchdog-SYSTEM")
$lnkPath  = Join-Path ([Environment]::GetFolderPath('Startup')) 'NihaiAlgo-LogonTetik.lnk'
$selfPath = $MyInvocation.MyCommand.Path

if ($Uninstall) {
    if (Test-Path $lnkPath) { Remove-Item $lnkPath -Force; Write-Host "KALDIRILDI: $lnkPath" }
    else { Write-Host "Zaten yok: $lnkPath" }
    return
}

if ($Install) {
    $ws = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut($lnkPath)
    $lnk.TargetPath  = "powershell.exe"
    $lnk.Arguments   = ('-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $selfPath)
    $lnk.WorkingDirectory = Split-Path $selfPath -Parent
    $lnk.Description = "NihaiAlgo: oturum acilisinda watchdog gorevlerini tetikler (AtLogOn esdegeri)"
    $lnk.Save()
    if (-not (Test-Path $lnkPath)) { throw "KISAYOL OLUSMADI: $lnkPath" }
    Write-Host "KURULDU: $lnkPath"
    Write-Host "Kaldirma: .\logon-watchdog-tetik.ps1 -Uninstall"
}

# Tetikleme (hem -Install sonrasi hem her oturum acilisinda calisan yol).
foreach ($name in $tasks) {
    $t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($null -eq $t) { Write-Host "ATLANDI (gorev yok): $name"; continue }
    if ($t.State -eq "Running") { Write-Host "ZATEN CALISIYOR: $name"; continue }
    try {
        Start-ScheduledTask -TaskName $name
        Write-Host "TETIKLENDI: $name"
    } catch {
        Write-Host "TETIKLENEMEDI: $name -> $($_.Exception.Message)"
    }
}
