<#
stop-hub.ps1 — NihaiAlgoritma veri hub'ini DURDURUR ve otomatik yeniden baslatmayi
KAPATIR.

Yaptigi:
  1) Bayrak dosyasini (logs\hub.should-run) siler  -> start-hub denetimi restart YAPMAZ.
  2) Calisan hub surecini (nihai.data.hub_server) sonlandirir.

Kullanim:
    .\stop-hub.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$logDir = Join-Path $PSScriptRoot "logs"
$flag   = Join-Path $logDir "hub.should-run"

if (Test-Path $flag) {
    Remove-Item $flag -Force
    Write-Host "Bayrak silindi -> auto-restart DEVRE DISI." -ForegroundColor Yellow
} else {
    Write-Host "Bayrak zaten yok (denetim calismyor olabilir)." -ForegroundColor DarkYellow
}

# nihai.data.hub_server surecini bul ve sonlandir
$procs = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -like '*nihai.data.hub_server*'
}
if ($procs) {
    foreach ($p in $procs) {
        Write-Host ("hub sureci sonlandiriliyor (PID {0})..." -f $p.ProcessId) -ForegroundColor Yellow
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "Calisan hub sureci bulunamadi." -ForegroundColor DarkYellow
}

# Denetim Tur-11: SYSTEM watchdog gorevi kayitliysa hub ~30 sn icinde OTOMATIK geri acilir
# (watchdog bayrak dosyasini okumaz, yalniz porta bakar). Kalici durdurma gorevi de kapatmali.
$taskName = "NihaiAlgo-HubWatchdog-SYSTEM"
try {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction Stop
    Disable-ScheduledTask -TaskName $taskName -ErrorAction Stop | Out-Null
    Write-Host "SYSTEM watchdog gorevi durduruldu ve devre disi birakildi ($taskName)." -ForegroundColor Yellow
} catch {
    Write-Host ("UYARI: '{0}' gorevi kapatilamadi ({1})." -f $taskName, $_.Exception.Message) -ForegroundColor Red
    Write-Host "Gorev aktifse hub ~30 sn icinde OTOMATIK geri baslar. Kalici durdurma icin YONETICI PowerShell'de:" -ForegroundColor Red
    Write-Host ("    Disable-ScheduledTask -TaskName {0}" -f $taskName) -ForegroundColor Red
}
Write-Host "Bitti." -ForegroundColor Green
