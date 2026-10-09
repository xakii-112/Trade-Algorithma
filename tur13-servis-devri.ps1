<#
tur13-servis-devri.ps1 - ESKI SYSTEM servis yiginini YENI interactive-user modeline
TEK SEFERDE devreder. YONETICI PowerShell'de calistirilmali (tek gereken elevated adim).

NEDEN (Tur-13 olcumleri, 2026-07-17):
  - Hub (8799) + kayitci (8765) + predict-watchdog SYSTEM baglaminda ZOMBI olarak calisiyor
    (2026-07-10'da baslamis; Tur-11/12 kod duzeltmelerini ve .env META_GATE=shadow'u GORMUYOR).
  - SYSTEM baglami yukseltisiz oturumlara GORUNMEZ: gorev listesi + surec komut satirlari
    bos gorunur -> "servisler kapali / gorevler kayitsiz" YANLIS teshisine yol acti (Tur-12).
    Interactive modelde her sey normal oturumdan gozlemlenebilir/yonetilebilir olur.
  - Bu makinede YENI gorev olusturma yukseltisiz reddediliyor (0x80070005) -> ilk kayit
    icin bu script yonetici ister; SONRAKI guncellemeler yonetici istemez.

YAPTIGI (sirayla):
  1) Eski SYSTEM gorevlerini kaldirir (ayni adlar yeniden interactive olarak kaydedilecek).
  2) SYSTEM watchdog powershell'lerini ve SYSTEM hub/kayitci python'larini sonlandirir.
  3) register-watchdog-system.ps1 + register-predict-watchdog.ps1 calistirir
     (interactive-user gorevleri kaydeder + HEMEN baslatir; watchdog'lar servisleri
     GUNCEL kodla + guncel .env ile yeniden ayaga kaldirir).
  4) Portlari dogrular ve ozet basar.

Kullanim (YONETICI PowerShell, repo kokunde):
    powershell -ExecutionPolicy Bypass -File .\tur13-servis-devri.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "HATA: Bu script YONETICI PowerShell ister (gorev olusturma + SYSTEM surec sonlandirma)." -ForegroundColor Red
    Write-Host "Baslat menusu -> 'PowerShell' -> sag tik -> Yonetici olarak calistir -> bu scripti yeniden calistir." -ForegroundColor Yellow
    exit 1
}

function Test-PortUp([int]$p) {
    $c = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $c.BeginConnect("127.0.0.1", $p, $null, $null)
        if ($iar.AsyncWaitHandle.WaitOne(1000, $false) -and $c.Connected) { return $true }
        return $false
    } catch { return $false } finally { $c.Close() }
}

Write-Host "=== 1/4 Eski SYSTEM gorevleri kaldiriliyor ===" -ForegroundColor Cyan
foreach ($tn in @("NihaiAlgo-HubWatchdog-SYSTEM", "NihaiAlgo-PredictWatchdog-SYSTEM")) {
    $t = Get-ScheduledTask -TaskName $tn -ErrorAction SilentlyContinue
    if ($t) {
        Stop-ScheduledTask -TaskName $tn -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $tn -Confirm:$false
        Write-Host "  Kaldirildi: $tn (principal: $($t.Principal.UserId))"
    } else {
        Write-Host "  Zaten yok: $tn"
    }
}

Write-Host "=== 2/4 SYSTEM zombi surecleri sonlandiriliyor ===" -ForegroundColor Cyan
# Watchdog powershell'leri ONCE oldur (yoksa python'u aninda geri baslatirlar).
$wd = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object {
    $_.CommandLine -like '*hub-watchdog.ps1*' -or $_.CommandLine -like '*predict-watchdog.ps1*'
}
foreach ($p in $wd) { Write-Host "  watchdog ps kill: PID $($p.ProcessId)"; Stop-Process -Id $p.ProcessId -Force }
$py = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -like '*nihai.data.hub_server*' -or
    ($_.CommandLine -like '*nihai.execution.main*' -and $_.CommandLine -like '*--serve-web*')
}
foreach ($p in $py) { Write-Host "  servis python kill: PID $($p.ProcessId)"; Stop-Process -Id $p.ProcessId -Force }
Start-Sleep -Seconds 3

Write-Host "=== 3/4 Interactive-user gorevleri kaydediliyor + baslatiliyor ===" -ForegroundColor Cyan
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "register-watchdog-system.ps1")
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "register-predict-watchdog.ps1")

Write-Host "=== 4/4 Dogrulama (servislerin acilmasi ~60-90 sn surebilir) ===" -ForegroundColor Cyan
$ok8799 = $false; $ok8765 = $false
for ($i = 0; $i -lt 24; $i++) {
    Start-Sleep -Seconds 5
    $ok8799 = Test-PortUp 8799
    $ok8765 = Test-PortUp 8765
    Write-Host ("  {0,3} sn: hub(8799)={1}  kayitci(8765)={2}" -f (($i + 1) * 5), $ok8799, $ok8765)
    if ($ok8799 -and $ok8765) { break }
}
Get-ScheduledTask | Where-Object TaskName -like "NihaiAlgo*" |
    Select-Object TaskName, State | Format-Table -AutoSize
if ($ok8799 -and $ok8765) {
    Write-Host "DEVIR TAMAM: servisler guncel kodla + guncel .env (META_GATE=shadow) ile interactive-user'da." -ForegroundColor Green
} else {
    Write-Host "UYARI: portlarin biri hala kapali - logs\hub_watchdog.log ve logs\predict_watchdog.log kuyruguna bak." -ForegroundColor Yellow
}
