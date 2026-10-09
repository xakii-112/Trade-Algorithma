<#
forward-harness.ps1 - Forward veri biriktirme harness'ini TEK SEFER calistirir
(resolve + snapshot + report) ve ciktiyi logs/forward-harness.log'a ekler.

Gorev Zamanlayici bunu gunde bir tetikler (register-forward-harness.ps1) -> her gun
vadesi gelen tahminler etiketlenir ve data/forward_eval/history.csv'ye bir snapshot
eklenir; boylece kenarin N&gun buyudukce evrimi zaman serisi olarak birikir.

Elle de calistirilabilir:
    .\forward-harness.ps1
    .\forward-harness.ps1 -MaxLogMB 20
    .\forward-harness.ps1 -SeansIcinde     # seans acikken de kos (varsayilan: kapanisi bekle ya da atla)
#>
param(
    [int]$MaxLogMB = 20,
    [switch]$SeansIcinde,
    [int]$BeklemeDk = 65,        # kapanisa bundan az kaldiysa kapanis beklenir; daha cok varsa kosu atlanir
    [int]$KapanisPayiSn = 300    # kapanistan sonra beklenen pay (son barlar + servis yazimi otursun)
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# Python ciktisini UTF-8 olarak coz - loga yazilan Turkce karakterler bozulmasin (O6).
# Konsolsuz ortamda (redirect) handle hatasi firlatabilir; kozmetik oldugu icin yut.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
# Cocuk python'un stdout+stderr'i de UTF-8 yazsin (loguru stderr'i ANSI yaziyordu).
$env:PYTHONIOENCODING = "utf-8"

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "venv python bulunamadi: $venvPy (.venv kurulu mu?)" }

$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$log = Join-Path $logDir "forward-harness.log"

# Tur-88: gunluk yazimi ISI DURDURAMAZ. 2026-09-22 23:30 kosusu, dosya baska bir surecte acikken
# (Git-Bash `tail -F` -> PS 5.1 Add-Content "dosya kullanimda") ErrorActionPreference=Stop altinda
# BASLAMADAN oldu (gorev sonucu 1, gunlukte tek satir yok) -> cozum + ogrenme + aday egitimi kosmadi.
# 10 x 0,5 sn dene; olmazsa yan dosyaya yaz. Asla firlatmaz.
function Write-Log([string]$Text) {
    for ($k = 0; $k -lt 10; $k++) {
        try { Add-Content -LiteralPath $log -Value $Text -Encoding utf8 -ErrorAction Stop; return }
        catch { Start-Sleep -Milliseconds 500 }
    }
    Add-Content -Path "$log.kilitli" -Value $Text -Encoding utf8 -ErrorAction SilentlyContinue
}

# Boyut sinirini asarsa logu sifirla (sonsuz buyumeyi onle).
if ((Test-Path $log) -and ((Get-Item $log).Length -gt ($MaxLogMB * 1MB))) {
    Clear-Content -Path $log -ErrorAction SilentlyContinue
}

# Tur-106 (BULGU-106-36): seans ACIKKEN kosma. Gorev tetikleri yerel duvar saatidir; ABD yaz saati bitince
# (2 Kasim -> Mart ortasi, TR sabit) seans TR 17:30-00:00 olur ve 23:30 tetigi 15:30 ET'ye duser: resolve()
# defteri bastan yazarken canli servis AYRI surecten satir ekler (olculdu: yazma penceresi 1-2 sn, gecede
# ~%2-3 olasilikla kayip ek), aday egitimi seansin son yarim saatine biner, kalibrasyon dosyalari seans
# ortasinda tazelenir. Uykudan uyanista kacan ogle tetigi de (StartWhenAvailable) seans icinde kosabiliyordu.
# Kapanisa $BeklemeDk'dan az kaldiysa kapanis + pay beklenir (kisin gece kosusu ~00:05); daha cok varsa kosu
# ATLANIR (sonraki tetik gunu tamamlar). Banner'dan ONCE: atlanan kosu "calisiyor" yazmaz. Takvim
# okunamazsa eski davranis: kos.
function Get-SeansKalanSn {
    $onceki = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $cikti = & $venvPy -m nihai.data.market_calendar --kalan-sn 2>$null | Select-Object -Last 1
    $kod = $LASTEXITCODE
    $ErrorActionPreference = $onceki
    $sn = -1
    if ($kod -eq 0 -and [int]::TryParse(("$cikti").Trim(), [ref]$sn)) { return $sn }
    return -1
}
if (-not $SeansIcinde) {
    $kalanSn = Get-SeansKalanSn
    if ($kalanSn -gt 0 -and $kalanSn -le ($BeklemeDk * 60)) {
        $an = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $bekle = $kalanSn + $KapanisPayiSn
        Write-Log "`n[$an] seans acik, kapanisa $kalanSn sn: $bekle sn bekleniyor (kapanis + $KapanisPayiSn sn pay)"
        Start-Sleep -Seconds $bekle
        $kalanSn = Get-SeansKalanSn
    }
    if ($kalanSn -gt 0) {
        $an = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Log "`n[$an] seans ACIK (kapanisa $kalanSn sn) - kosu ATLANDI; sonraki tetik gunu tamamlar (elle: -SeansIcinde)"
        exit 0
    }
}

# Tur-13: onceki kosu "calisiyor" banner'i yazip "BITTI" yazamadan sonlandiysa LOUD isaretle
# (or. gorev zaman-limiti kill, guc kaybi). Gozlem: 2026-07-16 19:53 kosusu boyle sessiz oldu.
if (Test-Path $log) {
    $tail = @(Get-Content $log -Tail 400 -ErrorAction SilentlyContinue)
    $sonBaslama = ($tail | Select-String -Pattern 'forward-harness calisiyor' | Select-Object -Last 1).LineNumber
    $sonBitis   = ($tail | Select-String -Pattern 'forward-harness BITTI' | Select-Object -Last 1).LineNumber
    # NOT: BITTI isaretleyicisi Tur-13'te eklendi; log'da HIC BITTI yokken uyari verme
    # (eski basarili kosular da BITTI icermez -> ilk kosuda yanlis-pozitif olurdu).
    if ($sonBaslama -and $sonBitis -and ($sonBitis -lt $sonBaslama)) {
        Write-Log "[UYARI] onceki kosu BITTI yazamadan sonlanmis (muhtemel zaman-limiti kill / guc kaybi) - o kosunun history satiri eksik olabilir."
    }
}

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Log "`n===== $stamp  forward-harness calisiyor ====="

# -m ile calistir: cwd ($PSScriptRoot) sys.path[0] olur -> nihai/research import edilir.
# stderr de yakalanip loga eklenir (native exe ciktisini string olarak topla).
#
# ONEMLI (PS 5.1 tuzagi): ErrorActionPreference=Stop altinda native komutun 2>&1
# yonlendirmesi, python stderr'e TEK satir bile yazsa (or. loguru bilgi mesaji)
# NativeCommandError firlatir -> script cikti kaydetmeden olur, gorev sonucu 1
# (gozlem: 2026-07-02 02:29 kosusu boyle sessiz oldu). Cagri suresince Continue'ya
# dusur, cikis kodunu kendimiz kontrol edelim.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$out = & $venvPy -m research.forward_harness --resolve --snapshot --report 2>&1 |
    ForEach-Object { $_.ToString() } | Out-String
$code = $LASTEXITCODE
$ErrorActionPreference = $prevEAP
Write-Log $out
Write-Output $out
if ($code -ne 0) {
    Write-Log "[HATA] forward_harness cikis kodu: $code"
}

# Tur-83: tahmin kalibrasyon egrisini (data/forecast_calibration.json) CANLI defterden
# yeniden kur. Egri 23 Haziran'dan beri hic tazelenmemisti ve sabit taban orandan KOTU
# cikmisti (BULGU-83-3). Yuruyen dogrulama + taban-oran kapisi; kapi gecilmezse egri DUZ
# (taban oran) yazilir. Fail-soft: hatasi harness'in NIHAI cikis kodunu ETKILEMEZ.
# NOT: `daily_summary --calib` BU dosyayi YAZMAZ (o AI kalibratorlerini tazeler).
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$calOut = & $venvPy -m research.calibrate_forecast 2>&1 |
    ForEach-Object { $_.ToString() } | Out-String
$calCode = $LASTEXITCODE
$ErrorActionPreference = $prevEAP
Write-Log $calOut
if ($calCode -ne 0) {
    Write-Log "[UYARI] calibrate_forecast cikis kodu: $calCode (harness sonucu etkilenmedi)"
}

# FAZ AI-4 (Tur-11): gunluk sistem-sagligi ozeti - karar VERMEZ, ozetler (verdikt+kapsama+
# servis durumu). BILINCLI: harness $code!=0 olsa BILE calisir - tam da boyle bir gunde
# (or. 2026-07-11 OneDrive kilit cokmesi) operator ozeti/Telegram'i EN COK o gun gormeli.
# Fail-soft: ozet hatasi script'in NIHAI cikis kodunu ETKILEMEZ; Telegram yapilandirilmamissa
# ozet yalnizca log/dosyaya yazilir.
# Tur-83: --calib. daily_summary docstring'i "gece gorevi --calib ile kosar" diyordu ama bu
# satir onu hic vermiyordu -> ai_calibrators/ai_adaptive_calibration 12 Agustos'tan beri
# BAYATTI. Olculdu (yalitilmis veri dizini): 7 sn; KARAR-DISI artefaktlar.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$sumOut = & $venvPy -m nihai.notifier.daily_summary --calib --send 2>&1 |
    ForEach-Object { $_.ToString() } | Out-String
$sumCode = $LASTEXITCODE
$ErrorActionPreference = $prevEAP
Write-Log $sumOut
if ($sumCode -ne 0) {
    Write-Log "[UYARI] daily_summary cikis kodu: $sumCode (harness sonucu etkilenmedi)"
}

# Tur-13: tamamlanma isareti - yarim-kosu tespiti bu banner'a bakar.
$stampBitis = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Log "===== $stampBitis  forward-harness BITTI (cikis=$code, ozet=$sumCode) ====="

# Script'in NIHAI cikis kodu forward_harness'inki - Gorev Zamanlayici/izleme dogru
# basarisizligi gorsun (ozet cagrisi bunun icin geciktirilmedi/atlanmadi, yalniz SONRAYA alindi).
if ($code -ne 0) { exit $code }
