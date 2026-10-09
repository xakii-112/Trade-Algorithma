<#
run-duo.ps1 — İki strateji varyasyonunu (champion/challenger) TEK process'te,
PAYLAŞIMLI veri havuzuyla A/B çalıştırır (duo.py).

run.ps1 ile aynı tek-process .venv mekanizması (venv redirector stub'ını atlar):
base interpreter'ı __PYVENV_LAUNCHER__ ile doğrudan venv-modunda çalıştırır.

Kullanım:
    .\run-duo.ps1            # canlı A/B
    .\run-duo.ps1 --once     # tek tur (test)

Çıktı: logs\system.log içinde [duo:compare] satırları (iki kasayı yan yana).
Kasalar: data\duo\champion\ ve data\duo\challenger\ (ayrı state.json + equity_curve.csv).
#>
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$cfg    = Join-Path $PSScriptRoot ".venv\pyvenv.cfg"
if (-not (Test-Path $cfg)) { throw ".venv\pyvenv.cfg bulunamadı — .venv kurulu mu?" }

$basePy = (Select-String -Path $cfg -Pattern '^\s*executable\s*=\s*(.+?)\s*$').Matches.Groups[1].Value
if (-not (Test-Path $basePy)) { throw "Base python bulunamadı: $basePy" }

$env:__PYVENV_LAUNCHER__ = $venvPy
& $basePy -m research.duo @args
exit $LASTEXITCODE
