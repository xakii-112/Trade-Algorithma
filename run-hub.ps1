<#
run-hub.ps1 - Process-disi veri hub'ini (hub_server.py) TEK process .venv modunda baslatir.

Hub, Alpaca'ya tek upstream baglanti tutar ve veriyi yerel HTTP'den (varsayilan
http://127.0.0.1:8799) sunar. Ana proje ve HighWR70 kopyasi DATA_PROVIDER=hub ile
buradan okur (1 baglanti, ozdes veri, ~yari API). ONCE bunu baslat, sonra projeleri.

run.ps1 ile ayni redirector-atlama mantigi (tek python process, .venv izolasyonu).

Kullanim:
    .\run-hub.ps1

Durdurma: pencerede Ctrl+C.
Guvenli geri donus (2-process redirector yolu):
    .\.venv\Scripts\python.exe -m nihai.data.hub_server
#>
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$cfg    = Join-Path $PSScriptRoot ".venv\pyvenv.cfg"
if (-not (Test-Path $cfg)) { throw "pyvenv.cfg bulunamadi - .venv kurulu mu?" }

# Base interpreter'i pyvenv.cfg'den oku (executable = ...python.exe)
$basePy = (Select-String -Path $cfg -Pattern '^\s*executable\s*=\s*(.+?)\s*$').Matches.Groups[1].Value
if (-not (Test-Path $basePy)) { throw "Base python bulunamadi: $basePy" }

# Base python'u venv-modunda, tek process olarak calistir
$env:__PYVENV_LAUNCHER__ = $venvPy
& $basePy -m nihai.data.hub_server @args
exit $LASTEXITCODE
