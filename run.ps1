<#
run.ps1 — NihaiAlgoritma'yı TEK process olarak .venv modunda başlatır.

Sorun: Bu Python 3.12 kurulumunda `.venv\Scripts\python.exe` bir "redirector"
stub'ı; çalıştırınca base interpreter'ı child process olarak başlatıp kendisi
boşta bekler → her zaman 2 process (boşta .venv stub + asıl işi yapan base-python
child). Davranış zararsız (child yine .venv paketlerini kullanır) ama gereksiz.

Çözüm: redirector'ın içeride yaptığını manuel yaparız — base interpreter'ı
`__PYVENV_LAUNCHER__` ortam değişkeniyle DOĞRUDAN venv-modunda çalıştırırız.
Sonuç: TEK python process, yine .venv izolasyonu (sys.prefix = .venv).

Kullanım:
    .\run.ps1
    .\run.ps1 --once --screen-now

NOT: `__PYVENV_LAUNCHER__` CPython'un dahili (belgelenmemiş) değişkenidir; bir üst
sürümde değişirse bu script kırılabilir. Güvenli geri dönüş her zaman mümkün:
    .\.venv\Scripts\python.exe -m nihai.execution.main --once   (2-process redirector yolu)
#>
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$cfg    = Join-Path $PSScriptRoot ".venv\pyvenv.cfg"
if (-not (Test-Path $cfg)) { throw ".venv\pyvenv.cfg bulunamadi - .venv kurulu mu?" }

# Base interpreter'ı pyvenv.cfg'den oku (executable = ...python.exe)
$basePy = (Select-String -Path $cfg -Pattern '^\s*executable\s*=\s*(.+?)\s*$').Matches.Groups[1].Value
if (-not (Test-Path $basePy)) { throw "Base python bulunamadi: $basePy" }

# Base python'u venv-modunda, tek process olarak çalıştır
$env:__PYVENV_LAUNCHER__ = $venvPy
& $basePy -m nihai.execution.main @args
exit $LASTEXITCODE
