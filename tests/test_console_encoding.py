"""nihai.console + KOD TABANI GENELİ kodlama güvenliği (Tur-24).

BULGU: Windows'ta `python -m ...` çıktısı bir dosyaya/pipe'a yönlendirildiğinde stdout
konsol kod sayfasına (bu makinede cp1254) düşer; `→ ⚠ Δ λ 📈` gibi karakterler orada YOK
→ ilk yazdırmada UnicodeEncodeError ve süreç ÇÖKER. Watchdog/zamanlanmış görevler çıktıyı
tam da böyle yönlendirdiği için hata yolu gözetimsiz ÜRETİM yoludur.

`test_tum_main_modulleri_utf8_guvenli` bu sınıfı KİLİTLER: yeni bir CLI giriş noktası
guard'sız eklenirse test kırılır (kaynak-tarayan invariant testi, çalıştırma gerektirmez).
"""
from __future__ import annotations

import ast
import io
import re
from pathlib import Path

import pytest

from nihai.console import ensure_utf8_stdout

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("nihai", "research", "tools")   # Tur-58: tools/ da giriş noktası taşır
_MAIN_RE = re.compile(r'^if __name__ == ["\']__main__["\']:', re.MULTILINE)
# Guard'ın iki kabul edilen biçimi: paylaşılan yardımcı VEYA (eski, 59 dosyadaki) elle desen.
_GUARD_RE = re.compile(r"ensure_utf8_stdout\(\)|stdout\.reconfigure")


def _entry_points() -> list[Path]:
    """`if __name__ == "__main__":` bloğu olan tüm üretim/araştırma modülleri."""
    out: list[Path] = []
    for pkg in PACKAGES:
        for p in sorted((ROOT / pkg).rglob("*.py")):
            if _MAIN_RE.search(p.read_text(encoding="utf-8")):
                out.append(p)
    return out


def test_en_az_bir_giris_noktasi_bulundu():
    """Tarayıcı bozulursa (yol/regex) test sessizce boş küme üstünde YEŞİL kalmasın."""
    assert len(_entry_points()) > 50


@pytest.mark.parametrize("path", _entry_points(), ids=lambda p: str(p.relative_to(ROOT)))
def test_tum_main_modulleri_utf8_guvenli(path: Path):
    """Her CLI giriş noktası ya `ensure_utf8_stdout()` çağırmalı ya da eşdeğer elle
    guard'ı taşımalı — aksi halde yönlendirilmiş stdout'ta ok/emoji basan ilk satır çöker."""
    assert _GUARD_RE.search(path.read_text(encoding="utf-8")), (
        f"{path.relative_to(ROOT)} bir CLI giriş noktası ama stdout kodlama guard'ı yok; "
        "`from nihai.console import ensure_utf8_stdout` ekleyip __main__ bloğunun ilk "
        "satırında çağır (bkz. nihai/console.py)"
    )


def test_ensure_utf8_stdout_reconfigure_edilemeyen_akista_cokmez(monkeypatch):
    """Fail-soft sözleşmesi: `reconfigure`'ı olmayan akış (pytest capsys, bazı pipe
    sarmalayıcıları) istisna DEĞİL sessiz atlama üretmeli."""
    monkeypatch.setattr("sys.stdout", io.StringIO())
    monkeypatch.setattr("sys.stderr", io.StringIO())
    ensure_utf8_stdout()   # patlamamalı


def test_ensure_utf8_stdout_reconfigure_varsa_utf8_ister():
    cagrilar: list[str] = []

    class _Akis:
        def reconfigure(self, **kw):
            cagrilar.append(kw["encoding"])

    import sys as _sys
    eski_out, eski_err = _sys.stdout, _sys.stderr
    _sys.stdout, _sys.stderr = _Akis(), _Akis()   # type: ignore[assignment]
    try:
        ensure_utf8_stdout()
    finally:
        _sys.stdout, _sys.stderr = eski_out, eski_err
    assert cagrilar == ["utf-8", "utf-8"]   # stderr de kapsanır (traceback'ler de çöküyordu)


def test_ensure_utf8_stdout_reconfigure_hata_firlatirsa_yutulur():
    class _Kizgin:
        def reconfigure(self, **kw):
            raise OSError("desteklenmiyor")

    import sys as _sys
    eski = _sys.stdout
    _sys.stdout = _Kizgin()   # type: ignore[assignment]
    try:
        ensure_utf8_stdout()
    finally:
        _sys.stdout = eski


def test_selfcheck_raporu_cp1254_disi_karakter_iceriyor():
    """Regresyonun GERÇEKLİĞİNİ kilitler: guard'ı kaldırmak bu modülü yeniden çökertirdi
    (rapor metni cp1254'te olmayan karakterler üretiyor)."""
    src = (ROOT / "nihai" / "ai" / "selfcheck.py").read_text(encoding="utf-8")
    yazdirilabilir: set[str] = set()
    tree = ast.parse(src)
    dokumanlar = {ast.get_docstring(n, clean=False) or ""
                  for n in ast.walk(tree)
                  if isinstance(n, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value not in dokumanlar:
            for ch in node.value:
                try:
                    ch.encode("cp1254")
                except UnicodeEncodeError:
                    yazdirilabilir.add(ch)
    assert yazdirilabilir, "selfcheck artık cp1254-dışı karakter basmıyorsa bu test güncellenmeli"
