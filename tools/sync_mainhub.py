"""sync_mainhub.py — MainHub'ın paylaşılan sağlayıcı dosyalarını `nihai/data/`'dan ÜRET.

SORUN (Tur-58 girdisi, dış rapor §2.B "Code Drift"): `MainHub/` üç sağlayıcı dosyasını
`nihai/data/` ile **kopyala-yapıştır** paylaşıyordu. Tur-57 `nihai` kopyasını güncelledi
(`trade_count`/`vwap` korunması, OPRA zenginleştirme, `asset_flags`, `ALPACA_BARS_FEED`);
MainHub kopyası 2026-07-26'da kaldı. Sonuç: `DATA_PROVIDER=hub` koşan süreç, hub'ın
düşürdüğü kolonları hiç göremez — yani bir düzeltme yapıldı ama TÜKETİCİYE ULAŞMADI.

Rapor bunun için "ortak `nihai_core` paketi" öneriyor. Bu depoda o **fazla büyük** bir
çözüm: MainHub ayrı bir venv'de, düz (paketsiz) modül düzeniyle çalışır; paketleme göçü
iki uygulamayı da kırar. Sürüklenmeyi bitiren asgari şey paket DEĞİL, **tek kaynak +
otomatik türetme + sürüklenme kapısı**:

    nihai/data/X.py   ── (import yeniden yazımı + LF) ──▶   MainHub/X.py

`nihai` KANONİKTİR. MainHub kopyası bir ARTEFAKTTIR; elle düzenlenmez.

Kullanım:
    python -m tools.sync_mainhub             # yalnız kontrol: sürüklenme var mı? (çıkış 1 = var)
    python -m tools.sync_mainhub --apply     # MainHub kopyalarını yeniden üret

Kapı: `tests/test_tur58_hub_koprusu.py` kontrol kipini (`generate(apply=False)`) koşar. MainHub'da elle yapılan
her düzenleme kırmızı test verir → düzeltme `nihai/` tarafında yapılıp yeniden üretilir.

KAPSAM DIŞI (bilinçli): `config.py` ve `hub_server.py`. MainHub'ın `config.py`'si gerçek
bir ALT KÜMEDİR (trading/risk ayarları yoktur) ve `hub_server.py` MainHub'da dosya-köklü
log dizini kullanır — ikisi de mekanik türetilemez, ikisi de bilinçli olarak yereldir.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../NihaiAlgoritma
MAINHUB = ROOT.parent / "MainHub"

# nihai/data/<ad>.py  →  MainHub/<ad>.py
SHARED = ("data_provider", "alpaca_provider", "alpaca_stream_provider")

_HEADER = (
    "# ---------------------------------------------------------------------------\n"
    "# ÜRETİLMİŞ DOSYA — ELLE DÜZENLEME.  Kaynak: NihaiAlgoritma/nihai/data/{name}.py\n"
    "# Yeniden üret:  python -m tools.sync_mainhub --apply   (NihaiAlgoritma kökünden)\n"
    "# Tek fark import yollarıdır; davranış kaynakla BİREBİR aynıdır (Tur-58).\n"
    "# ---------------------------------------------------------------------------\n"
)

# `from nihai.data.x import` → `from x import` · `from nihai.config import` → `from config import`
_RE_FROM = re.compile(r"^(\s*)from nihai\.(?:data|ai)\.([a-z_0-9]+) import ", re.M)
_RE_CFG = re.compile(r"^(\s*)from nihai\.config import ", re.M)
_RE_IMPORT = re.compile(r"^(\s*)import nihai\.(?:data|ai)\.([a-z_0-9]+)\b", re.M)


def rewrite(src: str, name: str) -> str:
    """Kaynağı MainHub'ın düz modül düzenine çevir (SAF — dosya sistemi gerekmez)."""
    out = _RE_FROM.sub(r"\1from \2 import ", src)
    out = _RE_CFG.sub(r"\1from config import ", out)
    out = _RE_IMPORT.sub(r"\1import \2", out)
    out = out.replace("\r\n", "\n")                  # MainHub ağacı LF
    return _HEADER.format(name=name) + out


def _isort(paths: list[Path]) -> None:
    """Yeniden yazım import SIRASINI bozar (`nihai.data.x` ≠ `x`) → ruff ile düzelt.

    ruff yoksa sessizce atlanır: sıralama kozmetiktir, üretimi bloke etmemeli."""
    import subprocess
    try:
        subprocess.run([sys.executable, "-m", "ruff", "check", "--select", "I", "--fix", "-q",
                        *[str(p) for p in paths]], cwd=str(paths[0].parent) if paths else None, check=False,
                       capture_output=True, timeout=120)
    except Exception:                                # noqa: BLE001
        pass


def _materialize() -> dict[str, str]:
    """Beklenen MainHub içeriğini üret — GEÇİCİ dizinde, isort UYGULANMIŞ hâliyle.

    isort adımı zorunludur: `nihai.data.alpaca_provider` → `alpaca_provider` yeniden
    yazımı import SIRASINI değiştirir. Sıralamayı yazdıktan SONRA uygulamak `--check`'i
    idempotent olmaktan çıkarırdı (yazılan dosya kontrol edilen metinden farklı olurdu),
    bu yüzden beklenen metin de aynı normalizasyondan geçirilir."""
    import tempfile
    out: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="mainhub_sync_") as tmp:
        tmpdir = Path(tmp)
        paths = []
        for name in SHARED:
            src = (ROOT / "nihai" / "data" / f"{name}.py").read_text(encoding="utf-8")
            p = tmpdir / f"{name}.py"
            p.write_text(rewrite(src, name), encoding="utf-8", newline="\n")
            paths.append(p)
        _isort(paths)
        for name, p in zip(SHARED, paths, strict=True):
            out[name] = p.read_text(encoding="utf-8").replace("\r\n", "\n")
    return out


def generate(apply: bool) -> list[str]:
    """Sürüklenmiş dosya adlarını döndür. `apply` ise MainHub kopyalarını yazar."""
    drifted: list[str] = []
    for name, want in _materialize().items():
        dst = MAINHUB / f"{name}.py"
        have = dst.read_text(encoding="utf-8").replace("\r\n", "\n") if dst.exists() else ""
        if have == want:
            continue
        drifted.append(name)
        if apply:
            dst.write_text(want, encoding="utf-8", newline="\n")
    return drifted


def main() -> int:
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()          # Tur-24: çıktı dosyaya/pipe'a yönlenince cp1254'e düşer
    ap = argparse.ArgumentParser(description="MainHub paylaşılan dosyalarını nihai/'dan üret")
    ap.add_argument("--apply", action="store_true", help="yaz (varsayılan: yalnız kontrol)")
    a = ap.parse_args()
    if not MAINHUB.is_dir():
        print(f"[sync] MainHub bulunamadı: {MAINHUB} — atlanıyor")
        return 0
    drifted = generate(apply=a.apply)
    if not drifted:
        print("[sync] MainHub güncel — sürüklenme yok")
        return 0
    if a.apply:
        print(f"[sync] yeniden üretildi: {', '.join(drifted)}")
        return 0
    print(f"[sync] SÜRÜKLENME: {', '.join(drifted)}\n"
          f"       düzelt: python -m tools.sync_mainhub --apply")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
