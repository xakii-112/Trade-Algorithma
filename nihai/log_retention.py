"""log_retention.py — BULGU B-7: log saklama politikası (günlük rotasyon vardı, SİLME yoktu).

Ölçülen (2026-07-25): `logs/` 368 MB, `data/` 2.089 MB — hepsi OneDrive senkronize
klasöründe. Tek günlük `decision_log_2026_07_25.txt` 35 MB (tarayıcı 5 sn'de bir 30
sembol yazıyor). Rotasyon zaten günlük ama hiçbir şey ESKİYİ SİLMİYORDU, dolayısıyla
klasör tek yönlü büyüyordu. Bu yalnız disk meselesi değil: bu depo geçmişte OneDrive
kilidi yüzünden gerçek bir çöküş yaşadı (bkz. `nihai/atomic_io.py`); senkronize edilen
hacim büyüdükçe o riskin yüzeyi de büyür.

TASARIM — bu modül SİLER, o yüzden bilinçli olarak temkinli:
  * Yalnız `logs/` altındaki DÖNGÜSEL/tarihli dosyalara dokunur (aşağıdaki desenler).
    `data/` (tahmin logu, özellik logu, dataset'ler = ARAŞTIRMA VERİSİ) ASLA hedef değil.
  * Varsayılan KURU KOŞU. Gerçekten silmek için `--apply` şarttır.
  * Bugünün ve dünün dosyalarına yaş ne olursa olsun dokunmaz (canlı süreç yazıyor olabilir).
  * Dosya ADINDAKİ tarihe bakar; ad tarihsizse mtime'a düşer. Tarihi çözülemeyen dosya
    ATLANIR (silme kararı asla tahmine dayanmaz).

Kullanım:
    python -m nihai.log_retention                 # ne silinecekti? (kuru koşu)
    python -m nihai.log_retention --apply         # gerçekten sil
    python -m nihai.log_retention --days 60 --apply
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from nihai.config import LOG_DIR
from nihai.console import ensure_utf8_stdout

# Silinebilir sayılan desenler — HEPSİ yeniden üretilebilir/tarihli operasyon logudur.
# Yeni desen eklerken soru şudur: "bu dosya kaybolursa geri döndürülemez bir ARAŞTIRMA
# kaydı kaybeder miyim?" Cevap evetse buraya EKLEME.
PATTERNS: tuple[str, ...] = (
    r"^decision_log_\d{4}_\d{2}_\d{2}\.txt$",
    r"^auto_screen_\d{8}_\d{6}\.csv$",
    r"^.*\.log\.\d+$",            # boyutla döndürülmüş eskiler (system.log.1 vb.)
)
DEFAULT_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "30"))
# Tur-58 — SİLME ile SAKLAMA arasındaki orta bant: bu yaştan eski ama saklama süresini
# aşmamış dosyalar SIKIŞTIRILIR. Dış rapor §H'nin eksik yarısı: retention vardı, sıkıştırma
# yoktu. Karar günlüğü düz metindir ve tekrar eden satır yapısı taşır → gzip oranı yüksektir.
# Bu klasör OneDrive ile senkronlandığı için kazanç yalnız disk değil YÜKLEME BANDIdır.
DEFAULT_COMPRESS_DAYS = int(os.getenv("LOG_COMPRESS_DAYS", "7"))
# Bugün + dün DOKUNULMAZ: canlı süreç açık tutuyor olabilir.
GRACE_DAYS = 2

_DATE_IN_NAME = re.compile(r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})")


@dataclass(frozen=True)
class Aday:
    yol: Path
    tarih: date
    boyut: int


def _dosya_tarihi(p: Path) -> date | None:
    """Önce ADdaki tarih (güvenilir), yoksa mtime. Çözülemezse None → dosya atlanır."""
    m = _DATE_IN_NAME.search(p.name)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    try:
        return date.fromtimestamp(p.stat().st_mtime)
    except OSError:
        return None


def _hedef_mi(ad: str) -> bool:
    return any(re.match(pat, ad) for pat in PATTERNS)


def adaylar(log_dir: Path | None = None, days: int = DEFAULT_DAYS,
            *, today: date | None = None) -> list[Aday]:
    """Saklama süresini aşmış, silinebilir desendeki dosyalar (SAF seçim; IO=stat)."""
    kok = Path(log_dir) if log_dir else LOG_DIR
    bugun = today or date.today()
    kesim = bugun - timedelta(days=max(days, GRACE_DAYS))
    if not kok.exists():
        return []
    out: list[Aday] = []
    for p in sorted(kok.glob("*")):
        if not p.is_file() or not _hedef_mi(p.name.removesuffix(".gz")):
            continue
        t = _dosya_tarihi(p)
        if t is None or t > kesim:
            continue
        try:
            out.append(Aday(p, t, p.stat().st_size))
        except OSError:
            continue
    return out


def sikistirma_adaylari(log_dir: Path | None = None,
                        compress_days: int = DEFAULT_COMPRESS_DAYS,
                        days: int = DEFAULT_DAYS,
                        *, today: date | None = None) -> list[Aday]:
    """`compress_days`..`days` bandındaki, HENÜZ sıkıştırılmamış hedef dosyalar.

    Silme bandına girmiş dosya buraya ALINMAZ: onu sıkıştırmak bir sonraki koşuda
    silinecek şeye CPU harcamak olurdu. `.gz` dosyaları desenlere uymaz → yeniden
    sıkıştırılmaz (idempotent)."""
    kok = Path(log_dir) if log_dir else LOG_DIR
    bugun = today or date.today()
    ust = bugun - timedelta(days=max(compress_days, GRACE_DAYS))     # bundan eski
    alt = bugun - timedelta(days=max(days, GRACE_DAYS))              # bundan yeni
    if not kok.exists():
        return []
    out: list[Aday] = []
    for p in sorted(kok.glob("*")):
        if not p.is_file() or not _hedef_mi(p.name):
            continue
        t = _dosya_tarihi(p)
        if t is None or t > ust or t <= alt:
            continue
        try:
            out.append(Aday(p, t, p.stat().st_size))
        except OSError:
            continue
    return out


def compress(log_dir: Path | None = None, compress_days: int = DEFAULT_COMPRESS_DAYS,
             days: int = DEFAULT_DAYS, *, apply: bool = False,
             today: date | None = None) -> dict:
    """Orta bandı gzip'le. apply=False (VARSAYILAN) → hiçbir dosyaya dokunulmaz.

    Sıra ÖNEMLİ: önce `.gz` yazılır, sonra kaynak silinir. Ters sırada bir kesinti
    logu YOK EDERDİ. Yazım sırasında hata olursa yarım `.gz` temizlenir ve kaynak
    OLDUĞU GİBİ kalır (atomic_io'nun aynı disiplini, tek dosyalık hâli)."""
    import gzip
    import shutil
    ad = sikistirma_adaylari(log_dir, compress_days, days, today=today)
    hatalar: list[str] = []
    n = 0
    once = sum(a.boyut for a in ad)
    sonra = 0
    if not apply:
        return {"aday": len(ad), "sikistirilan": 0, "once": once, "sonra": 0,
                "apply": False, "hatalar": hatalar}
    for a in ad:
        hedef = a.yol.with_suffix(a.yol.suffix + ".gz")
        try:
            with open(a.yol, "rb") as src, gzip.open(hedef, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        except OSError as e:
            hatalar.append(f"{a.yol.name}: {e}")
            try:
                hedef.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        try:
            sonra += hedef.stat().st_size
            a.yol.unlink()
        except OSError as e:
            hatalar.append(f"{a.yol.name} (kaynak silinemedi): {e}")
            continue
        n += 1
    return {"aday": len(ad), "sikistirilan": n, "once": once, "sonra": sonra,
            "apply": True, "hatalar": hatalar}


def purge(log_dir: Path | None = None, days: int = DEFAULT_DAYS,
          *, apply: bool = False, today: date | None = None) -> dict:
    """Saklama politikasını uygula. apply=False (VARSAYILAN) → hiçbir şey silinmez.

    Dönen: {'aday', 'silinen', 'bayt', 'apply', 'hatalar'}. Silinemeyen dosya (OneDrive
    kilidi/AV) sayımı düşürür ama işlemi DURDURMAZ — bir sonraki koşu tekrar dener."""
    ad = adaylar(log_dir, days, today=today)
    hatalar: list[str] = []
    silinen = 0
    bayt = 0
    if not apply:
        return {"aday": len(ad), "silinen": 0, "bayt": sum(a.boyut for a in ad),
                "apply": False, "hatalar": hatalar}
    for a in ad:
        try:
            a.yol.unlink()
        except OSError as e:
            hatalar.append(f"{a.yol.name}: {e}")
            continue
        silinen += 1
        bayt += a.boyut
    return {"aday": len(ad), "silinen": silinen, "bayt": bayt,
            "apply": True, "hatalar": hatalar}


def main() -> None:
    ap = argparse.ArgumentParser(description="logs/ saklama politikası (B-7)")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS,
                    help=f"bu kadar günden eskiler silinir (varsayılan {DEFAULT_DAYS})")
    ap.add_argument("--compress-days", type=int, default=DEFAULT_COMPRESS_DAYS,
                    help=f"bu kadar günden eskiler gzip'lenir (varsayılan {DEFAULT_COMPRESS_DAYS})")
    ap.add_argument("--apply", action="store_true",
                    help="GERÇEKTEN uygula (yoksa yalnız ne olacağını gösterir)")
    args = ap.parse_args()

    sk = sikistirma_adaylari(compress_days=args.compress_days, days=args.days)
    smb = sum(a.boyut for a in sk) / 1e6
    print(f"logs/ sıkıştırma: {args.compress_days}-{args.days} gün bandı · "
          f"aday {len(sk)} dosya · {smb:.1f} MB")
    ad = adaylar(days=args.days)
    mb = sum(a.boyut for a in ad) / 1e6
    print(f"logs/ saklama: {args.days} gün · aday {len(ad)} dosya · {mb:.1f} MB")
    for a in ad[:10]:
        print(f"   {a.tarih}  {a.boyut / 1e6:7.2f} MB  {a.yol.name}")
    if len(ad) > 10:
        print(f"   ... +{len(ad) - 10} dosya daha")
    if not args.apply:
        print("KURU KOŞU — hiçbir şey silinmedi/sıkıştırılmadı. Uygulamak için: --apply")
        return
    if sk:
        cz = compress(compress_days=args.compress_days, days=args.days, apply=True)
        oran = (cz["sonra"] / cz["once"]) if cz["once"] else 1.0
        print(f"SIKIŞTIRILDI: {cz['sikistirilan']} dosya · "
              f"{cz['once'] / 1e6:.1f} → {cz['sonra'] / 1e6:.1f} MB (×{1 / oran:.1f})")
        for h in cz["hatalar"]:
            print(f"   atlandı (kilitli?): {h}")
    ozet = purge(days=args.days, apply=True)
    print(f"SİLİNDİ: {ozet['silinen']} dosya · {ozet['bayt'] / 1e6:.1f} MB boşaldı")
    for h in ozet["hatalar"]:
        print(f"   atlandı (kilitli?): {h}")


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    main()
