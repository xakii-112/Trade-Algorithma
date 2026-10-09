"""
deadman.py — ÖLÜ ADAM ANAHTARI: kanıt akışı DURDUYSA haber ver.

NEDEN (Tur-59, dış rapor G2/P1): bu depoda izleme VAR (`observability` → `/metrics`,
`daily_summary` → günlük özet) ama ikisi de **süreç yaşıyorken** konuşur. Bir servis
öldüğünde `/metrics` yanıt vermez ve günlük özet ertesi güne kadar susar; operatör
Bayesyen öğrenmenin ve canlı tahmin defterinin DURDUĞUNU günler sonra fark eder.
Projenin kendi ölçümü bunun gerçek olduğunu söylüyor: servisler 2026-06-29→07-10 arası
kapalı kaldı, Tur-25'te nöral köprünün canlı kanıtı **6 satırdı**.

NE YAPAR (üçü de ölçüm, hiçbiri karar):
  1. **Tazelik denetimi** — canlı kanıt dosyaları (`predictions_log.csv`, `state.json`,
     `live_features_log.csv`) son N dakikada YAZILDI mı?
  2. **Telegram uyarısı** — bayatsa tek satırlık uyarı (`notify_alert`).
  3. **Dış ping (Dead Man's Snitch)** — SAĞLIKLIYSA `DEADMAN_SNITCH_URL`e GET atar. Bu
     mantık TERSTİR ve kasıtlıdır: makinenin kendisi ölürse (elektrik/ağ/OneDrive kilidi)
     hiçbir yerel uyarı çıkamaz; ping'in YOKLUĞU dış servisi tetikler.

SEANS KAPISI — YANLIŞ ALARM ÜRETMEZ. Piyasa kapalıyken tahmin üretilmemesi normaldir;
denetim yalnız seans açıkken (+ `--grace` dakika sonra) uyarır. `is_market_open`
okunamazsa hiçbir şey iddia edilmez (`BILINMIYOR`) — ölçüm yokluğu bir ALARM gerekçesi
değildir (Tur-31 disiplini).

Kullanım:
    python -m nihai.notifier.deadman                 # denetle + yazdır
    python -m nihai.notifier.deadman --send          # bayatsa Telegram uyarısı
    python -m nihai.notifier.deadman --send --ping   # + sağlıklıysa dış snitch ping'i

Zamanlama: mevcut watchdog script'lerinin yanına (ör. 15 dk'da bir) eklenir; bu modül
KENDİ zamanlayıcısını KURMAZ (bu depoda görev kaydı operatörün bilinçli adımıdır).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from nihai.config import DATA_DIR, LOG_DIR

# İzlenen kanıt dosyaları: (ad, yol, seans-içi azami bayatlık dk).
# Eşikler dosyanın YAZIM RİTMİNDEN türer, keyfi değildir:
#   predictions_log  — tahmin servisi her barda yazar (PREDICT_REFRESH_SEC ~20 sn)
#   state.json       — tarayıcı döngüsü her turda yazar
#   live_features_log— predictions_log ile AYNI çağrıda yazılır (BULGU-59-1 sonrası her bar)
_WATCHED: tuple[tuple[str, str, int], ...] = (
    ("predictions_log", "predictions_log.csv", 30),
    ("state", "state.json", 30),
    ("live_features_log", "live_features_log.csv", 60),
)
# Seanstan BAĞIMSIZ günlük kanıt (Tur-88): öğrenme hafızasını Tur-87'den beri YALNIZ gece görevi
# (12:00 + 23:30) yazar. 26 sa = iki ardışık tetik kaçtı + 2 sa pay; tek kaçış yedekle kapanır.
# Ölçüldü: 2026-09-21/22 uykusu iki tetiği birden kaçırdı, hafıza ~28 sa sessizce bayat kaldı.
_DAILY: tuple[tuple[str, str, int], ...] = (
    ("ai_learning_state", "ai_learning_state.json", 26 * 60),
)


def _age_minutes(path: Path) -> float | None:
    """Dosyanın son yazımından bu yana geçen dakika; dosya yoksa None (ÖLÇÜM YOK)."""
    try:
        return (time.time() - path.stat().st_mtime) / 60.0
    except OSError:
        return None


def _proje_kapali(log_dir: Path | None = None) -> bool:
    """Operatör projeyi BİLEREK kapattı mı? (`logs/project.stopped` bayrağı)

    BULGU-61-8 (Tur-61): `deadman` bu bayrağı bilmiyordu. `proje-kapat.ps1` servisleri
    kasten durdurur ve watchdog'lar bayrağı GÖRÜP yeniden başlatmayı bırakır
    (`predict_watchdog.log`: *"STOP bayragi bulundu - yeniden baslatma DURDURULDU"*).
    Ölü adam anahtarı o hâlde seans boyunca her 15 dakikada bir alarm verirdi — ve
    bilinçli bir duruma bağıran bir izleyicinin sonu SUSTURULMAKTIR; asıl arıza günü
    çalışmayan bir alarm, hiç olmayandan kötüdür. Bayrak watchdog'larla AYNI
    sözleşmedir; `deadman` de ona uyar."""
    kok = Path(log_dir) if log_dir else LOG_DIR
    try:
        return (kok / "project.stopped").exists()
    except OSError:
        return False


def _market_open() -> bool | None:
    """Seans açık mı? Takvim okunamazsa None — 'bilinmiyor' alarm gerekçesi değildir."""
    try:
        from nihai.data.market_calendar import is_market_open
        return bool(is_market_open())
    except Exception:
        return None


def check(*, grace_min: float = 0.0, data_dir: Path | None = None,
          log_dir: Path | None = None) -> dict:
    """Tazelik karnesi. SAF: hiçbir şey göndermez, hiçbir şey yazmaz.

    Dönen sözlük:
      market_open : True/False/None (None = takvim okunamadı)
      files       : {ad: {"age_min": float|None, "limit_min": int, "stale": bool|None}}
      stale       : bayat olan dosya adları (seans KAPALIYKEN yalnız `_DAILY` girebilir)
      verdict     : "SAGLIKLI" | "BAYAT" | "SEANS_KAPALI" | "PROJE_KAPALI" | "BILINMIYOR"

    `stale` üç durumlu: True (bayat), False (taze), **None (dosya yok = ölçüm yok)**.
    Dosyanın yokluğu bayatlık DEĞİLDİR — henüz hiç koşmamış bir kurulumu "öldü" diye
    raporlamak yanlış alarmdır (Tur-31)."""
    root = Path(data_dir) if data_dir else DATA_DIR
    kapali = _proje_kapali(log_dir)
    mo = _market_open()
    files: dict[str, dict] = {}
    stale: list[str] = []
    for name, fn, limit in _WATCHED:
        age = _age_minutes(root / fn)
        is_stale: bool | None
        if age is None:
            is_stale = None
        else:
            is_stale = age > (limit + grace_min)
        files[name] = {"age_min": age, "limit_min": limit, "stale": is_stale}
        # BULGU-61-8: bilerek kapatılmış projede bayatlık BEKLENEN durumdur, alarm DEĞİL.
        # Tazelik ÖLÇÜMÜ yine de raporlanır (bilgi gizlenmez, yalnız alarma dönüşmez).
        if is_stale and mo is True and not kapali:
            stale.append(name)
    for name, fn, limit in _DAILY:
        age = _age_minutes(root / fn)
        is_stale = None if age is None else age > limit
        files[name] = {"age_min": age, "limit_min": limit, "stale": is_stale}
        if is_stale and not kapali:
            stale.append(name)
    if kapali:
        verdict = "PROJE_KAPALI"
    elif stale:
        verdict = "BAYAT"
    elif mo is None:
        verdict = "BILINMIYOR"
    elif mo is False:
        verdict = "SEANS_KAPALI"
    else:
        verdict = "BAYAT" if stale else "SAGLIKLI"
    return {"market_open": mo, "project_stopped": kapali, "files": files,
            "stale": stale, "verdict": verdict}


def _snitch(url: str, timeout: float = 10.0) -> bool:
    """Dış 'hâlâ hayattayım' ping'i. Başarısızlık SESSİZ değildir ama AKIŞI BOZMAZ."""
    try:
        import requests
        requests.get(url, timeout=timeout).raise_for_status()
        return True
    except Exception as e:                                        # noqa: BLE001
        print(f"[deadman] snitch ping başarısız: {type(e).__name__}", file=sys.stderr)
        return False


def format_report(res: dict) -> str:
    """Tek ekranlık insan-okur özet."""
    mo = {True: "AÇIK", False: "KAPALI", None: "?"}[res["market_open"]]
    kapali = " proje=KAPATILDI (logs/project.stopped)" if res.get("project_stopped") else ""
    lines = [f"[deadman] seans={mo} verdikt={res['verdict']}{kapali}"]
    for name, d in res["files"].items():
        age = d["age_min"]
        yas = "yok" if age is None else f"{age:.1f} dk"
        bayrak = {True: "BAYAT", False: "taze", None: "ölçüm yok"}[d["stale"]]
        lines.append(f"  {name:<20} yaş={yas:<10} sınır={d['limit_min']} dk  → {bayrak}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Canlı kanıt akışı durdu mu? (ölü adam anahtarı)")
    ap.add_argument("--send", action="store_true", help="bayatsa Telegram uyarısı gönder")
    ap.add_argument("--ping", action="store_true",
                    help="SAĞLIKLIYSA DEADMAN_SNITCH_URL'e GET at (dış ölü-adam servisi)")
    ap.add_argument("--grace", type=float, default=float(os.getenv("DEADMAN_GRACE_MIN", "5")),
                    help="seans açılışından/eşikten sonra ek tolerans (dk)")
    args = ap.parse_args()

    res = check(grace_min=args.grace)
    print(format_report(res))

    if res["verdict"] == "BAYAT" and args.send:
        from nihai.notifier import telegram_bot
        detay = ", ".join(
            f"{n}={res['files'][n]['age_min']:.0f}dk" for n in res["stale"])
        telegram_bot.notify_alert(
            "KRITIK", f"Kanıt akışı DURDU: {detay}. "
                      "Tahmin defteri ve Bayesyen öğrenme birikmiyor.")
    if res["verdict"] == "SAGLIKLI" and args.ping:
        url = (os.getenv("DEADMAN_SNITCH_URL") or "").strip()
        if url:
            _snitch(url)
        else:
            print("[deadman] --ping verildi ama DEADMAN_SNITCH_URL boş — atlandı",
                  file=sys.stderr)

    # Çıkış kodu MAKİNE tarafından okunabilir olsun (watchdog script'i dallanabilsin):
    # 0 = sağlıklı/kapalı/bilinmiyor, 1 = BAYAT.
    print(json.dumps({"verdict": res["verdict"], "stale": res["stale"]}, ensure_ascii=False))
    sys.exit(1 if res["verdict"] == "BAYAT" else 0)


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout

    ensure_utf8_stdout()
    main()
