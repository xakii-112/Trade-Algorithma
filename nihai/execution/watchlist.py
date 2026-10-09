"""
watchlist.py — Çalışma anında değişebilen, kalıcı (JSON) dinamik watchlist.

Mimari:
- config.WATCHLIST sabit kalır → yalnızca ilk açılış tohumu.
- Asıl canlı liste data/watchlist.json'da tutulur; oto-tarayıcı (auto_screener)
  taramanın en iyi N skoruyla bu listeyi yeniler (top-N dinamik mod).
- main döngüsü her turda current() okur → arka plan thread güncellediğinde
  bir sonraki tur yeni listeyle döner. Thread-safe (RLock).

Dosya şeması:
    {
      "symbols": ["AAPL", "NVDA", ...],          # sıralı (en iyi → en kötü)
      "meta": {"AAPL": {"score": 1.23, "metric": "expectancy",
                         "added": "2026-06-02T09:31:00", "source": "auto"}},
      "updated": "2026-06-02T09:31:00",
      "last_scan": "2026-06-02T09:31:00"
    }
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime

from loguru import logger

from nihai.atomic_io import atomic_write_text
from nihai.config import WATCHLIST, WATCHLIST_FILE

_lock = threading.RLock()
_state: dict | None = None
_stamp: tuple | None = None


def _file_stamp() -> tuple:
    try:
        info = WATCHLIST_FILE.stat()
        return (str(WATCHLIST_FILE), info.st_mtime_ns, info.st_size)
    except FileNotFoundError:
        return (str(WATCHLIST_FILE), None, None)


@contextmanager
def _transaction():
    """Web, Telegram ve ayrı tarayıcı süreçlerinin pin/block yazıları kaybolmasın."""
    with _lock:
        WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        with WATCHLIST_FILE.with_suffix(".lock").open("a+b") as handle:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _seed_state() -> dict:
    """config.WATCHLIST'ten tohum durum üret."""
    syms = [s.strip().upper() for s in WATCHLIST if s.strip()]
    meta = {s: {"score": None, "metric": None, "added": _now_iso(), "source": "seed"}
            for s in syms}
    return {"symbols": syms, "meta": meta, "pins": [], "blocks": [],
            "updated": _now_iso(), "last_scan": None}


def _load() -> dict:
    """Diskten yükle; yoksa config tohumuyla başlat ve yaz."""
    global _state, _stamp
    with _lock:
        stamp = _file_stamp()
        if _state is not None and stamp == _stamp:
            return _state
        try:
            if WATCHLIST_FILE.exists():
                data = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("symbols"), list):
                    data.setdefault("meta", {})
                    data.setdefault("pins", [])
                    data.setdefault("blocks", [])
                    data.setdefault("updated", _now_iso())
                    data.setdefault("last_scan", None)
                    _state = data
                    _stamp = stamp
                    logger.info(f"[watchlist] yüklendi: {len(data['symbols'])} sembol "
                                f"(son tarama: {data.get('last_scan') or 'yok'})")
                    return _state
                raise ValueError("watchlist şeması geçersiz")
        except Exception as e:
            logger.warning(f"[watchlist] {WATCHLIST_FILE} okunamadı ({e})")
            # Bozuk dosyayı tohumla ezme; operatörün verisi korunur.
            raise
        _state = _seed_state()
        _save_locked()
        logger.info(f"[watchlist] tohum oluşturuldu: {_state['symbols']}")
        return _state


def _save_locked() -> None:
    """_lock tutulurken çağrılır."""
    global _stamp, _state
    try:
        WATCHLIST_FILE.parent.mkdir(exist_ok=True)
        # Atomik + OneDrive-kilidine dayanıklı yazma (Tur-12, nihai/atomic_io; state.py ile aynı
        # gerekçe): çökme watchlist.json'u yarım bırakmasın (yoksa _load config tohumuna düşüp
        # dinamik watchlist'i kaybeder); OneDrive senkron kilidi ham os.replace'i düşürmesin.
        atomic_write_text(WATCHLIST_FILE,
                          json.dumps(_state, ensure_ascii=False, indent=2))
        _stamp = _file_stamp()
    except Exception as e:
        _state = None  # başarısız yazımı bellekte başarılı gibi sunma
        logger.error(f"[watchlist] yazma hatası: {e}")
        raise


# ---------------------------------------------------------------------------
# Genel API
# ---------------------------------------------------------------------------
def current() -> list[str]:
    """Canlı watchlist sembolleri (sıralı kopya)."""
    with _transaction():
        return list(_load()["symbols"])


def partition(default_n: int) -> dict[str, list[str]]:
    """Son taramanın Top-N seçimi bütün canlı süreçlerde aynı katman sınırıdır."""
    with _transaction():
        st = _load()
        count = st.get("focused_n", default_n)
        if type(count) is not int or count < 0:
            count = default_n
        return {"focused": list(st["symbols"][:count]), "broad": list(st["symbols"][count:])}


def meta() -> dict:
    """Sembol → metadata kopyası."""
    with _transaction():
        return dict(_load().get("meta", {}))


def last_scan() -> str | None:
    with _transaction():
        return _load().get("last_scan")


def risk_mult(symbol: str) -> float:
    """A5 — sembolün güven-bazlı risk çarpanı (meta'dan). Yoksa/geçersizse 1.0."""
    with _transaction():
        m = _load().get("meta", {}).get(symbol.strip().upper(), {})
        v = m.get("risk_mult")
        return float(v) if isinstance(v, (int, float)) and v > 0 else 1.0


def replace(symbols: list[str], *, meta_by_sym: dict | None = None,
            source: str = "auto", mark_scan: bool = True, focused_n: int | None = None) -> list[str]:
    """
    Listeyi tamamen yenile (top-N / dinamik mod). Manuel kontrol korunur:
    - PIN'ler her zaman dahil edilir ve başa konur (oto-tarama silemez).
    - BLOCK'lar her zaman hariç tutulur (oto-tarama geri ekleyemez).
    Zaten listede olan sembollerin ilk eklenme tarihi ('added') bozulmaz.
    """
    if focused_n is not None and (type(focused_n) is not int or focused_n < 0):
        raise ValueError("Katman sınırı negatif olmayan tamsayı olmalı.")
    with _transaction():
        st = _load()
        old_meta = st.get("meta", {})
        pins = [s.strip().upper() for s in st.get("pins", [])]
        blocks = {s.strip().upper() for s in st.get("blocks", [])}
        new_syms, new_meta = [], {}
        # Önce pinler (manuel, her zaman dahil), sonra otomatikler (bloklananlar hariç)
        for s in [*pins, *symbols]:
            s = s.strip().upper()
            if not s or s in new_meta or s in blocks:
                continue
            prev = old_meta.get(s, {})
            m = (meta_by_sym or {}).get(s, {})
            is_pin = s in pins
            new_meta[s] = {
                "score": m.get("score", prev.get("score")),
                "metric": m.get("metric", prev.get("metric")),
                "added": prev.get("added", _now_iso()),  # ilk eklenmeyi koru
                "source": "manual" if is_pin else source,
                # A5 — güven-bazlı risk çarpanı (boyutlamada uygulanır; default 1.0)
                "risk_mult": m.get("risk_mult", prev.get("risk_mult", 1.0)),
            }
            new_syms.append(s)
        st["symbols"] = new_syms
        st["meta"] = new_meta
        if focused_n is not None:
            st["focused_n"] = focused_n
        st["updated"] = _now_iso()
        if mark_scan:
            st["last_scan"] = _now_iso()
        _save_locked()
        return list(new_syms)


def pins() -> list[str]:
    """Manuel sabitlenen (her zaman dahil) semboller."""
    with _transaction():
        return list(_load().get("pins", []))


def blocks() -> list[str]:
    """Manuel engellenen (her zaman hariç) semboller."""
    with _transaction():
        return list(_load().get("blocks", []))


def add(symbol: str, *, score: float | None = None, metric: str | None = None,
        source: str = "manual", pin: bool = False) -> bool:
    """
    Tek sembol ekle. pin=True ise kalıcı sabitlenir (oto-tarama silemez) ve
    varsa engel listesinden çıkarılır. Hiçbir şey değişmediyse False döner.
    """
    symbol = symbol.strip().upper()
    if not symbol:
        return False
    with _transaction():
        st = _load()
        changed = False
        if pin and symbol not in st["pins"]:
            st["pins"].append(symbol)
            changed = True
        if symbol in st["blocks"]:           # manuel ekleme engeli geçersiz kılar
            st["blocks"].remove(symbol)
            changed = True
        if symbol not in st["symbols"]:
            st["symbols"].insert(0 if pin else len(st["symbols"]), symbol)
            st["meta"][symbol] = {"score": score, "metric": metric,
                                  "added": _now_iso(),
                                  "source": "manual" if pin else source}
            changed = True
        if changed:
            st["updated"] = _now_iso()
            _save_locked()
            logger.info(f"[watchlist] eklendi: {symbol} (pin={pin})")
        return changed


def remove(symbol: str, *, block: bool = False) -> bool:
    """
    Tek sembol çıkar. block=True ise engel listesine eklenir (oto-tarama geri
    ekleyemez). block=False ise ve sembol zaten engelliler listesindeyse, engeli kaldırır.
    Hiçbir şey değişmediyse False döner.
    """
    symbol = symbol.strip().upper()
    with _transaction():
        st = _load()
        changed = False
        if symbol in st["symbols"]:
            st["symbols"].remove(symbol)
            st["meta"].pop(symbol, None)
            changed = True
        if symbol in st["pins"]:
            st["pins"].remove(symbol)
            changed = True
        if block and symbol not in st["blocks"]:
            st["blocks"].append(symbol)
            changed = True
        elif not block and symbol in st["blocks"]:
            st["blocks"].remove(symbol)
            changed = True
        if changed:
            st["updated"] = _now_iso()
            _save_locked()
            logger.info(f"[watchlist] çıkarıldı/unblock yapıldı: {symbol} (block={block})")
        return changed



# ---------------------------------------------------------------------------
# CLI — agent-native parite: web/telegram'daki manuel kontrol terminalden de.
#   python watchlist.py                 # listele
#   python watchlist.py add NVDA        # pinle (oto-tarama silemez)
#   python watchlist.py remove TSLA     # engelle (oto-tarama geri ekleyemez)
#   python watchlist.py add NVDA --no-pin / remove TSLA --no-block  # geçici
# ---------------------------------------------------------------------------
def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Dinamik watchlist yönetimi")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="watchlist + pin/block + son tarama")
    pa = sub.add_parser("add", help="sembol ekle (varsayılan: pinle)")
    pa.add_argument("symbol")
    pa.add_argument("--no-pin", action="store_true", help="pinleme (geçici, oto-tarama silebilir)")
    pr = sub.add_parser("remove", help="sembol çıkar (varsayılan: engelle)")
    pr.add_argument("symbol")
    pr.add_argument("--no-block", action="store_true", help="engelleme (oto-tarama geri ekleyebilir)")
    args = ap.parse_args(argv)

    if args.cmd == "add":
        changed = add(args.symbol, pin=not args.no_pin, source="cli")
        print(f"{args.symbol.upper()}: {'eklendi' if changed else 'zaten vardı'} "
              f"(pin={not args.no_pin})")
    elif args.cmd == "remove":
        changed = remove(args.symbol, block=not args.no_block)
        print(f"{args.symbol.upper()}: {'çıkarıldı' if changed else 'listede yoktu'} "
              f"(block={not args.no_block})")
    else:  # list (varsayılan)
        cur = current()
        print(f"Watchlist ({len(cur)}): {', '.join(cur) or '—'}")
        print(f"📌 pin   : {', '.join(pins()) or '—'}")
        print(f"🚫 block : {', '.join(blocks()) or '—'}")
        print(f"son tarama: {last_scan() or '—'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
