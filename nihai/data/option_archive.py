"""
option_archive.py — OPSİYON ZİNCİRİ ARŞİVİ: geçmişi SATIN ALAMIYORSAN BUGÜNDEN BİRİKTİR.

NEDEN (Tur-59, dış rapor A2): sistemin opsiyon veri kaynakları arasında **tarihsel akış
YOK**. DoltHub `option_chain` gerçek bid/ask + tam greek taşır ama **AÇIK POZİSYON (OI) ve
HACİM KOLONU YOKTUR** (CLAUDE.md, Tur-54) → put/call oranı, olağandışı aktivite ve $GEX
gibi AKIŞ ölçüleri o kaynaktan tarihsel olarak HESAPLANAMAZ. Alpaca OPRA + yfinance ise
OI ve hacmi verir ama **yalnız ANLIK** — dün ne olduğunu soramazsın.

`nihai/ai/option_flow.py` (Tur-57) bu ölçüleri hesaplayabiliyor; eksik olan tek şey
GEÇMİŞTİ. Tarihsel OPRA akışı ücretli ve pahalıdır; ücretsiz ve kesin olan tek yol
**bugünden itibaren günlük anlık görüntü biriktirmek**tir. Bu modül onu yapar.

NE YAPMAZ: kenar iddia etmez, karar yoluna bağlanmaz, mevcut hiçbir ölçümü değiştirmez.
Ürettiği şey bir VERİ SETİDİR; o veri setinden bir aile çıkarılacaksa `research/edge_lab`
kanıt kapısından geçmeli ve defterin `n_trials` sayacını artırmalıdır (Tur-55).

ŞEMA (günlük, gzip'li CSV — `option_chain_YYYY-MM-DD.csv.gz`):
    asof · symbol · spot · expiry · strike · cp · bid · ask · last · iv ·
    delta · gamma · theta · vega · open_interest · volume · source

- `asof` yazım anının ISO damgasıdır (UTC değil YEREL — dosya adı zaten günü taşır).
- Aynı gün birden çok kez koşulursa satırlar EKLENİR (gün içi kesitler); tüketici
  `asof`u okuyarak istediği kesiti seçer. Tekilleştirme YAPILMAZ: iki kesit arasındaki
  OI/hacim FARKI tam da ölçülmek istenen şeydir.
- Ölçülemeyen alan BOŞ kalır, 0 YAZILMAZ (Tur-31/56: ölçüm yokluğu ölçüm gibi sunulamaz —
  OI'si bilinmeyen bir kontrat "OI=0" değildir).

DİZİN: `OPTION_ARCHIVE_DIR` (varsayılan `DATA_DIR/option_archive`). Varsayılanın
OneDrive içinde olması BİLİNÇLİDİR: bu arşiv yeniden ÜRETİLEMEZ (dünün zinciri yarın
alınamaz), yani senkronlanması bir maliyet değil bir YEDEKTİR — 10 GB'lık DoltHub
önbelleğinden (yeniden indirilebilir) farkı budur. Ölçek büyürse env ile taşınır.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
from datetime import date, datetime
from pathlib import Path

from nihai.config import DATA_DIR

FIELDS = ["asof", "symbol", "spot", "expiry", "strike", "cp", "bid", "ask", "last",
          "iv", "delta", "gamma", "theta", "vega", "open_interest", "volume", "source"]

# yfinance/Alpaca zincir kolonları → arşiv alanları (TEK kaynak).
_COLMAP = {
    "strike": "strike", "bid": "bid", "ask": "ask", "lastPrice": "last",
    "impliedVolatility": "iv", "delta": "delta", "gamma": "gamma",
    "theta": "theta", "vega": "vega", "openInterest": "open_interest",
    "volume": "volume",
}


def archive_dir() -> Path:
    d = os.getenv("OPTION_ARCHIVE_DIR")
    return Path(d) if d else (DATA_DIR / "option_archive")


def _num(v) -> str:
    """Sonlu sayı → yuvarlanmış metin; aksi halde BOŞ (uydurma 0 yok)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if f != f or f in (float("inf"), float("-inf")):
        return ""
    return f"{f:.6g}"


def chain_rows(symbol: str, chain: dict, *, asof: str, spot: float | None = None) -> list[dict]:
    """Bir zincir sözlüğünü arşiv satırlarına çevir (SAF fonksiyon, I/O yok).

    `chain` `DataProvider.get_option_chain` çıktısıdır: `{"expiry", "calls", "puts", ...}`.
    Beklenen kolonu olmayan zincir BOŞ liste döndürür (istisna değil — fail-soft)."""
    if not chain:
        return []
    expiry = str(chain.get("expiry") or "")
    src = str(chain.get("option_source") or "yfinance")
    out: list[dict] = []
    for side, cp in (("calls", "C"), ("puts", "P")):
        df = chain.get(side)
        if df is None or getattr(df, "empty", True):
            continue
        cols = [c for c in _COLMAP if c in df.columns]
        for rec in df[cols].to_dict("records"):
            row = {"asof": asof, "symbol": symbol, "spot": _num(spot),
                   "expiry": expiry, "cp": cp, "source": src}
            row.update({_COLMAP[c]: _num(rec.get(c)) for c in cols})
            for f in FIELDS:
                row.setdefault(f, "")
            out.append(row)
    return out


def write_rows(rows: list[dict], *, day: date | None = None,
               out_dir: Path | None = None) -> Path | None:
    """Satırları o günün gzip'li CSV'sine EKLE. Dosya yolu (yazıldıysa) döner.

    gzip append doğrudan desteklenir (çoklu üye akışı): `gzip.open(..., "at")` yeni bir
    gzip ÜYESİ yazar ve `gzip.open(..., "rt")` üyeleri şeffafça birleştirerek okur —
    yani dosyayı yeniden yazmadan gün içi kesit eklenebilir. Başlık yalnız ilk üyeye
    yazılır (çünkü okurken tek akış görünür)."""
    if not rows:
        return None
    d = out_dir or archive_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"option_chain_{(day or date.today()).isoformat()}.csv.gz"
    fresh = not path.exists() or path.stat().st_size == 0
    with gzip.open(path, "at", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if fresh:
            w.writeheader()
        w.writerows(rows)
    return path


def read_day(day: date | str, out_dir: Path | None = None) -> list[dict]:
    """Bir günün arşivini oku (tüketici tarafı; yoksa boş liste)."""
    d = out_dir or archive_dir()
    key = day.isoformat() if isinstance(day, date) else str(day)
    path = d / f"option_chain_{key}.csv.gz"
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def snapshot(symbols: list[str], *, provider=None, expiries: int = 1,
             out_dir: Path | None = None) -> dict:
    """Sembol listesi için anlık zinciri çek ve arşive ekle. Karne döndürür.

    `expiries` kaç vadenin alınacağıdır. **Varsayılan 1'dir ve bu bir maliyet kararıdır:**
    her ek vade bir sağlayıcı çağrısı demektir ve akış ölçüleri (GEX/PCR/UOA) ağırlıklı
    olarak en yakın vadede yoğunlaşır. Daha geniş yüzey isteyen çağıran sayıyı artırır."""
    if provider is None:
        from nihai.data.data_provider import get_provider
        provider = get_provider()
    asof = datetime.now().isoformat(timespec="seconds")
    rows: list[dict] = []
    ok, fail = 0, 0
    for sym in symbols:
        spot = None
        try:
            q = provider.get_quote(sym)
            spot = float(getattr(q, "last", 0) or 0) or None
        except Exception:
            spot = None
        exps: list[str | None] = [None]
        try:
            first = provider.get_option_chain(sym)
            alls = first.get("all_expiries") or []
            if expiries > 1 and alls:
                exps = list(alls[:expiries])
            else:
                r = chain_rows(sym, first, asof=asof, spot=spot)
                rows.extend(r)
                ok += 1 if r else 0
                fail += 0 if r else 1
                continue
        except Exception as e:                                    # noqa: BLE001
            print(f"[opt-arşiv] {sym}: zincir alınamadı ({e})", file=sys.stderr)
            fail += 1
            continue
        for exp in exps:
            try:
                ch = provider.get_option_chain(sym, exp)
                r = chain_rows(sym, ch, asof=asof, spot=spot)
                rows.extend(r)
                ok += 1 if r else 0
                fail += 0 if r else 1
            except Exception as e:                                # noqa: BLE001
                print(f"[opt-arşiv] {sym} {exp}: {e}", file=sys.stderr)
                fail += 1
    path = write_rows(rows, out_dir=out_dir)
    return {"asof": asof, "symbols": len(symbols), "rows": len(rows),
            "chains_ok": ok, "chains_fail": fail,
            "path": str(path) if path else None,
            "bytes": path.stat().st_size if path else 0}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="OPRA/yfinance opsiyon zincirinin GÜNLÜK anlık arşivi (OI+hacim dahil)")
    ap.add_argument("--symbols", default=os.getenv("OPTION_ARCHIVE_SYMBOLS", "SPY,QQQ"),
                    help="virgülle ayrık semboller")
    ap.add_argument("--expiries", type=int, default=1, help="kaç vade (varsayılan 1)")
    ap.add_argument("--dir", default=None, help="çıktı dizini (vars: OPTION_ARCHIVE_DIR)")
    args = ap.parse_args()
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    res = snapshot(syms, expiries=args.expiries,
                   out_dir=Path(args.dir) if args.dir else None)
    print(f"[opt-arşiv] {res['rows']} satır · zincir ok/fail={res['chains_ok']}/"
          f"{res['chains_fail']} · {res['bytes'] / 1e6:.2f} MB · {res['path']}")


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout

    ensure_utf8_stdout()
    main()
