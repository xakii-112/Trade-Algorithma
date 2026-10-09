"""
backfill_iv_history.py — `data/iv_history/<SYM>.csv`'yi DoltHub'dan geriye doldurur.

SORUN: `nihai/data/iv_history.py` docstring'i açıkça söylüyor — "yfinance opsiyon
tarihsel IV vermiyor. Bu modül her gün ATM straddle IV'yi ekler; zamanla 52w rolling
oluşur." Yani IV Rank (Madde 16) sıfırdan başlayan bir kovaya su taşıyor: 30 güne
kadar LOW_CONFIDENCE, 100 güne kadar PARTIAL, ~252 gün sonra OK. Servisler kapalı
kaldıkça sayaç da duruyor. Pratikte `get_iv_rank` üretimde hep 50.0 (NO_HISTORY)
dönüyordu ve `config.iv_rank_high_threshold=50 / low=30` eşikleri hiç sınanamadı.

ÇÖZÜM: DoltHub `post-no-preference/options` 2019-02-09'dan bugüne 2321 sembol için
günlük `iv_current` tutuyor. Onu `iv_history.py`'nin ZATEN OKUDUĞU `date,iv` biçimine
yazarız → `get_iv_rank` tek satır değişmeden gerçek 252-gün penceresiyle çalışır.
(Mevcut koda dokunmamak bilinçli: çalışan bir okuyucuyu yeniden yazmak yerine
beslemek en kısa ve en az riskli yol.)

KAYNAK KARIŞTIRMA UYARISI: canlı biriken seri yfinance ATM straddle IV'sidir, Dolt
serisi başka bir sağlayıcının `iv_current`'ıdır. İkisini aynı CSV'de karıştırmak
`alpaca_history.py`'nin ham/düzeltilmiş barlar için uyardığı sessiz bozulmanın
aynısıdır (min/max aynı pencerede iki farklı ölçekten gelirse rank anlamsızlaşır).
Bu yüzden VARSAYILAN davranış: eski dosyayı `.yedek-<zaman>` olarak sakla, yerine
TEK KAYNAKLI Dolt serisi yaz. Karıştırmak isteyen `--merge` demek zorunda.

Kullanım:
    python -m research.backfill_iv_history --symbols AAPL,MSFT,SPY
    python -m research.backfill_iv_history --from-env          # .env STREAM_SYMBOLS
    python -m research.backfill_iv_history --symbols AAPL --as-of HEAD~50   # PIT
    python -m research.backfill_iv_history --symbols AAPL --dry-run
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd
from loguru import logger

from nihai.data import dolt_options
from nihai.data.iv_history import IV_HIST_DIR, _path, get_iv_rank, history_stats


def _symbols_from_env() -> list[str]:
    raw = os.getenv("STREAM_SYMBOLS", "")
    return [s.strip().upper() for s in raw.replace(";", ",").split(",") if s.strip()]


def already_backfilled(symbol: str, *, min_rows: int = 200) -> bool:
    """Bu sembol daha önce dolduruldu mu? (devam ettirilebilirlik için)

    Ölçüt satır sayısıdır: canlı birikim günde 1 satır ekler ve servisler aylardır
    aralıklı çalıştığı için mevcut dosyalar ~20-30 satırdır; Dolt dolgusu 1000+.
    `min_rows` bu ikisini ayırmaya yeter.
    """
    p = _path(symbol.upper())
    if not p.exists():
        return False
    try:
        return len(pd.read_csv(p)) >= min_rows
    except Exception:
        return False


def backfill_symbol(symbol: str, *, start: str | None = None, as_of: str | None = None,
                    merge: bool = False, dry_run: bool = False,
                    df: pd.DataFrame | None = None) -> dict:
    """Tek sembolü doldur. Dönen sözlük: ne yazıldı, kaç satır, kaynak durumu.

    `df` verilirse (toplu kip) ağa hiç çıkılmaz; verilmezse SQL API'den çekilir.
    """
    symbol = symbol.upper()
    if df is None:
        df = dolt_options.volatility_history(symbol, start=start, as_of=as_of)
    if df.empty or "iv_current" not in df.columns:
        return {"symbol": symbol, "durum": "VERI_YOK", "satir": 0}
    if start:
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(start)]

    new = (df[["date", "iv_current"]]
           .rename(columns={"iv_current": "iv"})
           .dropna())
    new["date"] = pd.to_datetime(new["date"]).dt.strftime("%Y-%m-%d")
    new = new[(new["iv"] > 0) & (new["iv"] <= 5.0)]      # iv_history ile aynı elek
    if new.empty:
        return {"symbol": symbol, "durum": "GECERLI_SATIR_YOK", "satir": 0}

    p = _path(symbol)
    onceki = len(pd.read_csv(p)) if p.exists() else 0

    if merge and p.exists():
        eski = pd.read_csv(p)
        # Dolt satırları taban; canlı satırlar YALNIZ Dolt'ta olmayan tarihlerde eklenir
        eksik = eski[~eski["date"].isin(set(new["date"]))]
        out = pd.concat([new, eksik], ignore_index=True)
        kaynak = f"KARISIK(dolt={len(new)}, canli={len(eksik)})"
    else:
        out = new
        kaynak = "DOLT_TEK_KAYNAK"

    out = out.sort_values("date").drop_duplicates("date", keep="first")

    if dry_run:
        return {"symbol": symbol, "durum": "DRY_RUN", "satir": len(out),
                "onceki": onceki, "kaynak": kaynak}

    IV_HIST_DIR.mkdir(parents=True, exist_ok=True)
    if p.exists() and not merge:
        yedek = p.with_suffix(f".csv.yedek-{datetime.now():%Y%m%d-%H%M%S}")
        p.replace(yedek)
        logger.info(f"[backfill] {symbol}: eski dosya → {yedek.name}")
    out.to_csv(p, index=False)
    return {"symbol": symbol, "durum": "YAZILDI", "satir": len(out),
            "onceki": onceki, "kaynak": kaynak}


def main() -> int:
    ap = argparse.ArgumentParser(description="IV history'yi DoltHub'dan geriye doldur")
    ap.add_argument("--symbols", help="virgülle ayrık liste, örn AAPL,MSFT,SPY")
    ap.add_argument("--from-env", action="store_true", help=".env STREAM_SYMBOLS kullan")
    ap.add_argument("--from-existing", action="store_true",
                    help="data/iv_history/ içinde dosyası olan TÜM sembolleri doldur "
                         "(toplu CSV yerelken ek ağ maliyeti yok)")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (varsayılan: tüm geçmiş)")
    ap.add_argument("--as-of", default=None,
                    help="PIT commit'i, örn HEAD~50 (survivorship-temiz okuma)")
    ap.add_argument("--merge", action="store_true",
                    help="mevcut CANLI satırları koru (KAYNAK KARIŞIR — docstring'i oku)")
    ap.add_argument("--dry-run", action="store_true", help="yazma, yalnız raporla")
    ap.add_argument("--skip-existing", action="store_true",
                    help="zaten dolu (>=200 satır) sembolleri atla — koşuyu sürdürür")
    ap.add_argument("--api", action="store_true",
                    help="toplu CSV yerine SQL API kullan (yavaş + hız sınırlı; "
                         "yalnız birkaç sembol için)")
    ap.add_argument("--force-download", action="store_true",
                    help="toplu CSV önbelleği taze olsa da yeniden indir")
    args = ap.parse_args()

    syms = []
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.from_env:
        syms = _symbols_from_env()
    elif args.from_existing:
        syms = sorted(p.stem.upper() for p in IV_HIST_DIR.glob("*.csv"))
    if not syms:
        print("sembol yok — --symbols / --from-env / --from-existing verin")
        return 2

    kip = "SQL API" if args.api else "toplu CSV"
    print(f"{len(syms)} sembol dolduruluyor  kip={kip} "
          f"(as_of={args.as_of or 'HEAD'}, merge={args.merge}, dry_run={args.dry_run})\n")

    if args.skip_existing:
        atlanacak = {s for s in syms if already_backfilled(s)}
        for s in sorted(atlanacak):
            print(f"  {s:<6} ATLANDI  (zaten dolu)")
        syms = [s for s in syms if s not in atlanacak]
    else:
        atlanacak = set()

    # Toplu kip: TEK indirme, sonra yerelde süz. SQL API'nin sembol-başına tam
    # taraması ve hız sınırı bu yolda hiç devreye girmez.
    toplu: pd.DataFrame | None = None
    if not args.api and syms:
        toplu = dolt_options.volatility_history_bulk(
            syms, ref=args.as_of, force=args.force_download)
        if toplu.empty:
            print("  toplu CSV boş/indirilemedi — --api ile tek tek deneyebilirsiniz")
            return 1

    ok = 0
    atlanan = len(atlanacak)
    for s in syms:
        try:
            pay = None
            if toplu is not None:
                pay = toplu[toplu["act_symbol"].str.upper() == s].copy()
            r = backfill_symbol(s, start=args.start, as_of=args.as_of,
                                merge=args.merge, dry_run=args.dry_run, df=pay)
        except Exception as e:                       # tek sembol partiyi düşürmesin
            print(f"  {s:<6} HATA: {e}")
            continue
        if r["durum"] in ("YAZILDI", "DRY_RUN"):
            ok += 1
            print(f"  {s:<6} {r['durum']:<8} {r['satir']:>5} satır "
                  f"(önce {r.get('onceki', 0)}) {r.get('kaynak', '')}")
        else:
            print(f"  {s:<6} {r['durum']}")

    print(f"\n{ok}/{len(syms)} sembol tamam"
          + (f" ({atlanan} atlandı)." if atlanan else "."))
    if ok and not args.dry_run:
        # Kanıt: rank artık gerçekten hesaplanıyor mu? (önce hep NO_HISTORY idi)
        print("\nIV Rank kontrolü (mevcut IV = serinin son değeri):")
        for s in syms[:5]:
            st = history_stats(s)
            if not st["days"]:
                continue
            son_iv = pd.read_csv(_path(s))["iv"].iloc[-1]
            rank = get_iv_rank(s, float(son_iv))
            print(f"  {s:<6} gün={st['days']:>5} {st['first_date']}→{st['last_date']} "
                  f"rank={rank.iv_rank:5.1f} güven={rank.confidence}")
    return 0


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout

    ensure_utf8_stdout()
    raise SystemExit(main())
