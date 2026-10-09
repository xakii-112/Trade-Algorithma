"""
alpaca_coverage_probe.py — Alpaca tarihsel veri KAPSAMA sondası (offline eğitim fizibilitesi).

Soru: Alpaca'dan geçmiş yılların verisiyle meta-model eğitmek mümkün mü? Bunun iki
ön şartı ölçülür:
1. WATCHLIST kapsaması: canlı watchlist'teki tüm sembollerin günlük bar geçmişi ne
   kadar geriye gidiyor? (eğitim penceresi derinliği)
2. DELİST kapsaması (asıl kritik): pit_universe.removed_since() isimlerinin — yani
   survivorship düzeltmesinin geri eklediği, bugün endekste OLMAYAN şirketlerin —
   verisi Alpaca'da var mı? Bu YOKSA Alpaca dataset'i de survivorship-şişkin olur
   ve plan çöker (ACI DERS #1).
Ek: küçük bir örneklemde SAATLİK bar derinliği (1h PIT dataset fizibilitesi).

Salt-okuma API çağrıları; hiçbir servis başlatmaz, üretim dosyası değiştirmez.
Çıktı: stdout raporu + data/alpaca_coverage_probe.csv (sembol-başına kapsama).

Kullanım (repo kökünden):
    python -m research.alpaca_coverage_probe
    python -m research.alpaca_coverage_probe --as-of 2018-01-01 --hourly-sample 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv()

from nihai.config import DATA_DIR  # noqa: E402
from nihai.execution import watchlist as watchlist_mod  # noqa: E402

PROBE_START = datetime(2015, 1, 1)
# SIP'te son 15 dk Basic planda yasak; güvenli taraf için dünü uç al.
PROBE_END = datetime.now() - timedelta(days=1)
BATCH = 100          # çoklu-sembol bar isteği başına sembol
HOURLY_START = datetime(2016, 1, 1)


def _alpaca_sym(t: str) -> str:
    """yfinance/wiki tarzı 'BRK-B' → Alpaca tarzı 'BRK.B'."""
    return t.strip().upper().replace("-", ".")


def _client():
    from alpaca.data.historical import StockHistoricalDataClient
    key, sec = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not sec:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY env'de yok (.env)")
    return StockHistoricalDataClient(key, sec)


def _bars_batch(client, symbols: list[str], timeframe, start: datetime,
                end: datetime) -> dict[str, dict]:
    """Bir sembol grubunun bar kapsamasını ölç: {sym: {first, last, n}}.
    Yanıtta hiç görünmeyen sembol = o pencerede veri YOK."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    # Enum (str değil): alpaca-py string'i zaten aynı değere çözüyor — tip güvenliği için.
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=timeframe,
                           start=start, end=end, adjustment=Adjustment.ALL,
                           feed=DataFeed.SIP)
    out: dict[str, dict] = {}
    resp = client.get_stock_bars(req)
    data = getattr(resp, "data", {}) or {}
    for sym, bars in data.items():
        if not bars:
            continue
        out[sym] = {"first": bars[0].timestamp, "last": bars[-1].timestamp, "n": len(bars)}
    return out


def probe(symbols: list[str], group: str, client, timeframe, start: datetime,
          end: datetime) -> pd.DataFrame:
    """Sembolleri BATCH'lik gruplar halinde sorgula; rate-limit'e nazik davran."""
    rows = []
    for i in range(0, len(symbols), BATCH):
        chunk = [_alpaca_sym(s) for s in symbols[i:i + BATCH]]
        try:
            got = _bars_batch(client, chunk, timeframe, start, end)
        except Exception as e:   # tek batch hatası tüm sondayı düşürmesin
            print(f"  [{group}] batch {i//BATCH}: HATA {type(e).__name__}: {e}")
            got = {}
        for s in chunk:
            r = got.get(s)
            rows.append({
                "symbol": s, "group": group,
                "covered": r is not None,
                "first_bar": (r["first"].date().isoformat() if r else ""),
                "last_bar": (r["last"].date().isoformat() if r else ""),
                "n_bars": (r["n"] if r else 0),
            })
        print(f"  [{group}] {min(i+BATCH, len(symbols))}/{len(symbols)} sorgulandı "
              f"(kapsanan: {sum(1 for x in rows if x['group']==group and x['covered'])})")
        time.sleep(0.5)   # 200 istek/dk limitinin çok altında kal
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame, group: str) -> None:
    g = df[df["group"] == group]
    if g.empty:
        print(f"\n--- {group}: sembol yok, atlandı")
        return
    cov = g[g["covered"]]
    print(f"\n--- {group}: {len(cov)}/{len(g)} kapsanıyor (%{100*len(cov)/len(g):.1f})")
    if not cov.empty:
        firsts = pd.to_datetime(cov["first_bar"])
        print(f"    ilk bar medyanı : {firsts.median().date()}  "
              f"(en erken {firsts.min().date()}, en geç {firsts.max().date()})")
        print(f"    bar/sembol medyan: {int(cov['n_bars'].median())}")
        early = (firsts <= pd.Timestamp("2016-06-30")).sum()
        print(f"    2016H1 ve öncesinden başlayan: {early}/{len(cov)}")
    missing = g[~g["covered"]]["symbol"].tolist()
    if missing:
        show = ", ".join(missing[:15]) + (" …" if len(missing) > 15 else "")
        print(f"    VERİSİ OLMAYAN ({len(missing)}): {show}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpaca tarihsel kapsama sondası")
    ap.add_argument("--as-of", default="2018-01-01",
                    help="PIT üyelik tarihi; removed_since bu tarihten beri çıkarılanlar")
    ap.add_argument("--hourly-sample", type=int, default=30,
                    help="saatlik derinlik ölçülecek sembol sayısı (yarısı delist)")
    ap.add_argument("--skip-removed", action="store_true",
                    help="delist sondasını atla (Wikipedia erişimi yoksa)")
    args = ap.parse_args()

    from alpaca.data.timeframe import TimeFrame

    client = _client()
    watch = watchlist_mod.current()
    print(f"=== ALPACA KAPSAMA SONDASI | watchlist: {len(watch)} sembol | "
          f"pencere: {PROBE_START.date()} → {PROBE_END.date()}")

    frames = [probe(watch, "watchlist", client, TimeFrame.Day, PROBE_START, PROBE_END)]

    removed: list[str] = []
    if not args.skip_removed:
        try:
            from nihai.data.pit_universe import removed_since
            removed = removed_since(args.as_of)
            print(f"\nPIT removed-since({args.as_of}): {len(removed)} delist/çıkarılmış isim")
            frames.append(probe(removed, "removed_since", client, TimeFrame.Day,
                                PROBE_START, PROBE_END))
        except Exception as e:
            print(f"UYARI: removed_since alınamadı ({type(e).__name__}: {e}) — "
                  "delist sondası atlandı; fizibilite kararı için TEKRAR dene.")

    # Saatlik derinlik örneklemi: yarısı watchlist, yarısı delist (varsa).
    k = max(args.hourly_sample // 2, 1)
    hourly_syms = watch[:k] + removed[:k]
    if hourly_syms:
        frames.append(probe(hourly_syms, "hourly_sample", client, TimeFrame.Hour,
                            HOURLY_START, PROBE_END))

    df = pd.concat(frames, ignore_index=True)
    out = DATA_DIR / "alpaca_coverage_probe.csv"
    df.to_csv(out, index=False)

    print(f"\n{'='*66}\nÖZET\n{'='*66}")
    for grp in ("watchlist", "removed_since", "hourly_sample"):
        _summary(df, grp)
    print(f"\n→ Detay: {out}")
    print("\nFizibilite kriteri (ACI DERS #1): removed_since kapsaması yüksek değilse")
    print("Alpaca dataset'i de survivorship-şişkin olur — o isimler evrende TUTULMALI.")


if __name__ == "__main__":
    main()
