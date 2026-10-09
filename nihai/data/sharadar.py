"""
sharadar.py — Sharadar (Nasdaq Data Link) point-in-time, survivorship'siz veri adaptörü.

NEDEN AYRI KATMAN: Sharadar canlı quote/opsiyon vermez → `data_provider.py`'nin canlı
DataProvider arayüzüne UYMAZ. Onun verdiği şey EDGE_FINDINGS'in tek dürüst sonraki adımı:
  - SHARADAR/TICKERS : her ticker'ın firstpricedate/lastpricedate + delisting durumu
    → bir tarihte HAYATTA olan evren (sonradan çıkarılanlar DAHİL = survivorship'siz).
  - SHARADAR/SEP     : günlük OHLCV + closeadj (split/temettü düzeltilmiş) — çıkarılan
    isimlerin verisini de içerir (yfinance içermez → §13.5'teki deflasyonun sebebi).

DÜRÜST ÇERÇEVE: Bu modül "kenar" üretmez; sadece backtest'i survivorship-temiz yapar.
Mevcut momentum/PEAD ölçümleri (xsect/pead_test) güncel-üyelik evreninde şişkindi
(EDGE_FINDINGS §13.5/§16); bu adaptör onları GERÇEK point-in-time evrende yeniden
ölçmeyi mümkün kılar.

KURULUM (kullanıcı, key gelince):
    pip install nasdaq-data-link
    # .env:  NASDAQ_DATA_LINK_API_KEY=...   (Sharadar SEP+SF1 aboneliği gerekir,
    #        data.nasdaq.com → Sharadar Equity Bundle, ~$30-50/ay)

Anahtar/paket yoksa: net RuntimeError (çökme değil) — çağıran taraf yakalayıp kullanıcıya
kurulum yönergesini gösterir. Hiçbir import zamanı yan etkisi yok.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import pandas as pd
from loguru import logger

# Sharadar scalemarketcap etiketleri (TICKERS tablosundaki güncel sınıflandırma).
# NOT: bu GÜNCEL bir snapshot — tam point-in-time market-cap değil. Small-cap evrenini
# yaklaşık seçmek için yeterli; kesin PIT market-cap SHARADAR/DAILY (marketcap-by-date)
# ister (ileride rafine edilebilir, aşağıda not).
_SMALLCAP_SCALES = ("1 - Nano", "2 - Micro", "3 - Small")
_SETUP_HINT = (
    "Sharadar erişimi yok. Kurulum:\n"
    "  1) pip install nasdaq-data-link\n"
    "  2) .env'e: NASDAQ_DATA_LINK_API_KEY=<key>\n"
    "  3) data.nasdaq.com → Sharadar Equity Bundle aboneliği (SEP+TICKERS+SF1)\n"
    "Anahtar gelince bu modül ve research/smallcap_momentum.py tek komutla çalışır."
)


class SharadarUnavailable(RuntimeError):
    """nasdaq-data-link paketi veya API anahtarı yok — çağıran yakalayıp yönerge gösterir."""


def _client():
    """nasdaqdatalink modülünü lazy yükle + anahtarı env'den ayarla. Yoksa SharadarUnavailable."""
    key = os.getenv("NASDAQ_DATA_LINK_API_KEY") or os.getenv("QUANDL_API_KEY")
    if not key:
        raise SharadarUnavailable("NASDAQ_DATA_LINK_API_KEY tanımlı değil.\n" + _SETUP_HINT)
    try:
        import nasdaqdatalink
    except ImportError as e:
        raise SharadarUnavailable("nasdaq-data-link paketi kurulu değil.\n" + _SETUP_HINT) from e
    nasdaqdatalink.ApiConfig.api_key = key
    return nasdaqdatalink


@lru_cache(maxsize=1)
def tickers_table() -> pd.DataFrame:
    """SHARADAR/TICKERS (yalnız SEP tablosu = ABD hisseleri). lru_cache: tek indirme.

    Kolonlar (ilgili): ticker, name, isdelisted, category, scalemarketcap,
    firstpricedate, lastpricedate, exchange, sector, industry.
    """
    ndl = _client()
    logger.info("[sharadar] TICKERS indiriliyor (tek sefer, cache'lenir)...")
    df = ndl.get_table("SHARADAR/TICKERS", table="SEP", paginate=True)
    # Tarih kolonlarını datetime'a çevir (boşlar NaT → hayatta-olma kontrolünde elenir)
    for col in ("firstpricedate", "lastpricedate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    logger.info(f"[sharadar] TICKERS: {len(df)} sembol (delisted dahil)")
    return df


def universe_as_of(
    as_of: str,
    *,
    smallcap_only: bool = True,
    scales: tuple[str, ...] = _SMALLCAP_SCALES,
    limit: int | None = None,
) -> list[str]:
    """`as_of` (YYYY-MM-DD) tarihinde AKTİF işlem gören evren — survivorship'siz.

    Bir ticker as_of'ta hayattadır ⇔ firstpricedate ≤ as_of ≤ lastpricedate.
    Bu, o tarihten SONRA çıkarılan/iflas eden isimleri DAHİL eder (survivorship düzeltmesi).

    smallcap_only=True → scalemarketcap ∈ scales (varsayılan Nano/Micro/Small). UYARI:
    scalemarketcap GÜNCEL snapshot, tam PIT değil; small-cap'i yaklaşık seçer.
    """
    df = tickers_table()
    cutoff = pd.Timestamp(as_of)
    alive = (df["firstpricedate"] <= cutoff) & (
        df["lastpricedate"].isna() | (df["lastpricedate"] >= cutoff)
    )
    sub = df[alive]
    if smallcap_only and "scalemarketcap" in sub.columns:
        sub = sub[sub["scalemarketcap"].isin(scales)]
    syms = sorted(str(t).strip().upper() for t in sub["ticker"].dropna() if str(t).strip())
    if limit:
        syms = syms[:limit]
    n_delisted = int(sub["isdelisted"].astype(str).str.upper().eq("Y").sum()) if "isdelisted" in sub.columns else 0
    logger.info(
        f"[sharadar] evren @ {as_of}: {len(syms)} sembol "
        f"({n_delisted} bugün delisted — survivorship düzeltmesi)"
        + (f", smallcap={scales}" if smallcap_only else "")
    )
    return syms


def price_panel(
    tickers: list[str],
    from_date: str,
    to_date: str,
    *,
    field: str = "closeadj",
    chunk: int = 100,
) -> pd.DataFrame:
    """SHARADAR/SEP günlük fiyatlarını ortak tarih indeksli panele topla (xsect formatı).

    Dönüş: index=tarih, columns=ticker, değer=`field` (varsayılan closeadj = düzeltilmiş
    kapanış). `run_momentum`/`build_panel` çıktısıyla aynı şekil → doğrudan beslenebilir.
    Çıkarılan isimler de gelir (varsa lastpricedate'e kadar). İstek boyutu için chunk'lar.
    """
    ndl = _client()
    cols = ["ticker", "date", field]
    frames: list[pd.DataFrame] = []
    uniq = sorted(set(t.strip().upper() for t in tickers if t.strip()))
    for i in range(0, len(uniq), chunk):
        batch = uniq[i : i + chunk]
        try:
            part = ndl.get_table(
                "SHARADAR/SEP",
                ticker=batch,
                date={"gte": from_date, "lte": to_date},
                qopts={"columns": cols},
                paginate=True,
            )
            if not part.empty:
                frames.append(part)
        except Exception as e:
            logger.warning(f"[sharadar] SEP batch {i//chunk} hata: {e}")
        logger.info(f"[sharadar] SEP {min(i+chunk, len(uniq))}/{len(uniq)} sembol")
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    panel = raw.pivot_table(index="date", columns="ticker", values=field, aggfunc="last")
    return panel.sort_index()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sharadar PIT evren/fiyat hızlı testi")
    ap.add_argument("--as-of", default="2018-01-01")
    ap.add_argument("--smallcap", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    try:
        syms = universe_as_of(args.as_of, smallcap_only=args.smallcap, limit=args.limit)
        print(f"\nSharadar evren @ {args.as_of} (ilk {len(syms)}): {', '.join(syms)}")
    except SharadarUnavailable as e:
        print(e)
        sys.exit(0)
