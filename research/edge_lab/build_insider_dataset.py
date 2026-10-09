"""
build_insider_dataset.py — BACKLOG #5: Form 4 açık-piyasa ALIM adayı (Tur-52).

Fikir (Lakonishok-Lee 2001; Cohen-Malloy-Pomorski 2012): şirket içindekilerin AÇIK
PİYASADAN kendi parasıyla hisse alması bilgilidir; özellikle "fırsatçı" (rutin olmayan)
ve KÜME hâlindeki alımlar. Satış bilgisiz kabul edilir (ücret/RSU likidasyonu).

TASARIM KİLİTLERİ

* **FILING tarihi, transaction tarihi DEĞİL.** Form 4 işlemden 2 iş günü sonrasına kadar
  dosyalanabilir; transaction tarihine hizalamak, o gün BİLİNMEYEN bilgiyle işlem açmaktır.
* **Kod "P" (açık piyasa alımı) + edinim yönü "A".** "A" kodu (ödül/hibe), "M" (opsiyon
  icrası), "S" (satış) ayıklanır — hibe almak bir kanaat beyanı değildir.
* **feature-lag 1 (ACI DERS #7).** Filing günü akşam yayımlanmış olabilir; giriş SONRAKİ
  barda varsayılır. "İlk bar etkisi" beş ailede aday öldürdü, varsayılan şüphedir.
* **PIT/survivorship-TEMİZ evren.** Fiyatlar `data/alpaca_hist/*_1d_all.csv` — bugün
  endekste olmayan ~90 isim DAHİL.
* **Örtüşmesizlik.** Sembol başına ardışık olaylar `horizon` bardan yakınsa ilki alınır.

VERİ KAYNAĞI — neden per-filing EDGAR DEĞİL. İlk sürüm sembol başına `submissions.json`
+ HER Form 4 için ayrı XML çekiyordu: ÖLÇÜLDÜ, **~140 sn/sembol** (AAPL tek başına 621
dosyalama) → tam evren ~20 saat, dolayısıyla ancak ÖRNEKLENMİŞ bir pilot mümkündü.
SEC'in **çeyreklik toplu Form 345 setleri** aynı alanları (`TRANS_CODE`, `TRANS_SHARES`,
`TRANS_PRICEPERSHARE`, `FILING_DATE`, `ISSUERTRADINGSYMBOL`) ~14 MB'lık tek ZIP'te verir:
22 çeyrek = 22 istek, dakikalar, ve **örneklem değil TAM evren**. Örnekleme yapmak
zorunda kalmamak bir hız kazancı değil bir KANIT kazancıdır — 140 sembollük bir pilot
`gate_symbol` bölmesini kurmakta zaten zorlanırdı.

AMENDMANLAR (`4/A`) DIŞARIDA: bir düzeltme aynı işlemi YENİDEN bildirir; ikisini birden
saymak alım değerini çift sayardı.
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from nihai import net
from nihai.console import ensure_utf8_stdout

PRICE_GLOB = "data/alpaca_hist/*_1d_all.csv"
CACHE_DIR = Path("data/sec_form345_cache")
BULK_URL = ("https://www.sec.gov/files/structureddata/data/"
            "insider-transactions-data-sets/{q}_form345.zip")
OUT_CSV = "data/edge_lab_insider_v1.csv"
TRADING_DAYS = 252

FEATURE_COLS = [
    "f_kume",          # o gün ALIM içeren ayrı dosyalama sayısı (cluster ölçütü)
    "f_log_deger",     # log1p(alım $ değeri)
    "f_deger_dv",      # alım değeri / 20g ortalama dolar hacim (göreli büyüklük)
    "f_gun_onceki",    # önceki alım olayından bu yana geçen işlem günü
    "f_ret_5",         # son 5 barın getirisi (bağlam)
    "f_ret_21",
    "f_zirve_uzak",    # 252g en yüksekten uzaklık (0 = zirvede)
]
SCHEMA_COLS = ["ts", "symbol", *FEATURE_COLS, "fwd_ret", "mkt_ret", "excess",
               "pnl", "label"]


# ---------------------------------------------------------------------------------------
# SEC toplu Form 345 — çeyrek başına tek ZIP
# ---------------------------------------------------------------------------------------
def quarters(since: str, until: str) -> list[str]:
    """`since`..`until` arasını kapsayan `YYYYqN` etiketleri."""
    rng = pd.period_range(pd.Timestamp(since), pd.Timestamp(until), freq="Q")
    return [f"{p.year}q{p.quarter}" for p in rng]


def _fetch_quarter(q: str) -> pd.DataFrame:
    """Bir çeyreğin AÇIK PİYASA ALIMLARI → (symbol, filing_date, buy_value, n_filings).

    Çeyrek başına yalnız SÜZÜLMÜŞ tablo diske yazılır (ham ZIP ~14 MB, süzülmüş ~%1).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{q}_buys.csv"
    if cache.exists():
        d = pd.read_csv(cache, parse_dates=["filing_date"])
        return d

    req = urllib.request.Request(BULK_URL.format(q=q),
                                 headers={"User-Agent": net.contact_ua("NihaiAlgoritma research")})
    with urllib.request.urlopen(req, timeout=120) as r:      # noqa: S310 (sabit SEC ana bilgisayarı)
        z = zipfile.ZipFile(io.BytesIO(r.read()))

    sub = pd.read_csv(z.open("SUBMISSION.tsv"), sep="\t", dtype=str,
                      usecols=["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE",
                               "ISSUERTRADINGSYMBOL"])
    sub = sub[sub["DOCUMENT_TYPE"] == "4"]                    # düzeltmeler (4/A) HARİÇ
    trx = pd.read_csv(z.open("NONDERIV_TRANS.tsv"), sep="\t", dtype=str,
                      usecols=["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES",
                               "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
    trx = trx[(trx["TRANS_CODE"] == "P") & (trx["TRANS_ACQUIRED_DISP_CD"] == "A")]

    df = trx.merge(sub, on="ACCESSION_NUMBER", how="inner")
    df["shares"] = pd.to_numeric(df["TRANS_SHARES"], errors="coerce")
    df["price"] = pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce")
    df["symbol"] = df["ISSUERTRADINGSYMBOL"].str.strip().str.upper()
    df["filing_date"] = pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y",
                                       errors="coerce")
    df = df.dropna(subset=["shares", "price", "symbol", "filing_date"])
    df = df[(df["shares"] > 0) & (df["price"] > 0)]
    df["deger"] = df["shares"] * df["price"]

    out = (df.groupby(["symbol", "filing_date"])
             .agg(buy_value=("deger", "sum"),
                  n_filings=("ACCESSION_NUMBER", "nunique"))
             .reset_index())
    out.to_csv(cache, index=False)
    logger.info(f"[insider] {q}: {len(out):,} sembol-gün alımı, "
                f"{out['symbol'].nunique():,} sembol")
    return out


def load_buys(since: str, until: str) -> pd.DataFrame:
    """Çeyrekleri topla. YAYIMLANMAMIŞ çeyrek koşuyu durdurmaz ama SUSMAZ da.

    SEC toplu setleri gecikmeli yayımlanır; istenen aralığın son çeyreği henüz yoksa
    veri seti SESSİZCE kısalırdı — Tur-51'in 20 aylık deliğinin ta kendisi. Eksik çeyrek
    UYARI olarak loglanır ve gerçekleşen kapsam çağırana `attrs` ile taşınır.
    """
    parts, eksik = [], []
    for q in quarters(since, until):
        try:
            parts.append(_fetch_quarter(q))
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            eksik.append(q)
            logger.warning(f"[insider] {q} ALINAMADI ({type(e).__name__}) — "
                           f"kapsam bu çeyreği İÇERMİYOR")
    if not parts:
        raise SystemExit("hiçbir çeyrek alınamadı")
    if eksik:
        logger.warning(f"[insider] EKSİK ÇEYREK(LER): {', '.join(eksik)} — "
                       f"sonuç bu dönemler hakkında hüküm vermez")
    df = pd.concat(parts, ignore_index=True)
    # Çeyrek sınırında aynı sembol-gün iki dosyada görünebilir → topla.
    out = (df.groupby(["symbol", "filing_date"], as_index=False)
             .agg({"buy_value": "sum", "n_filings": "sum"}))
    # Uyarı LOGA yazılır ama log bir SÖZLEŞME değildir; eksik kapsam makine-okunur
    # taşınmazsa çağıran onu denetleyemez (Tur-38: yazılıp okunmayan alan kusurdur).
    out.attrs["eksik_ceyrekler"] = eksik
    return out


# ---------------------------------------------------------------------------------------
def _load_price(sym: str) -> pd.DataFrame | None:
    try:
        px = pd.read_csv(f"data/alpaca_hist/{sym}_1d_all.csv",
                         usecols=["timestamp", "close", "volume"])
    except (OSError, ValueError):
        return None
    px["date"] = pd.to_datetime(px["timestamp"], errors="coerce").dt.normalize()
    px = px.dropna(subset=["date", "close"]).sort_values("date", ignore_index=True)
    return px if len(px) > 300 else None


def _rows_for(sym: str, ev: pd.DataFrame, px: pd.DataFrame,
              horizon: int, cost_pct: float) -> list[dict]:
    """`ev`: filing_date indeksli (buy_value, n_filings) tablosu."""
    p = px.set_index("date")
    close = p["close"].astype(float)
    dv = (close * p["volume"].astype(float)).rolling(20).mean()
    ret5 = close.pct_change(5)
    ret21 = close.pct_change(21)
    zirve = close / close.rolling(TRADING_DAYS, min_periods=60).max() - 1.0
    fwd = close.shift(-horizon) / close - 1.0
    sira = pd.Series(np.arange(len(close)), index=close.index)

    out: list[dict] = []
    son_idx: int | None = None
    for ts, row in ev.iterrows():
        # feature-lag 1: filing günü akşam yayımlanmış olabilir → SONRAKİ bar
        loc = int(close.index.searchsorted(ts, side="right"))
        if loc >= len(close):
            continue
        d = close.index[loc]
        i = int(sira.loc[d])
        if son_idx is not None and i - son_idx < horizon:      # örtüşmesizlik
            continue
        if not (np.isfinite(fwd.loc[d]) and np.isfinite(dv.loc[d]) and dv.loc[d] > 0):
            continue
        if not (np.isfinite(ret21.loc[d]) and np.isfinite(zirve.loc[d])):
            continue
        deger = float(row["buy_value"])
        out.append({
            "ts": d, "symbol": sym,
            "f_kume": float(row["n_filings"]),
            "f_log_deger": float(np.log1p(max(deger, 0.0))),
            "f_deger_dv": float(deger / float(dv.loc[d])),
            "f_gun_onceki": float(min(i - son_idx, 999)) if son_idx is not None else 999.0,
            "f_ret_5": float(ret5.loc[d]) if np.isfinite(ret5.loc[d]) else 0.0,
            "f_ret_21": float(ret21.loc[d]),
            "f_zirve_uzak": float(zirve.loc[d]),
            "fwd_ret": float(fwd.loc[d]),
        })
        son_idx = i
    for r in out:
        r["pnl"] = r["fwd_ret"] - cost_pct / 100.0
        r["label"] = int(r["pnl"] > 0)
    return out


def market_forward(horizon: int) -> pd.Series:
    """AYNI evrenin, her gün için eşit-ağırlık `horizon`-bar ileri getirisi.

    **Neden veri setinde, bir yan script'te değil.** Bir olay-ailesinin ham ileri
    getirisi bir kenar iddiası DEĞİLDİR: 2021-2025 boğa piyasasında her olay ailesi
    pozitif görünür (§6 beta tuzağı). Bu tur `xsect`te aynı kusuru bulup düzeltti
    (al-tut kontrolü sessizce 0,0 dönüyordu) — kontrolü çağırana bırakmak, onu
    unutulabilir kılmaktır. Kolon şemada olduğu için her okuyucu FAZLALIĞI görür.
    """
    ser = {}
    for f in glob.glob(PRICE_GLOB):
        sym = os.path.basename(f).split("_1d_all")[0]
        d = pd.read_csv(f, usecols=["timestamp", "close"])
        d["date"] = pd.to_datetime(d["timestamp"], errors="coerce").dt.normalize()
        d = d.dropna(subset=["date", "close"])
        if len(d) > 300:
            ser[sym] = d.set_index("date")["close"].astype(float)
    panel = pd.DataFrame(ser).sort_index()
    return (panel.shift(-horizon) / panel - 1.0).mean(axis=1)


def build(horizon: int, cost_pct: float, since: str, until: str,
          out_csv: str) -> pd.DataFrame:
    syms = sorted({os.path.basename(f).split("_1d_all")[0] for f in glob.glob(PRICE_GLOB)})
    logger.info(f"[insider] evren: {len(syms)} sembol (PIT yerel), {since} → {until}")

    buys = load_buys(since, until)
    logger.info(f"[insider] SEC toplu: {len(buys):,} sembol-gün alımı / "
                f"{buys['symbol'].nunique():,} sembol (TÜM piyasa)")
    buys = buys[buys["symbol"].isin(syms)]
    logger.info(f"[insider] evrenle kesişim: {len(buys):,} sembol-gün / "
                f"{buys['symbol'].nunique():,} sembol")

    rows: list[dict] = []
    for sym, g in buys.groupby("symbol"):
        px = _load_price(str(sym))
        if px is None:
            continue
        ev = g.set_index("filing_date")[["buy_value", "n_filings"]].sort_index()
        rows.extend(_rows_for(str(sym), ev, px, horizon, cost_pct))

    df = pd.DataFrame(rows).sort_values(["ts", "symbol"], ignore_index=True)
    if not df.empty:
        mkt = market_forward(horizon)
        df["mkt_ret"] = df["ts"].map(mkt)
        df["excess"] = df["fwd_ret"] - df["mkt_ret"]
    df = df.reindex(columns=SCHEMA_COLS)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if df.empty:
        logger.warning("[insider] SIFIR satır — evrende açık-piyasa alımı bulunamadı")
        return df
    logger.info(f"[insider] YAZILDI {out_csv}: {len(df):,} satır / "
                f"{df['symbol'].nunique()} sembol ({df['ts'].min().date()} → "
                f"{df['ts'].max().date()})")
    logger.info(f"[insider] taban: ort pnl {100 * df['pnl'].mean():+.3f}%, "
                f"kazanan {df['label'].mean():.1%} "
                f"(= 'her alımı al' TRİVİYAL kuralı; model bunu GEÇMELİ)")
    logger.info(f"[insider] BETA KONTROLÜ: piyasa aynı pencerede "
                f"{100 * df['mkt_ret'].mean():+.3f}% → FAZLALIK "
                f"{100 * df['excess'].mean():+.3f}% "
                f"(kazanan {100 * (df['excess'] > 0).mean():.1f}%) — "
                f"ham getiri bir kenar iddiası DEĞİLDİR (§6)")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Form 4 acik-piyasa alim veri seti (BACKLOG #5)")
    ap.add_argument("--horizon", type=int, default=21, help="ileri getiri ufku (islem gunu)")
    ap.add_argument("--cost", type=float, default=0.1, help="round-trip maliyet (yuzde)")
    ap.add_argument("--since", default="2021-01-01", help="SEC ceyrek alt siniri")
    ap.add_argument("--until", default="2026-06-30", help="SEC ceyrek ust siniri")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()
    build(args.horizon, args.cost, args.since, args.until, args.out)
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()
    raise SystemExit(main())
