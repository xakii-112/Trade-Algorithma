"""
build_shortflow_dataset.py — BACKLOG #8: GÜNLÜK AÇIĞA SATIŞ AKIŞI (FINRA) adayı (Tur-53).

Fikir (Boehmer-Jones-Zhang 2008 JF "Which Shorts Are Informed?"; Diether-Lee-Werner 2009
RFS): açığa satanlar ortalamada BİLGİLİDİR. Günlük konsolide tape'te bir hissenin açığa
satış hacmi oranı (SVR = short/total) KENDİ geçmişine göre olağandışı yükseldiğinde
sonraki günlerin getirisi negatife, olağandışı düştüğünde pozitife eğilir. Bu, dış
öneri listesindeki **"trade flow / order-flow imbalance"** maddesinin ABD hissesindeki
ÜCRETSİZ ve GÜNLÜK karşılığıdır (kripto order-book'u yok; bu var).

Önceki beş aile ile YAPISAL olarak farklıdır: olay (kazanç/rebalans/insider) veya fiyat
türevi (lead-lag/VRP) değil, **KONUMLANMA AKIŞI**. Bu yüzden ayrı bir aile sayılır ve
defterin çoklu-test sayacını bir artırır.

TASARIM KİLİTLERİ (hepsi önceki turların bedelini ödediği derslerdir)

* **feature-lag 1 (ACI DERS #7).** FINRA D gününün dosyasını D'nin KAPANIŞINDAN SONRA
  yayımlar. Giriş barı D+1'dir ve giriş fiyatı D+1 KAPANIŞIdır — yayım ile giriş arasında
  tam bir seans vardır. Beş ailede aday öldüren "ilk bar etkisi" böylece yapısal olarak
  dışarıda; `exec_lag_1` kapısı bunun ÜSTÜNE bir bar daha ekler.
* **PIT/survivorship-TEMİZ evren.** `data/alpaca_hist/*_1d_all.csv` — bugün endekste
  olmayan isimler DAHİL (ACI DERS #1).
* **BETA kontrolü ŞEMADA (§6, Tur-52 dersi).** `mkt_ret`/`excess` kolonları veri setinin
  kendisinde; yön piyasa-nötr FAZLALIK üstünden puanlanır. Boğa piyasasında bir LONG
  ailesi kendiliğinden pozitif, bir SHORT ailesi kendiliğinden negatif görünürdü.
* **Örtüşmesizlik.** Sembol içinde ardışık satırlar en az `horizon` bar aralıklıdır
  (bağımsızlık birimi gözlem değildir — Tur-27).
* **EŞİK TARAMASI YOK (ön-kayıt).** Ne olay eşiği (|z|>k) ne ufuk taraması yapıldı: ufuk
  **5 bar** ve evren TÜM uygun barlar olarak koşudan ÖNCE sabitlendi. Tarama yapılsaydı
  DSR deflasyonunun `n_trials`'ı ayrıca çarpılmalıydı (§23 ufuk süpürmesi dersi).
* **AÇIĞA SATIŞ MALİYETİ MODELDE.** Short bacağa yıllık borç maliyeti (varsayılan 100 bps)
  ufka orantılı düşülür. Gerçekte ağır açığa satılan isimler tam da PAHALI ödünç
  olanlardır (D'Avolio 2002) → bu varsayılan İYİMSERDİR ve öyle raporlanır.

VERİ: `https://cdn.finra.org/equity/regsho/daily/CNMSshvol<YYYYMMDD>.txt`,
`Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market`, gün başına ~12k sembol /
~0,5 MB. Anahtar/kayıt GEREKMEZ. Yalnız evrene süzülmüş hâli AY BAZINDA diske yazılır.
"""
from __future__ import annotations

import argparse
import http.client
import io
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from nihai import net
from nihai.console import ensure_utf8_stdout
from research.edge_lab.panel import PRICE_GLOB, load_panel, market_forward, puanla

CACHE_DIR = Path("data/finra_shortvol_cache")
DAILY_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"
OUT_CSV = "data/edge_lab_shortflow_v1.csv"
TRADING_DAYS = 252          # zirve penceresi; puanlamadaki yıllıklandırma panel.py'de
Z_WINDOW = 60          # SVR'nin KENDİ geçmişine göre normalize penceresi (işlem günü)
Z_MIN_PERIODS = 40

# Z_WINDOW/zirve pencereleri 60 işlem günü ≈ 84 takvim günü. FINRA'nın kendi başlangıcı
# bununla ÖRTÜŞÜYOR (ÖLÇÜLDÜ: doğan 2021-03-31, etkin beyan 2021-03-26 → 5 gün).
ISINMA_GUN = 84

FEATURE_COLS = [
    "f_svr",         # ham açığa satış hacim oranı (0-1), karar günü D
    "f_svr_z",       # SVR'nin sembol-içi 60g z-skoru (>0 = olağandışı YOĞUN açığa satış)
    "f_svr_absz",    # |z| — sinyalin ŞİDDETİ (yön f_is_short'ta)
    "f_is_short",    # 1 = short bacak (z>0), 0 = long bacak; bacaklar simetrik DEĞİL
    "f_svr_5",       # 5g ortalama SVR − 60g ortalama (ısrar/persistans)
    "f_dvol_z",      # log dolar hacmin 60g z-skoru (dikkat/aktivite bağlamı)
    "f_ret_5",       # bağlam: son 5 bar getirisi
    "f_ret_21",
    "f_zirve_uzak",  # 252g en yüksekten uzaklık (0 = zirvede)
]
SCHEMA_COLS = ["ts", "symbol", *FEATURE_COLS, "fwd_ret", "mkt_ret", "excess",
               "pnl", "label"]


# ---------------------------------------------------------------------------------------
# FINRA günlük konsolide açığa satış hacmi — gün başına tek istek, AY bazında önbellek
# ---------------------------------------------------------------------------------------
def _fetch_day(gun: pd.Timestamp, evren: set[str], denemeler: int = 3) -> pd.DataFrame | None:
    """Tek günün dosyası → evrene süzülmüş (date, symbol, short_vol, total_vol).

    `IncompleteRead` bir `HTTPException`'dır, `OSError` DEĞİL — ilk sürüm onu yakalamıyordu
    ve CDN'in tek geçici kesintisi 1378 günlük koşuyu 22. ayda öldürdü. Geçici hata SESSİZCE
    "o gün yok" sayılamaz (kapsam sessizce daralır, Tur-51); önce YENİDEN DENENİR.
    """
    url = DAILY_URL.format(d=gun.strftime("%Y%m%d"))
    raw = b""
    for deneme in range(denemeler):
        req = urllib.request.Request(url, headers={"User-Agent": net.contact_ua("NihaiAlgoritma research")})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (sabit FINRA ana bilgisayarı)
                raw = r.read()
            break
        except urllib.error.HTTPError:
            return None                                   # gerçekten yok (403/404) — tekrar etme
        except (urllib.error.URLError, http.client.HTTPException, OSError):
            if deneme == denemeler - 1:
                return None
            time.sleep(1.0 + deneme)
    d = pd.read_csv(io.BytesIO(raw), sep="|", dtype=str)
    if "Symbol" not in d.columns:
        return None
    d["symbol"] = d["Symbol"].str.strip().str.upper()
    d = d[d["symbol"].isin(evren)]
    d["short_vol"] = pd.to_numeric(d["ShortVolume"], errors="coerce")
    d["total_vol"] = pd.to_numeric(d["TotalVolume"], errors="coerce")
    d = d.dropna(subset=["short_vol", "total_vol"])
    d = d[d["total_vol"] > 0]
    d = d.assign(date=gun)
    return d[["date", "symbol", "short_vol", "total_vol"]].reset_index(drop=True)


def _month_cache(ay: pd.Period) -> Path:
    return CACHE_DIR / f"{ay}.csv"


def load_shortvol(gunler: pd.DatetimeIndex, evren: set[str],
                  workers: int = 6) -> pd.DataFrame:
    """`gunler`in TAMAMI için FINRA akışını topla (ay-ay önbellekli, eksik gün SUSMAZ).

    Önbellek YALNIZ istenen günlerin tamamını içeren ay için geçerlidir: yarım ay
    diskteyse ay yeniden çekilir. Tur-51'in 20 aylık sessiz deliği bu yüzden sessiz
    kalabilmişti.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parcalar: list[pd.DataFrame] = []
    eksik_toplam = 0
    for ay, idx in pd.Series(gunler, index=gunler).groupby(gunler.to_period("M")):
        istenen = pd.DatetimeIndex(idx.values)
        cache = _month_cache(ay)
        if cache.exists():
            d = pd.read_csv(cache, parse_dates=["date"])
            if set(istenen) <= set(d["date"].unique()):
                parcalar.append(d)
                continue
        with ThreadPoolExecutor(max_workers=workers) as ex:
            sonuc = list(ex.map(lambda g: _fetch_day(g, evren), istenen))
        alinan = [s for s in sonuc if s is not None and not s.empty]
        eksik = [g for g, s in zip(istenen, sonuc, strict=True) if s is None or s.empty]
        eksik_toplam += len(eksik)
        if eksik:
            logger.warning(f"[shortflow] {ay}: {len(eksik)}/{len(istenen)} gün ALINAMADI "
                           f"— kapsam bu günleri İÇERMİYOR")
        if not alinan:
            continue
        d = pd.concat(alinan, ignore_index=True)
        d.to_csv(cache, index=False)
        logger.info(f"[shortflow] {ay}: {len(d):,} sembol-gün ({d['symbol'].nunique()} sembol)")
        parcalar.append(d)
    if not parcalar:
        raise SystemExit("FINRA akışı alınamadı — hiçbir gün indirilemedi")
    out = pd.concat(parcalar, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out.attrs["eksik_gun"] = eksik_toplam
    return out


# ---------------------------------------------------------------------------------------
# Satır üretimi
# ---------------------------------------------------------------------------------------
def rows_for(sym: str, px: pd.DataFrame, svr_gun: pd.Series, horizon: int) -> list[dict]:
    """`svr_gun`: karar günü (D) tarihine indeksli açığa satış oranı serisi.

    Giriş barı D'den SONRAKİ bardır (feature-lag 1); ileri getiri o barın KAPANIŞINDAN
    ölçülür. Yön: z>0 (olağandışı yoğun açığa satış) → SHORT, z<0 → LONG.
    """
    p = px.set_index("date")
    close = p["close"].astype(float)
    svr = svr_gun.reindex(close.index)
    m60 = svr.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean()
    s60 = svr.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std()
    z = (svr - m60) / s60.replace(0.0, np.nan)
    svr5 = svr.rolling(5).mean() - m60

    ldv = np.log1p(close * p["volume"].astype(float))
    ldv_m = ldv.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean()
    ldv_s = ldv.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std().replace(0.0, np.nan)
    dvz = (ldv - ldv_m) / ldv_s
    ret5 = close.pct_change(5)
    ret21 = close.pct_change(21)
    zirve = close / close.rolling(TRADING_DAYS, min_periods=60).max() - 1.0
    fwd = close.shift(-horizon) / close - 1.0

    out: list[dict] = []
    son_idx: int | None = None
    for i in range(len(close) - 1):
        zi = z.iloc[i]
        if not np.isfinite(zi) or zi == 0.0:
            continue
        j = i + 1                                   # feature-lag 1: giriş SONRAKİ bar
        if son_idx is not None and j - son_idx < horizon:      # örtüşmesizlik
            continue
        deger = [fwd.iloc[j], svr.iloc[i], svr5.iloc[i], dvz.iloc[i],
                 ret5.iloc[i], ret21.iloc[i], zirve.iloc[i]]
        if not all(np.isfinite(v) for v in deger):
            continue
        out.append({
            "ts": close.index[j], "symbol": sym,
            "f_svr": float(svr.iloc[i]),
            "f_svr_z": float(zi),
            "f_svr_absz": float(abs(zi)),
            "f_is_short": 1.0 if zi > 0 else 0.0,
            "f_svr_5": float(svr5.iloc[i]),
            "f_dvol_z": float(dvz.iloc[i]),
            "f_ret_5": float(ret5.iloc[i]),
            "f_ret_21": float(ret21.iloc[i]),
            "f_zirve_uzak": float(zirve.iloc[i]),
            "fwd_ret": float(fwd.iloc[j]),
        })
        son_idx = j
    return out


def build(horizon: int, cost_pct: float, borrow_bps_yr: float, since: str, until: str,
          out_csv: str, workers: int = 6) -> pd.DataFrame:
    panel = load_panel(since, until)
    if not panel:
        raise SystemExit(f"fiyat paneli boş — {PRICE_GLOB} okunamadı")
    gunler = pd.DatetimeIndex(sorted({d for x in panel.values() for d in x["date"]}))
    logger.info(f"[shortflow] evren: {len(panel)} sembol (PIT yerel), "
                f"{len(gunler)} işlem günü {gunler.min().date()} → {gunler.max().date()}")

    akis = load_shortvol(gunler, set(panel), workers=workers)
    logger.info(f"[shortflow] FINRA: {len(akis):,} sembol-gün / "
                f"{akis['symbol'].nunique()} sembol; eksik gün {akis.attrs.get('eksik_gun', 0)}")
    akis["svr"] = akis["short_vol"] / akis["total_vol"]

    rows: list[dict] = []
    for sym, g in akis.groupby("symbol"):
        px = panel.get(str(sym))
        if px is None:
            continue
        ser = g.set_index("date")["svr"].sort_index()
        ser = ser[~ser.index.duplicated(keep="last")]
        rows.extend(rows_for(str(sym), px, ser, horizon))

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("[shortflow] SIFIR satır")
        return df
    df = df.sort_values(["ts", "symbol"], ignore_index=True)
    mkt = market_forward(panel, horizon)
    df["mkt_ret"] = df["ts"].map(mkt)
    df["excess"] = df["fwd_ret"] - df["mkt_ret"]
    df = df.dropna(subset=["mkt_ret"]).reset_index(drop=True)
    df = puanla(df, horizon, cost_pct, borrow_bps_yr)
    df = df.reindex(columns=SCHEMA_COLS)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    kisa = df["f_is_short"] == 1.0
    logger.info(f"[shortflow] YAZILDI {out_csv}: {len(df):,} satır / "
                f"{df['symbol'].nunique()} sembol "
                f"({df['ts'].min().date()} → {df['ts'].max().date()}); "
                f"short bacak %{100 * kisa.mean():.1f}")
    logger.info(f"[shortflow] taban: ort pnl {100 * df['pnl'].mean():+.4f}%, "
                f"kazanan {df['label'].mean():.1%} "
                f"(= 'her satırı al' TRİVİYAL kuralı; model bunu GEÇMELİ)")
    logger.info(f"[shortflow] BETA KONTROLÜ: piyasa {100 * df['mkt_ret'].mean():+.3f}%, "
                f"ham ileri {100 * df['fwd_ret'].mean():+.3f}% → FAZLALIK "
                f"{100 * df['excess'].mean():+.3f}%; "
                f"short bacak fazlalığı {100 * df.loc[kisa, 'excess'].mean():+.4f}% / "
                f"long bacak {100 * df.loc[~kisa, 'excess'].mean():+.4f}% "
                f"(literatür: short bacakta NEGATİF fazlalık bekleniyor)")
    return df


def kapilardan_gecir(out_csv: str, name: str = "shortflow_v1", *,
                     since: str, until: str) -> dict:
    """Veri setini `harness.run_candidate`'ın BEŞ kapısına ver ve verdikti bas.

    Bu script KENAR İDDİASI YAPMAZ — yalnız veri taşır; hüküm kapılarındır.
    """
    from research.edge_lab.harness import run_candidate

    v = run_candidate(name=name, dataset_csv=out_csv, feature_cols=FEATURE_COLS,
                      label_col="label", ts_col="ts", symbol_col="symbol", pnl_col="pnl",
                      expected_start=since, expected_end=until,
                      warmup_days=ISINMA_GUN)
    print(f"\n=== {v['candidate']} — {v['n_rows']:,} satır ===")
    for g in v["gates"]:
        print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['name']:<12} {g['detail']}")
    print(f"  DSR={v.get('dsr')}  exec_lag_etkisi={v.get('exec_lag_effect')}")
    print(f"VERDİKT: {v['verdict']}")
    return v


def main() -> int:
    ap = argparse.ArgumentParser(
        description="FINRA gunluk aciga satis akisi veri seti (BACKLOG #8)")
    ap.add_argument("--kapi", action="store_true",
                    help="veri seti kurulduktan sonra harness kapilarini kostur")
    ap.add_argument("--horizon", type=int, default=5, help="ileri getiri ufku (islem gunu)")
    ap.add_argument("--cost", type=float, default=0.1, help="round-trip maliyet (yuzde)")
    ap.add_argument("--borrow-bps-yr", type=float, default=100.0,
                    help="short bacak yillik odunc maliyeti (bps) — ETB icin iyimser")
    ap.add_argument("--since", default="2021-01-01")
    ap.add_argument("--until", default="2026-06-30")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()
    df = build(args.horizon, args.cost, args.borrow_bps_yr, args.since, args.until,
               args.out, args.workers)
    if args.kapi and not df.empty:
        kapilardan_gecir(args.out, since=args.since, until=args.until)
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()
    raise SystemExit(main())
