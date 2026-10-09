"""
build_vixts_dataset.py — VIX VADE YAPISI × SWING KIRILMASI (Tur-74, dış liste Madde 9).

Madde 9 şunu iddia ediyor: *"Contango (VIX < VIX3M) boğa swinglerine izin verirken,
Backwardation (VIX > VIX3M) TÜM swing alımlarını kilitler."* Bu bir KENAR iddiasıdır ve
bu depoda ölçülmemiş hiçbir bileşen karara giremez (ACI DERS #5). Tur-74 vade yapısını
KARAR-DIŞI rapor olarak bağladı; burada iddianın KENDİSİ kanıt kapısına sokulur.

TASARIM — HEPSİ ÖN-KAYITLI, TARAMA YOK
--------------------------------------
* **Taban kural:** 20 barlık en yükseğin YUKARI kırılması = klasik swing ALIMI. Madde 9'un
  kilitlemek istediği şey budur; kural `nihai.signals`ten DEĞİL buradan gelir çünkü
  `edge_lab` üretim sinyaline import bağı ALMAZ (paket docstring'i).
* **Rejim YÖN ÜRETMEZ, ÖZELLİK olarak girer** (`build_babmacro_dataset` ile aynı disiplin).
  Rejimi yön üreticisi yapmak, dilimleye dilimleye sonuç aramak olurdu; özellik yapmak
  kapıya *"model rejime bakınca ayırt edebiliyor mu?"* sorusunu sordurur.
* **Ufuk 5 bar** (swing), maliyet %0,1, feature-lag 1 (D barına kadarki veri, giriş D+1),
  sembol içinde örtüşmezlik ≥ ufuk.
* **PIT/survivorship-temiz evren** (`*_1d_all`, ACI DERS #1).

ÖLÇÜT NEDEN **HAM** GETİRİ (fazlalık DEĞİL) — bu ailenin ŞART KOŞTUĞU seçim
--------------------------------------------------------------------------
Diğer ailelerde ölçüt `excess = fwd_ret − mkt_ret`tir. Burada bu ÖLÇÜT YANLIŞ olurdu:
Madde 9'un iddiası bir PİYASA ZAMANLAMASI iddiasıdır (*"backwardation'da alma"*), yani
tam olarak piyasanın TOPLU hareketini konu alır. Piyasaya göre fazlalık almak, ölçülmek
istenen etkiyi TANIM GEREĞİ görünmez kılardı.

Boğa-piyasası artefaktı (`panel.market_forward` docstring'i: *"2021-2026 boğasında her
LONG ailesi kendiliğinden pozitif görünür"*) bu ailede FARKLILAŞTIRMAYLA değil
KIYAS TABANIYLA nötrlenir: `harness`ın kapı simülasyonu ve DSR aşaması adayı
**"kapıyı hiç uygulama, her kırılmayı al"** triviyal kuralına karşı ölçer (Tur-31
onarımı) — sürüklenme İKİ kolda da vardır. Şeffaflık için piyasaya göre fazlalık
`excess_duz` kolonunda AYRICA taşınır.

⚠️ BAĞIMSIZLIK BİRİMİ **GÜN**DÜR, SATIR DEĞİL (Tur-53 / BULGU-53-1)
------------------------------------------------------------------
Vade yapısı özellikleri PİYASA GENELİDİR: aynı gün kırılan 40 sembol AYNI `f_ts_slope`
değerini taşır. Dolayısıyla satır sayısı etkin örneklem büyüklüğü DEĞİLDİR ve
`harness`ın SEMBOL bölmesi bu özellikler üzerinde yapısal olarak ayırt edici değildir
(sembol-düzeyi kontrol kolonları ona içerik verir). Bu script bu yüzden hem SATIR hem
GÜN düzeyi tabanı basar; hükmü okurken gün sayısına bakılmalıdır.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from nihai.console import ensure_utf8_stdout
from nihai.data import cboe_data
from research.edge_lab.panel import PRICE_GLOB, load_panel, market_forward, puanla

OUT_CSV = "data/edge_lab_vixts_v1.csv"

BREAKOUT_WINDOW = 20            # swing kırılması penceresi (ÖN-KAYITLI, taranmadı)
DVOL_WINDOW = 60                # dolar-hacim z-skoru penceresi
ISINMA_GUN = 90                 # 60 işlem günü ≈ 90 takvim günü (dvol z-skoru en uzunu)

FEATURE_COLS = [
    # --- piyasa geneli: VADE YAPISI (Madde 9'un konusu) ---
    "f_ts_slope",       # VIX3M/VIX  >1 contango · <1 backwardation
    "f_backwardation",  # 1 = Madde 9'un "kilitle" dediği rejim
    "f_near_slope",     # VIX/VIX9D  ön uç dikliği
    "f_far_slope",      # VIX6M/VIX  uzun uç (Madde 9 bunu AYRICA istiyordu)
    "f_ts_chg20",       # ts_slope 20 günlük değişim (rejim YÖNÜ, seviyesi değil)
    "f_vix_pctile",     # VIX genişleyen-pencere yüzdeliği (look-ahead YOK)
    "f_vvix",
    "f_skew",
    # --- sembol düzeyi kontroller (kesitsel içerik) ---
    "f_ret_21",
    "f_dvol_z",
    "f_atr_pct",
    "f_ext_pct",        # kırılmanın 20-bar en yükseğini ne kadar aştığı
]
SCHEMA_COLS = ["ts", "symbol", "f_is_short", *FEATURE_COLS,
               "fwd_ret", "mkt_ret", "excess", "excess_duz", "pnl", "label"]


def vade_yapisi_gunluk(gunler: pd.DatetimeIndex) -> pd.DataFrame:
    """CBOE vade yapısını işlem günü ızgarasına taşı.

    **ffill YOK.** Eksik bir VIX günü, o gün rejimin ÖLÇÜLMEDİĞİ anlamına gelir ve o
    günün kırılmaları veri setine GİRMEZ. Son bilinen rejimi ileri taşımak, Tur-51'in
    düzensiz-seri dersini tekrarlamak olurdu (sınırsız ffill bir haftalık eski IV'yi
    "bugünkü" diye sunuyordu).
    """
    ts = cboe_data.term_structure(fetch=False)
    if ts.empty:
        return pd.DataFrame(index=gunler)
    ts = ts.copy()
    ts["ts_chg20"] = ts["ts_slope"] - ts["ts_slope"].shift(20)
    d = ts.reindex(gunler)
    kapsama = {c: float(d[c].notna().mean()) for c in
               ("ts_slope", "near_slope", "far_slope", "vix_pctile") if c in d.columns}
    logger.info("[vixts] vade yapısı kapsaması: "
                + " · ".join(f"{k} %{100 * v:.1f}" for k, v in kapsama.items()))
    return d


def rows_for(sym: str, px: pd.DataFrame, vy: pd.DataFrame, horizon: int) -> list[dict]:
    """Bir sembolün 20-bar kırılma girişleri (feature-lag 1, örtüşmesiz)."""
    p = px.set_index("date")
    close = p["close"].astype(float)
    if len(close) < DVOL_WINDOW + BREAKOUT_WINDOW + horizon + 2:
        return []
    vol = p["volume"].astype(float)

    # 20-bar en yüksek: KENDİ barını DIŞLAR (shift(1)) — aksi halde kırılma tanımı
    # kendi kendine referans verir ve her bar "kırılma" olurdu.
    hi20 = close.rolling(BREAKOUT_WINDOW).max().shift(1)
    kirilma = close > hi20
    ext = (close / hi20 - 1.0)

    r = close.pct_change()
    atr_pct = r.rolling(14).std()
    ldv = np.log1p(close * vol)
    dvz = ((ldv - ldv.rolling(DVOL_WINDOW, min_periods=40).mean())
           / ldv.rolling(DVOL_WINDOW, min_periods=40).std().replace(0.0, np.nan))
    ret21 = close.pct_change(21)
    fwd = close.shift(-horizon) / close - 1.0
    v = vy.reindex(close.index)

    out: list[dict] = []
    son_idx: int | None = None
    for i in range(len(close) - 1):
        if not bool(kirilma.iloc[i]):
            continue
        j = i + 1                                   # feature-lag 1: giriş ERTESİ bar
        if son_idx is not None and j - son_idx < horizon:
            continue                                # örtüşmezlik
        satir = {
            "f_ts_slope": v["ts_slope"].iloc[i] if "ts_slope" in v else np.nan,
            "f_near_slope": v["near_slope"].iloc[i] if "near_slope" in v else np.nan,
            "f_far_slope": v["far_slope"].iloc[i] if "far_slope" in v else np.nan,
            "f_ts_chg20": v["ts_chg20"].iloc[i] if "ts_chg20" in v else np.nan,
            "f_vix_pctile": v["vix_pctile"].iloc[i] if "vix_pctile" in v else np.nan,
            "f_vvix": v["vvix_ratio"].iloc[i] if "vvix_ratio" in v else np.nan,
            "f_skew": v["skew"].iloc[i] if "skew" in v else np.nan,
            "f_ret_21": ret21.iloc[i], "f_dvol_z": dvz.iloc[i],
            "f_atr_pct": atr_pct.iloc[i], "f_ext_pct": ext.iloc[i],
        }
        if not all(np.isfinite(x) for x in satir.values()) or not np.isfinite(fwd.iloc[j]):
            continue
        satir["f_backwardation"] = 1.0 if satir["f_ts_slope"] < 1.0 else 0.0
        out.append({"ts": close.index[j], "symbol": sym, "f_is_short": 0.0,
                    **{k: float(x) for k, x in satir.items()},
                    "fwd_ret": float(fwd.iloc[j])})
        son_idx = j
    return out


def build(horizon: int, cost_pct: float, since: str, until: str, out_csv: str) -> pd.DataFrame:
    panel = load_panel(since, until)
    if not panel:
        raise SystemExit(f"fiyat paneli boş — {PRICE_GLOB} okunamadı")
    wide = pd.DataFrame({s: d.set_index("date")["close"].astype(float)
                         for s, d in panel.items()}).sort_index()
    gunler = pd.DatetimeIndex(wide.index)
    logger.info(f"[vixts] evren: {len(panel)} sembol, {len(gunler)} işlem günü")

    vy = vade_yapisi_gunluk(gunler)
    if vy.empty or "ts_slope" not in vy.columns:
        raise SystemExit("CBOE vade yapısı alınamadı — ÖN-KOŞUL sağlanmadı "
                         "(önbelleği doldur: python -m nihai.data.cboe_data)")

    rows: list[dict] = []
    for sym, px in panel.items():
        rows.extend(rows_for(sym, px, vy, horizon))

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("[vixts] SIFIR satır")
        return df
    df = df.sort_values(["ts", "symbol"], ignore_index=True)
    mkt = market_forward(panel, horizon)
    df["mkt_ret"] = df["ts"].map(mkt)
    df["excess_duz"] = df["fwd_ret"] - df["mkt_ret"]      # şeffaflık: piyasaya göre
    df["excess"] = df["fwd_ret"]                          # ÖLÇÜT: HAM (bkz. modül docstring)
    df = df.dropna(subset=["mkt_ret"]).reset_index(drop=True)
    df = puanla(df, horizon, cost_pct, borrow_bps_yr=0.0)   # long-only → ödünç maliyeti YOK
    df = df.reindex(columns=SCHEMA_COLS)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    bw = df["f_backwardation"] == 1.0
    n_gun = df["ts"].nunique()
    logger.info(f"[vixts] YAZILDI {out_csv}: {len(df):,} satır / {df['symbol'].nunique()} "
                f"sembol / {n_gun} GÜN ({df['ts'].min().date()} → {df['ts'].max().date()})")
    logger.info(f"[vixts] taban (TRİVİYAL kural = her kırılmayı al): ort pnl "
                f"{100 * df['pnl'].mean():+.4f}%, kazanan {df['label'].mean():.1%}")
    # Madde 9'un iddiasının DOĞRUDAN okuması — satır VE gün düzeyinde (Tur-53)
    gun = df.groupby("ts").agg(pnl=("pnl", "mean"), bw=("f_backwardation", "max"))
    logger.info(f"[vixts] MADDE 9 İDDİASI — contango ↔ backwardation ort pnl: "
                f"SATIR {100 * df.loc[~bw, 'pnl'].mean():+.4f}% ↔ "
                f"{100 * df.loc[bw, 'pnl'].mean():+.4f}% (n={int((~bw).sum()):,}/{int(bw.sum()):,}) | "
                f"GÜN {100 * gun.loc[gun['bw'] == 0, 'pnl'].mean():+.4f}% ↔ "
                f"{100 * gun.loc[gun['bw'] == 1, 'pnl'].mean():+.4f}% "
                f"(n={int((gun['bw'] == 0).sum())}/{int((gun['bw'] == 1).sum())} gün)")
    return df


def kapilardan_gecir(out_csv: str, name: str = "vixts_v1", *,
                     since: str, until: str) -> dict:
    """Kanıt kapıları. Bu script KENAR İDDİASI YAPMAZ; hüküm kapılarındır."""
    from research.edge_lab.harness import run_candidate

    v = run_candidate(name=name, dataset_csv=out_csv, feature_cols=FEATURE_COLS,
                      label_col="label", ts_col="ts", symbol_col="symbol", pnl_col="pnl",
                      expected_start=since, expected_end=until, warmup_days=ISINMA_GUN)
    print(f"\n=== {v['candidate']} — {v['n_rows']:,} satır ===")
    for g in v["gates"]:
        print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['name']:<14} {g['detail']}")
    print(f"  DSR={v.get('dsr')}  exec_lag_etkisi={v.get('exec_lag_effect')}")
    print(f"VERDİKT: {v['verdict']}")
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description="VIX vade yapısı × swing kırılması (Madde 9)")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--cost", type=float, default=0.1)
    ap.add_argument("--since", default="2021-01-01")
    ap.add_argument("--until", default="2026-06-30")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--kapi", action="store_true")
    args = ap.parse_args()
    df = build(args.horizon, args.cost, args.since, args.until, args.out)
    if args.kapi and not df.empty:
        kapilardan_gecir(args.out, since=args.since, until=args.until)
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()
    raise SystemExit(main())
