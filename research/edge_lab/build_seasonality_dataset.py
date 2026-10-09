"""
build_seasonality_dataset.py — BACKLOG #7: MEVSİMSELLİK ailesi (Tur-55).

Yön kuralı **Heston-Sadka 2008 (JFE, "Seasonality in the Cross-Section of Stock
Returns")**: bir hissenin GEÇMİŞ yıllarda AYNI TAKVİM AYINDA verdiği ortalama getiri,
bu yılın aynı ayındaki getirisini kesitsel olarak öngörür. Kesitsel medyanın ÜSTÜ → LONG,
ALTI → SHORT.

**Neden "ay sonu / Ocak etkisi" DEĞİL.** Takvim etkilerinin çoğu (ay dönümü, Ocak,
tatil-öncesi) PİYASA GENELİdir: o günlerde bütün hisseler aynı yöne gider. Bu harness
piyasa-nötr FAZLALIK üstünden puanlar (§6 beta tuzağı kontrolü), yani piyasa-geneli bir
takvim etkisi tanımı gereği SIFIRA gider ve ölçülemez. Heston-Sadka etkisi ise
KESİTSELDİR (hangi hisse hangi ayda) ve bu çerçevede ölçülebilir. Aile seçimi bu yüzden
literatür değil, ÖLÇÜLEBİLİRLİK gerekçesiyle daraltıldı — ve bu daraltma sonucun
kapsamıdır: *"takvim ailesi ölçüldü"* değil, *"takvimin KESİTSEL kolu ölçüldü"*.

TASARIM KİLİTLERİ

* **Look-ahead yok.** Karar günü D = ayın SON işlem günü; `f_seas` yalnız D'den ÖNCEKİ
  yılların aynı takvim ayı getirilerinden hesaplanır (bu yılın hedef ayı ASLA girmez).
* **feature-lag 1 (ACI DERS #7).** Giriş D+1 (yeni ayın ilk barı) kapanışıdır.
* **PIT/survivorship-temiz evren** (`_1d_all`).
* **Örtüşmesizlik** yapısı gereği: sembol başına ayda bir satır, ufuk 21 bar.
* **TARAMA YOK (ön-kayıt):** geçmiş pencere en fazla 5 yıl, en az 2 gözlem; eşik kesitsel
  MEDYAN; ufuk 21 bar. Çoklu-test riski bu ailede EN YÜKSEKTİR (BACKLOG'un kendi notu),
  bu yüzden hiçbir varyant denenmedi.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from nihai.console import ensure_utf8_stdout
from research.edge_lab.panel import PRICE_GLOB, load_panel, market_forward, puanla

OUT_CSV = "data/edge_lab_seasonality_v1.csv"
GECMIS_YIL = 5                  # aynı takvim ayı için en fazla kaç yıl geriye bakılır
MIN_GOZLEM = 2                  # en az kaç geçmiş yıl olmalı

# Aile YAPISI gereği pencere içinde MIN_GOZLEM (=2) yıllık aynı-ay geçmişi ister; panel
# `since` ile kırpıldığı için o geçmiş de pencere İÇİNDEN gelir → ilk karar ayı ancak
# `since + 2 yıl` civarında doğar (ÖLÇÜLDÜ: 2023-02, istenen 2021-01). Bu bir veri arızası
# değil, ailenin cevap veremeyeceği dönemin BEYANIDIR.
ISINMA_GUN = 730

FEATURE_COLS = [
    "f_seas",       # geçmiş yılların AYNI takvim ayındaki ortalama getirisi
    "f_seas_pct",   # o karar günündeki kesitsel yüzdelik dilim (0-1)
    "f_is_short",   # 1 = medyanın altı (SHORT bacağı)
    "f_seas_n",     # kaç geçmiş yıl kullanıldı (2-5)
    "f_ay",         # hedef takvim ayı (1-12)
    "f_ret_21",     # bağlam: son ayın getirisi (kısa vadeli tersine dönüş kontrolü)
    "f_ret_252",    # 12 aylık momentum bağlamı
    "f_vol_60",     # gerçekleşen oynaklık
    "f_dvol_z",     # log dolar hacim z-skoru
]
SCHEMA_COLS = ["ts", "symbol", *FEATURE_COLS, "fwd_ret", "mkt_ret", "excess",
               "pnl", "label"]


def _ay_sonu_gunleri(gunler: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Her takvim ayının SON işlem günü (karar günü D)."""
    s = pd.Series(gunler, index=gunler)
    return pd.DatetimeIndex(s.groupby(gunler.to_period("M")).max().values)


def sembol_satirlari(sym: str, px: pd.DataFrame, karar_gunleri: pd.DatetimeIndex,
                     horizon: int) -> list[dict]:
    """Karar günü = ay sonu; özellik yalnız GEÇMİŞ yılların aynı ayından."""
    p = px.set_index("date")
    close = p["close"].astype(float)
    r = close.pct_change()

    # aylık getiri serisi (ay sonundan ay sonuna)
    ay_kapanis = close.groupby(close.index.to_period("M")).last()
    ay_ret = ay_kapanis.pct_change()

    vol60 = r.rolling(60, min_periods=40).std()
    ldv = np.log1p(close * p["volume"].astype(float))
    dvz = (ldv - ldv.rolling(60, min_periods=40).mean()) / \
        ldv.rolling(60, min_periods=40).std().replace(0.0, np.nan)
    ret21 = close.pct_change(21)
    ret252 = close.pct_change(252)
    fwd = close.shift(-horizon) / close - 1.0
    sira = pd.Series(np.arange(len(close)), index=close.index)

    out: list[dict] = []
    for d in karar_gunleri:
        if d not in close.index:
            continue
        i = int(sira.loc[d])
        if i + 1 >= len(close):
            continue
        j = i + 1                                    # feature-lag 1: yeni ayın ilk barı
        hedef = close.index[j].to_period("M")        # girilen (hedef) ay
        # GEÇMİŞ yılların aynı takvim ayı — hedef ay ASLA dâhil değil
        gecmis = [ay_ret.get(pd.Period(year=hedef.year - k, month=hedef.month, freq="M"),
                             np.nan) for k in range(1, GECMIS_YIL + 1)]
        gecmis = [g for g in gecmis if np.isfinite(g)]
        if len(gecmis) < MIN_GOZLEM:
            continue
        deger = [fwd.iloc[j], vol60.iloc[i], dvz.iloc[i], ret21.iloc[i], ret252.iloc[i]]
        if not all(np.isfinite(v) for v in deger):
            continue
        out.append({
            "ts": close.index[j], "symbol": sym,
            "karar_gunu": d,
            "f_seas": float(np.mean(gecmis)),
            "f_seas_n": float(len(gecmis)),
            "f_ay": float(hedef.month),
            "f_ret_21": float(ret21.iloc[i]),
            "f_ret_252": float(ret252.iloc[i]),
            "f_vol_60": float(vol60.iloc[i]),
            "f_dvol_z": float(dvz.iloc[i]),
            "fwd_ret": float(fwd.iloc[j]),
        })
    return out


def build(horizon: int, cost_pct: float, borrow_bps_yr: float, since: str, until: str,
          out_csv: str) -> pd.DataFrame:
    panel = load_panel(since, until)
    if not panel:
        raise SystemExit(f"fiyat paneli boş — {PRICE_GLOB} okunamadı")
    gunler = pd.DatetimeIndex(sorted({d for x in panel.values() for d in x["date"]}))
    karar = _ay_sonu_gunleri(gunler)
    logger.info(f"[seasonality] evren: {len(panel)} sembol, {len(karar)} karar günü "
                f"({karar.min().date()} → {karar.max().date()})")

    rows: list[dict] = []
    for sym, px in panel.items():
        rows.extend(sembol_satirlari(sym, px, karar, horizon))

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("[seasonality] SIFIR satır")
        return df

    # Kesitsel yüzdelik: AYNI karar gününün kesitinden (gelecekten değil)
    df["f_seas_pct"] = df.groupby("karar_gunu")["f_seas"].rank(pct=True)
    df["f_is_short"] = (df["f_seas_pct"] <= 0.5).astype(float)   # düşük mevsimsellik = SHORT

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
    logger.info(f"[seasonality] YAZILDI {out_csv}: {len(df):,} satır / "
                f"{df['symbol'].nunique()} sembol ({df['ts'].min().date()} → "
                f"{df['ts'].max().date()}); short bacak %{100 * kisa.mean():.1f}")
    logger.info(f"[seasonality] taban: ort pnl {100 * df['pnl'].mean():+.4f}%, "
                f"kazanan {df['label'].mean():.1%} (TRİVİYAL kural; model bunu GEÇMELİ)")
    logger.info(f"[seasonality] BETA KONTROLÜ: piyasa {100 * df['mkt_ret'].mean():+.3f}% → "
                f"YÜKSEK-mevsimsellik (long) bacak fazlalığı "
                f"{100 * df.loc[~kisa, 'excess'].mean():+.4f}% / DÜŞÜK (short) bacak "
                f"{100 * df.loc[kisa, 'excess'].mean():+.4f}% "
                f"(literatür: long bacakta YÜKSEK olmalı)")
    return df


def kapilardan_gecir(out_csv: str, name: str = "seasonality_v1", *,
                     since: str, until: str) -> dict:
    """Beş kanıt kapısı. Bu script KENAR İDDİASI YAPMAZ; hüküm kapılarındır."""
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
    ap = argparse.ArgumentParser(description="Mevsimsellik veri seti (BACKLOG #7)")
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--cost", type=float, default=0.1)
    ap.add_argument("--borrow-bps-yr", type=float, default=100.0)
    ap.add_argument("--since", default="2021-01-01")
    ap.add_argument("--until", default="2026-06-30")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--kapi", action="store_true")
    args = ap.parse_args()
    df = build(args.horizon, args.cost, args.borrow_bps_yr, args.since, args.until, args.out)
    if args.kapi and not df.empty:
        kapilardan_gecir(args.out, since=args.since, until=args.until)
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()
    raise SystemExit(main())
