"""
build_rmom_dataset.py — BACKLOG #11: KALINTI MOMENTUMU ailesi (Tur-86).

Yön kuralı **Blitz, Huij ve Martens 2011 (J. Empirical Finance 18(3), "Residual
momentum")**: hissenin aylık aşırı getirisi (r − RF) son 36 ayda Fama-French üç faktörü
(Mkt-RF, SMB, HML) üzerine sabitli EKK ile regres edilir; skor, **t−12…t−2** aylarının
kalıntı TOPLAMININ aynı 11 ayın kalıntı std'sine oranıdır (son ay HARİÇ — kısa vadeli
tersine dönüş). Kesitsel medyanın ÜSTÜ → LONG, ALTI → SHORT.

**Neden bu aile.** Defterdeki on ailenin hiçbiri momentum değil. Toplam getiri momentumu
faktör maruziyetini de taşır ve o maruziyet tersine döndüğünde çöker; kalıntı sürümü
faktörleri ayıklayıp yalnız hisseye özgü sürüklenmeyi sıralar. Veri tamamen yerel: PIT
`_1d_all` paneli + Ken French önbelleği (`french_factors(fetch=False)`) → ağ yok, yeni
bağımlılık yok.

TASARIM KİLİTLERİ

* **PIT ÜYELİK (bu ailenin en kritik kilidi).** Panel pencerede SONRADAN eklenen isimleri
  de taşır ve endekse eklenme çoğu zaman yükselişin SONUCUDUR. Süzülmezse geleceğin
  kazananları geçmiş kesite girer ve momentum SAHTE kâr gösterir (evren seçiminde
  look-ahead). Kesit her karar gününde `pit_universe.sp500_uyelik` ile O GÜNKÜ üyelere
  süzülür.
* **Look-ahead yok.** Regresyon ve skor yalnız karar günü D'ye (ayın son işlem günü) KADAR
  olan aylardan; D'nin ayı tam aydır. Faktör serisi bir ay gecikmelidir: D ayının faktörü
  yoksa satır ÜRETİLMEZ (eksik ay sıfırla doldurulmaz).
* **feature-lag 1 (ACI DERS #7).** Giriş D+1 kapanışı, ufuk 21 bar → sembol başına ayda
  bir satır, örtüşme yok.
* **Kıyas tabanı AYNI PIT kesitidir.** `panel.market_forward` tüm paneli (sonradan
  eklenenler dahil) ortalar; burada fazlalık aynı karar gününün kesitinin eşit-ağırlık ileri
  getirisine göre ölçülür.
* **TARAMA YOK (ön-kayıt, BACKLOG #11):** pencere 36 ay (tam 36 gözlem), oluşum t−12…t−2,
  eşik kesitsel MEDYAN, ufuk 21, maliyet %0,1 + ödünç 100 bps/yıl (Tur-55 ile aynı).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from nihai.console import ensure_utf8_stdout
from research.edge_lab.panel import PRICE_GLOB, load_panel, puanla

OUT_CSV = "data/edge_lab_rmom_v1.csv"
PENCERE_AY = 36                 # regresyon penceresi (ay)
OLUSUM_AY = 11                  # t−12…t−2 = pencerenin son 12 ayının İLK 11'i

# Pencere içinde 36 aylık getiri ister; panel 2020-06'da başladığı için ilk aylık getiri
# 2020-07'dir ve ilk karar ayı 2023-06 civarında doğar. Bu bir veri arızası değil, ailenin
# cevap veremeyeceği dönemin BEYANIDIR (Tur-63 kapsam kapısı).
ISINMA_GUN = 1130

FEATURE_COLS = [
    "f_rmom",       # standardize kalıntı momentumu (Blitz-Huij-Martens)
    "f_rmom_pct",   # o karar günündeki kesitsel yüzdelik (0-1)
    "f_is_short",   # 1 = medyanın altı (SHORT bacağı)
    "f_tmom",       # toplam getiri momentumu t−12…t−2 (Jegadeesh-Titman bağlamı)
    "f_beta",       # 36 aylık piyasa betası
    "f_ivol",       # 36 aylık kalıntı std'si (idiosenkratik oynaklık, aylık)
    "f_ret_21",     # son ayın getirisi (kısa vadeli tersine dönüş bağlamı)
    "f_vol_60",     # gerçekleşen oynaklık
    "f_dvol_z",     # log dolar hacim z-skoru
]
SCHEMA_COLS = ["ts", "symbol", *FEATURE_COLS, "fwd_ret", "mkt_ret", "excess",
               "pnl", "label"]


def aylik_faktorler(ff: pd.DataFrame) -> pd.DataFrame:
    """Günlük FF faktörlerini aylık BİLEŞİK orana çevir (Period[M] indeksli).

    Aşırı getirinin (Mkt-RF) günlük bileşimi, aylık Mkt − aylık RF'nin yaklaşığıdır; fark
    RF'nin karesi mertebesindedir ve kesitsel sıralamayı değiştirmez."""
    g = ff[["mkt_rf", "smb", "hml", "rf"]].astype(float)
    return (1.0 + g).groupby(g.index.to_period("M")).prod() - 1.0


def kalinti_skoru(y: np.ndarray, F: np.ndarray) -> tuple[float, float, float] | None:
    """Sabitli EKK (y ~ 1 + F) → (skor, piyasa betası, kalıntı std'si).

    `y` hissenin aylık aşırı getirisi (PENCERE_AY,), `F` faktörler (PENCERE_AY, 3), son
    eleman karar ayı. Oluşum t−12…t−2 = `e[-12:-1]` (son ay HARİÇ). Kalıntı std'si
    sayısal sıfırsa (dejenere seri: faktörlerle TAM açıklanan getiri, ya da işlem görmeyen
    hisse) skor TANIMSIZDIR → None. Eşik 1e-12'dir, 0 değil: tam uyumda EKK kalıntısı
    ~1e-17 mertebesinde kayan nokta gürültüsüdür ve `toplam / std` onu anlamlı bir skor
    gibi basardı (Tur-28 dejenere girdi dersi)."""
    X = np.column_stack([np.ones(len(y)), F])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ coef
    olusum = e[-12:-1]
    sd = float(np.std(olusum, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        return None
    return float(olusum.sum() / sd), float(coef[1]), float(np.std(e, ddof=1))


def _ay_sonu_gunleri(gunler: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Her takvim ayının SON işlem günü (karar günü D)."""
    s = pd.Series(gunler, index=gunler)
    return pd.DatetimeIndex(s.groupby(gunler.to_period("M")).max().values)


def sembol_satirlari(sym: str, px: pd.DataFrame, karar_gunleri: pd.DatetimeIndex,
                     fa: pd.DataFrame, horizon: int,
                     uyeler: dict[pd.Timestamp, set[str]]) -> list[dict]:
    """Karar günü = ay sonu; skor yalnız D'ye kadarki 36 aydan, kesit o günün üyeleri."""
    p = px.set_index("date")
    close = p["close"].astype(float)
    r = close.pct_change()

    ay_kapanis = close.groupby(close.index.to_period("M")).last()
    ay_ret = ay_kapanis.pct_change()
    # Seride ay BOŞLUĞU varsa `pct_change` boşluktan sonraki ayda iki aylık getiriyi tek ay
    # sayar (Tur-51: bir kaydırma, seri düzenli değilse zaman kaydırması değildir) → o ay NaN;
    # onu içeren her 36 aylık pencere aşağıdaki sonluluk kontrolünde düşer.
    donem = ay_kapanis.index.to_series()
    ardisik = (donem - donem.shift(1)).map(lambda x: getattr(x, "n", None) == 1)
    ay_ret = ay_ret.where(ardisik.to_numpy(dtype=bool))
    ortak = ay_ret.index.intersection(fa.index)
    ay_ret = ay_ret.loc[ortak]
    fa_o = fa.loc[ortak]
    fazla = (ay_ret - fa_o["rf"]).to_numpy()
    F = fa_o[["mkt_rf", "smb", "hml"]].to_numpy()
    ham = ay_ret.to_numpy()
    ay_sira = {per: k for k, per in enumerate(ortak)}

    vol60 = r.rolling(60, min_periods=40).std()
    ldv = np.log1p(close * p["volume"].astype(float))
    dvz = (ldv - ldv.rolling(60, min_periods=40).mean()) / \
        ldv.rolling(60, min_periods=40).std().replace(0.0, np.nan)
    ret21 = close.pct_change(21)
    fwd = close.shift(-horizon) / close - 1.0
    sira = pd.Series(np.arange(len(close)), index=close.index)

    out: list[dict] = []
    for d in karar_gunleri:
        if d not in close.index or sym not in uyeler.get(d, ()):
            continue
        k = ay_sira.get(d.to_period("M"))
        if k is None or k + 1 < PENCERE_AY:
            continue
        pencere = slice(k + 1 - PENCERE_AY, k + 1)          # karar ayı DAHİL, 36 ay
        y, Fw = fazla[pencere], F[pencere]
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(Fw))):
            continue
        skor = kalinti_skoru(y, Fw)
        if skor is None:
            continue
        i = int(sira.loc[d])
        j = i + 1                                           # feature-lag 1: D+1 kapanışı
        if j >= len(close):
            continue
        deger = [fwd.iloc[j], vol60.iloc[i], dvz.iloc[i], ret21.iloc[i]]
        if not all(np.isfinite(v) for v in deger):
            continue
        out.append({
            "ts": close.index[j], "symbol": sym, "karar_gunu": d,
            "f_rmom": skor[0], "f_beta": skor[1], "f_ivol": skor[2],
            "f_tmom": float(np.prod(1.0 + ham[pencere][-12:-1]) - 1.0),
            "f_ret_21": float(ret21.iloc[i]),
            "f_vol_60": float(vol60.iloc[i]),
            "f_dvol_z": float(dvz.iloc[i]),
            "fwd_ret": float(fwd.iloc[j]),
        })
    return out


def kesit_fazlaligi(df: pd.DataFrame) -> pd.DataFrame:
    """Kıyas = AYNI karar gününün PIT kesitinin eşit-ağırlık ileri getirisi."""
    df["mkt_ret"] = df.groupby("karar_gunu")["fwd_ret"].transform("mean")
    df["excess"] = df["fwd_ret"] - df["mkt_ret"]
    return df


def build(horizon: int, cost_pct: float, borrow_bps_yr: float, since: str, until: str,
          out_csv: str) -> pd.DataFrame:
    from nihai.data.factors import french_factors
    from nihai.data.pit_universe import sp500_uyelik

    # Pencere KARAR günlerini sınırlar; 21 barlık ETİKET pencerenin sonrasına taşar (etiket
    # tanımı gereği gelecektir — look-ahead değil). Etiket verisi `until`'de kesilseydi son
    # iki ayın etiketi kurulamaz ve beyan edilen pencereyle üretilen pencere ayrışırdı (Tur-63
    # kapsam kapısı bunu Tur-86'nın ilk koşusunda "son kesme %5,5" diye yakaladı).
    etiket_sonu = str((pd.Timestamp(until) + pd.Timedelta(days=2 * horizon + 10)).date())
    panel = load_panel(since, etiket_sonu)
    if not panel:
        raise SystemExit(f"fiyat paneli boş — {PRICE_GLOB} okunamadı")
    ff = french_factors("ff5_daily", fetch=False)
    if ff.empty:
        raise SystemExit("faktör önbelleği boş — önce `python -m nihai.data.factors`")
    fa = aylik_faktorler(ff)

    gunler = pd.DatetimeIndex(sorted({d for x in panel.values() for d in x["date"]}))
    karar = _ay_sonu_gunleri(gunler)
    karar = karar[karar <= pd.Timestamp(until)]
    uyelik = sp500_uyelik([str(d.date()) for d in karar])
    uyeler = {d: set(uyelik[str(d.date())]) for d in karar}
    logger.info(f"[rmom] evren: {len(panel)} sembol, {len(karar)} karar günü "
                f"({karar.min().date()} → {karar.max().date()}); faktör son ay "
                f"{fa.index.max()}")

    rows: list[dict] = []
    for sym, px in panel.items():
        rows.extend(sembol_satirlari(sym, px, karar, fa, horizon, uyeler))

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("[rmom] SIFIR satır")
        return df

    # Kesitsel yüzdelik: AYNI karar gününün (PIT) kesitinden — gelecekten değil
    df["f_rmom_pct"] = df.groupby("karar_gunu")["f_rmom"].rank(pct=True)
    df["f_is_short"] = (df["f_rmom_pct"] <= 0.5).astype(float)
    df = kesit_fazlaligi(df)
    df = df.sort_values(["ts", "symbol"], ignore_index=True)
    iz = kosulsuz_iz(df)
    df = puanla(df, horizon, cost_pct, borrow_bps_yr)
    df = df.reindex(columns=SCHEMA_COLS)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"[rmom] YAZILDI {out_csv}: {len(df):,} satır / "
                f"{df['symbol'].nunique()} sembol ({df['ts'].min().date()} → "
                f"{df['ts'].max().date()}); short bacak %{100 * df['f_is_short'].mean():.1f}")
    logger.info(f"[rmom] taban: ort pnl {100 * df['pnl'].mean():+.4f}%, "
                f"kazanan {df['label'].mean():.1%} (TRİVİYAL kural; model bunu GEÇMELİ)")
    logger.info(f"[rmom] KOŞULSUZ İZ: {iz}")
    return df


def kosulsuz_iz(df: pd.DataFrame) -> dict:
    """Uzun-kısa bacak farkı, SATIR ve GÜN düzeyinde (BULGU-53-1: birim t'nin işaretini
    bile çevirebilir). Maliyet ÖNCESİ fazlalık; kenar iddiası değildir."""
    kisa = df["f_is_short"] == 1.0
    gun = df.groupby(["karar_gunu", kisa])["excess"].mean().unstack()
    fark = (gun[False] - gun[True]).dropna()
    t_gun = (float(fark.mean() / (fark.std(ddof=1) / np.sqrt(len(fark))))
             if len(fark) > 2 and fark.std(ddof=1) > 0 else None)
    uzun_x, kisa_x = df.loc[~kisa, "excess"], df.loc[kisa, "excess"]
    se = float(np.sqrt(uzun_x.var(ddof=1) / len(uzun_x) + kisa_x.var(ddof=1) / len(kisa_x)))
    return {
        "long_fazlalik_pct": round(100 * float(uzun_x.mean()), 4),
        "short_fazlalik_pct": round(100 * float(kisa_x.mean()), 4),
        "fark_pct": round(100 * float(uzun_x.mean() - kisa_x.mean()), 4),
        "t_satir": round(float((uzun_x.mean() - kisa_x.mean()) / se), 3) if se > 0 else None,
        "t_gun": round(t_gun, 3) if t_gun is not None else None,
        "n_gun": int(len(fark)),
        "pozitif_gun_payi": round(float((fark > 0).mean()), 3) if len(fark) else None,
    }


def kapilardan_gecir(out_csv: str, name: str = "rmom_v1", *,
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
    ap = argparse.ArgumentParser(description="Kalıntı momentumu veri seti (BACKLOG #11)")
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--cost", type=float, default=0.1)
    ap.add_argument("--borrow-bps-yr", type=float, default=100.0)
    ap.add_argument("--since", default="2020-06-01")
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
