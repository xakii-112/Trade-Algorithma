"""
build_babmacro_dataset.py — BACKLOG #10: DÜŞÜK BETA (BAB) × MAKRO REJİM (Tur-55).

Yön kuralı **Frazzini-Pedersen 2014 (JFE, "Betting Against Beta")**: kaldıraç kısıtlı
yatırımcılar yüksek betayı fiyat olarak pahalılaştırır; düşük betalı hisseler
RİSK-AYARLI olarak yüksek betalıları geçer. Kesitsel medyanın ALTINDAKİ beta → LONG,
ÜSTÜNDEKİ → SHORT.

Makro rejim (dış öneri listesinin *"rejim tespiti — bunu özellikle ciddiye al"* maddesi)
burada YÖN üretmez, ÖZELLİK olarak girer: vade spreadi (T10Y2Y) ve finansal koşullar
endeksi (NFCI) ile seviyeleri/değişimleri. Rejimi yön üreticisi yapmak, dilimleye
dilimleye bir sonuç aramak olurdu; özellik yapmak kapıya *"model rejime bakınca
ayırt edebiliyor mu?"* sorusunu sordurur.

**BETA-AYARLI FAZLALIK — bu ailenin ŞART KOŞTUĞU ölçüt.** Diğer ailelerde `excess =
fwd_ret − mkt_ret`tir. BAB'da bu ÖLÇÜT YANLIŞ olurdu: betası 0,7 olan bir hisse yükselen
piyasada tanımı gereği piyasanın altında kalır, yani düz fazlalık long bacağı yapısal
olarak cezalandırır. Literatürün iddiası ALFA'dır, o yüzden burada
`excess = fwd_ret − beta × mkt_ret`. Düz fazlalık da HESAPLANIR ve raporlanır
(`excess_duz`) — ölçütün seçimi gizlenmez.

TASARIM KİLİTLERİ

* **feature-lag 1 (ACI DERS #7):** beta ve makro durum D barına kadarki veriden; giriş D+1.
* **Look-ahead yok:** beta 120 barlık GERİYE-BAKAN pencere; kesitsel medyan O GÜNÜN
  kesitinden (gelecekten değil); makro seri yalnız o güne kadar bilinen değerle
  (NFCI haftalıktır → en fazla 10 iş günü nedensel ffill, sınırsız DEĞİL).
* **PIT/survivorship-temiz evren** (`_1d_all`).
* **Örtüşmesizlik:** sembol içinde ≥ ufuk (21 bar — FP aylık yeniden dengeler).
* **TARAMA YOK (ön-kayıt):** beta penceresi 120, ufuk 21, eşik = kesitsel MEDYAN.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from nihai.console import ensure_utf8_stdout
from nihai.data import fred_data
from research.edge_lab.panel import PRICE_GLOB, load_panel, market_forward, puanla

OUT_CSV = "data/edge_lab_babmacro_v1.csv"
BETA_WINDOW = 120
BETA_MIN = 80
MACRO_FFILL_LIMIT = 10          # NFCI haftalıktır; SINIRLI nedensel doldurma (Tur-51)
MACRO_SERIES = ["T10Y2Y", "NFCI"]

# ÖLÇÜLDÜ (Tur-63): ilk satır 2021-08-23'te doğuyor = panel başlangıcından 160 İŞLEM günü
# sonra, BETA_MIN'in İKİ katı. Sebep BİLEŞİK ısınmadır: `f_ivol` artık serisinin rolling
# std'sidir ve o artık serisi ZATEN beta ısınmasını (80 bar) bekler → 80+80. Naif beyan
# (tek pencere) gerçeği 4 ay eksik gösterirdi. 160 işlem günü ≈ 224 takvim günü.
ISINMA_GUN = 224

FEATURE_COLS = [
    "f_beta",        # 120 barlık geriye-bakan beta (eşit-ağırlık evrene karşı)
    "f_beta_pct",    # o GÜNÜN kesitindeki beta yüzdelik dilimi (0-1)
    "f_is_short",    # 1 = yüksek beta bacağı (kesitsel medyanın üstü)
    "f_ivol",        # 120 barlık artık (idiyosinkratik) oynaklık
    "f_term",        # T10Y2Y seviyesi
    "f_term_chg",    # T10Y2Y 20 günlük değişim
    "f_nfci",        # NFCI seviyesi (>0 = sıkı finansal koşullar)
    "f_nfci_chg",    # NFCI 20 günlük değişim
    "f_ret_21",
    "f_dvol_z",
]
SCHEMA_COLS = ["ts", "symbol", *FEATURE_COLS, "fwd_ret", "mkt_ret", "excess",
               "excess_duz", "pnl", "label"]


def makro_gunluk(gunler: pd.DatetimeIndex, since: str) -> pd.DataFrame:
    """FRED serilerini işlem günü ızgarasına taşı (SINIRLI nedensel ffill)."""
    ham = fred_data.get_all(MACRO_SERIES, since=since)
    if ham.empty:
        return pd.DataFrame(index=gunler)
    # Tur-63: kırpılma ölçümü artık KAYNAKTAN gelir (`get_all` attrs) — burada elle
    # yeniden hesaplamak, aynı kuralın iki yerde yaşamasıydı (Tur-28 dersi).
    kirpik = list(ham.attrs.get("kirpik") or [])
    if kirpik:
        logger.warning(f"[babmacro] KIRPIK seri(ler): {kirpik} — o dönemde ölçüm yok")
    d = ham.reindex(ham.index.union(gunler)).ffill(limit=MACRO_FFILL_LIMIT).reindex(gunler)
    for c in MACRO_SERIES:
        if c in d.columns:
            d[f"{c}_chg"] = d[c] - d[c].shift(20)
    return d


def rows_for(sym: str, px: pd.DataFrame, mkt_ret: pd.Series, makro: pd.DataFrame,
             beta_pct: pd.DataFrame, horizon: int) -> list[dict]:
    """`beta_pct`: gün × sembol kesitsel beta yüzdelikleri (o günün kesitinden)."""
    p = px.set_index("date")
    close = p["close"].astype(float)
    r = close.pct_change()
    m = mkt_ret.reindex(close.index)

    cov = r.rolling(BETA_WINDOW, min_periods=BETA_MIN).cov(m)
    var = m.rolling(BETA_WINDOW, min_periods=BETA_MIN).var()
    beta = cov / var.replace(0.0, np.nan)
    ivol = (r - beta * m).rolling(BETA_WINDOW, min_periods=BETA_MIN).std()

    ldv = np.log1p(close * p["volume"].astype(float))
    dvz = (ldv - ldv.rolling(60, min_periods=40).mean()) / \
        ldv.rolling(60, min_periods=40).std().replace(0.0, np.nan)
    ret21 = close.pct_change(21)
    fwd = close.shift(-horizon) / close - 1.0
    pct = beta_pct[sym].reindex(close.index) if sym in beta_pct.columns else None
    mk = makro.reindex(close.index)

    out: list[dict] = []
    son_idx: int | None = None
    for i in range(len(close) - 1):
        if pct is None or not np.isfinite(pct.iloc[i]):
            continue
        j = i + 1                                   # feature-lag 1
        if son_idx is not None and j - son_idx < horizon:
            continue
        deger = [fwd.iloc[j], beta.iloc[i], ivol.iloc[i], dvz.iloc[i], ret21.iloc[i],
                 mk["T10Y2Y"].iloc[i], mk["T10Y2Y_chg"].iloc[i],
                 mk["NFCI"].iloc[i], mk["NFCI_chg"].iloc[i]]
        if not all(np.isfinite(v) for v in deger):
            continue
        out.append({
            "ts": close.index[j], "symbol": sym,
            "f_beta": float(beta.iloc[i]),
            "f_beta_pct": float(pct.iloc[i]),
            "f_is_short": 1.0 if pct.iloc[i] > 0.5 else 0.0,   # yüksek beta = SHORT
            "f_ivol": float(ivol.iloc[i]),
            "f_term": float(mk["T10Y2Y"].iloc[i]),
            "f_term_chg": float(mk["T10Y2Y_chg"].iloc[i]),
            "f_nfci": float(mk["NFCI"].iloc[i]),
            "f_nfci_chg": float(mk["NFCI_chg"].iloc[i]),
            "f_ret_21": float(ret21.iloc[i]),
            "f_dvol_z": float(dvz.iloc[i]),
            "fwd_ret": float(fwd.iloc[j]),
            "_beta_giris": float(beta.iloc[i]),
        })
        son_idx = j
    return out


def build(horizon: int, cost_pct: float, borrow_bps_yr: float, since: str, until: str,
          out_csv: str) -> pd.DataFrame:
    panel = load_panel(since, until)
    if not panel:
        raise SystemExit(f"fiyat paneli boş — {PRICE_GLOB} okunamadı")
    wide = pd.DataFrame({s: d.set_index("date")["close"].astype(float)
                         for s, d in panel.items()}).sort_index()
    gunler = pd.DatetimeIndex(wide.index)
    logger.info(f"[babmacro] evren: {len(panel)} sembol, {len(gunler)} işlem günü")

    mkt_ret = wide.pct_change().mean(axis=1)        # eşit-ağırlık evren getirisi
    makro = makro_gunluk(gunler, since)
    if makro.empty or "NFCI" not in makro.columns:
        raise SystemExit("FRED makro serileri alınamadı — ÖN-KOŞUL sağlanmadı")

    # Kesitsel beta yüzdelikleri: O GÜNÜN kesitinden (gelecekten değil)
    rets = wide.pct_change()
    cov = rets.rolling(BETA_WINDOW, min_periods=BETA_MIN).cov(mkt_ret)
    var = mkt_ret.rolling(BETA_WINDOW, min_periods=BETA_MIN).var()
    betas = cov.div(var.replace(0.0, np.nan), axis=0)
    beta_pct = betas.rank(axis=1, pct=True)
    logger.info(f"[babmacro] beta paneli: {int(betas.notna().sum().sum()):,} gözlem")

    rows: list[dict] = []
    for sym, px in panel.items():
        rows.extend(rows_for(sym, px, mkt_ret, makro, beta_pct, horizon))

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("[babmacro] SIFIR satır")
        return df
    df = df.sort_values(["ts", "symbol"], ignore_index=True)
    mkt = market_forward(panel, horizon)
    df["mkt_ret"] = df["ts"].map(mkt)
    df["excess_duz"] = df["fwd_ret"] - df["mkt_ret"]
    df["excess"] = df["fwd_ret"] - df["_beta_giris"] * df["mkt_ret"]   # BETA-AYARLI
    df = df.dropna(subset=["mkt_ret"]).reset_index(drop=True)
    df = puanla(df, horizon, cost_pct, borrow_bps_yr)
    df = df.reindex(columns=SCHEMA_COLS)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    kisa = df["f_is_short"] == 1.0
    logger.info(f"[babmacro] YAZILDI {out_csv}: {len(df):,} satır / "
                f"{df['symbol'].nunique()} sembol ({df['ts'].min().date()} → "
                f"{df['ts'].max().date()}); short (yüksek beta) bacak %{100 * kisa.mean():.1f}")
    logger.info(f"[babmacro] taban: ort pnl {100 * df['pnl'].mean():+.4f}%, "
                f"kazanan {df['label'].mean():.1%} (TRİVİYAL kural; model bunu GEÇMELİ)")
    logger.info(f"[babmacro] ÖLÇÜT: beta-ayarlı fazlalık düşük-beta bacakta "
                f"{100 * df.loc[~kisa, 'excess'].mean():+.4f}% / yüksek-beta bacakta "
                f"{100 * df.loc[kisa, 'excess'].mean():+.4f}% "
                f"(literatür: düşük betada YÜKSEK olmalı) | DÜZ fazlalık aynı bacaklarda "
                f"{100 * df.loc[~kisa, 'excess_duz'].mean():+.4f}% / "
                f"{100 * df.loc[kisa, 'excess_duz'].mean():+.4f}%")
    return df


def kapilardan_gecir(out_csv: str, name: str = "babmacro_v1", *,
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
    ap = argparse.ArgumentParser(description="BAB x makro rejim veri seti (BACKLOG #10)")
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
