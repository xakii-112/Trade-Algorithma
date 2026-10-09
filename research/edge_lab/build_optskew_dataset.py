"""
build_optskew_dataset.py — BACKLOG #9: OPSİYON YÜZEYİNİN YÖN SİNYALİ (Tur-54).

İki klasik bulgu, tek aile:

* **Cremers-Weinbaum 2010 (JFQA)** — put-call PARİTE SAPMASI. Aynı vade+kullanım fiyatında
  call ve put örtük oynaklıkları teoride eşittir; ayrıştıklarında fark (`VS = IV_call −
  IV_put`) bilgili taraf tarafından fiyatlanmış YÖN taşır: VS yüksekken hisse sonraki
  günlerde pozitif, düşükken negatif getiri verir. **Bu ailenin YÖN KURALI budur.**
* **Xing-Zhang-Zhao 2010 (JFQA)** — "smirk": OTM put IV'si ATM call IV'sinin ne kadar
  üstünde (`SKEW`). Dik gülümseme sonraki getiriyi DÜŞÜRÜR. Tek taraflı bir tahmin
  olduğu için burada YÖN değil ÖZELLİK olarak girer.

VRP ailesinden (Tur-51) FARKLIDIR ve ayrı sayılır: orada etiket ileri GERÇEKLEŞEN
OYNAKLIKTI (prim satıcısının sorusu), burada etiket HİSSENİN ileri getirisidir (yön
sorusu). Aynı veri kaynağı iki ayrı soruyu yanıtlıyor.

TASARIM KİLİTLERİ

* **feature-lag 1 (ACI DERS #7).** Zincir gün SONU verisidir → giriş D+1 KAPANIŞI.
* **PIT.** Fiyat evreni `_1d_all` (bugün endekste olmayanlar dâhil); opsiyon tarafında
  Dolt SÜRÜMLÜ olduğu için `--ref` ile commit sabitlenebilir.
* **ÖN-KAYITLI SEÇİM — tarama YOK.** Vade: DTE ∈ [20,60] içinde 30'a EN YAKIN olan.
  ATM call: |Δ−0,50| en küçük (≤0,15 olmalı). OTM put: |Δ+0,20| en küçük (≤0,10).
  VS: aynı kullanım fiyatındaki eşleşen çiftlerin ortalaması, call Δ ∈ [0,20 · 0,80].
  Ufuk 5 bar. Bu sayıların HİÇBİRİ taranmadı; tarama `n_trials`'ı ayrıca çarpardı (§23).
* **KOTASYON KALİTESİ.** `bid > 0` ve `ask > bid` şartı — sıfır-bid opsiyonun IV'si
  ölçüm değil gürültüdür. (`vol` alanı sağlayıcının IV'sidir.)
* **BETA kontrolü ŞEMADA** (`mkt_ret`/`excess`) ve short bacağa ödünç maliyeti
  (`panel.puanla`, Tur-52/53 ile aynı gövde).
* **Örtüşmesizlik:** sembol içinde ardışık satırlar ≥ ufuk.

VERİ: DoltHub `post-no-preference/options` → `option_chain` toplu CSV (~10 GB).
`DOLT_CACHE_DIR` ile OneDrive DIŞINA indirilmelidir; 10 GB'lık bir araştırma önbelleği
senkronlanacak bir şey değildir.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from nihai.console import ensure_utf8_stdout
from research.edge_lab.panel import PRICE_GLOB, load_panel, market_forward, puanla

OUT_CSV = "data/edge_lab_optskew_v1.csv"
CHAIN_COLS = ["date", "act_symbol", "expiration", "strike", "call_put", "bid", "ask",
              "vol", "delta"]
Z_WINDOW = 60
Z_MIN_PERIODS = 40

# --- ön-kayıtlı seçim sabitleri (taranmadı) ---------------------------------------------
DTE_MIN, DTE_MAX, DTE_HEDEF = 20, 60, 30
ATM_DELTA, ATM_TOL = 0.50, 0.15
OTM_PUT_DELTA, OTM_PUT_TOL = -0.20, 0.10
VS_DELTA_LO, VS_DELTA_HI = 0.20, 0.80
VS_MIN_CIFT = 2

# Ailenin KENDİ ısınması yalnız z-penceresidir (Z_MIN_PERIODS=40 işlem günü ≈ 56 takvim
# günü). Veri seti buna rağmen 3,7 yıl GEÇ doğdu (sağlayıcı kapsaması) → kapı FAIL eder ve
# etmelidir: BULGU-54-1'in ta kendisi, artık makinede.
ISINMA_GUN = 56

FEATURE_COLS = [
    "f_vs",        # put-call parite sapması: ortalama (IV_call − IV_put), eşleşen çiftler
    "f_vs_z",      # VS'nin sembol-içi 60g z-skoru  (>0 = call'lar zengin = yukarı yönlü)
    "f_vs_absz",   # |z| — sinyalin şiddeti (yön f_is_short'ta)
    "f_is_short",  # 1 = short bacak (vs_z<0); bacaklar simetrik DEĞİL (ödünç maliyeti)
    "f_vs_chg",    # VS'nin 5 günlük değişimi (Cremers-Weinbaum ikinci ölçüsü)
    "f_skew",      # IV(OTM put) − IV(ATM call) — Xing-Zhang-Zhao smirk
    "f_skew_z",    # smirk'in sembol-içi 60g z-skoru
    "f_iv_atm",    # ATM call IV seviyesi (bağlam)
    "f_ret_5",
    "f_ret_21",
]
SCHEMA_COLS = ["ts", "symbol", *FEATURE_COLS, "fwd_ret", "mkt_ret", "excess",
               "pnl", "label"]


# ---------------------------------------------------------------------------------------
# Zincir → gün×sembol yüzey ölçüleri (akış hâlinde; dosya belleğe SIĞMAZ)
# ---------------------------------------------------------------------------------------
def _yuzey_olc(g: pd.DataFrame) -> dict | None:
    """Tek (gün, sembol) için VS · SKEW · ATM IV. Ölçülemezse None (uydurma YOK)."""
    dte = (g["expiration"] - g["date"]).dt.days
    uygun = g[(dte >= DTE_MIN) & (dte <= DTE_MAX)]
    if uygun.empty:
        return None
    d = (uygun["expiration"] - uygun["date"]).dt.days
    hedef = uygun.loc[(d - DTE_HEDEF).abs().idxmin(), "expiration"]
    v = uygun[uygun["expiration"] == hedef]

    calls = v[v["call_put"] == "Call"]
    puts = v[v["call_put"] == "Put"]
    if calls.empty or puts.empty:
        return None

    # ATM call: delta 0,50'ye en yakın
    i = (calls["delta"] - ATM_DELTA).abs().idxmin()
    if abs(float(calls.loc[i, "delta"]) - ATM_DELTA) > ATM_TOL:
        return None
    iv_atm = float(calls.loc[i, "vol"])

    # OTM put: delta −0,20'ye en yakın
    j = (puts["delta"] - OTM_PUT_DELTA).abs().idxmin()
    if abs(float(puts.loc[j, "delta"]) - OTM_PUT_DELTA) > OTM_PUT_TOL:
        return None
    iv_otm_put = float(puts.loc[j, "vol"])

    # VS: aynı kullanım fiyatında eşleşen call/put çiftleri (yalnız orta bölge)
    c = calls[(calls["delta"] >= VS_DELTA_LO) & (calls["delta"] <= VS_DELTA_HI)]
    m = c.merge(puts, on="strike", suffixes=("_c", "_p"))
    if len(m) < VS_MIN_CIFT:
        return None
    vs = float((m["vol_c"] - m["vol_p"]).mean())

    if not (np.isfinite(vs) and np.isfinite(iv_atm) and np.isfinite(iv_otm_put)
            and iv_atm > 0):
        return None
    return {"date": g["date"].iloc[0], "symbol": g["act_symbol"].iloc[0],
            "vs": vs, "skew": iv_otm_put - iv_atm, "iv_atm": iv_atm,
            "n_cift": int(len(m))}


def yuzey_serisi(chain_csv: str | Path, evren: set[str], since: str, until: str,
                 chunksize: int = 4_000_000) -> pd.DataFrame:
    """Toplu zinciri akıtarak gün×sembol yüzey ölçülerini kur.

    Dosya birincil anahtar sırasındadır (date, act_symbol, …). Bir (gün, sembol) grubu
    parça sınırına denk gelirse YARIM ölçülürdü; bu yüzden her parçanın SON grubu bir
    sonraki parçaya DEVREDİLİR. Sıralama varsayımı kör değildir: çıktıda tekrar eden
    anahtar sayılır ve sıfır değilse UYARI verilir.
    """
    kalan = pd.DataFrame()
    parcalar: list[dict] = []
    okunan = 0
    for chunk in pd.read_csv(chain_csv, usecols=CHAIN_COLS, chunksize=chunksize,
                             dtype={"date": "string", "act_symbol": "string",
                                    "expiration": "string", "call_put": "string"}):
        okunan += len(chunk)
        # StringDtype karşılaştırması pd.NA taşıyabilir ve NA'lı maske ile indeksleme
        # patlar; 10 GB'lık dış bir dosyada tek bozuk satır koşuyu öldürmemeli.
        maske = ((chunk["date"] >= since) & (chunk["date"] <= until)
                 & chunk["act_symbol"].isin(evren)).fillna(False).to_numpy(dtype=bool)
        chunk = chunk[maske]
        chunk = chunk[(chunk["bid"] > 0) & (chunk["ask"] > chunk["bid"])
                      & (chunk["vol"] > 0)]
        if len(kalan):
            chunk = pd.concat([kalan, chunk], ignore_index=True)
        if chunk.empty:
            kalan = pd.DataFrame()
            continue
        anahtar = chunk["date"].astype(str) + "|" + chunk["act_symbol"].astype(str)
        son = anahtar.iloc[-1]
        kalan = chunk[anahtar == son].copy()
        islenecek = chunk[anahtar != son]
        if not islenecek.empty:
            parcalar.extend(_parca_olc(islenecek))
        logger.info(f"[optskew] {okunan:,} satır okundu → {len(parcalar):,} gün×sembol")
    if len(kalan):
        parcalar.extend(_parca_olc(kalan))

    out = pd.DataFrame(parcalar)
    if out.empty:
        return out
    yinelenen = int(out.duplicated(subset=["date", "symbol"]).sum())
    if yinelenen:
        logger.warning(f"[optskew] {yinelenen} YİNELENEN gün×sembol — dosya PK sırasında "
                       f"değil, ölçüler bölünmüş olabilir (ilki tutuldu)")
        out = out.drop_duplicates(subset=["date", "symbol"], keep="first")
    return out.sort_values(["symbol", "date"], ignore_index=True)


def _parca_olc(df: pd.DataFrame) -> list[dict]:
    """`_yuzey_olc`'in VEKTÖRLEŞTİRİLMİŞ hâli — aynı kural, grup döngüsü yok.

    Referans gövde ÖLÇÜLDÜ: 31.911 grup için **378 sn** (grup başına ~12 ms; her grupta
    üç `idxmin` + bir `merge`). Tam dosyada ~675.000 grup ≈ 2,3 saat ederdi. Bu sürüm
    aynı kararları `transform("min")` + tek `merge` ile toplu verir.

    **Denklik iddiası testle kilitlidir** (`test_tur54_...::test_vektor_referansla_AYNI`):
    `_yuzey_olc` üretimde çağrılmaz ama silinmez — bu fonksiyonun ORAKLIDIR. Eşitlik
    bozucuları da aynı seçilir: |dte−30| eşitse KÜÇÜK dte, delta eşitse İLK satır (dosya
    PK sırasındadır, yani küçük kullanım fiyatı).
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], format="%Y-%m-%d", errors="coerce")
    d["expiration"] = pd.to_datetime(d["expiration"], format="%Y-%m-%d", errors="coerce")
    d = d.dropna(subset=["date", "expiration", "delta", "vol", "strike"])
    if d.empty:
        return []
    K = ["date", "act_symbol"]

    d["dte"] = (d["expiration"] - d["date"]).dt.days
    d = d[(d["dte"] >= DTE_MIN) & (d["dte"] <= DTE_MAX)]
    if d.empty:
        return []

    # hedef vade: |dte−30| en küçük; eşitlikte küçük dte
    d["_d30"] = (d["dte"] - DTE_HEDEF).abs()
    d = d[d["_d30"] == d.groupby(K, sort=False)["_d30"].transform("min")]
    d = d[d["dte"] == d.groupby(K, sort=False)["dte"].transform("min")]

    calls = d[d["call_put"] == "Call"]
    puts = d[d["call_put"] == "Put"]
    if calls.empty or puts.empty:
        return []

    def _en_yakin(x: pd.DataFrame, hedef: float, tol: float) -> pd.Series:
        a = (x["delta"] - hedef).abs()
        x = x.assign(_a=a)
        x = x[x["_a"] == x.groupby(K, sort=False)["_a"].transform("min")]
        x = x[x["_a"] <= tol].drop_duplicates(subset=K, keep="first")
        return x.set_index(K)["vol"]

    iv_atm = _en_yakin(calls, ATM_DELTA, ATM_TOL).rename("iv_atm")
    iv_otm = _en_yakin(puts, OTM_PUT_DELTA, OTM_PUT_TOL).rename("iv_otm")
    if iv_atm.empty or iv_otm.empty:
        return []

    orta = calls[(calls["delta"] >= VS_DELTA_LO) & (calls["delta"] <= VS_DELTA_HI)]
    m = orta[[*K, "strike", "vol"]].merge(puts[[*K, "strike", "vol"]],
                                          on=[*K, "strike"], suffixes=("_c", "_p"))
    if m.empty:
        return []
    agg = (m.assign(_f=m["vol_c"] - m["vol_p"])
             .groupby(K, sort=False)["_f"].agg(vs="mean", n_cift="size"))
    agg = agg[agg["n_cift"] >= VS_MIN_CIFT]

    out = agg.join(iv_atm, how="inner").join(iv_otm, how="inner")
    out = out[np.isfinite(out["vs"]) & np.isfinite(out["iv_atm"])
              & np.isfinite(out["iv_otm"]) & (out["iv_atm"] > 0)]
    if out.empty:
        return []
    out = out.reset_index().rename(columns={"act_symbol": "symbol"})
    out["skew"] = out["iv_otm"] - out["iv_atm"]
    out["n_cift"] = out["n_cift"].astype(int)
    return out[["date", "symbol", "vs", "skew", "iv_atm", "n_cift"]].to_dict("records")


# ---------------------------------------------------------------------------------------
# Yüzey serisi + fiyat → satırlar
# ---------------------------------------------------------------------------------------
def rows_for(sym: str, px: pd.DataFrame, yuzey: pd.DataFrame, horizon: int) -> list[dict]:
    """Karar günü D'nin yüzey ölçüsü → giriş D+1 kapanışı (feature-lag 1)."""
    p = px.set_index("date")
    close = p["close"].astype(float)
    y = yuzey.set_index("date").reindex(close.index)
    vs, skew = y["vs"], y["skew"]

    vs_z = (vs - vs.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean()) / \
        vs.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std().replace(0.0, np.nan)
    sk_z = (skew - skew.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean()) / \
        skew.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std().replace(0.0, np.nan)
    vs_chg = vs - vs.shift(5)
    ret5, ret21 = close.pct_change(5), close.pct_change(21)
    fwd = close.shift(-horizon) / close - 1.0

    out: list[dict] = []
    son_idx: int | None = None
    for i in range(len(close) - 1):
        zi = vs_z.iloc[i]
        if not np.isfinite(zi) or zi == 0.0:
            continue
        j = i + 1                                   # feature-lag 1
        if son_idx is not None and j - son_idx < horizon:
            continue
        deger = [fwd.iloc[j], vs.iloc[i], skew.iloc[i], sk_z.iloc[i], vs_chg.iloc[i],
                 y["iv_atm"].iloc[i], ret5.iloc[i], ret21.iloc[i]]
        if not all(np.isfinite(v) for v in deger):
            continue
        out.append({
            "ts": close.index[j], "symbol": sym,
            "f_vs": float(vs.iloc[i]),
            "f_vs_z": float(zi),
            "f_vs_absz": float(abs(zi)),
            "f_is_short": 0.0 if zi > 0 else 1.0,   # VS yüksek = call zengin = LONG
            "f_vs_chg": float(vs_chg.iloc[i]),
            "f_skew": float(skew.iloc[i]),
            "f_skew_z": float(sk_z.iloc[i]),
            "f_iv_atm": float(y["iv_atm"].iloc[i]),
            "f_ret_5": float(ret5.iloc[i]),
            "f_ret_21": float(ret21.iloc[i]),
            "fwd_ret": float(fwd.iloc[j]),
        })
        son_idx = j
    return out


def build(chain_csv: str | Path, horizon: int, cost_pct: float, borrow_bps_yr: float,
          since: str, until: str, out_csv: str, chunksize: int = 4_000_000,
          yuzey_csv: str | None = None) -> pd.DataFrame:
    panel = load_panel(since, until)
    if not panel:
        raise SystemExit(f"fiyat paneli boş — {PRICE_GLOB} okunamadı")
    logger.info(f"[optskew] evren: {len(panel)} sembol (PIT yerel), {since} → {until}")

    # Yüzey serisi 10,7 GB'lık bir taramanın (116,5M satır / ~5,5 dk) ÜRÜNÜDÜR. Diske
    # alınmazsa her teşhis sorusu o taramayı yeniden ödetir; ara ürünü saklamak bu depoda
    # kuraldır (Tur-35: "ölçüm bütçesini OKUYAN da öder").
    yc = Path(yuzey_csv) if yuzey_csv else None
    if yc is not None and yc.exists():
        logger.info(f"[optskew] yüzey serisi ÖNBELLEKTEN: {yc}")
        yuzey = pd.read_csv(yc, parse_dates=["date"])
    else:
        yuzey = yuzey_serisi(chain_csv, set(panel), since, until, chunksize)
        if yc is not None and not yuzey.empty:
            yc.parent.mkdir(parents=True, exist_ok=True)
            yuzey.to_csv(yc, index=False)
            logger.info(f"[optskew] yüzey serisi YAZILDI: {yc}")
    if yuzey.empty:
        logger.warning("[optskew] yüzey serisi BOŞ")
        return pd.DataFrame()
    logger.info(f"[optskew] yüzey: {len(yuzey):,} gün×sembol / {yuzey['symbol'].nunique()} "
                f"sembol ({yuzey['date'].min().date()} → {yuzey['date'].max().date()}); "
                f"ortalama {yuzey['n_cift'].mean():.1f} eşleşen çift")

    rows: list[dict] = []
    for sym, g in yuzey.groupby("symbol"):
        px = panel.get(str(sym))
        if px is None:
            continue
        rows.extend(rows_for(str(sym), px, g, horizon))

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("[optskew] SIFIR satır")
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
    logger.info(f"[optskew] YAZILDI {out_csv}: {len(df):,} satır / "
                f"{df['symbol'].nunique()} sembol "
                f"({df['ts'].min().date()} → {df['ts'].max().date()}); "
                f"short bacak %{100 * kisa.mean():.1f}")
    logger.info(f"[optskew] taban: ort pnl {100 * df['pnl'].mean():+.4f}%, "
                f"kazanan {df['label'].mean():.1%} (TRİVİYAL kural; model bunu GEÇMELİ)")
    logger.info(f"[optskew] BETA KONTROLÜ: piyasa {100 * df['mkt_ret'].mean():+.3f}%, "
                f"ham ileri {100 * df['fwd_ret'].mean():+.3f}% → FAZLALIK "
                f"{100 * df['excess'].mean():+.3f}%; VS-yüksek (long) bacak fazlalığı "
                f"{100 * df.loc[~kisa, 'excess'].mean():+.4f}% / VS-düşük (short) bacak "
                f"{100 * df.loc[kisa, 'excess'].mean():+.4f}% "
                f"(literatür: long bacakta POZİTİF fazlalık bekleniyor)")
    return df


def kapilardan_gecir(out_csv: str, name: str = "optskew_v1", *,
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
    ap = argparse.ArgumentParser(
        description="Opsiyon yuzeyi yon sinyali veri seti (BACKLOG #9)")
    ap.add_argument("--chain-csv", required=True,
                    help="option_chain toplu CSV yolu (OneDrive DISINDA olmali)")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--cost", type=float, default=0.1)
    ap.add_argument("--borrow-bps-yr", type=float, default=100.0)
    ap.add_argument("--since", default="2021-01-01")
    ap.add_argument("--until", default="2026-06-30")
    ap.add_argument("--chunksize", type=int, default=4_000_000)
    ap.add_argument("--yuzey-csv", default="data/edge_lab_optskew_yuzey.csv",
                    help="yuzey serisi onbellegi (varsa okunur, yoksa yazilir)")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--kapi", action="store_true", help="kurulduktan sonra kapilari kostur")
    args = ap.parse_args()
    df = build(args.chain_csv, args.horizon, args.cost, args.borrow_bps_yr,
               args.since, args.until, args.out, args.chunksize, args.yuzey_csv)
    if args.kapi and not df.empty:
        kapilardan_gecir(args.out, since=args.since, until=args.until)
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()
    raise SystemExit(main())
