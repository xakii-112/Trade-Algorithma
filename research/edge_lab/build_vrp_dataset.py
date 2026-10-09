"""
build_vrp_dataset.py — VARYANS RİSK PRİMİ (VRP) aday veri seti (Tur-51, BACKLOG.md #2).

BACKLOG'da bu aday "Veri kaynağı (Alpaca/ücretsiz) henüz belirlenmedi" notuyla BEKLIYOR'du.
Bu not 2026-08-13'te ESKİDİ: `nihai/data/dolt_options.py` o gün DoltHub
`post-no-preference/options` toplu CSV'sini indirdi — 2321 sembol × 1,9M satır, her satırda
`iv_current` (örtük) ve `hv_current` (gerçekleşen) oynaklık. Yakıt VARDI, defter bilmiyordu.

FİKİR (literatürün en dayanıklı primlerinden): örtük oynaklık, sonradan GERÇEKLEŞEN
oynaklığın üstünde fiyatlanma eğilimindedir — opsiyon satıcısı bu farkı prim olarak alır.
Aday soru KENDİ primin varlığı DEĞİL (o bilinen bir olgu), **zamanlanabilir/seçilebilir
olup olmadığıdır**: t anında bilinen değişkenlerle, "her satırda prim sat" TRİVİYAL
kuralından daha iyi bir alt küme seçilebiliyor mu? Harness'ın kapı simülasyonu tam bunu
sorar (Tur-31'den beri taban = triviyal kural, sıfır DEĞİL).

Bu script KENAR İDDİASI YAPMAZ — yalnız veri taşır. Verdikt `harness.run_candidate`'ın
beş kapısındadır (purged_cv → gate_time → gate_symbol → dsr → exec_lag_1).

────────────────────────────────────────────────────────────────────────────────────────
K0 — İLERİ OYNAKLIK FİYATTAN HESAPLANIR, `hv_current` KAYDIRILARAK DEĞİL (Tur-51 BULGU-51-1)
────────────────────────────────────────────────────────────────────────────────────────
`research/iv_rank_threshold_test.py` ileri oynaklığı `hv_current.shift(-W)` ile vekilliyor
ve W'yi "sağlayıcı belgelemiyor" diyerek TARIYOR (varsayılan 21). Pencere ÖLÇÜLDÜ
(10 sembol × 10.132 ortak gün, fiyattan geriye-bakan yıllıklandırılmış RV ile eşleştirme):

    W:      5       10      14      20      21      22      25    **30**    44      63
    kor:  0,671   0,806   0,869   0,936   0,945   0,953   0,974  **0,99962** 0,947  0,900
    MAE:  0,087   0,059   0,046   0,029   0,027   0,024   0,017  **0,00143** 0,025  0,036

`hv_current` = **geriye-bakan 30 işlem günlük** gerçekleşen oynaklık (medyan oran 0,9999).
Sonuç: `hv_current(t+W)` (t+W−30, t+W] aralığını ölçer. W < 30 iken bu aralığın
**(30−W)/30'u t'den ÖNCEDİR** — yani "ileri" etiketin bir kısmı KARAR BARINDAN ÖNCEKİ
veridir ve `iv_current(t)` ile aynı bilgiyi taşır. W=21'de etiketin **%30'u** böyledir.
Bir KAYDIRMA MİKTARI, kaydırdığı PENCERE okunmadan seçilemez (Tur-33'ün "bir oran paydası
okunmadan okunamaz" kuralının zaman eksenindeki hâli).

Bu script kaydırmayı hiç kullanmaz: ileri RV, `data/alpaca_hist/SYM_1d_all.csv`
(bölünme/temettü DÜZELTİLMİŞ) kapanışlarından, karar barından SONRAKİ W barın log
getirileriyle hesaplanır → örtüşme yapısı gereği SIFIR, W serbestçe seçilebilir.

────────────────────────────────────────────────────────────────────────────────────────
EVREN — PIT/SURVIVORSHIP-TEMİZ (ACI DERS #1)
────────────────────────────────────────────────────────────────────────────────────────
`data/alpaca_hist/*_1d_all.csv` önbelleği endeks-rebalans pilotu için `sp500_as_of`
(point-in-time) evreniyle BİR KEZ çekilmişti; bu script onu yeniden Wikipedia'ya sormadan
kullanır (ağ bağımsız + aynı evren). Survivorship-temizliği DOLAYLI değil DOĞRUDAN kanıt:
609 sembolün **45'inin** fiyat serisi evren-sonundan >90 gün ÖNCE biter (TIF/LVMH,
CXO/ConocoPhillips, XLNX/AMD, ALXN/AstraZeneca, MXIM/ADI, KSU/CP, FLIR, WLTW, HFC…) —
survivorship-yanlı bir evrende bu isimlerden HİÇBİRİ bulunmaz. Dolt kesişimi 605 sembol.

────────────────────────────────────────────────────────────────────────────────────────
ÖZELLİKLER — hepsi t'de bilinir, hepsi YILLIKLANDIRILMIŞ OYNAKLIK BİRİMİNDE
────────────────────────────────────────────────────────────────────────────────────────
Harness `LogisticRegression`'ı ÖLÇEKLEMEDEN kurar ve L2 cezası ölçek-değişmez DEĞİLDİR;
farklı büyüklükteki kolonlar katsayıları haksız cezalandırır. Çözüm harness'ı değiştirmek
(adaya göre kural değiştirmek) değil, birimi DÜZELTMEKTİR: VIX 20 zaten "%20 yıllık
oynaklık"tır → 100'e bölününce `iv_current` ile AYNI birimdedir. Tüm kolonlar ~[0, 1,5].

  f_iv          iv_current(t)                        — örtük oynaklık
  f_hv          hv_current(t)                        — geriye-bakan 30g gerçekleşen (K0)
  f_vrp         iv_current − hv_current              — ANLIK prim (adayın çekirdeği)
  f_iv_rank     252-gözlem min-max rank / 100        — üretimdeki `get_iv_rank` matematiği
  f_iv_chg_w    iv_current − iv_current(t−5)         — primin kısa vadeli yönü
  f_iv_chg_m    iv_current − iv_current(t−21)        — (sağlayıcı kolonu DEĞİL, BULGU-51-2)
  f_hv_chg_m    hv_current − hv_current(t−21)        — gerçekleşen oynaklığın yönü
  f_iv_pos_yr   (iv − iv_year_low)/(iv_year_high−low)— yıl bandındaki konum
  f_vix         VIX kapanış / 100                    — piyasa-geneli rejim
  f_vix_ts      VIX3M/VIX − 1                        — vade yapısı (contango>0)

NEDENSELLİK ÖLÇÜLDÜ: `iv_year_high_date`/`iv_year_low_date` 1,78M satırın **%0,000000**'ında
satır tarihinin ilerisinde — sağlayıcı alanları geriye-bakan (Tur-34 `causal` disiplini).

────────────────────────────────────────────────────────────────────────────────────────
ETİKET ve PnL — PRİM SATAN taraf, MALİYET DÜŞÜLMÜŞ
────────────────────────────────────────────────────────────────────────────────────────
    kenar_net(t) = iv_current(t) − rv_ileri(t) − maliyet
    label        = 1 if kenar_net > 0 else 0
    pnl          = kenar_net            (harness `pnl_col`)

`rv_ileri` = t'den SONRAKİ W barın yıllıklandırılmış RV'si. Yön PRİM SATAN taraftır çünkü
ölçülen taban POZİTİFTİR (uzun opsiyon tarafı ortalama kaybeder); harness'ın triviyal
kuralı "her satırda sat" olur ve model bunu GEÇMEK zorundadır.

**PnL'İN SINIRI (ACI DERS #3 — okunmadan sayı okunamaz):** birim OYNAKLIK PUANIDIR, dolar
DEĞİL. Delta-hedge'li gamma P&L'i ≈ ½Γ S²(σ²_gerçek − σ²_örtük)dt yaklaşımının İŞARETİ
doğrudur, BÜYÜKLÜĞÜ vega'ya bağlıdır ve burada modellenmez. `--cost` (varsayılan 1,0
oynaklık puanı) opsiyon makasının kaba karşılığıdır; gerçek zincir bid/ask'i DEĞİLDİR.
Bu veri seti bir kenarı KANITLAYAMAZ — yalnız ÇÜRÜTEBİLİR.

ÖRTÜŞME (ACI DERS #8): ardışık günlerin ileri pencereleri neredeyse aynı aralığı ölçer →
t şişer. Sembol başına her W. gözlem alınır (`--overlap` ile kapatılabilir; VERDİKT için
ÖNERİLMEZ).

VERİ GECİKMESİ (ACI DERS #7): sağlayıcı satırı gün SONUNDA yayımlanıyor olabilir; t
kapanışında işlem varsayımı iyimserdir. `--feature-lag` VARSAYILAN 1'dir (güvenli yön):
özellikler bir önceki gözlemden okunur. 0 yalnız duyarlılık kontrolü içindir.

Kullanım:
    python -m research.edge_lab.build_vrp_dataset
    python -m research.edge_lab.build_vrp_dataset --horizon 21 --cost 1.0
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from nihai.console import ensure_utf8_stdout
from nihai.data.iv_history import MIN_DAYS_OK, ROLLING_WINDOW_DAYS

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Sabitler --------------------------------------------------------------------------
DEFAULT_HORIZON = 21          # bir opsiyon döngüsü (~1 işlem ayı); ÖNCEDEN seçildi, taranmadı
DEFAULT_COST_VOLPTS = 1.0     # oynaklık PUANI (= 0,01 mutlak) — kaba opsiyon makası vekili
RANK_WINDOW = ROLLING_WINDOW_DAYS   # iv_history ile AYNI pencere (sabit kopyalanmaz)
# BULGU-52-1 — `min_periods`, pencerenin KENDİSİ olamaz. Izgara dolt'un yalnız ~%92'sini
# taşır ve eksikler ÖBEKLİ; `min_periods=252` istemek "son 252 işlem gününün HEPSİNDE
# IV olsun" demektir → tek bir öbek kesinti, SONRAKİ 252 günün rank'ını da NaN yapar ve
# ortak `dropna` o satırları sessizce siler. Tur-51 veri setinde bunun bedeli 2023-12 →
# 2025-07 arası 20 AYIN tamamen kaybıydı (kaynakta 2024 için 96.369 satır DURUYORDU).
# Üretim sözleşmesi zaten 252 değil: `iv_history` 30 gözlemden itibaren rank döner,
# 100'den itibaren "OK" der. Araştırma tarafı üretimden SIKI olamaz — sabit oradan okunur.
TRADING_DAYS = 252            # yıllıklandırma
PRICE_GLOB = "data/alpaca_hist/*_1d_all.csv"
VIX_CSV = "data/cboe_cache/VIX.csv"
VIX3M_CSV = "data/cboe_cache/VIX3M.csv"
OUT_CSV = "data/edge_lab_vrp_v1.csv"

FEATURE_COLS: list[str] = [
    "f_iv", "f_hv", "f_vrp", "f_iv_rank",
    "f_iv_chg_w", "f_iv_chg_m", "f_hv_chg_m", "f_iv_pos_yr",
    "f_vix", "f_vix_ts",
]
SCHEMA_COLS: list[str] = ["ts", "symbol", *FEATURE_COLS, "rv_fwd", "pnl", "label"]

_VOL_COLS = ["date", "act_symbol", "iv_current", "hv_current",
             "iv_year_high", "iv_year_low"]

# Değişim özellikleri sağlayıcının `*_week_ago`/`*_month_ago` kolonlarından OKUNMAZ,
# `iv_current`/`hv_current` serisinden HESAPLANIR — iki nedenle (Tur-51 BULGU-51-2):
#
# (1) **Sağlayıcı 2024'ten beri bu kolonları hiç doldurmuyor.** Yıl-yıl NaN oranı:
#     2019-2022 ≈ %0 · 2023 %10 · **2024/2025/2026 %100**. Bu kolonlara bağlanan bir
#     hat 2023 sonunda SESSİZCE biter — ilk sürüm tam bunu yaptı (veri seti 2026-08
#     yerine 2023-11-29'da bitiyordu, satırların %55'i düşmüştü). Boş kolon hatası
#     yoktur, yalnız daha kısa bir tarih aralığı vardır; okunmazsa fark edilmez.
# (2) Doldurulduğu yerde bile TANIMI belirsiz: `iv_month_ago ≈ iv_current(t−k)`
#     eşleştirmesi k∈[18,23] aralığında MONOTON artan hata verir (en iyi k=18, ARAMA
#     ARALIĞININ SINIRI) = basit bir gecikme DEĞİL. `iv_rank_threshold_test.py`'nin
#     HV penceresi için terk ettiği "sınırda çıkan L uydurmadır" teşhisinin aynısı.
#
# Kendi gecikmemiz belgeli ve nedenseldir: t'deki gözlemden N gözlem geriye.
_LAG_WEEK = 5
_LAG_MONTH = 21
# Dolt satırı eksik olan işlem gününde son bilinen değer en fazla bu kadar taşınır
# (`_to_price_calendar`). Daha eskisi "bugünkü IV" sayılamaz.
_STALE_LIMIT = 5


# ---------------------------------------------------------------------------------------
# Fiyat tarafı — İLERİ gerçekleşen oynaklık (K0: kaydırma YOK, fiyattan hesap)
# ---------------------------------------------------------------------------------------
def forward_realized_vol(close: pd.Series, horizon: int) -> pd.Series:
    """t barından SONRAKİ `horizon` barın yıllıklandırılmış gerçekleşen oynaklığı.

    r_{i+1} = log(c_{i+1}/c_i) getirisi (t_i, t_{i+1}] aralığını kapsar — yani i. barın
    KAPANIŞINDAN sonrasını. İleri pencere r_{i+1}..r_{i+horizon} getirileridir; bu yüzden
    `rolling(horizon)` sonucu `shift(-horizon)` ile i'ye hizalanır. i'ye kadarki hiçbir
    getiri pencereye GİRMEZ (bkz. modül docstring K0) — bu, `hv_current.shift(-W)`
    vekilinin taşıdığı (30−W)/30 kirlenmesinin yapısal çözümüdür.
    """
    lr = np.log(close.astype(float)).diff()
    return lr.rolling(horizon).std(ddof=1).shift(-horizon) * np.sqrt(TRADING_DAYS)


def _load_prices(horizon: int) -> pd.DataFrame:
    """Tüm `*_1d_all.csv` (DÜZELTİLMİŞ) → (symbol, date, rv_fwd) uzun tablo."""
    frames = []
    for path in sorted(glob.glob(PRICE_GLOB)):
        sym = os.path.basename(path).split("_1d_all")[0]
        try:
            px = pd.read_csv(path, usecols=["timestamp", "close"])
        except (OSError, ValueError) as e:                # bozuk/eksik dosya koşuyu durdurmaz
            logger.debug(f"[vrp] {sym}: fiyat okunamadı ({e})")
            continue
        if len(px) < horizon + 2:
            continue
        px["date"] = pd.to_datetime(px["timestamp"], errors="coerce").dt.normalize()
        px = px.dropna(subset=["date", "close"]).sort_values("date")
        frames.append(pd.DataFrame({
            "symbol": sym,
            "date": px["date"].to_numpy(),
            "rv_fwd": forward_realized_vol(px["close"], horizon).to_numpy(),
        }))
    if not frames:
        raise SystemExit(f"fiyat önbelleği boş: {PRICE_GLOB}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------------------
# Oynaklık tarafı — Dolt toplu CSV
# ---------------------------------------------------------------------------------------
def _load_vol(symbols: set[str]) -> pd.DataFrame:
    from nihai.data import dolt_options
    path = dolt_options.bulk_csv("volatility_history")
    if path is None:
        raise SystemExit("toplu CSV yok — `python -m research.backfill_iv_history` koşun")
    df = dolt_options.read_bulk(path, cols=_VOL_COLS, symbols=symbols)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "iv_current", "hv_current"])
    return df.rename(columns={"act_symbol": "symbol"}).sort_values(["symbol", "date"])


def _load_vix() -> pd.DataFrame:
    """VIX seviyesi + VIX3M/VIX vade yapısı. Dosya yoksa kolonlar NaN kalır (koşu durmaz)."""
    def _read(path: str, name: str) -> pd.Series | None:
        if not Path(path).exists():
            logger.warning(f"[vrp] {path} yok — {name} özelliği NaN kalacak")
            return None
        d = pd.read_csv(path, usecols=["date", "close"])
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        return d.dropna(subset=["date"]).set_index("date")["close"].rename(name)

    vix, vix3m = _read(VIX_CSV, "vix"), _read(VIX3M_CSV, "vix3m")
    if vix is None:
        return pd.DataFrame(columns=["date", "f_vix", "f_vix_ts"])
    out = pd.DataFrame({"vix": vix})
    out["vix3m"] = vix3m if vix3m is not None else np.nan
    out["f_vix"] = out["vix"] / 100.0                       # VIX 20 → 0,20 = iv birimi
    out["f_vix_ts"] = np.where(out["vix"] > 0, out["vix3m"] / out["vix"] - 1.0, np.nan)
    return out.reset_index()[["date", "f_vix", "f_vix_ts"]]


# ---------------------------------------------------------------------------------------
# Özellikler
# ---------------------------------------------------------------------------------------
def _to_price_calendar(vol: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Dolt serisini DÜZENLİ işlem-günü ızgarasına taşı (Tur-51 BULGU-51-3).

    **Bir SATIR kaydırması, seri düzensizse bir ZAMAN kaydırması DEĞİLDİR.** Dolt serisi
    sembol başına fiyat barlarının ~%93,4'ünü taşır ve eksikler tek tek değil ÖBEKLER
    hâlindedir (evren genelinde 1713 günün 623'ünde kapsama %50'nin altında). Ölçüldü:

      `groupby(symbol).shift(-30)` hedefi 30 işlem günü ≈ **42-44 takvim günü**;
      GERÇEKTE ortanca **70**, ortalama 65,2, std **34,0** gün — %59,6'sı 60 günü aşıyor.
      Geri yönde `shift(21)` (hedef 29-31 gün): ortanca **49**, std 26,7.

    Yani ufuk sembole ve döneme göre DEĞİŞİR. Bu, `iv_rank_threshold_test`in ölçtüğü
    "monoton ilişki"nin de kaynağıdır (aşağıdaki BULGU-51-4) ve bu script'in KENDİ
    değişim özelliklerini de vuruyordu — ders kendi koduna da uygulanır.

    Çözüm: fiyat takvimi TAM olduğu için (`_1d_all` her işlem günü bir bar) dolt değerleri
    ona sol-birleştirilir ve NEDENSEL ffill ile taşınır. ffill SINIRLIDIR (`_STALE_LIMIT`
    işlem günü): sınırın ötesinde değer NaN kalır ve satır düşer — bir haftadan eski IV'yi
    "bugünkü IV" diye sunmak, ölçüm yokluğunu ölçüm gibi raporlamak olurdu (Tur-31).
    Izgara düzenli olunca `shift(N)` TAM OLARAK N işlem günüdür.
    """
    grid = px[["symbol", "date"]].drop_duplicates()
    out = (grid.merge(vol, on=["symbol", "date"], how="left")
                .sort_values(["symbol", "date"], ignore_index=True))
    cols = [c for c in out.columns if c not in ("symbol", "date")]
    out[cols] = out.groupby("symbol", sort=False)[cols].ffill(limit=_STALE_LIMIT)
    tasima = out["iv_current"].notna().sum()
    logger.info(f"[vrp] fiyat takvimine taşındı: {len(out):,} ızgara satırı, "
                f"{tasima:,} ({tasima / max(len(out), 1):.1%}) IV taşıyor "
                f"(ffill sınırı {_STALE_LIMIT} işlem günü)")
    return out


def _add_features(df: pd.DataFrame, min_rank_obs: int) -> pd.DataFrame:
    g = df.groupby("symbol", sort=False)["iv_current"]
    lo = g.transform(lambda s: s.rolling(RANK_WINDOW, min_periods=min_rank_obs).min())
    hi = g.transform(lambda s: s.rolling(RANK_WINDOW, min_periods=min_rank_obs).max())
    span = hi - lo
    # `nihai/data/iv_history.get_iv_rank` ile AYNI matematik; /100 ile oynaklık birimine
    rank = np.where(span > 1e-9, (df["iv_current"] - lo) / span, np.nan)

    gs = df.groupby("symbol", sort=False)
    yr_span = df["iv_year_high"] - df["iv_year_low"]
    df["f_iv"] = df["iv_current"]
    df["f_hv"] = df["hv_current"]
    df["f_vrp"] = df["iv_current"] - df["hv_current"]
    df["f_iv_rank"] = np.clip(rank, 0.0, 1.0)
    df["f_iv_chg_w"] = df["iv_current"] - gs["iv_current"].shift(_LAG_WEEK)
    df["f_iv_chg_m"] = df["iv_current"] - gs["iv_current"].shift(_LAG_MONTH)
    df["f_hv_chg_m"] = df["hv_current"] - gs["hv_current"].shift(_LAG_MONTH)
    df["f_iv_pos_yr"] = np.where(yr_span > 1e-9,
                                 (df["iv_current"] - df["iv_year_low"]) / yr_span, np.nan)
    return df


def build(horizon: int, cost_volpts: float, feature_lag: int, min_rank_obs: int,
          non_overlap: bool, out_csv: str) -> pd.DataFrame:
    px = _load_prices(horizon)
    syms = set(px["symbol"].unique())
    logger.info(f"[vrp] fiyat: {len(px):,} satır / {len(syms):,} sembol")

    vol = _load_vol(syms)
    logger.info(f"[vrp] dolt: {len(vol):,} satır / {vol['symbol'].nunique():,} sembol "
                f"({vol['date'].min().date()} → {vol['date'].max().date()})")

    vol = _to_price_calendar(vol, px)
    df = _add_features(vol, min_rank_obs)

    # ACI DERS #7 — sağlayıcı satırı gün SONUNDA yayımlanmış olabilir: özellikleri
    # `feature_lag` gözlem geriden oku (etiket ve tarih t'de KALIR; kaydırılan yalnız
    # KARARIN dayandığı bilgidir). Örtüşme kaldırılmadan ÖNCE uygulanır ki lag gerçek
    # işlem günü olsun, W günlük örnekleme adımı değil.
    if feature_lag:
        df[FEATURE_COLS[:8]] = df.groupby("symbol", sort=False)[FEATURE_COLS[:8]].shift(feature_lag)
        df["iv_current"] = df.groupby("symbol", sort=False)["iv_current"].shift(feature_lag)

    vix = _load_vix()
    df = df.merge(vix, on="date", how="left") if len(vix) else df.assign(f_vix=np.nan, f_vix_ts=np.nan)

    df = df.merge(px, on=["symbol", "date"], how="inner")

    # PRİM SATAN taraf, maliyet düşülmüş. `iv_current` burada zaten lag'lenmiş değerdir:
    # işlem o fiyattan kurulur, sonuç gerçek ileri RV ile ölçülür.
    df["pnl"] = df["iv_current"] - df["rv_fwd"] - cost_volpts / 100.0
    df["label"] = (df["pnl"] > 0).astype(int)

    df = df.dropna(subset=[*FEATURE_COLS, "pnl"])
    df = df.sort_values(["symbol", "date"], ignore_index=True)

    if non_overlap:
        before = len(df)
        df = df[df.groupby("symbol", sort=False).cumcount() % horizon == 0]
        logger.info(f"[vrp] örtüşme kaldırıldı: {before:,} → {len(df):,} "
                    f"(sembol başına her {horizon}. gözlem)")

    out = (df.rename(columns={"date": "ts"})[SCHEMA_COLS]
             .sort_values(["ts", "symbol"], ignore_index=True))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    logger.info(f"[vrp] YAZILDI {out_csv}: {len(out):,} satır / {out['symbol'].nunique():,} "
                f"sembol ({out['ts'].min().date()} → {out['ts'].max().date()})")
    logger.info(f"[vrp] taban: ort pnl {out['pnl'].mean():+.5f} oynaklık puanı, "
                f"kazanan oran {out['label'].mean():.1%} "
                f"(= 'her satırda prim sat' TRİVİYAL kuralı; model bunu GEÇMELİ)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="VRP aday veri seti (BACKLOG #2)")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                    help="ileri RV penceresi (işlem günü); ÖNCEDEN seçilir, taranmaz")
    ap.add_argument("--cost", type=float, default=DEFAULT_COST_VOLPTS,
                    help="oynaklık PUANI cinsinden kaba opsiyon makası")
    ap.add_argument("--feature-lag", type=int, default=1,
                    help="özellikleri N gözlem geriden oku (ACI DERS #7; 0 = duyarlılık)")
    ap.add_argument("--min-rank-obs", type=int, default=MIN_DAYS_OK,
                    help=f"IV rank için asgari gözlem (üretim sözleşmesi: "
                         f"iv_history.MIN_DAYS_OK={MIN_DAYS_OK}, pencere {RANK_WINDOW})")
    ap.add_argument("--overlap", action="store_true",
                    help="örtüşmeyi KORU (t şişer — verdikt için ÖNERİLMEZ)")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    build(args.horizon, args.cost, args.feature_lag, args.min_rank_obs,
          not args.overlap, args.out)
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()
    raise SystemExit(main())
