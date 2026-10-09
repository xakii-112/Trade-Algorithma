"""
build_leadlag_dataset.py — Sektör-içi lead-lag (büyük→küçük gecikmeli yayılım) aday veri seti
(edge_lab BACKLOG.md #3). KARAR-DIŞI/ARAŞTIRMA.

PILOT: mevcut-likit 30 sembol = SURVIVORSHIP-ŞİŞKİN evren (ACI DERS #1). Bu script KENAR
İDDİASI YAPMAZ — yalnız veri taşır; hüküm `harness.run_candidate`'ın dört kapısınındır.

Fikir: aynı sektördeki BÜYÜK (likit) bir hissenin hareketi, aynı sektörün daha KÜÇÜK
hisselerine bir bar GECİKMEYLE yayılır (Lo & MacKinlay 1990 lead-lag; Hou 2007 sektör-içi
gecikmeli tepki). `t` barının KAPANIŞINDA lider'in o bardaki getirisi bilinir; aday soru
şudur: bu, takipçinin `t+1` barındaki getirisinin İŞARETİNİ tahmin ediyor mu?

⚠️ ACI DERS #7 (exec-lag) RİSKİ BU AİLEDE ÖZELLİKLE YÜKSEK (BACKLOG.md notu). Kenarın
sinyal-sonrası İLK barda yoğunlaşması tam olarak bu ailenin beklenen yanılgısıdır
(bid-ask sıçraması / mikroyapı gürültüsü, gerçek yayılım değil). `run_candidate`'ın
`exec_lag_1` kapısı bunu ölçer ve bu aday için asıl belirleyici kapıdır.

## Lider seçimi — HİNDSIGHT YOK (kritik tasarım kararı)

Lider sektörün "en büyüğü" diye ELLE sabitlenmedi. Bugünün bilgisiyle "NVDA yarıiletkenlerin
lideridir" demek, 2 yıl önceki veriye bugünün sıralamasını taşımak olurdu (sinsi bir
look-ahead — CLAUDE.md "Look-ahead yasak"). Bunun yerine lider HER BAR'da yeniden ve
YALNIZ GERİYE bakan bir ölçüyle belirlenir:

    lider(sektör, t) = son 20 barın MEDYAN dolar-hacmi (close × volume) en yüksek olan üye

Medyan (ortalama değil) tek bir haber-günü hacim patlamasının liderliği devirmesini engeller.
`t` barının close/volume'u `t` kapanışında BİLİNİR → eşik-anında erişilebilir bilgi, sızıntı yok.
Sıralama bar-bar değişebilir; bir sembol bir barda lider, ertesi barda takipçi olabilir.

## Satır üretimi

Her (sektör, t) için: rank-1 üye LİDER'dir ve KENDİSİ İÇİN SATIR ÜRETMEZ (bir sembol kendini
"lead" edemez). Kalan üyelerin her biri bir takipçi satırı verir.

## Özellikler (TÜMÜ yalnız t barının KAPANIŞINA kadarki veriden — Madde 3)

  f_leader_ret1    — liderin t barındaki getirisi (close_t/close_{t-1} − 1)  ← ÇEKİRDEK sinyal
  f_leader_ret3    — liderin son 3 barlık kümülatif getirisi
  f_leader_ret1_z  — f_leader_ret1'in liderin son 20 barlık getiri std'sine bölünmüşü
                     (oynaklık rejiminden arındırılmış "şok büyüklüğü")
  f_self_ret1      — takipçinin t barındaki KENDİ getirisi (oto-korelasyon kontrolü)
  f_gap            — f_leader_ret1 − f_self_ret1 (AYRIŞMA / "yetişme" terimi; lead-lag
                     hipotezinin doğrudan ifadesi: lider oynadı, takipçi henüz oynamadı)
  f_self_vol20     — takipçinin son 20 barlık getiri std'si
  f_volume_z       — takipçinin hacim z-skoru (son 20 bar ort/std)
  f_hour           — barın saati (UTC; lead-lag literatürde gün-içi seans başında yoğunlaşır)

## Etiket

`label` = takipçinin t+1 barındaki getirisi maliyet eşiğinin (5 bps) ÜSTÜNDE ise 1, değilse 0.
Bu değer t kapanışından SONRA gerçekleşir → hiçbir özellik kolonunda YER ALMAZ.

**AYNI-GÜN kısıtı:** t ve t+1 AYNI takvim gününde değilse satır DÜŞÜRÜLÜR. Aksi halde etiket
gece boşluğunu (overnight getirisi) içerirdi ve bu, BACKLOG #1'de ayrıca ölçülmüş (ve 5/5 kapıda
FAIL etmiş) BAŞKA bir hipotezle karışırdı. Hipotezi temiz tutmak için gün-içi yayılıma bakılır.

CSV şeması (harness.run_candidate ile birebir): ts,symbol,sector,leader,<8 özellik>,label

Kullanım:
    python -m research.edge_lab.build_leadlag_dataset
    python -m research.edge_lab.build_leadlag_dataset --years 2 --interval 1h
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from nihai.console import ensure_utf8_stdout

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Sabitler --------------------------------------------------------------------------
COST_THRESHOLD_BPS = 5.0
COST_THRESHOLD = COST_THRESHOLD_BPS / 10_000.0
DEFAULT_YEARS = 2
DEFAULT_INTERVAL = "1h"
RANK_WINDOW = 20          # lider sıralaması için geriye-bakan pencere (bar)
VOL_WINDOW = 20           # oynaklık/hacim z pencereleri (bar)
MIN_HISTORY_BARS = 21     # en geniş pencere + 1 bar getiri ısınması

# Sektör grupları — PILOT evren (.env STREAM_SYMBOLS, ETF'ler HARİÇ).
# SPY/QQQ kasıtlı DIŞARIDA: bunlar sektör üyesi değil, piyasa vekilidir; sektör-içi yayılım
# hipotezine sokmak "lider" kavramını piyasa-betasına çevirirdi (hipotez bulanır).
# NOT: gruplama sabit ve ELLE — ama LİDER sabit değil (yukarıdaki geriye-bakan sıralama).
SECTORS: dict[str, list[str]] = {
    "yariiletken":     ["NVDA", "AVGO", "TSM", "AMD", "QCOM", "MU", "MRVL", "SNDK"],
    "yazilim_bulut":   ["MSFT", "ORCL", "GOOGL", "NOW", "CRWD", "NBIS"],
    "donanim_bt":      ["AAPL", "IBM", "CSCO", "DELL", "HPQ"],
    "enerji":          ["XOM", "CVX"],
    "finans":          ["JPM", "V"],
    "tuketici":        ["AMZN", "TSLA"],
    "sanayi_savunma":  ["GE", "LMT", "ONDS"],
}

FEATURE_COLS: list[str] = [
    "f_leader_ret1", "f_leader_ret3", "f_leader_ret1_z",
    "f_self_ret1", "f_gap", "f_self_vol20", "f_volume_z", "f_hour",
]
SCHEMA_COLS: list[str] = ["ts", "symbol", "sector", "leader", *FEATURE_COLS, "label"]

DEFAULT_OUT = ROOT / "data" / "edge_leadlag_pilot.csv"

# Hedefler (BEKLENTİ — ulaşılamazsa koşu durmaz, yalnız uyarı loglanır).
TARGET_MIN_SYMBOLS = 15
TARGET_MIN_ROWS = 20_000


# ---------------------------------------------------------------------------------------
# Veri çekimi — alpaca_history (savunmacı: hata → None, koşu durmaz)
# ---------------------------------------------------------------------------------------
def fetch_bars(symbol: str, years: int = DEFAULT_YEARS,
               interval: str = DEFAULT_INTERVAL) -> pd.DataFrame | None:
    """Sembolün OHLCV barları. Hata/boş → None (çağıran sembolü atlar)."""
    try:
        from nihai.data.alpaca_history import get_history
    except Exception as e:  # pragma: no cover - import-zamanı arıza
        logger.debug(f"[leadlag] alpaca_history import edilemedi: {e}")
        return None
    end = datetime.now(UTC)
    start = end - timedelta(days=int(years * 365.25))
    try:
        df = get_history(symbol, interval, start=start, end=end)  # type: ignore[arg-type]
    except Exception as e:
        logger.warning(f"[leadlag] {symbol} çekim hatası: {e}")
        return None
    if df is None or df.empty:
        return None
    return df


# ---------------------------------------------------------------------------------------
# Sektör paneli: üyeleri ORTAK zaman ızgarasında hizala
# ---------------------------------------------------------------------------------------
def _panel(frames: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    """Üye başına tek kolonlu geniş tablo (index=ts). Yalnız TÜM üyelerde bulunan barlar
    tutulur (inner join) — lider ile takipçinin barları birebir aynı ana denk gelmezse
    'gecikmeli yayılım' ölçümü anlamını yitirir."""
    cols = {}
    for sym, df in frames.items():
        s = df[field].astype(float)
        s = s[~s.index.duplicated(keep="last")]
        cols[sym] = s
    panel = pd.DataFrame(cols).dropna(how="any")
    return panel.sort_index()


def build_features_for_sector(frames: dict[str, pd.DataFrame], sector: str) -> pd.DataFrame | None:
    """Bir sektörün üye OHLCV'lerinden takipçi satırları üret (leak-free).

    frames: {sembol: DatetimeIndex'li OHLCV df}. En az 2 üye ve MIN_HISTORY_BARS ortak bar
    gerekir; aksi halde None."""
    frames = {s: d for s, d in frames.items()
              if d is not None and not d.empty and {"close"}.issubset(d.columns)}
    if len(frames) < 2:
        return None

    close = _panel(frames, "close")
    if len(close) < MIN_HISTORY_BARS:
        return None
    has_volume = all("volume" in d.columns for d in frames.values())
    volume = _panel(frames, "volume").reindex(close.index) if has_volume else None

    ret1 = close.pct_change()
    ret3 = close.pct_change(3)
    vol20 = ret1.rolling(VOL_WINDOW).std()

    # --- Lider sıralaması: son RANK_WINDOW barın MEDYAN dolar-hacmi (geriye-bakan) ---
    if volume is not None:
        dollar_vol = (close * volume).rolling(RANK_WINDOW).median()
    else:
        # Hacim yoksa dolar-hacim vekili: fiyat düzeyi (büyük hisse ≈ yüksek nominal işlem
        # hacmi varsayımı DEĞİL — yalnız degrade yol; test/sentetik veri için).
        dollar_vol = close.rolling(RANK_WINDOW).median()
        logger.warning(f"[leadlag] {sector}: hacim kolonu yok — lider sıralaması fiyat vekiliyle")

    # Her bar için rank-1 üye. DİKKAT: rolling ısınma penceresinde (ilk RANK_WINDOW-1 bar)
    # satırın TAMAMI NaN'dır ve pandas'ın idxmax'i orada ValueError atar — o barlarda lider
    # tanımsızdır (<NA>) ve satır zaten aşağıda `leader_sym.notna()` ile elenir.
    valid = dollar_vol.notna().any(axis=1)
    leader_sym = pd.Series(pd.NA, index=close.index, dtype=object)
    if bool(valid.any()):
        leader_sym.loc[valid] = dollar_vol.loc[valid].idxmax(axis=1)

    if volume is not None:
        vmean = volume.rolling(VOL_WINDOW).mean()
        vstd = volume.rolling(VOL_WINDOW).std().replace(0.0, np.nan)
        volume_z = (volume - vmean) / vstd
    else:
        volume_z = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    # Etiket: takipçinin t+1 barındaki getirisi. shift(-1) GELECEK — yalnız label'da.
    # `NaN > eşik` False döner (NaN değil) → son barda sahte label=0 sızardı; NaN'ı
    # np.where ile AÇIKÇA koru (build_overnight_dataset.py'deki aynı tuzak).
    fwd_ret = ret1.shift(-1)

    # AYNI-GÜN kısıtı: t+1 barı ertesi güne düşüyorsa satır geçersiz (gece boşluğu
    # etikete karışır ve BACKLOG #1 hipoteziyle çakışır).
    idx = pd.DatetimeIndex(close.index)
    same_day = pd.Series(idx.date, index=close.index).shift(-1) == pd.Series(idx.date, index=close.index)

    # Liderin serileri sembole BAĞLI DEĞİL — sektör başına bir kez hesaplanır (her takipçi
    # için yeniden üretmek aynı sonucu üye sayısı kadar tekrar hesaplamak olurdu).
    def _by_leader(src: pd.DataFrame) -> pd.Series:
        return pd.Series(
            [src.at[t, leader_sym.at[t]] if isinstance(leader_sym.at[t], str) else np.nan
             for t in close.index], index=close.index, dtype=float)

    lead_ret1 = _by_leader(ret1)
    lead_ret3 = _by_leader(ret3)
    lead_vol = _by_leader(vol20)
    lead_ret1_z = lead_ret1 / lead_vol.replace(0.0, np.nan)

    rows: list[pd.DataFrame] = []
    for sym in close.columns:
        is_follower = leader_sym.ne(sym) & leader_sym.notna()
        fwd = fwd_ret[sym]
        label_raw = np.where(fwd.isna(), np.nan, (fwd > COST_THRESHOLD).astype(float))

        feat = pd.DataFrame({
            "ts": close.index,
            "symbol": sym,
            "sector": sector,
            "leader": leader_sym.to_numpy(),
            "f_leader_ret1": lead_ret1.to_numpy(),
            "f_leader_ret3": lead_ret3.to_numpy(),
            "f_leader_ret1_z": lead_ret1_z.to_numpy(),
            "f_self_ret1": ret1[sym].to_numpy(),
            "f_gap": (lead_ret1 - ret1[sym]).to_numpy(),
            "f_self_vol20": vol20[sym].to_numpy(),
            "f_volume_z": volume_z[sym].to_numpy(),
            "f_hour": idx.hour.astype(float),
            "label": label_raw,
            "_keep": (is_follower & same_day).to_numpy(),
        })
        feat = feat[feat["_keep"]].drop(columns="_keep")
        rows.append(feat)

    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=[*FEATURE_COLS, "label"]).reset_index(drop=True)
    if out.empty:
        return None
    out["label"] = out["label"].astype(int)
    return out[SCHEMA_COLS]


# ---------------------------------------------------------------------------------------
# Çoklu-sektör inşa
# ---------------------------------------------------------------------------------------
def build_dataset(
    sectors: dict[str, list[str]] | None = None, *,
    years: int = DEFAULT_YEARS, interval: str = DEFAULT_INTERVAL,
    fetch_fn: Callable[[str, int, str], pd.DataFrame | None] | None = None,
) -> pd.DataFrame:
    """Sektörleri gez, üyelerin barlarını çek, takipçi satırlarını üret.
    `fetch_fn` testlerde ağ'a çıkmadan sentetik veri enjekte etmek için override edilir."""
    fetch = fetch_fn or fetch_bars
    groups = sectors if sectors is not None else SECTORS
    frames_out: list[pd.DataFrame] = []
    cache: dict[str, pd.DataFrame | None] = {}

    for sector, members in groups.items():
        frames: dict[str, pd.DataFrame] = {}
        for sym in members:
            if sym not in cache:
                cache[sym] = fetch(sym, years, interval)
            df = cache[sym]
            if df is None or df.empty or "close" not in df.columns:
                logger.warning(f"[leadlag] {sym}: veri alınamadı/eksik — atlanıyor")
                continue
            frames[sym] = df
        if len(frames) < 2:
            logger.warning(f"[leadlag] {sector}: 2'den az kullanılabilir üye — sektör atlanıyor")
            continue
        feat = build_features_for_sector(frames, sector)
        if feat is None or feat.empty:
            logger.warning(f"[leadlag] {sector}: yetersiz ortak geçmiş — atlanıyor")
            continue
        logger.info(f"[leadlag] {sector}: {len(feat)} satır / {feat['symbol'].nunique()} takipçi")
        frames_out.append(feat)

    if not frames_out:
        return pd.DataFrame(columns=SCHEMA_COLS)
    out = pd.concat(frames_out, ignore_index=True)
    return out.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sektör-içi lead-lag aday veri seti (edge_lab #3)")
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--interval", default=DEFAULT_INTERVAL)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    logger.info(f"[leadlag] {len(SECTORS)} sektör / "
                f"{sum(len(v) for v in SECTORS.values())} sembol (years={args.years}, "
                f"interval={args.interval})")
    df = build_dataset(years=args.years, interval=args.interval)

    n_symbols = int(df["symbol"].nunique()) if not df.empty else 0
    n_rows = len(df)
    logger.info(f"[leadlag] toplandı: {n_symbols} takipçi sembol, {n_rows} satır")
    if n_symbols < TARGET_MIN_SYMBOLS:
        logger.warning(f"[leadlag] HEDEF ALTI: {n_symbols} sembol < {TARGET_MIN_SYMBOLS}")
    if n_rows < TARGET_MIN_ROWS:
        logger.warning(f"[leadlag] HEDEF ALTI: {n_rows} satır < {TARGET_MIN_ROWS}")
    if not df.empty:
        logger.info(f"[leadlag] taban oran (label=1): {df['label'].mean():.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"[leadlag] yazıldı: {out_path}")
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    raise SystemExit(main())
