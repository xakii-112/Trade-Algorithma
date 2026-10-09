"""
build_overnight_dataset.py — Overnight/intraday getiri ayrışması PİLOT veri seti inşası
(Tur-17, edge_lab İLK GERÇEK ADAY, BACKLOG.md #1).

PILOT: mevcut-likit 30 sembol = SURVIVORSHIP-ŞİŞKİN evren; PIT tam ölçüm ayrı iş.

Fikir: hisse getirisi iki ayrık rejime bölünebilir — gece (t kapanışından t+1 açılışına,
"overnight") ve gündüz (t açılışından t kapanışına, "intraday"). Literatürde bilinen bir
anomali overnight getirilerin intraday'den sistematik ayrıştığıdır. Bu script, `t` gününün
KAPANIŞINA KADAR bilinen özelliklerden (yalnız t ve öncesi barlar) `t+1` gecesi overnight
getirisinin İŞARETİNİ tahmin etmeye çalışan bir aday veri seti üretir; `research.edge_lab.
harness.run_candidate`'ın dört sert kapısından (purged CV → time+symbol çift-bölme → DSR
→ exec-lag-1) geçirilmek üzeredir. Bu script'in kendisi KENAR İDDİASI YAPMAZ — yalnız veri
taşır (CLAUDE.md ACI DERS #1: survivorship bias kenar uydurur; bu evren SURVIVORSHIP-ŞİŞKİN
olduğu için "kenar" sonucu bile burada nihai değildir).

Veri kaynağı: `nihai.data.alpaca_history.get_history` (SIP, ücretsiz planda 5y tarihsel;
.env'deki ALPACA_API_KEY/SECRET ile). Alpaca başarısız/boş dönerse yfinance'e (aynı sembol,
aynı yıl aralığı) düşülür; o da başarısızsa sembol atlanır (log uyarısı, koşu durmaz).
5 yıl veri yoksa (yeni listelenmiş sembol vb.) 2 yıla düşülür (`--years`/`FALLBACK_YEARS`).

Sembol listesi: .env `STREAM_SYMBOLS` (proje genelinde kullanılan ≤30 likit-sembol
konvansiyonu — bkz. `nihai/data/alpaca_stream_provider.py`) veya `--symbols` ile override.

Özellikler (TÜMÜ yalnız t gününün KAPANIŞINA kadarki barlardan — look-ahead yok, Madde 3):
  f_overnight_mean5 / f_overnight_mean20  — son 5/20 günün overnight-getiri ortalaması
                                             (overnight_t = (open_t - close_{t-1})/close_{t-1};
                                             open_t kapanış-t'den ÖNCE gerçekleşir → güvenli)
  f_intraday_mean5 / f_intraday_mean20    — son 5/20 günün intraday-getiri ortalaması
                                             (intraday_t = (close_t - open_t)/open_t)
  f_vol20                                 — son 20 günün günlük (close-close) getiri
                                             std sapması (oynaklık rejimi)
  f_volume_z                              — hacim z-skoru (son 20 gün ortalama/std'ye göre)
  f_dow                                   — gün-of-hafta (0=Pzt..4=Cuma), sayısal (one-hot DEĞİL)

Etiket (label, ikili): `t+1` GECESİ overnight getirisi — (open_{t+1} - close_t)/close_t —
maliyet eşiğinin (varsayılan 5 bps net) ÜSTÜNDE ise 1, aksi halde 0. Bu değer t'nin
KAPANIŞINDAN SONRA gerçekleşir → satırın ÖZELLİK kolonlarında YER ALMAZ, yalnız `label`
kolonunda bulunur (leak-free ayrımı `tests/test_edge_overnight_dataset.py` ile korunur).

CSV şeması (harness.run_candidate ile birebir): ts,symbol,f_overnight_mean5,
f_overnight_mean20,f_intraday_mean5,f_intraday_mean20,f_vol20,f_volume_z,f_dow,label

Kullanım:
    python -m research.edge_lab.build_overnight_dataset
    python -m research.edge_lab.build_overnight_dataset --symbols AAPL,MSFT --years 2
"""
from __future__ import annotations

import argparse
import os
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
COST_THRESHOLD = COST_THRESHOLD_BPS / 10_000.0     # 5 bps net eşik (etiket için)
DEFAULT_YEARS = 5
FALLBACK_YEARS = 2
MIN_HISTORY_DAYS = 20                              # en geniş rolling pencere (mean20/vol20)

FEATURE_COLS: list[str] = [
    "f_overnight_mean5", "f_overnight_mean20",
    "f_intraday_mean5", "f_intraday_mean20",
    "f_vol20", "f_volume_z", "f_dow",
]
SCHEMA_COLS: list[str] = ["ts", "symbol", *FEATURE_COLS, "label"]

DEFAULT_OUT = ROOT / "data" / "edge_overnight_pilot.csv"

# Hedefler (BEKLENTİ — ulaşılamazsa koşu durmaz, yalnız uyarı loglanır).
TARGET_MIN_SYMBOLS = 20
TARGET_MIN_ROWS = 15_000


def _stream_symbols() -> list[str]:
    """.env STREAM_SYMBOLS'tan sembol listesi (alpaca_stream_provider.py ile AYNI desen)."""
    raw = os.getenv("STREAM_SYMBOLS", "")
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------------------
# Veri çekimi — alpaca_history ÖNCE, yfinance FALLBACK (her ikisi de savunmacı: hata → None)
# ---------------------------------------------------------------------------------------
def _fetch_alpaca(symbol: str, years: int) -> pd.DataFrame | None:
    try:
        from nihai.data.alpaca_history import get_history
    except Exception as e:  # pragma: no cover - import-zamanı arıza
        logger.debug(f"[overnight_pilot] alpaca_history import edilemedi: {e}")
        return None
    end = datetime.now(UTC)
    start = end - timedelta(days=int(years * 365.25))
    try:
        df = get_history(symbol, "1d", start=start, end=end)
    except Exception as e:
        logger.warning(f"[overnight_pilot] {symbol} alpaca çekim hatası: {e}")
        return None
    if df is None or df.empty:
        return None
    return df


def _fetch_yfinance(symbol: str, years: int) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except Exception as e:  # pragma: no cover - opsiyonel bağımlılık
        logger.debug(f"[overnight_pilot] yfinance import edilemedi: {e}")
        return None
    try:
        raw = yf.Ticker(symbol).history(period=f"{years}y", interval="1d", auto_adjust=False)
    except Exception as e:
        logger.warning(f"[overnight_pilot] {symbol} yfinance çekim hatası: {e}")
        return None
    if raw is None or raw.empty:
        return None
    raw = raw.rename(columns={c: str(c).lower() for c in raw.columns})
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    return raw


def fetch_daily_bars(symbol: str, years: int = DEFAULT_YEARS) -> pd.DataFrame | None:
    """Günlük OHLCV: alpaca_history dene, boş/hatalıysa yfinance'e düş."""
    df = _fetch_alpaca(symbol, years)
    if df is not None and not df.empty:
        return df
    df = _fetch_yfinance(symbol, years)
    if df is not None and not df.empty:
        return df
    return None


# ---------------------------------------------------------------------------------------
# Özellik + etiket çıkarımı — yalnız df_slice (t'ye kadar) kullanılır; label t+1'e bakar
# ---------------------------------------------------------------------------------------
def build_features_for_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """Tek sembolün günlük OHLCV'sinden leak-free özellik + t+1 overnight-etiket satırları.

    df: DatetimeIndex, en az 'open'/'close' kolonları (varsa 'volume'). Yetersiz/eksik
    veri → None (çağıran sembolü atlar)."""
    if df is None or df.empty or not {"open", "close"}.issubset(df.columns):
        return None
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    prev_close = close.shift(1)

    overnight_ret = (open_ - prev_close) / prev_close          # t: (t-1 kapanış)->(t açılış)
    intraday_ret = (close - open_) / open_                      # t: (t açılış)->(t kapanış)
    daily_ret = close.pct_change()

    if "volume" in df.columns:
        volume = df["volume"].astype(float)
        vol_mean20 = volume.rolling(20).mean()
        vol_std20 = volume.rolling(20).std().replace(0.0, np.nan)
        volume_z = (volume - vol_mean20) / vol_std20
    else:
        volume_z = pd.Series(np.nan, index=df.index)

    # Etiket t+1 GECESİ overnight getirisi — GELECEK veri, YALNIZ label'da (leak-free ayrımı:
    # overnight_ret bir üst-satır (t) özelliklerinde ASLA .shift(-1) kullanılmaz). DİKKAT:
    # `NaN > eşik` pandas'ta NaN DEĞİL False döner — son satırda t+1 verisi yokken bunu
    # doğrudan .astype(float) yaparsak sahte label=0 SIZAR (dropna onu YAKALAYAMAZ, çünkü
    # artık NaN değil). Bu yüzden NaN'ı ÖNCE np.where ile açıkça koru, sonra dropna'ya bırak.
    future_overnight = overnight_ret.shift(-1)
    label_raw = np.where(future_overnight.isna(), np.nan,
                         (future_overnight > COST_THRESHOLD).astype(float))

    feat = pd.DataFrame({
        "ts": df.index,
        "symbol": symbol,
        "f_overnight_mean5": overnight_ret.rolling(5).mean(),
        "f_overnight_mean20": overnight_ret.rolling(20).mean(),
        "f_intraday_mean5": intraday_ret.rolling(5).mean(),
        "f_intraday_mean20": intraday_ret.rolling(20).mean(),
        "f_vol20": daily_ret.rolling(20).std(),
        "f_volume_z": volume_z,
        "f_dow": df.index.dayofweek.astype(float) if hasattr(df.index, "dayofweek") else 0.0,
        "label": label_raw,
    })
    # shift(-1) NaN olduğu son satır + rolling ısınma satırları (ilk MIN_HISTORY_DAYS-1) düşer.
    feat = feat.dropna(subset=[*FEATURE_COLS, "label"]).reset_index(drop=True)
    feat["label"] = feat["label"].astype(int)
    if feat.empty:
        return None
    return feat[SCHEMA_COLS]


# ---------------------------------------------------------------------------------------
# Çoklu-sembol inşa
# ---------------------------------------------------------------------------------------
def build_dataset(symbols: list[str], *, years: int = DEFAULT_YEARS,
                  fallback_years: int = FALLBACK_YEARS,
                  fetch_fn: Callable[[str, int], pd.DataFrame | None] | None = None) -> pd.DataFrame:
    """Sembol listesini gez, her biri için OHLCV çek (fetch_fn override edilebilir —
    testlerde ağ'a çıkmadan sentetik veri enjekte etmek için) + özellik/etiket üret.
    Başarısız/eksik-kolonlu/yetersiz-geçmişli semboller ATLANIR (koşu durmaz)."""
    fetch = fetch_fn or fetch_daily_bars
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = fetch(sym, years)
        if df is None or df.empty:
            df = fetch(sym, fallback_years)
        if df is None or df.empty:
            logger.warning(f"[overnight_pilot] {sym}: veri alınamadı — atlanıyor")
            continue
        if not {"open", "close"}.issubset(df.columns):
            logger.warning(f"[overnight_pilot] {sym}: open/close kolonu eksik — atlanıyor")
            continue
        feat = build_features_for_symbol(df, sym)
        if feat is None or len(feat) == 0:
            logger.warning(
                f"[overnight_pilot] {sym}: yetersiz geçmiş (min {MIN_HISTORY_DAYS}g gerekli) — atlanıyor"
            )
            continue
        frames.append(feat)
    if not frames:
        return pd.DataFrame(columns=SCHEMA_COLS)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=None, help="virgülle; boşsa .env STREAM_SYMBOLS")
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--fallback-years", type=int, default=FALLBACK_YEARS)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
              if args.symbols else _stream_symbols())
    if not symbols:
        logger.error("[overnight_pilot] sembol listesi boş (.env STREAM_SYMBOLS yok) — çıkılıyor")
        return 1

    logger.info(f"[overnight_pilot] {len(symbols)} sembol için pilot veri seti inşa ediliyor "
               f"(years={args.years}, fallback={args.fallback_years})")
    df = build_dataset(symbols, years=args.years, fallback_years=args.fallback_years)

    n_symbols = int(df["symbol"].nunique()) if not df.empty else 0
    n_rows = len(df)
    logger.info(f"[overnight_pilot] toplandı: {n_symbols} sembol, {n_rows} satır")
    if n_symbols < TARGET_MIN_SYMBOLS:
        logger.warning(f"[overnight_pilot] HEDEF ALTI: {n_symbols} sembol < {TARGET_MIN_SYMBOLS}")
    if n_rows < TARGET_MIN_ROWS:
        logger.warning(f"[overnight_pilot] HEDEF ALTI: {n_rows} satır < {TARGET_MIN_ROWS}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"[overnight_pilot] yazıldı: {out_path}")
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    raise SystemExit(main())
