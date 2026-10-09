"""
iv_history.py — Günlük ATM IV cache + rolling IV Rank.

PDF Referansları:
- Madde 16 : IV Rank / Percentile (gerçek hesaplama)

yfinance opsiyon tarihsel IV vermiyor. Bu modül her gün ATM straddle IV'yi
data/iv_history/<SYM>.csv'ye ekler; zamanla 52w rolling oluşur.

Backfill stratejisi: Sıfırdan başlandığında ilk 30+ iş günü "LOW_CONFIDENCE" rank
döner; 100+ gün sonra "PARTIAL"; 252+ gün (~52w) sonra "OK". Polygon hesabı
varsa backfill scripti ile geriye dönük tarihsel IV serisi de doldurulabilir.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from nihai.config import DATA_DIR

IV_HIST_DIR = DATA_DIR / "iv_history"
IV_HIST_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_WINDOW_DAYS = 252      # ~52 hafta trading day
MIN_DAYS_OK = 100
MIN_DAYS_PARTIAL = 30


def _path(symbol: str) -> Path:
    return IV_HIST_DIR / f"{symbol.upper()}.csv"


@dataclass
class IVRankResult:
    iv_rank: float            # 0-100
    confidence: str           # OK | PARTIAL | LOW_CONFIDENCE | NO_HISTORY | NO_RANGE
    sample_size: int
    # ÖLÇÜLDÜ mü, yoksa 50.0 UYDURULDU mu? Ölçüm yokluğunda dönen 50.0 gerçek bir rank'ten
    # ayırt edilemiyordu ve `strategy.signal_quality_score`'un [30,50] bandının TAM İÇİNDE
    # kalıyordu → IV geçmişi olmayan her sinyal +0.5 bonusu KOŞULSUZ aldı (ölçüm yokluğu
    # ödüllendirildi). Skorlama yoluna giren her tüketici bu bayrağı sormalı.
    # Varsayılan False = güvenli yön (bonusu hak etmeden verme).
    measured: bool = False


def append_iv_today(symbol: str, atm_iv: float) -> None:
    """Bugünün ATM IV'sini cache'e ekle (idempotent — aynı gün üst üste yazılır)."""
    if atm_iv <= 0 or atm_iv > 5.0:        # absürt değerleri at
        logger.debug(f"[ivhist] gecersiz IV {symbol}: {atm_iv}")
        return
    p = _path(symbol)
    today_iso = date.today().isoformat()
    if p.exists():
        df = pd.read_csv(p)
        if (df["date"] == today_iso).any():
            df.loc[df["date"] == today_iso, "iv"] = atm_iv
        else:
            df = pd.concat([df, pd.DataFrame([{"date": today_iso, "iv": atm_iv}])],
                           ignore_index=True)
    else:
        df = pd.DataFrame([{"date": today_iso, "iv": atm_iv}])
    df.sort_values("date", inplace=True)
    df.to_csv(p, index=False)


def get_iv_rank(symbol: str, current_iv: float) -> IVRankResult:
    """Persistent cache üzerinden rolling 52w IV rank hesapla."""
    p = _path(symbol)
    if not p.exists():
        return IVRankResult(50.0, "NO_HISTORY", 0)
    try:
        df = pd.read_csv(p).tail(ROLLING_WINDOW_DAYS)
    except Exception as e:
        logger.debug(f"[ivhist] CSV okuma hatasi {symbol}: {e}")
        return IVRankResult(50.0, "NO_HISTORY", 0)

    n = len(df)
    if n < MIN_DAYS_PARTIAL:
        return IVRankResult(50.0, f"LOW_CONFIDENCE_{n}", n)

    lo, hi = float(df["iv"].min()), float(df["iv"].max())
    if hi - lo < 1e-6:
        return IVRankResult(50.0, "NO_RANGE", n)

    rank = float(np.clip((current_iv - lo) / (hi - lo) * 100, 0, 100))
    conf = "OK" if n >= MIN_DAYS_OK else f"PARTIAL_{n}"
    return IVRankResult(rank, conf, n, measured=True)


def history_stats(symbol: str) -> dict:
    """Dahili kullanım: cache durumunu raporla."""
    p = _path(symbol)
    if not p.exists():
        return {"symbol": symbol, "days": 0, "min": None, "max": None}
    df = pd.read_csv(p)
    return {
        "symbol": symbol,
        "days": len(df),
        "min": float(df["iv"].min()),
        "max": float(df["iv"].max()),
        "first_date": df["date"].iloc[0] if len(df) else None,
        "last_date": df["date"].iloc[-1] if len(df) else None,
    }
