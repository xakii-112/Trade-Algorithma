"""
spread_estimate.py — Etkin bid-ask spread'inin BARDAN tahmini (F3, 2026-07-26 planı).

Neden: canlı tahminlerde medyan mutlak hareket **~13,5 bps**. Bu isimlerde etkin spread
2–5 bps mertebesindedir → hareketin %15–40'ı. Şu ana kadar hiçbir ölçümde modellenmedi;
dolayısıyla bugünkü %46,8 isabet ve −0,0072% getiri bile **iyimser**. Bu modül, tarihsel
quote verisi OLMADAN (ücretsiz plan) yalnız OHLC'den etkin spread tahmini üretir.

İki klasik tahminci:
  * **Corwin–Schultz (2012)** — ardışık iki barın high/low aralığından. Fiyat aralığı hem
    gerçek oynaklığı hem spread'i içerir; iki-bar penceresinde oynaklık zamanla ölçeklenir,
    spread ölçeklenmez → ayrıştırılabilir.
  * **Roll (1984)** — ardışık fiyat değişimlerinin NEGATİF otokovaryansından. Bid-ask
    sıçraması (bid'de al / ask'te sat) getirilerde yapay negatif otokorelasyon yaratır.

DÜRÜSTLÜK NOTU: ikisi de *tahmincidir*, gerçek spread değildir. Corwin–Schultz gün-içi
1m barlarda genelde YUKARI yanlı (aralık gürültüsü spread'e yazılır), Roll ise trendli
seride negatif kovaryans bulamayıp 0 dönebilir (AŞAĞI yanlı). Bu yüzden `estimate()`
ikisini de raporlar ve maliyet kapısında **muhafazakâr olan (büyük olan)** kullanılabilir
(`method="max"`). Maliyeti KÜÇÜK göstermek kenar uydurmanın en kolay yoludur (ACI DERS #7).

Seans sınırı: gece boşluğu (overnight gap) iki-bar penceresini şişirir → farkı bar
aralığının 2 katından büyük olan ardışık çiftler ATILIR.

Kullanım:
    python -m research.spread_estimate --symbols AAPL,MSFT --interval 1m --days 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from nihai.console import ensure_utf8_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 3 − 2√2 (Corwin–Schultz sabiti)
_K = 3.0 - 2.0 * np.sqrt(2.0)


def _pair_mask(index: pd.Index, interval_min: float) -> np.ndarray:
    """Ardışık (t, t+1) çiftlerinden AYNI seansta olanların maskesi.

    Gece/hafta sonu boşluğu Corwin–Schultz'un γ terimini şişirir ve spread'i olduğundan
    büyük gösterir. Bar aralığının 2 katından uzun boşluklar elenir."""
    if len(index) < 2:
        return np.zeros(0, dtype=bool)
    ts = pd.to_datetime(pd.Series(index))
    gap_min = ts.diff().dt.total_seconds().to_numpy()[1:] / 60.0
    return np.isfinite(gap_min) & (gap_min <= 2.0 * interval_min)


def corwin_schultz(df: pd.DataFrame, interval_min: float = 1.0) -> np.ndarray:
    """Bar-çifti başına Corwin–Schultz etkin spread'i (fiyatın KESRİ olarak).

    S = 2(e^α − 1)/(1 + e^α),  α = (√(2β) − √β)/(3−2√2) − √(γ/(3−2√2))
    β = ln(H_t/L_t)² + ln(H_{t+1}/L_{t+1})²,  γ = ln(max H / min L)²
    Negatif α → S negatif çıkar; literatür konvansiyonu 0'a kırpmaktır (spread negatif olamaz).
    """
    if df is None or len(df) < 2 or not {"high", "low"} <= set(df.columns):
        return np.zeros(0, dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    ok = np.isfinite(h) & np.isfinite(l) & (h > 0) & (l > 0)
    h = np.where(ok, h, np.nan)
    l = np.where(ok, l, np.nan)

    hl = np.log(h / l) ** 2
    beta = hl[:-1] + hl[1:]
    h2 = np.maximum(h[:-1], h[1:])
    l2 = np.minimum(l[:-1], l[1:])
    gamma = np.log(h2 / l2) ** 2

    with np.errstate(invalid="ignore"):
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = np.where(np.isfinite(s), s, np.nan)
    s = np.clip(s, 0.0, None)                       # negatif tahmin → 0 (konvansiyon)
    mask = _pair_mask(df.index, interval_min)
    if mask.size == s.size:
        s = np.where(mask, s, np.nan)
    return s


def roll(df: pd.DataFrame, interval_min: float = 1.0) -> float:
    """Roll (1984) etkin spread'i (fiyatın kesri). Cov(Δp_t, Δp_{t−1}) ≥ 0 → 0 döner.

    Seans-içi ardışıklık şartı: gece boşluğunu içeren getiri çiftleri elenir."""
    if df is None or len(df) < 4 or "close" not in df.columns:
        return 0.0
    c = df["close"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(np.where(c > 0, c, np.nan)))
    same = _pair_mask(df.index, interval_min)       # len = len(df)-1, r ile hizalı
    if same.size == r.size:
        r = np.where(same, r, np.nan)
    r0, r1 = r[1:], r[:-1]
    ok = np.isfinite(r0) & np.isfinite(r1)
    if ok.sum() < 30:
        return 0.0
    cov = float(np.cov(r0[ok], r1[ok])[0, 1])
    return float(2.0 * np.sqrt(-cov)) if cov < 0 else 0.0


def estimate(df: pd.DataFrame, interval_min: float = 1.0) -> dict[str, float]:
    """Bir sembolün etkin spread tahmini — bps cinsinden (1 bps = 0,01%).

    `corwin` = bar-çifti tahminlerinin MEDYANI (ortalama uç değerlerden bozulur),
    `roll`   = tüm örneklemden tek tahmin, `max` = ikisinin büyüğü (muhafazakâr)."""
    cs = corwin_schultz(df, interval_min)
    cs_med = float(np.nanmedian(cs)) if cs.size and np.isfinite(cs).any() else 0.0
    rl = roll(df, interval_min)
    return {
        "corwin_bps": round(cs_med * 1e4, 3),
        "roll_bps": round(rl * 1e4, 3),
        "max_bps": round(max(cs_med, rl) * 1e4, 3),
        "n_bar": int(len(df)) if df is not None else 0,
    }


def spread_table(symbols: list[str], interval: str = "1m", days: int = 30,
                 interval_min: float = 1.0) -> dict[str, dict[str, float]]:
    """Sembol → spread tahmini tablosu (alpaca_history üzerinden, cache'li)."""
    from nihai.data.alpaca_history import get_history
    start = datetime.now(UTC) - timedelta(days=days)
    out: dict[str, dict[str, float]] = {}
    for sym in symbols:
        try:
            df = get_history(sym, interval, start=start)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001 — tek sembol hatası tabloyu düşürmesin
            out[sym] = {"corwin_bps": 0.0, "roll_bps": 0.0, "max_bps": 0.0, "n_bar": 0.0}
            print(f"[spread] {sym} çekilemedi: {e}", file=sys.stderr)
            continue
        if df is None or df.empty:
            out[sym] = {"corwin_bps": 0.0, "roll_bps": 0.0, "max_bps": 0.0, "n_bar": 0.0}
            continue
        out[sym] = estimate(df, interval_min)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OHLC'den etkin spread tahmini (Corwin–Schultz / Roll)")
    ap.add_argument("--symbols", default="AAPL,MSFT,NVDA", help="virgülle ayrık")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--interval-min", type=float, default=1.0, help="bar aralığı (dakika)")
    a = ap.parse_args(argv)
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    tbl = spread_table(syms, a.interval, a.days, a.interval_min)
    vals = [v.get("max_bps", 0.0) for v in tbl.values() if v.get("n_bar", 0)]
    print(json.dumps({
        "semboller": tbl,
        "medyan_max_bps": round(float(np.median(vals)), 3) if vals else None,
        "NOT": ("Bunlar TAHMİNCİDİR. Corwin–Schultz 1m barda yukarı, Roll trendli seride "
                "aşağı yanlıdır; maliyet kapısında muhafazakâr (büyük) olanı kullan."),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    raise SystemExit(main())
