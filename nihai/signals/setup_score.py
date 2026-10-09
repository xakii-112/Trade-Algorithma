"""
setup_score.py — Canlı "kurulum kalitesi" skoru (B7) + benchmark'a görel güç (B8).

AI_trade tarayıcısının çok-boyutlu skorlamasının hisse karşılığı. Backtest skoru
"geçmişte iyiydi" der; bu skor "şu an kurulum taze mi" der (trend, ADX, RSI bölgesi,
MA mesafesi, aşırı-uzama, geri-çekilme, hacim, ve SPY'a görel güç). auto_screener
ikisini HİBRİT birleştirir (SCREEN_HYBRID_ENABLED) — varsayılan KAPALI.

⚠️ OVERFIT UYARISI: ağırlıklar AI_trade'den uyarlanmış EL-AYARI değerlerdir. Hisse
evreninde walk-forward ile kalibre edilmeden "alfa" sayma. Bu modül yalnızca skoru
ÜRETİR; sıralamaya etkisi config bayrağıyla kapılıdır (bkz. docs/EDGE_FINDINGS.md).

Look-ahead-güvenli: yalnızca verilen df'in son KAPANMIŞ barına kadarki veri okunur.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from nihai.signals.strategy import adx, rsi


def _ext_enabled() -> bool:
    """Madde 2 — ek-göstergeleri skora kat (varsayılan KAPALI)."""
    return os.getenv("SETUP_EXT_INDICATORS_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on")


def _ext_weight() -> float:
    """Ek-gösterge bileşenlerinin toplu ağırlığı (agresif test için yükseltilebilir)."""
    try:
        return float(os.getenv("SETUP_EXT_WEIGHT", "1.0"))
    except ValueError:
        return 1.0


def _extended_components(df: pd.DataFrame) -> dict[str, float]:
    """Madde 2 — TAMAMLAYICI (kolinear OLMAYAN) ek-gösterge bileşenleri.

    Trend/RSI/ADX'i TEKRARLAMAZ; üç ortogonal sinyal ekler:
      • TD Sequential  → dönüş zamanlaması (boğa/ayı oyu)
      • Force Index    → hacim-fiyat akış yönü
      • VHF            → trend vs range rejimi

    ⚠️ EDGE_FINDINGS ilkesi: daha çok gösterge = daha çok edge DEĞİLDİR. Ağırlıklar
    el-ayarıdır; canlı sermayeye açmadan ÖNCE research/ext_indicator_eval.py ile
    walk-forward OOS'ta doğrula. Agresif ağırlık optimizasyonu in-sample'ı şişirir,
    OOS'ta deflate olur (PEAD/meta/momentum'da görülen desen)."""
    comp: dict[str, float] = {}
    try:
        from nihai.signals.indicators_ext import force_index, td_sequential, vhf
    except Exception:
        return comp
    try:
        comp["ext_td"] = float(td_sequential(df).get("vote", 0)) * 8.0
    except Exception:
        pass
    try:
        fi = force_index(df, 13).dropna()
        if len(fi) >= 5:
            last, prev = float(fi.iloc[-1]), float(fi.iloc[-5])
            comp["ext_force"] = (6.0 if last > 0 and last >= prev else
                                 -6.0 if last < 0 and last <= prev else 0.0)
    except Exception:
        pass
    try:
        v = float(vhf(df["close"], 28).dropna().iloc[-1])
        comp["ext_vhf"] = 5.0 if v >= 0.40 else (-5.0 if v <= 0.25 else 0.0)
    except Exception:
        pass
    return comp


@dataclass(frozen=True)
class SetupScore:
    score: float
    components: dict = field(default_factory=dict)
    rs_short: float | None = None   # B8 — sembol−benchmark getirisi (kısa ufuk), %
    rs_long: float | None = None    # B8 — (uzun ufuk), %


def _pct_return(close: pd.Series, bars: int) -> float | None:
    """Son `bars` bar getirisi (%). Yetersiz veri → None."""
    if len(close) <= bars:
        return None
    prev = float(close.iloc[-1 - bars])
    if prev <= 0:
        return None
    return (float(close.iloc[-1]) / prev - 1.0) * 100.0


def benchmark_returns(bench_df: pd.DataFrame | None,
                      horizons: tuple[int, int] = (20, 60)) -> dict[int, float]:
    """Benchmark (SPY) için ufuk→getiri(%) sözlüğü. Tarama başına BİR kez hesaplanıp
    her sembol skorlamasına küçük bir dict olarak geçirilir (df pickle'lamaktan kaçınır)."""
    out: dict[int, float] = {}
    if bench_df is None or bench_df.empty or "close" not in bench_df.columns:
        return out
    c = bench_df["close"].dropna()
    for h in horizons:
        r = _pct_return(c, h)
        if r is not None:
            out[h] = r
    return out


def compute_setup_score(df: pd.DataFrame | None,
                        bench_returns: dict[int, float] | None = None,
                        *, runup_bars: int = 20,
                        rs_horizons: tuple[int, int] = (20, 60)) -> SetupScore:
    """Bar df'inden kurulum kalitesi skoru üret. Bileşenler ~[-100, +100] aralığında
    toplanır (sınır kesin değil). bench_returns verilirse RS bileşenleri eklenir (B8).

    İKİ AYRI ufuk (kasıtlı): runup_bars KISA (son aşırı-uzama/geri-çekilme tazeliği),
    rs_horizons UZUN (1–3 aylık görel güç — dokümante momentum ufku). Karıştırmak RS'i
    3-günlük gürültüye çevirirdi. Çağıran (auto_screener) bunları işlem-günü→bar çevirir."""
    if df is None or df.empty or "close" not in df.columns or len(df) < 30:
        return SetupScore(0.0, {"reason": "yetersiz_veri"})

    c = df["close"].dropna()
    price = float(c.iloc[-1])
    comp: dict[str, float] = {}

    # Trend: fiyat uzun SMA'nın üstünde mi (veri yetiyorsa 200, yoksa 50)
    sma_long_p = 200 if len(c) >= 200 else 50
    sma_long = float(c.rolling(sma_long_p).mean().iloc[-1])
    comp["trend"] = 20.0 if price > sma_long else -20.0

    # ADX — trend gücü
    try:
        adx_v = float(adx(df).iloc[-1])
    except Exception:
        adx_v = 0.0
    comp["adx"] = 20.0 if adx_v >= 25 else (8.0 if adx_v >= 18 else -10.0)

    # RSI bölgesi — aşırı-alımı cezalandır, sağlıklı momentumu ödüllendir
    try:
        rsi_v = float(rsi(c).iloc[-1])
    except Exception:
        rsi_v = 50.0
    if 45 <= rsi_v <= 68:
        comp["rsi"] = 15.0
    elif 68 < rsi_v <= 74:
        comp["rsi"] = 3.0
    elif rsi_v > 74:
        comp["rsi"] = -15.0
    elif rsi_v < 40:
        comp["rsi"] = -10.0
    else:
        comp["rsi"] = 0.0

    # MA mesafesi (SMA50) — fazla uzamış girişleri cezalandır
    sma50 = float(c.rolling(min(50, len(c))).mean().iloc[-1])
    dist_pct = (price / sma50 - 1.0) * 100.0 if sma50 > 0 else 0.0
    comp["distance"] = (10.0 if abs(dist_pct) <= 3 else
                        4.0 if abs(dist_pct) <= 6 else
                        -10.0 if abs(dist_pct) >= 9 else 0.0)

    # Aşırı-uzama: son `runup_bars` bar koşusu çok dikse cezalandır (KISA ufuk)
    run = _pct_return(c, runup_bars)
    if run is not None:
        comp["runup"] = -10.0 if run > 12 else (5.0 if 0 <= run <= 6 else 0.0)

    # Geri-çekilme: son `runup_bars`-bar tepeden makul geri-çekilme (taze giriş) iyi
    n = min(runup_bars, len(df))
    hi = float(df["high"].iloc[-n:].max()) if "high" in df.columns else price
    pull = (hi - price) / hi * 100.0 if hi > 0 else 0.0
    comp["pullback"] = 8.0 if pull <= 3 else (-8.0 if pull >= 8 else 0.0)

    # Hacim oranı — sağlıklı katılım
    if "volume" in df.columns and len(df) >= 20:
        vma = float(df["volume"].iloc[-20:].mean())
        vr = float(df["volume"].iloc[-1]) / vma if vma > 0 else 1.0
        comp["volume"] = 8.0 if 0.8 <= vr <= 2.0 else (-8.0 if vr > 3 else 0.0)

    # B8 — benchmark'a görel güç (RS): sembol getirisi − SPY getirisi (UZUN ufuk)
    rs_short = rs_long = None
    if bench_returns:
        sym_s, sym_l = _pct_return(c, rs_horizons[0]), _pct_return(c, rs_horizons[1])
        if sym_s is not None and rs_horizons[0] in bench_returns:
            rs_short = sym_s - bench_returns[rs_horizons[0]]
            comp["rs_short"] = max(-25.0, min(rs_short * 2.0, 25.0))
        if sym_l is not None and rs_horizons[1] in bench_returns:
            rs_long = sym_l - bench_returns[rs_horizons[1]]
            comp["rs_long"] = max(-30.0, min(rs_long * 2.5, 40.0))

    # Madde 2 — ek göstergeler (flag ile, varsayılan KAPALI → mevcut çıktı değişmez)
    if _ext_enabled():
        w = _ext_weight()
        for k, val in _extended_components(df).items():
            comp[k] = round(val * w, 2)

    return SetupScore(round(sum(comp.values()), 2), comp, rs_short, rs_long)
