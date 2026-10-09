"""
market_regime.py — Piyasa-geneli rejim tespiti → risk çarpanı (A4).

AI_trade'in DEFENSIVE/NEUTRAL/BULLISH rejim→risk-çarpanı mekanizmasının hisse
karşılığı. Kripto'da BTC trendi + getiri kullanılıyordu; burada benchmark (SPY/QQQ)
200/50-SMA trendi + son N-bar getirisi (+ isteğe bağlı VIX) kullanılır.

Look-ahead-güvenli: yalnızca verilen df'in KAPANMIŞ barları okunur (çağıran karar
barına kadarki dilimi geçmeli). Çıktı risk çarpanı RiskManager.regime_multiplier'a
yazılır ve adaptif drawdown freni (Madde 9) ile ÇARPIMSAL istiflenir.

VARSAYILAN KAPALI (CONFIG.regime.enable_regime_risk). Açmadan önce A/B doğrula.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nihai.config import CONFIG


@dataclass(frozen=True)
class RegimeResult:
    regime: str            # "DEFENSIVE" | "NEUTRAL" | "BULLISH"
    multiplier: float      # risk çarpanı
    detail: str            # insan-okur gerekçe (pano/log)


def compute_regime(bench_df: pd.DataFrame | None, *, vix: float | None = None) -> RegimeResult:
    """Benchmark bar df'inden (kapanmış barlar) rejim + risk çarpanı üret.

    Yetersiz veri / df yok → NEUTRAL (güvenli orta nokta). VIX yalnızca çağıran
    geçerse katkı yapar (None ise yok sayılır)."""
    cfg = CONFIG.regime
    if bench_df is None or bench_df.empty or "close" not in bench_df.columns:
        return RegimeResult("NEUTRAL", cfg.neutral_mult, "veri yok → NEUTRAL")

    c = bench_df["close"].dropna()
    if len(c) < cfg.sma_long + 1:
        return RegimeResult("NEUTRAL", cfg.neutral_mult,
                            f"yetersiz veri ({len(c)}<{cfg.sma_long}) → NEUTRAL")

    price = float(c.iloc[-1])
    sma_long = float(c.rolling(cfg.sma_long).mean().iloc[-1])
    sma_short = float(c.rolling(cfg.sma_short).mean().iloc[-1])
    n = cfg.recent_return_bars
    ref = float(c.iloc[-1 - n]) if len(c) > n else float(c.iloc[0])
    recent_ret = (price / ref - 1.0) * 100.0 if ref > 0 else 0.0

    vix_high = vix is not None and cfg.vix_high > 0 and vix >= cfg.vix_high
    vix_calm = vix is not None and cfg.vix_low > 0 and vix < cfg.vix_low

    # DEFENSIVE: trend kırık VEYA sert düşüş VEYA panik VIX
    if price < sma_long or recent_ret < cfg.defensive_return_pct or vix_high:
        why = []
        if price < sma_long:
            why.append(f"fiyat<SMA{cfg.sma_long}")
        if recent_ret < cfg.defensive_return_pct:
            why.append(f"{n}bar getiri {recent_ret:+.1f}%")
        if vix_high:
            why.append(f"VIX {vix:.0f}≥{cfg.vix_high:.0f}")
        return RegimeResult("DEFENSIVE", cfg.defensive_mult, "DEFENSIVE: " + ", ".join(why))

    # BULLISH: SMA dizilimi yukarı + pozitif momentum + panik yok
    if price > sma_long and sma_short > sma_long and recent_ret > 0 and not vix_high:
        tag = "BULLISH: trend yukarı, momentum +"
        if vix_calm:
            tag += f", VIX {vix:.0f}<{cfg.vix_low:.0f}"
        return RegimeResult("BULLISH", cfg.bullish_mult, tag)

    return RegimeResult("NEUTRAL", cfg.neutral_mult, "NEUTRAL: karışık sinyaller")


def regime_shadow(bench_df: pd.DataFrame | None) -> dict | None:
    """Rejimin OLASILIKSAL gölge okuması (Tur-37) — KARAR-DIŞI, `compute_regime`'in YANINDA.

    `compute_regime` rejimi KESKİN bir etiketle okur: bar ya "DEFENSIVE"dır ya değil. Bu
    yeterliydi çünkü risk çarpanı bir sayı ister. Ama rejim gözlenmez, ÇIKARSANIR; keskin
    etiket belirsizliği sessizce sıfıra yuvarlar ve rejim SINIRLARINDA (riskin tam da arttığı
    yerde) en çok yanılan okuma budur. `nihai/ai/regime_hmm.py` bu boşluğu kapatmak için
    yazılmıştı — ve üretimde HİÇ ÇAĞRILMIYORDU. Bu fonksiyon onun tek üretim çağırıcısıdır.

    NEDEN AYRI FONKSİYON, neden `compute_regime`'in İÇİNDE değil: `compute_regime`'in dönüşü
    `RiskManager.regime_multiplier`'a yazılır — yani KARAR YOLUDUR. Anayasa (`nihai/ai`
    sinyale/eşiğe bağlanmaz) yalnız değer kullanılmadığı için değil, KOD YOLU ayrı durduğu
    için de korunmalıdır: karar fonksiyonunun gövdesinde bir AI importu, sonraki düzenleyici
    için "buradan besleyebilirim" davetidir. Çarpanı üreten fonksiyonda tek satır AI yok.

    Dönüş: `observatory.bar_shadow` sözlüğü (hmm_mode_prob / hmm_switch_prob / hmm_label +
    mikroyapı ölçüleri) · ölçülemezse/kapalıysa None. Yalnız RAPOR katmanı okur (pano şeridi,
    günlük özet); hiçbir çarpan, eşik veya emir bu değere bakmaz."""
    if bench_df is None or bench_df.empty:
        return None
    try:
        from nihai.ai.observatory import bar_shadow
        return bar_shadow(bench_df, "BENCH")
    except Exception:      # noqa: BLE001 — gölge okuma rejim yolunu asla düşürmez
        return None


def beta_vs_benchmark(sym_close, bench_close, *, window: int = 60,
                      min_obs: int = 30) -> float | None:
    """Sembolün benchmark'a göre OLS betası — ölçülemezse **None** (BULGU-59-6).

    beta = Cov(r_sym, r_bench) / Var(r_bench), ORTAK zaman damgalarında hizalanmış
    log-getirilerin son `window` gözlemi üzerinden. Nedenseldir (yalnız verilen serilerin
    geçmişi) ve iki seri farklı takvimlerde olabilir → indeks KESİŞİMİ alınır.

    **None DÖNMESİ bir kusur değil sözleşmedir** (Tur-31/Tur-34: ölçüm yokluğu ölçüm gibi
    sunulmaz). 1.0 uydurmak portföy beta maruziyetini sessizce "piyasa kadar" gösterirdi;
    çağıran ölçülmemiş pozisyonları AYRI sayar."""
    try:
        a = pd.Series(sym_close).astype(float).dropna()
        b = pd.Series(bench_close).astype(float).dropna()
    except (TypeError, ValueError):
        return None
    if a.empty or b.empty:
        return None
    common = a.index.intersection(b.index)
    if len(common) < min_obs + 1:
        return None
    ra = np.diff(np.log(np.clip(a.loc[common].to_numpy(dtype=float), 1e-12, None)))
    rb = np.diff(np.log(np.clip(b.loc[common].to_numpy(dtype=float), 1e-12, None)))
    ra, rb = ra[-window:], rb[-window:]
    ok = np.isfinite(ra) & np.isfinite(rb)
    ra, rb = ra[ok], rb[ok]
    if ra.size < min_obs:
        return None
    var_b = float(np.var(rb, ddof=1))
    if not np.isfinite(var_b) or var_b <= 0:
        return None          # benchmark DÜZ → beta tanımsız (Tur-28 dejenere-girdi dersi)
    cov = float(np.cov(ra, rb, ddof=1)[0, 1])
    beta = cov / var_b
    return beta if np.isfinite(beta) else None


def regime_multiplier_series(bench_df: pd.DataFrame | None) -> pd.Series:
    """Backtest için: benchmark df'inin HER barı için rejim risk çarpanı (vektörel).
    compute_regime ile AYNI mantık (VIX'siz) ama tüm seriye uygulanır → bench.py ile
    A4'ü A/B etmek mümkün olur. Look-ahead: her bar yalnızca kendine kadarki kapanışları
    kullanır (rolling SMA/getiri); çağıran (backtest) ayrıca 1-bar GECİKMEYLE uygular."""
    cfg = CONFIG.regime
    if bench_df is None or bench_df.empty or "close" not in bench_df.columns:
        return pd.Series(dtype=float)
    c = bench_df["close"].astype(float)
    sma_l = c.rolling(cfg.sma_long).mean()
    sma_s = c.rolling(cfg.sma_short).mean()
    ret = (c / c.shift(cfg.recent_return_bars) - 1.0) * 100.0

    mult = pd.Series(cfg.neutral_mult, index=c.index)
    bullish = (c > sma_l) & (sma_s > sma_l) & (ret > 0)
    defensive = (c < sma_l) | (ret < cfg.defensive_return_pct)
    mult[bullish] = cfg.bullish_mult
    mult[defensive] = cfg.defensive_mult        # defensive, bullish'i ezer (compute_regime ile aynı)
    mult[sma_l.isna()] = cfg.neutral_mult        # yeterli veri yok → nötr
    return mult
