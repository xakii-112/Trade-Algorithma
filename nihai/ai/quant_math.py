"""
quant_math.py — Derin FİNANS MATEMATİĞİ kütüphanesi (AI v2 §1, Tur-13 WP-A). KARAR-DIŞI.

Saf fonksiyonlar: yan etkisiz, IO/ağ YOK; numpy/scipy/scikit-learn (+ opsiyonel-guard'lı
statsmodels) dışında bağımlılık yok. `nihai/ai/decision.py` (GÖLGE karar motoru) ve
`nihai/ai/knowledge.py` bu modülü salt-okunur biçimde referanslar; BURADAN hiçbir üretim
sinyaline/eşiğe/META_GATE'e bağlanmaz (bkz. AI_V2_BEYIN_SPEC.md "Değişmez anayasa").

Savunmacı sözleşme (spec §1, madde 4):
  - Girdi kısa/dönüştürülemez/sonlu-olmayan (NaN/Inf) İSE: dönüş tipi `X | None` olan
    fonksiyonlar None döner. Dönüş tipi None İÇERMEYEN yapısal dönüşümler (simple_returns/
    log_returns/zscore, mc_gbm_paths, mc_bootstrap_terminal, classify_regime) KENDİ
    tipinde zararsız bir değerle (boş dizi / None-alanlı dict / {"label":"unknown"}) döner.
    Hiçbir girdi İSTİSNA fırlatmaz.
  - Her stokastik fonksiyon (mc_gbm_paths, mc_bootstrap_terminal) ZORUNLU `seed` alır ve
    DETERMİNİSTİKTİR (aynı seed → bit-bit aynı çıktı; np.random.default_rng kullanılır).
  - "Per-bar" diye işaretli oynaklık tahmincileri (vol_*, ewma_vol) YILLIKLANDIRILMAMIŞTIR;
    yıllıklandırma için ayrıca `annualize_vol` çağrılır.

Kaynak/konvansiyonlar: Parkinson (1980), Garman-Klass (1980), Yang-Zhang (2000), RiskMetrics
EWMA (λ=0.94 varsayılan), Lo-MacKinlay variance ratio, Engle-Granger (1987) iki-adımlı
kointegrasyon testi, pairs-trading literatüründeki OU yarı-ömrü/Kalman-beta yaklaşımları,
Cornish-Fisher VaR genişlemesi, Black-Scholes (1973) kapalı-form (kendi implementasyonu —
py_vollib'e bağımlı DEĞİL; py_vollib yalnız test dosyasında çapraz-doğrulama için kullanılır).
"""
from __future__ import annotations

import math
from functools import lru_cache
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import lfilter as _lfilter
from scipy.signal import welch as _welch
from scipy.spatial import cKDTree

# sklearn LAZY (Tur-26): `LedoitWolf`/`PCA` bu dosyada YALNIZ `ledoit_wolf_cov` ve
# `pca_risk` içinde kullanılıyor, ama modül-düzeyi import'ları `import nihai.ai`
# zincirinin 1,3 sn'sini tek başına oluşturuyordu (ölçüm: `python -X importtime`).
# Fonksiyon-içi import CPython'da sys.modules'tan gelir → ikinci çağrıdan sonra bedava.

try:
    from statsmodels.tsa.stattools import adfuller as _adfuller
except ImportError:  # pragma: no cover - opsiyonel bağımlılık (anayasa madde 3: guard'lı)
    _adfuller = None


# ---------------------------------------------------------------------------
# Girdi temizleme yardımcıları (modül-içi; dışa açık API değil)
# ---------------------------------------------------------------------------
def _as_1d(x: ArrayLike) -> np.ndarray:
    """Gevşek 1B float dönüşüm: çevrilemezse boş dizi (İSTİSNA YOK). NaN/Inf'e DOKUNULMAZ —
    yalnız None döndürmeyen yapısal dönüşümlerde (simple_returns/log_returns/zscore/
    classify_regime) kullanılır; onlar için kısalık dışında sessiz veri reddi uygun değildir."""
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)
    return arr


def _clean_1d(x: ArrayLike, min_len: int = 1) -> np.ndarray | None:
    """Katı 1B float dönüşüm: kısa/dönüştürülemez/sonlu-olmayan (NaN/Inf) içerik → None.
    `X | None` dönen istatistiksel tahmincilerin tümünde kullanılır (spec: "kısa/NaN'lı
    girdi → None")."""
    arr = _as_1d(x)
    if arr.size < min_len:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _clean_2d(x: ArrayLike, min_rows: int = 2, min_cols: int = 1) -> np.ndarray | None:
    """Katı 2B float dönüşüm (kovaryans/PCA girdileri: satır=zaman barı, sütun=varlık)."""
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < min_rows or arr.shape[1] < min_cols:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


# ---------------------------------------------------------------------------
# Getiriler / normalizasyon
# ---------------------------------------------------------------------------
def simple_returns(prices: ArrayLike) -> np.ndarray:
    """Basit getiri serisi: r[i] = p[i]/p[i-1] - 1. <2 fiyat → boş dizi (çökme yok).
    Sıfır/negatif fiyat elemanları o noktada nan/inf üretebilir (dizi kırpılmaz)."""
    arr = _as_1d(prices)
    if arr.size < 2:
        return np.array([], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return arr[1:] / arr[:-1] - 1.0


def log_returns(prices: ArrayLike) -> np.ndarray:
    """Log getiri serisi: r[i] = ln(p[i]/p[i-1]). <2 fiyat → boş dizi (çökme yok)."""
    arr = _as_1d(prices)
    if arr.size < 2:
        return np.array([], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(arr[1:] / arr[:-1])


def annualize_return(mean_ret: float, ppy: float, *, log: bool = False) -> float:
    """Bar-başı ortalama getiriyi BİLEŞİK olarak yıllıklandırır: (1+mean_ret)**ppy - 1.
    `log=True`: girdi LOG getiri ortalamasıdır → exp(ppy·mean_ret) − 1. Log girdiye basit
    formülü uygulamak ikinci dereceden sapar (Tur-78, R2-B2: günlük 0,005'te −1,1 pp).
    Geçersiz girdi (ppy<=0, sonlu-değil) → 0.0; taban negatifse (mean_ret<-1, kesirli üsle
    tanımsız) → nan. Hiçbir durumda istisna fırlatmaz (dönüş tipi her zaman float)."""
    try:
        mr, p = float(mean_ret), float(ppy)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(mr) or not math.isfinite(p) or p <= 0:
        return 0.0
    if log:
        try:
            return float(math.expm1(p * mr))
        except OverflowError:
            return float("nan")
    base = 1.0 + mr
    if base < 0:
        return float("nan")
    try:
        return float(base**p - 1.0)
    except (OverflowError, ValueError):
        return float("nan")


def annualize_vol(vol: float, ppy: float) -> float:
    """Bar-başı volatiliteyi karekök-zaman kuralıyla yıllıklandırır: vol * sqrt(ppy).
    Geçersiz girdi (vol<0, ppy<=0, sonlu-değil) → 0.0 (çökme yok)."""
    try:
        v, p = float(vol), float(ppy)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v) or not math.isfinite(p) or v < 0 or p <= 0:
        return 0.0
    return float(v * math.sqrt(p))


def zscore(x: ArrayLike, window: int | None = None) -> np.ndarray:
    """Standardize edilmiş seri. window=None → tam-örneklem z-skoru (popülasyon std, ddof=0).
    window verilirse HER nokta yalnız KENDİSİNE KADARKİ `window`-barlık pencereyle hesaplanır
    (look-ahead yok); ilk window-1 nokta nan. Sıfır dağılım (std=0) → 0.0 (0/0 tanımsızlığı
    yerine "fark yok" okunur). Boş/kısa girdi → uygun şekilde boş/nan dizi (çökme yok)."""
    arr = _as_1d(x)
    if arr.size == 0:
        return np.array([], dtype=float)
    if window is None:
        mu = float(np.nanmean(arr))
        sd = float(np.nanstd(arr))
        if sd == 0.0 or not math.isfinite(sd):
            return np.zeros_like(arr)
        return (arr - mu) / sd
    w = int(window)
    if w < 2 or w > arr.size:
        return np.full(arr.shape, np.nan)
    out = np.full(arr.shape, np.nan)
    for i in range(w - 1, arr.size):
        seg = arr[i - w + 1 : i + 1]
        mu = float(np.nanmean(seg))
        sd = float(np.nanstd(seg))
        out[i] = 0.0 if (sd == 0.0 or not math.isfinite(sd)) else (seg[-1] - mu) / sd
    return out


# ---------------------------------------------------------------------------
# Oynaklık tahmincileri (per-bar; YILLIKLANDIRILMAMIŞ)
# ---------------------------------------------------------------------------
def vol_close_to_close(close: ArrayLike) -> float | None:
    """Kapanıştan-kapanışa gerçekleşen volatilite: log getirilerin örneklem std'si (ddof=1).
    En az 3 fiyat (2 getiri) ister; azı/NaN'lı → None."""
    arr = _clean_1d(close, min_len=3)
    if arr is None:
        return None
    r = log_returns(arr)
    if r.size < 2 or not np.all(np.isfinite(r)):
        return None
    return float(np.std(r, ddof=1))


def vol_parkinson(high: ArrayLike, low: ArrayLike) -> float | None:
    """Parkinson (1980) bar-başı vol tahmincisi: sqrt( mean(ln(H/L)^2) / (4 ln2) ).
    H,L eşit uzunlukta ve H>L>0 olmalı; aksi halde None."""
    h = _clean_1d(high, min_len=1)
    if h is None:
        return None
    lo = _clean_1d(low, min_len=1)
    if lo is None:
        return None
    if h.size != lo.size or np.any(h <= 0) or np.any(lo <= 0) or np.any(h < lo):
        return None
    term = np.log(h / lo) ** 2
    var = float(np.mean(term)) / (4.0 * math.log(2.0))
    return math.sqrt(var) if var >= 0 else None


def vol_garman_klass(o: ArrayLike, h: ArrayLike, l: ArrayLike, c: ArrayLike) -> float | None:
    """Garman-Klass (1980) bar-başı vol tahmincisi:
    sqrt( mean( 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2 ) ). OHLC eşit uzunlukta, hepsi >0 olmalı."""
    oa = _clean_1d(o, min_len=1)
    if oa is None:
        return None
    ha = _clean_1d(h, min_len=1)
    if ha is None:
        return None
    la = _clean_1d(l, min_len=1)
    if la is None:
        return None
    ca = _clean_1d(c, min_len=1)
    if ca is None:
        return None
    n = oa.size
    if not (ha.size == la.size == ca.size == n):
        return None
    if np.any(oa <= 0) or np.any(ha <= 0) or np.any(la <= 0) or np.any(ca <= 0) or np.any(ha < la):
        return None
    log_hl = np.log(ha / la)
    log_co = np.log(ca / oa)
    term = 0.5 * log_hl**2 - (2.0 * math.log(2.0) - 1.0) * log_co**2
    var = float(np.mean(term))
    return math.sqrt(var) if var >= 0 else None


def vol_yang_zhang(o: ArrayLike, h: ArrayLike, l: ArrayLike, c: ArrayLike) -> float | None:
    """Yang-Zhang (2000) bar-başı vol tahmincisi (gece-getirisi + gündüz-getirisi + Rogers-
    Satchell bileşimi; drift-bağımsız). En az 3 bar ister (ilk barın önceki kapanışı yoktur →
    gece getirisi n-1 örnek verir; ddof=1 varyans için n-1>=2 gerekir)."""
    oa = _clean_1d(o, min_len=3)
    if oa is None:
        return None
    ha = _clean_1d(h, min_len=3)
    if ha is None:
        return None
    la = _clean_1d(l, min_len=3)
    if la is None:
        return None
    ca = _clean_1d(c, min_len=3)
    if ca is None:
        return None
    n = oa.size
    if not (ha.size == la.size == ca.size == n):
        return None
    if np.any(oa <= 0) or np.any(ha <= 0) or np.any(la <= 0) or np.any(ca <= 0):
        return None
    prev_close = ca[:-1]
    o_t, h_t, l_t, c_t = oa[1:], ha[1:], la[1:], ca[1:]
    m = o_t.size
    if m < 2:
        return None
    overnight = np.log(o_t / prev_close)
    close_open = np.log(c_t / o_t)
    u = np.log(h_t / o_t)
    d = np.log(l_t / o_t)
    rs = u * (u - close_open) + d * (d - close_open)
    v_o = float(np.var(overnight, ddof=1))
    v_c = float(np.var(close_open, ddof=1))
    v_rs = float(np.mean(rs))
    k = 0.34 / (1.34 + (m + 1) / (m - 1))
    var_yz = v_o + k * v_c + (1.0 - k) * v_rs
    return math.sqrt(var_yz) if var_yz >= 0 else None


def volatility_cone(o: ArrayLike, h: ArrayLike, l: ArrayLike, c: ArrayLike, *,
                    windows: tuple[int, ...] = (10, 20, 30, 60, 90, 180, 252),
                    estimator: str = "yang_zhang", iv_current: float | None = None,
                    ppy: float = 252, min_history: int = 20) -> dict | None:
    """OHLC gerçekleşen vol konisi; yıllık oranlar, IV sırası [0,100].

    T penceresi T günün hareketini ölçer (Yang–Zhang ayrıca önceki kapanışı ister).
    Tarihsel örneklem kısa olan vadeler atlanır; eksik IV yerine RV konulmaz.
    IV sırası bağlarda orta sıradır. Bu bir prim/işlem önerisi değildir.
    """
    arrays = [_clean_1d(x, min_len=3) for x in (o, h, l, c)]
    if any(x is None for x in arrays):
        return None
    op, hi, lo, close = [x for x in arrays if x is not None]
    if len({x.size for x in arrays if x is not None}) != 1:
        return None
    if (np.any(lo <= 0) or np.any(hi < np.maximum(op, close))
            or np.any(lo > np.minimum(op, close))):
        return None
    try:
        ppy = float(ppy)
        if (not math.isfinite(ppy) or ppy <= 0 or int(min_history) != min_history or min_history < 2
                or estimator not in ("parkinson", "yang_zhang", "close_to_close")
                or not windows or any(int(w) != w or w < 2 for w in windows)):
            return None
        wins = sorted(set(int(w) for w in windows))
        iv = None if iv_current is None else float(iv_current)
    except (ValueError, TypeError, OverflowError):
        return None
    if iv is not None and (not math.isfinite(iv) or iv <= 0):
        iv = None
    cone: dict = {k: [] for k in ("windows", "min", "p5", "p25", "p50", "p75", "p95", "max",
                                  "current", "n_history", "iv_percentile")}
    from numpy.lib.stride_tricks import sliding_window_view

    with np.errstate(all="ignore"):
        log_oc = np.log(op[1:]) - np.log(close[:-1])
        log_co = np.log(close[1:]) - np.log(op[1:])
        log_ho = np.log(hi[1:]) - np.log(op[1:])
        log_lo = np.log(lo[1:]) - np.log(op[1:])
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        for w in wins:
            n = close.size if estimator == "parkinson" else close.size - 1
            if n - w + 1 < min_history:
                continue
            if estimator == "parkinson":
                v = sliding_window_view((np.log(hi) - np.log(lo)) ** 2, w).mean(axis=1) / (4 * math.log(2))
            elif estimator == "close_to_close":
                v = sliding_window_view(np.diff(np.log(close)), w).var(axis=1, ddof=1)
            else:
                k = 0.34 / (1.34 + (w + 1) / (w - 1))
                v = (sliding_window_view(log_oc, w).var(axis=1, ddof=1)
                     + k * sliding_window_view(log_co, w).var(axis=1, ddof=1)
                     + (1 - k) * sliding_window_view(rs, w).mean(axis=1))
            rv = np.sqrt(np.maximum(v, 0) * ppy)
            if not np.all(np.isfinite(rv)):
                return None
            cone["windows"].append(w)
            for key, value in zip(("min", "p5", "p25", "p50", "p75", "p95", "max"),
                                  np.percentile(rv, [0, 5, 25, 50, 75, 95, 100]), strict=True):
                cone[key].append(float(value))
            cone["current"].append(float(rv[-1]))
            cone["n_history"].append(int(rv.size))
            cone["iv_percentile"].append(None if iv is None else
                float(100 * (np.sum(rv < iv) + 0.5 * np.sum(rv == iv)) / rv.size))
    if not cone["windows"]:
        return None
    return {**cone, "estimator": estimator, "ppy": ppy, "iv_current": iv}


def ewma_vol(returns: ArrayLike, lam: float = 0.94) -> float | None:
    """RiskMetrics-tarzı EWMA volatilite (bar-başı, özyinelemeli):
    var_0 = r[0]^2; var_t = λ·var_{t-1} + (1-λ)·r_t^2. λ ∈ (0,1) olmalı; aksi/kısa/NaN → None.

    KARE `x*x` ile alınır, `x**2` ile DEĞİL: CPython'da float `**` taşmada OverflowError
    FIRLATIR, çarpma ise numpy gibi `inf` verir (Tur-49). Modülün "hiçbir girdi istisna
    fırlatmaz" sözleşmesi buna bağlıdır; taşan yol artık istisna değil None üretir."""
    arr = _clean_1d(returns, min_len=1)
    if arr is None or not (0.0 < lam < 1.0):
        return None
    var = float(arr[0]) * float(arr[0])
    for r in arr[1:]:
        rf = float(r)
        var = lam * var + (1.0 - lam) * rf * rf
    return math.sqrt(var) if math.isfinite(var) and var >= 0 else None


# ---------------------------------------------------------------------------
# Risk / kayıp metrikleri (VaR ailesi, drawdown)
# ---------------------------------------------------------------------------
def var_historical(returns: ArrayLike, alpha: float = 0.05) -> float | None:
    """Tarihsel VaR: kayıp BÜYÜKLÜĞÜ (≥0). alpha-yüzdelik getiri negatifse o kadar kayıp;
    pozitifse (o eşikte hiç kayıp gözlenmemiş) 0.0'a kırpılır. alpha ∉ (0,1) veya kısa/NaN
    girdi → None."""
    arr = _clean_1d(returns, min_len=2)
    if arr is None or not (0.0 < alpha < 1.0):
        return None
    q = float(np.percentile(arr, alpha * 100.0))
    return max(0.0, -q)


def cvar_historical(returns: ArrayLike, alpha: float = 0.05) -> float | None:
    """Koşullu VaR / Expected Shortfall: alpha-kuyruğundaki getirilerin ORTALAMA kayıp
    BÜYÜKLÜĞÜ (≥0). alpha ∉ (0,1) veya kısa/NaN girdi → None."""
    arr = _clean_1d(returns, min_len=2)
    if arr is None or not (0.0 < alpha < 1.0):
        return None
    q = np.percentile(arr, alpha * 100.0)
    tail = arr[arr <= q]
    if tail.size == 0:
        k = max(1, int(math.ceil(alpha * arr.size)))
        tail = np.sort(arr)[:k]
    return max(0.0, float(-np.mean(tail)))


def var_cornish_fisher(returns: ArrayLike, alpha: float = 0.05) -> float | None:
    """Cornish-Fisher genişlemesiyle çarpıklık/basıklık-ayarlı VaR (kayıp BÜYÜKLÜĞÜ, ≥0).
    En az 4 gözlem ister (anlamlı çarpıklık/basıklık için); azı/NaN/alpha∉(0,1) → None."""
    arr = _clean_1d(returns, min_len=4)
    if arr is None or not (0.0 < alpha < 1.0):
        return None
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd == 0.0:
        return 0.0
    s = float(stats.skew(arr))
    k = float(stats.kurtosis(arr, fisher=True))
    z = _norm_ppf(alpha)
    z_cf = (
        z
        + (z**2 - 1.0) * s / 6.0
        + (z**3 - 3.0 * z) * k / 24.0
        - (2.0 * z**3 - 5.0 * z) * (s**2) / 36.0
    )
    return max(0.0, float(-(mu + z_cf * sd)))


def max_drawdown(equity: ArrayLike) -> tuple[float, int] | None:
    """Ekuiti eğrisinden maksimum düşüş → (dd_frac, süre_bar).

    dd_frac: en derin düşüşün BÜYÜKLÜĞÜ (≥0; ör. 0.25 = %25 düşüş).
    süre_bar: en derin düşüşün ZİRVEden dibe kadar sürdüğü bar sayısı (dip_index - zirve_index;
        zirve = dibin ÖNCESİNDEKİ en-son eşit-zirve barı). Toparlanma süresi DEĞİLDİR.
    Girdi <2 nokta veya sıfır/negatif eleman içeriyorsa (oran hesaplanamaz) → None."""
    arr = _clean_1d(equity, min_len=2)
    if arr is None or np.any(arr <= 0):
        return None
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    trough_idx = int(np.argmin(dd))
    worst = float(dd[trough_idx])
    if worst >= 0.0:
        return (0.0, 0)
    matches = np.where(arr[: trough_idx + 1] == peak[trough_idx])[0]
    peak_idx = int(matches[-1]) if matches.size else 0
    return (abs(worst), trough_idx - peak_idx)


def downside_deviation(returns: ArrayLike, mar: float = 0.0) -> float | None:
    """Aşağı-yönlü sapma (Sortino semi-deviation): sqrt( mean( min(r-mar,0)^2 ) ), tüm n
    üzerinden (yalnız kayıp sayısı değil). Kısa/NaN girdi → None."""
    arr = _clean_1d(returns, min_len=1)
    if arr is None:
        return None
    downside = np.minimum(arr - mar, 0.0)
    return float(np.sqrt(np.mean(downside**2)))


def ulcer_index(equity: ArrayLike) -> float | None:
    """Ulcer Index: sqrt( mean( ((peak-equity)/peak)^2 ) ) — düşüşün derinliği VE süresini
    birlikte cezalandıran risk ölçütü. Kısa/NaN/sıfır-negatif eleman → None."""
    arr = _clean_1d(equity, min_len=1)
    if arr is None or np.any(arr <= 0):
        return None
    peak = np.maximum.accumulate(arr)
    dd_pct = (peak - arr) / peak
    return float(np.sqrt(np.mean(dd_pct**2)))


# ---------------------------------------------------------------------------
# Risk-ayarlı performans oranları
# ---------------------------------------------------------------------------
def sharpe(returns: ArrayLike, rf: float = 0.0, ppy: float = 252) -> float | None:
    """Yıllıklandırılmış Sharpe: (mean(r) - rf/ppy) / std(r,ddof=1) * sqrt(ppy). rf YILLIK
    oran kabul edilir. std=0 veya ppy<=0 veya kısa/NaN girdi → None."""
    arr = _clean_1d(returns, min_len=2)
    if arr is None or ppy <= 0:
        return None
    sd = float(np.std(arr, ddof=1))
    if sd == 0.0:
        return None
    excess = float(np.mean(arr)) - rf / ppy
    return float(excess / sd * math.sqrt(ppy))


def sortino(returns: ArrayLike, rf: float = 0.0, ppy: float = 252) -> float | None:
    """Yıllıklandırılmış Sortino: (mean(r) - rf/ppy) / downside_deviation(r, mar=rf/ppy) *
    sqrt(ppy). Aşağı-sapma 0 (kayıp yok) veya kısa/NaN girdi → None."""
    arr = _clean_1d(returns, min_len=2)
    if arr is None or ppy <= 0:
        return None
    rf_period = rf / ppy
    dd = downside_deviation(arr, mar=rf_period)
    if dd is None or dd == 0.0:
        return None
    excess = float(np.mean(arr)) - rf_period
    return float(excess / dd * math.sqrt(ppy))


def calmar(returns: ArrayLike, ppy: float = 252) -> float | None:
    """Calmar oranı: geometrik-yıllıklandırılmış getiri / max_drawdown büyüklüğü. `returns`'ten
    kümülatif ekuiti (başlangıç=1.0) türetilir. max_drawdown=0 veya kısa/NaN/negatif-ekuiti
    → None."""
    arr = _clean_1d(returns, min_len=2)
    if arr is None or ppy <= 0:
        return None
    equity = np.concatenate(([1.0], np.cumprod(1.0 + arr)))
    if np.any(equity <= 0):
        return None
    mdd = max_drawdown(equity)
    if mdd is None or mdd[0] == 0.0:
        return None
    n = arr.size
    total_growth = float(equity[-1] / equity[0])
    if total_growth <= 0:
        return None
    ann_ret = float(total_growth ** (ppy / n) - 1.0)
    return float(ann_ret / mdd[0])


def omega(returns: ArrayLike, threshold: float = 0.0) -> float | None:
    """Omega oranı: eşik-üstü kazanç alanı / eşik-altı kayıp alanı. Kayıp alanı 0 (tanımsız/
    sonsuz oran) veya kısa/NaN girdi → None."""
    arr = _clean_1d(returns, min_len=1)
    if arr is None:
        return None
    gains = float(np.sum(np.maximum(arr - threshold, 0.0)))
    losses = float(np.sum(np.maximum(threshold - arr, 0.0)))
    if losses == 0.0:
        return None
    return float(gains / losses)


def profit_factor(pnls: ArrayLike) -> float | None:
    """Kâr faktörü: toplam kazanç / |toplam kayıp|. Kayıp yok (tanımsız/sonsuz oran) veya
    kısa/NaN girdi → None."""
    arr = _clean_1d(pnls, min_len=1)
    if arr is None:
        return None
    gains = float(np.sum(arr[arr > 0]))
    losses = float(-np.sum(arr[arr < 0]))
    if losses == 0.0:
        return None
    return float(gains / losses)


# ---------------------------------------------------------------------------
# Pozisyon boyutlama (Kelly ailesi)
# ---------------------------------------------------------------------------
def kelly_binary(p_win: float, win_loss_ratio: float) -> float:
    """İkili-bahis Kelly kesri: f* = p - (1-p)/b. NEGATİF kenar (f*<=0) veya geçersiz girdi
    (b<=0, sonlu-değil) → 0.0 (asla negatif pozisyon önermez)."""
    try:
        p, b = float(p_win), float(win_loss_ratio)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(p) or not math.isfinite(b) or b <= 0.0:
        return 0.0
    p = min(max(p, 0.0), 1.0)
    f = p - (1.0 - p) / b
    return float(f) if f > 0.0 else 0.0


def kelly_continuous(mean_ret: float, var_ret: float) -> float:
    """Sürekli/normal-getiri Kelly kesri: f* = mean_ret / var_ret. var_ret<=0 veya sonlu-değil
    girdi → 0.0 (bölme-sıfır korumalı). Not: kelly_binary'nin aksine işaretli (kısa öneriler
    de mümkün) — long-only kırpma çağıranın (ör. fractional_kelly) sorumluluğudur."""
    try:
        m, v = float(mean_ret), float(var_ret)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(m) or not math.isfinite(v) or v <= 0.0:
        return 0.0
    return float(m / v)


def fractional_kelly(k: float, fraction: float = 0.25, cap: float = 0.02) -> float:
    """Kesirli Kelly: clip(k*fraction, 0, cap). Negatif/sonlu-olmayan girdi → 0.0 (asla
    negatif pozisyon önermez); cap negatifse 0'a düzeltilir."""
    try:
        kv, fr, c = float(k), float(fraction), float(cap)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(kv) or not math.isfinite(fr):
        return 0.0
    if not math.isfinite(c) or c < 0.0:
        c = 0.0
    return float(min(max(kv * fr, 0.0), c))


# ---------------------------------------------------------------------------
# Ortalamaya-dönüş / kointegrasyon istatistikleri
# ---------------------------------------------------------------------------
def ou_half_life(series: ArrayLike) -> float | None:
    """Ornstein-Uhlenbeck yarı-ömrü, AR(1) katsayısından (pairs-trading konvansiyonu):
    Δy_t = a + b·y_{t-1} + ε (OLS), half_life = -ln(2)/b. b>=0 (mean-reversion YOK) veya
    b sonlu-değil veya <5 nokta → None."""
    arr = _clean_1d(series, min_len=5)
    if arr is None:
        return None
    y = arr[1:]
    y_lag = arr[:-1]
    dy = y - y_lag
    design = np.column_stack([np.ones_like(y_lag), y_lag])
    try:
        coef, *_ = np.linalg.lstsq(design, dy, rcond=None)
    except np.linalg.LinAlgError:
        return None
    b = float(coef[1])
    if not math.isfinite(b) or b >= 0:
        return None
    hl = -math.log(2.0) / b
    if not math.isfinite(hl) or hl <= 0:
        return None
    return float(hl)


def hurst_exponent(series: ArrayLike, max_lag: int = 20) -> float | None:
    """Genelleştirilmiş Hurst üsteli: log(std(lag-farkları)) ~ H*log(lag) + sabit ilişkisinin
    eğimi (yapı-fonksiyonu yöntemi). H<0.5 → ortalamaya-dönüş eğilimi, H≈0.5 → rastgele-yürüyüş,
    H>0.5 → trend-sürekliliği. En az max(20, 2*max_lag) nokta ister; yetersiz/dejenere → None."""
    arr = _clean_1d(series, min_len=1)
    ml = int(max_lag)
    if arr is None or ml < 4 or arr.size < max(20, ml * 2):
        return None
    lags = np.arange(2, ml)
    tau = np.full(lags.shape, np.nan)
    for idx, lag in enumerate(lags):
        diffs = arr[lag:] - arr[:-lag]
        sd = float(np.std(diffs))
        if sd > 0:
            tau[idx] = sd
    mask = np.isfinite(tau) & (tau > 0)
    if int(np.sum(mask)) < 3:
        return None
    slope, _ = np.polyfit(np.log(lags[mask].astype(float)), np.log(tau[mask]), 1)
    h = float(slope)
    return h if math.isfinite(h) else None


def variance_ratio(series: ArrayLike, q: int) -> float | None:
    """Lo-MacKinlay variance-ratio noktatahmini: Var(q-barlık örtüşen getiri) / (q*Var(1-barlık
    getiri)). VR≈1 rastgele-yürüyüş, VR<1 ortalamaya-dönüş, VR>1 trend. q<2 veya yetersiz
    veri/NaN → None."""
    arr = _clean_1d(series, min_len=1)
    qi = int(q)
    if arr is None or qi < 2:
        return None
    r = log_returns(arr)
    n = r.size
    if n < 2 * qi or not np.all(np.isfinite(r)):
        return None
    var_1 = float(np.var(r, ddof=1))
    if var_1 == 0.0:
        return None
    rq = np.array([float(np.sum(r[i : i + qi])) for i in range(0, n - qi + 1)])
    var_q = float(np.var(rq, ddof=1))
    vr = var_q / (qi * var_1)
    return float(vr) if math.isfinite(vr) else None


def adf_test(series: ArrayLike) -> tuple[float, float] | None:
    """Augmented Dickey-Fuller durağanlık testi → (adf_stat, pvalue). statsmodels YOKSA
    (opsiyonel bağımlılık) veya girdi kısa/NaN/regresyon başarısızsa → None."""
    if _adfuller is None:
        return None
    arr = _clean_1d(series, min_len=8)
    if arr is None:
        return None
    try:
        result = _adfuller(arr, autolag="AIC")
    except (ValueError, np.linalg.LinAlgError):
        return None
    stat, pvalue = float(result[0]), float(result[1])
    return stat, pvalue


def engle_granger(y: ArrayLike, x: ArrayLike) -> dict | None:
    """Engle-Granger (1987) iki-adımlı kointegrasyon testi: y = alpha + beta*x + resid (OLS),
    ardından resid üzerinde ADF + OU yarı-ömrü. adf_test None dönerse (statsmodels yok)
    adf_stat/pvalue None kalır ama beta/alpha/half_life yine de hesaplanır (kısmi-defensif).
    En az 10 eşit-uzunlukta nokta ister; azı/uyumsuz boy → None.

    Dönüş: {'beta','alpha','adf_stat','pvalue','half_life'}."""
    y_arr = _clean_1d(y, min_len=10)
    if y_arr is None:
        return None
    x_arr = _clean_1d(x, min_len=10)
    if x_arr is None or y_arr.size != x_arr.size:
        return None
    design = np.column_stack([np.ones_like(x_arr), x_arr])
    try:
        coef, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    except np.linalg.LinAlgError:
        return None
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y_arr - (alpha + beta * x_arr)
    adf = adf_test(resid)
    adf_stat, pvalue = adf if adf is not None else (None, None)
    half_life = ou_half_life(resid)
    return {"beta": beta, "alpha": alpha, "adf_stat": adf_stat, "pvalue": pvalue,
            "half_life": half_life}


def kalman_beta(y: ArrayLike, x: ArrayLike, q: float = 1e-5, r: float = 1e-2) -> np.ndarray | None:
    """1B skaler Kalman filtresiyle DİNAMİK (zamanla-değişen) beta yolu (pairs-trading
    konvansiyonu). Durum: beta_t = beta_{t-1} + süreç-gürültüsü(q); gözlem: y_t = beta_t*x_t +
    gözlem-gürültüsü(r). Başlangıç önseli beta_0=0, P_0=1 (belgelenmiş, veri-bağımsız).
    q<=0/r<=0/uyumsuz-boy/kısa girdi → None."""
    y_arr = _clean_1d(y, min_len=2)
    if y_arr is None:
        return None
    x_arr = _clean_1d(x, min_len=2)
    if x_arr is None or y_arr.size != x_arr.size or q <= 0 or r <= 0:
        return None
    n = y_arr.size
    beta = np.empty(n, dtype=float)
    p = 1.0
    beta_est = 0.0
    for t in range(n):
        p += q
        xt = float(x_arr[t])
        s = xt * xt * p + r
        k_gain = (p * xt / s) if s > 0 else 0.0
        beta_est = beta_est + k_gain * (float(y_arr[t]) - beta_est * xt)
        p = (1.0 - k_gain * xt) * p
        beta[t] = beta_est
    return beta


# ---------------------------------------------------------------------------
# Kovaryans / risk ayrıştırma
# ---------------------------------------------------------------------------
def ledoit_wolf_cov(ret_matrix: ArrayLike) -> np.ndarray | None:
    """Ledoit-Wolf küçültmeli (shrinkage) kovaryans matrisi (sklearn.covariance.LedoitWolf).
    Girdi (n_obs, n_asset); n_obs<n_asset olsa dahi (shrinkage sayesinde) çalışır. Geçersiz/
    dejenere girdi → None."""
    arr = _clean_2d(ret_matrix, min_rows=2, min_cols=1)
    if arr is None:
        return None
    from sklearn.covariance import LedoitWolf  # lazy — dosya başındaki gerekçeye bak
    try:
        return np.asarray(LedoitWolf().fit(arr).covariance_)
    except (ValueError, np.linalg.LinAlgError):
        return None


def ewma_cov(ret_matrix: ArrayLike, lam: float = 0.94) -> np.ndarray | None:
    """RiskMetrics-tarzı EWMA kovaryans matrisi (özyinelemeli, sıfır-ortalama varsayımıyla —
    kısa-ufuklu getirilerde standart basitleştirme). λ ∈ (0,1) olmalı; aksi/kısa/dejenere
    girdi → None."""
    arr = _clean_2d(ret_matrix, min_rows=2, min_cols=1)
    if arr is None or not (0.0 < lam < 1.0):
        return None
    cov = np.outer(arr[0], arr[0])
    for t in range(1, arr.shape[0]):
        cov = lam * cov + (1.0 - lam) * np.outer(arr[t], arr[t])
    return np.asarray(cov)


def pca_risk(ret_matrix: ArrayLike, n: int = 3) -> dict | None:
    """PCA risk ayrıştırması → {'explained': [açıklanan varyans oranı,...], 'loadings':
    [[bileşen yükleri],...]}. n, mevcut satır/sütun sayısıyla otomatik kırpılır (k=min(n,
    n_asset,n_obs)). Geçersiz/dejenere girdi → None."""
    arr = _clean_2d(ret_matrix, min_rows=2, min_cols=1)
    if arr is None:
        return None
    k = min(int(n), arr.shape[1], arr.shape[0])
    if k < 1:
        return None
    from sklearn.decomposition import PCA  # lazy — dosya başındaki gerekçeye bak
    try:
        pca = PCA(n_components=k)
        pca.fit(arr)
    except ValueError:
        return None
    return {"explained": pca.explained_variance_ratio_.tolist(),
            "loadings": pca.components_.tolist()}


# ---------------------------------------------------------------------------
# Monte Carlo simülasyonu
# ---------------------------------------------------------------------------
def mc_gbm_paths(s0: float, mu: float, sigma: float, days: int, n_paths: int, seed: int) -> np.ndarray:
    """Geometrik Brownian Motion Monte Carlo fiyat yolları → şekil (n_paths, days+1); sütun 0
    = s0. Günlük adım (dt=1/252; annualize_* fonksiyonlarındaki ppy=252 konvansiyonuyla
    tutarlı). `seed` ZORUNLU determinizm için: aynı seed → bit-bit aynı çıktı. Geçersiz girdi
    (days<0, n_paths<=0, s0<=0, sigma<0, sonlu-değil) → boş/sabit dizi (İSTİSNA YOK)."""
    try:
        d, npaths = int(days), int(n_paths)
        s0f, muf, sigf = float(s0), float(mu), float(sigma)
    except (TypeError, ValueError):
        return np.zeros((0, 1))
    if npaths <= 0:
        return np.zeros((0, max(d, 0) + 1))
    if d < 0 or s0f <= 0 or not math.isfinite(muf) or not math.isfinite(sigf) or sigf < 0:
        fallback = s0f if math.isfinite(s0f) and s0f > 0 else 0.0
        return np.full((npaths, 1), fallback)
    if d == 0:
        return np.full((npaths, 1), s0f)
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    z = rng.standard_normal((npaths, d))
    increments = (muf - 0.5 * sigf**2) * dt + sigf * math.sqrt(dt) * z
    log_cum = np.cumsum(increments, axis=1)
    log_cum = np.concatenate([np.zeros((npaths, 1)), log_cum], axis=1)
    return s0f * np.exp(log_cum)


def mc_bootstrap_terminal(returns: ArrayLike, horizon: int, n_paths: int, seed: int,
                          block: int = 5) -> dict:
    """Blok-bootstrap ile `horizon` barlık terminal (ufuk-sonu) kümülatif getiri dağılımı.
    Tarihsel `returns`'ten `block`-uzunluklu ardışık bloklar RASTGELE (seed'li, tekrarlı; devre-
    dışı sarma) örneklenip uç-uca eklenerek `horizon`a tamamlanır; `n_paths` yol için terminal
    büyüme-1 hesaplanır. `seed` ZORUNLU determinizm için. Dönüş TİPİ her zaman dict (None
    DEĞİL) — geçersiz/kısa girdi → tüm alanlar None (İSTİSNA YOK).

    Dönüş: {'p5','p50','p95','prob_loss'}."""
    empty = {"p5": None, "p50": None, "p95": None, "prob_loss": None}
    arr = _clean_1d(returns, min_len=2)
    if arr is None:
        return empty
    try:
        h, npaths, blk = int(horizon), int(n_paths), max(1, int(block))
    except (TypeError, ValueError):
        return empty
    if h <= 0 or npaths <= 0:
        return empty
    n = arr.size
    rng = np.random.default_rng(seed)
    max_start = max(1, n - blk + 1)
    # --- Katalog #21 (Tur-67): yol başına Python döngüsü → tek matris. ÖLÇÜLDÜ:
    # n=1500·h=21·4000 yol 146,22 → 2,11 ms (69,2×); n=250·h=63 314,42 → 5,91 ms (53,2×).
    # Çıktı BİT-DENKTİR (testle kilitli) ve bu iki ayrıntıya bağlıdır:
    #   1) `rng.integers(0, max_start, size=(npaths, n_blk))` skaler çağrı akışıyla
    #      ÖZDEŞ dizi verir (ölçüldü: 5 ayrı `max_start` için akış eşitliği) ve satır-
    #      öncelikli sıra, eski döngünün yol-yol çekiş sırasının aynısıdır.
    #   2) Çarpım `np.multiply.accumulate(...)[:, -1]` ile ARDIŞIK yapılır. `np.prod`
    #      eksende pairwise indirger; eski gövde her yolu ayrı `np.prod` ile 1-B
    #      çarptığı için sıra farkı son bitlerde ayrışırdı.
    # Kuyruk kırpılmaz: `start ≤ n-blk` olduğundan blok hep tam uzunluktadır; n < blk
    # ise max_start=1 ve dilim `arr` kadardır — `seg_len` ikisini de karşılar.
    seg_len = min(blk, n)
    n_blk = -(-h // seg_len)                    # ceil(h / seg_len)
    starts = rng.integers(0, max_start, size=(npaths, n_blk))
    ofs = np.arange(seg_len)
    idx = (starts[:, :, None] + ofs[None, None, :]).reshape(npaths, n_blk * seg_len)[:, :h]
    terminal = np.multiply.accumulate(1.0 + arr[idx], axis=1)[:, -1] - 1.0
    return {
        "p5": float(np.percentile(terminal, 5)),
        "p50": float(np.percentile(terminal, 50)),
        "p95": float(np.percentile(terminal, 95)),
        "prob_loss": float(np.mean(terminal < 0.0)),
    }


# ---------------------------------------------------------------------------
# Opsiyon fiyatlama (Black-Scholes kapalı-form)
#
# SKALER normal dağılım — stdlib `statistics.NormalDist` (Tur-36 ölçümü; Tur-34'ün "sıkı
# döngüdeki genel-amaçlı sarmalayıcı" dersinin ÜÇÜNCÜ tekrarı):
#   `scipy.stats.norm.cdf` TEK skaler için ~39,3 µs harcar (array-API gönderimi, argsreduce,
#   dtype yükseltme); `NormalDist().cdf` ~0,23 µs = **171×**. `ppf` **509×**. `bs_price` çağrı
#   başına iki `cdf` çağırır ve `implied_vol` onu Brent döngüsünde ~10 kez çağırır →
#   `bs_price` **53×** (85,5 → 1,6 µs), `implied_vol` **29×** (864,1 → 29,4 µs).
#   Çözüm yeni bir bağımlılık değil merdivenin STDLIB basamağıydı.
#
# DENKLİK — tek noktada değil ızgarada ölçüldü (Tur-36 dersi: "bir denklik iddiası TEK
# NOKTADA ölçülmez"; ilk okumada ppf p=0,75'te scipy ile TAM eşit çıkmış ve yanlışlıkla
# 'bit-denk' yazılmıştı, 20.011 noktalık ızgarada iddia çöktü). Üç AYRI iddia:
#   `_norm_pdf` → BİT-DENK (%100) · `_norm_cdf` → ≤2,2e-16 · `_norm_ppf` → ≤1,8e-15.
# Karar düzeyine taşınmış hâli: 1 sentin üstündeki her opsiyon fiyatında bağıl fark
# 8,5e-13, `implied_vol` gidiş-dönüşünde maks |Δσ| 9e-8. (Bağıl fark yalnız fiyat ~0'a
# giderken büyür = "değersiz opsiyon" bölgesi; orada MUTLAK fark 1e-13'tür. Bir bağıl hata
# metriği, paydası sıfıra giderken RAPOR EDİLEMEZ.)
#
# DİZİ girdilerinde scipy KALIR (orada vektörel ve hızlıdır) — değişen yalnız SKALER
# çağrılardır. `_SND` modül düzeyinde bir kez kurulur: `NormalDist()` kurucusu da ölçülebilir
# bir maliyettir ve bu üç fonksiyon sıkı döngüden çağrılır.
# ---------------------------------------------------------------------------
_SND = NormalDist()


def _norm_cdf(x: float) -> float:
    """Standart normal CDF — SKALER. Dizi için `scipy.stats.norm.cdf` kullanın."""
    return _SND.cdf(x)


def _norm_pdf(x: float) -> float:
    """Standart normal PDF — SKALER (scipy ile BİT-DENK)."""
    return _SND.pdf(x)


def _norm_ppf(p: float) -> float:
    """Standart normal ters CDF — SKALER. p ∈ (0,1) dışında ValueError (stdlib sözleşmesi)."""
    return _SND.inv_cdf(p)


def _bs_d1_d2(s: float, k: float, t: float, r: float, sigma: float) -> tuple[float, float]:
    sqt = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * sqt)
    return d1, d1 - sigma * sqt


def _bs_inputs_valid(s: float, k: float, t: float, r: float, sigma: float) -> bool:
    if not all(math.isfinite(v) for v in (s, k, t, r, sigma)):
        return False
    return s > 0 and k > 0 and t > 0 and sigma > 0


def bs_price(s: float, k: float, t: float, r: float, sigma: float, kind: str = "c") -> float | None:
    """Black-Scholes (1973) kapalı-form Avrupa opsiyon fiyatı (kendi implementasyonu;
    scipy.stats.norm ile). kind: 'c'/'call' veya 'p'/'put'. Geçersiz girdi (s,k,t,sigma<=0,
    sonlu-değil, bilinmeyen kind) → None."""
    try:
        sf, kf, tf, rf_, sigf = float(s), float(k), float(t), float(r), float(sigma)
    except (TypeError, ValueError):
        return None
    if not _bs_inputs_valid(sf, kf, tf, rf_, sigf):
        return None
    kind_l = (kind or "").strip().lower()
    if kind_l not in ("c", "call", "p", "put"):
        return None
    d1, d2 = _bs_d1_d2(sf, kf, tf, rf_, sigf)
    disc = math.exp(-rf_ * tf)
    if kind_l in ("c", "call"):
        return float(sf * _norm_cdf(d1) - kf * disc * _norm_cdf(d2))
    return float(kf * disc * _norm_cdf(-d2) - sf * _norm_cdf(-d1))


def bs_greeks(s: float, k: float, t: float, r: float, sigma: float, kind: str = "c") -> dict | None:
    """Black-Scholes kapalı-form Greek'ler → {'delta','gamma','vega','theta','rho'}. vega ham
    (1.0 vol birimi başına, yani sigma=0.20→0.21 değişimi); theta YILLIK (bar-başına çevirmek
    isteyen /365 yapar). Geçersiz girdi → None (bkz. bs_price)."""
    try:
        sf, kf, tf, rf_, sigf = float(s), float(k), float(t), float(r), float(sigma)
    except (TypeError, ValueError):
        return None
    if not _bs_inputs_valid(sf, kf, tf, rf_, sigf):
        return None
    kind_l = (kind or "").strip().lower()
    if kind_l not in ("c", "call", "p", "put"):
        return None
    sqt = math.sqrt(tf)
    d1, d2 = _bs_d1_d2(sf, kf, tf, rf_, sigf)
    pdf_d1 = _norm_pdf(d1)
    is_call = kind_l in ("c", "call")
    disc = math.exp(-rf_ * tf)
    delta = _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0
    gamma = pdf_d1 / (sf * sigf * sqt)
    vega = sf * pdf_d1 * sqt
    if is_call:
        theta = -(sf * pdf_d1 * sigf) / (2.0 * sqt) - rf_ * kf * disc * _norm_cdf(d2)
        rho = kf * tf * disc * _norm_cdf(d2)
    else:
        theta = -(sf * pdf_d1 * sigf) / (2.0 * sqt) + rf_ * kf * disc * _norm_cdf(-d2)
        rho = -kf * tf * disc * _norm_cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ---------------------------------------------------------------------------
# Rejim sınıflandırma
# ---------------------------------------------------------------------------
def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Basit hareketli ortalama (pandas'sız, kümülatif-toplam ile). İlk window-1 eleman nan."""
    out = np.full(arr.shape, np.nan)
    if window <= 0 or arr.size < window:
        return out
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    out[window - 1 :] = (csum[window:] - csum[:-window]) / window
    return out


def classify_regime(close: ArrayLike, high: ArrayLike | None = None,
                    low: ArrayLike | None = None) -> dict:
    """Kural-tabanlı piyasa rejimi sınıflandırması (bkz. AI_V2_BEYIN_SPEC.md §1):
      - trend: SMA50 eğim işareti (son değer vs 5-bar-önceki; göreli eşik ±5bps → 'flat').
      - vol_bucket: gerçekleşen 20-bar vol'un KENDİ trailing (≤252 bar) dağılımındaki
        yüzdelik sırası (<%33 low, >%67 high, aksi mid). `high`/`low` verilmişse (aynı
        uzunlukta, sonlu) rolling Parkinson vol kullanılır (OHLC-menzil bilgisiyle daha
        verimli); verilmezse close-to-close log-getiri std'sine düşülür.
      - mean_reversion: hurst_exponent(close) < 0.45.
    En az 60 bar ister; azı/NaN → {"label":"unknown", diğerleri None} (çökme yok)."""
    arr = _as_1d(close)
    if arr.size < 60 or not np.all(np.isfinite(arr)):
        return {"trend": None, "vol_bucket": None, "mean_reversion": None, "label": "unknown"}

    sma = _rolling_mean(arr, 50)
    last = sma[-1]
    prior = sma[-6] if arr.size >= 56 and math.isfinite(sma[-6]) else sma[-2]
    if not math.isfinite(last) or not math.isfinite(prior) or prior == 0:
        trend = "flat"
    else:
        rel = (last - prior) / abs(prior)
        trend = "up" if rel > 0.0005 else "down" if rel < -0.0005 else "flat"

    window = 20
    h_arr = _as_1d(high) if high is not None else np.array([])
    l_arr = _as_1d(low) if low is not None else np.array([])
    use_hl = (
        h_arr.size == arr.size and l_arr.size == arr.size and arr.size > 0
        and np.all(np.isfinite(h_arr)) and np.all(np.isfinite(l_arr))
    )
    vol_series: list[float] = []
    if use_hl:
        for i in range(window, arr.size + 1):
            v = vol_parkinson(h_arr[i - window : i], l_arr[i - window : i])
            if v is not None:
                vol_series.append(v)
    else:
        rets = log_returns(arr)
        vol_series = [
            float(np.std(rets[i - window : i], ddof=1))
            for i in range(window, rets.size + 1)
            if np.all(np.isfinite(rets[i - window : i]))
        ]
    if len(vol_series) < 5:
        vol_bucket = "mid"
    else:
        hist = np.asarray(vol_series[-252:])
        cur = hist[-1]
        pct = float(np.mean(hist <= cur))
        vol_bucket = "low" if pct < 0.33 else "high" if pct > 0.67 else "mid"

    h_exp = hurst_exponent(arr)
    mean_reversion = bool(h_exp is not None and h_exp < 0.45)
    return {"trend": trend, "vol_bucket": vol_bucket, "mean_reversion": mean_reversion,
            "label": f"{trend}_{vol_bucket}"}


# ===========================================================================
# v3 (Tur-16 WP-M) — Derin FİNANS MATEMATİĞİ genişlemesi. Aşağıdaki tüm bölümler
# yalnız numpy/scipy/sklearn kullanır (statsmodels'e YENİ bağımlılık eklenmez);
# aynı savunmacı sözleşme (kısa/NaN/dönüştürülemez veri -> None, İSTİSNA YOK)
# ve aynı ArrayLike imza deseni korunur. bkz. AI_V3_DERIN_BEYIN_SPEC.md §3.4.
# ===========================================================================

# ---------------------------------------------------------------------------
# GARCH(1,1) volatilite modeli
# ---------------------------------------------------------------------------
# Durağanlık kısıtının üst ucu. Hem alpha/beta tek tek üst sınırı hem alpha+beta kısıtı
# BU sabittir: fit tam bu değere oturduysa "durağan bulundu" DEĞİL "kısıt bağlandı"dır.
_GARCH_PERSIST_MAX = 0.999


def _garch11_sigma2_path(omega: float, alpha: float, beta: float,
                         resid2: np.ndarray, var0: float) -> np.ndarray:
    """sigma2_t = omega + alpha*eps_{t-1}^2 + beta*sigma2_{t-1} yinelemesi, sigma2_0=var0.

    Bu bir BİRİNCİ-DERECE DOĞRUSAL fark denklemidir; `scipy.signal.lfilter` (C) tam olarak
    bunu hesaplar: b=[1], a=[1,-beta], girdi u_k = omega + alpha*eps_k^2, başlangıç durumu
    zi=[beta*var0]. Python döngüsüyle BİT-DENK (ölçüldü: 12 parametre×3 uzunluk kombinasyonu,
    `np.array_equal` True) ve n=1000'de 295,8 µs → 14,9 µs (**19,9×**). Yinelemenin kendisi
    sıralıdır — vektörleştirilemez; kazanç yorumlayıcı döngüsünün C'ye taşınmasındandır."""
    n = int(resid2.size)
    if n <= 1:
        return np.full(1, var0, dtype=float)
    u = omega + alpha * resid2[:-1]
    tail, _ = _lfilter([1.0], [1.0, -beta], u, zi=np.array([beta * var0]))
    return np.concatenate((np.array([var0]), tail))


def _garch11_neg_loglik(params: np.ndarray, resid2: np.ndarray, var0: float) -> float:
    """GARCH(1,1) negatif log-olabilirlik (Gauss yenilik varsayımı, z_t~N(0,1)). Girdi
    KARELENMİŞ artıklardır (`resid2`): olabilirlik yalnız eps^2'ye bağlıdır ve kareyi her
    çağrıda yeniden almak boşunadır (optimizasyon binlerce çağrı yapar). Dejenere/
    sonlu-olmayan varyans yolunda BÜYÜK ceza (1e10) döner — scipy.optimize.minimize'ı
    çökertmeden kısıt-dışı bölgeden uzaklaştırır (arama sırasında geçici olarak ihlal
    edilebilir; nihai kısıt kontrolü garch11_fit'te)."""
    omega, alpha, beta = params
    sigma2 = _garch11_sigma2_path(float(omega), float(alpha), float(beta), resid2, var0)
    if np.any(sigma2 <= 0.0) or not np.all(np.isfinite(sigma2)):
        return 1e10
    nll = 0.5 * float(np.sum(np.log(2.0 * np.pi * sigma2) + resid2 / sigma2))
    return nll if math.isfinite(nll) else 1e10


def garch11_fit(returns: ArrayLike) -> dict | None:
    """GARCH(1,1) parametrelerini scipy MLE (SLSQP, kısıtlı) ile tahmin eder:
    sigma2_t = omega + alpha*eps_{t-1}^2 + beta*sigma2_{t-1}, eps_t = r_t - ortalama,
    z_t = eps_t/sigma_t ~ N(0,1). Kısıtlar: omega>0, alpha>=0, beta>=0, alpha+beta<1
    (durağanlık şartı — aksi halde koşulsuz varyans sonsuz/tanımsız). sigma2_0, örneklem
    varyansına (var0) sabitlenir (standart başlangıç önseli). En az 50 gözlem ister (GARCH
    MLE az veride kararsızdır); yakınsamazsa/kısıt ihlal edilirse None (İSTİSNA YOK).

    EWMA/RiskMetrics KÖPRÜSÜ: omega->0 VE alpha+beta->1 (alpha=1-lambda, beta=lambda) özel
    durumunda GARCH(1,1) yinelemesi `ewma_vol`ün TAM OLARAK aynısına indirgenir — RiskMetrics
    EWMA'sı, sabit-terimsiz (omega=0) ve durağan-OLMAYAN (alpha+beta=1, literatürde "IGARCH")
    bir GARCH(1,1) özel halidir. GARCH bunun sabit-omega'lı, durağan (koşulsuz varyansa
    ortalamaya-dönen) genellemesidir — bkz. `uncond_var` alanı ve `garch11_forecast`.

    BAŞLANGIÇ-NOKTASI DUYARLILIĞI: GARCH log-olabilirlik yüzeyi düz bölgeler/yerel-optimum
    içerebilir (bilinen bir GARCH-MLE zorluğu). Tek başlangıç noktası bazı serilerde erken
    (sahte) yakınsamaya yol açabildiği için ÇOKLU-BAŞLANGIÇ (5 aday alpha0/beta0 çifti)
    denenir; kısıtları sağlayan başarılı sonuçlar arasından EN YÜKSEK olabilirlikli (en
    düşük NLL) seçilir. Deterministiktir (rastgelelik yok — sabit aday listesi).

    ÖLÇEK (Tur-49 onarımı — ÇOKLU-BAŞLANGIÇ TEK BAŞINA YETMEZ): MLE, artıklar BİRİM
    VARYANSA normalize edildikten sonra koşulur; omega geri ölçeklenir. GARCH(1,1) ölçek-
    eşdeğişkendir (r -> c*r ise omega -> c^2*omega, alpha/beta DEĞİŞMEZ), dolayısıyla bu
    dönüşüm modeli değiştirmez — yalnız optimize edilen yüzeyi düzeltir. Günlük getiride
    var ~ 1e-4'tür; SLSQP'nin varsayılan sonlu-fark adımı (~1,5e-8) ve `ftol` bu ölçekte
    omega yönünde ANLAMSIZ kalır ve arama sık sık köşeye (omega alt sınırında + alpha+beta
    -> 0,999, ya da alpha=beta=0) çakılır. ÖLÇÜLDÜ (296 gerçek S&P sembolü, 1000 günlük
    log-getiri): onarımdan ÖNCE %8,4'te `uncond_var` örneklem varyansının 5 katından
    büyük, %1,4'ünde 0,2 katından küçük, %1,7'sinde fit tamamen None; en kötü sapma
    18.068x ve 1,3e-06x. SONRA: fit None %0,0 · 0,2x altı %0,0 · 5x üstü **1/296**
    (FICO, persistence 0,996 — köşede DEĞİL, yüksek kalıcılıkta 1/(1-a-b) teriminin
    GERÇEK duyarlılığı; kırpmak aşağıdaki IGARCH kusurunu geri getirirdi). En kötü
    sapma 5,13x / 0,81x, oran p90 3,57 -> 1,20 (bkz. tests/test_tur49_*).

    Dönüş: {'omega','alpha','beta','mean','loglik','persistence','uncond_var','last_sigma2',
    'last_resid2','n'}. **`uncond_var` None OLABİLİR**: `persistence` durağanlık kısıtının
    kenarına (0,999) oturduğunda seri IGARCH-benzeridir, koşulsuz varyans TANIMSIZDIR ve
    omega/(1-a-b) kısıttan türeyen bir yapaydan ibarettir — sayı basmak yerine None döner.
    Kısa-ufuklu `garch11_forecast` bu durumda da geçerlidir (koşullu varyans tanımlıdır);
    tanımsız olan yalnız uzun-ufuk SINIRIdır."""
    arr = _clean_1d(returns, min_len=50)
    if arr is None:
        return None
    mean = float(np.mean(arr))
    resid = arr - mean
    var0 = float(np.var(resid))
    if var0 <= 0.0 or not math.isfinite(var0):
        return None
    # ÖLÇEK NORMALİZASYONU (Tur-49) — bkz. docstring "ÖLÇEK" bölümü.
    resid2_scaled = (resid * resid) / var0          # birim varyanslı artıkların karesi
    bounds = [(1e-12, None), (0.0, _GARCH_PERSIST_MAX), (0.0, _GARCH_PERSIST_MAX)]
    constraints = [{"type": "ineq", "fun": lambda x: _GARCH_PERSIST_MAX - x[1] - x[2]}]
    starts = ((0.05, 0.90), (0.10, 0.85), (0.05, 0.80), (0.15, 0.75), (0.02, 0.95))
    best = None
    for a0, b0 in starts:
        x0 = np.array([max(1.0 - a0 - b0, 1e-10), a0, b0])
        try:
            result = minimize(
                _garch11_neg_loglik, x0, args=(resid2_scaled, 1.0), method="SLSQP",
                bounds=bounds, constraints=constraints, options={"maxiter": 300, "ftol": 1e-12},
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if not result.success:
            continue
        omega, alpha, beta = (float(v) for v in result.x)
        if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0:
            continue
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        return None
    omega_scaled, alpha, beta = (float(v) for v in best.x)
    omega = omega_scaled * var0                     # ölçek-eşdeğişkenlik: omega -> c^2*omega
    resid2 = resid * resid
    sigma2 = _garch11_sigma2_path(omega, alpha, beta, resid2, var0)
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
        return None
    # Jacobian düzeltmesi: y=c*x ise log L_Y = log L_X - n*ln(c); c=sqrt(var0).
    loglik = float(-best.fun) - 0.5 * arr.size * math.log(var0)
    # IGARCH KÖŞESİ: alpha+beta durağanlık KISITININ kenarına çakıldıysa, 1-(alpha+beta)
    # tahmin edilmiş bir büyüklük değil KISITIN ARTIĞIdır; omega/(1-a-b) o zaman bir
    # koşulsuz varyans DEĞİL, kısıttan türeyen bir yapaydır. Ölçüldü (296 gerçek sembol):
    # köşedeki fitlerin oranı örneklem varyansının 34-528 katı. Sayı basmak yerine None.
    persistence = alpha + beta
    at_boundary = persistence >= _GARCH_PERSIST_MAX - 1e-9
    return {
        "omega": omega, "alpha": alpha, "beta": beta, "mean": mean,
        "loglik": loglik, "persistence": float(persistence),
        "uncond_var": None if at_boundary else float(omega / (1.0 - persistence)),
        "last_sigma2": float(sigma2[-1]), "last_resid2": float(resid2[-1]),
        "n": int(arr.size),
    }


def garch11_forecast(returns_or_params: ArrayLike | dict, horizon: int = 1) -> float | None:
    """GARCH(1,1) h-adım ileri VOLATİLİTE (sigma; varyans DEĞİL — modülün per-bar vol
    konvansiyonu, bkz. dosya başı sözleşme) öngörüsü. İlk argüman ÖNCEDEN `garch11_fit` ile
    üretilmiş bir sözlük OLABİLİR (yeniden-fit maliyeti yok) VEYA ham getiri dizisi olabilir
    (bu durumda içeride garch11_fit çağrılır — kısa/yakınsamayan veri None üretir).

    Yineleme: E[sigma2_{t+1}] = omega + alpha*eps_t^2 + beta*sigma2_t (bilinen son bilgiyle);
    h>=2 için E[sigma2_{t+h}] = omega + (alpha+beta)*E[sigma2_{t+h-1}] (standart GARCH
    varyans-öngörü yinelemesi). horizon büyüdükçe tahmin `uncond_var`a yakınsar (durağansa);
    EWMA'da (omega=0 kabulü) böyle bir durağan sınır YOKTUR — GARCH'ı EWMA'dan ayıran budur
    (bkz. garch11_fit docstring EWMA köprüsü). horizon<1/geçersiz parametre → None."""
    if isinstance(returns_or_params, dict):
        params = returns_or_params
        required = ("omega", "alpha", "beta", "last_sigma2", "last_resid2")
        if not all(k in params for k in required):
            return None
        try:
            omega = float(params["omega"])
            alpha = float(params["alpha"])
            beta = float(params["beta"])
            sigma2 = float(params["last_sigma2"])
            resid2 = float(params["last_resid2"])
        except (TypeError, ValueError):
            return None
    else:
        fit = garch11_fit(returns_or_params)
        if fit is None:
            return None
        omega, alpha, beta = fit["omega"], fit["alpha"], fit["beta"]
        sigma2, resid2 = fit["last_sigma2"], fit["last_resid2"]
    try:
        h = int(horizon)
    except (TypeError, ValueError):
        return None
    if h < 1:
        return None
    if not all(math.isfinite(v) for v in (omega, alpha, beta, sigma2, resid2)):
        return None
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0:
        return None
    forecast_var = omega + alpha * resid2 + beta * sigma2
    if not math.isfinite(forecast_var) or forecast_var < 0.0:
        return None
    persistence = alpha + beta
    for _ in range(h - 1):
        forecast_var = omega + persistence * forecast_var
        if not math.isfinite(forecast_var) or forecast_var < 0.0:
            return None
    return float(math.sqrt(forecast_var))


# ---------------------------------------------------------------------------
# Aşırı Değer Teorisi (EVT) — kuyruk indeksi + POT/GPD VaR/ES
# ---------------------------------------------------------------------------
def evt_hill_tail_index(returns: ArrayLike, k: int) -> float | None:
    """Hill (1975) kuyruk-indeksi tahmincisi: |returns| büyüklük-sıralı istatistiklerinin
    en-büyük k tanesinden alfa (kuyruk ÜSTELİ; büyük alfa = ince kuyruk, küçük alfa [~2-4]
    finansal getirilerde tipik kalın-kuyruk bölgesi) tahmin eder:
    gamma_hill = (1/k)*sum(ln(X_(i)/X_(k+1))) (i=1..k, X BÜYÜKTEN KÜÇÜĞE sıralı büyüklük);
    alfa = 1/gamma_hill. k, [2, n-1) aralığında olmalı (k çok küçükse gürültülü, çok
    büyükse merkez-kütleyi kirletir — klasik "Hill ufku" ödünleşimi, bkz. evt_gpd_var_es
    threshold_q ile aynı ödünleşim). Kısa/NaN veri veya k sınır-dışı → None."""
    arr = _clean_1d(returns, min_len=10)
    if arr is None:
        return None
    ki = int(k)
    n = arr.size
    if ki < 2 or ki >= n:
        return None
    mags = np.sort(np.abs(arr))[::-1]
    mags = mags[mags > 0.0]
    if mags.size <= ki:
        return None
    top = mags[: ki + 1]
    x_k1 = float(top[-1])
    if x_k1 <= 0.0:
        return None
    logs = np.log(top[:-1] / x_k1)
    gamma_hill = float(np.mean(logs))
    if gamma_hill <= 0.0 or not math.isfinite(gamma_hill):
        return None
    return float(1.0 / gamma_hill)


def evt_gpd_var_es(returns: ArrayLike, alpha: float = 0.01,
                    threshold_q: float = 0.95) -> dict | None:
    """POT (Peaks-Over-Threshold) + Genelleştirilmiş Pareto Dağılımı (GPD) ile kuyruk-ötesi
    VaR ve Expected Shortfall (McNeil & Frey 2000) — Cornish-Fisher'in (var_cornish_fisher)
    momentlere dayalı yaklaşıklığının kuyruk-teorisi tabanlı eşi. Kayıp = -returns; eşik u =
    kayıpların threshold_q yüzdelik değeri; u-üstü AŞIMLAR (exceedance = kayıp-u) scipy
    genpareto.fit (floc=0, konum sıfıra sabit — aşımlar tanım gereği sıfırdan başlar) ile
    MLE'lenir → şekil (xi) + ölçek (beta).

    VaR_alpha = u + (beta/xi)*[ (n/Nu·alpha)^(-xi) - 1 ]   (xi != 0; xi->0 limitinde
                u + beta*ln(Nu/(n·alpha)) — kod bu limiti AYRICA hesaplar, 0/0 riski yok)
    ES_alpha  = VaR_alpha/(1-xi) + (beta - xi·u)/(1-xi)    (xi<1 şartıyla sonlu)

    `alpha` (tail probability) EŞİĞİN ÖTESİNE ekstrapolasyon anlamlıdır — alpha >= empirik
    aşım-oranı (Nu/n) İSE (istenen eşik zaten gözlenenden daha SIK) None (POT'un amacı
    dışı kullanım). threshold_q/alpha (0,1) dışıysa, aşım sayısı Nu<10 (dejenere fit) veya
    xi>=1 (ES sonsuz) ise None. En az 50 gözlem ister.

    Dönüş: {'threshold','shape','scale','n_exceed','var','es'}."""
    arr = _clean_1d(returns, min_len=50)
    if arr is None or not (0.0 < alpha < 1.0) or not (0.0 < threshold_q < 1.0):
        return None
    losses = -arr
    n = losses.size
    u = float(np.percentile(losses, threshold_q * 100.0))
    exceed = losses[losses > u] - u
    nu = int(exceed.size)
    if nu < 10:
        return None
    freq = nu / float(n)
    if alpha >= freq:
        return None
    try:
        xi, _loc, beta = stats.genpareto.fit(exceed, floc=0.0)
    except (ValueError, RuntimeError):
        return None
    xi, beta = float(xi), float(beta)
    if not math.isfinite(xi) or not math.isfinite(beta) or beta <= 0.0 or xi >= 1.0:
        return None
    if xi == 0.0:
        var_p = u + beta * math.log(freq / alpha)
    else:
        var_p = u + (beta / xi) * ((alpha / freq) ** (-xi) - 1.0)
    if not math.isfinite(var_p):
        return None
    es_p = var_p / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)
    if not math.isfinite(es_p) or es_p < var_p:
        return None
    return {"threshold": u, "shape": xi, "scale": beta, "n_exceed": nu,
            "var": float(var_p), "es": float(es_p)}


# ---------------------------------------------------------------------------
# Kopula bağımlılık modelleri (kuyruk-bağımlılık dahil)
# ---------------------------------------------------------------------------
def copula_fit_gaussian(u: ArrayLike, v: ArrayLike) -> dict | None:
    """İki-değişkenli GAUSS kopulası uyumu: u,v ham getiri veya ÖNCEDEN uniform pseudo-
    gözlem OLABİLİR — içeride ampirik rank/(n+1) dönüşümüyle uniform hale getirilir (copula
    parametreleri marjinal dağılımdan BAĞIMSIZDIR; Sklar teoreminin temel sonucu). Uniform
    gözlemler standart-normale (stats.norm.ppf) taşınır; rho = normal-skorların Pearson
    korelasyonu (Gauss kopulasında bu KAPALI-FORM MLE'dir, sayısal optimizasyon gerekmez).
    En az 10 eşleşmiş nokta ister; uyumsuz-uzunluk/kısa/NaN/dejenere (sabit seri) → None.

    Dönüş: {'rho','loglik','n'}."""
    ua = _clean_1d(u, min_len=10)
    if ua is None:
        return None
    va = _clean_1d(v, min_len=10)
    if va is None or ua.size != va.size:
        return None
    n = ua.size
    ru = stats.rankdata(ua) / (n + 1.0)
    rv = stats.rankdata(va) / (n + 1.0)
    x = stats.norm.ppf(ru)
    y = stats.norm.ppf(rv)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return None
    rho = float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / (sx * sy))
    rho = float(np.clip(rho, -0.999, 0.999))
    one_minus_r2 = 1.0 - rho**2
    quad = (rho**2 * (x**2 + y**2) - 2.0 * rho * x * y) / (2.0 * one_minus_r2)
    loglik = float(np.sum(-0.5 * math.log(one_minus_r2) - quad))
    if not math.isfinite(loglik):
        return None
    return {"rho": rho, "loglik": loglik, "n": int(n)}


def copula_fit_t(u: ArrayLike, v: ArrayLike) -> dict | None:
    """İki-değişkenli Student-t kopulası uyumu: rho, Kendall tau'dan KAPALI-FORM tahmin
    edilir (rho = sin(pi/2·tau) — eliptik kopulaların rho-tau bağıntısı, dağılım-serbest ve
    sağlam); serbestlik derecesi (nu) ise rho SABİTLENEREK 1B profil-olabilirlik üzerinde
    scipy.optimize.minimize_scalar (yöntem='bounded', nu∈[2.01,60]) ile MLE'lenir
    (DETERMİNİSTİK — rastgelelik yok, sınırlı-Brent araması). t-kopulası Gauss kopulasının
    aksine KUYRUK BAĞIMLILIĞI taşır (nu küçüldükçe artar; bkz. tail_dependence) — finansal
    ortak-çöküş (co-crash) riskini Gauss kopulasından daha gerçekçi yakalar. En az 10 nokta
    ister; uyumsuz-uzunluk/kısa/NaN/dejenere → None.

    Dönüş: {'rho','nu','loglik','n'}."""
    ua = _clean_1d(u, min_len=10)
    if ua is None:
        return None
    va = _clean_1d(v, min_len=10)
    if va is None or ua.size != va.size:
        return None
    n = ua.size
    tau_result = stats.kendalltau(ua, va)
    tau = float(tau_result[0])
    if not math.isfinite(tau):
        return None
    rho = float(np.clip(math.sin(math.pi / 2.0 * tau), -0.999, 0.999))
    ru = stats.rankdata(ua) / (n + 1.0)
    rv = stats.rankdata(va) / (n + 1.0)

    def _neg_loglik_nu(nu_val: float) -> float:
        nu = float(nu_val)
        if nu <= 2.0 or not math.isfinite(nu):
            return 1e10
        x = stats.t.ppf(ru, df=nu)
        y = stats.t.ppf(rv, df=nu)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return 1e10
        one_minus_r2 = 1.0 - rho**2
        const = (
            math.lgamma((nu + 2.0) / 2.0) + math.lgamma(nu / 2.0)
            - 2.0 * math.lgamma((nu + 1.0) / 2.0) - 0.5 * math.log(one_minus_r2)
        )
        quad = (nu + 2.0) / 2.0 * np.log1p(
            (x**2 - 2.0 * rho * x * y + y**2) / (nu * one_minus_r2)
        )
        marg = (nu + 1.0) / 2.0 * (np.log1p(x**2 / nu) + np.log1p(y**2 / nu))
        total = float(np.sum(const - quad + marg))
        return -total if math.isfinite(total) else 1e10

    try:
        result = minimize_scalar(_neg_loglik_nu, bounds=(2.01, 60.0), method="bounded")
    except (ValueError, FloatingPointError):
        return None
    if not result.success:
        return None
    nu_hat = float(result.x)
    return {"rho": rho, "nu": nu_hat, "loglik": float(-result.fun), "n": int(n)}


def _pseudo_uniform_pair(u: ArrayLike, v: ArrayLike,
                         min_len: int = 10) -> tuple[np.ndarray, np.ndarray] | None:
    """(u,v) → ampirik pseudo-gözlem çifti (rank/(n+1)). Kopula parametreleri marjinal
    dağılımdan BAĞIMSIZDIR (Sklar) — bu yüzden ham getiri de uniform girdi de kabul edilir.
    Uyumsuz uzunluk / kısa / NaN / dejenere (tüm-eşit) → None."""
    ua = _clean_1d(u, min_len=min_len)
    if ua is None:
        return None
    va = _clean_1d(v, min_len=min_len)
    if va is None or ua.size != va.size:
        return None
    if float(np.std(ua)) == 0.0 or float(np.std(va)) == 0.0:
        return None
    n = ua.size
    return stats.rankdata(ua) / (n + 1.0), stats.rankdata(va) / (n + 1.0)


def _clayton_loglik(theta: float, ru: np.ndarray, rv: np.ndarray) -> float:
    """Clayton kopula log-olabilirliği. Yoğunluk:
    c(u,v) = (1+θ)·(uv)^(−θ−1)·(u^−θ + v^−θ − 1)^(−2−1/θ)."""
    if theta <= 0.0 or not math.isfinite(theta):
        return -math.inf
    s = ru ** (-theta) + rv ** (-theta) - 1.0
    if np.any(s <= 0.0):
        return -math.inf
    ll = float(np.sum(math.log1p(theta) - (theta + 1.0) * (np.log(ru) + np.log(rv))
                      - (2.0 + 1.0 / theta) * np.log(s)))
    return ll if math.isfinite(ll) else -math.inf


def _gumbel_loglik(theta: float, ru: np.ndarray, rv: np.ndarray) -> float:
    """Gumbel kopula log-olabilirliği. x=−ln u, y=−ln v, w=(x^θ+y^θ)^(1/θ):
    c(u,v) = C(u,v)·(uv)^(−1)·(xy)^(θ−1)·w^(1−2θ)·(w+θ−1),  C(u,v)=exp(−w).
    θ=1'de yoğunluk özdeş 1'dir (bağımsızlık) — birim testin çapası budur."""
    if theta < 1.0 or not math.isfinite(theta):
        return -math.inf
    x = -np.log(ru)
    y = -np.log(rv)
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        return -math.inf
    w = (x ** theta + y ** theta) ** (1.0 / theta)
    if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
        return -math.inf
    ll = float(np.sum(-w - np.log(ru) - np.log(rv) + (theta - 1.0) * (np.log(x) + np.log(y))
                      + (1.0 - 2.0 * theta) * np.log(w) + np.log(w + theta - 1.0)))
    return ll if math.isfinite(ll) else -math.inf


def _fit_archimedean(u: ArrayLike, v: ArrayLike, family: str) -> dict | None:
    """Clayton/Gumbel ortak uyum yolu: Kendall tau TERSİNMESİ ile başlangıç + 1B sınırlı
    MLE inceltmesi (`minimize_scalar`, bounded Brent — DETERMİNİSTİK, rastgelelik yok;
    `copula_fit_t`'nin nu profili ile AYNI desen).

    ÖNEMLİ SÖZLEŞME: iki aile de YALNIZ POZİTİF bağımlılık modeller (Clayton θ>0,
    Gumbel θ≥1 ⇔ tau≥0). Ampirik tau ≤ 0 ise uyum ZORLANMAZ → None + `reason`
    döndürülmez, doğrudan None (aile uygulanamaz; uydurma θ üretmek Tur-31 IV Rank
    dersinin tekrarı olurdu). Aile seçimini `copula_select` yapar."""
    pair = _pseudo_uniform_pair(u, v)
    if pair is None:
        return None
    ru, rv = pair
    tau_res = stats.kendalltau(ru, rv)
    tau = float(tau_res[0])
    if not math.isfinite(tau) or tau <= 0.0:
        return None
    tau_c = min(tau, 0.95)          # tersinme tau→1'de patlar; sayısal tavan
    if family == "clayton":
        theta0 = 2.0 * tau_c / (1.0 - tau_c)          # tau = θ/(θ+2)
        bounds, loglik = (1e-4, 60.0), _clayton_loglik
    else:
        theta0 = 1.0 / (1.0 - tau_c)                  # tau = 1 − 1/θ
        bounds, loglik = (1.0, 40.0), _gumbel_loglik
    try:
        res = minimize_scalar(lambda t: -loglik(float(t), ru, rv),
                              bounds=bounds, method="bounded")
    except (ValueError, FloatingPointError):   # pragma: no cover - savunmacı
        return None
    theta = float(res.x) if res.success else float(np.clip(theta0, *bounds))
    ll = loglik(theta, ru, rv)
    if not math.isfinite(ll):
        return None
    lam = copula_tail_dependence(family, {"theta": theta}) or {}
    return {"family": family, "theta": theta, "theta_tau": float(theta0),
            "tau": tau, "loglik": ll, "n_params": 1, "aic": float(2.0 - 2.0 * ll),
            "lambda_lower": lam.get("lower"), "lambda_upper": lam.get("upper"),
            "n": int(ru.size)}


def copula_fit_clayton(u: ArrayLike, v: ArrayLike) -> dict | None:
    """CLAYTON kopulası (asimetrik, ALT kuyruk bağımlı) — `knowledge.py`'nin kendi
    kataloğunda *"asimetrik kopulalar (ör. Clayton/Gumbel) bu modülde YOKTUR"* diye yazılı
    boşluğun yarısı.

    NEDEN FİNANSTA ÖNEMLİ: Gauss kopulası her iki kuyrukta da bağımlılığı SIFIR varsayar,
    t kopulası ise SİMETRİK varsayar (alt=üst). Gerçek hisse getirilerinde düşüş kuyruğu
    yükseliş kuyruğundan KALINDIR — ortak çöküş (co-crash) ortak ralliden daha olasıdır.
    Clayton tam olarak bunu modeller: λ_L = 2^(−1/θ) > 0, λ_U = 0.

    θ Kendall tau tersinmesiyle başlatılır (τ = θ/(θ+2)), sonra 1B sınırlı MLE ile
    inceltilir. **Pozitif bağımlılık gerektirir** — ampirik τ ≤ 0 ise None (aile
    uygulanamaz; bu bir hata değil, ölçülmüş bir UYGUNSUZLUKTUR).

    Dönüş: {'family','theta','theta_tau','tau','loglik','aic','lambda_lower',
    'lambda_upper','n_params','n'}."""
    return _fit_archimedean(u, v, "clayton")


def copula_fit_gumbel(u: ArrayLike, v: ArrayLike) -> dict | None:
    """GUMBEL kopulası (asimetrik, ÜST kuyruk bağımlı) — boşluğun diğer yarısı.
    λ_U = 2 − 2^(1/θ) > 0, λ_L = 0; θ=1 bağımsızlıktır.

    Clayton'ın AYNASIDIR: ortak RALLİ riskini (ör. kısa pozisyonların birlikte sıkışması,
    momentum sepetinin birlikte fırlaması) yakalar. İkisini birlikte uydurup AIC ile
    kıyaslamak, bağımlılığın hangi kuyrukta olduğunu VERİDEN sordurur — varsaymak yerine
    (`copula_select`).

    Dönüş: `copula_fit_clayton` ile AYNI anahtarlar."""
    return _fit_archimedean(u, v, "gumbel")


def copula_tail_dependence(family: str, params: dict) -> dict | None:
    """Aile + parametre → TEORİK kuyruk-bağımlılık katsayıları (kapalı form). Ampirik
    `tail_dependence()`in model karşılığıdır: biri veriden sayar, bu türetir; ikisinin
    AYRIŞMASI aile seçiminin yanlış olduğunun teşhisidir.

      · gaussian : λ_L = λ_U = 0  (ρ<1 iken — Gauss kopulasının bilinen kusuru)
      · t        : λ_L = λ_U = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ)))  (SİMETRİK)
      · clayton  : λ_L = 2^(−1/θ), λ_U = 0
      · gumbel   : λ_L = 0, λ_U = 2 − 2^(1/θ)

    Bilinmeyen aile / eksik-geçersiz parametre → None (varsayılan 0 UYDURULMAZ)."""
    fam = str(family).strip().lower()
    if fam == "gaussian":
        rho = params.get("rho")
        if rho is None or not math.isfinite(float(rho)) or abs(float(rho)) >= 1.0:
            return None
        return {"family": fam, "lower": 0.0, "upper": 0.0}
    if fam == "t":
        rho, nu = params.get("rho"), params.get("nu")
        if rho is None or nu is None:
            return None
        r, d = float(rho), float(nu)
        if not (math.isfinite(r) and math.isfinite(d)) or abs(r) >= 1.0 or d <= 0.0:
            return None
        lam = 2.0 * float(stats.t.cdf(-math.sqrt((d + 1.0) * (1.0 - r) / (1.0 + r)), df=d + 1.0))
        lam = float(np.clip(lam, 0.0, 1.0))
        return {"family": fam, "lower": lam, "upper": lam}
    if fam in {"clayton", "gumbel"}:
        theta = params.get("theta")
        if theta is None or not math.isfinite(float(theta)):
            return None
        t = float(theta)
        if fam == "clayton":
            if t <= 0.0:
                return None
            return {"family": fam, "lower": float(2.0 ** (-1.0 / t)), "upper": 0.0}
        if t < 1.0:
            return None
        return {"family": fam, "lower": 0.0,
                "upper": float(np.clip(2.0 - 2.0 ** (1.0 / t), 0.0, 1.0))}
    return None


def copula_select(u: ArrayLike, v: ArrayLike) -> dict | None:
    """DÖRT aileyi (gaussian · t · clayton · gumbel) aynı pseudo-gözlemlerde uydurup AIC
    ile sıralar ve kuyruk yapısını VERİYE sordurur.

    NEDEN AIC: aileler farklı sayıda parametreye sahiptir (gaussian 1, t 2, arşimedyen 1);
    çıplak log-olabilirlik karşılaştırması t kopulasını yapısı gereği kayırırdı.

    DÜRÜSTLÜK KAPILARI:
      · Pozitif bağımlılık gerektiren aileler (clayton/gumbel) τ ≤ 0'da UYDURULMAZ ve
        tabloya `None` olarak girer — "uymadı" ile "denenmedi" ayrımı korunur.
      · `empirical` alanı ampirik λ_L/λ_U'yu taşır; kazanan ailenin TEORİK λ'sıyla
        kıyaslanabilsin diye. Seçim yalnız AIC'ye bakar — ampirik λ bir KONTROL,
        kapı değildir (n küçükken ampirik kuyruk tahmini çok gürültülüdür).
      · ΔAIC < 2 ise `decisive=False`: AIC farkı bu mertebede aile ayırt EDEMEZ
        (Burnham & Anderson kuralı). Kazananı "bulundu" diye okumak Tur-23'ün
        "kapı PASS dese bile n_kept'e bak" dersinin kopula karşılığıdır.

    Dönüş: {'best','decisive','delta_aic','fits':{aile: dict|None},'empirical':dict|None,'n'}
    veya girdi kullanılamazsa None."""
    pair = _pseudo_uniform_pair(u, v, min_len=20)
    if pair is None:
        return None
    ru, rv = pair
    fits: dict[str, dict | None] = {
        "gaussian": copula_fit_gaussian(ru, rv),
        "t": copula_fit_t(ru, rv),
        "clayton": copula_fit_clayton(ru, rv),
        "gumbel": copula_fit_gumbel(ru, rv),
    }
    n_par = {"gaussian": 1, "t": 2, "clayton": 1, "gumbel": 1}
    for fam, fit in fits.items():
        if fit is None:
            continue
        fit.setdefault("family", fam)
        fit.setdefault("n_params", n_par[fam])
        fit["aic"] = float(2.0 * n_par[fam] - 2.0 * float(fit["loglik"]))
        if "lambda_lower" not in fit:
            lam = copula_tail_dependence(fam, fit) or {}
            fit["lambda_lower"], fit["lambda_upper"] = lam.get("lower"), lam.get("upper")
    scored = sorted(((f["aic"], k) for k, f in fits.items() if f is not None))
    if not scored:
        return None
    best_aic, best = scored[0]
    delta = float(scored[1][0] - best_aic) if len(scored) > 1 else None
    return {"best": best, "decisive": bool(delta is not None and delta >= 2.0),
            "delta_aic": delta, "fits": fits,
            "empirical": tail_dependence(ru, rv), "n": int(ru.size)}


def tail_dependence(u: ArrayLike, v: ArrayLike, q: float = 0.10) -> dict | None:
    """Ampirik kuyruk-bağımlılık katsayıları (alt lambda_L, üst lambda_U) — u,v'nin (ham
    veya uniform) ORTAK kuyruk-olay olasılığı: lambda_U(q) ~= P(V>1-q | U>1-q) tahmini =
    #{pu_i>1-q VE pv_i>1-q} / (n·q); lambda_L(q) simetrik alt-kuyruk tanımı. q->0 teorik
    limite yaklaşır; sabit küçük q (varsayılan 0.10) pratik/gürültü ödünleşimidir (küçük q
    → az gözlem → yüksek varyans, bkz. evt_hill_tail_index k ödünleşimi). Bağımsız
    değişkenlerde lambda~0, komonoton (u=v) değişkenlerde lambda~1 beklenir. q∉(0,0.5)
    veya <20 eşleşmiş nokta/NaN/uyumsuz-uzunluk → None.

    Dönüş: {'lower','upper','n'}."""
    ua = _clean_1d(u, min_len=20)
    if ua is None:
        return None
    va = _clean_1d(v, min_len=20)
    if va is None or ua.size != va.size:
        return None
    qf = float(q)
    if not (0.0 < qf < 0.5):
        return None
    n = ua.size
    pu = stats.rankdata(ua) / (n + 1.0)
    pv = stats.rankdata(va) / (n + 1.0)
    upper = float(np.mean((pu > 1.0 - qf) & (pv > 1.0 - qf)) / qf)
    lower = float(np.mean((pu <= qf) & (pv <= qf)) / qf)
    return {"lower": float(np.clip(lower, 0.0, 1.0)), "upper": float(np.clip(upper, 0.0, 1.0)),
            "n": int(n)}


# ---------------------------------------------------------------------------
# Kesirli fark alma (durağanlık <-> hafıza dengesi)
# ---------------------------------------------------------------------------
def _ffd_weights(d: float, threshold: float) -> np.ndarray | None:
    """López de Prado FFD ağırlıkları: w_0=1, w_k = -w_{k-1}·(d-k+1)/k (binom serisinin
    (1-L)^d açılımı). |w_k|<threshold olduğunda kesilir. 10.000 adımda hâlâ eşiğin altına
    inmezse (yakınsamıyor) None (güvenlik sınırı — pencere sonsuza büyümez)."""
    weights = [1.0]
    k = 1
    max_k = 10_000
    while k <= max_k:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            return np.array(weights, dtype=float)
        weights.append(w_k)
        k += 1
    return None


def frac_diff_ffd(series: ArrayLike, d: float, threshold: float = 1e-5) -> np.ndarray | None:
    """López de Prado Sabit-Pencereli Kesirli Fark Alma (FFD, "Advances in Financial Machine
    Learning" Böl. 5): d=0 → seri AYNEN korunur (bellek tam), d=1 → sıradan birinci farka
    yakınsar (bellek sıfır, np.diff ile örtüşür); ARADAKİ kesirli d değerleri durağanlık
    (ADF anlamında) ile hafıza/öngörülebilirlik arasında AYARLANABİLİR bir denge kurar —
    tamsayı fark alma genellikle GEREĞİNDEN FAZLA bellek siler. Pencere `_ffd_weights` ile
    sabit uzunlukta kesilir (genişleyen-pencere DEĞİL — "FFD" adının kaynağı); ilk
    (pencere-1) nokta NaN (yetersiz geçmiş). threshold<=0/yakınsamayan ağırlık/pencereden
    kısa seri/NaN girdi → None.

    Not: `threshold` küçüldükçe pencere büyür (daha uzun hafıza, daha yavaş hesap)."""
    arr = _clean_1d(series, min_len=2)
    if arr is None:
        return None
    try:
        df = float(d)
        th = float(threshold)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(df) or not math.isfinite(th) or th <= 0.0:
        return None
    weights = _ffd_weights(df, th)
    if weights is None:
        return None
    window = weights.size
    if window > arr.size:
        return None
    out = np.full(arr.size, np.nan)
    for t in range(window - 1, arr.size):
        seg = arr[t - window + 1 : t + 1][::-1]
        out[t] = float(np.dot(weights, seg))
    return out


# ---------------------------------------------------------------------------
# Spektral / dalgacık ayrıştırma
# ---------------------------------------------------------------------------
def welch_psd(series: ArrayLike, fs: float = 1.0) -> dict | None:
    """Welch yöntemiyle güç spektral yoğunluğu (PSD) tahmini (scipy.signal.welch;
    ortalamalı-periodogram — ham periodogramdan düşük varyanslı). fs = örnekleme frekansı
    (varsayılan 1.0 = "bar başına 1 birim", ör. günlük seri için 1 gün^-1). nperseg otomatik
    min(256, n)'e ayarlanır. DC (frekans=0, ortalama seviye) bileşeni `dominant_freq`
    aramasından HARİÇ tutulur (döngüsellik değil, seviye bilgisi taşır). En az 8 gözlem
    ister; fs<=0/kısa/NaN/hesap hatası → None.

    Dönüş: {'freqs','psd','dominant_freq','total_power'}."""
    arr = _clean_1d(series, min_len=8)
    if arr is None:
        return None
    try:
        fsf = float(fs)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fsf) or fsf <= 0.0:
        return None
    nperseg = min(256, arr.size)
    try:
        freqs, psd = _welch(arr, fs=fsf, nperseg=nperseg)
    except ValueError:
        return None
    if not np.all(np.isfinite(psd)) or psd.size < 2:
        return None
    dominant_idx = int(np.argmax(psd[1:]) + 1)
    return {
        "freqs": freqs.tolist(), "psd": psd.tolist(),
        "dominant_freq": float(freqs[dominant_idx]), "total_power": float(np.sum(psd)),
    }


def haar_dwt_vol(returns: ArrayLike, levels: int = 3) -> dict | None:
    """Saf-numpy Haar Ayrık Dalgacık Dönüşümü (DWT) ile ÇOK-ÖLÇEKLİ volatilite ayrışımı.
    Her seviyede: a[i]=(x[2i]+x[2i+1])/sqrt2 (yaklaşıklık/düşük-frekans), d[i]=(x[2i]-x[2i+1])
    /sqrt2 (detay/yüksek-frekans); yaklaşıklık bir SONRAKİ seviyeye özyinelemeli girdi olur
    (Mallat piramidi). `energy_ratio[level_k]` = o seviyenin detay-enerjisinin TOPLAM
    enerjiye oranı — hangi zaman-ölçeğinin volatiliteye ne kadar katkı verdiğini gösterir
    (ör. yüksek level_1 payı = kısa-vadeli/gürültü baskın). Seri, look-ahead YARATMADAN
    (yalnız KIRPMA — baştan değil SONDAN alınır) 2^levels'in katı uzunluğa indirgenir.
    levels<1 veya seri 2^levels'ten kısa/NaN → None.

    Dönüş: {'levels','detail_energy','approx_energy','energy_ratio','n_used'}."""
    arr = _clean_1d(returns, min_len=4)
    if arr is None:
        return None
    lv = int(levels)
    if lv < 1:
        return None
    step = 2**lv
    max_len = (arr.size // step) * step
    if max_len < step:
        return None
    x = arr[-max_len:].astype(float)
    detail_energy: list[float] = []
    approx = x
    for _ in range(lv):
        n = approx.size
        if n % 2 != 0 or n < 2:
            return None
        evens = approx[0::2]
        odds = approx[1::2]
        a = (evens + odds) / math.sqrt(2.0)
        d = (evens - odds) / math.sqrt(2.0)
        detail_energy.append(float(np.sum(d**2)))
        approx = a
    approx_energy = float(np.sum(approx**2))
    total_energy = approx_energy + sum(detail_energy)
    energy_ratio = {
        f"level_{i + 1}": (e / total_energy if total_energy > 0.0 else 0.0)
        for i, e in enumerate(detail_energy)
    }
    return {
        "levels": lv, "detail_energy": detail_energy, "approx_energy": approx_energy,
        "energy_ratio": energy_ratio, "n_used": int(max_len),
    }


# ---------------------------------------------------------------------------
# Optimal yürütme (Almgren-Chriss) — YALNIZ RAPOR; execution/emir sistemine BAĞLANMAZ
# ---------------------------------------------------------------------------
def almgren_chriss_cost(q: float, sigma: float, eta: float, gamma_risk: float,
                        t: float, n: int) -> dict | None:
    """Almgren-Chriss (2001) ortalama-varyans optimal tasfiye yörüngesi ve maliyet ayrışımı
    (yalnız GEÇİCİ/temporary etki modellenir — kalıcı/permanent etki parametresi bu
    basitleştirilmiş sürümde YOKTUR; sözleşmede yalnız `eta` var). q = tasfiye edilecek
    miktar, sigma = birim-zaman volatilite, eta = geçici-etki katsayısı, gamma_risk = risk-
    kaçınma (lambda), t = ufuk, n = eşit-adım sayısı.

    kappa = arccosh(1 + gamma_risk·sigma²·tau²/(2·eta)) / tau  (tau=t/n); optimal elde-tutma
    yörüngesi x_j = q·sinh(kappa·(t-t_j)) / sinh(kappa·t). gamma_risk->0 VEYA sigma->0
    (risk-nötr sınır) durağan biçimde DOĞRUSAL (TWAP) yörüngeye indirgenir — bu KAPALI-FORM
    bir yaklaşım DEĞİL, açık kod-dallanmasıdır (0/0 riskini önler). Maliyet, yörüngeden İKİ
    BAĞIMSIZ standart bileşenle hesaplanır (kapalı-form E[maliyet] türetimi yerine —
    sayısal sağlamlık tercih edilmiştir): geçici-etki maliyeti = sum(eta·trade_j²/tau),
    tutma-riski varyansı = sigma²·sum(x_j²·tau) (j=0..n-1 aralığında tutulan pozisyon).

    ANAYASA NOTU: bu fonksiyon YALNIZ RAPOR/gölge amaçlıdır; hiçbir execution/emir
    sistemine bağlanmaz (bkz. AI_V3_DERIN_BEYIN_SPEC.md madde 1). Geçersiz girdi
    (eta<=0, t<=0, n<1, sigma<0, gamma_risk<0, sonlu-değil) → None.

    Dönüş: {'trajectory','trades','expected_temp_impact_cost','cost_variance',
            'risk_adjusted_cost','kappa'}."""
    try:
        qf, sigf, etaf, gr, tf = float(q), float(sigma), float(eta), float(gamma_risk), float(t)
        ni = int(n)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (qf, sigf, etaf, gr, tf)):
        return None
    if etaf <= 0.0 or tf <= 0.0 or ni < 1 or sigf < 0.0 or gr < 0.0:
        return None
    tau = tf / ni
    j = np.arange(0, ni + 1, dtype=float)
    t_j = j * tau
    kappa: float | None
    if gr <= 0.0 or sigf == 0.0:
        trajectory = qf * (1.0 - j / ni)
        kappa = None
    else:
        cosh_arg = 1.0 + gr * sigf**2 * tau**2 / (2.0 * etaf)
        if cosh_arg <= 1.0 + 1e-12:
            trajectory = qf * (1.0 - j / ni)
            kappa = None
        else:
            kap = math.acosh(cosh_arg) / tau
            denom = math.sinh(kap * tf)
            if denom == 0.0 or not math.isfinite(denom):
                trajectory = qf * (1.0 - j / ni)
                kappa = None
            else:
                trajectory = qf * np.sinh(kap * (tf - t_j)) / denom
                kappa = float(kap)
    trajectory = np.asarray(trajectory, dtype=float)
    if not np.all(np.isfinite(trajectory)):
        return None
    trades = -np.diff(trajectory)
    temp_impact_cost = float(np.sum(etaf * trades**2 / tau))
    variance_cost = float(sigf**2 * np.sum((trajectory[:-1] ** 2) * tau))
    risk_adjusted = temp_impact_cost + gr * variance_cost
    return {
        "trajectory": trajectory.tolist(), "trades": trades.tolist(),
        "expected_temp_impact_cost": temp_impact_cost, "cost_variance": variance_cost,
        "risk_adjusted_cost": float(risk_adjusted), "kappa": kappa,
    }


# ---------------------------------------------------------------------------
# Bilgi-kuramsal desen ölçüleri (entropi / karşılıklı bilgi)
# ---------------------------------------------------------------------------
def permutation_entropy(series: ArrayLike, m: int = 3, tau: int = 1) -> float | None:
    """Bandt-Pompe permütasyon entropisi (NORMALİZE, [0,1] — 0=tam düzenli/öngörülebilir,
    1=maksimum düzensiz), m-boyutlu gecikme-gömme vektörlerinin SIRALAMA (ordinal) örüntü
    dağılımının Shannon entropisi / log(m!) ile normalize edilir. tau = gecikme adımı.
    Fiyat/getiri serisinin DEĞERİNE değil SIRALAMA yapısına bakar — ölçek/monoton-dönüşüme
    dayanıklı bir düzensizlik ölçüsüdür (bkz. sample_entropy: genlik-tabanlı alternatif).
    m<2 (tekil örüntü, entropi tanımsız) veya m>10 (aşırı-büyük çarpınım, hesap/bellek
    güvenliği) veya tau<1 veya yetersiz-gömme-vektörü/kısa/NaN girdi → None."""
    arr = _clean_1d(series, min_len=1)
    mi, taui = int(m), int(tau)
    if arr is None or mi < 2 or mi > 10 or taui < 1:
        return None
    n_vectors = arr.size - (mi - 1) * taui
    if n_vectors < 1:
        return None
    counts: dict[tuple[int, ...], int] = {}
    for i in range(n_vectors):
        window = arr[i : i + mi * taui : taui]
        order = tuple(int(v) for v in np.argsort(window, kind="stable"))
        counts[order] = counts.get(order, 0) + 1
    total = float(n_vectors)
    probs = np.array([c / total for c in counts.values()])
    ent = float(-np.sum(probs * np.log(probs)))
    max_ent = math.log(math.factorial(mi))
    if max_ent <= 0.0 or not math.isfinite(ent):
        return None
    return float(ent / max_ent)


def sample_entropy(series: ArrayLike, m: int = 2, r: float = 0.2) -> float | None:
    """Sample Entropy (SampEn; Richman & Moorman 2000) — GENLİK-tabanlı düzensizlik ölçüsü
    (permutation_entropy'nin sıralama-tabanlı alternatifi). r, seri std'sinin BİR ORANI
    olarak yorumlanır (standart konvansiyon: mutlak tolerans = r·std(series)). m-boyutlu ve
    (m+1)-boyutlu şablon çiftleri arasında Chebyshev (max-norm) mesafe<=tolerans eşleşme
    SAYILARI A (m+1) ve B (m) hesaplanır (ORTAK şablon aralığı n-m — Richman-Moorman
    tanımıyla birebir); SampEn = -ln(A/B). Kendine-eşleşme HARİÇ tutulur. Sabit seri
    (std=0), eşleşme yok (A=0 veya B=0, log(0) riski) veya kısa/NaN/geçersiz r → None.
    Not: eşleşme sayımı k-d ağacıyla yapılır (aşağıya bkz.); kaba-kuvvet O(n²)
    referans gövdeyle BİT-DENKtir ve `tests/test_sample_entropy_kdtree.py` bunu kilitler."""
    arr = _clean_1d(series, min_len=1)
    mi = int(m)
    try:
        rf = float(r)
    except (TypeError, ValueError):
        return None
    if arr is None or mi < 1 or rf <= 0.0 or not math.isfinite(rf):
        return None
    n = arr.size
    if n < mi + 3:
        return None
    sd = float(np.std(arr, ddof=0))
    if sd == 0.0:
        return None
    tol = rf * sd
    n_templates = n - mi
    if n_templates < 2:
        return None

    def _match_count(dim: int) -> int:
        # Eski gövde n_templates² Chebyshev mesafesini AÇIKÇA kuruyordu (her i için
        # (n−i, dim) geçici dizi) — O(n²) hem zaman hem ayırma. cKDTree(p=inf) aynı
        # sayımı ağaçla yapar. `count_neighbors` SIRALI çiftleri sayar ve kendine-
        # eşleşmeleri İÇERİR (= n_templates + 2·çift), bu yüzden n_templates çıkarılır
        # → 2·çift, eski `total * 2` ile BİT-DENK: sayımlar tamsayıdır, mesafe aynı
        # max|a−b| aritmetiğidir ve sınır her ikisinde de `<= tol` (dahil).
        # ÖLÇÜLDÜ (bu makine, UÇTAN UCA `sample_entropy` çağrısı — yalnız sayım değil):
        # n=5000 0,867 → 0,089 s (**9,7×**) · n=10000 3,415 → 0,202 s (**16,9×**).
        # Uç noktanın tavanı n=1500'''de 22 ms. (Sayım çekirdeği tek başına 48,7× ama
        # `_clean_1d` + iki ağaç kurulumu payı var; yukarıdakiler gerçek ölçüm.)
        win = np.lib.stride_tricks.sliding_window_view(arr, dim)[:n_templates]
        tree = cKDTree(win)
        return int(tree.count_neighbors(tree, tol, p=np.inf)) - n_templates

    b_count = _match_count(mi)
    a_count = _match_count(mi + 1)
    if b_count == 0 or a_count == 0:
        return None
    return float(-math.log(a_count / b_count))


def mutual_info(x: ArrayLike, y: ArrayLike, bins: int = 16) -> float | None:
    """numpy 2B-histogram tabanlı karşılıklı bilgi (mutual information) tahmini —
    DETERMİNİSTİK (KNN-tabanlı sklearn.feature_selection.mutual_info_* tahmincilerinin
    aksine rastgelelik YOK). MI(X,Y) = sum(p(x,y)·log(p(x,y)/(p(x)p(y)))), 2B histogram
    hücrelerinden ampirik olasılıklarla. Bağımsız değişkenlerde ~0 (sonlu-örneklem
    yanlılığıyla küçük pozitif), tam-bağımlı (y=x) değişkenlerde yüksek. Doğrusal-OLMAYAN
    bağımlılığı da yakalar (korelasyonun aksine) — ör. y=x² gibi ilişkilerde korelasyon~0
    iken MI>0 olabilir. Sayısal gürültüyle oluşan hafif-negatif değer 0'a kırpılır.
    bins<2/uyumsuz-uzunluk/kısa/NaN girdi → None."""
    xa = _clean_1d(x, min_len=2)
    if xa is None:
        return None
    ya = _clean_1d(y, min_len=2)
    if ya is None or xa.size != ya.size:
        return None
    b = int(bins)
    if b < 2:
        return None
    hist_2d, _xe, _ye = np.histogram2d(xa, ya, bins=b)
    total_count = float(hist_2d.sum())
    if total_count <= 0.0:
        return None
    pxy = hist_2d / total_count
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pxy / (px * py)
    safe_ratio = np.where(pxy > 0.0, ratio, 1.0)  # pxy=0 hücreleri log(1)=0 ile maskelenir
    log_term = np.where(pxy > 0.0, np.log(safe_ratio), 0.0)
    mi_val = float(np.sum(pxy * log_term))
    if not math.isfinite(mi_val):
        return None
    return max(0.0, mi_val)


# ---------------------------------------------------------------------------
# Yüksek-moment ortak-risk tensörleri (coskewness / cokurtosis)
# ---------------------------------------------------------------------------
def coskewness(ret_matrix: ArrayLike) -> np.ndarray | None:
    """Standardize edilmiş 3-indisli ortak-çarpıklık (coskewness) tensörü: S[i,j,k] =
    mean_t(z[t,i]·z[t,j]·z[t,k]), z = (r-mean)/std sütun-standardize getiri. KÖŞEGEN
    S[i,i,i] tek-varlık örneklem çarpıklığına (scipy.stats.skew, bias=True) eşittir —
    tensör bunun çok-varlık genellemesidir; portföy çarpıklığı yalnız varyans/kovaryanstan
    (ledoit_wolf_cov/ewma_cov) DEĞİL bu ortak-momentlerden de etkilenir (negatif portföy
    çarpıklığı = eşzamanlı büyük-düşüş riski, çeşitlendirmeyle azalmayabilir). Simetriktir:
    S[i,j,k]=S[j,i,k]=... (herhangi indis permütasyonu altında değişmez). Girdi (n_obs,
    n_asset), n_obs>=10 ister; sıfır-varyans sütun/kısa/NaN/dejenere → None."""
    arr = _clean_2d(ret_matrix, min_rows=10, min_cols=1)
    if arr is None:
        return None
    n_obs = arr.shape[0]
    mu = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=0)
    if np.any(sd == 0.0):
        return None
    z = (arr - mu) / sd
    tensor = np.einsum("ti,tj,tk->ijk", z, z, z) / n_obs
    if not np.all(np.isfinite(tensor)):
        return None
    return np.asarray(tensor)


def cokurtosis(ret_matrix: ArrayLike) -> np.ndarray | None:
    """Standardize edilmiş 4-indisli ortak-basıklık (cokurtosis) tensörü: K[i,j,k,l] =
    mean_t(z[t,i]·z[t,j]·z[t,k]·z[t,l]). KÖŞEGEN K[i,i,i,i] tek-varlık HAM (Pearson, Fisher-
    DEĞİL — fazladan 3 çıkarılmamış) basıklığa (scipy.stats.kurtosis, fisher=False,
    bias=True) eşittir. coskewness'in dördüncü-moment genellemesidir; yüksek cokurtosis =
    varlıkların EŞ ZAMANLI aşırı-hareket etme eğilimi (çeşitlendirme kuyruk-riskini
    beklenenden az azaltır — portfolio.py'de HRP/CVaR-optimizasyonu için gölge girdi
    adayıdır, bkz. AI_V3_DERIN_BEYIN_PLAN §WP-P). Simetriktir (herhangi indis permütasyonu
    altında değişmez). Girdi (n_obs, n_asset), n_obs>=15 ister (4. moment daha çok veri
    ister); sıfır-varyans sütun/kısa/NaN/dejenere → None."""
    arr = _clean_2d(ret_matrix, min_rows=15, min_cols=1)
    if arr is None:
        return None
    n_obs = arr.shape[0]
    mu = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=0)
    if np.any(sd == 0.0):
        return None
    z = (arr - mu) / sd
    tensor = np.einsum("ti,tj,tk,tl->ijkl", z, z, z, z) / n_obs
    if not np.all(np.isfinite(tensor)):
        return None
    return np.asarray(tensor)


# ---------------------------------------------------------------------------
# Asimetrik GARCH ailesi (Tur-72 — yol haritası §6.1)
# ---------------------------------------------------------------------------
_EABS_Z = math.sqrt(2.0 / math.pi)       # Gauss yenilikte E|z|
_EGARCH_BETA_MAX = 0.999                 # log-varyans AR(1) durağanlık kenarı


def _gjr_sigma2_path(omega: float, alpha: float, gamma: float, beta: float,
                     resid: np.ndarray, var0: float) -> np.ndarray:
    """GJR(1,1,1) koşullu varyans yolu:
    sigma2_t = omega + (alpha + gamma·1[eps_{t-1}<0])·eps_{t-1}² + beta·sigma2_{t-1}.

    GARCH(1,1) ile AYNI birinci-derece doğrusal fark denklemidir — yalnız girdi terimi
    işarete bağlıdır. `_garch11_sigma2_path`'in `lfilter` kısayolu bu yüzden burada da
    geçerlidir (b=[1], a=[1,−beta], zi=[beta·var0])."""
    n = int(resid.size)
    if n <= 1:
        return np.full(1, var0, dtype=float)
    e2 = resid * resid
    neg = (resid < 0.0).astype(float)
    u = omega + (alpha + gamma * neg[:-1]) * e2[:-1]
    tail, _ = _lfilter([1.0], [1.0, -beta], u, zi=np.array([beta * var0]))
    return np.concatenate((np.array([var0]), tail))


def _gjr_neg_loglik(params: np.ndarray, resid: np.ndarray, var0: float) -> float:
    """GJR-GARCH negatif log-olabilirlik (Gauss yenilik). Dejenere varyans yolunda 1e10."""
    omega, alpha, gamma, beta = (float(v) for v in params)
    sigma2 = _gjr_sigma2_path(omega, alpha, gamma, beta, resid, var0)
    if np.any(sigma2 <= 0.0) or not np.all(np.isfinite(sigma2)):
        return 1e10
    nll = 0.5 * float(np.sum(np.log(2.0 * np.pi * sigma2) + (resid * resid) / sigma2))
    return nll if math.isfinite(nll) else 1e10


def gjr_garch_fit(returns: ArrayLike) -> dict | None:
    """GJR-GARCH(1,1,1) — Glosten-Jagannathan-Runkle (1993) EŞİK asimetrisi:

        sigma2_t = omega + (alpha + gamma·1[eps_{t-1} < 0])·eps_{t-1}² + beta·sigma2_{t-1}

    GARCH(1,1) şoku KARESİYLE ölçer, dolayısıyla +%3 ile −%3 aynı oynaklık tepkisini
    üretir. Hisse senedi getirilerinde bu ampirik olarak yanlıştır (kaldıraç etkisi):
    NEGATİF şok oynaklığı daha çok artırır. `gamma` tam olarak bu farktır ve **gamma > 0
    kaldıraç etkisi** demektir. gamma = 0'da model GARCH(1,1)'e indirgenir — yani
    `garch11_fit` bu ailenin kısıtlı özel halidir ve iki fitin log-olabilirliği
    karşılaştırılabilir (fark ≈ ½·χ²(1) olabilirlik-oranı testi).

    Kısıtlar: omega>0, alpha≥0, alpha+gamma≥0 (negatif barda da varyans katkısı negatif
    olamaz), beta≥0 ve `persistence` = alpha + gamma/2 + beta < 0,999. gamma/2 katsayısı
    simetrik yenilikte E[1[eps<0]] = ½ olmasından gelir.

    `garch11_fit`'in ÖLÇEK NORMALİZASYONU (Tur-49) burada da uygulanır: MLE birim
    varyanslı artıklarla koşulur, omega geri ölçeklenir (GJR de ölçek-eşdeğişkendir).
    Çoklu-başlangıç deterministiktir. En az 50 gözlem; yakınsamazsa None (İSTİSNA YOK).

    Dönüş: {'model','omega','alpha','gamma','beta','mean','loglik','persistence',
    'uncond_var','last_sigma2','last_resid','last_resid2','leverage','n'}.
    `uncond_var` None olabilir (persistence durağanlık kenarına oturduysa — bkz.
    `garch11_fit` IGARCH notu). `leverage` = gamma > 0."""
    arr = _clean_1d(returns, min_len=50)
    if arr is None:
        return None
    mean = float(np.mean(arr))
    resid = arr - mean
    var0 = float(np.var(resid))
    if var0 <= 0.0 or not math.isfinite(var0):
        return None
    resid_scaled = resid / math.sqrt(var0)
    pmax = _GARCH_PERSIST_MAX
    bounds = [(1e-12, None), (0.0, pmax), (-pmax, pmax), (0.0, pmax)]
    constraints = [
        {"type": "ineq", "fun": lambda x: pmax - (x[1] + 0.5 * x[2] + x[3])},
        {"type": "ineq", "fun": lambda x: x[1] + x[2]},
    ]
    starts = ((0.03, 0.06, 0.88), (0.05, 0.10, 0.80), (0.02, 0.02, 0.94),
              (0.08, 0.00, 0.85), (0.10, 0.15, 0.70))
    best = None
    for a0, g0, b0 in starts:
        x0 = np.array([max(1.0 - a0 - 0.5 * g0 - b0, 1e-10), a0, g0, b0])
        try:
            result = minimize(
                _gjr_neg_loglik, x0, args=(resid_scaled, 1.0), method="SLSQP",
                bounds=bounds, constraints=constraints, options={"maxiter": 300, "ftol": 1e-12},
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if not result.success:
            continue
        omega, alpha, gamma, beta = (float(v) for v in result.x)
        if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + gamma < -1e-12:
            continue
        if alpha + 0.5 * gamma + beta >= 1.0:
            continue
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        return None
    omega_scaled, alpha, gamma, beta = (float(v) for v in best.x)
    omega = omega_scaled * var0
    sigma2 = _gjr_sigma2_path(omega, alpha, gamma, beta, resid, var0)
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
        return None
    loglik = float(-best.fun) - 0.5 * arr.size * math.log(var0)
    persistence = alpha + 0.5 * gamma + beta
    at_boundary = persistence >= _GARCH_PERSIST_MAX - 1e-9
    return {
        "model": "gjr", "omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
        "mean": mean, "loglik": loglik, "persistence": float(persistence),
        "uncond_var": None if at_boundary else float(omega / (1.0 - persistence)),
        "last_sigma2": float(sigma2[-1]), "last_resid": float(resid[-1]),
        "last_resid2": float(resid[-1] * resid[-1]), "leverage": bool(gamma > 0.0),
        "n": int(arr.size),
    }


def _egarch_logs2_path(omega: float, alpha: float, gamma: float, beta: float,
                       resid: np.ndarray, logvar0: float) -> np.ndarray:
    """EGARCH(1,1) log-varyans yolu:
    ln sigma2_t = omega + alpha·(|z_{t-1}| − E|z|) + gamma·z_{t-1} + beta·ln sigma2_{t-1},
    z_t = eps_t / sigma_t.

    GARCH/GJR'nin aksine `lfilter`'a İNDİRGENEMEZ: z_{t-1} bir önceki sigma'ya bölünür,
    yani girdi terimi durumun kendisine bağlıdır (doğrusal DEĞİL). Saf Python döngüsü
    bilinçlidir. Taşmayı önlemek için log-varyans [−50, 50] aralığına kırpılır — bu
    aralığın dışı zaten sayısal olarak anlamsızdır (sigma² < 2e−22 veya > 5e21)."""
    n = int(resid.size)
    out = np.empty(n, dtype=float)
    out[0] = logvar0
    prev = logvar0
    eps = resid.tolist()
    for t in range(1, n):
        e = eps[t - 1]
        z = e / math.exp(0.5 * prev)
        prev = omega + alpha * (abs(z) - _EABS_Z) + gamma * z + beta * prev
        if prev > 50.0:
            prev = 50.0
        elif prev < -50.0:
            prev = -50.0
        out[t] = prev
    return out


def _egarch_neg_loglik(params: np.ndarray, resid: np.ndarray, logvar0: float) -> float:
    """EGARCH negatif log-olabilirlik (Gauss yenilik)."""
    omega, alpha, gamma, beta = (float(v) for v in params)
    logs2 = _egarch_logs2_path(omega, alpha, gamma, beta, resid, logvar0)
    if not np.all(np.isfinite(logs2)):
        return 1e10
    sigma2 = np.exp(logs2)
    nll = 0.5 * float(np.sum(logs2 + math.log(2.0 * np.pi) + (resid * resid) / sigma2))
    return nll if math.isfinite(nll) else 1e10


def egarch_fit(returns: ArrayLike) -> dict | None:
    """EGARCH(1,1) — Nelson (1991) ÜSTEL asimetrik oynaklık:

        ln sigma2_t = omega + alpha·(|z_{t-1}| − √(2/π)) + gamma·z_{t-1} + beta·ln sigma2_{t-1}

    GJR eşik-tabanlıdır (şok negatif Mİ?), EGARCH ise SÜREKLİdir: `gamma` standartlaştırılmış
    şokun İŞARETİNE, `alpha` BÜYÜKLÜĞÜNE tepki verir. **gamma < 0 kaldıraç etkisidir**
    (GJR'nin tersi işaret konvansiyonu — negatif z log-varyansı artırır). Log ölçekte
    çalıştığı için pozitiflik kısıtı YOKTUR: omega/alpha/gamma serbesttir, tek durağanlık
    şartı |beta| < 0,999'dur (log-varyans bir AR(1) sürecidir).

    `uncond_var` DEĞİL `uncond_var_median` döndürülür: E[ln sigma²] = omega/(1−beta) tanımlıdır
    ama exp'i E[sigma²] değil MEDYAN (geometrik) seviyedir; log-normal düzeltmesi yenilik
    dağılımının tüm momentlerini gerektirir. Sayıyı doğru adla basmak, yanlış adla basmaktan
    yeğdir (ölçüm-yokluğu disiplini).

    MLE birim varyanslı artıklarla koşulur; EGARCH log-ölçekte omega → omega + (1−beta)·ln(var0)
    ile geri ölçeklenir (ln sigma² doğrusal kaydırma altında eşdeğişkendir). En az 50 gözlem;
    yakınsamazsa None.

    Dönüş: {'model','omega','alpha','gamma','beta','mean','loglik','persistence' (=beta),
    'uncond_log_var','uncond_var_median','last_sigma2','last_z','leverage','n'}."""
    arr = _clean_1d(returns, min_len=50)
    if arr is None:
        return None
    mean = float(np.mean(arr))
    resid = arr - mean
    var0 = float(np.var(resid))
    if var0 <= 0.0 or not math.isfinite(var0):
        return None
    resid_scaled = resid / math.sqrt(var0)
    bounds = [(-5.0, 5.0), (-2.0, 2.0), (-2.0, 2.0), (-_EGARCH_BETA_MAX, _EGARCH_BETA_MAX)]
    starts = ((0.10, -0.05, 0.90), (0.15, -0.10, 0.85), (0.05, 0.00, 0.95))
    best = None
    for a0, g0, b0 in starts:
        x0 = np.array([0.0, a0, g0, b0])
        try:
            result = minimize(
                _egarch_neg_loglik, x0, args=(resid_scaled, 0.0), method="L-BFGS-B",
                bounds=bounds, options={"maxiter": 400, "ftol": 1e-12},
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if not result.success:
            continue
        if abs(float(result.x[3])) >= _EGARCH_BETA_MAX - 1e-9:
            continue
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        return None
    omega_s, alpha, gamma, beta = (float(v) for v in best.x)
    omega = omega_s + (1.0 - beta) * math.log(var0)      # log-ölçek geri kaydırma
    logs2 = _egarch_logs2_path(omega, alpha, gamma, beta, resid, math.log(var0))
    if not np.all(np.isfinite(logs2)):
        return None
    # Jacobian düzeltmesi: y = c·x ise log L_Y = log L_X − n·ln(c); c = sqrt(var0).
    loglik = float(-best.fun) - 0.5 * arr.size * math.log(var0)
    sigma_last = math.exp(0.5 * float(logs2[-1]))
    uncond_log = omega / (1.0 - beta)
    return {
        "model": "egarch", "omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
        "mean": mean, "loglik": loglik, "persistence": float(beta),
        "uncond_log_var": float(uncond_log),
        "uncond_var_median": float(math.exp(uncond_log)) if abs(uncond_log) < 500 else None,
        "last_sigma2": float(math.exp(float(logs2[-1]))),
        "last_z": float(resid[-1] / sigma_last) if sigma_last > 0 else None,
        "leverage": bool(gamma < 0.0), "n": int(arr.size),
    }


def asym_garch_forecast(fit: dict, horizon: int = 1) -> float | None:
    """`gjr_garch_fit` / `egarch_fit` sözlüğünden h-adım ileri VOLATİLİTE (sigma; modülün
    per-bar konvansiyonu — `garch11_forecast` ile aynı).

    GJR: bir adım İLERİ tam olarak bilinir (son şokun İŞARETİ gözlenmiştir):
    sigma²_{t+1} = omega + (alpha + gamma·1[eps_t<0])·eps_t² + beta·sigma²_t. h ≥ 2'de
    işaret bilinmediğinden beklenen değer alınır (E[1[eps<0]] = ½) ve yineleme
    `persistence` = alpha + gamma/2 + beta ile koşulsuz varyansa yakınsar.

    EGARCH: bir adım ileri ln sigma²_{t+1} de tam bilinir (z_t gözlenmiştir). h ≥ 2'de
    E[|z|−E|z|] = 0 ve E[z] = 0 olduğundan beklenen log-varyans omega + beta·ln sigma²
    yinelemesiyle ilerler. Dönen sigma bu YOLUN exp'idir = MEDYAN öngörü (E[sigma] değil);
    `egarch_fit` docstring'indeki aynı dürüstlük notu geçerlidir.

    Geçersiz sözlük / horizon < 1 / sonlu olmayan sonuç → None."""
    if not isinstance(fit, dict):
        return None
    try:
        h = int(horizon)
    except (TypeError, ValueError):
        return None
    if h < 1:
        return None
    model = fit.get("model")
    try:
        if model == "gjr":
            omega = float(fit["omega"])
            alpha = float(fit["alpha"])
            gamma = float(fit["gamma"])
            beta = float(fit["beta"])
            s2 = float(fit["last_sigma2"])
            e2 = float(fit["last_resid2"])
            neg = 1.0 if float(fit["last_resid"]) < 0.0 else 0.0
            s2 = omega + (alpha + gamma * neg) * e2 + beta * s2
            persist = alpha + 0.5 * gamma + beta
            for _ in range(h - 1):
                s2 = omega + persist * s2
            out = math.sqrt(s2) if s2 > 0 else None
        elif model == "egarch":
            omega = float(fit["omega"])
            alpha = float(fit["alpha"])
            gamma = float(fit["gamma"])
            beta = float(fit["beta"])
            z = fit.get("last_z")
            ls2 = math.log(float(fit["last_sigma2"]))
            if z is None:
                return None
            ls2 = omega + alpha * (abs(float(z)) - _EABS_Z) + gamma * float(z) + beta * ls2
            for _ in range(h - 1):
                ls2 = omega + beta * ls2
            out = math.exp(0.5 * ls2) if abs(ls2) < 500 else None
        else:
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return out if (out is not None and math.isfinite(out)) else None


# ---------------------------------------------------------------------------
# Rastgele matris teorisi — gürültü temizleme (Tur-72 — yol haritası §10.3)
# ---------------------------------------------------------------------------
def mp_bounds(n_assets: int, n_obs: int, sigma2: float = 1.0) -> tuple[float, float] | None:
    """Marčenko-Pastur (1967) özdeğer DESTEĞİ: q = n_obs/n_assets > 1 için

        lambda_± = sigma²·(1 ± √(1/q))²

    SAF GÜRÜLTÜDEN oluşan bir örneklem korelasyon matrisinin özdeğerleri (n,T→∞ limitinde)
    bu aralığa düşer. lambda_+ üstündeki özdeğerler gürültüyle AÇIKLANAMAZ = bilgi taşır.
    q ≤ 1 (gözlemden çok varlık) durumunda matris tekildir ve MP desteği bu biçimde
    tanımlı değildir → None. Geçersiz/sonlu olmayan girdi → None."""
    try:
        n = int(n_assets)
        t = int(n_obs)
        s2 = float(sigma2)
    except (TypeError, ValueError):
        return None
    if n < 1 or t < 1 or s2 <= 0.0 or not math.isfinite(s2):
        return None
    q = t / n
    if q <= 1.0:
        return None
    root = math.sqrt(1.0 / q)
    return s2 * (1.0 - root) ** 2, s2 * (1.0 + root) ** 2


def greek_neutral_hedge(a: ArrayLike, current: ArrayLike,
                        lower: ArrayLike | float | None = None,
                        upper: ArrayLike | float | None = None) -> dict | None:
    """min ||w||², A w = -G, lower <= w <= upper; salt okunur sürekli-adet hesabı.

    A: seçilen Greekler x hedge araçları; her sütun BİR PAY/KONTRATIN maruziyeti.
    G aynı birimlerde toplam mevcut maruziyettir. Çarpanları çağıran uygular.
    Bağımlı eşitlikler SVD ile indirgenir; tutarsızlık/olanaksız sınır/yakınsamama None.
    Amaç para maliyeti değildir. Kesirli kontrat sayıları yuvarlanmaz/icra edilmez.
    """
    from nihai.ai.advanced_math import qp_solve

    try:
        mat, g = np.asarray(a, dtype=float), np.asarray(current, dtype=float)
        if (mat.ndim != 2 or not 1 <= mat.shape[0] <= 3 or not 1 <= mat.shape[1] <= 256
                or g.shape != (mat.shape[0],) or not np.isfinite(mat).all() or not np.isfinite(g).all()):
            return None
        n = mat.shape[1]
        lo = np.broadcast_to(np.asarray(-np.inf if lower is None else lower, dtype=float), (n,))
        hi = np.broadcast_to(np.asarray(np.inf if upper is None else upper, dtype=float), (n,))
        if np.isnan(lo).any() or np.isnan(hi).any() or (lo > hi).any() or np.isposinf(lo).any() or np.isneginf(hi).any():
            return None
        scale = np.maximum(np.maximum(np.abs(mat).max(axis=1), np.abs(g)), 1e-12)
        scaled, rhs = mat / scale[:, None], -g / scale
        u, s, vh = np.linalg.svd(scaled, full_matrices=False)
        rank = int(np.count_nonzero(s > (s[0] * 1e-12 if s.size else 0)))
        projected = u[:, :rank].T @ rhs
        if np.max(np.abs(rhs - u[:, :rank] @ projected)) > 1e-8:
            return None
        w = qp_solve(2 * np.eye(n), np.zeros(n),
                     a_eq=vh[:rank] if rank else None,
                     b_eq=projected / s[:rank] if rank else None, lb=lo, ub=hi)
        if w is None:
            return None
        after = g + mat @ w
        if ((w < lo - 1e-8).any() or (w > hi + 1e-8).any()
                or not np.isfinite(after).all() or not np.isfinite(w @ w)
                or (np.abs(after) > 1e-7 * (1 + np.abs(g))).any()):
            return None
        return {"weights": w, "before": g.copy(), "after": after, "rank": rank,
                "objective": float(w @ w), "max_residual": float(np.max(np.abs(after))),
                "continuous": True}
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return None


def denoise_corr_mp(returns: ArrayLike) -> dict | None:
    """Korelasyon matrisini Marčenko-Pastur eşiğiyle GÜRÜLTÜDEN ARINDIRIR (RMT filtresi).

    Örneklem korelasyon matrisi, T gözlem ve N varlıkta N(N−1)/2 parametre tahmin eder;
    T/N küçüldükçe özdeğerlerin çoğu SADECE örneklem gürültüsüdür. Portföy optimizasyonu
    kovaryansı TERSLER ve tersin en küçük özdeğerlere duyarlılığı en yüksektir — yani
    optimizasyon tam olarak matrisin en gürültülü yönlerine yaslanır. Filtre:

      1) korelasyon matrisini özayrıştır,
      2) lambda_+ (bkz. `mp_bounds`) altındaki özdeğerleri GÜRÜLTÜ ilan et,
      3) hepsini ORTALAMALARIYLA değiştir (sabit-artık yöntemi — iz korunur, bilgi
         taşıyan özvektörlere DOKUNULMAZ),
      4) yeniden kur ve köşegeni 1'e ölçekle.

    `ledoit_wolf_cov` ile aynı sorunu hedefler ama farklı yolla: Ledoit-Wolf tüm matrisi
    hedefe doğru BÜZER (shrinkage), RMT yalnız gürültü ALT-UZAYINI düzler ve sinyal
    özvektörlerini olduğu gibi bırakır. İkisi rakip değil, farklı önsellerdir.

    Girdi (n_obs, n_asset) getiri paneli; n_obs > n_asset ŞARTTIR (aksi halde MP desteği
    tanımsız → None). Sıfır-varyans sütun / kısa / NaN → None.

    Dönüş: {'corr','corr_denoised','cov_denoised','eigenvalues','eigenvalues_denoised',
    'lambda_plus','lambda_minus','n_signal','n_noise','q','cond_before','cond_after','n',
    'n_assets'} — özdeğerler AZALAN sırada."""
    arr = _clean_2d(returns, min_rows=10, min_cols=2)
    if arr is None:
        return None
    n_obs, n_assets = arr.shape
    if n_obs <= n_assets:
        return None
    sd = arr.std(axis=0, ddof=1)
    if np.any(sd <= 0.0) or not np.all(np.isfinite(sd)):
        return None
    corr = np.corrcoef(arr, rowvar=False)
    if not np.all(np.isfinite(corr)):
        return None
    try:
        vals, vecs = np.linalg.eigh(corr)
    except np.linalg.LinAlgError:
        return None
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    bounds = mp_bounds(n_assets, n_obs)
    if bounds is None:
        return None
    lam_minus, lam_plus = bounds
    noise_mask = vals <= lam_plus
    new_vals = vals.copy()
    n_noise = int(np.count_nonzero(noise_mask))
    if n_noise > 0:
        new_vals[noise_mask] = float(np.mean(vals[noise_mask]))
    denoised = vecs @ np.diag(new_vals) @ vecs.T
    diag = np.sqrt(np.clip(np.diag(denoised), 1e-300, None))
    denoised = denoised / np.outer(diag, diag)
    np.fill_diagonal(denoised, 1.0)
    if not np.all(np.isfinite(denoised)):
        return None
    cov_denoised = denoised * np.outer(sd, sd)
    eig_after = np.linalg.eigvalsh(denoised)[::-1]

    def _cond(ev: np.ndarray) -> float | None:
        lo = float(np.min(ev))
        hi = float(np.max(ev))
        if lo <= 1e-12 or not math.isfinite(hi):
            return None
        return hi / lo

    return {
        "corr": np.asarray(corr), "corr_denoised": np.asarray(denoised),
        "cov_denoised": np.asarray(cov_denoised),
        "eigenvalues": np.asarray(vals), "eigenvalues_denoised": np.asarray(eig_after),
        "lambda_plus": float(lam_plus), "lambda_minus": float(lam_minus),
        "n_signal": int(n_assets - n_noise), "n_noise": n_noise,
        "q": float(n_obs / n_assets), "cond_before": _cond(vals), "cond_after": _cond(eig_after),
        "n": int(n_obs), "n_assets": int(n_assets),
    }


def mst_edges(corr: ArrayLike) -> dict | None:
    """Korelasyon matrisinden MİNİMUM YAYILAN AĞAÇ (Mantegna 1999 piyasa ağı).

    Metrik: d_ij = √(2·(1 − rho_ij)) — bu bir GERÇEK uzaklıktır (üçgen eşitsizliğini
    sağlar; ham 1−rho sağlamaz). MST, n varlığı n−1 kenarla bağlayan en kısa ağaçtır:
    piyasanın "iskeleti". Yüksek DERECELİ düğüm, korelasyon yapısının merkezindedir
    (şok o düğümden yayılır); yaprak düğümler çeşitlendirme adaylarıdır.

    Girdi kare, simetrik, köşegeni ~1 olan korelasyon matrisi (n ≥ 3). |rho| > 1 veya
    NaN/kısa girdi → None.

    Dönüş: {'edges': [{'i','j','dist','corr'}...] (uzaklığa göre artan), 'degree': list[int],
    'total_length': float, 'n': int}."""
    from scipy.sparse.csgraph import minimum_spanning_tree

    mat = _clean_2d(corr, min_rows=3, min_cols=3)
    if mat is None or mat.shape[0] != mat.shape[1]:
        return None
    n = int(mat.shape[0])
    if np.max(np.abs(mat)) > 1.0 + 1e-8:
        return None
    sym = 0.5 * (mat + mat.T)
    dist = np.sqrt(np.clip(2.0 * (1.0 - sym), 0.0, None))
    # csgraph 0'ı "kenar YOK" sayar; mükemmel korele çift (d=0) ağacı KOPARIRDI.
    # EŞİK ÖLÇÜLDÜ: scipy'nin MST'si 1e-9 ağırlıklı kenarı da SESSİZCE düşürür
    # (1e-12 ve 1e-9'da kenar ağaca girmiyor, 1e-6'da giriyor) — yani "sıfırdan
    # büyük olsun" yetmez. 1e-6 gerçek uzaklıklara (d ∈ [0, 2]) göre ihmal
    # edilebilir ama csgraph'ın eşiğinin üstündedir.
    dist[dist <= 0.0] = 1e-6
    np.fill_diagonal(dist, 0.0)
    try:
        tree = minimum_spanning_tree(dist).toarray()
    except (ValueError, np.linalg.LinAlgError):
        return None
    rows, cols = np.nonzero(tree)
    if rows.size == 0:
        return None
    degree = [0] * n
    edges = []
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        edges.append({"i": int(i), "j": int(j), "dist": float(tree[i, j]),
                      "corr": float(sym[i, j])})
        degree[int(i)] += 1
        degree[int(j)] += 1
    edges.sort(key=lambda e: e["dist"])
    return {"edges": edges, "degree": degree,
            "total_length": float(sum(e["dist"] for e in edges)), "n": n}


# ---------------------------------------------------------------------------
# Kümülatif Beklenti Teorisi (Tur-72 — yol haritası §9.1)
# ---------------------------------------------------------------------------
def cpt_value(x: ArrayLike, alpha: float = 0.88, beta: float = 0.88,
              lam: float = 2.25) -> np.ndarray | None:
    """Tversky-Kahneman (1992) DEĞER fonksiyonu (referans noktası 0):

        v(x) = x^alpha            (x ≥ 0)
        v(x) = −lam·(−x)^beta     (x < 0)

    `lam` ZARARDAN KAÇINMA katsayısıdır (makalede 2,25): aynı büyüklükteki kayıp,
    kazancın 2,25 katı kadar hissedilir. alpha,beta < 1 azalan duyarlılıktır
    (100→200 farkı, 0→100 farkından küçük hissedilir).

    Bu bir NORMATİF öneri değil, insan karar veren için AÇIKLAYICI bir modeldir;
    Kelly'nin log-fayda tepe noktasıyla yan yana okunduğunda "matematiksel optimum"
    ile "operatörün katlanabileceği" arasındaki farkı sayısallaştırır.

    Geçersiz parametre (alpha/beta ≤ 0 veya > 1, lam ≤ 0) / NaN girdi → None."""
    arr = _clean_1d(x, min_len=1)
    if arr is None:
        return None
    try:
        a = float(alpha)
        b = float(beta)
        lm = float(lam)
    except (TypeError, ValueError):
        return None
    if not (0.0 < a <= 1.0 and 0.0 < b <= 1.0) or lm <= 0.0 or not math.isfinite(lm):
        return None
    pos = np.where(arr >= 0.0, arr, 0.0)
    neg = np.where(arr < 0.0, -arr, 0.0)
    out = np.power(pos, a) - lm * np.power(neg, b)
    return np.asarray(out, dtype=float) if np.all(np.isfinite(out)) else None


def prelec_weight(p: ArrayLike, delta: float = 0.65, gamma: float = 1.0) -> np.ndarray | None:
    """Prelec (1998) OLASILIK AĞIRLIKLANDIRMA fonksiyonu:

        w(p) = exp(−gamma·(−ln p)^delta),   w(0)=0, w(1)=1

    delta < 1'de eğri ters-S biçimlidir: KÜÇÜK olasılıklar AŞIRI büyütülür (piyango
    ve kuyruk-sigortası talebinin kaynağı), orta-yüksek olasılıklar küçümsenir.
    delta = 1, gamma = 1'de w(p) = p (ağırlıklandırma yok — rasyonel sınır).

    p aralık dışı (<0 veya >1) / delta ≤ 0 / gamma ≤ 0 → None."""
    arr = _clean_1d(p, min_len=1)
    if arr is None:
        return None
    try:
        d = float(delta)
        g = float(gamma)
    except (TypeError, ValueError):
        return None
    if d <= 0.0 or g <= 0.0 or not math.isfinite(d) or not math.isfinite(g):
        return None
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        return None
    out = np.zeros_like(arr)
    inner = (arr > 0.0) & (arr < 1.0)
    if np.any(inner):
        out[inner] = np.exp(-g * np.power(-np.log(arr[inner]), d))
    out[arr >= 1.0] = 1.0
    return np.asarray(out, dtype=float) if np.all(np.isfinite(out)) else None


def cpt_prospect(outcomes: ArrayLike, probs: ArrayLike, alpha: float = 0.88,
                 beta: float = 0.88, lam: float = 2.25, delta_gain: float = 0.61,
                 delta_loss: float = 0.69) -> dict | None:
    """KÜMÜLATİF Beklenti Teorisi ile bir kumarın öznel değeri.

    "Kümülatif" olan kısım şudur: ağırlık tek tek olasılıklara DEĞİL, sıralanmış
    sonuçların KUYRUK olasılıklarına uygulanır (Quiggin'in rank-bağımlı yaklaşımı;
    aksi halde model birinci-derece stokastik baskınlığı ihlal eder). Kazanç tarafı
    en İYİ sonuçtan, kayıp tarafı en KÖTÜ sonuçtan başlayarak yığılır:

        pi_i^+ = w^+(p_i+…+p_n) − w^+(p_{i+1}+…+p_n)
        pi_i^- = w^-(p_1+…+p_i) − w^-(p_1+…+p_{i-1})
        CPT   = sum_i pi_i · v(x_i)

    Kesinlik eşdeğeri v^{-1}(CPT) ile geri çevrilir. `premium` = EV − CE: pozitifse
    operatör bu kumarı beklenen değerinden UCUZA satardı (risk/zarardan kaçınma primi).

    Uzunluklar eşit olmalı, olasılıklar ≥ 0 ve toplamları ~1 (1e-6 tolerans).
    Aksi halde / geçersiz parametre → None."""
    xs = _clean_1d(outcomes, min_len=1)
    ps = _clean_1d(probs, min_len=1)
    if xs is None or ps is None or xs.size != ps.size:
        return None
    if np.any(ps < 0.0) or abs(float(np.sum(ps)) - 1.0) > 1e-6:
        return None
    vals = cpt_value(xs, alpha=alpha, beta=beta, lam=lam)
    if vals is None:
        return None
    order = np.argsort(xs)
    xs_s = xs[order]
    ps_s = ps[order]
    vals_s = vals[order]
    n = xs_s.size
    weights = np.zeros(n, dtype=float)
    # Kayıp bacağı: en kötüden yukarı doğru kümülatif.
    loss_idx = np.nonzero(xs_s < 0.0)[0]
    if loss_idx.size:
        cum = np.cumsum(ps_s[: loss_idx[-1] + 1])
        w_cum = prelec_weight(np.clip(cum, 0.0, 1.0), delta=delta_loss)
        if w_cum is None:
            return None
        prev = 0.0
        for idx in range(loss_idx[-1] + 1):
            weights[idx] = float(w_cum[idx]) - prev
            prev = float(w_cum[idx])
    # Kazanç bacağı: en iyiden aşağı doğru kümülatif.
    gain_idx = np.nonzero(xs_s >= 0.0)[0]
    if gain_idx.size:
        start = int(gain_idx[0])
        tail = ps_s[start:][::-1]
        cum = np.cumsum(tail)
        w_cum = prelec_weight(np.clip(cum, 0.0, 1.0), delta=delta_gain)
        if w_cum is None:
            return None
        prev = 0.0
        for pos in range(tail.size):
            weights[n - 1 - pos] = float(w_cum[pos]) - prev
            prev = float(w_cum[pos])
    cpt = float(np.sum(weights * vals_s))
    try:
        a = float(alpha)
        b = float(beta)
        lm = float(lam)
        ce = cpt ** (1.0 / a) if cpt >= 0.0 else -((-cpt / lm) ** (1.0 / b))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    ev = float(np.sum(ps * xs))
    if not all(math.isfinite(v) for v in (cpt, ce, ev)):
        return None
    return {"cpt_value": cpt, "certainty_equivalent": float(ce), "ev": ev,
            "premium": float(ev - ce), "weights": np.asarray(weights),
            "outcomes_sorted": np.asarray(xs_s), "probs_sorted": np.asarray(ps_s),
            "loss_aversion": float(lam), "n": int(n)}


# ---------------------------------------------------------------------------
# Sualtı süreleri ve kuyruk performansı (Tur-73)
# ---------------------------------------------------------------------------
def time_underwater(equity: ArrayLike) -> dict | None:
    """Pozitif sermayede tepe_t = max(equity[:t+1]), sualtı_t = equity_t < tepe_t.

    Kesintisiz sualtı dizilerinin en uzun/ortalama süresini, son dizinin süresini ve
    sualtı bar sayısı / n oranını verir. Tamamlanmamış son dizi de ortalamaya katılır;
    zirveye eşitlik diziyi bitirir. Hiç sualtı yoksa süreler ve oran sıfırdır.
    n < 2, equity <= 0, NaN/Inf veya dönüştürülemeyen girdi → None.
    """
    try:
        arr = _clean_1d(equity, min_len=2)
    except (OverflowError, FloatingPointError):
        return None
    if arr is None or np.any(arr <= 0.0):
        return None
    underwater = arr < np.maximum.accumulate(arr)
    lengths: list[int] = []
    current = 0
    for submerged in underwater:
        if submerged:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return {
        "max_bar": max(lengths, default=0),
        "ortalama_bar": float(np.mean(lengths)) if lengths else 0.0,
        "mevcut_bar": current,
        "sualti_orani": float(np.count_nonzero(underwater) / arr.size),
    }


def gain_to_pain(returns: ArrayLike) -> float | None:
    """Gain-to-Pain = sum(r) / sum(|r[r < 0]|).

    profit_factor kazanç/kayıp oranıdır; burada pay NET toplam getiridir.
    n < 1, negatif getiri yokluğu (payda=0), NaN/Inf, dönüştürülemeyen girdi veya
    sonlu olmayan hesap sonucu → None.
    """
    try:
        arr = _clean_1d(returns, min_len=1)
        if arr is None:
            return None
        with np.errstate(all="ignore"):
            pain = float(-np.sum(arr[arr < 0.0]))
            if pain == 0.0 or not math.isfinite(pain):
                return None
            result = float(np.sum(arr) / pain)
        return result if math.isfinite(result) else None
    except (OverflowError, FloatingPointError):
        return None


def tail_ratio(returns: ArrayLike, q: float = 0.95) -> float | None:
    """Kuyruk oranı = |quantile(r,q)| / |quantile(r,1-q)|.

    NumPy varsayılan kuantil yöntemi, lineer interpolasyon kullanılır; 0.5 <= q <= 1.
    n < 2, payda=0, geçersiz q, NaN/Inf, dönüştürülemeyen girdi veya sonlu olmayan
    hesap sonucu → None.
    """
    try:
        arr = _clean_1d(returns, min_len=2)
        quantile = float(q)
        if arr is None or not 0.5 <= quantile <= 1.0:
            return None
        with np.errstate(all="ignore"):
            lower, upper = np.abs(np.quantile(arr, [1.0 - quantile, quantile]))
            if lower == 0.0 or not np.all(np.isfinite([lower, upper])):
                return None
            result = float(upper / lower)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return None


# ---------------------------------------------------------------------------
# Histogram entropileri — ortak binleme, doğal logaritma (Tur-73)
# ---------------------------------------------------------------------------
def _hist_probs(x: ArrayLike, bins: int = 20) -> np.ndarray | None:
    """Eşit genişlikli np.histogram bölmelerinden p_i = adet_i/n; boş bölmeler atılır.

    bins tam sayı değilse, bins < 2, n < bins, aralık=0, NaN/Inf, dönüştürülemeyen
    girdi veya sayısal olarak kurulamayan histogram → None.
    """
    try:
        arr = _clean_1d(x, min_len=2)
        count = float(bins)
        if arr is None or not math.isfinite(count) or not count.is_integer():
            return None
        if count < 2 or count > arr.size or np.max(arr) == np.min(arr):
            return None
        with np.errstate(all="ignore"):
            counts, _ = np.histogram(arr, bins=int(count))
            return np.asarray(counts[counts > 0] / arr.size, dtype=float)
    except (TypeError, ValueError, OverflowError, FloatingPointError, IndexError):
        return None


def shannon_entropy(x: ArrayLike, bins: int = 20) -> float | None:
    """Histogram Shannon entropisi: H = -sum(p_i * ln(p_i)), birim nat (doğal log).

    p_i=0 bölmeleri atlanır. bins tam sayı değilse, bins < 2, n < bins, tüm değerler
    eşitse, NaN/Inf, dönüştürülemeyen girdi veya kurulamayan histogram → None.
    """
    probs = _hist_probs(x, bins=bins)
    if probs is None:
        return None
    with np.errstate(all="ignore"):
        result = float(-np.sum(probs * np.log(probs)))
    return result if math.isfinite(result) else None


def renyi_entropy(x: ArrayLike, alpha: float = 2.0, bins: int = 20) -> float | None:
    """Rényi entropisi: H_a = ln(sum(p_i**a)) / (1-a), a > 0; doğal logaritma.

    |a-1| < 1e-6 ise matematiksel Shannon limiti shannon_entropy ile hesaplanır.
    Aynı histogram ve dolu bölmeler kullanılır. a <= 0, geçersiz bins (tam sayı >= 2
    olmalı), n < bins, sabit seri, NaN/Inf, dönüştürülemeyen girdi veya sonlu olmayan
    hesap sonucu → None.
    """
    try:
        a = float(alpha)
        if not math.isfinite(a) or a <= 0.0:
            return None
        if abs(a - 1.0) < 1e-6:
            return shannon_entropy(x, bins=bins)
        probs = _hist_probs(x, bins=bins)
        if probs is None:
            return None
        with np.errstate(all="ignore"):
            log_sum = float(np.logaddexp.reduce(a * np.log(probs)))
            result = log_sum / (1.0 - a)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return None


def tsallis_entropy(x: ArrayLike, q: float = 2.0, bins: int = 20) -> float | None:
    """Tsallis entropisi: S_q = (1-sum(p_i**q)) / (q-1).

    |q-1| < 1e-6 ise matematiksel Shannon limiti (nat) shannon_entropy ile hesaplanır.
    Sonlu gerçek q için ortak histogramın yalnız dolu bölmeleri kullanılır (q=0 için
    dolu bölme sayısı - 1). Geçersiz bins (tam sayı >= 2 olmalı), n < bins, sabit seri,
    NaN/Inf, dönüştürülemeyen girdi veya sonlu olmayan hesap sonucu → None.
    """
    try:
        exponent = float(q)
        if not math.isfinite(exponent):
            return None
        if abs(exponent - 1.0) < 1e-6:
            return shannon_entropy(x, bins=bins)
        probs = _hist_probs(x, bins=bins)
        if probs is None:
            return None
        with np.errstate(all="ignore"):
            log_sum = float(np.logaddexp.reduce(exponent * np.log(probs)))
            result = float(-np.expm1(log_sum) / (exponent - 1.0))
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return None


# ---------------------------------------------------------------------------
# Kritik yavaşlama ve çok-fraktallı ölçekleme (Tur-73)
# ---------------------------------------------------------------------------
def critical_slowing_down(x: ArrayLike, window: int = 50,
                          step: int | None = None) -> dict | None:
    """Kayan pencerede AR(1)=corr(x[:-1],x[1:]) ve örneklem varyansı (ddof=1).

    Her ikisinin pencere zamanına göre trendi Kendall tau ile ölçülür; devrilme,
    İKİ tau da > 0 ve İKİ p de < 0.05 olduğunda True'dur (iki taraflı p değerleri).
    Sabit pencere veya sabit lag dilimi AR(1)'i tanımsız kılar; aynı pencere iki
    trendden de atılır.

    `step` = pencereler arası KAYDIRMA; verilmezse **window** (ÖRTÜŞMEYEN pencere).
    Bu varsayılan bir zevk meselesi değil, ÖLÇÜLMÜŞ bir kusurun düzeltmesidir (Tur-73):
    Kendall p-değeri gözlemlerin BAĞIMSIZ olduğunu varsayar, oysa birer bar kaydırılan
    pencereler neredeyse aynı veriyi paylaşır → p felaket derecede anti-muhafazakâr olur.
    DURAĞAN AR(1)=0,2 gürültüsünde 200 tekrarla ölçüldü (n=1200, window=100):
        step=1  (örtüşen)     → yanlış-pozitif %19,5   ← nominal %5'in ~4 katı
        step=25 (örtüşen)     → %5,0
        step=100 (örtüşmeyen) → %0,0
    Gerçek bir devrilmede (AR1 0,05→0,95 + artan varyans) yakalama gücü ÜÇ ayarda da
    %100 — yani örtüşmeyi bırakmanın GÜÇ MALİYETİ YOKTUR. `step < window` bilinçle
    seçilebilir (eğriyi yoğun çizmek için) ama o zaman p-değeri yalnız betimleyicidir;
    dönüşteki `ortusen` bayrağı bunu bildirir.

    Dönüşe `adim` ve `ortusen` eklenir. window tam sayı değilse veya < 10, step < 1, n < 2*window,
    geçerli pencere < 5, tanımsız Kendall trendi, NaN/Inf veya dönüştürülemeyen girdi
    → None.
    UYARI: Scheffer/Dakos yaklaşımı bir KENAR/sinyal iddiası DEĞİL, betimleyici bir
    istatistiktir.
    """
    try:
        arr = _clean_1d(x, min_len=20)
        width = float(window)
        if arr is None or not math.isfinite(width) or not width.is_integer() or width < 10:
            return None
        w = int(width)
        adim = w if step is None else int(step)
        if adim < 1 or arr.size < 2 * w:
            return None
        times: list[int] = []
        ar1: list[float] = []
        variances: list[float] = []
        with np.errstate(all="ignore"):
            for end in range(w, arr.size + 1, adim):
                segment = arr[end - w : end]
                variance = float(np.var(segment, ddof=1))
                if variance <= 0.0 or not math.isfinite(variance):
                    continue
                if np.std(segment[:-1]) == 0.0 or np.std(segment[1:]) == 0.0:
                    continue
                correlation = float(np.corrcoef(segment[:-1], segment[1:])[0, 1])
                if math.isfinite(correlation):
                    times.append(end)
                    ar1.append(correlation)
                    variances.append(variance)
            if len(times) < 5:
                return None
            ar_tau, ar_p = (float(v) for v in stats.kendalltau(times, ar1))
            var_tau, var_p = (float(v) for v in stats.kendalltau(times, variances))
        if not all(math.isfinite(v) for v in (ar_tau, ar_p, var_tau, var_p)):
            return None
        return {
            "ar1_tau": ar_tau,
            "var_tau": var_tau,
            "ar1_p": ar_p,
            "var_p": var_p,
            "n_pencere": len(times),
            "adim": adim,
            "ortusen": adim < w,
            "devrilme": ar_tau > 0.0 and ar_p < 0.05 and var_tau > 0.0 and var_p < 0.05,
        }
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return None


def mfdfa(
    x: ArrayLike,
    q_list: ArrayLike | None = None,
    scales: ArrayLike | None = None,
    order: int = 1,
) -> dict | None:
    """Kantelhardt (2002) çok-fraktallı DFA: Y = cumsum(x-mean(x)).

    Her s ölçeğinde ileri ve geri N_s=floor(n/s) pencere (toplam 2*N_s) kullanılır.
    order dereceli polinom çıkarılır; F2=mean(kalıntı**2). F_q=(mean(F2**(q/2)))**(1/q),
    q=0 için F_0=exp(0.5*mean(ln(F2))). log(F_q) ~ log(s) eğimi h(q)'dur.
    tau=q*h-1, alpha=gradient(tau,q), f_alpha=q*alpha-tau; genislik=max(alpha)-min(alpha).
    h2=h(2) ayrıca hesaplanır; özel q_list içinde 2 bulunması gerekmez. q değerleri
    sıralanır ve tekilleştirilir; türev için en az iki farklı q gerekir.

    Varsayılan q arange(-5,5.5,0.5); otomatik ölçekler 16..n//4 arasında log aralıklı
    tamsayılardır ve en az 6 farklı ölçek ister. Özel ölçekler de tamsayı olmalıdır;
    order+1 < s <= n dışındakiler atılır. F2 veya F_q <= 0 / sonlu değilse ölçek atılır.
    n < 100, negatif/kesirli order, geçerli ölçek < 4, sabit seri, NaN/Inf,
    dönüştürülemeyen girdi veya başarısız/sonlu olmayan regresyon → None.
    """
    try:
        arr = _clean_1d(x, min_len=100)
        degree_value = float(order)
        if arr is None or not math.isfinite(degree_value) or not degree_value.is_integer():
            return None
        if degree_value < 0:
            return None
        degree = int(degree_value)
        n = int(arr.size)
        qs = np.arange(-5.0, 5.5, 0.5) if q_list is None else _clean_1d(q_list, min_len=2)
        if qs is None:
            return None
        qs = np.unique(qs)
        if qs.size < 2:
            return None
        if scales is None:
            if n // 4 < 16:
                return None
            sizes = np.unique(np.geomspace(16, n // 4, num=20).astype(int))
            if sizes.size < 6:
                return None
        else:
            supplied = _clean_1d(scales, min_len=4)
            if supplied is None or np.any(supplied != np.floor(supplied)):
                return None
            sizes = np.unique(supplied[(supplied > degree + 1) & (supplied <= n)]).astype(int)
        sizes = sizes[sizes > degree + 1]
        if sizes.size < 4:
            return None
        q_fit = np.union1d(qs, [2.0])
        valid_sizes: list[int] = []
        log_fluctuations: list[np.ndarray] = []
        with np.errstate(all="ignore"):
            magnitude = float(np.max(np.abs(arr)))
            if magnitude == 0.0:
                return None
            # Sabit çarpan h ve Legendre spektrumunu değiştirmez; profil taşmasını önler.
            scaled = arr / magnitude
            profile = np.cumsum(scaled - np.mean(scaled))
            for size in sizes:
                s = int(size)
                count = n // s
                forward = profile[: count * s].reshape(count, s)
                backward = profile[n - count * s :].reshape(count, s)
                segments = np.vstack((forward, backward)).T
                design = np.polynomial.polynomial.polyvander(np.linspace(-1.0, 1.0, s), degree)
                coefficients, _, rank, _ = np.linalg.lstsq(design, segments, rcond=None)
                if rank < degree + 1:
                    continue
                residuals = segments - design @ coefficients
                f2 = np.mean(residuals * residuals, axis=0)
                if np.any(f2 <= 0.0) or not np.all(np.isfinite(f2)):
                    continue
                logs = np.log(f2)
                log_f = np.empty(q_fit.size, dtype=float)
                for j, q in enumerate(q_fit):
                    if q == 0.0:
                        log_f[j] = 0.5 * np.mean(logs)
                    else:
                        log_f[j] = (np.logaddexp.reduce(0.5 * q * logs) - np.log(f2.size)) / q
                fluctuations = np.exp(log_f)
                if np.any(fluctuations <= 0.0) or not np.all(np.isfinite(fluctuations)):
                    continue
                valid_sizes.append(s)
                log_fluctuations.append(log_f)
            if len(valid_sizes) < 4:
                return None
            regression = np.column_stack((np.ones(len(valid_sizes)), np.log(valid_sizes)))
            fitted, _, _, _ = np.linalg.lstsq(regression, np.asarray(log_fluctuations), rcond=None)
            h = fitted[1, np.searchsorted(q_fit, qs)]
            h2 = float(fitted[1, np.searchsorted(q_fit, 2.0)])
            tau = qs * h - 1.0
            alpha = np.gradient(tau, qs)
            spectrum = qs * alpha - tau
            width = float(np.max(alpha) - np.min(alpha))
        if not np.all(np.isfinite([h, tau, alpha, spectrum])) or not math.isfinite(width + h2):
            return None
        return {
            "q": qs.tolist(),
            "h": h.tolist(),
            "tau": tau.tolist(),
            "alpha": alpha.tolist(),
            "f_alpha": spectrum.tolist(),
            "genislik": width,
            "h2": h2,
        }
    except (TypeError, ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
        return None


# ---------------------------------------------------------------------------
# Getiri eğrisi ve faktörlerden arındırılmış momentum (Tur-73)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1024)
def _nss_design(ms_bytes: bytes, tau1: float, tau2: float) -> np.ndarray:
    """NSS tasarım matrisi — (vadeler, tau1, tau2) ÖNBELLEKLİ, salt-okunur.

    Tur-81 (BULGU-81-5): `nss_fit` tau ızgarasında 16×16−16 = **240** tasarım matrisi
    kurar ve matris YALNIZ vadelere + tau çiftine bağlıdır; `y` ona hiç girmez. Pano
    `/api/quant/curve` aynı vade vektörüyle son 250 günü AYRI AYRI uyduruyordu →
    aynı 240 matris **250 kez** yeniden kuruluyordu (60.000 kuruluş, 240 yeterliyken).
    Anahtar vade dizisinin BAYTLARIdır (tam eşitlik; kayan nokta yuvarlaması yok), bu
    yüzden `lstsq`ın girdisi birebir aynı kalır ve sonuç BİT-DENKTİR.

    Dönen dizi PAYLAŞILANDIR ve `writeable=False` ile kilitlenmiştir: çağıran yalnız okur
    (`lstsq` ve `design @ betas` girdilerini değiştirmez); ileride biri yazmayı denerse
    önbelleği sessizce zehirlemek yerine hata alır."""
    ms = np.frombuffer(ms_bytes, dtype=np.float64)
    with np.errstate(all="ignore"):
        x1, x2 = ms / tau1, ms / tau2
        exp1, exp2 = np.exp(-x1), np.exp(-x2)
        load1 = np.divide(-np.expm1(-x1), x1, out=np.ones_like(ms), where=x1 != 0.0)
        load2 = np.divide(-np.expm1(-x2), x2, out=np.ones_like(ms), where=x2 != 0.0)
        design = np.column_stack((np.ones(ms.size), load1, load1 - exp1, load2 - exp2))
    design.flags.writeable = False
    return design


def nss_fit(
    maturities: ArrayLike,
    yields: ArrayLike,
    tau1: float | None = None,
    tau2: float | None = None,
) -> dict | None:
    """Nelson-Siegel-Svensson: x1=m/tau1, x2=m/tau2, L(x)=(1-exp(-x))/x.

    y(m)=b0+b1*L(x1)+b2*(L(x1)-exp(-x1))+b3*(L(x2)-exp(-x2)); m pozitif YIL'dır.
    Sabit tau'larda beta'lar doğrusal en küçük karelerle çözülür. Verilmeyen her tau
    0.1..30 yıl arasındaki, 2.0 ve 5.0'ı içeren sabit ızgarada aranır; tau1 != tau2.
    En küçük SSR kazanır; rmse=sqrt(SSR/n), fit=girdi vadelerindeki uydurulan değerler.
    seviye=b0, egim=-b1 (uzun eksi kısa vade işaret konvansiyonu), egrilik=b2.
    Izgara dışındaki gerçek tau'lar için tam geri kazanım iddiası yoktur.

    EKONOMİK KABUL EDİLEBİLİRLİK (Tur-73, ÖLÇÜLMÜŞ KUSUR ÜZERİNE): NSS aşırı
    parametrelenmiştir; salt-SSR araması gözlenen eğriyi mükemmel uydururken b0'ı
    (uzun-vade asimptotu) anlamsız bir yere koyabilir. CANLI ABD Hazine eğrisinde
    ölçüldü: son 500 işlem gününün **%98,0'ında** salt-SSR kazananı b0'ı gözlenen getiri
    aralığının, **%81,0'ında** ise aşağıdaki TOLERANSLI bandın bile DIŞINA atıyordu (2026-09-03: rmse %0,029 ile **b0 = −2,08**, 30Y %5,25
    iken). Panoda "seviye" diye gösterilecek sayı buydu. Bu yüzden arama İKİ aday tutar:
    SSR-en-iyi ve b0'ı [min(y), max(y)] aralığına DÜŞEN SSR-en-iyi. Kabul edilebilir aday
    varsa BİRİNCİL sonuç odur; `dejenere` bayrağı salt-SSR adayının reddedildiğini,
    `beta0_kisitsiz`/`rmse_kisitsiz` onun ne söyleyeceğini bildirir. Hiç kabul edilebilir
    aday yoksa salt-SSR adayı `dejenere=True` ile döner (sayı GİZLENMEZ, İŞARETLENİR).

    n < 4, uzunluk uyuşmazlığı, m <= 0, tau <= 0/eşit tau'lar, NaN/Inf,
    dönüştürülemeyen girdi, tam ranklı tasarım bulunamaması veya sayısal taşma → None.
    """
    try:
        ms = _clean_1d(maturities, min_len=4)
        ys = _clean_1d(yields, min_len=4)
        if ms is None or ys is None or ms.size != ys.size or np.any(ms <= 0.0):
            return None
        grid = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)
        first = grid if tau1 is None else (float(tau1),)
        second = grid if tau2 is None else (float(tau2),)
        if any(t <= 0.0 or not math.isfinite(t) for t in (*first, *second)):
            return None
        best: dict | None = None
        best_ssr = math.inf
        kabul: dict | None = None            # b0'ı gözlenen getiri aralığına düşen en iyi
        kabul_ssr = math.inf
        # Kabul bandı gözlenen aralığın TAM kendisi DEĞİLDİR: b0 bir ASİMPTOTTUR ve
        # eğri ona yaklaşır, ulaşmaz — gerçek b0 aralığın biraz dışında olabilir (sentetik
        # (4.5, −1.2, 2.0, −0.8) eğrisinde max(y)=4,42 iken gerçek b0=4,50). Bant bu yüzden
        # aralığın genişliğinin yarısı kadar (en az 0,5 puan) taşırılır; canlı eğrideki
        # dejenere uyum (b0=−2,08, bant [3,12; 5,96]) bu toleransla da REDDEDİLİR.
        y_lo, y_hi = float(np.min(ys)), float(np.max(ys))
        pay = max(0.5, 0.5 * (y_hi - y_lo))
        y_lo, y_hi = y_lo - pay, y_hi + pay
        ms_key = ms.astype(np.float64, copy=False).tobytes()
        with np.errstate(all="ignore"):
            for t1 in first:
                for t2 in second:
                    if t1 == t2:
                        continue
                    design = _nss_design(ms_key, t1, t2)
                    try:
                        betas, _, rank, _ = np.linalg.lstsq(design, ys, rcond=None)
                    except np.linalg.LinAlgError:
                        continue
                    if rank < 4 or not np.all(np.isfinite(betas)):
                        continue
                    fitted = design @ betas
                    residuals = ys - fitted
                    ssr = float(np.sum(residuals * residuals))
                    if not math.isfinite(ssr):
                        continue
                    aday = {
                        "beta0": float(betas[0]),
                        "beta1": float(betas[1]),
                        "beta2": float(betas[2]),
                        "beta3": float(betas[3]),
                        "tau1": t1,
                        "tau2": t2,
                        "rmse": math.sqrt(ssr / ms.size),
                        "seviye": float(betas[0]),
                        "egim": float(-betas[1]),
                        "egrilik": float(betas[2]),
                        "fit": fitted.tolist(),
                    }
                    if ssr < best_ssr:
                        best_ssr, best = ssr, aday
                    if ssr < kabul_ssr and y_lo <= aday["beta0"] <= y_hi:
                        kabul_ssr, kabul = ssr, aday
        if best is None:
            return None
        secilen = kabul if kabul is not None else best
        secilen["dejenere"] = kabul is None
        secilen["beta0_kisitsiz"] = best["beta0"]
        secilen["rmse_kisitsiz"] = best["rmse"]
        return secilen
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return None


def residual_momentum(
    asset_returns: ArrayLike,
    factor_returns: ArrayLike,
    lookback: int = 252,
    skip: int = 21,
) -> dict | None:
    """Son lookback gözlemde sabit terimli OLS: r_asset = alpha + F*beta + e.

    F tek faktör için 1B, birden çok faktör için n*k'dır. Son skip kalıntı atılır
    (skip=0 ise tümü); kalan m gözlemde resid_mom=sum(e), resid_vol=std(e,ddof=1),
    resid_mom_z=sum(e)/(std(e)*sqrt(m)). r2=1-SSR/TSS; n=OLS gözlem sayısı (lookback).
    Bu toplam kalıntı momentumu 12-1 konvansiyonudur; bileşik getiri değildir.
    Uzunluk uyuşmazlığı, kesirli lookback/skip, skip < 0, lookback <= skip+10,
    n < lookback, NaN/Inf, dönüştürülemeyen girdi, sayısal olarak sıfır resid_vol
    veya başarısız/sonlu olmayan OLS sonucu → None.
    """
    try:
        arr = _clean_1d(asset_returns, min_len=11)
        raw_factors = np.asarray(factor_returns, dtype=float)
        if raw_factors.ndim == 1:
            raw_factors = raw_factors[:, None]
        factors = _clean_2d(raw_factors)
        lb_value, skip_value = float(lookback), float(skip)
        if not all(math.isfinite(v) and v.is_integer() for v in (lb_value, skip_value)):
            return None
        lb, omitted = int(lb_value), int(skip_value)
        if arr is None or factors is None or arr.size != factors.shape[0]:
            return None
        if omitted < 0 or lb <= omitted + 10 or arr.size < lb:
            return None
        with np.errstate(all="ignore"):
            ys = arr[-lb:]
            design = np.column_stack((np.ones(lb), factors[-lb:]))
            coefficients, _, _, _ = np.linalg.lstsq(design, ys, rcond=None)
            fitted = design @ coefficients
            residuals = ys - fitted
            momentum = residuals[:-omitted] if omitted else residuals
            vol = float(np.std(momentum, ddof=1))
            scale = float(np.max(np.abs(ys)))
            tolerance = np.finfo(float).eps * max(design.shape) * scale
            if not math.isfinite(vol) or vol <= tolerance:
                return None
            total = float(np.sum(momentum))
            z = total / (vol * math.sqrt(momentum.size))
            centered = ys - np.mean(ys)
            tss = float(np.sum(centered * centered))
            if tss <= 0.0 or not math.isfinite(tss):
                return None
            r2 = 1.0 - float(np.sum(residuals * residuals)) / tss
        if not np.all(np.isfinite(coefficients)) or not all(math.isfinite(v) for v in (total, z, r2)):
            return None
        return {
            "resid_mom": total,
            "resid_vol": vol,
            "resid_mom_z": z,
            "alpha": float(coefficients[0]),
            "beta": coefficients[1:].tolist(),
            "r2": r2,
            "n": lb,
        }
    except (TypeError, ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
        return None


# ---------------------------------------------------------------------------
# Günlük kaldıraç ve bileşik getiri erimesi (Tur-73)
# ---------------------------------------------------------------------------
def leveraged_etf_drag(
    index_returns: ArrayLike,
    beta: float = 3.0,
    expense: float = 0.0095,
    ppy: float = 252,
) -> dict | None:
    """Günlük kaldıraçlı ETF: r_lev=beta*r_idx-expense/ppy (expense yıllık gider).

    bilesik_lev=prod(1+r_lev)-1, bilesik_idx=prod(1+r_idx)-1,
    naif=beta*bilesik_idx, suruklenme=bilesik_lev-naif (negatif değer erime).

    İKİ SÜRÜKLENME AYNI ŞEY DEĞİLDİR ve karıştırmak sayıyı 14 kat yanlış okutur (Tur-73'te
    ölçüldü). `suruklenme` ARİTMETİKTİR (bileşik farkı); Cheng-Madhavan yaklaşımı ise
    LOG uzayındadır. Karşılaştırılacak büyüklük `log_suruklenme` = ln(1+bilesik_lev) −
    beta*ln(1+bilesik_idx)'tir ve `teorik_drag` TAM ONU yaklaşıklar (±%5, expense=0'da):
      ±%5 salınan 252 günlük seride log_suruklenme −1,9216 ↔ teorik −1,8900 (%1,7 fark),
      düz +%0,1'lik trendde −0,00077 ↔ −0,000756 (%1,4 fark).
    AYNI seride `suruklenme` sırasıyla −0,1317 ve **+0,2680**'dir: **aritmetik sürüklenme
    TRENDDE POZİTİF OLABİLİR** — kaldıraçlı bileşik, tek yönlü bir seride naif katı GEÇER.
    "Vol drag her zaman negatiftir" cümlesi ARİTMETİK ölçüde YANLIŞTIR; negatif olan
    log ölçüsüdür ve orada da yalnız sum(r²) sürüklenmeyi bastırdığı ölçüde.

    Cheng-Madhavan yaklaşımı teorik_drag=-0.5*beta*(beta-1)*sum(r_idx**2);
    yillik_vol=std(r_idx,ddof=1)*sqrt(ppy). Yaklaşım gideri içermez, tam bileşik
    farkına eşit olmak zorunda değildir. Herhangi bir 1+r_lev <= 0 günü ETF'yi
    kalıcı sıfırlar: bilesik_lev=-1.0, sifirlandi=True; sonraki getiriler canlandırmaz.
    n < 2, beta <= 0, ppy <= 0, NaN/Inf, dönüştürülemeyen girdi veya sonlu olmayan
    hesap sonucu → None.
    """
    try:
        arr = _clean_1d(index_returns, min_len=2)
        leverage, fee, periods = float(beta), float(expense), float(ppy)
        if arr is None or not all(math.isfinite(v) for v in (leverage, fee, periods)):
            return None
        if leverage <= 0.0 or periods <= 0.0:
            return None
        with np.errstate(all="ignore"):
            leveraged_returns = leverage * arr - fee / periods
            growth = 1.0 + leveraged_returns
            if not np.all(np.isfinite(growth)):
                return None
            wiped_out = bool(np.any(growth <= 0.0))
            compounded_lev = -1.0 if wiped_out else float(np.prod(growth) - 1.0)
            compounded_idx = float(np.prod(1.0 + arr) - 1.0)
            naive = leverage * compounded_idx
            drag = compounded_lev - naive
            # log sürüklenme: teorik_drag'in YAKLAŞIKLADIĞI büyüklük. Sıfırlanmış ETF'de
            # ln(0) tanımsızdır → None (uydurma sayı basılmaz).
            if compounded_lev > -1.0 and compounded_idx > -1.0:
                log_drag: float | None = float(
                    math.log1p(compounded_lev) - leverage * math.log1p(compounded_idx))
                if log_drag is not None and not math.isfinite(log_drag):
                    log_drag = None
            else:
                log_drag = None
            theoretical = -0.5 * leverage * (leverage - 1.0) * float(np.sum(arr * arr))
            volatility = float(np.std(arr, ddof=1)) * math.sqrt(periods)
        if not all(math.isfinite(v) for v in (compounded_lev, compounded_idx, naive, drag,
                                            theoretical, volatility)):
            return None
        return {
            "bilesik_lev": compounded_lev,
            "bilesik_idx": compounded_idx,
            "naif": naive,
            "suruklenme": drag,
            "log_suruklenme": log_drag,
            "teorik_drag": theoretical,
            "yillik_vol": volatility,
            "n": int(arr.size),
            "sifirlandi": wiped_out,
        }
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return None


# ---------------------------------------------------------------------------
# 21) TUR-75 — ASİMETRİK OYNAKLIK, ENTROPİK RİSK ve RİSK KATKISI
#
# Dış geliştirme listesinin BÖLÜM 6/10'undan bu modülde karşılığı OLMAYAN beş madde.
# Hepsi KARAR-DIŞI ölçüdür; hiçbiri sinyale/eşiğe bağlanmaz (ACI DERS #3/#5).
# ---------------------------------------------------------------------------
def realized_semivariance(returns: ArrayLike) -> dict | None:
    """Aşağı/yukarı gerçekleşen yarı-varyans (Barndorff-Nielsen 2010 / Madde 108).

        RS⁻ = Σ_{r<0} r²      ·      RS⁺ = Σ_{r>0} r²      ·      RS⁻ + RS⁺ = RV

    Toplam varyans "kötü" ile "iyi" hareketi aynı sayar; yarı-varyans ayırır.
    `asymmetry` = (RS⁻ − RS⁺)/RV ∈ [−1, 1]: pozitif = düşüş oynaklığı baskın.

    Sıfır getiriler hiçbir tarafa yazılmaz (r² = 0 zaten katkısızdır) ama SAYILARI
    ayrı taşınır — `n_zero` düz/halt barlarını görünür kılar.
    """
    r = _clean_1d(returns, 2)
    if r is None:
        return None
    neg = r[r < 0.0]
    poz = r[r > 0.0]
    rs_n = float(np.sum(neg * neg))
    rs_p = float(np.sum(poz * poz))
    rv = rs_n + rs_p
    return {
        "rv": rv,
        "rs_minus": rs_n,
        "rs_plus": rs_p,
        "asymmetry": ((rs_n - rs_p) / rv) if rv > 0 else None,
        "n_neg": int(neg.size), "n_pos": int(poz.size),
        "n_zero": int(r.size - neg.size - poz.size),
        "n": int(r.size),
    }


def _garch_m_neg_loglik(params: np.ndarray, r: np.ndarray, var0: float) -> float:
    mu, delta, omega, alpha, beta = (float(v) for v in params)
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 0.999:
        return 1e18
    n = r.size
    s2 = np.empty(n)
    s2[0] = var0
    ll = 0.0
    for t in range(n):
        if t > 0:
            e = r[t - 1] - (mu + delta * s2[t - 1])
            s2[t] = omega + alpha * e * e + beta * s2[t - 1]
            if not (s2[t] > 0.0) or not math.isfinite(s2[t]):
                return 1e18
        e_t = float(r[t] - (mu + delta * s2[t]))
        # KAREYİ ALMADAN ÖNCE sınırla: |e| > 1e150'de e² taşar ve numpy sessiz bir
        # `inf` + RuntimeWarning üretir. Arama sırasında kısıt-dışı bölgeye
        # düşmek meşrudur; oradan BÜYÜK CEZAYLA dönülür (`_garch11_neg_loglik` deseni).
        if not math.isfinite(e_t) or abs(e_t) > 1e150:
            return 1e18
        ll += math.log(2.0 * math.pi * s2[t]) + e_t * e_t / s2[t]
        if not math.isfinite(ll):
            return 1e18
    # SABİT TERİM (ln 2π) DAHİLDİR — `_garch11_neg_loglik` ile AYNI ölçekte olsun
    # diye. Atlanırsa iki modelin log-olabilirlikleri n·ln(2π)/2 kadar ayrışır ve
    # LR istatistiği (2·Δloglik) tamamen anlamsız çıkar (ölçüldü: LR 3675,8).
    return 0.5 * float(ll)


def garch_m_fit(returns: ArrayLike) -> dict | None:
    """GARCH-in-Mean (Engle-Lilien-Robins 1987 / Madde 107).

        r_t = μ + δ·σ²_t + ε_t   ·   σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1}

    `delta` risk-getiri ödünleşiminin doğrudan tahminidir: δ > 0 ise oynaklık
    ARTTIKÇA beklenen getiri artıyordur (risk primi).

    Tur-49'un ÖLÇEK NORMALİZASYONU burada da uygulanır — getiriler birim varyansa
    çekilir, parametreler geri ölçeklenir. Atlanırsa optimizatör ω yönünde ANLAMSIZ
    bir adım büyüklüğüyle arar ve köşeye çakılır (o kusur `garch11_fit`te 296 gerçek
    sembolde ölçülmüştü).

    **δ'nın ANLAMLILIĞI BURADA TEST EDİLMEZ.** Pozitif bir δ tek başına "oynaklık
    yükselince al" kuralını doğrulamaz; o bir kenar iddiasıdır ve `edge_lab` kanıt
    kapısından geçmelidir (ACI DERS #3/#5). `loglik` ile `loglik_garch11` birlikte
    döner → farkın iki katı, δ=0 kısıtı için bir χ²(1) olabilirlik-oranı istatistiğidir.
    """
    r0 = _clean_1d(returns, 100)
    if r0 is None:
        return None
    olcek = float(np.std(r0, ddof=1))
    if not math.isfinite(olcek) or olcek <= 0.0:
        return None
    r = r0 / olcek
    var0 = float(np.var(r, ddof=1))
    baz = garch11_fit(r)
    if baz is None:
        return None
    sinir = [(-10.0, 10.0), (-50.0, 50.0), (1e-8, 10.0), (0.0, 0.999), (0.0, 0.999)]
    r_ort = float(np.mean(r))
    # ÇOKLU BAŞLANGIÇ ZORUNLUDUR ve nedeni ölçülmüştür: μ ile δ·σ² neredeyse
    # eşdoğrusaldır (σ²'nin ortalaması sabit gibi davranır) → δ=0'dan başlayan tek
    # bir arama SIRTA takılır. Bilinen bir DGP'de (δ=30) tek başlangıç δ̂=0,00 ve
    # nll 4170,14 verirken çoklu başlangıç δ̂=29,13 ve nll 4162,84 buluyor.
    # Her başlangıçta μ, ima edilen ortalama ÖRNEKLEM ortalamasında kalacak şekilde
    # seçilir (μ = r̄ − δ·E[σ²]) — sırtın kendisi böylece kaldırılır.
    m_s2 = float(baz.get("uncond_var") or var0) or 1.0
    en_iyi = None
    for d0 in (0.0, 0.25 / m_s2, -0.25 / m_s2, 1.0 / m_s2, -1.0 / m_s2):
        x0 = np.array([r_ort - d0 * m_s2, d0, float(baz["omega"]),
                       float(baz["alpha"]), float(baz["beta"])])
        try:
            res = minimize(_garch_m_neg_loglik, x0, args=(r, var0), method="L-BFGS-B",
                           bounds=sinir)
        except (ValueError, FloatingPointError):    # pragma: no cover - fail-soft
            continue
        if not np.all(np.isfinite(res.x)) or res.fun >= 1e17:
            continue
        if en_iyi is None or res.fun < en_iyi[0]:
            en_iyi = (float(res.fun), np.asarray(res.x, dtype=float))
    if en_iyi is None:
        return None
    nll, cozum = en_iyi
    mu, delta, omega, alpha, beta = (float(v) for v in cozum)
    # Geri ölçekleme: r = olcek·r̃ ⇒ σ² = olcek²·σ̃², μ = olcek·μ̃, δ = δ̃/olcek.
    o2 = olcek * olcek
    baz_ll = baz.get("loglik")
    return {
        "mu": mu * olcek,
        "delta": delta / olcek,
        "omega": omega * o2,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        # Jacobian düzeltmesi: ölçeklenmiş seride hesaplanan log-olabilirlik ham
        # seriye çevrilirken −n·ln(olcek) alır (aksi hâlde iki model FARKLI
        # ölçeklerde kıyaslanır ve LR istatistiği anlamsızlaşır).
        "loglik": -nll - r.size * math.log(olcek),
        "loglik_garch11": (float(baz_ll) - r.size * math.log(olcek)
                           if baz_ll is not None else None),
        "n": int(r.size),
        "scale": olcek,
    }


def evar(returns: ArrayLike, alpha: float = 0.05) -> dict | None:
    """Entropik Riske Maruz Değer (Ahmadi-Javid 2012 / Madde 211).

        EVaR_α(L) = inf_{z>0} { z·ln( E[e^{L/z}] / α ) }        (L = KAYIP = −getiri)

    Chernoff sınırından türer ve **CVaR'ın ÜST SINIRIDIR**: EVaR ≥ CVaR ≥ VaR. Bu
    sıralama YAPISALDIR ve test onu arar — sıralamayı bozan bir uygulama sayıyı değil
    TANIMI yanlış hesaplıyordur.

    `alpha` bu paketin geri kalanıyla AYNI konvansiyondadır: KUYRUK OLASILIĞI
    (0,05 = en kötü %5). BULGU-73-1'in kapattığı çelişki tekrarlanmasın diye burada
    ayrıca yazılıdır.
    """
    r = _clean_1d(returns, 5)
    if r is None:
        return None
    try:
        a = float(alpha)
    except (TypeError, ValueError):
        return None
    if not (0.0 < a < 1.0):
        return None
    loss = -r
    olcek = float(np.max(np.abs(loss)))
    if olcek <= 0.0:      # tamamen düz seri: her risk ölçüsü tam olarak 0'dır
        return {"evar": 0.0, "cvar": 0.0, "var": 0.0, "z": None,
                "alpha": a, "n": int(r.size)}

    def _hedef(logz: float) -> float:
        z = math.exp(logz)
        # log-sum-exp ile taşma güvenli: ln E[e^{L/z}] = m + ln mean(e^{L/z − m})
        u = loss / z
        m = float(np.max(u))
        return float(z * (m + math.log(float(np.mean(np.exp(u - m)))) - math.log(a)))

    try:
        res = minimize_scalar(_hedef, bounds=(math.log(olcek * 1e-4),
                                              math.log(olcek * 1e4)), method="bounded")
    except (ValueError, FloatingPointError):        # pragma: no cover - fail-soft
        return None
    if not res.success or not math.isfinite(res.fun):
        return None
    return {
        "evar": float(res.fun),
        "cvar": cvar_historical(r, a),
        "var": var_historical(r, a),
        "z": float(math.exp(res.x)),
        "alpha": a,
        "n": int(r.size),
    }


def _quantile_regression(x: np.ndarray, y: np.ndarray,
                         tau: float) -> tuple[float, float] | None:
    """Pinball kaybını minimize eden (a, b): y ≈ a + b·x. Türevsiz (Powell)."""
    def _kayip(p: np.ndarray) -> float:
        e = y - (p[0] + p[1] * x)
        return float(np.sum(np.where(e >= 0, tau * e, (tau - 1.0) * e)))

    vx = float(np.var(x, ddof=1))
    b0 = float(np.cov(x, y, ddof=1)[0, 1] / vx) if vx > 0 else 0.0
    try:
        res = minimize(_kayip, np.array([float(np.quantile(y, tau)), b0]),
                       method="Powell")
    except (ValueError, FloatingPointError):        # pragma: no cover - fail-soft
        return None
    if not np.all(np.isfinite(res.x)):
        return None
    return float(res.x[0]), float(res.x[1])


def covar(asset_returns: ArrayLike, system_returns: ArrayLike,
          alpha: float = 0.05) -> dict | None:
    """CoVaR ve ΔCoVaR (Adrian-Brunnermeier 2016 / Madde 203).

    Kuantil regresyonuyla: SİSTEM getirisinin α-kuantili, VARLIK kendi α-kuantilinde
    iken. ΔCoVaR = CoVaR(varlık sıkıntıda) − CoVaR(varlık MEDYANDA) — yani varlığın
    sisteme kattığı EK kuyruk riski.

    **ΔCoVaR bir NEDENSELLİK ölçüsü DEĞİLDİR.** Ortak bir faktöre maruz iki varlık
    birbirine hiç bulaşmadan da yüksek ΔCoVaR verir; ölçülen şey EŞ-HAREKETTİR.
    `alpha` yine KUYRUK OLASILIĞIDIR.
    """
    a_r = _clean_1d(asset_returns, 30)
    s_r = _clean_1d(system_returns, 30)
    if a_r is None or s_r is None or a_r.size != s_r.size:
        return None
    try:
        al = float(alpha)
    except (TypeError, ValueError):
        return None
    if not (0.0 < al < 0.5):
        return None
    if float(np.var(a_r, ddof=1)) <= 0.0:
        return None
    fit = _quantile_regression(a_r, s_r, al)
    if fit is None:
        return None
    a0, b = fit
    var_asset = float(np.quantile(a_r, al))
    med_asset = float(np.quantile(a_r, 0.5))
    covar_d = a0 + b * var_asset
    covar_m = a0 + b * med_asset
    return {
        "covar": covar_d,
        "covar_median": covar_m,
        "delta_covar": covar_d - covar_m,
        "beta_tail": b,
        "var_asset": var_asset,
        "var_system": float(np.quantile(s_r, al)),
        "alpha": al, "n": int(a_r.size),
    }


def marginal_risk_contributions(weights: ArrayLike, cov: ArrayLike) -> dict | None:
    """Marjinal risk katkısı ve bileşen riski (Madde 202).

        MCR_i = (Σw)_i / σ_p     ·     CR_i = w_i · MCR_i     ·     Σ CR_i = σ_p

    Son eşitlik Euler'in homojen fonksiyon teoremidir, YAPISALDIR ve test onu makine
    hassasiyetinde arar. `share` bileşenin toplam riskteki PAYIDIR ve AĞIRLIK payıyla
    KARIŞTIRILMAMALIDIR: %10 ağırlıklı bir pozisyon riskin %40'ını üretebilir —
    listenin Madde 202'si tam olarak bunu görünür kılmayı istiyor.

    `component_var`: aynı ayrışmanın parametrik (normal) VaR karşılığı, `z₀,₉₅·CR_i`.
    """
    w = _clean_1d(weights, 1)
    S = _clean_2d(cov, 1, 1)
    if w is None or S is None or S.shape[0] != S.shape[1] or S.shape[0] != w.size:
        return None
    var_p = float(w @ S @ w)
    if not math.isfinite(var_p) or var_p <= 0.0:
        return None
    sigma = math.sqrt(var_p)
    mcr = (S @ w) / sigma
    cr = w * mcr
    return {
        "portfolio_vol": sigma,
        "mcr": mcr,
        "component_risk": cr,
        "share": cr / sigma,
        "component_var": _norm_ppf(0.95) * cr,
        "max_share_idx": int(np.argmax(np.abs(cr))),
        "n": int(w.size),
    }
