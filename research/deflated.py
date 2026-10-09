"""
deflated.py — Dürüst-ölçüm istatistikleri: Probabilistic & Deflated Sharpe + binom/Wilson.

Bu modül YALNIZ ölçüm/raporlama sağlar; hiçbir verdikt/karar/eşik mantığı içermez
(CLAUDE.md "kenar UYDURMADAN dürüst ölçüm" disiplini). Amaç: bir Sharpe/isabet sayısına
bakıp "bu, şanstan ne kadar ayırt edilebilir?" ve "N konfig denedim, en iyisi şansla ne
kadar iyi görünürdü?" sorularını dürüstçe yanıtlamak.

Formül kaynakları:
- Bailey, D. H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality." Journal of Portfolio Management,
  40(5), 94-107.  (PSR, E[max SR], DSR)
- López de Prado, M. (2018). "Advances in Financial Machine Learning", Ch. 8 (Backtest
  Statistics) — PSR/DSR/beklenen-maksimum Sharpe türetimi.
- Acklam, P. J. (2003). "An algorithm for computing the inverse normal cumulative
  distribution function." (Phi_inv rasyonel yaklaşımı, |hata| < 1.15e-9.)

Normal dağılım: scipy varsa scipy.stats.norm (venv'de mevcut); yoksa saf-stdlib fallback
(Phi = math.erf; Phi_inv = Acklam). BAĞIMLILIK EKLENMEZ — scipy zaten kuruludur; import-guard
yalnız taşınabilirlik/tekil-fonksiyon testi içindir. Fallback ve scipy yolları ~1e-9 örtüşür.
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

_SQRT2 = math.sqrt(2.0)
_EULER_GAMMA = 0.5772156649015329   # Euler-Mascheroni sabiti (γ)

try:
    from scipy.stats import norm as _scipy_norm
    _HAS_SCIPY = True
except ImportError:   # pragma: no cover - fallback fonksiyonları ayrıca doğrudan test edilir
    _scipy_norm = None
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Standart normal CDF (Phi) ve ters-CDF (Phi_inv) — stdlib fallback + scipy tercihi
# ---------------------------------------------------------------------------
def _phi_erf(x: float) -> float:
    """Standart normal CDF, math.erf tabanlı (stdlib, tam-hassas): Phi(x)=½·(1+erf(x/√2))."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


# Acklam (2003) rasyonel yaklaşımı katsayıları; mutlak hata |ε| < 1.15e-9.
# a,b: merkez bölge (0.02425 ≤ p ≤ 0.97575); c,d: alt/üst kuyruklar.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def _phi_inv_acklam(p: float) -> float:
    """Standart normal ters-CDF (kuantil), Acklam rasyonel yaklaşımı. |hata| < 1.15e-9.
    Sınır davranışı (scipy.ppf ile uyumlu): p ≤ 0 → -inf, p ≥ 1 → +inf."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    if p < _P_LOW:   # alt kuyruk
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if p > _P_HIGH:   # üst kuyruk (simetri)
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    q = p - 0.5   # merkez
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
           (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)


def phi(x: float) -> float:
    """Standart normal CDF Φ(x). scipy varsa scipy.stats.norm.cdf, yoksa erf fallback."""
    if _HAS_SCIPY:
        return float(_scipy_norm.cdf(x))
    return _phi_erf(x)


def phi_inv(p: float) -> float:
    """Standart normal ters-CDF Φ⁻¹(p) (kuantil). scipy varsa ppf, yoksa Acklam fallback."""
    if _HAS_SCIPY:
        return float(_scipy_norm.ppf(p))
    return _phi_inv_acklam(p)


# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio (PSR)
# ---------------------------------------------------------------------------
def psr(sharpe_hat: float, n: int, skew: float = 0.0, kurt: float = 3.0,
        sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2014; LdP 2018 Ch.8).

        PSR(SR*) = Φ( (ŜR − SR*)·√(n−1) / √(1 − γ₃·ŜR + (γ₄−1)/4·ŜR²) )

    ŜR = örneklem Sharpe'ı (BOYUTSUZ; SR* ile aynı periyodiklikte), n = gözlem sayısı,
    γ₃ = çarpıklık (skew), γ₄ = basıklık (kurt; normal dağılım = 3). Getirilerin
    non-normalliğini (fat-tail/asimetri) Sharpe'ın belirsizliğine katar → "ŜR gerçekten
    SR*'ı aşıyor mu?" olasılığı. SR*=0 varsayılan (Sharpe > 0 olasılığı).

    Payda ≤ 0 (aşırı negatif skew + yüksek SR birleşimi) veya n < 2 → NaN + logger.warning
    (sessiz çökme yok; dürüstçe 'ölçülemez' der)."""
    if n < 2:
        logger.warning("psr: n<2 (n=%s) → NaN (Sharpe belirsizliği tanımsız)", n)
        return float("nan")
    var_term = 1.0 - skew * sharpe_hat + (kurt - 1.0) / 4.0 * sharpe_hat * sharpe_hat
    if var_term <= 0.0:
        logger.warning("psr: payda ≤ 0 (=%.4g; skew=%.3g kurt=%.3g SR=%.3g) → NaN",
                       var_term, skew, kurt, sharpe_hat)
        return float("nan")
    z = (sharpe_hat - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(var_term)
    return phi(z)


# ---------------------------------------------------------------------------
# Beklenen maksimum Sharpe (çoklu-test deflasyon çekirdeği) ve DSR
# ---------------------------------------------------------------------------
def expected_max_sharpe(n_trials: int, var_trials: float, T: int) -> float:
    """N bağımsız denemenin (konfig) H0 altında beklenen EN İYİ Sharpe'ı (LdP 2018 Ch.8):

        E[max SR] ≈ √V · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ],  γ = Euler-Mascheroni.

    V = var_trials = denemeler-arası Sharpe TAHMİNLERİNİN varyansı, N = n_trials. Çoklu-test
    deflasyonunun kalbi: "çok konfig deneyince en iyisi ŞANSLA bu kadar iyi görünür". Değer
    N ile monoton artar. N ≤ 1 → 0.0 (tek deneme = deflasyon yok); V ≤ 0 → 0.0.

    T (gözlem/track uzunluğu) imza tamlığı içindir; DSR bunu aşağı-akışta PSR'nin n'i olarak
    kullanır — bu yaklaşımda E[max SR] doğrudan T'ye bağlı değildir."""
    if n_trials <= 1 or var_trials <= 0.0:
        return 0.0
    max_z = ((1.0 - _EULER_GAMMA) * phi_inv(1.0 - 1.0 / n_trials)
             + _EULER_GAMMA * phi_inv(1.0 - 1.0 / (n_trials * math.e)))
    return math.sqrt(var_trials) * max_z


def dsr(sharpe_hat: float, n: int, n_trials: int, var_trials: float,
        skew: float = 0.0, kurt: float = 3.0) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014): PSR ama benchmark SR*=0 yerine
    SR* = E[max SR] (N deneme içinden şansla beklenen en iyi Sharpe). Böylece "çok konfig
    denedim, en iyisini seçtim" seçilim yanlılığı düşülür.

    E[max SR] ≥ 0 olduğundan DSR ≤ PSR; N=1 → E[max]=0 → DSR == PSR (deflasyon yok)."""
    sr_star = expected_max_sharpe(n_trials, var_trials, n)
    return psr(sharpe_hat, n, skew=skew, kurt=kurt, sr_benchmark=sr_star)


def sharpe_estimator_var(sharpe_hat: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Örneklem Sharpe TAHMİNCİSİNİN varyansı (Lo 2002; PSR paydasının /(n−1)'i):

        Var[ŜR] ≈ (1 − γ₃·ŜR + (γ₄−1)/4·ŜR²) / (n−1).

    DSR'ye var_trials verilmediğinde makul H0 varsayılanı: bağımsız denemelerin her biri bu
    örnekleme varyansına sahip kabul edilir. n < 2 → NaN."""
    if n < 2:
        return float("nan")
    return (1.0 - skew * sharpe_hat + (kurt - 1.0) / 4.0 * sharpe_hat * sharpe_hat) / (n - 1)


# ---------------------------------------------------------------------------
# Getiri momentleri (postmortem konvansiyonuyla hizalı)
# ---------------------------------------------------------------------------
def sharpe_moments(returns: np.ndarray, ddof: int = 0) -> dict:
    """Getiri serisinden PSR/DSR için gereken momentler:
        sr   = ort/std  (BOYUTSUZ örneklem Sharpe),
        skew = Pearson çarpıklık (g₁, popülasyon momenti; normal = 0),
        kurt = Pearson basıklık  (g₂+3, popülasyon momenti; normal = 3),
        n    = gözlem sayısı.

    NOT (konvansiyon): Görev taslağı ddof=1 önerdi; ancak postmortem.compute_metrics'in
    raporladığı `sharpe_trade` = mean/pstdev (ddof=0). Tutarlılık için ddof VARSAYILAN 0
    (statistics.pstdev ile aynı tanım) — görevin "uyumsuzsa postmortem'inkine uy" yönergesi.
    std=0 veya n<2 → sr=0 ve nötr momentler (skew=0, kurt=3) döner (çökme yok). skew n<3,
    kurt n<4 iken nötr sabitlenir (yüksek moment tanımsız/aşırı gürültülü)."""
    r = np.asarray(returns, dtype=float).ravel()
    n = int(r.size)
    if n < 2:
        return {"sr": 0.0, "skew": 0.0, "kurt": 3.0, "n": n}
    mean = float(r.mean())
    std = float(r.std(ddof=ddof))
    dev = r - mean
    m2 = float((dev ** 2).mean())
    if std <= 0.0 or m2 <= 0.0:
        return {"sr": 0.0, "skew": 0.0, "kurt": 3.0, "n": n}
    sr = mean / std
    skew = float((dev ** 3).mean() / m2 ** 1.5) if n >= 3 else 0.0
    kurt = float((dev ** 4).mean() / m2 ** 2) if n >= 4 else 3.0
    return {"sr": sr, "skew": skew, "kurt": kurt, "n": n}


# ---------------------------------------------------------------------------
# Binom oran istatistikleri: Wilson %95 GA + binom p-değeri (yönsüz)
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Binom oran için Wilson skor güven aralığı (k başarı / n deneme). Normal-yaklaşım
    (Wald) aralığından küçük/orta n'de daha doğru; [0,1] dışına ASLA taşmaz. z=1.96 → %95.
    n ≤ 0 → (0.0, 1.0) (bilgi yok). Dönüş: (alt, üst).

    NOT: forward_harness._wilson_ci ile AYNI formül ama BİLİNÇLİ olarak ayrı bırakıldı —
    verdikt makinesine (forward_harness) bu turda dokunulmuyor; delegasyon ileride ayrı,
    parite-testli bir değişiklik olarak yapılabilir. Uç k=0/k=n'de float hatası sınırı
    epsilon kadar taşırabildiğinden [0,1]'e kelepçelenir (Wilson matematiksel olarak
    [0,1] içindedir; kelepçe yalnız ~1e-17'lik float artığını temizler)."""
    if n <= 0:
        return 0.0, 1.0
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def binom_p_value(k: int, n: int, p0: float = 0.5) -> float:
    """H0: p = p0 için İKİ-KUYRUK binom testi p-değeri (normal yaklaşım; n binlerce iken
    kesin binomla ~aynı):  z = (k − n·p0)/√(n·p0·(1−p0)),  p = 2·Φ(−|z|).
    n ≤ 0 veya varyans 0 → NaN.

    YÖNSÜZ yorum: küçük p YALNIZCA "gözlenen oran p0'dan anlamlı sapıyor" der; sapmanın
    YÖNÜ (üstünde mi altında mı) veya bir "kenar" ima ETMEZ — oran p0'ın ALTINDA da anlamlı
    çıkabilir. Kenar iddiası için ayrıca yön + OOS doğrulama gerekir (CLAUDE.md ACI DERSLER)."""
    if n <= 0:
        return float("nan")
    var = n * p0 * (1.0 - p0)
    if var <= 0.0:
        return float("nan")
    z = (k - n * p0) / math.sqrt(var)
    return 2.0 * phi(-abs(z))


# ---------------------------------------------------------------------------
# Sabit-blok-olasılıklı (stationary) bootstrap — RC/SPA'nın örnekleme çekirdeği
# ---------------------------------------------------------------------------
def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int, seed: int) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap indeks matrisi, şekil (n_boot, n).

    Her adımda `q = 1/mean_block` olasılıkla YENİ rastgele bir başlangıç seçilir, aksi
    hâlde bir öncekinin devamına (mod n, DAİRESEL) geçilir. Blok uzunlukları geometrik
    dağılır → ortalama `mean_block`. IID bootstrap'ın aksine seri korelasyonu KORUR;
    RC/SPA'nın günlük getiri gibi bağımlı serilerde geçerli olmasının nedeni budur.

    `mean_block <= 1` → q=1 (IID bootstrap; blok yok). n <= 0 / n_boot <= 0 → boş dizi.
    DETERMİNİSTİK: aynı seed → bit-bit aynı indeksler."""
    if n <= 0 or n_boot <= 0:
        return np.empty((0, 0), dtype=np.int64)
    q = 1.0 / max(1.0, float(mean_block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n))
    yeni_blok = rng.random((n_boot, n)) < q
    yeni_blok[:, 0] = True                      # ilk eleman her zaman rastgele başlangıç
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):                        # ponytail: n adımlık Python döngüsü;
        devam = (idx[:, t - 1] + 1) % n          # vektörleştirilemez (özyinelemeli bağımlılık)
        idx[:, t] = np.where(yeni_blok[:, t], starts[:, t], devam)
    return idx


def _perf_matrix(perf) -> np.ndarray | None:
    """(T, M) performans farkı matrisi: sonlu, T>=8, M>=1. Uymayan girdi → None."""
    try:
        arr = np.asarray(perf, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] < 8 or arr.shape[1] < 1:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def reality_check(perf, mean_block: float = 5.0, n_boot: int = 1000, seed: int = 0) -> dict | None:
    """White (2000) Reality Check: "M aday içinden EN İYİSİ, hepsi değersizken şansla bu
    kadar iyi görünür müydü?"

    `perf` = (T, M) matris; sütun k, k'nci adayın BENCHMARK'A GÖRE dönem başına performans
    FARKI d_{k,t} (pozitif = aday daha iyi). Tek aday için 1B dizi kabul edilir.

        İstatistik:  V = max_k √T · d̄_k
        Bootstrap :  V*_b = max_k √T · (d̄*_{k,b} − d̄_k)      ← H0 altına yeniden merkezleme
        p-değeri  :  P(V* >= V)

    Dönüş: {"stat", "p_value", "en_iyi": int, "en_iyi_ortalama", "n_boot", "T", "M"}.
    Yorum: p KÜÇÜK ise "en iyi aday, çoklu-test şansıyla açıklanamıyor". p BÜYÜK ise
    veri madenciliği yanlılığı dışlanamaz. Bu bir KENAR İDDİASI DEĞİL, bir eleme kapısıdır
    (DSR'nin analitik deflasyonunun bootstrap karşılığı; DSR normallik varsayar, RC varsaymaz).

    UYARI (Hansen 2005): RC, kötü adaylarla SULANDIRILABİLİR — değersiz aday eklemek
    p-değerini yükseltir. Bu yüzden `hansen_spa` tercih edilmelidir.
    Kısa/NaN/boş girdi veya n_boot <= 0 → None."""
    arr = _perf_matrix(perf)
    if arr is None or n_boot <= 0:
        return None
    t_n, m = arr.shape
    dbar = arr.mean(axis=0)
    sqrt_t = math.sqrt(t_n)
    stat = float(np.max(sqrt_t * dbar))
    idx = stationary_bootstrap_indices(t_n, mean_block, n_boot, seed)
    boot_mean = arr[idx].mean(axis=1)                       # (n_boot, M)
    v_star = np.max(sqrt_t * (boot_mean - dbar), axis=1)    # (n_boot,)
    p_val = float(np.mean(v_star >= stat))
    en_iyi = int(np.argmax(dbar))
    return {"stat": stat, "p_value": p_val, "en_iyi": en_iyi,
            "en_iyi_ortalama": float(dbar[en_iyi]), "n_boot": int(n_boot),
            "T": int(t_n), "M": int(m)}


def hansen_spa(perf, mean_block: float = 5.0, n_boot: int = 1000, seed: int = 0) -> dict | None:
    """Hansen (2005) Superior Predictive Ability testi — RC'nin sulandırmaya dayanıklı hâli.

    RC'den İKİ farkı vardır:
      1. STUDENTİZASYON: istatistik  T^SPA = max(0, max_k √T·d̄_k / ŵ_k)  — oynaklığı yüksek
         adaylar ham ortalamayla öne çıkamaz. ŵ_k, √T·d̄*_k'nin bootstrap std'sidir.
      2. YENİDEN MERKEZLEME EŞİĞİ: açıkça kötü adaylar (d̄_k çok negatif) H0'a çekilmez,
         OLDUKLARI yerde bırakılır → değersiz aday eklemek p'yi ARTIRMAZ. Hansen üç
         merkezleme verir; üçü de döner:
           spa_u (upper, gevşek)  : g_k = d̄_k                    → RC ile aynı merkezleme
           spa_c (consistent)     : g_k = d̄_k·1{ d̄_k >= −A_k },  A_k = (ŵ_k/√T)·√(2·ln ln T)
           spa_l (lower, katı)    : g_k = d̄_k·1{ d̄_k >= 0 }
         p_l <= p_c <= p_u her zaman; `p_value` = spa_c (Hansen'in önerdiği tutarlı testtir).

    Dönüş: {"stat", "p_value", "p_lower", "p_consistent", "p_upper", "en_iyi": int,
            "en_iyi_t": float, "w": list[float], "n_boot", "T", "M"}.
    T < 8 (ln ln T tanımsız/anlamsız) / NaN / boş / n_boot <= 0 → None.
    Bir adayın ŵ_k'si 0 ise (sabit fark) o aday istatistikten DÜŞÜLÜR; hepsi düşerse None."""
    arr = _perf_matrix(perf)
    if arr is None or n_boot <= 0:
        return None
    t_n, _ = arr.shape
    if t_n < 8:
        return None
    dbar = arr.mean(axis=0)
    sqrt_t = math.sqrt(t_n)
    idx = stationary_bootstrap_indices(t_n, mean_block, n_boot, seed)
    boot_mean = arr[idx].mean(axis=1)                       # (n_boot, M)
    w = np.std(sqrt_t * boot_mean, axis=0, ddof=1)          # ŵ_k (Hansen 2005, §2.3)
    gecerli = w > 0
    if not np.any(gecerli):
        return None
    dbar_g, w_g = dbar[gecerli], w[gecerli]
    boot_g = boot_mean[:, gecerli]
    stat = float(max(0.0, np.max(sqrt_t * dbar_g / w_g)))
    # Hansen eşiği A_k: ln ln T, T>=8'de pozitif (ln ln 8 = 0.7321).
    a_k = (w_g / sqrt_t) * math.sqrt(2.0 * math.log(math.log(t_n)))
    merkez = {"upper": dbar_g,
              "consistent": np.where(dbar_g >= -a_k, dbar_g, 0.0),
              "lower": np.where(dbar_g >= 0.0, dbar_g, 0.0)}
    p_out: dict[str, float] = {}
    for ad, g in merkez.items():
        z_star = np.max(sqrt_t * (boot_g - g) / w_g, axis=1)
        z_star = np.maximum(z_star, 0.0)
        p_out[ad] = float(np.mean(z_star >= stat))
    t_k = sqrt_t * dbar_g / w_g
    en_iyi_local = int(np.argmax(t_k))
    en_iyi = int(np.flatnonzero(gecerli)[en_iyi_local])
    return {"stat": stat, "p_value": p_out["consistent"], "p_lower": p_out["lower"],
            "p_consistent": p_out["consistent"], "p_upper": p_out["upper"],
            "en_iyi": en_iyi, "en_iyi_t": float(t_k[en_iyi_local]),
            "w": [float(v) for v in w], "n_boot": int(n_boot),
            "T": int(t_n), "M": int(arr.shape[1])}
