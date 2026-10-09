"""
advanced_math.py — Araştırma-düzeyi finans matematiği (AI v3 §WP-M2, Tur-16 Dalga-2). KARAR-DIŞI.

`quant_math.py`'ın ikinci dalgası: sıçrama-difüzyon, stokastik volatilite/opsiyon fiyatlama,
Amerikan opsiyon (PDE + Monte Carlo regresyon), olay-kümelenmesi (Hawkes), çevrimiçi rejim-
kırılması (BOCPD), durum-uzay filtreleri, kaba-volatilite (rough vol), yol-imzaları (path
signatures), motif/anomali taraması ve portföy/numerik altyapı (QP, adaptif integral,
log-sum-exp, FFT konvolüsyon). Saf fonksiyonlar: yan etkisiz, IO/ağ YOK; yalnız numpy/scipy
(+ okuma-amaçlı `nihai/ai/quant_math` içe aktarımı — `hurst_exponent`/`log_returns`) kullanılır.
`nihai/ai/lab.py` ve `nihai/ai/portfolio.py` (Dalga-2 kardeşleri) bu modülü LAZY biçimde
referanslayabilir; BURADAN hiçbir üretim sinyaline/eşiğe/META_GATE'e/execution'a bağlanmaz
(bkz. AI_V3_DERIN_BEYIN_SPEC.md "Değişmez anayasa").

Savunmacı sözleşme (`quant_math.py` ile AYNI, spec §3.8):
  - Girdi kısa/dönüştürülemez/sonlu-olmayan (NaN/Inf) İSE: dönüş tipi `X | None` olan
    fonksiyonlar None döner. Dönüş tipi None İÇERMEYEN yapısal dönüşümler (jump_detect_bipower,
    hawkes_intensity, bocpd, signature_level2/3) KENDİ tipinde zararsız bir değerle (boş dizi /
    None-alanlı dict) döner. Hiçbir girdi İSTİSNA fırlatmaz.
  - Her stokastik fonksiyon (particle_filter_light) ZORUNLU `seed` alır ve DETERMİNİSTİKTİR
    (aynı seed → bit-bit aynı çıktı; np.random.default_rng kullanılır). MLE/kalibrasyon
    fonksiyonları (merton_jump_fit, heston_calibrate, hawkes_fit, garch-tarzı) ÇOKLU-BAŞLANGIÇ
    optimizasyonu kullanır ama rastgelelik İÇERMEZ (sabit aday listesi) → deterministiktir.
  - Sayısal kararsızlık riski taşıyan fonksiyonlar (Heston/FFT, CN-PDE) yakınsamama/kısıt-
    ihlali durumunda çökme yerine None döner; hiçbir çıktı "kesin doğru fiyat" iddiasında
    bulunmaz — Almgren-Chriss tipi TCA (transaction cost analysis) raporlarına benzer biçimde
    TÜM çıktılar YALNIZ bilgi/rapor amaçlıdır, execution/sinyal/eşiğe ASLA bağlanmaz.

Kaynak/konvansiyonlar: Merton (1976) sıçrama-difüzyon MLE, Barndorff-Nielsen & Shephard (2004)
bipower varyasyon sıçrama testi, Heston (1993) stokastik volatilite + Albrecher ve ark. (2007)
"Little Heston Trap" (numerik-kararlı karakteristik fonksiyon), Carr-Madan (1999) FFT opsiyon
fiyatlama, Brennan-Schwartz (1977) Amerikan opsiyon PDE algoritması (upwind ayrıklaştırma ile
M-matris garantisi), Longstaff-Schwartz (2001) en-küçük-kareler Monte Carlo, Hawkes (1971) üstel-
çekirdek öz-uyarımlı süreç MLE'si, Adams & MacKay (2007) Bayesian Online Changepoint Detection
(Student-t öngörülü, Normal-Ters-Gamma eşlenik önsel), Baum-Welch/Hamilton (1989) rejim-değişimli
gizli Markov filtresi, bootstrap parçacık filtresi (Gordon ve ark. 1993), Gatheral-Jaisson-
Rosenbaum (2018) kaba volatilite, Chen (1957) yinelemeli imza kombinasyon kimliği (path
signatures), Yeh ve ark. (2009) STOMP/matrix-profile basitleştirmesi.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.integrate import quad as _quad
from scipy.optimize import least_squares, minimize
from scipy.special import gammaln as _gammaln

from nihai.ai.quant_math import hurst_exponent as _hurst_exponent
from nihai.ai.quant_math import log_returns as _log_returns


# ---------------------------------------------------------------------------
# Girdi temizleme yardımcıları (modül-içi; quant_math'tan bağımsız KENDİ kopyası —
# iki modül birbirine dahili API üzerinden bağımlı olmasın diye kasıtlı tekrar).
# ---------------------------------------------------------------------------
def _as_1d(x: ArrayLike) -> np.ndarray:
    """Gevşek 1B float dönüşüm: çevrilemezse boş dizi (İSTİSNA YOK)."""
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)
    return arr


def _clean_1d(x: ArrayLike, min_len: int = 1) -> np.ndarray | None:
    """Katı 1B float dönüşüm: kısa/dönüştürülemez/sonlu-olmayan (NaN/Inf) içerik → None."""
    arr = _as_1d(x)
    if arr.size < min_len:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _clean_2d(x: ArrayLike, min_rows: int = 2, min_cols: int = 1) -> np.ndarray | None:
    """Katı 2B float dönüşüm (satır=zaman/yol, sütun=boyut/varlık)."""
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < min_rows or arr.shape[1] < min_cols:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _as_float(v: object) -> float | None:
    """Tek skaler değeri sonlu float'a çevirir; başarısızsa None (İSTİSNA YOK)."""
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ===========================================================================
# Sıçrama-difüzyon
# ===========================================================================
def _merton_neg_loglik(params: np.ndarray, r: np.ndarray, k_max: int) -> float:
    """Merton sıçrama-difüzyon karışım modeli negatif log-olabilirlik (Poisson-ağırlıklı
    Gauss karışımı, log-toplam-üstel-kararlı biçimde). Kısıt-dışı/dejenere bölgede BÜYÜK
    ceza (1e10) döner (scipy.optimize.minimize'ı çökertmeden arama uzayını sınırlar)."""
    mu, sigma, lam, jump_mu, jump_sigma = params
    if sigma <= 0.0 or jump_sigma <= 0.0 or lam < 0.0:
        return 1e10
    k = np.arange(k_max + 1, dtype=float)
    lam_safe = max(lam, 1e-300)
    with np.errstate(all="ignore"):
        log_poisson = k * math.log(lam_safe) - lam_safe - _gammaln(k + 1.0)
    var_k = sigma**2 + k * jump_sigma**2
    mean_k = mu + k * jump_mu
    diff = r[:, None] - mean_k[None, :]
    with np.errstate(all="ignore"):
        log_normal = -0.5 * np.log(2.0 * np.pi * var_k)[None, :] - 0.5 * diff**2 / var_k[None, :]
        log_terms = log_poisson[None, :] + log_normal
    m = np.max(log_terms, axis=1)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(all="ignore"):
        inner = np.sum(np.exp(log_terms - m_safe[:, None]), axis=1)
    if np.any(inner <= 0.0) or not np.all(np.isfinite(inner)):
        return 1e10
    log_dens = m_safe + np.log(inner)
    if not np.all(np.isfinite(log_dens)):
        return 1e10
    nll = float(-np.sum(log_dens))
    return nll if math.isfinite(nll) else 1e10


def merton_jump_fit(returns: ArrayLike) -> dict | None:
    """Merton (1976) sıçrama-difüzyon modelinin MLE tahmini (Poisson-ağırlıklı Gauss
    karışımı: N adet sıçrama ~ Poisson(lam), her sıçrama büyüklüğü ~ N(jump_mu,jump_sigma²)).
    scipy.optimize.minimize (L-BFGS-B, sigma/jump_sigma>0 ve lam∈[0,20] kısıtlı) ile ÇOKLU-
    BAŞLANGIÇ (5 sabit aday — karışım-modeli log-olabilirlik yüzeyi çok-modlu olabildiği için
    tek başlangıç riskli) aranır; en yüksek olabilirlikli (en düşük NLL) kabul edilebilir
    sonuç seçilir. Deterministiktir (rastgelelik yok).

    λ→0 KÖPRÜSÜ: veri sıçramasız (saf Gauss) İSE, k=0 terimi (Poisson(0;lam→0)=1) modeli düz
    Normal dağılıma indirger; MLE bu durumda lam'ı 0'a yakın bir değere iter ve mu/sigma örneklem
    ortalama/std'sine yakınsar (bkz. test dosyası: normal MLE karşılaştırması).

    En az 30 gözlem ister (5-parametreli karışım modeli az veride kararsızdır); yakınsamazsa
    None (İSTİSNA YOK).

    Dönüş: {'mu','sigma','lam','jump_mu','jump_sigma','loglik'}."""
    arr = _clean_1d(returns, min_len=30)
    if arr is None:
        return None
    mu0 = float(np.mean(arr))
    sd0 = float(np.std(arr, ddof=1))
    if sd0 <= 0.0 or not math.isfinite(sd0):
        return None
    k_max = 20
    bounds = [(None, None), (1e-8, None), (0.0, 20.0), (None, None), (1e-8, None)]
    starts = (
        (mu0, sd0 * 0.90, 0.01, 0.0, sd0 * 1.5),
        (mu0, sd0 * 0.70, 0.05, 0.0, sd0 * 2.0),
        (mu0, sd0 * 0.50, 0.10, -sd0 * 0.5, sd0 * 2.5),
        (mu0, sd0 * 0.80, 0.02, sd0 * 0.5, sd0 * 1.2),
        (mu0, sd0 * 0.95, 0.001, 0.0, sd0 * 3.0),
    )
    best = None
    for x0 in starts:
        try:
            res = minimize(
                _merton_neg_loglik, np.array(x0), args=(arr, k_max), method="L-BFGS-B",
                bounds=bounds, options={"maxiter": 500, "ftol": 1e-12},
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if not res.success or not math.isfinite(float(res.fun)):
            continue
        mu, sigma, lam, jump_mu, jump_sigma = (float(v) for v in res.x)
        if sigma <= 0.0 or jump_sigma <= 0.0 or lam < 0.0:
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        return None
    mu, sigma, lam, jump_mu, jump_sigma = (float(v) for v in best.x)
    return {
        "mu": mu, "sigma": sigma, "lam": lam, "jump_mu": jump_mu, "jump_sigma": jump_sigma,
        "loglik": float(-best.fun),
    }


def jump_detect_bipower(returns: ArrayLike, z_thresh: float = 4.0) -> np.ndarray:
    """Barndorff-Nielsen & Shephard (2004) bipower varyasyon tabanlı sıçrama tespiti.
    Bipower varyasyon BV = (π/2) * mean(|r_i|·|r_{i-1}|) sürekli (difüzyon) varyansı SIÇRAMALARA
    KARŞI DAYANIKLI tahmin eder (bir sıçrama yalnızca komşu İKİ çarpım terimini kirletir, n
    büyüdükçe asimptotik olarak ihmal edilir). Her bar için z_t = |r_t| / sqrt(BV); z_t>z_thresh
    OLAN bar indeksleri döner (int dizi). Kısa/NaN/dejenere (BV<=0) girdi veya z_thresh<=0 →
    boş int dizi (yapısal dönüşüm — None DEĞİL, çökme yok)."""
    arr = _clean_1d(returns, min_len=3)
    zt = _as_float(z_thresh)
    if arr is None or zt is None or zt <= 0.0:
        return np.array([], dtype=int)
    abs_r = np.abs(arr)
    mu1_sq = 2.0 / math.pi
    bp_var = float(np.mean(abs_r[1:] * abs_r[:-1])) / mu1_sq
    if bp_var <= 0.0 or not math.isfinite(bp_var):
        return np.array([], dtype=int)
    sigma_bp = math.sqrt(bp_var)
    z = abs_r / sigma_bp
    idx = np.where(z > zt)[0]
    return idx.astype(int)


# ===========================================================================
# Stokastik volatilite (Heston + Carr-Madan FFT)
# ===========================================================================
_HESTON_KEYS = ("kappa", "theta", "sigma", "rho", "v0")


def _parse_heston_params(params: object) -> tuple[float, float, float, float, float] | None:
    if not isinstance(params, dict):
        return None
    try:
        kappa, theta, sigma, rho, v0 = (float(params[k]) for k in _HESTON_KEYS)
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (kappa, theta, sigma, rho, v0)):
        return None
    if kappa <= 0.0 or theta <= 0.0 or sigma <= 0.0 or v0 < 0.0 or abs(rho) >= 1.0:
        return None
    return kappa, theta, sigma, rho, v0


def heston_charfn(u: ArrayLike, params: dict, s0: float, r: float, T: float) -> complex | np.ndarray:
    """Heston (1993) log-fiyatın karakteristik fonksiyonu, Albrecher ve ark. (2007) "Little
    Heston Trap" biçiminde (uzun-vadeli T'lerde karmaşık-log dal-kesme süreksizliğini önleyen
    NUMERİK KARARLI form; -d ve g=(...-d)/(...+d) kullanılır, naif +d/g1 formu DEĞİL).
    `params`: {'kappa','theta','sigma','rho','v0'}. Geçersiz params/s0<=0/T<=0/|rho|>=1 veya
    dönüştürülemez girdi → nan+nanj (skaler girdi için) veya nan+nanj dizisi (İSTİSNA YOK —
    bu fonksiyonun dönüş tipinde None YOKTUR, bkz. spec WP-M2 tablosu)."""
    scalar_input = np.isscalar(u) or (isinstance(u, np.ndarray) and u.ndim == 0)
    try:
        uc = np.atleast_1d(np.asarray(u, dtype=complex))
    except (TypeError, ValueError):
        uc = np.array([complex(float("nan"), float("nan"))])
        scalar_input = True
    nan_val = complex(float("nan"), float("nan"))
    parsed = _parse_heston_params(params)
    s0f = _as_float(s0)
    rf = _as_float(r)
    tf = _as_float(T)
    if parsed is None or s0f is None or s0f <= 0.0 or rf is None or tf is None or tf <= 0.0:
        out = np.full(uc.shape, nan_val, dtype=complex)
        return complex(out[0]) if scalar_input else out
    kappa, theta, sigma, rho, v0 = parsed
    i = 1j
    with np.errstate(all="ignore"):
        d = np.sqrt((rho * sigma * i * uc - kappa) ** 2 + sigma**2 * (i * uc + uc**2))
        g = (kappa - rho * sigma * i * uc - d) / (kappa - rho * sigma * i * uc + d)
        exp_dT = np.exp(-d * tf)
        c_term = rf * i * uc * tf + (kappa * theta / sigma**2) * (
            (kappa - rho * sigma * i * uc - d) * tf - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
        )
        d_term = (kappa - rho * sigma * i * uc - d) / sigma**2 * (1.0 - exp_dT) / (1.0 - g * exp_dT)
        result = np.exp(c_term + d_term * v0 + i * uc * math.log(s0f))
    return complex(result[0]) if scalar_input else result


def carr_madan_fft_price(
    s0: float, strikes: ArrayLike, r: float, T: float, heston_params: dict, alpha: float = 1.5,
) -> np.ndarray | None:
    """Carr-Madan (1999) FFT ile Heston-altında Avrupa CALL fiyatları. N=4096 nokta / η=0.25
    frekans-adımı sabit ızgara üzerinde tek FFT hesaplanır, ardından istenen `strikes`'a
    log-strike üzerinden lineer enterpolasyon yapılır. Sayısal-gürültülü tekil noktalar
    (|integrand| sonlu-değil) SIFIRLANIR (bkz. kod) — TÜM ızgaranın >%5'i geçersizse None
    (kondisyon uyarısı, çökme yerine savunmacı None). Geçersiz girdi (s0/T/alpha<=0, eksik/
    geçersiz heston_params) → None.

    BS KÖPRÜSÜ: vol-of-vol (sigma)→0 VE v0=theta (sabit varyans) özel durumunda Heston fiyatı
    Black-Scholes'e yakınsar (bkz. test dosyası)."""
    s0f = _as_float(s0)
    rf = _as_float(r)
    tf = _as_float(T)
    alphaf = _as_float(alpha)
    strikes_arr = _clean_1d(strikes, min_len=1)
    if s0f is None or s0f <= 0.0 or tf is None or tf <= 0.0 or alphaf is None or alphaf <= 0.0:
        return None
    if rf is None or strikes_arr is None or np.any(strikes_arr <= 0.0):
        return None
    if _parse_heston_params(heston_params) is None:
        return None
    n_grid = 4096
    eta = 0.25
    lam = 2.0 * math.pi / (n_grid * eta)
    b = n_grid * lam / 2.0
    u = eta * np.arange(n_grid)
    ku = -b + lam * np.arange(n_grid)
    i = 1j
    charfn_vals = heston_charfn(u - (alphaf + 1.0) * i, heston_params, s0f, rf, tf)
    denom = alphaf**2 + alphaf - u**2 + i * (2.0 * alphaf + 1.0) * u
    with np.errstate(all="ignore"):
        psi = np.exp(-rf * tf) * charfn_vals / denom
    simpson = np.ones(n_grid)
    simpson[1:-1:2] = 4.0
    simpson[2:-1:2] = 2.0
    x = np.exp(i * u * b) * psi * eta * (simpson / 3.0)
    bad = ~(np.isfinite(x.real) & np.isfinite(x.imag))
    if bad.mean() > 0.05:
        return None
    x = np.where(bad, 0.0, x)
    y = np.fft.fft(x)
    call_grid = (np.exp(-alphaf * ku) / math.pi) * y.real
    strike_grid = np.exp(ku)
    valid = np.isfinite(call_grid) & (strike_grid > 0.0)
    if int(np.sum(valid)) < 10:
        return None
    log_k_target = np.log(strikes_arr)
    log_k_grid = ku[valid]
    order = np.argsort(log_k_grid)
    prices = np.interp(log_k_target, log_k_grid[order], call_grid[valid][order])
    prices = np.maximum(prices, 0.0)
    if not np.all(np.isfinite(prices)):
        return None
    return prices


def heston_calibrate(
    strikes: ArrayLike, market_prices: ArrayLike, s0: float, r: float, T: float,
) -> dict | None:
    """Piyasa opsiyon fiyatlarından Heston parametrelerini kısıtlı en-küçük-kareler (scipy.
    optimize.least_squares, kutu-kısıtlı) ile kalibre eder — model fiyatı `carr_madan_fft_price`
    ile hesaplanır, hedef fonksiyon (model-piyasa) fark vektörüdür. Başlangıç: kappa=1.5,
    theta=0.04, sigma=0.3, rho=-0.5, v0=0.04 (piyasa-tipik tek başlangıç — kalibrasyon zaten
    çok-nokta gözlem/optimizasyon içerir, MLE'nin aksine tek başlangıç yeterli kabul edilir).
    En az 3 (strike,fiyat) çifti ister; boyut uyuşmazlığı/geçersiz s0,T veya yakınsamama →
    None. NOT: parametre kimliklenebilirliği (identifiability) zayıf olabilir — dönüş,
    KALİBRE EDİLMİŞ FİYATLARIN piyasaya yakınlığını garanti eder, gerçek 'doğru' Heston
    parametrelerini DEĞİL (bkz. test dosyası: fiyat-eşleşme testi, parametre-eşleşme değil).

    Dönüş: {'kappa','theta','sigma','rho','v0','cost','feller','feller_ok'}.

    FELLER KOŞULU (Tur-76, Madde 049): `2·κ·θ > ξ²` sağlanmazsa varyans süreci sıfıra
    DOKUNABİLİR ve orada yerel olarak dejenere olur (Monte Carlo ayrıklaştırması negatif
    varyans üretir, örtük vol yüzeyi kısa vadede patlar). Kalibrasyon bunu bir KISIT
    olarak DAYATMAZ — dayatmak, piyasanın gerçekten Feller-dışı fiyatladığı bir yüzeyi
    zorla içeri çekip uyum hatasını gizlerdi. Bunun yerine ÖLÇÜLÜR ve taşınır:
    `feller = 2κθ − ξ²` ve `feller_ok = feller > 0`. Bu, Tur-49'un "bir optimizatör
    çıktısını raporlamadan önce sınırda mı durdu diye SOR ve cevabı çıktıyla TAŞI"
    kuralının Heston tarafıdır."""
    strikes_arr = _clean_1d(strikes, min_len=3)
    prices_arr = _clean_1d(market_prices, min_len=3)
    s0f = _as_float(s0)
    rf = _as_float(r)
    tf = _as_float(T)
    if strikes_arr is None or prices_arr is None or strikes_arr.size != prices_arr.size:
        return None
    if s0f is None or s0f <= 0.0 or tf is None or tf <= 0.0 or rf is None:
        return None

    def resid(x: np.ndarray) -> np.ndarray:
        kappa, theta, sigma, rho, v0 = x
        params = {"kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho, "v0": v0}
        model = carr_madan_fft_price(s0f, strikes_arr, rf, tf, params)
        if model is None:
            return np.full(strikes_arr.size, 1e3)
        return model - prices_arr

    x0 = np.array([1.5, 0.04, 0.3, -0.5, 0.04])
    lb = np.array([1e-3, 1e-4, 1e-3, -0.999, 1e-4])
    ub = np.array([20.0, 4.0, 5.0, 0.999, 4.0])
    try:
        res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=300)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None
    if res.x is None or not np.all(np.isfinite(res.x)):
        return None
    kappa, theta, sigma, rho, v0 = (float(v) for v in res.x)
    feller = 2.0 * kappa * theta - sigma * sigma
    return {"kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho, "v0": v0,
            "cost": float(res.cost),
            # Madde 049 (Tur-76): kısıt DAYATILMAZ, ÖLÇÜLÜR ve taşınır.
            "feller": float(feller), "feller_ok": bool(feller > 0.0)}


# ===========================================================================
# Amerikan opsiyon (PDE Crank-Nicolson + Longstaff-Schwartz LSMC)
# ===========================================================================
def american_crank_nicolson(
    s0: float, k: float, r: float, sigma: float, T: float, n_s: int = 200, n_t: int = 200,
) -> float | None:
    """Amerikan PUT fiyatı: log-fiyat ızgarasında (sabit PDE katsayıları — sayısal kararlılık)
    Crank-Nicolson zaman-ayrıklaştırması + Brennan-Schwartz (1977) algoritmasıyla erken-
    kullanım kısıtının (V>=payoff) TEK-GEÇİŞ (iteratif PSOR YERİNE — çok daha hızlı, aynı
    LCP'nin kesin çözümü) tridiagonal çözümü. İlk-türev (drift) terimi UPWIND ayrıklaştırılır
    (drift işaretine göre ileri/geri fark) — bu, ayrıklaştırılmış operatörün M-matris (Brennan-
    Schwartz'ın geçerliliği için gerekli köşegen-baskın işaret yapısı) olmasını GARANTİ eder;
    merkezi-fark kullanımı büyük drift/küçük ızgara-aralığında bu garantiyi bozabilirdi.

    Alt sınır (S≈0): American put orada HER ZAMAN tam içsel değere (K) eşittir (Avrupa'nın
    aksine — anında kullanım her zaman mümkün); üst sınır (S→∞): V→0.

    Geçersiz girdi (s0/k/T/sigma<=0, n_s<10, n_t<5, sonlu-değil) → None. r=0 özel durumunda
    (erken-kullanım avantajı YOK — nakit tutmanın faiz kazancı olmadığından) Avrupa put'a
    YAKINSAR (bkz. test dosyası); r>0'da erken-kullanım primi nedeniyle Avrupa'dan BÜYÜK olmalı.

    NOT (bilinçli kapsam): yalnız PUT — Amerikan CALL (temettüsüz) zaten erken-kullanılmaz
    (Avrupa'ya eşittir), ayrı bir çözücü gerektirmez."""
    s0f = _as_float(s0)
    kf = _as_float(k)
    rf = _as_float(r)
    sigf = _as_float(sigma)
    tf = _as_float(T)
    try:
        ns, nt = int(n_s), int(n_t)
    except (TypeError, ValueError):
        return None
    if s0f is None or s0f <= 0.0 or kf is None or kf <= 0.0 or tf is None or tf <= 0.0:
        return None
    if rf is None or sigf is None or sigf <= 0.0 or ns < 10 or nt < 5:
        return None

    x0 = math.log(s0f)
    spread = sigf * math.sqrt(tf) * 6.0 + 1.0
    x = np.linspace(x0 - spread, x0 + spread, ns)
    dx = float(x[1] - x[0])
    if dx <= 0.0:
        return None
    dt = tf / nt
    s_grid = np.exp(x)
    payoff = np.maximum(kf - s_grid, 0.0)
    v = payoff.copy()

    drift = rf - 0.5 * sigf**2
    diff_coef = 0.5 * sigf**2 / dx**2
    if drift >= 0.0:
        a_coef = diff_coef
        b_coef = -2.0 * diff_coef - drift / dx - rf
        c_coef = diff_coef + drift / dx
    else:
        a_coef = diff_coef - drift / dx
        b_coef = -2.0 * diff_coef + drift / dx - rf
        c_coef = diff_coef

    lo = -0.5 * dt * a_coef
    diag = 1.0 - 0.5 * dt * b_coef
    up = -0.5 * dt * c_coef
    lo_rhs = 0.5 * dt * a_coef
    diag_rhs = 1.0 + 0.5 * dt * b_coef
    up_rhs = 0.5 * dt * c_coef

    m = ns - 2
    if m < 1 or diag == 0.0:
        return None
    v_left = float(payoff[0])   # American: S≈0 sınırında TAM içsel değer (K), τ'dan bağımsız
    v_right = 0.0
    payoff_int = payoff[1:-1]

    for _ in range(nt):
        rhs_int = lo_rhs * v[:-2] + diag_rhs * v[1:-1] + up_rhs * v[2:]
        rhs_int = rhs_int.copy()
        rhs_int[0] -= lo * v_left
        rhs_int[-1] -= up * v_right

        c_star = np.empty(m)
        d_star = np.empty(m)
        c_star[0] = up / diag
        d_star[0] = rhs_int[0] / diag
        for idx in range(1, m):
            denom = diag - lo * c_star[idx - 1]
            if denom == 0.0 or not math.isfinite(denom):
                return None
            c_star[idx] = up / denom
            d_star[idx] = (rhs_int[idx] - lo * d_star[idx - 1]) / denom
        if not np.all(np.isfinite(c_star)) or not np.all(np.isfinite(d_star)):
            return None

        v_int = np.empty(m)
        v_int[-1] = max(d_star[-1], payoff_int[-1])
        for idx in range(m - 2, -1, -1):
            v_int[idx] = max(d_star[idx] - c_star[idx] * v_int[idx + 1], payoff_int[idx])

        v_new = np.empty(ns)
        v_new[0] = v_left
        v_new[-1] = v_right
        v_new[1:-1] = v_int
        v = v_new

    if not np.all(np.isfinite(v)):
        return None
    price = float(np.interp(x0, x, v))
    return price if math.isfinite(price) and price >= 0.0 else None


# Bilinçli sapma (WP-M2 çekirdek brifi bu ismi kullanıyordu; SPEC §3.8 `american_crank_nicolson`
# diyor — iki taraf uyumlu tek isim + geriye-dönük takma ad).
american_put_crank_nicolson = american_crank_nicolson


def longstaff_schwartz_lsmc(paths: ArrayLike, k: float, r: float, dt: float) -> float | None:
    """Longstaff-Schwartz (2001) en-küçük-kareler Monte Carlo — Amerikan PUT fiyatı.
    `paths`: (n_paths, n_steps+1) simüle fiyat yolları (ör. `quant_math.mc_gbm_paths` çıktısı;
    sütun 0 = t=0). Geriye-doğru: her adımda İÇİNDE-PARA (ITM) yollarda devam-değeri, taban
    {1,x,x²} polinom regresyonuyla (López de Prado/Longstaff-Schwartz konvansiyonu — basit
    polinom, Laguerre değil) tahmin edilir; kullanım-değeri>devam-değeri İSE o an kullanılır.
    Regresyon dejenere (lstsq başarısız) olan adımda o adımda kullanım yapılmaz (güvenli
    varsayılan — devam edilir). `dt` sabit adım aralığı (iskonto: exp(-r*dt)).

    Geçersiz girdi (paths<2 satır/sütun, k/dt<=0, sonlu-değil) → None.

    BS/CN KÖPRÜSÜ: yeterli yol/adım sayısında `american_crank_nicolson` ile YAKLAŞIK
    eşleşmesi beklenir (Monte Carlo gürültüsü + ayrıklaştırma nedeniyle TAM DEĞİL — bkz. test
    dosyası tolerans notu)."""
    arr = _clean_2d(paths, min_rows=2, min_cols=2)
    kf = _as_float(k)
    rf = _as_float(r)
    dtf = _as_float(dt)
    if arr is None or kf is None or kf <= 0.0 or dtf is None or dtf <= 0.0 or rf is None:
        return None
    n_paths, n_cols = arr.shape
    n_steps = n_cols - 1
    if n_steps < 1:
        return None
    discount = math.exp(-rf * dtf)
    payoff = np.maximum(kf - arr, 0.0)
    cashflow = payoff[:, -1].copy()

    for t in range(n_steps - 1, 0, -1):
        cashflow = cashflow * discount
        itm = payoff[:, t] > 0.0
        n_itm = int(np.sum(itm))
        if n_itm < 3:
            continue
        x_v = arr[itm, t]
        y_v = cashflow[itm]
        design = np.column_stack([np.ones_like(x_v), x_v, x_v**2])
        try:
            coef, *_ = np.linalg.lstsq(design, y_v, rcond=None)
        except np.linalg.LinAlgError:
            continue
        continuation = design @ coef
        exercise = payoff[itm, t]
        exer_mask = exercise > continuation
        idx_itm = np.where(itm)[0]
        cashflow[idx_itm[exer_mask]] = exercise[exer_mask]

    price = float(np.mean(cashflow) * discount)
    return price if math.isfinite(price) and price >= 0.0 else None


# ===========================================================================
# Olay kümelenmesi (Hawkes süreci)
# ===========================================================================
def _hawkes_neg_loglik(params: np.ndarray, t_sorted: np.ndarray, big_t: float) -> float:
    mu, alpha, beta = params
    if mu <= 0.0 or alpha < 0.0 or beta <= 0.0 or alpha >= beta:
        return 1e10   # alpha<beta: durağanlık (dallanma oranı<1) şartı
    n = t_sorted.size
    a_rec = np.zeros(n)
    for idx in range(1, n):
        a_rec[idx] = math.exp(-beta * (t_sorted[idx] - t_sorted[idx - 1])) * (1.0 + a_rec[idx - 1])
    lam = mu + alpha * a_rec
    if np.any(lam <= 0.0) or not np.all(np.isfinite(lam)):
        return 1e10
    term1 = float(np.sum(np.log(lam)))
    term2 = mu * big_t
    with np.errstate(all="ignore"):
        term3 = (alpha / beta) * float(np.sum(1.0 - np.exp(-beta * (big_t - t_sorted))))
    nll = -(term1 - term2 - term3)
    return nll if math.isfinite(nll) else 1e10


def hawkes_fit(event_times: ArrayLike, T: float) -> dict | None:
    """Tek-boyutlu üstel-çekirdek Hawkes (1971) sürecinin MLE tahmini: λ(t) = mu + alpha *
    sum_{t_i<t} exp(-beta*(t-t_i)). Log-olabilirlik özyinelemeli A_i biçiminde hesaplanır
    (Ozaki 1979 — O(n), tam O(n²) toplam yerine). Kısıt: alpha<beta (dallanma oranı=alpha/beta
    <1, durağanlık şartı — aksi halde süreç patlar). ÇOKLU-BAŞLANGIÇ (5 aday, veri-bağımlı
    ortalama-oran ölçekli) + L-BFGS-B; en yüksek olabilirlikli kabul edilebilir sonuç seçilir.
    Deterministiktir. Olay zamanları [0,T] aralığında olmalı; en az 5 olay ister.

    Geçersiz girdi (T<=0, olay [0,T] dışı, <5 olay/NaN) → None.

    Dönüş: {'mu','alpha','beta','loglik','branching'} (branching=alpha/beta, dallanma oranı)."""
    arr = _clean_1d(event_times, min_len=5)
    tf = _as_float(T)
    if arr is None or tf is None or tf <= 0.0:
        return None
    t_sorted = np.sort(arr)
    if t_sorted[0] < 0.0 or t_sorted[-1] > tf:
        return None
    n = t_sorted.size
    mean_rate = n / tf
    bounds = [(1e-9, None), (0.0, None), (1e-6, None)]
    starts = (
        (mean_rate * 0.5, 0.3, 1.0), (mean_rate * 0.3, 0.5, 2.0), (mean_rate * 0.7, 0.2, 0.5),
        (mean_rate * 0.4, 0.6, 1.5), (mean_rate * 0.2, 0.1, 3.0),
    )
    best = None
    for x0 in starts:
        try:
            res = minimize(
                _hawkes_neg_loglik, np.array(x0), args=(t_sorted, tf), method="L-BFGS-B",
                bounds=bounds, options={"maxiter": 400, "ftol": 1e-12},
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if not res.success or not math.isfinite(float(res.fun)):
            continue
        mu, alpha, beta = (float(v) for v in res.x)
        if mu <= 0.0 or alpha < 0.0 or beta <= 0.0 or alpha >= beta:
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        return None
    mu, alpha, beta = (float(v) for v in best.x)
    return {"mu": mu, "alpha": alpha, "beta": beta, "loglik": float(-best.fun),
            "branching": alpha / beta}


def hawkes_intensity(event_times: ArrayLike, t_grid: ArrayLike, params: dict) -> np.ndarray:
    """Verilen `params` ({'mu','alpha','beta'}) ile `t_grid` noktalarında Hawkes yoğunluğu
    λ(t)=mu+alpha*sum_{t_i<t} exp(-beta*(t-t_i)). Geçersiz params/boş grid → nan-dolu (aynı
    şekilde) dizi; boş `t_grid` → boş dizi (İSTİSNA YOK, yapısal dönüşüm — None DEĞİL)."""
    grid = _as_1d(t_grid)
    if grid.size == 0:
        return np.array([], dtype=float)
    try:
        mu = float(params["mu"])
        alpha = float(params["alpha"])
        beta = float(params["beta"])
    except (KeyError, TypeError, ValueError, AttributeError):
        return np.full(grid.shape, np.nan)
    if not all(math.isfinite(v) for v in (mu, alpha, beta)) or beta <= 0.0:
        return np.full(grid.shape, np.nan)
    events = _as_1d(event_times)
    events = np.sort(events[np.isfinite(events)]) if events.size else events
    out = np.empty(grid.shape, dtype=float)
    for idx, t in enumerate(grid):
        past = events[events < t] if events.size else events
        if past.size == 0:
            out[idx] = mu
        else:
            out[idx] = mu + alpha * float(np.sum(np.exp(-beta * (t - past))))
    return out


# ===========================================================================
# Çevrimiçi rejim kırılması (BOCPD)
# ===========================================================================
def bocpd(x: ArrayLike, hazard: float = 1.0 / 250.0) -> dict:
    """Adams & MacKay (2007) Bayesian Online Changepoint Detection — Student-t öngörülü
    (Normal-Ters-Gamma eşlenik önsel), sabit tehlike (hazard) oranlı. Her `t`'de aktif
    run-length (r) hipotezleri için posterior R_t(r) çevrimiçi güncellenir; hesaplama
    maliyetini sınırlamak için run-length listesi max 300 hipotezle KIRPILIR (en uzun/en
    az-olası kuyruk atılır, kalan yeniden normalize edilir — pratik BOCPD uygulaması,
    Adams-MacKay orijinal makalesinde de önerilen bir optimizasyon).

    Önseller veri-bağımlı ölçeklenir (mu0=örneklem ortalaması, beta0=örneklem varyansı) —
    veri-bağımsız sabit önsel yerine sayısal kararlılık için. Deterministiktir (rastgelelik
    yok — yalnız veriye bağlı).

    Kısa (<5 nokta)/NaN/hazard∉(0,1) girdi → tüm alanlar boş dizi (İSTİSNA YOK, yapısal
    dönüşüm — None DEĞİL).

    NOT (matematiksel köşe-durum): R_t(0) TEK BAŞINA (yalnız run-length=0 hipotezi) sabit
    tehlike oranında HER ZAMAN tam olarak `hazard`'a eşittir — bu, özyinelemenin
    (cp=hazard·S, growth toplamı=(1-hazard)·S, toplam=S) cebirsel bir SONUCUDUR, veriye
    duyarlı DEĞİLDİR (dolayısıyla tek başına yararsız bir sinyaldir). Bu yüzden
    `changepoint_prob`, PRATİKTE bilgilendirici olan toplu büyüklüğü raporlar: KISA run-length
    hipotezlerinin (r<=min(3,run_count-1)) toplam posterior kütlesi — bu, gerçek bir
    değişim-noktasından hemen sonra belirgin biçimde YÜKSELİR (bkz. test dosyası: `map_run_
    length` de aynı davranışı çok daha net gösterir — zirveye tırmanır, değişimde 1'e düşer,
    yeniden tırmanır; birincil/en güvenilir sinyal ODUR).

    Dönüş: {'changepoint_prob' (kısa run-length toplu posterior kütlesi — bkz. yukarı not),
    'map_run_length' (MAP run-length tahmini; değişim noktasında keskin düşüş gösterir),
    'n'}."""
    empty = {"changepoint_prob": np.array([], dtype=float),
              "map_run_length": np.array([], dtype=int), "n": 0}
    arr = _as_1d(x)
    hz = _as_float(hazard)
    if arr.size < 5 or not np.all(np.isfinite(arr)) or hz is None or not (0.0 < hz < 1.0):
        return empty

    n = arr.size
    mu0 = float(np.mean(arr))
    var0 = float(np.var(arr))
    var0 = var0 if var0 > 0.0 else 1.0
    kappa0, alpha0, beta0 = 1.0, 1.0, max(var0, 1e-8)
    max_run = min(n, 300)

    mus = [mu0]
    kappas = [kappa0]
    alphas = [alpha0]
    betas = [beta0]
    r_probs = np.array([1.0])

    cp_prob = np.empty(n)
    map_run = np.empty(n, dtype=int)

    for t in range(n):
        xt = float(arr[t])
        mus_a = np.array(mus)
        kappas_a = np.array(kappas)
        alphas_a = np.array(alphas)
        betas_a = np.array(betas)
        dof = 2.0 * alphas_a
        scale = np.sqrt(betas_a * (kappas_a + 1.0) / (alphas_a * kappas_a))
        with np.errstate(all="ignore"):
            pred = stats.t.pdf(xt, df=dof, loc=mus_a, scale=scale)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        joint = r_probs * pred
        growth = joint * (1.0 - hz)
        cp = float(np.sum(joint * hz))
        new_r = np.concatenate(([cp], growth))
        total = float(np.sum(new_r))
        new_r = new_r / total if total > 0.0 and math.isfinite(total) else np.full(
            new_r.shape, 1.0 / new_r.size)

        new_mus = [mu0]
        new_kappas = [kappa0]
        new_alphas = [alpha0]
        new_betas = [beta0]
        for i in range(len(mus)):
            k_i, a_i, b_i, m_i = kappas[i], alphas[i], betas[i], mus[i]
            k_new = k_i + 1.0
            m_new = (k_i * m_i + xt) / k_new
            a_new = a_i + 0.5
            dx_i = xt - m_i                       # `**2` DEĞİL: CPython float `**` taşmada
            b_new = b_i + (k_i * dx_i * dx_i) / (2.0 * k_new)   # OverflowError fırlatır (Tur-49)
            new_mus.append(m_new)
            new_kappas.append(k_new)
            new_alphas.append(a_new)
            new_betas.append(b_new)

        if len(new_mus) > max_run:
            new_mus = new_mus[:max_run]
            new_kappas = new_kappas[:max_run]
            new_alphas = new_alphas[:max_run]
            new_betas = new_betas[:max_run]
            new_r = new_r[:max_run]
            r_sum = float(np.sum(new_r))
            new_r = new_r / r_sum if r_sum > 0.0 else np.full(new_r.shape, 1.0 / new_r.size)

        mus, kappas, alphas, betas = new_mus, new_kappas, new_alphas, new_betas
        r_probs = new_r
        short_cut = min(3, r_probs.size - 1)
        cp_prob[t] = float(np.sum(r_probs[: short_cut + 1]))
        map_run[t] = int(np.argmax(r_probs))

    return {"changepoint_prob": cp_prob, "map_run_length": map_run, "n": n}


# ===========================================================================
# Durum-uzay filtreleri (rejim-değişimli HMM + bootstrap parçacık filtresi)
# ===========================================================================
def kalman_regime_switching(y: ArrayLike, n_regime: int = 2) -> dict | None:
    """Basit 2-rejimli Gauss gizli-Markov filtresi (Hamilton 1989 tarzı rejim-değişimi;
    Baum-Welch ileri-geri (forward-backward) EM ile tahmin — YALNIZ n_regime=2 desteklenir,
    "basit" sözleşmesi gereği genel n-rejim genellemesi YOK). Başlangıç: medyan-bölünmüş iki
    küme ortalama/std'si; 30 sabit EM iterasyonu (yakınsama garantisi yerine sınırlı-bütçe —
    deterministik, rastgelelik yok). Sayısal kararsızlık (sıfır/negatif ölçek, ıraksak forward
    olasılığı) → None.

    En az 20 gözlem ister; n_regime!=2 veya kısa/NaN girdi → None.

    Dönüş: {'mu' (2), 'sigma' (2), 'trans' (2x2 geçiş matrisi), 'filtered_prob' (n,2 yumuşatılmış
    posterior — gamma), 'labels' (n, argmax rejim), 'loglik'}."""
    arr = _clean_1d(y, min_len=20)
    try:
        n_reg = int(n_regime)
    except (TypeError, ValueError):
        return None
    if arr is None or n_reg != 2:
        return None
    n = arr.size
    median = float(np.median(arr))
    lo_mask = arr <= median
    hi_mask = ~lo_mask
    if int(np.sum(lo_mask)) < 2 or int(np.sum(hi_mask)) < 2:
        return None
    mu = np.array([float(np.mean(arr[lo_mask])), float(np.mean(arr[hi_mask]))])
    if mu[0] > mu[1]:
        mu = mu[::-1]
        lo_mask, hi_mask = hi_mask, lo_mask
    global_sd = float(np.std(arr)) or 1.0
    sd = np.array([
        float(np.std(arr[lo_mask])) or global_sd,
        float(np.std(arr[hi_mask])) or global_sd,
    ])
    sd = np.where(sd <= 0.0, global_sd, sd)
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi0 = np.array([0.5, 0.5])

    gamma = None
    c = None
    for _ in range(30):
        emis = np.empty((n, 2))
        for j in range(2):
            emis[:, j] = stats.norm.pdf(arr, loc=mu[j], scale=max(float(sd[j]), 1e-8))
        emis = np.clip(emis, 1e-300, None)

        alpha_f = np.empty((n, 2))
        c = np.empty(n)
        alpha_f[0] = pi0 * emis[0]
        c[0] = float(alpha_f[0].sum())
        if c[0] <= 0.0 or not math.isfinite(c[0]):
            return None
        alpha_f[0] /= c[0]
        for t in range(1, n):
            alpha_f[t] = (alpha_f[t - 1] @ trans) * emis[t]
            c[t] = float(alpha_f[t].sum())
            if c[t] <= 0.0 or not math.isfinite(c[t]):
                return None
            alpha_f[t] /= c[t]

        beta_b = np.empty((n, 2))
        beta_b[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta_b[t] = trans @ (emis[t + 1] * beta_b[t + 1]) / c[t + 1]

        gamma = alpha_f * beta_b
        row_sum = gamma.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0.0, row_sum, 1.0)
        gamma = gamma / row_sum

        xi_sum = np.zeros((2, 2))
        for t in range(n - 1):
            xi_t = np.outer(alpha_f[t], emis[t + 1] * beta_b[t + 1]) * trans / c[t + 1]
            xi_sum += xi_t
        row_tot = xi_sum.sum(axis=1, keepdims=True)
        row_tot = np.where(row_tot > 0.0, row_tot, 1.0)
        new_trans = xi_sum / row_tot

        gamma_sum = gamma.sum(axis=0)
        gamma_sum = np.where(gamma_sum > 0.0, gamma_sum, 1.0)
        new_mu = (gamma.T @ arr) / gamma_sum
        new_sd = np.empty(2)
        for j in range(2):
            new_sd[j] = math.sqrt(float(np.sum(gamma[:, j] * (arr - new_mu[j]) ** 2)) / gamma_sum[j])
        new_sd = np.where(new_sd <= 0.0, global_sd, new_sd)

        if np.max(np.abs(new_mu - mu)) < 1e-8 and np.max(np.abs(new_sd - sd)) < 1e-8:
            mu, sd, trans = new_mu, new_sd, new_trans
            break
        mu, sd, trans = new_mu, new_sd, new_trans
        pi0 = gamma[0]

    if gamma is None or c is None or not np.all(np.isfinite(c)) or np.any(c <= 0.0):
        return None
    loglik = float(np.sum(np.log(c)))
    labels = np.argmax(gamma, axis=1).astype(int)
    return {"mu": mu.tolist(), "sigma": sd.tolist(), "trans": trans.tolist(),
            "filtered_prob": gamma, "labels": labels, "loglik": loglik}


def particle_filter_light(y: ArrayLike, n_particles: int = 1000, seed: int = 42) -> dict | None:
    """Bootstrap parçacık filtresi (Gordon ve ark. 1993), yerel-seviye durum-uzay modeli
    (y_t = seviye_t + gözlem-gürültüsü, seviye_t = seviye_{t-1} + süreç-gürültüsü) — dinamik
    beta/Kalman-beta'nın (bkz. quant_math.kalman_beta) doğrusal-olmayan/parçacık-tabanlı
    genellemesi. Gürültü ölçekleri veriden kaba sezgiyle tahmin edilir (obs_std=0.5*std(y),
    proc_std=0.5*std(diff(y))) — tam MLE kalibrasyonu YOK (fonksiyon adında "light").
    Etkin-örneklem-büyüklüğü (ESS) < n/2 olduğunda sistematik-YERİNE multinomial yeniden-
    örnekleme (rng.choice, ağırlıklı) yapılır. `seed` ZORUNLU determinizm için (aynı seed →
    bit-bit aynı çıktı). n_particles [10,5000] aralığına KIRPILIR (savunmacı üst sınır).

    Kısa (<5 nokta)/NaN/n_particles<10 girdi → None.

    Dönüş: {'filtered_mean','filtered_std','n_particles'}."""
    arr = _clean_1d(y, min_len=5)
    if arr is None:
        return None
    try:
        n_part = int(n_particles)
        seed_i = int(seed)
    except (TypeError, ValueError):
        return None
    if n_part < 10:
        return None
    n_part = min(n_part, 5000)

    rng = np.random.default_rng(seed_i)
    n = arr.size
    diffs = np.diff(arr)
    global_sd = float(np.std(arr))
    obs_std = global_sd * 0.5 if global_sd > 0.0 else 1.0
    diff_sd = float(np.std(diffs)) if diffs.size else 0.0
    proc_std = diff_sd * 0.5 if diff_sd > 0.0 else max(obs_std * 0.1, 1e-6)

    particles = rng.normal(float(arr[0]), obs_std, size=n_part)
    filtered_mean = np.empty(n)
    filtered_std = np.empty(n)
    filtered_mean[0] = float(np.mean(particles))
    filtered_std[0] = float(np.std(particles))

    for t in range(1, n):
        particles = particles + rng.normal(0.0, proc_std, size=n_part)
        w = stats.norm.pdf(float(arr[t]), loc=particles, scale=max(obs_std, 1e-8))
        w_sum = float(w.sum())
        weights = w / w_sum if w_sum > 0.0 and math.isfinite(w_sum) else np.full(n_part, 1.0 / n_part)
        ess = 1.0 / float(np.sum(weights**2))
        if ess < n_part / 2.0:
            idx = rng.choice(n_part, size=n_part, p=weights)
            particles = particles[idx]
            weights = np.full(n_part, 1.0 / n_part)
        filtered_mean[t] = float(np.sum(weights * particles))
        filtered_std[t] = float(np.sqrt(np.sum(weights * (particles - filtered_mean[t]) ** 2)))

    if not np.all(np.isfinite(filtered_mean)) or not np.all(np.isfinite(filtered_std)):
        return None
    return {"filtered_mean": filtered_mean, "filtered_std": filtered_std, "n_particles": n_part}


# ===========================================================================
# Kaba volatilite (rough volatility)
# ===========================================================================
_RV_WINDOW = 20


def _rolling_realized_vol(closes: np.ndarray, window: int = _RV_WINDOW) -> np.ndarray | None:
    """Kapanışlardan log-getiri + `window`-barlık kayan-pencere gerçekleşen-vol serisi (ddof=1).
    Yetersiz veri → None."""
    r = _log_returns(closes)
    if r.size < window + 20:
        return None
    vols = [float(np.std(r[i - window:i], ddof=1)) for i in range(window, r.size + 1)]
    arr = np.asarray(vols, dtype=float)
    return arr if arr.size >= 40 and np.all(arr > 0.0) else None


def realized_vol_hurst(closes: ArrayLike) -> float | None:
    """Gerçekleşen (20-bar kayan-pencere log-getiri std'si) volatilite SERİSİNİN Hurst üsteli
    (bkz. `quant_math.hurst_exponent` — aynı yapı-fonksiyonu yöntemi, burada vol serisine
    uygulanır). Gatheral-Jaisson-Rosenbaum (2018): gerçekleşen vol tipik olarak H<0.5 ("kaba"/
    rough) sergiler. En az 100 kapanış ister (20-bar pencere + Hurst'ün kendi min-örneklem
    şartı için yeterli vol-serisi uzunluğu); kısa/NaN/dejenere → None."""
    arr = _clean_1d(closes, min_len=100)
    if arr is None:
        return None
    vol_arr = _rolling_realized_vol(arr)
    if vol_arr is None:
        return None
    return _hurst_exponent(vol_arr, max_lag=min(20, vol_arr.size // 3))


def rough_vol_forecast(closes: ArrayLike, horizon: int = 5) -> dict | None:
    """Kaba-volatilite (rough vol) tabanlı BASİT öngörü (bilinçli yaklaşıklık — tam rBergomi
    kalibrasyonu DEĞİL): log-vol serisi uzun-vadeli ortalamasına, Hurst'e bağlı bir hızla
    üstel olarak geri döner (kappa = max(0.05, 1-H) — H<0.5 [kaba/anti-kalıcı] daha hızlı
    ortalamaya-dönüş anlamına gelir, bu basit haritalama docstring'te AÇIKÇA yaklaşıklık
    olarak işaretlenir). forecast_log[h] = mean_log_vol + (last_log_vol-mean_log_vol)*
    exp(-kappa*h), h=1..horizon.

    En az 100 kapanış/horizon<1/NaN/dejenere vol-serisi → None.

    Dönüş: {'hurst','last_vol','forecast' (liste, uzunluk=horizon),'horizon'}."""
    arr = _clean_1d(closes, min_len=100)
    if arr is None:
        return None
    try:
        hz = int(horizon)
    except (TypeError, ValueError):
        return None
    if hz < 1:
        return None
    vol_arr = _rolling_realized_vol(arr)
    if vol_arr is None:
        return None
    h_val = _hurst_exponent(vol_arr, max_lag=min(20, vol_arr.size // 3))
    if h_val is None:
        return None
    log_vol = np.log(vol_arr)
    mean_log_vol = float(np.mean(log_vol))
    last_log_vol = float(log_vol[-1])
    kappa = max(0.05, 1.0 - h_val)
    steps = np.arange(1, hz + 1, dtype=float)
    forecast_log = mean_log_vol + (last_log_vol - mean_log_vol) * np.exp(-kappa * steps)
    forecast_vol = np.exp(forecast_log)
    if not np.all(np.isfinite(forecast_vol)):
        return None
    return {"hurst": float(h_val), "last_vol": float(vol_arr[-1]),
            "forecast": forecast_vol.tolist(), "horizon": hz}


# ===========================================================================
# Yol imzaları (path signatures) — Chen (1957) yinelemeli kombinasyon kimliği
# ===========================================================================
def _chen_combine(
    s1_a: np.ndarray, s2_a: np.ndarray, s1_b: np.ndarray, s2_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """İki ardışık yol-parçasının seviye-1/2 imzalarını Chen kimliğiyle birleştirir (a=önce
    biriken, b=yeni parça): S1=S1_a+S1_b, S2=S2_a + S1_a⊗S1_b + S2_b."""
    s1 = s1_a + s1_b
    s2 = s2_a + np.outer(s1_a, s1_b) + s2_b
    return s1, s2


def _chen_combine3(
    s1_a: np.ndarray, s2_a: np.ndarray, s3_a: np.ndarray,
    s1_b: np.ndarray, s2_b: np.ndarray, s3_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seviye-3 için Chen kimliği (seviye-1/2 `_chen_combine` ile aynı; seviye-3 ek terim
    S3 = S3_a + S2_a⊗S1_b + S1_a⊗S2_b + S3_b)."""
    s1, s2 = _chen_combine(s1_a, s2_a, s1_b, s2_b)
    s3 = (
        s3_a
        + np.einsum("ij,k->ijk", s2_a, s1_b)
        + np.einsum("i,jk->ijk", s1_a, s2_b)
        + s3_b
    )
    return s1, s2, s3


def signature_level2(path_2d: ArrayLike) -> np.ndarray:
    """Yol-imzası (path signature), seviye-2'ye kesilmiş — Chen (1957) yinelemeli kombinasyon
    kimliğiyle parça-doğrusal (piecewise-linear) yol için TAM (yaklaşıklık DEĞİL) kapalı-form
    hesaplama. `path_2d`: (n_nokta, d_boyut). Dönüş: [S1 (d) | S2.ravel() (d²)] tek boyutlu
    dizi (uzunluk d+d²). Model-bağımsız zaman-serisi özellik üretimi (neural girdisine ek —
    bkz. AI_V3_DERIN_BEYIN_PLAN §C). <2 nokta/dejenere girdi → boş dizi (İSTİSNA YOK, yapısal
    dönüşüm — None DEĞİL).

    ANALİTİK ÇAPA (test dosyasında doğrulanır): 1-boyutlu yol için S2 = 0.5*(toplam-artış)²,
    yol ARA-NOKTALARDAN BAĞIMSIZ (yalnız uç noktalara bağlı) — Chen kimliğinin bilinen sonucu."""
    arr = _clean_2d(path_2d, min_rows=2, min_cols=1)
    if arr is None:
        return np.array([], dtype=float)
    d = arr.shape[1]
    s1 = np.zeros(d)
    s2 = np.zeros((d, d))
    for k in range(1, arr.shape[0]):
        delta = arr[k] - arr[k - 1]
        d1 = delta
        d2 = 0.5 * np.outer(delta, delta)
        s1, s2 = _chen_combine(s1, s2, d1, d2)
    return np.concatenate([s1, s2.ravel()]).astype(float)


def signature_level3(path_2d: ArrayLike) -> np.ndarray:
    """`signature_level2` ile AYNI Chen-kimliği yöntemi, seviye-3'e genişletilmiş. Dönüş:
    [S1 (d) | S2.ravel() (d²) | S3.ravel() (d³)] tek boyutlu dizi (uzunluk d+d²+d³). <2 nokta/
    dejenere girdi → boş dizi (İSTİSNA YOK, yapısal dönüşüm)."""
    arr = _clean_2d(path_2d, min_rows=2, min_cols=1)
    if arr is None:
        return np.array([], dtype=float)
    d = arr.shape[1]
    s1 = np.zeros(d)
    s2 = np.zeros((d, d))
    s3 = np.zeros((d, d, d))
    for k in range(1, arr.shape[0]):
        delta = arr[k] - arr[k - 1]
        d1 = delta
        d2 = 0.5 * np.outer(delta, delta)
        d3 = np.einsum("i,j,k->ijk", delta, delta, delta) / 6.0
        s1, s2, s3 = _chen_combine3(s1, s2, s3, d1, d2, d3)
    return np.concatenate([s1, s2.ravel(), s3.ravel()]).astype(float)


# ===========================================================================
# Motif/anomali (basitleştirilmiş matrix profile / STOMP)
# ===========================================================================
def matrix_profile_light(x: ArrayLike, m: int) -> dict | None:
    """STOMP (Yeh ve ark. 2016) basitleştirmesi — O(n²) KABUL EDİLİR (n küçükken; büyük n için
    gerçek STOMP/MASS-FFT gerekir, burada YOK). Her uzunluk-`m` alt-diziyi z-normalize edip
    diğer tüm alt-dizilere Öklid uzaklığı hesaplar; kendi-eşleşme bölgesi (±m/2, "trivial match
    zone") DIŞLANIR. `profile[i]` = en-yakın (dışlama-bölgesi hariç) alt-dizinin uzaklığı.

    motif_idx: profili EN KÜÇÜK olan (i,j) çifti (en çok tekrar eden desen).
    discord_idx: profili EN BÜYÜK olan i (en anormal/tekrar-etmeyen alt-dizi).

    Kısa (<4 nokta)/m<2/m>n//2/NaN girdi → None.

    Dönüş: {'profile','profile_index','motif_idx','discord_idx'}."""
    arr = _clean_1d(x, min_len=4)
    if arr is None:
        return None
    try:
        mlen = int(m)
    except (TypeError, ValueError):
        return None
    n = arr.size
    if mlen < 2 or mlen > n // 2:
        return None
    num_sub = n - mlen + 1
    subs = np.empty((num_sub, mlen))
    for i in range(num_sub):
        seg = arr[i:i + mlen]
        mu = float(np.mean(seg))
        sd = float(np.std(seg))
        subs[i] = (seg - mu) / sd if sd > 0.0 else np.zeros(mlen)

    profile = np.full(num_sub, np.inf)
    profile_index = np.full(num_sub, -1, dtype=int)
    exclusion = max(1, mlen // 2)
    for i in range(num_sub):
        dists = np.sqrt(np.sum((subs - subs[i]) ** 2, axis=1))
        lo_ex = max(0, i - exclusion)
        hi_ex = min(num_sub, i + exclusion + 1)
        dists[lo_ex:hi_ex] = np.inf
        j = int(np.argmin(dists))
        profile[i] = float(dists[j])
        profile_index[i] = j if math.isfinite(dists[j]) else -1

    finite_mask = np.isfinite(profile)
    if not np.any(finite_mask):
        return None
    motif_i = int(np.where(finite_mask)[0][np.argmin(profile[finite_mask])])
    motif_j = int(profile_index[motif_i])
    discord_idx = int(np.where(finite_mask)[0][np.argmax(profile[finite_mask])])
    motif_idx = (motif_i, motif_j) if motif_j >= 0 else None
    return {"profile": profile, "profile_index": profile_index,
            "motif_idx": motif_idx, "discord_idx": discord_idx}


# ===========================================================================
# Numerik altyapı (QP, adaptif integral, log-sum-exp, FFT konvolüsyon)
# ===========================================================================
def qp_solve(
    p: ArrayLike, q: ArrayLike, a_eq: ArrayLike | None = None, b_eq: ArrayLike | None = None,
    lb: ArrayLike | float | None = None, ub: ArrayLike | float | None = None,
) -> np.ndarray | None:
    """min 0.5·xᵀPx + qᵀx  k.k.  A_eq·x=b_eq, lb<=x<=ub — scipy.optimize.minimize (SLSQP)
    sarmalı. P simetrik-OLMAYABİLİR girdide 0.5(P+Pᵀ) ile simetrize edilir (savunmacı — QP
    tanımı simetrik P varsayar). `lb`/`ub` skaler verilirse TÜM boyutlara yayılır (broadcast).

    Boyut uyuşmazlığı/dönüştürülemez P,q/A_eq sütun-sayısı≠n/yakınsamama → None.

    NOT: portfolio.py (WP-P, Dalga-2 kardeşi) bu fonksiyonu FONKSİYON-İÇİNDE lazy import
    edip yoksa kendi SLSQP yedeğine düşer — Dalga-2 içi bağımlılık kasıtlı gevşek (paralel
    ajan teslimini bloklamaz, bkz. AI_V3_DERIN_BEYIN_SPEC.md §3.10)."""
    p_arr = _clean_2d(p, min_rows=1, min_cols=1)
    if p_arr is None or p_arr.shape[0] != p_arr.shape[1]:
        return None
    n = p_arr.shape[0]
    q_arr = _clean_1d(q, min_len=n)
    if q_arr is None or q_arr.size != n:
        return None
    p_sym = 0.5 * (p_arr + p_arr.T)

    a_eq_arr = None
    b_eq_arr = None
    if a_eq is not None:
        try:
            a_eq_arr = np.asarray(a_eq, dtype=float)
        except (TypeError, ValueError):
            return None
        if a_eq_arr.ndim == 1:
            a_eq_arr = a_eq_arr.reshape(1, -1)
        if a_eq_arr.ndim != 2 or a_eq_arr.shape[1] != n or not np.all(np.isfinite(a_eq_arr)):
            return None
        if b_eq is None:
            return None
        b_eq_arr = _clean_1d(b_eq, min_len=a_eq_arr.shape[0])
        if b_eq_arr is None or b_eq_arr.size != a_eq_arr.shape[0]:
            return None

    def _bound_vec(v: ArrayLike | float | None, default: float) -> np.ndarray | None:
        if v is None:
            return np.full(n, default)
        try:
            va = np.asarray(v, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if va.size == 1:
            return np.full(n, float(va[0]))
        return va if va.size == n else None

    lb_arr = _bound_vec(lb, -np.inf)
    ub_arr = _bound_vec(ub, np.inf)
    if lb_arr is None or ub_arr is None:
        return None
    bounds = list(zip(lb_arr.tolist(), ub_arr.tolist(), strict=True))

    def obj(xv: np.ndarray) -> float:
        return float(0.5 * xv @ p_sym @ xv + q_arr @ xv)

    def jac(xv: np.ndarray) -> np.ndarray:
        return p_sym @ xv + q_arr

    constraints = []
    if a_eq_arr is not None:
        constraints.append({"type": "eq", "fun": lambda xv: a_eq_arr @ xv - b_eq_arr,
                             "jac": lambda xv: a_eq_arr})

    finite_lb = np.where(np.isfinite(lb_arr), lb_arr, -1e6)
    finite_ub = np.where(np.isfinite(ub_arr), ub_arr, 1e6)
    x0 = np.clip(np.zeros(n), finite_lb, finite_ub)
    try:
        res = minimize(obj, x0, jac=jac, method="SLSQP", bounds=bounds, constraints=constraints,
                        options={"maxiter": 300, "ftol": 1e-14})
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None
    if not res.success or not np.all(np.isfinite(res.x)):
        return None
    return np.asarray(res.x, dtype=float)


def adaptive_quad(f: object, a: float, b: float) -> float | None:
    """Uyarlamalı sayısal integral (scipy.integrate.quad sarmalı). `f` çağrılabilir olmalı
    (keyfi kullanıcı fonksiyonu — İÇİNDE ne fırlatırsa fırlatsın YAKALANIR, İSTİSNA DIŞARI
    SIZMAZ). Geçersiz sınır (a/b sonlu-değil)/çağrılamaz `f`/yakınsamama → None."""
    af = _as_float(a)
    bf = _as_float(b)
    if af is None or bf is None or not callable(f):
        return None
    try:
        val, _err = _quad(f, af, bf)
    except Exception:
        return None
    return float(val) if math.isfinite(val) else None


def logsumexp_stable(x: ArrayLike) -> float:
    """Sayısal kararlı log-sum-exp: log(sum(exp(x))), max-çıkarma hilesiyle (taşma/düşük-akış
    korumalı). Boş girdi → -inf; NaN içeren girdi → nan; (NaN yokken) +inf içeren girdi → +inf;
    tamamı -inf (hiç sonlu değer yok, NaN da yok) → -inf (matematiksel olarak doğru: her terim
    exp(-inf)=0, log(0)=-inf). Hiçbir girdi İSTİSNA fırlatmaz."""
    arr = _as_1d(x)
    if arr.size == 0:
        return float("-inf")
    if np.any(np.isnan(arr)):
        return float("nan")
    if np.any(np.isposinf(arr)):
        return float("inf")
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return float("-inf")   # tamamı -inf (NaN/+inf yukarıda elendi)
    vals = arr[finite_mask]
    m = float(np.max(vals))
    s = float(np.sum(np.exp(vals - m)))
    if s <= 0.0:
        return float("-inf")
    result = m + math.log(s)
    return float(result) if math.isfinite(result) else result


def fft_convolve(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """FFT-tabanlı TAM (full) konvolüsyon — `np.convolve(a,b,mode='full')` ile SAYISAL OLARAK
    eşdeğer (bkz. test dosyası), büyük dizilerde O(n log n) (np.convolve'nin doğrudan O(n²)
    yerine). Boş girdi (a veya b) → boş dizi (İSTİSNA YOK, yapısal dönüşüm)."""
    arr_a = _as_1d(a)
    arr_b = _as_1d(b)
    if arr_a.size == 0 or arr_b.size == 0:
        return np.array([], dtype=float)
    n_out = arr_a.size + arr_b.size - 1
    n_fft = 1
    while n_fft < n_out:
        n_fft *= 2
    fa = np.fft.rfft(arr_a, n_fft)
    fb = np.fft.rfft(arr_b, n_fft)
    conv = np.fft.irfft(fa * fb, n_fft)[:n_out]
    return conv.astype(float)
