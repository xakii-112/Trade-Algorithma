"""Çok değişkenli/uzun hafızalı oynaklık ve Haar MODWT. KARAR-DIŞI, saf, IO/ağ yok.

Kaynaklar: Engle (2002), JBES 20(3); Engle–Sheppard (2001), NBER w8554;
Baillie–Bollerslev–Mikkelsen (1996), Journal of Econometrics 74;
Percival–Walden (2000), Wavelet Methods for Time Series Analysis, Bölüm 5 ve 8.
Şartnamedeki denklemler uygulanır; parametre fitleri örneklem içidir, alfa iddiası yoktur.
Kısa/sabit/yanlış şekilli/sonlu olmayan girdi veya çözülemeyen hesap -> None.
Döngüsel MODWT ve örneklemden başlatılan varyans yolları nedensel canlı özellik değildir.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.signal import lfilter

from nihai.ai.quant_math import _garch11_sigma2_path, garch11_fit


def _clean(x: ArrayLike, ndim: int, minimum: int) -> np.ndarray | None:
    """Şekli koruyan katı dönüşüm; sabit sütun ve taşan varyans da ölçülemezdir."""
    try:
        if np.iscomplexobj(x):
            return None
        arr = np.asarray(x, dtype=float)
        if arr.ndim != ndim or arr.shape[0] < minimum or not np.all(np.isfinite(arr)):
            return None
        if ndim == 2 and arr.shape[1] < 2:
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            var = np.var(arr, axis=0)
        if np.any(var <= 0) or not np.all(np.isfinite(var)):
            return None
        return arr
    except (TypeError, ValueError, OverflowError):
        return None


def _positive_int(value: Any) -> bool:
    """Kesirli ufukları sessizce kesmez; bool bir ufuk değildir."""
    return bool(isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value >= 1)


def _corr(q: np.ndarray) -> np.ndarray:
    """Son iki eksendeki kovaryans matrislerini korelasyona dönüştürür."""
    sd = np.sqrt(np.diagonal(q, axis1=-2, axis2=-1))
    return q / sd[..., :, None] / sd[..., None, :]


def _dcc_q(outer: np.ndarray, a: float, b: float, qbar: np.ndarray) -> np.ndarray:
    """Q[0]=Qbar; lfilter yalnız önceki şoku kullanır, son satır Q[T+1]'dir."""
    forcing = (1.0 - a - b) * qbar + a * outer
    tail, _ = lfilter([1.0], [1.0, -b], forcing, axis=0, zi=(b * qbar)[None])
    return np.concatenate((qbar[None], tail), axis=0)


def _dcc_loglik(eps: np.ndarray, r: np.ndarray) -> float:
    """Cholesky ile Gauss korelasyon olabilirliği; tekil matris büyük ceza alır."""
    try:
        chol = np.linalg.cholesky(r)
        z = np.linalg.solve(chol, eps[..., None])[..., 0]
        logdet = 2.0 * np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
        value = -0.5 * float(np.sum(logdet + np.sum(z * z - eps * eps, axis=1)))
        return value if np.isfinite(value) else -1e100
    except (ValueError, np.linalg.LinAlgError):
        return -1e100


def dcc_correlation_path(
    std_resid: ArrayLike, a: float, b: float, Qbar: ArrayLike | None = None,
) -> dict | None:
    """Sabit parametreli DCC yolu: R (T,N,N), Qbar, Q_last, Q_next, causal.

    Qbar dışarıdan sabitse R[t] yalnız eps[:t] kullanır (`causal=True`);
    Qbar verilmezse tüm örneklemden hesaplanır ve `causal=False` taşınır.
    Bayrak parametrelerin geçmişten tahmin edildiğini garanti etmez.
    """
    eps = _clean(std_resid, 2, 2)
    if eps is None:
        return None
    try:
        a, b = float(a), float(b)
        if not np.isfinite([a, b]).all() or a < 0 or b < 0 or a + b > 0.999 + 1e-12:
            return None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            outer = eps[:, :, None] * eps[:, None, :]
            if Qbar is not None and np.iscomplexobj(Qbar):
                return None
            qb = outer.mean(axis=0) if Qbar is None else np.asarray(Qbar, dtype=float)
            if qb.shape != (eps.shape[1], eps.shape[1]):
                return None
            if not np.isfinite(qb).all() or not np.allclose(qb, qb.T, atol=1e-12, rtol=0):
                return None
            np.linalg.cholesky(qb)
            q = _dcc_q(outer, a, b, qb)
            r = _corr(q[:-1])
            if not np.isfinite(q).all() or not np.isfinite(r).all():
                return None
            np.linalg.cholesky(r)
        return {"R": r, "Qbar": qb.copy(), "Q_last": q[-2], "Q_next": q[-1],
                "causal": Qbar is not None}
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return None


def dcc_garch_fit(returns: ArrayLike, *, starts: ArrayLike | None = None) -> dict | None:
    """Engle iki-adımlı QMLE; T>=250, N>=2. Sabit başlangıç ızgarası, CCC kıyası.

    Korelasyon LR değeri tanısaldır; a=0 altında b tanımlanamadığından sıradan
    chi-kare(2) kritik değeri kalibre edilmiş bir p-değeri değildir.
    Q_next öngörü için saklanır; tüm fit ve hedef Qbar örneklem içidir.
    """
    arr = _clean(returns, 2, 250)
    if arr is None:
        return None
    if starts is None:
        grid = np.array([(a, b) for a in (0.01, 0.03, 0.06) for b in (0.85, 0.92, 0.96)
                         if a + b <= 0.999])
    else:
        try:
            if np.iscomplexobj(starts):
                return None
            grid = np.asarray(starts, dtype=float)
            if (grid.ndim != 2 or grid.shape[1] != 2 or len(grid) == 0
                    or not np.isfinite(grid).all() or np.any(grid < 0)
                    or np.any(grid.sum(axis=1) > 0.999)):
                return None
        except (TypeError, ValueError, OverflowError):
            return None
    fits = []
    eps = np.empty_like(arr)
    for i in range(arr.shape[1]):
        fit = garch11_fit(arr[:, i])
        if fit is None:
            return None
        fits.append(fit)
        resid = arr[:, i] - fit["mean"]
        s2 = _garch11_sigma2_path(fit["omega"], fit["alpha"], fit["beta"],
                                resid * resid, float(np.var(resid)))
        eps[:, i] = resid / np.sqrt(s2)
    baseline = dcc_correlation_path(eps, 0.0, 0.0)
    if baseline is None:
        return None
    qb = baseline["Qbar"]
    outer = eps[:, :, None] * eps[:, None, :]
    lccc = _dcc_loglik(eps, baseline["R"])
    if lccc <= -1e99:
        return None

    def objective(params: np.ndarray) -> float:
        a, b = params
        if a < 0 or b < 0 or a + b > 0.999 + 1e-12:
            return 1e100
        r = _corr(_dcc_q(outer, a, b, qb)[:-1])
        return -_dcc_loglik(eps, r) / len(eps)

    best_x, best_fun = np.zeros(2), -lccc / len(eps)
    for start in grid:
        try:
            result = minimize(objective, start, method="SLSQP", jac="3-point",
                              bounds=[(0.0, 0.999)] * 2,
                              constraints=[{"type": "ineq", "fun": lambda p: 0.999 - p.sum()}],
                              options={"maxiter": 250, "ftol": 1e-12})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if result.success and np.isfinite(result.fun) and result.fun < best_fun:
            best_x, best_fun = result.x, float(result.fun)
    a, b = map(float, best_x)
    path = dcc_correlation_path(eps, a, b, qb)
    if path is None:
        return None
    n, dim = arr.shape
    pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    loglik = _dcc_loglik(eps, path["R"])
    return {"a": a, "b": b, "persistence": a + b, "loglik_corr": loglik,
            "loglik_ccc": lccc, "lr_vs_ccc": 2 * (loglik - lccc), "n": n, "N": dim,
            "Qbar": qb, "R_last": path["R"][-1], "Q_next": path["Q_next"],
            "corr_path": path["R"][:, *np.triu_indices(dim, 1)], "pairs": pairs,
            "univariate": fits, "sinirda": {"a_sifir": a < 1e-7, "persistence_ust": a + b > 0.9989999},
            "ic_orneklem": True, "causal": False}


def dcc_forecast(fit: dict, horizon: int) -> dict | None:
    """Q_next'ten Engle–Sheppard yaklaşımı; k=1 tam, k>1 yaklaşık korelasyon."""
    if not isinstance(fit, dict) or not _positive_int(horizon):
        return None
    try:
        a, b = float(fit["a"]), float(fit["b"])
        if np.iscomplexobj(fit["Qbar"]) or np.iscomplexobj(fit["Q_next"]):
            return None
        qb, qnext = np.asarray(fit["Qbar"], dtype=float), np.asarray(fit["Q_next"], dtype=float)
        if (not np.isfinite([a, b]).all() or a < 0 or b < 0 or a + b > 0.999 + 1e-12
                or qb.ndim != 2 or qb.shape[0] < 2 or qb.shape[0] != qb.shape[1]
                or qnext.shape != qb.shape):
            return None
        for q in (qb, qnext):
            if not np.isfinite(q).all() or not np.allclose(q, q.T, atol=1e-12, rtol=0):
                return None
            np.linalg.cholesky(q)
        weight = ((a + b) ** np.arange(horizon))[:, None, None]
        r = _corr((1 - weight) * qb + weight * qnext)
        return {"R": r, "yaklasim": "engle_sheppard"}
    except (KeyError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return None


def _figarch_weights(d: float, beta: float, phi: float, trunc: int) -> np.ndarray:
    """(1-beta L)-(1-phi L)(1-L)^d polinomunun c[1:trunc+1] katsayıları."""
    k = np.arange(1, trunc + 1, dtype=float)
    pi = np.concatenate(([1.0], np.cumprod((k - 1 - d) / k)))
    c = -(pi[1:] - phi * pi[:-1])
    c[0] -= beta
    return c


def _figarch_sigma2(
    eps2: np.ndarray, omega: float, d: float, beta: float, phi: float, trunc: int, var0: float,
) -> np.ndarray:
    """Sabit parametreli yol; sigma2[0]=var0, t<0 için eps2[t]=var0.

    Doğrudan konvolüsyon geçmiş şokları toplar; eksik geçmiş ters kümülatif c toplamıdır.
    FFT kullanılmaz: tüm pencerenin yuvarlama hatası geçmişi değiştirmemelidir.
    d=0'da eski GARCH yolu birebir kullanılır (alfa=phi-beta, eski geçmiş katkısı sıfır).
    İç yardımcı doğrulanmış parametre/kare dizisi alır; negatif varyansı kırpmaz.
    """
    if d == 0:
        return _garch11_sigma2_path(omega, phi - beta, beta, eps2, var0)
    c = _figarch_weights(d, beta, phi, trunc)
    shock = np.convolve(eps2, c)[:len(eps2) - 1]
    missing = np.concatenate((np.cumsum(c[:0:-1])[::-1], [0.0]))
    m = min(len(shock), trunc)
    shock[:m] += var0 * missing[:m]
    tail, _ = lfilter([1.0], [1.0, -beta], omega + shock, zi=[beta * var0])
    return np.concatenate(([var0], tail))


def _figarch_constraints(p: np.ndarray) -> np.ndarray:
    """Şartnamedeki BBM yeterli eşitsizliklerinin pozitif payları."""
    _, d, beta, phi = p
    return np.array([phi - beta + d, (2 - d) / 3 - phi,
                     beta * (phi - beta + d) - d * (phi - (1 - d) / 2)])


def figarch_fit(returns: ArrayLike, *, trunc: int = 1000) -> dict | None:
    """FIGARCH(1,d,1), Gauss QMLE; T>=250, sabit başlangıç ızgarası + GARCH(1,1) eşlemesi.

    Birim varyansla optimize edilir; omega ve loglik Jacobian ile geri ölçeklenir.
    Örneklem öncesi kare artık ve sigma2[0] örneklem varyansıdır. Kesme sayısı taşınır.
    d=0 GARCH özdeşliğidir; LR için sınırda chi-kare karışımı gerekir (yalnız tanısal).

    BBM KOŞULLARI DAYATILMAZ, ÖLÇÜLÜR (Tur-79, BULGU-79-3): BBM (1996) eşitsizlikleri koşullu
    varyansın pozitifliği için YETERLİ koşullardır, gerekli değil; ilk sürüm onları sert kısıt
    olarak dayatıyordu ve φ<=(2-d)/3 d=0'da GARCH kalıcılığını α+β<=2/3'e sınırlıyordu. ÖLÇÜLDÜ:
    α=0,08·β=0,90 GARCH sürecinde (n=500) FIGARCH, İÇİNDE BARINDIRDIĞI GARCH(1,1)'den DÜŞÜK
    log-olabilirlik veriyordu (LR = −0,78 < 0 — iç içe modelde imkânsız). Artık yalnız yol
    pozitifliği (her σ²>0) ve kutu sınırları uygulanır; GARCH(1,1) fitinin d=0 eşlemesi ek
    başlangıç noktasıdır; BBM koşulunun sağlanıp sağlanmadığı `kisit_ihlali` ile RAPORLANIR
    (ihlal = örneklem dışında pozitiflik garantisi yok). Tur-76 Feller ilkesinin aynısı.
    Başarısız yakınsama da `converged=False` ile taşınır; sonlu aday atılmaz.
    """
    arr = _clean(returns, 1, 250)
    if arr is None or not _positive_int(trunc):
        return None
    resid = arr - np.mean(arr)
    var0 = float(np.var(resid))
    eps2 = (resid / np.sqrt(var0)) ** 2
    garch = garch11_fit(arr)

    def objective(p: np.ndarray) -> float:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            s2 = _figarch_sigma2(eps2, p[0], p[1], p[2], p[3], trunc, 1.0)
            if np.any(s2 <= 0) or not np.isfinite(s2).all():
                return 1e100
            value = 0.5 * float(np.mean(np.log(2 * np.pi * s2) + eps2 / s2))
        return value if np.isfinite(value) else 1e100

    starts = []
    for d, beta, phi in ((0.2, 0.1, 0.1), (0.4, 0.3, 0.1), (0.6, 0.4, 0.2),
                         (0.8, 0.5, 0.1), (0.05, 0.3, 0.4), (0.0, 0.2, 0.5)):
        csum = float(_figarch_weights(d, beta, phi, trunc).sum())
        starts.append(np.array([max(1 - beta - csum, 0.01), d, beta, phi]))
    if garch is not None:
        # d=0 eşlemesi: α=φ−β → φ=α+β; ω birim-varyans ölçeğinde. İç içe modelin optimumu
        # aramaya DAHİL edilir → FIGARCH log-olabilirliği GARCH(1,1)'in altına düşemez.
        starts.append(np.array([max(garch["omega"] / var0, 1e-12), 0.0, garch["beta"],
                                min(garch["alpha"] + garch["beta"], 0.999)]))
    best = None
    for start in starts:
        try:
            result = minimize(objective, start, method="SLSQP", jac="3-point",
                              bounds=[(1e-12, None), (0.0, 1.0), (0.0, 0.999), (-0.999, 0.999)],
                              options={"maxiter": 400, "ftol": 1e-13})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if np.isfinite(result.fun) and result.fun < 1e99 and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        return None
    omega_s, d, beta, phi = map(float, best.x)
    sigma2 = _figarch_sigma2(eps2, omega_s, d, beta, phi, trunc, 1.0) * var0
    if not np.isfinite(sigma2).all() or np.any(sigma2 <= 0):
        return None
    loglik = -float(best.fun) * len(arr) - 0.5 * len(arr) * np.log(var0)
    violation = bool(omega_s <= 0 or not 0 <= d <= 1 or not 0 <= beta <= 1
                     or np.any(_figarch_constraints(best.x) < -1e-8))
    return {"omega": omega_s * var0, "d": d, "beta": beta, "phi": phi, "loglik": float(loglik),
            "aic": float(8 - 2 * loglik), "bic": float(4 * np.log(len(arr)) - 2 * loglik),
            "n": len(arr), "trunc": int(trunc), "sinirda": {"d_sifir": d < 1e-7, "d_bir": d > 1 - 1e-7},
            "kisit_ihlali": violation, "lambda_toplam": float(_figarch_weights(d, beta, phi, trunc).sum()),
            "last_sigma2": float(sigma2[-1]), "uncond_var": None,
            "uncond_neden": "FIGARCH'ta d>0 iken koşulsuz varyans tanımsız (λ(1)=1)",
            "garch11": {"loglik": None if garch is None else garch["loglik"],
                        "lr": None if garch is None else float(2 * (loglik - garch["loglik"]))},
            "converged": bool(best.success), "ic_orneklem": True, "causal": False}


def modwt_haar(x: ArrayLike, *, levels: int | None = None) -> dict | None:
    """Döngüsel Haar MODWT: W(J,N), V(N), levels, causal=False; N>=8.

    Filtre N'den uzunsa sınırdan arınmış katsayı kalmaz; varyans tüketicisi None taşır.
    """
    arr = _clean(x, 1, 8)
    if arr is None:
        return None
    jmax = max(1, int(np.log2(len(arr))) - 1) if levels is None else levels
    if not _positive_int(jmax):
        return None
    v, w = arr.copy(), []
    for j in range(jmax):
        shifted = np.roll(v, pow(2, j, len(arr)))
        w.append(0.5 * v - 0.5 * shifted)
        v = 0.5 * v + 0.5 * shifted
    return {"W": np.asarray(w), "V": v, "levels": int(jmax), "causal": False}


def modwt_mra(x: ArrayLike, *, levels: int | None = None) -> dict | None:
    """Sıfır fazlı ters MODWT: D(J,N), S(N), levels; X=sum(D)+S, causal=False."""
    wt = modwt_haar(x, levels=levels)
    if wt is None:
        return None
    details = []
    for j, w in enumerate(wt["W"]):
        d = 0.5 * (w - np.roll(w, -pow(2, j, len(w))))
        for k in range(j - 1, -1, -1):
            d = 0.5 * (d + np.roll(d, -pow(2, k, len(d))))
        details.append(d)
    smooth = wt["V"].copy()
    for j in range(wt["levels"] - 1, -1, -1):
        smooth = 0.5 * (smooth + np.roll(smooth, -pow(2, j, len(smooth))))
    return {"D": np.asarray(details), "S": smooth, "levels": wt["levels"], "causal": False}


def wavelet_variance(x: ArrayLike, *, levels: int | None = None) -> dict | None:
    """Sınır katsayıları atılmış Haar varyansı ve log2 ölçek eğimi.

    Enerji payının paydası ham X enerjisidir; düzgün bileşen nedeniyle paylar toplamı
    birden küçük olabilir. M<1 -> nu2=None. Sıfır varyans log eğimine alınmaz.
    """
    wt = modwt_haar(x, levels=levels)
    if wt is None:
        return None
    w, v = wt["W"], wt["V"]
    n, jmax = len(v), wt["levels"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        biased = np.mean(w * w, axis=1)
        energy = float(biased.sum() + np.mean(v * v))
        if not np.isfinite(energy) or energy <= 0:
            return None
        counts = [n - 2 ** (j + 1) + 1 for j in range(jmax)]
        nu2 = [float(np.mean(w[j, n - m:] ** 2)) if m > 0 else None for j, m in enumerate(counts)]
    valid = [(j, np.log2(value)) for j, value in enumerate(nu2)
             if value is not None and value > 0 and np.isfinite(value)]
    slope = None
    if len(valid) >= 3:
        scales, values = np.asarray(valid).T
        slope = float(np.polyfit(scales, values, 1)[0])
    return {"olcek": [2 ** j for j in range(jmax)], "nu2": nu2, "nu2_yanli": biased.tolist(),
            "enerji_payi": (biased / energy).tolist(), "M": counts, "log_egim": slope, "causal": False}
