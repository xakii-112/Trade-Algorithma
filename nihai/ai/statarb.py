"""
statarb.py — Öz-portföy/kalıntı analizi ve çok değişkenli kointegrasyon. KARAR-DIŞI.

Saf fonksiyonlar: yan etkisiz, IO/ağ YOK; yalnız numpy/scipy ve opsiyonel-guard'lı
statsmodels kullanılır. Üretim sinyaline/eşiğe/META_GATE'e bağlanmaz; salt-okunur rapor
katmanıdır. Skorlar ve rank testleri örneklem içi istatistiklerdir, kazanç kanıtı değildir.
Tüm fit'ler verilen pencerenin tamamını kullanır; geçmiş değerlendirmede pencere karar
barında bitmelidir. Bir pencerenin ECT serisi, yürüyen örneklem-dışı tahmin serisi değildir.

Savunmacı sözleşme (`quant_math.py` ile aynı): kısa/dönüştürülemez/NaN/Inf girdi veya
tanımsız/sayısal olarak çözülemeyen fit → None; eksik gözlem sessizce atılmaz. Getiri
panelinde satır=zaman barı, sütun=varlık; OU zamanı dt=1 bardır, yıllıklandırma yapılmaz.

Kaynaklar:
  - Avellaneda ve Lee (2010), Statistical arbitrage in the US equities market,
    Quantitative Finance 10(7), 761–782, özellikle öz-portföyler, §4 ve Ek A.
    https://doi.org/10.1080/14697680903124632
  - Johansen (1988), Statistical analysis of cointegration vectors,
    https://doi.org/10.1016/0165-1889(88)90041-3
  - Johansen (1991), Estimation and Hypothesis Testing of Cointegration Vectors
    in Gaussian Vector Autoregressive Models, https://doi.org/10.2307/2938278
  - Osterwald-Lenum (1992), A Note with Quantiles of the Asymptotic Distribution
    of the Maximum Likelihood Cointegration Rank Test Statistics, 54(3), 461–472.
    https://doi.org/10.1111/j.1468-0084.1992.tb00013.x
    Tablo aktarımı: https://warwick.ac.uk/fac/soc/economics/staff/jsmith/manual/statisticaltables.pdf
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg

try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen as _coint_johansen
except ImportError:  # pragma: no cover - opsiyonel bağımlılık, yerel indirgeme kullanılabilir.
    _coint_johansen = None


# Osterwald-Lenum (1992): satırlar k-r=1..5, sütunlar %90/%95/%99.
# det_order=0: kısıtsız sabit (durum III); det_order=1: kısıtsız sabit+trend (durum V).
# Bunlar statsmodels'in MacKinnon-Haug-Michelis tablosu DEĞİLDİR. İstatistik kaynağı
# değişse de bu modülün kritik değerleri aynı tablodan gelir; tablolar dışa kopyalanır.
_TRACE_CRIT = {
    0: (
        (2.69, 3.76, 6.65),
        (13.33, 15.41, 20.04),
        (26.79, 29.68, 35.65),
        (43.95, 47.21, 54.46),
        (64.84, 68.52, 76.07),
    ),
    1: (
        (2.57, 3.74, 6.40),
        (16.06, 18.17, 23.46),
        (31.42, 34.55, 40.49),
        (50.74, 54.64, 61.24),
        (73.40, 77.74, 85.78),
    ),
}
_MAX_CRIT = {
    0: (
        (2.69, 3.76, 6.65),
        (12.07, 14.07, 18.63),
        (18.60, 20.97, 25.52),
        (24.73, 27.07, 32.24),
        (30.90, 33.46, 38.77),
    ),
    1: (
        (2.57, 3.74, 6.40),
        (14.84, 16.87, 21.47),
        (21.53, 23.78, 28.83),
        (27.76, 30.33, 35.68),
        (33.74, 36.41, 41.58),
    ),
}


# ---------------------------------------------------------------------------
# Girdi temizleme yardımcıları (modül-içi; dışa açık API değil)
# ---------------------------------------------------------------------------
def _clean_1d(x: ArrayLike, min_len: int = 1) -> np.ndarray | None:
    """Katı 1B float dönüşüm: kısa/dönüştürülemez/sonlu-olmayan içerik → None.

    `quant_math._clean_1d` gibi düzleştirir; karmaşık sayının sanal kısmını atmaz.
    """
    try:
        raw = np.asarray(x)
        if np.iscomplexobj(raw):
            return None
        arr = np.asarray(raw, dtype=float).reshape(-1)
        if arr.size < min_len or not np.all(np.isfinite(arr)):
            return None
        return arr
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return None


def _clean_2d(x: ArrayLike, min_rows: int = 2, min_cols: int = 1) -> np.ndarray | None:
    """Katı 2B float dönüşüm: satır=zaman, sütun=varlık; kusurlu panel → None."""
    try:
        raw = np.asarray(x)
        if np.iscomplexobj(raw):
            return None
        arr = np.asarray(raw, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < min_rows or arr.shape[1] < min_cols:
            return None
        if not np.all(np.isfinite(arr)):
            return None
        return arr
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return None


def _int_arg(value: object, minimum: int) -> int | None:
    """Tamsayı parametresi; kesirleri kırpmaz, bool/NaN/Inf/geçersiz tip → None."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return None
    result = int(value)
    return result if result >= minimum else None


# ---------------------------------------------------------------------------
# Avellaneda–Lee öz-portföyleri, OU kalıntısı ve ham s-score
# ---------------------------------------------------------------------------
def eigenportfolio_returns(returns: ArrayLike, n_factors: int = 3) -> dict | None:
    """Korelasyonun en büyük özdeğerlerinden Avellaneda–Lee öz-portföyleri kurar.

    En az 60 bar ve 3 varlık gerekir. Örnek std (ddof=1) ile korelasyon hesaplanır;
    sıfır std → None. Ağırlık w_jk=v_jk/sigma_j, faktör getirisi F=R@w'dir: F için
    HAM getiri kullanılır, z-skor kullanılmaz. Ağırlıklar sermaye toplamına normalize
    edilmez. `explained`, korelasyon varyansındaki paydır, tahmin başarısı değildir.
    `n_factors` 1'den küçükse None, varlık sayısını aşarsa varlık sayısına kırpılır.
    """
    arr = _clean_2d(returns, min_rows=60, min_cols=3)
    nf = _int_arg(n_factors, 1)
    if arr is None or nf is None:
        return None
    n, n_assets = arr.shape
    nf = min(nf, n_assets)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            # Sabit ondalık sütunda ortalama yuvarlaması sahte pozitif std üretebilir.
            if np.any(np.ptp(arr, axis=0) == 0.0):
                return None
            centered = arr - arr.mean(axis=0)
            sigma = arr.std(axis=0, ddof=1)
            if np.any(sigma <= 0.0):
                return None
            standardized = centered / sigma
            corr = standardized.T @ standardized / (n - 1)
            eigenvalues, evec = np.linalg.eigh((corr + corr.T) / 2.0)
            if np.min(eigenvalues) < -1e-10 * n_assets:
                return None
            eigenvalues = np.maximum(eigenvalues, 0.0)
            order = np.argsort(eigenvalues)[::-1][:nf]
            selected = eigenvalues[order]
            vectors = evec[:, order]
            # Özvektör işareti belirsizdir; en büyük mutlak bileşeni pozitif seç.
            pivots = np.argmax(np.abs(vectors), axis=0)
            vectors = vectors * np.sign(vectors[pivots, np.arange(nf)])
            weights = vectors / sigma[:, None]
            factors = arr @ weights
            explained = selected / eigenvalues.sum()
            if not all(np.all(np.isfinite(x)) for x in (weights, factors, selected, explained)):
                return None
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError, FloatingPointError):
        return None
    return {
        "factor_returns": factors,
        "weights": weights,
        "eigenvalues": selected,
        "explained": explained,
        "n_factors": nf,
        "n": n,
        "n_assets": n_assets,
    }


def residual_ou_fit(residual_cumsum: ArrayLike) -> dict | None:
    """Kümülatif kalıntı X üzerinde X[t+1]=a+b*X[t]+zeta OLS/OU modeli kurar.

    Girdi kalıntı GETİRİSİ değil, onun kümülatif toplamıdır; en az 30 gözlem ister.
    dt=1 bar: kappa=-log(b), m=a/(1-b), sigma_eq=sqrt(var(zeta)/(1-b**2)).
    `sigma_resid` yenilik std'si (ddof=0), `sigma_eq` durağan dağılım std'sidir.
    `half_life=log(2)/kappa` bar cinsindedir; büyük değer yavaş dönüş demektir.
    `r2` bir adım OLS uyumudur. b<=0 veya b>=1, sabit seri ya da sıfır yenilik
    varyansı → None; böyle bir fit'ten ortalamaya dönen OU skoru üretilmez.
    """
    arr = _clean_1d(residual_cumsum, min_len=30)
    if arr is None:
        return None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            previous, following = arr[:-1], arr[1:]
            x_mean, y_mean = previous.mean(), following.mean()
            x, y = previous - x_mean, following - y_mean
            sxx, syy = x @ x, y @ y
            if sxx <= 0.0 or syy <= 0.0:
                return None
            b = float((x @ y) / sxx)
            if not 0.0 < b < 1.0:
                return None
            a = float(y_mean - b * x_mean)
            zeta = y - b * x
            variance = float(np.var(zeta))
            if variance <= 0.0:
                return None
            kappa = float(-np.log(b))
            m = a / (1.0 - b)
            sigma_eq = float(np.sqrt(variance / ((1.0 - b) * (1.0 + b))))
            sigma_resid = float(np.sqrt(variance))
            half_life = float(np.log(2.0) / kappa)
            r2 = float(1.0 - (zeta @ zeta) / syy)
            if not np.all(np.isfinite([a, b, kappa, m, sigma_eq, sigma_resid, half_life, r2])):
                return None
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError, FloatingPointError):
        return None
    return {
        "a": a, "b": b, "kappa": kappa, "m": m, "sigma_eq": sigma_eq,
        "half_life": half_life, "sigma_resid": sigma_resid, "r2": r2, "n": int(arr.size),
    }


def avellaneda_lee_scores(
    returns: ArrayLike,
    n_factors: int = 3,
    symbols: list[str] | None = None,
    esik: float = 1.25,
) -> dict | None:
    """Öz-portföy → sabitli çoklu OLS → kümülatif kalıntı → OU → ham s-score.

    s=(X_son-m)/sigma_eq. İŞARET: POZİTİF s, kalıntının aşırı yükseldiğini ve modelde
    ortalamaya dönüşün AŞAĞI beklendiğini belirtir: s>esik → SHORT, s<-esik → LONG;
    eşikler dahil kalan aralık → FLAT. Bunlar rapor etiketleridir, üretim sinyali değildir.
    `beta` listesi [sabit, F1 katsayısı, ...], `r2` getiri regresyonunun uyumudur.
    `reliable` yalnız OU tanımlılığı ve half_life<=n/2 koşuludur; kâr/güven testi değildir.
    Yavaş dönüşte skor/yön korunur fakat reliable=False olur. OU yoksa satır korunur,
    s_score=None ve yon=FLAT döner. Sayısal sıfır kalıntı da OU ölçümü sayılmaz.
    Eşik sonlu ve pozitif, verilmiş semboller varlık sayısı kadar string olmalıdır.
    `residual_cumsum` (n, n_assets) her varlığın X_j(t) YOLUdur — s-score bunun SON
    noktasının OU denge dağılımındaki z-skorudur; yol çizilmeden skor tek başına
    kalıntının nereden geldiğini söylemez.
    """
    arr = _clean_2d(returns, min_rows=60, min_cols=3)
    if arr is None:
        return None
    n, n_assets = arr.shape
    if symbols is not None and (
        not isinstance(symbols, list)
        or len(symbols) != n_assets
        or not all(isinstance(symbol, str) for symbol in symbols)
    ):
        return None
    try:
        if not np.isscalar(esik) or isinstance(esik, (bool, np.bool_)):
            return None
        threshold = float(esik)
        if not np.isfinite(threshold) or threshold <= 0.0:
            return None
        factors = eigenportfolio_returns(arr, n_factors)
        if factors is None:
            return None
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            design = np.column_stack((np.ones(n), factors["factor_returns"]))
            coefficients = np.linalg.lstsq(design, arr, rcond=None)[0]
            residuals = arr - design @ coefficients
            cumulative = np.cumsum(residuals, axis=0)
            sse = np.sum(residuals**2, axis=0)
            sst = np.sum((arr - arr.mean(axis=0))**2, axis=0)
            r2 = 1.0 - sse / sst
            if not all(np.all(np.isfinite(x)) for x in (coefficients, cumulative, r2)):
                return None
            assets: list[dict] = []
            for j in range(n_assets):
                x_last = float(cumulative[-1, j])
                # Tam faktör açıklaması sonrası yuvarlama artığını OU diye okumama.
                roundoff = np.finfo(float).eps * max(design.shape) * np.linalg.norm(arr[:, j])
                ou = (
                    None if np.linalg.norm(residuals[:, j]) <= roundoff
                    else residual_ou_fit(cumulative[:, j])
                )
                asset: dict = {
                    "symbol": None if symbols is None else symbols[j],
                    "index": j,
                    "beta": coefficients[:, j].tolist(),
                    "r2": float(r2[j]),
                    "s_score": None,
                    "yon": "FLAT",
                    "kappa": None,
                    "half_life": None,
                    "sigma_eq": None,
                    "m": None,
                    "x_last": x_last,
                    "reliable": False,
                    "neden": "ortalamaya donus yok (b araligi disinda)",
                }
                if ou is not None:
                    score = float((x_last - ou["m"]) / ou["sigma_eq"])
                    if not np.isfinite(score):
                        return None
                    reliable = bool(ou["half_life"] <= n / 2.0)
                    asset.update({
                        "s_score": score,
                        "yon": "SHORT" if score > threshold else "LONG" if score < -threshold else "FLAT",
                        "kappa": ou["kappa"],
                        "half_life": ou["half_life"],
                        "sigma_eq": ou["sigma_eq"],
                        "m": ou["m"],
                        "reliable": reliable,
                        "neden": None if reliable else "yarim omur n/2 bari asiyor; yavas donus/gurultu",
                    })
                assets.append(asset)
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError, FloatingPointError):
        return None
    return {"assets": assets, "factors": factors, "residual_cumsum": cumulative,
            "esik": threshold, "n": n}


# ---------------------------------------------------------------------------
# Johansen indirgemesi ve VECM (sabitli hata düzeltme)
# ---------------------------------------------------------------------------
def _lagged_system(panel: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Temiz panelden aynı t için dY[t], Y[t-1], [dY[t-1], ...] matrislerini kurar."""
    differences = np.diff(panel, axis=0)
    dy = differences[lags:]
    levels = panel[lags:-1]
    lagged = (
        np.column_stack([differences[lags - j:len(differences) - j] for j in range(1, lags + 1)])
        if lags else np.empty((len(dy), 0), dtype=float)
    )
    return dy, levels, lagged


def _johansen_covariances(
    panel: np.ndarray, det_order: int, lags: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Temiz panelin R0/R1 kalıntılarından S00, S01, S11 kovaryanslarını hesaplar.

    Her iki regresyonda AYNI gecikmeli farklar ve deterministik terimler kullanılır.
    det_order=-1 terimsiz, 0 sabitli, 1 sabit ve doğrusal trendli indirgemedir.
    """
    dy, levels, lagged = _lagged_system(panel, lags)
    regressors = lagged
    if det_order >= 0:
        regressors = np.column_stack((regressors, np.ones(len(dy))))
    if det_order == 1:
        regressors = np.column_stack((regressors, np.linspace(-1.0, 1.0, len(dy))))
    if regressors.shape[1]:
        r0 = dy - regressors @ np.linalg.lstsq(regressors, dy, rcond=None)[0]
        r1 = levels - regressors @ np.linalg.lstsq(regressors, levels, rcond=None)[0]
    else:
        r0, r1 = dy, levels
    count = len(dy)
    return r0.T @ r0 / count, r0.T @ r1 / count, r1.T @ r1 / count


def _sequential_rank(statistics: np.ndarray, critical: np.ndarray) -> int:
    """%95 testinde r=0'dan ilerler; ilk reddedilemeyen H0'da durur (sayım değildir)."""
    for rank, statistic in enumerate(statistics):
        if statistic <= critical[rank, 1]:
            return rank
    return len(statistics)


def johansen_test(panel: ArrayLike, det_order: int = 0, k_ar_diff: int = 1) -> dict | None:
    """Düzey panelinde Johansen trace ve max-eigen kointegrasyon rank testleri.

    En az 30 gözlem ve 2 seri gerekir; etkin örneklem gecikme/deterministik regresör
    sayısı + seri sayısından büyük olmalıdır. det_order=-1/0/1, fark gecikmesi >=0.
    Statsmodels varsa önce onun coint_johansen çıktısı kullanılır; yoksa R0/R1 üzerinde
    azaltılmış rank regresyonu ve simetrik genelleştirilmiş özdeğer problemi çözülür.
    `kaynak` istatistik hesaplayıcısını belirtir; iki yolun eşdeğerliği varsayılmaz.

    Kritik değerler HER İKİ yolda Osterwald-Lenum (1992), durum III (0) / V (1),
    k<=5 tablosudur; satır r=0..k-1, sütun %90/%95/%99. Tablo dışında kritik değerler
    ve rank'ler None, `neden` açıklamalıdır. Rank, %95'te ilk reddedilemeyen H0:rank<=r
    değeridir (max-eigen alternatifi r+1); tümü reddedilirse k'dır. Evec sütunları
    azalan özdeğer sırasındadır. `n` ham gözlem sayısı, istatistikte T=n-k_ar_diff-1.
    Tekil kovaryans/tanımsız logaritma → None; ridge ile yapay rank üretilmez.
    """
    arr = _clean_2d(panel, min_rows=30, min_cols=2)
    order = _int_arg(det_order, -1)
    lags = _int_arg(k_ar_diff, 0)
    if arr is None or order not in (-1, 0, 1) or lags is None:
        return None
    n, k = arr.shape
    if n - lags - 1 <= k * (lags + 1) + order + 1:
        return None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            if np.any(np.ptp(arr, axis=0) == 0.0):
                return None
            if _coint_johansen is not None:
                with warnings.catch_warnings():
                    # Tablo kapsamı zaten `neden` ile döner; konsola uyarı yazılmaz.
                    warnings.filterwarnings(
                        "ignore", message="Critical values are only available for time series",
                        category=UserWarning,
                    )
                    result = _coint_johansen(arr, order, lags)
                eigenvalues = np.asarray(result.eig)
                evec = np.asarray(result.evec)
                trace_stat = np.asarray(result.lr1)
                max_eig_stat = np.asarray(result.lr2)
                source = "statsmodels"
            else:
                s00, s01, s11 = _johansen_covariances(arr, order, lags)
                # solve(S00, S01) aynı indirgemedir; açık matris tersi oluşturulmaz.
                np.linalg.cholesky(s00)
                reduced = s01.T @ np.linalg.solve(s00, s01)
                eigenvalues, evec = linalg.eigh((reduced + reduced.T) / 2.0, s11)
                indices = np.argsort(eigenvalues)[::-1]
                eigenvalues, evec = eigenvalues[indices], evec[:, indices]
                if np.any(eigenvalues < -1e-10) or np.any(eigenvalues >= 1.0):
                    return None
                # Yuvarlama düzeyindeki negatif özdeğer sıfır kabul edilir.
                eigenvalues = np.maximum(eigenvalues, 0.0)
                max_eig_stat = -(n - lags - 1) * np.log1p(-eigenvalues)
                trace_stat = np.cumsum(max_eig_stat[::-1])[::-1]
                source = "yerel"
            arrays = (eigenvalues, evec, trace_stat, max_eig_stat)
            if any(np.iscomplexobj(x) or not np.all(np.isfinite(x)) for x in arrays):
                return None
            if np.any(eigenvalues < -1e-10) or np.any(eigenvalues >= 1.0):
                return None
            trace_crit = max_crit = None
            rank_trace = rank_maxeig = None
            reason = None
            if k > 5 or order not in _TRACE_CRIT:
                reason = "kritik deger tablosu yalniz k<=5 ve det_order=0/1 icin var; rank olculmedi"
            else:
                trace_crit = np.array(_TRACE_CRIT[order][:k][::-1], dtype=float)
                max_crit = np.array(_MAX_CRIT[order][:k][::-1], dtype=float)
                rank_trace = _sequential_rank(trace_stat, trace_crit)
                rank_maxeig = _sequential_rank(max_eig_stat, max_crit)
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError, FloatingPointError):
        return None
    return {
        "eigenvalues": eigenvalues,
        "trace_stat": trace_stat,
        "max_eig_stat": max_eig_stat,
        "trace_crit": trace_crit,
        "max_crit": max_crit,
        "rank_trace": rank_trace,
        "rank_maxeig": rank_maxeig,
        "evec": evec,
        "n": n,
        "k": k,
        "kaynak": source,
        "neden": reason,
    }


def vecm_fit(panel: ArrayLike, rank: int = 1, k_ar_diff: int = 1) -> dict | None:
    """Johansen vektörleriyle sabitli VECM: dY[t]=alpha*beta'Y[t-1]+Gamma*dY_gecikme+mu.

    Rank 1..k arasında KULLANICININ model seçimidir; test sonucu otomatik dayatılmaz.
    İlk rank özvektörünün uzayı korunarak beta' S11 beta=I normalizasyonu yapılır;
    S11, Y[t-1]'in gecikmeli farklar+sabite regresyonundan kalanların kovaryansıdır.
    Alpha (k,rank) OLS yüklemeleri; gamma (k,k*k_ar_diff) gecikme sıralı bloklardır,
    gecikme yoksa None. Sabit fit'e dahildir, gamma'ya dahil değildir.

    `ect_series` OLS ile hizalı Y[k_ar_diff:-1]@beta (n-k_ar_diff-1 satır), `ect_last`
    ise son gözlenen düzey panel[-1]@beta'dır. Her ECT'nin AR(1) yarı-ömrü bar cinsinden;
    b<=0/b>=1 veya yetersiz/tanımsız fit → ilgili listede None. Alpha'nın tek başına
    işareti dönüş kanıtı değildir; beta'nın işareti/birimiyle birlikte okunmalıdır.
    En az 30 gözlem/2 seri gerekir; tekil veya yetersiz regresyon → None.
    """
    arr = _clean_2d(panel, min_rows=30, min_cols=2)
    selected_rank = _int_arg(rank, 1)
    lags = _int_arg(k_ar_diff, 0)
    if arr is None or selected_rank is None or lags is None or selected_rank > arr.shape[1]:
        return None
    fit = johansen_test(arr, det_order=0, k_ar_diff=lags)
    if fit is None:
        return None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            _, _, s11 = _johansen_covariances(arr, 0, lags)
            basis = fit["evec"][:, :selected_rank]
            gram = basis.T @ s11 @ basis
            chol = np.linalg.cholesky((gram + gram.T) / 2.0)
            beta = linalg.solve_triangular(chol, basis.T, lower=True).T
            dy, levels, lagged = _lagged_system(arr, lags)
            ect_series = levels @ beta
            ect_last = arr[-1] @ beta
            design = np.column_stack((ect_series, lagged, np.ones(len(dy))))
            coefficients, _, design_rank, _ = np.linalg.lstsq(design, dy, rcond=None)
            if design_rank != design.shape[1]:
                return None
            alpha = coefficients[:selected_rank].T.copy()
            gamma = coefficients[selected_rank:-1].T.copy() if lags else None
            if not all(np.all(np.isfinite(x)) for x in (beta, coefficients, ect_series, ect_last)):
                return None
            half_life: list[float | None] = []
            for j in range(selected_rank):
                ou = residual_ou_fit(ect_series[:, j])
                half_life.append(None if ou is None else ou["half_life"])
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError, FloatingPointError):
        return None
    return {
        "beta": beta,
        "alpha": alpha,
        "gamma": gamma,
        "ect_last": ect_last,
        "ect_series": ect_series,
        "half_life": half_life,
        "rank": selected_rank,
        "n": arr.shape[0],
        "k": arr.shape[1],
    }


if __name__ == "__main__":
    # Tur-24 kuralı: yönlendirilmiş stdout Windows'ta cp1254'e düşer ve Türkçe/ok karakteri
    # basan ilk satır süreci ÇÖKERTİR (watchdog ve zamanlanmış görevler tam da böyle çağırır).
    # Tur-72 bu modülü eklerken guard'ı atlamıştı; Tur-73'te kapatıldı.
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()

    rng = np.random.default_rng(0)
    # Ortak getiri faktörü + durağan özgül düzeylerin farkları.
    sample_n = 600
    common = rng.normal(0.0, 0.015, size=sample_n)
    specific = np.zeros((sample_n + 1, 4))
    innovations = rng.normal(0.0, 0.003, size=(sample_n, 4))
    for t in range(sample_n):
        specific[t + 1] = 0.8 * specific[t] + innovations[t]
    synthetic_returns = common[:, None] * np.array([0.8, 1.0, 1.2, 1.4]) + np.diff(specific, axis=0)
    scores = avellaneda_lee_scores(synthetic_returns, n_factors=1)
    assert scores is not None
    assert all(asset["s_score"] is not None and np.isfinite(asset["s_score"]) for asset in scores["assets"])

    y1 = np.cumsum(rng.normal(size=800))
    stationary = np.zeros(800)
    noise = rng.normal(scale=0.5, size=800)
    for t in range(1, 800):
        stationary[t] = 0.6 * stationary[t - 1] + noise[t]
    cointegrated = np.column_stack((y1, 2.0 * y1 + stationary))
    johansen = johansen_test(cointegrated)
    assert johansen is not None and johansen["rank_trace"] >= 1
    vecm = vecm_fit(cointegrated)
    assert vecm is not None and np.all(np.isfinite(vecm["beta"]))
    assert vecm["half_life"][0] is not None

    for function in (eigenportfolio_returns, residual_ou_fit, avellaneda_lee_scores, johansen_test, vecm_fit):
        assert function([]) is None
        assert function(np.full((60, 4), np.nan)) is None
        assert function(np.full((60, 4), np.inf)) is None
        assert function(np.full((60, 4), 0.1)) is None
    print("statarb ozdenetim OK")
