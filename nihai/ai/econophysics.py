"""
econophysics.py — EKONOFİZİK, KAOS ve SİNYAL AYRIŞTIRMA (Tur-75). KARAR-DIŞI.

`quant_math` ölçekleme/entropi/MFDFA ailesini, `advanced_math` sıçrama/rejim/imza
ailesini taşıyor; dış geliştirme listesinin BÖLÜM 5/6'sında ikisinde de olmayan sekiz
madde kaldı. Hepsi FİYAT SERİSİNDEN hesaplanabilir — uydu/patent/13F gerektiren
maddeler burada YOKTUR (gerekçeler `docs/TUR75_RAPORU`).

| Madde | Fonksiyon |
|---|---|
| 086 LPPLS balon tekilliği | `lppls_fit` |
| 089 Hilbert-Huang (EMD + anlık frekans) | `emd` · `hilbert_spectrum` |
| 090 Transfer entropisi | `transfer_entropy` |
| 093 Takens gömmesi + Lyapunov üssü | `takens_embedding` · `lyapunov_rosenstein` |
| 095 CUSUM yapısal kırılma filtresi | `cusum_filter` |
| 098 Perkolasyon / dev bileşen | `percolation_profile` |
| 100 Dalgacık gürültü ayıklama | `wavelet_denoise` |
| 101 Diebold-Yılmaz taşma endeksi | `spillover_index` |

ANAYASA: KARAR-DIŞI. Listenin maddeleri kural öneriyor ("t_c yaklaşırken long'ları
kapat" — Madde 086); o kurallar BURAYA YAZILMAZ (ACI DERS #3/#5).

## Üç dürüstlük sınırı (gizlenmiyor)

1. **LPPLS bir TAHMİN DEĞİL bir UYUMDUR.** `t_c` çıkması "çöküş tarihi bulundu" demek
   değildir; literatürde bile LPPLS'in tahmin gücü tartışmalıdır ve `t_c` küçük veri
   değişimlerinde büyük oynar. Bu yüzden çıktı `t_c_std` (çoklu-başlangıç dağılımı) ve
   `filter_pass` (Filimonov-Sornette kısıtları) taşır; kısıtları geçmeyen uyum
   RAPORLANIR ama `filter_pass=False` ile.
2. **Lyapunov üssü pozitif çıkması KAOS KANITI DEĞİLDİR.** Rosenstein tahmincisi
   renkli gürültüde de pozitif değer verir; ayırt edici sınav bir **karıştırılmış
   (surrogate) seriyle** kıyastır ve fonksiyon onu `lambda_surrogate` olarak KENDİSİ
   koşar (Tur-60: yalnız reddeden değil, POZİTİF KONTROLÜ olan kapı).
3. **`spillover_index` YÖNLÜDÜR ama NEDENSELLİK DEĞİLDİR.** Genelleştirilmiş FEVD
   (Pesaran-Shin) sıralamadan bağımsızdır, ancak "A'dan B'ye taşma" ifadesi öngörü
   hatası varyansı payıdır — nedensel bir iddia değildir.

Sözleşme `quant_math` ile aynı: kısa/NaN/dönüştürülemez girdi → None, istisna YOK.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal as sp_signal
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

# Filimonov-Sornette (2013) kabul aralıkları — ön-kayıtlı, TARANMADI.
LPPLS_M_RANGE = (0.1, 0.9)
LPPLS_OMEGA_RANGE = (6.0, 13.0)
# EMD durdurma ölçütü (Huang 1998, Cauchy tipi): ardışık elemelerin bağıl farkı.
EMD_SD_TOL = 0.2
EMD_MAX_SIFT = 50
# Haar dalgacığında evrensel eşik ölçeği: σ̂ = MAD(d1)/0.6745 (Donoho-Johnstone).
MAD_TO_SIGMA = 0.6745


def _clean_1d(x: ArrayLike, min_len: int = 1) -> np.ndarray | None:
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < min_len or not np.all(np.isfinite(arr)):
        return None
    return arr


def _clean_2d(x: ArrayLike, min_rows: int = 2, min_cols: int = 2) -> np.ndarray | None:
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
# Madde 086 — LPPLS (Sornette)
# ---------------------------------------------------------------------------
def _lppls_lineer(t: np.ndarray, y: np.ndarray, tc: float, m: float,
                  omega: float) -> tuple[np.ndarray, float] | None:
    """(tc, m, ω) sabitken A, B, C1, C2 KAPALI FORMLA (EKK) çözülür.

    LPPLS bu üç parametrede doğrusaldır; nonlineer aramayı 7 boyuttan 3'e indiren
    şey budur (Filimonov-Sornette 2013'ün asıl katkısı).
    """
    dt = tc - t
    if np.any(dt <= 0.0):
        return None
    f = dt ** m
    lg = np.log(dt)
    g = f * np.cos(omega * lg)
    h = f * np.sin(omega * lg)
    X = np.column_stack([np.ones_like(t), f, g, h])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)):
        return None
    return beta, float(np.sum((y - X @ beta) ** 2))


def lppls_fit(log_prices: ArrayLike, n_starts: int = 12,
              seed: int = 42) -> dict | None:
    """Log-periyodik güç yasası tekilliği uyumu (Madde 086).

        ln p(t) = A + B(t_c−t)^m + C(t_c−t)^m·cos(ω·ln(t_c−t) − φ)

    Zaman ekseni 0..n−1 BAR indeksidir; `t_c` de bar cinsindendir (`t_c − (n−1)` =
    bugünden kaç bar sonra). Çoklu rastgele başlangıçtan en iyi SSE seçilir ve
    başlangıçlar arası `t_c` dağılımı `t_c_std` olarak RAPORLANIR — dar bir SSE
    minimumu ile geniş bir `t_c` dağılımı BİRLİKTE görülebilir ve o durumda tarih
    okunmamalıdır (modül docstring'i §1).

    `filter_pass`: 0,1 ≤ m ≤ 0,9 VE 6 ≤ ω ≤ 13 VE B < 0 (süper-üstel yükseliş).
    <30 gözlem / sabit seri / hiçbir başlangıç yakınsamazsa → None.
    """
    y = _clean_1d(log_prices, 30)
    if y is None or float(np.std(y)) <= 0.0:
        return None
    n = y.size
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    try:
        ns = max(1, int(n_starts))
    except (TypeError, ValueError):
        return None

    def _sse(p: np.ndarray) -> float:
        out = _lppls_lineer(t, y, float(p[0]), float(p[1]), float(p[2]))
        return float(out[1]) if out is not None else 1e18

    sinirlar = [(float(n), float(n) * 1.5), LPPLS_M_RANGE, LPPLS_OMEGA_RANGE]
    en_iyi: tuple[float, np.ndarray] | None = None
    tc_ler: list[float] = []
    for _ in range(ns):
        x0 = np.array([
            n * (1.0 + 0.4 * rng.random()),
            rng.uniform(*LPPLS_M_RANGE),
            rng.uniform(*LPPLS_OMEGA_RANGE),
        ])
        try:
            res = minimize(_sse, x0, method="L-BFGS-B", bounds=sinirlar)
        except (ValueError, FloatingPointError):        # pragma: no cover - fail-soft
            continue
        if not res.success or not np.isfinite(res.fun) or res.fun >= 1e17:
            continue
        tc_ler.append(float(res.x[0]))
        if en_iyi is None or res.fun < en_iyi[0]:
            en_iyi = (float(res.fun), np.asarray(res.x, dtype=float))
    if en_iyi is None:
        return None
    sse, (tc, m, omega) = en_iyi[0], en_iyi[1]
    lin = _lppls_lineer(t, y, tc, m, omega)
    if lin is None:
        return None
    beta = lin[0]
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "t_c": float(tc),
        "t_c_bars_ahead": float(tc - (n - 1)),
        "t_c_std": float(np.std(tc_ler, ddof=1)) if len(tc_ler) > 1 else None,
        "m": float(m),
        "omega": float(omega),
        "A": float(beta[0]), "B": float(beta[1]),
        "C1": float(beta[2]), "C2": float(beta[3]),
        "sse": sse,
        "r2": float(1.0 - sse / ss_tot) if ss_tot > 0 else None,
        "n_converged": len(tc_ler),
        "n": int(n),
        "filter_pass": bool(LPPLS_M_RANGE[0] <= m <= LPPLS_M_RANGE[1]
                            and LPPLS_OMEGA_RANGE[0] <= omega <= LPPLS_OMEGA_RANGE[1]
                            and beta[1] < 0.0),
    }


# ---------------------------------------------------------------------------
# Madde 089 — EMD / Hilbert-Huang
# ---------------------------------------------------------------------------
def _zarf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Yerel ekstremumlardan kübik spline üst/alt zarf. <2 tepe veya <2 dip → None."""
    n = x.size
    idx = np.arange(n)
    tepe = sp_signal.argrelextrema(x, np.greater)[0]
    dip = sp_signal.argrelextrema(x, np.less)[0]
    if tepe.size < 2 or dip.size < 2:
        return None
    # Uç noktalar zarfa DAHİL edilir (aksi hâlde spline uçlarda kontrolsüz salınır).
    tepe = np.unique(np.r_[0, tepe, n - 1])
    dip = np.unique(np.r_[0, dip, n - 1])
    try:
        ust = CubicSpline(tepe, x[tepe])(idx)
        alt = CubicSpline(dip, x[dip])(idx)
    except ValueError:                                   # pragma: no cover - fail-soft
        return None
    return ust, alt


def emd(x: ArrayLike, max_imf: int = 8) -> dict | None:
    """Ampirik Kip Ayrıştırması: seri → IMF'ler + artık (Madde 089).

    Huang (1998) eleme (sifting) algoritması; durdurma ölçütü Cauchy tipi bağıl fark
    (`EMD_SD_TOL`). **Durağanlık VARSAYMAZ** — Fourier/dalgacıktan farkı budur.

    `x = Σ imfs + residue` özdeşliği YAPISALDIR (her IMF çıkarılarak ilerlenir), bu
    yüzden test onu makine hassasiyetinde arar. Ekstremumu yetmeyen seride (monoton
    trend) IMF ÜRETİLMEZ ve seri tamamen artığa düşer — bu bir hata değil, EMD'nin
    doğru cevabıdır.
    """
    s = _clean_1d(x, 8)
    if s is None:
        return None
    try:
        k_max = max(1, int(max_imf))
    except (TypeError, ValueError):
        return None
    artik = s.astype(float).copy()
    imfler: list[np.ndarray] = []
    for _ in range(k_max):
        h = artik.copy()
        basarili = False
        for _ in range(EMD_MAX_SIFT):
            z = _zarf(h)
            if z is None:
                break
            ort = (z[0] + z[1]) / 2.0
            yeni = h - ort
            payda = float(np.sum(h * h))
            sd = float(np.sum((h - yeni) ** 2)) / payda if payda > 0 else 0.0
            h = yeni
            basarili = True
            if sd < EMD_SD_TOL:
                break
        if not basarili:
            break
        imfler.append(h)
        artik = artik - h
        if float(np.ptp(artik)) <= 0.0:
            break
    return {
        "imfs": imfler,
        "residue": artik,
        "n_imf": len(imfler),
        "n": int(s.size),
    }


def hilbert_spectrum(imf: ArrayLike) -> dict | None:
    """Bir IMF'in analitik sinyalinden anlık genlik/faz/frekans (Madde 089).

    `inst_freq` faz farkının 2π'ye bölümüdür (BAR başına çevrim); n−1 uzunluktadır
    ve dizinin ilk elemanı bar 0→1 geçişine aittir. Faz `np.unwrap` ile sürekli
    kılınır, aksi hâlde her ±π sıçraması sahte bir frekans zirvesi üretir.
    """
    v = _clean_1d(imf, 4)
    if v is None:
        return None
    z = sp_signal.hilbert(v)
    genlik = np.abs(z)
    faz = np.unwrap(np.angle(z))
    frek = np.diff(faz) / (2.0 * math.pi)
    return {
        "amplitude": genlik,
        "phase": faz,
        "inst_freq": frek,
        "mean_freq": float(np.mean(frek)),
        "n": int(v.size),
    }


# ---------------------------------------------------------------------------
# Madde 090 — Transfer entropisi
# ---------------------------------------------------------------------------
def _ortak_hist(kodlar: np.ndarray, boyutlar: tuple[int, ...]) -> np.ndarray:
    """Çok boyutlu sayım → olasılık tablosu."""
    duz = np.ravel_multi_index(kodlar, boyutlar)
    say = np.bincount(duz, minlength=int(np.prod(boyutlar))).astype(float)
    return (say / say.sum()).reshape(boyutlar)


def transfer_entropy(source: ArrayLike, target: ArrayLike, bins: int = 6,
                     lag: int = 1) -> dict | None:
    """T_{X→Y}: X'in geçmişinin Y'nin geleceğine kattığı BİLGİ (Madde 090, nat).

        T_{X→Y} = Σ p(y_{t+1}, y_t, x_t) · log[ p(y_{t+1}|y_t,x_t) / p(y_{t+1}|y_t) ]

    Kutulama EŞİT GENİŞLİKTEDİR ve `bins` çağıranındır; TE kutu sayısına duyarlıdır,
    bu yüzden İKİ YÖN AYNI kutulamayla hesaplanır ve `net` (T_{X→Y} − T_{Y→X}) asıl
    okunacak büyüklüktür — ortak kutulama yanlılığı farkta büyük ölçüde sadeleşir.

    **Sonlu örneklem yanlılığı SIFIR DEĞİLDİR:** bağımsız serilerde bile TE pozitif
    çıkar. `te_shuffled` kaynağı karıştırıp AYNI hesabı koşar = gürültü tabanı;
    `te_xy` ondan büyük değilse bilgi akışı İDDİA EDİLEMEZ (Tur-60 pozitif-kontrol
    disiplini).
    """
    x = _clean_1d(source, 20)
    y = _clean_1d(target, 20)
    if x is None or y is None or x.size != y.size:
        return None
    try:
        nb, lg = int(bins), int(lag)
    except (TypeError, ValueError):
        return None
    if nb < 2 or lg < 1 or x.size <= lg + 10:
        return None
    if float(np.ptp(x)) <= 0.0 or float(np.ptp(y)) <= 0.0:
        return None

    def _kodla(v: np.ndarray) -> np.ndarray:
        kenar = np.linspace(v.min(), v.max(), nb + 1)
        return np.clip(np.searchsorted(kenar, v, side="right") - 1, 0, nb - 1)

    kx, ky = _kodla(x), _kodla(y)

    def _te(a: np.ndarray, b: np.ndarray) -> float:
        """T_{a→b}: a'nın geçmişi b'nin geleceğine ne katıyor."""
        b1, b0, a0 = b[lg:], b[:-lg], a[:-lg]
        p3 = _ortak_hist(np.vstack([b1, b0, a0]), (nb, nb, nb))
        p_b0a0 = p3.sum(axis=0)
        p_b1b0 = p3.sum(axis=2)
        p_b0 = p3.sum(axis=(0, 2))
        toplam = 0.0
        nz = np.nonzero(p3)
        for i, j, k in zip(*nz, strict=True):
            pay = p3[i, j, k] * p_b0[j]
            payda = p_b0a0[j, k] * p_b1b0[i, j]
            if pay > 0.0 and payda > 0.0:
                toplam += p3[i, j, k] * math.log(pay / payda)
        return float(toplam)

    rng = np.random.default_rng(0)
    karisik = _te(rng.permutation(kx), ky)
    return {
        "te_xy": _te(kx, ky),
        "te_yx": _te(ky, kx),
        "net": _te(kx, ky) - _te(ky, kx),
        "te_shuffled": karisik,
        "bins": nb, "lag": lg, "n": int(x.size),
    }


# ---------------------------------------------------------------------------
# Madde 093 — Takens gömmesi + Lyapunov
# ---------------------------------------------------------------------------
def takens_embedding(x: ArrayLike, dim: int = 3, tau: int = 1) -> np.ndarray | None:
    """Gecikmeli koordinat matrisi [x(t), x(t−τ), …, x(t−(m−1)τ)] (Madde 093)."""
    v = _clean_1d(x, 3)
    if v is None:
        return None
    try:
        m, t = int(dim), int(tau)
    except (TypeError, ValueError):
        return None
    if m < 1 or t < 1:
        return None
    span = (m - 1) * t
    if v.size <= span:
        return None
    n = v.size - span
    return np.column_stack([v[span - i * t: span - i * t + n] for i in range(m)])


def lyapunov_rosenstein(x: ArrayLike, dim: int = 3, tau: int = 1,
                        max_t: int = 20, theiler: int | None = None,
                        seed: int = 0) -> dict | None:
    """En büyük Lyapunov üssü (Rosenstein 1993) + KARIŞTIRILMIŞ kontrol (Madde 093).

    Yakın komşuların ortalama log ayrışma eğrisinin eğimi. Zamansal olarak komşu
    noktalar dışlanır (Theiler penceresi; varsayılan `(m−1)τ`) — dışlanmazsa
    "komşu" aslında AYNI yörüngenin bir sonraki noktasıdır ve λ yapay çıkar.

    `lambda_surrogate`: seri KARIŞTIRILIP aynı hesap koşulur. λ > 0 tek başına kaos
    kanıtı DEĞİLDİR (modül docstring'i §2); ayırt edici sayı `lambda_ratio`dur.
    """
    v = _clean_1d(x, 50)
    if v is None or float(np.std(v)) <= 0.0:
        return None
    try:
        m, t, mt = int(dim), int(tau), int(max_t)
    except (TypeError, ValueError):
        return None
    if m < 2 or t < 1 or mt < 3:
        return None

    def _lam(seri: np.ndarray) -> tuple[float, np.ndarray, float | None] | None:
        emb = takens_embedding(seri, m, t)
        if emb is None or emb.shape[0] < mt + 10:
            return None
        n = emb.shape[0]
        w = theiler if theiler is not None else (m - 1) * t
        d = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=2)
        idx = np.arange(n)
        d[np.abs(idx[:, None] - idx[None, :]) <= w] = np.inf
        komsu = np.argmin(d, axis=1)
        gecerli = np.isfinite(d[idx, komsu])
        egri = np.full(mt, np.nan)
        for k in range(mt):
            i = idx[gecerli]
            j = komsu[gecerli]
            ok = (i + k < n) & (j + k < n)
            if not np.any(ok):
                break
            ayrim = np.linalg.norm(emb[i[ok] + k] - emb[j[ok] + k], axis=1)
            ayrim = ayrim[ayrim > 0]
            if ayrim.size == 0:
                break
            egri[k] = float(np.mean(np.log(ayrim)))
        ok = np.isfinite(egri)
        if ok.sum() < 3:
            return None
        kk = np.flatnonzero(ok).astype(float)
        yy = egri[np.flatnonzero(ok)]
        egim, kesim = np.polyfit(kk, yy, 1)
        art = yy - (egim * kk + kesim)
        ss_tot = float(np.sum((yy - yy.mean()) ** 2))
        r2 = float(1.0 - np.sum(art ** 2) / ss_tot) if ss_tot > 0 else None
        return float(egim), egri, r2

    ana = _lam(v)
    if ana is None:
        return None
    rng = np.random.default_rng(seed)
    kar = _lam(rng.permutation(v))
    lam_s = kar[0] if kar is not None else None
    r2_s = kar[2] if kar is not None else None
    return {
        "lambda": ana[0],
        "divergence_curve": ana[1],
        # AYIRT EDİCİ BÜYÜKLÜK BUDUR (ölçüldü, aşağıya bak): λ'nın kendisi değil,
        # ayrışma eğrisinin UZUN bir aralıkta DOĞRUSAL kalıp kalmadığı.
        "r2": ana[2],
        "lambda_surrogate": lam_s,
        "r2_surrogate": r2_s,
        "lambda_ratio": (ana[0] / lam_s) if (lam_s is not None and lam_s != 0.0) else None,
        "predictability_horizon": (1.0 / ana[0]) if ana[0] > 0 else None,
        "dim": m, "tau": t, "n": int(v.size),
    }


# ---------------------------------------------------------------------------
# Madde 095 — CUSUM yapısal kırılma filtresi
# ---------------------------------------------------------------------------
def cusum_filter(x: ArrayLike, threshold: float | None = None) -> dict | None:
    """Simetrik CUSUM olay filtresi (López de Prado Ch.2 / Madde 095).

        S⁺ = max(0, S⁺ + Δ)   ·   S⁻ = min(0, S⁻ + Δ)
        |S| > h olduğunda OLAY kaydedilir ve İLGİLİ taraf sıfırlanır.

    `threshold=None` ise h = Δ'nın standart sapması (veriye göre ölçekli, ön-kayıtlı
    çarpan YOK). Çıktı `events` **bar indeksleridir** ve amaç bar örneklemesidir:
    sabit zaman aralığı yerine BİLGİ geldiğinde örneklemek.

    Sıfırlama yalnız TETİKLEYEN tarafa uygulanır (López de Prado'nun kanonik biçimi).
    Pratikte iki biriktiricinin aynı anda sıfırdan uzak olması `max(0,·)`/`min(0,·)`
    kırpması yüzünden ZORDUR; yani asimetri burada bir DAVRANIŞ FARKI değil, kanonik
    biçime sadakattir — abartılı bir iddia kurulmuyor.
    """
    v = _clean_1d(x, 3)
    if v is None:
        return None
    d = np.diff(v)
    if d.size == 0:
        return None
    if threshold is None:
        h = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    else:
        try:
            h = float(threshold)
        except (TypeError, ValueError):
            return None
    if not np.isfinite(h) or h <= 0.0:
        return None
    sp = sn = 0.0
    olaylar: list[int] = []
    yonler: list[int] = []
    for i, dd in enumerate(d, start=1):
        sp = max(0.0, sp + float(dd))
        sn = min(0.0, sn + float(dd))
        if sp > h:
            sp = 0.0
            olaylar.append(i)
            yonler.append(1)
        elif sn < -h:
            sn = 0.0
            olaylar.append(i)
            yonler.append(-1)
    return {
        "events": np.asarray(olaylar, dtype=int),
        "directions": np.asarray(yonler, dtype=int),
        "threshold": h,
        "n_events": len(olaylar),
        "event_rate": float(len(olaylar)) / float(d.size),
        "n": int(v.size),
    }


# ---------------------------------------------------------------------------
# Madde 098 — Perkolasyon / dev bileşen
# ---------------------------------------------------------------------------
def _bilesenler(adj: np.ndarray) -> np.ndarray:
    """Birleştir-bul ile bağlı bileşen etiketleri (scipy'siz, N küçük)."""
    n = adj.shape[0]
    kok = np.arange(n)

    def bul(a: int) -> int:
        while kok[a] != a:
            kok[a] = kok[kok[a]]
            a = int(kok[a])
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                ri, rj = bul(i), bul(j)
                if ri != rj:
                    kok[ri] = rj
    return np.array([bul(i) for i in range(n)])


def percolation_profile(corr: ArrayLike,
                        thresholds: ArrayLike | None = None) -> dict | None:
    """Korelasyon eşiği düşürüldükçe DEV BİLEŞENİN büyümesi (Madde 098).

    Her eşik için |ρ| ≥ eşik olan çiftler kenar sayılır ve en büyük bağlı bileşenin
    PAYI ölçülür. `critical_threshold`, payın en HIZLI sıçradığı eşiktir (ardışık
    farkın maksimumu) — perkolasyon geçişinin sonlu-örneklem vekili.

    **Sıçrama noktası bir FAZ GEÇİŞİ İDDİASI DEĞİLDİR:** N varlıkla sonlu bir ağda
    kritik nokta keskin değildir; `jump` (sıçramanın büyüklüğü) birlikte raporlanır
    ve küçükse "geçiş bulundu" denemez.
    """
    c = _clean_2d(corr, 2, 2)
    if c is None or c.shape[0] != c.shape[1]:
        return None
    n = c.shape[0]
    a = np.abs(c)
    np.fill_diagonal(a, 0.0)
    if thresholds is None:
        esikler = np.linspace(0.95, 0.05, 19)
    else:
        e = _clean_1d(thresholds, 2)
        if e is None:
            return None
        esikler = np.sort(e)[::-1]
    paylar = np.empty(esikler.size)
    for k, e in enumerate(esikler):
        etiket = _bilesenler(a >= e)
        paylar[k] = float(np.bincount(etiket).max()) / n
    fark = np.diff(paylar)
    kidx = int(np.argmax(fark)) if fark.size else 0
    return {
        "thresholds": esikler,
        "giant_share": paylar,
        "critical_threshold": float(esikler[kidx]),
        "jump": float(fark[kidx]) if fark.size else 0.0,
        "n_assets": n,
    }


# ---------------------------------------------------------------------------
# Madde 100 — Dalgacık gürültü ayıklama
# ---------------------------------------------------------------------------
def wavelet_denoise(x: ArrayLike, levels: int = 3,
                    mode: str = "soft") -> dict | None:
    """Haar DWT + evrensel eşik + yeniden kurma (Madde 100).

    Eşik Donoho-Johnstone evrensel eşiğidir: λ = σ̂·√(2 ln n), σ̂ = MAD(d₁)/0.6745
    (en ince detay seviyesi gürültünün en saf tahminidir). `mode="soft"` büzer
    (sürekli), `"hard"` keser (süreksiz ama enerji korur).

    **Haar seçimi bir SADELEŞTİRME DEĞİL bir KARARDIR:** `quant_math.haar_dwt_vol`
    zaten Haar kullanıyor; ikinci bir dalgacık ailesi (Daubechies-4) eklemek `pywt`
    bağımlılığı isterdi ve depo yeni bağımlılık almıyor. Haar süreksizlikleri iyi,
    düzgün eğrileri kötü temsil eder — sonuç `wavelet="haar"` ile taşınır.

    Seri uzunluğu 2^levels'in katı değilse SON kısım kırpılır ve `n_used` raporlanır
    (sessizce sıfırla doldurmak kenarda sahte bir süreksizlik yaratırdı).

    **`kept_coeffs` OKUNMADAN çıktı okunmaz.** Evrensel eşik gürültülü serilerde
    agresiftir; sıfır kalan katsayı, çıktının bir ayıklama değil L. seviye YAKLAŞIMI
    olduğu anlamına gelir (Tur-23'ün "kapı PASS dese bile n_kept oku" kuralının
    sinyal-işleme karşılığı).
    """
    v = _clean_1d(x, 4)
    if v is None:
        return None
    try:
        L = int(levels)
    except (TypeError, ValueError):
        return None
    if L < 1 or mode not in ("soft", "hard"):
        return None
    blok = 2 ** L
    n_used = (v.size // blok) * blok
    if n_used < blok * 2:
        return None
    s = v[:n_used].astype(float).copy()

    yaklasim = s.copy()
    detaylar: list[np.ndarray] = []
    for _ in range(L):
        cift = yaklasim[0::2]
        tek = yaklasim[1::2]
        detaylar.append((cift - tek) / math.sqrt(2.0))
        yaklasim = (cift + tek) / math.sqrt(2.0)

    d1 = detaylar[0]
    mad = float(np.median(np.abs(d1 - np.median(d1))))
    sigma = mad / MAD_TO_SIGMA
    lam = sigma * math.sqrt(2.0 * math.log(max(n_used, 2)))
    esikli = []
    kalan = 0
    toplam_kats = 0
    for d in detaylar:
        toplam_kats += int(d.size)
        kalan += int(np.count_nonzero(np.abs(d) > lam))
        if mode == "soft":
            esikli.append(np.sign(d) * np.clip(np.abs(d) - lam, 0.0, None))
        else:
            esikli.append(np.where(np.abs(d) > lam, d, 0.0))

    rec = yaklasim
    for d in reversed(esikli):
        cift = (rec + d) / math.sqrt(2.0)
        tek = (rec - d) / math.sqrt(2.0)
        rec = np.empty(cift.size * 2, dtype=float)
        rec[0::2] = cift
        rec[1::2] = tek
    return {
        "denoised": rec,
        "residual": s - rec,
        "sigma": sigma,
        "threshold": lam,
        "levels": L,
        "wavelet": "haar",
        "mode": mode,
        "n_used": int(n_used),
        "n": int(v.size),
        # ÖLÇÜM, GÖRÜNÜRLÜK İÇİN: evrensel eşik gürültülü finans serilerinde çoğu
        # zaman TÜM detay katsayılarını sıfırlar; o durumda çıktı bir "gürültüsü
        # ayıklanmış seri" değil, düpedüz L. seviye YAKLAŞIMIDIR (2^L örneklik
        # parçalar hâlinde sabit). `kept_coeffs=0` bunu açıkça söyler — eşiğin ne
        # yaptığını gizleyen bir ayıklayıcı, ölçüm yokluğunu ölçüm gibi sunar.
        "kept_coeffs": kalan,
        "total_coeffs": toplam_kats,
        "noise_removed_share": (float(np.var(s - rec) / np.var(s))
                                if float(np.var(s)) > 0 else None),
    }


# ---------------------------------------------------------------------------
# Madde 101 — Diebold-Yılmaz taşma endeksi
# ---------------------------------------------------------------------------
def _var_ols(y: np.ndarray, p: int) -> tuple[list[np.ndarray], np.ndarray] | None:
    """VAR(p) EKK: Φ₁..Φ_p katsayıları + artık kovaryansı Σ."""
    T, k = y.shape
    if T <= p * k + k + 1:
        return None
    Y = y[p:]
    X = np.column_stack([np.ones(T - p)] + [y[p - i - 1: T - i - 1] for i in range(p)])
    try:
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)):
        return None
    resid = Y - X @ beta
    sigma = resid.T @ resid / float(T - p - X.shape[1])
    phi = [np.asarray(beta[1 + i * k: 1 + (i + 1) * k].T, dtype=float) for i in range(p)]
    return phi, np.asarray(sigma, dtype=float)


def spillover_index(returns: ArrayLike, horizon: int = 10,
                    lags: int = 2) -> dict | None:
    """Diebold-Yılmaz genelleştirilmiş taşma endeksi (Madde 101).

    Pesaran-Shin genelleştirilmiş FEVD (Cholesky DEĞİL) — değişken SIRASINDAN
    bağımsızdır, o yüzden "hangi sektör önce" tartışması sonucu değiştirmez.

        θ_ij(H) = σ_jj⁻¹ · Σ_h (e_i' A_h Σ e_j)² / Σ_h (e_i' A_h Σ A_h' e_i)

    Satırlar 1'e normalize edilir (genelleştirilmiş FEVD'de ham satır toplamı 1
    DEĞİLDİR — normalize etmeyen bir uygulama endeksi sistematik olarak kaydırır).

    Dönen: `total` (toplam taşma %) · `to_others`/`from_others`/`net` (varlık başına) ·
    `table` (yönlü matris). Girdi (T × N) GETİRİ matrisidir.
    """
    y = _clean_2d(returns, 20, 2)
    if y is None:
        return None
    try:
        H, p = int(horizon), int(lags)
    except (TypeError, ValueError):
        return None
    if H < 1 or p < 1:
        return None
    k = y.shape[1]
    var = _var_ols(y, p)
    if var is None:
        return None
    phi, sigma = var
    if not np.all(np.isfinite(sigma)) or np.any(np.diag(sigma) <= 0):
        return None

    # MA(∞) katsayıları: A₀ = I, A_h = Σ_{i=1..min(h,p)} Φ_i A_{h−i}
    A = [np.eye(k)]
    for h in range(1, H):
        acc = np.zeros((k, k))
        for i in range(1, min(h, p) + 1):
            acc += phi[i - 1] @ A[h - i]
        A.append(acc)

    pay = np.zeros((k, k))
    payda = np.zeros(k)
    for Ah in A:
        AS = Ah @ sigma
        pay += AS ** 2
        payda += np.diag(AS @ Ah.T)
    sjj = np.diag(sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = (pay / sjj[None, :]) / payda[:, None]
    if not np.all(np.isfinite(theta)):
        return None
    theta = theta / theta.sum(axis=1, keepdims=True)

    kosegen_disi = float(theta.sum() - np.trace(theta))
    to_others = theta.sum(axis=0) - np.diag(theta)      # j'den DİĞERLERİNE
    from_others = theta.sum(axis=1) - np.diag(theta)    # i'ye DİĞERLERİNDEN
    return {
        "total": 100.0 * kosegen_disi / float(k),
        "table": theta,
        "to_others": 100.0 * to_others / float(k),
        "from_others": 100.0 * from_others / float(k),
        "net": 100.0 * (to_others - from_others) / float(k),
        "horizon": H, "lags": p, "n_assets": k, "n": int(y.shape[0]),
    }
