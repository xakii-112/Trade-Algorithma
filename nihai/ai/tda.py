"""Topolojik veri analizi: Vietoris–Rips H0/H1 ve kalıcılık manzaraları. KARAR-DIŞI.

Saf, deterministik ölçüm; IO/ağ, emir, sinyal ve eşik üretimi yoktur. Çöküş öngörüsü
literatür iddiasıdır, bu sistemde ÖLÇÜLMEMİŞTİR. Getiriler hazır log-getiri panelidir.
Geçersiz veya sayısal olarak temsil edilemeyen girdi → None; boş diagram geçerlidir.

Kaynaklar: Edelsbrunner–Harer (2010), Computational Topology §VII;
Zomorodian–Carlsson (2005), Computing persistent homology; Bubenik (2015), JMLR 16,
Statistical topological data analysis using persistence landscapes;
Adamaszek–Adams (2017), The Vietoris–Rips complexes of a circle;
Gidea–Katz (2018), Physica A 491:820–834, Landscapes of crashes.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import pdist


def _array(value: ArrayLike) -> np.ndarray | None:
    """Karmaşık sayıları ve dönüştürülemeyen girdileri reddet."""
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            return None
        return np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return None


def _integer(value: object, minimum: int) -> bool:
    return bool(isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_)) and value >= minimum)


def _nonnegative(value: Any) -> float | None:
    try:
        if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
            return None
        number = float(value)
        return number if np.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _diagram(value: ArrayLike) -> tuple[np.ndarray, int] | None:
    """Sonlu doğum ve sonlu/+inf ölüm; ters çubuk geçersizdir."""
    arr = _array(value)
    if arr is None:
        return None
    if arr.shape == (0,):
        arr = arr.reshape(0, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    if (not np.all(np.isfinite(arr[:, 0])) or np.any(np.isnan(arr[:, 1]))
            or np.any(arr[:, 1] < arr[:, 0])):
        return None
    finite = np.isfinite(arr[:, 1])
    return arr[finite], int(np.sum(~finite))


# Tur-81 (BULGU-81-6): üçgen köşe listesi YALNIZ nokta sayısına bağlıdır; kayan pencere
# her adımda AYNI n ile çağırır ve liste her seferinde yeniden kuruluyordu (399×10 panelde
# 175 kez × C(50,3)=19.600 üçlü = ölçülen **0,749 sn**, toplam sürenin %11,6'sı).
# ponytail: tavan n<=64 (C(64,3)=41.664 üçlü ≈ 1 MB); üstü her çağrıda yeniden kurulur —
# C(200,3) 31,5 MB'dır ve 7/24 süreçte tutulamaz (Tur-80'in "defteri bellekte tutma" takası).
# maxsize=2 gerçek yükü karşılar: kayan pencere ile son tam-pencere çağrısı AYNI n'i kullanır.
_TRI_CACHE_MAX_N = 64


@lru_cache(maxsize=2)
def _ucgen_koseleri(n: int) -> np.ndarray:
    """C(n,3) üçlüsü — salt-okunur, paylaşılan (çağıran yalnız indeksler)."""
    v = np.array(list(combinations(range(n), 3)), dtype=int)
    v.flags.writeable = False
    return v


def rips_persistence(
    points: ArrayLike, *, max_edge: float | None = None, n_max: int = 200,
    omur_tol: float = 1e-12,
) -> dict[str, Any] | None:
    """Öklid Rips kompleksinin H0/H1 çubukları (Z/2, üçgenlere kadar).

    Kenarlar (uzunluk,i,j), üçgenler (çap,en büyük kenar indeksi,diğer indeksler)
    sırasındadır. Sınır sütunları Python int bit maskeleriyle XOR indirgenir.
    Sonlu H0/H1 çiftlerinde death-birth <= omur_tol*max(1,abs(death)) olanlar
    atılır ve n_sifir_omur'a sayılır; varsayılan 1e-12 sahte mikro-çubukları eler.
    omur_tol=0 yalnız tam sıfır ömrü eler. Kesilmiş komplekste HER bağlantılı
    bileşen sonsuz H0 verir; kesildi yalnız sonsuz H1 sayısıdır.
    Üçgen sayısı O(n³); n_max kaynak sınırıdır, büyük bulutlara ölçeklenmez.
    """
    arr = _array(points)
    tol = _nonnegative(omur_tol)
    cutoff = _nonnegative(max_edge) if max_edge is not None else None
    if (arr is None or arr.ndim != 2 or not _integer(n_max, 3)
            or not 3 <= len(arr) <= n_max or arr.shape[1] < 1
            or not np.all(np.isfinite(arr)) or tol is None
            or (max_edge is not None and cutoff is None)):
        return None
    n = len(arr)
    with np.errstate(over="ignore", invalid="ignore"):
        distances = pdist(arr)
    if not np.all(np.isfinite(distances)):
        return None
    ii, jj = np.triu_indices(n, 1)
    order = np.lexsort((jj, ii, distances))
    if cutoff is not None:
        order = order[distances[order] <= cutoff]
    lengths = distances[order]
    edges = list(zip(ii[order].tolist(), jj[order].tolist(), strict=True))
    edge_index = np.full((n, n), -1, dtype=int)
    edge_index[ii[order], jj[order]] = np.arange(len(order))
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    h0: list[tuple[float, float]] = []
    births: set[int] = set()
    n_zero = 0
    for e, (i, j) in enumerate(edges):
        a, b = root(i), root(j)
        if a == b:
            births.add(e)
        else:
            parent[b] = a
            death = float(lengths[e])
            if death <= tol * max(1.0, abs(death)):
                n_zero += 1
            else:
                h0.append((0.0, death))
    h0.extend((0.0, float("inf")) for _ in {root(i) for i in range(n)})

    vertices = (_ucgen_koseleri(n) if n <= _TRI_CACHE_MAX_N
                else np.array(list(combinations(range(n), 3)), dtype=int))
    triangles = np.column_stack((edge_index[vertices[:, 0], vertices[:, 1]],
                                 edge_index[vertices[:, 0], vertices[:, 2]],
                                 edge_index[vertices[:, 1], vertices[:, 2]]))
    triangles = np.sort(triangles[np.all(triangles >= 0, axis=1)], axis=1)
    diameters = lengths[triangles[:, 2]]
    tri_order = np.lexsort((triangles[:, 1], triangles[:, 0], triangles[:, 2], diameters))
    # Tur-81 (BULGU-81-6): indirgeme döngüsü bu modülün EN SICAK yeridir — 399×10 getiri
    # panelinde (pencere 50, adım 2) `bit_length` ve sözlük araması **41,3 milyon** kez
    # çağrılıyordu. Sözlük yerine KENAR-İNDEKSLİ liste: pivot her zaman geçerli bir kenar
    # indeksidir (sütunun en yüksek biti), yani aralık kontrolü gerekmez ve C düzeyi liste
    # indekslemesi hash + metot çağrısını tamamen kaldırır. `0` güvenli bir BOŞ nöbetçisidir:
    # saklanan sütun `(1<<a)|(1<<b)|(1<<c)` ya da onun XOR indirgemesidir ve saklandığı anda
    # `while column` koşulundan geçmiştir → ASLA 0 olamaz. Algoritma DEĞİŞMEZ; çıktı bit-denk.
    pivots: list[int] = [0] * len(edges)
    h1: list[tuple[float, float]] = []
    for a, b, c in triangles[tri_order].tolist():
        column = (1 << a) | (1 << b) | (1 << c)
        while column:
            pivot = column.bit_length() - 1
            previous = pivots[pivot]
            if not previous:
                pivots[pivot] = column
                births.remove(pivot)
                birth, death = float(lengths[pivot]), float(lengths[c])
                if death - birth <= tol * max(1.0, abs(death)):
                    n_zero += 1
                else:
                    h1.append((birth, death))
                break
            column ^= previous
    h1.extend((float(lengths[e]), float("inf")) for e in births)
    return {"h0": np.array(h0).reshape(-1, 2), "h1": np.array(sorted(h1)).reshape(-1, 2),
            "n_points": n, "n_edges": len(edges), "n_triangles": len(triangles),
            "max_edge": cutoff, "kesildi": len(births), "n_sifir_omur": n_zero}


def persistence_landscape(
    diagram: ArrayLike, *, k_max: int = 5, grid: ArrayLike | None = None, n_grid: int = 400,
) -> dict[str, Any] | None:
    """Çadır fonksiyonlarının noktasal azalan sırası λ₁..λₖ.

    Sonsuz ölümler hariçtir ve sayılır. Sonlu çubuk yoksa otomatik grid [0,1],
    manzara sıfırdır. Özel grid sonlu, en az iki noktalı ve kesin artan olmalıdır;
    çıktının sütun sayısını özel grid'in uzunluğu belirler (n_grid yerine).
    """
    cleaned = _diagram(diagram)
    if cleaned is None or not _integer(k_max, 1) or not _integer(n_grid, 2):
        return None
    bars, n_infinite = cleaned
    times: np.ndarray | None
    with np.errstate(over="ignore", invalid="ignore"):
        if grid is None:
            lo, hi = (float(bars[:, 0].min()), float(bars[:, 1].max())) if len(bars) else (0.0, 1.0)
            times = np.linspace(lo, hi, n_grid)
        else:
            times = _array(grid)
            if times is None or times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
                return None
        if not np.all(np.isfinite(times)):
            return None
        tents = np.maximum(0.0, np.minimum(times[None, :] - bars[:, 0, None],
                                         bars[:, 1, None] - times[None, :]))
    if not np.all(np.isfinite(tents)):
        return None
    levels = np.zeros((k_max, len(times)))
    count = min(k_max, len(bars))
    if count:
        levels[:count] = np.sort(tents, axis=0)[-count:][::-1]
    return {"grid": times, "landscapes": levels, "n_bar": len(bars), "n_sonsuz": n_infinite}


def landscape_norms(
    diagram: ArrayLike, *, k_max: int = 5, n_grid: int = 2000,
) -> dict[str, Any] | None:
    """İlk k seviyenin L1/L2 normları; kesin L1 TÜM sonlu çubukları kapsar.

    l2 = sqrt(Σₖ ∫λₖ²), seviye normlarının toplamı değildir. Sonsuz ölümler
    hariçtir; n_sonsuz ayrıca taşınır. Boş diagramın normları gerçek sıfırdır.
    """
    landscape = persistence_landscape(diagram, k_max=k_max, n_grid=n_grid)
    cleaned = _diagram(diagram)
    if landscape is None or cleaned is None:
        return None
    bars, n_infinite = cleaned
    with np.errstate(over="ignore", invalid="ignore"):
        lifetime = bars[:, 1] - bars[:, 0]
        values, times = landscape["landscapes"], landscape["grid"]
        l1_levels = np.trapezoid(values, times, axis=1)
        l2_squared = np.trapezoid(values * values, times, axis=1)
        exact = float(np.sum((lifetime / 2) * (lifetime / 2)))
        total_life = float(lifetime.sum())
        l1, l2 = float(l1_levels.sum()), float(np.sqrt(l2_squared.sum()))
    if not np.all(np.isfinite([exact, total_life, l1, l2])):
        return None
    return {"l1_seviye": l1_levels, "l2_seviye": np.sqrt(l2_squared), "l1": l1, "l2": l2,
            "l1_toplam_kesin": exact, "toplam_omur": total_life,
            "max_omur": float(lifetime.max()) if len(lifetime) else 0.0,
            "n_bar": len(bars), "n_sonsuz": n_infinite}


def sliding_window_tda(
    returns: ArrayLike, *, window: int = 50, step: int = 1, max_edge: float | None = None,
    k_max: int = 5, standardize: bool = False,
) -> dict[str, Any] | None:
    """Yalnız pencere sonuna kadarki getirilerle nedensel H1 norm serileri.

    standardize=True: pencere içi sütun std'sine (ddof=0) böl, merkezleme yok.
    Sıfır std veya sayısal ölçümsüzlükte o satırın TÜM ölçümleri None'dır;
    bu durumda seriler object dizisi olur, yoksa sayısal dizidir. Ham sabit
    bulut geçerlidir (H1 yok). window için Rips'in 3..200 sınırı geçerlidir.
    n_h1 sonsuz çubukları da sayar; normlar yalnız sonlu çubukları kapsar.
    h0_toplam_omur yalnız sonlu H0 ömürlerini toplar.
    """
    arr = _array(returns)
    if (arr is None or arr.ndim != 2 or arr.shape[1] < 2 or not np.all(np.isfinite(arr))
            or not _integer(window, 3) or window > 200 or len(arr) < window
            or not _integer(step, 1) or not _integer(k_max, 1)
            or not isinstance(standardize, (bool, np.bool_))
            or (max_edge is not None and _nonnegative(max_edge) is None)):
        return None
    ends = np.arange(window - 1, len(arr), step, dtype=int)
    keys = ("l1", "l2", "l1_toplam_kesin", "n_h1", "max_omur", "h0_toplam_omur")
    series: dict[str, list[float | int | None]] = {key: [] for key in keys}
    truncated = 0
    for end in ends:
        cloud: np.ndarray | None = arr[end - window + 1:end + 1]
        if standardize and cloud is not None:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                std = cloud.std(axis=0)
                cloud = cloud / std if np.all(np.isfinite(std) & (std > 0)) else None
        persistence = rips_persistence(cloud, max_edge=max_edge) if cloud is not None else None
        norms = landscape_norms(persistence["h1"], k_max=k_max) if persistence is not None else None
        row: dict[str, Any] = {}
        if persistence is not None:
            truncated += persistence["kesildi"]
        if persistence is not None and norms is not None:
            h0 = persistence["h0"][:, 1]
            with np.errstate(over="ignore"):
                h0_life = float(h0[np.isfinite(h0)].sum())
            if np.isfinite(h0_life):
                row = {**norms, "n_h1": len(persistence["h1"]), "h0_toplam_omur": h0_life}
        for key in keys:
            series[key].append(row.get(key))
    return {"end_index": ends, **{key: np.array(values) for key, values in series.items()},
            "window": window, "step": step, "causal": True,
            "standardize": bool(standardize), "kesildi_toplam": truncated}
