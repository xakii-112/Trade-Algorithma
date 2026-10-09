"""sample_entropy k-d ağacı gövdesi ↔ kaba-kuvvet O(n²) referansı: BİT-DENKLİK kilidi.

Tur-71: `_match_count` `cKDTree(p=inf).count_neighbors` ile yeniden yazıldı (ölçüldü
n=10000'de 48,7×). Sayımlar TAMSAYI olduğu için sonucun yaklaşık değil BİREBİR aynı
kalması gerekir; bu dosya onu kilitler. Kaba-kuvvet referans BİLEREK burada tutulur —
üretim gövdesi değişirse test onunla birlikte değişmesin diye bağımsız kopyadır.
"""
import math

import numpy as np
import pytest

from nihai.ai.quant_math import sample_entropy


def _sample_entropy_bruteforce(arr: np.ndarray, m: int, r: float) -> float | None:
    """Richman-Moorman referansı — eski üretim gövdesinin birebir kopyası."""
    mi = int(m)
    n = arr.size
    if n < mi + 3:
        return None
    sd = float(np.std(arr, ddof=0))
    if sd == 0.0:
        return None
    tol = float(r) * sd
    n_templates = n - mi
    if n_templates < 2:
        return None

    def _count(dim: int) -> int:
        templates = np.array([arr[i : i + dim] for i in range(n_templates)])
        total = 0
        for i in range(n_templates - 1):
            dists = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            total += int(np.sum(dists <= tol))
        return total * 2

    b_count, a_count = _count(mi), _count(mi + 1)
    if b_count == 0 or a_count == 0:
        return None
    return float(-math.log(a_count / b_count))


def _series(kind: int, n: int, rng) -> np.ndarray:
    if kind == 0:
        return rng.normal(0, 1, n)
    if kind == 1:
        return np.round(rng.normal(0, 1, n), 2)          # çok sayıda BERABERLİK
    if kind == 2:
        return np.cumsum(rng.normal(0, 0.3, n))          # trendli / kendiyle ilintili
    return rng.integers(0, 5, n).astype(float)           # aşırı tekrar (sınırda eşitlik)


@pytest.mark.parametrize("trial", range(24))
def test_kdtree_govdesi_kaba_kuvvetle_bit_denk(trial):
    rng = np.random.default_rng(1000 + trial)
    n = int(rng.integers(20, 400))
    x = _series(trial % 4, n, rng)
    m = int(rng.integers(1, 4))
    r = float(rng.choice([0.1, 0.2, 0.5]))
    got, ref = sample_entropy(x, m=m, r=r), _sample_entropy_bruteforce(x, m, r)
    if ref is None:
        assert got is None
    else:
        assert got == ref, f"AYRIŞMA n={n} m={m} r={r}: {got!r} != {ref!r}"


def test_sinirda_esitlik_dahil_sayilir():
    """Mesafe TAM `tol` olan çiftler her iki gövdede de SAYILIR (`<=`, dışlayıcı değil)."""
    x = np.array([0.0, 1.0, 2.0, 3.0] * 12, dtype=float)
    r = 1.0 / float(np.std(x))                    # tol = 1.0 → komşu şablonlar tam sınırda
    assert sample_entropy(x, m=2, r=r) == _sample_entropy_bruteforce(x, 2, r)


def test_dejenere_girdiler_none():
    assert sample_entropy(np.full(50, 3.0), m=2, r=0.2) is None      # std=0
    assert sample_entropy(np.array([1.0, 2.0]), m=2, r=0.2) is None  # çok kısa
    assert sample_entropy(np.random.normal(0, 1, 50), m=2, r=0.0) is None  # r<=0


def test_buyuk_n_makul_surede_biter():
    """Eski O(n²) gövde n=8000'de ~3 sn sürüyordu; ağaç gövdesi saniyenin altında."""
    import time
    x = np.random.default_rng(7).normal(0, 1, 8000)
    t0 = time.perf_counter()
    val = sample_entropy(x, m=2, r=0.2)
    assert val is not None and math.isfinite(val)
    assert time.perf_counter() - t0 < 1.5, "k-d ağacı yolu devre dışı kalmış olabilir"
