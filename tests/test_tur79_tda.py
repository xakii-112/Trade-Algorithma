"""Tur-79 T: bağımsız yoğun GF(2), SciPy ve kapalı-form topoloji hakemleri."""
from __future__ import annotations

from itertools import combinations
from time import perf_counter

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

from nihai.ai.tda import (
    landscape_norms,
    persistence_landscape,
    rips_persistence,
    sliding_window_tda,
)


def _polygon(n: int, radius: float = 1.0) -> np.ndarray:
    theta = np.arange(n) * (2 * np.pi / n)
    return radius * np.column_stack((np.cos(theta), np.sin(theta)))


def _rank_gf2(matrix: np.ndarray) -> int:
    """Bağımsız YOĞUN satır Gauss elemesi; bit maskesi/sütun indirgemesi yok."""
    a = matrix.copy()
    rank = 0
    for col in range(a.shape[1]):
        candidates = np.flatnonzero(a[rank:, col])
        if not len(candidates):
            continue
        pivot = rank + candidates[0]
        a[[rank, pivot]] = a[[pivot, rank]]
        for row in range(rank + 1, len(a)):
            if a[row, col]:
                a[row] = (a[row] + a[rank]) % 2
        rank += 1
    return rank


def _betti(distances: np.ndarray, radius: float) -> tuple[int, int]:
    n = len(distances)
    edges = [(i, j) for i, j in combinations(range(n), 2) if distances[i, j] <= radius]
    lookup = {edge: col for col, edge in enumerate(edges)}
    boundary1 = np.zeros((n, len(edges)), dtype=np.uint8)
    for col, (i, j) in enumerate(edges):
        boundary1[[i, j], col] = 1
    triangles = [((i, j), (i, k), (j, k)) for i, j, k in combinations(range(n), 3)
                 if max(distances[i, j], distances[i, k], distances[j, k]) <= radius]
    boundary2 = np.zeros((len(edges), len(triangles)), dtype=np.uint8)
    for col, triangle in enumerate(triangles):
        for edge in triangle:
            boundary2[lookup[edge], col] = 1
    rank1, rank2 = _rank_gf2(boundary1), _rank_gf2(boundary2)
    return n - rank1, len(edges) - rank1 - rank2


@pytest.mark.parametrize("seed", range(5))
def test_h0_scipy_single_linkage(seed: int) -> None:
    points = np.random.default_rng(seed).normal(size=(20 + 10 * seed, 2 + seed % 3))
    result = rips_persistence(points)
    deaths = result["h0"][:, 1]
    assert np.isinf(deaths).sum() == 1
    assert_allclose(np.sort(deaths[np.isfinite(deaths)]),
                    np.sort(linkage(pdist(points), method="single")[:, 2]), rtol=0, atol=1e-12)


def test_square() -> None:
    result = rips_persistence([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert_allclose(result["h1"], [[1, np.sqrt(2)]], rtol=0, atol=1e-12)
    assert_array_equal(result["h0"], [[0, 1], [0, 1], [0, 1], [0, np.inf]])
    assert (result["n_points"], result["n_edges"], result["n_triangles"]) == (4, 6, 4)
    assert result["n_sifir_omur"] == 2
    assert result["kesildi"] == 0


@pytest.mark.parametrize(("n", "radius"), [(6, 1), (12, 2)])
def test_regular_polygons(n: int, radius: float) -> None:
    result = rips_persistence(_polygon(n, radius))
    assert_allclose(result["h1"], [[2 * radius * np.sin(np.pi / n), radius * np.sqrt(3)]],
                    rtol=0, atol=1e-9)
    # Sıfır çiftler + pozitif çiftler, tüm çevrim doğumlarını tüketir.
    assert result["n_sifir_omur"] + len(result["h1"]) == result["n_edges"] - n + 1


def test_two_distant_polygons() -> None:
    points = np.vstack((_polygon(12, 2), _polygon(12, 2) + [20, 0]))
    result = rips_persistence(points)
    long = result["h1"][np.diff(result["h1"], axis=1).ravel() > 1]
    assert_allclose(long, np.tile([4 * np.sin(np.pi / 12), 2 * np.sqrt(3)], (2, 1)),
                    rtol=0, atol=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_isometry_and_permutation(seed: int) -> None:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(24, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    moved = (points @ rotation + [10, -5, 2])[rng.permutation(len(points))]
    original, transformed = rips_persistence(points), rips_persistence(moved)
    for key in ("h0", "h1"):
        assert_allclose(original[key], transformed[key], rtol=0, atol=1e-9)
        assert_array_equal(original[key], rips_persistence(points)[key])


@pytest.mark.parametrize("seed", range(20))
def test_dense_gf2_oracle_every_filtration(seed: int) -> None:
    points = np.random.default_rng(seed).normal(size=(7 + seed % 4, 2 + seed % 3))
    distances = squareform(pdist(points))
    result = rips_persistence(points)
    radii = np.unique(distances)
    for radius in radii:
        expected = _betti(distances, radius)
        actual = tuple(int(np.sum((result[key][:, 0] <= radius) & (radius < result[key][:, 1])))
                       for key in ("h0", "h1"))
        assert actual == expected, (seed, radius, actual, expected)
    # Kesilmiş kompleks de aynı yoğun hakemle doğrulanır; çoklu sonsuz H0 dahil.
    for radius in radii[[1, len(radii) // 3, len(radii) // 2]]:
        cut = rips_persistence(points, max_edge=float(radius))
        assert tuple(int(np.isinf(cut[key][:, 1]).sum()) for key in ("h0", "h1")) == _betti(
            distances, radius)


def test_single_bar_landscape_and_norms() -> None:
    result = persistence_landscape([[2, 6]], k_max=2, grid=np.array([1, 2, 3, 4, 5, 6, 7]))
    assert_array_equal(result["landscapes"], [[0, 0, 1, 2, 1, 0, 0], [0] * 7])
    norms = landscape_norms([[2, 6]], k_max=2, n_grid=2001)
    assert_allclose(norms["l1_seviye"], [4, 0], rtol=0, atol=1e-12)
    assert_allclose(norms["l2_seviye"], [np.sqrt(64 / 12), 0], rtol=1e-6)
    assert norms["l1_toplam_kesin"] == 4
    assert norms["toplam_omur"] == norms["max_omur"] == 4


@pytest.mark.parametrize("seed", range(20))
def test_landscape_exact_integrals(seed: int) -> None:
    rng = np.random.default_rng(seed)
    births = rng.uniform(-2, 2, 12)
    deaths = births + rng.uniform(.1, 4, 12)
    diagram = np.column_stack((births, deaths))
    result = landscape_norms(diagram, k_max=12)
    exact = np.sum((deaths - births) ** 2) / 4
    assert_allclose(result["l1_toplam_kesin"], exact, rtol=1e-14)
    assert_allclose(result["toplam_omur"], np.sum(deaths - births), rtol=1e-14)
    assert result["max_omur"] == np.max(deaths - births)
    assert result["n_bar"] == 12
    # Ölçülen en büyük bağıl hata 4,633e-7; yaklaşık 3 kat pay.
    assert abs(result["l1"] / exact - 1) < 1.5e-6
    assert_allclose(result["l2"] ** 2, np.sum((deaths - births) ** 3) / 12, rtol=3e-6)
    assert_allclose(result["l2"] ** 2, np.sum(result["l2_seviye"] ** 2), rtol=1e-14)
    assert_allclose(result["l1"], result["l1_seviye"].sum(), rtol=1e-14)
    # Noktasal sıralama bağımsız Python çadır hesabıyla karşılaştırılır.
    landscape = persistence_landscape(diagram, k_max=4, n_grid=71)
    expected = [[sorted([max(0, min(t - b, d - t)) for b, d in diagram], reverse=True)[k]
                 for t in landscape["grid"]] for k in range(4)]
    assert_array_equal(landscape["landscapes"], expected)
    assert landscape_norms(diagram, k_max=1)["l1"] < exact


@pytest.mark.parametrize("seed", range(5))
def test_cutoff(seed: int) -> None:
    points = np.random.default_rng(seed).normal(size=(14, 2))
    full = rips_persistence(points)
    cutoff = float(np.median(pdist(points)))
    expected = full["h1"][full["h1"][:, 0] <= cutoff].copy()
    expected[expected[:, 1] > cutoff, 1] = np.inf
    cut = rips_persistence(points, max_edge=cutoff)
    assert_array_equal(cut["h1"], expected)
    assert cut["kesildi"] == np.isinf(expected[:, 1]).sum()
    square = rips_persistence([[0, 0], [1, 0], [1, 1], [0, 1]], max_edge=1)
    assert_array_equal(square["h1"], [[1, np.inf]])
    assert square["kesildi"] == 1
    disconnected = rips_persistence(points, max_edge=0)
    assert disconnected["h0"].shape == (14, 2)
    assert np.isinf(disconnected["h0"][:, 1]).all()


@pytest.mark.parametrize("standardize", [False, True])
def test_sliding_causality_and_window_reference(standardize: bool) -> None:
    returns = np.random.default_rng(79).normal(size=(36, 3))
    original = returns.copy()
    baseline = sliding_window_tda(returns, window=12, step=3, standardize=standardize)
    changed = returns.copy()
    changed[24:] = 1000 * np.random.default_rng(80).normal(size=changed[24:].shape)
    altered = sliding_window_tda(changed, window=12, step=3, standardize=standardize)
    prefix = sliding_window_tda(returns[:24], window=12, step=3, standardize=standardize)
    mask = baseline["end_index"] < 24
    for key in ("l1", "l2", "l1_toplam_kesin", "n_h1", "max_omur", "h0_toplam_omur"):
        assert_array_equal(baseline[key][mask], altered[key][mask])
        assert_array_equal(baseline[key][mask], prefix[key])
    assert_array_equal(baseline["end_index"], np.arange(11, 36, 3))
    assert baseline["causal"] is True
    for row, end in enumerate(baseline["end_index"]):
        cloud = returns[end - 11:end + 1]
        if standardize:
            cloud = cloud / cloud.std(axis=0)
        persistence = rips_persistence(cloud)
        norms = landscape_norms(persistence["h1"])
        assert baseline["l1"][row] == norms["l1"]
        assert baseline["l2"][row] == norms["l2"]
        assert_allclose(baseline["h0_toplam_omur"][row],
                        linkage(pdist(cloud), method="single")[:, 2].sum(), atol=1e-12)
    assert_array_equal(returns, original)


def test_sliding_square_closed_form_and_missing_rows() -> None:
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    result = sliding_window_tda(square, window=4)
    assert_allclose(result["l1_toplam_kesin"], [(np.sqrt(2) - 1) ** 2 / 4], atol=1e-14)
    assert_array_equal(result["h0_toplam_omur"], [3])
    cut = sliding_window_tda(square, window=4, max_edge=1)
    assert cut["kesildi_toplam"] == 1
    assert_array_equal(cut["n_h1"], [1])
    assert_array_equal(cut["l1"], [0])
    returns = np.vstack((np.zeros((4, 2)), square))
    result = sliding_window_tda(returns, window=4, standardize=True)
    for key in ("l1", "l2", "l1_toplam_kesin", "n_h1", "max_omur", "h0_toplam_omur"):
        assert result[key][0] is None
        assert result[key][-1] is not None
    raw = sliding_window_tda(np.ones((6, 2)), window=4)
    assert_array_equal(raw["l1"], [0, 0, 0])


@pytest.mark.parametrize("bad", [None, [], [1], [1, 2, 3], [[1, 2]], [[0, 0], [1, 1]],
                                  [[np.nan, 0]] * 3, [[np.inf, 0]] * 3,
                                  [[1j, 0]] * 3, [["x", 0]] * 3, [[0], [1, 2]],
                                  np.ones((201, 2)), np.empty((3, 0))])
def test_invalid_points(bad: object) -> None:
    assert rips_persistence(bad) is None


@pytest.mark.parametrize("options", [{"max_edge": -1}, {"max_edge": np.inf}, {"max_edge": np.nan},
                                      {"max_edge": [1]}, {"max_edge": "x"}, {"n_max": 2},
                                      {"n_max": 3.5}, {"n_max": None}, {"omur_tol": -1},
                                      {"omur_tol": np.nan}, {"omur_tol": None}])
def test_invalid_rips_options(options: dict) -> None:
    assert rips_persistence(_polygon(4), **options) is None


@pytest.mark.parametrize("bad", [None, [1], [[1, 2, 3]], [[np.nan, 2]], [[0, np.nan]],
                                  [[np.inf, np.inf]], [[0, -np.inf]], [[2, 1]], [[1j, 2]], [["x", 2]]])
def test_invalid_diagrams(bad: object) -> None:
    assert persistence_landscape(bad) is None
    assert landscape_norms(bad) is None


def test_empty_infinite_zero_and_overflow() -> None:
    for diagram in ([], np.empty((0, 2)), [[0, np.inf]], [[1, 1]]):
        landscape = persistence_landscape(diagram)
        assert_array_equal(landscape["landscapes"], np.zeros((5, 400)))
        norms = landscape_norms(diagram)
        assert norms["l1"] == norms["l2"] == norms["max_omur"] == 0
    mixed = landscape_norms([[0, 2], [0, np.inf]])
    assert mixed["n_sonsuz"] == mixed["n_bar"] == 1
    assert mixed["l1_toplam_kesin"] == 1
    repeated = rips_persistence(np.ones((4, 2)))
    assert_array_equal(repeated["h0"], [[0, np.inf]])
    assert repeated["h1"].shape == (0, 2)
    assert repeated["n_sifir_omur"] == 6
    assert rips_persistence([[1e308, 0], [-1e308, 0], [0, 1]]) is None
    assert landscape_norms([[-1e308, 1e308]]) is None


@pytest.mark.parametrize("options", [{"k_max": 0}, {"k_max": 1.5}, {"n_grid": 1},
                                      {"n_grid": None}, {"k_max": True}])
def test_invalid_landscape_options(options: dict) -> None:
    assert persistence_landscape([[0, 1]], **options) is None
    assert landscape_norms([[0, 1]], **options) is None


@pytest.mark.parametrize("grid", [[], [1], [0, 0], [1, 0], [0, np.nan], [0, np.inf], [[0, 1]]])
def test_invalid_grid(grid: object) -> None:
    assert persistence_landscape([[0, 1]], grid=grid) is None


@pytest.mark.parametrize("bad", [None, [], [1, 2, 3], np.ones((3, 1)), np.ones((2, 2)),
                                  np.full((5, 2), np.nan), np.full((5, 2), np.inf)])
def test_invalid_returns(bad: object) -> None:
    assert sliding_window_tda(bad, window=3) is None


@pytest.mark.parametrize("options", [{"window": 2}, {"window": 201}, {"window": 5.5}, {"window": None},
                                      {"step": 0}, {"step": 1.5}, {"k_max": 0},
                                      {"max_edge": -1}, {"standardize": "yes"}])
def test_invalid_sliding_options(options: dict) -> None:
    assert sliding_window_tda(np.ones((60, 2)), **options) is None


def test_lifetime_tolerance() -> None:
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * 1e-13
    assert rips_persistence(points)["h1"].shape == (0, 2)
    raw = rips_persistence(points, omur_tol=0)
    assert raw["h1"].shape == (1, 2)
    assert_allclose(raw["h1"], [[1e-13, np.sqrt(2) * 1e-13]], rtol=1e-14, atol=0)


def test_performance_loose_limit() -> None:
    points = np.random.default_rng(79).normal(size=(50, 4))
    rips_persistence(points)
    start = perf_counter()
    result = rips_persistence(points)
    assert result["n_triangles"] == 19600
    assert perf_counter() - start < 2.0
