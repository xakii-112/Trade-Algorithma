"""Tur-79 E: bağımsız binom, skaler yineleme, DGP ve enerji özdeşliği kilitleri.

Tohumlar 0..4 DCC: max |a-.04|=.003186, |b-.93|=.009511 -> .0064/.0191.
Tohumlar 0..2 FIGARCH: max |d-.4|=.047225 -> .095.
Haar beyaz gürültü (N=32768,J=8): max bağıl hata .081917 -> .165;
log eğim max hatası beyaz .011307 / yürüyüş .067428 -> .023 / .135.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import binom

from nihai.ai import econometrics as E
from nihai.ai.quant_math import _garch11_sigma2_path

QBAR = np.array([[1., .3, .5], [.3, 1., .2], [.5, .2, 1.]])


def _dcc_dgp(seed: int, a: float = .04, b: float = .93) -> np.ndarray:
    """Üretim fonksiyonundan bağımsız, skaler zaman döngülü DGP; 500 ısınma."""
    rng = np.random.default_rng(seed)
    q = QBAR.copy()
    out = []
    for _ in range(4500):
        r = q / np.sqrt(np.outer(np.diag(q), np.diag(q)))
        e = np.linalg.cholesky(r) @ rng.normal(size=3)
        out.append(e)
        q = (1 - a - b) * QBAR + a * np.outer(e, e) + b * q
    return np.array(out[500:])


def _binomial_weights(d: float, beta: float, phi: float, trunc: int) -> np.ndarray:
    """scipy genelleştirilmiş binomu ve doğrudan polinom çarpımı."""
    k = np.arange(trunc + 1)
    pi = (-1.) ** k * binom(d, k)
    c = -np.convolve([1., -phi], pi)[1:trunc + 1]
    c[0] -= beta
    return c


def _fig_dgp(seed: int, d: float = .4, beta: float = .3, phi: float = .1) -> np.ndarray:
    """T=6000, kesme=1000; binom katsayılarıyla 2000 bar ısınmış DGP."""
    c = _binomial_weights(d, beta, phi, 1000)
    omega = .05
    var0 = omega / (1 - beta - c.sum())
    eps2 = np.full(9000, var0)
    z = np.random.default_rng(seed).normal(size=8000)
    s = var0
    out = []
    for t, noise in enumerate(z, 1000):
        s = omega + beta * s + c @ eps2[t - 1000:t][::-1]
        e = np.sqrt(s) * noise
        eps2[t] = e * e
        out.append(e)
    return np.array(out[2000:])


@pytest.mark.parametrize("seed", range(5))
def test_dcc_recovery(seed: int) -> None:
    fit = E.dcc_garch_fit(_dcc_dgp(seed))
    assert fit is not None
    assert abs(fit['a'] - .04) < .0064
    assert abs(fit['b'] - .93) < .0191
    assert fit['ic_orneklem'] and not fit['causal']
    assert fit['pairs'] == [(0, 1), (0, 2), (1, 2)]
    assert fit['corr_path'].shape == (4000, 3)
    np.testing.assert_allclose(fit['corr_path'][-1], fit['R_last'][np.triu_indices(3, 1)])
    for univ in fit['univariate']:
        assert univ['persistence'] == univ['alpha'] + univ['beta']


def test_dcc_ccc_null_five_seeds() -> None:
    lr = []
    for seed in range(5):
        fit = E.dcc_garch_fit(_dcc_dgp(seed, 0., 0.))
        assert fit is not None
        lr.append(fit['lr_vs_ccc'])
    # Ölçülen [0,.018775,0,.317106,.283335]: 5/5 geçiyor; sözleşme çoğunluğu ister.
    assert sum(value < 5.991 for value in lr) >= 3
    assert min(lr) >= -1e-9


def test_dcc_ccc_identity_and_independent_likelihood() -> None:
    eps = np.random.default_rng(19).normal(size=(400, 3))
    qb = QBAR * np.outer([2., 3., 4.], [2., 3., 4.])
    path = E.dcc_correlation_path(eps, 0, 0, qb)
    assert path is not None
    expected = np.broadcast_to(QBAR, (400, 3, 3))
    np.testing.assert_allclose(path['R'], expected, atol=1e-12, rtol=0)
    independent = -.5 * (len(eps) * np.linalg.slogdet(QBAR)[1]
                        + np.einsum('ti,ij,tj->', eps, np.linalg.inv(QBAR), eps)
                        - np.sum(eps * eps))
    assert abs(E._dcc_loglik(eps, path['R']) - independent) < 1e-9
    assert E._dcc_loglik(eps, np.ones((400, 3, 3))) == -1e100


def test_dcc_path_scalar_reference_causality_and_forecast() -> None:
    eps = np.random.default_rng(8).normal(size=(500, 3))
    a, b = .04, .93
    path = E.dcc_correlation_path(eps, a, b, QBAR)
    assert path is not None and path['causal']
    expected, q = [], QBAR.copy()
    for e in eps:
        expected.append(q / np.sqrt(np.outer(np.diag(q), np.diag(q))))
        q = (1 - a - b) * QBAR + a * np.outer(e, e) + b * q
    np.testing.assert_allclose(path['R'], expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(path['Q_next'], q, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.diagonal(path['R'], axis1=1, axis2=2), 1., atol=1e-12, rtol=0)
    assert np.linalg.eigvalsh(path['R']).min() > 0
    changed = eps.copy()
    changed[230:] *= 30
    later = E.dcc_correlation_path(changed, a, b, QBAR)
    np.testing.assert_array_equal(path['R'][:231], later['R'][:231])
    assert not E.dcc_correlation_path(eps, a, b)['causal']
    forecast = E.dcc_forecast({'a': a, 'b': b, 'Qbar': QBAR, 'Q_next': q}, 2000)
    assert forecast is not None and forecast['yaklasim'] == 'engle_sheppard'
    for k in (0, 1, 9, 1999):
        hq = (1 - (a + b) ** k) * QBAR + (a + b) ** k * q
        hr = hq / np.sqrt(np.outer(np.diag(hq), np.diag(hq)))
        np.testing.assert_allclose(forecast['R'][k], hr, atol=1e-12, rtol=0)
    np.testing.assert_allclose(forecast['R'][-1], QBAR, atol=1e-12, rtol=0)


def test_dcc_determinism_and_univariate_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    x = _dcc_dgp(4)[:300]
    first = E.dcc_garch_fit(x, starts=[(.03, .92)])
    second = E.dcc_garch_fit(x, starts=[(.03, .92)])
    np.testing.assert_array_equal(first['corr_path'], second['corr_path'])
    assert first['a'] == second['a'] and first['b'] == second['b']
    original = E.garch11_fit
    calls = []

    def fail_second(arr: np.ndarray) -> dict | None:
        calls.append(arr)
        return None if len(calls) == 2 else original(arr)

    monkeypatch.setattr(E, 'garch11_fit', fail_second)
    assert E.dcc_garch_fit(x) is None
    assert len(calls) == 2


@pytest.mark.parametrize('trunc', [1, 20, 1000])
def test_figarch_d_zero_identity(trunc: int) -> None:
    eps2 = np.random.default_rng(4).normal(size=1500) ** 2
    actual = E._figarch_sigma2(eps2, .03, 0, .9, .96, trunc, 2.)
    expected = _garch11_sigma2_path(.03, .96 - .9, .9, eps2, 2.)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    weights = E._figarch_weights(0., .9, .96, trunc)
    np.testing.assert_array_equal(weights[1:], 0.)
    assert actual[0] == 2.


@pytest.mark.parametrize('d', [0., .1, .4, .9, 1.])
def test_figarch_weights_polynomial(d: float) -> None:
    np.testing.assert_allclose(E._figarch_weights(d, .3, .1, 1000),
                               _binomial_weights(d, .3, .1, 1000), atol=1e-14, rtol=0)


def test_figarch_scalar_reference_and_prehistory() -> None:
    eps2 = np.random.default_rng(5).normal(size=80) ** 2
    c = _binomial_weights(.4, .3, .1, 100)
    expected = [1.3]
    for t in range(1, len(eps2)):
        shock = sum(c[k - 1] * (eps2[t - k] if t >= k else 1.3) for k in range(1, 101))
        expected.append(.04 + .3 * expected[-1] + shock)
    actual = E._figarch_sigma2(eps2, .04, .4, .3, .1, 100, 1.3)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    changed = eps2.copy()
    changed[40:] *= 100
    later = E._figarch_sigma2(changed, .04, .4, .3, .1, 100, 1.3)
    np.testing.assert_array_equal(actual[:41], later[:41])


@pytest.mark.parametrize('seed', range(3))
def test_figarch_recovery_and_scale(seed: int) -> None:
    x = _fig_dgp(seed)
    fit = E.figarch_fit(x)
    assert fit is not None and abs(fit['d'] - .4) < .095
    assert not fit['kisit_ihlali'] and fit['converged']
    assert fit['uncond_var'] is None and fit['ic_orneklem'] and not fit['causal']
    assert fit['aic'] == 8 - 2 * fit['loglik']
    assert fit['bic'] == 4 * np.log(len(x)) - 2 * fit['loglik']
    sigma2 = E._figarch_sigma2((x - x.mean()) ** 2, fit['omega'], fit['d'],
                               fit['beta'], fit['phi'], 1000, float(x.var()))
    ll = -.5 * np.sum(np.log(2 * np.pi * sigma2) + (x - x.mean()) ** 2 / sigma2)
    assert abs(ll - fit['loglik']) < 1e-8
    assert abs(sigma2[-1] - fit['last_sigma2']) < 1e-10
    assert abs(fit['lambda_toplam'] - _binomial_weights(fit['d'], fit['beta'], fit['phi'], 1000).sum()) < 1e-12
    big = E.figarch_fit(x * 100)
    assert big is not None
    for key in ['d', 'beta', 'phi']:
        assert abs(fit[key] - big[key]) < 1e-6
    assert abs(big['omega'] / (10000 * fit['omega']) - 1) < 1e-6
    assert abs(big['loglik'] - fit['loglik'] + len(x) * np.log(100)) < 1e-7


def test_figarch_garch_null() -> None:
    fit = E.figarch_fit(_fig_dgp(10, d=0., beta=.3, phi=.5))
    assert fit is not None and fit['garch11']['lr'] < 5.991
    assert fit['garch11']['lr'] > -1e-6
    assert fit['sinirda']['d_sifir']


def test_figarch_constraint_violation_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def invalid_opt(fun, *args, **kwargs):
        p = np.array([.3, .4, .3, .55])
        return SimpleNamespace(x=p, fun=fun(p), success=False)

    monkeypatch.setattr(E, 'minimize', invalid_opt)
    x = np.random.default_rng(2).choice([-1., 1.], size=400)
    fit = E.figarch_fit(x)
    assert fit is not None and fit['kisit_ihlali'] and not fit['converged']


@pytest.mark.parametrize('seed', range(5))
def test_modwt_energy_mra_and_shift(seed: int) -> None:
    x = np.random.default_rng(seed).normal(size=1003)
    wt = E.modwt_haar(x)
    assert wt is not None and wt['levels'] == 8 and not wt['causal']
    energy = np.sum(wt['W'] ** 2) + np.sum(wt['V'] ** 2)
    assert abs(energy / np.sum(x * x) - 1) < 1e-10
    mra = E.modwt_mra(x)
    np.testing.assert_allclose(mra['D'].sum(axis=0) + mra['S'], x, atol=1e-10, rtol=0)
    moved = E.modwt_haar(np.roll(x, 71))
    np.testing.assert_array_equal(moved['W'], np.roll(wt['W'], 71, axis=1))
    np.testing.assert_array_equal(moved['V'], np.roll(wt['V'], 71))


def test_modwt_independent_filters_and_zero_phase() -> None:
    x = np.random.default_rng(0).normal(size=32)
    wt = E.modwt_haar(x, levels=4)
    index = np.arange(len(x))
    for j in range(1, 5):
        # Etkin Haar filtresi: ilk yarı +1/2^j, ikinci yarı -1/2^j.
        width = 2 ** j
        expected = sum((1 if k < width // 2 else -1) * x[(index - k) % len(x)]
                       for k in range(width)) / width
        np.testing.assert_allclose(wt['W'][j - 1], expected, atol=1e-14, rtol=0)
    impulse = np.zeros(32)
    impulse[16] = 1.
    mra = E.modwt_mra(impulse, levels=4)
    for component in [*mra['D'], mra['S']]:
        np.testing.assert_allclose(component[(16 + index) % 32], component[(16 - index) % 32],
                                   atol=1e-14, rtol=0)


@pytest.mark.parametrize('seed', range(3))
def test_wavelet_white_noise_and_random_walk(seed: int) -> None:
    x = np.random.default_rng(seed).normal(size=2 ** 15)
    white = E.wavelet_variance(x, levels=8)
    walk = E.wavelet_variance(x.cumsum(), levels=8)
    assert white is not None and walk is not None
    assert np.max(np.abs(np.array(white['nu2']) * 2. ** np.arange(1, 9) - 1)) < .165
    assert abs(white['log_egim'] + 1) < .023
    assert abs(walk['log_egim'] - 1) < .135


def test_wavelet_variance_boundaries_and_energy_shares() -> None:
    x = np.arange(8.)
    result = E.wavelet_variance(x, levels=4)
    wt = E.modwt_haar(x, levels=4)
    assert result['M'] == [7, 5, 1, -7]
    assert result['nu2'][-1] is None
    for j in range(3):
        expected = sum(wt['W'][j, t] ** 2 for t in range(2 ** (j + 1) - 1, 8)) / result['M'][j]
        assert abs(result['nu2'][j] - expected) < 1e-14
    np.testing.assert_allclose(result['nu2_yanli'], (wt['W'] ** 2).mean(axis=1), atol=1e-14)
    np.testing.assert_allclose(result['enerji_payi'], (wt['W'] ** 2).sum(axis=1) / (x * x).sum(), atol=1e-14)
    assert E.wavelet_variance(x, levels=2)['log_egim'] is None
    assert E.wavelet_variance(x, levels=8)['nu2'][3:] == [None] * 5


@pytest.mark.parametrize('bad', [None, [], [1.], [0.] * 300, [1.] * 300, [np.nan] * 300,
                                      [np.inf] * 300, [[1., 2.]], 'bad', [1 + 2j] * 300,
                                      [1e308, -1e308] * 150])
def test_invalid_inputs(bad) -> None:
    for func in [E.dcc_garch_fit, E.figarch_fit, E.modwt_haar, E.modwt_mra, E.wavelet_variance]:
        assert func(bad) is None
    assert E.dcc_correlation_path(bad, .04, .93) is None


def test_bad_dimensions_parameters_and_singular_covariance() -> None:
    rng = np.random.default_rng(1)
    eps = rng.normal(size=(300, 3))
    for bad in [eps[:249], eps[:, :1], np.ones((300, 3)), np.stack([eps] * 2)]:
        assert E.dcc_garch_fit(bad) is None
    assert E.dcc_garch_fit(np.column_stack([eps[:, 0], eps[:, 0]])) is None
    for func in [E.figarch_fit, E.modwt_haar, E.modwt_mra, E.wavelet_variance]:
        assert func(eps) is None
    assert E.figarch_fit(eps[:249, 0]) is None
    assert E.modwt_haar(eps[:7, 0]) is None
    for bad in [[], [(1., 1.)], [(.1, np.nan)], 'x', None, 1.5, np.inf, -1, 0, True]:
        if bad is not None:
            assert E.dcc_garch_fit(eps, starts=bad) is None
        assert E.figarch_fit(eps[:, 0], trunc=bad) is None
        assert E.dcc_forecast({}, bad) is None
        if bad is not None:
            assert E.modwt_haar(eps[:, 0], levels=bad) is None
    for a, b in [(None, .3), (-.1, .3), (.4, .7), (np.inf, .1), (.1, np.nan)]:
        assert E.dcc_correlation_path(eps, a, b) is None
    for qb in [np.ones((3, 3)), np.eye(2), np.eye(3) * np.nan, QBAR + np.triu(np.ones((3, 3))),
               QBAR.astype(complex) + 1j]:
        assert E.dcc_correlation_path(eps, .04, .93, qb) is None
    assert E.dcc_forecast(None, 1) is None
    for qb in [np.ones((3, 3)), np.eye(2), np.eye(3) * np.nan, QBAR.astype(complex) + 1j]:
        assert E.dcc_forecast({'a': .04, 'b': .93, 'Qbar': qb, 'Q_next': QBAR}, 3) is None
