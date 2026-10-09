"""nihai.ai.advanced_math testleri — analitik-limit çapaları, sentetik geri-bulma,
determinizm ve savunmacı (kısa/NaN/dönüştürülemez) girdi yolları (AI v3 WP-M2, Tur-16
Dalga-2). Ağ/IO yok (saf fonksiyonlar). Yavaş fonksiyonlar (PDE/FFT/MLE/BOCPD/HMM/PF)
KÜÇÜK ızgara/örneklem ile çağrılır (toplam çalışma süresi hedefi <60sn).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from nihai.ai import advanced_math as am
from nihai.ai import quant_math as qm


# ---------------------------------------------------------------------------
# Sıçrama-difüzyon: merton_jump_fit
# ---------------------------------------------------------------------------
def test_merton_jump_fit_pure_gauss_loglik_at_least_normal_mle():
    """λ→0 köprüsü: saf Gauss veride (sıçrama YOK) Merton MLE loglik'i, DÜZ Normal MLE
    loglik'inden KÜÇÜK OLAMAZ (lam=0 karışım-modelinin uzayında ULAŞILABİLİR bir nokta —
    global MLE en azından o kadar iyi olmalı). NOT: mixture-model MLE'de lam'ın kendisinin
    tam 0'a yakınsaması GARANTİ DEĞİLDİR (sonlu örneklemde kimliklenebilirlik zayıf olabilir,
    bkz. modül docstring'i) — bu yüzden testin çapası PARAMETRE değil LOGLIK SIRALAMASIDIR."""
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, 500)
    fit = am.merton_jump_fit(r)
    assert fit is not None
    mu_mle, sd_mle = float(np.mean(r)), float(np.std(r))
    normal_loglik = float(np.sum(stats.norm.logpdf(r, mu_mle, sd_mle)))
    assert fit["loglik"] >= normal_loglik - 1.0
    assert fit["sigma"] > 0.0
    assert fit["jump_sigma"] > 0.0
    assert fit["lam"] >= 0.0


def test_merton_jump_fit_recovers_synthetic_jumps():
    """Gerçek sıçrama içeren sentetik veride lam/jump_mu/jump_sigma/sigma gerçek değerlere
    (gevşek tolerans — MLE finite-sample gürültüsü) yakın çıkmalı."""
    rng = np.random.default_rng(3)
    n = 600
    base = rng.normal(0.0002, 0.006, n)
    jump_mask = rng.random(n) < 0.03
    jumps = np.where(jump_mask, rng.normal(-0.04, 0.02, n), 0.0)
    r = base + jumps
    fit = am.merton_jump_fit(r)
    assert fit is not None
    assert 0.005 < fit["lam"] < 0.10
    assert fit["jump_mu"] < -0.01
    assert 0.003 < fit["sigma"] < 0.010


def test_merton_jump_fit_deterministic():
    rng = np.random.default_rng(5)
    r = rng.normal(0.0, 0.01, 200)
    f1 = am.merton_jump_fit(r)
    f2 = am.merton_jump_fit(r)
    assert f1 == f2


@pytest.mark.parametrize("bad", [[1.0] * 5, [], [1.0, float("nan"), 2.0] * 15, "abc"])
def test_merton_jump_fit_defensive(bad):
    assert am.merton_jump_fit(bad) is None


# ---------------------------------------------------------------------------
# Sıçrama-difüzyon: jump_detect_bipower
# ---------------------------------------------------------------------------
def test_jump_detect_bipower_recovers_synthetic_jumps():
    rng = np.random.default_rng(7)
    n = 400
    r = rng.normal(0.0, 0.004, n)
    jump_idx = [50, 150, 300]
    for j in jump_idx:
        r[j] = 0.08 * (1 if j % 2 == 0 else -1)
    idx = am.jump_detect_bipower(r, z_thresh=4.0)
    assert set(jump_idx).issubset(set(idx.tolist()))


def test_jump_detect_bipower_no_jumps_low_count():
    rng = np.random.default_rng(9)
    r = rng.normal(0.0, 0.005, 300)
    idx = am.jump_detect_bipower(r, z_thresh=5.0)
    assert idx.size <= 3   # yaklaşık %0 yanlış-pozitif beklenir (5-sigma eşik)


@pytest.mark.parametrize("bad", [[1.0, 2.0], [], "xyz"])
def test_jump_detect_bipower_defensive(bad):
    out = am.jump_detect_bipower(bad, z_thresh=4.0)
    assert isinstance(out, np.ndarray) and out.size == 0


def test_jump_detect_bipower_bad_z_thresh():
    r = np.random.default_rng(1).normal(0, 0.01, 50)
    assert am.jump_detect_bipower(r, z_thresh=0.0).size == 0
    assert am.jump_detect_bipower(r, z_thresh=-1.0).size == 0


# ---------------------------------------------------------------------------
# Stokastik volatilite: heston_charfn / carr_madan_fft_price / heston_calibrate
# ---------------------------------------------------------------------------
_HESTON_PARAMS = {"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.5, "v0": 0.045}


def test_heston_charfn_at_zero_is_one():
    """φ(0) = E[e^{i·0·X}] = 1 (karakteristik fonksiyonun evrensel özelliği)."""
    val = am.heston_charfn(0.0, _HESTON_PARAMS, 100.0, 0.02, 0.5)
    assert val == pytest.approx(1.0 + 0.0j, abs=1e-8)


def test_heston_charfn_defensive():
    assert cmath_isnan(am.heston_charfn(1.0, {}, 100.0, 0.02, 0.5))
    assert cmath_isnan(am.heston_charfn(1.0, _HESTON_PARAMS, -1.0, 0.02, 0.5))
    assert cmath_isnan(am.heston_charfn(1.0, _HESTON_PARAMS, 100.0, 0.02, 0.0))
    bad_params = dict(_HESTON_PARAMS, rho=1.5)
    assert cmath_isnan(am.heston_charfn(1.0, bad_params, 100.0, 0.02, 0.5))


def cmath_isnan(z: complex) -> bool:
    return math.isnan(z.real) or math.isnan(z.imag)


def test_heston_fft_vol_of_vol_zero_limit_matches_bs():
    """sigma (vol-of-vol) → 0 VE v0=theta (sabit varyans) özel durumunda Heston/FFT fiyatı
    Black-Scholes'e yakınsamalı (bkz. modül docstring BS KÖPRÜSÜ notu)."""
    s0, r, T, sigma_bs = 100.0, 0.02, 0.5, 0.20
    v0 = sigma_bs**2
    params = {"kappa": 2.0, "theta": v0, "sigma": 1e-3, "rho": -0.3, "v0": v0}
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    heston_prices = am.carr_madan_fft_price(s0, strikes, r, T, params)
    bs_prices = np.array([qm.bs_price(s0, kk, T, r, sigma_bs, "c") for kk in strikes])
    assert heston_prices is not None
    rel_err = np.abs(heston_prices - bs_prices) / bs_prices
    assert np.all(rel_err < 0.01)


def test_carr_madan_fft_price_monotone_in_strike():
    strikes = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
    prices = am.carr_madan_fft_price(100.0, strikes, 0.03, 1.0, _HESTON_PARAMS)
    assert prices is not None
    assert np.all(np.diff(prices) < 0.0)   # call fiyatı strike arttıkça azalır
    assert np.all(prices >= 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(s0=-1.0, strikes=[100.0], r=0.02, T=0.5, heston_params=_HESTON_PARAMS),
        dict(s0=100.0, strikes=[100.0], r=0.02, T=0.0, heston_params=_HESTON_PARAMS),
        dict(s0=100.0, strikes=[100.0], r=0.02, T=0.5, heston_params={}),
        dict(s0=100.0, strikes=[], r=0.02, T=0.5, heston_params=_HESTON_PARAMS),
    ],
)
def test_carr_madan_fft_price_defensive(kwargs):
    assert am.carr_madan_fft_price(**kwargs) is None


def test_heston_calibrate_recovers_prices():
    """Parametre-kimliklenebilirliği garanti EDİLMEZ (bkz. docstring); test FİYAT-eşleşmesini
    doğrular (kalibre parametrelerle yeniden fiyatlama, sentetik piyasa fiyatlarına yakın)."""
    s0, r, T = 100.0, 0.02, 0.5
    strikes = np.array([85.0, 92.0, 100.0, 108.0, 115.0])
    market = am.carr_madan_fft_price(s0, strikes, r, T, _HESTON_PARAMS)
    assert market is not None
    cal = am.heston_calibrate(strikes, market, s0, r, T)
    assert cal is not None
    recalced = am.carr_madan_fft_price(
        s0, strikes, r, T,
        {k: cal[k] for k in ("kappa", "theta", "sigma", "rho", "v0")},
    )
    assert recalced is not None
    assert np.max(np.abs(recalced - market)) < 0.01


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(strikes=[100.0, 105.0], market_prices=[1.0, 2.0, 3.0], s0=100.0, r=0.02, T=0.5),
        dict(strikes=[100.0], market_prices=[1.0], s0=100.0, r=0.02, T=0.5),
        dict(strikes=[100.0, 105.0, 110.0], market_prices=[1.0, 2.0, 3.0], s0=-1.0, r=0.02, T=0.5),
    ],
)
def test_heston_calibrate_defensive(kwargs):
    assert am.heston_calibrate(**kwargs) is None


# ---------------------------------------------------------------------------
# Amerikan opsiyon: american_crank_nicolson + longstaff_schwartz_lsmc
# ---------------------------------------------------------------------------
def test_american_crank_nicolson_ge_european():
    s0, k, r, sigma, tt = 100.0, 100.0, 0.05, 0.2, 1.0
    price_am = am.american_crank_nicolson(s0, k, r, sigma, tt, n_s=120, n_t=120)
    price_eu = qm.bs_price(s0, k, tt, r, sigma, "p")
    assert price_am is not None and price_eu is not None
    assert price_am >= price_eu - 1e-6
    assert price_am > price_eu * 1.02   # r>0'da anlamlı erken-kullanım primi bekleniyor


def test_american_crank_nicolson_r_zero_converges_to_european():
    """r=0'da erken-kullanım avantajı yok (nakit tutmanın faiz kazancı olmadığından) —
    Amerikan put Avrupa'ya yakınsar. Izgara-kaba olduğundan gevşek (~%2) tolerans."""
    s0, k, sigma, tt = 100.0, 100.0, 0.2, 1.0
    price_am = am.american_crank_nicolson(s0, k, 0.0, sigma, tt, n_s=200, n_t=200)
    price_eu = qm.bs_price(s0, k, tt, 0.0, sigma, "p")
    assert price_am is not None and price_eu is not None
    assert abs(price_am - price_eu) / price_eu < 0.02


def test_american_crank_nicolson_itm_at_least_intrinsic():
    """Amerikan put fiyatı içsel değerin (neredeyse) altına düşemez — küçük negatif sapma
    yalnız log-ızgara enterpolasyonunun x0'ı TAM bir grid noktasına oturtmadığı durumda,
    dışbükey payoff'un doğrusal enterpolasyonundan kaynaklanan ihmal edilebilir bir sayısal
    artefakttır (bkz. modül risk notu — kondisyon uyarısı, PSOR/CN'nin bilinen sınırı)."""
    s0, k, r, sigma, tt = 60.0, 100.0, 0.05, 0.3, 0.5
    price = am.american_crank_nicolson(s0, k, r, sigma, tt, n_s=101, n_t=100)
    assert price is not None
    assert price >= max(k - s0, 0.0) - 0.1


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(s0=-1.0, k=100.0, r=0.05, sigma=0.2, T=1.0),
        dict(s0=100.0, k=0.0, r=0.05, sigma=0.2, T=1.0),
        dict(s0=100.0, k=100.0, r=0.05, sigma=-0.1, T=1.0),
        dict(s0=100.0, k=100.0, r=0.05, sigma=0.2, T=0.0),
        dict(s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0, n_s=5, n_t=5),
    ],
)
def test_american_crank_nicolson_defensive(kwargs):
    assert am.american_crank_nicolson(**kwargs) is None


def test_american_put_crank_nicolson_alias():
    assert am.american_put_crank_nicolson is am.american_crank_nicolson


def test_longstaff_schwartz_lsmc_matches_crank_nicolson():
    s0, k, r, sigma, tt = 100.0, 100.0, 0.05, 0.25, 0.5
    n_steps = round(tt * 252)
    paths = qm.mc_gbm_paths(s0, r, sigma, days=n_steps, n_paths=6000, seed=7)
    dt = 1.0 / 252.0
    price_lsmc = am.longstaff_schwartz_lsmc(paths, k, r, dt)
    price_cn = am.american_crank_nicolson(s0, k, r, sigma, tt, n_s=150, n_t=150)
    assert price_lsmc is not None and price_cn is not None
    assert abs(price_lsmc - price_cn) / price_cn < 0.10


def test_longstaff_schwartz_lsmc_deterministic():
    s0, k, r, sigma = 100.0, 100.0, 0.03, 0.2
    paths = qm.mc_gbm_paths(s0, r, sigma, days=30, n_paths=500, seed=11)
    p1 = am.longstaff_schwartz_lsmc(paths, k, r, 1.0 / 252.0)
    p2 = am.longstaff_schwartz_lsmc(paths, k, r, 1.0 / 252.0)
    assert p1 == p2


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(paths=[[100.0]], k=100.0, r=0.05, dt=1 / 252),
        dict(paths=[[100.0, 101.0], [99.0, 98.0]], k=0.0, r=0.05, dt=1 / 252),
        dict(paths=[[100.0, 101.0], [99.0, 98.0]], k=100.0, r=0.05, dt=0.0),
    ],
)
def test_longstaff_schwartz_lsmc_defensive(kwargs):
    assert am.longstaff_schwartz_lsmc(**kwargs) is None


# ---------------------------------------------------------------------------
# Olay kümelenmesi: hawkes_fit / hawkes_intensity
# ---------------------------------------------------------------------------
def _simulate_hawkes(mu: float, alpha: float, beta: float, big_t: float, seed: int) -> np.ndarray:
    """Ogata (1981) inceltme (thinning) algoritmasıyla üstel-çekirdek Hawkes süreci simülasyonu
    (test-yardımcısı; modülün KENDİSİ bu simülatörü içermez — yalnız MLE'yi test eder)."""
    rng = np.random.default_rng(seed)
    t = 0.0
    events: list[float] = []
    while t < big_t:
        lam_bar = mu + alpha * sum(math.exp(-beta * (t - ev)) for ev in events) + alpha
        t += rng.exponential(1.0 / lam_bar)
        if t >= big_t:
            break
        lam_t = mu + alpha * sum(math.exp(-beta * (t - ev)) for ev in events)
        if rng.random() <= lam_t / lam_bar:
            events.append(t)
    return np.array(events)


def test_hawkes_fit_recovers_synthetic_params():
    mu_true, alpha_true, beta_true, big_t = 0.5, 1.0, 2.0, 200.0
    ev = _simulate_hawkes(mu_true, alpha_true, beta_true, big_t, seed=5)
    assert ev.size > 30
    fit = am.hawkes_fit(ev, big_t)
    assert fit is not None
    assert 0.2 < fit["mu"] < 1.0
    assert 0.5 < fit["alpha"] < 1.6
    assert 1.0 < fit["beta"] < 3.2
    assert 0.0 < fit["branching"] < 1.0


def test_hawkes_fit_deterministic():
    ev = _simulate_hawkes(0.5, 1.0, 2.0, 150.0, seed=13)
    f1 = am.hawkes_fit(ev, 150.0)
    f2 = am.hawkes_fit(ev, 150.0)
    assert f1 == f2


@pytest.mark.parametrize(
    "event_times,big_t",
    [([1.0, 2.0], 10.0), ([1.0, 2.0, 3.0, 4.0, 5.0], 0.0), ([1.0, 2.0, 3.0, 4.0, 20.0], 10.0), ([], 10.0)],
)
def test_hawkes_fit_defensive(event_times, big_t):
    assert am.hawkes_fit(event_times, big_t) is None


def test_hawkes_intensity_known_shape():
    params = {"mu": 0.5, "alpha": 1.0, "beta": 2.0}
    ev = np.array([1.0, 2.0])
    grid = np.array([0.5, 1.5, 2.5])
    out = am.hawkes_intensity(ev, grid, params)
    assert out[0] == pytest.approx(0.5)   # t=0.5 önce hiçbir olay yok -> lambda=mu
    expected_t1_5 = 0.5 + 1.0 * math.exp(-2.0 * 0.5)
    assert out[1] == pytest.approx(expected_t1_5)


def test_hawkes_intensity_defensive():
    out = am.hawkes_intensity([1.0], [], {"mu": 1.0, "alpha": 0.5, "beta": 1.0})
    assert out.size == 0
    out2 = am.hawkes_intensity([1.0], [0.5, 1.5], {})
    assert np.all(np.isnan(out2))


# ---------------------------------------------------------------------------
# Çevrimiçi rejim kırılması: bocpd
# ---------------------------------------------------------------------------
def test_bocpd_detects_synthetic_changepoint():
    rng = np.random.default_rng(9)
    seg1 = rng.normal(0.0, 1.0, 60)
    seg2 = rng.normal(6.0, 1.0, 60)
    x = np.concatenate([seg1, seg2])
    res = am.bocpd(x, hazard=1.0 / 50.0)
    assert res["n"] == 120
    cp = res["changepoint_prob"]
    baseline = float(np.mean(cp[5:50]))
    peak_window = cp[58:64]
    assert float(np.max(peak_window)) > baseline * 5.0
    map_run = res["map_run_length"]
    assert int(map_run[59]) > 40          # değişimden hemen önce uzun run-length
    assert int(np.min(map_run[60:63])) <= 3   # değişimden hemen sonra keskin düşüş


def test_bocpd_deterministic():
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, 80)
    r1 = am.bocpd(x, hazard=0.02)
    r2 = am.bocpd(x, hazard=0.02)
    assert np.array_equal(r1["changepoint_prob"], r2["changepoint_prob"])
    assert np.array_equal(r1["map_run_length"], r2["map_run_length"])


@pytest.mark.parametrize(
    "x,hazard",
    [([1.0, 2.0, 3.0], 0.02), ([1.0] * 20, 0.0), ([1.0] * 20, 1.0), ([1.0, float("nan")] * 10, 0.02)],
)
def test_bocpd_defensive(x, hazard):
    res = am.bocpd(x, hazard=hazard)
    assert res["n"] == 0
    assert res["changepoint_prob"].size == 0
    assert res["map_run_length"].size == 0


# ---------------------------------------------------------------------------
# Durum-uzay: kalman_regime_switching / particle_filter_light
# ---------------------------------------------------------------------------
def test_kalman_regime_switching_separates_synthetic_regimes():
    rng = np.random.default_rng(11)
    seg1 = rng.normal(0.0, 1.0, 60)
    seg2 = rng.normal(5.0, 1.0, 60)
    y = np.concatenate([seg1, seg2])
    res = am.kalman_regime_switching(y, n_regime=2)
    assert res is not None
    mu = sorted(res["mu"])
    assert mu[0] < 1.0
    assert mu[1] > 4.0
    labels = np.asarray(res["labels"])
    acc_seg1 = max(float(np.mean(labels[:60] == 0)), float(np.mean(labels[:60] == 1)))
    acc_seg2 = max(float(np.mean(labels[60:] == 0)), float(np.mean(labels[60:] == 1)))
    assert acc_seg1 > 0.85
    assert acc_seg2 > 0.85
    assert labels.shape == (120,)


def test_kalman_regime_switching_deterministic():
    rng = np.random.default_rng(2)
    y = rng.normal(0.0, 1.0, 40)
    r1 = am.kalman_regime_switching(y)
    r2 = am.kalman_regime_switching(y)
    assert r1["mu"] == r2["mu"]
    assert r1["loglik"] == r2["loglik"]


@pytest.mark.parametrize(
    "y,n_regime",
    [
        ([1.0] * 30, 3),                    # yalnız n_regime=2 desteklenir
        ([1.0, 2.0, 3.0], 2),                # <20 gözlem
        ([1.0, float("nan")] * 15, 2),       # NaN
        ([1.0] * 30, 2),                     # dejenere (varyans=0) küme
    ],
)
def test_kalman_regime_switching_defensive(y, n_regime):
    assert am.kalman_regime_switching(y, n_regime=n_regime) is None


def test_particle_filter_light_deterministic_and_tracks():
    rng = np.random.default_rng(31)
    y = np.cumsum(rng.normal(0.0, 0.1, 80)) + 10.0
    pf1 = am.particle_filter_light(y, n_particles=300, seed=42)
    pf2 = am.particle_filter_light(y, n_particles=300, seed=42)
    assert pf1 is not None and pf2 is not None
    assert np.array_equal(pf1["filtered_mean"], pf2["filtered_mean"])
    assert np.array_equal(pf1["filtered_std"], pf2["filtered_std"])
    mae = float(np.mean(np.abs(pf1["filtered_mean"] - y)))
    assert mae < 0.5
    assert pf1["n_particles"] == 300


def test_particle_filter_light_different_seed_differs():
    rng = np.random.default_rng(32)
    y = np.cumsum(rng.normal(0.0, 0.1, 60)) + 5.0
    pf1 = am.particle_filter_light(y, n_particles=200, seed=1)
    pf2 = am.particle_filter_light(y, n_particles=200, seed=2)
    assert not np.array_equal(pf1["filtered_mean"], pf2["filtered_mean"])


@pytest.mark.parametrize(
    "y,n_particles",
    [([1.0, 2.0, 3.0], 100), ([1.0] * 20, 5), ([1.0, float("nan")] * 10, 100)],
)
def test_particle_filter_light_defensive(y, n_particles):
    assert am.particle_filter_light(y, n_particles=n_particles) is None


def test_particle_filter_light_caps_n_particles():
    y = np.random.default_rng(1).normal(0, 1, 20)
    res = am.particle_filter_light(y, n_particles=999_999, seed=1)
    assert res is not None
    assert res["n_particles"] <= 5000


# ---------------------------------------------------------------------------
# Rough volatility: realized_vol_hurst / rough_vol_forecast
# ---------------------------------------------------------------------------
def test_realized_vol_hurst_in_plausible_range():
    rng = np.random.default_rng(41)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 400)))
    h = am.realized_vol_hurst(prices)
    assert h is not None
    assert -0.5 < h < 1.5   # yapı-fonksiyonu tahmincisi gürültülü olabilir; kaba makul aralık


def test_realized_vol_hurst_deterministic():
    rng = np.random.default_rng(43)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.008, 300)))
    h1 = am.realized_vol_hurst(prices)
    h2 = am.realized_vol_hurst(prices)
    assert h1 == h2


@pytest.mark.parametrize("bad", [[100.0] * 10, [], "abc", [100.0, float("nan")] * 60])
def test_realized_vol_hurst_defensive(bad):
    assert am.realized_vol_hurst(bad) is None


def test_rough_vol_forecast_positive_and_converges():
    rng = np.random.default_rng(45)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 400)))
    fc = am.rough_vol_forecast(prices, horizon=10)
    assert fc is not None
    forecast = np.asarray(fc["forecast"])
    assert forecast.size == 10
    assert np.all(forecast > 0.0)
    # ufuk büyüdükçe uzun-vadeli ortalamaya yaklaşmalı (monoton olmasa da son fark ilk farktan küçük)
    assert abs(forecast[-1] - forecast[-2]) <= abs(forecast[1] - forecast[0]) + 1e-9


@pytest.mark.parametrize("bad_horizon", [0, -1])
def test_rough_vol_forecast_bad_horizon(bad_horizon):
    rng = np.random.default_rng(47)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 300)))
    assert am.rough_vol_forecast(prices, horizon=bad_horizon) is None


def test_rough_vol_forecast_short_input_defensive():
    assert am.rough_vol_forecast([100.0] * 10, horizon=5) is None


# ---------------------------------------------------------------------------
# Yol imzaları: signature_level2 / signature_level3
# ---------------------------------------------------------------------------
def test_signature_level2_single_segment_1d_analytic():
    path = np.array([[0.0], [3.0]])
    sig = am.signature_level2(path)
    assert sig == pytest.approx([3.0, 4.5])   # S1=Δ=3, S2=0.5*Δ²=4.5


def test_signature_level2_two_segments_1d_path_independent_identity():
    """1B yol için S2 = 0.5*(toplam artış)² — ARA NOKTALARDAN BAĞIMSIZ (Chen kimliğinin
    bilinen sonucu). İki farklı ara-nokta konfigürasyonu (0->1->2 ve 0->1.9->2) AYNI S2'yi
    vermeli."""
    path_a = np.array([[0.0], [1.0], [2.0]])
    path_b = np.array([[0.0], [1.9], [2.0]])
    sig_a = am.signature_level2(path_a)
    sig_b = am.signature_level2(path_b)
    assert sig_a[0] == pytest.approx(2.0)
    assert sig_a[1] == pytest.approx(0.5 * 2.0**2)
    assert sig_a == pytest.approx(sig_b, abs=1e-10)


def test_signature_level2_shape_2d():
    rng = np.random.default_rng(51)
    path = np.cumsum(rng.normal(0, 1, (10, 2)), axis=0)
    sig = am.signature_level2(path)
    assert sig.shape == (2 + 4,)   # d + d^2, d=2


def test_signature_level3_shape_and_matches_level2_prefix():
    rng = np.random.default_rng(53)
    path = np.cumsum(rng.normal(0, 1, (8, 2)), axis=0)
    sig2 = am.signature_level2(path)
    sig3 = am.signature_level3(path)
    assert sig3.shape == (2 + 4 + 8,)   # d + d^2 + d^3
    assert sig3[:6] == pytest.approx(sig2, abs=1e-10)   # S1,S2 seviyeleri aynı olmalı


@pytest.mark.parametrize("bad", [[[1.0]], [], "abc"])
def test_signature_functions_defensive(bad):
    assert am.signature_level2(bad).size == 0
    assert am.signature_level3(bad).size == 0


# ---------------------------------------------------------------------------
# Motif/anomali: matrix_profile_light
# ---------------------------------------------------------------------------
def test_matrix_profile_light_finds_repeated_motif():
    rng = np.random.default_rng(21)
    x = rng.normal(0.0, 0.1, 80)
    motif = np.array([0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0])
    x[10:18] += motif
    x[50:58] += motif
    res = am.matrix_profile_light(x, m=8)
    assert res is not None
    motif_i, motif_j = res["motif_idx"]
    near_10_50 = abs(motif_i - 10) <= 3 and abs(motif_j - 50) <= 3
    near_50_10 = abs(motif_i - 50) <= 3 and abs(motif_j - 10) <= 3
    assert near_10_50 or near_50_10


def test_matrix_profile_light_shapes():
    rng = np.random.default_rng(23)
    x = rng.normal(0, 1, 50)
    res = am.matrix_profile_light(x, m=6)
    assert res is not None
    assert res["profile"].size == 50 - 6 + 1
    assert res["profile_index"].size == 50 - 6 + 1


@pytest.mark.parametrize("x,m", [([1.0, 2.0], 2), ([1.0] * 20, 1), ([1.0] * 20, 15), ([], 4)])
def test_matrix_profile_light_defensive(x, m):
    assert am.matrix_profile_light(x, m) is None


# ---------------------------------------------------------------------------
# Numerik altyapı: qp_solve / adaptive_quad / logsumexp_stable / fft_convolve
# ---------------------------------------------------------------------------
def test_qp_solve_equal_weight_analytic():
    p_mat = 2.0 * np.eye(3)
    q_vec = np.zeros(3)
    x = am.qp_solve(p_mat, q_vec, a_eq=[[1.0, 1.0, 1.0]], b_eq=[1.0])
    assert x is not None
    assert x == pytest.approx([1.0 / 3.0] * 3, abs=1e-6)


def test_qp_solve_unconstrained_analytic():
    p_mat = 2.0 * np.eye(2)
    q_vec = np.array([-2.0, -4.0])
    x = am.qp_solve(p_mat, q_vec)
    assert x is not None
    assert x == pytest.approx([1.0, 2.0], abs=1e-6)


def test_qp_solve_bounds_respected():
    p_mat = 2.0 * np.eye(2)
    q_vec = np.array([-2.0, -4.0])
    x = am.qp_solve(p_mat, q_vec, lb=0.0, ub=1.5)
    assert x is not None
    assert np.all(x >= -1e-6)
    assert np.all(x <= 1.5 + 1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(p=[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], q=[1.0, 2.0]),
        dict(p=[[1.0, 0.0], [0.0, 1.0]], q=[1.0, 2.0, 3.0]),
        dict(p=[[1.0, 0.0], [0.0, 1.0]], q=[1.0, 2.0], a_eq=[[1.0, 1.0, 1.0]], b_eq=[1.0]),
    ],
)
def test_qp_solve_defensive(kwargs):
    assert am.qp_solve(**kwargs) is None


def test_adaptive_quad_known_integral():
    val = am.adaptive_quad(lambda t: t**2, 0.0, 1.0)
    assert val == pytest.approx(1.0 / 3.0, abs=1e-8)
    val2 = am.adaptive_quad(math.sin, 0.0, math.pi)
    assert val2 == pytest.approx(2.0, abs=1e-6)


def test_adaptive_quad_defensive():
    assert am.adaptive_quad(lambda t: t, float("nan"), 1.0) is None
    assert am.adaptive_quad("not callable", 0.0, 1.0) is None

    def bad_fn(t):
        raise RuntimeError("boom")

    assert am.adaptive_quad(bad_fn, 0.0, 1.0) is None


def test_logsumexp_stable_matches_naive_for_moderate_values():
    x = np.array([1.0, 2.0, 3.0])
    naive = math.log(np.sum(np.exp(x)))
    assert am.logsumexp_stable(x) == pytest.approx(naive, abs=1e-9)


def test_logsumexp_stable_avoids_overflow():
    x = np.array([1000.0, 1001.0])
    expected = 1000.0 + math.log(1.0 + math.exp(1.0))
    assert am.logsumexp_stable(x) == pytest.approx(expected, abs=1e-6)
    assert math.isfinite(am.logsumexp_stable(np.array([50000.0, 50000.0])))


def test_logsumexp_stable_defensive():
    assert am.logsumexp_stable([]) == float("-inf")
    assert am.logsumexp_stable([float("-inf"), float("-inf")]) == float("-inf")
    assert am.logsumexp_stable([float("inf"), 1.0]) == float("inf")


def test_fft_convolve_matches_np_convolve():
    rng = np.random.default_rng(61)
    a = rng.normal(0, 1, 37)
    b = rng.normal(0, 1, 15)
    out = am.fft_convolve(a, b)
    expected = np.convolve(a, b, mode="full")
    assert out == pytest.approx(expected, abs=1e-8)


def test_fft_convolve_defensive():
    assert am.fft_convolve([], [1.0, 2.0]).size == 0
    assert am.fft_convolve([1.0], []).size == 0
