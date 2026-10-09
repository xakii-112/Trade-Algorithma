"""Tur-73: on bir saf istatistiğin analitik, sayısal ve dejenere girdi doğrulamaları.

Sentetik rastgele örneklerin tümü sabit SEED ile üretilir; dış veri/ağ kullanılmaz.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import ArrayLike

from nihai.ai import quant_math as qm

SEED = 73
FUNCTIONS = (
    "time_underwater", "gain_to_pain", "tail_ratio", "shannon_entropy", "renyi_entropy",
    "tsallis_entropy", "critical_slowing_down", "mfdfa", "nss_fit", "residual_momentum",
    "leveraged_etf_drag",
)
ENTROPIES = (qm.shannon_entropy, qm.renyi_entropy, qm.tsallis_entropy)


def _call(name: str, values: ArrayLike) -> dict | float | None:
    """İkinci dizi isteyen iki fonksiyona sabit uzunlukta geçerli eşlikçi sağlar."""
    if name == "nss_fit":
        return qm.nss_fit(np.linspace(0.1, 30.0, 512), values)
    if name == "residual_momentum":
        return qm.residual_momentum(values, np.sin(np.arange(512) * 0.17))
    return getattr(qm, name)(values)


@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, "gecersiz", None, 10**1000])
def test_sonlu_olmayan_ve_donusturulemeyen_veri_none(name, bad):
    # Uzunluk kapısı geçilir: bozuk eleman sırf kısa girdi olduğu için reddedilemez.
    values = np.random.default_rng(SEED).normal(0.0, 0.01, 512)
    if name == "time_underwater":
        values = np.cumprod(1.0 + values)
    data = values.tolist()
    data[127] = bad
    assert _call(name, data) is None


@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("bad", [[], None, "gecersiz", [[1, 2], [3]]])
def test_bos_ve_bozuk_yapi_none(name, bad):
    assert _call(name, bad) is None


@pytest.mark.parametrize("name", FUNCTIONS)
def test_saflik_girdi_korunur_ve_sonuc_tekrarlanir(name):
    values = np.random.default_rng(SEED).normal(0.0, 0.01, 512)
    if name == "time_underwater":
        values = np.cumprod(1.0 + values)
    before = values.copy()
    result = _call(name, values)
    assert result is not None
    assert _call(name, values) == result
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("name", FUNCTIONS)
def test_sonlu_asiri_buyuk_girdi_istisna_ve_sonsuz_sonuc_uretmez(name):
    values = np.tile([1e308, -1e308], 256)
    if name == "time_underwater":
        values = np.abs(values)
    with np.errstate(all="raise"):
        result = _call(name, values)
    if result is not None:
        parts = result.values() if isinstance(result, dict) else [result]
        assert all(np.all(np.isfinite(value)) for value in parts)


# ---------------------------------------------------------------------------
# Sualtı ve getiri oranları
# ---------------------------------------------------------------------------
def test_time_underwater_tamamlanan_ve_acik_diziler():
    result = qm.time_underwater([100, 90, 95, 100, 105, 104, 105, 103, 102, 101])
    assert result == {"max_bar": 3, "ortalama_bar": 2.0, "mevcut_bar": 3, "sualti_orani": 0.6}
    assert type(result["max_bar"]) is int and type(result["mevcut_bar"]) is int


@pytest.mark.parametrize("equity", [[1, 1], [1, 2, 3, 4]])
def test_time_underwater_hic_sualti_yok(equity):
    assert qm.time_underwater(equity) == {
        "max_bar": 0, "ortalama_bar": 0.0, "mevcut_bar": 0, "sualti_orani": 0.0,
    }


def test_time_underwater_zirveye_esitlik_sureyi_bitirir():
    assert qm.time_underwater([5, 4, 3, 5]) == {
        "max_bar": 2, "ortalama_bar": 2.0, "mevcut_bar": 0, "sualti_orani": 0.5,
    }


@pytest.mark.parametrize("equity", [[1], [1, 0], [1, -1]])
def test_time_underwater_dejenere(equity):
    assert qm.time_underwater(equity) is None


def test_gain_to_pain_net_toplam_kar_faktorunden_bir_eksik():
    returns = [-2, 1, 3, -1]
    assert qm.gain_to_pain(returns) == pytest.approx(1 / 3)
    assert qm.profit_factor(returns) == pytest.approx(4 / 3)
    assert qm.gain_to_pain([-0.1]) == -1.0
    assert qm.gain_to_pain([-1, 1]) == 0.0


@pytest.mark.parametrize("returns", [[0], [1, 2], []])
def test_gain_to_pain_kayip_yoksa_none(returns):
    assert qm.gain_to_pain(returns) is None


@pytest.mark.parametrize("q,expected", [(0.5, 1.0), (0.75, 1.5), (0.95, 1.5), (1.0, 1.5)])
def test_tail_ratio_lineer_kuantiller_ve_sinirlar(q, expected):
    assert qm.tail_ratio([-4, -2, 1, 3, 6], q=q) == pytest.approx(expected)


def test_tail_ratio_iki_kuyrukta_da_mutlak_deger():
    assert qm.tail_ratio([-4, -3, -2, -1], q=1) == 0.25
    assert qm.tail_ratio([1, 2], q=1) == 2.0
    assert qm.tail_ratio([0, 0]) is None
    assert qm.tail_ratio([1]) is None


@pytest.mark.parametrize("q", [0.49, 1.01, np.nan, np.inf, None, "gecersiz", [], 10**1000])
def test_tail_ratio_gecersiz_q(q):
    assert qm.tail_ratio([-2, 1, 3], q=q) is None


# ---------------------------------------------------------------------------
# Entropiler — kapalı form ve iki yönlü Shannon limiti
# ---------------------------------------------------------------------------
def test_entropiler_bilinen_esit_olmayan_olasiliklar():
    values = [0, 1, 1, 2]  # p=(1/4,1/2,1/4), üç eşit genişlikli bölme.
    assert qm.shannon_entropy(values, bins=3) == pytest.approx(1.5 * math.log(2))
    assert qm.renyi_entropy(values, bins=3) == pytest.approx(-math.log(3 / 8))
    assert qm.tsallis_entropy(values, bins=3) == pytest.approx(5 / 8)


@pytest.mark.parametrize("alpha", [0.5, 2.0, 12.0, 500.0])
def test_duzgun_dagilimda_renyi_ln_bolme_sayisi(alpha):
    assert qm.renyi_entropy(np.arange(20), alpha=alpha) == pytest.approx(math.log(20))


def test_duzgun_dagilim_shannon_ve_tsallis():
    assert qm.shannon_entropy(np.arange(20)) == pytest.approx(math.log(20))
    assert qm.tsallis_entropy(np.arange(20)) == pytest.approx(19 / 20)


def test_bos_histogram_bolmeleri_atlanir():
    values = [0] * 4 + [1] * 4
    assert qm.shannon_entropy(values, bins=4) == pytest.approx(math.log(2))
    assert qm.renyi_entropy(values, bins=4) == pytest.approx(math.log(2))
    assert qm.tsallis_entropy(values, bins=4) == pytest.approx(0.5)
    assert qm.tsallis_entropy(values, q=0, bins=4) == pytest.approx(1.0)


@pytest.mark.parametrize("delta", [-1e-7, 0.0, 1e-7])
def test_renyi_ve_tsallis_tam_shannon_limiti(delta):
    values = [0, 1, 1, 2]
    shannon = qm.shannon_entropy(values, bins=3)
    assert qm.renyi_entropy(values, alpha=1 + delta, bins=3) == shannon
    assert qm.tsallis_entropy(values, q=1 + delta, bins=3) == shannon


@pytest.mark.parametrize("side", [-1, 1])
def test_limite_gecis_dalinin_disinda_da_shannona_yakinsar(side):
    values = [0, 1, 1, 2]
    shannon = qm.shannon_entropy(values, bins=3)
    for fn, keyword in ((qm.renyi_entropy, "alpha"), (qm.tsallis_entropy, "q")):
        far = fn(values, bins=3, **{keyword: 1 + side * 1e-2})
        near = fn(values, bins=3, **{keyword: 1 + side * 1e-5})
        assert shannon is not None and far is not None and near is not None
        assert abs(near - shannon) < abs(far - shannon) / 100
        assert near == pytest.approx(shannon, abs=1e-5)


@pytest.mark.parametrize("fn", ENTROPIES)
@pytest.mark.parametrize("bins", [0, 1, 2.5, 100, np.nan, np.inf, "gecersiz", None, 10**1000])
def test_entropiler_gecersiz_bins(fn, bins):
    assert fn(np.arange(20), bins=bins) is None


@pytest.mark.parametrize("fn", ENTROPIES)
def test_entropiler_kisa_ve_sabit(fn):
    assert fn(np.arange(19)) is None
    assert fn(np.ones(100)) is None


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf, "gecersiz", None, 10**1000])
def test_renyi_gecersiz_alpha(alpha):
    assert qm.renyi_entropy(np.arange(20), alpha=alpha) is None


@pytest.mark.parametrize("q", [np.nan, np.inf, "gecersiz", None, 10**1000])
def test_tsallis_gecersiz_q(q):
    assert qm.tsallis_entropy(np.arange(20), q=q) is None


# ---------------------------------------------------------------------------
# Kritik yavaşlama — pozitif kontrol ve iki bağımsız negatif kontrol
# ---------------------------------------------------------------------------
def test_kritik_yavaslama_monoton_tau_ve_analitik_p():
    # Kayan karesel seride korelasyon ve varyans kesin artar: tau=1, p=2/n!.
    result = qm.critical_slowing_down(np.arange(60, dtype=float)**2, window=10, step=1)
    assert result is not None
    assert result["ar1_tau"] == 1.0 and result["var_tau"] == 1.0
    assert result["ar1_p"] == pytest.approx(2 / math.factorial(51), rel=1e-12, abs=0)
    assert result["var_p"] == pytest.approx(2 / math.factorial(51), rel=1e-12, abs=0)
    assert result["n_pencere"] == 51
    assert result["devrilme"] is True


def test_yalniz_varyans_artisi_devrilme_degildir():
    values = np.random.default_rng(SEED).standard_normal(1000) * np.linspace(1, 10, 1000)
    result = qm.critical_slowing_down(values)
    assert result is not None
    assert result["var_tau"] > 0 and result["var_p"] < 0.05
    assert result["ar1_tau"] < 0
    assert result["devrilme"] is False


def test_yalniz_ar1_artisi_devrilme_degildir():
    result = qm.critical_slowing_down(np.sqrt(np.arange(1, 241)), window=20)
    assert result is not None
    assert result["ar1_tau"] > 0 and result["ar1_p"] < 0.05
    assert result["var_tau"] < 0
    assert result["devrilme"] is False


def test_iki_pozitif_ama_anlamsiz_trend_devrilme_degildir():
    result = qm.critical_slowing_down(np.random.default_rng(SEED).standard_normal(200), step=1)
    assert result is not None
    assert result["ar1_tau"] > 0 and result["var_tau"] > 0
    assert result["ar1_p"] > 0.05 and result["var_p"] > 0.05
    assert result["devrilme"] is False


def test_sabit_pencereler_ve_sabit_lag_dilimi_atlanir():
    values = np.r_[np.zeros(20), np.arange(1, 41, dtype=float)**2]
    result = qm.critical_slowing_down(values, window=10, step=1)
    assert result is not None
    assert result["n_pencere"] == 39  # 51 - 11 düz pencere - 1 sabit sol lag dilimi.
    assert result["devrilme"] is True
    assert qm.critical_slowing_down(np.r_[np.zeros(95), [1, 4, 9, 16, 25]],
                                    window=10, step=1) is None


@pytest.mark.parametrize("window", [0, 9, 10.5, 101, np.nan, np.inf, "gecersiz", None, 10**1000])
def test_kritik_yavaslama_gecersiz_pencere(window):
    assert qm.critical_slowing_down(np.arange(200, dtype=float)**2, window=window) is None


def test_kritik_yavaslama_sabit_tanimsiz_ve_kisa():
    assert qm.critical_slowing_down(np.ones(200)) is None
    assert qm.critical_slowing_down(np.tile([-1, 1], 100)) is None
    assert qm.critical_slowing_down(np.arange(99)) is None


# ---------------------------------------------------------------------------
# MF-DFA — aynı n, beyaz gürültü ve iki-terimli kaskad
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def multifractal_samples():
    white = np.random.default_rng(SEED).standard_normal(8192)
    cascade = np.array([1.0])
    for _ in range(13):
        cascade = np.kron(cascade, [0.3, 0.7])
    assert white.size == cascade.size == 8192
    return white, cascade


def test_mfdfa_beyaz_gurultu_ve_kaskad_dogrulama_hedefleri(multifractal_samples):
    white, cascade = (qm.mfdfa(values) for values in multifractal_samples)
    assert white is not None and cascade is not None
    assert white["h2"] == pytest.approx(0.5, abs=0.10)
    assert white["genislik"] < 0.5
    assert cascade["genislik"] > white["genislik"] + 0.5
    assert cascade["genislik"] > 2 * white["genislik"]
    assert white["q"] == np.arange(-5, 5.5, 0.5).tolist()
    assert white["h2"] == white["h"][white["q"].index(2.0)]


def test_mfdfa_kaskadin_analitik_legendre_spektrumu(multifractal_samples):
    # Binomial ölçünün tau(q)=-log2(p**q+(1-p)**q). Sonlu ölçek etkisi h'yi sabit
    # kaydırabilir; Legendre spektrumu ve genişlik bu kaymaya değişmezdir.
    qs = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    result = qm.mfdfa(multifractal_samples[1], q_list=qs, scales=2**np.arange(4, 12))
    assert result is not None
    theory_tau = -np.log2(0.3**qs + 0.7**qs)
    theory_alpha = np.gradient(theory_tau, qs)
    theory_spectrum = qs * theory_alpha - theory_tau
    assert result["genislik"] == pytest.approx(np.ptp(theory_alpha), abs=1e-8)
    np.testing.assert_allclose(result["f_alpha"], theory_spectrum, atol=1e-8)
    theory_h0 = -math.log(0.3 * 0.7) / (2 * math.log(2))
    theory_h2 = (1 - math.log2(0.3**2 + 0.7**2)) / 2
    assert result["h"][2] - result["h2"] == pytest.approx(theory_h0 - theory_h2, abs=1e-8)


def test_mfdfa_sabit_dalgalanmanin_kapali_formu():
    # Çift uzunlukta her pencerede profil 1,0,1,0...; sabit trend sonrası F2=1/4.
    # Her q için Fq=1/2, dolayısıyla h=0, tau=-1, alpha=0, f=1.
    result = qm.mfdfa(np.tile([1, -1], 256), q_list=[-2, 0, 2], scales=[16, 24, 32, 48, 64], order=0)
    assert result is not None
    np.testing.assert_allclose(result["h"], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["tau"], -1.0, atol=1e-12)
    np.testing.assert_allclose(result["alpha"], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["f_alpha"], 1.0, atol=1e-12)
    assert result["h2"] == pytest.approx(0.0, abs=1e-12)
    assert result["genislik"] == pytest.approx(0.0, abs=1e-12)


def test_mfdfa_ozel_q_siralanir_h2_ayrica_hesaplanir():
    values = np.random.default_rng(SEED).standard_normal(512)
    full = qm.mfdfa(values)
    custom = qm.mfdfa(values, q_list=[4, 0, -2, 0])
    assert full is not None and custom is not None
    assert custom["q"] == [-2.0, 0.0, 4.0]
    assert custom["h2"] == full["h2"]


def test_mfdfa_geri_pencereler_sondaki_artigi_olcume_katar():
    values = np.random.default_rng(SEED).standard_normal(260)
    changed = values.copy()
    changed[-4:] += [10, -10, 10, -10]
    # Bu dört ölçeğin ileri bölmeleri 256'da biter; son dört barı yalnız geri
    # bölmeler görür. Eklemenin toplamı sıfırdır, genel ortalamayı değiştirmez.
    before = qm.mfdfa(values, scales=[16, 32, 64, 128])
    after = qm.mfdfa(changed, scales=[16, 32, 64, 128])
    assert before is not None and after is not None
    assert abs(before["h2"] - after["h2"]) > 0.01


def test_mfdfa_sifir_f2_olan_olcegi_dusurur():
    values = np.r_[np.zeros(16), np.tile([1.0, -1.0], 248)]
    without_zero = qm.mfdfa(values, scales=[32, 48, 64, 96])
    assert without_zero is not None
    assert qm.mfdfa(values, scales=[16, 32, 48, 64, 96]) == without_zero
    assert qm.mfdfa(values, scales=[16, 32, 64, 96]) is None  # Yalnız üç geçerli ölçek.


@pytest.mark.parametrize("kwargs", [
    {"q_list": [2]}, {"q_list": [2, 2]}, {"q_list": [-1, np.nan, 2]},
    {"q_list": "gecersiz"}, {"scales": [16, 32, 64]}, {"scales": [16, 16, 32, 32]},
    {"scales": [1, 2, 3, 1024]}, {"scales": [16, 20, 32, np.inf]},
    {"scales": [16, 20.5, 32, 64]}, {"scales": "gecersiz"},
    {"order": -1}, {"order": 1.5}, {"order": np.inf}, {"order": "gecersiz"}, {"order": 10**1000},
])
def test_mfdfa_gecersiz_parametreler(kwargs):
    assert qm.mfdfa(np.random.default_rng(SEED).standard_normal(512), **kwargs) is None


def test_mfdfa_kisa_ve_sifir_dalgalanma():
    assert qm.mfdfa(np.arange(99)) is None
    assert qm.mfdfa(np.ones(512)) is None
    assert qm.mfdfa(np.zeros(512)) is None


# ---------------------------------------------------------------------------
# NSS — beta ve tau geri kazanımı
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def nss_curve():
    maturities = np.array([1/12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    betas = np.array([0.04, -0.025, 0.03, -0.015])
    x1, x2 = maturities / 2.0, maturities / 5.0
    l1, l2 = (1 - np.exp(-x1)) / x1, (1 - np.exp(-x2)) / x2
    yields = betas[0] + betas[1] * l1 + betas[2] * (l1 - np.exp(-x1)) + betas[3] * (l2 - np.exp(-x2))
    return maturities, yields, betas


@pytest.mark.parametrize("taus", [{}, {"tau1": 2.0}, {"tau2": 5.0}, {"tau1": 2.0, "tau2": 5.0}])
def test_nss_sentetik_egri_tam_geri_kazanim(nss_curve, taus):
    maturities, yields, betas = nss_curve
    result = qm.nss_fit(maturities, yields, **taus)
    assert result is not None
    assert result["rmse"] < 1e-6
    assert result["tau1"] == 2.0 and result["tau2"] == 5.0
    np.testing.assert_allclose([result[f"beta{i}"] for i in range(4)], betas, atol=1e-4, rtol=0)
    np.testing.assert_allclose(result["fit"], yields, atol=1e-6, rtol=0)
    assert result["seviye"] == result["beta0"]
    assert result["egim"] == -result["beta1"]
    assert result["egrilik"] == result["beta2"]


def test_nss_vade_sirasi_ve_girdiler_korunur(nss_curve):
    maturities, yields, _ = nss_curve
    permutation = np.random.default_rng(SEED).permutation(maturities.size)
    ms, ys = maturities[permutation], yields[permutation]
    ms_before, ys_before = ms.copy(), ys.copy()
    result = qm.nss_fit(ms, ys, tau1=2, tau2=5)
    assert result is not None
    np.testing.assert_allclose(result["fit"], ys, atol=1e-12)
    np.testing.assert_array_equal(ms, ms_before)
    np.testing.assert_array_equal(ys, ys_before)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, "gecersiz", 10**1000])
def test_nss_gecersiz_vade_ve_tau(nss_curve, bad):
    maturities, yields, _ = nss_curve
    ms = maturities.tolist()
    ms[2] = bad
    assert qm.nss_fit(ms, yields) is None
    assert qm.nss_fit(maturities, yields, tau1=bad) is None
    assert qm.nss_fit(maturities, yields, tau2=bad) is None


def test_nss_eksik_veri_ve_rank(nss_curve):
    maturities, yields, _ = nss_curve
    assert qm.nss_fit(maturities[:3], yields[:3]) is None
    assert qm.nss_fit(maturities[:-1], yields) is None
    assert qm.nss_fit(maturities, yields, tau1=2, tau2=2) is None
    assert qm.nss_fit(np.ones(12), yields) is None


# ---------------------------------------------------------------------------
# Kalıntı momentum — ortogonal Fourier faktörleriyle bilinen OLS çözümü
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("skip", [0, 8])
def test_residual_momentum_analitik_ols_ve_momentum(k, skip):
    n = 48
    t = 2 * np.pi * np.arange(n) / n
    factors = 0.01 * np.column_stack((np.cos(t), np.sin(t)))[:, :k]
    betas = np.array([1.2, -0.7])[:k]
    errors = 0.005 * np.cos(2 * t)
    asset = 0.0002 + factors @ betas + errors
    passed_factors = factors[:, 0] if k == 1 else factors
    before = passed_factors.copy()
    result = qm.residual_momentum(asset, passed_factors, lookback=n, skip=skip)
    assert result is not None
    expected_errors = errors[:-skip] if skip else errors
    expected_sum = float(expected_errors.sum())
    expected_std = float(expected_errors.std(ddof=1))
    assert result["alpha"] == pytest.approx(0.0002, abs=1e-12)
    np.testing.assert_allclose(result["beta"], betas, atol=1e-12)
    assert result["resid_mom"] == pytest.approx(expected_sum, abs=1e-12)
    assert result["resid_vol"] == pytest.approx(expected_std, abs=1e-12)
    assert result["resid_mom_z"] == pytest.approx(
        expected_sum / (expected_std * math.sqrt(n - skip)), abs=1e-12,
    )
    explained = 0.01**2 * np.sum(betas**2)
    assert result["r2"] == pytest.approx(explained / (explained + 0.005**2), abs=1e-12)
    assert result["n"] == n
    np.testing.assert_array_equal(passed_factors, before)


def test_residual_momentum_yalniz_son_lookback_ile_ols():
    rng = np.random.default_rng(SEED)
    factors = rng.normal(0, 0.01, (100, 2))
    asset = 0.002 + factors @ [1.2, -0.4] + rng.normal(0, 0.005, 100)
    expected = qm.residual_momentum(asset, factors, lookback=100, skip=10)
    actual = qm.residual_momentum(
        np.r_[np.full(20, 1e3), asset], np.vstack((np.full((20, 2), -1e3), factors)),
        lookback=100, skip=10,
    )
    assert actual is not None and actual == expected


@pytest.mark.parametrize("kwargs", [
    {"lookback": 513}, {"lookback": 31, "skip": 21}, {"lookback": 32.5},
    {"lookback": np.nan}, {"lookback": "gecersiz"}, {"lookback": 10**1000},
    {"skip": -1}, {"skip": 1.5}, {"skip": np.inf}, {"skip": "gecersiz"},
])
def test_residual_momentum_gecersiz_pencereler(kwargs):
    assert qm.residual_momentum(np.sin(np.arange(512)), np.cos(np.arange(512)), **kwargs) is None


def test_residual_momentum_dejenere_faktor_ve_sifir_kalinti():
    rng = np.random.default_rng(SEED)
    factor = rng.normal(0, 0.01, 300)
    asset = 0.002 + 1.2 * factor
    assert qm.residual_momentum(asset, factor) is None
    assert qm.residual_momentum(np.zeros(300), factor) is None
    assert qm.residual_momentum(asset, factor[:-1]) is None
    assert qm.residual_momentum(asset, np.zeros((300, 1, 1))) is None
    assert qm.residual_momentum(asset, np.empty((300, 0))) is None
    for bad in (np.nan, np.inf, "gecersiz", 10**1000):
        broken = factor.tolist()
        broken[5] = bad  # OLS penceresinin dışındaki NaN/Inf de reddedilmeli.
        assert qm.residual_momentum(asset, broken) is None


# ---------------------------------------------------------------------------
# Kaldıraç erimesi — tam iki günlük çözüm, gider ve kalıcı sıfırlanma
# ---------------------------------------------------------------------------
def test_leveraged_etf_iki_gun_kapali_form():
    result = qm.leveraged_etf_drag([0.05, -0.05], expense=0)
    assert result is not None
    assert result["bilesik_idx"] == pytest.approx(-0.0025)
    assert result["bilesik_lev"] == pytest.approx(-0.0225)
    assert result["naif"] == pytest.approx(-0.0075)
    assert result["suruklenme"] == pytest.approx(-0.015)
    assert result["teorik_drag"] == pytest.approx(-0.015)
    assert result["yillik_vol"] == pytest.approx(0.05 * math.sqrt(2 * 252))
    assert result["n"] == 2 and result["sifirlandi"] is False


def test_leveraged_etf_salinimda_negatif_ve_ayni_mertebe():
    result = qm.leveraged_etf_drag(np.tile([0.05, -0.05], 10), beta=3, expense=0)
    assert result is not None
    assert result["suruklenme"] < 0 and result["teorik_drag"] < 0
    assert 0.5 < result["suruklenme"] / result["teorik_drag"] < 2.0
    assert result["bilesik_lev"] == pytest.approx((1 - 0.15**2)**10 - 1)
    assert result["bilesik_idx"] == pytest.approx((1 - 0.05**2)**10 - 1)
    assert result["teorik_drag"] == pytest.approx(-0.15)


def test_leveraged_etf_bir_kat_kaldirac_ve_yillik_gider():
    returns = [0.01, -0.02, 0.03, -0.01]
    one = qm.leveraged_etf_drag(returns, beta=1, expense=0)
    assert one is not None
    assert one["bilesik_lev"] == one["bilesik_idx"]
    assert one["suruklenme"] == 0.0 and one["teorik_drag"] == 0.0
    fee = qm.leveraged_etf_drag(np.zeros(10), expense=0.01, ppy=10)
    assert fee is not None
    assert fee["bilesik_lev"] == pytest.approx(0.999**10 - 1)
    assert fee["yillik_vol"] == 0.0 and fee["teorik_drag"] == 0.0


@pytest.mark.parametrize("returns", [[-1/3, 0.5, 0.5], [-0.4, -0.4, 0.5], [0.1, -0.4, 1.0]])
def test_leveraged_etf_sifirlaninca_tekrar_canlanmaz(returns):
    result = qm.leveraged_etf_drag(returns, expense=0)
    assert result is not None
    assert result["sifirlandi"] is True and result["bilesik_lev"] == -1.0
    assert result["suruklenme"] == -1.0 - result["naif"]


@pytest.mark.parametrize("kwargs", [
    {"beta": 0}, {"beta": -1}, {"beta": np.nan}, {"beta": np.inf}, {"beta": "gecersiz"},
    {"beta": 10**1000}, {"ppy": 0}, {"ppy": -1}, {"ppy": np.nan}, {"ppy": "gecersiz"},
    {"expense": np.nan}, {"expense": np.inf}, {"expense": "gecersiz"},
])
def test_leveraged_etf_gecersiz_parametreler(kwargs):
    assert qm.leveraged_etf_drag([0.01, -0.01], **kwargs) is None


def test_leveraged_etf_kisa():
    assert qm.leveraged_etf_drag([0.01]) is None


def test_kritik_yavaslama_VARSAYILAN_ortusmeyen_ve_yanlis_pozitif_uretmiyor():
    """Tur-73 kusuru: Kendall p'si BAĞIMSIZ gözlem varsayar; örtüşen pencereler onu
    anti-muhafazakâr yapar. Ölçüldü (durağan AR(1)=0,2, 40 tekrar, n=1200, window=100):
    step=1 yanlış-pozitif ~%18, varsayılan (örtüşmeyen) %0 — güç kaybı YOK."""
    n, tekrar = 1200, 40
    yp_ortusen = yp_varsayilan = 0
    for s in range(tekrar):
        rng = np.random.default_rng(4100 + s)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.2 * x[i - 1] + rng.normal()
        yp_ortusen += qm.critical_slowing_down(x, 100, step=1)["devrilme"]
        yp_varsayilan += qm.critical_slowing_down(x, 100)["devrilme"]
    assert yp_ortusen >= 4, "örtüşen kip artık yanlış-pozitif üretmiyorsa kurgu bozulmuş"
    assert yp_varsayilan == 0, f"örtüşmeyen kip {yp_varsayilan} yanlış-pozitif verdi"

    # GÜÇ: gerçek bir devrilme varsayılan ayarda da yakalanmalı (aksi halde düzeltme,
    # kusuru testi kırarak 'çözerdi').
    yakalanan = 0
    for s in range(10):
        rng = np.random.default_rng(4200 + s)
        x = np.zeros(n)
        phis, sds = np.linspace(0.05, 0.95, n), np.linspace(0.5, 2.0, n)
        for i in range(1, n):
            x[i] = phis[i] * x[i - 1] + rng.normal(0, sds[i])
        yakalanan += qm.critical_slowing_down(x, 100)["devrilme"]
    assert yakalanan == 10


def test_kritik_yavaslama_adim_alanlari_ve_gecersiz_adim():
    z = np.random.default_rng(7).standard_normal(1200)
    var = qm.critical_slowing_down(z, 100)
    yogun = qm.critical_slowing_down(z, 100, step=10)
    assert var["adim"] == 100 and var["ortusen"] is False and var["n_pencere"] == 12
    assert yogun["adim"] == 10 and yogun["ortusen"] is True
    assert yogun["n_pencere"] > var["n_pencere"]
    assert qm.critical_slowing_down(z, 100, step=0) is None
    assert qm.critical_slowing_down(z, 100, step=-5) is None


# ---------------------------------------------------------------------------
# Tur-73 DÜZELTME KİLİTLERİ — üç ölçülmüş kusurun geri gelmemesi için
# ---------------------------------------------------------------------------
def test_nss_dejenere_beta0_reddediliyor_ve_kisitsiz_raporlaniyor():
    """CANLI ABD eğrisinde ölçüldü: salt-SSR kazananı son 500 günün %98'inde b0'ı gözlenen
    getiri aralığının DIŞINA atıyordu. Uyum artık [min(y),max(y)] içindeki en iyi adaydır."""
    mats = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    ys = np.array([3.83, 3.89, 3.95, 4.11, 4.34, 4.41, 4.52, 4.63, 4.77, 5.25, 5.25])
    r = qm.nss_fit(mats, ys)
    assert r is not None
    pay = max(0.5, 0.5 * float(ys.max() - ys.min()))     # b0 bir ASİMPTOT: bant taşırılır
    assert ys.min() - pay <= r["beta0"] <= ys.max() + pay, "b0 kabul bandının dışında"
    assert r["seviye"] == r["beta0"]
    assert r["dejenere"] is False
    # Kısıtsız uyum bu eğride GERÇEKTEN bandın dışındaydı — kusur kurgu değil, ölçüm.
    assert r["beta0_kisitsiz"] < ys.min() - pay
    assert r["beta0_kisitsiz"] < 0.0                     # 30Y %5,25 iken b0 = −2,08
    assert r["rmse_kisitsiz"] <= r["rmse"] + 1e-12       # kısıtsız her zaman >= iyi uyar
    assert r["rmse"] < 0.10                              # kabul edilebilir uyum yine de iyi


def test_nss_sentetik_geri_kazanim_kisitla_BOZULMADI():
    """Kısıt, gerçek b0 zaten aralık içindeyken devreye girmemeli (regresyon koruması)."""
    mats = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    b0, b1, b2, b3, t1, t2 = 4.5, -1.2, 2.0, -0.8, 2.0, 5.0
    x1, x2 = mats / t1, mats / t2
    l1 = (1 - np.exp(-x1)) / x1
    ys = b0 + b1 * l1 + b2 * (l1 - np.exp(-x1)) + b3 * ((1 - np.exp(-x2)) / x2 - np.exp(-x2))
    r = qm.nss_fit(mats, ys)
    assert r["rmse"] < 1e-9 and r["dejenere"] is False
    assert r["beta0"] == pytest.approx(b0, abs=1e-4)
    assert r["beta3"] == pytest.approx(b3, abs=1e-4)


def test_letf_log_suruklenme_cheng_madhavan_ile_ORTUSUYOR():
    """`teorik_drag` LOG uzayındadır; `suruklenme` ARİTMETİK. Karşılaştırılacak büyüklük
    `log_suruklenme`'dir ve yaklaşımla ±%5 içinde örtüşmelidir."""
    for r in (np.tile([0.05, -0.05], 126), np.full(252, 0.001),
              np.random.default_rng(SEED).normal(0.0004, 0.012, 252)):
        o = qm.leveraged_etf_drag(r, beta=3.0, expense=0.0)
        assert o is not None and o["log_suruklenme"] is not None
        assert o["log_suruklenme"] / o["teorik_drag"] == pytest.approx(1.0, abs=0.05)
        assert o["log_suruklenme"] < 0.0                 # log sürüklenme HER ZAMAN negatif


def test_letf_aritmetik_suruklenme_TRENDDE_POZITIF_OLABILIR():
    """'Vol drag her zaman negatiftir' ARİTMETİK ölçüde YANLIŞTIR — düz trendde 3x
    bileşik, naif 3 katı GEÇER. İki ölçü aynı seride ZIT işaret taşır."""
    o = qm.leveraged_etf_drag(np.full(252, 0.001), beta=3.0, expense=0.0)
    assert o["suruklenme"] > 0.0 and o["log_suruklenme"] < 0.0
    # Salınan seride ikisi de negatif (asıl erime rejimi).
    o2 = qm.leveraged_etf_drag(np.tile([0.05, -0.05], 126), beta=3.0, expense=0.0)
    assert o2["suruklenme"] < 0.0 and o2["log_suruklenme"] < 0.0


def test_letf_sifirlanmis_etfde_log_suruklenme_None():
    """ln(0) tanımsız → uydurma sayı basılmaz (ölçüm yokluğu ölçüm gibi sunulamaz)."""
    o = qm.leveraged_etf_drag([-0.40, 0.10, 0.10], beta=3.0, expense=0.0)
    assert o["sifirlandi"] is True and o["bilesik_lev"] == -1.0
    assert o["log_suruklenme"] is None
