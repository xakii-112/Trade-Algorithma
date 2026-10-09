"""nihai.ai.quant_math testleri — bilinen-değer çapaları + savunmacı (kısa/NaN) girdi yolları.

Ağ/IO yok (saf fonksiyonlar). py_vollib YALNIZ burada, opsiyonel çapraz-doğrulama için
(kurulu değilse ilgili test atlanır — modülün KENDİSİ py_vollib'e bağımlı değildir).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nihai.ai import quant_math as qm

try:
    from py_vollib.black_scholes import black_scholes as _vollib_bs
except ImportError:  # pragma: no cover - opsiyonel bağımlılık
    _vollib_bs = None


# ---------------------------------------------------------------------------
# Getiriler / normalizasyon
# ---------------------------------------------------------------------------
def test_simple_returns_known_values():
    out = qm.simple_returns([100, 110, 121])
    assert out == pytest.approx([0.10, 0.10])


def test_simple_returns_short_input_bos_dizi():
    assert qm.simple_returns([100]).size == 0
    assert qm.simple_returns([]).size == 0


def test_log_returns_known_values():
    out = qm.log_returns([100, 100 * math.e])
    assert out == pytest.approx([1.0])


def test_annualize_return_bilinen():
    assert qm.annualize_return(0.0, 252) == 0.0
    assert qm.annualize_return(0.01, 252) == pytest.approx(1.01**252 - 1.0)
    assert qm.annualize_return(0.01, 0) == 0.0   # ppy<=0 -> 0.0, çökme yok


def test_annualize_vol_bilinen():
    assert qm.annualize_vol(0.01, 252) == pytest.approx(0.01 * math.sqrt(252))
    assert qm.annualize_vol(-1.0, 252) == 0.0    # negatif vol -> 0.0


def test_zscore_tam_orneklem_bilinen():
    out = qm.zscore([1, 2, 3, 4, 5])
    assert out == pytest.approx([-1.41421356, -0.70710678, 0.0, 0.70710678, 1.41421356], abs=1e-6)


def test_zscore_pencereli_bilinen():
    out = qm.zscore([1, 2, 3, 4, 5, 6], window=3)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2:] == pytest.approx([math.sqrt(1.5)] * 4, abs=1e-6)


def test_zscore_sabit_dizi_sifira_kirpilir():
    assert qm.zscore([5, 5, 5]) == pytest.approx([0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Oynaklık tahmincileri
# ---------------------------------------------------------------------------
def test_vol_close_to_close_kisa_none():
    assert qm.vol_close_to_close([100, 101]) is None
    assert qm.vol_close_to_close([100, 101, 99]) is not None


def test_vol_parkinson_bilinen_sabit_seri():
    """H/L = e olacak şekilde kurgulanmış seri -> Parkinson sabiti 1/sqrt(4 ln2) (bağımsız
    matematiksel referans; quant_math'i ÇAĞIRMADAN hesaplanır)."""
    high = [100 * math.e, 100 * math.e]
    low = [100.0, 100.0]
    expected = math.sqrt(1.0 / (4.0 * math.log(2.0)))
    assert qm.vol_parkinson(high, low) == pytest.approx(expected)


def test_vol_garman_klass_bilinen_sabit_seri():
    """O=C, H/L=e -> GK terimi 0.5*1^2 - 0 = 0.5 -> sqrt(0.5) (bağımsız referans)."""
    o = [100.0, 100.0]
    h = [100 * math.e, 100 * math.e]
    low = [100.0, 100.0]
    c = [100.0, 100.0]
    assert qm.vol_garman_klass(o, h, low, c) == pytest.approx(math.sqrt(0.5))


def test_vol_garman_klass_uyumsuz_uzunluk_none():
    assert qm.vol_garman_klass([100, 100], [101], [99, 99], [100, 100]) is None


def test_vol_yang_zhang_kisa_girdi_none_ve_pozitif():
    assert qm.vol_yang_zhang([100, 101], [102, 103], [99, 100], [101, 102]) is None
    rng = np.random.default_rng(9)
    n = 40
    o = 100 + np.cumsum(rng.normal(0, 0.2, n))
    c = o + rng.normal(0, 0.3, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.2, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.2, n))
    v = qm.vol_yang_zhang(o, h, lo, c)
    assert v is not None and v >= 0.0


def test_ewma_vol_bilinen():
    assert qm.ewma_vol([0.1, -0.1], lam=0.5) == pytest.approx(0.1)
    assert qm.ewma_vol([0.1], lam=1.5) is None   # lam disi aralik


# ---------------------------------------------------------------------------
# Risk / kayıp metrikleri
# ---------------------------------------------------------------------------
def test_var_cvar_historical_bilinen():
    rets = [-0.05, -0.03, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10]
    assert qm.var_historical(rets, alpha=0.1) == pytest.approx(0.032)
    assert qm.cvar_historical(rets, alpha=0.1) == pytest.approx(0.05)
    # CVaR (kuyruk ortalaması) her zaman VaR (tek yüzdelik) BÜYÜKLÜĞÜNDEN >= olmalı
    assert qm.cvar_historical(rets, alpha=0.1) >= qm.var_historical(rets, alpha=0.1)


def test_var_historical_gecersiz_alpha_none():
    assert qm.var_historical([0.01, 0.02], alpha=0.0) is None
    assert qm.var_historical([0.01, 0.02], alpha=1.0) is None


def test_var_cornish_fisher_negatif_olmayan_ve_kisa_none():
    rets = [-0.05, -0.03, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10]
    v = qm.var_cornish_fisher(rets, alpha=0.05)
    assert v is not None and v >= 0.0
    assert qm.var_cornish_fisher([0.01, 0.02], alpha=0.05) is None   # <4 nokta


def test_max_drawdown_bilinen_egri():
    dd = qm.max_drawdown([100, 150, 75, 50, 120])
    assert dd is not None
    frac, dur = dd
    assert frac == pytest.approx(2.0 / 3.0)
    assert dur == 2


def test_max_drawdown_monoton_artan_sifir():
    assert qm.max_drawdown([100, 110, 120, 130]) == (0.0, 0)


def test_max_drawdown_kisa_ve_negatif_none():
    assert qm.max_drawdown([100]) is None
    assert qm.max_drawdown([100, -5]) is None


def test_downside_deviation_bilinen():
    dd = qm.downside_deviation([0.1, -0.2, 0.05, -0.1])
    assert dd == pytest.approx(math.sqrt(0.0125))


def test_ulcer_index_bilinen():
    ui = qm.ulcer_index([100, 150, 75, 50, 120])
    expected = math.sqrt(((0.0**2 + 0.0**2 + 0.5**2 + (2.0 / 3.0) ** 2 + 0.2**2)) / 5.0)
    assert ui == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Risk-ayarlı performans oranları
# ---------------------------------------------------------------------------
def test_sharpe_bilinen():
    rets = [0.01, 0.02, -0.01, 0.03, 0.00]
    mean = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    expected = mean / sd * math.sqrt(252)
    assert qm.sharpe(rets, rf=0.0, ppy=252) == pytest.approx(expected)


def test_sortino_kayipsiz_none():
    assert qm.sortino([0.01, 0.02, 0.03]) is None


def test_calmar_sifir_drawdown_none():
    assert qm.calmar([0.01] * 50) is None   # monoton artan -> max_drawdown=0


def test_calmar_normal_durumda_float():
    rng = np.random.default_rng(1)
    rets = rng.normal(0.001, 0.01, 300)
    val = qm.calmar(rets)
    assert val is not None and math.isfinite(val)


def test_omega_bilinen():
    assert qm.omega([0.02, -0.01, 0.03, -0.02]) == pytest.approx(5.0 / 3.0)
    assert qm.omega([0.01, 0.02]) is None   # kayıp yok -> tanımsız


def test_profit_factor_bilinen():
    assert qm.profit_factor([10, -5, 20, -10, 5]) == pytest.approx(35.0 / 15.0)
    assert qm.profit_factor([10, 20]) is None   # kayıp yok -> tanımsız


# ---------------------------------------------------------------------------
# Kelly ailesi
# ---------------------------------------------------------------------------
def test_kelly_binary_elle_hesap():
    assert qm.kelly_binary(0.6, 2.0) == pytest.approx(0.4)


def test_kelly_binary_negatif_kenar_sifira_kirpilir():
    assert qm.kelly_binary(0.4, 1.0) == 0.0
    assert qm.kelly_binary(0.5, -1.0) == 0.0   # geçersiz b


def test_kelly_continuous_bilinen_ve_sifir_varyans():
    assert qm.kelly_continuous(0.02, 0.01) == pytest.approx(2.0)
    assert qm.kelly_continuous(0.02, 0.0) == 0.0


def test_fractional_kelly_cap():
    assert qm.fractional_kelly(0.4, fraction=0.25, cap=0.02) == pytest.approx(0.02)
    assert qm.fractional_kelly(0.04, fraction=0.25, cap=0.02) == pytest.approx(0.01)
    assert qm.fractional_kelly(-1.0, fraction=0.25, cap=0.02) == 0.0


# ---------------------------------------------------------------------------
# Ortalamaya-dönüş / kointegrasyon istatistikleri
# ---------------------------------------------------------------------------
def test_ou_half_life_ar1_yaklasik():
    """phi=0.9 AR(1) sentetik seri -> regresyon-formülünün POPÜLASYON değeri ln2/(1-phi)
    (Chan pairs-trading konvansiyonu; -ln2/b ile b=phi-1)."""
    rng = np.random.default_rng(123)
    n, phi = 2000, 0.9
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = phi * ou[i - 1] + rng.normal(0, 1)
    target = math.log(2.0) / (1.0 - phi)
    hl = qm.ou_half_life(ou)
    assert hl is not None and hl == pytest.approx(target, abs=1.0)


def test_ou_half_life_random_walk_none():
    rng = np.random.default_rng(4)
    rw = np.cumsum(rng.normal(0, 1, 200))
    # rastgele yürüyüşte b>=0 olabilir (mean-reversion yok) -> None BEKLENİR ya da pozitif
    # yarı-ömür üretmez; katı tekil-değer yerine None/pozitif olması davranışı doğrulanır.
    hl = qm.ou_half_life(rw)
    assert hl is None or hl > 0


def test_hurst_exponent_trend_vs_meanreversion_siralama():
    rng = np.random.default_rng(42)
    trend = np.cumsum(rng.normal(0.05, 1.0, 400))
    ou = np.zeros(400)
    for i in range(1, 400):
        ou[i] = 0.5 * ou[i - 1] + rng.normal(0, 1)
    h_trend = qm.hurst_exponent(trend)
    h_meanrev = qm.hurst_exponent(ou)
    assert h_trend is not None and h_meanrev is not None
    assert h_meanrev < 0.45 < h_trend or h_meanrev < h_trend


def test_hurst_exponent_kisa_seri_none():
    assert qm.hurst_exponent(list(range(10))) is None


def test_variance_ratio_random_walk_vs_meanreversion():
    rng = np.random.default_rng(5)
    rw_price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000)))
    vr_rw = qm.variance_ratio(rw_price, 5)
    assert vr_rw is not None and vr_rw == pytest.approx(1.0, abs=0.2)

    ou = np.zeros(2000)
    for i in range(1, 2000):
        ou[i] = 0.3 * ou[i - 1] + rng.normal(0, 0.02)
    mr_price = 100 * np.exp(ou)
    vr_mr = qm.variance_ratio(mr_price, 5)
    assert vr_mr is not None and vr_mr < 0.6


def test_variance_ratio_gecersiz_q_none():
    assert qm.variance_ratio([1, 2, 3, 4, 5], q=1) is None


def test_adf_test_durgun_ve_rastgele_yuruyus():
    if qm._adfuller is None:
        pytest.skip("statsmodels kurulu değil")
    rng = np.random.default_rng(42)
    stationary = rng.normal(0, 1, 300)
    stat_res = qm.adf_test(stationary)
    assert stat_res is not None
    assert stat_res[1] < 0.05   # durağan seride küçük p-değeri (birim-kök reddedilir)

    rw = np.cumsum(rng.normal(0, 1, 300))
    rw_res = qm.adf_test(rw)
    assert rw_res is not None
    assert rw_res[1] > 0.10    # rastgele yürüyüşte birim-kök reddedilemez


def test_adf_test_kisa_girdi_none():
    assert qm.adf_test([1, 2, 3]) is None


def test_engle_granger_sentetik_kointegre_cift_beta_yaklasik_2():
    """y = 2x + OU-gürültü (kointegre çift) -> beta ≈ 2 (spec test çapası)."""
    rng = np.random.default_rng(7)
    n = 300
    x = np.cumsum(rng.normal(0, 1, n))
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.3 * ou[i - 1] + rng.normal(0, 0.5)
    y = 2.0 * x + ou
    result = qm.engle_granger(y, x)
    assert result is not None
    assert result["beta"] == pytest.approx(2.0, abs=0.15)
    assert set(result) == {"beta", "alpha", "adf_stat", "pvalue", "half_life"}


def test_engle_granger_kisa_girdi_none():
    assert qm.engle_granger([1, 2, 3], [1, 2, 3]) is None


def test_kalman_beta_yakinsar():
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(0, 1, n)
    y = 2.0 * x + rng.normal(0, 0.05, n)
    beta = qm.kalman_beta(y, x)
    assert beta is not None and beta.shape == (n,)
    assert float(np.mean(beta[-20:])) == pytest.approx(2.0, abs=0.3)


def test_kalman_beta_uyumsuz_uzunluk_none():
    assert qm.kalman_beta([1, 2, 3], [1, 2]) is None
    assert qm.kalman_beta([1, 2], [1, 2], q=-1.0) is None


# ---------------------------------------------------------------------------
# Kovaryans / risk ayrıştırma
# ---------------------------------------------------------------------------
def test_ledoit_wolf_cov_sekil_ve_simetri():
    retm = np.random.default_rng(3).normal(0, 0.01, (50, 3))
    cov = qm.ledoit_wolf_cov(retm)
    assert cov is not None and cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)


def test_ledoit_wolf_cov_gecersiz_ndim_none():
    assert qm.ledoit_wolf_cov([1, 2, 3]) is None


def test_ewma_cov_sekil_ve_simetri():
    retm = np.random.default_rng(3).normal(0, 0.01, (50, 3))
    cov = qm.ewma_cov(retm)
    assert cov is not None and cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)


def test_pca_risk_aciklanan_varyans_sinirlari():
    retm = np.random.default_rng(3).normal(0, 0.01, (50, 3))
    result = qm.pca_risk(retm, n=2)
    assert result is not None
    assert len(result["explained"]) == 2
    assert all(0.0 <= e <= 1.0 for e in result["explained"])
    assert len(result["loadings"]) == 2


# ---------------------------------------------------------------------------
# Monte Carlo simülasyonu
# ---------------------------------------------------------------------------
def test_mc_gbm_paths_seed_determinizmi():
    p1 = qm.mc_gbm_paths(100, 0.05, 0.2, 30, 50, seed=42)
    p2 = qm.mc_gbm_paths(100, 0.05, 0.2, 30, 50, seed=42)
    assert np.array_equal(p1, p2)
    assert p1.shape == (50, 31)
    assert np.all(p1[:, 0] == 100.0)


def test_mc_gbm_paths_farkli_seed_farkli_sonuc():
    p1 = qm.mc_gbm_paths(100, 0.05, 0.2, 10, 20, seed=1)
    p2 = qm.mc_gbm_paths(100, 0.05, 0.2, 10, 20, seed=2)
    assert not np.array_equal(p1, p2)


def test_mc_gbm_paths_gecersiz_girdi_cokmez():
    out = qm.mc_gbm_paths(100, 0.05, 0.2, -1, 10, seed=1)
    assert out.shape[0] == 10
    out2 = qm.mc_gbm_paths(100, 0.05, 0.2, 10, 0, seed=1)
    assert out2.shape[0] == 0


def test_mc_bootstrap_terminal_seed_determinizmi_ve_anahtarlar():
    rng = np.random.default_rng(3)
    rets = rng.normal(0.001, 0.02, 200)
    b1 = qm.mc_bootstrap_terminal(rets, horizon=20, n_paths=300, seed=1)
    b2 = qm.mc_bootstrap_terminal(rets, horizon=20, n_paths=300, seed=1)
    assert b1 == b2
    assert set(b1) == {"p5", "p50", "p95", "prob_loss"}
    assert b1["p5"] <= b1["p50"] <= b1["p95"]
    assert 0.0 <= b1["prob_loss"] <= 1.0


def test_mc_bootstrap_terminal_kisa_girdi_none_alanli_dict():
    out = qm.mc_bootstrap_terminal([0.01], horizon=10, n_paths=100, seed=1)
    assert out == {"p5": None, "p50": None, "p95": None, "prob_loss": None}


# ---------------------------------------------------------------------------
# Opsiyon fiyatlama
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_vollib_bs is None, reason="py_vollib kurulu değil")
def test_bs_price_py_vollib_capraz_dogrulama():
    for kind, flag in (("c", "c"), ("p", "p")):
        mine = qm.bs_price(100, 100, 1.0, 0.05, 0.2, kind=kind)
        ref = _vollib_bs(flag, 100.0, 100.0, 1.0, 0.05, 0.2)
        assert mine == pytest.approx(ref, abs=1e-8)


def test_bs_price_gecersiz_girdi_none():
    assert qm.bs_price(-1, 100, 1.0, 0.05, 0.2) is None
    assert qm.bs_price(100, 100, 1.0, 0.05, 0.2, kind="x") is None


def test_bs_greeks_call_put_parity_ve_sinirlar():
    call = qm.bs_greeks(100, 100, 1.0, 0.05, 0.2, kind="c")
    put = qm.bs_greeks(100, 100, 1.0, 0.05, 0.2, kind="p")
    assert call is not None and put is not None
    assert call["delta"] == pytest.approx(put["delta"] + 1.0)
    assert 0.0 <= call["delta"] <= 1.0
    assert -1.0 <= put["delta"] <= 0.0
    assert call["gamma"] == pytest.approx(put["gamma"])
    assert call["vega"] == pytest.approx(put["vega"])
    assert call["gamma"] > 0


# ---------------------------------------------------------------------------
# Rejim sınıflandırma
# ---------------------------------------------------------------------------
def test_classify_regime_kisa_veri_unknown():
    out = qm.classify_regime([1, 2, 3])
    assert out == {"trend": None, "vol_bucket": None, "mean_reversion": None, "label": "unknown"}


def test_classify_regime_trend_yukselen():
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0.003, 0.01, 100)))
    out = qm.classify_regime(close)
    assert out["trend"] == "up"
    assert out["label"] == "up_" + out["vol_bucket"]
    assert out["vol_bucket"] in ("low", "mid", "high")
    assert isinstance(out["mean_reversion"], bool)


def test_classify_regime_high_low_ile_calisir():
    rng = np.random.default_rng(11)
    n = 100
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    high = close + np.abs(rng.normal(0, 0.3, n))
    low = close - np.abs(rng.normal(0, 0.3, n))
    out = qm.classify_regime(close, high=high, low=low)
    assert out["trend"] in ("up", "down", "flat")
    assert out["vol_bucket"] in ("low", "mid", "high")


# ---------------------------------------------------------------------------
# Savunmacı sözleşme (kısa/NaN girdi -> None) — birkaç fonksiyon üzerinden konsolide
# ---------------------------------------------------------------------------
def test_savunmaci_nan_girdi_none_doner():
    nanli = [0.01, float("nan"), 0.02, 0.03]
    assert qm.sharpe(nanli) is None
    assert qm.sortino(nanli) is None
    assert qm.var_historical(nanli) is None
    assert qm.max_drawdown([100, float("nan"), 105]) is None
    assert qm.vol_close_to_close([100, float("nan"), 105, 106]) is None


def test_savunmaci_bos_girdi_cokmez():
    assert qm.simple_returns([]).size == 0
    assert qm.log_returns([]).size == 0
    assert qm.zscore([]).size == 0
    assert qm.sharpe([]) is None
    assert qm.kelly_binary(float("nan"), 2.0) == 0.0
    assert qm.mc_bootstrap_terminal([], horizon=5, n_paths=10, seed=1)["p5"] is None


# ===========================================================================
# v3 (Tur-16 WP-M) testleri — GARCH/EVT/kopula/FFD/spektral/Almgren-Chriss/
# entropi/yüksek-moment fonksiyonları. Bilinen-değer çapaları + savunmacı-girdi
# + determinizm. Sayısal toleranslar C:\...\scratchpad\verify_quant_math_v3.py
# ile önceden doğrulandı (aynı seed'lerle burada tekrarlanır).
# ===========================================================================

# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------
def _sentetik_garch11(n, omega, alpha, beta, seed):
    """Test-yardımcısı (üretim kodu DEĞİL): bilinen parametrelerle GARCH(1,1) simülasyonu."""
    rng = np.random.default_rng(seed)
    resid = np.empty(n)
    sigma2 = np.empty(n)
    sigma2[0] = omega / (1.0 - alpha - beta)
    resid[0] = math.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, n):
        sigma2[t] = omega + alpha * resid[t - 1] ** 2 + beta * sigma2[t - 1]
        resid[t] = math.sqrt(sigma2[t]) * rng.standard_normal()
    return resid


def test_garch11_fit_sentetik_parametre_geri_bulma():
    omega_true, alpha_true, beta_true = 1e-5, 0.10, 0.85
    resid = _sentetik_garch11(2000, omega_true, alpha_true, beta_true, seed=11)
    fit = qm.garch11_fit(resid)
    assert fit is not None
    # Persistence (alpha+beta) GARCH MLE'de en iyi-tanımlı büyüklüktür -> sıkı tolerans.
    assert (fit["alpha"] + fit["beta"]) == pytest.approx(alpha_true + beta_true, abs=0.05)
    # alpha/beta tek tek daha gürültülü tahmin edilir (klasik GARCH-MLE ödünleşimi) -> gevşek.
    assert fit["alpha"] == pytest.approx(alpha_true, abs=0.12)
    assert fit["beta"] == pytest.approx(beta_true, abs=0.15)
    assert set(fit) == {"omega", "alpha", "beta", "mean", "loglik", "persistence",
                        "uncond_var", "last_sigma2", "last_resid2", "n"}


def test_garch11_fit_kisa_veri_none():
    assert qm.garch11_fit([0.01, -0.02, 0.03]) is None


def test_garch11_fit_nan_veri_none():
    veri = [0.01, float("nan")] + [0.001] * 60
    assert qm.garch11_fit(veri) is None


def test_garch11_fit_determinizm():
    resid = _sentetik_garch11(500, 1e-5, 0.08, 0.85, seed=3)
    f1 = qm.garch11_fit(resid)
    f2 = qm.garch11_fit(resid)
    assert f1 == f2


def test_garch11_forecast_tek_adim_elle_dogrulama():
    params = {"omega": 1e-6, "alpha": 0.08, "beta": 0.90, "last_sigma2": 2e-4, "last_resid2": 3e-4}
    out = qm.garch11_forecast(params, horizon=1)
    manual = math.sqrt(params["omega"] + params["alpha"] * params["last_resid2"]
                        + params["beta"] * params["last_sigma2"])
    assert out == pytest.approx(manual)


def test_garch11_forecast_uzun_ufuk_kosullu_varyansa_yakinsar():
    params = {"omega": 1e-6, "alpha": 0.05, "beta": 0.90, "last_sigma2": 5e-4, "last_resid2": 1e-3}
    long_run = qm.garch11_forecast(params, horizon=5000)
    uncond_vol = math.sqrt(params["omega"] / (1.0 - params["alpha"] - params["beta"]))
    assert long_run == pytest.approx(uncond_vol, rel=1e-3)


def test_garch11_forecast_ham_getiriden_dict_ile_tutarli():
    resid = _sentetik_garch11(500, 1e-5, 0.08, 0.85, seed=3)
    fit = qm.garch11_fit(resid)
    assert fit is not None
    from_dict = qm.garch11_forecast(fit, horizon=1)
    from_raw = qm.garch11_forecast(resid, horizon=1)
    assert from_dict == pytest.approx(from_raw)


def test_garch11_forecast_gecersiz_ufuk_none():
    params = {"omega": 1e-6, "alpha": 0.08, "beta": 0.90, "last_sigma2": 2e-4, "last_resid2": 3e-4}
    assert qm.garch11_forecast(params, horizon=0) is None
    assert qm.garch11_forecast({"omega": 1e-6}, horizon=1) is None   # eksik anahtar


# ---------------------------------------------------------------------------
# EVT — Hill tahmincisi + GPD/POT VaR-ES
# ---------------------------------------------------------------------------
def test_evt_hill_tail_index_pareto_bilinen_deger():
    """Tip-I Pareto(alpha=3) örneklemi -> Hill tahmincisi alfa'yı geri bulmalı (bilinen-değer)."""
    rng = np.random.default_rng(3)
    alpha_true = 3.0
    u = rng.uniform(0.0, 1.0, 5000)
    sample = (1.0 - u) ** (-1.0 / alpha_true)
    est = qm.evt_hill_tail_index(sample, k=250)
    assert est is not None and est == pytest.approx(alpha_true, abs=0.3)


def test_evt_hill_tail_index_k_siniri_none():
    veri = list(range(1, 20))
    assert qm.evt_hill_tail_index(veri, k=19) is None   # k>=n
    assert qm.evt_hill_tail_index(veri, k=1) is None    # k<2


def test_evt_hill_tail_index_kisa_veri_none():
    assert qm.evt_hill_tail_index([1, 2, 3], k=1) is None


def test_evt_gpd_var_es_es_var_ve_esik_iliskisi():
    rng = np.random.default_rng(5)
    body = rng.normal(0.0, 0.01, 3000)
    result = qm.evt_gpd_var_es(body, alpha=0.01, threshold_q=0.95)
    assert result is not None
    assert result["es"] >= result["var"] >= result["threshold"]
    assert result["n_exceed"] >= 10
    assert set(result) == {"threshold", "shape", "scale", "n_exceed", "var", "es"}


def test_evt_gpd_var_es_alpha_freq_disinda_none():
    rng = np.random.default_rng(5)
    body = rng.normal(0.0, 0.01, 3000)
    # alpha, eşik-aşım oranından (~0.05) çok büyük -> ekstrapolasyon-dışı kullanım -> None.
    assert qm.evt_gpd_var_es(body, alpha=0.5, threshold_q=0.95) is None


def test_evt_gpd_var_es_gecersiz_parametre_none():
    rng = np.random.default_rng(5)
    body = rng.normal(0.0, 0.01, 3000)
    assert qm.evt_gpd_var_es(body, alpha=0.0, threshold_q=0.95) is None
    assert qm.evt_gpd_var_es(body, alpha=0.01, threshold_q=1.0) is None


def test_evt_gpd_var_es_kisa_veri_none():
    assert qm.evt_gpd_var_es([0.01, 0.02, -0.01]) is None


# ---------------------------------------------------------------------------
# Kopula (Gauss/Student-t) + kuyruk-bağımlılık
# ---------------------------------------------------------------------------
def test_copula_fit_gaussian_rho_geri_bulma():
    rng = np.random.default_rng(7)
    z1 = rng.standard_normal(2000)
    z2 = 0.7 * z1 + math.sqrt(1.0 - 0.7**2) * rng.standard_normal(2000)
    out = qm.copula_fit_gaussian(z1, z2)
    assert out is not None
    assert out["rho"] == pytest.approx(0.7, abs=0.05)
    assert math.isfinite(out["loglik"])
    assert out["n"] == 2000


def test_copula_fit_gaussian_uyumsuz_uzunluk_none():
    assert qm.copula_fit_gaussian(list(range(20)), list(range(10))) is None


def test_copula_fit_gaussian_sabit_seri_none():
    assert qm.copula_fit_gaussian([1.0] * 20, list(range(20))) is None


def test_copula_fit_t_rho_kendall_formulune_esit():
    """rho KAPALI-FORM'dur (sin(pi/2*tau)) — sayısal optimizasyona bağlı değil, bağımsız
    olarak scipy.stats.kendalltau ile çapraz-doğrulanabilir (qm modülünün kendi `stats`'ı)."""
    rng = np.random.default_rng(7)
    z1 = rng.standard_normal(500)
    z2 = 0.5 * z1 + math.sqrt(1.0 - 0.5**2) * rng.standard_normal(500)
    out = qm.copula_fit_t(z1, z2)
    assert out is not None
    tau = float(qm.stats.kendalltau(z1, z2)[0])
    expected_rho = math.sin(math.pi / 2.0 * tau)
    assert out["rho"] == pytest.approx(expected_rho, abs=1e-9)
    assert out["nu"] > 2.0 and math.isfinite(out["nu"])
    assert math.isfinite(out["loglik"])


def test_copula_fit_t_kisa_veri_none():
    assert qm.copula_fit_t([1, 2, 3], [1, 2, 3]) is None


def test_tail_dependence_bagimsiz_dusuk_komonoton_yuksek():
    rng = np.random.default_rng(4)
    indep = qm.tail_dependence(rng.uniform(0, 1, 3000), rng.uniform(0, 1, 3000))
    assert indep is not None
    assert indep["upper"] < 0.3 and indep["lower"] < 0.3

    comon = rng.uniform(0, 1, 3000)
    dep = qm.tail_dependence(comon, comon)
    assert dep is not None
    assert dep["upper"] == pytest.approx(1.0) and dep["lower"] == pytest.approx(1.0)
    assert dep["upper"] > indep["upper"]


def test_tail_dependence_gecersiz_q_none():
    veri = list(range(30))
    assert qm.tail_dependence(veri, veri, q=0.6) is None
    assert qm.tail_dependence(veri, veri, q=0.0) is None


# ---------------------------------------------------------------------------
# Asimetrik (arşimedyen) kopulalar — Tur-34
# ---------------------------------------------------------------------------
def _clayton_ornekle(theta: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Clayton kopulasından KOŞULLU ÖRNEKLEME ile üretim (kapalı form — MCMC yok):
    v = (u^−θ·(w^(−θ/(θ+1)) − 1) + 1)^(−1/θ), u,w ~ U(0,1) bağımsız."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    w = rng.uniform(size=n)
    v = (u ** (-theta) * (w ** (-theta / (theta + 1.0)) - 1.0) + 1.0) ** (-1.0 / theta)
    return u, v


def test_clayton_bilinen_parametreyi_geri_buluyor():
    """θ=2 üretilmiş veride MLE θ̂≈2, τ≈0.5 (=θ/(θ+2)) ve λ_L≈2^(−1/2)=0.707."""
    u, v = _clayton_ornekle(2.0, 3000, seed=1)
    fit = qm.copula_fit_clayton(u, v)
    assert fit is not None
    assert fit["theta"] == pytest.approx(2.0, abs=0.15)
    assert fit["tau"] == pytest.approx(0.5, abs=0.05)
    assert fit["lambda_lower"] == pytest.approx(2.0 ** -0.5, abs=0.02)
    assert fit["lambda_upper"] == 0.0          # Clayton ÜST kuyrukta bağımsızdır


def test_gumbel_theta_1_bagimsizlik_capasi():
    """θ=1'de Gumbel yoğunluğu özdeş 1 → log-olabilirlik TAM 0 (analitik çapa)."""
    rng = np.random.default_rng(7)
    ru, rv = rng.uniform(size=500), rng.uniform(size=500)
    assert qm._gumbel_loglik(1.0, ru, rv) == pytest.approx(0.0, abs=1e-9)


def test_gumbel_ust_kuyruk_clayton_alt_kuyruk():
    """İki aile birbirinin AYNASIDIR: Clayton alt, Gumbel üst kuyrukta bağımlıdır.
    Bu asimetri, t kopulasının simetrik varsayımının kapatamadığı boşluktur."""
    cl = qm.copula_tail_dependence("clayton", {"theta": 2.0})
    gu = qm.copula_tail_dependence("gumbel", {"theta": 2.0})
    assert cl["lower"] > 0.0 and cl["upper"] == 0.0
    assert gu["upper"] > 0.0 and gu["lower"] == 0.0
    t_lam = qm.copula_tail_dependence("t", {"rho": 0.5, "nu": 4})
    assert t_lam["lower"] == pytest.approx(t_lam["upper"])   # t SİMETRİKTİR
    assert qm.copula_tail_dependence("gaussian", {"rho": 0.9}) == {
        "family": "gaussian", "lower": 0.0, "upper": 0.0}


def test_copula_tail_dependence_gecersiz_girdi_none():
    """Bilinmeyen aile / geçersiz parametre → None (varsayılan 0 UYDURULMAZ)."""
    assert qm.copula_tail_dependence("frank", {"theta": 3.0}) is None
    assert qm.copula_tail_dependence("clayton", {"theta": -1.0}) is None
    assert qm.copula_tail_dependence("gumbel", {"theta": 0.5}) is None
    assert qm.copula_tail_dependence("t", {"rho": 0.5}) is None
    assert qm.copula_tail_dependence("gaussian", {"rho": 1.0}) is None


def test_copula_select_dogru_aileyi_seciyor():
    """Clayton verisinde AIC Clayton'ı seçmeli ve karar KESİN olmalı (ΔAIC ≥ 2).
    Ampirik λ_L teorik λ_L ile aynı mertebede olmalı (kontrol, kapı değil)."""
    u, v = _clayton_ornekle(2.0, 3000, seed=1)
    sel = qm.copula_select(u, v)
    assert sel is not None
    assert sel["best"] == "clayton" and sel["decisive"] is True
    assert sel["delta_aic"] > 2.0
    assert sel["fits"]["gumbel"]["aic"] > sel["fits"]["clayton"]["aic"]
    assert sel["empirical"]["lower"] > sel["empirical"]["upper"]


def test_copula_select_negatif_bagimlilikta_arsimedyen_denenmiyor():
    """τ ≤ 0'da Clayton/Gumbel UYGULANAMAZ → tabloda None ('denenmedi' ile 'uymadı'
    ayrımı korunur); eliptik aileler yine de uydurulur ve seçim onlardan yapılır."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=1500)
    y = -x + 0.25 * rng.normal(size=1500)          # güçlü NEGATİF bağımlılık
    sel = qm.copula_select(x, y)
    assert sel is not None
    assert sel["fits"]["clayton"] is None and sel["fits"]["gumbel"] is None
    assert sel["best"] in {"gaussian", "t"}


def test_arsimedyen_kopula_savunmaci_girdiler():
    assert qm.copula_fit_clayton([1, 2, 3], [1, 2, 3]) is None          # çok kısa
    assert qm.copula_fit_gumbel(list(range(20)), list(range(10))) is None  # uyumsuz
    assert qm.copula_fit_clayton([1.0] * 20, list(range(20))) is None   # dejenere
    assert qm.copula_select(list(range(5)), list(range(5))) is None     # n<20


def test_tail_dependence_kisa_veri_none():
    assert qm.tail_dependence([1, 2, 3], [1, 2, 3]) is None


# ---------------------------------------------------------------------------
# Kesirli fark alma (FFD)
# ---------------------------------------------------------------------------
def test_frac_diff_ffd_d0_seri_kendisi():
    rng = np.random.default_rng(4)
    series = 100 + np.cumsum(rng.normal(0, 1, 200))
    out = qm.frac_diff_ffd(series, 0.0)
    assert out is not None
    assert out == pytest.approx(series)


def test_frac_diff_ffd_d1_birinci_farka_yakinsar():
    rng = np.random.default_rng(4)
    series = 100 + np.cumsum(rng.normal(0, 1, 200))
    out = qm.frac_diff_ffd(series, 1.0)
    assert out is not None
    assert np.isnan(out[0])
    assert out[1:] == pytest.approx(np.diff(series), abs=1e-8)


def test_frac_diff_ffd_kesirli_d_sonlu_ve_nan_prefiksli():
    rng = np.random.default_rng(4)
    series = 100 + np.cumsum(rng.normal(0, 1, 200))
    out = qm.frac_diff_ffd(series, 0.4, threshold=1e-2)
    assert out is not None
    assert out.shape == series.shape
    nan_count = int(np.sum(np.isnan(out)))
    assert 0 < nan_count < series.size   # baştaki birkaç nokta NaN, geri kalan sonlu
    assert np.all(np.isfinite(out[nan_count:]))


def test_frac_diff_ffd_pencere_asimi_none():
    """Varsayılan threshold (1e-5) ile d=0.4 ağırlık penceresi kısa seriyi (200) aşar ->
    None (López de Prado FFD'nin bilinen ödünleşimi: küçük threshold = uzun pencere)."""
    rng = np.random.default_rng(4)
    series = 100 + np.cumsum(rng.normal(0, 1, 200))
    assert qm.frac_diff_ffd(series, 0.4) is None


def test_frac_diff_ffd_gecersiz_girdi_none():
    assert qm.frac_diff_ffd([1, 2, float("nan"), 4, 5], 0.5) is None
    assert qm.frac_diff_ffd([1, 2, 3, 4, 5], 0.5, threshold=-1.0) is None
    assert qm.frac_diff_ffd([1], 0.5) is None


# ---------------------------------------------------------------------------
# Spektral analiz (Welch) + Haar dalgacık vol ayrışımı
# ---------------------------------------------------------------------------
def test_welch_psd_bilinen_sinuzoid_frekans():
    n = 1024
    f0 = 0.125
    t_idx = np.arange(n)
    sine = np.sin(2.0 * math.pi * f0 * t_idx)
    out = qm.welch_psd(sine, fs=1.0)
    assert out is not None
    assert out["dominant_freq"] == pytest.approx(f0, abs=1e-6)
    assert out["total_power"] > 0.0
    assert len(out["freqs"]) == len(out["psd"])


def test_welch_psd_kisa_veri_none():
    assert qm.welch_psd([1, 2, 3]) is None


def test_welch_psd_gecersiz_fs_none():
    assert qm.welch_psd(list(range(20)), fs=-1.0) is None


def test_haar_dwt_vol_tam_enerji_detay_seviyesinde():
    """[1,-1,1,-1,...] periyot-2 salınımı: TÜM enerji seviye-1 detayında, yaklaşıklık SIFIR
    (bilinen-değer, elle doğrulanabilir Haar dönüşümü)."""
    osc = np.array([1.0, -1.0] * 4)
    out = qm.haar_dwt_vol(osc, levels=1)
    assert out is not None
    assert out["energy_ratio"]["level_1"] == pytest.approx(1.0)
    assert out["approx_energy"] == pytest.approx(0.0, abs=1e-10)


def test_haar_dwt_vol_sabit_seri_sifir_detay():
    sabit = np.full(8, 5.0)
    out = qm.haar_dwt_vol(sabit, levels=2)
    assert out is not None
    assert out["energy_ratio"]["level_1"] == pytest.approx(0.0)
    assert out["energy_ratio"]["level_2"] == pytest.approx(0.0)


def test_haar_dwt_vol_gecersiz_levels_veya_kisa_seri_none():
    assert qm.haar_dwt_vol([1, 2, 3, 4], levels=0) is None
    assert qm.haar_dwt_vol([1, 2, 3], levels=3) is None


# ---------------------------------------------------------------------------
# Almgren-Chriss optimal yürütme maliyeti (YALNIZ rapor amaçlı)
# ---------------------------------------------------------------------------
def test_almgren_chriss_cost_islemler_toplami_q_ve_sinirlar():
    out = qm.almgren_chriss_cost(q=1000, sigma=0.02, eta=0.001, gamma_risk=1.0, t=1.0, n=10)
    assert out is not None
    assert sum(out["trades"]) == pytest.approx(1000.0)
    assert out["trajectory"][0] == pytest.approx(1000.0)
    assert out["trajectory"][-1] == pytest.approx(0.0, abs=1e-6)
    assert out["expected_temp_impact_cost"] >= 0.0
    assert out["cost_variance"] >= 0.0


def test_almgren_chriss_cost_risk_notr_dogrusal_twap():
    out = qm.almgren_chriss_cost(q=1000, sigma=0.02, eta=0.001, gamma_risk=0.0, t=1.0, n=10)
    assert out is not None
    assert out["kappa"] is None
    assert out["trajectory"][1] == pytest.approx(900.0)   # doğrusal (TWAP) yörünge


def test_almgren_chriss_cost_risk_kacinma_on_yukleme():
    """Daha yüksek risk-kaçınma (gamma_risk) -> daha hızlı (ön-yüklemeli) tasfiye ->
    1. adım sonrası kalan miktar risk-nötr (doğrusal) duruma göre DAHA AZ olmalı."""
    dusuk = qm.almgren_chriss_cost(q=1000, sigma=0.02, eta=0.001, gamma_risk=1.0, t=1.0, n=10)
    yuksek = qm.almgren_chriss_cost(q=1000, sigma=0.02, eta=0.001, gamma_risk=50.0, t=1.0, n=10)
    assert dusuk is not None and yuksek is not None
    assert yuksek["trajectory"][1] < dusuk["trajectory"][1] < 900.0
    assert yuksek["kappa"] is not None and yuksek["kappa"] > 0.0


def test_almgren_chriss_cost_gecersiz_parametre_none():
    assert qm.almgren_chriss_cost(1000, 0.02, -0.001, 1.0, 1.0, 10) is None   # eta<=0
    assert qm.almgren_chriss_cost(1000, 0.02, 0.001, 1.0, 1.0, 0) is None     # n<1
    assert qm.almgren_chriss_cost(1000, 0.02, 0.001, 1.0, 0.0, 10) is None    # t<=0
    assert qm.almgren_chriss_cost(1000, -0.02, 0.001, 1.0, 1.0, 10) is None   # sigma<0


def test_almgren_chriss_cost_determinizm():
    a = qm.almgren_chriss_cost(q=500, sigma=0.015, eta=0.002, gamma_risk=5.0, t=2.0, n=20)
    b = qm.almgren_chriss_cost(q=500, sigma=0.015, eta=0.002, gamma_risk=5.0, t=2.0, n=20)
    assert a == b


# ---------------------------------------------------------------------------
# Bilgi-kuramsal desen ölçüleri (entropi / karşılıklı bilgi)
# ---------------------------------------------------------------------------
def test_permutation_entropy_duzenli_kucuk_karisik_buyuk():
    duzenli = np.arange(200, dtype=float)
    rng = np.random.default_rng(9)
    karisik = rng.normal(0, 1, 200)
    pe_duzenli = qm.permutation_entropy(duzenli, m=3, tau=1)
    pe_karisik = qm.permutation_entropy(karisik, m=3, tau=1)
    assert pe_duzenli is not None and pe_karisik is not None
    assert pe_duzenli == pytest.approx(0.0, abs=1e-9)
    assert pe_duzenli < pe_karisik
    assert 0.0 <= pe_karisik <= 1.0


def test_permutation_entropy_sinir_disi_m_none():
    veri = list(range(50))
    assert qm.permutation_entropy(veri, m=1) is None
    assert qm.permutation_entropy(veri, m=11) is None


def test_permutation_entropy_kisa_veri_none():
    assert qm.permutation_entropy([1, 2], m=3) is None


def test_sample_entropy_duzenli_kucuk_gurultu_buyuk():
    rng = np.random.default_rng(13)
    sinuzoid = np.sin(2.0 * math.pi * 0.05 * np.arange(500)) + rng.normal(0, 0.01, 500)
    gurultu = rng.normal(0, 1, 500)
    se_sin = qm.sample_entropy(sinuzoid, m=2, r=0.2)
    se_noise = qm.sample_entropy(gurultu, m=2, r=0.2)
    assert se_sin is not None and se_noise is not None
    assert se_sin < se_noise


def test_sample_entropy_sabit_seri_none():
    assert qm.sample_entropy(np.ones(50), m=2, r=0.2) is None


def test_sample_entropy_gecersiz_r_veya_kisa_none():
    assert qm.sample_entropy(list(range(20)), m=2, r=-0.1) is None
    assert qm.sample_entropy([1, 2, 3], m=2, r=0.2) is None


def test_mutual_info_bagimsiz_kucuk_bagimli_buyuk():
    rng = np.random.default_rng(17)
    x = rng.normal(0, 1, 3000)
    y_bagimsiz = rng.normal(0, 1, 3000)
    y_bagimli = x + rng.normal(0, 0.01, 3000)
    mi_bagimsiz = qm.mutual_info(x, y_bagimsiz, bins=10)
    mi_bagimli = qm.mutual_info(x, y_bagimli, bins=10)
    assert mi_bagimsiz is not None and mi_bagimli is not None
    assert 0.0 <= mi_bagimsiz < 0.2
    assert mi_bagimsiz < mi_bagimli
    assert mi_bagimli > 1.0


def test_mutual_info_uyumsuz_uzunluk_veya_az_bin_none():
    assert qm.mutual_info([1, 2, 3], [1, 2]) is None
    assert qm.mutual_info([1, 2, 3], [1, 2, 3], bins=1) is None


# ---------------------------------------------------------------------------
# Yüksek-moment ortak-risk tensörleri (coskewness / cokurtosis)
# ---------------------------------------------------------------------------
def test_coskewness_kosegen_scipy_skew_ile_esit_ve_simetrik():
    rng = np.random.default_rng(21)
    carpik = rng.standard_gamma(2.0, 500) - 2.0
    duz = rng.normal(0, 1, 500)
    ret_mat = np.column_stack([carpik, duz])
    tensor = qm.coskewness(ret_mat)
    assert tensor is not None
    assert tensor.shape == (2, 2, 2)
    assert tensor[0, 0, 0] == pytest.approx(float(qm.stats.skew(carpik, bias=True)))
    assert tensor[1, 1, 1] == pytest.approx(float(qm.stats.skew(duz, bias=True)))
    assert np.allclose(tensor, np.transpose(tensor, (1, 0, 2)))
    assert np.allclose(tensor, np.transpose(tensor, (2, 1, 0)))


def test_coskewness_kisa_veya_dejenere_none():
    assert qm.coskewness([[1, 2], [3, 4]]) is None   # <10 satır
    dejenere = np.column_stack([np.ones(20), np.random.default_rng(1).normal(0, 1, 20)])
    assert qm.coskewness(dejenere) is None   # sıfır-varyans sütun


def test_cokurtosis_kosegen_scipy_kurtosis_ile_esit_ve_simetrik():
    rng = np.random.default_rng(21)
    carpik = rng.standard_gamma(2.0, 500) - 2.0
    duz = rng.normal(0, 1, 500)
    ret_mat = np.column_stack([carpik, duz])
    tensor = qm.cokurtosis(ret_mat)
    assert tensor is not None
    assert tensor.shape == (2, 2, 2, 2)
    assert tensor[0, 0, 0, 0] == pytest.approx(float(qm.stats.kurtosis(carpik, fisher=False, bias=True)))
    assert np.allclose(tensor, np.transpose(tensor, (1, 0, 2, 3)))
    assert np.allclose(tensor, np.transpose(tensor, (3, 2, 1, 0)))


def test_cokurtosis_kisa_veya_dejenere_none():
    assert qm.cokurtosis(np.random.default_rng(2).normal(0, 1, (10, 2))) is None   # <15 satır
    dejenere = np.column_stack([np.ones(20), np.random.default_rng(1).normal(0, 1, 20)])
    assert qm.cokurtosis(dejenere) is None


# ---------------------------------------------------------------------------
# v3 konsolide: savunmacı-girdi + determinizm (birden çok fonksiyon)
# ---------------------------------------------------------------------------
def test_v3_savunmaci_nan_girdi_none_konsolide():
    nan = float("nan")
    assert qm.garch11_fit([0.01, nan] + [0.001] * 60) is None
    assert qm.evt_hill_tail_index([0.01, nan, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09], k=3) is None
    assert qm.evt_gpd_var_es([0.01, nan, 0.02]) is None
    assert qm.copula_fit_gaussian([1, 2, nan, 4, 5, 6, 7, 8, 9, 10], list(range(10))) is None
    assert qm.frac_diff_ffd([1, nan, 3, 4, 5], 0.5) is None
    assert qm.welch_psd([1, 2, nan, 4, 5, 6, 7, 8]) is None
    assert qm.haar_dwt_vol([1, nan, 3, 4]) is None
    assert qm.permutation_entropy([1, nan, 3, 4, 5], m=2) is None
    assert qm.sample_entropy([1, nan, 3, 4, 5], m=2) is None
    assert qm.mutual_info([1, nan, 3], [1, 2, 3]) is None
    assert qm.coskewness([[1, nan], [3, 4]] * 6) is None


def test_v3_bos_girdi_cokmez_konsolide():
    assert qm.garch11_fit([]) is None
    assert qm.evt_hill_tail_index([], k=2) is None
    assert qm.copula_fit_gaussian([], []) is None
    assert qm.frac_diff_ffd([], 0.5) is None
    assert qm.welch_psd([]) is None
    assert qm.haar_dwt_vol([]) is None
    assert qm.permutation_entropy([], m=2) is None
    assert qm.sample_entropy([], m=2) is None
    assert qm.mutual_info([], []) is None
    assert qm.coskewness([]) is None
    assert qm.cokurtosis([]) is None
    assert qm.almgren_chriss_cost(0, 0, 0, 0, 0, 0) is None


def test_v3_determinizm_konsolide():
    rng = np.random.default_rng(31)
    z1 = rng.standard_normal(300)
    z2 = 0.4 * z1 + math.sqrt(1 - 0.4**2) * rng.standard_normal(300)
    assert qm.copula_fit_gaussian(z1, z2) == qm.copula_fit_gaussian(z1, z2)
    assert qm.copula_fit_t(z1, z2) == qm.copula_fit_t(z1, z2)
    assert qm.evt_hill_tail_index(np.abs(z1) + 1.0, k=50) == qm.evt_hill_tail_index(np.abs(z1) + 1.0, k=50)
    ret_mat = np.column_stack([z1, z2])
    assert np.array_equal(qm.coskewness(ret_mat), qm.coskewness(ret_mat))
    assert np.array_equal(qm.cokurtosis(ret_mat), qm.cokurtosis(ret_mat))
    assert qm.permutation_entropy(z1, m=3) == qm.permutation_entropy(z1, m=3)
    assert qm.sample_entropy(z1, m=2) == qm.sample_entropy(z1, m=2)
    assert qm.mutual_info(z1, z2, bins=8) == qm.mutual_info(z1, z2, bins=8)
