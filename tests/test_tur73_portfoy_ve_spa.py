"""test_tur73_portfoy_ve_spa.py — Tur-73'ün portföy / istatistik / opsiyon ailelerinin kilidi.

Kapsanan maddeler (`finans genel geliştirme listesi.txt`, 320 madde):
  Madde 116  NCO (iç içe kümelenmiş optimizasyon) → portfolio.nco_weights
  Madde 117  Maksimum çeşitlendirme               → portfolio.max_diversification_weights
  Madde 118  Volatilite hedefleme                 → portfolio.vol_target_scale
  Madde 119  Bayesyen Kelly                       → portfolio.bayesian_kelly
  Madde 120  Likidite-ayarlı VaR                  → portfolio.liquidity_adjusted_var
  Madde 281  CDaR optimizasyonu                   → portfolio.cdar_lp_weights / cdar
  Madde 283  CPPI                                 → portfolio.cppi_allocation
  Madde 177  White Reality Check + Hansen SPA     → deflated.reality_check / hansen_spa
  Madde 279  İma edilen korelasyon                → options_math.implied_correlation
  Madde 145/211/212  Hazine eğrisi kablosu        → fred_data.TREASURY_MATURITIES

TEST FELSEFESİ (Tur-49/69/72 dersleri): "koştu ve patlamadı" bir test DEĞİLDİR. Her aile
ya BİLİNEN BİR ÖZDEŞLİĞE (ρ=1'de DR=1; N bağımsız varlıkta DR=√N; blok-köşegen kovaryansta
NCO ≡ MinVar; ρ üretilip geri çıkarma) ya da DİFERANSİYEL bir karşılaştırmaya (CDaR-LP'nin
CDaR'ı CVaR-LP'ninkinden düşük; RC sulandırılırken SPA_c sabit) bağlanır.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nihai.ai import options_math as om
from nihai.ai import portfolio as P
from nihai.data import fred_data
from research import deflated as D

SEED = 20260906


# ---------------------------------------------------------------------------
# Madde 117 — maksimum çeşitlendirme
# ---------------------------------------------------------------------------
def test_mdp_rho1_dr_tam_bir():
    """Tüm çiftler ρ=1 → çeşitlendirme YOK → DR her ağırlıkta TAM 1."""
    sig = np.array([0.10, 0.25, 0.40])
    cov = np.outer(sig, sig)                      # ρ=1 her yerde
    w = P.max_diversification_weights(cov)
    assert w is not None
    assert P.diversification_ratio(w, cov) == pytest.approx(1.0, abs=1e-9)


def test_mdp_bagimsizda_ters_vol_ve_dr_sqrt_n():
    """ρ=0 → MDP'nin kapalı formu ters-volatilitedir ve DR = √N."""
    sig = np.array([0.10, 0.25, 0.40])
    cov = np.diag(sig**2)
    w = P.max_diversification_weights(cov)
    kapali = (1.0 / sig) / np.sum(1.0 / sig)
    assert np.max(np.abs(w - kapali)) < 1e-6
    assert P.diversification_ratio(w, cov) == pytest.approx(math.sqrt(3.0), abs=1e-4)


def test_dr_ve_mdp_dejenere_girdi():
    assert P.max_diversification_weights(np.array([[np.nan, 0.0], [0.0, 1.0]])) is None
    assert P.max_diversification_weights(np.zeros((2, 2))) is None      # σ=0
    assert P.diversification_ratio([0.5, 0.5], np.zeros((2, 2))) is None
    assert P.diversification_ratio([0.5, 0.5], np.eye(3)) is None       # boyut uyumsuz


# ---------------------------------------------------------------------------
# Madde 116 — NCO
# ---------------------------------------------------------------------------
def _blok_cov(sig: np.ndarray, blok_boy: int, rho: float) -> np.ndarray:
    n = sig.size
    c = np.zeros((n, n))
    for a in range(0, n, blok_boy):
        c[a:a + blok_boy, a:a + blok_boy] = rho
    np.fill_diagonal(c, 1.0)
    return np.outer(sig, sig) * c


def test_nco_blok_kosegende_minvar_ile_OZDES():
    """TAM blok-köşegen kovaryansta NCO'nun ayrıştırması EXACT'tır → MinVar ile özdeş."""
    sig = np.array([0.12, 0.18, 0.25, 0.31, 0.14, 0.22, 0.28, 0.35])
    cov = _blok_cov(sig, 4, 0.85)
    w_nco, w_min = P.nco_weights(cov), P.min_variance_weights(cov)
    assert w_nco is not None and w_min is not None
    assert np.max(np.abs(w_nco - w_min)) < 1e-12


def test_nco_kumeleri_dogru_buluyor_ve_kisiti_ihlal_etmiyor():
    sig = np.full(8, 0.20)
    cov = _blok_cov(sig, 4, 0.85)
    etiket = P._corr_clusters(cov, None)
    assert len(set(etiket[:4].tolist())) == 1 and len(set(etiket[4:].tolist())) == 1
    assert etiket[0] != etiket[4]
    w = P.nco_weights(cov, bounds=(np.zeros(8), np.full(8, 0.20)))
    assert w is not None
    assert w.max() <= 0.20 + 1e-9 and w.min() >= -1e-9
    assert np.sum(w) == pytest.approx(1.0, abs=1e-6)


def test_nco_gurultulu_kovaryansta_minvardan_SAGLAM():
    """NCO'nun ASIL iddiası: T~2N gürültü rejiminde GERÇEK kovaryansta daha düşük varyans.
    Tahmin edilen kovaryansla çöz, GERÇEK kovaryansta değerlendir (tek dürüst kurulum)."""
    rng = np.random.default_rng(SEED)
    n, t_n, tekrar = 20, 40, 40
    sig = 0.10 + 0.30 * rng.random(n)
    cov_true = _blok_cov(sig, 7, 0.70)
    chol = np.linalg.cholesky(cov_true + 1e-12 * np.eye(n))
    v_nco, v_min = [], []
    for _ in range(tekrar):
        cov_hat = np.cov(rng.standard_normal((t_n, n)) @ chol.T, rowvar=False)
        a, b = P.nco_weights(cov_hat), P.min_variance_weights(cov_hat)
        assert a is not None and b is not None
        v_nco.append(float(a @ cov_true @ a))
        v_min.append(float(b @ cov_true @ b))
    kazanan = int(np.sum(np.array(v_nco) < np.array(v_min)))
    assert kazanan > tekrar * 0.70, f"NCO {kazanan}/{tekrar} kazandı — saglamlik iddiasi cokmus"
    assert np.mean(v_nco) < np.mean(v_min)


def test_nco_dejenere_girdi():
    assert P.nco_weights(np.array([[np.inf]])) is None
    assert P.nco_weights(np.eye(2)) is not None          # n<=2 düz min-varyansa düşer


# ---------------------------------------------------------------------------
# Madde 118 — volatilite hedefleme
# ---------------------------------------------------------------------------
def test_vol_hedefleme_kapali_form_ve_asimetri():
    cov = np.diag(np.full(4, 0.02**2))
    r = P.vol_target_scale(np.full(4, 0.25), cov, target_vol=0.10, ppy=252)
    assert r["sigma_port"] == pytest.approx(0.01 * math.sqrt(252), rel=1e-12)
    assert r["olcek"] == pytest.approx(0.10 / r["sigma_port"], rel=1e-12)
    assert r["nakit_orani"] == pytest.approx(1.0 - r["olcek"], rel=1e-12)
    # ASİMETRİ: oynaklık hedefin ALTINDAYKEN büyütme YOK (max_leverage=1)
    dusuk = P.vol_target_scale(np.full(4, 0.25), np.diag(np.full(4, 0.001**2)), target_vol=0.10)
    assert dusuk["sigma_port"] < 0.10 and dusuk["olcek"] == pytest.approx(1.0)
    assert dusuk["nakit_orani"] == pytest.approx(0.0)


def test_vol_hedefleme_dejenere():
    assert P.vol_target_scale([0.5, 0.5], np.zeros((2, 2)), 0.1) is None
    assert P.vol_target_scale([0.5, 0.5], np.eye(2), target_vol=0.0) is None
    assert P.vol_target_scale([0.5, 0.5], np.eye(3), 0.1) is None


# ---------------------------------------------------------------------------
# Madde 281 — CDaR
# ---------------------------------------------------------------------------
def _cdar_cvar_paneli() -> np.ndarray:
    """İki ölçütü AYIRAN kurgu: A yavaş kanar (kötü ÇEKİLME, iyi günlük kuyruk), B keskin
    şok alır ve hemen toparlar (kötü KUYRUK, iyi çekilme). Köşe çözümü üretmeyen bir panel
    olmadan CDaR-LP ile CVaR-LP ayırt EDİLEMEZ — ilk kurgu tam bu yüzden bir mutasyonu
    kaçırmıştı (Tur-73)."""
    rng = np.random.default_rng(4242)
    t_n = 400
    a = rng.normal(0.0, 0.004, t_n)
    a[80:260] -= 0.0016
    b = rng.normal(0.0006, 0.004, t_n)
    for t in range(20, t_n - 1, 40):
        b[t] -= 0.030
        b[t + 1] += 0.030
    return np.column_stack([a, b])


def test_cdar_lp_ve_cvar_lp_HER_BIRI_KENDI_olcutunde_kazaniyor():
    """İki yöntem AYNI şeyi optimize etmiyorsa ağırlıkları AYRIŞMALI ve her biri kendi
    ölçütünde diğerini YENMELİ. Yalnız 'CDaR-LP daha düşük CDaR verir' demek yetmez:
    iki yöntem köşede buluştuğunda o iddia bilgi taşımadan da doğru çıkar."""
    x = _cdar_cvar_paneli()
    w_cdar = P.cdar_lp_weights(x, alpha=0.05)
    w_cvar = P.cvar_lp_weights(x, alpha=0.05)
    assert w_cdar is not None and w_cvar is not None
    assert not np.allclose(w_cdar, w_cvar, atol=1e-3), "iki yöntem AYNI ağırlığı verdi"
    # her biri KENDİ ölçütünde önde
    assert P.cdar(x, w_cdar)["cdar"] < P.cdar(x, w_cvar)["cdar"]
    from nihai.ai.quant_math import cvar_historical
    assert cvar_historical(x @ w_cvar, 0.05) < cvar_historical(x @ w_cdar, 0.05)
    # ve ikisi de eşit-ağırlığı kendi ölçütünde geçiyor
    assert P.cdar(x, w_cdar)["cdar"] < P.cdar(x, np.array([0.5, 0.5]))["cdar"]
    assert cvar_historical(x @ w_cvar, 0.05) < cvar_historical(x @ np.array([0.5, 0.5]), 0.05)
    assert np.sum(w_cdar) == pytest.approx(1.0, abs=1e-6)


def test_BULGU_73_1_alpha_KUYRUK_olasiligi_konvansiyonu():
    """`cvar_lp_weights` alpha'yı GÜVEN DÜZEYİ sayıyordu (payda (1−α)T) → varsayılan 0,05
    ile en kötü %95'i ortalıyor, yani ORTALAMA KAYBI minimize ediyordu. Artık payda α·T ve
    çözüm kaba kuvvet ızgarasının en iyisiyle örtüşüyor."""
    from nihai.ai.quant_math import cvar_historical
    x = _cdar_cvar_paneli()
    w = P.cvar_lp_weights(x, alpha=0.05)
    izgara = np.linspace(0.0, 1.0, 201)
    en_iyi = min(cvar_historical(x @ np.array([g, 1 - g]), 0.05) for g in izgara)
    assert cvar_historical(x @ w, 0.05) <= en_iyi + 1e-9, "LP kaba kuvvetin gerisinde"
    # Konvansiyon: α büyüdükçe daha GENİŞ bir kuyruk ortalanır → çözüm ortalama-kayba kayar.
    w95 = P.cvar_lp_weights(x, alpha=0.95)
    assert cvar_historical(x @ w95, 0.05) > cvar_historical(x @ w, 0.05)


def test_cdar_lp_alpha_KUYRUK_konvansiyonu_olcumle_TUTARLI():
    """`cdar_lp_weights` ile `cdar()` AYNI α'yı aynı anlamda okumalı: α=0,05 ile çözülen
    ağırlık, α=0,95 ile çözülenden CDaR@5%'te KESİN daha iyi olmalı. Payda `(1−α)·T`ye
    dönerse iki çözüm ÇAKIŞIR ve bu eşitsizlik düşer (mutasyonla doğrulandı)."""
    x = _cdar_cvar_paneli()
    w05 = P.cdar_lp_weights(x, alpha=0.05)
    w95 = P.cdar_lp_weights(x, alpha=0.95)
    assert w05 is not None and w95 is not None
    assert P.cdar(x, w05, 0.05)["cdar"] < P.cdar(x, w95, 0.05)["cdar"]


def test_cdar_olcumu_bilinen_seride():
    """Elle hesaplanabilir seri: kümülatif [+1,+1,-3,+1] → çekilme [0,0,3,2]."""
    r = np.array([[1.0], [1.0], [-3.0], [1.0]])
    d = P.cdar(r, [1.0], alpha=0.25)
    assert d["max_dd"] == pytest.approx(3.0)
    assert d["dar"] == pytest.approx(float(np.quantile([0.0, 0.0, 3.0, 2.0], 0.75)))


def test_cdar_dejenere():
    assert P.cdar_lp_weights(np.zeros((1, 2))) is None
    assert P.cdar_lp_weights(np.zeros((5, 2)), alpha=1.5) is None
    assert P.cdar(np.zeros((5, 2)), [1.0]) is None


# ---------------------------------------------------------------------------
# Madde 283 — CPPI (ve kayan-taban tuzağı)
# ---------------------------------------------------------------------------
def test_cppi_kayan_taban_TUZAGI_sabit_oran_uretir():
    """floor_pct güncel sermayeden türetilirse taban sermayeyle DÜŞER → riskli oran SABİT
    kalır ve sigorta anlamını yitirir. Bayrak bunu raporlamalı."""
    oranlar = [P.cppi_allocation(e, floor_pct=0.90, multiplier=3.0)["riskli_oran"]
               for e in (100.0, 90.0, 70.0, 50.0)]
    assert all(abs(o - oranlar[0]) < 1e-12 for o in oranlar)
    assert oranlar[0] == pytest.approx(min(3.0 * 0.10, 1.0))
    assert P.cppi_allocation(100.0)["taban_kayan"] is True


def test_cppi_mutlak_taban_derisk_ediyor():
    onceki = 1.0
    for e in (100.0, 95.0, 92.0, 90.0, 85.0):
        c = P.cppi_allocation(e, multiplier=3.0, floor_value=90.0)
        assert c["riskli_oran"] <= onceki + 1e-12
        onceki = c["riskli_oran"]
        assert c["taban_kayan"] is False
    assert P.cppi_allocation(90.0, floor_value=90.0)["riskli_oran"] == 0.0
    assert P.cppi_allocation(85.0, floor_value=90.0)["tabanin_altinda"] is True


def test_cppi_dejenere():
    assert P.cppi_allocation(0.0) is None
    assert P.cppi_allocation(100.0, multiplier=0.0) is None
    assert P.cppi_allocation(100.0, floor_pct=1.0) is None


# ---------------------------------------------------------------------------
# Madde 119 — Bayesyen Kelly (ve doğrusallık gerçeği)
# ---------------------------------------------------------------------------
def test_bayes_kelly_posterior_ORTALAMASI_bayes_optimaldir():
    """İkili bahiste E[log-büyüme] p'de DOĞRUSAL → posterior ortalamasındaki klasik Kelly
    Bayes-optimalin TA KENDİSİDİR. Sayısal maksimizasyonla kanıtlanır (uydurma yok)."""
    from scipy.integrate import quad
    from scipy.stats import beta as sp_beta
    r = P.bayesian_kelly(60, 40, payoff_ratio=1.0)
    a, b = r["posterior_a"], r["posterior_b"]

    def neg_e(f: float) -> float:
        return -quad(lambda p: sp_beta.pdf(p, a, b)
                     * (p * math.log(1 + f) + (1 - p) * math.log(1 - f)), 0.0, 1.0, limit=200)[0]

    izgara = np.linspace(0.01, 0.60, 60)
    kaba = float(izgara[int(np.argmin([neg_e(f) for f in izgara]))])
    ince = np.linspace(max(1e-3, kaba - 0.02), kaba + 0.02, 81)
    f_sayisal = float(ince[int(np.argmin([neg_e(f) for f in ince]))])
    assert abs(f_sayisal - r["f_bayes"]) < 5e-4


def test_bayes_kelly_kesinti_n_ile_ERIYOR():
    """Belirsizlik indirimi KUANTİLDEN gelir ve örneklem büyüdükçe 0'a iner."""
    kesintiler = [P.bayesian_kelly(int(0.6 * n), n - int(0.6 * n))["kesinti"]
                  for n in (10, 100, 1000, 10000)]
    assert all(kesintiler[i] > kesintiler[i + 1] for i in range(len(kesintiler) - 1))
    assert kesintiler[-1] < 0.02
    assert P.bayesian_kelly(600, 400)["f_robust"] < P.bayesian_kelly(600, 400)["f_bayes"]


def test_bayes_kelly_negatif_kenar_sifira_kirpilir():
    r = P.bayesian_kelly(20, 80, payoff_ratio=1.0)
    assert r["f_bayes"] == 0.0 and r["f_robust"] == 0.0


def test_bayes_kelly_dejenere():
    assert P.bayesian_kelly(-1, 5) is None
    assert P.bayesian_kelly(5, 5, payoff_ratio=0.0) is None
    assert P.bayesian_kelly(5, 5, quantile=0.9) is None


# ---------------------------------------------------------------------------
# Madde 120 — L-VaR
# ---------------------------------------------------------------------------
def test_lvar_kucuk_pozisyonda_carpan_bire_yakin_buyukte_artiyor():
    kucuk = P.liquidity_adjusted_var(10_000.0, 50_000_000.0, 0.02)
    buyuk = P.liquidity_adjusted_var(20_000_000.0, 50_000_000.0, 0.02)
    assert kucuk["ufuk_gun"] == 1.0 and kucuk["carpan"] < 1.02
    assert buyuk["ufuk_gun"] == pytest.approx(2.0)
    assert buyuk["carpan"] > kucuk["carpan"] * 1.3
    assert buyuk["l_var"] == pytest.approx(
        buyuk["var_ufuk"] + buyuk["makas_maliyeti"] + buyuk["etki_maliyeti"], rel=1e-12)


def test_lvar_dejenere():
    assert P.liquidity_adjusted_var(-1.0, 1.0, 0.02) is None
    assert P.liquidity_adjusted_var(1.0, 1.0, 0.02, alpha=0.9) is None
    assert P.liquidity_adjusted_var(1.0, 1.0, 0.02, participation=0.0) is None


# ---------------------------------------------------------------------------
# Madde 177 — White Reality Check + Hansen SPA
# ---------------------------------------------------------------------------
def test_stationary_bootstrap_ortalama_blok_uzunlugu():
    """Politis-Romano: ardışıklık oranı 1−q → ortalama blok = 1/q = mean_block."""
    n = 1000
    idx = D.stationary_bootstrap_indices(n, 5.0, 200, seed=SEED)
    assert idx.shape == (200, n)
    ardisik = float(np.mean(idx[:, 1:] == (idx[:, :-1] + 1) % n))
    assert 1.0 / (1.0 - ardisik) == pytest.approx(5.0, rel=0.05)
    # determinizm
    assert np.array_equal(idx, D.stationary_bootstrap_indices(n, 5.0, 200, seed=SEED))


def test_spa_sulandirmaya_dayanikli_RC_degil():
    """Hansen'in RC'ye getirdiği asıl düzeltme: DEĞERSİZ aday eklemek RC'nin p'sini
    şişirir, SPA_c'ninkini DEĞİŞTİRMEZ (tutarlı yeniden merkezleme)."""
    rng = np.random.default_rng(SEED)
    t_n = 500
    iyi = rng.normal(2.0 * 0.01 / math.sqrt(t_n), 0.01, size=(t_n, 1))   # t ≈ 2
    notr = rng.normal(0.0, 0.01, size=(t_n, 4))
    kotu = rng.normal(-0.010, 0.01, size=(t_n, 150))
    dar = np.hstack([iyi, notr])
    genis = np.hstack([iyi, notr, kotu])
    rc_dar = D.reality_check(dar, 5.0, 800, seed=1)["p_value"]
    rc_genis = D.reality_check(genis, 5.0, 800, seed=1)["p_value"]
    spa_dar = D.hansen_spa(dar, 5.0, 800, seed=1)["p_consistent"]
    spa_genis = D.hansen_spa(genis, 5.0, 800, seed=1)["p_consistent"]
    assert rc_genis > rc_dar + 0.20, f"RC sulanmadi: {rc_dar:.3f} -> {rc_genis:.3f}"
    assert abs(spa_genis - spa_dar) < 1e-12, f"SPA_c kaydi: {spa_dar} -> {spa_genis}"


def test_spa_p_siralamasi_ve_guclu_adayi_yakalamasi():
    rng = np.random.default_rng(SEED + 1)
    d = rng.normal(0.0, 0.01, size=(400, 8))
    d[:, 3] += 0.004                                   # açıkça üstün aday
    s = D.hansen_spa(d, 5.0, 500, seed=2)
    assert s["p_lower"] <= s["p_consistent"] <= s["p_upper"] + 1e-12
    assert s["p_value"] < 0.01 and s["en_iyi"] == 3
    assert D.reality_check(d, 5.0, 500, seed=2)["en_iyi"] == 3


def test_rc_spa_H0_altinda_asiri_reddetmiyor():
    """Tümü değersizken 0.05 düzeyinde red oranı ~0.05 olmalı (boyut kalibrasyonu)."""
    red_rc = red_spa = 0
    tekrar = 60
    for i in range(tekrar):
        d = np.random.default_rng(SEED + i).normal(0.0, 0.01, size=(300, 8))
        red_rc += D.reality_check(d, 5.0, 300, seed=i)["p_value"] < 0.05
        red_spa += D.hansen_spa(d, 5.0, 300, seed=i)["p_value"] < 0.05
    assert red_rc / tekrar < 0.20 and red_spa / tekrar < 0.20


def test_rc_spa_dejenere():
    assert D.reality_check(np.zeros((3, 2))) is None          # T<8
    assert D.hansen_spa(np.zeros((3, 2))) is None
    assert D.reality_check([[1.0, np.nan]] * 20) is None
    assert D.hansen_spa(np.zeros((20, 2))) is None            # tüm ŵ=0
    assert D.stationary_bootstrap_indices(0, 5.0, 10, 0).size == 0


# ---------------------------------------------------------------------------
# Madde 279 — ima edilen korelasyon
# ---------------------------------------------------------------------------
def test_implied_correlation_uretilen_rho_yu_GERI_cikariyor():
    rng = np.random.default_rng(SEED)
    n = 8
    w = rng.random(n)
    w /= w.sum()
    sig = 0.15 + 0.35 * rng.random(n)
    for rho in (0.0, 0.35, 0.90, 1.0):
        c = np.full((n, n), rho)
        np.fill_diagonal(c, 1.0)
        sigma_idx = math.sqrt(float(w @ (np.outer(sig, sig) * c) @ w))
        r = om.implied_correlation(sigma_idx, w, sig)
        assert r["rho_imp"] == pytest.approx(rho, abs=1e-12)
        assert r["gecerli"] is True          # ρ=1 sınırı kayan-nokta tozuyla düşmemeli


def test_implied_correlation_tutarsiz_girdi_gecersiz_isaretleniyor():
    r = om.implied_correlation(5.0, [0.5, 0.5], [0.2, 0.2])   # endeks vol'ü absürt yüksek
    assert r["rho_imp"] > 1.0 and r["gecerli"] is False       # KIRPILMAZ, raporlanır


def test_implied_correlation_gerceklesen_fark():
    n = 5
    w = np.full(n, 1.0 / n)
    sig = np.full(n, 0.25)
    c_gercek = np.full((n, n), 0.30)
    np.fill_diagonal(c_gercek, 1.0)
    c_ima = np.full((n, n), 0.55)
    np.fill_diagonal(c_ima, 1.0)
    sigma_idx = math.sqrt(float(w @ (np.outer(sig, sig) * c_ima) @ w))
    r = om.implied_correlation(sigma_idx, w, sig, pairwise=c_gercek)
    assert r["rho_imp"] == pytest.approx(0.55, abs=1e-12)
    assert r["rho_realized"] == pytest.approx(0.30, abs=1e-12)
    assert r["fark"] == pytest.approx(0.25, abs=1e-12)


def test_implied_correlation_agirlik_normalize_ve_dejenere():
    r = om.implied_correlation(0.2, [2.0, 2.0, 2.0], [0.2, 0.2, 0.2])
    assert r["agirlik_toplami"] == pytest.approx(6.0)   # ham toplam raporlanır
    assert om.implied_correlation(0.2, [1.0], [0.3]) is None
    assert om.implied_correlation(-0.2, [0.5, 0.5], [0.2, 0.2]) is None
    assert om.implied_correlation(0.2, [0.5, 0.5], [0.2, float("nan")]) is None
    assert om.implied_correlation(0.2, [1.0, 0.0], [0.2, 0.2]) is None   # payda 0


# ---------------------------------------------------------------------------
# Madde 145/211/212 — Hazine eğrisi kablosu (AĞ İSTEĞİ YOK; yalnız sözleşme)
# ---------------------------------------------------------------------------
def test_hazine_vade_haritasi_tutarli():
    m = fred_data.TREASURY_MATURITIES
    assert m["DGS1MO"] == pytest.approx(1 / 12) and m["DGS10"] == 10.0
    assert list(m.values()) == sorted(m.values()), "vadeler artan sırada olmalı"
    assert len(m) >= 8 and all(v > 0 for v in m.values())
    assert set(fred_data.CURVE_SPREADS) == {"T10Y2Y", "T10Y3M"}
    assert set(fred_data.BREAKEVEN) == {"T5YIE", "T10YIE"}
