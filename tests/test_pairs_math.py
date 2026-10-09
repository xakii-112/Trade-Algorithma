"""research.pairs_test matematiği için birim testleri — AĞSIZ, <10 sn, deterministik.

Kapsam: (1) Kalman dinamik-beta sentetik bilinen-beta'ya yakınsar; (2) statik z FORMATION
parametreleriyle hesaplanır (trade penceresinde yeniden ölçekleme YOK = look-ahead yok);
(3) simulate_pair state-machine giriş→çıkış işlemi üretir + maliyet doğru; (4) select_window_pairs
coint MOCK'lanarak eleme/sıralama mantığı (ağsız). coint gerçek çağrısı testlerde asla yapılmaz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.pairs_test import (
    kalman_beta_z,
    ols_beta_alpha,
    select_window_pairs,
    simulate_pair,
    static_spread_stats,
    static_z,
)


# ---------------------------------------------------------------------------
# (1) Kalman: sabit gerçek-beta'ya yakınsama
# ---------------------------------------------------------------------------
def test_kalman_bilinen_betaya_yakinsar():
    """log_a = 2·log_b + 0.5 + küçük gürültü, sabit gerçek beta=2. Filtre yanlış beta0=1'den
    başlasa bile serinin sonunda beta ≈ 2'ye yakınsamalı (dinamik hedge doğru öğreniyor)."""
    rng = np.random.default_rng(0)
    log_b = np.cumsum(rng.standard_normal(400)) * 0.02 + 4.0   # yavaş random-walk log-fiyat
    beta_true, alpha_true = 2.0, 0.5
    log_a = beta_true * log_b + alpha_true + rng.standard_normal(400) * 0.005
    z, betas = kalman_beta_z(log_a, log_b, beta0=1.0, alpha0=0.0, obs_var=1e-4, delta=1e-5)
    assert abs(betas[-1] - beta_true) < 0.15, f"beta yakınsamadı: {betas[-1]:.3f}"
    # innovation z'ler standartlaştırılmış → makul ölçekte (patlamamış)
    assert np.isfinite(z).all() and np.abs(z[50:]).mean() < 5.0


def test_kalman_z_ortalama_sifira_yakin():
    """Model doğruyken (spread ~ beyaz gürültü) standartlaştırılmış innovation ortalaması
    warmup sonrası ~0 olmalı (sistematik sapma yok)."""
    rng = np.random.default_rng(1)
    log_b = np.cumsum(rng.standard_normal(300)) * 0.02 + 3.0
    log_a = 1.5 * log_b - 0.3 + rng.standard_normal(300) * 0.004
    z, _ = kalman_beta_z(log_a, log_b, beta0=1.5, alpha0=-0.3, obs_var=1e-4, delta=1e-6)
    assert abs(float(np.mean(z[50:]))) < 0.6


# ---------------------------------------------------------------------------
# (2) Statik z: FORMATION parametreleri kullanılır — look-ahead YOK
# ---------------------------------------------------------------------------
def test_statik_z_formation_parametreleri_kullanir_lookahead_yok():
    """Trade penceresinde spread'e SABİT bir C kayması eklenirse, statik z bunu C/sigma_form
    olarak GÖRMELİ (formation mu ile ölçekliyor). Eğer trade penceresinden yeniden merkezleseydi
    (look-ahead), z ortalaması 0 çıkardı. z ≈ C/sigma → yeniden-ölçekleme YOK kanıtı."""
    rng = np.random.default_rng(2)
    log_b_form = np.cumsum(rng.standard_normal(252)) * 0.02 + 4.0
    beta, alpha = 1.8, 0.2
    resid_form = rng.standard_normal(252) * 0.01
    log_a_form = beta * log_b_form + alpha + resid_form
    b_est, a_est = ols_beta_alpha(log_a_form, log_b_form)
    mu, sigma = static_spread_stats(log_a_form, log_b_form, b_est, a_est)

    # Trade: resid'i tam olarak (mu + shift) sabitine ayarla → z her noktada shift/sigma olmalı.
    shift = 3.0 * sigma
    log_b_trade = np.cumsum(rng.standard_normal(63)) * 0.02 + 4.2
    log_a_trade = b_est * log_b_trade + a_est + (mu + shift)
    z = static_z(log_a_trade, log_b_trade, b_est, a_est, mu, sigma)
    assert np.allclose(z, shift / sigma, atol=1e-6)
    # Trade penceresinden yeniden merkezlenseydi ortalama 0 olurdu; burada ~3 (kanıt).
    assert abs(float(np.mean(z)) - 3.0) < 1e-3


# ---------------------------------------------------------------------------
# (3) simulate_pair: giriş→çıkış işlemi + maliyet
# ---------------------------------------------------------------------------
def test_simulate_pair_giris_cikis_islemi_uretir():
    """z barız girişi (>=2) sonra ortalamaya dönüşü (<=0.5) → tam bir işlem, exit_reason='mean'."""
    n = 10
    dates = pd.to_datetime([f"2022-01-{d:02d}" for d in range(1, n + 1)])
    close_a = np.full(n, 100.0)
    close_b = np.full(n, 100.0)
    beta_hold = np.ones(n)
    z = np.array([0.0, 2.5, 2.5, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])   # gir @1, çık @4
    ser, trades = simulate_pair(dates, close_a, close_b, z, beta_hold, z_entry=2.0,
                                z_exit=0.5, z_stop=3.5, cost_bps=5.0, involves_removed=False)
    assert len(trades) == 1
    tr = trades[0]
    assert tr["side"] == -1 and tr["exit_reason"] == "mean"   # z>0 → short spread
    # Sabit fiyat → brüt 0; net = yalnız round-trip maliyet ≈ -4·5bps = -20 bps.
    assert abs(tr["net_ret"] - (-4 * 5e-4)) < 1e-9


def test_simulate_pair_time_stop_pencere_sonu():
    """z hep girişte kalırsa (dönüş yok) pozisyon pencere sonunda time-stop ile kapanır."""
    n = 6
    dates = pd.to_datetime([f"2022-02-{d:02d}" for d in range(1, n + 1)])
    close_a = np.full(n, 50.0)
    close_b = np.full(n, 50.0)
    beta_hold = np.ones(n)
    z = np.array([-3.0, -3.0, -3.0, -3.0, -3.0, -3.0])   # sürekli giriş (long spread), dönüş yok
    ser, trades = simulate_pair(dates, close_a, close_b, z, beta_hold, z_entry=2.0,
                                z_exit=0.5, z_stop=3.5, cost_bps=0.0, involves_removed=True)
    assert len(trades) == 1
    assert trades[0]["exit_reason"] == "time" and trades[0]["side"] == 1
    assert trades[0]["involves_removed"] is True


# ---------------------------------------------------------------------------
# (4) select_window_pairs: coint MOCK'lanır (ağsız, hızlı) — eleme + sıralama
# ---------------------------------------------------------------------------
def _make_panel(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=40, freq="B")
    cols = {}
    for s in ["AAA", "BBB", "CCC", "DDD", "EEE"]:
        cols[s] = 100.0 * np.exp(np.cumsum(rng.standard_normal(40) * 0.01))
    return pd.DataFrame(cols, index=idx)


def test_select_window_pairs_coint_mock_siralar_ve_eler(monkeypatch):
    """coint MOCK'lanır: p-değeri sembol adına göre deterministik. Sonuç p'ye göre ARTAN sıralı;
    PIT üye olmayan / formation TAM dolu olmayan semboller elenir. Gerçek coint asla çağrılmaz."""
    panel = _make_panel()

    def fake_coint(la, lb, *a, **k):
        # p = ilk elemanların uzaklığına bağlı deterministik sahte değer (ağsız)
        return (-3.0, float(abs(la[0] - lb[0]) % 0.1), np.array([0, 0, 0]))

    monkeypatch.setattr("research.pairs_test.coint", fake_coint)
    members = {"AAA", "BBB", "CCC", "DDD", "EEE"}
    pairs = select_window_pairs(panel, t0=30, form=30, members=members, max_cand=10)
    assert len(pairs) >= 3
    pvals = [p["pval"] for p in pairs]
    assert pvals == sorted(pvals)                     # p'ye göre artan sıralı
    for p in pairs:                                   # beta/mu/sigma hesaplanmış
        assert np.isfinite(p["beta"]) and p["sigma"] >= 0.0


def test_select_window_pairs_uye_olmayani_eler(monkeypatch):
    """PIT üye kümesinde OLMAYAN sembol hiçbir çiftte görünmemeli (look-ahead/survivorship
    koruması: yalnız formation-tarihli üyeler)."""
    panel = _make_panel()
    monkeypatch.setattr("research.pairs_test.coint",
                        lambda la, lb, *a, **k: (-3.0, 0.01, np.array([0, 0, 0])))
    members = {"AAA", "BBB", "CCC"}                   # DDD, EEE üye değil
    pairs = select_window_pairs(panel, t0=30, form=30, members=members, max_cand=10)
    assert pairs, "en az bir çift beklenirdi"
    syms = {p["a"] for p in pairs} | {p["b"] for p in pairs}
    assert "DDD" not in syms and "EEE" not in syms


def test_select_window_pairs_formation_nan_eler(monkeypatch):
    """Formation penceresinde NaN (eksik veri = IPO sonrası/delist) olan sembol elenir."""
    panel = _make_panel()
    panel.iloc[5, panel.columns.get_loc("BBB")] = np.nan   # formation içinde boşluk
    monkeypatch.setattr("research.pairs_test.coint",
                        lambda la, lb, *a, **k: (-3.0, 0.02, np.array([0, 0, 0])))
    members = {"AAA", "BBB", "CCC", "DDD", "EEE"}
    pairs = select_window_pairs(panel, t0=30, form=30, members=members, max_cand=10)
    syms = {p["a"] for p in pairs} | {p["b"] for p in pairs}
    assert "BBB" not in syms
