"""portfolio.py testleri — GÖLGE portföy yönetimi (Tur-16 WP-P)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nihai.ai import portfolio, quant_math
from nihai.ai.reasoner import CLOSING


@pytest.fixture(autouse=True)
def _tmp_file(tmp_path, monkeypatch):
    monkeypatch.setattr(portfolio, "PORTFOLIO_PROPOSALS_FILE", tmp_path / "ai_portfolio_proposals.jsonl")
    return tmp_path


def _make_panel(symbols, n_bars=150, seed=1, mus=None, sigmas=None):
    idx = pd.bdate_range("2021-01-04", periods=n_bars)
    data = {}
    for i, s in enumerate(symbols):
        mu = mus[i] if mus is not None else 0.05 + 0.01 * i
        sigma = sigmas[i] if sigmas is not None else 0.15 + 0.02 * i
        paths = quant_math.mc_gbm_paths(100.0, mu, sigma, n_bars - 1, 1, seed + i)
        data[s] = paths[0]
    return pd.DataFrame(data, index=idx)


def _candidates(symbols, **overrides):
    out = []
    for s in symbols:
        c = {"symbol": s, "strategy": "event_breakout"}
        c.update(overrides.get(s, {}))
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# _project_capped_simplex — kısıt ihlali imkânsızlığının matematiksel temeli
# ---------------------------------------------------------------------------
def test_project_capped_simplex_kutu_ihlali_imkansiz_adversarial():
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([0.3, 0.3, 0.3])
    w = np.array([np.nan, np.inf, -np.inf])
    out = portfolio._project_capped_simplex(w, lo, hi, 1.0)
    assert np.all(out >= lo - 1e-9)
    assert np.all(out <= hi + 1e-9)
    assert np.all(np.isfinite(out))


def test_project_capped_simplex_butce_ulasilabilir():
    lo = np.zeros(4)
    hi = np.full(4, 0.5)
    w = np.array([10.0, -5.0, 0.2, 0.3])
    out = portfolio._project_capped_simplex(w, lo, hi, 1.0)
    assert np.all(out >= -1e-9) and np.all(out <= 0.5 + 1e-9)
    assert abs(float(np.sum(out)) - 1.0) < 1e-6


def test_project_capped_simplex_butce_ulasilmaz_kutu_korunur():
    lo = np.zeros(5)
    hi = np.full(5, 0.1)   # n*hi = 0.5 < budget=1.0 -> ulaşılamaz
    w = np.full(5, 0.2)
    out = portfolio._project_capped_simplex(w, lo, hi, 1.0)
    assert np.all(out <= 0.1 + 1e-9)
    assert np.all(out >= -1e-9)
    assert float(np.sum(out)) <= 0.5 + 1e-6


def test_project_capped_simplex_zaten_uygun_degismez():
    lo, hi = np.zeros(2), np.ones(2)
    w = np.array([0.4, 0.6])
    out = portfolio._project_capped_simplex(w, lo, hi, 1.0)
    assert np.allclose(out, w)


# ---------------------------------------------------------------------------
# min-varyans
# ---------------------------------------------------------------------------
def test_min_variance_iki_varlik_korelasyonsuz_bilinen():
    cov = np.diag([0.04, 0.01])   # ters-varyans: w ∝ [1/0.04, 1/0.01] -> [0.2, 0.8]
    w = portfolio.min_variance_weights(cov)
    assert w is not None
    assert np.allclose(w, [0.2, 0.8], atol=1e-3)


def test_min_variance_gecersiz_girdi_none():
    assert portfolio.min_variance_weights(np.array([[1.0, 2.0, 3.0]])) is None
    assert portfolio.min_variance_weights(np.array([[np.nan, 0.0], [0.0, 1.0]])) is None


def test_min_variance_tek_varlik():
    w = portfolio.min_variance_weights(np.array([[0.05]]))
    assert w is not None and np.allclose(w, [1.0])


# ---------------------------------------------------------------------------
# maks-Sharpe
# ---------------------------------------------------------------------------
def test_max_sharpe_yuksek_getirili_varliga_yogunlasir():
    mu = np.array([0.20, 0.01, 0.01])
    cov = np.diag([0.04, 0.04, 0.04])
    w = portfolio.max_sharpe_weights(mu, cov, bounds=(np.zeros(3), np.ones(3)))
    assert w is not None
    assert w[0] > w[1] and w[0] > w[2]
    assert abs(float(np.sum(w)) - 1.0) < 1e-6


def test_max_sharpe_gecersiz_girdi_none():
    assert portfolio.max_sharpe_weights(np.array([1.0]), np.array([[1.0, 0.0], [0.0, 1.0]])) is None


# ---------------------------------------------------------------------------
# HRP
# ---------------------------------------------------------------------------
def test_hrp_bilinen_iki_varlik_ters_varyans_esdeger():
    cov = np.diag([0.04, 0.01])
    w = portfolio.hrp_weights(cov)
    assert w is not None
    assert np.allclose(w, [0.2, 0.8], atol=1e-6)   # n=2 için HRP == ters-varyans (kapalı-form)


def test_hrp_tek_varlik():
    w = portfolio.hrp_weights(np.array([[0.02]]))
    assert w is not None and np.allclose(w, [1.0])


def test_hrp_uc_varlik_toplami_bir_ve_pozitif():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(50, 3))
    cov = np.cov(a.T)
    w = portfolio.hrp_weights(cov)
    assert w is not None
    assert np.all(w >= -1e-9)
    assert abs(float(np.sum(w)) - 1.0) < 1e-6


def test_hrp_gecersiz_girdi_none():
    assert portfolio.hrp_weights(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])) is None
    assert portfolio.hrp_weights(np.array([[np.inf, 0.0], [0.0, 1.0]])) is None


def test_quasi_diag_order_kucuk_n():
    assert portfolio._quasi_diag_order(np.eye(1)) == [0]
    assert portfolio._quasi_diag_order(np.eye(2)) == [0, 1]


# ---------------------------------------------------------------------------
# ERC
# ---------------------------------------------------------------------------
def test_erc_esit_risk_katkisi_korelasyonsuz():
    sigmas = np.array([0.2, 0.3, 0.1])
    cov = np.diag(sigmas ** 2)
    w = portfolio.erc_weights(cov)
    assert w is not None
    assert abs(float(np.sum(w)) - 1.0) < 1e-4
    # w ∝ 1/sigma (ters-vol, ters-varyans DEĞİL) — korelasyonsuz ERC kapalı-formu
    expected = (1.0 / sigmas) / np.sum(1.0 / sigmas)
    assert np.allclose(w, expected, atol=5e-3)
    # risk katkıları eşit mi (asıl tanım)
    port_var = float(w @ cov @ w)
    rc = (w * (cov @ w)) / port_var
    assert np.allclose(rc, 1.0 / 3.0, atol=5e-3)


def test_erc_gecersiz_girdi_none():
    assert portfolio.erc_weights(np.array([[1.0, 2.0, 3.0]])) is None


def test_erc_tek_varlik():
    w = portfolio.erc_weights(np.array([[0.03]]))
    assert w is not None and np.allclose(w, [1.0])


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------
def test_bl_gorus_yok_mu_esittir_pi():
    cov = np.diag([0.04, 0.09, 0.01])
    mw = np.array([0.5, 0.3, 0.2])
    bl = portfolio.black_litterman(cov, mw, views=None)
    assert bl is not None
    assert bl["n_views"] == 0
    assert np.allclose(bl["mu_bl"], bl["pi"])
    assert np.allclose(bl["pi"], portfolio._BL_DELTA_DEFAULT * cov @ mw)


def test_bl_gorus_yok_piyasa_agirligina_limit():
    cov = np.diag([0.04, 0.09, 0.01])
    mw = portfolio._inverse_vol_weights(cov)
    bl = portfolio.black_litterman(cov, mw, views=None)
    assert bl is not None
    w = portfolio.max_sharpe_weights(bl["mu_bl"], bl["cov_bl"], bounds=(np.zeros(3), np.ones(3)))
    assert w is not None
    assert np.allclose(w, mw, atol=3e-2)


def test_bl_gorus_ile_mu_degisir_yon_dogru():
    cov = np.diag([0.04, 0.04])
    mw = np.array([0.5, 0.5])
    bl_no_view = portfolio.black_litterman(cov, mw, views=None)
    bl_view = portfolio.black_litterman(cov, mw, views=[{"index": 0, "q": 0.10, "confidence": 0.9}])
    assert bl_no_view is not None and bl_view is not None
    assert bl_view["mu_bl"][0] > bl_no_view["mu_bl"][0]
    assert bl_view["n_views"] == 1


def test_bl_gecersiz_girdi_none():
    assert portfolio.black_litterman(np.array([[1.0, 2.0, 3.0]]), np.array([1.0])) is None
    assert portfolio.black_litterman(np.eye(2), np.array([1.0, 2.0, 3.0])) is None


def test_bl_bozuk_gorus_formati_degrade_eder():
    cov = np.diag([0.04, 0.09])
    mw = np.array([0.5, 0.5])
    bl = portfolio.black_litterman(cov, mw, views=[{"index": "bozuk"}])
    assert bl is not None and bl["n_views"] == 0   # bozuk görüş -> görüşsüz davranışa degrade


# ---------------------------------------------------------------------------
# CVaR-LP
# ---------------------------------------------------------------------------
def test_cvar_lp_kucuk_analitik_ornek_risksiz_varliga_yogunlasir():
    n_scn = 20
    r0 = np.full(n_scn, 0.01)                 # risksiz (sabit küçük kazanç)
    r1 = np.full(n_scn, 0.02)
    r1[-1] = -0.5                              # 1/20 = %5 kuyrukta büyük çöküş
    returns = np.column_stack([r0, r1])
    w = portfolio.cvar_lp_weights(returns, alpha=0.05)
    assert w is not None
    assert w[0] > 0.99
    assert w[1] < 0.01


def test_cvar_lp_gecersiz_alpha_none():
    returns = np.random.default_rng(0).normal(size=(20, 3)) * 0.01
    assert portfolio.cvar_lp_weights(returns, alpha=0.0) is None
    assert portfolio.cvar_lp_weights(returns, alpha=1.0) is None


def test_cvar_lp_gecersiz_returns_none():
    assert portfolio.cvar_lp_weights(np.array([[1.0, 2.0]])) is None
    assert portfolio.cvar_lp_weights(np.array([[np.nan, 0.0], [0.1, 0.2]])) is None


# ---------------------------------------------------------------------------
# Çok-varlık Kelly
# ---------------------------------------------------------------------------
def test_multi_asset_kelly_bilinen_iki_varlik_korelasyonsuz():
    mu = np.array([0.08, 0.02])
    cov = np.diag([0.04, 0.01])
    w = portfolio.multi_asset_kelly(mu, cov, fraction=0.25, cap=1.0)
    assert w is not None
    assert np.allclose(w, [0.5, 0.5], atol=1e-6)   # (0.08/0.04)*0.25=0.5, (0.02/0.01)*0.25=0.5


def test_multi_asset_kelly_cap_uygulanir():
    mu = np.array([0.08, 0.02])
    cov = np.diag([0.04, 0.01])
    w = portfolio.multi_asset_kelly(mu, cov, fraction=0.25, cap=0.3)
    assert w is not None
    assert np.allclose(w, [0.3, 0.3], atol=1e-9)
    assert np.all(np.abs(w) <= 0.3 + 1e-9)


def test_multi_asset_kelly_gecersiz_girdi_none():
    assert portfolio.multi_asset_kelly(np.array([1.0]), np.array([[1.0, 0.0], [0.0, 1.0]])) is None


# ---------------------------------------------------------------------------
# Ciro cezası
# ---------------------------------------------------------------------------
def test_turnover_cost_monotonik_bps():
    symbols = ["A", "B"]
    w = np.array([0.6, 0.4])
    prev = {"A": 0.2, "B": 0.8}
    c10 = portfolio._turnover_cost(symbols, w, prev, 10.0)
    c20 = portfolio._turnover_cost(symbols, w, prev, 20.0)
    assert c10 > 0.0
    assert abs(c20 - 2.0 * c10) < 1e-12


def test_turnover_cost_prev_yoksa_sifir():
    assert portfolio._turnover_cost(["A"], np.array([0.5]), None, 10.0) == 0.0
    assert portfolio._turnover_cost(["A"], np.array([0.5]), {"A": 0.1}, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Kovaryans yardımcıları
# ---------------------------------------------------------------------------
def test_single_factor_cov_kisa_veri_none():
    assert portfolio._single_factor_cov(np.zeros((2, 3))) is None


def test_single_factor_cov_calisir():
    rng = np.random.default_rng(5)
    r = rng.normal(scale=0.01, size=(80, 4))
    cov = portfolio._single_factor_cov(r)
    assert cov is not None
    assert cov.shape == (4, 4)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T, atol=1e-12)   # simetrik olmalı


def test_condition_number_tekil_matris_cokmez():
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    cn = portfolio._condition_number(singular)
    assert cn > 1e6 or cn == float("inf")


def test_inverse_vol_weights_sum_bir():
    cov = np.diag([0.04, 0.09, 0.01])
    w = portfolio._inverse_vol_weights(cov)
    assert abs(float(np.sum(w)) - 1.0) < 1e-9
    assert w[2] > w[0] > w[1]   # vol sırası (0.1 < 0.2 < 0.3) -> ağırlık ters sıralı


# ---------------------------------------------------------------------------
# lazy qp_solve
# ---------------------------------------------------------------------------
def test_lazy_qp_solve_modul_yokken_none(monkeypatch):
    # advanced_math artık MEVCUT (Dalga-2) — yokluk senaryosu sys.modules blokajıyla simüle.
    import sys

    import nihai.ai
    monkeypatch.delattr(nihai.ai, "advanced_math", raising=False)
    monkeypatch.setitem(sys.modules, "nihai.ai.advanced_math", None)
    assert portfolio._lazy_qp_solve() is None


def test_lazy_qp_solve_modul_varken_callable():
    fn = portfolio._lazy_qp_solve()
    assert fn is not None and callable(fn)


# ---------------------------------------------------------------------------
# Panel normalizasyonu
# ---------------------------------------------------------------------------
def test_normalize_panel_dataframe_girdi():
    df = _make_panel(["AAA", "BBB"], n_bars=30)
    symbols, prices = portfolio._normalize_panel(df, ["AAA", "BBB"])
    assert symbols == ["AAA", "BBB"]
    assert prices.shape == (30, 2)


def test_normalize_panel_dict_girdi():
    data = {"AAA": [100.0, 101.0, 102.0, 103.0, 104.0], "BBB": [50.0, 50.5, 51.0, 51.5, 52.0]}
    symbols, prices = portfolio._normalize_panel(data, ["AAA", "BBB"])
    assert symbols == ["AAA", "BBB"]
    assert prices.shape == (5, 2)


def test_normalize_panel_negatif_fiyat_reddedilir():
    data = {"AAA": [100.0, -5.0, 102.0, 103.0], "BBB": [50.0, 50.5, 51.0, 51.5]}
    symbols, prices = portfolio._normalize_panel(data, ["AAA", "BBB"])
    assert symbols == [] and prices.size == 0


def test_normalize_panel_yetersiz_sutun():
    data = {"CCC": [1.0, 2.0, 3.0]}
    symbols, prices = portfolio._normalize_panel(data, ["AAA", "BBB"])
    assert symbols == [] and prices.size == 0


def test_normalize_panel_gecersiz_tip():
    symbols, prices = portfolio._normalize_panel(object(), ["AAA"])
    assert symbols == [] and prices.size == 0


# ---------------------------------------------------------------------------
# compare_methods — tek yönteme körlük yok
# ---------------------------------------------------------------------------
def test_compare_methods_tum_beklenen_anahtarlar_var():
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    df = _make_panel(symbols, n_bars=120)
    log_p = np.log(df[symbols].to_numpy())
    returns = np.diff(log_p, axis=0)
    cov = quant_math.ledoit_wolf_cov(returns)
    mu = returns.mean(axis=0)
    bounds = (np.zeros(4), np.full(4, 0.5))
    methods = portfolio.compare_methods(symbols, returns, cov, mu, _candidates(symbols), bounds,
                                        1.0, 0.05, 0.25, 0.5)
    for key in ("min_var", "max_sharpe", "hrp", "erc", "black_litterman", "cvar_lp", "kelly_multi_asset"):
        assert key in methods, f"eksik yöntem: {key}"
    assert methods["kelly_multi_asset"]["budget_constrained"] is False
    for key in ("min_var", "max_sharpe", "hrp", "erc", "black_litterman", "cvar_lp"):
        w = np.array(list(methods[key]["weights"].values()))
        assert abs(float(np.sum(w)) - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# propose() — uçtan uca
# ---------------------------------------------------------------------------
def test_propose_bos_candidates():
    r = portfolio.propose([], {})
    assert r["methods"] == {}
    assert CLOSING in r["text"]
    assert "boş" in r["notes"][0]


def test_propose_gecersiz_sembol_alani():
    r = portfolio.propose([{"strategy": "x"}], {})
    assert r["methods"] == {}
    assert CLOSING in r["text"]


def test_propose_yetersiz_panel_verisi():
    r = portfolio.propose(_candidates(["AAA", "BBB"]), {"AAA": [1.0]})
    assert r["methods"] == {}
    assert CLOSING in r["text"]


def test_propose_tam_akis_yontem_karsilastirma():
    symbols = ["AAA", "BBB", "CCC"]
    df = _make_panel(symbols, n_bars=100)
    r = portfolio.propose(_candidates(symbols), df)
    assert r["universe"] == symbols
    assert isinstance(r["condition_number"], float)
    assert len(r["methods"]) >= 4
    assert CLOSING in r["text"]
    assert r["stress"]["monte_carlo"]["n_paths"] > 0
    assert "by_symbol" in r["attribution"]


def test_propose_dusen_sembol_notu():
    symbols = ["AAA", "BBB"]
    df = _make_panel(symbols, n_bars=60)
    cands = _candidates([*symbols, "ZZZ"])
    r = portfolio.propose(cands, df)
    assert any("ZZZ" in n for n in r["notes"])


def test_propose_kisit_ihlali_imkansiz_dar_max_weight():
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    df = _make_panel(symbols, n_bars=90)
    r = portfolio.propose(_candidates(symbols), df, constraints={"max_weight": 0.1})
    assert r["methods"]
    for name, m in r["methods"].items():
        if not m.get("budget_constrained", True):
            continue
        for w in m["weights"].values():
            assert w <= 0.1 + 1e-6, f"{name} pozisyon tavanını ihlal etti: {w}"
            assert w >= -1e-9


def test_propose_determinizm():
    symbols = ["AAA", "BBB", "CCC"]
    df = _make_panel(symbols, n_bars=80)
    r1 = portfolio.propose(_candidates(symbols), df, seed=11)
    r2 = portfolio.propose(_candidates(symbols), df, seed=11)
    assert r1["condition_number"] == r2["condition_number"]
    for name in r1["methods"]:
        w1 = r1["methods"][name]["weights"]
        w2 = r2["methods"][name]["weights"]
        for s in w1:
            assert w1[s] == pytest.approx(w2[s], abs=1e-9)
    assert r1["stress"]["monte_carlo"] == r2["stress"]["monte_carlo"]


def test_propose_atomik_jsonl_kaydi(tmp_path):
    symbols = ["AAA", "BBB"]
    df = _make_panel(symbols, n_bars=60)
    portfolio.propose(_candidates(symbols), df)
    assert portfolio.PORTFOLIO_PROPOSALS_FILE.exists()
    lines = portfolio.PORTFOLIO_PROPOSALS_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert "gölge" in rec and "GÖLGE" in rec["gölge"]


def test_propose_kapanis_cumlesi_text_icinde():
    symbols = ["AAA", "BBB"]
    df = _make_panel(symbols, n_bars=60)
    r = portfolio.propose(_candidates(symbols), df)
    assert r["text"].strip().endswith(CLOSING)


def test_propose_allow_short_negatif_agirlik_mumkun():
    symbols = ["AAA", "BBB", "CCC"]
    df = _make_panel(symbols, n_bars=100, mus=[0.30, 0.05, -0.30], sigmas=[0.15, 0.15, 0.15])
    r = portfolio.propose(_candidates(symbols), df, constraints={"allow_short": True, "max_weight": 0.6})
    assert "max_sharpe" in r["methods"]
    w = r["methods"]["max_sharpe"]["weights"]
    for val in w.values():
        assert -0.6 - 1e-6 <= val <= 0.6 + 1e-6


def test_propose_prev_weights_turnover_uygulanir():
    symbols = ["AAA", "BBB", "CCC"]
    df = _make_panel(symbols, n_bars=90)
    prev = {"AAA": 0.0, "BBB": 0.0, "CCC": 1.0}
    r = portfolio.propose(_candidates(symbols), df,
                          constraints={"turnover_penalty_bps": 50.0, "prev_weights": prev})
    assert any(m["turnover_cost_est"] > 0.0 for m in r["methods"].values() if m.get("budget_constrained", True))


def test_propose_bl_gorusleri_candidates_posterior_kullanir():
    symbols = ["AAA", "BBB", "CCC"]
    df = _make_panel(symbols, n_bars=90)
    cands = _candidates(symbols, AAA={"posterior": 0.9, "conformal_interval": (0.6, 0.95)})
    r = portfolio.propose(cands, df)
    assert "black_litterman" in r["methods"]
    assert r["methods"]["black_litterman"]["n_views"] >= 1


def test_build_bl_views_gecersiz_posterior_atlanir():
    views = portfolio._build_bl_views([{"symbol": "AAA", "posterior": "bozuk"}], ["AAA"])
    assert views == []


def test_attribution_by_symbol_and_strategy():
    symbols = ["AAA", "BBB"]
    w = np.array([0.6, 0.4])
    mu = np.array([0.01, 0.02])
    cands = [{"symbol": "AAA", "strategy": "s1"}, {"symbol": "BBB", "strategy": "s2"}]
    attr = portfolio._attribution(symbols, w, mu, cands)
    assert attr["by_symbol"]["AAA"] == pytest.approx(0.006)
    assert attr["by_symbol"]["BBB"] == pytest.approx(0.008)
    assert set(attr["by_strategy"]) == {"s1", "s2"}
