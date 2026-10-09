"""test_tur72_yol_haritasi.py — dış yol haritasından uygulanan altı ailenin kilidi.

Kapsanan başlıklar (`quant geliştirme yol haritası.txt`):
  §2.1  Dupire yerel volatilite      → options_math.dupire_local_vol_svi
  §6.1  EGARCH / GJR-GARCH           → quant_math.egarch_fit / gjr_garch_fit
  §6.3  Johansen + VECM              → statarb.johansen_test / vecm_fit
  §7.1  Avellaneda-Lee öz-portföy    → statarb.avellaneda_lee_scores
  §7.5  Varyans swap replikasyonu    → options_math.variance_swap_strike
  §9.1  Kümülatif Beklenti Teorisi   → quant_math.cpt_prospect
  §10.3 Marčenko-Pastur / RMT + MST  → quant_math.denoise_corr_mp / mst_edges

TEST FELSEFESİ (Tur-49/Tur-69 dersleri): "koştu ve patlamadı" bir test değildir.
Her aile ya BİLİNEN BİR ÖZDEŞLİĞE (düz yüzeyde yerel vol = örtük vol; sabit-BS
zincirinde K_vol = σ; yerel Johansen = statsmodels) ya da DİFERANSİYEL bir
karşılaştırmaya (asimetrik seri ↔ simetrik seri) bağlanır. Ayrıca uç sözleşmesi
ayrıca sınanır: yazılıp panoda OKUNMAYAN alan bu depoda kusurdur (Tur-37 kuralı).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_tur56_quant_modelleri import _FakeProvider

import nihai.web.api_quant as aq
from nihai.ai import options_math as om
from nihai.ai import quant_math as qm
from nihai.ai import statarb as sa
from nihai.web.web_dashboard import app

client = TestClient(app)
DASHBOARD_HTML = Path(__file__).resolve().parents[1] / "nihai" / "web" / "web_dashboard.html"


# ---------------------------------------------------------------------------
# Sentetik seriler — deterministik (seed sabit), ağ YOK
# ---------------------------------------------------------------------------
def _gjr_series(gamma: float, n: int = 900, seed: int = 7) -> np.ndarray:
    """Bilinen kaldıraçlı GJR süreci. gamma=0 ⇒ süreç SİMETRİK GARCH(1,1)'dir."""
    rng = np.random.default_rng(seed)
    om_, al, be = 1e-6, 0.03, 0.88
    s2 = np.empty(n)
    e = np.empty(n)
    s2[0] = om_ / (1.0 - al - gamma / 2.0 - be)
    for t in range(n):
        e[t] = math.sqrt(s2[t]) * rng.standard_normal()
        if t + 1 < n:
            s2[t + 1] = om_ + (al + gamma * (e[t] < 0)) * e[t] ** 2 + be * s2[t]
    return e + 0.0003


def _factor_panel(n: int = 300, k: int = 8, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 1)) * 0.012
    beta = rng.uniform(0.5, 1.5, (1, k))
    return f @ beta + rng.standard_normal((n, k)) * 0.006


@pytest.fixture
def fake_provider(monkeypatch):
    monkeypatch.setattr(aq, "_provider", lambda: _FakeProvider())
    monkeypatch.setattr("nihai.execution.watchlist.current", lambda: ["AAA", "BBB", "CCC"])
    return _FakeProvider()


# ===========================================================================
# §6.1 — ASİMETRİK GARCH
# ===========================================================================
def test_gjr_kaldiraci_bulur_ve_simetride_bulmaz():
    """DİFERANSİYEL: aynı kod, iki seri. Kaldıraçlı seride gamma açıkça pozitif,
    simetrik seride sıfıra yakın olmalı — tek seride "pozitif çıktı" demek
    tahminciyi değil şansı ölçerdi."""
    lev = qm.gjr_garch_fit(_gjr_series(gamma=0.12))
    sym = qm.gjr_garch_fit(_gjr_series(gamma=0.0))
    assert lev is not None and sym is not None
    assert lev["gamma"] > 0.05, lev["gamma"]
    assert lev["leverage"] is True
    assert abs(sym["gamma"]) < lev["gamma"] / 2.0, (sym["gamma"], lev["gamma"])


def test_gjr_garch11in_genellemesidir_olabilirlik_dusmez():
    """GARCH(1,1) = GJR'nin gamma=0 kısıtlı hali. İç içe modelde kısıtsız fit'in
    log-olabilirliği kısıtlıdan KÜÇÜK ÇIKAMAZ; çıkıyorsa optimizasyon kusurludur."""
    r = _gjr_series(gamma=0.12)
    g11 = qm.garch11_fit(r)
    gjr = qm.gjr_garch_fit(r)
    assert g11 is not None and gjr is not None
    assert gjr["loglik"] >= g11["loglik"] - 1e-6, (gjr["loglik"], g11["loglik"])
    assert 2.0 * (gjr["loglik"] - g11["loglik"]) > 3.84   # χ²(1) %95


def test_egarch_kaldiraci_TERS_isaretle_bulur():
    """EGARCH parametrizasyonunda kaldıraç gamma < 0'dır (GJR'de gamma > 0).
    İşaret konvansiyonunu karıştırmak, kaldıracı ters okumak demektir."""
    eg = qm.egarch_fit(_gjr_series(gamma=0.12))
    assert eg is not None
    assert eg["gamma"] < 0.0, eg["gamma"]
    assert eg["leverage"] is True
    assert abs(eg["beta"]) < 1.0


def test_asimetrik_ongoru_uzun_ufukta_kosulsuz_seviyeye_yaklasir():
    r = _gjr_series(gamma=0.12)
    gjr = qm.gjr_garch_fit(r)
    assert gjr is not None and gjr["uncond_var"] is not None
    target = math.sqrt(gjr["uncond_var"])
    f1 = qm.asym_garch_forecast(gjr, 1)
    f200 = qm.asym_garch_forecast(gjr, 200)
    assert f1 is not None and f200 is not None
    assert abs(f200 - target) < abs(f1 - target) + 1e-12
    assert abs(f200 - target) / target < 0.05


@pytest.mark.parametrize("fn", [qm.gjr_garch_fit, qm.egarch_fit])
def test_asimetrik_garch_none_kapilari(fn):
    assert fn([1.0, 2.0]) is None                       # kısa
    assert fn([float("nan")] * 100) is None             # NaN
    assert fn(np.zeros(200)) is None                    # sıfır varyans
    assert fn("abc") is None                            # dönüştürülemez


def test_asym_forecast_gecersiz_sozlukte_none():
    assert qm.asym_garch_forecast({}, 1) is None
    assert qm.asym_garch_forecast({"model": "yok"}, 1) is None
    assert qm.asym_garch_forecast(qm.gjr_garch_fit(_gjr_series(0.1)), 0) is None


# ===========================================================================
# §10.3 — RASTGELE MATRİS TEORİSİ + PİYASA AĞI
# ===========================================================================
def test_mp_siniri_kapali_formu_tutturur():
    lo, hi = qm.mp_bounds(20, 500)
    q = 500 / 20
    assert hi == pytest.approx((1 + math.sqrt(1 / q)) ** 2)
    assert lo == pytest.approx((1 - math.sqrt(1 / q)) ** 2)
    assert qm.mp_bounds(500, 20) is None                # q ≤ 1 → MP desteği tanımsız
    assert qm.mp_bounds(0, 100) is None


def test_rmt_tek_faktoru_bulur_ve_kosul_sayisini_dusurur():
    """Panel BİR ortak faktörle üretildi; λ⁺ üstünde tam olarak 1 özdeğer kalmalı."""
    d = qm.denoise_corr_mp(_factor_panel(n=400, k=15))
    assert d is not None
    assert d["n_signal"] == 1, (d["n_signal"], d["eigenvalues"][:3])
    assert d["n_signal"] + d["n_noise"] == 15
    assert d["cond_after"] <= d["cond_before"]


def test_rmt_ciktisi_gecerli_korelasyon_matrisidir():
    d = qm.denoise_corr_mp(_factor_panel(n=400, k=12))
    assert d is not None
    c = np.asarray(d["corr_denoised"])
    assert np.allclose(np.diag(c), 1.0)
    assert np.allclose(c, c.T)
    assert float(np.min(np.linalg.eigvalsh(c))) > -1e-10          # PSD
    assert float(np.trace(c)) == pytest.approx(12.0, abs=1e-9)    # iz korunur


def test_rmt_gozlemden_cok_varlikta_olcum_yapmaz():
    """T ≤ N'de MP desteği tanımsızdır; sayı basmak yerine None (ölçüm-yokluğu)."""
    rng = np.random.default_rng(1)
    assert qm.denoise_corr_mp(rng.standard_normal((10, 20))) is None
    assert qm.denoise_corr_mp(rng.standard_normal((20, 20))) is None


def test_mst_zincir_korelasyonunda_ZİNCİRİ_bulur():
    """rho_ij = 0,9^|i−j| bir YOL grafiğidir; MST tam olarak i—i+1 kenarlarını
    seçmelidir. Bu, "n−1 kenar döndü" gibi zayıf bir sayımdan farklı olarak
    ağacın DOĞRU olduğunu sınar."""
    n = 7
    idx = np.arange(n)
    corr = 0.9 ** np.abs(idx[:, None] - idx[None, :])
    m = qm.mst_edges(corr)
    assert m is not None
    edges = {tuple(sorted((e["i"], e["j"]))) for e in m["edges"]}
    assert edges == {(i, i + 1) for i in range(n - 1)}
    assert sum(m["degree"]) == 2 * (n - 1)
    assert min(m["degree"]) >= 1                                   # bağlı


def test_mst_none_kapilari():
    assert qm.mst_edges([[1.0, 0.5], [0.5, 1.0]]) is None          # n < 3
    assert qm.mst_edges(np.full((4, 4), 2.0)) is None              # |rho| > 1
    assert qm.mst_edges("x") is None


# ===========================================================================
# §9.1 — KÜMÜLATİF BEKLENTİ TEORİSİ
# ===========================================================================
def test_cpt_deger_fonksiyonu_zarardan_kacinmayi_tam_lam_kati_yapar():
    v = qm.cpt_value([-100.0, 0.0, 100.0], lam=2.25)
    assert v is not None
    assert v[1] == 0.0
    assert -v[0] / v[2] == pytest.approx(2.25, rel=1e-12)
    assert qm.cpt_value([1.0], alpha=0.0) is None
    assert qm.cpt_value([1.0], lam=-1.0) is None


def test_prelec_ters_S_ve_rasyonel_sinir():
    w = qm.prelec_weight([0.01, 0.10, 0.50, 0.90, 0.99], delta=0.65)
    assert w is not None
    assert w[0] > 0.01 and w[3] < 0.90                    # küçük şişer, büyük küçülür
    assert np.all(np.diff(w) > 0)                         # monoton
    ident = qm.prelec_weight([0.0, 0.25, 0.5, 0.75, 1.0], delta=1.0, gamma=1.0)
    assert ident is not None
    assert np.allclose(ident, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert qm.prelec_weight([1.5]) is None                # aralık dışı


def test_cpt_pozitif_EV_bahsi_zarardan_kacinmayla_reddedilebilir():
    """p=0,55 · b=1,5 bahsi beklenen değeri POZİTİFtir; CPT bunu λ=2,25'te
    NEGATİF değerler. Kelly ile CPT'nin ayrıştığı yer tam olarak burasıdır."""
    r = qm.cpt_prospect([1.5, -1.0], [0.55, 0.45])
    assert r is not None
    assert r["ev"] > 0.0
    assert r["cpt_value"] < 0.0
    assert r["certainty_equivalent"] < r["ev"]
    assert r["premium"] > 0.0
    w = np.asarray(r["weights"])
    assert np.all(w >= -1e-12)
    assert float(w.sum()) <= 1.0 + 1e-9                   # alt-kesinlik (subcertainty)


def test_cpt_lam_1de_kayip_agirligi_kalkar():
    lo = qm.cpt_prospect([1.5, -1.0], [0.55, 0.45], lam=1.0)
    hi = qm.cpt_prospect([1.5, -1.0], [0.55, 0.45], lam=3.0)
    assert lo is not None and hi is not None
    assert lo["cpt_value"] > hi["cpt_value"]


def test_cpt_none_kapilari():
    assert qm.cpt_prospect([1.0], [0.5]) is None          # olasılık toplamı ≠ 1
    assert qm.cpt_prospect([1.0, -1.0], [0.5]) is None    # uzunluk uyuşmaz
    assert qm.cpt_prospect([1.0, -1.0], [1.5, -0.5]) is None


# ===========================================================================
# §2.1 — DUPIRE YEREL VOLATİLİTE
# ===========================================================================
def _flat_slices(sig: float, ts=(0.1, 0.25, 0.5, 1.0)):
    return [{"t": t, "params": {"a": sig * sig * t, "b": 0.0, "rho": 0.0,
                                "m": 0.0, "sigma": 0.1}} for t in ts]


def test_dupire_duz_yuzeyde_yerel_vol_ortuk_vole_ESITTIR():
    """Bilinen özdeşlik: w(k,T) = σ²T (k'dan bağımsız) ise Dupire σ_LV = σ verir.
    Bu, formülün payını/paydasını birlikte sınayan tek testtir."""
    d = om.dupire_local_vol_svi(_flat_slices(0.25), n_k=15)
    assert d is not None
    lv = np.asarray(d["local_vol"])
    assert np.all(np.isfinite(lv))
    assert float(np.max(np.abs(lv - 0.25))) < 1e-9
    assert d["gecerli_oran"] == 1.0
    assert d["n_calendar_ihlal"] == 0 and d["n_butterfly_ihlal"] == 0


def test_dupire_takvim_arbitrajinda_SAYI_BASMAZ():
    """Toplam varyans vade ile AZALIYORSA yerel varyans negatiftir. Kırpıp pozitif
    bir sayı basmak olmayan bir ölçümü var göstermek olurdu — NaN döner."""
    bad = [{"t": 0.1, "params": {"a": 0.05, "b": 0.0, "rho": 0.0, "m": 0.0, "sigma": 0.1}},
           {"t": 0.5, "params": {"a": 0.01, "b": 0.0, "rho": 0.0, "m": 0.0, "sigma": 0.1}}]
    d = om.dupire_local_vol_svi(bad, n_k=9)
    assert d is not None
    assert d["n_calendar_ihlal"] > 0
    assert d["gecerli_oran"] == 0.0
    assert not np.any(np.isfinite(np.asarray(d["local_vol"])))


def test_dupire_none_kapilari():
    assert om.dupire_local_vol_svi(_flat_slices(0.25, ts=(0.5,))) is None    # tek dilim
    assert om.dupire_local_vol_svi("x") is None
    assert om.dupire_local_vol_svi([{"t": 0.5, "params": {"a": 0.01}},
                                    {"t": 1.0, "params": {"a": 0.02}}]) is None
    # Aynı vadenin iki kopyası ∂w/∂T'yi sıfıra bölerdi → tekilleştirilip None.
    same = _flat_slices(0.25, ts=(0.5, 0.5))
    assert om.dupire_local_vol_svi(same) is None


# ===========================================================================
# §7.5 — VARYANS SWAP REPLİKASYONU
# ===========================================================================
def _bs_chain(sig: float, lo: float = 20.0, hi: float = 201.0, s: float = 100.0,
              t: float = 0.5, r: float = 0.0):
    ks = np.arange(lo, hi, 1.0)
    c = np.array([qm.bs_price(s, float(k), t, r, sig, "c") for k in ks])
    p = np.array([qm.bs_price(s, float(k), t, r, sig, "p") for k in ks])
    return ks, c, p, s * math.exp(r * t), t, r


def test_varyans_swap_sabit_BS_zincirinde_sigmayi_geri_verir():
    """Sabit volatiliteli bir zincirde model-bağımsız K_vol tam olarak σ olmalıdır;
    replikasyonun ağırlıkları (ΔK/K²) yanlışsa bu özdeşlik bozulur."""
    ks, c, p, fwd, t, r = _bs_chain(0.30)
    v = om.variance_swap_strike(ks, c, ks, p, fwd, t, r)
    assert v is not None
    assert v["k_vol"] == pytest.approx(0.30, rel=2e-3)
    assert v["k_var"] == pytest.approx(0.09, rel=4e-3)
    assert v["k0"] <= fwd
    assert v["forward_term"] >= 0.0


def test_varyans_swap_dar_kanatta_ASAGI_yanlidir():
    """Replikasyon sürekli strike sürekliliği varsayar; zincir dar kesilirse kuyruk
    örneklenmez ve K_vol AŞAĞI yanlı çıkar. `kanat_kirpimi` bunu raporlar."""
    ks, c, p, fwd, t, r = _bs_chain(0.30)
    wide = om.variance_swap_strike(ks, c, ks, p, fwd, t, r)
    kn, cn, pn, *_ = _bs_chain(0.30, lo=85.0, hi=116.0)
    narrow = om.variance_swap_strike(kn, cn, kn, pn, fwd, t, r)
    assert wide is not None and narrow is not None
    assert narrow["k_vol"] < wide["k_vol"]
    assert narrow["kanat_kirpimi"] > wide["kanat_kirpimi"]


def test_varyans_swap_none_kapilari():
    ks, c, p, fwd, t, r = _bs_chain(0.30)
    assert om.variance_swap_strike([1.0], [1.0], [1.0], [1.0], 100.0, 0.5) is None
    assert om.variance_swap_strike(ks, c, ks, p, -1.0, t, r) is None      # forward ≤ 0
    assert om.variance_swap_strike(ks, c, ks, p, fwd, 0.0, r) is None     # t ≤ 0
    assert om.variance_swap_strike(ks, c[:-1], ks, p, fwd, t, r) is None  # uzunluk


# ===========================================================================
# §6.3 / §7.1 — JOHANSEN, VECM, AVELLANEDA-LEE
# ===========================================================================
def _coint_pair(n: int = 400, seed: int = 11):
    rng = np.random.default_rng(seed)
    y1 = np.cumsum(rng.standard_normal(n))
    y2 = 2.0 * y1 + rng.standard_normal(n) * 0.5
    return np.column_stack([y1, y2])


def test_yerel_johansen_statsmodelsle_BIT_BIT_ORTUSUR(monkeypatch):
    """Bu modülün statsmodels'siz ortamda da doğru olduğunun TEK kanıtı budur:
    aynı panelde iki yol ayrı ayrı koşulur ve istatistikler karşılaştırılır."""
    pytest.importorskip("statsmodels")
    panel = np.column_stack([_coint_pair(), np.cumsum(
        np.random.default_rng(3).standard_normal(400))])
    ref = sa.johansen_test(panel, det_order=0, k_ar_diff=1)
    assert ref is not None and ref["kaynak"] == "statsmodels"
    monkeypatch.setattr(sa, "_coint_johansen", None)
    loc = sa.johansen_test(panel, det_order=0, k_ar_diff=1)
    assert loc is not None and loc["kaynak"] == "yerel"
    assert np.allclose(ref["trace_stat"], loc["trace_stat"], atol=1e-8)
    assert np.allclose(ref["max_eig_stat"], loc["max_eig_stat"], atol=1e-8)
    assert np.allclose(ref["eigenvalues"], loc["eigenvalues"], atol=1e-10)
    assert ref["rank_trace"] == loc["rank_trace"]


def test_johansen_kointegreyi_bulur_bagimsizda_bulmaz():
    """DİFERANSİYEL: kointegre çiftte rank ≥ 1, bağımsız rastgele yürüyüşlerde 0."""
    rng = np.random.default_rng(4)
    coint = sa.johansen_test(_coint_pair(), det_order=0, k_ar_diff=1)
    indep = sa.johansen_test(np.cumsum(rng.standard_normal((400, 3)), axis=0),
                             det_order=0, k_ar_diff=1)
    assert coint is not None and indep is not None
    assert coint["rank_trace"] >= 1
    assert indep["rank_trace"] == 0


def test_vecm_gercek_kointegrasyon_oranini_geri_verir():
    """y2 = 2·y1 ⇒ durağan bileşim y1 − 0,5·y2'dir. β ilk bileşene normalize
    edilince ikinci bileşen −0,5 civarında çıkmalıdır."""
    v = sa.vecm_fit(_coint_pair(), rank=1, k_ar_diff=1)
    assert v is not None
    beta = np.asarray(v["beta"], dtype=float).ravel()
    ratio = beta[1] / beta[0]
    assert ratio == pytest.approx(-0.5, abs=0.05), ratio
    assert np.asarray(v["ect_series"]).shape[1] == 1
    assert v["rank"] == 1


def test_avellaneda_lee_en_gerilmis_kalintiyi_en_uc_skorla_isaretler():
    """Panele TEK bir varlığın kalıntısına bilinen bir gerilme eklenir; s-skorun
    mutlak değeri o varlıkta en büyük olmalıdır. Skorların "sonlu çıkması"
    tahmincinin doğru varlığı bulduğunu göstermezdi."""
    panel = _factor_panel(n=300, k=6, seed=9)
    bump = np.zeros(300)
    bump[-40:] = 0.004                       # son 40 barda kalıntıyı yukarı it
    panel[:, 2] += bump
    res = sa.avellaneda_lee_scores(panel, n_factors=2,
                                   symbols=[f"S{i}" for i in range(6)])
    assert res is not None
    scores = [abs(a["s_score"]) if a["s_score"] is not None else -1.0
              for a in res["assets"]]
    assert int(np.argmax(scores)) == 2, scores
    assert res["assets"][2]["s_score"] > 0.0          # yukarı gerilme ⇒ pozitif s
    assert np.asarray(res["residual_cumsum"]).shape == (300, 6)


def test_avellaneda_lee_yon_etiketi_esikle_tutarli():
    res = sa.avellaneda_lee_scores(_factor_panel(n=300, k=6, seed=2), n_factors=2,
                                   esik=0.5)
    assert res is not None
    for a in res["assets"]:
        if a["s_score"] is None:
            assert a["yon"] == "FLAT"
        elif a["s_score"] > 0.5:
            assert a["yon"] == "SHORT"
        elif a["s_score"] < -0.5:
            assert a["yon"] == "LONG"
        else:
            assert a["yon"] == "FLAT"


def test_statarb_none_kapilari():
    assert sa.eigenportfolio_returns(np.zeros((10, 3))) is None       # kısa
    assert sa.eigenportfolio_returns(np.zeros((100, 3))) is None      # sıfır varyans
    assert sa.avellaneda_lee_scores(np.full((100, 3), np.nan)) is None
    assert sa.avellaneda_lee_scores(_factor_panel(100, 3), symbols=["A"]) is None
    assert sa.residual_ou_fit(np.arange(50.0)) is None                # b ≥ 1 (trend)
    assert sa.johansen_test("x") is None
    assert sa.vecm_fit(_coint_pair(), rank=0) is None                 # rank < 1


# ===========================================================================
# UÇ SÖZLEŞMESİ — yeni alanlar GERÇEKTEN doluyor mu?
# ===========================================================================
def test_vol_ucu_asimetrik_garchi_yayinliyor(fake_provider):
    d = client.get("/api/quant/vol?symbol=SPY&interval=1d&lookback=400").json()
    assert d["ok"] is True
    a = d["asym"]
    assert a is not None and a["gjr"] and a["egarch"]
    assert len(a["eps_grid"]) == len(a["gjr"]["news"]) == len(a["egarch"]["news"])
    assert len(a["gjr"]["cond_vol"]) == len(d["realized_vol"])
    assert len(a["gjr"]["forecast"]) == d["horizon"]
    assert a["gjr"]["lr_vs_garch"] is not None


def test_portfoy_ucu_rmt_ve_agi_yayinliyor(fake_provider):
    d = client.get("/api/quant/portfolio?symbols=AAA,BBB,CCC,DDD&"
                   "interval=1d&lookback=250").json()
    assert d["ok"] is True
    r, m = d["rmt"], d["mst"]
    assert r is not None and len(r["eigenvalues"]) == 4
    assert r["n_signal"] + r["n_noise"] == 4
    assert r["lambda_plus"] > r["lambda_minus"] > 0
    assert m is not None and len(m["edges"]) == 3          # n−1
    assert m["merkez"] in d["symbols"]


def test_smile_ucu_dupire_ve_varyans_swapi_yayinliyor(fake_provider):
    d = client.get("/api/quant/smile?symbol=SPY").json()
    assert d["ok"] is True
    vs, dp = d["varswap"], d["dupire"]
    assert vs is not None and vs["k_vol"] is not None
    assert vs["n_strikes"] >= 5 and len(vs["contrib"]) == vs["n_strikes"]
    assert set(vs["otm_tip"]) <= {"put", "call", "atm"}
    assert dp is not None and dp["n_slices"] >= 2
    assert len(dp["local_vol"]) == dp["n_slices"]
    assert len(dp["local_vol"][0]) == len(dp["k"])


def test_kelly_ucu_cpti_yayinliyor():
    d = client.get("/api/quant/kelly?p_win=0.55&win_loss=1.5").json()
    assert d["ok"] is True
    c = d["cpt"]
    assert c is not None
    assert c["ev"] > 0 and c["cpt_value"] < 0 and c["kabul"] is False
    assert len(c["lam_grid"]) == len(c["ce_by_lam"])


def test_statarb_ucu_skor_johansen_ve_gerekce_veriyor(fake_provider):
    d = client.get("/api/quant/statarb?symbols=AAA,BBB,CCC,DDD,EEE&"
                   "interval=1d&lookback=300").json()
    assert d["ok"] is True
    assert len(d["assets"]) == 5
    assert len(d["residual_cumsum"]) == 5
    assert len(d["residual_cumsum"][0]) == len(d["dates"])
    for a in d["assets"]:
        assert a["yon"] in {"LONG", "SHORT", "FLAT"}
        assert isinstance(a["reliable"], bool)
    jo = d["johansen"]
    assert jo is not None and jo["kaynak"] in {"statsmodels", "yerel"}
    assert len(jo["trace_stat"]) == jo["k"]
    # VECM YA vardır YA DA neden'i yazılıdır — sessiz boşluk yok (Tur-37).
    assert (d["vecm"] is not None) != (d["vecm_neden"] is not None)


def test_statarb_ucu_fail_soft(monkeypatch):
    """Sağlayıcı yoksa 500 ATMAZ; sekme kararmaz, neden yazılır."""
    monkeypatch.setattr(aq, "_provider", lambda: (_ for _ in ()).throw(RuntimeError("yok")))
    monkeypatch.setattr("nihai.execution.watchlist.current", lambda: ["AAA", "BBB"])
    r = client.get("/api/quant/statarb?symbols=AAA,BBB,CCC")
    assert r.status_code == 200
    assert r.json()["ok"] is False and r.json()["reason"]


# ===========================================================================
# PANO BAĞLANTISI — yazılıp OKUNMAYAN alan kusurdur (Tur-37 kuralı)
# ===========================================================================
@pytest.mark.parametrize("expr", [
    "d.asym", "a.gjr.news", "a.egarch.news", "a.news_garch",        # vol
    "d.rmt", "d.mst", "rmt.lambda_plus", "mst.merkez",              # portföy
    "d.dupire", "d.varswap", "vw.otm_tip", "dp.local_vol",          # gülümseme
    "d.cpt", "d.cpt.lam_grid",                                      # kelly
    "d.assets", "d.residual_cumsum", "d.johansen", "d.vecm_neden",  # statarb
])
def test_yeni_uc_alanlari_panoda_okunuyor(expr):
    assert expr in DASHBOARD_HTML.read_text(encoding="utf-8"), expr


def test_statarb_paneli_katalog_sozlesmesine_uyuyor():
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    for hook in ("qm-p-statarb", "qm-head-statarb", "qm-s-statarb", "qm-f-statarb"):
        assert f'id="{hook}"' in html, hook
    assert "statarb: qmRenderStatarb" in html
    assert "quant/statarb?symbols=" in html
    listed = re.search(r"const QM_MODELS = \[(.*?)\];", html, re.S).group(1)
    assert "'statarb'" in listed


# ===========================================================================
# MUTASYON KAPILARI — yukarıdaki testlerin KAÇIRDIĞI dört sapma (Tur-70 dersi:
# "kilidi mutasyonla sına"). Her biri belirli bir katsayıyı/eşiği tek başına pinler.
# ===========================================================================
def test_rmt_SAF_GURULTUDE_sinyal_bulmaz():
    """λ⁺ eşiğini TEK BAŞINA pinler. Faktörlü panelde gürültü özdeğerleri λ⁻'nin de
    altına düşer, o yüzden orada λ⁺↔λ⁻ karışıklığı görünmez. Saf gürültüde özdeğerler
    [λ⁻, λ⁺] aralığına YAYILIR: yanlış eşik yarısını 'sinyal' ilan eder."""
    rng = np.random.default_rng(21)
    d = qm.denoise_corr_mp(rng.standard_normal((500, 20)))
    assert d is not None
    lo, hi = qm.mp_bounds(20, 500)
    ev = np.asarray(d["eigenvalues"])
    assert float(np.min(ev)) < lo * 1.6         # özdeğerler gerçekten yayılmış
    assert float(np.max(ev)) > hi * 0.8
    # Tracy-Widom dalgalanması en büyüğü λ⁺'nin bir tık üstüne itebilir → 1 tolere.
    assert d["n_signal"] <= 1, (d["n_signal"], ev[:5], hi)


def test_ou_denge_std_teorik_degeri_tutturur():
    """sigma_eq = sqrt(var(zeta)/(1−b²)) formülünü pinler. Düzeltmesiz sqrt(var(zeta))
    yazmak, b→1'de denge yayılımını KATBEKAT hafife alır (b=0,9'da 2,29×)."""
    rng = np.random.default_rng(33)
    b_true, s_true, n = 0.90, 0.05, 20000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = b_true * x[t - 1] + s_true * rng.standard_normal()
    fit = sa.residual_ou_fit(x)
    assert fit is not None
    assert fit["b"] == pytest.approx(b_true, abs=0.02)
    teorik = s_true / math.sqrt(1.0 - b_true ** 2)          # ≈ 0,1147
    assert fit["sigma_eq"] == pytest.approx(teorik, rel=0.05), (fit["sigma_eq"], teorik)
    assert fit["sigma_eq"] > 2.0 * fit["sigma_resid"]       # düzeltme GERÇEKTEN var
    assert fit["half_life"] == pytest.approx(math.log(2) / -math.log(b_true), rel=0.1)


def test_vecm_beta_S11_normalizasyonunu_gercekten_uygular():
    """β' S11 β = I sözleşmesini pinler. Normalizasyon rank-1'de ORANI değiştirmez,
    bu yüzden oran testi atlanmasını göremez; ölçeği burada sınarız."""
    panel = _coint_pair()
    v = sa.vecm_fit(panel, rank=1, k_ar_diff=1)
    assert v is not None
    _, _, s11 = sa._johansen_covariances(panel, 0, 1)
    beta = np.asarray(v["beta"], dtype=float)
    gram = beta.T @ s11 @ beta
    assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-8), gram


def test_dupire_FIYAT_UZAYINDAKI_ozgun_turevle_ortusur():
    """Toplam-varyans biçimindeki Dupire'i, BAĞIMSIZ TÜRETİLMİŞ fiyat-uzayı biçimiyle
    karşılaştırır:  σ_LV² = (∂C/∂T) / (½·K²·∂²C/∂K²)   (r=q=0, forward ölçüsü).
    Düz yüzey testi paydadaki (−¼ − 1/w + k²/w²)·w'² terimini GÖREMEZ (w'=0'dır);
    eğimli/kavisli bir yüzeyde bu terim canlıdır ve işareti burada pinlenir."""
    fwd, base_t = 100.0, 0.5
    params = {"a": 0.010, "b": 0.055, "rho": -0.65, "m": 0.02, "sigma": 0.18}
    ts = (base_t - 0.02, base_t, base_t + 0.02)
    slices = [{"t": t, "params": params} for t in ts]     # w(k,T) = t·w0(k)/base_t?
    # SVI parametreleri vadeden bağımsız verilirse w vade ile SABİT kalır (takvim
    # ihlali). Gerçekçi bir yüzey için toplam varyansı vadeyle ORANTILI ölçekleriz.
    slices = [{"t": t, "params": {k: (v * t / base_t if k in ("a", "b") else v)
                                 for k, v in params.items()}} for t in ts]
    ks = np.array([-0.20, -0.08, 0.0, 0.08, 0.20])
    got = om.dupire_local_vol_svi(slices, k_grid=ks)
    assert got is not None
    lv = np.asarray(got["local_vol"])[1]                  # orta dilim (merkezî fark)
    assert np.all(np.isfinite(lv))

    def _call(k: float, t: float) -> float:
        w = om.svi_total_variance(np.array([k]), {
            key: (val * t / base_t if key in ("a", "b") else val)
            for key, val in params.items()})
        sig = math.sqrt(max(float(w[0]), 1e-12) / t)
        return float(qm.bs_price(fwd, fwd * math.exp(k), t, 0.0, sig, "c"))

    dk_h, dt_h = 1e-4, 1e-5
    for i, k in enumerate(ks):
        strike = fwd * math.exp(float(k))
        dcdt = (_call(k, base_t + dt_h) - _call(k, base_t - dt_h)) / (2 * dt_h)
        kp = math.log((strike + dk_h) / fwd)
        km = math.log((strike - dk_h) / fwd)
        d2c = (_call(kp, base_t) - 2 * _call(k, base_t) + _call(km, base_t)) / dk_h ** 2
        ref = math.sqrt(dcdt / (0.5 * strike * strike * d2c))
        assert lv[i] == pytest.approx(ref, rel=2e-3), (k, lv[i], ref)


def test_mst_MUKEMMEL_KORELE_ciftte_agaci_koparmaz():
    """`scipy.sparse.csgraph` sıfırı "kenar YOK" sayar; ρ=1 olan bir çift d=0 üretir ve
    guard olmadan o kenar DÜŞER → ağaç kopar (n−2 kenar, izole düğüm). Zincir testi bu
    dalı göremez (orada hiçbir çift birebir korele değildir)."""
    corr = np.array([[1.0, 1.0, 0.3, 0.2],
                     [1.0, 1.0, 0.3, 0.2],
                     [0.3, 0.3, 1.0, 0.4],
                     [0.2, 0.2, 0.4, 1.0]])
    m = qm.mst_edges(corr)
    assert m is not None
    assert len(m["edges"]) == 3, m["edges"]              # n−1: ağaç KOPMADI
    assert min(m["degree"]) >= 1, m["degree"]            # izole düğüm yok
    assert {tuple(sorted((e["i"], e["j"]))) for e in m["edges"]} >= {(0, 1)}
