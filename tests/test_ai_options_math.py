"""nihai.ai.options_math testleri — opsiyon/volatilite yüzeyi matematiği (Tur-32).

Doğrulama stratejisi (test_ai_quant_math.py deseni): kapalı-form iddiaları BAĞIMSIZ bir
kaynağa karşı sınanır — Greek'ler SONLU FARK ile (analitik formül ile sayısal türev aynı
sayıyı vermeli), IV kök bulma GİDİŞ-DÖNÜŞ ile (bs_price → implied_vol → aynı σ), risk-nötr
yoğunluk BİLİNEN dağılımla (BS fiyatlarından çıkarılan yoğunluk lognormal olmalı). Ağ/IO yok.

py_vollib YALNIZ opsiyonel çapraz doğrulama için (kurulu değilse ilgili test atlanır).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from nihai.ai import options_math as om
from nihai.ai import quant_math as qm

try:
    from py_vollib.black_scholes.implied_volatility import implied_volatility as _vollib_iv
except ImportError:  # pragma: no cover - opsiyonel bağımlılık
    _vollib_iv = None


# ---------------------------------------------------------------------------
# 1) Örtük volatilite
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["c", "p"])
@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
@pytest.mark.parametrize("sigma", [0.08, 0.35, 0.90])
def test_implied_vol_gidis_donus(kind, strike, sigma):
    """bs_price(σ) → implied_vol → σ (parite/moneyness/vol rejimlerinde)."""
    px = qm.bs_price(100.0, strike, 0.5, 0.02, sigma, kind)
    assert px is not None
    assert om.implied_vol(px, 100.0, strike, 0.5, 0.02, kind) == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_arbitraj_bandi_disinda_none():
    """Model-bağımsız sınırın dışındaki (ve TAM ÜSTÜNDEKİ) fiyat için IV TANIMSIZDIR —
    uydurma bir σ döndürmek yerine None. Sınırdaki fiyat σ→0/σ→∞ limitidir."""
    lo, hi = om.bs_no_arb_bounds(100.0, 100.0, 0.5, 0.02, "c")
    assert om.implied_vol(hi + 1.0, 100.0, 100.0, 0.5, 0.02, "c") is None
    assert om.implied_vol(lo, 100.0, 100.0, 0.5, 0.02, "c") is None
    assert om.implied_vol(hi, 100.0, 100.0, 0.5, 0.02, "c") is None


def test_bs_no_arb_bounds_bilinen_degerler():
    lo, hi = om.bs_no_arb_bounds(100.0, 90.0, 1.0, 0.0, "c")
    assert (lo, hi) == pytest.approx((10.0, 100.0))
    lo_p, hi_p = om.bs_no_arb_bounds(100.0, 90.0, 1.0, 0.0, "p")
    assert (lo_p, hi_p) == pytest.approx((0.0, 90.0))


def test_implied_vol_savunmaci_girdiler():
    assert om.implied_vol("x", 100, 100, 0.5) is None
    assert om.implied_vol(5.0, -100, 100, 0.5) is None
    assert om.implied_vol(5.0, 100, 100, 0.0) is None
    assert om.implied_vol(5.0, 100, 100, 0.5, 0.0, "zort") is None
    assert om.bs_no_arb_bounds(100, 100, 0.5, 0.0, "zort") is None


@pytest.mark.skipif(_vollib_iv is None, reason="py_vollib kurulu değil")
def test_implied_vol_vollib_ile_capraz_dogrulama():
    px = qm.bs_price(100.0, 105.0, 0.75, 0.01, 0.27, "c")
    assert om.implied_vol(px, 100.0, 105.0, 0.75, 0.01, "c") == pytest.approx(
        _vollib_iv(px, 100.0, 105.0, 0.75, 0.01, "c"), abs=1e-5)


def test_implied_vol_chain_cozulemeyen_nan_kalir():
    strikes = [90.0, 100.0, 110.0]
    prices = [qm.bs_price(100.0, k, 0.5, 0.0, 0.25, "c") for k in strikes]
    prices[1] = 1e9                       # arbitraj bandı dışı → NaN
    out = om.implied_vol_chain(prices, strikes, 100.0, 0.5, 0.0, "c")
    assert out.shape == (3,)
    assert math.isnan(out[1]) and not math.isnan(out[0]) and not math.isnan(out[2])
    assert om.implied_vol_chain([1.0], [1.0, 2.0], 100, 0.5) is None


# ---------------------------------------------------------------------------
# 2) Yüksek mertebeden Greek'ler — sonlu farkla doğrulama
# ---------------------------------------------------------------------------
_S, _K, _T, _R, _SIG = 100.0, 105.0, 0.6, 0.03, 0.28


def _g(s=_S, t=_T, sig=_SIG, kind="c"):
    return qm.bs_greeks(s, _K, t, _R, sig, kind)


@pytest.mark.parametrize("kind", ["c", "p"])
def test_extended_greeks_sonlu_fark_ile_ayni(kind):
    """Analitik formüller `quant_math.bs_greeks`'in SAYISAL türevleriyle eşleşmeli.
    Zamana bağlı olanlar (charm/veta/color) ∂/∂t (TAKVİM zamanı) yönündedir — theta ile
    aynı eksen. Bu test o işaret sözleşmesinin kilididir."""
    ext = om.bs_greeks_extended(_S, _K, _T, _R, _SIG, kind)
    h, hs = 1e-5, 1e-3
    beklenen = {
        "vanna": (_g(sig=_SIG + h, kind=kind)["delta"] - _g(sig=_SIG - h, kind=kind)["delta"]) / (2 * h),
        "volga": (_g(sig=_SIG + h, kind=kind)["vega"] - _g(sig=_SIG - h, kind=kind)["vega"]) / (2 * h),
        "charm": -(_g(t=_T + h, kind=kind)["delta"] - _g(t=_T - h, kind=kind)["delta"]) / (2 * h),
        "veta": -(_g(t=_T + h, kind=kind)["vega"] - _g(t=_T - h, kind=kind)["vega"]) / (2 * h),
        "speed": (_g(s=_S + hs, kind=kind)["gamma"] - _g(s=_S - hs, kind=kind)["gamma"]) / (2 * hs),
        "zomma": (_g(sig=_SIG + h, kind=kind)["gamma"] - _g(sig=_SIG - h, kind=kind)["gamma"]) / (2 * h),
        "color": -(_g(t=_T + h, kind=kind)["gamma"] - _g(t=_T - h, kind=kind)["gamma"]) / (2 * h),
    }
    for ad, sayisal in beklenen.items():
        assert ext[ad] == pytest.approx(sayisal, rel=1e-4), ad


def test_dual_greeks_strike_turevi():
    """dual_delta/dual_gamma strike'a göre 1./2. türevdir (dijital fiyat ve BL yoğunluğu)."""
    ext = om.bs_greeks_extended(_S, _K, _T, _R, _SIG, "c")
    h = 1e-3
    d1 = (qm.bs_price(_S, _K + h, _T, _R, _SIG, "c") - qm.bs_price(_S, _K - h, _T, _R, _SIG, "c")) / (2 * h)
    d2 = (qm.bs_price(_S, _K + h, _T, _R, _SIG, "c") - 2 * qm.bs_price(_S, _K, _T, _R, _SIG, "c")
          + qm.bs_price(_S, _K - h, _T, _R, _SIG, "c")) / h ** 2
    assert ext["dual_delta"] == pytest.approx(d1, rel=1e-6)
    assert ext["dual_gamma"] == pytest.approx(d2, rel=1e-4)


def test_extended_greeks_temettusuz_charm_call_put_esit():
    """q=0'da Δ_put = Δ_call − 1 kimliği charm'ı call/put için EŞİT yapar (sadeleştirme
    değil, kimlik — docstring iddiasının kilidi)."""
    c = om.bs_greeks_extended(_S, _K, _T, _R, _SIG, "c")
    p = om.bs_greeks_extended(_S, _K, _T, _R, _SIG, "p")
    assert c["charm"] == pytest.approx(p["charm"])
    assert c["color"] == pytest.approx(p["color"])
    assert c["vanna"] == pytest.approx(p["vanna"])


def test_extended_greeks_savunmaci():
    assert om.bs_greeks_extended(100, 100, -1, 0.0, 0.2) is None
    assert om.bs_greeks_extended(100, 100, 0.5, 0.0, 0.0) is None
    assert om.bs_greeks_extended(100, 100, 0.5, 0.0, 0.2, "zort") is None
    assert om.bs_greeks_extended("x", 100, 0.5, 0.0, 0.2) is None


# ---------------------------------------------------------------------------
# 3) Put-call paritesi
# ---------------------------------------------------------------------------
def test_put_call_parity_bs_fiyatlarinda_gap_sifir():
    c = qm.bs_price(100.0, 100.0, 0.5, 0.03, 0.2, "c")
    p = qm.bs_price(100.0, 100.0, 0.5, 0.03, 0.2, "p")
    out = om.put_call_parity(c, p, 100.0, 100.0, 0.5, 0.03)
    assert out["gap"] == pytest.approx(0.0, abs=1e-9)
    assert out["implied_forward"] == pytest.approx(100.0 * math.exp(0.03 * 0.5), rel=1e-9)
    assert out["implied_div_yield"] == pytest.approx(0.0, abs=1e-9)


def test_put_call_parity_temettu_ortaya_cikar():
    """Spot yerine S·e^{−qT} ile fiyatlanan zincirde parite q'yu geri verir."""
    q, s, t = 0.04, 100.0, 0.5
    s_adj = s * math.exp(-q * t)
    c = qm.bs_price(s_adj, 100.0, t, 0.03, 0.2, "c")
    p = qm.bs_price(s_adj, 100.0, t, 0.03, 0.2, "p")
    out = om.put_call_parity(c, p, s, 100.0, t, 0.03)
    assert out["implied_div_yield"] == pytest.approx(q, abs=1e-9)
    assert out["gap"] < 0            # temettü call'u ucuzlatır → parite açığı negatif


def test_put_call_parity_savunmaci():
    assert om.put_call_parity(1, 1, -1, 100, 0.5) is None
    assert om.put_call_parity("x", 1, 100, 100, 0.5) is None


# ---------------------------------------------------------------------------
# 4) SVI
# ---------------------------------------------------------------------------
_SVI_TRUE = {"a": 0.04, "b": 0.4, "rho": -0.4, "m": 0.03, "sigma": 0.18}


def test_svi_fit_parametre_geri_kazanimi():
    """Bilinen SVI'dan üretilen toplam varyans yeniden fit edilince parametreler geri gelir."""
    k = np.linspace(-0.6, 0.5, 25)
    w = om.svi_total_variance(k, _SVI_TRUE)
    fit = om.svi_fit(k, w)
    assert fit["success"] and fit["rmse"] < 1e-5
    for ad, deger in _SVI_TRUE.items():
        assert fit[ad] == pytest.approx(deger, abs=1e-3), ad


def test_svi_durrleman_arbitrajsiz_dilimde_pozitif():
    k = np.linspace(-0.6, 0.5, 41)
    g = om.svi_durrleman_g(k, _SVI_TRUE)
    assert np.all(g > 0)
    rapor = om.svi_no_arbitrage(_SVI_TRUE, (-0.6, 0.5))
    assert rapor["butterfly_free"] and rapor["wing_ok"] and rapor["positive"]


def test_svi_no_arbitrage_kotu_parametreyi_yakalar():
    """Kanat sınırını (b(1+|ρ|) ≤ 2) aşan parametre butterfly arbitrajı üretir; Durrleman
    minimumu negatiftir. Kapı GEREKLİ koşulu da yeterli koşulu da yakalamalı."""
    kotu = {"a": 0.001, "b": 1.9, "rho": -0.95, "m": 0.0, "sigma": 0.02}
    rapor = om.svi_no_arbitrage(kotu)
    assert rapor["wing_ok"] is False
    assert rapor["durrleman_min"] < 0
    assert rapor["butterfly_free"] is False


def test_svi_savunmaci_girdiler():
    assert om.svi_fit([1, 2, 3], [1, 2, 3]) is None                # <5 nokta
    assert om.svi_total_variance([0.0], {"a": 1}) is None          # eksik anahtar
    assert om.svi_total_variance([0.0], {**_SVI_TRUE, "rho": 1.5}) is None
    assert om.svi_durrleman_g([0.0], {**_SVI_TRUE, "sigma": -1}) is None
    assert om.svi_no_arbitrage({"a": 1}) is None
    k = np.linspace(-0.5, 0.5, 10)
    w = om.svi_total_variance(k, _SVI_TRUE)
    assert om.svi_fit(k, w, weights=[1, 2]) is None                # uyumsuz ağırlık


# ---------------------------------------------------------------------------
# 4b) SSVI — arbitrajsız yüzey (Tur-34)
# ---------------------------------------------------------------------------
_SSVI_TRUE = {"rho": -0.4, "eta": 0.9, "gamma": 0.35}
_SSVI_THETAS = (0.010, 0.020, 0.035, 0.060)
_SSVI_EXPIRIES = (0.08, 0.17, 0.33, 0.50)


def _ssvi_dilimler(n_k: int = 15) -> list[dict]:
    """Gerçek SSVI parametrelerinden dört vadelik TEMİZ (arbitrajsız) yüzey."""
    k = np.linspace(-0.35, 0.35, n_k)
    return [{"log_moneyness": k,
             "total_variance": om.ssvi_total_variance(k, th, _SSVI_TRUE), "t": t}
            for th, t in zip(_SSVI_THETAS, _SSVI_EXPIRIES, strict=True)]


def _dik_kanat_dilimler() -> list[dict]:
    """Raw SVI'nin kelebek arbitrajı ÜRETECEĞİ yüzey: kanat eğimi b·(1+|ρ|) > 2
    (Lee moment sınırının ihlali)."""
    k = np.linspace(-0.30, 0.30, 13)
    out = []
    for i in range(3):
        b = 3.0 * (i + 1) / 2.0
        w = 0.004 + b * np.sqrt(k * k + 0.02 ** 2)
        out.append({"log_moneyness": k, "total_variance": w, "t": 0.1 * (i + 1)})
    return out


def test_ssvi_parametre_geri_kazanimi():
    """Kendi ürettiği yüzeyi makine hassasiyetinde geri bulmalı (ρ, η, γ ve tüm θ_t)."""
    fit = om.ssvi_fit(_ssvi_dilimler())
    assert fit is not None and fit["success"]
    assert fit["rho"] == pytest.approx(_SSVI_TRUE["rho"], abs=1e-3)
    assert fit["eta"] == pytest.approx(_SSVI_TRUE["eta"], abs=1e-3)
    assert fit["gamma"] == pytest.approx(_SSVI_TRUE["gamma"], abs=1e-3)
    assert fit["thetas"] == pytest.approx(list(_SSVI_THETAS), abs=1e-4)
    assert fit["rmse"] < 1e-5
    assert fit["n_slices"] == 4 and fit["n_points"] == 60


def test_ssvi_atm_toplam_varyans_theta_ya_esit():
    """k=0'da w(0,θ) = θ — SSVI'nin tanımı gereği (θ ATM toplam varyansıdır)."""
    w = om.ssvi_total_variance([0.0], 0.04, _SSVI_TRUE)
    assert float(w[0]) == pytest.approx(0.04, abs=1e-12)


def test_ssvi_arbitrajsizlik_kosullari():
    """η·(1+|ρ|) ≤ 2 ve γ ≤ ½ sağlanınca arb_free; ihlal edilince ilgili bayrak düşer."""
    ok = om.ssvi_no_arbitrage(_SSVI_TRUE, _SSVI_THETAS)
    assert ok["arb_free"] is True and ok["calendar"] is True
    kotu = om.ssvi_no_arbitrage({"rho": -0.5, "eta": 1.8, "gamma": 0.35}, _SSVI_THETAS)
    assert kotu["butterfly_wing"] is False and kotu["arb_free"] is False
    gam = om.ssvi_no_arbitrage({**_SSVI_TRUE, "gamma": 0.9}, _SSVI_THETAS)
    assert gam["gamma_ok"] is False and gam["arb_free"] is False
    azalan = om.ssvi_no_arbitrage(_SSVI_TRUE, (0.06, 0.02))     # takvim ihlali
    assert azalan["calendar"] is False and azalan["arb_free"] is False


def test_ssvi_kisitli_fit_arbitrajsiz_bolgede_kaliyor():
    """DÜŞMANCA veride bile `enforce_arb_free=True` arbitrajsız bölgeden ÇIKMAZ —
    kısıt bir umut değil, optimizasyon sınırıdır."""
    fit = om.ssvi_fit(_dik_kanat_dilimler(), enforce_arb_free=True)
    assert fit is not None and fit["enforced"] is True
    assert fit["no_arb"]["arb_free"] is True
    assert fit["eta"] * (1.0 + abs(fit["rho"])) <= 2.0 + 1e-9
    serbest = om.ssvi_fit(_dik_kanat_dilimler(), enforce_arb_free=False)
    assert serbest["enforced"] is False
    # Kısıtsız fit DAHA İYİ uyar ama arbitrajsızlığı İDDİA EDEMEZ — takas budur.
    assert serbest["rmse"] < fit["rmse"]
    assert serbest["no_arb"]["arb_free"] is False


def test_ssvi_diagnose_temiz_yuzeyde_arbitraj_yok():
    tani = om.ssvi_diagnose(_ssvi_dilimler())
    assert tani is not None
    assert tani["verdict"] == "ARBITRAJ_YOK" and tani["svi_violations"] == 0


def test_ssvi_diagnose_veride_arbitraji_ayirt_ediyor():
    """Tur-32 açık kalem #5: raw SVI arbitraj buluyor VE arbitrajsız yüzey veriyi
    açıklayamıyorsa ihlal FİT'te değil VERİDEDİR."""
    tani = om.ssvi_diagnose(_dik_kanat_dilimler())
    assert tani["verdict"] == "VERIDE_ARBITRAJ"
    assert tani["svi_violations"] == 3
    assert tani["rmse_ratio"] > om._SSVI_FIT_RATIO
    assert tani["svi_rmse"] < tani["ssvi_rmse"]      # raw SVI iyi uyuyor — ama arbitrajlı


def test_ssvi_diagnose_uyum_artefaktini_ayirt_ediyor():
    """Karşı durum: GÜRÜLTÜLÜ ama özünde arbitrajsız bir yüzeyde dilim başına 5 serbest
    parametreli raw SVI gürültüyü kovalayıp bir dilimde kelebek ihlali üretir; arbitrajsız
    SSVI aynı veriyi BENZER hatayla (oran 1,16) açıklar → ihlal VERİDE değil FİT'tedir."""
    rng = np.random.default_rng(0)
    k = np.linspace(-0.35, 0.35, 15)
    gurultulu = [
        {"log_moneyness": k, "t": t,
         "total_variance": np.maximum(
             om.ssvi_total_variance(k, th, _SSVI_TRUE) + rng.normal(0, 0.006, k.size), 1e-4)}
        for th, t in zip(_SSVI_THETAS, _SSVI_EXPIRIES, strict=True)]
    tani = om.ssvi_diagnose(gurultulu)
    assert tani is not None
    assert tani["verdict"] == "UYUM_ARTEFAKTI"
    assert tani["svi_violations"] >= 1
    assert tani["rmse_ratio"] <= om._SSVI_FIT_RATIO


def test_ssvi_diagnose_takvim_ihlalinde_belirsiz():
    """θ vadeye göre AZALAN bir yüzeyde kısıtlı fit çözüm bulamaz → dürüstçe 'BELIRSIZ'.
    Yanlış bir verdikt vermektense ölçemediğini SÖYLEMEK (Tur-31 dersi)."""
    dilimler = _ssvi_dilimler()
    ters = om.svi_total_variance(dilimler[0]["log_moneyness"],
                                 {"a": 0.004, "b": 2.4, "rho": 0.0, "m": 0.0, "sigma": 0.02})
    tani = om.ssvi_diagnose([{**dilimler[0], "total_variance": ters}] + dilimler[1:])
    assert tani is not None and tani["verdict"] == "BELIRSIZ"


def test_ssvi_savunmaci_girdiler():
    assert om.ssvi_fit([]) is None
    assert om.ssvi_fit("dilim degil") is None
    assert om.ssvi_fit([{"log_moneyness": [0.0], "total_variance": [0.1]}]) is None  # <3
    assert om.ssvi_fit([{"log_moneyness": [0, 1, 2],
                         "total_variance": [0.1, -0.1, 0.2]}]) is None   # w ≤ 0
    assert om.ssvi_total_variance([0.0], -1.0, _SSVI_TRUE) is None
    assert om.ssvi_total_variance([0.0], 0.04, {"rho": 1.5, "eta": 1, "gamma": 0.3}) is None
    assert om.ssvi_no_arbitrage({"rho": 2.0}, [0.1]) is None
    assert om.ssvi_no_arbitrage(_SSVI_TRUE, [-0.1]) is None
    assert om.ssvi_diagnose([]) is None


# ---------------------------------------------------------------------------
# 5) SABR
# ---------------------------------------------------------------------------
def test_sabr_atm_kapali_form_ve_sureklilik():
    """f=k dalı ayrı ele alınır (0/0 belirsizliği). ATM değeri α/f^{1−β}·(1+corr·t)'dir ve
    f→k limitinde genel dal ile SÜREKLİ olmalıdır (klasik SABR NaN hatasının kilidi)."""
    f, t, alpha, beta = 100.0, 0.75, 0.30, 0.5
    atm = om.sabr_lognormal_vol(f, f, t, alpha, beta, -0.3, 0.4)
    komsu = om.sabr_lognormal_vol(f, f * (1 + 1e-8), t, alpha, beta, -0.3, 0.4)
    assert atm == pytest.approx(komsu, rel=1e-6)
    assert atm == pytest.approx(alpha / f ** (1 - beta), rel=0.05)


def test_sabr_calibrate_parametre_geri_kazanimi():
    f, t = 100.0, 0.75
    strikes = np.linspace(70, 140, 15)
    vols = np.array([om.sabr_lognormal_vol(f, float(k), t, 0.30, 0.5, -0.30, 0.40)
                     for k in strikes])
    cal = om.sabr_calibrate(f, strikes, vols, t, beta=0.5)
    assert cal["rmse"] < 1e-4
    assert cal["alpha"] == pytest.approx(0.30, abs=5e-3)
    assert cal["rho"] == pytest.approx(-0.30, abs=5e-2)
    assert cal["nu"] == pytest.approx(0.40, abs=5e-2)


def test_sabr_savunmaci():
    assert om.sabr_lognormal_vol(100, 100, 1, 0.3, 0.5, 1.5, 0.4) is None   # |ρ|≥1
    assert om.sabr_lognormal_vol(100, 100, 1, -0.1, 0.5, 0.0, 0.4) is None  # α≤0
    assert om.sabr_lognormal_vol(100, 100, 1, 0.3, 1.5, 0.0, 0.4) is None   # β∉[0,1]
    assert om.sabr_calibrate(100, [90, 100], [0.2, 0.2], 1.0) is None       # <3 nokta
    assert om.sabr_calibrate(-1, [90, 100, 110], [0.2] * 3, 1.0) is None


# ---------------------------------------------------------------------------
# 6) Gülümseme ölçüleri
# ---------------------------------------------------------------------------
def test_smile_metrics_hisse_skew_isareti():
    """Hisse tipik gülümsemesi (sola yatık): rr25 NEGATİF, bf25 POZİTİF, skew<0."""
    k = np.linspace(-0.4, 0.3, 21)
    v = 0.25 - 0.25 * k + 0.4 * k ** 2
    m = om.smile_metrics(k, v, 100.0, 0.5)
    assert m["atm_vol"] == pytest.approx(0.25, abs=2e-3)
    assert m["skew"] == pytest.approx(-0.25, abs=1e-2)
    assert m["curvature"] == pytest.approx(0.8, abs=5e-2)
    assert m["rr25"] < 0 and m["bf25"] > 0


def test_smile_metrics_duz_gulumsemede_rr_bf_sifir():
    k = np.linspace(-0.4, 0.4, 21)
    m = om.smile_metrics(k, np.full(21, 0.30), 100.0, 0.5)
    assert m["rr25"] == pytest.approx(0.0, abs=1e-9)
    assert m["bf25"] == pytest.approx(0.0, abs=1e-9)
    assert m["skew"] == pytest.approx(0.0, abs=1e-9)


def test_smile_metrics_atm_disinda_none():
    """k=0 veri aralığının dışındaysa ekstrapolasyon YAPILMAZ (np.interp uçları sabitler
    ve bu sahte bir düz kanat üretirdi) → None."""
    assert om.smile_metrics([0.1, 0.2, 0.3], [0.2, 0.21, 0.22], 100.0, 0.5) is None
    assert om.smile_metrics([1, 2, 3], [0.2, 0.2, 0.2], -1.0, 0.5) is None


# ---------------------------------------------------------------------------
# 7) Risk-nötr yoğunluk
# ---------------------------------------------------------------------------
def test_risk_neutral_density_lognormali_geri_verir():
    """BS fiyatlarından Breeden-Litzenberger ile çıkarılan yoğunluk, BS'in KENDİ lognormal
    risk-nötr yoğunluğu olmalı — bağımsız kaynakla doğrulama."""
    s, r, t, sig = 100.0, 0.02, 0.5, 0.25
    strikes = np.linspace(40, 180, 281)
    calls = np.array([qm.bs_price(s, float(k), t, r, sig, "c") for k in strikes])
    out = om.risk_neutral_density(strikes, calls, t, r)
    beklenen = stats.lognorm.pdf(out["strikes"], s=sig * math.sqrt(t),
                                 scale=s * math.exp((r - 0.5 * sig ** 2) * t))
    assert np.max(np.abs(out["density"] - beklenen)) < 1e-5
    assert out["mass"] == pytest.approx(1.0, abs=1e-3)
    assert out["mean"] == pytest.approx(s * math.exp(r * t), rel=1e-3)   # ileri fiyat
    assert out["n_negative"] == 0
    assert out["skew"] > 0                                              # lognormal sağa çarpık


def test_risk_neutral_density_arbitraji_bastirmaz():
    """Dışbükeyliği bozulmuş (butterfly arbitrajlı) fiyatlarda negatif yoğunluk RAPORLANIR,
    sıfırlanmaz — veriyi güzelleştirmek yalan söylemek olurdu."""
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    calls = np.array([12.0, 8.0, 6.5, 2.0, 1.0])     # 100'de dışbükeylik ihlali
    out = om.risk_neutral_density(strikes, calls, 0.5, 0.0)
    assert out["n_negative"] >= 1


def test_risk_neutral_density_savunmaci():
    assert om.risk_neutral_density([1, 2, 3], [1, 2, 3], 0.5) is None          # <4 strike
    assert om.risk_neutral_density([1, 1, 2, 3], [1, 2, 3, 4], 0.5) is None    # yinelenen
    assert om.risk_neutral_density([1, 2, 3, 4], [1, 2, 3, 4], -1) is None


# ---------------------------------------------------------------------------
# 8) Vade yapısı
# ---------------------------------------------------------------------------
def test_vol_term_structure_contango_ve_ileri_vol():
    out = om.vol_term_structure([0.25, 1.0], [0.20, 0.25])
    assert out["contango"] is True and out["calendar_free"] is True
    # w1=0.01, w2=0.0625 → ileri varyans = 0.0525/0.75 = 0.07 → ileri vol ≈ 0.2646
    assert out["forward_vols"][0] == pytest.approx(math.sqrt(0.07), rel=1e-9)


def test_vol_term_structure_calendar_arbitraji_yakalar():
    """Toplam varyans vade boyunca AZALAMAZ; azalıyorsa ileri varyans negatiftir."""
    out = om.vol_term_structure([0.1, 0.5], [0.30, 0.10])
    assert out["calendar_free"] is False
    assert out["n_violations"] == 1
    assert math.isnan(out["forward_vols"][0])


def test_vol_term_structure_savunmaci():
    assert om.vol_term_structure([1.0], [0.2]) is None
    assert om.vol_term_structure([1.0, 1.0], [0.2, 0.3]) is None    # yinelenen vade
    assert om.vol_term_structure([1.0, 2.0], [0.2, -0.3]) is None
