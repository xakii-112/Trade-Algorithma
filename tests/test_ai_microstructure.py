"""nihai.ai.microstructure testleri — gerçekleşen oynaklık + mikroyapı (Tur-32).

Doğrulama stratejisi: her tahminci BİLİNEN bir üretici süreçten sentetik veri üzerinde
sınanır ve GERÇEK parametreyi geri vermesi beklenir (HAR katsayıları, Roll makası, Kyle λ,
TSRV'nin gürültüden arınmış varyansı). Sabit `default_rng` seed'leriyle deterministik;
ağ/IO yok.
"""
from __future__ import annotations

import numpy as np
import pytest

from nihai.ai import microstructure as ms


@pytest.fixture()
def har_serisi() -> np.ndarray:
    """Bilinen HAR süreci: RV_{t+1} = 1e-5 + 0.35·d + 0.35·w + 0.25·m + küçük gürültü."""
    rng = np.random.default_rng(7)
    n = 1200
    rv = np.full(n, 1e-4)
    for t in range(22, n):
        d, w, m = rv[t - 1], rv[t - 5:t].mean(), rv[t - 22:t].mean()
        rv[t] = max(1e-8, 1e-5 + 0.35 * d + 0.35 * w + 0.25 * m + rng.normal(0, 2e-6))
    return rv


# ---------------------------------------------------------------------------
# 1) Gerçekleşen varyans / sıçrama
# ---------------------------------------------------------------------------
def test_realized_variance_bilinen_deger():
    assert ms.realized_variance([0.1, -0.2, 0.3]) == pytest.approx(0.14)
    assert ms.realized_variance([]) is None
    assert ms.realized_variance([1.0, np.nan]) is None


def test_bipower_variation_saf_difuzyonda_rv_ye_yakin():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 4000)
    rv, bv = ms.realized_variance(r), ms.bipower_variation(r)
    assert bv == pytest.approx(rv, rel=0.05)      # sıçrama yoksa BV ≈ RV
    assert ms.bipower_variation([0.1]) is None


def test_jump_split_sicramayi_ayirir():
    """Saf difüzyonda jump_share ≈ 0; tek bir dev getiri eklenince belirgin biçimde artar."""
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.01, 500)
    assert ms.jump_split(r)["jump_share"] < 0.05
    r_jump = r.copy()
    r_jump[250] = 0.30
    out = ms.jump_split(r_jump)
    assert out["jump_share"] > 0.4
    assert out["continuous"] + out["jump"] == pytest.approx(out["rv"])
    assert ms.jump_split([0.1]) is None


# ---------------------------------------------------------------------------
# 2) Gürültü / TSRV / imza
# ---------------------------------------------------------------------------
def test_two_scale_rv_gurultuyu_temizler():
    """Etkin fiyat + iid gürültü: RV_all gerçek varyansı KAT KAT şişirir, TSRV geri getirir."""
    rng = np.random.default_rng(3)
    n, sig = 4000, 4e-4
    etkin = np.cumsum(rng.normal(0, sig, n))
    gozlenen = np.exp(etkin + rng.normal(0, 8e-4, n))
    gercek = n * sig ** 2
    out = ms.two_scale_rv(gozlenen, k=10)
    assert out["rv_all"] > 3 * gercek                      # ham RV ağır yanlı
    assert out["tsrv"] == pytest.approx(gercek, rel=0.15)  # TSRV yanlılığı giderir
    assert out["noise_var"] > 0


def test_volatility_signature_gurultuyu_tespit_eder():
    """Gürültülü seride bar-başına RV örnekleme adımıyla DÜŞER (negatif eğim); temiz seride
    imza düzdür."""
    rng = np.random.default_rng(3)
    n, sig = 4000, 4e-4
    etkin = np.cumsum(rng.normal(0, sig, n))
    gurultulu = ms.volatility_signature(np.exp(etkin + rng.normal(0, 8e-4, n)), 15)
    temiz = ms.volatility_signature(np.exp(etkin), 15)
    assert gurultulu["noise_suspected"] is True and gurultulu["noise_slope"] < 0
    assert temiz["noise_suspected"] is False
    assert temiz["rv_per_bar"][-1] == pytest.approx(sig ** 2, rel=0.3)


def test_tsrv_ve_imza_savunmaci():
    assert ms.two_scale_rv([1, 2, 3], k=5) is None
    assert ms.two_scale_rv([1.0] * 100, k=1) is None
    assert ms.two_scale_rv([-1.0] * 100, k=5) is None       # fiyat ≤ 0
    assert ms.volatility_signature([1, 2, 3], 20) is None


# ---------------------------------------------------------------------------
# 3) HAR
# ---------------------------------------------------------------------------
def test_har_rv_fit_katsayilari_geri_kazanir(har_serisi):
    fit = ms.har_rv_fit(har_serisi)
    assert fit["n"] > 1000 and fit["r2"] > 0.9
    assert fit["coef"] == pytest.approx([0.35, 0.35, 0.25], abs=0.12)
    assert fit["persistence"] == pytest.approx(0.95, abs=0.1)


def test_har_rv_forecast_makul_bant(har_serisi):
    """Öngörü serinin son değerlerinin mertebesinde olmalı (h=1 ve h=5 için)."""
    fit = ms.har_rv_fit(har_serisi)
    son = float(har_serisi[-1])
    for h in (1, 5):
        tahmin = ms.har_rv_forecast(fit, har_serisi, horizon=h)
        assert 0.3 * son < tahmin < 3.0 * son


def test_har_log_target_ham_olcekte_doner(har_serisi):
    """log_target modelinde çağıran HAM seri verir, HAM ölçekte tahmin alır (log alması
    beklenmez) ve yarı-varyans düzeltmesi uygulanır."""
    fit = ms.har_rv_fit(har_serisi, log_target=True)
    assert fit["log_target"] is True
    tahmin = ms.har_rv_forecast(fit, har_serisi, horizon=1)
    assert 0.3 * float(har_serisi[-1]) < tahmin < 3.0 * float(har_serisi[-1])


def test_har_cj_sicrama_katsayisi_alani(har_serisi):
    jump = np.zeros_like(har_serisi)
    fit = ms.har_cj_fit(har_serisi, jump)
    assert len(fit["coef"]) == 4 and "jump_beta" in fit
    assert fit["coef"][:3] == pytest.approx([0.35, 0.35, 0.25], abs=0.12)


def test_har_savunmaci():
    assert ms.har_rv_fit([1, 2, 3]) is None
    assert ms.har_rv_fit([-1.0] * 40) is None
    assert ms.har_rv_fit([0.0] * 40, log_target=True) is None
    assert ms.har_rv_forecast({"coef": [1, 2]}, [1.0] * 30) is None
    assert ms.har_rv_forecast({}, [1.0] * 30) is None
    assert ms.har_cj_fit([1.0] * 40, [1.0] * 10) is None


# ---------------------------------------------------------------------------
# 4) Likidite / makas / etki
# ---------------------------------------------------------------------------
def test_amihud_hacimsiz_barlari_raporlayarak_duser():
    out = ms.amihud_illiquidity([0.01, -0.02, 0.0], [1e6, 2e6, 0.0])
    assert out["n_used"] == 2 and out["n_input"] == 3
    assert out["illiq"] == pytest.approx(0.5 * (0.01 + 0.01))
    assert ms.amihud_illiquidity([0.01], [0.0]) is None
    assert ms.amihud_illiquidity([0.01, 0.02], [1e6]) is None


def test_roll_spread_bid_ask_sicramasini_geri_verir():
    rng = np.random.default_rng(9)
    orta = 100 + np.cumsum(rng.normal(0, 0.01, 3000))
    makas = 0.05
    gozlenen = orta + np.where(rng.random(3000) < 0.5, makas / 2, -makas / 2)
    out = ms.roll_spread(gozlenen)
    assert out["positive_autocov"] is False
    assert out["spread"] == pytest.approx(makas, rel=0.10)


def test_roll_spread_pozitif_otokovaryansta_olcum_yoklugunu_soyler():
    """Trend baskınsa tahminci TANIMSIZDIR: uydurma 0.0 yerine None + bayrak (Tur-31'in
    'ölçüm yokluğu ödüllendirilmez' dersinin bu modüldeki karşılığı)."""
    out = ms.roll_spread(np.arange(50.0) + 100.0)
    assert out["spread"] is None and out["positive_autocov"] is True
    assert ms.roll_spread([1.0, 2.0]) is None


def test_corwin_schultz_makul_ve_negatifleri_raporlar():
    rng = np.random.default_rng(2)
    orta = 100 + np.cumsum(rng.normal(0, 0.01, 500))
    out = ms.corwin_schultz_spread(orta + 0.2, orta - 0.2)
    assert out["spread"] >= 0 and out["n"] == 499
    assert "spread_raw" in out and "n_negative" in out
    assert ms.corwin_schultz_spread([1.0, 2.0], [3.0, 4.0]) is None   # H < L
    assert ms.corwin_schultz_spread([1.0], [1.0]) is None


# --- BULGU-33-4/5 (Tur-33) — dejenere rejimde ölçüm YOKLUĞU söylenmeli --------------
def test_corwin_schultz_alan_disinda_makas_uydurmaz():
    """Tur-32 açık kalem #6 kapanışı. Tahminci GÜNLÜK bar içindir; dakikalık seride
    varsayım çöker ve negatiflerin oranı ~%50'ye koşar. Kırpma E[max(S,0)] SİMETRİK
    gürültüden bile pozitif bir sayı üretir → gerçek makas 0 iken "1,4 bps makas".
    Sözleşme `roll_spread` ile aynı olmalı: yansız tahmin ≤ 0 → `spread` None."""
    rng = np.random.default_rng(7)
    n = 400
    orta = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))       # makassız (S=0) süreç
    hi = orta * (1 + np.abs(rng.normal(0, 0.0006, n)))
    lo = orta * (1 - np.abs(rng.normal(0, 0.0006, n)))
    out = ms.corwin_schultz_spread(hi, lo)

    assert out["spread_raw"] < 0                    # yansız tahmin: ölçülebilir makas YOK
    assert out["spread_clipped"] > 0                # ama kırpılmış ortalama POZİTİF (artefakt)
    assert out["reliable"] is False
    assert out["spread"] is None, "alan dışı rejimde kırpma artefaktı makas diye raporlanıyor"
    assert 0.0 <= out["negative_share"] <= 1.0
    assert out["negative_share"] == pytest.approx(out["n_negative"] / out["n"])


def test_corwin_schultz_duz_seride_sifir_makas_demez():
    """H==L (halt / hiç hareket yok): makas SIFIR değil BİLİNMEZDİR."""
    duz = np.full(30, 100.0)
    out = ms.corwin_schultz_spread(duz, duz)
    assert out["spread"] is None and out["reliable"] is False
    assert out["spread_clipped"] == pytest.approx(0.0)


def test_corwin_schultz_gercek_makasi_hala_olcuyor():
    """Düzeltme fazla ileri gitmemeli: GERÇEK makas varken güvenilir ve pozitif kalmalı."""
    rng = np.random.default_rng(11)
    n = 600
    orta = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    s = 0.02                                                        # %2 etkin makas
    out = ms.corwin_schultz_spread(orta * (1 + s / 2), orta * (1 - s / 2))
    assert out["reliable"] is True and out["spread"] is not None
    assert out["spread"] > 0 and out["negative_share"] < 0.5


def test_roll_spread_kisa_seride_uyari_basmaz(recwarn):
    """BULGU-33-5: n=3'te `np.cov(..., ddof=1)` tek gözlemli dilimle çağrılıyor ve üç
    RuntimeWarning basıyordu (7/24 süreçte defter kirliliği, `-W error` altında çökme)."""
    assert ms.roll_spread([100.0, 101.0, 100.5]) is None
    assert [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)] == []


def test_roll_spread_dort_fiyattan_itibaren_calisir():
    assert ms.roll_spread([100.0, 101.0, 100.5, 101.5]) is not None


def test_kyle_lambda_bilinen_egimi_bulur():
    rng = np.random.default_rng(1)
    imzali = rng.normal(0, 1000, 800)
    dp = 5e-4 * imzali + rng.normal(0, 0.01, 800)
    out = ms.kyle_lambda(dp, imzali)
    assert out["lam"] == pytest.approx(5e-4, rel=0.05)
    assert out["r2"] > 0.9
    assert ms.kyle_lambda([1, 2], [1, 2]) is None


def test_bvc_ve_vpin_tek_yonlu_akisi_ayirt_eder():
    """VPIN dengeli akışta düşük, tek yönlü akışta belirgin yüksektir."""
    rng = np.random.default_rng(6)
    hacim = rng.gamma(2, 500, 900)
    dengeli = 100 + np.cumsum(rng.normal(0, 0.1, 900))
    tek_yonlu = 100 + np.cumsum(np.abs(rng.normal(0, 0.1, 900)))
    v_dengeli = ms.vpin(dengeli, hacim, 20)
    v_tek = ms.vpin(tek_yonlu, hacim, 20)
    assert 0.0 <= v_dengeli["vpin"] <= 1.0
    assert v_tek["vpin"] > 2 * v_dengeli["vpin"]
    bvc = ms.bulk_volume_classification(dengeli, hacim)
    assert np.allclose(bvc["buy"] + bvc["sell"], hacim[1:])


def test_bvc_vpin_savunmaci():
    assert ms.bulk_volume_classification([1.0] * 10, [1.0] * 10) is None   # düz seri σ=0
    assert ms.bulk_volume_classification([1, 2, 3], [1, 2]) is None
    assert ms.vpin([1, 2, 3], [1, 2, 3], 50) is None
    assert ms.vpin([1, 2, 3], [1, 2, 3], 1) is None
    assert ms.vpin([1, 2, 3], [1, 2, 3], np.inf) is None


def test_vpin_buyuk_bar_kova_sinirinda_orantili_bolunur(monkeypatch):
    # İki 50-birim kova: [50 alım] ve [40 alım, 10 satım] → 1.0, 0.6.
    bvc = {"buy": np.array([90., 0, 0, 0, 0, 0]), "sell": np.array([0., 10, 0, 0, 0, 0]), "n": 6}
    monkeypatch.setattr(ms, "bulk_volume_classification", lambda *a: bvc)
    result = ms.vpin([], [], 2)
    np.testing.assert_allclose(result["buckets"], [1., .6])
    assert result["vpin"] == pytest.approx(.8)
    assert result["bucket_volume"] * result["n_buckets"] == 100
    np.testing.assert_array_equal(result["buckets"], ms.vpin([], [], 2)["buckets"])


# ---------------------------------------------------------------------------
# PIN (EKOP) — Tur-34
# ---------------------------------------------------------------------------
_PIN_GERCEK = {"alpha": 0.4, "delta": 0.5, "mu": 60.0, "eps_b": 40.0, "eps_s": 45.0}


def _ekop_ornekle(n: int = 400, seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """EKOP üretici sürecinden örnekleme: olay var/yok → haber iyi/kötü → bilgili akış
    haberin YÖNÜNDE Poisson(mu) ekler."""
    g = _PIN_GERCEK
    rng = np.random.default_rng(seed)
    buys = np.empty(n, dtype=float)
    sells = np.empty(n, dtype=float)
    for i in range(n):
        olay = rng.random() < g["alpha"]
        kotu = olay and rng.random() < g["delta"]
        buys[i] = rng.poisson(g["eps_b"] + (g["mu"] if (olay and not kotu) else 0.0))
        sells[i] = rng.poisson(g["eps_s"] + (g["mu"] if kotu else 0.0))
    return buys, sells


def test_pin_bilinen_parametreleri_geri_buluyor():
    """EKOP'un beş parametresi de sentetik veriden geri gelmeli; PIN gerçek değere yakın."""
    b, s = _ekop_ornekle()
    out = ms.pin_ekop(b, s)
    assert out is not None and out["converged"] is True
    g = _PIN_GERCEK
    gercek_pin = g["alpha"] * g["mu"] / (g["alpha"] * g["mu"] + g["eps_b"] + g["eps_s"])
    assert out["pin"] == pytest.approx(gercek_pin, abs=0.05)
    assert out["alpha"] == pytest.approx(g["alpha"], abs=0.1)
    assert out["mu"] == pytest.approx(g["mu"], rel=0.2)
    assert out["eps_buy"] == pytest.approx(g["eps_b"], rel=0.2)
    assert out["eps_sell"] == pytest.approx(g["eps_s"], rel=0.2)
    assert 0.0 <= out["pin"] <= 1.0


def test_pin_olcek_degismez():
    """PIN = αμ/(αμ+ε_b+ε_s); tüm sayımlar c ile çarpılınca pay ve payda birlikte ölçeklenir
    → PIN DEĞİŞMEZ. `pin_from_bars`'ın vekil sayım kullanabilmesinin dayanağı budur.

    Tolerans 1e-3'tür, 0 DEĞİL: değişmezlik TAHMİNCİNİNDİR, optimizatörün değil — L-BFGS-B
    ölçeklenmiş problemde biraz farklı bir noktada durur (ölçüldü: |Δ| ≈ 3e-5)."""
    b, s = _ekop_ornekle()
    assert ms.pin_ekop(b * 3.0, s * 3.0)["pin"] == pytest.approx(ms.pin_ekop(b, s)["pin"],
                                                                abs=1e-3)


def test_pin_deterministik():
    """Başlangıç ızgarası SABİT, rastgele yeniden-başlatma YOK → bit-denk sonuç."""
    b, s = _ekop_ornekle(n=120, seed=9)
    assert ms.pin_ekop(b, s) == ms.pin_ekop(b, s)


def test_pin_bilgisiz_akista_dusuk():
    """Bilgili işlemci OLMAYAN (μ=0) veride PIN ~0 olmalı — modelin sıfır-bilgi çapası."""
    rng = np.random.default_rng(4)
    b = rng.poisson(50, 300).astype(float)
    s = rng.poisson(50, 300).astype(float)
    out = ms.pin_ekop(b, s)
    assert out is not None and out["pin"] < 0.15


def test_pin_savunmaci():
    assert ms.pin_ekop([1, 2], [1, 2]) is None                 # <20 dönem
    assert ms.pin_ekop([-1.0] * 30, [1.0] * 30) is None        # negatif sayım
    assert ms.pin_ekop([0.0] * 30, [0.0] * 30) is None         # tüm-sıfır
    assert ms.pin_ekop(list(range(30)), list(range(20))) is None   # uyumsuz uzunluk
    assert ms.pin_ekop([1.0, np.nan] * 20, [1.0] * 40) is None     # NaN


def test_pin_from_bars_vekil_bayragi_tasiyor():
    """Bar tabanlı PIN bir VEKİLDİR ve bunu `proxy=True` ile TAŞIR — gerçek tick PIN'i ile
    aynı şey sanılmasın (ölçüm koşulunun görünürlüğü kuralı)."""
    rng = np.random.default_rng(8)
    fiyat = 100 + np.cumsum(rng.normal(0, 0.15, 400))
    hacim = rng.gamma(2, 500, 400)
    out = ms.pin_from_bars(fiyat, hacim)
    assert out is not None
    assert out["proxy"] is True and out["unit"] > 0.0
    assert 0.0 <= out["pin"] <= 1.0


def test_pin_from_bars_savunmaci():
    assert ms.pin_from_bars([1, 2, 3], [1, 2, 3]) is None            # çok kısa
    assert ms.pin_from_bars([1.0] * 40, [1.0] * 40) is None          # düz seri (BVC None)
    rng = np.random.default_rng(1)
    fiyat = 100 + np.cumsum(rng.normal(0, 0.15, 60))
    assert ms.pin_from_bars(fiyat, rng.gamma(2, 500, 60), unit=0.0) is None
    assert ms.pin_from_bars(fiyat, rng.gamma(2, 500, 60), unit=float("nan")) is None
