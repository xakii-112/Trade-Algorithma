"""research.pbo testleri — CSCV/PBO null-taban, ekili-kenar, aşırı-uyum, determinizm, guard'lar.

Kaynak: Bailey, Borwein, López de Prado & Zhu (2016), "The Probability of Backtest Overfitting".
Ağ/IO yok; tüm veri seed'li sentetik. Hız için s=8/10 (C=70/252 kombinasyon); s=16 YALNIZ guard
testinde anılır (asla çalıştırılmaz).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from research.pbo import cscv_pbo, pbo_report


# ---------------------------------------------------------------------------
# NULL taban: iid Gauss gürültü → PBO ~0.5 (H0: hiçbir konfigin gerçek kenarı yok)
# ---------------------------------------------------------------------------
def test_null_pbo_yaklasik_yarim():
    """Bağımsız aynı-dağılımlı (skill'siz) konfiglerde IS-en-iyi, OOS'ta medyanın üstünde/altında
    OLMA olasılığı eşit → PBO teorik olarak ~0.5. Örneklem gürültüsüne geniş tolerans: 0.30-0.70."""
    rng = np.random.default_rng(42)
    m = rng.normal(size=(800, 40))
    res = cscv_pbo(m, s=10)
    assert 0.30 <= res["pbo"] <= 0.70
    # logits uzunluğu == kombinasyon sayısı == C(s, s/2)
    assert res["n_combinations"] == math.comb(10, 5) == 252
    assert res["logits"].shape == (252,)
    assert 0.0 <= res["prob_oos_loss"] <= 1.0
    assert res["n_configs"] == 40 and res["n_periods_used"] == 800 and res["metric"] == "sharpe"


# ---------------------------------------------------------------------------
# Ekili kenar: tek kolonda güçlü drift → IS-en-iyi hep o kolon, OOS'ta da üst sıra → PBO ~0
# ---------------------------------------------------------------------------
def test_ekili_kenar_dusuk_pbo():
    """Bir konfigürasyona güçlü pozitif drift eklenirse (gerçek kenar), her kombinasyonda IS-en-iyi
    O kolon olur ve OOS'ta da en üstte kalır → λ>0 hep → PBO ≈ 0. degradation_slope raporlanır."""
    rng = np.random.default_rng(7)
    m = rng.normal(size=(600, 20))
    m[:, 3] += 0.6   # güçlü, kalıcı pozitif drift = ayırt edilebilir kenar
    res = cscv_pbo(m, s=10)
    assert res["pbo"] <= 0.15
    # performans-bozunumu alanları her zaman raporlanır (sonlu float)
    assert "degradation_slope" in res and math.isfinite(res["degradation_slope"])
    assert "degradation_intercept" in res and math.isfinite(res["degradation_intercept"])
    # gerçek kenar → IS-en-iyinin OOS'ta zarar etme olasılığı da düşük
    assert res["prob_oos_loss"] <= 0.15


# ---------------------------------------------------------------------------
# Tam-overfit senaryosu: kısa T + saf gürültü kolonları → PBO yükselebilir (≥0.4)
# ---------------------------------------------------------------------------
def test_tam_overfit_yuksek_pbo():
    """Kısa T'de bağımsız saf-gürültü konfigleri arasından IS-en-iyiyi seçmek, OOS'ta çoğunlukla
    çöker (şans sıralaması ters döner) → PBO yüksek çıkabilir. Seed sabit, tolerans geniş: ≥0.4.
    (Saf gürültünün PBO'su ~0.5 etrafında rastgele bir çekiliştir; seed=2 bu bölgede 0.60 verir.)"""
    rng = np.random.default_rng(2)
    m = rng.normal(size=(120, 20))
    res = cscv_pbo(m, s=8)
    assert res["pbo"] >= 0.4
    assert res["n_combinations"] == math.comb(8, 4) == 70


# ---------------------------------------------------------------------------
# Determinizm: CSCV çekirdeği deterministik — seed'siz iki çağrı BİREBİR aynı
# ---------------------------------------------------------------------------
def test_determinizm_seedsiz_ve_seed_etkisiz():
    """CSCV kombinasyonları deterministik sıralıdır; seed verilmese de aynı girdi → aynı çıktı.
    Ayrıca seed YALNIZ stokastik metric-callable içindir: yerleşik sharpe'ta seed sonucu
    DEĞİŞTİRMEZ (global RNG tohumlansa bile logits birebir aynı)."""
    rng = np.random.default_rng(1)
    m = rng.normal(size=(200, 10))
    a = cscv_pbo(m, s=8)
    b = cscv_pbo(m, s=8)
    assert np.array_equal(a["logits"], b["logits"])
    assert a["pbo"] == b["pbo"]
    # seed yerleşik metrik için etkisiz (deterministik çekirdek)
    c = cscv_pbo(m, s=8, seed=123)
    assert np.array_equal(a["logits"], c["logits"])


# ---------------------------------------------------------------------------
# "mean" ve callable metrik yolları
# ---------------------------------------------------------------------------
def test_mean_ve_callable_metrik():
    """'mean' metriği çalışır; aynı işlevi yapan callable BİREBİR aynı logits üretir (dispatch
    doğru). callable metric etiketi 'callable' olarak raporlanır."""
    rng = np.random.default_rng(11)
    m = rng.normal(size=(300, 12))
    r_str = cscv_pbo(m, s=8, metric="mean")
    r_cb = cscv_pbo(m, s=8, metric=lambda col: float(col.mean()))
    assert np.allclose(r_str["logits"], r_cb["logits"])
    assert r_str["metric"] == "mean" and r_cb["metric"] == "callable"
    assert 0.0 <= r_str["pbo"] <= 1.0


# ---------------------------------------------------------------------------
# Guard'lar — hepsi Türkçe mesajlı ValueError
# ---------------------------------------------------------------------------
def test_guard_s_tek():
    with pytest.raises(ValueError, match="ÇİFT"):
        cscv_pbo(np.zeros((100, 10)), s=7)


def test_guard_s_kucuk():
    with pytest.raises(ValueError, match="s ≥ 4"):
        cscv_pbo(np.zeros((100, 10)), s=2)


def test_guard_T_kucuk():
    with pytest.raises(ValueError, match="T ≥ 2"):
        cscv_pbo(np.zeros((10, 10)), s=8)   # T=10 < 2·8=16


def test_guard_N_kucuk():
    with pytest.raises(ValueError, match="N ≥ 5"):
        cscv_pbo(np.zeros((100, 4)), s=8)


def test_guard_nan():
    m = np.zeros((100, 10))
    m[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        cscv_pbo(m, s=8)


def test_guard_kombinasyon_patlamasi():
    """s=16 sınırda GEÇER (C(16,8)=12870 ≤ 12870); s=18 patlamayı önlemek için ValueError."""
    assert math.comb(16, 8) == 12870   # sınır değeri belgele (çalıştırma yok)
    with pytest.raises(ValueError, match="12870"):
        cscv_pbo(np.zeros((400, 10)), s=18)


# ---------------------------------------------------------------------------
# Rapor metni — anahtar alanları içerir, düz metin döner
# ---------------------------------------------------------------------------
def test_pbo_report_metni():
    rng = np.random.default_rng(3)
    res = cscv_pbo(rng.normal(size=(240, 15)), s=8)
    txt = pbo_report(res)
    assert isinstance(txt, str)
    assert "PBO=" in txt and "bozunum" in txt and "DSR" in txt
