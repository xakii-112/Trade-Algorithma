"""nihai.ai.online testleri — çevrimiçi öğrenme / uyarlama (Tur-32).

Doğrulama stratejisi: her mekanizma İKİ senaryoda sınanır — (a) mekanizmanın YAKALAMASI
gereken durum (sürüklenme var / bir uzman gerçekten iyi / özellikler gerçekten bilgili) ve
(b) yakalamaMASI gereken durum (durgun seri / hepsi gürültü). İkincisi bu projede daha
önemlidir: uyarlama mekanizmaları yok olan kenarı VAR göstermemelidir (ACI DERS #3/#9).

NEDENSELLİK: birleştiricilerin t anındaki tahmini yalnız t'den önceki sonuçlara bakmalıdır;
`test_running_base_rate_nedensel` ve `test_hedge_ilk_adim_esit_agirlik` bunu kilitler.
"""
from __future__ import annotations

import numpy as np
import pytest

from nihai.ai import online as ol


# ---------------------------------------------------------------------------
# 1) ADWIN
# ---------------------------------------------------------------------------
def test_adwin_durgun_seride_sahte_alarm_vermez():
    rng = np.random.default_rng(11)
    out = ol.adwin(rng.normal(0, 1, 1500))
    assert out["drift_detected"] is False
    assert out["n_drifts"] == 0
    assert out["final_window"] == 1500          # hiçbir şey atılmadı


def test_adwin_ani_kaymayi_yakalar_ve_pencereyi_kucultur():
    """700. gözlemde ortalama 0→2.5 kayıyor: değişim tespit edilmeli, pencere yeni rejime
    daralmalı ve pencere ortalaması YENİ ortalamaya yakın olmalı."""
    rng = np.random.default_rng(11)
    seri = np.concatenate([rng.normal(0, 1, 700), rng.normal(2.5, 1, 700)])
    out = ol.adwin(seri)
    assert out["drift_detected"] is True
    assert 690 <= out["change_points"][0] <= 760
    assert out["final_window"] <= 720
    assert out["final_mean"] == pytest.approx(2.5, abs=0.3)


def test_adwin_savunmaci():
    assert ol.adwin([1.0]) is None
    assert ol.adwin([1.0, 2.0], delta=0.0) is None
    assert ol.adwin([1.0, 2.0], delta=1.5) is None
    assert ol.adwin([1.0, np.nan, 2.0]) is None
    assert ol.adwin([1.0, 2.0], max_window=3) is None


# ---------------------------------------------------------------------------
# 2) Koşan taban oranı
# ---------------------------------------------------------------------------
def test_running_base_rate_nedensel():
    """t anındaki değer YALNIZ t'den öncekilerden hesaplanır: ilk eleman prior'dır ve
    hiçbir eleman kendi sonucunu görmez."""
    out = ol.running_base_rate([1, 1, 1, 0, 0], prior=0.5, prior_weight=2.0)
    assert out[0] == pytest.approx(0.5)                      # veri yok → prior
    assert out[1] == pytest.approx((1 + 1.0) / (1 + 2.0))    # tek gözlem (1)
    assert out[4] == pytest.approx((3 + 1.0) / (4 + 2.0))
    assert out.size == 5


def test_running_base_rate_savunmaci():
    assert ol.running_base_rate([0.3, 0.7]) is None    # {0,1} dışı
    assert ol.running_base_rate([1, 0], prior=1.5) is None
    assert ol.running_base_rate([1, 0], prior_weight=0.0) is None


# ---------------------------------------------------------------------------
# 3) Hedge / fixed-share
# ---------------------------------------------------------------------------
@pytest.fixture()
def uzmanlar():
    """Üç uzman: (a) gerçekten bilgili, (b) saf gürültü, (c) sabit 0.5."""
    rng = np.random.default_rng(21)
    n = 3000
    y = (rng.random(n) < 0.5).astype(float)
    iyi = np.where(y == 1, 0.75, 0.25)
    return np.column_stack([iyi, rng.random(n), np.full(n, 0.5)]), y


def test_hedge_en_iyi_uzmana_yakinsar_ve_regret_kucuk(uzmanlar):
    preds, y = uzmanlar
    out = ol.hedge_combine(preds, y, eta=1.0)
    assert out["best_expert"] == 0
    assert out["regret"] < 0.01                  # en iyi uzmana karşı pişmanlık küçük
    assert out["final_weights"][0] > 0.9         # ağırlık bilgili uzmanda toplandı
    assert out["vs_base"] > 0                    # taban orandan GERÇEKTEN iyi


def test_hedge_ilk_adim_esit_agirlik(uzmanlar):
    """Prequential sözleşme: ilk tahmin hiçbir sonuç görmeden yapılır → eşit ağırlık."""
    preds, y = uzmanlar
    out = ol.hedge_combine(preds, y, eta=1.0, include_base=False)
    assert np.allclose(out["weights"][0], 1.0 / 3.0)
    assert out["combined"][0] == pytest.approx(float(np.mean(preds[0])))


def test_hedge_tum_uzmanlar_gurultuyse_taban_orani_gecemez():
    """Bilgi yoksa birleşim sabit taban tahmincisini geçemez — uyarlama kenar UYDURMAZ
    (ACI DERS #9'un bu moduldeki kilidi)."""
    rng = np.random.default_rng(4)
    n = 3000
    y = (rng.random(n) < 0.45).astype(float)
    preds = rng.random((n, 3))
    out = ol.hedge_combine(preds, y, eta=1.0)
    assert out["vs_base"] <= 0.0
    assert out["best_expert"] == out["n_experts"] - 1     # en iyi "uzman" = taban oranı


def test_fixed_share_rejim_degisiminde_saf_hedgeden_iyi():
    """İki uzman rol değiştiriyor (t=1500'de). Saf Hedge ölü ağırlığı geri kazanamaz;
    fixed-share her adımda ağırlığın α'sını yeniden dağıtır → belirgin daha düşük kayıp."""
    rng = np.random.default_rng(31)
    n = 3000
    y = (rng.random(n) < 0.5).astype(float)
    a = np.where(y == 1, 0.8, 0.2)
    b = 1.0 - a
    e1 = np.concatenate([a[:1500], b[1500:]])
    e2 = np.concatenate([b[:1500], a[1500:]])
    preds = np.column_stack([e1, e2])
    saf = ol.hedge_combine(preds, y, eta=1.0, share=0.0, include_base=False)
    esnek = ol.hedge_combine(preds, y, eta=1.0, share=0.02, include_base=False)
    assert esnek["loss_combined"] < 0.5 * saf["loss_combined"]


def test_hedge_savunmaci():
    assert ol.hedge_combine([[0.5]], [2]) is None                 # y ∉ {0,1}
    assert ol.hedge_combine([[1.5]], [1]) is None                 # olasılık aralığı dışı
    assert ol.hedge_combine([[0.5], [0.5]], [1]) is None          # uyumsuz uzunluk
    assert ol.hedge_combine([[0.5]], [1], eta=0.0) is None
    assert ol.hedge_combine([[0.5]], [1], share=1.5) is None


# ---------------------------------------------------------------------------
# 4) FTRL-Proximal
# ---------------------------------------------------------------------------
def test_ftrl_ogrenilebilir_veride_tabani_gecer():
    rng = np.random.default_rng(41)
    m = 800
    x = rng.normal(0, 1, (m, 5))
    w_true = np.array([1.5, -1.0, 0.0, 0.0, 0.5])
    y = (rng.random(m) < 1 / (1 + np.exp(-(x @ w_true)))).astype(float)
    out = ol.ftrl_prequential(x, y, alpha=0.3, l1=0.05)
    assert out["uplift"] > 0.1                       # tabana karşı GERÇEK kazanç
    assert out["weights"][0] > 0.8 and out["weights"][1] < -0.4   # işaretler doğru


def test_ftrl_gurultude_taban_orani_gecemez():
    """Bilgisiz özelliklerde uplift ≈ 0 (hatta hafif negatif) olmalı — çevrimiçi öğrenici
    olmayan bilgiyi icat etmemeli."""
    rng = np.random.default_rng(42)
    m = 800
    out = ol.ftrl_prequential(rng.normal(0, 1, (m, 5)),
                              (rng.random(m) < 0.47).astype(float), alpha=0.3, l1=0.05)
    assert out["uplift"] < 0.02


def test_ftrl_l1_gercek_sifir_uretir():
    """Yüksek L1 ile ağırlıkların TAMAMI sıfırlanır (proximal kapalı-form); model taban
    tahminciye çöker — 'ölü özellik' doğrudan okunabilir olmalı."""
    rng = np.random.default_rng(43)
    m = 300
    model = ol.FTRLProximal(n_features=4, alpha=0.1, l1=1e6)
    out = model.fit_stream(rng.normal(0, 1, (m, 4)), (rng.random(m) < 0.5).astype(float))
    assert out["nonzero"] == 0
    assert np.allclose(out["preds"], 0.5)


def test_ftrl_prequential_test_then_train():
    """`update_one` ÖNCE tahmin eder sonra öğrenir: ilk örnekte tahmin 0.5 (hiç öğrenmemiş)
    ve dönen kayıp ln2'dir."""
    model = ol.FTRLProximal(n_features=2)
    assert model.predict_one([1.0, 2.0]) == pytest.approx(0.5)
    kayip = model.update_one([1.0, 2.0], 1.0)
    assert kayip == pytest.approx(np.log(2.0))
    assert model.n_seen == 1
    assert model.predict_one([1.0, 2.0]) > 0.5      # artık öğrendi


def test_ftrl_gecersiz_girdide_durum_degismez():
    model = ol.FTRLProximal(n_features=3)
    model.update_one([1.0, 2.0, 3.0], 1.0)
    z_before = model._z.copy()
    assert model.update_one([1.0, 2.0], 1.0) is None          # boyut uyumsuz
    assert model.update_one([1.0, 2.0, 3.0], 0.5) is None     # y ∉ {0,1}
    assert model.predict_one([1.0]) is None
    assert np.array_equal(model._z, z_before)


def test_ftrl_kurucu_dogrulama_ve_savunmaci():
    with pytest.raises(ValueError):
        ol.FTRLProximal(n_features=0)
    with pytest.raises(ValueError):
        ol.FTRLProximal(n_features=2, alpha=0.0)
    assert ol.FTRLProximal(n_features=2).fit_stream([[1.0, 2.0]], [0.5]) is None
    assert ol.FTRLProximal(n_features=2).fit_stream([[1.0]], [1]) is None
    assert ol.ftrl_prequential([1, 2, 3], [1, 0, 1]) is None    # 2B değil


def test_ftrl_deterministik():
    """Aynı veri → bit-denk sonuç (anayasa madde 6: rastgelelik yok)."""
    rng = np.random.default_rng(44)
    x = rng.normal(0, 1, (200, 3))
    y = (rng.random(200) < 0.5).astype(float)
    a = ol.ftrl_prequential(x, y)
    b = ol.ftrl_prequential(x, y)
    assert np.array_equal(a["preds"], b["preds"])
    assert a["mean_loss"] == b["mean_loss"]
