"""nihai.ai.ml_hygiene testleri — etiketleme + aşırı-öğrenme karşıtı araçlar (Tur-75).

En kritik iki test:

* `test_mahalanobis_kolon_bazli_guardin_KACIRDIGINI_yakalar` — mevcut `neural_shadow`
  OOD guard'ı kolon başına |z| okur. Korelasyona AYKIRI bir nokta her kolonda normal
  görünür ve o guard'dan geçer; Mahalanobis geçirmez. Testin ölçtüğü şey budur.
* `test_oof_kodlama_naif_kodlamanin_sizintisini_kapatir` — naif hedef kodlama nadir
  kategoride etiketin KOPYASIDIR; test iki kodlamanın hedefle korelasyonunu kıyaslar.
"""
from __future__ import annotations

import numpy as np
import pytest

from nihai.ai import ml_hygiene as mh


# ---------------------------------------------------------------------------
# 1) Madde 127 — üçlü bariyer
# ---------------------------------------------------------------------------
def test_ucl_bariyer_ust_temasi_kapali_form():
    """Düzgün yükselen seride üst bariyere değilir → etiket +1."""
    p = 100.0 * np.exp(np.r_[np.zeros(25), np.arange(1, 11) * 0.02])
    d = mh.triple_barrier_labels(p, events=[24], upper=1.0, lower=1.0,
                                 vertical=10, volatility=np.full(p.size, 0.01))
    assert d["label"][0] == 1
    assert d["bars_held"][0] == 1          # ilk bar +0,02 > +1σ = +0,01


def test_ucl_bariyer_dikey_temasi_etiketi_SIFIRDIR():
    """Zaman aşımı 0'dır — getirinin işareti DEĞİL (bilgi yok edilmez)."""
    p = 100.0 * np.exp(np.r_[np.zeros(25), np.arange(1, 11) * 0.0005])
    d = mh.triple_barrier_labels(p, events=[24], upper=5.0, lower=5.0,
                                 vertical=10, volatility=np.full(p.size, 0.01))
    assert d["label"][0] == 0
    assert d["ret"][0] > 0.0, "getiri pozitif ama etiket 0 — ayrım korunmalı"


def test_ucl_bariyer_side_ile_etiket_KAR_ZARARA_doner():
    """SHORT tarafta fiyat YÜKSELİRSE etiket −1 (zarar) olmalı."""
    p = 100.0 * np.exp(np.r_[np.zeros(25), np.arange(1, 11) * 0.02])
    d = mh.triple_barrier_labels(p, events=[24], upper=1.0, lower=1.0, vertical=10,
                                 volatility=np.full(p.size, 0.01), side=[-1])
    assert d["label"][0] == -1


def test_ucl_bariyer_son_barlar_kirpik_isaretlenir():
    rng = np.random.default_rng(0)
    p = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    d = mh.triple_barrier_labels(p, upper=2.0, lower=1.0, vertical=20)
    assert d["n_truncated"] == 20
    assert d["truncated"][-1] and not d["truncated"][0]


def test_ucl_bariyer_etiket_kumesi_ve_gecersiz_girdi():
    rng = np.random.default_rng(1)
    d = mh.triple_barrier_labels(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    assert set(d["counts"]) == {-1, 0, 1}
    assert sum(d["counts"].values()) == d["n"]
    assert mh.triple_barrier_labels([100.0, -5.0, 3.0, 2.0, 1.0]) is None   # negatif fiyat
    assert mh.triple_barrier_labels(np.full(50, 100.0)) is None             # σ=0


# ---------------------------------------------------------------------------
# 2) Madde 126 — CPCV
# ---------------------------------------------------------------------------
def test_cpcv_bolme_ve_patika_sayisi_kapali_form():
    """C(6,2)=15 bölme, 15·2/6 = 5 OOS patikası."""
    p = mh.cpcv_paths(6, 2)
    assert p["n_splits"] == 15 and p["n_paths"] == 5
    assert len(mh.cpcv_splits(600, n_groups=6, n_test=2)) == 15


def test_cpcv_egitim_ve_test_ortusmez():
    for tr, te, _ in mh.cpcv_splits(600, n_groups=6, n_test=2, embargo=5):
        assert len(set(tr.tolist()) & set(te.tolist())) == 0
        assert te.size == 200


def test_cpcv_purge_bitisik_bloklarin_ICINI_kesmez():
    """{2,3} gibi bitişik bloklarda iç sınır test içindedir; orada purge GEREKMEZ.

    Blok blok purge eden bir uygulama iç sınırı da keser ve eğitimi sessizce küçültür;
    bu test ayrık ({0,5}) ve bitişik ({2,3}) kombinasyonların eğitim boyutunu kıyaslar.
    """
    bolmeler = {k: (tr, te) for tr, te, k in
                mh.cpcv_splits(600, n_groups=6, n_test=2, embargo=10)}
    bitisik = bolmeler[(2, 3)][0].size          # tek blok gibi: 2 dış sınır
    ayrik = bolmeler[(1, 4)][0].size            # iki ayrı blok: 4 dış sınır
    # {0,5} BİLEREK kullanılmaz: dizinin uçlarındadır ve purge kırpılır → iki
    # kombinasyon tesadüfen EŞİT çıkar (ölçüldü: 380 = 380).
    assert bitisik > ayrik
    assert (bitisik, ayrik) == (380, 360)


def test_cpcv_gecersiz_parametrede_none():
    assert mh.cpcv_splits(100, n_groups=3, n_test=3) is None      # k_test == n_groups
    assert mh.cpcv_paths(2, 0) is None


# ---------------------------------------------------------------------------
# 3) Madde 129 — düşmanca doğrulama
# ---------------------------------------------------------------------------
def test_adversarial_ayni_dagilimda_kayma_bildirmez():
    rng = np.random.default_rng(0)
    d = mh.adversarial_validation(rng.normal(0, 1, (500, 4)), rng.normal(0, 1, (500, 4)))
    assert d["auc"] < 0.65 and d["shift_detected"] is False


def test_adversarial_kaymis_kolonu_isaretle():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (500, 4))
    b = rng.normal(0, 1, (500, 4))
    b[:, 2] += 3.0
    d = mh.adversarial_validation(a, b)
    assert d["shift_detected"] is True and d["auc"] > 0.9
    assert d["worst_feature"] == 2


def test_adversarial_auc_OOS_olculur():
    """İç-örneklem ölçülseydi ayırt edilemez veride bile AUC 1'e giderdi."""
    rng = np.random.default_rng(3)
    # p ≈ n: iç-örneklemde tam ayrılabilir, OOS'ta ayrılamaz.
    d = mh.adversarial_validation(rng.normal(0, 1, (30, 10)), rng.normal(0, 1, (30, 10)))
    assert d["auc"] < 0.85


def test_adversarial_gecersiz_girdide_none():
    rng = np.random.default_rng(0)
    assert mh.adversarial_validation(rng.normal(0, 1, (50, 3)),
                                     rng.normal(0, 1, (50, 4))) is None
    assert mh.adversarial_validation(np.zeros((5, 2)), np.zeros((50, 2))) is None


# ---------------------------------------------------------------------------
# 4) Madde 130 — kümeli MDA
# ---------------------------------------------------------------------------
def test_kumeli_mda_kopya_ozellikleri_birlikte_karistirir():
    """Üç kopya + bir gürültü: kopya kümesi BÜYÜK önem, gürültü kümesi ~0 almalı.

    Tek tek permütasyonda üç kopyanın da önemi ~0 çıkardı (biri karışınca model
    diğerini okur) — bu maddenin var oluş nedeni tam olarak budur.
    """
    rng = np.random.default_rng(0)
    n = 600
    z = rng.normal(0, 1, n)
    X = np.column_stack([z, z + rng.normal(0, 0.05, n), z + rng.normal(0, 0.05, n),
                         rng.normal(0, 1, n)])
    y = (z + rng.normal(0, 0.5, n) > 0).astype(float)
    d = mh.clustered_mda(X, y, n_clusters=2, n_repeats=5)
    onem = sorted(d["cluster_importance"].values())
    assert onem[-1] > 0.20, "bilgi taşıyan küme önemli çıkmadı"
    assert abs(onem[0]) < 0.05, "gürültü kümesi önemli göründü"


def test_kumeli_mda_gecersiz_girdide_none():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (100, 3))
    assert mh.clustered_mda(X, np.zeros(100)) is None            # tek sınıf
    assert mh.clustered_mda(X, rng.random(100) > 0.5, n_clusters=9) is None


# ---------------------------------------------------------------------------
# 5) Madde 132 — Mahalanobis OOD
# ---------------------------------------------------------------------------
def test_mahalanobis_kolon_bazli_guardin_KACIRDIGINI_yakalar():
    """ρ=0,95 referansta [2,2] normal, [2,−2] İMKÂNSIZ — ikisi de kolon-|z|=2.

    `neural_shadow`ın `max|z|` guard'ı ikisini AYIRAMAZ; Mahalanobis ayırır.
    """
    rng = np.random.default_rng(0)
    R = rng.multivariate_normal([0, 0], [[1.0, 0.95], [0.95, 1.0]], 2000)
    d = mh.mahalanobis_ood(R, np.array([[2.0, 2.0], [2.0, -2.0]]))
    # İki nokta da her KOLONDA aynı |z|'ye sahip:
    z = (np.array([[2.0, 2.0], [2.0, -2.0]]) - R.mean(0)) / R.std(0)
    assert np.abs(z).max(axis=1)[0] == pytest.approx(np.abs(z).max(axis=1)[1], abs=0.1)
    # Ama Mahalanobis onları AYIRIYOR:
    assert d["is_ood"][0] is np.False_ and d["is_ood"][1] is np.True_
    assert d["d2"][1] > 10.0 * d["d2"][0]


def test_mahalanobis_tekil_kovaryansta_buzme_raporlanir():
    """İki kolon TAM özdeş → kovaryans tekil; büzme uygulanır ve GİZLENMEZ."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 200)
    R = np.column_stack([x, x])
    d = mh.mahalanobis_ood(R, np.array([[0.0, 0.0]]))
    assert d is not None and d["shrinkage"] > 0.0


def test_mahalanobis_esik_boyutla_buyur():
    rng = np.random.default_rng(2)
    d2 = mh.mahalanobis_ood(rng.normal(0, 1, (500, 2)), np.zeros((1, 2)))
    d8 = mh.mahalanobis_ood(rng.normal(0, 1, (500, 8)), np.zeros((1, 8)))
    assert d8["threshold"] > d2["threshold"]


def test_mahalanobis_gecersiz_girdide_none():
    rng = np.random.default_rng(0)
    assert mh.mahalanobis_ood(rng.normal(0, 1, (50, 2)), np.zeros((1, 3))) is None
    assert mh.mahalanobis_ood(rng.normal(0, 1, (50, 2)), np.zeros((1, 2)),
                              alpha=1.5) is None


# ---------------------------------------------------------------------------
# 6) Madde 133 — nötrleştirme
# ---------------------------------------------------------------------------
def test_notrlestirme_maruziyet_korelasyonunu_sifirlar():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 1, (300, 1))
    F = np.column_stack([2.0 * e[:, 0] + rng.normal(0, 0.3, 300), rng.normal(0, 1, 300)])
    d = mh.neutralize(F, e)
    assert abs(np.corrcoef(F[:, 0], e[:, 0])[0, 1]) > 0.95
    assert abs(np.corrcoef(d["neutral"][:, 0], e[:, 0])[0, 1]) < 1e-10
    assert d["beta"][0, 0] == pytest.approx(2.0, abs=0.1)


def test_notrlestirme_SABIT_TERIMI_korur_ortalamayi_degil():
    """Çıktının ortalaması β₀'dır; maruziyetin ortalaması 0 değilse mean(F) DEĞİLDİR.

    Bu ayrım ölçülmüştür: β₀ = mean(F) − β·mean(E). İkisini karıştıran bir uygulama
    çıktıyı β·mean(E) kadar kaydırır.
    """
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1, (300, 1))
    F = (5.0 + 2.0 * e[:, 0] + rng.normal(0, 0.3, 300)).reshape(-1, 1)
    d = mh.neutralize(F, e)
    beklenen = float(F.mean()) - float(d["beta"][0, 0]) * float(e.mean())
    assert float(d["neutral"].mean()) == pytest.approx(beklenen, abs=1e-9)
    # Maruziyet ORTALANMIŞSA ikisi çakışır — iddia böylece iki yönlü kilitlenir.
    d2 = mh.neutralize(F, e - e.mean())
    assert float(d2["neutral"].mean()) == pytest.approx(float(F.mean()), abs=1e-9)


def test_notrlestirme_oransal_uygulanir():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, (300, 1))
    F = (2.0 * e[:, 0] + rng.normal(0, 0.3, 300)).reshape(-1, 1)
    yari = mh.neutralize(F, e, proportion=0.5)
    tam = mh.neutralize(F, e, proportion=1.0)
    k_yari = abs(np.corrcoef(yari["neutral"][:, 0], e[:, 0])[0, 1])
    k_tam = abs(np.corrcoef(tam["neutral"][:, 0], e[:, 0])[0, 1])
    assert k_tam < 1e-10 < k_yari < 0.99


def test_notrlestirme_gecersiz_girdide_none():
    rng = np.random.default_rng(0)
    assert mh.neutralize(rng.normal(0, 1, (10, 2)), rng.normal(0, 1, (5, 1))) is None
    assert mh.neutralize(rng.normal(0, 1, (10, 2)), rng.normal(0, 1, (10, 1)),
                         proportion=2.0) is None


# ---------------------------------------------------------------------------
# 7) Madde 146 — fBM
# ---------------------------------------------------------------------------
def test_fbm_uretilen_hurst_geri_olculur():
    """H = 0,2 / 0,5 / 0,8 patikalarında ölçülen Hurst SIRALI ve yakın olmalı."""
    from nihai.ai.quant_math import hurst_exponent
    olculen = [hurst_exponent(mh.fbm_paths(2048, hurst=h, seed=1)[0], max_lag=40)
               for h in (0.2, 0.5, 0.8)]
    assert olculen[0] < olculen[1] < olculen[2]
    assert olculen[0] == pytest.approx(0.2, abs=0.10)
    assert olculen[2] == pytest.approx(0.8, abs=0.12)


def test_fbm_sekil_ve_determinizm():
    a = mh.fbm_paths(512, hurst=0.7, n_paths=3, seed=5)
    b = mh.fbm_paths(512, hurst=0.7, n_paths=3, seed=5)
    assert a.shape == (3, 512) and np.array_equal(a, b)
    assert not np.array_equal(a, mh.fbm_paths(512, hurst=0.7, n_paths=3, seed=6))


def test_fbm_gecersiz_hurst_none():
    assert mh.fbm_paths(512, hurst=0.0) is None
    assert mh.fbm_paths(512, hurst=1.0) is None
    assert mh.fbm_paths(3, hurst=0.5) is None


# ---------------------------------------------------------------------------
# 8) Madde 147 — OOF hedef kodlama
# ---------------------------------------------------------------------------
def test_oof_kodlama_naif_kodlamanin_sizintisini_kapatir():
    """NADİR kategoride naif kod etiketin KOPYASIDIR; OOF kod prior'a düşer."""
    rng = np.random.default_rng(0)
    n = 300
    cat = np.array(["ortak"] * (n - 1) + ["nadir"])
    y = np.r_[rng.normal(0.0, 1.0, n - 1), [9.0]]
    d = mh.oof_target_encode(cat, y, n_folds=5, smoothing=0.0)
    # Naif kod: nadir kategorinin ortalaması = 9,0 (etiketin ta kendisi).
    assert d["full_map"]["nadir"] == pytest.approx(9.0)
    # OOF kod: satır kendi katında yok → kategori eğitimde HİÇ görülmez → prior.
    assert d["encoded"][-1] == pytest.approx(float(y[: n - 1].mean()), abs=0.3)


def test_oof_kodlama_sinyali_tasir():
    """Gerçek sinyalde OOF kod hedefle korele olmalı (sızıntı kapatmak sağır etmez)."""
    rng = np.random.default_rng(1)
    cat = rng.choice(["a", "b", "c"], 600)
    y = np.where(cat == "a", 1.0, 0.0) + rng.normal(0, 0.2, 600)
    d = mh.oof_target_encode(cat, y)
    assert np.corrcoef(d["encoded"], y)[0, 1] > 0.8
    assert d["full_map"]["a"] > d["full_map"]["b"]


def test_oof_buzme_az_gozlemli_kategoriyi_priora_ceker():
    rng = np.random.default_rng(2)
    cat = np.array(["cok"] * 200 + ["az"] * 3)
    y = np.r_[rng.normal(0.0, 0.1, 200), [5.0, 5.0, 5.0]]
    az_sm = mh.oof_target_encode(cat, y, smoothing=0.0)["full_map"]["az"]
    cok_sm = mh.oof_target_encode(cat, y, smoothing=50.0)["full_map"]["az"]
    assert az_sm > cok_sm, "büzme az gözlemli kategoriyi prior'a çekmedi"


def test_oof_gecersiz_girdide_none():
    assert mh.oof_target_encode(["a", "b"], [1.0, 2.0, 3.0]) is None
    assert mh.oof_target_encode(["a"] * 10, np.arange(10.0), n_folds=1) is None
