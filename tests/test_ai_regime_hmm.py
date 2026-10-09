"""nihai.ai.regime_hmm testleri — bilinen-parametre geri-bulma + look-ahead sözleşmesi
+ savunmacı girdi yolları. Ağ/IO yok (saf fonksiyonlar), rastgelelik YALNIZ sentetik veri
üretiminde (sabit seed) — modülün kendisi deterministiktir.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from nihai.ai import regime_hmm as H

# Sentetik iki-rejim gerçeği: yapışkan geçişler, farklı oynaklık.
_A_TRUE = np.array([[0.98, 0.02], [0.05, 0.95]])
_MU_TRUE = (0.0005, -0.0015)
_SD_TRUE = (0.006, 0.020)


def _iki_rejim(n: int = 2000, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Markov zincirinden durum örnekle, duruma göre Gauss gözlem üret."""
    rng = np.random.default_rng(seed)
    s = 0
    states = np.empty(n, dtype=int)
    x = np.empty(n, dtype=float)
    for t in range(n):
        states[t] = s
        x[t] = rng.normal(_MU_TRUE[s], _SD_TRUE[s])
        s = int(rng.choice(2, p=_A_TRUE[s]))
    return x, states


@pytest.fixture(scope="module")
def veri() -> tuple[np.ndarray, np.ndarray]:
    return _iki_rejim()


@pytest.fixture(scope="module")
def model(veri) -> dict:
    m = H.gaussian_hmm_fit(veri[0], n_states=2)
    assert m is not None
    return m


# ---------------------------------------------------------------------------
# A. Uyum: bilinen parametreleri geri buluyor mu
# ---------------------------------------------------------------------------
def test_fit_oynakliklari_geri_buluyor(model):
    """Durumlar ORTALAMAYA göre sıralanır → kimlik kararlıdır. Bu veride negatif-ortalamalı
    durum YÜKSEK oynaklıklıdır (gerçek: mu=-0.0015, sd=0.020)."""
    assert model["stds"][0] == pytest.approx(_SD_TRUE[1], rel=0.15)
    assert model["stds"][1] == pytest.approx(_SD_TRUE[0], rel=0.15)
    assert model["means"][0] < model["means"][1]
    assert model["converged"] is True and model["n"] == 2000


def test_fit_gecis_matrisi_geri_buluyor(model):
    """Yapışkanlık geri bulunmalı: her iki durumun köşegeni > 0.9 ve satırlar toplamı 1."""
    trans = np.asarray(model["transition"])
    assert np.allclose(trans.sum(axis=1), 1.0)
    assert trans[0, 0] > 0.85 and trans[1, 1] > 0.9
    # sıralanmış durum 1 = gerçek durum 0 (kalıcılık 0.98) → daha yapışkan olmalı
    assert trans[1, 1] > trans[0, 0]


def test_fit_bic_ve_serbest_parametre_sayisi(model):
    """K durumda serbest parametre = 2K (ortalama+std) + K(K−1) geçiş + (K−1) başlangıç."""
    assert model["n_params"] == 2 * 2 + 2 * 1 + 1
    assert model["bic"] == pytest.approx(
        model["n_params"] * np.log(model["n"]) - 2.0 * model["loglik"])


def test_fit_deterministik(veri):
    """Rastgelelik YOK: aynı girdi → bit-denk parametreler (anayasa madde 6)."""
    a = H.gaussian_hmm_fit(veri[0], n_states=2)
    b = H.gaussian_hmm_fit(veri[0], n_states=2)
    assert a == b


def test_fit_savunmaci_girdiler():
    assert H.gaussian_hmm_fit([1, 2, 3], 2) is None                  # çok kısa
    assert H.gaussian_hmm_fit([1.0] * 100, 2) is None                # sabit seri
    assert H.gaussian_hmm_fit(list(range(100)), 1) is None           # n_states < 2
    assert H.gaussian_hmm_fit(list(range(100)), 7) is None           # n_states > 6
    assert H.gaussian_hmm_fit([1.0, np.nan] * 50, 2) is None         # NaN
    assert H.gaussian_hmm_fit(list(range(15)), 2) is None            # n < 10K


def test_varyans_tabani_dejenere_cozumu_engelliyor():
    """Aynı değerin çok kez tekrarlandığı seride bir durum tek noktaya ÇÖKEBİLİR ve
    olabilirlik sonsuza gider. σ tabanı bunu engeller: fit ya sonlu döner ya None."""
    x = np.concatenate([np.full(150, 0.01), np.random.default_rng(0).normal(0, 0.02, 150)])
    m = H.gaussian_hmm_fit(x, 2)
    if m is not None:
        assert np.all(np.isfinite(m["stds"])) and min(m["stds"]) > 0.0
        assert np.isfinite(m["loglik"])


# ---------------------------------------------------------------------------
# B. Look-ahead sözleşmesi — modülün EN kritik kuralı
# ---------------------------------------------------------------------------
def test_filtre_nedensel_bayragi_ve_gecmise_bagimsizlik(veri, model):
    """`hmm_filter` NEDENSELDİR: t anındaki değer, t'den SONRAKİ gözlemler değişse bile
    AYNI kalmalıdır. Look-ahead yasağının makine kontrolü budur."""
    x = veri[0]
    tam = H.hmm_filter(x, model)
    kesik = H.hmm_filter(x[:500], model)
    assert tam["causal"] is True and kesik["causal"] is True
    assert np.allclose(tam["filtered"][:500], kesik["filtered"])
    assert np.allclose(tam["predicted"][:500], kesik["predicted"])


def test_duzeltme_look_ahead_bayragi_tasiyor(veri, model):
    """`hmm_smooth`/`hmm_viterbi` geleceği görür → `causal=False` MAKİNE-OKUNUR biçimde
    taşınmalı (çağıran yanlışlıkla nedensel sanmasın)."""
    assert H.hmm_smooth(veri[0], model)["causal"] is False
    assert H.hmm_viterbi(veri[0], model)["causal"] is False


def test_duzeltme_filtreden_daha_isabetli(veri, model):
    """Geleceği gören düzeltme, nedensel filtreden İSABETLİ olmalı — aksi hâlde ileri-geri
    özyinelemesi yanlış kurulmuş demektir (matematiksel çapa)."""
    x, gercek = veri
    filt = H.hmm_filter(x, model)["filtered"].argmax(axis=1)
    duz = H.hmm_smooth(x, model)["smoothed"].argmax(axis=1)
    # Durum kimliği permütasyona kadar tanımlı → iki eşlemenin İYİSİ alınır.
    isabet_f = max((filt == gercek).mean(), (filt != gercek).mean())
    isabet_d = max((duz == gercek).mean(), (duz != gercek).mean())
    assert isabet_f > 0.85
    assert isabet_d >= isabet_f


def test_olasiliklar_normalize(veri, model):
    filt = H.hmm_filter(veri[0], model)
    assert np.allclose(filt["filtered"].sum(axis=1), 1.0)
    assert np.allclose(filt["predicted"].sum(axis=1), 1.0)
    assert np.allclose(H.hmm_smooth(veri[0], model)["smoothed"].sum(axis=1), 1.0)


def test_viterbi_yolu_gecerli_ve_isabetli(veri, model):
    vit = H.hmm_viterbi(veri[0], model)
    assert vit["path"].shape == (2000,)
    assert set(np.unique(vit["path"])) <= {0, 1}
    isabet = max((vit["path"] == veri[1]).mean(), (vit["path"] != veri[1]).mean())
    assert isabet > 0.9


# ---------------------------------------------------------------------------
# C. Boşluğu kapatan çıktı: geçiş OLASILIĞI
# ---------------------------------------------------------------------------
def test_sonraki_durum_olasiliklari_nedensel(veri, model):
    """`hmm_next_state_probs` yalnız geçmişe bakar ve h adım ileri taşır."""
    out = H.hmm_next_state_probs(veri[0], model, horizon=1)
    assert out["causal"] is True and out["horizon"] == 1
    assert sum(out["next"]) == pytest.approx(1.0)
    assert sum(out["current"]) == pytest.approx(1.0)
    assert 0.0 <= out["switch_prob"] <= 1.0
    assert out["mode_prob"] == pytest.approx(max(out["next"]))


def test_uzun_ufuk_duragan_dagilima_yaklasiyor(veri, model):
    """h → büyük: başlangıç durumundan BAĞIMSIZ durağan dağılıma yakınsar (Markov
    zincirinin temel özelliği — geçiş matrisinin doğru kurulduğunun çapası)."""
    uzak = H.hmm_next_state_probs(veri[0], model, horizon=2000)
    stat = H.hmm_transition_summary(model)["stationary"]
    assert np.allclose(uzak["next"], stat, atol=0.02)


def test_sonraki_durum_savunmaci(veri, model):
    assert H.hmm_next_state_probs(veri[0], model, horizon=0) is None
    assert H.hmm_next_state_probs(veri[0], {"means": [1.0]}) is None
    assert H.hmm_next_state_probs([np.nan] * 50, model) is None


def test_gecis_ozeti_sure_ve_duragan(model):
    ozet = H.hmm_transition_summary(model)
    trans = np.asarray(model["transition"])
    assert ozet["persistence"] == pytest.approx(list(np.diag(trans)))
    for p, d in zip(ozet["persistence"], ozet["expected_duration"], strict=False):
        assert d == pytest.approx(1.0 / (1.0 - p))
    assert sum(ozet["stationary"]) == pytest.approx(1.0)
    # Durağan dağılım geçiş matrisinin SOL öz-vektörüdür: π·P = π
    assert np.allclose(np.asarray(ozet["stationary"]) @ trans, ozet["stationary"], atol=1e-8)


def test_mutlak_yapiskan_durumda_sure_none():
    """P(i→i)=1 → beklenen süre SONSUZDUR; sonsuz bir SAYI değildir → None (uydurulmaz)."""
    ozet = H.hmm_transition_summary(
        {"means": [0.0, 1.0], "stds": [1.0, 1.0], "start": [0.5, 0.5],
         "transition": [[1.0, 0.0], [0.1, 0.9]]})
    assert ozet["expected_duration"][0] is None
    assert ozet["expected_duration"][1] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# D. Etiketleme + durum sayısı seçimi
# ---------------------------------------------------------------------------
def test_durum_etiketleri_classify_regime_sozlugunde(model):
    labels = H.hmm_state_labels(model)
    assert len(labels) == 2
    for lab in labels:
        yon, vol = lab.split("_")
        assert yon in {"up", "down", "flat"} and vol in {"low", "mid", "high"}
    # Yüksek oynaklıklı durum (index 0) 'high', diğeri 'low' olmalı.
    assert labels[0].endswith("high") and labels[1].endswith("low")


def test_durum_sayisi_secimi_bic(veri):
    """İki-rejim verisinde BIC 2'yi seçmeli ve karar KESİN olmalı (ΔBIC ≥ 2)."""
    sec = H.hmm_select_states(veri[0], candidates=(2, 3, 4))
    assert sec is not None
    assert sec["best"] == 2 and sec["decisive"] is True
    assert sec["delta_bic"] > 2.0
    assert sec["fits"][2]["bic"] < sec["fits"][3]["bic"]


def test_durum_secimi_hicbiri_uymazsa_none():
    assert H.hmm_select_states([1, 2, 3], candidates=(2, 3)) is None


# ---------------------------------------------------------------------------
# E. Optimizasyon denkliği (Tur-34) — iddia ÖLÇÜLEBİLİR olmalı
# ---------------------------------------------------------------------------
def _forward_referans(log_emis, log_start, log_trans):
    """`_forward`ın ÖZGÜN (Tur-34 öncesi) hâli: adım başına scipy.logsumexp, log uzayı.
    Ölçeklenmiş uygulama bunun BİT-DENKİ olamaz (toplama sırası değişti) — bu yüzden
    denklik İDDİA edilmez, ÖLÇÜLÜR (Tur-33 dersi #4)."""
    t_len, k = log_emis.shape
    log_alpha = np.empty((t_len, k), dtype=float)
    log_alpha[0] = log_start + log_emis[0]
    for t in range(1, t_len):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + log_emis[t]
    return log_alpha, float(logsumexp(log_alpha[-1]))


def _backward_referans(log_emis, log_trans):
    t_len, k = log_emis.shape
    log_beta = np.zeros((t_len, k), dtype=float)
    for t in range(t_len - 2, -1, -1):
        log_beta[t] = logsumexp(log_trans + (log_emis[t + 1] + log_beta[t + 1])[None, :],
                                axis=1)
    return log_beta


def test_olceklenmis_ozyineleme_log_uzayiyla_denk(veri, model):
    """Ölçeklenmiş (Rabiner) özyineleme ile özgün log-uzayı özyinelemesi AYNI sayıyı
    vermelidir. Ölçüldü (T=2000, K=2): log_alpha BAĞIL fark 1,5e-14, loglik bağıl fark
    1,5e-14, filtreli olasılıklarda mutlak fark 5,6e-16 ve **argmax kararında 0/2000 fark**.
    Eşikler bu ölçümün biraz üstünde tutuldu — sıkılaştırma değil, REGRESYON kilidi."""
    means, stds, log_trans, log_start = H._unpack(model)
    log_emis = H._log_emissions(veri[0], means, stds)

    la_ref, ll_ref = _forward_referans(log_emis, log_start, log_trans)
    la_new, ll_new = H._forward(log_emis, log_start, log_trans)
    assert np.max(np.abs((la_ref - la_new) / la_ref)) < 1e-12
    assert abs((ll_ref - ll_new) / ll_ref) < 1e-12

    lb_ref = _backward_referans(log_emis, log_trans)
    lb_new = H._backward(log_emis, log_trans)
    sonlu = np.isfinite(lb_ref) & (np.abs(lb_ref) > 0)
    assert np.max(np.abs((lb_ref[sonlu] - lb_new[sonlu]) / lb_ref[sonlu])) < 1e-12


def test_olceklenmis_filtre_kararda_ayni(veri, model):
    """Asıl ölçüt SAYI değil KARARDIR: filtreli dağılımın argmax'ı hiçbir barda
    değişmemeli (Tur-33'ün '24.900 panel satırında 0 oy farkı' kuralının HMM karşılığı)."""
    means, stds, log_trans, log_start = H._unpack(model)
    log_emis = H._log_emissions(veri[0], means, stds)
    t_len, k = log_emis.shape
    ref = np.empty((t_len, k))
    log_prior = log_start
    for t in range(t_len):
        post = log_prior + log_emis[t]
        norm = logsumexp(post)
        ref[t] = np.exp(post - norm)
        log_prior = logsumexp((post - norm)[:, None] + log_trans, axis=0)
    yeni = H.hmm_filter(veri[0], model)["filtered"]
    assert np.max(np.abs(ref - yeni)) < 1e-12
    assert int((ref.argmax(axis=1) != yeni.argmax(axis=1)).sum()) == 0


def test_bozuk_model_sozlugu_none(veri):
    bozuk = {"means": [0.0, 1.0], "stds": [0.0, 1.0], "start": [0.5, 0.5],
             "transition": [[0.5, 0.5], [0.5, 0.5]]}          # σ=0 geçersiz
    assert H.hmm_filter(veri[0], bozuk) is None
    assert H.hmm_smooth(veri[0], bozuk) is None
    assert H.hmm_viterbi(veri[0], bozuk) is None
    assert H.hmm_transition_summary(bozuk) is None
    assert H.hmm_state_labels(bozuk) is None
    assert H.hmm_filter(veri[0], "model degil") is None
