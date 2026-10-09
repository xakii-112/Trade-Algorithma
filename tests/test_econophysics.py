"""nihai.ai.econophysics testleri — ekonofizik/kaos/sinyal ayrıştırma (Tur-75).

Doğrulama stratejisi: her tahminci PARAMETRESİ BİLİNEN bir üretici süreçten geri
kazanım ile sınanır (LPPLS'in t_c/m/ω'sı, Hénon'un λ'sı, A→B nedensel kurgusunda
taşma yönü), ya da bir ÖZDEŞLİĞE bağlanır (EMD'nin Σimf+artık = x kimliği, FEVD
satır toplamı = 1). Ayrıca her aile için bir NEGATİF kontrol koşar: bağımsız/karışık
seride yöntemin bir şey BULMAMASI gerekir (Tur-60 pozitif+negatif kontrol disiplini).
"""
from __future__ import annotations

import numpy as np
import pytest

from nihai.ai import econophysics as ep


def _henon(n: int = 2000) -> np.ndarray:
    x, y = 0.1, 0.3
    out = []
    for _ in range(n + 500):
        x, y = 1.0 - 1.4 * x * x + y, 0.3 * x
        out.append(x)
    return np.asarray(out[500:], dtype=float)


# ---------------------------------------------------------------------------
# 1) Madde 086 — LPPLS
# ---------------------------------------------------------------------------
def test_lppls_uretilen_parametreleri_geri_kazanir():
    """Bilinen (t_c=240, m=0.4, ω=9.0) LPPLS serisinden parametreler geri çıkmalı."""
    rng = np.random.default_rng(0)
    n, tc, m, om = 200, 240.0, 0.4, 9.0
    t = np.arange(n, dtype=float)
    y = (5.0 - 0.02 * (tc - t) ** m
         + 0.004 * (tc - t) ** m * np.cos(om * np.log(tc - t) - 1.0)
         + rng.normal(0, 0.001, n))
    d = ep.lppls_fit(y)
    assert d["t_c"] == pytest.approx(tc, abs=2.0)
    assert d["m"] == pytest.approx(m, abs=0.05)
    assert d["omega"] == pytest.approx(om, abs=0.5)
    assert d["r2"] > 0.99
    assert d["filter_pass"] is True


def test_lppls_B_pozitifse_filtre_gecmez():
    """Süper-üstel YÜKSELİŞ B<0 ister; düz gürültüde filtre geçmemeli."""
    rng = np.random.default_rng(3)
    d = ep.lppls_fit(np.cumsum(rng.normal(0, 0.01, 200)))
    assert d is not None
    assert isinstance(d["filter_pass"], bool)      # verdikt üretiliyor, sayı gizlenmiyor
    assert d["t_c"] > 199.0, "t_c geçmişte olamaz"


def test_lppls_tc_dagilimi_raporlanir():
    rng = np.random.default_rng(1)
    d = ep.lppls_fit(np.cumsum(rng.normal(0, 0.01, 120)), n_starts=6)
    assert d["t_c_std"] is not None and d["n_converged"] >= 1


def test_lppls_kisa_ve_sabit_seride_none():
    assert ep.lppls_fit(np.arange(10.0)) is None
    assert ep.lppls_fit(np.full(60, 5.0)) is None


# ---------------------------------------------------------------------------
# 2) Madde 089 — EMD / Hilbert
# ---------------------------------------------------------------------------
def test_emd_ayristirma_kimligi_makine_hassasiyetinde():
    """x = Σ IMF + artık — bu YAPISAL bir kimliktir, yaklaşık değil."""
    t = np.linspace(0, 20 * np.pi, 512)
    x = np.sin(t) + 0.3 * np.sin(4 * t) + np.linspace(0, 2, 512)
    d = ep.emd(x)
    rec = sum(d["imfs"]) + d["residue"]
    assert np.abs(rec - x).max() < 1e-12
    assert d["n_imf"] >= 2


def test_emd_monoton_seride_imf_uretmez():
    """Ekstremumu olmayan seri tamamen ARTIĞA düşer — hata değil, doğru cevap."""
    d = ep.emd(np.linspace(0.0, 10.0, 64))
    assert d["n_imf"] == 0
    assert np.allclose(d["residue"], np.linspace(0.0, 10.0, 64))


def test_hilbert_anlik_frekans_bilinen_sinuste_dogru():
    """20 örnekte bir çevrim → anlık frekans ≈ 0,05 çevrim/bar."""
    x = np.sin(2 * np.pi * np.arange(400) / 20.0)
    d = ep.hilbert_spectrum(x)
    # Kenar etkileri var; ORTA bölge kullanılır.
    assert float(np.median(d["inst_freq"][50:-50])) == pytest.approx(0.05, abs=0.005)
    assert np.all(d["amplitude"][50:-50] == pytest.approx(1.0, abs=0.05))


def test_emd_ve_hilbert_gecersiz_girdide_none():
    assert ep.emd([1.0, 2.0]) is None
    assert ep.hilbert_spectrum([1.0, 2.0]) is None


# ---------------------------------------------------------------------------
# 3) Madde 090 — transfer entropisi
# ---------------------------------------------------------------------------
def test_transfer_entropisi_yonu_dogru_bulur():
    """B[i] = 0.8·A[i−1] + gürültü → T_{A→B} ≫ T_{B→A}."""
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.normal(0, 1, n)
    b = np.zeros(n)
    for i in range(1, n):
        b[i] = 0.7 * a[i - 1] + 0.3 * rng.normal()
    d = ep.transfer_entropy(a, b)
    assert d["te_xy"] > 0.3
    assert d["te_xy"] > 5.0 * d["te_yx"]
    assert d["net"] > 0.0


def test_transfer_entropisi_bagimsiz_seride_gurultu_tabaninda_kalir():
    """NEGATİF KONTROL: bağımsız serilerde TE, karıştırılmış tabandan ayrılmamalı."""
    rng = np.random.default_rng(7)
    d = ep.transfer_entropy(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000))
    assert d["te_xy"] < 3.0 * d["te_shuffled"] + 0.05


def test_transfer_entropisi_duz_seride_none():
    assert ep.transfer_entropy(np.ones(100), np.arange(100.0)) is None
    assert ep.transfer_entropy(np.arange(5.0), np.arange(5.0)) is None


# ---------------------------------------------------------------------------
# 4) Madde 093 — Takens + Lyapunov
# ---------------------------------------------------------------------------
def test_takens_gomme_boyutu_ve_gecikmesi():
    x = np.arange(10.0)
    e = ep.takens_embedding(x, dim=3, tau=2)
    assert e.shape == (6, 3)
    # İlk satır: [x(4), x(2), x(0)] = [4,2,0]
    assert list(e[0]) == [4.0, 2.0, 0.0]
    assert ep.takens_embedding(np.arange(3.0), dim=3, tau=2) is None


def test_lyapunov_henon_bilinen_ussu_geri_verir():
    """Hénon haritasının en büyük Lyapunov üssü ≈ 0,42."""
    d = ep.lyapunov_rosenstein(_henon(2000), dim=3, tau=1, max_t=15)
    assert d["lambda"] == pytest.approx(0.42, abs=0.12)
    assert d["r2"] > 0.95, "kaotik seride ayrışma eğrisi UZUN aralıkta doğrusaldır"
    assert d["predictability_horizon"] is not None


def test_lyapunov_ayirt_edici_buyukluk_egrinin_dogrusalligidir():
    """NEGATİF KONTROL: beyaz gürültü de pozitif λ verir — ayıran şey r²'dir."""
    rng = np.random.default_rng(0)
    kaos = ep.lyapunov_rosenstein(_henon(2000), dim=3, tau=1, max_t=15)
    gur = ep.lyapunov_rosenstein(rng.normal(0, 1, 2000), dim=3, tau=1, max_t=15)
    assert gur["lambda"] > 0.0, "gürültüde de λ>0 çıkar (modül docstring'i §2)"
    assert gur["r2"] < 0.6 < kaos["r2"]
    # Karıştırılmış kontrol: gürültünün λ'sı kendi surrogate'inden AYRILMAZ.
    assert gur["lambda_ratio"] == pytest.approx(1.0, abs=0.25)
    assert kaos["lambda_ratio"] > 2.0


def test_lyapunov_kisa_ve_sabit_seride_none():
    assert ep.lyapunov_rosenstein(np.arange(20.0)) is None
    assert ep.lyapunov_rosenstein(np.full(200, 3.0)) is None


# ---------------------------------------------------------------------------
# 5) Madde 095 — CUSUM
# ---------------------------------------------------------------------------
def test_cusum_kapali_form_esikte_tetikler():
    """Eşik 1,0; +0,4'lük üç adım kümülatifi 1,2'ye taşır → 3. adımda olay."""
    d = ep.cusum_filter(np.array([0.0, 0.4, 0.8, 1.2, 1.6]), threshold=1.0)
    assert list(d["events"]) == [3]
    assert list(d["directions"]) == [1]


def test_cusum_olay_dizisi_kapali_form_izler():
    """Δ = [+1,5 · −0,1 · −1,4 · −1,5], h=1 → olaylar 1(+), 3(−), 4(−).

    Yukarı olaydan sonra `sp` sıfırlanır ama `sn` kendi birikimini SÜRDÜRÜR:
    3. adımda −0,1−1,4 = −1,5 eşiği aşar. Tetikleyen tarafın sıfırlanması burada
    izlenebilir tek yerdir.
    """
    d = ep.cusum_filter(np.array([0.0, 1.5, 1.4, 0.0, -1.5]), threshold=1.0)
    assert list(d["events"]) == [1, 3, 4]
    assert list(d["directions"]) == [1, -1, -1]


def test_cusum_rejim_kaymasini_yakalar():
    rng = np.random.default_rng(1)
    sakin = np.cumsum(rng.normal(0, 0.001, 300))
    kayma = sakin[-1] + 0.5 + np.cumsum(rng.normal(0, 0.001, 50))
    d = ep.cusum_filter(np.r_[sakin, kayma])
    assert d["n_events"] >= 1
    assert d["events"][-1] >= 295, "kayma barının civarında olay yok"


def test_cusum_gecersiz_esikte_none():
    assert ep.cusum_filter([1.0, 2.0, 3.0], threshold=0.0) is None
    assert ep.cusum_filter([5.0, 5.0, 5.0]) is None       # σ(Δ)=0 → eşik yok


# ---------------------------------------------------------------------------
# 6) Madde 098 — perkolasyon
# ---------------------------------------------------------------------------
def test_perkolasyon_iki_bloklu_agda_birlesmeyi_bulur():
    """Blok-içi ρ=0,9 · bloklar-arası ρ=0,1 → pay 0,5'te durur, 0,1'de 1,0 olur."""
    n = 20
    c = np.where((np.arange(n)[:, None] < 10) == (np.arange(n)[None, :] < 10), 0.9, 0.1)
    np.fill_diagonal(c, 1.0)
    d = ep.percolation_profile(c)
    assert d["giant_share"][0] == pytest.approx(1.0 / n)        # eşik 0,95 → kenar yok
    assert 0.5 in np.round(d["giant_share"], 6)
    assert d["giant_share"][-1] == pytest.approx(1.0)           # eşik 0,05 → hepsi bağlı
    assert d["jump"] > 0.0


def test_perkolasyon_bagimsiz_agda_dev_bilesen_gec_dogar():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, (400, 15))
    c = np.corrcoef(x.T)
    d = ep.percolation_profile(c)
    assert d["giant_share"][0] < 0.3, "bağımsız varlıklarda yüksek eşikte dev bileşen yok"


def test_perkolasyon_gecersiz_girdide_none():
    assert ep.percolation_profile([[1.0, 0.5]]) is None          # kare değil
    assert ep.percolation_profile([[1.0, np.nan], [np.nan, 1.0]]) is None


# ---------------------------------------------------------------------------
# 7) Madde 100 — dalgacık gürültü ayıklama
# ---------------------------------------------------------------------------
def test_dalgacik_gurultuyu_azaltir_sinyali_korur():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 4 * np.pi, 512)
    temiz = np.sin(t)
    gur = temiz + rng.normal(0, 0.3, 512)
    d = ep.wavelet_denoise(gur, levels=3)
    rmse_ham = float(np.sqrt(np.mean((gur - temiz) ** 2)))
    rmse_tmz = float(np.sqrt(np.mean((d["denoised"] - temiz[:d["n_used"]]) ** 2)))
    assert rmse_tmz < 0.5 * rmse_ham


def test_dalgacik_duzgun_seride_neredeyse_dokunmaz():
    """DÜZGÜN bir seride en ince detaylar ~0 → σ̂≈0 → eşik≈0 → seri geri gelir.

    Sinyal ne kadar hızlı salınırsa d₁ o kadar büyür ve evrensel eşik GERÇEK yapıyı
    da siler; bu bir kusur değil, MAD(d₁)'i gürültü tahmini saymanın bedelidir.
    """
    t = np.linspace(0, 2 * np.pi, 256)
    d = ep.wavelet_denoise(np.sin(t), levels=2)
    assert np.abs(d["denoised"] - np.sin(t)[:d["n_used"]]).max() < 0.05
    # Detayların TAMAMI eşiğin altında kaldı → çıktı 2. seviye YAKLAŞIMdır.
    assert d["kept_coeffs"] == 0


def test_dalgacik_hard_ve_soft_ancak_KATSAYI_KALIRSA_ayrisir():
    """Eşiği aşan katsayı yoksa iki kip ÖZDEŞTİR — ayrışma ancak `kept_coeffs>0` ile.

    Bu, `kept_coeffs`in neden çıktıda olduğunun kanıtı: onu okumadan iki kipin
    "aynı sonucu verdiği" gözlemi bir kusur sanılırdı.
    """
    rng = np.random.default_rng(2)
    # (a) Düşük gürültü + BÜYÜK basamak → bazı katsayılar eşiği aşar.
    # Basamak 100'e konur, 128'e DEĞİL: 128 her seviyede bir ÇİFT SINIRIDIR ve
    # basamağı hiçbir detay katsayısı görmez (ölçüldü — kept_coeffs 0 çıkıyordu).
    x = np.r_[np.zeros(100), np.full(156, 20.0)] + rng.normal(0, 0.1, 256)
    s = ep.wavelet_denoise(x, levels=3, mode="soft")
    h = ep.wavelet_denoise(x, levels=3, mode="hard")
    assert s["kept_coeffs"] > 0
    assert not np.allclose(s["denoised"], h["denoised"])
    assert float(np.var(s["residual"])) > float(np.var(h["residual"]))
    # (b) DEĞİŞMEZ: kalan katsayı sayısı SIFIRSA iki kip ÖZDEŞ, değilse AYRIŞIK.
    # Gürültü düzeyi taranır; iddia her düzeyde `kept_coeffs`ten TÜRETİLİR.
    for sd in (0.05, 0.2, 0.4, 0.8):
        y = np.sin(np.linspace(0, 8 * np.pi, 256)) + np.random.default_rng(11).normal(0, sd, 256)
        s2 = ep.wavelet_denoise(y, levels=3, mode="soft")
        h2 = ep.wavelet_denoise(y, levels=3, mode="hard")
        ayni = np.allclose(s2["denoised"], h2["denoised"])
        assert ayni == (s2["kept_coeffs"] == 0), f"sd={sd} kept={s2['kept_coeffs']}"


def test_dalgacik_kirpma_raporlanir_ve_gecersiz_kip_none():
    d = ep.wavelet_denoise(np.arange(70.0), levels=3)
    assert d["n"] == 70 and d["n_used"] == 64      # 8'in katına kırpıldı, gizlenmedi
    assert ep.wavelet_denoise(np.arange(70.0), mode="orta") is None


# ---------------------------------------------------------------------------
# 8) Madde 101 — Diebold-Yılmaz taşma
# ---------------------------------------------------------------------------
def test_tasma_yonu_dogru_ve_satirlar_bire_normalize():
    """B[i]=0.8·A[i−1] → A NET İHRACATÇI, B net ithalatçı, C nötr."""
    rng = np.random.default_rng(0)
    T = 800
    a = rng.normal(0, 1, T)
    c = rng.normal(0, 1, T)
    b = np.zeros(T)
    for i in range(1, T):
        b[i] = 0.8 * a[i - 1] + 0.2 * rng.normal()
    d = ep.spillover_index(np.column_stack([a, b, c]), horizon=10, lags=2)
    assert np.allclose(d["table"].sum(axis=1), 1.0)
    assert d["net"][0] > 20.0 and d["net"][1] < -20.0
    assert abs(d["net"][2]) < 5.0
    assert 0.0 < d["total"] < 100.0


def test_tasma_bagimsiz_seride_dusuk():
    """NEGATİF KONTROL: bağımsız varlıklarda toplam taşma küçük olmalı."""
    rng = np.random.default_rng(5)
    d = ep.spillover_index(rng.normal(0, 1, (800, 4)), horizon=10, lags=2)
    assert d["total"] < 15.0


def test_tasma_net_toplami_sifirdir():
    """Kimlik: Σ net = 0 (biri ihraç ediyorsa başkası ithal ediyordur)."""
    rng = np.random.default_rng(6)
    d = ep.spillover_index(rng.normal(0, 1, (600, 3)), horizon=8, lags=1)
    assert float(np.sum(d["net"])) == pytest.approx(0.0, abs=1e-9)


def test_tasma_gecersiz_girdide_none():
    assert ep.spillover_index(np.zeros((10, 2))) is None          # T çok kısa
    assert ep.spillover_index(np.random.default_rng(0).normal(0, 1, (100, 2)),
                              lags=0) is None
