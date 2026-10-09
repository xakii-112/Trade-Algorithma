"""nihai.ai.volume_profile testleri — hacim/piyasa profili (Tur-75).

Doğrulama stratejisi `test_ai_microstructure` ile aynı: her fonksiyon KAPALI FORMLA
bilinen bir kurguya karşı sınanır (elle hesaplanabilir histogram, tek FVG'li seri,
tam süpürme barı), sonra dejenere girdilerle (düz seri, sıfır hacim, tek bar)
sözleşme denetlenir. Ağ/IO yok, `default_rng` seed'li.
"""
from __future__ import annotations

import numpy as np
import pytest

from nihai.ai import volume_profile as vp


# ---------------------------------------------------------------------------
# 1) Madde 069 — hacim profili
# ---------------------------------------------------------------------------
def test_profil_hacmi_kesisimle_dagitir_toplam_korunur():
    """Bar hacmi kutulara kesişim uzunluğuyla bölünür; TOPLAM hacim korunmalı."""
    d = vp.volume_profile(high=[10.0, 12.0], low=[8.0, 10.0], volume=[100.0, 200.0], bins=4)
    assert d["total_volume"] == pytest.approx(300.0)
    assert d["hist"].sum() == pytest.approx(300.0)
    # Aralık [8,12] → kutular [8,9),[9,10),[10,11),[11,12]. Bar1 [8,10] iki kutuya
    # eşit (50/50), bar2 [10,12] iki kutuya eşit (100/100) dağılır.
    assert list(np.round(d["hist"], 6)) == [50.0, 50.0, 100.0, 100.0]


def test_poc_en_cok_hacmin_gectigi_kutudur():
    """Dar bir barda yığılan hacim POC'yi oraya çeker."""
    d = vp.volume_profile(high=[10.0, 10.1, 20.0], low=[9.9, 10.0, 10.0],
                          volume=[1000.0, 1000.0, 10.0], bins=10)
    assert 9.9 <= d["poc"] <= 10.6


def test_deger_alani_hedefi_kapsar_ve_poc_yi_icerir():
    rng = np.random.default_rng(11)
    mid = 100.0 + rng.normal(0, 1.0, 400)
    d = vp.volume_profile(high=mid + 0.2, low=mid - 0.2,
                          volume=rng.uniform(50, 150, 400), bins=40)
    assert d["value_area_share"] >= 0.70
    assert d["val"] <= d["poc"] <= d["vah"]


def test_deger_alani_beraberlikte_deterministiktir():
    """Simetrik histogramda genişletme AŞAĞI yönü seçer (belgelenmiş konvansiyon)."""
    # Tek bir bar tüm aralığı eşit kaplar → histogram DÜZ, her komşu beraberdir.
    d = vp.volume_profile(high=[10.0], low=[0.0], volume=[100.0], bins=10)
    d2 = vp.volume_profile(high=[10.0], low=[0.0], volume=[100.0], bins=10)
    assert (d["val"], d["vah"]) == (d2["val"], d2["vah"])
    assert d["val"] < d["poc"], "beraberlikte aşağı genişlemedi"


def test_profil_dejenere_girdide_none():
    assert vp.volume_profile([10.0, 10.0], [10.0, 10.0], [5.0, 5.0]) is None   # h==l
    assert vp.volume_profile([10.0, 11.0], [9.0, 10.0], [0.0, 0.0]) is None    # hacim 0
    assert vp.volume_profile([10.0, 11.0], [9.0, np.nan], [1.0, 1.0]) is None  # NaN
    assert vp.volume_profile([10.0, 9.0], [11.0, 10.0], [1.0, 1.0]) is None    # h < l


# ---------------------------------------------------------------------------
# 2) Madde 070 — CVD
# ---------------------------------------------------------------------------
def test_cvd_yukselen_seride_pozitif_egimli():
    p = np.linspace(100.0, 110.0, 60)
    d = vp.cumulative_volume_delta(p, np.full(60, 1000.0))
    assert d["slope"] > 0 and d["last"] > 0


def test_cvd_uyumsuzlugu_uc_durumludur():
    """Fiyat yeni zirve yaparken CVD yapamıyorsa −1; eşitlik/nötr durumda 0."""
    # Fiyat sona doğru yeni zirve, ama son barların hacmi ÇOK küçük → CVD tepe yapamaz.
    p = np.array([100, 101, 102, 103, 104, 103.5, 104.5], dtype=float)
    v = np.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1.0])
    assert vp.cumulative_volume_delta(p, v)["divergence"] == -1
    # Düz olmayan ama zirve/dip kırmayan seri → 0
    p2 = np.array([100, 102, 101, 103, 102, 103, 102.5], dtype=float)
    assert vp.cumulative_volume_delta(p2, np.full(7, 1000.0))["divergence"] == 0


def test_cvd_duz_seride_none():
    """σ_Δp = 0 → BVC None → CVD None (sıfır bilgi 'nötr akış' diye raporlanmaz)."""
    assert vp.cumulative_volume_delta([100.0] * 10, [1000.0] * 10) is None


# ---------------------------------------------------------------------------
# 3) Madde 078/084 — çıpalanmış VWAP + bantlar
# ---------------------------------------------------------------------------
def test_avwap_sabit_fiyatta_fiyatin_kendisi_sigma_sifir():
    d = vp.anchored_vwap([10.0] * 5, [10.0] * 5, [10.0] * 5, [100.0] * 5)
    assert d["last_vwap"] == pytest.approx(10.0)
    assert d["sigma"][-1] == pytest.approx(0.0, abs=1e-12)
    assert d["last_z"] is None, "σ=0'da z ölçülemez, uydurulmamalı"


def test_avwap_hacim_agirlikli_kapali_form():
    """İki bar: tp=10 (v=100) ve tp=20 (v=300) → VWAP = (1000+6000)/400 = 17,5."""
    d = vp.anchored_vwap(high=[10.0, 20.0], low=[10.0, 20.0], close=[10.0, 20.0],
                         volume=[100.0, 300.0])
    assert d["last_vwap"] == pytest.approx(17.5)
    # Hacim-ağırlıklı varyans = E[p²]−E[p]² = (100·100+300·400)/400 − 17,5² = 18,75
    assert d["sigma"][-1] == pytest.approx(np.sqrt(18.75))


def test_avwap_negatif_capa_sondan_sayar():
    h = np.arange(1.0, 21.0)
    d = vp.anchored_vwap(h, h, h, np.full(20, 10.0), anchor=-5)
    assert d["anchor"] == 15 and d["n"] == 5


def test_avwap_bantlari_simetrik_ve_sirali():
    rng = np.random.default_rng(5)
    c = 100 + np.cumsum(rng.normal(0, 0.5, 100))
    d = vp.anchored_vwap(c + 0.3, c - 0.3, c, rng.uniform(10, 100, 100))
    b1, b3 = d["bands"][1.0], d["bands"][3.0]
    assert np.all(b3["upper"][-10:] >= b1["upper"][-10:])
    assert np.allclose(b1["upper"] - d["vwap"], d["vwap"] - b1["lower"])


def test_avwap_gecersiz_capa_none():
    assert vp.anchored_vwap([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 1.0], anchor=1) is None
    assert vp.anchored_vwap([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [0.0, 0.0]) is None


# ---------------------------------------------------------------------------
# 4) Madde 080 — TPO
# ---------------------------------------------------------------------------
def test_tpo_tek_baski_hizli_gecisi_isaretler():
    """Fiyat bir bölgeden tek dilimde geçerse orası tek baskıdır."""
    # 5 bar aşağıda, 1 bar hızlı geçiş, 5 bar yukarıda → orta bölge 1 dilim görür.
    h = np.array([10.1] * 5 + [20.0] + [20.1] * 5)
    lo = np.array([10.0] * 5 + [10.1] + [19.9] * 5)
    d = vp.tpo_profile(h, lo, bars_per_bracket=1, bins=20)
    assert d["n_brackets"] == 11
    assert len(d["single_prints_interior"]) > 0
    # Orta bölge (12-19 civarı) tek baskı olmalı; uçlar yoğun.
    assert any(12.0 < x < 19.0 for x in d["single_prints_interior"])


def test_tpo_dilimleme_bar_sayisini_bolerek_calisir():
    h = np.arange(1.0, 13.0)
    d = vp.tpo_profile(h, h - 0.5, bars_per_bracket=4, bins=10)
    assert d["n_brackets"] == 3


def test_tpo_kenar_tek_baskilari_ic_listeden_dusulur():
    h = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    d = vp.tpo_profile(h, h - 0.1, bars_per_bracket=1, bins=10)
    assert len(d["single_prints_interior"]) <= len(d["single_prints"])


# ---------------------------------------------------------------------------
# 5) Madde 085 — emilim
# ---------------------------------------------------------------------------
def test_emilim_yuksek_hacim_dar_barda_pozitif():
    rng = np.random.default_rng(3)
    n = 60
    h = 100 + rng.normal(0, 0.5, n)
    lo = h - rng.uniform(0.8, 1.2, n)
    v = rng.uniform(900, 1100, n)
    lo[-1] = h[-1] - 0.05          # ÇOK dar bar
    v[-1] = 5000.0                 # ÇOK yüksek hacim
    assert vp.absorption_score(h, lo, v, window=20)["last"] > 2.0


def test_emilim_dusuk_hacim_genis_barda_negatif():
    rng = np.random.default_rng(4)
    n = 60
    h = 100 + rng.normal(0, 0.5, n)
    lo = h - rng.uniform(0.8, 1.2, n)
    v = rng.uniform(900, 1100, n)
    lo[-1] = h[-1] - 5.0
    v[-1] = 50.0
    assert vp.absorption_score(h, lo, v, window=20)["last"] < -2.0


def test_emilim_duz_seride_olcum_yoklugu_nan():
    """σ=0 → skor NaN; 0 basmak 'nötr emilim' demek olurdu."""
    d = vp.absorption_score([10.0] * 30, [9.0] * 30, [100.0] * 30, window=10)
    assert d["last"] is None and d["n_measured"] == 0


def test_emilim_pencere_dolmadan_uretmez():
    rng = np.random.default_rng(9)
    n = 40
    h = 100 + rng.normal(0, 1, n)
    lo = h - rng.uniform(0.5, 1.5, n)            # bar boyu DEĞİŞMELİ, yoksa σ_range=0
    d = vp.absorption_score(h, lo, rng.uniform(10, 100, n), window=20)
    assert np.isnan(d["score"][:19]).all() and np.isfinite(d["score"][19])


# ---------------------------------------------------------------------------
# 6) Madde 074 — likidite süpürme
# ---------------------------------------------------------------------------
def test_supurme_deler_ve_iceri_kapanirsa_isaretlenir():
    h = np.array([10.0] * 5 + [12.0])
    lo = np.array([9.0] * 5 + [9.5])
    c = np.array([9.5] * 5 + [9.8])          # eşiğin (10,0) ALTINDA kapandı
    d = vp.liquidity_sweep(h, lo, c, lookback=5)
    assert d["sweep_high"][-1] == 1 and d["last"] == 1


def test_supurme_disarida_kapanirsa_isaretlenmez():
    h = np.array([10.0] * 5 + [12.0])
    lo = np.array([9.0] * 5 + [9.5])
    c = np.array([9.5] * 5 + [11.5])         # kırılım DOĞRULANDI → süpürme değil
    assert vp.liquidity_sweep(h, lo, c, lookback=5)["sweep_high"][-1] == 0


def test_supurme_penceresi_bari_icermez():
    """Pencere barın kendisini içerseydi koşul asla sağlanamazdı."""
    h = np.array([10.0] * 5 + [12.0])
    d = vp.liquidity_sweep(h, h - 1.0, np.full(6, 9.5), lookback=5)
    assert d["n_sweep_high"] == 1


def test_supurme_isinma_barlari_sifir():
    h = np.arange(1.0, 11.0)
    d = vp.liquidity_sweep(h, h - 0.5, h - 0.4, lookback=5)
    assert (d["sweep_high"][:5] == 0).all()


# ---------------------------------------------------------------------------
# 7) Madde 105 — FVG
# ---------------------------------------------------------------------------
def test_fvg_tek_yukselis_boslugu_bulunur():
    """low[2]=12 > high[0]=10 → [10,12] boşluğu."""
    d = vp.fair_value_gaps(high=[10.0, 15.0, 16.0], low=[9.0, 11.0, 12.0])
    assert d["n_gaps"] == 1
    g = d["gaps"][0]
    assert g["direction"] == 1 and g["lower"] == 10.0 and g["upper"] == 12.0
    assert g["filled"] is False


def test_fvg_doldurma_yalniz_SONRAKI_barlara_bakar():
    """Boşluğun kendi barı doldurma sayılsaydı her boşluk anında dolardı."""
    h = [10.0, 15.0, 16.0, 13.0]
    lo = [9.0, 11.0, 12.0, 10.5]      # 4. bar [10,12] aralığına giriyor
    d = vp.fair_value_gaps(h, lo)
    assert d["gaps"][0]["filled"] is True and d["n_open"] == 0
    assert d["fill_rate"] == pytest.approx(1.0)


def test_fvg_dusus_boslugu_ve_min_size():
    d = vp.fair_value_gaps(high=[16.0, 12.0, 10.0], low=[15.0, 11.0, 9.0])
    assert d["gaps"][0]["direction"] == -1 and d["gaps"][0]["size"] == pytest.approx(5.0)
    assert vp.fair_value_gaps([16.0, 12.0, 10.0], [15.0, 11.0, 9.0],
                              min_size=6.0)["n_gaps"] == 0


def test_fvg_bosluk_yoksa_bos_liste():
    h = [10.0, 10.5, 11.0, 11.5]
    d = vp.fair_value_gaps(h, [x - 2.0 for x in h])
    assert d["n_gaps"] == 0 and d["fill_rate"] is None


# ---------------------------------------------------------------------------
# 8) Madde 083 — bar-içi momentler
# ---------------------------------------------------------------------------
def test_momentler_normal_orneklemde_beklenene_yakin():
    rng = np.random.default_rng(2)
    d = vp.intrabar_moments(rng.normal(0, 0.01, 20000))
    assert abs(d["skew"]) < 0.1
    assert abs(d["excess_kurtosis"]) < 0.2


def test_momentler_sola_carpik_seride_negatif_skew():
    r = np.r_[np.full(99, 0.001), [-0.20]]      # tek büyük negatif = sol kuyruk
    d = vp.intrabar_moments(r)
    assert d["skew"] < -5.0 and d["excess_kurtosis"] > 50.0


def test_momentler_duz_seride_none():
    assert vp.intrabar_moments([0.0] * 50) is None
    assert vp.intrabar_moments([0.1, 0.2]) is None       # n < 4


def test_kurtosis_ile_excess_arasinda_tam_uc_fark_var():
    rng = np.random.default_rng(6)
    d = vp.intrabar_moments(rng.normal(0, 1, 500))
    assert d["kurtosis"] - d["excess_kurtosis"] == pytest.approx(3.0)
