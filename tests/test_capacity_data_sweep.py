"""capacity_data_sweep._verdikt testleri — P2-11 hükmünün mantığı.

Bu hüküm "daha çok veri toplayalım mı yoksa sinyal ailesini mi değiştirelim" sorusunu
yanıtlıyor; yanlış olursa tüm P2-11 sonucu saptırır.

⚠️ Bu dosyanın ilk sürümü BOZUK bir hükmü doğruluyordu: hüküm yalnız log-doğrusal eğime
bakıyordu ve 5 noktalı eğride EN KÜÇÜK n'nin gürültüsü (hata payı orada en büyük) doğruyu
tek başına döndürüyordu. Gerçek koşularda:
  * 97k PIT  → ilk nokta yüksekti  → "küçük n şanstı"   (ilk nokta hariç eğim -0.0015 = DÜZ)
  * 234k     → ilk nokta düşüktü   → "veri sınırlı"     (ilk nokta hariç eğim -0.0116 = İŞARET DÖNDÜ)
…oysa ikisinde de DSR her n'de ≈ 0 idi. Etiket raporun geri kalanıyla ÇELİŞİYORDU = ACI
DERS #8'in aynısı. Düzeltilmiş sıra: ÖNCE kanıt kapısı (DSR), SONRA eğim; eğim iki kez
hesaplanır (tüm noktalar + ilk nokta hariç) ve uyuşmazlarsa hüküm VERİLMEZ.
"""
from __future__ import annotations

from research.capacity_data_sweep import _verdikt


def _nokta(n, auc, dsr=0.0):
    return {"n": n, "auc_mean": auc, "auc_std": 0.01, "dsr": dsr}


# ---------------------------------------------------------------------------
# 1) KANIT KAPISI ÖNCE — baskın gerçek
# ---------------------------------------------------------------------------
def test_dsr_hicbir_nde_gecmiyorsa_kenar_yok():
    """ASIL ÖLÇÜLEN VAKA (97k + 234k): AUC ne yaparsa yapsın, DSR≈0 ise hüküm budur."""
    v = _verdikt([_nokta(12_000, 0.494), _nokta(25_000, 0.539, 0.0249),
                  _nokta(53_000, 0.530, 0.0032), _nokta(111_000, 0.534),
                  _nokta(234_000, 0.525)])
    assert v["label"] == "KENAR_YOK_HER_N"
    assert v["dsr_max"] == 0.0249
    assert "sinyalin kendisi" in v["text"]


def test_dsr_kapisi_yukselen_auc_egrisini_de_susturur():
    """Yükselen AUC eğrisi bile DSR olmadan 'veri işe yarıyor' DEMEZ (ACI DERS #8)."""
    v = _verdikt([_nokta(10_000, 0.505), _nokta(30_000, 0.530),
                  _nokta(100_000, 0.560), _nokta(300_000, 0.590)])
    assert v["label"] == "KENAR_YOK_HER_N"


def test_dsr_yoksa_da_kenar_yok():
    v = _verdikt([{"n": 10_000, "auc_mean": 0.52}, {"n": 50_000, "auc_mean": 0.53},
                  {"n": 200_000, "auc_mean": 0.54}])
    assert v["label"] == "KENAR_YOK_HER_N"
    assert v["dsr_max"] is None


# ---------------------------------------------------------------------------
# 2) DSR geçtiğinde eğim — ama tek uç noktaya bağımlıysa hüküm YOK
# ---------------------------------------------------------------------------
def test_veri_sinirli_dsr_gecerken_ve_egim_kararli():
    v = _verdikt([_nokta(10_000, 0.505, 0.30), _nokta(30_000, 0.530, 0.32),
                  _nokta(100_000, 0.560, 0.35), _nokta(300_000, 0.590, 0.40)])
    assert v["label"] == "VERI_SINIRLI"
    assert v["egim_per_dekad"] > 0.010 and v["egim_ilk_nokta_haric"] > 0.010


def test_egri_kararsiz_ilk_nokta_hukmu_ceviriyorsa():
    """234k'nın GERÇEK şekli: ilk nokta atılınca eğimin İŞARETİ dönüyor → hüküm verme."""
    v = _verdikt([_nokta(12_000, 0.494, 0.30), _nokta(25_000, 0.539, 0.32),
                  _nokta(53_000, 0.530, 0.31), _nokta(111_000, 0.534, 0.33),
                  _nokta(234_000, 0.525, 0.30)])
    assert v["label"] == "EGRI_KARARSIZ"
    assert v["egim_per_dekad"] > 0 > v["egim_ilk_nokta_haric"]
    assert "tek bir uç noktaya bağımlı" in v["text"].lower() or "TEK bir uç" in v["text"]


def test_kucuk_n_sansiydi_dsr_gecerken():
    v = _verdikt([_nokta(10_000, 0.585, 0.30), _nokta(30_000, 0.560, 0.28),
                  _nokta(100_000, 0.535, 0.26), _nokta(300_000, 0.508, 0.25)])
    assert v["label"] == "KUCUK_N_SANSIYDI"


def test_gurultu_sinirli_duz_egri_dsr_gecerken():
    v = _verdikt([_nokta(10_000, 0.551, 0.30), _nokta(30_000, 0.549, 0.31),
                  _nokta(100_000, 0.552, 0.30), _nokta(234_000, 0.550, 0.29)])
    assert v["label"] == "GURULTU_SINIRLI"


# ---------------------------------------------------------------------------
# 3) savunmacı
# ---------------------------------------------------------------------------
def test_yetersiz_nokta_hukum_vermez():
    v = _verdikt([_nokta(10_000, 0.50), _nokta(100_000, 0.60)])
    assert v["label"] == "YETERSIZ_NOKTA"
    assert "egim_per_dekad" not in v


def test_auc_none_olan_noktalar_elenir():
    v = _verdikt([_nokta(10_000, 0.50), {"n": 30_000, "auc_mean": None},
                  _nokta(100_000, 0.50), _nokta(234_000, 0.50)])
    assert v["label"] == "KENAR_YOK_HER_N"


def test_uc_noktada_kirpik_egim_hesaplanmaz():
    """3 nokta varsa ilk nokta atılınca 2 kalır — eğim anlamsız; kırpık eğim None."""
    v = _verdikt([_nokta(10_000, 0.55, 0.3), _nokta(50_000, 0.56, 0.3),
                  _nokta(200_000, 0.57, 0.3)])
    assert v["egim_ilk_nokta_haric"] is None
    assert v["label"] in ("VERI_SINIRLI", "GURULTU_SINIRLI")


def test_ozet_alanlari_dolu():
    v = _verdikt([_nokta(10_000, 0.50), _nokta(50_000, 0.51),
                  _nokta(234_000, 0.52, dsr=0.03)])
    assert v["auc_ilk"] == 0.50 and v["auc_son"] == 0.52
    assert v["auc_en_iyi"] == 0.52
    assert v["en_buyuk_n"] == 234_000 and v["en_buyuk_dsr"] == 0.03
