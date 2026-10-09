"""nihai.ai.knowledge testleri — kavram sayısı, formül köprüsü doğrulaması, arama/sözlük/açıklama.

Ağ/dosya YOK: knowledge.py saf statik bir tablo + sorgu API'sidir. formul_ref'lerin gerçekten
nihai.ai.quant_math'te var olduğu getattr ile doğrulanır (uydurma fonksiyon adı koruması).
"""
from __future__ import annotations

import re

import pytest

from nihai.ai import knowledge, quant_math, selfcheck

# Spec §5 zorunlu kavram listesi (docs/AI_V2_BEYIN_SPEC.md) — hepsi CONCEPTS'te olmalı.
ZORUNLU_KAVRAMLAR = [
    "rejim", "momentum", "mean_reversion", "pead", "earnings_drift", "survivorship_bias",
    "look_ahead_bias", "exec_lag", "bid_ask_bounce", "kelly", "var", "cvar", "sharpe",
    "sortino", "deflated_sharpe", "pbo", "purged_cv", "embargo", "meta_labeling",
    "kointegrasyon", "hurst", "ou_process", "half_life", "parkinson_vol", "garman_klass_vol",
    "yang_zhang_vol", "ewma_vol", "kalman_filter", "ledoit_wolf", "pca_risk", "monte_carlo",
    "block_bootstrap", "black_scholes", "delta", "gamma", "vega", "theta", "implied_vol",
    "put_call_parity", "max_drawdown", "position_sizing", "drift_detection", "brier",
    "beta_binomial", "carry",
]


def test_en_az_40_kavram():
    assert len(knowledge.CONCEPTS) >= 40


def test_zorunlu_kavramlar_hepsi_mevcut():
    eksik = [k for k in ZORUNLU_KAVRAMLAR if k not in knowledge.CONCEPTS]
    assert eksik == [], f"spec §5 zorunlu kavramlar eksik: {eksik}"


def test_concept_id_dict_anahtariyla_tutarli():
    for key, c in knowledge.CONCEPTS.items():
        assert c.id == key


def test_tum_formul_ref_formul_modullerinde_var():
    """Her formul_ref stringinin FORMÜL MODÜLLERİNDEN birinde (quant_math/options_math/
    microstructure) GERÇEKTEN çağrılabilir bir fonksiyon olduğunu doğrular (uydurma ad
    koruması). Tur-32: köprü üç modüle genişledi — kavramın işaret ettiği uygulama gerçekten
    var olmalı, hangi dosyada olduğu değişebilir."""
    moduller = selfcheck._FORMULA_MODULES
    kontrol_edilen = 0
    for c in knowledge.CONCEPTS.values():
        for fn_name in c.formul_ref:
            hedef = next((getattr(m, fn_name) for m in moduller if hasattr(m, fn_name)), None)
            assert hedef is not None, f"{c.id}.formul_ref → hiçbir formül modülünde {fn_name} yok"
            assert callable(hedef), f"{fn_name} çağrılabilir değil"
            kontrol_edilen += 1
    assert kontrol_edilen > 0  # en az bir formül referansı gerçekten test edilmiş olmalı


def test_iliskili_referanslar_gecerli_kavramlara_isaret_eder():
    """iliskili alanındaki her id, CONCEPTS sözlüğünde gerçekten var olmalı (kırık graf yok)."""
    for c in knowledge.CONCEPTS.values():
        for rel_id in c.iliskili:
            assert rel_id in knowledge.CONCEPTS, f"{c.id}.iliskili → '{rel_id}' CONCEPTS'te yok"


def test_ders_ref_format_aci_n():
    """ders_ref yalnızca CLAUDE.md ACI DERSLER listesine işaret eder (ör. 'ACI-1').

    Tur-35: desen 1-7 idi ve BAYATTI — CLAUDE.md'de ders #8 (sınıf dengesizliği DSR'yi
    şişirir) ve #9 (kalibrasyon kazancı SABİT tahminciyle kıyaslanmalı) 2026-07/08'de
    eklenmişti. Aralık belgeye göre 1-9'a genişletildi (kural gevşetilmedi; belgeyle
    yeniden hizalandı)."""
    desen = re.compile(r"^ACI-[1-9]$")
    for c in knowledge.CONCEPTS.values():
        for ref in c.ders_ref:
            assert desen.match(ref), f"{c.id}.ders_ref beklenmeyen biçim: {ref!r}"


def test_glossary_alfabetik_ve_tum_id_leri_icerir():
    g = knowledge.glossary()
    assert g == sorted(g)
    assert set(g) == set(knowledge.CONCEPTS.keys())


def test_search_bulur_ve_kucuk_harf_duyarsiz():
    sonuc = knowledge.search("KELLY")
    ids = [c.id for c in sonuc]
    assert "kelly" in ids


def test_search_tanim_icinde_de_arar():
    # "kazanma" kelimesi kelly kavramının tanımında geçmez ama 'oynaklık' pek çok tanımda geçer
    sonuc = knowledge.search("oynaklık")
    assert len(sonuc) >= 3


def test_search_bos_sorgu_bos_liste():
    assert knowledge.search("") == []
    assert knowledge.search("   ") == []


def test_search_bulunamayan_sorgu_bos_liste():
    assert knowledge.search("zzz_boyle_bir_kavram_yok_xyzzy") == []


def test_explain_bilinen_kavram_tum_bolumleri_icerir():
    metin = knowledge.explain("sharpe")
    assert metin is not None
    assert "Tanım:" in metin
    assert "Formül köprüsü" in metin
    assert "sharpe" in metin  # formül adı geçmeli
    assert "Tuzaklar:" in metin
    assert "Ders bağları:" in metin
    assert "İlişkili kavramlar:" in metin


def test_explain_formulu_olmayan_kavram_yok_notu_verir():
    metin = knowledge.explain("survivorship_bias")
    assert metin is not None
    assert "kavramsal/metodolojik" in metin


def test_explain_bilinmeyen_kavram_none():
    assert knowledge.explain("bu_kavram_yok") is None


def test_formula_for_bilinen_kavram():
    assert knowledge.formula_for("kelly") == ["kelly_binary", "kelly_continuous", "fractional_kelly"]


def test_formula_for_bilinmeyen_kavram_bos_liste():
    assert knowledge.formula_for("bu_kavram_yok") == []


def test_formula_for_kopya_liste_dondurur_kaynagi_bozmaz():
    liste = knowledge.formula_for("kelly")
    liste.append("uydurma_fonksiyon")
    assert "uydurma_fonksiyon" not in knowledge.CONCEPTS["kelly"].formul_ref


@pytest.mark.parametrize("concept_id", ZORUNLU_KAVRAMLAR)
def test_her_zorunlu_kavram_explain_ile_patlamiyor(concept_id):
    """Regresyon: her zorunlu kavram için explain() None dönmemeli ve çökmemeli."""
    assert knowledge.explain(concept_id) is not None


# ===========================================================================
# v3 (Tur-16 WP-K) testleri — +58 kavram (toplam >=100), `prereqs` alanı,
# `curriculum()` topolojik sıralama + çevrim tespiti.
# ===========================================================================

# 16 quant_math v3 fonksiyonu — knowledge.py'de en az bir kavramda köprülenmiş olmalı.
V3_QUANT_MATH_FONKSIYONLARI = [
    "garch11_fit", "garch11_forecast", "evt_hill_tail_index", "evt_gpd_var_es",
    "copula_fit_gaussian", "copula_fit_t", "tail_dependence", "frac_diff_ffd",
    "welch_psd", "haar_dwt_vol", "almgren_chriss_cost", "permutation_entropy",
    "sample_entropy", "mutual_info", "coskewness", "cokurtosis",
]


def test_toplam_kavram_sayisi_en_az_100():
    assert len(knowledge.CONCEPTS) >= 100


def test_v3_en_az_55_yeni_kavram_eklendi():
    yeni_sayisi = len(knowledge.CONCEPTS) - len(ZORUNLU_KAVRAMLAR)
    assert yeni_sayisi >= 55


def test_eski_45_kavram_degismedi_spot_kontrol():
    """kelly kavramının id/ad/tanim/formul_ref AYNEN korunmuş olmalı (regresyon guard)."""
    kelly = knowledge.CONCEPTS["kelly"]
    assert kelly.ad == "Kelly Kriteri"
    assert kelly.tanim == (
        "Uzun-vadeli logaritmik servet büyümesini maksimize eden optimal bahis/pozisyon "
        "oranı."
    )
    assert kelly.formul_ref == ["kelly_binary", "kelly_continuous", "fractional_kelly"]


def test_prereqs_alani_eski_kavramlarda_varsayilan_bos():
    for concept_id in ZORUNLU_KAVRAMLAR:
        assert knowledge.CONCEPTS[concept_id].prereqs == []


def test_prereqs_alani_yeni_kavramlarda_gecerli_id_isaret_eder():
    for c in knowledge.CONCEPTS.values():
        for p in c.prereqs:
            assert p in knowledge.CONCEPTS, f"{c.id}.prereqs -> '{p}' CONCEPTS'te yok"


def test_v3_quant_math_fonksiyonlari_en_az_bir_kavramda_referansli():
    tum_formul_refler: set[str] = set()
    for c in knowledge.CONCEPTS.values():
        tum_formul_refler.update(c.formul_ref)
    eksik = [fn for fn in V3_QUANT_MATH_FONKSIYONLARI if fn not in tum_formul_refler]
    assert eksik == [], f"knowledge.py'de köprülenmemiş v3 fonksiyonlar: {eksik}"


def test_garch_kavrami_dogru_fonksiyonlara_koprulu():
    assert knowledge.CONCEPTS["garch"].formul_ref == ["garch11_fit", "garch11_forecast"]


def test_evt_kavramlari_dogru_fonksiyonlara_koprulu():
    assert knowledge.CONCEPTS["hill_estimator"].formul_ref == ["evt_hill_tail_index"]
    assert knowledge.CONCEPTS["gpd_pot"].formul_ref == ["evt_gpd_var_es"]


def test_kopula_kavramlari_dogru_fonksiyonlara_koprulu():
    assert knowledge.CONCEPTS["gaussian_copula"].formul_ref == ["copula_fit_gaussian"]
    assert knowledge.CONCEPTS["student_t_copula"].formul_ref == ["copula_fit_t"]
    assert knowledge.CONCEPTS["kuyruk_bagimliligi"].formul_ref == ["tail_dependence"]


def test_entropi_kavramlari_dogru_fonksiyonlara_koprulu():
    assert knowledge.CONCEPTS["permutasyon_entropisi"].formul_ref == ["permutation_entropy"]
    assert knowledge.CONCEPTS["ornek_entropisi"].formul_ref == ["sample_entropy"]
    assert knowledge.CONCEPTS["karsilikli_bilgi"].formul_ref == ["mutual_info"]


def test_coskewness_cokurtosis_kavramlari_dogru_fonksiyonlara_koprulu():
    assert knowledge.CONCEPTS["coskewness"].formul_ref == ["coskewness"]
    assert knowledge.CONCEPTS["cokurtosis"].formul_ref == ["cokurtosis"]


def test_tum_kavramlarda_ad_ve_tanim_bos_degil():
    for c in knowledge.CONCEPTS.values():
        assert c.ad.strip() != "", f"{c.id}: ad boş"
        assert c.tanim.strip() != "", f"{c.id}: tanim boş"


def test_yeni_kavramlarda_tuzaklar_bos_degil():
    """v3 kavramları mevcut-45'in üslup/derinliğinde olmalı: her biri en az bir tuzak içerir."""
    for concept_id in knowledge.CONCEPTS:
        if concept_id in ZORUNLU_KAVRAMLAR:
            continue
        c = knowledge.CONCEPTS[concept_id]
        assert len(c.tuzaklar) >= 1, f"{c.id}: tuzaklar boş (v3 kavramı derinlik şartını karşılamıyor)"


# ---------------------------------------------------------------------------
# curriculum()
# ---------------------------------------------------------------------------
def test_curriculum_uzunluk_kavram_sayisiyla_esit():
    order = knowledge.curriculum()
    assert len(order) == len(knowledge.CONCEPTS)
    assert set(order) == set(knowledge.CONCEPTS.keys())   # tekrar/eksik yok


def test_curriculum_topolojik_tutarlilik():
    """Her kavram, TÜM prereqs'lerinden SONRA gelmeli (indeks karşılaştırması)."""
    order = knowledge.curriculum()
    pos = {cid: i for i, cid in enumerate(order)}
    for c in knowledge.CONCEPTS.values():
        for p in c.prereqs:
            assert pos[p] < pos[c.id], f"{c.id} önkoşulu {p}'den ÖNCE geliyor"


def test_curriculum_deterministik():
    assert knowledge.curriculum() == knowledge.curriculum()


def test_curriculum_ilk_katman_alfabetik_sirali():
    """Önkoşulu olmayan (kök) kavramlar ilk katmanı oluşturur ve alfabetik sıralanır
    (eşit-öncelikli düğümler arası deterministik tekrarlanabilirlik için)."""
    order = knowledge.curriculum()
    roots = sorted(cid for cid, c in knowledge.CONCEPTS.items() if not c.prereqs)
    assert order[: len(roots)] == roots


def test_curriculum_cevrim_value_error(monkeypatch):
    cyclic = {
        "a": knowledge.Concept(id="a", ad="A", tanim="t", prereqs=["b"]),
        "b": knowledge.Concept(id="b", ad="B", tanim="t", prereqs=["a"]),
    }
    monkeypatch.setattr(knowledge, "CONCEPTS", cyclic)
    with pytest.raises(ValueError):
        knowledge.curriculum()


def test_curriculum_kendine_donen_cevrim_value_error(monkeypatch):
    kendine = {"a": knowledge.Concept(id="a", ad="A", tanim="t", prereqs=["a"])}
    monkeypatch.setattr(knowledge, "CONCEPTS", kendine)
    with pytest.raises(ValueError):
        knowledge.curriculum()


def test_curriculum_bilinmeyen_prereq_sessizce_yoksayilir(monkeypatch):
    veri = {
        "a": knowledge.Concept(id="a", ad="A", tanim="t", prereqs=["yok_boyle_bir_id"]),
        "b": knowledge.Concept(id="b", ad="B", tanim="t", prereqs=["a"]),
    }
    monkeypatch.setattr(knowledge, "CONCEPTS", veri)
    assert knowledge.curriculum() == ["a", "b"]


def test_curriculum_bos_concepts_bos_liste(monkeypatch):
    monkeypatch.setattr(knowledge, "CONCEPTS", {})
    assert knowledge.curriculum() == []


# ---------------------------------------------------------------------------
# Sorgu API'sinin yeni kavramlarla uyumu
# ---------------------------------------------------------------------------
def test_search_yeni_kavrami_bulur():
    sonuc = knowledge.search("GARCH")
    ids = [c.id for c in sonuc]
    assert "garch" in ids


def test_explain_yeni_kavram_tum_bolumleri_icerir():
    metin = knowledge.explain("gpd_pot")
    assert metin is not None
    assert "Tanım:" in metin
    assert "evt_gpd_var_es" in metin
    assert "Tuzaklar:" in metin
    assert "Ders bağları:" in metin


def test_formula_for_yeni_kavram():
    assert knowledge.formula_for("almgren_chriss") == ["almgren_chriss_cost"]
    assert knowledge.formula_for("bu_kavram_yok_yeni") == []


def test_glossary_yeni_idleri_icerir():
    g = knowledge.glossary()
    for yeni_id in ("garch", "gpd_pot", "coskewness", "cokurtosis", "backtest_tuzaklari"):
        assert yeni_id in g


def test_backtest_tuzaklari_ders_ref_ve_prereqs_zengin():
    """Yüksek-yoğunluklu bir v3 kavramının (üç ACI dersini birleştiren) alan bütünlüğü."""
    c = knowledge.CONCEPTS["backtest_tuzaklari"]
    assert set(c.ders_ref) == {"ACI-1", "ACI-3", "ACI-5"}
    assert set(c.prereqs) == {"overfitting", "data_leakage", "survivorship_bias"}
    assert len(c.tuzaklar) >= 1
