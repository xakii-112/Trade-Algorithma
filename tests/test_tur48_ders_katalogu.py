"""Tur-48 — ACI DERS kataloğu ile CLAUDE.md ARASINDAKİ bağ artık DENETLENİYOR.

Kusur: `episodic.LESSONS` ACI-7'de duruyordu; CLAUDE.md'de #8 (sınıf dengesizliği DSR'yi
şişirir) ve #9 (kalibrasyon kazancı sabit tahminciyle kıyaslanmalı) 2026-07/08'den beri
vardı. Sonuç: `check_against_lessons` — `agent`/`brain`/`decision`'ın okuduğu TEK ders
kapısı — projenin en çok atıf alan iki dersi için ASLA ateşlenemiyordu. Üstelik ikisi de
tam olarak KANIT KAPILARININ bozulmasını anlatıyor.

Tur-35 boşluğun yalnız BİÇİM tarafını görmüştü (`test_ders_ref_format_aci_n` desenini
1-7 → 1-9 genişletti). Desen genişledi, KATALOG genişlemedi → `knowledge.ders_ref=["ACI-9"]`
sessizce boşa işaret etti; hiçbir test bunu yakalayamazdı çünkü kontrol BİÇİMDİ, VARLIK değil.

Ders: bir çapraz referansı "gevşek" bırakmak IMPORT bağımsızlığıdır, DENETİMSİZLİK değil.
Çalışma zamanında import yok; testte VARLIK kontrolü var.
"""
from __future__ import annotations

import re
from pathlib import Path

from nihai.ai import episodic, knowledge

_CLAUDE_MD = Path(__file__).resolve().parents[1] / "CLAUDE.md"


def _claude_md_ders_numaralari() -> set[str]:
    """CLAUDE.md "⚠️ ACI DERSLER" bölümündeki numaralı liste maddeleri."""
    metin = _CLAUDE_MD.read_text(encoding="utf-8")
    bas = metin.index("## ⚠️ ACI DERSLER")
    son = metin.index("## Çalışma ilkeleri", bas)
    return set(re.findall(r"^([0-9])\. \*\*", metin[bas:son], re.MULTILINE))


def _katalog_aci_idleri() -> set[str]:
    return {les.id.split("-")[1] for les in episodic.LESSONS if les.id.startswith("ACI-")}


def test_katalog_CLAUDE_MD_ile_ayni_dersleri_tasiyor():
    """Asıl kusur buydu: belge 9 ders sayarken katalog 7'de duruyordu."""
    belge, katalog = _claude_md_ders_numaralari(), _katalog_aci_idleri()
    assert belge, "CLAUDE.md ACI DERSLER bölümü ayrıştırılamadı"
    assert katalog == belge, (
        f"katalog ↔ belge AYRIŞTI — belgede {sorted(belge)}, katalogda {sorted(katalog)}; "
        f"eksik: {sorted(belge - katalog)}, fazla: {sorted(katalog - belge)}"
    )


def test_her_ders_ref_gercek_bir_derse_isaret_eder():
    """BİÇİM değil VARLIK kontrolü — Tur-35'in kaçırdığı yarı.

    `knowledge` çalışma zamanında `episodic`i import ETMEZ (bilinçli); bağ burada,
    testte denetlenir."""
    mevcut = {les.id for les in episodic.LESSONS}
    for c in knowledge.CONCEPTS.values():
        for ref in c.ders_ref:
            assert ref in mevcut, f"{c.id}.ders_ref={ref!r} kataloğda YOK (boşa işaret)"


def test_her_dersin_anahtar_kelimesi_var():
    """Anahtarsız bir ders kataloğa girer ama `check_against_lessons`'ta ÖLÜDÜR."""
    for les in episodic.LESSONS:
        kws = episodic._LESSON_KEYWORDS.get(les.id, ())
        assert kws, f"{les.id} anahtar kelimesiz — hiçbir metinle eşleşemez"


def test_yeni_dersler_gercekten_ATESLENIYOR():
    """Katalog kaydı tek başına yetmez: kapının fiilen ateşlendiği ÖLÇÜLMELİ."""
    for metin, bekle in (
        ("DSR 1.0 çıktı ama OOS-AUC 0.50", "ACI-8"),
        ("sözde-getiri çoğunluk kuralına göre ölçülmeli", "ACI-8"),
        ("Brier düştü, kalibrasyon iyileşti", "ACI-9"),
        ("sabit taban tahmincisiyle kıyasla", "ACI-9"),
    ):
        ids = [les.id for les in episodic.check_against_lessons(metin)]
        assert bekle in ids, f"{metin!r} → {ids} ({bekle} ateşlenmedi)"


def test_anahtarlar_GOMULU_parca_olarak_eslesmiyor():
    """REGRESYON KAPISI: çıplak 'ece' (ECE metriği) 'gelecek'in İÇİNDE eşleşirdi.

    Kısa bir anahtar kapıyı gürültüye boğar ve uyarı değersizleşir."""
    for zararsiz in ("gelecek barlarda ne olur", "sürecin tamamı", "tamamen alakasiz qqqq"):
        assert episodic.check_against_lessons(zararsiz) == [], \
            f"{zararsiz!r} yanlışlıkla ders tetikledi"


def test_dersin_KONUSU_olan_kavram_o_derse_isaret_eder():
    """Kataloğa eklemek yetmez — dersin TA KENDİSİ hakkında olan kavram onu ANMALI.

    `deflated_sharpe` (ders #8'in konusu) ve `brier` (ders #9'un konusu) `ders_ref=[]`
    ile duruyordu: #8/#9 kataloğa hiç girmediği için bağ hiç kurulmamıştı. Bu, aynı
    kusurun ikinci katmanıdır (Tur-48)."""
    for kavram, ders in (("deflated_sharpe", "ACI-8"), ("brier", "ACI-9")):
        assert ders in knowledge.CONCEPTS[kavram].ders_ref, \
            f"{kavram} kavramı {ders} dersini ANMIYOR"


def test_yeni_derslerin_tuzak_metni_KURALI_tasiyor():
    """Bir ders_ref işaretçidir; kavramın kendi metni KURALI da söylemelidir.

    Yoksa okuyan 'ACI-8' görür, ne olduğunu görmez (Tur-42: işaretçi cevabı taşımalı)."""
    dsr = " ".join(knowledge.CONCEPTS["deflated_sharpe"].tuzaklar)
    assert "TRİVİYAL" in dsr and "FAZLALIK" in dsr
    br = " ".join(knowledge.CONCEPTS["brier"].tuzaklar)
    assert "paired_brier_t" in br and "taban" in br.lower()
