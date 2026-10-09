"""nihai.ai.agent testleri — ask() (RAG + dürüstlük guardı) ve plan/next_task/complete_task
(hipotez kuyruğu). Ağ YOK: router.complete HER testte monkeypatch'lenir (reasoner.py test
deseniyle aynı — .env'de gerçek bir sağlayıcı anahtarı varsa bile testler asla ağa çıkmaz).
Dosya yolları tmp_path'e monkeypatch'lenir (memory.py/episodic.py test deseni).
"""
from __future__ import annotations

import importlib
import json

import pytest

from nihai.ai import agent, episodic, reasoner, router


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Varsayılan: LLM yok (router.complete None döner) — deterministik şablon kullanılır.
    Ağa asla çıkılmaz (bu proje .env'sinde gerçek bir GEMINI/ANTHROPIC anahtarı olabilir)."""
    monkeypatch.setattr(router, "complete", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _tmp_agenda(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "AGENDA_FILE", tmp_path / "ai_agenda.jsonl")
    return tmp_path


# ---------------------------------------------------------------------------
# ask() — temel davranış
# ---------------------------------------------------------------------------
def test_ask_bos_soru_cokmeden_cevap_doner():
    out = agent.ask("")
    assert out["sources"] == []
    assert reasoner.CLOSING in out["answer"]
    assert out["closing"] == reasoner.CLOSING


def test_ask_bilinen_kavram_knowledge_kaynagi_verir():
    out = agent.ask("PEAD nedir")
    assert any(s.startswith("knowledge/") for s in out["sources"])
    assert reasoner.CLOSING in out["answer"]


def test_ask_cevap_dict_semasi():
    out = agent.ask("momentum")
    assert set(out) == {"answer", "sources", "closing"}
    assert isinstance(out["sources"], list)


def test_ask_cevap_her_zaman_closing_ile_biter():
    for soru in ("", "survivorship bias nedir", "tamamen alakasız zzz123 soru"):
        out = agent.ask(soru)
        assert out["answer"].rstrip().endswith(reasoner.CLOSING)


def test_ask_eslesme_yoksa_durust_bulunamadi_cevabi():
    out = agent.ask("zzz_hicbir_yerde_eslesmeyecek_uydurma_terim_123")
    assert "bulunamadı" in out["answer"].lower()
    assert out["sources"] == []


def test_ask_kurumsal_ders_kaynagi_survivorship():
    out = agent.ask("survivorship point-in-time PIT evren")
    assert any(s.startswith("episodic/lesson/ACI-1") for s in out["sources"])


def test_ask_kaynaklar_tekrarsiz_ve_siniirli():
    out = agent.ask("kenar exec-lag pairs kointegrasyon gecikme")
    assert len(out["sources"]) == len(set(out["sources"]))
    assert len(out["sources"]) <= agent._MAX_SOURCES


def test_ask_docs_taramasi(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "EDGE_FINDINGS.md").write_text(
        "# Bulgular\n## PEAD PIT sonucu\nPEAD PIT S&P500 ozel_test_anahtari sonucu.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(agent, "DOCS_DIR", docs)
    out = agent.ask("ozel_test_anahtari")
    assert any("EDGE_FINDINGS.md" in s for s in out["sources"])
    assert "İlgili belgeler" in out["answer"]


def test_ask_docs_dizini_yoksa_cokmez(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "DOCS_DIR", tmp_path / "yok_boyle_bir_dizin")
    out = agent.ask("herhangi bir soru")
    assert reasoner.CLOSING in out["answer"]


# ---------------------------------------------------------------------------
# ask() — DÜRÜSTLÜK GUARDI (reasoner._apply_llm ile aynı sözleşme)
# ---------------------------------------------------------------------------
def test_ask_llm_kaynaklari_korumazsa_atilir(monkeypatch):
    monkeypatch.setattr(router, "complete", lambda *a, **k: "Her şey harika görünüyor!")
    out = agent.ask("PEAD nedir")
    # LLM çıktısı kaynak etiketlerini/kapanışı korumadığından deterministik metin kullanılmalı
    assert "harika" not in out["answer"]
    assert reasoner.CLOSING in out["answer"]


def test_ask_llm_korursa_kullanilir(monkeypatch):
    det_holder: dict[str, str] = {}
    orig_apply = agent._apply_llm

    def spy(det_text, must_tokens):
        det_holder["det"] = det_text
        return orig_apply(det_text, must_tokens)

    monkeypatch.setattr(agent, "_apply_llm", spy)

    def sahte_llm(task, prompt, **k):
        det = det_holder.get("det", "")
        # kaynak etiketlerini + kapanışı AYNEN koruyan akıcı cevap üret
        return "AKICI CEVAP: " + det

    monkeypatch.setattr(router, "complete", sahte_llm)
    out = agent.ask("PEAD nedir")
    assert out["answer"].startswith("AKICI CEVAP:")


def test_ask_router_istisna_firlatirsa_fail_soft(monkeypatch):
    def patlayan(*a, **k):
        raise RuntimeError("ağ hatası simülasyonu")

    monkeypatch.setattr(router, "complete", patlayan)
    out = agent.ask("PEAD nedir")
    assert reasoner.CLOSING in out["answer"]


# ---------------------------------------------------------------------------
# episodic/consolidate — ImportError-dayanıklılık
# ---------------------------------------------------------------------------
def test_lesson_lines_importerror_dayanikli(monkeypatch):
    def raiser(name):
        if name.startswith("nihai.ai."):
            raise ImportError("simülasyon: modül yok")
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", raiser)
    lines, sources, tokens = agent._lesson_lines("survivorship")
    assert lines == []
    assert sources == []
    assert tokens == []


def test_semantic_lines_importerror_dayanikli(monkeypatch):
    def raiser(name):
        if name.startswith("nihai.ai."):
            raise ImportError("simülasyon: modül yok")
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", raiser)
    lines, sources = agent._semantic_lines("herhangi bir soru")
    assert lines == []
    assert sources == []


def test_ask_ic_modul_importerror_olsa_da_cevap_doner(monkeypatch):
    def raiser(name):
        if name.startswith("nihai.ai."):
            raise ImportError("simülasyon")
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", raiser)
    out = agent.ask("survivorship PEAD")
    assert reasoner.CLOSING in out["answer"]
    # knowledge.search hâlâ çalışır (agent.py'de doğrudan import — lazy değil)
    assert any(s.startswith("knowledge/") for s in out["sources"])


# ---------------------------------------------------------------------------
# plan / next_task / complete_task — hipotez kuyruğu
# ---------------------------------------------------------------------------
def test_plan_kanonik_sirayi_kuyruga_ekler():
    pid = agent.plan("VRP ailesini değerlendir")
    records = agent._read_all()
    tasks = [r for r in records if r.get("plan_id") == pid]
    assert [t["task_type"] for t in tasks] == list(agent.ALLOWED_TASK_TYPES)
    assert all(t["status"] == "pending" for t in tasks)


def test_plan_bilinmeyen_gorev_tipi_valueerror():
    with pytest.raises(ValueError):
        agent.plan("hedef", steps=["uretim_dosyasi_yaz"])


def test_plan_derinlik_limiti_valueerror():
    with pytest.raises(ValueError):
        agent.plan("hedef", depth=agent.MAX_DEPTH + 1)
    # limitin TAM ÜSTÜ hâlâ kabul edilir (sınır dahil değil)
    agent.plan("hedef", depth=agent.MAX_DEPTH)


def test_plan_bos_hedef_valueerror():
    with pytest.raises(ValueError):
        agent.plan("   ")


def test_plan_gunluk_tavan_asilirsa_valueerror(monkeypatch):
    monkeypatch.setattr(agent, "DAILY_TASK_CAP", 5)
    agent.plan("hedef-1")  # 4 adım ekler (kanonik sıra)
    with pytest.raises(ValueError):
        agent.plan("hedef-2")  # +4 adım daha → tavanı aşar


def test_next_task_ve_complete_task_kaldigi_yerden_devam():
    pid = agent.plan("hedef")
    t1 = agent.next_task()
    assert t1 is not None and t1["task_type"] == "veri_kontrol" and t1["plan_id"] == pid

    assert agent.complete_task(t1["id"], "veri sağlıklı") is True
    t2 = agent.next_task()
    assert t2 is not None and t2["task_type"] == "lab_deney"
    assert t2["id"] != t1["id"]


def test_complete_task_bilinmeyen_id_false():
    assert agent.complete_task("olmayan-id", "özet") is False


def test_complete_task_resolution_satiri_eskiyi_silmez(tmp_path):
    agent.plan("hedef")
    t1 = agent.next_task()
    agent.complete_task(t1["id"], "özet-1")
    lines_before = agent.AGENDA_FILE.read_text(encoding="utf-8").strip().splitlines()
    n_before = len(lines_before)
    # ikinci kez tamamlamaya çalışmak (idempotent değil ama YİNE satır ekler — append-only,
    # var olan id ise True döner; eskiyi silmediğini doğrulamak asıl amaç)
    agent.complete_task(t1["id"], "özet-2")
    lines_after = agent.AGENDA_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_after) == n_before + 1
    recs = [json.loads(x) for x in lines_after]
    resolutions = [r for r in recs if r.get("resolution_of") == t1["id"]]
    assert len(resolutions) == 2
    assert resolutions[0]["result_summary"] == "özet-1"
    assert resolutions[1]["result_summary"] == "özet-2"


def test_next_task_bos_kuyrukta_none():
    assert agent.next_task() is None


def test_agenda_status_sayaclari():
    agent.plan("hedef-a")
    status = agent.agenda_status()
    assert status["toplam"] == len(agent.ALLOWED_TASK_TYPES)
    assert status["bekleyen"] == len(agent.ALLOWED_TASK_TYPES)
    assert status["tamamlanan"] == 0
    t = agent.next_task()
    agent.complete_task(t["id"], "x")
    status2 = agent.agenda_status()
    assert status2["tamamlanan"] == 1
    assert status2["bekleyen"] == len(agent.ALLOWED_TASK_TYPES) - 1


def test_plan_ozel_steps_ile_kisaltilmis_sira():
    pid = agent.plan("hızlı kontrol", steps=["veri_kontrol", "verdikt_ozeti"])
    tasks = [r for r in agent._read_all() if r.get("plan_id") == pid]
    assert [t["task_type"] for t in tasks] == ["veri_kontrol", "verdikt_ozeti"]


def test_agenda_bozuk_satir_atlanir(tmp_path):
    agent.AGENDA_FILE.parent.mkdir(parents=True, exist_ok=True)
    agent.AGENDA_FILE.write_text(
        json.dumps({"id": "saglam-1", "plan_id": "p1", "goal": "g", "task_type": "veri_kontrol",
                   "step_index": 0, "depth": 0, "status": "pending",
                   "created_at": "2026-01-01T00:00:00+00:00"}) + "\n"
        + "{bozuk json\n", encoding="utf-8")
    records = agent._read_all()
    assert len(records) == 1
    assert records[0]["id"] == "saglam-1"


def test_check_against_lessons_gercek_api_ile_calisir():
    """agent._lesson_lines gerçek episodic.check_against_lessons'ı çağırır (ImportError
    simülasyonu OLMADAN) — modülün normal/mutlu yol davranışı."""
    lessons = episodic.check_against_lessons("meta_gate kapı gürültü")
    lines, sources, tokens = agent._lesson_lines("meta_gate kapı gürültü")
    assert len(lines) == len(lessons) == len(sources) == len(tokens)
