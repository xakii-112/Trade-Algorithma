"""
agent.py — SORU-CEVAP + HİPOTEZ KUYRUĞU (Tur-16 WP-A, docs/AI_V3_DERIN_BEYIN_PLAN §WP-A). KARAR-DIŞI.

⚙️ KULLANIM KİPİ: **MANUEL / operatör-tetikli.** Üretim akışında bu modülü kendiliğinden
çağıran yoktur; tek üretim yüzeyi panonun AI uç noktasıdır (operatör soru sorduğunda çalışır)
ve gündem kuyruğu boş kalabilir (denetim 2026-07-26'da 0/0/0 ölçüldü). Bu bir arıza DEĞİL,
tasarım: soru-cevap katmanı kendi kendine gündem üretip canlı döngüye yük bindirmez.
Denetim B-8 "ölü kod mu?" diye sorduğu için etiketlendi.

AI'ı "çağrılan fonksiyon"dan "muhatap alınabilen, kendini sınayan sistem"e taşıyan iki API:

1. `ask(question)` — Türkçe soru-cevap (RAG): knowledge.search (kavram tanımı/formül köprüsü/
   tuzak) + episodic.check_against_lessons (kurumsal ders taraması; metin-tabanlı olduğundan
   TEK gerçek "episodik" kaynak burada budur) + consolidate.semantic_recall (varsa;
   ImportError-DAYANIKLI — modül gelecekte kaldırılsa/opsiyonel hale gelse de ask çökmez;
   NOT: semantic_recall SAYISAL özellik vektörü ister, salt-metin sorudan doğal bir sayısal
   bağlam TÜRETİLEMEZ — bu yüzden burada boş sözlükle çağrılır [defensif no-op; sayısal
   bağlamlı çağıranlar için ileride ayrı bir yol açılabilir; bilinçli, belgeli sapma]) +
   docs/*.md (EDGE_FINDINGS dahil) başlık/satır taraması (salt-okunur). Deterministik şablon
   cevap ÖNCE üretilir; router (LLM) varsa YALNIZ akıcılaştırır — reasoner._apply_llm ile AYNI
   DÜRÜSTLÜK GUARDI (kaynak/kapanış korunmazsa LLM çıktısı atılır). Cevap HER ZAMAN
   reasoner.CLOSING ile biter (ACI DERS #3/#4).

2. `plan(goal)` / `next_task()` / `complete_task(id, result_summary)` — çok-adımlı "hipotez
   kuyruğu": bir hedefi ÖNCEDEN TANIMLI, yalnız-analiz görev-tipi şablonlarına böler
   (`ALLOWED_TASK_TYPES`) ve `data/ai_agenda.jsonl`'e atomik ekler. SINIR (anayasa): kuyruk
   YALNIZ analiz/rapor görevleri üretebilir — üretim dosyası yazan bir görev tipi YOKTUR ve
   asla eklenemez (bilinmeyen tip → ValueError). Derinlik (>12) ve günlük görev tavanı (50)
   sert sınırlardır (ValueError) — sınırsız kendi-kendini-çoğaltan ajan döngüsü riskine karşı.
   Kuyruk YALNIZ metadata tutar; agent.py hiçbir görevi OTOMATİK ÇALIŞTIRMAZ (edge_lab/lab
   koşusu insan/ayrı araç tarafından tetiklenir) — bu da anayasanın "üretim dosyası yazmaz"
   ilkesini yapısal olarak garanti eder (agent.py `research/`e import bağı ALMAZ).

Kalıcılık: JSONL append-only (episodic.py deseni); `complete_task` AYRI bir "resolution"
satırı ekler (eskiyi silmez/değiştirmez) — okuyucular birleştirir. Süreç yeniden başlasa
bile `next_task()` dosyadan kaldığı yerden devam eder (durum bellekte tutulmaz).
"""
from __future__ import annotations

import importlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from nihai.ai import knowledge, reasoner, router
from nihai.config import DATA_DIR, ROOT
from nihai.console import ensure_utf8_stdout

# Modül-düzeyi yollar: testler monkeypatch'ler (memory.py/episodic.py deseni).
AGENDA_FILE = DATA_DIR / "ai_agenda.jsonl"
DOCS_DIR = ROOT / "docs"
_LOCK = threading.Lock()

# Kuyruğun üretebileceği TEK görev-tipi kümesi (anayasa: üretim-dosyası-yazan tip YOK).
ALLOWED_TASK_TYPES: tuple[str, ...] = ("veri_kontrol", "lab_deney", "edge_harness", "verdikt_ozeti")

MAX_DEPTH = 12
DAILY_TASK_CAP = 50
_MAX_SOURCES = 10
_MAX_DOC_HITS = 3


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# ask() — kaynaklı soru-cevap (RAG)
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset({
    "nedir", "nasıl", "neden", "niçin", "mi", "mı", "mu", "mü", "ile", "ve", "veya",
    "bir", "bu", "şu", "için", "gibi", "kadar", "midir", "mıdır",
})


def _question_tokens(question: str) -> list[str]:
    """Soruyu arama-anahtar-kelimelerine böl (≥3 harf, durak-kelimeler hariç, tekrarsız,
    sıra korunur). knowledge.search TAM DİZGE alt-dizge eşleşmesi yaptığından ("PEAD nedir"
    kavram tanımının İÇİNDE bir alt-dizge DEĞİLDİR) — çok-kelimeli doğal dil sorularının
    tek-tek anahtar kelimelere bölünmesi gerekir."""
    import re

    seen: set[str] = set()
    out: list[str] = []
    for tok in re.split(r"\W+", question.lower()):
        if len(tok) >= 3 and tok not in _STOPWORDS and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _knowledge_lines(question: str) -> tuple[list[str], list[str], list[str]]:
    """knowledge.search eşleşmelerinden açıklama satırları + kaynak etiketleri + HAM
    kavram-id'leri (dürüstlük guard'ının 'AYNEN korunmalı' token'ları — bkz. `ask`).

    Önce TÜM soru string'i denenir (kullanıcı doğrudan bir kavram adı/id yazmış olabilir),
    sonra tek-tek anahtar kelimeler (`_question_tokens`) — ilk 3 BENZERSİZ eşleşme alınır."""
    hits: list[Any] = []
    seen_ids: set[str] = set()
    try:
        candidates = [question, *_question_tokens(question)]
        for cand in candidates:
            if len(hits) >= 3:
                break
            for c in knowledge.search(cand):
                if c.id in seen_ids:
                    continue
                seen_ids.add(c.id)
                hits.append(c)
                if len(hits) >= 3:
                    break
    except Exception as e:  # noqa: BLE001 — bilgi taraması ask'ı asla düşürmesin
        logger.debug(f"[ai.agent] knowledge.search hatası: {e}")
        return [], [], []
    lines: list[str] = []
    sources: list[str] = []
    tokens: list[str] = []
    for c in hits:
        exp = knowledge.explain(c.id)
        if exp:
            lines.append(exp)
        sources.append(f"knowledge/{c.id}")
        tokens.append(c.id)
    return lines, sources, tokens


def _lazy_module(name: str) -> Any | None:
    """`nihai.ai.<name>` alt-modülünü importlib ile YÜKLE; ImportError → None (dayanıklı).
    `importlib.import_module` kullanımı BİLİNÇLİ: `from nihai.ai import X` sözdiziminin
    Python'un modül-önbelleğiyle (parent paket zaten `X`'i attribute olarak taşıyorsa)
    ImportError'ı hiç TETİKLEMEDEN geçmesi riskini önler — testler bu fonksiyonu
    monkeypatch'leyerek 'modül YOK/kaldırıldı' senaryosunu güvenilir şekilde simüle eder."""
    try:
        return importlib.import_module(f"nihai.ai.{name}")
    except ImportError as e:
        logger.debug(f"[ai.agent] nihai.ai.{name} içe aktarılamadı: {e}")
        return None


def _lesson_lines(question: str) -> tuple[list[str], list[str], list[str]]:
    """episodic.check_against_lessons — TEK metin-tabanlı episodik/kurumsal-ders kaynağı."""
    episodic = _lazy_module("episodic")
    if episodic is None:
        return [], [], []
    try:
        lessons = episodic.check_against_lessons(question)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.agent] check_against_lessons hatası: {e}")
        return [], [], []
    lines = [f"{les.id}: {les.claim} (guard: {les.guard})" for les in lessons]
    sources = [f"episodic/lesson/{les.id}" for les in lessons]
    tokens = [les.id for les in lessons]
    return lines, sources, tokens


def _semantic_lines(question: str) -> tuple[list[str], list[str]]:
    """consolidate.semantic_recall — ImportError-dayanıklı (modül docstring'i: metinden
    sayısal özellik vektörü türetilemediği için boş sözlükle çağrılır → bugün için doğal
    olarak no-op, ama ileride sayısal bağlamlı çağıranlar için kanca hazır kalır)."""
    consolidate = _lazy_module("consolidate")
    if consolidate is None:
        return [], []
    try:
        recs = consolidate.semantic_recall({}, k=3)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.agent] semantic_recall hatası: {e}")
        return [], []
    if not recs:
        return [], []
    lines = [f"{r.get('pattern', '?')} (n={r.get('n')}, isabet={r.get('isabet')})" for r in recs]
    sources = ["episodic/semantic_recall"] * len(recs)
    return lines, sources


def _doc_hits(question: str, limit: int = _MAX_DOC_HITS) -> list[dict[str, str]]:
    """docs/*.md (EDGE_FINDINGS dahil) başlık/satır taraması — salt-okunur, IO hatası
    yutulur (dürüstlük: sadece bulunanları raporlar, dosya sistemi hatasında çökmez)."""
    q = (question or "").strip().lower()
    if not q or not DOCS_DIR.is_dir():
        return []
    hits: list[dict[str, str]] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        heading = path.stem
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                heading = stripped.lstrip("#").strip() or heading
                continue
            if q in stripped.lower():
                hits.append({"file": path.name, "heading": heading, "line": stripped[:200]})
                if len(hits) >= limit:
                    return hits
    return hits


def _dedup_cap(items: list[str], limit: int) -> list[str]:
    """Tekrarsızlaştır (ilk-görülme sırası korunur) + `limit`'e kırp."""
    return list(dict.fromkeys(items))[:limit]


def _strip_llm_preamble(out: str, must_tokens: list[str]) -> str:
    """LLM'lerin sızdırdığı önsöz satırlarını ('İşte akıcılaştırılmış metin:' vb.) soyar:
    baştaki KISA (≤100 kr), iki-nokta/ünlemle biten, must_token ve CLOSING içermeyen
    satırlar atılır; ilk içerik satırında durulur (içerik asla kırpılmaz — Tur-17 düzeltmesi)."""
    lines = out.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        is_preamble = (len(ln) <= 100 and ln.endswith((":", "!"))
                       and reasoner.CLOSING not in ln
                       and not any(tok in ln for tok in must_tokens))
        if is_preamble:
            i += 1
            continue
        break
    return "\n".join(lines[i:]).lstrip("\n")


def _apply_llm(det_text: str, must_tokens: list[str]) -> str:
    """router varsa cevabı akıcılaştır; `must_tokens` (kavram/ders id'leri + doküman adları —
    det_text'in İÇİNDE HARFİYEN geçen HAM kimlikler, `sources` alanının süslü etiketleri
    DEĞİL) VEYA kapanış kaybolursa (uydurma koruması) LLM çıktısı ATILIR — reasoner
    ._apply_llm ile AYNI sözleşme."""
    prompt = ("Aşağıdaki Türkçe quant-sistem cevabını akıcılaştır. Kavram/ders kimliklerini "
              "(parantez içindeki id'ler, ör. '(pead)', 'ACI-6') ve son satırdaki kapanış "
              "cümlesini AYNEN koru; yeni iddia/sayı EKLEME. YALNIZ akıcılaştırılmış metnin "
              "kendisini döndür — 'İşte...' gibi önsöz/sonsöz cümlesi YAZMA:\n\n" + det_text)
    try:
        out = router.complete("judgment", prompt)
    except Exception as e:  # noqa: BLE001 — LLM hiçbir koşulda cevabı düşürmesin
        logger.debug(f"[ai.agent] router hatası: {e}")
        return det_text
    if not out:
        return det_text
    out = _strip_llm_preamble(out, must_tokens)
    if reasoner.CLOSING not in out or any(tok not in out for tok in must_tokens):
        logger.debug("[ai.agent] LLM çıktısı kaynak/kapanış korumadı → deterministik")
        return det_text
    return out


def ask(question: str) -> dict[str, Any]:
    """Türkçe soru → kaynaklı cevap. Girdi boş/anlamsız olsa da ÇÖKMEZ (savunmacı); hiçbir
    dahili kaynak (bilgi/ders/anlamsal/doküman) bulunamazsa dürüst 'bulunamadı' cevabı döner.

    Dönüş: {'answer' (CLOSING dahil tam metin), 'sources' (kaynak etiketi listesi,
    tekrarsız, en fazla 10), 'closing' (reasoner.CLOSING — kolaylık için ayrıca)}."""
    q = (question or "").strip()
    if not q:
        text = "Soru boş — cevap üretilemedi.\n" + reasoner.CLOSING
        return {"answer": text, "sources": [], "closing": reasoner.CLOSING}

    k_lines, k_sources, k_tokens = _knowledge_lines(q)
    l_lines, l_sources, l_tokens = _lesson_lines(q)
    s_lines, s_sources = _semantic_lines(q)
    doc_hits = _doc_hits(q)

    sections: list[str] = []
    if k_lines:
        sections.append("[Bilgi]\n" + "\n\n".join(k_lines))
    if l_lines:
        sections.append("[Kurumsal ders uyarıları]\n" + "\n".join(l_lines))
    if s_lines:
        sections.append("[Anlamsal hafıza]\n" + "\n".join(s_lines))
    doc_sources: list[str] = []
    doc_tokens: list[str] = []
    if doc_hits:
        doc_lines = [f"- {h['file']} § {h['heading']}: {h['line']}" for h in doc_hits]
        sections.append("[İlgili belgeler]\n" + "\n".join(doc_lines))
        doc_sources = [f"{h['file']}#{h['heading']}" for h in doc_hits]
        doc_tokens = [h["file"] for h in doc_hits]

    all_sources = _dedup_cap(k_sources + l_sources + s_sources + doc_sources, _MAX_SOURCES)
    # DÜRÜSTLÜK GUARDININ token'ları: det_text İÇİNDE HARFİYEN geçen HAM kimlikler (kavram-id,
    # ders-id, doküman-adı) — `sources` alanının 'knowledge/'/'episodic/lesson/' gibi süslü
    # etiketleri det_text'te doğal biçimde GEÇMEZ, o yüzden AYRI tutulur (bkz. _apply_llm).
    must_tokens = _dedup_cap(k_tokens + l_tokens + doc_tokens, _MAX_SOURCES)
    if not sections:
        sections.append(
            "Bilgi tabanında, kurumsal derslerde, hafızada veya dokümanlarda bu soruyla "
            "doğrudan eşleşen bir kaynak bulunamadı — şablon-cevap üretilemedi."
        )
    det_text = "\n\n".join(sections) + "\n" + reasoner.CLOSING
    final_text = _apply_llm(det_text, must_tokens)
    return {"answer": final_text, "sources": all_sources, "closing": reasoner.CLOSING}


# ---------------------------------------------------------------------------
# Hipotez kuyruğu — plan / next_task / complete_task
# ---------------------------------------------------------------------------
@dataclass
class _AgendaTask:
    """Kuyruktaki tek bir görev satırı (JSONL kaydının Python karşılığı)."""

    id: str
    plan_id: str
    goal: str
    task_type: str
    step_index: int
    depth: int
    status: str = "pending"
    created_at: str = field(default_factory=_utc_now_iso)


def _append_line(rec: dict) -> None:
    """Tek satır JSONL ekle (thread-safe append; memory.py/episodic.py deseni)."""
    line = json.dumps(rec, ensure_ascii=False)
    with _LOCK:
        AGENDA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with AGENDA_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _read_all() -> list[dict]:
    """Tüm geçerli JSONL kayıtları oku (bozuk satır atlanır, sayısı loglanır)."""
    if not AGENDA_FILE.exists():
        return []
    out: list[dict] = []
    corrupt = 0
    try:
        with AGENDA_FILE.open(encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except ValueError:
                    corrupt += 1
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
                else:
                    corrupt += 1
    except OSError as e:
        logger.warning(f"[ai.agent] okunamadı ({AGENDA_FILE.name}): {e}")
        return out
    if corrupt:
        logger.warning(f"[ai.agent] {corrupt} bozuk satır atlandı ({AGENDA_FILE.name})")
    return out


def _task_records(records: list[dict]) -> list[dict]:
    """Yalnız görev kayıtları (resolution satırları HARİÇ — 'resolution_of' alanı yok)."""
    return [r for r in records if "resolution_of" not in r]


def _resolutions(records: list[dict]) -> dict[str, dict]:
    """task_id → EN SON resolution (append-only; sonraki üzerine yazar)."""
    out: dict[str, dict] = {}
    for r in records:
        tid = r.get("resolution_of")
        if isinstance(tid, str):
            out[tid] = r
    return out


def _today_task_count(records: list[dict]) -> int:
    today = datetime.now(UTC).date().isoformat()
    return sum(1 for r in _task_records(records) if str(r.get("created_at", "")).startswith(today))


def plan(goal: str, *, steps: list[str] | None = None, depth: int = 0) -> str:
    """Hedefi ÖNCEDEN TANIMLI görev-tipi şablonlarına böl ve kuyruğa atomik ekle.

    `steps` verilmezse KANONİK sıra kullanılır: veri_kontrol → lab_deney → edge_harness →
    verdikt_ozeti (spec örneği: "VRP ailesini değerlendir" → veri-kontrol → lab ön-analiz →
    edge_lab harness → verdikt-özeti). `steps` verilirse HER elemanı `ALLOWED_TASK_TYPES`
    içinde olmalı — değilse ValueError (anayasa: üretim-dosyası-yazan görev tipi YOKTUR,
    yalnız bu 4 analiz/rapor tipi var olabilir).

    Sert sınırlar (ValueError): `depth` > `MAX_DEPTH` (12) — kendi-kendini-çoğaltan ajan
    döngüsü riskine karşı; günlük EKLENEN görev sayısı `DAILY_TASK_CAP` (50) aşılırsa.

    Dönüş: `plan_id` (bu hedefin TÜM adımlarını gruplayan kimlik; `next_task`/`complete_task`
    ile devam ettirilir — süreç yeniden başlasa bile kuyruk dosyadan okunur)."""
    goal_s = (goal or "").strip()
    if not goal_s:
        raise ValueError("goal boş olamaz")
    if depth > MAX_DEPTH:
        raise ValueError(f"derinlik limiti aşıldı: depth={depth} > {MAX_DEPTH}")
    task_types = list(steps) if steps is not None else list(ALLOWED_TASK_TYPES)
    if not task_types:
        raise ValueError("en az bir görev-tipi gerekir")
    for t in task_types:
        if t not in ALLOWED_TASK_TYPES:
            raise ValueError(
                f"bilinmeyen görev tipi: {t!r} (izinli: {ALLOWED_TASK_TYPES}) — "
                "üretim-dosyası-yazan görev tipi YOKTUR (anayasa)"
            )
    existing = _read_all()
    if _today_task_count(existing) + len(task_types) > DAILY_TASK_CAP:
        raise ValueError(f"günlük görev tavanı ({DAILY_TASK_CAP}) aşılıyor")

    plan_id = uuid.uuid4().hex[:12]
    ts = _utc_now_iso()
    for i, t in enumerate(task_types):
        task = _AgendaTask(id=f"{plan_id}-{i}", plan_id=plan_id, goal=goal_s, task_type=t,
                            step_index=i, depth=depth, created_at=ts)
        _append_line({"id": task.id, "plan_id": task.plan_id, "goal": task.goal,
                      "task_type": task.task_type, "step_index": task.step_index,
                      "depth": task.depth, "status": task.status, "created_at": task.created_at})
    return plan_id


def next_task() -> dict | None:
    """Sıradaki BEKLEYEN görevi döndür (dosya sırasıyla — en eski ilk); hiç yoksa None.
    Kaldığı-yerden-devam: süreç yeniden başlasa bile dosyadan okunduğundan durum kaybolmaz."""
    records = _read_all()
    resolved_ids = set(_resolutions(records).keys())
    for r in _task_records(records):
        if r.get("id") not in resolved_ids and r.get("status") == "pending":
            return r
    return None


def complete_task(task_id: str, result_summary: str) -> bool:
    """Görevi TAMAMLANDI işaretle: AYRI bir resolution satırı ekler (episodic.py'nin
    resolution deseni — eskiyi SİLMEZ/DEĞİŞTİRMEZ, append-only). Bilinmeyen id → False
    (satır yazılmaz — sessiz veri bozulması yok)."""
    records = _read_all()
    ids = {r.get("id") for r in _task_records(records)}
    if task_id not in ids:
        return False
    _append_line({"resolution_of": str(task_id), "ts": _utc_now_iso(),
                 "status": "done", "result_summary": str(result_summary)})
    return True


def agenda_status() -> dict:
    """Kuyruğun kısa özeti (salt-okunur): toplam/bekleyen/tamamlanan görev sayısı + bugün
    eklenen sayı (günlük tavan izlemi için)."""
    records = _read_all()
    tasks = _task_records(records)
    resolved_ids = set(_resolutions(records).keys())
    pending = sum(1 for r in tasks if r.get("id") not in resolved_ids)
    return {
        "toplam": len(tasks),
        "bekleyen": pending,
        "tamamlanan": len(tasks) - pending,
        "bugun_eklenen": _today_task_count(records),
        "gunluk_tavan": DAILY_TASK_CAP,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: `--ask "soru"` | `--plan "hedef"` | `--next` | `--status` (manuel/tanılama amaçlı;
    entegrasyon [WP-I, `brain --ask`] bu dosyanın kapsamı DIŞINDA)."""
    import argparse
    ap = argparse.ArgumentParser(description="AI ajanlık katmanı (KARAR-DIŞI)")
    ap.add_argument("--ask", metavar="SORU")
    ap.add_argument("--plan", metavar="HEDEF")
    ap.add_argument("--next", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)
    if args.ask:
        out = ask(args.ask)
        print(out["answer"])
        print("Kaynaklar: " + ", ".join(out["sources"]) if out["sources"] else "Kaynaklar: (yok)")
        return 0
    if args.plan:
        pid = plan(args.plan)
        print(f"plan_id={pid}")
        return 0
    if args.next:
        t = next_task()
        print(json.dumps(t, ensure_ascii=False, indent=1) if t else "Bekleyen görev yok.")
        return 0
    if args.status:
        print(json.dumps(agenda_status(), ensure_ascii=False, indent=1))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    raise SystemExit(main())
