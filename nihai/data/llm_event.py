"""
llm_event.py — LLM (Claude) 8-K olay-çıkarımı ADAY sinyali (C3).

DÜRÜSTLÜK / DİSİPLİN (docs/EDGE_FINDINGS.md):
- Bu bir KENAR (alpha) İDDİASI DEĞİLDİR. ADAY sinyaldir: önce ölçülür
  (research/llm_event_test.py → 2 periyot + al-tut beta + pit_universe survivorship),
  PEAD/insider/sentiment'in §13.5/§16'da yaşadığı survivorship deflasyonu burada da
  BEKLENİR. ÖLÇ, SHIP ETME. EDGE_FINDINGS'in gösterdiği gibi (Ölçüm A/B/C/D): negatif-kenarlı
  tabanın üstüne model sofistikasyonu eklemek overfit yüzeyini büyütür — bu modül de yalnız
  YENİ, BAĞIMSIZ, PIT-doğrulanmış bir sinyal KAYNAĞI olduğunu KANITLARSA değer taşır.
- LLM burada TAHMİNCİ değil, ÇIKARICI'dır (parser): yapılandırılmamış 8-K metnini →
  yapılandırılmış olay alanına (tip/yön/materyallik/güven) çevirir. Çıktı doğrudan işleme
  bağlanmaz; aday sinyal olarak research harness'ından geçer. (Görev B — bkz. tartışma.)
- Fiyat-göstergesi DEĞİL; olay-bazlı (kurumsal duyuru). Fiyat-gösterge ailesinden bağımsız.

NEDEN insider/PEAD ile AYNI AİLE, sentiment'ten DAHA İYİ test edilebilir:
- 8-K dosyalamaları FİLİNG tarihine göre indekslenir; bilgi ancak dosyalanınca halka açık.
  Giriş = dosyalama tarihinin ERTESİ barı → ileri-bakış yok (insider deseni).
- Reddit'in aksine EDGAR full-text TARİHSELDİR ve point-in-time'dır → bu modül SOĞUK
  başlangıçta bile geçmişe gidebilir (forward-accumulation gerekmez). Bu yüzden PIT
  survivorship testi (pead_pit_test/insider_test deseni) doğrudan uygulanabilir.

SINIRLAR (dürüstçe):
- Model AĞIRLIKLARI eğitim-kesimi sonrası → kuramsal "benzer şirkete ne oldu biliyor" sızıntısı
  riski var; ama model YALNIZ dosyalama metnini görür (gelecekteki fiyatı değil). Yine de
  PIT sonuçları temkinli okunmalı.
- Maliyet: her 8-K = 1 Claude çağrısı. Bu yüzden accession başına AGRESİF cache (disk) —
  backtest sepet×yıl üzerinde binlerce çağrı olurdu; cache ile her dosyalama bir kez işlenir.
- Ücretsiz veri PIT değil (CIK/isim değişiklikleri, delist tickerlar). Altın standart Sharadar/CRSP.

GEREKSİNİM (.env):
- ANTHROPIC_API_KEY  (Claude API anahtarı)
- SEC_USER_AGENT     (insider.py ile paylaşılır — kibar User-Agent, SEC şartı)
- LLM_EVENT_MODEL    (opsiyonel; varsayılan claude-opus-4-8. Toplu çıkarımda maliyet için
                      LLM_EVENT_MODEL=claude-haiku-4-5 tercih edilebilir — KARARı kullanıcı verir.)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import pandas as pd
from loguru import logger

from nihai.config import DATA_DIR

# EDGAR plumbing'i insider.py'den yeniden kullan (CIK eşlemesi + kibar throttle'lı GET).
from nihai.data.insider import _get, ticker_to_cik

LLM_DIR = DATA_DIR / "llm_event_cache"
LLM_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Varsayılan model = claude-opus-4-8 (claude-api skill kuralı: maliyet için kendiliğinden
# düşürme; karar kullanıcının). Toplu 8-K çıkarımı maliyet-duyarlıysa kullanıcı .env'de
# LLM_EVENT_MODEL=claude-haiku-4-5 ayarlayabilir.
LLM_EVENT_MODEL = os.getenv("LLM_EVENT_MODEL", "claude-opus-4-8")

# Sağlayıcı seçimi: "anthropic" (Claude, ücretli) | "gemini" (Google AI Studio, ÜCRETSİZ katman).
# Bu görev (8-K → yapılandırılmış olay) bir sınıflandırma işi → ucuz/Flash modeller yeterli.
LLM_EVENT_PROVIDER = os.getenv("LLM_EVENT_PROVIDER", "anthropic").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
# Model-id env'den ayarlanabilir (SDK/sürüm farkına karşı). aistudio.google.com'da güncel
# ücretsiz Flash id'sini gör; farklıysa GEMINI_MODEL=... ile değiştir.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# 8-K gövdesi exhibit'lerle çok büyük olabilir; çıkarım için baş kısmı yeterli. Sessiz DEĞİL —
# bilinçli bir tasarım sınırı (sınıflandırma görevi, kullanıcı içeriği değil).
MAX_DOC_CHARS = int(os.getenv("LLM_EVENT_MAX_CHARS", "12000"))
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# 1M token başına (girdi, çıktı) — claude-api referansı (cached 2026-06). compare_models
# maliyet tablosu için. Fable 5 dahil edilmedi (bu görev için aşırı pahalı/gereksiz).
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
}

_client = None                                   # anthropic.Anthropic (lazy, cache)
_gemini_client = None                            # google.genai.Client (lazy, cache)
_events_cache: dict[str, pd.DataFrame | None] = {}

# LLM'in dolduracağı yapılandırılmış olay şeması (output_config.format = json_schema).
# Sayısal aralık kısıtı (min/max) structured-outputs'ta desteklenmez → kodda clamp edilir.
EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {
            "type": "string",
            "enum": ["guidance_raise", "guidance_cut", "mna", "executive_change",
                     "legal_regulatory", "product", "financing", "restructuring",
                     "other", "none"],
        },
        "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "magnitude": {"type": "number", "description": "Materyallik 0..1 (0=önemsiz, 1=çok önemli)"},
        "confidence": {"type": "number", "description": "Çıkarım güveni 0..1"},
        "rationale": {"type": "string", "description": "Tek cümle gerekçe (≤120 karakter)"},
    },
    "required": ["event_type", "direction", "magnitude", "confidence", "rationale"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are a financial filing analyst. You are given the body text of an SEC Form 8-K "
    "(material event disclosure). Classify ONLY what the text states — do not speculate about "
    "price impact, do not use outside knowledge of what later happened. Judge 'direction' as the "
    "likely sign of the disclosed news for equity holders (bullish/bearish/neutral), 'magnitude' "
    "as how material the event is (0..1), and 'confidence' as how certain your classification is "
    "(0..1). If the filing is routine/administrative with no material news, use event_type='none', "
    "direction='neutral', low magnitude."
)


@dataclass
class EventExtract:
    """Tek 8-K'nın yapılandırılmış çıkarımı. score = signed materyallik∈[-1,+1]."""
    event_type: str = "none"
    direction: str = "neutral"
    magnitude: float = 0.0
    confidence: float = 0.0
    rationale: str = ""

    @property
    def score(self) -> float:
        sign = {"bullish": 1.0, "bearish": -1.0}.get(self.direction, 0.0)
        return round(sign * _clamp(self.magnitude) * _clamp(self.confidence), 4)

    def as_row(self) -> dict:
        return {"event_type": self.event_type, "direction": self.direction,
                "magnitude": _clamp(self.magnitude), "confidence": _clamp(self.confidence),
                "score": self.score, "rationale": self.rationale[:120]}


def _clamp(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _get_client():
    """Lazy anthropic client. Anahtar/SDK yok → None (sessiz devre-dışı)."""
    global _client
    if _client is not None:
        return _client
    if not ANTHROPIC_API_KEY:
        logger.debug("[llm_event] ANTHROPIC_API_KEY yok → devre-dışı")
        return None
    try:
        import anthropic
        _client = anthropic.Anthropic()      # anahtarı ortamdan çözer
        return _client
    except Exception as e:
        logger.debug(f"[llm_event] anthropic init başarısız: {e}")
        return None


# ---------------------------------------------------------------------------
# LLM çıkarım — yapılandırılmış olay (Claude API, output_config.format)
# ---------------------------------------------------------------------------
def _to_extract(data: dict) -> EventExtract:
    """JSON dict → EventExtract (ortak, savunmacı: eksik/bozuk alan = güvenli varsayılan)."""
    return EventExtract(
        event_type=str(data.get("event_type", "none")),
        direction=str(data.get("direction", "neutral")),
        magnitude=_clamp(data.get("magnitude", 0.0)),
        confidence=_clamp(data.get("confidence", 0.0)),
        rationale=str(data.get("rationale", "")),
    )


def _loads_lenient(text: str) -> dict | None:
    """JSON ayrıştır; ```json ... ``` sarmalı veya önek/sonek varsa ilk {...} bloğunu ayıkla."""
    if not text:
        return None
    t = text.strip().strip("`")
    if t[:4].lower() == "json":
        t = t[4:]
    try:
        return json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                return None
    return None


def _extract_anthropic(doc_text: str, model: str) -> tuple[EventExtract | None, dict]:
    """Claude yolu: structured outputs (output_config.format). (ex, usage) döndürür.

    Geçici hatalar (429 rate-limit / 529 overloaded / 5xx) kısa backoff'la yeniden
    denenir — Gemini yolundaki dayanıklılıkla parite (toplu 8-K çıkarımında tek anlık
    yük hatası dosyalamayı sessizce atlamasın; cache'lenmediği için sonraki koşuda
    zaten tekrar denenirdi ama o koşu günler sonra olabilir)."""
    client = _get_client()
    if client is None or not doc_text.strip():
        return None, {}
    resp = None
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                system=_SYSTEM,
                output_config={"format": {"type": "json_schema", "schema": EVENT_SCHEMA}},
                messages=[{"role": "user",
                           "content": f"8-K filing body:\n\n{doc_text[:MAX_DOC_CHARS]}"}],
            )
            break
        except Exception as e:
            msg = str(e)
            transient = any(c in msg for c in ("429", "529", "503", "500",
                                               "overloaded", "rate_limit", "Overloaded"))
            logger.debug(f"[llm_event] anthropic deneme {attempt+1}/3 ({model}): {e}")
            if not transient or attempt == 2:
                return None, {}
            time.sleep(2.0 * (attempt + 1))
    if resp is None:
        return None, {}
    u = getattr(resp, "usage", None)
    usage = {"input": int(getattr(u, "input_tokens", 0) or 0),
             "output": int(getattr(u, "output_tokens", 0) or 0)}
    if getattr(resp, "stop_reason", None) == "refusal":
        return EventExtract(rationale="REFUSAL"), usage
    text = next((getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"), "")
    data = _loads_lenient(text)
    return (_to_extract(data) if data else None), usage


# Gemini structured-output: SDK/model sürüm farkına karşı response_schema yerine
# response_mime_type=application/json + sistem promptunda alan şeması + savunmacı parse.
_GEMINI_SYSTEM = _SYSTEM + (
    " Respond with ONLY a JSON object (no prose, no markdown fences). Keys EXACTLY: "
    "event_type (one of guidance_raise|guidance_cut|mna|executive_change|legal_regulatory|"
    "product|financing|restructuring|other|none), direction (bullish|bearish|neutral), "
    "magnitude (number 0..1), confidence (number 0..1), rationale (string <=120 chars)."
)


def _get_gemini_client():
    """Lazy google-genai client. Anahtar/SDK yok → None (sessiz devre-dışı)."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    if not GEMINI_API_KEY:
        logger.debug("[llm_event] GEMINI_API_KEY yok → devre-dışı")
        return None
    try:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        return _gemini_client
    except Exception as e:
        logger.debug(f"[llm_event] gemini init başarısız: {e}")
        return None


def _extract_gemini(doc_text: str, model: str) -> tuple[EventExtract | None, dict]:
    """Gemini yolu: generate_content + JSON mime. (ex, usage) döndürür (ücretsiz katman → ~$0)."""
    client = _get_gemini_client()
    if client is None or not doc_text.strip():
        return None, {}
    try:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=_GEMINI_SYSTEM,
            response_mime_type="application/json",
            temperature=0.0,
        )
    except Exception as e:
        logger.debug(f"[llm_event] gemini config başarısız: {e}")
        return None, {}
    # Geçici 503/429 (yük) anlıktır → kısa backoff ile birkaç deneme.
    resp = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=f"8-K filing body:\n\n{doc_text[:MAX_DOC_CHARS]}",
                config=cfg,
            )
            break
        except Exception as e:
            msg = str(e)
            transient = any(c in msg for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            logger.debug(f"[llm_event] gemini deneme {attempt+1}/3 ({model}): {e}")
            if not transient or attempt == 2:
                return None, {}
            time.sleep(2.0 * (attempt + 1))
    if resp is None:
        return None, {}
    um = getattr(resp, "usage_metadata", None)
    usage = {"input": int(getattr(um, "prompt_token_count", 0) or 0),
             "output": int(getattr(um, "candidates_token_count", 0) or 0)}
    data = _loads_lenient(getattr(resp, "text", "") or "")
    return (_to_extract(data) if data else None), usage


def _extract_with_usage(doc_text: str, model: str) -> tuple[EventExtract | None, dict]:
    """Sağlayıcıya göre çıkarım (LLM_EVENT_PROVIDER). compare_models Claude'a doğrudan gider."""
    if LLM_EVENT_PROVIDER == "gemini":
        return _extract_gemini(doc_text, model)
    return _extract_anthropic(doc_text, model)


def extract_event(doc_text: str, model: str | None = None) -> EventExtract | None:
    """8-K gövde metnini → yapılandırılmış olaya çevir. None = istemci yok / hata / refusal.
    model=None → sağlayıcının varsayılan modeli (LLM_EVENT_MODEL / GEMINI_MODEL)."""
    default = GEMINI_MODEL if LLM_EVENT_PROVIDER == "gemini" else LLM_EVENT_MODEL
    ex, _ = _extract_with_usage(doc_text, model or default)
    return ex


# ---------------------------------------------------------------------------
# EDGAR — 8-K dosyalamalarını listele + birincil belge metnini çek
# ---------------------------------------------------------------------------
def _iter_8k(cik: int) -> list[tuple[str, str, str]]:
    """CIK'in TÜM 8-K dosyalamaları → [(filingDate, accession, primaryDocument), ...].

    insider._iter_form4 deseni: submissions 'recent' + 'files' arşiv parçaları."""
    out: list[tuple[str, str, str]] = []

    def _collect(block: dict) -> None:
        forms = block.get("form", [])
        dates = block.get("filingDate", [])
        accs = block.get("accessionNumber", [])
        docs = block.get("primaryDocument", [])
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            try:
                out.append((dates[i], accs[i], docs[i] if i < len(docs) else ""))
            except Exception:
                continue

    root = _get(SUBMISSIONS_URL.format(cik=cik))
    if not isinstance(root, dict):
        return out
    filings = root.get("filings", {})
    if isinstance(filings.get("recent"), dict):
        _collect(filings["recent"])
    for extra in filings.get("files", []) or []:
        name = extra.get("name")
        if not name:
            continue
        block = _get(f"https://data.sec.gov/submissions/{name}")
        if isinstance(block, dict):
            _collect(block)
    return out


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_EX99_RE = re.compile(r"(?i)ex(?:hibit)?[._\-]?99")


def _strip_html(raw: str) -> str:
    text = _TAG_RE.sub(" ", raw)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&#160;", " ").replace("&#8217;", "'"))
    return _WS_RE.sub(" ", text).strip()


def _fetch_doc(cik: int, acc_nodash: str, name: str) -> str:
    raw = _get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{name}",
               as_json=False)
    return _strip_html(raw) if raw else ""


def _filing_files(cik: int, acc_nodash: str) -> list[str]:
    """Filing dizinindeki dosya adları (EDGAR index.json). Hata → []."""
    idx = _get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/index.json")
    items = (idx.get("directory") or {}).get("item") or [] if isinstance(idx, dict) else []
    return [it.get("name", "") for it in items if it.get("name")]


def _fetch_8k_text(cik: int, accession: str, primary_doc: str) -> str:
    """8-K birincil belgesi + (varsa) Exhibit 99.x basın bülteni → düz metin. Hata → ''.

    KRİTİK: Kazanç/guidance 8-K'larında ASIL içerik (rakamlar, yön) gövdede DEĞİL,
    Exhibit 99.1'dedir; gövde yalnız 'Exhibit 99.1 olarak eklenmiştir' der. Bu yüzden gövde
    bir EX-99 basın bültenine atıf yapıyorsa o ek de çekilip eklenir → çıkarıcı kazanç yönünü
    görebilir. Atıf yoksa ek çekilmez (gereksiz EDGAR isteği yok)."""
    acc_nodash = accession.replace("-", "")
    primary = primary_doc or f"{accession}.txt"
    body = _fetch_doc(cik, acc_nodash, primary)[:5000]
    low = body.lower()
    if not ("exhibit 99" in low or "99.1" in low or "item 2.02" in low):
        return body                       # ek yok / atıf yok → yalnız gövde
    parts = [body] if body else []
    ex_names = [n for n in _filing_files(cik, acc_nodash)
                if _EX99_RE.search(n) and n.lower().endswith((".htm", ".html", ".txt"))]
    for n in ex_names[:2]:                # en fazla 2 EX-99 (basın bülteni + finansal tablo)
        ex = _fetch_doc(cik, acc_nodash, n)
        if ex:
            parts.append(f"[EXHIBIT {n}]\n{ex[:7000]}")
    return "\n\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Olay tablosu kurulumu — accession-keyed disk cache (LLM çağrısını TEKRARLAMA)
# ---------------------------------------------------------------------------
def _cache_path(symbol: str):
    return LLM_DIR / f"{symbol.upper()}.json"


def _load_raw_cache(symbol: str) -> dict:
    p = _cache_path(symbol)
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            pass
    return {"as_of": "", "filings": {}}


def _build_events(symbol: str, *, max_filings: int = 0) -> pd.DataFrame | None:
    """Sembolün 8-K'larını çıkar (cache'lenmemiş accession'ları LLM'e gönder), FİLİNG tarihine
    göre topla. Aynı gün birden çok 8-K → en materyal (|score| en büyük) olan alınır.

    Sütunlar: event_type, direction, magnitude, confidence, score, rationale.
    max_filings>0 → yalnız en yeni N dosyalamayı işle (maliyet bütçesi)."""
    cik = ticker_to_cik(symbol)
    if cik is None:
        return None
    filings = _iter_8k(cik)
    if not filings:
        return None
    # En yeniler önce (date string YYYY-MM-DD sıralanabilir)
    filings.sort(key=lambda t: t[0], reverse=True)
    if max_filings > 0:
        filings = filings[:max_filings]

    raw = _load_raw_cache(symbol)
    store: dict = raw.get("filings", {})
    dirty = False
    for fdate, acc, doc in filings:
        if acc in store:                          # cache hit → LLM çağrısı YOK
            continue
        text = _fetch_8k_text(cik, acc, doc)
        ex = extract_event(text) if text else None
        if ex is None:                            # istemci yok / hata → atla (cache'leme)
            continue
        row = ex.as_row()
        row["date"] = fdate
        store[acc] = row
        dirty = True

    if dirty:
        try:
            raw["filings"] = store
            raw["as_of"] = time.strftime("%Y-%m-%d")
            _cache_path(symbol).write_text(json.dumps(raw), "utf-8")
        except Exception:
            pass

    if not store:
        return None
    # accession satırlarını filing-tarihine göre topla (gün başına en materyal olay)
    by_date: dict[str, dict] = {}
    for row in store.values():
        d = row.get("date")
        if not d:
            continue
        cur = by_date.get(d)
        if cur is None or abs(row.get("score", 0.0)) > abs(cur.get("score", 0.0)):
            by_date[d] = row
    if not by_date:
        return None
    idx = pd.to_datetime(list(by_date.keys()))
    df = pd.DataFrame(list(by_date.values()), index=[d.date() for d in idx])
    return df.sort_index()


def cached_events(symbol: str) -> pd.DataFrame | None:
    """SALT disk cache'ten olay tablosu — EDGAR/AI ÇAĞIRMAZ → web GET için güvenli.
    (_build_events'in toplama mantığının ağsız aynası; çıkarım yapmaz.)"""
    store = _load_raw_cache(symbol).get("filings", {})
    if not store:
        return None
    by_date: dict[str, dict] = {}
    for row in store.values():
        d = row.get("date")
        if not d:
            continue
        cur = by_date.get(d)
        if cur is None or abs(row.get("score", 0.0)) > abs(cur.get("score", 0.0)):
            by_date[d] = row
    if not by_date:
        return None
    idx = pd.to_datetime(list(by_date.keys()))
    df = pd.DataFrame(list(by_date.values()), index=[dd.date() for dd in idx])
    return df.sort_index()


def llm_events(symbol: str, *, refresh: bool = False, max_filings: int = 0
               ) -> pd.DataFrame | None:
    """Sembolün LLM-çıkarımlı 8-K olay tablosu (filing-tarihli, cache'li: bellek + disk).

    İndeks = datetime.date (filing günü). None → CIK yok / 8-K yok / istemci yok / ağ hatası."""
    s = symbol.upper()
    if not refresh and s in _events_cache:
        return _events_cache[s]
    df = _build_events(s, max_filings=max_filings)
    _events_cache[s] = df
    return df


# ---------------------------------------------------------------------------
# SAF karar — önceki bar materyal bir 8-K olay günü müydü? (sentiment/insider deseni)
# ---------------------------------------------------------------------------
def llm_event_signal(events: pd.DataFrame | None, prev_date,
                     *, score_min: float = 0.4, allow_short: bool = True
                     ) -> tuple[str, list[str]]:
    """SAF karar: prev_date'te materyal bir 8-K çıkarıldı mı?

    Dönüş: ('LONG'|'SHORT'|'WAIT', reasons). score = signed materyallik×güven ∈[-1,+1].
    boğa & score≥score_min → LONG; ayı & score≤-score_min → SHORT (allow_short).
    KENAR İDDİASI YOK — eşikler research/llm_event_test.py ile ölçülür."""
    if events is None or events.empty:
        return "WAIT", ["NO_LLM_EVENT_DATA"]
    if prev_date not in events.index:
        return "WAIT", ["NO_LLM_EVENT"]
    row = events.loc[prev_date]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    score = float(row.get("score", 0.0))
    etype = str(row.get("event_type", "none"))
    if score >= score_min:
        return "LONG", [f"LLM_EVENT_BULL_{etype}_{score:+.2f}", f"filed_{prev_date}"]
    if allow_short and score <= -score_min:
        return "SHORT", [f"LLM_EVENT_BEAR_{etype}_{score:+.2f}", f"filed_{prev_date}"]
    return "WAIT", [f"LLM_EVENT_WEAK_{etype}_{score:+.2f}"]


# ---------------------------------------------------------------------------
# TANI: aynı dosyalama setini 3 modelle çıkar → anlaşma + maliyet (model seçimi için)
# ---------------------------------------------------------------------------
def compare_models(symbol: str, *, models: list[str] | None = None,
                   max_filings: int = 10) -> None:
    """Aynı 8-K setini Haiku/Sonnet/Opus ile çıkar; yön+tip anlaşması ve gerçek maliyeti
    raporla. Cache'i KİRLETMEZ (doğrudan _extract_with_usage). Amaç: "ucuz model yeter mi?"
    sorusunu kafadan değil ÖLÇEREK cevaplamak — anlaşma yüksekse Haiku'da kal.

    NOT: bu mod 8-K başına |models| çağrı yapar (cache'siz) → maliyetli; max_filings'i küçük tut."""
    models = models or list(MODEL_PRICES)
    if _get_client() is None:
        print("ANTHROPIC_API_KEY yok → karşılaştırma çalışmaz.")
        return
    cik = ticker_to_cik(symbol)
    if cik is None:
        print(f"{symbol}: CIK bulunamadı.")
        return
    filings = _iter_8k(cik)
    if not filings:
        print(f"{symbol}: 8-K bulunamadı.")
        return
    filings.sort(key=lambda t: t[0], reverse=True)
    filings = filings[:max_filings] if max_filings > 0 else filings

    costs = {m: 0.0 for m in models}
    extracts: dict[str, list[EventExtract]] = {m: [] for m in models}
    dir_full_agree = type_full_agree = comparable = 0

    print(f"=== MODEL KARŞILAŞTIRMA | {symbol} | {len(filings)} dosyalama | {', '.join(models)} ===\n")
    print(f"  {'tarih':<11} " + " ".join(f"{m.split('-')[1][:3]:>6}" for m in models) + "   anlaşma")
    for fdate, acc, doc in filings:
        text = _fetch_8k_text(cik, acc, doc)
        if not text:
            continue
        per: dict[str, EventExtract] = {}
        for m in models:
            ex, usage = _extract_anthropic(text, m)   # compare = Claude'a özel tanı
            if ex is None:
                continue
            extracts[m].append(ex)
            pin, pout = MODEL_PRICES.get(m, (0.0, 0.0))
            costs[m] += usage.get("input", 0) / 1e6 * pin + usage.get("output", 0) / 1e6 * pout
            per[m] = ex
        if len(per) == len(models):                      # tüm modeller döndüyse karşılaştır
            comparable += 1
            dirs = {ex.direction for ex in per.values()}
            types = {ex.event_type for ex in per.values()}
            if len(dirs) == 1:
                dir_full_agree += 1
            if len(types) == 1:
                type_full_agree += 1
            mark = "✓" if len(dirs) == 1 else "✗"
            print(f"  {fdate:<11} " +
                  " ".join(f"{per[m].direction[:3]:>6}" for m in models) + f"   {mark}")

    print(f"\n{'='*60}")
    if comparable:
        print(f"Karşılaştırılabilir dosyalama: {comparable}")
        print(f"  YÖN tam-anlaşma (3/3 aynı):  {dir_full_agree}/{comparable} "
              f"= %{100*dir_full_agree/comparable:.0f}")
        print(f"  TİP tam-anlaşma (3/3 aynı):  {type_full_agree}/{comparable} "
              f"= %{100*type_full_agree/comparable:.0f}")
    print("\nGERÇEK MALİYET (bu çalıştırma):")
    for m in models:
        n = len(extracts[m])
        per1 = costs[m] / n if n else 0.0
        print(f"  {m:<20} {n:>3} çağrı  toplam ${costs[m]:.4f}  (8-K başı ${per1:.5f})")
    print(f"{'='*60}")
    print("Yorum: yön-anlaşması yüksekse (≳%85) çıkarım MODELDEN bağımsız → en ucuzu (Haiku)")
    print("kullan; LLM_EVENT_MODEL=claude-haiku-4-5. Asıl soru model değil, sinyalde KENAR")
    print("var mı — research/llm_event_test.py (§13.5/§16). Düşük anlaşma → yalnız o zaman Opus.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LLM (Claude) 8-K olay-çıkarımı tablosu")
    ap.add_argument("symbol")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--max-filings", type=int, default=20,
                    help="işlenecek en yeni N dosyalama (0=hepsi; maliyet bütçesi)")
    ap.add_argument("--compare-models", action="store_true",
                    help="aynı seti Haiku/Sonnet/Opus ile çıkar → anlaşma + maliyet tanısı")
    args = ap.parse_args()
    if args.compare_models:
        compare_models(args.symbol, max_filings=args.max_filings)
        sys.exit(0)
    ev = llm_events(args.symbol, refresh=args.refresh, max_filings=args.max_filings)
    if ev is None or ev.empty:
        print(f"{args.symbol}: LLM olayı yok (CIK/8-K/ANTHROPIC_API_KEY/ağ).")
        if not ANTHROPIC_API_KEY:
            print("NOT: ANTHROPIC_API_KEY .env'de yok → LLM çıkarımı devre-dışı.")
    else:
        material = ev[ev["score"].abs() >= 0.4]
        print(f"{args.symbol}: {len(ev)} olay günü, {len(material)} materyal "
              f"(CIK={ticker_to_cik(args.symbol)}, model={LLM_EVENT_MODEL})")
        print(material.tail(15).to_string())
