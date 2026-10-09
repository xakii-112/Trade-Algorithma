"""
router.py — Model YÖNLENDİRİCİ (FAZ AI, GELISTIRME_RAPORU §1.2). KARAR-DIŞI.

Görev profilleri → model sınıfı (env ile override):
  - "parse"    (ucuz çıkarım işleri)  → haiku-sınıfı   (AI_MODEL_PARSE)
  - "summary"  (günlük özet)          → haiku-sınıfı   (AI_MODEL_SUMMARY)
  - "judgment" (derin yorum)          → sonnet-sınıfı  (AI_MODEL_JUDGMENT)

Sağlayıcı çözümleme sırası (kota-duyarlı fallback zinciri):
  1) ANTHROPIC_API_KEY doluysa   → anthropic
  2) yoksa GEMINI_API_KEY doluysa → gemini (ücretsiz katman; llm_event konvansiyonu)
  3) ikisi de yoksa               → None (çağıran DETERMİNİSTİK fallback kullanır)

`complete(task, prompt)` başarısız/None döndürebilir: bu bir hata değil, sözleşmedir —
çağıran (ör. reasoner) her zaman deterministik metne düşebilmelidir.

429 / kota hatası: o sağlayıcı süreç-içi COOLDOWN'a alınır (varsayılan 6 saat; opsiyonel
kalıcılık `data/ai_router_state.json`, atomik yazım). Zincirdeki sonraki sağlayıcı denenir;
retry fırtınası YOK (her çağrı sağlayıcı başına en fazla bir deneme = en fazla bir fallback).

HTTP: llm_event.py'deki SDK deseni (lazy import, ağ yalnız `_call_*` içinde). Yeni pip
bağımlılığı EKLENMEZ (anthropic / google-genai zaten llm_event tarafından kullanılıyor).
"""
from __future__ import annotations

import json
import os
import time

from loguru import logger

from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR

# Görev profili → Anthropic varsayılan modeli + override env değişkeni.
_TASK_ENV: dict[str, str] = {
    "parse": "AI_MODEL_PARSE",
    "summary": "AI_MODEL_SUMMARY",
    "judgment": "AI_MODEL_JUDGMENT",
}
_MODEL_DEFAULTS: dict[str, str] = {
    "parse": "claude-haiku-4-5",
    "summary": "claude-haiku-4-5",
    "judgment": "claude-sonnet-5",
}
_DEFAULT_TASK = "summary"

# Cooldown: 429/kota'da sağlayıcıyı bu kadar SÜREYLE atla (retry fırtınası önleme).
COOLDOWN_HOURS = float(os.getenv("AI_ROUTER_COOLDOWN_HOURS", "6"))
# Opsiyonel kalıcılık (süreç yeniden başlasa bile 6 saatlik kota-yasağı korunur).
ROUTER_STATE = DATA_DIR / "ai_router_state.json"

# GERÇEK kota/limit işaretçileri (llm_event transient ailesinden). NOT: 5xx/overloaded
# gibi ANLIK yük hataları BURADA YOK — onlar uzun cooldown'a sokulmaz, yalnız o çağrıda
# sıradaki sağlayıcıya düşürülür (geçici blip 6 saat sağlayıcı kaybettirmesin).
_QUOTA_MARKERS = ("429", "resource_exhausted", "quota", "rate_limit",
                  "rate limit", "insufficient_quota", "too many requests")

# Süreç-içi cooldown durumu: sağlayıcı → (bu epoch'a kadar yasak).
_COOLDOWN: dict[str, float] = {}


def _now() -> float:
    """Test'te monkeypatch'lenebilir zaman kaynağı (cooldown determinizmi)."""
    return time.time()


def _gemini_key() -> str:
    """llm_event ile aynı env: GEMINI_API_KEY veya GOOGLE_API_KEY."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def _provider_chain() -> list[str]:
    """Anahtar durumuna göre çözümleme sırası. Boş liste → LLM yok (çağıran deterministik)."""
    chain: list[str] = []
    if os.getenv("ANTHROPIC_API_KEY"):
        chain.append("anthropic")
    if _gemini_key():
        chain.append("gemini")
    return chain


def _model_for(task: str, provider: str) -> str:
    """(task, provider) → model adı. Anthropic'te profil-modeli (env override); Gemini'de
    llm_event konvansiyonu: tek ücretsiz Flash modeli (GEMINI_MODEL), kota-duyarlı."""
    t = task if task in _TASK_ENV else _DEFAULT_TASK
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return os.getenv(_TASK_ENV[t], _MODEL_DEFAULTS[t])


# ---------------------------------------------------------------------------
# Cooldown durumu — süreç-içi + opsiyonel kalıcı (atomik)
# ---------------------------------------------------------------------------
def _load_state() -> None:
    """ai_router_state.json'dan süresi DOLMAMIŞ cooldown'ları tohumla (yok/bozuk → sessiz)."""
    try:
        raw = json.loads(ROUTER_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(raw, dict):
        return
    now = _now()
    for prov, until in raw.items():
        try:
            u = float(until)
        except (TypeError, ValueError):
            continue
        if u > now:
            _COOLDOWN[str(prov)] = u


def _save_state() -> None:
    """Cooldown haritasını atomik yaz (OneDrive/AV kilidine dayanıklı). Hata → sessiz-degrade
    (kalıcılık opsiyonel; süreç-içi cooldown yine de çalışır)."""
    try:
        atomic_write_text(ROUTER_STATE, json.dumps(_COOLDOWN))
    except OSError as e:
        logger.debug(f"[ai.router] cooldown durumu yazılamadı: {e}")


def _in_cooldown(provider: str) -> bool:
    return _COOLDOWN.get(provider, 0.0) > _now()


def _enter_cooldown(provider: str) -> None:
    _COOLDOWN[provider] = _now() + COOLDOWN_HOURS * 3600.0
    _save_state()


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _QUOTA_MARKERS)


# ---------------------------------------------------------------------------
# Sağlayıcı çağrıları (llm_event SDK deseni) — ağ YALNIZ burada; test monkeypatch'ler
# ---------------------------------------------------------------------------
def _call_anthropic(model: str, prompt: str, max_tokens: int, timeout: float) -> str | None:
    """Claude Messages API. Anahtarı ortamdan çözer. Metin bloğu yoksa None."""
    import anthropic
    client = anthropic.Anthropic(timeout=timeout)
    resp = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])
    # getattr: anthropic SDK 1.8'in blok birleşimi `.text`i yalnız TextBlock'ta taşır (mypy daraltması).
    return next((getattr(b, "text", "") for b in resp.content
                 if getattr(b, "type", None) == "text"), "") or None


def _call_gemini(model: str, prompt: str, max_tokens: int, timeout: float) -> str | None:
    """Gemini generate_content (ücretsiz katman). timeout Gemini SDK'sında en-iyi-çaba
    (per-call kısıt sürüm-bağımlı) → burada uygulanmaz; kota koruması cooldown'dadır."""
    from google import genai
    client = genai.Client(api_key=_gemini_key())
    resp = client.models.generate_content(model=model, contents=prompt)
    return (getattr(resp, "text", "") or "").strip() or None


def active_provider() -> str | None:
    """Şu an kullanılabilir (cooldown'da OLMAYAN) ilk sağlayıcı; hiçbiri yoksa None."""
    for p in _provider_chain():
        if not _in_cooldown(p):
            return p
    return None


def complete(task: str, prompt: str, *, max_tokens: int = 512,
             timeout: float = 30.0) -> str | None:
    """Görev-profiline göre LLM'i çağır. None = LLM yok / cooldown / başarısız (çağıran
    DETERMİNİSTİK fallback yapar — bu bir hata değil, sözleşmedir).

    Zincir: anthropic → gemini. 429/kota → o sağlayıcı cooldown + sıradakine düş. Sağlayıcı
    başına en fazla bir deneme (retry fırtınası YOK)."""
    for provider in _provider_chain():
        if _in_cooldown(provider):
            continue
        model = _model_for(task, provider)
        try:
            if provider == "anthropic":
                out = _call_anthropic(model, prompt, max_tokens, timeout)
            else:
                out = _call_gemini(model, prompt, max_tokens, timeout)
        except Exception as e:  # noqa: BLE001 — sağlayıcı SDK'sının her hatası tolere edilir
            if _is_quota_error(e):
                _enter_cooldown(provider)
                logger.warning(
                    f"[ai.router] {provider} kota/429 → {COOLDOWN_HOURS} saat cooldown ({e})")
            else:
                logger.debug(f"[ai.router] {provider} çağrı hatası: {e}")
            continue
        if out:
            return out
    return None


_load_state()
