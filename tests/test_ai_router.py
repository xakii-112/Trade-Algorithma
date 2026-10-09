"""nihai.ai.router testleri — sağlayıcı zinciri, kota-cooldown, env override, kalıcılık.

Ağ YOK: _call_anthropic/_call_gemini monkeypatch'lenir; anahtarlar sahte env değerleridir.
"""
from __future__ import annotations

import pytest

from nihai.ai import router


@pytest.fixture(autouse=True)
def temiz(tmp_path, monkeypatch):
    """Her test izole: anahtarlar silinir, cooldown sıfırlanır, state tmp'e yönlenir."""
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
              "AI_MODEL_PARSE", "AI_MODEL_SUMMARY", "AI_MODEL_JUDGMENT", "GEMINI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(router, "ROUTER_STATE", tmp_path / "ai_router_state.json")
    router._COOLDOWN.clear()
    yield
    router._COOLDOWN.clear()


def test_anahtar_yoksa_none():
    assert router._provider_chain() == []
    assert router.active_provider() is None
    assert router.complete("summary", "merhaba") is None


def test_zincir_sirasi(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert router._provider_chain() == ["gemini"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert router._provider_chain() == ["anthropic", "gemini"]


def test_model_secimi_ve_override(monkeypatch):
    assert router._model_for("summary", "anthropic") == "claude-haiku-4-5"
    assert router._model_for("judgment", "anthropic") == "claude-sonnet-5"
    # bilinmeyen görev → varsayılan profil (summary)
    assert router._model_for("bilinmeyen", "anthropic") == "claude-haiku-4-5"
    monkeypatch.setenv("AI_MODEL_SUMMARY", "claude-opus-x")
    assert router._model_for("summary", "anthropic") == "claude-opus-x"
    assert router._model_for("summary", "gemini") == "gemini-2.5-flash"
    monkeypatch.setenv("GEMINI_MODEL", "gemini-ozel")
    assert router._model_for("parse", "gemini") == "gemini-ozel"


def test_kota_cooldown_ve_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("GEMINI_API_KEY", "g")

    def patlayan(*a, **k):
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(router, "_call_anthropic", patlayan)
    monkeypatch.setattr(router, "_call_gemini", lambda *a, **k: "tamam")
    assert router.complete("summary", "p") == "tamam"
    assert router._in_cooldown("anthropic")          # kota → cooldown'a girdi
    assert not router._in_cooldown("gemini")

    # İkinci çağrıda anthropic HİÇ denenmemeli (cooldown atlar)
    def cagrilmamali(*a, **k):
        raise AssertionError("cooldown'daki sağlayıcı çağrıldı")

    monkeypatch.setattr(router, "_call_anthropic", cagrilmamali)
    assert router.complete("summary", "p") == "tamam"


def test_kota_disi_hata_cooldown_yapmaz(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setattr(router, "_call_anthropic",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("gecici patlama")))
    monkeypatch.setattr(router, "_call_gemini", lambda *a, **k: "yedek")
    assert router.complete("summary", "p") == "yedek"
    # Anlık hata 6 saatlik cooldown'a SOKMAZ (yalnız o çağrıda sıradakine düşer)
    assert not router._in_cooldown("anthropic")


def test_cooldown_suresi_dolunca_acilir(monkeypatch):
    router._COOLDOWN["anthropic"] = router._now() + 60
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert router.active_provider() is None
    monkeypatch.setattr(router, "_now", lambda: router._COOLDOWN["anthropic"] + 1)
    assert router.active_provider() == "anthropic"


def test_cooldown_kaliciligi(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    router._enter_cooldown("gemini")
    assert router.ROUTER_STATE.exists()
    router._COOLDOWN.clear()
    router._load_state()   # süreç yeniden başlamış gibi
    assert router._in_cooldown("gemini")


def test_bos_yanit_none(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setattr(router, "_call_gemini", lambda *a, **k: None)
    assert router.complete("summary", "p") is None
