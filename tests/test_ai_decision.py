"""nihai.ai.decision testleri — GÖLGE Bayes karar motoru.

Ağ/gerçek dosya YOK: DECISIONS_FILE tmp_path'e monkeypatch'lenir (memory.py deseniyle aynı).
Sınır (boundary) modülleri — learning/episodic/quant_math — kendi fonksiyonları üzerinden
monkeypatch'lenir (test_ai_router.py'nin _call_anthropic/_call_gemini mock deseniyle aynı
felsefe): decision.py'nin KENDİ mantığı, komşu modüllerin gerçek implementasyonundan
bağımsız test edilir. Kelly/rejim fonksiyonları saf/deterministik olduğundan çoğu testte
gerçek quant_math implementasyonu bilerek MOCKLANMADAN bırakılmıştır (uçtan-uca doğrulama).
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from nihai.ai import decision, episodic, learning, quant_math


class _FakeLesson:
    """episodic.Lesson için minimal duck-type stand-in (id/claim/guard yeter)."""

    def __init__(self, id_: str, claim: str, guard: str):
        self.id = id_
        self.claim = claim
        self.guard = guard


def _strong_state(strategy: str = "event_breakout", mean: float = 0.79,
                  lo: float = 0.66, hi: float = 0.89) -> dict:
    """Güçlü öğrenme-durumu prior'u (kabaca 40 kazanç/10 kayıptan türetilmiş Beta posterior'a
    yakın; tam sayısal kesinlik gerekmiyor — yalnızca CI 0.5'i dışlasın diye)."""
    return {"strategies": {strategy: {"n": 50, "win": 40, "loss": 10,
                                       "post_mean": mean, "lo95": lo, "hi95": hi}}}


@pytest.fixture(autouse=True)
def izole(tmp_path, monkeypatch):
    """Her test izole: AI_DECISION_MODE temiz başlar, DECISIONS_FILE tmp'e yönlenir, epizot
    yazımı varsayılan no-op'tur (ihtiyaç halinde testler override eder)."""
    monkeypatch.delenv("AI_DECISION_MODE", raising=False)
    monkeypatch.setattr(decision, "DECISIONS_FILE", tmp_path / "ai_decisions.jsonl")
    monkeypatch.setattr(episodic, "append_episode", lambda **kw: None)
    yield


# ---------------------------------------------------------------------------
# resolve_mode / kip sözleşmesi
# ---------------------------------------------------------------------------
def test_resolve_mode_varsayilan_off(monkeypatch):
    monkeypatch.delenv("AI_DECISION_MODE", raising=False)
    assert decision.resolve_mode() == "off"


def test_resolve_mode_shadow_buyuk_kucuk_harf_ve_bosluk_duyarsiz(monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "SHADOW")
    assert decision.resolve_mode() == "shadow"
    monkeypatch.setenv("AI_DECISION_MODE", "  shadow  ")
    assert decision.resolve_mode() == "shadow"


def test_resolve_mode_gecersiz_deger_off(monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "enforce")
    assert decision.resolve_mode() == "off"
    monkeypatch.setenv("AI_DECISION_MODE", "gölge")
    assert decision.resolve_mode() == "off"


# ---------------------------------------------------------------------------
# Prior / ABSTAIN
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bos_state", [
    {},
    {"strategies": {}},
    {"strategies": {"baska_strateji": {"post_mean": 0.9, "lo95": 0.8, "hi95": 0.95}}},
])
def test_bos_prior_abstain(monkeypatch, bos_state):
    monkeypatch.setattr(learning, "load_state", lambda: bos_state)
    d = decision.decide("AAPL", "event_breakout", {})
    assert d.action == "ABSTAIN"
    assert d.conviction == 0.0
    assert d.kelly_suggest == 0.0
    assert d.prior_mean == pytest.approx(0.5)


def test_guclu_prior_40w10l_yon_ve_conviction_pozitif(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state())
    d = decision.decide("AAPL", "event_breakout", {})
    assert d.action in ("LONG_BIAS", "SHORT_BIAS")
    assert d.conviction > 0.0
    assert d.prior_mean == pytest.approx(0.79)


def test_ci_gate_pwin_ile_asilmaz(monkeypatch):
    """Prior CI 0.5'i kapsıyorsa (belirsiz), p_win ne kadar güçlü olursa olsun ABSTAIN
    korunmalı — adım-5'teki CI testi HER ZAMAN adım-1'in ham CI'ını kullanır (spec §6)."""
    state = {"strategies": {"event_breakout": {"post_mean": 0.5, "lo95": 0.3, "hi95": 0.7}}}
    monkeypatch.setattr(learning, "load_state", lambda: state)
    d = decision.decide("AAPL", "event_breakout", {}, p_win=0.97)
    assert d.action == "ABSTAIN"
    assert d.kelly_suggest == 0.0


# ---------------------------------------------------------------------------
# Kip / kalıcılık (off vs shadow)
# ---------------------------------------------------------------------------
def test_mode_off_varsayilan_dosya_yazmaz(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state())
    d = decision.decide("AAPL", "event_breakout", {})
    assert d.mode == "off"
    assert not decision.DECISIONS_FILE.exists()


def test_mode_off_env_ile_acikca_de_yazmaz(monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state())
    decision.decide("AAPL", "event_breakout", {})
    assert not decision.DECISIONS_FILE.exists()


def test_mode_gecersiz_deger_off_davranir_dosya_yazmaz(monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "enforce")
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state())
    decision.decide("AAPL", "event_breakout", {})
    assert not decision.DECISIONS_FILE.exists()


def test_mode_shadow_dosya_yazar_ve_epizot_cagrilir(monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state())
    epizot_cagrilari = []
    monkeypatch.setattr(episodic, "append_episode", lambda **kw: epizot_cagrilari.append(kw))

    d = decision.decide("AAPL", "event_breakout", {"ret_20": 0.01})

    assert d.mode == "shadow"
    assert decision.DECISIONS_FILE.exists()
    satirlar = decision.DECISIONS_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(satirlar) == 1
    kayit = json.loads(satirlar[0])
    assert kayit["symbol"] == "AAPL"
    assert kayit["strategy"] == "event_breakout"
    assert kayit["action"] == d.action
    assert kayit["mode"] == "shadow"

    assert len(epizot_cagrilari) == 1
    assert epizot_cagrilari[0]["kind"] == "ai_decision"
    assert epizot_cagrilari[0]["symbol"] == "AAPL"


def test_mode_shadow_iken_abstain_da_yazar(monkeypatch):
    """ABSTAIN da bir gözlemdir — shadow modda dosyaya yazılır (yalnız aksiyonlu kararlar değil)."""
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    monkeypatch.setattr(learning, "load_state", lambda: {})
    d = decision.decide("AAPL", "event_breakout", {})
    assert d.action == "ABSTAIN"
    assert decision.DECISIONS_FILE.exists()


# ---------------------------------------------------------------------------
# Kelly
# ---------------------------------------------------------------------------
def test_kelly_abstain_sifir_ve_fonksiyonlar_cagrilmaz(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: {})  # boş prior → ABSTAIN
    cagrilar: list[str] = []
    monkeypatch.setattr(quant_math, "kelly_binary", lambda *a, **k: cagrilar.append("kb") or 0.9)
    monkeypatch.setattr(quant_math, "fractional_kelly", lambda *a, **k: cagrilar.append("fk") or 0.9)

    d = decision.decide("AAPL", "event_breakout", {})

    assert d.action == "ABSTAIN"
    assert d.kelly_suggest == 0.0
    assert cagrilar == []  # ABSTAIN'de kelly fonksiyonları hiç çağrılmamalı


def test_kelly_non_abstain_dogru_cagrilir(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state())
    cagrilar: dict = {}

    def fake_kb(p, wl):
        cagrilar["kb"] = (p, wl)
        return 0.3

    def fake_fk(k, fraction=0.25, cap=0.02):
        cagrilar["fk"] = (k, fraction, cap)
        return 0.075

    monkeypatch.setattr(quant_math, "kelly_binary", fake_kb)
    monkeypatch.setattr(quant_math, "fractional_kelly", fake_fk)

    d = decision.decide("AAPL", "event_breakout", {})

    assert d.action != "ABSTAIN"
    assert d.kelly_suggest == 0.075
    kb_p, kb_wl = cagrilar["kb"]
    assert kb_p == pytest.approx(d.posterior)
    assert kb_wl == 1.0
    assert cagrilar["fk"][0] == 0.3


# ---------------------------------------------------------------------------
# Evidence / p_win harmanı
# ---------------------------------------------------------------------------
def test_evidence_zinciri_dolu_ve_turkce_gerekce_icerir(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state())
    d = decision.decide("AAPL", "event_breakout", {})
    assert len(d.evidence) >= 2
    assert all(isinstance(e, str) and e for e in d.evidence)
    assert any("Prior" in e for e in d.evidence)
    assert any("Karar:" in e for e in d.evidence)


def test_p_win_harmani_dogru_hesaplanir(monkeypatch):
    mean, lo, hi = 0.6, 0.55, 0.65  # dar CI, 0.5 dışında → non-abstain garantili
    monkeypatch.setattr(learning, "load_state", lambda: {
        "strategies": {"event_breakout": {"post_mean": mean, "lo95": lo, "hi95": hi}}
    })
    d = decision.decide("AAPL", "event_breakout", {}, p_win=0.9)
    beklenen = 0.7 * mean + 0.3 * 0.9
    assert d.posterior == pytest.approx(beklenen, abs=1e-9)


def test_p_win_kirpilir(monkeypatch):
    """p_win aralık dışı verilse bile [0.01, 0.99]'a kırpılır (0/1 uçları harmanı bozmaz)."""
    mean, lo, hi = 0.6, 0.55, 0.65
    monkeypatch.setattr(learning, "load_state", lambda: {
        "strategies": {"event_breakout": {"post_mean": mean, "lo95": lo, "hi95": hi}}
    })
    d = decision.decide("AAPL", "event_breakout", {}, p_win=5.0)  # aşırı değer
    beklenen = 0.7 * mean + 0.3 * 0.99
    assert d.posterior == pytest.approx(beklenen, abs=1e-9)


# ---------------------------------------------------------------------------
# Yön (side_hint / momentum fallback)
# ---------------------------------------------------------------------------
def test_side_hint_yon_belirler(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state(mean=0.8, lo=0.7, hi=0.9))

    assert decision.decide("AAPL", "event_breakout", {}, side_hint="SHORT").action == "SHORT_BIAS"
    assert decision.decide("AAPL", "event_breakout", {}, side_hint=-1).action == "SHORT_BIAS"
    assert decision.decide("AAPL", "event_breakout", {}, side_hint="LONG").action == "LONG_BIAS"


def test_momentum_fallback_side_hint_yokken(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state(mean=0.8, lo=0.7, hi=0.9))

    d_neg = decision.decide("AAPL", "event_breakout", {"ret_20": -0.05})
    assert d_neg.action == "SHORT_BIAS"

    d_pos = decision.decide("AAPL", "event_breakout", {"ret_20": 0.05})
    assert d_pos.action == "LONG_BIAS"


# ---------------------------------------------------------------------------
# Rejim uyumu
# ---------------------------------------------------------------------------
def test_rejim_ayari_trend_takip_stratejisi(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state(mean=0.55, lo=0.51, hi=0.59))
    monkeypatch.setattr(
        quant_math, "classify_regime",
        lambda *_a, **_k: {"trend": "up", "vol_bucket": "mid", "mean_reversion": False, "label": "up_mid"},
    )
    d = decision.decide("AAPL", "event_breakout", {}, closes=[1.0] * 60)
    assert d.regime == {"trend": "up", "vol_bucket": "mid", "mean_reversion": False, "label": "up_mid"}
    assert any("Rejim uyumu" in e for e in d.evidence)
    assert d.posterior == pytest.approx(0.57, abs=1e-9)  # 0.55 + 0.02 (trend uyumlu)


def test_rejim_ayari_mean_reversion_stratejisi(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: {
        "strategies": {"mean_reversion": {"post_mean": 0.55, "lo95": 0.51, "hi95": 0.59}}
    })
    monkeypatch.setattr(
        quant_math, "classify_regime",
        lambda *_a, **_k: {"trend": "flat", "vol_bucket": "low", "mean_reversion": True, "label": "flat_low"},
    )
    d = decision.decide("EURUSD", "mean_reversion", {}, closes=[1.0] * 60)
    assert d.posterior == pytest.approx(0.57, abs=1e-9)  # rejim mean-reverting + strateji uyumlu


def test_rejim_yok_regime_none_ayar_yapilmaz(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state(mean=0.55, lo=0.51, hi=0.59))
    d = decision.decide("AAPL", "event_breakout", {})  # closes=None → regime hiç hesaplanmaz
    assert d.regime is None
    assert d.posterior == pytest.approx(0.55, abs=1e-9)


def test_classify_regime_hata_toleransli(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state())
    monkeypatch.setattr(
        quant_math, "classify_regime",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    d = decision.decide("AAPL", "event_breakout", {}, closes=[1.0] * 60)
    assert d.regime is None  # hata → savunmacı düşüş, karar süreci çökmedi


# ---------------------------------------------------------------------------
# Ders uyarıları (şeffaflık, ceza değil)
# ---------------------------------------------------------------------------
def test_ders_uyarilari_pwin_ile_tetiklenir(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state())

    def fake_check(text):
        if "meta_labeling" in text:
            return [_FakeLesson("ACI-2", "Meta-labeling PIT'te kenar yok.", "Kapıyı kör açma.")]
        return []

    monkeypatch.setattr(episodic, "check_against_lessons", fake_check)

    d_with = decision.decide("AAPL", "event_breakout", {}, p_win=0.7)
    assert any("ACI-2" in w for w in d_with.lesson_warnings)
    assert any("ACI DERS" in e for e in d_with.evidence)

    d_without = decision.decide("AAPL", "event_breakout", {})
    assert d_without.lesson_warnings == []


def test_check_against_lessons_hata_toleransli(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state())
    monkeypatch.setattr(
        episodic, "check_against_lessons",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    d = decision.decide("AAPL", "event_breakout", {})
    assert d.lesson_warnings == []  # hata → savunmacı düşüş, karar süreci çökmedi


# ---------------------------------------------------------------------------
# decision_report
# ---------------------------------------------------------------------------
def test_decision_report_bos_kayit_yok():
    assert "kayıt yok" in decision.decision_report()


def test_decision_report_bozuk_satir_tolere_edilir():
    decision.DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with decision.DECISIONS_FILE.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"ts": decision._utc_now_iso(), "action": "ABSTAIN", "conviction": 0.0}) + "\n")
        f.write("{bozuk json satiri\n")
    rapor = decision.decision_report(days=7)
    assert "n=1" in rapor


def test_decision_report_icerik_dogru():
    now = decision._utc_now_iso()
    kayitlar = [
        {"ts": now, "action": "ABSTAIN", "conviction": 0.0, "lesson_warnings": []},
        {"ts": now, "action": "LONG_BIAS", "conviction": 0.4, "lesson_warnings": ["x"]},
        {"ts": now, "action": "LONG_BIAS", "conviction": 0.6, "lesson_warnings": []},
    ]
    decision.DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with decision.DECISIONS_FILE.open("w", encoding="utf-8") as f:
        for k in kayitlar:
            f.write(json.dumps(k) + "\n")
    rapor = decision.decision_report(days=7)
    assert "n=3" in rapor
    assert "abstain %33.3" in rapor
    assert "ders-uyarılı 1/3" in rapor


def test_decision_report_karar_disi_kapanis_cumlesi_icerir():
    now = decision._utc_now_iso()
    decision.DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with decision.DECISIONS_FILE.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now, "action": "ABSTAIN", "conviction": 0.0}) + "\n")
    rapor = decision.decision_report()
    assert "KARAR-DIŞI" in rapor


# =============================================================================
# v3 (WP-D, docs/AI_V3_DERIN_BEYIN_SPEC.md §3.6) — hiyerarşik birleşim + fayda + üç kapı
# =============================================================================

# ---------------------------------------------------------------------------
# Kapı kombinasyonları (a=CI/maliyet, b=fayda, c=conformal)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prior_kw, extra_kw, beklenen_action",
    [
        # hepsi geçer → yön (conformal pasif)
        (dict(mean=0.8, lo=0.7, hi=0.9), {}, "LONG_BIAS"),
        # kapı-a (CI) başarısız → ABSTAIN
        (dict(mean=0.5, lo=0.3, hi=0.7), {}, "ABSTAIN"),
        # kapı-a geçer ama kapı-c (conformal 0.5'i kapsıyor) → ABSTAIN
        (dict(mean=0.8, lo=0.7, hi=0.9), dict(conformal_interval=(0.4, 0.6)), "ABSTAIN"),
        # kapı-a geçer, kapı-c 0.5'i KAPSAMIYOR → yön
        (dict(mean=0.8, lo=0.7, hi=0.9), dict(conformal_interval=(0.55, 0.95)), "LONG_BIAS"),
    ],
)
def test_kapi_kombinasyonlari(monkeypatch, prior_kw, extra_kw, beklenen_action):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state(**prior_kw))
    d = decision.decide("AAPL", "event_breakout", {"ret_20": 0.01}, **extra_kw)
    assert d.action == beklenen_action


def test_kapi_b_fayda_basarisiz_abstain(monkeypatch):
    """CI (kapı-a) geçer ama fayda (kapı-b, E≤0) başarısız → ABSTAIN — v2'de (fayda kapısı
    yokken) bu senaryo yön verirdi; v3 SIKILAŞTIRIR, gevşetmez."""
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state(mean=0.7, lo=0.6, hi=0.8))
    monkeypatch.setattr(decision, "_cvar_penalty", lambda rets: 10.0)
    d = decision.decide("AAPL", "event_breakout", {}, closes=[100.0, 101.0, 99.0] * 21)
    assert d.action == "ABSTAIN"
    assert d.utility is not None and d.utility <= 0


def test_conformal_interval_bozuk_pasif_kalir(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state(mean=0.8, lo=0.7, hi=0.9))
    d = decision.decide("AAPL", "event_breakout", {"ret_20": 0.01},
                        conformal_interval=("bozuk", None))
    assert d.action == "LONG_BIAS"   # kapı-c savunmacı pasif kaldı, diğer kapılar geçti


# ---------------------------------------------------------------------------
# Brier-ağırlık monotonluğu (_source_weight)
# ---------------------------------------------------------------------------
def test_source_weight_brier_monoton():
    iyi_kalibre = {"pwin": {"brier": 0.05, "n": 100}}
    kotu_kalibre = {"pwin": {"brier": 0.22, "n": 100}}
    w_iyi, b_iyi = decision._source_weight(iyi_kalibre, "pwin", decision._PWIN_WEIGHT)
    w_kotu, b_kotu = decision._source_weight(kotu_kalibre, "pwin", decision._PWIN_WEIGHT)
    assert w_iyi > w_kotu   # düşük Brier (iyi kalibrasyon) → yüksek ağırlık
    assert b_iyi == pytest.approx(0.05)
    assert b_kotu == pytest.approx(0.22)


def test_source_weight_taban_01():
    w, b = decision._source_weight({"pwin": {"brier": 0.9}}, "pwin", decision._PWIN_WEIGHT)
    assert w == pytest.approx(decision._BRIER_WEIGHT_FLOOR)   # çok kötü kalibrasyon → taban
    assert b == pytest.approx(0.9)


@pytest.mark.parametrize("sources", [None, {}, {"prior": {}}, {"prior": {"brier": "bozuk"}}])
def test_source_weight_eksik_brier_fallback(sources):
    w, b = decision._source_weight(sources, "prior", decision._PRIOR_WEIGHT)
    assert w == decision._PRIOR_WEIGHT
    assert b is None


def test_sources_alaniyla_uctan_uca_agirlik_degisir(monkeypatch):
    """learning.load_state()['sources'] doluysa ağırlıklar Brier'e göre kayar — çok-iyi-
    kalibre bir pwin kaynağı, sabit %30'dan çok daha yüksek ağırlık alabilir."""
    state = {
        "strategies": {"event_breakout": {"post_mean": 0.5, "lo95": 0.3, "hi95": 0.7}},
        "sources": {"prior": {"brier": 0.24, "n": 10}, "pwin": {"brier": 0.02, "n": 500}},
    }
    monkeypatch.setattr(learning, "load_state", lambda: state)
    d = decision.decide("AAPL", "event_breakout", {}, p_win=0.9)
    beklenen = (0.01 * 0.5 + 0.23 * 0.9) / 0.24
    assert d.posterior == pytest.approx(beklenen, abs=1e-6)
    assert (0.23 / 0.24) > decision._PWIN_WEIGHT


def test_sources_bos_dict_sabit_agirliga_denk(monkeypatch):
    """sources anahtarı VAR ama BOŞ ({}) — v2 sabit ağırlığına SAYISAL DENK davranmalı
    (yalnız sources'ın TAMAMEN yok olması değil, boş olması da aynı degrade'i tetikler)."""
    state = {
        "strategies": {"event_breakout": {"post_mean": 0.7, "lo95": 0.6, "hi95": 0.8}},
        "sources": {},
    }
    monkeypatch.setattr(learning, "load_state", lambda: state)
    d = decision.decide("AAPL", "event_breakout", {}, p_win=0.55)
    beklenen = decision._PRIOR_WEIGHT * 0.7 + decision._PWIN_WEIGHT * 0.55
    assert d.posterior == pytest.approx(beklenen, abs=1e-9)


# ---------------------------------------------------------------------------
# neural_pwin — std büyüdükçe katkı kısılması
# ---------------------------------------------------------------------------
def test_neural_std_arttikca_katki_azalir(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: {
        "strategies": {"event_breakout": {"post_mean": 0.5, "lo95": 0.3, "hi95": 0.7}}
    })
    d_kesin = decision.decide("AAPL", "event_breakout", {}, neural_pwin=(0.9, 0.01))
    d_orta = decision.decide("AAPL", "event_breakout", {}, neural_pwin=(0.9, 0.15))
    d_belirsiz = decision.decide("AAPL", "event_breakout", {}, neural_pwin=(0.9, 5.0))
    assert d_kesin.posterior > d_orta.posterior > d_belirsiz.posterior
    assert d_kesin.posterior > 0.5   # kesin (düşük-std) neural, nötr 0.5 prior'ı belirgin çekti


def test_neural_pwin_bozuk_savunmaci_atlanir(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state(mean=0.5, lo=0.3, hi=0.7))
    d = decision.decide("AAPL", "event_breakout", {}, neural_pwin=("bozuk", "veri"))
    assert d.posterior == pytest.approx(0.5, abs=1e-9)   # neural kaynağı atlandı, yalnız prior


# ---------------------------------------------------------------------------
# PARİTE (spec §3.6 zorunlu şart): sources yok + neural_pwin=None → v2 ile SAYISAL DENK
# ---------------------------------------------------------------------------
def test_parite_v2_sadece_prior(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state(mean=0.55, lo=0.51, hi=0.59))
    d = decision.decide("AAPL", "event_breakout", {})
    assert d.posterior == pytest.approx(0.55, abs=1e-9)   # v2: p_win yokken post=prior_mean


def test_parite_v2_prior_ve_pwin(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state(mean=0.7, lo=0.6, hi=0.8))
    d = decision.decide("AAPL", "event_breakout", {"ret_20": 0.01}, p_win=0.55)
    # v2 FORMÜLÜ dondurulmuş "oracle" olarak burada — decision.py'nin iç yolundan bağımsız:
    beklenen = decision._PRIOR_WEIGHT * 0.7 + decision._PWIN_WEIGHT * 0.55
    assert d.posterior == pytest.approx(beklenen, abs=1e-9)
    assert d.action == "LONG_BIAS"


def test_parite_v2_rejim_ayariyla(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state(mean=0.55, lo=0.51, hi=0.59))
    monkeypatch.setattr(
        quant_math, "classify_regime",
        lambda *_a, **_k: {"trend": "up", "vol_bucket": "mid", "mean_reversion": False, "label": "up_mid"},
    )
    d = decision.decide("AAPL", "event_breakout", {}, closes=[1.0] * 60)
    assert d.posterior == pytest.approx(0.57, abs=1e-9)   # v2: 0.55 prior + 0.02 rejim ayarı


# ---------------------------------------------------------------------------
# Sıkılaşma yönü: fayda kapısı yönü ASLA değiştirmez, yalnız ABSTAIN üretebilir
# ---------------------------------------------------------------------------
def test_sikilasma_fayda_kapisi_yonu_asla_degistirmez_sadece_abstain_uretir(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state(mean=0.7, lo=0.6, hi=0.8))
    d_baseline = decision.decide("AAPL", "event_breakout", {"ret_20": 0.01})
    assert d_baseline.action == "LONG_BIAS"

    monkeypatch.setattr(decision, "_cvar_penalty", lambda rets: 10.0)
    d_penalized = decision.decide("AAPL", "event_breakout", {"ret_20": 0.01},
                                  closes=[100.0, 101.0, 99.0] * 21)
    assert d_penalized.action == "ABSTAIN"   # ASLA "SHORT_BIAS" gibi ters bir yön değil


# ---------------------------------------------------------------------------
# Evidence içerik sözleşmesi
# ---------------------------------------------------------------------------
def test_evidence_kaynak_katki_satirlari_ve_kapi_gerekcesi(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state(mean=0.8, lo=0.7, hi=0.9))
    d = decision.decide("AAPL", "event_breakout", {"ret_20": 0.01}, p_win=0.6,
                        neural_pwin=(0.7, 0.1))
    assert any("Kaynak[prior]" in e for e in d.evidence)
    assert any("Kaynak[pwin]" in e for e in d.evidence)
    assert any("Kaynak[neural]" in e for e in d.evidence)
    assert any("Kapı-b fayda" in e for e in d.evidence)


def test_evidence_abstain_kapi_gerekcelerini_listeler(monkeypatch):
    state = {"strategies": {"event_breakout": {"post_mean": 0.5, "lo95": 0.3, "hi95": 0.7}}}
    monkeypatch.setattr(learning, "load_state", lambda: state)
    d = decision.decide("AAPL", "event_breakout", {}, conformal_interval=(0.2, 0.8))
    assert d.action == "ABSTAIN"
    son = d.evidence[-1]
    assert son.startswith("Karar: ABSTAIN (")
    assert "prior CI" in son


# ---------------------------------------------------------------------------
# AI_UTILITY_LAMBDA env — çağrı-anında okunur (resolve_mode deseni)
# ---------------------------------------------------------------------------
def test_ai_utility_lambda_env_okunur(monkeypatch):
    monkeypatch.setattr(learning, "load_state", lambda: _strong_state(mean=0.7, lo=0.6, hi=0.8))
    monkeypatch.setattr(decision, "_estimate_gain_loss", lambda rets: (1.0, 1.0))
    monkeypatch.setattr(decision, "_cvar_penalty", lambda rets: 0.1)
    closes = [100.0, 101.0, 99.0] * 21

    monkeypatch.delenv("AI_UTILITY_LAMBDA", raising=False)
    d_default = decision.decide("AAPL", "event_breakout", {}, closes=closes)   # λ=0.5 varsayılan
    # Tur-92 (BULGU-92-2): kuyruk cezası yalnız kaybeden dalda → E = p·1 − (1−p)·(1 + λ·CVaR)
    p = d_default.posterior
    assert d_default.utility == pytest.approx(p - (1 - p) * (1 + 0.5 * 0.1), abs=1e-9)
    assert d_default.action == "LONG_BIAS"

    monkeypatch.setenv("AI_UTILITY_LAMBDA", "100")
    d_big = decision.decide("AAPL", "event_breakout", {}, closes=closes)
    p = d_big.posterior
    assert d_big.utility == pytest.approx(p - (1 - p) * (1 + 100.0 * 0.1), abs=1e-9)
    assert d_big.action == "ABSTAIN"
    assert d_big.utility < d_default.utility


def test_ai_utility_lambda_env_bozuksa_varsayilana_duser(monkeypatch):
    monkeypatch.setenv("AI_UTILITY_LAMBDA", "not-a-number")
    assert decision._resolve_utility_lambda() == decision._UTILITY_LAMBDA_DEFAULT
    monkeypatch.setenv("AI_UTILITY_LAMBDA", "")
    assert decision._resolve_utility_lambda() == decision._UTILITY_LAMBDA_DEFAULT
    monkeypatch.delenv("AI_UTILITY_LAMBDA", raising=False)
    assert decision._resolve_utility_lambda() == decision._UTILITY_LAMBDA_DEFAULT


# ---------------------------------------------------------------------------
# ShadowDecision alan sırası (JSONL geriye-uyumu)
# ---------------------------------------------------------------------------
def test_shadowdecision_alan_sirasi_korunur():
    isimler = [f.name for f in dataclasses.fields(decision.ShadowDecision)]
    v2_sira = ["ts", "symbol", "strategy", "action", "conviction", "prior_mean",
              "posterior", "evidence", "kelly_suggest", "regime", "lesson_warnings", "mode"]
    assert isimler[:len(v2_sira)] == v2_sira
    assert isimler[len(v2_sira):] == ["utility"]   # yalnız SONA eklendi


# ---------------------------------------------------------------------------
# Fayda-katmanı yardımcı fonksiyonları (birim testleri)
# ---------------------------------------------------------------------------
def test_estimate_gain_loss_none_notr():
    assert decision._estimate_gain_loss(None) == (1.0, 1.0)


def test_estimate_gain_loss_tek_yonlu_notr():
    rets = np.array([0.01, 0.02, 0.005])   # hepsi pozitif → kayıp örneği yok
    assert decision._estimate_gain_loss(rets) == (1.0, 1.0)


def test_estimate_gain_loss_karisik_gercek_deger():
    rets = np.array([0.02, -0.01, 0.04, -0.03])
    gain, loss = decision._estimate_gain_loss(rets)
    assert gain == pytest.approx((0.02 + 0.04) / 2)
    assert loss == pytest.approx((0.01 + 0.03) / 2)


def test_cvar_penalty_none_ise_sifir():
    assert decision._cvar_penalty(None) == 0.0


def test_returns_from_closes_kisa_veri_none():
    assert decision._returns_from_closes([100.0, 101.0]) is None   # <3 nokta
    assert decision._returns_from_closes(None) is None


def test_returns_from_closes_gercek_veri():
    rets = decision._returns_from_closes([100.0, 101.0, 99.0, 102.0])
    assert rets is not None
    assert len(rets) == 3


@pytest.mark.parametrize("std,beklenen", [(0.0, 1.0), (-5.0, 1.0)])
def test_neural_std_confidence_sinir_degerler(std, beklenen):
    assert decision._neural_std_confidence(std) == pytest.approx(beklenen)


def test_neural_std_confidence_monoton():
    c1 = decision._neural_std_confidence(0.01)
    c2 = decision._neural_std_confidence(1.0)
    c3 = decision._neural_std_confidence(10.0)
    assert c1 > c2 > c3
