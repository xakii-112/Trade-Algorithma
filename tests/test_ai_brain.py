"""brain.py testleri — düşünme döngüsü orkestratörü (KARAR-DIŞI/GÖLGE)."""
from __future__ import annotations

import numpy as np
import pytest

from nihai.ai import brain, episodic, learning, memory, reasoner
from nihai.ai import decision as decision_mod


@pytest.fixture()
def _tmp_paths(tmp_path, monkeypatch):
    """Tüm ai durum/veri yollarını geçici dizine yönlendir (üretim data/ korunur)."""
    monkeypatch.setattr(memory, "MEM_FILE", tmp_path / "ai_memory.jsonl")
    monkeypatch.setattr(episodic, "EPISODES_FILE", tmp_path / "ai_episodes.jsonl")
    monkeypatch.setattr(learning, "LEARNING_STATE", tmp_path / "ai_learning_state.json")
    monkeypatch.setattr(decision_mod, "DECISIONS_FILE", tmp_path / "ai_decisions.jsonl")
    monkeypatch.setattr(reasoner, "HISTORY_CSV", tmp_path / "history.csv")
    monkeypatch.setattr(reasoner, "PRED_LOG", tmp_path / "predictions_log.csv")
    monkeypatch.setattr(reasoner, "FEATS_LOG", tmp_path / "live_features_log.csv")
    return tmp_path


def test_demo_yazmasiz_ve_kapanisli(_tmp_paths, monkeypatch):
    """--demo: uçtan uca koşar, kapanış cümlesi + gölge karar satırı içerir, dosya YAZMAZ."""
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    out = brain._demo(seed=7)
    assert brain.CLOSING in out
    assert "Gölge karar" in out
    assert not (_tmp_paths / "ai_decisions.jsonl").exists()
    assert not (_tmp_paths / "ai_memory.jsonl").exists()
    assert not (_tmp_paths / "ai_episodes.jsonl").exists()


def test_demo_deterministik(_tmp_paths, monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    a, b = brain._demo(seed=11), brain._demo(seed=11)

    # ts satırı değişir; karar/rejim satırları aynı olmalı → ts içermeyen satırları kıyasla
    def strip(s: str) -> str:
        return "\n".join(ln for ln in s.splitlines() if not ln.startswith("🧠 Düşünce"))

    assert strip(a) == strip(b)


def test_think_shadow_hafizaya_yazar(_tmp_paths, monkeypatch):
    """shadow modda düşünce digest'i operatör hafızasına düşer (karar dosyası decision testinde)."""
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    closes = 100.0 * np.cumprod(1 + 0.001 * np.ones(120))
    t = brain.think("TST", "event_breakout", features={"ret_5": 0.01, "rsi": 55.0},
                    closes=closes, p_win=0.52)
    assert t.decision is not None
    assert t.narrative and brain.CLOSING in t.narrative
    rec = memory.last("thought_digest")
    assert rec is not None and rec["payload"]["symbol"] == "TST"


def test_think_veri_yokken_cokmez(_tmp_paths, monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    t = brain.think("BOS", "bilinmeyen_strateji")
    assert t.decision is not None
    assert t.risk["vol_yillik"] is None
    assert brain.CLOSING in t.narrative


def test_daily_brief_bos_ortamda_calisir(_tmp_paths, monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    out = brain.daily_brief()
    assert "Quant Beyin" in out
    assert brain.CLOSING in out


def test_risk_block_bilinen_seri():
    closes = np.array([100, 110, 121, 108.9, 119.79, 131.769], dtype=float)
    rb = brain._risk_block(np.concatenate([closes, closes * 1.01, closes * 1.02,
                                           closes * 1.03, closes * 1.04, closes * 1.05]))
    assert rb["max_dd"] is not None and rb["max_dd"] >= 0.0


def test_daily_summary_brain_blogu(_tmp_paths, monkeypatch):
    """--brain köprüsü: öğrenme raporu döner, hata durumunda '' (özet asla düşmez)."""
    from nihai.notifier import daily_summary
    out = daily_summary._brain_block()
    assert "[Öğrenme]" in out            # boş ortamda bile nötr-prior raporu üretir
    monkeypatch.setattr(learning, "learning_report",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kasıtlı")))
    assert daily_summary._brain_block() == ""    # fail-soft


# ---------------------------------------------------------------------------
# v3 (Tur-16): ÖNGÖR + ÖZ-ELEŞTİR + anlamsal HATIRLA köprüleri
# ---------------------------------------------------------------------------
from nihai.ai import consolidate


class _StubNeural:
    """NeuralCapacity duck-parite stub'ı — brain torch'a import bağı almaz, test de almaz."""
    is_fitted = True
    feature_names = ["ret_5", "rsi"]

    def predict_with_uncertainty(self, X, T: int = 20):
        n = np.asarray(X).shape[0]
        return np.full(n, 0.61), np.full(n, 0.05)


def test_think_ongoru_stub_ile(_tmp_paths, monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    t = brain.think("TST", "event_breakout", features={"ret_5": 0.01, "rsi": 55.0},
                    neural_model=_StubNeural())
    assert t.neural_pwin is not None
    assert abs(t.neural_pwin[0] - 0.61) < 1e-9
    assert abs(t.neural_pwin[1] - 0.05) < 1e-9
    assert "ÖNGÖR" in t.narrative


def test_think_bozuk_neural_cokmez(_tmp_paths, monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "off")

    class _Bozuk:
        is_fitted = True
        feature_names = ["ret_5"]

        def predict_with_uncertainty(self, X, T: int = 20):
            raise RuntimeError("kasıtlı")

    t = brain.think("TST", "event_breakout", features={"ret_5": 0.01}, neural_model=_Bozuk())
    assert t.decision is not None
    assert t.neural_pwin is None


def test_think_fit_edilmemis_neural_pasif(_tmp_paths, monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "off")

    class _Fitsiz:
        is_fitted = False
        feature_names = ["ret_5"]

    t = brain.think("TST", "event_breakout", features={"ret_5": 0.01}, neural_model=_Fitsiz())
    assert t.neural_pwin is None


def test_think_oz_elestiri_deterministik(_tmp_paths, monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    t = brain.think("TST", "event_breakout", features={"ret_5": 0.01})
    assert t.weak_links, "öz-eleştiri boş olmamalı"
    assert "Öz-eleştiri" in t.narrative
    assert any("neural" in w for w in t.weak_links)   # model verilmedi → zayıf halka
    t2 = brain.think("TST", "event_breakout", features={"ret_5": 0.01})
    assert t.weak_links == t2.weak_links               # deterministik


def test_think_anlamsal_hatirlama_koprusu(_tmp_paths, monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    monkeypatch.setattr(consolidate, "semantic_recall",
                        lambda features, k=3: [{"pattern": "down_high event_breakout",
                                                "n": 42, "isabet": 0.44}])
    t = brain.think("TST", "event_breakout", features={"ret_5": 0.01})
    assert t.semantic
    assert "down_high" in t.semantic[0]
    assert "Anlamsal hafıza" in t.narrative


def test_daily_brief_kaynak_kalibrasyon_satiri(_tmp_paths, monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    import json as _json
    (_tmp_paths / "ai_learning_state.json").write_text(_json.dumps(
        {"strategies": {}, "sources": {"pwin": {"brier": 0.31, "n": 12}}}), encoding="utf-8")
    out = brain.daily_brief()
    assert "Kaynak kalibrasyonu" in out
    assert "pwin" in out


# ---------------------------------------------------------------------------
# Tur-17 köprüleri: daily_brief KARNE satırı (salt-okunur) + öz-eleştiri LAB önerisi
# ---------------------------------------------------------------------------
def _karne_kaydi(ts: str, skor: float, isabet: float, trend: str = "BOZULUYOR (drift)") -> dict:
    """selfcheck.report_card()'ın JSONL şemasıyla birebir örnek kayıt."""
    return {"ts": ts,
            "knowledge": {"n": 103, "score": skor, "dag_tutarli": True},
            "calibration": {"durum": "OLCULDU", "isabet_posterior": isabet, "trend": trend},
            "replay": {"rejim_dogruluk": 1.0, "abstain_orani": 1.0, "n_senaryo": 3}}


def _yaz_karne(path, kayitlar: list[dict]) -> None:
    import json as _json
    path.write_text("".join(_json.dumps(k, ensure_ascii=False) + "\n" for k in kayitlar),
                    encoding="utf-8")


def test_daily_brief_karne_satiri_son_kaydi_kullanir(_tmp_paths, monkeypatch):
    from nihai.ai import selfcheck
    f = _tmp_paths / "ai_selfcheck.jsonl"
    _yaz_karne(f, [_karne_kaydi("2026-07-20T05:00:00+00:00", 0.90, 0.512, "İYİLEŞİYOR"),
                   _karne_kaydi("2026-07-25T05:00:00+00:00", 1.00, 0.458)])
    monkeypatch.setattr(selfcheck, "SELFCHECK_FILE", f)
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    out = brain.daily_brief()
    assert "[Karne]" in out
    assert "2026-07-25T05:00:00+00:00" in out      # SON kayıt
    assert "0.458" in out and "BOZULUYOR" in out
    assert "2026-07-20" not in out                  # eski kayıt raporda yer almaz


def test_daily_brief_karne_dosyasi_yoksa_satir_hic_eklenmez(_tmp_paths, monkeypatch):
    from nihai.ai import selfcheck
    monkeypatch.setattr(selfcheck, "SELFCHECK_FILE", _tmp_paths / "yok.jsonl")
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    out = brain.daily_brief()
    assert "[Karne]" not in out
    assert brain.CLOSING in out                     # brif yine de tam


def test_daily_brief_karne_bozuk_satiri_atlar(_tmp_paths, monkeypatch):
    """Bozuk JSONL satırı brifi düşürmez; son GEÇERLİ kayıt kullanılır (memory.py deseni)."""
    import json as _json

    from nihai.ai import selfcheck
    f = _tmp_paths / "ai_selfcheck.jsonl"
    f.write_text(_json.dumps(_karne_kaydi("2026-07-24T05:00:00+00:00", 1.0, 0.470)) + "\n"
                 + "{ bozuk json\n" + "\n", encoding="utf-8")
    monkeypatch.setattr(selfcheck, "SELFCHECK_FILE", f)
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    out = brain.daily_brief()
    assert "[Karne]" in out and "0.470" in out


def test_daily_brief_karne_yeni_sinav_kosmaz_ve_dosyaya_yazmaz(_tmp_paths, monkeypatch):
    """SALT-OKUNUR sözleşmesi: brif üretimi `report_card()` ÇAĞIRMAZ (üç sınav koşmak
    pahalı + JSONL'e yeni satır ekler); yalnız var olan son satırı okur."""
    from nihai.ai import selfcheck
    f = _tmp_paths / "ai_selfcheck.jsonl"
    _yaz_karne(f, [_karne_kaydi("2026-07-25T05:00:00+00:00", 1.0, 0.458)])
    monkeypatch.setattr(selfcheck, "SELFCHECK_FILE", f)
    monkeypatch.setattr(selfcheck, "report_card",
                        lambda *a, **k: pytest.fail("daily_brief yeni sınav KOŞMAMALI"))
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    onceki = f.read_text(encoding="utf-8")
    out = brain.daily_brief()
    assert "[Karne]" in out
    assert f.read_text(encoding="utf-8") == onceki   # dosya BÜYÜMEDİ


def test_oz_elestiri_lab_onerisi_zemin_zayifken_gorunur(_tmp_paths, monkeypatch):
    """REGRESYON: öneri 5-madde tavanına kurban gitmemeli. features=None + closes=None
    iken üstteki tanılar zaten 5 maddeyi doldurur; öneri yine de SON madde olarak kalır."""
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    t = brain.think("TST", "event_breakout")
    assert len(t.weak_links) == 5
    assert "lab.run_experiment" in t.weak_links[-1]
    assert "yalnız-analiz" in t.weak_links[-1]


def test_oz_elestiri_lab_onerisi_zemin_saglamken_yok(_tmp_paths, monkeypatch):
    """Özellik + yeterli fiyat serisi varsa (rejim hesaplanıyor) deney önerisi çıkmaz."""
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    closes = 100.0 * np.cumprod(1 + 0.001 * np.ones(120))
    t = brain.think("TST", "event_breakout", features={"ret_5": 0.01, "rsi": 55.0}, closes=closes)
    assert t.regime is not None
    assert not any("lab.run_experiment" in w for w in t.weak_links)


def test_daily_summary_brain_blogunda_karne_satiri(_tmp_paths, monkeypatch):
    """Tur-17: `daily_summary --brain` bölümü karne satırını da taşır (salt-okunur)."""
    from nihai.ai import selfcheck
    from nihai.notifier import daily_summary
    f = _tmp_paths / "ai_selfcheck.jsonl"
    _yaz_karne(f, [_karne_kaydi("2026-07-25T05:00:00+00:00", 1.0, 0.458)])
    monkeypatch.setattr(selfcheck, "SELFCHECK_FILE", f)
    out = daily_summary._brain_block()
    assert "[Öğrenme]" in out                       # mevcut bölüm korunur
    assert "[Karne]" in out and "0.458" in out


def test_daily_summary_brain_blogu_karnesiz_de_calisir(_tmp_paths, monkeypatch):
    """Karne hiç üretilmemişse (dosya yok) satır eklenmez; bölümün geri kalanı sağlam."""
    from nihai.ai import selfcheck
    from nihai.notifier import daily_summary
    monkeypatch.setattr(selfcheck, "SELFCHECK_FILE", _tmp_paths / "yok.jsonl")
    out = daily_summary._brain_block()
    assert "[Öğrenme]" in out
    assert "[Karne]" not in out


def test_daily_summary_brain_blogu_karne_hatasi_bolumu_dusurmez(_tmp_paths, monkeypatch):
    """Fail-soft: karne okuması patlasa bile öğrenme/karar bölümleri kaybolmaz."""
    from nihai.ai import selfcheck
    from nihai.notifier import daily_summary
    monkeypatch.setattr(selfcheck, "last_card_line",
                        lambda: (_ for _ in ()).throw(RuntimeError("kasıtlı")))
    out = daily_summary._brain_block()
    assert "[Öğrenme]" in out
    assert "[Karne]" not in out


def test_oz_elestiri_lab_onerisi_deneyi_CALISTIRMAZ(_tmp_paths, monkeypatch):
    """Anayasa: öneri METİNDİR. brain hiçbir koşulda lab.run_experiment ÇAĞIRMAZ
    (otomatik yürütme yapısal olarak yok — insan/ayrı araç tetikler)."""
    from nihai.ai import lab
    monkeypatch.setattr(lab, "run_experiment",
                        lambda *a, **k: pytest.fail("brain deneyi otomatik ÇALIŞTIRMAMALI"))
    monkeypatch.setenv("AI_DECISION_MODE", "off")
    t = brain.think("TST", "event_breakout")
    assert any("lab.run_experiment" in w for w in t.weak_links)
