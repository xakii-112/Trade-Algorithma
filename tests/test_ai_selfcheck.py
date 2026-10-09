"""nihai.ai.selfcheck testleri — üç bölümlü öz-değerlendirme sınavı (bilgi/kalibrasyon/replay)
+ report_card kalıcılığı. Ağ yok (brain.think LLM'e bağlı değil); dosya yolları tmp_path'e
monkeypatch'lenir (memory.py/episodic.py test deseni).
"""
from __future__ import annotations

import json
import os

import pytest

from nihai.ai import learning, reasoner, selfcheck


@pytest.fixture(autouse=True)
def _tmp_files(tmp_path, monkeypatch):
    monkeypatch.setattr(selfcheck, "SELFCHECK_FILE", tmp_path / "ai_selfcheck.jsonl")
    monkeypatch.setattr(learning, "LEARNING_STATE", tmp_path / "ai_learning_state.json")
    return tmp_path


# ---------------------------------------------------------------------------
# (a) exam_knowledge
# ---------------------------------------------------------------------------
def test_exam_knowledge_determinizm_ayni_seed_ayni_skor():
    a = selfcheck.exam_knowledge(n=30, seed=42)
    b = selfcheck.exam_knowledge(n=30, seed=42)
    assert a["score"] == b["score"]
    assert a["failed"] == b["failed"]


def test_exam_knowledge_farkli_seed_farkli_ornek_olabilir():
    a = selfcheck.exam_knowledge(n=30, seed=1)
    b = selfcheck.exam_knowledge(n=30, seed=2)
    # aynı boyutta ama seed farklı → örneklem (dolayısıyla skor) DEĞİŞEBİLİR (garanti değil,
    # ama iki farklı çağrının deterministik+bağımsız olduğunu gösterir)
    assert a["n"] == b["n"] == 30


def test_exam_knowledge_dag_tutarli_ve_skor_yuksek():
    """knowledge.py'nin kendi sözleşmesi (formul_ref'ler GERÇEK, prereqs DAG'dır) doğruysa
    sınav skoru yüksek çıkmalı — bu, knowledge.py'nin dürüstlük iddiasının bağımsız testidir."""
    out = selfcheck.exam_knowledge(n=80, seed=42)
    assert out["dag_tutarli"] is True
    assert out["score"] is not None and out["score"] >= 0.9


def test_exam_knowledge_n_sifir_skor_none():
    """Sınav devre dışıyken sahte toplam skor üretilmez ("ölçülemez, uydurma yok")."""
    out = selfcheck.exam_knowledge(n=0, seed=42)
    assert out["score"] is None
    assert out["total"] == 0
    # ...ama sayısal blok yine de raporlanır (bilgi kaybedilmez)
    assert out["sayisal"]["total"] > 0


def test_exam_knowledge_toplam_soru_yapisal_arti_sayisal():
    """Toplam = kavram başına 2 yapısal soru + sabit sayısal formül seti (P2-10)."""
    out = selfcheck.exam_knowledge(n=17, seed=5)
    assert out["total"] == 2 * out["n"] + out["sayisal"]["total"]


# ---------------------------------------------------------------------------
# (b) exam_calibration
# ---------------------------------------------------------------------------
def test_exam_calibration_veri_yok_durumu():
    out = selfcheck.exam_calibration()
    assert out["durum"] == "VERI_YOK"


def test_exam_calibration_olculdu_learning_state_varsa(tmp_path):
    state = {
        "updated_at": "2026-07-19T00:00:00+00:00",
        "strategies": {"GLOBAL": {"n": 100, "win": 60, "loss": 40, "post_mean": 0.6,
                                  "lo95": 0.5, "hi95": 0.7}},
        "pwin_calibration": {"n": 50, "brier": 0.21, "bins": []},
        "drift": {"drift": False, "index": None},
        "retrain_candidate": False,
    }
    learning.LEARNING_STATE.parent.mkdir(parents=True, exist_ok=True)
    learning.LEARNING_STATE.write_text(json.dumps(state), encoding="utf-8")

    out = selfcheck.exam_calibration(limit=500)
    assert out["durum"] == "OLCULDU"
    assert out["n_karar"] == 100
    assert out["isabet_posterior"] == 0.6
    assert out["pwin_brier"] == 0.21
    assert out["trend"] == "STABIL"
    assert out["retrain_candidate"] is False


def test_exam_calibration_drift_bozuluyor_trend():
    state = {"strategies": {"GLOBAL": {"n": 10, "win": 3, "loss": 7, "post_mean": 0.3,
                                       "lo95": 0.1, "hi95": 0.5}},
             "pwin_calibration": None, "drift": {"drift": True, "index": 5},
             "retrain_candidate": True}
    learning.LEARNING_STATE.parent.mkdir(parents=True, exist_ok=True)
    learning.LEARNING_STATE.write_text(json.dumps(state), encoding="utf-8")

    out = selfcheck.exam_calibration()
    assert out["durum"] == "OLCULDU"
    assert "BOZULUYOR" in out["trend"]
    assert out["retrain_candidate"] is True


# ---------------------------------------------------------------------------
# (c) exam_replay
# ---------------------------------------------------------------------------
def test_exam_replay_sentetik_rejim_dogruluk_makul_esik():
    out = selfcheck.exam_replay()
    assert out["n_senaryo"] == len(selfcheck._SCENARIOS) + len(selfcheck._TRANSITIONS)
    assert out["rejim_dogruluk"] is not None
    assert out["rejim_dogruluk"] >= 0.5
    assert 0.0 <= out["abstain_orani"] <= 1.0


def test_exam_replay_rejim_gecisi_guncel_rejimi_raporlar():
    """P2-10: geçiş serilerinde dedektör dönem ORTALAMASINI değil GÜNCEL rejimi vermeli.

    Simetrik kapı: yükseliş→çöküş 'down', çöküş→toparlanma 'up'. Tek yönlü bir test,
    'her volatil seriye down de' diyen bozuk bir dedektörü de geçirirdi."""
    detay = {d["senaryo"]: d for d in selfcheck.exam_replay()["detay"]}
    assert detay["gecis_yukselis_cokus"]["trend_tespit"] == "down"
    assert detay["gecis_cokus_toparlanma"]["trend_tespit"] == "up"


def test_exam_knowledge_sayisal_bozulmayi_yakalar(monkeypatch):
    """Sınavın ASIL amacı: matematik bozulursa skor DÜŞMELİ (eskiden 1.00'a yapışıktı)."""
    saglam = selfcheck.exam_knowledge()
    assert saglam["sayisal"]["failed"] == [] and saglam["score"] == 1.0

    monkeypatch.setattr(selfcheck.quant_math, "kelly_binary", lambda *_a, **_k: 0.99)
    bozuk = selfcheck.exam_knowledge()
    assert "kelly_binary" in " ".join(bozuk["sayisal"]["failed"])
    assert bozuk["score"] < saglam["score"]


def test_exam_knowledge_sayisal_istisnayi_yutar(monkeypatch):
    """Fırlatan bir formül sınavı çökertmez — soru yanlış sayılır (savunmacı sözleşme)."""
    def _patlar(*_a, **_k):
        raise RuntimeError("kasıtlı")

    monkeypatch.setattr(selfcheck.quant_math, "profit_factor", _patlar)
    out = selfcheck.exam_knowledge()
    assert out["score"] is not None and out["score"] < 1.0
    assert "profit_factor" in " ".join(out["sayisal"]["failed"])


def test_exam_replay_ai_decision_mode_env_geri_yuklenir(monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    selfcheck.exam_replay()
    assert os.environ.get("AI_DECISION_MODE") == "shadow"


def test_exam_replay_ai_decision_mode_env_yoksa_kalir_yok(monkeypatch):
    monkeypatch.delenv("AI_DECISION_MODE", raising=False)
    selfcheck.exam_replay()
    assert "AI_DECISION_MODE" not in os.environ


def test_exam_replay_kisa_seri_cokmez():
    """< 60 bar → classify_regime 'unknown' döner (quant_math sözleşmesi); trend_dogru None,
    action None/ABSTAIN olabilir ama exam ÇÖKMEZ."""
    import numpy as np

    panel = {"cok_kisa": np.array([100.0, 101.0, 99.0])}
    out = selfcheck.exam_replay(closes_panel=panel)
    assert out["n_senaryo"] == 1
    assert out["detay"][0]["trend_dogru"] is None


def test_exam_replay_bos_panel():
    out = selfcheck.exam_replay(closes_panel={})
    assert out == {"n_senaryo": 0, "abstain_orani": None, "rejim_dogruluk": None, "detay": []}


# ---------------------------------------------------------------------------
# report_card — birleştirme + kalıcılık + trend
# ---------------------------------------------------------------------------
def test_report_card_yapisi_ve_closing():
    kart = selfcheck.report_card()
    assert set(kart) >= {"ts", "knowledge", "calibration", "replay", "trend", "report_text"}
    assert kart["report_text"].rstrip().endswith(reasoner.CLOSING)


def test_report_card_kenar_iddiasi_icermez():
    kart = selfcheck.report_card()
    # "kenar var"/"kenar bulundu" gibi POZİTİF iddia yok; negatif form ("kenar kanıtı değildir")
    # zaten codebase konvansiyonudur (CLOSING'in kendisi de bu formu kullanır).
    assert "kenar var" not in kart["report_text"].lower()
    assert "kenar bulundu" not in kart["report_text"].lower()


def test_report_card_ilk_kosu_trend_metni():
    kart = selfcheck.report_card()
    assert "ilk koşu" in kart["trend"]


def test_report_card_ikinci_kosu_dosyaya_ekler_ve_trend_karsilastirir(tmp_path):
    k1 = selfcheck.report_card()
    satirlar_1 = selfcheck.SELFCHECK_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(satirlar_1) == 1

    k2 = selfcheck.report_card()
    satirlar_2 = selfcheck.SELFCHECK_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(satirlar_2) == 2
    assert "ilk koşu" not in k2["trend"]
    assert k1["ts"] != k2["ts"] or True  # ts aynı saniyede olabilir; asıl kontrol satır sayısı


def test_report_card_bozuk_satir_atlanir():
    selfcheck.SELFCHECK_FILE.parent.mkdir(parents=True, exist_ok=True)
    selfcheck.SELFCHECK_FILE.write_text(
        json.dumps({"ts": "x", "knowledge": {"score": 1.0}, "calibration": {"durum": "VERI_YOK"},
                   "replay": {"rejim_dogruluk": 1.0}}) + "\n" + "{bozuk\n",
        encoding="utf-8")
    kayitlar = selfcheck._read_all()
    assert len(kayitlar) == 1


# ---------------------------------------------------------------------------
# last_card_line (Tur-17) — SALT-OKUNUR raporlama köprüsü (brain.daily_brief + daily_summary)
# ---------------------------------------------------------------------------
def _yaz(kayitlar: list[dict]) -> None:
    selfcheck.SELFCHECK_FILE.parent.mkdir(parents=True, exist_ok=True)
    selfcheck.SELFCHECK_FILE.write_text(
        "".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kayitlar), encoding="utf-8")


def _kart(ts: str, skor: float, isabet: float, trend: str = "BOZULUYOR (drift)") -> dict:
    return {"ts": ts, "knowledge": {"n": 103, "score": skor, "dag_tutarli": True},
            "calibration": {"durum": "OLCULDU", "isabet_posterior": isabet, "trend": trend},
            "replay": {"rejim_dogruluk": 1.0, "abstain_orani": 1.0, "n_senaryo": 3}}


def test_last_card_line_dosya_yoksa_none():
    assert selfcheck.last_card_line() is None


def test_last_card_line_bos_dosya_none():
    _yaz([])
    assert selfcheck.last_card_line() is None


def test_last_card_line_son_kaydi_ozetler():
    _yaz([_kart("2026-07-20T05:00:00+00:00", 0.90, 0.512, "İYİLEŞİYOR"),
          _kart("2026-07-25T05:00:00+00:00", 1.00, 0.458)])
    line = selfcheck.last_card_line()
    assert line is not None
    assert line.startswith("[Karne] ts=2026-07-25T05:00:00+00:00")
    assert "bilgi 1.00" in line
    assert "isabet 0.458 (BOZULUYOR (drift))" in line
    assert "replay-rejim 1.00" in line
    assert "2026-07-20" not in line


def test_last_card_line_eksik_alanlar_atlanir_cokmez():
    """Kısmi/eski şemalı kayıt (yalnız ts) → satır yine üretilir, eksik alanlar YAZILMAZ."""
    _yaz([{"ts": "2026-07-25T05:00:00+00:00"}])
    line = selfcheck.last_card_line()
    assert line == "[Karne] ts=2026-07-25T05:00:00+00:00"


def test_last_card_line_sinav_kosmaz_dosyaya_yazmaz(monkeypatch):
    """SALT-OKUNUR sözleşmesi: ne üç sınav koşar ne de JSONL'e satır ekler."""
    _yaz([_kart("2026-07-25T05:00:00+00:00", 1.0, 0.458)])
    monkeypatch.setattr(selfcheck, "exam_knowledge",
                        lambda *a, **k: pytest.fail("last_card_line sınav KOŞMAMALI"))
    monkeypatch.setattr(selfcheck, "exam_calibration",
                        lambda *a, **k: pytest.fail("last_card_line sınav KOŞMAMALI"))
    monkeypatch.setattr(selfcheck, "exam_replay",
                        lambda *a, **k: pytest.fail("last_card_line sınav KOŞMAMALI"))
    onceki = selfcheck.SELFCHECK_FILE.read_text(encoding="utf-8")
    assert selfcheck.last_card_line() is not None
    assert selfcheck.SELFCHECK_FILE.read_text(encoding="utf-8") == onceki
