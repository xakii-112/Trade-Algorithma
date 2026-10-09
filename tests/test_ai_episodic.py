"""episodic.py testleri — epizodik hafıza + LESSONS ders kataloğu."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from nihai.ai import episodic


@pytest.fixture(autouse=True)
def _tmp_file(tmp_path, monkeypatch):
    monkeypatch.setattr(episodic, "EPISODES_FILE", tmp_path / "ai_episodes.jsonl")
    monkeypatch.setattr(episodic, "LESSON_CANDIDATES_FILE", tmp_path / "ai_lesson_candidates.jsonl")
    return tmp_path


def test_append_recent_gidis_donus():
    ep = episodic.append_episode("gozlem", symbol="AAPL",
                                 features={"rsi": 55.0, "ret_5": 0.01, "vol": 0.02},
                                 summary="test epizotu", tags=["deneme"])
    assert len(ep.id) == 12
    got = episodic.recent_episodes(kind="gozlem")
    assert len(got) == 1
    assert got[0].symbol == "AAPL"
    assert got[0].features["rsi"] == 55.0
    assert episodic.recent_episodes(kind="baska") == []


def test_sayisal_olmayan_ozellik_dusurulur():
    ep = episodic.append_episode("gozlem", features={"a": 1, "s": "metin", "b": True, "n": float("nan")})
    assert set(ep.features) == {"a"}          # str/bool/NaN elenir


def test_resolve_birlesimi_son_kazanir():
    ep = episodic.append_episode("karar", features={"a": 1.0, "b": 2.0, "c": 3.0})
    assert episodic.resolve_episode(ep.id, {"win": False}) is True
    assert episodic.resolve_episode(ep.id, {"win": True}) is True
    merged = episodic.recent_episodes(kind="karar")[0]
    assert merged.outcome == {"win": True}    # en son çözüm kazanır
    assert episodic.resolve_episode("yok0000000ok", {"win": True}) is False


def test_similar_bilinen_komsu():
    a = episodic.append_episode("gozlem", features={"x": 1.0, "y": 1.0, "z": 1.0})
    episodic.append_episode("gozlem", features={"x": 10.0, "y": 10.0, "z": 10.0})
    episodic.append_episode("gozlem", features={"x": 1.0, "y": 2.0})   # 2 ortak anahtar → atlanır
    got = episodic.similar_episodes({"x": 1.1, "y": 1.1, "z": 0.9}, k=2)
    assert got and got[0][0].id == a.id
    assert all(len(set(e.features) & {"x", "y", "z"}) >= 3 for e, _ in got)
    assert got[0][1] <= got[-1][1]


def test_similar_bos_durumlar():
    assert episodic.similar_episodes({}) == []
    assert episodic.similar_episodes({"x": 1.0, "y": 1.0, "z": 1.0}) == []   # havuz boş


def test_stats_by_tag():
    e1 = episodic.append_episode("karar", tags=["breakout"], features={"a": 1.0})
    episodic.append_episode("karar", tags=["breakout"], features={"a": 2.0})
    episodic.resolve_episode(e1.id, {"win": True})
    st = episodic.stats_by_tag("breakout")
    assert st["n"] == 2 and st["win"] == 1 and st["loss"] == 0
    assert st["win_rate"] == 1.0
    assert episodic.stats_by_tag("olmayan")["win_rate"] is None


def test_bozuk_satir_atlanir(_tmp_file):
    episodic.append_episode("gozlem", summary="saglam")
    with episodic.EPISODES_FILE.open("a", encoding="utf-8") as f:
        f.write("{yarim json...\n")
    got = episodic.recent_episodes()
    assert len(got) == 1 and got[0].summary == "saglam"


def test_trim_boyut_korumasi(monkeypatch):
    monkeypatch.setattr(episodic, "MAX_BYTES", 200)
    monkeypatch.setattr(episodic, "KEEP_LINES", 3)
    for i in range(10):
        episodic.append_episode("gozlem", summary=f"kayit-{i}")
    lines = episodic.EPISODES_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) <= 4
    assert json.loads(lines[-1])["summary"] == "kayit-9"


def test_lessons_katalogu():
    assert len(episodic.LESSONS) >= 10
    ids = [les.id for les in episodic.LESSONS]
    assert len(ids) == len(set(ids))
    assert all(les.claim and les.guard and les.source for les in episodic.LESSONS)


def test_check_against_lessons():
    hits = episodic.check_against_lessons("tek bölme PF çok iyi görünüyor")
    assert any(les.id == "ACI-3" for les in hits)
    hits2 = episodic.check_against_lessons(["survivorship", "pit"])
    assert any(les.id == "ACI-1" for les in hits2)
    assert episodic.check_against_lessons("") == []
    assert episodic.check_against_lessons("tamamen alakasiz qqqq") == []


# =============================================================================
# v3 (WP-H, docs/AI_V3_DERIN_BEYIN_SPEC.md §3.7) — cosine metriği + unutma + ders-adayı
# =============================================================================

# ---------------------------------------------------------------------------
# similar_episodes(metric=...) — cosine vs rms
# ---------------------------------------------------------------------------
def test_similar_episodes_rms_varsayilan_ile_acik_metric_ayni():
    """metric parametresi verilmeden çağrı == metric='rms' açıkça verilen çağrı (bit-denk)."""
    episodic.append_episode("gozlem", features={"a": 1.0, "b": 2.0, "c": 3.0})
    episodic.append_episode("gozlem", features={"a": 5.0, "b": 6.0, "c": 7.0})
    varsayilan = episodic.similar_episodes({"a": 1.1, "b": 2.1, "c": 2.9})
    acik = episodic.similar_episodes({"a": 1.1, "b": 2.1, "c": 2.9}, metric="rms")
    assert [(ep.id, d) for ep, d in varsayilan] == [(ep.id, d) for ep, d in acik]


def test_similar_episodes_cosine_en_yakin_ayni_epizotu_bulur():
    a = episodic.append_episode("gozlem", features={"x": 1.0, "y": 1.0, "z": 1.0})
    episodic.append_episode("gozlem", features={"x": 10.0, "y": 10.0, "z": 10.0})
    episodic.append_episode("gozlem", features={"x": 1.0, "y": 2.0})   # 2 ortak anahtar → atlanır
    got = episodic.similar_episodes({"x": 1.1, "y": 1.1, "z": 0.9}, k=2, metric="cosine")
    assert got and got[0][0].id == a.id
    assert got[0][1] <= got[-1][1]   # küçük=yakın sıralama sözleşmesi rms ile TUTARLI


def test_similar_episodes_bilinmeyen_metric_valueerror():
    with pytest.raises(ValueError):
        episodic.similar_episodes({"x": 1.0}, metric="euclidean")
    # boş features ile bile ValueError ÖNCE gelir (programcı hatası, veri-yoksulluğu değil)
    with pytest.raises(ValueError):
        episodic.similar_episodes({}, metric="euclidean")


def test_similar_episodes_cosine_regime_tag_epizotu_etiketsizden_ayirir():
    """Aynı özelliklere sahip iki epizottan yalnız biri rejim-etiketi taşıyorsa, cosine
    query'yi (etiketsiz) etiketsiz epizota daha yakın bulmalı — rms bu bilgiyi HİÇ kullanmaz."""
    episodic.append_episode("gozlem", features={"x": 50.0, "y": 60.0, "z": 70.0})   # varyans için
    tagli = episodic.append_episode("gozlem", features={"x": 1.0, "y": 2.0, "z": 3.0}, tags=["up_mid"])
    etiketsiz = episodic.append_episode("gozlem", features={"x": 1.0, "y": 2.0, "z": 3.0})
    rms_sonuc = episodic.similar_episodes({"x": 1.0, "y": 2.0, "z": 3.0}, metric="rms", k=3)
    cosine_sonuc = episodic.similar_episodes({"x": 1.0, "y": 2.0, "z": 3.0}, metric="cosine", k=3)
    d_rms_tagli = next(d for ep, d in rms_sonuc if ep.id == tagli.id)
    d_rms_etiketsiz = next(d for ep, d in rms_sonuc if ep.id == etiketsiz.id)
    assert d_rms_tagli == pytest.approx(d_rms_etiketsiz)   # rms: fark yok
    d_cos_tagli = next(d for ep, d in cosine_sonuc if ep.id == tagli.id)
    d_cos_etiketsiz = next(d for ep, d in cosine_sonuc if ep.id == etiketsiz.id)
    assert d_cos_etiketsiz < d_cos_tagli   # cosine: etiketsiz query'ye daha yakın


def test_similar_episodes_cosine_sonuc_bilgisini_kullanir():
    episodic.append_episode("gozlem", features={"x": 50.0, "y": 60.0, "z": 70.0})
    cozulmus = episodic.append_episode("gozlem", features={"x": 1.0, "y": 2.0, "z": 3.0})
    episodic.resolve_episode(cozulmus.id, {"win": True})
    cozulmemis = episodic.append_episode("gozlem", features={"x": 1.0, "y": 2.0, "z": 3.0})
    cosine_sonuc = episodic.similar_episodes({"x": 1.0, "y": 2.0, "z": 3.0}, metric="cosine", k=3)
    d_cozulmus = next(d for ep, d in cosine_sonuc if ep.id == cozulmus.id)
    d_cozulmemis = next(d for ep, d in cosine_sonuc if ep.id == cozulmemis.id)
    assert d_cozulmemis < d_cozulmus   # query'nin sonucu da "bilinmiyor" — eşleşen taraf yakın


# ---------------------------------------------------------------------------
# Kosinüs yardımcı fonksiyonları (birim testleri)
# ---------------------------------------------------------------------------
def test_regime_one_hot_eslesme():
    vec = episodic._regime_one_hot(["event_breakout", "up_mid", "long_bias"])
    assert sum(vec) == pytest.approx(1.0)
    idx = episodic._REGIME_LABELS.index("up_mid")
    assert vec[idx] == 1.0


def test_regime_one_hot_eslesme_yok_sifir_vektor():
    vec = episodic._regime_one_hot(["event_breakout", "long_bias"])
    assert vec == tuple(0.0 for _ in episodic._REGIME_LABELS)
    assert episodic._regime_one_hot([]) == tuple(0.0 for _ in episodic._REGIME_LABELS)


def test_outcome_vector_bilinen():
    assert episodic._outcome_vector({"win": True}) == (1.0, 1.0)
    assert episodic._outcome_vector({"win": False}) == (1.0, -1.0)


def test_outcome_vector_bilinmeyen_notr():
    assert episodic._outcome_vector(None) == (0.0, 0.0)
    assert episodic._outcome_vector({"other": 1}) == (0.0, 0.0)
    assert episodic._outcome_vector({"win": "evet"}) == (0.0, 0.0)   # bool değil → nötr


def test_cosine_distance_ozdes_vektor_sifir():
    v = (1.0, 2.0, -3.0)
    assert episodic._cosine_distance(v, v) == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_sifir_vektor_maksimum():
    assert episodic._cosine_distance((0.0, 0.0), (1.0, 1.0)) == 1.0
    assert episodic._cosine_distance((0.0, 0.0), (0.0, 0.0)) == 1.0


def test_cosine_distance_zit_yon_maksimuma_yakin():
    assert episodic._cosine_distance((1.0, 1.0), (-1.0, -1.0)) == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Unutma eğrisi (_forget_score / _maybe_trim)
# ---------------------------------------------------------------------------
def test_forget_score_yas_monoton():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    eski = episodic.Episode(id="a", ts=(now - timedelta(days=60)).isoformat(), kind="k")
    yeni = episodic.Episode(id="b", ts=now.isoformat(), kind="k")
    freq = {("k", ()): 2}
    assert episodic._forget_score(yeni, freq, now) > episodic._forget_score(eski, freq, now)


def test_forget_score_bilgi_degeri_cozulmus_epizotu_yukseltir():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    cozulmus = episodic.Episode(id="a", ts=now.isoformat(), kind="k", outcome={"win": True})
    cozulmemis = episodic.Episode(id="b", ts=now.isoformat(), kind="k", outcome=None)
    freq = {("k", ()): 2}
    assert (episodic._forget_score(cozulmus, freq, now)
            > episodic._forget_score(cozulmemis, freq, now))


def test_forget_score_sik_kalip_yukseltir():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    sik = episodic.Episode(id="a", ts=now.isoformat(), kind="k", tags=["x"])
    nadir = episodic.Episode(id="b", ts=now.isoformat(), kind="k", tags=["y"])
    freq = {("k", ("x",)): 10, ("k", ("y",)): 1}
    assert episodic._forget_score(sik, freq, now) > episodic._forget_score(nadir, freq, now)


def test_maybe_trim_dusuk_skor_once_kirpilir(monkeypatch):
    """Kırpma artık salt-kronolojik DEĞİL: çok eski/değersiz bir kayıt, ondan SONRA eklenmiş
    ama daha az 'değerli' olmayan kayıtlardan önce gider (skor-tabanlı, rastgele değil)."""
    monkeypatch.setattr(episodic, "MAX_BYTES", 10)   # her zaman tetiklensin
    monkeypatch.setattr(episodic, "KEEP_LINES", 2)
    now = datetime.now(UTC)
    eski = json.dumps({"id": "eski", "ts": (now - timedelta(days=200)).isoformat(),
                       "kind": "k", "tags": [], "features": {}, "summary": "eski", "outcome": None})
    orta = json.dumps({"id": "orta", "ts": (now - timedelta(days=1)).isoformat(),
                       "kind": "k", "tags": [], "features": {}, "summary": "orta", "outcome": None})
    yeni = json.dumps({"id": "yeni", "ts": now.isoformat(),
                       "kind": "k", "tags": [], "features": {}, "summary": "yeni", "outcome": None})
    episodic.EPISODES_FILE.parent.mkdir(parents=True, exist_ok=True)
    episodic.EPISODES_FILE.write_text(eski + "\n" + orta + "\n" + yeni + "\n", encoding="utf-8")
    episodic._maybe_trim()
    kalan = episodic.EPISODES_FILE.read_text(encoding="utf-8").strip().splitlines()
    ozetler = [json.loads(x)["summary"] for x in kalan]
    assert "eski" not in ozetler   # en düşük skorlu (en yaşlı + çözülmemiş) önce gitti
    assert set(ozetler) == {"orta", "yeni"}


# ---------------------------------------------------------------------------
# Ders-adayı üretici (propose_lesson_candidate)
# ---------------------------------------------------------------------------
def test_propose_lesson_candidate_dosyaya_yazar_ve_katalogu_degistirmez():
    onceki_uzunluk = len(episodic.LESSONS)
    rec = episodic.propose_lesson_candidate(
        "Drift tespit edildi", "Brier 0.30→0.05", "learning.page_hinkley")
    assert rec["status"] == "candidate"
    assert episodic.LESSON_CANDIDATES_FILE.exists()
    satirlar = episodic.LESSON_CANDIDATES_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(satirlar) == 1
    assert json.loads(satirlar[0])["claim"] == "Drift tespit edildi"
    assert len(episodic.LESSONS) == onceki_uzunluk   # KATALOĞA OTOMATİK TERFİ YOK


def test_propose_lesson_candidate_ard_arda_append_eder():
    episodic.propose_lesson_candidate("A", "kanit-1", "src1")
    episodic.propose_lesson_candidate("B", "kanit-2", "src2")
    satirlar = episodic.LESSON_CANDIDATES_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(satirlar) == 2
    assert [json.loads(s)["claim"] for s in satirlar] == ["A", "B"]


def test_propose_lesson_candidate_alanlari_dogru():
    rec = episodic.propose_lesson_candidate("iddia", "kanit", "kaynak")
    assert set(rec) == {"id", "ts", "claim", "evidence", "source", "status"}
    assert len(rec["id"]) == 12
