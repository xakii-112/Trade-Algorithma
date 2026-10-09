"""consolidate.py testleri — deterministik epizot kümeleme + semantik bellek konsolidasyonu.

Ağ/gerçek dosya YOK: hem episodic.EPISODES_FILE hem consolidate.SEMANTIC_MEMORY_FILE
tmp_path'e monkeypatch'lenir (episodic.py test deseniyle aynı — consolidate() episodic'in
GERÇEK okuma API'sini [recent_episodes/append_episode] kullandığından ikisi de yönlenmeli).
"""
from __future__ import annotations

import json

import pytest

from nihai.ai import consolidate, episodic


@pytest.fixture(autouse=True)
def _tmp_files(tmp_path, monkeypatch):
    monkeypatch.setattr(episodic, "EPISODES_FILE", tmp_path / "ai_episodes.jsonl")
    monkeypatch.setattr(consolidate, "SEMANTIC_MEMORY_FILE", tmp_path / "ai_semantic_memory.jsonl")
    return tmp_path


def _add(n, kind="ai_decision", tags=None, features=None, win=None):
    """n adet epizot ekler (varsayılan: AYNI özelliklerle — deterministik/gürültüsüz test
    verisi, tek kümeye doğal olarak birleşsinler diye); `win` verilmişse HEPSİ o sonuçla
    çözülür (None → çözülmemiş kalır)."""
    tags = list(tags) if tags is not None else ["event_breakout"]
    eps = []
    for _ in range(n):
        feats = dict(features or {"ret_20": 0.01, "rsi": 55.0})
        ep = episodic.append_episode(kind, features=feats, tags=list(tags), symbol="AAPL")
        if win is not None:
            episodic.resolve_episode(ep.id, {"win": win})
        eps.append(ep)
    return eps


def _lines() -> list[str]:
    if not consolidate.SEMANTIC_MEMORY_FILE.exists():
        return []
    return consolidate.SEMANTIC_MEMORY_FILE.read_text(encoding="utf-8").strip().splitlines()


# ---------------------------------------------------------------------------
# consolidate() — temel akış / min_cluster eşiği
# ---------------------------------------------------------------------------
def test_consolidate_bos_havuz_sifir_rapor():
    rapor = consolidate.consolidate()
    assert rapor == {
        "n_episodes": 0, "clusters_found": 0, "clusters_qualifying": 0,
        "clusters_written": 0, "clusters_skipped_existing": 0,
    }
    assert not consolidate.SEMANTIC_MEMORY_FILE.exists()


def test_consolidate_min_cluster_esigini_gecmeyen_kume_yazilmaz():
    _add(3, tags=["nadir_desen"])
    rapor = consolidate.consolidate(min_cluster=5)
    assert rapor["clusters_found"] == 1
    assert rapor["clusters_qualifying"] == 0
    assert rapor["clusters_written"] == 0
    assert not consolidate.SEMANTIC_MEMORY_FILE.exists()


def test_consolidate_min_cluster_esigini_gecen_kume_yazilir():
    _add(5, tags=["sik_desen"])
    rapor = consolidate.consolidate(min_cluster=5)
    assert rapor["clusters_qualifying"] == 1
    assert rapor["clusters_written"] == 1
    assert consolidate.SEMANTIC_MEMORY_FILE.exists()
    assert len(_lines()) == 1


# ---------------------------------------------------------------------------
# Determinizm
# ---------------------------------------------------------------------------
def test_cluster_episodes_determinizm():
    _add(6, tags=["desen_a"])
    episodes = episodic.recent_episodes(limit=10_000)
    k1 = consolidate._cluster_episodes(episodes)
    k2 = consolidate._cluster_episodes(episodes)
    assert [[ep.id for ep in c] for c in k1] == [[ep.id for ep in c] for c in k2]


# ---------------------------------------------------------------------------
# İdempotens
# ---------------------------------------------------------------------------
def test_consolidate_idempotent_ikinci_kosu_satir_eklemez():
    _add(5, tags=["desen_b"])
    ilk = consolidate.consolidate(min_cluster=5)
    assert ilk["clusters_written"] == 1
    n_ilk = len(_lines())

    ikinci = consolidate.consolidate(min_cluster=5)
    assert ikinci["clusters_written"] == 0
    assert ikinci["clusters_skipped_existing"] == 1
    assert len(_lines()) == n_ilk   # dosya BÜYÜMEDİ


def test_consolidate_yeni_epizot_eklenince_yeni_satir_yazar():
    """İdempotens 'asla yazma' demek değildir: küme içeriği (n) GERÇEKTEN değişirse
    yeni bir özet satırı eklenir (append-only felsefesi — episodic.py'nin resolution
    satırı deseniyle aynı ruh)."""
    _add(5, tags=["desen_c"])
    consolidate.consolidate(min_cluster=5)
    n_once = len(_lines())

    _add(3, tags=["desen_c"])   # aynı kümeye 3 epizot daha → n: 5→8
    rapor = consolidate.consolidate(min_cluster=5)
    assert rapor["clusters_written"] == 1
    assert len(_lines()) == n_once + 1


# ---------------------------------------------------------------------------
# Atomik yazım / içerik doğruluğu
# ---------------------------------------------------------------------------
def test_consolidate_atomik_yazim_dosya_icerigi_dogru():
    _add(5, tags=["desen_d"], win=True)
    consolidate.consolidate(min_cluster=5)
    satirlar = _lines()
    assert len(satirlar) == 1
    rec = json.loads(satirlar[0])
    assert {"pattern", "n", "isabet", "ts", "n_resolved", "feature_centroid"}.issubset(rec)
    assert rec["n"] == 5
    assert rec["isabet"] == pytest.approx(1.0)


def test_consolidate_pattern_alani_kind_ve_tags_icerir():
    _add(5, kind="ai_decision", tags=["event_breakout", "long_bias"])
    consolidate.consolidate(min_cluster=5)
    rec = json.loads(_lines()[0])
    assert "ai_decision" in rec["pattern"]
    assert "event_breakout" in rec["pattern"]
    assert "long_bias" in rec["pattern"]


def test_consolidate_isabet_none_cozulmemis_epizotlarda():
    _add(5, tags=["desen_e"])   # win=None → hiç çözülmedi
    consolidate.consolidate(min_cluster=5)
    rec = json.loads(_lines()[0])
    assert rec["isabet"] is None
    assert rec["n_resolved"] == 0


def test_consolidate_isabet_karisik_sonuc_orani_dogru():
    eps = _add(5, tags=["desen_f"])
    for i, ep in enumerate(eps):
        episodic.resolve_episode(ep.id, {"win": i < 2})   # 2 kazanç / 5
    consolidate.consolidate(min_cluster=5)
    rec = json.loads(_lines()[0])
    assert rec["isabet"] == pytest.approx(0.4)
    assert rec["n_resolved"] == 5


# ---------------------------------------------------------------------------
# semantic_recall
# ---------------------------------------------------------------------------
def test_semantic_recall_bos_dosya_bos_liste():
    assert consolidate.semantic_recall({"ret_20": 0.01}) == []


def test_semantic_recall_bos_features_bos_liste():
    _add(5, tags=["desen_g"])
    consolidate.consolidate(min_cluster=5)
    assert consolidate.semantic_recall({}) == []


def test_semantic_recall_siralama_en_yakin_once():
    _add(5, tags=["yakin"], features={"ret_20": 0.01, "rsi": 55.0})
    _add(5, tags=["uzak"], features={"ret_20": 5.0, "rsi": 95.0})
    consolidate.consolidate(min_cluster=5)
    sonuc = consolidate.semantic_recall({"ret_20": 0.01, "rsi": 55.0}, k=2)
    assert len(sonuc) == 2
    assert "yakin" in sonuc[0]["pattern"]


def test_semantic_recall_k_limitler():
    for i in range(3):
        _add(5, tags=[f"desen_{i}"], features={"ret_20": float(i), "rsi": 50.0 + i})
    consolidate.consolidate(min_cluster=5)
    sonuc = consolidate.semantic_recall({"ret_20": 0.0, "rsi": 50.0}, k=2)
    assert len(sonuc) <= 2


# ---------------------------------------------------------------------------
# _cluster_episodes — kova/eşik davranışı
# ---------------------------------------------------------------------------
def test_cluster_episodes_farkli_kind_ayri_kumeler():
    episodic.append_episode("ai_decision", features={"a": 1.0}, tags=["x"])
    episodic.append_episode("gozlem", features={"a": 1.0}, tags=["x"])
    episodes = episodic.recent_episodes(limit=10_000)
    kumeler = consolidate._cluster_episodes(episodes)
    assert len(kumeler) == 2   # farklı kind → farklı kova → farklı küme


def test_cluster_episodes_uzak_ozellikler_esik_disinda_ayri_kumeler():
    for _ in range(3):
        episodic.append_episode("gozlem", features={"a": 1.0, "b": 1.0, "c": 1.0}, tags=["x"])
    for _ in range(3):
        episodic.append_episode("gozlem", features={"a": 100.0, "b": 100.0, "c": 100.0}, tags=["x"])
    episodes = episodic.recent_episodes(limit=10_000)
    kumeler = consolidate._cluster_episodes(episodes)
    # aynı (kind,tags) kovası ama iki UZAK öbek → z-normalize RMS eşiği aşılır → 2 ayrı küme
    assert len(kumeler) == 2
    assert sorted(len(c) for c in kumeler) == [3, 3]


# ---------------------------------------------------------------------------
# Bozuk satır toleransı
# ---------------------------------------------------------------------------
def test_read_all_patterns_bozuk_satir_atlanir():
    consolidate.SEMANTIC_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    consolidate.SEMANTIC_MEMORY_FILE.write_text(
        json.dumps({"pattern": "saglam", "n": 1, "isabet": None, "ts": "x"}) + "\n"
        + "{bozuk json\n", encoding="utf-8")
    kayitlar = consolidate._read_all_patterns()
    assert len(kayitlar) == 1
    assert kayitlar[0]["pattern"] == "saglam"
