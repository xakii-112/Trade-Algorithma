"""nihai.ai.memory testleri — append/last/recent, bozuk satır toleransı, boyut kırpması.

Ağ yok; MEM_FILE tmp_path'e monkeypatch'lenir (üretim data/'sına dokunulmaz).
"""
from __future__ import annotations

import json

import pytest

from nihai.ai import memory


@pytest.fixture()
def mem(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEM_FILE", tmp_path / "ai_memory.jsonl")
    return memory


def test_append_last_recent(mem):
    mem.append("daily_digest", {"resolved": 100})
    mem.append("not", {"x": 1})
    mem.append("daily_digest", {"resolved": 150})

    son = mem.last("daily_digest")
    assert son is not None and son["payload"] == {"resolved": 150}
    # ts UTC ve tz-aware yazılır
    assert str(son["ts"]).endswith("+00:00")

    assert mem.last("yok") is None
    assert [r["kind"] for r in mem.recent()] == ["daily_digest", "not", "daily_digest"]
    assert len(mem.recent(kind="daily_digest")) == 2


def test_payload_kopyalanir(mem):
    p = {"a": 1}
    mem.append("k", p)
    p["a"] = 999  # çağıranın sonradan değiştirmesi kaydı ETKİLEMEMELİ
    assert mem.last("k")["payload"] == {"a": 1}


def test_bozuk_satir_tolere_edilir(mem):
    mem.append("k", {"i": 1})
    with memory.MEM_FILE.open("a", encoding="utf-8") as f:
        f.write("{yarim json satiri\n")
    mem.append("k", {"i": 2})
    kayitlar = mem.recent(kind="k")
    assert [r["payload"]["i"] for r in kayitlar] == [1, 2]


def test_boyut_kirpmasi(mem, monkeypatch):
    monkeypatch.setattr(memory, "MAX_BYTES", 200)
    monkeypatch.setattr(memory, "KEEP_LINES", 3)
    for i in range(12):
        mem.append("k", {"i": i})
    satirlar = memory.MEM_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(satirlar) <= 3
    # En YENİ kayıtlar korunur (eskiler düşer)
    assert json.loads(satirlar[-1])["payload"]["i"] == 11
