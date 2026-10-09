"""Dinamik watchlist testleri — tohum, ekle/çıkar, replace, kalıcılık."""
import importlib

import pytest


@pytest.fixture
def wl(tmp_path, monkeypatch):
    """İzole watchlist modülü: WATCHLIST_FILE tmp'ye, _state sıfır, sabit tohum."""
    import nihai.execution.watchlist as _wl
    monkeypatch.setattr(_wl, "WATCHLIST_FILE", tmp_path / "watchlist.json")
    monkeypatch.setattr(_wl, "WATCHLIST", ["AAPL", "MSFT", "SPY"])
    monkeypatch.setattr(_wl, "_state", None)
    return _wl


def test_seed_from_config(wl):
    assert wl.current() == ["AAPL", "MSFT", "SPY"]
    assert wl.last_scan() is None
    assert wl.meta()["AAPL"]["source"] == "seed"


def test_file_created_on_seed(wl):
    wl.current()  # tetikler
    assert wl.WATCHLIST_FILE.exists()


def test_add_and_dedup(wl):
    assert wl.add("NVDA", score=1.5, metric="expectancy") is True
    assert "NVDA" in wl.current()
    assert wl.meta()["NVDA"]["score"] == 1.5
    # zaten var → False, çift eklenmez
    assert wl.add("nvda") is False
    assert wl.current().count("NVDA") == 1


def test_remove(wl):
    assert wl.remove("MSFT") is True
    assert "MSFT" not in wl.current()
    assert "MSFT" not in wl.meta()
    # yoksa False
    assert wl.remove("ZZZZ") is False


def test_replace_topn_and_scores(wl):
    new = wl.replace(["TSLA", "AMD"],
                     meta_by_sym={"TSLA": {"score": 9.0, "metric": "sharpe"}},
                     source="auto")
    assert new == ["TSLA", "AMD"]
    assert wl.current() == ["TSLA", "AMD"]
    assert wl.meta()["TSLA"]["score"] == 9.0
    assert wl.last_scan() is not None  # mark_scan=True (default)


def test_replace_preserves_added_timestamp(wl):
    wl.add("NVDA")
    first_added = wl.meta()["NVDA"]["added"]
    # NVDA'yı içeren yeni replace → ilk eklenme zamanı korunmalı
    wl.replace(["NVDA", "GOOGL"], source="auto")
    assert wl.meta()["NVDA"]["added"] == first_added


def test_replace_dedups_input(wl):
    new = wl.replace(["AAA", "AAA", "BBB", "aaa"], source="auto")
    assert new == ["AAA", "BBB"]


def test_persistence_across_reload(wl, monkeypatch):
    wl.add("NVDA", score=2.0, metric="expectancy")
    # _state'i sıfırla → diskten yeniden yüklemeye zorla
    monkeypatch.setattr(wl, "_state", None)
    assert "NVDA" in wl.current()
    assert wl.meta()["NVDA"]["score"] == 2.0


def test_mark_scan_false_keeps_last_scan(wl):
    wl.replace(["AAA"], source="auto", mark_scan=False)
    assert wl.last_scan() is None


# --- Manuel kontrol: pin / block ---
def test_pin_survives_auto_replace(wl):
    wl.add("NVDA", pin=True)
    assert "NVDA" in wl.pins()
    # oto-tarama tamamen farklı liste yazsa bile pin kalmalı, üstelik başta
    new = wl.replace(["AAA", "BBB"], source="auto")
    assert new[0] == "NVDA"
    assert "NVDA" in new
    assert wl.meta()["NVDA"]["source"] == "manual"


def test_block_excluded_from_auto_replace(wl):
    wl.remove("AAPL", block=True)
    assert "AAPL" in wl.blocks()
    # oto-tarama AAPL'ı geri eklemeye çalışsa bile hariç kalmalı
    new = wl.replace(["AAPL", "TSLA"], source="auto")
    assert "AAPL" not in new
    assert "TSLA" in new


def test_manual_add_overrides_block(wl):
    wl.remove("AAPL", block=True)
    assert "AAPL" in wl.blocks()
    wl.add("AAPL", pin=True)          # manuel ekleme engeli kaldırır
    assert "AAPL" not in wl.blocks()
    assert "AAPL" in wl.pins()


def test_add_idempotent_returns_false(wl):
    assert wl.add("MSFT") is False    # zaten tohumda var, pin değil → değişiklik yok


def test_remove_pin_also_unpins(wl):
    wl.add("NVDA", pin=True)
    wl.remove("NVDA")
    assert "NVDA" not in wl.pins()
    assert "NVDA" not in wl.current()


def test_pins_blocks_persist_across_reload(wl, monkeypatch):
    wl.add("NVDA", pin=True)
    wl.remove("AAPL", block=True)
    monkeypatch.setattr(wl, "_state", None)   # diskten yeniden yükle
    assert "NVDA" in wl.pins()
    assert "AAPL" in wl.blocks()


# --- CLI (agent-native parite) ---
def test_cli_add_pins(wl, capsys):
    rc = wl._cli(["add", "nvda"])
    assert rc == 0
    assert "NVDA" in wl.pins() and "NVDA" in wl.current()
    assert "eklendi" in capsys.readouterr().out


def test_cli_remove_blocks(wl, capsys):
    rc = wl._cli(["remove", "AAPL"])
    assert rc == 0
    assert "AAPL" in wl.blocks() and "AAPL" not in wl.current()


def test_cli_add_no_pin(wl):
    wl._cli(["add", "TSLA", "--no-pin"])
    assert "TSLA" in wl.current()
    assert "TSLA" not in wl.pins()   # geçici, pinlenmedi


def test_cli_list_default(wl, capsys):
    wl._cli([])
    out = capsys.readouterr().out
    assert "Watchlist" in out and "AAPL" in out


def test_risk_mult_roundtrips_through_replace(wl):
    # A5 — risk_mult meta'da korunur, accessor okur; bilinmeyen sembol → 1.0
    wl.replace(["NVDA"], meta_by_sym={"NVDA": {"score": 1.0, "metric": "expectancy",
                                               "risk_mult": 0.5}})
    assert wl.risk_mult("NVDA") == 0.5
    assert wl.risk_mult("MSFT") == 1.0


def test_external_screener_updates_visible_and_manual_controls_preserved(wl):
    import json

    from nihai.atomic_io import atomic_write_text
    wl.add("NVDA", pin=True)
    external = json.loads(wl.WATCHLIST_FILE.read_text(encoding="utf-8"))
    external.update(symbols=["NVDA", "AMD"], blocks=["AAPL"])
    atomic_write_text(wl.WATCHLIST_FILE, json.dumps(external))
    assert wl.current() == ["NVDA", "AMD"]
    assert wl.replace(["AAPL", "MSFT"]) == ["NVDA", "MSFT"]
    assert wl.blocks() == ["AAPL"]
    wl.WATCHLIST_FILE.write_text("broken", encoding="utf-8")
    with pytest.raises(ValueError):
        wl.add("TSLA", pin=True)
    assert wl.WATCHLIST_FILE.read_text(encoding="utf-8") == "broken"
