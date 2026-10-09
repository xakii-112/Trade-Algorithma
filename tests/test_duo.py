"""duo.py testleri — ayrı kasa, doğru strateji modu, paylaşımlı tarama döngüsü.

Ağ yok: scan_symbol ve provider mock'lanır; sadece duo orkestrasyon mantığı
(varyasyon kurulumu, sembol-dış/varyasyon-iç döngü, kalıcılık) sınanır.
"""
from __future__ import annotations

from types import SimpleNamespace

import research.duo as duo


def test_build_variation_ayri_kasa_ve_mod(tmp_path, monkeypatch):
    monkeypatch.setattr(duo, "DATA_DIR", tmp_path)
    v = duo.build_variation("champ", "event_breakout", 25000.0)
    assert v.name == "champ"
    assert v.strategy_mode == "event_breakout"
    assert v.portfolio.equity == 25000.0
    assert v.start_equity == 25000.0
    # kasa kendi dizininde
    assert (tmp_path / "duo" / "champ").is_dir()
    assert v.store.path == tmp_path / "duo" / "champ" / "state.json"
    assert v.equity_csv == tmp_path / "duo" / "champ" / "equity_curve.csv"


def test_iki_varyasyonun_kasalari_ayri(tmp_path, monkeypatch):
    monkeypatch.setattr(duo, "DATA_DIR", tmp_path)
    a = duo.build_variation("champion", "event_breakout", 25000.0)
    b = duo.build_variation("challenger", "trend_pullback", 25000.0)
    # ayrı state nesneleri + ayrı yollar + ayrı portföyler
    assert a.store.path != b.store.path
    assert a.portfolio is not b.portfolio
    assert a.pmgr is not b.pmgr
    assert a.risk_mgr is not b.risk_mgr


def test_run_duo_once_paylasimli_tarama(tmp_path, monkeypatch):
    """sembol-dış/varyasyon-iç döngü: her sembol her iki modla taranır + kasalar yazılır."""
    monkeypatch.setattr(duo, "DATA_DIR", tmp_path)
    monkeypatch.setattr(duo, "INITIAL_EQUITY", 25000.0)
    monkeypatch.setattr(duo, "is_market_open", lambda *a, **k: True)
    monkeypatch.setattr(duo, "get_provider",
                        lambda *a, **k: SimpleNamespace(name="fake_havuz"))  # poz yok → metot çağrılmaz
    monkeypatch.setattr(duo.watchlist, "current", lambda: ["AAA", "BBB"])

    calls: list[tuple[str, str | None]] = []

    def fake_scan(sym, provider, risk, store, breaker, pmgr, strategy_mode=None):
        calls.append((sym, strategy_mode))

    monkeypatch.setattr(duo, "scan_symbol", fake_scan)
    duo.run_duo(once=True)

    # 2 sembol × 2 varyasyon = 4 tarama
    assert len(calls) == 4
    # her iki strateji modu da kullanıldı
    assert {m for _, m in calls} == {"event_breakout", "trend_pullback"}
    # sembol-dış sıra: AAA(champion), AAA(challenger), BBB(champion), BBB(challenger)
    assert [s for s, _ in calls] == ["AAA", "AAA", "BBB", "BBB"]
    # iki kasa da kalıcılaştı
    assert (tmp_path / "duo" / "champion" / "state.json").exists()
    assert (tmp_path / "duo" / "challenger" / "state.json").exists()
    assert (tmp_path / "duo" / "champion" / "equity_curve.csv").exists()


def test_run_duo_market_kapali_tarama_yok(tmp_path, monkeypatch):
    monkeypatch.setattr(duo, "DATA_DIR", tmp_path)
    monkeypatch.setattr(duo, "INITIAL_EQUITY", 25000.0)
    monkeypatch.setattr(duo, "is_market_open", lambda *a, **k: False)
    monkeypatch.setattr(duo, "get_provider", lambda *a, **k: SimpleNamespace(name="fake_havuz"))
    called = []
    monkeypatch.setattr(duo, "scan_symbol", lambda *a, **k: called.append(1))
    duo.run_duo(once=True)
    assert called == []  # market kapalı → hiç tarama yok
