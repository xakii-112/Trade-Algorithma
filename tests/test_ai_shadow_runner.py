"""nihai.ai.shadow_runner testleri — Quant Beyin GÖLGE-KOŞUCU (Tur-17).

Ağ ÇAĞRISI YAPMA: provider her testte monkeypatch'lenir (gerçek yfinance/hub'a
dokunulmaz); strateji._VIX_LOADED False'a sabitlenir (özellik çıkarımı yolu asla
`_load_vix()`'i tetiklemez — shadow_runner._cheap_features'ın ağ-tetiklememe
sözleşmesi doğrulanır, ayrıca test izolasyonu için de gereklidir).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nihai.ai import decision, episodic, shadow_runner


def _make_df(n: int = 90, start: float = 100.0) -> pd.DataFrame:
    """Deterministik, hafif yükselen sentetik OHLCV (n bar, günlük)."""
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    closes = start + np.arange(n, dtype=float) * 0.1
    return pd.DataFrame({
        "open": closes - 0.05,
        "high": closes + 0.2,
        "low": closes - 0.2,
        "close": closes,
        "volume": np.full(n, 1_000_000.0),
    }, index=idx)


def _make_noisy_df(n: int = 90, start: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """Seed'li rastgele-yürüyüş OHLCV — portföy köprüsü için gereken (dejenere olmayan)
    kovaryans. `_make_df` düz-doğrusal olduğundan getiri varyansı ~0'dır ve gerçek
    `portfolio.propose` orada anlamlı bir çözüm üretemez."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    closes = start * np.exp(np.cumsum(rng.normal(0.0003, 0.012, size=n)))
    return pd.DataFrame({
        "open": closes * 0.999,
        "high": closes * 1.004,
        "low": closes * 0.996,
        "close": closes,
        "volume": rng.uniform(8e5, 1.2e6, size=n),
    }, index=idx)


class _FakeProvider:
    """get_bars(symbol, interval, lookback_bars) duck-tipli sahte sağlayıcı.

    `table`: sembol → DataFrame | Exception | None (None/boş-DataFrame de "başarısız"
    olarak ele alınır — gerçek provider'ların boş DataFrame dönme davranışını taklit eder).
    """

    def __init__(self, table: dict):
        self.table = table
        self.calls: list[str] = []

    def get_bars(self, symbol: str, interval: str, lookback_bars: int = 200):
        self.calls.append(symbol)
        v = self.table.get(symbol, pd.DataFrame())
        if isinstance(v, Exception):
            raise v
        return v


@pytest.fixture(autouse=True)
def izole(tmp_path, monkeypatch):
    """Her test izole: AI_DECISION_MODE temiz başlar, DECISIONS_FILE/SHADOW_LOG_FILE tmp'e
    yönlenir, epizot yazımı no-op (test_ai_decision.py izole fixture'ıyla aynı desen),
    VIX önbelleği soğuk sabitlenir (ağ tetiklenmesin)."""
    monkeypatch.delenv("AI_DECISION_MODE", raising=False)
    monkeypatch.delenv("WATCHLIST", raising=False)
    monkeypatch.setattr(decision, "DECISIONS_FILE", tmp_path / "ai_decisions.jsonl")
    monkeypatch.setattr(shadow_runner, "SHADOW_LOG_FILE", tmp_path / "ai_shadow_runner_log.jsonl")
    monkeypatch.setattr(shadow_runner, "WATCHLIST_FILE", tmp_path / "watchlist.json")
    monkeypatch.setattr(episodic, "append_episode", lambda **kw: None)
    from nihai.signals import strategy as strategy_mod
    monkeypatch.setattr(strategy_mod, "_VIX_LOADED", False, raising=False)
    # Koşu-başı ısıtma bu testte KAPALI: buradaki sözleşme `_cheap_features`'ın ağı KENDİ
    # tetiklemediğidir; `_warm_vix` ayrı ve AÇIK bir yoldur (test_tur40_neural_kablo.py).
    monkeypatch.setattr(shadow_runner, "WARM_VIX", False, raising=False)
    # Portföy köprüsü (Tur-17) gölge-önerilerini de tmp'e yönlendir: kök conftest DATA_DIR'ı
    # zaten geçici dizine alıyor, bu açık yönlendirme testin kendi izolasyonunu belgeler.
    from nihai.ai import portfolio as portfolio_mod
    monkeypatch.setattr(portfolio_mod, "PORTFOLIO_PROPOSALS_FILE",
                        tmp_path / "ai_portfolio_proposals.jsonl")
    yield


def _strong_prior(monkeypatch, strategy: str = "event_breakout") -> None:
    """Yön kararı üretebilen (ABSTAIN olmayan) güçlü öğrenme-durumu prior'u — test_ai_decision
    ile aynı desen (CI 0.5'i dışlasın diye)."""
    from nihai.ai import learning
    state = {"strategies": {strategy: {"n": 50, "win": 40, "loss": 10,
                                       "post_mean": 0.79, "lo95": 0.66, "hi95": 0.89}}}
    monkeypatch.setattr(learning, "load_state", lambda: state)


# ---------------------------------------------------------------------------
# run_once — temel akış: kısmi başarısızlık koşuyu düşürmez
# ---------------------------------------------------------------------------
def test_run_once_iki_ok_bir_fail(monkeypatch):
    table = {
        "AAPL": _make_df(),
        "MSFT": _make_df(start=200.0),
        "BADSYM": RuntimeError("provider patladı"),
    }
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["AAPL", "MSFT", "BADSYM"])
    assert summary["n_ok"] == 2
    assert summary["n_fail"] == 1


def test_run_once_bos_dataframe_de_fail_sayilir(monkeypatch):
    table = {"AAPL": _make_df(), "EMPTY": pd.DataFrame()}
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["AAPL", "EMPTY"])
    assert summary["n_ok"] == 1
    assert summary["n_fail"] == 1


def test_run_once_close_kolonu_yoksa_fail(monkeypatch):
    df_no_close = pd.DataFrame({"open": [1.0, 2.0, 3.0, 4.0, 5.0]})
    table = {"NOCLOSE": df_no_close}
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["NOCLOSE"])
    assert summary["n_ok"] == 0
    assert summary["n_fail"] == 1


# ---------------------------------------------------------------------------
# AI_DECISION_MODE kipi — off/shadow ayrımı
# ---------------------------------------------------------------------------
def test_off_kipinde_uyari_notu_ve_karar_dosyasi_yazilmaz(monkeypatch):
    monkeypatch.delenv("AI_DECISION_MODE", raising=False)   # → resolve_mode() "off"
    table = {"AAPL": _make_df()}
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["AAPL"])
    assert summary["mode"] == "off"
    assert "kip off" in summary["note"]
    assert not decision.DECISIONS_FILE.exists()


def test_shadow_kipinde_karar_ve_telemetri_jsonl_olusur(monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    _strong_prior(monkeypatch)
    table = {"AAPL": _make_df()}
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["AAPL"])
    assert summary["mode"] == "shadow"
    assert "note" not in summary
    assert decision.DECISIONS_FILE.exists()
    assert shadow_runner.SHADOW_LOG_FILE.exists()
    # decision dosyasında en az bir geçerli JSONL satırı var
    lines = decision.DECISIONS_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    json.loads(lines[0])


def test_shadow_kipinde_telemetri_ozeti_dogru_semayla_yazilir(monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    _strong_prior(monkeypatch)
    table = {"AAPL": _make_df()}
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    shadow_runner.run_once(symbols=["AAPL"])
    lines = shadow_runner.SHADOW_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["n_ok"] == 1 and rec["mode"] == "shadow"


# ---------------------------------------------------------------------------
# Karantina (bozuk seri) → ABSTAIN + quarantined sayacı
# ---------------------------------------------------------------------------
def test_karantinali_seri_abstain_ve_quarantined_sayaci_artar(monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    _strong_prior(monkeypatch)
    # < _MIN_BARS(5) bar → data_health.health_check quarantine=True zorunlu ABSTAIN üretir.
    short_df = _make_df(n=3)
    table = {"BOZUK": short_df}
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["BOZUK"])
    assert summary["n_ok"] == 1
    assert summary["quarantined"] == 1
    assert summary["abstain"] == 1
    assert summary["long_bias"] == 0
    assert summary["short_bias"] == 0


def test_saglikli_seri_karantinaya_dusmez(monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    _strong_prior(monkeypatch)
    table = {"AAPL": _make_df()}
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["AAPL"])
    assert summary["quarantined"] == 0


# ---------------------------------------------------------------------------
# Watchlist kaynak önceliği: json > env > boş
# ---------------------------------------------------------------------------
def test_watchlist_json_env_uzerinde_oncelikli(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCHLIST", "ENVSYM")
    wl_file = tmp_path / "watchlist.json"
    wl_file.write_text(json.dumps({"symbols": ["JSONSYM"]}), encoding="utf-8")
    monkeypatch.setattr(shadow_runner, "WATCHLIST_FILE", wl_file)
    assert shadow_runner._resolve_watchlist() == ["JSONSYM"]


def test_watchlist_json_yoksa_env_kullanilir(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCHLIST", "ENVSYM1,ENVSYM2")
    monkeypatch.setattr(shadow_runner, "WATCHLIST_FILE", tmp_path / "yok.json")
    assert shadow_runner._resolve_watchlist() == ["ENVSYM1", "ENVSYM2"]


def test_watchlist_ikisi_de_yoksa_bos(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCHLIST", raising=False)
    monkeypatch.setattr(shadow_runner, "WATCHLIST_FILE", tmp_path / "yok.json")
    assert shadow_runner._resolve_watchlist() == []


def test_watchlist_bozuk_json_dosyasi_env_e_duser(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCHLIST", "FALLBACK")
    wl_file = tmp_path / "watchlist.json"
    wl_file.write_text("{ bozuk json", encoding="utf-8")
    monkeypatch.setattr(shadow_runner, "WATCHLIST_FILE", wl_file)
    assert shadow_runner._resolve_watchlist() == ["FALLBACK"]


# ---------------------------------------------------------------------------
# --symbols override + max_symbols kırpması
# ---------------------------------------------------------------------------
def test_run_once_symbols_override_watchlist_yerine_kullanilir(monkeypatch, tmp_path):
    wl_file = tmp_path / "watchlist.json"
    wl_file.write_text(json.dumps({"symbols": ["WLSYM"]}), encoding="utf-8")
    monkeypatch.setattr(shadow_runner, "WATCHLIST_FILE", wl_file)
    table = {"OVERRIDE": _make_df()}
    provider = _FakeProvider(table)
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: provider)
    shadow_runner.run_once(symbols=["OVERRIDE"])
    assert provider.calls == ["OVERRIDE"]


def test_run_once_max_symbols_kirpar(monkeypatch):
    table = {s: _make_df() for s in ("A", "B", "C", "D")}
    provider = _FakeProvider(table)
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: provider)
    shadow_runner.run_once(symbols=["A", "B", "C", "D"], max_symbols=2)
    assert provider.calls == ["A", "B"]


# ---------------------------------------------------------------------------
# Özet şeması + boş watchlist (provider hiç çağrılmaz)
# ---------------------------------------------------------------------------
def test_ozet_semasi_anahtarlar(monkeypatch):
    table = {"AAPL": _make_df()}
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["AAPL"])
    for key in ("ts", "n_ok", "n_fail", "abstain", "long_bias", "short_bias",
               "quarantined", "mode"):
        assert key in summary


# ---------------------------------------------------------------------------
# PORTFÖY BAKIŞI köprüsü (Tur-17) — GÖLGE öneri, tur sonunda bir kez
# ---------------------------------------------------------------------------
def _panel_candidates(panel: dict[str, pd.Series]) -> list[dict]:
    return [{"symbol": s, "strategy": "event_breakout", "posterior": 0.5} for s in panel]


def _fake_propose(captured: dict):
    """`portfolio.propose` yerine geçen deterministik sahte (gerçek optimizasyon yolu
    `test_ai_portfolio.py`'nin işi; burada KÖPRÜ sözleşmesi test edilir)."""
    def _inner(candidates, closes_panel, **kw):
        captured["candidates"] = candidates
        captured["panel"] = closes_panel
        return {"methods": {"hrp": {"weights": {"A": 0.5, "B": 0.3, "C": 0.2}},
                            "erc": {"weights": {"A": 0.34, "B": 0.33, "C": 0.33}}}}
    return _inner


def test_portfoy_bakisi_off_kipinde_calismaz(monkeypatch):
    """Anayasa: gölge-dışı kipte hiçbir portföy önerisi ÜRETİLMEZ (kip=off → birikim yok)."""
    monkeypatch.delenv("AI_DECISION_MODE", raising=False)
    table = {s: _make_noisy_df(seed=i) for i, s in enumerate(("A", "B", "C"))}
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["A", "B", "C"])
    assert summary["mode"] == "off"
    assert "portfolio" not in summary


def test_portfoy_bakisi_uc_sembolden_az_evrende_yok(monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    _strong_prior(monkeypatch)
    table = {"A": _make_noisy_df(seed=1), "B": _make_noisy_df(seed=2)}
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["A", "B"])
    assert summary["n_ok"] == 2
    assert "portfolio" not in summary


def test_portfoy_bakisi_golge_kipte_ozet_uretir(monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    _strong_prior(monkeypatch)
    captured: dict = {}
    from nihai.ai import portfolio as portfolio_mod
    monkeypatch.setattr(portfolio_mod, "propose", _fake_propose(captured))
    table = {s: _make_noisy_df(seed=i) for i, s in enumerate(("A", "B", "C"))}
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["A", "B", "C"])
    assert summary["portfolio"]["n_universe"] == 3
    assert summary["portfolio"]["n_bars"] == 90
    assert summary["portfolio"]["n_methods"] == 2
    assert summary["portfolio"]["hrp_top3"][0] == ["A", 0.5]      # |ağırlık| sırasına göre
    # Köprü decision posterior'ını BL görüşü olarak taşır (gölge görüş = karar posterior'u)
    assert {c["symbol"] for c in captured["candidates"]} == {"A", "B", "C"}
    assert all("posterior" in c for c in captured["candidates"])


def test_portfoy_bakisi_gercek_propose_ile_ucdan_uca(monkeypatch):
    """Sahte YOK: gerçek `portfolio.propose` çalışır (köprünün ürettiği panelin gerçek
    optimizasyon yolunu besleyebildiğini kanıtlar) ve gölge-JSONL'e tek satır düşer."""
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    _strong_prior(monkeypatch)
    from nihai.ai import portfolio as portfolio_mod
    table = {s: _make_noisy_df(seed=i + 10) for i, s in enumerate(("A", "B", "C"))}
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["A", "B", "C"])
    assert summary["portfolio"]["n_universe"] == 3
    assert summary["portfolio"]["n_methods"] >= 3
    assert len(summary["portfolio"]["hrp_top3"]) == 3
    lines = portfolio_mod.PORTFOLIO_PROPOSALS_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["methods"]


def test_portfoy_bakisi_kisa_ortak_pencerede_atlanir(monkeypatch):
    """≥60 ORTAK bar şartı: 40 barlık seriler karantinaya düşmez ama portföy bakışını
    tetiklemez (kovaryans için yetersiz pencere sessizce zorlanmaz)."""
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    _strong_prior(monkeypatch)
    table = {s: _make_noisy_df(n=40, seed=i) for i, s in enumerate(("A", "B", "C"))}
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["A", "B", "C"])
    assert summary["n_ok"] == 3
    assert "portfolio" not in summary


def test_portfoy_bakisi_tarih_kesisimiyle_hizalar(monkeypatch):
    """REGRESYON: hizalama POZİSYONEL kuyrukla değil TARİHLE yapılır. B 10 gün kaymışsa
    kuyruk-hizalaması farklı günleri aynı satıra koyup kovaryansı sessizce bozardı;
    doğru davranış ORTAK tarih kesişimidir. Ayrıca panelin DatetimeIndex'i korunmalı —
    `portfolio._panel_dates` yalnız o zaman tarihsel stres penceresini hesaplayabilir."""
    captured: dict = {}
    from nihai.ai import portfolio as portfolio_mod
    monkeypatch.setattr(portfolio_mod, "propose", _fake_propose(captured))
    idx_a = pd.date_range("2026-01-01", periods=90, freq="D")
    idx_b = pd.date_range("2026-01-11", periods=90, freq="D")     # 10 gün kaymış
    panel = {
        "A": pd.Series(np.linspace(100.0, 110.0, 90), index=idx_a),
        "B": pd.Series(np.linspace(200.0, 220.0, 90), index=idx_b),
        "C": pd.Series(np.linspace(50.0, 55.0, 90), index=idx_a),
    }
    out = shadow_runner._shadow_portfolio_view(_panel_candidates(panel), panel, "shadow")
    assert out is not None
    assert out["n_bars"] == 80                                     # 2026-01-11 .. 2026-03-31
    got = captured["panel"]
    assert isinstance(got.index, pd.DatetimeIndex)
    assert list(got.index) == list(idx_a.intersection(idx_b))
    assert not got.isna().any().any()


def test_portfoy_bakisi_kesisim_60_barin_altinda_none(monkeypatch):
    """Kayma büyükse ORTAK pencere 60'ın altına düşer → öneri üretilmez (kuyruk
    hizalamasında bu durum FARK EDİLMEZDİ: her seri 90 bar 'görünür')."""
    from nihai.ai import portfolio as portfolio_mod

    def _boom(*a, **kw):
        raise AssertionError("kesişim yetersizken propose ÇAĞRILMAMALIYDI")

    monkeypatch.setattr(portfolio_mod, "propose", _boom)
    idx_a = pd.date_range("2026-01-01", periods=90, freq="D")
    idx_b = pd.date_range("2026-02-20", periods=90, freq="D")      # kesişim 40 bar
    panel = {
        "A": pd.Series(np.linspace(100.0, 110.0, 90), index=idx_a),
        "B": pd.Series(np.linspace(200.0, 220.0, 90), index=idx_b),
        "C": pd.Series(np.linspace(50.0, 55.0, 90), index=idx_a),
    }
    assert shadow_runner._shadow_portfolio_view(_panel_candidates(panel), panel, "shadow") is None


def test_portfoy_bakisi_hatasi_kosuyu_dusurmez(monkeypatch):
    """Fail-soft: propose patlasa bile tur özeti üretilir, telemetri yazılır, koşu düşmez."""
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    _strong_prior(monkeypatch)
    from nihai.ai import portfolio as portfolio_mod

    def _boom(*a, **kw):
        raise RuntimeError("portföy motoru patladı")

    monkeypatch.setattr(portfolio_mod, "propose", _boom)
    table = {s: _make_noisy_df(seed=i) for i, s in enumerate(("A", "B", "C"))}
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["A", "B", "C"])
    assert summary["n_ok"] == 3
    assert "portfolio" not in summary
    assert shadow_runner.SHADOW_LOG_FILE.exists()


def test_portfoy_bakisi_karantinali_sembolu_evrene_almaz(monkeypatch):
    """Veri-sağlık kapısından KARANTİNA ile dönen sembol portföy evrenine girmez
    (bozuk seri ağırlık önerisine katkı veremez) — kalan 2 sembol de eşiği düşürmez."""
    captured: dict = {}
    from nihai.ai import portfolio as portfolio_mod
    monkeypatch.setattr(portfolio_mod, "propose", _fake_propose(captured))
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    _strong_prior(monkeypatch)
    table = {"A": _make_noisy_df(seed=1), "B": _make_noisy_df(seed=2),
             "BOZUK": _make_df(n=3)}          # < _MIN_BARS → zorunlu karantina
    monkeypatch.setattr(shadow_runner, "get_provider", lambda: _FakeProvider(table))
    summary = shadow_runner.run_once(symbols=["A", "B", "BOZUK"])
    assert summary["quarantined"] == 1
    assert "portfolio" not in summary          # evren 2'ye düştü (<3)
    assert "panel" not in captured             # propose hiç çağrılmadı


def test_bos_watchlist_provider_hic_cagrilmaz(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCHLIST", raising=False)
    monkeypatch.setattr(shadow_runner, "WATCHLIST_FILE", tmp_path / "yok.json")
    called = {"n": 0}

    def _boom():
        called["n"] += 1
        raise AssertionError("provider hiç kurulmamalıydı — watchlist boş")

    monkeypatch.setattr(shadow_runner, "get_provider", _boom)
    summary = shadow_runner.run_once()
    assert summary["n_ok"] == 0 and summary["n_fail"] == 0
    assert called["n"] == 0
