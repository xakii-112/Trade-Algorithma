"""META_GATE gölge (shadow) modu testleri.

Kritik değişmezler:
- shadow: p_win hesaplanır + metadata/reason yazılır ama sinyal ASLA bastırılmaz
  (aksiyon/stop/hedef/skor birebir korunur) → canlı kanıt döngüsü yaşar.
- enforce: eski davranış BİREBİR (eşik altı WAIT + meta_gated + META_GATE_P reason).
- off: kapı hiç çağrılmaz.
- META_GATE env parse: "shadow"/"gölge" → shadow; _envb-doğru token'lar → enforce; diğer → off.

Kapı sahte (get_gate monkeypatch) → model dosyası/ağ yok; CONFIG dataclasses.replace +
modül-attr monkeypatch ile değiştirilir (test_risk_manager/test_backtest deseni)."""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pandas as pd
import pytest

import nihai.config as config
import nihai.signals.meta_gate as mg
import nihai.signals.strategy as st
from nihai.config import CONFIG
from nihai.signals.strategy import SignalAction


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def _make_df(n=320, seed=17):
    """test_strategy_perf_parity._make_df ile aynı sentetik OHLCV (ağsız, deterministik)."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = rng.integers(1e5, 5e6, n).astype(float)
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="h")
    return pd.DataFrame({"open": open_, "high": np.maximum(high, close),
                         "low": np.minimum(low, close), "close": close,
                         "volume": vol}, index=idx)


@pytest.fixture(scope="module")
def actionable_df():
    """generate_signal'in LONG/SHORT ürettiği bir dilim bul (kapı kapalıyken).

    Modül kapsamı: pahalı arama bir kez koşar. Arama sırasında gerçek CONFIG kullanılır;
    testler DATA_DIR'i tmp'ye yönlendirdiğinden model dosyası yok → kapı zaten pasif."""
    for mode in ("ensemble", "donchian", "macd", "vol_breakout"):
        df = _make_df()
        cache = st._SharedIndicators(df)
        for i in range(80, len(df)):
            sig = st.generate_signal("SYN", df.iloc[:i + 1], strategy_mode=mode,
                                     shared=cache.upto(i + 1))
            if sig.is_actionable():
                return df.iloc[:i + 1], mode
    pytest.fail("sentetik df'te aksiyonlu sinyal bulunamadı (fixture varsayımı kırıldı)")


class _FakeGate:
    """Sabit p_win döndüren sahte kapı (model dosyası/özellik çıkarımı yok)."""
    threshold = 0.5

    def __init__(self, p: float):
        self._p = p

    def probability(self, df, sig, **_):
        # Tur-43: gerçek imza `shared=` kwarg'ı ile genişledi. Test ikizi de kabul
        # etmeli — `generate_signal` kapı çağrısını `except Exception: pass` ile
        # sardığı için imza uyuşmazlığı ÜRETİMDE sessizce kapıyı kapatırdı.
        return self._p


def _patch_mode(monkeypatch, mode: str, thr: float = 0.6):
    """CONFIG'i güvenle değiştir: frozen dataclass → replace + modül-attr monkeypatch
    (test sonunda otomatik geri alınır)."""
    strat = dataclasses.replace(CONFIG.strategy, meta_gate_mode=mode, meta_gate_threshold=thr)
    monkeypatch.setattr(st, "CONFIG", dataclasses.replace(CONFIG, strategy=strat))


def _signal(actionable, monkeypatch, mode: str, p: float | None = None, thr: float = 0.6):
    df, strat_mode = actionable
    _patch_mode(monkeypatch, mode, thr)
    if p is not None:
        monkeypatch.setattr(mg, "get_gate", lambda: _FakeGate(p))
    return st.generate_signal("SYN", df, strategy_mode=strat_mode)


# ---------------------------------------------------------------------------
# (a) shadow + düşük p → aksiyon KORUNUR, yalnız işaretlenir
# ---------------------------------------------------------------------------
def test_shadow_low_p_keeps_action_and_marks(actionable_df, monkeypatch):
    base = _signal(actionable_df, monkeypatch, "off")
    s = _signal(actionable_df, monkeypatch, "shadow", p=0.30)
    assert base.action != SignalAction.WAIT          # fixture sözleşmesi
    assert s.action == base.action                   # sinyal BASTIRILMADI
    assert (s.price, s.stop, s.target, s.quality_score, s.rr_ratio) == \
           (base.price, base.stop, base.target, base.quality_score, base.rr_ratio)
    assert s.metadata["meta_p_win"] == 0.3           # p_win yazıldı (kanıt döngüsü girdisi)
    assert s.metadata["meta_shadow_gated"] == base.action.value
    assert "meta_gated" not in s.metadata            # enforce işareti YOK
    assert "META_SHADOW_P0.30<0.60" in s.reasons
    assert not any(r.startswith("META_GATE_P") for r in s.reasons)


# ---------------------------------------------------------------------------
# (b) enforce + düşük p → WAIT + meta_gated (mevcut davranış regresyonu)
# ---------------------------------------------------------------------------
def test_enforce_low_p_gates_to_wait(actionable_df, monkeypatch):
    base = _signal(actionable_df, monkeypatch, "off")
    s = _signal(actionable_df, monkeypatch, "enforce", p=0.30)
    assert s.action == SignalAction.WAIT
    assert s.metadata["meta_gated"] == base.action.value   # elenen yön saklanır (tracker)
    assert s.metadata["meta_p_win"] == 0.3
    assert "meta_shadow_gated" not in s.metadata
    assert "META_GATE_P0.30<0.60" in s.reasons
    assert (s.stop, s.target) == (base.stop, base.target)  # seviyeler WAIT'e taşınır


# ---------------------------------------------------------------------------
# p ≥ eşik → her iki kipte mevcut META_P reason'ı, işaret yok
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_high_p_passes_with_meta_reason(actionable_df, monkeypatch, mode):
    base = _signal(actionable_df, monkeypatch, "off")
    s = _signal(actionable_df, monkeypatch, mode, p=0.80)
    assert s.action == base.action
    assert "META_P0.80" in s.reasons
    assert "meta_gated" not in s.metadata and "meta_shadow_gated" not in s.metadata


# ---------------------------------------------------------------------------
# (c) off → kapı HİÇ çağrılmaz
# ---------------------------------------------------------------------------
def test_off_mode_never_calls_gate(actionable_df, monkeypatch):
    calls: list[int] = []

    def _spy():
        calls.append(1)
        return _FakeGate(0.0)

    monkeypatch.setattr(mg, "get_gate", _spy)
    s = _signal(actionable_df, monkeypatch, "off")
    assert calls == []                               # get_gate'e hiç dokunulmadı
    assert "meta_p_win" not in s.metadata


# ---------------------------------------------------------------------------
# Geriye uyum: yalnız enable_meta_gate bilen eski/fake config → semantik korunur
# ---------------------------------------------------------------------------
def test_legacy_config_flag_maps_to_enforce(actionable_df, monkeypatch):
    real = CONFIG.strategy
    attrs = {f.name: getattr(real, f.name) for f in dataclasses.fields(real)}
    attrs.pop("meta_gate_mode")                      # eski config'te kip alanı yok
    attrs["meta_gate_threshold"] = 0.6
    old_strat = types.SimpleNamespace(**attrs, enable_meta_gate=True)
    cfg_attrs = {f.name: getattr(CONFIG, f.name) for f in dataclasses.fields(CONFIG)}
    cfg_attrs["strategy"] = old_strat
    monkeypatch.setattr(st, "CONFIG", types.SimpleNamespace(**cfg_attrs))
    monkeypatch.setattr(mg, "get_gate", lambda: _FakeGate(0.30))

    df, strat_mode = actionable_df
    s = st.generate_signal("SYN", df, strategy_mode=strat_mode)
    assert s.action == SignalAction.WAIT             # True → enforce gibi davranır
    assert s.metadata.get("meta_gated")


# ---------------------------------------------------------------------------
# (d) META_GATE env parse — üç değerli kip + geriye-uyum bayrağı
# ---------------------------------------------------------------------------
def test_meta_gate_mode_env_parse(monkeypatch):
    cases = {"shadow": "shadow", "SHADOW": "shadow", " Shadow ": "shadow",
             "gölge": "shadow", "GÖLGE": "shadow", "golge": "shadow",
             "true": "enforce", "TRUE": "enforce", "1": "enforce",
             "yes": "enforce", "on": "enforce",
             "false": "off", "": "off", "0": "off", "maybe": "off", "no": "off"}
    for raw, want in cases.items():
        monkeypatch.setenv("META_GATE", raw)
        assert config._meta_gate_mode() == want, raw
    monkeypatch.delenv("META_GATE", raising=False)
    assert config._meta_gate_mode() == "off"         # unset → varsayılan kapalı


def test_strategy_filters_mode_and_backcompat_property(monkeypatch):
    monkeypatch.setenv("META_GATE", "shadow")
    sf = config.StrategyFilters()
    assert sf.meta_gate_mode == "shadow" and sf.enable_meta_gate is True
    monkeypatch.setenv("META_GATE", "true")
    sf = config.StrategyFilters()
    assert sf.meta_gate_mode == "enforce" and sf.enable_meta_gate is True
    monkeypatch.setenv("META_GATE", "false")
    sf = config.StrategyFilters()
    assert sf.meta_gate_mode == "off" and sf.enable_meta_gate is False
