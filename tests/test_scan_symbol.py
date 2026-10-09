"""test_scan_symbol.py — orkestrasyon karar-yolu (main.scan_symbol) + C2/C4/C6 regresyonları.

scan_symbol'ün guard→sinyal→açılış yolu sağlayıcı/telegram/state sahteleriyle sürülür
(gerçek ağ/log yok; kök conftest üretim data/log'unu geçici dizine yönlendirir). Ayrıca:
- C2  alert-dedup: (symbol,bar_ts) başına tek alarm (Telegram spam önleme).
- C4  ortak kapanış defteri: _apply_close_bookkeeping (T+1 takas + kill-switch dahil).
- C6  equity eğrisi: yalnız değişince yaz."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

import nihai.execution.main as main
from nihai.risk.position_manager import SlippageModel
from nihai.signals.strategy import Signal, SignalAction

_NOW = datetime(2024, 1, 2, 15, 0)


@pytest.fixture(autouse=True)
def _reset_alert_dedup():
    """Modül-düzeyi alarm-dedup'ı her testten önce temizle (test izolasyonu)."""
    main._alerted_bars.clear()
    yield
    main._alerted_bars.clear()


def _htf_df(rows: int = 80):
    idx = pd.date_range("2024-01-02 09:30", periods=rows, freq="h")
    px = [100.0 + i * 0.1 for i in range(rows)]
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": [1e6] * rows},
                        index=idx)


class _Provider:
    def __init__(self, df):
        self._df = df

    def get_bars(self, sym, interval, lookback_bars=200):
        return self._df

    def get_quote(self, sym):
        return None                      # quote-sağlık yolunu atla

    def get_option_chain(self, sym, *a, **k):
        return None                      # opsiyon-zincir yolunu atla

    def is_quote_healthy(self, q):
        return (True, "")


class _State:
    def __init__(self, *, kill=False, cooldown=False):
        self.kill_switch_active = kill
        self._cooldown = cooldown
        self.cooldown_set: list[str] = []

    def cooldown_active(self, sym, now):
        return self._cooldown

    def wash_sale_check(self, sym, now, asset_type=""):   # Tur-58: araç türü
        return (False, "")

    def good_faith_warning(self, notional):
        return (False, "")

    def set_cooldown(self, sym, now):
        self.cooldown_set.append(sym)


class _Risk:
    def __init__(self, approved=True):
        self._approved = approved
        self.regime_label = "—"
        self.regime_multiplier = 1.0

    def size_stock(self, **kw):
        return SimpleNamespace(approved=self._approved, reason="cap doldu",
                               sizing={"notional": 1000.0, "shares": 10, "risk_dollars": 100.0})


class _Breaker:
    def __init__(self):
        self.emits = 0

    def can_emit(self):
        return (True, "")

    def record_emit(self):
        self.emits += 1


class _Pmgr:
    def __init__(self):
        self.opened: list = []
        self.slip = SlippageModel()   # Tur-89: scan_symbol dolum fiyatında RR sınar

    def open_from_signal(self, sig, sizing, **kw):
        self.opened.append((sig, sizing, kw))
        return SimpleNamespace(symbol=sig.symbol)


def _install(monkeypatch, *, approved=True):
    """main modül-düzeyi guard'larını sahte, deterministik davranışa sabitle."""
    blocks: list[tuple] = []
    tg = SimpleNamespace(signals=[], alerts=[])
    monkeypatch.setattr(main, "now_et_naive", lambda: _NOW)
    monkeypatch.setattr(main, "get_sector", lambda s: "Technology")
    monkeypatch.setattr(main, "can_enter_new_position", lambda: (True, "ok"))
    monkeypatch.setattr(main, "event_guard", lambda s: (False, ""))
    monkeypatch.setattr(main, "corporate_action_guard", lambda s: (False, ""))
    monkeypatch.setattr(main, "halt_or_luld_guard", lambda df: (False, ""))
    monkeypatch.setattr(main, "log_block", lambda code, msg="": blocks.append((code, msg)))
    monkeypatch.setattr(main, "log_signal", lambda *a, **k: None)
    monkeypatch.setattr(main, "log_decision", lambda *a, **k: None)
    monkeypatch.setattr(main, "telegram_bot",
                        SimpleNamespace(notify_signal=lambda *a, **k: tg.signals.append((a, k)),
                                        notify_alert=lambda *a, **k: tg.alerts.append((a, k))))
    sig = Signal(SignalAction.LONG, "AAPL", 100.0, 95.0, 115.0,
                 reasons=["breakout"], quality_score=1.0, rr_ratio=3.0)
    monkeypatch.setattr(main, "generate_signal", lambda *a, **k: sig)
    return SimpleNamespace(blocks=blocks, tg=tg, state=_State(), risk=_Risk(approved),
                           breaker=_Breaker(), pmgr=_Pmgr())


def _scan(fx, df):
    main.scan_symbol("AAPL", _Provider(df), fx.risk, fx.state, fx.breaker, fx.pmgr,
                     dashboard=None, lock=None)


# --------------------------------------------------------------------- scan_symbol
def test_kill_switch_blocks_and_returns(monkeypatch):
    fx = _install(monkeypatch)
    fx.state.kill_switch_active = True
    _scan(fx, _htf_df())
    assert ("KILL_SWITCH_ACTIVE", ) == (fx.blocks[0][0], )
    assert fx.pmgr.opened == [] and fx.tg.signals == []


def test_empty_df_blocks_data_empty(monkeypatch):
    fx = _install(monkeypatch)
    _scan(fx, pd.DataFrame())
    assert any(code == "DATA_EMPTY" for code, _ in fx.blocks)
    assert fx.pmgr.opened == [] and fx.tg.signals == []


def test_valid_approved_opens_once_notifies_and_sets_cooldown(monkeypatch):
    fx = _install(monkeypatch, approved=True)
    _scan(fx, _htf_df())
    assert len(fx.pmgr.opened) == 1                # tek pozisyon açıldı
    inputs = fx.pmgr.opened[0][2]
    assert inputs["signal_price"] == 100
    assert inputs["adv_shares"] == 1e6 and inputs["bar_sigma"] > 0
    assert inputs["spread_pct"] is None  # sağlayıcı bu testte kote vermiyor
    assert len(fx.tg.signals) == 1                 # tek bildirim (üst-alarm; açılış çift göndermez)
    assert fx.state.cooldown_set == ["AAPL"]       # açılış sonrası cooldown set
    assert fx.breaker.emits == 1


def test_live_anchor_preserves_cost_reference_and_observed_spread(monkeypatch):
    fx = _install(monkeypatch, approved=True)
    monkeypatch.setattr(main, "LIVE_ENTRY_ANCHOR", True)
    quote = SimpleNamespace(bid=100.9, ask=101.1, last=101, timestamp=_NOW, spread_measured=True)
    monkeypatch.setattr(_Provider, "get_quote", lambda self, sym: quote)
    _scan(fx, _htf_df())
    assert len(fx.pmgr.opened) == 1
    signal, _, inputs = fx.pmgr.opened[0]
    assert signal.price == 101 and inputs["signal_price"] == 100
    assert abs(inputs["spread_pct"] - .2 / 101) < 1e-12


def test_risk_reject_no_open_and_alert_deduped_across_cycles(monkeypatch):
    # Risk reddederse pozisyon AÇILMAZ; ayrıca aynı bar iki tur taransa da alarm TEK kez
    # (C2: eski davranış her ~60s yeniden alarmlıyordu → Telegram spam).
    fx = _install(monkeypatch, approved=False)
    df = _htf_df()
    _scan(fx, df)
    _scan(fx, df)                                  # aynı bar_ts → ikinci turda alarm yok
    assert fx.pmgr.opened == []                    # açılış yok
    assert fx.state.cooldown_set == []             # cooldown yalnız açılışta
    assert len(fx.tg.signals) == 1                 # iki tura rağmen tek alarm (dedup)


# --------------------------------------------------------------------- C2 _alert_once
def test_alert_once_dedup_per_bar():
    main._alerted_bars.clear()
    assert main._alert_once("AAPL", "b1") is True
    assert main._alert_once("AAPL", "b1") is False   # aynı bar → sustur
    assert main._alert_once("AAPL", "b2") is True     # yeni bar → yeni alarm
    assert main._alert_once("MSFT", "b1") is True     # farklı sembol bağımsız


# --------------------------------------------------------------------- C4 kapanış defteri
def _fake_ce(pnl=-50.0, asset_type="stock", exit_price=90.0, qty=10):
    pos = SimpleNamespace(symbol="AAPL", asset_type=asset_type, quantity=qty, side="LONG")
    return SimpleNamespace(realized_pnl=pnl, pos=pos, exit_price=exit_price, exit_time=_NOW)


class _BookStore:
    def __init__(self, kill=False):
        self.kill_switch_active = kill
        self.pnls: list = []
        self.wash: list = []
        self.unsettled: list = []

    def record_trade_pnl(self, pnl):
        self.pnls.append(pnl)

    def register_close_for_wash_sale(self, sym, pnl, when, asset_type=""):  # Tur-58
        self.wash.append((sym, pnl, when))

    def add_unsettled_proceeds(self, amount, day, side="LONG"):  # Tur-89: taraf
        self.unsettled.append((amount, day))


def test_apply_close_bookkeeping_full_ledger(monkeypatch):
    alerts: list = []
    monkeypatch.setattr(main, "telegram_bot",
                        SimpleNamespace(notify_alert=lambda *a, **k: alerts.append((a, k))))
    st = _BookStore(kill=False)
    main._apply_close_bookkeeping(st, _fake_ce(pnl=-50.0))
    assert st.pnls == [-50.0]
    assert st.wash == [("AAPL", -50.0, _NOW)]
    # Madde 23 — T+1 takas edilmemiş gelir: hisse mult=1 (eski flatten_all yolunda EKSİKTİ)
    assert st.unsettled == [(90.0 * 10 * 1, _NOW.date())]
    assert alerts == []                            # kill-switch pasif → uyarı yok


def test_apply_close_bookkeeping_option_mult_and_killswitch(monkeypatch):
    alerts: list = []
    monkeypatch.setattr(main, "telegram_bot",
                        SimpleNamespace(notify_alert=lambda *a, **k: alerts.append((a, k))))
    st = _BookStore(kill=True)
    main._apply_close_bookkeeping(st, _fake_ce(pnl=20.0, asset_type="option", exit_price=3.0, qty=2))
    assert st.unsettled == [(3.0 * 2 * 100, _NOW.date())]   # opsiyon çarpanı 100
    assert len(alerts) == 1                        # kill-switch aktif → CRIT uyarısı


# --------------------------------------------------------------------- C6 equity eğrisi
def test_append_equity_point_only_on_change(tmp_path, monkeypatch):
    csv = tmp_path / "equity_curve.csv"
    monkeypatch.setattr(main, "EQUITY_CURVE_CSV", csv)
    monkeypatch.setattr(main, "_last_equity_point", None)
    main.append_equity_point(1000.0, _NOW)
    main.append_equity_point(1000.0, _NOW)         # özdeş → yazılmaz
    main.append_equity_point(1001.0, _NOW)         # değişti → yazılır
    lines = csv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "timestamp,equity"
    assert len(lines) == 3                          # başlık + 2 farklı nokta
