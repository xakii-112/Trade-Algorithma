"""Tur-89 — projeyi ÇALIŞTIRINCA görünen mantıksızlıklar (2026-09-23).

Kilitler (her biri kusurun ölçülmüş hâlinden):
  BULGU-89-1  IEX ince defter makası (%9-12) "ölçülmüş makas" sayılıyor → %5-6 hayali giriş kayması,
              dolum HEDEFİN ötesinde; RR/boyut sinyal fiyatında kuruluyordu.
  BULGU-89-2  Reg T alım gücü pozisyon başına 4× ve açık pozisyonları düşmüyordu (brüt 35× ölçüldü).
  BULGU-89-3  SHORT kapanışı (geri alım) T+1 "satış hasılatı" sayılıyordu.
  BULGU-89-4  İki tarama katmanı seans dışında 5 sn'de bir 550 sembolü tarayıp reddediyordu.
  BULGU-89-5  Yeniden başlatmada taşınan pozisyon deftere varsayılan ad + planlanan RR 0 ile düşüyordu.
"""
from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import nihai.execution.main as main
from nihai.config import ACTIVE_STRATEGY, CONFIG
from nihai.execution.dual_tier import TierScanner
from nihai.execution.state import StateStore
from nihai.risk.position_manager import (
    MAX_CREDIBLE_STOCK_SPREAD,
    PositionManager,
    SlippageModel,
    quote_spread_pct,
)
from nihai.risk.risk_manager import PortfolioState, RiskManager, VirtualPosition
from nihai.signals.strategy import Signal, SignalAction

KOK = Path(__file__).resolve().parents[1]


# ============================================================ BULGU-89-1: makas + dolum RR'ı
def _kote(bid: float, ask: float):
    return SimpleNamespace(bid=bid, ask=ask, last=(bid + ask) / 2, spread_measured=True)


def test_makas_tavani_ince_defteri_olcum_saymaz():
    assert quote_spread_pct(_kote(99.5, 100.5), 100) == pytest.approx(0.01)
    # 2026-09-22 PPG/MAR/ECHO: %10-12'lik IEX "makası" artık ölçüm değil → taban kayma
    assert quote_spread_pct(_kote(95.0, 106.0), 100) is None
    assert quote_spread_pct(_kote(99.0, 101.0), 100) is None
    assert MAX_CREDIBLE_STOCK_SPREAD == 0.01


def test_hayali_kayma_artik_hedefin_otesine_doldurmaz():
    """PPG 2026-09-22: sinyal 106,485, stop 106,18, hedef 109,67; IEX kotası %10,9 makas."""
    slip = SlippageModel(impact_coef=0.0)
    q = _kote(100.68, 112.29)
    fill = slip.apply_entry(106.485, "LONG", "stock", spread_pct=quote_spread_pct(q, 106.485))
    assert fill == pytest.approx(106.485 * 1.001)       # eskiden 112,27 (+%5,4)
    assert fill < 109.67


class _Provider:
    def __init__(self, df, q):
        self._df, self._q = df, q

    def get_bars(self, sym, interval, lookback_bars=200):
        return self._df

    def get_quote(self, sym):
        return self._q

    def get_option_chain(self, sym, *a, **k):
        return None

    def is_quote_healthy(self, q):
        return (True, "")


class _State:
    kill_switch_active = False

    def cooldown_active(self, sym, now):
        return False

    def wash_sale_check(self, sym, now, asset_type=""):
        return (False, "")

    def good_faith_warning(self, notional):
        return (False, "")

    def set_cooldown(self, sym, now):
        pass


def _kur(monkeypatch):
    kayit = SimpleNamespace(bloklar=[], boyut=[], acilan=[])
    now = datetime(2024, 1, 2, 15, 0)
    monkeypatch.setattr(main, "now_et_naive", lambda: now)
    monkeypatch.setattr(main, "get_sector", lambda s: "Technology")
    monkeypatch.setattr(main, "can_enter_new_position", lambda: (True, "ok"))
    monkeypatch.setattr(main, "event_guard", lambda s: (False, ""))
    monkeypatch.setattr(main, "corporate_action_guard", lambda s: (False, ""))
    monkeypatch.setattr(main, "halt_or_luld_guard", lambda df: (False, ""))
    monkeypatch.setattr(main, "log_block", lambda code, msg="": kayit.bloklar.append(code))
    monkeypatch.setattr(main, "log_signal", lambda *a, **k: None)
    monkeypatch.setattr(main, "log_decision", lambda *a, **k: None)
    monkeypatch.setattr(main, "LIVE_ENTRY_ANCHOR", False)
    monkeypatch.setattr(main, "OPTION_MODE", "stock")
    monkeypatch.setattr(main.shadow_bridge, "evaluate", lambda *a, **k: None)
    monkeypatch.setattr(main.ai_advisor, "advise", lambda *a, **k: None)
    monkeypatch.setattr(main, "telegram_bot", SimpleNamespace(notify_signal=lambda *a, **k: None,
                                                               notify_alert=lambda *a, **k: None))
    sig = Signal(SignalAction.LONG, "AAPL", 100.0, 99.5, 101.0,
                 reasons=["breakout"], quality_score=5.0, rr_ratio=2.0)
    monkeypatch.setattr(main, "generate_signal", lambda *a, **k: sig)
    main._alerted_bars.clear()

    def _size(**kw):
        kayit.boyut.append(kw)
        return SimpleNamespace(approved=True, reason="",
                               sizing={"notional": 1000.0, "shares": 10, "risk_dollars": 10.0})

    risk = SimpleNamespace(size_stock=_size, regime_label="—", regime_multiplier=1.0)
    pmgr = SimpleNamespace(slip=SlippageModel(impact_coef=0.0),
                           open_from_signal=lambda s, sz, **k: kayit.acilan.append(s.symbol))
    breaker = SimpleNamespace(can_emit=lambda: (True, ""), record_emit=lambda: None)
    idx = pd.date_range("2024-01-02 09:30", periods=80, freq="h")
    px = [100.0] * 80
    df = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": [1e6] * 80}, index=idx)

    def tara(q):
        main.scan_symbol("AAPL", _Provider(df, q), risk, _State(), breaker, pmgr, None, None,
                         strategy_mode="event_breakout")
    return kayit, tara


def test_dolum_rr_kapisi_gercek_makasta_girisi_durdurur(monkeypatch):
    kayit, tara = _kur(monkeypatch)
    tara(_kote(99.55, 100.45))          # %0,9 GERÇEK makas: dolum 100,45 → RR 0,58 < 1,2
    assert "FILL_SLIPPAGE" in kayit.bloklar
    assert kayit.acilan == [] and kayit.boyut == []


def test_ince_defter_makasi_girisi_hayali_kaymayla_bozmaz_ve_boyut_dolumdan(monkeypatch):
    kayit, tara = _kur(monkeypatch)
    tara(_kote(95.0, 105.0))            # %10 IEX "makası" → ölçüm değil → taban %0,1
    assert "FILL_SLIPPAGE" not in kayit.bloklar
    assert kayit.acilan == ["AAPL"]
    # boyut SİNYAL fiyatında değil DOLUM fiyatında kurulur (gerçek risk = dolum − stop)
    assert kayit.boyut[0]["entry"] == pytest.approx(100.1)


# ============================================================ BULGU-89-2: toplam alım gücü
def test_reg_t_alim_gucu_portfoy_toplamidir_ve_2x(monkeypatch):
    monkeypatch.setitem(vars(CONFIG.portfolio), "max_position_notional", 0.0)
    ps = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    rm = RiskManager(ps)
    units, why = rm._magnitude_cap_units(unit_notional=100.0, unit_risk=0.1, sign=+1)
    assert units == pytest.approx(2000) and why == "INSUFFICIENT_BUYING_POWER"   # 2× (eskiden 4×)
    # 150 k$'lık açık SHORT alım gücünün 150 k$'ını TÜKETİR (eskiden hiç düşülmüyordu)
    ps.open_positions.append(VirtualPosition(symbol="XOM", asset_type="stock", side="SHORT",
                                             quantity=1500, entry_price=100.0,
                                             entry_time=date(2024, 1, 2), stop=100.05))
    units, why = rm._magnitude_cap_units(unit_notional=100.0, unit_risk=0.1, sign=+1)
    assert units == pytest.approx(500) and why == "INSUFFICIENT_BUYING_POWER"


def test_eski_4x_formulu_geri_gelmedi():
    kaynak = (KOK / "nihai/risk/risk_manager.py").read_text(encoding="utf-8")
    assert "buying_power / (margin * unit_notional)" not in kaynak


# ============================================================ BULGU-89-3: T+1 hasılat
def test_short_kapanisi_takas_hasilati_uretmez(tmp_path):
    st = StateStore(tmp_path / "state.json")
    st.load_or_init(10_000)
    st.add_unsettled_proceeds(5_000, date(2026, 9, 22), "SHORT")
    assert st.unsettled_lots == []
    st.add_unsettled_proceeds(5_000, date(2026, 9, 22), "LONG")
    st.add_unsettled_proceeds(3_000, date(2026, 9, 22))          # varsayılan LONG (geri uyum)
    assert [lot["amount"] for lot in st.unsettled_lots] == [5_000, 3_000]


# ============================================================ BULGU-89-4: seans kapısı
def _katman(gate):
    taranan: list[str] = []
    t = TierScanner(name="test_katman", provider=None, get_symbols=lambda: ["A", "B"],
                    scan_one=lambda s, p: taranan.append(s), cycle_gap=0.01, gate=gate)
    t.start()
    time.sleep(0.3)
    t.stop()
    t.join(2)
    return taranan, t


def test_katman_seans_kapaliyken_taramaz():
    taranan, t = _katman(lambda: (False, "MARKET_CLOSED"))
    assert taranan == [] and t.cycles == 0


def test_katman_seans_acikken_tarar():
    taranan, t = _katman(lambda: (True, "OK"))
    assert taranan[:2] == ["A", "B"] and t.cycles >= 1


def test_main_katmanlara_seans_kapisini_baglar():
    kaynak = (KOK / "nihai/execution/main.py").read_text(encoding="utf-8")
    assert "gate=can_enter_new_position" in kaynak
    assert 'OBS.record_loop("scan", _elapsed)' not in kaynak       # periyot "gecikme" sanılıyordu


# ============================================================ BULGU-89-5: taşınan pozisyon adı
def test_yeniden_baslatmada_strateji_adi_ve_rr_korunur():
    ps = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    ps.open_positions += [
        VirtualPosition(symbol="PCG", asset_type="stock", side="SHORT", quantity=10,
                        entry_price=13.0, entry_time=datetime(2026, 9, 21, 11, 19),
                        stop=13.5, target=12.0, strategy="trend_pullback V1"),
        VirtualPosition(symbol="ADM", asset_type="stock", side="LONG", quantity=10,
                        entry_price=80.0, entry_time=datetime(2026, 9, 21, 12, 17),
                        stop=79.0, target=82.0),             # eski state: ad alanı yok
    ]
    pm = PositionManager(ps, record_logs=False)
    assert pm.rehydrate_trackers() == 2
    trk = {t.pos.symbol: t for t in pm._trackers.values()}
    assert trk["PCG"].strategy_name == "trend_pullback V1" and trk["PCG"].rr_planned == 2.0
    assert trk["ADM"].strategy_name == f"{ACTIVE_STRATEGY} V1" and trk["ADM"].rr_planned == 2.0
    assert "EventBreakout" not in trk["ADM"].strategy_name


def test_acilan_pozisyon_strateji_adini_tasir():
    ps = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    pm = PositionManager(ps, SlippageModel(impact_coef=0.0), record_logs=False)
    sig = Signal(SignalAction.LONG, "AAPL", 100.0, 99.0, 102.0, reasons=[], quality_score=1.0,
                 rr_ratio=2.0)
    pos = pm.open_from_signal(sig, {"shares": 10}, now=datetime(2026, 9, 22, 10, 0),
                              strategy_name="event_breakout V1")
    assert pos.strategy == "event_breakout V1"


# ============================================================ pano: [object Object] + saat dilimi
def test_pano_karne_ic_ice_alanlari_sayi_olarak_basar():
    html = (KOK / "nihai/web/web_dashboard.html").read_text(encoding="utf-8")
    assert "bilgi ${c.knowledge != null ? c.knowledge" not in html
    assert "_alan(c.knowledge, 'score')" in html


def test_pano_saat_dilimli_damgayi_yerel_saate_cevirir():
    html = (KOK / "nihai/web/web_dashboard.html").read_text(encoding="utf-8")
    govde = html.split("function fmtTs(ts)")[1].split("\n        }\n")[0]
    assert "new Date(s)" in govde
    assert "Bloke (T+1)" not in html


def test_hub_akis_karnesini_saglayici_zincirinden_okur(monkeypatch):
    from nihai.data import data_provider as dp

    hub = dp.HubProvider(base_url="http://127.0.0.1:1")
    monkeypatch.setattr(hub, "_get", lambda path, params: SimpleNamespace(
        json=lambda: {"streams": {"connected": True, "subscribed": ["AAPL"]}}))
    assert dp.stream_status(hub) == {"connected": True, "subscribed": ["AAPL"]}
    monkeypatch.setattr(hub, "_get", lambda path, params: None)
    assert dp.stream_status(hub) is None


def test_katman_kapisi_threadi_bekletmez():
    """Kapı kapalıyken 30 sn bekleme durdurmaya DUYARLI olmalı (servis kapanışı takılmasın)."""
    t = TierScanner(name="test_bekleme", provider=None, get_symbols=lambda: [],
                    scan_one=lambda s, p: None, cycle_gap=5.0, gate=lambda: (False, ""))
    t.start()
    time.sleep(0.1)
    t0 = time.monotonic()
    t.stop()
    t.join(3)
    assert not t.is_alive() and time.monotonic() - t0 < 2
