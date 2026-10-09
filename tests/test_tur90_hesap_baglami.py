"""Tur-90 — çalışan sistemin HESAP BAĞLAMI hataları (2026-09-23).

Kilitler (her biri kusurun ölçülmüş hâlinden):
  BULGU-90-1  Canlı Alpaca barları bölünme/temettü DÜZELTİLMEMİŞ (APH 1h −%49,8, CRWD 1d −%74,9).
  BULGU-90-2  Canlı Alpaca barları uzatılmış seansı taşıyor (Madde 22); 09:00 saat barı seans-öncesini içeriyor.
  BULGU-90-3  Akış sağlayıcısının 1d serisinde her gün İKİ KEZ vardı (UTC 00:00 canlı + 04:00 REST).
  BULGU-90-4  PDT penceresi takvim bloğuydu (yürüyen 5 iş günü değil); pano açılış gününü bir gün geri yazıyordu.
  BULGU-90-5  yfinance yedeği ET-tz'li index döndürüyordu ("naive = UTC" sözleşmesi dışı).
  BULGU-90-6  Karar motoru kapı-a kenarı |2p−1|: çözülen tarafa ALEYHTE posterior pozitif kenar sayılıyordu.
  BULGU-90-7  Gölge AI kararı kapalı piyasada yazılıyor, 1m kararları 5 GÜNLÜK ufuklu epizodik hafızaya giriyordu.
"""
from __future__ import annotations

from collections import deque
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import nihai.data.alpaca_stream_provider as asp
from nihai.data import alpaca_provider as ap
from nihai.data import data_provider as dp
from nihai.risk.risk_manager import PortfolioState, pdt_sessions

KOK = Path(__file__).resolve().parents[1]


def _utc(*ts: str) -> pd.DatetimeIndex:
    """ET duvar saatleri → naive UTC index (sistem sözleşmesi)."""
    return pd.DatetimeIndex(ts).tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)


# ============================================================ BULGU-90-2: seans süzgeci
def test_rth_mask_uzatilmis_seans_ve_takvim():
    idx = _utc("2026-09-22 08:00", "2026-09-22 09:00", "2026-09-22 15:00", "2026-09-22 16:00",
               "2026-07-03 10:00",               # Bağımsızlık Günü (gözlenen) — seans yok
               "2026-11-27 12:00", "2026-11-27 13:00")   # Şükran ertesi yarım gün (13:00 kapanış)
    assert dp.rth_mask(idx, 60).tolist() == [False, True, True, False, False, True, False]
    # 30 dk alt bar: 09:00–09:30 seans-öncesidir, 09:30 değil
    assert dp.rth_mask(_utc("2026-09-22 09:00", "2026-09-22 09:30"), 30).tolist() == [False, True]


def _alt_barlar() -> pd.DataFrame:
    idx = _utc("2026-09-22 09:00", "2026-09-22 09:30", "2026-09-22 10:00", "2026-09-22 10:30",
               "2026-09-22 16:00")
    return pd.DataFrame({"open": [90.0, 100, 102, 103, 99], "high": [150.0, 101, 104, 105, 99],
                         "low": [50.0, 99, 101, 102, 99], "close": [95.0, 100.5, 103, 104, 99],
                         "volume": [10.0, 1000, 800, 600, 5], "trade_count": [1.0, 50, 40, 30, 1],
                         "vwap": [95.0, 100.0, 103.0, 104.0, 99.0]}, index=idx.tz_localize("UTC"))


def test_normalize_1h_alt_barlardan_saf_seans():
    out = ap._normalize_alpaca_bars(_alt_barlar(), "X", "1h", 200, rth=True, src_minutes=30)
    et = out.index.tz_localize("UTC").tz_convert("America/New_York").strftime("%H:%M").tolist()
    # Tur-91 (BULGU-91-1): ızgara seans açılışına çapalı (yfinance yedeğiyle aynı); Tur-90'da 09:00/10:00'dı
    assert et == ["09:30", "10:30"], "16:00 seans-sonrası barı kalmış ya da ızgara seansa çapalı değil"
    b1 = out.iloc[0]
    # 09:30 barı 09:30–10:30: seans-öncesinin 150/50 tepe-dibi ve 90 açılışı YOK
    assert (b1.open, b1.high, b1.low, b1.close, b1.volume, b1.trade_count) == (
        100.0, 104.0, 99.0, 103.0, 1800.0, 90.0)
    assert b1.vwap == pytest.approx((100 * 1000 + 103 * 800) / 1800)
    b2 = out.iloc[1]
    assert (b2.open, b2.high, b2.close, b2.volume) == (103.0, 105.0, 104.0, 600.0)


def test_normalize_varsayilan_suzmez_arastirma_yolu_degismez():
    out = ap._normalize_alpaca_bars(_alt_barlar(), "X", "1h", 200)
    assert len(out) == 5


class _YakalayanIstemci:
    def __init__(self):
        self.req = None

    def get_stock_bars(self, req):
        self.req = req
        return SimpleNamespace(df=_alt_barlar())


def test_canli_get_bars_duzeltmeli_ve_30dk_alt_bar_ister(monkeypatch):
    """BULGU-90-1 + 90-2: canlı istek adjustment=all taşır ve 1h'yi 30 dk alt bardan kurar."""
    monkeypatch.delenv("ALPACA_HIST_ADJUSTMENT", raising=False)
    ist = _YakalayanIstemci()
    p = ap.AlpacaProvider(api_key="k", secret_key="s", data_client=ist)
    def kip():
        return str(getattr(ist.req.adjustment, "value", ist.req.adjustment))
    df = p.get_bars("X", "1h", lookback_bars=50)
    # Tur-91 (BULGU-91-2): gün içi yalnız bölünme (yfinance yedeği gibi); günlük bölünme + temettü
    assert kip() == "split"
    assert ist.req.timeframe.amount_value == 30 and "min" in str(ist.req.timeframe.unit_value).lower()
    assert len(df) == 2
    p.get_bars("X", "15m", lookback_bars=50)
    assert ist.req.timeframe.amount_value == 15 and kip() == "split"
    p.get_bars("X", "1d", lookback_bars=50)
    assert kip() == "all"


def test_duzeltme_kipi_tek_kaynak():
    from nihai.data import alpaca_history as ah
    assert ah._adjustment is ap._adjustment, "araştırma ve canlı yol ayrı düzeltme kipi okuyor"


# ============================================================ BULGU-90-3: akış 1d birleşimi
class _Tohum:
    def get_bars(self, symbol, interval, lookback):
        idx = pd.to_datetime(["2026-09-18 04:00", "2026-09-21 04:00", "2026-09-22 04:00"])
        return pd.DataFrame({"open": [1.0, 2, 3], "high": [1.0, 2, 3], "low": [1.0, 2, 3],
                             "close": [1.0, 2, 3], "volume": [1.0, 1, 1]}, index=idx)


def _akis_saglayici() -> asp.AlpacaStreamProvider:
    p = asp.AlpacaStreamProvider(symbols=[], rest=_Tohum(), max_symbols=30)
    dk = []
    for gun in ("2026-09-21", "2026-09-22"):
        for t in pd.date_range(gun + " 12:00", periods=500, freq="1min", tz="UTC"):  # 08:00 ET'den
            dk.append((t, 5.0, 6.0, 0.5, 5.5, 10.0))
    p._minutes["AAPL"] = deque(dk)
    return p


def test_akis_1d_gunler_tekil_ve_rest_etiketinde():
    out = _akis_saglayici().get_bars("AAPL", "1d", lookback_bars=10)
    assert not out.index.duplicated().any()
    assert list(out.index) == list(pd.to_datetime(["2026-09-18 04:00", "2026-09-21 04:00",
                                                   "2026-09-22 04:00"]))
    son = out.iloc[-1]
    # REST açılışı korunur, tepe/dip birleşir, kapanış canlı, hacim yalnız SEANS dakikaları (390×10)
    assert (son.open, son.high, son.low, son.close, son.volume) == (3.0, 6.0, 0.5, 5.5, 3900.0)


def test_akis_gun_ici_toplam_seans_disi_dakikayi_almaz(monkeypatch):
    p = _akis_saglayici()
    p._rest = SimpleNamespace(get_bars=lambda *a, **k: pd.DataFrame())   # yalnız canlı toplam
    monkeypatch.setattr(asp, "datetime", SimpleNamespace(
        now=lambda tz=None: pd.Timestamp("2026-09-23 12:00", tz="UTC").to_pydatetime()))
    # Tur-106 (BULGU-106-5): tohum yokken tampon isteği KARŞILAMALI → 2 seans × 7 saat = 14 bar.
    # Süzgeç kalkarsa son 14 bar 16:00 ET saatini de taşır ve aşağıdaki iddia yine kırmızı olur.
    out = p.get_bars("AAPL", "1h", lookback_bars=14)
    et = out.index.tz_localize("UTC").tz_convert("America/New_York").hour.tolist()
    assert 8 not in et and all(9 <= h <= 15 for h in et)


# ============================================================ BULGU-90-5: yfinance index sözleşmesi
def test_yfinance_index_naive_utc(monkeypatch):
    gun_ici = pd.DataFrame({"Open": [1.0, 1.0], "High": [1.0, 1.0], "Low": [1.0, 1.0],
                            "Close": [1.0, 1.0], "Volume": [1, 1]},
                           index=pd.DatetimeIndex(["2026-09-21 15:30", "2026-09-22 15:30"],
                                                  tz="America/New_York"))
    gunluk = gun_ici.copy()
    gunluk.index = pd.DatetimeIndex(["2026-09-21", "2026-09-22"])
    monkeypatch.setattr(dp.yf, "download", lambda *a, **k: (gunluk if k.get("interval") == "1d"
                                                            else gun_ici).copy())
    y = dp.YFinanceProvider.__new__(dp.YFinanceProvider)
    y._session, y._yf_period = None, {}
    h = y.get_bars("SPY", "1h")
    assert h.index.tz is None and h.index[-1] == pd.Timestamp("2026-09-22 19:30")
    d = y.get_bars("SPY", "1d")
    assert d.index[-1] == pd.Timestamp("2026-09-22 04:00"), "günlük etiket Alpaca 1d ile aynı değil"


# ============================================================ BULGU-90-4: PDT yürüyen penceresi
def _ps(**kw) -> PortfolioState:
    return PortfolioState(equity=19_500.0, ath_equity=1e6, cash_balance=19_500.0, **kw)


def test_pdt_seanslari_hafta_sonunu_atlar():
    assert pdt_sessions(date(2026, 9, 27), 5) == [date(2026, 9, d) for d in (21, 22, 23, 24, 25)]


def test_pdt_pencere_yuruyen_blok_degil():
    """Eski blok kuralı: 25 Eyl'de başlayan pencere 1 Eki'de sıfırlanır → 30 Eyl'in 3'ü + 1 Eki'nin 3'ü
    = 5 iş gününde 6 day-trade. Yürüyen pencerede 30 Eyl'in işlemleri 1 Eki'de HÂLÂ sayılır."""
    p = _ps(day_trade_dates=[date(2026, 9, 30)] * 3)
    assert p.pdt_refresh(date(2026, 10, 1), 5) == 3
    assert p.pdt_refresh(date(2026, 10, 6), 5) == 3      # 30 Eyl, 5. seans (30,1,2,5,6)
    assert p.pdt_refresh(date(2026, 10, 7), 5) == 0


def test_pdt_acilis_gunu_ilk_uygun_seans():
    p = _ps(day_trade_dates=[date(2026, 9, 21)] * 33 + [date(2026, 9, 22)] * 64)
    assert p.pdt_unlock_date(date(2026, 9, 23), 5, 3) == date(2026, 9, 29)   # Salı; Cumartesi DEĞİL
    assert _ps().pdt_unlock_date(date(2026, 9, 23), 5, 3) is None
    # sınırda: 28 Eyl'de pencerede TAM 3 (22 Eyl) kalır → hâlâ kilitli; kural "< limit"
    sinir = _ps(day_trade_dates=[date(2026, 9, 21)] * 5 + [date(2026, 9, 22)] * 3)
    assert sinir.pdt_unlock_date(date(2026, 9, 23), 5, 3) == date(2026, 9, 29)


def test_pdt_kaydi_gunu_tutar():
    p = _ps()
    p.register_day_trade(date(2026, 9, 22))
    assert (p.day_trade_dates, p.day_trades_count) == ([date(2026, 9, 22)], 1)


def test_pdt_tarihsiz_eski_sayac_pencere_basina_sayilir():
    p = _ps(day_trades_count=97, day_trades_window_start=date(2026, 9, 21))
    assert p.pdt_refresh(date(2026, 9, 23), 5) == 97
    assert p.pdt_refresh(date(2026, 9, 28), 5) == 0


def test_pdt_gocu_defterden_kurar(tmp_path):
    from nihai.execution.state import StateStore
    (tmp_path / "trades.csv").write_text(
        "exit_time,entry_time\n"
        f"{date.today().isoformat()}T15:50:00,{date.today().isoformat()}T10:00:00\n"
        f"{date.today().isoformat()}T15:50:00,2020-01-02T10:00:00\n", encoding="utf-8")
    st = StateStore(path=tmp_path / "state.json")
    assert st._day_trades_from_ledger() == [date.today()]


def test_pdt_status_ucu_yuruyen_sayim_ve_acilis_gunu(monkeypatch):
    from fastapi.testclient import TestClient

    import nihai.data.market_calendar as mc
    import nihai.web.web_dashboard as wd
    ps = _ps(day_trade_dates=[date(2026, 9, 21)] * 33 + [date(2026, 9, 22)] * 64)
    monkeypatch.setattr(wd, "get_portfolio_status", lambda: (wd.StateStore(), ps))
    monkeypatch.setattr(mc, "now_et_naive", lambda: pd.Timestamp("2026-09-23 02:00").to_pydatetime())
    d = TestClient(wd.app).get("/api/status").json()
    assert (d["day_trades_count"], d["pdt_unlock_date"]) == (97, "2026-09-29")


def test_pdt_kaynak_kapilari():
    rm = (KOK / "nihai/risk/risk_manager.py").read_text(encoding="utf-8")
    assert "date.today() - self.state.day_trades_window_start" not in rm
    html = (KOK / "nihai/web/web_dashboard.html").read_text(encoding="utf-8")
    assert "d.toISOString().slice(0, 10)} açılır" not in html
    assert "pdt_unlock_date" in (KOK / "nihai/web/web_dashboard.py").read_text(encoding="utf-8")


# ============================================================ BULGU-90-6: işaretli kenar
def test_karar_aleyhte_posteriorla_tarafi_onaylamaz(monkeypatch, tmp_path):
    from nihai.ai import decision, episodic, learning
    monkeypatch.delenv("AI_DECISION_MODE", raising=False)
    monkeypatch.setattr(decision, "DECISIONS_FILE", tmp_path / "d.jsonl")
    monkeypatch.setattr(episodic, "append_episode", lambda **kw: None)
    monkeypatch.setattr(learning, "load_state", lambda: {"strategies": {"event_breakout": {
        "n": 3000, "win": 1590, "loss": 1410, "post_mean": 0.53, "lo95": 0.51, "hi95": 0.55}}})
    monkeypatch.setattr(decision, "_estimate_gain_loss", lambda rets: (10.0, 0.1))   # kapı-b geçsin
    monkeypatch.setattr(decision, "_cvar_penalty", lambda rets: 0.0)
    d = decision.decide("LMT", "event_breakout", {}, p_win=0.05, side_hint="LONG",
                        closes=np.linspace(100, 101, 63))
    assert d.posterior < 0.5
    assert d.action == "ABSTAIN", "LONG için %39 posterior LONG_BIAS onayı aldı (|2p−1| kenarı)"
    assert "(2*" in " ".join(d.evidence)
    # taraf YOKSA mutlak kenar korunur (yön posterior işaretinden seçilir)
    d2 = decision.decide("LMT", "event_breakout", {}, p_win=0.95, closes=np.linspace(100, 101, 63))
    assert d2.action == "LONG_BIAS"


# ============================================================ BULGU-90-7: gölge AI bağlamı
def test_gun_ici_bar_karari_epizot_yazmaz(monkeypatch):
    from nihai.execution import ai_advisor
    gun_ici = pd.DataFrame(index=pd.date_range("2026-09-22 13:30", periods=40, freq="1min"))
    gunluk = pd.DataFrame(index=pd.bdate_range("2026-08-01", periods=40))
    assert ai_advisor._gunluk_bar(gun_ici) is False
    assert ai_advisor._gunluk_bar(gunluk) is True
    assert ai_advisor._gunluk_bar(None) is True


def test_record_episode_false_ani_yazmaz(monkeypatch, tmp_path):
    from nihai.ai import decision, episodic
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    monkeypatch.setattr(decision, "DECISIONS_FILE", tmp_path / "d.jsonl")
    yazilan = []
    monkeypatch.setattr(episodic, "append_episode", lambda **kw: yazilan.append(kw))
    decision.decide("X", "event_breakout", {}, record_episode=False)
    assert (tmp_path / "d.jsonl").exists() and yazilan == []
    decision.decide("X", "event_breakout", {})
    assert len(yazilan) == 1


def test_kapali_piyasada_golge_cikarim_yok():
    src = (KOK / "nihai/execution/live_predict_service.py").read_text(encoding="utf-8")
    assert "if side is not None and self._cycle_market_open is not False:" in src
    assert "record_episode=_gunluk_bar(df)" in (KOK / "nihai/execution/ai_advisor.py").read_text(
        encoding="utf-8")
