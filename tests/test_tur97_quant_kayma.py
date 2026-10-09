"""Tur-97 — algoritmaların quant tarafına kaydırılması: yetenek sözleşmeleri + regresyon kapıları.

Kilitlenen: `nihai/signals/quant.py` (nedensellik, skor, EWMA σ, günlük bar sözleşmesi), `quant_trend`
kipinin karar yolu (TA kapılarından muafiyet, σ stop, kalite ölçeği, df_daily), quant pozisyon çıkışı
(izleyen stop, sabit hedef/seans düzleşmesi/zaman stopu yok), backtest'in strateji etiketi, canlı
günlük bar önbelleği ve `research.quant_kayma` gecikme sarmalayıcısı (BULGU-96-5 tekrarı yok).
"""
from __future__ import annotations

import inspect
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nihai.config import CONFIG
from nihai.signals import quant as Q
from nihai.signals.strategy import (
    BASE_STRATEGY_MODES,
    STRATEGY_MODES,
    Signal,
    SignalAction,
    _SharedIndicators,
    detect_quant_trend,
    generate_signal,
    is_quant_label,
)

ROOT = Path(__file__).resolve().parents[1]


def _gunluk(close, bas="2015-01-02", birim="us") -> pd.DataFrame:
    c = np.asarray(close, float)
    idx = pd.bdate_range(bas, periods=c.size).as_unit(birim)
    return pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c,
                         "volume": np.full(c.size, 1e6)}, index=idx)


def _yuruyus(n=600, mu=0.0008, sd=0.01, seed=0) -> np.ndarray:
    r = np.random.default_rng(seed).normal(mu, sd, n)
    return 100 * np.exp(np.cumsum(r))


# ---------------------------------------------------------------------------------------------------
# quant.py çekirdeği
# ---------------------------------------------------------------------------------------------------
def test_seriler_nedensel_gelecek_eklemek_gecmisi_degistirmez():
    c = _yuruyus(700, seed=1)
    tam_s, tam_g = Q.seriler(c)
    for k in (260, 400, 699):
        s, g = Q.seriler(c[:k])
        np.testing.assert_array_equal(s, tam_s[:k])
        np.testing.assert_allclose(g, tam_g[:k], rtol=0, atol=0)


def test_skor_ufuk_isaretlerinin_ortalamasi_ve_252_oncesi_tanimsiz():
    c = np.linspace(100, 200, 400)                       # hep yükselen → üç ufuk da +
    s, _ = Q.seriler(c)
    assert np.isnan(s[: Q.UFUKLAR[-1]]).all() and np.isfinite(s[Q.UFUKLAR[-1]:]).all()
    assert s[-1] == 1.0
    assert Q.seriler(c[::-1])[0][-1] == -1.0
    # 21 günlük düşüş, 63 ve 252 günlük yükseliş → (−1 + 1 + 1)/3
    c2 = np.concatenate([np.linspace(100, 200, 380), np.linspace(199, 190, 21)])
    assert abs(Q.seriler(c2)[0][-1] - 1 / 3) < 1e-12


def test_sigma_ewma_elle_ozyinelemeyle_bit_denk():
    c = _yuruyus(300, seed=2)
    lr = np.diff(np.log(c))
    v = lr[0] ** 2
    beklenen = [np.nan, np.sqrt(v)]
    for x in lr[1:]:
        v = Q.EWMA_LAMBDA * v + (1 - Q.EWMA_LAMBDA) * x * x
        beklenen.append(np.sqrt(v))
    np.testing.assert_allclose(Q.seriler(c)[1], np.array(beklenen), rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("birim", ["ns", "us", "s"])
def test_gunluk_mu_indeks_biriminden_bagimsiz(birim):
    """BULGU-97-1: pandas 3 indeksi `datetime64[us]` tutar; `asi8` + sabit ns eşiği günlük barı
    saatlik sanıyordu ve quant kipi 14 piyasanın HEPSİNDE sıfır işlem üretiyordu."""
    gun = pd.bdate_range("2020-01-01", periods=30).as_unit(birim)
    saat = pd.date_range("2020-01-02 09:30", periods=30, freq="h").as_unit(birim)
    assert Q.gunluk_mu(gun) is True or Q.gunluk_mu(gun) == np.True_
    assert not Q.gunluk_mu(saat)
    assert Q.gunluk_mu(pd.RangeIndex(30))                # tarihsiz sentetik veri: çağırana güven


def test_gunluk_mu_kaynaginda_asi8_yok():
    assert ".asi8" not in inspect.getsource(Q.gunluk_mu)


def test_stop_hedef_uc_sigma_ve_uc_r():
    s, h = Q.stop_hedef(100.0, "LONG", 0.02)
    assert (s, h) == pytest.approx((94.0, 118.0))
    s, h = Q.stop_hedef(100.0, "SHORT", 0.02)
    assert (s, h) == pytest.approx((106.0, 82.0))


def test_kalite_ta_olceginde_ve_hedef_agirlik_tavanli():
    assert Q.kalite(1.0)[0] == 6.5 and Q.kalite(-1 / 3)[0] == pytest.approx(2.17)
    assert Q.hedef_agirlik(1.0, 0.01, 0.02) == 2.0          # tavan 2
    assert Q.hedef_agirlik(-1 / 3, 0.02, 0.01) == pytest.approx(-1 / 6)
    assert Q.hedef_agirlik(1.0, 0.0, 0.01) == 0.0


def test_durum_ozeti():
    d = Q.durum(_gunluk(np.linspace(100, 150, 300)))
    assert d["hazir"] and d["yon"] == "LONG" and d["skor"] == 1.0 and set(d["ufuk_getiri"]) == {"21", "63", "252"}
    assert Q.durum(_gunluk(np.linspace(100, 150, 100)))["neden"] == "YETERSIZ_GECMIS"


# ---------------------------------------------------------------------------------------------------
# Strateji kipi
# ---------------------------------------------------------------------------------------------------
def test_kip_katalogu():
    assert "quant_trend" in STRATEGY_MODES
    assert "quant_trend" not in BASE_STRATEGY_MODES      # Tur-96 Hedge uzman listesi değişmesin
    from nihai.runtime_strategy import validate
    assert validate({"strategy": "quant_trend", "min_votes": 2, "stop_atr": 1.5, "target_atr": 3.0})
    assert is_quant_label("QuantTrend") and is_quant_label("quant_trend V1") and not is_quant_label("MACD V1")
    assert not is_quant_label("") and not is_quant_label(None)


def test_uzun_yalniz_varsayilan_kisa_bayrakla():
    df = _gunluk(np.linspace(200, 100, 300))
    a, r = detect_quant_trend(df)
    assert a == SignalAction.WAIT and "QUANT_KISA_KAPALI" in r
    eski = CONFIG.strategy.quant_allow_short
    object.__setattr__(CONFIG.strategy, "quant_allow_short", True)
    try:
        assert detect_quant_trend(df)[0] == SignalAction.SHORT
    finally:
        object.__setattr__(CONFIG.strategy, "quant_allow_short", eski)


def test_saatlik_barla_quant_reddedilir_gunlukle_calisir():
    c = _yuruyus(600, seed=3)
    saat = _gunluk(c)
    saat.index = pd.date_range("2024-01-02 09:30", periods=len(c), freq="h")
    sig = generate_signal("X", saat, strategy_mode="quant_trend")
    assert sig.action == SignalAction.WAIT and sig.reasons == ["QUANT_GUNLUK_BAR_GEREKLI"]
    gunluk = _gunluk(np.linspace(100, 160, 300))
    sig = generate_signal("X", saat, strategy_mode="quant_trend", df_daily=gunluk)
    assert sig.action == SignalAction.LONG and sig.metadata["quant"]["skor"] == 1.0


def test_paylasimli_onbellek_dilimle_bit_denk():
    df = _gunluk(_yuruyus(700, seed=4))
    sh = _SharedIndicators(df)
    for i in (300, 450, 699):
        a1 = detect_quant_trend(df.iloc[: i + 1], shared=sh.upto(i + 1))
        a2 = detect_quant_trend(df.iloc[: i + 1])
        assert a1 == a2


def test_quant_stop_sigma_kalite_skor_ve_ta_kapilarindan_muaf():
    """Dik yükselişte RSI aşırı-uzama freni TA kiplerini keser; quant kipi (momentumun tanımı gereği)
    girmeli. Stop = fiyat − 3σ·fiyat, hedef 3R, kalite = 6,5·|skor|."""
    c = np.concatenate([np.full(60, 100.0), 100 * np.exp(np.cumsum(np.full(260, 0.004)))])
    c = c * (1 + 0.002 * np.sin(np.arange(c.size)))       # σ > 0
    df = _gunluk(c)
    eski = CONFIG.strategy.rsi_chase_long
    object.__setattr__(CONFIG.strategy, "rsi_chase_long", 50.0)   # her yükselişte "aşırı uzamış"
    try:
        sig = generate_signal("X", df, strategy_mode="quant_trend")
    finally:
        object.__setattr__(CONFIG.strategy, "rsi_chase_long", eski)
    assert sig.action == SignalAction.LONG
    assert not any(r.startswith("RSI_") for r in sig.reasons)
    s, g = Q.seriler(df["close"].to_numpy())
    beklenen_stop, beklenen_hedef = Q.stop_hedef(float(df["close"].iloc[-1]), "LONG", float(g[-1]))
    assert sig.stop == pytest.approx(beklenen_stop) and sig.target == pytest.approx(beklenen_hedef)
    assert sig.rr_ratio == pytest.approx(3.0) and sig.quality_score == pytest.approx(6.5 * abs(s[-1]))
    assert sig.metadata["strategy"] == "QuantTrend"


def test_quant_sinyali_ta_meta_kapisina_sokulmaz(monkeypatch):
    """BULGU-97-4: TA ile eğitilmiş meta kapı quant sinyalinde p_win=0,00 (dağılım-dışı doyum)
    yazıyordu; gölgede bile kanıt döngüsünü kirletir. Quant'ta kapı hiç çağrılmaz."""
    import nihai.signals.meta_gate as MG

    def patla():
        raise AssertionError("meta kapı quant sinyalinde çağrıldı")

    monkeypatch.setattr(MG, "get_gate", patla)
    eski = CONFIG.strategy.meta_gate_mode
    object.__setattr__(CONFIG.strategy, "meta_gate_mode", "shadow")
    try:
        sig = generate_signal("X", _gunluk(np.linspace(100, 160, 300)), strategy_mode="quant_trend")
    finally:
        object.__setattr__(CONFIG.strategy, "meta_gate_mode", eski)
    assert sig.action == SignalAction.LONG and "META_GATE_ERROR" not in sig.reasons
    assert sig.metadata["meta_gate"] == "KAPSAM_DISI_QUANT" and "meta_p_win" not in sig.metadata


def test_quant_opsiyon_zincirini_cekmez():
    df = _gunluk(np.linspace(100, 160, 300))
    cagri = []
    sig = generate_signal("X", df, strategy_mode="quant_trend",
                          chain_eval=lambda: cagri.append(1) or (True, 50.0))
    assert sig.action == SignalAction.LONG and cagri == []


# ---------------------------------------------------------------------------------------------------
# Pozisyon çıkışı
# ---------------------------------------------------------------------------------------------------
def _pmgr():
    from nihai.risk.position_manager import PositionManager, SlippageModel
    from nihai.risk.risk_manager import PortfolioState
    ps = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    return ps, PositionManager(ps, SlippageModel(stock_slippage_pct=0.0, option_slippage_pct=0.0))


def _ac(pmgr, strateji, *, now=datetime(2024, 1, 2, 10, 0)):
    sig = Signal(action=SignalAction.LONG, symbol="AAA", price=100.0, stop=94.0, target=118.0, rr_ratio=3.0)
    return pmgr.open_from_signal(sig, {"shares": 10, "risk_dollars": 60.0}, now=now,
                                 asset_type="stock", strategy_name=strateji)


def test_quant_pozisyon_izleyen_stop_hedefsiz():
    _, pm = _pmgr()
    pos = _ac(pm, "QuantTrend V1")
    # Hedefin (118) ötesine gider: TA pozisyonu TARGET ile kapanırdı, quant koşmaya devam eder
    kapanan = pm.update_all_with_bars({"AAA": (101.0, 125.0, 100.0, 124.0)}, now=datetime(2024, 1, 3))
    assert kapanan == []
    assert pos.stop == pytest.approx(125.0 - 6.0)           # tepe − başlangıç riski (1R = 6)
    # Stop yalnız lehe ratchet: geri çekilme stop'u gevşetmez
    pm.update_all_with_bars({"AAA": (123.0, 123.5, 120.0, 121.0)}, now=datetime(2024, 1, 4))
    assert pos.stop == pytest.approx(119.0)
    kapanan = pm.update_all_with_bars({"AAA": (120.0, 120.5, 118.0, 118.5)}, now=datetime(2024, 1, 5))
    assert len(kapanan) == 1 and kapanan[0].exit_trigger == "STOP"


def test_quant_izleyen_mesafe_yeniden_baslatmada_korunur():
    """BULGU-97-3: tracker bellekte yaşar; yeniden başlatmada başlangıç stopu pozisyonun O ANKİ
    (lehe kaymış) stopundan kurulur. Kalıcı mesafe olmadan 3σ mesafe |giriş − stop|'a büzülürdü."""
    from nihai.execution.state import _deserialize_position, _serialize
    from nihai.risk.position_manager import PositionManager, SlippageModel
    from nihai.risk.risk_manager import PortfolioState
    _, pm = _pmgr()
    pos = _ac(pm, "QuantTrend V1")
    assert pos.izleyen_mesafe == pytest.approx(6.0)
    pm.update_all_with_bars({"AAA": (101.0, 110.0, 100.0, 109.0)}, now=datetime(2024, 1, 3))
    assert pos.stop == pytest.approx(104.0)                 # 110 − 6
    # "yeniden başlatma": JSON gidiş-dönüş + yeni yönetici + rehydrate
    geri = _deserialize_position(_serialize(pos))
    assert geri.izleyen_mesafe == pytest.approx(6.0) and geri.strategy == "QuantTrend V1"
    ps2 = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    ps2.open_positions.append(geri)
    pm2 = PositionManager(ps2, SlippageModel(stock_slippage_pct=0.0, option_slippage_pct=0.0))
    assert pm2.rehydrate_trackers() == 1
    pm2.update_all_with_bars({"AAA": (109.0, 115.0, 108.0, 114.0)}, now=datetime(2024, 1, 4))
    assert geri.stop == pytest.approx(115.0 - 6.0)          # büzülmüş |100−104|=4 değil, 6
    # eski kayıt (alan yok) okunabilir ve TA pozisyonunda mesafe 0
    eski = {k: v for k, v in _serialize(pos).items() if k != "izleyen_mesafe"}
    assert _deserialize_position(eski).izleyen_mesafe == 0.0
    _, pm3 = _pmgr()
    assert _ac(pm3, "EventBreakout V1").izleyen_mesafe == 0.0


def test_quant_capala_mesafeyi_dolum_fiyatina_tasir_ta_dokunulmaz():
    """BULGU-97-2: motor stopu karar kapanışına göre kurup açılıştan giriyordu; gece boşluğu / +1 bar
    gecikme LONG stopunu girişin ÜSTÜNE taşıyabiliyordu (SPY: gecikmede GAP_STOP 17 → 33)."""
    from nihai.signals.strategy import quant_capala, reanchor_to_live
    q = Signal(action=SignalAction.LONG, symbol="A", price=100.0, stop=94.0, target=118.0, rr_ratio=3.0,
               metadata={"strategy": "QuantTrend"})
    t = Signal(action=SignalAction.LONG, symbol="A", price=100.0, stop=94.0, target=118.0, rr_ratio=3.0,
               metadata={"strategy": "EventBreakout"})
    q2 = quant_capala(q, 92.0)
    assert (q2.stop, q2.target, q2.price) == pytest.approx((86.0, 110.0, 100.0))
    assert quant_capala(t, 92.0) is t
    # canlı çapa: TA'da fiyat stopun ötesinde → BAYAT; quant'ta mesafe taşınır, RR aynı kalır
    assert reanchor_to_live(t, 92.0, min_rr=1.2)[0] is None
    s, why = reanchor_to_live(q, 92.0, min_rr=1.2)
    assert why == "" and s.stop == pytest.approx(86.0) and s.rr_ratio == pytest.approx(3.0) and s.price == 92.0
    from nihai.execution import backtest as B
    assert inspect.getsource(B.backtest_symbol).count("quant_capala(sig, float(_o[i + 1]))") == 1
    assert inspect.getsource(B.backtest_portfolio).count("quant_capala(sig, float(o_a[i + 1]))") == 1


def test_ta_pozisyonu_eski_davranista():
    _, pm = _pmgr()
    _ac(pm, "EventBreakout V1")
    kapanan = pm.update_all_with_bars({"AAA": (101.0, 125.0, 100.0, 124.0)}, now=datetime(2024, 1, 3))
    assert len(kapanan) == 1 and kapanan[0].exit_trigger == "TARGET"


def test_quant_pozisyonu_seans_sonu_ve_zaman_stopundan_muaf():
    """Canlıda FLAT_BEFORE_CLOSE_MIN=10 ve TIME_STOP_MINUTES=90: günlük ufuklu pozisyonu her gün
    kapatıp ertesi gün yeniden açtırırdı. TA pozisyonu aynı barda SESSION_FLAT ile kapanır."""
    kapanis_oncesi = datetime(2024, 1, 3, 15, 55)          # ET, seans kapanışına 5 dk
    for ad, beklenen in (("QuantTrend V1", []), ("EventBreakout V1", ["SESSION_FLAT"])):
        _, pm = _pmgr()
        _ac(pm, ad, now=datetime(2024, 1, 3, 10, 0))
        kapanan = pm.update_all_with_bars({"AAA": (100.0, 100.2, 99.9, 100.1)}, now=kapanis_oncesi)
        assert [k.exit_trigger for k in kapanan] == beklenen, ad


def test_backtest_pozisyona_strateji_etiketini_yazar():
    from nihai.execution import backtest as B
    src = inspect.getsource(B.backtest_symbol) + inspect.getsource(B.backtest_portfolio)
    assert src.count("strategy_name=f\"{sig.metadata.get('strategy', 'EventBreakout')} V1\"") == 2


def test_backtest_quant_kipi_hedefle_cikmaz():
    from nihai.execution.backtest import backtest_symbol
    r = np.random.default_rng(7).normal(0.0006, 0.012, 900)
    df = _gunluk(100 * np.exp(np.cumsum(r)))
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["high"] = df[["open", "close"]].max(axis=1) * 1.004
    df["low"] = df[["open", "close"]].min(axis=1) * 0.996
    res = backtest_symbol("SYN", "1d", df=df, strategy_mode="quant_trend", warmup_bars=260,
                          disable_kill_switch=True)
    assert res.trades_count > 0


# ---------------------------------------------------------------------------------------------------
# Canlı günlük bar önbelleği
# ---------------------------------------------------------------------------------------------------
def test_canli_gunluk_bar_gunde_bir_kez_ve_bos_yanit_onbelleklenmez(monkeypatch):
    import nihai.execution.main as M
    cagri = []

    class P:
        bos = True

        def get_bars(self, sym, interval, lookback_bars=200):
            cagri.append((sym, interval, lookback_bars))
            return pd.DataFrame() if self.bos else _gunluk(np.linspace(100, 120, 300))

    monkeypatch.setattr(M, "_GUNLUK_ONBELLEK", {})
    p = P()
    assert M._gunluk_barlar("AAA", p).empty and M._gunluk_barlar("AAA", p).empty
    assert len(cagri) == 2                                  # boş yanıt önbelleğe yazılmadı
    p.bos = False
    for _ in range(5):
        assert len(M._gunluk_barlar("AAA", p)) == 300
    assert len(cagri) == 3 and cagri[-1] == ("AAA", "1d", M._QUANT_GUNLUK_BAR)
    assert M._QUANT_GUNLUK_BAR >= Q.MIN_BAR


def test_canli_tarama_quant_kipinde_gunluk_bar_gecirir():
    import nihai.execution.main as M
    src = inspect.getsource(M.scan_symbol)
    assert 'strategy_mode == "quant_trend" and HTF_INTERVAL != "1d"' in src
    assert "df_daily=df_daily" in src


def test_canli_quant_sembol_basina_tek_pozisyon_ve_agdan_once():
    """BULGU-97-8: durum sinyali her taramada yeniden ateşler; küme sınırı (2) TA için kuruldu. Replay
    (2026-09-25, AAPL+MSFT) muhafızsız 3 pozisyon açtı. Muhafız bar çekiminden ÖNCE durmalı."""
    import nihai.execution.main as M
    src = inspect.getsource(M.scan_symbol)
    i = src.index('if strategy_mode == "quant_trend" and any(p.symbol == symbol')
    assert 'log_block("QUANT_POZISYON_VAR"' in src[i:i + 300]
    assert i < src.index("provider.get_bars(symbol, HTF_INTERVAL")


def test_quant_muhafizi_davranista_ikinci_pozisyonu_engeller(monkeypatch):
    import nihai.execution.main as M
    engellenen, cekilen = [], []
    monkeypatch.setattr(M, "log_block", lambda tag, msg: engellenen.append(tag))
    monkeypatch.setattr(M, "can_enter_new_position", lambda: (True, ""))
    monkeypatch.setattr(M, "event_guard", lambda sym: (True, "sahte"))   # muhafızı geçen sembol burada durur
    monkeypatch.setattr(M, "_quant_odak", lambda: frozenset({"AAA", "BBB"}))   # Tur-101: quant evreni = odak

    class Prov:
        def get_bars(self, *a, **k):
            cekilen.append(a)
            return pd.DataFrame()

    class Store:
        kill_switch_active = False
        symbol_cooldowns: dict = {}                                     # Tur-101: aynı-gün kuralı okur

        def cooldown_active(self, *a):
            return False

    _, pm = _pmgr()
    _ac(pm, "quant_trend V1")
    M.scan_symbol("AAA", Prov(), None, Store(), None, pm, strategy_mode="quant_trend")
    assert engellenen == ["QUANT_POZISYON_VAR"] and cekilen == []
    M.scan_symbol("BBB", Prov(), None, Store(), None, pm, strategy_mode="quant_trend")
    assert engellenen == ["QUANT_POZISYON_VAR", "EVENT_GUARD"]            # pozisyonsuz sembol geçer
    M.scan_symbol("AAA", Prov(), None, Store(), None, pm, strategy_mode="event_breakout")
    assert engellenen[-1] == "EVENT_GUARD"                                 # TA kipinde küme sınırı yine geçerli


# ---------------------------------------------------------------------------------------------------
# Araştırma düzeneği
# ---------------------------------------------------------------------------------------------------
def test_gecikme_sarmalayicisi_zincirlenmez_ve_geri_yuklenir():
    import nihai.execution.backtest as B
    from research import quant_kayma as QK
    orj = B.generate_signal
    gorulen = []

    def casus(symbol, df, *a, shared=None, **kw):
        gorulen.append((len(df), shared.nbars if shared is not None else None))
        return orj(symbol, df, *a, shared=shared, **kw)

    QK._ORIJINAL.pop("gs", None)
    B.generate_signal = casus
    try:
        df = _gunluk(_yuruyus(300, seed=5))
        sh = _SharedIndicators(df)
        for _ in range(3):                                  # BULGU-96-5: 3 görev ardışık → yine 1 bar
            with QK.gecikmeli(1):
                B.generate_signal("X", df.iloc[:280], shared=sh.upto(280))
        assert gorulen == [(279, 279)] * 3
    finally:
        B.generate_signal = orj
        QK._ORIJINAL.pop("gs", None)
    assert B.generate_signal is orj


def test_fark_t_eslestirilmis():
    from research import quant_kayma as QK
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    a = b + 0.001
    d = QK.fark_t(a, b)
    assert d["n"] == 400 and d["yillik_pp"] == pytest.approx(25.2) and (d["t"] is None or d["t"] > 10)


def test_modul_ve_kip_siniflandirildi():
    from research import coklu_piyasa as K
    assert K.MODUL_SINIFI["nihai.signals.quant"][0] == "algoritma"
    assert K.ALGORITMALAR["quant_trend"].kaynak.endswith("detect_quant_trend")


def test_test_paketi_operatorun_stratejisini_izlemez():
    """Canlı `.env` quant_trend'e geçince stratejisiz `generate_signal` çağıran üç Tur-44 testi kırmızıya düştü:
    paket operatörün `.env`'ini okuyordu. conftest kod varsayılanını sabitler."""
    import os

    from nihai import config
    assert os.environ["ACTIVE_STRATEGY"] == "event_breakout" == config.ACTIVE_STRATEGY
    assert 'os.environ["ACTIVE_STRATEGY"] = "event_breakout"' in (ROOT / "conftest.py").read_text(encoding="utf-8")


def test_env_ornegi_beyanli():
    ornek = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "QUANT_KISA=" in ornek and "quant_trend" in ornek
