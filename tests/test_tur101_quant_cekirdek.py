"""Tur-101 — quant çekirdeğinin kendi kararları (çıkış, skor boyutu, portföy) + canlı/backtest sadakati.

Kilitlenen: `research/quant_cekirdek.py` (skor nedenselliği, kol yamaları ve geri yüklenmesi, sinyal çıkışının
zamanı, stopsuz kol, skor boyutu, FX kısa bayrağı, portföy planı, backtest_portfolio paritesi, G/P kapıları,
Z doğrulamasının bağlantı kuralı, pencere/hazırlık kuralları), `nihai/execution/backtest.py` (BULGU-101-5: günlük
sayaçlar her seans sıfırlanır), `nihai/execution/main.py` (BULGU-101-1 quant yalnız odak katmanında, BULGU-101-4
çıkıştan sonra aynı gün yeniden giriş yok) ve karnenin ön-kayıtlı hükmü.
"""
from __future__ import annotations

import inspect
import json
import math
import pickle
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nihai.config import CONFIG
from nihai.signals import quant as Q
from nihai.signals.strategy import Signal, SignalAction
from research import quant_cekirdek as QC

ROOT = Path(__file__).resolve().parents[1]


def _gunluk(close, bas="2012-01-02") -> pd.DataFrame:
    c = np.asarray(close, float)
    idx = pd.bdate_range(bas, periods=c.size)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996,
                         "close": c, "volume": np.full(c.size, 1e6)}, index=idx)


def _yuruyus(n=900, mu=0.0006, sd=0.012, seed=7) -> np.ndarray:
    return 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(mu, sd, n)))


def _pmgr():
    from nihai.risk.position_manager import PositionManager, SlippageModel
    from nihai.risk.risk_manager import PortfolioState
    ps = PortfolioState(equity=100_000, ath_equity=100_000, cash_balance=100_000)
    return ps, PositionManager(ps, SlippageModel(stock_slippage_pct=0.0, option_slippage_pct=0.0))


def _ac(pmgr, strateji="QuantTrend V1", sym="AAA"):
    sig = Signal(action=SignalAction.LONG, symbol=sym, price=100.0, stop=94.0, target=118.0, rr_ratio=3.0)
    return pmgr.open_from_signal(sig, {"shares": 10, "risk_dollars": 60.0}, now=datetime(2024, 1, 2, 10, 0),
                                 asset_type="stock", strategy_name=strateji)


def _motor(df, kol="q0", warmup=260):
    from nihai.execution.backtest import backtest_symbol
    from research import quant_karar as QKR
    with QC.cekirdek(kol, {"SYN": QC.skor_once(df)}), QKR.defter() as kay:
        r = backtest_symbol("SYN", "1d", df=df, strategy_mode="quant_trend", warmup_bars=warmup,
                            disable_kill_switch=True)
    return r, QKR.islemler(kay, df.index)


# ---------------------------------------------------------------------------------------------------
# Araştırma düzeneği — kollar
# ---------------------------------------------------------------------------------------------------
def test_skor_once_nedensel_ve_bir_bar_kaydirilmis():
    df = _gunluk(_yuruyus(400))
    sk, _ = Q.seriler(df["close"].to_numpy(float))
    so = QC.skor_once(df)
    ayni = [(so[df.index[i + 1]] == sk[i]) or (math.isnan(so[df.index[i + 1]]) and math.isnan(sk[i]))
            for i in range(len(df) - 1)]
    assert all(ayni)                                  # TÜM barlar: skor ardışık barlarda sık aynıdır (tek nokta dejenere)
    assert len({round(x, 6) for x in sk[252:] if not math.isnan(x)}) > 1, "kurgu dejenere: skor hiç değişmiyor"
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("close")] *= 3.0                    # gelecek değişir, geçmiş değişmez
    so2 = QC.skor_once(df2)
    assert all(so[t] == so2[t] or (math.isnan(so[t]) and math.isnan(so2[t])) for t in list(so)[:-1])


def test_q0_kolu_motoru_degistirmez():
    from nihai.execution.backtest import backtest_symbol
    df = _gunluk(_yuruyus())
    ham = backtest_symbol("SYN", "1d", df=df, strategy_mode="quant_trend", warmup_bars=260, disable_kill_switch=True)
    r, _ = _motor(df, "q0")
    assert ham.equity_curve == r.equity_curve and ham.trades_count == r.trades_count > 0


def test_q1_sinyal_cikisi_skor_dondugu_ilk_acilista():
    c = np.r_[_yuruyus(450, mu=0.002, sd=0.008, seed=1), _yuruyus(350, mu=-0.004, sd=0.008, seed=2) * 0]
    c[450:] = c[449] * np.exp(np.cumsum(np.random.default_rng(3).normal(-0.004, 0.008, 350)))
    df = _gunluk(c)
    sk, _ = Q.seriler(df["close"].to_numpy(float))
    _, isl = _motor(df, "q1")
    sinyal = [x for x in isl if x["neden"] == "QUANT_SINYAL"]
    assert sinyal, "kurgu dejenere: sinyal çıkışı yok"
    for x in sinyal:                                   # karar b−1 kapanışında, infaz b açılışında
        assert sk[x["b"] - 1] <= 0 < np.nanmin(sk[x["a"] - 1: x["b"] - 1])
    assert {x["neden"] for x in isl} <= {"QUANT_SINYAL", "STOP", "GAP_STOP", None}


def test_q1_izleyen_stopu_kapatir_q0_acik_birakir():
    for kol, beklenen in (("q0", 125.0 - 6.0), ("q1", 94.0)):
        _, pm = _pmgr()
        pos = _ac(pm)
        with QC.cekirdek(kol, {}):
            pm.update_all_with_bars({"AAA": (101.0, 125.0, 100.0, 124.0)}, now=datetime(2024, 1, 3))
        assert pos.stop == pytest.approx(beklenen)


def test_q1n_stopsuz_kolda_stop_kapatmaz_ve_stop_alani_korunur():
    _, pm = _pmgr()
    pos = _ac(pm)
    with QC.cekirdek("q1n", {}):
        kapanan = pm.update_all_with_bars({"AAA": (95.0, 95.5, 80.0, 81.0)}, now=datetime(2024, 1, 3))
    assert kapanan == [] and pos.stop == pytest.approx(94.0)
    _, pm2 = _pmgr()
    _ac(pm2)
    with QC.cekirdek("q1", {}):                        # sabit stop q1'de yine çalışır
        kapanan = pm2.update_all_with_bars({"AAA": (95.0, 95.5, 80.0, 81.0)}, now=datetime(2024, 1, 3))
    assert len(kapanan) == 1 and kapanan[0].exit_trigger == "STOP"


def test_q0s_riski_skor_buyuklugu_ile_olcekler():
    from nihai.risk.risk_manager import PortfolioState, RiskManager
    def adet():
        rm = RiskManager(PortfolioState(equity=1_000_000, ath_equity=1_000_000, cash_balance=1_000_000))
        return rm.size_stock(entry=100.0, stop=95.0, symbol="AAA", side="LONG").sizing["shares"]
    tam = adet()
    with QC.cekirdek("q0s", {}):
        QC._DURUM["son_skor"] = 1.0 / 3.0
        ucte = adet()
        QC._DURUM["son_skor"] = None
        yok = adet()
    assert ucte == pytest.approx(tam / 3.0, abs=1) and yok == tam


def test_yamalar_geri_yuklenir_zincirlenmez_ve_kisa_bayragi_yalniz_fxte():
    import nihai.execution.backtest as B
    from nihai.risk.position_manager import PositionManager as PM
    from nihai.risk.risk_manager import RiskManager as RM
    once = (PM.check_exit, PM._apply_trailing, RM.size_stock, B.generate_signal, CONFIG.strategy.quant_allow_short)
    for kol in ("q2", "q0k", "q1n", "q1k"):          # SON kol kısa bayrağını AÇAR → geri yükleme sınanır
        with QC.cekirdek(kol, {}):
            assert CONFIG.strategy.quant_allow_short is (kol in QC.FX_KOLLAR)
    assert (PM.check_exit, PM._apply_trailing, RM.size_stock, B.generate_signal,
            CONFIG.strategy.quant_allow_short) == once


# ---------------------------------------------------------------------------------------------------
# Portföy planı + parite
# ---------------------------------------------------------------------------------------------------
class _Poz:
    def __init__(self, sektor, nominal):
        self.sector, self.signed_notional = sektor, nominal


def test_plan_siralama_sektor_tavani_ve_carpan():
    ad = [("D", 1 / 3, 0.02, "TECH"), ("A", 1.0, 0.01, "TECH"), ("B", 1.0, 0.02, "TECH"),
          ("C", 1 / 3, 0.01, "FIN"), ("E", 1.0, 0.02, "TECH")]
    sira = {s: k for k, s in enumerate("EDCBA")}
    sec, m = QC.plan_kur(ad, [_Poz("TECH", 0.0)], sira, 1.0, 0.015, 2.0, 3, 50)
    assert sec == ["E", "B", "C"]                       # skor azalan, eşitlikte liste sırası; TECH 1 açık + 2 yeni
    talep = 0.015 / 0.06 + 0.015 / 0.06 + 0.015 / 0.03
    assert m == pytest.approx(min(1.0, 2.0 / talep))
    _, m_bol = QC.plan_kur(ad[:1], [], sira, 1.0, 0.015, 2.0, 3, 50)
    assert m_bol == 1.0                                 # talep kapasitenin altında → tam risk
    _, m_dolu = QC.plan_kur(ad, [_Poz("X", 2.5e6)], sira, 1e6, 0.015, 2.0, 3, 50)
    assert m_dolu == 0.0                                # net yön tavanı dolu
    sec_s, m_s = QC.plan_kur(ad, [], sira, 1.0, 0.015, 2.0, 3, 50, skor_boyutu=True)
    assert m_s >= m                                     # skor boyutu talebi küçültür


def test_portfoy_kopyasi_backtest_portfolio_ile_birebir():
    from nihai.execution.backtest import backtest_portfolio
    data = {s: _gunluk(_yuruyus(700, seed=k)) for k, s in enumerate(("AAA", "BBB", "CCC"))}
    sira = list(data)
    r = backtest_portfolio(sira, "1d", data=dict(data), strategy_mode="quant_trend", disable_kill_switch=True,
                           warmup_bars=60)
    k = QC.portfoy_kos(data, sira, bas=None, plan=False)
    assert np.asarray(k["eq"]) == pytest.approx(np.asarray(r.equity_curve[1:1 + len(k["eq"])]), abs=0.0)
    assert k["islem"] == r.trades_count > 0


def test_plan_acikken_yalniz_secilen_sembole_girilir(monkeypatch):
    data = {s: _gunluk(_yuruyus(700, seed=k)) for k, s in enumerate(("AAA", "BBB", "CCC"))}
    monkeypatch.setattr(QC, "plan_kur", lambda ad, *a, **k: ([s for s, *_ in ad if s == "BBB"], 1.0))
    k = QC.portfoy_kos(data, list(data), bas=None, plan=True)
    assert k["farkli_sembol"] == 1 and k["m"] and k["m"]["ort"] == 1.0


# ---------------------------------------------------------------------------------------------------
# Kapılar ve hüküm
# ---------------------------------------------------------------------------------------------------
def _b(z=2.0, olumlu=10, n=14, izl=0.1):
    return {"z": z, "olumlu_piyasa": olumlu, "n_piyasa": n, "izleme_ds": izl}


def test_g_kapisi_uc_sart_ve_uc_pencere():
    assert QC.g_kapisi(_b())
    assert not QC.g_kapisi(_b(z=1.6))
    assert not QC.g_kapisi(_b(olumlu=9))                # ⌈⅔·14⌉ = 10
    assert QC.g_kapisi(_b(olumlu=10, n=15))             # ⌈⅔·15⌉ = 10
    assert not QC.g_kapisi(_b(izl=0.0)) and not QC.g_kapisi(_b(izl=None))
    iyi = {"parite": {"B": 0.0, "C": 0.0}, "birlesik": {h: {p: _b() for p in QC.PENCERELER} for h in QC.HIPOTEZLER}}
    assert QC.hukum(iyi)["H1"]["gecti"] and QC.hukum(iyi)["aday"] == "q2"
    kotu = json.loads(json.dumps(iyi))
    kotu["birlesik"]["H1"]["C"]["z"] = 0.1
    assert not QC.hukum(kotu)["H1"]["gecti"] and QC.hukum(kotu)["aday"] == "q0s"
    par = json.loads(json.dumps(iyi))
    par["parite"]["B"] = 1e-9
    assert QC.hukum(par)["aday"] is None                # parite tutmazsa HÜKÜM YOK


def test_z_dogrulamasi_kalirsa_plan_baglanmaz(tmp_path, monkeypatch):
    monkeypatch.setattr(QC, "SONUC_DIR", tmp_path)
    idx = pd.bdate_range("2019-01-01", periods=300)
    rng = np.random.default_rng(0)

    def yaz(pen, kol, duzen, tohum, mu, sd=0.01):
        r = pd.Series(rng.normal(mu, sd, idx.size), index=idx)
        with open(tmp_path / f"pf_{pen}_{kol}_{duzen}_{tohum}.pkl", "wb") as f:
            pickle.dump({"r": r, "islem": 10, "maruziyet": 1.0, "ort_acik": 5.0, "farkli_sembol": 5, "hhi": 0.2,
                         "sembol": ["A"], "m": None}, f)
    for pen in QC.PORTFOY_PENCERE:
        for kol in ("q0", "q1"):
            yaz(pen, kol, "liste", 0, 0.0)
            yaz(pen, kol, "plan", 0, 0.004 if pen != "Z" else -0.004)
            for t in QC.PERM_TOHUM:
                yaz(pen, kol, "perm", t, 0.0)
    b2 = QC.bolum2()
    assert b2["kapi_p_gecti"] and not b2["z1"] and not b2["plan_baglanir"]


# ---------------------------------------------------------------------------------------------------
# Pencere ve hazırlık kuralları
# ---------------------------------------------------------------------------------------------------
def test_hazirla_arsiv_pencere_siniri_ve_kusur(monkeypatch):
    from research import piyasalar as P
    idx = pd.bdate_range("1995-01-02", "2012-12-31")
    c = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, idx.size)))
    ham = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6}, index=idx)
    monkeypatch.setattr(P, "yukle", lambda s, *a, **k: ham)
    df, neden = QC.hazirla_arsiv("XYZ", "A")
    assert neden is None and df.index[0] >= pd.Timestamp(QC.ARSIV_BAS) and df.index[-1] < pd.Timestamp("2009-01-01")
    kirik = ham.copy()
    kirik.loc[kirik.index[2500]:, ["open", "high", "low", "close"]] *= 2.0     # tek günlük +%100 = kusur
    monkeypatch.setattr(P, "yukle", lambda s, *a, **k: kirik)
    df2, neden2 = QC.hazirla_arsiv("XYZ", "A")
    assert df2 is None and "log getiri" in neden2
    dfz, _ = QC.hazirla_arsiv("XYZ", "Z")
    assert dfz is None or (dfz.index[0] >= pd.Timestamp("1987-01-01") and dfz.index[-1] < pd.Timestamp("2001-01-01"))


def test_pencere_verisi_or_ve_test_kurallari(monkeypatch):
    from research import coklu_piyasa as C
    idx = pd.bdate_range("2008-01-02", "2026-09-01")
    df = _gunluk(_yuruyus(idx.size), bas="2008-01-02")
    monkeypatch.setattr(C, "hazirla", lambda s: df)
    d, t0 = QC.pencere_verisi("S", "B")
    assert d.index[-1] < pd.Timestamp("2018-01-01") and t0 == max(int((d.index < pd.Timestamp("2009-01-01")).sum()), 60)
    d, t0 = QC.pencere_verisi("S", "C")
    assert len(d) == len(df) and t0 == int((df.index < pd.Timestamp("2018-01-01")).sum())
    kisa = df[df.index >= "2016-06-01"]
    monkeypatch.setattr(C, "hazirla", lambda s: kisa)
    assert QC.pencere_verisi("S", "B") is None          # B'de < 500 değerlendirme barı
    assert QC.pencere_verisi("S", "C") is not None      # C'de şart yok (Tur-97 TEST kuralı)


# ---------------------------------------------------------------------------------------------------
# Üretim: backtest seans sayaçları (BULGU-101-5)
# ---------------------------------------------------------------------------------------------------
def test_backtest_gunluk_sayaclar_her_seans_sifirlanir(monkeypatch):
    from nihai.execution.backtest import backtest_symbol
    from nihai.risk.risk_manager import PortfolioState
    df = _gunluk(_yuruyus(900, mu=0.0002, sd=0.015, seed=11))

    def kos(soguma, gunluk):
        once = (CONFIG.risk.consecutive_stops_cooldown, CONFIG.risk.daily_max_loss_pct)
        object.__setattr__(CONFIG.risk, "consecutive_stops_cooldown", soguma)   # dondurulmuş dataclass
        object.__setattr__(CONFIG.risk, "daily_max_loss_pct", gunluk)
        try:
            return backtest_symbol("SYN", "1d", df=df, strategy_mode="quant_trend", warmup_bars=260,
                                   disable_kill_switch=True)
        finally:
            object.__setattr__(CONFIG.risk, "consecutive_stops_cooldown", once[0])
            object.__setattr__(CONFIG.risk, "daily_max_loss_pct", once[1])
    serbest = kos(0, 0.0)
    kayip = np.asarray(serbest.trade_pnls) < 0
    ardisik = max((len(s) for s in "".join("1" if k else "0" for k in kayip).split("0")), default=0)
    assert ardisik >= 3, "kurgu dejenere: 3 ardışık zarar yok"
    assert kos(3, 0.10).trades_count == serbest.trades_count           # .env.example varsayılanları
    monkeypatch.setattr(PortfolioState, "roll_session_if_new_day", lambda self, d: False)   # kontrol: eski kusur
    assert kos(3, 0.10).trades_count < serbest.trades_count


def test_iki_hisse_backtestinde_seans_baslangici_gecmiste():
    from nihai.execution import backtest as B
    for f in (B.backtest_symbol, B.backtest_portfolio):
        src = inspect.getsource(f)
        assert "portfolio.last_session_date = date.min" in src and "roll_session_if_new_day(" in src


# ---------------------------------------------------------------------------------------------------
# Üretim: canlı quant sadakati (BULGU-101-1 / 101-4)
# ---------------------------------------------------------------------------------------------------
class _Store:
    kill_switch_active = False

    def __init__(self):
        self.symbol_cooldowns: dict = {}
        self.pnl: list = []

    def record_trade_pnl(self, p):
        self.pnl.append(p)

    def register_close_for_wash_sale(self, *a):
        pass

    def add_unsettled_proceeds(self, *a):
        pass

    def set_cooldown(self, sym, now):
        self.symbol_cooldowns[sym] = now

    def cooldown_active(self, *a):
        return False


def test_kapanis_defteri_yalniz_quant_cikisini_damgalar():
    import nihai.execution.main as M
    for strateji, damga in (("quant_trend V1", True), ("event_breakout V1", False)):
        _, pm = _pmgr()
        pos = _ac(pm, strateji)
        st = _Store()
        ce = pm.force_close(pos, 97.0, "STOP", datetime(2024, 1, 5, 11, 0))
        M._apply_close_bookkeeping(st, ce)
        assert (st.symbol_cooldowns.get("AAA") == datetime(2024, 1, 5, 11, 0)) is damga


def test_quant_odak_disi_gunde_bir_satir_ve_ayni_gun_yeniden_giris_yok(monkeypatch):
    import nihai.execution.main as M
    loglar, cekilen = [], []
    monkeypatch.setattr(M, "log_block", lambda tag, msg: loglar.append(tag))
    monkeypatch.setattr(M, "can_enter_new_position", lambda: (True, ""))
    monkeypatch.setattr(M, "event_guard", lambda sym: (True, "sahte"))
    monkeypatch.setattr(M, "_quant_odak", lambda: frozenset({"AAA"}))
    monkeypatch.setattr(M, "now_et_naive", lambda: datetime(2026, 9, 28, 10, 0))
    monkeypatch.setattr(M, "_QUANT_ODAK_LOG", {"gun": None, "s": set()})

    class Prov:
        def get_bars(self, *a, **k):
            cekilen.append(a)
            return pd.DataFrame()
    _, pm = _pmgr()
    st = _Store()
    for _ in range(3):
        M.scan_symbol("ZZZ", Prov(), None, st, None, pm, strategy_mode="quant_trend")
    assert loglar == ["QUANT_ODAK_DISI"] and cekilen == []           # geniş katman: günde tek satır, bar çekilmez
    st.symbol_cooldowns["AAA"] = datetime(2026, 9, 28, 9, 50)           # bugün stopla çıktı
    M.scan_symbol("AAA", Prov(), None, st, None, pm, strategy_mode="quant_trend")
    assert loglar[-1] == "QUANT_AYNI_GUN" and cekilen == []
    st.symbol_cooldowns["AAA"] = datetime(2026, 9, 25, 15, 0)           # önceki seans → geçer
    M.scan_symbol("AAA", Prov(), None, st, None, pm, strategy_mode="quant_trend")
    assert loglar[-1] == "EVENT_GUARD"
    M.scan_symbol("ZZZ", Prov(), None, st, None, pm, strategy_mode="event_breakout")
    assert loglar[-1] == "EVENT_GUARD"                                  # TA kipinde odak kuralı yok


def test_quant_odak_60_sn_onbellekli(monkeypatch):
    import nihai.execution.main as M
    sayac = {"n": 0, "t": 1000.0}

    def partition(n):
        sayac["n"] += 1
        return {"focused": ["AAA"], "broad": ["ZZZ"]}
    monkeypatch.setattr(M.watchlist, "partition", partition)
    monkeypatch.setattr(M.time_mod, "monotonic", lambda: sayac["t"])
    monkeypatch.setattr(M, "_QUANT_ODAK", {"t": float("-inf"), "k": frozenset()})
    assert M._quant_odak() == {"AAA"} and M._quant_odak() == {"AAA"} and sayac["n"] == 1
    sayac["t"] += 61
    M._quant_odak()
    assert sayac["n"] == 2


def test_odak_ve_ayni_gun_kurali_bar_cekiminden_once():
    import nihai.execution.main as M
    src = inspect.getsource(M.scan_symbol)
    assert src.index("_quant_odak()") < src.index("provider.get_bars") and "QUANT_AYNI_GUN" in src
    assert src.index("QUANT_AYNI_GUN") < src.index("provider.get_bars")


# ---------------------------------------------------------------------------------------------------
# Karne — ön-kayıtlı hüküm
# ---------------------------------------------------------------------------------------------------
def test_karne_on_kayitli_hukmu_tasir():
    k = json.loads((ROOT / "data" / "quant_cekirdek_karnesi.json").read_text(encoding="utf-8"))
    hk, b2 = k["hukum"], k["bolum2"]
    assert hk["parite_gecti"] and hk["aday"] is None
    assert not hk["H1"]["gecti"] and not hk["H2"]["gecti"]
    assert b2["kapi_p_gecti"] and not b2["z1"] and b2["z2"] and not b2["plan_baglanir"]
    assert set(b2) >= set(QC.PORTFOY_PENCERE) and "parmak_izi" in k
    assert date.fromisoformat(k["uretim"][:10]) >= date(2026, 9, 27)
