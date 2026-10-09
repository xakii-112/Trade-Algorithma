"""Tur-99 — ML'in KARAR mekanizması: VY boyut kararı laboratuvarı, birleşik kapı, AI köprüsünün kapsamı.

Kilitlenen: işlem defterinden kurulan SABİT replikanın motorun MTM getirisini birebir vermesi (laboratuvarın
ön-koşulu), VY'nin tek değişkenli ve nedensel olması (σ̂[t−1], bant, maliyet, tavan; giriş/çıkış aynı),
korelasyon düzeltmeli birleşik z'nin ortak piyasayı çift saymaması, MDE formülü, ön-kayıtlı kapı, AI karar
köprüsünün quant adayına 1m kanıtıyla görüş YAZMAMASI (BULGU-99-1/2) ve karnelerin girdi parmak izi taşıması.
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import coklu_piyasa as C
from research import quant_karar as K
from research import quant_kayma as QK

ROOT = Path(__file__).resolve().parents[1]


def _ohlc(close, bas="2015-01-01"):
    c = np.asarray(close, float)
    o = np.r_[c[0], c[:-1]] * 1.001
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99, "close": c},
                        index=pd.bdate_range(bas, periods=c.size))


def _islem(a, b, u=10.0, F=100.0, X=110.0, E=1000.0, neden="STOP"):
    return {"a": a, "b": b, "u": u, "F": F, "X": X, "E": E, "neden": neden, "pnl": u * (X - F)}


# ---------------------------------------------------------------------------------------------------
# Laboratuvar: SABİT replika = motor
# ---------------------------------------------------------------------------------------------------
def test_sabit_replika_motorun_mtm_getirisini_birebir_verir():
    """Gerçek motor (`backtest_symbol` quant_trend) + yakalanan defter → replika günlük getirisi ≤ 1e-8."""
    rng = np.random.default_rng(7)
    lr = rng.normal(0.0004, 0.012, 900) + 0.002 * np.sin(np.arange(900) / 60)
    df = _ohlc(100 * np.exp(np.cumsum(lr)))
    t0 = 300
    with K.defter() as kay:
        r = C._algoritma_kos("SYN", df, t0, "quant_trend")
    isl = K.islemler(kay, df.index)
    assert len(isl) >= 5
    sab = K.simule(df["open"].to_numpy(), df["close"].to_numpy(), t0, isl, 1.0)
    rep = sab["E"][1:] / sab["E"][:-1] - 1.0
    np.testing.assert_allclose(rep, np.asarray(r["r"], float), atol=1e-6)   # r float32 saklanır
    assert all(x["a"] > t0 for x in isl)
    assert all(x["b"] is None or x["b"] > x["a"] for x in isl)


def test_defter_orijinalleri_geri_koyar_ve_zincirlenmez():
    from nihai.risk.position_manager import PositionManager as PM
    orj = (PM.open_from_signal, PM.update_all_with_bars, PM.force_close)
    for _ in range(3):                                           # BULGU-96-5: havuz işçisi ardışık görevler
        with K.defter():
            assert PM.open_from_signal is not orj[0]
        assert (PM.open_from_signal, PM.update_all_with_bars, PM.force_close) == orj


def test_giris_gunu_isaret_dolumda_kalir_ve_cikis_pnli_toplamdir():
    O = np.full(10, 100.0)
    Cl = np.array([100, 100, 105, 108, 104, 100, 100, 100, 100, 100], float)
    isl = [_islem(a=2, b=5, u=10, F=101.0, X=103.0, E=1000.0)]
    E = K.simule(O, Cl, 0, isl, 1000.0)["E"]
    assert E[2] == 1000.0                                        # giriş barında MTM = dolum (motor konvansiyonu)
    assert E[3] == pytest.approx(1000 + 10 * (108 - 101))
    assert E[5] == pytest.approx(1000 + 10 * (103 - 101))        # toplam = u·(X − F)
    assert E[-1] == E[5]


def test_bt_end_cikisi_acik_pozisyon_sayilir():
    class Ce:
        def __init__(self, pos, trig, ts):
            self.pos, self.exit_trigger, self.exit_time, self.exit_price, self.realized_pnl = pos, trig, ts, 1.0, 0.0

    class Pos:
        quantity, entry_price = 5, 100.0
    idx = pd.bdate_range("2020-01-01", periods=10)
    p = Pos()
    kay = {"giris": [{"pos": p, "now": idx[2], "E": 1000.0}], "cikis": {id(p): Ce(p, "BT_END", idx[-1])}}
    x = K.islemler(kay, idx)[0]
    assert x["a"] == 3 and x["b"] is None


# ---------------------------------------------------------------------------------------------------
# VY — tek değişken, nedensel
# ---------------------------------------------------------------------------------------------------
def _yol(n=60, seed=1):
    rng = np.random.default_rng(seed)
    Cl = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    O = np.r_[Cl[0], Cl[:-1]] * (1 + rng.normal(0, 0.002, n))
    return O, Cl


def test_vy_sabit_sigmada_sabitle_aynidir():
    O, Cl = _yol()
    isl = [_islem(a=5, b=30, u=10, F=O[5], X=Cl[29], E=1000.0), _islem(a=35, b=55, u=8, F=O[35], X=Cl[54], E=1000.0)]
    sab = K.simule(O, Cl, 0, isl, 1000.0)
    vy = K.simule(O, Cl, 0, isl, 1000.0, np.full(60, 0.01), bant=0.25, maliyet=0.001)
    np.testing.assert_allclose(vy["E"], sab["E"], rtol=0, atol=1e-9)
    assert vy["devir_gunluk"] == 0.0


def test_vy_sigma_ikiye_katlaninca_ertesi_acilista_adet_yariya_iner_ve_maliyet_yazilir():
    O = np.full(20, 100.0)
    Cl = np.full(20, 100.0)
    sig = np.full(20, 0.01)
    sig[7:] = 0.02                                               # t=7 kapanışında σ̂ ikiye katlanır
    isl = [_islem(a=3, b=15, u=10, F=100.0, X=100.0, E=1000.0)]
    E = K.simule(O, Cl, 0, isl, 1000.0, sig, bant=0.25, maliyet=0.001)["E"]
    assert E[7] == pytest.approx(1000.0)                         # t=7 açılışı σ̂[6]'yı görür → işlem yok
    assert E[8] == pytest.approx(1000.0 - 0.001 * 5 * 100.0)     # t=8 açılışı: 10 → 5 adet, maliyet |Δu|·O·c


def test_vy_kararinin_bugunku_sigmayi_gormedigi():
    O, Cl = _yol(80, 3)
    rng = np.random.default_rng(4)
    sig = 0.01 * np.exp(rng.normal(0, 0.4, 80))
    isl = [_islem(a=5, b=70, u=10, F=O[5], X=Cl[69], E=1000.0)]
    E1 = K.simule(O, Cl, 0, isl, 1000.0, sig)["E"]
    for k in (20, 40, 60):
        s2 = sig.copy()
        s2[k] *= 5.0
        E2 = K.simule(O, Cl, 0, isl, 1000.0, s2)["E"]
        np.testing.assert_array_equal(E1[: k + 1], E2[: k + 1])  # σ̂[k] ancak k+1 açılışında karara girer


def test_vy_bant_ici_degisimde_islem_yok_ve_tavan_uygulanir():
    O = np.full(20, 100.0)
    Cl = np.full(20, 100.0)
    isl = [_islem(a=3, b=15, u=10, F=100.0, X=100.0, E=1000.0)]
    assert K.simule(O, Cl, 0, isl, 1000.0, np.r_[np.full(8, 0.01), np.full(12, 0.009)])["devir_gunluk"] == 0.0
    sig = np.r_[np.full(8, 0.01), np.full(12, 0.001)]             # σ̂ onda bire → hedef 100 adet
    vy = K.simule(O, Cl, 0, isl, 1000.0, sig, tavan=2.0)
    # maruziyet = tüm günlerin ortalaması (düz günler 0): 6 gün ×1 + 6 gün ×2 → 18/19; tavansız 6×1 + 6×10 → 66/19
    assert vy["maruziyet"] == pytest.approx(18 / 19, rel=0.01)
    assert K.simule(O, Cl, 0, isl, 1000.0, sig)["maruziyet"] == pytest.approx(66 / 19, rel=0.01)
    assert K.KARAR["f_tavan"] == 2.0


def test_f_tavan_motorun_alim_gucu_tavanidir():
    from nihai.risk.risk_manager import PortfolioState, RiskManager
    p = PortfolioState(equity=1_000_000, ath_equity=1_000_000, cash_balance=1_000_000)
    rc = RiskManager(p).size_stock(entry=100.0, stop=99.99, sector="X", symbol="S", side="LONG")
    assert rc.approved and rc.sizing["notional"] / 1_000_000 == pytest.approx(K.KARAR["f_tavan"], rel=1e-3)


# ---------------------------------------------------------------------------------------------------
# Birleşik kapı + MDE
# ---------------------------------------------------------------------------------------------------
def test_birlesik_z_bagimsizda_stouffer_ozdes_piyasada_cift_saymaz():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=1000)
    f = {k: pd.Series(rng.normal(0, 1, idx.size), index=idx) for k in "abcd"}
    b = K.birlesik_z({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}, f)
    assert b["z"] == pytest.approx(4 / math.sqrt(4), abs=0.1)
    ikiz = {"a": f["a"], "a_kopya": f["a"].copy()}
    assert K.birlesik_z({"a": 1.5, "a_kopya": 1.5}, ikiz)["z"] == pytest.approx(1.5)   # 2·1,5/√4
    assert K.birlesik_z({"a": 1.0}, f)["z"] is None
    assert K.birlesik_z({"a": None, "b": 2.0}, f)["z"] is None


def test_mde_formulu():
    assert K.mde({"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1}, 4.0) == pytest.approx((1.645 + 0.8416) * 0.1 / 2, abs=1e-4)
    assert K.mde({"a": 0.1}, 4.0) > K.mde({"a": 0.1}, 1.0)        # korelasyon etkin sayıyı düşürür
    assert K.mde({}, 4.0) is None


def test_sharpe_fark_isaret_ve_standart_hata():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2018-01-01", periods=1500)
    b = pd.Series(rng.normal(0.0003, 0.01, idx.size), index=idx)
    a = b + pd.Series(rng.normal(0.0002, 0.002, idx.size), index=idx)
    m = K.sharpe_fark(a, b, 252)
    assert m["ds"] > 0 and m["z"] > 0 and m["se"] > 0
    assert K.sharpe_fark(b, a, 252)["z"] == pytest.approx(-m["z"], abs=1e-3)
    assert K.sharpe_fark(a.iloc[:30], b.iloc[:30], 252)["z"] is None


def _karne(z_a, z_b, olumlu=12, izl=0.02, parite=True):
    bl = {pen: {"z": z, "olumlu_piyasa": olumlu, "n_piyasa": 15, "izleme_ds": izl}
          for pen, z in zip(QK.SIGMA_PENCERE, (z_a, z_b), strict=True)}
    return {"birlesik": {h: bl for h in K.HIPOTEZLER}, "parite": {"gecti": parite}}


def test_hukum_on_kayitli_kapi():
    assert K.hukum(_karne(2.0, 2.0))["H3a"]["gecti"] is True
    assert K.hukum(_karne(2.0, 2.0))["aday"] == "ml21_vy"
    assert K.hukum(_karne(2.0, 1.6))["H3a"]["gecti"] is False     # iki pencerede de
    assert K.hukum(_karne(2.0, 2.0, olumlu=9))["H3a"]["gecti"] is False
    assert K.hukum(_karne(2.0, 2.0, izl=-0.01))["H3a"]["gecti"] is False
    assert K.hukum(_karne(2.0, 2.0, parite=False))["H3a"]["gecti"] is False   # laboratuvar ön-koşulu
    k = _karne(2.0, 2.0)
    k["birlesik"]["H3a"] = _karne(1.0, 1.0)["birlesik"]["H3a"]
    assert K.hukum(k)["aday"] is None                              # H3b tek başına aday üretmez


def test_on_kayit_belgesi():
    doc = K.__doc__
    for ifade in ("ÖN-KAYIT", "H3a", "H3b", "TEŞHİS (SONRADAN", "MDE (koşudan ÖNCE", "BİRLEŞİK z", "≤ 1e-8",
                  "F_TAVAN = 2,0", "BANT (0,25)"):
        assert ifade in doc, ifade
    assert K.KARAR["bant"] == 0.25 and K.KARAR["birlesik_z"] == 1.645 and K.KARAR["min_piyasa"] == 10
    assert K.HIPOTEZLER["H3a"] == ("ewma_vy", "ewma_sabit", True)
    assert K.HIPOTEZLER["H3b"] == ("ml21_vy", "ewma_vy", True)
    assert K.HIPOTEZLER["teshis_tur98_h1"][2] is False             # sonradan kol hüküm veremez


def test_tur99_karnesi_aday_yok_ve_canliya_baglanmadi():
    karne = json.loads((ROOT / "data" / "quant_karar_karnesi.json").read_text(encoding="utf-8"))
    assert karne["parite"]["gecti"] is True
    assert karne["hukum"]["aday"] is None
    assert karne["hukum"]["H3a"]["gecti"] is False and karne["hukum"]["H3b"]["gecti"] is False
    assert karne["parmak_izi"]["n_sembol"] > 100
    for p in (ROOT / "nihai").rglob("*.py"):
        assert "quant_karar" not in p.read_text(encoding="utf-8"), p


# ---------------------------------------------------------------------------------------------------
# BULGU-99-1/2 — AI karar köprüsü quant adayına kapsam dışı
# ---------------------------------------------------------------------------------------------------
def _sig(action="LONG", strategy="QuantTrend"):
    from nihai.signals.strategy import Signal, SignalAction
    return Signal(action=SignalAction(action), symbol="AAA", price=100.0, stop=94.0, target=118.0,
                  reasons=["test"], quality_score=3.0, rr_ratio=3.0, metadata={"strategy": strategy})


@pytest.mark.parametrize("kip", ["advise", "enforce"])
def test_ai_koprusu_quant_adayina_1m_kanitiyla_gorus_yazmaz(monkeypatch, kip):
    from nihai.execution import ai_advisor as A

    def _patlat(*a, **k):
        raise AssertionError("quant adayı karar motoruna sorulmamalı")
    monkeypatch.setattr("nihai.ai.decision.decide", _patlat)
    monkeypatch.setenv("AI_LIVE_MODE", kip)
    df = _ohlc(np.linspace(100, 120, 300))
    once = A.status()["n_kapsam_disi"]
    for sig, strat in ((_sig(), "quant_trend"), (_sig(), None), (_sig(strategy="EventBreakout"), "quant_trend")):
        adv = A.advise("AAA", df, sig, strategy=strat)
        assert adv.action == A.KAPSAM_DISI and adv.veto is False and adv.size_mult == 1.0
        assert A.KAPSAM_DISI in adv.reason and "5 bar" in adv.reason
    assert A.status()["n_kapsam_disi"] == once + 3
    monkeypatch.setenv("AI_LIVE_MODE", "off")
    assert A.advise("AAA", df, _sig(), strategy="quant_trend") is None


def test_ai_koprusu_quant_disi_adayi_motora_sormaya_devam_eder(monkeypatch):
    from nihai.ai.decision import ShadowDecision
    from nihai.execution import ai_advisor as A
    cagri = []

    def _fake(symbol, strategy, features, **kw):
        cagri.append(strategy)
        return ShadowDecision(ts="2026-09-26T00:00:00+00:00", symbol=symbol, strategy=strategy, action="ABSTAIN",
                              conviction=0.0, prior_mean=0.5, posterior=0.47, evidence=["Karar: ABSTAIN."],
                              kelly_suggest=0.0, regime=None, lesson_warnings=[], mode="off", utility=None)
    monkeypatch.setattr("nihai.ai.decision.decide", _fake)
    monkeypatch.setenv("AI_LIVE_MODE", "enforce")
    adv = A.advise("AAA", _ohlc(np.linspace(100, 120, 300)), _sig(strategy="EventBreakout"),
                   strategy="event_breakout")
    assert cagri == ["event_breakout"] and adv.veto is True        # eski davranış (ABSTAIN → veto) korunur


def test_pano_karti_canli_kopruyle_ayni_kapsam_kuralini_kullanir():
    kaynak = (ROOT / "nihai" / "web" / "web_dashboard.py").read_text(encoding="utf-8")
    # Tur-106 (BULGU-106-27): strateji canlı taramayla AYNI kaynaktan (`runtime_strategy.apply`), sabit config adından değil.
    assert "_aia.kapsam_disi(sig, strateji)" in kaynak and "strateji = _strateji_sec()" in kaynak
    from nihai.execution import ai_advisor as A
    assert inspect.getsource(A.advise).count("kapsam_disi(sig, strategy)") == 1


# ---------------------------------------------------------------------------------------------------
# Tur-98 §9 #4 — karnelerin girdi parmak izi
# ---------------------------------------------------------------------------------------------------
def test_girdi_parmak_izi_girdiye_duyarli_ve_birimden_bagimsiz(monkeypatch):
    df = _ohlc(np.linspace(100, 130, 50))
    df = df.set_axis(df.index.as_unit("ns"))                     # pandas 3 bdate_range'i zaten "us" üretir
    tablo = {"A": df, "B": df * 1.0}
    monkeypatch.setattr(C, "hazirla", lambda s: tablo[s])
    p1 = C.girdi_parmak_izi(["B", "A"])
    assert p1 == C.girdi_parmak_izi(["A", "B"]) and p1["n_sembol"] == 2
    assert p1["sembol"]["A"]["son"] == str(df.index[-1].date())
    tablo["A"] = df.set_axis(df.index.as_unit("us"))
    assert C.girdi_parmak_izi(["A", "B"])["ozet"] == p1["ozet"]   # pandas 3: indeks birimi özeti değiştirmez
    d2 = df.copy()
    d2.iloc[10, d2.columns.get_loc("close")] += 1e-9
    tablo["A"] = d2
    p2 = C.girdi_parmak_izi(["A", "B"])
    assert p2["ozet"] != p1["ozet"] and p2["sembol"]["B"] == p1["sembol"]["B"]


def test_karne_yazicilari_parmak_izi_tasir():
    for mod in ("quant_kayma", "quant_ml", "quant_ml_motor", "coklu_piyasa", "quant_karar"):
        assert 'karne["parmak_izi"]' in (ROOT / "research" / f"{mod}.py").read_text(encoding="utf-8"), mod
