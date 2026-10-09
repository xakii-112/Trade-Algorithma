"""Tur-78 — bağımsız denetimin (Codex R2/R3 + Claude doğrulaması) kilitleri.

BULGU-78-1: backtest döngüleri bar i+1'de kapanan pozisyonu gördükten sonra bar i kapanışında
karar verip AYNI barın açılışında yeniden giriyordu (çıkış bar İÇİNDE, giriş AÇILIŞTA → tek
pozisyon kuralı zamanda deliniyor, kapı gelecekten besleniyor). Her testte eski davranışın
`REENTRY_POLICY=same_bar` ile hâlâ üretildiği de sınanır: koruma ateşlenmiyorsa test boştur.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nihai.config import CONFIG
from nihai.execution import backtest as bt
from nihai.signals.strategy import Signal, SignalAction


def _df(next_low: float = 94.0) -> pd.DataFrame:
    df = pd.DataFrame(dict(open=[100.] * 25, high=[101.] * 25, low=[99.] * 25,
                           close=[100.] * 25, volume=[10000.] * 25),
                      index=pd.date_range("2024-01-02 09:30", periods=25, freq="5min"))
    df.iloc[22, df.columns.get_loc("low")] = next_low      # yalnız bar 22 stop 95'i deler
    return df


def _run(runner, policy: str):
    opens, closes = [], []
    orig_open, orig_close = bt.PositionManager.open_from_signal, bt.PositionManager._close

    def opened(self, *a, **k):
        opens.append(k["now"])
        return orig_open(self, *a, **k)

    def closed(self, pos, raw_price, trigger, now):
        closes.append((trigger, now))
        return orig_close(self, pos, raw_price, trigger, now)

    with patch.object(bt, "get_sector", return_value="UNKNOWN"), \
            patch.object(bt, "generate_signal", lambda s, d, **k: Signal(SignalAction.LONG, s, 100, 95, 120)), \
            patch.object(bt.RiskManager, "size_stock", return_value=SimpleNamespace(approved=True, sizing={"shares": 10})), \
            patch.object(bt.PositionManager, "open_from_signal", opened), \
            patch.object(bt.PositionManager, "_close", closed), \
            patch.dict(vars(CONFIG.backtest), impact_coef=0, reentry_policy=policy), \
            patch.dict(vars(CONFIG.regime), enable_regime_risk=False), \
            patch.dict(vars(CONFIG.risk), flat_before_close_min=0, enable_breakeven=False, enable_trailing=False):
        runner()
    return opens, closes


@pytest.mark.parametrize("kind", ["symbol", "portfolio"])
def test_cikis_bari_ayni_iterasyonda_yeniden_giris_yok(kind):
    df = _df()

    def runner():
        if kind == "symbol":
            bt.backtest_symbol("AUDIT", df=df, warmup_bars=20)
        else:
            bt.backtest_portfolio(["AUDIT"], data={"AUDIT": df}, warmup_bars=20)

    opens, closes = _run(runner, "next_bar")
    stop_t = [t for trig, t in closes if trig == "STOP"]
    assert stop_t == [df.index[22].to_pydatetime()]
    # İkinci giriş kararı en erken çıkış barının KAPANIŞINDA (bar 22) verilir → giriş bar 23.
    assert len(opens) == 2 and opens[1] >= stop_t[0]

    opens_old, closes_old = _run(runner, "same_bar")
    assert opens_old[1] < [t for trig, t in closes_old if trig == "STOP"][0]   # eski kusur üretilebiliyor


def test_sentetik_opsiyon_cikis_iterasyonunda_giris_yok():
    n, w = 12, 3
    df = pd.DataFrame(dict(open=[100.] * n, high=[101.] * n, low=[89.] * n, close=[90.] * n,
                           volume=[1e6] * n), index=pd.date_range("2024-01-01", periods=n, freq="D"))
    calls: list[int] = []

    def sig(s, d, **k):
        calls.append(len(d))
        return Signal(SignalAction.LONG, s, 100, 95, 120)

    def fake_bs(flag, S, K, T, r, sigma):          # giriş (S_open=100) 1,0 · yönetim (S_close=90) 0,1
        return 1.0 if S == 100 else 0.1

    def koş(policy):
        calls.clear()
        with patch.object(bt, "get_sector", return_value="UNKNOWN"), patch.object(bt, "generate_signal", sig), \
                patch("vollib.black_scholes.black_scholes", fake_bs), \
                patch.object(bt.RiskManager, "size_long_option",
                             return_value=SimpleNamespace(approved=True, sizing={"contracts": 1})), \
                patch.dict(vars(CONFIG.backtest), reentry_policy=policy):
            r = bt.backtest_option_synth("AUDIT", df=df, warmup_bars=w)
        return list(calls), r

    yeni, r_yeni = koş("next_bar")
    eski, r_eski = koş("same_bar")
    assert r_yeni.trades_count > 0 and np.all(np.diff(yeni) >= 2)      # çıkış iterasyonu atlanır
    assert np.all(np.diff(eski) == 1)                                 # eski: her iterasyonda karar


def test_etkin_sinir_tavani_normalizasyonla_delmez():
    """R2-B1: 3 varlık × %30 tavan → ulaşılabilir bütçe 0,90; sınır noktası tavanı aşamaz."""
    from nihai.ai import portfolio as pf
    from nihai.web import api_quant as aq

    r = np.random.default_rng(78).normal(0, .01, (100, 3)) * [1, 2, 3]
    panel = pd.DataFrame(100 * np.exp(np.vstack([np.zeros(3), r.cumsum(axis=0)])), columns=list("ABC"))
    rets = np.diff(np.log(panel.to_numpy()), axis=0)
    got = dict(panel=panel, names=list(panel), missing=[], rets=rets, ppy=252)
    with patch.object(aq, "_symbol_panel", return_value=(got, None)), patch.object(aq, "_faktor_bloku", return_value=None):
        api = aq.quant_portfolio(covariance_model="rmt", max_weight=.30)
    assert api["ok"] and api["frontier"]["vol"]
    cov = np.asarray(pf.propose([{"symbol": s} for s in panel], panel,
                                {"covariance_model": "rmt", "max_weight": .30})["covariance"])
    w_mv = pf.min_variance_weights(cov, (np.zeros(3), np.full(3, .30)), 1.0)
    assert float(np.sum(w_mv)) == pytest.approx(.90) and float(np.max(w_mv)) <= .30 + 1e-9
    assert api["frontier"]["vol"][0] == pytest.approx(float(np.sqrt(w_mv @ cov @ w_mv * 252)), abs=6e-5)


def test_vpin_ve_bvc_nedensel_olmadigini_tasir():
    """R3: σ tüm pencereden → yalnız SON fiyat değişince ilk kova değişir; bayrak bunu söyler."""
    from nihai.ai.microstructure import bulk_volume_classification, vpin
    p = 100 + np.r_[0, np.cumsum([1., -1, 2, -2, 1, 2, -1, 1, -2, 2, -1, 1])]
    v = np.ones(p.size) * 10
    q = p.copy()
    q[-1] += 100
    a, b = vpin(p, v, 2), vpin(q, v, 2)
    assert abs(a["buckets"][0] - b["buckets"][0]) > .01          # geçmiş kova gelecekten etkileniyor
    assert a["causal"] is False and bulk_volume_classification(p, v)["causal"] is False


def test_sec_retry_after_http_tarihi_erken_istek_gondermez():
    """R3: `Retry-After: <HTTP tarihi>` eskiden 30 sn sayılıyordu; sunucu 120 sn istemişti."""
    from datetime import UTC, datetime
    from email.utils import format_datetime

    import requests

    from nihai.data import insider
    start = datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    clock, dispatch = [0.0], []
    header = format_datetime(datetime.fromtimestamp(start + 120, UTC), usegmt=True)

    def fake_get(*a, **k):
        dispatch.append(clock[0])
        if len(dispatch) == 1:
            def fail():
                raise requests.HTTPError("429")
            return SimpleNamespace(status_code=429, headers={"Retry-After": header}, raise_for_status=fail)
        return SimpleNamespace(status_code=200, json=lambda: {}, raise_for_status=lambda: None)

    fake_time = SimpleNamespace(monotonic=lambda: clock[0], time=lambda: start + clock[0],
                                sleep=lambda d: clock.__setitem__(0, clock[0] + d))
    with patch.object(insider, "time", fake_time), patch.object(insider, "_THROTTLE", 0), \
            patch.object(insider, "_retry_after", 0), patch.object(insider, "_headers", return_value={}), \
            patch.object(requests, "get", fake_get):
        assert insider._get("mock:ilk") is None
        assert insider._get("mock:ikinci") == {}
    assert dispatch[-1] >= 120.0


@pytest.mark.parametrize("kw", [dict(equity=None), dict(multiplier=None), dict(floor_value=[]),
                                dict(equity="x"), dict(risky_cap=object())])
def test_cppi_donusturulemeyen_girdi_none(kw):
    from nihai.ai.portfolio import cppi_allocation
    args = dict(equity=100.0, floor_pct=.9, multiplier=3.0, risky_cap=1.0) | kw
    assert cppi_allocation(**args) is None


def test_log_getiri_yillikligi_bilesik_formulle_karismaz():
    """R2-B2: log fark girdisine (1+m)^ppy−1 uygulanıyordu. Bağımsız çapa: fiyat oranı."""
    from nihai.ai import quant_math as qm
    from nihai.web import api_quant as aq
    m = .005
    panel = pd.DataFrame(100 * np.exp(m * np.arange(253))[:, None] * np.ones(2), columns=["A", "B"])
    oran = float(panel["A"].iloc[-1] / panel["A"].iloc[0]) - 1          # 252 bar = 1 yıl → 2,525421
    assert qm.annualize_return(m, 252, log=True) == pytest.approx(oran, rel=1e-12)
    assert qm.annualize_return(m, 252) == pytest.approx(1.005 ** 252 - 1)            # basit yol aynen
    rets = np.diff(np.log(panel.to_numpy()), axis=0) + np.random.default_rng(1).normal(0, 1e-4, (252, 2))
    got = dict(panel=panel, names=list(panel), missing=[], rets=rets, ppy=252)
    with patch.object(aq, "_symbol_panel", return_value=(got, None)), patch.object(aq, "_faktor_bloku", return_value=None):
        api = aq.quant_portfolio(max_weight=1.0)
    beklenen = [round(math.expm1(252 * float(x)), 4) for x in rets.mean(axis=0)]
    assert api["asset_stats"]["return_annual"] == beklenen
    assert abs(api["asset_stats"]["return_annual"][0] - oran) < .02
    # Yöntem tablosu satırı ve `propose` özeti de log yıllıklandırır (mutasyon A8/A9 kaçıyordu):
    # `propose` getirisini gürültüsüz panelden kurar → bütçe 1'de her yöntem 2,5254 (basit bileşik 2,5144).
    tam = [row for row in api["methods"].values()
           if row["return_annual"] is not None and abs(sum(row["weights"]) - 1) < 1e-3]
    assert tam and all(abs(row["return_annual"] - oran) < 2e-3 for row in tam)
    from nihai.ai.portfolio import _summarize
    ozet = _summarize(["A", "B"], np.array([.5, .5]), np.array([m, m]), np.eye(2) * 1e-4, 252)
    assert ozet["expected_return_annualized"] == pytest.approx(oran, rel=1e-12)


@pytest.mark.parametrize("strikes,ivs,spot,beklenen", [
    ([70., 100., 130.], [.9, np.nan, np.nan], 100., None),      # R1-B7: 70'in IV'si ATM sayılamaz
    ([95., 105.], [.2, .3], 100., .25),
    ([100., 105.], [.2, .3], 102., .24),                        # strike'ta doğrusal enterpolasyon
    ([95., 100., 105.], [.2, np.nan, .3], 100., None),          # spot tam strike'ta ve IV'si yok
    ([95., 105., 110.], [np.nan, .3, .4], 100., .3),            # yalnız bir komşu ölçülü
])
def test_atm_iv_yalniz_spotu_cevreleyen_strikelardan(strikes, ivs, spot, beklenen):
    from datetime import date, timedelta

    from nihai.web import api_quant as aq

    class Saglayici:
        def get_option_chain(self, *a):
            side = pd.DataFrame(dict(strike=strikes, impliedVolatility=ivs))
            return dict(calls=side, puts=side, underlying_price=spot,
                        expiry=(date.today() + timedelta(days=30)).isoformat())

    with patch.object(aq, "_provider", lambda: Saglayici()):
        iv, _ = aq._current_atm_iv("AUDIT")
    assert iv == (None if beklenen is None else pytest.approx(beklenen))
    if beklenen is None:
        x = np.full(300, 100.)
        data = dict(open=x, high=x * 1.01, low=x * .99, close=x, dates=["2026-09-14"] * 300)
        with patch.object(aq, "_provider", lambda: Saglayici()):
            koni = aq._cone_report(data, "AUDIT", 252, "parkinson")
        assert koni["iv_percentile"] in (None, [None] * 7)


def test_mc_gbm_medyan_yolu_tarihsel_log_suruklenmeyi_korur():
    """Tur-78: log ortalaması GBM μ'sü diye veriliyordu → İtô düzeltmesi iki kez düşülüyor,
    medyan yol ½σ²T kadar aşağı kayıyordu (bu seride eski kod −%0,94). Bu seride m=0."""
    from nihai.web import api_quant as aq
    a = .03
    close = 100 * np.exp(np.cumsum(np.r_[0.0, np.tile([a, -a], 150)]))
    data = dict(dates=["2026-01-01"] * close.size, open=close, high=close, low=close, close=close, volume=None)
    with patch.object(aq, "_ohlc", return_value=data):
        out = aq.quant_mc("AUDIT", horizon=21, paths=20000, seed=7)
    assert out["ok"] and abs(out["terminal"]["p50"]) < .003


def test_opsiyon_dolumu_yalniz_sonraki_nyse_seansinda():
    """R1-B2: karar 2024-02-05; 02-06 dayanak barı ve kotasyonu yok → giriş İPTAL. Eskiden döngünün
    sonraki günü (02-07, iki seans sonra) 3,7 ask ile alınıyordu. Barlar tamken 02-06'da alınır."""
    from dataclasses import replace

    days = pd.bdate_range("2024-02-05", periods=5)
    bars = pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=1e6), index=days)
    chain = pd.DataFrame([dict(date=d, act_symbol="AAA", expiration="2024-03-15", strike=k, call_put="Call",
                               bid=b, ask=a, vol=.3, delta=.5 if k == 100 else .8, gamma=.02, theta=-.03, vega=.1)
                          for d, b, a in zip(days, [1.9, 2.2, 3.5, 3., 2.7], [2., 2.5, 3.7, 3.2, 2.9], strict=True)
                          for k in (100., 105.)])

    def kos(b, c):
        sig = lambda s, data, **k: SimpleNamespace(action=SignalAction.LONG, is_actionable=lambda: len(data) == 1)  # noqa: E731
        cfg = replace(bt.CONFIG, option_strategy=replace(bt.CONFIG.option_strategy,
                                                          premium_target_gain_pct=.25, premium_stop_loss_pct=.5))
        with patch.object(bt, "generate_signal", sig), patch.object(bt, "CONFIG", cfg), \
                patch.object(bt.RiskManager, "size_long_option",
                             lambda *a, **k: SimpleNamespace(approved=True, sizing={"contracts": 2})):
            return bt.backtest_option_chain("AAA", df=b, chain_df=c, warmup_bars=0, initial_equity=10000)

    eksik = kos(bars.drop(days[1]), chain.loc[chain.date != days[1]])
    assert eksik.option_data["fills"] == [] and eksik.option_data["unfilled_entries"] == 1
    tam = kos(bars, chain)
    assert tam.option_data["fills"][0]["date"] == "2024-02-06"


@pytest.mark.parametrize("varlik,beklenen", [(1e-308, True), (1000.0, True), (-5.0, None), (None, None)])
def test_mohanram_tahakkuk_tutar_kiyasi_tasmaz(varlik, beklenen):
    """R3-B6: VA=1e-308'de iki oran da inf → `inf > inf` False. CFO=20 > NI=10 ⟺ sağlıklı."""
    from nihai.ai.fundamentals import _bol, mohanram_g_score
    cari = {"varliklar": varlik, "net_kar": 10.0, "faaliyet_nakit": 20.0}
    assert mohanram_g_score(cari, {})["sinyaller"]["tahakkuk_saglikli"] is beklenen
    assert _bol(10.0, 1e-308) is None and _bol(10.0, 4.0) == 2.5


def test_edgar_ve_fda_bozuk_girdide_istisna_yok():
    """R3-B7: warm(None) ve NaN submissions/products TypeError atıyordu."""
    from nihai.data.edgar_facts import warm
    from nihai.data.fda_calendar import approval_events
    assert warm(None) is None and warm(["AAA"], workers="x") is None
    kayit = {"application_number": "NDA1", "products": float("nan"),
             "submissions": [{"submission_status": "AP", "submission_status_date": "20250101",
                              "submission_type": "ORIG", "submission_number": "1"}]}
    olay = approval_events({"results": [kayit, {"submissions": float("nan")}]}, "2025-01-01", "2025-12-31")
    assert len(olay) == 1 and olay[0]["products"] == []


def _denetim_kovaryanslari(adet: int = 400) -> list[np.ndarray]:
    """09-14 fark testinin üreteci (tohum 11): düşük ranklı faktör + özgün varyans, N 2–8."""
    rng = np.random.default_rng(11)
    out = []
    for _ in range(adet):
        n = int(rng.integers(2, 9))
        b = rng.normal(0, 1, (n, int(rng.integers(1, 3))))
        cov = b @ b.T * rng.uniform(.5, 2) + np.diag(rng.uniform(.05, 1.0, n))
        out.append((cov + cov.T) / 2)
    return out


def test_erc_konveks_cozum_paylari_esitler_ve_herc_none_donmez():
    """Tur-78: SLSQP'nin konveks olmayan amacı 400 kovaryansın 149'unda bir ağırlığı 0'a
    yapıştırıyordu (pay hatası tam 1/n); katı Armijo'lu ilk Newton sürümü 6'sında yuvarlamaya
    takılıp aynı yedeğe düşüyordu. Ölçek değişmezliği de sınanır: aynı korelasyon, farklı ölçek."""
    from nihai.ai.portfolio import erc_weights, herc_weights

    def pay_hatasi(w, cov):
        return float(np.max(np.abs(w * (cov @ w) / float(w @ cov @ w) - 1 / cov.shape[0])))

    # Eşik 1e-10: mutasyon A14 (Armijo yuvarlama payı 0) 1e-8 eşiğinden KAÇTI ama ölçüldü — aynı 400
    # kovaryansta maks pay hatası 1,5e-11 → 1,6e-9, 5,4× yavaş; patolojik 3000'de None 21 → 43.
    for cov in _denetim_kovaryanslari():
        for olcek in (1.0, 1 / float(np.diag(cov).max()), 1e-6):
            w = erc_weights(cov * olcek)
            assert w is not None and pay_hatasi(w, cov) < 1e-10
        assert herc_weights(cov, 1) is not None


def test_erc_iki_varlikta_korelasyondan_bagimsiz_kapali_form():
    """İki varlıkta ERC her korelasyonda w ∝ 1/σ'dır (RC eşitliği w1σ1 = w2σ2'ye indirgenir)."""
    from nihai.ai.portfolio import erc_weights
    for rho in (-.95, -.5, 0.0, .5, .95, .999):
        s = np.array([.1, .4])
        cov = np.array([[s[0] ** 2, rho * s[0] * s[1]], [rho * s[0] * s[1], s[1] ** 2]])
        np.testing.assert_allclose(erc_weights(cov), (1 / s) / np.sum(1 / s), atol=1e-12)


def _cvar_denetim_vakalari(k: int) -> list[tuple[np.ndarray, float]]:
    """Codex P3 salt-okunur ölçümünün üreteci (tohum 20260915), çağrı SIRASI birebir."""
    rng = np.random.default_rng(20260915)
    out = []
    for _ in range(k):
        t, n = int(rng.integers(40, 260)), int(rng.integers(2, 5))
        alpha, df = float(rng.choice([.05, .1, .025])), float(rng.uniform(2.5, 8))
        sigma, mu = rng.uniform(.005, .03, n), rng.uniform(-.001, .001, n)
        out.append((rng.standard_t(df, size=(t, n)) * sigma + mu, alpha))
    return out


def test_cvar_paritesi_kinkte_cozumu_olan_problemde_none_donmez():
    """P3: 300 problemin 40'ında tam çözüm VARDI (bağımsız düzgün dual + KKT, hata ~1e-15) ama kink
    LP'si payları A_eq ile TAM 1/N istediği için None dönüyordu (vaka 0, 10). ±1e-5 bant biçimi köşeyi
    bant sınırına oturttuğu için 202'ye çıkıyordu (vaka 1). Minimaks LP üçünü de çözer."""
    from nihai.ai.portfolio import _cvar_tail_weights, cvar_risk_parity

    vakalar = _cvar_denetim_vakalari(11)
    for i in (0, 1, 10):
        r, alpha = vakalar[i]
        res = cvar_risk_parity(r, alpha=alpha)
        assert res is not None and res["max_error"] <= 1e-5 and res["kink"] is True
        w = res["weights"]
        assert abs(w.sum() - 1) < 1e-12 and (w > 0).all()
        np.testing.assert_allclose(res["contributions"].sum(), res["es"], rtol=1e-6)

        # Bağımsız optimallik: f(x) = ES(x) − mean(log x) sıkı konveks; minimumda ES(x*) = 1 olduğundan
        # x* = w / ES(w). Rastgele küçük çarpımsal sapmalar amacı DÜŞÜREMEZ (alt-türev LP'sinden bağımsız).
        def f(x, r=r, alpha=alpha):
            losses = -r @ x
            return float(_cvar_tail_weights(losses, alpha) @ losses) - float(np.mean(np.log(x)))

        losses = -r @ w
        x_star = w / float(_cvar_tail_weights(losses, alpha) @ losses)
        rng = np.random.default_rng(i)
        f0 = f(x_star)
        assert min(f(x_star * np.exp(1e-3 * rng.normal(size=len(w)))) for _ in range(200)) >= f0 - 1e-9
