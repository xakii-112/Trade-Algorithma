"""Tur-82 — canlı veri yolu kilitleri (rapor: docs/TUR82_RAPORU_2026-09-18.md).

BULGU-82-1  Alpaca IEX akışı 30 sembol × (bars+quotes+trades) istiyordu; Basic planda trades+quotes
            TOPLAMI ≤30'dur (canlı hesapta ölçüldü) → sunucu isteği ATOMİK reddetti (405) ve akış
            "bağlı ama 0 mesaj" kaldı. Hata yalnız stdlib log'una düşüyordu.
BULGU-82-2  `HubProvider` kesicisi ReadTimeout'ta da açılıyordu (docstring "yalnız bağlantı").
BULGU-82-3  `scan_symbol` zinciri (2 çağrı) ve kotasyonu HER taramada çekiyordu; 6209 WAIT'in hiçbiri
            zincirin girdiği kalite adımına ulaşmadı → `generate_signal(chain_eval=...)`.
Ek          IV örnekleme kısıtı (IV_COLLECT_MIN) · pozisyon 1m bar çekimi havuzda.
"""
from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest

import nihai.execution.main as main
import nihai.signals.strategy as strategy_mod
from nihai.data import alpaca_stream_provider as asp
from nihai.data.data_provider import HubProvider
from nihai.risk.position_manager import PositionManager, SlippageModel
from nihai.signals.strategy import generate_signal


# ============================================================ BULGU-82-1: akış abonelik bütçesi
@pytest.mark.parametrize("n", range(0, 41))
def test_kanal_plani_butceyi_asmaz_ve_kotasyonu_once_verir(n):
    syms = [f"S{i}" for i in range(n)]
    bars, quotes, trades = asp._channel_plan(syms, 30)
    assert bars == syms, "bars sınıra SAYILMAZ (ölçüldü: 150'de kabul) → hepsi abone"
    assert len(quotes) + len(trades) <= 30, "trades+quotes toplamı Basic bütçeyi aşarsa 405 → HİÇBİR şey"
    assert quotes == syms[:30]
    assert set(trades) <= set(quotes)
    if n <= 15:
        assert trades == syms, "bütçe yetiyorsa trades de verilir (last = son işlem)"


def test_canli_listesi_30_sembolde_trades_acilmaz():
    _, quotes, trades = asp._channel_plan([f"S{i}" for i in range(30)], 30)
    assert len(quotes) == 30 and trades == []


def _saglayici(symbols):
    return asp.AlpacaStreamProvider(symbols=symbols, rest=SimpleNamespace(), max_symbols=30)


def test_run_loop_sunucuya_butce_icinde_abonelik_ister(monkeypatch):
    import alpaca.data.live as live
    kurulan = []
    prov = _saglayici([f"S{i}" for i in range(30)])

    class _SahteAkis:
        def __init__(self, *a, **k):
            self.istek: dict[str, int] = {}
            kurulan.append(self)

        def subscribe_bars(self, h, *s):
            self.istek["bars"] = len(s)

        def subscribe_quotes(self, h, *s):
            self.istek["quotes"] = len(s)

        def subscribe_trades(self, h, *s):
            self.istek["trades"] = len(s)

        def run(self):
            prov._stop.set()

    monkeypatch.setattr(live, "StockDataStream", _SahteAkis)
    prov._server_subs = {"bars": 30, "quotes": 30, "trades": 0}      # ÖNCEKİ bağlantının onayı
    prov._run_loop()
    assert prov.stream_status()["server_subscriptions"] is None, "yeni bağlantı eski onayı taşımamalı"
    akis = kurulan[0]
    assert akis.istek == {"bars": 30, "quotes": 30}, "30 × 3 kanal = 60 → 405 (BULGU-82-1)"
    assert akis._nihai_control == prov._on_control, "sunucu denetim mesajları sağlayıcıya bağlı değil"


def test_sunucu_405_hatasi_ve_onayi_durumda_gorunur():
    prov = _saglayici(["AAPL"])
    assert prov.stream_status()["server_subscriptions"] is None
    prov._on_control({"T": "error", "code": 405, "msg": "symbol limit exceeded"})
    st = prov.stream_status()
    assert "405" in st["last_error"] and "symbol limit" in st["last_error"]
    prov._on_control({"T": "subscription", "bars": ["AAPL"], "quotes": ["AAPL"], "trades": []})
    assert prov.stream_status()["server_subscriptions"] == {"bars": 1, "quotes": 1, "trades": 0}


def test_dispatch_yamasi_denetim_mesajini_saglayiciya_iletir():
    import alpaca.data.live.websocket as ws
    assert ws.DataStream._dispatch.__name__ == "_patched_dispatch"
    gelen: list[dict] = []
    akis = SimpleNamespace(_nihai_control=gelen.append,
                           _handlers={"trades": {}, "quotes": {}, "bars": {}})
    asyncio.run(ws.DataStream._dispatch(akis, {"T": "error", "code": 405, "msg": "symbol limit exceeded"}))
    asyncio.run(ws.DataStream._dispatch(akis, {"T": "subscription", "bars": ["A"]}))
    assert [m["T"] for m in gelen] == ["error", "subscription"]
    # sağlayıcı bağlamadıysa (ör. MainHub dışı kullanım) eski davranış: yalnız SDK logu
    asyncio.run(ws.DataStream._dispatch(SimpleNamespace(_handlers={"bars": {}}),
                                        {"T": "error", "code": 1, "msg": "x"}))


def test_dinamik_abonelik_trades_acmaz_ve_kanal_butcesini_korur():
    prov = _saglayici([])
    cagri: list[tuple[str, str]] = []
    prov._stream = SimpleNamespace(subscribe_bars=lambda h, s: cagri.append(("bars", s)),
                                   subscribe_quotes=lambda h, s: cagri.append(("quotes", s)),
                                   subscribe_trades=lambda h, s: cagri.append(("trades", s)))
    prov._stream_ok = True
    prov._subscribed = {f"S{i}" for i in range(25)}
    prov._trade_syms = {f"S{i}" for i in range(4)}
    prov._ensure_subscribed("YENI1")   # 25 kotasyon + 4 trades = 29 < 30 → eklenir
    prov._ensure_subscribed("YENI2")   # 26 + 4 = 30 → bütçe dolu, REST'e bırakılır
    assert cagri == [("bars", "YENI1"), ("quotes", "YENI1")]


def test_seed_kapanmis_bari_otoritedir_eksik_canli_saat_ezmez(monkeypatch):
    """Akış yeniden canlanınca devreye giren yol: kopma sonrası EKSİK dakikalardan kurulan saat,
    REST seed'inin TAM barını ezmemeli; seed'den SONRAKİ kapanmış saat canlıdan gelir."""
    import datetime as _dt
    from collections import deque

    # Tur-91: ızgara seans açılışına çapalı (13:30 UTC = 09:30 EDT); senaryo 30 dk kaydırıldı.
    simdi = _dt.datetime(2025, 6, 2, 16, 0, tzinfo=_dt.UTC)
    monkeypatch.setattr(asp, "datetime", SimpleNamespace(now=lambda tz=None: simdi.astimezone(tz)))

    class _Rest:
        def get_bars(self, symbol, interval, lookback):   # 13:30 kapanmış TAM bar
            return pd.DataFrame({"open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0],
                                 "volume": [5000.0]}, index=pd.to_datetime(["2025-06-02 13:30"]))

    prov = asp.AlpacaStreamProvider(symbols=[], rest=_Rest(), max_symbols=30)
    prov._minutes["AAPL"] = deque([
        (pd.Timestamp("2025-06-02 14:10", tz="UTC"), 100.5, 100.6, 100.4, 100.5, 100.0),  # kopma: eksik saat
        (pd.Timestamp("2025-06-02 14:30", tz="UTC"), 101.0, 101.5, 100.8, 101.2, 300.0),
        (pd.Timestamp("2025-06-02 15:29", tz="UTC"), 101.2, 101.9, 101.1, 101.8, 400.0),
        (pd.Timestamp("2025-06-02 15:40", tz="UTC"), 101.8, 102.2, 101.7, 102.0, 200.0),  # oluşan saat
    ])
    out = prov.get_bars("AAPL", "1h")
    assert list(out.index) == list(pd.to_datetime(["2025-06-02 13:30", "2025-06-02 14:30"]))
    assert out.loc["2025-06-02 13:30", ["high", "volume"]].tolist() == [102.0, 5000.0], "eksik saat ezdi"
    assert out.loc["2025-06-02 14:30", ["open", "high", "close", "volume"]].tolist() == [101.0, 101.9, 101.8, 700.0]


def test_alpaca_rest_429da_uyumaz_hemen_yedege_duser(monkeypatch):
    """Hub'da 429 yeniden denemesi (3 sn × 3) istemcinin 6 sn zaman aşımını aşıyordu."""
    from alpaca.data.historical import StockHistoricalDataClient

    from nihai.data.alpaca_provider import AlpacaProvider
    gercek = StockHistoricalDataClient("k", "s")          # ağa çıkmaz; yalnız kurulur
    assert hasattr(gercek, "_retry"), "alpaca-py yeniden deneme alanının adı değişmiş — Tur-82 ayarı boşa düşer"
    prov = AlpacaProvider(api_key="k", secret_key="s")
    assert prov._get_client()._retry == 0

    import alpaca.common.rest as rest
    cagri = []

    class _Yanit:
        status_code, text = 429, '{"message": "too many requests."}'

        def raise_for_status(self):
            import requests
            raise requests.HTTPError(response=self)

    monkeypatch.setattr(prov._get_client()._session, "request", lambda *a, **k: cagri.append(1) or _Yanit())
    monkeypatch.setattr(rest.time, "sleep", lambda s: pytest.fail(f"429'da {s} sn uyudu"))
    with pytest.raises(rest.APIError):
        prov._get_client()._request("GET", "/stocks/bars")
    assert cagri == [1], "429 tek istekte hataya dönmeli (yeniden deneme YOK)"


# ============================================================ BULGU-82-2: kesici yalnız "hub kapalı"da
@pytest.fixture
def _temiz_kesici():
    HubProvider._down_until.clear()
    yield
    HubProvider._down_until.clear()


class _HataIstemci:
    def __init__(self, hata: BaseException):
        self.hata = hata
        self.cagri = 0

    def get(self, url, params=None):
        self.cagri += 1
        raise self.hata


def _hub(istemci) -> HubProvider:
    h = HubProvider(base_url="http://127.0.0.1:8799")
    h._client = istemci
    return h


@pytest.mark.parametrize("hata", [httpx.ReadTimeout("yavaş"), httpx.WriteTimeout("yavaş"),
                                  httpx.PoolTimeout("dolu"), httpx.RemoteProtocolError("kopuk yanıt")])
def test_yavas_ama_ayakta_hub_kesiciyi_ACMAZ(hata, _temiz_kesici):
    c = _HataIstemci(hata)
    h = _hub(c)
    assert h._get("/quote", {}) is None      # BU çağrı yedeğe düşer
    assert h._breaker_open() is False, "tek geç yanıt süreci 30 sn yfinance'e yollamamalı"
    h._get("/quote", {})
    assert c.cagri == 2, "sonraki çağrı yine hub'a gitmeli"


@pytest.mark.parametrize("hata", [httpx.ConnectError("reddedildi"), httpx.ConnectTimeout("syn"),
                                  ConnectionRefusedError()])
def test_kapali_hub_kesiciyi_acar(hata, _temiz_kesici):
    c = _HataIstemci(hata)
    h = _hub(c)
    assert h._get("/bars", {}) is None
    assert h._breaker_open() is True
    h._get("/bars", {})
    assert c.cagri == 1


# ============================================================ BULGU-82-3: tembel zincir (karar-denk)
def _cerceve(seed: int, n: int = 200) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    ret = r.normal(0.0004, 0.01, n)
    if seed % 3 == 0:                                    # son barlarda kırılım → kalite adımı
        ret = ret + np.where(np.arange(n) > n - 8, r.normal(0.01, 0.01), 0)
    c = 100 * np.exp(np.cumsum(ret))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + r.uniform(0, 0.006, n))
    lo = np.minimum(o, c) * (1 - r.uniform(0, 0.006, n))
    v = r.uniform(5e5, 2e6, n)
    v[-1] *= r.choice([0.5, 3.0])
    idx = pd.date_range("2026-01-05 09:30", periods=n, freq="h")
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v}, index=idx)


def _imza(s):
    return (s.action, s.symbol, s.price, s.stop, s.target, tuple(s.reasons), s.quality_score,
            s.rr_ratio, s.metadata)


def _kalite_adimi(sig) -> bool:
    return any(r.startswith("QUALITY_") for r in sig.reasons)


@pytest.mark.parametrize("iv_kip", ["off", "band"])
def test_tembel_zincir_istekli_ile_bit_denk_ve_yalniz_kalite_adiminda_cagrilir(monkeypatch, iv_kip):
    cfg = strategy_mod.CONFIG
    monkeypatch.setattr(strategy_mod, "CONFIG", dataclasses.replace(
        cfg, strategy=dataclasses.replace(cfg.strategy, iv_regime_bonus_mode=iv_kip)))
    ulasan = oncesi = ceviren = 0
    for seed in range(240):
        df = _cerceve(seed)
        eylem: set[bool] = set()
        for ok, ivr in ((True, None), (False, None), (True, 40.0), (False, 10.0)):
            istekli = generate_signal("TST", df, chain_health_ok=ok, iv_rank_val=ivr,
                                      strategy_mode="ensemble")
            cagri: list[int] = []
            tembel = generate_signal("TST", df, strategy_mode="ensemble",
                                     chain_eval=lambda ok=ok, ivr=ivr, c=cagri: c.append(1) or (ok, ivr))
            assert _imza(tembel) == _imza(istekli), f"seed={seed} ok={ok} ivr={ivr}"
            assert len(cagri) == int(_kalite_adimi(tembel)), "zincir yalnız kalite adımında istenmeli"
            eylem.add(istekli.is_actionable())
        if _kalite_adimi(tembel):
            ulasan += 1
        else:
            oncesi += 1
        ceviren += len(eylem) > 1
    # kurgu dejenere değil: üç dalın üçü de kapsandı (Tur-73 dersi)
    assert ulasan > 20 and oncesi > 100 and ceviren > 0, (ulasan, oncesi, ceviren)


# ----------------------------------------------------------- scan_symbol çağrı sayıları
class _SayacliSaglayici:
    def __init__(self, df, kote=None):
        self.df = df
        self.kote = kote
        self.sayac = {"bars": 0, "quote": 0, "chain": 0}

    def get_bars(self, sym, interval, lookback_bars=200):
        self.sayac["bars"] += 1
        return self.df

    def get_quote(self, sym):
        self.sayac["quote"] += 1
        return self.kote

    def get_option_chain(self, sym, expiry=None):
        self.sayac["chain"] += 1
        return {"calls": pd.DataFrame({"strike": [100.0]}), "expiry": "2026-10-16",
                "all_expiries": ["2026-10-16"], "underlying_price": 100.0}

    def is_quote_healthy(self, q):
        return (q.bid > 0, "BOZUK_KOTE")


class _Durum:
    kill_switch_active = False

    def cooldown_active(self, s, n):
        return False

    def wash_sale_check(self, s, n, k=""):
        return (False, "")

    def good_faith_warning(self, x):
        return (False, "")

    def set_cooldown(self, s, n):
        pass


def _kur(monkeypatch):
    kayit = SimpleNamespace(blok=[], sinyal=[], acilan=[])
    monkeypatch.setattr(main, "can_enter_new_position", lambda: (True, "ok"))
    monkeypatch.setattr(main, "event_guard", lambda s: (False, ""))
    monkeypatch.setattr(main, "corporate_action_guard", lambda s: (False, ""))
    monkeypatch.setattr(main, "halt_or_luld_guard", lambda df: (False, ""))
    monkeypatch.setattr(main, "get_sector", lambda s: "Technology")
    monkeypatch.setattr(main, "log_block", lambda code, msg="": kayit.blok.append(code))
    monkeypatch.setattr(main, "log_signal", lambda sig, accepted, extra="": kayit.sinyal.append(
        (sig.action.value, accepted, extra)))
    monkeypatch.setattr(main, "log_decision", lambda *a, **k: None)
    monkeypatch.setattr(main, "filter_liquid_chain", lambda df, **k: SimpleNamespace(
        accepted=df, rejected_count=0, reasons={}))
    monkeypatch.setattr(main, "estimate_atm_iv", lambda *a, **k: None)
    monkeypatch.setattr(main.shadow_bridge, "evaluate", lambda *a, **k: None)
    monkeypatch.setattr(main.ai_advisor, "advise", lambda *a, **k: None)
    monkeypatch.setattr(main, "telegram_bot", SimpleNamespace(notify_signal=lambda *a, **k: None,
                                                               notify_alert=lambda *a, **k: None))
    monkeypatch.setattr(main, "LIVE_ENTRY_ANCHOR", False)
    risk = SimpleNamespace(size_stock=lambda **k: SimpleNamespace(
        approved=True, reason="", sizing={"notional": 1.0, "shares": 1, "risk_dollars": 1.0}),
        regime_label="—", regime_multiplier=1.0)
    pmgr = SimpleNamespace(open_from_signal=lambda sig, sz, **k: kayit.acilan.append(sig.symbol),
                           slip=SlippageModel())
    breaker = SimpleNamespace(can_emit=lambda: (True, ""), record_emit=lambda: None)
    main._IV_LAST.clear()
    main._alerted_bars.clear()

    def tara(prov):
        main.scan_symbol("TST", prov, risk, _Durum(), breaker, pmgr, None, None, strategy_mode="ensemble")
    return kayit, tara


def _ilk(kosul):
    for s in range(300):
        df = _cerceve(s)
        if kosul(generate_signal("TST", df, strategy_mode="ensemble")):
            return df
    raise AssertionError("kurgu bulunamadı")


def test_wait_taramasi_kotasyon_cekmez_zinciri_yalniz_iv_araliginda_ceker(monkeypatch):
    kayit, tara = _kur(monkeypatch)
    prov = _SayacliSaglayici(_ilk(lambda s: not _kalite_adimi(s)))
    tara(prov)                                   # ilk tarama: IV örneklemesi ZAMANI (hiç yok)
    assert prov.sayac == {"bars": 1, "quote": 0, "chain": 1}
    tara(prov)                                   # hemen ardından: örnekleme zamanı değil
    assert prov.sayac == {"bars": 2, "quote": 0, "chain": 1}
    main._IV_LAST["TST"] = time.monotonic() - main.IV_COLLECT_SEC - 1
    tara(prov)                                   # aralık doldu → yeniden örnekler
    assert prov.sayac == {"bars": 3, "quote": 0, "chain": 2}
    assert kayit.sinyal == [("WAIT", False, "")] * 3 and kayit.blok == []


def test_eyleme_donuk_taramada_zincir_bir_kez_kotasyon_bir_kez(monkeypatch):
    kayit, tara = _kur(monkeypatch)
    kote = SimpleNamespace(bid=99.9, ask=100.1, last=100.0, timestamp=None, spread_measured=True)
    prov = _SayacliSaglayici(_ilk(lambda s: s.is_actionable() and s.action.value == "LONG"), kote)
    tara(prov)
    assert (prov.sayac["quote"], prov.sayac["chain"]) == (1, 1)   # bars: HTF + beta kıyası
    assert kayit.acilan == ["TST"]


def test_saglıksiz_kote_eyleme_donuk_sinyali_hala_durdurur(monkeypatch):
    kayit, tara = _kur(monkeypatch)
    bozuk = SimpleNamespace(bid=0.0, ask=100.1, last=100.0, timestamp=None, spread_measured=False)
    prov = _SayacliSaglayici(_ilk(lambda s: s.is_actionable() and s.action.value == "LONG"), bozuk)
    tara(prov)
    assert "QUOTE_UNHEALTHY" in kayit.blok and kayit.acilan == []


def test_iv_ornekleme_kisiti_zamanla():
    main._IV_LAST.clear()
    assert main._iv_collect_due("X") is True
    main._IV_LAST["X"] = time.monotonic()
    assert main._iv_collect_due("X") is False
    main._IV_LAST["X"] = time.monotonic() - main.IV_COLLECT_SEC - 0.5
    assert main._iv_collect_due("X") is True


# ============================================================ pozisyon 1m bar çekimi havuzda
def test_1m_bar_cekimi_havuzda_sirali_dongu_ile_ayni_sozluk():
    nesne = {s: object() for s in "ABCDEFGH"}
    cagri: list[str] = []
    kilit = threading.Lock()

    class _Yavas:
        def get_bars(self, sym, interval, lookback_bars=3):
            with kilit:
                cagri.append(sym)
            time.sleep(0.05 if sym in "ACE" else 0.01)   # bitiş sırası karışsın
            if sym == "D":
                raise RuntimeError("ağ")
            return nesne[sym]

    sahte_pm = SimpleNamespace(slip=SimpleNamespace(impact_coef=0.0))
    girdi = list("ABCDDEFGHA")                               # tekrarlar tek çekilir
    t0 = time.perf_counter()
    out = PositionManager.fetch_1m_bars(sahte_pm, _Yavas(), girdi)
    sure = time.perf_counter() - t0
    beklenen = {s: (None if s == "D" else nesne[s]) for s in dict.fromkeys(girdi)}
    assert list(out) == list(beklenen) and all(out[k] is beklenen[k] for k in beklenen)
    assert sorted(cagri) == sorted(dict.fromkeys(girdi)), "her sembol TAM bir kez çekilmeli"
    assert sure < 0.15, f"havuz yok gibi: {sure:.3f} sn (sıralı ≥ 0,20 sn)"
