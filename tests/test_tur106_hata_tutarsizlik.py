"""Tur-106 — projedeki hata ve tutarsızlıklar (canlı günlüklerden izlenen kusurlar).

Her test bir BULGU'yu kilitler; kusurun kendi ifadesi kaynakta ARANIR (regresyon kapısı).
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# BULGU-106-1 — hub istemcisinin boşta bağlantı ömrü sunucununkine eşitti (keep-alive yarışı)
# ---------------------------------------------------------------------------
def test_hub_istemcisi_bosta_omru_sunucunun_keepalive_suresinden_kisa():
    import uvicorn
    from fastapi import FastAPI

    from nihai.data.data_provider import HubProvider

    hp = HubProvider(base_url="http://127.0.0.1:9")
    try:
        expiry = hp._client._transport._pool._keepalive_expiry
    finally:
        hp._client.close()
    sunucu = uvicorn.Config(FastAPI()).timeout_keep_alive   # hub_server `uvicorn.run` varsayılanı
    # ≥ 1 sn pay: istemcinin saati yanıtı OKUYUNCA başlar, sunucununki yazınca → istemci geride kalır.
    assert expiry is not None and expiry <= sunucu - 1.0, (expiry, sunucu)


def test_hub_istemcisi_varsayilan_havuzla_kurulmuyor():
    kaynak = (KOK / "nihai/data/data_provider.py").read_text(encoding="utf-8")
    assert "self._client = httpx.Client(timeout=timeout)\n" not in kaynak


# ---------------------------------------------------------------------------
# BULGU-106-2 — "[state] yüklendi" satırı birden çok durum dosyasında her yüklemede basılıyordu
# ---------------------------------------------------------------------------
def test_state_yukleme_satiri_dosya_basina_yalniz_degisince(tmp_path):
    from loguru import logger

    from nihai.execution.state import StateStore

    yollar = [tmp_path / f"kasa{i}.json" for i in range(3)]
    for i, y in enumerate(yollar):
        s = StateStore(y)
        s.load_or_init(1000.0 * (i + 1))   # farklı özsermaye: tek süreç-genel anahtar her seferinde değişir
        s.save()

    satirlar: list[str] = []
    hid = logger.add(lambda m: satirlar.append(str(m)), level="INFO",
                     filter=lambda r: "[state] yüklendi" in r["message"])
    try:
        for _ in range(5):                     # gölge kasa motoru gibi sırayla, tekrar tekrar
            for y in yollar:
                StateStore(y).load_or_init(1.0)
        assert len(satirlar) == 3              # dosya başına BİR kez (eskisi 15)

        s = StateStore(yollar[0])              # bir dosya DEĞİŞİNCE yalnız o yeniden yazılır
        s.load_or_init(1.0)
        assert s.portfolio is not None
        s.portfolio.equity += 7.0
        s.save()
        for y in yollar:
            StateStore(y).load_or_init(1.0)
        assert len(satirlar) == 4
    finally:
        logger.remove(hid)


# ---------------------------------------------------------------------------
# BULGU-106-3 — hub stderr'i (loguru) cp1254'te kalıyordu → hub_server.err.log bozuk bayt
# ---------------------------------------------------------------------------
def test_hub_sunucusu_ortak_utf8_yardimcisini_cagirir():
    kaynak = (KOK / "nihai/data/hub_server.py").read_text(encoding="utf-8")
    cagrilar = {n.func.id for n in ast.walk(ast.parse(kaynak))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "ensure_utf8_stdout" in cagrilar
    assert 'sys.stdout.reconfigure(encoding="utf-8")' not in kaynak


def test_hub_sunucusu_import_sonrasi_stderr_utf8_yazar():
    # Ortam değişkeniyle UTF-8 VERİLMEZ (BULGU-81-4'ün tersi): modül kendisi çevirmeli.
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    kod = "import sys, nihai.data.hub_server; sys.stderr.write('bar çekme hatası ğüşİ\\n')"
    r = subprocess.run([sys.executable, "-c", kod], cwd=KOK, env=env,
                       capture_output=True, timeout=180)
    assert r.returncode == 0, r.stderr[-500:]
    assert "bar çekme hatası ğüşİ".encode() in r.stderr


# ---------------------------------------------------------------------------
# BULGU-106-4 — otomatik tarayıcı aktif quant stratejisini 1h barla backtest edip hep 0 işlem alıyordu
# ---------------------------------------------------------------------------
def _sentetik(freq: str, n: int, egim: float):
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2024-01-02 14:30", periods=n, freq=freq)
    rng = np.random.default_rng(7)
    c = 100.0 * np.exp(np.cumsum(egim + 0.01 * rng.standard_normal(n)))
    return pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c,
                         "volume": 1e6}, index=idx)


def _strateji(monkeypatch, ad: str) -> None:
    """Çalışma anı strateji seçimi (pano `runtime_strategy.json` > `.env`) — generate_signal'ın kaynağı."""
    import nihai.runtime_strategy as R

    monkeypatch.setattr(R, "settings", lambda: {"strategy": ad})


def test_olculemeyen_strateji_yuklemi(monkeypatch):
    from nihai.execution import auto_screener as A

    _strateji(monkeypatch, "quant_trend")
    monkeypatch.setattr(A, "AUTO_SCREEN_INTERVAL", "1h")
    assert A.olculemeyen_strateji() is not None
    monkeypatch.setattr(A, "AUTO_SCREEN_INTERVAL", "1d")
    assert A.olculemeyen_strateji() is None
    _strateji(monkeypatch, "event_breakout")
    monkeypatch.setattr(A, "AUTO_SCREEN_INTERVAL", "1h")
    assert A.olculemeyen_strateji() is None


def test_yuklem_stratejinin_gercek_davranisina_bagli():
    """Yüklem bir varsayım değil: quant 1h barda 0 işlem, günlük barda işlem açar. Quant bir gün
    gün-içi bar okumayı öğrenirse ilk iddia kırmızı olur ve atlama kuralı bayatlamış demektir."""
    from nihai.execution.backtest import backtest_symbol

    saatlik = backtest_symbol("SNT", "1h", df=_sentetik("h", 900, 0.002), strategy_mode="quant_trend",
                              disable_kill_switch=True)
    gunluk = backtest_symbol("SNT", "1d", df=_sentetik("B", 700, 0.002), strategy_mode="quant_trend",
                             disable_kill_switch=True)
    assert saatlik.trades_count == 0
    assert gunluk.trades_count > 0


def test_tarayici_olculemeyen_stratejide_backtest_kosmaz(monkeypatch):
    from nihai.execution import auto_screener as A

    _strateji(monkeypatch, "quant_trend")
    monkeypatch.setattr(A, "AUTO_SCREEN_INTERVAL", "1h")
    monkeypatch.setattr(A, "DUAL_TIER_ENABLED", True)
    monkeypatch.setattr(A.watchlist, "current", lambda: ["AAPL", "MSFT"])

    def _tarama(**_):
        raise AssertionError("ölçülemeyen stratejide evren taranmamalı")

    monkeypatch.setattr(A, "scan_universe", _tarama)
    assert A.scan_and_update() == ["AAPL", "MSFT"]

    # Ölçülebilir stratejide tarama çağrılır (atlama kuralı her şeyi susturmuyor).
    cagri: list[int] = []
    _strateji(monkeypatch, "event_breakout")
    monkeypatch.setattr(A, "scan_universe", lambda **_: cagri.append(1) or [])
    A.scan_and_update()
    assert cagri == [1]


# ---------------------------------------------------------------------------
# BULGU-106-5 — REST tohumu düşünce akış sağlayıcısı tampondan kurulan EKSİK seriyi döndürüyordu
# ---------------------------------------------------------------------------
class _Rest:
    name = "rest"

    def __init__(self, bos: bool):
        self.bos = bos

    def get_bars(self, symbol, interval, lookback_bars=200):
        import pandas as pd

        if self.bos:                                   # 429 → alpaca_provider boş döner
            return pd.DataFrame()
        idx = pd.date_range("2020-01-02", periods=lookback_bars, freq="B")
        return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)

    def get_quote(self, symbol):
        return None


def _akis_iki_gunluk_tampon():
    from collections import deque

    import pandas as pd

    from nihai.data.alpaca_stream_provider import AlpacaStreamProvider

    p = AlpacaStreamProvider(symbols=["AAPL"], rest=_Rest(bos=True))
    buf = p._minutes.setdefault("AAPL", deque(maxlen=2000))
    for gun in ("2026-01-05", "2026-01-06"):           # 2 seans (09:30-16:00 ET = 14:30-21:00 UTC)
        for ts in pd.date_range(f"{gun} 14:30", periods=390, freq="1min", tz="UTC"):
            buf.append((ts, 10.0, 10.0, 10.0, 10.0, 100.0))
    return p


def test_tohum_yokken_istegi_karsilayamayan_tampon_cevap_vermez():
    p = _akis_iki_gunluk_tampon()
    assert p.get_bars("AAPL", "1d", 300).empty          # eskisi 2 günlük "seri" döndürürdü
    assert p.get_bars("AAPL", "1h", 200).empty
    assert len(p.get_bars("AAPL", "1m", 220)) == 220     # tampon yeterliyse yine cevap verir


def test_tohum_yokken_yedek_devreye_girer():
    from nihai.data.data_provider import FailoverProvider

    f = FailoverProvider(_akis_iki_gunluk_tampon(), _Rest(bos=False))
    assert len(f.get_bars("AAPL", "1d", 300)) == 300
    assert f.secondary_hits == 1


# ---------------------------------------------------------------------------
# BULGU-106-6 → 106-36 — "özellik logu artmadı": sayım yerel takvim gününe değil AÇILMIŞ son seansa aittir
# (106-6: seans açılmadan/hafta sonu yanlış alarm; 106-36: öğle yedeğinde ve gece yarısını aşan kış seansında suskunluk)
# ---------------------------------------------------------------------------
def test_ozellik_logu_sayimi_ve_uyarisi_son_seansa_aittir(tmp_path, monkeypatch):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from nihai.ai import reasoner
    from nihai.data.market_calendar import ET, son_seans_acilisi_yerel

    ist = ZoneInfo("Europe/Istanbul")

    def acilis_et(dt, **kw):                       # yerel damga → ET duvar saati (makinenin saat diliminden bağımsız iddia)
        return datetime.fromisoformat(son_seans_acilisi_yerel(dt, **kw)).astimezone().astimezone(ET).replace(tzinfo=None)

    # yaz: seans TR 16:30-23:00 = 09:30-16:00 ET
    assert acilis_et(datetime(2026, 10, 1, 23, 30, tzinfo=ist)) == datetime(2026, 10, 1, 9, 30)    # gece koşusu: bugünün seansı
    assert acilis_et(datetime(2026, 10, 1, 12, 0, tzinfo=ist)) == datetime(2026, 9, 30, 9, 30)     # öğle yedeği: dünkü seans
    assert acilis_et(datetime(2026, 10, 3, 23, 30, tzinfo=ist)) == datetime(2026, 10, 2, 9, 30)    # Cumartesi → Cuma
    assert acilis_et(datetime(2026, 10, 5, 12, 0, tzinfo=ist)) == datetime(2026, 10, 2, 9, 30)     # Pazartesi öğlen → Cuma
    # kış (ABD yaz saati bitti, TR sabit): seans TR 17:30-00:00 → kapanış sonrası koşu yerel ERTESİ gündedir
    assert acilis_et(datetime(2026, 12, 2, 0, 5, tzinfo=ist)) == datetime(2026, 12, 1, 9, 30)
    assert acilis_et(datetime(2026, 12, 26, 12, 0, tzinfo=ist)) == datetime(2026, 12, 24, 9, 30)   # Noel + hafta sonu → 24 Aralık
    # yeni açılan seans pay süresince (30 dk) sayılmaz: henüz satır üretmemiş olabilir
    assert acilis_et(datetime(2026, 10, 1, 16, 45, tzinfo=ist)) == datetime(2026, 9, 30, 9, 30)
    assert acilis_et(datetime(2026, 10, 1, 17, 1, tzinfo=ist)) == datetime(2026, 10, 1, 9, 30)
    assert acilis_et(datetime(2026, 10, 1, 16, 45, tzinfo=ist), pay_dk=0) == datetime(2026, 10, 1, 9, 30)
    assert acilis_et(datetime(2026, 10, 1, 16, 0)) == datetime(2026, 10, 1, 9, 30)                 # saat dilimsiz = ET (modül sözleşmesi)

    fl = tmp_path / "live_features_log.csv"
    monkeypatch.setattr(reasoner, "FEATS_LOG", fl)
    monkeypatch.setattr(reasoner, "HISTORY_CSV", tmp_path / "yok.csv")
    monkeypatch.setattr(reasoner, "PRED_LOG", tmp_path / "yok2.csv")

    def damga(dt, dk):                             # son açılıştan `dk` dakika sonrası, defterin biçiminde (yerel, saat dilimsiz)
        return (datetime.fromisoformat(reasoner.seans_esigi(dt)) + timedelta(minutes=dk)).isoformat(timespec="seconds")

    def uyari(facts):
        return [s for s in reasoner._dikkat_lines(facts, None, datetime(2026, 12, 2, 0, 5)) if "özellik logu artmadı" in s]

    kis = datetime(2026, 12, 2, 0, 5, tzinfo=ist)              # kış gece koşusu (kapanış + 5 dk)
    gun = reasoner.seans_esigi(kis)[:10]
    # seans öncesi 2 satır (aynı yerel gün; sayılmaz) + seans içi 2 satır + kapanış barının yerel gece yarısından SONRA
    # yazılan satırı → eşik sayımı 3; yerel gün ÖNEKİYLE saymak 4 (açılış günü) ya da 1 (koşu günü) verirdi
    fl.write_text("logged_at,symbol\n" + "".join(f"{damga(kis, dk)},AAPL\n" for dk in (-120, -60, 10, 380, 391)),
                  encoding="utf-8")
    facts = reasoner.gather_facts(now=kis)
    assert (facts["live_features_today"], facts["live_features_day"]) == (3, gun)     # eskiden: yerel "bugün" → 1
    assert not uyari(facts)
    # aynı seans hiç satır üretmediyse (kayıt servisi kapalı) uyarı KONUŞUR ve seansı adıyla söyler (eskiden: susardı)
    fl.write_text("logged_at,symbol\n" + f"{damga(kis, -60)},AAPL\n", encoding="utf-8")
    bos = reasoner.gather_facts(now=kis)
    assert bos["live_features_today"] == 0 and len(uyari(bos)) == 1 and gun in uyari(bos)[0]
    # öğle yedeği dünkü seansı raporlar: dün boşsa uyarır (eskiden hiç konuşmazdı), dün doluysa "bugün +0" yanlış alarmı yok
    ogle = datetime(2026, 10, 1, 12, 0, tzinfo=ist)
    fl.write_text("logged_at,symbol\n", encoding="utf-8")
    assert len(uyari(reasoner.gather_facts(now=ogle))) == 1
    fl.write_text("logged_at,symbol\n" + f"{damga(ogle, 30)},MSFT\n", encoding="utf-8")
    dolu = reasoner.gather_facts(now=ogle)
    assert dolu["live_features_today"] == 1 and not uyari(dolu)
    # etiket: sayım yerel bugüne aitse "bugün", değilse seans günü; seansı bilinmeyen sayım alarm üretmez
    assert reasoner.seans_etiketi({"today": "2026-10-01", "live_features_day": "2026-10-01"}) == "bugün"
    assert reasoner.seans_etiketi({"today": "2026-12-02", "live_features_day": "2026-12-01"}) == "son seans (2026-12-01)"
    assert reasoner.seans_etiketi({"today": "2026-12-02", "live_features_day": "2026-12-01"}, ek="ta") == "son seansta (2026-12-01)"
    assert "son seans (2026-12-01) canlı özellik +7" in reasoner._durum_line(
        {"today": "2026-12-02", "live_features_day": "2026-12-01", "live_features_today": 7})
    assert uyari({"today": "2026-10-01", "live_features_day": "2026-10-01", "live_features_today": 0}) == [
        "Dikkat: bugün canlı özellik logu artmadı (kayıt servisi durmuş olabilir)."]
    assert not uyari({"live_features_today": 0})
    # takvim okunamazsa yerel güne düşülür (eski davranış: uyarı susmaz)
    import nihai.data.market_calendar as mc

    monkeypatch.setattr(mc, "son_seans_acilisi_yerel", lambda *a, **k: 1 / 0)
    assert reasoner.seans_esigi(datetime(2026, 10, 1, 12, 0)) == "2026-10-01"


# ---------------------------------------------------------------------------
# BULGU-106-7 — pano NLV'si açık K/Z'yi, "nakit"i alım bedelini hiç içermiyordu (ikisi aynı sayıydı)
# ---------------------------------------------------------------------------
def test_status_nlv_piyasa_degeri_ve_nakit_pozisyonlar_sonrasi(monkeypatch):
    from datetime import datetime

    from fastapi.testclient import TestClient

    import nihai.web.web_dashboard as wd
    from nihai.risk.risk_manager import PortfolioState, VirtualPosition

    ps = PortfolioState(equity=20_000.0, ath_equity=25_000.0, cash_balance=20_000.0)
    t = datetime(2026, 10, 1, 10, 0)
    ps.open_positions += [
        VirtualPosition(symbol="AAA", asset_type="stock", side="LONG", quantity=10, entry_price=100.0,
                        entry_time=t, stop=95.0, target=110.0, last_mark=103.0),     # +30, bağlı 1000
        VirtualPosition(symbol="BBB", asset_type="stock", side="SHORT", quantity=5, entry_price=50.0,
                        entry_time=t, stop=52.0, target=45.0, last_mark=51.0),       # −5, hasılat 250
    ]
    monkeypatch.setattr(wd, "get_portfolio_status", lambda: (wd.StateStore(), ps))
    d = TestClient(wd.app).get("/api/status").json()
    assert d["equity"] == 20_000.0                     # kurallar (PDT, boyut) gerçekleşmişle çalışır
    assert d["open_pnl"] == 25.0
    assert d["nlv"] == 20_025.0                        # nakit + pozisyon değeri
    assert d["cash_committed"] == 750.0
    assert d["cash"] == 19_250.0                       # 20.000 − 1.000 (alım) + 250 (açığa satış)
    # özdeşlik: NLV = nakit + uzunların piyasa değeri − kısaların piyasa değeri
    assert d["nlv"] == d["cash"] + 10 * 103.0 - 5 * 51.0


def test_pano_pdt_ve_gunluk_yuzde_gerceklesmis_ozkaynakla():
    kaynak = (KOK / "nihai/web/web_dashboard.html").read_text(encoding="utf-8")
    assert "if (data.nlv >= pdtEsik) {" not in kaynak
    assert "(data.daily_pnl / (data.nlv - data.daily_pnl) * 100)" not in kaynak
    assert "<span>Nakit Rezervi</span>" not in kaynak


# ---------------------------------------------------------------------------
# BULGU-106-8 — gölge kasalar canlı yolun seans giriş kapısını sormuyordu (ilk 5 dk + 15:50 ET sonrası)
# ---------------------------------------------------------------------------
def _kasa_kos(monkeypatch, tmp_path, bas_utc: str, *, stop: float = 99.0, hedef: float = 103.0):
    """`bas_utc`'den başlayan 6 dakikalık bar; aday 2. barda, dolum 3. barın açılışında."""
    import pandas as pd

    from nihai.execution import golge_kasalar as G
    from nihai.signals.strategy import Signal, SignalAction

    idx = pd.date_range(bas_utc, periods=6, freq="1min")
    df = pd.DataFrame({"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "volume": 1e3}, index=idx)

    def aday(symbol, d, shared=None):
        if d.index[-1] != idx[1]:
            return None
        return Signal(SignalAction.LONG, symbol, 100.0, stop, hedef, reasons=["t"],
                      metadata={"strategy": "EventBreakout"})

    monkeypatch.setattr(G, "aday", aday)
    monkeypatch.setattr(G, "olc", lambda symbol, d, sig, shared=None, *, conf_bar=None: G.Olcum())
    m = G.Motor(kok=tmp_path, bakiye=1_000_000.0)
    for k in range(len(df)):
        m.bar_isle("AAA", df.iloc[: k + 1], simdi=(df.index[k] + pd.Timedelta(seconds=61)).to_pydatetime())
    return m


def test_kasa_seans_ortasinda_girer(monkeypatch, tmp_path):
    m = _kasa_kos(monkeypatch, tmp_path, "2026-10-01 14:00")          # 10:00 ET
    assert m.sayac["acilan"] >= 1 and m.sayac["seans_kapisi"] == 0


def test_kasa_mocdan_sonra_ve_acilista_girmez(monkeypatch, tmp_path):
    gec = _kasa_kos(monkeypatch, tmp_path / "gec", "2026-10-01 19:49")  # 15:49 ET → dolum 15:51
    assert gec.sayac["acilan"] == 0 and gec.sayac["seans_kapisi"] >= 1
    assert all(not k.portfoy.open_positions for k in gec.kasalar.values())
    erken = _kasa_kos(monkeypatch, tmp_path / "erken", "2026-10-01 13:30")  # 09:30 ET → dolum 09:32
    assert erken.sayac["acilan"] == 0 and erken.sayac["seans_kapisi"] >= 1


def test_takvim_saat_dilimsiz_zamani_et_sayar():
    """Sistem sözleşmesi: saat dilimsiz = ET (`now_et_naive`). `astimezone` onu makinenin yerel saati sayardı."""
    from datetime import datetime, timedelta

    from nihai.data import market_calendar as mc

    naive = datetime(2026, 10, 1, 15, 58)
    assert mc.is_market_open(naive) is True
    assert mc.can_enter_new_position(naive) == mc.can_enter_new_position(naive.replace(tzinfo=mc.ET))
    assert mc.can_enter_new_position(naive) == (False, "PAST_MOC_CUTOFF")
    assert mc.session_remaining(naive) == timedelta(minutes=2)
    assert mc.can_enter_new_position(datetime(2026, 10, 1, 9, 31)) == (False, "OPENING_VOLATILITY_5MIN")
    assert mc.can_enter_new_position(datetime(2026, 10, 1, 11, 0)) == (True, "OK")


# ---------------------------------------------------------------------------
# BULGU-106-10 — gölge kasalar canlı yolun RR kapılarını (çapa + kaymalı dolum) uygulamıyordu
# ---------------------------------------------------------------------------
def test_kasa_dolumda_rr_kapisi_canli_yolla_ayni(monkeypatch, tmp_path):
    """Ölçülen canlı örüntü: 1m sinyalde risk %0,1, ödül %0,2 (RR 2); %0,1 taban kayma ≈ 1R → dolumda RR 0,5."""
    dar = _kasa_kos(monkeypatch, tmp_path / "dar", "2026-10-01 14:00", stop=99.9, hedef=100.2)
    assert dar.sayac["acilan"] == 0 and dar.sayac["rr_kapisi"] >= 1
    genis = _kasa_kos(monkeypatch, tmp_path / "genis", "2026-10-01 14:00", stop=99.0, hedef=103.0)
    assert genis.sayac["acilan"] >= 1 and genis.sayac["rr_kapisi"] == 0


def test_kasa_rr_kapisi_canli_yolun_fonksiyonunu_kullanir():
    kaynak = (KOK / "nihai/execution/golge_kasalar.py").read_text(encoding="utf-8")
    assert "reanchor_to_live(emir.sig, o, min_rr=min_rr)" in kaynak
    assert "k.pm.slip.apply_entry(o, taraf, \"stock\")" in kaynak


# ---------------------------------------------------------------------------
# BULGU-106-11 — quant'ın günlük durumu her 1h barda yeniden "kabul edilen sinyal" olarak uyarılıyordu
# ---------------------------------------------------------------------------
def test_uyari_anahtari_quantta_gunluk_bar(monkeypatch):
    import pandas as pd

    from nihai.execution import main as M

    gunluk = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-09-30 04:00", "2026-10-01 04:00"]))
    saat_a = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2026-10-01 14:30"]))
    saat_b = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2026-10-01 15:30"]))
    monkeypatch.setattr(M, "_alerted_bars", {})
    assert M._alert_once("MU", M._uyari_bari(saat_a, gunluk)) is True
    assert M._alert_once("MU", M._uyari_bari(saat_b, gunluk)) is False      # aynı günlük durum: tek uyarı
    # TA kipinde (günlük bar yok) anahtar HTF barıdır — eski davranış
    assert M._alert_once("XYZ", M._uyari_bari(saat_a, None)) is True
    assert M._alert_once("XYZ", M._uyari_bari(saat_b, None)) is True


def test_uyari_cagrisi_yardimciyi_kullanir():
    kaynak = (KOK / "nihai/execution/main.py").read_text(encoding="utf-8")
    assert "_alert_once(symbol, str(df_htf.index[-1]))" not in kaynak
    assert "_alert_once(symbol, _uyari_bari(df_htf, df_daily))" in kaynak


# ---------------------------------------------------------------------------
# BULGU-106-12 — backtest/screen CLI'larında bayat strateji listesi; quant 1h'de sessizce 0 işlem
# ---------------------------------------------------------------------------
def test_cli_strateji_secenekleri_tek_kaynaktan():
    for yol in ("nihai/execution/backtest.py", "nihai/execution/screen.py"):
        kaynak = (KOK / yol).read_text(encoding="utf-8")
        assert '"macd", "donchian", "squeeze", "ensemble"]' not in kaynak, yol
        assert "choices=list(STRATEGY_MODES)" in kaynak, yol


def test_cli_quant_gunluk_olmayan_aralikta_acik_hata_verir(monkeypatch, capsys):
    import pytest

    import nihai.execution.backtest as B
    import nihai.execution.screen as S

    _strateji(monkeypatch, "quant_trend")
    for modul, argv in ((B, ["backtest", "--symbol", "AAPL"]), (S, ["screen", "--limit", "1"])):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as e:
            modul.main()
        assert e.value.code == 2
        assert "quant_trend günlük bar ister" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# BULGU-106-13 — yürüyen kalibrasyon raporu ölçtüğü DÖNEMİ söylemiyordu (kaynak kesilince eski dilim "güncel")
# ---------------------------------------------------------------------------
def test_yuruyen_kalibrasyon_raporu_olculen_donemi_yazar(monkeypatch):
    from nihai.ai import adaptive_calibration as A

    deneme = {"method": "platt", "n_eval": 2543, "window": None,
              "eval_range": ["2026-09-22T16:26:00", "2026-09-24T18:32:00"],
              "ham": {}, "adaptif": {}, "taban": {}}
    monkeypatch.setattr(A, "load_artifact", lambda: {"neural": {"durum": "KAPI_GECILEMEDI", "denemeler": [deneme]}})
    assert "neural: KAPI_GECILEMEDI method=platt n_eval=2543 pencere=genişleyen dönem=2026-09-22→2026-09-24" in A.report()


# ---------------------------------------------------------------------------
# BULGU-106-14 — `/metrics` HELP/TYPE'ı etiketli her örnekte tekrarlıyordu; `[rate]` satırı sayaçların yarısını basıyordu
# ---------------------------------------------------------------------------
def test_metrics_her_aile_tek_help_type_ve_toplam_etiketi_yok():
    import nihai.observability as O
    from nihai.data.alpaca_provider import AlpacaProvider

    AlpacaProvider._tick("get_trades")
    O.record_loop("t106_a", 1.0)
    O.record_loop("t106_b", 2.0)
    satirlar = O.prometheus_text().splitlines()
    for tur in ("# HELP ", "# TYPE "):
        adlar = [x.split()[2] for x in satirlar if x.startswith(tur)]
        assert len(adlar) == len(set(adlar)), tur
    # TYPE, ailenin ilk örneğinden ÖNCE ve aile tek blok
    for ad in ("nihai_api_calls_total", "nihai_loop_seconds"):
        sira = [i for i, x in enumerate(satirlar) if x.split("{")[0].split(" ")[0] == ad]
        tip = satirlar.index(next(x for x in satirlar if x.startswith(f"# TYPE {ad} ")))
        assert tip < sira[0] and sira == list(range(sira[0], sira[0] + len(sira)))
    assert any(x.startswith('nihai_api_calls_total{method="get_trades"}') for x in satirlar)
    assert not [x for x in satirlar if 'method="total"' in x]


def test_rate_satiri_tum_sayaclari_basar():
    kaynak = (KOK / "nihai/execution/main.py").read_text(encoding="utf-8")
    assert "bars={_ac['get_bars']} quote={_ac['get_quote']}" not in kaynak
    assert "for k, v in _ac.items() if v and k != 'total'" in kaynak


# ---------------------------------------------------------------------------
# BULGU-106-15 — karar hunisi Tur-89/97/101 guard'larını tanımıyordu (günde ~42 bin deneme sayım dışı)
# ---------------------------------------------------------------------------
def test_huni_quant_ve_dolum_asamalarini_sayar(tmp_path, monkeypatch):
    from nihai.signals import signal_engine as se
    from nihai.signals.strategy import Signal, SignalAction

    monkeypatch.setattr(se, "LOG_DIR", tmp_path)
    monkeypatch.setenv("AUDIT_CHAIN", "off")
    for _ in range(3):
        se.log_block("QUANT_POZISYON_VAR", "AAA: açık quant pozisyonu var (sembol başına tek)")
    se.log_block("QUANT_ODAK_DISI", "BBB: quant yalnız odak katmanında (ölçülen evren)")
    se.log_block("QUANT_AYNI_GUN", "CCC: bugün quant girişi/çıkışı var — yeniden giriş ertesi seans")
    sig = Signal(action=SignalAction.LONG, symbol="DDD", price=100.0, stop=99.0, target=101.5,
                 reasons=["x"], quality_score=0.7, rr_ratio=1.5)
    se.log_signal(sig, accepted=False, extra="FILL_SLIPPAGE")           # canlı yolun ÇİFT yazımı
    se.log_block("FILL_SLIPPAGE", "DDD: dolum 100.1000 — STALE_RR: canlı fiyatta RR=1.07 < 1.20")
    h = se.karar_hunisi()
    el = {a["asama"]: a["elenen"] for a in h["asamalar"]}
    assert h["taninmayan"] == {}
    assert (el["QUANT_POZISYON_VAR"], el["QUANT_ODAK_DISI"], el["QUANT_AYNI_GUN"], el["FILL_SLIPPAGE"]) == (3, 1, 1, 1)
    assert el["STRATEJI"] == 0 and h["toplam_deneme"] == 6


def test_huni_main_yazdigi_her_kodu_tanir():
    """Elle bakılan liste sessizce eskir (Tur-76): `scan_symbol`ün yazdığı her eylem kodu ya bir huni aşaması ya
    bilgi satırı olmalı. Opsiyon dalları sanal hisse kararından SONRA koşar (deneme bitmiştir) → bilinçli dışarıda."""
    import re

    from nihai.signals.signal_engine import _BILGI_SATIRI, _CIFT_YAZILAN, HUNI_ASAMALARI

    kaynak = (KOK / "nihai/execution/main.py").read_text(encoding="utf-8")
    kodlar = set(re.findall(r'log_(?:block|decision)\(\s*"([A-Z_]+)"', kaynak))
    ekstra = set(re.findall(r'accepted=False, extra="([A-Z_]+)"', kaynak))
    disari = {"OPTION_NO_CONTRACT", "OPTION_RISK_REJECT"}
    assert kodlar and kodlar - set(HUNI_ASAMALARI) - _BILGI_SATIRI - disari == set()
    assert ekstra and ekstra <= set(_CIFT_YAZILAN)


def test_pano_hunisi_her_asamayi_adlandirir_ve_taninmayani_gosterir(tmp_path):
    import importlib.util
    import json
    import re

    import pytest

    from nihai.signals.signal_engine import HUNI_ASAMALARI

    html = (KOK / "nihai/web/web_dashboard.html").read_text(encoding="utf-8")
    govde = html[html.index("function renderDecisionFunnel"):html.index("async function loadDecisionFunnel")]
    etiketli = set(re.findall(r"([A-Z_]+):\s*\"", govde[govde.index("const labels"):govde.index("const fmt")]))
    assert set(HUNI_ASAMALARI) <= etiketli
    # Gerçek çizim (Tur-77 node düzeneği): eşlenmemiş kod sayıma girmiyorsa bunu kart SÖYLEMELİ.
    spec = importlib.util.spec_from_file_location("t77pano", KOK / "tests" / "test_tur77_pano_modul.py")
    t77 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t77)
    if t77.NODE is None:
        pytest.skip("node kurulu değil — JS ölçülemedi")
    ornek = {"ok": True, "dosya": "d.txt", "toplam_deneme": 50, "kabul": 0, "uyari": 0, "pozisyon_acilis": 0,
             "n_sembol": 3, "taninmayan": {"OPTION_NO_CONTRACT": 7}, "sembol_en_ileri": {},
             "asamalar": [{"asama": "QUANT_POZISYON_VAR", "elenen": 50, "kalan": 0, "alt_neden": {}}]}
    yol = tmp_path / "ornek.json"
    yol.write_text(json.dumps(ornek), encoding="utf-8")
    r = t77._kos(tmp_path, ornek=yol)
    assert "Quant: açık pozisyon" in r["huni"]
    assert "Huniye eşlenmemiş kod" in r["huni"] and "OPTION_NO_CONTRACT 7" in r["huni"]


# ---------------------------------------------------------------------------
# BULGU-106-17 — canlı quant ve gölge koşucu oluşmakta olan BUGÜN günlük barıyla karar veriyordu
# ---------------------------------------------------------------------------
def _gunluk_df(son_gun: str, n: int = 300):
    import numpy as np
    import pandas as pd

    gunler = pd.bdate_range(end=son_gun, periods=n) + pd.Timedelta(hours=4)     # Alpaca etiketi 04:00 UTC
    kapanis = np.linspace(100.0, 130.0, n)
    return pd.DataFrame({"open": kapanis, "high": kapanis, "low": kapanis, "close": kapanis,
                         "volume": 1e6}, index=gunler)


def test_kapanmis_gunluk_yalniz_seans_acikken_bugunu_atar():
    from datetime import datetime

    from nihai.data.market_calendar import kapanmis_gunluk

    df = _gunluk_df("2026-09-30")
    acik, kapali, sabah = datetime(2026, 9, 30, 11, 0), datetime(2026, 9, 30, 17, 0), datetime(2026, 9, 30, 8, 0)
    assert str(kapanmis_gunluk(df, acik).index[-1].date()) == "2026-09-29"       # seans açık → yarım bar atılır
    assert len(kapanmis_gunluk(df, kapali)) == len(df)                           # kapanış sonrası bar kapanmıştır
    assert len(kapanmis_gunluk(df, sabah)) == len(df)                            # seans öncesi (bugün barı yok)
    dun = _gunluk_df("2026-09-29")
    assert len(kapanmis_gunluk(dun, acik)) == len(dun)                           # sağlayıcı bugünü vermediyse dokunma


def test_canli_quant_ve_golge_kosucu_kapanmis_gunleri_okur(monkeypatch):
    from datetime import datetime

    import nihai.ai.shadow_runner as SR
    import nihai.execution.main as M
    from nihai.data import market_calendar as MC

    simdi = datetime(2026, 9, 30, 11, 0, tzinfo=MC.ET)
    monkeypatch.setattr(MC, "now_et", lambda: simdi)
    monkeypatch.setattr(M, "now_et_naive", lambda: simdi.replace(tzinfo=None))
    monkeypatch.setattr(M, "_GUNLUK_ONBELLEK", {})

    class P:
        def get_bars(self, symbol, interval, lookback_bars=200):
            return _gunluk_df("2026-09-30")

    assert str(M._gunluk_barlar("AAA", P()).index[-1].date()) == "2026-09-29"
    assert str(SR._fetch_bars(P(), "AAA").index[-1].date()) == "2026-09-29"


# ---------------------------------------------------------------------------
# BULGU-106-18 — "bekleyen birikimi artıyor" kapanış kuyruğunu (ertesi seansı bekleyen) birikim sanıyordu
# ---------------------------------------------------------------------------
def test_gecikmis_bekleyen_yalniz_cozulmesi_gerekeni_sayar():
    from datetime import UTC, datetime

    from nihai.ai.reasoner import _gecikmis

    kapanis = "2026-10-02 19:57:00"                  # 15:57 ET, ufuk 5 dk → kapanışı aşar, ertesi seansı bekler
    assert not _gecikmis(kapanis, "5", "1m", datetime(2026, 10, 2, 20, 30, tzinfo=UTC))   # 23:30 TR gece koşusu
    assert not _gecikmis(kapanis, "5", "1m", datetime(2026, 10, 3, 9, 0, tzinfo=UTC))     # 12:00 TR, hafta sonu
    assert _gecikmis(kapanis, "5", "1m", datetime(2026, 10, 5, 14, 10, tzinfo=UTC))       # Pazartesi açılış + 40 dk
    gun_ici = "2026-10-02 15:00:00"
    assert not _gecikmis(gun_ici, "5", "1m", datetime(2026, 10, 2, 15, 20, tzinfo=UTC))   # 30 dk payı içinde
    assert _gecikmis(gun_ici, "5", "1m", datetime(2026, 10, 2, 15, 40, tzinfo=UTC))
    assert not _gecikmis("bozuk", "5", "1m", datetime(2026, 10, 5, 14, 10, tzinfo=UTC))    # ölçülemeyen → alarm yok


def test_birikim_uyarisi_kapanis_kuyrugunda_susar():
    from datetime import datetime

    from nihai.ai.reasoner import _dikkat_lines

    simdi = datetime(2026, 10, 2, 23, 30)
    kuyruk = {"pred_pending_recent": 119, "pred_overdue_recent": 0, "pred_tail_n": 500}
    assert not any("bekleyen" in u for u in _dikkat_lines(kuyruk, {"pred_pending_recent": 90}, simdi))
    takili = {**kuyruk, "pred_overdue_recent": 37}
    assert any("çözülmüş olması gereken 37" in u for u in _dikkat_lines(takili, None, simdi))


def test_lab_strateji_formu_quantin_parametreleri_kullanmadigini_soyler():
    html = (KOK / "nihai/web/lab_dashboard.html").read_text(encoding="utf-8")
    form = html[html.index('id="lab-strategy"'):html.index('onclick="labReadGuards()"')]
    assert "quant_trend</code> üçünü de kullanmaz" in form


def test_forward_hukmu_beraberlik_haric_orani_adlandirir():
    from research.forward_harness import _verdict

    haric = _verdict({"resolved": 160000, "days_span": 99, "hit_rate": 0.481, "hit_rate_excl_ties": 0.486,
                      "tie_share": 0.01})["text"]
    assert "48.6% beraberlik hariç" in haric
    assert "beraberlik" not in _verdict({"resolved": 160000, "days_span": 99, "hit_rate": 0.481})["text"]


def test_gate_hurts_onerisi_golge_kipiyle_celismez():
    """GATE_HURTS'ün belgelenmiş cevabı `META_GATE=shadow`dur; kapıyı kapatmak kanıt döngüsünü öldürür."""
    from nihai.execution import prediction_tracker as pt

    def satir(pwin, ret):
        return {"pwin": str(pwin), "pwin_side": "LONG", "direction": "+", "fwd_ret_pct": str(ret)}

    gs = pt.gate_sim([satir(0.60, -1.0) for _ in range(5)] + [satir(0.30, +1.0) for _ in range(5)],
                     threshold=0.45, min_side=3)
    assert gs["label"] == "GATE_HURTS"
    assert "kapatmayı" not in gs["text"] and "META_GATE=shadow" in gs["text"]


# ---------------------------------------------------------------------------
# BULGU-106-19 — başlatıcılar yeniden başlatmada stderr günlüğünü sıfırlıyordu (kanıt kaybı)
# ---------------------------------------------------------------------------
_BASLATICILAR = (("hub-watchdog.ps1", "$hubErr"), ("predict-watchdog.ps1", "$svcErr"), ("proje-ac.ps1", "$errLog"))


def test_baslaticilar_eski_stderr_gunlugunu_yeni_surecten_once_saklar():
    for ad, var in _BASLATICILAR:
        metin = (KOK / ad).read_text(encoding="utf-8-sig")
        tasima = metin.index(f"Move-Item -LiteralPath {var} -Destination ({var} -replace '\\.log$', '.prev.log')")
        baslat = metin.index(f"-RedirectStandardError {var}")
        assert tasima < baslat, ad
        # taşıma, yönlendirmeyi kuran AYNI Start-Process çağrısından hemen önce (araya başka başlatma girmez)
        assert metin.count("Start-Process", tasima, baslat) == 1, ad


def test_stderr_saklama_satiri_powershellde_gercekten_tasir(tmp_path):
    import shutil

    import pytest

    ps = shutil.which("powershell") or shutil.which("pwsh")
    if ps is None:
        pytest.skip("PowerShell yok")
    metin = (KOK / "hub-watchdog.ps1").read_text(encoding="utf-8-sig")
    satir = next(s.strip() for s in metin.splitlines() if "Move-Item -LiteralPath $hubErr" in s)
    err = tmp_path / "hub_server.err.log"
    err.write_text("asili surecin son satiri", encoding="ascii")
    bos = tmp_path / "bos.err.log"
    bos.write_bytes(b"")
    betik = f"$hubErr = '{err}'; {satir}; $hubErr = '{bos}'; {satir}"
    subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command", betik], check=True, timeout=60)
    assert not err.exists()
    assert (tmp_path / "hub_server.err.prev.log").read_text(encoding="ascii") == "asili surecin son satiri"
    assert bos.exists() and not (tmp_path / "bos.err.prev.log").exists()   # boş günlük öncekini EZMEZ


# ---------------------------------------------------------------------------
# BULGU-106-20 — resolve() seans dışında ertesi açılışı bekleyen satırlar için de bar çekiyordu
# ---------------------------------------------------------------------------
def test_ilk_bar_ani_seans_ici_ertesi_dakika_seans_disi_sonraki_acilis():
    from datetime import UTC, datetime

    import pandas as pd

    from nihai.data.market_calendar import ET, ilk_bar_ani

    cuma_seans = datetime(2026, 10, 2, 15, 0, tzinfo=ET)
    assert ilk_bar_ani(cuma_seans) == pd.Timestamp("2026-10-02 19:01", tz="UTC")
    kapanis_sonrasi = datetime(2026, 10, 2, 20, 3, tzinfo=UTC)          # 16:03 ET
    assert ilk_bar_ani(kapanis_sonrasi) == pd.Timestamp("2026-10-05 13:30", tz="UTC")   # Pzt açılış
    assert ilk_bar_ani(datetime(2026, 10, 2, 15, 0)) == ilk_bar_ani(cuma_seans)   # saat dilimsiz = ET


def test_resolve_ertesi_acilisi_bekleyen_satir_icin_bar_cekmez(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    import pandas as pd

    from nihai.execution import prediction_tracker as pt

    monkeypatch.setattr(pt, "PRED_LOG", tmp_path / "predictions_log.csv")
    monkeypatch.setattr(pt, "FEATS_LOG", tmp_path / "live_features_log.csv")
    # Cuma 15:58 ET (19:58 UTC) + 5 dk → hedef kapanıştan sonra → Pazartesi ilk barını bekler
    satir = {"symbol": "AAPL", "action": "WAIT", "direction": "+", "bar_ts": "2026-10-02 19:58:00",
             "confidence": 0.6, "prob_cal": 0.52, "price": 100.0}
    assert pt.record([satir], "1m", horizon=5) == 1
    an = {"t": datetime(2026, 10, 3, 12, 0, tzinfo=UTC)}

    class Saat(datetime):
        @classmethod
        def now(cls, tz=None):
            return an["t"] if tz is not None else an["t"].replace(tzinfo=None)

    monkeypatch.setattr(pt, "datetime", Saat)
    cagri: list = []
    idx = pd.date_range("2026-10-02 19:55", periods=5, freq="1min").append(
        pd.date_range("2026-10-05 13:30", periods=3, freq="1min"))

    class Saglayici:
        def get_bars(self, symbol, interval, lookback_bars=400):
            cagri.append((symbol, interval))
            return pd.DataFrame({"open": 101.0, "high": 101.0, "low": 101.0, "close": 101.0}, index=idx)

    sonuc = pt.resolve(Saglayici())
    assert cagri == [] and sonuc["pending"] == 1 and sonuc["resolved"] == 0   # Cumartesi: ağ yok
    an["t"] = datetime(2026, 10, 5, 13, 31, tzinfo=UTC)                         # Pzt açılış + 1 dk
    sonuc = pt.resolve(Saglayici())
    assert cagri == [("AAPL", "1m")] and sonuc["resolved"] == 1


# ---------------------------------------------------------------------------
# BULGU-106-21 — README "Madde Eşlemesi" silinmiş/hiç olmamış tanımları anıyordu (margin_ok, /api/reset-breaker)
# ---------------------------------------------------------------------------
def test_readme_madde_tablosu_yalniz_var_olan_tanimlari_anar():
    import re

    from nihai.config import CONFIG

    metin = (KOK / "README.md").read_text(encoding="utf-8")
    tablo = metin[metin.index("## Madde Eşlemesi"):metin.index("## Dizin Yapısı")]
    moduller = {p.stem: p for p in (KOK / "nihai").rglob("*.py")}

    def tanimlar(p: Path) -> dict[str, set[str]]:
        d: dict[str, set[str]] = {}
        for n in ast.parse(p.read_text(encoding="utf-8")).body:
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef):
                d[n.name] = set()
            elif isinstance(n, ast.ClassDef):
                d[n.name] = ({m.name for m in n.body if isinstance(m, ast.FunctionDef | ast.AsyncFunctionDef)}
                             | {m.target.id for m in n.body
                                if isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name)})
            elif isinstance(n, ast.Assign):
                d.update({t.id: set() for t in n.targets if isinstance(t, ast.Name)})
        return d

    siniflar = {ad: uye for p in moduller.values() for ad, uye in tanimlar(p).items() if uye}
    eksik = []
    for kod in re.findall(r"`([^`]+)`", tablo):
        for parca in re.split(r"\s*(?:/|·|\+)\s*", kod):
            parca = parca.split("(")[0].strip()
            if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", parca) or parca.endswith((".py", ".txt", ".csv")):
                continue
            bas, ad, *alt = parca.split(".")
            if bas == "CONFIG":
                nesne: object = CONFIG
                ok = True
                for a in (ad, *alt):
                    ok = hasattr(nesne, a)
                    if not ok:
                        break
                    nesne = getattr(nesne, a)
            elif bas in moduller:
                t = tanimlar(moduller[bas])
                ok = (ad in t and (not alt or alt[0] in t[ad])) or (not alt and any(ad in u for u in t.values()))
            else:
                ok = bas in siniflar and ad in siniflar[bas]
            if not ok:
                eksik.append(parca)
    assert not eksik, eksik
    assert "/api/reset-breaker" not in tablo   # böyle bir uç yok; devre kesici Telegram /dur · /devam ile


# ---------------------------------------------------------------------------
# BULGU-106-22 — kanonik gölge görev betiği (tur17) hafta sonu da kuruyordu; tur20'nin dersi taşınmamıştı
# ---------------------------------------------------------------------------
def test_golge_gorev_betigi_yalniz_hafta_ici_ve_tek_kanonik():
    tur17 = (KOK / "tur17-shadow-gorev.ps1").read_text(encoding="utf-8-sig")
    assert "New-ScheduledTaskTrigger -Daily" not in tur17
    assert "-DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $At" in tur17
    assert "market_calendar --at" in tur17                       # saat hâlâ seanstan türetiliyor (Tur-58)
    assert "09:30'dan itibaren 7 saat" not in tur17               # bayat kullanım satırı
    tur20 = (KOK / "tur20-shadow-gorev.ps1").read_text(encoding="utf-8-sig")
    assert tur20.lstrip("<#\r\n").startswith("ESKI (Tur-106): kanonik betik tur17-shadow-gorev.ps1")


# ---------------------------------------------------------------------------
# BULGU-106-23 — Laboratuvar hub düğmeleri kanonik hub yerine MainHub kopyasını açıyordu
# ---------------------------------------------------------------------------
def test_lab_hub_baslat_kanonik_hubu_bu_depodan_acar(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from nihai.web import api_operations as ops

    kok = tmp_path / "depo"
    piton = kok / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    piton.parent.mkdir(parents=True)
    piton.write_text("", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "hub.web-stopped").write_text("Panelden durduruldu\n", encoding="utf-8")
    monkeypatch.setattr(ops, "ROOT", kok)
    monkeypatch.setattr(ops, "MAINHUB", tmp_path / "MainHub")
    monkeypatch.setattr(ops, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(ops, "HUB_URL", "http://127.0.0.1:8799")
    saglik = iter([RuntimeError("kapalı"), {"pid": 4242, "service_root": str(kok)}])

    def istek(_yol):
        cevap = next(saglik)
        if isinstance(cevap, Exception):
            raise cevap
        return cevap

    acilan: list = []
    monkeypatch.setattr(ops, "hub_request", istek)
    monkeypatch.setattr(ops.subprocess, "Popen", lambda argv, **k: acilan.append((argv, k["cwd"]))
                        or SimpleNamespace(poll=lambda: None))
    assert ops._start_hub()["health"]["pid"] == 4242
    assert acilan == [([str(piton), "-m", "nihai.data.hub_server"], kok)]       # gözcü/proje-ac ile AYNI süreç
    assert not (tmp_path / "logs" / "hub.web-stopped").exists()


def test_proje_ac_panel_durdurma_bayragini_kaldirir_ve_panel_mainhub_demez():
    pa = (KOK / "proje-ac.ps1").read_text(encoding="utf-8")
    assert "Remove-Item -LiteralPath (Join-Path $logDir 'hub.web-stopped')" in pa
    assert "MainHub işlemi" not in (KOK / "nihai/web/lab_dashboard.js").read_text(encoding="utf-8")
    assert "MainHub süreç ve akış" not in (KOK / "nihai/web/lab_dashboard.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# BULGU-106-24 — ana süreçte tek-kopya kilidi yoktu (Tur-80 çift kayıtçı olayı yapısal olarak açıktı)
# ---------------------------------------------------------------------------
def test_ana_surec_ikinci_kopyayi_web_sunucusu_acmadan_reddeder(tmp_path, monkeypatch):
    import time

    import pytest

    from nihai.execution import main as M

    kod = ("import sys,time; from pathlib import Path; from nihai.atomic_io import tek_yazar_kilidi; "
           "f = tek_yazar_kilidi(Path(sys.argv[1]) / 'main.lock'); assert f; print('TUTUYOR', flush=True); "
           "time.sleep(30)")
    cocuk = subprocess.Popen([sys.executable, "-c", kod, str(tmp_path)], cwd=KOK, stdout=subprocess.PIPE, text=True,
                             env={**os.environ, "PYTHONPATH": str(KOK)})
    try:
        assert cocuk.stdout is not None and cocuk.stdout.readline().strip() == "TUTUYOR"
        monkeypatch.setattr(M, "DATA_DIR", tmp_path)
        monkeypatch.setattr(M, "_ANA_KILIT", None)
        monkeypatch.setattr(M, "validate_config", lambda: ([], []))
        monkeypatch.setattr(M, "run_loop", lambda **k: pytest.fail("ikinci kopya tarama döngüsüne girdi"))
        monkeypatch.setattr(sys, "argv", ["main", "--once"])
        with pytest.raises(SystemExit, match="main.lock"):
            M.main()
    finally:
        cocuk.kill()
        cocuk.wait()
    time.sleep(0.2)
    assert M._tek_ana_surec() is True                     # kilit sahibi ölünce işletim sistemi bırakır


def test_canli_ozellik_logu_cand_side_sutunu_tipli_okunur(tmp_path):
    """Gece görevi her koşuda DtypeWarning basıyordu: 28 Eylül'den beri boş olan `cand_side` parça parça float/str okunuyordu."""
    import warnings

    import pandas as pd

    from nihai.signals.meta_features import FEATURE_NAMES
    from research import build_live_dataset as B

    yol = tmp_path / "live_features_log.csv"
    n = 300_000                                             # pandas'ın parça sınırını aşan dosya: ilk yarı dolu, kalan boş
    pd.DataFrame({"logged_at": "2026-09-20T15:00:00", "symbol": "AAPL", "bar_ts": [f"t{i}" for i in range(n)],
                  "interval": "1m", **{k: 0.0 for k in FEATURE_NAMES},
                  "cand_side": ["LONG"] * (n // 2) + [""] * (n - n // 2)}).to_csv(yol, index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.DtypeWarning)
        B.build_unlabeled(feats_path=yol, pred_path=tmp_path / "yok.csv")


# ---------------------------------------------------------------------------
# BULGU-106-25 — 106-7 NLV düzeltmesi kardeş yüzeylere taşınmamıştı (Telegram /status, terminal panosu, uyarılar)
# ---------------------------------------------------------------------------
def _acik_pozisyonlu_portfoy():
    from datetime import datetime

    from nihai.risk.risk_manager import PortfolioState, VirtualPosition

    ps = PortfolioState(equity=20_000.0, ath_equity=25_000.0, cash_balance=20_000.0)
    t = datetime(2026, 10, 1, 10, 0)
    ps.open_positions += [
        VirtualPosition(symbol="AAA", asset_type="stock", side="LONG", quantity=10, entry_price=100.0,
                        entry_time=t, stop=95.0, target=110.0, last_mark=103.0),     # +30
        VirtualPosition(symbol="BBB", asset_type="stock", side="SHORT", quantity=5, entry_price=50.0,
                        entry_time=t, stop=52.0, target=45.0, last_mark=51.0),       # −5
    ]
    return ps


def test_terminal_panosu_nlv_acik_kz_dahil_pano_ile_ayni_sayi(monkeypatch):
    from rich.console import Console

    from nihai.notifier import terminal_dashboard as td

    ps = _acik_pozisyonlu_portfoy()
    assert ps.open_pnl == 25.0
    monkeypatch.setattr(td, "_ai_strip", lambda: "")
    konsol = Console(record=True, width=200)
    konsol.print(td._flat_panel(ps))
    metin = konsol.export_text()
    assert "20,025.00 USD" in metin                     # pano /api/status → nlv 20.025 ile aynı sayı
    assert "gerceklesmis 20,000.00" in metin            # kuralların baktığı özkaynak ayrıca görünür
    assert "40,000.00 USD" in metin                     # alım gücü gerçekleşmiş özkaynaktan (kural tarafı değişmedi)


def test_telegram_ve_uyarilar_nlv_etiketiyle_gerceklesmis_ozkaynak_basmaz():
    kaynak = (KOK / "nihai/execution/main.py").read_text(encoding="utf-8")
    assert kaynak.count("NLV ${portfolio.equity + portfolio.open_pnl:,.0f}") == 4     # /status, açılış, kapanış, heartbeat
    assert "NLV ${portfolio.equity:,.0f}" not in kaynak
    pano = (KOK / "nihai/web/web_dashboard.py").read_text(encoding="utf-8")
    assert "acik_kz = portfolio.open_pnl" in pano       # tek tanım: pano da aynı özelliği okur


# ---------------------------------------------------------------------------
# BULGU-106-26 — Emir Akışı: kapalı seansta çapalı pencere her yenilemede yeniden tam çekiliyordu;
#                canlı yenileme ilk geçici hatada kalıcı duruyordu
# ---------------------------------------------------------------------------
def test_emir_akisi_kapali_seansta_capali_pencereyi_yeniden_cekmez():
    from datetime import UTC, datetime, timedelta

    import numpy as np

    from nihai.data import orderflow as OF

    OF._HISSE_DEPO.clear()
    OF._CAPA.clear()
    kapanis = [datetime(2026, 10, 2, 20, 0, tzinfo=UTC)]            # cuma son işlem
    cagrilar = []

    def cek(symbol, bas, son, feed, limit):
        cagrilar.append((bas, son, limit))
        if bas > kapanis[0]:                                        # son işlemden sonraki pencere: boş
            return OF._bos()
        ust = min(son, kapanis[0])
        if limit == 1:
            return OF.Islemler(np.array([int(ust.timestamp() * 1000) - 1]), np.array([100.0]), np.array([10.0]))
        t = np.arange(int(bas.timestamp() * 1000), int(ust.timestamp() * 1000), 1000, dtype=np.int64)
        return OF.Islemler(t, 100 + (t % 7) / 100, np.full(t.size, 10.0))

    simdi = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)                # cumartesi
    try:
        ilk = OF._hisse_akisi("AAA", 60, 60, "iex", 0.0, 50, simdi, cek)
        assert ilk["ok"] and len(cagrilar) == 3                      # boş yoklama + çapa + çapalı pencere
        assert ilk["yenile_sn"] == 60 and any("çapalandı" in u for u in ilk["uyari"])
        for k in (1, 2, 3):
            n0 = len(cagrilar)
            d = OF._hisse_akisi("AAA", 60, 60, "iex", 0.0, 50, simdi + timedelta(seconds=60 * k), cek)
            yeni = cagrilar[n0:]
            assert len(yeni) == 1 and yeni[0][0] > kapanis[0], yeni  # yalnız "seans açıldı mı?" yoklaması
            assert d["n_islem"] == ilk["n_islem"] and d["son"] == ilk["son"]
        # seans açılınca (canlı pencerede işlem var) depo yenilenir ve canlı hızına dönülür
        kapanis[0] = simdi + timedelta(hours=3)
        acik = OF._hisse_akisi("AAA", 60, 60, "iex", 0.0, 50, simdi + timedelta(hours=2), cek)
        assert acik["ok"] and acik["son"] > ilk["son"] and acik["uyari"] == [] and acik["yenile_sn"] == 5
        OF._HISSE_DEPO.clear()
        assert OF._hisse_akisi("AAA", 60, 60, "sip", 0.0, 50, simdi + timedelta(hours=2), cek)["yenile_sn"] == 30
    finally:
        OF._HISSE_DEPO.clear()
        OF._CAPA.clear()


_OF_HATA_JS = r"""
const fs = require("fs"); const vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8"), veri = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const blok = /<script>\s*\/\/ ===== \[pano modülü m11-emir-akisi\][\s\S]*?<\/script>/.exec(html)[0].replace(/<\/?script>/g, "");
const ctx = new Proxy({}, { get(t, k) { return k in t ? t[k] : () => {}; }, set(t, k, v) { t[k] = v; return true; } });
const el = () => ({ checked: true, value: "50", style: {}, clientWidth: 1400, clientHeight: 660, width: 0, height: 0,
  offsetWidth: 100, offsetHeight: 40, textContent: "", innerHTML: "", classList: { contains: () => true },
  getContext: () => ctx, addEventListener() {}, getBoundingClientRect: () => ({ left: 0, top: 0 }) });
const dom = {};
const document = { getElementById: (id) => (dom[id] = dom[id] || el()), createElement: () => el(), querySelector: () => null };
const yanit = [], zam = [];
async function apiCall(yol) { return yol.startsWith("orderflow/gex") ? { ok: false, reason: "yok" } : yanit.shift(); }
const sb = { document, console, Math, JSON, Number, String, Date, Array, Object, isFinite, URLSearchParams, Promise, apiCall,
             setTimeout: (f, ms) => { zam.push(ms); return zam.length; }, clearTimeout() {} };
sb.window = sb; sb.__PANO_MODULLERI = []; sb.devicePixelRatio = 1; sb.addEventListener = () => {};
vm.createContext(sb); vm.runInContext(blok, sb);
Object.assign(sb, { yanit, zam, veri, process });
vm.runInContext(`(async () => {
  const o = {}; const al = () => { const z = zam.slice(); zam.length = 0; return z; };
  document.getElementById('of-sym').value = 'AAA';
  yanit.push(veri); await ofYukle(); o.ilk = al();
  yanit.push({ ok: false, reason: 'Alpaca hız sınırı (429)' }); await ofYukle(true); o.sessizHata = al();
  yanit.push({ ok: false, reason: 'ısınıyor', yeniden_dene_sn: 3 }); await ofYukle(true); o.isiniyor = al();
  yanit.push({ ok: false, reason: 'Geçersiz sembol' }); await ofYukle(); o.elleHata = al();
  console.log(JSON.stringify(o));
})().catch((e) => { console.error(e && e.stack || e); process.exitCode = 3; });`, sb);
"""


def test_emir_akisi_canli_yenileme_gecici_hatada_durmaz(tmp_path):
    import json

    import pytest

    import tests.test_tur104_gex_ve_buyuk_islem as t104
    from nihai.data import orderflow as OF
    from tests.test_tur103_emir_akisi import _rastgele

    if t104.NODE is None:
        pytest.skip("node kurulu değil — JS ölçülemedi")
    t, p, q, s = _rastgele(n=600)
    v = OF._paketle(t, p, q, s, 60, 0.0, 0.01, 20)
    v.update({"ok": True, "symbol": "AAA", "feed": "iex", "piyasa": "hisse", "kaynak": "test", "taraf_yontemi": "tick",
              "bas": int(t[0]), "son": int(t[-1]), "uyari": [], "yenile_sn": 5, "isi": None, "isi_neden": "yok"})
    (tmp_path / "v.json").write_text(json.dumps(v), encoding="utf-8")
    (tmp_path / "c.js").write_text(_OF_HATA_JS, encoding="utf-8")
    (tmp_path / "p.html").write_text(t104.HTML, encoding="utf-8")
    r = subprocess.run([t104.NODE, str(tmp_path / "c.js"), str(tmp_path / "p.html"), str(tmp_path / "v.json")],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[:800]
    o = json.loads(r.stdout)
    assert o["ilk"] == [5000]                      # başarılı yükleme kendi hızında zamanlar
    assert o["sessizHata"] == [30000]              # otomatik yenilemede geçici hata: 30 sn sonra yeniden dener
    assert o["isiniyor"] == [3000]                 # sunucunun verdiği yeniden deneme süresi önceliklidir
    assert o["elleHata"] == []                     # elle girilen bozuk istek döngüye girmez


# ---------------------------------------------------------------------------
# BULGU-106-27 — Hisse Detay (/api/analyze) quant kipinde canlı taramanın kararını göstermiyordu
# ---------------------------------------------------------------------------
class _DetaySaglayici:
    name = "sahte"

    def __init__(self, saatlik, gunluk):
        self.df = {"1h": saatlik, "1d": gunluk}
        self.cagri = []

    def get_bars(self, sym, interval, lookback_bars=200):
        self.cagri.append((interval, lookback_bars))
        return self.df[interval].tail(lookback_bars)

    def get_quote(self, sym):
        return None


def _detay_duzenegi(monkeypatch, tmp_path, strateji, *, bugun_cokus=False):
    import numpy as np
    import pandas as pd

    from nihai import runtime_strategy
    from nihai.ai import decision
    from nihai.data.market_calendar import now_et_naive
    from nihai.execution import live_predict_service as lps
    from nihai.signals import strategy
    from nihai.web import web_dashboard as wd

    # Tek başına koşunca conftest'in VIX yalıtımı modülü henüz göremez (gövdede import) → burada sabitle.
    monkeypatch.setattr(strategy, "_VIX_LOADED", True, raising=False)
    rng = np.random.default_rng(5)
    g = wd._gelecek_barlar(pd.Timestamp("2026-06-01 13:30"), "1h", 600)
    kap = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, len(g))))
    saatlik = pd.DataFrame({"open": kap, "high": kap * 1.002, "low": kap * 0.998, "close": kap, "volume": 1e5},
                           index=pd.DatetimeIndex([pd.Timestamp(t, unit="s") for t in g]))
    gun = pd.date_range(end=pd.Timestamp(now_et_naive().date()), periods=321, freq="D")
    kg = 50.0 * np.exp(np.cumsum(np.full(321, 0.002) + rng.normal(0, 0.006, 321)))      # belirgin yükseliş: skor +1
    if bugun_cokus:
        kg[-1] = kg[-2] * 0.3                           # oluşmakta olan BUGÜN barı: üç ufku da eksiye çevirir
    gunluk = pd.DataFrame({"open": kg, "high": kg * 1.01, "low": kg * 0.99, "close": kg, "volume": 1e6}, index=gun)
    s = _DetaySaglayici(saatlik, gunluk)
    monkeypatch.setattr(wd, "_analysis_provider", lambda: s)
    monkeypatch.setattr(wd, "is_market_open", lambda: False)
    monkeypatch.setattr(lps, "_seans_acilisi", lambda *a, **k: None)
    monkeypatch.setattr(runtime_strategy, "apply", lambda: strateji)
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    monkeypatch.setattr(decision, "DECISIONS_FILE", tmp_path / "ai_decisions.jsonl")
    return wd, s


def test_detay_quant_kipinde_gun_ici_aralikta_canli_karari_gosterir(monkeypatch, tmp_path):
    from nihai.execution import ai_advisor
    from nihai.execution import main as M
    from nihai.signals import meta_gate, quant

    class _Kapi:                                              # model yüklüymüş gibi: quant sinyaline SORULMAMALI
        soru = 0

        def probability(self, *a, **k):
            _Kapi.soru += 1
            return 0.0

        explain = probability

    monkeypatch.setattr(meta_gate, "get_gate", lambda: _Kapi())
    wd, s = _detay_duzenegi(monkeypatch, tmp_path, "quant_trend")
    d = wd.analyze_symbol("AAA", "1h")
    sig = d["signal"]
    assert d["ok"] and sig["action"] == "LONG", sig
    assert any(r.startswith("QUANT_SKOR_+") for r in sig["reasons"]) and "QUANT_GUNLUK_BAR_GEREKLI" not in sig["reasons"]
    assert ("1d", quant.CANLI_GUNLUK_BAR) in s.cagri          # canlı taramayla AYNI pencere
    assert M._QUANT_GUNLUK_BAR == quant.CANLI_GUNLUK_BAR      # iki yüzey tek sabiti okur
    assert sig["pwin"] is None and sig["meta_explain"] is None and _Kapi.soru == 0   # meta kapı quant'ta kapsam dışı
    assert d["quant"]["action"] == ai_advisor.KAPSAM_DISI
    assert sig["hedef_nominal"] is True and sig["target"] is not None   # quant sabit hedefle çıkmaz (106-28)
    plan = d["order_plan"]
    assert plan["hedef_nominal"] is True and plan["bracket"]["take_profit"] is None and plan["bracket"]["stop_loss"]


def test_detay_quant_karari_olusmakta_olan_bugun_barini_okumaz(monkeypatch, tmp_path):
    from nihai.data import market_calendar

    wd, _ = _detay_duzenegi(monkeypatch, tmp_path, "quant_trend", bugun_cokus=True)
    monkeypatch.setattr(market_calendar, "is_market_open", lambda dt=None: True)     # seans açık: bugün barı yarım
    assert wd.analyze_symbol("AAA", "1h")["signal"]["action"] == "LONG"             # karar T−1 kapanışında (106-17)
    monkeypatch.setattr(market_calendar, "is_market_open", lambda dt=None: False)    # kapanıştan sonra bar kapanmıştır
    assert wd.analyze_symbol("AAA", "1h")["signal"]["action"] == "WAIT"


def test_detay_gunluk_aralikta_quant_tam_pencereyi_kullanir_ve_ta_kipinde_ek_cekim_yok(monkeypatch, tmp_path):
    from nihai.signals import quant

    wd, s = _detay_duzenegi(monkeypatch, tmp_path, "quant_trend")
    sig = wd.analyze_symbol("AAA", "1d")["signal"]            # analiz penceresi 220 bar < MIN_BAR; quant 300 bar çeker
    assert sig["action"] == "LONG" and not any("YETERSIZ" in r for r in sig["reasons"]), sig
    wd, s = _detay_duzenegi(monkeypatch, tmp_path, "event_breakout")
    d = wd.analyze_symbol("AAA", "1h")
    assert d["ok"] and ("1d", quant.CANLI_GUNLUK_BAR) not in s.cagri              # TA kipinde quant penceresi çekilmez
    assert d["signal"]["hedef_nominal"] is False
    assert d["quant"]["action"] != "KAPSAM_DISI_QUANT"


def test_live_predict_cli_quant_kipinde_canli_taramayla_ayni_girdileri_gecer(monkeypatch, tmp_path):
    from nihai.data import market_calendar
    from nihai.execution import live_predict as L
    from nihai.signals import quant

    class _Kapi:
        soru = 0

        def probability(self, *a, **k):
            _Kapi.soru += 1
            return 0.0

    _, s = _detay_duzenegi(monkeypatch, tmp_path, "quant_trend", bugun_cokus=True)
    monkeypatch.setattr(L, "_strateji_sec", lambda: "quant_trend")
    monkeypatch.setattr(market_calendar, "is_market_open", lambda dt=None: True)    # bugünün barı yarım → atılır
    r = L._row("AAA", s, _Kapi(), "1h")
    assert r["action"] == "LONG" and r["why"].startswith("QUANT_SKOR_+"), r
    assert ("1d", quant.CANLI_GUNLUK_BAR) in s.cagri
    assert r["pwin"] is None and _Kapi.soru == 0                # meta kapı quant sinyalinde kapsam dışı
    assert "python live_predict.py" not in L.__doc__            # belgelenen komut çalışan biçimde (BULGU-95-1 sınıfı)
    assert "python -m nihai.execution.live_predict" in L.__doc__


# ---------------------------------------------------------------------------
# BULGU-106-28 — pano, sabit hedefte KAPANMAYAN pozisyonun (quant / trailing) hedefini çıkış seviyesi gibi gösteriyordu
# ---------------------------------------------------------------------------
def test_sabit_hedef_kurali_tek_kaynak_ve_pano_nominal_hedefi_isaretler(monkeypatch):
    from dataclasses import replace
    from datetime import datetime

    from fastapi.testclient import TestClient

    import nihai.web.web_dashboard as wd
    from nihai.config import CONFIG
    from nihai.risk import position_manager as PM
    from nihai.risk.risk_manager import PortfolioState, VirtualPosition

    assert PM.CONFIG is CONFIG
    eski = CONFIG.risk
    try:
        object.__setattr__(CONFIG, "risk", replace(eski, enable_trailing=False, trail_drop_target=False))
        assert PM.sabit_hedefle_cikar("EventBreakout V1") is True and PM.sabit_hedefle_cikar(None) is True
        assert PM.sabit_hedefle_cikar("quant_trend V1") is False and PM.sabit_hedefle_cikar("QuantTrend") is False
        ps = PortfolioState(equity=20_000.0, ath_equity=25_000.0, cash_balance=20_000.0)
        t = datetime(2026, 10, 1, 10, 0)
        ortak = dict(asset_type="stock", side="LONG", quantity=10, entry_price=100.0, entry_time=t, stop=95.0,
                     target=115.0, last_mark=103.0)
        ps.open_positions += [VirtualPosition(symbol="QQQ", strategy="quant_trend V1", **ortak),
                              VirtualPosition(symbol="TTT", strategy="event_breakout V1", **ortak)]
        monkeypatch.setattr(wd, "get_portfolio_status", lambda: (wd.StateStore(), ps))
        poz = {p["symbol"]: p for p in TestClient(wd.app).get("/api/positions").json()}
        assert poz["QQQ"]["hedef_nominal"] is True and poz["QQQ"]["trailing_enabled"] is True    # quant: bayraktan bağımsız
        assert poz["TTT"]["hedef_nominal"] is False and poz["TTT"]["trailing_enabled"] is False
        object.__setattr__(CONFIG, "risk", replace(eski, enable_trailing=True, trail_drop_target=True))
        assert PM.sabit_hedefle_cikar("EventBreakout V1") is False                               # trailing kipi: TA da nominal
        object.__setattr__(CONFIG, "risk", replace(eski, enable_trailing=True, trail_drop_target=False))
        assert PM.sabit_hedefle_cikar("EventBreakout V1") is True
    finally:
        object.__setattr__(CONFIG, "risk", eski)
    kaynak = (KOK / "nihai/risk/position_manager.py").read_text(encoding="utf-8")
    assert "use_target = sabit_hedefle_cikar(p.strategy)" in kaynak                              # check_exit aynı kuralı okur
    html = (KOK / "nihai/web/web_dashboard.html").read_text(encoding="utf-8")
    assert 'id="az-target-sub"' in html and "sig.hedef_nominal" in html and "pos.hedef_nominal ? 'HEDEF (nominal)'" in html


def test_telegram_sinyal_uyarisi_nominal_hedefi_kar_al_seviyesi_gibi_basmaz(monkeypatch):
    import tests.test_scan_symbol as T
    from nihai.execution import main
    from nihai.notifier import telegram_bot
    from nihai.signals import strategy
    from nihai.signals.strategy import Signal, SignalAction

    # Tek başına koşunca conftest'in VIX yalıtımı modülü henüz göremez (gövdede import) → burada sabitle.
    monkeypatch.setattr(strategy, "_VIX_LOADED", True, raising=False)
    giden = []
    monkeypatch.setattr(telegram_bot, "send", giden.append)
    telegram_bot.notify_signal("AAA", "LONG", 100.0, 3.0, 6.5, stop=95.0, target=115.0, hedef_nominal=True)
    telegram_bot.notify_signal("AAA", "LONG", 100.0, 3.0, 6.5, stop=95.0, target=115.0)
    assert "Hedef (nominal" in giden[0] and "izleyen stop" in giden[0] and "`115.00`" in giden[0]
    assert "🎯 Hedef: `115.00`" in giden[1] and "nominal" not in giden[1]
    # canlı tarama bayrağı sinyalin strateji etiketinden üretir (quant → nominal; TA → sabit hedef)
    for etiket, beklenen in (("QuantTrend", True), ("EventBreakout", False)):
        fx = T._install(monkeypatch, approved=True)
        sig = Signal(SignalAction.LONG, "AAPL", 100.0, 95.0, 115.0, reasons=["x"], quality_score=1.0, rr_ratio=3.0,
                     metadata={"strategy": etiket})
        monkeypatch.setattr(main, "generate_signal", lambda *a, _s=sig, **k: _s)
        monkeypatch.setattr(main, "_ALERTED", {}, raising=False)
        T._scan(fx, T._htf_df())
        assert len(fx.tg.signals) == 1 and fx.tg.signals[0][1]["hedef_nominal"] is beklenen, (etiket, fx.tg.signals)


# ---------------------------------------------------------------------------
# BULGU-106-29 — proje-kapat ayrı gölge kasa koşucusunu (`golge_kasalar --kos`) durdurmuyordu
# ---------------------------------------------------------------------------
def test_proje_kapat_ayri_kasa_kosucusunu_da_durdurur():
    import re
    import shutil

    import pytest

    ps = shutil.which("powershell") or shutil.which("pwsh")
    if ps is None:
        pytest.skip("PowerShell yok")
    betik = (KOK / "proje-kapat.ps1").read_text(encoding="utf-8")
    m = re.search(r"\$svc = Get-CimInstance Win32_Process[^\n]*\|\s*Where-Object \{(.*?)\}\n", betik, re.S)
    assert m, "servis süzgeci bulunamadı"
    satirlar = ["python.exe -m nihai.execution.golge_kasalar --kos",      # ayrı koşucu: durmalı
                "python.exe -m nihai.execution.golge_kasalar",            # rapor CLI'ı (kısa ömürlü): dokunulmaz
                "python.exe -m nihai.execution.main --serve-web", "python.exe -m nihai.data.hub_server",
                "python.exe -m baska.proje"]
    komut = ("@(" + ",".join(f"'{s}'" for s in satirlar) + ") | ForEach-Object { [pscustomobject]@{CommandLine=$_} } | "
             "Where-Object {" + m.group(1) + "} | ForEach-Object { $_.CommandLine }")
    r = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command", komut], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[:400]
    assert r.stdout.split() and [s.strip() for s in r.stdout.strip().splitlines()] == [satirlar[0], satirlar[2], satirlar[3]]


def test_emir_plani_nominal_hedefi_kar_al_bacagi_olarak_onermez():
    import nihai.web.web_dashboard as wd

    sabit = wd._order_plan(price=100.0, atr_val=2.0, action="LONG", stop=95.0, target=115.0)
    assert sabit["hedef_nominal"] is False and sabit["bracket"]["take_profit"]["price"] == 115.0
    assert sabit["multi_leg_option"]["legs"][1]["strike_approx"] == 115
    nominal = wd._order_plan(price=100.0, atr_val=2.0, action="LONG", stop=95.0, target=115.0, hedef_nominal=True)
    assert nominal["hedef_nominal"] is True and nominal["bracket"]["take_profit"] is None     # sabit kâr-al yok
    assert nominal["bracket"]["stop_loss"]["trigger"] == 95.0 and nominal["stop_market"]["trigger"] == 95.0
    assert nominal["multi_leg_option"]["legs"][1]["strike_approx"] == 102                      # hedefe değil ATR'ye dayalı
    html = (KOK / "nihai/web/web_dashboard.html").read_text(encoding="utf-8")
    assert "op.hedef_nominal ? 'giriş + zarar-kes · sabit kâr-al yok: çıkış izleyen stopla'" in html


# ---------------------------------------------------------------------------
# BULGU-106-30 — pano operatör eylemleri diskteki kopyaya yazıyor, ana döngü bellekteki portföyü geri yazıyordu
# ---------------------------------------------------------------------------
_KILIT_TUTAN = ("import sys,time; from pathlib import Path; from nihai.atomic_io import tek_yazar_kilidi; "
                "f = tek_yazar_kilidi(Path(sys.argv[1]) / 'main.lock'); assert f; print('TUTUYOR', flush=True); "
                "time.sleep(30)")


def _operator_duzenegi(monkeypatch, tmp_path, n_poz=1):
    from datetime import datetime
    from types import SimpleNamespace

    import nihai.web.web_dashboard as wd
    from nihai.execution import main as M
    from nihai.execution.state import StateStore
    from nihai.risk.position_manager import PositionManager, SlippageModel
    from nihai.risk.risk_manager import VirtualPosition

    yol = tmp_path / "state.json"
    store = StateStore(yol)                                   # "ana sürecin" bellekteki deposu
    ps = store.load_or_init(20_000.0)
    for k in range(n_poz):
        ps.open_positions.append(VirtualPosition(
            symbol="AAA" if k == 0 else f"BB{k}", asset_type="stock", side="LONG", quantity=10, entry_price=100.0,
            entry_time=datetime(2026, 10, 1, 10, 0), stop=95.0, target=110.0, last_mark=101.0))
    pmgr = PositionManager(ps, SlippageModel(), record_logs=False)
    pmgr.rehydrate_trackers()
    pmgr.add_on_close(lambda ce: M._apply_close_bookkeeping(store, ce))
    store.save()
    kot = SimpleNamespace(get_quote=lambda s: SimpleNamespace(last=102.0))
    monkeypatch.setattr(wd, "get_provider", lambda *a, **k: kot)
    monkeypatch.setattr(wd, "DATA_DIR", tmp_path)
    monkeypatch.setattr(wd, "StateStore", lambda: StateStore(yol))
    monkeypatch.setattr(wd, "_CANLI", {})
    return wd, store, ps, pmgr, yol


def _diskteki(yol):
    from nihai.execution.state import StateStore

    s = StateStore(yol)
    return s, s.load_or_init(1.0)


def test_pano_operator_eylemleri_canli_portfoye_uygulanir_ve_dongu_geri_yazmaz(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    wd, store, ps, pmgr, yol = _operator_duzenegi(monkeypatch, tmp_path)
    wd.canli_baglam_kaydet(store, pmgr)                       # main.run_loop'un yaptığı kayıt
    c = TestClient(wd.app)
    r = c.post("/api/update-position-limits", json={"symbol": "AAA", "asset_type": "stock", "stop": 97.0, "target": 120.0})
    assert r.status_code == 200, r.text
    assert (ps.open_positions[0].stop, ps.open_positions[0].target) == (97.0, 120.0)      # BELLEKTEKİ pozisyon
    assert next(iter(pmgr._trackers.values())).initial_stop == 97.0
    store.kill_switch_active = True
    assert c.post("/api/reset-killswitch").status_code == 200 and store.kill_switch_active is False
    r = c.post("/api/close-position", json={"symbol": "AAA", "asset_type": "stock"})
    assert r.status_code == 200, r.text
    assert ps.open_positions == [] and ps.equity != 20_000.0  # canlı portföy kapandı
    assert len(store.recent_trades_pnl) == 1                  # defter BİR kez: pmgr kancası; uç ikinci kez yazmaz
    store.save()                                              # ana döngünün bir sonraki turu
    assert _diskteki(yol)[1].open_positions == []             # pozisyon geri GELMEZ


def test_pano_flatten_canli_portfoyde_ve_defter_tek_kez(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    wd, store, ps, pmgr, yol = _operator_duzenegi(monkeypatch, tmp_path, n_poz=2)
    wd.canli_baglam_kaydet(store, pmgr)
    r = TestClient(wd.app).post("/api/flatten")
    assert r.status_code == 200, r.text
    assert ps.open_positions == [] and len(store.recent_trades_pnl) == 2
    store.save()
    assert _diskteki(yol)[1].open_positions == []


def test_pano_tek_basina_ana_surec_baska_surecteyse_eylemi_reddeder(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    wd, _store, _ps, _pmgr, yol = _operator_duzenegi(monkeypatch, tmp_path)
    once = yol.read_bytes()
    cocuk = subprocess.Popen([sys.executable, "-c", _KILIT_TUTAN, str(tmp_path)], cwd=KOK, stdout=subprocess.PIPE,
                             text=True, env={**os.environ, "PYTHONPATH": str(KOK)})
    try:
        assert cocuk.stdout is not None and cocuk.stdout.readline().strip() == "TUTUYOR"
        c = TestClient(wd.app)
        limit = {"symbol": "AAA", "asset_type": "stock", "stop": 97.0, "target": 120.0}
        for uc, govde in (("close-position", {"symbol": "AAA", "asset_type": "stock"}), ("flatten", None),
                          ("reset-killswitch", None), ("update-position-limits", limit)):
            r = c.post("/api/" + uc, json=govde) if govde else c.post("/api/" + uc)
            assert r.status_code == 409 and "Ana süreç" in r.json()["detail"], (uc, r.status_code, r.text)
        assert yol.read_bytes() == once                       # ezilecek hiçbir şey yazılmadı
    finally:
        cocuk.kill()
        cocuk.wait()


def test_pano_tek_basina_ana_surec_yokken_diske_yazar_ve_defteri_kendi_tutar(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    wd, _store, _ps, _pmgr, yol = _operator_duzenegi(monkeypatch, tmp_path)
    r = TestClient(wd.app).post("/api/close-position", json={"symbol": "AAA", "asset_type": "stock"})
    assert r.status_code == 200, r.text
    s, p = _diskteki(yol)
    assert p.open_positions == [] and len(s.recent_trades_pnl) == 1
    r = TestClient(wd.app).post("/api/close-position", json={"symbol": "YOK", "asset_type": "stock"})
    assert r.status_code == 404                               # durum kodu artık 500'e çevrilmiyor


def test_cli_flatten_ana_surec_calisirken_reddedilir_ve_dongu_baglami_kaydeder(tmp_path, monkeypatch):
    import inspect
    import time

    import pytest

    from nihai.execution import main as M

    cocuk = subprocess.Popen([sys.executable, "-c", _KILIT_TUTAN, str(tmp_path)], cwd=KOK, stdout=subprocess.PIPE,
                             text=True, env={**os.environ, "PYTHONPATH": str(KOK)})
    monkeypatch.setattr(M, "DATA_DIR", tmp_path)
    monkeypatch.setattr(M, "_ANA_KILIT", None)
    monkeypatch.setattr(M, "validate_config", lambda: ([], []))
    monkeypatch.setattr(sys, "argv", ["main", "--flatten"])
    try:
        assert cocuk.stdout is not None and cocuk.stdout.readline().strip() == "TUTUYOR"
        monkeypatch.setattr(M, "flatten_all_action", lambda: pytest.fail("ana süreç çalışırken disk kopyası kapatıldı"))
        with pytest.raises(SystemExit, match="ETKİSİZ"):
            M.main()
    finally:
        cocuk.kill()
        cocuk.wait()
    time.sleep(0.2)
    cagri = []
    monkeypatch.setattr(M, "flatten_all_action", lambda: cagri.append(1))
    M.main()                                                  # ana süreç yokken flatten koşar
    assert cagri == [1]
    assert "_pano.canli_baglam_kaydet(store, pmgr)" in inspect.getsource(M.run_loop)


# ---------------------------------------------------------------------------
# BULGU-106-31 — yeniden başlatmada taşınan quant pozisyonunun riski/RR'ı/tepesi o anki izleyen stoptan kuruluyordu
# ---------------------------------------------------------------------------
def test_yeniden_baslatmada_quant_riski_rr_ve_tepe_giristeki_mesafeden_kurulur():
    from datetime import datetime

    import pytest

    from nihai.risk.position_manager import PositionManager, SlippageModel
    from nihai.risk.risk_manager import PortfolioState, VirtualPosition

    ps = PortfolioState(equity=20_000.0, ath_equity=25_000.0, cash_balance=20_000.0)
    t0 = datetime(2026, 9, 29, 9, 36)
    # canlı TSM: giriş 448,62 · başlangıç mesafesi 24,72 (stop 423,90) · izleyen stop girişin ÜSTÜNE çıkmış (450,04)
    quant = VirtualPosition(symbol="TSM", asset_type="stock", side="LONG", quantity=5, entry_price=448.62, entry_time=t0,
                            stop=450.04, target=521.55, last_mark=473.67, strategy="quant_trend V1", izleyen_mesafe=24.72)
    ta = VirtualPosition(symbol="TTT", asset_type="stock", side="LONG", quantity=10, entry_price=100.0, entry_time=t0,
                         stop=95.0, target=110.0, last_mark=101.0, strategy="event_breakout V1")
    ps.open_positions += [quant, ta]
    pmgr = PositionManager(ps, SlippageModel(), record_logs=False)
    assert pmgr.rehydrate_trackers() == 2
    tq, tt = (next(t for t in pmgr._trackers.values() if t.pos is x) for x in (quant, ta))
    assert tq.sizing["risk_dollars"] == pytest.approx(24.72 * 5)         # eskiden |448,62 − 450,04| × 5 = 7,1 $
    assert tq.rr_planned == pytest.approx((521.55 - 448.62) / 24.72, abs=0.01)   # eskiden 51,36
    assert tq.high_water == pytest.approx(450.04 + 24.72) and tq.initial_stop == 450.04
    assert tq.mfe_pct == pytest.approx((474.76 - 448.62) / 448.62 * 100)  # tepe yeniden başlatmada kaybolmaz
    # TA pozisyonu (kalıcı mesafe yok): eski kural aynen
    assert tt.sizing["risk_dollars"] == pytest.approx(50.0) and tt.rr_planned == 2.0 and tt.high_water == 0.0
    # izleyen stop yeniden başlatmadan sonra ÇALIŞMAYA devam eder: yeni tepe → stop = tepe − mesafe
    pmgr.update_all_with_bars({"TSM": (474.0, 480.0, 473.0, 479.0)}, now=datetime(2026, 10, 5, 10, 0))
    assert quant.stop == pytest.approx(480.0 - 24.72) and quant in ps.open_positions
    kapanan = pmgr.update_all_with_bars({"TSM": (456.0, 457.0, 454.0, 455.0)}, now=datetime(2026, 10, 5, 10, 1))
    assert [c.pos.symbol for c in kapanan] == ["TSM"] and kapanan[0].mfe_pct == pytest.approx((480.0 - 448.62) / 448.62 * 100)


# ---------------------------------------------------------------------------
# BULGU-106-32 — Sinyal Akışı (/api/signals) bugünün yerine DÜNÜN son 200 karar satırını döndürüyordu
# ---------------------------------------------------------------------------
def _karar_gunlugu(dizin, gun, n, *, gurultu=0, doldur=0):
    """`gun` (YYYY-MM-DD) için n gerçek karar satırı + sonda `gurultu` adet kapalı-seans satırı. `doldur` > 0 ise satırlar
    o kadar bayt uzatılır (512 KB üstü dosya = uçtaki sondan-okuma yolu)."""
    bas = "Dosya: x\nZAMAN               | EYLEM         | GEREKCE / DURUM DETAYI\n" + "-" * 80 + "\n"
    govde = "".join(f"{gun} 16:{i // 60 % 60:02d}:{i % 60:02d} | SKIP          | S{i:05d} sinyal yok{' ' * doldur}\n"
                    for i in range(n))
    kapali = "".join(f"{gun} 23:{i % 60:02d}:42 | SESSION_CLOSED | market kapalı — sinyal taraması yok\n" for i in range(gurultu))
    (dizin / f"decision_log_{gun.replace('-', '_')}.txt").write_text(bas + govde + kapali, encoding="utf-8")


def test_sinyal_akisi_bugunun_kararlarini_en_yeniden_eskiye_dondurur(monkeypatch, tmp_path):
    import nihai.web.web_dashboard as wd

    monkeypatch.setattr(wd, "LOG_DIR", tmp_path)
    monkeypatch.setattr(wd, "StateStore", lambda: type("S", (), {"recent_signals": [], "load_or_init": lambda s, e: None})())
    _karar_gunlugu(tmp_path, "2026-10-01", 1500, gurultu=60)
    for n, doldur in ((1200, 0), (3000, 300)):                    # küçük dosya yolu · 512 KB üstü (sondan okuma) yolu
        _karar_gunlugu(tmp_path, "2026-10-02", n, doldur=doldur)
        r = wd.get_signals()
        assert len(r) == 200 and all(x.startswith("2026-10-02") for x in r), (n, r[0][:19])   # eskiden 200'ü de 1 Ekim'di
        assert f"S{n - 1:05d}" in r[0] and f"S{n - 200:05d}" in r[-1]                        # en yeni önce, bitişik
    # seans yeni açıldı (bugün 50 satır): önce bugünün 50'si, sonra dünün EN YENİ 150'si (kapalı-seans satırları elenir)
    _karar_gunlugu(tmp_path, "2026-10-02", 50)
    r = wd.get_signals()
    assert [x[:10] for x in r] == ["2026-10-02"] * 50 + ["2026-10-01"] * 150
    assert "S00049" in r[0] and "S01499" in r[50] and not any("market kapalı" in x for x in r)
    # hafta sonu: son iki dosya yalnız kapalı-seans gürültüsü → cuma kararları yine görünür
    _karar_gunlugu(tmp_path, "2026-10-03", 0, gurultu=1200)
    _karar_gunlugu(tmp_path, "2026-10-04", 0, gurultu=1200)
    r = wd.get_signals()
    assert len(r) == 200 and r[0].startswith("2026-10-02") and "S00049" in r[0]
    # 200 satır dolunca ESKİ dosyalar hiç açılmaz (uç 5 sn'de bir yoklanır): eski "dosya" açılamaz bir dizin
    ikinci = tmp_path / "b"
    ikinci.mkdir()
    monkeypatch.setattr(wd, "LOG_DIR", ikinci)
    _karar_gunlugu(ikinci, "2026-10-02", 300)
    (ikinci / "decision_log_2026_10_01.txt").mkdir()
    assert len(wd.get_signals()) == 200


# ---------------------------------------------------------------------------
# BULGU-106-33 — "son sinyaller": tarama iş parçacıkları yazmıyor, tarihsiz satırlar güncel görünüyor, pano ters çeviriyor
# ---------------------------------------------------------------------------
def test_son_sinyal_satirlari_tarihli_kilitli_ve_eski_bicim_ayiklanir():
    import re
    import threading

    from nihai.notifier import terminal_dashboard as td

    d = td.LiveDashboard()
    d.add_signal_line("AAPL LONG @ 333.69 | hedef(nominal) 376.92 stop 319.41")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| AAPL LONG @ 333\.69 .*", d.recent_signals[0])
    is_parcaciklari = [threading.Thread(target=lambda k=k: [d.add_signal_line(f"T{k} {i}") for i in range(300)])
                       for k in range(8)]
    for t in is_parcaciklari:
        t.start()
    for t in is_parcaciklari:
        t.join()
    assert len(d.recent_signals) == 20 and all(td.TARIHLI_SATIR.match(s) for s in d.recent_signals)
    # yalnız-saat taşıyan eski satır hangi güne ait olduğunu söylemez → yeniden başlatmada geri yüklenmez
    karisik = ["22:50:43 | CLOSE PPG @ 107.77 | SESSION_FLAT | PnL -2575.64", "2026-10-05 16:31:02 | MU LONG @ 1.00", None]
    assert td.tarihli_satirlar(karisik) == ["2026-10-05 16:31:02 | MU LONG @ 1.00"]


def test_sinyal_akisi_one_cikanlari_tarihli_ve_en_ustte_verir_pano_ters_cevirmez(monkeypatch, tmp_path):
    import nihai.web.web_dashboard as wd

    bellek = ["22:50:43 | CLOSE PPG @ 107.77 | SESSION_FLAT | PnL -2575.64",       # 22 Eylül'den kalma, tarihsiz
              "2026-10-02 16:31:02 | MU LONG @ 190.10 | hedef(nominal) 220.00 stop 180.00",
              "2026-10-02 16:35:40 | CLOSE JPM @ 325.80 | STOP | PnL -44.11"]
    monkeypatch.setattr(wd, "LOG_DIR", tmp_path)
    monkeypatch.setattr(wd, "StateStore", lambda: type("S", (), {"recent_signals": bellek, "load_or_init": lambda s, e: None})())
    _karar_gunlugu(tmp_path, "2026-10-02", 300)
    r = wd.get_signals()
    assert r[0].startswith("2026-10-02 16:35:40 | CLOSE JPM") and r[1].startswith("2026-10-02 16:31:02 | MU LONG")
    assert "S00299" in r[2] and len(r) == 200                    # sonra günlük: en yeni karar önce
    assert not any("CLOSE PPG" in x for x in r)                   # tarihsiz eski satır güncelmiş gibi gösterilmez
    html = (KOK / "nihai/web/web_dashboard.html").read_text(encoding="utf-8")
    assert "[...rawSignals].reverse()" not in html and "const sortedSignals = rawSignals;" in html


def test_tarama_is_parcaciklari_pano_satir_defterine_yazar(monkeypatch):
    import inspect

    import tests.test_scan_symbol as T
    from nihai.execution import main as M
    from nihai.notifier import terminal_dashboard as td
    from nihai.signals import strategy

    monkeypatch.setattr(strategy, "_VIX_LOADED", True, raising=False)      # tek başına koşuda conftest yalıtımı görmez
    monkeypatch.setattr(M, "_alerted_bars", {})
    fx = T._install(monkeypatch, approved=True)
    pano = td.LiveDashboard()                                     # Live'a girilmedi: yalnız satır defteri
    M.scan_symbol("AAPL", T._Provider(T._htf_df()), fx.risk, fx.state, fx.breaker, fx.pmgr, pano, None)
    assert pano.recent_signals and all(td.TARIHLI_SATIR.match(s) for s in pano.recent_signals)
    assert any("AAPL LONG @ 100.00" in s for s in pano.recent_signals)
    kaynak = inspect.getsource(M.run_loop)
    assert "pmgr, None, PORTFOLIO_LOCK" not in kaynak             # iki-katman + tek-katman havuzu: pano artık None değil
    assert kaynak.count("scan_symbol(sym, prov, risk_mgr, store, breaker, pmgr, dash, PORTFOLIO_LOCK)") == 2
    assert kaynak.index("dash = terminal_dashboard.LiveDashboard()") < kaynak.index("def _scan_tier")
    assert "with dash:" in kaynak and "tarihli_satirlar(store.recent_signals)" in kaynak


# ---------------------------------------------------------------------------
# BULGU-106-34 — quant kipinde odak katmanı aynı GÜNLÜK kararı ~6 sn'de bir yeniden üretip yazıyordu
# ---------------------------------------------------------------------------
def test_quant_taramasi_sembol_basina_seyreltilir_ve_muhafiz_satiri_gunde_bir(monkeypatch, tmp_path):
    from datetime import datetime, timedelta
    from types import SimpleNamespace

    import pandas as pd

    import nihai.execution.main as M
    from nihai.execution.state import StateStore

    saat = [datetime(2026, 10, 5, 10, 0, 0)]
    loglar, cekilen = [], []
    monkeypatch.setattr(M, "now_et_naive", lambda: saat[0])
    monkeypatch.setattr(M, "log_block", lambda tag, msg="": loglar.append(tag))
    monkeypatch.setattr(M, "can_enter_new_position", lambda: (True, ""))
    monkeypatch.setattr(M, "event_guard", lambda sym: (False, ""))
    monkeypatch.setattr(M, "corporate_action_guard", lambda sym: (False, ""))
    monkeypatch.setattr(M, "_quant_odak", lambda: frozenset({"AAA", "BBB"}))

    class Prov:
        def get_bars(self, sym, *a, **k):
            cekilen.append(sym)
            return pd.DataFrame()                                   # DATA_EMPTY: değerlendirme buraya kadar geldi

    store = StateStore(tmp_path / "state.json")
    store.load_or_init(20_000.0)
    pm = SimpleNamespace(portfolio=SimpleNamespace(open_positions=[SimpleNamespace(symbol="BBB")]))

    def tara(sym):
        M.scan_symbol(sym, Prov(), None, store, None, pm, strategy_mode="quant_trend")

    for _ in range(10):                                             # odak katmanı ~6 sn'de bir döner
        tara("AAA")
        tara("BBB")
        saat[0] += timedelta(seconds=6)
    # 60 sn'de AAA en çok iki kez değerlendirilir (0. ve 48. sn; eskiden 10), açık pozisyonlu BBB tek satır yazar (eskiden 10)
    assert cekilen == ["AAA", "AAA"] and loglar.count("DATA_EMPTY") == 2
    assert loglar.count("QUANT_POZISYON_VAR") == 1
    saat[0] += timedelta(days=1)                                    # yeni ET günü: muhafız satırı yeniden bir kez
    tara("BBB")
    tara("BBB")
    assert loglar.count("QUANT_POZISYON_VAR") == 2
    # TA kipi seyreltilmez (olay sinyali her taramada sorulur)
    n0 = len(cekilen)
    for _ in range(3):
        M.scan_symbol("AAA", Prov(), None, store, None, pm, strategy_mode="event_breakout")
    assert len(cekilen) == n0 + 3
    # izler state.json'a yazılmaz (süreç ömürlü, geçici)
    store.save()
    assert "quant_son_tarama" not in (tmp_path / "state.json").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# BULGU-106-35 — tepe/dip/bar sayısı kalıcı değildi: çok günlük pozisyonun MAE/MFE'si her yeniden başlatmada sıfırlanıyordu
# ---------------------------------------------------------------------------
def test_mae_mfe_ve_bar_sayisi_yeniden_baslatmadan_sonra_kaldigi_yerden_surer(tmp_path):
    from datetime import datetime

    import pytest

    from nihai.execution.state import StateStore
    from nihai.risk.position_manager import PositionManager, SlippageModel
    from nihai.risk.risk_manager import VirtualPosition

    yol = tmp_path / "state.json"
    store = StateStore(yol)
    ps = store.load_or_init(20_000.0)
    ps.open_positions.append(VirtualPosition(
        symbol="QQQ", asset_type="stock", side="LONG", quantity=10, entry_price=100.0, entry_time=datetime(2026, 9, 29, 9, 36),
        stop=91.0, target=127.0, last_mark=100.0, strategy="quant_trend V1", izleyen_mesafe=9.0))
    pm = PositionManager(ps, SlippageModel(), record_logs=False)
    pm.rehydrate_trackers()
    # 1. oturum: önce dip (94), sonra tepe (112) → izleyen stop 103'e çıkar
    pm.update_all_with_bars({"QQQ": (99.0, 100.0, 94.0, 96.0)}, now=datetime(2026, 9, 29, 10, 0))
    pm.update_all_with_bars({"QQQ": (97.0, 112.0, 96.0, 111.0)}, now=datetime(2026, 9, 29, 10, 1))
    poz = ps.open_positions[0]
    assert (poz.en_iyi_fiyat, poz.en_kotu_fiyat, poz.bar_sayisi) == (112.0, 94.0, 2) and poz.stop == pytest.approx(103.0)
    store.save()
    # proje kapatılıp açıldı: yeni süreç durumu diskten kurar
    store2 = StateStore(yol)
    ps2 = store2.load_or_init(1.0)
    pm2 = PositionManager(ps2, SlippageModel(), record_logs=False)
    pm2.rehydrate_trackers()
    t = next(iter(pm2._trackers.values()))
    assert (t.high_water, t.low_water, t.bars_held) == (112.0, 94.0, 2)      # eskiden: tepe yeniden kurulur, dip = giriş, bar 0
    assert t.mae_pct == pytest.approx(-6.0) and t.mfe_pct == pytest.approx(12.0)
    # 2. oturum: daha düşük bir tepe MFE'yi DÜŞÜRMEZ; yeni tepe izleyen stopu YÜRÜTMEYE devam eder (başlangıç stopu da
    # geri kurulur); stop çıkışında defter tüm tutma süresini ölçer
    pm2.update_all_with_bars({"QQQ": (108.0, 109.0, 107.0, 108.0)}, now=datetime(2026, 9, 30, 10, 0))
    assert t.mfe_pct == pytest.approx(12.0) and ps2.open_positions[0].stop == pytest.approx(103.0)
    pm2.update_all_with_bars({"QQQ": (110.0, 115.0, 109.0, 114.0)}, now=datetime(2026, 9, 30, 10, 1))
    assert ps2.open_positions[0].stop == pytest.approx(106.0)
    kapanan = pm2.update_all_with_bars({"QQQ": (107.0, 108.0, 105.0, 105.5)}, now=datetime(2026, 9, 30, 10, 2))
    assert len(kapanan) == 1
    ce = kapanan[0]
    assert ce.mfe_pct == pytest.approx(15.0) and ce.mae_pct == pytest.approx(-6.0) and ce.bars_held >= 4
    # eski kayıt (alanlar yok → 0): 106-31'in geri kurulumu geçerli kalır
    eski = VirtualPosition(symbol="OLD", asset_type="stock", side="LONG", quantity=1, entry_price=50.0,
                           entry_time=datetime(2026, 9, 29, 9, 36), stop=52.0, target=65.0, last_mark=55.0,
                           strategy="quant_trend V1", izleyen_mesafe=5.0)
    ps2.open_positions.append(eski)
    pm2.rehydrate_trackers()
    te = next(x for x in pm2._trackers.values() if x.pos is eski)
    assert te.high_water == pytest.approx(57.0) and te.low_water == 50.0 and te.bars_held == 0


# ---------------------------------------------------------------------------
# BULGU-106-36 — gece görevi seans AÇIKKEN koşuyordu (kış saatinde 23:30 = 15:30 ET); özet "bugün"e bağlıydı
# ---------------------------------------------------------------------------
def test_gunluk_ozet_ozellik_satirini_seansiyla_etiketler(tmp_path, monkeypatch):
    from datetime import datetime, timedelta

    import nihai.notifier.daily_summary as ds
    from nihai.ai import reasoner

    taban = {"today": "2026-12-02", "history_last": None, "history_prev": None, "gate_sim": None, "coverage": None,
             "services": {}, "live_features_today": 42}
    assert "Canlı özellik logu: bugün +42 satır." in ds.build_summary(taban)                       # gün bilgisi yok → eski metin
    assert "Canlı özellik logu: bugün +42 satır." in ds.build_summary({**taban, "live_features_day": "2026-12-02"})
    assert "Canlı özellik logu: son seans (2026-12-01) +42 satır." in ds.build_summary({**taban, "live_features_day": "2026-12-01"})
    # collect: eşik reasoner'ın tek kaynağından; eşikten ÖNCEKİ satır sayılmaz (yerel gün aynı olsa bile)
    esik = "2026-12-01T17:30:00"
    monkeypatch.setattr(reasoner, "seans_esigi", lambda now=None: esik)
    feats = tmp_path / "live_features_log.csv"
    t0 = datetime.fromisoformat(esik)
    feats.write_text("logged_at,symbol\n" + "".join(
        f"{(t0 + timedelta(minutes=dk)).isoformat(timespec='seconds')},AAPL\n" for dk in (-400, -1, 0, 5, 391)), encoding="utf-8")
    monkeypatch.setattr(ds, "FEATS_LOG", feats)
    for ad in ("HISTORY_CSV", "GATE_SIM_JSON", "COVERAGE_JSON", "SHADOW_PROMOTE_JSON"):
        monkeypatch.setattr(ds, ad, tmp_path / f"{ad}.yok")
    data = ds.collect(ports={})
    assert (data["live_features_today"], data["live_features_day"]) == (3, "2026-12-01")


def test_takvim_cli_kalan_saniye_seans_kapisi_icin(monkeypatch, capsys):
    from datetime import timedelta

    import nihai.data.market_calendar as mc

    def bas(kalan):
        monkeypatch.setattr(mc, "session_remaining", lambda dt=None: kalan)
        monkeypatch.setattr("sys.argv", ["market_calendar", "--kalan-sn"])
        mc.main()
        return capsys.readouterr().out.strip()

    assert bas(None) == "-1"                                   # kapalı (hafta sonu dahil: hata değil)
    assert bas(timedelta(minutes=30, seconds=0.7)) == "1800"
    assert bas(timedelta(seconds=0.2)) == "1"                  # açık seans asla 0/-1 basmaz (kapı "açık"ı > 0 ile okur)


def test_gece_gorevi_seans_acikken_kosmaz_kapanisi_bekler_ya_da_atlar(tmp_path):
    """Gerçek PowerShell + gerçek venv başlatıcısı; `nihai`/`research` sahte modüllerle (takvim cevabı dosyadan)."""
    import os
    import shutil
    import sys

    import pytest

    ps = shutil.which("powershell") or shutil.which("pwsh")
    venv = Path(sys.prefix)
    if ps is None or not (venv / "pyvenv.cfg").exists() or not (venv / "Scripts" / "python.exe").exists():
        pytest.skip("PowerShell ya da Windows venv başlatıcısı yok")
    kok = tmp_path / "depo"
    (kok / ".venv" / "Scripts").mkdir(parents=True)
    shutil.copy2(venv / "Scripts" / "python.exe", kok / ".venv" / "Scripts" / "python.exe")
    shutil.copy2(venv / "pyvenv.cfg", kok / ".venv" / "pyvenv.cfg")
    shutil.copy2(KOK / "forward-harness.ps1", kok / "forward-harness.ps1")
    for paket in ("nihai", "nihai/data", "nihai/notifier", "research"):
        (kok / paket).mkdir(parents=True, exist_ok=True)
        (kok / paket / "__init__.py").write_text("", encoding="utf-8")
    iz = "import pathlib, sys\npathlib.Path('izler.txt').open('a').write({ad!r} + ' ' + ' '.join(sys.argv[1:]) + '\\n')\n"
    (kok / "research" / "forward_harness.py").write_text(iz.format(ad="harness"), encoding="utf-8")
    (kok / "research" / "calibrate_forecast.py").write_text(iz.format(ad="kalibrasyon"), encoding="utf-8")
    (kok / "nihai" / "notifier" / "daily_summary.py").write_text(iz.format(ad="ozet"), encoding="utf-8")
    # sahte takvim: kalan.txt'in İLK cevabını basar ve tüketir (bekleme sonrası ikinci soru sıradaki cevabı alır)
    (kok / "nihai" / "data" / "market_calendar.py").write_text(
        "import pathlib, sys\n"
        "p = pathlib.Path('kalan.txt')\n"
        "s = p.read_text().split()\n"
        "pathlib.Path('sorular.txt').open('a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        "p.write_text(' '.join(s[1:]) or s[0])\n"
        "if s[0] == 'HATA':\n"
        "    sys.exit(3)\n"
        "print(s[0])\n", encoding="utf-8")
    ortam = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    import time

    def kos(cevaplar, *args):
        (kok / "kalan.txt").write_text(cevaplar, encoding="ascii")
        for ad in ("izler.txt", "sorular.txt", "logs/forward-harness.log"):
            (kok / ad).unlink(missing_ok=True)
        r = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                            str(kok / "forward-harness.ps1"), *args], capture_output=True, text=True, timeout=180, env=ortam)
        oku = lambda ad: (kok / ad).read_text(encoding="utf-8-sig").splitlines() if (kok / ad).exists() else []  # noqa: E731
        return r.returncode, oku("izler.txt"), "\n".join(oku("logs/forward-harness.log")), oku("sorular.txt")

    tam = ["harness --resolve --snapshot --report", "kalibrasyon ", "ozet --calib --send"]
    # kapalı seans: eskisi gibi koşar (takvime BİR kez sorulur)
    kod, izler, gunluk, sorular = kos("-1")
    assert (kod, izler, sorular) == (0, tam, ["--kalan-sn"]) and "forward-harness calisiyor" in gunluk and "BITTI" in gunluk
    # seans açık, kapanışa 2 saat: koşu ATLANIR — hiçbir modül koşmaz, "calisiyor" banner'ı yazılmaz, çıkış 0
    kod, izler, gunluk, sorular = kos("7200")
    assert (kod, izler) == (0, []) and "ATLANDI" in gunluk and "calisiyor" not in gunluk and len(sorular) == 1
    # bekleme penceresinin sınırı: -BeklemeDk 1 → 61 sn kala atlanır (beklenmez)
    kod, izler, gunluk, sorular = kos("61", "-BeklemeDk", "1", "-KapanisPayiSn", "0")
    assert izler == [] and "ATLANDI" in gunluk and "bekleniyor" not in gunluk
    # kapanışa az kala (kış 23:30 tetiği): kapanış beklenir, takvime YENİDEN sorulur, kapalıysa koşar
    t0 = time.monotonic()
    kod, izler, gunluk, sorular = kos("1 -1", "-KapanisPayiSn", "4")
    assert (kod, izler, len(sorular)) == (0, tam, 2) and "kapanisa 1 sn: 5 sn bekleniyor" in gunluk   # kalan + pay
    assert time.monotonic() - t0 >= 5.0                                   # süre gerçekten beklendi
    assert gunluk.index("bekleniyor") < gunluk.index("forward-harness calisiyor")
    # bekledikten sonra seans yine açıksa (makine uyudu, ertesi seansa uyandı): atlanır
    kod, izler, gunluk, sorular = kos("1 7200", "-KapanisPayiSn", "0")
    assert (kod, izler, len(sorular)) == (0, [], 2) and "ATLANDI" in gunluk
    # elle zorlamada takvime sorulmaz; takvim okunamazsa eski davranış (koş)
    kod, izler, gunluk, sorular = kos("7200", "-SeansIcinde")
    assert (izler, sorular) == (tam, [])
    kod, izler, gunluk, sorular = kos("HATA")
    assert (kod, izler) == (0, tam)


# ---------------------------------------------------------------------------
# BULGU-106-37 — gölge görev betiği kesirli seans süresini tam saate yuvarlıyordu (6,5 → 6; yarım gün 3,5 → 4)
# ---------------------------------------------------------------------------
def test_golge_gorev_tekrar_suresi_kesirli_saati_korur():
    import re
    import shutil

    import pytest

    ps = shutil.which("powershell") or shutil.which("pwsh")
    if ps is None:
        pytest.skip("PowerShell yok")
    betik = (KOK / "tur17-shadow-gorev.ps1").read_text(encoding="utf-8-sig")
    m = re.search(r"-RepetitionDuration (\(New-TimeSpan .*\))\s*\n", betik)
    assert m, "tekrar süresi ifadesi bulunamadı"
    ifade = m.group(1)
    # betiğin KENDİ ifadesi gerçek PowerShell'de: normal seans 6,5 sa = 390 dk, yarım gün 3,5 sa = 210 dk, tam saat aynı
    komut = "; ".join(f"$DurationHours = [double]'{s}'; ({ifade}).TotalMinutes" for s in ("6.5", "3.5", "7", "6.50"))
    r = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command", komut], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[:400]
    assert [float(x.replace(",", ".")) for x in r.stdout.split()] == [390.0, 210.0, 420.0, 390.0]
    # kusurun eski ifadesi (Int32'ye yuvarlanan -Hours $DurationHours) kanonik betikte kalmadı
    assert "New-TimeSpan -Hours $DurationHours" not in betik
