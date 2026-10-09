"""Tur-91 — canlı barın SAĞLAYICIDAN bağımsızlığı (2026-09-23).

Kilitler (her biri kusurun ölçülmüş hâlinden):
  BULGU-91-1  Canlı 1h ızgarası sağlayıcıya bağlıydı: Alpaca/akış saate (09:00, 10:00…), yfinance yedeği
              seansa (09:30, 10:30…) hizalı; hub 429'da seans içinde günde 471–1.446 1h çekimi yedeğe
              düşüyor, aynı veriden iki ızgara eyleme dönük kararların %64,4'ünde ayrışıyordu.
  BULGU-91-2  Alpaca gün-içi geçmişi temettü DE düzeltiliyordu ('all'), yfinance gün-içi yalnız bölünme:
              her temettü tarihinden geriye fiyat düzeyi bir temettü verimi kadar basamaklanıyordu (%3,9 karar).
  BULGU-91-3  yfinance seans sonrası son günlük satırı OHLC'siz döndürüyor (25/25 sembol); yedek atmıyordu.
  BULGU-91-4  Otomatik tarayıcı kapalı piyasada aynı kapanmış barları 4 saatte bir yeniden tarıyordu.
"""
from __future__ import annotations

import inspect
from collections import deque
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from nihai.data import alpaca_provider as ap
from nihai.data import alpaca_stream_provider as asp
from nihai.data import data_provider as dp
from nihai.execution import auto_screener as ascr

YF_IZGARA = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]


def _et(idx: pd.DatetimeIndex) -> list[str]:
    return idx.tz_localize("UTC").tz_convert("America/New_York").strftime("%H:%M").tolist()


def _seans_dakikalari(gun: str, kapanis: str = "16:00") -> pd.DataFrame:
    """Bir seansın 1m barları (naive UTC); fiyat dakika sırası = açılıştan geçen dakika."""
    et = pd.date_range(f"{gun} 09:30", f"{gun} {kapanis}", freq="1min", inclusive="left",
                       tz="America/New_York")
    n = len(et)
    fiyat = pd.Series(range(n), dtype=float)
    return pd.DataFrame({"open": fiyat.values, "high": fiyat.values + 0.5, "low": fiyat.values - 0.5,
                         "close": fiyat.values, "volume": 1.0}, index=et.tz_convert("UTC").tz_localize(None))


# ============================================================ BULGU-91-1: tek ızgara
def test_resample_seans_acilisina_capali_yaz_ve_kis():
    for gun in ("2026-09-22", "2026-12-15"):          # EDT (13:30 UTC) ve EST (14:30 UTC)
        out = dp.resample_bars(_seans_dakikalari(gun), 60)
        assert _et(out.index) == YF_IZGARA, gun
        ilk, son = out.iloc[0], out.iloc[-1]
        assert (ilk.open, ilk.close, ilk.volume) == (0.0, 59.0, 60.0)     # 09:30–10:30 tam saat
        assert (son.open, son.close, son.volume) == (360.0, 389.0, 30.0)  # 15:30–16:00 yarım bar


def test_yarim_gun_son_bari_kapanista_biter():
    out = dp.resample_bars(_seans_dakikalari("2026-11-27", "13:00"), 60)
    assert _et(out.index) == ["09:30", "10:30", "11:30", "12:30"]
    assert out.iloc[-1].volume == 30.0


def test_alt_saat_araliklari_degismez():
    """5m/15m/30m'de çapa ofseti 0: 09:30 zaten kova sınırıdır (davranış Tur-90 ile aynı)."""
    d = _seans_dakikalari("2026-09-22")
    for dk in (5, 15, 30):
        out = dp.resample_bars(d, dk)
        ikinci = (pd.Timestamp("2026-09-22 09:30") + pd.Timedelta(minutes=dk)).strftime("%H:%M")
        assert _et(out.index)[:2] == ["09:30", ikinci]
        assert len(out) == 390 // dk


def _otuz_dk(gun: str) -> pd.DataFrame:
    d = _seans_dakikalari(gun)
    ek = pd.DataFrame({"open": 9.0, "high": 999.0, "low": -99.0, "close": 9.0, "volume": 5.0},
                      index=pd.DatetimeIndex([d.index[0] - pd.Timedelta(minutes=30),     # 09:00 ET
                                              d.index[-1] + pd.Timedelta(minutes=1)]))    # 16:00 ET
    otuz = dp.resample_bars(pd.concat([d, ek]).sort_index(), 30)
    otuz.index = otuz.index.tz_localize("UTC")
    return otuz


def test_canli_rest_1h_yfinance_izgarasinda_ve_seans_disi_yok():
    out = ap._normalize_alpaca_bars(_otuz_dk("2026-09-22"), "X", "1h", 200, rth=True, src_minutes=30)
    assert _et(out.index) == YF_IZGARA
    assert out.high.max() < 999.0 and out.low.min() > -99.0, "seans dışı alt bar toplama girdi"


def test_akis_toplami_rest_ile_ayni_izgara(monkeypatch):
    """Akışın canlı dakika toplamı ile REST tohumu AYNI damgaları üretmeli (birleşik seri tek ızgara)."""
    d = _seans_dakikalari("2026-09-22")
    canli = asp._aggregate_minutes(d, "1h")
    rest = ap._normalize_alpaca_bars(_otuz_dk("2026-09-22"), "X", "1h", 200, rth=True, src_minutes=30)
    assert list(canli.index) == list(rest.index)
    assert canli[["open", "high", "low", "close", "volume"]].equals(rest[["open", "high", "low", "close",
                                                                           "volume"]])


def test_akis_birlesimi_tohum_sonrasini_ayni_izgarada_ekler(monkeypatch):
    d = _seans_dakikalari("2026-09-22")
    tohum = dp.resample_bars(d.iloc[:180], 60)            # 09:30–12:30 REST'ten
    p = asp.AlpacaStreamProvider(symbols=[], rest=SimpleNamespace(get_bars=lambda *a, **k: tohum),
                                 max_symbols=30)
    p._minutes["AAPL"] = deque(
        (t.tz_localize("UTC"), r.open, r.high, r.low, r.close, r.volume) for t, r in d.iterrows())
    monkeypatch.setattr(asp, "datetime", SimpleNamespace(
        now=lambda tz=None: pd.Timestamp("2026-09-22 21:00", tz="UTC").to_pydatetime()))
    out = p.get_bars("AAPL", "1h", lookback_bars=50)
    assert _et(out.index) == YF_IZGARA and not out.index.duplicated().any()


def test_izgara_kaynak_kapilari():
    kaynak = inspect.getsource(dp.resample_bars)
    assert "offset" in kaynak and "_SEANS_ACILIS_DK" in kaynak
    assert "resample_bars(" in inspect.getsource(asp._aggregate_minutes)
    assert not hasattr(asp, "_RESAMPLE_FREQ"), "saate hizalı ikinci toplama yolu geri geldi"


# ============================================================ BULGU-91-2: gün-içi düzeltme kipi
def test_canli_duzeltme_gun_icinde_temettusuz(monkeypatch):
    beklenen = {"all": ("split", "all"), "split": ("split", "split"),
                "dividend": ("raw", "dividend"), "raw": ("raw", "raw")}
    for env, (gun_ici, gunluk) in beklenen.items():
        monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", env)
        assert (ap._canli_duzeltme("1h"), ap._canli_duzeltme("1m"), ap._canli_duzeltme("1d")) == (
            gun_ici, gun_ici, gunluk), env
    monkeypatch.delenv("ALPACA_HIST_ADJUSTMENT")
    assert ap._adjustment() == "all", "araştırma yolunun varsayılanı değişmemeli"


def test_canli_istek_gun_icinde_split_gunlukte_all(monkeypatch):
    monkeypatch.delenv("ALPACA_HIST_ADJUSTMENT", raising=False)
    istekler = []
    istemci = SimpleNamespace(get_stock_bars=lambda req: istekler.append(req) or SimpleNamespace(
        df=_otuz_dk("2026-09-22")))
    p = ap.AlpacaProvider(api_key="k", secret_key="s", data_client=istemci)
    p.get_bars("X", "1h", 50)
    p.get_bars("X", "1d", 50)
    assert [str(getattr(r.adjustment, "value", r.adjustment)) for r in istekler] == ["split", "all"]


def test_arastirma_yolu_canli_kipi_okumaz():
    from nihai.data import alpaca_history as ah
    kaynak = inspect.getsource(ah)
    assert "_canli_duzeltme" not in kaynak and "Adjustment(_adjustment())" in kaynak


# ============================================================ BULGU-91-3: yfinance boş günlük satır
def test_yfinance_ohlcsiz_son_gunluk_satiri_atar(monkeypatch):
    idx = pd.DatetimeIndex(["2026-09-18", "2026-09-21", "2026-09-22"])
    ham = pd.DataFrame({"Open": [1.0, 2.0, float("nan")], "High": [1.0, 2.0, float("nan")],
                        "Low": [1.0, 2.0, float("nan")], "Close": [1.0, 2.0, float("nan")],
                        "Volume": [10, 20, 30]}, index=idx)
    monkeypatch.setattr(dp.yf, "download", lambda *a, **k: ham.copy())
    out = dp.YFinanceProvider().get_bars("SPY", "1d", 10)
    assert len(out) == 2 and out.close.iloc[-1] == 2.0


# ============================================================ BULGU-91-4: tarayıcı zamanlaması
_KAPALI = datetime.fromisoformat("2026-09-23T12:00:00+03:00")        # 05:00 ET, seans yok
_ACIK = datetime.fromisoformat("2026-09-22T18:00:00+03:00")          # 11:00 ET


def test_kapali_piyasada_taze_tarama_tekrarlanmaz(monkeypatch):
    monkeypatch.setattr(ascr.watchlist, "last_scan", lambda: "2026-09-23T09:05:53+03:00")
    assert ascr.girdi_tazelenmedi(_KAPALI) is True                  # son kapanış 09-22 20:00 UTC'den sonra
    monkeypatch.setattr(ascr.watchlist, "last_scan", lambda: "2026-09-22T22:59:00+03:00")   # 15:59 ET
    assert ascr.girdi_tazelenmedi(_KAPALI) is False                 # kapanıştan ÖNCEKİ tarama bayat
    monkeypatch.setattr(ascr.watchlist, "last_scan", lambda: "2026-09-22T17:30:00+03:00")
    assert ascr.girdi_tazelenmedi(_ACIK) is False                   # seans açık: yeni dakika kapanıyor
    monkeypatch.setattr(ascr.watchlist, "last_scan", lambda: None)
    assert ascr.girdi_tazelenmedi(_KAPALI) is False


def test_dongu_ertelemede_taramaz_seans_acilinca_tarar(monkeypatch):
    durum = {"taze": True, "tarama": 0}
    s = ascr.AutoScreener(interval_hours=1.0, run_immediately=True)
    monkeypatch.setattr(ascr, "girdi_tazelenmedi", lambda: durum["taze"])

    def tara(stop_event=None):
        durum["tarama"] += 1
        s._stop.set()
        return []
    monkeypatch.setattr(ascr, "scan_and_update", tara)
    beklemeler = []

    def bekle(sn):
        beklemeler.append(sn)
        if len(beklemeler) == 3:
            durum["taze"] = False                                   # seans açıldı
        return s._stop.is_set()
    monkeypatch.setattr(s._stop, "wait", bekle)
    s._loop()
    assert durum["tarama"] == 1 and len(beklemeler) >= 3, "ertelenen tarama yapılmadı ya da erken yapıldı"


# ============================================================ replay: canlıyla aynı bar kurulumu
def test_replay_htf_barini_1m_seans_dakikalarindan_turetir(tmp_path, monkeypatch):
    from nihai.data import alpaca_history
    from nihai.execution import replay

    monkeypatch.setattr(alpaca_history, "HIST_DIR", tmp_path)
    monkeypatch.setattr(alpaca_history, "_fetch", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("ağ")))
    seans = _seans_dakikalari("2026-09-08")
    once = seans.iloc[:90].copy()
    once.index = once.index - pd.Timedelta(minutes=90)          # 08:00–09:29 ET seans-öncesi
    once[["high", "close"]] = 777.0
    pd.concat([once, seans]).to_csv(alpaca_history._cache_path("AAA", "1m"))
    sahte = pd.DataFrame({"open": 555.0, "high": 555.0, "low": 555.0, "close": 555.0, "volume": 1.0},
                         index=pd.date_range("2026-09-08 12:00", "2026-09-08 20:00", freq="h"))
    sahte.to_csv(alpaca_history._cache_path("AAA", "1h"))     # araştırma önbelleği: saat ızgarası
    yuk, eksik = replay.load_history(["AAA"], "2026-09-08", "2026-09-08", ["1m", "1h", "1d"])
    dk, saat = yuk[("AAA", "1m")], yuk[("AAA", "1h")]
    assert len(dk) == 390 and 777.0 not in dk.close.values, "seans-öncesi dakika replay'e girdi"
    assert _et(saat.index) == YF_IZGARA and 555.0 not in saat.close.values
    assert not [e for e in eksik if e["reason"] == "SESSION_GAPS"], eksik
