"""Tur-84 — HESAPLAMALARIN doğruluğu: ufuk tanımı, öngörü aralığı, gün-içi değişim.

Üç bulgu kilitlenir (ölçümler: docs/TUR84_RAPORU_2026-09-19.md):
  BULGU-84-1  Defterin "5 bar ileri" çözümü KONUMLA seçiliyordu; IEX'te işlem olmayan dakika bar
              üretmez → satırların %18,6'sında ufuk >5 dk, 411 satırda gece/hafta sonu aşılıyordu.
              → ufuk ZAMANLA (düzenli ızgarada eski kuralla birebir).
  BULGU-84-2  "Öngörü" = fiyat ± ATR (yön renginde) "fiyat değişmez"den ölçülerek kötüydü
              (MAE 24,47 ↔ 21,29 bps, gün-kümeli t=+8,7). → kalibre ARALIK: ölçek ATR14, kuantiller
              defterin `atr_pct` kolonundan gece görevinde; yürüyen kapı koşulsuz aralığa karşı.
  BULGU-84-3  "Değişim" / "% (gün)" gerçek seans açılışını satırların yalnız %3,5'inde kullanıyordu.
  BULGU-84-4  Aralığın çapası kotasyon değil SON BAR KAPANIŞI: aynı 83.147 satırda aralık skoru
              74,4 ↔ 96,9 bps (eşleştirilmiş gün-kümeli t=−10,7). Şema kapısı `referans` alanını arar.
  +           alpaca_history önbellek anahtarı feed taşımıyordu (IEX isteği SIP dosyasını okuyup ezerdi);
              geçici okuma hatası aralık/eğri yükleyicisinde önbelleğe yazılıyordu (Gemini incelemesi).
"""
from __future__ import annotations

import csv
import inspect
import io
import json
import os
import tokenize
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nihai.execution import live_predict_service as lps
from nihai.execution import prediction_tracker as pt
from nihai.signals.strategy import SignalAction, atr
from research import calibrate_forecast as cf

KOK = Path(__file__).resolve().parents[1]


def _kod(yol: str) -> str:
    """Kaynağın YORUM ve DİZGİ içermeyen hâli — regresyon kapısı anlatıya takılmasın (Tur-49)."""
    kaynak = (KOK / yol).read_text(encoding="utf-8")
    parca = [t.string for t in tokenize.generate_tokens(io.StringIO(kaynak).readline)
             if t.type not in (tokenize.COMMENT, tokenize.STRING)]
    return " ".join(parca)


def _seri(dakikalar, t0="2026-09-18 14:00", *, taban=100.0, tz=None) -> pd.DataFrame:
    """1m barlar; kapanış = taban + dakika → hangi barın okunduğu sayıdan görünür."""
    idx = pd.DatetimeIndex([pd.Timestamp(t0) + pd.Timedelta(minutes=m) for m in dakikalar])
    if tz:
        idx = idx.tz_localize(tz)
    kap = [taban + m for m in dakikalar]
    return pd.DataFrame({"open": kap, "high": kap, "low": kap, "close": kap, "volume": 1.0},
                        index=idx)


def _ileri(df, i=0, horizon=5, interval="1m"):
    return pt._forward_price(df, pt._pos_map(df), str(df.index[i]), horizon, interval)


@pytest.fixture
def defter(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "PRED_LOG", tmp_path / "predictions_log.csv")
    monkeypatch.setattr(pt, "FEATS_LOG", tmp_path / "live_features_log.csv")
    return pt.PRED_LOG


def _oku(yol) -> list[dict]:
    with yol.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ================================================ 1. ufuk ZAMANLA ölçülür (BULGU-84-1)
def test_duzenli_izgarada_konum_kuraliyla_birebir():
    df = _seri(range(12))
    for i in range(7):
        assert _ileri(df, i) == float(df["close"].iloc[i + 5])          # eski kural
    # hedefte BAŞLAYAN bar son bar olsa bile hemen çözülür (eskisi gibi; arkası beklenmez)
    assert _ileri(_seri(range(6))) == 105.0


def test_boslukta_hedefteki_son_fiyat_konum_degil():
    df = _seri([0, 1, 2, 9, 10, 11, 12])      # 3..8. dakikalarda işlem yok → bar yok (IEX)
    assert _ileri(df) == 102.0                # t0+5'teki son fiyat
    assert float(df["close"].iloc[5]) == 111.0   # eski konum kuralı 11 dk sonrasını okurdu


def test_hedef_gecmeden_cozulmez_seans_sonu_kapanisla_cozulur():
    kapanis = _seri([0, 1, 2])
    assert _ileri(kapanis) is None            # hedef an henüz geçmedi (arkasında bar yok)
    ertesi = pd.concat([kapanis, _seri([0, 1, 2, 3], t0="2026-09-19 13:30", taban=500.0)])
    assert _ileri(ertesi) == 102.0            # ertesi günün fiyatı DEĞİL: hedefteki son fiyat
    assert float(ertesi["close"].iloc[5]) == 502.0   # eski kural geceyi aşardı


def test_gunluk_aralikta_konum_kurali_kalir():
    idx = pd.date_range("2026-09-01 04:00", periods=8, freq="D")
    df = pd.DataFrame({"close": np.arange(8.0) + 100.0}, index=idx)
    assert pt._forward_price(df, pt._pos_map(df), str(idx[0]), 5, "1d") == 105.0
    assert pt._forward_price(df, pt._pos_map(df), str(idx[0]), 5, None) == 105.0


def test_tz_farkinda_index_ayni_sonuc():
    df = _seri([0, 1, 2, 9, 10], tz="UTC")
    assert _ileri(df) == 102.0


def test_resolve_ucdan_uca_zaman_kurali(defter):
    df = _seri([0, 1, 2, 9, 10, 11, 12])

    class _Barlar:
        def get_bars(self, *a, **k):
            return df

    satir = {"symbol": "LMT", "bar_ts": str(df.index[0]), "direction": "+", "action": "WAIT",
             "confidence": 0.5, "prob_cal": 0.5, "price": 100.0, "atr_pct": 0.1}
    assert pt.record([satir], "1m", horizon=5) == 1
    assert pt.resolve(provider=_Barlar())["resolved"] == 1
    row = _oku(defter)[-1]
    assert float(row["fwd_price"]) == 102.0 and row["hit"] == "1"


def test_ufuk_kurali_kaynakta():
    src = inspect.getsource(pt._forward_price)
    assert "searchsorted(hedef" in src and "INTERVAL_MINUTES" in src


# ======================================== 2. defter: yayın ölçeği `atr_pct` (BULGU-84-2)
def test_atr_pct_kolonu_sona_eklendi():
    assert pt.FIELDS[-1] == "atr_pct"
    assert pt.FIELDS.index("atr_pct") > pt.FIELDS.index("conf_live")   # araya değil SONA


@pytest.mark.parametrize("deger,beklenen", [(0.123456789, "0.123457"), (None, ""),
                                            (float("nan"), ""), (0.0, ""), (-1.0, ""),
                                            ("x", "")])
def test_record_atr_pct(defter, deger, beklenen):
    satir = {"symbol": "AAPL", "bar_ts": "2026-09-18 14:00:00", "direction": "+",
             "action": "WAIT", "confidence": 0.5, "prob_cal": 0.5, "price": 100.0,
             "atr_pct": deger}
    assert pt.record([satir], "1m") == 1
    assert _oku(defter)[-1]["atr_pct"] == beklenen


def test_eski_basliktan_kayipsiz_goc(defter):
    eski = [f for f in pt.FIELDS if f != "atr_pct"]
    with defter.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=eski)
        w.writeheader()
        w.writerow({**dict.fromkeys(eski, ""), "logged_at": "2026-09-18T17:00:00",
                    "symbol": "X", "interval": "1m", "bar_ts": "2026-09-18 13:59:00",
                    "direction": "+", "price": "10", "horizon_bars": "5"})
    satir = {"symbol": "X", "bar_ts": "2026-09-18 14:00:00", "direction": "+", "action": "WAIT",
             "confidence": 0.5, "prob_cal": 0.5, "price": 10.0, "atr_pct": 0.1}
    assert pt.record([satir], "1m") == 1
    with defter.open(encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == pt.FIELDS
    rows = _oku(defter)
    assert (rows[0]["symbol"], rows[0]["price"], rows[0]["atr_pct"]) == ("X", "10", "")
    assert rows[1]["atr_pct"] == "0.1"


# ============================================== 3. aralık: yükleme + uygulama (BULGU-84-2)
# Tur-85: geçerli şema ölçek üssünü taşır (`olcek` adı değişti, `us` zorunlu); us=1,0 = ATR14'ün kendisi
# → bu dosyanın bütün sayıları Tur-84'teki gibi kalır.
_BANT = {"meta": {"olcek": "atr14_yuzde_us", "durum": "GECTI", "referans": "bar_kapanisi",
                  "interval": "1m", "horizon": 5, "yuruyen": {"kapsama": 0.79}},
         "us": 1.0,
         "havuz": {"z_lo": -2.8, "z_hi": 2.7, "z_med_abs": 1.3, "n": 1000},
         "sembol": {"AAPL": {"z_lo": -2.0, "z_hi": 2.0, "z_med_abs": 1.0, "n": 400}}}


def test_apply_band_sembol_ya_da_havuz():
    assert lps._apply_band(_BANT, "AAPL", 0.1) == pytest.approx((-0.2, 0.2, 0.1))
    assert lps._apply_band(_BANT, "LMT", 0.1) == pytest.approx((-0.28, 0.27, 0.13))
    for a in (None, 0.0, -0.1, float("nan"), float("inf")):
        assert lps._apply_band(_BANT, "AAPL", a) is None
    assert lps._apply_band(None, "AAPL", 0.1) is None


@pytest.fixture
def veri(tmp_path, monkeypatch):
    monkeypatch.setattr("nihai.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(lps, "_BAND_ONBELLEK", None)
    return tmp_path


def _yaz_bant(dizin, bant):
    (dizin / "forecast_band.json").write_text(json.dumps(bant), encoding="utf-8")


def test_load_band_gecerli(veri):
    _yaz_bant(veri, _BANT)
    assert lps._load_band("1m")["havuz"]["z_lo"] == -2.8


@pytest.mark.parametrize("bozuk", [{"olcek": "atr"}, {"durum": "KAPI_GECILEMEDI"},
                                   {"referans": "yayin_fiyati"}, {"interval": "1h"},
                                   {"horizon": 7}])
def test_load_band_alani_eslesmeyen_reddedilir(veri, bozuk):
    _yaz_bant(veri, {**_BANT, "meta": {**_BANT["meta"], **bozuk}})
    assert lps._load_band("1m") is None


def test_load_band_yok_ya_da_kuantilsiz(veri):
    assert lps._load_band("1m") is None                       # dosya yok
    _yaz_bant(veri, {"meta": _BANT["meta"]})                  # GECTI ama kuantil yok
    assert lps._load_band("1m") is None


def test_load_band_taze_okur(veri):
    _yaz_bant(veri, _BANT)
    assert lps._load_band("1m") is not None
    _yaz_bant(veri, {**_BANT, "meta": {**_BANT["meta"], "durum": "KAPI_GECILEMEDI",
                                      "not": "kapı bu gece geçilemedi"}})
    assert lps._load_band("1m") is None                       # süreç yeniden başlamadan


class _Sag:
    """Sahte sağlayıcı: 1m çerçeve, 1d çerçeve (ayrı), sabit kotasyon; çağrılan aralıkları sayar."""

    def __init__(self, df, last, gunluk=None):
        self.df, self.last, self.gunluk, self.name = df, last, gunluk, "sahte"
        self.cagri: list[str] = []

    def get_bars(self, _sym, interval, lookback_bars=200):
        self.cagri.append(interval)
        return self.gunluk if interval == "1d" else self.df

    def get_quote(self, _sym):
        return SimpleNamespace(last=self.last, bid=self.last - 0.01, ask=self.last + 0.01)


def _servis(monkeypatch, *, bant=None, e20=100.0, e50=100.0, day_open=98.0):
    """Bar kapanışı 99,5 (aralığın ÇAPASI); kotasyon 100,0 (gösterim); ATR 0,5 → ölçek 0,5/99,5."""
    monkeypatch.setenv("SIGNAL_ON_BAR_CLOSE", "true")
    idx = pd.date_range("2026-09-18 17:00", periods=80, freq="1min")
    df = pd.DataFrame({"open": 99.5, "high": 99.5, "low": 99.5, "close": 99.5,
                       "volume": 1.0}, index=idx)
    svc = lps.PredictionService()
    svc._provider = _Sag(df, 100.0)
    svc._gate = SimpleNamespace(model_id=None)
    svc._track = False
    svc._cycle_market_open = True
    svc._cycle_regime = None
    svc._calib = None
    svc._band = bant
    sig = SimpleNamespace(action=SignalAction.WAIT, price=99.5, stop=0.0, target=0.0,
                          rr_ratio=0.0, quality_score=0.0, metadata={}, reasons=[])
    svc._sig_cache["AAPL"] = {"ts": str(df.index[-1]), "sig": sig, "last_close": 99.5,
                              "e20": e20, "e50": e50, "r": 50.0, "a": 0.5,
                              "adx_v": 20.0, "htf_dir": 0, "day_open": day_open}
    return svc


def test_satir_aralik_uretir_nokta_tahmini_yok(monkeypatch):
    row = _servis(monkeypatch, bant=_BANT)._row("AAPL")
    assert "projected" not in row
    olcek = 0.5 / 99.5 * 100.0                                  # bar kapanışına göre (kotasyona değil)
    assert row["atr_pct"] == pytest.approx(olcek, abs=1e-6)
    assert row["band_lo"] == pytest.approx(round(99.5 * (1 - 2.0 * olcek / 100), 2))   # AAPL z ±2
    assert row["band_hi"] == pytest.approx(round(99.5 * (1 + 2.0 * olcek / 100), 2))
    assert row["exp_move_pct"] == pytest.approx(round(olcek, 2))  # z_med_abs 1,0 × ölçek (yönsüz)
    assert row["band_lo"] < row["bar_close"] < row["band_hi"]


def test_satir_aralik_yoksa_uydurmaz_ama_olcegi_loglar(monkeypatch):
    row = _servis(monkeypatch, bant=None)._row("AAPL")
    assert row["band_lo"] is None and row["band_hi"] is None and row["exp_move_pct"] is None
    assert row["atr_pct"] == pytest.approx(0.5 / 99.5 * 100.0, abs=1e-6)   # kuantil kurulabilsin


def test_aralik_yon_iddia_etmez(monkeypatch):
    asagi = _servis(monkeypatch, bant=_BANT, e20=100.0)._row("AAPL")
    yukari = _servis(monkeypatch, bant=_BANT, e20=99.0, e50=98.0)._row("AAPL")
    assert (asagi["direction"], yukari["direction"]) == ("-", "+")
    for k in ("band_lo", "band_hi", "exp_move_pct"):
        assert asagi[k] == yukari[k], k


# ========================================= 4. gün-içi değişim SEANS açılışından (BULGU-84-3)
def _gunluk(satirlar):
    idx = pd.DatetimeIndex([pd.Timestamp(g) + pd.Timedelta(hours=4) for g, _ in satirlar])
    acilis = [o for _, o in satirlar]
    return pd.DataFrame({"open": acilis, "high": acilis, "low": acilis, "close": acilis}, index=idx)


def test_satir_degisim_seans_acilisindan(monkeypatch):
    row = _servis(monkeypatch, day_open=98.0)._row("AAPL")
    assert row["chg_pct"] == pytest.approx(round((100.0 - 98.0) / 98.0 * 100.0, 2))
    assert _servis(monkeypatch, day_open=None)._row("AAPL")["chg_pct"] is None


def test_seans_acilisi_bugunun_gunluk_barindan():
    prov = _Sag(None, 0.0, gunluk=_gunluk([("2026-09-17", 90.0), ("2026-09-18", 95.0)]))
    assert lps._seans_acilisi(prov, "AAPL", date(2026, 9, 18)) == 95.0
    assert lps._seans_acilisi(prov, "AAPL", date(2026, 9, 19)) is None   # seans öncesi: bar yok


def test_seans_acilisi_hata_ve_bos():
    class _Patlak:
        def get_bars(self, *a, **k):
            raise RuntimeError("hub kapalı")

    assert lps._seans_acilisi(_Patlak(), "X", date(2026, 9, 18)) is None
    assert lps._seans_acilisi(_Sag(None, 0.0, gunluk=pd.DataFrame()), "X",
                              date(2026, 9, 18)) is None


def test_et_gunu_utc_ve_tz():
    assert lps._et_gunu("2026-09-18 03:30:00") == date(2026, 9, 17)   # 23:30 ET, önceki gün
    assert lps._et_gunu("2026-09-18 13:30:00") == date(2026, 9, 18)
    assert lps._et_gunu(pd.Timestamp("2026-09-18 01:00", tz="America/New_York")) == date(2026, 9, 18)


def test_gun_acilisi_gunde_bir_cekim_basarisiz_kisa_sure_bekletilir(monkeypatch):
    """Başarı gün boyu önbellekte; başarısızlık (seans öncesi) `_ACILIS_NEG_TTL` boyunca YENİDEN
    ÇEKİLMEZ (her yeni barda 30 sembolün 1d çekimi — Gemini incelemesi) ama gün boyu da kalmaz."""
    import time
    svc = _servis(monkeypatch)
    prov = _Sag(None, 0.0, gunluk=_gunluk([("2026-09-17", 90.0)]))
    svc._provider = prov
    assert svc._gun_acilisi("AAPL", "2026-09-18 13:00:00") is None       # bugünün barı henüz yok
    prov.gunluk = _gunluk([("2026-09-17", 90.0), ("2026-09-18", 95.0)])
    assert svc._gun_acilisi("AAPL", "2026-09-18 13:01:00") is None       # TTL içinde: çekim YOK
    assert prov.cagri.count("1d") == 1
    gun, _, _ = svc._acilis_memo["AAPL"]
    svc._acilis_memo["AAPL"] = (gun, None, time.monotonic() - lps._ACILIS_NEG_TTL - 1)
    assert svc._gun_acilisi("AAPL", "2026-09-18 13:31:00") == 95.0       # TTL doldu → yeniden
    n = prov.cagri.count("1d")
    for m in range(32, 59):
        assert svc._gun_acilisi("AAPL", f"2026-09-18 13:{m:02d}:00") == 95.0
    assert prov.cagri.count("1d") == n                                    # gün içinde tekrar yok


@pytest.mark.parametrize("dosya", ["forecast_band.json", "forecast_calibration.json"])
def test_gecici_okuma_hatasi_onbellege_yazilmaz(veri, monkeypatch, dosya):
    """OneDrive/AV kilidi gibi GEÇİCİ bir okuma hatası imzayla önbelleğe yazılsaydı dosya bir
    sonraki gece değişene kadar aralık/eğri kapalı kalırdı (Gemini incelemesi; iki yükleyici)."""
    for ad, v in (("_CALIB_SIG", None), ("_CALIB_PTS", None), ("_CALIB_META", {}),
                  ("_CALIB_SEEN", False)):
        monkeypatch.setattr(lps, ad, v)
    monkeypatch.delenv("SIGNAL_ON_BAR_CLOSE", raising=False)
    _yaz_bant(veri, _BANT)
    (veri / "forecast_calibration.json").write_text(json.dumps(
        {"points": [[0.0, 0.3], [1.0, 0.7]],
         "meta": {"girdi": "yayinlanan_guven", "yon": "bar", "interval": "1m"}}), encoding="utf-8")
    yukle = lps._load_band if dosya == "forecast_band.json" else lps._load_calibration
    asil = Path.read_text
    kilit = {"kalan": 1}

    def kilitli(self, *a, **k):
        if self.name == dosya and kilit["kalan"]:
            kilit["kalan"] -= 1
            raise PermissionError("dosya kilitli")
        return asil(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", kilitli)
    assert yukle("1m") is None                                   # kilitli okuma → bu çağrı "—"
    assert yukle("1m") is not None                               # dosya DEĞİŞMEDEN yeniden dener


def test_kapali_piyasada_aralik_gosterilmez(monkeypatch):
    """Kuantiller AÇIK seansın 5 dk'lık hareketinden kuruldu; kapalıyken alan dışı → '—'."""
    svc = _servis(monkeypatch, bant=_BANT)
    svc._cycle_market_open = False
    row = svc._row("AAPL")
    assert row["band_lo"] is None and row["band_hi"] is None and row["exp_move_pct"] is None
    assert row["atr_pct"] is not None


def test_eski_gece_yarisi_formulu_iki_yuzeyde_de_yok():
    for yol in ("nihai/execution/live_predict_service.py", "nihai/web/web_dashboard.py"):
        kod = _kod(yol)
        assert "normalize ( )" not in kod, yol
        assert "_seans_acilisi" in kod, yol


# ===================================================== 5. hisse detayı (/api/analyze)
def test_hisse_detay_aralik_ve_seans_acilisi(veri, monkeypatch):
    from nihai.web import web_dashboard as wd
    _yaz_bant(veri, _BANT)
    idx = pd.date_range("2026-09-18 12:00", periods=120, freq="1min")   # 08:00 ET: seans öncesi
    df = pd.DataFrame({"open": np.linspace(90.0, 99.0, 120), "high": 100.2, "low": 99.8,
                       "close": 100.0, "volume": 1000.0}, index=idx)
    gunluk = _gunluk([("2026-09-17", 97.0), ("2026-09-18", 98.0)])
    monkeypatch.setattr(wd, "_analysis_provider", lambda: _Sag(df, 99.0, gunluk=gunluk))
    monkeypatch.setattr(wd, "is_market_open", lambda: True)
    d = wd.analyze_symbol("AAPL", "1m")
    assert d["ok"] and "projected" not in d and "projected_atr" not in d["fair_value"]
    assert d["chg_pct"] == pytest.approx(round((99.0 - 98.0) / 98.0 * 100.0, 2))  # 90 değil
    # Çapa bar kapanışı (100), kotasyon (99) DEĞİL — İKİ uç da tam değerle: yalnız sıralamaya bakan
    # eski kilit, `band_lo`yu tek başına kotasyona çapalayan mutasyonu kaçırıyordu (98,21 < 100 < 100,8).
    olcek = float(atr(df, 14).iloc[-1]) / 100.0 * 100.0
    assert d["band_lo"] == pytest.approx(round(100.0 * (1 - 2.0 * olcek / 100), 2))   # AAPL z ±2
    assert d["band_hi"] == pytest.approx(round(100.0 * (1 + 2.0 * olcek / 100), 2))
    assert d["exp_move_pct"] == pytest.approx(round(olcek, 2))
    h = wd.analyze_symbol("AAPL", "1h")
    assert h["band_lo"] is None and h["exp_move_pct"] is None      # yalnız kendi aralığında
    monkeypatch.setattr(wd, "is_market_open", lambda: False)
    k = wd.analyze_symbol("AAPL", "1m")
    assert k["band_lo"] is None and k["exp_move_pct"] is None      # kapalı piyasa: alan dışı
    assert k["chg_pct"] == d["chg_pct"]                             # gün değişimi yine verilir


# ========================================= 6. gece görevi kurucusu (yürüyen kapı)
def _satirlar(gun_sayisi=8, n_gun=400, *, bilgili=True, tohum=3, semboller=("AAA", "BBB")):
    """z ~ t(5); getiri = z × ATR% (bilgili) ya da ATR%'den BAĞIMSIZ (bilgisiz ölçek)."""
    rng = np.random.default_rng(tohum)
    out = []
    for d in range(gun_sayisi):
        g = f"2026-09-{d + 1:02d}"
        for i in range(n_gun):
            a = float(rng.uniform(0.02, 0.4))                   # ölçek 20 kat değişir
            z = float(rng.standard_t(5))
            out.append((g, semboller[i % len(semboller)], z * a if bilgili else z * 0.1, a))
    return out


def test_bilgili_olcek_kapiyi_gecer():
    out = cf.bant_kalibre_et(_satirlar(), kaynak="test", interval="1m", horizon=5)
    m, k = out["meta"], out["meta"]["yuruyen"]
    assert m["durum"] == "GECTI" and k["gun_t"] < -cf.T_ESIK
    assert 0.74 < k["kapsama"] < 0.86                           # nominal %80 civarı
    assert set(out["sembol"]) == {"AAA", "BBB"} and out["havuz"]["n"] == 3200
    assert m["olcek"] == "atr14_yuzde_us" and m["interval"] == "1m" and m["horizon"] == 5


def test_bilgisiz_olcek_kuantil_yayinlanmaz():
    out = cf.bant_kalibre_et(_satirlar(bilgili=False), kaynak="test", interval="1m", horizon=5)
    assert out["meta"]["durum"] == "KAPI_GECILEMEDI"
    assert "havuz" not in out and "sembol" not in out


def test_yetersiz_test_gunu_hukum_yok():
    assert cf.bant_kalibre_et(_satirlar(gun_sayisi=7), kaynak="t", interval="1m",
                              horizon=5) is None                # 3 ısınma + 4 test < 5
    assert cf.bant_kalibre_et([], kaynak="t", interval="1m", horizon=5) is None


def test_bant_yuruyen_nedensel_yalniz_gecmisten(monkeypatch):
    boylar: list[int] = []
    asil = cf.bant_modeli
    monkeypatch.setattr(cf, "bant_modeli",
                        lambda s, us=1.0: (boylar.append(len(s)), asil(s, us))[1])
    k = cf.bant_yuruyen(_satirlar(gun_sayisi=6, n_gun=100), isinma_gun=3, min_test_gun=1)
    # gün 4/5/6 → önceki 3/4/5 gün; Tur-85'ten beri iki aday (sabit ATR + seçilen üs) aynı sırayla yürür
    assert boylar == [300, 400, 500] * 2
    assert k["test_gun"] == 3 and k["n_test"] == 300


def test_bant_yuruyen_yayinlanan_temsili_puanlar():
    assert "_apply_band" in inspect.getsource(cf._yuruyen)


def test_bant_modeli_sembol_esigi():
    n = cf.BANT_MIN_SEMBOL
    sat = [("g", "AZ", 0.1, 0.1)] * (n - 1) + [("g", "COK", 0.01 * i, 0.1) for i in range(n)]
    m = cf.bant_modeli(sat)
    assert "COK" in m["sembol"] and "AZ" not in m["sembol"]
    assert m["havuz"]["n"] == 2 * n - 1


def test_aralik_skoru_elle():
    """lo=−1, hi=1, α=0,2: içeride genişlik 2; y=2 → 2 + 10·1; y=−3 → 2 + 10·2."""
    assert cf._aralik_skoru(-1.0, 1.0, 0.0) == pytest.approx(2.0)
    assert cf._aralik_skoru(-1.0, 1.0, 2.0) == pytest.approx(12.0)
    assert cf._aralik_skoru(-1.0, 1.0, -3.0) == pytest.approx(22.0)


def test_bant_satirlari_filtre_ve_bar_kapanisindan():
    """Getiri aralığın çapası olan BAR KAPANIŞINDAN ölçülür, kotasyondan (`price`) değil."""
    iyi = {"logged_at": "2026-09-18T17:00:00", "symbol": "AAPL", "interval": "1m",
           "horizon_bars": "5", "hit": "1", "price": "101", "bar_close": "100",
           "fwd_price": "100.5", "atr_pct": "0.2"}
    kotu = [{**iyi, "atr_pct": ""}, {**iyi, "atr_pct": "0"}, {**iyi, "hit": ""},
            {**iyi, "interval": "1h"}, {**iyi, "horizon_bars": "3"}, {**iyi, "bar_close": ""},
            {**iyi, "bar_close": "x"}]
    s = cf.bant_satirlari([iyi, *kotu], "1m", 5)
    assert len(s) == 1
    g, sym, r, a = s[0]
    assert (g, sym, a) == ("2026-09-18", "AAPL", 0.2) and r == pytest.approx(0.5)


def _defter_satirlari(sat):
    rows = []
    for i, (g, s, r, a) in enumerate(sat):
        fwd = 100.0 * (1 + r / 100.0)
        rows.append({"logged_at": f"{g}T17:{(i // 60) % 60:02d}:{i % 60:02d}", "symbol": s,
                     "interval": "1m", "horizon_bars": "5", "hit": "1" if fwd > 100 else "0",
                     "direction": "+", "confidence": "0.5", "dir_bar": "+", "conf_bar": "0.5",
                     "dir_live": "", "conf_live": "", "price": "100.0", "bar_close": "100.0",
                     "fwd_price": f"{fwd:.6f}", "atr_pct": f"{a:.6f}"})
    return rows


def _yaz_defter(yol, rows):
    with yol.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_cli_aralik_dosyasini_yazar(tmp_path, monkeypatch):
    monkeypatch.delenv("SIGNAL_ON_BAR_CLOSE", raising=False)
    log, bant = tmp_path / "pl.csv", tmp_path / "band.json"
    _yaz_defter(log, _defter_satirlari(_satirlar()))
    assert cf.main(["--log", str(log), "--out", str(tmp_path / "fc.json"), "--bant-out",
                    str(bant), "--interval", "1m", "--horizon", "5"]) == 0
    d = json.loads(bant.read_text(encoding="utf-8"))
    assert d["meta"]["durum"] == "GECTI" and d["meta"]["kaynak"] == "canli_defter"


def test_cli_yetersizde_mevcut_araliga_dokunmaz(tmp_path, monkeypatch):
    monkeypatch.delenv("SIGNAL_ON_BAR_CLOSE", raising=False)
    log, bant = tmp_path / "pl.csv", tmp_path / "band.json"
    _yaz_defter(log, _defter_satirlari(_satirlar(gun_sayisi=4, n_gun=100)))
    bant.write_text("ONYUKLEME", encoding="utf-8")
    assert cf.main(["--log", str(log), "--out", str(tmp_path / "fc.json"), "--bant-out",
                    str(bant), "--interval", "1m", "--horizon", "5"]) == 0
    assert bant.read_text(encoding="utf-8") == "ONYUKLEME"


def test_barlardan_onyukleme_iex_ham_ve_zaman_kurali(monkeypatch):
    df = _seri([*range(31), 31, 32, 40, 41, 42])                # 30. dakikadan sonra boşluk
    cagri = []

    def sahte(sym, interval, *, start, end, feed, use_cache):
        cagri.append((sym, interval, feed, use_cache, os.environ.get("ALPACA_HIST_ADJUSTMENT")))
        return df

    monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", "all")
    rows = [{"logged_at": "2026-09-18T17:31:39", "symbol": "LMT", "interval": "1m",
             "horizon_bars": "5", "hit": "1", "bar_ts": str(df.index[30]), "price": "131.0"}]
    out = cf.barlardan_bant_satirlari(rows, "1m", 5, get_history=sahte)
    assert cagri == [("LMT", "1m", "iex", False, "raw")]        # canlı hub'ın serisi, düzeltmesiz
    assert os.environ["ALPACA_HIST_ADJUSTMENT"] == "all"        # geri yüklendi
    g, sym, r, a = out[0]
    assert r == pytest.approx((132.0 - 130.0) / 130.0 * 100.0)  # çapa bar kapanışı (130), 131 değil
    assert a == pytest.approx(float(atr(df, 14).iloc[30]) / 130.0 * 100.0)


# ===================================== 7. alpaca_history önbellek anahtarı feed taşır
def test_onbellek_anahtari_feed_tasir(monkeypatch, tmp_path):
    from nihai.data import alpaca_history as ah
    monkeypatch.setattr(ah, "HIST_DIR", tmp_path)
    monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", "raw")
    assert ah._cache_path("AAPL", "1m").name == "AAPL_1m.csv"          # SIP: eski ad korunur
    assert ah._cache_path("AAPL", "1m", "iex").name == "AAPL_1m_iex.csv"
    monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", "all")
    assert ah._cache_path("AAPL", "1m", "iex").name == "AAPL_1m_all_iex.csv"


def test_iex_istegi_sip_onbellegini_okumaz_ve_ezmez(monkeypatch, tmp_path):
    from nihai.data import alpaca_history as ah
    monkeypatch.setattr(ah, "HIST_DIR", tmp_path)
    monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", "raw")
    idx = pd.date_range("2026-09-01 14:00", periods=3, freq="1min")
    sip = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
                       index=idx)
    sip.to_csv(ah._cache_path("AAPL", "1m"))                  # taze SIP önbelleği
    iex = sip.assign(close=2.0)
    monkeypatch.setattr(ah, "_fetch", lambda s, i, st, en, feed: iex if feed == "iex" else sip)
    got = ah.get_history("AAPL", "1m", start="2026-09-01", end="2026-09-02", feed="iex")
    assert (got["close"] == 2.0).all()
    assert (pd.read_csv(ah._cache_path("AAPL", "1m"), index_col=0)["close"] == 1.0).all()
    assert ah._cache_path("AAPL", "1m", "iex").exists()


# ================================================================ 8. pano sözleşmesi
def test_pano_nokta_ongoruyu_gostermez_araligi_gosterir():
    html = (KOK / "nihai/web/web_dashboard.html").read_text(encoding="utf-8")
    assert "r.projected" not in html and "d.projected" not in html
    assert "sortPredict('projected')" not in html
    assert "r.band_lo" in html and "d.band_lo" in html and "d.band_hi" in html
    assert "m.band_coverage" in html
    assert "d.chg_pct == null" in html                          # seans açılışı yoksa '—'
