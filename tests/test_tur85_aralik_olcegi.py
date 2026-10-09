"""Tur-85 — ARALIĞIN ÖLÇEĞİ: ATR14^us; üs gece görevinde yalnız geçmişten seçilir ve ancak sabit ATR'yi
yürüyen sınavda yenerse yayınlanır.

Ölçümler: docs/TUR85_RAPORU_2026-09-19.md.
  BULGU-85-0  20 aday ölçeğin (gerçekleşen varyans 10–120 dk, zaman-normalize TR², EWMA, önceki gün RV,
              seans-içi ATR) HİÇBİRİ ATR14'ü yenmedi — "daha hızlı ölçek" (Tur-84 açık kalemi) çürüdü.
  BULGU-85-1  ATR14 aralığı sembol-içi ATR ondalığında %70,6 (en sakin) → %86,4 (en oynak) kapsıyordu
              (ortalamaya dönüş). ATR^us (düzeye büzme): Haziran–Eylül −0,63 bps (t=−3,51, 29/32 gün);
              bulguların hiç bakılmadığı Mart–Mayıs'ta −0,53 bps (t=−8,46, 60/60 gün us=0,8).
  BULGU-85-2  Son yarım saat için kapanış çarpanı Haziran–Eylül'de kapıyı geçti (t=−2,75) ama örneklem-dışı
              dönemde BİR KEZ bile seçilmedi → eklenmedi (aşağıdaki regresyon kilidi).
"""
from __future__ import annotations

import inspect
import io
import json
import tokenize
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nihai.execution import live_predict_service as lps
from nihai.signals.strategy import atr
from research import calibrate_forecast as cf

KOK = Path(__file__).resolve().parents[1]

_Q = {"z_lo": -2.0, "z_hi": 2.0, "z_med_abs": 1.0, "n": 400}


def _bant(us=0.8, **meta):
    return {"meta": {"olcek": "atr14_yuzde_us", "durum": "GECTI", "referans": "bar_kapanisi",
                     "interval": "1m", "horizon": 5, "yuruyen": {"kapsama": 0.8}, **meta},
            "us": us, "havuz": {"z_lo": -2.8, "z_hi": 2.7, "z_med_abs": 1.3, "n": 1000},
            "sembol": {"AAPL": dict(_Q)}}


def _kod(yol: str) -> str:
    """Kaynağın YORUM ve DİZGİ içermeyen hâli — kilit anlatıya takılmasın (Tur-49)."""
    kaynak = (KOK / yol).read_text(encoding="utf-8")
    return " ".join(t.string for t in tokenize.generate_tokens(io.StringIO(kaynak).readline)
                    if t.type not in (tokenize.COMMENT, tokenize.STRING))


# ============================================================ 1. uygulama: ölçek ATR^us
def test_apply_band_us_uygular():
    a = 0.25
    s = a ** 0.8
    assert lps._apply_band(_bant(0.8), "AAPL", a) == pytest.approx((-2.0 * s, 2.0 * s, 1.0 * s))
    assert lps._apply_band(_bant(0.8), "LMT", a) == pytest.approx((-2.8 * s, 2.7 * s, 1.3 * s))


def test_apply_band_us_yoksa_atrnin_kendisi():
    """Saf fonksiyon: `us` taşımayan model = Tur-84 (ölçek ATR14'ün kendisi)."""
    b = {k: v for k, v in _bant().items() if k != "us"}
    assert lps._apply_band(b, "AAPL", 0.25) == pytest.approx((-0.5, 0.5, 0.25))
    assert lps._apply_band(_bant(1.0), "AAPL", 0.25) == lps._apply_band(b, "AAPL", 0.25)


@pytest.mark.parametrize("us", [0.0, -0.8, float("nan"), float("inf"), 2.5, "x", None])
def test_apply_band_gecersiz_us_uydurmaz(us):
    assert lps._apply_band(_bant(us), "AAPL", 0.25) is None


# ============================================================ 2. şema kapısı
@pytest.fixture
def veri(tmp_path, monkeypatch):
    monkeypatch.setattr("nihai.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(lps, "_BAND_ONBELLEK", None)
    return tmp_path


def _yaz(dizin, bant):
    (dizin / "forecast_band.json").write_text(json.dumps(bant), encoding="utf-8")


def test_load_band_us_li_dosya_yuklenir(veri):
    _yaz(veri, _bant(0.8))
    b = lps._load_band("1m")
    assert b is not None and b["us"] == 0.8


def test_load_band_tur84_dosyasi_reddedilir(veri):
    """Üssü bilmeyen kod yeni dosyayı yanlış ölçekte uygulamasın diye `olcek` adı değişti; eski ad
    (ATR'yi ham ölçek sayan Tur-84 önyüklemesi) artık yüklenmez — gece görevi yenisini yazar."""
    eski = _bant(0.8)
    eski["meta"]["olcek"] = "atr14_yuzde"
    del eski["us"]
    _yaz(veri, eski)
    assert lps._load_band("1m") is None


@pytest.mark.parametrize("olcek", ["atr14_yuzde", "atr14_yuzde_us_dilim", "rv30_yuzde_us"])
def test_load_band_baska_olcek_adi_us_olsa_da_reddedilir(veri, olcek):
    """Ad, ölçeğin TANIMIDIR: `us` taşısa bile başka bir ölçek tanımının kuantilleri ATR14^us'a
    uygulanamaz (yukarıdaki test eski dosyayı `us`suz kurduğu için adı ayrıca sınamıyordu)."""
    b = _bant(0.8)
    b["meta"]["olcek"] = olcek
    _yaz(veri, b)
    assert lps._load_band("1m") is None


@pytest.mark.parametrize("us", ["YOK", 0.0, 5.0, "x", None])
def test_load_band_us_eksik_ya_da_gecersiz_reddedilir(veri, us):
    b = _bant(0.8)
    if us == "YOK":
        del b["us"]
    else:
        b["us"] = us
    _yaz(veri, b)
    assert lps._load_band("1m") is None


# ============================================================ 3. kurucu: model + üs seçimi
def _satirlar(gun_sayisi=10, n_gun=600, *, beta=1.0, tohum=7, semboller=("AAA", "BBB")):
    """z ~ t(5); getiri = z × ATR%^beta; ATR% 20 kat değişir → üs bilgisi taşır."""
    rng = np.random.default_rng(tohum)
    out = []
    for d in range(gun_sayisi):
        g = f"2026-09-{d + 1:02d}"
        for i in range(n_gun):
            a = float(rng.uniform(0.02, 0.4))
            out.append((g, semboller[i % len(semboller)], float(rng.standard_t(5)) * a ** beta, a))
    return out


def test_bant_modeli_us_z_tanimi():
    sat = _satirlar(gun_sayisi=2, n_gun=400)
    m1, m5 = cf.bant_modeli(sat), cf.bant_modeli(sat, 0.5)
    z1 = np.array([r / a for _, _, r, a in sat])
    z5 = np.array([r / a ** 0.5 for _, _, r, a in sat])
    assert m1["us"] == 1.0 and m5["us"] == 0.5
    assert m1["havuz"]["z_lo"] == round(float(np.quantile(z1, 0.1)), 4)   # us=1: Tur-84 ile birebir
    assert m5["havuz"]["z_hi"] == round(float(np.quantile(z5, 0.9)), 4)
    assert m1["sembol"]["AAA"]["z_lo"] == round(float(np.quantile(z1[0::2], 0.1)), 4)   # sembol kodu


def test_us_secimi_gercek_ussu_bulur():
    assert cf.bant_us_sec(_satirlar(beta=0.5)) == 0.5
    assert cf.bant_us_sec(_satirlar(beta=1.0)) == 1.0


def test_us_secimi_esitlikte_bire_doner():
    """Ölçek sabitse bütün üsler aynı aralığı verir → en basit olan (1,0)."""
    rng = np.random.default_rng(1)
    sat = [("g", "AAA", float(rng.standard_t(5)), 1.0) for _ in range(900)]
    assert cf.bant_us_sec(sat) == 1.0


def test_secim_olcutu_yayinlanan_temsili_puanlar():
    """Üs seçimi dizide puanlar; puanladığı şey servisin `_apply_band` temsiliyle BİREBİR olmalı."""
    sat = _satirlar(gun_sayisi=3, n_gun=400, beta=0.7)
    d = cf._bant_diziler(sat)
    for us in (1.0, 0.7):
        m = cf.bant_modeli(sat, us)
        tek = [float(cf._aralik_skoru(*lps._apply_band(m, s, x)[:2], y)) for _, s, y, x in sat]
        assert cf._ic_skor(*d, us) == pytest.approx(float(np.mean(tek)), rel=1e-12)


def test_aralik_skoru_skaler_ve_dizi_ayni():
    lo, hi, y = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]), np.array([0.0, 2.0, -3.0])
    dizi = cf._aralik_skoru(lo, hi, y)
    assert list(dizi) == [float(cf._aralik_skoru(-1.0, 1.0, v)) for v in y] == [2.0, 12.0, 22.0]


# ============================================================ 4. yürüyen sınav + yayın kapısı
def test_us_secimi_yalniz_gecmisten(monkeypatch):
    boylar: list[int] = []
    asil = cf.bant_us_sec
    monkeypatch.setattr(cf, "bant_us_sec", lambda s: (boylar.append(len(s)), asil(s))[1])
    cf.bant_yuruyen(_satirlar(gun_sayisi=6, n_gun=100), isinma_gun=3, min_test_gun=1)
    assert boylar == [300, 400, 500]                       # gün 4/5/6 → önceki 3/4/5 gün


def test_bilgili_us_sabit_atryi_yener_ve_yayinlanir():
    sat = _satirlar(gun_sayisi=10, beta=0.5)
    k = cf.bant_yuruyen(sat)
    uk = k["us_kiyas"]
    assert uk["gecti"] and uk["gun_t"] < -cf.T_ESIK and uk["fark_bps"] < 0
    sabit = cf._yuruyen_ozeti(cf._yuruyen(sat, cf.ISINMA_GUN, us_sec=False), cf.MIN_TEST_GUN)
    assert k["skor_model_bps"] < sabit["skor_model_bps"]    # özet YAYINLANAN (seçilen üs) adayın
    out = cf.bant_kalibre_et(sat, kaynak="test", interval="1m", horizon=5)
    assert out["meta"]["olcek"] == "atr14_yuzde_us" and out["us"] == 0.5
    assert out["meta"]["durum"] == "GECTI"


def test_bilgisiz_us_yayinlanmaz_sabit_atr_kalir():
    sat = _satirlar(gun_sayisi=10, beta=1.0)
    assert not cf.bant_yuruyen(sat)["us_kiyas"]["gecti"]
    assert cf.bant_kalibre_et(sat, kaynak="test", interval="1m", horizon=5)["us"] == 1.0


def test_kapi_gecilmezse_tum_veri_secimi_de_yayinlanmaz(monkeypatch):
    """Örneklem-içi seçim ne derse desin, yürüyen sınavı geçmeyen üs yayınlanmaz."""
    asil = cf.bant_yuruyen

    def sinav(sat, **kw):
        k = asil(sat, **kw)
        k["us_kiyas"]["gecti"] = False
        return k

    monkeypatch.setattr(cf, "bant_yuruyen", sinav)
    monkeypatch.setattr(cf, "bant_us_sec", lambda s: 0.7)
    out = cf.bant_kalibre_et(_satirlar(gun_sayisi=10, beta=0.5), kaynak="t", interval="1m", horizon=5)
    assert out["us"] == 1.0


def test_adaylar_farkli_satir_puanlarsa_eslestirme_durur(monkeypatch):
    """Seçilen üs temsili geçersiz kılarsa (satırlar atlanır) eşleştirilmiş t anlamsızdır → dur."""
    monkeypatch.setattr(cf, "bant_us_sec", lambda s: 3.0)          # `_band_us` aralığı dışında
    with pytest.raises(RuntimeError, match="eşleştirme"):
        cf.bant_yuruyen(_satirlar(gun_sayisi=6, n_gun=100))


def test_cli_yazdigi_dosya_servis_kapisindan_gecer(tmp_path, monkeypatch, veri):
    """Uçtan uca: gece görevinin yazdığı aralık dosyası servisin şema kapısından geçer ve üssü taşır."""
    monkeypatch.delenv("SIGNAL_ON_BAR_CLOSE", raising=False)
    rows = []
    for i, (g, s, r, a) in enumerate(_satirlar(gun_sayisi=10, beta=0.5)):
        fwd = 100.0 * (1 + r / 100.0)
        rows.append({"logged_at": f"{g}T17:{(i // 60) % 60:02d}:{i % 60:02d}", "symbol": s,
                     "interval": "1m", "horizon_bars": "5", "hit": "1" if fwd > 100 else "0",
                     "direction": "+", "confidence": "0.5", "dir_bar": "+", "conf_bar": "0.5",
                     "price": "100.0", "bar_close": "100.0", "fwd_price": f"{fwd:.6f}",
                     "atr_pct": f"{a:.6f}"})
    log = tmp_path / "pl.csv"
    pd.DataFrame(rows).to_csv(log, index=False)
    assert cf.main(["--log", str(log), "--out", str(tmp_path / "fc.json"), "--bant-out",
                    str(veri / "forecast_band.json"), "--interval", "1m", "--horizon", "5"]) == 0
    b = lps._load_band("1m")
    assert b is not None and b["us"] == 0.5 and b["meta"]["yuruyen"]["us_kiyas"]["gecti"]


# ============================================================ 5. servis + hisse detayı uçtan uca
class _Sag:
    def __init__(self, df, last, gunluk):
        self.df, self.last, self.gunluk, self.name = df, last, gunluk, "sahte"

    def get_bars(self, _sym, interval, lookback_bars=200):
        return self.gunluk if interval == "1d" else self.df

    def get_quote(self, _sym):
        return SimpleNamespace(last=self.last, bid=self.last - 0.01, ask=self.last + 0.01)


def test_hisse_detay_us_li_aralik(veri, monkeypatch):
    from nihai.web import web_dashboard as wd
    _yaz(veri, _bant(0.5))
    idx = pd.date_range("2026-09-18 14:00", periods=120, freq="1min")
    df = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 1000.0},
                      index=idx)
    gunluk = pd.DataFrame({"open": [98.0], "high": [98.0], "low": [98.0], "close": [98.0]},
                          index=pd.DatetimeIndex([pd.Timestamp("2026-09-18 04:00")]))
    monkeypatch.setattr(wd, "_analysis_provider", lambda: _Sag(df, 99.0, gunluk))
    monkeypatch.setattr(wd, "is_market_open", lambda: True)
    d = wd.analyze_symbol("AAPL", "1m")
    s = (float(atr(df, 14).iloc[-1]) / 100.0 * 100.0) ** 0.5
    assert d["band_lo"] == pytest.approx(round(100.0 * (1 - 2.0 * s / 100), 2))
    assert d["band_hi"] == pytest.approx(round(100.0 * (1 + 2.0 * s / 100), 2))
    assert d["exp_move_pct"] == pytest.approx(round(s, 2))


def test_servis_satiri_us_li_aralik(monkeypatch):
    """Tahmin sekmesi: aralık ölçeği ATR%^us; deftere yazılan `atr_pct` HAM ATR% kalır (gece görevi
    dönüşümü kendisi uygular — defter şeması değişmedi)."""
    from nihai.signals.strategy import SignalAction
    monkeypatch.setenv("SIGNAL_ON_BAR_CLOSE", "true")
    idx = pd.date_range("2026-09-18 17:00", periods=80, freq="1min")
    df = pd.DataFrame({"open": 99.5, "high": 99.5, "low": 99.5, "close": 99.5, "volume": 1.0}, index=idx)
    svc = lps.PredictionService()
    svc._provider = _Sag(df, 100.0, None)
    svc._gate = SimpleNamespace(model_id=None)
    svc._track = False
    svc._cycle_market_open = True
    svc._cycle_regime = None
    svc._calib = None
    svc._band = _bant(0.5)
    sig = SimpleNamespace(action=SignalAction.WAIT, price=99.5, stop=0.0, target=0.0, rr_ratio=0.0,
                          quality_score=0.0, metadata={}, reasons=[])
    svc._sig_cache["AAPL"] = {"ts": str(df.index[-1]), "sig": sig, "last_close": 99.5, "e20": 100.0,
                              "e50": 100.0, "r": 50.0, "a": 0.5, "adx_v": 20.0, "htf_dir": 0,
                              "day_open": 98.0}
    row = svc._row("AAPL")
    ham = 0.5 / 99.5 * 100.0
    s = ham ** 0.5
    assert row["band_lo"] == pytest.approx(round(99.5 * (1 - 2.0 * s / 100), 2))
    assert row["band_hi"] == pytest.approx(round(99.5 * (1 + 2.0 * s / 100), 2))
    assert row["atr_pct"] == pytest.approx(ham, abs=1e-6)


# ============================================================ 6. regresyon kilidi (BULGU-85-2)
def test_ornek_disi_tekrarlanmayan_kapanis_carpani_yok():
    """Kapanış çarpanı Haziran–Eylül'de kapıyı geçti (t=−2,75), Mart–Mayıs'ta 60 günün HİÇBİRİNDE
    seçilmedi. Her gece yeniden sınanan boş bir özellik örtüşen veride er geç sahte bir "geçti" üretir →
    eklenmedi. Geri getirmek yeni bir ÖRNEKLEM-DIŞI kanıt ister (bu kilit o kararla birlikte değişir)."""
    assert "kapanis" not in cf.bant_modeli(_satirlar(gun_sayisi=1, n_gun=50))
    assert list(inspect.signature(lps._apply_band).parameters) == ["band", "symbol", "atr_pct"]
    for yol in ("research/calibrate_forecast.py", "nihai/execution/live_predict_service.py"):
        kod = _kod(yol)
        assert "BANT_KAPANIS" not in kod and "_band_kapanis" not in kod, yol
