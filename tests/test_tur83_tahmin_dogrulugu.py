"""Tur-83 — tahmin DOĞRULUĞU: yön girdisi, kalibre olasılık, beraberlik hükmü.

Üç bulgu kilitlenir (ölçümler: docs/TUR83_RAPORU_2026-09-18.md):
  BULGU-83-1  Yayınlanan yön canlı kotasyondan hesaplanıyordu; kotasyon bar kapanışından
              sapıp EMA20'nin öbür yanına geçtiğinde sapma geri dönüyor ve o yönler %26
              isabet ediyordu. AYNI satırlarda AYNI referanstan bar-yön +1,92 puan (t=+12,8).
              → varsayılan yön girdisi bar kapanışı (`_yon_kipi`), karşı varyant loglanır.
  BULGU-83-2  `hit_bar` canlı-yön için İLERİ-BAKIŞ taşıyordu (belge + gölge terfi yüzeyi).
  BULGU-83-3  "Olasılık (kal.)" eğrisi sabit taban orandan KÖTÜYDÜ (t=+4,38): örneklem-içi,
              yanlış hedef, yanlış girdi, 3 ay bayat. → canlı defterden yürüyen + kapılı eğri,
              alanı eşleşmeyen eğri REDDEDİLİR, gece görevi tazeler.
  +           Beraberlik `hit`te ıskadır → 0,5'e kıyaslayan hüküm beraberlik HARİÇ oranı okur.
"""
from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nihai.execution import live_predict_service as lps
from nihai.execution import prediction_tracker as pt
from nihai.signals.strategy import SignalAction
from research import calibrate_forecast as cf

KOK = Path(__file__).resolve().parents[1]


# ============================================================================ 1. yön kipi
def test_yon_kipi_varsayilan_bar(monkeypatch):
    monkeypatch.delenv("SIGNAL_ON_BAR_CLOSE", raising=False)
    assert lps._yon_kipi() == "bar"


@pytest.mark.parametrize("deger", ["false", "0", "off", "no", "FALSE", " Off "])
def test_yon_kipi_acikca_kapatilir(monkeypatch, deger):
    monkeypatch.setenv("SIGNAL_ON_BAR_CLOSE", deger)
    assert lps._yon_kipi() == "canli"


@pytest.mark.parametrize("deger", ["true", "1", "yes", "on", "flase", "hayir", ""])
def test_yon_kipi_yanlis_yazim_kovalayan_varyanta_dusmez(monkeypatch, deger):
    """Tanınmayan değer ÖLÇÜLMÜŞ olarak kötü varyanta (canlı) SESSİZCE düşemez."""
    monkeypatch.setenv("SIGNAL_ON_BAR_CLOSE", deger)
    assert lps._yon_kipi() == "bar"


# ================================================= 2. servis satırı — işlevsel (BULGU-83-1)
class _Sag:
    """Sahte sağlayıcı: düz bar çerçevesi + EMA20'nin öbür yanında bir kotasyon."""

    def __init__(self, df, last):
        self.df, self.last, self.name = df, last, "sahte"

    def get_bars(self, *a, **k):
        return self.df

    def get_quote(self, _sym):
        return SimpleNamespace(last=self.last, bid=self.last - 0.01, ask=self.last + 0.01)


def _servis(monkeypatch, kip: str, calib=None):
    """Bar kapanışı 99,5 (EMA20=100'ün ALTINDA) · kotasyon 100,5 (ÜSTÜNDE) · e20=e50, RSI 50.

    Bileşenler: bar → [0, 0, −1] → "−" · canlı → [0, 0, +1] → "+". HTF −1 (bar yönüyle
    UYUMLU) → yayınlanan güven bar kipinde ×1,15, canlı kipte ×0,65 — girdiyi ayırt eder."""
    monkeypatch.setenv("SIGNAL_ON_BAR_CLOSE", kip)
    idx = pd.date_range("2026-09-18 14:00", periods=80, freq="1min")
    df = pd.DataFrame({"open": 99.5, "high": 99.5, "low": 99.5, "close": 99.5,
                       "volume": 1.0}, index=idx)
    svc = lps.PredictionService()
    svc._provider = _Sag(df, 100.5)
    svc._gate = SimpleNamespace(model_id=None)
    svc._track = False
    svc._cycle_market_open = True
    svc._cycle_regime = None
    svc._calib = calib
    sig = SimpleNamespace(action=SignalAction.WAIT, price=99.5, stop=0.0, target=0.0,
                          rr_ratio=0.0, quality_score=0.0, metadata={}, reasons=[])
    svc._sig_cache["AAPL"] = {"ts": str(df.index[-1]), "sig": sig, "last_close": 99.5,
                              "e20": 100.0, "e50": 100.0, "r": 50.0, "a": 0.5,
                              "adx_v": 20.0, "htf_dir": -1, "day_open": 99.5}
    return svc


def test_bar_kipinde_yon_bar_kapanisindan(monkeypatch):
    row = _servis(monkeypatch, "true")._row("AAPL")
    assert row["direction"] == "-"                       # bar kapanışı EMA20'nin altında
    assert row["_dirbar"]["direction"] == "-"
    assert row["_dirlive"]["direction"] == "+"            # karşı varyant yine loglanıyor
    assert row["price"] == pytest.approx(100.5)           # gösterim/referans canlı kalır


def test_canli_kipinde_eski_davranis(monkeypatch):
    row = _servis(monkeypatch, "false")._row("AAPL")
    assert row["direction"] == "+"
    assert row["_dirlive"]["direction"] == "+" and row["_dirbar"]["direction"] == "-"


def test_prob_cal_yayinlanan_guveni_kullanir(monkeypatch):
    """Eğri yayınlanan güven (HTF/rejim DAHİL) üzerinde kurulur → girdi o olmalı.

    Bar yönü HTF ile uyumlu: güven (1/3·0,6 + 0,5·0,4)·1,15 = 0,46; saf-baz 0,40, karşı
    varyant 0,26. Eğri [(0; 0,2), (1; 0,8)] → 0,476 yalnız YAYINLANAN güvenle çıkar."""
    row = _servis(monkeypatch, "true", calib=[(0.0, 0.2), (1.0, 0.8)])._row("AAPL")
    assert row["confidence"] == pytest.approx(0.46)
    assert row["prob_cal"] == pytest.approx(0.2 + 0.6 * 0.46, abs=1e-3)


def test_dirlive_web_satirindan_dusurulur():
    src = inspect.getsource(lps)
    assert 'r.pop("_dirlive", None)' in src


# ================================================ 3. kalibrasyon ŞEMA KAPISI (BULGU-83-3)
def _yaz_egri(dizin, meta, noktalar=((0.0, 0.3), (1.0, 0.7))):
    (dizin / "forecast_calibration.json").write_text(
        json.dumps({"points": [list(p) for p in noktalar], "meta": meta}), encoding="utf-8")


@pytest.fixture
def veri(tmp_path, monkeypatch):
    monkeypatch.setattr("nihai.config.DATA_DIR", tmp_path)
    monkeypatch.delenv("SIGNAL_ON_BAR_CLOSE", raising=False)
    lps._CALIB_SIG, lps._CALIB_PTS, lps._CALIB_META, lps._CALIB_SEEN = None, None, {}, False
    yield tmp_path
    lps._CALIB_SIG, lps._CALIB_PTS, lps._CALIB_META, lps._CALIB_SEEN = None, None, {}, False


_GECERLI = {"girdi": "yayinlanan_guven", "yon": "bar", "interval": "1m"}


def test_eski_bicim_egri_reddedilir(veri):
    """23 Haziran eğrisinin BİREBİR biçimi (meta'da girdi/yon yok, 'brier' var) → None."""
    _yaz_egri(veri, {"interval": "1m", "horizon": 5, "samples": 51845, "brier": 0.3261,
                     "note": "tek-zaman-dilimli base güven; HTF/rejim hariç; tüm-örneklem"})
    assert lps._load_calibration() is None
    assert lps._load_calibration("1m") is None


def test_gecerli_egri_kabul_edilir(veri):
    _yaz_egri(veri, _GECERLI)
    assert lps._load_calibration() == [(0.0, 0.3), (1.0, 0.7)]
    assert lps._load_calibration("1m") == [(0.0, 0.3), (1.0, 0.7)]


def test_aralik_eslesmezse_uygulanmaz(veri):
    """1m/5 bar ufkunda kurulan olasılık 1h detay analizinin olasılığı DEĞİLDİR."""
    _yaz_egri(veri, _GECERLI)
    assert lps._load_calibration("1h") is None


def test_yon_kipi_eslesmezse_reddedilir_ve_kip_onbellek_anahtarinda(veri, monkeypatch):
    _yaz_egri(veri, {**_GECERLI, "yon": "canli"})
    assert lps._load_calibration() is None                # servis bar kipinde
    monkeypatch.setenv("SIGNAL_ON_BAR_CLOSE", "false")     # dosya DEĞİŞMEDEN kip değişti
    assert lps._load_calibration() == [(0.0, 0.3), (1.0, 0.7)]


def test_girdi_etiketi_yoksa_reddedilir(veri):
    _yaz_egri(veri, {"yon": "bar", "interval": "1m"})
    assert lps._load_calibration() is None


# ====================================== 4. kalibrasyon KURUCUSU — canlı defter (BULGU-83-3)
def _defter(gun_sayisi=10, n_gun=600, *, egim=0.0, tohum=7):
    """Sentetik çözülmüş defter: P(isabet | güven) = 0,5 + egim·(güven − 0,5).

    Bar varyantı `dir_bar`/`conf_bar`; `direction`/`hit` KASITEN karşı varyantın sonucudur
    (kurucu bar kipinde onları OKUMAMALI)."""
    rng = np.random.default_rng(tohum)
    rows = []
    for d in range(gun_sayisi):
        gun = f"2026-09-{d + 1:02d}"
        for i in range(n_gun):
            c = float(rng.uniform(0, 1))
            isabet = rng.uniform() < 0.5 + egim * (c - 0.5)
            yon = "+" if rng.uniform() < 0.5 else "-"
            s = 1.0 if yon == "+" else -1.0
            fwd = 100.0 * (1 + s * 0.001 * (1 if isabet else -1))
            rows.append({"logged_at": f"{gun}T17:{i // 60:02d}:{i % 60:02d}", "interval": "1m",
                         "horizon_bars": "5", "hit": "0" if isabet else "1",
                         "direction": "-" if yon == "+" else "+", "confidence": "0.9",
                         "dir_bar": yon, "conf_bar": f"{c:.4f}", "dir_live": "", "conf_live": "",
                         "price": "100.0", "fwd_price": f"{fwd:.4f}"})
    return rows


def test_bilgili_guven_kapiyi_gecer_ve_egri_artar():
    out = cf.kalibre_et(_defter(egim=0.8), kip="bar", interval="1m", horizon=5)
    m = out["meta"]
    assert m["durum"] == "BILGI_VAR" and m["yontem"] == "platt"
    assert m["yuruyen"]["gun_t"] < -cf.T_ESIK
    ys = [p for _, p in out["points"]]
    assert len(ys) == 21 and ys[-1] - ys[0] > 0.3
    assert all(b >= a for a, b in zip(ys, ys[1:], strict=False))
    assert m["girdi"] == "yayinlanan_guven" and m["yon"] == "bar" and m["interval"] == "1m"


def test_bilgisiz_guven_duz_taban_egrisi_yazar():
    out = cf.kalibre_et(_defter(egim=0.0), kip="bar", interval="1m", horizon=5)
    m = out["meta"]
    assert m["durum"] == "TABAN" and m["yontem"] == "taban"
    b = m["overall_hit_rate"]
    assert out["points"] == [[0.0, b], [1.0, b]]
    assert not m["yuruyen"]["gecti"]


def test_yuruyen_kapi_nedensel_yalniz_gecmisten_fit(monkeypatch):
    """`g` gününü puanlayan eğri YALNIZ önceki günlerden kurulur (ileri-bakış yok)."""
    rows = _defter(gun_sayisi=6, n_gun=50)
    ciftler = cf.canli_ciftler(rows, "bar", "1m", 5)
    boylar: list[int] = []
    asil = cf._egri
    monkeypatch.setattr(cf, "_egri", lambda p: (boylar.append(len(p)), asil(p))[1])
    k = cf.yuruyen_kapi(ciftler, isinma_gun=3, min_test_gun=1)
    assert boylar == [150, 200, 250]                      # gün 4/5/6 → önceki 3/4/5 gün
    assert k["n_test"] == 150 and k["test_gun"] == 3      # ısınma günleri PUANLANMAZ


def test_ciftler_bar_varyantini_canli_referanstan_olcer():
    """Bar kipinde `direction`/`hit` (karşı varyant) OKUNMAZ; beraberlik = ıska (0)."""
    rows = _defter(gun_sayisi=1, n_gun=20)
    rows.append({**rows[0], "fwd_price": "100.0"})        # beraberlik
    rows.append({**rows[0], "dir_bar": ""})               # varyantı olmayan satır
    rows.append({**rows[0], "interval": "1h"})            # başka aralık
    ciftler = cf.canli_ciftler(rows, "bar", "1m", 5)
    assert len(ciftler) == 21
    for r, (_g, c, y) in zip(rows[:20], ciftler[:20], strict=True):
        s = 1 if r["dir_bar"] == "+" else -1
        assert y == int(s * (float(r["fwd_price"]) - 100.0) > 0)
        assert c == pytest.approx(float(r["conf_bar"]))
        assert y != int(r["hit"])                         # karşı varyantın sonucu DEĞİL
    assert ciftler[20][2] == 0


def test_ciftler_canli_kipi_eski_satirda_direction_okur():
    eski = {"logged_at": "2026-08-01T17:00:00", "interval": "1m", "horizon_bars": "5",
            "hit": "1", "direction": "+", "confidence": "0.7", "dir_live": "", "conf_live": "",
            "price": "100", "fwd_price": "101"}
    yeni = {**eski, "direction": "-", "confidence": "0.2", "dir_live": "+", "conf_live": "0.6"}
    assert cf.canli_ciftler([eski, yeni], "canli", "1m", 5) == [
        ("2026-08-01", 0.7, 1), ("2026-08-01", 0.6, 1)]


def test_gun_kumeli_t_satir_agirlikli():
    """Elle: fark [−1,−1,−1,−1,+1], günler [a,a,a,a,b] → ort. −0,6; küme artık toplamları
    −1,6 / +1,6; Var = 5,12/25·2/1 = 0,4096 → t = −0,6/0,64 = −0,9375. Gün ORTALAMALARINI
    eşit tartan sürüm (−1 ve +1) burada t = 0 verirdi: 1 satırlık gün 4 satırlık kadar oy alırdı."""
    fark = np.array([-1.0, -1.0, -1.0, -1.0, 1.0])
    assert cf._gun_kumeli_t(fark, np.array(list("aaaab"))) == pytest.approx(-0.9375)
    assert cf._gun_kumeli_t(np.array([0.1, 0.2]), np.array(["a", "a"])) is None   # tek küme
    assert cf._gun_kumeli_t(np.array([0.1]), np.array(["a"])) is None


def test_kurucu_eski_orneklem_ici_yontemi_tasimiyor():
    src = (KOK / "research/calibrate_forecast.py").read_text(encoding="utf-8")
    assert "for t in range(50, n - horizon)" not in src    # tarihsel bar taraması
    assert "bin_w" not in src and "min_bin" not in src      # ham kova eğrisi
    assert "_apply_calibration" in src                      # yayınlanacak temsil puanlanıyor


def test_cli_kopya_deftere_yazar(tmp_path, monkeypatch):
    monkeypatch.delenv("SIGNAL_ON_BAR_CLOSE", raising=False)
    log = tmp_path / "pl.csv"
    rows = _defter(gun_sayisi=6, n_gun=60)
    with log.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "fc.json"
    assert cf.main(["--log", str(log), "--out", str(out), "--interval", "1m",
                    "--horizon", "5"]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["meta"]["girdi"] == "yayinlanan_guven" and d["meta"]["yon"] == "bar"


# ======================================================= 5. beraberlik hükmü (iki kardeş)
def _cozulmus(n_hit, n_iska, n_berabere):
    rows = ([{"hit": "1", "fwd_ret_pct": "0.1"}] * n_hit
            + [{"hit": "0", "fwd_ret_pct": "-0.1"}] * n_iska
            + [{"hit": "0", "fwd_ret_pct": "0.0000"}] * n_berabere)
    return [dict(r, confidence="0.5") for r in rows]


def test_rapor_beraberligi_ayri_tasir():
    rep = pt.report(_cozulmus(460, 460, 80))
    assert rep["hit_rate"] == pytest.approx(0.46)          # geriye uyum: DEĞİŞMEDİ
    assert rep["tie_share"] == pytest.approx(0.08)
    assert rep["hit_rate_excl_ties"] == pytest.approx(0.5)


def test_hukum_yazi_turayi_ters_okumaz():
    """%8 beraberlikte yazı-tura `hit_rate` 0,46 verir; eski hüküm bunu INVERTED okurdu."""
    rep = pt.report(_cozulmus(460, 460, 80))
    assert pt.verdict({"resolved": 1000, "hit_rate": 0.46})["label"] == "INVERTED"   # eski yol
    v = pt.verdict(rep)
    assert v["label"] == "NO_EDGE" and v["tie_share"] == pytest.approx(0.08)


def test_kardes_hukum_forward_harness_ayni_kural():
    from research import forward_harness as fh
    ev = {"resolved": 20000, "days_span": 60, "hit_rate": 0.46,
          "hit_rate_excl_ties": 0.5, "tie_share": 0.08}
    assert fh._verdict(ev)["label"] == "NO_EDGE"
    assert fh._verdict({**ev, "hit_rate_excl_ties": None})["label"] == "INVERTED"


# ============================================== 6. defter: karşı varyant (dir_live) kaydı
def test_dir_live_kolonlari_sona_eklendi():
    i = pt.FIELDS.index("dir_live")
    assert pt.FIELDS[i:i + 2] == ["dir_live", "conf_live"]
    assert i > pt.FIELDS.index("meta_strategy")           # araya değil SONA: eski hiza korunur


def test_record_dir_live_yazar():
    satir = {"symbol": "AAPL", "bar_ts": "2026-09-18T14:00:00", "direction": "-",
             "action": "WAIT", "confidence": 0.46, "prob_cal": 0.5, "price": 100.5,
             "bar_close": 99.5, "pwin": None, "pwin_side": None,
             "_dirbar": {"direction": "-", "confidence": 0.46},
             "_dirlive": {"direction": "+", "confidence": 0.26}}
    assert pt.record([satir], "1m") == 1
    with pt.PRED_LOG.open(encoding="utf-8", newline="") as f:
        row = list(csv.DictReader(f))[-1]
    assert (row["dir_bar"], row["dir_live"], row["conf_live"]) == ("-", "+", "0.26")


def test_golge_terfi_uyusma_yuzeyi_canli_varyanti_okur():
    """Bar kipinde `direction == dir_bar`; yüzey yozlaşmasın diye canlı taraf `dir_live`."""
    from research.shadow_promote import _dirbar_agrees
    assert _dirbar_agrees({"dir_bar": "-", "direction": "-", "dir_live": "+"}) is False
    assert _dirbar_agrees({"dir_bar": "-", "direction": "+", "dir_live": ""}) is False
    assert _dirbar_agrees({"dir_bar": "+", "direction": "+", "dir_live": "+"}) is True


# ======================================================= 7. yüzeyler + gece görevi bağı
def test_calibration_input_silindi_iki_yuzeyde_de():
    for yol in ("nihai/execution/live_predict_service.py", "nihai/web/web_dashboard.py"):
        assert "_calibration_input" not in (KOK / yol).read_text(encoding="utf-8"), yol


def test_hisse_detay_ayni_kurali_okur(monkeypatch):
    """`/api/analyze` yönü, tahmin servisiyle AYNI kuralla hangi fiyattan hesaplıyor?

    Casus `_bias_and_confidence`a giden fiyatı yakalar (gösterge aritmetiğine bağlı kalmaz):
    bar kipinde bar kapanışı (100), canlı kipte kotasyon (99). Gösterilen fiyat canlı kalır."""
    from nihai.web import web_dashboard as wd
    idx = pd.date_range("2026-09-18 13:00", periods=120, freq="1min")
    df = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0,
                       "volume": 1000.0}, index=idx)
    monkeypatch.setattr(wd, "_analysis_provider", lambda: _Sag(df, 99.0))
    gorulen: list[float] = []
    asil = lps._bias_and_confidence

    def casus(price, *a, **k):
        gorulen.append(price)
        return asil(price, *a, **k)

    monkeypatch.setattr(lps, "_bias_and_confidence", casus)
    monkeypatch.setenv("SIGNAL_ON_BAR_CLOSE", "true")
    bar = wd.analyze_symbol("AAPL", "1m")
    monkeypatch.setenv("SIGNAL_ON_BAR_CLOSE", "false")
    canli = wd.analyze_symbol("AAPL", "1m")
    assert bar["ok"] and canli["ok"]
    assert gorulen == [100.0, 99.0]
    assert bar["price"] == pytest.approx(99.0)


def test_hisse_detay_egriyi_yalniz_kendi_araliginda_uygular(veri, monkeypatch):
    """1m/5 bar eğrisi 1h detay analizine UYGULANMAZ (eskiden uygulanıyordu)."""
    from nihai.web import web_dashboard as wd
    _yaz_egri(veri, _GECERLI)
    idx = pd.date_range("2026-09-18 13:00", periods=120, freq="1min")
    df = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0,
                       "volume": 1000.0}, index=idx)
    monkeypatch.setattr(wd, "_analysis_provider", lambda: _Sag(df, 99.0))
    assert wd.analyze_symbol("AAPL", "1m")["prob_cal"] is not None
    assert wd.analyze_symbol("AAPL", "1h")["prob_cal"] is None


def test_gece_gorevi_kalibrasyonu_kurar():
    """Eğriyi YAZAN adım `calibrate_forecast`tır (`--calib` onu YAZMAZ — AI kalibratörlerini
    tazeler); ikisi de gece görevinde koşmalı, eğri çözümden (--resolve) SONRA kurulmalı."""
    ps1 = (KOK / "forward-harness.ps1").read_text(encoding="utf-8-sig")
    assert "-m research.calibrate_forecast" in ps1
    assert ps1.index("research.forward_harness") < ps1.index("research.calibrate_forecast")
    assert "daily_summary --calib" in ps1


def test_env_example_yeni_varsayilani_belgeler():
    env = (KOK / ".env.example").read_text(encoding="utf-8")
    assert "SIGNAL_ON_BAR_CLOSE=true" in env and "SIGNAL_ON_BAR_CLOSE=false" not in env
