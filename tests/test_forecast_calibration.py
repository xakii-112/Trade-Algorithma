"""Tahmin kalibrasyonu alan-tutarlılığı testleri.

Tur-83'e kadar forecast_calibration.json tarihsel barlarda HTF/rejim HARİÇ saf-baz güvenle
üretiliyordu ve lookup da o yüzden saf-baz güvenle yapılıyordu (`_calibration_input`).
Tur-83 eğriyi CANLI defterde, YAYINLANAN güven (HTF/rejim DAHİL) üzerinde kurdu → lookup
yayınlanan güvenle yapılır, `_calibration_input` silindi. Alan eşleşmesi artık bir yorumla
değil `_load_calibration`ın ŞEMA KAPISIYLA korunur (meta.girdi/yon/interval; bkz.
tests/test_tur83_tahmin_dogrulugu.py).
"""
import json

from nihai.execution import live_predict_service as lps
from nihai.execution.live_predict_service import (
    _apply_calibration,
    _bias_and_confidence,
)

# price, e20, e50, r, adx_v → yukarı trend (e20>e50, price>e20, rsi>50)
_UP = (105.0, 104.0, 100.0, 60.0, 30.0)


def test_direction_invariant_to_htf_and_regime():
    d0, _b, _c = _bias_and_confidence(*_UP, htf_dir=0, regime_tilt=1.0)
    d1, _b1, _c1 = _bias_and_confidence(*_UP, htf_dir=-1, regime_tilt=0.5)
    assert d0 == d1 == "+"                       # yön HTF/rejimden bağımsız


def test_sifir_bilgi_girdisi_yuksek_guvenli_yon_uretmez():
    """Tur-28: price==e20==e50 ve rsi==50 = HİÇBİR yön bilgisi yok.

    Eskiden iki karşılaştırma da `>` idi → eşitlik sessizce −1.0 (tam SHORT) sayılıyor,
    bileşenler [−1, 0, −1] → bias −0,667 → **conf 0,60 ile SHORT**. Yani sıfır bilgili
    girdiye yüksek güven. Artık eşitlik 0.0 (bilgi yok) → mevcut nötr dalına düşer."""
    _d, bias, _c = _bias_and_confidence(100.0, 100.0, 100.0, 50.0, adx_v=float("nan"))
    assert abs(bias) <= 1e-3               # nötr dalının işaretlemesi, −0,667 DEĞİL
    _d2, _b2, conf = _bias_and_confidence(100.0, 100.0, 100.0, 50.0, adx_v=float("nan"))
    assert conf < 0.01                     # ADX de yoksa güven ~0 (dürüst cevap)


def test_esitlik_disinda_bit_denk():
    """Düzeltme YALNIZ eşitlik dalını ekler: eşitlik içermeyen girdilerde davranış aynı."""
    import numpy as np

    def _eski(price, e20, e50, r, adx_v):
        comps = [1.0 if e20 > e50 else -1.0,
                 float(np.clip((r - 50.0) / 25.0, -1.0, 1.0)),
                 1.0 if price > e20 else -1.0]
        bias = float(np.mean(comps))
        if abs(bias) < 1e-9:
            bias = 1e-3 if price >= e50 else -1e-3
        adx_norm = float(np.clip(adx_v / 40.0, 0.0, 1.0))
        return ("+" if bias >= 0 else "-", bias,
                float(np.clip(abs(bias) * 0.6 + adx_norm * 0.4, 0.0, 1.0)))

    rng = np.random.default_rng(0)
    for _ in range(2000):
        p, e1, e2 = rng.uniform(50, 200, 3)      # sürekli dağılım → eşitlik olasılığı ~0
        r, a = rng.uniform(0, 100), rng.uniform(0, 60)
        assert _bias_and_confidence(p, e1, e2, r, a) == _eski(p, e1, e2, r, a)


def test_apply_calibration_clamps_out_of_range():
    pts = [[0.1, 0.55], [0.9, 0.48]]
    assert _apply_calibration(pts, 0.0) == 0.55      # alt uç clamp
    assert _apply_calibration(pts, 1.0) == 0.48      # üst uç clamp (modüle güven taşması)
    mid = _apply_calibration(pts, 0.5)
    assert 0.48 < mid < 0.55                          # parçalı-doğrusal interpolasyon
    assert _apply_calibration(None, 0.5) is None      # kalibrasyon yoksa None


# --------------------------------------------------------------------------- Tur-28
# Eğri TAZE okunmalı: 7/24 süreçler (tahmin servisi + web panosu) eskiden dosyayı bir kez
# okuyup sonsuza dek saklıyordu → calibrate_forecast.py yeniden ürettiğinde görmüyorlardı.
def _yaz(dizin, noktalar):
    # Tur-83: meta'sız (eski biçim) eğri şema kapısında REDDEDİLİR; tazelenme testleri
    # kabul edilen biçimi yazar (varsayılan kip 'bar' — testler SIGNAL_ON_BAR_CLOSE'u siler).
    (dizin / "forecast_calibration.json").write_text(
        json.dumps({"points": noktalar,
                    "meta": {"girdi": "yayinlanan_guven", "yon": "bar", "interval": "1m"}}),
        encoding="utf-8")


def _cache_sifirla():
    lps._CALIB_SIG, lps._CALIB_PTS, lps._CALIB_SEEN = None, None, False


def test_load_calibration_dosya_degisince_tazelenir(tmp_path, monkeypatch):
    monkeypatch.setattr("nihai.config.DATA_DIR", tmp_path)
    monkeypatch.delenv("SIGNAL_ON_BAR_CLOSE", raising=False)
    _cache_sifirla()
    _yaz(tmp_path, [[0.1, 0.55], [0.9, 0.48]])
    assert lps._load_calibration() == [(0.1, 0.55), (0.9, 0.48)]

    # Aynı dosya → yeniden ayrıştırma YOK (imza değişmedi): read_text patlarsa cache konuşur.
    monkeypatch.setattr("pathlib.Path.read_text",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("tekrar okundu")))
    assert lps._load_calibration() == [(0.1, 0.55), (0.9, 0.48)]
    monkeypatch.undo()
    monkeypatch.setattr("nihai.config.DATA_DIR", tmp_path)
    monkeypatch.delenv("SIGNAL_ON_BAR_CLOSE", raising=False)

    # Yeni eğri BOYUTU da farklı: imza (mtime_ns, size) ikinci bileşenden de değişsin —
    # test aynı-milisaniye yazımında mtime çözünürlüğüne bağlı kalmasın (flake önleme).
    _yaz(tmp_path, [[0.2, 0.61], [0.5, 0.5], [0.8, 0.44]])   # calibrate_forecast yeniden koştu
    assert lps._load_calibration() == [(0.2, 0.61), (0.5, 0.5), (0.8, 0.44)]
    _cache_sifirla()


def test_load_calibration_dosya_yok_veya_bozuk_none(tmp_path, monkeypatch):
    monkeypatch.setattr("nihai.config.DATA_DIR", tmp_path)
    monkeypatch.delenv("SIGNAL_ON_BAR_CLOSE", raising=False)
    _cache_sifirla()
    assert lps._load_calibration() is None            # dosya yok
    (tmp_path / "forecast_calibration.json").write_text("{bozuk", encoding="utf-8")
    assert lps._load_calibration() is None            # bozuk JSON → çökme yok
    _yaz(tmp_path, [])
    assert lps._load_calibration() is None            # boş nokta listesi → kalibrasyon pasif
    _cache_sifirla()
