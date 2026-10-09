"""calibration_loop.py testleri — zaman-bölmeli yalnız-OOS kalibrasyon döngüsü.

Ağırlık merkezi: **sızıntısızlık** (OOS'u bozmak FIT'i değiştirmemeli), **kronolojik bölme**
(rastgele YASAK), **yeterlilik kapısı** (az veri → YETERSİZ, asla 'iyileşti' iddiası) ve
**kalıcılık kapısı** (yalnız HEM Brier HEM ECE OOS'ta iyileşirse kaydedilir). Bu dört şey
modülün en kolay sessizce bozulacağı yerler (CLAUDE.md: look-ahead yasak; ACI DERS #3).
"""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pytest

from nihai.ai import calibration_loop

FIELDS = ["logged_at", "symbol", "interval", "bar_ts", "direction", "action", "confidence",
          "prob_cal", "pwin", "pwin_side", "price", "bar_close", "horizon_bars",
          "resolved_at", "fwd_price", "hit", "fwd_ret_pct", "hit_bar", "slip_bps",
          "neural_pwin", "neural_pwin_side", "neural_pwin_std", "dir_bar", "conf_bar"]

_BASE = datetime(2026, 1, 1, 0, 0, 0)


@pytest.fixture(autouse=True)
def _tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_loop, "PRED_LOG", tmp_path / "predictions_log.csv")
    monkeypatch.setattr(calibration_loop, "CALIBRATORS_FILE", tmp_path / "ai_calibrators.json")
    return tmp_path


def _ts(i: int) -> str:
    return (_BASE + timedelta(minutes=i)).isoformat()


def _row(i: int, *, direction="LONG", pwin="", pwin_side="", fwd="",
         neural_pwin="", neural_pwin_side="", resolved_at="AUTO") -> dict:
    bar_ts = _ts(i)
    if resolved_at == "AUTO":
        resolved_at = (_BASE + timedelta(minutes=i, hours=2)).isoformat()
    return {"logged_at": bar_ts, "symbol": "TST", "interval": "1h", "bar_ts": bar_ts,
            "direction": direction, "action": direction, "resolved_at": resolved_at,
            "pwin": pwin, "pwin_side": pwin_side, "fwd_ret_pct": fwd,
            "neural_pwin": neural_pwin, "neural_pwin_side": neural_pwin_side}


def _write_log(rows: list[dict]) -> None:
    with calibration_loop.PRED_LOG.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: "" for k in FIELDS}, **r})


def _seeded_rows(n: int, seed: int, *, direction: str = "LONG") -> list[dict]:
    """n satır, kronolojik bar_ts, pwin~U(0.2,0.8) ve y~Bernoulli(pwin) (yani KALİBRE-YAKIN
    ham veri — leak-free/bölme testleri için; 'gerçekten iyileşen' senaryo AYRI (aşırı-güvenli)
    bir üreteç kullanır, bkz. `_overconfident_rows`)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = float(np.clip(rng.uniform(0.2, 0.8), 0.02, 0.98))
        y = 1 if rng.random() < p else 0
        rows.append(_row(i, direction=direction, pwin=f"{p:.6f}", pwin_side=direction,
                         fwd=("1.0" if y == 1 else "-1.0")))
    return rows


def _overconfident_rows(n: int = 1000, seed: int = 42, overconf: float = 1.8) -> list[dict]:
    """Sistematik olarak FAZLA-GÜVENLİ ham olasılıklar: gerçek q'dan daha uç bir p raporlanır
    (klasik aşırı-güven deseni). Platt/izotonik bunu OOS'ta ÖLÇÜLEBİLİR şekilde düzeltmeli —
    seed=42/n=1000/split=0.6 ile önceden DOĞRULANDI (platt Δbrier=-0.016 Δece=-0.067; izotonik
    Δbrier=-0.013 Δece=-0.044 — bkz. görev notları)."""
    rng = np.random.default_rng(seed)
    q_levels = [0.30, 0.40, 0.50, 0.60, 0.70]
    rows = []
    for i in range(n):
        q = q_levels[i % len(q_levels)]
        p = float(min(max(0.5 + overconf * (q - 0.5), 0.02), 0.98))
        y = 1 if rng.random() < q else 0
        rows.append(_row(i, pwin=f"{p:.6f}", pwin_side="LONG", fwd=("1.0" if y == 1 else "-1.0")))
    return rows


def _bilgisiz_asiri_guvenli_rows(n: int = 1200, seed: int = 7, taban: float = 0.46) -> list[dict]:
    """Aşırı-güvenli AMA BİLGİSİZ ham olasılıklar: p geniş bir aralıkta salınır, y ise p'den
    BAĞIMSIZ olarak Bernoulli(taban)'dır.

    Bu, ACI DERS #9'un tam senaryosudur ve canlı `pwin`'in ölçülmüş hâlidir: kalibrasyon
    Brier'i ve ECE'yi ham'a göre GERÇEKTEN düzeltir (çünkü p'yi taban orana büzer), ama
    kazanılan hiçbir BİLGİ yoktur — sabit taban-oran tahmincisi aynı işi yapar. Eski kapı
    bunu `KAYITLI` yazardı (ve gerçek artefaktta aylarca öyle yazdı)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = float(np.clip(rng.uniform(0.15, 0.85), 0.02, 0.98))
        y = 1 if rng.random() < taban else 0        # y ⟂ p — hiçbir bilgi yok
        rows.append(_row(i, pwin=f"{p:.6f}", pwin_side="LONG", fwd=("1.0" if y == 1 else "-1.0")))
    return rows


# ---------------------------------------------------------------------------
# learning.py'den KOPYALANMIŞ yardımcılar — parite kilidi
# ---------------------------------------------------------------------------
def test__f_gecerli_ve_gecersiz_deger():
    assert calibration_loop._f("0.5") == 0.5
    assert calibration_loop._f("bozuk") is None
    assert calibration_loop._f(None) is None
    assert calibration_loop._f(float("nan")) is None


def test__norm_dir_kanonik_formlar():
    assert calibration_loop._norm_dir("+") == "LONG"
    assert calibration_loop._norm_dir("short") == "SHORT"
    assert calibration_loop._norm_dir("DOWN") == "SHORT"
    assert calibration_loop._norm_dir("???") == ""


# ---------------------------------------------------------------------------
# Zaman damgası kanonikleştirme (BULGU-16)
# ---------------------------------------------------------------------------
def test_parse_ts_utc_naive_aynen_doner():
    assert calibration_loop._parse_ts_utc("2026-01-01T10:00:00") == datetime(2026, 1, 1, 10, 0, 0)


def test_parse_ts_utc_tz_aware_utcye_cevrilir():
    # -05:00 (ET kışı) -> UTC +5 saat; gerçek predictions_log verisinde ölçülen biçimle aynı
    # (bkz. modül docstring'i: bar_ts bazen 'YYYY-MM-DD HH:MM:SS-04:00' formatında).
    dt = calibration_loop._parse_ts_utc("2026-01-15 15:59:00-05:00")
    assert dt == datetime(2026, 1, 15, 20, 59, 0)


def test_parse_ts_utc_bozuk_veya_bos_none():
    assert calibration_loop._parse_ts_utc("bozuk-tarih") is None
    assert calibration_loop._parse_ts_utc("") is None


def test_row_ts_bar_ts_oncelikli():
    header = ["bar_ts", "logged_at", "resolved_at"]
    idx = {n: i for i, n in enumerate(header)}
    row = ["2026-01-01T00:00:00", "2026-01-02T00:00:00", "2026-01-03T00:00:00"]
    assert calibration_loop._row_ts(row, idx) == datetime(2026, 1, 1)


def test_row_ts_bar_ts_bozuksa_logged_ata_duser():
    header = ["bar_ts", "logged_at", "resolved_at"]
    idx = {n: i for i, n in enumerate(header)}
    row = ["", "2026-01-02T00:00:00", "2026-01-03T00:00:00"]
    assert calibration_loop._row_ts(row, idx) == datetime(2026, 1, 2)


def test_row_ts_hicbiri_ayristirilamazsa_none():
    header = ["bar_ts", "logged_at", "resolved_at"]
    idx = {n: i for i, n in enumerate(header)}
    assert calibration_loop._row_ts(["", "", ""], idx) is None


# ---------------------------------------------------------------------------
# _load_pairs
# ---------------------------------------------------------------------------
def test_load_pairs_bilinmeyen_source_value_error():
    with pytest.raises(ValueError):
        calibration_loop._load_pairs("bogus-source")


def test_load_pairs_dosya_yokken_bos_liste():
    assert calibration_loop._load_pairs("pwin") == []


def test_load_pairs_temel_etiketleme_taraf_isaretli():
    rows = [
        _row(0, pwin="0.60", pwin_side="LONG", fwd="1.5"),    # side==direction, fwd>0 -> y=1
        _row(1, pwin="0.40", pwin_side="SHORT", fwd="1.5"),   # side!=direction -> -1.5 -> y=0
    ]
    _write_log(rows)
    pairs = calibration_loop._load_pairs("pwin")
    assert len(pairs) == 2
    assert pairs[0][1] == pytest.approx(0.60) and pairs[0][2] == 1
    assert pairs[1][1] == pytest.approx(0.40) and pairs[1][2] == 0
    assert pairs[0][0] < pairs[1][0]   # kronolojik bar_ts korunuyor


def test_load_pairs_cozulmemis_ve_unresolvable_atlanir():
    rows = [
        _row(0, pwin="0.6", pwin_side="LONG", fwd="1.0", resolved_at=""),
        _row(1, pwin="0.6", pwin_side="LONG", fwd="1.0", resolved_at="UNRESOLVABLE"),
        _row(2, pwin="0.6", pwin_side="LONG", fwd="1.0"),
    ]
    _write_log(rows)
    assert len(calibration_loop._load_pairs("pwin")) == 1


def test_load_pairs_neural_kolon_yoksa_bos_liste():
    fields_no_neural = [f for f in FIELDS if not f.startswith("neural_")]
    with calibration_loop.PRED_LOG.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_no_neural)
        w.writeheader()
        w.writerow({**{k: "" for k in fields_no_neural}, "direction": "LONG",
                   "resolved_at": "2026-01-01T00:00:00", "bar_ts": "2026-01-01T00:00:00"})
    assert calibration_loop._load_pairs("neural") == []


def test_load_pairs_gercek_sema_arti_eksi_yon():
    rows = [_row(0, direction="+", pwin="0.7", pwin_side="LONG", fwd="2.0")]
    _write_log(rows)
    pairs = calibration_loop._load_pairs("pwin")
    assert len(pairs) == 1 and pairs[0][2] == 1


# ---------------------------------------------------------------------------
# evaluate_source — yeterlilik kapısı
# ---------------------------------------------------------------------------
def test_evaluate_source_bilinmeyen_source_value_error():
    with pytest.raises(ValueError):
        calibration_loop.evaluate_source("bogus-source", "platt")


def test_evaluate_source_yetersiz_n_oos_az():
    rows = _seeded_rows(60, seed=5)   # split 0.6 -> n_oos=24 < min_n=200 (varsayılan)
    _write_log(rows)
    r = calibration_loop.evaluate_source("pwin", "platt")
    assert r["durum"] == "YETERSIZ"
    assert "calibrator" not in r
    assert r["n_oos"] == 24
    assert "n_oos=24" in r["gerekce"]


def test_evaluate_source_yetersiz_fit_tek_sinif():
    n = 600
    cut = int(n * calibration_loop.DEFAULT_SPLIT)   # 360
    rows = [_row(i, pwin="0.9", pwin_side="LONG", fwd="1.0") for i in range(cut)]  # FIT: hep y=1
    rng = np.random.default_rng(21)
    for i in range(cut, n):   # OOS: karışık, yeterli (>=200)
        y = 1 if rng.random() < 0.5 else 0
        rows.append(_row(i, pwin="0.5", pwin_side="LONG", fwd=("1.0" if y else "-1.0")))
    _write_log(rows)
    r = calibration_loop.evaluate_source("pwin", "platt")
    assert r["durum"] == "YETERSIZ"
    assert "tek-sınıf" in r["gerekce"]
    assert r["n_oos"] >= 200   # YALNIZ tek-sınıf yüzünden yetersiz, n_oos zaten yeterliydi
    assert "calibrator" not in r


def test_evaluate_source_neural_n6_spec_senaryosu():
    """Görevde ölçülen GERÇEK durum: neural_pwin+resolved n=6 — YETERSİZ olmalı, asla
    'kalibratör üretildi' denmemeli."""
    rows = [_row(i, neural_pwin=f"{0.5 + 0.02 * i:.2f}", neural_pwin_side="LONG",
                 fwd=("1.0" if i % 2 == 0 else "-1.0")) for i in range(6)]
    _write_log(rows)
    assert len(calibration_loop._load_pairs("neural")) == 6
    r = calibration_loop.evaluate_source("neural", "platt")
    assert r["durum"] == "YETERSIZ"
    assert "calibrator" not in r
    assert r.get("improved") is None   # 'improved' alanı hiç YOK — iyileşti iddiası yok


def test_evaluate_source_bilinmeyen_method_value_error_yeterli_veride():
    rows = _seeded_rows(600, seed=17)
    _write_log(rows)
    with pytest.raises(ValueError):
        calibration_loop.evaluate_source("pwin", "dogrusal-olmayan")


# ---------------------------------------------------------------------------
# evaluate_source — ölçülen (OLCULDU) yol + alan sözleşmesi
# ---------------------------------------------------------------------------
def test_evaluate_source_olculdu_alan_sozlesmesi():
    rows = _seeded_rows(600, seed=3)
    _write_log(rows)
    r = calibration_loop.evaluate_source("pwin", "platt")
    assert r["durum"] == "OLCULDU"
    for key in ("kaynak", "method", "n_fit", "n_oos", "fit_range", "oos_range",
               "brier_raw", "brier_cal", "ece_raw", "ece_cal", "delta_brier", "delta_ece",
               "improved", "verdict", "calibrator", "bins_raw", "bins_cal"):
        assert key in r, f"eksik alan: {key}"
    assert r["kaynak"] == "pwin" and r["method"] == "platt"
    assert isinstance(r["improved"], bool)
    assert r["n_fit"] + r["n_oos"] == 600


# ---------------------------------------------------------------------------
# B4.2 — zaman bölmesi kronolojiktir (dosya sırasından BAĞIMSIZ)
# ---------------------------------------------------------------------------
def test_zaman_bolmesi_kronolojik_dosya_sirasindan_bagimsiz():
    rows = _seeded_rows(600, seed=11)
    shuffled = list(rows)
    random.Random(3).shuffle(shuffled)   # dosyaya KARIŞIK sırada yaz
    _write_log(shuffled)
    r = calibration_loop.evaluate_source("pwin", "platt")
    assert r["durum"] == "OLCULDU"
    fit_end = datetime.fromisoformat(r["fit_range"][1])
    oos_start = datetime.fromisoformat(r["oos_range"][0])
    assert fit_end <= oos_start


def test_zaman_bolmesi_oranini_kullanir():
    rows = _seeded_rows(1000, seed=23)
    _write_log(rows)
    r = calibration_loop.evaluate_source("pwin", "platt", split=0.7)
    assert r["n_fit"] == 700 and r["n_oos"] == 300


# ---------------------------------------------------------------------------
# B4.1 — SIZINTI YOK: OOS'u bozmak FIT kalibratörünü DEĞİŞTİRMEZ
# ---------------------------------------------------------------------------
def test_leak_free_oos_bozulunca_fit_calibratoru_degismez():
    n = 600
    base_rows = _seeded_rows(n, seed=7)
    _write_log(base_rows)
    r1 = calibration_loop.evaluate_source("pwin", "platt")
    assert r1["durum"] == "OLCULDU"

    cut = int(n * calibration_loop.DEFAULT_SPLIT)
    bozuk_rows = list(base_rows)
    rng = np.random.default_rng(999)
    for j in range(cut, n):   # yalnız OOS dilimini (aynı zaman damgalarında) BOZ
        p2 = float(np.clip(rng.uniform(0.02, 0.98), 0.02, 0.98))
        y2 = 1 if rng.random() < 0.5 else 0
        bozuk_rows[j] = {**bozuk_rows[j], "pwin": f"{p2:.6f}",
                         "fwd_ret_pct": "1.0" if y2 == 1 else "-1.0"}
    _write_log(bozuk_rows)
    r2 = calibration_loop.evaluate_source("pwin", "platt")
    assert r2["durum"] == "OLCULDU"

    assert r2["n_fit"] == r1["n_fit"]
    assert r2["fit_range"] == r1["fit_range"]
    assert r2["calibrator"] == r1["calibrator"], (
        "OOS bozulması FIT kalibratörünü değiştirdi — LOOK-AHEAD sızıntısı")
    # OOS'a bağlı metrikler GÖRÜNÜR şekilde değişmeli (test vakumlu değil — perturbasyon etkili)
    assert (r2["brier_raw"], r2["ece_raw"]) != (r1["brier_raw"], r1["ece_raw"])


def test_leak_free_isotonic_icin_de_gecerli():
    n = 600
    base_rows = _seeded_rows(n, seed=8)
    _write_log(base_rows)
    r1 = calibration_loop.evaluate_source("pwin", "isotonic")
    assert r1["durum"] == "OLCULDU"

    cut = int(n * calibration_loop.DEFAULT_SPLIT)
    bozuk_rows = list(base_rows)
    for j in range(cut, n):
        bozuk_rows[j] = {**bozuk_rows[j], "pwin": "0.5", "fwd_ret_pct": "-1.0"}
    _write_log(bozuk_rows)
    r2 = calibration_loop.evaluate_source("pwin", "isotonic")
    assert r2["calibrator"] == r1["calibrator"]


# ---------------------------------------------------------------------------
# B4.4 — kalıcılık kapısı: yalnız Brier iyileşip ECE kötüleşirse KAYDEDİLMEZ
# ---------------------------------------------------------------------------
def test_kalicilik_kapisi_yalniz_brier_iyilesirse_kaydedilmez(monkeypatch):
    def fake_eval(source, method, *, split=calibration_loop.DEFAULT_SPLIT,
                 tail_n=calibration_loop.DEFAULT_TAIL_N, min_n=calibration_loop.DEFAULT_MIN_N):
        # Brier iyileşiyor (0.25->0.20) AMA ECE kötüleşiyor (0.05->0.08) — İKİSİ BİRDEN şartı YOK
        return {"kaynak": source, "method": method, "n_fit": 300, "n_oos": 250,
                "fit_range": ["2026-01-01T00:00:00", "2026-01-03T00:00:00"],
                "oos_range": ["2026-01-03T00:01:00", "2026-01-05T00:00:00"],
                "durum": "OLCULDU", "calibrator": {"method": method, "n": 300},
                "brier_raw": 0.25, "brier_cal": 0.20,
                "ece_raw": 0.05, "ece_cal": 0.08,
                "delta_brier": -0.05, "delta_ece": 0.03,
                "improved": False, "verdict": "sentetik test",
                "bins_raw": [], "bins_cal": []}
    monkeypatch.setattr(calibration_loop, "evaluate_source", fake_eval)
    out = calibration_loop.refresh(sources=("pwin",), methods=("platt", "isotonic"))
    assert out["pwin"]["durum"] == "KAPI_GECILEMEDI"
    assert "calibrator" not in out["pwin"]
    persisted = calibration_loop.load_calibrators()
    assert persisted["pwin"]["durum"] == "KAPI_GECILEMEDI"
    assert "calibrator" not in persisted["pwin"]


# ---------------------------------------------------------------------------
# B4.5 — gerçekten iyileşen sentetik veri → kaydedilir, deltalar negatif
# ---------------------------------------------------------------------------
def test_gercekten_iyilesen_asiri_guvenli_veri_kaydedilir_ve_delta_negatif():
    _write_log(_overconfident_rows())
    out = calibration_loop.refresh(sources=("pwin",), methods=("platt", "isotonic"))
    entry = out["pwin"]
    assert entry["durum"] == "KAYITLI"
    assert entry["delta_brier"] < 0
    assert entry["delta_ece"] < 0
    assert entry["method"] in ("platt", "isotonic")

    persisted = calibration_loop.load_calibrators()["pwin"]
    assert persisted["durum"] == "KAYITLI"
    assert persisted["delta_brier"] < 0 and persisted["delta_ece"] < 0
    assert "calibrator" in persisted


def test_bilgisiz_kaynak_hami_yener_ama_tabani_yenemez_kaydedilmez():
    """Tur-28 / ACI DERS #9 KAPIDA: kalibrasyon ham'ı Brier+ECE'de düzeltse bile, kazanç
    yalnız taban orana BÜZÜLMEDEN geliyorsa kalibratör KAYDEDİLMEZ.

    Eski kapı (yalnız ham'a bakan) bunu `KAYITLI` yazıyordu — gerçek `data/ai_calibrators.json`
    da tam bu yüzden aylarca yanıltıcı bir `pwin: KAYITLI` taşıdı."""
    _write_log(_bilgisiz_asiri_guvenli_rows())
    out = calibration_loop.refresh(sources=("pwin",), methods=("platt", "isotonic"))
    entry = out["pwin"]
    assert entry["durum"] == "KAPI_GECILEMEDI"

    denemeler = {d["method"]: d for d in entry["denemeler"]}
    platt = denemeler["platt"]
    assert platt["ham_yener"] is True          # kalibrasyon ham'ı gerçekten düzeltiyor
    assert platt["taban_yener"] is False       # ama sabit tabandan daha bilgili DEĞİL
    assert platt["improved"] is False
    assert platt["taban_t"] > -calibration_loop.T_ESIK
    assert "BÜZÜLME" in platt["verdict"]
    # Kalibratör diske YAZILMAZ → calibrate() geçiş-şeffaf kalır.
    assert calibration_loop.calibrate("pwin", 0.9) == pytest.approx(0.9)


def test_kayitli_girdi_taban_alanlarini_tasir():
    """Kaydedilen artefakt taban-oran kanıtını da taşımalı (sonraki operatör görebilsin)."""
    _write_log(_overconfident_rows())
    calibration_loop.refresh(sources=("pwin",), methods=("platt", "isotonic"))
    d = calibration_loop.load_calibrators()["pwin"]["denemeler"][0]
    for alan in ("brier_base", "base_rate", "taban_brier_fark", "taban_t",
                 "ham_yener", "taban_yener"):
        assert alan in d, alan
    assert d["taban_yener"] is True             # bilgili kaynak tabanı gerçekten yener


def test_calibrate_kayitli_kalibratoru_gercekten_uygular():
    _write_log(_overconfident_rows())
    calibration_loop.refresh(sources=("pwin",), methods=("platt", "isotonic"))
    kalibre = calibration_loop.calibrate("pwin", 0.9)   # aşırı-uçtaki ham p
    assert 0.0 <= kalibre <= 1.0
    assert kalibre != pytest.approx(0.9)   # kalibratör GERÇEKTEN bir dönüşüm uyguladı


# ---------------------------------------------------------------------------
# refresh — kısmi kaynak güncellemesi diğerini silmez; KAPI_GECILEMEDI görünür kalır
# ---------------------------------------------------------------------------
def test_refresh_kismi_kaynak_digerinin_kaydini_silmez():
    calibration_loop._save_artifact({"neural": {"durum": "YETERSIZ", "not": "eski-kayit"}})
    _write_log(_seeded_rows(600, seed=13))
    calibration_loop.refresh(sources=("pwin",), methods=("platt",))
    persisted = calibration_loop.load_calibrators()
    assert persisted["neural"]["durum"] == "YETERSIZ"
    assert persisted["neural"]["not"] == "eski-kayit"
    assert "pwin" in persisted


def test_refresh_neural_n6_artefaktta_yetersiz_gorunur_kalir():
    rows = [_row(i, neural_pwin=f"{0.5 + 0.02 * i:.2f}", neural_pwin_side="LONG",
                 fwd=("1.0" if i % 2 == 0 else "-1.0")) for i in range(6)]
    _write_log(rows)
    out = calibration_loop.refresh(sources=("neural",))
    assert out["neural"]["durum"] == "YETERSIZ"
    assert "calibrator" not in out["neural"]


# ---------------------------------------------------------------------------
# B4.6 — bozuk artefakt: load_calibrators()={} , calibrate() p'yi aynen döner, çökme yok
# ---------------------------------------------------------------------------
def test_load_calibrators_bozuk_json_bos_dict():
    calibration_loop.CALIBRATORS_FILE.write_text("{bozuk json", encoding="utf-8")
    assert calibration_loop.load_calibrators() == {}


def test_load_calibrators_dosya_yok_bos_dict():
    assert calibration_loop.load_calibrators() == {}


def test_load_calibrators_json_liste_ise_bos_dict():
    calibration_loop.CALIBRATORS_FILE.write_text("[1, 2, 3]", encoding="utf-8")
    assert calibration_loop.load_calibrators() == {}


def test_calibrate_bozuk_artefaktta_p_aynen_doner():
    calibration_loop.CALIBRATORS_FILE.write_text("{bozuk json", encoding="utf-8")
    assert calibration_loop.calibrate("pwin", 0.42) == pytest.approx(0.42)


def test_calibrate_kayit_yokken_p_aynen_doner():
    assert calibration_loop.calibrate("pwin", 0.37) == pytest.approx(0.37)


def test_calibrate_kapi_gecilemedi_durumunda_p_aynen_doner():
    calibration_loop._save_artifact({"pwin": {"durum": "KAPI_GECILEMEDI"}})
    assert calibration_loop.calibrate("pwin", 0.61) == pytest.approx(0.61)


def test_calibrate_bozuk_p_value_error():
    with pytest.raises(ValueError):
        calibration_loop.calibrate("pwin", float("nan"))


# ---------------------------------------------------------------------------
# B4.7 — atomik yazım kullanılır (monkeypatch ile doğrulanır)
# ---------------------------------------------------------------------------
def test_refresh_atomic_write_text_kullanir(monkeypatch):
    calls = []

    def fake_atomic_write(path, text, **kw):
        calls.append((path, text))

    monkeypatch.setattr(calibration_loop, "atomic_write_text", fake_atomic_write)
    calibration_loop.refresh(sources=("pwin",), methods=("platt",))
    assert len(calls) == 1
    assert calls[0][0] == calibration_loop.CALIBRATORS_FILE
    json.loads(calls[0][1])   # geçerli JSON metni üretildi


def test_save_artifact_oserror_cokmez(monkeypatch):
    def kaboom(path, text, **kw):
        raise OSError("disk dolu (sentetik)")

    monkeypatch.setattr(calibration_loop, "atomic_write_text", kaboom)
    calibration_loop._save_artifact({"pwin": {"durum": "YETERSIZ"}})   # çökmemeli, yalnız uyarı loglar


# ---------------------------------------------------------------------------
# B4.8 — boş/başlıksız predictions_log: çökme yok, 'ölçülemedi'
# ---------------------------------------------------------------------------
def test_bos_log_cokme_yok_ve_olculemedi_raporu():
    assert calibration_loop._load_pairs("pwin") == []
    r = calibration_loop.evaluate_source("pwin", "platt")
    assert r["durum"] == "YETERSIZ"
    rep = calibration_loop.calibration_report()
    assert "ölçülemedi" in rep


def test_basliksiz_bos_dosya_cokme_yok():
    calibration_loop.PRED_LOG.write_text("", encoding="utf-8")
    assert calibration_loop._load_pairs("pwin") == []
    assert calibration_loop._load_pairs("neural") == []


def test_refresh_veri_yokken_cokmez_yetersiz_yazar():
    out = calibration_loop.refresh(sources=("pwin", "neural"), methods=("platt", "isotonic"))
    assert out["pwin"]["durum"] == "YETERSIZ"
    assert out["neural"]["durum"] == "YETERSIZ"


# ---------------------------------------------------------------------------
# calibration_report — metin biçimi
# ---------------------------------------------------------------------------
def test_calibration_report_kayitli_satiri():
    calibration_loop._save_artifact({"pwin": {
        "durum": "KAYITLI", "method": "platt", "n_fit": 300, "n_oos": 250,
        "brier_raw": 0.25, "brier_cal": 0.20, "ece_raw": 0.05, "ece_cal": 0.03,
        "delta_brier": -0.05, "delta_ece": -0.02}})
    rep = calibration_loop.calibration_report()
    assert "KAYITLI" in rep and "platt" in rep and "0.25" in rep and "pwin" in rep


def test_calibration_report_kapi_gecilemedi_satiri():
    calibration_loop._save_artifact({"pwin": {"durum": "KAPI_GECILEMEDI"}})
    rep = calibration_loop.calibration_report()
    assert "KAPI_GECILEMEDI" in rep


def test_calibration_report_yetersiz_satiri():
    calibration_loop._save_artifact({"neural": {"durum": "YETERSIZ", "not": "n_oos=6 < min_n=200"}})
    rep = calibration_loop.calibration_report()
    assert "YETERSIZ" in rep and "neural" in rep


def test_calibration_report_kenar_iddiasi_icermez():
    calibration_loop._save_artifact({"pwin": {
        "durum": "KAYITLI", "method": "platt", "n_fit": 300, "n_oos": 250,
        "brier_raw": 0.25, "brier_cal": 0.20, "ece_raw": 0.05, "ece_cal": 0.03,
        "delta_brier": -0.05, "delta_ece": -0.02}})
    rep = calibration_loop.calibration_report()
    assert "kenar kanıtı değildir" in rep


# ---------------------------------------------------------------------------
# CLI (main) — argparse doğru bağlanmış mı
# ---------------------------------------------------------------------------
def test_main_bos_argv_yardim_basar_ve_cokmez(capsys):
    rc = calibration_loop.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "usage" in out.lower()


def test_main_report_veri_yokken_olculemedi(capsys):
    rc = calibration_loop.main(["--report"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ölçülemedi" in out


def test_main_refresh_json_gecerli_json_uretir(capsys):
    _write_log(_seeded_rows(600, seed=29))
    rc = calibration_loop.main(["--refresh", "--source", "pwin", "--method", "platt",
                                "--method", "isotonic", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "pwin" in data


def test_main_refresh_min_n_ve_split_override(capsys):
    _write_log(_seeded_rows(60, seed=31))
    rc = calibration_loop.main(["--refresh", "--source", "pwin", "--min-n", "10",
                                "--split", "0.5", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["pwin"]["durum"] in ("KAYITLI", "KAPI_GECILEMEDI")   # artık YETERSİZ değil (n_oos>=10)
