"""Tur-77 · Madde 5.3 — Venn–Abers aralık kalibrasyonu (`adaptive_calibration`).

Dört soru:
  1. **Matematik doğru mu?** Elle hesaplanmış kapalı-form örnekler + her test skorunda
     kalibrasyon kümesine etiket ekleyip sklearn `IsotonicRegression`'ı yeniden uyduran
     NAİF tanıma karşı diferansiyel eşleşme (bağlı skorlar, aralık dışı skorlar dahil).
  2. **Nedensel mi?** Yürüyen değerlendirmede bir bloğu puanlayan tablo YALNIZ o bloktan
     önceki örneklerden kurulmalı (look-ahead yasağı).
  3. **Kapı dürüst mü?** Bilgisiz kaynakta VA sabit tabanı YENEMEZ (ACI DERS #9); bilgi taşıyan
     kaynakta yenebilmeli (ölü kapı olmamalı).
  4. **Uydurma aralık yok mu?** Nokta kalibratörden (platt) aralık TÜRETİLMEZ → None.

Yardımcılar `test_ai_adaptive_calibration.py`'den KOPYALANDI (projenin "kopyalandı, import
edilmedi" deseni).
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from nihai.ai import adaptive_calibration as ac
from nihai.ai import calibration_loop

FIELDS = ["logged_at", "symbol", "interval", "bar_ts", "direction", "action", "confidence",
          "prob_cal", "pwin", "pwin_side", "price", "bar_close", "horizon_bars",
          "resolved_at", "fwd_price", "hit", "fwd_ret_pct", "hit_bar", "slip_bps",
          "neural_pwin", "neural_pwin_side", "neural_pwin_std", "dir_bar", "conf_bar"]

_BASE = datetime(2026, 1, 1, 0, 0, 0)


@pytest.fixture(autouse=True)
def _tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_loop, "PRED_LOG", tmp_path / "predictions_log.csv")
    monkeypatch.setattr(ac, "ARTIFACT_FILE", tmp_path / "ai_adaptive_calibration.json")
    return tmp_path


def _row(i: int, *, pwin: str, fwd: str) -> dict:
    bar_ts = (_BASE + timedelta(minutes=i)).isoformat()
    return {"logged_at": bar_ts, "symbol": "TST", "interval": "1m", "bar_ts": bar_ts,
            "direction": "LONG", "action": "LONG", "pwin": pwin, "pwin_side": "LONG",
            "fwd_ret_pct": fwd,
            "resolved_at": (_BASE + timedelta(minutes=i, hours=2)).isoformat()}


def _write_log(rows: list[dict]) -> None:
    with calibration_loop.PRED_LOG.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: "" for k in FIELDS}, **r})


def _bilgisiz_rows(n: int = 2000, seed: int = 7, taban: float = 0.47) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = float(rng.uniform(0.25, 0.75))
        y = 1 if rng.random() < taban else 0
        rows.append(_row(i, pwin=f"{p:.6f}", fwd=("1.0" if y == 1 else "-1.0")))
    return rows


def _bilgili_rows(n: int = 2000, seed: int = 42, overconf: float = 1.8) -> list[dict]:
    rng = np.random.default_rng(seed)
    q_levels = [0.30, 0.40, 0.50, 0.60, 0.70]
    rows = []
    for i in range(n):
        q = q_levels[i % len(q_levels)]
        p = float(min(max(0.5 + overconf * (q - 0.5), 0.02), 0.98))
        y = 1 if rng.random() < q else 0
        rows.append(_row(i, pwin=f"{p:.6f}", fwd=("1.0" if y == 1 else "-1.0")))
    return rows


# ---------------------------------------------------------------------------
# 1 — matematik: kapalı form + naif tanıma karşı diferansiyel eşleşme
# ---------------------------------------------------------------------------
_CAL = [(0.2, 0), (0.4, 0), (0.6, 1), (0.8, 1)]


@pytest.mark.parametrize(("s", "p0", "p1"), [
    (0.5, 0.0, 1.0),            # iki komşunun arasında: y=0 → 0, y=1 → 1 (zaten monoton)
    (0.9, 2.0 / 3.0, 1.0),      # en üstte y=0 → (1,0) havuz 0,5 < önceki 1 → (1,1,0) = 2/3
    (0.1, 0.0, 1.0 / 3.0),      # en altta y=1 → (1,0,0) havuzlanır → 1/3
    (0.4, 0.0, 0.5),            # BAĞ: 0,4 kovası (0+1)/2 = 0,5 (ağırlık 2)
])
def test_kapali_form_ornekleri(s, p0, p1):
    iv = ac.venn_abers_predict(ac.venn_abers_fit(_CAL), [s])[0]
    assert iv is not None
    assert iv[0] == pytest.approx(p0, abs=1e-12)
    assert iv[1] == pytest.approx(p1, abs=1e-12)


def test_birlestirme_kapali_formu_ve_aralik_icinde_kalir():
    assert ac.venn_abers_merge(0.0, 1.0) == pytest.approx(0.5)
    assert ac.venn_abers_merge(2.0 / 3.0, 1.0) == pytest.approx(0.75)
    assert ac.venn_abers_merge(0.0, 1.0 / 3.0) == pytest.approx(0.25)
    rng = np.random.default_rng(3)
    for _ in range(500):
        a, b = sorted(rng.uniform(0, 1, 2))
        m = ac.venn_abers_merge(a, b)
        assert a - 1e-12 <= m <= b + 1e-12


def _naif(cal: list[tuple[float, int]], s: float) -> tuple[float, float]:
    from sklearn.isotonic import IsotonicRegression
    xs = np.array([p for p, _ in cal], dtype=float)
    ys = np.array([y for _, y in cal], dtype=float)
    vals = []
    for etiket in (0.0, 1.0):
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(np.append(xs, s), np.append(ys, etiket))
        vals.append(float(iso.predict([s])[0]))
    return vals[0], vals[1]


def test_naif_tanimla_diferansiyel_eslesme_bagli_skorlar_dahil():
    rng = np.random.default_rng(11)
    q = rng.uniform(0, 1, 300)
    cal = [(float(np.round(v, 2)), int(rng.random() < v)) for v in q]   # 2 hane → bol bağ
    testler = [float(np.round(v, 3)) for v in rng.uniform(-0.05, 1.05, 120)]
    testler += [p for p, _ in cal[:40]]                                  # birebir eşit skorlar
    tablo = ac.venn_abers_fit(cal)
    sonuc = ac.venn_abers_predict(tablo, testler)
    for s, iv in zip(testler, sonuc, strict=True):
        assert iv is not None
        n0, n1 = _naif(cal, s)
        assert iv[0] == pytest.approx(n0, abs=1e-9)
        assert iv[1] == pytest.approx(n1, abs=1e-9)
        assert iv[0] <= iv[1] + 1e-12                                    # yapısal p0 ≤ p1


def test_gecersiz_girdiler_uydurmaz():
    assert ac.venn_abers_fit([]) is None
    assert ac.venn_abers_fit([(0.3, 1)]) is None
    assert ac.venn_abers_fit([(0.3, 1), (0.6, 1), (0.9, 1)]) is None     # tek sınıf
    tablo = ac.venn_abers_fit([(0.3, 0), (float("nan"), 1), (True, 1), (0.7, 1)])
    assert tablo is not None and tablo["n"] == 2                          # NaN ve bool elendi
    assert ac.venn_abers_predict(tablo, [float("nan"), float("inf")]) == [None, None]
    assert ac.venn_abers_predict(None, [0.5]) == [None]
    assert ac.venn_abers_predict({"method": "platt"}, [0.5]) == [None]
    assert ac.venn_abers_predict({"method": "venn_abers", "scores": [0.1], "counts": [0],
                                  "positives": [0]}, [0.5]) == [None]
    assert ac.venn_abers_interval([], 0.5) is None


def test_tek_cagri_araligi_ve_notr_bayragi():
    iv = ac.venn_abers_interval(_CAL, 0.5)
    assert iv == {"p0": 0.0, "p1": 1.0, "p": 0.5, "genislik": 1.0, "notr": True}
    iv2 = ac.venn_abers_interval(_CAL, 0.1)
    assert iv2 is not None and iv2["notr"] is False


def test_determinizm():
    rng = np.random.default_rng(5)
    cal = [(float(v), int(rng.random() < v)) for v in rng.uniform(0, 1, 400)]
    ps = [float(v) for v in rng.uniform(0, 1, 50)]
    a = ac.venn_abers_predict(ac.venn_abers_fit(cal), ps)
    b = ac.venn_abers_predict(ac.venn_abers_fit(cal), ps)
    assert a == b


# ---------------------------------------------------------------------------
# 2 — nedensellik: yürüyen tablo yalnız GEÇMİŞTEN kurulur
# ---------------------------------------------------------------------------
def test_yuruyen_tablo_gelecegi_gormez(monkeypatch):
    _write_log(_bilgisiz_rows(1200))
    triples = calibration_loop._load_pairs("pwin", 60_000)
    triples.sort(key=lambda t: t[0])
    beklenen = [(p, y) for _, p, y in triples]

    gorulen: list[list[tuple[float, int]]] = []
    gercek = ac.venn_abers_fit
    monkeypatch.setattr(ac, "venn_abers_fit",
                        lambda pairs: (gorulen.append(list(pairs)) or gercek(pairs)))

    ac.prequential_eval("pwin", "venn_abers", warmup=200, refit_every=100, min_eval=100)

    yuruyen = gorulen[:-1]            # son çağrı: TÜM veriden kalıcılık tablosu (puana dahil değil)
    assert yuruyen
    for k, pairs in enumerate(yuruyen):
        uzunluk = 200 + k * 100
        assert len(pairs) == uzunluk
        assert pairs == beklenen[:uzunluk]


# ---------------------------------------------------------------------------
# 3 — kapı: bilgisiz kaynakta taban YENİLMEZ, bilgili kaynakta yenilebilir
# ---------------------------------------------------------------------------
def test_bilgisiz_kaynakta_tabani_yenemez():
    _write_log(_bilgisiz_rows(2000))
    r = ac.prequential_eval("pwin", "venn_abers", warmup=500, refit_every=100, min_eval=500)
    assert r["durum"] == "OLCULDU"
    assert r["kazandi"] is False              # bilgi hükmü KAPIDAN gelir, kapsama payından değil
    va = r["venn_abers"]
    assert va["n_aralik"] == r["n_eval"]
    for alan in ("notr_payi", "taban_kapsayan_payi", "kapsama_payi"):
        assert 0.0 <= va[alan] <= 1.0
    assert len(va["kova"]) == 10
    assert r["calibrator"]["method"] == "venn_abers"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_aralik_genisligi_bilgiyi_degil_kume_yogunlugunu_olcer(seed):
    """Ölçülen sınırın kilidi: aynı dağılımda kalibrasyon kümesi 30 kat büyüyünce ortalama
    aralık genişliği 11–13 kat daraldı (Tur-77). Genişliği "bilgisizlik" diye okumak yanlıştır;
    bu yüzden 0,5-kapsama kuralı bir bilgi dedektörü sayılmaz (modül notu)."""
    rng = np.random.default_rng(seed)
    grid = [float(v) for v in np.linspace(0.05, 0.95, 37)]

    def ort_genislik(n: int) -> float:
        q = rng.uniform(0, 1, n)
        cal = [(float(v), int(rng.random() < v)) for v in q]
        ivs = ac.venn_abers_predict(ac.venn_abers_fit(cal), grid)
        return float(np.mean([b - a for a, b in ivs]))       # type: ignore[misc]

    assert ort_genislik(100) > 5.0 * ort_genislik(3000)


def test_bilgili_kaynakta_tabani_yenebilir():
    _write_log(_bilgili_rows(2500))
    r = ac.prequential_eval("pwin", "venn_abers", warmup=500, refit_every=100, min_eval=500)
    assert r["durum"] == "OLCULDU"
    assert r["taban_yener"] is True
    assert r["venn_abers"]["notr_payi"] < 0.9


def test_yetersiz_orneklem_verdikt_uretmez():
    _write_log(_bilgisiz_rows(300))
    r = ac.prequential_eval("pwin", "venn_abers", warmup=200, refit_every=100, min_eval=500)
    assert r["durum"] == "YETERSIZ"
    assert "venn_abers" not in r


# ---------------------------------------------------------------------------
# 4 — uydurma aralık yok + CLI bağlantısı
# ---------------------------------------------------------------------------
def test_calibrate_interval_yalniz_kayitli_va_icin():
    assert ac.calibrate_interval("pwin", 0.5) is None                  # artefakt yok
    tablo = ac.venn_abers_fit(_CAL)
    ac.ARTIFACT_FILE.write_text(json.dumps({"pwin": {"durum": "KAYITLI", "calibrator": tablo}}),
                                encoding="utf-8")
    iv = ac.calibrate_interval("pwin", 0.5)
    assert iv is not None and iv["notr"] is True and iv["p"] == pytest.approx(0.5)
    assert ac.calibrate("pwin", 0.1) == pytest.approx(0.25)
    platt = {"method": "platt", "coef": 1.0, "intercept": 0.0}
    ac.ARTIFACT_FILE.write_text(json.dumps({"pwin": {"durum": "KAYITLI", "calibrator": platt}}),
                                encoding="utf-8")
    assert ac.calibrate_interval("pwin", 0.5) is None                  # nokta kalibratörden aralık YOK
    ac.ARTIFACT_FILE.write_text(json.dumps({"pwin": {"durum": "KAPI_GECILEMEDI",
                                                     "calibrator": tablo}}), encoding="utf-8")
    assert ac.calibrate_interval("pwin", 0.5) is None                  # kapıyı geçmemiş kayıt okunmaz


def test_cli_yontemi_taniyor_ve_gunluk_kume_va_icerir():
    # Günlük `--calib` görevi `refresh()`i varsayılanlarla çağırır; VA orada yoksa teşhisi
    # hiçbir rapora ulaşmaz. Platt İLK kalır: `report()` KAPI_GECILEMEDI'de ilk denemeyi basar.
    assert ac.DEFAULT_METHODS == ("platt", "isotonic", "venn_abers")
    _write_log(_bilgisiz_rows(1500))
    assert ac.main(["--eval", "--source", "pwin", "--method", "venn_abers", "--warmup", "500",
                    "--min-eval", "500", "--json"]) == 0
    kayit = ac.load_artifact()["pwin"]
    assert kayit["durum"] == "KAPI_GECILEMEDI"
    assert any(isinstance(d.get("venn_abers"), dict) for d in kayit["denemeler"])
