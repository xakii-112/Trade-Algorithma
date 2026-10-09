"""learning.py testleri — kalibrasyon, Beta posterior, Page-Hinkley, canlı-log öğrenmesi.

v3 EKLERİ (Tur-16, WP-L): sources şeması, fit_calibrator/apply_calibration (yalnız-OOS),
retrain_candidate bayrağı, geriye-uyumluluk (eski state.json'da 'sources' yok)."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from nihai.ai import learning

FIELDS = ["logged_at", "symbol", "interval", "bar_ts", "direction", "action", "confidence",
          "prob_cal", "pwin", "pwin_side", "price", "horizon_bars",
          "resolved_at", "fwd_price", "hit", "fwd_ret_pct"]


@pytest.fixture(autouse=True)
def _tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "PRED_LOG", tmp_path / "predictions_log.csv")
    monkeypatch.setattr(learning, "LEARNING_STATE", tmp_path / "ai_learning_state.json")
    return tmp_path


def _write_log(rows: list[dict]) -> None:
    with learning.PRED_LOG.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: "" for k in FIELDS}, **r})


def _row(direction="LONG", hit="1", resolved="2026-07-19T10:00:00", pwin="", pwin_side="", fwd=""):
    return {"logged_at": "2026-07-19T09:00:00", "symbol": "TST", "interval": "1h",
            "bar_ts": "2026-07-19T09:00:00", "direction": direction, "action": direction,
            "resolved_at": resolved, "hit": hit, "pwin": pwin, "pwin_side": pwin_side,
            "fwd_ret_pct": fwd}


# --- kalibrasyon -----------------------------------------------------------
def test_brier_bilinen_degerler():
    assert learning.brier_score([(1.0, 1)]) == 0.0
    assert learning.brier_score([(0.5, 1)]) == 0.25
    assert learning.brier_score([(0.5, 1), (0.5, 0)]) == 0.25
    assert learning.brier_score([]) is None
    assert learning.brier_score([(float("nan"), 1)]) is None


def test_reliability_bins():
    bins = learning.reliability_bins([(0.05, 0), (0.95, 1), (0.92, 1)], n_bins=10)
    assert len(bins) == 10
    assert bins[0]["n"] == 1 and bins[0]["y_rate"] == 0.0
    assert bins[9]["n"] == 2 and bins[9]["y_rate"] == 1.0
    assert bins[5]["n"] == 0 and bins[5]["p_mean"] is None


# --- Beta posterior --------------------------------------------------------
def test_beta_posterior_elle():
    mean, lo, hi = learning.beta_posterior(2, 0)          # uniform prior → Beta(3,1)
    assert mean == pytest.approx(0.75)
    assert 0.0 < lo < mean < hi < 1.0
    mean0, lo0, hi0 = learning.beta_posterior(0, 0)
    assert mean0 == pytest.approx(0.5)
    assert hi0 - lo0 > 0.9                                # kanıt yok → çok geniş CI
    assert learning.beta_posterior(-5, -5)[0] == pytest.approx(0.5)   # negatif kırpılır


# --- Page-Hinkley ----------------------------------------------------------
def test_page_hinkley_sabit_seride_yok():
    assert learning.page_hinkley([0.5] * 100)["drift"] is False
    assert learning.page_hinkley([0.5, 0.5])["drift"] is False        # kısa seri


def test_page_hinkley_kirilmada_var():
    seri = [0.55] * 50 + [0.35] * 50
    out = learning.page_hinkley(seri)
    assert out["drift"] is True and out["index"] is not None and out["index"] >= 50
    yukari = [0.35] * 50 + [0.60] * 50
    assert learning.page_hinkley(yukari)["drift"] is True


def test_rolling_mean():
    assert learning._rolling_mean([1.0, 2.0, 3.0, 4.0], 2) == [1.5, 2.5, 3.5]
    assert learning._rolling_mean([1.0], 5) == []


# --- canlı-log öğrenmesi ---------------------------------------------------
def test_update_dosya_yokken_not_birakir():
    st = learning.update_from_predictions()
    assert st["strategies"] == {}
    assert st["note"] and learning.LEARNING_STATE.exists()


def test_onizleme_ve_yazma_hatasi_kalici_hafizayi_degistirmez(monkeypatch):
    from nihai.web.api_extra import ai_learning

    _write_log([_row("LONG", "1") for _ in range(3)])
    assert learning.update_from_predictions()["saved"] is True
    before = learning.LEARNING_STATE.read_bytes()
    _write_log([_row("SHORT", "0") for _ in range(4)])
    preview = ai_learning(tail=200)["state"]
    assert preview["saved"] is False
    assert preview["strategies"]["GLOBAL"]["n"] == 4
    assert learning.LEARNING_STATE.read_bytes() == before

    learning.PRED_LOG.unlink()
    assert learning.update_from_predictions(persist=False)["note"]
    assert learning.LEARNING_STATE.read_bytes() == before

    def locked(*args, **kwargs):
        raise OSError("locked")

    monkeypatch.setattr(learning, "atomic_write_text", locked)
    assert learning.update_from_predictions()["saved"] is False
    _write_log([_row()])
    assert learning.update_from_predictions()["saved"] is False
    assert learning.LEARNING_STATE.read_bytes() == before


def test_update_sentetik_logdan_sayim():
    rows = [_row("LONG", "1") for _ in range(6)] + [_row("LONG", "0") for _ in range(2)]
    rows += [_row("SHORT", "0") for _ in range(3)]
    rows += [_row("LONG", "", resolved="")]                    # bekleyen → sayılmaz
    rows += [_row("LONG", "", resolved="UNRESOLVABLE")]        # terminal → sayılmaz
    # pwin kalibrasyonu: taraf-işaretli getiri mantığı
    rows += [_row("LONG", "1", pwin="0.60", pwin_side="LONG", fwd="1.5")]    # y=1
    rows += [_row("LONG", "1", pwin="0.40", pwin_side="SHORT", fwd="1.5")]   # side_ret=-1.5 → y=0
    _write_log(rows)
    st = learning.update_from_predictions(tail_n=500)
    g = st["strategies"]["GLOBAL"]
    assert g["n"] == 13 and g["win"] == 8 and g["loss"] == 5
    lng = st["strategies"]["LONG"]
    assert lng["n"] == 10 and lng["win"] == 8
    assert st["strategies"]["SHORT"]["loss"] == 3
    cal = st["pwin_calibration"]
    assert cal["n"] == 2
    assert cal["brier"] == pytest.approx(((0.6 - 1) ** 2 + (0.4 - 0) ** 2) / 2, abs=1e-4)
    # durum dosyası atomik yazıldı ve yeniden yüklenebilir
    disk = json.loads(learning.LEARNING_STATE.read_text(encoding="utf-8"))
    assert disk["strategies"]["GLOBAL"]["n"] == 13
    assert learning.load_state()["strategies"]["LONG"]["win"] == 8


def test_update_gercek_sema_arti_eksi_yon():
    """Canlı log direction'ı '+'/'-' yazar (LONG/SHORT değil) — normalizasyon şart."""
    rows = [_row("+", "1") for _ in range(4)] + [_row("-", "0") for _ in range(2)]
    # side==direction (+→LONG) ve fwd>0 → taraf kazandı (y=1)
    rows += [_row("+", "1", pwin="0.70", pwin_side="LONG", fwd="2.0")]
    # side!=direction (-→SHORT vs LONG) → side_ret=-2.0 → y=0
    rows += [_row("-", "1", pwin="0.30", pwin_side="LONG", fwd="2.0")]
    _write_log(rows)
    st = learning.update_from_predictions(tail_n=500)
    assert st["strategies"]["LONG"]["n"] == 5 and st["strategies"]["LONG"]["win"] == 5
    assert st["strategies"]["SHORT"]["n"] == 3
    cal = st["pwin_calibration"]
    assert cal["n"] == 2
    assert cal["brier"] == pytest.approx(((0.7 - 1) ** 2 + (0.3 - 0) ** 2) / 2, abs=1e-4)


def test_load_state_bozuk_dosyada_varsayilan():
    learning.LEARNING_STATE.write_text("{bozuk", encoding="utf-8")
    assert learning.load_state() == {"strategies": {}, "updated_at": None}


def test_learning_report_bos_ve_dolu():
    bos = learning.learning_report({})
    assert "nötr" in bos
    st = {"strategies": {"GLOBAL": {"n": 5, "win": 3, "loss": 2, "post_mean": 0.57,
                                    "lo95": 0.25, "hi95": 0.85}},
          "drift": {"drift": True, "window": 50}}
    dolu = learning.learning_report(st)
    assert "GLOBAL" in dolu and "DRIFT" in dolu
    assert "kenar kanıtı değildir" in dolu


# ---------------------------------------------------------------------------
# v3 — sources şeması (SPEC §3.2, WP-D hiyerarşik birleşim kaynağı: prior/pwin/neural)
# ---------------------------------------------------------------------------
def test_sources_varsayilan_yapi_dosya_yokken():
    st = learning.update_from_predictions()
    # v4 (Tur-22): her kaynak Brier'in YANINDA ECE taşır (yalnız-kalibrasyon ölçüsü).
    # v5 (Tur-32): + BSS (sabit tabana karşı beceri) ve resolution (Murphy bilgi terimi).
    bos = {"brier": None, "ece": None, "bss": None, "resolution": None, "n": 0}
    assert st["sources"] == {"prior": bos, "pwin": bos, "neural": bos}
    assert st["retrain_candidate"] is False


def test_sources_pwin_pwin_calibration_ile_ayni():
    rows = [_row("LONG", "1", pwin="0.60", pwin_side="LONG", fwd="1.5") for _ in range(3)]
    rows += [_row("LONG", "1", pwin="0.40", pwin_side="SHORT", fwd="1.5") for _ in range(2)]
    _write_log(rows)
    st = learning.update_from_predictions(tail_n=500)
    assert st["sources"]["pwin"]["brier"] == st["pwin_calibration"]["brier"]
    assert st["sources"]["pwin"]["n"] == st["pwin_calibration"]["n"] == 5


def test_sources_prior_global_post_mean_ile_hesaplaniyor():
    rows = [_row("LONG", "1") for _ in range(8)] + [_row("LONG", "0") for _ in range(2)]
    _write_log(rows)
    st = learning.update_from_predictions(tail_n=500)
    g = st["strategies"]["GLOBAL"]
    beklenen = learning.brier_score([(g["post_mean"], 1)] * 8 + [(g["post_mean"], 0)] * 2)
    assert st["sources"]["prior"]["brier"] == pytest.approx(round(beklenen, 4))
    assert st["sources"]["prior"]["n"] == 10


def test_sources_neural_kolon_yoksa_none():
    rows = [_row("LONG", "1") for _ in range(5)]
    _write_log(rows)   # FIELDS'te neural_pwin YOK — mevcut/eski şema
    st = learning.update_from_predictions(tail_n=500)
    assert st["sources"]["neural"] == {"brier": None, "ece": None, "bss": None,
                                       "resolution": None, "n": 0}


# ---------------------------------------------------------------------------
# v4 (Tur-22) — ECE: yalnız-kalibrasyon ölçüsü (Brier'in ayrıştıramadığı eksen)
# ---------------------------------------------------------------------------
def test_ece_kusursuz_kalibre_sifir():
    """Kova içindeki tahmin ortalaması gerçekleşen oranla eşitse ECE tam 0'dır —
    model ayırt ETMESE bile (0.5 diyen bir model, gerçekten %50 kazanıyorsa)."""
    ciftler = [(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)]
    assert learning.expected_calibration_error(ciftler) == pytest.approx(0.0)


def test_ece_tam_ters_bir():
    assert learning.expected_calibration_error([(1.0, 0)] * 4) == pytest.approx(1.0)


def test_ece_bos_ve_gecersiz_none():
    assert learning.expected_calibration_error([]) is None
    assert learning.expected_calibration_error([(float("nan"), 1), (0.5, 5)]) is None


def test_ece_brierden_farkli_ekseni_olcer():
    """AYIRT EDEN ama AŞIRI-GÜVENLİ model: Brier düşük değil ama asıl sorun kalibrasyon.
    Kusursuz kalibre-ama-ayırt etmeyen modelle karşılaştırıldığında ECE ayrımı gösterir."""
    asiri_guvenli = [(0.95, 1), (0.95, 0), (0.05, 1), (0.05, 0)]      # gerçek oran her kovada 0.5
    kalibre = [(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)]
    assert learning.expected_calibration_error(asiri_guvenli) > 0.4
    assert learning.expected_calibration_error(kalibre) == pytest.approx(0.0)


def test_ece_kovalari_reliability_bins_ile_tutarli():
    """Tek doğruluk kaynağı: ECE `reliability_bins` kovalarından türer — pano eğrisiyle
    tek-sayılı özet birbirini tutmalı."""
    ciftler = [(0.15, 0), (0.25, 1), (0.85, 1), (0.95, 1)]
    bins = learning.reliability_bins(ciftler, n_bins=10)
    toplam = sum(b["n"] for b in bins)
    beklenen = sum((b["n"] / toplam) * abs(b["p_mean"] - b["y_rate"])
                   for b in bins if b["n"])
    assert learning.expected_calibration_error(ciftler) == pytest.approx(beklenen)


def test_learning_report_kaynak_karnesi_satiri():
    rows = [_row("LONG", "1", pwin="0.60", pwin_side="LONG", fwd="1.5") for _ in range(3)]
    rows += [_row("LONG", "0", pwin="0.40", pwin_side="LONG", fwd="-1.5") for _ in range(2)]
    _write_log(rows)
    st = learning.update_from_predictions(tail_n=500)
    rapor = learning.learning_report(st)
    assert "ECE=" in rapor
    assert "kaynak karnesi:" in rapor


def test_sources_neural_kolon_varsa_dolduruluyor():
    """neural_pwin/neural_pwin_side (Dalga-2 köprüsü) mevcutsa pwin ile AYNI mantıkla
    (taraf-işaretli getiri → isabet) hesaplanır; kolonlar yoksa None kalır (üstteki test)."""
    fields = FIELDS + ["neural_pwin", "neural_pwin_side"]
    rows = []
    for _ in range(4):
        r = _row("LONG", "1", fwd="1.5")
        r["neural_pwin"], r["neural_pwin_side"] = "0.65", "LONG"
        rows.append(r)
    for _ in range(2):
        r = _row("LONG", "0", fwd="1.5")
        r["neural_pwin"], r["neural_pwin_side"] = "0.30", "SHORT"
        rows.append(r)
    with learning.PRED_LOG.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: "" for k in fields}, **r})
    st = learning.update_from_predictions(tail_n=500)
    assert st["sources"]["neural"]["n"] == 6
    assert st["sources"]["neural"]["brier"] is not None


# ---------------------------------------------------------------------------
# v3 — retrain_candidate bayrağı (otomatik eğitim YOK, yalnız bayrak+rapor satırı)
# ---------------------------------------------------------------------------
def test_retrain_candidate_false_yeterli_veri_drift_yok():
    rows = [_row("LONG", "1") for _ in range(60)]
    _write_log(rows)
    st = learning.update_from_predictions(tail_n=500)
    assert st["drift"]["drift"] is False
    assert st["retrain_candidate"] is False


def test_retrain_candidate_true_drift_tetiklenince():
    rows = [_row("LONG", "1") for _ in range(70)] + [_row("LONG", "0") for _ in range(70)]
    _write_log(rows)
    st = learning.update_from_predictions(tail_n=500)
    assert st["drift"]["drift"] is True
    assert st["retrain_candidate"] is True


def test_retrain_candidate_drift_yokken_state_alaninda_false():
    st = learning.update_from_predictions()   # log yok → drift=None
    assert st["drift"] is None
    assert st["retrain_candidate"] is False


def test_learning_report_retrain_adayi_satiri_varken():
    st = {"strategies": {}, "drift": {"drift": True, "window": 50}, "retrain_candidate": True}
    rapor = learning.learning_report(st)
    assert "RETRAIN ADAYI" in rapor


def test_learning_report_retrain_adayi_satiri_yokken():
    st = {"strategies": {}, "drift": {"drift": False}, "retrain_candidate": False}
    rapor = learning.learning_report(st)
    assert "RETRAIN ADAYI" not in rapor


# ---------------------------------------------------------------------------
# v3 — fit_calibrator / apply_calibration (YALNIZ verilen OOS çiftlerinde fit)
# ---------------------------------------------------------------------------
def test_fit_calibrator_bilinmeyen_method_value_error():
    with pytest.raises(ValueError):
        learning.fit_calibrator([(0.5, 1), (0.6, 0)], method="dogrusal-olmayan")


def test_fit_calibrator_bos_liste_none():
    assert learning.fit_calibrator([], method="platt") is None


def test_fit_calibrator_tek_veri_none():
    assert learning.fit_calibrator([(0.5, 1)], method="platt") is None


def test_fit_calibrator_tek_sinif_none():
    assert learning.fit_calibrator([(0.4, 1), (0.6, 1), (0.5, 1)], method="isotonic") is None


def test_fit_calibrator_gecersiz_ciftler_elense_yetersiz_kalirsa_none():
    assert learning.fit_calibrator([(float("nan"), 1), (0.5, 2)], method="platt") is None


def test_fit_calibrator_platt_alanlari():
    ys = [0] * 20 + [1] * 20
    ps = [0.30 + 0.02 * i for i in range(20)] + [0.50 + 0.02 * i for i in range(20)]
    res = learning.fit_calibrator(list(zip(ps, ys, strict=True)), method="platt")
    assert res is not None
    for key in ("method", "n", "coef", "intercept", "brier_once", "brier_sonra"):
        assert key in res
    assert res["method"] == "platt"
    assert res["n"] == 40


def test_fit_calibrator_isotonic_brier_iyilesir_veya_esit():
    """İzotonik regresyon AYNI (x-sıralı) veri üstünde SSE'yi minimize eder — ham p zaten
    x-sırasıyla monoton-artan olduğundan izotonik arama uzayında bir ADAY sayılır; bu yüzden
    brier_sonra <= brier_once MATEMATİKSEL OLARAK garantidir (şans/veri-bağımlı değil)."""
    ys = [0] * 20 + [1] * 20
    ps = [0.30 + 0.02 * i for i in range(20)] + [0.50 + 0.02 * i for i in range(20)]
    res = learning.fit_calibrator(list(zip(ps, ys, strict=True)), method="isotonic")
    assert res is not None
    assert res["brier_sonra"] <= res["brier_once"] + 1e-9


def test_apply_calibration_calib_none_p_ayenen_doner():
    assert learning.apply_calibration(0.42, None) == pytest.approx(0.42)


def test_apply_calibration_calib_dict_degil_p_ayenen_doner():
    assert learning.apply_calibration(0.42, "not-a-dict") == pytest.approx(0.42)


def test_apply_calibration_bozuk_p_value_error():
    with pytest.raises(ValueError):
        learning.apply_calibration(float("nan"), None)


def test_apply_calibration_platt_elle_hesap():
    calib = {"method": "platt", "coef": 2.0, "intercept": -1.0}
    beklenen = 1.0 / (1.0 + np.exp(-(2.0 * 0.5 - 1.0)))
    assert learning.apply_calibration(0.5, calib) == pytest.approx(beklenen)


def test_apply_calibration_platt_eksik_katsayi_p_ayenen_doner():
    assert learning.apply_calibration(0.5, {"method": "platt"}) == pytest.approx(0.5)


def test_apply_calibration_isotonic_np_interp_ile_ayni():
    calib = {"method": "isotonic", "x_thresholds": [0.0, 0.5, 1.0], "y_thresholds": [0.1, 0.5, 0.9]}
    beklenen = float(np.interp(0.25, [0.0, 0.5, 1.0], [0.1, 0.5, 0.9]))
    assert learning.apply_calibration(0.25, calib) == pytest.approx(beklenen)


def test_apply_calibration_isotonic_esiksiz_p_ayenen_doner():
    assert learning.apply_calibration(0.5, {"method": "isotonic"}) == pytest.approx(0.5)


def test_apply_calibration_bilinmeyen_method_p_ayenen_doner():
    assert learning.apply_calibration(0.3, {"method": "gizemli"}) == pytest.approx(0.3)


def test_fit_calibrator_apply_calibration_ucdan_uca_platt():
    """fit_calibrator çıktısı doğrudan apply_calibration'a beslenebilir (uçtan-uca sözleşme)."""
    ys = [0] * 15 + [1] * 15
    ps = [0.35] * 15 + [0.65] * 15
    calib = learning.fit_calibrator(list(zip(ps, ys, strict=True)), method="platt")
    kalibre = learning.apply_calibration(0.65, calib)
    assert 0.0 <= kalibre <= 1.0


# ---------------------------------------------------------------------------
# v3 — geriye-uyumluluk (eski state.json şemasında 'sources'/'retrain_candidate' YOK)
# ---------------------------------------------------------------------------
def test_load_state_eski_semada_sources_eksik_cokmez():
    eski = {"strategies": {"GLOBAL": {"n": 5, "win": 3, "loss": 2, "post_mean": 0.6,
                                      "lo95": 0.2, "hi95": 0.9}},
            "updated_at": "2026-01-01T00:00:00+00:00"}   # 'sources' YOK — eski kayıt
    learning.LEARNING_STATE.write_text(json.dumps(eski), encoding="utf-8")
    loaded = learning.load_state()
    assert loaded["strategies"]["GLOBAL"]["n"] == 5
    assert "sources" not in loaded   # load_state() sahte alan ENJEKTE ETMEZ — okuyan degrade eder


def test_learning_report_eski_semada_sources_yokken_calisir():
    eski = {"strategies": {}, "drift": None}   # retrain_candidate/sources YOK
    rapor = learning.learning_report(eski)
    assert "RETRAIN ADAYI" not in rapor
    assert "kenar kanıtı değildir" in rapor


def test_update_from_predictions_mevcut_alanlar_v3_ile_bozulmaz():
    """v3 ekleri (sources/retrain_candidate) MEVCUT alanları (strategies/pwin_calibration/
    drift/note) bozmaz — geriye-uyum bütünsel testi."""
    rows = [_row("LONG", "1") for _ in range(6)] + [_row("LONG", "0") for _ in range(2)]
    rows += [_row("SHORT", "0") for _ in range(3)]
    _write_log(rows)
    st = learning.update_from_predictions(tail_n=500)
    assert st["strategies"]["GLOBAL"]["n"] == 11
    assert st["strategies"]["LONG"]["win"] == 6
    assert st["strategies"]["SHORT"]["loss"] == 3
    assert "sources" in st
    assert "retrain_candidate" in st


# ===========================================================================
# v5 (Tur-32) — beceri skorları: BSS · Murphy ayrışması · log-kayıp
# ===========================================================================
def test_log_loss_bilinen_degerler():
    assert learning.log_loss_score([(0.5, 1), (0.5, 0)]) == pytest.approx(np.log(2.0))
    assert learning.log_loss_score([(1.0, 1), (0.0, 0)]) == pytest.approx(0.0, abs=1e-5)
    assert learning.log_loss_score([]) is None
    assert learning.log_loss_score([("x", 1), (0.5, 3)]) is None     # geçersizler atlanır


def test_log_loss_hatali_kesinligi_brierden_sert_cezalandirir():
    """Aynı iki tahminci: biri temkinli-yanlış, diğeri emin-yanlış. Brier farkı ılımlı,
    log-kayıp farkı büyüktür — 'fazla-güvenli model' tam olarak burada görünür."""
    temkinli = [(0.4, 1)] * 100
    emin = [(0.02, 1)] * 100
    assert learning.brier_score(emin) / learning.brier_score(temkinli) < 3
    assert learning.log_loss_score(emin) / learning.log_loss_score(temkinli) > 4


def test_brier_skill_score_sabit_tabana_gore():
    """BSS: sabit taban-oran tahmincisi tam 0, mükemmel tahminci 1, taban-altı NEGATİF."""
    pairs_taban = [(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)]
    assert learning.brier_skill_score(pairs_taban) == pytest.approx(0.0)
    assert learning.brier_skill_score([(1.0, 1), (0.0, 0)] * 5) == pytest.approx(1.0)
    kotu = [(0.0, 1), (1.0, 0), (0.0, 1), (1.0, 0)]
    assert learning.brier_skill_score(kotu) < 0


def test_brier_skill_score_tek_sinifli_orneklemde_none():
    """Taban Brier'i 0 ise (hep aynı sonuç) beceri TANIMSIZDIR — 0'a bölme yerine None."""
    assert learning.brier_skill_score([(0.7, 1), (0.6, 1)]) is None
    assert learning.brier_skill_score([]) is None
    assert learning.brier_skill_score([(0.5, 1), (0.5, 0)], base_rate="x") is None


def test_murphy_ayrisma_kimligi():
    """BS = REL − RES + UNC (kova-içi kalan `residual` ile açıkça raporlanır)."""
    rng = np.random.default_rng(3)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int)          # kusursuz kalibre, bilgili
    out = learning.murphy_decomposition(list(zip(p, y, strict=True)))
    assert out["brier"] == pytest.approx(
        out["reliability"] - out["resolution"] + out["uncertainty"] + out["residual"], abs=1e-12)
    assert out["reliability"] < 0.01                # kalibre → REL küçük
    assert out["resolution"] > 0.05                 # bilgili → RES belirgin


def test_murphy_resolution_bilgisiz_tahmincide_sifira_yakin():
    """Bu projenin en çok ihtiyaç duyduğu ayrım: sabit tahminci Brier'de idare eder ama
    ÇÖZÜNÜRLÜĞÜ (bilgi) sıfırdır (ACI DERS #9)."""
    rng = np.random.default_rng(5)
    y = (rng.random(2000) < 0.45).astype(int)
    sabit = [(0.45, int(v)) for v in y]
    out = learning.murphy_decomposition(sabit)
    assert out["resolution"] == pytest.approx(0.0, abs=1e-12)
    assert out["uncertainty"] == pytest.approx(0.45 * 0.55, abs=0.02)
    assert learning.murphy_decomposition([]) is None


def test_source_card_bss_ve_resolution_tasiyor():
    """Kaynak karnesi Tur-32'de iki alan kazandı; boş kaynakta hepsi None (şema sabit)."""
    kart = learning._source_card([(0.9, 1), (0.1, 0)] * 20)
    assert set(kart) == {"brier", "ece", "bss", "resolution", "n"}
    assert kart["bss"] > 0.9 and kart["resolution"] > 0.2
    bos = learning._source_card([])
    assert bos["n"] == 0 and bos["bss"] is None and bos["resolution"] is None
