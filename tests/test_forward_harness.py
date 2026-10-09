"""forward_harness + prediction_tracker pwin-şema migrasyonu testleri.

Değişmezler: metrikler doğru ve çökmüyor; örneklem kapısı altında ASLA kenar iddiası
yok; history append-only; eski (pwin'siz) log append'te bozulmadan yükseliyor."""
import csv
import sys

import numpy as np
import pytest

import nihai.execution.prediction_tracker as pt
import research.forward_harness as fh


# ---------------------------------------------------------------------------
# Saf metrikler
# ---------------------------------------------------------------------------
def test_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1.0])
    assert fh._auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0      # mükemmel ayrım
    assert fh._auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5      # hepsi eşit → 0.5
    assert fh._auc(np.array([1, 1.0]), np.array([0.3, 0.7])) is None  # tek sınıf → None


def test_brier_reference():
    y = np.array([0, 1, 0, 1.0])
    assert fh._brier(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.25)
    assert fh._brier(y, y) == 0.0  # mükemmel


def test_ece_perfectly_calibrated_is_low():
    # p == gerçek frekans → ECE ~0
    y = np.array([0, 0, 1, 1.0] * 25)
    p = np.where(y == 1, 0.95, 0.05)
    assert fh._ece(y, p) < 0.1


# ---------------------------------------------------------------------------
# Örneklem kapısı — KRİTİK dürüstlük değişmezi
# ---------------------------------------------------------------------------
def test_verdict_gate_blocks_edge_claim_on_small_sample(monkeypatch):
    monkeypatch.setattr(fh, "MIN_RESOLVED", 500)
    monkeypatch.setattr(fh, "MIN_DAYS", 10.0)
    ev = {"resolved": 50, "days_span": 2.0, "hit_rate": 0.70}  # cazip ama küçük
    v = fh._verdict(ev)
    assert v["label"] == "INSUFFICIENT"  # 70% isabete RAĞMEN kenar iddia etmez


def test_verdict_no_edge_when_near_coin_flip(monkeypatch):
    monkeypatch.setattr(fh, "MIN_RESOLVED", 10)
    monkeypatch.setattr(fh, "MIN_DAYS", 0.0)
    monkeypatch.setattr(fh, "EDGE_PP", 3.0)
    assert fh._verdict({"resolved": 600, "days_span": 30, "hit_rate": 0.49})["label"] == "NO_EDGE"


def test_verdict_inverted_and_weak_positive(monkeypatch):
    monkeypatch.setattr(fh, "MIN_RESOLVED", 10)
    monkeypatch.setattr(fh, "MIN_DAYS", 0.0)
    monkeypatch.setattr(fh, "EDGE_PP", 3.0)
    assert fh._verdict({"resolved": 600, "days_span": 30, "hit_rate": 0.42})["label"] == "INVERTED"
    assert fh._verdict({"resolved": 600, "days_span": 30, "hit_rate": 0.58})["label"] == "WEAK_POSITIVE"


def test_verdict_no_data():
    assert fh._verdict({"resolved": 0, "hit_rate": None})["label"] == "NO_DATA"


# ---------------------------------------------------------------------------
# B4 — pwin metrikleri TARAF sonucuyla etiketlenir (yön-isabeti hit ile DEĞİL)
# ---------------------------------------------------------------------------
def test_pwin_label_uses_side_outcome_not_direction():
    """SHORT aday, yön '+', ham +2% → YÖN isabet (hit=1) ama SHORT taraf KAYBEDER →
    pwin etiketi 0. LONG aynı senaryoda kazanır → 1. (hit ile etiketlemek SHORT'ları
    ters ölçerdi = dürüstlük panosu bozulurdu.)"""
    short = {"pwin": "0.7", "pwin_side": "SHORT", "direction": "+", "fwd_ret_pct": "2.0", "hit": "1"}
    long = {"pwin": "0.7", "pwin_side": "LONG", "direction": "+", "fwd_ret_pct": "2.0", "hit": "1"}
    assert fh._pwin_label(short) == 0 and int(short["hit"]) == 1   # taraf≠yön
    assert fh._pwin_label(long) == 1
    # pwin_side yok (eski şema) → None (dürüstçe dışarıda)
    assert fh._pwin_label({"pwin": "0.7", "direction": "+", "fwd_ret_pct": "2.0", "hit": "1"}) is None


def test_col_pwin_short_labels_are_side_not_hit():
    """_col pwin sütununda label_fn=_pwin_label → SHORT (yön isabetli ama taraf kaybeden)
    satırların etiketi 0; aynı satırlar hit ile etiketlense 1 (yanlış) olurdu."""
    rows = [{"pwin": "0.7", "pwin_side": "SHORT", "direction": "+",
             "fwd_ret_pct": "2.0", "hit": "1"} for _ in range(25)]
    y_side, _p = fh._col(rows, "pwin", label_fn=fh._pwin_label)
    assert (y_side == 0).all() and len(y_side) == 25   # taraf: SHORT + fiyat yukarı = kayıp
    y_hit, _ = fh._col(rows, "pwin")                   # label_fn yok → hit (yanlış olurdu)
    assert (y_hit == 1).all()


# ---------------------------------------------------------------------------
# Honest-improvement — binom %95 GA nokta-tahmin kenarını doğrular
# ---------------------------------------------------------------------------
def test_verdict_ci_downgrades_small_sample_positive(monkeypatch):
    """Nokta-isabet 0.5 üstünde (edge≥EDGE_PP) ama %95 GA 0.5'i içeriyorsa WEAK_POSITIVE
    DEĞİL NO_EDGE. Aynı isabet büyük örneklemde (GA alt>0.5) WEAK_POSITIVE olur."""
    monkeypatch.setattr(fh, "MIN_RESOLVED", 10)
    monkeypatch.setattr(fh, "MIN_DAYS", 0.0)
    monkeypatch.setattr(fh, "EDGE_PP", 3.0)
    assert fh._verdict({"resolved": 30, "days_span": 30, "hit_rate": 0.60})["label"] == "NO_EDGE"
    assert fh._verdict({"resolved": 1000, "days_span": 30, "hit_rate": 0.60})["label"] == "WEAK_POSITIVE"


def test_wilson_ci_bounds():
    lo, hi = fh._wilson_ci(600, 1000)      # phat 0.6, dar aralık
    assert 0.0 <= lo < 0.6 < hi <= 1.0 and lo > 0.5
    assert fh._wilson_ci(0, 0) == (0.0, 1.0)   # bölme-sıfır güvenli


# ---------------------------------------------------------------------------
# history append-only
# ---------------------------------------------------------------------------
def test_snapshot_appends_history(tmp_path, monkeypatch):
    monkeypatch.setattr(fh, "HISTORY", tmp_path / "forward_eval" / "history.csv")
    ev = {"resolved": 600, "days_span": 12.0, "symbols": 30, "hit_rate": 0.48,
          "avg_signed_ret_pct": -0.02, "auc": {"prob_cal": 0.50, "pwin": 0.51},
          "brier": {"pwin": 0.25}, "ece": {"pwin": 0.04},
          "verdict": {"label": "NO_EDGE"}}
    fh.snapshot(ev)
    fh.snapshot({**ev, "resolved": 700})
    rows = fh.read_history()
    assert len(rows) == 2
    assert rows[0]["resolved"] == "600" and rows[1]["resolved"] == "700"
    assert rows[0]["auc_pwin"] == "0.51" and rows[0]["verdict"] == "NO_EDGE"


# ---------------------------------------------------------------------------
# prediction_tracker pwin-şema migrasyonu
# ---------------------------------------------------------------------------
def test_record_migrates_old_schema_then_appends(tmp_path, monkeypatch):
    log = tmp_path / "predictions_log.csv"
    # Eski (pwin'siz) başlık + 1 satır
    old_fields = [f for f in pt.FIELDS if f != "pwin"]
    with log.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=old_fields)
        w.writeheader()
        w.writerow({"logged_at": "2026-06-01T10:00:00", "symbol": "AAA", "interval": "1h",
                    "bar_ts": "2026-06-01 09:30:00", "direction": "+", "confidence": "0.6",
                    "prob_cal": "0.55", "price": "100", "horizon_bars": "5",
                    "resolved_at": "", "fwd_price": "", "hit": "", "fwd_ret_pct": ""})
    monkeypatch.setattr(pt, "PRED_LOG", log)

    # pwin taşıyan yeni tahmin kaydet → şema yükselir, append kaymaz
    n = pt.record([{"symbol": "BBB", "action": "LONG", "direction": "-", "confidence": 0.4,
                    "prob_cal": 0.3, "pwin": 0.62, "price": 50, "bar_ts": "2026-06-02 09:30:00"}],
                  interval="1h")
    assert n == 1
    with log.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == pt.FIELDS  # pwin sütunu eklendi
    with log.open(encoding="utf-8", newline="") as f:
        rows = {r["symbol"]: r for r in csv.DictReader(f)}
    assert rows["AAA"]["pwin"] == ""        # eski satır boş pwin (kayma yok)
    assert rows["BBB"]["pwin"] == "0.62"    # yeni satır pwin doğru hizada
    assert rows["BBB"]["direction"] == "-"  # sütunlar kaymadı


def test_evaluate_no_crash_on_missing_log(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "PRED_LOG", tmp_path / "nope.csv")
    ev = fh.evaluate()
    assert ev["verdict"]["label"] == "NO_DATA"


def test_gunluk_ogrenme_cozumden_sonra_rapor_ziyaretinden_bagimsiz(tmp_path, monkeypatch, capsys):
    from nihai.ai import learning
    from research import auto_retrain

    pred = tmp_path / "predictions.csv"
    saved = tmp_path / "learning.json"
    monkeypatch.setattr(learning, "PRED_LOG", pred)
    monkeypatch.setattr(learning, "LEARNING_STATE", saved)
    events = []

    def resolve():
        events.append("resolve")
        pred.write_text("resolved_at,hit,direction\n2026-09-21T10:00:00,1,+\n", encoding="utf-8")
        return {}

    def snapshot(ev):
        assert learning.load_state()["strategies"]["LONG"]["n"] == 1
        events.append("snapshot")

    monkeypatch.setattr(pt, "resolve", resolve)
    monkeypatch.setattr(fh, "evaluate", lambda: {})
    monkeypatch.setattr(fh, "snapshot", snapshot)
    monkeypatch.setattr(fh, "publish_side_reports", lambda: events.append("side_reports"))
    monkeypatch.setattr(fh, "_print_report", lambda ev: events.append("report"))
    monkeypatch.setattr(fh, "read_history", lambda: [])
    monkeypatch.setattr(fh, "_print_history", lambda rows: events.append("history"))
    monkeypatch.setattr(auto_retrain, "run", lambda: events.append("retrain") or {"status": "UNCHANGED"})
    monkeypatch.setattr(sys, "argv", ["harness", "--resolve", "--snapshot", "--report"])
    fh.main()
    assert events == ["resolve", "snapshot", "side_reports", "report", "retrain"]
    assert "öğrenme → n=1" in capsys.readouterr().out
    before = saved.read_bytes()

    # Salt-okunur raporlar başarısız bir yazma yoluna bile girmemeli.
    def forbidden():
        events.append("unexpected_learning")
        raise OSError("test failure")

    monkeypatch.setattr(learning, "update_from_predictions", forbidden)
    for flag in ("--report", "--history"):
        monkeypatch.setattr(sys, "argv", ["harness", flag])
        fh.main()
    assert "unexpected_learning" not in events
    assert events.count("retrain") == 1
    assert saved.read_bytes() == before

    # Hesap ve atomik yazma hataları görünür; diğer raporlar yine tamamlanır.
    for updater in (forbidden, lambda: {"saved": False}):
        monkeypatch.setattr(learning, "update_from_predictions", updater)
        monkeypatch.setattr(sys, "argv", ["harness", "--snapshot", "--report"])
        fh.main()
        assert events[-4:] == ["snapshot", "side_reports", "report", "retrain"]
        assert "ÖĞRENME GÜNCELLENEMEDİ" in capsys.readouterr().out
        assert saved.read_bytes() == before
