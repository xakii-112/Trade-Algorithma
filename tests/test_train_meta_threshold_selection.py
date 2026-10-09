"""KALEM 1 — eşik SEÇİMİ artık TRAIN'de yapılır, TEST yalnız DOĞRULAR.

Bağlam: eskiden evaluate() dört adayı (0.45/0.50/0.55/0.60) doğrudan TEST pnl'i üzerinde
tarayıp en iyisini seçiyordu — bu, test setini KULLANARAK seçim yapmak demekti (test-sızıntısı
+ çoklu-test yanlılığı: 4 şansın en iyisi rapor edilir). Düzeltme: adaylar TRAIN'de (kept_tr≥5,
e_tr>base_e_tr, pf_tr>1.0) taranır, en yüksek TRAIN-expectancy'li eşik SEÇİLİR; o eşik SONRA
TEST'te yalnızca DOĞRULANIR (e_te>base_e_te ve pf_te>1.0 değilse best_gate=None).

evaluate()'in model.predict_proba çağrılarını (önce Xtr, sonra Xte — kodda bu sırayla) sahte bir
modelle deterministik kontrol ederek train/test olasılıklarını tam olarak kurguluyoruz; böylece
gerçek bir model fit etmeden seçim mantığının SADECE train'e baktığını kanıtlıyoruz."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.signals.meta_features import FEATURE_NAMES
from research.train_meta import evaluate


class _FakeModel:
    """evaluate() önce predict_proba(Xtr) sonra predict_proba(Xte) çağırır (kod sırası).
    Çağrı SIRASINA göre önceden hazırlanmış olasılık dizilerini döndürür — model X'in
    içeriğine bakmaz, yalnız çağrı sırasına göre train/test olasılıklarını ayırt eder."""

    kind = "fake"

    def __init__(self, tr_probs: np.ndarray, te_probs: np.ndarray):
        self._queue = [tr_probs, te_probs]

    def predict_proba(self, X):
        return self._queue.pop(0)


def _mk_df(n: int) -> pd.DataFrame:
    """FEATURE_NAMES + label + pnl kolonlu, içeriği önemsiz bir df (probaları _FakeModel
    kurguluyor; evaluate yalnız pnl + FEATURE_NAMES'i okur, X'in sayısal değeri kararı etkilemez)."""
    return pd.DataFrame(
        {**{f: np.zeros(n) for f in FEATURE_NAMES},
         "label": np.zeros(n), "pnl": np.zeros(n)})


# --- TRAIN verisi: 20 satır, 3 olasılık kümesi (0.62 / 0.52 / 0.20).
# Yalnız thr=0.55 TRAIN'de en yüksek expectancy'yi verir (bkz modül docstring'indeki hesap):
#   thr .45/.50 → kept=10 satır (0.62'ler + 0.52'ler), e_tr=27.5
#   thr .55/.60 → kept=5 satır  (yalnız 0.62'ler),      e_tr=50.0  ← en yüksek, 0.55 önce görülür
TR_PROBS = np.array([0.62] * 5 + [0.52] * 5 + [0.20] * 10)
TR_PNL = np.array([50.0] * 5 + [5.0] * 5 + [-10.0] * 10)


def _train_df():
    df = _mk_df(20)
    df["pnl"] = TR_PNL
    return df


def test_selected_threshold_fails_on_test_yields_no_gate():
    """TRAIN'de thr=0.55 güçlü bir 'kenar' gösterir (e_tr=50 >> base_e_tr=8.75) AMA aynı eşikte
    TEST'te tutulan işlemler ZARARLI çıkarsa (model train'de ezberlemiş/overfit) best_gate None
    OLMALI — TRAIN'deki görünüşteki kenar TEK BAŞINA yeterli değildir, test DOĞRULAMALIDIR."""
    tr = _train_df()
    # TEST: pte=0.62 olan 5 satır ZARARLI (-20), pte=0.10 olan 15 satır KÂRLI (+5).
    # thr=0.55 (TRAIN'de seçilen) → kept_te = 5 satır (pte=0.62), hepsi zararlı → pf_te=0, e_te=-20.
    te_probs = np.array([0.62] * 5 + [0.10] * 15)
    te_pnl = np.array([-20.0] * 5 + [5.0] * 15)
    te = _mk_df(20)
    te["pnl"] = te_pnl

    model = _FakeModel(TR_PROBS, te_probs)
    r = evaluate(model, tr, te, label="test-fail")

    assert r["thr_source"] == "train"
    assert r["n_thr_tried"] == 4
    assert r["best_gate"] is None    # TRAIN'de seçilen eşik TEST'te doğrulanamadı


def test_selected_threshold_holds_on_test_yields_gate():
    """AYNI TRAIN verisi (aynı sel_thr=0.55) ama bu kez TEST'te de o eşik KÂRLI çıkarsa
    best_gate dolu döner ve eşiği 0.55'tir (TRAIN'de seçilenle AYNI — bkz alttaki bağımsızlık testi)."""
    tr = _train_df()
    te_probs = np.array([0.62] * 5 + [0.10] * 15)
    te_pnl = np.array([30.0] * 5 + [-1.0] * 15)   # bu kez pte>=0.55 kümesi KÂRLI
    te = _mk_df(20)
    te["pnl"] = te_pnl

    model = _FakeModel(TR_PROBS, te_probs)
    r = evaluate(model, tr, te, label="test-pass")

    assert r["best_gate"] is not None
    assert r["best_gate"][0] == pytest.approx(0.55)


def test_threshold_selection_independent_of_test_content(capsys):
    """AYNI TRAIN verisi, İKİ FARKLI (birbirine zıt) TEST senaryosu: seçilen eşik (tabloda
    '← seçilen (train)' işaretiyle basılan) HER İKİSİNDE DE 0.55 olmalı — seçim test içeriğine
    bakmadan yalnız TRAIN'den yapıldığı için test verisi ne olursa olsun değişmemeli."""
    tr = _train_df()

    def _selected_threshold_line(te_probs, te_pnl):
        te = _mk_df(20)
        te["pnl"] = te_pnl
        model = _FakeModel(TR_PROBS.copy(), te_probs)
        evaluate(model, tr, te, label="probe")
        out = capsys.readouterr().out
        lines = [ln for ln in out.splitlines() if "seçilen (train)" in ln]
        assert len(lines) == 1, f"tam bir satır işaretlenmeli, bulunan: {lines}"
        return lines[0]

    # Senaryo A: test'te thr=0.55 kümesi zararlı (best_gate None olur)
    line_a = _selected_threshold_line(
        np.array([0.62] * 5 + [0.10] * 15), np.array([-20.0] * 5 + [5.0] * 15))
    # Senaryo B: test'te thr=0.55 kümesi kârlı (best_gate dolu olur)
    line_b = _selected_threshold_line(
        np.array([0.62] * 5 + [0.10] * 15), np.array([30.0] * 5 + [-1.0] * 15))

    assert line_a.strip().startswith("0.55")
    assert line_b.strip().startswith("0.55")


def test_no_threshold_qualifies_on_train_yields_no_gate():
    """TRAIN'de HİÇBİR eşik (kept_tr≥5, e_tr>base_e_tr, pf_tr>1.0) şartını geçmezse
    sel_thr=None → best_gate MUTLAKA None (test'e hiç bakılmadan)."""
    n = 20
    tr = _mk_df(n)
    tr["pnl"] = -10.0    # her satır zararlı → hiçbir eşikte e_tr>base_e_tr olamaz
    tr_probs = np.array([0.70] * n)   # herkes her eşiği geçer ama hepsi zararlı
    te = _mk_df(n)
    te["pnl"] = 100.0    # test kârlı olsa BİLE seçim yapılmadığından önemsiz
    te_probs = np.array([0.70] * n)

    model = _FakeModel(tr_probs, te_probs)
    r = evaluate(model, tr, te, label="no-train-edge")
    assert r["best_gate"] is None


def test_ece_and_auc_reported():
    """evaluate() dönüşünde ECE test alanı var ve [0,1] aralığında (kalibrasyon tanısı)."""
    tr = _train_df()
    te_probs = np.array([0.62] * 5 + [0.10] * 15)
    te_pnl = np.array([30.0] * 5 + [-1.0] * 15)
    te = _mk_df(20)
    te["pnl"] = te_pnl
    model = _FakeModel(TR_PROBS.copy(), te_probs)
    r = evaluate(model, tr, te, label="ece-check")
    assert "ece_test" in r
    assert 0.0 <= r["ece_test"] <= 1.0
