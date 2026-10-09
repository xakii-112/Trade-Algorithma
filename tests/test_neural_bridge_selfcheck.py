"""BULGU B-8 — nöral gölge köprüsünün CANLI çıktı ürettiğini doğrulama.

Denetimde `predictions_log.csv`'nin neural_* kolonları 0 dolu görünüyordu. İki apayrı
açıklama vardı ve logdan ayırt edilemiyordu:
    (a) piyasa kapalıydı, henüz aday sinyal olmadı  → zararsız
    (b) model hiç çıktı vermiyor (köprü ÖLÜ)        → sessiz arıza
Çünkü canlı yoldaki her hata bilinçli olarak yutuluyor (`_neural_shadow_pwin` → None).
`selfcheck()` gerçek canlı özellik vektörlerini modele vererek farkı ölçer; bu testler
onun sözleşmesini + guard davranışını kilitler.

ÖLÇÜLEN (2026-07-26, gerçek live_features_log): 600/600 üretildi, susturma yok →
köprü canlı. Nötr/sentetik girdide None dönmesi guard'ın DOĞRU çalışmasıydı.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nihai.execution import neural_shadow as ns
from nihai.execution.prediction_tracker import _neural_cells

# --- Köprü sözleşmesi: satırdaki '_neural' → CSV hücreleri ----------------------


def test_gecerli_golge_cikarimi_hucrelere_yazilir():
    pw, side, std = _neural_cells({"_neural": {"pwin": 0.0585, "side": "LONG", "std": 0.035}})
    assert side == "LONG"
    assert float(pw) == pytest.approx(0.0585, abs=1e-4)
    assert float(std) == pytest.approx(0.035, abs=1e-4)


@pytest.mark.parametrize("satir", [
    {},                                              # gölge yok
    {"_neural": None},
    {"_neural": {"pwin": 0.5, "std": 0.01}},         # taraf yok
    {"_neural": {"pwin": 0.5, "side": "UZUN", "std": 0.01}},   # geçersiz taraf
])
def test_bozuk_golge_kaydi_engellemez(satir):
    """ANAYASA: gölge katman tahmin satırından daha az önemli — kayıt ASLA engellenmez."""
    assert _neural_cells(satir) == ("", "", "")


# --- selfcheck() sözleşmesi -----------------------------------------------------


def test_selfcheck_log_yoksa_ok_degil(tmp_path, monkeypatch):
    monkeypatch.setattr(ns, "get_shadow", lambda: _SahteGolge(0.5))
    monkeypatch.setattr("nihai.config.DATA_DIR", tmp_path)
    r = ns.selfcheck()
    assert r["ok"] is False
    assert "hata" in r


def test_selfcheck_uretilen_orani_olcer(tmp_path, monkeypatch):
    _feats_log(tmp_path, satir=3)
    monkeypatch.setattr(ns, "get_shadow", lambda: _SahteGolge(0.42))
    monkeypatch.setattr("nihai.config.DATA_DIR", tmp_path)
    r = ns.selfcheck()
    assert r["ok"] is True
    assert r["denenen"] == 6 and r["uretildi"] == 6      # 3 satır × LONG/SHORT
    assert r["oran"] == 1.0
    assert r["pwin_ort"] == pytest.approx(0.42, abs=1e-3)


def test_selfcheck_sessiz_kopruyu_yakalar(tmp_path, monkeypatch):
    """ASIL AMAÇ: model her girdide susuyorsa bunu 'ok: False' ile SÖYLEMELİ —
    sessiz arıza bir daha kolonların boşluğunda saklanmasın."""
    _feats_log(tmp_path, satir=3)
    monkeypatch.setattr(ns, "get_shadow", lambda: _SahteGolge(None))
    monkeypatch.setattr("nihai.config.DATA_DIR", tmp_path)
    r = ns.selfcheck()
    assert r["ok"] is False
    assert r["uretildi"] == 0 and r["susturuldu"] == 6
    assert r["oran"] == 0.0


# --- yardımcılar ----------------------------------------------------------------


class _SahteGolge:
    """pwin(): sabit sonuç veya None (doygunluk guard'ı taklidi)."""

    card_hash = "test1234"

    def __init__(self, p: float | None):
        self._p = p

    def pwin(self, feats: dict, side: str):   # noqa: ARG002
        return None if self._p is None else (self._p, 0.03)


def _feats_log(kok, satir: int) -> None:
    from nihai.signals.meta_features import FEATURE_NAMES
    kok.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{k: 0.1 for k in FEATURE_NAMES} for _ in range(satir)])
    df.insert(0, "symbol", "AAPL")
    df.to_csv(kok / "live_features_log.csv", index=False)
