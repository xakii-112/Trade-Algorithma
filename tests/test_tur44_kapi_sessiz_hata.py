"""
test_tur44_kapi_sessiz_hata.py — BULGU-44: SESSİZ degrade ≠ ZARİF degrade.

ÖLÇÜLEN KUSUR (Tur-43 sırasında CANLI olarak yaşandı). `_MetaGate.probability`
imzası `shared=` kwarg'ı ile genişleyince eski imzalı çağıranlar TypeError verdi.
İki katman birden yuttu:
  · `probability` kendi gövdesini `except Exception: return None` ile sarıyor,
  · `generate_signal` kapı çağrısını `except Exception: pass` ile sarıyor.
Sonuç: kapı KONFİGÜRE AÇIK ve model YÜKLÜ iken sessizce pasifleşti. Testler
yakaladı; üretim YAKALAMAZDI — çünkü tek dış gözlem `gate.active`'dir ve o hâlâ
True döner ("AKTİF (logistic)" yazar).

Zarar teorik değil: gölge kipin varlık nedeni canlı kanıt döngüsüdür
(prediction_tracker p_win logu → gate_sim). Kapı susarsa o log BOŞ birikir ve
"kapı işe yarıyor mu?" sorusu canlıda ASLA ölçülemez (ACI DERS #2/#5).

Bu dosya sözleşmeyi kilitler: hata YUTULMAYA DEVAM EDER (sinyal bloke edilmez)
ama artık SAYILIR, LOGLANIR, sinyale İZ DÜŞER ve `gate_health()` üzerinden
sorgulanabilir.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from nihai.config import CONFIG
from nihai.signals import meta_gate as mg
from nihai.signals import strategy as st
from nihai.signals.strategy import SignalAction, generate_signal


@pytest.fixture(autouse=True)
def _temiz_defter():
    """Her test kendi defterinden başlasın (defter modül düzeyinde ve kalıcıdır)."""
    mg.reset_health()
    yield
    mg.reset_health()


# seed 7 ARANARAK bulundu: `generate_signal` bu seride LONG üretir (kalite 3,05 · RR 2,0)
# → kapı GERÇEKTEN çağrılır. Rastgele bir seri çoğunlukla WAIT verir ve kapıya hiç
# ulaşmaz; o durumda test "yeşil" görünüp HİÇBİR ŞEY doğrulamaz — sessiz atlama,
# bu dosyanın konusu olan sessiz yutmanın ta kendisidir.
_AKSIYON_SEED = 7


def _df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(_AKSIYON_SEED)
    c = 100 + np.cumsum(rng.normal(0.05, 1, n))
    return pd.DataFrame(
        {"open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
         "volume": rng.integers(1e5, 1e6, n).astype(float)},
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )


def test_on_kosul_bu_seri_kapiya_ULASIR():
    """Diğer testlerin ön koşulu. Strateji eşikleri değişip seri WAIT'e düşerse bu
    test KIRMIZI olur — aşağıdakiler sessizce anlamsızlaşmasın diye."""
    mg.reset_health()
    sig = generate_signal("AAA", _df())
    assert sig.action in (SignalAction.LONG, SignalAction.SHORT),         "test verisi artık aksiyonlu sinyal üretmiyor → kapı çağrılmıyor, _AKSIYON_SEED yenile"


class _EskiImzaliKapi:
    """Tur-43 öncesi imza: `shared=` KABUL ETMEZ → TypeError.

    Bu, uydurma bir senaryo değil; Tur-43'te beş testin kırmızıya dönme sebebidir."""
    threshold = 0.5
    kind = "logistic"
    active = True

    def probability(self, df, sig):        # noqa: ARG002 — dar imza KASITLI
        return 0.9


class _PatlayanKapi:
    threshold = 0.5
    kind = "logistic"
    active = True

    def probability(self, df, sig, **_):   # noqa: ARG002
        raise RuntimeError("model bozuk")


def _kip(monkeypatch, mode: str = "shadow", thr: float = 0.6):
    strat = dataclasses.replace(CONFIG.strategy, meta_gate_mode=mode, meta_gate_threshold=thr)
    monkeypatch.setattr(st, "CONFIG", dataclasses.replace(CONFIG, strategy=strat))


# ---------------------------------------------------------------------------
# 1) ASIL KUSUR: imza kırılması artık SESSİZ değil
# ---------------------------------------------------------------------------
def test_imza_kirilmasi_sinyale_iz_birakir(monkeypatch):
    _kip(monkeypatch)
    monkeypatch.setattr(mg, "_GATE", _EskiImzaliKapi())
    sig = generate_signal("AAA", _df())
    assert "META_GATE_ERROR" in sig.reasons, "imza kırılması SESSİZCE yutuldu"
    assert "TypeError" in sig.metadata.get("meta_gate_error", "")


def test_imza_kirilmasi_sinyali_BLOKE_ETMEZ(monkeypatch):
    """Zarif degrade korunmalı: kapı patlasa da yön/stop/hedef aynen üretilir."""
    _kip(monkeypatch)
    monkeypatch.setattr(mg, "_GATE", None)
    saglam = generate_signal("AAA", _df())
    mg.reset_health()
    monkeypatch.setattr(mg, "_GATE", _EskiImzaliKapi())
    bozuk = generate_signal("AAA", _df())
    assert bozuk.action == saglam.action
    assert bozuk.stop == saglam.stop and bozuk.target == saglam.target
    assert bozuk.quality_score == saglam.quality_score


def test_hata_defterde_sayilir(monkeypatch):
    _kip(monkeypatch)
    monkeypatch.setattr(mg, "_GATE", _EskiImzaliKapi())
    for _ in range(3):
        generate_signal("AAA", _df())
    h = mg.gate_health()
    assert h["failures"] >= 1
    assert h["where"] == "generate_signal"
    assert "TypeError" in (h["last_error"] or "")


def test_ilk_hata_UYARI_olarak_loglanir(monkeypatch, caplog):
    """Sessizliğin panzehiri: ilk hata operatörün göreceği yere yazılmalı."""
    import logging

    from loguru import logger

    kayit: list[str] = []
    sink = logger.add(lambda m: kayit.append(str(m)), level="WARNING")
    try:
        mg.note_error(TypeError("deneme"), "probability")
    finally:
        logger.remove(sink)
    assert any("meta_gate" in k and "TypeError" in k for k in kayit)
    assert logging  # caplog importu loguru ile karışmasın diye no-op


def test_log_hiz_sinirli():
    """Backtest bar başına çağırır — 300 hata 300 satır log ETMEMELİ."""
    from loguru import logger
    kayit: list[str] = []
    sink = logger.add(lambda m: kayit.append(str(m)), level="WARNING")
    try:
        for _ in range(300):
            mg.note_error(ValueError("x"), "probability")
    finally:
        logger.remove(sink)
    # 1. · 100. · 200. · 300. hata → dört satır (300 değil)
    assert len(kayit) == 4, f"ilk + her 100'de bir bekleniyordu, {len(kayit)} satır çıktı"
    assert mg.gate_health()["failures"] == 300     # sayım TAM, yalnız LOG seyrek


# ---------------------------------------------------------------------------
# 2) `active` ≠ "üretiyor" — durum ayrımı
# ---------------------------------------------------------------------------
def test_gate_health_durumlari(monkeypatch):
    monkeypatch.setattr(mg, "_GATE", _PatlayanKapi())
    assert mg.gate_health()["durum"] == "HENUZ_SORULMADI"   # aktif ama hiç sorulmadı

    mg.note_call()
    assert mg.gate_health()["durum"] == "SAGLIKLI"

    mg.note_error(RuntimeError("x"), "probability")
    assert mg.gate_health()["durum"] == "SUSUYOR", "1 soru / 1 hata = fiilen ölü kapı"

    mg.note_call()
    assert mg.gate_health()["durum"] == "KISMI_HATA"


def test_model_yokken_durum_MODEL_YOK_ve_hata_sayilmaz(monkeypatch, tmp_path):
    """Model dosyası yoksa kapı pasiftir — bu HATA DEĞİLDİR, sayılmamalı."""
    monkeypatch.setattr(mg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mg, "_GATE", None)
    g = mg.get_gate()
    assert g.active is False
    assert g.probability(_df(), object()) is None
    h = mg.gate_health()
    assert h["durum"] == "MODEL_YOK"
    assert h["calls"] == 0 and h["failures"] == 0


def test_aktif_kapinin_ic_hatasi_da_sayilir(monkeypatch, tmp_path):
    """`probability`'nin KENDİ except'i de defterden geçmeli (iki katman da sessizdi)."""
    monkeypatch.setattr(mg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mg, "_GATE", None)
    g = mg.get_gate()
    g.kind = "logistic"          # aktif say
    g._log = None                # ...ama matematiği bozuk → TypeError
    assert g.probability(_df(), object()) is None
    h = mg.gate_health()
    assert h["calls"] == 1 and h["failures"] == 1
    assert h["durum"] == "SUSUYOR"
    assert h["where"] == "probability"


# ---------------------------------------------------------------------------
# 3) Durum canlı yüzeye ULAŞIYOR mu
# ---------------------------------------------------------------------------
def test_servis_meta_kapi_sagligini_tasir():
    """Pano `snapshot()["meta"]` okur; oraya girmeyen bir teşhis operatöre ULAŞMAZ."""
    import inspect

    from nihai.execution import live_predict_service as lps
    kaynak = inspect.getsource(lps.PredictionService._run)
    assert '"meta_gate": gate_health()' in kaynak, \
        "kapı sağlığı döngü meta'sına yazılmıyor → pano hâlâ yalnız `meta_model` görüyor"


def test_saglikli_kapida_iz_YOK(monkeypatch):
    """Gürültü yapmamalı: kapı çalışıyorken META_GATE_ERROR görünmemeli."""
    _kip(monkeypatch)

    class _Saglam:
        threshold = 0.5
        kind = "logistic"
        active = True

        def probability(self, df, sig, **_):   # noqa: ARG002
            return 0.9

    monkeypatch.setattr(mg, "_GATE", _Saglam())
    sig = generate_signal("AAA", _df())
    assert "META_GATE_ERROR" not in sig.reasons
    assert "meta_gate_error" not in sig.metadata
