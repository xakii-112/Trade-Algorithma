"""Tur-77 · Madde 6.4b — KARAR HUNİSİ: yazıcı ↔ ayrıştırıcı gidiş-dönüşü.

Tur-29 dersi: "Günlük Defteri" sekmesi 330 kapanışta boştu çünkü test fixture'ı gerçek yazıcının
biçimi değildi. Burada log satırları ELLE yazılmaz; `log_block`/`log_signal`/`log_decision`
gerçekten çağrılır ve huni onların ürettiği dosyayı okur.
"""
from __future__ import annotations

import pytest

from nihai.signals import signal_engine as se
from nihai.signals.strategy import Signal, SignalAction


@pytest.fixture(autouse=True)
def _gecici_log(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "LOG_DIR", tmp_path)
    monkeypatch.setenv("AUDIT_CHAIN", "off")
    return tmp_path


def _sig(sym: str, action: SignalAction) -> Signal:
    return Signal(action=action, symbol=sym, price=100.0, stop=98.0, target=104.0,
                  reasons=["kirilim"], quality_score=0.7, rr_ratio=2.0)


def _gun_yaz() -> None:
    for _ in range(3):
        se.log_block("SESSION_GUARD", "AAA: MARKET_CLOSED")
    se.log_block("COOLDOWN", "AAA: re-entry bekleniyor")
    se.log_block("EVENT_GUARD", "BBB: EARNINGS_IN_2_DAYS_2026-09-02")
    se.log_decision("CHAIN_HEALTH", "BBB: kabul=0 red=15")                     # bilgi satırı
    se.log_signal(_sig("CCC", SignalAction.WAIT), accepted=False)                  # strateji WAIT
    se.log_signal(_sig("DDD", SignalAction.LONG), accepted=False, extra="STALE_SIGNAL")
    se.log_block("STALE_SIGNAL", "DDD: STALE_RR: canlı fiyatta RR=0.46 < 1.20")   # ÇİFT yazılan dal
    se.log_decision("SHADOW", "DDD: SHADOW[advise] health=90")                    # bilgi satırı
    se.log_signal(_sig("EEE", SignalAction.SHORT), accepted=True, extra="SIGNAL_ALERT (risk-bağımsız)")
    se.log_signal(_sig("EEE", SignalAction.SHORT), accepted=False,
                  extra="RISK_REJECT: MAX_OPEN_POSITIONS")
    se.log_signal(_sig("FFF", SignalAction.LONG), accepted=True, extra="SIGNAL_ALERT (risk-bağımsız)")
    se.log_signal(_sig("FFF", SignalAction.LONG), accepted=True, extra="sizing={'qty': 10}")
    se.log_decision("POSITION_OPEN", "FFF LONG 10x @ 100.0")
    se.log_block("CIRCUIT_BREAKER", "TRIPPED: MANUEL")                            # sembolsüz satır
    se.log_decision("YENI_KOD", "GGG: bilinmeyen aşama")


def test_gidis_donus_asama_sayimlari():
    _gun_yaz()
    h = se.karar_hunisi()
    assert h is not None
    el = {a["asama"]: a["elenen"] for a in h["asamalar"]}
    assert el["SESSION_GUARD"] == 3 and el["COOLDOWN"] == 1 and el["EVENT_GUARD"] == 1
    assert el["STRATEJI"] == 1                       # yalnız CCC WAIT; DDD'nin SKIP satırı sayılmadı
    assert el["STALE_SIGNAL"] == 1 and el["RISK_REJECT"] == 1 and el["CIRCUIT_BREAKER"] == 1
    assert h["kabul"] == 1 and h["uyari"] == 2 and h["pozisyon_acilis"] == 1
    assert h["toplam_deneme"] == sum(el.values()) + h["kabul"] == 10
    assert h["taninmayan"] == {"YENI_KOD": 1}
    alt = {a["asama"]: a["alt_neden"] for a in h["asamalar"]}
    assert alt["EVENT_GUARD"] == {"EARNINGS": 1}
    assert alt["STALE_SIGNAL"] == {"STALE_RR": 1}
    assert alt["RISK_REJECT"] == {"MAX_OPEN_POSITIONS": 1}
    assert alt["STRATEJI"] == {"WAIT": 1}


def test_kalan_monoton_ve_sonda_kabule_esit():
    _gun_yaz()
    h = se.karar_hunisi()
    kalanlar = [a["kalan"] for a in h["asamalar"]]
    assert all(x >= y for x, y in zip(kalanlar, kalanlar[1:], strict=False))
    assert kalanlar[-1] == h["kabul"]


def test_sembol_en_ileri_asama():
    _gun_yaz()
    h = se.karar_hunisi()
    # AAA: SESSION_GUARD'da 3 kez, COOLDOWN'da 1 kez → en ileri COOLDOWN. FFF → KABUL.
    assert h["sembol_en_ileri"]["COOLDOWN"] == 1
    assert h["sembol_en_ileri"]["KABUL"] == 1
    assert "TRIPPED" not in str(h["sembol_en_ileri"]) and h["n_sembol"] == 6   # AAA..FFF; kesici sembolsüz


def test_log_yoksa_none(tmp_path):
    assert se.karar_hunisi(tmp_path / "decision_log_1999_01_01.txt") is None


def test_uc_tarih_dogrulamasi_ve_bugun_dosyasi(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from nihai.web import api_operations as ops
    monkeypatch.setattr(ops, "LOG_DIR", tmp_path)
    _gun_yaz()
    r = ops.decision_funnel()
    assert r["ok"] is True and r["kabul"] == 1 and r["dosya"] == se._today_log().name
    with pytest.raises(HTTPException):
        ops.decision_funnel(tarih="../../etc")
    assert ops.decision_funnel(tarih="1999-01-01")["reason"] == "LOG_YOK"


def test_engellenen_sinyaller_yazicinin_dosya_adini_kullanir():
    """Önceki okuyucu ET tarihiyle dosya arıyordu; yazıcı yerel tarihle yazar."""
    from nihai.web import api_operations as ops
    se.log_block("HALT_LULD", "ZZZ: LULD_PAUSE")
    r = ops.blocked_signals()
    assert r["file"] == se._today_log().name and r["counts"].get("HALT_LULD") == 1
