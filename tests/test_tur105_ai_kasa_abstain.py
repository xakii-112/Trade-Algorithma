"""Tur-105 — gölge `ai_karar` kasasında ABSTAIN artık veto DEĞİL (`nihai/execution/golge_kasalar.py`).

Tur-102 kasası canlı enforce sözleşmesini (ABSTAIN = veto) birebir uyguluyordu; prior'ın lo95'i iki yönde de
0,5'in altında olduğu için 2026-09-28 → 10-01 arasında 5.531 adayın 5.531'i ABSTAIN aldı ve kasa tek işlem
açmadı (BULGU-105-1). Üç soru kilitlenir:
  1. KURAL: ABSTAIN geçer (boyut = posterior rampası), TERS yön hâlâ veto, aynı yön değişmedi.
  2. MOTOR: ABSTAIN adayı kasada T+1 açılışında küçültülmüş adetle açılır; defterin `ai_veto` kolonu canlı
     enforce'un cevabını (1) yazmaya devam eder — kasanın hükmü o kolondan AYRIDIR.
  3. KAPSAM: canlı `ai_advisor.advise` enforce'u DEĞİŞMEDİ (ABSTAIN veto — test_tur38 kilitler); taraf verilince
     motor ters yön üretemez, yani kasada pratikte veto kalmaz (BULGU-105-2).
"""
from __future__ import annotations

import inspect

import pytest

from nihai.ai import decision
from nihai.execution import ai_advisor
from nihai.execution import golge_kasalar as G
from tests.test_tur102_golge_kasalar import _barlar, _kararlar, _kur, _yuru


def _k(**alan):
    return G.karar("ai_karar", G.Olcum(**alan), meta_esik=0.45, noral_esik=0.486)


# --------------------------------------------------------------------------- 1. KURAL
@pytest.mark.parametrize("post", [0.1859, 0.3681, 0.4999])
def test_abstain_gecer_boyut_taban(post):
    gir, carpan, neden = _k(ai_action="ABSTAIN", ai_post=post, ai_veto=1)
    assert gir is True
    assert carpan == pytest.approx(ai_advisor.enforce_boyut(post)) == pytest.approx(0.25)   # rampa tabanı
    assert "ABSTAIN" in neden


def test_abstain_yuksek_posteriorda_rampa_buyutur():
    # canlı defterin en yüksek posterioru 0,553 (5.531 adayın 33'ü ≥ 0,50) → tabanın üstünde, 1,0'ın altında
    gir, carpan, _ = _k(ai_action="ABSTAIN", ai_post=0.553, ai_veto=1)
    assert gir and carpan == pytest.approx(ai_advisor.enforce_boyut(0.553)) == pytest.approx(0.6475)


@pytest.mark.parametrize("ters", ["SHORT_BIAS", "LONG_BIAS"])
def test_ters_yon_hala_veto(ters):
    gir, carpan, neden = _k(ai_action=ters, ai_post=0.62, ai_veto=1)
    assert (gir, carpan) == (False, 0.0)
    assert "ters yön" in neden


def test_ayni_yon_ve_olcumsuz_degismedi():
    assert _k(ai_action="LONG_BIAS", ai_post=0.62, ai_veto=0)[:2] == (True, 1.0)
    assert _k()[:2] == (True, 1.0)


def test_eski_sozlesme_ifadesi_kaynakta_yok():
    kaynak = inspect.getsource(G)
    assert "(ABSTAIN dahil) VETO" not in kaynak
    assert "ters/ABSTAIN veto" not in G.ACIKLAMA["ai_karar"]
    assert "ABSTAIN geçer" in G.ACIKLAMA["ai_karar"]


# --------------------------------------------------------------------------- 2. MOTOR
def test_abstain_adayi_kasada_acilir_defter_canli_vetoyu_yazar(monkeypatch, tmp_path):
    df = _barlar(6)
    m = _kur(monkeypatch, tmp_path, aday_barlari={df.index[2]},
             olcum=G.Olcum(ai_action="ABSTAIN", ai_post=0.39, ai_veto=1))
    _yuru(m, df)
    ai = m.kasalar["ai_karar"].portfoy.open_positions
    kontrol = m.kasalar["kontrol"].portfoy.open_positions
    assert len(ai) == 1 and len(kontrol) == 1
    assert ai[0].entry_time == kontrol[0].entry_time                     # aynı T+1 açılışı
    assert ai[0].quantity == int(kontrol[0].quantity * 0.25)             # çarpan tavandan SONRA (BULGU-102-7)
    satir = _kararlar(tmp_path)[0]
    assert satir["ai_karar"] == "gir×0.25"
    assert satir["ai_veto"] == "1"                                       # canlı enforce'un cevabı korunur


def test_ters_yon_adayi_kasada_acilmaz(monkeypatch, tmp_path):
    df = _barlar(6)
    m = _kur(monkeypatch, tmp_path, aday_barlari={df.index[2]},
             olcum=G.Olcum(ai_action="SHORT_BIAS", ai_post=0.7, ai_veto=1))
    _yuru(m, df)
    assert m.kasalar["ai_karar"].portfoy.open_positions == []
    assert len(m.kasalar["kontrol"].portfoy.open_positions) == 1
    assert _kararlar(tmp_path)[0]["ai_karar"].startswith("veto: ")


# --------------------------------------------------------------------------- 3. KAPSAM
@pytest.mark.parametrize("taraf, isaret", [("LONG", 1), ("SHORT", -1)])
def test_taraf_verilince_motor_ters_yon_uretemez(taraf, isaret):
    # kasa `shadow_verdict(side=taraf)` çağırır → yön side_hint'ten gelir, momentum tersini gösterse bile
    ters_momentum = {"ret_20": -isaret * 5.0, "ema_gap": -isaret * 2.0}
    assert decision._direction_sign(taraf, ters_momentum) == isaret
