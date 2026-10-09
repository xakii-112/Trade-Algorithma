"""Tur-77 · Madde 5.4 — K-NN tarihsel anı özeti (`episodic.analog_ozeti`) + `/api/ai/episodes/similar`.

Üç soru:
  1. **Doğru komşuları mı buluyor?** Kosinüs yakınlığı bilinen kurguda en yakın çözülmüş epizotu
     ilk sıraya koymalı; çözülmemiş epizot komşu listesine GİRMEMELİ ("nasıl çözüldü" sorusunun
     cevabı olmayan anı gösterilmez).
  2. **Taban var mı?** Komşu ortalaması havuzun koşulsuz ortalamasıyla birlikte döner (ACI DERS #9);
     az örnekte `yeterli=False`.
  3. **Bağlandı mı?** Uç sembolün son epizodunu sorgu yapar, kendisini ve (varsayılan) aynı sembolü dışlar.
"""
from __future__ import annotations

import pytest

from nihai.ai import episodic


@pytest.fixture(autouse=True)
def _tmp_episodes(tmp_path, monkeypatch):
    monkeypatch.setattr(episodic, "EPISODES_FILE", tmp_path / "ai_episodes.jsonl")
    return tmp_path


def _ep(symbol: str, x: float, y: float, z: float, fwd: float | None):
    e = episodic.append_episode("ai_decision", symbol=symbol,
                                features={"x": x, "y": y, "z": z, "w": 1.0},
                                summary="event_breakout: ABSTAIN", tags=["event_breakout", "abstain"])
    if fwd is not None:
        episodic.resolve_episode(e.id, {"win": None, "fwd_ret_pct": fwd, "horizon_bars": 5, "side": 0})
    return e


def test_en_yakin_cozulmus_komsu_ilk_sirada_cozulmemis_listede_yok():
    yakin = _ep("AAA", 1.0, 2.0, 3.0, fwd=+1.5)
    _ep("BBB", 1.0, 2.0, 3.05, fwd=None)                 # en yakın AMA çözülmemiş
    _ep("CCC", -3.0, 5.0, -1.0, fwd=-0.5)
    _ep("DDD", 9.0, -4.0, 2.0, fwd=+0.2)
    r = episodic.analog_ozeti({"x": 1.0, "y": 2.0, "z": 3.0, "w": 1.0}, k=2, min_taban=1)
    ids = [k["id"] for k in r["komsular"]]
    assert ids[0] == yakin.id
    assert all(k["symbol"] != "BBB" for k in r["komsular"])
    assert r["n_taban"] == 3
    assert r["taban_fwd_ret_ort"] == pytest.approx((1.5 - 0.5 + 0.2) / 3, abs=1e-4)
    assert r["komsular"][0]["benzerlik"] <= 1.0


def test_haric_id_ve_haric_sembol():
    a = _ep("AAA", 1.0, 2.0, 3.0, fwd=+1.0)
    _ep("AAA", 1.0, 2.0, 3.1, fwd=+2.0)
    b = _ep("BBB", 1.2, 2.1, 2.9, fwd=-1.0)
    r = episodic.analog_ozeti({"x": 1.0, "y": 2.0, "z": 3.0, "w": 1.0}, k=5,
                              haric_id=a.id, haric_sembol="AAA", min_taban=1)
    assert [k["id"] for k in r["komsular"]] == [b.id]
    assert r["n_taban"] == 1


def test_az_ornekte_yeterli_degil_ve_bos_havuz_nedeni():
    _ep("AAA", 1.0, 2.0, 3.0, fwd=+1.0)
    r = episodic.analog_ozeti({"x": 1.0, "y": 2.0, "z": 3.0, "w": 1.0}, k=5)
    assert r["yeterli"] is False and r["reason"].startswith("AZ_ORNEK")
    bos = episodic.analog_ozeti({"x": 1.0, "y": 2.0, "z": 3.0, "w": 1.0}, k=5, haric_sembol="AAA")
    assert bos["komsular"] == [] and bos["reason"] == "COZULMUS_KOMSU_YOK"
    assert bos["taban_fwd_ret_ort"] is None                          # taban uydurulmaz


def test_sonlu_olmayan_ileri_getiri_havuza_girmez():
    _ep("AAA", 1.0, 2.0, 3.0, fwd=float("nan"))
    _ep("BBB", 1.0, 2.0, 3.0, fwd=+0.4)
    r = episodic.analog_ozeti({"x": 1.0, "y": 2.0, "z": 3.0, "w": 1.0}, k=5, min_taban=1)
    assert [k["symbol"] for k in r["komsular"]] == ["BBB"]


def test_ayni_karar_bari_kopyalari_tek_ani_sayilir():
    """Tur-77 ölçümü: gerçek defterde 1.000 çözümde 210 benzersiz (sembol, karar barı) vardı ve
    ilk 5 komşunun beşi aynı karardı. Okuma tarafı kopyayı TEK anı sayar, tabanı da tekilleştirir."""
    for _ in range(4):                                         # aynı IBM kararı 4 kez yazılmış
        e = episodic.append_episode("ai_decision", symbol="IBM",
                                    features={"x": 1.0, "y": 2.0, "z": 3.0, "w": 1.0})
        episodic.resolve_episode(e.id, {"win": None, "fwd_ret_pct": -0.04, "horizon_bars": 5,
                                        "decision_bar": "2026-08-21T00:00:00+00:00"})
    tek = _ep("V", 1.1, 2.0, 3.0, fwd=+2.8)
    r = episodic.analog_ozeti({"x": 1.0, "y": 2.0, "z": 3.0, "w": 1.0}, k=5, min_taban=1)
    assert [k["symbol"] for k in r["komsular"]].count("IBM") == 1
    assert r["n_taban"] == 2 and r["n_kopya_elenen"] == 3
    assert tek.id in {k["id"] for k in r["komsular"]}
    assert r["taban_fwd_ret_ort"] == pytest.approx((-0.04 + 2.8) / 2, abs=1e-4)


def test_yazma_tarafi_ayni_gun_kopyasini_yazmaz(monkeypatch):
    kw = dict(symbol="NVDA", features={"x": 1.0, "y": 2.0}, summary="s: ABSTAIN", tags=["s", "abstain"])
    a = episodic.append_episode("ai_decision", **kw, ayni_gun_tekil=True)
    b = episodic.append_episode("ai_decision", **kw, ayni_gun_tekil=True)
    assert b.id == a.id                                           # yeni satır YOK, var olan döner
    episodic.append_episode("ai_decision", **{**kw, "features": {"x": 1.0, "y": 2.5}},
                            ayni_gun_tekil=True)                  # vektör değişti → yeni anı
    monkeypatch.setattr(episodic, "_utc_now_iso", lambda: "2099-01-02T00:00:00+00:00")
    episodic.append_episode("ai_decision", **kw, ayni_gun_tekil=True)   # ertesi gün → yeni anı
    satirlar = episodic.EPISODES_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(satirlar) == 3
    episodic.append_episode("ai_decision", **kw)                  # varsayılan: eski davranış
    assert len(episodic.EPISODES_FILE.read_text(encoding="utf-8").strip().splitlines()) == 4


def test_golge_karar_kaydi_tekilleştirmeyi_acar(monkeypatch):
    from types import SimpleNamespace

    from nihai.ai import decision
    yakalanan: dict = {}
    monkeypatch.setattr(decision, "_append_decision_file", lambda d: None)
    monkeypatch.setattr(decision.episodic, "append_episode",
                        lambda **kw: yakalanan.update(kw))
    decision._record_shadow(SimpleNamespace(symbol="AAA", strategy="s", action="ABSTAIN",
                                            conviction=0.0), {"x": 1.0})
    assert yakalanan.get("ayni_gun_tekil") is True


def test_uc_son_epizodu_sorgular_ve_ayni_sembolu_disarida_birakir():
    from nihai.web import api_extra
    _ep("AAA", 1.0, 2.0, 3.0, fwd=+1.0)
    _ep("BBB", 1.1, 2.0, 3.0, fwd=+0.5)
    son = _ep("AAA", 1.0, 2.0, 3.0, fwd=None)            # sorgu: AAA'nın en son epizodu
    r = api_extra.ai_episodes_similar(symbol="aaa", k=5)
    assert r["ok"] is True and r["sorgu"]["id"] == son.id
    assert [k["symbol"] for k in r["komsular"]] == ["BBB"]
    r2 = api_extra.ai_episodes_similar(symbol="AAA", k=5, ayni_sembol=True)
    assert {k["symbol"] for k in r2["komsular"]} == {"AAA", "BBB"}
    assert api_extra.ai_episodes_similar(symbol="ZZZ")["reason"] == "EPIZOT_YOK"
