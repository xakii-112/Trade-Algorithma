"""Gölge yüzey TERFİ ölçeri — Tur-60 kilidi (`research/shadow_promote.py`).

Üç soruyu birden sorar:
  1. KAPI ELİYOR MU? — ters kalibrasyon, kayma artefaktı, yetersiz örneklem ve
     küme-içi sahte anlamlılık terfi ETTİRİLMEMELİ.
  2. KAPI GEÇİRİYOR MU? — yalnız reddeden bir kapı, kapı değil bir `return False`tir;
     gerçekten ayıran bir yüzey PROMOTE almalı (pozitif kontrol).
  3. BAĞLANDI MI? — ölçüm günlük göreve yazılıyor ve özet onu OKUYOR mu (bu depoda
     yazılıp okunmayan rapor kusurdur, Tur-38).
"""
from __future__ import annotations

import json
import random

import pytest

from research import shadow_promote as sp


def _row(*, sym="AAPL", day="2026-08-01", direction="+", ret=0.0, price=100.0,
         bar_close=None, **extra) -> dict:
    """Tek çözülmüş tahmin satırı. `bar_close=None` → kayma YOK (price == bar_close).

    `fwd_price` getiriden TÜRETİLİR ki `_dir_return` ile `_bar_return` aynı ileri fiyatı
    görsün — kayma kontrolünün anlamlı olması buna bağlıdır."""
    bc = price if bar_close is None else bar_close
    sgn = 1.0 if direction == "+" else -1.0
    fwd = price * (1.0 + sgn * ret / 100.0)
    r = {"symbol": sym, "logged_at": f"{day}T12:00:00", "direction": direction,
         "hit": "1" if ret > 0 else "0", "fwd_ret_pct": f"{ret:.6f}",
         "price": f"{price:.6f}", "bar_close": f"{bc:.6f}", "fwd_price": f"{fwd:.6f}"}
    r.update({k: ("" if v is None else str(v)) for k, v in extra.items()})
    return r


def _ayrisan(n_gun: int, n_sym: int, hi: float, lo: float, seed: int = 7) -> list[dict]:
    """GERÇEKTEN ayıran ama GÜRÜLTÜLÜ yüzey: yüksek sağlık → iyi getiri + gürültü.

    Gürültü şart: tam ayrışmada artıklar SIFIR olur, kümele-dayanıklı SE de sıfır çıkar
    ve `_cluster_t` dürüstçe 'hüküm yok' der (dejenere kurgu bir kanıt değildir)."""
    rnd = random.Random(seed)
    rows = []
    for g in _gunler(n_gun):
        for k in range(n_sym):
            rows.append(_row(sym=f"S{k}", day=g, ret=hi + rnd.gauss(0, 0.35), mkt_health=95))
            rows.append(_row(sym=f"S{k}", day=g, ret=lo + rnd.gauss(0, 0.35), mkt_health=10))
    return rows


def _gunler(n: int, start: int = 1) -> list[str]:
    """n ayrı takvim günü — kümele-dayanıklı t'nin küme sayısını doyurması için."""
    return [f"2026-08-{((start + i - 1) % 28) + 1:02d}" for i in range(n)]


def _yuzey(rep: dict, key: str) -> dict:
    return rep["surfaces"][key]


# --------------------------------------------------------------- 1) kapı ELİYOR MU
def test_yetersiz_ornekleme_hukum_verilmez():
    """2*min_side altındaki satırda etiket INSUFFICIENT — kenar uydurulmaz."""
    rows = [_row(sym=f"S{i}", day=g, ret=1.0, mkt_health=90)
            for i, g in enumerate(_gunler(20))]
    res = _yuzey(sp.run(rows, min_side=30), "mkt_health")
    assert res["label"] == "INSUFFICIENT"
    assert "hüküm YOK" in res["text"]


def test_ters_kalibrasyon_terfi_etmez():
    """p_win'de 'DÜŞÜK tut' kazanırsa etiket INVERTED — tersine çevirme YASAK."""
    rows = []
    for g in _gunler(24):
        for k in range(6):
            # düşük p_win → KAZANIYOR, yüksek p_win → kaybediyor (model ters sıralıyor)
            rows.append(_row(sym=f"S{k}", day=g, ret=+1.0, pwin=0.20, pwin_side="LONG"))
            rows.append(_row(sym=f"S{k}", day=g, ret=-1.0, pwin=0.70, pwin_side="LONG"))
    res = _yuzey(sp.run(rows, min_side=30), "meta_pwin")
    assert res["label"] == "INVERTED"
    assert res["best"]["keep"] == "low"
    assert "ACI DERS" in res["text"]


def test_kayma_artefakti_terfi_etmez():
    """Uplift ortak referansta işaret değiştirirse etiket SLIP_ARTIFACT.

    Kurgu: ileri fiyat HER satırda bar kapanışının aynı oranda üstünde (yani gerçek
    yön becerisi SABİT), ama 'uyuşmaz' satırlarda giriş baskısı kovalanmış → satırın
    KENDİ referansından ölçülen getiri düşük. Kural bilgi taşımıyor, yalnız böleni
    ayırıyor; kapı bunu yakalamalı."""
    rows = []
    for g in _gunler(24):
        for k in range(6):
            # uyuşan: giriş = bar kapanışı → kayma yok
            rows.append(_row(sym=f"S{k}", day=g, ret=+0.20, price=100.0, bar_close=100.0,
                             dir_bar="+", conf_bar=0.5))
        # uyuşmaz: giriş bar kapanışının %0,5 üstünde (kovalanmış); AYNI ileri fiyat
        r = _row(sym="S9", day=g, ret=-0.30, price=100.5, bar_close=100.0,
                 dir_bar="-", conf_bar=0.5)
        # ileri fiyat: bar kapanışından +%0,20 (uyuşanlarla AYNI yön becerisi)
        r["fwd_price"] = f"{100.0 * 1.002:.6f}"
        r["fwd_ret_pct"] = f"{(float(r['fwd_price']) / 100.5 - 1.0) * 100.0:.6f}"
        rows.append(r)
    res = _yuzey(sp.run(rows, min_side=20), "dir_bar_agree")
    assert res["label"] == "SLIP_ARTIFACT", res
    assert res["best"]["uplift_pp"] > 0            # satırın kendi referansında POZİTİF
    assert res["slip_control"]["uplift_pp"] <= 0   # ortak referansta yok oluyor


def test_kume_ici_fark_sahte_anlamlilik_uretmez():
    """Tüm 'iyi' satırlar TEK sembol-gününde toplanırsa t kümele-dayanıklı kalmalı.

    Naif t bu kurguda devasa çıkar (n=120); bağımsız gözlem sayısı ise 2'dir."""
    rows = [_row(sym="AAA", day="2026-08-01", ret=+1.0, mkt_health=90) for _ in range(60)]
    rows += [_row(sym="BBB", day="2026-08-02", ret=-1.0, mkt_health=10) for _ in range(60)]
    res = _yuzey(sp.run(rows, min_side=30), "mkt_health")
    assert res["label"] != "PROMOTE"
    assert res["n_clusters"] == 2


# ------------------------------------------------------------- 2) kapı GEÇİRİYOR MU
def test_gercek_ayirim_terfi_eder():
    """Pozitif kontrol: kayma YOK, etki çok kümeye yayılı → PROMOTE + OOS onayı.

    Bu test olmadan kapı 'her şeyi reddet' ile de yeşil kalırdı."""
    res = _yuzey(sp.run(_ayrisan(26, 8, +0.50, -0.50), min_side=30), "mkt_health")
    assert res["label"] == "PROMOTE", res["text"]
    assert res["best"]["keep"] == "high"
    assert res["oos"]["label"] == "CONFIRMED"


def test_oos_bolmesi_yuzeyin_KENDI_satirlarinda_yapilir():
    """Kolon yalnız SON günlerde doluysa IS yarısı boş kalmamalı.

    Gölge kolonlar farklı turlarda eklendi; bölme tüm havuzda yapılırsa o kolonun
    tüm satırları OOS'a düşer ve terfi 'NO_RULE' ile sessizce imkânsızlaşır."""
    rows = [_row(sym=f"S{i % 8}", day=g, ret=0.01) for i, g in enumerate(_gunler(20))]
    rows += _ayrisan(26, 8, +0.50, -0.50)
    res = _yuzey(sp.run(rows, min_side=30), "mkt_health")
    assert res["oos"]["label"] == "CONFIRMED"
    assert res["oos"]["n"] > 0


def test_kapsama_gunu_raporlanir():
    """Terfiyi bugün engelleyen şey çoğu yüzeyde ÖRNEKLEM; kaç GÜN koştuğu görünmeli."""
    rows = [_row(sym=f"S{i % 4}", day=g, ret=0.1, mkt_health=50)
            for i, g in enumerate(_gunler(5)) for _ in range(20)]
    res = _yuzey(sp.run(rows, min_side=30), "mkt_health")
    assert res["coverage_days"] == 5


# --------------------------------------------------------------- 3) BAĞLANDI MI
def test_harness_shadow_promote_json_yazar(tmp_path, monkeypatch):
    """Günlük görev terfi ölçümünü diske yazmalı (özet bu dosyadan okur)."""
    import research.forward_harness as fh

    yol = tmp_path / "shadow_promote.json"
    monkeypatch.setattr(fh, "SHADOW_PROMOTE_JSON", yol)
    out = fh.publish_side_reports(_ayrisan(26, 8, +0.50, -0.50))
    assert out["shadow_promote"] is True
    veri = json.loads(yol.read_text(encoding="utf-8"))
    assert veri["source"] == "research.forward_harness"
    assert "mkt_health" in veri["surfaces"]
    # sweep DÜŞÜRÜLÜR (özet yalnız etiketi okur; ayrıntı CLI'da)
    assert "sweep" not in veri["surfaces"]["mkt_health"]


@pytest.mark.parametrize("promote,beklenen", [([], "YOK (hepsi gölgede)"), (["mkt_health"], "Veri sağlığı")])
def test_ozet_terfi_satirini_basar(promote, beklenen):
    """`daily_summary` etiketi ve terfi listesini OKUMALI — boşsa da AÇIKÇA söylemeli."""
    from nihai.notifier import daily_summary as ds

    data = {
        "today": "2026-08-31", "history_last": None, "history_prev": None,
        "gate_sim": None, "coverage": None, "services": {},
        "live_features_today": None,
        "shadow_promote": {
            "generated_at": "2026-08-31T12:00:00", "promote": promote,
            "surfaces": {"mkt_health": {"name": "Veri sağlığı skoru", "label": "PROMOTE"},
                         "meta_pwin": {"name": "Meta-kapı p_win", "label": "INVERTED"}},
        },
    }
    metin = ds.build_summary(data)
    assert "Gölge terfi (2 yüzey)" in metin
    assert "INVERTED 1" in metin
    assert beklenen in metin


def test_ozet_veri_yoksa_sessiz_kalmaz():
    from nihai.notifier import daily_summary as ds

    metin = ds.build_summary({"today": "2026-08-31", "history_last": None,
                              "history_prev": None, "gate_sim": None, "coverage": None,
                              "services": {}, "live_features_today": None,
                              "shadow_promote": None})
    assert "Gölge terfi: veri yok" in metin
