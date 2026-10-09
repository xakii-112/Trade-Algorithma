"""nihai.ai.reasoner testleri — facts toplama, delta, uyarılar, DÜRÜSTLÜK GUARDI.

Ağ yok; dosya yolları tmp'e monkeypatch'lenir; router.complete sahte fonksiyonlarla taklit.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from nihai.ai import reasoner, router

BUGUN = date.today().isoformat()


def _history_yaz(p, satirlar):
    baslik = "run_at,resolved,days_span,symbols,hit_rate,auc_prob_cal,avg_signed_ret_pct,verdict\n"
    p.write_text(baslik + "".join(satirlar), encoding="utf-8")


@pytest.fixture()
def yollar(tmp_path, monkeypatch):
    h = tmp_path / "history.csv"
    pl = tmp_path / "predictions_log.csv"
    fl = tmp_path / "live_features_log.csv"
    monkeypatch.setattr(reasoner, "HISTORY_CSV", h)
    monkeypatch.setattr(reasoner, "PRED_LOG", pl)
    monkeypatch.setattr(reasoner, "FEATS_LOG", fl)
    return SimpleNamespace(h=h, pl=pl, fl=fl)


def test_dosyalar_yokken_cokmez(yollar):
    facts = reasoner.gather_facts()
    assert facts["history_last"] is None
    assert facts["pred_total"] is None
    assert facts["live_features_today"] is None
    metin = reasoner.deterministic_reasoning(facts, None)
    assert reasoner.CLOSING in metin
    assert "ilk koşu" in metin


def test_facts_delta_ve_uyarilar(yollar):
    simdi = datetime.now()
    _history_yaz(yollar.h, [
        f"{(simdi - timedelta(days=2)).isoformat(timespec='seconds')},100,10,36,0.500,0.49,-0.02,NO_EDGE\n",
        f"{(simdi - timedelta(hours=40)).isoformat(timespec='seconds')},150,11,36,0.470,0.49,-0.02,NO_EDGE\n",
    ])
    # Tur-106 (BULGU-106-18): uyarı "dünden fazla bekleyen" değil, çözülmüş olması GEREKEN bekleyen sayar.
    # MSFT 2026-09-28 seans içi → gecikmiş; NVDA uzak gelecekte → henüz çözülemez (uyarıya girmez).
    yollar.pl.write_text(
        "symbol,bar_ts,action,resolved_at,horizon_bars,interval\n"
        "AAPL,2026-09-28 14:00:00,BUY,2026-01-01T00:00:00,5,1m\n"
        "MSFT,2026-09-28 14:00:00,BUY,,5,1m\n"
        "NVDA,2099-01-02 14:00:00,WAIT,,5,1m\n", encoding="utf-8")
    yollar.fl.write_text(
        "logged_at,symbol\n"
        f"{simdi.isoformat(timespec='seconds')},AAPL\n"      # Tur-106 (BULGU-106-36): sayım son seans açılışından beri
        "2020-01-01T10:00:00,MSFT\n", encoding="utf-8")

    facts = reasoner.gather_facts()
    assert facts["pred_total"] == 3
    assert facts["pred_pending_recent"] == 2
    assert facts["live_features_today"] == 1

    onceki = {"resolved": 100, "hit_rate": 0.500, "verdict": "NO_EDGE", "pred_pending_recent": 0}
    metin = reasoner.deterministic_reasoning(facts, onceki, now=simdi)
    assert "çözülen +50" in metin
    assert "isabet -3.0pp" in metin
    assert "verdikt aynı (NO_EDGE)" in metin
    # son koşu 40 saat önce → boşluk uyarısı (eşik 26s)
    assert "boşluk" in metin
    # çözülmüş olması gereken 1 bekleyen → birikim uyarısı
    assert facts["pred_overdue_recent"] == 1
    assert "bekleyen birikimi: çözülmüş olması gereken 1 tahmin" in metin
    assert reasoner.CLOSING in metin


def test_durustluk_guardi_sayi_uydurmayi_atar(yollar, monkeypatch):
    _history_yaz(yollar.h, ["2026-07-16T23:30:00,200,12,36,0.480,0.49,-0.02,NO_EDGE\n"])
    yakalanan: list[tuple[str, dict]] = []
    sahte_hafiza = SimpleNamespace(
        last=lambda kind: None,
        append=lambda kind, payload: yakalanan.append((kind, payload)),
    )
    # LLM sayıları/verdikti KORUMAYAN bir metin döndürüyor → deterministik metin kullanılmalı
    monkeypatch.setattr(router, "complete", lambda *a, **k: "Her şey harika görünüyor!")
    metin = reasoner.reason(reasoner.gather_facts(), sahte_hafiza)
    assert "NO_EDGE" in metin and "200" in metin
    assert reasoner.CLOSING in metin
    # digest hafızaya yazıldı
    assert yakalanan and yakalanan[0][0] == "daily_digest"
    assert yakalanan[0][1]["resolved"] == 200


def test_llm_korursa_kullanilir(yollar, monkeypatch):
    _history_yaz(yollar.h, ["2026-07-16T23:30:00,200,12,36,0.480,0.49,-0.02,NO_EDGE\n"])
    sahte_hafiza = SimpleNamespace(last=lambda kind: None, append=lambda *a, **k: None)

    def sahte_llm(task, prompt, **k):
        # deterministik metindeki kritik değerleri ve kapanışı AYNEN koruyan akıcı metin
        return ("Sistem durumu: verdikt NO_EDGE, çözülen 200, isabet %48.0. "
                "Değişim yok. ") + reasoner.CLOSING

    monkeypatch.setattr(router, "complete", sahte_llm)
    metin = reasoner.reason(reasoner.gather_facts(), sahte_hafiza)
    assert metin.startswith("Sistem durumu:")


def test_summary_sections_kapanis_icermez(yollar):
    """daily_summary kendi kapanış notunu basar → summary_sections tekrar ETMEMELİ."""
    _history_yaz(yollar.h, ["2026-07-16T23:30:00,200,12,36,0.480,0.49,-0.02,NO_EDGE\n"])
    blok = reasoner.summary_sections(reasoner.gather_facts(), None)
    assert reasoner.CLOSING not in blok
    assert blok.startswith("Değişim:")
