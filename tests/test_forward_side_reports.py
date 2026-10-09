"""BULGU B-1 — gate_sim.json / coverage.json ÜRETİCİSİ.

Bulgunun kendisi kadar önemli: NEDEN yeşil testlerin arkasında saklandı?
`test_daily_summary.py` yalnızca "dosya YOK → 'veri yok' de" yolunu doğruluyordu.
Tüketici sözleşmesi test edilmiş, üretici hiç yazılmamıştı; dolayısıyla "veri yok"
satırı sonsuza kadar DOĞRU test sonucu üretti. Buradaki testler zinciri uçtan uca
bağlar: üretici yazar → tüketici okur → özette GERÇEK verdikt görünür.
"""
from __future__ import annotations

import json

import pytest

from nihai.notifier import daily_summary as ds
from research import forward_harness as fh

SATIRLAR = [
    # çözülmüş + pwin'li aday satırlar (gate_sim bunların üzerinde çalışır)
    {"symbol": "AAPL", "hit": "1", "pwin": "0.62", "pwin_side": "+", "fwd_ret_pct": "0.4",
     "logged_at": "2026-07-20T15:00:00", "action": "LONG"},
    {"symbol": "MSFT", "hit": "0", "pwin": "0.41", "pwin_side": "+", "fwd_ret_pct": "-0.3",
     "logged_at": "2026-07-20T16:00:00", "action": "WAIT"},
    {"symbol": "NVDA", "hit": "", "pwin": "", "pwin_side": "", "fwd_ret_pct": "",
     "logged_at": "2026-07-21T16:00:00", "action": "LONG"},   # bekleyen
]


@pytest.fixture
def yollar(tmp_path, monkeypatch):
    monkeypatch.setattr(fh, "GATE_SIM_JSON", tmp_path / "gate_sim.json")
    monkeypatch.setattr(fh, "COVERAGE_JSON", tmp_path / "coverage.json")
    # Tur-60: üçüncü yan rapor (gölge terfi ölçümü) — yollanmazsa test ÜRETİM
    # data/ dizinine yazardı (conftest zaten yönlendiriyor ama kilit burada açık olsun).
    monkeypatch.setattr(fh, "SHADOW_PROMOTE_JSON", tmp_path / "shadow_promote.json")
    return tmp_path


def test_uretici_iki_dosyayi_da_yazar(yollar):
    durum = fh.publish_side_reports(SATIRLAR)
    # Tur-60: yan rapor kümesi 2 → 3 (shadow_promote). Sayı bir KARARDIR, kilit
    # bilerek TAM KÜME üzerindedir: sessizce bir rapor düşerse test kırmızı olsun.
    assert durum == {"gate_sim": True, "coverage": True, "shadow_promote": True}
    assert (yollar / "gate_sim.json").exists()
    assert (yollar / "coverage.json").exists()
    assert (yollar / "shadow_promote.json").exists()


def test_uretilen_json_tuketicinin_bekledigi_anahtarlari_tasir(yollar):
    """Üretici↔tüketici SÖZLEŞMESİ: daily_summary label/uplift_pp ve total/resolved/pending okur."""
    fh.publish_side_reports(SATIRLAR)
    gs = json.loads((yollar / "gate_sim.json").read_text(encoding="utf-8"))
    cov = json.loads((yollar / "coverage.json").read_text(encoding="utf-8"))
    assert gs.get("label") or gs.get("verdict")      # biri olmalı (özet ikisini de kabul eder)
    assert {"total", "resolved", "pending"} <= set(cov)
    assert cov["total"] == 3 and cov["resolved"] == 2 and cov["pending"] == 1
    for d in (gs, cov):
        assert d["source"] == "research.forward_harness"
        assert d["generated_at"]                     # bayatlık kontrolünün dayanağı


def test_zincir_ozette_gercek_verdikti_gosterir(yollar):
    """Regresyon: üretici koştuktan sonra özet artık 'veri yok' DEMEMELİ."""
    fh.publish_side_reports(SATIRLAR)
    gs = json.loads((yollar / "gate_sim.json").read_text(encoding="utf-8"))
    cov = json.loads((yollar / "coverage.json").read_text(encoding="utf-8"))
    metin = ds.build_summary({"today": gs["generated_at"][:10], "gate_sim": gs, "coverage": cov})
    assert "Meta-kapı canlı sim: veri yok." not in metin
    assert "Kapsama: toplam 3" in metin


def test_uretici_bozuk_veride_cokmez(yollar, monkeypatch):
    """Fail-soft: bir tarafın hesabı patlasa DİĞERİ yine yazılmalı (gözlem hattı
    günlük harness'i düşürmemeli)."""
    def patla(*_a, **_k):
        raise RuntimeError("bilerek")
    monkeypatch.setattr(fh.pt, "gate_sim", patla)
    durum = fh.publish_side_reports(SATIRLAR)
    assert durum == {"gate_sim": False, "coverage": True, "shadow_promote": True}
    assert (yollar / "coverage.json").exists()
    assert (yollar / "shadow_promote.json").exists()


# --- Bayatlık koruması ----------------------------------------------------------


def test_taze_rapor_bayat_etiketi_almaz():
    metin = ds.build_summary({
        "today": "2026-07-26",
        "gate_sim": {"label": "GATE_HURTS", "uplift_pp": -0.013, "generated_at": "2026-07-26T12:00:00"},
    })
    assert 'Meta-kapı canlı sim: "GATE_HURTS" (uplift -0.013pp)' in metin
    assert "BAYAT" not in metin


def test_bayat_rapor_isaretlenir():
    """Üretici görev ölürse dosya diskte KALIR; özet onu taze sanmamalı."""
    metin = ds.build_summary({
        "today": "2026-07-26",
        "gate_sim": {"label": "GATE_HURTS", "uplift_pp": -0.013, "generated_at": "2026-07-20T12:00:00"},
    })
    assert "BAYAT: 6 gün önce" in metin


def test_damgasiz_rapor_yas_bilinmiyor_der():
    metin = ds.build_summary({
        "today": "2026-07-26",
        "gate_sim": {"label": "GATE_HURTS", "uplift_pp": -0.013},
    })
    assert "yaş bilinmiyor" in metin
