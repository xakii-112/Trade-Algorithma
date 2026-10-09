"""shadow_runner.repeat_report testleri — denetim bulgusu B-6 / plan maddesi O5.

Bulgu: gölge tur tetiği ABD seansından ÖNCE koşuyordu; günlük barlar turlar arasında
değişmediği için kararların %75.8'i aynı (gün, sembol, posterior) üçlüsünün tekrarıydı.
Plan O5 "tekrar oranı %10 altına insin" diyordu ama bunu ÖLÇEN araç yoktu — yazılı olup
ölçülemeyen bir kabul kriteri kapı değildir.
"""
from __future__ import annotations

import json

import pytest

from nihai.ai import shadow_runner


@pytest.fixture()
def gunluk(tmp_path, monkeypatch):
    """decision.DECISIONS_FILE'ı geçici dosyaya yönlendiren yazıcı döndürür."""
    path = tmp_path / "ai_decisions.jsonl"
    from nihai.ai import decision as decision_mod
    monkeypatch.setattr(decision_mod, "DECISIONS_FILE", path)

    def yaz(kayitlar):
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kayitlar) + "\n",
                        encoding="utf-8")
    return path, yaz


def _kayit(gun: str, symbol: str, posterior: float) -> dict:
    return {"ts": f"{gun}T12:00:00+00:00", "symbol": symbol, "posterior": posterior,
            "action": "ABSTAIN", "strategy": "event_breakout"}


def test_tekrar_orani_hesaplanir(gunluk):
    _, yaz = gunluk
    # AAPL aynı gün 3 kez AYNI posterior → 2 fazlalık; MSFT tek → 0
    yaz([_kayit("2026-07-25", "AAPL", 0.5228),
         _kayit("2026-07-25", "AAPL", 0.5228),
         _kayit("2026-07-25", "AAPL", 0.5228),
         _kayit("2026-07-25", "MSFT", 0.4910)])
    r = shadow_runner.repeat_report(days=0)
    assert r["n"] == 4 and r["essiz"] == 2
    assert r["fazlalik"] == 2
    assert r["oran"] == pytest.approx(0.5)
    assert "GEÇİLMEDİ" in r["verdikt"]


def test_farkli_gun_tekrar_sayilmaz(gunluk):
    """Aynı sembol+posterior FARKLI günlerde tekrar değildir (gün anahtarın parçası)."""
    _, yaz = gunluk
    yaz([_kayit("2026-07-24", "AAPL", 0.52), _kayit("2026-07-25", "AAPL", 0.52)])
    r = shadow_runner.repeat_report(days=0)
    assert r["fazlalik"] == 0 and r["oran"] == 0.0
    assert "KABUL" in r["verdikt"]


def test_farkli_posterior_tekrar_sayilmaz(gunluk):
    _, yaz = gunluk
    yaz([_kayit("2026-07-25", "AAPL", 0.5228), _kayit("2026-07-25", "AAPL", 0.5301)])
    r = shadow_runner.repeat_report(days=0)
    assert r["fazlalik"] == 0


def test_posterior_4_haneye_yuvarlanir(gunluk):
    """Kayan-nokta gürültüsü sahte 'eşsiz' üretmemeli."""
    _, yaz = gunluk
    yaz([_kayit("2026-07-25", "AAPL", 0.52280001), _kayit("2026-07-25", "AAPL", 0.52280009)])
    r = shadow_runner.repeat_report(days=0)
    assert r["essiz"] == 1 and r["fazlalik"] == 1


def test_gun_penceresi_eski_kayitlari_eler(gunluk):
    _, yaz = gunluk
    yaz([_kayit("2020-01-01", "AAPL", 0.52), _kayit("2020-01-01", "AAPL", 0.52)])
    assert shadow_runner.repeat_report(days=7)["n"] == 0
    assert shadow_runner.repeat_report(days=0)["n"] == 2


def test_bozuk_satir_sayima_girmez(gunluk):
    path, yaz = gunluk
    yaz([_kayit("2026-07-25", "AAPL", 0.52)])
    with path.open("a", encoding="utf-8") as f:
        f.write("{bozuk json\n")
        f.write(json.dumps({"ts": "2026-07-25T12:00:00+00:00"}) + "\n")   # eksik alan
        f.write("\n")
    r = shadow_runner.repeat_report(days=0)
    assert r["n"] == 1


def test_gunluk_yoksa_cokmez(tmp_path, monkeypatch):
    from nihai.ai import decision as decision_mod
    monkeypatch.setattr(decision_mod, "DECISIONS_FILE", tmp_path / "yok.jsonl")
    r = shadow_runner.repeat_report()
    assert r["n"] == 0 and r["oran"] is None
    assert "yok" in r["not"]


def test_cli_repeat_report_tur_calistirmaz(gunluk, capsys, monkeypatch):
    """--repeat-report salt-okunur olmalı: run_once ASLA çağrılmamalı."""
    _, yaz = gunluk
    yaz([_kayit("2026-07-25", "AAPL", 0.52)])

    def _patlar(*_a, **_k):
        raise AssertionError("run_once çağrılmamalıydı (salt-okunur rapor)")

    monkeypatch.setattr(shadow_runner, "run_once", _patlar)
    assert shadow_runner.main(["--repeat-report", "--days", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 1
