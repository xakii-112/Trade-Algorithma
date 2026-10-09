"""research.edge_lab.ledger testleri — PROGRAM-DÜZEYİ çoklu-test muhasebesi.

Kilitlenen sözleşme üç maddedir:
  1. BOŞ defterde deflasyon eski sabitle BİT-DENK (kapı sıkılaştırması geçmişi bozmaz;
     ACI DERSLER: sıkılaştırma yeni sonuç ÜRETEMEZ).
  2. Defterdeki AYRIK aday sayısı arttıkça `n_trials` artar ve DSR MONOTON DÜŞER —
     yani aradıkça kanıt eşiği sıkışır (BACKLOG.md çoklu-test muhasebesi).
  3. Aynı adayı tekrar koşmak sayacı ŞİŞİRMEZ (geliştirme döngüsü eşiği kaçırmasın),
     veri yükleme hatası ise deftere HİÇ yazılmaz (yanlış dosya yolu bir deneme değildir).

Gerçek piyasa verisi YOK; tüm veri setleri burada üretilir.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.deflated import dsr, sharpe_estimator_var, sharpe_moments
from research.edge_lab import harness, ledger
from research.edge_lab.harness import run_candidate


@pytest.fixture()
def defter(tmp_path):
    return tmp_path / "defter.jsonl"


@pytest.fixture()
def noise_csv(tmp_path):
    """Etiket özelliklerden bağımsız — kapılar FAIL etmeli (yanlış-pozitif üretmemeli)."""
    rng = np.random.default_rng(0)
    n = 600
    p = tmp_path / "noise.csv"
    pd.DataFrame({
        "ts": pd.date_range("2022-01-01", periods=n, freq="h"),
        "symbol": [f"SYM{i % 20}" for i in range(n)],
        "f1": rng.normal(size=n), "f2": rng.normal(size=n),
        "label": rng.integers(0, 2, size=n),
    }).to_csv(p, index=False)
    return p


def _satir(name: str, *, passed: bool = False) -> dict:
    """`record`'un beklediği en küçük verdikt biçimi."""
    return {"candidate": name, "dataset": "x.csv", "n_rows": 10,
            "gates": [{"name": "dsr", "passed": passed, "detail": "", "metrics": {"n_trials": 4}}],
            "dsr": 0.1, "exec_lag_effect": None, "generated_at": "2026-01-01T00:00:00+00:00"}


# ---------------------------------------------------------------------------
# 1) Geriye dönük denklik + monotonluk
# ---------------------------------------------------------------------------
def test_bos_defterde_n_trials_eski_sabitle_bit_denk(defter):
    """Defter boşken deflasyon `_DSR_N_TRIALS`'in ta kendisi olmalı: bu değişiklik
    bugüne kadarki DÖRT koşunun verdiktini DEĞİŞTİREMEZ."""
    assert ledger.n_trials_for("ilk", len(harness._THR_CANDIDATES), defter) == harness._DSR_N_TRIALS


def test_defter_buyudukce_n_trials_artar(defter):
    for ad in ("overnight", "leadlag", "rebalance"):
        ledger.record(_satir(ad), path=defter)
    # 3 ayrık geçmiş + bu koşu = 4 aday x 4 eşik
    assert ledger.n_trials_for("vrp", 4, defter) == 16
    # Defterde ZATEN olan bir ad yeni aday saymaz
    assert ledger.n_trials_for("leadlag", 4, defter) == 12


def test_ayni_adi_tekrar_kosmak_sayaci_sismez(defter):
    """Hata düzeltip yeniden koşmak bir 'deneme' değildir — yoksa geliştirme döngüsü
    kanıt eşiğini kaçırırdı. Defter yine de her koşuyu kaydeder (denetim izi)."""
    for _ in range(5):
        ledger.record(_satir("ayni"), path=defter)
    assert len(ledger.read_entries(defter)) == 5
    assert ledger.distinct_candidates(defter) == ["ayni"]
    assert ledger.n_trials_for("ayni", 4, defter) == 4


def test_artan_n_trials_DSR_yi_MONOTON_dusurur():
    """Asıl bilimsel özellik: AYNI seride yalnız çoklu-test sayısı büyüyünce DSR düşer.
    (Deflasyonun yönü yanlış olsaydı 'çok denedim' kanıtı GÜÇLENDİRİRDİ.)"""
    rng = np.random.default_rng(7)
    seri = rng.normal(loc=0.08, scale=1.0, size=400)
    mom = sharpe_moments(seri)
    var_t = sharpe_estimator_var(mom["sr"], mom["n"], skew=mom["skew"], kurt=mom["kurt"])
    degerler = [dsr(mom["sr"], mom["n"], n, var_t, skew=mom["skew"], kurt=mom["kurt"])
                for n in (1, 4, 16, 64, 256)]
    assert all(a >= b for a, b in zip(degerler, degerler[1:], strict=False)), degerler
    assert degerler[0] > degerler[-1], "deflasyon hiç ısırmıyor"


def test_stage_dsr_n_trials_metrige_ve_metne_yazilir():
    """Kullanılan eşik DENETLENEBİLİR olmalı: hangi deflasyonla PASS/FAIL verildiği
    rapordan okunabilmeli (aksi halde defter değişince verdiktler kıyaslanamaz)."""
    seri = np.random.default_rng(1).normal(loc=0.05, size=200)
    gate, _ = harness._stage_dsr((seri, 0.0), 40)
    assert gate.metrics["n_trials"] == 40
    assert "n_trials=40" in gate.detail


# ---------------------------------------------------------------------------
# 2) run_candidate ile uçtan uca
# ---------------------------------------------------------------------------
def test_run_candidate_deftere_yazar_ve_sonraki_esigi_yukseltir(noise_csv, defter):
    v1 = run_candidate("aday_a", noise_csv, ["f1", "f2"], "label", n_folds=4, ledger_path=defter)
    v2 = run_candidate("aday_b", noise_csv, ["f1", "f2"], "label", n_folds=4, ledger_path=defter)

    satirlar = ledger.read_entries(defter)
    assert [r["candidate"] for r in satirlar] == ["aday_a", "aday_b"]
    assert satirlar[0]["features"] == ["f1", "f2"]      # ad yeniden kullanımı denetlenebilsin
    assert satirlar[0]["all_passed"] is False
    assert "purged_cv" in satirlar[0]["failed_gates"]

    n1 = next(g for g in v1["gates"] if g["name"] == "dsr")["metrics"].get("n_trials")
    n2 = next(g for g in v2["gates"] if g["name"] == "dsr")["metrics"].get("n_trials")
    assert (n1, n2) == (4, 8), "ikinci aday PROGRAM-düzeyi sayaçla deflate edilmeliydi"


def test_prior_runs_BU_kosudan_ONCEKI_gecmisi_raporlar(noise_csv, defter):
    v1 = run_candidate("tekrar", noise_csv, ["f1", "f2"], "label", n_folds=4, ledger_path=defter)
    v2 = run_candidate("tekrar", noise_csv, ["f1", "f2"], "label", n_folds=4, ledger_path=defter)
    assert v1["prior_runs"] == {"n": 0, "fail": 0, "pass": 0, "last": None}
    assert v2["prior_runs"]["n"] == 1 and v2["prior_runs"]["fail"] == 1


def test_veri_yukleme_hatasi_deftere_YAZILMAZ(tmp_path, defter):
    """Yanlış dosya yolu bir deneme değildir; sonraki adayın eşiğini yükseltmemeli."""
    v = run_candidate("yok", tmp_path / "olmayan.csv", ["f1"], "label", ledger_path=defter)
    assert v["gates"][0]["name"] == "veri_yukleme"
    assert ledger.read_entries(defter) == []


def test_record_run_false_yazmaz_ama_esik_yine_program_duzeyi(noise_csv, defter):
    ledger.record(_satir("gecmis_1"), path=defter)
    ledger.record(_satir("gecmis_2"), path=defter)
    v = run_candidate("kesif", noise_csv, ["f1", "f2"], "label", n_folds=4,
                      ledger_path=defter, record_run=False)
    assert len(ledger.read_entries(defter)) == 2, "record_run=False yazmamalıydı"
    assert next(g for g in v["gates"] if g["name"] == "dsr")["metrics"].get("n_trials") == 12


# ---------------------------------------------------------------------------
# 3) Defterin dayanıklılığı
# ---------------------------------------------------------------------------
def test_bozuk_ve_bos_satirlar_atlanir(defter):
    """Yalnız-ekleme günlüğünde çökme SON satırı yarım bırakabilir; okuma düşmemeli."""
    defter.write_text(
        json.dumps({"candidate": "iyi", "all_passed": False}) + "\n"
        + "\n" + "{yarim satir" + "\n" + json.dumps(["liste", "dict degil"]) + "\n",
        encoding="utf-8")
    assert [r["candidate"] for r in ledger.read_entries(defter)] == ["iyi"]
    assert ledger.n_trials_for("yeni", 4, defter) == 8


def test_ledger_path_env_ile_yonlendirilebilir(tmp_path, monkeypatch):
    hedef = tmp_path / "env_defteri.jsonl"
    monkeypatch.setenv("NIHAI_EDGE_LEDGER", str(hedef))
    ledger.record(_satir("env_aday"))
    assert ledger.ledger_path() == hedef
    assert [r["candidate"] for r in ledger.read_entries()] == ["env_aday"]


def test_summary_bos_ve_dolu_defterde_okunabilir(defter):
    assert "BOŞ" in ledger.summary(defter)
    ledger.record(_satir("aday_x"), path=defter)
    metin = ledger.summary(defter)
    assert "aday_x" in metin and "FAIL" in metin
    assert "2 ayrık aday" in metin      # bir sonraki YENİ adayın sayacı
