"""lab.py testleri — kod yazma/çalıştırma laboratuvarı (Tur-16 WP-C): sandbox kaçış
denemeleri, şablon determinizmi, LLM-yok fail-soft, JSONL log atomikliği."""
from __future__ import annotations

import json

import numpy as np
import pytest

from nihai.ai import lab, quant_math


@pytest.fixture(autouse=True)
def _tmp_log(tmp_path, monkeypatch):
    """Her test kendi log dosyasını kullanır (gerçek data/ai_lab_log.jsonl'e ASLA yazılmaz)."""
    monkeypatch.setattr(lab, "AI_LAB_LOG_FILE", tmp_path / "ai_lab_log.jsonl")
    return tmp_path


def _log_lines():
    if not lab.AI_LAB_LOG_FILE.exists():
        return []
    return [json.loads(x) for x in lab.AI_LAB_LOG_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# SANDBOX KAÇIŞ TESTLERİ (zorunlu)
# ---------------------------------------------------------------------------
def test_sandbox_open_engellenir():
    r = lab.run_sandbox("open('kacis_denemesi.txt', 'w')")
    assert r["ok"] is False
    assert "NameError" in r["error"]


def test_sandbox_dunder_import_os_engellenir():
    r = lab.run_sandbox("__import__('os')")
    assert r["ok"] is False
    assert "ImportError" in r["error"]


def test_sandbox_import_os_statement_engellenir():
    r = lab.run_sandbox("import os\nresult = os.getcwd()")
    assert r["ok"] is False
    assert "ImportError" in r["error"]


def test_sandbox_import_subprocess_engellenir():
    r = lab.run_sandbox("import subprocess")
    assert r["ok"] is False
    assert "ImportError" in r["error"]


def test_sandbox_import_socket_engellenir():
    """Ağ erişimi: socket'in KENDİSİ beyaz-liste dışı olduğundan import bile başarısız."""
    r = lab.run_sandbox("import socket")
    assert r["ok"] is False
    assert "ImportError" in r["error"]


def test_sandbox_eval_exec_compile_input_engellenir():
    for expr in ("eval('1+1')", "exec('x=1')", "compile('1', '<s>', 'eval')", "input()"):
        r = lab.run_sandbox(f"result = {expr}")
        assert r["ok"] is False, expr
        assert "NameError" in r["error"], expr


def test_sandbox_dosya_yazma_open_uzerinden_engellenir(tmp_path):
    hedef = tmp_path / "ureti_dosyasi.txt"
    r = lab.run_sandbox(f"open(r'{hedef}', 'w').write('kaçak veri')")
    assert r["ok"] is False
    assert not hedef.exists()


def test_sandbox_goreli_import_reddedilir():
    r = lab.run_sandbox("from . import something")
    assert r["ok"] is False


def test_sandbox_sonsuz_dongu_zaman_asimi():
    r = lab.run_sandbox("while True:\n    pass", timeout_s=1)
    assert r["ok"] is False
    assert "zaman aşımı" in r["error"]
    assert r["elapsed_s"] < 5.0  # timeout_s+makul_tolerans içinde DÖNMELİ (arka planda beklemez)


# ---------------------------------------------------------------------------
# Meşru kullanım — beyaz-liste erişilebilir, girdi/çıktı sözleşmesi
# ---------------------------------------------------------------------------
def test_sandbox_numpy_pandas_math_scipy_beyaz_liste():
    r = lab.run_sandbox(
        "import math\nimport numpy as npx\nimport pandas as pdx\nimport scipy.stats as st\n"
        "result = float(npx.mean([1.0, 2.0, 3.0])) + math.sqrt(4.0) + float(st.norm.cdf(0.0)) "
        "+ float(len(pdx.Series([1, 2])))"
    )
    assert r["ok"] is True
    assert r["result"] == pytest.approx(2.0 + 2.0 + 0.5 + 2.0, rel=1e-6)


def test_sandbox_onceden_yuklu_modul_isimleri_import_gerektirmez():
    r = lab.run_sandbox("result = float(np.mean(closes))", inputs={"closes": np.array([2.0, 4.0, 6.0])})
    assert r["ok"] is True
    assert r["result"] == pytest.approx(4.0)


def test_sandbox_stdout_yakalanir():
    r = lab.run_sandbox("print('merhaba')\nresult = 1")
    assert r["ok"] is True
    assert "merhaba" in r["output"]


def test_sandbox_result_atanmazsa_none():
    r = lab.run_sandbox("x = 1 + 1")
    assert r["ok"] is True
    assert r["result"] is None


def test_sandbox_calisma_zamani_hatasi_yakalanir():
    r = lab.run_sandbox("result = 1 / 0")
    assert r["ok"] is False
    assert "ZeroDivisionError" in r["error"]


# ---------------------------------------------------------------------------
# Loglama — atomik JSONL bütünlüğü
# ---------------------------------------------------------------------------
def test_sandbox_her_calisma_loglanir():
    lab.run_sandbox("result = 1")
    lab.run_sandbox("result = 2")
    rows = _log_lines()
    assert len(rows) == 2
    assert all(r["kind"] == "sandbox" for r in rows)
    assert all("ts" in r and "code" in r for r in rows)


def test_log_dosyasi_her_satir_bagimsiz_json():
    for i in range(5):
        lab.run_sandbox(f"result = {i}")
    text = lab.AI_LAB_LOG_FILE.read_text(encoding="utf-8")
    lines = [x for x in text.splitlines() if x.strip()]
    assert len(lines) == 5
    for line in lines:
        json.loads(line)


# ---------------------------------------------------------------------------
# run_experiment — hazır şablonlar (determinizm + dürüstlük guard'ı)
# ---------------------------------------------------------------------------
def _synthetic_closes(n=250, seed=11):
    return quant_math.mc_gbm_paths(100.0, 0.05, 0.20, n - 1, 1, seed)[0]


def test_run_experiment_bilinmeyen_isim_value_error():
    with pytest.raises(ValueError):
        lab.run_experiment("olmayan_sablon")


def test_vol_model_karsilastirma_4_tahminci_ve_not_alani():
    closes = _synthetic_closes()
    r = lab.run_experiment("vol_model_karsilastirma", closes=closes)
    assert set(r["estimates"]) == {"close_to_close", "ewma", "rolling_realized", "garch11"}
    assert r["not"] == "tanımlayıcı analiz; kenar iddiası değildir"


def test_vol_model_karsilastirma_determinizm():
    closes = _synthetic_closes()
    r1 = lab.run_experiment("vol_model_karsilastirma", closes=closes)
    r2 = lab.run_experiment("vol_model_karsilastirma", closes=closes)
    assert r1["estimates"] == r2["estimates"]


def test_vol_model_karsilastirma_yetersiz_veri_hata_dict():
    r = lab.run_experiment("vol_model_karsilastirma", closes=[100.0, 101.0, 99.0])
    assert "error" in r and r["not"] == "tanımlayıcı analiz; kenar iddiası değildir"


def test_korelasyon_rejimi_panel_dict():
    closes_a = _synthetic_closes(seed=21)
    closes_b = _synthetic_closes(seed=22)
    r = lab.run_experiment("korelasyon_rejimi", panel={"A": closes_a, "B": closes_b}, window=40)
    assert r["regime"] in {"dusuk_korelasyon", "orta_korelasyon", "yuksek_korelasyon"}
    assert -1.0 <= r["overall_mean_corr"] <= 1.0


def test_korelasyon_rejimi_gecersiz_panel_hata():
    r = lab.run_experiment("korelasyon_rejimi", panel={"tek_sembol": [1.0, 2.0, 3.0]})
    assert "error" in r


def test_olay_etudu_car_aar_sekli():
    closes = _synthetic_closes(n=200, seed=31)
    r = lab.run_experiment("olay_etudu", closes=closes, event_indices=[50, 100, 150], pre=5, post=5)
    assert r["n_events"] == 3
    assert len(r["aar"]) == 5 + 5 + 1
    assert "car_mean" in r
    assert r["model"].startswith("ortalama-düzeltmeli")


def test_olay_etudu_gecersiz_indeksler_hata():
    closes = _synthetic_closes(n=50, seed=32)
    r = lab.run_experiment("olay_etudu", closes=closes, event_indices=[9999])
    assert "error" in r


def test_dagilim_testi_normal_ve_non_normal_ayrimi():
    rng = np.random.default_rng(42)
    normal_returns = rng.normal(0, 0.01, 500)
    r_normal = lab.run_experiment("dagilim_testi", returns=normal_returns)
    assert "jarque_bera_p" in r_normal

    # ağır-kuyruklu (t-dağılımı, düşük serbestlik derecesi) -> normallik reddedilmeli
    skewed = rng.standard_t(df=2, size=500) * 0.01
    r_skewed = lab.run_experiment("dagilim_testi", returns=skewed)
    assert r_skewed["excess_kurtosis"] > r_normal["excess_kurtosis"]


def test_dagilim_testi_yetersiz_veri_hata():
    r = lab.run_experiment("dagilim_testi", returns=[0.01, 0.02])
    assert "error" in r


def test_ozellik_tarama_determinizm_ve_siralama():
    rng = np.random.default_rng(7)
    n = 200
    x_signal = rng.normal(0, 1, n)
    x_noise = rng.normal(0, 1, n)
    y = (x_signal + rng.normal(0, 0.1, n) > 0).astype(int)
    X = np.column_stack([x_signal, x_noise])
    r1 = lab.run_experiment("ozellik_tarama", X=X, y=y, feature_names=["sinyal", "gurultu"], seed=5)
    r2 = lab.run_experiment("ozellik_tarama", X=X, y=y, feature_names=["sinyal", "gurultu"], seed=5)
    assert r1["ranking_by_permutation_importance"] == r2["ranking_by_permutation_importance"]
    assert r1["ranking_by_permutation_importance"][0] == "sinyal"


def test_ozellik_tarama_yetersiz_veri_hata():
    r = lab.run_experiment("ozellik_tarama", X=[[1.0, 2.0]], y=[1])
    assert "error" in r


def test_run_experiment_loglanir():
    lab.run_experiment("dagilim_testi", returns=list(np.random.default_rng(1).normal(0, 1, 50)))
    rows = _log_lines()
    assert any(r["kind"] == "experiment" and r["name"] == "dagilim_testi" for r in rows)


def test_tum_sablonlar_kenar_iddiasi_tasimaz():
    closes = _synthetic_closes(n=200, seed=99)
    results = [
        lab.run_experiment("vol_model_karsilastirma", closes=closes),
        lab.run_experiment("korelasyon_rejimi", panel={"A": closes, "B": _synthetic_closes(n=200, seed=98)}),
        lab.run_experiment("olay_etudu", closes=closes, event_indices=[50, 100]),
        lab.run_experiment("dagilim_testi", returns=np.diff(np.log(closes))),
    ]
    for r in results:
        assert r["not"] == "tanımlayıcı analiz; kenar iddiası değildir"


# ---------------------------------------------------------------------------
# ask_llm_analysis — FAIL-SOFT (router.complete monkeypatch'lenir, gerçek ağ ÇAĞRISI YAPILMAZ)
# ---------------------------------------------------------------------------
def test_ask_llm_analysis_fail_soft_sablon(monkeypatch):
    monkeypatch.setattr(lab.router, "complete", lambda *a, **k: None)
    r = lab.ask_llm_analysis("volatilite nedir?", {"vol": 0.2})
    assert r["source"] == "template"
    assert r["not"] == "tanımlayıcı analiz; kenar iddiası değildir"
    assert "ŞABLON YANIT" in r["answer"]


def test_ask_llm_analysis_llm_basarili(monkeypatch):
    monkeypatch.setattr(lab.router, "complete", lambda *a, **k: "LLM yanıtı burada.")
    r = lab.ask_llm_analysis("volatilite nedir?", {"vol": 0.2})
    assert r["source"] == "llm"
    assert r["answer"] == "LLM yanıtı burada."


def test_ask_llm_analysis_sablon_determinizm(monkeypatch):
    monkeypatch.setattr(lab.router, "complete", lambda *a, **k: None)
    r1 = lab.ask_llm_analysis("soru", {"a": 1})
    r2 = lab.ask_llm_analysis("soru", {"a": 1})
    assert r1["answer"] == r2["answer"]


def test_ask_llm_analysis_context_none_calisir(monkeypatch):
    monkeypatch.setattr(lab.router, "complete", lambda *a, **k: None)
    r = lab.ask_llm_analysis("bağlamsız soru")
    assert r["source"] == "template"
