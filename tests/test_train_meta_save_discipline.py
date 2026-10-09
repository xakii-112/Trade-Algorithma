"""train_meta --save kaydetme disiplini — REGRESYON GUARD'ı.

Bağlam (2026-07-07): 10y saatlik PIT setinde logistic time-split 38.858 işlemden
yalnız 6'sını tutup PF 3.68 gösterdi; symbol-split 24 işlem gösterdi. Eski --save
mantığı "iki bölmede de best_gate var" diye modeli KAYDETTİ — oysa purged CV aynı
modelde kenar YOK diyordu. Avuç-dolusu işleme dayanan "kenar" istatistiksel gürültü.
Düzeltme: her bölmede tutulan-işlem sayısı >= MIN_GATE_KEEP şartı.

Bu test o davranışı doğrudan çağırıp (main değil, saf karar mantığı) gürültünün
DAYANIKLI sayılmadığını sabitler.
"""
from __future__ import annotations


def _durable_kinds(results: dict, models: list[str], min_gate_keep: int) -> dict:
    """train_meta.main() --save bloğundaki dayanıklılık kararının saf kopyası.
    best_gate = (thr, pf, e, wr, kept). Her bölmede kept >= min_gate_keep şartı."""
    durable: dict[str, float] = {}
    for mk in models:
        gt = results.get((mk, "time"), {}).get("best_gate")
        gs = results.get((mk, "symbol"), {}).get("best_gate")
        if gt and gs and gt[4] >= min_gate_keep and gs[4] >= min_gate_keep:
            durable[mk] = min(gt[2], gs[2])
    return durable


def test_handful_of_trades_not_durable():
    """6 ve 24 işlemlik 'kenar' (gerçek 2026-07-07 gözlemi) DAYANIKLI sayılmamalı."""
    results = {
        ("logistic", "time"):   {"best_gate": (0.55, 3.68, 15697.0, 33.3, 6)},
        ("logistic", "symbol"): {"best_gate": (0.45, 1.86, 7821.0, 34.0, 24)},
    }
    assert _durable_kinds(results, ["logistic"], min_gate_keep=50) == {}


def test_ample_two_split_edge_is_durable():
    """Her iki bölmede de bol örneklemli (>=50) gerçek kenar KAYDEDİLEBİLİR sayılır."""
    results = {
        ("gbm", "time"):   {"best_gate": (0.55, 1.40, 900.0, 40.0, 120)},
        ("gbm", "symbol"): {"best_gate": (0.50, 1.35, 700.0, 39.0, 200)},
    }
    d = _durable_kinds(results, ["gbm"], min_gate_keep=50)
    assert "gbm" in d and d["gbm"] == 700.0   # muhafazakâr: kötü bölmenin E'si


def test_one_split_ample_other_thin_not_durable():
    """Bir bölme bol, diğeri ince → yine dayanıklı DEĞİL (tek-bölme overfit koruması)."""
    results = {
        ("xgb", "time"):   {"best_gate": (0.60, 1.86, 5934.0, 41.0, 231)},
        ("xgb", "symbol"): {"best_gate": (0.55, 2.0, 500.0, 45.0, 8)},
    }
    assert _durable_kinds(results, ["xgb"], min_gate_keep=50) == {}


def test_no_gate_at_all_not_durable():
    results = {
        ("gbm", "time"):   {"best_gate": None},
        ("gbm", "symbol"): {"best_gate": (0.50, 1.35, 700.0, 39.0, 200)},
    }
    assert _durable_kinds(results, ["gbm"], min_gate_keep=50) == {}
