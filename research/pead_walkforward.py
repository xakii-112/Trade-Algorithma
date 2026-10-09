"""
pead_walkforward.py — PEAD kesit-ML'i YIL-YIL walk-forward ile sına (regime robustluğu).

build_pead_dataset çıktısını (2018 as-of, ~2021-2026 olayları — yfinance earnings ~4yıl
kısıtı) alır; her test yılı Y için `entry_year < Y` üzerinde eğitir, `Y` yılında test eder
(purged: gelecek sızıntısı yok, eğitim tümüyle Y öncesi). Her yıl: OOS-AUC + kapı(≥thr)
BASE'e karşı + SURVIVORSHIP (removed-since'te kapı pozitif mi). §19 disiplininin yıl-bazlı
versiyonu: kenar GERÇEKSE çoğu yılda + removed-since'te görünmeli.

    python -m research.pead_walkforward
    python -m research.pead_walkforward --csv data/pead_ml_dataset_pit_sp500_2018-01-01.csv --thr 0.55
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.config import DATA_DIR
from research.train_meta import _auc, _pf_exp
from research.train_pead_ml import PEAD_FEATURES, _build_model, prepare


def _gate_line(pnl: np.ndarray, p: np.ndarray, thr: float) -> tuple[str, bool]:
    """BASE vs kapı(≥thr) + kazandı-mı bayrağı."""
    b_pf, b_e, b_wr = _pf_exp(pnl)
    kept = pnl[p >= thr]
    if len(kept) < 5:
        return f"BASE WR{b_wr:4.0f} PF{b_pf:4.2f} E{b_e:+7.0f} | kapı: az işlem", False
    g_pf, g_e, g_wr = _pf_exp(kept)
    ok = g_e > b_e and g_pf > 1.0
    return (f"BASE WR{b_wr:4.0f} PF{b_pf:4.2f} E{b_e:+7.0f} | "
            f"kapı al{len(kept):<4} WR{g_wr:4.0f} PF{g_pf:4.2f} E{g_e:+7.0f} {'✓' if ok else '✗'}"), ok


def main():
    ap = argparse.ArgumentParser(description="PEAD kesit-ML yıl-yıl walk-forward")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--thr", type=float, default=0.55)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--min-train", type=int, default=200, help="asgari eğitim örneği")
    args = ap.parse_args()

    csv_path = args.csv
    if not csv_path:
        cands = sorted(glob.glob(str(DATA_DIR / "pead_ml_dataset_*.csv")), key=os.path.getsize)
        csv_path = cands[-1] if cands else None
    if not csv_path:
        print("Dataset yok. Önce: python -m research.build_pead_dataset --as-of 2018-01-01")
        return

    df = prepare(pd.read_csv(csv_path))
    df["yr"] = pd.to_datetime(df["entry_time"]).dt.year
    years = sorted(df["yr"].unique())
    print(f"Dataset: {csv_path}")
    print(f"Olay {len(df)} | yıllar {years[0]}-{years[-1]} | özellik {len(PEAD_FEATURES)} (SUE dahil)")
    print(f"Walk-forward: her yıl için eğitim=önceki tüm yıllar, test=o yıl | kapı eşik {args.thr}\n")

    for model_kind in ("logistic", "gbm"):
        print(f"{'='*78}\n  MODEL = {model_kind.upper()}\n{'='*78}")
        print(f"  {'test-yıl':>8} {'eğitim-n':>9} {'test-n':>7} {'OOS-AUC':>8}   genel-kapı / removed-since-kapı")
        wins = rm_wins = n_eval = 0
        for Y in years:
            tr = df[df["yr"] < Y]
            te = df[df["yr"] == Y]
            if len(tr) < args.min_train or len(te) < 30 or tr["label"].nunique() < 2:
                print(f"  {Y:>8} {len(tr):>9} {len(te):>7}   (atlandı — yetersiz eğitim/test)")
                continue
            m = _build_model(model_kind, tr[PEAD_FEATURES].to_numpy(float),
                             tr["label"].to_numpy(float), l2=args.l2)
            pte = m.predict_proba(te[PEAD_FEATURES].to_numpy(float))
            pnlte = te["pnl"].to_numpy(float)
            rm = te["removed_since"].to_numpy(int)
            auc = _auc(te["label"].to_numpy(float), pte)
            all_line, all_ok = _gate_line(pnlte, pte, args.thr)
            rm_line, rm_ok = _gate_line(pnlte[rm == 1], pte[rm == 1], args.thr)
            n_eval += 1
            wins += all_ok
            rm_wins += rm_ok
            print(f"  {Y:>8} {len(tr):>9} {len(te):>7} {auc:>8.3f}   {all_line}")
            print(f"  {'':>8} {'':>9} {'':>7} {'':>8}   rm→ {rm_line}")
        print(f"\n  ÖZET {model_kind.upper()}: kapı genelde kazandığı yıl {wins}/{n_eval} | "
              f"removed-since'te kazandığı yıl {rm_wins}/{n_eval}")
        durable = n_eval > 0 and wins >= n_eval - 1 and rm_wins >= n_eval - 1
        print(f"  → {'DAYANIKLI (çoğu yıl + removed-since)' if durable else 'kenar YOK / dayanıklı değil (yıllar arası tutarsız)'}\n")


if __name__ == "__main__":
    main()
