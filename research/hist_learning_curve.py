"""
hist_learning_curve.py — Tur-94 pilotu: TARİHSEL 1m veriyle eğitmek CANLIDA işe yarıyor mu?

Soru (kullanıcı): canlı veri yerine son 2 (işe yararsa 10) yılın verisiyle eğitsek model iyileşir mi?
Tur-93'ün 282M modeli 16 bin canlı satırla eğitildi (basamağın alt sınırı 3,6 milyon) ve becerisi yoktu.

Tasarım (`capacity_data_sweep`in öğrenme eğrisi, ama DOĞRU test kümesiyle):
  eğitim : `build_hist_1m_dataset` satırları, CANLI DÖNEMDEN ÖNCE (canlı ilk gün − embargo);
           her noktada kesime EN YAKIN son n satır ("elimde n satır geçmiş olsaydı")
  test A : CANLI 1m eğitim seti (`build_live_dataset`) — modelin kullanılacağı asıl dağılım
  test B : aynı dönemin tarihsel satırları (tüm semboller, sabit tohumlu alt örneklem)
  test C : aynı dönemin tarihsel satırlarından yalnız CANLI semboller (B↔C farkı = evren, C↔A = etiket)
  ölçü   : AUC + Brier ↔ SABİT taban (eğitim taban oranı; ACI DERS #9) + BSS + gün-kümeli t
           (satır t'si örtüşen satırlarda şişer; Tur-53). Negatif t = model tabandan iyi.
Kart KAYDEDİLMEZ, canlıya hiçbir şey girmez — bu bir ÖLÇÜMdür (kayıt `train_neural --register`in işi).

Kullanım (repo kökünden):
    python -m research.hist_learning_curve --hist C:\\Users\\Kagan\\nihai_hist\\meta_dataset_hist_1m_2y.csv \
        --tier quant_transformer_moe_xl --device cuda --optimizer adafactor --amp --batch-size 128 \
        --json data/hist_learning_curve_moe_xl_2026-09-24.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nihai.console import ensure_utf8_stdout
from nihai.signals.meta_features import FEATURE_NAMES, ensure_feature_columns

CANLI_VARSAYILAN = "data/meta_dataset_live_1m_2026-09-23b.csv"


def yukle(path: str) -> tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray]:
    """(X float32, y, UTC zaman, sembol) — kronolojik sıralı. Karışık damga güvenli (BULGU-88-2)."""
    from research.train_meta import _chrono_key
    keep = {*FEATURE_NAMES, "label", "entry_time", "symbol"}
    df = pd.read_csv(path, usecols=lambda c: c in keep)
    df, _ = ensure_feature_columns(df)
    t = _chrono_key(df["entry_time"])
    df = df.assign(_t=t).dropna(subset=["_t"]).sort_values("_t", kind="stable").reset_index(drop=True)
    return (df[FEATURE_NAMES].to_numpy(dtype=np.float32), df["label"].to_numpy(dtype=float),
            df["_t"], df["symbol"].to_numpy())


def olc(y: np.ndarray, p: np.ndarray, taban: float, gun: np.ndarray) -> dict:
    """AUC + Brier ↔ sabit taban tahmincisi (eşleştirilmiş, gün-kümeli t)."""
    from nihai.ai.neural import evaluation_metrics
    from research.horizon_sweep import _cluster_t
    from research.train_meta import _auc
    d = (p - y) ** 2 - (taban - y) ** 2
    t, k = _cluster_t(d, gun)
    b, b0 = float(np.mean((p - y) ** 2)), float(np.mean((taban - y) ** 2))
    return {"n": int(len(y)), "taban_orani": round(float(y.mean()), 4), "auc": round(float(_auc(y, p)), 4),
            "brier": round(b, 5), "brier_sabit": round(b0, 5), "bss": round(1.0 - b / b0, 5),
            "t_gun": (round(t, 2) if t is not None else None), "gun": k,
            "ece": evaluation_metrics(y, p)["ece"], "p_std": round(float(np.std(p)), 4)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tarihsel 1m öğrenme eğrisi → canlı sette ölçüm")
    ap.add_argument("--hist", required=True)
    ap.add_argument("--live", default=CANLI_VARSAYILAN)
    ap.add_argument("--points", default="16000,64000,256000,1000000,all")
    ap.add_argument("--tier", default="deep_mlp")
    ap.add_argument("--embargo-days", type=int, default=1)
    ap.add_argument("--epoch-budget", type=int, default=400_000,
                    help="nokta başına max_epochs = budget/n (1..--max-epochs); büyük n'de 1 geçiş")
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--eval-hist-max", type=int, default=100_000)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"])
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", required=True)
    args = ap.parse_args(argv)

    from nihai.ai.neural import NeuralCapacity
    from research.capacity_data_sweep import _vram_sifirla, _vram_zirve

    Xh, yh, th, sh = yukle(args.hist)
    Xa, ya, ta, sa = yukle(args.live)
    canli_bas = ta.iloc[0].floor("D")
    kesim = canli_bas - pd.Timedelta(days=args.embargo_days)
    tr = np.flatnonzero((th < kesim).to_numpy())
    te_tum = np.flatnonzero((th >= canli_bas).to_numpy())
    te = te_tum
    if len(te) > args.eval_hist_max:
        te = np.sort(np.random.default_rng(args.seed).choice(te, args.eval_hist_max, replace=False))
    # test C: canlı sembollerin aynı dönemdeki tarihsel satırları — evren farkını etiket farkından ayırır
    te30 = te_tum[np.isin(sh[te_tum], np.unique(sa))]
    Xb, yb, Xc, yc = Xh[te], yh[te], Xh[te30], yh[te30]
    gun_a = ta.dt.strftime("%Y-%m-%d").to_numpy()
    gun_b = th.iloc[te].dt.strftime("%Y-%m-%d").to_numpy()
    gun_c = th.iloc[te30].dt.strftime("%Y-%m-%d").to_numpy()

    # Eğitimde sabit (σ≈0) kolon: tüm kümelerde sıfırla (neutralize_degenerate sözleşmesi, eğitimden).
    dej = np.flatnonzero(np.nanstd(Xh[tr], axis=0) < 1e-6)
    for M in (Xh, Xa, Xb, Xc):
        M[:, dej] = 0.0

    recete: dict = {"device": args.device, "optimizer_name": args.optimizer, "amp": args.amp}
    if args.batch_size:
        recete["batch_size"] = int(args.batch_size)
    noktalar = sorted({len(tr) if p.strip() == "all" else min(int(p), len(tr))
                       for p in args.points.split(",") if p.strip()})
    out: dict = {"hist": args.hist, "live": args.live, "tier": args.tier, "recete": recete,
                 "egitim_havuzu": int(len(tr)), "kesim_utc": str(kesim),
                 "egitim_araligi": [str(th.iloc[tr[0]]), str(th.iloc[tr[-1]])] if len(tr) else None,
                 "canli_araligi": [str(ta.iloc[0]), str(ta.iloc[-1])],
                 "sifirlanan_ozellik": [FEATURE_NAMES[i] for i in dej], "noktalar": []}
    print(f"eğitim havuzu {len(tr):,} (< {kesim.date()}) · test A canlı {len(ya):,} · "
          f"test B tarihsel {len(yb):,} · tier={args.tier} · noktalar {noktalar}", flush=True)

    for n in noktalar:
        idx = tr[-n:]
        ytr = yh[idx]
        ep = max(1, min(args.max_epochs, round(args.epoch_budget / n)))
        model = NeuralCapacity(args.tier, feature_names=list(FEATURE_NAMES), seed=args.seed)
        _vram_sifirla(str(args.device))
        t0 = time.perf_counter()
        model.fit(Xh[idx], ytr, max_epochs=ep, patience=args.patience, **recete)
        sure = time.perf_counter() - t0
        taban = float(ytr.mean())
        kayit = {"n": int(n), "max_epochs": ep, "egitim_sn": round(sure, 1),
                 "egitim_ilk": str(th.iloc[idx[0]]), "taban_egitim": round(taban, 4),
                 "param_count": model.param_count(), "zirve_vram_mb": _vram_zirve(str(model._device)),
                 "canli": olc(ya, model.predict_proba(Xa), taban, gun_a),
                 "tarihsel": olc(yb, model.predict_proba(Xb), taban, gun_b),
                 "tarihsel_canli_sembol": olc(yc, model.predict_proba(Xc), taban, gun_c)}
        out["noktalar"].append(kayit)
        tmp = Path(args.json).with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(args.json)
        a, b, c = kayit["canli"], kayit["tarihsel"], kayit["tarihsel_canli_sembol"]
        print(f"n={n:>9,} ep≤{ep:<2} {sure / 60:6.1f} dk | CANLI AUC {a['auc']:.4f} BSS {a['bss']:+.5f} "
              f"t {a['t_gun']} | TARİHSEL AUC {b['auc']:.4f} BSS {b['bss']:+.5f} t {b['t_gun']} | "
              f"TARİHSEL-CANLI-SEMBOL AUC {c['auc']:.4f} BSS {c['bss']:+.5f} t {c['t_gun']}", flush=True)
        del model
        gc.collect()
        if str(args.device) != "cpu":
            import torch
            torch.cuda.empty_cache()
    print(f"→ {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()
    raise SystemExit(main())
