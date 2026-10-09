"""
train_pead_ml.py — PEAD KESİT-ML modelini eğit + survivorship-temiz OOS doğrula.

build_pead_dataset.py çıktısını (pead_ml_dataset_*.csv) tüketir. Fikir: birincil PEAD
sinyali (|surprise|≥2%) YÖN üretir; bu ikincil model her olay için "drift kâra dönecek mi?"
olasılığını tahmin edip düşük-olasılıklı olayları eler (meta-labeling; López de Prado).

Bu script train_meta.py / purged_cv.py'yi ÜÇ noktada genişletir:
  1. Genişletilmiş özellik seti: 19 generic + surprise + abs_surprise + cap_tier
     (dolar-hacim yüzdeliği). surprise/abs_surprise winsorize (aşırı-küçük payda → 100%+
     sürprizler gürültü; robust aralığa kırpılır).
  2. SURVIVORSHIP-BÖLMELİ kapı değerlendirmesi (§13.5 kapısı — train_meta'da YOK):
     kapı kenarı removed-since VE hayatta-kalan alt-kümelerinde AYRI raporlanır. Kenar
     yalnız hayatta-kalanlarda ise = survivorship artefaktı, ship EDİLMEZ.
  3. Purged+embargolu K-fold (zaman-serisi-dürüst OOS) genişletilmiş özelliklerle.

DİSİPLİN (CLAUDE.md ACI DERSLER §16): karar AUC değil KAPI SİMÜLASYONU. Kenar iddiası
için HEM time HEM symbol bölmesi + purged CV + removed-since'te de pozitif ŞART. Aksi =
"kenar YOK" (geçerli, dürüst sonuç). META_GATE deseni gibi ship yalnız dayanıklıysa.

    python -m research.train_pead_ml
    python -m research.train_pead_ml --model all --folds 5
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
from nihai.signals.meta_features import FEATURE_NAMES, ensure_feature_columns
from research.purged_cv import purged_folds, suggest_purge
from research.train_meta import (
    GBMModel,
    LogisticModel,
    XGBModel,
    _auc,
    _pf_exp,
    chrono_sort,
    split_symbol,
    split_time,
)

# Genişletilmiş kesit-özellik seti (PEAD'e özgü sinyaller generic'in ÜSTÜNE).
# sue_w = standardize sürpriz (ham surprise'ın daha öngörücü versiyonu — §19 teşhisi:
# model ham surprise'ı yok saymıştı, SUE ona daha temiz bir earnings-sinyali verir).
PEAD_EXTRA = ["surprise_w", "abs_surprise_w", "sue_w", "abs_sue_w", "cap_tier"]
PEAD_FEATURES = FEATURE_NAMES + PEAD_EXTRA


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorize surprise/sue + cap_tier (dolar-hacim yüzdeliği) ekle. Look-ahead-nötr
    (bar/zaman ekseninde): surprise/sue/dollar_vol_med zaten giriş-öncesi; buradaki
    dönüşümler yalnız ölçekleme, gelecekteki BAR'a bakmaz.

    ⚠️ BİLİNEN SINIR (Tur-11 denetimi, PLAUSIBLE, düzeltilMEDİ — bilinçli erteleme):
    winsorize sınırları (`quantile(0.01/0.99)`) ve `cap_tier` rank'i bu fonksiyon TÜM df
    (main()'de split'ten ÖNCE) üzerinde hesaplanıyor → train+test birleşik dağılımdan
    fit edilen ölçek, hafif TRANSDÜKTİF sızıntı (etiket sızıntısı DEĞİL; yalnız ölçekleme
    dağılımı). Doğru fix: winsorize/rank'i yalnız TRAIN diliminde fit edip TEST'e aynı
    sınırlarla uygulamak (evaluate() + run_purged() fold döngüsünde ayrı ayrı) — bu,
    prepare()'in tek-df imzasını fit/apply ikilisine ayırmayı gerektiren orta ölçekli bir
    refactor. Ertelendi çünkü: (a) bu script'in GÜNCEL verdikti zaten 'kenar YOK' (§19) —
    sızıntı varsa OOS-AUC'yi OLDUĞUNDAN YÜKSEK gösterir, yani mevcut 'kenar yok' hükmü bu
    sızıntıya rağmen duruyor = hüküm GÜVENDE; (b) düzeltme "kenar yok"u asla "kenar var"a
    çeviremez, yalnız daha muhafazakâr yapar. Yeni veriyle yeniden koşulursa ÖNCE bu
    düzeltilmeli. Bkz. DENETIM_BULGULARI_2026-07-09.md Tur-11 eki.

    B7 (2026-07-09) NaN-güvenliği: (1) generic özellik kolonları ensure_feature_columns ile
    şemaya yükseltilir + NaN'ları nötrlenir (eski-şema KeyError'ı + NaN sızıntısı önlenir);
    (2) winsorize NaN'ı KORUR → clip sonrası fillna(0.0) ŞART, yoksa eksik surprise/sue
    (ör. eşleşmeyen earnings, küçük payda) NaN olarak modele gider ve sessizce bozar."""
    df = df.copy()
    # (1) generic 19+ özellik: eksik kolon nötr eklenir, mevcut NaN'lar nötrle doldurulur.
    df, missing = ensure_feature_columns(df)
    if missing:
        print(f"UYARI: eski-şema PEAD dataset, {len(missing)} generic özellik nötr-dolduruldu: {missing}")
    # (2) surprise/sue sayısala zorla (eksik kolon → nötr 0), winsorize + fillna(0.0).
    for col in ("surprise", "sue"):
        df[col] = (pd.to_numeric(df[col], errors="coerce") if col in df.columns
                   else pd.Series(0.0, index=df.index))
    lo, hi = df["surprise"].quantile(0.01), df["surprise"].quantile(0.99)
    df["surprise_w"] = df["surprise"].clip(lo, hi).fillna(0.0)
    df["abs_surprise_w"] = df["surprise_w"].abs()
    slo, shi = df["sue"].quantile(0.01), df["sue"].quantile(0.99)
    df["sue_w"] = df["sue"].clip(slo, shi).fillna(0.0)
    df["abs_sue_w"] = df["sue_w"].abs()
    # cap_tier: evren-içi dolar-hacim yüzdeliği [0,1] (PIT market-cap ücretsiz yok → likidite vekili).
    dv_src = (pd.to_numeric(df["dollar_vol_med"], errors="coerce") if "dollar_vol_med" in df.columns
              else pd.Series(np.nan, index=df.index))
    df["cap_tier"] = dv_src.replace(0, np.nan).rank(pct=True).fillna(0.0)
    return df


def _build_model(kind: str, X, y, *, l2: float):
    if kind == "logistic":
        return LogisticModel.fit(X, y, l2=l2)
    if kind == "xgb":
        return XGBModel.fit(X, y)
    return GBMModel.fit(X, y)


def _subset_gate(pnl: np.ndarray, p: np.ndarray, thr: float) -> str:
    """Bir alt-küme için BASE vs kapı(≥thr) tek-satır özeti."""
    if len(pnl) == 0:
        return "(işlem yok)"
    b_pf, b_e, b_wr = _pf_exp(pnl)
    kept = pnl[p >= thr]
    if len(kept) < 5:
        return f"n={len(pnl):<4} BASE WR{b_wr:4.0f} PF{b_pf:4.2f} E{b_e:+7.0f} | kapı: az işlem"
    g_pf, g_e, g_wr = _pf_exp(kept)
    win = "✓" if (g_e > b_e and g_pf > 1.0) else "✗"
    return (f"n={len(pnl):<4} BASE WR{b_wr:4.0f} PF{b_pf:4.2f} E{b_e:+7.0f} | "
            f"kapı al{len(kept):<4} WR{g_wr:4.0f} PF{g_pf:4.2f} E{g_e:+7.0f} {win}")


def evaluate(model, tr: pd.DataFrame, te: pd.DataFrame, *, features: list[str],
             label: str, thr: float) -> dict:
    Xtr, ytr = tr[features].to_numpy(float), tr["label"].to_numpy(float)
    Xte, yte = te[features].to_numpy(float), te["label"].to_numpy(float)
    pnlte = te["pnl"].to_numpy(float)
    ptr, pte = model.predict_proba(Xtr), model.predict_proba(Xte)

    print(f"\n{'='*72}\n  MODEL={model.kind.upper()}  BÖLME={label}  "
          f"(eğitim {len(tr)} / test {len(te)})\n{'='*72}")
    print(f"  AUC eğitim {_auc(ytr, ptr):.3f} | AUC test (OOS) {_auc(yte, pte):.3f}")

    base_pf, base_e, base_wr = _pf_exp(pnlte)
    print(f"  {'eşik':>6} {'al':>6} {'%al':>5} {'WR%':>6} {'PF':>7} {'E':>10}")
    print(f"  {'BASE':>6} {len(pnlte):>6} {'100':>5} {base_wr:>6.1f} {base_pf:>7.2f} {base_e:>10.1f}")
    best = None
    for t in (0.45, 0.50, 0.55, 0.60):
        kept = pnlte[pte >= t]
        if len(kept) < 5:
            continue
        pf, e, wr = _pf_exp(kept)
        flag = "  ←" if (e > base_e and pf > base_pf and pf > 1.0) else ""
        print(f"  {t:>6.2f} {len(kept):>6} {100*len(kept)/len(pnlte):>4.0f}% "
              f"{wr:>6.1f} {pf:>7.2f} {e:>10.1f}{flag}")
        if e > base_e and pf > 1.0 and (best is None or e > best[2]):
            best = (t, pf, e, wr, len(kept))

    # --- SURVIVORSHIP BÖLMESİ (§13.5 kapısı) ---
    use_thr = best[0] if best else thr
    rm = te["removed_since"].to_numpy(int) if "removed_since" in te.columns else np.zeros(len(te), int)
    print(f"  --- survivorship (kapı eşik {use_thr:.2f}) ---")
    print(f"    removed-since : {_subset_gate(pnlte[rm == 1], pte[rm == 1], use_thr)}")
    print(f"    hayatta-kalan : {_subset_gate(pnlte[rm == 0], pte[rm == 0], use_thr)}")

    # removed-since'te kapı pozitif mi? (dürüstlük bayrağı)
    rm_pnl, rm_p = pnlte[rm == 1], pte[rm == 1]
    rm_ok = False
    if len(rm_pnl) >= 5:
        rk = rm_pnl[rm_p >= use_thr]
        if len(rk) >= 5:
            _, rme, _ = _pf_exp(rm_pnl)
            _, gke, _ = _pf_exp(rk)
            rm_ok = gke > rme and gke > 0
    return {"oos_auc": _auc(yte, pte), "best_gate": best, "removed_gate_ok": rm_ok}


def run_purged(df: pd.DataFrame, kind: str, *, features: list[str], folds: int,
               embargo_frac: float, thr: float, l2: float) -> None:
    # Tur-12: lexical sort_values yerine tz-güvenli chrono_sort (kardeşlerle tutarlı) — karışık
    # naive/aware VEYA 'T'-vs-boşluk entry_time formatı purged fold sırasını bozmasın.
    df = chrono_sort(df).reset_index(drop=True) if "entry_time" in df.columns else df
    n = len(df)
    embargo = int(round(n * embargo_frac))
    # P2-1: sol purge'ü etiket horizonundan (bars_held) türet; kolon yoksa embargo-simetrik.
    purge = suggest_purge(df)
    X = df[features].to_numpy(float)
    y = df["label"].to_numpy(float)
    pnl = df["pnl"].to_numpy(float)
    rm = df["removed_since"].to_numpy(int) if "removed_since" in df.columns else np.zeros(n, int)

    purge_txt = str(purge) if purge is not None else f"simetrik({embargo})"
    print(f"\n{'='*72}\n  MODEL={kind.upper()}  —  {folds}-fold PURGED CV "
          f"(embargo {embargo}, purge {purge_txt})\n{'='*72}")
    aucs, base_es = [], []
    gate_pairs: list[tuple[float, float]] = []   # (base_e, gate_e) — yalnız kapı çalışan foldlar (hizalı)
    rm_flags: list[bool] = []
    kept_tot = base_tot = 0
    imp: dict[str, list[float]] = {f: [] for f in features}
    for fi, (tr_i, te_i) in enumerate(purged_folds(n, folds, embargo, purge=purge)):
        if len(np.unique(y[tr_i])) < 2:
            print(f"  fold {fi}: eğitimde tek sınıf, atlandı")
            continue
        m = _build_model(kind, X[tr_i], y[tr_i], l2=l2)
        pte = m.predict_proba(X[te_i])
        pnlte, rmte = pnl[te_i], rm[te_i]
        auc = _auc(y[te_i], pte)
        b_pf, b_e, _ = _pf_exp(pnlte)
        kept = pnlte[pte >= thr]
        gate_ran = len(kept) >= 5
        g_pf, g_e, _ = _pf_exp(kept) if gate_ran else (0.0, 0.0, 0.0)
        aucs.append(auc)
        base_es.append(b_e)
        if gate_ran:
            gate_pairs.append((b_e, g_e))
        kept_tot += len(kept)
        base_tot += len(pnlte)
        # removed-since kapı pozitif mi bu fold'da? (survivorship kapısı)
        rk = pnlte[(pte >= thr) & (rmte == 1)]
        rm_base = pnlte[rmte == 1]
        rm_ok = (len(rk) >= 5 and len(rm_base) >= 5
                 and _pf_exp(rk)[1] > _pf_exp(rm_base)[1] and _pf_exp(rk)[1] > 0)
        rm_flags.append(bool(rm_ok))
        flag = "  ←" if (gate_ran and g_e > b_e and g_pf > 1.0) else ""
        print(f"  fold {fi}: OOS-AUC {auc:.3f} | base E {b_e:+7.1f} PF {b_pf:4.2f} | "
              f"kapı(≥{thr}) al {len(kept):>4} E {g_e:+7.1f} PF {g_pf:4.2f} | rm-kapı {'✓' if rm_ok else '✗'}{flag}")
        rng = np.random.default_rng(fi)
        for j, f in enumerate(features):
            Xp = X[te_i].copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            imp[f].append(auc - _auc(y[te_i], m.predict_proba(Xp)))

    if not aucs:
        return
    a = np.array(aucs)
    print(f"\n  ORTALAMA OOS-AUC: {a.mean():.3f} ± {a.std():.3f} (min {a.min():.3f} / max {a.max():.3f})")
    durable = False
    if gate_pairs:
        gb = np.array([b for b, _ in gate_pairs])
        ge = np.array([g for _, g in gate_pairs])
        pos = int((ge > gb).sum())
        rm_pos = int(sum(rm_flags))
        print(f"  KAPI vs BASE: kapı E ort {ge.mean():+.1f} | base E ort {gb.mean():+.1f} | "
              f"kapı kazandığı fold {pos}/{len(ge)} | tutma %{100*kept_tot/base_tot:.0f}")
        print(f"  SURVIVORSHIP: removed-since'te kapı pozitif fold {rm_pos}/{len(rm_flags)} "
              f"(kenar gerçekse burada da kazanmalı)")
        durable = (a.mean() > 0.53 and ge.mean() > gb.mean() and pos >= len(ge) - 1
                   and ge.mean() > 0 and rm_pos >= len(rm_flags) - 1)
    print(f"  → VERDİKT: {'DAYANIKLI (yine de forward/paper ile doğrula)' if durable else 'kenar YOK / dayanıklı değil → META_GATE kapalı kalmalı'}")
    ranked = sorted(((np.mean(v) if v else 0.0, f) for f, v in imp.items()), reverse=True)
    print("  En etkili özellikler (OOS-AUC düşüşü):  " + "  ".join(f"{f}={d:+.3f}" for d, f in ranked[:8]))


def main():
    ap = argparse.ArgumentParser(description="PEAD kesit-ML eğit + survivorship-temiz OOS doğrula")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--model", default="both", choices=["logistic", "gbm", "xgb", "both", "all"])
    ap.add_argument("--split", default="both", choices=["time", "symbol", "both"])
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo", type=float, default=0.01)
    ap.add_argument("--thr", type=float, default=0.55)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--no-cv", action="store_true", help="purged CV'yi atla (yalnız time/symbol bölmesi)")
    args = ap.parse_args()

    csv_path = args.csv
    if not csv_path:
        cands = sorted(glob.glob(str(DATA_DIR / "pead_ml_dataset_*.csv")), key=os.path.getsize)
        if not cands:
            print("Dataset bulunamadı. Önce: python -m research.build_pead_dataset")
            return
        csv_path = cands[-1]
    df = prepare(pd.read_csv(csv_path))
    rm_n = int(df["removed_since"].sum()) if "removed_since" in df.columns else 0
    print(f"Dataset: {csv_path}")
    print(f"Olay: {len(df)} | sembol: {df['symbol'].nunique()} | genel kazanma %{100*df['label'].mean():.1f} "
          f"| removed-since {rm_n} ({100*rm_n/max(len(df),1):.0f}%)")
    print(f"Özellikler: {len(PEAD_FEATURES)} (generic 19 + {PEAD_EXTRA})")

    models = {"both": ["logistic", "gbm"], "all": ["logistic", "gbm", "xgb"]}.get(args.model, [args.model])
    splits = ["time", "symbol"] if args.split == "both" else [args.split]

    results = {}
    for sp in splits:
        tr, te = (split_time(df, args.train_frac) if sp == "time"
                  else split_symbol(df, args.train_frac))
        for mk in models:
            m = _build_model(mk, tr[PEAD_FEATURES].to_numpy(float), tr["label"].to_numpy(float), l2=args.l2)
            results[(mk, sp)] = evaluate(m, tr, te, features=PEAD_FEATURES, label=sp, thr=args.thr)

    if not args.no_cv:
        for mk in models:
            run_purged(df, mk, features=PEAD_FEATURES, folds=args.folds,
                       embargo_frac=args.embargo, thr=args.thr, l2=args.l2)

    print(f"\n{'='*72}\nÖZET — kenar dayanıklı SAYILIR ancak: her iki bölme (←) + purged CV DAYANIKLI")
    print(f"        + removed-since'te de kapı pozitif. Aksi = ship etme (§16).\n{'='*72}")
    for (mk, sp), r in results.items():
        g = r["best_gate"]
        gate = (f"eşik {g[0]:.2f}→PF {g[1]:.2f} E{g[2]:+.0f} (al {g[4]})" if g else "kenar YOK")
        rmok = "removed✓" if r["removed_gate_ok"] else "removed✗"
        print(f"  {mk:<9} {sp:<7} OOS-AUC {r['oos_auc']:.3f} | {gate} | {rmok}")


if __name__ == "__main__":
    main()
