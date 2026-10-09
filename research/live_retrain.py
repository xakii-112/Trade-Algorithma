"""
live_retrain.py — CANLI birikmiş veriden meta-modeli TEK KOMUTLA yeniden eğit + dürüst değerlendir.

Kapalı eğitim döngüsünün son halkası (ACI DERS #5'in canlı-kanıt yolu):
    prediction_tracker (canlı özellik+etiket birikimi)
      → build_live_dataset.build()  (survivorship-temiz, leak-free dataset)
      → bu modül: guardrail → time+symbol kapı simülasyonu → purged K-fold CV → VERDİKT
      → --promote (yalnız DAYANIKLI verdiktte): modeli data/'ya yaz (eskisi arşivlenir)
      → canlı süreçte reload_gate() / servis restart ile devreye girer.

DÜRÜSTLÜK SÖZLEŞMESİ (CLAUDE.md ACI DERSLER):
- #3: AUC değil KAPI karar verir; kenar HEM time HEM symbol bölmesinde çıkmalı.
- Purged+embargolu K-fold (López de Prado Ch.7) tek-bölme şansını eler.
- Örneklem guardrail'leri (satır/sembol/gün) altında hiçbir sonuç kenar sayılmaz,
  --promote reddedilir. Terfi asla otomatik değildir; bu komut elle çalıştırılır.

Kullanım (repo kökünden):
    python -m research.live_retrain                 # değerlendir, verdikt yaz
    python -m research.live_retrain --promote       # DAYANIKLI ise modeli kaydet
    python -m research.live_retrain --csv data/meta_dataset_live_2026-08-01.csv
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.signals.meta_features import FEATURE_NAMES, ensure_feature_columns
from research.build_live_dataset import build as build_live
from research.purged_cv import _build_model, null_auc_baseline, purged_folds, suggest_purge
from research.train_meta import (
    _auc,
    _chrono_key,
    _pf_exp,
    _provenance,
    chrono_sort,
    evaluate,
    save_model,
    split_symbol,
    split_time,
)

# Varsayılan guardrail'ler: bunların altında sonuç GÜRÜLTÜDÜR, verdikt INSUFFICIENT.
MIN_ROWS = 300
MIN_SYMBOLS = 10
MIN_SPAN_DAYS = 10.0


def _span_days(df: pd.DataFrame) -> float:
    """entry_time aralığı (gün). Parse edilemezse 0 (guardrail'i geçemez — dürüst taraf).
    Ek-bulgu #3: karışık naive/aware + 'T'li/boşluklu format utc=True ile hizalanır (aksi
    halde karışım tek satırı NaT yapıp _span_days=0 → sebepsiz kalıcı INSUFFICIENT üretirdi)."""
    if "entry_time" not in df.columns:
        return 0.0
    ts = _chrono_key(df["entry_time"]).dropna()
    if len(ts) < 2:
        return 0.0
    return float((ts.max() - ts.min()).total_seconds() / 86400.0)


def _purged_gate(df: pd.DataFrame, kind: str, *, folds: int = 5, embargo_frac: float = 0.01,
                 thr: float = 0.45, l2: float = 1.0) -> dict:
    """purged_cv'nin verdikt çekirdeği (permütasyon önemi olmadan — hız): fold başına
    OOS-AUC + kapı-vs-base + etiket-karıştırma NULL taban (1 tekrar, sabit). purged_cv.py
    ile AYNI standart: ort AUC>0.53, kapı ortalamada base'i geçer, en fazla 1 fold kaybeder,
    kapı E net pozitif VE gözlenen AUC null tabanından AYRIŞIR (KALEM 2). `thr` kapı eşiği
    çağıran tarafından geçilir (ek-bulgu #1: assess CV'yi TERFİ edilecek eşikte sınar).
    Dönüş: {mean_auc, pos_folds, n_folds, gate_e, base_e, durable, null_auc, separated}."""
    df = chrono_sort(df)   # ek-bulgu #3: string değil datetime sıralaması (karışık format)
    n = len(df)
    X = df[FEATURE_NAMES].to_numpy(float)
    y = df["label"].to_numpy(float)
    pnl = df["pnl"].to_numpy(float)
    embargo = int(round(n * embargo_frac))
    # P2-1: sol purge'ü etiket horizonundan (bars_held) türet; kolon yoksa embargo-simetrik.
    purge = suggest_purge(df)
    aucs: list[float] = []
    gate_es: list[tuple[float, float]] = []
    for tr_idx, te_idx in purged_folds(n, folds, embargo, purge=purge):
        ytr = y[tr_idx]
        if len(np.unique(ytr)) < 2:
            continue
        model = _build_model(kind, X[tr_idx], ytr, l2=l2)
        pte = model.predict_proba(X[te_idx])
        aucs.append(_auc(y[te_idx], pte))
        _, base_e, _ = _pf_exp(pnl[te_idx])
        kept = pnl[te_idx][pte >= thr]
        if len(kept) >= 5:
            _, g_e, _ = _pf_exp(kept)
            gate_es.append((g_e, base_e))
    if not aucs:
        return {"mean_auc": 0.5, "pos_folds": 0, "n_folds": 0, "gate_e": 0.0,
                "base_e": 0.0, "durable": False, "null_auc": None, "separated": None}
    mean_auc = float(np.mean(aucs))
    # KALEM 2: purged_cv ile aynı null taban (1 tekrar, sabit). Ayrışma yoksa AUC şans olabilir.
    nb = null_auc_baseline(kind, X, y, n, folds, embargo, purge=purge, l2=l2, reps=1)
    null_auc: float | None = None
    separated: bool | None = None
    if nb is not None:
        null_auc, null_std = nb
        separated = bool(mean_auc > null_auc + max(0.02, 2 * null_std))
    if gate_es:
        ge = np.array([g for g, _ in gate_es])
        be = np.array([b for _, b in gate_es])
        pos = int((ge > be).sum())
        durable = (mean_auc > 0.53 and ge.mean() > be.mean()
                   and pos >= len(ge) - 1 and ge.mean() > 0)
        if separated is not None:
            durable = durable and separated
        return {"mean_auc": mean_auc, "pos_folds": pos, "n_folds": len(ge),
                "gate_e": float(ge.mean()), "base_e": float(be.mean()), "durable": durable,
                "null_auc": null_auc, "separated": separated}
    return {"mean_auc": mean_auc, "pos_folds": 0, "n_folds": 0, "gate_e": 0.0,
            "base_e": 0.0, "durable": False, "null_auc": null_auc, "separated": separated}


def assess(df: pd.DataFrame, *, kinds: list[str], train_frac: float = 0.6,
           min_rows: int = MIN_ROWS, min_symbols: int = MIN_SYMBOLS,
           min_span_days: float = MIN_SPAN_DAYS, folds: int = 5,
           min_gate_keep: int = 50, verbose: bool = True) -> dict:
    """Dataset'i tam disiplinle değerlendir. Dönüş:
    {verdict: INSUFFICIENT|NO_EDGE|DURABLE, reason, durable_kind, threshold, results}.
    Test edilebilirlik için saf: dosya yazmaz, model kaydetmez."""
    n = len(df)
    n_sym = df["symbol"].nunique() if n else 0
    span = _span_days(df) if n else 0.0
    if n < min_rows or n_sym < min_symbols or span < min_span_days:
        return {"verdict": "INSUFFICIENT",
                "reason": f"örneklem guardrail: satır {n}<{min_rows} veya sembol "
                          f"{n_sym}<{min_symbols} veya süre {span:.1f}g<{min_span_days}g",
                "durable_kind": None, "threshold": None, "results": {}}

    results: dict = {}
    for kind in kinds:
        split_res: dict[str, dict | None] = {}
        for sp, splitter in (("time", split_time), ("symbol", split_symbol)):
            tr, te = splitter(df, train_frac)
            if len(tr) < 20 or len(te) < 20:
                split_res[sp] = None
                continue
            model = _build_model(kind, tr[FEATURE_NAMES].to_numpy(float),
                                 tr["label"].to_numpy(float), l2=1.0)
            if verbose:
                r = evaluate(model, tr, te, label=f"{sp} (canlı)")
            else:   # test modunda çıktı bastırma: evaluate print-ağır
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    r = evaluate(model, tr, te, label=sp)
            split_res[sp] = {"model": model, **r}
        # Ek-bulgu #1: CV'yi TERFİ edilecek eşikte sına. Terfi eşiği = time bölmesinin
        # (TRAIN'de seçilen) best_gate eşiği; sabit 0.45 kullanmak DURABLE'ın CV ayağını
        # terfi edilen eşikte hiç sınamamış bırakıyordu. best_gate yoksa kind zaten
        # dayanıklı olamaz → taban 0.45 zararsız.
        st_bg = split_res.get("time")
        gate_thr = st_bg["best_gate"][0] if (st_bg and st_bg.get("best_gate")) else 0.45
        cv = _purged_gate(df, kind, folds=folds, thr=gate_thr)
        if verbose:
            print(f"\n  [{kind}] purged {cv['n_folds']}-fold @eşik{gate_thr}: ort OOS-AUC "
                  f"{cv['mean_auc']:.3f} | kapı E {cv['gate_e']:+.3f} vs base {cv['base_e']:+.3f} | "
                  f"kazanılan fold {cv['pos_folds']}/{cv['n_folds']} | ayrışma {cv['separated']}")
        results[kind] = {"splits": split_res, "cv": cv}

    # DAYANIKLI = aynı model iki bölmede de kapı kenarı VE purged CV verdikti pozitif
    # VE her bölmedeki kapı kenarı yeterli örnekleme (>= min_gate_keep tutulan işlem)
    # dayanıyor. B1 (2026-07-09): eskiden bu asgari-örneklem şartı YOKTU → canlı --promote
    # avuç-dolusu işlemle (ör. time-split'te 6 işlem, PF 3.68 = şans) sahte kenarı ÜRETİME
    # yazabilirdi. train_meta --save aynı şartı zaten uyguluyordu (best_gate[4] >= MIN_GATE_KEEP);
    # üretim modelini YAZAN diğer yol (live_retrain) da artık aynı disiplinde.
    # best_gate = (thr, pf, e, wr, kept); [4] = tutulan işlem sayısı.
    durable_kinds: dict[str, float] = {}
    for kind, r in results.items():
        st, ss = r["splits"].get("time"), r["splits"].get("symbol")
        if (st and ss and st.get("best_gate") and ss.get("best_gate")
                and st["best_gate"][4] >= min_gate_keep
                and ss["best_gate"][4] >= min_gate_keep
                and r["cv"]["durable"]):
            durable_kinds[kind] = min(st["best_gate"][2], ss["best_gate"][2])
    if durable_kinds:
        kind = max(durable_kinds, key=lambda k: durable_kinds[k])
        thr = results[kind]["splits"]["time"]["best_gate"][0]
        return {"verdict": "DURABLE",
                "reason": f"{kind}: time+symbol kapı kenarı + purged CV pozitif",
                "durable_kind": kind, "threshold": thr, "results": results}
    return {"verdict": "NO_EDGE",
            "reason": "hiçbir model iki bölme + purged CV üçlüsünü birden geçemedi",
            "durable_kind": None, "threshold": None, "results": results}


def main() -> None:
    ap = argparse.ArgumentParser(description="Canlı veriden meta-modeli yeniden eğit + değerlendir")
    ap.add_argument("--csv", default=None,
                    help="hazır dataset CSV'si (vars: canlı loglardan taze build edilir)")
    ap.add_argument("--json", default=None, help="Terfi incelemesini JSON'a yaz")
    ap.add_argument("--model", default="both", choices=["logistic", "gbm", "both"])
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-gate-keep", type=int, default=50,
                    help="DAYANIKLI verdikt için her bölmede kapı kenarının dayanması "
                         "gereken asgari tutulan-işlem sayısı (altı = gürültü; train_meta "
                         "--save ile aynı disiplin)")
    ap.add_argument("--promote", action="store_true",
                    help="verdikt DAYANIKLI ise modeli data/'ya kaydet (eskisi arşivlenir)")
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        df, missing = ensure_feature_columns(df)
        if missing:
            print(f"UYARI: eski-şema dataset, {len(missing)} özellik nötr-dolduruldu: {missing}")
        src = args.csv
    else:
        df = build_live()
        src = "canlı loglar (build_live_dataset)"
    print(f"=== CANLI YENİDEN-EĞİTİM | kaynak: {src} | satır: {len(df)}")

    kinds = ["logistic", "gbm"] if args.model == "both" else [args.model]
    res = assess(df, kinds=kinds, min_rows=args.min_rows, folds=args.folds,
                 min_gate_keep=args.min_gate_keep)
    if args.json:
        import json
        from pathlib import Path

        from nihai.atomic_io import atomic_write_text
        serial = {**res, "promoted": False, "rows": len(df), "source": src,
                  "results": {k: {**v, "splits": {
                      s: {a: b for a, b in detail.items() if a != "model"} if detail else None
                      for s, detail in v["splits"].items()}} for k, v in res["results"].items()}}
        atomic_write_text(Path(args.json), json.dumps(serial, ensure_ascii=False, default=str))

    print(f"\n{'='*66}\nVERDİKT [{res['verdict']}]: {res['reason']}\n{'='*66}")
    if res["verdict"] != "DURABLE":
        if args.promote:
            print("--promote REDDEDİLDİ: dayanıklı kenar yok (ACI DERS #3/#4 — "
                  "META_GATE varsayılan kapalı kalır).")
        return
    kind, thr = res["durable_kind"], res["threshold"]
    print(f"Dayanıklı aday: {kind} (kapı eşiği {thr}). "
          "Yine de canlı gate_sim ile izlemeye devam (forward_harness --report).")
    if args.promote:
        # B5 (2026-07-09): model↔eşik TUTARLILIĞI. Eşik (thr) TIME alt-modelinin OOS
        # kapı taramasından gelir; bu yüzden TAM O alt-modeli deploy ederiz. Eskiden burada
        # TÜM-veride TAZE bir model eğitilip alt-modelin eşiğiyle kaydediliyordu → tam-veri
        # modelinin olasılık ölçeği farklıdır, eşik anlamını yitirir (sessiz uyuşmazlık).
        # OOS'ta ölçülen modeli deploy etmek dürüst + tutarlı; sembol bölmesi + purged CV
        # dayanıklılığı doğrular. (Kenar zaten yok — amaç doğruluk, kenar uydurmak değil.)
        model = res["results"][kind]["splits"]["time"]["model"]
        auc = res["results"][kind]["cv"]["mean_auc"]
        # KALEM 3: provenance. Canlı build ise dataset_sha256=None + source='live_logs';
        # --csv verilmişse dosya hash'lenir + source='csv'. ECE time alt-modelinden gelir.
        ece = res["results"][kind]["splits"]["time"].get("ece_test")
        n_cfg = 4 * len(kinds) * 2   # eşik × model × (time+symbol)
        prov = _provenance(dataset_path=(args.csv or src), n_rows=len(df), train_frac=0.6,
                           min_gate_keep=args.min_gate_keep, n_configs_tried=n_cfg,
                           calibrate=None, ece_test=ece)
        prov["source"] = "csv" if args.csv else "live_logs"
        path = save_model(model, trained_on=src, oos_auc=auc, gate_threshold=thr, provenance=prov)
        if args.json:
            serial.update(promoted=True, model_path=str(path))
            atomic_write_text(Path(args.json), json.dumps(serial, ensure_ascii=False, default=str))
        print(f"→ TERFİ: {path} yazıldı (öncekisi .bak-<tarih> olarak arşivlendi).")
        print("  Canlıda etkinleştirme: servis restart VEYA reload_gate(); "
              "META_GATE=true bilinçli karar ister (CLAUDE.md).")


if __name__ == "__main__":
    main()
