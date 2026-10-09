"""
purged_cv.py — Purged + embargolu K-fold CV ile meta-labeling modelini DÜRÜST değerlendir.

Neden (López de Prado, "Advances in Financial ML", Ch.7): tek 60/40 zaman-bölmesi tek bir
OOS noktası verir → şanslı/şanssız bölmeye karşı kırılgan. K-fold CV birden çok OOS penceresi
verir, AMA finansal seride satırlar zaman-bağımlı: bir foldun eğitimi test foldunun hemen
bitişiğindeki örnekleri içerirse SIZINTI olur. Çözüm:
  - PURGE: test penceresine zaman olarak değen eğitim örneklerini at.
  - EMBARGO: test penceresinden hemen SONRAKİ küçük bir bandı da eğitimden çıkar (otokorelasyon
    sinyalin test sınırını aşmasını engeller).

Bu, train_meta.py'nin tek-bölme kapı simülasyonunu TAMAMLAR: aynı "olasılık-eşiği filtresi
expectancy/PF'yi BASE'in üstüne çıkarıyor mu?" sorusunu K bağımsız OOS penceresinde sorar.
Kenar yalnız 1-2 foldda görünüp ortalamada kayboluyorsa = dayanıklı değil (train_meta'daki
EDGE_FINDINGS standardının zaman-serisi-doğru versiyonu).

Ayrıca PERMÜTASYON ÖNEMİ: her özelliği OOS'ta karıştırıp AUC düşüşünü ölçer → model GERÇEKTEN
neye bakıyor? (Sızıntı özellikleri burada anormal yüksek çıkar — teşhis aracı.)

    python -m research.purged_cv                          # en büyük dataset, logistic+gbm
    python -m research.purged_cv --model all --folds 5 --embargo 0.01
    python -m research.purged_cv --csv data/meta_dataset_pit_sp500_event_breakout_1d_2022-01-01.csv
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

# train_meta.py'deki model sınıflarını ve metrikleri YENİDEN KULLAN (tek kaynak).
from research.train_meta import (  # noqa: E402
    GBMModel,
    LogisticModel,
    XGBModel,
    _auc,
    _pf_exp,
    chrono_sort,
)


def _build_model(kind: str, X: np.ndarray, y: np.ndarray, *, l2: float):
    if kind == "logistic":
        return LogisticModel.fit(X, y, l2=l2)
    if kind == "xgb":
        return XGBModel.fit(X, y)
    return GBMModel.fit(X, y)


def purged_folds(n: int, k: int, embargo: int, *, purge: int | None = None):
    """Zaman-sıralı [0..n) indekslerini k ardışık test bloğuna böl. Her test bloğu için
    eğitim = (blok - purge) ÖNCESİ ve (blok + embargo) SONRASI. Yield: (train_idx, test_idx).

    İKİ TARAFLI temizlik (López de Prado Ch.7):
      - SAĞ EMBARGO (embargo): test'ten hemen SONRAKİ bant eğitimden çıkar (otokorelasyon
        sinyalin test sınırını geriye aşmasını engeller).
      - SOL PURGE (purge): test'ten hemen ÖNCEKİ bant da eğitimden çıkar. B2 (2026-07-09)
        DÜZELTMESİ: eskiden sol taraf 'test'ten önce → güvenli' varsayılıp TÜM `[0, lo)`
        eğitime alınıyordu. AMA finansal etiket ileriye bakar (giriş t → sonuç t+horizon):
        test bloğuna BİTİŞİK bir eğitim satırının etiket penceresi test dönemine TAŞAR →
        gelecek-bilgisi eğitime sızar, OOS-AUC şişer. Sol purge bu sızıntıyı keser (kenarı
        UYDURMAZ; yalnız sahte şişkinliği kaldırır — sonuç daha katı/dürüst).
      purge=None → simetrik (embargo kadar). Etiket horizonu biliniyorsa purge>=horizon geç."""
    left_purge = embargo if purge is None else max(int(purge), 0)
    bounds = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        lo, hi = bounds[i], bounds[i + 1]
        test_idx = np.arange(lo, hi)
        left = np.arange(0, max(lo - left_purge, 0))  # SOL PURGE: etiket penceresi test'e taşmasın
        # SAĞ tarafı da horizon-purge et (Tur-12): finansal etiket İLERİYE bakar → test bloğunun
        # HEMEN SONRASINDAKİ eğitim satırlarının etiket penceresi test dönemiyle ÖRTÜŞÜR (López de
        # Prado iki tarafı da horizon kadar purge eder, üstüne embargo). B2 yalnız SOL'u horizon-aware
        # yapmıştı; sağ ham embargo kalınca kümeli/çok-sembollü veride sağ AZ-purge edilip OOS-AUC'yi
        # şişirebiliyordu (kenar UYDURMAZ ama sahte-DAYANIKLI verdikt riski). max(embargo, sol-purge).
        right = np.arange(min(hi + max(embargo, left_purge), n), n)
        train_idx = np.concatenate([left, right])
        yield train_idx, test_idx


def suggest_purge(df: pd.DataFrame, *, q: float = 0.95, cap_frac: float = 1 / 3) -> int | None:
    """P2-1: SOL PURGE satır sayısını dataset'in kendi `bars_held`'inden türet.

    purged_folds SATIR indeksiyle çalışır ama etiket horizonu BAR cinsindendir; çok-sembollü
    event dataset'lerinde aynı bara birden çok satır düşer. Dönüşüm satır/bar yoğunluğuyla
    yapılır ve hep FAZLA-purge yönüne yuvarlanır — sızıntı yerine biraz eğitim verisi feda
    etmek dürüst taraftır.
      - horizon = q95(bars_held): tek aykırı işlem purge'ü şişirmesin (max değil).
      - rows_per_bar (ek-bulgu #2): KÜRESEL ortalama (n/benzersiz-bar) değil, bar başına
        satır sayısının q95'i (en az küresel ortalama). Kümeli veride (satırların çoğu birkaç
        yoğun barda) ortalama YEREL yoğunluğu hafife alıp AZ-purge ederdi (anti-muhafazakâr);
        yoğun barları gören yüksek-quantile fazla-purge yönünü korur.
      - bars_held yok/boş/≤0 → None (çağıran embargo-simetrik varsayılana düşer).
      - tavan cap_frac*n: dejenere fold önlemi (gerçekçi horizonların çok üstü; tavana
        çarpan bars_held zaten şüpheli veridir, uyarıyı çağıran verir).
    """
    if "bars_held" not in df.columns or len(df) == 0:
        return None
    bh = pd.to_numeric(df["bars_held"], errors="coerce").dropna()
    bh = bh[bh > 0]
    if bh.empty:
        return None
    horizon = int(np.ceil(float(bh.quantile(q))))
    n = len(df)
    rows_per_bar = 1.0
    if "entry_time" in df.columns:
        vc = df["entry_time"].astype(str).value_counts()
        if len(vc) > 0:
            # q95 yoğun barları görür; küresel ortalama tabanı korunur (ikisinin büyüğü).
            rows_per_bar = max(1.0, float(vc.quantile(0.95)), n / len(vc))
    purge = int(np.ceil(horizon * rows_per_bar))
    return min(purge, max(1, int(n * cap_frac)))


def permutation_importance(model, X: np.ndarray, y: np.ndarray, *, base_auc: float,
                           seed: int = 0) -> dict[str, float]:
    """Her özellik sütununu karıştırıp OOS-AUC düşüşünü ölç. Büyük düşüş = model o özelliğe
    güveniyor. Anormal büyük düşüş + mantıksız özellik = sızıntı şüphesi."""
    rng = np.random.default_rng(seed)
    drops: dict[str, float] = {}
    for j, name in enumerate(FEATURE_NAMES):
        # D3 (Tur-10): özellik başına tam X.copy() yerine tek sütunu yerinde permüte edip
        # GERİ YÜKLE (bit-exact: rng tüketimi ve değerler aynı; predict_proba X'i değiştirmez;
        # finally → istisna yolunda bile çağıranın X'i bozulmaz).
        orig = X[:, j].copy()
        try:
            X[:, j] = rng.permutation(orig)
            drops[name] = base_auc - _auc(y, model.predict_proba(X))
        finally:
            X[:, j] = orig
    return drops


def null_auc_baseline(kind: str, X_all: np.ndarray, y_all: np.ndarray, n: int, folds: int,
                      embargo: int, *, purge: int | None = None, l2: float = 1.0,
                      reps: int = 1) -> tuple[float, float] | None:
    """Etiket-karıştırma NULL taban (KALEM 2): y'yi permüte edip AYNI purged_folds döngüsüyle
    (permütasyon-önem ve kapı simülasyonu YOK) fold OOS-AUC ortalamasını hesapla. Gerçek
    sinyal/sızıntı yoksa gözlenen AUC bu tabandan AYRIŞMAMALI (kenar-yok ile tutarlı).

    rep başına default_rng(1000+rep) ile bağımsız permütasyon; dönüş (ort, std) rep'ler
    üstünden (reps=1 → std=0 → çağıran marj=max(0.02, 2·std) kullanır). Hiç fold
    değerlenemezse (tek-sınıf) None."""
    rep_means: list[float] = []
    for rep in range(reps):
        rng = np.random.default_rng(1000 + rep)
        y_perm = rng.permutation(y_all)
        aucs: list[float] = []
        for tr_idx, te_idx in purged_folds(n, folds, embargo, purge=purge):
            ytr = y_perm[tr_idx]
            if len(np.unique(ytr)) < 2:
                continue
            model = _build_model(kind, X_all[tr_idx], ytr, l2=l2)
            aucs.append(_auc(y_perm[te_idx], model.predict_proba(X_all[te_idx])))
        if aucs:
            rep_means.append(float(np.mean(aucs)))
    if not rep_means:
        return None
    return float(np.mean(rep_means)), float(np.std(rep_means))


def main():
    ap = argparse.ArgumentParser(description="Purged+embargolu K-fold CV ile meta-labeling doğrula")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--json", default=None, help="Doğrulama ve özellik önemini JSON'a yaz")
    ap.add_argument("--model", default="both", choices=["logistic", "gbm", "xgb", "both", "all"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo", type=float, default=0.01,
                    help="test'ten sonra eğitimden çıkarılan bant (toplam satırın oranı)")
    ap.add_argument("--purge", type=int, default=None,
                    help="sol purge (satır); verilmezse bars_held'den otomatik türetilir (P2-1)")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--thr", type=float, default=0.55, help="kapı simülasyonu olasılık eşiği")
    ap.add_argument("--null-reps", type=int, default=5,
                    help="etiket-karıştırma NULL taban tekrar sayısı (0=atla). Gözlenen AUC "
                         "bu tabandan ayrışmıyorsa kenar-yok ile tutarlı (KALEM 2)")
    args = ap.parse_args()

    csv_path = args.csv
    if not csv_path:
        cands = sorted(glob.glob(str(DATA_DIR / "meta_dataset_*.csv")), key=os.path.getsize)
        if not cands:
            print("Dataset bulunamadı. Önce research/build_meta_dataset.py çalıştır.")
            return
        csv_path = cands[-1]
    df = pd.read_csv(csv_path)
    df, missing = ensure_feature_columns(df)
    if missing:
        print(f"UYARI: eski-şema dataset, {len(missing)} özellik nötr-dolduruldu: {missing}")
    df = chrono_sort(df)   # ek-bulgu #3: string değil datetime sıralaması (karışık format)
    n = len(df)
    embargo = int(round(n * args.embargo))
    # P2-1: sol purge'ü etiket horizonundan (bars_held) otomatik türet; --purge ile ezilebilir.
    purge = args.purge if args.purge is not None else suggest_purge(df)
    purge_txt = str(purge) if purge is not None else f"simetrik({embargo})"
    print(f"Dataset: {csv_path}")
    print(f"İşlem: {n} | sembol: {df['symbol'].nunique()} | kazanma %{100*df['label'].mean():.1f} | "
          f"fold={args.folds} embargo={embargo} purge={purge_txt} satır eşik={args.thr}")

    X_all = df[FEATURE_NAMES].to_numpy(float)
    y_all = df["label"].to_numpy(float)
    pnl_all = df["pnl"].to_numpy(float)

    models = {"both": ["logistic", "gbm"], "all": ["logistic", "gbm", "xgb"]}.get(
        args.model, [args.model])
    report = {"dataset": csv_path, "rows": n, "folds": args.folds, "embargo": embargo,
              "purge": purge, "models": {}}

    for kind in models:
        print(f"\n{'='*68}\n  MODEL = {kind.upper()}  —  {args.folds}-fold purged CV\n{'='*68}")
        fold_aucs, fold_base_e = [], []
        # (kapı E, base E) ÇİFT olarak saklanır: kapı yalnız kept>=5 foldlarda ölçülür;
        # ayrı listelerde base HER foldda birikince ge>be karşılaştırması yanlış fold
        # çiftlerini kıyaslar ya da boyut uyuşmazlığıyla çöker (BULGU-12).
        fold_gate_pairs: list[tuple[float, float]] = []
        kept_n_tot = base_n_tot = 0
        imp_accum: dict[str, list[float]] = {f: [] for f in FEATURE_NAMES}
        for fi, (tr_idx, te_idx) in enumerate(purged_folds(n, args.folds, embargo, purge=purge)):
            ytr = y_all[tr_idx]
            if len(np.unique(ytr)) < 2:
                print(f"  fold {fi}: eğitimde tek sınıf, atlandı")
                continue
            model = _build_model(kind, X_all[tr_idx], ytr, l2=args.l2)
            Xte, yte, pnlte = X_all[te_idx], y_all[te_idx], pnl_all[te_idx]
            pte = model.predict_proba(Xte)
            auc = _auc(yte, pte)
            base_pf, base_e, base_wr = _pf_exp(pnlte)
            kept = pnlte[pte >= args.thr]
            g_pf, g_e, g_wr = _pf_exp(kept) if len(kept) >= 5 else (0.0, 0.0, 0.0)
            fold_aucs.append(auc)
            fold_base_e.append(base_e)
            if len(kept) >= 5:
                fold_gate_pairs.append((g_e, base_e))
            kept_n_tot += len(kept)
            base_n_tot += len(pnlte)
            flag = "  ←" if (len(kept) >= 5 and g_e > base_e and g_pf > 1.0) else ""
            print(f"  fold {fi}: OOS-AUC {auc:.3f} | base E {base_e:+7.1f} PF {base_pf:4.2f} "
                  f"| kapı(≥{args.thr}) al {len(kept):>4} E {g_e:+7.1f} PF {g_pf:4.2f}{flag}")
            for f, d in permutation_importance(model, Xte, yte, base_auc=auc, seed=fi).items():
                imp_accum[f].append(d)

        if not fold_aucs:
            continue
        auc_arr = np.array(fold_aucs)
        print(f"\n  ORTALAMA OOS-AUC: {auc_arr.mean():.3f} ± {auc_arr.std():.3f}  "
              f"(min {auc_arr.min():.3f} / max {auc_arr.max():.3f})")

        # KALEM 2: etiket-karıştırma NULL taban. Gözlenen AUC bu tabandan ayrışmıyorsa
        # kenar-yok ile tutarlı; ayrışıyorsa gerçek sinyal VEYA sızıntı (perm önemine bak).
        separated: bool | None = None
        if args.null_reps > 0:
            nb = null_auc_baseline(kind, X_all, y_all, n, args.folds, embargo,
                                   purge=purge, l2=args.l2, reps=args.null_reps)
            if nb is not None:
                null_mean, null_std = nb
                margin = max(0.02, 2 * null_std)
                separated = bool(auc_arr.mean() > null_mean + margin)
                print(f"  NULL taban (etiket-karıştırma, {args.null_reps} tekrar): "
                      f"ort {null_mean:.3f} ± {null_std:.3f}  (marj {margin:.3f})")
                print("  → " + ("gözlenen AUC null tabanından AYRIŞIYOR — gerçek sinyal VEYA "
                                "sızıntı; permütasyon önemine bak" if separated else
                                "gözlenen AUC null tabanından AYRIŞMIYOR (kenar-yok ile tutarlı)"))

        if fold_gate_pairs:
            ge = np.array([g for g, _ in fold_gate_pairs])
            be = np.array([b for _, b in fold_gate_pairs])
            pos_folds = int((ge > be).sum())
            print(f"  KAPI vs BASE: kapı E ort {ge.mean():+.1f} | base E ort {be.mean():+.1f} | "
                  f"kapı kazandığı fold {pos_folds}/{len(ge)} | tutma %{100*kept_n_tot/base_n_tot:.0f}")
            durable = (auc_arr.mean() > 0.53 and ge.mean() > be.mean() and pos_folds >= len(ge) - 1
                       and ge.mean() > 0)
            # KALEM 2: null hesaplandıysa AYRIŞMA da şart (AUC şans eseri yüksek olabilir).
            if separated is not None:
                durable = durable and separated
            print(f"  → VERDİKT: {'DAYANIKLI kenar sinyali (yine de forward/paper ile doğrula)' if durable else 'kenar YOK / dayanıklı değil (META_GATE kapalı kalmalı)'}")
        else:
            print("  → VERDİKT: eşiği geçen yeterli işlem yok → kenar YOK")

        # En etkili 8 özellik (ortalama OOS-AUC düşüşü)
        ranked = sorted(((np.mean(v) if v else 0.0, f) for f, v in imp_accum.items()), reverse=True)
        report["models"][kind] = {
            "auc": float(auc_arr.mean()), "auc_std": float(auc_arr.std()),
            "fold_aucs": fold_aucs, "null_separated": separated,
            "gate_pairs": fold_gate_pairs, "kept": kept_n_tot, "total": base_n_tot,
            "feature_importance": [{"feature": f, "auc_drop": float(d)} for d, f in ranked],
        }
        print("  En etkili özellikler (OOS-AUC düşüşü):  " +
              "  ".join(f"{f}={d:+.3f}" for d, f in ranked[:8]))
        # KALEM 2: sızıntı bayrağı — tek özellik AUC'yi tek başına taşıyorsa look-ahead şüphesi.
        if ranked and ranked[0][0] >= 0.08:
            top_drop, top_feat = ranked[0]
            print(f"  SIZINTI ŞÜPHESİ: '{top_feat}' AUC'yi tek başına taşıyor "
                  f"(+{top_drop:.3f}) — look-ahead/etiket-koreli özellik denetle.")
    if args.json:
        import json
        from pathlib import Path

        from nihai.atomic_io import atomic_write_text
        atomic_write_text(Path(args.json), json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
