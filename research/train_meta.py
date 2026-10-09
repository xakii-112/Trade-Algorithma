"""
train_meta.py — Meta-labeling modeli eğit + GERÇEK out-of-sample doğrula.

İki model: bağımlılıksız numpy lojistik (production'da hafif) VE sklearn
GradientBoosting (doğrusal-olmayan, genelde daha güçlü). İki bölme stratejisi:
- time   : işlemleri zamana sırala, son %X test (gelecek işlemler)
- symbol : sembollerin %X'i TAMAMEN test (hiç görülmemiş hisseler → genelleme sınavı)

EDGE_FINDINGS standardı: asıl soru AUC değil, KAPI SİMÜLASYONU — modeli 'düşük
olasılıkları ele' filtresi olarak kullanınca görülmemiş işlemlerde expectancy/PF
gerçekten BASE'in üstüne çıkıyor mu? İki bölmede de çıkmalı (dayanıklı kenar).

    python train_meta.py --model both --split time
    python train_meta.py --model gbm  --split symbol --save
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pickle
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.config import DATA_DIR, ROOT
from nihai.signals.meta_features import FEATURE_NAMES, ensure_feature_columns


# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------
def _auc(y: np.ndarray, p: np.ndarray) -> float:
    pos_n = int((y == 1).sum())
    neg_n = int((y == 0).sum())
    if pos_n == 0 or neg_n == 0:
        return 0.5
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    # D2 (Tur-10): saf-Python rank döngüsü → cumsum ile vektörize. Bit-exact: her eleman
    # eskisi gibi int + (c+1)/2.0 float64 toplamı (parite: test_perf_parity_round10).
    cum = np.concatenate(([0], np.cumsum(counts)[:-1]))
    avg_rank = cum + (counts + 1) / 2.0
    ranks = avg_rank[inv]
    sum_pos = ranks[y == 1].sum()
    return (sum_pos - pos_n * (pos_n + 1) / 2.0) / (pos_n * neg_n)


def _pf_exp(pnls: np.ndarray) -> tuple[float, float, float]:
    if len(pnls) == 0:
        return 0.0, 0.0, 0.0
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gl = abs(losses.sum())
    pf = (wins.sum() / gl) if gl > 0 else float("inf")
    return pf, float(pnls.mean()), float(100 * len(wins) / len(pnls))


def _ece(y: np.ndarray, p: np.ndarray, *, bins: int = 10) -> float:
    """Beklenen Kalibrasyon Hatası (ECE): olasılıkları `bins` eşit-genişlikli kovaya böl;
    her kovada |ort tahmin − gerçek frekans| kova-ağırlığıyla topla. 0'a yakın = kalibre
    (model 0.6 dediğinde canlıda ~%60 kazanıyor). prob_sizing güvenilirliğinin ölçüsü —
    kenar İDDİASI DEĞİL, olasılığın ne kadar dürüst okunabileceğinin tanısıdır."""
    n = len(y)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        # son kova [lo,1.0] kapalı; diğerleri [lo,hi) — her tahmin tam bir kovaya düşer.
        mask = (p >= lo) & (p <= hi) if b == bins - 1 else (p >= lo) & (p < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return ece


def _chrono_key(s: pd.Series) -> pd.Series:
    """entry_time'ı kronolojik sıralama/ölçüm için datetime'a çevir. Karışık naive/aware
    ve 'T'li/boşluklu format güvenli (utc=True hepsini UTC'ye hizalar). Parse edilemeyen
    → NaT (sıralamada sona, guardrail'de dürüstçe düşer). Denetim ek-bulgu #3: STRING
    sıralama karışık formatta kronolojiyi bozardı (tz-offset'li satırlar yanlış yere gider);
    datetime anahtarı tek doğru sıralamadır."""
    return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")


def chrono_sort(df: pd.DataFrame) -> pd.DataFrame:
    """df'i entry_time'a göre KRONOLOJİK sırala (karışık-format güvenli, bkz _chrono_key).
    entry_time yoksa df aynen döner. Orijinal entry_time kolonu (string) korunur — yalnız
    satır sırası değişir. train_meta/purged_cv/live_retrain aynı sıralamayı paylaşır."""
    if "entry_time" not in df.columns:
        return df
    return (df.assign(_ord=_chrono_key(df["entry_time"]))
              .sort_values("_ord", kind="stable")
              .drop(columns="_ord").reset_index(drop=True))


def _retention_sweep(pte: np.ndarray, pnlte: np.ndarray) -> None:
    """Precision↔return ödünleşim eğrisi (kullanıcı sorusunun doğrudan cevabı):
    sinyalleri kazanma-olasılığına göre sırala, en yüksek %X'i tut. Seçicilik
    arttıkça WR (isabet) yükseliyor mu — ve toplam kâr (return) ne kadar korunuyor?

    'Aynı/yakın kâr oranıyla isabeti artır' = retain% yüksek kalırken WR/PF yükselen
    satırı bul. WR yükselip retain% çökerse (az ama küçük kazanç) = §7 tuzağı."""
    base_total = float(pnlte.sum())
    base_pf, base_e, base_wr = _pf_exp(pnlte)
    print("\n  --- PRECISION↔RETURN taraması (olasılık-sırALI top-%X tut) ---")
    print(f"  {'tut%':>5} {'işlem':>6} {'eşik~':>6} {'WR%':>6} {'PF':>6} "
          f"{'E':>10} {'toplam-$':>11}")
    order = np.argsort(-pte)             # yüksek olasılık önce
    pnl_sorted = pnlte[order]
    p_sorted = pte[order]
    n = len(pnl_sorted)
    print(f"  {'100':>5} {n:>6} {p_sorted[-1]:>6.2f} {base_wr:>6.1f} "
          f"{base_pf:>6.2f} {base_e:>10.1f} {base_total:>11.0f}")
    for frac in (0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20):
        k = int(round(n * frac))
        if k < 5:
            continue
        kept = pnl_sorted[:k]
        thr_at = float(p_sorted[k - 1])     # bu retention'ı veren olasılık eşiği
        pf, e, wr = _pf_exp(kept)
        # ideal (base zararlı olabilir → mutlak ölç): WR↑ + PF↑ + E↑ + net pozitif
        flag = "  ←" if (wr > base_wr and pf > base_pf and e > base_e and e > 0) else ""
        print(f"  {100*frac:>5.0f} {k:>6} {thr_at:>6.2f} {wr:>6.1f} {pf:>6.2f} "
              f"{e:>10.1f} {kept.sum():>11.0f}{flag}")
    print("  (← = WR↑ + PF↑ + E↑ + net pozitif. NOT: base zararlıysa amaç 'kârı korumak'")
    print("   değil, kaybeden tabanı seçicilikle kâra çevirmek — toplam-$ işareti dönmeli.)")


# ---------------------------------------------------------------------------
# Numpy lojistik (bağımlılıksız)
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticModel:
    kind = "logistic"

    def __init__(self, w, mu, sd):
        self.w, self.mu, self.sd = w, mu, sd

    @classmethod
    def fit(cls, X, y, *, l2=1.0, lr=0.1, iters=3000):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        n, d = Xs.shape
        Xb = np.hstack([np.ones((n, 1)), Xs])
        w = np.zeros(d + 1)
        for _ in range(iters):
            p = _sigmoid(Xb @ w)
            grad = Xb.T @ (p - y) / n
            grad[1:] += l2 * w[1:] / n
            w -= lr * grad
        return cls(w, mu, sd)

    def predict_proba(self, X):
        Xs = (X - self.mu) / self.sd
        Xb = np.hstack([np.ones((len(X), 1)), Xs])
        return _sigmoid(Xb @ self.w)


class GBMModel:
    kind = "gbm"

    def __init__(self, clf):
        self.clf = clf

    @classmethod
    def fit(cls, X, y, *, seed=42):
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=seed)
        clf.fit(X, y)
        return cls(clf)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]


class XGBModel:
    """XGBoost (gradient-boosted ağaçlar) — GBMModel'den daha güçlü/düzenlileştirilmiş.

    İmbalans (kazanma oranı ~%40-50) scale_pos_weight=neg/pos ile telafi edilir.
    `calibrate` verilirse CalibratedClassifierCV (izotonik/sigmoid) ile olasılıklar
    EĞİTİM verisi üstünde çapraz-doğrulanarak kalibre edilir (test'e dokunmaz → OOS
    dürüstlüğü korunur). Kalibrasyon, olasılık→boyut (prob_sizing) için ŞART: ham GBM
    olasılıkları sınıf-frekansını yansıtmaz, sizing'i yanıltır.

    Serileştirme: meta_gate.py pkl'i yükleyip `clf.predict_proba` çağırır → bu sınıfın
    `.clf`'i (ham XGBClassifier ya da CalibratedClassifierCV) doğrudan takılır. Üretimde
    xgboost kurulu olmalı; değilse meta_gate try/except ile pasife düşer (güvenli)."""

    kind = "xgb"

    def __init__(self, clf):
        self.clf = clf

    @classmethod
    def fit(cls, X, y, *, seed=42, calibrate: str | None = None):
        from xgboost import XGBClassifier
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        spw = (neg / pos) if pos > 0 else 1.0
        base = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
            scale_pos_weight=spw, eval_metric="logloss",
            tree_method="hist", random_state=seed, n_jobs=1)
        # B8 (2026-07-09): n_jobs=1 → TEK iş parçacığı. n_jobs=0 (tüm çekirdek) ağaç
        # toplamlarının kayan-nokta indirgeme SIRASINI çekirdek sayısına göre değiştirir →
        # aynı veride farklı makinada farklı model = belirlenimsiz terfi. Tek-iş = tekrarlanabilir.
        if calibrate in ("isotonic", "sigmoid"):
            from sklearn.calibration import CalibratedClassifierCV
            clf = CalibratedClassifierCV(estimator=base, method=calibrate, cv=3)
            clf.fit(X, y)
        else:
            base.fit(X, y)
            clf = base
        return cls(clf)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Bölme stratejileri
# ---------------------------------------------------------------------------
def split_time(df, frac):
    df = chrono_sort(df)   # ek-bulgu #3: string değil datetime sıralaması (karışık format)
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]


def split_symbol(df, frac, seed=42):
    syms = sorted(df["symbol"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(syms)
    cut = int(len(syms) * frac)
    train_syms = set(syms[:cut])
    tr = df[df["symbol"].isin(train_syms)]
    te = df[~df["symbol"].isin(train_syms)]
    return tr, te


# ---------------------------------------------------------------------------
# Değerlendirme + kapı simülasyonu
# ---------------------------------------------------------------------------
def evaluate(model, tr, te, *, label: str):
    Xtr = tr[FEATURE_NAMES].to_numpy(float)
    ytr = tr["label"].to_numpy(float)
    Xte = te[FEATURE_NAMES].to_numpy(float)
    yte = te["label"].to_numpy(float)
    pnltr = tr["pnl"].to_numpy(float)
    pnlte = te["pnl"].to_numpy(float)
    ptr = model.predict_proba(Xtr)
    pte = model.predict_proba(Xte)

    print(f"\n{'='*66}\n  MODEL={model.kind.upper()}  BÖLME={label}  "
          f"(eğitim {len(tr)} / test {len(te)})\n{'='*66}")
    ece_te = _ece(yte, pte, bins=10)
    print(f"  AUC eğitim: {_auc(ytr, ptr):.3f}  |  AUC test (OOS): {_auc(yte, pte):.3f}"
          f"  |  ECE test: {ece_te:.3f}")

    # KALEM 1: eşik SEÇİMİ TRAIN üzerinde yapılır (test'e HİÇ bakmadan → test-sızıntısı +
    # çoklu-test yanlılığı önlenir). Adaylardan TRAIN kapı şartını (kept≥5, e>base, pf>1)
    # geçen en yüksek TRAIN-expectancy'li eşik seçilir; TEST yalnız DOĞRULAR.
    thr_candidates = (0.45, 0.50, 0.55, 0.60)
    _, base_e_tr, _ = _pf_exp(pnltr)
    sel_thr: float | None = None
    sel_e_tr: float = float("-inf")
    for thr in thr_candidates:
        kept_tr = pnltr[ptr >= thr]
        if len(kept_tr) < 5:
            continue
        pf_tr, e_tr, _ = _pf_exp(kept_tr)
        if e_tr > base_e_tr and pf_tr > 1.0 and e_tr > sel_e_tr:
            sel_thr, sel_e_tr = thr, e_tr

    base_pf, base_e, base_wr = _pf_exp(pnlte)
    print(f"  {'eşik':>6} {'alınan':>7} {'%al':>5} {'kazanma%':>9} {'PF':>7} {'expectancy':>12}"
          "   (seçim TRAIN'de yapılır; tablo şeffaflık içindir)")
    print(f"  {'BASE':>6} {len(pnlte):>7} {'100':>5} {base_wr:>8.1f} {base_pf:>7.2f} {base_e:>12.2f}")
    for thr in thr_candidates:
        kept = pnlte[pte >= thr]
        is_sel = sel_thr is not None and thr == sel_thr
        if len(kept) < 5 and not is_sel:
            continue
        pf, e, wr = _pf_exp(kept)
        mark = "  ← seçilen (train)" if is_sel else ""
        pct = 100 * len(kept) / len(pnlte) if len(pnlte) else 0.0
        print(f"  {thr:>6.2f} {len(kept):>7} {pct:>4.0f}% "
              f"{wr:>8.1f} {pf:>7.2f} {e:>12.2f}{mark}")

    # Seçilen eşik TEST'te DOĞRULANIR: yalnız e_te>base_e ve pf_te>1 ise kenar sayılır;
    # TRAIN'de hiçbir eşik geçmediyse (sel_thr None) best_gate=None.
    best = None
    if sel_thr is not None:
        kept_te = pnlte[pte >= sel_thr]
        pf_te, e_te, wr_te = _pf_exp(kept_te)
        if e_te > base_e and pf_te > 1.0:
            best = (sel_thr, pf_te, e_te, wr_te, len(kept_te))
    _retention_sweep(pte, pnlte)
    return {"oos_auc": _auc(yte, pte), "base_pf": base_pf, "base_e": base_e,
            "best_gate": best, "thr_source": "train", "n_thr_tried": len(thr_candidates),
            "ece_test": ece_te}


def _code_sha() -> str | None:
    """Üretilen modelin provenance'ı için mevcut commit'in kısa SHA'sı. Git yok/başarısız
    → None (provenance yine yazılır, code_sha alanı None)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5, cwd=str(ROOT))
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:
        return None


def _dataset_sha256(path: str) -> str | None:
    """CSV gerçek bir dosyaysa içeriğinin sha256'sı; değilse (canlı build vb.) None.
    Aynı modelin hangi veriden çıktığını denetlenebilir kılar (tekrar-üretilebilirlik)."""
    try:
        p = Path(path)
        if p.is_file():
            return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return None
    return None


def _provenance(*, dataset_path: str, n_rows: int, train_frac: float, min_gate_keep: int,
                n_configs_tried: int, calibrate: str | None, ece_test: float | None,
                seed: int = 42) -> dict:
    """Model provenance (dürüstlük + tekrar-üretilebilirlik izi). save_model'e verilir;
    meta_gate._load fazladan anahtarı yok sayar → yükleme sözleşmesi bozulmaz. Eşik seçimi
    KALEM 1 ile TRAIN'de yapıldığından threshold_selection='train' sabittir; n_configs_tried
    çoklu-test deflasyon bağlamını (eşik×model×bölme) kaydeder."""
    return {
        "code_sha": _code_sha(),
        "dataset_sha256": _dataset_sha256(dataset_path),
        "n_rows": int(n_rows),
        "n_features": len(FEATURE_NAMES),
        "seed": int(seed),
        "train_frac": float(train_frac),
        "min_gate_keep": int(min_gate_keep),
        "threshold_selection": "train",
        "n_configs_tried": int(n_configs_tried),
        "calibrate": calibrate,
        "ece_test": (round(float(ece_test), 4) if ece_test is not None else None),
        "saved_at": datetime.now(UTC).isoformat(),
    }


def save_model(model, *, trained_on: str, oos_auc: float, gate_threshold: float,
               provenance: dict | None = None, output_dir: Path | None = None):
    """Modeli üretim formatında data/'ya yaz (meta_gate.py'nin okuduğu sözleşme).
    Mevcut model dosyası varsa üzerine yazmadan önce .bak-<tarih> olarak arşivlenir
    (kötü giden terfi geri alınabilsin). Yazılan yolu döndürür.

    provenance verilirse meta['provenance']'a gömülür (json + pkl + meta_model_info.json).
    meta_gate._load yalnız bilinen anahtarları okur → fazla alan sözleşmeyi bozmaz.
    `output_dir` verilirse aday ayrı dizine yazılır; üretim dosyaları değişmez."""
    stamp = date.today().isoformat()
    meta: dict = {"features": FEATURE_NAMES, "kind": model.kind, "trained_on": str(trained_on),
                  "oos_auc": round(float(oos_auc), 4), "gate_threshold": float(gate_threshold)}
    if provenance is not None:
        meta["provenance"] = provenance
    target = DATA_DIR if output_dir is None else Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "meta_model.json"
    pkl_path = target / "meta_model.pkl"
    if model.kind == "logistic":
        if json_path.exists():
            json_path.replace(json_path.with_suffix(f".json.bak-{stamp}"))
        # B3 (2026-07-09): BAYAT .pkl'i de arşivle. meta_gate._load .pkl'i .json'dan ÖNCE
        # okur → kalan eski GBM/XGB .pkl'i yeni lojistik modeli SESSİZCE gölgeler (yanlış
        # model üretime çıkar). Arşivleyerek yalnız yeni lojistik yüklenir.
        if pkl_path.exists():
            pkl_path.replace(pkl_path.with_suffix(f".pkl.bak-{stamp}"))
        meta.update({"w": model.w.tolist(), "mu": model.mu.tolist(), "sd": model.sd.tolist()})
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return json_path
    if pkl_path.exists():
        pkl_path.replace(pkl_path.with_suffix(f".pkl.bak-{stamp}"))
    # B3 simetrik: gbm/xgb kaydında bayat .json'ı arşivle. Öncelik .pkl'de olduğundan
    # yükleme doğru olurdu, ama eski .json geri-dönüş/denetimde yanıltmasın diye arşivlenir.
    if json_path.exists():
        json_path.replace(json_path.with_suffix(f".json.bak-{stamp}"))
    with open(pkl_path, "wb") as f:
        pickle.dump({"clf": model.clf, **meta}, f)
    (target / "meta_model_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return pkl_path


def main():
    ap = argparse.ArgumentParser(description="Meta-labeling eğit + OOS doğrula")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--model", default="both",
                    choices=["logistic", "gbm", "xgb", "both", "all"],
                    help="both=logistic+gbm, all=logistic+gbm+xgb")
    ap.add_argument("--split", default="both", choices=["time", "symbol", "both"])
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--calibrate", default=None, choices=["isotonic", "sigmoid"],
                    help="xgb olasılıklarını eğitim verisinde kalibre et (prob_sizing için)")
    ap.add_argument("--save", action="store_true", help="en iyi modeli data/'ya kaydet")
    ap.add_argument("--min-gate-keep", type=int, default=50,
                    help="--save için: her bölmede kapı kenarının dayanması gereken "
                         "asgari tutulan-işlem sayısı (altı = gürültü, kaydedilmez)")
    args = ap.parse_args()

    csv_path = args.csv
    if not csv_path:
        cands = sorted(glob.glob(str(DATA_DIR / "meta_dataset_*.csv")), key=lambda p: __import__("os").path.getsize(p))
        if not cands:
            print("Dataset bulunamadı. Önce build_meta_dataset.py çalıştır.")
            return
        csv_path = cands[-1]  # en büyük (en çok işlemli) dataset
    df = pd.read_csv(csv_path)
    df, missing = ensure_feature_columns(df)
    if missing:
        print(f"UYARI: eski-şema dataset, {len(missing)} özellik nötr-dolduruldu: {missing}")
    print(f"Dataset: {csv_path}\nİşlem: {len(df)} | sembol: {df['symbol'].nunique()} | "
          f"genel kazanma %{100*df['label'].mean():.1f}")

    if args.model == "both":
        models = ["logistic", "gbm"]
    elif args.model == "all":
        models = ["logistic", "gbm", "xgb"]
    else:
        models = [args.model]
    splits = ["time", "symbol"] if args.split == "both" else [args.split]
    results = {}
    for sp in splits:
        tr, te = split_time(df, args.train_frac) if sp == "time" else split_symbol(df, args.train_frac)
        # D10 (Tur-10): Xtr/ytr yalnız bölmeye bağlı → model döngüsünün DIŞINA
        # (--model both/all'da aynı numpy dönüşümü 2-3 kez tekrarlanıyordu).
        Xtr = tr[FEATURE_NAMES].to_numpy(float)
        ytr = tr["label"].to_numpy(float)
        for mk in models:
            if mk == "logistic":
                model = LogisticModel.fit(Xtr, ytr, l2=args.l2)
            elif mk == "xgb":
                model = XGBModel.fit(Xtr, ytr, calibrate=args.calibrate)
            else:
                model = GBMModel.fit(Xtr, ytr)
            results[(mk, sp)] = (model, evaluate(model, tr, te, label=sp))

    print(f"\n{'='*66}")
    print("ÖZET — bir model HER İKİ bölmede de kapı kenarı (←) gösteriyorsa dayanıklı.")
    print(f"{'='*66}")
    for (mk, sp), (_, r) in results.items():
        g = r["best_gate"]
        gate = (f"eşik {g[0]:.2f} → PF {g[1]:.2f} E {g[2]:+.0f} (al {g[4]})" if g
                else "kenar YOK")
        print(f"  {mk:<9} {sp:<7} OOS-AUC {r['oos_auc']:.3f} | base PF {r['base_pf']:.2f} | {gate}")

    # KALEM 1: çoklu-test deflasyon bağlamı — kaç konfig denendi? (eşik×model×bölme).
    n_thr = 4
    n_configs = n_thr * len(models) * len(splits)
    print(f"\n  Denenen konfig: {n_thr} eşik × {len(models)} model × {len(splits)} bölme "
          f"= {n_configs} (çoklu-test bağlamı; eşik seçimi TRAIN'de yapılır — test yalnız doğrular).")

    if args.save:
        # ACI DERS #3: dayanıklılık = AYNI modelin HEM time HEM symbol bölmesinde kapı
        # kenarı göstermesi. Eskiden tek bölmedeki kenar bile kaydediliyordu (özet bunu
        # söylüyor ama kod zorlamıyordu) → overfit model üretime sızabilirdi.
        if len(splits) < 2:
            print("\nKaydedilmedi: dayanıklılık İKİ bölme ister → --split both ile çalıştır.")
            return
        # ACI DERS #3+: çift-bölme kenarı YETMEZ — bir bölmedeki "kenar" avuç dolusu
        # işleme dayanıyorsa istatistiksel gürültüdür (gözlem: 10y PIT'te logistic
        # time-split 38.858'den yalnız 6 işlem tuttu, PF 3.68 → şans; purged CV aynı
        # modelde kenar YOK dedi). best_gate=(thr,pf,e,wr,kept); kept < MIN_GATE_KEEP
        # olan bölme dayanıklılık sayılmaz. Örneklem şartı olmadan --save gürültüyü yazar.
        MIN_GATE_KEEP = args.min_gate_keep
        durable: dict[str, float] = {}
        for mk in models:
            gt = results.get((mk, "time"), (None, {}))[1].get("best_gate")
            gs = results.get((mk, "symbol"), (None, {}))[1].get("best_gate")
            if gt and gs and gt[4] >= MIN_GATE_KEEP and gs[4] >= MIN_GATE_KEEP:
                durable[mk] = min(gt[2], gs[2])   # kötü bölmenin expectancy'si (muhafazakâr)
        if not durable:
            print(f"\nKaydedilmedi: hiçbir model HER İKİ bölmede yeterli örneklemle "
                  f"(≥{MIN_GATE_KEEP} işlem) kapı kenarı göstermedi (ACI DERS #3 — tek "
                  f"bölme / avuç-dolusu işlem overfit ya da gürültü olabilir).")
            return
        mk = max(durable, key=lambda k: durable[k])
        # B5 (2026-07-09): model↔eşik TUTARLILIĞI. Kaydedilen model = TIME alt-modeli;
        # eşik + oos_auc de AYNI alt-modelin OOS değerlendirmesinden gelir. Böylece
        # deploy edilen modelin olasılık ölçeği ile saklanan eşik aynı dağılımdan çıkar.
        # (Tam-veride TAZE eğitilmiş bir modeli alt-model eşiğiyle kaydetmek uyuşmazlık
        # olurdu — olasılık ölçeği kayar, eşik anlamını yitirir. live_retrain de aynı
        # tutarlılığı uygular.)
        model, r = results[(mk, "time")]
        # KALEM 3: provenance izi (code_sha, dataset_sha256, ECE, konfig sayısı, saved_at).
        prov = _provenance(dataset_path=csv_path, n_rows=len(df), train_frac=args.train_frac,
                           min_gate_keep=args.min_gate_keep, n_configs_tried=n_configs,
                           calibrate=args.calibrate, ece_test=r.get("ece_test"))
        path = save_model(model, trained_on=csv_path, oos_auc=r["oos_auc"],
                          gate_threshold=r["best_gate"][0], provenance=prov)
        print(f"\n→ {model.kind.upper()} model kaydedildi (iki-bölme dayanıklı): {path} "
              f"(eşik {r['best_gate'][0]})")


if __name__ == "__main__":
    main()
