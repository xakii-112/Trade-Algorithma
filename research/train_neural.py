"""
train_neural.py — Sinir ağı katmanının (nihai/ai/neural.py) purged+embargolu K-fold CV
ile DÜRÜST ölçümü (AI v3 WP-L, Tur-16).

`research/purged_cv.py`'nin bölme mantığını (`purged_folds`) YENİDEN KULLANIR (import;
kopyalama yok — tek doğruluk kaynağı orada kalır). DSR (Deflated Sharpe) raporu için
`research/deflated.py` kullanılır. Sonuç NE ÇIKARSA o raporlanır — "kenar bulundu" imalı
dil YASAK (CLAUDE.md ACI DERSLER + AI_V3_DERIN_BEYIN_SPEC.md §0.3).

İki veri kaynağı:
  --dataset yol.csv   : GERÇEK veri, train_meta.py/purged_cv.py ile AYNI şema
                        (`nihai.signals.meta_features.FEATURE_NAMES` + `label` kolonu;
                        varsa `entry_time`/`bars_held` kronolojik sıralama + purge önerisi
                        için kullanılır — purged_cv.py ile birebir tutarlı).
  --synthetic N       : BÖLÜM F sentetik müfredat (GBM + blok-bootstrap + OU + rejim
                        karışımı; TEK bir sentetik log-fiyat serisinden kayan pencereyle
                        N örnek türetilir; seed'li → deterministik; gerçek veri sızıntısı
                        YAPISAL OLARAK imkânsız, çünkü hiçbir gerçek veri okunmaz).
  İkisi de verilmezse çalıştırılamaz (kullanım hatası, argparse dışı — açık mesajla çıkar).
  İkisi de verilirse --dataset önceliklidir (gerçek veri sentetiğe tercih edilir).

`--save path_base` VERİLMEDİKÇE HİÇBİR DOSYA YAZILMAZ. Verilirse TÜM veri üstünde eğitilen
TEK bir nihai model `NeuralCapacity.save(path_base)` ile kaydedilir — YALNIZ verilen
path_base'e (models.py/neural.py sözleşmesi). Üretim `data/meta_model.*` dosyalarına bu
script'ten YAZILMAZ (anayasa madde 2) — `--save` için o yolu vermek çağıranın sorumluluğu
DIŞINDA bırakılmıştır (bu script hiçbir zaman kendiliğinden o yola yazmaz).

v4 (Tur-22): eğitim reçetesi bayrakları (`--scheduler cosine --ema 0.999 --class-weight
balanced --label-smoothing 0.05 --early-stop-metric val_auc --calibrate --device auto`) ve
derin topluluk (`--ensemble K`). HEPSİ opt-in — bayraksız koşu v3 ile bit-denktir, yani
eski ölçümler geçerliliğini korur. Rapora AUC'nin yanına Brier + ECE eklendi: bu projede
model kenar bulmak zorunda değildir ama DÜRÜST olasılık üretmek zorundadır.

Kullanım (repo kökünden, -m ile):
    python -m research.train_neural --tier deep_mlp --folds 5 --synthetic 2000
    python -m research.train_neural --tier quant_transformer --folds 5 --dataset data/meta_dataset_live_2026-07-20.csv
    python -m research.train_neural --tier deep_mlp --folds 3 --synthetic 5000 --save data/tmp/neural_deneme
    python -m research.train_neural --tier deep_mlp --folds 5 --dataset data/x.csv \
        --scheduler cosine --ema 0.999 --class-weight balanced --calibrate --ensemble 5
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.ai import neural  # noqa: E402 — sys.path/encoding ayarından SONRA (üstteki desenle tutarlı)
from research import deflated
from research.purged_cv import purged_folds, suggest_purge

# purged_cv.py'nin ZATEN yeniden kullandığı model-değerlendirme yardımcıları (tek kaynak).
from research.train_meta import _auc, chrono_sort

# ---------------------------------------------------------------------------
# Sentetik müfredat (BÖLÜM F): GBM + blok-bootstrap + OU + rejim karışımı.
# Tek bir sentetik log-fiyat serisinden kayan pencereyle örnekler türetilir; hiçbir gerçek
# veri okunmaz → sızıntı YAPISAL OLARAK imkânsız. h_/d_/w_ önek konvansiyonu (neural.py
# §3.1 çoklu zaman-ölçeği) 3 farklı bakış-penceresiyle (10/20/100 bar) taklit edilir.
# ---------------------------------------------------------------------------
SYNTHETIC_FEATURES: list[str] = [
    "h_ret1", "h_ret5", "h_vol10", "h_mom10", "h_rsi14",
    "d_ret20", "d_vol20", "d_trend20",
    "w_ret100", "w_vol100", "w_trend100",
    "skew20",
]
_MAX_LOOKBACK = 100    # w_* özellikleri için gereken en geniş pencere
_HORIZON = 5           # etiket ufku (bar) — leak-free: yalnız GELECEK barlardan okunur


def _generate_synthetic_series(n_bars: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """GBM + OU + blok-bootstrap KARIŞIMINDAN tek bir log-fiyat serisi (n_bars uzunluğunda)
    + üretici rejim kimliği dizisi (0=GBM 1=OU 2=blok-bootstrap) üretir. Blok uzunlukları
    geometrik dağılımdan (ortalama ~200 bar) — "rejim karışımı" (BÖLÜM F). Deterministik:
    aynı (n_bars, seed) → aynı seri (anayasa madde 6)."""
    rng = np.random.default_rng(seed)
    log_p = np.empty(n_bars, dtype=float)
    gen_type = np.zeros(n_bars, dtype=int)
    log_p[0] = 0.0
    base_steps = rng.normal(0.0001, 0.008, size=500)   # blok-bootstrap tampon serisi
    i = 1
    while i < n_bars:
        block_len = min(n_bars - i, int(rng.geometric(1.0 / 200)) + 20)
        regime = int(rng.integers(0, 3))
        gen_type[i:i + block_len] = regime
        if regime == 0:                                  # GBM
            mu = rng.uniform(-0.0002, 0.0004)
            sigma = rng.uniform(0.003, 0.02)
            steps = rng.normal(mu, sigma, size=block_len)
            log_p[i:i + block_len] = log_p[i - 1] + np.cumsum(steps)
        elif regime == 1:                                 # OU (ortalamaya-dönüş)
            theta = rng.uniform(0.02, 0.15)
            sigma = rng.uniform(0.002, 0.01)
            target = log_p[i - 1] + rng.normal(0, 0.02)
            x = log_p[i - 1]
            path = np.empty(block_len)
            for j in range(block_len):
                x = x + theta * (target - x) + rng.normal(0, sigma)
                path[j] = x
            log_p[i:i + block_len] = path
        else:                                              # blok-bootstrap
            start = int(rng.integers(0, max(1, len(base_steps) - block_len)))
            steps = base_steps[start:start + block_len]
            if len(steps) < block_len:
                steps = np.pad(steps, (0, block_len - len(steps)))
            log_p[i:i + block_len] = log_p[i - 1] + np.cumsum(steps)
        i += block_len
    return log_p, gen_type


def build_synthetic_dataset(n_samples: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """(X, y, feature_names, aux) üretir. X:(n_samples,12) — SYNTHETIC_FEATURES sırasıyla,
    her satır YALNIZ o bardan GERİYE dönük pencerelerden hesaplanır (leak-free: `.shift`/
    `.rolling` ileri bakmaz). y = ufuk-sonu getiri işareti (0/1). aux = {'magnitude','vol',
    'regime'} — SPEC §3.1 çok-görevli hedefleri; sentetik olduğundan GERÇEK üretici rejimi
    (0..5 = 3 tip × 2 vol-kovası) doğrudan bilinir (gerçek veride bu bir varsayım/etiketleme
    seçimi olurdu; burada zemin-gerçek)."""
    n_bars = int(n_samples) + _MAX_LOOKBACK + _HORIZON
    log_p, gen_type = _generate_synthetic_series(n_bars, seed)
    price = pd.Series(np.exp(log_p))
    ret1 = price.pct_change()

    feat = pd.DataFrame(index=price.index)
    feat["h_ret1"] = ret1
    feat["h_ret5"] = price.pct_change(5)
    feat["h_vol10"] = ret1.rolling(10).std()
    roll_min10, roll_max10 = price.rolling(10).min(), price.rolling(10).max()
    feat["h_mom10"] = (price - roll_min10) / (roll_max10 - roll_min10 + 1e-9)
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    feat["h_rsi14"] = 100.0 - 100.0 / (1.0 + gain / (loss + 1e-9))
    feat["d_ret20"] = price.pct_change(20)
    feat["d_vol20"] = ret1.rolling(20).std()
    feat["d_trend20"] = (price - price.shift(20)) / 20.0
    feat["w_ret100"] = price.pct_change(100)
    feat["w_vol100"] = ret1.rolling(100).std()
    feat["w_trend100"] = (price - price.shift(100)) / 100.0
    feat["skew20"] = ret1.rolling(20).skew()
    feat = feat[SYNTHETIC_FEATURES]

    fwd_ret = price.shift(-_HORIZON) / price - 1.0            # ETİKET: yalnız GELECEK (izinli)
    fwd_vol = ret1.rolling(_HORIZON).std().shift(-_HORIZON)    # ileri-ufuk gerçekleşen vol (aux hedef)
    vol_bucket = pd.Series(0, index=price.index)
    med_vol = pd.Series(feat["h_vol10"]).median()
    vol_bucket[feat["h_vol10"] > med_vol] = 1
    regime = pd.Series(gen_type, index=price.index) * 2 + vol_bucket   # 0..5 (3 tip × 2 vol-kovası)

    lo, hi = _MAX_LOOKBACK, n_bars - _HORIZON
    idx = np.arange(lo, hi)
    X = feat.iloc[idx].to_numpy(dtype=float)
    y = (fwd_ret.iloc[idx].to_numpy(dtype=float) > 0).astype(float)
    abs_fwd = fwd_ret.iloc[idx].abs().to_numpy(dtype=float)
    magnitude = np.clip(
        pd.qcut(pd.Series(abs_fwd), 5, labels=False, duplicates="drop").to_numpy(dtype=float), 0, 4)
    aux = {
        "magnitude": magnitude,
        "vol": fwd_vol.iloc[idx].fillna(0.0).to_numpy(dtype=float),
        "regime": regime.iloc[idx].to_numpy(dtype=float),
    }
    # Satır bütünlüğü: rolling/shift NaN üretebilir (w_* ısınma dönemi lo'dan ÖNCE bitmiş
    # olmalı — savunmacı ikinci kontrol; kalan NaN varsa 0'a düşür, çökme yok).
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)
    return X, y, list(SYNTHETIC_FEATURES), aux


def load_real_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str], int | None]:
    """`train_meta.py`/`purged_cv.py` ile AYNI şema (FEATURE_NAMES + label); kronolojik
    sıralama `chrono_sort` iledir (purged_folds zaman-sıralı indeks varsayar). Dönüşün son
    elemanı `suggest_purge(df)` — dataset'te `bars_held` varsa purged_cv.py ile BİREBİR aynı
    sol-purge önerisini türetir (yoksa None → `purged_folds` embargo-simetriğine düşer)."""
    from nihai.signals.meta_features import FEATURE_NAMES, ensure_feature_columns
    df = pd.read_csv(csv_path)
    df, missing = ensure_feature_columns(df)
    if missing:
        print(f"UYARI: eski-şema dataset, {len(missing)} özellik nötr-dolduruldu: {missing}")
    df = chrono_sort(df)
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=float)
    X = neutralize_degenerate(X, list(FEATURE_NAMES))
    purge = suggest_purge(df)
    return X, y, list(FEATURE_NAMES), purge


_SIGMA_MIN = 1e-6


def neutralize_degenerate(X: np.ndarray, names: list[str],
                          sigma_min: float = _SIGMA_MIN) -> np.ndarray:
    """σ < `sigma_min` olan (fiilen SABİT) özellikleri 0'a sıfırla ve GÖRÜNÜR kıl (F6).

    NEDEN: `rr_ratio` eğitim setinde σ=3,2e-15, `confluence` sabit 1,0 ölçüldü (2026-07-26
    denetimi P1-5). Sabit bir kolon hiçbir bilgi taşımaz ama σ'ya bölen her normalizasyonda
    ~1e-15'lik bölen = sayısal patlama ("bölme mayını"); ağırlığı da keyfî büyür.

    Kolon DÜŞÜRÜLMEZ, sıfırlanır: `neural_shadow` kayıtlı `feature_names`'in FEATURE_NAMES
    ile BİREBİR eşleşmesini şart koşuyor (train/serve kilidi) — kolonu atmak canlı köprüyü
    kırardı. Sıfırlanmış kolon hem mayını etkisizleştirir hem sözleşmeyi korur."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return X
    sd = np.nanstd(X, axis=0)
    dej = np.flatnonzero(~np.isfinite(sd) | (sd < sigma_min))
    if dej.size:
        print(f"UYARI (F6): σ<{sigma_min:g} olan {dej.size} özellik SIFIRLANDI (bilgi yok + "
              f"bölme mayını): {[names[i] for i in dej if i < len(names)]}")
        X = X.copy()
        X[:, dej] = 0.0
    return X


# ---------------------------------------------------------------------------
# DSR raporu — deflated.py'yi YENİDEN KULLAN (kopyalama yok)
# ---------------------------------------------------------------------------
def excess_pseudo_returns(p: np.ndarray, y_test: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Sözde-getiri = MODELİN isabeti − TRİVİYAL (çoğunluk-sınıfı) kuralın isabeti.

    NEDEN FAZLALIK: ham isabet (+1/−1) kullanılırsa taban oranı %50'den uzak veri setlerinde
    "hep çoğunluk sınıfını söyle" kuralı bile yüksek Sharpe üretir ve büyük n'de PSR/DSR 1.0'a
    fırlar. Ölçüldü (2026-07-25, PIT S&P500, taban %36.2): OOS-AUC 0.500 — yani SIFIR ayırt
    etme — iken ham seri Sharpe 0.2652 / DSR 1.0 veriyordu. DSR bir KANIT KAPISI olduğundan
    bu, sınıf dengesizliğini beceri sanan bir yanlış-pozitiftir (CLAUDE.md ACI DERSLER #3:
    metrik yeşil diye kenar var demek değildir).

    Çoğunluk sınıfı YALNIZ eğitim fold'undan hesaplanır (test etiketlerine bakılmaz → sızıntı
    yok). Beceri yoksa seri ~0 etrafındadır; trivial kuralın kendisi tam olarak 0 üretir."""
    hits = ((np.asarray(p) >= 0.5).astype(float) == np.asarray(y_test, dtype=float)).astype(float)
    majority = 1.0 if float(np.mean(np.asarray(y_train, dtype=float))) >= 0.5 else 0.0
    base_hits = (np.asarray(y_test, dtype=float) == majority).astype(float)
    return hits - base_hits


def dsr_report(pseudo_returns: list[float], n_trials: int) -> dict:
    """Fold'lar üstünde biriken sözde-getiri serisinden (bkz. `excess_pseudo_returns`:
    çoğunluk-kuralına göre FAZLALIK isabet) Sharpe + PSR + DSR. `n_trials` = bağımsız fold
    sayısı (çoklu-test bağlamı — kaç 'deneme' üstünden en iyi görünüm seçildi). Yetersiz veri
    (<2 örnek) VEYA sıfır varyans (modelin trivial kuraldan farkı yok) → tüm alanlar None
    (çökme yok; deflated.py'nin kendi NaN sözleşmesiyle tutarlı)."""
    arr = np.asarray(pseudo_returns, dtype=float)
    if arr.size < 2:
        return {"n": int(arr.size), "sharpe": None, "psr": None, "dsr": None, "n_trials": int(n_trials)}
    if float(np.std(arr)) == 0.0:
        # Sharpe = 0/0. Trivial kuralla BİREBİR aynı davranan model için doğru cevap
        # "ölçülemedi"dir — sıfır bölme NaN'ını deflated.py'ye taşımaya gerek yok.
        return {"n": int(arr.size), "sharpe": None, "psr": None, "dsr": None, "n_trials": int(n_trials)}
    m = deflated.sharpe_moments(arr)
    sr, n_obs, skew, kurt = m["sr"], m["n"], m["skew"], m["kurt"]
    var_trials = deflated.sharpe_estimator_var(sr, n_obs, skew=skew, kurt=kurt)
    d = deflated.dsr(sr, n_obs, n_trials=max(1, int(n_trials)), var_trials=var_trials, skew=skew, kurt=kurt)
    p = deflated.psr(sr, n_obs, skew=skew, kurt=kurt)
    return {
        "n": int(n_obs), "sharpe": round(float(sr), 4),
        "psr": (round(float(p), 4) if p == p else None),      # p==p → NaN eleme (self-eşitsizlik)
        "dsr": (round(float(d), 4) if d == d else None),
        "n_trials": int(n_trials),
    }


# ---------------------------------------------------------------------------
# CV döngüsü
# ---------------------------------------------------------------------------
def build_model(tier: str, feature_names: list[str], seed: int, n_members: int = 1) -> Any:
    """`n_members<=1` → `NeuralCapacity`; aksi hâlde `NeuralEnsemble` (derin topluluk).
    İkisi de AYNI API'yi sunar (fit/predict_proba/param_count/evaluate/save) — çağıranın
    dallanmasına gerek yoktur."""
    if int(n_members) <= 1:
        return neural.NeuralCapacity(tier, feature_names=feature_names, seed=seed)
    return neural.NeuralEnsemble(tier, feature_names=feature_names, seed=seed,
                                 n_members=int(n_members))


def run_cv(X: np.ndarray, y: np.ndarray, feature_names: list[str], *, tier: str, folds: int,
          max_epochs: int, aux: dict[str, np.ndarray] | None = None, embargo_frac: float = 0.01,
          seed: int = 42, purge: int | None = None, n_members: int = 1,
          recipe: dict | None = None, progress=None) -> dict:
    """purged_folds ile K bağımsız OOS penceresinde `tier` modelini eğitip ölç. Fold-başı
    OOS-AUC/Brier/ECE yazdırır; DSR raporu + özet verdikt döner. Sonuç NE ÇIKARSA o
    raporlanır. `purge=None` → `purged_folds` embargo-simetrik sol-purge kullanır (real
    dataset'te `load_real_dataset`'in döndürdüğü `suggest_purge` sonucu; sentetikte ufuk+1).

    v4 (Tur-22): `recipe` doğrudan `fit()`'e geçen eğitim reçetesi (scheduler/ema/
    class_weight/label_smoothing/early_stop_metric/calibrate/device); `n_members>1` derin
    topluluk. AUC'nin YANINDA kalibrasyon metrikleri (Brier/ECE) raporlanır — çünkü bu
    projede modelin işi kenar bulmak değil, DÜRÜST olasılık üretmektir; AUC~0.50'de bile
    kalibrasyon ölçülebilir ve iyileşebilir (Tur-20 dersi)."""
    n = len(y)
    embargo = max(1, int(round(n * embargo_frac)))
    kw = dict(recipe or {})
    fold_aucs: list[float] = []
    fold_briers: list[float] = []
    fold_eces: list[float] = []
    pseudo_returns: list[float] = []
    n_folds_done = 0
    param_count: int | None = None
    for fi, (tr_idx, te_idx) in enumerate(purged_folds(n, folds, embargo, purge=purge)):
        ytr = y[tr_idx]
        if len(tr_idx) < 4 or len(te_idx) < 1 or len(np.unique(ytr)) < 2:
            print(f"  fold {fi}: eğitim için yetersiz/tek-sınıf veri, atlandı")
            continue
        aux_tr = {k: v[tr_idx] for k, v in aux.items()} if aux else None
        # Fold'lar arası seed BLOKLARI ayrık: topluluk üyeleri seed..seed+K-1 kullandığından
        # fold başına K adım atlanır (aksi hâlde farklı fold'lar aynı ilklendirmeyi paylaşırdı).
        model = build_model(tier, feature_names, seed + fi * max(1, int(n_members)), n_members)
        progress_kw = {"progress": lambda event, fold=fi: progress({"fold": fold, **event})} if progress else {}
        model.fit(X[tr_idx], ytr, aux=aux_tr, max_epochs=max_epochs, **kw, **progress_kw)
        p = model.predict_proba(X[te_idx])
        yte = y[te_idx]
        auc = _auc(yte, p)
        met = neural.evaluation_metrics(yte, p)
        fold_aucs.append(auc)
        if met["brier"] is not None:
            fold_briers.append(float(met["brier"]))
        if met["ece"] is not None:
            fold_eces.append(float(met["ece"]))
        pseudo_returns.extend(excess_pseudo_returns(p, yte, ytr).tolist())
        n_folds_done += 1
        param_count = model.param_count()
        print(f"  fold {fi}: OOS-AUC {auc:.3f} | Brier {met['brier']:.4f} | ECE {met['ece']:.4f} "
              f"(n_tr={len(tr_idx)} n_te={len(te_idx)}, param_count={param_count})")

    dsr_res = dsr_report(pseudo_returns, n_trials=max(1, n_folds_done))
    auc_arr = np.array(fold_aucs) if fold_aucs else np.array([])
    return {
        "tier": tier, "n_folds_done": n_folds_done, "fold_aucs": fold_aucs,
        "auc_mean": (float(auc_arr.mean()) if auc_arr.size else None),
        "auc_std": (float(auc_arr.std()) if auc_arr.size else None),
        "brier_mean": (float(np.mean(fold_briers)) if fold_briers else None),
        "ece_mean": (float(np.mean(fold_eces)) if fold_eces else None),
        "param_count": param_count,
        "n_members": int(n_members),
        "recipe": kw,
        "dsr_report": dsr_res,
    }


def _torch_version() -> str | None:
    """Model kartına yazılacak torch sürümü; torch yoksa None (fail-soft — kart yine üretilir)."""
    try:
        import torch
        return str(torch.__version__)
    except Exception:  # noqa: BLE001 — sürüm bilgisi kartı düşürmemeli
        return None


def build_model_card(*, tier: str, feature_names: list[str], param_count: int | None,
                     data_source: str, n_rows: int, positive_rate: float, purge: int | None,
                     folds: int, max_epochs: int, seed: int, cv_result: dict,
                     save_path: str) -> dict:
    """`nihai.ai.registry.register` için MODEL KARTI içeriği (Tur-17 köprüsü).

    KENAR İDDİASI İÇERMEZ: kart, modelin nasıl eğitildiğini + OOS ölçümünün NE ÇIKTIĞINI
    kaydeder; "dayanıklı/kârlı/kenar var" gibi bir yorum taşımaz (CLAUDE.md ACI DERSLER,
    SPEC §0.3). Zaman damgası BİLEREK yok: registry hash'i içerikten türetir, ts eklemek
    aynı eğitimi her seferinde farklı hash'e sokardı (`registered_at`'i registry ekler)."""
    dsr = cv_result.get("dsr_report") or {}
    return {
        "kind": "neural",
        "tier": tier,
        "framework": "torch",
        "param_count": param_count,
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "data": {"source": data_source, "n_rows": int(n_rows),
                 "positive_rate": round(float(positive_rate), 4), "purge": purge},
        "hyperparams": {"folds": int(folds), "max_epochs": int(max_epochs), "seed": int(seed),
                        # v4: eğitim reçetesi + topluluk boyutu kartın PARÇASIDIR — aynı
                        # tier'ın farklı reçeteyle eğitilmiş iki kopyası artık ayırt edilir
                        # (registry hash'i içerikten türediği için kart da otomatik ayrışır).
                        "ensemble": int(cv_result.get("n_members") or 1),
                        "recipe": cv_result.get("recipe") or {}},
        "oos": {"auc_mean": cv_result.get("auc_mean"), "auc_std": cv_result.get("auc_std"),
                "brier_mean": cv_result.get("brier_mean"), "ece_mean": cv_result.get("ece_mean"),
                "n_folds_done": cv_result.get("n_folds_done"),
                "method": "purged+embargolu K-fold (research.purged_cv.purged_folds)"},
        "dsr": {"dsr": dsr.get("dsr"), "psr": dsr.get("psr"), "sharpe": dsr.get("sharpe"),
                "n_trials": dsr.get("n_trials"), "n": dsr.get("n")},
        # Topluluk artefaktı `<base>.json` manifestosu + üye başına `<base>.m{i}.json/.pt`
        # olarak durur; kart bunu DOĞRU listelemeli (yanlış dosya listesi = izlenebilirlik
        # kaybı ve `neural_shadow`'un "artefakt diskte yok" sanmasına giden yol).
        "artifacts": {"base_path": save_path,
                      "files": ([f"{save_path}.json", f"{save_path}.pt"]
                                if int(cv_result.get("n_members") or 1) <= 1 else
                                [f"{save_path}.json"] + [f"{save_path}.m{i}.{e}"
                                                          for i in range(int(cv_result["n_members"]))
                                                          for e in ("json", "pt")])},
        "env": {"python": platform.python_version(), "torch": _torch_version(),
                "platform": platform.platform()},
        "not": "KAPASİTE/kalibrasyon kaydıdır; kenar iddiası DEĞİLDİR. Üretim sinyaline "
               "bağlanmaz (anayasa: KARAR-DIŞI/GÖLGE).",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sinir ağı katmanının (neural.py) purged+embargolu K-fold CV ölçümü")
    ap.add_argument("--tier", default="quant_transformer", choices=neural.NEURAL_TIERS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--progress", action="store_true", help="Epoch/son minibatch kaybını anlık yaz")
    ap.add_argument("--json", default=None, help="CV ölçümlerini JSON'a yaz")
    ap.add_argument("--dataset", default=None, help="gerçek CSV (FEATURE_NAMES+label şeması)")
    ap.add_argument("--synthetic", type=int, default=None,
                    help="N örnek sentetik müfredat üret (--dataset verilmemişse kullanılır)")
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--save", default=None,
                    help="verilirse TÜM veriyle eğitilen model NeuralCapacity.save(path_base) ile "
                         "kaydedilir; verilmezse HİÇBİR dosya yazılmaz. data/meta_model* YASAK "
                         "(anayasa madde 2 — bu script o yola asla kendiliğinden yazmaz).")
    ap.add_argument("--register", action="store_true",
                    help="--save ile BİRLİKTE: eğitilen modelin KARTINI (hiperparametre + OOS-AUC "
                         "+ DSR + veri dönemi/boyutu + ortam) nihai.ai.registry'ye yazar. Tek "
                         "başına --register hiçbir şey yazmaz (artefaktsız kart anlamsız).")
    ap.add_argument("--registry-dir", default=None,
                    help="kart kökü (varsayılan: registry.AI_MODELS_DIR = data/ai_models)")
    # --- v4 (Tur-22) eğitim reçetesi: HEPSİ OPT-IN, varsayılanlar v3 ile bit-denk ---
    ap.add_argument("--ensemble", type=int, default=1,
                    help="K>1 → derin topluluk (K üye, üye başına farklı seed). Kapasite K×; "
                         "belirsizlik aleatorik/epistemik olarak ayrışır.")
    ap.add_argument("--scheduler", default="none", choices=["none", "cosine"],
                    help="ısınmalı kosinüs LR (varsayılan: sabit LR = v3 davranışı)")
    ap.add_argument("--ema", type=float, default=None, metavar="DECAY",
                    help="ağırlık EMA katsayısı (ör. 0.999); verilmezse EMA yok")
    ap.add_argument("--class-weight", default=None, choices=["balanced"],
                    help="sınıf dengesizliğini kayıpta dengele (YALNIZ eğitim diliminin oranından)")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--early-stop-metric", default="val_loss", choices=["val_loss", "val_auc"])
    ap.add_argument("--calibrate", action="store_true",
                    help="fit sonrası val diliminde sıcaklık ölçekleme (sıralamayı DEĞİŞTİRMEZ; "
                         "yalnız olasılıkları dürüstleştirir)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"],
                    help="v9 (Tur-42) BELLEK REÇETESİ: `adafactor` ikinci momenti satır×sütun "
                         "çarpanlarına ayırır ve birinci momenti hiç tutmaz → optimizatör "
                         "durumu O(n)→O(√n) (ULTRA'da ölçüldü: 6,17 GB → 0,01 GB). FARKLI bir "
                         "algoritmadır: ağırlıklar AdamW koşusuyla BİT-DENK OLMAZ, bu yüzden "
                         "seçim model kartına ve `meta()['recipe']['optimizer']`e yazılır.")
    ap.add_argument("--amp", action="store_true",
                    help="v9 (Tur-42): ileri geçişi bf16 autocast'e al (YALNIZ CUDA'da; "
                         "ağırlıklar float32 kalır, GradScaler gerekmez). Aktivasyon "
                         "belleğini ≈yarıya indirir. CPU'da SESSİZCE yok sayılır.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Tur-93: eğitim toplu-işi (verilmezse `fit` varsayılanı 256 = eski "
                         "davranış). 282M basamağı 8 GB kartta 256'da 7,18 GB, 128'de 5,07 GB ölçüldü.")
    ap.add_argument("--pretrain", action="store_true",
                    help="v7 (Tur-35): ETİKETSİZ maskeli özellik ön-eğitimi + warm start. "
                         "Etiket GEREKTİRMEZ — gövde önce özellikler arası yapıyı öğrenir, "
                         "sonra etiketli eğitim onun üstüne gelir. `deep_mlp`'te GEÇERSİZ.")
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--pretrain-mask-ratio", type=float, default=0.25,
                    help="satır başına maskelenecek özellik oranı (0<r<1)")
    ap.add_argument("--pretrain-dataset", default=None,
                    help="ön-eğitim için AYRI (etiketsiz olabilir) CSV; verilmezse ana "
                         "dataset'in X'i kullanılır")
    args = ap.parse_args(argv)

    if not neural.TORCH_AVAILABLE:
        print("torch kurulu değil — neural katmanı devre dışı (fail-soft). "
              "`pip install torch --index-url https://download.pytorch.org/whl/cpu`")
        return 1

    # Tur-65 — KAPALI KALAN DÜĞMEYİ GÖRÜNÜR KIL. `--device`/`--amp` Tur-42'de yazıldı
    # ama makinede CPU-derlemesi torch vardı, yani yol üretimde bir kez bile koşmadı
    # ("yazılıp koşulmayan izleyici, izleme değildir" dersinin eğitim tarafı). Torch
    # artık cu130 derlemesi ve ÖLÇÜLDÜ (RTX 4070 Laptop, n=4.000×32, 4 epoch):
    #   quant_transformer  cpu 135,21 sn → cuda 3,94 sn (34,3×) → +amp 1,81 sn (74,7×)
    #   deep_mlp           cpu   1,92 sn → cuda 0,58 sn ( 3,3×) → +amp 0,42 sn ( 4,6×)
    # VARSAYILAN BİLEREK `cpu` KALDI: GPU/bf16 farklı kayan-nokta sırası demektir,
    # yani geçmiş koşularla bit-denk DEĞİLDİR (bkz. `--optimizer adafactor` kararı).
    # Seçim operatörün; burada yalnız SEÇENEK olduğu söylenir.
    if args.device == "cpu" and neural.torch is not None and neural.torch.cuda.is_available():
        print(f"[ipucu] CUDA kullanılabilir ({neural.torch.cuda.get_device_name(0)}) ama "
              f"--device cpu seçildi. `--device auto --amp` ölçüldü: quant_transformer "
              f"eğitiminde 74,7× (135,2 → 1,8 sn). Sonuçlar bit-denk OLMAZ.")

    aux: dict[str, np.ndarray] | None = None
    purge: int | None = None
    if args.dataset:
        X, y, feature_names, purge = load_real_dataset(args.dataset)
        print(f"Dataset (gerçek): {args.dataset} | satır={len(y)} | özellik={len(feature_names)} | "
              f"pozitif oran %{100 * float(y.mean()):.1f}")
    elif args.synthetic:
        X, y, feature_names, aux = build_synthetic_dataset(args.synthetic, seed=42)
        purge = _HORIZON + 1   # etiket ufku bilinir (sentetik) — sol-purge doğrudan türetilir
        print(f"Dataset (SENTETİK müfredat): n={len(y)} | özellik={len(feature_names)} "
              f"({', '.join(feature_names)}) | pozitif oran %{100 * float(y.mean()):.1f}")
    else:
        print("Veri kaynağı yok: --dataset yol.csv VEYA --synthetic N verilmeli.")
        return 1

    recipe: dict[str, Any] = {
        "scheduler": args.scheduler, "ema_decay": args.ema, "class_weight": args.class_weight,
        "label_smoothing": args.label_smoothing, "early_stop_metric": args.early_stop_metric,
        "calibrate": bool(args.calibrate), "device": args.device,
        "optimizer_name": args.optimizer, "amp": bool(args.amp),
    }
    if args.batch_size:
        recipe["batch_size"] = int(args.batch_size)
    n_members = max(1, int(args.ensemble))
    print(f"tier={args.tier} | folds={args.folds} | max_epochs={args.max_epochs} | "
          f"topluluk={n_members} | reçete={ {k: v for k, v in recipe.items() if v not in (None, 'none', 0.0, False, 'cpu', 'adamw')} or 'v3 varsayılan'}")
    progress = (lambda event: print("EPOCH " + json.dumps(event), flush=True)) if args.progress else None
    result = run_cv(X, y, feature_names, tier=args.tier, folds=args.folds,
                    max_epochs=args.max_epochs, aux=aux, purge=purge,
                    n_members=n_members, recipe=recipe, progress=progress)

    print(f"\n{'=' * 66}")
    if result["auc_mean"] is not None:
        print(f"  ORTALAMA OOS-AUC: {result['auc_mean']:.3f} ± {result['auc_std']:.3f} "
              f"({result['n_folds_done']}/{args.folds} fold değerlendirildi)")
    else:
        print("  Hiçbir fold değerlendirilemedi (yetersiz/tek-sınıf veri).")
    if result.get("brier_mean") is not None:
        # AUC ~0.50 iken bile bu iki sayı ANLAMLIDIR: model ayırt edemiyor olabilir ama
        # olasılıkları dürüst mü? (Tur-20: kazanç kalibrasyon hizasındaydı, kenar değil.)
        print(f"  Kalibrasyon: Brier={result['brier_mean']:.4f} | ECE={result['ece_mean']:.4f} "
              f"(taban oranı %{100 * float(y.mean()):.1f})")
    if result.get("param_count"):
        print(f"  Kapasite: param_count={result['param_count']:,} "
              f"(topluluk={result.get('n_members')} üye)".replace(",", "."))
    dr = result["dsr_report"]
    if dr["dsr"] is not None:
        print(f"  DSR raporu: Sharpe(sözde-getiri)={dr['sharpe']} | PSR={dr['psr']} | "
              f"DSR={dr['dsr']} (n_trials={dr['n_trials']}, n_gözlem={dr['n']})")
    else:
        print(f"  DSR raporu: ölçülemedi (n_gözlem={dr['n']} yetersiz).")
    print("  Not: bu ölçüm KAPASİTE/kalibrasyon raporudur — 'kenar' iddiası içermez; "
          "sonuç ne çıkarsa o raporlanmıştır (SPEC §0.3/BÖLÜM F).")
    print(f"{'=' * 66}")

    if args.save:
        print(f"\n--save verildi: TÜM veriyle ({len(y)} satır) nihai model eğitiliyor...")
        final_model = build_model(args.tier, feature_names, 42, n_members)
        if args.pretrain:
            # Ön-eğitim ETİKET GÖRMEZ: ayrı (daha büyük, etiketsiz) bir X verilebilir.
            X_pre = X
            if args.pretrain_dataset:
                X_pre, _, pre_names, _ = load_real_dataset(args.pretrain_dataset)
                if list(pre_names) != list(feature_names):
                    print("HATA: --pretrain-dataset özellik şeması ana dataset ile uyuşmuyor.")
                    return 1
            print(f"→ maskeli ön-eğitim: n={len(X_pre)} satır | mask_ratio="
                  f"{args.pretrain_mask_ratio} | epochs={args.pretrain_epochs}")
            final_model.pretrain(X_pre, mask_ratio=args.pretrain_mask_ratio,
                                 max_epochs=args.pretrain_epochs, device=args.device)
            pre = (final_model.members[0] if hasattr(final_model, "members")
                   else final_model)._pretrain_info
            print(f"  rekonstrüksiyon MSE (val): {pre.get('val_recon_mse_first')} → "
                  f"{pre.get('val_recon_mse_best')} | {pre.get('n_epochs')} epoch | "
                  f"{pre.get('seconds')} sn")
            print("  NOT: rekonstrüksiyon iyileşmesi bir KENAR iddiası DEĞİLDİR (ACI DERS #5) — "
                  "yön tahmini ayrı ölçülür.")
            recipe["warm_start"] = True
        progress_kw = {"progress": lambda event: progress({"fold": "final", **event})} if progress else {}
        final_model.fit(X, y, aux=aux, max_epochs=args.max_epochs, **recipe, **progress_kw)
        final_model.save(args.save)
        print(f"→ kaydedildi: {args.save}.json / {args.save}.pt "
              f"(param_count={final_model.param_count()})")
        if args.register:
            from nihai.ai import registry
            card = build_model_card(
                tier=args.tier, feature_names=feature_names,
                param_count=final_model.param_count(),
                data_source=(args.dataset if args.dataset else f"synthetic:{args.synthetic}"),
                n_rows=len(y), positive_rate=float(y.mean()), purge=purge,
                folds=args.folds, max_epochs=args.max_epochs, seed=42,
                cv_result=result, save_path=args.save)
            written = registry.register(card, artifacts_base_path=args.registry_dir)
            base = Path(args.registry_dir) if args.registry_dir else registry.AI_MODELS_DIR
            print(f"→ model kartı yazıldı: hash={written['hash'][:16]}… "
                  f"({base / written['hash'] / 'card.json'})")
    else:
        print("\n--save verilmedi: hiçbir dosya yazılmadı.")
        if args.register:
            print("UYARI: --register YOK SAYILDI (--save olmadan kart yazılmaz — "
                  "artefaktı olmayan kart yanıltıcıdır).")

    if args.json:
        from nihai.atomic_io import atomic_write_text
        atomic_write_text(Path(args.json), json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
