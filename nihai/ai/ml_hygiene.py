"""
ml_hygiene.py — ETİKETLEME ve AŞIRI-ÖĞRENME KARŞITI ARAÇLAR (Tur-75). KARAR-DIŞI.

Dış geliştirme listesinin BÖLÜM 7'si 25 madde sayıyor; çoğu ZATEN var (purged CV ·
CSCV/PBO · DSR · Platt/izotonik · conformal aralık · SHAP · BOCPD) ya da bu donanımda
ölçülemez (PPO/TabNet/MAML/GNN/TFT — ACI DERS #5: yeni bir sınıflandırıcı kenar
yokluğunu çözmez). Geriye ölçülebilir ve karşılığı OLMAYAN sekiz madde kaldı:

| Madde | Fonksiyon |
|---|---|
| 126 Kombinatoryal purged CV (CPCV) | `cpcv_splits` · `cpcv_paths` |
| 127 Üçlü bariyer etiketleme | `triple_barrier_labels` |
| 129 Düşmanca doğrulama (veri kayması) | `adversarial_validation` |
| 130 Kümeli özellik önemi (clustered MDA) | `clustered_mda` |
| 132 Mahalanobis dağılım-dışı (OOD) tespiti | `mahalanobis_ood` |
| 133 Özellik nötrleştirme | `neutralize` |
| 146 fBM sentetik stres üreteci | `fbm_paths` |
| 147 Katman-dışı hedef kodlama | `oof_target_encode` |

ANAYASA: KARAR-DIŞI. Bu modül ETİKET ve DOĞRULAMA üretir, sinyal üretmez; hiçbiri
`signals`/`risk`/`execution` yolundan çağrılmaz.

## Bu modülün var oluş nedeni tek cümlede

Depodaki kanıt kapıları (`purged_cv` · `pbo` · `deflated` · `edge_lab.harness`) TEK bir
OOS PATİKASI üretir ve etiketi HORIZON-BAZLI getiri işareti alır. İkisi de bilinçli
sadeleştirmelerdi; ikisinin de bilinen bedeli var:

* Tek patika = tek bir bölmenin şansı. CPCV `C(N,k)` bölmeden **çok sayıda** OOS patikası
  kurar ve PBO'yu doğrudan besler (López de Prado Ch.12).
* Horizon etiketi, stop'a ÖNCE dokunan bir işlemi kazanan sayabilir. Üçlü bariyer
  ilk temas noktasına göre etiketler — sistemin CANLI çıkış mantığına daha yakındır.

**İKİSİ DE ÜRETİM KAPISINI DEĞİŞTİRMEZ.** `edge_lab.harness`'ın etiketi ve fold üreticisi
DOKUNULMADAN bırakıldı: bir kanıt kapısını, geçmiş 9 ailenin verdikti yeniden
ölçülmeden değiştirmek, sonuçları kıyaslanamaz kılardı (Tur-23/Tur-31 disiplini).
Buradakiler çağıranın AÇIKÇA seçtiği araçlardır.

Sözleşme `quant_math` ile aynı: geçersiz/kısa girdi → None, istisna YOK.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from numpy.typing import ArrayLike

# Adversarial validation'da "kayma var" eşiği — literatürün yaygın çıtası.
ADV_AUC_THRESHOLD = 0.65
# Mahalanobis OOD ki-kare güven düzeyi.
OOD_ALPHA = 0.001


def _clean_1d(x: ArrayLike, min_len: int = 1) -> np.ndarray | None:
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < min_len or not np.all(np.isfinite(arr)):
        return None
    return arr


def _clean_2d(x: ArrayLike, min_rows: int = 2, min_cols: int = 1) -> np.ndarray | None:
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < min_rows or arr.shape[1] < min_cols:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    """Sıra-tabanlı AUC (beraberliklerde ortalama sıra). Tek sınıflı hedefte None."""
    n1 = float(np.sum(y == 1))
    n0 = float(y.size) - n1
    if n1 == 0.0 or n0 == 0.0:
        return None
    sira = np.empty(p.size, dtype=float)
    duzen = np.argsort(p, kind="mergesort")
    sirali = p[duzen]
    i = 0
    while i < sirali.size:
        j = i
        while j + 1 < sirali.size and sirali[j + 1] == sirali[i]:
            j += 1
        sira[duzen[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((np.sum(sira[y == 1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ---------------------------------------------------------------------------
# Madde 127 — Üçlü bariyer etiketleme
# ---------------------------------------------------------------------------
def triple_barrier_labels(prices: ArrayLike, events: ArrayLike | None = None,
                          upper: float = 2.0, lower: float = 1.0,
                          vertical: int = 20,
                          volatility: ArrayLike | None = None,
                          side: ArrayLike | None = None) -> dict | None:
    """İlk TEMAS noktasına göre {+1, −1, 0} etiketi (López de Prado Ch.3 / Madde 127).

    Bariyerler oynaklık ÖLÇEKLİDİR: üst = +`upper`·σ_t, alt = −`lower`·σ_t, dikey =
    `vertical` bar. `volatility` verilmezse 20 barlık geriye-bakan getiri std'si
    kullanılır (NEDENSEL: t barındaki σ yalnız t'ye kadarki veriden).

    `side` verilirse (LONG=+1 / SHORT=−1) bariyerler yöne göre ÇEVRİLİR ve etiket
    KÂR/ZARAR anlamına gelir (meta-labeling kurulumu); verilmezse etiket YÖN anlamına
    gelir (üst bariyer = +1).

    **Dikey bariyerin etiketi 0'dır, getirinin işareti DEĞİL.** Yaygın varyant zaman
    aşımını da getirinin işaretiyle etiketler; o varyant "hiçbir bariyere değmedi"
    bilgisini yok eder ve üç-durumlu etiketi ikiliye çökertir. Getiri ayrıca
    `ret` kolonunda taşınır — çağıran isterse kendisi çökertir.

    Son `vertical` bar için bariyerlerin TAMAMI görülemez; o satırlar `truncated=True`
    ile İŞARETLENİR ve etiketleri KISMİ bilgiyle verilir (dışlamak, en taze veriyi
    sessizce atmak olurdu — sayı `n_truncated`ta durur).
    """
    p = _clean_1d(prices, 5)
    if p is None or np.any(p <= 0.0):
        return None
    try:
        vb = int(vertical)
        up, dn = float(upper), float(lower)
    except (TypeError, ValueError):
        return None
    if vb < 1 or up <= 0.0 or dn <= 0.0:
        return None
    n = p.size
    logp = np.log(p)

    if volatility is None:
        r = np.diff(logp, prepend=logp[0])
        sig = np.full(n, np.nan)
        w = 20
        if n > w:
            win = np.lib.stride_tricks.sliding_window_view(r, w)
            sig[w - 1:] = win.std(axis=1, ddof=1)
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
    else:
        v = _clean_1d(volatility, n)
        if v is None or v.size != n:
            return None
        sig = np.where(v > 0, v, np.nan)

    if events is None:
        ev = np.flatnonzero(np.isfinite(sig))
    else:
        e = np.asarray(events).reshape(-1)
        if e.dtype == bool:
            ev = np.flatnonzero(e)
        else:
            try:
                ev = np.asarray(e, dtype=int)
            except (TypeError, ValueError):
                return None
        ev = ev[(ev >= 0) & (ev < n) & np.isfinite(sig[np.clip(ev, 0, n - 1)])]
    if ev.size == 0:
        return None

    if side is None:
        yon = np.ones(ev.size)
    else:
        s = _clean_1d(side, 1)
        if s is None or s.size != ev.size:
            return None
        yon = np.sign(s)
        if np.any(yon == 0):
            return None

    etiket = np.zeros(ev.size, dtype=int)
    dokunma = np.zeros(ev.size, dtype=int)
    getiri = np.zeros(ev.size, dtype=float)
    kirpik = np.zeros(ev.size, dtype=bool)
    for k, i_ in enumerate(ev):
        i = int(i_)
        son = int(min(i + vb, n - 1))
        kirpik[k] = (i + vb) > n - 1
        yol = yon[k] * (logp[i + 1: son + 1] - logp[i])
        ust_b, alt_b = up * sig[i], -dn * sig[i]
        ust_i = np.flatnonzero(yol >= ust_b)
        alt_i = np.flatnonzero(yol <= alt_b)
        t_ust = int(ust_i[0]) if ust_i.size else np.iinfo(np.int32).max
        t_alt = int(alt_i[0]) if alt_i.size else np.iinfo(np.int32).max
        if t_ust == t_alt == np.iinfo(np.int32).max:
            etiket[k], dokunma[k] = 0, son - i
        elif t_ust <= t_alt:            # eşitlikte ÜST önce — belgelenmiş konvansiyon
            etiket[k], dokunma[k] = 1, t_ust + 1
        else:
            etiket[k], dokunma[k] = -1, t_alt + 1
        getiri[k] = float(yol[dokunma[k] - 1]) if dokunma[k] >= 1 and yol.size else 0.0
    return {
        "events": ev,
        "label": etiket,
        "ret": getiri,
        "bars_held": dokunma,
        "truncated": kirpik,
        "n_truncated": int(kirpik.sum()),
        "sigma": sig[ev],
        "counts": {int(v): int(np.sum(etiket == v)) for v in (-1, 0, 1)},
        "n": int(ev.size),
    }


# ---------------------------------------------------------------------------
# Madde 126 — Kombinatoryal purged CV
# ---------------------------------------------------------------------------
def cpcv_splits(n: int, n_groups: int = 6, n_test: int = 2,
                embargo: int = 0, purge: int | None = None):
    """C(N, k) kombinasyonlu purged CV bölmeleri (López de Prado Ch.12 / Madde 126).

    `purged_folds` her bloğu BİR KEZ test eder (N bölme, 1 OOS patikası). CPCV k bloğu
    BİRLİKTE test eder → C(N,k) bölme ve `C(N,k)·k/N` OOS patikası. 6 grup / 2 test ile
    15 bölme, 5 patika.

    Purge/embargo `purged_folds` ile AYNI sözleşmededir ve her test bloğunun İKİ
    yanına da uygulanır — bloklar bitişik olabileceği için (ör. {2,3}) sınır iç
    tarafta kalır ve orada purge GEREKMEZ; kod bunu blok BİRLEŞİMİ üzerinden
    hesaplar, blok blok değil (blok blok yapan bir uygulama iç sınırı gereksiz yere
    keser ve eğitim verisini sessizce küçültür).
    """
    try:
        N, k, emb = int(n), int(n_groups), int(embargo)
        kt = int(n_test)
    except (TypeError, ValueError):
        return None
    if N < k or k < 2 or kt < 1 or kt >= k or emb < 0:
        return None
    sol = emb if purge is None else max(int(purge), 0)
    sinir = np.linspace(0, N, k + 1).astype(int)
    bolmeler = []
    for kombo in itertools.combinations(range(k), kt):
        maske_test = np.zeros(N, dtype=bool)
        for g in kombo:
            maske_test[sinir[g]:sinir[g + 1]] = True
        # Yasak bant: her test satırının SOLUNA `sol`, SAĞINA `emb` kadar.
        yasak = maske_test.copy()
        idx = np.flatnonzero(maske_test)
        for i in idx:
            yasak[max(0, i - sol): i] = True
            yasak[i + 1: min(N, i + 1 + max(emb, sol))] = True
        train = np.flatnonzero(~yasak)
        bolmeler.append((train, np.flatnonzero(maske_test), tuple(kombo)))
    return bolmeler


def cpcv_paths(n_groups: int = 6, n_test: int = 2) -> dict | None:
    """CPCV'nin ürettiği OOS PATİKA sayısı ve her grubun kaç bölmede test edildiği.

    Patika sayısı = C(N,k)·k/N. Bu sayı PBO'ya doğrudan girer: tek patikalı bir
    kapı, aşırı-uyum olasılığını ÖLÇEMEZ (yalnız tahmin eder).
    """
    try:
        k, kt = int(n_groups), int(n_test)
    except (TypeError, ValueError):
        return None
    if k < 2 or kt < 1 or kt >= k:
        return None
    n_split = math.comb(k, kt)
    return {
        "n_splits": n_split,
        "n_paths": n_split * kt // k,
        "tests_per_group": n_split * kt // k,
        "n_groups": k,
        "n_test": kt,
    }


# ---------------------------------------------------------------------------
# Madde 129 — Düşmanca doğrulama
# ---------------------------------------------------------------------------
def _logistic_fit(X: np.ndarray, y: np.ndarray, iters: int = 200,
                  lr: float = 0.5, l2: float = 1e-3) -> np.ndarray:
    """Küçük, bağımlılıksız lojistik (Newton'suz gradyan inişi + L2)."""
    n, p = X.shape
    Xb = np.column_stack([np.ones(n), X])
    w = np.zeros(p + 1)
    for _ in range(iters):
        z = np.clip(Xb @ w, -30.0, 30.0)
        pr = 1.0 / (1.0 + np.exp(-z))
        grad = Xb.T @ (pr - y) / n + l2 * np.r_[0.0, w[1:]]
        w -= lr * grad
    return w


def adversarial_validation(train: ArrayLike, live: ArrayLike,
                           n_folds: int = 3, seed: int = 0) -> dict | None:
    """Eğitim ↔ canlı özellik dağılımı AYRILABİLİR mi? (Madde 129)

    Canlıya 1, eğitime 0 etiketi verilir ve bir sınıflandırıcı **ÇAPRAZ DOĞRULAMAYLA**
    ikisini ayırmaya çalışır. AUC ≈ 0,5 = dağılımlar ayırt edilemez (iyi haber).
    AUC > `ADV_AUC_THRESHOLD` = REJİM KAYMASI uyarısı.

    **AUC iç-örneklemde değil OOS'ta ölçülür** — iç-örneklem AUC'si yeterince esnek
    bir modelle HER ZAMAN 1,0'a gider ve "kayma var" yanlış alarmını garanti ederdi.

    `per_feature`: her özellik TEK BAŞINA ne kadar ayırıyor (tek-değişkenli AUC) —
    kaymanın hangi kolondan geldiğini söyler. Toplu AUC yüksek ama tek tek hepsi
    0,5 ise kayma BİRLEŞİK dağılımdadır (korelasyon yapısı değişmiştir).
    """
    a = _clean_2d(train, 10, 1)
    b = _clean_2d(live, 10, 1)
    if a is None or b is None or a.shape[1] != b.shape[1]:
        return None
    try:
        kf = int(n_folds)
    except (TypeError, ValueError):
        return None
    if kf < 2:
        return None
    X = np.vstack([a, b])
    y = np.r_[np.zeros(a.shape[0]), np.ones(b.shape[0])]
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Xs = (X - mu) / np.where(sd > 0, sd, 1.0)
    rng = np.random.default_rng(seed)
    duzen = rng.permutation(X.shape[0])
    katlar = np.array_split(duzen, kf)
    oos_p = np.full(X.shape[0], np.nan)
    for t in range(kf):
        te = katlar[t]
        tr = np.concatenate([katlar[j] for j in range(kf) if j != t])
        if np.unique(y[tr]).size < 2:
            continue
        w = _logistic_fit(Xs[tr], y[tr])
        oos_p[te] = 1.0 / (1.0 + np.exp(-np.clip(
            np.column_stack([np.ones(te.size), Xs[te]]) @ w, -30.0, 30.0)))
    ok = np.isfinite(oos_p)
    if ok.sum() < 10:
        return None
    auc = _auc(y[ok].astype(int), oos_p[ok])
    if auc is None:
        return None
    tekil = []
    for j in range(X.shape[1]):
        aj = _auc(y.astype(int), Xs[:, j])
        tekil.append(None if aj is None else float(max(aj, 1.0 - aj)))
    return {
        "auc": float(auc),
        "shift_detected": bool(auc > ADV_AUC_THRESHOLD),
        "threshold": ADV_AUC_THRESHOLD,
        "per_feature": tekil,
        "worst_feature": (int(np.argmax([x if x is not None else 0.0 for x in tekil]))
                          if tekil else None),
        "n_train": int(a.shape[0]), "n_live": int(b.shape[0]),
    }


# ---------------------------------------------------------------------------
# Madde 130 — Kümeli özellik önemi (clustered MDA)
# ---------------------------------------------------------------------------
def _corr_clusters(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Korelasyon uzaklığı üzerinde basit hiyerarşik (tek-bağlantı) kümeleme."""
    p = X.shape[1]
    c = np.corrcoef(X.T)
    c = np.where(np.isfinite(c), c, 0.0)
    d = np.sqrt(np.clip(2.0 * (1.0 - c), 0.0, None))
    np.fill_diagonal(d, np.inf)
    etiket = np.arange(p)
    for _ in range(max(0, p - n_clusters)):
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        if not np.isfinite(d[i, j]):
            break
        eski, yeni = etiket[j], etiket[i]
        etiket[etiket == eski] = yeni
        d[i, :] = np.minimum(d[i, :], d[j, :])
        d[:, i] = d[i, :]
        d[j, :] = np.inf
        d[:, j] = np.inf
        d[i, i] = np.inf
    _, tekil = np.unique(etiket, return_inverse=True)
    return tekil


def clustered_mda(X: ArrayLike, y: ArrayLike, n_clusters: int = 3,
                  n_repeats: int = 5, seed: int = 0) -> dict | None:
    """Küme BLOKLARINI birlikte karıştırarak permütasyon önemi (Madde 130).

    Tek tek permütasyon, birbirinin yerine geçebilen (yüksek korelasyonlu) iki
    özelliğin ÖNEMİNİ SIFIR gösterir: biri karıştırıldığında model diğerini okur.
    Küme bloğu birlikte karıştırılınca bilgi GERÇEKTEN kaybolur.

    Ölçüt OOS AUC düşüşüdür (`baseline_auc − permuted_auc`); model küçük bir
    lojistiktir — amacı en iyi tahmin DEĞİL, karşılaştırma için sabit bir referanstır.
    """
    Xa = _clean_2d(X, 20, 2)
    ya = _clean_1d(y, 20)
    if Xa is None or ya is None or Xa.shape[0] != ya.size:
        return None
    yb = np.asarray(ya > 0.5, dtype=float)
    if np.unique(yb).size < 2:
        return None
    try:
        nc, nr = int(n_clusters), int(n_repeats)
    except (TypeError, ValueError):
        return None
    p = Xa.shape[1]
    if nc < 1 or nc > p or nr < 1:
        return None
    n = Xa.shape[0]
    kesme = int(n * 0.7)
    if kesme < 10 or n - kesme < 10:
        return None
    mu, sd = Xa[:kesme].mean(axis=0), Xa[:kesme].std(axis=0)
    Xs = (Xa - mu) / np.where(sd > 0, sd, 1.0)
    if np.unique(yb[:kesme]).size < 2 or np.unique(yb[kesme:]).size < 2:
        return None
    w = _logistic_fit(Xs[:kesme], yb[:kesme])

    def _skor(M: np.ndarray) -> float | None:
        z = np.clip(np.column_stack([np.ones(M.shape[0]), M]) @ w, -30.0, 30.0)
        return _auc(yb[kesme:].astype(int), 1.0 / (1.0 + np.exp(-z)))

    taban = _skor(Xs[kesme:])
    if taban is None:
        return None
    kume = _corr_clusters(Xs[:kesme], nc)
    rng = np.random.default_rng(seed)
    onem: dict[int, float] = {}
    for c in np.unique(kume):
        sut = np.flatnonzero(kume == c)
        dususler = []
        for _ in range(nr):
            M = Xs[kesme:].copy()
            perm = rng.permutation(M.shape[0])
            M[:, sut] = M[perm][:, sut]     # BLOK halinde, aynı satır düzeniyle
            s = _skor(M)
            if s is not None:
                dususler.append(taban - s)
        onem[int(c)] = float(np.mean(dususler)) if dususler else float("nan")
    return {
        "baseline_auc": float(taban),
        "clusters": kume,
        "cluster_importance": onem,
        "n_clusters": int(np.unique(kume).size),
        "n_train": kesme, "n_test": int(n - kesme),
    }


# ---------------------------------------------------------------------------
# Madde 132 — Mahalanobis OOD
# ---------------------------------------------------------------------------
def mahalanobis_ood(reference: ArrayLike, query: ArrayLike,
                    alpha: float = OOD_ALPHA) -> dict | None:
    """D_M² > χ²_{p,1−α} ⇒ dağılım DIŞI (Madde 132).

    `neural_shadow`ın mevcut OOD guard'ı KOLON BAŞINA z-skoru okur (`max|z|`); o kural
    özellikler arası KORELASYON YAPISINI göremez — her kolonu tek tek normal olan bir
    nokta, ortak dağılımda imkânsız olabilir. Mahalanobis tam bunu ölçer.

    Kovaryans tekil/kötü koşulluysa **Ledoit-Wolf tipi büzme** uygulanır ve
    `shrinkage` çıktıda taşınır (sessizce sözde-ters almak, ölçüm yokluğunu ölçüm
    gibi sunmak olurdu).
    """
    R = _clean_2d(reference, 5, 1)
    Q = _clean_2d(query, 1, 1)
    if R is None or Q is None or R.shape[1] != Q.shape[1]:
        return None
    try:
        a = float(alpha)
    except (TypeError, ValueError):
        return None
    if not (0.0 < a < 1.0):
        return None
    p = R.shape[1]
    mu = R.mean(axis=0)
    S = np.cov(R.T, ddof=1)
    S = np.atleast_2d(S)
    iz = float(np.trace(S)) / p if p > 0 else 0.0
    buzme = 0.0
    for lam in (0.0, 0.01, 0.1, 0.3, 0.6, 0.9):
        Sl = (1.0 - lam) * S + lam * iz * np.eye(p)
        if np.linalg.cond(Sl) < 1e10:
            buzme = lam
            S = Sl
            break
    else:
        return None
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:                # pragma: no cover - fail-soft
        return None
    d = Q - mu
    d2 = np.einsum("ij,jk,ik->i", d, Sinv, d)
    # χ² kuantili: Wilson-Hilferty yaklaşımı (scipy'siz, p≥1 için ≤%1 hata).
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1.0 - a)
    esik = p * (1.0 - 2.0 / (9.0 * p) + z * math.sqrt(2.0 / (9.0 * p))) ** 3
    return {
        "d2": d2,
        "distance": np.sqrt(np.clip(d2, 0.0, None)),
        "threshold": float(esik),
        "is_ood": d2 > esik,
        "ood_share": float(np.mean(d2 > esik)),
        "shrinkage": buzme,
        "dim": p, "n_reference": int(R.shape[0]), "n_query": int(Q.shape[0]),
    }


# ---------------------------------------------------------------------------
# Madde 133 — Özellik nötrleştirme
# ---------------------------------------------------------------------------
def neutralize(features: ArrayLike, exposures: ArrayLike,
               proportion: float = 1.0) -> dict | None:
    """Özellikleri maruziyet(ler)e (ör. piyasa getirisi) karşı REGRESE ET, artığı al.

    `proportion=1` tam nötrleştirmedir; 0,5 yarı. Nötrleştirilmiş özelliğin maruziyetle
    korelasyonu YAPISAL OLARAK 0'dır ve test bunu makine hassasiyetinde arar.

    **KORUNAN ŞEY SABİT TERİMDİR, ÖRNEKLEM ORTALAMASI DEĞİL.** Tam artık almak
    (`F − uyum`) ortalaması 0 olan bir seri verir ve özelliği ölçeğinden koparır;
    burada yalnız MARUZİYET bileşeni çıkarılır, yani çıktının ortalaması "maruziyet
    sıfırken beklenen değer"e (β₀) oturur. Maruziyetin ortalaması 0 DEĞİLSE β₀ ile
    `mean(F)` FARKLIDIR (ölçüldü: β₀ = mean(F) − β·mean(E)) ve ikisini karıştırmak
    çıktıyı sistematik olarak kaydırır.

    **Nötrleştirme BİLGİ SİLER.** Piyasa yönü gerçekten öngörü taşıyorsa, onu çıkarmak
    modeli KÖTÜLEŞTİRİR. Ölçüt "korelasyon düştü mü" değil, "OOS beklentisi düzeldi mi"
    olmalıdır (ACI DERS #3). Bu fonksiyon yalnız DÖNÜŞÜMÜ yapar, iddiayı değil.
    """
    F = _clean_2d(features, 3, 1)
    if F is None:
        return None
    E_ham = np.asarray(exposures, dtype=float)
    if E_ham.ndim == 1:
        E_ham = E_ham.reshape(-1, 1)
    E = _clean_2d(E_ham, 3, 1)
    if E is None or E.shape[0] != F.shape[0]:
        return None
    try:
        pr = float(proportion)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= pr <= 1.0):
        return None
    X = np.column_stack([np.ones(E.shape[0]), E])
    try:
        beta, *_ = np.linalg.lstsq(X, F, rcond=None)
    except np.linalg.LinAlgError:                # pragma: no cover - fail-soft
        return None
    uyum = X @ beta
    # Sabit terim korunur: yalnız MARUZİYET bileşeni çıkarılır, ortalama değil.
    sabit = beta[0][None, :]
    out = F - pr * (uyum - sabit)
    return {
        "neutral": out,
        "beta": np.asarray(beta[1:], dtype=float),
        "proportion": pr,
        "explained_var": float(np.mean(1.0 - np.var(F - uyum, axis=0)
                                       / np.where(np.var(F, axis=0) > 0,
                                                  np.var(F, axis=0), 1.0))),
        "n": int(F.shape[0]), "n_features": int(F.shape[1]),
    }


# ---------------------------------------------------------------------------
# Madde 146 — Fraksiyonel Brown hareketi (sentetik stres)
# ---------------------------------------------------------------------------
def fbm_paths(n: int, hurst: float = 0.5, n_paths: int = 1,
              seed: int = 0) -> np.ndarray | None:
    """Davies-Harte ile TAM (yaklaşık değil) fBM patikaları (Madde 146).

    H > 0,5 kalıcı/trendli, H = 0,5 klasik rassal yürüyüş, H < 0,5 ortalamaya dönen.
    Stratejinin hangi Hurst bölgesinde kırıldığını haritalamak için üretilir.

    Davies-Harte, dairesel gömme özdeğerlerinin NEGATİF çıkması durumunda geçersizdir;
    o durumda **None döner** (negatifleri kırpıp "yaklaşık fBM" üretmek, üretilen
    serinin Hurst'ünü sessizce değiştirirdi). Ölçüldü: 0,05 ≤ H ≤ 0,95'te sorun yok.
    """
    try:
        N, np_ = int(n), int(n_paths)
        H = float(hurst)
    except (TypeError, ValueError):
        return None
    if N < 4 or np_ < 1 or not (0.0 < H < 1.0):
        return None
    k = np.arange(N)
    # fGn otokovaryansı: γ(k) = ½(|k+1|^{2H} − 2|k|^{2H} + |k−1|^{2H})
    g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H)
               + np.abs(k - 1) ** (2 * H))
    dolasim = np.r_[g, g[-2:0:-1]]
    ozdeger = np.fft.fft(dolasim).real
    if np.any(ozdeger < -1e-10):
        return None
    ozdeger = np.clip(ozdeger, 0.0, None)
    m = dolasim.size
    rng = np.random.default_rng(seed)
    yollar = np.empty((np_, N))
    for i in range(np_):
        z = rng.normal(size=m) + 1j * rng.normal(size=m)
        w = np.fft.fft(np.sqrt(ozdeger / (2.0 * m)) * z)
        fgn = np.real(w[:N]) * math.sqrt(2.0)
        yollar[i] = np.cumsum(fgn)
    return yollar


# ---------------------------------------------------------------------------
# Madde 147 — Katman-dışı hedef kodlama
# ---------------------------------------------------------------------------
def oof_target_encode(categories: ArrayLike, target: ArrayLike, n_folds: int = 5,
                      smoothing: float = 10.0, seed: int = 0) -> dict | None:
    """Kategorik değişkeni hedef ortalamasıyla kodla — SIZINTISIZ (Madde 147).

    Her satırın kodu, o satırın BULUNMADIĞI katlardan hesaplanır. Naif kodlama
    (tüm veriden ortalama) satırın KENDİ etiketini kendi özelliğine yazar; nadir bir
    kategoride bu, etiketin kopyasıdır ve OOS'ta çöker.

    Büzme: `(toplam + smoothing·genel_ortalama) / (sayım + smoothing)` — az gözlemli
    kategori genel ortalamaya çekilir. Eğitimde GÖRÜLMEYEN kategori genel ortalamayı
    alır (NaN değil: eksik kod, aşağı akıştaki her modeli ayrıca kırardı).

    `full_map` PRODÜKSİYON kodudur (tüm veriden, sızıntı sorunu yok çünkü canlı satır
    eğitim setinde değildir); `encoded` ise EĞİTİM için kullanılacak OOS koddur. İkisi
    farklıdır ve karıştırmak tam da bu maddenin engellediği hatadır.
    """
    c = np.asarray(categories).reshape(-1)
    y = _clean_1d(target, 5)
    if y is None or c.size != y.size:
        return None
    try:
        kf = int(n_folds)
        sm = float(smoothing)
    except (TypeError, ValueError):
        return None
    if kf < 2 or sm < 0.0 or y.size < kf:
        return None
    kodlar, ters = np.unique(c, return_inverse=True)
    genel = float(y.mean())
    rng = np.random.default_rng(seed)
    katlar = np.array_split(rng.permutation(y.size), kf)
    kodlanmis = np.full(y.size, genel)
    for t in range(kf):
        te = katlar[t]
        tr = np.concatenate([katlar[j] for j in range(kf) if j != t])
        toplam = np.bincount(ters[tr], weights=y[tr], minlength=kodlar.size)
        sayim = np.bincount(ters[tr], minlength=kodlar.size).astype(float)
        tr_ort = float(y[tr].mean())
        # sm=0 VE sayım=0 → 0/0. Bölmeye SOKULMADAN kat ortalamasına düşülür
        # (uyarı bastırmak yerine tanımsız hesabı hiç yapmamak).
        payda = sayim + sm
        harita = np.where(payda > 0, (toplam + sm * tr_ort) / np.where(payda > 0, payda, 1.0),
                          tr_ort)
        harita[sayim == 0] = tr_ort
        kodlanmis[te] = harita[ters[te]]
    toplam = np.bincount(ters, weights=y, minlength=kodlar.size)
    sayim = np.bincount(ters, minlength=kodlar.size).astype(float)
    tam = (toplam + sm * genel) / (sayim + sm)
    return {
        "encoded": kodlanmis,
        "full_map": {kodlar[i]: float(tam[i]) for i in range(kodlar.size)},
        "prior": genel,
        "counts": {kodlar[i]: int(sayim[i]) for i in range(kodlar.size)},
        "n_categories": int(kodlar.size), "n": int(y.size), "smoothing": sm,
    }
