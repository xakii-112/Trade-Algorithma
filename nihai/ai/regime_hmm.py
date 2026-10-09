"""
regime_hmm.py — REJİM GEÇİŞ OLASILIKLARI (Gauss saklı Markov modeli). KARAR-DIŞI/GÖLGE.

NEDEN BU MODÜL VAR: `knowledge.py`'nin KENDİ kataloğunda, `classify_regime` kavramının
sınırlamaları arasında şu satır yazılıdır —

    "Rejim etiketi KATEGORİKTİR (ör. 'up_high') — geçiş OLASILIĞI
     (ör. '%70 ihtimalle yeni rejimdeyiz') bilgisi mevcut modülde yoktur."

Sistem rejimi her zaman KESKİN bir etiketle okudu: bir bar "up_high"dır ya da değildir.
Oysa rejim gözlenmez, ÇIKARSANIR; keskin etiket belirsizliği sessizce sıfıra yuvarlar ve
rejim SINIRLARINDA (tam da riskin arttığı yerde) en çok yanılan okuma budur. HMM üç şeyi
birden verir: (1) her bar için rejim OLASILIK DAĞILIMI, (2) rejimler arası GEÇİŞ matrisi,
(3) bir rejimin beklenen SÜRESİ. Üçü de "hangi rejimdeyiz" sorusunu "ne kadar eminiz ve ne
kadar sürer" sorusuna çevirir.

⚠️ LOOK-AHEAD SINIRI — bu modülün EN ÖNEMLİ sözleşmesi (Çalışma ilkeleri: look-ahead yasak):

  · `hmm_filter`  → **NEDENSELDİR**. P(S_t | x_1..x_t): yalnız t'ye kadarki veri. Canlı
                    kullanıma UYGUN olan tek çıktı budur.
  · `hmm_smooth`  → **LOOK-AHEAD İÇERİR**. P(S_t | x_1..x_T) tüm seriyi görür. GEÇMİŞ
                    analizi/etiketleme için doğru araçtır; sinyal/özellik hesabında
                    KULLANILAMAZ. Dönüşünde `causal=False` bayrağı TAŞINIR.
  · `hmm_viterbi` → LOOK-AHEAD içerir (en olası YOL tüm seriye bakar). Aynı kural.
  · `gaussian_hmm_fit` parametreleri TÜM örneklemden tahmin eder. Canlı kullanımda
    model bir EĞİTİM penceresinde fit edilmeli, sonra `hmm_filter` ile İLERİ
    uygulanmalıdır — `train_meta`/`purged_cv` disiplininin HMM karşılığı.

DETERMİNİZM (anayasa madde 6): ilklendirme QUANTILE tabanlıdır, rastgelelik YOKTUR —
`seed` argümanı da yoktur. Aynı girdi her ortamda aynı parametreleri verir.

SAVUNMACI SÖZLEŞME (modül ailesinin ortak kuralı): kısa/NaN/dejenere girdi → `None`.
Uydurma parametre, uydurma olasılık YOK.
"""
from __future__ import annotations

import copy
import math
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

# Varyans tabanı: bir durum tek bir gözlemin üstüne çökerse (σ→0) olabilirlik SONSUZA
# gider ve EM anlamsız bir "kusursuz uyum" bulur — Gauss karışımlarının klasik dejenere
# çözümü. Taban, serinin KENDİ std'sinin bu oranıdır (mutlak sabit değil: ölçek-değişmez).
_VAR_FLOOR_FRAC = 1e-3
# Geçiş matrisi tabanı: gözlenmemiş geçiş 0 olursa log(0) = −inf ve o geçiş SONSUZA DEK
# imkânsız ilan edilir. Küçük taban, "hiç görülmedi" ile "imkânsız" ayrımını korur.
_TRANS_FLOOR = 1e-8
# Uyum önbelleği (katalog #26). Pano tek sembol/pencere yoklar; 8 giriş üç ayrı
# durum sayısı × birkaç pencereyi rahat taşır ve bellek maliyeti ihmal edilebilir
# (girdi başına T·8 bayt, T=500 → 4 KB).
_FIT_CACHE_MAX = 8


def _lse(a: np.ndarray, axis: int | tuple[int, ...] | None = None,
         keepdims: bool = False) -> Any:
    """`scipy.special.logsumexp`in yerel, dar kapsamlı karşılığı — max-kaydırmalı.

    NEDEN (Tur-34 profili): ileri-geri özyinelemesi gözlem BAŞINA çağrı yapar; T=2000,
    K=2, 20 EM adımında **86.021 çağrı** ölçüldü ve profilin tamamına yakını scipy'nin
    array-API gönderimi + dtype yükseltme sarmalayıcısındaydı — matematikte DEĞİL. Aynı
    ders Tur-33'te `spearman`'da öğrenilmişti: pahalı olan formül değil, çağrı başına
    kurulan altyapı.

    Sözleşme scipy ile aynı (aynı max-kaydırma aritmetiği): tüm elemanlar −inf ise sonuç
    −inf'tir (m sonlu değilse 0'a çekilir, exp(−inf)=0, log(0)=−inf)."""
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):     # tüm-(−inf) satırda log(0) = −inf BEKLENEN
        out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if keepdims:
        return out
    if axis is None:
        return float(out.reshape(-1)[0])
    return np.squeeze(out, axis=axis)


def _clean_1d(x: ArrayLike, min_len: int = 1) -> np.ndarray | None:
    """1B sonlu float dizi (NaN/inf içeren veya kısa girdi → None)."""
    try:
        arr = np.asarray(x, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size < min_len or not np.all(np.isfinite(arr)):
        return None
    return arr


def _log_emissions(x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """(T,) gözlem × (K,) Gauss durum → (T, K) log olasılık yoğunluğu."""
    z = (x[:, None] - means[None, :]) / stds[None, :]
    return -0.5 * z * z - np.log(stds)[None, :] - 0.5 * math.log(2.0 * math.pi)


def _forward(log_emis: np.ndarray, log_start: np.ndarray,
             log_trans: np.ndarray) -> tuple[np.ndarray, float]:
    """İleri özyineleme LOG uzayında (ölçekleme yerine max-kaydırma — taşma/alt-taşma yok).
    Dönüş: (log_alpha (T,K), log-olabilirlik).

    max-kaydırma `_lse` ÇAĞRILMADAN yerinde yapılır: bu döngü gözlem başına bir kez döner
    (T=2000 · 20 EM adımı = 40.000 kez) ve ölçüldü — `_lse` sarmalayıcısının kendisi çağrı
    başına 20,9 µs, açılmış hâli 11,2 µs (scipy.logsumexp: 151 µs). Aritmetik AYNI; tasarruf
    tamamen çağrı/gönderim katmanındandır. Ölçekleme değerleri her zaman sonludur
    (`_TRANS_FLOOR` sıfır olasılığı engeller) → `_lse`'nin isfinite koruması burada gereksiz.

    İKİNCİ ADIM (aynı tur): kaydırma SKALER seçilir (satırın kendi maksimumu). Bu, (K,K)
    matris üzerinde eksen-boyu max+exp+sum yerine tek bir MATRİS-VEKTÖR çarpımı bırakır:
        α_t[i] = m + log( Σ_j exp(α_{t−1}[j] − m)·A[j,i] ) = m + log( (e·A)[i] )
    exp(·−m) ∈ (0,1] olduğu için taşma yoktur ve A'nın tabanı (>0) toplamı sıfırdan uzak
    tutar. BLAS'a devredilen çarpım, Python katmanındaki üç dizi işlemini birden siler.

    ÜÇÜNCÜ ADIM (profil yine yol gösterdi): kalan maliyetin tamamı adım başına `max`/`exp`
    çağrılarıydı (T·EM_adımı kadar; ölçüldü: yalnız `.max()` 0,97 s). Rabiner (1989 §V.A)
    ÖLÇEKLEME şemasına geçildi — özyineleme normalize LİNEER uzayda döner, ölçek çarpanı
    ayrı bir log toplamında biriktirilir:
        â_t ∝ (â_{t−1}·A) ⊙ b̂_t ,  b̂ = exp(log_emis − satır maks.)  [DÖNGÜ DIŞINDA, tek çağrı]
        log α_t = log â_t + Σ_{u≤t} log c_u
    Böylece döngü içinde `max` ve `exp` KALMAZ (adım başına: 1 matvec + 1 çarpım + 1 toplam
    + 1 bölme + 1 skaler log). Alt-taşma imkânsızdır: â her adımda 1'e normalize edilir.
    Kayan-nokta toplama sırası değiştiği için sonuç bit-denk DEĞİLDİR; denklik ÖLÇÜLMÜŞTÜR
    (bkz. tests/test_ai_regime_hmm.py::test_olceklenmis_ozyineleme_log_uzayiyla_denk)."""
    t_len, k = log_emis.shape
    trans = np.exp(log_trans)                       # döngü DIŞINDA bir kez
    m_all = log_emis.max(axis=1)                    # (T,) — tek çağrı
    b_hat = np.exp(log_emis - m_all[:, None])       # (T,K) — tek çağrı
    m_start = float(log_start.max())
    a_hat = np.empty((t_len, k), dtype=float)
    log_c = np.empty(t_len, dtype=float)

    raw = np.exp(log_start - m_start) * b_hat[0]
    s = float(raw.sum())
    a_hat[0] = raw / s
    log_c[0] = m_start + float(m_all[0]) + math.log(s)
    for t in range(1, t_len):
        raw = (a_hat[t - 1] @ trans) * b_hat[t]
        s = float(raw.sum())
        a_hat[t] = raw / s
        log_c[t] = float(m_all[t]) + math.log(s)
    cum = np.cumsum(log_c)
    with np.errstate(divide="ignore"):
        log_alpha = np.log(a_hat) + cum[:, None]
    return log_alpha, float(cum[-1])


def _backward(log_emis: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    """Geri özyineleme — `_forward` ile AYNI ölçekleme şeması (β̂ her adımda normalize
    edilir, ölçek çarpanı geriye doğru biriken bir log toplamında tutulur)."""
    t_len, k = log_emis.shape
    trans = np.exp(log_trans)
    m_all = log_emis.max(axis=1)
    b_hat = np.exp(log_emis - m_all[:, None])
    beta_hat = np.empty((t_len, k), dtype=float)
    log_d = np.zeros(t_len, dtype=float)
    beta_hat[-1] = 1.0
    for t in range(t_len - 2, -1, -1):
        raw = trans @ (b_hat[t + 1] * beta_hat[t + 1])
        s = float(raw.sum())
        beta_hat[t] = raw / s
        log_d[t] = float(m_all[t + 1]) + math.log(s) + log_d[t + 1]
    with np.errstate(divide="ignore"):
        return np.log(beta_hat) + log_d[:, None]


def _init_by_quantiles(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DETERMİNİSTİK ilklendirme: seri k eşit-olasılıklı dilime (quantile) bölünür, her
    dilimin ortalama/std'si bir durumun başlangıcı olur; geçiş matrisi bu KESKİN atamanın
    ampirik geçiş sayımlarından gelir. k-means++ veya rastgele ilklendirme yerine bu
    seçildi çünkü sonuç ORTAM-BAĞIMSIZ yeniden üretilebilir olmalıdır."""
    edges = np.quantile(x, np.linspace(0.0, 1.0, k + 1)[1:-1]) if k > 1 else np.array([])
    labels = np.searchsorted(edges, x, side="right") if k > 1 else np.zeros(x.size, dtype=int)
    means = np.empty(k, dtype=float)
    stds = np.empty(k, dtype=float)
    overall = float(np.std(x))
    floor = max(overall * _VAR_FLOOR_FRAC, 1e-12)
    for j in range(k):
        sel = x[labels == j]
        means[j] = float(np.mean(sel)) if sel.size else float(np.mean(x))
        stds[j] = max(float(np.std(sel)) if sel.size > 1 else overall, floor)
    trans = np.full((k, k), _TRANS_FLOOR, dtype=float)
    for a, b in zip(labels[:-1], labels[1:], strict=False):
        trans[a, b] += 1.0
    trans /= trans.sum(axis=1, keepdims=True)
    return means, stds, trans


def gaussian_hmm_fit(x: ArrayLike, n_states: int = 2, max_iter: int = 200,
                     tol: float = 1e-6) -> dict | None:
    """Gauss saklı Markov modelini Baum-Welch (EM) ile uydurur — DETERMİNİSTİK.

    `x` genelde LOG GETİRİ serisidir (fiyat DEĞİL: fiyat durağan değildir ve Gauss
    yayılımı anlamsızlaşır). En az 30 gözlem ve `n_states` ∈ [2, 6] ister.

    Dejenere çözüm KORUMASI: her M adımında σ, serinin kendi std'sinin `_VAR_FLOOR_FRAC`
    katının ALTINA inemez. Bu bir kozmetik değil ZORUNLULUKTUR — Gauss karışımlarında
    olabilirlik, bir bileşen tek gözlemin üstüne çöktüğünde SONSUZA gider ve EM o sahte
    optimuma kilitlenir.

    ⚠️ Parametreler TÜM örneklemden tahmin edilir → bu bir İÇ-ÖRNEKLEM uyumudur. Canlı
    kullanımda eğitim penceresinde fit edip `hmm_filter` ile ileri uygulayın.

    Dönüş: {'n_states','means','stds','transition','start','loglik','bic','n_iter',
    'converged','n'} · başarısız/dejenere → None.

    ÖNBELLEKLİ (dış katalog #26 — memoization). Bu fonksiyon SAF ve DETERMİNİSTİKTİR:
    ilklendirme quantile tabanlıdır, `seed` argümanı bile yoktur (bkz. modül başlığı) →
    aynı girdi her zaman aynı çıktıyı verir, yani memoize edilebilir olması modülün ZATEN
    verdiği bir sözdür. Önbellek ANAHTARI serinin baytlarıdır (`arr.tobytes()`): tek bir
    bar değişirse anahtar değişir, dolayısıyla bayat model DÖNEMEZ.

    ÖLÇÜLDÜ (`/api/quant/regime`, SPY 1d 500 bar): uç `hmm_select_states` üzerinden k=2,3,4
    için ÜÇ ayrı EM koşturur; profil 458 `_forward` + 455 `_backward` çağrısı gösterdi
    (fit başına ~152 EM adımı) ve ısınmış p50 **2334,8 ms** idi — panonun 55 GET ucunun
    toplam maliyetinin %75'i tek uçtaydı. Önbellekle tekrar çağrılar **~0 ms**.

    Neden ÇÖZÜM BURADA, uçta değil: `hmm_select_states` de, uç da, `research/` de aynı
    fit'i çağırır; kapıyı çağıranların her birine koymak aynı yamanın üç kopyası olurdu.
    """
    arr = _clean_1d(x, min_len=30)
    if arr is None:
        return None
    try:
        k = int(n_states)
    except (TypeError, ValueError):
        return None
    if not 2 <= k <= 6 or arr.size < 10 * k:
        return None
    if float(np.std(arr)) <= 0.0:
        return None            # sabit seri: rejim kavramı tanımsız
    out = _hmm_fit_cached(np.ascontiguousarray(arr, dtype=float).tobytes(),
                          k, int(max_iter), float(tol))
    # Önbellekteki sözlük PAYLAŞILIR; çağıran onu değiştirirse sonraki isabet kirlenirdi.
    return copy.deepcopy(out) if out is not None else None


@lru_cache(maxsize=_FIT_CACHE_MAX)
def _hmm_fit_cached(key: bytes, k: int, max_iter: int, tol: float) -> dict | None:
    """`gaussian_hmm_fit`in ÖNBELLEKLENEN gövdesi. `key` serinin ham float64 baytlarıdır
    (bit-birebir): `np.frombuffer` ile geri açılır, kopya değil GÖRÜNÜM döner ve dizi
    salt-okunurdur — EM yalnız türetilmiş diziler üzerinde yazar, girdiye dokunmaz."""
    arr = np.frombuffer(key, dtype=float)
    overall = float(np.std(arr))
    floor = max(overall * _VAR_FLOOR_FRAC, 1e-12)

    means, stds, trans = _init_by_quantiles(arr, k)
    start = np.full(k, 1.0 / k, dtype=float)
    prev_ll = -math.inf
    converged = False
    it = 0
    for step in range(1, int(max_iter) + 1):
        it = step
        log_emis = _log_emissions(arr, means, stds)
        log_trans = np.log(np.maximum(trans, _TRANS_FLOOR))
        log_start = np.log(np.maximum(start, _TRANS_FLOOR))
        log_alpha, ll = _forward(log_emis, log_start, log_trans)
        if not math.isfinite(ll):
            return None
        log_beta = _backward(log_emis, log_trans)
        log_gamma = log_alpha + log_beta
        log_gamma -= _lse(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        # xi: (T-1, K, K) ortak geçiş sorumluluğu — log uzayında kurulup normalize edilir.
        log_xi = (log_alpha[:-1, :, None] + log_trans[None, :, :]
                  + (log_emis[1:] + log_beta[1:])[:, None, :])
        log_xi -= _lse(log_xi, axis=(1, 2), keepdims=True)
        xi = np.exp(log_xi)

        start = gamma[0] / max(gamma[0].sum(), 1e-300)
        denom = gamma[:-1].sum(axis=0)[:, None]
        trans = xi.sum(axis=0) / np.maximum(denom, 1e-300)
        trans = np.maximum(trans, _TRANS_FLOOR)
        trans /= trans.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        means = (gamma * arr[:, None]).sum(axis=0) / np.maximum(w, 1e-300)
        var = (gamma * (arr[:, None] - means[None, :]) ** 2).sum(axis=0) / np.maximum(w, 1e-300)
        stds = np.maximum(np.sqrt(np.maximum(var, 0.0)), floor)

        if abs(ll - prev_ll) < float(tol) * max(1.0, abs(prev_ll)):
            converged = True
            prev_ll = ll
            break
        prev_ll = ll

    # Son parametrelerle olabilirliği YENİDEN ölç (M adımı sonrası değer raporlanmalı).
    log_emis = _log_emissions(arr, means, stds)
    _, ll_final = _forward(log_emis, np.log(np.maximum(start, _TRANS_FLOOR)),
                           np.log(np.maximum(trans, _TRANS_FLOOR)))
    if not math.isfinite(ll_final):
        return None
    # Serbest parametre: K ortalama + K std + K(K−1) geçiş + (K−1) başlangıç.
    n_par = 2 * k + k * (k - 1) + (k - 1)
    bic = float(n_par * math.log(arr.size) - 2.0 * ll_final)
    order = np.argsort(means)          # durumlar ortalamaya göre SIRALANIR → kararlı kimlik
    return {"n_states": k,
            "means": [float(v) for v in means[order]],
            "stds": [float(v) for v in stds[order]],
            "transition": [[float(v) for v in row] for row in trans[np.ix_(order, order)]],
            "start": [float(v) for v in start[order]],
            "loglik": float(ll_final), "bic": bic, "n_params": int(n_par),
            "n_iter": int(it), "converged": bool(converged), "n": int(arr.size)}


def _unpack(model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """model sözlüğü → (means, stds, log_trans, log_start); bozuk model → None."""
    try:
        means = np.asarray(model["means"], dtype=float)
        stds = np.asarray(model["stds"], dtype=float)
        trans = np.asarray(model["transition"], dtype=float)
        start = np.asarray(model["start"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    k = means.size
    if k < 2 or stds.size != k or trans.shape != (k, k) or start.size != k:
        return None
    if not (np.all(np.isfinite(means)) and np.all(stds > 0.0) and np.all(trans >= 0.0)):
        return None
    return (means, stds, np.log(np.maximum(trans, _TRANS_FLOOR)),
            np.log(np.maximum(start, _TRANS_FLOOR)))


def hmm_filter(x: ArrayLike, model: dict) -> dict | None:
    """**NEDENSEL** rejim olasılıkları — canlı kullanıma uygun TEK çıktı.

    İki ayrı dizi döner ve farkları önemlidir:
      · `predicted[t]` = P(S_t | x_1..x_{t−1}) — bar AÇILMADAN önceki inanç.
      · `filtered[t]`  = P(S_t | x_1..x_t)     — bar KAPANDIKTAN sonraki inanç.
    Bir özellik hesabında hangisinin kullanılacağı karar barı sözleşmesine bağlıdır
    (`SIGNAL_ON_BAR_CLOSE` tartışmasının olasılık karşılığı) — bu modül seçmez, İKİSİNİ DE
    verir ki çağıran bilinçli seçsin.

    Dönüş: {'filtered' (T,K), 'predicted' (T,K), 'loglik', 'causal': True, 'n'}."""
    arr = _clean_1d(x, min_len=1)
    unpacked = _unpack(model) if isinstance(model, dict) else None
    if arr is None or unpacked is None:
        return None
    means, stds, log_trans, log_start = unpacked
    log_emis = _log_emissions(arr, means, stds)
    t_len, k = log_emis.shape
    filtered = np.empty((t_len, k), dtype=float)
    predicted = np.empty((t_len, k), dtype=float)
    # ÖLÇEKLENMİŞ ileri algoritma (Rabiner 1989 §V.A): önsel dağılım LİNEER uzayda
    # normalize taşınır, gözlem yoğunluğu ise log'da kalır (yoğunluk alt-taşabilir, olasılık
    # taşamaz). Böylece adım başına log/exp sayısı düşer ve geçiş tek matris-vektör
    # çarpımına iner — sonuç log-uzayı biçimiyle aynı, bkz. denklik testi.
    trans = np.exp(log_trans)
    prior = np.exp(log_start - log_start.max())
    prior = prior / prior.sum()
    total = 0.0
    for t in range(t_len):
        predicted[t] = prior
        le = log_emis[t]
        m = float(le.max())
        un = prior * np.exp(le - m)
        s = float(un.sum())
        if s <= 0.0:                       # tüm durumlar imkânsız → ölçüm yapılamaz
            return None
        total += m + math.log(s)
        filtered[t] = un / s
        prior = filtered[t] @ trans
    return {"filtered": filtered, "predicted": predicted, "loglik": float(total),
            "causal": True, "n": int(t_len)}


def hmm_smooth(x: ArrayLike, model: dict) -> dict | None:
    """P(S_t | x_1..x_T) — ileri-geri (Baum-Welch E adımı).

    ⚠️ **LOOK-AHEAD İÇERİR**: t anındaki olasılık t'den SONRAKİ gözlemleri kullanır.
    Geçmişi etiketlemek/analiz etmek için doğru araçtır; sinyal veya ML özelliği olarak
    KULLANILAMAZ. Dönüşte `causal=False` bayrağı bunu makine-okunur biçimde taşır —
    çağıran yanlışlıkla nedensel sanmasın (Tur-31 dersi: ölçüm yokluğu gibi, ölçümün
    KOŞULU da açıkça taşınmalıdır).

    Dönüş: {'smoothed' (T,K), 'loglik', 'causal': False, 'n'}."""
    arr = _clean_1d(x, min_len=2)
    unpacked = _unpack(model) if isinstance(model, dict) else None
    if arr is None or unpacked is None:
        return None
    means, stds, log_trans, log_start = unpacked
    log_emis = _log_emissions(arr, means, stds)
    log_alpha, ll = _forward(log_emis, log_start, log_trans)
    if not math.isfinite(ll):
        return None
    log_beta = _backward(log_emis, log_trans)
    log_gamma = log_alpha + log_beta
    log_gamma -= _lse(log_gamma, axis=1, keepdims=True)
    return {"smoothed": np.exp(log_gamma), "loglik": float(ll), "causal": False,
            "n": int(arr.size)}


def hmm_viterbi(x: ArrayLike, model: dict) -> dict | None:
    """En olası durum YOLU (Viterbi, log uzayında). ⚠️ LOOK-AHEAD: yol tüm seriye bakarak
    seçilir — `hmm_filter`'ın argmax'ı ile AYNI DEĞİLDİR ve nedensel değildir.

    Dönüş: {'path' (T,), 'loglik', 'causal': False, 'n'}."""
    arr = _clean_1d(x, min_len=1)
    unpacked = _unpack(model) if isinstance(model, dict) else None
    if arr is None or unpacked is None:
        return None
    means, stds, log_trans, log_start = unpacked
    log_emis = _log_emissions(arr, means, stds)
    t_len, k = log_emis.shape
    delta = log_start + log_emis[0]
    psi = np.zeros((t_len, k), dtype=int)
    for t in range(1, t_len):
        cand = delta[:, None] + log_trans
        psi[t] = np.argmax(cand, axis=0)
        delta = cand[psi[t], np.arange(k)] + log_emis[t]
    path = np.empty(t_len, dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(t_len - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return {"path": path, "loglik": float(np.max(delta)), "causal": False, "n": int(t_len)}


def hmm_next_state_probs(x: ArrayLike, model: dict, horizon: int = 1) -> dict | None:
    """**Boşluğu kapatan fonksiyon**: P(S_{T+h} | x_1..x_T) — "şu an hangi rejimdeyiz"
    yerine "bir sonraki barda hangi rejimde OLACAĞIZ, ne olasılıkla".

    Tümüyle NEDENSELDİR (yalnız `hmm_filter`'ın son filtreli dağılımını geçiş matrisiyle
    h kez ileri taşır). `current` bugünün filtreli dağılımı, `next` h adım sonrasıdır.
    `switch_prob` = 1 − P(aynı durumda kal) — "rejim değişiyor mu" sorusunun tek sayılı
    cevabı. `mode`/`mode_prob` en olası durum ve olasılığı: **`mode_prob` okunmadan `mode`
    okunamaz** (Tur-33'ün "bir ORAN, paydası okunmadan okunamaz" kuralının buradaki
    karşılığı — 0,34'lük bir tepe rejim ilan etmez).

    Dönüş: {'current','next','mode','mode_prob','switch_prob','horizon','causal': True}."""
    unpacked = _unpack(model) if isinstance(model, dict) else None
    if unpacked is None:
        return None
    try:
        h = int(horizon)
    except (TypeError, ValueError):
        return None
    if h < 1:
        return None
    filt = hmm_filter(x, model)
    if filt is None:
        return None
    trans = np.asarray(model["transition"], dtype=float)
    trans = np.maximum(trans, 0.0)
    row_sum = trans.sum(axis=1, keepdims=True)
    trans = trans / np.maximum(row_sum, 1e-300)
    current = filt["filtered"][-1]
    nxt = current @ np.linalg.matrix_power(trans, h)
    nxt = nxt / max(float(nxt.sum()), 1e-300)
    mode = int(np.argmax(nxt))
    stay = float(current @ np.diag(np.linalg.matrix_power(trans, h)))
    return {"current": [float(v) for v in current], "next": [float(v) for v in nxt],
            "mode": mode, "mode_prob": float(nxt[mode]),
            "switch_prob": float(np.clip(1.0 - stay, 0.0, 1.0)),
            "horizon": h, "causal": True}


def hmm_transition_summary(model: dict) -> dict | None:
    """Geçiş matrisinin OKUNABİLİR özeti:
      · `persistence[i]`        = P(i→i) (köşegen) — rejimin yapışkanlığı.
      · `expected_duration[i]`  = 1/(1−P(i→i)) — geometrik bekleme süresinin ortalaması
                                  (bar cinsinden). P(i→i)→1'de sonsuza gider → `None`
                                  (sonsuz süre bir SAYI değildir, uydurulmaz).
      · `stationary`            = uzun-dönem durum dağılımı (geçiş matrisinin 1
                                  öz-değerine ait sol öz-vektörü). İndirgenemez olmayan
                                  zincirde tanımsız → None.

    Dönüş: {'persistence','expected_duration','stationary','n_states'} · bozuk → None."""
    unpacked = _unpack(model) if isinstance(model, dict) else None
    if unpacked is None:
        return None
    trans = np.asarray(model["transition"], dtype=float)
    trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1e-300)
    diag = np.diag(trans)
    duration: list[float | None] = [
        (float(1.0 / (1.0 - d)) if d < 1.0 - 1e-12 else None) for d in diag]
    stationary: list[float] | None = None
    try:
        vals, vecs = np.linalg.eig(trans.T)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        if abs(vals[idx] - 1.0) < 1e-6:
            v = np.real(vecs[:, idx])
            total = float(v.sum())
            if abs(total) > 1e-12:
                v = v / total
                if np.all(v >= -1e-9):
                    stationary = [float(max(0.0, val)) for val in v]
    except np.linalg.LinAlgError:   # pragma: no cover - savunmacı
        stationary = None
    return {"persistence": [float(d) for d in diag], "expected_duration": duration,
            "stationary": stationary, "n_states": int(trans.shape[0])}


def hmm_state_labels(model: dict, vol_low_q: float = 0.33,
                     vol_high_q: float = 0.67) -> list[str] | None:
    """Durumlara `classify_regime` SÖZLÜĞÜYLE uyumlu okunabilir ad verir:
    `{up|down|flat}_{low|mid|high}`. Yön ortalamanın işaretinden (0'a çok yakınsa `flat`),
    oynaklık kovası durumun σ'sının DİĞER durumlar arasındaki sırasından gelir.

    Ad bir YORUMDUR, ölçüm değildir — durum kimliği `means` sırasıyla sabittir ve karar
    hiçbir zaman ada bakmaz. `classify_regime` ile aynı sözlük seçildi ki iki kaynak
    karşılaştırılabilsin (kural-tabanlı etiket ↔ olasılıksal rejim)."""
    unpacked = _unpack(model) if isinstance(model, dict) else None
    if unpacked is None:
        return None
    means, stds, _, _ = unpacked
    scale = float(np.mean(stds))
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    lo = float(np.quantile(stds, max(0.0, min(1.0, vol_low_q))))
    hi = float(np.quantile(stds, max(0.0, min(1.0, vol_high_q))))
    labels: list[str] = []
    for mu, sd in zip(means, stds, strict=False):
        # Yön eşiği: ortalamanın kendi oynaklığına GÖRE büyüklüğü (ölçek-değişmez).
        yon = "flat" if abs(mu) < 0.1 * scale else ("up" if mu > 0 else "down")
        vol = "low" if sd <= lo else ("high" if sd >= hi else "mid")
        labels.append(f"{yon}_{vol}")
    return labels


def hmm_select_states(x: ArrayLike, candidates: tuple[int, ...] = (2, 3, 4)) -> dict | None:
    """Durum SAYISINI veriden seçer (BIC en küçük). Rejim sayısı bir MODEL KARARIDIR;
    elle 2 seçmek "iki rejim vardır" varsayımını gizli bırakır.

    ΔBIC < 2 ise `decisive=False` — bu mertebede BIC modelleri ayırt EDEMEZ (aynı okuma
    kuralı `quant_math.copula_select`'te de var). Kazananı "bulundu" diye okumak, kapı
    PASS dese bile `n_kept`e bakma kuralının model-seçimi karşılığıdır.

    Dönüş: {'best','decisive','delta_bic','fits':{k: model|None}} · hiçbiri uymazsa None."""
    fits: dict[int, dict | None] = {}
    for k in candidates:
        try:
            kk = int(k)
        except (TypeError, ValueError):
            continue
        fits[kk] = gaussian_hmm_fit(x, n_states=kk)
    scored = sorted((float(m["bic"]), k) for k, m in fits.items() if m is not None)
    if not scored:
        return None
    best_bic, best_k = scored[0]
    delta = float(scored[1][0] - best_bic) if len(scored) > 1 else None
    return {"best": int(best_k), "decisive": bool(delta is not None and delta >= 2.0),
            "delta_bic": delta, "fits": fits}
