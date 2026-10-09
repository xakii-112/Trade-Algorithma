"""
online.py — ÇEVRİMİÇİ ÖĞRENME ve UYARLAMA çekirdeği (AI v5, Tur-32). KARAR-DIŞI.

Sistemdeki uyarlama bugüne kadar TEK bir eksende vardı: `adaptive_calibration` olasılık
EŞLEMESİNİ zaman içinde yeniden fit eder. Eksik olan üç ayak burada:

  (1) SÜRÜKLENME TESPİTİ — `learning.page_hinkley` tek yönlü, sabit-eşikli bir dedektördür ve
      pencere boyutu seçimini kullanıcıya bırakır. `adwin` (Bifet & Gavaldà 2007) pencereyi
      VERİDEN seçer: istatistiksel olarak ayrışan en eski kısmı atar, atılan miktar
      sürüklenmenin kendisidir. Yanlış-alarm oranı δ ile matematiksel olarak sınırlıdır.
  (2) UZMAN BİRLEŞTİRME — `decision.py` kaynakları SABİT ağırlıklarla (Brier'den türetilmiş)
      harmanlar. `hedge_combine` ağırlıkları ÇEVRİMİÇİ öğrenir ve en iyi uzmana karşı
      pişmanlığı (regret) O(√T) ile sınırlar; "fixed-share" karışımı rejim DEĞİŞİMİNDE
      ağırlığı yeniden dağıtabilir (Herbster & Warmuth 1998) — bu, sabit ağırlığın
      yapısal olarak yapamadığı şeydir.
  (3) AKIŞ İÇİNDE ÖĞRENEN MODEL — `FTRLProximal` (McMahan ve ark. 2013): tek geçişli,
      koordinat-başına uyarlanır adım boyutlu, L1 ile seyrekleşen lojistik regresyon.
      Toplu yeniden-eğitim (research/live_retrain) ile aynı şey DEĞİLDİR: her yeni çözülmüş
      tahminde O(aktif özellik) maliyetle güncellenir, veri seti biriktirmeyi beklemez.

ACI DERS #9 MAKİNEYE GÖMÜLÜ: `hedge_combine(include_base=True)` KOŞAN TABAN ORANI'nı otomatik
olarak bir uzman olarak ekler. Böylece "birleşim iyileşti" iddiası, sabit-taban tahmincisini
GERÇEKTEN geçmediği sürece raporda kendini gösteremez (`vs_base` alanı).

ANAYASA: KARAR-DIŞI. Hiçbir çıktı sinyale/eşiğe/META_GATE'e/execution'a bağlanmaz. Bir
uyarlama mekanizmasının kayıpları düşürmesi KENAR demek değildir (ACI DERS #3/#9).

Nedensellik sözleşmesi (bu modülün en önemli kuralı): TÜM birleşik tahminler PREQUENTIAL'dır
— t anındaki tahmin YALNIZ t'den ÖNCEKİ sonuçlardan öğrenilmiş ağırlıklarla üretilir. Geriye
dönük bir "en iyi ağırlık" hesabı sızıntıdır ve burada hiçbir yerde yapılmaz.

Savunmacı sözleşme `quant_math` ile aynı: kısa/NaN/uyumsuz girdi → None; istisna YOK.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

# Olasılıklar log-kayıpta bu bantla kırpılır (log(0) = −inf koruması).
_P_EPS = 1e-6
# ADWIN varsayılanları: δ = yanlış pozitif olasılığı üst sınırı; pencere üst sınırı bellek/
# maliyet freni (tam ADWIN üstel histogramla O(log W) tutar — burada O(W) kabul edilir,
# gerekçe: bu projede seriler 10⁴ mertebesinde ve kesinlik yaklaşıklığa yeğlenir).
_ADWIN_DELTA = 0.002
_ADWIN_MAX_WINDOW = 5000
_ADWIN_MIN_SUB = 5


def _clean_1d(x: ArrayLike, min_len: int = 1) -> np.ndarray | None:
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < min_len or not np.all(np.isfinite(arr)):
        return None
    return arr


def _clean_2d(x: ArrayLike, min_rows: int = 1, min_cols: int = 1) -> np.ndarray | None:
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < min_rows or arr.shape[1] < min_cols:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _binary_outcomes(y: ArrayLike) -> np.ndarray | None:
    """{0,1} sonuç vektörü — 0/1 dışı değer varsa None (sessizce eşiklemek veri hatasını
    gizlerdi)."""
    arr = _clean_1d(y, min_len=1)
    if arr is None:
        return None
    if not np.all((arr == 0.0) | (arr == 1.0)):
        return None
    return arr


def _log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    pc = np.clip(p, _P_EPS, 1.0 - _P_EPS)
    return -(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))


# ---------------------------------------------------------------------------
# 1) ADWIN — veriden pencere seçen sürüklenme dedektörü
# ---------------------------------------------------------------------------
def adwin(values: ArrayLike, delta: float = _ADWIN_DELTA,
          max_window: int = _ADWIN_MAX_WINDOW) -> dict | None:
    """ADWIN (ADaptive WINdowing, Bifet & Gavaldà 2007) — sürüklenme tespiti + pencere seçimi.

    Her yeni gözlemde pencere W tüm (W₀|W₁) bölünmelerinde sınanır; ortalamalar
        |μ₀ − μ₁| ≥ ε_cut = √(2σ²_W·ln(2/δ')/m) + 2·ln(2/δ')/(3m),   m = (1/n₀ + 1/n₁)⁻¹
    kadar ayrışıyorsa W₀ (ESKİ kısım) ATILIR ve o nokta bir DEĞİŞİM olarak kaydedilir. δ'
    Bonferroni düzeltmesidir (δ/ln n) — bu yüzden yanlış-alarm oranı δ ile sınırlıdır.

    `page_hinkley`'e üstünlüğü: eşik/pencere ELDE ayarlanmaz, ölçek-bağımsızdır (varyansı
    kullanır) ve değişimden SONRA pencere kendini küçültür → tahminci hızla yeni rejime
    uyarlanır. Zayıflığı: O(W) maliyet (bkz. `_ADWIN_MAX_WINDOW`).

    Dönüş: {'n','drift_detected','n_drifts','change_points','final_window','final_mean',
    'window_sizes'} · kısa/NaN/geçersiz δ → None."""
    x = _clean_1d(values, min_len=2)
    try:
        d = float(delta)
        cap = int(max_window)
    except (TypeError, ValueError):
        return None
    if x is None or not (0.0 < d < 1.0) or cap < 2 * _ADWIN_MIN_SUB:
        return None

    # Pencere Python listesi DEĞİL, numpy tamponu üzerinde [start, end) dilimidir
    # (Tur-33 optimizasyonu). Eskiden her gözlemde `np.asarray(window)` çağrılıyordu:
    # 6.000 gözlem × ~3.000 elemanlık liste = profilde toplam sürenin **%34'ü**, saf
    # kutu-açma maliyeti. `buf[start:end]` BİTİŞİK bir görünümdür → np.var/np.cumsum
    # aynı bellek düzeninde aynı toplama sırasını kullanır, sonuç BİT-DENK kalır.
    win_cap = min(cap, int(x.size))
    buf = np.empty(2 * win_cap + 2, dtype=float)
    start = end = 0
    change_points: list[int] = []
    sizes = np.empty(x.size, dtype=int)
    # `j` yalnız n'e bağlı; her turda yeniden üretmek yerine bir kez kurup dilimlenir.
    j_all = np.arange(win_cap + 1, dtype=int)
    for i, v in enumerate(x):
        if end == buf.size:                       # sağ uç doldu → yaşayan dilimi başa çek
            buf[:end - start] = buf[start:end]
            end -= start
            start = 0
        buf[end] = v
        end += 1
        if end - start > cap:
            start += 1
        n = end - start
        if n >= 2 * _ADWIN_MIN_SUB:
            arr = buf[start:end]
            var_w = float(np.var(arr))
            delta_prime = d / max(math.log(max(n, 2)), 1e-12)
            log_term = math.log(2.0 / delta_prime)
            csum = np.cumsum(arr)
            total = float(csum[-1])
            # Tüm kesme noktaları vektörel olarak sınanır (döngüsüz — O(W) ama sabit çarpanı
            # küçük). j = sol parçanın uzunluğu.
            j = j_all[_ADWIN_MIN_SUB:n - _ADWIN_MIN_SUB + 1]
            if j.size:
                n0 = j.astype(float)
                n1 = float(n) - n0
                mu0 = csum[j - 1] / n0
                mu1 = (total - csum[j - 1]) / n1
                m_harm = 1.0 / (1.0 / n0 + 1.0 / n1)
                eps = np.sqrt(2.0 * var_w * log_term / m_harm) + 2.0 * log_term / (3.0 * m_harm)
                hit = np.abs(mu0 - mu1) >= eps
                if np.any(hit):
                    cut = int(j[np.argmax(hit)])       # en ESKİ geçerli kesme
                    start += cut
                    change_points.append(int(i))
        sizes[i] = end - start
    final = np.array(buf[start:end], dtype=float)
    return {
        "n": int(x.size),
        "drift_detected": bool(change_points),
        "n_drifts": len(change_points),
        "change_points": change_points,
        "final_window": int(final.size),
        "final_mean": float(np.mean(final)) if final.size else None,
        "window_sizes": sizes,
    }


# ---------------------------------------------------------------------------
# 2) Hedge / fixed-share — çevrimiçi uzman birleştirme
# ---------------------------------------------------------------------------
def running_base_rate(outcomes: ArrayLike, prior: float = 0.5,
                      prior_weight: float = 2.0) -> np.ndarray | None:
    """KOŞAN taban oranı — t anındaki tahmin yalnız t'den ÖNCEKİ sonuçların (Laplace
    düzeltmeli) ortalamasıdır: (Σy_{<t} + prior·w) / (t + w).

    Bu, projedeki tüm kalibrasyon/birleştirme kapılarının kıyas tabanıdır (ACI DERS #9) ve
    NEDENSELDİR — ilk gözlemde prior'a eşittir, geleceği hiç görmez. Geçersiz → None."""
    y = _binary_outcomes(outcomes)
    try:
        p0, w0 = float(prior), float(prior_weight)
    except (TypeError, ValueError):
        return None
    if y is None or not (0.0 <= p0 <= 1.0) or w0 <= 0:
        return None
    csum = np.concatenate([[0.0], np.cumsum(y)[:-1]])
    idx = np.arange(y.size, dtype=float)
    return np.asarray((csum + p0 * w0) / (idx + w0), dtype=float)


def hedge_combine(expert_preds: ArrayLike, outcomes: ArrayLike, eta: float = 1.0,
                  share: float = 0.0, include_base: bool = True) -> dict | None:
    """Üstel ağırlıklı uzman birleştirme (Hedge / Vovk toplama) + opsiyonel FIXED-SHARE.

    Her adımda:
        p̂_t = Σ_i w_i,t · p_i,t         (ağırlıklar t'den ÖNCEKİ kayıplardan — prequential)
        w_i ← w_i · exp(−η·ℓ(p_i,t, y_t)),  sonra normalize
        fixed-share:  w_i ← (1−α)·w_i + α/N     (α = `share`)

    `share > 0` NEDEN: saf Hedge tek bir uzmana yakınsar ve rejim değişince ölü ağırlığı geri
    kazanamaz; fixed-share her adımda ağırlığın α'sını yeniden dağıtır → en iyi uzman DİZİSİNE
    (switching) karşı pişmanlık sınırı verir. Uyarlamanın açık knob'u budur.

    `include_base=True` ise `running_base_rate` ek bir uzman olarak EKLENİR (son sütun) ve
    `vs_base` alanı birleşimin sabit-taban tahmincisini GERÇEKTEN geçip geçmediğini söyler —
    ACI DERS #9'un bu modüldeki karşılığı. Kayıp ölçüsü LOG-KAYIP'tır (uygun/proper skor).

    Dönüş: {'combined','weights','final_weights','loss_combined','loss_experts','best_expert',
    'regret','vs_base','n','n_experts'} · uyumsuz/kısa/NaN/aralık dışı → None."""
    preds = _clean_2d(expert_preds, min_rows=1, min_cols=1)
    y = _binary_outcomes(outcomes)
    try:
        eta_f, share_f = float(eta), float(share)
    except (TypeError, ValueError):
        return None
    if preds is None or y is None or preds.shape[0] != y.size:
        return None
    if not (eta_f > 0) or not (0.0 <= share_f <= 1.0):
        return None
    if np.any(preds < 0.0) or np.any(preds > 1.0):
        return None
    if include_base:
        base = running_base_rate(y)
        if base is None:
            return None
        preds = np.column_stack([preds, base])
    n, k = preds.shape
    w = np.full(k, 1.0 / k, dtype=float)
    combined = np.empty(n, dtype=float)
    weights_hist = np.empty((n, k), dtype=float)
    # Tur-35 OPTİMİZASYONU: uzman kayıpları adım-başı DEĞİL, tek seferde (n,k) olarak
    # hesaplanır. Kayıp yalnız (p_i,t, y_t)'ye bağlıdır — ağırlıklara DEĞİL — yani önceden
    # hesaplamak nedenselliği bozmaz (ağırlık güncellemesi hâlâ yalnız o adımın satırını
    # okur). `_log_loss` eleman-bazlıdır, indirgeme yoktur → dönüşüm BİT-DENK.
    # Ölçüldü (n=6000, k=3): 110,3 → 24,5 ms (4,50×); çıktı farkı 0,000e+00.
    loss_mat = _log_loss(preds, y.reshape(-1, 1))       # (n,k) ← (n,k) ⊗ (n,1) yayınlama
    row_min = loss_mat.min(axis=1)
    exp_step = np.exp(-eta_f * (loss_mat - row_min.reshape(-1, 1)))   # kaydırma: taşma koruması
    uniform = np.full(k, 1.0 / k)
    for t in range(n):
        weights_hist[t] = w
        combined[t] = float(np.dot(w, preds[t]))
        w = w * exp_step[t]
        total = float(np.sum(w))
        w = uniform.copy() if total <= 0 or not math.isfinite(total) else w / total
        if share_f > 0:
            w = (1.0 - share_f) * w + share_f / k
    loss_comb = float(np.mean(_log_loss(combined, y)))
    loss_experts = np.asarray([float(np.mean(_log_loss(preds[:, i], y))) for i in range(k)])
    best = int(np.argmin(loss_experts))
    vs_base = None if not include_base else float(loss_experts[k - 1] - loss_comb)
    return {
        "combined": combined, "weights": weights_hist, "final_weights": w,
        "loss_combined": loss_comb, "loss_experts": loss_experts,
        "best_expert": best, "regret": float(loss_comb - loss_experts[best]),
        "vs_base": vs_base, "n": int(n), "n_experts": int(k),
    }


# ---------------------------------------------------------------------------
# 3) FTRL-Proximal — akış içinde öğrenen lojistik regresyon
# ---------------------------------------------------------------------------
@dataclass
class FTRLProximal:
    """FTRL-Proximal çevrimiçi lojistik regresyon (McMahan ve ark. 2013, Google Ad Click).

    Neden bu algoritma: (a) tek geçişlidir — veri setini bellekte tutmaz, her örnekten sonra
    atılabilir; (b) koordinat-BAŞINA uyarlanır adım boyutu (AdaGrad'in FTRL biçimi) seyrek ve
    ÇOK FARKLI ölçekli özelliklerde SGD'den kararlıdır; (c) L1 düzenlileştirmesi GERÇEK sıfır
    üretir (proximal kapalı-form) — hangi özelliğin ölü olduğu doğrudan okunur.

    Güncelleme (koordinat i, gradyan g = (σ(wᵀx) − y)·x_i):
        σ_i = (√(n_i + g²) − √n_i)/α ;  z_i += g − σ_i·w_i ;  n_i += g²
        w_i = 0                                  eğer |z_i| ≤ λ₁
        w_i = −(z_i − sgn(z_i)λ₁)/((β+√n_i)/α + λ₂)   aksi halde

    KARAR-DIŞI: bu sınıf hiçbir canlı karar yolundan çağrılmaz; gölge/araştırma aracıdır.
    Determinizm: durum yalnız görülen örneklere bağlıdır, rastgelelik YOKTUR (anayasa madde 6).
    """

    n_features: int
    alpha: float = 0.1
    beta: float = 1.0
    l1: float = 0.0
    l2: float = 1.0
    _z: np.ndarray = field(init=False, repr=False)
    _n: np.ndarray = field(init=False, repr=False)
    n_seen: int = field(default=0, init=False)
    loss_sum: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        f = int(self.n_features)
        if f < 1:
            raise ValueError("n_features >= 1 olmalı")
        if self.alpha <= 0 or self.beta < 0 or self.l1 < 0 or self.l2 < 0:
            raise ValueError("alpha>0, beta/l1/l2 >= 0 olmalı")
        self.n_features = f
        self._z = np.zeros(f + 1, dtype=float)      # son eleman = bias (sabit terim)
        self._n = np.zeros(f + 1, dtype=float)

    def weights(self) -> np.ndarray:
        """Yürürlükteki ağırlıklar (L1 proximal kapalı-formundan TÜRETİLİR; ayrı saklanmaz —
        FTRL'nin tanımı budur). Son eleman bias'tır."""
        sign = np.sign(self._z)
        w = np.zeros_like(self._z)
        active = np.abs(self._z) > self.l1
        denom = (self.beta + np.sqrt(self._n[active])) / self.alpha + self.l2
        w[active] = -(self._z[active] - sign[active] * self.l1) / denom
        return w

    def _sigmoid(self, xa: np.ndarray, w: np.ndarray) -> float:
        """σ(wᵀx + b), z kırpmalı. `w` DIŞARIDAN verilir — `weights()` FTRL'de her çağrıda
        yeniden türetilen (saklanmayan) bir büyüklüktür; aynı adımda iki kez hesaplamak
        saf israftı (bkz. `_step`)."""
        z = float(np.dot(w[:-1], xa) + w[-1])
        return float(1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z)))))

    def predict_one(self, x: ArrayLike) -> float | None:
        """σ(wᵀx + b). Boyut uyumsuz/NaN → None (öğrenme durumu DEĞİŞMEZ)."""
        xa = _clean_1d(x, min_len=1)
        if xa is None or xa.size != self.n_features:
            return None
        return self._sigmoid(xa, self.weights())

    def _step(self, xa: np.ndarray, yf: float) -> tuple[float, float]:
        """Doğrulanmış tek örnek → (p, loss) ve durum güncellemesi. `weights()` BİR KEZ.

        Tur-33 optimizasyonu: eskiden `fit_stream` örnek başına `predict_one`ı İKİ kez
        çağırıyordu (bir kez kendisi, bir kez `update_one`ın içinden) ve `update_one` ayrıca
        kendi `weights()`ini alıyordu → örnek başına **3× `weights()`** (her biri ~6 dizi
        ayırır) + 2× sigmoid. Aritmetik değişmedi, yalnız tekrar kaldırıldı: çıktı bit-denk.
        """
        w = self.weights()
        p = self._sigmoid(xa, w)
        loss = float(_log_loss(np.array([p]), np.array([yf]))[0])
        full_x = np.concatenate([xa, [1.0]])
        g = (p - yf) * full_x
        g2 = g * g
        sigma = (np.sqrt(self._n + g2) - np.sqrt(self._n)) / self.alpha
        self._z += g - sigma * w
        self._n += g2
        self.n_seen += 1
        self.loss_sum += loss
        return p, loss

    def update_one(self, x: ArrayLike, y: float) -> float | None:
        """Tek örnekle güncelle; ÖNCE tahmin edip (prequential) log-kaybı döndürür.
        Geçersiz girdi → None ve durum DEĞİŞMEZ (kısmi güncelleme yok)."""
        xa = _clean_1d(x, min_len=1)
        if xa is None or xa.size != self.n_features:
            return None
        try:
            yf = float(y)
        except (TypeError, ValueError):
            return None
        if yf not in (0.0, 1.0):
            return None
        return self._step(xa, yf)[1]

    def fit_stream(self, x: ArrayLike, y: ArrayLike) -> dict | None:
        """Bir diziyi SIRAYLA işler (tek geçiş, prequential "test-then-train").
        Dönüş: {'preds','losses','mean_loss','n','nonzero','weights'} · geçersiz → None."""
        xm = _clean_2d(x, min_rows=1, min_cols=1)
        ya = _binary_outcomes(y)
        if xm is None or ya is None or xm.shape[0] != ya.size or xm.shape[1] != self.n_features:
            return None
        preds = np.empty(ya.size, dtype=float)
        losses = np.empty(ya.size, dtype=float)
        for i in range(ya.size):
            # Girdi `_clean_2d` ile zaten sonlu ve doğru genişlikte; satır başına ikinci
            # bir doğrulama turu gereksizdi.
            preds[i], losses[i] = self._step(xm[i], float(ya[i]))
        w = self.weights()
        return {"preds": preds, "losses": losses, "mean_loss": float(np.mean(losses)),
                "n": int(ya.size), "nonzero": int(np.sum(w != 0.0)), "weights": w}


def ftrl_prequential(x: ArrayLike, y: ArrayLike, alpha: float = 0.1, beta: float = 1.0,
                     l1: float = 0.0, l2: float = 1.0, compare_base: bool = True) -> dict | None:
    """`FTRLProximal.fit_stream` + KIYAS: aynı örneklerde koşan taban oranı.

    `uplift` = taban_kaybı − model_kaybı (POZİTİF = model tabanı geçiyor). Bu alan olmadan
    bir çevrimiçi öğrenici raporu okunamaz: log-kaybın düşmesi tek başına hiçbir şey söylemez,
    çünkü taban oranını öğrenmek de kaybı düşürür (ACI DERS #9). Geçersiz → None."""
    try:
        model = FTRLProximal(n_features=int(np.asarray(x).shape[1]), alpha=alpha, beta=beta,
                             l1=l1, l2=l2)
    except (ValueError, IndexError, AttributeError, TypeError):
        return None
    res = model.fit_stream(x, y)
    if res is None:
        return None
    if not compare_base:
        res["base_loss"] = None
        res["uplift"] = None
        return res
    ya = _binary_outcomes(y)
    base = None if ya is None else running_base_rate(ya)
    if ya is None or base is None:
        res["base_loss"] = None
        res["uplift"] = None
        return res
    base_loss = float(np.mean(_log_loss(base, ya)))
    res["base_loss"] = base_loss
    res["uplift"] = float(base_loss - res["mean_loss"])
    return res
