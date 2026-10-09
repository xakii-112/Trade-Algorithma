"""
microstructure.py — GERÇEKLEŞEN OYNAKLIK ve PİYASA MİKROYAPISI (AI v5, Tur-32). KARAR-DIŞI.

`quant_math` bar-içi oynaklık TAHMİNCİLERİNİ (Parkinson/Garman-Klass/Yang-Zhang/EWMA/GARCH)
ve `advanced_math` sıçrama/rejim ailesini taşır; ikisinde de eksik olan iki aile burada:

  (1) GERÇEKLEŞEN oynaklığın ÖNGÖRÜ modeli — HAR-RV (Corsi 2009) ve sıçrama-ayrıştırmalı
      HAR-CJ (Andersen-Bollerslev-Diebold 2007). Projede oynaklık hep ÖLÇÜLÜYOR, hiç
      TAHMİN EDİLMİYORDU; oysa oynaklık finansın en öngörülebilir büyüklüğüdür (yön
      değildir — ACI DERS #2/#4 yönle ilgilidir, oynaklıkla DEĞİL).
  (2) MİKROYAPI ölçüleri — likidite/etki/bilgi-akışı (Amihud, Roll, Corwin-Schultz, Kyle λ,
      VPIN). Bunlar ACI DERS #7'nin ("kenar sinyal-sonrası İLK bardaydı = bid-ask sıçraması")
      TEŞHİS aracıdır: bir adayın kârı gerçek mi yoksa makas artefaktı mı, ancak etkin makas
      ve etki katsayısı ölçülürse söylenebilir. Bugüne kadar bu ölçüler sistemde YOKTU.

ANAYASA: KARAR-DIŞI. Buradaki hiçbir ölçü sinyale/eşiğe/META_GATE'e bağlanmaz. Bir mikroyapı
ölçüsünün "iyi" görünmesi kenar demek DEĞİLDİR (ACI DERS #3).

Savunmacı sözleşme `quant_math` ile aynı: kısa/NaN/dönüştürülemez girdi → None; istisna YOK.
Saf fonksiyonlar; numpy dışında bağımlılık yok (scipy yalnız VPIN'in normal CDF'i için).

Birim konvansiyonu: tüm "variance" çıktıları BAR-BAŞINA (yıllıklandırılmamış) — `quant_math.
annualize_vol` ile çevrilir. Getiriler LOG getiridir (gerçekleşen oynaklık literatürünün
standardı; toplanabilirlik ancak log'da geçerlidir).

Kaynaklar: Corsi (2009) HAR-RV, Andersen-Bollerslev-Diebold (2007) HAR-CJ, Barndorff-Nielsen
& Shephard (2004) bipower variation, Zhang-Mykland-Aït-Sahalia (2005) two-scale RV, Amihud
(2002), Roll (1984), Corwin & Schultz (2012), Kyle (1985), Easley-López de Prado-O'Hara
(2012) VPIN/BVC.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln

# HAR pencereleri — Corsi'nin gün/hafta/ay ayrışımı (günlük barda 1/5/22).
HAR_DAILY = 1
HAR_WEEKLY = 5
HAR_MONTHLY = 22

# Bipower ölçekleme sabiti: μ₁ = E|Z| = √(2/π); BV = μ₁^{−2}·Σ|r_i||r_{i−1}|.
_MU1 = math.sqrt(2.0 / math.pi)
_BV_SCALE = 1.0 / (_MU1 * _MU1)

# Corwin-Schultz sabiti: 3 − 2√2.
_CS_CONST = 3.0 - 2.0 * math.sqrt(2.0)


def _clean_1d(x: ArrayLike, min_len: int = 1) -> np.ndarray | None:
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < min_len or not np.all(np.isfinite(arr)):
        return None
    return arr


def _ols(x: np.ndarray, y: np.ndarray) -> dict | None:
    """Sabit terimli OLS (lstsq) → {'coef','intercept','r2','resid_std','n'}.
    Tekil/kötü koşullu tasarım matrisinde None (uydurma katsayı döndürmez)."""
    n, p = x.shape
    if n <= p + 1:
        return None
    design = np.column_stack([np.ones(n), x])
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)):
        return None
    fitted = design @ beta
    resid = y - fitted
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(resid ** 2))
    dof = n - p - 1
    return {
        "coef": np.asarray(beta[1:], dtype=float),
        "intercept": float(beta[0]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "resid_std": float(math.sqrt(ss_res / dof)) if dof > 0 else None,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# 1) Gerçekleşen varyans ve sıçrama ayrıştırması
# ---------------------------------------------------------------------------
def realized_variance(returns: ArrayLike) -> float | None:
    """RV = Σ r_i² — bar-içi log getirilerden gerçekleşen varyans (bir GÜN için o günün
    bar getirileri verilir). Kısa/NaN → None."""
    r = _clean_1d(returns, min_len=1)
    return None if r is None else float(np.dot(r, r))


def bipower_variation(returns: ArrayLike) -> float | None:
    """BV = μ₁⁻² Σ|r_i||r_{i−1}| (Barndorff-Nielsen & Shephard 2004) — SIÇRAMAYA DAYANIKLI
    varyans. Tek bir dev getiri RV'yi şişirir ama BV'yi (komşusuyla çarpıldığı için) çok daha
    az etkiler; RV−BV farkı sıçrama bileşenidir.

    `advanced_math.jump_detect_bipower` AYNI fikri BAR-BAŞINA bayrak üretmek için kullanır;
    bu fonksiyon GÜN-BAŞINA skaler bileşen verir (HAR-CJ'nin girdisi). <2 getiri → None."""
    r = _clean_1d(returns, min_len=2)
    if r is None:
        return None
    a = np.abs(r)
    return float(_BV_SCALE * np.sum(a[1:] * a[:-1]))


def jump_split(returns: ArrayLike) -> dict | None:
    """RV'yi SÜREKLİ (C) ve SIÇRAMA (J) bileşenlerine ayırır: J = max(RV − BV, 0), C = RV − J.
    Kırpma (max(·,0)) literatürün standardıdır: sıçrama bileşeni tanımı gereği negatif olamaz,
    negatif fark örnekleme gürültüsüdür. `jump_share` = J/RV (0 = saf difüzyon).
    <2 getiri → None."""
    rv = realized_variance(returns)
    bv = bipower_variation(returns)
    if rv is None or bv is None:
        return None
    jump = max(rv - bv, 0.0)
    cont = rv - jump
    return {"rv": rv, "bv": bv, "jump": jump, "continuous": cont,
            "jump_share": float(jump / rv) if rv > 0 else 0.0}


def two_scale_rv(prices: ArrayLike, k: int = 5) -> dict | None:
    """TSRV (Zhang-Mykland-Aït-Sahalia 2005) — MİKROYAPI GÜRÜLTÜSÜNDEN arındırılmış
    gerçekleşen varyans.

    Sorun: yüksek frekansta RV, gerçek oynaklık DEĞİL gürültü varyansını ölçer (n arttıkça
    RV → 2n·σ²_gürültü diverjansı). Çözüm: seyrek ızgarada K alt-örneklemin ortalaması
    (RV_avg) ile tam ızgara RV'sinin (RV_all) yanlılık düzeltilmiş farkı:
        TSRV = (1 − n̄/n)⁻¹ · (RV_avg − (n̄/n)·RV_all)
    Ayrıca gürültü varyansı σ²_gürültü ≈ RV_all/(2n) ile tahmin edilir — bu sayı tek başına
    "bu seri ne kadar makas gürültüsü taşıyor" sorusunun cevabıdır (ACI DERS #7 teşhisi).

    k<2, fiyat ≤0 veya <3k gözlem → None."""
    p = _clean_1d(prices, min_len=1)
    try:
        k_int = int(k)
    except (TypeError, ValueError):
        return None
    if p is None or k_int < 2 or p.size < 3 * k_int or np.any(p <= 0):
        return None
    log_p = np.log(p)
    r_all = np.diff(log_p)
    n = int(r_all.size)
    rv_all = float(np.dot(r_all, r_all))
    rv_sub, n_bar_total = [], 0
    for start in range(k_int):
        sub = log_p[start::k_int]
        if sub.size < 2:
            continue
        r_sub = np.diff(sub)
        rv_sub.append(float(np.dot(r_sub, r_sub)))
        n_bar_total += int(r_sub.size)
    if not rv_sub:
        return None
    rv_avg = float(np.mean(rv_sub))
    n_bar = n_bar_total / len(rv_sub)
    ratio = n_bar / n
    if ratio >= 1.0:
        return None
    tsrv = (rv_avg - ratio * rv_all) / (1.0 - ratio)
    return {"tsrv": float(tsrv), "rv_all": rv_all, "rv_avg": rv_avg, "k": k_int, "n": n,
            "noise_var": float(rv_all / (2.0 * n)) if n > 0 else None}


def volatility_signature(prices: ArrayLike, max_scale: int = 20) -> dict | None:
    """Oynaklık İMZA GRAFİĞİ: RV'nin örnekleme adımına (q=1..max_scale) göre değişimi.

    Düz bir imza = gürültüsüz seri. q arttıkça DÜŞEN imza = pozitif mikroyapı gürültüsü
    (bid-ask sıçraması); YÜKSELEN imza = negatif seri-korelasyon/trend. `noise_slope` ilk beş
    ölçekteki eğimdir (log-ölçekte değil ham; işareti okunur, büyüklüğü seriye özgüdür).
    <3·max_scale gözlem / geçersiz → None."""
    p = _clean_1d(prices, min_len=1)
    try:
        m = int(max_scale)
    except (TypeError, ValueError):
        return None
    if p is None or m < 2 or p.size < 3 * m or np.any(p <= 0):
        return None
    log_p = np.log(p)
    scales, rvs = [], []
    for q in range(1, m + 1):
        sub = log_p[::q]
        if sub.size < 3:
            break
        r = np.diff(sub)
        # BAR-BAŞINA normalize: q-adımlık getirinin varyansı q'ya bölünür. Bu bölme
        # imzanın TAMAMIDIR — gürültüsüz seride (q·σ²)/q = σ² sabit çıkar, iid gürültülü
        # seride (q·σ² + 2σ²_gürültü)/q = σ² + 2σ²_gürültü/q olur ve q ile DÜŞER.
        scales.append(q)
        rvs.append(float(np.dot(r, r) / max(1, r.size) / q))
    if len(scales) < 3:
        return None
    arr_s = np.asarray(scales, dtype=float)
    arr_v = np.asarray(rvs, dtype=float)
    head = min(5, arr_s.size)
    slope = float(np.polyfit(arr_s[:head], arr_v[:head], 1)[0])
    return {"scales": arr_s, "rv_per_bar": arr_v, "noise_slope": slope,
            "noise_suspected": bool(slope < 0)}


# ---------------------------------------------------------------------------
# 2) HAR — gerçekleşen oynaklığın öngörüsü
# ---------------------------------------------------------------------------
def _har_design(rv: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(X, y) — gün/hafta/ay ortalamaları ile t+1 hedefi. Tüm pencereler t'ye kadarki veriden
    (kapsayıcı) → LOOK-AHEAD YOK; hedef t+1'dir. Yetersiz uzunluk → None."""
    n = rv.size
    if n < HAR_MONTHLY + 2:
        return None
    rows, targets = [], []
    for t in range(HAR_MONTHLY - 1, n - 1):
        d = float(rv[t])
        w = float(np.mean(rv[t - HAR_WEEKLY + 1:t + 1]))
        m = float(np.mean(rv[t - HAR_MONTHLY + 1:t + 1]))
        rows.append((d, w, m))
        targets.append(float(rv[t + 1]))
    if not rows:
        return None
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def har_rv_fit(rv_series: ArrayLike, log_target: bool = False) -> dict | None:
    """HAR-RV (Corsi 2009): RV_{t+1} = c + β_d·RV_t + β_w·RV^{(5)}_t + β_m·RV^{(22)}_t.

    Basitliğine rağmen çok-fraktal uzun-hafızalı oynaklığı yakalar ve GARCH sınıfını çoğu
    varlıkta OOS'ta geçer — bu yüzden gerçekleşen-oynaklık literatürünün fiilî tabanıdır.

    `log_target=True` model log(RV) üzerinde kurulur (artıklar çok daha Gauss'çu; pozitiflik
    yapısal olarak garanti). Bu durumda `forecast` GERİ DÖNÜŞÜMDE yarı-varyans düzeltmesi
    (exp(μ + σ²/2)) uygular — düz exp(μ) SİSTEMATİK olarak düşük tahmin verir.

    Dönüş: {'coef'(3),'intercept','r2','resid_std','n','log_target','persistence'} —
    `persistence` = β_d+β_w+β_m (1'e yakınsa uzun hafıza). <24 gözlem / geçersiz → None.
    R² İÇSEL uyumdur, OOS kenar iddiası DEĞİLDİR (ACI DERS #3)."""
    rv = _clean_1d(rv_series, min_len=HAR_MONTHLY + 2)
    if rv is None or np.any(rv < 0):
        return None
    if log_target:
        if np.any(rv <= 0):
            return None
        rv = np.log(rv)
    built = _har_design(rv)
    if built is None:
        return None
    x, y = built
    res = _ols(x, y)
    if res is None:
        return None
    res["log_target"] = bool(log_target)
    res["persistence"] = float(np.sum(res["coef"]))
    return res


def har_rv_forecast(model: dict, rv_history: ArrayLike, horizon: int = 1) -> float | None:
    """HAR ile h-adım ileri öngörü (yinelemeli: tahmin geçmişe eklenip yeniden beslenir).

    `model` `har_rv_fit` çıktısıdır. `rv_history` HAM RV serisidir (log_target modelinde
    dönüşüm İÇERİDE yapılır — çağıranın log alması beklenmez, bu tipik bir kullanım hatası
    olurdu). Dönüş HAM ölçekte RV tahmini. Geçersiz/kısa → None."""
    if not isinstance(model, dict):
        return None
    coef = model.get("coef")
    intercept = model.get("intercept")
    if coef is None or intercept is None:
        return None
    coef_arr = np.asarray(coef, dtype=float).reshape(-1)
    if coef_arr.size != 3 or not np.all(np.isfinite(coef_arr)):
        return None
    try:
        h = int(horizon)
        b0 = float(intercept)
    except (TypeError, ValueError):
        return None
    if h < 1:
        return None
    hist = _clean_1d(rv_history, min_len=HAR_MONTHLY)
    if hist is None or np.any(hist < 0):
        return None
    log_mode = bool(model.get("log_target", False))
    if log_mode:
        if np.any(hist <= 0):
            return None
        hist = np.log(hist)
    buf = list(hist)
    pred = float("nan")
    for _ in range(h):
        d = buf[-1]
        w = float(np.mean(buf[-HAR_WEEKLY:]))
        m = float(np.mean(buf[-HAR_MONTHLY:]))
        pred = b0 + coef_arr[0] * d + coef_arr[1] * w + coef_arr[2] * m
        if not math.isfinite(pred):
            return None
        buf.append(pred)
    if not log_mode:
        return float(pred)
    sigma = model.get("resid_std")
    half_var = 0.0
    if isinstance(sigma, (int, float)) and math.isfinite(float(sigma)):
        half_var = 0.5 * float(sigma) ** 2
    return float(math.exp(pred + half_var))


def har_cj_fit(rv_series: ArrayLike, jump_series: ArrayLike) -> dict | None:
    """HAR-CJ (Andersen-Bollerslev-Diebold 2007): sürekli ve sıçrama bileşenleri AYRI
    regresörler. Bulgu (literatür): sürekli bileşen kalıcıdır, sıçrama bileşeni neredeyse
    KALICI DEĞİLDİR — ikisini tek RV'de toplamak öngörüyü bozar.

    Regresörler: C_t, C^{(5)}, C^{(22)}, J_t (C = RV − J). `coef` sırası aynıdır.
    Uzunluklar eşit değilse / kısa / geçersiz → None."""
    rv = _clean_1d(rv_series, min_len=HAR_MONTHLY + 2)
    jp = _clean_1d(jump_series, min_len=HAR_MONTHLY + 2)
    if rv is None or jp is None or rv.size != jp.size or np.any(rv < 0) or np.any(jp < 0):
        return None
    cont = np.maximum(rv - jp, 0.0)
    built = _har_design(cont)
    if built is None:
        return None
    x_c, _ = built
    y = rv[HAR_MONTHLY:]
    jump_col = jp[HAR_MONTHLY - 1:-1].reshape(-1, 1)
    if x_c.shape[0] != y.size or jump_col.shape[0] != y.size:
        return None
    res = _ols(np.column_stack([x_c, jump_col]), y)
    if res is None:
        return None
    res["log_target"] = False
    res["persistence"] = float(np.sum(res["coef"][:3]))
    res["jump_beta"] = float(res["coef"][3])
    return res


# ---------------------------------------------------------------------------
# 3) Likidite / etkin makas / etki
# ---------------------------------------------------------------------------
def amihud_illiquidity(returns: ArrayLike, dollar_volume: ArrayLike,
                       scale: float = 1e6) -> dict | None:
    """Amihud (2002) ILLIQ = ort(|r_t| / DolarHacim_t) — "1 dolarlık işlem fiyatı yüzde kaç
    oynatıyor". Fiyat ETKİSİNİN en yaygın düşük-frekans vekilidir.

    `scale` çarpanı okunabilirlik içindir (varsayılan 1e6 → milyon dolar başına). Hacmi ≤0
    olan barlar DÜŞÜRÜLÜR (bölme tanımsız); kaç barın düştüğü `n_used`/`n_input` ile
    raporlanır (sessiz filtre yok). Kullanılabilir bar yoksa → None."""
    r = _clean_1d(returns, min_len=1)
    v = _clean_1d(dollar_volume, min_len=1)
    try:
        sc = float(scale)
    except (TypeError, ValueError):
        return None
    if r is None or v is None or r.size != v.size or not math.isfinite(sc) or sc <= 0:
        return None
    mask = v > 0
    if not np.any(mask):
        return None
    ratio = np.abs(r[mask]) / v[mask] * sc
    return {"illiq": float(np.mean(ratio)), "median": float(np.median(ratio)),
            "n_used": int(np.sum(mask)), "n_input": int(r.size)}


def roll_spread(prices: ArrayLike) -> dict | None:
    """Roll (1984) etkin makas: S = 2·√(−Cov(Δp_t, Δp_{t−1})).

    Sezgi: makas sıçraması (bid↔ask arasında zıplama) fiyat değişimlerine NEGATİF birinci
    dereceden otokovaryans sokar; kovaryansın büyüklüğü makasın karesiyle orantılıdır.

    Kovaryans POZİTİF çıkarsa (trend/momentum baskın) tahminci TANIMSIZDIR: `spread` None ve
    `positive_autocov`=True döner. Literatürde bu duruma sıfır atamak yaygındır ama bu, ölçüm
    yokluğunu "makas yok" diye raporlamak olurdu (Tur-31'in IV Rank dersinin aynısı).

    <4 fiyat / geçersiz → None. Alt sınır 3 DEĞİL 4'tür (BULGU-33-5): n=3'te Δp yalnız 2
    eleman olur, `np.cov(..., ddof=1)` TEK gözlemli dilimlerle çağrılır ve üç RuntimeWarning
    ("Degrees of freedom <= 0", "divide by zero", "invalid value") basıp NaN döndürür. Sonuç
    yine None'dı ama 7/24 süreçte defteri kirletiyordu ve `-W error` altında ÇÖKERDİ."""
    p = _clean_1d(prices, min_len=4)
    if p is None:
        return None
    dp = np.diff(p)
    if dp.size < 3:
        return None
    cov = float(np.cov(dp[1:], dp[:-1], ddof=1)[0, 1])
    if not math.isfinite(cov):
        return None
    if cov >= 0:
        return {"spread": None, "cov": cov, "positive_autocov": True, "n": int(p.size)}
    spread = 2.0 * math.sqrt(-cov)
    return {"spread": float(spread), "cov": cov, "positive_autocov": False, "n": int(p.size)}


def corwin_schultz_spread(high: ArrayLike, low: ArrayLike) -> dict | None:
    """Corwin & Schultz (2012) YÜKSEK-DÜŞÜK makas tahmincisi — yalnız OHLC gerektirir
    (tick verisi YOK), bu yüzden bu projenin bar verisiyle doğrudan uygulanabilir.

    Fikir: iki günlük yüksek/düşük aralığı oynaklık ∝ süre ile büyür, makas bileşeni BÜYÜMEZ;
    1-günlük ve 2-günlük aralıkların karşılaştırılması ikisini ayırır.

    Negatif tahminler örnekleme gürültüsüdür; ham ortalama (`spread_raw`) ve literatürün
    önerdiği sıfır-kırpılmış ortalama (`spread_clipped`) AYRI AYRI raporlanır.

    **BULGU-33-4 (Tur-33) — kırpma dejenere rejimde MAKAS UYDURUR.** Tahminci GÜNLÜK bar
    için türetilmiştir (oynaklık süreyle büyür, makas büyümez varsayımı). Saatlik/dakikalık
    barda varsayım çöker: Tur-32 saatlikte %36 negatif ölçtü, 1-dakikalık sentetik seride
    **%61,7** ölçüldü. Kırpma E[max(S,0)] işlemi SIMETRİK gürültüden bile POZİTİF bir sayı
    üretir — gerçek makas sıfır olsa dahi. Ölçülen örnek: `spread_raw = −0,000607`
    (yani "ölçülebilir makas yok") iken kırpılmış ortalama **+0,000140** = "1,4 bps makas".
    Bu, Tur-31'in IV Rank dersinin birebir tekrarıdır: ölçüm YOKLUĞU bir ÖLÇÜM gibi
    raporlanıyordu.

    Artık `roll_spread` ile AYNI sözleşme: yansız tahmin (`spread_raw`) pozitif değilse
    `spread` **None** ve `reliable=False` döner. Kırpılmış değer kaybolmaz
    (`spread_clipped`), ama varsayılan okuma artık uydurmuyor. `negative_share` alan
    uyumunun teşhisidir: günlük barda tipik olarak <%20, dejenere rejimde ~%50'ye koşar.
    <2 bar / H<L / geçersiz → None."""
    h = _clean_1d(high, min_len=2)
    low_arr = _clean_1d(low, min_len=2)
    if h is None or low_arr is None or h.size != low_arr.size:
        return None
    if np.any(h <= 0) or np.any(low_arr <= 0) or np.any(h < low_arr):
        return None
    hl = np.log(h / low_arr) ** 2
    beta = hl[:-1] + hl[1:]
    h2 = np.maximum(h[:-1], h[1:])
    l2 = np.minimum(low_arr[:-1], low_arr[1:])
    gamma = np.log(h2 / l2) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_CONST - np.sqrt(gamma / _CS_CONST)
    exp_a = np.exp(alpha)
    spread = 2.0 * (exp_a - 1.0) / (1.0 + exp_a)
    finite = np.isfinite(spread)
    if not np.any(finite):
        return None
    s_ok = spread[finite]
    raw = float(np.mean(s_ok))
    clipped = float(np.mean(np.maximum(s_ok, 0.0)))
    n_neg = int(np.sum(s_ok < 0))
    reliable = bool(raw > 0)
    return {"spread": clipped if reliable else None,
            "spread_clipped": clipped,
            "spread_raw": raw,
            "reliable": reliable,
            "negative_share": float(n_neg / s_ok.size),
            "n_negative": n_neg, "n": int(s_ok.size)}


def kyle_lambda(price_changes: ArrayLike, signed_volume: ArrayLike) -> dict | None:
    """Kyle (1985) λ — FİYAT ETKİSİ katsayısı: Δp = λ·(imzalı hacim) + ε (OLS eğimi).

    λ, "birim net alım baskısı fiyatı ne kadar iter" sorusunun cevabıdır: piyasa derinliğinin
    tersidir (λ ↑ = sığ piyasa). İmzalı hacim yoksa `bulk_volume_classification` ile
    üretilebilir. <5 gözlem / uyumsuz / tekil → None."""
    dp = _clean_1d(price_changes, min_len=5)
    sv = _clean_1d(signed_volume, min_len=5)
    if dp is None or sv is None or dp.size != sv.size:
        return None
    res = _ols(sv.reshape(-1, 1), dp)
    if res is None:
        return None
    return {"lam": float(res["coef"][0]), "intercept": res["intercept"],
            "r2": res["r2"], "n": res["n"]}


# ---------------------------------------------------------------------------
# PIN — bilgiye dayalı işlem OLASILIĞI (EKOP yapısal modeli), Tur-34
# ---------------------------------------------------------------------------
# Yan-Zhang (2012) tarzı DETERMİNİSTİK başlangıç ızgarası: EKOP olabilirliği çok-tepelidir
# ve tek başlangıçla yerel optimuma kilitlenir. Rastgele yeniden-başlatma yerine SABİT
# ızgara seçildi (anayasa madde 6 — aynı girdi her ortamda aynı sonucu vermeli).
_PIN_ALPHA_GRID = (0.2, 0.5, 0.8)
_PIN_DELTA_GRID = (0.3, 0.5, 0.7)
_PIN_GAMMA_GRID = (0.3, 0.6)      # bilgisiz alım oranının toplam alıma payı
_PIN_EPS = 1e-8
# Kaba eleme bütçesi ve ince çözüme kalan başlangıç sayısı (bkz. pin_ekop içindeki gerekçe).
_PIN_COARSE_ITER = 12
_PIN_REFINE_TOP = 3


def _pin_neg_loglik(theta: np.ndarray, b: np.ndarray, s: np.ndarray,
                    const: float) -> float:
    """EKOP negatif log-olabilirliği — Lin & Ke (2011) ÇARPANLAMASIYLA.

    Naif form P(B|eps_b)*P(S|eps_s+mu) çarpımlarını doğrudan hesaplar ve B,S birkaç yüzü
    geçince eps^B taşar/alt-taşar (literatürde belgelenmiş sayısal çöküş). Ortak çarpanlar
    dışarı alınıp köşeli parantez logsumexp ile toplandığında ifade her B,S için sonludur.

        L = (1/(B!S!))*exp(-eps_b-eps_s)*(eps_b+mu)^B*(eps_s+mu)^S * [ ... ]
        [ ... ] = (1-a)*x_b^B*x_s^S + a*d*x_b^B*exp(-mu) + a*(1-d)*x_s^S*exp(-mu)
        x_b = eps_b/(eps_b+mu),  x_s = eps_s/(eps_s+mu)
    """
    a, d, mu, eb, es = (float(v) for v in theta)
    if not (0.0 <= a <= 1.0 and 0.0 <= d <= 1.0) or mu < 0 or eb < 0 or es < 0:
        return 1e12
    xb = eb / max(eb + mu, _PIN_EPS)
    xs = es / max(es + mu, _PIN_EPS)
    lxb = math.log(max(xb, _PIN_EPS))
    lxs = math.log(max(xs, _PIN_EPS))
    e1 = b * lxb + s * lxs
    e2 = b * lxb - mu
    e3 = s * lxs - mu
    # max-kaydırma AÇILMIŞ hâlde (Tur-34 profili): `scipy.special.logsumexp` bu boyutta
    # çağrı başına ~151 µs array-API gönderim maliyeti taşıyordu ve `_pin_neg_loglik`
    # optimizasyon başına yüzlerce kez çağrılıyor. Aritmetik AYNI (aynı max-kaydırma),
    # yalnız üç terim elle yazıldı — vstack + broadcast_to + gönderim katmanı gitti.
    m = np.maximum(np.maximum(e1, e2), e3)
    w1, w2, w3 = (max(1.0 - a, _PIN_EPS), max(a * d, _PIN_EPS), max(a * (1.0 - d), _PIN_EPS))
    bracket = m + np.log(w1 * np.exp(e1 - m) + w2 * np.exp(e2 - m) + w3 * np.exp(e3 - m))
    ll = float(np.sum(-eb - es + b * math.log(max(eb + mu, _PIN_EPS))
                      + s * math.log(max(es + mu, _PIN_EPS)) + bracket)) + const
    return -ll if math.isfinite(ll) else 1e12


def pin_ekop(buys: ArrayLike, sells: ArrayLike) -> dict | None:
    """PIN — Easley-Kiefer-O'Hara-Paperman (1996) yapısal modelinin MLE uyumu.
    `knowledge.py`'nin kendi kataloğunda *"Bu modülde doğrudan bir PIN tahmincisi YOKTUR"*
    diye yazılı boşluğun kapanışı.

    MODEL: her dönemde `alpha` olasılıkla bir BİLGİ OLAYI olur; olay varsa `delta`
    olasılıkla kötü haberdir. Bilgisiz alım/satım Poisson(eps_b)/Poisson(eps_s) gelir;
    bilgili işlemciler haberin YÖNÜNDE Poisson(mu) ekler.

        PIN = alpha*mu / (alpha*mu + eps_b + eps_s)

    yani bir işlemin bilgili olma OLASILIĞIDIR. `roll_spread`/`corwin_schultz_spread`
    makasın BÜYÜKLÜĞÜNÜ ölçer; PIN o makasın NEDENİNİ (ters seçim payını) ölçer — ACI DERS
    #7'nin ("kenar ilk bardaki makas sıçramasıydı") teşhis takımının eksik parçası.

    ÖLÇEK DEĞİŞMEZLİĞİ: tüm sayımlar c ile çarpılırsa mu/eps'ler de c ile çarpılır ve PIN
    DEĞİŞMEZ. Bu yüzden orantılı vekil sayımlar (bkz. `pin_from_bars`) geçerli bir PIN
    verir — ama Poisson varsayımı SAYIM üzerinedir, hacim üzerine değil.

    En az 20 dönem, negatif olmayan tam-sayımsı girdi ister. Tüm-sıfır/dejenere → None.

    Dönüş: {'pin','alpha','delta','mu','eps_buy','eps_sell','loglik','n','converged'}.
    """
    b = _clean_1d(buys, min_len=20)
    s = _clean_1d(sells, min_len=20)
    if b is None or s is None or b.size != s.size:
        return None
    if np.any(b < 0) or np.any(s < 0):
        return None
    mean_b, mean_s = float(np.mean(b)), float(np.mean(s))
    if mean_b <= 0.0 or mean_s <= 0.0:
        return None
    # Faktöriyel terimleri parametrelerden BAĞIMSIZDIR; optimizasyonu etkilemez ama
    # raporlanan log-olabilirliğin gerçek değerini verir.
    const = float(-np.sum(gammaln(b + 1.0) + gammaln(s + 1.0)))
    bounds = [(1e-6, 1.0 - 1e-6), (1e-6, 1.0 - 1e-6), (1e-6, None), (1e-6, None),
              (1e-6, None)]
    starts: list[np.ndarray] = []
    for a0 in _PIN_ALPHA_GRID:
        for d0 in _PIN_DELTA_GRID:
            for g0 in _PIN_GAMMA_GRID:
                eb0 = max(g0 * mean_b, 1e-3)
                mu0 = max((mean_b - eb0) / max(a0 * (1.0 - d0), 1e-3), 1e-3)
                es0 = max(mean_s - a0 * d0 * mu0, 1e-3)
                starts.append(np.array([a0, d0, mu0, eb0, es0], dtype=float))

    def _solve(x0: np.ndarray, maxiter: int | None) -> dict | None:
        opts = {"maxiter": maxiter} if maxiter is not None else None
        try:
            res = minimize(_pin_neg_loglik, x0, args=(b, s, const), method="L-BFGS-B",
                           bounds=bounds, options=opts)
        except (ValueError, FloatingPointError):   # pragma: no cover - savunmacı
            return None
        if not np.isfinite(res.fun):
            return None
        return {"fun": float(res.fun), "x": np.asarray(res.x, dtype=float),
                "ok": bool(res.success)}

    # KABA ELEME → İNCE ÇÖZÜM (Tur-34): 18 başlangıcın tamamını sonuna kadar koşturmak
    # yerine hepsi kısa bütçeyle (maxiter=_PIN_COARSE_ITER) taranır, yalnız en iyi
    # `_PIN_REFINE_TOP` tanesi tam bütçeyle çözülür. Küresel optimum kaba turda ZATEN en
    # iyi birkaç arasındadır (kötü başlangıç ilk adımlarda belli olur); ölçüldü:
    # verdikt/parametreler değişmedi, süre düştü. Determinizm korunur — ızgara sabit,
    # sıralama sabit, rastgelelik yok.
    coarse = [(r["fun"], i, r) for i, x0 in enumerate(starts)
              if (r := _solve(x0, _PIN_COARSE_ITER)) is not None]
    if not coarse:
        return None
    coarse.sort(key=lambda t: (t[0], t[1]))
    best: dict | None = None
    for _, i, _r in coarse[:_PIN_REFINE_TOP]:
        fine = _solve(starts[i], None)
        if fine is not None and (best is None or fine["fun"] < best["fun"]):
            best = fine
    if best is None:
        return None
    a, d, mu, eb, es = (float(v) for v in best["x"])
    denom = a * mu + eb + es
    if denom <= 0.0:
        return None
    return {"pin": float(a * mu / denom), "alpha": a, "delta": d, "mu": mu,
            "eps_buy": eb, "eps_sell": es, "loglik": float(-best["fun"]),
            "n": int(b.size), "converged": bool(best["ok"])}


def pin_from_bars(prices: ArrayLike, volumes: ArrayLike,
                  unit: float | None = None) -> dict | None:
    """OHLCV barlarından PIN VEKİLİ: `bulk_volume_classification` ile alım/satım HACMİ
    tahmin edilir, `unit` ile bölünüp sayıma çevrilir ve `pin_ekop`a verilir.

    ⚠️ DÜRÜSTLÜK: EKOP'un Poisson varsayımı İŞLEM SAYISI üzerinedir. Hacim, işlem
    büyüklüğü dağılımı yüzünden Poisson'dan daha AŞIRI-DAĞILIMLIDIR (overdispersed) →
    `alpha`/`mu` ayrışması gürültülenir. PIN ölçek-değişmez olduğu için sayı YİNE DE
    yorumlanabilir, ama sonuç `proxy=True` bayrağıyla döner ve gerçek tick verisiyle
    hesaplanmış PIN ile AYNI ŞEY DEĞİLDİR. Bu bayrak, sonucun ne olduğunu okuyucunun
    gözünden saklamamak içindir (Tur-31/33'ün ölçüm-koşulu taşıma kuralı).

    `unit` verilmezse ortanca bar hacminin yüzde biri kullanılır (~100 birim/bar mertebesi;
    ölçek-değişmezlik nedeniyle PIN bu seçimden etkilenmez, yalnız sayısal koşullanma
    etkilenir). Dönüş `pin_ekop` sözlüğü + {'proxy': True, 'unit': ...} · geçersiz → None.
    """
    bvc = bulk_volume_classification(prices, volumes)
    if bvc is None or bvc["n"] < 20:
        return None
    buy, sell = np.asarray(bvc["buy"], dtype=float), np.asarray(bvc["sell"], dtype=float)
    if unit is None:
        med = float(np.median(buy + sell))
        unit = max(med / 100.0, 1e-12)
    else:
        unit = float(unit)
        if not math.isfinite(unit) or unit <= 0.0:
            return None
    out = pin_ekop(buy / unit, sell / unit)
    if out is None:
        return None
    out["proxy"] = True
    out["unit"] = float(unit)
    return out


def bulk_volume_classification(prices: ArrayLike, volumes: ArrayLike) -> dict | None:
    """BVC (Easley-López de Prado-O'Hara 2012): tick-yönü OLMADAN alıcı/satıcı hacmi tahmini.
      V_alım_i = V_i · Φ(Δp_i / σ_Δp)      ·      V_satım_i = V_i − V_alım_i
    Sezgi: büyük POZİTİF fiyat değişimi barındaki hacmin çoğunun alıcı-başlatmalı olduğunu
    ima eder; Φ bunu 0–1 arası yumuşak bir orana çevirir (tick kuralının sürekli karşılığı).
    <3 bar / σ=0 (düz seri) / geçersiz → None."""
    p = _clean_1d(prices, min_len=3)
    v = _clean_1d(volumes, min_len=3)
    if p is None or v is None or p.size != v.size or np.any(v < 0):
        return None
    dp = np.diff(p)
    sd = float(np.std(dp, ddof=1))
    if not math.isfinite(sd) or sd <= 0:
        return None
    frac = np.asarray(stats.norm.cdf(dp / sd), dtype=float)
    vol = v[1:]
    buy = vol * frac
    # Tur-78 (R3): σ pencerenin TAMAMINDAN → erken barların sınıfı sonraki fiyatlara bağlı.
    return {"buy": buy, "sell": vol - buy, "signed": 2.0 * buy - vol, "n": int(vol.size), "causal": False}


def vpin(prices: ArrayLike, volumes: ArrayLike, n_buckets: int = 50) -> dict | None:
    """VPIN (bilgiye dayalı işlem olasılığının hacim-senkronize vekili):
      VPIN = ort_j( |V_alım,j − V_satım,j| / V_j )   — j: EŞİT HACİMLİ kovalar

    Kova hacmi = toplam hacim / n_buckets (zaman değil HACİM saati — bilgi akışı zamanla
    değil işlem hacmiyle gelir). Yüksek VPIN = tek yönlü akış = ters seçim riski; 2010 Flash
    Crash öncesinde yükseldiği gösterilmiştir.

    Dürüstlük notu: VPIN'in öngörü gücü literatürde TARTIŞMALIDIR (Andersen & Bondarenko
    2014 eleştirisi). Burada TEŞHİS ölçüsü olarak sunulur, sinyal olarak değil.
    <n_buckets·3 bar / geçersiz → None."""
    try:
        nb = int(n_buckets)
    except (TypeError, ValueError, OverflowError):
        return None
    if nb < 2:
        return None
    bvc = bulk_volume_classification(prices, volumes)
    if bvc is None or bvc["n"] < 3 * nb:
        return None
    buy, sell = bvc["buy"], bvc["sell"]
    volume = buy + sell
    active = volume > 0
    cumulative_volume = np.r_[0., np.cumsum(volume[active])]
    cumulative_flow = np.r_[0., np.cumsum((buy - sell)[active])]
    total = float(cumulative_volume[-1])
    if not math.isfinite(total) or total <= 0 or not np.isfinite(cumulative_flow).all():
        return None
    bucket_size = total / nb
    # ponytail: bar içinde BVC oranı sabit; tick verisi varsa gerçek akış dağılımı kullanılabilir.
    # Kümülatif akışı hacim sınırlarında böl: bir bar birden fazla kovaya taşabilir.
    flow_at_edges = np.interp(np.linspace(0., total, nb + 1), cumulative_volume, cumulative_flow)
    arr = np.clip(np.abs(np.diff(flow_at_edges)) / bucket_size, 0., 1.)
    # causal=False: BVC σ'sı tüm pencereden; geçmiş kovalar o gün bilinen değer DEĞİLDİR (Tur-78, R3).
    return {"vpin": float(np.mean(arr)), "buckets": arr, "n_buckets": int(arr.size),
            "bucket_volume": float(bucket_size), "causal": False}


# ---------------------------------------------------------------------------
# 6) BAR-TÜRETİLMİŞ AKIŞ ÖLÇÜLERİ (Tur-57)
#
# Alpaca barları `trade_count` ve `vwap` kolonlarını ZATEN taşıyor (ölçüldü,
# 2026-08-29); sistem bunları normalize adımında düşürüyordu. Kolonlar geri
# geldiğine göre üç ölçü artık HESAPLANABİLİR: göreceli hacim (RVOL), ortalama
# işlem büyüklüğü (perakende kalabalığı ↔ kurumsal blok) ve VWAP sapması.
# Üçü de KARAR-DIŞI teşhistir; hiçbiri sinyale bağlanmaz (ACI DERS #3).
# ---------------------------------------------------------------------------
def relative_volume(volumes: ArrayLike, slots: ArrayLike,
                    min_history: int = 3) -> dict | None:
    """RVOL — son barın hacmi, geçmişte AYNI gün-içi diliminde görülen hacme göre.

    `slots` her bara bir dilim etiketi verir (ör. gün içi dakika; günlük barda hafta
    günü). Kıyas AYNI dilimin geçmişiyle yapılır: açılışın ilk 15 dakikası öğlene göre
    zaten 4-5 kat hacimlidir, düz ortalamaya bölmek bu mevsimselliği "patlama" sanır.

    Medyan tabanlı oran ESAS, ortalama tabanlı ikincil olarak döner (tek bir haber
    gününün ortalamayı şişirmesi RVOL'ü sessizce bastırır). Aynı dilimden
    `min_history` gözlem yoksa → None (uydurma taban YOK)."""
    v = _clean_1d(volumes, min_len=2)
    if v is None:
        return None
    try:
        s = np.asarray(slots).reshape(-1)
    except (TypeError, ValueError):
        return None
    if s.size != v.size:
        return None
    cur_slot = s[-1]
    hist = v[:-1][s[:-1] == cur_slot]
    hist = hist[np.isfinite(hist) & (hist >= 0)]
    if hist.size < max(1, int(min_history)):
        return None
    med = float(np.median(hist))
    mean = float(np.mean(hist))
    cur = float(v[-1])
    return {"rvol": (cur / med) if med > 0 else None,
            "rvol_mean": (cur / mean) if mean > 0 else None,
            "volume": cur, "slot_median": med, "slot_mean": mean,
            "n_history": int(hist.size), "slot": (cur_slot.item()
                                                  if hasattr(cur_slot, "item") else cur_slot)}


def trade_size_profile(volumes: ArrayLike, trade_counts: ArrayLike,
                       window: int = 20) -> dict | None:
    """Ortalama işlem büyüklüğü = hacim / işlem sayısı — kalabalık mı, blok mu?

    Küçük lot + yüksek işlem sayısı = perakende kalabalığı; büyük lot + düşük sayı =
    kurumsal blok. Seviye tek başına sembole göre anlamsızdır (SPY ile küçük bir hisse
    kıyaslanmaz), bu yüzden esas çıktı son barın kendi `window` geçmişine göre
    z-skorudur. `trade_count` 0/eksik olan barlar ÖLÇÜLMEMİŞ sayılır (0 kabul edilmez).

    Yalnız Alpaca barlarında ölçülebilir; yfinance `trade_count` VERMEZ → None."""
    v = _clean_1d(volumes, min_len=2)
    if v is None:
        return None
    try:
        tc = np.asarray(trade_counts, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if tc.size != v.size:
        return None
    ok = np.isfinite(tc) & (tc > 0) & np.isfinite(v)
    if not np.any(ok):
        return None
    avg = np.full(v.size, np.nan)
    avg[ok] = v[ok] / tc[ok]
    if not np.isfinite(avg[-1]):
        return None
    w = max(3, int(window))
    hist = avg[-(w + 1):-1]
    hist = hist[np.isfinite(hist)]
    z = None
    if hist.size >= 3:
        sd = float(np.std(hist, ddof=1))
        z = float((avg[-1] - float(np.mean(hist))) / sd) if sd > 0 else None
    return {"avg_trade_size": float(avg[-1]), "series": avg, "z": z,
            "baseline": (float(np.mean(hist)) if hist.size else None),
            "n_history": int(hist.size),
            "whale": (bool(z is not None and z >= 2.0)),
            "n_measured": int(np.count_nonzero(ok))}


def turnover_rate(volume: float, float_shares: float) -> float | None:
    """El değiştirme oranı = bar hacmi / halka açık hisse adedi.

    Float SİSTEMDE YOKTUR (proje temel-veri sağlayıcısı kullanmaz); bu fonksiyon float'ı
    çağırandan alır ve bir kaynak bağlanana kadar panoda "ölçülmedi" olarak kalır.
    Uydurma float ile hesaplamak, ölçüm yokluğunu ölçüm gibi göstermek olurdu."""
    try:
        v, f = float(volume), float(float_shares)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(v) and math.isfinite(f)) or f <= 0 or v < 0:
        return None
    return v / f


def vwap_deviation(closes: ArrayLike, vwaps: ArrayLike, window: int = 20) -> dict | None:
    """Kapanışın VWAP'tan sapması (bps) ve bu sapmanın kendi geçmişine göre z-skoru.

    VWAP hacim ağırlıklı ORTALAMA fiyattır: gün içinde işlem gören ortalama maliyet.
    Fiyatın VWAP'ın kaç sapma üstünde/altında olduğu, ortalamaya dönüş ile kırılım
    arasındaki farkı okumanın en ucuz yoludur. VWAP kolonu YOKSA (yfinance) → None."""
    c = _clean_1d(closes, min_len=2)
    if c is None:
        return None
    try:
        w = np.asarray(vwaps, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if w.size != c.size:
        return None
    ok = np.isfinite(w) & (w > 0) & np.isfinite(c)
    if not np.any(ok) or not ok[-1]:
        return None
    dev = np.full(c.size, np.nan)
    dev[ok] = (c[ok] - w[ok]) / w[ok] * 1e4
    n = max(3, int(window))
    hist = dev[-(n + 1):-1]
    hist = hist[np.isfinite(hist)]
    z = None
    if hist.size >= 3:
        sd = float(np.std(hist, ddof=1))
        z = float((dev[-1] - float(np.mean(hist))) / sd) if sd > 0 else None
    return {"dev_bps": float(dev[-1]), "series": dev, "z": z,
            "vwap": float(w[-1]), "close": float(c[-1]), "n_history": int(hist.size)}


# ---------------------------------------------------------------------------
# 7) GÜN İÇİ U-EĞRİSİ (Tur-77, Madde 4.5)
# ---------------------------------------------------------------------------

def _intraday_data(index, values, *, index_tz: str, agg: str, slot_minutes: int,
                   session_tz: str, session_open: str, session_close: str,
                   min_days: int) -> dict | None:
    """Ortak seans/slot hazırlığı; saat dilimi tahmin edilmez, eksik bar sıfırla doldurulmaz."""
    from zoneinfo import ZoneInfo

    import pandas as pd

    if not isinstance(agg, str) or agg not in ("sum", "median", "mean"):
        raise ValueError("agg yalnız sum, median veya mean olabilir")
    try:
        if (isinstance(slot_minutes, bool) or not isinstance(slot_minutes, (int, np.integer))
                or slot_minutes < 1 or isinstance(min_days, bool)
                or not isinstance(min_days, (int, np.integer)) or min_days < 1):
            return None
        bounds: list[int] = []
        for clock in (session_open, session_close):
            hour, minute = map(int, clock.split(":"))
            if not (0 <= hour < 24 and 0 <= minute < 60):
                return None
            bounds.append(hour * 60 + minute)
        start, end = bounds
        if start >= end:
            return None
        stamps = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
        vals = np.asarray(values, dtype=float)
        if vals.ndim != 1 or len(stamps) != vals.size or not vals.size:
            return None
        if stamps.tz is None:
            stamps = stamps.tz_localize(ZoneInfo(index_tz), ambiguous="NaT", nonexistent="NaT")
        stamps = stamps.tz_convert(ZoneInfo(session_tz))
        minute_of_day = (stamps.hour * 60 + stamps.minute + stamps.second / 60
                         + stamps.microsecond / 60_000_000 + stamps.nanosecond / 60_000_000_000)
        inside = stamps.notna() & (minute_of_day >= start) & (minute_of_day < end)
        stamps, vals = stamps[inside], vals[inside]
        if not len(stamps):
            return None
        elapsed = np.asarray(minute_of_day[inside] - start, dtype=float)
        frame = pd.DataFrame({
            "session": stamps.strftime("%Y-%m-%d"), "slot": (elapsed // slot_minutes).astype(int),
            "minute": elapsed, "value": np.where(np.isfinite(vals), vals, np.nan),
        }, index=stamps).sort_index()
        # Son seansın tüm değerleri eksik olsa bile önceki günü "bugün" yapma.
        daily = frame.dropna(subset=["value"]).groupby(["session", "slot"])["value"].agg(agg)
        daily = daily[np.isfinite(daily)]
        if daily.empty:
            return None
        return {"frame": frame, "daily": daily, "start": start, "end": end}
    except (TypeError, ValueError, OverflowError, KeyError, AttributeError):
        return None


def intraday_profile(index, values, *, index_tz: str, agg: str = "sum", slot_minutes: int = 30,
                     session_tz: str = "America/New_York", session_open: str = "09:30",
                     session_close: str = "16:00", min_days: int = 5,
                     exclude_last_session: bool = True) -> dict | None:
    """Geçmiş seansların gün/slot toplamlarından U-eğrisi; KARAR-DIŞI teşhis.

    Naive zamanlar zorunlu index_tz ile yerelleştirilir; aware zamanlar session_tz'ye
    çevrilir. Varsayılan olarak son yerel seans dışlanır. Eksik günler sıfır sayılmaz;
    min_days altında kalan slotların üç çeyreklik ölçüsü de None olur.
    """
    data = _intraday_data(
        index, values, index_tz=index_tz, agg=agg, slot_minutes=slot_minutes,
        session_tz=session_tz, session_open=session_open, session_close=session_close, min_days=min_days,
    )
    if data is None:
        return None
    daily = data["daily"]
    excluded = str(data["frame"]["session"].max()) if exclude_last_session else None
    if excluded is not None:
        daily = daily[daily.index.get_level_values("session") != excluded]
    n_sessions = int(daily.index.get_level_values("session").nunique())
    if n_sessions < min_days:
        return None
    slots: list[str] = []
    medians: list[float | None] = []
    q25: list[float | None] = []
    q75: list[float | None] = []
    counts: list[int] = []
    for slot, minute in enumerate(range(data["start"], data["end"], int(slot_minutes))):
        hist = daily[daily.index.get_level_values("slot") == slot].to_numpy(dtype=float)
        slots.append(f"{minute // 60:02d}:{minute % 60:02d}")
        counts.append(int(hist.size))
        if hist.size < min_days:
            medians.append(None)
            q25.append(None)
            q75.append(None)
        else:
            low, med, high = np.quantile(hist, [0.25, 0.5, 0.75])
            medians.append(float(med))
            q25.append(float(low))
            q75.append(float(high))
    return {"slots": slots, "median": medians, "q25": q25, "q75": q75, "n_days": counts,
            "n_sessions": n_sessions, "slot_minutes": int(slot_minutes), "agg": agg,
            "excluded_session": excluded, "session_tz": session_tz}


def intraday_anomaly(index, values, *, index_tz: str, agg: str = "sum", slot_minutes: int = 30,
                     session_tz: str = "America/New_York", session_open: str = "09:30",
                     session_close: str = "16:00", min_days: int = 5,
                     z_threshold: float = 3.0) -> dict | None:
    """Son seansın son tamamlanmış slotunu geçmişteki aynı slotla kıyaslar.

    Bar damgası barın başlangıcıdır. Son seanstaki pozitif zaman farklarının medyanı
    bar süresini verir; son damga + süre slot bitişine ulaşmıyorsa slot kısmi sayılır.
    Pozitif gün/slot değerlerinde log-MAD kullanılır; ölçülemeyen z None kalır.
    """
    profile = intraday_profile(
        index, values, index_tz=index_tz, agg=agg, slot_minutes=slot_minutes,
        session_tz=session_tz, session_open=session_open, session_close=session_close,
        min_days=min_days, exclude_last_session=True,
    )
    if profile is None:
        return None
    try:
        threshold = float(z_threshold)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(threshold) or threshold <= 0:
        return None
    data = _intraday_data(
        index, values, index_tz=index_tz, agg=agg, slot_minutes=slot_minutes,
        session_tz=session_tz, session_open=session_open, session_close=session_close, min_days=min_days,
    )
    if data is None:
        return None
    session = profile["excluded_session"]
    current = data["frame"][data["frame"]["session"] == session]
    differences = current.index.to_series().diff().dt.total_seconds().to_numpy() / 60
    positive = differences[differences > 0]
    if not positive.size:
        return None
    bar_minutes = float(np.median(positive))
    observed_end = float(current["minute"].iloc[-1]) + bar_minutes
    daily = data["daily"]
    today = daily[daily.index.get_level_values("session") == session]
    completed = [
        int(slot) for slot in current["slot"].unique()
        if min((int(slot) + 1) * slot_minutes, data["end"] - data["start"]) <= observed_end
    ]
    if not completed:
        return None
    slot = max(completed)
    if (session, slot) not in today.index:
        return None
    value = float(today.loc[(session, slot)])
    median = profile["median"][slot]
    ratio = value / median if median is not None and median > 0 else None
    if ratio is not None and not math.isfinite(ratio):
        ratio = None
    hist = daily[(daily.index.get_level_values("session") != session)
                 & (daily.index.get_level_values("slot") == slot)].to_numpy(dtype=float)
    hist = hist[hist > 0]
    z = None
    if value > 0 and hist.size >= min_days:
        logs = np.log(hist)
        center = float(np.median(logs))
        mad = float(np.median(np.abs(logs - center)))
        if mad > 0:
            candidate = (math.log(value) - center) / (1.4826 * mad)
            if math.isfinite(candidate):
                z = float(candidate)
    return {"session": session, "slot": profile["slots"][slot], "value": value, "median": median,
            "ratio": ratio, "z_robust": z, "n_days": profile["n_days"][slot],
            "anormal": bool(z is not None and abs(z) >= threshold), "bar_minutes": bar_minutes}


def liquidity_withdrawal(volume_anomaly: dict | None, spread_anomaly: dict | None,
                         z_threshold: float = 3.0) -> dict | None:
    """Aynı seans/slotta makas genişlerken hacim daralıyor mu? Yalnız KARAR-DIŞI teşhis."""
    if not isinstance(volume_anomaly, dict) or not isinstance(spread_anomaly, dict):
        return None
    session, slot = volume_anomaly.get("session"), volume_anomaly.get("slot")
    if (not isinstance(session, str) or not session or not isinstance(slot, str) or not slot
            or session != spread_anomaly.get("session") or slot != spread_anomaly.get("slot")):
        return None
    try:
        volume_z = float(volume_anomaly["z_robust"])
        spread_z = float(spread_anomaly["z_robust"])
        threshold = float(z_threshold)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in (volume_z, spread_z, threshold)) or threshold <= 0:
        return None
    return {"cekilme": bool(spread_z >= threshold and volume_z <= -threshold),
            "volume_z": volume_z, "spread_z": spread_z, "session": session, "slot": slot}
