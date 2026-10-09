"""
options_math.py — OPSİYON & VOLATİLİTE YÜZEYİ matematiği (AI v5, Tur-32). KARAR-DIŞI.

Bu modül `knowledge.py`'nin KENDİ kataloğunda "uygulaması YOK" diye işaretlediği iki boşluğu
kapatır (bkz. eski satırlar: *"quant_math'te doğrudan bir 'IV' fonksiyonu yoktur"* ve
*"doğrudan bir smile-fit fonksiyonu YOKTUR"*). Artık ikisi de burada: implied vol kök-bulmayla
tersine çözülür, gülümseme SVI/SABR ile parametrize edilir, risk-nötr yoğunluk fiyatlardan
çıkarılır. Proje 2026-08-13'te GERÇEK opsiyon zinciri verisine (DoltHub `option_chain`:
bid/ask + tam greeks, 2321 sembol) kavuştuğu için bu aile artık ölçülebilir girdiye sahiptir.

ANAYASA: `nihai/ai` paketinin geri kalanı gibi KARAR-DIŞI. Buradan hiçbir sinyale/eşiğe/
META_GATE'e/execution'a bağlanmaz; çıktılar rapor/analiz/gölge-kayıttır. Bir yüzey ölçüsünün
"iyi görünmesi" kenar demek DEĞİLDİR (ACI DERS #3).

Savunmacı sözleşme (quant_math.py ile AYNI): girdi kısa/dönüştürülemez/sonlu-değil ise
`X | None` dönen her fonksiyon None döner; hiçbir girdi İSTİSNA fırlatmaz. Saf fonksiyonlar:
yan etkisiz, IO/ağ yok, numpy/scipy dışında bağımlılık yok (py_vollib YALNIZ testte çapraz
doğrulama için — modül ona bağımlı DEĞİL, `nihai/risk/options_analyzer.py`'nin aksine).

Konvansiyonlar:
  - `t` = yıl cinsinden vadeye kalan süre; `r` = sürekli bileşik faiz; temettü YOK varsayılır
    (q=0) — gerekirse çağıran `s` yerine `s·e^{−qT}` geçer (standart hile).
  - `k` = log-moneyness = ln(K/F) (FORWARD'a göre; spot'a göre DEĞİL — SVI literatürünün
    konvansiyonu budur ve calendar-arbitraj kontrolünü anlamlı kılan tek seçenektir).
  - `w` = TOPLAM örtük varyans = σ_BS(k)²·t (SVI bu değişkende parametriktir).
  - Greek birimleri `quant_math.bs_greeks` ile aynı: vega 1.0 vol birimi başına, zamana bağlı
    greekler (theta/charm/veta/color) YILLIK ve ∂/∂t (takvim zamanı) işaretindedir.

Kaynaklar: Black-Scholes (1973), Gatheral (2004) "A parsimonious arbitrage-free implied
volatility parameterization" (raw SVI), Gatheral & Jacquier (2014) "Arbitrage-free SVI
volatility surfaces" (Durrleman koşulu, kanat sınırı), Lee (2004) moment formülü, Hagan ve
ark. (2002) "Managing smile risk" (SABR lognormal asimptotiği), Breeden & Litzenberger (1978)
risk-nötr yoğunluk, Reiswich & Wystup (2010) 25Δ risk-reversal/butterfly konvansiyonu.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq, minimize

# quant_math'in BS çekirdeği TEK KAYNAK olarak kullanılır (kendi d1/d2'sini yazmak ikinci bir
# doğruluk kaynağı yaratırdı). `_bs_d1_d2`/`_bs_inputs_valid` modül-özel adlar ama BİLEREK
# import edilir: BS formülü sabittir, kopyalamak ıraksama riskini paket içine sokardı.
from nihai.ai.quant_math import (
    _bs_d1_d2,
    _bs_inputs_valid,
    _norm_cdf,
    _norm_pdf,
    _norm_ppf,
    bs_price,
)

# Kök-bulma sınırları: IV araması bu bantta yapılır. Alt sınır 0 değil çünkü σ=0'da BS
# türevlenemez; üst sınır %1000 yıllık — kripto/0DTE dahil hiçbir gerçek kotasyon aşmaz.
_IV_LO = 1e-6
_IV_HI = 10.0
_IV_XTOL = 1e-10

# SVI kalibrasyonu: |ρ|<1 ve σ>0 katı eşitsizliklerdir; sınırlara ε kadar yaklaşılır.
_SVI_EPS = 1e-6
_SVI_MAX_B = 50.0
# Durrleman g(k) taraması — kanatlar veri aralığının 1.5 katına kadar yoklanır (arbitraj
# tipik olarak EKSTRAPOLE bölgede doğar; yalnız veri noktalarında bakmak yanıltır).
_DURRLEMAN_GRID = 201
_DURRLEMAN_SPAN = 1.5

_INV_NORM_75 = _norm_ppf(0.75)   # 25Δ konvansiyonu için ±0.6745


# ---------------------------------------------------------------------------
# Girdi temizleme (quant_math._clean_1d ile aynı sözleşme — orada modül-özel)
# ---------------------------------------------------------------------------
def _clean_1d(x: ArrayLike, min_len: int = 1) -> np.ndarray | None:
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < min_len or not np.all(np.isfinite(arr)):
        return None
    return arr


def _as_float(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _is_call(kind: str) -> bool | None:
    """'c'/'call' → True, 'p'/'put' → False, aksi → None (bilinmeyen tip)."""
    k = (kind or "").strip().lower()
    if k in ("c", "call"):
        return True
    if k in ("p", "put"):
        return False
    return None


# ---------------------------------------------------------------------------
# 1) Örtük volatilite (implied vol) — kök bulma
# ---------------------------------------------------------------------------
def bs_no_arb_bounds(s: float, k: float, t: float, r: float, kind: str = "c") -> tuple[float, float] | None:
    """Avrupa opsiyonu için model-BAĞIMSIZ fiyat sınırları (arbitrajsızlık):
      call ∈ [max(0, S − K·e^{−rt}), S]   ·   put ∈ [max(0, K·e^{−rt} − S), K·e^{−rt}]
    Bu bandın DIŞINDAKİ bir kotasyon için IV TANIMSIZDIR (σ ne olursa olsun BS o fiyatı
    üretemez) — `implied_vol` bu yüzden None döner, uydurma bir σ döndürmez.
    Geçersiz girdi → None."""
    sf, kf, tf, rf_ = _as_float(s), _as_float(k), _as_float(t), _as_float(r)
    if sf is None or kf is None or tf is None or rf_ is None:
        return None
    is_call = _is_call(kind)
    if is_call is None or sf <= 0 or kf <= 0 or tf <= 0:
        return None
    disc_k = kf * math.exp(-rf_ * tf)
    if is_call:
        return max(0.0, sf - disc_k), sf
    return max(0.0, disc_k - sf), disc_k


def implied_vol(price: float, s: float, k: float, t: float, r: float = 0.0,
                kind: str = "c") -> float | None:
    """Black-Scholes ÖRTÜK VOLATİLİTE — `bs_price`'ın σ'ya göre tersi (Brent kök bulma).

    Neden kök bulma: BS fiyatı σ'da KESİN ARTANDIR (vega>0), bu yüzden çözüm varsa TEKTİR ve
    parantezleme (bracketing) güvenlidir; Newton-Raphson derin OTM'de vega≈0 olduğu için
    ıraksayabilir, Brent ıraksayamaz. `nihai/risk/options_analyzer.py` aynı işi py_vollib ile
    yapar — bu paket opsiyonel bağımlılık kullanamaz (anayasa madde 3), bu yüzden kendi
    çözücüsü vardır. Testte py_vollib ile çapraz doğrulanır.

    None dönüş koşulları (hepsi VERİ sorunu, çökme değil):
      · girdi geçersiz/sonlu-değil, s/k/t ≤ 0, bilinmeyen kind,
      · fiyat arbitrajsızlık bandının dışında (bkz. `bs_no_arb_bounds`) — sınıra tam OTURAN
        fiyat da dahil (σ→0 veya σ→∞ limitidir, sonlu kök yoktur),
      · [1e-6, 10.0] bandında kök yok (uç kotasyon).
    """
    p = _as_float(price)
    sf, kf, tf, rf_ = _as_float(s), _as_float(k), _as_float(t), _as_float(r)
    if p is None or sf is None or kf is None or tf is None or rf_ is None:
        return None
    bounds = bs_no_arb_bounds(sf, kf, tf, rf_, kind)
    if bounds is None:
        return None
    lo_p, hi_p = bounds
    if not (lo_p < p < hi_p):
        return None

    def _obj(sig: float) -> float:
        px = bs_price(sf, kf, tf, rf_, sig, kind)
        return float("nan") if px is None else px - p

    f_lo, f_hi = _obj(_IV_LO), _obj(_IV_HI)
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)) or f_lo * f_hi > 0:
        return None
    try:
        root = brentq(_obj, _IV_LO, _IV_HI, xtol=_IV_XTOL, maxiter=200)
    except (ValueError, RuntimeError):
        return None
    return float(root) if math.isfinite(root) else None


def implied_vol_chain(prices: ArrayLike, strikes: ArrayLike, s: float, t: float,
                      r: float = 0.0, kind: str = "c") -> np.ndarray | None:
    """Bir vade dilimindeki TÜM kotasyonlar için IV (eleman-başına `implied_vol`).
    Çözülemeyen kontrat NaN olur (satır DÜŞÜRÜLMEZ — çağıran hangi strike'ın çözülmediğini
    görmelidir). Uzunluklar uyumsuz/kısa → None."""
    px = _clean_1d(prices, min_len=1)
    ks = _clean_1d(strikes, min_len=1)
    if px is None or ks is None or px.size != ks.size:
        return None
    out = np.full(px.size, np.nan, dtype=float)
    for i in range(px.size):
        iv = implied_vol(float(px[i]), s, float(ks[i]), t, r, kind)
        if iv is not None:
            out[i] = iv
    return out


# ---------------------------------------------------------------------------
# 2) Yüksek mertebeden Greek'ler
# ---------------------------------------------------------------------------
def bs_greeks_extended(s: float, k: float, t: float, r: float, sigma: float,
                       kind: str = "c") -> dict | None:
    """`quant_math.bs_greeks`'in ÜSTÜNE ikinci/üçüncü mertebeden duyarlılıklar:

      vanna  = ∂²V/∂S∂σ  — delta'nın vol'e duyarlılığı (skew riskinin taşıyıcısı)
      volga  = ∂²V/∂σ²   — vega'nın vol'e duyarlılığı (vomma; vol-of-vol maruziyeti)
      ultima = ∂³V/∂σ³   — volga'nın vol'e duyarlılığı (Madde 045; vol şoklarında
                           vega hedge'inin ikinci mertebeden kayması)
      charm  = ∂²V/∂S∂t  — delta'nın zaman aşınması (hafta sonu hedge kayması)
      veta   = ∂²V/∂σ∂t  — vega'nın zaman aşınması
      speed  = ∂³V/∂S³   — gamma'nın spot'a duyarlılığı
      zomma  = ∂³V/∂S²∂σ — gamma'nın vol'e duyarlılığı
      color  = ∂³V/∂S²∂t — gamma'nın zaman aşınması
      dual_delta = ∂V/∂K — strike duyarlılığı (= −e^{−rt}·N(d2) call; dijital fiyatı)
      dual_gamma = ∂²V/∂K² — Breeden-Litzenberger yoğunluğunun e^{rt} katı

    q=0 (temettüsüz) varsayımı: bu durumda charm/color call ve put için AYNIDIR (Δp = Δc − 1
    olduğundan zamana göre türevleri eşittir) — bu bir sadeleştirme değil, kimliktir.
    Geçersiz girdi → None (bkz. `bs_price`)."""
    sf, kf, tf, rf_, sig = (_as_float(s), _as_float(k), _as_float(t),
                            _as_float(r), _as_float(sigma))
    if None in (sf, kf, tf, rf_, sig):
        return None
    assert sf is not None and kf is not None and tf is not None  # mypy daraltma
    assert rf_ is not None and sig is not None
    if not _bs_inputs_valid(sf, kf, tf, rf_, sig):
        return None
    is_call = _is_call(kind)
    if is_call is None:
        return None
    sqt = math.sqrt(tf)
    d1, d2 = _bs_d1_d2(sf, kf, tf, rf_, sig)
    pdf1 = _norm_pdf(d1)
    pdf2 = _norm_pdf(d2)
    disc = math.exp(-rf_ * tf)
    vega = sf * pdf1 * sqt
    gamma = pdf1 / (sf * sig * sqt)
    charm = -pdf1 * (2.0 * rf_ * tf - d2 * sig * sqt) / (2.0 * tf * sig * sqt)
    return {
        "vanna": -pdf1 * d2 / sig,
        "volga": vega * d1 * d2 / sig,
        # Ultima (Tur-76, Madde 045): −vega/σ² · (d1·d2·(1 − d1·d2) + d1² + d2²).
        # Sonlu farkla doğrulanır (test_tur76): ∂volga/∂σ ile örtüşmeli.
        "ultima": -vega / (sig * sig) * (d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2),
        "charm": charm,
        # veta/color: literatürdeki kapalı formlar ∂/∂τ (VADEYE KALAN süre) yönündedir;
        # burada theta/charm ile aynı eksende olsunlar diye ∂/∂t'ye çevrilir (işaret ters).
        # Sözleşme sonlu-fark testiyle kilitlidir (test_ai_options_math.py).
        "veta": sf * pdf1 * sqt * (rf_ * d1 / (sig * sqt) - (1.0 + d1 * d2) / (2.0 * tf)),
        "speed": -gamma / sf * (d1 / (sig * sqt) + 1.0),
        "zomma": gamma * (d1 * d2 - 1.0) / sig,
        "color": pdf1 / (2.0 * sf * tf * sig * sqt)
                 * (1.0 + (2.0 * rf_ * tf - d2 * sig * sqt) * d1 / (sig * sqt)),
        "dual_delta": (-disc * _norm_cdf(d2) if is_call
                       else disc * _norm_cdf(-d2)),
        "dual_gamma": disc * pdf2 / (kf * sig * sqt),
    }


# ---------------------------------------------------------------------------
# 3) Put-call paritesi — örtük forward ve temettü
# ---------------------------------------------------------------------------
def put_call_parity(call: float, put: float, s: float, k: float, t: float,
                    r: float = 0.0) -> dict | None:
    """Parite: C − P = S·e^{−qt} − K·e^{−rt}. Üç çıktı:

      gap             — gözlenen ihlal (C − P) − (S − K·e^{−rt}); q=0 varsayımıyla
      implied_forward — F = (C − P)·e^{rt} + K   (modelden BAĞIMSIZ; IV/SVI'nin doğru
                        çapası budur, spot DEĞİL — temettü/repo/borç maliyeti buradadır)
      implied_div_yield — q = r − ln(F/S)/t      (gap'in yıllık temettü/taşıma karşılığı)

    Bu üçlü bir arbitraj İDDİASI değildir: makas/asenkron kotasyon/Amerikan erken kullanım
    primi de aynı gap'i üretir. Ölçüm amacı, gülümseme kalibrasyonunda hangi forward'ın
    kullanılacağını dürüstçe belirlemektir. Geçersiz girdi → None."""
    cf, pf, sf, kf, tf, rf_ = (_as_float(call), _as_float(put), _as_float(s),
                               _as_float(k), _as_float(t), _as_float(r))
    if None in (cf, pf, sf, kf, tf, rf_):
        return None
    assert cf is not None and pf is not None and sf is not None
    assert kf is not None and tf is not None and rf_ is not None
    if sf <= 0 or kf <= 0 or tf <= 0:
        return None
    disc = math.exp(-rf_ * tf)
    fwd = (cf - pf) / disc + kf
    div_q = None
    if fwd > 0:
        div_q = rf_ - math.log(fwd / sf) / tf
    return {
        "gap": (cf - pf) - (sf - kf * disc),
        "implied_forward": float(fwd),
        "implied_div_yield": None if div_q is None else float(div_q),
    }


# ---------------------------------------------------------------------------
# 4) SVI gülümseme parametrizasyonu (raw)
# ---------------------------------------------------------------------------
def _parse_svi(params: object) -> tuple[float, float, float, float, float] | None:
    """{'a','b','rho','m','sigma'} → beşli; eksik anahtar / dönüştürülemez / geçersiz alan
    (b<0, σ≤0, |ρ|≥1, sonlu-değil) → None. Üç SVI fonksiyonunun ORTAK kapısıdır — parametre
    geçerliliği tek yerde tanımlıdır (üç ayrı kopya ıraksayabilirdi)."""
    if not isinstance(params, dict):
        return None
    try:
        a = float(params["a"])
        b = float(params["b"])
        rho = float(params["rho"])
        m = float(params["m"])
        sig = float(params["sigma"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (a, b, rho, m, sig)):
        return None
    if b < 0 or sig <= 0 or abs(rho) >= 1:
        return None
    return a, b, rho, m, sig


def svi_total_variance(k: ArrayLike, params: dict) -> np.ndarray | None:
    """Raw SVI (Gatheral 2004): w(k) = a + b·(ρ(k−m) + √((k−m)² + σ²)).
    `params` = {'a','b','rho','m','sigma'}; eksik/geçersiz anahtar → None.
    `k` log-moneyness (FORWARD'a göre), dönüş TOPLAM varyans (σ²·t)."""
    ks = _clean_1d(k, min_len=1)
    if ks is None or not isinstance(params, dict):
        return None
    parsed = _parse_svi(params)
    if parsed is None:
        return None
    a, b, rho, m, sig = parsed
    d = ks - m
    return np.asarray(a + b * (rho * d + np.sqrt(d * d + sig * sig)), dtype=float)


def _svi_derivatives(k: np.ndarray, a: float, b: float, rho: float, m: float,
                     sig: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(w, w', w'') — Durrleman koşulu için analitik türevler (sonlu fark DEĞİL: g(k)
    ikinci türeve doğrudan bakar, sayısal gürültü sahte arbitraj bulur)."""
    d = k - m
    root = np.sqrt(d * d + sig * sig)
    w = a + b * (rho * d + root)
    w1 = b * (rho + d / root)
    w2 = b * sig * sig / (root ** 3)
    return w, w1, w2


def svi_durrleman_g(k: ArrayLike, params: dict) -> np.ndarray | None:
    """Durrleman fonksiyonu g(k) — butterfly arbitrajının OPERATİF testi:
      g = (1 − k·w'/(2w))² − (w'²/4)·(1/w + 1/4) + w''/2
    g(k) ≥ 0 ∀k ⟺ dilim butterfly-arbitrajsız (risk-nötr yoğunluk her yerde ≥ 0).
    Geçersiz parametre / w ≤ 0 → None."""
    ks = _clean_1d(k, min_len=1)
    if ks is None or not isinstance(params, dict):
        return None
    parsed = _parse_svi(params)
    if parsed is None:
        return None
    a, b, rho, m, sig = parsed
    w, w1, w2 = _svi_derivatives(ks, a, b, rho, m, sig)
    if np.any(w <= 0):
        return None
    return np.asarray((1.0 - ks * w1 / (2.0 * w)) ** 2
                      - (w1 * w1 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0, dtype=float)


def svi_no_arbitrage(params: dict, k_range: tuple[float, float] | None = None) -> dict | None:
    """Dilim-içi (butterfly) arbitrajsızlık raporu — İKİ ayrı kontrol:

      wing_ok  : b·(1+|ρ|) ≤ 2 — Lee (2004) moment formülünün SVI karşılığı (toplam varyans
                 asimptotik eğimi 2'yi aşamaz). GEREKLİ koşul, yeterli DEĞİL.
      positive : a + b·σ·√(1−ρ²) ≥ 0 — w(k) > 0 (SVI'nin minimumu bu ifadedir).
      durrleman_min : g(k)'nin taranan bölgedeki minimumu; < 0 ise arbitraj VARDIR.

    Yeterli kontrol Durrleman'dır; kanat sınırı ucuz bir ön elemedir. `k_range` verilmezse
    m ± 1 bandının 1.5 katı taranır (ekstrapolasyon bölgesi bilerek dahil — arbitraj tipik
    olarak orada doğar). Geçersiz parametre → None."""
    if not isinstance(params, dict):
        return None
    parsed = _parse_svi(params)
    if parsed is None:
        return None
    a, b, rho, m, sig = parsed
    if k_range is None:
        lo, hi = m - 1.0, m + 1.0
    else:
        try:
            lo, hi = float(k_range[0]), float(k_range[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            return None
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * _DURRLEMAN_SPAN
    grid = np.linspace(mid - half, mid + half, _DURRLEMAN_GRID)
    g = svi_durrleman_g(grid, params)
    g_min = None if g is None else float(np.min(g))
    wing_ok = bool(b * (1.0 + abs(rho)) <= 2.0)
    positive = bool(a + b * sig * math.sqrt(1.0 - rho * rho) >= 0.0)
    return {
        "wing_ok": wing_ok,
        "positive": positive,
        "durrleman_min": g_min,
        "butterfly_free": bool(g_min is not None and g_min >= -1e-10 and positive),
    }


def svi_fit(log_moneyness: ArrayLike, total_variance: ArrayLike,
            weights: ArrayLike | None = None) -> dict | None:
    """Raw SVI'yi bir vade dilimine kalibre eder (ağırlıklı en küçük kareler, SLSQP).

    Parametre alanı KISITLIDIR (kalibrasyon-öncesi arbitraj savunması): b ≥ 0, |ρ| ≤ 1−ε,
    σ ≥ ε, a ≥ −b·σ·√(1−ρ²) kısıtı SONRADAN `svi_no_arbitrage` ile RAPORLANIR (kısıt olarak
    dayatılmaz — dayatmak kötü veriyi sessizce "düzeltir"; biz bunun yerine ölçüp söyleriz).

    İlklendirme veriye dayanır (a₀=min(w), b₀ kanat eğimlerinin ortalaması, m₀=argmin k) —
    SVI kayıp yüzeyi çok-modludur, sabit ilklendirme kötü yerel minimuma oturur.

    Dönüş: {'a','b','rho','m','sigma','rmse','n','success'} · <5 nokta veya çözüm yok → None.
    (5 = parametre sayısı; daha azı için problem belirlenmemiştir ve uyduran bir fit
    döndürmek dürüst değildir.)"""
    ks = _clean_1d(log_moneyness, min_len=5)
    ws = _clean_1d(total_variance, min_len=5)
    if ks is None or ws is None or ks.size != ws.size or np.any(ws <= 0):
        return None
    if weights is None:
        wt = np.ones_like(ws)
    else:
        wt_arr = _clean_1d(weights, min_len=1)
        if wt_arr is None or wt_arr.size != ws.size or np.any(wt_arr < 0) or np.all(wt_arr == 0):
            return None
        wt = wt_arr
    order = np.argsort(ks)
    ks, ws, wt = ks[order], ws[order], wt[order]

    span = float(ks[-1] - ks[0])
    if span <= 0:
        return None
    i_min = int(np.argmin(ws))
    a0 = float(np.min(ws)) * 0.5
    slope_r = float((ws[-1] - ws[i_min]) / max(ks[-1] - ks[i_min], 1e-6)) if i_min < ks.size - 1 else 0.1
    slope_l = float((ws[0] - ws[i_min]) / max(ks[i_min] - ks[0], 1e-6)) if i_min > 0 else 0.1
    b0 = max(1e-3, 0.5 * (abs(slope_r) + abs(slope_l)))
    x0 = np.array([a0, b0, 0.0, float(ks[i_min]), max(0.1 * span, 1e-2)], dtype=float)

    def _sse(p: np.ndarray) -> float:
        a, b, rho, m, sig = p
        d = ks - m
        w_hat = a + b * (rho * d + np.sqrt(d * d + sig * sig))
        res = (w_hat - ws) * np.sqrt(wt)
        return float(np.dot(res, res))

    bounds = [(-np.max(ws), np.max(ws) * 5.0), (0.0, _SVI_MAX_B),
              (-1.0 + _SVI_EPS, 1.0 - _SVI_EPS),
              (float(ks[0]) - span, float(ks[-1]) + span), (_SVI_EPS, 5.0 * span + 1.0)]
    try:
        res = minimize(_sse, x0, method="SLSQP", bounds=bounds,
                       options={"maxiter": 500, "ftol": 1e-12})
    except (ValueError, RuntimeError, TypeError):
        return None
    if not np.all(np.isfinite(res.x)):
        return None
    a, b, rho, m, sig = (float(v) for v in res.x)
    sse = _sse(res.x)
    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sig,
            "rmse": float(math.sqrt(sse / max(1.0, float(np.sum(wt))))),
            "n": int(ks.size), "success": bool(res.success)}


# ---------------------------------------------------------------------------
# 4b) SSVI — ARBİTRAJSIZ yüzey (Gatheral & Jacquier 2014), Tur-34
# ---------------------------------------------------------------------------
# Güç yasası φ: φ(θ) = η·θ^(−γ)·(1+θ)^(γ−1).
# Gatheral-Jacquier (2014) Corollary 5.1: γ ∈ (0, ½] ve η·(1+|ρ|) ≤ 2 ise yüzey KELEBEK
# arbitrajından YAPISI GEREĞİ muaftır. Takvim arbitrajı ayrıca θ_t'nin vadeye göre
# AZALMAMASI ile sağlanır. Yani SSVI'de arbitrajsızlık bir SONUÇ değil bir KISITTIR; raw
# SVI'de ise fit'ten SONRA taranıp umut edilir. `ssvi_diagnose`'un dayanağı tam da bu fark.
_SSVI_GAMMA_MAX = 0.5
_SSVI_ETA_MAX = 10.0


def _ssvi_phi(theta: np.ndarray | float, eta: float, gamma: float) -> np.ndarray:
    """Güç-yasası φ. θ>0 varsayılır (ATM toplam varyans)."""
    th = np.asarray(theta, dtype=float)
    return eta * th ** (-gamma) * (1.0 + th) ** (gamma - 1.0)


def ssvi_total_variance(k: ArrayLike, theta: float, params: dict) -> np.ndarray | None:
    """SSVI toplam varyansı:

        w(k, θ) = θ/2 · { 1 + ρ·φ·k + √( (φ·k + ρ)² + (1 − ρ²) ) }

    `theta` o vadenin ATM toplam varyansıdır (σ_ATM²·T) — SSVI'nin dilimleri birbirine
    bağlayan tek sayısı budur. `params` {'rho','eta','gamma'} taşır. Geçersiz parametre /
    θ ≤ 0 → None."""
    ka = _clean_1d(k, min_len=1)
    if ka is None or not isinstance(params, dict):
        return None
    th = _as_float(theta)
    rho, eta, gam = (_as_float(params.get(x)) for x in ("rho", "eta", "gamma"))
    if th is None or th <= 0.0 or rho is None or eta is None or gam is None:
        return None
    if abs(rho) >= 1.0 or eta <= 0.0 or gam <= 0.0:
        return None
    phi = float(_ssvi_phi(th, eta, gam))
    x = phi * ka
    return 0.5 * th * (1.0 + rho * x + np.sqrt((x + rho) ** 2 + (1.0 - rho * rho)))


def ssvi_no_arbitrage(params: dict, thetas: ArrayLike) -> dict | None:
    """SSVI parametrelerinin STATİK arbitrajsızlık koşulları (Gatheral-Jacquier §4-5):

      butterfly_wing : η·(1+|ρ|) ≤ 2                  (γ ∈ (0,½] ile birlikte YETERLİ)
      butterfly_sq   : θ·φ(θ)²·(1+|ρ|) ≤ 4            (Thm 4.2 ikinci koşul)
      calendar       : θ_t vadeye göre AZALMIYOR
      gamma_ok       : 0 < γ ≤ ½

    `svi_no_arbitrage` fit SONRASI g(k) tarar ve arbitraj BULABİLİR; burada koşullar
    sağlandığında arbitraj yapısı gereği YOKTUR. Geçersiz → None."""
    if not isinstance(params, dict):
        return None
    rho, eta, gam = (_as_float(params.get(x)) for x in ("rho", "eta", "gamma"))
    th = _clean_1d(thetas, min_len=1)
    if rho is None or eta is None or gam is None or th is None:
        return None
    if abs(rho) >= 1.0 or eta <= 0.0 or np.any(th <= 0.0):
        return None
    phi = _ssvi_phi(th, eta, gam)
    wing = bool(eta * (1.0 + abs(rho)) <= 2.0 + 1e-12)
    sq = bool(np.all(th * phi * phi * (1.0 + abs(rho)) <= 4.0 + 1e-12))
    cal = bool(np.all(np.diff(th) >= -1e-12))
    gam_ok = bool(0.0 < gam <= _SSVI_GAMMA_MAX + 1e-12)
    return {"butterfly_wing": wing, "butterfly_sq": sq, "calendar": cal,
            "gamma_ok": gam_ok, "arb_free": bool(wing and sq and cal and gam_ok)}


def ssvi_fit(slices: list[dict], enforce_arb_free: bool = True) -> dict | None:
    """SSVI'yi ÇOK VADELİ yüzeye ORTAK kalibre eder.

    `slices`: [{'log_moneyness': [...], 'total_variance': [...], 't': vade}, ...] — `t`
    verilirse dilimler ona göre sıralanır (takvim koşulu vade sırasına bakar; verilmezse
    listedeki sıra kullanılır).

    Serbest parametreler: global (ρ, η, γ) + dilim başına θ_t. Raw SVI dilim başına 5
    parametre isterken SSVI 3 + n_dilim ister — yüzey olarak çok daha KISITLIDIR. Bu bir
    dezavantaj değil, tam da amaçtır: arbitrajsızlık serbestlik derecesi azaltılarak elde
    edilir.

    `enforce_arb_free=True` (varsayılan) parametreleri arbitrajsız bölgeye SINIRLAR
    (η ≤ 2/(1+|ρ|), γ ≤ ½, θ artan). False ise kısıt kalkar — fit daha iyi olabilir ama
    arbitrajsızlık İDDİA EDİLEMEZ; `no_arb` raporu yine döner ve durumu söyler.

    Dönüş: {'rho','eta','gamma','thetas','expiries','rmse','rmse_per_slice','n_slices',
    'n_points','no_arb','success','enforced'} · geçersiz/yetersiz → None."""
    if not isinstance(slices, list) or len(slices) < 1:
        return None
    parsed: list[tuple[np.ndarray, np.ndarray, float]] = []
    for i, sl in enumerate(slices):
        if not isinstance(sl, dict):
            return None
        k_raw, w_raw = sl.get("log_moneyness"), sl.get("total_variance")
        if k_raw is None or w_raw is None:
            return None
        ks = _clean_1d(k_raw, min_len=3)
        ws = _clean_1d(w_raw, min_len=3)
        if ks is None or ws is None or ks.size != ws.size or np.any(ws <= 0):
            return None
        t_raw = _as_float(sl.get("t"))
        parsed.append((ks, ws, float(i) if t_raw is None else t_raw))
    parsed.sort(key=lambda r: r[2])
    n_sl = len(parsed)
    n_pts = int(sum(p[0].size for p in parsed))
    if n_pts < 3 + n_sl:
        return None

    # θ ilklendirmesi: her dilimin k=0'a en yakın noktasındaki toplam varyans.
    theta0 = np.maximum(
        np.array([float(ws[int(np.argmin(np.abs(ks)))]) for ks, ws, _ in parsed]), 1e-8)
    x0 = np.concatenate([[0.0, 0.5, 0.4], theta0])

    def _split(p: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        return float(p[0]), float(p[1]), float(p[2]), np.asarray(p[3:], dtype=float)

    def _sse(p: np.ndarray) -> float:
        rho, eta, gam, th = _split(p)
        if abs(rho) >= 1.0 or eta <= 0.0 or gam <= 0.0 or np.any(th <= 0.0):
            return 1e12
        total = 0.0
        for (ks, ws, _), t_th in zip(parsed, th, strict=True):
            x = float(_ssvi_phi(t_th, eta, gam)) * ks
            w_hat = 0.5 * t_th * (1.0 + rho * x + np.sqrt((x + rho) ** 2 + (1.0 - rho * rho)))
            d = w_hat - ws
            total += float(np.dot(d, d))
        return total if math.isfinite(total) else 1e12

    th_hi = float(np.max(theta0)) * 20.0 + 1.0
    bounds = [(-1.0 + _SVI_EPS, 1.0 - _SVI_EPS), (_SVI_EPS, _SSVI_ETA_MAX),
              (_SVI_EPS, _SSVI_GAMMA_MAX if enforce_arb_free else 2.0)]
    bounds += [(1e-8, th_hi)] * n_sl
    constraints: list[dict] = []
    if enforce_arb_free:
        constraints.append({"type": "ineq",           # 2 − η·(1+|ρ|) ≥ 0
                            "fun": lambda p: 2.0 - p[1] * (1.0 + abs(p[0]))})
        for i in range(n_sl - 1):                     # takvim: θ artan
            constraints.append({"type": "ineq",
                                "fun": (lambda p, j=i: p[3 + j + 1] - p[3 + j])})
    try:
        res = minimize(_sse, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints if constraints else (),
                       options={"maxiter": 800, "ftol": 1e-14})
    except (ValueError, RuntimeError, TypeError):
        return None
    if not np.all(np.isfinite(res.x)):
        return None
    rho, eta, gam, th = _split(np.asarray(res.x, dtype=float))
    params = {"rho": rho, "eta": eta, "gamma": gam}
    per_slice: list[float] = []
    for (ks, ws, _), t_th in zip(parsed, th, strict=True):
        w_hat = ssvi_total_variance(ks, float(t_th), params)
        if w_hat is None:
            return None
        per_slice.append(float(math.sqrt(float(np.mean((w_hat - ws) ** 2)))))
    return {**params, "thetas": [float(v) for v in th],
            "expiries": [float(p[2]) for p in parsed],
            "rmse": float(math.sqrt(_sse(res.x) / n_pts)),
            "rmse_per_slice": per_slice, "n_slices": n_sl, "n_points": n_pts,
            "no_arb": ssvi_no_arbitrage(params, th),
            "success": bool(res.success), "enforced": bool(enforce_arb_free)}


# Uyum bozulması eşiği: SSVI'nin RMSE'si raw SVI'nin bu KATINDAN fazlaysa arbitrajsızlık
# kısıtı veriye "sığmıyor" demektir → ihlalin kaynağı büyük olasılıkla VERİDEDİR.
_SSVI_FIT_RATIO = 2.0


def ssvi_diagnose(slices: list[dict]) -> dict | None:
    """**Tur-32 açık kalem #5'in kapanışı**: bir yüzeyde bulunan kelebek arbitrajı UYUM
    ARTEFAKTI mı, yoksa VERİDE mi?

    Yöntem — iki bağımsız kalibrasyon AYNI veriye:
      · raw SVI, dilim başına serbest (5 param/dilim) → `svi_no_arbitrage` ile g(k) taranır.
      · SSVI, arbitrajsız bölgeye KISITLI ortak yüzey (3 + n_dilim param).

    Okuma:
      · SVI arbitraj bulmuyorsa                       → 'ARBITRAJ_YOK'
      · SVI buluyor AMA SSVI veriyi benzer hatayla uyduruyorsa (oran ≤ 2)
                                                      → 'UYUM_ARTEFAKTI'
        (arbitrajsız bir yüzey bu veriyi AÇIKLAYABİLİYOR demektir)
      · SVI buluyor VE SSVI belirgin biçimde daha kötü uyduruyorsa
                                                      → 'VERIDE_ARBITRAJ'
        (kotasyonlar arbitrajsız hiçbir yüzeyle bağdaşmıyor: asenkron EOD kotasyonu,
        Amerikan erken-kullanım primi ya da gerçek arbitraj — bu test üçünü BİRBİRİNDEN
        ayırmaz, ama hepsini "fit hatası" şıkkından AYIRIR)
      · SSVI kısıtlı fit başarısızsa                  → 'BELIRSIZ'

    Verdikt bir KENAR iddiası değildir ve hiçbir karara bağlanmaz (anayasa).

    Dönüş: {'verdict','svi_rmse','ssvi_rmse','rmse_ratio','svi_violations','n_slices',
    'ssvi','svi_per_slice'} · geçersiz → None."""
    ssvi = ssvi_fit(slices, enforce_arb_free=True)
    if ssvi is None:
        return None
    svi_rows: list[dict] = []
    sse_svi, n_pts = 0.0, 0
    for sl in slices:
        k_raw, w_raw = sl.get("log_moneyness"), sl.get("total_variance")
        ks = None if k_raw is None else _clean_1d(k_raw, min_len=5)
        ws = None if w_raw is None else _clean_1d(w_raw, min_len=5)
        if ks is None or ws is None or ks.size != ws.size:
            svi_rows.append({"fit": None, "arb": None})
            continue
        fit = svi_fit(ks, ws)
        svi_rows.append({"fit": fit, "arb": svi_no_arbitrage(fit) if fit is not None else None})
        if fit is not None:
            sse_svi += float(fit["rmse"]) ** 2 * ks.size
            n_pts += int(ks.size)
    svi_rmse = float(math.sqrt(sse_svi / n_pts)) if n_pts > 0 else None
    violations = sum(1 for r in svi_rows
                     if r["arb"] is not None and not r["arb"]["butterfly_free"])
    ratio = (float(ssvi["rmse"] / svi_rmse)
             if svi_rmse is not None and svi_rmse > 0 else None)
    if not ssvi["success"] or not (ssvi["no_arb"] or {}).get("arb_free"):
        verdict = "BELIRSIZ"
    elif violations == 0:
        verdict = "ARBITRAJ_YOK"
    elif ratio is not None and ratio <= _SSVI_FIT_RATIO:
        verdict = "UYUM_ARTEFAKTI"
    else:
        verdict = "VERIDE_ARBITRAJ"
    return {"verdict": verdict, "svi_rmse": svi_rmse, "ssvi_rmse": float(ssvi["rmse"]),
            "rmse_ratio": ratio, "svi_violations": int(violations),
            "n_slices": int(ssvi["n_slices"]), "ssvi": ssvi,
            "svi_per_slice": [None if r["fit"] is None else float(r["fit"]["rmse"])
                              for r in svi_rows]}


# ---------------------------------------------------------------------------
# 5) SABR (Hagan 2002 lognormal asimptotiği)
# ---------------------------------------------------------------------------
def sabr_lognormal_vol(f: float, k: float, t: float, alpha: float, beta: float,
                       rho: float, nu: float) -> float | None:
    """Hagan ve ark. (2002) SABR → Black (lognormal) örtük vol yaklaşımı.

    ATM (f≈k) dalında z/x(z) → 1 limiti AYRI kolla ele alınır: genel formülde 0/0 belirsizliği
    vardır ve naif değerlendirme NaN üretir (bu, SABR uygulamalarının klasik hatasıdır).
    Yaklaşım küçük t·ν² için geçerlidir; uzun vade/yüksek vol-of-vol'de sapar (dürüstlük
    notu — burada düzeltilmez, çünkü düzeltme ayrı bir model seçimidir).
    Geçersiz girdi (f,k,t,alpha ≤ 0, |ρ|≥1, ν<0) → None."""
    ff, kf, tf = _as_float(f), _as_float(k), _as_float(t)
    al, be, rh, nuf = _as_float(alpha), _as_float(beta), _as_float(rho), _as_float(nu)
    if None in (ff, kf, tf, al, be, rh, nuf):
        return None
    assert ff is not None and kf is not None and tf is not None and al is not None
    assert be is not None and rh is not None and nuf is not None
    if ff <= 0 or kf <= 0 or tf <= 0 or al <= 0 or nuf < 0 or abs(rh) >= 1 or not 0.0 <= be <= 1.0:
        return None
    one_b = 1.0 - be
    fk_pow = (ff * kf) ** (one_b / 2.0)
    log_fk = math.log(ff / kf)
    # Zamana bağlı düzeltme terimi (her iki dalda ORTAK)
    corr = (one_b ** 2 / 24.0 * al * al / ((ff * kf) ** one_b)
            + 0.25 * rh * be * nuf * al / fk_pow
            + (2.0 - 3.0 * rh * rh) / 24.0 * nuf * nuf)
    if abs(log_fk) < 1e-9:                      # ATM dalı (z/x(z) → 1)
        return float(al / (ff ** one_b) * (1.0 + corr * tf))
    denom = fk_pow * (1.0 + one_b ** 2 / 24.0 * log_fk ** 2
                      + one_b ** 4 / 1920.0 * log_fk ** 4)
    z = nuf / al * fk_pow * log_fk
    inner = math.sqrt(1.0 - 2.0 * rh * z + z * z) + z - rh
    if inner <= 0 or (1.0 - rh) <= 0:
        return None
    x_z = math.log(inner / (1.0 - rh))
    if abs(x_z) < 1e-12:
        return None
    return float(al / denom * (z / x_z) * (1.0 + corr * tf))


def sabr_calibrate(f: float, strikes: ArrayLike, vols: ArrayLike, t: float,
                   beta: float = 0.5) -> dict | None:
    """SABR'ı bir vade dilimine kalibre eder (α, ρ, ν; β SABİT).

    β neden sabit: (α, β) neredeyse tam ÖZDEŞLEŞMEZ (aynı gülümseme farklı β'larla eşit iyi
    fit edilir) — β'yı da optimize etmek kalibrasyonu keyfileştirir. Piyasa konvansiyonu
    β'yı varlık sınıfına göre sabitler (hisse ≈0.5–1.0, faiz ≈0–0.5) ve kalan üçlüyü çözer.

    Dönüş: {'alpha','beta','rho','nu','rmse','n','success'} · <3 nokta / geçersiz → None."""
    ff, tf, bef = _as_float(f), _as_float(t), _as_float(beta)
    ks = _clean_1d(strikes, min_len=3)
    vs = _clean_1d(vols, min_len=3)
    if None in (ff, tf, bef) or ks is None or vs is None or ks.size != vs.size:
        return None
    assert ff is not None and tf is not None and bef is not None
    if ff <= 0 or tf <= 0 or not 0.0 <= bef <= 1.0 or np.any(ks <= 0) or np.any(vs <= 0):
        return None

    def _sse(p: np.ndarray) -> float:
        al, rh, nuf = float(p[0]), float(p[1]), float(p[2])
        total = 0.0
        for kk, vv in zip(ks, vs, strict=True):
            model = sabr_lognormal_vol(ff, float(kk), tf, al, bef, rh, nuf)
            if model is None:
                return 1e12
            total += (model - float(vv)) ** 2
        return total

    atm_v = float(np.interp(ff, ks, vs))
    x0 = np.array([max(1e-4, atm_v * ff ** (1.0 - bef)), 0.0, 0.3], dtype=float)
    bounds = [(1e-6, 10.0), (-1.0 + _SVI_EPS, 1.0 - _SVI_EPS), (1e-6, 5.0)]
    try:
        res = minimize(_sse, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 400, "ftol": 1e-14})
    except (ValueError, RuntimeError, TypeError):
        return None
    if not np.all(np.isfinite(res.x)):
        return None
    return {"alpha": float(res.x[0]), "beta": float(bef), "rho": float(res.x[1]),
            "nu": float(res.x[2]), "rmse": float(math.sqrt(_sse(res.x) / ks.size)),
            "n": int(ks.size), "success": bool(res.success)}


# ---------------------------------------------------------------------------
# 6) Gülümseme ölçüleri (25Δ konvansiyonu)
# ---------------------------------------------------------------------------
def _strike_for_delta(f: float, t: float, target_d1: float, smile_k: np.ndarray,
                      smile_v: np.ndarray, max_iter: int = 50) -> float | None:
    """Verilen d1 hedefine karşılık gelen strike — SABİT NOKTA iterasyonu.
    K = F·exp(σ(K)²t/2 − d1·σ(K)·√t); σ gülümsemeden geldiği için K'ya bağlıdır → iterasyon.
    Yakınsamazsa (uçuk gülümseme) None (uydurma değer DÖNDÜRÜLMEZ)."""
    sqt = math.sqrt(t)
    k_log = 0.0
    for _ in range(max_iter):
        vol = float(np.interp(k_log, smile_k, smile_v))
        if not math.isfinite(vol) or vol <= 0:
            return None
        new_k = vol * vol * t / 2.0 - target_d1 * vol * sqt
        if abs(new_k - k_log) < 1e-10:
            return float(f * math.exp(new_k))
        k_log = new_k
    return None


def smile_metrics(log_moneyness: ArrayLike, vols: ArrayLike, f: float, t: float) -> dict | None:
    """Gülümsemenin piyasa-konvansiyonu özeti (Reiswich & Wystup 2010):

      atm_vol   — k=0'daki vol (forward-ATM; spot-ATM DEĞİL)
      rr25      — 25Δ risk reversal = σ(25Δ call) − σ(25Δ put)  → SKEW'in işareti/şiddeti
      bf25      — 25Δ butterfly = ½(σ(25Δc)+σ(25Δp)) − σ_ATM    → KUYRUK kalınlığı
      skew      — dσ/dk (ATM'de merkezî fark)
      curvature — d²σ/dk² (ATM'de merkezî fark)

    rr25/bf25 yalnız 25Δ strike'ları veri aralığına DÜŞÜYORSA hesaplanır; aksi halde None
    (ekstrapolasyon sessizce yapılmaz — `np.interp` uçları sabitler ve bu sahte bir düz
    kanat üretirdi). <3 nokta / geçersiz → None."""
    ks = _clean_1d(log_moneyness, min_len=3)
    vs = _clean_1d(vols, min_len=3)
    ff, tf = _as_float(f), _as_float(t)
    if ks is None or vs is None or ks.size != vs.size or ff is None or tf is None:
        return None
    if ff <= 0 or tf <= 0 or np.any(vs <= 0):
        return None
    order = np.argsort(ks)
    ks, vs = ks[order], vs[order]
    if not (ks[0] <= 0.0 <= ks[-1]):
        return None
    atm = float(np.interp(0.0, ks, vs))
    h = min(0.05, max(1e-3, float(ks[-1] - ks[0]) / 20.0))
    v_up = float(np.interp(h, ks, vs))
    v_dn = float(np.interp(-h, ks, vs))
    skew = (v_up - v_dn) / (2.0 * h)
    curv = (v_up - 2.0 * atm + v_dn) / (h * h)

    rr25 = bf25 = None
    k_call = _strike_for_delta(ff, tf, -_INV_NORM_75, ks, vs)   # 25Δ call: N(d1)=0.25
    k_put = _strike_for_delta(ff, tf, _INV_NORM_75, ks, vs)     # 25Δ put:  N(−d1)=0.25
    if k_call is not None and k_put is not None:
        lk_c, lk_p = math.log(k_call / ff), math.log(k_put / ff)
        if ks[0] <= lk_p and lk_c <= ks[-1]:
            v_c = float(np.interp(lk_c, ks, vs))
            v_p = float(np.interp(lk_p, ks, vs))
            rr25 = v_c - v_p
            bf25 = 0.5 * (v_c + v_p) - atm
    return {"atm_vol": atm, "rr25": rr25, "bf25": bf25,
            "skew": float(skew), "curvature": float(curv), "n": int(ks.size)}


# ---------------------------------------------------------------------------
# 7) Risk-nötr yoğunluk (Breeden-Litzenberger)
# ---------------------------------------------------------------------------
def risk_neutral_density(strikes: ArrayLike, call_prices: ArrayLike, t: float,
                         r: float = 0.0) -> dict | None:
    """Breeden & Litzenberger (1978): q(K) = e^{rt}·∂²C/∂K².

    İkinci türev DÜZGÜN OLMAYAN strike ızgarasında üç-nokta bölünmüş farkla alınır (eşit
    aralık VARSAYILMAZ — gerçek zincirlerde strike adımı paranın içinde/dışında değişir).
    Negatif yoğunluk butterfly arbitrajının doğrudan gözlemidir: bastırılmaz, `n_negative` ile
    RAPORLANIR (sessizce sıfırlamak veriyi güzelleştirip yalan söylerdi).

    Dönüş: {'strikes','density','mass','n_negative','mean','std','skew','kurtosis'} — momentler
    NORMALİZE edilmiş yoğunluk üzerinden (mass ≈ 1 değilse ızgara dar demektir; mass yine de
    ham haliyle raporlanır). <4 strike / geçersiz → None."""
    ks = _clean_1d(strikes, min_len=4)
    cs = _clean_1d(call_prices, min_len=4)
    tf, rf_ = _as_float(t), _as_float(r)
    if ks is None or cs is None or ks.size != cs.size or tf is None or rf_ is None or tf <= 0:
        return None
    order = np.argsort(ks)
    ks, cs = ks[order], cs[order]
    if np.any(np.diff(ks) <= 0):                 # yinelenen strike → türev tanımsız
        return None
    h1 = ks[1:-1] - ks[:-2]
    h2 = ks[2:] - ks[1:-1]
    d2c = 2.0 / (h1 + h2) * ((cs[2:] - cs[1:-1]) / h2 - (cs[1:-1] - cs[:-2]) / h1)
    dens = math.exp(rf_ * tf) * d2c
    grid = ks[1:-1]
    n_neg = int(np.sum(dens < 0))
    mass = float(np.trapezoid(dens, grid))
    if not math.isfinite(mass) or abs(mass) < 1e-12:
        return {"strikes": grid, "density": dens, "mass": mass, "n_negative": n_neg,
                "mean": None, "std": None, "skew": None, "kurtosis": None}
    p = dens / mass
    mean = float(np.trapezoid(grid * p, grid))
    var = float(np.trapezoid((grid - mean) ** 2 * p, grid))
    std = math.sqrt(var) if var > 0 else 0.0
    if std <= 0:
        return {"strikes": grid, "density": dens, "mass": mass, "n_negative": n_neg,
                "mean": mean, "std": std, "skew": None, "kurtosis": None}
    skew = float(np.trapezoid(((grid - mean) / std) ** 3 * p, grid))
    kurt = float(np.trapezoid(((grid - mean) / std) ** 4 * p, grid))
    return {"strikes": grid, "density": dens, "mass": mass, "n_negative": n_neg,
            "mean": mean, "std": std, "skew": skew, "kurtosis": kurt}


# ---------------------------------------------------------------------------
# 8) Vade yapısı — calendar arbitraj
# ---------------------------------------------------------------------------
def vol_term_structure(expiries: ArrayLike, atm_vols: ArrayLike) -> dict | None:
    """ATM vol vade yapısı + CALENDAR arbitraj kontrolü.

    Kural (Gatheral): TOPLAM varyans w(t)=σ²t vade boyunca AZALAMAZ. Azalıyorsa ileri
    (forward) varyans NEGATİFTİR — takvim arbitrajı. `forward_vols` her ardışık dilim için
    √(Δw/Δt) verir (negatif ileri varyansta NaN).

    slope = (σ_uzun − σ_kısa)/(t_uzun − t_kısa); contango = slope > 0 (kısa vade ucuz).
    <2 vade / geçersiz → None."""
    ts = _clean_1d(expiries, min_len=2)
    vs = _clean_1d(atm_vols, min_len=2)
    if ts is None or vs is None or ts.size != vs.size or np.any(ts <= 0) or np.any(vs <= 0):
        return None
    order = np.argsort(ts)
    ts, vs = ts[order], vs[order]
    if np.any(np.diff(ts) <= 0):
        return None
    w = vs * vs * ts
    dw, dt = np.diff(w), np.diff(ts)
    fwd_var = dw / dt
    fwd_vols = np.where(fwd_var > 0, np.sqrt(np.abs(fwd_var)), np.nan)
    return {
        "expiries": ts, "total_variance": w, "forward_vols": fwd_vols,
        "calendar_free": bool(np.all(dw >= -1e-12)),
        "n_violations": int(np.sum(dw < -1e-12)),
        "slope": float((vs[-1] - vs[0]) / (ts[-1] - ts[0])),
        "contango": bool(vs[-1] > vs[0]),
    }


# ---------------------------------------------------------------------------
# Dupire yerel volatilite (Tur-72 — yol haritası §2.1)
# ---------------------------------------------------------------------------
def dupire_local_vol_svi(slices: object, k_grid: ArrayLike | None = None,
                         n_k: int = 41) -> dict | None:
    """Dupire (1994) YEREL volatilitesini, vade-vade uydurulmuş SVI dilimlerinden çıkarır.

    ÖRTÜK vol bir DİLİM ortalamasıdır: "bu strike ve vadeye kadar ortalama oynaklık ne
    fiyatlanmış?" YEREL vol ise ANLIK'tır: "spot şu anda K'da ve zaman T ise oynaklık
    kaç?" İkisini karıştırmak klasik hatadır — örtük vol yüzeyi düz olsa bile yerel vol
    yüzeyi eğimli olabilir. Gatheral'in toplam-varyans (w = σ²t) biçimindeki Dupire
    denklemi kullanılır (fiyat türevleri yerine w türevleri — sayısal olarak çok daha
    kararlıdır):

        σ_LV²(k,T) = (∂w/∂T) / [ 1 − (k/w)·∂w/∂k
                                 + ¼·(−¼ − 1/w + k²/w²)·(∂w/∂k)²
                                 + ½·∂²w/∂k² ]

    ∂w/∂k ve ∂²w/∂k² SVI'dan ANALİTİK alınır (`_svi_derivatives`); yalnız ∂w/∂T sonlu
    farktır (dilimler ayrık vadelerde). Uçlarda tek-yanlı, içeride eşit-aralıksız
    üç noktalı ikinci derece fark kullanılır.

    İKİ ARBİTRAJ KAPISI, ve ikisi de SESSİZ GEÇİLMEZ:
      - Komşu vadelerde toplam varyans AZALIYORSA takvim arbitrajı → iki uç NaN.
      - Durrleman g < 0 kelebek arbitrajıdır. Pay/payda sıfırsa yerel vol burada
        yine tanımsız bırakılır; sınırdaki eşitlik tek başına arbitraj sayılmaz.
    Yerel varyans NEGATİF çıkamaz; kırpıp pozitif bir sayı basmak, olmayan bir ölçümü
    var göstermek olurdu. Kaç noktanın düştüğü `n_calendar_ihlal`/`n_butterfly_ihlal`
    ile RAPORLANIR.

    `slices`: [{'t': yıl, 'params': {'a','b','rho','m','sigma'}}, ...] — en az 2 dilim,
    pozitif ve BİRBİRİNDEN FARKLI vadeler. Geçersiz/eksik dilim atılır; 2'den az kalırsa
    None. `k_grid` verilmezse [−0,4; +0,4] aralığında `n_k` nokta kullanılır.

    Dönüş: {'k','t','local_vol' (m×n_k, tanımsız nokta NaN), 'implied_vol' (aynı ızgarada
    SVI örtük vol — karşılaştırma için), 'n_calendar_ihlal','n_butterfly_ihlal',
    'n_slices','gecerli_oran'}."""
    if not isinstance(slices, (list, tuple)) or len(slices) < 2:
        return None
    parsed: list[tuple[float, tuple[float, float, float, float, float]]] = []
    for sl in slices:
        if not isinstance(sl, dict):
            continue
        try:
            t_i = float(sl["t"])
        except (KeyError, TypeError, ValueError):
            continue
        pp = _parse_svi(sl.get("params"))
        if pp is None or not math.isfinite(t_i) or t_i <= 0.0:
            continue
        parsed.append((t_i, pp))
    parsed.sort(key=lambda x: x[0])
    # Aynı vadeden birden çok dilim gelirse ilki tutulur (∂w/∂T sıfıra bölünürdü).
    uniq: list[tuple[float, tuple[float, float, float, float, float]]] = []
    for t_i, pp in parsed:
        if not uniq or t_i - uniq[-1][0] > 1e-9:
            uniq.append((t_i, pp))
    if len(uniq) < 2:
        return None
    if k_grid is None:
        try:
            nk = max(5, min(int(n_k), 401))
        except (TypeError, ValueError, OverflowError):
            return None
        ks = np.linspace(-0.4, 0.4, nk)
    else:
        given = _clean_1d(k_grid, min_len=3)
        if given is None:
            return None
        ks = np.sort(given)
    m = len(uniq)
    ts = np.array([t for t, _ in uniq], dtype=float)
    w_all = np.empty((m, ks.size), dtype=float)
    w1_all = np.empty_like(w_all)
    w2_all = np.empty_like(w_all)
    for i, (_t, (a, b, rho, mm, sig)) in enumerate(uniq):
        w_all[i], w1_all[i], w2_all[i] = _svi_derivatives(ks, a, b, rho, mm, sig)
    dwdt = np.asarray(np.gradient(w_all, ts, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        kk = ks[None, :]
        denom = (1.0 - (kk / w_all) * w1_all
                 + 0.25 * (-0.25 - 1.0 / w_all + (kk * kk) / (w_all * w_all)) * w1_all ** 2
                 + 0.5 * w2_all)
        lv2 = dwdt / denom
        iv = np.sqrt(np.maximum(w_all, 0.0) / ts[:, None])
    bad_w = ~np.isfinite(w_all) | (w_all <= 0.0)
    calendar_slope = np.diff(w_all, axis=0) / np.diff(ts)[:, None]
    calendar_slope = np.where(bad_w[:-1] | bad_w[1:], np.nan, calendar_slope)
    cal_intervals = calendar_slope < -1e-10
    cal = np.zeros_like(bad_w)
    cal[:-1] |= cal_intervals
    cal[1:] |= cal_intervals  # merkezî fark ara vadedeki azalışı gizleyemez
    fly = (denom < -1e-10) & ~bad_w
    lv2 = np.where(bad_w | cal | (dwdt <= 0) | (denom <= 0) | ~np.isfinite(lv2) | (lv2 <= 0), np.nan, lv2)
    local_vol = np.sqrt(lv2)
    total = int(local_vol.size)
    valid = int(np.count_nonzero(np.isfinite(local_vol)))
    return {
        "k": np.asarray(ks), "t": [float(x) for x in ts],
        "local_vol": np.asarray(local_vol),
        "implied_vol": np.asarray(np.where(~bad_w & np.isfinite(iv), iv, np.nan)),
        "durrleman_g": np.where(bad_w | ~np.isfinite(denom), np.nan, denom),
        "calendar_slope": np.where(np.isfinite(calendar_slope), calendar_slope, np.nan),
        "arbitrage_tolerance": 1e-10,
        "n_calendar_ihlal": int(np.count_nonzero(cal_intervals)),
        "n_butterfly_ihlal": int(np.count_nonzero(fly)),
        "n_slices": m, "gecerli_oran": (valid / total) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Varyans swap adil kullanım fiyatı — log-kontrat replikasyonu (§7.5)
# ---------------------------------------------------------------------------
def variance_swap_strike(call_strikes: ArrayLike, call_mid: ArrayLike,
                         put_strikes: ArrayLike, put_mid: ArrayLike,
                         forward: float, t: float, r: float = 0.0) -> dict | None:
    """MODELDEN BAĞIMSIZ varyans swap adil kullanım fiyatı (Demeterfi-Derman-Kamal-Zou
    1999 log-kontrat replikasyonu; CBOE'nin VIX'i hesaplarken kullandığı ayrık biçim):

        K_var = (2/T)·Σ_i (ΔK_i/K_i²)·e^{rT}·Q(K_i) − (1/T)·(F/K₀ − 1)²

    Q(K_i) = o strike'taki PARA-DIŞI opsiyonun orta fiyatı (K < K₀ ise put, K > K₀ ise
    call, K₀'da ikisinin ortalaması), K₀ = forward'ın ALTINDAKİ ilk strike,
    ΔK_i = komşu strike'ların yarı farkı.

    Neden "modelden bağımsız": bu formül BİR VOLATİLİTE MODELİ VARSAYMAZ. Log-kontratın
    (−2/T)·E[ln(S_T/F)] beklentisi, tüm strike'lardaki opsiyonlardan 1/K² ağırlıklı bir
    portföyle TAM olarak replike edilir. Black-Scholes'un tersine tek bir σ'ya
    indirgemez — gülümsemenin TAMAMINI kullanır. Bu yüzden K_vol genellikle ATM örtük
    vol'den YÜKSEKTİR: fark, doğrudan gülümsemenin (skew/kurtoz) katkısıdır.

    Ölçümün SINIRI dürüstçe raporlanır: replikasyon SÜREKLİ bir strike sürekliliği
    varsayar; gerçek zincir ayrık ve SONLUdur. `kanat_kirpimi` en dış strike'ın
    forward'a oranını verir — 1'e yakınsa kuyruk hiç örneklenmemiştir ve K_var AŞAĞI
    yanlıdır. Sıçrama (jump) varsa log-kontrat replikasyonu ayrıca yanlıdır; bu formül
    "gerçekleşen varyans" tahmini DEĞİL, o varyansın PİYASA FİYATIdır.

    En az 5 kullanılabilir strike, forward>0, t>0 ister; aksi halde None (İSTİSNA YOK).

    Dönüş: {'k_var' (yıllık varyans), 'k_vol' (=√k_var), 'variance_term','forward_term',
    'k0','forward','t','n_strikes','strikes','contrib' (strike başına katkı),
    'otm_tip' (list['put'|'call'|'atm']), 'kanat_alt','kanat_ust','kanat_kirpimi'}."""
    ck = _clean_1d(call_strikes, min_len=1)
    cm = _clean_1d(call_mid, min_len=1)
    pk = _clean_1d(put_strikes, min_len=1)
    pm = _clean_1d(put_mid, min_len=1)
    if ck is None or cm is None or pk is None or pm is None:
        return None
    if ck.size != cm.size or pk.size != pm.size:
        return None
    fwd = _as_float(forward)
    tt = _as_float(t)
    rr = _as_float(r)
    if fwd is None or tt is None or rr is None or fwd <= 0.0 or tt <= 0.0:
        return None
    calls = {float(k): float(v) for k, v in zip(ck, cm, strict=True) if k > 0 and v > 0}
    puts = {float(k): float(v) for k, v in zip(pk, pm, strict=True) if k > 0 and v > 0}
    below = [k for k in puts if k <= fwd]
    if not below:
        return None
    k0 = max(below)
    grid = sorted(set(list(calls) + list(puts)))
    rows: list[tuple[float, float, str]] = []
    for k in grid:
        if k < k0:
            q, tip = puts.get(k), "put"
        elif k > k0:
            q, tip = calls.get(k), "call"
        else:
            cq, pq = calls.get(k), puts.get(k)
            if cq is not None and pq is not None:
                q, tip = 0.5 * (cq + pq), "atm"
            else:
                q, tip = (cq if cq is not None else pq), "atm"
        if q is not None and q > 0.0 and math.isfinite(q):
            rows.append((k, q, tip))
    if len(rows) < 5:
        return None
    ks = np.array([r_[0] for r_ in rows], dtype=float)
    qs = np.array([r_[1] for r_ in rows], dtype=float)
    dk = np.empty_like(ks)
    dk[1:-1] = (ks[2:] - ks[:-2]) / 2.0
    dk[0] = ks[1] - ks[0]
    dk[-1] = ks[-1] - ks[-2]
    disc = math.exp(rr * tt)
    contrib = (2.0 / tt) * (dk / (ks * ks)) * disc * qs
    variance_term = float(np.sum(contrib))
    forward_term = float((1.0 / tt) * (fwd / k0 - 1.0) ** 2)
    k_var = variance_term - forward_term
    if not math.isfinite(k_var):
        return None
    return {
        "k_var": float(k_var), "k_vol": float(math.sqrt(k_var)) if k_var > 0 else None,
        "variance_term": variance_term, "forward_term": forward_term,
        "k0": float(k0), "forward": float(fwd), "t": float(tt),
        "n_strikes": int(ks.size), "strikes": np.asarray(ks),
        "contrib": np.asarray(contrib), "otm_tip": [r_[2] for r_ in rows],
        "kanat_alt": float(ks[0] / fwd), "kanat_ust": float(ks[-1] / fwd),
        "kanat_kirpimi": float(min(ks[0] / fwd, fwd / ks[-1])),
    }


# ---------------------------------------------------------------------------
# İma edilen korelasyon (dispersion / korelasyon swap çekirdeği)
# ---------------------------------------------------------------------------
_RHO_TOL = 1e-9   # ρ=±1 sınırında kayan-nokta tozu `gecerli` bayrağını çevirmesin


def implied_correlation(index_vol: float, weights: ArrayLike, component_vols: ArrayLike,
                        pairwise: ArrayLike | None = None) -> dict | None:
    """Endeks ile bileşenlerinin örtük vol'lerinden İMA EDİLEN ORTALAMA KORELASYON.

    Portföy varyansı kimliği (σ_I² = Σ_i w_i²σ_i² + Σ_{i≠j} w_i w_j σ_i σ_j ρ_ij) tek bir
    ortak ρ varsayımıyla çözülür:

        ρ_imp = ( σ_I² − Σ_i w_i²σ_i² ) / ( Σ_{i≠j} w_i w_j σ_i σ_j )

    Payda = (Σ_i w_iσ_i)² − Σ_i w_i²σ_i² olarak hesaplanır (O(n), n² çift döngüsü YOK).
    `pairwise` verilirse (gerçekleşen korelasyon matrisi, n×n) aynı payda ile GERÇEKLEŞEN
    ortalama korelasyon da hesaplanır ve `fark = ρ_imp − ρ_realize` döner: dispersion
    ticaretinin ölçtüğü büyüklük tam olarak budur (örtük korelasyon primi).

    NE ÖLÇER, NE ÖLÇMEZ: bu bir FİYAT kimliğidir, tahmin değildir. ρ_imp'in [0,1] dışına
    çıkması bir hata DEĞİL, tutarsız bir girdi kümesinin (farklı vade/kaynak vol'leri,
    eksik ağırlık, endeks ≠ bileşen sepeti) işaretidir; bu yüzden değer KIRPILMAZ ve
    `gecerli` bayrağıyla RAPORLANIR (ρ=±1 gibi SINIR değerler kayan-nokta tozuyla bandın
    dışına düşmesin diye `_RHO_TOL`=1e-9 toleransla; ölçülen: tam ρ=1 girdisinde geri çıkan
    değer 1 + 6,7e−16'dır ve bu bir tutarsızlık DEĞİLDİR).

    Ağırlıklar 1'e normalize EDİLİR (`agirlik_toplami` ham toplamı bildirir; endeksin
    yalnız bir alt kümesi verildiyse bu 1'den küçüktür ve ρ_imp yanlıdır).

    Dönüş: {'rho_imp','pay','payda','sigma_index','sigma_basket_indep','n',
    'agirlik_toplami','gecerli','rho_realized','fark'}. Kısa/NaN/negatif vol, uzunluk
    uyumsuzluğu, sıfır payda (tek bileşen ya da tüm ağırlık tek isimde) → None."""
    w = _clean_1d(weights, min_len=2)
    sig = _clean_1d(component_vols, min_len=2)
    idx_vol = _as_float(index_vol)
    if w is None or sig is None or idx_vol is None:
        return None
    if w.size != sig.size or idx_vol <= 0.0:
        return None
    if np.any(sig < 0.0) or np.any(w < 0.0):
        return None
    ham_toplam = float(np.sum(w))
    if ham_toplam <= 0.0:
        return None
    wn = w / ham_toplam
    ws = wn * sig
    payda = float(np.sum(ws) ** 2 - np.sum(ws**2))
    if payda <= 0.0:
        return None
    bagimsiz = float(np.sum(ws**2))
    pay = float(idx_vol**2 - bagimsiz)
    rho = pay / payda
    out: dict = {
        "rho_imp": rho, "pay": pay, "payda": payda, "sigma_index": float(idx_vol),
        "sigma_basket_indep": math.sqrt(bagimsiz), "n": int(w.size),
        "agirlik_toplami": ham_toplam, "gecerli": bool(-1.0 - _RHO_TOL <= rho <= 1.0 + _RHO_TOL),
        "rho_realized": None, "fark": None,
    }
    if pairwise is None:
        return out
    try:
        corr = np.asarray(pairwise, dtype=float)
    except (TypeError, ValueError):
        return out
    if corr.shape != (w.size, w.size) or not np.all(np.isfinite(corr)):
        return out
    outer = np.outer(ws, ws)
    np.fill_diagonal(outer, 0.0)
    rho_r = float(np.sum(outer * corr) / payda)
    out["rho_realized"] = rho_r
    out["fark"] = rho - rho_r
    return out


def implied_correlation_series(index_vols: ArrayLike, component_vols: ArrayLike, weights: ArrayLike,
                               *, min_kapsama: float = 0.9) -> dict | None:
    """`implied_correlation`ın GÜNLÜK SERİ hâli (Tur-79, yol haritası §7.5 dispersiyon).

    `index_vols` (T,) endeks örtük vol'ü; `component_vols` (T, N) bileşen örtük vol'leri —
    o gün kotasyonu olmayan hücre NaN OLABİLİR; `weights` (N,) negatif olmayan endeks ağırlığı.
    Her gün `implied_correlation`ın kimliği O GÜN ölçülen bileşenlerle çözülür: ağırlıklar
    ölçülen alt kümede 1'e yeniden normalize edilir (aynı fonksiyonun alt küme davranışı).

    KAPSAMA KAPISI: ölçülen bileşenlerin ham ağırlık payı `min_kapsama`nın altındaysa o gün
    ρ ÜRETİLMEZ (NaN, `olculdu=False`). Eksik ağır bir bileşeni sessizce atıp kalanla çözmek
    ρ'yu yanlılaştırır ve bir ölçüm gibi sunulurdu (Tur-31: ölçüm yokluğu ölçüm değildir).

    Dönüş: {'rho_imp' (T,), 'kapsama' (T,), 'n' (T,), 'olculdu' (T,), 'gecerli' (T,)} —
    `gecerli` = ölçüldü VE ρ ∈ [−1, 1] (`_RHO_TOL` ile). Değer KIRPILMAZ. Şekil uyuşmazlığı,
    negatif/sonlu olmayan ağırlık, N<2 → None."""
    try:
        idx = np.asarray(index_vols, dtype=float).reshape(-1)
        comp = np.asarray(component_vols, dtype=float)
        w = np.asarray(weights, dtype=float).reshape(-1)
        kap_min = float(min_kapsama)
    except (TypeError, ValueError):
        return None
    if comp.ndim != 2 or comp.shape[0] != idx.size or comp.shape[1] != w.size or w.size < 2:
        return None
    if idx.size == 0 or not np.all(np.isfinite(w)) or np.any(w < 0.0) or not math.isfinite(kap_min):
        return None
    ham = float(np.sum(w))
    if ham <= 0.0:
        return None
    olcum = np.isfinite(comp) & (comp >= 0.0)
    w_m = np.where(olcum, w[None, :], 0.0)
    w_top = w_m.sum(axis=1)
    kapsama = w_top / ham
    idx_ok = np.isfinite(idx) & (idx > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        wn = w_m / w_top[:, None]
        ws = np.where(olcum, wn * np.where(olcum, comp, 0.0), 0.0)
        bagimsiz = np.sum(ws * ws, axis=1)
        payda = np.sum(ws, axis=1) ** 2 - bagimsiz
        rho = (idx * idx - bagimsiz) / payda
    olculdu = idx_ok & (kapsama >= kap_min) & (w_top > 0.0) & (payda > 0.0) & np.isfinite(rho)
    rho = np.where(olculdu, rho, np.nan)
    gecerli = olculdu & (rho >= -1.0 - _RHO_TOL) & (rho <= 1.0 + _RHO_TOL)
    return {"rho_imp": rho, "kapsama": kapsama, "n": olcum.sum(axis=1).astype(int),
            "olculdu": olculdu, "gecerli": gecerli}


def forward_variance_curve(tenors_years: ArrayLike, vols: ArrayLike) -> dict | None:
    """Sabit vadeli örtük vol'lerden İLERİ VARYANS eğrisi (Tur-79, yol haritası §7.5).

    Toplam varyans w(T) = σ(T)²·T; ardışık iki vade arası ileri varyans
    σ²_ileri(T_i, T_{i+1}) = (w_{i+1} − w_i) / (T_{i+1} − T_i). Varyans swap fiyatlamasının
    (ve Bergomi tipi ileri-varyans modellerinin) girdisidir: piyasanın 30. günden 93. güne
    kadar olan dönem için ŞİMDİ fiyatladığı oynaklık budur.

    KAPALI ÖZDEŞLİK (testte kilitli): Σ_i σ²_ileri,i · ΔT_i = w_K − w_1 (teleskop).

    NEGATİF İLERİ VARYANS bir hesap hatası DEĞİLDİR: takvim arbitrajı (toplam varyansın vadeyle
    azalması) ya da farklı saatte kapanan kotasyonların işaretidir. Değer KIRPILMAZ; vol'ü
    TANIMSIZ (NaN) bırakılır ve `negatif` maskesiyle SAYILIR — sıfıra kırpmak olmayan bir
    ölçümü "sıfır oynaklık" diye sunardı (Tur-33).

    `tenors_years` (K,) kesin artan pozitif; `vols` (K,) ya da (T, K) YILLIK ondalık vol
    (%20 → 0,20). NaN vol yalnız ilgili segmentleri NaN yapar. Dönüş:
    {'toplam_varyans' (T,K), 'fwd_var' (T,K−1), 'fwd_vol' (T,K−1), 'negatif' (T,K−1),
    'segmentler': [(T_i, T_{i+1})]}. Geçersiz vade dizisi / şekil / K<2 → None."""
    tn = _clean_1d(tenors_years, min_len=2)
    if tn is None or np.any(tn <= 0.0) or np.any(np.diff(tn) <= 0.0):
        return None
    try:
        v = np.asarray(vols, dtype=float)
    except (TypeError, ValueError):
        return None
    if v.ndim == 1:
        v = v[None, :]
    if v.ndim != 2 or v.shape[1] != tn.size or v.shape[0] == 0:
        return None
    with np.errstate(invalid="ignore"):
        v = np.where(v >= 0.0, v, np.nan)
        w = v * v * tn[None, :]
        fwd_var = np.diff(w, axis=1) / np.diff(tn)[None, :]
        negatif = fwd_var < 0.0
        fwd_vol = np.where(fwd_var >= 0.0, np.sqrt(np.where(fwd_var >= 0.0, fwd_var, 0.0)), np.nan)
    fwd_vol = np.where(np.isfinite(fwd_var), fwd_vol, np.nan)
    return {"toplam_varyans": w, "fwd_var": fwd_var, "fwd_vol": fwd_vol, "negatif": negatif,
            "segmentler": [(float(a), float(b)) for a, b in zip(tn[:-1], tn[1:], strict=True)]}
