"""
portfolio.py — PORTFÖY YÖNETİMİ (Tur-16 WP-P, docs/AI_V3_DERIN_BEYIN_SPEC.md §3.10). GÖLGE.

KARAR-DIŞI/GÖLGE: bu modül tekil-sinyal dünyasından portföy-düzeyi bir ÖNERİ RAPORUNA
geçiş sağlar. TÜM çıktılar `propose()` çağrısının ürettiği bir sözlük + JSONL gölge
kaydıdır; execution/emir sistemine, strategy/main'e veya META_GATE'e HİÇBİR KOŞULDA
bağlanmaz (anayasa madde 1). Sistem zaten sinyal-only olduğundan bu modülün ürettiği
"ağırlık" hiçbir yerde otomatik pozisyon boyutuna dönüşmez — yalnız RAPORLANIR.

Ana akış: `propose(candidates, closes_panel, constraints=None)` →
  1) closes_panel'den ortak-tarihli log-getiri matrisi çıkar (eksik sembol düşürülür,
     not olarak raporlanır — savunmacı, çökme yok).
  2) Kovaryans: quant_math.ledoit_wolf_cov → (başarısızsa) ewma_cov → (başarısızsa)
     köşegen-varyans yedeği; ayrıca tek-faktör (eşit-ağırlıklı piyasa proxy'si) kovaryansı
     hesaplanır (raporda AYRICA gösterilir); kondisyon sayısı (np.linalg.cond) raporlanır.
  3) YÖNTEM KARŞILAŞTIRMASI: min-varyans, kısıtlı maks-Sharpe, HRP (López de Prado),
     ERC (eşit risk katkısı), Black-Litterman (görüşler = candidates'teki gölge
     posterior+CI genişliği), CVaR-LP (Rockafellar-Uryasev), çok-varlık fractional Kelly
     — HEPSİ AYNI raporda (tek yönteme körlük yok, spec §3.10).
  4) Kısıtlar: pozisyon tavanı (`max_weight`), net-directional cap, ciro cezası (bps).
     Kısıt ihlali MATEMATİKSEL OLARAK İMKÂNSIZ kılınır: her yöntemin ham çıktısı, çözücü
     başarısız/patlamış olsa DAHİ, `_project_capped_simplex` ile kutu-kısıtına (asla
     ihlal edilemez — `np.clip` garantisi) + bütçeye (mümkün olduğunca) projekte edilir.
  5) Stres: tarihsel pencere replay (panelde varsa; yoksa None) + MC (quant_math.mc_gbm_paths,
     varlık-başına bağımsız simülasyon — korelasyon MC'de YOK, tarihsel replay YAKALAR;
     bu sınırlama docstring'lerde açık raporlanır).
  6) Atribüsyon: sembol/strateji bazlı beklenen-getiri katkısı.
  7) `data/ai_portfolio_proposals.jsonl`'e atomik gölge kayıt; metin raporu reasoner.CLOSING
     ile biter (anayasa madde 3: "kenar bulundu" imalı DİL YOK).

QP: `nihai.ai.advanced_math.qp_solve` FONKSİYON İÇİNDE lazy denenir (Dalga-2 paralel
teslim — mevcut olmayabilir); yoksa/hatalıysa KENDİ scipy SLSQP QP yedeğine düşülür
(bkz. `min_variance_weights`). Bu modül advanced_math'i MODÜL SEVİYESİNDE import ETMEZ.

Determinizm: hiçbir yöntem RNG kullanmaz (SLSQP/linprog/HRP hepsi deterministik, sabit
başlangıç noktası x0=1/n); yalnız MC stres testi RNG kullanır ve `seed` parametreli
(quant_math.mc_gbm_paths sözleşmesi — aynı seed+veri → bit-bit aynı çıktı).
"""
from __future__ import annotations

import json
import math
import threading
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform

from nihai.ai import quant_math
from nihai.ai.reasoner import CLOSING
from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR

# Modül-düzeyi yol: testler monkeypatch'ler (episodic.py/decision.py deseni).
PORTFOLIO_PROPOSALS_FILE = DATA_DIR / "ai_portfolio_proposals.jsonl"
_LOCK = threading.Lock()

PORTFOLIO_BANNER = (
    "[GÖLGE PORTFÖY ÖNERİSİ — KARAR-DIŞI] Bu rapor execution/emir sistemine BAĞLANMAZ; "
    "yalnız yöntem-karşılaştırmalı ağırlık önerisidir (anayasa madde 1)."
)

# Varsayılan kısıtlar (propose() constraints=None iken kullanılır; belgeli sabitler).
_DEFAULT_MAX_WEIGHT = 0.30
_DEFAULT_NET_DIRECTIONAL_CAP = 1.0
_DEFAULT_TURNOVER_BPS = 0.0
_DEFAULT_CVAR_ALPHA = 0.05
_DEFAULT_KELLY_FRACTION = 0.25
_DEFAULT_PPY = 252.0

# BL görüş-eşlemesi: posterior 0.5'ten sapma → aşırı-getiri görüşü (belgeli sezgisel
# ölçek; posterior=1.0 → +2*_BL_VIEW_SCALE, posterior=0.0 → -2*_BL_VIEW_SCALE).
_BL_VIEW_SCALE = 0.05
_BL_TAU_DEFAULT = 0.05
_BL_DELTA_DEFAULT = 2.5

# Tarihsel stres pencereleri (YYYY-MM-DD, ISO string karşılaştırması için kapsayıcı).
STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "2008_kriz": ("2008-09-01", "2008-12-31"),
    "2020_covid": ("2020-02-15", "2020-04-15"),
    "2022_ayi": ("2022-01-01", "2022-10-15"),
}

# ponytail: 10 açık senaryo; OCC parametre dosyaları/broker ek teminatları kapsamda değil.
# IV şoku göreli değişimdir: +%20, sigma=0.25'i 0.30 yapar. Ufuk anlıktır (T sabit).
TIMS_SCENARIOS = ((-.12, .20), (-.08, .10), (-.06, 0.), (-.03, -.10), (0., -.20),
                  (0., .20), (.03, -.10), (.06, 0.), (.08, .10), (.12, .20))


def tims_stress(positions: list[dict], available_margin: float | None = None) -> dict | None:
    """TIMS tarzı spot/IV stresi; tüm bacaklar yeniden fiyatlanır, sonra PnL netlenir.

    Pozisyon: kind=stock/c/p, spot, quantity (+uzun/−kısa), multiplier; opsiyonda ayrıca
    strike, t (yıl), sigma, r gerekir; q temettü varsayımı varsayılan 0'dır.
    Farklı dayanaklara aynı göreli spot şoku uygulanır. Avrupa BS, sabit vade/curve;
    erken kullanım, likidite ve broker teminat kuralları modellenmez.
    available_margin verilmezse teminat açığı None'dır; negatifse mevcut açık da sayılır.
    """
    if not isinstance(positions, list) or not positions or len(positions) > 256:
        return None
    try:
        margin = float(available_margin) if available_margin is not None else None
        if margin is not None and not math.isfinite(margin):
            return None
        base_values, shocked_values = [], []
        for pos in positions:
            kind = pos["kind"]
            spot, qty, mult = (float(pos[k]) for k in ("spot", "quantity", "multiplier"))
            if not all(math.isfinite(v) for v in (spot, qty, mult)) or spot <= 0 or mult <= 0:
                return None
            base: float | None
            values: list[float | None]
            if kind == "stock":
                base = spot
                values = [spot * (1 + ds) for ds, _ in TIMS_SCENARIOS]
            elif kind in ("c", "p"):
                strike, t, sigma, r = (float(pos[k]) for k in ("strike", "t", "sigma", "r"))
                q = float(pos.get("q", 0.))
                if (not all(math.isfinite(v) for v in (strike, t, sigma, r, q))
                        or min(strike, t, sigma) <= 0):
                    return None
                prepaid_spot = spot * math.exp(-q * t)
                base = quant_math.bs_price(prepaid_spot, strike, t, r, sigma, kind)
                values = [quant_math.bs_price(prepaid_spot * (1 + ds), strike, t, r, sigma * (1 + dv), kind)
                          for ds, dv in TIMS_SCENARIOS]
            else:
                return None
            if base is None or any(v is None for v in values):
                return None
            base_values.append(base * qty * mult)
            shocked_values.append(np.asarray(values, dtype=float) * qty * mult)
        baseline = np.asarray(base_values)
        pnl_legs = np.asarray(shocked_values).T - baseline
        pnl = pnl_legs.sum(axis=1)
        if not (np.isfinite(pnl_legs).all() and np.isfinite(pnl).all()
                and np.isfinite(baseline).all() and np.isfinite(baseline.sum())):
            return None
        worst = int(np.argmin(pnl))
        requirement = max(0., -float(pnl[worst]))
        return {"scenarios": [{"spot_shock": ds, "iv_shock": dv, "pnl": float(pnl[i]),
                               "leg_pnl": pnl_legs[i].tolist()} for i, (ds, dv) in enumerate(TIMS_SCENARIOS)],
                "baseline_value": float(baseline.sum()), "worst_scenario": worst,
                "stress_requirement": requirement, "available_margin": margin,
                "margin_shortfall": None if margin is None else max(0., requirement - margin),
                "iv_shock_unit": "relative", "n_positions": len(positions),
                "note": "10 varsayımsal anlık spot/IV senaryosu; Avrupa BS ve tüm dayanaklarda ortak şok. OCC/broker teminat hesabı değildir."}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# advanced_math.qp_solve — LAZY (Dalga-2 paralel teslim; yoksa/hatalıysa None)
# ---------------------------------------------------------------------------
def _lazy_qp_solve() -> Any:
    """`nihai.ai.advanced_math.qp_solve`'u fonksiyon-içinde dener. Modül henüz
    yazılmamışsa (ImportError) VEYA fonksiyon henüz eklenmemişse (AttributeError) →
    None (çağıran kendi SLSQP QP yedeğine düşer — Dalga-2 içi bağımlılık gevşek)."""
    try:
        from nihai.ai import advanced_math
    except ImportError:
        return None
    return getattr(advanced_math, "qp_solve", None)


# ---------------------------------------------------------------------------
# Kutu + bütçe kısıtına PROJEKSİYON — kısıt ihlalini MATEMATİKSEL OLARAK imkânsız kılar
# ---------------------------------------------------------------------------
def _resolve_bounds(bounds: tuple[np.ndarray, np.ndarray] | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        return np.zeros(n), np.ones(n)
    lo, hi = bounds
    lo_arr = np.broadcast_to(np.asarray(lo, dtype=float), (n,)).copy()
    hi_arr = np.broadcast_to(np.asarray(hi, dtype=float), (n,)).copy()
    return lo_arr, hi_arr


def _project_capped_simplex(w: np.ndarray, lo: np.ndarray, hi: np.ndarray, budget: float) -> np.ndarray:
    """`w`'yi [lo_i, hi_i] kutusuna + sum(w)≈budget bütçesine projekte eder (su-doldurma /
    kaydırma-ikilik arama). Kutu kısıtı `np.clip` ile HER ZAMAN, çözücü ıraksamış/NaN/Inf
    üretmiş OLSA DAHİ sağlanır (önce nan_to_num+clip, sonra kaydırma) — "kısıt ihlali
    imkânsızlığı" garantisinin matematiksel temeli budur. Bütçe yalnız [sum(lo),sum(hi)]
    içinde ULAŞILABİLİRSE tam sağlanır; aksi halde en yakın ulaşılabilir bütçeye kırpılır
    (kutu kısıtı YİNE bozulmaz — savunmacı öncelik sırası: kutu > bütçe)."""
    w_clean = np.clip(np.nan_to_num(np.asarray(w, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), lo, hi)
    lo_sum, hi_sum = float(np.sum(lo)), float(np.sum(hi))
    target = float(np.clip(budget, lo_sum, hi_sum))
    cur = float(np.sum(w_clean))
    if abs(cur - target) < 1e-9:
        return w_clean
    span = float(np.max(hi) - np.min(lo)) + 10.0

    def f(tau: float) -> float:
        return float(np.sum(np.clip(w_clean + tau, lo, hi))) - target

    lo_tau, hi_tau = -span, span
    f_lo, f_hi = f(lo_tau), f(hi_tau)
    if f_lo > 0.0 or f_hi < 0.0:  # erişilemez (target zaten [lo_sum,hi_sum]'e kırpıldığından pratikte olmaz)
        return w_clean
    for _ in range(100):
        mid = 0.5 * (lo_tau + hi_tau)
        f_mid = f(mid)
        if abs(f_mid) < 1e-12:
            lo_tau = hi_tau = mid
            break
        if f_mid < 0.0:
            lo_tau = mid
        else:
            hi_tau = mid
    tau = 0.5 * (lo_tau + hi_tau)
    return np.clip(w_clean + tau, lo, hi)


# ---------------------------------------------------------------------------
# Kovaryans yardımcıları
# ---------------------------------------------------------------------------
def _single_factor_cov(returns: np.ndarray) -> np.ndarray | None:
    """Tek-faktör (eşit-ağırlıklı piyasa proxy'si) kovaryans modeli: Σ ≈ σ_m²·ββᵀ + diag(idio).
    Gerçek piyasa-kapı verisi YOK — proxy eşit-ağırlıklı ortalama getiridir (belgeli
    basitleştirme). <3 gözlem veya sıfır piyasa-varyansı → None (savunmacı)."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[0] < 3 or not np.all(np.isfinite(r)):
        return None
    mkt = r.mean(axis=1)
    var_m = float(np.var(mkt, ddof=1))
    if var_m <= 0.0:
        return None
    betas = np.array([float(np.cov(r[:, i], mkt, ddof=1)[0, 1]) / var_m for i in range(r.shape[1])])
    resid = r - np.outer(mkt, betas)
    idio_var = np.var(resid, axis=0, ddof=1)
    idio_var = np.where(np.isfinite(idio_var) & (idio_var >= 0.0), idio_var, 0.0)
    return var_m * np.outer(betas, betas) + np.diag(idio_var)


def _condition_number(cov: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(cov))
    except np.linalg.LinAlgError:
        return float("inf")


def _inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    """Ters-volatilite ağırlığı — BL'in 'piyasa dengesi' önseli için PROXY.

    ÖNCÜL BAYATLADI (Tur-79): *"gerçek piyasa kapitalizasyonu verisi bu modülde YOK"*
    cümlesi yazıldığında doğruydu; `nihai/data/etf_holdings` artık SPDR'ın günlük sepet
    dosyasından GERÇEK kapitalizasyon ağırlıklarını okuyor. Bu modül VERİ katmanını
    import ETMEZ (katman kuralı) → gerçek ağırlıklar `compare_methods`/`propose`'a
    DIŞARIDAN verilir; verilmezse bu proxy kullanılır ve kaynak çıktıda BEYAN edilir.
    Proxy bir yaklaşıklık DEĞİL FARKLI bir önseldir: ters-vol düşük oynaklıklı ismi
    piyasadan bağımsız olarak ağırlaştırır (π = δΣw sonucu bambaşka çıkar)."""
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    inv = 1.0 / std
    s = float(inv.sum())
    return inv / s if s > 0 else np.full(cov.shape[0], 1.0 / cov.shape[0])


# ---------------------------------------------------------------------------
# Klasik optimizasyon: min-varyans, kısıtlı maks-Sharpe
# ---------------------------------------------------------------------------
def min_variance_weights(cov: Any, bounds: tuple[np.ndarray, np.ndarray] | None = None,
                         budget: float = 1.0) -> np.ndarray | None:
    """Min-varyans ağırlıkları: minimize wᵀΣw, s.t. sum(w)=budget, lo<=w<=hi. Önce
    `advanced_math.qp_solve` (varsa) dener, aksi halde kendi SLSQP QP yedeğine düşer.
    Geçersiz (kare-olmayan/NaN/Inf) kovaryans → None."""
    cov_arr = np.asarray(cov, dtype=float)
    n = cov_arr.shape[0]
    if cov_arr.ndim != 2 or cov_arr.shape != (n, n) or n < 1 or not np.all(np.isfinite(cov_arr)):
        return None
    lo, hi = _resolve_bounds(bounds, n)
    a_eq = np.ones((1, n))
    b_eq = np.array([budget])
    qp = _lazy_qp_solve()
    if qp is not None:
        try:
            w_qp = qp(cov_arr, np.zeros(n), a_eq=a_eq, b_eq=b_eq, lb=lo, ub=hi)
        except Exception as e:  # noqa: BLE001 — Dalga-2 paralel teslim, gevşek bağımlılık
            logger.debug(f"[ai.portfolio] advanced_math.qp_solve hata, SLSQP yedeğe düşülüyor: {e}")
            w_qp = None
        if w_qp is not None:
            return _project_capped_simplex(np.asarray(w_qp, dtype=float), lo, hi, budget)

    x0 = np.full(n, budget / n)

    def obj(w: np.ndarray) -> float:
        return float(w @ cov_arr @ w)

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - budget)}]
    b = list(zip(lo.tolist(), hi.tolist(), strict=True))
    res = minimize(obj, x0, method="SLSQP", bounds=b, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-14})
    w_out = np.asarray(res.x, dtype=float) if res.success else x0
    return _project_capped_simplex(w_out, lo, hi, budget)


def max_sharpe_weights(mu: Any, cov: Any, bounds: tuple[np.ndarray, np.ndarray] | None = None,
                       budget: float = 1.0, rf: float = 0.0) -> np.ndarray | None:
    """Kısıtlı maks-Sharpe ağırlıkları (SLSQP, negatif-Sharpe'ı minimize). Geçersiz girdi
    → None; sıfır-varyans noktalarında büyük ceza (1e10) ile ıraksama önlenir."""
    mu_arr = np.asarray(mu, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)
    n = mu_arr.shape[0]
    if cov_arr.shape != (n, n) or n < 1 or not np.all(np.isfinite(mu_arr)) or not np.all(np.isfinite(cov_arr)):
        return None
    lo, hi = _resolve_bounds(bounds, n)
    x0 = np.full(n, budget / n)

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ mu_arr - rf)
        port_var = float(w @ cov_arr @ w)
        if port_var <= 1e-18:
            return 1e10
        return -port_ret / math.sqrt(port_var)

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - budget)}]
    b = list(zip(lo.tolist(), hi.tolist(), strict=True))
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=b, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    w_out = np.asarray(res.x, dtype=float) if res.success else x0
    return _project_capped_simplex(w_out, lo, hi, budget)


# ---------------------------------------------------------------------------
# HRP — Hierarchical Risk Parity (López de Prado)
# ---------------------------------------------------------------------------
def _corr_linkage(cov: np.ndarray) -> np.ndarray:
    """HRP/HERC/NCO için ortak korelasyon mesafesi ve single-linkage ağacı."""
    std = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # simetri float-artığını temizle (squareform şart koşar)
    condensed = squareform(dist, checks=False)
    return linkage(condensed, method="single")


def _quasi_diag_order(cov: np.ndarray) -> list[int]:
    """Ortak korelasyon ağacının yaprak sırası (LdP quasi-diagonalization)."""
    if cov.shape[0] <= 2:
        return list(range(cov.shape[0]))
    return [int(x) for x in dendrogram(_corr_linkage(cov), no_plot=True)["leaves"]]


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    """Küme-içi TERS-VARYANS ağırlıklı portföy varyansı (LdP konvansiyonu — küme-içi
    çapraz-terimleri yok sayar, yalnız köşegeni kullanır; hız/basitlik için standart)."""
    sub = cov[np.ix_(idx, idx)]
    diag = np.clip(np.diag(sub), 1e-18, None)
    inv = 1.0 / diag
    w = inv / inv.sum()
    return float(w @ sub @ w)


def _hrp_recursive_bisection(cov: np.ndarray, sort_ix: list[int]) -> np.ndarray:
    n = cov.shape[0]
    w: dict[int, float] = dict.fromkeys(range(n), 1.0)
    clusters: list[list[int]] = [list(sort_ix)]
    while clusters:
        next_level: list[list[int]] = []
        for c in clusters:
            if len(c) <= 1:
                continue
            mid = len(c) // 2
            c0, c1 = c[:mid], c[mid:]
            var0 = _cluster_var(cov, c0)
            var1 = _cluster_var(cov, c1)
            denom = var0 + var1
            alpha = (1.0 - var0 / denom) if denom > 0 else 0.5
            for i in c0:
                w[i] *= alpha
            for i in c1:
                w[i] *= (1.0 - alpha)
            next_level.append(c0)
            next_level.append(c1)
        clusters = next_level
    return np.array([w[i] for i in range(n)], dtype=float)


def hrp_weights(cov: Any, bounds: tuple[np.ndarray, np.ndarray] | None = None,
               budget: float = 1.0) -> np.ndarray | None:
    """Hierarchical Risk Parity (López de Prado): korelasyon-kümeleme (quasi-diag) +
    özyinelemeli risk-paylaşımı (bisection). Kovaryans-TERS-ÇEVİRME gerektirmediğinden
    (yalnız köşegen/alt-matris kullanır) min-var/max-Sharpe'a göre dejenere/ill-conditioned
    kovaryansta daha DAYANIKLIDIR (spec §3.10 gerekçesi). n=2 için TERS-VARYANS ağırlığına
    matematiksel olarak eşdeğerdir (bilinen-küçük-örnek test temeli). Geçersiz girdi → None."""
    cov_arr = np.asarray(cov, dtype=float)
    n = cov_arr.shape[0]
    if cov_arr.ndim != 2 or cov_arr.shape != (n, n) or n < 1 or not np.all(np.isfinite(cov_arr)):
        return None
    lo, hi = _resolve_bounds(bounds, n)
    if n == 1:
        return _project_capped_simplex(np.array([1.0]), lo, hi, budget)
    try:
        order = _quasi_diag_order(cov_arr)
        raw = _hrp_recursive_bisection(cov_arr, order)
    except Exception as e:  # noqa: BLE001 — hesap hatası çökmeye değil None'a düşer
        logger.debug(f"[ai.portfolio] hrp hesap hatası: {e}")
        return None
    return _project_capped_simplex(raw, lo, hi, budget)


# ---------------------------------------------------------------------------
# ERC — Equal Risk Contribution
# ---------------------------------------------------------------------------
def _erc_spinu(cov: np.ndarray) -> np.ndarray | None:
    """Uzun-yalnız, bütçesi 1 ERC: min ½yᵀCy − (1/n)Σ log y (Spinu 2013), C korelasyon.

    Log bariyer kesin konveks → tekil çözüm; KKT y_i(Cy)_i = 1/n doğrudan eşit risk
    katkısıdır. Sönümlü Newton (pozitiflik korunur). w_i ∝ y_i/σ_i kovaryans ölçeğine döner.

    Durma ölçütü göreli pay kalıntısıdır max|n·y_i(Cy)_i − 1|. Optimuma yakın Armijo testi
    YUVARLAMA gürültüsüne takılır (f'nin değişimi ~1e-16 altına iner) — Tur-78 ölçümü: katı test
    09-14 üretecinin 400 kovaryansının 6'sında (cond 42–89) MAXITER'e düştü ve SLSQP yedeği bir
    ağırlığı 0'a yapıştırdı (pay hatası tam 1/n). Bu yüzden Armijo'ya yuvarlama payı verilir ve
    en iyi iterasyon tutulur; kalıntısı 1e-6'yı aşarsa (cond ≳ 1e8 bölgesi) None."""
    n = cov.shape[0]
    sd = np.sqrt(np.diag(cov))
    if not (np.isfinite(sd).all() and (sd > 0).all()):
        return None
    c = cov / np.outer(sd, sd)
    s = float(np.ones(n) @ c @ np.ones(n))
    if not math.isfinite(s) or s <= 0:
        return None
    y = np.full(n, 1.0 / math.sqrt(s))
    best, best_res = y, math.inf

    def f(v: np.ndarray) -> float:
        return float(0.5 * v @ c @ v - np.log(v).mean())

    for _ in range(100):
        cy = c @ y
        res = float(np.max(np.abs(n * y * cy - 1.0)))
        if res < best_res:
            best, best_res = y, res
        if res < 1e-10:
            break
        g = cy - 1.0 / (n * y)
        try:
            d = -np.linalg.solve(c + np.diag(1.0 / (n * y * y)), g)
        except np.linalg.LinAlgError:
            break
        slope = float(g @ d)
        if not np.isfinite(d).all() or slope >= 0:
            break
        t, f0 = 1.0, f(y)
        tol_f = 1e-12 * max(1.0, abs(f0))
        # `or` kısa devresi: f yalnız pozitif noktada değerlendirilir.
        while ((y + t * d) <= 0).any() or f(y + t * d) > f0 + 1e-4 * t * slope + tol_f:
            t *= 0.5
            if t < 1e-12:
                break
        else:
            y = y + t * d
            continue
        break
    if best_res > 1e-6:
        return None
    w = best / sd
    return w / w.sum()


def erc_weights(cov: Any, bounds: tuple[np.ndarray, np.ndarray] | None = None,
               budget: float = 1.0) -> np.ndarray | None:
    """Eşit Risk Katkısı (ERC) ağırlıkları: her varlığın toplam portföy riskine katkı PAYI
    eşitlenir (RC_i = w_i·(Σw)_i / (wᵀΣw) → 1/n hedefi). Kutu bağlamıyorsa konveks Spinu
    çözümü (`_erc_spinu`, kesin); bağlıyorsa o çözümden başlayan SLSQP (genel kutu-kısıtlı
    durumda kapalı-form yok). Köşegen (korelasyonsuz) kovaryansta KAPALI-FORM çözümle
    örtüşür: w_i ∝ 1/σ_i (bilinen-örnek test temeli). Geçersiz girdi → None."""
    cov_arr = np.asarray(cov, dtype=float)
    n = cov_arr.shape[0]
    if cov_arr.ndim != 2 or cov_arr.shape != (n, n) or n < 1 or not np.all(np.isfinite(cov_arr)):
        return None
    lo, hi = _resolve_bounds(bounds, n)
    if n == 1:
        return _project_capped_simplex(np.array([1.0]), lo, hi, budget)
    # Tur-78: aşağıdaki SLSQP amacı konveks DEĞİL; w_i=0 sınırında sahte durağan noktalar var.
    # Ölçüldü: rastgele iyi koşullu (cond 15–175) 400 kovaryansın %37'sinde (149) payları 1/n'den
    # 0,12–0,33 saptı, 2 varlıkta kapalı formdan 0,64. Kutu bağlamıyorsa konveks çözüm kesindir:
    # aynı 400'de 0 sapma, gerçekçi 1500 Ledoit-Wolf kovaryansında 0 None (maks pay hatası 1,5e-11).
    exact = _erc_spinu(cov_arr)
    if exact is not None:
        w_exact = budget * exact
        if (w_exact >= lo - 1e-12).all() and (w_exact <= hi + 1e-12).all():
            return w_exact
    x0 = _project_capped_simplex(budget * exact, lo, hi, budget) if exact is not None else np.full(n, budget / n)
    target = 1.0 / n

    def obj(w: np.ndarray) -> float:
        port_var = float(w @ cov_arr @ w)
        if port_var <= 1e-18:
            return 1e10
        mrc = cov_arr @ w
        rc_norm = (w * mrc) / port_var
        return float(np.sum((rc_norm - target) ** 2))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - budget)}]
    b = list(zip(lo.tolist(), hi.tolist(), strict=True))
    res = minimize(obj, x0, method="SLSQP", bounds=b, constraints=cons,
                   options={"maxiter": 800, "ftol": 1e-16})
    if res.success:
        return _project_capped_simplex(np.asarray(res.x, dtype=float), lo, hi, budget)
    # yakınsamazsa ters-volatilite sezgisi (korelasyonsuzsa TAM ERC çözümü) — deterministik yedek
    std = np.sqrt(np.clip(np.diag(cov_arr), 1e-18, None))
    inv = 1.0 / std
    w_fallback = budget * inv / inv.sum()
    return _project_capped_simplex(w_fallback, lo, hi, budget)


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------
def herc_weights(cov: Any, n_clusters: int | None = None,
                 bounds: tuple[np.ndarray, np.ndarray] | None = None,
                 budget: float = 1.) -> np.ndarray | None:
    """Planın küme-içi ERC içeren HERC varyantı; gerçek dendrogram dalları izlenir.

    Terminal kümelerde tam kovaryansla ERC; iki kardeş küme portföyünün risk katkısı
    ters standart sapma oranıyla eşitlenir. Kümeler, sıralı varlıkları ortadan ikiye
    keserek oluşturulmaz. ERC yereldir; tüm varlıkların global ERC'si iddia edilmez.
    Sonradan ağırlık kırpmak yerel ERC'yi bozacağından, sınırlar buna izin vermiyorsa
    None döner. Ağ ve ERC hesabı veri içinde deterministiktir, OOS üstünlük iddiası yok.
    """
    try:
        mat = np.asarray(cov, dtype=float)
        n = mat.shape[0] if mat.ndim == 2 else 0
        if (not n or mat.shape != (n, n) or not np.isfinite(mat).all()
                or not np.allclose(mat, mat.T) or (np.diag(mat) <= 0).any()
                or not math.isfinite(budget) or budget <= 0):
            return None
        if np.linalg.eigvalsh(mat).min() < -1e-10 * np.diag(mat).max():
            return None
        lo, hi = _resolve_bounds(bounds, n)
        if np.isnan(lo).any() or np.isnan(hi).any() or (lo > hi).any():
            return None
        # ponytail: mevcut NCO gibi sqrt(N) kesimi; küme sayısı açık parametredir, taranmaz.
        k = max(1, int(round(math.sqrt(n)))) if n_clusters is None else n_clusters
        if not isinstance(k, (int, np.integer)) or not 1 <= k <= n:
            return None
        tree = _corr_linkage(mat) if n > 1 else np.empty((0, 4))
        labels = fcluster(tree, t=k, criterion="maxclust") if n > 1 else np.ones(1)
        members = {i: [i] for i in range(n)}
        for i, row in enumerate(tree):
            members[n + i] = members[int(row[0])] + members[int(row[1])]

        def allocate(node: int) -> np.ndarray | None:
            ids = members[node]
            if len(set(labels[ids])) == 1:
                sub = mat[np.ix_(ids, ids)]
                inner = erc_weights(sub / np.diag(sub).max())
                if inner is None:
                    return None
                risk = float(inner @ sub @ inner)
                if risk <= 0 or np.max(np.abs(inner * (sub @ inner) / risk - 1 / len(ids))) > 1e-5:
                    return None
                out = np.zeros(n)
                out[ids] = inner
                return out
            left = allocate(int(tree[node - n, 0]))
            right = allocate(int(tree[node - n, 1]))
            if left is None or right is None:
                return None
            vl, vr = float(left @ mat @ left), float(right @ mat @ right)
            if min(vl, vr) <= 0:
                return None
            share = math.sqrt(vr) / (math.sqrt(vl) + math.sqrt(vr))
            return share * left + (1 - share) * right

        raw = allocate(2 * n - 2)
        if raw is None:
            return None
        weights = budget * raw
        total_risk = float(weights @ mat @ weights)
        if (not np.isfinite(weights).all() or (weights < lo - 1e-8).any()
                or (weights > hi + 1e-8).any() or not math.isfinite(total_risk) or total_risk <= 0):
            return None
        return weights
    except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError):
        return None


def black_litterman(cov: Any, market_weights: Any, views: list[dict[str, Any]] | None = None,
                    tau: float = _BL_TAU_DEFAULT, delta: float = _BL_DELTA_DEFAULT) -> dict | None:
    """Black-Litterman posterior beklenen-getiri + kovaryans (standart formülasyon):
      π = δ·Σ·w_mkt                                     (piyasa-dengesi ima edilen getiri)
      μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹·[(τΣ)⁻¹π + PᵀΩ⁻¹Q]      (görüş-ağırlıklı posterior)
      Σ_BL = Σ + [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹

    `views`: [{'index': i, 'q': aşırı-getiri, 'confidence': 0..1}, ...] — P her görüş için
    tekil varlık üstünde MUTLAK görüş (P satırı bir-hot; göreli görüş desteklenmez, spec
    kapsamı yeterli). Ω köşegeni güvenden türetilir: düşük güven → yüksek varyans
    (standart LdP/He-Litterman sezgisi: omega_i = (1/conf_i - 1)·τ·(P_iΣP_iᵀ)).

    GÖRÜŞ YOKSA (views None/boş): μ_BL == π (SAF piyasa-dengesi) — bu, "görüş-yoksa
    piyasa-ağırlığına limit" özelliğinin matematiksel temelidir (π=δΣw_mkt ile maks-Sharpe
    koşulursa tanjant portföyü TAM OLARAK w_mkt'e eşittir, ölçek/normalize farkı hariç).
    Geçersiz (kare-olmayan/NaN/Inf) girdi → None.

    Dönüş: {'mu_bl','cov_bl','pi','n_views'}."""
    cov_arr = np.asarray(cov, dtype=float)
    mw = np.asarray(market_weights, dtype=float)
    n = cov_arr.shape[0]
    if (cov_arr.ndim != 2 or cov_arr.shape != (n, n) or mw.shape != (n,) or n < 1
            or not np.all(np.isfinite(cov_arr)) or not np.all(np.isfinite(mw)) or tau <= 0.0):
        return None
    pi = delta * cov_arr @ mw
    if not views:
        return {"mu_bl": pi, "cov_bl": cov_arr + tau * cov_arr, "pi": pi, "n_views": 0}
    try:
        idxs = [int(v["index"]) for v in views]
        q_vec = np.array([float(v["q"]) for v in views], dtype=float)
        conf = np.array([_clip(float(v.get("confidence", 0.5)), 1e-6, 1.0) for v in views], dtype=float)
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"[ai.portfolio] BL görüş formatı bozuk: {e}")
        return {"mu_bl": pi, "cov_bl": cov_arr + tau * cov_arr, "pi": pi, "n_views": 0}
    if any(i < 0 or i >= n for i in idxs) or not np.all(np.isfinite(q_vec)):
        return {"mu_bl": pi, "cov_bl": cov_arr + tau * cov_arr, "pi": pi, "n_views": 0}
    p_mat = np.zeros((len(views), n))
    for row, i in enumerate(idxs):
        p_mat[row, i] = 1.0
    tau_sigma = tau * cov_arr
    omega_diag = np.array([
        (1.0 / conf[k] - 1.0) * tau * float(p_mat[k] @ cov_arr @ p_mat[k]) if conf[k] < 1.0 else 1e-10
        for k in range(len(views))
    ])
    omega_diag = np.clip(omega_diag, 1e-12, None)
    omega = np.diag(omega_diag)
    inv_tau_sigma = np.linalg.pinv(tau_sigma)
    inv_omega = np.linalg.pinv(omega)
    m_mat = inv_tau_sigma + p_mat.T @ inv_omega @ p_mat
    m_inv = np.linalg.pinv(m_mat)
    mu_bl = m_inv @ (inv_tau_sigma @ pi + p_mat.T @ inv_omega @ q_vec)
    cov_bl = cov_arr + m_inv
    return {"mu_bl": mu_bl, "cov_bl": cov_bl, "pi": pi, "n_views": len(views)}


def _build_bl_views(candidates: list[dict[str, Any]], symbols: list[str]) -> list[dict[str, Any]]:
    """`candidates`'teki gölge posterior + conformal/CI genişliğinden BL görüşü türetir
    (spec §3.10: "görüşler = candidates içindeki gölge posterior+CI"). posterior yok/NaN
    veya sembol evrende yok → o aday atlanır (savunmacı, hata fırlatmaz)."""
    idx = {s: i for i, s in enumerate(symbols)}
    views: list[dict[str, Any]] = []
    for c in candidates or []:
        if not isinstance(c, dict):
            continue
        sym = c.get("symbol")
        if sym not in idx:
            continue
        post_raw = c.get("posterior", c.get("shadow_posterior", c.get("post")))
        if post_raw is None:
            continue
        try:
            p = float(post_raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(p):
            continue
        q = (p - 0.5) * 2.0 * _BL_VIEW_SCALE
        conf = 0.5
        ci = c.get("conformal_interval", c.get("ci"))
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            try:
                width = abs(float(ci[1]) - float(ci[0]))
                conf = _clip(1.0 - width, 0.05, 0.95)
            except (TypeError, ValueError):
                pass
        views.append({"index": idx[sym], "q": q, "confidence": conf})
    return views


# ---------------------------------------------------------------------------
# CVaR-LP (Rockafellar-Uryasev)
# ---------------------------------------------------------------------------
def cvar_lp_weights(returns: Any, alpha: float = _DEFAULT_CVAR_ALPHA,
                    bounds: tuple[np.ndarray, np.ndarray] | None = None,
                    budget: float = 1.0) -> np.ndarray | None:
    """CVaR-minimize eden ağırlıklar (Rockafellar-Uryasev LP formülasyonu, scipy.linprog):
    değişkenler w(N), ζ (VaR-proxy), u_t(T,>=0); minimize ζ + 1/(α·T)·Σu_t s.t.
    u_t >= -(r_tᵀw) - ζ, sum(w)=budget, lo<=w<=hi. `returns` T×N (satır=senaryo/bar).

    **α = KUYRUK OLASILIĞIDIR** (0,05 = en kötü %5), `quant_math.cvar_historical` ile AYNI
    konvansiyon. BULGU-73-1: payda önceden `(1−α)·T` idi — bu Rockafellar-Uryasev'in
    GÜVEN DÜZEYİ biçimidir ve varsayılan α=0,05 ile en kötü **%95**'i ortalıyordu, yani
    ortalama kaybı minimize ediyordu. Ölçüldü (yavaş-kanama ↔ keskin-şok panelinde, kaba
    kuvvet ızgarasına karşı): eski yol w=[0,00; 1,00] verip CVaR@5%'i **0,017720**'de
    bırakıyordu; doğru payda ile w=[0,7744; 0,2256] ve CVaR@5% **0,007225** (201 noktalı
    kaba kuvvet en iyisi 0,007301 — LP sürekli uzayda daha iyisini buluyor).
    Geçersiz girdi (T<2, N<1, NaN/Inf, alpha dışı-(0,1)) veya LP başarısız → None."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[0] < 2 or r.shape[1] < 1 or not np.all(np.isfinite(r)):
        return None
    if not (0.0 < alpha < 1.0):
        return None
    t_n, n_assets = r.shape
    lo, hi = _resolve_bounds(bounds, n_assets)
    nvar = n_assets + 1 + t_n
    c = np.zeros(nvar)
    c[n_assets] = 1.0
    c[n_assets + 1:] = 1.0 / (alpha * t_n)      # BULGU-73-1: α = KUYRUK olasılığı
    a_ub = np.zeros((t_n, nvar))
    for t in range(t_n):
        a_ub[t, :n_assets] = -r[t]
        a_ub[t, n_assets] = -1.0
        a_ub[t, n_assets + 1 + t] = -1.0
    b_ub = np.zeros(t_n)
    a_eq = np.zeros((1, nvar))
    a_eq[0, :n_assets] = 1.0
    b_eq = np.array([budget])
    bnds = list(zip(lo.tolist(), hi.tolist(), strict=True)) + [(None, None)] + [(0.0, None)] * t_n
    try:
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bnds, method="highs")
    except ValueError as e:
        logger.debug(f"[ai.portfolio] cvar_lp linprog hata: {e}")
        return None
    if not res.success:
        return None
    w_out = np.asarray(res.x[:n_assets], dtype=float)
    return _project_capped_simplex(w_out, lo, hi, budget)


# ---------------------------------------------------------------------------
# Çok-varlık fractional Kelly (bütçe/simpleks kısıtı YOK — kaldıraç doğası gereği)
# ---------------------------------------------------------------------------
def _cvar_tail_weights(losses: np.ndarray, alpha: float) -> np.ndarray:
    """Tam alpha*T kuyruk kütlesi; sınırdaki eşit kayıplar aynı payı alır."""
    mass = alpha * len(losses)
    cutoff = np.partition(losses, len(losses) - math.ceil(mass))[len(losses) - math.ceil(mass)]
    above, tied = losses > cutoff, losses == cutoff
    prob = above.astype(float) / mass
    prob[tied] = max(0., 1 - prob.sum()) / tied.sum()
    return prob


def cvar_risk_parity(returns: Any, alpha: float = .05,
                     bounds: tuple[np.ndarray, np.ndarray] | None = None,
                     budget: float = 1.) -> dict | None:
    """Uzun-yalnız eşit ES katkısı: min ES(x) − mean(log x), sonra bütçeye ölçekle.

    ES örneklemde parçalı doğrusaldır. Destek düzlemleriyle küçük, konveks log problem
    çözülür; böylece her iterasyonda T ek slack değişkenli yoğun NLP kurulmaz.
    Kuyruk eşitliklerinde geçerli bir ES alt-türevi LP ile denetlenir. Ölçekleme sonrası
    Euler payları 1/N'den 1e-5 fazla ayrışırsa veya kutu sınırı bozulursa None döner.
    Sıfır/negatif minimum ES'de log problemin pozitif-risk varsayımı sağlanmaz → None.

    KİNK (Tur-78, P3): optimum çoğunlukla örneklem VaR'ında kayıpların eşitlendiği bir kinktedir; orada ES
    türevlenemez ve Euler katkısı TEK DEĞİLDİR. `kink=True` iken `shares` göreli 1e-7 bağ toleransındaki
    alt-türevlerden pariteye EN YAKIN olanıdır (minimaks LP); tek kuyruk atamasıyla yeniden hesaplanan paylar
    ağırlıktaki yuvarlama ölçeğinde bir sapmayla çok farklı çıkabilir. Ölçüldü (300 problem, bağımsız düzgün
    dual + KKT referansı): ağırlık farkı ≤4,4e-7, pay hatası ≤4,8e-7; eski tam-eşitlik LP'si çözümü OLAN
    40 problemde None dönüyordu, bant (A_ub ±1e-5) biçimi ise köşe çözümü bant sınırına oturduğu için 202.
    """
    try:
        r = np.asarray(returns, dtype=float)
        if (r.ndim != 2 or r.shape[0] < 20 or r.shape[1] < 1 or not np.isfinite(r).all()
                or not math.isfinite(alpha) or not 0 < alpha < 1
                or not math.isfinite(budget) or budget <= 0):
            return None
        t, n = r.shape
        lo, hi = _resolve_bounds(bounds, n)
        if (np.isnan(lo).any() or np.isnan(hi).any() or (lo > hi).any()
                or (hi <= 0).any() or np.maximum(lo, 0).sum() > budget or hi.sum() < budget):
            return None
        scale = max(float(np.std(r, axis=0, ddof=1).mean()), float(np.abs(r).mean()))
        if not math.isfinite(scale) or scale <= 0:
            return None
        scaled = r / scale
        min_w = cvar_lp_weights(scaled, alpha)
        if min_w is None:
            return None
        min_losses = -scaled @ min_w
        min_es = float(_cvar_tail_weights(min_losses, alpha) @ min_losses)
        if min_es <= 1e-10:
            return None
        x0 = np.full(n, 1 / n)
        loss0 = -scaled @ x0
        marginal0 = -scaled.T @ _cvar_tail_weights(loss0, alpha)
        # Pozitif alt sınır, ilk destek düzlemlerinin log problemini sınırlı tutar.
        cuts = [marginal0, np.full(n, min_es * .5)]
        y = np.r_[x0, float(marginal0 @ x0)]

        def objective(z):
            return float(z[-1] - np.log(z[:-1]).mean())

        def gradient(z):
            return np.r_[-1 / (n * z[:-1]), 1.]

        for _iteration in range(100):
            a = np.asarray(cuts)
            jac = np.column_stack([-a, np.ones(len(a))])
            fit = minimize(objective, y, jac=gradient, method="SLSQP",
                           bounds=[(1e-10, None)] * n + [(None, None)],
                           constraints=[{"type": "ineq", "fun": lambda z, a=a: z[-1] - a @ z[:-1],
                                         "jac": lambda z, jac=jac: jac}],
                           options={"ftol": 1e-12, "maxiter": 300})
            if not fit.success or not np.isfinite(fit.x).all():
                return None
            y = np.asarray(fit.x)
            losses = -scaled @ y[:-1]
            marginal = -scaled.T @ _cvar_tail_weights(losses, alpha)
            risk = float(marginal @ y[:-1])
            if risk - y[-1] <= 1e-8 * max(1., abs(risk)):
                break
            cuts.append(marginal)
        else:
            return None
        w = budget * y[:-1] / y[:-1].sum()
        if (w < lo - 1e-8).any() or (w > hi + 1e-8).any():
            return None
        losses = -r @ w
        prob = _cvar_tail_weights(losses, alpha)
        risk = float(prob @ losses)
        if risk <= 0 or not math.isfinite(risk):
            return None
        shares = w * (-r.T @ prob) / risk
        kink = bool(np.max(np.abs(shares - 1 / n)) > 1e-5)
        if kink:
            # Örneklem VaR'ındaki kink'te tek bir gradyan yoktur. Geçerli kuyruk ağırlıkları
            # içinde pariteye EN YAKIN olanı bul (min s: |euler·q − 1/N| ≤ s); kabul yine 1e-5.
            # Tam eşitlik (A_eq) yaklaşık adayda olursuzdu; ±1e-5 bandı köşeyi bant sınırına iter.
            mass = alpha * t
            cutoff = np.partition(losses, t - math.ceil(mass))[t - math.ceil(mass)]
            tol = max(1e-12, float(np.max(np.abs(losses))) * 1e-7)
            cap = 1 / mass
            q_bounds = [(cap, cap) if loss > cutoff + tol else
                        (0., 0.) if loss < cutoff - tol else (0., cap) for loss in losses]
            euler = (-r.T) * w[:, None] / risk
            ones = np.ones((n, 1))
            dual = linprog(np.r_[np.zeros(t), 1.], A_ub=np.block([[euler, -ones], [-euler, -ones]]),
                           b_ub=np.r_[np.full(n, 1 / n), np.full(n, -1 / n)],
                           A_eq=np.r_[np.ones(t), 0.][None, :], b_eq=[1.],
                           bounds=q_bounds + [(0., None)], method="highs")
            if not dual.success:
                return None
            prob = np.asarray(dual.x[:t])
            shares = euler @ prob
            if abs(float(prob @ losses) - risk) > 1e-7 * risk:
                return None
        error = float(np.max(np.abs(shares - 1 / n)))
        if error > 1e-5:
            return None
        return {"weights": w, "es": risk, "contributions": shares * risk, "shares": shares,
                "max_error": error, "alpha": alpha, "n_tail": int(np.count_nonzero(prob > 1e-12)),
                "iterations": _iteration + 1, "kink": kink}
    except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError):
        return None


def multi_asset_kelly(mu: Any, cov: Any, fraction: float = _DEFAULT_KELLY_FRACTION,
                      cap: float = _DEFAULT_MAX_WEIGHT) -> np.ndarray | None:
    """Kovaryans-ayarlı kesirli Kelly: f = fraction·Σ⁻¹μ, her varlık [-cap,cap]'e kırpılır.
    kelly_binary/kelly_continuous'un ÇOK-VARLIK genellemesi (Σ⁻¹μ, korelasyonu hesaba
    katan optimal-büyüme yönü). Simpleks/bütçe kısıtı BİLİNÇLİ olarak YOK — Kelly kaldıraç
    kabul eder, sum(w)=1 zorlaması ekonomik anlamını bozar (rapor tablosunda
    `budget_constrained=False` ile işaretlenir). Geçersiz/tekil kovaryans → None."""
    mu_arr = np.asarray(mu, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)
    n = mu_arr.shape[0]
    if cov_arr.shape != (n, n) or n < 1 or not np.all(np.isfinite(mu_arr)) or not np.all(np.isfinite(cov_arr)):
        return None
    try:
        inv_cov = np.linalg.pinv(cov_arr)
    except np.linalg.LinAlgError:
        return None
    raw = inv_cov @ mu_arr
    if not np.all(np.isfinite(raw)):
        return None
    scaled = raw * float(fraction)
    c = abs(float(cap))
    return np.clip(scaled, -c, c)


# ---------------------------------------------------------------------------
# Stres testleri
# ---------------------------------------------------------------------------
def _historical_window_replay(dates: np.ndarray | None, returns: np.ndarray, weights: np.ndarray,
                              windows: dict[str, tuple[str, str]] | None = None) -> dict[str, Any]:
    """Panel gerçekten o tarihleri kapsıyorsa pencere-başı portföy getirisi; kapsamıyorsa
    o pencere için None (uydurma yok)."""
    if dates is None:
        return {}
    windows = windows or STRESS_WINDOWS
    out: dict[str, Any] = {}
    for label, (start, end) in windows.items():
        mask = (dates >= start) & (dates <= end)
        if not np.any(mask):
            out[label] = None
            continue
        port_ret = returns[mask] @ weights
        out[label] = {
            "total_return": float(np.prod(1.0 + port_ret) - 1.0),
            "worst_day": float(np.min(port_ret)),
            "n_days": int(np.sum(mask)),
        }
    return out


def _panel_dates(closes_panel: Any, symbols: list[str]) -> np.ndarray | None:
    """Panel bir pandas DataFrame + DatetimeIndex ise ISO-string tarih dizisi (returns'e
    hizalı: ilk satır diff ile düşer); aksi halde None (tarihsel replay atlanır)."""
    if not isinstance(closes_panel, pd.DataFrame) or not isinstance(closes_panel.index, pd.DatetimeIndex):
        return None
    cols = [s for s in symbols if s in closes_panel.columns]
    if not cols:
        return None
    sub = closes_panel[cols].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if len(sub) < 3:
        return None
    return sub.index[1:].strftime("%Y-%m-%d").to_numpy()


def _mc_stress(mu: np.ndarray, sigma_diag: np.ndarray, weights: np.ndarray, seed: int,
              horizon: int = 63, n_paths: int = 2000) -> dict[str, Any]:
    """MC stres: HER VARLIK bağımsız GBM (quant_math.mc_gbm_paths, spec §3.10 zorunlu
    kütüphane) ile simüle edilir — varlıklar-arası KORELASYON bu MC'de YAKALANMAZ (bilinen
    sınırlama, `_historical_window_replay` bunu telafi eder). Deterministik (seed)."""
    n = len(weights)
    terminal = np.zeros(n_paths)
    for i in range(n):
        paths = quant_math.mc_gbm_paths(1.0, float(mu[i]), float(sigma_diag[i]), horizon, n_paths, seed + i)
        if paths.size == 0 or paths.shape[1] < 2:
            continue
        asset_term = paths[:, -1] - 1.0
        terminal = terminal + float(weights[i]) * asset_term
    return {
        "p5": float(np.percentile(terminal, 5)),
        "p50": float(np.percentile(terminal, 50)),
        "p95": float(np.percentile(terminal, 95)),
        "prob_loss": float(np.mean(terminal < 0.0)),
        "horizon_bars": horizon,
        "n_paths": n_paths,
        "not": "varlıklar bağımsız simüle edildi (MC'de korelasyon yok; tarihsel replay yakalar).",
    }


def _attribution(symbols: list[str], weights: np.ndarray, mu: np.ndarray,
                 candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Sembol/strateji bazlı BEKLENEN-GETİRİ katkısı (w_i·mu_i). Strateji bilgisi yoksa
    '?' altında toplanır."""
    contrib = {sym: float(weights[i] * mu[i]) for i, sym in enumerate(symbols)}
    sym_to_strategy = {c.get("symbol"): str(c.get("strategy", "?"))
                       for c in (candidates or []) if isinstance(c, dict)}
    by_strategy: dict[str, float] = {}
    for sym, val in contrib.items():
        strat = sym_to_strategy.get(sym, "?")
        by_strategy[strat] = by_strategy.get(strat, 0.0) + val
    return {"by_symbol": contrib, "by_strategy": by_strategy}


def _turnover_cost(symbols: list[str], w: np.ndarray, prev_weights: dict[str, float] | None,
                   bps: float) -> float:
    """Ciro cezası tahmini: sum(|w_yeni - w_önceki|)·bps·1e-4. `prev_weights` yok/bps<=0
    → 0.0 (ceza yok — geriye-uyum/opsiyonel). BPS'e göre DOĞRUSAL ölçeklenir (monoton)."""
    if not prev_weights or bps <= 0.0:
        return 0.0
    prev = np.array([float(prev_weights.get(s, 0.0)) for s in symbols])
    turnover = float(np.sum(np.abs(w - prev)))
    return turnover * bps * 1e-4


# ---------------------------------------------------------------------------
# Panel normalizasyonu + özet
# ---------------------------------------------------------------------------
def _normalize_panel(closes_panel: Any, symbols_req: list[str]) -> tuple[list[str], np.ndarray]:
    """closes_panel'i (pandas.DataFrame VEYA dict[str, ArrayLike]) ORTAK-tarihli,
    NaN'sız kapanış-fiyatı matrisine indirger. Uygun veri yoksa ([], boş dizi)."""
    if isinstance(closes_panel, pd.DataFrame):
        df = closes_panel
    elif isinstance(closes_panel, dict):
        try:
            df = pd.DataFrame(closes_panel)
        except (ValueError, TypeError):
            return [], np.zeros((0, 0))
    else:
        return [], np.zeros((0, 0))
    cols = [s for s in symbols_req if s in df.columns]
    if not cols:
        return [], np.zeros((0, 0))
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if len(sub) < 3 or (sub <= 0).any().any():
        return [], np.zeros((0, 0))
    return cols, sub.to_numpy(dtype=float)


def _summarize(symbols: list[str], w: np.ndarray, mu: np.ndarray, cov: np.ndarray,
              ppy: float = _DEFAULT_PPY) -> dict[str, Any]:
    port_ret = float(w @ mu)
    port_var = float(w @ cov @ w)
    port_vol = math.sqrt(max(port_var, 0.0))
    sharpe = (port_ret / port_vol) if port_vol > 0 else None
    return {
        "weights": {s: float(wi) for s, wi in zip(symbols, w, strict=True)},
        "expected_return_per_bar": port_ret,
        "vol_per_bar": port_vol,
        "sharpe_per_bar": sharpe,
        # `propose` getirileri log fark; basit bileşik formül burada yanlış (Tur-78, R2-B2).
        "expected_return_annualized": quant_math.annualize_return(port_ret, ppy, log=True),
        "vol_annualized": quant_math.annualize_vol(port_vol, ppy),
        "budget_constrained": True,
    }


def _run_method(name: str, fn: Any, symbols: list[str], mu: np.ndarray, cov: np.ndarray) -> dict[str, Any] | None:
    try:
        w = fn()
    except Exception as e:  # noqa: BLE001 — bir yöntemin hatası TÜM raporu düşürmemeli
        logger.debug(f"[ai.portfolio] yöntem '{name}' hata: {e}")
        return None
    if w is None:
        return None
    return _summarize(symbols, np.asarray(w, dtype=float), mu, cov)


def compare_methods(symbols: list[str], returns: np.ndarray, cov: np.ndarray, mu: np.ndarray,
                    candidates: list[dict[str, Any]], bounds: tuple[np.ndarray, np.ndarray],
                    budget: float, alpha_cvar: float, kelly_fraction: float,
                    max_weight: float, herc_clusters: int | None = None,
                    market_weights: Any | None = None) -> dict[str, dict[str, Any]]:
    """Tüm yöntemleri koşup TEK sözlükte toplar (spec §3.10 "yöntem karşılaştırma
    tablosu" — tek yönteme körlük yok). Bir yöntem None dönerse (veri/çözücü yetersizliği)
    o anahtar tabloda BULUNMAZ (uydurma değer yok).

    `market_weights` (Tur-79): Black-Litterman'ın π = δΣw_mkt önselini kuran ağırlıklar.
    **None = eski davranış** (ters-vol proxy) → çıktı BİT-DENK. Geçersiz/uyumsuz bir
    vektör sessizce proxy'ye DÜŞMEZ, `ValueError` verir: yanlış bir piyasa önselini
    "piyasa dengesi" diye raporlamak, proxy kullanmaktan daha kötüdür."""
    methods: dict[str, dict[str, Any]] = {}
    for name, fn in (
        ("min_var", lambda: min_variance_weights(cov, bounds, budget)),
        ("max_sharpe", lambda: max_sharpe_weights(mu, cov, bounds, budget)),
        ("hrp", lambda: hrp_weights(cov, bounds, budget)),
        ("erc", lambda: erc_weights(cov, bounds, budget)),
        ("herc", lambda: herc_weights(cov, herc_clusters, bounds, budget)),
        # Tur-73: NCO ve maks-çeşitlendirme aynı tabloya girer — "tek yönteme körlük yok"
        # kuralı yeni yöntemleri de kapsar; ayrı bir yüzey açmak tabloyu ikiye bölerdi.
        ("nco", lambda: nco_weights(cov, None, bounds, budget)),
        ("max_div", lambda: max_diversification_weights(cov, bounds, budget)),
    ):
        m = _run_method(name, fn, symbols, mu, cov)
        if m is not None:
            methods[name] = m

    if market_weights is None:
        mkt_w = _inverse_vol_weights(cov)
        mkt_kaynak = "ters_vol_proxy"
    else:
        mkt_w = np.asarray(market_weights, dtype=float)
        toplam = float(mkt_w.sum()) if mkt_w.size else 0.0
        if (mkt_w.shape != (cov.shape[0],) or not np.all(np.isfinite(mkt_w))
                or np.any(mkt_w < 0.0) or toplam <= 0.0):
            raise ValueError("market_weights: (n,) şeklinde, sonlu, negatif olmayan ve "
                             "toplamı pozitif olmalı")
        mkt_w = mkt_w / toplam          # sepet alt kümesi 1'e toplanmaz; π ÖLÇEĞE duyarlıdır
        mkt_kaynak = "gercek"
    bl_views = _build_bl_views(candidates, symbols)
    bl = black_litterman(cov, mkt_w, views=bl_views)
    if bl is not None:
        m = _run_method("black_litterman", lambda: max_sharpe_weights(bl["mu_bl"], bl["cov_bl"], bounds, budget),
                        symbols, mu, cov)
        if m is not None:
            m["n_views"] = bl["n_views"]
            m["market_weights_source"] = mkt_kaynak
            m["market_weights"] = {s: float(w) for s, w in zip(symbols, mkt_w, strict=False)}
            methods["black_litterman"] = m

    m_cvar = _run_method("cvar_lp", lambda: cvar_lp_weights(returns, alpha_cvar, bounds, budget),
                         symbols, mu, cov)
    if m_cvar is not None:
        methods["cvar_lp"] = m_cvar
    parity = cvar_risk_parity(returns, alpha_cvar, bounds, budget)
    if parity is not None:
        m_rp = _summarize(symbols, parity["weights"], mu, cov)
        m_rp["tail_risk"] = {"es": parity["es"], "alpha": parity["alpha"],
                             "shares": parity["shares"].tolist(), "max_error": parity["max_error"],
                             "n_tail": parity["n_tail"], "kink": parity["kink"]}
        methods["cvar_rp"] = m_rp

    # Tur-73: CDaR-LP. CVaR "en kötü günü", CDaR "en kötü ÇEKİLMEYİ" cezalandırır — ikisi
    # farklı ağırlık üretir ve tabloda YAN YANA durmaları gerekir (ölçüldü: CDaR-LP'nin
    # CDaR'ı CVaR-LP'ninkinden düşük, tersi de doğru).
    m_cdar = _run_method("cdar_lp", lambda: cdar_lp_weights(returns, alpha_cvar, bounds, budget),
                         symbols, mu, cov)
    if m_cdar is not None:
        methods["cdar_lp"] = m_cdar

    w_kelly = multi_asset_kelly(mu, cov, fraction=kelly_fraction, cap=max_weight)
    if w_kelly is not None:
        m_kelly = _summarize(symbols, w_kelly, mu, cov)
        m_kelly["budget_constrained"] = False
        methods["kelly_multi_asset"] = m_kelly
    return methods


def _append_jsonl(result: dict[str, Any]) -> None:
    """`PORTFOLIO_PROPOSALS_FILE`'a atomik gölge kayıt (episodic.propose_lesson_candidate
    deseni: düşük-frekanslı → oku+ekle+ATOMİK-tam-yeniden-yaz; OP-2 dersi)."""
    line = json.dumps(result, ensure_ascii=False, default=str)
    try:
        with _LOCK:
            PORTFOLIO_PROPOSALS_FILE.parent.mkdir(parents=True, exist_ok=True)
            prev = (PORTFOLIO_PROPOSALS_FILE.read_text(encoding="utf-8")
                    if PORTFOLIO_PROPOSALS_FILE.exists() else "")
            atomic_write_text(PORTFOLIO_PROPOSALS_FILE, prev + line + "\n")
    except OSError as e:
        logger.warning(f"[ai.portfolio] {PORTFOLIO_PROPOSALS_FILE.name} yazılamadı: {e}")


# ---------------------------------------------------------------------------
# Ana API
# ---------------------------------------------------------------------------
def propose(candidates: list[dict[str, Any]], closes_panel: Any,
           constraints: dict[str, Any] | None = None, *, seed: int = 42) -> dict[str, Any]:
    """GÖLGE portföy önerisi üretir (spec §3.10). KARAR-DIŞI: execution/emir sistemine
    BAĞLANMAZ. `candidates`: [{'symbol':..., 'strategy':..., 'posterior':0..1 (ops),
    'conformal_interval':(lo,hi) (ops)}, ...]. `closes_panel`: pandas.DataFrame
    (kolon=sembol) VEYA dict[str, ArrayLike]. `constraints` (hepsi opsiyonel, belgeli
    varsayılanlarla):
      max_weight (0.30), allow_short (False), net_directional_cap (1.0),
      turnover_penalty_bps (0.0), prev_weights (dict), cvar_alpha (0.05),
      kelly_fraction (0.25), covariance_model ('ledoit_wolf' veya 'rmt').

    Dönüş: {'ts','gölge','seed','universe','methods','condition_number',
    'single_factor_condition_number','constraints_used','stress','attribution','notes','text'}.
    Her çağrı `PORTFOLIO_PROPOSALS_FILE`'a atomik gölge kaydedilir (anayasa madde 5)."""
    ts = _utc_now_iso()
    constraints = dict(constraints or {})
    max_weight = float(constraints.get("max_weight", _DEFAULT_MAX_WEIGHT))
    allow_short = bool(constraints.get("allow_short", False))
    net_cap = float(constraints.get("net_directional_cap", _DEFAULT_NET_DIRECTIONAL_CAP))
    turnover_bps = float(constraints.get("turnover_penalty_bps", _DEFAULT_TURNOVER_BPS))
    prev_weights = constraints.get("prev_weights") or {}
    alpha_cvar = float(constraints.get("cvar_alpha", _DEFAULT_CVAR_ALPHA))
    kelly_fraction = float(constraints.get("kelly_fraction", _DEFAULT_KELLY_FRACTION))
    covariance_model = str(constraints.get("covariance_model", "ledoit_wolf"))
    herc_clusters = constraints.get("herc_clusters") or None
    seed_i = int(seed)

    result: dict[str, Any] = {
        "ts": ts, "gölge": PORTFOLIO_BANNER, "seed": seed_i,
        "universe": [], "n_candidates": len(candidates or []),
        "methods": {}, "condition_number": None, "single_factor_condition_number": None,
        "constraints_used": {
            "max_weight": max_weight, "allow_short": allow_short,
            "net_directional_cap": net_cap, "turnover_penalty_bps": turnover_bps,
            "cvar_alpha": alpha_cvar, "kelly_fraction": kelly_fraction,
            "covariance_model": covariance_model,
            "herc_clusters": herc_clusters,
        },
        "stress": {}, "attribution": {}, "notes": [], "text": "",
    }

    if not candidates:
        result["notes"].append("candidates boş — öneri üretilemedi.")
        result["text"] = "Portföy önerisi üretilemedi: aday listesi boş.\n" + CLOSING
        _append_jsonl(result)
        return result

    symbols_req: list[str] = []
    for c in candidates:
        sym = c.get("symbol") if isinstance(c, dict) else None
        if sym and str(sym) not in symbols_req:
            symbols_req.append(str(sym))
    if not symbols_req:
        result["notes"].append("candidates içinde geçerli 'symbol' alanı yok.")
        result["text"] = "Portföy önerisi üretilemedi: aday sembolleri okunamadı.\n" + CLOSING
        _append_jsonl(result)
        return result

    symbols, prices = _normalize_panel(closes_panel, symbols_req)
    if len(symbols) < 2:
        result["notes"].append(f"panelde yeterli örtüşen-veri sembol sayısı yok (bulunan={len(symbols)}, en az 2 gerekir).")
        result["text"] = "Portföy önerisi üretilemedi: fiyat panelinde yeterli örtüşen veri yok.\n" + CLOSING
        _append_jsonl(result)
        return result
    result["universe"] = symbols

    dropped = [s for s in symbols_req if s not in symbols]
    if dropped:
        result["notes"].append(f"panelde veri bulunamayan {len(dropped)} sembol düşürüldü: {dropped}")

    log_prices = np.log(prices)
    returns = np.diff(log_prices, axis=0)
    n = len(symbols)

    if covariance_model not in ("ledoit_wolf", "rmt"):
        result["notes"].append("Bilinmeyen kovaryans modeli; ölçüm üretilmedi.")
        result["text"] = result["notes"][-1] + "\n" + CLOSING
        _append_jsonl(result)
        return result
    rmt = quant_math.denoise_corr_mp(returns) if covariance_model == "rmt" else None
    cov = rmt["cov_denoised"] if rmt is not None else None
    cov_source = covariance_model
    if covariance_model == "rmt" and cov is None:
        result["notes"].append("RMT ölçülemedi: T>N, en az 10 getiri ve pozitif sütun varyansları gerekir.")
        result["text"] = result["notes"][-1] + "\n" + CLOSING
        _append_jsonl(result)
        return result
    if covariance_model == "ledoit_wolf":
        cov = quant_math.ledoit_wolf_cov(returns)
    if cov is None:
        cov = quant_math.ewma_cov(returns)
        cov_source = "ewma"
    if cov is None:
        var = np.var(returns, axis=0, ddof=1) if returns.shape[0] >= 2 else np.ones(n)
        var = np.where(np.isfinite(var) & (var > 0), var, 1e-6)
        cov = np.diag(var)
        cov_source = "diagonal_fallback"
    cov = np.asarray(cov, dtype=float)
    result["condition_number"] = _condition_number(cov)
    result["cov_source"] = cov_source
    result["covariance"] = cov.tolist()

    single_factor = _single_factor_cov(returns)
    if single_factor is not None:
        result["single_factor_condition_number"] = _condition_number(single_factor)

    mu = returns.mean(axis=0)

    lo = np.full(n, -max_weight if allow_short else 0.0)
    hi = np.full(n, max_weight)
    if n * max_weight < 1e-9:
        result["notes"].append("max_weight*n≈0 — hiçbir bütçe tahsis edilemez (kısıt çok sıkı).")
    # Bütçe (hedef sum(w)): uzun-yalnız iken [0,1]'e kırpılmış net_cap (varsayılan 1.0 =
    # tam yatırım); kısaya izin varken hedef net-uzun 1.0'dır ama [-net_cap,net_cap]'e
    # kırpılır (net-directional cap burada devreye girer). Kutu kısıtı (lo/hi) HER
    # durumda `_project_capped_simplex` içinde önceliklidir (bkz. fonksiyon docstring'i).
    budget = _clip(net_cap, 0.0, 1.0) if not allow_short else _clip(1.0, -net_cap, net_cap)
    bounds = (lo, hi)

    # Tur-79 (§4.1): gerçek piyasa-kapitalizasyonu ağırlıkları ÇAĞIRANDAN gelir
    # (bu modül veri katmanını import etmez). Geçersizse proxy'ye düşülür ama SESSİZCE
    # değil — not yazılır; "piyasa dengesi" diye yanlış bir önsel raporlanamaz.
    mkt_w_in = (constraints or {}).get("market_weights")
    if mkt_w_in is not None and not isinstance(mkt_w_in, (list, tuple, np.ndarray)):
        mkt_w_in = [mkt_w_in.get(s) for s in symbols] if hasattr(mkt_w_in, "get") else None
    if mkt_w_in is not None and any(w is None for w in np.atleast_1d(mkt_w_in).tolist()):
        result["notes"].append("market_weights: bazı semboller sepette YOK → ters-vol proxy.")
        mkt_w_in = None
    try:
        methods = compare_methods(symbols, returns, cov, mu, candidates, bounds, budget,
                                  alpha_cvar, kelly_fraction, max_weight, herc_clusters,
                                  market_weights=mkt_w_in)
    except ValueError as e:
        result["notes"].append(f"market_weights reddedildi ({e}) → ters-vol proxy.")
        methods = compare_methods(symbols, returns, cov, mu, candidates, bounds, budget,
                                  alpha_cvar, kelly_fraction, max_weight, herc_clusters)
    if not methods:
        result["notes"].append("hiçbir yöntem çözüm üretemedi (veri/kısıt yetersizliği).")
    for method in ("herc", "cvar_rp"):
        if method not in methods:
            result["notes"].append(f"{method}: veri/çözücü/kısıtlar altında eşit risk katkısı ölçülemedi.")
    for m in methods.values():
        m["turnover_cost_est"] = _turnover_cost(
            symbols, np.array([m["weights"][s] for s in symbols]), prev_weights, turnover_bps)
    result["methods"] = methods

    base_method = "hrp" if "hrp" in methods else (next(iter(methods), None))
    if base_method is not None:
        w_ref = np.array([methods[base_method]["weights"][s] for s in symbols])
        dates = _panel_dates(closes_panel, symbols)
        result["stress"] = {
            "reference_method": base_method,
            "tims_per_unit_equity": tims_stress([
                {"kind": "stock", "spot": 1., "quantity": float(w), "multiplier": 1.} for w in w_ref]),
            "historical": _historical_window_replay(dates, returns, w_ref),
            "monte_carlo": _mc_stress(mu, np.sqrt(np.clip(np.diag(cov), 1e-12, None)), w_ref, seed_i),
        }
        result["attribution"] = _attribution(symbols, w_ref, mu, candidates)

    lines = [PORTFOLIO_BANNER, f"Zaman: {ts}", f"Evren: {n} sembol ({', '.join(symbols)}).",
             f"Kovaryans kaynağı: {cov_source} (kondisyon sayısı={result['condition_number']:.4g})."]
    for name, m in methods.items():
        sh = m.get("sharpe_per_bar")
        sh_txt = f"{sh:.3f}" if sh is not None else "n/a"
        lines.append(f"  [{name}] beklenen-getiri(bar)={m['expected_return_per_bar']:+.5f} "
                    f"vol(bar)={m['vol_per_bar']:.5f} Sharpe(bar)={sh_txt} "
                    f"ciro-cezası≈{m['turnover_cost_est']:.6f}")
    if result["notes"]:
        lines.append("Notlar: " + " | ".join(result["notes"]))
    lines.append(CLOSING)
    result["text"] = "\n".join(lines)

    _append_jsonl(result)
    return result


# ---------------------------------------------------------------------------
# Tur-73: maks-çeşitlendirme · NCO · vol hedefleme · CDaR · CPPI · Bayes Kelly · L-VaR
# (Madde 117 · 116 · 118 · 281 · 283 · 119 · 120). Hepsi KARAR-DIŞI; `propose` raporu.
# ---------------------------------------------------------------------------
def max_diversification_weights(cov: Any, bounds: tuple[np.ndarray, np.ndarray] | None = None,
                                budget: float = 1.0) -> np.ndarray | None:
    """Maksimum çeşitlendirme portföyü (Choueifaty & Coignard 2008): Çeşitlendirme Oranını

        DR(w) = (Σ_i w_i·σ_i) / √(wᵀΣw)

    MAKSİMİZE eder. Pay = varlıklar bağımsız sayılsaydı çıkacak toplam risk, payda = gerçek
    portföy riski; oran, korelasyonun ne kadar risk SÖNDÜRDÜĞÜNÜ ölçer. DR = 1 yalnız tüm
    çiftler ρ=1 iken (çeşitlendirme yok) olur; aksi hâlde DR > 1'dir.

    min-varyanstan FARKI: min-varyans düşük-vol varlığa yığılabilir; MDP yüksek-vol ama
    DÜŞÜK-KORELASYONLU varlığı da alır. SLSQP ile −DR minimize edilir (oran ölçek-değişmez
    olduğundan bütçe kısıtı çözümü daraltmaz, yine de projeksiyonla uygulanır).
    Geçersiz/NaN kovaryans, σ_i <= 0 → None."""
    cov_arr = np.asarray(cov, dtype=float)
    n = cov_arr.shape[0] if cov_arr.ndim == 2 else 0
    if n < 1 or cov_arr.shape != (n, n) or not np.all(np.isfinite(cov_arr)):
        return None
    std = np.sqrt(np.clip(np.diag(cov_arr), 0.0, None))
    if np.any(std <= 0.0):
        return None
    lo, hi = _resolve_bounds(bounds, n)

    def neg_dr(w: np.ndarray) -> float:
        var = float(w @ cov_arr @ w)
        if var <= 1e-18:
            return 1e10
        return -float(w @ std) / math.sqrt(var)

    x0 = np.full(n, budget / n)
    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - budget)}]
    b = list(zip(lo.tolist(), hi.tolist(), strict=True))
    res = minimize(neg_dr, x0, method="SLSQP", bounds=b, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-14})
    w_out = np.asarray(res.x, dtype=float) if res.success else x0
    return _project_capped_simplex(w_out, lo, hi, budget)


def diversification_ratio(weights: Any, cov: Any) -> float | None:
    """Çeşitlendirme Oranı DR = (Σ |w_i| σ_i)/√(wᵀΣw). Geçersiz girdi / sıfır varyans → None."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.ndim != 2 or cov_arr.shape != (w.size, w.size):
        return None
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(cov_arr))):
        return None
    var = float(w @ cov_arr @ w)
    if var <= 0.0:
        return None
    return float(np.abs(w) @ np.sqrt(np.clip(np.diag(cov_arr), 0.0, None)) / math.sqrt(var))


def _corr_clusters(cov: np.ndarray, n_clusters: int | None) -> np.ndarray:
    """Korelasyon-mesafesi üzerinde küme etiketleri (1..k). `_quasi_diag_order` ile AYNI
    bağlantı/mesafe tanımını kullanır — HRP ile NCO aynı ağacı görsün diye."""
    n = cov.shape[0]
    if n <= 2:
        return np.ones(n, dtype=int)
    link = _corr_linkage(cov)
    k = int(n_clusters) if n_clusters else max(1, min(n - 1, int(round(math.sqrt(n)))))
    k = max(1, min(k, n))
    return np.asarray(fcluster(link, t=k, criterion="maxclust"), dtype=int)


def nco_weights(cov: Any, n_clusters: int | None = None,
                bounds: tuple[np.ndarray, np.ndarray] | None = None,
                budget: float = 1.0) -> np.ndarray | None:
    """İç içe kümelenmiş optimizasyon — NCO (López de Prado 2019, "A Robust Estimator of the
    Efficient Frontier"). Markowitz'in gürültü büyütmesini KÜMELEME ile sınırlar:

      1. Korelasyon mesafesinde hiyerarşik kümeleme (HRP ile AYNI ağaç, k = round(sqrt(N))).
      2. Her kümede AYRI min-varyans (küçük ve iyi-koşullu alt problem).
      3. Küme portföylerinin İNDİRGENMİŞ kovaryansı (KᵀΣK) üzerinde yine min-varyans.
      4. Nihai ağırlık = küme-içi x küme-arası.

    HRP'den farkı: HRP küme-içinde TERS-VARYANS kullanır (çapraz terimleri yok sayar),
    NCO gerçek min-varyans çözer; ikisi de aynı ağacı kullandığından karşılaştırılabilir.
    Kısıt yalnız NİHAİ vektöre uygulanır (alt problemler kısıtsız çözülür; kutu kısıtı
    `_project_capped_simplex` ile MATEMATİKSEL OLARAK garanti edilir).
    Geçersiz/NaN kovaryans veya bir alt problemin çözülememesi → None."""
    cov_arr = np.asarray(cov, dtype=float)
    n = cov_arr.shape[0] if cov_arr.ndim == 2 else 0
    if n < 1 or cov_arr.shape != (n, n) or not np.all(np.isfinite(cov_arr)):
        return None
    lo, hi = _resolve_bounds(bounds, n)
    if n <= 2:
        w_flat = min_variance_weights(cov_arr, budget=1.0)
        if w_flat is None:
            return None
        return _project_capped_simplex(w_flat, lo, hi, budget)
    etiket = _corr_clusters(cov_arr, n_clusters)
    ic_agirlik = np.zeros(n)
    kume_vekt: list[np.ndarray] = []
    for cid in sorted({int(v) for v in etiket}):
        idx = np.flatnonzero(etiket == cid)
        w_ic = min_variance_weights(cov_arr[np.ix_(idx, idx)], budget=1.0)
        if w_ic is None:
            return None
        ic_agirlik[idx] = w_ic
        k_vec = np.zeros(n)
        k_vec[idx] = w_ic
        kume_vekt.append(k_vec)
    k_mat = np.column_stack(kume_vekt)              # n x k
    cov_indirgenmis = k_mat.T @ cov_arr @ k_mat     # k x k
    w_arasi = min_variance_weights(cov_indirgenmis, budget=1.0)
    if w_arasi is None:
        return None
    w_out = k_mat @ np.asarray(w_arasi, dtype=float)
    return _project_capped_simplex(w_out, lo, hi, budget)


def vol_target_scale(weights: Any, cov: Any, target_vol: float = 0.10,
                     ppy: float = _DEFAULT_PPY,
                     max_leverage: float = 1.0) -> dict[str, Any] | None:
    """Volatilite hedefleme: verili ağırlık vektörünü YILLIK hedef oynaklığa ölçekleyen
    çarpanı hesaplar. `cov` DÖNEM-BAŞI (bar) kovaryanstır; yıllıklandırma sqrt(ppy) iledir.

        sigma_port = sqrt(wᵀΣw)·sqrt(ppy) ;  olcek = min(target_vol/sigma_port, max_leverage)

    `max_leverage=1.0` (varsayılan) KALDIRAÇ YOK demektir: oynaklık hedefin altındayken
    pozisyon büyütülmez, yalnız hedefin üstündeyken KÜÇÜLTÜLÜR — asimetrik ve bilinçli
    (kaldıraç bu sinyal-only sistemde bir öneri bile değildir). `nakit_orani` = 1 − olcek.

    Dönüş: {'sigma_port','sigma_hedef','olcek','nakit_orani','w_olcekli','max_leverage'}.
    Geçersiz kovaryans / target_vol <= 0 / ppy <= 0 / sigma_port = 0 → None."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.ndim != 2 or cov_arr.shape != (w.size, w.size) or target_vol <= 0.0 or ppy <= 0.0:
        return None
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(cov_arr))):
        return None
    var = float(w @ cov_arr @ w)
    if var <= 0.0:
        return None
    sigma = math.sqrt(var) * math.sqrt(ppy)
    olcek = min(target_vol / sigma, float(max_leverage))
    return {"sigma_port": sigma, "sigma_hedef": float(target_vol), "olcek": float(olcek),
            "nakit_orani": float(max(0.0, 1.0 - olcek)), "w_olcekli": w * olcek,
            "max_leverage": float(max_leverage)}


def cdar_lp_weights(returns: Any, alpha: float = _DEFAULT_CVAR_ALPHA,
                    bounds: tuple[np.ndarray, np.ndarray] | None = None,
                    budget: float = 1.0) -> np.ndarray | None:
    """CDaR-minimize eden ağırlıklar (Chekhlov, Uryasev & Zabarankin 2005) — LP.

    CVaR "en kötü %alpha GÜNÜ" cezalandırır; CDaR "en kötü %alpha ÇEKİLMEYİ" (alpha =
    KUYRUK OLASILIĞI; `cdar()` ölçümüyle ve `cvar_historical` ile AYNI konvansiyon —
    BULGU-73-1). Çekilme YOLA
    bağlıdır (tepe hafızası taşır), bu yüzden LP'ye kümülatif getiri ve koşan-tepe
    değişkenleri girer:

        x_t = (Σ_{s<=t} r_s)ᵀw  (kümülatif),  M_t = koşan tepe,  D_t = M_t − x_t
        min  zeta + 1/(alpha·T)·Σ_t u_t
        s.t. u_t >= D_t − zeta,  u_t >= 0,  M_t >= M_{t−1},  M_t >= x_t,
             Σw = budget,  lo <= w <= hi

    M_t iki eşitsizlikle yalnız ALTTAN sınırlanır; amaç fonksiyonu D_t'yi (dolayısıyla
    M_t'yi) aşağı ittiğinden optimumda M_t = max(M_{t−1}, x_t) SIKI olur — koşan tepe
    böylece tamsayı değişkeni olmadan doğru kurulur.

    ARİTMETİK KÜMÜLATİF: x_t getirilerin TOPLAMIDIR (bileşik değil); bu, problemi w'de
    doğrusal tutar. Log-getirilerle beslendiğinde toplam zaten bileşiğin kendisidir.
    Geçersiz girdi (T<2, N<1, NaN/Inf, alpha (0,1) dışında) veya LP başarısız → None."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[0] < 2 or r.shape[1] < 1 or not np.all(np.isfinite(r)):
        return None
    if not (0.0 < alpha < 1.0):
        return None
    t_n, n_assets = r.shape
    lo, hi = _resolve_bounds(bounds, n_assets)
    r_cum = np.cumsum(r, axis=0)                       # T x N
    i_w, i_z = 0, n_assets                             # w(N), zeta
    i_u, i_m = n_assets + 1, n_assets + 1 + t_n        # u(T), M(T)
    nvar = n_assets + 1 + 2 * t_n
    c = np.zeros(nvar)
    c[i_z] = 1.0
    c[i_u:i_u + t_n] = 1.0 / (alpha * t_n)      # BULGU-73-1: `cdar()` ile AYNI konvansiyon
    satir: list[np.ndarray] = []
    # (1) u_t >= M_t − x_t − zeta   →   −u_t + M_t − (r_cum_t)ᵀw − zeta <= 0
    for t in range(t_n):
        a = np.zeros(nvar)
        a[i_u + t] = -1.0
        a[i_m + t] = 1.0
        a[i_w:i_w + n_assets] = -r_cum[t]
        a[i_z] = -1.0
        satir.append(a)
    # (2) M_t >= x_t   →   −M_t + (r_cum_t)ᵀw <= 0
    for t in range(t_n):
        a = np.zeros(nvar)
        a[i_m + t] = -1.0
        a[i_w:i_w + n_assets] = r_cum[t]
        satir.append(a)
    # (3) M_t >= M_{t−1}   →   M_{t−1} − M_t <= 0
    for t in range(1, t_n):
        a = np.zeros(nvar)
        a[i_m + t - 1] = 1.0
        a[i_m + t] = -1.0
        satir.append(a)
    a_ub = np.vstack(satir)
    b_ub = np.zeros(a_ub.shape[0])
    a_eq = np.zeros((1, nvar))
    a_eq[0, i_w:i_w + n_assets] = 1.0
    bnds = (list(zip(lo.tolist(), hi.tolist(), strict=True)) + [(None, None)]
            + [(0.0, None)] * t_n + [(None, None)] * t_n)
    try:
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=np.array([budget]),
                      bounds=bnds, method="highs")
    except ValueError as e:
        logger.debug(f"[ai.portfolio] cdar_lp linprog hata: {e}")
        return None
    if not res.success:
        return None
    return _project_capped_simplex(np.asarray(res.x[:n_assets], dtype=float), lo, hi, budget)


def cdar(returns: Any, weights: Any, alpha: float = _DEFAULT_CVAR_ALPHA) -> dict[str, Any] | None:
    """Verili ağırlıkların CDaR/DaR'ı (ÖLÇÜM; optimizasyon değil). `cdar_lp_weights` ile
    AYNI aritmetik-kümülatif tanımı kullanır → ikisi doğrudan karşılaştırılabilir.
    Dönüş: {'cdar','dar','max_dd','n'}; dar = çekilme dağılımının (1−alpha) kuantili,
    cdar = o kuantilin ÜSTÜNDEKİ çekilmelerin ortalaması. Geçersiz girdi → None."""
    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if r.ndim != 2 or r.shape[0] < 2 or r.shape[1] != w.size:
        return None
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(w))) or not (0.0 < alpha < 1.0):
        return None
    x = np.cumsum(r @ w)
    dd = np.maximum.accumulate(x) - x
    dar = float(np.quantile(dd, 1.0 - alpha))
    kuyruk = dd[dd >= dar]
    return {"cdar": float(kuyruk.mean()) if kuyruk.size else dar, "dar": dar,
            "max_dd": float(dd.max()), "n": int(dd.size)}


def cppi_allocation(equity: float, floor_pct: float = 0.90, multiplier: float = 3.0,
                    risky_cap: float = 1.0,
                    floor_value: float | None = None) -> dict[str, Any] | None:
    """Sabit Oranlı Portföy Sigortası — CPPI (Perold & Sharpe 1988).

        taban  = floor_value (verilmezse floor_pct·equity)
        tampon = max(0, equity − taban)
        riskli = min(multiplier·tampon, risky_cap·equity)

    `floor_value` MUTLAK taban içindir ve DOĞRU kullanımdır: taban her çağrıda güncel
    sermayeden türetilirse (floor_pct·equity) taban sermayeyle birlikte DÜŞER, sigorta
    ANLAMINI YİTİRİR ve riskli oran sabit min(m·(1−f), cap) çıkar — hiçbir şey korunmaz.
    Bu tuzak `taban_kayan` bayrağıyla AÇIKÇA raporlanır.

    "Nakde geçiş" (riskli=0) tampon tükendiğinde MEKANİK olarak gerçekleşir; bir tavsiye
    veya sinyal değil, `propose` raporundaki referans tahsis eğrisidir.
    equity <= 0 / multiplier <= 0 / floor_pct [0,1) dışında / risky_cap < 0 → None.
    Dönüştürülemeyen girdi (None, liste, metin) de istisna değil None'dır (Tur-78, R2-B3)."""
    try:
        equity, floor_pct, multiplier, risky_cap = (float(v) for v in (equity, floor_pct, multiplier, risky_cap))
        taban = floor_pct * equity if floor_value is None else float(floor_value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in (equity, floor_pct, multiplier, risky_cap)):
        return None
    if equity <= 0.0 or multiplier <= 0.0 or not (0.0 <= floor_pct < 1.0) or risky_cap < 0.0:
        return None
    kayan = floor_value is None
    if not math.isfinite(taban) or taban < 0.0:
        return None
    tampon = max(0.0, float(equity) - taban)
    riskli = min(multiplier * tampon, risky_cap * float(equity))
    return {"taban": taban, "tampon": tampon, "riskli_tutar": riskli,
            "riskli_oran": riskli / float(equity), "nakit_oran": 1.0 - riskli / float(equity),
            "carpan": float(multiplier), "taban_kayan": kayan,
            "tabanin_altinda": bool(float(equity) <= taban)}


def tipp_allocation(equity: float, high_water: float, initial_floor: float,
                    max_drawdown: float = .10, multiplier: float = 3.,
                    risky_cap: float = 1.) -> dict | None:
    """TIPP: F=max(F0,(1−α)·tepe). F0 sabittir; tepe güncel NLV'yi de içerir.

    Yalnız tahsis raporu: aralıklı yeniden dengeleme ve fiyat sıçramaları tabanı
    delebilir. CPPI/TIPP tabanı gerçekleşmiş zarar garantisi değildir.
    """
    try:
        equity, peak, initial_floor, alpha, multiplier, risky_cap = map(
            float, (equity, high_water, initial_floor, max_drawdown, multiplier, risky_cap))
        if (not all(math.isfinite(v) for v in (equity, peak, initial_floor, alpha, multiplier, risky_cap))
                or equity <= 0 or peak <= 0 or initial_floor < 0 or not 0 <= alpha < 1 or risky_cap > 1):
            return None
        peak = max(equity, peak)
        floor = max(initial_floor, (1 - alpha) * peak)
        out = cppi_allocation(equity, multiplier=multiplier, risky_cap=risky_cap, floor_value=floor)
        if out is None:
            return None
        return {**out, "high_water": peak, "initial_floor": initial_floor, "max_drawdown": alpha,
                "drawdown": 1 - equity / peak, "floor_shortfall": max(0., floor - equity)}
    except (TypeError, ValueError, OverflowError):
        return None


def _beta_ppf(q: float, a: float, b: float) -> float:
    """Beta(a,b) ters-CDF. scipy.stats.beta LAZY import (modül açılışına 1,3 sn ekleme
    dersi, Tur-26) — bu dosyada YALNIZ `bayesian_kelly` kullanıyor."""
    from scipy.stats import beta as _beta
    return float(_beta.ppf(q, a, b))


def bayesian_kelly(wins: int, losses: int, payoff_ratio: float = 1.0,
                   prior_a: float = 1.0, prior_b: float = 1.0,
                   quantile: float = 0.05, fraction: float = 1.0) -> dict[str, Any] | None:
    """Parametre belirsizliği altında Kelly — Beta(a+kazanç, b+kayıp) posterioru ile.

    ÖLÇÜLEN GERÇEK (bu fonksiyonun asıl bulgusu): ikili bahiste beklenen log-büyüme p'de
    DOĞRUSALDIR — E_p[ p·ln(1+f·b) + (1−p)·ln(1−f) ] = E[p]·ln(1+f·b) + (1−E[p])·ln(1−f).
    Dolayısıyla posterior üzerinden ORTALAMA ALMAK Kelly'yi HİÇ küçültmez: Bayes-optimal
    kesir, posterior ORTALAMASINDAKİ klasik Kelly'nin TA KENDİSİDİR (`f_bayes`). Belirsizlik
    indirimi ortalamadan DEĞİL, KUANTİLDEN gelmek zorundadır (`f_robust`, p'nin `quantile`
    alt kuantilinde) — "Bayesyen olmak tek başına ihtiyatlı olmak değildir".

    Dönüş: {'p_mean','p_lo','f_bayes','f_robust','f_naive','n','posterior_a','posterior_b',
    'kesinti'}; `kesinti` = f_bayes − f_robust (belirsizlik indirimi; n büyüdükçe 0'a iner).
    `fraction` her iki kesri de ölçekler (kesirli Kelly). Kelly'ler `quant_math.kelly_binary`
    ile hesaplanır (tek kaynak) ve negatif kenar 0.0'a kırpılır.
    Negatif sayım / payoff_ratio <= 0 / quantile (0,0.5) dışında / prior <= 0 → None."""
    if wins < 0 or losses < 0 or payoff_ratio <= 0.0 or prior_a <= 0.0 or prior_b <= 0.0:
        return None
    if not (0.0 < quantile < 0.5) or not math.isfinite(fraction):
        return None
    a = prior_a + float(wins)
    b = prior_b + float(losses)
    p_mean = a / (a + b)
    p_lo = _beta_ppf(quantile, a, b)
    f_bayes = max(0.0, quant_math.kelly_binary(p_mean, payoff_ratio) or 0.0)
    f_robust = max(0.0, quant_math.kelly_binary(p_lo, payoff_ratio) or 0.0)
    n = int(wins + losses)
    f_naive = max(0.0, quant_math.kelly_binary(wins / n, payoff_ratio) or 0.0) if n > 0 else 0.0
    return {"p_mean": float(p_mean), "p_lo": float(p_lo), "f_bayes": float(f_bayes * fraction),
            "f_robust": float(f_robust * fraction), "f_naive": float(f_naive * fraction),
            "n": n, "posterior_a": float(a), "posterior_b": float(b),
            "kesinti": float((f_bayes - f_robust) * fraction)}


def liquidity_adjusted_var(position_value: float, adv_value: float, daily_vol: float,
                           spread_bps: float = 5.0, alpha: float = 0.05,
                           participation: float = 0.20,
                           impact_coef: float = 0.10) -> dict[str, Any] | None:
    """Likiditeye göre düzeltilmiş VaR — L-VaR (Bangia-Diebold-Schuermann-Stroughair 1999
    likidite bileşeni + tasfiye ufku genişletmesi).

    Standart VaR pozisyonun ANINDA piyasa fiyatından kapandığını varsayar; gerçekte günlük
    hacmin ancak bir kısmına katılınabilir:

        ufuk_gun = position_value / (participation·adv_value)      (en az 1 gün)
        VaR_ufuk = position_value · z_alpha · daily_vol · sqrt(ufuk_gun)
        makas    = position_value · (spread_bps/10000)/2           (yarı makas, tek yön)
        etki     = position_value · impact_coef · daily_vol · sqrt(position_value/adv_value)
        L-VaR    = VaR_ufuk + makas + etki

    Karekök piyasa etkisi, Tur-59'da eta=0'da bit-denk olduğu ölçülen AYNI ailedendir.
    Likidite cezası `carpan` = L-VaR / VaR_1gun ile raporlanır: 1'e yakınsa pozisyon
    likidite açısından önemsizdir, büyükse "VaR sayısı riskin bir kısmını saklıyor" demektir.
    Bu bir limit/kapı DEĞİL, bir görünürlük ölçütüdür.
    Negatif/sıfır girdi, alpha (0,0.5) dışında, participation (0,1] dışında → None."""
    vals = (position_value, adv_value, daily_vol, spread_bps, impact_coef)
    if not all(math.isfinite(float(v)) for v in vals):
        return None
    if position_value <= 0.0 or adv_value <= 0.0 or daily_vol <= 0.0:
        return None
    if spread_bps < 0.0 or impact_coef < 0.0 or not (0.0 < alpha < 0.5):
        return None
    if not (0.0 < participation <= 1.0):
        return None
    z = abs(quant_math._norm_ppf(alpha))
    ufuk = max(1.0, float(position_value) / (participation * float(adv_value)))
    var_1g = float(position_value) * z * float(daily_vol)
    var_ufuk = var_1g * math.sqrt(ufuk)
    makas = float(position_value) * (float(spread_bps) / 10_000.0) / 2.0
    etki = (float(position_value) * float(impact_coef) * float(daily_vol)
            * math.sqrt(float(position_value) / float(adv_value)))
    l_var = var_ufuk + makas + etki
    return {"l_var": l_var, "var_1gun": var_1g, "var_ufuk": var_ufuk,
            "makas_maliyeti": makas, "etki_maliyeti": etki, "ufuk_gun": ufuk,
            "carpan": l_var / var_1g if var_1g > 0 else None, "z": float(z)}
