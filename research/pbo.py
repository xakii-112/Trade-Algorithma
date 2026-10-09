"""
pbo.py — Backtest Overfitting Olasılığı (PBO) via CSCV (combinatorially symmetric CV).

Bu modül `research/deflated.py`'nin KARDEŞİDİR ve onun tamamlayıcısıdır: ikisi de YALNIZ
ölçüm/raporlama yapar, hiçbir verdikt/eşik/karar mantığı içermez (CLAUDE.md "kenar UYDURMADAN
dürüst ölçüm" disiplini). Aralarındaki işbölümü:

  - DSR (deflated.py) "KAÇ DENEME yaptın?" sorusunu yanıtlar: N konfig denenince en iyisi
    şansla ne kadar iyi görünür, ölçtüğün Sharpe bunu aşıyor mu?
  - PBO (bu modül) "KONFİG SEÇİMİN OOS'TA ÇÖKÜYOR MU?" sorusunu yanıtlar: örneklem-içinde (IS)
    en iyi görünen konfigürasyonu seçtiğinde, örneklem-dışında (OOS) medyanın altına düşme
    olasılığı nedir? PBO > 0.5 → seçim süreci gürültüye uyuyor (aşırı-uyum).

Bu yüzden her aday-sinyal raporuna İKİSİ BİRLİKTE girmeli: DSR seçilim-yanlılığını Sharpe'a
düşer, PBO seçim sürecinin OOS-dayanıklılığını ölçer. Farklı başarısızlık kiplerini yakalarlar.

Yöntem — CSCV (Bailey, Borwein, López de Prado & Zhu, 2016, "The Probability of Backtest
Overfitting", Journal of Computational Finance, 20(4), 39-69):
  1. T×N getiri matrisini (T dönem, N strateji-konfigürasyonu) T ekseni boyunca s eşit-uzunluklu
     ARDIŞIK bloğa böl.
  2. s/2'li tüm blok kombinasyonlarını IS (in-sample), tümleyeni OOS (out-of-sample) yap →
     C(s, s/2) simetrik bölünme (bu yüzden "combinatorially symmetric": her IS bölünmesinin
     tümleyeni de ayrı bir kombinasyonda IS olur).
  3. Her kombinasyonda: IS'te metrik-en-iyi konfigürasyonu (n*) seç; n*'in OOS metriğinin diğer
     konfiglere göre GÖRELİ RANK'ından ω = rank/(N+1) ∈ (0,1) ve logit λ = ln(ω/(1−ω)) hesapla.
  4. PBO = λ ≤ 0 olan kombinasyonların payı (= IS-optimal konfigin OOS medyanının ALTINA düşme
     olasılığı). Ek olarak performans-bozunumu (OOS~IS doğrusal fiti) ve OOS-zarar olasılığı.

Bağımlılık disiplini: SAF numpy + stdlib (itertools/math). scipy dahi GEREKMEZ — sıralama ve
logit ölçek-bağımsız olduğundan sharpe yıllıklanmaz. mlfinlab KULLANILMADI: ticari lisanslı
(Hudson & Thames); CSCV algoritması makaleden doğrudan, birkaç düzine satırda uygulanabilir.

Sharpe konvansiyonu deflated.py ile hizalı: ddof=0 (mean/pstdev), yıllıksız — CSCV YALNIZ
sıralama kullandığından ölçek/annualizasyon PBO'yu değiştirmez (docstring'de tekrar belirtilir).
"""
from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

# C(s, s/2) üst sınırı: s=16 → 12870. Bunun ötesi (s≥18) hesap patlaması → ValueError.
_MAX_COMBINATIONS = 12870
# ω'nın logit öncesi savunmacı kırpması. Ortalama-rank ω'yı zaten (1/(N+1), N/(N+1)) içinde
# tuttuğundan (aşağıya bak) pratikte ASLA tetiklenmez; yalnız ±inf'e karşı sigortadır.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Girdi doğrulama — hepsi Türkçe mesajlı ValueError (sessiz uydurma yok)
# ---------------------------------------------------------------------------
def _validate(returns_matrix: object, s: object,
              metric: str | Callable[[np.ndarray], float]) -> tuple[np.ndarray, int]:
    """returns_matrix'i T×N float ndarray'e çevirir ve tüm ön-koşulları denetler.

    Kurallar: s çift ve ≥4; C(s,s/2) ≤ 12870; T ≥ 2·s (blok başına ≥2 gözlem); N ≥ 5 (rank
    anlamı); getiriler sonlu (NaN/inf yok). İhlal → açıklayıcı Türkçe ValueError."""
    if not isinstance(s, (int, np.integer)) or isinstance(s, bool):
        raise ValueError(f"s tam sayı olmalı (verilen tip: {type(s).__name__}).")
    s = int(s)
    if s < 4:
        raise ValueError(f"s ≥ 4 olmalı (verilen s={s}); daha az blokla CSCV anlamsız.")
    if s % 2 != 0:
        raise ValueError(f"s ÇİFT olmalı (verilen s={s}); IS/OOS eşit blok bölünmesi gerekir.")
    n_comb = math.comb(s, s // 2)
    if n_comb > _MAX_COMBINATIONS:
        raise ValueError(
            f"s={s} → C({s},{s // 2})={n_comb} kombinasyon, üst sınır {_MAX_COMBINATIONS} (s=16) "
            f"aşıldı; kombinasyon sayısı patlar. Daha küçük çift s kullanın (s ≤ 16)."
        )
    if not (isinstance(metric, str) or callable(metric)):
        raise ValueError("metric 'sharpe'/'mean' ya da çağrılabilir (kolon→float) olmalı.")
    if isinstance(metric, str) and metric not in ("sharpe", "mean"):
        raise ValueError(f"Bilinmeyen metric='{metric}'; 'sharpe', 'mean' veya callable olmalı.")
    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"returns_matrix 2 boyutlu (T×N) olmalı (verilen ndim={m.ndim}).")
    n_periods, n_configs = m.shape
    if n_configs < 5:
        raise ValueError(f"N ≥ 5 konfigürasyon gerekir (verilen N={n_configs}); N<5'te rank/PBO anlamsız.")
    if n_periods < 2 * s:
        raise ValueError(
            f"T ≥ 2·s gerekir (verilen T={n_periods}, s={s} → en az {2 * s} dönem); "
            f"blok başına en az 2 gözlem şart."
        )
    if not np.isfinite(m).all():
        raise ValueError("returns_matrix NaN/sonsuz değer içeriyor; CSCV'den önce temizlenmeli.")
    return m, s


# ---------------------------------------------------------------------------
# Kolon-başına metrik (IS ve OOS alt-matrisleri için) — sharpe/mean vektörize, callable döngü
# ---------------------------------------------------------------------------
def _column_metrics(sub: np.ndarray, metric: str | Callable[[np.ndarray], float]) -> np.ndarray:
    """(L×N) alt-matriste her konfig (kolon) için tek skorluk metrik → (N,) vektör.

    'sharpe': mean/std, ddof=0 (deflated.sharpe_moments konvansiyonu), YILLIKSIZ; std=0 kolon → 0
    (çökme yok). 'mean': kolon ortalaması. callable: her kolona ayrı uygulanır (kolon→float).

    callable dalı ÖNCE denetlenir (str değil → callable narrow'u; _validate metric'i zaten
    doğrular, son raise savunmacıdır/erişilmez)."""
    if callable(metric):
        return np.array([float(metric(sub[:, j])) for j in range(sub.shape[1])], dtype=float)
    if metric == "sharpe":
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0)   # ddof=0 (popülasyon std) — deflated.py ile hizalı
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = mu / sd
        sr[~np.isfinite(sr)] = 0.0   # std=0 (sabit getiri) → sharpe 0
        return sr
    if metric == "mean":
        return sub.mean(axis=0)
    raise ValueError(f"Bilinmeyen metric: {metric!r}")   # savunmacı; _validate erişimi engeller


def _relative_rank(oos_perf: np.ndarray, sel: float) -> float:
    """sel değerinin oos_perf içindeki ORTALAMA-RANK'ı (1-tabanlı, artan: en yüksek değer = N).

    Beraberlikleri simetrik ele alan ortalama-rank: rank = (kesin küçük sayısı) + (eşit+1)/2.
    sel her zaman oos_perf'in bir elemanı olduğundan eşit ≥ 1 ⇒ rank ∈ [1, N] (garanti).
    En büyük tekil değer → rank N, en küçük → rank 1, hepsi eşit → rank (N+1)/2 (nötr)."""
    less = float(np.count_nonzero(oos_perf < sel))
    equal = float(np.count_nonzero(oos_perf == sel))
    return less + (equal + 1.0) / 2.0


def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """y ~ a·x + b en-küçük-kareler eğim/kesim. Var[x]=0 (tüm IS-en-iyiler özdeş) → eğim
    tanımsız; 0.0 eğim + ȳ kesim döner (+ logger.warning; sessiz uydurma yok)."""
    xbar = float(x.mean())
    ybar = float(y.mean())
    dx = x - xbar
    denom = float(np.dot(dx, dx))
    if denom <= 0.0:
        logger.warning("pbo: degradation fit — Var[IS]=0, eğim tanımsız → slope=0, intercept=ȳ")
        return 0.0, ybar
    slope = float(np.dot(dx, y - ybar) / denom)
    return slope, float(ybar - slope * xbar)


# ---------------------------------------------------------------------------
# Ana API: CSCV → PBO
# ---------------------------------------------------------------------------
def cscv_pbo(returns_matrix: object, s: int = 16, metric: str | Callable[[np.ndarray], float] = "sharpe",
             seed: int | None = None) -> dict:
    """Backtest Overfitting Olasılığı (PBO) — Bailey ve ark. (2016) CSCV yöntemiyle.

    returns_matrix : T×N (T dönem getirisi, N strateji-konfigürasyonu). np.ndarray veya
        list-of-lists kabul eder. SATIRLAR ZAMAN-SIRALI (eski→yeni) varsayılır.
    s : blok sayısı (ÇİFT, 4 ≤ s ≤ 16). C(s, s/2) kombinasyon üretilir (s=16 → 12870).
    metric : "sharpe" (varsayılan; ddof=0, YILLIKSIZ — CSCV yalnız SIRALAMA kullandığından ölçek
        PBO'yu değiştirmez), "mean" veya çağrılabilir (kolon-ndarray → float).
    seed : CSCV çekirdeği DETERMİNİSTİKTİR; seed YALNIZ stokastik bir metric-callable'ı (np.random
        kullanan) tekrarlanabilir kılmak için global RNG'yi tohumlar. Yerleşik sharpe/mean için
        etkisizdir; seed verilmese de aynı girdi → bit-birebir aynı çıktı.

    Yöntem: T ekseni s ardışık bloğa bölünür; s/2'li her blok kombinasyonu IS, tümleyeni OOS.
    Her kombinasyonda IS-en-iyi konfig n* seçilir; n*'in OOS metriğinin göreli rank'ından
    ω = rank/(N+1), logit λ = ln(ω/(1−ω)). PBO = pay(λ ≤ 0).

    Artık satırlar (T'nin s'e tam bölünmeyen kısmı) BAŞTAN atılır: satırlar zaman-sıralı kabul
    edildiğinden en GÜNCEL (son) veri korunur, en eski fazlalık düşer. n_periods_used bunu belgeler.

    Sayısal kenarlar: OOS metrikleri hep-eşit/std=0 kolonlar → rank nötr (ω=0.5, λ=0, PBO'ya
    dahil). ω logit öncesi [ε, 1−ε]'e kırpılır (ortalama-rank ω'yı zaten (0,1) içinde tutar; kırpma
    salt sigortadır). NaN/inf girdi → ValueError.

    Dönüş (dict):
      pbo                  : λ ≤ 0 payı (aşırı-uyum olasılığı; 0=temiz, 0.5=şans, →1=çöküyor).
      n_combinations       : C(s, s/2) — logits uzunluğu.
      logits               : np.ndarray, kombinasyon-başına λ (dağılımı incelemek için).
      degradation_slope    : OOS-metrik ~ IS-metrik doğrusal fit eğimi (IS-en-iyi konfigin
      degradation_intercept:   (IS,OOS) çiftleri üzerinden). Eğim ≤ 0 → güçlü performans bozunumu.
      prob_oos_loss        : IS-en-iyi konfigin OOS metriği < 0 olan kombinasyon payı.
      s, n_configs, n_periods_used, metric : parametre/kullanım özeti.
    """
    m, s = _validate(returns_matrix, s, metric)
    if seed is not None:
        np.random.seed(seed)   # yalnız stokastik metric-callable için; CSCV çekirdeği deterministik

    n_periods, n_configs = m.shape
    block_len = n_periods // s
    n_used = block_len * s
    m = m[n_periods - n_used:]   # BAŞTAN kırp → en güncel (son) satırlar kalır (satırlar zaman-sıralı)

    # Blok satır-indeksleri (ardışık dilimler) ve tüm-blok kümesi.
    blocks = [np.arange(i * block_len, (i + 1) * block_len) for i in range(s)]
    all_blocks = set(range(s))

    logits: list[float] = []
    is_best_is: list[float] = []    # x: IS-en-iyinin IS metriği (= max IS metriği)
    is_best_oos: list[float] = []   # y: AYNI konfigin OOS metriği

    for combo in itertools.combinations(range(s), s // 2):
        is_rows = np.concatenate([blocks[i] for i in combo])
        oos_rows = np.concatenate([blocks[i] for i in sorted(all_blocks - set(combo))])
        is_perf = _column_metrics(m[is_rows], metric)
        oos_perf = _column_metrics(m[oos_rows], metric)

        n_star = int(np.argmax(is_perf))   # IS-en-iyi konfig (beraberlikte ilk = deterministik)
        sel = float(oos_perf[n_star])
        rank = _relative_rank(oos_perf, sel)
        omega = rank / (n_configs + 1.0)
        omega = min(max(omega, _EPS), 1.0 - _EPS)
        logits.append(math.log(omega / (1.0 - omega)))
        is_best_is.append(float(is_perf[n_star]))
        is_best_oos.append(sel)

    logits_arr = np.asarray(logits, dtype=float)
    x = np.asarray(is_best_is, dtype=float)
    y = np.asarray(is_best_oos, dtype=float)
    slope, intercept = _ols_slope_intercept(x, y)

    return {
        "pbo": float(np.mean(logits_arr <= 0.0)),
        "n_combinations": int(logits_arr.size),
        "logits": logits_arr,
        "degradation_slope": slope,
        "degradation_intercept": intercept,
        "prob_oos_loss": float(np.mean(y < 0.0)),
        "s": int(s),
        "n_configs": int(n_configs),
        "n_periods_used": int(n_used),
        "metric": metric if isinstance(metric, str) else "callable",
    }


def pbo_report(result: dict) -> str:
    """cscv_pbo çıktısından Türkçe kısa özet metni (loglara/rapora gömmek için)."""
    return (
        f"PBO={result['pbo']:.3f}  ({result['n_combinations']} kombinasyon, s={result['s']}, "
        f"N={result['n_configs']} konfig, kullanılan T={result['n_periods_used']}, "
        f"metrik={result['metric']})\n"
        f"  OOS-zarar olasılığı = {result['prob_oos_loss']:.3f}\n"
        f"  performans bozunumu (OOS~IS): eğim={result['degradation_slope']:.4f}, "
        f"kesim={result['degradation_intercept']:.4f}\n"
        f"  Yorum: PBO, IS-optimal konfigin OOS medyanının ALTINA düşme olasılığıdır; PBO>0.5 ise "
        f"seçim süreci gürültüye uyuyor (aşırı-uyum). DSR (deflated.py) ile BİRLİKTE okunmalı."
    )
