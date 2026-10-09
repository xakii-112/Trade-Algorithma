"""
data_health.py — VERİ SAĞLIĞI kapısı (Tur-16 WP-O, docs/AI_V3_DERIN_BEYIN_SPEC.md §3.12). KARAR-DIŞI.

Ne yapar: brain/eğitim girdisi bu kapıdan geçer. `health_check(ohlcv_df, symbol=None)` bir
OHLCV DataFrame'i (kolonlar: open/high/low/close/volume — küçük/büyük harf duyarsız,
eksik olanlar atlanır) 0..100 arası bir SAĞLIK SKORUNA indirger + bulunan sorunları
listeler + karantina bayrağı üretir. Skor eşik-altıysa (`_QUARANTINE_THRESHOLD`)
`quarantine=True` — brain'in bu sembol için otomatik ABSTAIN'e düşmesi İÇİN kullanılacak
sinyaldir (bağlama KÖPRÜSÜ Dalga-2 entegrasyonunda kurulur; bu modül YALNIZ ölçer).

Denetimler (hepsi deterministik, savunmacı — çökme yok):
  1) NaN/tip           : sayısala çevrilemeyen/NaN hücre sayımı.
  2) OHLC geçerliliği   : negatif/sıfır fiyat, high<low.
  3) eksik-bar          : index ardışık-fark medyanının `_GAP_RATIO_THRESH` katından
                          büyük boşluklar (DatetimeIndex → gerçek zaman farkı; aksi halde
                          AYNI yöntem sayısal-index adım farkına uygulanır = "bar-aralığı
                          sezgisi", spec'in istediği iki-yollu davranış TEK fonksiyonda).
  4) donmuş-fiyat        : `_FROZEN_MIN_RUN`+ ardışık BİREBİR aynı kapanış bloğu.
  5) sıçrama-outlier     : bipower-varyasyon eşiği (advanced_math.jump_detect_bipower LAZY;
                          yoksa kendi tüm-örneklem bipower tahmincisi) → z>eşik barlar.
                          Her sıçrama ORANI klasik split-kesirleriyle (½,⅓,¼,...) karşılaştırılır:
                          eşleşirse "split_candidate" (CEZASIZ — meşru kurumsal aksiyon),
                          eşleşmezse "jump_outlier" (CEZALI — veri hatası/gerçek şok şüphesi).
  6) hacim anomalisi     : |z|>eşik hacim barları (kolon yoksa atlanır).

Skor: 100'den başlar, her sorun kategorisi KENDİ ağırlığıyla VE etkilenen-bar-oranıyla
ORANTILI (doygun, `min(1.0, frac*k)`) düşer — tek bir aykırı bar tüm skoru götürmez, ama
yaygın bozukluk sert cezalanır. split_candidate CEZASIZ (bilgilendirme amaçlı raporlanır).

Dönüş: {'score','issues'[{'type','detail','count'}],'quarantine','n_bars','symbol'}.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# --- Eşikler/ağırlıklar (belgeli sabitler; env override YOK — GÖLGE ölçüm sabit kalmalı) ---
_FROZEN_MIN_RUN = 5
_VOLUME_Z_THRESH = 4.0
_JUMP_Z_THRESH = 4.0
_GAP_RATIO_THRESH = 3.0
_SPLIT_RATIOS: tuple[float, ...] = (0.5, 1.0 / 3.0, 0.25, 0.2, 0.1, 2.0, 3.0, 4.0, 5.0, 10.0)
_SPLIT_TOLERANCE = 0.03
_QUARANTINE_THRESHOLD = 50.0
_MIN_BARS = 5

_PENALTY: dict[str, float] = {
    "nan_or_type": 20.0,
    "ohlc_invalid": 15.0,
    "missing_bars": 15.0,
    "frozen_price": 10.0,
    "jump_outlier": 10.0,
    "volume_anomaly": 6.0,
}
# ≥3 FARKLI cezalı kategori aynı anda bozuksa veri sistemik olarak güvenilmezdir —
# kategori-başına tavanlar tek tek aşılamasa bile bileşik ceza uygulanır
# ("yaygın bozukluk sert cezalanır" ilkesinin çok-kategori ayağı).
_MULTI_ISSUE_MIN_TYPES = 3
_MULTI_ISSUE_EXTRA = 10.0  # eşik üstündeki her ek kategori başına


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    """Kolonu küçük/büyük-harf duyarsız bulur, sayısala çevirir (dönüştürülemeyen → NaN)."""
    lower = {str(c).lower(): c for c in df.columns}
    for nm in names:
        if nm in lower:
            return pd.to_numeric(df[lower[nm]], errors="coerce")
    return None


# ---------------------------------------------------------------------------
# 3) eksik-bar (takvim VEYA sayısal-index sezgisi — aynı medyan-tabanlı yöntem)
# ---------------------------------------------------------------------------
def _missing_bar_count(index: Any) -> tuple[int, str]:
    try:
        if isinstance(index, pd.DatetimeIndex) and len(index) >= 3:
            diffs = np.diff(index.values).astype("timedelta64[s]").astype(float)
        else:
            arr = np.asarray(index, dtype=float)
            if arr.size < 3:
                return 0, "yetersiz index (eksik-bar denetlenemedi)"
            diffs = np.diff(arr)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size < 2:
            return 0, "yetersiz aralık (eksik-bar denetlenemedi)"
        med = float(np.median(diffs))
        if med <= 0:
            return 0, "medyan-aralık sıfır (eksik-bar denetlenemedi)"
        n_gaps = int(np.sum(diffs > med * _GAP_RATIO_THRESH))
        return n_gaps, f"{n_gaps} bar-aralığı medyanın {_GAP_RATIO_THRESH:.0f}x üstünde (medyan={med:.4g})"
    except (TypeError, ValueError) as e:
        logger.debug(f"[ai.data_health] eksik-bar hesabı hata: {e}")
        return 0, "eksik-bar hesaplanamadı"


# ---------------------------------------------------------------------------
# 4) donmuş-fiyat
# ---------------------------------------------------------------------------
def _frozen_price_runs(close: np.ndarray) -> list[int]:
    """`_FROZEN_MIN_RUN`'ı AŞAN ardışık-eşit kapanış bloklarının uzunluk listesi."""
    runs: list[int] = []
    n = close.size
    if n < _FROZEN_MIN_RUN:
        return runs
    i = 0
    while i < n:
        j = i + 1
        while j < n and np.isfinite(close[j]) and np.isfinite(close[i]) and close[j] == close[i]:
            j += 1
        run_len = j - i
        if run_len >= _FROZEN_MIN_RUN:
            runs.append(run_len)
        i = j
    return runs


# ---------------------------------------------------------------------------
# 5) sıçrama-outlier (bipower) + split-faktör ayrımı
# ---------------------------------------------------------------------------
def _lazy_jump_detect() -> Any:
    """`advanced_math.jump_detect_bipower`'ı fonksiyon-içinde dener (Dalga-2 paralel
    teslim — henüz mevcut olmayabilir). Modül/fonksiyon yok → None (kendi yedeğine düş)."""
    try:
        from nihai.ai import advanced_math
    except ImportError:
        return None
    return getattr(advanced_math, "jump_detect_bipower", None)


def _own_bipower_jump_flags(logret: np.ndarray) -> np.ndarray:
    """Kendi bipower-varyasyon tahmincisi (Barndorff-Nielsen & Shephard): BP = (π/2)⁻¹ ·
    mean(|r_t|·|r_{t-1}|); σ = sqrt(BP) TÜM-ÖRNEKLEM ölçeği (basitleştirme — araştırma-
    düzeyi bir yerel-pencere tahmincisi DEĞİL, sağlık-kapısı için yeterli sezgisel eşik).
    |r_t|/σ > `_JUMP_Z_THRESH` → sıçrama adayı."""
    n = logret.size
    if n < 10:
        return np.zeros(n, dtype=bool)
    abs_r = np.abs(logret)
    bp_terms = abs_r[1:] * abs_r[:-1]
    mu1_sq = 2.0 / math.pi
    bp_var = float(np.mean(bp_terms)) / mu1_sq if bp_terms.size else float("nan")
    sigma = math.sqrt(bp_var) if math.isfinite(bp_var) and bp_var > 0 else float(np.std(logret))
    if not math.isfinite(sigma) or sigma <= 0:
        return np.zeros(n, dtype=bool)
    return np.abs(logret) / sigma > _JUMP_Z_THRESH


def _bipower_jump_flags(logret: np.ndarray) -> np.ndarray:
    fn = _lazy_jump_detect()
    if fn is not None:
        try:
            raw = fn(logret)
            arr = np.asarray(raw)
            if arr.shape[0] == logret.shape[0]:
                return arr.astype(bool) if arr.dtype == bool else (np.asarray(arr, dtype=float) > 0.5)
        except Exception as e:  # noqa: BLE001 — Dalga-2 paralel teslim, gevşek bağımlılık
            logger.debug(f"[ai.data_health] advanced_math.jump_detect_bipower kullanılamadı: {e}")
    return _own_bipower_jump_flags(logret)


def _jump_and_split_analysis(close: np.ndarray) -> tuple[int, int, str]:
    """Dönüş: (jump_outlier sayısı, split_candidate sayısı, Türkçe özet)."""
    finite = np.isfinite(close) & (close > 0)
    if int(np.sum(finite)) < 12:
        return 0, 0, "yetersiz veri (sıçrama denetlenemedi)"
    c = np.where(finite, close, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.diff(np.log(c))
    valid = np.isfinite(logret)
    if int(np.sum(valid)) < 10:
        return 0, 0, "yetersiz veri (sıçrama denetlenemedi)"
    logret_clean = np.where(valid, logret, 0.0)
    flags = _bipower_jump_flags(logret_clean) & valid
    n_jump = n_split = 0
    for i in np.where(flags)[0]:
        c_i, c_i1 = c[i], c[i + 1]
        if not (np.isfinite(c_i) and np.isfinite(c_i1) and c_i > 0):
            continue
        ratio = float(c_i1 / c_i)
        if any(abs(ratio - r) / r < _SPLIT_TOLERANCE for r in _SPLIT_RATIOS):
            n_split += 1
        else:
            n_jump += 1
    detail = (f"{n_jump} açıklanamayan sıçrama + {n_split} olası split/kurumsal-aksiyon "
             f"(bipower z>{_JUMP_Z_THRESH:.0f})")
    return n_jump, n_split, detail


# ---------------------------------------------------------------------------
# 6) hacim anomalisi
# ---------------------------------------------------------------------------
def _volume_anomaly_count(volume: np.ndarray) -> int:
    v_all = volume
    finite = np.isfinite(v_all)
    v = v_all[finite]
    if v.size < 10:
        return 0
    mu = float(np.mean(v))
    sd = float(np.std(v, ddof=1))
    if sd <= 0:
        return 0
    z = np.abs((v_all - mu) / sd)
    return int(np.sum(finite & (z > _VOLUME_Z_THRESH)))


# ---------------------------------------------------------------------------
# Ana API
# ---------------------------------------------------------------------------
def health_check(ohlcv_df: Any, symbol: str | None = None) -> dict[str, Any]:
    """OHLCV verisini sağlık skoruna indirger (modül docstring'i — 6 denetim). Savunmacı:
    DataFrame değil/boş/close-kolonu-yok/n<`_MIN_BARS` → score=0.0, quarantine=True,
    açıklayıcı tek `issue`. Hiçbir girdi İSTİSNA fırlatmaz."""
    if not isinstance(ohlcv_df, pd.DataFrame) or ohlcv_df.empty:
        return {"score": 0.0, "issues": [{"type": "veri_yok", "detail": "DataFrame boş/geçersiz", "count": 0}],
               "quarantine": True, "n_bars": 0, "symbol": symbol}
    n = len(ohlcv_df)
    close = _col(ohlcv_df, "close", "kapanis", "adj_close", "adjclose")
    if close is None:
        return {"score": 0.0, "issues": [{"type": "close_yok", "detail": "'close' kolonu bulunamadı", "count": 0}],
               "quarantine": True, "n_bars": n, "symbol": symbol}

    open_ = _col(ohlcv_df, "open", "acilis")
    high = _col(ohlcv_df, "high", "yuksek")
    low = _col(ohlcv_df, "low", "dusuk")
    volume = _col(ohlcv_df, "volume", "vol", "hacim")

    issues: list[dict[str, Any]] = []
    score = 100.0

    # 1) NaN/tip
    present_cols = [c for c in (open_, high, low, close) if c is not None]
    nan_count = int(sum(int(c.isna().sum()) for c in present_cols))
    if nan_count > 0:
        frac = nan_count / max(1, n * len(present_cols))
        issues.append({"type": "nan_or_type", "detail": f"{nan_count} NaN/dönüştürülemez değer", "count": nan_count})
        score -= _PENALTY["nan_or_type"] * min(1.0, frac * 10.0)

    # 2) OHLC geçerliliği
    ohlc_bad = 0
    if high is not None and low is not None:
        both = high.notna() & low.notna()
        ohlc_bad += int((high[both] < low[both]).sum())
    for col in present_cols:
        ohlc_bad += int((col.dropna() <= 0).sum())
    if ohlc_bad > 0:
        frac = ohlc_bad / max(1, n)
        issues.append({"type": "ohlc_invalid", "detail": f"{ohlc_bad} negatif/sıfır fiyat veya high<low", "count": ohlc_bad})
        score -= _PENALTY["ohlc_invalid"] * min(1.0, frac * 10.0)

    # 3) eksik-bar
    gap_count, gap_detail = _missing_bar_count(ohlcv_df.index)
    if gap_count > 0:
        issues.append({"type": "missing_bars", "detail": gap_detail, "count": gap_count})
        frac = gap_count / max(1, n)
        score -= _PENALTY["missing_bars"] * min(1.0, frac * 5.0)

    close_arr = close.to_numpy(dtype=float)

    # 4) donmuş-fiyat
    frozen_runs = _frozen_price_runs(close_arr)
    if frozen_runs:
        total_frozen = sum(frozen_runs)
        issues.append({"type": "frozen_price",
                      "detail": f"{len(frozen_runs)} donmuş-fiyat bloğu ({total_frozen} bar, eşik={_FROZEN_MIN_RUN})",
                      "count": len(frozen_runs)})
        frac = total_frozen / max(1, n)
        score -= _PENALTY["frozen_price"] * min(1.0, frac * 5.0)

    # 5) sıçrama / split ayrımı
    n_jump, n_split, jump_detail = _jump_and_split_analysis(close_arr)
    if n_jump > 0:
        issues.append({"type": "jump_outlier", "detail": jump_detail, "count": n_jump})
        frac = n_jump / max(1, n)
        score -= _PENALTY["jump_outlier"] * min(1.0, frac * 8.0)
    if n_split > 0:
        issues.append({"type": "split_candidate",
                      "detail": f"{n_split} olası split/kurumsal-aksiyon (fiyat hatası SAYILMADI, ceza yok)",
                      "count": n_split})

    # 6) hacim anomalisi
    if volume is not None:
        vol_anom = _volume_anomaly_count(volume.to_numpy(dtype=float))
        if vol_anom > 0:
            issues.append({"type": "volume_anomaly",
                          "detail": f"{vol_anom} hacim aykırı-değeri (|z|>{_VOLUME_Z_THRESH:.0f})",
                          "count": vol_anom})
            frac = vol_anom / max(1, n)
            score -= _PENALTY["volume_anomaly"] * min(1.0, frac * 5.0)

    # 7) çok-kategorili bileşik ceza (yalnız cezalı tipler; split_candidate bilgilendirmedir)
    penalized_types = {iss["type"] for iss in issues if iss["type"] in _PENALTY}
    if len(penalized_types) >= _MULTI_ISSUE_MIN_TYPES:
        score -= _MULTI_ISSUE_EXTRA * (len(penalized_types) - (_MULTI_ISSUE_MIN_TYPES - 1))

    if n < _MIN_BARS:
        issues.insert(0, {"type": "yetersiz_veri", "detail": f"n={n} < {_MIN_BARS} (minimum bar sayısı)", "count": n})
        score = min(score, 0.0)

    score = float(max(0.0, min(100.0, score)))
    quarantine = bool(score < _QUARANTINE_THRESHOLD or n < _MIN_BARS)
    return {"score": score, "issues": issues, "quarantine": quarantine, "n_bars": n, "symbol": symbol}
