"""Model sürüklenme karnesi (Tur-79). KARAR-DIŞI; sinyal/model/kapı seçimine bağlanmaz.

PSI/CSI, iki örneklem testi, Wasserstein-1 ve normal ailesinde Fisher–Rao mesafesi.
NaN gözlemler atılır (PSI'de oranları taşınır); Inf, yanlış şekil/tip ve yetersiz
örneklem → None. Sabit seride tanımlı mesafeler korunur, tanımsız ölçekler uydurulmaz.
Kategori seçimi yalnız referansın benzersiz sayısına bağlıdır; sayısal kodlu
kategoriler desteklenir. Tarihsel raporda referans, izlenen dönemlerden önce bitmelidir.
CSV okuyucu dışında saf fonksiyonlar; ağ ve dosya yazımı yoktur.

Kaynaklar: Federal Reserve SR 11-7 (2011), Guidance on Model Risk Management;
Yurdakul (2018), Statistical Properties of Population Stability Index;
Atkinson & Mitchell (1981), Rao's distance measure; Villani (2009), Optimal Transport.
SR 11-7 model izleme çerçevesidir; PSI eşikleri düzenleyici zorunluluk değildir.
"""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats


def _array(values: ArrayLike) -> np.ndarray | None:
    """Şekli koruyan 1B dönüşüm; NaN kalır, Inf ve karmaşık sayılar reddedilir."""
    try:
        raw = np.asarray(values)
        if raw.ndim != 1 or np.iscomplexobj(raw):
            return None
        arr = np.asarray(values, dtype=float)
        return None if np.isinf(arr).any() else arr
    except (TypeError, ValueError, OverflowError):
        return None


def _clean_1d(values: ArrayLike) -> np.ndarray | None:
    arr = _array(values)
    if arr is None:
        return None
    arr = arr[~np.isnan(arr)]
    return arr if arr.size >= 2 else None


def _as_float(value: Any) -> float | None:
    try:
        if np.ndim(value) != 0 or isinstance(value, (str, bytes, bool)):
            return None
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _integer(value: Any, minimum: int) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= minimum


def _reference_edges(reference: np.ndarray, bins: int) -> np.ndarray | None:
    with np.errstate(over="ignore", invalid="ignore"):
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if not np.isfinite(edges).all():
        return None
    if edges.size == 1:
        return np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _category_counts(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Birleşik kategorilerde sayım; gözlem başına Python döngüsü yok."""
    categories, inverse = np.unique(np.concatenate((reference, current)), return_inverse=True)
    return np.vstack((np.bincount(inverse[:reference.size], minlength=categories.size),
                      np.bincount(inverse[reference.size:], minlength=categories.size)))


def psi(reference: ArrayLike, current: ArrayLike, *, bins: int = 10,
        edges: ArrayLike | None = None, eps: float = 1e-4,
        kategorik_max: int = 20) -> dict[str, Any] | None:
    """Σ(c−r) ln(c/r); özellik başına aynı ölçü CSI olarak da adlandırılır.

    <0,10 STABIL; 0,10–0,25 ORTA; >0,25 BUYUK: SEKTÖR KURALI, istatistiksel test değil.
    Referansta ≤kategorik_max benzersiz değer varsa birleşik kategoriler kullanılır.
    Açık `edges` bu seçimi ezer; katı artan ve −inf/+inf uçlu olmalıdır (veri kaybı yok).
    eps ile tabanlama sonrası yeniden normalizasyon yapılmaz. `n_bos_kova`, taraflardan
    en az birinin oranı eps altındaysa kovayı BİR kez sayar (yalnız tam sıfırlar değil).
    """
    a, b = _array(reference), _array(current)
    floor = _as_float(eps)
    if (a is None or b is None or not _integer(bins, 1) or not _integer(kategorik_max, 0)
            or floor is None or not 0 < floor < 1):
        return None
    r, c = a[~np.isnan(a)], b[~np.isnan(b)]
    if r.size < 2 or c.size < 2:
        return None
    categorical = edges is None and np.unique(r).size <= kategorik_max
    bin_edges = None
    if categorical:
        counts = _category_counts(r, c)
    else:
        if edges is None:
            bin_edges = _reference_edges(r, bins)
        else:
            try:
                raw = np.asarray(edges)
                if raw.ndim != 1 or np.iscomplexobj(raw):
                    return None
                bin_edges = np.asarray(edges, dtype=float)
            except (TypeError, ValueError, OverflowError):
                return None
        if (bin_edges is None or bin_edges.size < 2
                or not np.isneginf(bin_edges[0]) or not np.isposinf(bin_edges[-1])
                or not np.all(bin_edges[1:] > bin_edges[:-1])):
            return None
        counts = np.vstack((np.histogram(r, bin_edges)[0], np.histogram(c, bin_edges)[0]))
    r_ratio, c_ratio = counts / np.array([[r.size], [c.size]])
    empty = int(np.count_nonzero((r_ratio < floor) | (c_ratio < floor)))
    r_ratio, c_ratio = np.maximum(r_ratio, floor), np.maximum(c_ratio, floor)
    contributions = (c_ratio - r_ratio) * (np.log(c_ratio) - np.log(r_ratio))
    value = float(contributions.sum())
    return {"psi": value, "bins_used": int(counts.shape[1]), "kategorik": bool(categorical),
            "n_ref": int(r.size), "n_cur": int(c.size),
            "eksik_ref": float(1 - r.size / a.size), "eksik_cur": float(1 - c.size / b.size),
            "n_bos_kova": empty, "durum": "STABIL" if value < 0.1 else "ORTA" if value <= 0.25 else "BUYUK",
            "edges": None if bin_edges is None else bin_edges.tolist(), "katki": contributions.tolist()}


def ks_drift(reference: ArrayLike, current: ArrayLike) -> dict[str, Any] | None:
    """Referans ≤20 kategoriyse Pearson χ² (Yates düzeltmesiz), aksi halde iki-örneklem KS.

    Beklenen hücrelerin >%20'si <5 ise p=None/reliable=False. Tek kategoride
    serbestlik derecesi yok: stat=0, p ve Cramér V=None; anlamlılık uydurulmaz.
    """
    r, c = _clean_1d(reference), _clean_1d(current)
    if r is None or c is None:
        return None
    if np.unique(r).size <= 20:
        table = _category_counts(r, c)
        if table.shape[1] == 1:
            return {"stat": 0.0, "p": None, "cramer_v": None, "test": "chi2", "reliable": False}
        stat, p, _, expected = stats.chi2_contingency(table, correction=False)
        reliable = bool(np.mean(expected < 5) <= 0.2)
        return {"stat": float(stat), "p": float(p) if reliable else None,
                "cramer_v": float(np.sqrt(stat / table.sum())), "test": "chi2", "reliable": reliable}
    result = stats.ks_2samp(r, c)
    return {"stat": float(result.statistic), "p": float(result.pvalue), "test": "ks"}


def wasserstein_drift(reference: ArrayLike, current: ArrayLike) -> dict[str, Any] | None:
    """Wasserstein-1 ve referansın örneklem std'sine (ddof=1) oranı; sıfır std → None oran."""
    r, c = _clean_1d(reference), _clean_1d(current)
    if r is None or c is None:
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        w1 = float(stats.wasserstein_distance(r, c))
        sd = float(np.std(r, ddof=1))
        scaled = w1 / sd if sd > 0 and np.isfinite(sd) else None
    if not np.isfinite(w1):
        return None
    return {"w1": w1, "w1_std": scaled if scaled is not None and np.isfinite(scaled) else None}


def fisher_rao_normal(mu1: float, sd1: float, mu2: float, sd2: float) -> float | None:
    """Normal ailesinin Fisher–Rao jeodeziği; sonlu parametreler ve σ>0 gerekir.

    Ortak ölçekleme taşmayı önler. arccosh(1+x)=2 asinh(sqrt(x/2)) özdeşliği
    yakın dağılımlarda 1+x yuvarlanmasının sahte sıfır üretmesini engeller.
    """
    values = [_as_float(v) for v in (mu1, sd1, mu2, sd2)]
    if any(v is None for v in values):
        return None
    m1, s1, m2, s2 = (float(v) for v in values if v is not None)
    if s1 <= 0 or s2 <= 0:
        return None
    scale = max(abs(m1), abs(m2), s1, s2)
    m1, s1, m2, s2 = m1 / scale, s1 / scale, m2 / scale, s2 / scale
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        numerator = np.hypot((m1 - m2) / np.sqrt(2.0), s1 - s2)
        denominator = 2.0 * np.sqrt(s1) * np.sqrt(s2)
        distance = float(2.0 * np.sqrt(2.0) * np.arcsinh(numerator / denominator))
    return distance if np.isfinite(distance) else None


def fisher_rao_drift(reference: ArrayLike, current: ArrayLike) -> dict[str, Any] | None:
    """Örneklem momentleri (ddof=1) ile normal yaklaşımı; normalite testi DEĞİLDİR."""
    r, c = _clean_1d(reference), _clean_1d(current)
    if r is None or c is None:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        mr, sr, mc, sc = float(r.mean()), float(r.std(ddof=1)), float(c.mean()), float(c.std(ddof=1))
    distance = fisher_rao_normal(mr, sr, mc, sc)
    if distance is None:
        return None
    return {"fr": distance, "mu_ref": mr, "sd_ref": sr, "mu_cur": mc, "sd_cur": sc,
            "normal_varsayimi": True}


def feature_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                         features: Sequence[str] | None = None, *, min_n: int = 200,
                         alpha: float = 0.05, bins: int = 10) -> dict[str, Any] | None:
    """Özellik bazında düz anahtarlı dört ölçü ve geçerli p'lere Bonferroni karnesi.

    n büyüdükçe önemsiz kaymalar da anlamlı çıkar: ETKİ BÜYÜKLÜĞÜ (PSI, w1_std, fr)
    önce okunur. Testler bağımsız gözlem varsayar; zaman bağımlılığını düzeltmez.
    m_test yalnız güvenilir, ölçülebilmiş p'leri sayar; diğerlerinde anlamli=None.
    Varsayılan kolonlar iki çerçevenin ortak sayısal kolonlarıdır; FEATURE_NAMES kopyalanmaz.
    """
    a = _as_float(alpha)
    if (not isinstance(reference_df, pd.DataFrame) or not isinstance(current_df, pd.DataFrame)
            or not reference_df.columns.is_unique or not current_df.columns.is_unique
            or not _integer(min_n, 2) or not _integer(bins, 1) or a is None or not 0 < a < 1):
        return None
    if features is None:
        names = [name for name in reference_df.select_dtypes(include="number").columns
                 if name in current_df.select_dtypes(include="number").columns]
    else:
        if isinstance(features, (str, bytes)):
            return None
        try:
            names = list(features)
            if any(not isinstance(name, str) for name in names):
                return None
            names = list(dict.fromkeys(names))
        except TypeError:
            return None
    if not names:
        return None
    rows: dict[str, dict[str, Any]] = {}
    for name in names:
        if name not in reference_df or name not in current_df:
            rows[name] = {"yetersiz": True, "neden": "Kolon iki çerçevede de bulunmalı."}
            continue
        r, c = _clean_1d(reference_df[name]), _clean_1d(current_df[name])
        if r is None or c is None or min(r.size, c.size) < min_n:
            rows[name] = {"yetersiz": True, "neden": f"Geçerli 1B veri ve iki tarafta en az {min_n} gözlem gerekli."}
            continue
        population = psi(reference_df[name], current_df[name], bins=bins)
        if population is None:
            rows[name] = {"yetersiz": True, "neden": "Referans kovaları sayısal olarak ölçülemedi."}
            continue
        row = {**population, "yetersiz": False, "stat": None, "p": None, "test": None,
               "w1": None, "w1_std": None, "fr": None}
        for measure in (ks_drift(r, c), wasserstein_drift(r, c), fisher_rao_drift(r, c)):
            if measure is not None:
                row.update(measure)
        rows[name] = row
    measured = {name: row for name, row in rows.items() if not row["yetersiz"]}
    m = sum(row["p"] is not None for row in measured.values())
    for row in measured.values():
        p_bonf = min(1.0, row["p"] * m) if row["p"] is not None else None
        row.update(p_bonf=p_bonf, anlamli=p_bonf < a if p_bonf is not None else None)
    return {"ozellikler": rows, "siralama": sorted(measured, key=lambda name: -measured[name]["psi"]),
            "n_buyuk": sum(row["durum"] == "BUYUK" for row in measured.values()),
            "n_orta": sum(row["durum"] == "ORTA" for row in measured.values()),
            "n_stabil": sum(row["durum"] == "STABIL" for row in measured.values()),
            "n_yetersiz": len(rows) - len(measured), "m_test": m, "alpha": a, "min_n": min_n}


def load_live_features(path: str | PathLike[str], *, since: str | pd.Timestamp | None = None,
                       until: str | pd.Timestamp | None = None,
                       interval: str | None = None) -> pd.DataFrame | None:
    """Yerel CSV'yi oku; KANONİK-UTC bar_ts üzerinde [since, until) ve interval süzgeci uygula.

    `bar_ts` bu depoda KARIŞIK yazılır ve bu BİLİNEN bir durumdur: canlı sağlayıcı
    failover'ı bazı satırları tz-aware (ET), bazılarını naive bırakır
    (`prediction_tracker._canon_ts` docstring'i, Denetim Tur-11). Ölçüldü (2026-09-16,
    üretim logu 27.260 satır): satırların **%37,2'si tz-aware**, 30 sembolün 30'u da
    karışık ve karışım zamana yayılı (günlük oran %9→%50) — yani tz-aware'i REDDETMEK
    logun TAMAMINI reddetmek demekti (`format="mixed"` karışık tz'de zaten `ValueError`
    atar → fonksiyon None döndürüyordu, modül üretimde HİÇ okuyamıyordu).

    Bu yüzden sözleşme deponun ZATEN KANONİK olan dört kopyasıyla (`_canon_ts`,
    `calibration_loop._parse_ts_utc`, `ai/memory._parse_ts`, `ai/decision._parse_ts`)
    BİRLEŞTİRİLDİ: **tz-aware → UTC'ye çevrilip tz düşülür; naive → ZATEN UTC sayılır.**
    Naive satırların gerçekten UTC olduğu ölçülerek doğrulandı: iki grup da aynı UTC
    seans bandına (13-20) düşüyor ve `logged_at − bar_ts` gecikmesinin medyanı iki grupta
    da ~10.900 sn (yerel saat farkı + ~2 dk kayıt gecikmesi). Dönüş HER ZAMAN naive'dir
    ki karşılaştırmalar aware/naive TypeError'ına düşmesin.

    `since`/`until` hâlâ naive olmalıdır (tz-aware sınır → None): sınırın hangi tz'de
    okunacağı ÇAĞIRANIN kararıdır ve sessizce varsayılmaz. Bozuk/eksik tarih → None.
    Geçerli süzgecin boş sonucu boş DataFrame'dir; okuma hatası None'dır.
    Özelliklerdeki eksik değerler nötr doldurulmaz, ek şema kolonları korunur.
    """
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            frame = pd.read_csv(handle, low_memory=False)
        if "bar_ts" not in frame or frame.empty:
            return None
        # utc=True: naive ZATEN UTC sayılır, tz-aware UTC'ye çevrilir (tek sözleşme);
        # ardından tz düşürülür → sonuç naive-UTC, `_canon_ts` ile BİT-DENK.
        times = pd.to_datetime(frame["bar_ts"], format="mixed", errors="raise", utc=True)
        if times.isna().any():
            return None
        times = times.dt.tz_localize(None)
        start = pd.Timestamp(since) if since is not None else None
        end = pd.Timestamp(until) if until is not None else None
        if any(t is not None and (pd.isna(t) or t.tzinfo is not None) for t in (start, end)):
            return None
        if start is not None and end is not None and start >= end:
            return None
        frame["bar_ts"] = times
        mask = pd.Series(True, index=frame.index)
        if start is not None:
            mask &= times >= start
        if end is not None:
            mask &= times < end
        if interval is not None:
            if not isinstance(interval, str) or "interval" not in frame:
                return None
            mask &= frame["interval"] == interval
        return frame.loc[mask].copy()
    except (OSError, TypeError, ValueError, OverflowError, AttributeError):
        return None


def time_sliced_psi(reference: ArrayLike, current_values: ArrayLike, current_times: ArrayLike,
                    *, freq: str = "W", bins: int = 10, min_n: int = 100) -> dict[str, Any] | None:
    """Referans kovalarını bir kez kur; pandas resample dönemlerinde PSI ölç.

    n geçerli (NaN olmayan) gözlem sayısıdır; boş/az gözlemli dönem → None.
    Kategorik kip her dönemde referans + yalnız O DÖNEMİN kategorilerini kullanır;
    gelecekteki yeni kategoriler geçmiş PSI'yi değiştiremez. Etiketler pandas'ın
    dönem etiketleridir; dönem içindeki geleceği görmemek için kapanmış dönemleri verin.
    """
    r, values = _clean_1d(reference), _array(current_values)
    if (r is None or values is None or not _integer(min_n, 2) or not _integer(bins, 1)
            or not isinstance(freq, str) or np.count_nonzero(~np.isnan(values)) < 2):
        return None
    try:
        raw_times = np.asarray(current_times)
        if raw_times.ndim != 1 or raw_times.size != values.size:
            return None
        times = pd.DatetimeIndex(pd.to_datetime(raw_times, format="mixed", errors="raise"))
        if times.isna().any() or times.tz is not None:
            return None
        categorical = np.unique(r).size <= 20
        edges = None if categorical else _reference_edges(r, bins)
        if not categorical and edges is None:
            return None
        periods: list[str] = []
        scores: list[float | None] = []
        counts: list[int] = []
        for period, sample in pd.Series(values, index=times).resample(freq):
            count = int(sample.count())
            result = psi(r, sample.to_numpy(), bins=bins, edges=edges) if count >= min_n else None
            periods.append(str(period))
            scores.append(result["psi"] if result is not None else None)
            counts.append(count)
        return {"donem": periods, "psi": scores, "n": counts}
    except (TypeError, ValueError, OverflowError):
        return None
