"""
factors.py — Fama-French / Carhart faktör serileri (Ken French kütüphanesi). ÜCRETSİZ.

Listenin 224. maddesi ("Barra / Fama-French faktör maruziyeti") ve 221. maddesi
("faktör taklitçi portföy") tek bir veri köprüsüne bağlıydı ve o köprü ANAHTARSIZDIR:

    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/<set>_CSV.zip

Ölçüldü (2026-09-07, bu makine): 5 faktör günlük dosyası **200 · 150 KB · zip**,
1963-07-01 → **2026-07-31**. Yani seri BİR AY GECİKMELİDİR ve bu gizlenmez:
`attrs["son_tarih"]` taşınır, dünün getirisi faktörlere karşı REGRESE EDİLEMEZ.

## Neden bu köprü bu depoda önemli

Bu projenin merkezî sorusu *"kenar var mı?"*dır ve bugüne kadar her aday **kendi
başına** ölçüldü. Faktör regresyonu farklı bir soru sorar: *"bu getirinin ne kadarı
zaten BİLİNEN bir risk primidir?"* Pozitif ama tamamı piyasa betasından gelen bir
seri, kenar DEĞİL maruziyettir. `factor_attribution` alfayı bu maruziyetlerden ARTA
KALAN olarak ölçer.

## İki sessiz hata, ikisi de burada kapalı

**1. YÜZDE ↔ ORAN.** Ken French dosyaları YÜZDE cinsindendir (`-0.67` = %−0,67).
Getiri serisi oran cinsindeyse (0.0067) bölmeden regresyona sokmak alfayı **100 kat**
şişirir ve HATA VERMEZ. Burada seriler 100'e BÖLÜNÜR ve `birim="oran"` taşınır.

**2. OTOKORELASYON.** Günlük getiriler bağımsız değildir; sıradan EKK standart
hataları alfanın t'sini şişirir. `factor_attribution` **Newey-West (HAC)** standart
hatası kullanır ve `lag` seçimini çıktıda taşır — Tur-73'ün "örtüşen pencereden
hesaplanan bir p BAĞIMSIZLIK varsayar" dersinin regresyon tarafıdır.

Modül KARAR VERMEZ: eşik/kural içermez, `signals`/`risk`/`execution` bu modülü
import ETMEZ (testle kilitli).
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from nihai.config import DATA_DIR

BASE = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/")
CACHE_DIR = DATA_DIR / "factor_cache"
MAX_AGE_DAYS = 7
TIMEOUT = 60

# ad → (zip dosyası, kolonların kanonik adları)
DATASETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ff5_daily": ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
                  ("mkt_rf", "smb", "hml", "rmw", "cma", "rf")),
    "ff3_daily": ("F-F_Research_Data_Factors_daily_CSV.zip",
                  ("mkt_rf", "smb", "hml", "rf")),
    "mom_daily": ("F-F_Momentum_Factor_daily_CSV.zip", ("mom",)),
}


def _indir(dosya: str) -> bytes | None:
    try:
        import requests

        from nihai import net
        r = requests.get(BASE + dosya, timeout=TIMEOUT,
                         headers={"User-Agent": net.contact_ua("NihaiAlgoritma research")})
        r.raise_for_status()
        return r.content
    except Exception as e:                                # ağ hatası bayat kopyayı öldürmesin
        logger.warning(f"[faktor] {dosya} indirilemedi: {type(e).__name__}: {e}")
        return None


def _ayristir(ham: bytes, kolonlar: tuple[str, ...]) -> pd.DataFrame:
    """Ken French CSV'sini ayrıştır: başlık metni + YYYYMMDD satırları + telif dipnotu.

    Dosyanın YAPISI ayrıştırmanın parçasıdır: üstte serbest metin, ortada veri, altta
    "Copyright ..." satırı ve bazı setlerde İKİNCİ bir tablo (aylık) vardır. Bu yüzden
    satırlar `pandas`a körlemesine verilmez — YALNIZ 8 haneli tarihle başlayan satırlar
    alınır. Aksi hâlde aylık tablo günlük tablonun ALTINA eklenir ve tarihler sessizce
    ikiye katlanır.
    """
    with zipfile.ZipFile(io.BytesIO(ham)) as z:
        metin = z.read(z.namelist()[0]).decode("latin-1")
    satirlar = [s.strip() for s in metin.replace("\r", "").split("\n")]
    veri = [s for s in satirlar if len(s) > 8 and s[:8].isdigit() and s[8:9] in (",", "")]
    if not veri:
        return pd.DataFrame()
    d = pd.read_csv(io.StringIO("\n".join(veri)), header=None)
    if d.shape[1] < len(kolonlar) + 1:
        return pd.DataFrame()
    d = d.iloc[:, : len(kolonlar) + 1]
    d.columns = ["tarih", *kolonlar]
    d["tarih"] = pd.to_datetime(d["tarih"].astype(str), format="%Y%m%d", errors="coerce")
    d = d.dropna(subset=["tarih"]).set_index("tarih").sort_index()
    # YÜZDE → ORAN (modül docstring'i, sessiz hata 1). -99.99 = eksik gözlem işareti.
    for k in kolonlar:
        d[k] = pd.to_numeric(d[k], errors="coerce")
        d.loc[d[k] <= -99.0, k] = np.nan
        d[k] = d[k] / 100.0
    return d


def french_factors(dataset: str = "ff5_daily", *, max_age_days: int = MAX_AGE_DAYS,
                   fetch: bool = True, with_momentum: bool = False) -> pd.DataFrame:
    """Faktör serisi (ORAN cinsinden, tarih indeksli). Fail-soft: hata → boş çerçeve.

    `fetch=False` ÖNBELLEK-YALNIZ okur (BULGU-59-8). `with_momentum=True` Carhart
    momentum faktörünü AYRI dosyadan çeker ve ORTAK tarihlerde birleştirir — birleşim
    `join="inner"`dır: momentum serisi kısa kaldığında sessizce NaN taşımak yerine
    örneklem KISALIR ve `attrs["n"]` bunu söyler.
    """
    if dataset not in DATASETS:
        raise ValueError(f"bilinmeyen faktör seti: {dataset!r} (bkz. DATASETS)")
    dosya, kolonlar = DATASETS[dataset]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = Path(CACHE_DIR) / f"{dataset}.csv"

    taze = dest.exists() and (time.time() - dest.stat().st_mtime) / 86400.0 < max_age_days
    if fetch and not taze:
        ham = _indir(dosya)
        if ham is not None:
            d = _ayristir(ham, kolonlar)
            if not d.empty:
                d.to_csv(dest)
    if not dest.exists():
        return pd.DataFrame()
    try:
        out = pd.read_csv(dest, index_col=0, parse_dates=True)
    except (OSError, ValueError) as e:                    # pragma: no cover
        logger.warning(f"[faktor] {dataset} okunamadı: {e}")
        return pd.DataFrame()

    if with_momentum and dataset != "mom_daily":
        mom = french_factors("mom_daily", max_age_days=max_age_days, fetch=fetch)
        if not mom.empty:
            out = out.join(mom, how="inner")
    out.attrs["dataset"] = dataset
    out.attrs["birim"] = "oran"
    out.attrs["son_tarih"] = str(out.index[-1].date()) if len(out) else None
    out.attrs["n"] = int(len(out))
    return out


def _hac_se(x: np.ndarray, artik: np.ndarray, xtx_inv: np.ndarray, lag: int) -> np.ndarray:
    """Newey-West (HAC) standart hataları — Bartlett çekirdeği, `lag` gecikmeye kadar."""
    n = x.shape[0]
    u = x * artik[:, None]
    s = u.T @ u
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        g = u[L:].T @ u[:-L]
        s += w * (g + g.T)
    kov = xtx_inv @ s @ xtx_inv * (n / max(1, n - x.shape[1]))
    return np.sqrt(np.maximum(np.diag(kov), 0.0))


def factor_attribution(returns: pd.Series, factors: pd.DataFrame | None = None, *,
                       fazlalik: bool = True, lag: int | None = None,
                       fetch: bool = False) -> dict | None:
    """Getiri serisini faktörlere regres et → ALFA + yüklemeler (Madde 224/221).

    `returns` ORAN cinsinden GÜNLÜK getiri (0,01 = %1) olmalıdır; birim karışırsa alfa
    100 kat yanlış çıkar ve HATA VERMEZ (modül docstring'i). `fazlalik=True` ise sol
    taraf `r − rf`dir (faktörlerin kendisi de fazlalık getiridir; ham getiriyi fazlalık
    faktörlere regres etmek sabiti risksiz faizle kirletir).

    `lag=None` → Newey-West gecikmesi `floor(4·(n/100)^(2/9))` (Newey-West 1994 kuralı).
    Ortak tarih sayısı 60'ın altındaysa **None** döner: alfanın standart hatası o
    örneklemde okunacak bir sayı değildir.

    Dönüş: `{"alpha", "alpha_yillik", "alpha_t", "yuklemeler", "t", "r2", "n", "lag"}`.
    Bir KENAR İDDİASI DEĞİLDİR: alfa pozitif çıksa bile çoklu-test ve exec-lag kapıları
    ayrıca sorulmalıdır (ACI DERS #3/#7).
    """
    if factors is None:
        factors = french_factors(fetch=fetch)
    if factors is None or factors.empty or returns is None or len(returns) == 0:
        return None
    r = pd.Series(returns).dropna()
    r.index = pd.DatetimeIndex(r.index).tz_localize(None).normalize()
    f = factors.copy()
    f.index = pd.DatetimeIndex(f.index).tz_localize(None).normalize()
    ortak = r.index.intersection(f.index)
    if len(ortak) < 60:
        return None
    y = r.loc[ortak].to_numpy(dtype=float)
    rf = f.loc[ortak, "rf"].to_numpy(dtype=float) if "rf" in f.columns else np.zeros(len(ortak))
    if fazlalik:
        y = y - rf
    adlar = [c for c in f.columns if c != "rf"]
    x = np.column_stack([np.ones(len(ortak)),
                         f.loc[ortak, adlar].to_numpy(dtype=float)])
    iyi = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y, x = y[iyi], x[iyi]
    n = x.shape[0]
    if n < 60:
        return None
    try:
        xtx_inv = np.linalg.inv(x.T @ x)
    except np.linalg.LinAlgError:                          # tekil tasarım matrisi
        return None
    beta = xtx_inv @ (x.T @ y)
    artik = y - x @ beta
    L = int(lag) if lag is not None else int(4 * (n / 100.0) ** (2.0 / 9.0))
    se = _hac_se(x, artik, xtx_inv, max(0, L))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((artik ** 2).sum()) / ss_tot if ss_tot > 0 else None
    return {
        "alpha": float(beta[0]),
        "alpha_yillik": float(beta[0]) * 252.0,
        "alpha_t": (float(beta[0] / se[0]) if se[0] > 0 else None),
        "yuklemeler": {ad: float(b) for ad, b in zip(adlar, beta[1:], strict=True)},
        "t": {ad: (float(b / s) if s > 0 else None)
              for ad, b, s in zip(adlar, beta[1:], se[1:], strict=True)},
        "r2": r2, "n": int(n), "lag": max(0, L), "fazlalik": bool(fazlalik),
        "son_faktor_tarihi": str(f.index[-1].date()),
    }


def _cli() -> None:  # pragma: no cover - operatör aracı
    import argparse

    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    p = argparse.ArgumentParser(description="Ken French faktör köprüsü")
    p.add_argument("--dataset", default="ff5_daily", choices=sorted(DATASETS))
    p.add_argument("--momentum", action="store_true")
    a = p.parse_args()
    d = french_factors(a.dataset, with_momentum=a.momentum)
    if d.empty:
        print("faktör serisi alınamadı")
        return
    print(f"{a.dataset}: {len(d)} gün · {d.index[0].date()} → {d.index[-1].date()}")
    print(f"birim: {d.attrs.get('birim')} (yüzde DEĞİL)")
    print(d.tail(3).to_string())


if __name__ == "__main__":  # pragma: no cover
    _cli()
