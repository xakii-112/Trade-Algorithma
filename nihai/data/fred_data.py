"""
fred_data.py — FRED (St. Louis Fed) makro serileri. ANAHTAR GEREKMEZ.

BACKLOG'un "veri genişletme" tablosu FRED'i aylardır *"ücretsiz; anahtar açık madde"*
diye bekletiyordu. 2026-08-29'da ölçüldü: **API anahtarı gerekmiyor** —
`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERI>` düz CSV döndürür.

**SINIR (ölçüldü, gizlenmiyor):** grafik uç noktası her seri için kendi VARSAYILAN
penceresini uygular ve `cosd`/`coed` her seride dinlenmez. T10Y2Y ve NFCI 2020'den
itibaren geldi; `BAMLH0A0HYM2` (HY OAS) `cosd=2015` verilse bile yalnız son ~3 yılı
döndürdü. Bu yüzden `get()` istenen başlangıcı KONTROL EDER ve tutmayan seriyi
`attrs["kirpildi"]` ile işaretler — sessizce kısa bir seri döndürmek, Tur-51'in 20 aylık
deliğinin ta kendisi olurdu.

Bu modül CBOE köprüsüyle (`cboe_data.py`) AYNI sözleşmededir: yalnız veri taşır, hiçbir
karar vermez, ağ hatasında fail-soft (boş seri) döner.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from loguru import logger

from nihai.config import DATA_DIR

try:  # opsiyonel bağımlılık TÜM kapıyı düşürmemeli (Tur-26 dersi)
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
CACHE_DIR = DATA_DIR / "fred_cache"
MAX_AGE_DAYS = 7
TIMEOUT = 60
# Kırpılma bayrağının TOLERANSI: bir serinin ilk gözlemi istenen günden birkaç gün
# sonra olabilir (tatil, haftalık frekans). Toleranssız bayrak HER seride yanar ve
# her yerde yanan uyarı hiç yanmıyor demektir (Tur-48).
KIRPILMA_TOLERANS_GUN = 10


def get(series: str, *, since: str = "2015-01-01", max_age_days: int = MAX_AGE_DAYS,
        force: bool = False, fetch: bool = True) -> pd.Series:
    """Tek seriyi tarih-indeksli float Series olarak getir (önbellekli, fail-soft).

    `attrs["kirpildi"]` True ise seri istenen `since`'ten SONRA başlıyordur — yani
    uç nokta pencereyi kendisi kısmıştır ve o dönem için ÖLÇÜM YOKTUR.

    `fetch=False` ÖNBELLEK-YALNIZ okur ve ağa HİÇ çıkmaz (`cboe_data.term_structure`
    ile aynı sözleşme). Pano/uç yolları bunu kullanmalıdır: bir HTTP isteğinin içinde
    20 FRED serisi indirmek "ölçüm bütçesini OKUYAN da öder" kuralının ihlalidir
    (BULGU-59-8). Önbellek yoksa boş seri döner — "veri yok" değil "ölçmedim".
    """
    if requests is None:
        logger.warning("[fred] requests yok — seri alınamadı")
        return pd.Series(dtype=float)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = Path(CACHE_DIR) / f"{series}.csv"

    taze = dest.exists() and (time.time() - dest.stat().st_mtime) / 86400.0 < max_age_days
    if not fetch:
        if not dest.exists():
            return pd.Series(dtype=float)
    elif force or not taze:
        try:
            r = requests.get(FRED_CSV, params={"id": series, "cosd": since},
                             timeout=TIMEOUT)
            r.raise_for_status()
            dest.write_bytes(r.content)
        except Exception as e:                       # ağ hatası bayat kopyayı öldürmesin
            logger.warning(f"[fred] {series} indirilemedi: {e}")
            if not dest.exists():
                return pd.Series(dtype=float)

    try:
        d = pd.read_csv(dest)
    except (OSError, ValueError) as e:
        logger.warning(f"[fred] {series} okunamadı: {e}")
        return pd.Series(dtype=float)
    if d.shape[1] < 2:
        return pd.Series(dtype=float)
    tarih, deger = d.columns[0], d.columns[1]
    d[tarih] = pd.to_datetime(d[tarih], errors="coerce")
    # FRED eksik gözlemi "." ile yazar
    d[deger] = pd.to_numeric(d[deger].replace(".", pd.NA), errors="coerce")
    s = d.dropna(subset=[tarih]).set_index(tarih)[deger].astype(float).sort_index()
    s.name = series
    esik = pd.Timestamp(since) + pd.Timedelta(days=KIRPILMA_TOLERANS_GUN)
    s.attrs["kirpildi"] = bool(len(s) and s.index[0] > esik)
    if s.attrs["kirpildi"]:
        logger.warning(f"[fred] {series}: istenen {since}, gelen {s.index[0].date()} "
                       f"— uç nokta pencereyi KISTI, o dönem için ölçüm YOK")
    return s


def get_all(series: list[str], *, since: str = "2015-01-01") -> pd.DataFrame:
    """Birden çok seriyi tek çerçevede birleştir (dış birleşim, ileri doldurma YOK).

    İleri doldurma bilerek yapılmaz: haftalık NFCI'yi günlük ızgaraya yaymak ÇAĞIRANIN
    kararıdır ve ne kadar geriye doldurulacağı bir look-ahead sorusudur (Tur-51: sınırsız
    ffill bir haftadan eski değeri "bugünkü" diye sunar).

    `attrs["kirpik"]` (Tur-63): penceresi KISILMIŞ serilerin adları. `get()` bunu seri
    başına `attrs["kirpildi"]` diye işaretliyordu ama Series → DataFrame geçişinde attrs
    KAYBOLUYOR: bayrak yazılıyor, hiçbir yerden okunamıyordu (`build_babmacro_dataset` bu
    yüzden aynı ölçümü elle yeniden yapıyordu). Kırpılma bilgisi çağırana çerçeveyle
    BİRLİKTE ulaşmalı — ölçümü taşımayan bayrak, ölçüm yokluğuyla aynı şeydir.
    """
    kol = {}
    kirpik: list[str] = []
    for s in series:
        v = get(s, since=since)
        if not v.empty:
            kol[s] = v
            if v.attrs.get("kirpildi"):
                kirpik.append(s)
    if not kol:
        return pd.DataFrame()
    df = pd.DataFrame(kol).sort_index()
    df.attrs["kirpik"] = kirpik
    return df


# ---------------------------------------------------------------------------
# ABD Hazine getiri eğrisi (Madde 145/211/212) — hepsi ANAHTARSIZ, hepsi ücretsiz
# ---------------------------------------------------------------------------
# Sabit-vadeli (constant maturity) nominal Hazine getirileri. Değer = YILLIK YÜZDE.
# Anahtar = FRED seri kodu, değer = YIL cinsinden vade (nss_fit'in beklediği birim).
TREASURY_MATURITIES: dict[str, float] = {
    "DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0, "DGS2": 2.0,
    "DGS3": 3.0, "DGS5": 5.0, "DGS7": 7.0, "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0,
}
# Eğri türevleri: FRED bu farkları KENDİSİ yayımlar — elle çıkarmak yerine kaynağından
# okumak, iki serinin farklı günlerde güncellenmesinden doğan hayalet inversiyonları önler.
CURVE_SPREADS = ("T10Y2Y", "T10Y3M")
BREAKEVEN = ("T5YIE", "T10YIE")     # TIPS başabaş enflasyon (5Y / 10Y)


def treasury_curve(*, since: str = "2015-01-01") -> pd.DataFrame:
    """Sabit-vadeli ABD Hazine getiri eğrisi paneli (gün × vade), YILLIK YÜZDE.

    Kolonlar `TREASURY_MATURITIES` anahtarlarıdır; `attrs["vadeler"]` kolon → YIL
    eşlemesini, `attrs["kirpik"]` penceresi kısılmış serileri taşır (`get_all` sözleşmesi).

    İLERİ DOLDURMA YOK: bir vade o gün yayımlanmadıysa NaN kalır. Eğri uydurmadan önce
    çağıran satırı `dropna()` ile temizlemelidir — eksik vadeyi komşusundan türetmek
    eğrinin şeklini UYDURMAK olur ve `nss_fit`'in ölçtüğü şey tam olarak o şekildir."""
    df = get_all(list(TREASURY_MATURITIES), since=since)
    if df.empty:
        return df
    df.attrs["vadeler"] = {k: v for k, v in TREASURY_MATURITIES.items() if k in df.columns}
    return df


def curve_spreads(*, since: str = "2015-01-01") -> pd.DataFrame:
    """Getiri eğrisi farkları (10Y−2Y, 10Y−3M) ve TIPS başabaş enflasyonu (5Y, 10Y).

    Değerler YÜZDE PUANDIR: T10Y2Y < 0 = eğri TERS (resesyon öncüsü olarak izlenen hâl).
    Bu modül bir karar vermez — inversiyonun ne anlama geldiği çağıranın sorusudur."""
    return get_all(list(CURVE_SPREADS) + list(BREAKEVEN), since=since)
