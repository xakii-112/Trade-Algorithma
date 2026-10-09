"""
cboe_data.py — CBOE ücretsiz volatilite endeksleri (anahtarsız, resmî kaynak).

NEDEN: projenin test ettiği ALTI sinyal ailesi de (PEAD, meta-labeling, momentum,
pairs, lead-lag, endeks rebalans) hisse kesitiydi ve ACI DERS #1 (survivorship) ile
#7 (exec-lag) hepsini yedi. Buradaki seriler ENDEKS düzeyindedir: endeks delist
olmaz, kurucu değişmez, geriye dönük düzeltilmez → survivorship yapısı gereği YOKTUR.
Yani rejim/koşullama özelliği olarak kullanıldıklarında ACI DERS #1'i tetiklemezler.

Kaynak: `https://cdn.cboe.com/api/global/us_indices/daily_prices/<AD>_History.csv`
Anahtar/kayıt gerekmez. Doğrulandı (2026-08-13): beş seri de canlı, VIX 1990-01-02'den
bugüne, SKEW 1990'dan, VVIX 2006'dan.

MEVCUT KODLA İLİŞKİ: `strategy._load_vix()` VIX'i yfinance ile (`yf.download("^VIX")`,
10 yıl, modül-global) çeker. Bu modül onu DEĞİŞTİRMEZ — canlı sinyal yolundaki bir
girdiyi sessizce başka kaynağa taşımak ölçülmüş sonuçları geçersiz kılardı
("mevcut disipline saygı"). Burası araştırma/özellik tarafı içindir; canlıya
taşınacaksa A/B ölçümüyle taşınmalı.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from nihai.config import DATA_DIR

CBOE_BASE = "https://cdn.cboe.com/api/global/us_indices/daily_prices"
CBOE_CACHE_DIR = DATA_DIR / "cboe_cache"

# Doğrulanmış seriler. VIX6M bazı dönemlerde 404 verebilir → fail-soft ele alınır.
SERIES = {
    "VIX": "VIX_History.csv",        # 30 gün örtük vol       (1990→)
    "VIX9D": "VIX9D_History.csv",    # 9 gün                  (2011→)
    "VIX3M": "VIX3M_History.csv",    # 3 ay                   (2009→)
    "VIX6M": "VIX6M_History.csv",    # 6 ay                   (opsiyonel)
    "VVIX": "VVIX_History.csv",      # vol'ün vol'ü           (2006→)
    "SKEW": "SKEW_History.csv",      # kuyruk riski fiyatı    (1990→)
    # Tur-79 (yol haritası §7.5) — hepsi 2026-09-16'da HTTP 200 doğrulandı:
    "VIX1D": "VIX1D_History.csv",    # tek işlem günü (2022-05→); ileri varyansa GİRMEZ (aşağıda)
    "VIX1Y": "VIX1Y_History.csv",    # 1 yıl (2007→)
    "COR1M": "COR1M_History.csv",    # S&P 500 örtük korelasyon, ilk 50 bileşen, ~1 ay (2006→)
    "COR3M": "COR3M_History.csv",    # ~3 ay
    "COR6M": "COR6M_History.csv",
    "COR9M": "COR9M_History.csv",
    "COR1Y": "COR1Y_History.csv",
    "DSPX": "DSPX_History.csv",      # S&P 500 dispersiyon endeksi (2014-06→), tek kolon
}

# Vade yapısı endekslerinin HEDEF SABİT VADESİ (takvim günü). Kaynak: Cboe "Selected SPX
# Target Expected Volatility Term Indices" metodolojisi (2026-09-16 okundu): VIX3M **93**,
# VIX6M **184**, VIX1Y **366** gün (365 DEĞİL), zaman ölçüsü dakika/525.600 = gün/365.
# VIX1D bilerek YOK: vadesi o günün kalan İŞLEM DAKİKALARIDIR (sabit takvim vadesi değil),
# aynı toplam-varyans ekseninde karşılaştırılamaz.
VADE_GUN = {"VIX9D": 9, "VIX": 30, "VIX3M": 93, "VIX6M": 184, "VIX1Y": 366}

CACHE_MAX_AGE_DAYS = int(os.getenv("CBOE_CACHE_MAX_AGE_DAYS", "1"))


def _cache_path(name: str) -> Path:
    return CBOE_CACHE_DIR / f"{name}.csv"


def _read_cache(name: str, *, ignore_age: bool = False) -> pd.DataFrame | None:
    """Diskteki seriyi oku. `ignore_age=True` ise BAYAT önbellek de kabul edilir —
    bu, ağa ÇIKMAYAN okuyucular içindir (`fetch=False`); bayatlığı çağıran raporlar."""
    p = _cache_path(name)
    if not p.exists():
        return None
    try:
        age = (datetime.now().date() - datetime.fromtimestamp(p.stat().st_mtime).date()).days
        if not ignore_age and age >= CACHE_MAX_AGE_DAYS:
            return None
        return pd.read_csv(p, index_col=0, parse_dates=True)
    except Exception as e:
        logger.debug(f"[cboe] cache okunamadı {p.name}: {e}")
        return None


def get_index(name: str, *, use_cache: bool = True, fetch: bool = True) -> pd.DataFrame:
    """Tek seriyi getir. Index=tarih (tz-naive), kolonlar küçük harf.

    VIX/VIX9D/VIX3M OHLC verir, VVIX/SKEW tek kolon (close'a normalize edilir).
    Ağ/parse hatasında BOŞ df döner (fail-soft) — çağıran çökmez.

    **`fetch=False` = ÖNBELLEK-YALNIZ, ağa hiç çıkmaz** (bayat önbellek de kabul edilir,
    yoksa BOŞ df). Bir HTTP ucundan ya da sıkı bir döngüden çağıran HER okuyucu bunu
    kullanmalıdır: ölçüm bütçesini OKUYAN da öder (Tur-35/36) ve bir okuma yoluna saniyelik
    ağ çağrısı koymak Tur-59'un BULGU-59-8'inin ta kendisidir. Önbelleği DOLDURMAK ayrı ve
    bilinçli bir adımdır: `python -m nihai.data.cboe_data`.
    """
    name = name.upper()
    if name not in SERIES:
        raise ValueError(f"bilinmeyen seri {name!r}; geçerli: {sorted(SERIES)}")

    if use_cache:
        cached = _read_cache(name, ignore_age=not fetch)
        if cached is not None:
            return cached
    if not fetch:
        return pd.DataFrame()

    url = f"{CBOE_BASE}/{SERIES[name]}"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        logger.warning(f"[cboe] {name} indirilemedi: {e}")
        stale = _cache_path(name)
        if stale.exists():                      # bayat da olsa veri > veri yok
            try:
                return pd.read_csv(stale, index_col=0, parse_dates=True)
            except Exception:
                pass
        return pd.DataFrame()

    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        logger.warning(f"[cboe] {name} beklenmeyen şema: {list(df.columns)}")
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    # VVIX/SKEW tek kolonludur; adı seri adıdır → 'close'a normalize et
    if "close" not in df.columns and name.lower() in df.columns:
        df = df.rename(columns={name.lower(): "close"})
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    CBOE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(_cache_path(name))
    except Exception as e:
        logger.debug(f"[cboe] cache yazılamadı {name}: {e}")
    return df


def get_all(names: list[str] | None = None, *, fetch: bool = True) -> pd.DataFrame:
    """Seçili serilerin close'larını tek geniş df'te birleştir (kolon = seri adı).

    `fetch=False` → önbellek-yalnız (bkz. `get_index`)."""
    names = names or ["VIX", "VIX9D", "VIX3M", "VVIX", "SKEW"]
    cols = {}
    for n in names:
        df = get_index(n, fetch=fetch)
        if df.empty or "close" not in df.columns:
            logger.debug(f"[cboe] {n} atlandı (boş/close yok)")
            continue
        cols[n] = df["close"]
    return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()


def term_structure(*, fetch: bool = True) -> pd.DataFrame:
    """Vade yapısı rejim özellikleri.

    ts_slope  = VIX3M / VIX  → >1 contango (sakin), <1 backwardation (stres)
    near_slope= VIX / VIX9D  → ön uç dikliği
    far_slope = VIX6M / VIX  → uzun uç (dış liste Madde 9 bunu AYRICA istiyordu)
    vvix_ratio= VVIX / 100   → vol'ün vol'ü, normalize

    **VIX6M ÜÇÜNCÜ SINIF VERİDİR ve öyle davranılır:** CBOE onu düzensiz yayımlar
    (ölçüldü 2026-09-06: VIX 2026-09-04'e kadar, VIX6M yalnız **2026-08-12**'ye kadar =
    23 gün geride). Kolon YOKSA `far_slope` HİÇ üretilmez ve `far_slope_bayat_gun`
    bayatlığı taşır — eksik uzun ucu son bilinen değerle doldurmak, üç haftalık bir
    vade yapısını "bugünkü" diye sunmak olurdu (BULGU-51-2).

    Backwardation ('ts_slope < 1') piyasa stresinin klasik göstergesidir ve
    survivorship içermez; bir rejim FİLTRESİ olarak sınanabilir — ama bu depoda
    henüz SINANMADI, dolayısıyla hiçbir karar yolundan okunmaz (ACI DERS #5).

    `fetch=False` → önbellek-yalnız; HTTP ucundan çağıran bunu kullanır (bkz. `get_index`).
    """
    wide = get_all(["VIX9D", "VIX", "VIX3M", "VIX6M", "VVIX", "SKEW"], fetch=fetch)
    if wide.empty or "VIX" not in wide.columns:
        return pd.DataFrame()
    out = pd.DataFrame(index=wide.index)
    if "VIX3M" in wide.columns:
        out["ts_slope"] = wide["VIX3M"] / wide["VIX"]
        # BULGU-79-2: `(NaN < 1).astype(int)` = 0 → iki uçtan biri o gün YAYIMLANMAMIŞKEN
        # (ölçüldü 2026-09-16: VIX3M 09-15'e, VIX 09-14'e kadar) satır "contango" diye
        # yazılıyordu. Ölçüm yokluğu rejim etiketi olamaz: ts_slope NaN ise bayrak da NaN.
        out["backwardation"] = (out["ts_slope"] < 1.0).astype(float).where(out["ts_slope"].notna())
    if "VIX9D" in wide.columns:
        out["near_slope"] = wide["VIX"] / wide["VIX9D"]
    if "VIX6M" in wide.columns:
        # Uzun uç: hizalı olduğu günlerde hesaplanır, ffill YOK (bkz. docstring).
        out["far_slope"] = wide["VIX6M"] / wide["VIX"]
    if "VVIX" in wide.columns:
        out["vvix_ratio"] = wide["VVIX"] / 100.0
    if "SKEW" in wide.columns:
        out["skew"] = wide["SKEW"]
    out["vix"] = wide["VIX"]
    # VIX yüzdelik dilimi: yalnız GEÇMİŞE bakan genişleyen pencere (look-ahead yasak)
    out["vix_pctile"] = wide["VIX"].expanding(min_periods=252).rank(pct=True) * 100
    return out.dropna(how="all")


def forward_variance(*, fetch: bool = True) -> pd.DataFrame:
    """VIX ailesinden İLERİ VOL eğrisi (Tur-79, yol haritası §7.5 "forward varyans eğrisi").

    Vadeler `VADE_GUN`dan (Cboe metodolojisi); matematik `options_math.forward_variance_curve`
    (TEK kaynak). Kolonlar `fwd_<kısa>_<uzun>` (yıllık ondalık ileri vol; ör. `fwd_VIX_VIX3M` =
    30. günden 93. güne kadar fiyatlanan oynaklık) + her segment için `neg_<...>` (negatif
    ileri varyans = o gün toplam varyans vadeyle AZALDI).

    AYNI GÜN ŞARTI: bir segment yalnız iki ucu da O GÜN yayımlanmışsa hesaplanır; ffill YOK
    (VIX6M düzensiz yayımlanır — Tur-74 `far_slope` dersi). `fetch=False` → önbellek-yalnız.
    Karar yoluna BAĞLANMAZ (ACI DERS #5); rapor yüzeyidir."""
    from nihai.ai.options_math import forward_variance_curve

    adlar = ["VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y"]
    wide = get_all(adlar, fetch=fetch)
    mevcut = [a for a in adlar if a in wide.columns]
    if len(mevcut) < 2:
        return pd.DataFrame()
    tenors = [VADE_GUN[a] / 365.0 for a in mevcut]
    egri = forward_variance_curve(tenors, wide[mevcut].to_numpy(dtype=float) / 100.0)
    if egri is None:
        return pd.DataFrame()
    out = pd.DataFrame(index=wide.index)
    for i in range(len(mevcut) - 1):
        ad = f"{mevcut[i]}_{mevcut[i + 1]}"
        out[f"fwd_{ad}"] = egri["fwd_vol"][:, i]
        # NaN segment (uçlardan biri o gün yok) negatif SAYILMAZ; kolon bool değil 0/1/NaN
        out[f"neg_{ad}"] = pd.Series(egri["negatif"][:, i].astype(float), index=wide.index).where(
            np.isfinite(egri["fwd_var"][:, i]))
    out = out.dropna(how="all")
    out.attrs["vade_gun"] = {a: VADE_GUN[a] for a in mevcut}   # dropna'dan SONRA (attrs düşmesin)
    return out


if __name__ == "__main__":  # pragma: no cover - canlı ağ gerektirir
    from nihai.console import ensure_utf8_stdout

    ensure_utf8_stdout()
    for n in SERIES:
        d = get_index(n)
        if d.empty:
            print(f"{n:6} BOŞ")
        else:
            print(f"{n:6} {len(d):>6} satır  {d.index[0].date()} → {d.index[-1].date()}")
    ts = term_structure()
    print(f"\nvade yapısı: {len(ts)} satır, kolonlar={list(ts.columns)}")
    print(ts.tail(3).to_string())
