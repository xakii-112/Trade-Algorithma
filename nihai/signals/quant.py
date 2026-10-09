"""quant.py — İSTATİSTİKSEL (quant) sinyal çekirdeği (Tur-97).

Sistemin kural tabanlı detektörleri (kırılım, RSI uyumsuzluğu, MACD, Donchian…) GÖSTERGE desenine
bakar. Tur-96 onları 14 piyasada aynı motorla ölçtü: 10 algoritmanın Sharpe medyanı 14 piyasanın
13'ünde NEGATİF. Bu modül karar yoluna literatürün iki sağlam quant yapısını koyar:

  1. ZAMAN SERİSİ MOMENTUMU (Moskowitz-Ooi-Pedersen 2012; çok ufuklu biçim Hurst-Ooi-Pedersen 2017):
     yön skoru = 21 / 63 / 252 GÜNLÜK log getirinin işaretlerinin ortalaması ∈ {−1, −⅓, +⅓, +1}.
  2. OYNAKLIK ÖLÇEKLEME: risk birimi EWMA(λ=0,94; RiskMetrics) ile öngörülen GÜNLÜK σ'dır. Stop 3σ
     uzağa konur; boyut mevcut risk-tabanlı motordan gelir (risk $ / stop mesafesi) → nominal ∝ 1/σ,
     yani oynaklık hedefleme ayrı bir kural olmadan motorun kendisinden çıkar.

Kurallar ÖN-KAYITLIDIR (Tur-97, 14 piyasa sonucuna bakılmadan önce sabitlendi): ufuklar, λ, 3σ ve
"uzun-yalnız varsayılan". Tek istisna uzun-yalnız kararıdır — vektörel keşif koşusundan SONRA
seçildi (kısa bacak hisse piyasalarında her yerde zarar yazdı, BULGU-96-9'un kural karşılığı) ve bu
yüzden motor koşusu 2009-2017 ÖRNEKLEM-DIŞI penceresinde ayrıca sınandı (docs/TUR97_RAPORU).

Nedensellik sözleşmesi: `seriler(close)[k][t]` yalnız close[0..t]'den hesaplanır (EWMA özyinelemesi
ve geriye bakan farklar). Karar t kapanışında, infaz t+1 açılışında — motorun kendi konvansiyonu.

Günlük bar sözleşmesi: ufuklar GÜN cinsindendir. Saatlik barla koşulursa 252 "gün" 39 günlük bir
şeye dönüşür ve sessizce başka bir strateji olurdu → `gunluk_mu` saatlik indeksi reddeder; canlı
tarama günlük barları ayrıca çekip `generate_signal(df_daily=…)` ile geçer.

Saf numpy/pandas: `strategy.py`'ye bağımlı DEĞİL (SignalAction'ı döndüren ince sarmalayıcı orada).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

UFUKLAR: tuple[int, ...] = (21, 63, 252)   # gün — ~1 ay / 1 çeyrek / 1 yıl
EWMA_LAMBDA = 0.94                          # RiskMetrics günlük
STOP_SIGMA = 3.0                            # stop mesafesi = 3 × günlük σ × fiyat
HEDEF_R = 3.0                               # NOMİNAL hedef = 3 × stop mesafesi (RR kapısı + pano)
MIN_BAR = max(UFUKLAR) + 1                  # skorun tanımlı olduğu ilk bar
KALITE_OLCEK = 6.5                          # TA kalite skorunun üst sınırı; pano aynı ölçekte okur
CANLI_GUNLUK_BAR = 300                      # canlı tarama ve pano detayı quant için AYNI pencereyi çeker (≥ MIN_BAR)


def seriler(close) -> tuple[np.ndarray, np.ndarray]:
    """(skor, sigma) — her bar için NEDENSEL dizi çifti.

    skor[t]  = ortalama_h sign(ln c[t] − ln c[t−h]); herhangi bir ufuk tanımsızsa NaN.
    sigma[t] = EWMA σ (günlük, kesir) — t'ye kadarki getirilerle, t+1 için öngörü."""
    c = np.asarray(close, dtype=float)
    n = c.size
    if n == 0:
        return np.full(0, np.nan), np.full(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        lc = np.where(c > 0, np.log(c), np.nan)
    lr = np.full(n, np.nan)
    lr[1:] = np.diff(lc)
    v = pd.Series(lr * lr).ewm(alpha=1.0 - EWMA_LAMBDA, adjust=False, ignore_na=True).mean().to_numpy()
    sigma = np.sqrt(v)
    isaret = np.full((len(UFUKLAR), n), np.nan)
    for k, h in enumerate(UFUKLAR):
        if n > h:
            isaret[k, h:] = np.sign(lc[h:] - lc[:-h])
    return isaret.mean(axis=0), sigma      # tek NaN ufuk → skor NaN (kısmi skor YOK)


def gunluk_mu(index) -> bool:
    """Bar aralığı günlük mü? Son 10 aralığın medyanı ≥ 20 saat. Tarih indeksi yoksa (sentetik
    test verisi) True — ölçek çağıranın sorumluluğundadır."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return True
    # Birimden bağımsız fark: pandas 3 indeksi ns dışı çözünürlükte (us/s) tutabilir → `asi8`
    # tamsayısı birime göre değişir ve sabit ns eşiği günlük barı "saatlik" sanardı (ölçüldü).
    son = index[-11:]
    return (son[1:] - son[:-1]).median() >= pd.Timedelta(hours=20)


def stop_hedef(price: float, yon: str, sigma: float) -> tuple[float, float]:
    """3σ stop, 3R nominal hedef. yon: "LONG" | "SHORT"."""
    d = STOP_SIGMA * float(sigma) * float(price)
    if yon == "LONG":
        return price - d, price + HEDEF_R * d
    return price + d, price - HEDEF_R * d


def kalite(skor: float) -> tuple[float, dict]:
    """Quant kalite = |skor| × 6,5 (TA skoruyla AYNI ölçek; pano karşılaştırabilsin). KAPI DEĞİL:
    quant girişini ufuk uzlaşması belirler, TA kalite eşiği uygulanmaz."""
    s = abs(float(skor))
    return round(KALITE_OLCEK * s, 2), {"quant_skor": round(float(skor), 4),
                                         "ufuk_uzlasmasi": round(s, 4)}


def hedef_agirlik(skor: float, sigma: float, hedef_sigma: float, tavan: float = 2.0) -> float:
    """Vektörel/pano kullanımı için oynaklık hedefli ağırlık: skor × min(hedef_σ/σ, tavan).
    σ ≤ 0 ya da sonlu değilse 0."""
    if not (math.isfinite(sigma) and sigma > 0 and math.isfinite(skor)):
        return 0.0
    return float(skor) * min(float(hedef_sigma) / float(sigma), float(tavan))


def durum(df: pd.DataFrame) -> dict:
    """Son barın quant özeti (pano / açıklama). Veri yetersizse `hazir=False`."""
    if df is None or len(df) < MIN_BAR or "close" not in df:
        return {"hazir": False, "neden": "YETERSIZ_GECMIS", "gerekli_bar": MIN_BAR}
    if not gunluk_mu(df.index):
        return {"hazir": False, "neden": "GUNLUK_BAR_GEREKLI"}
    c = df["close"].to_numpy(dtype=float)
    skor, sigma = seriler(c)
    s, sg = float(skor[-1]), float(sigma[-1])
    if not (math.isfinite(s) and math.isfinite(sg) and sg > 0):
        return {"hazir": False, "neden": "TANIMSIZ"}
    lc = np.log(c)
    return {"hazir": True, "skor": round(s, 4), "sigma_gunluk": round(sg, 6),
            "sigma_yillik": round(sg * math.sqrt(252), 4),
            "ufuk_getiri": {str(h): round(float(lc[-1] - lc[-1 - h]), 4) for h in UFUKLAR},
            "yon": "LONG" if s > 0 else ("SHORT" if s < 0 else "YOK")}
