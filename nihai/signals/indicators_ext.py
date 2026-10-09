"""
indicators_ext.py — Genişletilmiş gösterge kütüphanesi (TradingView listesi kapsamı).

indicators.py'deki çekirdek seti TAMAMLAR: hareketli ortalama ailesi, ek osilatörler,
oynaklık ölçütleri, hacim/akış ve fiyat-dönüşümleri. `extended_rows(df, price)` her
göstergenin son değerini + yön oyunu (+1/−1) veya bilgi (0) olarak döndürür; panel
bunları tabloya ekler. Yön VEREN göstergeler konsensüse katılır, BİLGİ (oynaklık/
fiyat-dönüşümü) satırları katılmaz.

HARİÇ (gerekçeli): Implied Volatility (opsiyon verisi — options_analyzer'da),
TD Sequential (çok-barlı durum makinesi), Zig Zag (repaint — anlık değer üretmez).

DÜRÜST ÇERÇEVE: göstergedir, kesin tahmin değil. Sinyal-only; emir tetiklemez.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.signal import lfilter as _lfilter

from nihai.signals.strategy import _rma, atr, bollinger_bands, ema, last_valid, note3, vote


# ----------------------------------------------------------------- yardımcılar
def sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p).mean()


def _windows(s: pd.Series, p: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(pencere matrisi, çıktı tamponu) — `rolling(p).apply(..., raw=True)` yerine TEK
    vektörel adım. Python geri-çağırımı pencere başına yorumlayıcıya dönüyordu; profilde
    panel süresinin büyük kısmı buradaydı (Tur-33). Kısa seri → None."""
    v = s.to_numpy(dtype=float)
    if p < 1 or v.size < p:
        return None
    return np.lib.stride_tricks.sliding_window_view(v, p), np.full(v.size, np.nan)


def wma(s: pd.Series, p: int) -> pd.Series:
    w = np.arange(1, p + 1)
    built = _windows(s, p)
    if built is None:
        return pd.Series(np.full(len(s), np.nan), index=s.index, name=s.name)
    win, out = built
    out[p - 1:] = win @ w / w.sum()
    return pd.Series(out, index=s.index, name=s.name)


def smma(s: pd.Series, p: int) -> pd.Series:
    return _rma(s, p)


def dema(s: pd.Series, p: int) -> pd.Series:
    e = ema(s, p)
    return 2 * e - ema(e, p)


def tema(s: pd.Series, p: int) -> pd.Series:
    e1 = ema(s, p)
    e2 = ema(e1, p)
    return 3 * e1 - 3 * e2 + ema(e2, p)


def tma(s: pd.Series, p: int) -> pd.Series:
    return sma(sma(s, p), p)


def hma(s: pd.Series, p: int) -> pd.Series:
    return wma(2 * wma(s, p // 2) - wma(s, p), max(int(round(p ** 0.5)), 1))


def kama(s: pd.Series, p: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    change = (s - s.shift(p)).abs()
    vol = s.diff().abs().rolling(p).sum()
    er = (change / vol.replace(0, np.nan)).fillna(0.0)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    vals = s.to_numpy(dtype=float)
    scv = sc.to_numpy(dtype=float)
    k = vals[0]
    res = np.full(len(vals), np.nan)
    # Tur-34: uyarlanabilir düzgünleştirme ÖZYİNELEMELİDİR (k her adımda bir öncekine
    # bağlı) → vektörleştirilemez. Skaler `np.isnan(x)` yerine IEEE denkliği `x != x`
    # kullanılır (aynı test, ufunc gönderimi yok). Aritmetik DEĞİŞMEZ.
    for i in range(len(vals)):
        sc_i = scv[i]
        if sc_i != sc_i:
            res[i] = k
            continue
        k = k + sc_i * (vals[i] - k)
        res[i] = k
    return pd.Series(res, index=s.index)


def _linreg(s: pd.Series, p: int):
    """Rolling lineer regresyon → (endpoint/forecast, eğim).

    Tur-33: eski gövde İKİ ayrı `rolling.apply` geçişi yapıyordu ve `_end` içinde `_slope`ın
    tamamını yeniden hesaplıyordu — yani aynı regresyon pencere başına iki kez kuruluyordu.
    Ölçüldü: 5.000 barda 68,7 ms → 0,20 ms. Σ(x−x̄)(y−ȳ) = Σ(x−x̄)·y özdeşliği kullanılır
    (Σ(x−x̄)·ȳ = 0 olduğundan) → tek matris-vektör çarpımı."""
    x = np.arange(p)
    xm = float(x.mean())
    dev = x - xm
    denom = float((dev ** 2).sum())
    built = _windows(s, p)
    if built is None:
        bos = pd.Series(np.full(len(s), np.nan), index=s.index, name=s.name)
        return bos, bos.copy()
    win, end = built
    slope_out = np.full(len(s), np.nan)
    slope = (win @ dev) / denom
    slope_out[p - 1:] = slope
    end[p - 1:] = (win.mean(axis=1) - slope * xm) + slope * (p - 1)
    return (pd.Series(end, index=s.index, name=s.name),
            pd.Series(slope_out, index=s.index, name=s.name))


def _typical(df):
    return (df["high"] + df["low"] + df["close"]) / 3.0


def _median(df):
    return (df["high"] + df["low"]) / 2.0


def _last(s) -> float:
    # Tur-43: gövde `strategy.last_valid`e indi (tek kaynak) — değer birebir aynı.
    return last_valid(s)


def _slope_last(s, n: int = 10) -> float:
    s = pd.Series(s).dropna()
    if len(s) < n + 1:
        return 0.0
    return float(s.iloc[-1] - s.iloc[-n])


# ----------------------------------------------------------------- osilatörler
def cmo(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(p).sum()
    dn = (-d.clip(upper=0)).rolling(p).sum()
    return 100 * (up - dn) / (up + dn).replace(0, np.nan)


def demarker(df, p: int = 14) -> pd.Series:
    dh = df["high"].diff().clip(lower=0)
    dl = (-df["low"].diff()).clip(lower=0)
    return dh.rolling(p).sum() / (dh.rolling(p).sum() + dl.rolling(p).sum()).replace(0, np.nan)


def dpo(s: pd.Series, p: int = 20) -> pd.Series:
    return s - sma(s, p).shift(p // 2 + 1)


def ppo(s: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return (ema(s, fast) - ema(s, slow)) / ema(s, slow) * 100


def tsi(s: pd.Series, r: int = 25, sm: int = 13) -> pd.Series:
    m = s.diff()
    ds = ema(ema(m, r), sm)
    da = ema(ema(m.abs(), r), sm)
    return 100 * ds / da.replace(0, np.nan)


def rvi(df, p: int = 10) -> tuple[pd.Series, pd.Series]:
    """Relative Vigor Index + signal."""
    num = (df["close"] - df["open"])
    den = (df["high"] - df["low"])
    n = (num + 2 * num.shift() + 2 * num.shift(2) + num.shift(3)) / 6
    d = (den + 2 * den.shift() + 2 * den.shift(2) + den.shift(3)) / 6
    line = n.rolling(p).sum() / d.rolling(p).sum().replace(0, np.nan)
    sig = (line + 2 * line.shift() + 2 * line.shift(2) + line.shift(3)) / 6
    return line, sig


def rvgi_volatility(df, p: int = 10) -> pd.Series:
    """Relative Volatility Index — RSI'nin std üzerine uygulanışı."""
    std = df["close"].rolling(p).std(ddof=0)
    d = std.diff()
    up = _rma(d.clip(lower=0), p)
    dn = _rma((-d.clip(upper=0)), p)
    return 100 * up / (up + dn).replace(0, np.nan)


def imi(df, p: int = 14) -> pd.Series:
    gains = (df["close"] - df["open"]).clip(lower=0).rolling(p).sum()
    losses = (df["open"] - df["close"]).clip(lower=0).rolling(p).sum()
    return 100 * gains / (gains + losses).replace(0, np.nan)


def kairi(s: pd.Series, p: int = 20) -> pd.Series:
    m = sma(s, p)
    return (s - m) / m.replace(0, np.nan) * 100


def wavetrend(df, n1: int = 10, n2: int = 21) -> tuple[pd.Series, pd.Series]:
    tp = _typical(df)
    esa = ema(tp, n1)
    d = ema((tp - esa).abs(), n1)
    ci = (tp - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = ema(ci, n2)
    return wt1, sma(wt1, 4)


def schaff_trend_cycle(s: pd.Series, fast: int = 23, slow: int = 50, cyc: int = 10) -> pd.Series:
    m = ema(s, fast) - ema(s, slow)
    ll = m.rolling(cyc).min()
    k = 100 * (m - ll) / (m.rolling(cyc).max() - ll).replace(0, np.nan)
    d = ema(k, 3)
    ll2 = d.rolling(cyc).min()
    return 100 * (d - ll2) / (d.rolling(cyc).max() - ll2).replace(0, np.nan)


def cog(s: pd.Series, p: int = 10) -> pd.Series:
    """Center of Gravity osilatörü."""
    w = np.arange(p, 0, -1)                       # x[::-1]·(1..p) = x·(p..1)
    built = _windows(s, p)
    den = s.rolling(p).sum()
    if built is None:
        return pd.Series(np.full(len(s), np.nan), index=s.index, name=s.name)
    win, num = built
    num[p - 1:] = win @ w
    return -pd.Series(num, index=s.index, name=s.name) / den.replace(0, np.nan)


def force_index(df, p: int = 13) -> pd.Series:
    return ema(df["close"].diff() * df["volume"], p)


def adl(df) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    return (mfm.fillna(0.0) * df["volume"].fillna(0.0)).cumsum()


def williams_ad(df) -> pd.Series:
    # Tur-35: pd.concat(...).max/min(axis=1) → np.fmax/fmin (Tur-6 deseni; fmax NaN'ı yok
    # sayar = skipna → BİREBİR aynı değerler). Ölçüldü: 1,26 → 0,29 ms.
    c = df["close"].to_numpy(dtype=float)
    pc = np.concatenate(([np.nan], c[:-1])) if len(c) else c
    trh = np.fmax(df["high"].to_numpy(dtype=float), pc)
    trl = np.fmin(df["low"].to_numpy(dtype=float), pc)
    move = np.where(c > pc, c - trl, np.where(c < pc, c - trh, 0.0))
    return pd.Series(move, index=df.index).cumsum()


def pvt(df) -> pd.Series:
    return (df["close"].pct_change() * df["volume"]).cumsum()


def nvi(df) -> pd.Series:
    """Negatif Hacim Endeksi: hacim DÜŞTÜĞÜ barlarda getiriyle bileşiklenir, diğerlerinde
    sabit kalır.

    Tur-34 optimizasyonu: eski gövde bar başına `pct.iloc[i]` + `vol_dn.iloc[i]` çağırıyordu
    (pandas skaler indeksleme, çağrı başına ~µs mertebesi). Profilde `indicator_panel`in
    en pahalı TEK göstergesiydi (1500 barda 55 ms/çağrı; 12.080 `_getitem_axis`).

    Özyineleme bir ÇARPIM birikimidir → `np.cumprod` ile birebir aynı sıradadır:
    `res[i] = res[i−1]·f[i]`, `f[0] = 1000`. Soldan sağa çarpım sırası DEĞİŞMEDİĞİ için
    sonuç BİT-DENKTİR (Tur-33'ün `wma`/`_linreg` durumundan farkı budur — orada toplama
    sırası değişiyordu, burada değişmiyor)."""
    pct = df["close"].pct_change().fillna(0.0).to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)
    n = pct.size
    factors = np.ones(n, dtype=float)
    if n > 1:
        # i=0 döngüde HİÇ uygulanmıyordu (`i > 0` koruması) → ilk çarpan taban değerdir.
        dn = np.zeros(n, dtype=bool)
        dn[1:] = vol[1:] < vol[:-1]
        factors[dn] = 1.0 + pct[dn]
    factors[0] = 1000.0
    return pd.Series(np.cumprod(factors), index=df.index)


def swing_index(df, limit: float = 3.0) -> pd.Series:
    # Tur-35: iki `pd.concat(...).max(axis=1)` → `np.fmax` zinciri (Tur-6 deseni). `tr` zaten
    # `k`'nin bir adım genişletilmişiydi: max(a,b,|h−l|) = fmax(k, |h−l|) — eskiden aynı iki
    # terim İKİ KEZ birleştiriliyordu. Ölçüldü: 2,12 → 0,44 ms.
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    pc = np.concatenate(([np.nan], c[:-1])) if len(c) else c
    po = np.concatenate(([np.nan], o[:-1])) if len(o) else o
    k = np.fmax(np.abs(h - pc), np.abs(l - pc))
    tr = pd.Series(np.fmax(k, np.abs(h - l)), index=df.index)
    si = 50 * ((c - pc) + 0.5 * (c - o) + 0.25 * (pc - po)) / tr.replace(0, np.nan) * (k / limit)
    return si.fillna(0.0)


def mass_index(df, p: int = 25) -> pd.Series:
    rng = df["high"] - df["low"]
    e = ema(rng, 9)
    return (e / ema(e, 9)).rolling(p).sum()


def vhf(s: pd.Series, p: int = 28) -> pd.Series:
    hi = s.rolling(p).max()
    lo = s.rolling(p).min()
    return (hi - lo) / s.diff().abs().rolling(p).sum().replace(0, np.nan)


def spearman(s: pd.Series, p: int = 10) -> pd.Series:
    """Fiyatın ZAMAN sırasıyla Spearman sıra-korelasyonu (×100).

    BULGU-33-3: pencere TAMAMEN düzse her gözlem aynı ortalama sırayı ((n+1)/2) alır; d²
    sabit çıkar ve formül p=10'da **+50** üretir — yani sıfır bilgiden "orta güçlü YUKARI
    trend". Sıra varyansı yoksa katsayı TANIMSIZDIR → NaN.

    Tur-33 optimizasyonu: eski gövde `rolling(p).apply(..., raw=True)` içinde pencere
    BAŞINA bir `pd.Series` kurup `.rank()` çağırıyordu. Profilde `indicator_panel`
    süresinin **%25'i** buradaydı (300 barlık tek çağrıda 291 Series inşası). Şimdi tüm
    pencereler `sliding_window_view` ile TEK seferde sıralanıyor; bağ-ortalamalı sıralama
    (pandas `.rank()` varsayılanı) vektörel olarak birebir yeniden üretilir."""
    v = s.to_numpy(dtype=float)
    n_all = v.size
    out = np.full(n_all, np.nan, dtype=float)
    if p < 2 or n_all < p:
        return pd.Series(out, index=s.index, name=s.name)

    win = np.lib.stride_tricks.sliding_window_view(v, p)          # (n_all−p+1, p)
    # Geçerli pencere: NaN yok VE en az iki farklı değer (aksi halde sıra varyansı sıfır).
    ok = np.isfinite(win).all(axis=1) & (win != win[:, :1]).any(axis=1)
    if not np.any(ok):
        return pd.Series(out, index=s.index, name=s.name)
    w = win[ok]

    # BAĞ-ORTALAMALI sıra, sıralama YAPMADAN: bir elemanın ortalama sırası
    #     (kendisinden KÜÇÜK sayısı + kendisinden KÜÇÜK-EŞİT sayısı + 1) / 2
    # ile kapalı formdadır (bağsızda tam sıra, m bağda k+(m+1)/2 = bağların ortalaması).
    # pandas `.rank()` varsayılanının ("average") birebir karşılığı, satır başına
    # p×p karşılaştırma ile tek vektörel adımda.
    less = (w[:, None, :] < w[:, :, None]).sum(axis=2)
    leq = (w[:, None, :] <= w[:, :, None]).sum(axis=2)
    ranks = (less + leq + 1) / 2.0

    idx = np.arange(1.0, p + 1.0)
    d2 = ((ranks - idx) ** 2).sum(axis=1)
    rho = 1.0 - 6.0 * d2 / (p * (p * p - 1))
    tail = out[p - 1:]
    tail[ok] = rho * 100.0
    return pd.Series(out, index=s.index, name=s.name)


def alligator(df):
    med = _median(df)
    return _rma(med, 13).shift(8), _rma(med, 8).shift(5), _rma(med, 5).shift(3)


def williams_fractal(df) -> float:
    """Son tamamlanmış fraktal yönü: +1 up, -1 down, 0 yok (5-bar penceresi)."""
    if len(df) < 5:
        return 0.0
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    i = len(df) - 3
    if h[i] > h[i - 1] and h[i] > h[i - 2] and h[i] > h[i + 1] and h[i] > h[i + 2]:
        return 1.0
    if l[i] < l[i - 1] and l[i] < l[i - 2] and l[i] < l[i + 1] and l[i] < l[i + 2]:
        return -1.0
    return 0.0


def td_sequential(df) -> dict:
    """TD Sequential (DeMark) — Setup(9) + Countdown(13). DÖNÜŞ göstergesi:
    tamamlanan BUY setup/countdown = boğa dönüşü beklentisi, SELL = ayı dönüşü.
    Basitleştirilmiş (perfection/recycle yok); look-ahead-güvenli (geçmiş+mevcut bar).

    Döner: {setup_dir, setup_count(1-9), setup_complete, countdown_dir,
            countdown_count(0-13), vote(+1/−1/0), label}."""
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    n = len(c)
    empty = {"setup_dir": None, "setup_count": 0, "setup_complete": False,
             "countdown_dir": None, "countdown_count": 0, "vote": 0, "label": "veri yok"}
    if n < 6:
        return empty
    buy_s = sell_s = 0
    cd_dir = None
    cd_count = 0
    for i in range(4, n):
        # --- TD Setup: close vs 4 bar önceki close
        if c[i] < c[i - 4]:
            buy_s, sell_s = (buy_s + 1 if buy_s else 1), 0
        elif c[i] > c[i - 4]:
            sell_s, buy_s = (sell_s + 1 if sell_s else 1), 0
        else:
            buy_s = sell_s = 0
        # --- Setup tamamlanınca countdown uygunluğu (ters yön countdown'ı iptal eder)
        if buy_s == 9 and cd_dir != "buy":
            cd_dir, cd_count = "buy", 0
        elif sell_s == 9 and cd_dir != "sell":
            cd_dir, cd_count = "sell", 0
        # --- TD Countdown: buy → close ≤ low[2], sell → close ≥ high[2]
        if (cd_dir == "buy" and c[i] <= lo[i - 2]) or (cd_dir == "sell" and c[i] >= h[i - 2]):
            cd_count = min(cd_count + 1, 13)

    setup_dir = "buy" if buy_s else ("sell" if sell_s else None)
    setup_count = min(buy_s or sell_s, 9)
    setup_complete = (buy_s >= 9) or (sell_s >= 9)
    # Aktif yön: countdown ilerliyorsa o, yoksa mevcut setup yönü
    active = cd_dir if cd_count > 0 else setup_dir
    vote = 1 if active == "buy" else (-1 if active == "sell" else 0)
    if cd_count > 0:
        label = f"{'BUY' if cd_dir == 'buy' else 'SELL'} countdown {cd_count}/13"
    elif setup_dir:
        label = f"{'BUY' if setup_dir == 'buy' else 'SELL'} setup {setup_count}/9" + (" ✓TAMAM" if setup_complete else "")
    else:
        label = "nötr (sayım yok)"
    return {"setup_dir": setup_dir, "setup_count": setup_count, "setup_complete": setup_complete,
            "countdown_dir": cd_dir, "countdown_count": cd_count, "vote": vote, "label": label}


def td_sequential_series(df) -> list[dict]:
    """TD Sequential'ı BAR-BAZLI üret (grafik rakam-overlay için). Her anlamlı bar için
    {pos, kind('setup'/'cd'), dir('buy'/'sell'), num} döndürür. Look-ahead-güvenli."""
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    n = len(c)
    out: list[dict] = []
    if n < 6:
        return out
    buy_s = sell_s = 0
    cd_dir = None
    cd_count = 0
    for i in range(4, n):
        prev_cd = cd_count
        if c[i] < c[i - 4]:
            buy_s, sell_s = (buy_s + 1 if buy_s else 1), 0
        elif c[i] > c[i - 4]:
            sell_s, buy_s = (sell_s + 1 if sell_s else 1), 0
        else:
            buy_s = sell_s = 0
        if buy_s == 9 and cd_dir != "buy":
            cd_dir, cd_count = "buy", 0
        elif sell_s == 9 and cd_dir != "sell":
            cd_dir, cd_count = "sell", 0
        if (cd_dir == "buy" and c[i] <= lo[i - 2]) or (cd_dir == "sell" and c[i] >= h[i - 2]):
            cd_count = min(cd_count + 1, 13)
        # Bu bara yazılacak rakam: countdown bu barda arttıysa o; yoksa aktif setup (1-9)
        if cd_count != prev_cd and cd_count > 0:
            out.append({"pos": i, "kind": "cd", "dir": cd_dir, "num": cd_count})
        elif 0 < buy_s <= 9:
            out.append({"pos": i, "kind": "setup", "dir": "buy", "num": buy_s})
        elif 0 < sell_s <= 9:
            out.append({"pos": i, "kind": "setup", "dir": "sell", "num": sell_s})
    return out


def _aroon_osc(df, period: int = 25) -> pd.Series:
    # Aroon çekirdeği indicators.py'de TEK kaynaktır; buradaki kopya (aynı iki
    # rolling.apply) Tur-33'te kaldırıldı — iki uygulamanın ıraksama riski de gitti.
    from nihai.signals.indicators import _aroon_pair
    up, dn = _aroon_pair(df["high"], df["low"], period)
    return up - dn


# --------------------------------------------------------- volatilite/fiyat (bilgi)
def historical_volatility(s: pd.Series, p: int = 20) -> pd.Series:
    return np.log(s / s.shift()).rolling(p).std(ddof=0) * np.sqrt(252) * 100


def chaikin_volatility(df, p: int = 10) -> pd.Series:
    e = ema(df["high"] - df["low"], p)
    return (e - e.shift(p)) / e.shift(p).replace(0, np.nan) * 100


# ----------------------------------------------------------------- Tur-75 (Madde 112/113/114)
# Dış geliştirme listesinin BÖLÜM 6'sındaki üç Ehlers/Kaufman ailesi. KAMA (Madde 115) ve
# Vortex (Madde 116) ZATEN vardı; eksik olanlar bunlardı.
def fisher_transform(df, p: int = 10) -> tuple[pd.Series, pd.Series]:
    """Ehlers Fisher dönüşümü (Madde 112) → (fisher, tetik).

        x_t = 0,33·2·((med − min)/(max − min) − 0,5) + 0,67·x_{t−1}      (|x| < 1'e kırpılır)
        y_t = 0,5·ln((1+x)/(1−x)) + 0,5·y_{t−1}

    Fiyatın (yaklaşık) Gauss'a dönüştürülmesi uçları KESKİNLEŞTİRİR; ln'in tanım
    kümesi gereği x ±1'e ÇARPILMADAN kırpılır (±0,999) — kırpmasız bir uygulama
    dejenere pencerede ±inf üretir ve panel satırını sessizce düşürür.
    DÜZ pencerede (max == min) oran 0,5 kabul edilir, yani x KATKISIZDIR: sıfır bilgi
    sıfır sinyal verir (BULGU-33-1 disiplini).
    """
    med = (df["high"] + df["low"]) / 2.0
    hi = med.rolling(p).max()
    lo = med.rolling(p).min()
    span = (hi - lo)
    oran = np.where(span.to_numpy() > 0,
                    (med.to_numpy() - lo.to_numpy()) / np.where(span.to_numpy() > 0,
                                                                span.to_numpy(), 1.0),
                    0.5)
    n = len(med)
    x = np.zeros(n)
    y = np.zeros(n)
    onceki_x = 0.0
    onceki_y = 0.0
    for i in range(n):
        if not np.isfinite(oran[i]):
            x[i] = onceki_x
            y[i] = onceki_y
            continue
        xi = 0.66 * (oran[i] - 0.5) + 0.67 * onceki_x
        xi = float(np.clip(xi, -0.999, 0.999))
        yi = 0.5 * np.log((1.0 + xi) / (1.0 - xi)) + 0.5 * onceki_y
        x[i], y[i] = xi, yi
        onceki_x, onceki_y = xi, yi
    fis = pd.Series(y, index=med.index, name="fisher")
    return fis, fis.shift(1).rename("fisher_trigger")


def super_smoother(s: pd.Series, p: int = 10) -> pd.Series:
    """Ehlers 2-kutuplu Süper Yumuşatıcı (Madde 113) — Butterworth mantığı.

        a = e^{−√2·π/p} · b = 2a·cos(√2·π/p)
        c2 = b · c3 = −a² · c1 = 1 − c2 − c3
        y_t = c1·(x_t + x_{t−1})/2 + c2·y_{t−1} + c3·y_{t−2}

    Hareketli ortalamaların FAZ GECİKMESİNİ eklemeden yüksek frekansı süzer. Özyineli
    olduğu için vektörleştirilemez; `lfilter` ile kurulur (Tur-49 dersi: sıralı
    bağımlılık optimize EDİLEMEZ demek değildir).
    """
    per = max(2, int(p))
    a = math.exp(-math.sqrt(2.0) * math.pi / per)
    b = 2.0 * a * math.cos(math.sqrt(2.0) * math.pi / per)
    c2, c3 = b, -a * a
    c1 = 1.0 - c2 - c3
    x = s.to_numpy(dtype=float)
    # SEVİYE ÇIKARILIP GERİ EKLENİR. İki nedenle:
    #  (1) Sıfır durumdan başlayan bir IIR, sabit girdide bile ilk barlarda sıfırdan
    #      tırmanan SAHTE bir trend üretir (ölçüldü: sabit 42,0 serisinde 2. bar
    #      32,84 = 9,16 sapma). Sapma serisi 0'dan başladığı için o geçici yok olur.
    #  (2) DC kazancı float'ta TAM 1 değildir (c1 = 1 − c2 − c3 yuvarlanır). Sabit
    #      bir seride bu ~1e-14'lük bir artık bırakır ve `vote()` KESİN kıyas yaptığı
    #      için panelde HAYALET bir yön oyu doğar (BULGU-33-1: sıfır bilgi sıfır
    #      sinyal vermelidir; `tests/test_degenerate_panel.py` bunu yakaladı).
    #      Sapma tam 0 olduğunda filtre çıktısı da tam 0'dır → y = seviye, TAM.
    seviye = float(x[0])
    d = x - seviye
    girdi = np.empty_like(d)
    girdi[0] = d[0]
    girdi[1:] = (d[1:] + d[:-1]) / 2.0
    y = _lfilter([c1], [1.0, -c2, -c3], girdi) + seviye
    return pd.Series(y, index=s.index, name=f"ss{per}")


def frama(df, p: int = 16) -> pd.Series:
    """Fraktal Adaptif Hareketli Ortalama (Ehlers 2005 / Madde 114).

        D = [ln(N1 + N2) − ln(N3)] / ln 2 ,  N = (max−min)/uzunluk   (kutu sayma)
        α = e^{−4,6(D − 1)} , [0,01 ; 1] aralığına kırpılır
        FRAMA_t = α·p_t + (1 − α)·FRAMA_{t−1}

    Trendde D→1 (α→1, hızlı), gürültüde D→2 (α→0,01, neredeyse sabit). `p` ÇİFT
    olmalıdır (pencere ikiye bölünür); tek verilirse bir artırılır — sessizce
    yuvarlamak yarım pencereleri asimetrik yapardı.
    """
    per = max(4, int(p))
    if per % 2:
        per += 1
    yari = per // 2
    hi, lo = df["high"], df["low"]
    c = df["close"].to_numpy(dtype=float)
    n1 = ((hi.rolling(yari).max() - lo.rolling(yari).min()) / yari).to_numpy()
    n2 = ((hi.shift(yari).rolling(yari).max()
           - lo.shift(yari).rolling(yari).min()) / yari).to_numpy()
    n3 = ((hi.rolling(per).max() - lo.rolling(per).min()) / per).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (np.log(n1 + n2) - np.log(n3)) / math.log(2.0)
    alpha = np.exp(-4.6 * (d - 1.0))
    alpha = np.where(np.isfinite(alpha), np.clip(alpha, 0.01, 1.0), np.nan)
    out = np.empty(c.size)
    onceki = c[0]
    for i in range(c.size):
        a = alpha[i]
        onceki = c[i] if not np.isfinite(a) else a * c[i] + (1.0 - a) * onceki
        out[i] = onceki
    return pd.Series(out, index=df.index, name=f"frama{per}")


# ----------------------------------------------------------------- panel satırları
def _r(name, val, sig, note, rows):
    v = None if (val is None or (isinstance(val, float) and np.isnan(val))) else round(float(val), 4)
    rows.append({"name": name, "value": v, "signal": int(sig), "note": note})


def extended_rows(df: pd.DataFrame, price: float) -> list[dict]:
    """Genişletilmiş göstergeler → panel satırları. Her biri try/except ile sarılı;
    biri patlasa diğerleri etkilenmez. signal: +1/−1 yön oyu, 0 = bilgi (konsensüse girmez)."""
    rows: list[dict] = []
    c = df["close"]
    p = price

    def vs(name, series):
        v = _last(series)
        sig = vote(p, v)                       # eşitlik/NaN → 0 (BULGU-33-1)
        _r(name, v, sig, note3(sig, "fiyat MA üstünde (boğa)", "fiyat MA altında (ayı)",
                               "fiyat MA'ya eşit / MA yok"), rows)

    # --- Hareketli ortalama ailesi (fiyat vs MA = yön)
    for nm, fn, per in [("SMA(20)", sma, 20), ("WMA(20)", wma, 20), ("SMMA(20)", smma, 20),
                        ("DEMA(20)", dema, 20), ("TEMA(20)", tema, 20), ("TMA(20)", tma, 20),
                        ("HMA(20)", hma, 20), ("KAMA(10)", kama, 10),
                        ("SuperSmoother(10)", super_smoother, 10)]:
        try:
            vs(nm, fn(c, per))
        except Exception:
            pass

    # --- Osilatörler (yön). (name, değer-üretici, oy-kuralı, not) — lazy ki try sarsın.
    # Oy kuralı: `vote(v, eşik)` — eşitlik ve NaN 0'dır (BULGU-33-1). Eski `1 if v > x
    # else -1` biçimi düz seride 14 osilatörün TAMAMINI ayı oyuna çeviriyordu.
    osc_defs = [
        ("CMO(14)", lambda: _last(cmo(c)), lambda v: vote(v), "Chande momentum"),
        ("DeMarker", lambda: _last(demarker(df)), lambda v: vote(v, 0.5), ">0.5 boğa"),
        ("DPO(20)", lambda: _last(dpo(c)), lambda v: vote(v), "trend-arındırılmış"),
        ("PPO", lambda: _last(ppo(c)), lambda v: vote(v), "% MACD"),
        ("Price Osc.", lambda: _last(ema(c, 10) - ema(c, 21)), lambda v: vote(v), "kısa-uzun MA farkı"),
        ("Momentum(10)", lambda: _last(c - c.shift(10)), lambda v: vote(v), "fiyat ivmesi"),
        ("TSI", lambda: _last(tsi(c)), lambda v: vote(v), "true strength"),
        ("Smoothed ROC", lambda: _last(ema((c / c.shift(12) - 1) * 100, 5)), lambda v: vote(v), "yumuşatılmış ROC"),
        ("Kairi", lambda: _last(kairi(c)), lambda v: vote(v), "SMA'dan sapma %"),
        ("IMI(14)", lambda: _last(imi(df)), lambda v: vote(v, 50.0), "intraday momentum"),
        ("Rel. Vol. Index", lambda: _last(rvgi_volatility(df)), lambda v: vote(v, 50.0), "oynaklık RSI'si"),
        ("Schaff Trend Cyc.", lambda: _last(schaff_trend_cycle(c)), lambda v: vote(v, 50.0), "döngü"),
        ("Spearman", lambda: _last(spearman(c)), lambda v: vote(v), "sıra-korelasyon trend"),
        ("Aroon Osc.", lambda: _last(_aroon_osc(df)), lambda v: vote(v), "up−down"),
        ("Fisher Transform", lambda: _last(fisher_transform(df)[0]), lambda v: vote(v),
         "Gauss'a dönüştürülmüş uç"),
    ]
    for nm, valfn, rule, note in osc_defs:
        try:
            val = valfn()
            if not (isinstance(val, float) and np.isnan(val)):
                _r(nm, val, rule(val), note, rows)
        except Exception:
            pass

    # --- FRAMA: fiyat vs adaptif ortalama (MA ailesiyle aynı oy kuralı)
    try:
        vs("FRAMA(16)", frama(df))
    except Exception:
        pass

    # --- RVI / CoG / WaveTrend / Alligator (çift-çizgi → yön)
    try:
        line, sig = rvi(df)
        lv, sv = _last(line), _last(sig)
        rv_sig = vote(lv, sv)
        _r("Rel. Vigor Index", lv, rv_sig, note3(rv_sig, "RVI > signal", "RVI < signal",
                                                 "RVI = signal / hesaplanamadı"), rows)
    except Exception:
        pass
    try:
        cg = cog(c)
        cgv, cgs = _last(cg), _last(ema(cg, 3))
        cg_sig = vote(cgv, cgs)
        _r("Center of Gravity", cgv, cg_sig, note3(cg_sig, "CoG > signal", "CoG < signal",
                                                   "CoG = signal"), rows)
    except Exception:
        pass
    try:
        wt1, wt2 = wavetrend(df)
        w1, w2 = _last(wt1), _last(wt2)
        wt_sig = vote(w1, w2)
        _r("WaveTrend", w1, wt_sig, note3(wt_sig, "WT1 > WT2", "WT1 < WT2",
                                          "WT1 = WT2 / hesaplanamadı"), rows)
    except Exception:
        pass
    try:
        jaw, teeth, lips = alligator(df)
        jv, tv, lv = _last(jaw), _last(teeth), _last(lips)
        al = 1 if (lv > tv > jv) else (-1 if (lv < tv < jv) else 0)
        _r("Williams Alligator", lv, al, "lips/teeth/jaw dizilimi", rows)
        _r("Gator Osc.", abs(tv - jv), 0, "alligator yakınsama (bilgi)", rows)
    except Exception:
        pass

    # --- Hacim/akış (eğim → yön)
    flow_fns: list[tuple[str, Callable[..., pd.Series]]] = [
        ("Acc/Dist (ADL)", adl), ("Williams A/D", williams_ad), ("Price-Vol Trend", pvt),
        ("Elder Force Idx", force_index), ("Swing Index", swing_index),
        ("Acc. Swing Index", lambda d: swing_index(d).cumsum())]
    for nm, fn in flow_fns:
        try:
            s = fn(df)
            sl = _slope_last(s, 10)
            fl_sig = vote(sl)
            _r(nm, _last(s), fl_sig,
               "10-bar eğim " + note3(fl_sig, "yukarı", "aşağı", "yatay"), rows)
        except Exception:
            pass
    try:
        n = nvi(df)
        nv, ne = _last(n), _last(ema(n, 255))
        _r("Neg. Volume Idx", nv, vote(nv, ne), "akıllı para teyidi (vs EMA255)", rows)
    except Exception:
        pass

    # --- Lineer regresyon (yön)
    try:
        end, slope = _linreg(c, 20)
        sv, ev = _last(slope), _last(end)
        _r("LinReg Slope", sv, vote(sv), "regresyon eğimi", rows)
        _r("LinReg/TSF", ev, vote(p, ev), "fiyat vs regresyon ucu", rows)
        _r("Forecast Osc.", (p - ev) / p * 100 if p else 0.0, vote(p, ev), "tahmin sapması %", rows)
    except Exception:
        pass

    # --- %Bollinger / Donchian (yön) + bandwidth (bilgi)
    try:
        ma, up, lo = bollinger_bands(c)
        span = _last(up) - _last(lo)
        # span==0 (sıfır oynaklık) → %B TANIMSIZ. Eskiden 0.5 UYDURULUYOR ve `>=0.5`
        # kuralı bunu koşulsuz BOĞA oyuna çeviriyordu (BULGU-33-1'in aynı ailesi).
        pb = (p - _last(lo)) / span if span else float("nan")
        _r("%B (Bollinger)", pb, vote(pb, 0.5), "bant içi konum", rows)
        _r("BB Bandwidth", span / _last(ma) if _last(ma) else 0.0, 0, "bant genişliği (bilgi)", rows)
    except Exception:
        pass
    try:
        mid = (_last(df["high"].rolling(20).max()) + _last(df["low"].rolling(20).min())) / 2
        _r("Donchian/Price Ch.", mid, vote(p, mid), "kanal orta konumu", rows)
    except Exception:
        pass

    # --- TTM Squeeze (momentum yönü + sıkışma bilgisi)
    try:
        ma, up, lo = bollinger_bands(c)
        a = atr(df, 20)
        kc_up, kc_lo = ema(c, 20) + 1.5 * a, ema(c, 20) - 1.5 * a
        squeeze_on = (_last(up) < _last(kc_up)) and (_last(lo) > _last(kc_lo))
        mom = _last(c - sma(c, 20))
        ttm_sig = vote(mom)
        _r("TTM Squeeze", 1.0 if squeeze_on else 0.0, ttm_sig,
           ("SIKIŞMA AÇIK" if squeeze_on else "sıkışma yok")
           + " · momentum " + note3(ttm_sig, "+", "−", "0"), rows)
    except Exception:
        pass

    # --- Williams Fractal
    try:
        fr = williams_fractal(df)
        _r("Williams Fractal", fr, int(np.sign(fr)),
           "son fraktal " + ("zirve" if fr > 0 else ("dip" if fr < 0 else "yok")), rows)
    except Exception:
        pass

    # --- TD Sequential (DeMark): setup/countdown dönüş sayacı
    try:
        td = td_sequential(df)
        # Değer: işaretli sayaç (+ buy/boğa, − sell/ayı) — countdown varsa o, yoksa setup
        cnt = td["countdown_count"] if td["countdown_count"] > 0 else td["setup_count"]
        signed = cnt * (1 if td["vote"] >= 0 else -1)
        _r("TD Sequential", signed, td["vote"], td["label"], rows)
    except Exception:
        pass

    # --- ADXR (bilgi: trend gücü)
    try:
        from nihai.signals.strategy import adx as _adx
        ax = _adx(df)
        _r("ADXR", _last((ax + ax.shift(14)) / 2), 0, "trend gücü (yön değil)", rows)
    except Exception:
        pass

    # --- Oynaklık / fiyat-dönüşümü (BİLGİ — konsensüse girmez, signal=0)
    info_defs = [
        ("Historical Vol %", lambda: _last(historical_volatility(c)), "yıllık oynaklık"),
        ("Std Dev(20)", lambda: _last(c.rolling(20).std(ddof=0)), "standart sapma"),
        ("High-Low Vol %", lambda: _last((df["high"] - df["low"]) / c * 100), "bar aralığı %"),
        ("Chaikin Vol %", lambda: _last(chaikin_volatility(df)), "aralık değişimi"),
        ("Mass Index", lambda: _last(mass_index(df)), ">27 dönüş uyarısı"),
        ("VHF(28)", lambda: _last(vhf(c)), "trend/range gücü"),
        ("Typical Price", lambda: _last(_typical(df)), "(H+L+C)/3"),
        ("Median Price", lambda: _last(_median(df)), "(H+L)/2"),
        ("Weighted Close", lambda: _last((df["high"] + df["low"] + 2 * df["close"]) / 4), "(H+L+2C)/4"),
        ("Day Open-Close", lambda: _last(df["close"] - df["open"]), "bar gövdesi"),
        ("Percent Change %", lambda: _last(c.pct_change() * 100), "bar değişimi"),
        ("Volume Avg(20)", lambda: _last(df["volume"].rolling(20).mean()), "ortalama hacim"),
        ("Market Facil. Idx", lambda: _last((df["high"] - df["low"]) / df["volume"].replace(0, np.nan)), "BW MFI"),
    ]
    for nm, valfn, note in info_defs:
        try:
            val = valfn()
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                _r(nm, val, 0, note, rows)
        except Exception:
            pass

    return rows
