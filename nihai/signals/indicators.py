"""
indicators.py — Ek teknik göstergeler + tek-bakışta konsensüs yön paneli.

strategy.py mevcut göstergeleri (EMA/RSI/ATR/ADX/Bollinger/MACD) üretir; bu modül
onları TAMAMLAR: hisse seçimi/teyidi için yaygın ama eksik göstergeler (VWAP,
Stochastic, StochRSI, OBV, CCI, Williams %R, MFI, ROC, SuperTrend, Keltner,
Parabolic SAR, CMF, Pivot) + her birinin bull/bear/neutral yorumu.

DÜRÜST ÇERÇEVE: Bunlar GÖSTERGE; "kesin tahmin" değil. `indicator_panel` tek bir
konsensüs skoru (-1..+1) üretir — bu yön EĞİLİMİdir, garanti değil. Sistem hâlâ
sinyal-only: bu modül emir TETİKLEMEZ, yalnızca analiz/teyit verisi üretir.

Tüm fonksiyonlar look-ahead-güvenli (yalnızca geçmiş+mevcut bar), Wilder yumuşatması
gereken yerlerde strategy._rma yeniden kullanılır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nihai.signals.strategy import _true_range, atr, bollinger_bands, ema, last_valid, macd, note3, rsi, vote


# ---------------------------------------------------------------------------
# Hacim / akış göstergeleri
# ---------------------------------------------------------------------------
def vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling VWAP (hacim-ağırlıklı ortalama fiyat). Tipik fiyat (H+L+C)/3 kullanır.
    Anchored session VWAP yerine pencere-tabanlı → her interval'de tutarlı."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    v = df["volume"].replace(0, np.nan)
    pv = (tp * v).rolling(period).sum()
    vol = v.rolling(period).sum()
    return pv / vol


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — fiyat yönüne göre kümülatif hacim (akış/birikim teyidi)."""
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"].fillna(0.0)).cumsum()


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — hacim-ağırlıklı RSI (0-100). >80 aşırı alım, <20 aşırı satım."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"].fillna(0.0)
    delta = tp.diff()
    pos = mf.where(delta > 0, 0.0).rolling(period).sum()
    neg = mf.where(delta < 0, 0.0).rolling(period).sum()
    mr = pos / neg.replace(0, np.nan)
    out = 100 - (100 / (1 + mr))
    # BULGU-33-2 — `rsi()`nin Tur-28'de kapatılan 0/0 kusurunun BİREBİR kardeşi, burada
    # açık kalmıştı: neg==0 iken 100 ("azami aşırı-alım") dönülüyor; ama pos DA 0 ise
    # (düz seri / sıfır hacim / seans-dışı) oran 0/0 BELİRSİZDİR ve doğru cevap NÖTR 50'dir.
    # 100 demek panelde koşulsuz BOĞA oyu demekti — yani sıfır bilgi +1'e çevriliyordu.
    duz = (neg == 0) & (pos == 0)                  # NaN ısınma penceresinde False kalır
    return (out.where(neg != 0, 100.0)
               .mask(duz, 50.0)
               .where(pos.notna(), np.nan))


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow — bar-içi kapanış konumu × hacim. +: birikim, -: dağıtım."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    mfv = mfm.fillna(0.0) * df["volume"].fillna(0.0)
    return mfv.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


# ---------------------------------------------------------------------------
# Momentum / osilatörler
# ---------------------------------------------------------------------------
def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator (%K, %D). >80 aşırı alım, <20 aşırı satım."""
    ll = df["low"].rolling(k).min()
    hh = df["high"].rolling(k).max()
    pk = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    return pk, pk.rolling(d).mean()


def stoch_rsi(series: pd.Series, period: int = 14, k: int = 3, d: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI — RSI'nin stochastic'i (daha hassas dönüş sinyali)."""
    r = rsi(series, period)
    ll = r.rolling(period).min()
    hh = r.rolling(period).max()
    sr = 100 * (r - ll) / (hh - ll).replace(0, np.nan)
    pk = sr.rolling(k).mean()
    return pk, pk.rolling(d).mean()


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index. >+100 güçlü yukarı, <-100 güçlü aşağı momentum."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R (-100..0). > -20 aşırı alım, < -80 aşırı satım."""
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change (%) — period bar önceye göre fiyat değişimi."""
    return (series / series.shift(period) - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Trend / kanal göstergeleri
# ---------------------------------------------------------------------------
def keltner_channels(df: pd.DataFrame, period: int = 20, mult: float = 2.0):
    """Keltner Channels — EMA orta + ATR bantları. (orta, üst, alt)."""
    mid = ema(df["close"], period)
    a = atr(df, period)
    return mid, mid + mult * a, mid - mult * a


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """SuperTrend — ATR-tabanlı trend takip çizgisi + yön (+1 yukarı, -1 aşağı).
    Döner: (supertrend_line, direction). Yaygın trend/stop referansı."""
    a = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    n = len(df)
    st = np.full(n, np.nan)
    # Yön de NaN ile başlar: ısınma penceresinde (ATR henüz yok) ve i=0'da eski kod 1.0
    # = "YUKARI TREND" bırakıyordu. `_last()` NaN'ları düşürdüğü için düz bir seride bu
    # uydurma ısınma değeri son geçerli okuma sanılıp panele BOĞA oyu olarak giriyordu.
    dir_ = np.full(n, np.nan)
    close = df["close"].to_numpy()
    ub = upper.to_numpy()
    lb = lower.to_numpy()
    f_ub = np.copy(ub)
    f_lb = np.copy(lb)
    # Tur-34: özyineleme (f_ub/f_lb/st her adımda bir öncekine bağlı) VEKTÖRLEŞTİRİLEMEZ,
    # ama döngü GÖVDESİ ucuzlatılabilir. Skaler `np.isnan(x)` çağrısı ufunc gönderimi
    # yapar; IEEE'de `x != x` AYNI testtir ve saf Python karşılaştırmasıdır. Ayrıca dizi
    # elemanları yerel değişkene alınır (tekrar indeksleme yok). Aritmetik DEĞİŞMEZ.
    for i in range(1, n):
        ub_i, lb_i = ub[i], lb[i]
        if ub_i != ub_i or lb_i != lb_i:
            continue
        prev_ub, prev_lb, prev_close = f_ub[i - 1], f_lb[i - 1], close[i - 1]
        # İlk geçerli bantta önceki "final band" NaN'dir → temel bantla seedle
        # (aksi halde üst-bant rekürsiyonu NaN'a saplanır ve çizgi hep NaN kalır).
        if prev_ub != prev_ub:
            f_ub[i] = cur_ub = ub_i
        else:
            f_ub[i] = cur_ub = ub_i if (ub_i < prev_ub or prev_close > prev_ub) else prev_ub
        if prev_lb != prev_lb:
            f_lb[i] = cur_lb = lb_i
        else:
            f_lb[i] = cur_lb = lb_i if (lb_i > prev_lb or prev_close < prev_lb) else prev_lb
        prev = st[i - 1]
        close_i = close[i]
        if prev != prev:
            dir_i = 1.0 if close_i > cur_ub else -1.0
        elif prev == prev_ub:
            dir_i = -1.0 if close_i <= cur_ub else 1.0
        else:
            dir_i = 1.0 if close_i >= cur_lb else -1.0
        dir_[i] = dir_i
        if cur_ub == cur_lb:
            # BULGU-33-1b: ATR=0 (tamamen DÜZ pencere: halt / illikit isim / seans-dışı)
            # üst ve alt bant ÇAKIŞIR → "fiyat hangi bandın dışında" sorusu TANIMSIZDIR.
            # Yukarıdaki dallar `close > f_ub`u yanlışlayıp −1 (AŞAĞI TREND) yazıyordu:
            # sıfır bilgiden trend YÖNÜ. NaN = "yön yok"; `vote()` bunu 0'a çevirir.
            # ATR>0 olan her veride bu dal HİÇ çalışmaz → mevcut ölçümlerde bit-denk.
            dir_[i] = dir_i = np.nan
        st[i] = cur_lb if dir_i > 0 else cur_ub
    return pd.Series(st, index=df.index), pd.Series(dir_, index=df.index)


def parabolic_sar(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Parabolic SAR — trend takip / dur-ve-tersine-dön noktaları."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=df.index)
    up = True
    af = af_step
    ep = high[0]
    sar[0] = low[0]
    for i in range(1, n):
        prev = sar[i - 1]
        cur = prev + af * (ep - prev)
        if up:
            cur = min(cur, low[i - 1], low[max(i - 2, 0)])
            if low[i] < cur:                       # trend döndü → aşağı
                up = False
                cur = ep
                ep = low[i]
                af = af_step
            elif high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)
        else:
            cur = max(cur, high[i - 1], high[max(i - 2, 0)])
            if high[i] > cur:                      # trend döndü → yukarı
                up = True
                cur = ep
                ep = high[i]
                af = af_step
            elif low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)
        sar[i] = cur
    return pd.Series(sar, index=df.index)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin-Ashi mumları — gürültüyü yumuşatan sentetik OHLC (trend okuması kolaylaşır).

    ha_close = (O+H+L+C)/4 ; ha_open = (önceki ha_open + önceki ha_close)/2
    (seed: ilk barda (O+C)/2) ; ha_high/low = gerçek H/L ile ha_open/close'un uç değeri.
    Look-ahead-güvenli: her bar yalnız kendi + önceki HA değerini kullanır.

    Döner: ha_open/ha_high/ha_low/ha_close kolonlu DataFrame (df ile aynı indeks).
    Yeşil mum (ha_close>ha_open) = boğa eğilimi, kırmızı = ayı; ardışık aynı-renk +
    iğnesiz mum güçlü trend işaretidir (yorum indicator_panel'de).
    """
    o, h, l, c = (df["open"].to_numpy(), df["high"].to_numpy(),
                  df["low"].to_numpy(), df["close"].to_numpy())
    n = len(df)
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.full(n, np.nan)
    if n:
        ha_open[0] = (o[0] + c[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])
    return pd.DataFrame(
        {"ha_open": ha_open, "ha_high": ha_high, "ha_low": ha_low, "ha_close": ha_close},
        index=df.index,
    )


def pivot_points(df: pd.DataFrame) -> dict:
    """Klasik pivot noktaları (son KAPANMIŞ bara göre): destek/direnç seviyeleri.
    'Olması gereken fiyat' referansı olarak hedef/stop seçimine yardımcı."""
    if len(df) < 2:
        return {}
    h, l, c = float(df["high"].iloc[-1]), float(df["low"].iloc[-1]), float(df["close"].iloc[-1])
    p = (h + l + c) / 3.0
    return {
        "pivot": round(p, 2),
        "r1": round(2 * p - l, 2), "s1": round(2 * p - h, 2),
        "r2": round(p + (h - l), 2), "s2": round(p - (h - l), 2),
        "r3": round(h + 2 * (p - l), 2), "s3": round(l - 2 * (h - p), 2),
    }


# ---------------------------------------------------------------------------
# İkinci parti — ek trend/momentum/akış göstergeleri
# ---------------------------------------------------------------------------
def dmi(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series]:
    """Directional Movement (DI+ / DI-). DI+ > DI- → yukarı baskın yön."""
    h, l = df["high"], df["low"]
    up_move = h.diff().fillna(0.0)
    dn_move = (-l.diff()).fillna(0.0)
    plus_dm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=df.index)
    tr = _true_range(df).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / tr
    return plus_di, minus_di


def ichimoku(df: pd.DataFrame) -> dict:
    """Ichimoku — Tenkan(9)/Kijun(26) + Senkou A/B bulutu (mevcut bara yansıtılmış)."""
    h, l = df["high"], df["low"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2.0
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2.0
    span_a = ((tenkan + kijun) / 2.0).shift(26)          # buluttaki "şimdi" değeri
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2.0).shift(26)
    return {"tenkan": tenkan, "kijun": kijun, "span_a": span_a, "span_b": span_b}


def aroon(df: pd.DataFrame, period: int = 25) -> tuple[pd.Series, pd.Series]:
    """Aroon Up/Down — son zirveye/dibe kaç bar olduğunu ölçer (trend tazeliği).

    Tur-33: `rolling.apply` yerine vektörel argmax/argmin. `argmax` TAM SAYI döndürür →
    kayan-nokta toplama sırası yoktur, sonuç BİREBİR aynıdır (yalnız Python geri-çağırımı
    kalkar). NaN içeren pencere: eski `apply` min_periods gereği NaN veriyordu; burada
    aynı maske elle uygulanır."""
    return _aroon_pair(df["high"], df["low"], period)


def _aroon_pair(high: pd.Series, low: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    p1 = period + 1
    hv, lv = high.to_numpy(dtype=float), low.to_numpy(dtype=float)
    up = np.full(hv.size, np.nan)
    dn = np.full(lv.size, np.nan)
    if hv.size >= p1:
        swv = np.lib.stride_tricks.sliding_window_view
        wh, wl = swv(hv, p1), swv(lv, p1)
        gecerli_h = np.isfinite(wh).all(axis=1)
        gecerli_l = np.isfinite(wl).all(axis=1)
        up_t, dn_t = up[period:], dn[period:]
        up_t[gecerli_h] = 100.0 * np.argmax(wh[gecerli_h], axis=1) / period
        dn_t[gecerli_l] = 100.0 * np.argmin(wl[gecerli_l], axis=1) / period
    return (pd.Series(up, index=high.index, name=high.name),
            pd.Series(dn, index=low.index, name=low.name))


def vortex(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series]:
    """Vortex Indicator (VI+ / VI-). VI+ > VI- → yukarı trend baskın."""
    tr = _true_range(df)
    vm_plus = (df["high"] - df["low"].shift()).abs()
    vm_minus = (df["low"] - df["high"].shift()).abs()
    tr_sum = tr.rolling(period).sum().replace(0, np.nan)
    return vm_plus.rolling(period).sum() / tr_sum, vm_minus.rolling(period).sum() / tr_sum


def awesome_oscillator(df: pd.DataFrame) -> pd.Series:
    """Awesome Oscillator — SMA(median,5) - SMA(median,34). >0 yukarı momentum."""
    med = (df["high"] + df["low"]) / 2.0
    return med.rolling(5).mean() - med.rolling(34).mean()


def trix(series: pd.Series, period: int = 15) -> pd.Series:
    """TRIX — üçlü-EMA'nın yüzde değişimi (gürültü-filtreli momentum)."""
    e1 = ema(series, period)
    e3 = ema(ema(e1, period), period)
    return e3.pct_change() * 100.0


def ultimate_oscillator(df: pd.DataFrame, s: int = 7, m: int = 14, lng: int = 28) -> pd.Series:
    """Ultimate Oscillator (0-100) — üç periyodu harmanlar. >50 boğa."""
    # Tur-35: `pd.concat(...).min/max(axis=1)` → `np.fmin/fmax`. Bu, Tur-6'nın
    # `strategy._true_range`'te kurduğu deseni gösterge modülüne taşır (ders MODÜLE değil
    # DESENE aitti). `fmax/fmin` NaN'ı yok sayar = pandas'ın skipna'sı → BİREBİR aynı
    # değerler. Ayrıca `min_lc` eskiden İKİ KEZ hesaplanıyordu (bp ve tr satırlarında aynı
    # ifade); şimdi bir kez. Ölçüldü: 4,84 → 0,55 ms (panelin %5,7'siydi).
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    prev_c = np.concatenate(([np.nan], c[:-1])) if len(c) else c
    min_lc = np.fmin(l, prev_c)
    bp = pd.Series(c - min_lc, index=df.index)
    tr = pd.Series(np.fmax(h, prev_c) - min_lc, index=df.index)
    tr = tr.replace(0, np.nan)
    a1 = bp.rolling(s).sum() / tr.rolling(s).sum()
    a2 = bp.rolling(m).sum() / tr.rolling(m).sum()
    a3 = bp.rolling(lng).sum() / tr.rolling(lng).sum()
    return 100 * (4 * a1 + 2 * a2 + a3) / 7.0


def chaikin_oscillator(df: pd.DataFrame) -> pd.Series:
    """Chaikin Oscillator — ADL'nin EMA(3)-EMA(10) farkı. >0 birikim ivmesi."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    adl = (mfm.fillna(0.0) * df["volume"].fillna(0.0)).cumsum()
    return ema(adl, 3) - ema(adl, 10)


def elder_ray(df: pd.DataFrame, period: int = 13) -> tuple[pd.Series, pd.Series]:
    """Elder Ray — bull power (high-EMA) / bear power (low-EMA). Alıcı/satıcı gücü."""
    e = ema(df["close"], period)
    return df["high"] - e, df["low"] - e


# ---------------------------------------------------------------------------
# Konsensüs paneli — tüm göstergeleri tek skora indirger
# ---------------------------------------------------------------------------
def _last(s: pd.Series) -> float:
    # Tur-43: gövde `strategy.last_valid`e indi (tek kaynak) — değer birebir aynı.
    return last_valid(s)


def _entry(name: str, value: float, signal: int, note: str) -> dict:
    """signal: +1 bullish, -1 bearish, 0 neutral/na."""
    safe = None if (value is None or (isinstance(value, float) and np.isnan(value))) else round(value, 4)
    return {"name": name, "value": safe, "signal": signal, "note": note}


# Üç durumlu oy/not kuralı TEK KAYNAKTA (strategy.py) tanımlıdır — indicators_ext de
# aynı yerden alır; ikinci bir kopya BULGU-33-1'in tekrar doğmasına açık kapı olurdu.
_note3 = note3

# Konsensüs etiketi için gereken en az YÖNLÜ gösterge sayısı. Panel normal veride ~65
# yönlü oy üretir; 10 bunun yaklaşık altıda biridir ve "ölçüm var ama zayıf" ile "ölçüm
# yok" arasındaki eşiktir. Altındaki bir oranın payda gürültüsü işaretini belirler.
_MIN_DIRECTIONAL = 10


def indicator_panel(df: pd.DataFrame, live_price: float | None = None) -> dict:
    """Tüm ek göstergeleri hesapla, her birine bull/bear/neutral yorumu ver, bir
    konsensüs yön skoru (-1..+1) ve etiket döndür. Detay sekmesi bunu listeler.

    Döner: {"rows": [...], "consensus_score": float, "consensus": str,
            "bull": int, "bear": int, "neutral": int, "pivots": {...}}"""
    rows: list[dict] = []
    if df is None or df.empty or len(df) < 30:
        return {"rows": rows, "consensus_score": 0.0, "consensus": "VERİ_YOK",
                "bull": 0, "bear": 0, "neutral": 0, "pivots": {}}

    c = df["close"]
    price = float(live_price) if (live_price and live_price > 0) else float(c.iloc[-1])

    # --- Trend
    st_line, st_dir = supertrend(df)
    st_d = _last(st_dir)
    st_sig = vote(st_d)
    rows.append(_entry("SuperTrend", _last(st_line), st_sig,
                       _note3(st_sig, "Yukarı trend", "Aşağı trend")))

    sar = _last(parabolic_sar(df))
    sar_sig = vote(price, sar)
    rows.append(_entry("Parabolic SAR", sar, sar_sig,
                       _note3(sar_sig, "Fiyat SAR üstünde (yükseliş)",
                              "Fiyat SAR altında (düşüş)")))

    e20, e50 = _last(ema(c, 20)), _last(ema(c, 50))
    ema_sig = vote(e20, e50)
    rows.append(_entry("EMA20 vs EMA50", e20 - e50, ema_sig,
                       _note3(ema_sig, "Hızlı EMA üstte (boğa)", "Hızlı EMA altta (ayı)")))

    ha = heikin_ashi(df)
    ha_o, ha_c = _last(ha["ha_open"]), _last(ha["ha_close"])
    ha_diff = ha_c - ha_o
    ha_sig = vote(ha_diff)
    ha_green = ha_diff > 0
    # Ardışık aynı-renk mum sayısı (son bardan geriye) → trend ısrarı
    body = (ha["ha_close"] - ha["ha_open"]).dropna()
    streak = 0
    for v in reversed(body.to_numpy()):
        if (v > 0) == ha_green and v != 0:
            streak += 1
        else:
            break
    rows.append(_entry("Heikin-Ashi", ha_diff, ha_sig,
                       _note3(ha_sig, f"Yeşil mum, {streak} bar ardışık (boğa eğilimi)",
                              f"Kırmızı mum, {streak} bar ardışık (ayı eğilimi)",
                              "Doji/düz mum — yön yok")))

    kmid, kup, klo = keltner_channels(df)
    kmid_v, kup_v, klo_v = _last(kmid), _last(kup), _last(klo)
    k_sig = vote(price, kmid_v)
    rows.append(_entry("Keltner orta", kmid_v, k_sig,
                       _note3(k_sig, "Fiyat orta-bant üstünde", "Fiyat orta-bant altında",
                              "Fiyat orta-bantta")
                       + f" (üst {kup_v:.2f} / alt {klo_v:.2f})"))

    # --- Momentum / osilatör (YÖN = momentum okuması; aşırı alım/satım NOTta uyarılır,
    #     yön oyunu trendi silmesin diye seviye değil eğim/orta-çizgi temel alınır)
    pk, pd_ = stochastic(df)
    pk_v, pd_v = _last(pk), _last(pd_)
    pk_sig = vote(pk_v, pd_v)
    rows.append(_entry("Stochastic %K", pk_v, pk_sig,
                       ("Aşırı satım <20" if pk_v < 20 else ("Aşırı alım >80" if pk_v > 80 else "Orta bölge"))
                       + _note3(pk_sig, " · %K>%D yukarı", " · %K<%D aşağı", " · %K=%D yatay")))

    srk = _last(stoch_rsi(c)[0])
    srk_sig = vote(srk, 50.0)
    rows.append(_entry("StochRSI", srk, srk_sig,
                       "Aşırı satım" if srk < 20 else ("Aşırı alım" if srk > 80 else
                       _note3(srk_sig, "Yukarı yarı", "Aşağı yarı", "Tam orta / hesaplanamadı"))))

    cci_v = _last(cci(df))
    cci_sig = vote(cci_v)
    rows.append(_entry("CCI(20)", cci_v, cci_sig,
                       "Güçlü yukarı >100" if cci_v > 100 else ("Güçlü aşağı <-100" if cci_v < -100 else
                       _note3(cci_sig, "Pozitif", "Negatif"))))

    wr = _last(williams_r(df))
    wr_sig = vote(wr, -50.0)
    rows.append(_entry("Williams %R", wr, wr_sig,
                       "Aşırı satım" if wr < -80 else ("Aşırı alım" if wr > -20 else
                       _note3(wr_sig, "Üst yarı", "Alt yarı", "Tam orta / hesaplanamadı"))))

    roc_v = _last(roc(c))
    roc_sig = vote(roc_v)
    rows.append(_entry("ROC(12) %", roc_v, roc_sig,
                       _note3(roc_sig, "Momentum pozitif", "Momentum negatif", "Momentum sıfır")))

    # --- Hacim / akış
    mfi_v = _last(mfi(df))
    mfi_sig = vote(mfi_v, 50.0)
    rows.append(_entry("MFI(14)", mfi_v, mfi_sig,
                       "Aşırı satım" if mfi_v < 20 else ("Aşırı alım" if mfi_v > 80 else
                       _note3(mfi_sig, "Para girişi", "Para çıkışı", "Net akış yok"))))

    cmf_v = _last(cmf(df))
    cmf_sig = vote(cmf_v)
    rows.append(_entry("CMF(20)", cmf_v, cmf_sig,
                       _note3(cmf_sig, "Birikim (+)", "Dağıtım (-)", "Net akış yok")))

    obv_s = obv(df)
    obv_slope = _last(obv_s) - (float(obv_s.iloc[-10]) if len(obv_s) >= 10 else _last(obv_s))
    obv_sig = vote(obv_slope)
    rows.append(_entry("OBV eğimi (10 bar)", obv_slope, obv_sig,
                       _note3(obv_sig, "Hacim akışı yukarı", "Hacim akışı aşağı", "Hacim akışı yatay")))

    vwap_v = _last(vwap(df))
    vwap_sig = vote(price, vwap_v)
    rows.append(_entry("VWAP(20)", vwap_v, vwap_sig,
                       _note3(vwap_sig, "Fiyat VWAP üstünde (alıcı baskın)",
                              "Fiyat VWAP altında (satıcı baskın)", "Fiyat VWAP'ta")))

    # --- RSI/Bollinger (mevcut göstergeler de panele dahil — tek bakış)
    r = _last(rsi(c, 14))
    r_sig = vote(r, 50.0)
    rows.append(_entry("RSI(14)", r, r_sig,
                       "Sağlıklı yükseliş 50-70" if 50 < r <= 70 else ("Aşırı alım >70" if r > 70 else
                       ("Aşırı satım <30" if r < 30 else
                        _note3(r_sig, "Zayıf yukarı", "Zayıf <50", "Tam 50 — yön yok")))))

    ma, _, _ = bollinger_bands(c)   # üst/alt bant burada kullanılmaz, yalnız orta-bant
    ma_v = _last(ma)
    bb_sig = vote(price, ma_v)
    rows.append(_entry("Bollinger orta", ma_v, bb_sig,
                       _note3(bb_sig, "Fiyat orta-bant üstünde", "Fiyat orta-bant altında",
                              "Fiyat orta-bantta")))

    # --- İkinci parti
    _, _, hist = macd(c)
    hist_v = _last(hist)
    macd_sig = vote(hist_v)
    rows.append(_entry("MACD histogram", hist_v, macd_sig,
                       _note3(macd_sig, "Pozitif (boğa momentum)", "Negatif (ayı momentum)",
                              "Sıfır histogram — yön yok")))

    pdi, mdi = dmi(df)
    pdi_v, mdi_v = _last(pdi), _last(mdi)
    dmi_sig = vote(pdi_v, mdi_v)
    rows.append(_entry("DMI (DI+ − DI−)", pdi_v - mdi_v, dmi_sig,
                       f"DI+ {pdi_v:.0f} {'>' if dmi_sig > 0 else ('<' if dmi_sig < 0 else '=')} "
                       f"DI- {mdi_v:.0f}"))

    ich = ichimoku(df)
    sa, sb = _last(ich["span_a"]), _last(ich["span_b"])
    cloud_top = max(sa, sb) if not (np.isnan(sa) or np.isnan(sb)) else float("nan")
    cloud_bot = min(sa, sb) if not (np.isnan(sa) or np.isnan(sb)) else float("nan")
    ich_sig = 1 if (not np.isnan(cloud_top) and price > cloud_top) else (-1 if (not np.isnan(cloud_bot) and price < cloud_bot) else 0)
    rows.append(_entry("Ichimoku bulut", cloud_top, ich_sig,
                       "Fiyat bulut üstünde (boğa)" if ich_sig > 0 else ("Fiyat bulut altında (ayı)" if ich_sig < 0 else "Bulut içinde (kararsız)")))

    ar_up, ar_dn = aroon(df)
    aru, ard = _last(ar_up), _last(ar_dn)
    ar_sig = vote(aru, ard)
    rows.append(_entry("Aroon (Up−Down)", aru - ard, ar_sig,
                       f"Up {aru:.0f} / Down {ard:.0f}"))

    vip, vim = vortex(df)
    vip_v, vim_v = _last(vip), _last(vim)
    vx_sig = vote(vip_v, vim_v)
    rows.append(_entry("Vortex (VI+ − VI−)", vip_v - vim_v, vx_sig,
                       f"VI+ {vip_v:.2f} {'>' if vx_sig > 0 else ('<' if vx_sig < 0 else '=')} "
                       f"VI- {vim_v:.2f}"))

    ao = _last(awesome_oscillator(df))
    ao_sig = vote(ao)
    rows.append(_entry("Awesome Osc.", ao, ao_sig,
                       _note3(ao_sig, "Pozitif momentum", "Negatif momentum", "Sıfır momentum")))

    trix_v = _last(trix(c))
    trix_sig = vote(trix_v)
    rows.append(_entry("TRIX(15)", trix_v, trix_sig,
                       _note3(trix_sig, "Yukarı eğim", "Aşağı eğim", "Yatay eğim")))

    uo = _last(ultimate_oscillator(df))
    uo_sig = vote(uo, 50.0)
    rows.append(_entry("Ultimate Osc.", uo, uo_sig,
                       _note3(uo_sig, "Boğa >50", "Ayı <50", "Tam 50 — yön yok")))

    co = _last(chaikin_oscillator(df))
    co_sig = vote(co)
    rows.append(_entry("Chaikin Osc.", co, co_sig,
                       _note3(co_sig, "Birikim ivmesi (+)", "Dağıtım ivmesi (-)", "İvme yok")))

    bull_p, bear_p = elder_ray(df)
    er = _last(bull_p) + _last(bear_p)
    er_sig = vote(er)
    rows.append(_entry("Elder Ray", er, er_sig,
                       _note3(er_sig, "Alıcı gücü baskın", "Satıcı gücü baskın", "Güçler dengede")))

    # Genişletilmiş gösterge kütüphanesi (MA ailesi, ek osilatörler, oynaklık, hacim).
    # Lazy import: indicators_ext bu modülden bir şey almaz ama döngüsel riski sıfırlar.
    try:
        from nihai.signals.indicators_ext import extended_rows
        rows.extend(extended_rows(df, price))
    except Exception:
        pass

    # NaN/hesaplanamayan gösterge (value None) yön OYU vermemeli — aksi halde "x>v else -1"
    # deseni yetersiz/düz veride sessizce AYI oyu üretir ve konsensüsü yanlış aşağı çeker.
    # Bu satırlar nötr sayılır (yön sayımına girmez).
    bull = sum(1 for x in rows if x["signal"] > 0 and x["value"] is not None)
    bear = sum(1 for x in rows if x["signal"] < 0 and x["value"] is not None)
    neutral = len(rows) - bull - bear
    # Konsensüs YALNIZCA yön-veren göstergelerden (bilgi/oynaklık satırları nötr=0,
    # konsensüsü sulandırmasın diye paydaya katılmaz).
    directional = bull + bear
    score = round((bull - bear) / directional, 3) if directional else 0.0
    if directional < _MIN_DIRECTIONAL:
        # Tur-23 dersinin panodaki karşılığı: bir ORAN, paydası küçükken okunamaz.
        # `vote` eşitliği/NaN'ı artık nötr saydığı için dejenere veride yönlü oy sayısı
        # meşru olarak düşebilir — 2 oydan 0 elde edilen skor +1,000 olur ve etiket
        # "GÜÇLÜ AL" derdi. Yön VEREN gösterge sayısı okunmadan skor okunmamalıdır.
        label = "YETERSİZ ÖLÇÜM"
    elif score >= 0.5:
        label = "GÜÇLÜ AL"
    elif score >= 0.15:
        label = "AL EĞİLİMİ"
    elif score <= -0.5:
        label = "GÜÇLÜ SAT"
    elif score <= -0.15:
        label = "SAT EĞİLİMİ"
    else:
        label = "NÖTR / KARARSIZ"

    return {"rows": rows, "consensus_score": score, "consensus": label,
            "bull": bull, "bear": bear, "neutral": neutral,
            "directional": directional,
            "pivots": pivot_points(df)}
