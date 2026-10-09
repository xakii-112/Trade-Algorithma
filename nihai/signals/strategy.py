"""
strategy.py — Saf sinyal motoru (event-based entry + kalite skoru).

PDF Referansları:
- Madde 1  : Olay-bazlı giriş (crossover/breakout anı)
- Madde 3  : Look-ahead koruması (data_provider tarafından da garantilenir; tekrar guard)
- Madde 5  : Granüler test — yön HTF, kontrol 1m
- Madde 7  : Re-entry cooldown
- Madde 8  : Minimum R:R
- Madde 11 : Zaman-bazlı stop
- Madde 16 : Piyasa rejimi (ADX) + IV rejimi
- Madde 17 : Volatilite patlaması (ATR)
- Madde 18 : Sinyal kalitesi puanlama
- Madde 19 : Karar gerekçeleri loglanır (signal_engine üzerinden)

ÖNEMLİ: Bu dosya EMİR TETİKLEMEZ. Yalnızca GIRIS / CIKIS / BEKLE üretir.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from enum import Enum

import numpy as np
import pandas as pd

from nihai.config import ACTIVE_STRATEGY, CONFIG
from nihai.signals import quant as _quant


class SignalAction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    WAIT = "WAIT"


@dataclass
class Signal:
    action: SignalAction
    symbol: str
    price: float
    stop: float
    target: float
    reasons: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    rr_ratio: float = 0.0
    metadata: dict = field(default_factory=dict)

    def is_actionable(self) -> bool:
        return self.action in (SignalAction.LONG, SignalAction.SHORT, SignalAction.EXIT)


# ---------------------------------------------------------------------------
# Teknik göstergeler — kendi ellerimizle (ta kütüphanesi yoksa da çalışsın).
#
# DOĞRULUK NOTU: RSI/ATR/ADX, Wilder yumuşatması (RMA) kullanır — bu göstergelerin
# evrensel standardı (TradingView/TA-Lib varsayılanı). Basit SMA varyantı, 30/70
# (RSI) ve 20/25 (ADX) eşiklerinin anlamını bozar. RMA = ewm(alpha=1/period,
# adjust=False): y_t = ((period-1)·y_{t-1} + x_t)/period — Wilder rekürsiyonunun ta kendisi.
# ---------------------------------------------------------------------------
def _true_range(df: pd.DataFrame) -> pd.Series:
    # numpy fmax zinciri = eski pd.concat(...).max(axis=1) ile BİREBİR aynı değerler
    # (fmax NaN'ı yok sayar = skipna; ilk barda shift-NaN → h-l düşer). concat+max
    # backtest sıcak yolunda ~%40 CPU yiyordu (bkz. OPTIMIZASYON_LOG Tur-6).
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    prev_c = np.concatenate(([np.nan], c[:-1])) if len(c) else c
    tr = np.fmax(h - l, np.fmax(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr, index=df.index)


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder yumuşatması (RMA). min_periods=period → ısınma boşluğu NaN kalır
    (yarı-oluşmuş göstergeden sinyal üretilmez)."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def vote(x: float, y: float = 0.0) -> int:
    """ÜÇ DURUMLU yön oyu: +1 (x>y) · −1 (x<y) · **0 (eşit VEYA hesaplanamaz)**.

    Tur-28 bu kuralı `_bias_and_confidence` içinde ELDE uygulamıştı ("eşitlik → 0.0,
    bilgi yok"); ders modül düzeyinde kaldığı için KARDEŞ ÇAĞIRANLAR açıkta kaldı.
    BULGU-33-1: `indicators.indicator_panel` + `indicators_ext.extended_rows` toplam 51
    yerde `1 if x > y else -1` yazıyordu — bu desen hem EŞİTLİĞİ hem NaN'ı −1 (ayı)
    sayar. Ölçüldü: tamamen DÜZ bir pencerede (halt / illikit isim / seans-dışı bar)
    panel SIFIR bilgiden **"GÜÇLÜ SAT", skor −0,542 (46 ayı ↔ 11 boğa)** üretiyordu.

    Kural tek yerde tanımlıdır ki üçüncü bir kardeş çağıran doğduğunda yeniden
    ıraksamasın: eşitlik "yön yok" demektir ("aşağı" değil); NaN "ölçülemedi" demektir
    ("aşağı" değil — Tur-31: ölçüm yokluğu ödüllendirilmez, burada CEZALANDIRILIYORDU).
    """
    try:
        a, b = float(x), float(y)
    except (TypeError, ValueError):
        return 0
    if not (math.isfinite(a) and math.isfinite(b)):
        return 0
    return 1 if a > b else (-1 if a < b else 0)


def last_valid(s) -> float:
    """Serinin/dizinin son NaN-OLMAYAN değeri; hiç yoksa NaN. **TEK KAYNAK.**

    BULGU-43-2 (Tur-43): bu yardımcının DÖRT birebir kopyası vardı
    (`indicators._last`, `indicators_ext._last`, `live_predict_service._last_valid`
    ve pano üzerinden aynısı) ve dördü de `s.dropna()` ile TÜM seriyi — yeni bir
    Series + yeni bir Index — kuruyordu. Sorulan tek şey KUYRUKTAKİ ilk geçerli
    değerdir; gösterge serilerinde son bar neredeyse her zaman geçerli olduğundan
    ilk yinelemede dönülür. Ölçüldü: 4000 barlık seride 0,1204 → 0,0059 ms (20,4×),
    dönen değer birebir aynı. Panel bu yardımcıyı satır başına en az bir kez çağırır.

    `dropna` NaN'ı atar ama ±inf'i ATMAZ → kontrol `isnan` iledir (`isfinite` DEĞİL);
    aksi halde sonsuz bir gösterge değeri sessizce atlanır ve eleme kuralı değişirdi.
    """
    a = np.asarray(s, dtype=float)
    if a.ndim != 1:
        a = a.ravel()
    for k in range(a.size - 1, -1, -1):
        v = a[k]
        if not np.isnan(v):
            return float(v)
    return float("nan")


def note3(sig: int, up: str, dn: str, flat: str = "Yön yok (eşit/hesaplanamadı)") -> str:
    """`vote` üç durumluysa NOT da üç durumlu olmalı — aksi halde oy 0 iken operatör
    "Fiyat MA altında (ayı)" okur ve panel kendi kendisiyle çelişir."""
    return up if sig > 0 else (dn if sig < 0 else flat)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # Tur-43: Wilder yumuşatması (`_rma` → ewm) pandas'ta KALIR; onun ETRAFINDAKİ
    # elemanwise aritmetik numpy'a indi. Eski gövde 5 `Series.where/mask/clip` + bir
    # `replace` çağırıyordu ve profilde bunlar sürenin **%43'ü** idi (blok yöneticisi
    # + `__finalize__` + hizalama yükü). İşlem SIRASI değişmedi, yalnız sarmalayıcı
    # kalktı → IEEE düzeyinde birebir (denklik gerçek 22.825 barlık seride ÖLÇÜLDÜ).
    idx = series.index
    x = series.to_numpy(dtype=float)
    d = np.empty(x.size, dtype=float)
    if x.size:
        d[0] = np.nan
        d[1:] = x[1:] - x[:-1]          # == series.diff()
    nan_d = np.isnan(d)
    # clip(lower=0).fillna(0) ≡ max(d,0) sonra NaN→0 · (-clip(upper=0)).fillna(0) ≡ max(−d,0)
    up = np.where(nan_d, 0.0, np.maximum(d, 0.0))
    dn = np.where(nan_d, 0.0, np.maximum(-d, 0.0))
    roll_up = _rma(pd.Series(up, index=idx), period).to_numpy(dtype=float)
    roll_dn = _rma(pd.Series(dn, index=idx), period).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = roll_up / np.where(roll_dn == 0.0, np.nan, roll_dn)
        out = 100 - (100 / (1 + rs))
    # roll_dn==0 (hiç KAYIP yok) → RS=∞ → RSI=100. AMA roll_up DA 0 ise (tamamen DÜZ
    # pencere: halt, illikit isim, seans-dışı bar) RS=0/0 BELİRSİZDİR ve doğru cevap
    # NÖTR 50'dir (Tur-28). 100 demek "azami aşırı-alım" demektir ve düz bir seriden
    # _bias_and_confidence'a +1.0 (azami boğa) bileşeni sokar, meta_features'ın rsi14/
    # rsi_slope'unu da kirletir. Ölçüldü: canlı özellik logunun 6374 satırının 0'ında
    # tetiklenmiş (30 likit sembol her barda hareket ediyor) → düzeltme mevcut veride
    # BİT-DENK, yalnız latent yolu kapatıyor.
    duz = (roll_dn == 0) & (roll_up == 0)          # NaN ısınma penceresinde False kalır
    out = np.where(roll_dn != 0, out, 100.0)       # .where(cond, 100.0)
    out = np.where(duz, 50.0, out)                 # .mask(duz, 50.0)
    out = np.where(np.isnan(roll_up), np.nan, out)  # .where(roll_up.notna(), nan)
    return pd.Series(out, index=idx, name=series.name)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _rma(_true_range(df), period)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # Tur-43: `rsi` ile aynı gerekçe — dört `_rma` (ewm) pandas'ta kalır, aradaki
    # DI/DX aritmetiği numpy'a iner. `.abs()`, `.replace(0, nan)` ve iki Seri bölmesi
    # blok yöneticisinden geçiyordu. İşlem sırası korunur → birebir aynı sayılar.
    idx = df.index
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    d_h = np.empty(h.size, dtype=float)
    d_l = np.empty(l.size, dtype=float)
    if h.size:
        d_h[0] = np.nan
        d_h[1:] = h[1:] - h[:-1]
        d_l[0] = np.nan
        d_l[1:] = -(l[1:] - l[:-1])
    up_move = np.where(np.isnan(d_h), 0.0, d_h)        # h.diff().fillna(0.0)
    dn_move = np.where(np.isnan(d_l), 0.0, d_l)        # (-l.diff()).fillna(0.0)
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    atr_v = _rma(_true_range(df), period).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100 * _rma(pd.Series(plus_dm, index=idx), period).to_numpy(dtype=float) / atr_v
        minus_di = 100 * _rma(pd.Series(minus_dm, index=idx), period).to_numpy(dtype=float) / atr_v
        den = plus_di + minus_di
        dx = 100 * np.abs(plus_di - minus_di) / np.where(den == 0, np.nan, den)
    return _rma(pd.Series(dx, index=idx), period)


# ---------------------------------------------------------------------------
# Tur-7 — Paylaşımlı gösterge önbelleği (davranış bit-bit aynı, yalnız tekrar-hesap yok)
#
# İki kullanım:
# 1) Tek dilim içinde: ensemble'ın 6 detektörü + generate_signal kapıları aynı RSI/ADX/
#    BB/EMA'yı ayrı ayrı hesaplıyordu → tembel memo ile İLK istekte bir kez hesaplanır.
# 2) Backtest'te semboller arası barlar boyunca: buradaki TÜM göstergeler NEDENSELDİR
#    (ewm adjust=False rekürsiyonu + rolling pencereler yalnız geçmişe bakar) → tam df
#    üzerinde BİR kez hesaplanan serinin ilk n bara kırpılmışı, df.iloc[:n] dilimi
#    üzerinde hesaplanmış seriyle BİT-BİT aynıdır (test_strategy_perf_parity doğrular).
#    backtest_symbol sembol başına bir _SharedIndicators kurar, her bara .upto(i+1)
#    görünümü verir → O(n²) gösterge maliyeti O(n)'e iner.
# ---------------------------------------------------------------------------
class _SharedIndicators:
    """df'in TAMAMI üzerinde tembel-memoize gösterge erişimcileri."""
    __slots__ = ("df", "_memo")

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._memo: dict = {}

    @property
    def nbars(self) -> int:
        """Görünümün kapsadığı bar sayısı — çağıranın hizalama kontrolü için (Tur-43).
        `meta_features.extract_features` bu sayıyı kendi diliminkiyle karşılaştırır;
        uyuşmazlık sessiz LOOK-AHEAD demek olurdu, orada önbellek yok sayılır."""
        return len(self.df)

    def _get(self, key, builder):
        v = self._memo.get(key)
        if v is None:
            v = self._memo[key] = builder()
        return v

    def ema(self, period: int) -> pd.Series:
        return self._get(("ema", period), lambda: ema(self.df["close"], period))

    def rsi14(self) -> pd.Series:
        return self._get("rsi14", lambda: rsi(self.df["close"], 14))

    def atr14(self) -> pd.Series:
        return self._get("atr14", lambda: atr(self.df, 14))

    def adx_series(self) -> pd.Series:
        return self._get("adx", lambda: adx(self.df))

    def adx_np(self) -> np.ndarray:
        """ADX dizisi — numpy (Tur-49). `adx_last` bundan TEK sayı okur."""
        return self._get("adx_np", lambda: self.adx_series().to_numpy(dtype=float))

    def adx_last(self) -> float:
        return self.adx_np()[-1]

    def bb20(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        return self._get("bb20", lambda: bollinger_bands(self.df["close"]))

    def bb_width20(self) -> pd.Series:
        def _w():
            ma, upper, lower = self.bb20()
            return (upper - lower) / ma.replace(0, np.nan)
        return self._get("bbw20", _w)

    def macd(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        return self._get("macd", lambda: macd(self.df["close"]))

    # BULGU-43-3 devamı: aşağıdaki üç erişimci SKALER okur — kırpma da, sütun çıkarma
    # da yok. `detect_event` bar başına iki `.iloc[:n]` görünümü kuruyor ve ondan yalnız
    # DÖRT sayı okuyordu; O(n) iş, O(1) bilgi için. Değerler aynı seriden gelir.
    def closes(self) -> np.ndarray:
        """Kapanış dizisi (bir kez numpy'a alınır) — skaler okumalar için."""
        return self._get("closes_np", lambda: self.df["close"].to_numpy(dtype=float))

    def ema_np(self, period: int) -> np.ndarray:
        """EMA(period) dizisi — numpy (Tur-49). Skaler okumalar buradan yapılır:
        `Series.iloc[int]` ÖLÇÜLDÜ 2,573 µs, `ndarray[int]` 0,126 µs (**20,4×**);
        `ema_at` backtest profilinde bar başına ~4 kez çağrılıyordu."""
        return self._get(("ema_np", period), lambda: self.ema(period).to_numpy(dtype=float))

    def ema_at(self, period: int, k: int = -1) -> float:
        """Görünümün SONUNDAN k. bardaki EMA (k=-1 son bar, k=-2 önceki)."""
        return float(self.ema_np(period)[self.nbars + k])

    def col_np(self, name: str) -> np.ndarray | None:
        """Ham OHLCV sütununun numpy hâli (Tur-49) — sütun YOKSA None.

        `closes()` ile aynı gerekçe: `meta_features.extract_features` bar başına
        high/low/open/volume sütunlarını `df[...]` ile ÇIKARIP `.iloc[]` ile birkaç
        skaler okuyordu; `DataFrame.__getitem__` + `Series.iloc` yolu ölçümde çağrı
        başına ~2,6 µs, numpy konumsal okuma 0,126 µs."""
        def _c():
            if name not in self.df.columns:
                return None
            return self.df[name].to_numpy(dtype=float)
        return self._get(("col_np", name), _c)

    def macd_np(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD üçlüsü — numpy (Tur-49). `detect_macd` bundan ÜÇ sayı okur; kırpılmış
        `_SharedSlice.macd()` ise bar başına ÜÇ Series+DatetimeIndex kuruyordu."""
        def _mn():
            m, sg, h = self.macd()
            return (m.to_numpy(dtype=float), sg.to_numpy(dtype=float), h.to_numpy(dtype=float))
        return self._get("macd_np", _mn)

    def bb20_np(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger üçlüsü — numpy (Tur-49); `detect_mean_reversion` İKİ sayı okur."""
        def _bn():
            ma, up, lo = self.bb20()
            return (ma.to_numpy(dtype=float), up.to_numpy(dtype=float), lo.to_numpy(dtype=float))
        return self._get("bb20_np", _bn)

    def bb_width20_np(self) -> np.ndarray:
        """Band genişliği — numpy (Tur-49); `detect_squeeze` iki PENCERE okur."""
        return self._get("bbw20_np", lambda: self.bb_width20().to_numpy(dtype=float))

    def rsi14_np(self) -> np.ndarray:
        """Görünümün RSI(14) dizisi — numpy, KIRPMA YOK (Tur-49, `atr14_np` ile aynı gerekçe).
        `detect_rsi_divergence` bundan yalnız son `lookback` barlık bir PENCERE okur."""
        return self._get("rsi14_np", lambda: self.rsi14().to_numpy(dtype=float))

    def atr14_np(self) -> np.ndarray:
        """Görünümün ATR(14) dizisi — numpy, **KIRPMA YOK** (Tur-45).

        `volatility_explosion_filter` ATR serisinden yalnız ÜÇ şey okur: son değer,
        son 20 barın ortalaması ve "hepsi NaN mi?". Üçü de konumla okunabilirken
        `_SharedSlice.atr14()` bar başına O(n) bir Series+DatetimeIndex kuruyordu
        (profilin %13'ü) ve filtrenin kendisi %19'du. Dizi ebeveynde bir kez kurulur;
        alt görünüm `[:n]` ile **sıfır-kopya numpy görünümü** alır → nedensellik
        yapısaldır: dilim geleceği İÇERMEZ, geçilecek bir `n` parametresi de yoktur."""
        return self._get("atr14_np", lambda: self.atr14().to_numpy(dtype=float))

    def close_at(self, k: int = -1) -> float:
        return float(self.closes()[self.nbars + k])

    def breakout_levels(self, lookback: int = 20) -> tuple[pd.Series, pd.Series]:
        """Karar barı HARİÇ önceki `lookback` barın (en yükseği, en düşüğü) serisi.

        `df["high"].iloc[-21:-1].max()` ile BİT-DENK: konum j'de rolling(20) penceresi
        j-19..j'dir, `.shift(1)` onu j+1'e taşır → j+1'de okunan değer j-19..j
        aralığının maksimumudur = karar barının ÖNCEKİ 20 barı. `min_periods=1`
        ısınmada kısmi pencereyi aynen `.max()` gibi kullanır; ikisi de NaN'ı atlar.
        NEDENSELLİK: tam df üzerinde hesaplanır ama j'de okunan değer j-1 ve
        öncesine bakar → dilim-üzerinde-hesapla ile aynı (Tur-7 argümanı)."""
        def _b():
            hi = self.df["high"].rolling(lookback, min_periods=1).max().shift(1)
            lo = self.df["low"].rolling(lookback, min_periods=1).min().shift(1)
            return hi, lo
        return self._get(("brk", lookback), _b)

    def breakout_np(self, lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """`breakout_levels` çıktısının numpy hâli (Tur-49) — skaler okuma için."""
        def _bn():
            hi, lo = self.breakout_levels(lookback)
            return hi.to_numpy(dtype=float), lo.to_numpy(dtype=float)
        return self._get(("brk_np", lookback), _bn)

    def breakout_at(self, k: int = -1, lookback: int = 20) -> tuple[float, float]:
        hi, lo = self.breakout_np(lookback)
        pos = self.nbars + k
        return float(hi[pos]), float(lo[pos])

    def quant_np(self) -> tuple[np.ndarray, np.ndarray]:
        """(skor, sigma) quant dizileri (Tur-97) — `quant.seriler` nedensel olduğu için tam df'te
        bir kez hesaplanıp dilime sıfır-kopya görünümle verilir (atr14_np ile aynı argüman)."""
        return self._get("quant_np", lambda: _quant.seriler(self.closes()))

    def upto(self, n: int) -> _SharedSlice:
        """İlk n bara kırpılmış görünüm (backtest bar döngüsü için)."""
        return _SharedSlice(self, n)


class _SharedSlice:
    """_SharedIndicators'ın ilk n bara kırpılmış görünümü — aynı arayüz.

    Nedensellik garantisi: dönen her seri, df.iloc[:n] üzerinde sıfırdan hesaplanmış
    olanla bit-bit aynıdır; detektörler yalnız kuyruk değerlerine bakar (look-ahead yok)."""
    __slots__ = ("_p", "_n", "_crops")

    def __init__(self, parent: _SharedIndicators, n: int):
        self._crops: dict = {}
        self._p, self._n = parent, n

    @property
    def nbars(self) -> int:
        return self._n

    # BULGU-43-3 (Tur-43): Tur-7 gösterge HESABINI bir kereye indirdi ama KIRPMAYI
    # indirmedi — `.iloc[:n]` her erişimde yeni bir Series + yeni bir DatetimeIndex
    # kurar (indeks dilimlemesi pandas'ta freq çıkarımını da tetikler). Aynı bar
    # içinde `ema(50)` üç, `rsi14` iki çağırana gidiyordu → bar başına ~9 kırpma,
    # her biri O(n) → yeniden O(n²). Ölçüldü: kırpma makinesi (`_get_slice_axis` +
    # `Series._slice` + `managers.get_slice`) 4000 barlık backtest profilinin
    # **%36'sı**. Kırpma bar başına anahtar başına BİR kez yapılır ve AYNI nesne
    # döner → değer karşılaştırması bile gereksiz, kimlik korunur.
    def _crop(self, key, build):
        v = self._crops.get(key)
        if v is None:
            v = self._crops[key] = build()
        return v

    def ema(self, period: int) -> pd.Series:
        return self._crop(("ema", period), lambda: self._p.ema(period).iloc[:self._n])

    def rsi14(self) -> pd.Series:
        return self._crop("rsi14", lambda: self._p.rsi14().iloc[:self._n])

    def atr14(self) -> pd.Series:
        return self._crop("atr14", lambda: self._p.atr14().iloc[:self._n])

    def adx_last(self) -> float:
        return self._p.adx_np()[self._n - 1]

    def bb20(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        def _b():
            ma, upper, lower = self._p.bb20()
            return ma.iloc[:self._n], upper.iloc[:self._n], lower.iloc[:self._n]
        return self._crop("bb20", _b)

    def bb_width20(self) -> pd.Series:
        return self._crop("bbw20", lambda: self._p.bb_width20().iloc[:self._n])

    def macd(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        def _m():
            m, s, h = self._p.macd()
            return m.iloc[:self._n], s.iloc[:self._n], h.iloc[:self._n]
        return self._crop("macd", _m)

    # Skaler erişimciler ebeveynin TAM serisinden okur; konum `_n + k` olduğu için
    # görünümün ötesine ASLA bakmaz (k negatif) — kırpma kurmaya gerek yok.
    def ema_at(self, period: int, k: int = -1) -> float:
        return float(self._p.ema_np(period)[self._n + k])

    def atr14_np(self) -> np.ndarray:
        return self._p.atr14_np()[:self._n]      # sıfır-kopya görünüm; geleceği içermez

    def rsi14_np(self) -> np.ndarray:
        return self._p.rsi14_np()[:self._n]      # sıfır-kopya görünüm; geleceği içermez

    def macd_np(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, sg, h = self._p.macd_np()
        return m[:self._n], sg[:self._n], h[:self._n]

    def bb20_np(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ma, up, lo = self._p.bb20_np()
        return ma[:self._n], up[:self._n], lo[:self._n]

    def bb_width20_np(self) -> np.ndarray:
        return self._p.bb_width20_np()[:self._n]

    def col_np(self, name: str) -> np.ndarray | None:
        a = self._p.col_np(name)
        return None if a is None else a[:self._n]

    def closes(self) -> np.ndarray:
        return self._p.closes()[:self._n]        # sıfır-kopya görünüm; geleceği içermez

    def quant_np(self) -> tuple[np.ndarray, np.ndarray]:
        s, sg = self._p.quant_np()
        return s[:self._n], sg[:self._n]         # sıfır-kopya görünüm; geleceği içermez

    def close_at(self, k: int = -1) -> float:
        return float(self._p.closes()[self._n + k])

    def breakout_at(self, k: int = -1, lookback: int = 20) -> tuple[float, float]:
        hi, lo = self._p.breakout_np(lookback)
        pos = self._n + k
        return float(hi[pos]), float(lo[pos])


# ---------------------------------------------------------------------------
# Olay tetikleyiciler — Madde 1
# ---------------------------------------------------------------------------
def detect_event(df: pd.DataFrame, shared=None) -> tuple[SignalAction, list[str]]:
    """EMA(20)/EMA(50) crossover + 20-bar breakout. Olay yalnızca KAPANMIŞ son barda.
    shared: opsiyonel _SharedIndicators/_SharedSlice — aynı değerler, tekrar-hesap yok (Tur-7)."""
    if len(df) < 60:
        return SignalAction.WAIT, ["INSUFFICIENT_BARS"]

    # Tur-43: bu gövde bar başına iki `.iloc[:n]` görünümü + iki sütun çıkarma +
    # iki `.iloc[-21:-1].max/min()` yapıyordu; okunan bilgi ise ALTI SAYI. Önbellek
    # yoksa dilimin kendisi üzerinde kurulur (aynı fonksiyonlar → aynı değerler).
    if shared is None:
        shared = _SharedIndicators(df)
    e20_now, e20_prev = shared.ema_at(20, -1), shared.ema_at(20, -2)
    e50_now, e50_prev = shared.ema_at(50, -1), shared.ema_at(50, -2)

    # Crossover ANI: önceki bar farkı ≤ 0, son bar > 0 → bullish cross.
    cross_up = e20_prev <= e50_prev and e20_now > e50_now
    cross_dn = e20_prev >= e50_prev and e20_now < e50_now

    # Breakout: son 20 bar yüksek/düşük kırılımı (karar barı hariç).
    hh20, ll20 = shared.breakout_at(-1, 20)
    last_close = shared.close_at(-1)
    breakout_up = last_close > hh20
    breakout_dn = last_close < ll20

    reasons = []
    if cross_up:
        reasons.append("EMA20/50_BULLISH_CROSS")
    if cross_dn:
        reasons.append("EMA20/50_BEARISH_CROSS")
    if breakout_up:
        reasons.append(f"BREAKOUT_HIGH_{hh20:.2f}")
    if breakout_dn:
        reasons.append(f"BREAKDOWN_LOW_{ll20:.2f}")

    if cross_up or breakout_up:
        return SignalAction.LONG, reasons
    if cross_dn or breakout_dn:
        return SignalAction.SHORT, reasons
    return SignalAction.WAIT, ["NO_EVENT"]


# ---------------------------------------------------------------------------
# Bollinger Bands — Mean Reversion için
# ---------------------------------------------------------------------------
def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    # Standart Bollinger: popülasyon std (ddof=0); pandas varsayılanı ddof=1'dir.
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std(ddof=0)
    return ma, ma + std_mult * sd, ma - std_mult * sd


# ---------------------------------------------------------------------------
# ALT STRATEJİ — Mean Reversion (sideways piyasalar)
# ---------------------------------------------------------------------------
def detect_mean_reversion(df: pd.DataFrame, shared=None) -> tuple[SignalAction, list[str]]:
    """RSI aşırı bölge + Bollinger Band kenarı dokunuş + zayıf trend (ADX<25).
    shared: opsiyonel paylaşımlı gösterge önbelleği (Tur-7) — değerler birebir aynı."""
    if len(df) < 30:
        return SignalAction.WAIT, ["INSUFFICIENT_BARS"]

    if shared is None:
        shared = _SharedIndicators(df)
    r = shared.rsi14_np()[-1]
    a = shared.adx_last() if len(df) >= 30 else float("nan")
    # mean-reversion yalnız üst/alt banta bakar
    _, upper, lower = shared.bb20_np()      # Tur-49: iki sayı için üç Series kırpma YOK
    price = shared.close_at(-1)

    # Mean reversion sadece zayıf trendde anlamlı
    if not np.isnan(a) and a > 25:
        return SignalAction.WAIT, [f"ADX_{a:.0f}_STRONG_TREND_NO_MR"]

    if np.isnan(r):
        return SignalAction.WAIT, ["RSI_NA"]

    # Bullish reversal: aşırı satım + lower BB dokunuş/altı
    if r < 30 and price <= lower[-1] * 1.002:
        return SignalAction.LONG, [
            f"MR_BULL_RSI_{r:.0f}", f"BB_LOWER_{lower[-1]:.2f}",
            f"ADX_{a:.0f}" if not np.isnan(a) else "ADX_NA",
        ]
    # Bearish reversal: aşırı alım + upper BB dokunuş/üstü
    if r > 70 and price >= upper[-1] * 0.998:
        return SignalAction.SHORT, [
            f"MR_BEAR_RSI_{r:.0f}", f"BB_UPPER_{upper[-1]:.2f}",
            f"ADX_{a:.0f}" if not np.isnan(a) else "ADX_NA",
        ]
    return SignalAction.WAIT, ["NO_MR_SETUP"]


# ---------------------------------------------------------------------------
# ALT STRATEJİ — RSI Divergence (trend reversal)
# ---------------------------------------------------------------------------
def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 30,
                          shared=None) -> tuple[SignalAction, list[str]]:
    """
    Klasik divergence:
    - Bullish: price LL + RSI HL → reversal up
    - Bearish: price HH + RSI LH → reversal down
    Pencerede iki yarıdaki ekstremler karşılaştırılır.
    shared: opsiyonel paylaşımlı gösterge önbelleği (Tur-7) — değerler birebir aynı.
    """
    if len(df) < lookback + 5:
        return SignalAction.WAIT, ["INSUFFICIENT_BARS"]

    # Tur-49: gövde bar başına ~12 pandas dilim/etiket-arama yapıyordu (iki `.iloc[-30:]`
    # görünümü + 8 yarı-pencere min/max/idx + 2 `.loc[etiket]`) ve okuduğu bilgi SEKİZ
    # SAYI idi. Backtest profilinde `.iloc` çağıranları arasında AÇIK ARA BİRİNCİYDİ
    # (8.000 barlık koşuda 74.102 çağrı = bar başına 9,3; ikincinin iki katı). Pencere
    # artık iki sıfır-kopya numpy görünümüdür; işlem sırası ve karşılaştırmalar AYNI.
    if shared is None:
        shared = _SharedIndicators(df)
    c = shared.closes()[-lookback:]
    r = shared.rsi14_np()[-lookback:]
    if np.isnan(r).any():
        return SignalAction.WAIT, ["RSI_NA"]

    half = lookback // 2
    c_a, c_b = c[:half], c[half:]
    # nan* aileleri pandas'ın skipna=True varsayılanını BİREBİR karşılar; RSI kapanış
    # NaN'ını YAYMAZ (up/dn orada 0'a düşer), yani c'de r'de olmayan NaN bulunabilir.
    # Yarı-pencerenin TAMAMI NaN ise pandas `.idxmin()` ValueError fırlatırdı — detektörün
    # barı çökertmesi yerine WAIT. Ölçüldü: 2.285.455 gerçek 1h barında NaN kapanış 0 →
    # mevcut veride BİT-DENK, yalnız latent çökme yolu kapanıyor.
    if np.isnan(c_a).all() or np.isnan(c_b).all():
        return SignalAction.WAIT, ["CLOSE_NA"]

    # Lows
    p_low_1 = float(np.nanmin(c_a))
    j_low_2 = int(np.nanargmin(c_b))          # ilk-eşleşme: pandas idxmin ile aynı
    p_low_2 = float(c_b[j_low_2])
    r_low_1 = float(np.nanmin(r[:half]))
    r_low_2 = float(r[half + j_low_2])        # aynı bar'daki RSI

    if (p_low_2 < p_low_1 and r_low_2 > r_low_1 + 2 and r_low_2 < 45):
        return SignalAction.LONG, [
            "RSI_BULL_DIVERGENCE",
            f"price_LL_{p_low_1:.2f}->{p_low_2:.2f}",
            f"RSI_HL_{r_low_1:.0f}->{r_low_2:.0f}",
        ]

    # Highs
    p_hi_1 = float(np.nanmax(c_a))
    j_hi_2 = int(np.nanargmax(c_b))
    p_hi_2 = float(c_b[j_hi_2])
    r_hi_1 = float(np.nanmax(r[:half]))
    r_hi_2 = float(r[half + j_hi_2])

    if (p_hi_2 > p_hi_1 and r_hi_2 < r_hi_1 - 2 and r_hi_2 > 55):
        return SignalAction.SHORT, [
            "RSI_BEAR_DIVERGENCE",
            f"price_HH_{p_hi_1:.2f}->{p_hi_2:.2f}",
            f"RSI_LH_{r_hi_1:.0f}->{r_hi_2:.0f}",
        ]

    return SignalAction.WAIT, ["NO_DIVERGENCE"]


# ---------------------------------------------------------------------------
# MACD — momentum crossover
# ---------------------------------------------------------------------------
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def detect_macd(df: pd.DataFrame, shared=None) -> tuple[SignalAction, list[str]]:
    """MACD histogramı sıfırı keser (signal çizgisi crossover) → momentum dönüşü.
    Olay yalnızca son KAPANMIŞ barda (önceki bar ≤0/≥0, son bar ters).
    shared: opsiyonel paylaşımlı gösterge önbelleği (Tur-7) — değerler birebir aynı."""
    if len(df) < 35:
        return SignalAction.WAIT, ["INSUFFICIENT_BARS"]
    # sinyal çizgisi burada kullanılmaz, histogram dönüşü esas
    if shared is None:
        shared = _SharedIndicators(df)
    m, _, hist = shared.macd_np()          # Tur-49: üç sayı için üç Series kırpma YOK
    h_prev, h_cur = hist[-2], hist[-1]
    if np.isnan(h_prev) or np.isnan(h_cur):
        return SignalAction.WAIT, ["MACD_NA"]
    if h_prev <= 0 and h_cur > 0:
        return SignalAction.LONG, ["MACD_BULL_CROSS", f"hist_{h_cur:+.3f}",
                                   f"macd_{m[-1]:+.3f}"]
    if h_prev >= 0 and h_cur < 0:
        return SignalAction.SHORT, ["MACD_BEAR_CROSS", f"hist_{h_cur:+.3f}",
                                    f"macd_{m[-1]:+.3f}"]
    return SignalAction.WAIT, ["NO_MACD_CROSS"]


# ---------------------------------------------------------------------------
# Donchian kanal kırılımı (turtle) — yalnızca trendde (ADX filtresi)
# ---------------------------------------------------------------------------
def detect_donchian(df: pd.DataFrame, period: int = 20, shared=None) -> tuple[SignalAction, list[str]]:
    """Son `period` barın (mevcut hariç) en yüksek/düşük seviyesi kırılırsa giriş.
    Whipsaw'a karşı yalnızca trend rejiminde (ADX ≥ eşik) işler.
    shared: opsiyonel paylaşımlı gösterge önbelleği (Tur-7) — değerler birebir aynı."""
    if len(df) < period + 5:
        return SignalAction.WAIT, ["INSUFFICIENT_BARS"]
    # Tur-49: bu üç okuma zaten `breakout_levels(period)`/`closes()` önbelleğinde VARDI
    # (`detect_event` onu kullanıyor). `.iloc[-period-1:-1].max()` ile
    # `rolling(period, min_periods=1).max().shift(1)` son konumu BİT-DENKTİR — pencere
    # tamdır (len(df) >= period+5 kapısı) ve ikisi de NaN atlar. Yeni hesap DEĞİL,
    # var olanın PAYLAŞILMASI.
    if shared is None:
        shared = _SharedIndicators(df)
    upper, lower = shared.breakout_at(-1, period)
    price = shared.close_at(-1)
    a = shared.adx_last()
    if not np.isnan(a) and a < CONFIG.strategy.adx_trend_threshold:
        return SignalAction.WAIT, [f"ADX_{a:.0f}_WEAK_NO_DONCHIAN"]
    a_str = f"ADX_{a:.0f}" if not np.isnan(a) else "ADX_NA"
    if price > upper:
        return SignalAction.LONG, [f"DONCHIAN_BREAK_HIGH_{upper:.2f}", a_str]
    if price < lower:
        return SignalAction.SHORT, [f"DONCHIAN_BREAK_LOW_{lower:.2f}", a_str]
    return SignalAction.WAIT, ["NO_DONCHIAN_BREAK"]


# ---------------------------------------------------------------------------
# Bollinger Squeeze breakout — volatilite sıkışması sonrası patlama
# (mean_reversion'ın TERSİ: band kenarını fade etmez, kırılımı KOVALAR)
# ---------------------------------------------------------------------------
def detect_squeeze(df: pd.DataFrame, period: int = 20, lookback: int = 50,
                   squeeze_ratio: float = 0.6, calm_bars: int = 10,
                   shared=None) -> tuple[SignalAction, list[str]]:
    """Volatilite sıkışması sonrası kanal kırılımı.

    Sıkışma = son `calm_bars` barın band genişliği (kırılım barı hariç) HEPSI,
    `lookback` medyanının `squeeze_ratio` katının altında (kalıcı, rejim-göreli düşük
    volatilite). Kırılım, kendini-şişiren bant yerine önceki `period` kanalından ölçülür.
    shared: opsiyonel paylaşımlı gösterge önbelleği (Tur-7; yalnız period=20 varsayılanında
    kullanılır — farklı period isteyen çağıran kendi bandını hesaplar, davranış aynı).
    """
    if len(df) < lookback + period:
        return SignalAction.WAIT, ["INSUFFICIENT_BARS"]
    if shared is None:
        shared = _SharedIndicators(df)
    if period == 20:
        width = shared.bb_width20_np()
    else:
        ma, upper, lower = bollinger_bands(df["close"], period)
        width = ((upper - lower) / ma.replace(0, np.nan)).to_numpy(dtype=float)
    recent = width[-(calm_bars + 1):-1]                 # son calm_bars bar, mevcut hariç
    if np.isnan(recent).any():
        return SignalAction.WAIT, ["BB_NA"]
    baseline = float(np.nanmedian(width[-lookback:]))   # pandas .median() = skipna
    recent_max = float(np.max(recent))                  # recent'te NaN YOK (üstteki kapı)
    if not (baseline > 0 and recent_max < squeeze_ratio * baseline):
        return SignalAction.WAIT, ["NO_SQUEEZE"]
    prior_high, prior_low = shared.breakout_at(-1, period)   # Tur-49: paylaşılan önbellek
    price = shared.close_at(-1)
    ratio = recent_max / baseline
    if price > prior_high:
        return SignalAction.LONG, [f"SQUEEZE_BREAK_UP_{prior_high:.2f}", f"vol_{ratio:.2f}x_baseline"]
    if price < prior_low:
        return SignalAction.SHORT, [f"SQUEEZE_BREAK_DN_{prior_low:.2f}", f"vol_{ratio:.2f}x_baseline"]
    return SignalAction.WAIT, ["SQUEEZE_NO_BREAK"]


# ---------------------------------------------------------------------------
# YENİ ADAY — Trend Pullback (yükseliş trendinde DÜŞÜŞÜ al, breakout'u KOVALAMA)
# ---------------------------------------------------------------------------
# Gerekçe (docs/EDGE_FINDINGS.md kök-neden): 1h mega-cap'te EMA50 üstündeki "hizalı"
# breakout girişleri geç/yorgun (tepeyi alır). Pullback girişi bunun tersi: trend
# SAĞLAMKEN, fiyat hızlı-EMA'ya geri çekilip yeniden ele geçirdiğinde gir. Düşük stop
# mesafesi + daha az uzamış giriş. Trend rejimi uzun EMA (varsayılan 100) ile teyit.
# ---------------------------------------------------------------------------
def detect_trend_pullback(df: pd.DataFrame, fast: int = 20,
                          regime: int = 100, shared=None) -> tuple[SignalAction, list[str]]:
    if len(df) < regime + 5:
        return SignalAction.WAIT, ["INSUFFICIENT_BARS"]
    c = df["close"]
    e_fast = shared.ema(fast) if shared is not None else ema(c, fast)
    e_reg = shared.ema(regime) if shared is not None else ema(c, regime)
    price = float(c.iloc[-1])
    prev = float(c.iloc[-2])
    # Trend rejimi: fiyat VE hızlı-EMA, rejim-EMA'sının aynı tarafında.
    up = price > e_reg.iloc[-1] and e_fast.iloc[-1] > e_reg.iloc[-1]
    dn = price < e_reg.iloc[-1] and e_fast.iloc[-1] < e_reg.iloc[-1]
    # Geri-çekilme + yeniden ele geçirme: önceki bar hızlı-EMA'nın ters tarafında,
    # son bar geri döndü (olay = bu yeniden-ele-geçirme barı).
    reclaim_up = prev <= e_fast.iloc[-2] and price > e_fast.iloc[-1]
    reclaim_dn = prev >= e_fast.iloc[-2] and price < e_fast.iloc[-1]
    if up and reclaim_up:
        return SignalAction.LONG, [f"PULLBACK_RECLAIM_EMA{fast}", f"UPTREND_EMA{regime}"]
    if dn and reclaim_dn:
        return SignalAction.SHORT, [f"PULLBACK_RECLAIM_EMA{fast}", f"DOWNTREND_EMA{regime}"]
    return SignalAction.WAIT, ["NO_PULLBACK"]


# NOT (ampirik, docs/EDGE_FINDINGS.md §5): trend_pullback'in YANINDA üç aday daha
# ölçülüp ELENDİ — tsmom (time-series momentum), ker_breakout (Kaufman efficiency)
# ve hybrid (katmanlı AND birleşimi). Hiçbiri iki periyotta dayanıklı pozitif değildi;
# hybrid parçalarından KÖTÜ çıktı ("daha çok birleştirme = daha kötü"yü 3. kez doğruladı).
# Yeniden denemek için git geçmişine bak (bu commit'ten önce). trend_pullback en iyi
# ADAY olarak kaldı (en yüksek WR) ama çıkış optimizasyonu walk-forward'da çürüdü.


# ---------------------------------------------------------------------------
# ALTERNATİF VERİ BOYUTU — Hacim-teyitli kırılım (RVOL)
# ---------------------------------------------------------------------------
# Mevcut sinyaller fiyatı kullanıyordu; bu HACMİ birincil tetik yapar. Tez: hacim
# patlamasıyla (RVOL = anlık/ortalama) teyitli kırılım = kurumsal konvansiyon; hacimsiz
# kırılım = gürültü/sahte. event_breakout/donchian'dan farkı: hacim ZORUNLU kapı.
# Olay yalnızca son KAPANMIŞ barda; look-ahead yok (ortalama mevcut bar hariç).
# ---------------------------------------------------------------------------
def detect_vol_breakout(df: pd.DataFrame, lookback: int = 20,
                        rvol_mult: float = 1.5, shared=None) -> tuple[SignalAction, list[str]]:
    # shared: imza tekdüzeliği için kabul edilir (Tur-7); bu detektör yalnız hacim +
    # kuyruk high/low kullanır, paylaşılabilir gösterge yok — davranış değişmez.
    if len(df) < lookback + 5:
        return SignalAction.WAIT, ["INSUFFICIENT_BARS"]
    if "volume" not in df.columns:
        return SignalAction.WAIT, ["NO_VOLUME_DATA"]
    c, v = df["close"], df["volume"]
    avg_v = float(v.iloc[-lookback - 1:-1].mean())     # önceki ortalama, mevcut bar hariç
    if avg_v <= 0:
        return SignalAction.WAIT, ["VOLUME_NA"]
    rvol = float(v.iloc[-1]) / avg_v
    if rvol < rvol_mult:                                # konvansiyon yok → pas
        return SignalAction.WAIT, [f"RVOL_{rvol:.2f}_NO_CONVICTION"]
    upper = float(df["high"].iloc[-lookback - 1:-1].max())
    lower = float(df["low"].iloc[-lookback - 1:-1].min())
    price = float(c.iloc[-1])
    if price > upper:
        return SignalAction.LONG, [f"VOL_BREAKOUT_HIGH_{upper:.2f}", f"RVOL_{rvol:.2f}x"]
    if price < lower:
        return SignalAction.SHORT, [f"VOL_BREAKOUT_LOW_{lower:.2f}", f"RVOL_{rvol:.2f}x"]
    return SignalAction.WAIT, ["NO_VOL_BREAKOUT"]


# ---------------------------------------------------------------------------
# ALTERNATİF SİNYAL TİPİ — PEAD (Post-Earnings Announcement Drift)
# ---------------------------------------------------------------------------
# Fiyat-göstergesi DEĞİL; temel/olay-bazlı. Belgelenmiş anomali: bilanço sürprizi
# yönünde fiyat günler-haftalar boyunca "sürüklenir" (analistler/piyasa yavaş tepki
# verir). Giriş = earnings'in ERTESİ barı, sürpriz yönünde; çıkış = standart ATR
# stop/target + zaman-stop drift penceresini yönetir. Yalnızca GÜNLÜK bar anlamlı.
#
# Look-ahead güvenli: önceki barın earnings'i + bildirilmiş Surprise% o bar kapanışında
# bilinir; bu bar girişte kullanmak ileri-bakış değildir.
# Veri: yfinance Ticker.earnings_dates (geçmiş tarih + Surprise%). Sembol-başına bir
# kez çekilip cache'lenir (backtest döngüsünde her bar fetch felaket olurdu).
# ---------------------------------------------------------------------------
_PEAD_CACHE: dict[str, pd.DataFrame | None] = {}


def _pead_earnings(symbol: str) -> pd.DataFrame | None:
    """Sembolün geçmiş earnings tarihleri + Surprise% (cache'li). Sütun: 'surprise',
    index = tz-naive earnings tarihi (date)."""
    s = symbol.upper()
    if s in _PEAD_CACHE:
        return _PEAD_CACHE[s]
    try:
        import yfinance as yf
        ed = yf.Ticker(s).earnings_dates
        if ed is None or ed.empty or "Surprise(%)" not in ed.columns:
            _PEAD_CACHE[s] = None
            return None
        dates = [(i.date() if hasattr(i, "date") else i) for i in ed.index]
        out = pd.DataFrame({"surprise": ed["Surprise(%)"].to_numpy()}, index=dates)
        out = out.dropna(subset=["surprise"])
        _PEAD_CACHE[s] = out if not out.empty else None
        return _PEAD_CACHE[s]
    except Exception:
        _PEAD_CACHE[s] = None
        return None


def detect_pead(df: pd.DataFrame, symbol: str | None,
                surprise_min: float = 2.0) -> tuple[SignalAction, list[str]]:
    if symbol is None:
        return SignalAction.WAIT, ["PEAD_NO_SYMBOL"]
    if len(df) < 30:
        return SignalAction.WAIT, ["INSUFFICIENT_BARS"]
    e = _pead_earnings(symbol)
    if e is None or e.empty:
        return SignalAction.WAIT, ["NO_EARNINGS_DATA"]
    prev_idx = df.index[-2]
    prev_date = prev_idx.date() if hasattr(prev_idx, "date") else prev_idx
    # Önceki bar bir earnings günü müydü? (bugün = ilk tam tepki/giriş barı)
    hits = e[e.index == prev_date]
    if hits.empty:
        return SignalAction.WAIT, ["NO_EARNINGS_EVENT"]
    surprise = float(hits["surprise"].iloc[-1])
    if surprise >= surprise_min:
        return SignalAction.LONG, [f"PEAD_LONG_surprise_{surprise:+.1f}%", f"earnings_{prev_date}"]
    if surprise <= -surprise_min:
        return SignalAction.SHORT, [f"PEAD_SHORT_surprise_{surprise:+.1f}%", f"earnings_{prev_date}"]
    return SignalAction.WAIT, [f"PEAD_SURPRISE_{surprise:+.1f}%_BELOW_MIN"]


# ---------------------------------------------------------------------------
# INSIDER (SEC EDGAR Form 4 açık-piyasa alım) — olay-bazlı ADAY sinyal (C1)
# ---------------------------------------------------------------------------
# PEAD ile aynı mantık ailesi: fiyat-göstergesi DEĞİL, temel/olay-bazlı. Giriş =
# insider küme-alım Form 4'ünün DOSYALANDIĞI günün ERTESİ barı (look-ahead güvenli:
# bilgi ancak dosyalanınca halka açık). Veri/önbellek: nihai.data.insider (ücretsiz EDGAR).
# KENAR İDDİASI YOK — research/insider_test.py ile ölç (§13.5/§16 survivorship disiplini).
# ---------------------------------------------------------------------------
def detect_insider(df: pd.DataFrame, symbol: str | None,
                   min_buyers: int = 1) -> tuple[SignalAction, list[str]]:
    if symbol is None:
        return SignalAction.WAIT, ["INSIDER_NO_SYMBOL"]
    if len(df) < 30:
        return SignalAction.WAIT, ["INSUFFICIENT_BARS"]
    try:
        from nihai.data.insider import insider_events, insider_signal
    except Exception:
        return SignalAction.WAIT, ["INSIDER_MODULE_NA"]
    ev = insider_events(symbol)
    prev_idx = df.index[-2]
    prev_date = prev_idx.date() if hasattr(prev_idx, "date") else prev_idx
    action, reasons = insider_signal(ev, prev_date, min_buyers=min_buyers)
    return SignalAction(action), reasons


# ---------------------------------------------------------------------------
# SENTIMENT (Reddit retail sentiment) — davranışsal ADAY sinyal (C2)
# ---------------------------------------------------------------------------
# PEAD/insider ailesi: fiyat-göstergesi DEĞİL. Giriş = sentiment zirvesi gününün ERTESİ barı.
# Veri/geçmiş: nihai.data.sentiment (PRAW+VADER, İLERİYE-BİRİKİMLİ — Reddit tarihsel API'si yok,
# bkz. modül docstring). Geçmiş birikene dek WAIT döner. KENAR İDDİASI YOK (research/sentiment_test.py).
# ---------------------------------------------------------------------------
def detect_sentiment(df: pd.DataFrame, symbol: str | None) -> tuple[SignalAction, list[str]]:
    if symbol is None:
        return SignalAction.WAIT, ["SENTIMENT_NO_SYMBOL"]
    if len(df) < 30:
        return SignalAction.WAIT, ["INSUFFICIENT_BARS"]
    try:
        from nihai.data.sentiment import load_history, sentiment_signal
    except Exception:
        return SignalAction.WAIT, ["SENTIMENT_MODULE_NA"]
    hist = load_history(symbol)
    prev_idx = df.index[-2]
    prev_date = prev_idx.date() if hasattr(prev_idx, "date") else prev_idx
    action, reasons = sentiment_signal(hist, prev_date)
    return SignalAction(action), reasons


# ---------------------------------------------------------------------------
# LLM-EVENT (Claude 8-K olay-çıkarımı) — olay-bazlı ADAY sinyal (C3)
# ---------------------------------------------------------------------------
# PEAD/insider ailesi: fiyat-göstergesi DEĞİL. LLM TAHMİNCİ değil ÇIKARICI'dır (8-K metnini →
# yapılandırılmış olaya). Giriş = 8-K dosyalama gününün ERTESİ barı (look-ahead güvenli — bilgi
# ancak dosyalanınca halka açık). Veri: nihai.data.llm_event (EDGAR full-text + Claude, accession
# başına cache'li). KENAR İDDİASI YOK — research/llm_event_test.py ile ölç (§13.5/§16 disiplini).
# ---------------------------------------------------------------------------
def detect_llm_event(df: pd.DataFrame, symbol: str | None) -> tuple[SignalAction, list[str]]:
    if symbol is None:
        return SignalAction.WAIT, ["LLM_EVENT_NO_SYMBOL"]
    if len(df) < 30:
        return SignalAction.WAIT, ["INSUFFICIENT_BARS"]
    try:
        from nihai.data.llm_event import llm_event_signal, llm_events
    except Exception:
        return SignalAction.WAIT, ["LLM_EVENT_MODULE_NA"]
    ev = llm_events(symbol)
    prev_idx = df.index[-2]
    prev_date = prev_idx.date() if hasattr(prev_idx, "date") else prev_idx
    action, reasons = llm_event_signal(ev, prev_date)
    return SignalAction(action), reasons


def detect_quant_trend(df: pd.DataFrame, shared=None) -> tuple[SignalAction, list[str]]:
    """Tur-97 — çok ufuklu zaman serisi momentumu (`nihai/signals/quant.py`; kurallar orada).

    Skor > 0 → LONG. Skor < 0 → yalnız `QUANT_KISA` açıksa SHORT (varsayılan uzun-yalnız).
    `shared` yalnız `df`'in kendi önbelleğiyse geçilmelidir (backtest 1d); canlıda `df` ayrıca
    çekilen günlük barlardır ve önbellek yoktur."""
    if not _quant.gunluk_mu(df.index):
        return SignalAction.WAIT, ["QUANT_GUNLUK_BAR_GEREKLI"]
    if len(df) < _quant.MIN_BAR:
        return SignalAction.WAIT, [f"QUANT_YETERSIZ_GECMIS_{len(df)}<{_quant.MIN_BAR}"]
    skor_s, sig_s = shared.quant_np() if shared is not None else _quant.seriler(df["close"].to_numpy(dtype=float))
    skor, sig = float(skor_s[-1]), float(sig_s[-1])
    if not (math.isfinite(skor) and math.isfinite(sig) and sig > 0):
        return SignalAction.WAIT, ["QUANT_TANIMSIZ"]
    reasons = [f"QUANT_SKOR_{skor:+.2f}", f"QUANT_SIGMA_{sig * 100:.2f}%"]
    if skor > 0:
        return SignalAction.LONG, reasons
    if skor < 0 and CONFIG.strategy.quant_allow_short:
        return SignalAction.SHORT, reasons
    return SignalAction.WAIT, reasons + (["QUANT_KISA_KAPALI"] if skor < 0 else ["QUANT_NOTR"])


def is_quant_label(label: str | None) -> bool:
    """Sinyal/pozisyon etiketi quant kipine mi ait? ("QuantTrend", "quant_trend V1" …)."""
    return bool(label) and str(label).replace("_", "").lower().startswith("quant")


# ---------------------------------------------------------------------------
# Strateji dispatcher
# ---------------------------------------------------------------------------
# Tek kaynak: geçerli strateji modları (dispatcher + config doğrulama + CLI'ler)
STRATEGY_MODES = ("event_breakout", "mean_reversion", "rsi_divergence",
                  "macd", "donchian", "squeeze",
                  "trend_pullback", "pead", "insider", "sentiment", "llm_event",
                  "vol_breakout", "ensemble", "confluence", "quant_trend")

# Fiyat-only, kendi-kendine yeten (harici veri/sembol gerektirmeyen) baz stratejiler.
# Walk-forward / oto-tarama strateji-SEÇİMİ yalnız bunlar arasından yapılmalı:
#   - pead/insider/sentiment çalışma-anında HARİCİ veri çeker (earnings/EDGAR/Reddit) →
#     tarihsel backtest döngüsünde look-ahead + ağ-gecikmesi riski; çoğu sembolde veri
#     yokken hep WAIT döner (aday olarak gürültü). Bunlar kendi research/*_test.py
#     harness'larıyla survivorship-temiz ölçülür (§13.5/§16 disiplini).
#   - ensemble/confluence BİRLEŞTİRİCİ meta-modlar; "en iyi baz stratejiyi seç"
#     aramasında baz aday olamazlar (kendileri baz stratejileri çağırır).
#   - quant_trend (Tur-97) fiyat-only ama GÜNLÜK bar + ≥253 bar geçmiş ister; walk-forward/oto-tarama
#     1h pencerelerinde aday olsa her barda QUANT_GUNLUK_BAR_GEREKLI döner. Ayrıca Tur-96 düzeneği
#     BASE'i Hedge'in uzman listesi olarak kullanır — eklemek o ölçümü sessizce değiştirirdi.
BASE_STRATEGY_MODES = ("event_breakout", "mean_reversion", "rsi_divergence",
                       "macd", "donchian", "squeeze", "trend_pullback", "vol_breakout")
def detect_setup(df: pd.DataFrame, mode: str | None = None,
                 symbol: str | None = None, shared=None,
                 df_daily: pd.DataFrame | None = None) -> tuple[SignalAction, list[str], str]:
    """
    Aktif strateji moduna göre detect fonksiyonunu çağır.
    Dönüş: (action, reasons, strategy_label)
    symbol: olay-bazlı sinyaller (PEAD) için gerekli; fiyat-only modlar yok sayar.
    shared: opsiyonel paylaşımlı gösterge önbelleği (Tur-7) — fiyat-only detektörlere
    iletilir; olay-bazlılar (pead/insider/sentiment/llm_event) gösterge kullanmaz.
    df_daily: quant_trend için ayrıca çekilmiş GÜNLÜK barlar (Tur-97, canlı 1h tarama). Verilirse
    quant detektörü onu kullanır ve `shared` (df'in önbelleği) ona GEÇİLMEZ; diğer kipler yok sayar.
    """
    mode = (mode or ACTIVE_STRATEGY).lower()
    if mode == "quant_trend":
        if df_daily is not None:
            a, r = detect_quant_trend(df_daily)
        else:
            a, r = detect_quant_trend(df, shared=shared)
        return a, r, "QuantTrend"
    single: dict[str, tuple[Callable[..., tuple[SignalAction, list[str]]], str]] = {
        "event_breakout": (detect_event, "EventBreakout"),
        "mean_reversion": (detect_mean_reversion, "MeanReversion"),
        "rsi_divergence": (detect_rsi_divergence, "RSIDivergence"),
        "macd": (detect_macd, "MACD"),
        "donchian": (detect_donchian, "Donchian"),
        "squeeze": (detect_squeeze, "Squeeze"),
        "trend_pullback": (detect_trend_pullback, "TrendPullback"),
        "vol_breakout": (detect_vol_breakout, "VolBreakout"),
    }
    if mode in single:
        fn, label = single[mode]
        a, r = fn(df, shared=shared)
        return a, r, label
    if mode == "pead":
        a, r = detect_pead(df, symbol)
        return a, r, "PEAD"
    if mode == "insider":
        a, r = detect_insider(df, symbol)
        return a, r, "Insider"
    if mode == "sentiment":
        a, r = detect_sentiment(df, symbol)
        return a, r, "Sentiment"
    if mode == "llm_event":
        a, r = detect_llm_event(df, symbol)
        return a, r, "LLMEvent"
    if mode == "ensemble":
        from nihai.config import ENSEMBLE_MODE
        if ENSEMBLE_MODE == "first_trigger":
            return _detect_ensemble_first_trigger(df, shared=shared)
        return _detect_ensemble(df, shared=shared)
    if mode == "confluence":
        # Katı mutabakat: ≥CONFLUENCE_MIN_VOTES bağımsız detektör hemfikir olmalı (§16.2).
        from nihai.config import CONFLUENCE_MIN_VOTES
        return _detect_ensemble(df, min_votes=max(2, CONFLUENCE_MIN_VOTES), shared=shared)
    return SignalAction.WAIT, [f"UNKNOWN_MODE_{mode}"], mode


# Ensemble oylaması — (detect fn, okunur etiket, stop-routing mode anahtarı, trend mi?)
# Tip anotasyonu: imzalar heterojen (opsiyonel period/lookback paramları) → mypy'nin
# ortak çağrılabilir tipe indirgemesi için Callable[..., ...] gerekir (Tur-7).
_ENSEMBLE_DETECTORS: tuple[tuple[Callable[..., tuple[SignalAction, list[str]]],
                                 str, str, bool], ...] = (
    (detect_event, "EventBreakout", "event_breakout", True),
    (detect_macd, "MACD", "macd", True),
    (detect_donchian, "Donchian", "donchian", True),
    (detect_squeeze, "Squeeze", "squeeze", True),
    (detect_mean_reversion, "MeanReversion", "mean_reversion", False),
    (detect_rsi_divergence, "RSIDivergence", "rsi_divergence", False),
)


def _detect_ensemble_first_trigger(df: pd.DataFrame,
                                   shared=None) -> tuple[SignalAction, list[str], str]:
    """Eski davranış (A/B karşılaştırma için): sırayla dene, ilk tetikleyeni al."""
    if shared is None:
        shared = _SharedIndicators(df)   # Tur-7: detektörler arası RSI/ADX/BB paylaşımı
    for fn, label, mode_key, _is_trend in _ENSEMBLE_DETECTORS:
        a, r = fn(df, shared=shared)
        if a in (SignalAction.LONG, SignalAction.SHORT):
            return a, [*r, f"VIA_{label}"], mode_key
    return SignalAction.WAIT, ["ENSEMBLE_NO_TRIGGER"], "Ensemble"


def _detect_ensemble(df: pd.DataFrame, min_votes: int = 1,
                     shared=None) -> tuple[SignalAction, list[str], str]:
    """Confluence oylaması: tüm detektörleri çalıştır, yön çakışmasında BEKLE.

    "İlk tetikleyen kazanır" yerine: bağımsız kurulumların UZLAŞMASI aranır.
    - LONG ve SHORT aynı anda tetiklerse → whipsaw bölgesi → WAIT (kenarda dur).
    - Aksi halde baskın yön döner; kaç detektör hemfikirse `CONFLUENCE_N` olarak
      işaretlenir (generate_signal kalite skoruna bonus yazar).
    - `min_votes` > 1 ise (katı `confluence` modu) o sayıda detektör HEMFİKİR değilse
      WAIT — tek-detektör (gürültü) tetikleri elenir. §16.2 dersi: bağımsız ONAY
      precision'ı keskinleştirir (in-sample'da 53.8→66.1). KENAR İDDİASI DEĞİL,
      yalnız false-positive azaltma kaldıracı; survivorship-temiz ölçüm şart.
    - Stop/target stili: trend-takip oyu varsa ilk trend detektörünün moduna,
      yoksa mean-reversion moduna göre yönlendirilir.
    """
    if shared is None:
        shared = _SharedIndicators(df)   # Tur-7: detektörler arası RSI/ADX/BB paylaşımı
    longs: list[tuple[str, str, bool, list[str]]] = []
    shorts: list[tuple[str, str, bool, list[str]]] = []
    for fn, label, mode_key, is_trend in _ENSEMBLE_DETECTORS:
        a, r = fn(df, shared=shared)
        if a == SignalAction.LONG:
            longs.append((label, mode_key, is_trend, r))
        elif a == SignalAction.SHORT:
            shorts.append((label, mode_key, is_trend, r))

    if not longs and not shorts:
        return SignalAction.WAIT, ["ENSEMBLE_NO_TRIGGER"], "Ensemble"
    if longs and shorts:
        # Çelişki: bağımsız sinyaller ters yönde → düşük güven, gürültü. Pas geç.
        return (SignalAction.WAIT,
                [f"ENSEMBLE_CONFLICT_L{len(longs)}_S{len(shorts)}"], "Ensemble")

    side = SignalAction.LONG if longs else SignalAction.SHORT
    votes = longs or shorts
    if len(votes) < min_votes:
        return (SignalAction.WAIT,
                [f"CONFLUENCE_BELOW_MIN_{len(votes)}<{min_votes}"], "Ensemble")
    # Stop-routing: ilk trend oyunun modu, yoksa ilk (mean-reversion) oyun modu.
    routing_mode = next((v[1] for v in votes if v[2]), votes[0][1])
    reasons: list[str] = []
    for label, _mk, _t, r in votes:
        reasons.append(f"{label}:{r[0]}" if r else label)
    reasons.append(f"CONFLUENCE_{len(votes)}")
    return side, reasons, routing_mode


# ---------------------------------------------------------------------------
# Stops / targets — strateji-aware
# ---------------------------------------------------------------------------
def compute_stops_targets(df: pd.DataFrame, action: SignalAction, mode: str,
                          atr_val: float | None = None) -> tuple[float, float]:
    """Stop/target stratejiye göre farklı: trend takip vs mean reversion.
    atr_val: çağıran ATR(14) son değerini zaten hesapladıysa geçer (tekrar-hesap
    eliminasyonu, Tur-6); None ise eski davranış (kendisi hesaplar)."""
    c = df["close"]
    price = float(c.iloc[-1])
    a = float(atr(df, 14).iloc[-1]) if atr_val is None else float(atr_val)
    if np.isnan(a) or a <= 0:
        a = price * 0.01  # son çare %1

    # Etiket normalize: ensemble "MeanReversion"/"RSIDivergence" gibi okunur etiket
    # döndürebilir; alt çizgisiz/CamelCase de mean-reversion stop stilini seçmeli.
    mode_norm = mode.replace("_", "").replace("-", "").lower()
    if mode_norm in ("meanreversion", "rsidivergence"):
        # Mean reversion: hedef = orta bant; stop = dar (1 ATR)
        ma20 = float(c.rolling(20).mean().iloc[-1])
        if action == SignalAction.LONG:
            return price - a, ma20
        return price + a, ma20

    # Default trend takip: ATR-çarpanı stop/hedef (config'ten, env-ayarlanabilir).
    s_mult = CONFIG.strategy.trend_stop_atr
    t_mult = CONFIG.strategy.trend_target_atr
    if action == SignalAction.LONG:
        return price - s_mult * a, price + t_mult * a
    return price + s_mult * a, price - t_mult * a


# ---------------------------------------------------------------------------
# Filtreler — Madde 16, 17
# ---------------------------------------------------------------------------
def regime_filter(df: pd.DataFrame, adx_val: float | None = None) -> tuple[bool, str, float]:
    """Madde 16 — ADX < 20 ise risk %50 ↓ (zayıf trend). adx, durum, çarpan döner.
    adx_val: çağıran ADX son değerini zaten hesapladıysa geçer (Tur-6); None=eski davranış."""
    a = adx(df).iloc[-1] if adx_val is None else adx_val
    if np.isnan(a):
        return True, "ADX_NA", 1.0
    if a < CONFIG.strategy.adx_trend_threshold:
        return True, f"ADX_{a:.0f}_WEAK_TREND", 0.5
    return True, f"ADX_{a:.0f}_OK", 1.0


def volatility_explosion_filter(df: pd.DataFrame,
                                atr_series: pd.Series | None = None) -> tuple[bool, str]:
    """Madde 17 — anlık ATR > X * ortalama → giriş yok.
    atr_series: çağıran ATR(14) serisini zaten hesapladıysa geçer (Tur-6); None=eski davranış."""
    a = atr(df, 14) if atr_series is None else atr_series
    # Tur-45: gövde artık numpy üzerinde. `atr_series` pandas Series de olabilir (dış
    # çağıranlar/testler) — float64 seride `asarray` SIFIR-KOPYADIR, maliyet eklemez.
    # BULGU-43-2 (Tur-43): "hepsi NaN mi?" sorusu için `dropna()` YENİ bir Series+Index
    # tahsis ediyordu; bu kapı backtest'te BAR BAŞINA çağrılır → O(n²) tahsis. Son bar
    # geçerliyse tam tarama zaten kısa devre olur; tam-NaN yolu davranışı korur.
    v = a if isinstance(a, np.ndarray) else np.asarray(a, dtype=float)
    if v.size == 0 or (not np.isfinite(v[-1]) and bool(np.isnan(v).all())):
        return True, "ATR_NA"
    cur = float(v[-1])
    # `.iloc[-20:].mean()` (skipna) ≡ `np.nanmean` — ÖLÇÜLDÜ: ≤20 elemanlı 20.000
    # pencerede (NaN'lı/tam-NaN/büyük-ölçekli dahil) fark **0,000e+00**, bottleneck
    # kurulu olmasına rağmen. Tam-NaN kuyrukta nanmean UYARI basar → dal ayrıldı.
    tail = v[-20:]
    avg = float("nan") if bool(np.isnan(tail).all()) else float(np.nanmean(tail))
    if cur > CONFIG.strategy.atr_explosion_multiplier * avg:
        return False, f"ATR_EXPLOSION_{cur/avg:.2f}x"
    return True, "ATR_NORMAL"


# ---------------------------------------------------------------------------
# Kalite skoru — Madde 18
# ---------------------------------------------------------------------------
def signal_quality_score(
    df: pd.DataFrame, action: SignalAction, *,
    htf_trend_aligned: bool, iv_rank_val: float | None,
    chain_health_ok: bool, confluence: int = 1,
    adx_val: float | None = None, rsi_val: float | None = None,
) -> tuple[float, dict]:
    """0-6.5 arası skor. confluence = aynı yönde hemfikir bağımsız sinyal sayısı
    (ensemble); ilkin ötesindeki her uzlaşma +0.75 (max +1.5) — gürültü tek-gösterge
    tetikleri düşük skorda kalırken çoklu-onaylı kurulumlar eşiği rahat geçer.
    adx_val/rsi_val: çağıran son ADX/RSI değerini zaten hesapladıysa geçer (Tur-6
    tekrar-hesap eliminasyonu); None ise eski davranış (kendisi hesaplar)."""
    score = 0.0
    breakdown: dict = {}

    # 0) Confluence bonusu — ensemble uzlaşması (tek detektör=1 → bonus yok)
    if confluence > 1:
        conf_bonus = min((confluence - 1) * 0.75, 1.5)
        score += conf_bonus
        breakdown["confluence"] = round(conf_bonus, 2)

    # 1) Trend gücü
    a = adx(df).iloc[-1] if adx_val is None else adx_val
    trend_score = 0.0 if np.isnan(a) else min(a / 40.0, 1.0)
    score += trend_score
    breakdown["trend"] = round(trend_score, 2)

    # 2) RSI uyumu
    r = rsi(df["close"]).iloc[-1] if rsi_val is None else rsi_val
    if not np.isnan(r):
        if (action == SignalAction.LONG and 50 <= r <= 70) or \
           (action == SignalAction.SHORT and 30 <= r <= 50):
            score += 1.0
            breakdown["rsi"] = 1.0
        else:
            breakdown["rsi"] = 0.0

    # 3) HTF uyumu
    if htf_trend_aligned:
        score += 1.0
        breakdown["htf_align"] = 1.0

    # 4) Hacim onayı — volume sütunu garanti değil (bkz. detect_vol_breakout savunmacı
    # kapısı: bazı sağlayıcı/df'lerde hacim yok). Yoksa KeyError yerine bonusu atla.
    if "volume" in df.columns:
        vol = df["volume"].iloc[-20:]
        if df["volume"].iloc[-1] > vol.mean() * 1.2:
            score += 1.0
            breakdown["volume"] = 1.0

    # 5) Opsiyon zinciri sağlığı
    if chain_health_ok:
        score += 0.5
        breakdown["chain_health"] = 0.5

    # IV rejimi bonus. `iv_rank_val=None` = ÖLÇÜM YOK (Tur-31) → bileşen hiç oluşmaz;
    # `get_iv_rank`'ın uydurduğu 50.0 bandın içinde kaldığı için eskiden bonus koşulsuzdu.
    # Rejim `IV_REGIME_BONUS` ile seçilir; Tur-51'den beri varsayılan "off"
    # (bandın hiçbir yönde bilgi taşımadığı ölçüldü — EDGE_FINDINGS §28).
    if iv_rank_val is not None:
        kip = CONFIG.strategy.iv_regime_bonus_mode
        lo, hi = CONFIG.strategy.iv_rank_low_threshold, CONFIG.strategy.iv_rank_high_threshold
        hak_edildi = (iv_rank_val <= lo) if kip == "low" else (
            False if kip == "off" else lo <= iv_rank_val <= hi)
        if hak_edildi:
            score += 0.5
            breakdown["iv_regime"] = 0.5
        breakdown["iv_rank"] = round(iv_rank_val, 1)

    return score, breakdown


# ---------------------------------------------------------------------------
# R:R hesabı — Madde 8
# ---------------------------------------------------------------------------
def compute_rr(entry: float, stop: float, target: float, action: SignalAction) -> float:
    if action == SignalAction.LONG:
        risk = entry - stop
        reward = target - entry
    else:
        risk = stop - entry
        reward = entry - target
    if risk <= 0:
        return 0.0
    return reward / risk


# ---------------------------------------------------------------------------
# ALTERNATİF VERİ — Çapraz-varlık VIX rejimi (cache'li, look-ahead-güvenli)
# ---------------------------------------------------------------------------
# ^VIX günlük kapanışı bir kez çekilir. Karar gününden STRICT ÖNCEki kapanış kullanılır
# (intraday barlar dahil her zaman güvenli; VIX rejimi yavaş değişir, dünkü kapanış iyi proxy).
_VIX_DATES: list | None = None
_VIX_CLOSES: list | None = None
_VIX_LOADED = False


def _load_vix() -> None:
    global _VIX_DATES, _VIX_CLOSES, _VIX_LOADED
    if _VIX_LOADED:
        return
    _VIX_LOADED = True
    try:
        import yfinance as yf
        df = yf.download("^VIX", period="10y", interval="1d",
                         auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            return
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        col = "Close" if "Close" in df.columns else "close"
        _VIX_DATES = [(i.date() if hasattr(i, "date") else i) for i in df.index]
        _VIX_CLOSES = [float(x) for x in df[col].to_numpy()]
    except Exception:
        _VIX_DATES = None
        _VIX_CLOSES = None


def _vix_close_before(d) -> float | None:
    """Karar gününden STRICT önce en son VIX kapanışı (look-ahead-güvenli)."""
    _load_vix()
    if not _VIX_DATES or _VIX_CLOSES is None or d is None:
        return None
    import bisect
    lo = bisect.bisect_left(_VIX_DATES, d)   # ilk tarih >= d
    if lo == 0:
        return None
    return _VIX_CLOSES[lo - 1]


# ---------------------------------------------------------------------------
# Ana üretici
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CANLI GİRİŞ ÇAPASI (Tur-39) — sinyal fiyatı ile İŞLEM fiyatı aynı şey değildir.
# ---------------------------------------------------------------------------
def quant_capala(sig: Signal, fiyat: float) -> Signal:
    """Tur-97 (BULGU-97-2): quant stop/hedefi bir SEVİYE değil σ MESAFESİDİR → dolum fiyatına mesafe
    korunarak taşınır (fiyat alanı karar fiyatı olarak KALIR; çağıran onu ayrıca günceller).

    Ölçüldü: motor stopu karar barının kapanışına göre kurup girişi sonraki açılıştan yapınca, gece
    boşluğu ya da +1 bar gecikme stopu girişin ÜSTÜNE taşıyabiliyordu (SPY 2018-10-10: LONG giriş
    238,96, stop 244,71 → ertesi bar GAP_STOP); gecikmeli koşuda GAP_STOP 17 → 33'e çıktı. TA
    stopları yapısaldır (kırılım seviyesi) ve bu dönüşümden GEÇMEZ."""
    if not is_quant_label(sig.metadata.get("strategy")):
        return sig
    d = float(fiyat) - float(sig.price)
    return dc_replace(sig, stop=sig.stop + d, target=sig.target + d)


def reanchor_to_live(sig: Signal, live_price: float, *, min_rr: float) -> tuple[Signal | None, str]:
    """Sinyali CANLI fiyata çapala; kurulum artık geçerli değilse (None, sebep) döndür.

    NEDEN VAR: `generate_signal` fiyatı `df_htf["close"].iloc[-1]`, yani KAPANMIŞ (ya da
    yarı oluşmuş) bir HTF barının kapanışıdır. `HTF_INTERVAL=1h` iken bu fiyat ~59 dakika
    BAYAT olabilir; pozisyon ise 1m barlarla/kotasyonla yönetilir. Aradaki fark ölçüldü
    (510 sanal işlem, 2026-07-06→08-20):
      · işlemlerin **%46'sında MFE tam 0,00** — pozisyon giriş fiyatının lehine BİR KEZ
        bile geçmemiş, yani girilen fiyat piyasada YOKTU;
      · `GAP_STOP` 68 işlemde **68/68 aleyhte**, ortalama **−%2,45** ve **−1,497R**,
        toplam **−299.438 $** (sistemin TÜM net zararından büyük; bu sınıf olmadan kalan
        442 işlem +41.804 $ / PF 1,178);
      · 27'si girişten sonra **2 dakika içinde** tetiklenmiş.
    Gerçek bir "gap" dağılımı simetriktir; 68/68 aleyhte olması piyasa değil ÖLÇÜM
    kusurudur (Tur-37'nin uydurma-Greek dersiyle aynı sınıf: karar yolunda bir sayı,
    gerçeği değil bir varsayımı raporluyordu).

    NE YAPAR: girişi canlı fiyata taşır, stop/target'ı YAPISAL bırakır (kaydırmaz) ve
    RR'ı yeniden hesaplar. Fiyat stop'un ötesine geçmişse ya da RR eşiğin altına düşmüşse
    kurulum BAYATTIR → (None, sebep). Bu yeni bir filtre DEĞİL, mevcut `min_rr_ratio`
    kapısının GERÇEKTEN işlem görülebilecek fiyatta uygulanmasıdır: kaçmış bir kırılmayı
    kovalamak RR'ı yok eder, ve bunu ancak canlı fiyata bakan bir kapı görebilir.

    live_price geçersizse (<=0 / NaN) sinyal DEĞİŞMEDEN döner: kotasyon yokluğu bir
    RED gerekçesi değildir (ölçüm yokluğu ölçüm gibi kullanılamaz)."""
    if sig.action not in (SignalAction.LONG, SignalAction.SHORT):
        return sig, ""
    try:
        px = float(live_price)
    except (TypeError, ValueError):
        return sig, ""
    if not (px > 0) or px != px:          # <=0 veya NaN → çapa yok, davranış eskisi gibi
        return sig, ""
    sig = quant_capala(sig, px)           # Tur-97: quant σ-mesafesi canlı fiyata taşınır (TA'da no-op)

    if sig.action == SignalAction.LONG:
        risk, reward = px - sig.stop, sig.target - px
    else:
        risk, reward = sig.stop - px, px - sig.target
    drift_bps = (px - sig.price) / sig.price * 1e4 if sig.price else 0.0

    if risk <= 0:
        return None, (f"STALE_PRICE_THROUGH_STOP: canlı {px:.4f} stop'un ({sig.stop:.4f}) "
                      f"ötesinde (sinyal fiyatı {sig.price:.4f}, sapma {drift_bps:+.0f}bps)")
    rr = reward / risk
    if rr < min_rr:
        return None, (f"STALE_RR: canlı fiyatta RR={rr:.2f} < {min_rr:.2f} "
                      f"(sinyal RR={sig.rr_ratio:.2f}, sapma {drift_bps:+.0f}bps)")
    return dc_replace(sig, price=px, rr_ratio=round(rr, 2),
                      reasons=[*sig.reasons, f"LIVE_ANCHOR{drift_bps:+.0f}bps"]), ""


def generate_signal(
    symbol: str, df_htf: pd.DataFrame, df_1m: pd.DataFrame | None = None,
    *, iv_rank_val: float | None = None, chain_health_ok: bool = True,
    strategy_mode: str | None = None, shared=None, live_interval: str | None = None,
    chain_eval: Callable[[], tuple[bool, float | None]] | None = None,
    df_daily: pd.DataFrame | None = None,
) -> Signal:
    """
    df_htf: 1h veya 15m (yön kararı için)
    df_1m: 1 dakikalık (stop/granül kontrol için, Madde 5)
    strategy_mode: event_breakout | mean_reversion | rsi_divergence | ensemble
    shared: opsiyonel _SharedIndicators/_SharedSlice (Tur-7). Backtest, sembol başına
        göstergeleri BİR kez tam df üzerinde hesaplayıp her bara kırpılmış görünüm
        geçer (nedensel göstergelerde bit-bit aynı değerler; O(n²)→O(n)). None ise
        çağrı-içi tembel önbellek kurulur — dış çağıranlar için davranış aynı.
    chain_eval: verilirse `(chain_health_ok, iv_rank_val)` YALNIZ kalite adımına
        ulaşıldığında çağrılır ve iki argümanın yerine geçer (Tur-82). Opsiyon zinciri bu
        iki değerden başka hiçbir yere girmez ve kalite adımından önceki her WAIT
        onlardan bağımsızdır; canlı taramada zincir o yüzden yalnız gerektiğinde çekilir.
    df_daily: quant_trend kipinde GÜNLÜK barlar (Tur-97). Canlı tarama 1h ile karar verirken
        quant skoru/σ günlük barlardan gelir; verilmezse df_htf'in kendisi günlük sayılır
        (backtest 1d) ve günlük değilse detektör QUANT_GUNLUK_BAR_GEREKLI ile bekler.
    """
    if strategy_mode is None:
        from nihai.runtime_strategy import apply
        strategy_mode = apply()
    if len(df_htf) < 60:
        return Signal(SignalAction.WAIT, symbol, 0, 0, 0, reasons=["INSUFFICIENT_HTF_BARS"])

    if shared is None:
        shared = _SharedIndicators(df_htf)

    # Madde 17 — Volatilite filtresi. ATR(14) serisi BİR kez hesaplanır; aşağıda
    # stop/target ve NaN-kapısında yeniden kullanılır (Tur-6: aynı dilimde 3× ATR
    # hesabı elenmiştir — değerler birebir aynı, yalnız tekrar-hesap yok).
    # Tur-45: `shared.atr14()` bar başına O(n) kırpma kuruyordu; filtre ve stop/target
    # yalnız SON değeri ve son-20 ortalamasını okuyor → kırpmasız numpy görünümü yeter.
    atr_np = shared.atr14_np()
    vol_ok, vol_reason = volatility_explosion_filter(df_htf, atr_series=atr_np)
    if not vol_ok:
        return Signal(SignalAction.WAIT, symbol, df_htf["close"].iloc[-1], 0, 0, reasons=[vol_reason])

    # Strateji dispatcher (Madde 1 olay tetik + alternatif setup'lar)
    action, event_reasons, strat_label = detect_setup(df_htf, strategy_mode,
                                                      symbol=symbol, shared=shared, df_daily=df_daily)
    if action == SignalAction.WAIT:
        return Signal(action, symbol, df_htf["close"].iloc[-1], 0, 0, reasons=event_reasons,
                      metadata={"strategy": strat_label})

    # Yalnızca-long kipi: SHORT sinyalleri bastır (sürekli-yükselen evren tezi).
    if CONFIG.strategy.enable_long_only and action == SignalAction.SHORT:
        return Signal(SignalAction.WAIT, symbol, df_htf["close"].iloc[-1], 0, 0,
                      reasons=[*event_reasons, "LONG_ONLY_SKIP_SHORT"],
                      metadata={"strategy": strat_label})

    # Çapraz-varlık VIX rejim kapısı (opt-in) — yüksek-korku rejiminde yeni giriş yok.
    if CONFIG.strategy.enable_vix_regime_gate:
        _dt = df_htf.index[-1]
        _vix = _vix_close_before(_dt.date() if hasattr(_dt, "date") else None)
        if _vix is not None and _vix > CONFIG.strategy.vix_high_threshold:
            return Signal(SignalAction.WAIT, symbol, df_htf["close"].iloc[-1], 0, 0,
                          reasons=event_reasons + [f"VIX_{_vix:.1f}_REGIME_OFF"],
                          metadata={"strategy": strat_label})

    price = df_htf["close"].iloc[-1]

    # ---- Edge filtreleri (Madde 16 sertleştirme) — yalnızca TREND-takip kurulumlarına.
    # Mean-reversion/divergence kasıtlı olarak rejimi fade eder; kendi iç ADX<25 +
    # RSI-aşırı kapıları vardır, bu yüzden bu kapılardan muaftır.
    is_mr = strat_label.replace("_", "").replace("-", "").lower() in ("meanreversion", "rsidivergence")
    # Tur-97: quant kipi TA kapılarından MUAFTIR. RSI aşırı-uzama freni tam da momentumun en güçlü
    # olduğu yerde alımı keser (TSMOM'un tersine bahis) ve ADX/EMA50 günlük skorla ölçeği uyuşmayan
    # saatlik göstergelerdir; quant girişini yalnız ufuk uzlaşması belirler.
    is_quant = strat_label == "QuantTrend"
    # Tur-6/7: ADX/RSI/EMA50 paylaşımlı önbellekten — kalite skoru, HTF hizası ve
    # rejim filtresi aynı değerleri yeniden kullanır (tekrar-hesap yok, değerler aynı).
    if not is_mr and not is_quant:
        e50 = shared.ema(50).iloc[-1]
        htf_ok = (action == SignalAction.LONG and price > e50) or \
                 (action == SignalAction.SHORT and price < e50)
        # (1) HTF trend hizalama sert-kapı: EMA50'nin ters tarafında trend işlemi yok.
        if CONFIG.strategy.enable_trend_alignment_gate and not htf_ok:
            return Signal(SignalAction.WAIT, symbol, price, 0, 0,
                          reasons=event_reasons + ["HTF_TREND_MISALIGNED"],
                          metadata={"strategy": strat_label})
        # (2) ADX tabanı: zayıf trendde trend-takip girişi gürültüdür → pas.
        adx_v = shared.adx_last()
        if (CONFIG.strategy.enable_adx_floor_gate and not np.isnan(adx_v)
                and adx_v < CONFIG.strategy.adx_trend_threshold):
            return Signal(SignalAction.WAIT, symbol, price, 0, 0,
                          reasons=event_reasons + [f"ADX_{adx_v:.0f}_BELOW_FLOOR"],
                          metadata={"strategy": strat_label})
        # (3) RSI aşırı-uzama freni: hareketin sonunu kovalama.
        rsi_v = shared.rsi14().iloc[-1]
        if not np.isnan(rsi_v):
            if action == SignalAction.LONG and rsi_v > CONFIG.strategy.rsi_chase_long:
                return Signal(SignalAction.WAIT, symbol, price, 0, 0,
                              reasons=event_reasons + [f"RSI_{rsi_v:.0f}_OVEREXTENDED"],
                              metadata={"strategy": strat_label})
            if action == SignalAction.SHORT and rsi_v < CONFIG.strategy.rsi_chase_short:
                return Signal(SignalAction.WAIT, symbol, price, 0, 0,
                              reasons=event_reasons + [f"RSI_{rsi_v:.0f}_OVEREXTENDED"],
                              metadata={"strategy": strat_label})

    a = atr_np[-1] if atr_np.size else np.nan
    if np.isnan(a) or a <= 0:
        return Signal(SignalAction.WAIT, symbol, price, 0, 0, reasons=["ATR_NA"])

    # Stop/target strateji-aware (ATR yukarıda hesaplandı — yeniden kullan)
    q_skor = q_sig = float("nan")
    if is_quant:
        # Tur-97: risk birimi GÜNLÜK EWMA σ (quant.py). Detektör aynı diziyi okudu; burada
        # yeniden okunur (backtest'te önbellekten O(1), canlıda ~300 günlük barda ihmal edilir).
        _qs, _qg = (shared.quant_np() if df_daily is None
                    else _quant.seriler(df_daily["close"].to_numpy(dtype=float)))
        q_skor, q_sig = float(_qs[-1]), float(_qg[-1])
        stop, target = _quant.stop_hedef(float(price), action.value, q_sig)
    else:
        stop, target = compute_stops_targets(df_htf, action, strat_label.lower(), atr_val=float(a))

    rr = compute_rr(price, stop, target, action)
    if rr < CONFIG.strategy.min_rr_ratio:
        # WAIT döndür (action DEĞİL): tüketiciler yalnız is_actionable()'a bakar; action
        # korunursa RR-altı (ve mean-reversion'da hedefi ters tarafa düşen negatif-RR)
        # sinyaller backtest'e ve canlıya işlem olarak sızar. Diğer tüm kapılarla tutarlı.
        return Signal(SignalAction.WAIT, symbol, price, stop, target,
                      reasons=[f"RR_{rr:.2f}_BELOW_MIN"], rr_ratio=rr,
                      metadata={"strategy": strat_label})

    # Madde 18 — Kalite skoru (EMA50 paylaşımlı önbellekten — memo'da varsa bedava)
    e50_v = shared.ema(50).iloc[-1]
    htf_aligned = (action == SignalAction.LONG and df_htf["close"].iloc[-1] > e50_v) or \
                  (action == SignalAction.SHORT and df_htf["close"].iloc[-1] < e50_v)
    # Ensemble confluence sayısını gerekçelerden çöz (CONFLUENCE_N) — kalite bonusu.
    confluence = 1
    for _r in event_reasons:
        if _r.startswith("CONFLUENCE_"):
            try:
                confluence = int(_r.split("_", 1)[1])
            except ValueError:
                pass
            break
    if is_quant:
        # Tur-97: kalite = ufuk uzlaşması (TA ölçeğinde); KAPI DEĞİL. Opsiyon zinciri/IV rank TA
        # kalite skoruna girer, quant kararına girmez → zincir quant kipinde hiç çekilmez.
        score, breakdown = _quant.kalite(q_skor)
    else:
        if chain_eval is not None:
            chain_health_ok, iv_rank_val = chain_eval()
        score, breakdown = signal_quality_score(
            df_htf, action, htf_trend_aligned=htf_aligned,
            iv_rank_val=iv_rank_val, chain_health_ok=chain_health_ok,
            confluence=confluence, adx_val=shared.adx_last(), rsi_val=shared.rsi14().iloc[-1],
        )

    reasons = event_reasons + [f"RR_{rr:.2f}", f"QUALITY_{score:.2f}"]

    if not is_quant and score < CONFIG.strategy.min_signal_quality_score:
        # WAIT döndür (action DEĞİL): kalite-altı kurulumlar da is_actionable() üstünden
        # işleme sızmamalı (RR kapısıyla aynı tutarlılık gerekçesi).
        return Signal(SignalAction.WAIT, symbol, price, stop, target,
                      reasons=reasons + ["QUALITY_BELOW_THRESHOLD"],
                      quality_score=score, rr_ratio=rr,
                      metadata={"breakdown": breakdown})

    # Madde 16 — rejim çarpanı (risk modülüne iletilir; ADX paylaşımlı önbellekten)
    _, regime_reason, regime_mult = regime_filter(df_htf, adx_val=shared.adx_last())
    reasons.append(regime_reason)

    final = Signal(
        action=action, symbol=symbol, price=price, stop=stop, target=target,
        reasons=reasons, quality_score=score, rr_ratio=rr,
        metadata={"breakdown": breakdown, "atr": float(a),
                  "regime_mult": regime_mult, "strategy": strat_label},
    )
    if is_quant:
        final.metadata["quant"] = {"skor": round(q_skor, 4), "sigma_gunluk": round(q_sig, 6)}

    # Meta-labeling kapısı (opt-in) — ikincil model kazanma olasılığını tahmin eder.
    # KİP: "off" (hiç çağrılmaz) | "shadow" (skorla + logla, ASLA bastırma) | "enforce"
    # (eşik altı WAIT — eski davranış). Gölge modun varlık nedeni: canlı gate_sim kanıtı
    # GATE_HURTS derken kapı kapatılmalı AMA p_win loglama döngüsü (prediction_tracker →
    # gate_sim/pwin_calibration) ölmemeli. Geç import: strategy↔meta_gate dairesel
    # bağımlılığı önler. Model yoksa kapı her kipte pasiftir (probability None → eleme yok).
    _sc = CONFIG.strategy
    final.metadata["meta_strategy"] = strategy_mode
    gate_mode = getattr(_sc, "meta_gate_mode", None)
    if gate_mode is None:  # eski/fake config (yalnız enable_meta_gate bayrağı) → semantiği koru
        gate_mode = "enforce" if getattr(_sc, "enable_meta_gate", False) else "off"
    live_threshold = None
    if live_interval is not None:
        from nihai.signals.model_promotion import live_policy
        gate_mode, live_threshold = live_policy(gate_mode, live_interval, strategy_mode)
    if is_quant and gate_mode != "off":
        # Tur-97 (BULGU-97-4): meta kapı TA sinyallerinin özellikleriyle eğitildi (RR ~2, TA kalite
        # ölçeği, kırılım bağlamı). ÖLÇÜLDÜ (2026-09-26, hub, 10 sembol): 8 quant sinyalinin HEPSİNDE
        # p_win = 0,00; aynı anda TA sinyali 0,28. Bu bir olasılık değil dağılım-dışı doyumdur —
        # gölgede bile loglanırsa kanıt döngüsünü (gate_sim / pwin_calibration) kirletir.
        final.metadata["meta_gate"] = "KAPSAM_DISI_QUANT"
        gate_mode = "off"
    if gate_mode != "off":
        try:
            from nihai.signals.meta_gate import get_gate, note_error
            gate = get_gate()
            # Tur-43: paylaşımlı gösterge önbelleği kapıya da geçer — kapı EMA/ATR/BB/
            # RSI/ADX'i sıfırdan hesaplıyordu (bkz. meta_features BULGU-43-1).
            p_win = gate.probability(df_htf, final, shared=shared)
            if p_win is not None:
                thr = live_threshold if live_threshold is not None else (_sc.meta_gate_threshold or gate.threshold)
                final.metadata["meta_p_win"] = round(p_win, 3)
                final.metadata["meta_model_id"] = getattr(gate, "model_id", None)
                final.metadata["meta_strategy"] = strategy_mode
                if p_win < thr:
                    if gate_mode == "enforce":
                        # Elenen YÖN metadata'da saklanır (meta_gated): prediction_tracker
                        # kapı-engelli adayın p_win + tarafını da kaydedebilsin. Bu olmadan
                        # canlı kanıt yalnız eşik-ÜSTÜ sinyallerden birikir (seçilim yanlılığı)
                        # ve kapının işe yarayıp yaramadığı canlıda ASLA ölçülemez.
                        final.metadata["meta_gated"] = action.value
                        return Signal(SignalAction.WAIT, symbol, price, stop, target,
                                      reasons=reasons + [f"META_GATE_P{p_win:.2f}<{thr:.2f}"],
                                      quality_score=score, rr_ratio=rr,
                                      metadata=final.metadata)
                    # GÖLGE: sinyale DOKUNMA (aksiyon/stop/hedef/skor aynı) — yalnız işaretle.
                    # Sinyal LONG/SHORT kaldığı için _signal_pwin onu meta_p_win + tarafıyla
                    # zaten loglar; meta_shadow_gated 'enforce olsaydı elenirdi' izidir.
                    final.metadata["meta_shadow_gated"] = action.value
                    final.reasons.append(f"META_SHADOW_P{p_win:.2f}<{thr:.2f}")
                else:
                    final.reasons.append(f"META_P{p_win:.2f}")
        except Exception as exc:
            # BULGU-44 (Tur-44): ZARİF degrade doğru tasarım, SESSİZ degrade değil.
            # Bu blok kapı çağrısını sarar; `probability` kendi içindeki hatayı zaten
            # defterine yazar ama İMZA uyuşmazlığı (TypeError) burada, gate'in try'ının
            # DIŞINDA patlar — Tur-43'te tam bu oldu ve kapı sessizce pasifleşti.
            # Sinyal yine BLOKE EDİLMEZ; yalnız olan biten görünür kılınır: sinyalin
            # kendi gerekçesine iz düşer (prediction_tracker/pano onu okur) ve defter
            # ilk hatayı UYARI olarak loglar.
            final.reasons.append("META_GATE_ERROR")
            final.metadata["meta_gate_error"] = f"{type(exc).__name__}: {exc}"[:120]
            try:
                from nihai.signals.meta_gate import note_error
                note_error(exc, "generate_signal")
            except Exception:
                pass   # meta_gate hiç import edilemiyorsa iz zaten sinyalde duruyor

    return final
