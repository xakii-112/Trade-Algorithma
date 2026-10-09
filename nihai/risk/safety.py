"""
safety.py — Bot-düzeyi devre kesici (fat-finger), kaçak sinyal freni.

PDF Referansları:
- Madde 42 : Kaçak emir freni (dakikada/saatte max emir, sembol başına max açık)
- Madde 47 : Acil manuel müdahale hook'u (flatten-all yerine "sinyalleri durdur")

Sinyal-only sistemde "emir" yerine "sinyal akışı" frenleyici.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock

import numpy as np
from loguru import logger

from nihai.config import CONFIG


class CircuitBreaker:
    def __init__(self):
        self._lock = Lock()
        self._signal_times: deque[datetime] = deque()
        self._tripped: bool = False
        self._trip_reason: str = ""

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    def trip(self, reason: str) -> None:
        with self._lock:
            self._tripped = True
            self._trip_reason = reason
            logger.error(f"[safety] DEVRE KESİCİ TRİPLE — {reason}")

    def reset(self) -> None:
        with self._lock:
            self._tripped = False
            self._trip_reason = ""
            self._signal_times.clear()
            logger.info("[safety] devre kesici sıfırlandı")

    def can_emit(self) -> tuple[bool, str]:
        """Sinyal yayılabilir mi? İKİ AYRI mekanizma vardır ve karıştırılmamalıdır:

        - **Hız sınırı (Madde 42)** GEÇİCİDİR: pencere kendi kendini temizler, koşul 60 sn
          (ya da 1 saat) içinde kendiliğinden geçer. Bu yüzden MANDALLAMAZ — yalnız o
          çağrıyı reddeder.
        - **Devre kesici (Madde 47)** KALICIDIR: yalnız `trip()` kurar, yalnız `reset()`
          temizler. Operatörün bilinçli "sinyalleri durdur" kararıdır.

        Tur-50 öncesi hız sınırı da `_tripped`'i kuruyordu; geçici bir koşul kalıcı bir
        sonuç doğuruyor ve `reset()` hiçbir yerden çağrılmadığı için süreç ömrü boyunca
        MANDALLI kalıyordu. Canlı karar loglarında ölçüldü: 6 gerçek tetiklemeye karşı
        9.194 mandallı blok (%99,93)."""
        now = datetime.now()
        with self._lock:
            if self._tripped:
                return False, f"TRIPPED: {self._trip_reason}"
            # Pencereyi temizle
            cutoff = now - timedelta(hours=1)
            while self._signal_times and self._signal_times[0] < cutoff:
                self._signal_times.popleft()

            per_min = sum(1 for t in self._signal_times if t > now - timedelta(minutes=1))
            per_hr = len(self._signal_times)

            # Mandal YOK: pencere daraldığında sinyal kendiliğinden yeniden akar.
            if per_min >= CONFIG.safety.max_signals_per_minute:
                return False, f"RATE_LIMIT_MIN_{per_min}"
            if per_hr >= CONFIG.safety.max_signals_per_hour:
                return False, f"RATE_LIMIT_HOUR_{per_hr}"
        return True, "OK"

    def record_emit(self) -> None:
        with self._lock:
            self._signal_times.append(datetime.now())


# =====================================================================================
# Portföy VaR / CVaR (A3) — risk RAPORLAMA katmanı (sinyal-only; emir/boyut KISMAZ).
#
# İki bağımsız yöntem, ikisi de açık pozisyonların DAYANAK getirilerinden beslenir:
#   1) Tarihsel simülasyon  — geçmiş günlük portföy P&L dağılımının sol kuyruğu.
#   2) Monte Carlo          — getirilerden tahmin edilen ort/kovaryans ile çok-değişkenli
#                              normal örnekleme (korelasyonu korur).
# CVaR (beklenen kuyruk kaybı / expected shortfall) her ikisi için raporlanır.
#
# DÜRÜSTLÜK NOTU (docs/EDGE_FINDINGS.md ilkesi): VaR bir KESTİRİM değil, geçmiş
# oynaklığın özetidir. Normal varsayımı kuyruğu hafife alır; opsiyonlar delta ile
# DOĞRUSAL yaklaşıkdır (gamma/vega yok). Alfa iddiası taşımaz — yalnız maruziyet ölçer.
# =====================================================================================

# Güven seviyesi → sözlük anahtarı ("95"/"99")
def _lvl_key(level: float) -> str:
    return f"{level * 100:.0f}"


@dataclass
class VaRResult:
    """Tek yöntemin VaR/CVaR çıktısı (pozitif $ = beklenen KAYIP büyüklüğü)."""
    method: str
    horizon_days: int
    n_samples: int
    var: dict[str, float] = field(default_factory=dict)   # {"95": $, "99": $}
    cvar: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "horizon_days": self.horizon_days,
            "n_samples": self.n_samples,
            "var": self.var,
            "cvar": self.cvar,
            "note": self.note,
        }


def _tail_var_cvar(pnl: np.ndarray, levels: Sequence[float], scale: float
                   ) -> tuple[dict[str, float], dict[str, float]]:
    """Bir P&L örnek dizisinin sol kuyruğundan VaR ve CVaR (her ikisi de pozitif $)."""
    var: dict[str, float] = {}
    cvar: dict[str, float] = {}
    for lv in levels:
        q = float(np.percentile(pnl, (1.0 - lv) * 100.0))   # kayıp kuyruğu kuantili (≤0 beklenir)
        var[_lvl_key(lv)] = max(0.0, -q * scale)
        tail = pnl[pnl <= q]
        es = float(tail.mean()) if tail.size else q
        cvar[_lvl_key(lv)] = max(0.0, -es * scale)
    return var, cvar


def historical_var(pnl_samples: Sequence[float], *, levels: Sequence[float] = (0.95, 0.99),
                   horizon_days: int = 1) -> VaRResult:
    """Tarihsel simülasyon VaR/CVaR. pnl_samples = 1-günlük portföy P&L ($) serisi."""
    a = np.asarray([x for x in pnl_samples if x is not None and x == x], dtype=float)
    if a.size < 2:
        return VaRResult("historical", horizon_days, int(a.size),
                         note="yetersiz örnek (<2)")
    scale = math.sqrt(max(1, horizon_days))
    var, cvar = _tail_var_cvar(a, levels, scale)
    return VaRResult("historical", horizon_days, int(a.size), var, cvar)


def monte_carlo_var(returns_matrix: Sequence[Sequence[float]] | np.ndarray,
                    exposures: Sequence[float] | np.ndarray, *,
                    levels: Sequence[float] = (0.95, 0.99), horizon_days: int = 1,
                    n_sims: int = 10000, seed: int | None = None) -> VaRResult:
    """Monte Carlo VaR/CVaR. returns_matrix: (T, K) varlık-bazlı günlük getiriler;
    exposures: (K,) işaretli $ maruziyet (LONG +, SHORT −). Getirilerden ort/kovaryans
    tahmin edilip çok-değişkenli normalden n_sims senaryo çekilir."""
    R = np.asarray(returns_matrix, dtype=float)
    w = np.asarray(exposures, dtype=float)
    if R.ndim != 2 or R.shape[0] < 2 or R.shape[1] == 0 or w.size != R.shape[1]:
        return VaRResult("monte_carlo", horizon_days, 0, note="geçersiz/yetersiz girdi")
    mu = np.nanmean(R, axis=0)
    cov = np.atleast_2d(np.cov(R, rowvar=False))
    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(mu * horizon_days, cov * horizon_days, size=n_sims)
    pnl = sims @ w                       # horizon zaten ölçeklendi → scale=1
    var, cvar = _tail_var_cvar(pnl, levels, scale=1.0)
    return VaRResult("monte_carlo", horizon_days, int(n_sims), var, cvar)


def _position_exposure(pos, underlying_px: float) -> float:
    """İşaretli $ maruziyet. Hisse = yön·adet·fiyat; opsiyon = yön·delta·100·adet·dayanak
    (delta-DOĞRUSAL yaklaşık, greeks yoksa |delta|=0.5 varsayılır)."""
    sign = 1.0 if str(getattr(pos, "side", "LONG")).upper() == "LONG" else -1.0
    qty = float(getattr(pos, "quantity", 0) or 0)
    if getattr(pos, "asset_type", "stock") == "option":
        gk = getattr(pos, "greeks", None)
        delta = float(getattr(gk, "delta", 0.0)) if gk else (0.5 * sign)
        return delta * 100.0 * qty * underlying_px   # delta zaten işaretli (short put/call dahil)
    return sign * qty * underlying_px


def compute_portfolio_var(positions, provider, *, interval: str = "1d", lookback: int = 120,
                          levels: Sequence[float] = (0.95, 0.99), horizon_days: int = 1,
                          n_sims: int = 10000, seed: int | None = None) -> dict:
    """Açık pozisyonlardan portföy VaR/CVaR (tarihsel + Monte Carlo).

    Her pozisyonun DAYANAK günlük getirileri provider.get_bars(symbol, interval, lookback)
    ile çekilir, ortak tarih ekseninde hizalanır; işaretli $ maruziyetle çarpılıp portföy
    P&L dağılımı kurulur. Korelasyon otomatik korunur (aynı zaman ekseni)."""
    pos_list = list(positions or [])
    base = {
        "ok": False, "horizon_days": horizon_days, "levels": [_lvl_key(x) for x in levels],
        "portfolio_value": 0.0, "n_positions": len(pos_list),
        "historical": None, "monte_carlo": None, "exposures": [], "note": "",
    }
    if not pos_list:
        base["note"] = "açık pozisyon yok"
        return base

    series: dict[str, np.ndarray] = {}   # symbol → günlük getiri dizisi
    last_px: dict[str, float] = {}
    for pos in pos_list:
        sym = getattr(pos, "symbol", None)
        if not sym or sym in series:
            continue
        try:
            df = provider.get_bars(sym, interval, lookback)
        except Exception as e:
            logger.debug(f"[var] {sym} bar alınamadı: {e}")
            df = None
        if df is None or getattr(df, "empty", True) or len(df) < 3:
            continue
        close = df["close"].astype(float)
        rets = close.pct_change().dropna().to_numpy()
        if rets.size >= 2:
            series[sym] = rets
            last_px[sym] = float(close.iloc[-1])

    if not series:
        base["note"] = "dayanak getiri verisi alınamadı"
        return base

    # Tüm getiri dizilerini en kısa ortak uzunlukta (en yeni gözlemler) hizala
    min_len = min(arr.size for arr in series.values())
    symbols = list(series.keys())
    R = np.column_stack([series[s][-min_len:] for s in symbols])   # (min_len, K)

    exposures = []
    pv = 0.0
    for s in symbols:
        # Aynı dayanağa birden çok pozisyon varsa maruziyetleri topla
        exp = sum(_position_exposure(p, last_px[s])
                  for p in pos_list if getattr(p, "symbol", None) == s)
        exposures.append(exp)
        pv += abs(exp)
    w = np.asarray(exposures, dtype=float)

    portfolio_pnl = R @ w                       # (min_len,) günlük portföy P&L ($)
    hist = historical_var(portfolio_pnl, levels=levels, horizon_days=horizon_days)
    mc = monte_carlo_var(R, w, levels=levels, horizon_days=horizon_days,
                         n_sims=n_sims, seed=seed)

    base.update({
        "ok": True,
        "portfolio_value": round(pv, 2),
        "historical": hist.as_dict(),
        "monte_carlo": mc.as_dict(),
        "exposures": [{"symbol": s, "exposure": round(float(e), 2)}
                      for s, e in zip(symbols, exposures, strict=True)],
        "note": "delta-doğrusal yaklaşık; normal kuyruğu hafife alabilir",
    })
    return base
