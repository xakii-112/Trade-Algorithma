"""
backtest.py — Event-driven backtest motoru + Monte Carlo yol analizi.

PDF Referansları:
- Madde 3  : Look-ahead koruması (bar i'de yalnızca i-1'e kadar)
- Madde 4  : Gap simülasyonu (kapanış→açılış)
- Madde 5  : Granüler test (HTF yön + 1m granül kontrol — bu sürümde HTF tek başına)
- Madde 15 : Monte Carlo / yol bağımlılığı
- Madde 38 : Tamamlanan her işlem post-mortem (otomatik, position_manager)
- Madde 50 : Slipaj modellenir
- Madde 51 : Survivorship bias uyarısı (yfinance batanları içermez — README'de not)
- Madde 52 : Backtest → Paper → Canlı kapısı (bu motor sadece backtest)

Çalıştırma:
    python backtest.py --symbol AAPL --interval 1h --period 2y
    python backtest.py --symbol SPY MSFT --interval 1d --from 2023-01-01 --to 2024-12-31
    python backtest.py --symbol AAPL --interval 1h --period 1y --mc 1000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TypedDict, cast

# Windows console (cp1254/cp1252) Unicode kelepçesi → stdout'u UTF-8'e ayarla
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from nihai.config import CONFIG, DATA_PROVIDER, INITIAL_EQUITY, LOG_DIR, get_sector
from nihai.data.data_provider import Interval, get_provider
from nihai.risk.position_manager import CloseEvent, PositionManager, SlippageModel
from nihai.risk.risk_manager import PortfolioState, RiskManager
from nihai.signals.strategy import (
    BASE_STRATEGY_MODES,
    STRATEGY_MODES,
    SignalAction,
    _SharedIndicators,
    generate_signal,
    quant_capala,
)

# A4 — backtest'te piyasa rejimi çarpanı için benchmark serisi önbelleği (process-ömrü).
# (benchmark, interval, period) başına bir kez yüklenir → 600-sembol taramada worker
# başına tek SPY yüklemesi. Yalnızca CONFIG.regime.enable_regime_risk açıkken kullanılır.
_REGIME_SERIES_CACHE: dict = {}


def _regime_series(period: str | None, provider_name: str | None):
    """A4 — backtest için benchmark rejim-çarpanı serisi (önbellekli). Hata/veri yoksa None."""
    cfg = CONFIG.regime
    key = (cfg.benchmark, cfg.benchmark_interval, period or "")
    if key in _REGIME_SERIES_CACHE:
        return _REGIME_SERIES_CACHE[key]
    series = None
    try:
        from nihai.signals.market_regime import regime_multiplier_series
        bench = load_bars(cfg.benchmark, cfg.benchmark_interval, period=period,
                          provider_name=provider_name)
        s = regime_multiplier_series(bench)
        if s is not None and not s.empty:
            # tz/normalize-güvenli arama için indeksi tz-naive GÜN'e indir. Benchmark
            # günlük (tz-naive) iken sembol intraday (tz-aware) olabilir; karıştırmak
            # asof'u patlatır ve rejim sessizce uygulanmazdı (no-op bug).
            idx = s.index
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            s.index = idx.normalize()
            series = s[~s.index.duplicated(keep="last")].sort_index()
    except Exception as e:
        logger.debug(f"[bt-regime] {cfg.benchmark} serisi yüklenemedi: {e}")
    _REGIME_SERIES_CACHE[key] = series
    return series


def _regime_mult_at(series, ts) -> float:
    """A4 — ts'in 1 gün ÖNCESİNE ait rejim çarpanı (look-ahead-güvenli), tz/normalize-güvenli.
    Seri boş/erken → neutral_mult; beklenmeyen hata → 1.0 (etkisiz, asla çökmez)."""
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_localize(None)
        m = series.asof(t.normalize() - pd.Timedelta(days=1))
        return float(m) if m == m else CONFIG.regime.neutral_mult   # m==m: NaN değil
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Veri yükleme (backtest, yfinance üzerinden)
# ---------------------------------------------------------------------------
def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Bar index'ini tz-naive yap. Backtest motoru ve sistem geneli naive
    datetime.now() kullanır; tz-aware index 'offset-naive vs offset-aware'
    çıkarma hatalarına yol açar (örn. exit_time - entry_time)."""
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def _period_to_bars(period: str | None, interval: str) -> int:
    """yfinance tarzı period ('6mo','2y','60d') + bar aralığını yaklaşık bar sayısına
    çevirir (n-bar tabanlı sağlayıcılar: hub/alpaca için). Bolca tahmin eder; fazla
    çekilen barlar load_bars'ta start/end/period ile zaten kırpılır."""
    from nihai.config import INTERVAL_MINUTES
    if not period:
        return 2000
    import re
    m = re.match(r"^(\d+)(d|mo|y)$", period.strip().lower())
    if not m:
        return 2000
    days = int(m.group(1)) * {"d": 1, "mo": 30, "y": 365}[m.group(2)]
    bars_per_day = 390 / INTERVAL_MINUTES.get(interval, 60) if interval != "1d" else 1
    # ~0.7 takvim→işlem günü oranı + %20 emniyet payı
    est = int(days * 0.7 * bars_per_day * 1.2)
    return max(50, min(est, 10000))


# D6 — backtest hub/alpaca sağlayıcısı (with_cache=False) selection başına BİR kez kurulur.
# Eskiden her sembolde yeniden kuruluyordu → 600-sembol taramada sembol başına yeni
# httpx.Client/session. ProcessPool (varsayılan motor) her worker'da kendi cache'ini tutar
# → süreç-içi tek sağlayıcı; walk_forward/portföy (tek-thread) de paylaşır. Yalnız sağlayıcı
# NESNESİ paylaşılır, bar cache DEĞİL (with_cache=False korunur → barlar birebir aynı).
# httpx.Client eşzamanlı isteklerde thread-safe'tir; lock yalnız çift-kurulum yarışını önler.
_BT_PROVIDER_CACHE: dict[str, object] = {}
_BT_PROVIDER_LOCK = threading.Lock()


def _bt_provider(selection: str):
    """selection için önbellekli backtest sağlayıcısı (with_failover=True, with_cache=False)."""
    prov = _BT_PROVIDER_CACHE.get(selection)
    if prov is None:
        with _BT_PROVIDER_LOCK:
            prov = _BT_PROVIDER_CACHE.get(selection)
            if prov is None:
                prov = get_provider(selection, with_failover=True, with_cache=False)
                _BT_PROVIDER_CACHE[selection] = prov
    return prov


def load_bars(symbol: str, interval: str, period: str | None = None,
              start: str | None = None, end: str | None = None,
              provider_name: str | None = None) -> pd.DataFrame:
    """
    Backtest için tarihsel bar yükler. provider_name verilirse o sağlayıcıyı kullanır,
    yoksa env'deki DATA_PROVIDER (yfinance | tradingview).

    NOT: yfinance native period/start-end destekler. TradingView (tvDatafeed) yalnızca
    n_bars destekler — geniş pencere çekilir, sonra tarih filtresi uygulanır.
    """
    selection = (provider_name or DATA_PROVIDER).lower()

    if selection in ("tradingview", "tv"):
        import re
        from datetime import timedelta
        provider = get_provider("tradingview")
        # TV'de date range yok — bolca bar çek, sonra filtrele
        max_bars = {"1m": 5000, "5m": 5000, "15m": 5000, "1h": 5000, "1d": 5000}.get(interval, 2000)
        df = provider.get_bars(symbol, cast(Interval, interval), lookback_bars=max_bars)
        if df.empty:
            return df
        # Tarih filtresi (start/end)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        # Period filtresi (yfinance tarzı: 5d, 6mo, 2y) — TV'de native yok, çevir
        if period and not start:
            m = re.match(r"^(\d+)(d|mo|y)$", period.strip().lower())
            if m:
                n, unit = int(m.group(1)), m.group(2)
                days = n * {"d": 1, "mo": 30, "y": 365}[unit]
                cutoff = pd.Timestamp.now(tz=df.index.tz) - timedelta(days=days)
                df = df[df.index >= cutoff]
        df = df.dropna(subset=["open", "high", "low", "close"])
        return _strip_tz(df)

    # alpaca_hist: araştırma-kalitesi tarihsel çekim (SIP feed, start/end native,
    # delist edilmiş semboller dahil, disk-cache'li). Eğitim seti üretimi için —
    # canlı n-bar sağlayıcısının aksine gerçek tarih aralığı ve tam-piyasa hacmi verir.
    if selection in ("alpaca_hist", "alpaca_history"):
        from datetime import timedelta

        from nihai.data.alpaca_history import get_history
        if not start and period:
            import re
            m = re.match(r"^(\d+)(d|mo|y)$", period.strip().lower())
            days = (int(m.group(1)) * {"d": 1, "mo": 30, "y": 365}[m.group(2)]) if m else 365
            start = (pd.Timestamp.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = get_history(symbol, cast(Interval, interval), start=start or "2020-01-01", end=end)
        return df.dropna(subset=["open", "high", "low", "close"]) if not df.empty else df

    # hub / alpaca / alpaca_stream: DataProvider.get_bars (n-bar tabanlı) üzerinden çek.
    # yfinance gibi native period/start-end yok → period'u bar sayısına çevir, bolca
    # bar iste, sonra start/end ile filtrele. Bu sağlayıcılar eşzamanlı isteklere
    # dayanıklıdır → paralel taramanın asıl hızlandığı yol (yf 429'a girmeden).
    if selection in ("hub", "alpaca", "alpaca_stream", "alpaca_ws"):
        from datetime import timedelta
        # failover açık: hub/alpaca erişilemezse yfinance'e düş (canlı davranışla tutarlı).
        # cache kapalı: tarama her sembolün barını bir kez okur; 600 DataFrame'i cache'te
        # tutmak gereksiz RAM şişmesi olur.
        provider = _bt_provider(selection)   # D6: selection başına tek sağlayıcı (cache'li)
        n_bars = _period_to_bars(period, interval)
        df = provider.get_bars(symbol, cast(Interval, interval), lookback_bars=n_bars)
        if df.empty:
            return df
        df = _strip_tz(df)   # naive index → naive karşılaştırma (tüm filtreler güvenli)
        df.columns = [c.lower() for c in df.columns]
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        if period and not start:
            import re
            m = re.match(r"^(\d+)(d|mo|y)$", period.strip().lower())
            if m:
                days = int(m.group(1)) * {"d": 1, "mo": 30, "y": 365}[m.group(2)]
                cutoff = pd.Timestamp.now() - timedelta(days=days)
                df = df[df.index >= cutoff]
        return df.dropna(subset=["open", "high", "low", "close"])

    # Default: yfinance (native period/start-end). threads=False: tek sembol, kontrolsüz
    # thread pool yaratmasın (TierScanner/uvicorn thread'leriyle çakışma önlenir).
    kw = {"interval": interval, "auto_adjust": True, "progress": False, "prepost": False, "threads": False}
    if start:
        df = yf.download(symbol, start=start, end=end, **kw)
    else:
        df = yf.download(symbol, period=period or "1y", **kw)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna(subset=["open", "high", "low", "close"])
    return _strip_tz(df)


# ---------------------------------------------------------------------------
# Backtest sonuçları
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    symbol: str
    interval: str
    bars_total: int
    trades_count: int
    wins: int
    losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float
    final_equity: float | None
    initial_equity: float
    total_return_pct: float | None
    max_drawdown_pct: float | None
    sharpe_annualized: float | None
    # P0-4 — Sortino + profit_factor: mevcut alanların ARDINDAN eklendi (yeniden adlandırma/
    # yeniden hesaplama yok); varsayılanlı → eski pozisyonel BacktestResult(...) çağrıları
    # (örn. boş-veri erken-dönüşleri) değişmeden çalışmaya devam eder.
    sortino_annualized: float | None = 0.0
    profit_factor: float | None = None
    trade_pnls: list[float] = field(default_factory=list)
    equity_curve: Sequence[float | None] = field(default_factory=list)
    monte_carlo: dict | None = None
    # BULGU-59-2 — intra-bar SIRA belirsizliğinin MARUZİYETİ (kaç çıkış varsayıma dayandı).
    # `ambiguous` = aynı barda hem stop hem hedef delindi (açılış ikisinin arasında);
    # `resolved`/`flipped` yalnız 1m alt-bar çözücüsü bağlıyken dolar. Sayı 0'sa varsayım
    # hiç devreye girmemiştir; büyükse sonuç KÖTÜMSER bir varsayımın altındadır.
    intrabar_ambiguous: int = 0
    intrabar_resolved: int = 0
    intrabar_flipped: int = 0
    option_data: dict | None = None
    execution_costs: dict | None = None
    # Tur-95 (BULGU-95-3) — kill-switch KALICIDIR: tetiklendiği bardan sonra yeni giriş yok.
    # Varsayılan portföy koşularında 35-50. işlemde tetiklenip karar barlarının %93-96'sını
    # girişsiz bırakıyordu; getiri/Sharpe/MDD o zaman "dönem" değil "ilk birkaç ay" ölçüsüdür.
    # `kill_switch_at` girişin İLK engellendiği karar barı, `kill_switch_idle_pct` girişsiz
    # kalan karar barlarının payı. None = tetiklenmedi (ya da kapalı).
    kill_switch_at: str | None = None
    kill_switch_idle_pct: float | None = None

    def short_summary(self) -> str:
        pf_txt = f"{self.profit_factor:.2f}" if self.profit_factor is not None else "N/A"
        amb = ""
        if self.intrabar_ambiguous:
            pay = (f"/{self.trades_count}" if self.trades_count else "")
            amb = f" intrabar_belirsiz={self.intrabar_ambiguous}{pay}"
            if self.intrabar_resolved:
                amb += f" (çözülen={self.intrabar_resolved}, çevrilen={self.intrabar_flipped})"
        ks = (f" KILL_SWITCH@{self.kill_switch_at} (girişsiz %{self.kill_switch_idle_pct:.1f})"
              if self.kill_switch_at is not None and self.kill_switch_idle_pct is not None else "")
        return (f"{self.symbol} {self.interval} | trades={self.trades_count} "
                f"WR={self.win_rate*100:.1f}% E={self.expectancy:+.2f} "
                f"Ret={_metric_text(self.total_return_pct)}% MDD={_metric_text(self.max_drawdown_pct)}% "
                f"Sharpe={_metric_text(self.sharpe_annualized)} Sortino={_metric_text(self.sortino_annualized)} "
                f"PF={pf_txt}{amb}{ks}")


def _metric_text(value: float | None) -> str:
    return f"{value:,.2f}" if value is not None else "N/A"


# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------
def max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    mdd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd * 100


def sharpe_annualized(returns: list[float], periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    mean = statistics.mean(returns)
    sd = statistics.pstdev(returns)
    if sd == 0:
        return 0.0
    return (mean / sd) * (periods_per_year ** 0.5)


def sortino_annualized(returns: list[float], periods_per_year: int = 252,
                       target: float = 0.0) -> float:
    """Sortino oranı — `sharpe_annualized` ile AYNI TABAN: bar-başı getiri serisi
    (returns; işlem-bazlı DEĞİL) ve aynı `periods_per_year` annualizasyon faktörü.

    Downside deviation empyrical (`downside_risk`) konvansiyonu ile birebir: hedefin
    ALTINDAKİ sapmaların karesi TÜM seri N'e bölünür — yalnızca hedefin altında kalan
    bar sayısına değil. Yani hedefin ÜSTÜNDEKİ barlar paydaya sıfır katkı ile girer ve
    onu SULANDIRIR (küçültmez). Formül (matematiksel olarak empyrical.sortino_ratio ile
    özdeş — empyrical payı ann_factor'la payda sqrt(ann_factor)'la ölçekler, net etki
    aşağıdaki tek sqrt'tir; bkz. tests/test_backtest_metrics_empyrical.py):

        downside_dev = sqrt( mean( min(r - target, 0)^2 ) )     # N'e bölü, yıllıksız
        sortino      = (mean(returns) - target) / downside_dev * sqrt(periods_per_year)

    downside_dev==0 (hedefin altına hiç inilmemiş) durumunda Sharpe ile AYNI sözleşme:
    0.0 döner. empyrical bu durumda inf/nan üretir (numpy sessiz bölme); biz JSON/log
    güvenliği için sonlu değeri tercih ederiz (bkz. `profit_factor`'ın None kararı — aynı
    gerekçe: sonsuz/NaN değer JSON'u ve tüketicileri bozar)."""
    if len(returns) < 2:
        return 0.0
    n = len(returns)
    mean_adj = statistics.mean(returns) - target
    downside_sq_mean = sum(min(r - target, 0.0) ** 2 for r in returns) / n
    downside_dev = downside_sq_mean ** 0.5
    if downside_dev == 0:
        return 0.0
    return (mean_adj / downside_dev) * (periods_per_year ** 0.5)


def profit_factor(trade_pnls: list[float]) -> float | None:
    """Brüt kâr / brüt zarar (kapanan İŞLEMLERİN gerçekleşen PnL'i — trade_pnls; bar
    getirisi DEĞİL). Hiç zarar yoksa (hiç işlem yok VEYA tüm işlemler kazançlı/başabaş)
    None döner: inf değeri JSON'da geçersiz "Infinity" token'ı üretir (RFC 8259 dışı) ve
    gate.py/web/telegram gibi tüketicilerde sessizce bozulabilir — None, 'tanımsız'ı
    açıkça ifade eder ve JSON'da 'null' olarak güvenle serileşir."""
    gross_win = sum(p for p in trade_pnls if p > 0)
    gross_loss = -sum(p for p in trade_pnls if p < 0)
    if gross_loss == 0:
        return None
    return gross_win / gross_loss


# ---------------------------------------------------------------------------
# Monte Carlo — Madde 15
# ---------------------------------------------------------------------------
def monte_carlo_paths(trade_pnls: list[float], initial_equity: float,
                      n_iters: int = 1000, seed: int = 42, *, include_bands: bool = False) -> dict:
    """Trade PnL'lerini karıştırarak binlerce equity yolu üret, MDD/final dağılımı raporla.

    Katalog #21 (Tur-67): eğri kurma + MDD artık YOL BAŞINA Python döngüsü değil, tüm
    yolları taşıyan tek matriste vektörel. ÖLÇÜLDÜ (bu makine, p50): 200 işlem × 1000
    yineleme **180,05 → 70,48 ms (2,55×)**; 200×5000 **891,44 → 356,26 ms (2,50×)**.

    **Çıktı BİT-DENKTİR** ve öyle kalmak zorundadır (testle kilitli):

    * `rng.shuffle` sırası KORUNDU — karıştırma hâlâ `random.Random(seed)` ile yol yol
      yapılır. Karıştırmayı numpy'a devretmek RNG akışını değiştirir ve aynı seed farklı
      sayı verirdi; ölçüldü, karıştırma toplam sürenin yalnız **%33'ü** (59,68/187,16) —
      geri kalan %67 zaten vektörleşiyor, akışı bozmaya gerek YOK.
    * `initial_equity` diziye **İLK ELEMAN** olarak konur, cumsum'dan SONRA eklenmez.
      Sıra önemlidir: önce PnL'leri toplayıp sonra sermayeyi eklemek TOPLAMA SIRASINI
      değiştirir ve son bitlerde ayrışır (ölçüldü: maks |ΔMDD| 8,04e-14, `final` eşit
      DEĞİL). Sermaye ilk elemanken `cumsum` birebir `curve[-1] + pnl` sırasını izler.
      (`cumsum` her öneki üretmek zorunda olduğu için ardışıktır; `sum`ın pairwise
      toplaması burada devreye girmez.)

    **Tur-95 (BULGU-95-2) — karıştırma YALNIZ SIRAYA bağlı ölçüleri dağıtır.** Her yol aynı
    PnL kümesini taşır → `final_equity_p05/p50/p95` yapı gereği AYNI sayıdır (sermaye +
    ΣPnL); bir dağılım değil, tek değerdir. Eski `ruin_probability` bu nihai değerden
    (`final <= 0`) sayıldığı için ya 0 ya 1 çıkıyor, bir OLASILIK ölçmüyordu — ve Madde 52
    kapısı (`gate.MC_RUIN_LOW`) ile tarayıcı iflas filtresi tam bu alanı okuyor. Artık
    `ruin_probability` = sermayenin yolun HERHANGİ bir noktasında <= 0'a değdiği yolların
    payı (sıraya bağlı; nihai-iflas yolları da içinde → eskisinden asla KÜÇÜK değil).
    """
    if not trade_pnls:
        return {}
    rng = random.Random(seed)
    n_tr = len(trade_pnls)
    # Satır i = i'inci yolun [sermaye, pnl_1 … pnl_n] dizisi.
    mat = np.empty((n_iters, n_tr + 1), dtype=float)
    mat[:, 0] = initial_equity
    for i in range(n_iters):
        order = trade_pnls.copy()
        rng.shuffle(order)
        mat[i, 1:] = order
    curves = np.cumsum(mat, axis=1)                 # eğriler (sermaye dahil)
    peaks = np.maximum.accumulate(curves, axis=1)
    dd_mat = np.where(peaks > 0, (peaks - curves) / peaks, 0.0)
    final_equities: list[float] = curves[:, -1].tolist()
    drawdowns: list[float] = (dd_mat.max(axis=1) * 100.0).tolist()
    ruin_path = float(np.mean(np.any(curves <= 0, axis=1)))

    final_equities.sort()
    drawdowns.sort()

    def pct(arr, p):
        return arr[int(len(arr) * p)]

    result = {
        "iterations": n_iters,
        "final_equity_p05": pct(final_equities, 0.05),
        "final_equity_p50": pct(final_equities, 0.50),
        "final_equity_p95": pct(final_equities, 0.95),
        "max_drawdown_p05": pct(drawdowns, 0.05),
        "max_drawdown_p50": pct(drawdowns, 0.50),
        "max_drawdown_p95": pct(drawdowns, 0.95),
        "ruin_probability": ruin_path,
    }
    if include_bands:
        result["bands"] = dict(zip(("p05", "p50", "p95"),
                                    np.quantile(curves, [0.05, 0.5, 0.95], axis=0).tolist(), strict=True))
        result["ruin_during_path_probability"] = ruin_path   # eski ad (laboratuvar uyumu)
    return result


# ---------------------------------------------------------------------------
# Ana backtest döngüsü
# ---------------------------------------------------------------------------
def _bt_slippage() -> SlippageModel:
    """Backtest slipaj/komisyon modeli — config'den (Madde 32, 50)."""
    bc = CONFIG.backtest
    return SlippageModel(
        stock_slippage_pct=bc.stock_slippage_pct,
        option_slippage_pct=bc.option_slippage_pct,
        stock_commission_per_trade=bc.stock_commission_per_trade,
        impact_coef=bc.impact_coef,
    )


def _causal_adv(volume: np.ndarray, window: int) -> np.ndarray:
    """Nedensel (karar barına KADAR, bar dahil) hareketli ortalama hacim — BULGU-59-3.

    `pandas.rolling(window).mean()` ile O(n). Isınma
    barlarında (i+1 < window) o ana kadarki ortalama kullanılır (NaN yerine kısmi
    pencere: ADV'nin YOKLUĞU etkiyi kapatırdı, kısmi pencere ise gerçek bir ölçümdür).
    Sonuç dizisi i konumunda [0..i] barlarının ortalamasını taşır → i+1'deki infazda
    look-ahead YOKTUR."""
    v = np.asarray(volume, dtype=float)
    if v.ndim != 1 or not isinstance(window, int) or window <= 0:
        raise ValueError("Hacim tek boyutlu, pencere pozitif tamsayı olmalı")
    invalid = ~np.isfinite(v) | (v < 0)
    result = pd.Series(v).rolling(window, min_periods=1).mean().to_numpy()
    bad_window = pd.Series(invalid).rolling(window, min_periods=1).max().to_numpy(dtype=bool)
    return np.where(bad_window, np.nan, result)  # eksik hacim sıfır hacim değildir


def _bar_sigma(shared_full, i: int, price: float) -> float | None:
    """Bar i'deki ATR/fiyat = bar-başı oynaklık kesri (BULGU-59-3 etkisinin σ'sı).

    `_SharedIndicators` ZATEN kurulu (Tur-7 önbelleği) → ek hesap yok. Ölçülemezse
    (NaN/0 fiyat) None döner: etki terimi uygulanmaz, taban kayması kalır."""
    try:
        atr = float(shared_full.atr14_np()[i])
    except (IndexError, TypeError, ValueError, AttributeError):
        return None
    if not math.isfinite(price) or price <= 0 or not math.isfinite(atr) or atr < 0:
        return None
    return atr / price


def _mtm_equity(portfolio) -> float:
    """Piyasaya-göre-işaretlenmiş (mark-to-market) equity = realized equity + açık
    pozisyonların gerçekleşmemiş PnL'i (son bar kapanışından, last_mark).

    DOĞRULUK: portfolio.equity yalnızca KAPANIŞTA güncellenir (realized). Equity eğrisini
    yalnız-realized kurmak işlem-İÇİ ters gidişi gizler → max_drawdown olduğundan KÜÇÜK,
    Sharpe topaklı getirilerden çarpık çıkar. MTM eğrisi metrikleri dürüstleştirir
    (risk kararları hâlâ realized portfolio.equity'i kullanır; bu yalnız raporlama)."""
    unreal = sum(p.unrealized_pnl for p in portfolio.open_positions)
    return portfolio.equity + unreal


def backtest_symbol(symbol: str, interval: str = "1h", *,
                    period: str | None = None,
                    start: str | None = None, end: str | None = None,
                    initial_equity: float = INITIAL_EQUITY,
                    warmup_bars: int = 60,
                    mc_iters: int = 0,
                    provider_name: str | None = None,
                    strategy_mode: str | None = None,
                    df: pd.DataFrame | None = None,
                    recorder=None,
                    disable_kill_switch: bool = False,
                    verbose: bool = False) -> BacktestResult:
    """
    Tek sembol üzerinde event-driven backtest.

    Madde 3 — bar i'de karar bar i-1'e kadar olan veriye göre verilir.
    Madde 4 — pozisyon açıkken next bar'ın gap'i intra-bar high/low veya açılışla yakalanır.

    strategy_mode: event_breakout | mean_reversion | rsi_divergence | ensemble (None=config)
    df: önceden yüklenmiş bar (test/offline için); None ise load_bars çağrılır.
    recorder: opsiyonel meta_features.TradeRecorder — giriş özellikleri + çıkış sonucunu
        eşleştirip meta-labeling eğitim verisi toplar. None ise hiçbir davranış değişmez.
    """
    if df is None:
        df = load_bars(symbol, interval, period=period, start=start, end=end, provider_name=provider_name)
    if df.empty:
        logger.warning(f"[bt] {symbol}: veri yok")
        return BacktestResult(symbol, interval, 0, 0, 0, 0, 0, 0, 0, 0,
                              initial_equity, initial_equity, 0, 0, 0)

    portfolio = PortfolioState(equity=initial_equity, ath_equity=initial_equity,
                               cash_balance=initial_equity)
    # Tur-101 (BULGU-101-5): günlük sayaçlar (daily_pnl, consecutive_stops) canlıda her seans sıfırlanır
    # (`main` → roll_session_if_new_day); backtest HİÇ sıfırlamıyordu ve varsayılan `last_session_date`
    # BUGÜN olduğu için çağrı eklemek de yetmezdi. Ölçüldü (.env.example: soğutma 3, günlük %10): SPY 1d
    # quant 2018-02-07'den sonra 8,5 yıl TEK işlem açmadı. Operatör .env'i iki freni kapattığı için görünmüyordu.
    portfolio.last_session_date = date.min
    risk_mgr = RiskManager(portfolio)
    pmgr = PositionManager(portfolio, _bt_slippage(), record_logs=False)  # BULGU-9: backtest üretim loglarına yazmasın

    trade_pnls: list[float] = []
    equity_curve: list[float] = [initial_equity]
    bar_returns: list[float] = []
    closes: list[CloseEvent] = []

    # Madde 20 — Backtest'te de kill-switch realism: rolling 50 işlem expectancy
    recent_pnls: deque = deque(maxlen=50)
    kill_switch_tripped = {"v": False}

    def _bt_on_close(ce: CloseEvent):
        recent_pnls.append(ce.realized_pnl)
        if len(recent_pnls) >= 30 and sum(recent_pnls) / len(recent_pnls) < 0:
            kill_switch_tripped["v"] = True
    # disable_kill_switch: DATASET üretimi için (build_meta_dataset). Kill-switch canlı
    # gerçekçiliğidir; ama çok-yıllık pencerede bir kez tetiklenince kalıcı olduğundan
    # örneklemeyi tam da kayıp rejimlerinde SANSÜRLER (gözlem: 10y 1h PIT koşusunda
    # satırların %95'i 2016'ya sıkıştı — semboller ilk ~30 işlemde kilitlenip 9 yıl ölü
    # kaldı). Meta-labeling etiketi işlem-bazlıdır, portföy-simülasyonu değil →
    # dataset kipinde bilinçli kapatılır. Varsayılan False = eski davranış birebir.
    if not disable_kill_switch:
        pmgr.add_on_close(_bt_on_close)
    if recorder is not None:
        pmgr.add_on_close(recorder.on_close)

    sector = get_sector(symbol)

    # A4 — piyasa rejimi çarpanı serisi (yalnızca açıkken). Her sembolde aynı benchmark
    # serisi önbellekten gelir; sizing'e 1-bar gecikmeyle uygulanır (look-ahead-güvenli).
    regime_series = _regime_series(period, provider_name) if CONFIG.regime.enable_regime_risk else None

    # Tur-6 perf: OHLC + zaman damgaları döngüden ÖNCE bir kez çıkarılır. Eski hali
    # her barda df.iloc[i+1] Series kurup float() çekiyordu (bar-başına sabit pandas
    # yükü); değerler birebir aynı, yalnız kurulum maliyeti elenmiştir.
    _o = df["open"].to_numpy(dtype=float)
    _h = df["high"].to_numpy(dtype=float)
    _l = df["low"].to_numpy(dtype=float)
    _c = df["close"].to_numpy(dtype=float)
    # BULGU-59-3 — nedensel ADV (hacim yoksa None → etki terimi uygulanmaz).
    _adv = (_causal_adv(df["volume"].to_numpy(dtype=float), CONFIG.backtest.impact_adv_bars)
            if "volume" in df.columns else None)
    try:
        _ts = df.index.to_pydatetime()          # DatetimeIndex → datetime dizisi
    except (AttributeError, TypeError):
        _ts = None                              # datetime-dışı indeks: eski bar-başı yol

    # Tur-7: göstergeler sembol başına BİR kez tam df üzerinde hesaplanır; her bara
    # ilk i+1 bara kırpılmış görünüm verilir. Buradaki göstergeler nedensel olduğundan
    # değerler dilim-üzerinde-hesapla ile bit-bit aynıdır (parite testleri kilitler);
    # look-ahead yok — görünüm bar i'den sonrasını GÖSTERMEZ. O(n²) → O(n).
    shared_full = _SharedIndicators(df)
    ks_i: int | None = None   # Tur-95: kill-switch'in İLK görüldüğü iterasyon

    # C3 — SONRAKİ-BAR KONVANSİYONU (kasıtlı; golden-reference parite — DEĞİŞTİRME):
    # Bar i'nin KAPANIŞINA göre karar verilir, infaz bar i+1'in AÇILIŞINDA olur. Adım 1
    # (update_all_with_bars) her iterasyonun BAŞINDA bar i+1 ile ZATEN AÇIK pozisyonları
    # kontrol eder — pozisyon ise adım 2'de (aşağıda) bar i+1 açılışında AÇILIR, yani kendi
    # GİRİŞ barı (i+1) çıkış-kontrolünden geçmez; ilk çıkış-kontrolü bir sonraki iterasyonda
    # (bar i+2) yapılır. Böylece aynı-bar giriş+çıkış (iyimser sapma / look-ahead belirsizliği)
    # önlenir; giriş barında delinen bir stop KASITLI olarak bir bar sonra tetiklenir.
    for i in range(warmup_bars, len(df) - 1):  # son barı çıkış için sakla
        now_dt = _ts[i] if _ts is not None else (
            df.index[i].to_pydatetime() if hasattr(df.index[i], "to_pydatetime") else datetime.now())
        next_dt = _ts[i + 1] if _ts is not None else (
            df.index[i + 1].to_pydatetime() if hasattr(df.index[i + 1], "to_pydatetime") else datetime.now())

        # 1) Açık pozisyonları next bar'ın intra-bar high/low/open ile güncelle (Madde 4 gap)
        bars_by_symbol = {symbol: (float(_o[i + 1]), float(_h[i + 1]),
                                    float(_l[i + 1]), float(_c[i + 1]))}
        liquidity = ({symbol: {"adv_shares": float(_adv[i]) if _adv is not None else None,
                               "bar_sigma": _bar_sigma(shared_full, i, float(_c[i]))}}
                     if pmgr.slip.impact_coef else None)
        portfolio.roll_session_if_new_day(next_dt.date())     # BULGU-101-5: canlı seans paritesi
        new_closes = pmgr.update_all_with_bars(bars_by_symbol, now=next_dt, liquidity_by_symbol=liquidity)
        for ce in new_closes:
            trade_pnls.append(ce.realized_pnl)
            closes.append(ce)
        if ks_i is None and kill_switch_tripped["v"]:
            ks_i = i   # bu iterasyonun giriş kararından itibaren giriş kapalı

        # A4 — bu bar için rejim çarpanını ayarla (1-bar gecikmeli → look-ahead-güvenli)
        if regime_series is not None:
            risk_mgr.regime_multiplier = _regime_mult_at(regime_series, df.index[i])

        # Sadece açık pozisyon yoksa VE kill-switch tripped değilse yeni giriş ara.
        # BULGU-78-1: bu iterasyonda bar i+1'de kapanan pozisyon, bar i kapanışında AÇIKTI.
        if (not portfolio.open_positions and not kill_switch_tripped["v"]
                and not (new_closes and CONFIG.backtest.reentry_policy != "same_bar")):
            # 2) Bar i'ye kadar (inclusive) dilim ile sinyal üret. Karar bar i'nin
            #    KAPANIŞINA göre, infaz bar i+1'in AÇILIŞINA.
            #
            # Katalog #34 (Tur-68) — dilim KULLANILDIĞI yerde kurulur. Eskiden döngünün
            # başındaydı ve HER barda kuruluyordu, oysa yalnız bu dalda okunuyor: 2000
            # barlık koşuda 1939 dilimin 1418'i (%73) kurulup ATILIYORDU. `df.iloc[:n]`
            # ucuz değildir — yeni BlockManager + indeks dilimi + `__finalize__` demektir
            # (ölçüldü: bu tek satır backtest süresinin ~%15'i).
            # Look-ahead açısından hiçbir şey değişmez: dilim aynı `i`, aynı sınır.
            df_slice = df.iloc[: i + 1]
            sig = generate_signal(symbol, df_slice, strategy_mode=strategy_mode,
                                  shared=shared_full.upto(i + 1))
            if sig.is_actionable():
                sig = quant_capala(sig, float(_o[i + 1]))   # Tur-97: σ-mesafesi dolum fiyatına (TA'da no-op)
                rc = risk_mgr.size_stock(entry=float(_o[i + 1]), stop=sig.stop,
                                         sector=sector, symbol=symbol, side=sig.action.value)
                if rc.approved:
                    signal_price = sig.price
                    sig.price = float(_o[i + 1])
                    pos = pmgr.open_from_signal(
                        sig, rc.sizing, now=now_dt, asset_type="stock", sector=sector,
                        signal_price=signal_price,
                        # Tur-97: pozisyonu AÇAN stratejinin etiketi (canlı main.py ile aynı ilke);
                        # quant pozisyonunun çıkış kuralları bu etikete bağlı. EventBreakout'ta aynı.
                        strategy_name=f"{sig.metadata.get('strategy', 'EventBreakout')} V1",
                        # BULGU-59-3: ikisi de bar i'ye kadarki veriden (infaz i+1) →
                        # look-ahead yok. impact_coef=0 iken hiçbir etkisi olmaz.
                        adv_shares=(float(_adv[i]) if _adv is not None else None),
                        bar_sigma=_bar_sigma(shared_full, i, float(_o[i + 1])))
                    if recorder is not None:
                        recorder.on_open(pos, df_slice, sig)

        equity_curve.append(_mtm_equity(portfolio))   # MTM: işlem-içi DD/Sharpe dürüstlüğü
        # Bar return (PnL / önceki equity)
        ret = (equity_curve[-1] - equity_curve[-2]) / max(equity_curve[-2], 1e-9)
        bar_returns.append(ret)

    # Açık pozisyon kaldıysa son bar fiyatından kapat
    if portfolio.open_positions:
        last = df.iloc[-1]
        for pos in list(portfolio.open_positions):
            ce = pmgr.force_close(pos, float(last["close"]), "BT_END",
                                   df.index[-1].to_pydatetime() if hasattr(df.index[-1], "to_pydatetime") else datetime.now())
            trade_pnls.append(ce.realized_pnl)
            closes.append(ce)
        equity_curve.append(portfolio.equity)

    # Metrikler
    wins = sum(1 for p in trade_pnls if p > 0)
    losses = sum(1 for p in trade_pnls if p <= 0)
    win_rate = wins / len(trade_pnls) if trade_pnls else 0.0
    avg_win = statistics.mean([p for p in trade_pnls if p > 0]) if wins else 0.0
    avg_loss = statistics.mean([p for p in trade_pnls if p <= 0]) if losses else 0.0
    expectancy = statistics.mean(trade_pnls) if trade_pnls else 0.0

    # Sharpe/Sortino — interval'a göre yıllık ölçek (AYNI taban: bar-başı getiri serisi)
    periods_per_year = {"1d": 252, "1h": 252 * 6.5, "15m": 252 * 26,
                         "5m": 252 * 78, "1m": 252 * 390}.get(interval, 252)
    sharpe = sharpe_annualized(bar_returns, int(periods_per_year))
    sortino = sortino_annualized(bar_returns, int(periods_per_year))
    pf = profit_factor(trade_pnls)

    mc = monte_carlo_paths(trade_pnls, initial_equity, mc_iters) if mc_iters > 0 else None

    result = BacktestResult(
        symbol=symbol, interval=interval, bars_total=len(df),
        trades_count=len(trade_pnls), wins=wins, losses=losses,
        win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
        expectancy=expectancy,
        final_equity=portfolio.equity, initial_equity=initial_equity,
        total_return_pct=(portfolio.equity - initial_equity) / initial_equity * 100,
        max_drawdown_pct=max_drawdown(equity_curve),
        sharpe_annualized=sharpe,
        sortino_annualized=sortino,
        profit_factor=pf,
        trade_pnls=trade_pnls, equity_curve=equity_curve,
        monte_carlo=mc,
        execution_costs=pmgr.execution_costs,
        **_intrabar_stats(pmgr),
        **_kill_switch_stats(df.index[ks_i] if ks_i is not None else None,
                             (len(df) - 1 - ks_i) if ks_i is not None else 0,
                             len(df) - 1 - warmup_bars),
    )
    if verbose:
        logger.info(result.short_summary())
    return result


def _periods_per_year(interval: str) -> int:
    """Bar aralığı → yıllık bar sayısı (Sharpe + realized-vol annualizasyonu paylaşır)."""
    return {"1d": 252, "1h": int(252 * 6.5), "15m": 252 * 26,
            "5m": 252 * 78, "1m": 252 * 390}.get(interval, 252)


def _realized_iv(df: pd.DataFrame, idx: int, bc, ppy: int) -> float:
    """Bar idx'e kadarki (dahil) gerçekleşen volatiliteden IV tahmini (look-ahead-safe).

    Log-getiri std × sqrt(ppy) × risk primi, [floor, ceiling]'e kırpılır. Dinamik IV
    kapalıysa sabit option_bt_iv döner. Sentetik opsiyon backtest'inde IV'nin volatilite
    rejimiyle değişmesini (ve dolayısıyla vega/IV-crush P&L'ini) sağlar."""
    if not bc.option_bt_dynamic_iv:
        return bc.option_bt_iv
    lb = bc.option_bt_iv_lookback
    start = max(0, idx - lb + 1)
    closes = df["close"].iloc[start: idx + 1].to_numpy(dtype=float)
    if len(closes) < 3:
        return bc.option_bt_iv
    rets = np.diff(np.log(np.clip(closes, 1e-9, None)))
    rv = float(np.std(rets, ddof=1)) * (ppy ** 0.5)
    iv = rv * bc.option_bt_iv_risk_premium
    return float(min(max(iv, bc.option_bt_iv_floor), bc.option_bt_iv_ceiling))


class _IntrabarStats(TypedDict):
    intrabar_ambiguous: int
    intrabar_resolved: int
    intrabar_flipped: int


def _intrabar_stats(pmgr) -> _IntrabarStats:
    """PositionManager'ın intra-bar belirsizlik sayaçlarını sonuç alanlarına çevir
    (BULGU-59-2). pmgr None/eski ise sıfırlar → geriye uyumlu."""
    return {
        "intrabar_ambiguous": int(getattr(pmgr, "intrabar_ambiguous", 0) or 0),
        "intrabar_resolved": int(getattr(pmgr, "intrabar_resolved", 0) or 0),
        "intrabar_flipped": int(getattr(pmgr, "intrabar_flipped", 0) or 0),
    }


class _KillSwitchStats(TypedDict):
    kill_switch_at: str | None
    kill_switch_idle_pct: float | None


def _kill_switch_stats(at, idle_n: int, total_n: int) -> _KillSwitchStats:
    """Tur-95 (BULGU-95-3): kill-switch tetiklendiyse girişin ilk engellendiği an + girişsiz
    kalan karar barlarının yüzdesi. Tetiklenmediyse iki alan da None (0 "hiç boşta değil" olurdu)."""
    if at is None:
        return {"kill_switch_at": None, "kill_switch_idle_pct": None}
    return {"kill_switch_at": str(at), "kill_switch_idle_pct": round(100.0 * idle_n / max(total_n, 1), 2)}


def _metrics_from(symbol, interval, df_len, trade_pnls, equity_curve, bar_returns,
                  initial_equity, final_equity, mc_iters, pmgr=None,
                  kill_switch: _KillSwitchStats | None = None) -> BacktestResult:
    wins = sum(1 for p in trade_pnls if p > 0)
    losses = sum(1 for p in trade_pnls if p <= 0)
    win_rate = wins / len(trade_pnls) if trade_pnls else 0.0
    avg_win = statistics.mean([p for p in trade_pnls if p > 0]) if wins else 0.0
    avg_loss = statistics.mean([p for p in trade_pnls if p <= 0]) if losses else 0.0
    expectancy = statistics.mean(trade_pnls) if trade_pnls else 0.0
    ppy = _periods_per_year(interval)
    mc = monte_carlo_paths(trade_pnls, initial_equity, mc_iters) if mc_iters > 0 else None
    return BacktestResult(
        symbol=symbol, interval=interval, bars_total=df_len,
        trades_count=len(trade_pnls), wins=wins, losses=losses,
        win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss, expectancy=expectancy,
        final_equity=final_equity, initial_equity=initial_equity,
        total_return_pct=(final_equity - initial_equity) / initial_equity * 100,
        max_drawdown_pct=max_drawdown(equity_curve),
        sharpe_annualized=sharpe_annualized(bar_returns, ppy),
        sortino_annualized=sortino_annualized(bar_returns, ppy),
        profit_factor=profit_factor(trade_pnls),
        trade_pnls=trade_pnls, equity_curve=equity_curve, monte_carlo=mc,
        execution_costs=pmgr.execution_costs if pmgr is not None else None,
        **_intrabar_stats(pmgr),
        **(kill_switch or _kill_switch_stats(None, 0, 0)),
    )


# ---------------------------------------------------------------------------
# Portföy backtest — paylaşılan equity + Madde 12/14 tavanları (çok sembol)
# ---------------------------------------------------------------------------
def backtest_portfolio(symbols: list[str], interval: str = "1d", *,
                       period: str | None = None, start: str | None = None, end: str | None = None,
                       initial_equity: float = INITIAL_EQUITY, warmup_bars: int = 60,
                       mc_iters: int = 0, provider_name: str | None = None,
                       strategy_mode: str | None = None, disable_kill_switch: bool = False,
                       data: dict | None = None, verbose: bool = False) -> BacktestResult:
    """Çok sembol, PAYLAŞILAN portföy. Zaman damgasıyla hizalanır; her sembolün
    kararı kendi bir sonraki barında infaz edilir. Portföy tavanları (Madde 14)
    ve sektör/sembol kümelenmesi (Madde 12) sembol-arası uygulanır.

    data: {sym: df} önceden yüklenmiş (offline test); None ise load_bars."""
    if data is None:
        data = {}
        for s in symbols:
            d = load_bars(s, interval, period=period, start=start, end=end, provider_name=provider_name)
            if not d.empty:
                data[s] = d
    data = {s: d for s, d in data.items() if d is not None and not d.empty}
    if not data:
        return BacktestResult("PORTFOLIO", interval, 0, 0, 0, 0, 0, 0, 0, 0,
                              initial_equity, initial_equity, 0, 0, 0)

    portfolio = PortfolioState(equity=initial_equity, ath_equity=initial_equity,
                               cash_balance=initial_equity)
    portfolio.last_session_date = date.min     # BULGU-101-5 (bkz. backtest_symbol)
    risk_mgr = RiskManager(portfolio)
    pmgr = PositionManager(portfolio, _bt_slippage(), record_logs=False)  # BULGU-9: backtest üretim loglarına yazmasın

    trade_pnls: list[float] = []
    equity_curve: list[float] = [initial_equity]
    bar_returns: list[float] = []
    recent_pnls: deque = deque(maxlen=50)
    kill = {"v": False}

    def _on_close(ce: CloseEvent):
        recent_pnls.append(ce.realized_pnl)
        if len(recent_pnls) >= 30 and sum(recent_pnls) / len(recent_pnls) < 0:
            kill["v"] = True
    # disable_kill_switch: backtest_symbol ile aynı gerekçe. Kill-switch bir kez
    # tetiklenince KALICIDIR (yeniden kurulmaz) ve portföy kipinde TÜM sembolleri
    # birden susturur → çok-yıllık pencerede koşu ~30. işlemde ölür. Uzun-pencere
    # strateji karşılaştırmasında kapatılmalı. Varsayılan False = eski davranış.
    if not disable_kill_switch:
        pmgr.add_on_close(_on_close)

    # Tur-7 perf: sembol başına (a) ts→konum haritası (yinelenen ts'ler eski get_loc
    # davranışıyla uyumlu şekilde ATLANIR), (b) OHLC numpy dizileri, (c) datetime listesi,
    # (d) tam-df paylaşımlı gösterge önbelleği (_SharedIndicators — nedensel göstergelerde
    # kırpılmış görünüm bit-bit aynı; O(n²) gösterge maliyeti O(n)'e iner).
    prep: dict = {}
    for s, d in data.items():
        dup = d.index.duplicated(keep=False)
        pos_map = {ts_: j for j, (ts_, du) in enumerate(zip(d.index, dup, strict=True)) if not du}
        try:
            ts_list = d.index.to_pydatetime()
        except (AttributeError, TypeError):
            ts_list = [ix.to_pydatetime() if hasattr(ix, "to_pydatetime") else datetime.now()
                       for ix in d.index]
        adv_a = (_causal_adv(d["volume"].to_numpy(dtype=float), CONFIG.backtest.impact_adv_bars)
                 if "volume" in d.columns else None)
        prep[s] = (pos_map, d["open"].to_numpy(dtype=float), d["high"].to_numpy(dtype=float),
                   d["low"].to_numpy(dtype=float), d["close"].to_numpy(dtype=float),
                   ts_list, _SharedIndicators(d), adv_a)

    all_ts = sorted(set().union(*[set(d.index) for d in data.values()]))
    ks_k: int | None = None   # Tur-95: kill-switch'in tetiklendiği iterasyon (tek sembolle aynı tanım)
    for k, ts in enumerate(all_ts):
        for s, d in data.items():
            pos_map, o_a, h_a, l_a, c_a, ts_list, shared_full, adv_a = prep[s]
            i = pos_map.get(ts)
            if i is None or i < warmup_bars or i >= len(d) - 1:
                continue
            bars = {s: (float(o_a[i + 1]), float(h_a[i + 1]),
                        float(l_a[i + 1]), float(c_a[i + 1]))}
            liquidity = ({s: {"adv_shares": float(adv_a[i]) if adv_a is not None else None,
                              "bar_sigma": _bar_sigma(shared_full, i, float(c_a[i]))}}
                         if pmgr.slip.impact_coef else None)
            portfolio.roll_session_if_new_day(ts_list[i + 1].date())     # BULGU-101-5
            closed_now = pmgr.update_all_with_bars(bars, now=ts_list[i + 1], liquidity_by_symbol=liquidity)
            trade_pnls.extend(ce.realized_pnl for ce in closed_now)

            if kill["v"] or any(p.symbol == s for p in portfolio.open_positions):
                continue
            # BULGU-78-1 (bkz. backtest_symbol). ponytail: aynı zaman damgasında BAŞKA sembolün
            # bar-içi çıkışının serbest bıraktığı sermaye hâlâ bu girişin boyutuna girer.
            if closed_now and CONFIG.backtest.reentry_policy != "same_bar":
                continue
            sig = generate_signal(s, d.iloc[: i + 1], strategy_mode=strategy_mode,
                                  shared=shared_full.upto(i + 1))
            if sig.is_actionable():
                sig = quant_capala(sig, float(o_a[i + 1]))
                rc = risk_mgr.size_stock(entry=float(o_a[i + 1]), stop=sig.stop,
                                         sector=get_sector(s), symbol=s, side=sig.action.value)
                if rc.approved:
                    signal_price = sig.price
                    sig.price = float(o_a[i + 1])
                    pmgr.open_from_signal(
                        sig, rc.sizing, now=ts_list[i], asset_type="stock", sector=get_sector(s),
                        signal_price=signal_price,
                        strategy_name=f"{sig.metadata.get('strategy', 'EventBreakout')} V1",
                        adv_shares=(float(adv_a[i]) if adv_a is not None else None),
                        bar_sigma=_bar_sigma(shared_full, i, float(o_a[i + 1])))
        if ks_k is None and kill["v"]:
            ks_k = k   # bu zaman damgasında sıradaki sembollerin girişi zaten kesildi
        equity_curve.append(_mtm_equity(portfolio))   # MTM: işlem-içi DD/Sharpe dürüstlüğü
        bar_returns.append((equity_curve[-1] - equity_curve[-2]) / max(equity_curve[-2], 1e-9))

    # Kalan açık pozisyonları kapat
    for s, d in data.items():
        last = d.iloc[-1]
        last_dt = d.index[-1].to_pydatetime() if hasattr(d.index[-1], "to_pydatetime") else datetime.now()
        for pos in [p for p in list(portfolio.open_positions) if p.symbol == s]:
            ce = pmgr.force_close(pos, float(last["close"]), "BT_END", last_dt)
            trade_pnls.append(ce.realized_pnl)
    equity_curve.append(portfolio.equity)

    label = "PORTFOLIO(" + ",".join(data.keys()) + ")"
    r = _metrics_from(label, interval, sum(len(d) for d in data.values()), trade_pnls,
                      equity_curve, bar_returns, initial_equity, portfolio.equity, mc_iters,
                      pmgr=pmgr,
                      kill_switch=_kill_switch_stats(all_ts[ks_k] if ks_k is not None else None,
                                                     len(all_ts) - (ks_k or 0), len(all_ts)))
    if verbose:
        logger.info(r.short_summary())
    return r


# ---------------------------------------------------------------------------
# Sentetik opsiyon backtest — Madde 28/2 (tarihsel zincir yok → Black-Scholes)
# ---------------------------------------------------------------------------
def _measured_greeks_mask(candidates: pd.DataFrame, measured_greeks: Callable) -> pd.Series:
    """Sayısal adayları topluca süz; belirsiz dtype'ta eski satır/hata davranışını koru."""
    values = candidates[["delta", "gamma", "theta", "vega", "vol"]].to_numpy()
    if candidates.empty or values.dtype.kind not in "fiu":
        # Sayısal metin de np.isfinite(list) için TypeError'dır; zorla dönüştürmeyiz.
        return candidates.apply(lambda row: measured_greeks(row) is not None, axis=1)
    valid = (np.isfinite(values).all(axis=1)
             & (values[:, 0] >= -1) & (values[:, 0] <= 1)
             & (values[:, 1] >= 0) & (values[:, 3] >= 0) & (values[:, 4] > 0))
    return pd.Series(valid, index=candidates.index)


def backtest_option_chain(symbol: str, interval: str = "1d", *,
                          period: str | None = None, start: str | None = None, end: str | None = None,
                          initial_equity: float = INITIAL_EQUITY, warmup_bars: int = 60,
                          mc_iters: int = 0, provider_name: str | None = None,
                          strategy_mode: str | None = None, options_file: str | Path | None = None,
                          df: pd.DataFrame | None = None, chain_df: pd.DataFrame | None = None,
                          verbose: bool = False) -> BacktestResult:
    """Gerçek günlük Dolt kotasyonları: t kapanışında seçim, t+1 kapanışında icra.

    Kaynakta saat/işlem hacmi yok: günlük kapanış kotasyonu ve standart 100 pay
    çarpanı varsayılır, emir kuyruğu/dolum garantisi modellenmez. Eksik gün veya
    sıfır bid ile satış üretilmez. Prim stop/hedef kararı da sonraki bara gecikir.
    Veri boşluğu boyunca MTM, DD ve risk oranları None; sentetik fiyat yedeği yok.
    """
    from nihai.data.dolt_options import local_option_history, option_snapshot, prepare_option_history
    from nihai.data.market_calendar import NYSE
    from nihai.risk.options_analyzer import GreekSet
    from nihai.risk.risk_manager import OPTION_FEES

    if interval != "1d":
        raise ValueError("Dolt zinciri yalnız günlük tarihlidir; gerçek opsiyon testi için interval=1d gerekli.")
    if not np.isfinite(initial_equity) or initial_equity <= 0 or warmup_bars < 0:
        raise ValueError("Pozitif sonlu sermaye ve negatif olmayan ısınma süresi gerekli.")
    if df is None:
        df = load_bars(symbol, interval, period=period, start=start, end=end, provider_name=provider_name)
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Gerçek opsiyon testi için tarihli dayanak barları bulunamadı.")
    df = df.copy()
    df.index = df.index.tz_localize(None).normalize()
    if df.index.has_duplicates or not df.index.is_monotonic_increasing or df.index.hasnans:
        raise ValueError("Dayanak günleri artan ve tekil olmalı.")
    sessions = NYSE.sessions.tz_localize(None)
    df = df.loc[df.index.isin(sessions)]
    if len(df) < warmup_bars + 2:
        raise ValueError("Isınma ve bir bar gecikmeli opsiyon icrası için yeterli seans yok.")
    history = (local_option_history(symbol, path=options_file,
                                   start=df.index[0].date().isoformat(), end=df.index[-1].date().isoformat())
               if chain_df is None else prepare_option_history(chain_df, symbol))
    if history.empty:
        raise ValueError("Yerel Dolt zincirinde geçerli veri yok; sentetik fiyat kullanılmadı.")
    diagnostics = dict(source=history.attrs.get("source", "provided_frame"),
                       model="daily_close_bid_ask", exec_lag_bars=1,
                       pit_scope="quote_date_only_not_revision_vintage", multiplier=100,
                       rejected_rows=history.attrs.get("rejected_rows", 0),
                       no_chain_bars=0, unfilled_entries=0, rejected_entries=0,
                       missing_mark_bars=0, unfilled_exits=0, fills=[])
    by_day = {day: rows for day, rows in history.groupby("date", sort=True)}
    test_days = df.index[warmup_bars:]
    diagnostics["quoted_bars"] = sum(day in by_day for day in test_days)
    diagnostics["evaluated_bars"] = len(test_days)
    if not diagnostics["quoted_bars"]:
        raise ValueError("Seçilen test aralığında işlem günü kotasyonu yok.")
    keys = ["expiration", "strike", "call_put"]
    indexed = {day: rows.set_index(keys) for day, rows in by_day.items()}
    portfolio = PortfolioState(equity=initial_equity, ath_equity=initial_equity, cash_balance=initial_equity)
    risk_mgr = RiskManager(portfolio)
    bc = CONFIG.backtest
    fee = sum(OPTION_FEES.values())
    cash = initial_equity
    curve: list[float | None] = [initial_equity]
    pnls: list[float] = []
    recent: deque = deque(maxlen=50)
    pos: dict | None = None
    pending: dict | None = None

    def quote_at(day, contract):
        quotes = indexed.get(day)
        key = tuple(contract[k] for k in keys)
        return quotes.loc[key] if quotes is not None and key in quotes.index else None

    def measured_greeks(row):
        values = [row[k] for k in ("delta", "gamma", "theta", "vega", "vol")]
        if not np.isfinite(values).all() or not -1 <= values[0] <= 1 or min(values[1], values[3]) < 0 or values[4] <= 0:
            return None
        return GreekSet(*map(float, values))

    # Tur-80: normalize edilmiş ve seansları süzülmüş döngü verisini bir kez paylaş.
    shared_full = _SharedIndicators(df)
    for i in range(warmup_bars, len(df)):
        day = df.index[i]
        last = i == len(df) - 1
        if day not in by_day:
            diagnostics["no_chain_bars"] += 1
        # Önceden seçilen kontrat: t+1 zinciri kontrat SEÇİMİNİ değiştiremez.
        if pending is not None:
            fill = quote_at(day, pending)
            # Tur-78 (R1-B2): dolum YALNIZ karar gününden sonraki NYSE seansında. Dayanak barı
            # eksikse döngünün sonraki günü iki seans sonrasıdır; eskiden orada alınıyordu.
            j = sessions.searchsorted(pd.Timestamp(pending["decision_date"]), side="right")
            next_session = sessions[j] if j < len(sessions) else None
            if (last or fill is None or day != next_session
                    or (pending["expiration"] - day).days <= bc.option_bt_close_dte):
                diagnostics["unfilled_entries"] += 1
            else:
                greeks = measured_greeks(fill)
                rc = (risk_mgr.size_long_option(float(fill["ask"]), greeks, sector=get_sector(symbol),
                                               symbol=symbol, option_type=pending["call_put"])
                      if greeks is not None else None)
                n = int(rc.sizing["contracts"]) if rc is not None and rc.approved else 0
                if n > 0 and n * (float(fill["ask"]) * 100 + fee) <= cash:
                    pos = {**pending, "entry_date": day, "entry_fill": float(fill["ask"]),
                           "contracts": n, "exit_reason": None}
                    cash -= n * (pos["entry_fill"] * 100 + fee)
                    diagnostics["fills"].append(dict(side="buy", date=day.date().isoformat(),
                        decision_date=pending["decision_date"], expiration=pos["expiration"].date().isoformat(),
                        strike=pos["strike"], call_put=pos["call_put"], contracts=n,
                        price=pos["entry_fill"], bid=float(fill["bid"]), ask=float(fill["ask"]),
                        spread_cost=n * 100 * (float(fill["ask"]) - float(fill["bid"])) / 2,
                        fees=n * fee))
                else:
                    diagnostics["rejected_entries"] += 1
            pending = None  # tek sonraki seans; bayat sinyalle günler sonra giriş yok

        mark: float | None = cash
        if pos is not None:
            quote = quote_at(day, pos)
            dte = (pos["expiration"] - day).days
            reason = pos["exit_reason"] or ("BT_END" if last else "DTE_GUARD" if dte <= bc.option_bt_close_dte else None)
            if reason and day > pos["entry_date"] and quote is not None and quote["bid"] > 0:
                fill_price = float(quote["bid"])
                n = pos["contracts"]
                cash += n * (fill_price * 100 - fee)
                pnl = n * ((fill_price - pos["entry_fill"]) * 100 - 2 * fee)
                pnls.append(pnl)
                recent.append(pnl)
                diagnostics["fills"].append(dict(side="sell", date=day.date().isoformat(),
                    expiration=pos["expiration"].date().isoformat(), strike=pos["strike"], call_put=pos["call_put"],
                    contracts=n, price=fill_price, reason=reason, pnl=pnl,
                    bid=float(quote["bid"]), ask=float(quote["ask"]),
                    spread_cost=n * 100 * (float(quote["ask"]) - float(quote["bid"])) / 2, fees=n * fee))
                pos = None
                mark = cash
            else:
                if reason:
                    diagnostics["unfilled_exits"] += 1
                    pos["exit_reason"] = reason
                if quote is None:
                    mark = None
                    diagnostics["missing_mark_bars"] += 1
                else:
                    bid = float(quote["bid"])
                    mark = cash + pos["contracts"] * bid * 100
                    if bid <= pos["entry_fill"] * (1 - CONFIG.option_strategy.premium_stop_loss_pct):
                        pos["exit_reason"] = pos["exit_reason"] or "PREMIUM_STOP"
                    elif bid >= pos["entry_fill"] * (1 + CONFIG.option_strategy.premium_target_gain_pct):
                        pos["exit_reason"] = pos["exit_reason"] or "PREMIUM_TARGET"
        curve.append(mark)
        if mark is not None:
            portfolio.equity = mark
            portfolio.ath_equity = max(portfolio.ath_equity, mark)
        portfolio.cash_balance = cash

        if pos is not None or last or (len(recent) >= 30 and sum(recent) < 0):
            continue
        sig = generate_signal(symbol, df.iloc[:i + 1], strategy_mode=strategy_mode,
                              shared=shared_full.upto(i + 1))
        if not sig.is_actionable():
            continue
        # Sadece bugüne kadarki zincir. Geçmiş dilim tek güne indirgenmiştir;
        # 10 GB dosya ne bar başına tekrar okunur ne de geleceğe doğru aranır.
        snapshot = option_snapshot(by_day.get(day, pd.DataFrame()), as_of=day.date().isoformat())
        if snapshot.empty:
            continue
        opt_type = "c" if sig.action == SignalAction.LONG else "p"
        candidates = snapshot.loc[snapshot["call_put"] == opt_type].copy()
        candidates["dte"] = (candidates["expiration"] - day).dt.days
        candidates = candidates.loc[(candidates["dte"] > bc.option_bt_close_dte + 1)
                                    & _measured_greeks_mask(candidates, measured_greeks)]
        if candidates.empty:
            continue
        candidates["dte_distance"] = (candidates["dte"] - bc.option_bt_dte_entry).abs()
        candidates["delta_distance"] = (candidates["delta"].abs() - bc.option_bt_entry_delta).abs()
        selected = candidates.sort_values(["dte_distance", "delta_distance", "expiration", "strike"]).iloc[0]
        pending = {k: selected[k] for k in keys}
        pending["decision_date"] = day.date().isoformat()

    # Eksik MTM barlarını enterpole etmeyiz. İşlem PnL'si ölçülebilir kalırken
    # yol bağımlı metrikler ve ölçülemeyen son değer ayrı ayrı None olur.
    observed = [v for v in curve if v is not None]
    returns = [(b - a) / a for a, b in zip(curve, curve[1:], strict=False) if a is not None and b is not None and a > 0]
    result = _metrics_from(f"OPT:{symbol}", interval, len(df), pnls, observed, returns,
                           initial_equity, observed[-1], mc_iters if pos is None else 0)
    result.equity_curve = curve
    result.final_equity = curve[-1]
    result.total_return_pct = (curve[-1] / initial_equity - 1) * 100 if curve[-1] is not None else None
    if diagnostics["missing_mark_bars"]:
        result.max_drawdown_pct = result.sharpe_annualized = result.sortino_annualized = None
    diagnostics["complete"] = pos is None and not diagnostics["missing_mark_bars"]
    diagnostics["open_position"] = ({k: str(v) if isinstance(v, pd.Timestamp) else v for k, v in pos.items()}
                                    if pos is not None else None)
    diagnostics["spread_cost"] = sum(f["spread_cost"] for f in diagnostics["fills"])
    diagnostics["fees"] = sum(f["fees"] for f in diagnostics["fills"])
    result.option_data = diagnostics
    if verbose:
        logger.info(result.short_summary() + f" [Dolt günlük bid/ask; kapsam {diagnostics['quoted_bars']}/{len(test_days)}]")
    return result


def _strike_for_delta(S: float, target_delta: float, sigma: float, T: float,
                      r: float, opt_type: str) -> float:
    """Hedef delta'ya karşılık gelen strike (BS, kuru). target_delta pozitif büyüklük."""
    from scipy.stats import norm
    d1 = norm.ppf(target_delta if opt_type == "c" else 1 - target_delta)
    K = S * np.exp((r + 0.5 * sigma ** 2) * T - d1 * sigma * (T ** 0.5))
    return round(K, 2)


def backtest_option_synth(symbol: str, interval: str = "1d", *,
                          period: str | None = None, start: str | None = None, end: str | None = None,
                          initial_equity: float = INITIAL_EQUITY, warmup_bars: int = 60,
                          mc_iters: int = 0, provider_name: str | None = None,
                          strategy_mode: str | None = None,
                          df: pd.DataFrame | None = None, verbose: bool = False) -> BacktestResult:
    """Dayanak sinyalini SENTETİK uzun opsiyona çevirip Black-Scholes ile yeniden
    fiyatlar. YAKLAŞIKLAMADIR: tarihsel opsiyon zinciri kullanmaz. IV varsayılan olarak
    dayanağın gerçekleşen volatilitesinden türetilir ve her bar güncellenir (dinamik IV;
    CONFIG.backtest.option_bt_dynamic_iv=False ile sabit option_bt_iv'ye döner). Smile/skew
    modellenmez (tek IV, tüm strike'lar). Çıkışlar: prim stop/hedef + DTE guard (Madde 28)."""
    from vollib.black_scholes import black_scholes as bs_price

    from nihai.risk.options_analyzer import GreekSet

    bc = CONFIG.backtest
    if df is None:
        df = load_bars(symbol, interval, period=period, start=start, end=end, provider_name=provider_name)
    if df.empty:
        return BacktestResult(f"OPT:{symbol}", interval, 0, 0, 0, 0, 0, 0, 0, 0,
                              initial_equity, initial_equity, 0, 0, 0)

    portfolio = PortfolioState(equity=initial_equity, ath_equity=initial_equity, cash_balance=initial_equity)
    risk_mgr = RiskManager(portfolio)
    ppy = _periods_per_year(interval)
    r = bc.risk_free_rate

    trade_pnls: list[float] = []
    equity_curve: list[float] = [initial_equity]
    bar_returns: list[float] = []
    recent_pnls: deque = deque(maxlen=50)
    kill = False
    pos: dict | None = None

    def _bar_date(idx):
        v = df.index[idx]
        return v.date() if hasattr(v, "date") else date.today()

    # Tur-80: göstergeler bir kez kurulur; her karar yalnız kendi geçmişini görür.
    shared_full = _SharedIndicators(df)
    for i in range(warmup_bars, len(df) - 1):
        bar_next = df.iloc[i + 1]
        S_open = float(bar_next["open"])
        S_close = float(bar_next["close"])
        next_date = _bar_date(i + 1)

        # --- Açık opsiyonu yönet (yeniden fiyatla + çıkış) ---
        exited_now = False
        if pos is not None:
            dte_now = max(pos["dte_entry"] - (next_date - pos["entry_date"]).days, 0.5)
            T = dte_now / 365.0
            # IV'yi o ana kadarki gerçekleşen volatiliteyle güncelle → vega/IV-crush P&L
            iv_now = _realized_iv(df, i + 1, bc, ppy)
            prem_now = max(bs_price(pos["type"], S_close, pos["strike"], T, r, iv_now), 0.0)
            entry = pos["entry_prem"]
            trigger = None
            if prem_now <= entry * (1 - CONFIG.option_strategy.premium_stop_loss_pct):
                trigger = "PREMIUM_STOP"
            elif prem_now >= entry * (1 + CONFIG.option_strategy.premium_target_gain_pct):
                trigger = "PREMIUM_TARGET"
            elif dte_now <= bc.option_bt_close_dte:
                trigger = "DTE_GUARD"
            if trigger:
                exit_fill = prem_now * (1 - bc.option_slippage_pct)  # long çıkışta kötü tarafa
                from nihai.risk.risk_manager import OPTION_FEES
                pnl = (exit_fill - pos["entry_fill"]) * pos["contracts"] * 100
                pnl -= sum(OPTION_FEES.values()) * pos["contracts"] * 2
                portfolio.equity += pnl
                portfolio.cash_balance += pnl
                if portfolio.equity > portfolio.ath_equity:
                    portfolio.ath_equity = portfolio.equity
                trade_pnls.append(pnl)
                recent_pnls.append(pnl)
                if len(recent_pnls) >= 30 and sum(recent_pnls) / len(recent_pnls) < 0:
                    kill = True
                pos = None
                exited_now = True

        # --- Yeni giriş --- (BULGU-78-1: çıkış bar i+1 KAPANIŞINDA, giriş bar i+1 AÇILIŞINDA olurdu)
        if pos is None and not kill and not (exited_now and CONFIG.backtest.reentry_policy != "same_bar"):
            sig = generate_signal(symbol, df.iloc[: i + 1], strategy_mode=strategy_mode,
                                  shared=shared_full.upto(i + 1))
            if sig.is_actionable():
                opt_type = "c" if sig.action == SignalAction.LONG else "p"
                dte = bc.option_bt_dte_entry
                T = dte / 365.0
                # Giriş IV'si KARAR barına (i) kadarki veriden → look-ahead-safe
                iv_entry = _realized_iv(df, i, bc, ppy)
                strike = _strike_for_delta(S_open, bc.option_bt_entry_delta, iv_entry, T, r, opt_type)
                entry_prem = max(bs_price(opt_type, S_open, strike, T, r, iv_entry), 0.0)
                if entry_prem > 0.01:
                    dsign = bc.option_bt_entry_delta * (1 if opt_type == "c" else -1)
                    greeks = GreekSet(delta=dsign, gamma=0.02, theta=-entry_prem / dte,
                                      vega=entry_prem * 0.1, iv=iv_entry)
                    rc = risk_mgr.size_long_option(entry_prem, greeks, sector=get_sector(symbol),
                                                   symbol=symbol, option_type=opt_type)
                    if rc.approved:
                        pos = {
                            "type": opt_type, "strike": strike, "entry_prem": entry_prem,
                            "entry_fill": entry_prem * (1 + bc.option_slippage_pct),
                            "contracts": rc.sizing["contracts"], "dte_entry": dte,
                            "entry_date": _bar_date(i),
                        }

        equity_curve.append(portfolio.equity)
        bar_returns.append((equity_curve[-1] - equity_curve[-2]) / max(equity_curve[-2], 1e-9))

    label = f"OPT:{symbol}"
    r_out = _metrics_from(label, interval, len(df), trade_pnls, equity_curve, bar_returns,
                          initial_equity, portfolio.equity, mc_iters)   # sentetik opsiyon
                                                                          # yolu PositionManager
                                                                          # KULLANMAZ (intra-bar
                                                                          # sayacı yok)
    if verbose:
        iv_mode = "dinamik IV (gerçekleşen vol)" if bc.option_bt_dynamic_iv else "sabit IV"
        logger.info(r_out.short_summary() + f"  [SENTETİK opsiyon — {iv_mode}]")
    return r_out


# ---------------------------------------------------------------------------
# Walk-forward optimizasyon — parametre robustluğu / overfitting tespiti (Madde 51/52)
# ---------------------------------------------------------------------------
_WF_METRICS = {
    "expectancy": lambda r: r.expectancy,
    "sharpe": lambda r: r.sharpe_annualized,
    "return": lambda r: r.total_return_pct,
}


@dataclass
class WalkForwardFold:
    fold: int
    is_start: int
    oos_start: int
    oos_end: int
    chosen: str               # IS'te kazanan strateji
    is_metric: float          # seçimi yapan IS metrik değeri
    oos_metric: float         # aynı stratejinin OOS'taki gerçekleşen metriği
    oos_trades: int
    oos_return_pct: float


@dataclass
class WalkForwardResult:
    symbol: str
    interval: str
    metric: str
    candidates: list[str]
    folds: list[WalkForwardFold]
    oos_trades_count: int
    oos_win_rate: float
    oos_expectancy: float
    oos_total_return_pct: float
    oos_max_drawdown_pct: float
    oos_sharpe_annualized: float
    wf_efficiency: float      # mean(OOS metrik) / mean(IS metrik); <1 → overfitting
    strategy_picks: dict      # {strateji: kaç fold seçildi}
    initial_equity: float
    final_equity: float


def walk_forward(symbol: str, interval: str = "1d", *,
                 period: str | None = None, start: str | None = None, end: str | None = None,
                 initial_equity: float = INITIAL_EQUITY, is_bars: int = 180, oos_bars: int = 60,
                 warmup_bars: int = 60, candidates: list[str] | None = None,
                 metric: str = "expectancy", provider_name: str | None = None,
                 df: pd.DataFrame | None = None, verbose: bool = False) -> WalkForwardResult:
    """Yuvarlanan IS/OOS pencerelerinde strateji seçimini optimize edip OOS'ta test eder.

    Her fold: önceki `is_bars` barda aday stratejileri backtest'le, `metric`'e göre en
    iyiyi seç; sonra takip eden `oos_bars` barda (warmup lead-in'li) yalnızca o stratejiyi
    çalıştır. OOS sonuçları compound equity ile birleştirilir. IS-seçim metriği OOS'ta
    tutmuyorsa (wf_efficiency < 1) parametre/strateji seçimi overfit'tir (Madde 51/52).

    Look-ahead yok: OOS her zaman IS'ten SONRAKİ bar penceresidir; OOS warmup'ı IS kuyruğudur."""
    # Strateji-SEÇİMİ yalnız fiyat-only, backtest-güvenli baz modlar arasından
    # (pead/insider/sentiment harici veri çeker → look-ahead/gürültü; ensemble/confluence
    # birleştirici meta-modlar → baz aday olamaz). STRATEGY_MODES tam liste config doğrulama için.
    cands = list(candidates) if candidates else list(BASE_STRATEGY_MODES)
    score = _WF_METRICS.get(metric, _WF_METRICS["expectancy"])
    if is_bars <= warmup_bars:
        raise ValueError(f"is_bars ({is_bars}) > warmup_bars ({warmup_bars}) olmalı")
    if df is None:
        df = load_bars(symbol, interval, period=period, start=start, end=end, provider_name=provider_name)

    n = len(df)
    folds: list[WalkForwardFold] = []
    stitched_pnls: list[float] = []
    stitched_curve: list[float] = [initial_equity]
    stitched_returns: list[float] = []
    picks: dict[str, int] = {}
    is_sum = oos_sum = 0.0
    equity = initial_equity

    is_start = 0
    while is_start + is_bars + oos_bars <= n:
        oos_start = is_start + is_bars
        is_df = df.iloc[is_start:oos_start]

        best: str | None = None
        best_score = float("-inf")
        for c in cands:
            r = backtest_symbol(symbol, interval, df=is_df, warmup_bars=warmup_bars,
                                initial_equity=initial_equity, strategy_mode=c)
            s = score(r)
            if r.trades_count > 0 and s > best_score:
                best_score, best = s, c
        if best is None:                       # IS'te hiçbir aday işlem açmadı → fold'u atla
            is_start += oos_bars
            continue

        lead = max(oos_start - warmup_bars, 0)
        oos_df = df.iloc[lead:oos_start + oos_bars]
        oos_r = backtest_symbol(symbol, interval, df=oos_df, warmup_bars=warmup_bars,
                                initial_equity=equity, strategy_mode=best)
        for p in oos_r.trade_pnls:
            equity += p
            stitched_pnls.append(p)
            stitched_curve.append(equity)
            stitched_returns.append(p / max(stitched_curve[-2], 1e-9))

        oos_val = score(oos_r)
        picks[best] = picks.get(best, 0) + 1
        is_sum += best_score
        oos_sum += oos_val
        assert oos_r.total_return_pct is not None  # hisse WF yolu daima ölçülmüş değer döndürür
        folds.append(WalkForwardFold(
            fold=len(folds) + 1, is_start=is_start, oos_start=oos_start,
            oos_end=oos_start + oos_bars, chosen=best, is_metric=best_score,
            oos_metric=oos_val, oos_trades=oos_r.trades_count,
            oos_return_pct=oos_r.total_return_pct))
        is_start += oos_bars

    agg = _metrics_from(f"WF:{symbol}", interval, n, stitched_pnls, stitched_curve,
                        stitched_returns, initial_equity, equity, 0)
    nf = len(folds)
    wf_eff = (oos_sum / is_sum) if nf and abs(is_sum) > 1e-12 else 0.0
    assert agg.total_return_pct is not None and agg.max_drawdown_pct is not None and agg.sharpe_annualized is not None
    res = WalkForwardResult(
        symbol=symbol, interval=interval, metric=metric, candidates=cands, folds=folds,
        oos_trades_count=agg.trades_count, oos_win_rate=agg.win_rate,
        oos_expectancy=agg.expectancy, oos_total_return_pct=agg.total_return_pct,
        oos_max_drawdown_pct=agg.max_drawdown_pct, oos_sharpe_annualized=agg.sharpe_annualized,
        wf_efficiency=wf_eff, strategy_picks=picks,
        initial_equity=initial_equity, final_equity=equity)
    if verbose:
        logger.info(f"WF:{symbol} {nf} fold · OOS getiri {res.oos_total_return_pct:+.1f}% · "
                    f"verimlilik {wf_eff:.2f} · seçimler {picks}")
    return res


def print_walk_forward(r: WalkForwardResult) -> None:
    print("=" * 72)
    print(f"WALK-FORWARD — {r.symbol} ({r.interval}) · metrik: {r.metric}")
    print("=" * 72)
    print(f"Adaylar     : {', '.join(r.candidates)}")
    print(f"Fold sayisi : {len(r.folds)}")
    print("-" * 72)
    print(f"{'Fold':>4} {'Secilen':>14} {'IS':>10} {'OOS':>10} {'Islem':>6} {'OOS%':>8}")
    for f in r.folds:
        print(f"{f.fold:>4} {f.chosen:>14} {f.is_metric:>10.3f} {f.oos_metric:>10.3f} "
              f"{f.oos_trades:>6} {f.oos_return_pct:>+8.2f}")
    print("-" * 72)
    print(f"Strateji secimleri : {r.strategy_picks}")
    print(f"OOS islem / WR     : {r.oos_trades_count} / {r.oos_win_rate*100:.1f}%")
    print(f"OOS toplam getiri  : {r.oos_total_return_pct:+.2f}%")
    print(f"OOS max drawdown   : {r.oos_max_drawdown_pct:.2f}%")
    print(f"OOS Sharpe         : {r.oos_sharpe_annualized:.2f}")
    print(f"Initial -> Final   : {r.initial_equity:,.0f} -> {r.final_equity:,.0f} USD")
    eff = r.wf_efficiency
    verdict = "ROBUST" if eff >= 0.5 else ("ZAYIF" if eff > 0 else "OVERFIT/NEGATIF")
    print(f"WF verimliligi     : {eff:.2f}  → {verdict}  (OOS metrik / IS metrik; <1 düşüş)")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------
def print_report(r: BacktestResult) -> None:
    print("=" * 72)
    print(f"BACKTEST RAPORU - {r.symbol} ({r.interval})")
    print("=" * 72)
    print(f"Bar sayisi           : {r.bars_total}")
    print(f"Initial -> Final NLV : {r.initial_equity:,.2f} -> {_metric_text(r.final_equity)} USD")
    print(f"Toplam Getiri        : {_metric_text(r.total_return_pct)}%")
    print(f"Max Drawdown         : {_metric_text(r.max_drawdown_pct)}%")
    print(f"Sharpe (yillik)      : {_metric_text(r.sharpe_annualized)}")
    print(f"Sortino (yillik)     : {_metric_text(r.sortino_annualized)}")
    if r.kill_switch_at is not None and r.kill_switch_idle_pct is not None:
        print(f"Kill-switch          : {r.kill_switch_at} -> karar barlarinin %{r.kill_switch_idle_pct:.1f}'inde "
              "YENI GIRIS YOK (metrikler donemin tamamini olcmuyor)")
    if r.option_data is not None:
        print("Dolt veri/icra       : " + json.dumps({k: v for k, v in r.option_data.items() if k != "fills"}, ensure_ascii=False, default=str))
    print("-" * 72)
    print(f"Islem sayisi         : {r.trades_count} (W:{r.wins} / L:{r.losses})")
    print(f"Win Rate             : {r.win_rate*100:.1f}%")
    print(f"Ortalama Kazanc      : {r.avg_win:+.2f}")
    print(f"Ortalama Kayip       : {r.avg_loss:+.2f}")
    print(f"Expectancy (islem)   : {r.expectancy:+.2f} USD")
    pf_txt = f"{r.profit_factor:.2f}" if r.profit_factor is not None else "N/A (zarar yok)"
    print(f"Profit Factor        : {pf_txt}")
    print("-" * 72)
    if r.monte_carlo:
        mc = r.monte_carlo
        print(f"Monte Carlo ({mc['iterations']} iter, islem SIRASI karistirilir):")
        # Tur-95: her yol ayni PnL kumesini tasir → final p05=p50=p95; tek deger basilir.
        print(f"  Final NLV (siradan bagimsiz) : {mc['final_equity_p50']:,.0f}")
        print(f"  Max DD     p05/p50/p95 : "
              f"{mc['max_drawdown_p05']:.2f}% / {mc['max_drawdown_p50']:.2f}% / "
              f"{mc['max_drawdown_p95']:.2f}%")
        print(f"  Iflas (yolda <= 0)     : {mc['ruin_probability']*100:.2f}%")
    print("=" * 72)


def save_report(r: BacktestResult, path: Path | None = None) -> Path:
    symbol = re.sub(r'[^\w.,()-]', '_', r.symbol)
    path = path or (LOG_DIR / f"backtest_{symbol}_{r.interval}_{datetime.now():%Y%m%d_%H%M%S}.json")
    payload = asdict(r)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Nihai Algoritma — Backtest Motoru")
    ap.add_argument("--symbol", nargs="+", required=True, help="bir veya daha çok sembol")
    ap.add_argument("--interval", default=None, choices=["1m", "5m", "15m", "1h", "1d"])
    ap.add_argument("--period", default=None, help="örn. 6mo, 1y, 2y, 5y")
    ap.add_argument("--from", dest="start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--to", dest="end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--equity", type=float, default=INITIAL_EQUITY)
    ap.add_argument("--mc", type=int, default=0, help="Monte Carlo iterasyon sayısı (0=kapalı)")
    ap.add_argument("--save", action="store_true", help="JSON raporu logs/ altına kaydet")
    ap.add_argument("--provider", default=None, choices=[None, "yfinance", "tradingview"],
                    help="Veri sağlayıcı (varsayılan: env DATA_PROVIDER)")
    # Tur-106 (BULGU-106-12): seçenekler elle yazılmış 7'lik bayat bir alt kümeydi (STRATEGY_MODES 15 kip; quant_trend,
    # trend_pullback, vol_breakout … seçilemiyordu) ve varsayılan (canlı `quant_trend`) 1h'de her sembolde 0 işlem verip
    # sessizce boş rapor basıyordu.
    ap.add_argument("--strategy", default=None, choices=list(STRATEGY_MODES),
                    help="Strateji modu (varsayılan: config ACTIVE_STRATEGY)")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--portfolio", action="store_true",
                    help="Tüm sembolleri PAYLAŞILAN portföyde test et (Madde 14 tavanları)")
    modes.add_argument("--options", action="store_true",
                    help="Yerel Dolt zinciriyle günlük bid/ask opsiyon testi (varsayılan 1d)")
    modes.add_argument("--options-synthetic", action="store_true",
                    help="Açıkça sentetik Black-Scholes opsiyon deneyi (gerçek zincir kullanmaz)")
    ap.add_argument("--options-file", type=Path, help="Yerel option_chain CSV yolu (--options ile)")
    modes.add_argument("--walk-forward", dest="walk_forward", action="store_true",
                    help="Walk-forward: IS'te en iyi stratejiyi seç, OOS'ta test et (overfitting tespiti)")
    ap.add_argument("--is-bars", dest="is_bars", type=int, default=180, help="IS pencere bar sayısı")
    ap.add_argument("--oos-bars", dest="oos_bars", type=int, default=60, help="OOS pencere bar sayısı")
    ap.add_argument("--wf-metric", dest="wf_metric", default="expectancy",
                    choices=["expectancy", "sharpe", "return"], help="WF seçim metriği")
    args = ap.parse_args()
    args.interval = args.interval or ("1d" if args.options else "1h")
    if args.options and args.interval != "1d":
        ap.error("Dolt verisi günlük: --options için --interval 1d kullanın.")
    if args.options_file and not args.options:
        ap.error("--options-file yalnız --options ile kullanılabilir.")
    from nihai.runtime_strategy import (
        settings as _strateji,  # generate_signal'ın varsayılanı (pano seçimi > .env)
    )
    if (args.strategy or _strateji()["strategy"]) == "quant_trend" and args.interval != "1d" and not (args.walk_forward or args.options or args.options_synthetic):
        ap.error("quant_trend günlük bar ister (≥253 bar): --interval 1d --period 2y gibi kullanın "
                 "(1h barda her sembol 0 işlem verir).")

    if args.walk_forward:
        for sym in args.symbol:
            r = walk_forward(
                sym, interval=args.interval,
                period=args.period if not args.start else None,
                start=args.start, end=args.end, initial_equity=args.equity,
                is_bars=args.is_bars, oos_bars=args.oos_bars,
                metric=args.wf_metric, provider_name=args.provider, verbose=True)
            print_walk_forward(r)
        return

    common = {
        "interval": args.interval,
        "period": args.period if not args.start else None,
        "start": args.start, "end": args.end,
        "initial_equity": args.equity, "mc_iters": args.mc,
        "provider_name": args.provider, "strategy_mode": args.strategy, "verbose": True,
    }

    if args.portfolio:
        r = backtest_portfolio(args.symbol, **common)
        print_report(r)
        if args.save:
            print(f"\n→ Rapor kaydedildi: {save_report(r)}\n")
        return

    runner = backtest_option_chain if args.options else backtest_option_synth if args.options_synthetic else backtest_symbol
    if args.options:
        common["options_file"] = args.options_file
    for sym in args.symbol:
        try:
            r = runner(sym, **common)
        except ValueError as exc:
            ap.error(str(exc))
        print_report(r)
        if args.save:
            p = save_report(r)
            print(f"\n→ Rapor kaydedildi: {p}\n")


if __name__ == "__main__":
    main()
