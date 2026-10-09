"""
position_manager.py — Sanal pozisyon yaşam döngüsü + otomatik post-mortem.

PDF Referansları:
- Madde 4  : Gap (boşluk) yönetimi — açılış fiyatından çıkış simülasyonu
- Madde 6  : Disiplinli çıkış (robotik infaz)
- Madde 11 : Zaman-bazlı stop
- Madde 28 : Vade yakınında kapat (DTE eşiği)
- Madde 38 : Trade post-mortem otomatik tetik
- Madde 50 : Realized slippage

Bu modül HEM canlı sinyal döngüsünde HEM backtest motorunda kullanılır.
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
from loguru import logger

from nihai.config import ACTIVE_STRATEGY, CONFIG
from nihai.data.data_provider import DataProvider
from nihai.execution.concurrency import PORTFOLIO_LOCK
from nihai.execution.cost_attribution import COST_FIELDS, cost_summary, implementation_shortfall
from nihai.execution.dual_tier import current_tier
from nihai.risk.options_analyzer import GreekSet, days_to_expiry, is_pin_risk
from nihai.risk.risk_manager import PortfolioState, VirtualPosition
from nihai.signals.signal_engine import (
    TradeRecord,
    log_decision,
    log_trade_csv,
    log_trade_postmortem,
)
from nihai.signals.strategy import Signal, SignalAction, atr, is_quant_label


def liquidity_inputs(df) -> dict:
    """Sağlayıcının kapanmış barlarından hacim ve ATR/fiyat; ek veri isteği yok.

    Hacim ile volatilite AYNI bar ufkunda. ATR bir oynaklık vekilidir; eta ölçülmüş
    Kyle lambda değildir. Kısa/bozuk pencere tahmin üretmez.
    """
    missing = {"adv_shares": None, "bar_sigma": None}
    window = CONFIG.backtest.impact_adv_bars
    try:
        if len(df) < max(window, 15):
            return missing
        recent = df[["high", "low", "close", "volume"]].tail(max(window, 15)).to_numpy(dtype=float)
        if (not np.isfinite(recent).all() or (recent <= 0).any()
                or (recent[:, 0] < recent[:, 2]).any() or (recent[:, 1] > recent[:, 2]).any()):
            return missing
        sigma = float(atr(df).iloc[-1]) / float(recent[-1, 2])
        adv = float(recent[-window:, 3].mean())
        if not math.isfinite(sigma) or sigma < 0 or not math.isfinite(adv):
            return missing
        return {"adv_shares": adv, "bar_sigma": sigma}
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, OverflowError):
        return missing


# Tur-89 (BULGU-89-1): IEX tek-borsa kotasyonu NBBO değildir; defter pasifken bir taraf uzak kalır.
# Ölçüldü (2026-09-14→22, 298 giriş): 45 girişte "makas" %9-12 → yarım-makas kayması %5-6, 35 giriş
# HEDEFİN ÖTESİNDE doldu ve sanal hesap 257k → 19,5k $'a eridi. S&P evreninde NBBO makası <%0,5'tir.
# ponytail: sabit tavan; SIP beslemesine geçilirse ya da küçük hisse evrenine inilirse yeniden ölç.
MAX_CREDIBLE_STOCK_SPREAD = 0.01


def quote_spread_pct(quote, price: float) -> float | None:
    """Sağlığı çağıranda doğrulanmış kotenin tam makası / dolum referans fiyatı.

    Tavanı aşan makas ÖLÇÜM sayılmaz (None → taban kayma): tek borsanın ince defteri piyasanın
    dolum fiyatı değildir."""
    if getattr(quote, "spread_measured", False) is not True:
        return None
    try:
        bid, ask, reference = float(quote.bid), float(quote.ask), float(price)
        if all(math.isfinite(v) for v in (bid, ask, reference)) and 0 < bid <= ask and reference > 0:
            spread = (ask - bid) / reference
            return spread if spread <= MAX_CREDIBLE_STOCK_SPREAD else None
    except (AttributeError, TypeError, ValueError, OverflowError):
        pass
    return None


# ---------------------------------------------------------------------------
# Slipaj modeli — Madde 50
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlippageModel:
    """Yarım ölçülen makas + isteğe bağlı hacim katılımı etkisi (sanal icra).

        kayma/fiyat = taban + eta · (ATR/fiyat) · sqrt(Q / ortalama_bar_hacmi)

    Taban, mevcut tam makasın yarısıdır; kote yoksa yapılandırılmış sabit kesir.
    Q ve hacim aynı araç/adet biriminde; ATR ile hacim aynı bar ufkundadır.
    Ham Kyle lambda ($/adet), boyutsuz eta yerine konamaz. Eta bu evrende
    kalibre edilmedi; IMPACT_COEF=0 varsayılanı hacim etkisini kapatır.
    Kote de yokken eski sabit modelle bit-denktir. Eksik hacim/ATR etki üretmez.
    Yarım makas referans fiyat çevresinde simüle edilir; gerçek dolum kanıtı değil.
    """

    stock_slippage_pct: float = 0.001        # %0.1 hisse giriş/çıkış
    option_slippage_pct: float = 0.03        # %3 opsiyon (geniş makas)
    stock_commission_per_trade: float = 0.0  # ücretsiz brokerler
    impact_coef: float = field(default_factory=lambda: CONFIG.backtest.impact_coef)
    # Opsiyon kontrat başı maliyetler risk_manager.OPTION_FEES'den okunur

    def __post_init__(self):
        for value in (self.stock_slippage_pct, self.option_slippage_pct,
                      self.stock_commission_per_trade, self.impact_coef):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Slipaj/komisyon parametreleri sonlu ve negatif olmayan sayılar olmalı")
        if max(self.stock_slippage_pct, self.option_slippage_pct) >= 1:
            raise ValueError("Taban slipaj yüzde 100'den küçük olmalı")

    def slip_pct(self, asset_type: str, *, qty: float | None = None,
                 adv_shares: float | None = None, sigma: float | None = None,
                 spread_pct: float | None = None) -> float:
        """Efektif kayma kesri (taban + karekök etkisi). TEK kaynak: giriş ve çıkış
        aynı formülü kullanır, yalnız işaret farklıdır."""
        base = self.stock_slippage_pct if asset_type == "stock" else self.option_slippage_pct
        # Ölçülen makas varsa sabit tabanın YERİNE yarım makas; iki kez ücret yok.
        if spread_pct is not None and math.isfinite(spread_pct) and spread_pct >= 0:
            base = spread_pct / 2
        if base >= 1:
            raise ValueError("Makas referans fiyata göre aşırı büyük")
        if self.impact_coef <= 0:
            return base
        try:
            q, adv, s = float(qty or 0.0), float(adv_shares or 0.0), float(sigma or 0.0)
        except (TypeError, ValueError, OverflowError):
            return base
        if q <= 0 or adv <= 0 or s < 0 or not all(math.isfinite(v) for v in (q, adv, s)):
            return base          # ölçüm yok → taban (NaN dahil; NaN != NaN)
        slip = base + self.impact_coef * s * ((q / adv) ** 0.5)
        if not math.isfinite(slip) or slip >= 1:
            raise ValueError("Etki modeli geçerli pozitif dolum fiyatı üretemiyor")
        return slip

    def apply_entry(self, price: float, side: str, asset_type: str, *,
                    qty: float | None = None, adv_shares: float | None = None,
                    sigma: float | None = None, spread_pct: float | None = None) -> float:
        slip = self.slip_pct(asset_type, qty=qty, adv_shares=adv_shares, sigma=sigma, spread_pct=spread_pct)
        sign = +1 if side == "LONG" else -1
        fill = price * (1 + sign * slip)
        if not math.isfinite(fill) or fill <= 0:
            raise ValueError("Geçersiz dolum fiyatı")
        return fill

    def apply_exit(self, price: float, side: str, asset_type: str, *,
                   qty: float | None = None, adv_shares: float | None = None,
                   sigma: float | None = None, spread_pct: float | None = None) -> float:
        return self.apply_entry(price, "SHORT" if side == "LONG" else "LONG", asset_type,
                                qty=qty, adv_shares=adv_shares, sigma=sigma, spread_pct=spread_pct)


# ---------------------------------------------------------------------------
# In-flight tracker (MAE/MFE + bars_held). Diske yazılmaz; restart'ta sıfırlanır.
# ---------------------------------------------------------------------------
@dataclass
class PositionTracker:
    pos: VirtualPosition
    signal_reasons: list[str]
    sizing: dict
    bars_held: int = 0
    high_water: float = 0.0       # en iyi lehte fiyat
    low_water: float = 0.0        # en kötü aleyhte fiyat
    current_price: float = 0.0
    strategy_name: str = "EventBreakout V1"
    iv_at_entry: float | None = None
    iv_rank_at_entry: float | None = None
    rr_planned: float = 0.0
    initial_stop: float = 0.0     # breakeven için: girişteki stop (sonradan stop değişse de korunur)
    breakeven_done: bool = False  # breakeven bir kez uygulandı mı

    def update_with_bar(self, bar_high: float, bar_low: float, bar_close: float) -> None:
        """Intra-bar high/low ile MAE/MFE güncelle (gerçekçi)."""
        if self.high_water == 0 and self.low_water == 0:
            self.high_water = self.pos.entry_price
            self.low_water = self.pos.entry_price
            if self.initial_stop == 0:
                self.initial_stop = self.pos.stop
        if self.pos.side == "LONG":
            self.high_water = max(self.high_water, bar_high)
            self.low_water = min(self.low_water, bar_low)
        else:
            self.high_water = min(self.high_water, bar_low)   # kısa için "iyi" = düşük
            self.low_water = max(self.low_water, bar_high)
        self.current_price = bar_close
        self.bars_held += 1
        # BULGU-106-35: yeniden başlatmada kaybolmasın (state.json pozisyonla birlikte yazar).
        self.pos.en_iyi_fiyat, self.pos.en_kotu_fiyat, self.pos.bar_sayisi = self.high_water, self.low_water, self.bars_held

    def update_with_quote(self, price: float) -> None:
        """Canlı modda quote tabanlı (high/low yok)."""
        self.update_with_bar(price, price, price)

    @property
    def mae_pct(self) -> float:
        if self.pos.entry_price == 0:
            return 0.0
        if self.pos.side == "LONG":
            return (self.low_water - self.pos.entry_price) / self.pos.entry_price * 100
        return (self.pos.entry_price - self.low_water) / self.pos.entry_price * 100

    @property
    def mfe_pct(self) -> float:
        if self.pos.entry_price == 0:
            return 0.0
        if self.pos.side == "LONG":
            return (self.high_water - self.pos.entry_price) / self.pos.entry_price * 100
        return (self.pos.entry_price - self.high_water) / self.pos.entry_price * 100


# ---------------------------------------------------------------------------
# PositionManager
# ---------------------------------------------------------------------------
@dataclass
class CloseEvent:
    pos: VirtualPosition
    exit_price: float
    exit_trigger: str
    exit_time: datetime
    realized_pnl: float
    mae_pct: float
    mfe_pct: float
    bars_held: int
    cost_attribution: dict | None = None


def sabit_hedefle_cikar(strategy: str | None) -> bool:
    """Pozisyon sabit hedefte kapanır mı? Trailing kipinde (`trail_drop_target`) ve quant_trend'de HAYIR: hedef
    nominaldir (RR kapısı + pano), çıkışı izleyen stop yapar. `check_exit` ve pano AYNI kuralı okur (Tur-106)."""
    return not (CONFIG.risk.enable_trailing and CONFIG.risk.trail_drop_target) and not is_quant_label(strategy)


class PositionManager:
    """
    Sanal portföyde pozisyon aç/güncelle/kapat. Kapanışta otomatik
    trade_log post-mortem üretir (Madde 38).
    """

    def __init__(self, portfolio: PortfolioState, slippage: SlippageModel | None = None,
                 *, record_logs: bool = True):
        self.portfolio = portfolio
        self.slip = slippage or SlippageModel()
        self._trackers: dict[str, PositionTracker] = {}  # key: f"{symbol}_{entry_time_iso}"
        self._on_close_callbacks: list[Callable[[CloseEvent], None]] = []
        # Madde 25 (Tur-58) — ex-div uyarısı için (sembol, vade, strike, gün) defteri.
        # Süreç-ömürlü: yeniden başlatma uyarıyı bir kez tekrarlar, bu kabul edilebilir.
        self._exdiv_warned: set[tuple] = set()
        # BULGU-9: backtest aç/kapanışları üretim trade_log.txt / data/trades.csv /
        # decision_log'a akıyordu → postmortem/dashboard/VaR canlı-dışı veriyle
        # kirlenir. Backtest record_logs=False geçer; canlı (main/web/duo) yazar.
        self.record_logs = record_logs
        self.execution_costs = cost_summary([])
        # BULGU-59-2 (Tur-59) — INTRA-BAR SIRA: bir barda hem stop hem hedef DELİNİRSE
        # hangisinin önce olduğu bar OHLC'sinden çıkarılamaz. Bu kod stop'u önce sayar
        # (kötümser) ve bu DOĞRU varsayılan; kusur varsayımın kendisi değil, MARUZİYETİN
        # ÖLÇÜLMEMESİYDİ (Tur-33: "aradığım kusur olsaydı bu gösterge farklı görünür
        # müydü?"). Sayaçlar her koşuda kaç çıkışın belirsiz olduğunu söyler.
        #   intrabar_ambiguous : stop VE hedef aynı barda delindi (açılış ikisinin arasında)
        #   intrabar_resolved  : `intrabar_resolver` ile 1m alt-barla ÇÖZÜLDÜ
        #   intrabar_flipped   : çözüm stop-önce varsayımını ÇEVİRDİ (gerçek: hedef önce)
        # `intrabar_resolver(pos, bar_open, now_dt) -> "STOP"|"TARGET"|None`: opsiyonel.
        # None (varsayılan) → davranış eskisiyle BİT-DENK.
        self.intrabar_ambiguous = 0
        self.intrabar_resolved = 0
        self.intrabar_flipped = 0
        self.intrabar_resolver: Callable[[VirtualPosition, float, datetime], str | None] | None = None

    def add_on_close(self, callback: Callable[[CloseEvent], None]) -> None:
        """Madde 20 — Kapanışta kill-switch tetikleme vb. için hook."""
        self._on_close_callbacks.append(callback)

    # -----------------------------------------------------------------
    def rehydrate_trackers(self) -> int:
        """Madde 41 — Restart sonrası: state.json'dan yüklenen açık pozisyonlar için
        eksik tracker'ları yeniden oluştur. Aksi halde update_all_* bu pozisyonları
        'tracker yok' diye atlayıp asla stop/target/çıkış uygulamaz (izlenmeden kalır)."""
        n = 0
        with PORTFOLIO_LOCK:
            for pos in self.portfolio.open_positions:
                key = self._key(pos)
                if key not in self._trackers:
                    # Tur-89: ad ve planlanan RR pozisyondan geri kurulur (eskiden varsayılan ad + RR 0).
                    # Tur-106 (BULGU-106-31): risk GİRİŞTEKİ mesafedir. İzleyen stop lehe kaydıktan sonra
                    # |giriş − o anki stop| başlangıç riski değildir: çok günlük quant pozisyonu her gece yeniden
                    # başlatılan süreçte kapanınca deftere her stop çıkışı tam ±1R, planlanan RR onlarca kat
                    # yazılırdı (canlı TSM: stop girişin üstünde → RR 51, risk 7 $ ↔ gerçek 124 $). Kalıcı
                    # `izleyen_mesafe` (BULGU-97-3) varsa risk, RR ve tepe ondan kurulur.
                    kalici = pos.izleyen_mesafe if pos.izleyen_mesafe > 0 else 0.0
                    risk_ps = kalici or abs(pos.entry_price - pos.stop)
                    rr = abs(pos.target - pos.entry_price) / risk_ps if risk_ps > 0 and pos.target > 0 else 0.0
                    mult = 100 if pos.asset_type == "option" else 1
                    t = PositionTracker(
                        pos=pos, signal_reasons=[],
                        sizing={"risk_dollars": kalici * abs(pos.quantity) * mult if kalici else pos.risk_dollars},
                        current_price=pos.last_mark or pos.entry_price,
                        strategy_name=pos.strategy or f"{ACTIVE_STRATEGY} V1",
                        rr_planned=round(rr, 2),
                    )
                    if pos.en_iyi_fiyat > 0 and pos.en_kotu_fiyat > 0:
                        # BULGU-106-35: kalıcı izler varsa MAE/MFE/bar sayısı kaldığı yerden sürer. Başlangıç stopu
                        # ilk kurulumun yazacağı değerle aynı (o adım artık atlanır).
                        t.high_water, t.low_water, t.bars_held = pos.en_iyi_fiyat, pos.en_kotu_fiyat, pos.bar_sayisi
                        t.initial_stop = pos.stop
                    elif kalici and pos.stop > 0:
                        # stop = tepe ∓ mesafe (yalnız lehe ratchet) ⇒ tepe geri kurulabilir: MFE son oturumla
                        # sınırlı kalmaz. Dip (MAE) bilinmez → giriş; `update_with_bar` ilk kurulumu atlamasın
                        # diye başlangıç stopu da burada yazılır (izleyen stop onsuz çalışmaz).
                        t.high_water = pos.stop + kalici if pos.side == "LONG" else pos.stop - kalici
                        t.low_water = pos.entry_price
                        t.initial_stop = pos.stop
                    self._trackers[key] = t
                    n += 1
        return n

    # -----------------------------------------------------------------
    def _key(self, p: VirtualPosition) -> str:
        return f"{p.symbol}_{p.entry_time.isoformat()}_{p.side}_{id(p)}"

    # -----------------------------------------------------------------
    def open_from_signal(
        self,
        signal: Signal,
        sizing: dict,
        *,
        now: datetime,
        sector: str = "UNKNOWN",
        asset_type: str = "stock",
        option_type: str | None = None,
        strike: float | None = None,
        expiry: str | None = None,
        greeks: GreekSet | None = None,
        iv: float | None = None,
        iv_rank: float | None = None,
        strategy_name: str = "EventBreakout V1",
        adv_shares: float | None = None,
        bar_sigma: float | None = None,
        beta: float | None = None,
        signal_price: float | None = None,
        spread_pct: float | None = None,
    ) -> VirtualPosition:
        """Yeni sanal pozisyon. Slipaj uygulanmış giriş fiyatı kullanılır.

        adv_shares/bar_sigma (BULGU-59-3): hacim-katılımına duyarlı piyasa etkisi için
        NEDENSEL girdiler (karar barına kadarki ortalama hacim ve ATR/fiyat). Verilmezse
        (varsayılan) etki terimi uygulanmaz → taban kayması, eski davranışla bit-denk.
        Girdiler pozisyonda saklanır; bar güncellemesi varsa çıkıştan önce yenilenir.
        Yalnız quote/manüel çıkış yolunda yeni bar verisi yoksa son hacim/ATR taşınır.
        signal_price çapa/sonraki-bar değişiminden ÖNCEKİ referansı korur."""
        side = signal.action.value if signal.action in (SignalAction.LONG, SignalAction.SHORT) else "LONG"
        qty = sizing.get("contracts") or sizing.get("shares") or 0
        if not math.isfinite(qty) or qty <= 0 or int(qty) != qty:
            raise ValueError(f"Sıfır boyut: {sizing}")
        reference = signal.price if signal_price is None else signal_price
        if not all(math.isfinite(v) and v > 0 for v in (signal.price, reference)):
            raise ValueError("Sinyal ve karar fiyatı sonlu ve pozitif olmalı")

        fill_price = self.slip.apply_entry(signal.price, side, asset_type,
                                           qty=qty, adv_shares=adv_shares, sigma=bar_sigma,
                                           spread_pct=spread_pct)
        pos = VirtualPosition(
            symbol=signal.symbol,
            asset_type=asset_type,
            side=side,
            quantity=qty,
            entry_price=fill_price,
            # Tam datetime sakla (eskiden now.date() → gün-içi saat kaybı → post-mortem
            # "Pozisyon Suresi" gece-yarısından ölçülüp yanlış çıkıyordu). PDT/settle
            # mantığı zaten .date() ile çalışır, datetime date'in alt sınıfıdır.
            entry_time=now,   # tam datetime (yukarıdaki nota bak); önceki isinstance dalı ikisinde de now döndürüyordu
            stop=signal.stop,
            target=signal.target,
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            greeks=greeks,
            sector=sector,
            last_mark=fill_price,
            adv_shares=adv_shares,
            bar_sigma=bar_sigma,
            spread_pct=spread_pct,
            entry_signal_price=reference,
            entry_decision_price=signal.price,
            beta=beta,
            tier=current_tier(),
            strategy=strategy_name,
            izleyen_mesafe=(abs(fill_price - signal.stop)
                            if is_quant_label(strategy_name) and signal.stop > 0 else 0.0),
        )
        with PORTFOLIO_LOCK:
            self.portfolio.open_positions.append(pos)

        tracker = PositionTracker(
            pos=pos, signal_reasons=list(signal.reasons), sizing=sizing,
            strategy_name=strategy_name, iv_at_entry=iv, iv_rank_at_entry=iv_rank,
            rr_planned=signal.rr_ratio,
            current_price=fill_price,
        )
        self._trackers[self._key(pos)] = tracker

        slip_diff = fill_price - signal.price
        if self.record_logs:
            log_decision(
                "POSITION_OPEN",
                f"{pos.symbol} {side} {qty}x @ {fill_price:.4f} "
                f"(signal {signal.price:.4f}, slip {slip_diff:+.4f}) | "
                f"stop {pos.stop:.4f} target {pos.target:.4f} | RR {signal.rr_ratio:.2f}"
            )
        return pos

    # -----------------------------------------------------------------
    @staticmethod
    def _held_minutes(entry_time, now_dt) -> float | None:
        """Girişten bu yana geçen DAKİKA; okunamazsa None (çağıran yedek yola düşer).

        Tur-39: `bars_held` BAR değil GÜNCELLEME sayar ve canlı döngüde ~60 sn'de bir
        artar → tarama hızını değiştiren herhangi bir ayar (worker sayısı, uyku süresi)
        zaman-stopunun ufkunu SESSİZCE değiştiriyordu. Duvar saati bu bağı koparır."""
        try:
            return (now_dt - entry_time).total_seconds() / 60.0
        except (TypeError, AttributeError, ValueError):
            return None

    @staticmethod
    def _minutes_to_close(now_dt) -> float | None:
        """Seans kapanışına kalan DAKİKA; kapalı/okunamaz ise None (kural uygulanmaz).

        `session_remaining` erken kapanışları (yarım gün) da bilir — sabit 16:00
        karşılaştırması onları kaçırırdı."""
        try:
            from nihai.data.market_calendar import session_remaining
            rem = session_remaining(now_dt)      # saat dilimsiz = ET (takvim sözleşmesi)
        except Exception:
            return None
        return None if rem is None else rem.total_seconds() / 60.0

    def _intrabar_first(self, p: VirtualPosition, bar_open: float,
                        now_dt: datetime) -> str:
        """Aynı barda stop VE hedef delindi: hangisi önce? (BULGU-59-2)

        Sayacı her çağrıda artırır — MARUZİYET ölçümü resolver olmadan da birikir.
        `intrabar_resolver` yoksa (varsayılan) "STOP" döner = mevcut kötümser varsayım,
        bit-denk. Resolver 1m alt-barlarla gerçek sırayı verirse ve sıra ÇEVRİLİRSE
        `intrabar_flipped` artar; resolver'ın None'ı (veri yok) yine "STOP"a düşer —
        ölçüm yokluğu ölçüm gibi sunulmaz (Tur-31 disiplini)."""
        self.intrabar_ambiguous += 1
        fn = self.intrabar_resolver
        if fn is None:
            return "STOP"
        try:
            first = fn(p, bar_open, now_dt)
        except Exception:
            first = None
        if first not in ("STOP", "TARGET"):
            return "STOP"
        self.intrabar_resolved += 1
        if first == "TARGET":
            self.intrabar_flipped += 1
        return first

    def check_exit(self, t: PositionTracker, *,
                   bar_high: float, bar_low: float, bar_open: float,
                   bar_close: float | None = None,
                   now_dt: datetime) -> tuple[str, float] | None:
        """Çıkış koşullarını sırayla dene. Tetiklenirse (trigger, fill_price) döner.

        Madde 4 — Gap: bar açılışı stop'un üstünden/altından geçtiyse açılıştan çık.
        Madde 28 — Vade/DTE/pin riski.
        """
        p = t.pos
        # Madde 28 — Vade yakını DTE eşiği
        if p.asset_type == "option" and p.expiry:
            dte = days_to_expiry(p.expiry)
            # Madde 28 — eşik: strateji override'ı varsa O, yoksa global. Override alanı
            # (`OptionStrategyConfig.close_before_expiry_dte_override`) kendi yorumunda bu
            # zinciri TARİF EDİYORDU ama hiçbir kod onu okumuyordu: değer verilse sessizce
            # yok sayılırdı (Tur-50). Varsayılan None → davranış bit-denk.
            _ovr = CONFIG.option_strategy.close_before_expiry_dte_override
            dte_esik = _ovr if _ovr is not None else CONFIG.options.close_before_expiry_dte
            if dte <= dte_esik:
                return ("DTE_GUARD", bar_open)
            # Madde 28 — Pin riski (vade günü ATM)
            if p.strike:
                ref_price = bar_close if bar_close is not None else (bar_high + bar_low) / 2
                if is_pin_risk(p.strike, ref_price, dte):
                    return ("PIN_RISK", ref_price)

        # Trailing kipinde sabit hedefi kaldır (kazananı koştur) — yalnızca trailing stop çıkarır.
        # Tur-97: quant_trend pozisyonu HER ZAMAN böyle çıkar (trend takibi kazananı kesmez; nominal
        # hedef yalnız RR kapısı ve pano içindir) — küresel TRAIL bayrağından bağımsız.
        quant = is_quant_label(p.strategy)
        use_target = sabit_hedefle_cikar(p.strategy)

        # Madde 4 — Gap-aware stop/target
        # BULGU-59-2: AÇILIŞ bir SIRA KANITIDIR ve iki yönde de okunmalıdır. Kod gap'i
        # yalnız STOP tarafında okuyordu; bar hedefin ÖTESİNDE açıldığında (LONG için
        # open >= target) hedef daha ilk tikte doludur — sonra stop delinse bile. O bar
        # eskiden STOP yazıyordu: bu "kötümser" değil YANLIŞTIR (belirsiz değil,
        # belirlenmiş bir sıra ters okunuyordu). Gap dolumu `bar_open`tan yapılır
        # (hedeften daha iyi fiyat gerçekten alınırdı), stop gap'iyle simetrik.
        # Geriye kalan gerçek belirsizlik: açılış İKİSİNİN ARASINDA ve bar ikisini de
        # deliyor → stop-önce (kötümser) korunur, sayaç artar, resolver varsa sorulur.
        open_evidence = CONFIG.backtest.intrabar_policy != "stop_first"
        if p.side == "LONG":
            # Açılış stop'un altında → açılıştan çık (slipaj uygulanmıştır)
            if p.stop > 0 and bar_open <= p.stop:
                return ("GAP_STOP", bar_open)
            if open_evidence and use_target and p.target > 0 and bar_open >= p.target:
                return ("GAP_TARGET", bar_open)
            hit_stop = p.stop > 0 and bar_low <= p.stop
            hit_target = use_target and p.target > 0 and bar_high >= p.target
            if hit_stop and hit_target and self._intrabar_first(p, bar_open, now_dt) == "TARGET":
                return ("TARGET", p.target)
            if hit_stop:
                return ("STOP", p.stop)
            if hit_target:
                return ("TARGET", p.target)
        else:  # SHORT
            if p.stop > 0 and bar_open >= p.stop:
                return ("GAP_STOP", bar_open)
            if open_evidence and use_target and p.target > 0 and bar_open <= p.target:
                return ("GAP_TARGET", bar_open)
            hit_stop = p.stop > 0 and bar_high >= p.stop
            hit_target = use_target and p.target > 0 and bar_low <= p.target
            if hit_stop and hit_target and self._intrabar_first(p, bar_open, now_dt) == "TARGET":
                return ("TARGET", p.target)
            if hit_stop:
                return ("STOP", p.stop)
            if hit_target:
                return ("TARGET", p.target)

        # A2 — MFE give-back (R-tabanlı): lehte tepe (MFE) arm_r'yi aştıktan SONRA tepeden
        # giveback_r kadar R geri verilirse anında çık. ATR/R-uzayında olduğundan volatiliteye
        # uyarlanır ve breakeven/trailing ile tutarlıdır. high_water o ana dek görülen en iyi
        # fiyat → look-ahead-güvenli. Stop/target'tan SONRA, time-stop'tan ÖNCE değerlendirilir.
        if CONFIG.risk.enable_mfe_giveback and t.initial_stop > 0 and p.entry_price > 0:
            risk = abs(p.entry_price - t.initial_stop)
            cur = bar_close if bar_close is not None else t.current_price
            if risk > 0:
                if p.side == "LONG":
                    mfe_r = (t.high_water - p.entry_price) / risk
                    cur_r = (cur - p.entry_price) / risk
                else:
                    mfe_r = (p.entry_price - t.high_water) / risk
                    cur_r = (p.entry_price - cur) / risk
                if mfe_r >= CONFIG.risk.mfe_giveback_arm_r and (mfe_r - cur_r) >= CONFIG.risk.mfe_giveback_r:
                    return ("MFE_GIVEBACK", cur)

        # SEANS SONU DÜZLEŞME (Tur-39) — gecelik taşıma YOK. Zaman-stopundan ÖNCE ve
        # "yatay mı?" koşulundan BAĞIMSIZ: kapanışa dakikalar kala açık bir pozisyon,
        # kâr/zarar durumundan bağımsız olarak gecelik gap riskine maruzdur.
        # Tur-97: quant_trend GÜNLÜK ufuklu bir pozisyondur (21-252 günlük skor, 3σ günlük stop);
        # seans sonu düzleşmesi ve 90 dk'lık zaman stopu onu her gün kapatıp ertesi gün yeniden
        # açtırırdı (maliyet × ~250/yıl, strateji başka bir şeye dönüşürdü). Gecelik risk stop
        # mesafesinde zaten fiyatlıdır (σ gece boşluklarını içeren kapanış→kapanış getirisinden).
        if quant:
            return None
        if CONFIG.risk.flat_before_close_min > 0:
            to_close = self._minutes_to_close(now_dt)
            if to_close is not None and to_close <= CONFIG.risk.flat_before_close_min:
                return ("SESSION_FLAT", t.current_price)

        # Madde 11 — ZAMAN-bazlı stop (Tur-39: birim onarımı; gerekçe config.py'de).
        # Eşik artık DAKİKADIR; `bars_held` yalnız zaman okunamadığında yedek yoldur.
        # İKİ koşul da aranır (AND) ve bu BİLEREK böyledir:
        #   · `bars_held >= time_stop_bars` BACKTEST paritesini korur — orada her güncelleme
        #     GERÇEK bir bardır, dolayısıyla eşik eskisiyle BİREBİR aynı barda tetiklenir.
        #   · `held_min >= time_stop_minutes` CANLIDA gerçek ufku kurar — orada güncelleme
        #     ~60 sn'lik bir yoklamadır ve tek başına bars_held 12 dakikada dolardı.
        # AND olduğu için kural ESKİSİNDEN ERKEN tetiklenemez (yalnız geciktirir) —
        # değişimin yönü tek taraflıdır ve ölçülmemiş bir hızlanma getirmez.
        held_min = self._held_minutes(p.entry_time, now_dt)
        past_horizon = (t.bars_held >= CONFIG.risk.time_stop_bars
                        and (held_min is None or held_min >= CONFIG.risk.time_stop_minutes))
        if past_horizon:
            # Yatay seyrediyor mu? (±0.5 ATR yerine basit: fiyat giriş bandında)
            drift = abs(t.current_price - p.entry_price) / max(p.entry_price, 1e-9)
            if drift < 0.005:
                return ("TIME_STOP", t.current_price)

        return None

    # -----------------------------------------------------------------
    def update_all_with_bars(
        self, bars_by_symbol: dict[str, tuple[float, float, float, float]],
        *, now: datetime,
        liquidity_by_symbol: dict[str, dict] | None = None,
    ) -> list[CloseEvent]:
        """
        Tüm açık pozisyonları günceller. bars_by_symbol = {sym: (open, high, low, close)}.
        Çıkış tetiklenenleri kapatır, CloseEvent listesi döner.
        """
        closed: list[CloseEvent] = []
        with PORTFOLIO_LOCK:
            # Liste üzerinde değişiklik yaparken kopya gez.
            for pos in list(self.portfolio.open_positions):
                tracker = self._trackers.get(self._key(pos))
                if not tracker:
                    continue
                bars = bars_by_symbol.get(pos.symbol)
                if bars is None:
                    continue
                if pos.asset_type == "stock" and liquidity_by_symbol is not None:
                    inputs = liquidity_by_symbol.get(pos.symbol, {})
                    for name in ("adv_shares", "bar_sigma", "spread_pct"):
                        if name in inputs:
                            setattr(pos, name, inputs[name])
                o, h, l, c = bars
                tracker.update_with_bar(h, l, c)
                pos.last_mark = c   # canlı PnL / pano için

                exit_signal = self.check_exit(
                    tracker, bar_high=h, bar_low=l, bar_open=o, bar_close=c, now_dt=now,
                )
                if exit_signal is not None:
                    trigger, raw_price = exit_signal
                    close_evt = self._close(pos, raw_price, trigger, now)
                    closed.append(close_evt)
                else:
                    # Look-ahead-güvenli: stop'u yalnızca bu bar kapandıktan SONRA,
                    # gelecek barlar için breakeven'a/trailing'e çek.
                    self._apply_breakeven(tracker)
                    self._apply_trailing(tracker)
        return closed

    # -----------------------------------------------------------------
    def _apply_trailing(self, t: PositionTracker) -> None:
        """Trailing stop — stop'u en iyi-lehte fiyatın trail_r×(başlangıç risk) gerisinde
        sürükle (yalnızca lehe yönde). Sabit hedef kaldırıldıysa kazanan trail'e kadar koşar
        → ort.kazanç (dolayısıyla payoff/PF) WR'ı bozmadan büyür.

        Tur-97: quant_trend pozisyonu küresel bayraktan bağımsız olarak oynaklık İZLEYEN stopla
        (chandelier) yönetilir: mesafe = başlangıç riski (3σ), yalnız lehe ratchet."""
        p = t.pos
        quant = is_quant_label(p.strategy)
        if not CONFIG.risk.enable_trailing and not quant:
            return
        if t.initial_stop <= 0 or p.entry_price <= 0:
            return
        risk = abs(p.entry_price - t.initial_stop)
        if risk <= 0:
            return
        if quant:
            d = p.izleyen_mesafe if p.izleyen_mesafe > 0 else risk   # BULGU-97-3: kalıcı mesafe
            yeni = t.high_water - d if p.side == "LONG" else t.high_water + d
            if (p.side == "LONG" and yeni > p.stop) or (p.side != "LONG" and (p.stop == 0 or yeni < p.stop)):
                p.stop = yeni
            return

        # A1 — kademeli trail + kâr-tabanı kilidi. trail_tiered KAPALI → eski düz davranış
        # (byte-aynı): sabit trail_r, kilit yok. AÇIK → kâr (R) arttıkça mesafeyi sıkılaştır
        # ve kâr eşiği aşılınca tabanı kilitle. Her iki halde de stop yalnızca lehe ratchet eder.
        eff_trail_r = CONFIG.risk.trail_r
        lock_floor = 0.0
        if CONFIG.risk.trail_tiered:
            profit_r = ((t.high_water - p.entry_price) if p.side == "LONG"
                        else (p.entry_price - t.high_water)) / risk
            for at_r, tr in CONFIG.risk.trail_tiers:   # eşik-azalan sıralı: ilk eşleşen en sıkı
                if profit_r >= at_r:
                    eff_trail_r = tr
                    break
            if CONFIG.risk.trail_lock_trigger_r > 0 and profit_r >= CONFIG.risk.trail_lock_trigger_r:
                lock_floor = CONFIG.risk.trail_lock_floor_r * risk

        dist = eff_trail_r * risk
        if p.side == "LONG":
            new_stop = t.high_water - dist
            if lock_floor > 0:
                new_stop = max(new_stop, p.entry_price + lock_floor)
            if new_stop > p.stop:          # yalnızca yukarı sürükle
                p.stop = new_stop
        else:  # SHORT — high_water kısa için en iyi (en düşük) fiyat
            new_stop = t.high_water + dist
            if lock_floor > 0:
                new_stop = min(new_stop, p.entry_price - lock_floor)
            if p.stop == 0 or new_stop < p.stop:   # yalnızca aşağı sürükle
                p.stop = new_stop

    # -----------------------------------------------------------------
    def _apply_breakeven(self, t: PositionTracker) -> None:
        """Madde 11+ — Breakeven stop. Fiyat lehte +be_trigger_r×(başlangıç risk) gidince
        stop'u girişe (+ tampon) çek. Kazananı kesmez; potansiyel kaybedeni başabaşa çevirir."""
        if not CONFIG.risk.enable_breakeven or t.breakeven_done:
            return
        p = t.pos
        if t.initial_stop <= 0 or p.entry_price <= 0:
            return
        risk = abs(p.entry_price - t.initial_stop)
        if risk <= 0:
            return
        trigger_dist = CONFIG.risk.be_trigger_r * risk
        buf = CONFIG.risk.be_buffer_r * risk
        if p.side == "LONG":
            if t.high_water - p.entry_price >= trigger_dist:
                new_stop = p.entry_price + buf
                if new_stop > p.stop:          # yalnızca yukarı (asla gevşetme)
                    p.stop = new_stop
                    t.breakeven_done = True
        else:  # SHORT — high_water kısa için en iyi (en düşük) fiyat
            if p.entry_price - t.high_water >= trigger_dist:
                new_stop = p.entry_price - buf
                if p.stop == 0 or new_stop < p.stop:   # yalnızca aşağı
                    p.stop = new_stop
                    t.breakeven_done = True

    # -----------------------------------------------------------------
    def force_close(self, pos: VirtualPosition, price: float, reason: str,
                    now: datetime) -> CloseEvent:
        """Manuel / acil kapatma (Madde 47 flatten-all)."""
        with PORTFOLIO_LOCK:
            return self._close(pos, price, reason, now)

    def _close(self, pos: VirtualPosition, raw_price: float, trigger: str,
               now: datetime) -> CloseEvent:
        # NOT: Çağıran PORTFOLIO_LOCK'u tutmalıdır (update_all_*, force_close).
        fill_price = self.slip.apply_exit(raw_price, pos.side, pos.asset_type,
                                          qty=pos.quantity, adv_shares=pos.adv_shares,
                                          sigma=pos.bar_sigma, spread_pct=pos.spread_pct)

        # PnL
        multiplier = 100 if pos.asset_type == "option" else 1
        sign = +1 if pos.side == "LONG" else -1
        pnl = (fill_price - pos.entry_price) * pos.quantity * multiplier * sign

        # Madde 32 — Opsiyon komisyon / Madde 50 — hisse komisyonu
        if pos.asset_type == "option":
            from nihai.risk.risk_manager import OPTION_FEES
            per_contract = sum(OPTION_FEES.values())
            fees = per_contract * pos.quantity * 2
        else:
            fees = self.slip.stock_commission_per_trade * 2
        pnl -= fees

        entry_cost = implementation_shortfall(
            side="BUY" if sign == 1 else "SELL", quantity=pos.quantity,
            filled_quantity=pos.quantity, signal_price=pos.entry_signal_price,
            decision_price=pos.entry_decision_price, fill_price=pos.entry_price,
            fees=fees / 2, multiplier=multiplier)
        exit_cost = implementation_shortfall(
            side="SELL" if sign == 1 else "BUY", quantity=pos.quantity,
            filled_quantity=pos.quantity, signal_price=raw_price, decision_price=raw_price,
            fill_price=fill_price, fees=fees / 2, multiplier=multiplier)
        # ponytail: sanal dolum tam ve çıkış tetik→karar anlık; ölçülmüş broker gecikmesi değil.
        # Tur-78 (R3-B1): ölçülemeyen bacak (None) maliyeti bilinmez yapar, kapanışı durdurmaz.
        costs = {key: (entry_cost[key] + exit_cost[key]
                       if entry_cost and exit_cost and entry_cost[key] is not None
                       and exit_cost[key] is not None else None)
                 for key in COST_FIELDS}
        self.execution_costs = cost_summary([costs], self.execution_costs)
        tracker = self._trackers.pop(self._key(pos), None)

        # State güncelle
        self.portfolio.open_positions.remove(pos)
        self.portfolio.equity += pnl
        self.portfolio.cash_balance += pnl
        self.portfolio.daily_pnl += pnl
        if self.portfolio.equity > self.portfolio.ath_equity:
            self.portfolio.ath_equity = self.portfolio.equity
        if pnl < 0:
            self.portfolio.consecutive_stops += 1
        else:
            self.portfolio.consecutive_stops = 0

        # Madde 10, 23 — PDT day-trade sayacı (aynı gün aç-kapat)
        # Takip kapaliyken sayac hic isletilmez (config.regulation.pdt_enabled).
        if CONFIG.regulation.pdt_enabled:
            entry_date = pos.entry_time if isinstance(pos.entry_time, date) and \
                         not isinstance(pos.entry_time, datetime) else \
                         (pos.entry_time.date() if hasattr(pos.entry_time, "date") else now.date())
            if entry_date == now.date():
                self.portfolio.register_day_trade(now.date())   # Tur-90: yürüyen pencere günü

        mae = tracker.mae_pct if tracker else 0.0
        mfe = tracker.mfe_pct if tracker else 0.0
        bars_held = tracker.bars_held if tracker else 0

        # Madde 38 — otomatik post-mortem
        entry_dt = (datetime.combine(pos.entry_time, datetime.min.time())
                    if isinstance(pos.entry_time, date) and not isinstance(pos.entry_time, datetime)
                    else pos.entry_time)
        if pos.asset_type == "option":
            side_label = "Long Call" if pos.option_type == "c" else \
                         "Long Put" if pos.option_type == "p" else pos.side
            label = f"{pos.symbol} {pos.expiry} {(pos.option_type or '').upper()} {pos.strike} ({side_label})"
        else:
            label = f"{pos.symbol} ({pos.side} Stock x {pos.quantity})"

        rec = TradeRecord(
            asset_label=label,
            strategy=tracker.strategy_name if tracker else "EventBreakout V1",
            entry_time=entry_dt,
            entry_price=pos.entry_price,
            entry_greeks=pos.greeks,
            entry_iv=tracker.iv_at_entry if tracker else None,
            entry_iv_rank=tracker.iv_rank_at_entry if tracker else None,
            dte_at_entry=days_to_expiry(pos.expiry) if pos.expiry else None,
            risk_dollars=tracker.sizing.get("risk_dollars", 0.0) if tracker else 0.0,
            planned_rr=tracker.rr_planned if tracker else 0.0,
            contracts_or_shares=pos.quantity,
            exit_time=now,
            exit_price=fill_price,
            exit_trigger=trigger,
            mae_pct=mae,
            mfe_pct=mfe,
            realized_pnl=pnl,
            nlv_after=self.portfolio.equity,
            entry_reasons=list(tracker.signal_reasons) if tracker else [],
            realized_slippage=fill_price - raw_price,
            cost_attribution=costs,
        )
        if self.record_logs:
            log_trade_postmortem(rec)
            log_trade_csv(rec, pos)   # makine-okur kopya (data/trades.csv) — post-mortem/dashboard/VaR
            log_decision(
                "POSITION_CLOSE",
                f"{pos.symbol} {pos.side} {pos.quantity}x @ {fill_price:.4f} | "
                f"{trigger} | PnL {pnl:+.2f} | MAE {mae:+.1f}% MFE {mfe:+.1f}% | "
                f"NLV {self.portfolio.equity:.2f}"
            )

        close_evt = CloseEvent(
            pos=pos, exit_price=fill_price, exit_trigger=trigger, exit_time=now,
            realized_pnl=pnl, mae_pct=mae, mfe_pct=mfe, bars_held=bars_held,
            cost_attribution=costs,
        )

        # Madde 20 — kill-switch ve diğer hook'ları tetikle
        for cb in self._on_close_callbacks:
            try:
                cb(close_evt)
            except Exception as e:
                logger.error(f"[pmgr] on_close callback hatası: {e}")

        return close_evt

    # -----------------------------------------------------------------
    def _check_option_premium_exit(self, tracker: PositionTracker,
                                    current_premium: float) -> tuple[str, float] | None:
        """Madde 2 — Long opsiyon prim-bazlı çıkış (% loss/gain)."""
        entry = tracker.pos.entry_price
        if entry <= 0:
            return None
        cfg = CONFIG.option_strategy
        if current_premium <= entry * (1 - cfg.premium_stop_loss_pct):
            return ("PREMIUM_STOP", current_premium)
        if current_premium >= entry * (1 + cfg.premium_target_gain_pct):
            return ("PREMIUM_TARGET", current_premium)
        return None

    def _check_ex_dividend_exit(self, pos: VirtualPosition, current_premium: float,
                                provider: DataProvider) -> tuple[str, float] | None:
        """Madde 25 (Tur-58) — açık LONG CALL ex-temettüye ITM giriyorsa ne olur?

        Ex-tarihte dayanak ≈D düşer; uzun call temettüyü ALMAZ ve içsel değerin ≈D'sini
        yitirir. Bu sistem opsiyon KULLANAMAZ (sinyal-only, brokerda pozisyon yok) →
        tek uygulanabilir karşılık ex-tarihten ÖNCE KAPATMAKTIR.

        **Varsayılan yalnız İŞARETLER** (`pos.reasons`'a not + log): kural determinist
        olsa da defterin davranışını değiştirir ve bu projede davranış değişikliği
        opt-in'dir (Tur-57 borç bayrağı sözleşmesi). `EXDIV_GUARD=block` gerçek bir
        çıkış tetikleyicisine çevirir — kapatan taraf `EXDIV_EARLY_EXERCISE`tir.

        YALNIZ CANLI yoldan çağrılır (`update_all_with_quotes`): takvim BUGÜNÜN ex-div
        tarihini bilir, backtest barına uygulamak look-ahead olurdu (Madde 3).
        """
        if pos.asset_type != "option" or pos.option_type != "c" or not pos.strike:
            return None
        try:
            from nihai.data.event_calendar import ex_dividend_context
            from nihai.risk.options_analyzer import early_exercise_risk
            days_div, div_amt = ex_dividend_context(pos.symbol)
            if days_div is None:
                return None
            q = provider.get_quote(pos.symbol)
            underlying = float(q.last) if q and q.last > 0 else 0.0
            if underlying <= 0:
                return None                      # dayanak okunamadı → ölçüm yok, karar yok
            ee = early_exercise_risk(
                option_type="c", strike=float(pos.strike), underlying=underlying,
                option_mid=current_premium, days_to_ex_div=days_div,
                dividend=div_amt, symbol=pos.symbol,
            )
        except Exception as e:      # noqa: BLE001 — takvim yolu pozisyon yönetimini düşürmez
            logger.debug(f"[exdiv] {pos.symbol} eşik ölçülemedi: {e}")
            return None
        if not ee["risk"]:
            return None
        tag = "EXDIV_EARLY_EXERCISE" if ee["measured"] else "EXDIV_EARLY_EXERCISE_UNMEASURED"
        # Uyarı GÜNDE BİR kez (canlı döngü aynı pozisyonu ~60 sn'de bir yokluyor; her
        # yoklamada satır yazmak decision_log'u şişirirdi — Tur-44'ün "sayım TAM, log
        # SEYREK" deseni). Anahtar pozisyonu ve GÜNÜ taşır, `VirtualPosition`'a alan
        # EKLENMEZ: o dataclass state.json'a serileşir ve şema göçü gerektirirdi.
        key = (pos.symbol, pos.expiry, float(pos.strike), date.today().isoformat())
        if key not in self._exdiv_warned:
            self._exdiv_warned.add(key)
            logger.warning(
                f"[exdiv] {pos.symbol} {pos.expiry} C{pos.strike}: ex-div {days_div} gün, "
                f"ekstrinsik={ee['extrinsic']} D={ee['dividend']} → {ee['reason']}")
        if (os.getenv("EXDIV_GUARD") or "").strip().lower() != "block":
            return None                          # varsayılan: yalnız işaretle
        return (tag, current_premium)

    # -----------------------------------------------------------------
    def update_all_with_quotes(self, provider: DataProvider, now: datetime,
                               *, skip_stocks: bool = False) -> list[CloseEvent]:
        """
        Canlı mod: hisse için quote; opsiyon için zincirden premium yeniden çekilir.

        skip_stocks=True: hisseler bu çağrıda atlanır. Canlı döngüde hisseler zaten
        update_all_with_1m_bars ile granüler güncellendiği için, ikinci kez burada
        güncellenmesi bars_held'i çift sayıp TIME_STOP'u erken tetikliyordu (Madde 11).
        """
        from nihai.risk.option_strategy import get_option_premium_now

        closed: list[CloseEvent] = []
        with PORTFOLIO_LOCK:
            for pos in list(self.portfolio.open_positions):
                tracker = self._trackers.get(self._key(pos))
                if not tracker:
                    continue
                if skip_stocks and pos.asset_type == "stock":
                    continue

                if pos.asset_type == "option" and pos.option_type and pos.expiry and pos.strike:
                    # Opsiyon — anlık premium çek
                    prem = get_option_premium_now(provider, pos.symbol, pos.option_type,
                                                   pos.strike, pos.expiry)
                    if prem is None or prem <= 0:
                        continue
                    tracker.update_with_quote(prem)
                    pos.last_mark = prem   # canlı PnL / pano için
                    # 1) Madde 28 — DTE / pin riski (check_exit içinde)
                    # 2) Madde 2 — Premium-based exit
                    # 3) Madde 25 — ex-temettü erken-kullanım eşiği (Tur-58)
                    exit_sig = (self._check_ex_dividend_exit(pos, prem, provider)
                                or self._check_option_premium_exit(tracker, prem)
                                or self.check_exit(tracker, bar_high=prem, bar_low=prem,
                                                   bar_open=prem, bar_close=prem, now_dt=now))
                    if exit_sig:
                        trigger, raw_price = exit_sig
                        closed.append(self._close(pos, raw_price, trigger, now))
                else:
                    # Hisse — standart quote
                    q = provider.get_quote(pos.symbol)
                    if q and q.last > 0:
                        bars = (q.last, q.last, q.last, q.last)
                        healthy, _ = provider.is_quote_healthy(q)
                        spread = quote_spread_pct(q, q.last) if healthy else None
                        sub_close = self.update_all_with_bars(
                            {pos.symbol: bars}, now=now,
                            liquidity_by_symbol={pos.symbol: {"spread_pct": spread}})
                        closed.extend(sub_close)
        return closed

    # -----------------------------------------------------------------
    def fetch_1m_bars(self, provider: DataProvider,
                      symbols: list[str] | None = None) -> dict:
        """Açık hisse pozisyonlarının son 1m barlarını sembol başına BİR KEZ çeker.

        Ağ çağrısı güncellemeden ayrıdır ki canlı döngü onu PORTFOLIO_LOCK DIŞINDA
        yapabilsin (Tur-80): kilit N sembollük `get_bars` boyunca tutulunca pano uçları
        ve tarama thread'leri bekliyordu. Hata → None (güncelleme o sembolü atlar).
        """
        if symbols is None:
            symbols = [p.symbol for p in list(self.portfolio.open_positions) if p.asset_type == "stock"]
        lookback = max(3, CONFIG.backtest.impact_adv_bars + 16) if self.slip.impact_coef else 3

        def _cek(sym: str):
            try:
                return provider.get_bars(sym, "1m", lookback_bars=lookback)
            except Exception as e:
                logger.debug(f"[1m] {sym} bar fetch: {e}")
                return None

        # Tur-82: çağrılar birbirinden bağımsız ve ağ-bekleme ağırlıklı (Tur-81 `resolve`
        # deseni) → havuzla çekilir; dönen sözlük (anahtar SIRASI dahil) sıralı döngününkiyle aynı.
        uniq = list(dict.fromkeys(symbols))
        if len(uniq) < 2:
            return {sym: _cek(sym) for sym in uniq}
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(8, len(uniq)), thread_name_prefix="pm-1m") as havuz:
            return dict(zip(uniq, havuz.map(_cek, uniq), strict=True))

    def update_all_with_1m_bars(self, provider: DataProvider, now: datetime,
                                prefetched: dict | None = None) -> list[CloseEvent]:
        """
        Madde 5 — Granüler stop kontrolü. Hisse pozisyonları için son birkaç
        1 dakikalık bar'ı çek; intra-bar high/low ile stop fitili tetiklenirse yakala.
        Opsiyon pozisyonları premium-based exit kullandığı için bu metod onlara
        dokunmaz (update_all_with_quotes orada).

        `prefetched`: `fetch_1m_bars` çıktısı; onda olmayan sembol (arada açılan
        pozisyon) burada çekilir. BULGU-80-5: eskiden döngü POZİSYON başınaydı ve her
        tur `update_all_with_bars`ı o sembolün TÜM pozisyonlarına uyguluyordu → aynı
        sembolde 2 pozisyon (küme sınırı 2) her barı iki kez görüyordu: `bars_held` 2×,
        breakeven/trailing açıkken aynı barın taşıdığı stop aynı barda tetikleniyordu.
        """
        symbols = [p.symbol for p in list(self.portfolio.open_positions) if p.asset_type == "stock"]
        frames = dict(prefetched or {})
        frames.update(self.fetch_1m_bars(provider, [s for s in symbols if s not in frames]))
        bars: dict[str, tuple[float, float, float, float]] = {}
        liquidity: dict[str, dict] = {}
        for sym in dict.fromkeys(symbols):
            df = frames[sym]
            if df is None or df.empty:
                continue
            last = df.iloc[-1]
            bars[sym] = (float(last["open"]), float(last["high"]),
                         float(last["low"]), float(last["close"]))
            # Son barın İÇİNDEKİ stop için o barın nihai hacmini kullanma.
            inputs = liquidity_inputs(df.iloc[:-1]) if self.slip.impact_coef else {}
            inputs["spread_pct"] = None  # OHLCV, giriş kotasyonunun hâlâ geçerli olduğunu kanıtlamaz
            liquidity[sym] = inputs
        return self.update_all_with_bars(bars, now=now, liquidity_by_symbol=liquidity)

    # -----------------------------------------------------------------
    def flatten_all(self, provider: DataProvider, now: datetime, reason: str = "FLATTEN_ALL") -> list[CloseEvent]:
        """Madde 47 — acil tüm pozisyonları kapat."""
        closed = []
        with PORTFOLIO_LOCK:
            for pos in list(self.portfolio.open_positions):
                q = provider.get_quote(pos.symbol)
                price = q.last if q else pos.entry_price
                closed.append(self.force_close(pos, price, reason, now))
        return closed
