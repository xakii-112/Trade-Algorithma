"""
dual_tier.py — İki-katmanlı paralel tarama çalışma zamanı.

Mimari:
- "Odak" katmanı: en iyi N sembol (watchlist'in başı) → TradingView ile taranır.
- "Geniş" katmanı: geri kalan tüm evren → yfinance ile taranır.
- İki katman AYRI thread'lerde GERÇEKTEN paralel çalışır (yavaş kısım = veri çekme).
- Ortak portföy/state (pozisyon açma, risk sayaçları, devre kesici, state.json),
  eşzamanlı yazıma karşı PORTFOLIO_LOCK ile korunur. Her thread kilidi YALNIZCA
  sinyal-kabul→pozisyon-açma kritik bölümünde tutar; veri çekme + sinyal üretimi
  kilit dışında paralel kalır.

Tekrar/çakışma engeli: odak = watchlist[:N], geniş = watchlist[N:]. Örtüşme yok →
bir sembol iki katmanda birden taranmaz, çift sinyal olmaz.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable

from loguru import logger

from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR

# Tüm portföy/state mutasyonlarını seri hale getiren TEK küresel kilit.
# concurrency.py'daki kilidi yeniden ihraç ederiz — web_dashboard ve
# position_manager da aynı nesneyi kullanır; iki tarama thread'i + ana döngü +
# web uçları hepsi aynı kilitle dışlanır (ayrı kilit = yarış koşulu). RLock:
# aynı thread içinde yeniden girilebilir (scan_symbol içinde iç içe çağrı güvenli).
from nihai.execution.concurrency import PORTFOLIO_LOCK  # noqa: F401 (yeniden ihraç)

# Katman adları — TierScanner thread'i BU adlarla kurulur (main.py `_make_tier`).
TIER_NAMES = ("odak", "genis")


def current_tier() -> str:
    """Çağıran thread bir katman tarayıcısıysa katman adı, değilse "" (BULGU-59-4).

    Neden thread ADI: `TierScanner` zaten `threading.Thread(name=...)` ile kuruluyor ve
    `scan_one` doğrudan o thread'de koşuyor → yeni bir bağlam değişkeni/parametre zinciri
    GEREKMEZ. Ana döngü, pano ve watchdog thread'leri bu adları taşımaz → "" döner
    (dual-tier kapalıyken davranış değişmez)."""
    name = threading.current_thread().name
    return name if name in TIER_NAMES else ""


class TierScanner(threading.Thread):
    """Bir katmanın tarama döngüsü: kendi partisyonunu kendi sağlayıcısıyla,
    bitince TIER_CYCLE_GAP bekleyip yeniden tarar."""

    def __init__(self, *, name: str, provider, get_symbols: Callable[[], list[str]],
                 scan_one: Callable[[str, object], None], throttle: float = 0.0,
                 cycle_gap: float = 5.0,
                 gate: Callable[[], tuple[bool, str]] | None = None):
        super().__init__(name=name, daemon=True)
        self.tier_name = name
        self.provider = provider
        self.get_symbols = get_symbols
        self.scan_one = scan_one
        self.throttle = throttle
        self.cycle_gap = cycle_gap
        # Tur-89 (BULGU-89-4): kapı kapalıyken (seans dışı) tur KOŞULMAZ. Eskiden iki katman seans
        # dışında 5 sn'de bir 550 sembolü tarayıp her birini SESSION_GUARD ile reddediyordu:
        # 22 dk'da 31.404 karar satırı + 5 sn'de bir tier_*.json yazımı (OneDrive senkronu).
        self.gate = gate
        self._stop_event = threading.Event()
        self.cycles = 0
        self.last_cycle_secs = 0.0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info(f"[tier:{self.tier_name}] başladı | sağlayıcı={getattr(self.provider, 'name', '?')}")
        while not self._stop_event.is_set():
            if self.gate is not None and not self.gate()[0]:
                self._stop_event.wait(max(self.cycle_gap, 30.0))
                continue
            symbols = self.get_symbols()
            if not symbols:
                # Henüz watchlist dolmadı (ör. ilk tarama sürüyor) → kısa bekle
                self._stop_event.wait(self.cycle_gap)
                continue
            t0 = time.time()
            scanned = 0
            timings = []
            for sym in symbols:
                if self._stop_event.is_set():
                    break
                started = time.monotonic()
                error = None
                try:
                    self.scan_one(sym, self.provider)
                    scanned += 1
                except Exception as e:
                    error = type(e).__name__
                    logger.exception(f"[tier:{self.tier_name}] {sym} tarama hatası: {e}")
                timings.append({"symbol": sym, "seconds": round(time.monotonic() - started, 4), "error": error})
                if self.throttle > 0:
                    self._stop_event.wait(self.throttle)
            self.last_cycle_secs = time.time() - t0
            self.cycles += 1
            if self.tier_name in TIER_NAMES:
                try:
                    atomic_write_text(DATA_DIR / f"tier_{self.tier_name}.json", json.dumps({
                        "tier": self.tier_name, "measured": True, "updated_at": time.time(),
                        "provider": getattr(self.provider, "name", "?"), "cycles": self.cycles,
                        "last_cycle_seconds": self.last_cycle_secs, "symbols": timings}))
                except OSError as exc:
                    logger.warning(f"[tier:{self.tier_name}] karne yazılamadı: {type(exc).__name__}")
            # Tur-89: her tur değil, ilk ve her 60. tur (odak katmanı ~5 sn'de bir döner; karne
            # tier_*.json'da zaten her turda tazelenir). Hatalı tur her zaman yazılır.
            if self.cycles == 1 or self.cycles % 60 == 0 or scanned < len(symbols):
                logger.info(f"[tier:{self.tier_name}] tur {self.cycles} bitti: "
                            f"{scanned}/{len(symbols)} sembol, {self.last_cycle_secs:.0f} sn")
            # Bir sonraki tura başlamadan kısa nefes (durdurmaya duyarlı)
            self._stop_event.wait(self.cycle_gap)
        logger.info(f"[tier:{self.tier_name}] durduruldu ({self.cycles} tur)")
