"""
duo.py — İki strateji varyasyonunu TEK process'te, PAYLAŞIMLI veri havuzuyla
A/B (champion/challenger) çalıştırır.

Tasarım ("havuz" deseni):
- Tek CachingProvider = veri havuzu. Her sembol BİR KEZ çekilir; sembol-DIŞ/
  varyasyon-İÇ döngü sayesinde ikinci varyasyon ms sonra cache'ten BİREBİR aynı
  veriyi okur → adil A/B, çift çekme yok (~60 çağrı/dk korunur, 2× olmaz).
- Her varyasyon kendi DEFTERİ (kasası): ayrı state.json + equity_curve.csv
  (data/duo/<isim>/), ayrı RiskManager / PositionManager / CircuitBreaker.
- Ortak: veri havuzu + global gate CONFIG (frozen). Varyasyonlar strateji TİPİYLE
  ayrışır (strategy_mode) ve aynı tarama yolundan (main.scan_symbol) geçer.

Varyasyonlar (EDGE_FINDINGS dayanaklı): champion=event_breakout (en dayanıklı
baseline), challenger=trend_pullback (§9.1 en yüksek WR ~%44; aynı 1h yola uyumlu).

Kullanım:
    .venv\\Scripts\\python.exe duo.py            # canlı A/B
    .venv\\Scripts\\python.exe duo.py --once      # tek tur (test/CI)

NOT: Sinyal-only/paper. Mevcut tek-strateji run_loop'a (main.py) DOKUNMAZ.
"""
from __future__ import annotations

import argparse
import time as time_mod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

import nihai.execution.watchlist as watchlist
from nihai.config import DATA_DIR, DATA_PROVIDER, INITIAL_EQUITY
from nihai.console import ensure_utf8_stdout
from nihai.data.data_provider import get_provider
from nihai.execution.main import is_market_open, now_et_naive, scan_symbol
from nihai.execution.state import StateStore
from nihai.risk.position_manager import PositionManager, SlippageModel
from nihai.risk.risk_manager import PortfolioState, RiskManager
from nihai.risk.safety import CircuitBreaker

# (isim, strategy_mode) — her varyasyon aynı veriyle kendi seçimini yapar.
DUO_SPEC: tuple[tuple[str, str], ...] = (
    ("champion", "event_breakout"),
    ("challenger", "trend_pullback"),
)


@dataclass
class Variation:
    name: str
    strategy_mode: str
    store: StateStore
    portfolio: PortfolioState
    risk_mgr: RiskManager
    pmgr: PositionManager
    breaker: CircuitBreaker
    equity_csv: Path
    start_equity: float


def build_variation(name: str, strategy_mode: str, initial_equity: float) -> Variation:
    """Bir varyasyonun bağımsız defterini (kasa) kurar — ayrı state/equity yolları."""
    base = DATA_DIR / "duo" / name
    base.mkdir(parents=True, exist_ok=True)
    store = StateStore(path=base / "state.json")
    portfolio = store.load_or_init(initial_equity)
    risk_mgr = RiskManager(portfolio)
    pmgr = PositionManager(portfolio, SlippageModel())
    rehydrated = pmgr.rehydrate_trackers()
    if rehydrated:
        logger.info(f"[duo:{name}] {rehydrated} açık pozisyon tracker'ı yeniden kuruldu")
    breaker = CircuitBreaker()

    def _on_close(ce, _store: StateStore = store) -> None:  # her varyasyon kendi defterine yazar
        _store.record_trade_pnl(ce.realized_pnl)
        _store.register_close_for_wash_sale(ce.pos.symbol, ce.realized_pnl, ce.exit_time)
        mult = 100 if ce.pos.asset_type == "option" else 1
        _store.add_unsettled_proceeds(ce.exit_price * ce.pos.quantity * mult, ce.exit_time.date())
    pmgr.add_on_close(_on_close)

    return Variation(
        name=name, strategy_mode=strategy_mode, store=store, portfolio=portfolio,
        risk_mgr=risk_mgr, pmgr=pmgr, breaker=breaker,
        equity_csv=base / "equity_curve.csv", start_equity=portfolio.equity,
    )


def _append_equity(path: Path, equity: float, when: datetime) -> None:
    try:
        new = not path.exists()
        with path.open("a", encoding="utf-8") as f:
            if new:
                f.write("timestamp,equity\n")
            f.write(f"{when.isoformat()},{equity:.2f}\n")
    except Exception as e:
        logger.debug(f"[duo] equity yazılamadı {path}: {e}")


def _emit_comparison(variations: list[Variation], when: datetime) -> None:
    """İki kasayı yan yana logla (asıl kıyas: equity eğrisi, WR değil)."""
    parts = []
    for v in variations:
        p = v.portfolio
        ret = (p.equity / v.start_equity - 1.0) * 100 if v.start_equity else 0.0
        ks = " KILL" if v.store.kill_switch_active else ""
        parts.append(f"{v.name}[{v.strategy_mode}] NLV ${p.equity:,.0f} "
                     f"({ret:+.2f}%) poz {len(p.open_positions)} gPnL {p.daily_pnl:+.0f}{ks}")
    logger.info(f"[duo:compare] {when:%H:%M} | " + " || ".join(parts))


def run_duo(once: bool = False) -> None:
    provider = get_provider(DATA_PROVIDER)  # PAYLAŞIMLI HAVUZ (CachingProvider)
    logger.info(f"[duo] havuz (paylaşımlı veri): {provider.name}")
    variations = [build_variation(n, m, INITIAL_EQUITY) for n, m in DUO_SPEC]
    for v in variations:
        logger.info(f"[duo] varyasyon '{v.name}' | strateji={v.strategy_mode} | "
                    f"kasa=${v.portfolio.equity:,.0f} | açık poz={len(v.portfolio.open_positions)}")



    while True:
        if is_market_open():
            now = now_et_naive()
            # 1) Pozisyon yönetimi — her varyasyon kendi defteri (paylaşımlı havuz)
            for v in variations:
                try:
                    v.pmgr.update_all_with_1m_bars(provider, now)
                    v.pmgr.update_all_with_quotes(provider, now, skip_stocks=True)
                except Exception as e:
                    logger.exception(f"[duo:{v.name}] pozisyon güncelleme hatası: {e}")
            # 2) PAYLAŞIMLI tarama: sembol-DIŞ, varyasyon-İÇ → veri BİR KEZ çekilir,
            #    ikinci varyasyon cache'ten BİREBİR aynı barı okur (adil A/B).
            for sym in watchlist.current():
                for v in variations:
                    if v.store.kill_switch_active:
                        continue
                    try:
                        scan_symbol(sym, provider, v.risk_mgr, v.store, v.breaker, v.pmgr,
                                    strategy_mode=v.strategy_mode)
                    except Exception as e:
                        logger.exception(f"[duo:{v.name}] {sym} tarama hatası: {e}")
            # 3) Kalıcılık + kıyas
            for v in variations:
                _append_equity(v.equity_csv, v.portfolio.equity, now)
                v.store.save()
            _emit_comparison(variations, now)
        else:
            logger.info("[duo] market kapalı — tarama yok")

        if once:
            break
        time_mod.sleep(60)


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    ap = argparse.ArgumentParser(description="İki strateji varyasyonu — paylaşımlı havuz A/B")
    ap.add_argument("--once", action="store_true", help="tek tur çalış (test)")
    args = ap.parse_args()
    run_duo(once=args.once)
