"""
main.py — Orkestrasyon. Watchlist tarama, sinyal üretimi, log + bildirim.

PDF Referansları:
- Madde 22 : RTH dışı yeni sinyal yok
- Madde 36 : Heartbeat (seans-farkındalıklı)
- Madde 41 : Başlangıç mutabakatı (state ↔ runtime tutarlılık)
- Madde 42 : Devre kesici (safety) entegre
- Madde 45 : Data provider failover hook'u (henüz tek kaynak)

Çalıştırma:
    python main.py                # canlı sinyal modu (yfinance gecikmeli)
    python main.py --once         # tek tur tara, çık
"""
from __future__ import annotations

import argparse
import contextlib

# Yalnızca ana process dosya sink'i ekler. auto_screener paralel taramada
# ProcessPoolExecutor (spawn) kullanır; spawn edilen worker'lar main.py'yi yeniden
# import eder — guard olmadan her worker aynı system.log'a ikinci bir sink eklerdi
# (yinelenen satırlar + rotasyon yarışı). parent_process() ana'da None döner.
import multiprocessing as _mp
import os
import sys
import threading
import time as time_mod
from datetime import datetime, timedelta
from typing import cast

from loguru import logger

import nihai.execution.watchlist as watchlist
import nihai.observability as OBS
from nihai.config import (
    ACTIVE_STRATEGY,
    AUTO_SCREEN_ENABLED,
    AUTO_SCREEN_INTERVAL_HOURS,
    BROAD_PROVIDER,
    BROAD_THROTTLE,
    CONFIDENCE_SIZING_ENABLED,
    CONFIG,
    DATA_DIR,
    DATA_PROVIDER,
    DUAL_TIER_ENABLED,
    FOCUSED_PROVIDER,
    FOCUSED_THROTTLE,
    FOCUSED_TOP_N,
    HTF_INTERVAL,
    INITIAL_EQUITY,
    LIVE_ENTRY_ANCHOR,
    LIVE_SCAN_WORKERS,
    LOG_DIR,
    OPTION_MODE,
    PROB_SIZING_ENABLED,
    PROB_SIZING_FLOOR,
    PROB_SIZING_P_HI,
    PROB_SIZING_P_LO,
    RUN_MODE,
    SIGNAL_ALERT_ALWAYS,
    TIER_CYCLE_GAP,
    get_sector,
    validate_config,
)
from nihai.console import ensure_utf8_stdout
from nihai.data.data_provider import Interval, get_provider
from nihai.data.event_calendar import corporate_action_guard, event_guard
from nihai.data.iv_history import append_iv_today, get_iv_rank
from nihai.data.market_calendar import can_enter_new_position, is_market_open, kapanmis_gunluk, now_et_naive
from nihai.data.market_structure import halt_or_luld_guard, ssr_short_guard
from nihai.execution import ai_advisor, shadow_bridge
from nihai.execution.auto_screener import AutoScreener, scan_and_update
from nihai.execution.dual_tier import PORTFOLIO_LOCK, TierScanner
from nihai.execution.state import StateStore
from nihai.notifier import telegram_bot, terminal_dashboard
from nihai.risk.option_strategy import select_option_contract
from nihai.risk.options_analyzer import (
    days_to_expiry,
    estimate_atm_iv,
    filter_liquid_chain,
)
from nihai.risk.position_manager import (
    PositionManager,
    SlippageModel,
    liquidity_inputs,
    quote_spread_pct,
    sabit_hedefle_cikar,
)
from nihai.risk.risk_manager import RiskManager
from nihai.risk.safety import CircuitBreaker
from nihai.signals.quant import CANLI_GUNLUK_BAR
from nihai.signals.signal_engine import log_block, log_decision, log_signal, prune_logs
from nihai.signals.strategy import SignalAction, generate_signal, is_quant_label, reanchor_to_live

# NIHAI_DISABLE_FILE_LOG: dosya sink'i + retention'ı kapatır. pytest bunu conftest'te
# set eder — aksi halde main'i (dolaylı: research.duo) import eden HERHANGİ bir test,
# sink'i test sürecine ekler → tüm test logları ÜRETİM system.log'una akar ve
# prune_logs() gerçek log/CSV dosyalarını silebilir (2026-07-02'de canlıda görüldü).
if _mp.parent_process() is None and os.getenv("NIHAI_DISABLE_FILE_LOG", "").lower() not in (
        "1", "true", "yes", "on"):
    logger.add(LOG_DIR / "system.log", rotation="10 MB", retention="30 days")
    # Log retention: rotasyon arşivleri + günlük karar/tarama dosyaları sınırsız
    # birikmesin (system.log zaten retention'lı; diğerleri elle birikiyordu → 580MB).
    try:
        _pruned = prune_logs()
        if sum(_pruned.values()):
            logger.info(f"[log] retention: {_pruned} eski dosya silindi")
    except Exception as _e:  # best-effort; loglama hatası akışı durdurmasın
        logger.debug(f"[log] prune atlandı: {_e}")

EQUITY_CURVE_CSV = DATA_DIR / "equity_curve.csv"
_last_equity_point: float | None = None   # C6: değişmeyen dakikalık noktalar birikmesin


def append_equity_point(equity: float, when: datetime) -> None:
    """Madde 15/9 — canlı equity eğrisini diske ekle (backtest dışı kalıcı kayıt).

    C6 — equity yalnız pozisyon kapanınca (seans içinde) değişir; her dakika özdeş nokta
    yazmak dosyayı sınırsız şişiriyordu. Yalnız son yazılan noktadan FARKLIYSA ekle
    (0.005$ epsilon = ':.2f' yuvarlamasıyla aynı satır olan noktaları atla)."""
    global _last_equity_point
    if _last_equity_point is not None and abs(equity - _last_equity_point) < 0.005:
        return
    try:
        new = not EQUITY_CURVE_CSV.exists()
        with EQUITY_CURVE_CSV.open("a", encoding="utf-8") as f:
            if new:
                f.write("timestamp,equity\n")
            f.write(f"{when.isoformat(timespec='seconds')},{equity:.2f}\n")
        _last_equity_point = equity
    except Exception as e:
        logger.debug(f"[equity] yazma hatası: {e}")


# C2 — sinyal-alarm dedup: risk-reddi / dolu-portföyde actionable sinyal her ~60s Telegram
# spam'i üretiyordu (alarm başarılı açılıştan BAĞIMSIZ gönderilir; cooldown yalnız açılışta
# set edilir). (symbol,bar_ts) başına bir kez alarm ver; yeni bar → yeni alarm. Paralel
# katman/işçi taramaları için thread-safe.
_ALERT_DEDUP_LOCK = threading.Lock()
_alerted_bars: dict[str, str] = {}


def _uyari_bari(df_htf, df_daily) -> str:
    """Uyarı tekilleştirme anahtarı = sinyalin KARAR barı.

    Tur-106 (BULGU-106-11): quant kararı GÜNLÜK bara bağlıdır; anahtar HTF (1h) barı olunca aynı günlük "LONG"
    durumu her saat yeniden uyarılıp karar zincirine "kabul edilen sinyal" halkası olarak yazılıyordu (1 Eki:
    76 halka, MU/SNDK 9'ar)."""
    if df_daily is not None and not df_daily.empty:
        return str(df_daily.index[-1])
    return str(df_htf.index[-1])


def _alert_once(symbol: str, bar_ts: str) -> bool:
    """(symbol,bar_ts) için ilk çağrıda True (alarm ver), aynı barın tekrarında False."""
    with _ALERT_DEDUP_LOCK:
        if _alerted_bars.get(symbol) == bar_ts:
            return False
        _alerted_bars[symbol] = bar_ts
        return True


def _apply_close_bookkeeping(store: StateStore, ce) -> None:
    """Pozisyon kapanış defteri — run_loop._on_close ve flatten_all_action TEK kaynağı (C4):
    Madde 20 realized PnL + Madde 53 wash-sale + Madde 23 T+1 takas edilmemiş gelir +
    kill-switch tetiklenirse uyarı. flatten_all → _close bu hook'u her kapanışta çağırır."""
    store.record_trade_pnl(ce.realized_pnl)
    # Madde 53 — zararlı kapanışı wash-sale defterine yaz
    store.register_close_for_wash_sale(ce.pos.symbol, ce.realized_pnl, ce.exit_time,
                                       ce.pos.asset_type)
    # Madde 23 — satış geliri T+1 takas edilmemiş sayılır
    mult = 100 if ce.pos.asset_type == "option" else 1
    store.add_unsettled_proceeds(ce.exit_price * ce.pos.quantity * mult, ce.exit_time.date(),
                                ce.pos.side)
    # Tur-101 (BULGU-101-4): quant çıkışı damgalanır → `scan_symbol` aynı ET gününde yeniden girmez.
    # Yalnız quant: TA'nın soğutması GİRİŞTEN sayılır (Madde 7) ve değişmez.
    if is_quant_label(getattr(ce.pos, "strategy", None)):
        store.set_cooldown(ce.pos.symbol, ce.exit_time)
    if store.kill_switch_active:
        telegram_bot.notify_alert(
            "CRIT", "KILL-SWITCH AKTIF — rolling expectancy negatif. "
                    "Tum sinyal akisi durduruldu."
        )


# ---------------------------------------------------------------------------
# Madde 41 — başlangıç mutabakatı
# ---------------------------------------------------------------------------
def startup_reconciliation(store: StateStore) -> None:
    p = store.portfolio
    if p is None:
        raise RuntimeError("State yok")
    # Sanal portföyde "broker pozisyon listesi" yerine kendi state'imiz authoritative.
    # Yine de tutarsızlık tespiti yapalım: NLV vs equity, ATH > equity gibi.
    if p.equity > p.ath_equity:
        logger.info(f"[recon] yeni ATH tespit edildi: {p.equity:.2f} (eski {p.ath_equity:.2f})")
        p.ath_equity = p.equity
    if p.equity <= 0:
        raise RuntimeError("Equity sıfır veya negatif — manuel müdahale gerekli")
    # Madde 10 — yeni seans günüyse günlük sayaçları sıfırla (daily_pnl + consecutive_stops).
    # Aksi halde daily_pnl ömür-boyu birikir ve consecutive_stops cooldown'u kalıcı kilitlenir.
    today = now_et_naive().date()
    if p.roll_session_if_new_day(today):
        logger.info(f"[recon] yeni seans günü {today} — günlük sayaçlar sıfırlandı "
                    f"(daily_pnl, consecutive_stops)")
    # PDT 5-iş-günü penceresi — Tur-90: YÜRÜYEN son N NYSE seansı (eskiden takvim bloğu sıfırlanırdı)
    if CONFIG.regulation.pdt_enabled:
        p.pdt_refresh(today, CONFIG.regulation.pdt_lookback_days)
    # Madde 23 — takası dolmuş nakit dilimlerini düş
    store.settle_matured(today)
    _pdt_note = (f"day_trades={p.day_trades_count}/"
                 f"{CONFIG.regulation.pdt_max_day_trades_under_threshold}"
                 if CONFIG.regulation.pdt_enabled else "PDT takibi: kapali")
    logger.info(f"[recon] OK | equity={p.equity:.2f} | açık poz={len(p.open_positions)} | "
                f"{_pdt_note}")


# ---------------------------------------------------------------------------
# Sinyal döngüsü
# ---------------------------------------------------------------------------
# BULGU-59-6 — benchmark kapanış serisi önbelleği (beta ölçümü için). Sağlayıcı çağrısı
# TTL boyunca bir kez ödenir; hata/veri yoksa None ve beta ÖLÇÜLMEZ (uydurma 1.0 yok).
_BENCH_CACHE: dict[str, object] = {"ts": None, "closes": None}
_BENCH_TTL_SEC = float(os.getenv("BETA_BENCH_TTL_SEC", "1800"))


def _bench_closes(provider):
    """`CONFIG.regime.benchmark` kapanışları (HTF_INTERVAL) — TTL önbellekli, hata→None."""
    now = time_mod.time()
    ts = _BENCH_CACHE["ts"]
    if ts is not None and (now - float(ts)) < _BENCH_TTL_SEC:
        return _BENCH_CACHE["closes"]
    closes = None
    try:
        b = provider.get_bars(CONFIG.regime.benchmark, HTF_INTERVAL, lookback_bars=200)
        if b is not None and not b.empty and "close" in b.columns:
            closes = b["close"]
    except Exception as e:
        logger.debug(f"[beta] benchmark barları alınamadı ({e}) — beta ölçülmeyecek")
    _BENCH_CACHE["ts"], _BENCH_CACHE["closes"] = now, closes
    return closes


def _position_beta(df, provider) -> float | None:
    """Pozisyonun benchmark betası — ölçülemezse None (bkz. beta_vs_benchmark)."""
    bench = _bench_closes(provider)
    if bench is None or df is None or df.empty or "close" not in df.columns:
        return None
    try:
        from nihai.signals.market_regime import beta_vs_benchmark
        return beta_vs_benchmark(df["close"], bench)
    except Exception:
        return None


# Tur-82 — sembol başına son zincir/IV değerlendirmesinin monotonik zamanı (IV örnekleme kısıtı).
_IV_LAST: dict[str, float] = {}
IV_COLLECT_SEC = 60.0 * float(os.getenv("IV_COLLECT_MIN", "15"))


def _iv_collect_due(symbol: str) -> bool:
    last = _IV_LAST.get(symbol)
    return last is None or (time_mod.monotonic() - last) >= IV_COLLECT_SEC


def _chain_health_and_iv(symbol: str, provider) -> tuple[bool, float | None]:
    """Madde 21 zincir sağlığı + Madde 16 canlı IV Rank → (chain_health_ok, iv_rank_val).

    Gövde Tur-82'de `scan_symbol`'den BİREBİR taşındı; yalnız ne zaman çağrıldığı değişti."""
    _IV_LAST[symbol] = time_mod.monotonic()
    chain_health_ok = False
    iv_rank_val = None  # Madde 16 — aşağıda zincirden ATM IV ile hesaplanır
    chain = provider.get_option_chain(symbol)
    if chain and chain.get("calls") is not None and not chain["calls"].empty:
        res = filter_liquid_chain(
            chain["calls"], min_oi=CONFIG.options.min_open_interest,
            min_volume=CONFIG.options.min_daily_volume,
            max_spread_pct=CONFIG.options.max_spread_pct_of_mid,
        )
        chain_health_ok = not res.accepted.empty
        log_decision("CHAIN_HEALTH",
                     f"{symbol}: kabul={len(res.accepted)} red={res.rejected_count} "
                     f"sebepler={res.reasons}")

        # Madde 16 — Canlı IV Rank: ~30-45 DTE bandındaki bir vadeden ATM IV tahmin et.
        # (En yakın vade 1-2 DTE olabilir; kısa-vade IV'si 52w seriyi bozar.)
        try:
            underlying_px = float(chain.get("underlying_price") or 0)
            target_dte = (CONFIG.option_strategy.dte_min + CONFIG.option_strategy.dte_max) // 2
            expiries = [e for e in (chain.get("all_expiries") or [chain.get("expiry")]) if e]
            best_exp = min(expiries, key=lambda e: abs(days_to_expiry(e) - target_dte),
                           default=chain.get("expiry"))
            iv_chain = chain if best_exp == chain.get("expiry") else provider.get_option_chain(symbol, best_exp)
            calls = iv_chain.get("calls") if iv_chain else None
            atm_iv = (estimate_atm_iv(calls, underlying_px, days_to_expiry(best_exp))
                      if calls is not None and not calls.empty and best_exp else None)
            if atm_iv:
                append_iv_today(symbol, atm_iv)
                ivr = get_iv_rank(symbol, atm_iv)
                # ÖLÇÜLMEMİŞSE skorlamaya SOKMA: NO_HISTORY/NO_RANGE/LOW_CONFIDENCE
                # hepsi 50.0 döner ve bu değer bonus bandının ([30,50]) içindedir.
                iv_rank_val = ivr.iv_rank if ivr.measured else None
                log_decision("IV_RANK",
                             f"{symbol}: ATM_IV={atm_iv*100:.0f}% IVR={ivr.iv_rank:.0f} "
                             f"({ivr.confidence}) @DTE{days_to_expiry(best_exp)}")
        except Exception as e:
            logger.debug(f"[scan] {symbol} IV rank hesap hatası: {e}")
    return chain_health_ok, iv_rank_val


# Tur-97 — quant_trend GÜNLÜK barla karar verir. Oluşmakta olan gün barı sağlayıcıda dışlandığı
# için (Madde 3) günlük seri seans boyunca DEĞİŞMEZ → sembol başına ET günü başına BİR çekim.
# Geniş katmanda yüzlerce sembol dakikada bir taranır; önbelleksiz her tarama hub'a ~300 barlık
# istek atardı. Boş yanıt önbelleğe YAZILMAZ (geçici hata tüm günü kilitlemesin).
_QUANT_GUNLUK_BAR = CANLI_GUNLUK_BAR
_GUNLUK_ONBELLEK: dict[str, tuple] = {}
_GUNLUK_KILIT = threading.Lock()


def _gunluk_barlar(symbol: str, provider):
    gun = now_et_naive().date()
    with _GUNLUK_KILIT:
        kayit = _GUNLUK_ONBELLEK.get(symbol)
    if kayit is not None and kayit[0] == gun:
        return kayit[1]
    # Tur-106 (BULGU-106-17): yalnız KAPANMIŞ günler — önbellek günün ilk çekimini tutar; oluşmakta olan bugün barı
    # kararı başlatma saatindeki fiyata dondururdu (ölçülen kural T−1 kapanışında karar verir).
    df = kapanmis_gunluk(provider.get_bars(symbol, "1d", lookback_bars=_QUANT_GUNLUK_BAR))
    if df is not None and not df.empty:
        with _GUNLUK_KILIT:
            _GUNLUK_ONBELLEK[symbol] = (gun, df)
    return df


# Tur-101 (SADAKAT, BULGU-101-1) — quant evreni ODAK katmanıdır: quant_trend yalnız odak listesinde ölçüldü
# (Tur-97 ABD_izleme). Geniş katman oto-tarayıcı çıktısıdır (4 saatte bir değişir); quant kipinde tavanlar
# (net yön 2×, sektör 3) ~2× nominalde dolarken portföyü HANGİ sembole önce ulaşıldıysa o doldururdu — replay
# 2026-09-25 (odak 30 + geniş 30): açılan 9 pozisyonun 6'sı genişten, alfabetik tarama sırasıyla.
# Odak kümesi watchlist.json'dan okunur (dosya kilidi); geniş katman dakikada yüzlerce kez sorduğu için 60 sn önbellek.
_QUANT_ODAK: dict = {"t": float("-inf"), "k": frozenset()}
_QUANT_ODAK_LOG: dict = {"gun": None, "s": set()}     # odak-dışı ret sembol başına günde BİR satır
# Tur-106 (BULGU-106-34): quant kararı GÜNLÜKtür (kapanmış günlük barlar); odak katmanı ise aynı sembolü ~6 sn'de bir
# yeniden değerlendirip aynı sonucu yazıyordu. Ölçüldü (2 Eki, 5,6 saat): 33.850 "SKIP … RISK_REJECT" (10 sembol;
# her biri kotasyon + gölge köprü + AI_ADVICE + SHADOW satırı) ve 40.620 QUANT_POZISYON_VAR → 188 bin satırlık
# günlüğün ~%75'i. Sembol başına en çok bu aralıkta bir değerlendirilir; tek-katman döngüsü (60 sn) etkilenmez.
_QUANT_TARAMA_SN = 45.0


def _quant_gunde_bir(store, kod: str, symbol: str) -> bool:
    """Bu (kod, sembol) muhafız satırı bu ET günü İLK kez mi yazılıyor? Depo iz taşımıyorsa (test ikizi) hep True."""
    iz = getattr(store, "quant_gun_satiri", None)
    if iz is None:
        return True
    gun = now_et_naive().date()
    if iz["gun"] != gun:
        iz.update(gun=gun, s=set())
    if (kod, symbol) in iz["s"]:
        return False
    iz["s"].add((kod, symbol))
    return True


def _quant_odak() -> frozenset:
    simdi = time_mod.monotonic()
    if simdi - _QUANT_ODAK["t"] > 60:
        _QUANT_ODAK.update(t=simdi, k=frozenset(watchlist.partition(FOCUSED_TOP_N)["focused"]))
    return _QUANT_ODAK["k"]


def scan_symbol(symbol: str, provider, risk_mgr, state_store, breaker,
                pmgr: PositionManager, dashboard=None, lock=None,
                strategy_mode: str | None = None) -> None:
    """strategy_mode=None → global ACTIVE_STRATEGY (mevcut davranış). duo.py iki
    varyasyonu farklı modlarla (event_breakout / trend_pullback) aynı tarama
    yolundan geçirmek için bunu açıkça geçer."""
    # İki-katman paralel modda: sinyal-kabul→pozisyon-açma kritik bölümü ortak
    # portföy/state'i tek kilitle serileştirir. lock=None ise (tek-thread mod)
    # nullcontext → davranış değişmez.
    if strategy_mode is None:
        from nihai.runtime_strategy import apply
        strategy_mode = apply()
    _crit = lock if lock is not None else contextlib.nullcontext()
    try:
        # Kill-switch kontrolü
        if state_store.kill_switch_active:
            log_block("KILL_SWITCH_ACTIVE", f"{symbol}: yeni girişler engellendi (Kill-switch aktif)")
            return

        # Madde 22 — yalnızca RTH
        can, reason = can_enter_new_position()
        if not can:
            log_block("SESSION_GUARD", f"{symbol}: {reason}")
            return

        # Tur-97 (BULGU-97-8): quant_trend bir DURUM sinyalidir — trend sürdükçe her taramada yeniden ateşler.
        # Sembol başına küme sınırı (2) TA olay sinyali için kuruldu; quant'ta ikinci pozisyon aynı bahsi
        # ikiye katlar ve backtest'in ölçtüğü şey (sembol başına tek pozisyon) değildir. Ölçüldü (replay
        # 2026-09-25, AAPL+MSFT): 3 pozisyon açıldı, sonra SECTOR_CAP_3_TECH.
        if strategy_mode == "quant_trend" and any(p.symbol == symbol for p in list(pmgr.portfolio.open_positions)):
            if _quant_gunde_bir(state_store, "QUANT_POZISYON_VAR", symbol):
                log_block("QUANT_POZISYON_VAR", f"{symbol}: açık quant pozisyonu var (sembol başına tek)")
            return
        if strategy_mode == "quant_trend":
            gun = now_et_naive().date()
            if symbol not in _quant_odak():                          # Tur-101 SADAKAT (yukarıda)
                if _QUANT_ODAK_LOG["gun"] != gun:
                    _QUANT_ODAK_LOG.update(gun=gun, s=set())
                if symbol not in _QUANT_ODAK_LOG["s"]:
                    _QUANT_ODAK_LOG["s"].add(symbol)
                    log_block("QUANT_ODAK_DISI", f"{symbol}: quant yalnız odak katmanında (ölçülen evren)")
                return
            # Tur-101 (BULGU-101-4): backtest çıkış barında yeniden girmez, karar o günün KAPANIŞINDA verilir
            # ve giriş ertesi açılıştadır. Canlıda soğutma GİRİŞTEN sayılıyordu (2 sa) → gün içi stoptan
            # ~1 dk sonra, skor hâlâ > 0 olduğu için aynı gün yeniden giriliyordu (ölçülmemiş davranış).
            son = state_store.symbol_cooldowns.get(symbol)
            if son is not None and son.date() == gun:
                if _quant_gunde_bir(state_store, "QUANT_AYNI_GUN", symbol):
                    log_block("QUANT_AYNI_GUN", f"{symbol}: bugün quant girişi/çıkışı var — yeniden giriş ertesi seans")
                return
            # BULGU-106-34: aynı günlük kararı saniyeler içinde yeniden üretme (ET saatiyle: replay simüle saati kullanır).
            son_tarama = getattr(state_store, "quant_son_tarama", None)
            if son_tarama is not None:
                simdi = now_et_naive()
                onceki = son_tarama.get(symbol)
                if onceki is not None and 0 <= (simdi - onceki).total_seconds() < _QUANT_TARAMA_SN:
                    return
                son_tarama[symbol] = simdi

        # Madde 7 — re-entry cooldown
        if state_store.cooldown_active(symbol, now_et_naive()):
            log_block("COOLDOWN", f"{symbol}: re-entry bekleniyor")
            return

        # Madde 26 — Olay/bilanço guard
        evt_block, evt_why = event_guard(symbol)
        if evt_block:
            log_block("EVENT_GUARD", f"{symbol}: {evt_why}")
            return

        # Madde 25 — Kurumsal aksiyon (yakın split) guard
        ca_block, ca_why = corporate_action_guard(symbol)
        if ca_block:
            log_block("CORP_ACTION", f"{symbol}: {ca_why}")
            return

        # Veri çek (Madde 34 — sanity check provider içinde)
        df_htf = provider.get_bars(symbol, HTF_INTERVAL, lookback_bars=200)
        if df_htf.empty:
            log_block("DATA_EMPTY", f"{symbol}: {HTF_INTERVAL} bar yok")
            return

        # Madde 24 — Halt / LULD sezgisel guard
        halt_block, halt_why = halt_or_luld_guard(df_htf)
        if halt_block:
            log_block("HALT_LULD", f"{symbol}: {halt_why}")
            return

        # Tur-82 (BULGU-82-3): kotasyon ve opsiyon zinciri artık yalnız GEREKTİKLERİNDE çekilir.
        # ÖLÇÜLDÜ (2026-09-17 seansı, karar logu): zincir 6524 taramada İKİ kez çekildi (sembol
        # başına ~6 Yahoo isteği) ve 6209 WAIT'in HİÇBİRİ zincirin girdiği tek yere — kalite
        # adımına — ulaşmadı (6138 NO_EVENT · 71 RSI_OVEREXTENDED). `chain_eval` zinciri yalnız o
        # adımda ister → kararlar ve SKIP satırları BİREBİR aynı; CHAIN_HEALTH/IV_RANK bilgi
        # satırları (huni saymaz) yalnız zincir çekilen taramada yazılır. Kotasyon yalnız eyleme
        # dönük sinyalde çekilir: WAIT'te kullanan yok (QUOTE_UNHEALTHY son 20 günün karar
        # loglarında 0 kez görüldü; sağlıksız kotasyon eyleme dönük sinyali yine durdurur).
        _chain_memo: list[tuple[bool, float | None]] = []

        def _chain_eval() -> tuple[bool, float | None]:
            if not _chain_memo:
                _chain_memo.append(_chain_health_and_iv(symbol, provider))
            return _chain_memo[0]

        df_daily = (_gunluk_barlar(symbol, provider)
                    if strategy_mode == "quant_trend" and HTF_INTERVAL != "1d" else None)
        sig = generate_signal(symbol, df_htf, chain_eval=_chain_eval,
                              strategy_mode=strategy_mode, live_interval=HTF_INTERVAL,
                              df_daily=df_daily)

        if not sig.is_actionable():
            # IV geçmişi günde BİR satırdır ve günün SON gözlemini tutar (`append_iv_today`).
            # Eskiden her taramada yazılıyordu; şimdi en fazla IV_COLLECT_MIN'de bir. ÖLÇÜLDÜ
            # (17 Eylül, 510 sembol): 15 dk'lık aralık günün son değerini 13/510 sembolde
            # değiştirir (ort. 0,03 puan) — 30 dk'da 161/510.
            if not _chain_memo and _iv_collect_due(symbol):
                _chain_eval()
            log_signal(sig, accepted=False)
            return
        signal_price = sig.price  # canlı fiyat çapası gecikme referansını silmesin

        # Quote sağlığı (Madde 34)
        q = provider.get_quote(symbol)
        if q:
            healthy, why = provider.is_quote_healthy(q)
            if not healthy:
                log_block("QUOTE_UNHEALTHY", f"{symbol}: {why}")
                return
        live_px = float(q.last) if (q and LIVE_ENTRY_ANCHOR) else 0.0

        # ===== CANLI GİRİŞ ÇAPASI (Tur-39) — sinyal fiyatı BAYAT olabilir (HTF barının
        # kapanışı), işlem ise canlı fiyattan olur. Girişi gerçek fiyata taşı, stop/target'ı
        # YAPISAL bırak, RR'ı yeniden hesapla; kurulum artık geçerli değilse GİRME.
        # Bu yeni bir filtre değil, mevcut min_rr_ratio kapısının işlem görülebilir fiyatta
        # uygulanmasıdır. `LIVE_ENTRY_ANCHOR=off` eski davranışa döndürür (A/B için).
        if live_px > 0:
            sig_anchored, stale_why = reanchor_to_live(
                sig, live_px, min_rr=CONFIG.strategy.min_rr_ratio)
            if sig_anchored is None:
                log_signal(sig, accepted=False, extra="STALE_SIGNAL")
                log_block("STALE_SIGNAL", f"{symbol}: {stale_why}")
                return
            sig = sig_anchored

        # ===== DOLUM FİYATINDA RR (Tur-89, BULGU-89-1). Çapa RR'ı son İŞLEM fiyatında ölçer; sanal
        # dolum ise kayma kadar aleyhtedir. Ölçüldü: 35 giriş HEDEFİN ÖTESİNDE doldu (PPG sinyal
        # 106,49 → dolum 112,27, hedef 109,67) ve risk planın 20 katına çıktı. Aynı min_rr kapısı
        # DOLUM fiyatında uygulanır; boyut da gerçek riske (dolum − stop) göre kurulur.
        fill_px = sig.price
        if OPTION_MODE != "option" and sig.action in (SignalAction.LONG, SignalAction.SHORT):
            fill_px = pmgr.slip.apply_entry(sig.price, sig.action.value, "stock",
                                            spread_pct=quote_spread_pct(q, sig.price) if q else None)
            _, fill_why = reanchor_to_live(sig, fill_px, min_rr=CONFIG.strategy.min_rr_ratio)
            if fill_why:
                log_signal(sig, accepted=False, extra="FILL_SLIPPAGE")
                log_block("FILL_SLIPPAGE", f"{symbol}: dolum {fill_px:.4f} — {fill_why}")
                return

        # ===== GÖLGE KÖPRÜSÜ (Tur-60) — `SHADOW_BRIDGE` ile kipli, VARSAYILAN "advise".
        # Gözlemevi (mikroyapı + NEDENSEL rejim HMM + veri sağlığı) ve kovalama sapması
        # bugüne kadar YALNIZ live_predict_service'te, yani AYRI bir süreçte koşuyordu;
        # aday sinyali ÜRETEN yol onları hiç görmüyordu. Artık görüyor.
        # advise = ölçüler gerekçeye + karar loguna yazılır, KARAR DEĞİŞMEZ (veto False /
        # size_mult 1.0 yapısal). enforce = yalnız TERFİ ETMİŞ kural hüküm verir ve
        # `shadow_bridge.RULES` bugün hiç PROMOTE taşımaz → o kipte de sinyal bastırılmaz.
        # Çapadan SONRA duruyor: `chase_drift_bps` canlı fiyatı ister ve ölçtüğü şey tam
        # olarak çapanın gördüğü sapmadır.
        sh_v = shadow_bridge.evaluate(symbol, df_htf, sig, live_price=live_px or None)
        if sh_v is not None:
            sig.reasons.append(sh_v.reason)
            sig.metadata["shadow"] = sh_v.marks
            log_decision("SHADOW", f"{symbol}: {sh_v.reason}")
            if sh_v.veto:
                log_signal(sig, accepted=False, extra="SHADOW_VETO")
                log_block("SHADOW_VETO", f"{symbol}: {sh_v.reason}")
                return

        # Madde 30 — SSR farkındalığı (yalnızca SHORT sinyalleri; bloke etmez, işaretler)
        if sig.action == SignalAction.SHORT:
            try:
                df_d = provider.get_bars(symbol, "1d", lookback_bars=3)
                if df_d is not None and len(df_d) >= 2:
                    prev_close = float(df_d["close"].iloc[-2])
                    cur_px = float(df_d["close"].iloc[-1])
                    ssr_on, ssr_why = ssr_short_guard(prev_close, cur_px)
                    if ssr_on:
                        sig.reasons.append(ssr_why)
                        log_block("SSR_NOTICE", f"{symbol}: {ssr_why}")
            except Exception as e:
                logger.debug(f"[scan] {symbol} SSR kontrol hatası: {e}")

            # Madde 30'un EKSİK YARISI (Tur-57): SSR kodluydu ama hissenin O GÜN açığa
            # satılabilir OLUP OLMADIĞI hiç sorulmuyordu. Bayraklar Alpaca'nın ÜCRETSİZ
            # trading API'sinden gelir. SSR ile AYNI sözleşme: varsayılan İŞARETLER,
            # bloke ETMEZ (bayrak brokerine özgüdür; başka bir brokerde satılabilir
            # olabilir). `SHORT_BORROW_GUARD=block` bilinçli olarak kapıya çevirir.
            # Bayrak okunamazsa (None) hiçbir şey yapılmaz — "bilinmiyor" bir RED
            # gerekçesi değildir (Tur-31: ölçüm yokluğu ölçüm gibi kullanılamaz).
            try:
                from nihai.data.data_provider import asset_flags as _asset_flags
                flags = _asset_flags(symbol, provider)
                if flags is not None and not (flags.get("shortable") and
                                              flags.get("easy_to_borrow")):
                    why = ("NOT_SHORTABLE" if not flags.get("shortable")
                           else "HARD_TO_BORROW")
                    sig.reasons.append(why)
                    log_block("BORROW_NOTICE", f"{symbol}: {why}")
                    if (os.getenv("SHORT_BORROW_GUARD") or "").strip().lower() == "block":
                        log_signal(sig, accepted=False, extra=why)
                        return
            except Exception as e:
                logger.debug(f"[scan] {symbol} borç bayrağı hatası: {e}")

        # Madde 53 — Wash-sale farkındalığı (vergi muhasebesi; bloke etmez)
        # Yeni girişin ARAÇ türü: `OPTION_MODE` "both" iken iki dal da açıktır ve tek bir
        # tür seçmek yanlış olurdu → "" geçilir ve çapraz etiketi ARANMAZ (ölçüm yokluğu
        # uydurma bir sınıflandırmaya çevrilmez).
        _ws_kind = {"stock": "stock", "option": "option"}.get(OPTION_MODE, "")
        ws_hit, ws_why = state_store.wash_sale_check(symbol, now_et_naive(), _ws_kind)
        if ws_hit:
            sig.reasons.append(ws_why)
            log_decision("WASH_SALE", f"{symbol}: {ws_why}")

        # ===== İÇ AI KARAR KÖPRÜSÜ (Tur-38) — `AI_LIVE_MODE` ile kipli.
        # off (VARSAYILAN) → advise() None döner, aşağıdaki tek satır bile çalışmaz.
        # advise → görüş yalnızca gerekçeye/loga (ve oradan Telegram metnine) yazılır.
        # enforce → ABSTAIN/ters-yön VETOSU hem uyarıyı hem girişi durdurur; veto yoksa
        # posterior aşağıda boyutu KISABİLİR. Determinist guard'ların HEPSİNDEN SONRA
        # duruyor: AI, kuralların yerine değil ARDINA konur.
        ai_adv = ai_advisor.advise(symbol, df_htf, sig, strategy=strategy_mode)
        if ai_adv is not None:
            sig.reasons.append(ai_adv.reason)
            log_decision("AI_ADVICE", f"{symbol}: {ai_adv.reason}")
            if ai_adv.veto:
                log_signal(sig, accepted=False, extra="AI_VETO")
                log_block("AI_VETO", f"{symbol}: {ai_adv.reason}")
                return

        # ===== SİNYAL-ONLY UYARI — risk onayından BAĞIMSIZ (config.SIGNAL_ALERT_ALWAYS).
        # Kullanıcı emir değil sinyal istiyor: geçerli sinyal (strateji RR+kalite filtresini
        # geçen; gates/RR-altı/kalite-altı zaten WAIT veya filtreli) üretilir üretilmez
        # "hisse / yön / hedef / stop" bildirilir. Risk kapıları (soğutma/caps/buying-power/
        # PDT) yalnızca aşağıdaki SANAL pozisyon açımını etkiler, bu uyarıyı SUSTURMAZ.
        signal_alerted = False
        hedef_nominal = not sabit_hedefle_cikar((sig.metadata or {}).get("strategy"))   # BULGU-106-28
        valid_entry = (sig.rr_ratio >= CONFIG.strategy.min_rr_ratio
                       and sig.quality_score >= CONFIG.strategy.min_signal_quality_score)
        # C2 — (symbol,bar_ts) başına bir kez alarm ver: risk-reddi/dolu-portföy actionable
        # sinyalini her ~60s yeniden alarmlamasın (Telegram spam). Aynı barda başarılı açılış
        # olursa signal_alerted=True → açılış-yolu ikinci mesaj göndermez (çift mesaj önle).
        if SIGNAL_ALERT_ALWAYS and valid_entry and _alert_once(symbol, _uyari_bari(df_htf, df_daily)):
            telegram_bot.notify_signal(symbol, sig.action.value, sig.price, sig.rr_ratio,
                                       sig.quality_score, " | ".join(sig.reasons),
                                       stop=sig.stop, target=sig.target, hedef_nominal=hedef_nominal)
            log_signal(sig, accepted=True, extra="SIGNAL_ALERT (risk-bağımsız)")
            if dashboard:
                dashboard.add_signal_line(
                    f"{symbol} {sig.action.value} @ {sig.price:.2f} | "
                    f"hedef{'(nominal)' if hedef_nominal else ''} {sig.target:.2f} stop {sig.stop:.2f} | "
                    f"RR={sig.rr_ratio:.2f} Q={sig.quality_score:.2f}")
            signal_alerted = True

        # ===== KRİTİK BÖLÜM (kilitli) — sinyal kabulü + pozisyon açma + state yazımı.
        # Paralel katmanlar arasında "max pozisyon" kontrolü ile açmanın atomik olması
        # için risk-sizing de kilit içinde (güncel portföye göre yeniden değerlendirilir).
        with _crit:
            # Devre kesici (Madde 42)
            can_emit, why = breaker.can_emit()
            if not can_emit:
                log_block("CIRCUIT_BREAKER", why)
                return

            # A5 — düşük-güven kademeli boyutlama: watchlist meta'sındaki risk_mult'i uygula
            # (CONFIDENCE_SIZING kapalıyken 1.0 → davranış değişmez).
            conf_mult = watchlist.risk_mult(symbol) if CONFIDENCE_SIZING_ENABLED else 1.0

            # Olasılık-bazlı boyutlama (opt-in, varsayılan kapalı): meta_gate P(win)'i
            # risk çarpanına çevirip mevcut conf_mult ile ÇARP (istiflenir). Model pasif
            # veya hata → None → prob_to_size_mult cap (1.0) döndürür: nötr, bozmaz.
            if PROB_SIZING_ENABLED:
                from nihai.risk.prob_sizing import prob_to_size_mult
                from nihai.signals.meta_gate import get_gate
                pwin = get_gate().probability(df_htf, sig)
                conf_mult *= prob_to_size_mult(
                    pwin, p_lo=PROB_SIZING_P_LO, p_hi=PROB_SIZING_P_HI,
                    floor=PROB_SIZING_FLOOR)

            # İç AI boyut çarpanı (yalnız AI_LIVE_MODE=enforce; advise/off → 1.0).
            # prob_sizing ile AYNI sözleşme: tavan 1.0 → pozisyonu asla BÜYÜTMEZ.
            if ai_adv is not None:
                conf_mult *= ai_adv.size_mult

            # Gölge köprüsü boyut çarpanı (yalnız enforce + TERFİ ETMİŞ kural; bugün 1.0).
            # prob_sizing/ai_advisor ile AYNI sözleşme: pozisyonu asla BÜYÜTMEZ.
            if sh_v is not None:
                conf_mult *= sh_v.size_mult

            # Risk modülü (Madde 2, 13, 14, 27) — sektör Madde 12, sembol/yön Madde 14 için
            rc = risk_mgr.size_stock(entry=fill_px, stop=sig.stop, sector=get_sector(symbol),
                                     symbol=symbol, side=sig.action.value,
                                     confidence_multiplier=conf_mult)
            if not rc.approved:
                log_signal(sig, accepted=False, extra=f"RISK_REJECT: {rc.reason}")
                return

            # Madde 23 — good-faith farkındalığı (takas edilmemiş nakitle alım; bloke etmez)
            gf, gf_why = state_store.good_faith_warning(rc.sizing.get("notional", 0.0))
            if gf:
                sig.reasons.append(gf_why)
                log_decision("GOOD_FAITH", f"{symbol}: {gf_why}")

            # Sinyali kabul et — sanal hisse pozisyonu aç
            if OPTION_MODE in ("stock", "both"):
                log_signal(sig, accepted=True, extra=f"sizing={rc.sizing}")
                st_name = f"{strategy_mode} V1" if strategy_mode else f"{ACTIVE_STRATEGY} V1"
                pmgr.open_from_signal(sig, rc.sizing, now=now_et_naive(),
                                      asset_type="stock", sector=get_sector(symbol),
                                      strategy_name=st_name,
                                      signal_price=signal_price,
                                      spread_pct=quote_spread_pct(q, sig.price),
                                      **liquidity_inputs(df_htf),
                                      beta=_position_beta(df_htf, provider))
                # Uyarı zaten risk-bağımsız gönderildiyse tekrar gönderme (çift mesaj önle)
                if not signal_alerted:
                    telegram_bot.notify_signal(symbol, sig.action.value, sig.price, sig.rr_ratio,
                                               sig.quality_score, " | ".join(sig.reasons),
                                               stop=sig.stop, target=sig.target, hedef_nominal=hedef_nominal)
                breaker.record_emit()
                if dashboard:
                    # Karar-desteği bağlamı (sinyal-only): rejim + düşük-güven yalnızca
                    # ilgili olduklarında gösterilir (gürültü yapmasın).
                    ctx = ""
                    if CONFIG.regime.enable_regime_risk and risk_mgr.regime_label not in ("—", "BULLISH"):
                        ctx += f" | rejim={risk_mgr.regime_label}×{risk_mgr.regime_multiplier:.2f}"
                    if CONFIDENCE_SIZING_ENABLED and conf_mult < 1.0:
                        ctx += f" | güven↓×{conf_mult:.2f}"
                    dashboard.add_signal_line(
                        f"{symbol} {sig.action.value} @ {sig.price:.2f} | "
                        f"RR={sig.rr_ratio:.2f} Q={sig.quality_score:.2f} | "
                        f"adet={rc.sizing.get('shares','?')}{ctx}")

            # Madde 2, 16, 21, 27, 28 — opsiyon sinyali
            if OPTION_MODE in ("option", "both"):
                pick, reasons = select_option_contract(symbol, sig.action, provider)
                if pick is None:
                    log_block("OPTION_NO_CONTRACT", f"{symbol}: {' | '.join(reasons)}")
                else:
                    opt_rc = risk_mgr.size_long_option(pick.mid, pick.greeks,
                                                       sector=get_sector(symbol),
                                                       symbol=symbol, option_type=pick.option_type,
                                                       confidence_multiplier=conf_mult)
                    if not opt_rc.approved:
                        log_block("OPTION_RISK_REJECT",
                                  f"{symbol} {pick.label}: {opt_rc.reason}")
                    else:
                        # Sinyali opsiyon premiumuna göre yeniden ayarla (entry/stop/target prim-bazlı)
                        from dataclasses import replace as dc_replace
                        opt_sig = dc_replace(
                            sig,
                            price=pick.mid,
                            stop=pick.mid * (1 - CONFIG.option_strategy.premium_stop_loss_pct),
                            target=pick.mid * (1 + CONFIG.option_strategy.premium_target_gain_pct),
                            reasons=sig.reasons + pick.reasons,
                        )
                        log_signal(opt_sig, accepted=True,
                                   extra=f"OPT {pick.label} | sizing={opt_rc.sizing}")
                        st_name = f"{strategy_mode} V1 (option)" if strategy_mode else f"{ACTIVE_STRATEGY} V1 (option)"
                        pmgr.open_from_signal(
                            opt_sig, opt_rc.sizing, now=now_et_naive(),
                            asset_type="option", sector=get_sector(symbol),
                            option_type=pick.option_type, strike=pick.strike, expiry=pick.expiry,
                            greeks=pick.greeks, iv=pick.iv,
                            strategy_name=st_name,
                        )
                        telegram_bot.notify_signal(
                            symbol, f"{sig.action.value}_OPTION", pick.mid,
                            sig.rr_ratio, sig.quality_score,
                            f"{pick.label} | " + " | ".join(pick.reasons),
                        )
                        if dashboard:
                            dashboard.add_signal_line(
                                f"OPT {pick.label} @ {pick.mid:.2f} | "
                                f"Δ{pick.greeks.delta:+.2f} IV%{pick.iv*100:.0f} "
                                f"DTE={pick.dte} OI={pick.open_interest}"
                            )
                        breaker.record_emit()

            state_store.set_cooldown(symbol, now_et_naive())

    except Exception as e:
        logger.exception(f"[scan] {symbol} hata: {e}")
        telegram_bot.notify_alert("ERROR", f"{symbol} tarama hatası: {e}")


def flatten_all_action():
    """Madde 47 — Tüm açık pozisyonları kapat ve çık (acil müdahale)."""
    logger.warning("[main] FLATTEN-ALL talebi alındı")
    store = StateStore()
    portfolio = store.load_or_init(INITIAL_EQUITY)
    startup_reconciliation(store)
    provider = get_provider(DATA_PROVIDER)
    pmgr = PositionManager(portfolio, SlippageModel())
    # C4 — Madde 41: restart-taşınan açık pozisyonlar için tracker'ları geri yükle (aksi
    # halde force_close MAE/MFE/bars_held'i kaybeder ve post-mortem eksik olur).
    pmgr.rehydrate_trackers()

    if not portfolio.open_positions:
        logger.info("[main] kapatılacak açık pozisyon yok")
        return

    # C4 — kapanışlar run_loop._on_close ile AYNI defter yolundan geçsin: realized PnL +
    # wash-sale + T+1 takas edilmemiş gelir (eksikti) + kill-switch uyarısı. flatten_all →
    # _close bu hook'u her kapanışta tetikler; inline defter tutma KALDIRILDI (çift-sayma önle).
    pmgr.add_on_close(lambda ce: _apply_close_bookkeeping(store, ce))
    closed = pmgr.flatten_all(provider, now_et_naive(), reason="MANUAL_FLATTEN")
    for ce in closed:
        logger.warning(f"[main] kapatildi: {ce.pos.symbol} @ {ce.exit_price:.2f} "
                       f"PnL {ce.realized_pnl:+.2f}")
        telegram_bot.notify_alert(
            "WARN", f"FLATTEN: {ce.pos.symbol} @ {ce.exit_price:.2f} | "
                    f"PnL {ce.realized_pnl:+.2f}"
        )
    store.save()
    logger.info(f"[main] {len(closed)} pozisyon kapatildi. NLV={portfolio.equity:.2f}")


def run_loop(once: bool = False, screen_now: bool = False):
    logger.info(f"[main] Nihai Algoritma başlatılıyor | mod={RUN_MODE} | "
                f"watchlist={watchlist.current()} (kaynak: dinamik)")

    store = StateStore()
    portfolio = store.load_or_init(INITIAL_EQUITY)
    startup_reconciliation(store)

    provider = get_provider(DATA_PROVIDER)
    # Akış sağlığı `/metrics`e buradan ulaşır; observability KENDİ sağlayıcısını kurmaz
    # (kurarsa ikinci bir WebSocket bağlantısı açardı — hub'ın tek-upstream amacının tersi).
    OBS.set_provider(provider)
    logger.info(f"[main] data provider: {provider.name}")
    risk_mgr = RiskManager(portfolio)
    pmgr = PositionManager(portfolio, SlippageModel())
    # Madde 41 — restart'ta taşınan açık pozisyonlar için tracker'ları geri yükle
    rehydrated = pmgr.rehydrate_trackers()
    if rehydrated:
        logger.info(f"[recon] {rehydrated} açık pozisyon için tracker yeniden oluşturuldu")
    breaker = CircuitBreaker()

    # Madde 20 — pozisyon kapanışında kill-switch ve trade kaydı otomatik (C4: ortak defter)
    pmgr.add_on_close(lambda ce: _apply_close_bookkeeping(store, ce))
    # Tur-106 (BULGU-106-30): pano operatör eylemleri (kapat / flatten / stop-hedef / kill-switch) bu CANLI nesnelere
    # uygulanır. Eskiden diskteki ayrı bir kopyaya yazıyor, döngü bellekteki portföyü ≤60 sn'de geri yazıyordu.
    _pano = sys.modules.get("nihai.web.web_dashboard")      # yalnız --serve-web ile yüklenir
    if _pano is not None:
        _pano.canli_baglam_kaydet(store, pmgr)

    # Oto-tarayıcı (dinamik watchlist). --once'ta yalnızca --screen-now ile senkron taranır;
    # canlı döngüde arka plan thread'i periyodik tarar (60 sn döngüyü bloke etmez).
    screener: AutoScreener | None = None
    if (once or screen_now) and AUTO_SCREEN_ENABLED and screen_now:
        logger.info("[main] --screen-now: senkron evren taraması başlıyor (uzun sürebilir)…")
        try:
            scan_and_update()
        except Exception as e:
            logger.exception(f"[main] senkron tarama hatası: {e}")

    last_heartbeat = now_et_naive()
    last_market_open = is_market_open()
    last_regime: str | None = None        # A4 — rejim değişimini logla
    last_regime_check: datetime | None = None   # A4 — yenileme throttle'ı
    # B1 — FRED makro risk-off yenileme throttle'ı (default kapalı: MACRO_GATE)
    last_macro_check: datetime | None = None
    last_macro_label: str | None = None
    # C2 — Reddit sentiment ileriye-birikimi (default kapalı: SENTIMENT_COLLECT)
    last_sentiment_check: datetime | None = None
    sentiment_collect = os.getenv("SENTIMENT_COLLECT", "false").strip().lower() in (
        "1", "true", "yes", "on")
    sentiment_refresh_min = float(os.getenv("SENTIMENT_REFRESH_MIN", "60"))
    # Alpaca çağrı-hızı gözlemi (rate-limit): sınıf-düzeyi sayacı her tur ölç, gerçek
    # geçen süreye göre çağrı/dk hesapla. Alpaca kullanılmıyorsa sayaç 0 kalır (zararsız).
    from nihai.data.alpaca_provider import AlpacaProvider
    last_alpaca_total = AlpacaProvider.call_counts()["total"]
    last_rate_mono = time_mod.monotonic()

    if once:
        # Önce açık pozisyonları güncelle (varsa çıkış tetikleyebilir — kill-switch hook auto)
        closed = pmgr.update_all_with_quotes(provider, now_et_naive())
        for ce in closed:
            telegram_bot.notify_alert(
                "INFO" if ce.realized_pnl >= 0 else "WARN",
                f"Pozisyon kapandı: {ce.pos.symbol} @ {ce.exit_price:.2f} | "
                f"{ce.exit_trigger} | PnL {ce.realized_pnl:+.2f}"
            )
        for sym in watchlist.current():
            scan_symbol(sym, provider, risk_mgr, store, breaker, pmgr)
        terminal_dashboard.render_once(portfolio)
        store.save()
        return

    # Canlı modda oto-tarayıcı thread'ini başlat
    if AUTO_SCREEN_ENABLED:
        def _notify_update(new_list):
            telegram_bot.notify_alert(
                "INFO", f"Watchlist güncellendi ({len(new_list)} sembol): "
                        f"{', '.join(new_list[:12])}{' …' if len(new_list) > 12 else ''}")
        screener = AutoScreener(interval_hours=AUTO_SCREEN_INTERVAL_HOURS,
                                run_immediately=not screen_now,  # screen_now ise zaten taradık
                                on_update=_notify_update)
        screener.start()

    # İki-katman paralel tarama: odak (en iyi N, TradingView) + geniş (kalan, yfinance).
    # Her tarama thread'i KENDİ sağlayıcı örneğini kullanır (sağlayıcılar thread-safe değil).
    # Ana döngü pozisyon güncellemesi için AYRI bir sağlayıcı örneği tutar (provider).
    # Tur-106 (BULGU-106-33): pano nesnesi tarama iş parçacıklarından ÖNCE kurulur ve onlara verilir. Eskiden
    # `dashboard=None` geçiliyordu ("rich Live thread-safe değil") → iki-katman kipinde kabul edilen sinyal / açılış
    # satırları "son sinyaller"e HİÇ düşmüyordu (yalnız ana döngünün kapanışları). `add_signal_line` yalnız kilitli
    # bir listeye ekler; Live'a dokunan `refresh` ana döngüde kalır.
    dash = terminal_dashboard.LiveDashboard()
    dash.recent_signals = terminal_dashboard.tarihli_satirlar(store.recent_signals)[-20:]  # restart sonrası geri yükle
    tiers: list[TierScanner] = []
    if DUAL_TIER_ENABLED:
        def _scan_tier(sym, prov):
            # ortak state PORTFOLIO_LOCK ile korunur; pano yalnız satır biriktirir
            scan_symbol(sym, prov, risk_mgr, store, breaker, pmgr, dash, PORTFOLIO_LOCK)
        def _make_tier(name, prov_name, get_syms, throttle):
            # Sağlayıcı kurulamı (örn. tvDatafeed import hatası) tüm uygulamayı çökertmesin
            try:
                prov = get_provider(prov_name)
            except Exception as e:
                logger.error(f"[main] {name} katmanı sağlayıcısı ({prov_name}) kurulamadı: "
                             f"{e} — bu katman atlanıyor")
                return None
            return TierScanner(name=name, provider=prov, get_symbols=get_syms,
                               scan_one=_scan_tier, throttle=throttle, cycle_gap=TIER_CYCLE_GAP,
                               gate=can_enter_new_position)
        focused = _make_tier("odak", FOCUSED_PROVIDER,
                             lambda: watchlist.partition(FOCUSED_TOP_N)["focused"], FOCUSED_THROTTLE)
        broad = _make_tier("genis", BROAD_PROVIDER,
                           lambda: watchlist.partition(FOCUSED_TOP_N)["broad"], BROAD_THROTTLE)
        tiers = [t_ for t_ in (focused, broad) if t_ is not None]
        for t_ in tiers:
            t_.start()
        if tiers:
            logger.info(f"[main] iki-katman aktif ({len(tiers)} katman): "
                        f"odak={FOCUSED_PROVIDER}(ilk {FOCUSED_TOP_N}) + geniş={BROAD_PROVIDER}(kalan) "
                        f"| ana döngü yalnızca pozisyon yönetir")
        else:
            logger.error("[main] iki-katman: hiçbir katman başlatılamadı — sağlayıcıları kontrol et")

    # Telegram komut dinleyici (iki-yönlü) — operatör telefondan watchlist/durum yönetir.
    def _cmd_add(arg):
        if not arg:
            return "Kullanım: /add SEMBOL"
        return f"📌 {arg} pinlendi (oto-tarama silemez)" if watchlist.add(arg, pin=True) \
            else f"{arg} zaten listede"
    def _cmd_remove(arg):
        if not arg:
            return "Kullanım: /remove SEMBOL"
        return f"🚫 {arg} engellendi (oto-tarama geri ekleyemez)" if watchlist.remove(arg, block=True) \
            else f"{arg} listede yoktu"
    def _cmd_watchlist(arg):
        cur = watchlist.current()
        return (f"Watchlist ({len(cur)}): {', '.join(cur[:40])}{' …' if len(cur) > 40 else ''}\n"
                f"📌 pin: {', '.join(watchlist.pins()) or '—'}\n"
                f"🚫 block: {', '.join(watchlist.blocks()) or '—'}\n"
                f"son tarama: {watchlist.last_scan() or '—'}")
    def _cmd_status(arg):
        # Tur-106 (BULGU-106-25): "NLV" gerçekleşmiş özkaynaktı; pano (106-7) açık K/Z'yi katıyor → aynı anda iki sayı.
        return (f"NLV ${portfolio.equity + portfolio.open_pnl:,.0f} (gerçekleşmiş ${portfolio.equity:,.0f}) | "
                f"açık poz {len(portfolio.open_positions)}/"
                f"{CONFIG.portfolio.max_open_positions} | günlük PnL {portfolio.daily_pnl:+.2f}")
    def _cmd_grafik(arg):
        if not arg:
            return "Kullanım: /grafik SEMBOL (örn. /grafik AAPL)"
        from nihai.notifier import chart_render
        png = chart_render.render_price_png(arg)
        if png is None:
            return f"{arg} için grafik üretilemedi (veri yok?)."
        telegram_bot.send_photo(png, caption=f"📈 {arg}")
        return None  # foto gönderildi → ek metin yok
    def _cmd_yorum(arg):
        if not arg:
            return "Kullanım: /yorum SEMBOL (örn. /yorum AAPL) — grafik + AI okuması"
        from nihai.notifier import chart_analyst
        png, read = chart_analyst.analyze_chart(arg)
        if png is None:
            return f"{arg} için grafik üretilemedi (veri yok?)."
        if read is None:
            telegram_bot.send_photo(png, caption=f"📈 {arg} (AI okuması üretilemedi — anahtar/ağ?)")
            return None
        telegram_bot.send_photo(png, caption=chart_analyst.format_telegram(read, arg, "1h"))
        return None  # foto + AI okuması tek mesajda gönderildi
    def _cmd_beyin(arg):
        # AI dürüstlük karnesi (KARAR-DIŞI, SALT OKUNUR): kaynak-başı Brier/BSS + son
        # öz-değerlendirme kartı + kalibrasyon şeridi. Hiçbir sınav yeniden koşulmaz.
        return telegram_bot.ai_status_text()
    def _cmd_dur(arg):
        # Madde 47 — acil manuel müdahale: "flatten-all" DEĞİL "sinyalleri durdur".
        # Devre kesici KALICIDIR; yalnız /devam temizler (hız sınırı ile karıştırma).
        # Açık pozisyonlara DOKUNMAZ — çıkış yolu kesiciden bağımsızdır (bir operatör
        # frenini zarar-kesmenin önüne koymak bu sistemdeki en pahalı hata olurdu.)
        breaker.trip(f"MANUEL{('_' + arg) if arg else ''}")
        return "🔴 Devre kesici AÇIK — yeni sinyal yayılmayacak. Sürdürmek için /devam."
    def _cmd_devam(arg):
        if not breaker.tripped:
            return "🟢 Devre kesici zaten kapalı (sinyaller akıyor)."
        onceki = breaker.trip_reason
        breaker.reset()
        return f"🟢 Devre kesici sıfırlandı (önceki sebep: {onceki or '—'})."
    def _cmd_help(arg):
        return ("Komutlar:\n/add SEM — pinle\n/remove SEM — engelle\n"
                "/watchlist — liste\n/status — portföy\n/grafik SEM — fiyat grafiği\n"
                "/yorum SEM — grafik + AI okuması\n/beyin — AI dürüstlük karnesi\n"
                "/dur — sinyalleri durdur (Madde 47)\n/devam — sinyalleri sürdür\n/help")
    tg_handlers = {"add": _cmd_add, "remove": _cmd_remove, "watchlist": _cmd_watchlist,
                   "list": _cmd_watchlist, "status": _cmd_status,
                   "grafik": _cmd_grafik, "chart": _cmd_grafik,
                   "yorum": _cmd_yorum, "analiz": _cmd_yorum,
                   "beyin": _cmd_beyin, "ai": _cmd_beyin,
                   "dur": _cmd_dur, "stop": _cmd_dur,
                   "devam": _cmd_devam, "resume": _cmd_devam, "help": _cmd_help}
    tg_listener = telegram_bot.CommandListener(tg_handlers)
    tg_listener.start()  # telegram kapalıysa no-op

    # Canlı tarama paralelliği (tek-katman mod). DUAL_TIER zaten kendi paralel
    # katmanlarını kullanır; burada yalnızca tek-katman seri tarama hızlandırılır.
    # I/O-bound → thread havuzu. Her worker thread KENDİ provider'ını kurar
    # (thread-local), ortak state PORTFOLIO_LOCK ile korunur, dashboard=None
    # (rich Live thread-safe değil — dual-tier ile aynı kabul).
    scan_pool = None
    _scan_tls = threading.local()
    if not DUAL_TIER_ENABLED and LIVE_SCAN_WORKERS > 1:
        from concurrent.futures import ThreadPoolExecutor
        scan_pool = ThreadPoolExecutor(max_workers=LIVE_SCAN_WORKERS,
                                       thread_name_prefix="live-scan")
        logger.info(f"[main] canlı tarama paralel: {LIVE_SCAN_WORKERS} thread "
                    f"(thread-local provider, ortak state PORTFOLIO_LOCK ile korunur)")

    def _scan_worker(sym: str) -> None:
        try:
            prov = getattr(_scan_tls, "provider", None)
            if prov is None:
                prov = get_provider()   # thread'e özel örnek (sağlayıcılar paylaşılmaz)
                _scan_tls.provider = prov
            scan_symbol(sym, prov, risk_mgr, store, breaker, pmgr, dash, PORTFOLIO_LOCK)
        except Exception as e:  # bir sembolün hatası tüm turu bozmasın
            logger.warning(f"[live-scan] {sym}: {e}")

    # Canlı döngü — rich Live ile
    with dash:
        try:
            while True:
                _tur_basi = time_mod.monotonic()
                # Kill-switch aktifken programı durdurmuyoruz; sunucunun ayakta kalmasını sağlıyor,
                # yeni pozisyon girişlerini scan_symbol seviyesinde engelliyoruz.
                if store.kill_switch_active:
                    logger.warning("[main] KILL-SWITCH aktif — yeni pozisyon girişi engellendi, web sunucusu aktif kalmaya devam ediyor")

                if is_market_open():
                    # Madde 5 — 1m granüler stop fitili kontrolü (hisse pozisyonları).
                    # Barlar kilit DIŞINDA çekilir (Tur-80: ağ gecikmesi pano uçlarını ve
                    # tarama thread'lerini bekletiyordu); pozisyon mutasyonu kilitli.
                    frames = pmgr.fetch_1m_bars(provider)
                    with PORTFOLIO_LOCK:
                        closed = pmgr.update_all_with_1m_bars(provider, now_et_naive(), prefetched=frames)
                        # Opsiyonları premium-bazlı güncelle (hisseler 1m'de işlendi → çift sayma yok)
                        closed += pmgr.update_all_with_quotes(provider, now_et_naive(), skip_stocks=True)
                    for ce in closed:
                        dash.add_signal_line(
                            f"CLOSE {ce.pos.symbol} @ {ce.exit_price:.2f} | "
                            f"{ce.exit_trigger} | PnL {ce.realized_pnl:+.2f}")
                        telegram_bot.notify_alert(
                            "INFO" if ce.realized_pnl >= 0 else "WARN",
                            f"{ce.pos.symbol} kapandı | {ce.exit_trigger} | "
                            f"PnL {ce.realized_pnl:+.2f}")
                    # A4 — piyasa rejimi → risk çarpanı (sizing'den ÖNCE güncelle). GÜNLÜK
                    # benchmark barlarıyla (SMA200≈200 gün) hesaplanır; rejim yavaş değiştiği
                    # için refresh_minutes'te bir yenilenir. Default kapalı → hiç ekstra çağrı yok.
                    if CONFIG.regime.enable_regime_risk:
                        reg_now = now_et_naive()
                        due = (last_regime_check is None or
                               (reg_now - last_regime_check).total_seconds()
                               >= CONFIG.regime.refresh_minutes * 60)
                        if due:
                            try:
                                from nihai.signals.market_regime import compute_regime
                                bench_df = provider.get_bars(
                                    CONFIG.regime.benchmark,
                                    cast(Interval, CONFIG.regime.benchmark_interval),
                                    lookback_bars=CONFIG.regime.sma_long + 10)
                                res = compute_regime(bench_df)
                                risk_mgr.regime_multiplier = res.multiplier
                                risk_mgr.regime_label = res.regime
                                last_regime_check = reg_now
                                if res.regime != last_regime:
                                    logger.info(f"[regime] {res.detail} → risk ×{res.multiplier:.2f}")
                                    dash.add_signal_line(f"REGIME {res.regime} ×{res.multiplier:.2f}")
                                    last_regime = res.regime
                            except Exception as e:
                                logger.debug(f"[regime] hesaplanamadı ({e}) — çarpan 1.0'da bırakıldı")
                                risk_mgr.regime_multiplier = 1.0

                    # B1 — FRED makro risk-off çarpanı (sizing'den ÖNCE). Default KAPALI
                    # (MACRO_GATE). Makro yavaş değişir → MACRO_REFRESH_MIN'de bir yenilenir.
                    if CONFIG.macro.enable_macro_gate:
                        mc_now = now_et_naive()
                        due = (last_macro_check is None or
                               (mc_now - last_macro_check).total_seconds()
                               >= CONFIG.macro.refresh_minutes * 60)
                        if due:
                            try:
                                risk_mgr.refresh_macro()
                                last_macro_check = mc_now
                                if risk_mgr.macro_label != last_macro_label:
                                    logger.info(f"[macro] {risk_mgr.macro_label} → "
                                                f"risk ×{risk_mgr.macro_multiplier:.2f}")
                                    dash.add_signal_line(
                                        f"MACRO {risk_mgr.macro_label} "
                                        f"×{risk_mgr.macro_multiplier:.2f}")
                                    last_macro_label = risk_mgr.macro_label
                            except Exception as e:
                                # Kasıtlı: son bilinen çarpan KORUNUR (rejim yolunun aksine
                                # 1.0'a sıfırlanmaz) — geçici FRED kesintisi risk-off korumasını
                                # kaldırmasın. Makro yavaş değişir; bayat değer nötrden güvenli.
                                logger.debug(f"[macro] yenilenemedi ({e}) — çarpan son bilinen değerde")

                    # C2 — Reddit sentiment İLERİYE-BİRİKİMİ (Pushshift kapalı → canlıda birikir).
                    # Default KAPALI (SENTIMENT_COLLECT); ağ çağrısı yapar, geçmiş birikene dek
                    # sentiment stratejisi WAIT döner. Reddit creds yoksa sessizce no-op.
                    if sentiment_collect:
                        sc_now = now_et_naive()
                        due = (last_sentiment_check is None or
                               (sc_now - last_sentiment_check).total_seconds()
                               >= sentiment_refresh_min * 60)
                        if due:
                            try:
                                from nihai.data.sentiment import update_sentiment
                                for _s in watchlist.current():
                                    update_sentiment(_s)
                                last_sentiment_check = sc_now
                                logger.debug("[sentiment] watchlist güncellendi")
                            except Exception as e:
                                logger.debug(f"[sentiment] güncellenemedi ({e})")

                    # İki-katman modda yeni-sinyal taramasını paralel thread'ler yapar;
                    # ana döngü yalnızca pozisyon yönetir. Tek-thread modda burada taranır.
                    if not DUAL_TIER_ENABLED:
                        syms = watchlist.current()
                        if scan_pool is not None:
                            # Paralel: tümünü gönder, hepsi bitene kadar bekle (pozisyon
                            # yönetimi bir sonraki tura geçmeden tarama tamamlanmalı).
                            list(scan_pool.map(_scan_worker, syms))
                        else:
                            for sym in syms:
                                scan_symbol(sym, provider, risk_mgr, store, breaker, pmgr, dash)
                else:
                    log_decision("SESSION_CLOSED", "market kapalı — sinyal taraması yok")

                # Madde 36 — sofistike heartbeat: seans geçişi + detaylı durum
                now = now_et_naive()
                is_open = is_market_open()
                # Seans açılış/kapanış geçişi
                if is_open != last_market_open:
                    if is_open:
                        # Yeni seans günü → günlük sayaçları sıfırla (daily_pnl + consecutive_stops)
                        if portfolio.roll_session_if_new_day(now.date()):
                            logger.info(f"[main] yeni seans {now.date()} — günlük sayaçlar sıfırlandı")
                        telegram_bot.notify_alert(
                            "INFO",
                            f"Market acildi. NLV ${portfolio.equity + portfolio.open_pnl:,.0f} | "
                            f"poz {len(portfolio.open_positions)}/"
                            f"{CONFIG.portfolio.max_open_positions}"
                        )
                    else:
                        telegram_bot.notify_alert(
                            "INFO",
                            f"Market kapandi. Gunluk PnL {portfolio.daily_pnl:+.2f} | "
                            f"NLV ${portfolio.equity + portfolio.open_pnl:,.0f} "
                            f"(gerceklesmis ${portfolio.equity:,.0f})"
                        )
                    last_market_open = is_open
                # 30 dk'da bir detaylı heartbeat (sadece market açıkken)
                if is_open and now - last_heartbeat > timedelta(minutes=30):
                    telegram_bot.send(
                        f"heartbeat | NLV ${portfolio.equity + portfolio.open_pnl:,.0f} | "
                        f"poz {len(portfolio.open_positions)}/{CONFIG.portfolio.max_open_positions} | "
                        f"PnL {portfolio.daily_pnl:+.0f} | "
                        f"{now:%H:%M ET}"
                    )
                    last_heartbeat = now

                dash.refresh(portfolio)

                # State mutasyonu/kaydı paralel katmanlarla çakışmasın diye kilitli.
                with PORTFOLIO_LOCK:
                    # Madde 23 — takası dolmuş nakdi düş
                    store.settle_matured(now.date())
                    # Madde 15/9 — canlı equity eğrisi kalıcılığı (yalnızca seans açıkken;
                    # equity sadece pozisyon kapanınca = seans içinde değişir)
                    if is_open:
                        append_equity_point(portfolio.equity, now)
                    # recent_signals kalıcılığı
                    store.recent_signals = list(dash.recent_signals)
                    store.save()

                # Alpaca çağrı-hızı → MAKİNE-OKUNUR bütçe (Tur-58). Eskiden yalnız
                # loglanıyordu; artık aynı sayı `observability.BUDGET`e girer, `/metrics`
                # ile dışa açılır ve yumuşak tavanın ÜSTÜNDE tarama döngüsünü kısar.
                _ac = AlpacaProvider.call_counts()
                _now_mono = time_mod.monotonic()
                _elapsed = _now_mono - last_rate_mono
                _delta = _ac["total"] - last_alpaca_total
                _rate = OBS.BUDGET.observe()
                # Tur-89: ölçülen İŞ süresi (predict döngüsüyle aynı anlam). Eskiden 60 sn uykuyu da
                # içeren periyot yazılıyordu → pano "Döngü Gecikmesi 60,0 sn" gösteriyordu.
                OBS.record_loop("scan", _now_mono - _tur_basi)
                if is_open and _elapsed > 0:
                    logger.info(
                        f"[rate] alpaca {_delta / _elapsed * 60:.0f} çağrı/dk "
                        f"(son {_elapsed:.0f}s'de {_delta} çağrı) | kümülatif "
                        # Tur-106 (BULGU-106-14): hız TÜM sayaçlardan; yalnız bars/quote basılınca hub kipinde
                        # "5 çağrı/dk · bars=0 quote=0" çelişkisi (çağrılar Emir Akışı `get_trades`ti).
                        f"{' '.join(f'{k}={v}' for k, v in _ac.items() if v and k != 'total') or '0'} | limit 200/dk"
                    )
                last_alpaca_total = _ac["total"]
                last_rate_mono = _now_mono

                # Bir sonraki tarama: 1m beklemeli (1h barlar için aşırı sık tarama anlamsız)
                # Kısma yalnız yumuşak tavanın (vars. limitin %80'i) ÜSTÜNDE > 0 döner →
                # bugünkü rejimde ek uyku TAM 0.0'dır (testle kilitli).
                _extra = OBS.BUDGET.throttle_sleep(60.0)
                if _extra > 0:
                    logger.warning(f"[rate] kota freni: +{_extra:.0f}s (ölçülen {_rate:.0f}/dk, "
                                   f"yumuşak tavan {OBS.BUDGET.limit_per_min * OBS.BUDGET.soft_pct / 100:.0f}/dk)")
                time_mod.sleep(60 + _extra)
        except KeyboardInterrupt:
            logger.info("[main] kullanıcı durdurdu (Ctrl+C)")
            store.save()
        finally:
            if screener is not None:
                screener.stop()
            for t_ in tiers:
                t_.stop()
            if scan_pool is not None:
                scan_pool.shutdown(wait=False)
            tg_listener.stop()


# ---------------------------------------------------------------------------
_ANA_KILIT: object = None     # süreç ömrü boyunca açık: kilit ancak süreç ölünce bırakılır


def _tek_ana_surec() -> bool:
    """Veri dizinine TEK canlı ana süreç (tarama + kayıt + pano) yazar. Tur-106 (BULGU-106-24): kilit yoktu;
    2026-09-17'de gözcü ile `proje-ac` iki kayıtçı açtı ve ikisi aynı pozisyonları ayrı ayrı kapatıp
    `trades.csv`'ye çift satır yazdı (Tur-80). Tur-80 yalnız başlatma SIRASINI düzeltti; proje açıkken `run.ps1`/
    `python -m nihai.execution.main` ikinci bir tarayıcıyı hâlâ sessizce açıyordu."""
    global _ANA_KILIT
    from nihai.atomic_io import tek_yazar_kilidi

    _ANA_KILIT = tek_yazar_kilidi(DATA_DIR / "main.lock")
    return _ANA_KILIT is not None


def main():
    ap = argparse.ArgumentParser(description="Nihai Algoritma — Sinyal/Rapor Modu")
    ap.add_argument("--once", action="store_true", help="tek tur tara, çık")

    ap.add_argument("--screen-now", action="store_true",
                    help="başlangıçta senkron evren taraması yap, watchlist'i güncelle")
    ap.add_argument("--flatten", action="store_true",
                    help="Madde 47: tüm açık pozisyonları kapat ve çık (ana süreç ÇALIŞMIYORKEN; "
                         "çalışırken reddedilir → panodan Flatten All)")
    ap.add_argument("--serve-web", action="store_true",
                    help="Web arayüzünü (FastAPI) arka planda başlatır")
    args = ap.parse_args()

    # Başlangıç config doğrulaması — yanlış .env değerleri sessiz hataya dönüşmesin
    errors, warnings = validate_config()
    for w in warnings:
        logger.warning(f"[config] {w}")
    if errors:
        for e in errors:
            logger.error(f"[config] {e}")
        raise SystemExit("Config hatası — .env'i düzelt (ayrıntılar yukarıda).")

    if args.flatten:
        # Tur-106 (BULGU-106-30): ana süreç çalışırken AYRI süreçten flatten diskteki kopyayı kapatır; ana döngü
        # belleğindeki portföyü ≤60 sn'de geri yazar (pozisyonlar geri gelir, sonra ikinci kez kapanır). Kilit bu
        # süreçte alınır: ana süreç yoksa flatten güvenle koşar ve o sürerken yeni bir ana süreç de açılamaz.
        if not _tek_ana_surec():
            raise SystemExit("Ana süreç çalışıyor: --flatten ayrı süreçten ETKİSİZ olurdu (döngü portföyü geri yazar). "
                             "Acil kapatma: pano → Operatör Kontrol → Flatten All (canlı portföye uygulanır) "
                             "ya da önce proje-kapat.ps1, sonra --flatten.")
        flatten_all_action()
        return

    if not _tek_ana_surec():
        raise SystemExit("Bu veri dizinini başka bir ana süreç yazıyor (data/main.lock) — ikinci kopya açılmadı.")

    if args.serve_web:
        try:
            from nihai.web.web_dashboard import start_web_server
            start_web_server()
        except Exception as e:
            logger.error(f"[main] Web sunucusu başlatılamadı: {e}")

    run_loop(once=args.once, screen_now=args.screen_now)


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    main()
