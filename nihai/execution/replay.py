"""Madde 6.3: canlı scan_symbol + PositionManager hattında izole tarihsel replay.

Tek iş parçacığı; saat yamaları süreç geneline yönelik olduğundan canlı süreçte
kullanılmaz. run_replay API'si de CLI gibi HER koşuyu temiz alt süreçte çalıştırır.
Config ana süreçte import edilmez. .env load_dotenv(override=False) kullandığından
boş Telegram env'leri korunur. Alt süreç yazma çiti çıktı dışındaki yazıları reddeder.

YAPILMAYANLAR: run_loop kopyalanmaz; rejim/makro/sentiment tazeleme, heartbeat,
pano, katmanlı tarayıcılar, oto-screener, Telegram komutları çalıştırılmaz. Opsiyon
yolu boş zincirle kapsam dışıdır. Harici pead/insider/llm_event detektörleri WAIT
döner; olay guard'ları yalnız geçmişte alınmış ve TTL'i geçmemiş snapshot okur.
VIX indirilmez (stratejinin mevcut nötr fallback'i). Rejim/makro çarpanı 1 kalır.

Barlar naive-UTC BAŞLANGIÇ damgalıdır. 1d = 24 saat; config.INTERVAL_MINUTES[1d]
=390 işlem dakikasıdır, gece yarısı damgalı günlük barın kapanışı DEĞİLDİR.
Günlük barı 24 saat bekletmek ihtiyatlıdır (seans kapanışından sonra gecikme).
Kotasyon son kapanmış 1m close; makas ölçülmez. Eksik dakikada eski bar/quote
yeniden görülebilir: mevcut PositionManager bu barları dedup etmez.

required_intervals: 1m pozisyon yönetimi, HTF_INTERVAL sinyal/beta, 1d SSR. Canlıyla aynı
bar: 1m yalnız normal seans, gün-içi HTF o dakikalardan 09:30 ızgarasında türetilir (Tur-91).
Yükleyici 1d için 550 takvim günü (>=260 bar hedefi), intraday için >=60 gün ve
>=200 bar hedefler; eksik kapsama ayrıca raporlanır, aralık ikamesi yapılmaz.
Snapshot modeller/kalibrasyon/AI hafızası bugünkü dosya sürümüdür; PIT veya kâr
kanıtı değildir. state/trades/cooldown/kill-switch geçmişi kopyalanmaz: taze portföy.
Zaman envanteri AST ile execution/risk/signals/data altında çıkarılır; yamalanmayan
çağrılar özetin sapma listesinde adlarıyla bulunur. Loguru/mtime/monotonic ve PID/
UUID dosya metası gerçek kalır; bunlar özetin parçası değildir. Özet alanı dışlanmaz.
"""
from __future__ import annotations

import argparse
import ast
import csv
import importlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from nihai.data.data_provider import DataProvider, Interval, Quote, resample_bars, rth_mask

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[2]
DURATIONS = {"1m": timedelta(minutes=1), "5m": timedelta(minutes=5),
             "15m": timedelta(minutes=15), "1h": timedelta(hours=1), "1d": timedelta(days=1)}
# Sadece modülün kendi bağlaması değiştirilir; datetime/time stdlib modülleri değişmez.
DATE_MODULES = (
    "nihai.execution.state", "nihai.execution.watchlist", "nihai.risk.risk_manager",
    "nihai.risk.safety", "nihai.risk.position_manager", "nihai.risk.options_analyzer",
    "nihai.data.event_calendar", "nihai.data.iv_history", "nihai.signals.signal_engine",
    "nihai.audit_chain", "nihai.ai.decision", "nihai.ai.episodic", "nihai.ai.learning",
)
TIME_MODULES = {"nihai.execution.main": "time_mod", "nihai.signals.model_promotion": "time"}
INPUTS = (
    "meta_model.json", "meta_model.pkl", "meta_activation.json", "runtime_strategy.json",
    "watchlist.json", "sector_map.json", "iv_history", "events", "sentiment_history",
    "forecast_calibration.json", "ai_calibrators.json", "ai_learning_state.json",
    "ai_episodes.jsonl", "ai_lesson_candidates.jsonl",
)
DEVIATIONS = (
    "OPSİYON_KAPSAM_DIŞI: boş zincir; OPTION_MODE=stock.",
    "MAKAS_ÖLÇÜLMEDİ: bid=ask=son kapanmış 1m close; spread_measured=False; tick/NBBO yok.",
    "GÜNLÜK_BAR: başlangıç+24h; kapanış sonrası ihtiyatlı gecikme, 390 dk kullanılmaz.",
    "EKSİK_DAKİKA: son kapanmış bar/quote bayat olabilir; PM aynı barı tekrar işleyebilir.",
    "DÖNGÜ_KAPSAMI: rejim/makro çarpanları 1; tazeleme, heartbeat, pano, tier, screener, Telegram yok.",
    "VIX_YOK: indirme kapalı; mevcut nötr fallback. pead/insider/llm_event WAIT.",
    "BORÇ_BAYRAĞI_YOK: shortable/easy_to_borrow tarihsel snapshot yok; Alpaca yedeği sorgulanmaz.",
    "OLAY_SAATİ: eski naive fetched_at ET kabul edilir; özgün makine saat dilimi kayıtta yok.",
    "SNAPSHOT_PIT_DEĞİL: model, politika, sektör, watchlist, AI hafızası bugünkü kopya; tarihsel sürüm yok.",
    "KOPYALANMAZ: state/trades/cooldown/kill-switch (taze portföy); neural/ai_models ve prediction logları "
    "(scan_symbol okumaz); harici olay/insider/LLM cache (ilgili detektörler kapalı).",
    "KALİBRASYON: forecast_calibration/ai_calibrators kopyalanır; scan_symbol bunları tüketmez.",
    "GERÇEK_META: loguru/mtime/monotonic/PID/UUID; özet bunlara dayanmaz, özetten alan dışlanmaz.",
)


class SimulatedClock:
    """Naive girdi ET kabul edilir; aware girdi ET'ye çevrilir. Tek-thread kullanım."""

    def __init__(self, dt: datetime):
        self.set(dt)

    def set(self, dt: datetime) -> None:
        if not isinstance(dt, datetime) or pd.isna(dt):
            raise ValueError("Geçerli datetime gerekli")
        self.now = dt.replace(tzinfo=ET) if dt.tzinfo is None else dt.astimezone(ET)

    def advance(self, minutes: float) -> datetime:
        if not math.isfinite(minutes):
            raise ValueError("Dakika sonlu olmalı")
        self.set(self.now.astimezone(UTC) + timedelta(minutes=minutes))
        return self.now

    @contextmanager
    def patch(self) -> Iterator[SimulatedClock]:
        from nihai.data import market_calendar

        clock = self

        # isinstance gerçek datetime/date örneklerini tanımaya devam etmeli: StateStore
        # serializer ve PM'nin date-only ayrımı buna dayanır.
        class DateMeta(type):
            def __instancecheck__(cls, value):
                return isinstance(value, date)

        class DatetimeMeta(type):
            def __instancecheck__(cls, value):
                return isinstance(value, datetime)

        class ReplayDate(date, metaclass=DateMeta):
            @classmethod
            def today(cls):
                return clock.now.date()

        class ReplayDatetime(datetime, metaclass=DatetimeMeta):
            @classmethod
            def now(cls, tz: tzinfo | None = None):
                return clock.now.astimezone(tz) if tz else clock.now.replace(tzinfo=None)

            @classmethod
            def utcnow(cls):
                return clock.now.astimezone(UTC).replace(tzinfo=None)

        with ExitStack() as stack:
            stack.enter_context(patch.object(market_calendar, "now_et", lambda: clock.now))
            for name in DATE_MODULES:
                module = sys.modules.get(name)
                if module is None:
                    continue  # salt saat testi canlı hattı import etmez
                for attr, real, replacement in (("date", date, ReplayDate),
                                                 ("datetime", datetime, ReplayDatetime)):
                    if getattr(module, attr, None) is real:
                        stack.enter_context(patch.object(module, attr, replacement))
            for name, attr in TIME_MODULES.items():
                module = sys.modules.get(name)
                if module is not None:
                    proxy = SimpleNamespace(**vars(time))
                    proxy.time = lambda: clock.now.timestamp()
                    stack.enter_context(patch.object(module, attr, proxy))
            yield self


def _symbol(value: str) -> str:
    value = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,31}", value):
        raise ValueError(f"Geçersiz sembol: {value!r}")
    return value


class ReplayProvider(DataProvider):
    name = "replay"

    def __init__(self, bars: Mapping[tuple[str, str], pd.DataFrame], clock: SimulatedClock):
        self.clock = clock
        self.bars: dict[tuple[str, str], pd.DataFrame] = {}
        for (symbol, interval), frame in bars.items():
            if interval not in DURATIONS:
                raise ValueError(f"Desteklenmeyen aralık: {interval}")
            if frame.empty and not isinstance(frame.index, pd.DatetimeIndex):
                frame = frame.copy()
                frame.index = pd.DatetimeIndex([])
            if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is not None
                    or frame.index.hasnans or frame.index.has_duplicates):
                raise ValueError("Bar indeksi benzersiz, NaT içermeyen naive-UTC olmalı")
            if not frame.empty:
                values = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
                if not all(math.isfinite(v) for v in values.flat) or (values[:, :4] <= 0).any() \
                        or (values[:, 4] < 0).any():
                    raise ValueError("OHLCV sonlu, fiyat pozitif ve hacim negatif olmayan olmalı")
                if ((frame.high < frame[["open", "low", "close"]].max(axis=1)).any()
                        or (frame.low > frame[["open", "high", "close"]].min(axis=1)).any()):
                    raise ValueError("Tutarsız OHLC")
            self.bars[(_symbol(symbol), interval)] = frame.sort_index().copy()

    def get_bars(self, symbol: str, interval: Interval, lookback_bars: int = 200) -> pd.DataFrame:
        if lookback_bars <= 0:
            return pd.DataFrame()
        frame = self.bars.get((_symbol(symbol), interval))
        if frame is None:
            return pd.DataFrame()
        cutoff = self.clock.now.astimezone(UTC).replace(tzinfo=None) - DURATIONS[interval]
        end = frame.index.searchsorted(pd.Timestamp(cutoff), side="right")
        return frame.iloc[max(0, end - lookback_bars):end].copy()

    def get_quote(self, symbol: str) -> Quote | None:
        bars = self.get_bars(symbol, "1m", 1)
        if bars.empty:
            return None
        price = float(bars.close.iloc[-1])
        return Quote(_symbol(symbol), price, price, price, self.clock.now, spread_measured=False)

    def get_option_chain(self, symbol: str, expiry: str | None = None) -> dict:
        return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiry": expiry,
                "underlying_price": None}


def required_intervals() -> tuple[str, ...]:
    from nihai.config import HTF_INTERVAL
    return tuple(dict.fromkeys(("1m", HTF_INTERVAL, "1d")))


def load_history(symbols, start, end, intervals, *, fetch: bool = False) -> tuple[dict, list[dict]]:
    """(bar sözlüğü, eksik kapsam listesi). fetch=False yalnız cache, mtime şartı yok.

    fetch=True yalnız izole worker'da kullanılabilir; cache'in sağ ucu eskimişse
    get_history'nin yalnız sol uca bakan cache kontrolünü kullanmadan yeniden çeker.
    """
    from nihai.data import alpaca_history as history
    from nihai.data.market_calendar import NYSE

    if fetch and os.environ.get("NIHAI_REPLAY_WORKER") != "1":
        raise ValueError("fetch yalnız izole CLI alt sürecinde kullanılabilir")
    start_day, end_day = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if end_day < start_day:
        raise ValueError("Bitiş başlangıçtan önce")
    sessions = NYSE.sessions_in_range(str(start_day), str(end_day))
    result: dict = {}
    missing: list[dict] = []
    for symbol in sorted({_symbol(s) for s in symbols}):
        for interval in dict.fromkeys(intervals):
            if interval not in DURATIONS:
                raise ValueError(f"Desteklenmeyen aralık: {interval}")
            days = 550 if interval == "1d" else 60
            lo = datetime.combine(start_day - timedelta(days=days), datetime.min.time(), UTC)
            hi = datetime.combine(end_day + timedelta(days=1), datetime.min.time(), UTC)
            if interval not in ("1m", "1d") and (symbol, "1m") in result:
                # Tur-91: canlı gün-içi HTF barı seans dakikalarından 09:30 ızgarasında kurulur
                # (Tur-90/91); araştırma önbelleğinin saat barı (09:00, uzatılmış seans) o bar değildir.
                src = result[(symbol, "1m")]
                frame = (resample_bars(src, int(DURATIONS[interval].total_seconds() // 60))
                         if not src.empty else pd.DataFrame())
            else:
                frame = history.get_history(symbol, interval, start=lo, end=hi, fetch=False)
                if fetch:
                    # Açık fetch isteği; mevcut get_history API'si ile izole cache'e yazılır.
                    with patch.object(history, "_read_cache", return_value=None):
                        frame = history.get_history(symbol, interval, start=lo, end=hi, fetch=True)
                if interval == "1m" and not frame.empty:
                    frame = frame[rth_mask(frame.index, 1)]      # canlı 1m de yalnız seans (Madde 22)
            result[(symbol, interval)] = frame
            detail: dict = {"symbol": symbol, "interval": interval}
            if frame.empty:
                missing.append({**detail, "reason": "CACHE_EMPTY_OR_MISSING"})
                continue
            first_open = (NYSE.session_open(sessions[0]).tz_convert(None) if len(sessions)
                          else pd.Timestamp(start_day))
            warm = int((frame.index + DURATIONS[interval] <= first_open).sum())
            need = 260 if interval == "1d" else 200
            if warm < need:
                missing.append({**detail, "reason": "WARMUP_SHORT", "available": warm, "required": need})
            # Günlük veri timestamp'ı midnight ET; intraday her RTH başlangıcı aranır.
            for session in sessions:
                if interval == "1d":
                    present = any(t.date() == session.date() for t in frame.index)
                    count = 0 if present else 1
                else:
                    opening = NYSE.session_open(session).tz_convert(None)
                    closing = NYSE.session_close(session).tz_convert(None)
                    # Gün-içi barlar seans açılışına çapalıdır (09:30, 10:30 … — Tur-91).
                    expected = pd.date_range(opening, closing, freq=pd.Timedelta(DURATIONS[interval]),
                                             inclusive="left")
                    count = len(expected.difference(frame.index))
                if count:
                    missing.append({**detail, "reason": "SESSION_GAPS", "session": str(session.date()),
                                    "missing_bars": count})
    return result, missing


def time_inventory() -> dict[str, list[str]]:
    """Yorum/docstring değil gerçek çağrılar; time_mod ve date.today dahil."""
    inventory: dict[str, list[str]] = {"patched": [], "deviation": []}
    paths = {path for folder in ("execution", "risk", "signals", "data")
             for path in (ROOT / "nihai" / folder).glob("*.py")}
    paths.update(ROOT.joinpath(*module.split(".")).with_suffix(".py") for module in DATE_MODULES)
    for path in sorted(paths):
        if path.name == "replay.py":
            continue
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            expr = ast.unparse(node.func)
            if expr not in {"datetime.now", "datetime.utcnow", "date.today", "time.time", "time_mod.time"}:
                continue
            patched = ((module in DATE_MODULES and expr.startswith(("date.", "datetime.")))
                       or module in TIME_MODULES or module == "nihai.data.market_calendar")
            group = "patched" if patched else "deviation"
            inventory[group].append(f"{module}:{node.lineno}:{expr}")
    return {key: sorted(values) for key, values in inventory.items()}


def _source_dirs() -> tuple[Path, Path, Path]:
    # config importu mkdir yapar; ana süreçte yalnız dotenv'yi salt-okunur çöz.
    from dotenv import dotenv_values
    env = {**dotenv_values(ROOT / ".env"), **os.environ}
    data = Path(env.get("NIHAI_DATA_DIR") or ROOT / "data").resolve()
    logs = Path(env.get("NIHAI_LOG_DIR") or ROOT / "logs").resolve()
    hist = Path(env.get("ALPACA_HIST_DIR") or data / "alpaca_hist").resolve()
    return data, logs, hist


def _prepare(out_dir: str | Path) -> tuple[Path, dict[str, str], dict]:
    out = Path(out_dir).resolve()
    source, logs, hist = _source_dirs()
    for protected in (ROOT / "data", ROOT / "logs", source, logs, hist):
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError("Replay çıktısı üretim/veri kaynaklarıyla örtüşemez")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Determinizm için yeni veya boş çıktı dizini gerekli")
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    copied, absent = [], []
    for name in INPUTS:
        src, dest = source / name, out / "data" / name
        if not src.exists():
            absent.append(name)
        elif src.is_dir():
            shutil.copytree(src, dest)
            copied.append(name)
        else:
            shutil.copy2(src, dest)
            copied.append(name)
    env = dict(os.environ)
    env.update(NIHAI_DATA_DIR=str(out / "data"), NIHAI_LOG_DIR=str(out / "logs"),
               ALPACA_HIST_DIR=str(out / "data" / "alpaca_hist"),
               NIHAI_REPLAY_WORKER="1", TELEGRAM_BOT_TOKEN="", TELEGRAM_CHAT_ID="",
               NIHAI_DISABLE_FILE_LOG="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
               PYTHONIOENCODING="utf-8", OPTION_MODE="stock")
    return out, env, {"copied": copied, "missing": absent, "history_source": str(hist)}


def _install_write_fence(out: Path) -> None:
    """Yalnız disposable worker; audit hook kaldırılamaz. Dirfd işlemleri kullanılmaz."""
    def check(value):
        if isinstance(value, int) or value is None:
            return  # stdout/stderr pipe dosya tanıtıcısı
        if not Path(os.fsdecode(value)).resolve().is_relative_to(out):
            raise PermissionError(f"Replay çıktı dışına yazamaz: {value}")

    def audit(event, args):
        if event == "open":
            _, mode, flags = args
            if (mode and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
                check(args[0])
        elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime", "os.truncate"}:
            check(args[0])
        elif event in {"os.rename", "os.link", "os.symlink"}:
            check(args[0])
            check(args[1])
    sys.addaudithook(audit)


@contextmanager
def _offline(clock: SimulatedClock, issues: set[str]) -> Iterator[None]:
    """Canlı sorguları kaynak sınırında kes; guard ve karar fonksiyonlarını koru."""
    from nihai.data import event_calendar as events
    from nihai.signals import strategy

    def cached_events(symbol, *, refresh=False):
        result = events._load_cache(symbol)
        if result is not None and result.fetched_at is not None:
            fetched = result.fetched_at
            now = clock.now if fetched.tzinfo else clock.now.replace(tzinfo=None)
            if fetched <= now:
                return result
        issues.add(f"EVENT_SNAPSHOT_UNAVAILABLE:{symbol} (eksik, TTL aşılmış veya gelecekten)")
        return events.CompanyEvents(symbol, source="replay_missing")

    def no_network(*args, **kwargs):
        issues.add("NETWORK_ATTEMPT_BLOCKED")
        raise OSError("Replay sırasında ağ kapalı")

    with ExitStack() as stack:
        stack.enter_context(patch.object(events, "get_company_events", cached_events))
        stack.enter_context(patch("nihai.data.data_provider.asset_flags", return_value=None))
        for attr, value in (("_VIX_LOADED", True), ("_VIX_DATES", None), ("_VIX_CLOSES", None)):
            stack.enter_context(patch.object(strategy, attr, value))
        for name in ("detect_pead", "detect_insider", "detect_llm_event"):
            stack.enter_context(patch.object(strategy, name, lambda *a, **kw:
                               (strategy.SignalAction.WAIT, ["REPLAY_EXTERNAL_SOURCE_DISABLED"])))
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection",
                       "socket.getaddrinfo", "requests.sessions.Session.request"):
            stack.enter_context(patch(target, no_network))
        try:
            from curl_cffi import requests as curl
        except ImportError:
            pass
        else:
            stack.enter_context(patch.object(curl.Session, "request", no_network))
            async def no_async_network(*args, **kwargs):
                no_network()
            stack.enter_context(patch.object(curl.AsyncSession, "request", no_async_network))
        yield


def _run_local(provider, symbols, sessions, *, out: Path, scan_every: int, inputs: dict,
               missing: list[dict]) -> dict:
    from loguru import logger

    from nihai import config
    from nihai.data.market_calendar import NYSE
    from nihai.execution import main as live

    if config.DATA_DIR.resolve() != out / "data" or config.LOG_DIR.resolve() != out / "logs":
        raise RuntimeError("Config izole dizinleri import anında okumalı")
    if live.telegram_bot.is_enabled():
        raise RuntimeError("Replay Telegram kapalı olmalı")
    for name in (*DATE_MODULES, *TIME_MODULES):
        importlib.import_module(name)
    clock = provider.clock
    issues: set[str] = set()
    session_days = sorted({pd.Timestamp(d).date().isoformat() for d in sessions})
    schedule = [d for d in session_days if NYSE.is_session(d)]
    counts: Counter = Counter()
    errors: Counter = Counter()
    sink = logger.add(lambda m: errors.update([m.record["level"].name])
                      if m.record["level"].no >= 40 else None)
    try:
        with clock.patch(), _offline(clock, issues):
            if schedule:
                clock.set(NYSE.session_open(schedule[0]).to_pydatetime())
            store = live.StateStore(path=out / "data" / "state.json")
            portfolio = store.load_or_init(config.INITIAL_EQUITY)
            # dataclass default_factory date.today import anında bağlanır: açıkça kur.
            portfolio.last_session_date = clock.now.date() - timedelta(days=1)
            portfolio.day_trades_window_start = clock.now.date()
            risk_mgr = live.RiskManager(portfolio)
            pmgr = live.PositionManager(portfolio, live.SlippageModel())
            breaker = live.CircuitBreaker()

            def on_close(ce):
                # Canlı döngünün TEK kapanış defteri (C4) doğrudan çağrılır: kopya gövde, defter
                # değişince replay'i sessizce ayrıştırırdı (Tur-77 incelemesi — kopya opsiyon
                # çarpanını atlamıştı). İçindeki Telegram uyarısı worker'da kapalıdır.
                live._apply_close_bookkeeping(store, ce)
                counts["closed_positions"] += 1

            pmgr.add_on_close(on_close)
            for day in schedule:
                opening, closing = NYSE.session_open(day), NYSE.session_close(day)
                clock.set(opening.to_pydatetime())
                portfolio.roll_session_if_new_day(clock.now.date())
                store.settle_matured(clock.now.date())
                duration = int((closing - opening).total_seconds() // 60)
                # Açılış kontrolü dahil; son kapanmış 1m bar için kapanışta da güncelle.
                for minute in range(duration + 1):
                    clock.set((opening + pd.Timedelta(minutes=minute)).to_pydatetime())
                    now = clock.now.replace(tzinfo=None)
                    pmgr.update_all_with_1m_bars(provider, now)
                    pmgr.update_all_with_quotes(provider, now, skip_stocks=True)
                    if minute < duration and minute % scan_every == 0:
                        for symbol in symbols:
                            live.scan_symbol(symbol, provider, risk_mgr, store, breaker, pmgr)
                            counts["scans"] += 1
                counts["simulated_minutes"] += duration
                counts["minute_steps"] += duration + 1
                store.save()
            store.save()  # sıfır seans da açık, determinist çıktı verir
            nlv = portfolio.equity + sum(p.unrealized_pnl for p in portfolio.open_positions)
            open_count = len(portfolio.open_positions)
    finally:
        logger.remove(sink)
    from nihai.signals.signal_engine import HUNI_ASAMALARI

    decisions: Counter = Counter()
    for file in sorted((out / "logs").glob("decision_log_*.txt")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if re.match(r"\d{4}-\d{2}-\d{2} ", line):
                decisions[line.split("|", 2)[1].strip()] += 1
    trades = out / "data" / "trades.csv"
    with trades.open(encoding="utf-8", newline="") if trades.exists() else io.StringIO() as stream:
        trade_rows = sum(1 for _ in csv.DictReader(stream))
    inventory = time_inventory()
    summary = {
        "sessions": len(schedule), "session_dates": schedule,
        "simulated_minutes": counts["simulated_minutes"], "minute_steps": counts["minute_steps"],
        "scans": counts["scans"], "opened_positions": decisions["POSITION_OPEN"],
        "closed_positions": counts["closed_positions"], "remaining_positions": open_count,
        "trade_rows": trade_rows, "final_nlv": round(nlv, 8),
        # Tur-106 (BULGU-106-15): blok kodları karar hunisinin TEK listesinden — elle tutulan küme FILL_SLIPPAGE'ı
        # ve quant guard'larını kaçırıyordu.
        "signals": decisions["SIGNAL"], "warnings": sum(v for k, v in decisions.items()
            if k in HUNI_ASAMALARI or k.endswith("NOTICE") or k in {"OPTION_NO_CONTRACT", "OPTION_RISK_REJECT"}),
        "decision_counts": dict(sorted(decisions.items())), "errors": dict(sorted(errors.items())),
        "missing_history": missing, "inputs": {k: inputs[k] for k in ("copied", "missing")},
        "time_sources": inventory,
        "deviations": [*DEVIATIONS, *sorted(issues),
                       *(f"YAMALANMAYAN_SAAT:{item}" for item in inventory["deviation"])],
    }
    (out / "replay_ozet.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _spawn(out_dir, payload: dict) -> dict:
    out, env, inputs = _prepare(out_dir)
    payload["inputs"] = inputs
    result = subprocess.run([sys.executable, "-B", "-m", "nihai.execution.replay", "--worker", str(out)],
                            input=json.dumps(payload), text=True, encoding="utf-8", env=env,
                            cwd=ROOT, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Replay alt süreç başarısız ({result.returncode}):\n{result.stderr[-8000:]}")
    return json.loads((out / "replay_ozet.json").read_text(encoding="utf-8"))


def run_replay(provider: ReplayProvider, symbols, sessions, *, out_dir,
               scan_every_minutes: int = 1, clock: SimulatedClock | None = None) -> dict:
    """Her çağrı temiz alt süreç; ana süreçteki config/cache/saat değiştirilmez.

    sessions: tarih/datetime/ISO tarih iterable'ı (iki uç değil, günlerin listesi).
    Verilen clock başlangıcı taşınır; ana süreçteki saat nesnesi değiştirilmez.
    """
    if type(scan_every_minutes) is not int or scan_every_minutes < 1:
        raise ValueError("Tarama aralığı pozitif tamsayı olmalı")
    clock = clock or provider.clock
    payload = {"symbols": sorted({_symbol(s) for s in symbols}),
               "sessions": [pd.Timestamp(s).date().isoformat() for s in sessions],
               "scan_every": scan_every_minutes, "clock": clock.now.isoformat(),
               "bars": [{"symbol": s, "interval": i,
                         "frame": df.to_json(orient="split", date_format="iso", double_precision=15)}
                        for (s, i), df in sorted(provider.bars.items())]}
    return _spawn(out_dir, payload)


def _worker(out: Path, payload: dict) -> None:
    if os.getenv("NIHAI_REPLAY_WORKER") != "1":
        raise RuntimeError("Worker yalnız replay başlatıcısından çalıştırılır")
    _install_write_fence(out)
    clock = SimulatedClock(datetime.fromisoformat(payload.get("clock", "2000-01-01T00:00:00-05:00")))
    missing: list[dict] = []
    if "bars" in payload:
        bars = {}
        for item in payload["bars"]:
            frame = pd.read_json(io.StringIO(item["frame"]), orient="split")
            frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
            bars[(item["symbol"], item["interval"])] = frame
    else:
        from nihai.config import CONFIG
        from nihai.data import alpaca_history
        symbols = sorted(set(payload["symbols"]) | {_symbol(CONFIG.regime.benchmark)})
        # Kaynak cache sadece OKUNUR; fetch gerekirse izole cache'e gider.
        if not payload["fetch"]:
            with patch.object(alpaca_history, "HIST_DIR", Path(payload["inputs"]["history_source"])):
                bars, missing = load_history(symbols, payload["start"], payload["end"], required_intervals())
        else:
            bars, missing = load_history(symbols, payload["start"], payload["end"], required_intervals(), fetch=True)
    _run_local(ReplayProvider(bars, clock), payload["symbols"], payload["sessions"], out=out,
               scan_every=payload["scan_every"], inputs=payload["inputs"], missing=missing)


def main(argv: Sequence[str] | None = None) -> None:
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Canlı karar yolunda izole tarihsel replay")
    parser.add_argument("--symbols", default="AAPL,MSFT")
    parser.add_argument("--from", dest="start")
    parser.add_argument("--to", dest="end")
    parser.add_argument("--out")
    parser.add_argument("--scan-every", type=int, default=1)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        _worker(Path(args.worker).resolve(), json.load(sys.stdin))
        return
    if not args.start or not args.end or not args.out or args.scan_every < 1:
        parser.error("--from, --to, --out ve pozitif --scan-every gerekli")
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if end < start:
        parser.error("--to, --from öncesinde olamaz")
    sessions = [d.date().isoformat() for d in pd.date_range(start, end)]
    summary = _spawn(args.out, {"symbols": sorted({_symbol(s) for s in args.symbols.split(",")}),
                               "sessions": sessions, "start": args.start, "end": args.end,
                               "scan_every": args.scan_every, "fetch": args.fetch})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    main()
