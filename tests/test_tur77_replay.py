"""Tur-77: ağsız sentetik canlı replay, kapanış sınırı ve alt süreç izolasyonu."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from random import Random
from unittest.mock import patch

import pandas as pd
import pytest

from nihai.data import market_calendar as calendar
from nihai.execution import replay


def bars(index, close=100.0):
    index = pd.DatetimeIndex(index)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 10000.0}, index=index)


def test_clock_calendar_restore_and_exception():
    original = calendar.now_et
    clock = replay.SimulatedClock(datetime(2026, 9, 8, 10, 0))
    with clock.patch():
        assert calendar.now_et_naive() == datetime(2026, 9, 8, 10, 0)
        assert calendar.is_market_open()
        assert calendar.can_enter_new_position() == (True, "OK")
        assert calendar.session_remaining() == timedelta(hours=6)
        clock.advance(420)
        assert not calendar.is_market_open()
    assert calendar.now_et is original
    with pytest.raises(RuntimeError), clock.patch():
        raise RuntimeError("istisnada da geri yükle")
    assert calendar.now_et is original
    clock.set(datetime(2026, 9, 8, 14, tzinfo=UTC))
    assert clock.now.hour == 10


def test_clock_module_dates_and_time_restore():
    from nihai.execution import state
    from nihai.risk import risk_manager, safety
    from nihai.signals import model_promotion

    originals = state.datetime, risk_manager.date, safety.datetime, model_promotion.time
    clock = replay.SimulatedClock(datetime(2026, 9, 8, 10))
    with clock.patch():
        assert state.datetime.now() == datetime(2026, 9, 8, 10)
        assert risk_manager.date.today() == clock.now.date()
        assert safety.datetime.now() == datetime(2026, 9, 8, 10)
        assert model_promotion.time.time() == clock.now.timestamp()
        assert isinstance(datetime(2026, 9, 8), state.datetime)
        assert state._serialize(datetime(2026, 9, 8)) == "2026-09-08T00:00:00"
    assert originals == (state.datetime, risk_manager.date, safety.datetime, model_promotion.time)


def test_lookahead_200_random_times_and_exact_boundaries():
    rng = Random(77)
    start = datetime(2026, 8, 1)
    clock = replay.SimulatedClock(start.replace(tzinfo=UTC))
    data = {("AAA", interval): bars(pd.date_range(start, periods=400, freq=pd.Timedelta(delta)))
            for interval, delta in replay.DURATIONS.items()}
    provider = replay.ReplayProvider(data, clock)
    for _ in range(200):
        clock.set((start + timedelta(seconds=rng.randrange(50 * 86400))).replace(tzinfo=UTC))
        now = clock.now.astimezone(UTC).replace(tzinfo=None)
        for interval, delta in replay.DURATIONS.items():
            result = provider.get_bars("AAA", interval, 37)
            assert len(result) <= 37
            assert all(result.index + delta <= now)
    for interval, delta in replay.DURATIONS.items():
        clock.set((start + delta).replace(tzinfo=UTC))
        result = provider.get_bars("AAA", interval)
        assert list(result.index) == [pd.Timestamp(start)]
        clock.advance(-1 / 60)
        assert provider.get_bars("AAA", interval).empty
    clock.set((start + timedelta(days=600)).replace(tzinfo=UTC))
    returned = provider.get_bars("AAA", "1m")
    returned.iloc[0, 0] = 999
    assert provider.bars[("AAA", "1m")].open.max() == 100


def test_quote_and_missing_interval():
    clock = replay.SimulatedClock(datetime(2026, 9, 8, 10))
    frame = bars(["2026-09-08 13:58", "2026-09-08 13:59", "2026-09-08 14:00"])
    frame.loc[pd.Timestamp("2026-09-08 13:59"), ["close", "high"]] = 101.0
    provider = replay.ReplayProvider({("AAA", "1m"): frame}, clock)
    quote = provider.get_quote("AAA")
    assert quote is not None
    assert quote.bid == quote.ask == quote.last == 101.0
    assert quote.spread_measured is False
    assert quote.timestamp == clock.now
    assert provider.get_quote("BBB") is None
    assert provider.get_bars("AAA", "1h").empty
    assert provider.get_bars("AAA", "1m", 0).empty
    assert provider.get_option_chain("AAA")["calls"].empty


def test_provider_rejects_invalid_input():
    clock = replay.SimulatedClock(datetime(2026, 9, 8))
    for index in (pd.date_range("2026-09-08", periods=2, tz="UTC"),
                  pd.DatetimeIndex(["2026-09-08", "2026-09-08"]), pd.DatetimeIndex([pd.NaT])):
        with pytest.raises(ValueError):
            replay.ReplayProvider({("AAA", "1m"): bars(index)}, clock)
    with pytest.raises(ValueError):
        replay.ReplayProvider({("../state", "1m"): bars(["2026-09-08"])}, clock)


def synthetic_provider():
    clock = replay.SimulatedClock(datetime(2026, 9, 8, 9, 30))
    minute = bars(pd.date_range("2026-09-08 13:00", "2026-09-08 20:00", freq="min"))
    minute.loc[minute.index >= "2026-09-08 14:01", "high"] = 105.0
    hour = bars(pd.date_range("2026-08-01", "2026-09-08 20:00", freq="h"))
    return replay.ReplayProvider({(symbol, interval): frame
                                 for symbol in ("AAA", "BBB")
                                 for interval, frame in (("1m", minute), ("1h", hour))}, clock)


def hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
            for p in (replay.ROOT / "data" / "state.json", replay.ROOT / "data" / "trades.csv")}


def _uretim_degismedi(before) -> None:
    """Tur-88: canlı tarayıcı state.json'u ~60 sn'de bir yazar; test o pencereye denk gelince kırmızı veriyordu.
    Sızıntının izi replay'in SİMÜLE günüdür (2026-09-08). Tur-106: kardeş pozitif-kontrol testi bu toleransı
    almamıştı ve seans saatinde paket kırmızısı veriyordu → iki test aynı yardımcıyı kullanır."""
    for path, digest in hashes().items():
        if digest != before[path] and path.endswith("state.json"):
            assert '"2026-09-08T' not in Path(path).read_text(encoding="utf-8")
        else:
            assert digest == before[path]


def test_real_scan_deterministic_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HTF_INTERVAL", "1h")
    monkeypatch.setenv("ACTIVE_STRATEGY", "donchian")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "must-not-send")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "must-not-send")
    before = hashes()
    provider = synthetic_provider()
    summaries = [replay.run_replay(provider, ["AAA", "BBB"], ["2026-09-08"], out_dir=tmp_path / name,
                                    scan_every_minutes=30) for name in ("one", "two")]
    assert summaries[0] == summaries[1]  # Hiçbir özet alanı dışlanmıyor.
    summary = summaries[0]
    assert summary["sessions"] == 1
    assert summary["simulated_minutes"] == 390 and summary["minute_steps"] == 391
    assert summary["scans"] == 26
    assert summary["decision_counts"]["SESSION_GUARD"] == 2
    assert summary["decision_counts"]["SKIP"] > 0
    assert not summary["errors"]
    assert summary["final_nlv"] > 0
    assert all("NETWORK_ATTEMPT" not in issue for issue in summary["deviations"])
    for name in ("one", "two"):
        out = tmp_path / name
        state = json.loads((out / "data" / "state.json").read_text(encoding="utf-8"))
        assert state["saved_at"] == "2026-09-08T16:00:00"
        assert (out / "logs" / "decision_log_2026_09_08.txt").is_file()
        assert all(p.resolve().is_relative_to(out.resolve()) for p in out.rglob("*") if p.is_file())
    _uretim_degismedi(before)


def test_live_positive_control_open_close_and_order(tmp_path, monkeypatch):
    """Yalnız sinyal detektörü sentetik; scan, çapa, risk, PM ve defterler GERÇEK."""
    monkeypatch.setenv("HTF_INTERVAL", "1h")
    monkeypatch.setenv("AI_LIVE_MODE", "off")
    monkeypatch.setenv("SHADOW_BRIDGE", "off")
    monkeypatch.setenv("PROB_SIZING", "off")
    before = hashes()
    out, env, inputs = replay._prepare(tmp_path / "positive")
    # Test alt süreci de üretimdeki yazma çitini kullanır; ortam kalıtımı test guard'ını gerektirmez.
    code = '''
import json, sys
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from nihai.execution import replay
out = Path(sys.argv[1])
replay._install_write_fence(out)
from nihai.execution import main as live
from nihai.config import INITIAL_EQUITY
from nihai.signals.strategy import Signal, SignalAction
clock = replay.SimulatedClock(replay.datetime(2026, 9, 8, 9, 30))
index = pd.date_range("2026-09-01", "2026-09-08 20:00", freq="min")
frame = pd.DataFrame(dict(open=100., high=100.2, low=99.8, close=100., volume=10000.), index=index)
frame.loc[frame.index >= "2026-09-08 14:01", "high"] = 105.
provider = replay.ReplayProvider({(s,i): frame for s in ("AAA","BBB") for i in ("1m","1h")}, clock)
calls = []
original_scan = live.scan_symbol
original_bars = live.PositionManager.update_all_with_1m_bars
original_quotes = live.PositionManager.update_all_with_quotes
def signal(symbol, df, **kwargs):
    return Signal(SignalAction.LONG, symbol, 100., 98., 104., ["POSITIVE_CONTROL"], 3., 2.)
def scan(*args, **kwargs):
    assert calls[-1] in ("quotes", "scan")
    calls.append("scan")
    return original_scan(*args, **kwargs)
def update_bars(*args, **kwargs):
    calls.append("bars")
    return original_bars(*args, **kwargs)
def update_quotes(*args, **kwargs):
    assert calls[-1] == "bars" and kwargs["skip_stocks"] is True
    calls.append("quotes")
    return original_quotes(*args, **kwargs)
with patch.object(live, "generate_signal", signal), patch.object(live, "scan_symbol", scan), \\
     patch.object(live.PositionManager, "update_all_with_1m_bars", update_bars), \\
     patch.object(live.PositionManager, "update_all_with_quotes", update_quotes):
    result = replay._run_local(provider, ["AAA", "BBB"], ["2026-09-08"], out=out,
        scan_every=30, inputs=json.loads(sys.stdin.read()), missing=[])
assert calls.count("bars") == calls.count("quotes") == 391
assert result["opened_positions"] >= 2, result
assert result["closed_positions"] == result["trade_rows"] >= 2, result
assert not result["errors"], result
assert result["final_nlv"] > INITIAL_EQUITY, result
assert all("NETWORK_ATTEMPT" not in x for x in result["deviations"])
try:
    (replay.ROOT / "data" / "REPLAY_MUST_NOT_WRITE.txt").write_text("forbidden")
except PermissionError:
    pass
else:
    raise AssertionError("write fence missing")
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, str(out)], input=json.dumps(inputs),
                            env=env, cwd=replay.ROOT, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr[-8000:]
    _uretim_degismedi(before)


@pytest.mark.parametrize("sessions, count, minutes", [
    (["2026-09-05", "2026-09-06", "2026-09-07"], 0, 0),  # Labor Day
    (["2026-11-27"], 1, 210),  # Thanksgiving ertesi yarım gün
])
def test_holiday_and_half_day(tmp_path, sessions, count, minutes):
    provider = synthetic_provider()
    result = replay.run_replay(provider, [], sessions, out_dir=tmp_path / "calendar", scan_every_minutes=60)
    assert result["sessions"] == count and result["simulated_minutes"] == minutes
    assert result["scans"] == 0


def test_loader_cache_only_and_warmup(tmp_path, monkeypatch):
    from nihai.data import alpaca_history

    monkeypatch.setattr(alpaca_history, "HIST_DIR", tmp_path)
    monkeypatch.setattr(alpaca_history, "_fetch", lambda *a, **kw: pytest.fail("ağ yasak"))
    frame = bars(pd.date_range("2026-09-08 13:30", periods=3, freq="min"))
    frame.to_csv(alpaca_history._cache_path("AAA", "1m"))
    loaded, missing = replay.load_history(["AAA"], "2026-09-08", "2026-09-08", ["1m", "1d"])
    assert len(loaded[("AAA", "1m")]) == 3
    assert loaded[("AAA", "1d")].empty
    assert {entry["reason"] for entry in missing} == {"WARMUP_SHORT", "SESSION_GAPS", "CACHE_EMPTY_OR_MISSING"}
    assert len(list(tmp_path.iterdir())) == 1


def test_cli_missing_cache_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_HIST_DIR", str(tmp_path / "absent_cache"))
    result = subprocess.run([sys.executable, "-B", "-m", "nihai.execution.replay", "--symbols", "AAA",
                             "--from", "2026-09-08", "--to", "2026-09-08", "--scan-every", "120",
                             "--out", str(tmp_path / "cli")],
                            env=dict(os.environ), cwd=replay.ROOT, capture_output=True, text=True,
                            encoding="utf-8")
    assert result.returncode == 0, result.stderr[-8000:]
    summary = json.loads(result.stdout)
    assert summary["sessions"] == 1 and summary["scans"] == 4
    assert summary["missing_history"]
    assert not summary["errors"]
    assert summary["decision_counts"]["DATA_EMPTY"] == 3
    assert not (tmp_path / "absent_cache").exists()


def test_reject_unsafe_output_and_existing_output(tmp_path):
    with pytest.raises(ValueError):
        replay._prepare(replay.ROOT / "data" / "replay")
    (tmp_path / "existing").mkdir()
    (tmp_path / "existing" / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError):
        replay._prepare(tmp_path / "existing")


def test_snapshot_copy_and_dotenv_empty_credentials(tmp_path, monkeypatch):
    from dotenv import load_dotenv

    source = tmp_path / "source"
    source.mkdir()
    (source / "meta_model.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("NIHAI_DATA_DIR", str(source))
    out, env, inputs = replay._prepare(tmp_path / "copied")
    assert "meta_model.json" in inputs["copied"]
    assert (out / "data" / "meta_model.json").read_bytes() == (source / "meta_model.json").read_bytes()
    assert env["TELEGRAM_BOT_TOKEN"] == env["TELEGRAM_CHAT_ID"] == ""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    dotenv = tmp_path / "env"
    dotenv.write_text("TELEGRAM_BOT_TOKEN=not-empty\n", encoding="utf-8")
    load_dotenv(dotenv)
    assert os.environ["TELEGRAM_BOT_TOKEN"] == ""


def test_time_inventory_contains_events_and_unpatched_sources():
    inventory = replay.time_inventory()
    assert sum("event_calendar:" in item for item in inventory["patched"]) == 13
    assert any("main:" in item and "time_mod.time" in item for item in inventory["patched"])
    assert any("alpaca_history:" in item for item in inventory["deviation"])
