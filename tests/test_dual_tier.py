"""İki-katman testleri — TierScanner yaşam döngüsü + PORTFOLIO_LOCK yarış koruması."""
import threading
import time

from nihai.execution.dual_tier import PORTFOLIO_LOCK, TierScanner


def test_lock_is_reentrant():
    # RLock: aynı thread iki kez alabilmeli (scan_symbol içinde iç içe çağrı güvenli)
    with PORTFOLIO_LOCK:
        with PORTFOLIO_LOCK:
            assert True


def test_tier_scans_partition():
    seen = []
    lock = threading.Lock()
    def scan_one(sym, prov):
        with lock:
            seen.append(sym)
    t = TierScanner(name="t", provider="P", get_symbols=lambda: ["A", "B", "C"],
                    scan_one=scan_one, throttle=0, cycle_gap=0.01)
    t.start()
    time.sleep(0.2)
    t.stop()
    t.join(2)
    assert not t.is_alive()
    assert set(seen) >= {"A", "B", "C"}     # en az bir tam tur
    assert t.cycles >= 1


def test_tier_empty_symbols_no_crash():
    t = TierScanner(name="empty", provider=None, get_symbols=lambda: [],
                    scan_one=lambda s, p: None, cycle_gap=0.01)
    t.start()
    time.sleep(0.1)
    t.stop()
    t.join(2)
    assert not t.is_alive()
    assert t.cycles == 0   # boş partisyon → tur sayılmaz


def test_tier_exception_isolated():
    # scan_one patlasa bile thread çökmemeli, sonraki sembollere devam etmeli
    ok = []
    def scan_one(sym, prov):
        if sym == "BAD":
            raise ValueError("boom")
        ok.append(sym)
    t = TierScanner(name="t", provider=None, get_symbols=lambda: ["BAD", "GOOD"],
                    scan_one=scan_one, throttle=0, cycle_gap=0.01)
    t.start()
    time.sleep(0.15)
    t.stop()
    t.join(2)
    assert "GOOD" in ok          # BAD patladı ama GOOD işlendi
    assert not t.is_alive()


def test_parallel_lock_enforces_capacity():
    # İki katman aynı paylaşılan "max 3 pozisyon"u zorlasın → asla aşmamalı (yarış yok)
    state = {"positions": [], "max": 3, "violations": 0}
    def scan_one(sym, prov):
        with PORTFOLIO_LOCK:
            if len(state["positions"]) < state["max"]:
                time.sleep(0.0003)   # context-switch penceresi (kilit yoksa yarış)
                state["positions"].append(sym)
                if len(state["positions"]) > state["max"]:
                    state["violations"] += 1
    syms = [f"S{i}" for i in range(150)]
    a = TierScanner(name="A", provider=None, get_symbols=lambda: syms,
                    scan_one=scan_one, throttle=0, cycle_gap=0.01)
    b = TierScanner(name="B", provider=None, get_symbols=lambda: syms,
                    scan_one=scan_one, throttle=0, cycle_gap=0.01)
    a.start(); b.start()
    time.sleep(0.5)
    a.stop(); b.stop(); a.join(2); b.join(2)
    assert state["violations"] == 0           # kilit yarışı engelledi
    assert len(state["positions"]) == 3       # kapasite tam doldu, aşılmadı
