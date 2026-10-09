"""gate.py go/no-go karar mantığı testleri (Madde 52 — dağıtım kapısı).

Bu fonksiyonlar bir stratejinin gerçek parayla canlıya geçip geçmeyeceğini belirler;
eşik sınırları (>=, <=), eksik anahtar varsayılanları ve parity'nin E~0 özel dalı
yanlış olursa kötü strateji canlıya geçebilir. Eşik SABİTLERİNE göre relatif test
edilir ki ileride eşik ayarı testi sessizce geçersiz bırakmasın."""
import nihai.execution.gate as gate


def _passing_backtest() -> dict:
    """Tüm kapıları rahatça geçen referans sözlük (sınır testleri bundan saptırır)."""
    return {
        "symbol": "AAPL", "interval": "1d",
        "trades_count": gate.MIN_TRADES + 10,
        "win_rate": gate.MIN_WIN_RATE + 0.10,
        "expectancy": gate.MIN_EXPECTANCY + 0.50,
        "max_drawdown_pct": gate.MAX_DRAWDOWN_PCT - 5.0,
    }


def test_clean_backtest_passes_all_checks():
    rep = gate.evaluate_backtest_dict(_passing_backtest())
    assert rep.label == "BACKTEST"
    assert rep.passed is True
    assert all(c.passed for c in rep.checks)


def test_below_min_trades_fails():
    r = _passing_backtest()
    r["trades_count"] = gate.MIN_TRADES - 1
    rep = gate.evaluate_backtest_dict(r)
    assert rep.passed is False
    assert any(c.name == "MIN_TRADES" and not c.passed for c in rep.checks)


def test_min_trades_boundary_is_inclusive():
    # >= eşik: tam eşikte GEÇMELİ (off-by-one regresyon koruması)
    r = _passing_backtest()
    r["trades_count"] = gate.MIN_TRADES
    rep = gate.evaluate_backtest_dict(r)
    assert next(c for c in rep.checks if c.name == "MIN_TRADES").passed is True


def test_drawdown_boundary_is_inclusive():
    # <= eşik: tam eşikte GEÇMELİ
    r = _passing_backtest()
    r["max_drawdown_pct"] = gate.MAX_DRAWDOWN_PCT
    rep = gate.evaluate_backtest_dict(r)
    assert next(c for c in rep.checks if c.name == "MAX_DRAWDOWN").passed is True


def test_missing_keys_use_safe_defaults_and_block():
    # Boş sözlük çökmemeli; eksik metrikler güvenli tarafta (BLOCK) düşmeli
    rep = gate.evaluate_backtest_dict({})
    assert rep.passed is False
    names = {c.name for c in rep.checks}
    assert {"MIN_TRADES", "MIN_WIN_RATE", "POSITIVE_EXPECTANCY", "MAX_DRAWDOWN"} <= names


def test_monte_carlo_absent_omits_ruin_check():
    rep = gate.evaluate_backtest_dict(_passing_backtest())
    assert all(c.name != "MC_RUIN_LOW" for c in rep.checks)
    assert len(rep.checks) == 4


def test_monte_carlo_high_ruin_blocks():
    r = _passing_backtest()
    r["monte_carlo"] = {"ruin_probability": gate.MAX_MC_RUIN_PROB + 0.10}
    rep = gate.evaluate_backtest_dict(r)
    assert rep.passed is False
    assert any(c.name == "MC_RUIN_LOW" and not c.passed for c in rep.checks)


def test_monte_carlo_low_ruin_passes():
    r = _passing_backtest()
    r["monte_carlo"] = {"ruin_probability": 0.0}
    rep = gate.evaluate_backtest_dict(r)
    assert rep.passed is True
    assert any(c.name == "MC_RUIN_LOW" and c.passed for c in rep.checks)


def test_to_dict_shape():
    rep = gate.evaluate_backtest_dict(_passing_backtest())
    d = rep.to_dict()
    assert d["label"] == "BACKTEST" and d["passed"] is True
    assert isinstance(d["checks"], list) and d["checks"]
    assert {"name", "passed", "detail"} <= set(d["checks"][0])


# ---------------- evaluate_parity ----------------

def test_parity_near_zero_backtest_small_paper_passes():
    # backtest E~0 dalı: paper da küçükse parity OK
    rep = gate.evaluate_parity({"expectancy": 0.0, "win_rate": 0.5},
                               {"expectancy": 0.5, "win_rate": 0.5})
    assert next(c for c in rep.checks if c.name == "EXPECTANCY_PARITY").passed is True


def test_parity_near_zero_backtest_large_paper_fails():
    rep = gate.evaluate_parity({"expectancy": 0.0, "win_rate": 0.5},
                               {"expectancy": 2.0, "win_rate": 0.5})
    assert next(c for c in rep.checks if c.name == "EXPECTANCY_PARITY").passed is False


def test_parity_within_tolerance_passes():
    # paper = backtest * (1 + tol/2) → tolerans içinde
    bt = 1.00
    pp = bt * (1 + gate.PARITY_TOLERANCE_PCT / 2)
    rep = gate.evaluate_parity({"expectancy": bt, "win_rate": 0.5},
                               {"expectancy": pp, "win_rate": 0.5})
    assert rep.passed is True


def test_parity_beyond_tolerance_fails():
    bt = 1.00
    pp = bt * (1 + gate.PARITY_TOLERANCE_PCT * 2)  # toleransın çok ötesi
    rep = gate.evaluate_parity({"expectancy": bt, "win_rate": 0.5},
                               {"expectancy": pp, "win_rate": 0.5})
    assert rep.passed is False
    assert any(c.name == "EXPECTANCY_PARITY" and not c.passed for c in rep.checks)


def test_parity_win_rate_drift_blocks():
    # Expectancy uyumlu ama WR sapması >15pp → BLOCK
    rep = gate.evaluate_parity({"expectancy": 1.0, "win_rate": 0.60},
                               {"expectancy": 1.0, "win_rate": 0.30})
    assert rep.passed is False
    assert any(c.name == "WIN_RATE_PARITY" and not c.passed for c in rep.checks)
