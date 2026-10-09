"""A1 — post-mortem metrik/render testleri (saf fonksiyonlar, dosya gerektirmez)."""
import math
import statistics

import pytest

from research.postmortem import (
    _breakdown,
    _parse_period,
    compute_metrics,
    render_markdown,
)


def _trade(pnl, r="", strat="S1", sector="Tech", trig="TARGET", sym="AAPL"):
    return {
        "symbol": sym, "strategy": strat, "sector": sector,
        "realized_pnl": str(pnl), "r_multiple": str(r), "exit_trigger": trig,
    }


def test_compute_metrics_basic():
    trades = [_trade(100, 1.0), _trade(-50, -1.0), _trade(200, 2.0)]
    m = compute_metrics(trades, equity=[1000, 1100, 1050, 1250])
    assert m["n"] == 3
    assert m["wins"] == 2 and m["losses"] == 1
    assert round(m["win_rate"], 1) == 66.7
    assert round(m["total_pnl"], 2) == 250.0
    # PF = kazanç(300) / kayıp(50) = 6.0
    assert round(m["profit_factor"], 2) == 6.0
    assert round(m["expectancy"], 2) == round(250 / 3, 2)
    assert m["best"]["realized_pnl"] == "200"
    assert m["worst"]["realized_pnl"] == "-50"


def test_profit_factor_infinite_when_no_losses():
    m = compute_metrics([_trade(10), _trade(20)], equity=[])
    assert m["profit_factor"] == math.inf


def test_empty_trades_safe():
    m = compute_metrics([], equity=[])
    assert m["n"] == 0 and m["profit_factor"] == 0.0 and m["best"] is None


def test_breakdown_sorted_desc():
    trades = [_trade(100, strat="A"), _trade(-30, strat="B"), _trade(50, strat="A")]
    bd = _breakdown(trades, "strategy")
    assert bd[0] == ("A", 2, 150.0)
    assert bd[1] == ("B", 1, -30.0)


def test_parse_period():
    assert _parse_period("all") is None
    assert _parse_period(None) is None
    assert _parse_period("7d").days == 7
    assert _parse_period("2w").days == 14


def test_render_markdown_contains_summary():
    md = render_markdown([_trade(100, 1.0), _trade(-50, -1.0)], [1000, 900], "7d")
    assert "# Post-Mortem Raporu" in md
    assert "Profit Factor" in md
    assert "Strateji kırılımı" in md


# ---------------------------------------------------------------------------
# B6 — sharpe_trade = ort/std (boyutsuz); t_stat = ort/std·√n (anlamlılık) AYRI
# ---------------------------------------------------------------------------
def test_sharpe_is_per_trade_and_tstat_separate():
    """sharpe_trade artık *sqrt(n) İÇERMEZ (o bir t-istatistiğiydi, Sharpe değil)."""
    pnls = [100.0, -50.0, 200.0, -30.0, 80.0]
    m = compute_metrics([_trade(p) for p in pnls], equity=[])
    mean, sd = statistics.mean(pnls), statistics.pstdev(pnls)
    assert m["sharpe_trade"] == pytest.approx(mean / sd)
    assert m["t_stat"] == pytest.approx(mean / sd * math.sqrt(len(pnls)))


def test_sharpe_invariant_to_n_but_tstat_grows():
    """BOYUTSUZLUK değişmezi: aynı dağılımı 4× çoğalt → sharpe SABİT (n'den bağımsız),
    t_stat BÜYÜR (√n ile). Eski hatalı 'sharpe' n ile şişerdi = yanıltıcı."""
    pnls = [100.0, -50.0, 200.0, -30.0, 80.0]
    trades = [_trade(p) for p in pnls]
    m1 = compute_metrics(trades, equity=[])
    m4 = compute_metrics(trades * 4, equity=[])
    assert m4["sharpe_trade"] == pytest.approx(m1["sharpe_trade"])   # n'den bağımsız
    assert m4["t_stat"] > m1["t_stat"] * 1.5                         # √4 = 2× büyür


def test_render_markdown_has_sharpe_and_tstat_rows():
    md = render_markdown([_trade(100, 1.0), _trade(-50, -1.0)], [1000, 900], "7d")
    assert "Sharpe (işlem-bazlı" in md
    assert "t-istatistiği" in md
