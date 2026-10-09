"""
gate.py — Backtest → Paper → Canlı yetkilendirme kapısı.

PDF Referansları:
- Madde 52 : Dağıtım kapısı (kâğıt üzerinde mükemmel ama canlıda bozuk olanı
             gerçek parayla test etmeyi engeller)

Kullanım:
    python gate.py --check logs/backtest_AAPL_1d_20260531_230452.json
    python gate.py --parity backtest.json paper.json

Mantık:
- Backtest sonucu yüksek standartları (min işlem, expectancy>0, MDD<25, MC ruin<5%)
  geçerse "BACKTEST_OK"
- Paper sonucu backtest'le karşılaştırılır (expectancy parity ±%50)
- İki kapı da geçerse "LIVE_READY" — aksi halde "BLOCKED" gerekçesiyle
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from nihai.console import ensure_utf8_stdout

# Eşikler (config'e taşınabilir)
MIN_TRADES = 30
MIN_WIN_RATE = 0.40
MIN_EXPECTANCY = 0.01           # $ / işlem (sembolden bağımsız pozitif)
MAX_DRAWDOWN_PCT = 25.0
MAX_MC_RUIN_PROB = 0.05
PARITY_TOLERANCE_PCT = 0.50      # paper expectancy backtest'in ±%50'sinde olmalı


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class GateReport:
    label: str                  # BACKTEST | PAPER | PARITY | OVERALL
    passed: bool
    checks: list[GateCheck]
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "label": self.label, "passed": self.passed,
            "summary": self.summary,
            "checks": [asdict(c) for c in self.checks],
        }


def evaluate_backtest_dict(r: dict) -> GateReport:
    """Backtest JSON sözlüğü (BacktestResult.asdict)'ten kapı değerlendirmesi."""
    checks: list[GateCheck] = []
    checks.append(GateCheck(
        "MIN_TRADES",
        r.get("trades_count", 0) >= MIN_TRADES,
        f"{r.get('trades_count', 0)} >= {MIN_TRADES}",
    ))
    checks.append(GateCheck(
        "MIN_WIN_RATE",
        r.get("win_rate", 0) >= MIN_WIN_RATE,
        f"{r.get('win_rate', 0)*100:.1f}% >= {MIN_WIN_RATE*100:.0f}%",
    ))
    checks.append(GateCheck(
        "POSITIVE_EXPECTANCY",
        r.get("expectancy", 0) >= MIN_EXPECTANCY,
        f"E={r.get('expectancy', 0):+.2f}",
    ))
    checks.append(GateCheck(
        "MAX_DRAWDOWN",
        r.get("max_drawdown_pct", 999) <= MAX_DRAWDOWN_PCT,
        f"MDD={r.get('max_drawdown_pct', 0):.1f}% <= {MAX_DRAWDOWN_PCT}%",
    ))
    mc = r.get("monte_carlo") or {}
    if mc:
        checks.append(GateCheck(
            "MC_RUIN_LOW",
            mc.get("ruin_probability", 1) <= MAX_MC_RUIN_PROB,
            f"ruin={mc.get('ruin_probability', 0)*100:.2f}% <= {MAX_MC_RUIN_PROB*100:.0f}%",
        ))
    all_passed = all(c.passed for c in checks)
    return GateReport(
        label="BACKTEST", passed=all_passed, checks=checks,
        summary=f"{r.get('symbol', '?')} {r.get('interval', '?')}: "
                f"{'PASS' if all_passed else 'FAIL'}",
    )


def evaluate_parity(backtest: dict, paper: dict) -> GateReport:
    """Paper sonucu backtest'le karşılaştır — expectancy ±%X tolerans."""
    bt_e = backtest.get("expectancy", 0)
    pp_e = paper.get("expectancy", 0)
    if abs(bt_e) < 0.01:
        parity_ok = abs(pp_e) < 1.0
        detail = f"backtest E~0 ({bt_e:+.2f}), paper E={pp_e:+.2f}"
    else:
        ratio = (pp_e - bt_e) / abs(bt_e)
        parity_ok = abs(ratio) <= PARITY_TOLERANCE_PCT
        detail = f"backtest E={bt_e:+.2f}, paper E={pp_e:+.2f}, sapma {ratio*100:+.1f}%"
    checks = [GateCheck("EXPECTANCY_PARITY", parity_ok, detail)]
    # WR parity (sertif eşiği yumuşak)
    bt_wr, pp_wr = backtest.get("win_rate", 0), paper.get("win_rate", 0)
    wr_diff = abs(pp_wr - bt_wr)
    checks.append(GateCheck(
        "WIN_RATE_PARITY",
        wr_diff <= 0.15,
        f"WR bt={bt_wr*100:.1f}% paper={pp_wr*100:.1f}% diff={wr_diff*100:.1f}pp",
    ))
    all_passed = all(c.passed for c in checks)
    return GateReport(
        label="PARITY", passed=all_passed, checks=checks,
        summary=f"{'PASS' if all_passed else 'FAIL'}",
    )


def print_report(rep: GateReport) -> None:
    print(f"\n{'='*64}")
    print(f" GATE: {rep.label} - {'PASS' if rep.passed else 'BLOCKED'}")
    print(f" {rep.summary}")
    print("=" * 64)
    for c in rep.checks:
        mark = "[OK]" if c.passed else "[X] "
        print(f"  {mark} {c.name:<25} {c.detail}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Madde 52 — Backtest/Paper/Live kapısı")
    ap.add_argument("--check", help="Backtest JSON dosyasını kapıya sok")
    ap.add_argument("--parity", nargs=2, metavar=("BACKTEST", "PAPER"),
                    help="Backtest ve paper JSON'ları karşılaştır")
    args = ap.parse_args()

    if args.check:
        data = json.loads(Path(args.check).read_text(encoding="utf-8"))
        rep = evaluate_backtest_dict(data)
        print_report(rep)
        return

    if args.parity:
        bt = json.loads(Path(args.parity[0]).read_text(encoding="utf-8"))
        pp = json.loads(Path(args.parity[1]).read_text(encoding="utf-8"))
        bt_rep = evaluate_backtest_dict(bt)
        print_report(bt_rep)
        par_rep = evaluate_parity(bt, pp)
        print_report(par_rep)
        live_ready = bt_rep.passed and par_rep.passed
        print(f"{'='*64}\n  LIVE_READY: {live_ready}\n{'='*64}\n")
        return

    ap.print_help()


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    main()
