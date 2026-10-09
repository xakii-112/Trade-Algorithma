"""
ablation.py — Her stratejik kaldıracı İZOLE ölçer (hangisi gerçekten yardımcı?).

Barları sembol başına BİR KEZ indirir, sonra bellekteki df'yi tüm config
kombinasyonlarına karşı yeniden backtest eder (hızlı). Frozen CONFIG alanları
object.__setattr__ ile geçici override edilir; ENSEMBLE_MODE config modülünde
monkeypatch'lenir (detect_setup her çağrıda oradan okur).
"""
from __future__ import annotations

import statistics
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import nihai.config as config
from nihai.config import CONFIG
from nihai.execution.backtest import backtest_symbol, load_bars
from nihai.universe import BASKET

INTERVAL = "1h"
PERIOD = "1y"


def set_cfg(*, ensemble: str, quality: float, trend_gate: bool, adx_gate: bool):
    config.ENSEMBLE_MODE = ensemble
    object.__setattr__(CONFIG.strategy, "min_signal_quality_score", quality)
    object.__setattr__(CONFIG.strategy, "enable_trend_alignment_gate", trend_gate)
    object.__setattr__(CONFIG.strategy, "enable_adx_floor_gate", adx_gate)


CONFIGS = {
    "1_legacy(BEFORE)":   dict(ensemble="first_trigger", quality=1.0, trend_gate=False, adx_gate=False),
    "2_confluence_only":  dict(ensemble="confluence",    quality=1.0, trend_gate=False, adx_gate=False),
    "3_trendgate_only":   dict(ensemble="first_trigger", quality=1.0, trend_gate=True,  adx_gate=False),
    "4_adxgate_only":     dict(ensemble="first_trigger", quality=1.0, trend_gate=False, adx_gate=True),
    "5_quality2.5_only":  dict(ensemble="first_trigger", quality=2.5, trend_gate=False, adx_gate=False),
    "6_gates_both":       dict(ensemble="first_trigger", quality=1.0, trend_gate=True,  adx_gate=True),
    "7_all(AFTER)":       dict(ensemble="confluence",    quality=2.5, trend_gate=True,  adx_gate=True),
    "8_conf+trendgate":   dict(ensemble="confluence",    quality=1.0, trend_gate=True,  adx_gate=False),
}


def agg(results: dict[str, list[float]]):
    pnls_all = [p for pnls in results.values() for p in pnls]
    total_trades = len(pnls_all)
    weighted_e = statistics.mean(pnls_all) if pnls_all else 0.0
    per_symbol_e = [statistics.mean(p) for p in results.values() if p]
    profitable = sum(1 for p in results.values() if p and statistics.mean(p) > 0)
    tested = sum(1 for p in results.values() if p)
    wins = sum(1 for p in pnls_all if p > 0)
    wr = 100 * wins / total_trades if total_trades else 0.0
    return {
        "trades": total_trades,
        "weighted_E": round(weighted_e, 2),
        "mean_sym_E": round(statistics.mean(per_symbol_e), 2) if per_symbol_e else 0.0,
        "pct_profit": round(100 * profitable / tested, 1) if tested else 0.0,
        "win_rate": round(wr, 1),
    }


def main():
    print(f"Barlar indiriliyor ({len(BASKET)} sembol, {INTERVAL}/{PERIOD})...")
    bars = {}
    for s in BASKET:
        df = load_bars(s, INTERVAL, period=PERIOD)
        if not df.empty:
            bars[s] = df
    print(f"  {len(bars)} sembol hazır.\n")
    if not bars:
        raise ValueError("Ablasyon için bar alınamadı; sonuç ölçülmedi.")

    rows = {}
    for name, cfg in CONFIGS.items():
        set_cfg(**cfg)
        results = {}
        for s, df in bars.items():
            r = backtest_symbol(s, interval=INTERVAL, df=df, strategy_mode="ensemble")
            results[s] = r.trade_pnls
        rows[name] = agg(results)
        a = rows[name]
        print(f"{name:<20} trades={a['trades']:>4}  weighted_E={a['weighted_E']:>9}  "
              f"mean_sym_E={a['mean_sym_E']:>9}  pct_profit={a['pct_profit']:>5}  WR={a['win_rate']:>5}")

    print("\n" + "=" * 78)
    best = max(rows.items(), key=lambda kv: kv[1]["weighted_E"])
    print(f"EN İYİ (weighted_E): {best[0]} → {best[1]}")
    print("=" * 78)
    return [{"configuration": name, **metrics} for name, metrics in rows.items()]


if __name__ == "__main__":
    main()
