"""
comprehensive_benchmark.py — NihaiAlgoritma & MainHub Projesi Kapsamlı Benchmark Test Paketi

Bu script, sistemi çok boyutlu olarak test eder ve ölçümler:
1. Donanım ve Ortam Profili (CPU, RAM, Python, Kütüphaneler)
2. Mikro-Benchmark: Teknik İndikatörler (EMA, RSI, MACD, BB, ATR, Supertrend, HMA, CMO, TSI vb.)
3. Mikro-Benchmark: Opsiyon Matematiği & Black-Scholes Greeks Motoru (compute_greeks, bs_greeks, implied_vol)
4. Mikro-Benchmark: Quant Math & AI Motoru (Oynaklık tahmincileri, Cornish-Fisher VaR, GBM Monte Carlo, Meta-Features)
5. Uçtan Uca Sinyal Üretim Hattı (End-to-End Signal Pipeline Throughput & Latency)
6. Makro-Benchmark: BASKET Evreni Strateji Backtest Performansı
7. Eşzamanlılık ve İş Parçacığı Ölçeklenme Testi (Concurrency Scaling)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from typing import Any

# Windows console encoding ayarı
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import numpy as np
import pandas as pd

from nihai.ai import options_math, quant_math

# Nihai imports
from nihai.config import LOG_DIR
from nihai.execution.backtest import backtest_symbol
from nihai.risk import options_analyzer
from nihai.signals import indicators as ind
from nihai.signals import indicators_ext as ind_ext
from nihai.signals import meta_features
from nihai.signals import strategy as strat
from nihai.universe import BASKET


# ---------------------------------------------------------------------------
# Sentetik Veri Üreteci (Tekrarlanabilir Benchmark Verisi)
# ---------------------------------------------------------------------------
def generate_synthetic_ohlcv(n_bars: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    mu = 0.08
    sigma = 0.25
    returns = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_bars)
    price_paths = 100.0 * np.exp(np.cumsum(returns))

    highs = price_paths * (1.0 + rng.uniform(0.001, 0.015, n_bars))
    lows = price_paths * (1.0 - rng.uniform(0.001, 0.015, n_bars))
    opens = price_paths * (1.0 + rng.normal(0.0, 0.003, n_bars))
    closes = price_paths
    volumes = rng.integers(50000, 2000000, n_bars).astype(float)

    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=idx)
    return df


# ---------------------------------------------------------------------------
# 1. Mikro-Benchmark: Teknik İndikatörler
# ---------------------------------------------------------------------------
def benchmark_indicators(n_runs: int = 50) -> dict:
    df_1k = generate_synthetic_ohlcv(1000)
    df_5k = generate_synthetic_ohlcv(5000)
    df_20k = generate_synthetic_ohlcv(20000)

    results = {}

    bench_funcs = [
        ("EMA (p=20)", lambda df: strat.ema(df["close"], 20)),
        ("SMA (p=20)", lambda df: ind_ext.sma(df["close"], 20)),
        ("RSI (p=14, RMA)", lambda df: strat.rsi(df["close"], 14)),
        ("MACD (12, 26, 9)", lambda df: strat.macd(df["close"], 12, 26, 9)),
        ("Bollinger Bands (20, 2.0)", lambda df: strat.bollinger_bands(df["close"], 20, 2.0)),
        ("ATR (p=14, Wilder)", lambda df: strat.atr(df, 14)),
        ("ADX (p=14)", lambda df: strat.adx(df, 14)),
        ("SuperTrend (10, 3.0)", lambda df: ind.supertrend(df, 10, 3.0)),
        ("VWAP (Rolling 20)", lambda df: ind.vwap(df, 20)),
        ("OBV (Cumulative)", lambda df: ind.obv(df)),
        ("MFI (p=14)", lambda df: ind.mfi(df, 14)),
        ("StochRSI (14, 3, 3)", lambda df: ind.stoch_rsi(df["close"], 14, 3, 3)),
        ("Keltner Channels (20, 2.0)", lambda df: ind.keltner_channels(df, 20, 2.0)),
        ("Hull MA (HMA 20)", lambda df: ind_ext.hma(df["close"], 20)),
        ("Chande Momentum (CMO 14)", lambda df: ind_ext.cmo(df["close"], 14)),
        ("True Strength Index (TSI)", lambda df: ind_ext.tsi(df["close"], 25, 13)),
    ]

    for name, func in bench_funcs:
        # Isınma turu
        func(df_1k)
        
        # 1.000 bar ölçümü
        t0 = time.perf_counter()
        for _ in range(n_runs):
            func(df_1k)
        t_1k = (time.perf_counter() - t0) / n_runs * 1000  # ms

        # 5.000 bar ölçümü
        t0 = time.perf_counter()
        for _ in range(n_runs):
            func(df_5k)
        t_5k = (time.perf_counter() - t0) / n_runs * 1000  # ms

        # 20.000 bar ölçümü (10 tur)
        t0 = time.perf_counter()
        for _ in range(max(5, n_runs // 5)):
            func(df_20k)
        t_20k = (time.perf_counter() - t0) / max(5, n_runs // 5) * 1000  # ms

        throughput_bars_per_sec = int(5000 / (t_5k / 1000.0))

        results[name] = {
            "1k_bars_ms": round(t_1k, 3),
            "5k_bars_ms": round(t_5k, 3),
            "20k_bars_ms": round(t_20k, 3),
            "throughput_bars_sec": throughput_bars_per_sec,
        }

    return results


# ---------------------------------------------------------------------------
# 2. Mikro-Benchmark: Opsiyon Matematiği & Black-Scholes Greeks
# ---------------------------------------------------------------------------
def benchmark_options(n_contracts: int = 5000, n_runs: int = 20) -> dict:
    rng = np.random.default_rng(42)
    underlyings = rng.uniform(50.0, 500.0, n_contracts)
    strikes = underlyings * rng.uniform(0.8, 1.2, n_contracts)
    dtes = rng.integers(1, 90, n_contracts) / 365.0
    ivs = rng.uniform(0.15, 0.90, n_contracts)
    types = rng.choice(["c", "p"], n_contracts)

    # 1. options_analyzer.compute_greeks
    t0 = time.perf_counter()
    for _ in range(n_runs):
        for i in range(min(500, n_contracts)):
            options_analyzer.compute_greeks(
                option_type=types[i],
                S=underlyings[i],
                K=strikes[i],
                T=dtes[i],
                r=0.05,
                sigma=ivs[i],
            )
    t_single_greeks = (time.perf_counter() - t0) / (n_runs * min(500, n_contracts)) * 1e6  # mikrosaniye

    # 2. quant_math.bs_greeks (Kapalı form Saf Python/NumPy)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        for i in range(min(500, n_contracts)):
            quant_math.bs_greeks(
                s=underlyings[i],
                k=strikes[i],
                t=dtes[i],
                r=0.05,
                sigma=ivs[i],
                kind=types[i],
            )
    t_quant_greeks = (time.perf_counter() - t0) / (n_runs * min(500, n_contracts)) * 1e6  # mikrosaniye

    # 3. quant_math.bs_price
    t0 = time.perf_counter()
    for _ in range(n_runs):
        for i in range(min(500, n_contracts)):
            quant_math.bs_price(
                s=underlyings[i],
                k=strikes[i],
                t=dtes[i],
                r=0.05,
                sigma=ivs[i],
                kind=types[i],
            )
    t_bs_price = (time.perf_counter() - t0) / (n_runs * min(500, n_contracts)) * 1e6  # mikrosaniye

    # 4. Implied Volatility Çözücü (options_math.implied_vol)
    call_prices = [
        quant_math.bs_price(s=underlyings[i], k=strikes[i], t=dtes[i], r=0.05, sigma=ivs[i], kind="c") or 5.0
        for i in range(200)
    ]
    t0 = time.perf_counter()
    for _ in range(n_runs):
        for i in range(200):
            options_math.implied_vol(
                price=call_prices[i],
                s=underlyings[i],
                k=strikes[i],
                t=dtes[i],
                r=0.05,
                kind="c",
            )
    t_iv_solver = (time.perf_counter() - t0) / (n_runs * 200) * 1e6  # mikrosaniye

    return {
        "options_analyzer_greeks_latency_us": round(t_single_greeks, 2),
        "options_analyzer_throughput_per_sec": int(1e6 / t_single_greeks) if t_single_greeks > 0 else 0,
        "quant_math_bs_greeks_latency_us": round(t_quant_greeks, 2),
        "quant_math_greeks_throughput_per_sec": int(1e6 / t_quant_greeks) if t_quant_greeks > 0 else 0,
        "bs_price_latency_us": round(t_bs_price, 2),
        "bs_price_throughput_per_sec": int(1e6 / t_bs_price) if t_bs_price > 0 else 0,
        "implied_vol_solver_latency_us": round(t_iv_solver, 2),
        "implied_vol_solves_per_sec": int(1e6 / t_iv_solver) if t_iv_solver > 0 else 0,
    }


# ---------------------------------------------------------------------------
# 3. Mikro-Benchmark: Quant Math & AI Çekirdeği
# ---------------------------------------------------------------------------
def benchmark_quant_ai(n_runs: int = 50) -> dict:
    df_500 = generate_synthetic_ohlcv(500)
    df_2k = generate_synthetic_ohlcv(2000)

    # 1. Volatility Estimators (Parkinson, Garman-Klass, Yang-Zhang)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        quant_math.vol_parkinson(df_2k["high"], df_2k["low"])
    t_parkinson = (time.perf_counter() - t0) / n_runs * 1000  # ms

    t0 = time.perf_counter()
    for _ in range(n_runs):
        quant_math.vol_garman_klass(df_2k["open"], df_2k["high"], df_2k["low"], df_2k["close"])
    t_gk = (time.perf_counter() - t0) / n_runs * 1000  # ms

    t0 = time.perf_counter()
    for _ in range(n_runs):
        quant_math.vol_yang_zhang(df_2k["open"], df_2k["high"], df_2k["low"], df_2k["close"])
    t_yz = (time.perf_counter() - t0) / n_runs * 1000  # ms

    # 2. Cornish-Fisher VaR / CVaR
    rets = np.diff(np.log(df_2k["close"].to_numpy()))
    t0 = time.perf_counter()
    for _ in range(n_runs * 5):
        quant_math.var_cornish_fisher(rets, alpha=0.05)
    t_var = (time.perf_counter() - t0) / (n_runs * 5) * 1e6  # us

    # 3. Meta-Feature Extraction (25+ Features per bar)
    sig_sample = strat.generate_signal("AAPL", df_500, strategy_mode="ensemble")
    t0 = time.perf_counter()
    for _ in range(n_runs):
        meta_features.extract_features(df_500, sig_sample)
    t_meta_features = (time.perf_counter() - t0) / n_runs * 1000  # ms

    # 4. Monte Carlo GBM Path Generation (1.000 yol, 252 gün)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        quant_math.mc_gbm_paths(s0=100.0, mu=0.08, sigma=0.25, days=252, n_paths=1000, seed=42)
    t_mc_gbm = (time.perf_counter() - t0) / n_runs * 1000  # ms

    return {
        "vol_parkinson_2k_ms": round(t_parkinson, 3),
        "vol_garman_klass_2k_ms": round(t_gk, 3),
        "vol_yang_zhang_2k_ms": round(t_yz, 3),
        "var_cornish_fisher_latency_us": round(t_var, 2),
        "meta_features_extraction_500b_ms": round(t_meta_features, 3),
        "meta_features_throughput_bars_sec": int(500 / (t_meta_features / 1000.0)) if t_meta_features > 0 else 0,
        "monte_carlo_1k_paths_ms": round(t_mc_gbm, 3),
    }


# ---------------------------------------------------------------------------
# 4. Uçtan Uca Sinyal Üretim Hattı Benchmark'ı
# ---------------------------------------------------------------------------
def benchmark_pipeline_latency(n_runs: int = 30) -> dict:
    df_300 = generate_synthetic_ohlcv(300)
    
    # generate_signal test
    t0 = time.perf_counter()
    for _ in range(n_runs):
        strat.generate_signal("AAPL", df_300, strategy_mode="ensemble")
    t_signal = (time.perf_counter() - t0) / n_runs * 1000  # ms

    # indicator_panel test
    t0 = time.perf_counter()
    for _ in range(n_runs):
        ind.indicator_panel(df_300, live_price=df_300["close"].iloc[-1])
    t_panel = (time.perf_counter() - t0) / n_runs * 1000  # ms

    return {
        "generate_signal_ensemble_ms": round(t_signal, 2),
        "signals_per_sec_capacity": int(1000.0 / t_signal) if t_signal > 0 else 0,
        "indicator_panel_ms": round(t_panel, 2),
        "panels_per_sec_capacity": int(1000.0 / t_panel) if t_panel > 0 else 0,
    }


# ---------------------------------------------------------------------------
# 5. Makro-Benchmark: BASKET Üzerinde Strateji Backtest Kıyası
# ---------------------------------------------------------------------------
def benchmark_strategy_basket(symbols: list[str], interval: str = "1h", period: str = "1y") -> dict:
    print(f"\n[MAKRO-BENCHMARK] Sepet ({len(symbols)} sembol) Backtest Başlıyor ({interval}/{period})...")
    t0 = time.perf_counter()
    # Değerler HETEROJEN (symbol str, geri kalanı sayı); literalden çıkarılan birleşim
    # tipi aşağıdaki aritmetiği (r["expectancy"] * r["trades"]) tip hatasına çeviriyordu.
    rows: list[dict[str, Any]] = []
    
    for i, s in enumerate(symbols, 1):
        t_sym = time.perf_counter()
        try:
            r = backtest_symbol(s, interval=interval, period=period, strategy_mode="ensemble", mc_iters=0)
            elapsed_sym = (time.perf_counter() - t_sym) * 1000
            if r.bars_total == 0:
                print(f"  [{i:>2}/{len(symbols)}] {s:<5}: Veri yok (atlandı) ({elapsed_sym:.0f} ms)")
                continue
            
            rows.append({
                "symbol": s,
                "trades": r.trades_count,
                "win_rate_pct": round(r.win_rate * 100, 2),
                "expectancy": round(r.expectancy, 2),
                # Tur-77: bu alanlar `float | None` oldu (ölçülemeyen değer null — backtest.py);
                # `round(None)` tüketiciyi çökertirdi. Ölçüm yokluğu None olarak taşınır, 0 uydurulmaz.
                "total_return_pct": None if r.total_return_pct is None else round(r.total_return_pct, 2),
                "max_drawdown_pct": None if r.max_drawdown_pct is None else round(r.max_drawdown_pct, 2),
                "sharpe": None if r.sharpe_annualized is None else round(r.sharpe_annualized, 2),
                "profit_factor": round(getattr(r, "profit_factor", 0.0) or 0.0, 2),
                "elapsed_ms": round(elapsed_sym, 1),
            })
            print(f"  [{i:>2}/{len(symbols)}] {s:<5}: {r.trades_count:>3} işlem | WR: {r.win_rate*100:>5.1f}% | "
                  f"E: {r.expectancy:>+6.2f} | Getiri: {r.total_return_pct:>+6.2f}% | "
                  f"MDD: {r.max_drawdown_pct:>5.2f}% | Sharpe: {r.sharpe_annualized:>5.2f} ({elapsed_sym:.0f} ms)")
        except Exception as e:
            print(f"  [{i:>2}/{len(symbols)}] {s:<5}: Hata -> {e}")

    total_time = time.perf_counter() - t0

    if not rows:
        return {"error": "Backtest sonucu üretilemedi", "total_time_sec": total_time}

    total_trades = sum(r["trades"] for r in rows)
    weighted_e = (sum(r["expectancy"] * r["trades"] for r in rows) / total_trades) if total_trades else 0.0
    profitable_syms = [r for r in rows if r["expectancy"] > 0]
    rets = [r["total_return_pct"] for r in rows]
    sharpes = [r["sharpe"] for r in rows]
    wrs = [r["win_rate_pct"] for r in rows]
    mdds = [r["max_drawdown_pct"] for r in rows]

    summary = {
        "symbols_tested": len(rows),
        "total_trades": total_trades,
        "avg_trades_per_symbol": round(total_trades / len(rows), 1),
        "weighted_expectancy": round(weighted_e, 2),
        "mean_win_rate_pct": round(statistics.mean(wrs), 2),
        "mean_total_return_pct": round(statistics.mean(rets), 2),
        "median_total_return_pct": round(statistics.median(rets), 2),
        "mean_sharpe": round(statistics.mean(sharpes), 2),
        "mean_max_drawdown_pct": round(statistics.mean(mdds), 2),
        "pct_symbols_profitable": round(100 * len(profitable_syms) / len(rows), 1),
        "total_backtest_duration_sec": round(total_time, 2),
        "avg_time_per_symbol_sec": round(total_time / len(rows), 3),
    }

    return {"summary": summary, "symbol_details": rows}


# ---------------------------------------------------------------------------
# 6. Eşzamanlılık & İş Parçacığı Ölçeklenme Testi
# ---------------------------------------------------------------------------
def benchmark_concurrency_scaling() -> dict:
    df = generate_synthetic_ohlcv(500)

    def work_unit():
        sig = strat.generate_signal("AAPL", df, strategy_mode="ensemble")
        ind.indicator_panel(df, live_price=df["close"].iloc[-1])
        meta_features.extract_features(df, sig)

    n_tasks = 40
    scaling_results = {}

    for workers in [1, 2, 4, 8]:
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(work_unit) for _ in range(n_tasks)]
            concurrent.futures.wait(futures)
        dur = time.perf_counter() - t0
        throughput = n_tasks / dur
        scaling_results[f"{workers}_workers"] = {
            "duration_sec": round(dur, 3),
            "throughput_tasks_sec": round(throughput, 1),
            "speedup_vs_1w": 0.0,
        }

    t1 = scaling_results["1_workers"]["duration_sec"]
    for k in scaling_results:
        scaling_results[k]["speedup_vs_1w"] = round(t1 / scaling_results[k]["duration_sec"], 2)

    return scaling_results


# ---------------------------------------------------------------------------
# Ana Benchmark Koşucusu
# ---------------------------------------------------------------------------
def run_all_benchmarks(quick: bool = False) -> dict:
    print("=" * 75)
    print("      NİHAİ ALGORİTMA & MAİNHUB KAPSAMLI BENCHMARK TEST SUITE")
    print("=" * 75)
    
    # 0. Sistem Bilgisi
    sys_info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "benchmark_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"OS: {sys_info['os']}")
    print(f"CPU: {sys_info['processor']} ({sys_info['cpu_count_logical']} Logical Cores)")
    print(f"Python: {sys_info['python_version']} | NumPy: {sys_info['numpy_version']} | Pandas: {sys_info['pandas_version']}")
    print("-" * 75)

    # 1. Mikro-Benchmark: İndikatörler
    print("\n[1/5] Mikro-Benchmark: Teknik İndikatör Hesaplama Hızları Ölçülüyor...")
    ind_results = benchmark_indicators(n_runs=30 if quick else 60)
    print("  ✓ İndikatör hızları ölçüldü.")

    # 2. Mikro-Benchmark: Opsiyon & Greeks
    print("\n[2/5] Mikro-Benchmark: Opsiyon ve Greeks Motoru Ölçülüyor...")
    opt_results = benchmark_options(n_runs=15 if quick else 30)
    print(f"  ✓ Options Analyzer Greeks Throughput: {opt_results['options_analyzer_throughput_per_sec']:,} hesaplama/sn")
    print(f"  ✓ Quant Math BS Greeks Throughput: {opt_results['quant_math_greeks_throughput_per_sec']:,} hesaplama/sn")
    print(f"  ✓ Black-Scholes Fiyatlama Throughput: {opt_results['bs_price_throughput_per_sec']:,} fiyatlama/sn")
    print(f"  ✓ Implied Volatility Solver: {opt_results['implied_vol_solves_per_sec']:,} çözüm/sn")

    # 3. Mikro-Benchmark: Quant & AI
    print("\n[3/5] Mikro-Benchmark: Quant Math & AI Motoru Ölçülüyor...")
    quant_results = benchmark_quant_ai(n_runs=30 if quick else 60)
    print(f"  ✓ Meta-Feature Çıkarım Throughput: {quant_results['meta_features_throughput_bars_sec']:,} bar/sn")
    print(f"  ✓ Monte Carlo 1.000 Yol Süresi: {quant_results['monte_carlo_1k_paths_ms']} ms")

    # 4. Uçtan Uca Pipeline & Concurrency
    print("\n[4/5] Uçtan Uca Pipeline & Eşzamanlılık Test Ediliyor...")
    pipe_results = benchmark_pipeline_latency(n_runs=20 if quick else 40)
    scaling_results = benchmark_concurrency_scaling()
    print(f"  ✓ Tek Çekirdek Sinyal Üretim Kapasitesi: {pipe_results['signals_per_sec_capacity']} sembol/sn")
    print(f"  ✓ 4 Worker Ölçeklenme Hızı: {scaling_results.get('4_workers', {}).get('speedup_vs_1w', 1.0)}x")

    # 5. Makro-Benchmark: BASKET Strateji Backtesti
    print("\n[5/5] Makro-Benchmark: BASKET Evreni Strateji Kıyaslama Testi...")
    basket_symbols = BASKET
    macro_results = benchmark_strategy_basket(basket_symbols, interval="1h", period="1y")

    all_results = {
        "system_info": sys_info,
        "indicators_benchmark": ind_results,
        "options_greeks_benchmark": opt_results,
        "quant_ai_benchmark": quant_results,
        "pipeline_latency_benchmark": pipe_results,
        "concurrency_scaling_benchmark": scaling_results,
        "strategy_macro_benchmark": macro_results,
    }

    # Sonuçları JSON olarak kaydet
    out_file = LOG_DIR / f"comprehensive_benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_file.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 75)
    print(f" BENCHMARK TAMAMLANDI! Sonuçlar kaydedildi: {out_file}")
    print("=" * 75)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NihaiAlgoritma Kapsamlı Benchmark Testi")
    parser.add_argument("--quick", action="store_true", help="Hızlı modda çalıştır (daha az sembol ve iterasyon)")
    args = parser.parse_args()

    run_all_benchmarks(quick=args.quick)
