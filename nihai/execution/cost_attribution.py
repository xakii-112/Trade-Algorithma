"""İşlem fiyatlarından dört parçalı implementation shortfall (dolar).

--orders-csv: side,quantity,filled_quantity,signal_price,decision_price,fill_price,
fees,benchmark_price,multiplier sütunları; boş fiyat/ücret = ölçülmedi.
--trades-csv: sanal işlem defterindeki maliyetler ve ölçüm kapsamı.
Dosya verilmezse eski maliyet açık/kapalı backtest duyarlılık karşılaştırması.
Bu karşılaştırma boyut/çıkışları da değiştirebilir; fark saf maliyet atfı DEĞİLDİR.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from nihai.config import CONFIG
from nihai.console import ensure_utf8_stdout
from nihai.universe import BASKET

COST_FIELDS = ("delay_cost", "market_impact", "fees", "opportunity_cost", "total_shortfall")


def implementation_shortfall(*, side: str, quantity: float, filled_quantity: float,
                             signal_price: float | None, decision_price: float | None,
                             fill_price: float | None, fees: float | None,
                             benchmark_price: float | None = None,
                             multiplier: float = 1.0) -> dict | None:
    """Perold muhasebesi; BUY/SELL emir yönü, pozisyonun boğa/ayı yönü değildir.

    Gecikme ve icra farkı yalnız DOLAN miktara, fırsat maliyeti kalan miktara yazılır.
    Fırsat maliyeti sinyal fiyatını referans alır; kaçırılan zarar negatif kalır.
    Benchmark önceden seçilmiş değerlendirme ufkunun fiyatıdır, karar girdisi değil.
    'market_impact' varış→dolum farkıdır; nedensel kendi-emir etkisini kanıtlamaz.
    Eksik fiyat/ücret None kalır. Ücret negatifse açıkça girilmiş iadedir.
    Geçersiz girdi (yön, miktar, NaN/Inf/negatif fiyat, taşma) → None, istisna YOK (Tur-78, R3-B1):
    bu bir TEŞHİSTİR ve canlı kapanış muhasebesinden çağrılır; hesaplanamaması kapanışı durduramaz.
    """
    if side not in ("BUY", "SELL"):
        return None
    try:
        q, filled, mult = float(quantity), float(filled_quantity), float(multiplier)
        prices = [None if v is None else float(v)
                  for v in (signal_price, decision_price, fill_price, benchmark_price)]
        fee = None if fees is None else float(fees)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in (q, filled, mult)) or not (q > 0 and 0 <= filled <= q and mult > 0):
        return None
    if any(v is not None and (not math.isfinite(v) or v < 0) for v in prices):
        return None
    if fee is not None and not math.isfinite(fee):
        return None
    signal, decision, fill, benchmark = prices
    signed_mult = mult if side == "BUY" else -mult

    def change(amount, before, after):
        if amount == 0:
            return 0.0
        return None if before is None or after is None else signed_mult * amount * (after - before)

    delay = change(filled, signal, decision)
    impact = change(filled, decision, fill)
    opportunity = change(q - filled, signal, benchmark)
    execution = change(filled, signal, fill)
    total = (execution + opportunity + fee
             if execution is not None and opportunity is not None and fee is not None else None)
    paper = change(q, signal, benchmark)
    actual_gross = change(filled, fill, benchmark)
    actual = actual_gross - fee if actual_gross is not None and fee is not None else None
    out = dict(delay_cost=delay, market_impact=impact, fees=fee,
               opportunity_cost=opportunity, total_shortfall=total,
               paper_pnl=paper, actual_pnl=actual, unfilled_quantity=q - filled)
    if any(v is not None and not math.isfinite(v) for v in out.values()):
        return None  # sayısal sınır aşıldı
    return out


def cost_summary(records: Iterable[Mapping], previous: Mapping | None = None) -> dict:
    """Bilinen tutarların toplamı + her sütunun ayrı ölçüm sayısı; eksik != sıfır.

    previous ile sabit bellekli birikim; önceki sözlük değişmez. Kısmi toplamın
    kapsamı measured[field]/records ile okunur, tüm portföy maliyeti sayılmaz.
    Sonlu olmayan/sayı olmayan tutar ya da taşan toplam ölçüm SAYILMAZ (toplam ve sayaç
    değişmez, istisna yok — Tur-78, R3-B1); açık measured/records farkında görünür.
    """
    result = dict(previous or {"records": 0, **dict.fromkeys(COST_FIELDS)})
    measured = dict(result.get("measured", dict.fromkeys(COST_FIELDS, 0)))
    for record in records:
        result["records"] += 1
        for key in COST_FIELDS:
            value = record.get(key)
            if value is None:
                continue
            try:
                total = (result.get(key) or 0.0) + float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(total):
                result[key] = total
                measured[key] += 1
    result["measured"] = measured
    return result


def read_costs(path: Path, *, orders: bool = False) -> dict:
    """CSV'yi salt okunur işler; eski işlem defterinin eksik alanları bilinmez kalır."""
    records = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for number, row in enumerate(csv.DictReader(stream), start=2):
            try:
                if orders:
                    values = {k: float(row[k]) if (row.get(k) or "").strip() else None
                              for k in ("signal_price", "decision_price", "fill_price", "fees", "benchmark_price")}
                    record = implementation_shortfall(
                        side=row["side"], quantity=float(row["quantity"]),
                        filled_quantity=float(row["filled_quantity"]),
                        multiplier=float(row.get("multiplier") or 1), **values)
                    if record is None:  # CLI sınırı: kullanıcı dosyasındaki bozuk satır ADIYLA bildirilir
                        raise ValueError("geçersiz yön, miktar, fiyat veya ücret")
                else:
                    record = {k: float(row[k]) if (row.get(k) or "").strip() else None for k in COST_FIELDS}
                records.append(record)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"CSV satır {number}: {exc}") from exc
    return cost_summary(records)


def run_basket(bars: dict, mode: str, interval: str = "1h") -> tuple[float, int]:
    from nihai.execution.backtest import backtest_symbol

    total_pnl = 0.0
    trades = 0
    for s, df in bars.items():
        r = backtest_symbol(s, interval=interval, df=df, strategy_mode=mode)
        total_pnl += sum(r.trade_pnls)
        trades += len(r.trade_pnls)
    return total_pnl, trades


def set_costs(stock_slip: float, opt_slip: float, commission: float, impact_coef: float = 0.0) -> None:
    bc = CONFIG.backtest
    object.__setattr__(bc, "stock_slippage_pct", stock_slip)
    object.__setattr__(bc, "option_slippage_pct", opt_slip)
    object.__setattr__(bc, "stock_commission_per_trade", commission)
    object.__setattr__(bc, "impact_coef", impact_coef)


def main():
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser()
    files = ap.add_mutually_exclusive_group()
    files.add_argument("--orders-csv", type=Path)
    files.add_argument("--trades-csv", type=Path)
    ap.add_argument("--mode", default="event_breakout")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--periods", default="1y,2y")
    args = ap.parse_args()
    if args.orders_csv or args.trades_csv:
        try:
            report = read_costs(args.orders_csv or args.trades_csv, orders=bool(args.orders_csv))
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return
    saved = [(cfg, name, getattr(cfg, name)) for cfg, names in (
        (CONFIG.strategy, ("enable_trend_alignment_gate", "enable_adx_floor_gate", "min_signal_quality_score")),
        (CONFIG.backtest, ("stock_slippage_pct", "option_slippage_pct", "stock_commission_per_trade", "impact_coef")))
             for name in names]
    try:
        _compare_costs(args)
    finally:
        for cfg, name, value in saved:
            object.__setattr__(cfg, name, value)


def _compare_costs(args):
    from nihai.execution.backtest import load_bars

    periods = [p.strip() for p in args.periods.split(",") if p.strip()]

    # Saf strateji (kapılar kapalı, tutarlılık).
    object.__setattr__(CONFIG.strategy, "enable_trend_alignment_gate", False)
    object.__setattr__(CONFIG.strategy, "enable_adx_floor_gate", False)
    object.__setattr__(CONFIG.strategy, "min_signal_quality_score", 1.0)

    # Varsayılan maliyet değerleri (config'ten okunan).
    bc = CONFIG.backtest
    DEF_SLIP, DEF_OPT, DEF_COMM = bc.stock_slippage_pct, bc.option_slippage_pct, bc.stock_commission_per_trade
    default_impact = bc.impact_coef

    print(f"=== MALİYET ATFI · {args.mode} · {args.interval} (hisse) ===")
    print(f"(maliyet AÇIK: hisse slippage %{DEF_SLIP*100:.2f}/yön, komisyon ${DEF_COMM:.2f}/işlem)\n")
    hdr = f"{'periyot':>8} {'işlem':>6} {'PnL_açık':>12} {'PnL_kapalı':>12} {'maliyet_$':>11} {'maliyet/işlem':>13} {'E_açık':>9} {'E_kapalı':>9}"
    print(hdr)
    print("-" * len(hdr))

    for per in periods:
        bars = {}
        for s in BASKET:
            df = load_bars(s, args.interval, period=per)
            if not df.empty:
                bars[s] = df

        set_costs(DEF_SLIP, DEF_OPT, DEF_COMM, default_impact)
        pnl_on, n_on = run_basket(bars, args.mode, args.interval)
        set_costs(0.0, 0.0, 0.0)                       # KAPALI
        pnl_off, n_off = run_basket(bars, args.mode, args.interval)

        drag = pnl_off - pnl_on
        per_trade = drag / n_on if n_on else 0.0
        e_on = pnl_on / n_on if n_on else 0.0
        e_off = pnl_off / n_off if n_off else 0.0
        print(f"{per:>8} {n_on:>6} {pnl_on:>+12.0f} {pnl_off:>+12.0f} {drag:>11.0f} "
              f"{per_trade:>13.2f} {e_on:>+9.1f} {e_off:>+9.1f}")

    print("\nYorum: 'PnL_kapalı' bile pozitif değilse → kök-neden maliyet DEĞİL, kenar yokluğu.")
    print("Fark, boyut ve çıkış değişikliklerini de içerir; saf maliyet için --trades-csv kullanın.")


if __name__ == "__main__":
    main()
