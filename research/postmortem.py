"""
postmortem.py — Haftalık/aylık post-mortem rapor üreticisi (A1).

Kaynak: data/trades.csv (signal_engine.log_trade_csv ile her kapanışta yazılır) +
data/equity_curve.csv (main.append_equity_point). İkisi de makine-okur → bu modül
trade_log.txt'yi PARSE ETMEZ (kırılganlık yok).

Üretilen: WR, profit factor, expectancy, ortalama kazanç/kayıp, toplam PnL, Sharpe
(işlem-bazlı), max drawdown (equity eğrisinden), en iyi/kötü işlem, strateji/sektör
kırılımı → Markdown. Alfa İDDİASI YOK; dürüst muhasebe (docs/EDGE_FINDINGS.md ilkesi).

Kullanım:
    python -m research.postmortem                      # tüm geçmiş, stdout + dosya
    python -m research.postmortem --period 7d          # son 7 gün
    python -m research.postmortem --period 30d --telegram   # özet Telegram'a
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from datetime import datetime, timedelta
from pathlib import Path

from nihai.config import DATA_DIR
from nihai.console import ensure_utf8_stdout
from nihai.execution.backtest import max_drawdown

TRADES_CSV = DATA_DIR / "trades.csv"
EQUITY_CSV = DATA_DIR / "equity_curve.csv"


def _parse_dt(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _parse_period(period: str | None) -> timedelta | None:
    """'7d' / '30d' / '12h' / 'all' → timedelta (None = sınırsız)."""
    if not period or period.lower() == "all":
        return None
    period = period.strip().lower()
    units = {"d": "days", "h": "hours", "w": "weeks"}
    unit = period[-1]
    if unit in units and period[:-1].isdigit():
        return timedelta(**{units[unit]: int(period[:-1])})
    raise ValueError(f"Geçersiz period: {period!r} (örn. 7d, 30d, 12h, all)")


def load_trades(period: str | None = None) -> list[dict]:
    """data/trades.csv'yi oku; period verilirse exit_time'a göre filtrele."""
    if not TRADES_CSV.exists():
        return []
    cutoff = None
    delta = _parse_period(period)
    if delta is not None:
        cutoff = datetime.now() - delta
    rows: list[dict] = []
    with TRADES_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            xt = _parse_dt(row.get("exit_time", ""))
            if cutoff is not None and (xt is None or xt < cutoff):
                continue
            rows.append(row)
    return rows


def load_equity(period: str | None = None) -> list[float]:
    """data/equity_curve.csv'den equity serisi (period filtreli)."""
    if not EQUITY_CSV.exists():
        return []
    cutoff = None
    delta = _parse_period(period)
    if delta is not None:
        cutoff = datetime.now() - delta
    series: list[float] = []
    with EQUITY_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ts = _parse_dt(row.get("timestamp", ""))
            if cutoff is not None and (ts is None or ts < cutoff):
                continue
            try:
                series.append(float(row["equity"]))
            except (ValueError, KeyError, TypeError):
                continue
    return series


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (ValueError, TypeError):
        return 0.0


def compute_metrics(trades: list[dict], equity: list[float]) -> dict:
    """İşlem listesinden temel performans metrikleri."""
    pnls = [_f(t, "realized_pnl") for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    total = sum(pnls)
    pf = (gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)
    expectancy = (total / n) if n else 0.0
    # İşlem-bazlı Sharpe = ort/std (BOYUTSUZ, örneklem sayısından bağımsız dağılım metriği).
    # B6 (2026-07-09): eskiden buraya *sqrt(n) çarpılıyordu → o bir Sharpe DEĞİL, t-istatistiği
    # (anlamlılık) idi; n büyüdükçe şişip yanıltıyordu. Şimdi ikisi AYRI raporlanıyor:
    #   sharpe_trade = mean/std           → işlem-başı risk-ayarlı getiri (dürüst büyüklük)
    #   t_stat       = mean/std*sqrt(n)   → "ort. PnL sıfırdan farklı mı?" anlamlılık testi
    sharpe = 0.0
    t_stat = 0.0
    if n > 1:
        sd = statistics.pstdev(pnls)
        if sd > 0:
            sharpe = statistics.mean(pnls) / sd
            t_stat = sharpe * math.sqrt(n)
    rs = [_f(t, "r_multiple") for t in trades if t.get("r_multiple")]
    return {
        "n": n,
        "win_rate": (len(wins) / n * 100) if n else 0.0,
        "wins": len(wins), "losses": len(losses),
        "profit_factor": pf,
        "expectancy": expectancy,
        "total_pnl": total,
        "avg_win": (statistics.mean(wins) if wins else 0.0),
        "avg_loss": (statistics.mean(losses) if losses else 0.0),
        "avg_r": (statistics.mean(rs) if rs else 0.0),
        "sharpe_trade": sharpe,
        "t_stat": t_stat,
        "max_drawdown_pct": max_drawdown(equity) if equity else 0.0,
        "best": max(trades, key=lambda t: _f(t, "realized_pnl")) if trades else None,
        "worst": min(trades, key=lambda t: _f(t, "realized_pnl")) if trades else None,
    }


def _breakdown(trades: list[dict], key: str) -> list[tuple[str, int, float]]:
    """key (strategy/sector) bazında (ad, işlem, toplam_pnl) — pnl'e göre azalan."""
    agg: dict[str, list[float]] = {}
    for t in trades:
        agg.setdefault(t.get(key) or "—", []).append(_f(t, "realized_pnl"))
    out = [(name, len(v), sum(v)) for name, v in agg.items()]
    return sorted(out, key=lambda x: x[2], reverse=True)


def _pf_str(pf: float) -> str:
    return "∞" if pf == math.inf else f"{pf:.2f}"


def render_markdown(trades: list[dict], equity: list[float], period: str | None) -> str:
    m = compute_metrics(trades, equity)
    label = period or "tüm geçmiş"
    lines = [
        f"# Post-Mortem Raporu — {label}",
        f"_Üretim: {datetime.now():%Y-%m-%d %H:%M} · kaynak: data/trades.csv_",
        "",
        "> Alfa iddiası yok; dürüst muhasebe (docs/EDGE_FINDINGS.md).",
        "",
        "## Özet",
        "",
        "| Metrik | Değer |",
        "|---|---|",
        f"| İşlem sayısı | {m['n']} |",
        f"| Kazanma oranı (WR) | {m['win_rate']:.1f}% ({m['wins']}K/{m['losses']}Z) |",
        f"| Profit Factor | {_pf_str(m['profit_factor'])} |",
        f"| Expectancy ($/işlem) | {m['expectancy']:+.2f} |",
        f"| Ortalama R | {m['avg_r']:+.2f} |",
        f"| Toplam PnL | {m['total_pnl']:+,.2f} |",
        f"| Ort. kazanç / kayıp | {m['avg_win']:+,.2f} / {m['avg_loss']:+,.2f} |",
        f"| Sharpe (işlem-bazlı, ort/std) | {m['sharpe_trade']:.2f} |",
        f"| t-istatistiği (anlamlılık, ort/std·√n) | {m['t_stat']:.2f} |",
        f"| Max drawdown | {m['max_drawdown_pct']:.1f}% |",
        "",
    ]
    if m["best"] and m["worst"]:
        b, w = m["best"], m["worst"]
        lines += [
            "## Uç işlemler",
            "",
            f"- **En iyi:** {b.get('symbol')} {b.get('strategy')} "
            f"→ {_f(b,'realized_pnl'):+,.2f} ({b.get('exit_trigger')})",
            f"- **En kötü:** {w.get('symbol')} {w.get('strategy')} "
            f"→ {_f(w,'realized_pnl'):+,.2f} ({w.get('exit_trigger')})",
            "",
        ]
    for key, title in (("strategy", "Strateji"), ("sector", "Sektör")):
        bd = _breakdown(trades, key)
        if bd and any(name != "—" for name, _, _ in bd):
            lines += [f"## {title} kırılımı", "", f"| {title} | İşlem | Toplam PnL |", "|---|---|---|"]
            lines += [f"| {name} | {cnt} | {pnl:+,.2f} |" for name, cnt, pnl in bd]
            lines.append("")
    return "\n".join(lines)


def _telegram_summary(trades: list[dict], equity: list[float], period: str | None) -> str:
    m = compute_metrics(trades, equity)
    return (
        f"📊 Post-Mortem ({period or 'tüm geçmiş'})\n"
        f"İşlem: {m['n']} | WR: {m['win_rate']:.0f}% | PF: {_pf_str(m['profit_factor'])}\n"
        f"Expectancy: {m['expectancy']:+.2f} | Toplam: {m['total_pnl']:+,.0f}\n"
        f"Sharpe(işlem): {m['sharpe_trade']:.2f} | t-stat: {m['t_stat']:.1f} | "
        f"MaxDD: {m['max_drawdown_pct']:.1f}%"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-mortem rapor üreticisi (A1)")
    ap.add_argument("--period", default="all", help="7d | 30d | 12h | all (varsayılan all)")
    ap.add_argument("--out", default="", help="Markdown çıktı yolu (boş → data/postmortem_<tarih>.md)")
    ap.add_argument("--telegram", action="store_true", help="Özeti Telegram'a gönder")
    ap.add_argument("--no-file", action="store_true", help="Dosyaya yazma, sadece stdout")
    args = ap.parse_args()

    trades = load_trades(args.period)
    equity = load_equity(args.period)
    if not trades:
        print(f"[postmortem] {args.period} için kapanmış işlem yok (data/trades.csv).")
        return

    md = render_markdown(trades, equity, args.period)
    print(md)

    if not args.no_file:
        out = Path(args.out) if args.out else DATA_DIR / f"postmortem_{datetime.now():%Y%m%d}.md"
        out.write_text(md, encoding="utf-8")
        print(f"\n[postmortem] yazıldı: {out}")

    if args.telegram:
        try:
            from nihai.notifier.telegram_bot import send
            send(_telegram_summary(trades, equity, args.period))
            print("[postmortem] Telegram özeti gönderildi.")
        except Exception as e:
            print(f"[postmortem] Telegram gönderilemedi: {e}")


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    main()
