"""
gap_stop_diagnose.py — GAP_STOP kaybı mikroyapı mı, gerçek fiyat sıçraması mı?

Neden: canlı paper defterinde (`data/trades.csv`) işlemlerin ~%15'i `GAP_STOP` ile
kapanıyor ve TOPLAM kaybın tamamından fazlasını tek başına üretiyor (ort. R ≈ −1,46;
diğer tüm çıkışlar çıkarıldığında portföy artıya dönüyor). İki rakip açıklama var:

  H1 — MİKROYAPI: stop, etkin bid-ask makasının/ilk-bar gürültüsünün İÇİNE konmuş.
       O zaman çözüm stop yerleşimidir (makas tabanı, `--exec-lag 1`, ATR zemini).
  H2 — GERÇEK SIÇRAMA: fiyat stop'u gerçek bir hareketle atlıyor.
       O zaman makas ölçümü hiçbir şey kurtarmaz; sorun giriş zamanlaması/rejimidir.

Bu modül ikisini AYIRIR. `nihai/ai/microstructure` ailesinin üretim tarafına bağlı
olmayan makas tahmincileri (`research.spread_estimate`) burada ilk kez canlı defterle
karşılaştırılır.

Ölçüt (hepsi bps; 1 bps = 0,01%):
  * `stop_bps`      — PLANLI stop mesafesi = (risk_dollars / quantity) / entry_price
  * `gerceklesen`   — çıkışta fiilen yenen yön-ayarlı kayıp
  * `asim_bps`      — gerçekleşen − planlı (stop'un ötesine geçen kısım)
  * `spread_bps`    — Corwin-Schultz / Roll etkin makas tahmini (muhafazakâr = max)
  * `gurultu_bps`   — 1m bar mutlak getirisinin MEDYANI (bar-içi tipik hareket)

VERDİKT KURALI (önceden yazıldı, sonuca göre değil):
  H1 ancak `asim_bps` makasla AYNI MERTEBEDE ise (oran < 3×) desteklenir. Aşım makasın
  birkaç katıysa mikroyapı açıklaması YETMEZ → H2.

DÜRÜSTLÜK NOTU: makas/gürültü tahmini SON `--days` günün penceresinden gelir, işlemin
KENDİ anından değil. Bu bir vekildir; mertebe karşılaştırması için yeterli, tek tek
işlem atfı için değildir. Tahminciler zaten yanlıdır (Corwin-Schultz yukarı, Roll aşağı)
— maliyeti KÜÇÜK göstermek kenar uydurmanın en kolay yoludur (ACI DERS #7), o yüzden
muhafazakâr olan (büyük) kullanılır.

Kullanım:
    python -m research.gap_stop_diagnose
    python -m research.gap_stop_diagnose --days 20 --interval 1m
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from nihai.console import ensure_utf8_stdout
from research.spread_estimate import estimate

TRADES = "data/trades.csv"


def load_trades(path: str = TRADES) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    yon = np.where(d["side"] == "LONG", 1.0, -1.0)
    # PLANLI stop mesafesi: risk_dollars = quantity × (entry − stop)
    d["stop_bps"] = (d["risk_dollars"] / d["quantity"]) / d["entry_price"] * 1e4
    # Fiilen gerçekleşen yön-ayarlı getiri (negatif = kayıp)
    d["gerceklesen_bps"] = (d["exit_price"] - d["entry_price"]) / d["entry_price"] * 1e4 * yon
    # Stop'un ÖTESİNE geçen kısım (yalnız stop'u aşan kayıplarda anlamlı)
    d["asim_bps"] = -d["gerceklesen_bps"] - d["stop_bps"]
    return d


def bar_metrics(symbols: list[str], interval: str, days: int) -> dict[str, dict[str, float]]:
    """Sembol → {spread_bps, gurultu_bps, n_bar}. Tek sembol hatası tabloyu düşürmez."""
    from datetime import UTC, datetime, timedelta

    from nihai.data.alpaca_history import get_history

    start = datetime.now(UTC) - timedelta(days=days)
    interval_min = 1.0 if interval.endswith("m") else 60.0
    out: dict[str, dict[str, float]] = {}
    for i, sym in enumerate(symbols, 1):
        try:
            df = get_history(sym, interval, start=start)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(symbols)}] {sym}: çekilemedi ({e})", file=sys.stderr)
            continue
        if df is None or df.empty or len(df) < 100:
            print(f"[{i}/{len(symbols)}] {sym}: veri yetersiz", file=sys.stderr)
            continue
        sp = estimate(df, interval_min)
        r = np.abs(np.diff(np.log(df["close"].to_numpy(dtype=float))))
        out[sym] = {
            "spread_bps": float(sp["max_bps"]),
            "gurultu_bps": float(np.nanmedian(r) * 1e4),
            "n_bar": float(sp["n_bar"]),
        }
    return out


def _ozet(s: pd.Series) -> str:
    return f"medyan {s.median():7.1f}  ort {s.mean():7.1f}  (n={len(s)})"


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="GAP_STOP: mikroyapı mı, gerçek sıçrama mı?")
    ap.add_argument("--trades", default=TRADES)
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--days", type=int, default=10, help="makas/gürültü penceresi")
    ap.add_argument("--esik", type=float, default=3.0, help="H1 için aşım/makas üst sınırı")
    args = ap.parse_args(argv)

    d = load_trades(args.trades)
    gap = d[d["exit_trigger"] == "GAP_STOP"]
    if gap.empty:
        print("GAP_STOP işlemi yok — teşhis edilecek bir şey yok.")
        return 0

    print("=" * 64)
    print("GAP_STOP TEŞHİSİ — mikroyapı (H1) mı, gerçek sıçrama (H2) mı?")
    print("=" * 64)
    print(f"Defter: {args.trades}  ·  toplam {len(d)} işlem  ·  GAP_STOP {len(gap)} "
          f"({100 * len(gap) / len(d):.1f}%)")
    print(f"GAP_STOP toplam PnL: {gap['realized_pnl'].sum():,.0f} $   "
          f"(tüm defter: {d['realized_pnl'].sum():,.0f} $)")
    print()

    print("--- 1) PLANLI STOP MESAFESİ, çıkış tetiğine göre (bps) ---")
    for trig, g in d.groupby("exit_trigger"):
        print(f"  {trig:<10} {_ozet(g['stop_bps'])}")
    print("  → GAP_STOP'un stop'u diğerlerinden DAR mı? Değilse yerleşim tek başına suçlu değil.")
    print()

    print("--- 2) GAP_STOP: plan ↔ gerçekleşen ---")
    print(f"  planlı stop     {_ozet(gap['stop_bps'])}")
    print(f"  gerçekleşen     {_ozet(-gap['gerceklesen_bps'])}")
    print(f"  AŞIM (fark)     {_ozet(gap['asim_bps'])}")
    print()

    syms = sorted(gap["symbol"].unique())
    print(f"--- 3) Etkin makas + bar gürültüsü ({len(syms)} sembol, {args.interval}, "
          f"son {args.days} gün) ---")
    m = bar_metrics(syms, args.interval, args.days)
    if not m:
        print("  ÖLÇÜLEMEDİ (veri yok) — verdikt verilemez.")
        return 1
    tab = pd.DataFrame(m).T
    print(f"  makas (max)     medyan {tab['spread_bps'].median():.2f} bps  "
          f"[{tab['spread_bps'].min():.2f} – {tab['spread_bps'].max():.2f}]")
    print(f"  bar gürültüsü   medyan {tab['gurultu_bps'].median():.2f} bps")
    print(f"  ölçülen sembol  {len(tab)}/{len(syms)}")
    print()

    g2 = gap[gap["symbol"].isin(tab.index)].copy()
    g2["spread_bps"] = g2["symbol"].map(tab["spread_bps"])
    g2["gurultu_bps"] = g2["symbol"].map(tab["gurultu_bps"])
    g2["stop_x_makas"] = g2["stop_bps"] / g2["spread_bps"].replace(0, np.nan)
    g2["asim_x_makas"] = g2["asim_bps"] / g2["spread_bps"].replace(0, np.nan)

    print("--- 4) ORANLAR (asıl soru) ---")
    print(f"  stop / makas    medyan {g2['stop_x_makas'].median():.1f}×   "
          "(<3× ise stop gürültünün İÇİNDE)")
    print(f"  aşım / makas    medyan {g2['asim_x_makas'].median():.1f}×   "
          f"(<{args.esik:.0f}× ise mikroyapı açıklar)")
    print()

    oran = float(g2["asim_x_makas"].median())
    stop_oran = float(g2["stop_x_makas"].median())
    print("=" * 64)
    if not np.isfinite(oran):
        print("VERDİKT: ÖLÇÜLEMEDİ — makas tahmini sıfır/eksik.")
    elif oran < args.esik:
        print(f"VERDİKT: H1 (MİKROYAPI). Aşım makasın {oran:.1f}× katı, eşik {args.esik:.0f}× altında.")
        print("  → Stop yerleşimi düzeltilebilir: makas tabanı + ATR zemini + `--exec-lag 1`.")
    else:
        print(f"VERDİKT: H2 (GERÇEK SIÇRAMA). Aşım makasın {oran:.1f}× katı, eşiğin ÇOK üstünde;")
        print(f"  stop zaten makasın {stop_oran:.0f}× uzağında = gürültünün DIŞINDA.")
        print("  → Makas ölçümü bu kaybı KURTARMAZ. Kayıp gerçek ters fiyat hareketidir;")
        print("    çözüm stop yerleşimi değil pozisyon boyutu / giriş rejimi / gap maruziyetidir.")
    print("=" * 64)
    return 0


def demo() -> None:
    """Küçük öz-denetim: türetilen alanlar tutarlı mı (ağ gerektirmez)."""
    d = pd.DataFrame({
        "side": ["LONG", "SHORT"],
        "entry_price": [100.0, 50.0],
        "exit_price": [98.0, 51.0],
        "quantity": [10.0, 20.0],
        "risk_dollars": [10.0, 20.0],   # → 1,00 $/adet ve 1,00 $/adet
        "exit_trigger": ["GAP_STOP", "GAP_STOP"],
        "realized_pnl": [-20.0, -20.0],
        "entry_time": ["2026-01-01", "2026-01-01"],
        "exit_time": ["2026-01-02", "2026-01-02"],
        "symbol": ["A", "B"],
    })
    d.to_csv(_TMP, index=False)
    out = load_trades(_TMP)
    # LONG: stop 1/100 = 100 bps · gerçekleşen −2/100 = −200 bps · aşım 100 bps
    assert abs(out.loc[0, "stop_bps"] - 100.0) < 1e-6, out.loc[0, "stop_bps"]
    assert abs(out.loc[0, "gerceklesen_bps"] + 200.0) < 1e-6
    assert abs(out.loc[0, "asim_bps"] - 100.0) < 1e-6
    # SHORT: stop 1/50 = 200 bps · fiyat yükseldi → gerçekleşen −200 bps · aşım 0
    assert abs(out.loc[1, "stop_bps"] - 200.0) < 1e-6
    assert abs(out.loc[1, "gerceklesen_bps"] + 200.0) < 1e-6
    assert abs(out.loc[1, "asim_bps"] - 0.0) < 1e-6
    print("demo OK")


_TMP = "data/_gap_stop_demo.csv"

if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        raise SystemExit(main())
