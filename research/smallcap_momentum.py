"""
smallcap_momentum.py — Survivorship'siz SMALL-CAP cross-sectional momentum (Yol 3).

NEDEN SMALL-CAP: EDGE_FINDINGS, saf fiyat-momentum'unun likit MEGA-cap'lerde kenarsız
olduğunu gösterdi (§3). Momentum anomalisi tarihsel olarak küçük/likit-olmayan isimlerde
DAHA güçlüdür — ama tam orada survivorship en öldürücü (small-cap'ler ağır delisting
yaşar). Bu yüzden bu test ZORUNLU olarak point-in-time evren ister: `nihai/data/sharadar.py`
(çıkarılan isimler dahil). yfinance survivor'larında ölçmek bias'ı ARTIRIR — bu script
kasıtlı olarak yfinance'e DÜŞMEZ; Sharadar yoksa net yönerge verip çıkar.

YÖNTEM: her pencere için (a) pencerenin BAŞINDA hayatta olan small-cap evreni kur
(survivorship'siz), (b) o evrenin günlük closeadj panelini çek, (c) xsect.run_momentum'u
(kanıtlanmış, look-ahead-güvenli) bu panelde koştur. KRİTİK kontrol değişmedi (§6):
edge_vs_buyhold > 0 olmalı — yoksa sadece small-cap betası, kenar değil. İKİ pencerede de
geçmeli (tek-pencere = overfit, §3).

    # (anahtar gerekir: NASDAQ_DATA_LINK_API_KEY + pip install nasdaq-data-link)
    python -m research.smallcap_momentum
    python -m research.smallcap_momentum --windows 2016-01-01:2019-12-31,2020-01-01:2023-12-31 \
        --lookback 126 --skip 21 --hold 21 --top 20 --max-names 400
"""
from __future__ import annotations

import argparse
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.config import CONFIG
from nihai.data.sharadar import SharadarUnavailable, price_panel, universe_as_of
from research.xsect import run_momentum


def _parse_windows(arg: str) -> list[tuple[str, str]]:
    """'from:to,from:to' → [(from, to), ...]. İki-periyot dayanıklılık için ≥2 önerilir."""
    out: list[tuple[str, str]] = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise SystemExit(f"Geçersiz pencere {chunk!r} — biçim: FROM:TO (YYYY-MM-DD)")
        a, b = chunk.split(":", 1)
        out.append((a.strip(), b.strip()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Survivorship'siz small-cap momentum (Sharadar PIT)")
    ap.add_argument(
        "--windows",
        default="2016-01-01:2019-12-31,2020-01-01:2023-12-31",
        help="FROM:TO,FROM:TO — evren her pencerenin BAŞINDAN (PIT) kurulur",
    )
    ap.add_argument("--lookback", type=int, default=126, help="momentum penceresi (bar)")
    ap.add_argument("--skip", type=int, default=21, help="atlanan son bar (kısa-vade reversal)")
    ap.add_argument("--hold", type=int, default=21, help="rebalance aralığı (bar)")
    ap.add_argument("--top", type=int, default=20, help="tutulan en güçlü K sembol")
    ap.add_argument("--max-names", type=int, default=400,
                    help="evren büyüklüğü tavanı (API/hız) — PIT evrenden ilk N (alfabetik)")
    ap.add_argument("--all-caps", action="store_true",
                    help="small-cap filtresini KAPAT (tüm evren) — kıyas için")
    args = ap.parse_args()

    windows = _parse_windows(args.windows)
    cost = CONFIG.backtest.stock_slippage_pct
    smallcap = not args.all_caps

    print(f"=== Small-cap momentum (Sharadar PIT, survivorship'siz) | "
          f"{len(windows)} pencere | lookback={args.lookback} skip={args.skip} "
          f"hold={args.hold} top={args.top} ===")
    print("Hedef: edge_vs_buyhold > 0 İKİ pencerede de (momentum, small-cap betasını aşmalı).\n")

    try:
        results: list[tuple[str, dict]] = []
        for frm, to in windows:
            syms = universe_as_of(frm, smallcap_only=smallcap, limit=args.max_names)
            if not syms:
                print(f"--- {frm}→{to}: evren boş ---\n")
                continue
            panel = price_panel(syms, frm, to)
            if panel.empty:
                print(f"--- {frm}→{to}: fiyat verisi yok ---\n")
                continue
            m = run_momentum(panel, lookback=args.lookback, skip=args.skip,
                             hold=args.hold, top=args.top, cost_pct=cost)
            print(f"{'='*60}\n--- PENCERE: {frm}→{to} "
                  f"({panel.shape[1]} sembol, {panel.shape[0]} bar) ---\n{'='*60}")
            if not m:
                print("  (yeterli veri yok)\n")
                continue
            for k, v in m.items():
                print(f"  {k:<24} {v}")
            verdict = "KENAR VAR ✓" if m["edge_vs_buyhold_pct"] > 0 else "sadece beta (kenar yok)"
            print(f"  → {verdict}\n")
            results.append((f"{frm}→{to}", m))

        # İki-pencere dayanıklılık verdiği (§3 disiplini)
        if len(results) >= 2:
            all_positive = all(m["edge_vs_buyhold_pct"] > 0 for _, m in results)
            print("=" * 60)
            if all_positive:
                print("DAYANIKLILIK: TÜM pencerelerde edge_vs_buyhold > 0 → "
                      "small-cap momentum'da dayanıklı kenar ADAYI (PIT-temiz). "
                      "Yine de canlı paper ile teyit et.")
            else:
                print("DAYANIKLILIK: en az bir pencerede kenar yok → "
                      "dayanıklı DEĞİL (tek-pencere artefaktı, §3). Ship etme.")
            print("=" * 60)
    except SharadarUnavailable as e:
        print(e)
        sys.exit(0)


if __name__ == "__main__":
    main()
