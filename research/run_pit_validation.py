"""
run_pit_validation.py — Survivorship-temiz "gerçek kenar" doğrulama bataryası (tek komut).

EDGE_FINDINGS §18'in operasyonel karşılığı: tüm fiyat/gösterge/ML alanı tükendi ve her
survivorship-temiz ölçümde kenar YOK; darboğaz TEKNİK değil VERİ. Bu script o veriyi
beslediğin an çalışacak üç honest adımı TEK yerden, preflight-kontrollü koşturur:

  Adım 2 — small-cap momentum (Sharadar PIT, survivorship'siz)   → research.smallcap_momentum
  Adım 3 — PEAD point-in-time (S&P500 as-of, ücretsiz PIT evren) → research.pead_pit_test
  Adım 4 — canlı paper isabet/kalibrasyon (survivorship'SİZ)     → prediction_tracker --report

Tasarım: hiçbir adım diğerini bloke etmez. Sharadar anahtarı yoksa Adım 2 net yönergeyle
ATLANIR (çökme değil); Adım 3/4 ücretsiz veriyle yine koşar. Her adım kendi alt-süreçinde
çalışır → birinin hatası diğerini düşürmez. Çıkış kodu 0 (rapor aracı, gate değil).

KULLANIM:
    python -m research.run_pit_validation                  # üçünü de dene
    python -m research.run_pit_validation --only 3,4       # yalnız ücretsiz adımlar
    python -m research.run_pit_validation --as-of 2021-01-01 --windows 2016-01-01:2019-12-31,2020-01-01:2023-12-31

ANAHTAR GELİNCE (Adım 2'yi açmak için):
    pip install nasdaq-data-link
    .env →  NASDAQ_DATA_LINK_API_KEY=<key>     (data.nasdaq.com → Sharadar Equity Bundle, SEP+TICKERS+SF1)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib.util import find_spec

# .env'i erken yükle (config zaten yapıyor ama bu script config'ten önce key'i okumalı)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

PY = sys.executable or "python"


def _run(label: str, module_args: list[str]) -> int:
    """Bir doğrulama adımını alt-süreçte koştur. Dönüş = alt-süreç çıkış kodu (rapor amaçlı)."""
    print("\n" + "=" * 78)
    print(f"▶ {label}")
    print("=" * 78)
    cmd = [PY, "-m", *module_args]
    print("  $ " + " ".join(cmd) + "\n")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        print("  (kullanıcı kesti)")
        return 130
    except Exception as e:  # alt-süreç başlatılamadı → adımı atla, bataryayı düşürme
        print(f"  HATA: adım başlatılamadı: {e}")
        return 1


def _sharadar_ready() -> tuple[bool, str]:
    """Adım 2 ön-koşulu: nasdaq-data-link paketi + API anahtarı. (hazır mı, gerekçe)."""
    key = os.getenv("NASDAQ_DATA_LINK_API_KEY") or os.getenv("QUANDL_API_KEY")
    pkg = find_spec("nasdaqdatalink") is not None or find_spec("nasdaq_data_link") is not None
    if not key:
        return False, "NASDAQ_DATA_LINK_API_KEY .env'de yok"
    if not pkg:
        return False, "nasdaq-data-link paketi kurulu değil (pip install nasdaq-data-link)"
    return True, "hazır"


def main() -> None:
    ap = argparse.ArgumentParser(description="Survivorship-temiz gerçek-kenar doğrulama bataryası")
    ap.add_argument("--only", default="2,3,4",
                    help="koşturulacak adımlar (virgülle): 2=smallcap-mom 3=PEAD-PIT 4=canlı-paper")
    ap.add_argument("--as-of", default="2022-01-01", help="Adım 3 PEAD evren üyelik tarihi")
    ap.add_argument("--end", default=None, help="Adım 3 bitiş (YYYY-MM-DD)")
    ap.add_argument("--limit", type=int, default=0, help="Adım 3 ilk N sembol (0=hepsi; deneme için küçük tut)")
    ap.add_argument("--windows", default="2016-01-01:2019-12-31,2020-01-01:2023-12-31",
                    help="Adım 2 PIT pencereleri (FROM:TO,FROM:TO)")
    ap.add_argument("--resolve", action="store_true",
                    help="Adım 4 öncesi vadesi gelen canlı tahminleri etiketle (--resolve)")
    args = ap.parse_args()
    steps = {s.strip() for s in args.only.split(",") if s.strip()}

    print("SURVIVORSHIP-TEMİZ GERÇEK-KENAR DOĞRULAMA BATARYASI (EDGE_FINDINGS §18)")
    print("Hedef: kenar 'bugünün kazananlarında' değil, o tarihte İŞLEM YAPSAYDIN da var mıydı?")
    summary: list[str] = []

    # --- Adım 2: small-cap momentum (Sharadar PIT) ---
    if "2" in steps:
        ready, why = _sharadar_ready()
        if ready:
            rc = _run("Adım 2 — Small-cap momentum (Sharadar PIT, survivorship'siz)",
                      ["research.smallcap_momentum", "--windows", args.windows])
            summary.append(f"Adım 2 (smallcap-mom) : çalıştı (rc={rc}) — çıktıdaki 'edge_vs_buyhold>0 İKİ pencere' kararına bak")
        else:
            print("\n" + "=" * 78)
            print("▶ Adım 2 — Small-cap momentum  [ATLANDI]")
            print("=" * 78)
            print(f"  Önkoşul yok: {why}")
            print("  Açmak için:\n    pip install nasdaq-data-link")
            print("    .env → NASDAQ_DATA_LINK_API_KEY=<key>   (data.nasdaq.com → Sharadar Equity Bundle)")
            summary.append(f"Adım 2 (smallcap-mom) : ATLANDI — {why}")

    # --- Adım 3: PEAD point-in-time (ücretsiz S&P500 as-of evren) ---
    if "3" in steps:
        a3 = ["research.pead_pit_test", "--as-of", args.as_of]
        if args.end:
            a3 += ["--end", args.end]
        if args.limit:
            a3 += ["--limit", str(args.limit)]
        rc = _run("Adım 3 — PEAD point-in-time (S&P500 as-of, ücretsiz PIT evren; yfinance earnings)",
                  a3)
        summary.append(f"Adım 3 (PEAD-PIT)     : çalıştı (rc={rc}) — removed-since isimlerdeki PF'e dikkat (§13.5)")

    # --- Adım 4: canlı paper isabet/kalibrasyon (survivorship'SİZ, lokal) ---
    if "4" in steps:
        a4 = ["nihai.execution.prediction_tracker", "--report"]
        if args.resolve:
            a4.insert(1, "--resolve")
        rc = _run("Adım 4 — Canlı paper isabet + kalibrasyon (survivorship'siz, lokal log)", a4)
        summary.append(f"Adım 4 (canlı-paper)  : çalıştı (rc={rc}) — isabet>%50 VE pozitif getiri ararız (§17)")

    print("\n" + "=" * 78)
    print("ÖZET")
    print("=" * 78)
    for line in summary:
        print("  " + line)
    print("\nKARAR KURALI (§3/§6 disiplini): bir aday ancak İKİ bağımsız pencerede/holdout'ta")
    print("kenar gösterirse ship edilir. Tek-pencere pozitifi = overfit; ship etme.")


if __name__ == "__main__":
    main()
