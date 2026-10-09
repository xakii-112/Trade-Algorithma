"""Gerçek-piyasa opsiyon varlığı — yfinance (.options vade listesi). Projenin asıl kaynağı."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def check(sym: str):
    for attempt in range(3):
        try:
            opts = yf.Ticker(sym).options
            return sym, len(opts or [])
        except Exception:
            if attempt == 2:
                return sym, -1  # hata
            time.sleep(1.5 * (attempt + 1))


def main() -> None:
    syms = json.load(open("data/watchlist.json", encoding="utf-8"))["symbols"]
    print(f"Toplam sembol: {len(syms)}")

    result: dict[str, int] = {}   # sembol -> vade sayısı (0 = opsiyon yok)
    errors: dict[str, str] = {}

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(check, s): s for s in syms}
        for f in as_completed(futs):
            sym, n = f.result()
            if n < 0:
                errors[sym] = "fetch_fail"
            else:
                result[sym] = n
            done += 1
            if done % 50 == 0:
                print(f"  ... {done}/{len(syms)}")

    with_opts = sorted([s for s, n in result.items() if n > 0])
    no_opts = sorted([s for s, n in result.items() if n == 0])
    err = sorted(errors)

    print(f"\n✅ Opsiyonu OLAN   : {len(with_opts)}")
    print(f"❌ Opsiyonu OLMAYAN: {len(no_opts)}")
    print(f"⚠️  Sorgu HATASI   : {len(err)}")
    print("\n=== OPSİYONU OLMAYAN (yfinance) ===")
    print(", ".join(no_opts) if no_opts else "(yok — hepsinde opsiyon var)")
    if err:
        print("\n=== HATA (tekrar denenebilir) ===")
        print(", ".join(err))

    json.dump(
        {"source": "yfinance", "with_options": with_opts,
         "without_options": no_opts, "errors": err},
        open("data/options_availability_yf.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=2,
    )
    print("\n💾 data/options_availability_yf.json yazıldı")


if __name__ == "__main__":
    main()
