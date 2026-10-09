"""build_sector_map.py — Evren (S&P 500 + Nasdaq-100) için GICS sektör haritası üret.

Wikipedia tablolarındaki 'GICS Sector' sütununu parse eder (per-sembol yfinance
çağrısı YOK → hızlı, rate-limit'siz) ve data/sector_map.json'a yazar. get_sector()
bu dosyayı okur; böylece sektör-başına pozisyon tavanı (Madde 12) tüm evrende
gerçek sektörlere yayılır (eskiden ~%80 sembol 'UNKNOWN' tek kovaya düşüyordu).
"""
from __future__ import annotations

import json
import re
import sys

import pandas as pd

from nihai.config import DATA_DIR, SECTOR_MAP
from nihai.execution.screen import _read_html_tables

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

# GICS sektör adı → projedeki kısa kod (mevcut SECTOR_MAP kodlarıyla uyumlu)
GICS_TO_CODE = {
    "Information Technology": "TECH",
    "Communication Services": "COMM",
    "Consumer Discretionary": "DISCRETIONARY",
    "Consumer Staples": "STAPLES",
    "Financials": "FINANCE",
    "Health Care": "HEALTH",
    "Energy": "ENERGY",
    "Industrials": "INDUSTRIALS",
    "Materials": "MATERIALS",
    "Real Estate": "REAL_ESTATE",
    "Utilities": "UTILITIES",
}


def _norm(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")


def _extract_sym_sector(tables: list[pd.DataFrame]) -> dict[str, str]:
    out: dict[str, str] = {}
    for t in tables:
        cols = {str(c).lower(): c for c in t.columns}
        sym_col = next((cols[k] for k in cols if "ticker" in k or "symbol" in k), None)
        sec_col = next((cols[k] for k in cols
                        if "gics sector" in k or k == "sector"), None)
        if sym_col is None or sec_col is None:
            continue
        for sym, sec in zip(t[sym_col].tolist(), t[sec_col].tolist(), strict=True):
            s = _norm(sym)
            sec = re.sub(r"\s+", " ", str(sec)).strip()
            if not s or s == "NAN" or len(s) > 6:
                continue
            code = GICS_TO_CODE.get(sec)
            if code:
                out[s] = code
    return out


def main() -> int:
    mapping: dict[str, str] = {}
    for name, url in [
        ("S&P 500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
        ("Nasdaq-100", "https://en.wikipedia.org/wiki/Nasdaq-100"),
    ]:
        try:
            m = _extract_sym_sector(_read_html_tables(url))
            print(f"{name}: {len(m)} sembol sektörü çekildi")
            mapping.update(m)
        except Exception as e:
            print(f"{name}: ÇEKİLEMEDİ ({e})")

    # Elle tanımlı SECTOR_MAP (ETF/INDEX/VOL + override) HER ZAMAN öncelikli
    mapping.update(SECTOR_MAP)

    out_file = DATA_DIR / "sector_map.json"
    out_file.write_text(json.dumps(mapping, ensure_ascii=False, indent=2,
                                   sort_keys=True), encoding="utf-8")

    from collections import Counter
    dist = Counter(mapping.values())
    print(f"\n✅ Toplam {len(mapping)} sembol → {out_file}")
    print("Sektör dağılımı:", dict(sorted(dist.items(), key=lambda x: -x[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
