"""
pit_universe.py — Point-in-time (tarihsel) S&P 500 üyeliği yeniden inşası.

Survivorship bias düzeltmesi: backtest evrenini BUGÜNKÜ endeks üyeliğiyle (hayatta kalan
kazananlar) değil, test penceresinin BAŞINDAKİ üyelikle kurmak gerekir — o snapshot,
sonradan çıkarılan/iflas eden/satın alınan isimleri (survivorship'in attığı tam o
şirketleri) içerir.

Yöntem: Wikipedia'nın "List of S&P 500 companies" sayfasındaki (a) güncel üyeler tablosu
+ (b) tarihli değişiklikler tablosu (Added/Removed/Effective Date). D tarihindeki üyeliği,
güncel kümeden D'den SONRAki değişiklikleri geri-sararak elde ederiz:
  - D'den sonra EKLENEN ticker → D'de yoktu → kümeden çıkar.
  - D'den sonra ÇIKARILAN ticker → D'de vardı → kümeye geri ekle.

SINIR: bu hâlâ TAM survivorship-free değil — Wikipedia değişiklik tablosu ~2000'lerden
beri makul ama eksiksiz değil; ayrıca çıkarılan tickerların FİYAT/EARNINGS verisi yfinance'te
çoğu zaman (satın alma → delisting tarihine kadar) bulunur ama iflaslarda eksik olabilir.
Yine de "bugünün kazananları" evreninden ÇOK daha adil. Gerçek altın standart: Sharadar/CRSP.
"""
from __future__ import annotations

import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import pandas as pd
from loguru import logger

from nihai.execution.screen import _read_html_tables

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _norm(t: str) -> str:
    return str(t).strip().upper().replace(".", "-")


# BULGU-86-3: değişiklik tablosu 2026-08-11'de ayrı maddeye TAŞINDI (revizyon notu: "move to
# [[Historical components of the S&P 500]]"). Ana maddede `tabs[1]` sessizce bir navbox oldu ve
# `_current_and_changes` o günden beri KeyError('date') ile düşüyordu → PIT evren kuran HER
# kurucu kırıktı. Tablo artık KONUMLA değil kolon İMZASIYLA aranır, önce ana maddede sonra
# yeni maddede.
_SP500_CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"


def _degisiklik_tablosu(tabs: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Effective Date + Added + Removed kolonlarını birlikte taşıyan İLK tablo (yoksa None)."""
    for t in tabs:
        ust = [str(c[0] if isinstance(c, tuple) else c).lower() for c in t.columns]
        if (any("effective" in u for u in ust) and any("added" in u for u in ust)
                and any("removed" in u for u in ust)):
            return t
    return None


def _current_and_changes() -> tuple[list[str], pd.DataFrame]:
    tabs = _read_html_tables(_SP500_URL)
    current = tabs[0]
    sym_col = "Symbol" if "Symbol" in current.columns else current.columns[0]
    cur = [_norm(s) for s in current[sym_col].tolist() if str(s).strip()]
    raw = _degisiklik_tablosu(tabs)
    if raw is None:
        raw = _degisiklik_tablosu(_read_html_tables(_SP500_CHANGES_URL))
    if raw is None:
        # Değişiklik tablosu olmadan geri-sarma YAPILAMAZ; güncel kümeyle devam etmek
        # survivorship'i sessizce geri getirirdi (ACI DERS #1) → yüksek sesle dur.
        raise RuntimeError("S&P 500 değişiklik tablosu bulunamadı (ana madde + "
                           f"{_SP500_CHANGES_URL}) — PIT üyelik KURULAMAZ")
    # MultiIndex/düz kolonları pozisyonel tarayıp temiz bir DataFrame kur.
    out = {}
    for col in raw.columns:
        top = str(col[0] if isinstance(col, tuple) else col).lower()
        sub = str(col[-1] if isinstance(col, tuple) else col).lower()
        series = raw[col]
        if "effective" in top or (top == sub and "date" in top):
            out["date"] = series
        elif "added" in top and "ticker" in sub:
            out["added"] = series
        elif "removed" in top and "ticker" in sub:
            out["removed"] = series
    eksik = {"date", "added", "removed"} - set(out)
    if eksik:
        # Tur-86 (Gemini incelemesi, doğrulandı): alt başlık "Ticker" dışında bir ada dönerse
        # `added`/`removed` HİÇ kurulmaz, `_geri_sar` hiçbir şeyi geri sarmaz ve GÜNCEL küme
        # geçmiş üyelik diye döner = SESSİZ survivorship. Hata vermeyen bu dal, KeyError'dan
        # daha tehlikelidir → yüksek sesle dur.
        raise RuntimeError(f"S&P 500 değişiklik tablosunda {sorted(eksik)} kolonu kurulamadı "
                           f"(başlıklar: {[str(c) for c in raw.columns][:8]}) — PIT üyelik KURULAMAZ")
    changes = pd.DataFrame(out)
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
    return cur, changes.dropna(subset=["date"])


def sp500_as_of(as_of: str) -> list[str]:
    """`as_of` (YYYY-MM-DD) tarihindeki S&P 500 üyeliğini döndür (çıkarılanlar dahil)."""
    cur, changes = _current_and_changes()
    result = _geri_sar(cur, changes, as_of)
    removed_since = sorted(set(result) - set(cur))
    logger.info(f"[pit] S&P500 @ {as_of}: {len(result)} sembol "
                f"({len(removed_since)} bugün artık YOK — survivorship düzeltmesi)")
    return result


def sp500_uyelik(tarihler: list[str]) -> dict[str, list[str]]:
    """Birden çok tarih için PIT üyelik; Wikipedia'yı TEK kez çeker (Tur-86).

    Aylık kesit kuran bir aile `sp500_as_of`'u her ay sonu için çağırsaydı aynı iki
    tabloyu tarih başına yeniden indirirdi (37 ay = 37 istek). Geri-sarma gövdesi
    `sp500_as_of` ile ORTAKTIR (`_geri_sar`) — iki kopya birbirinden sessizce ayrışamaz."""
    cur, changes = _current_and_changes()
    return {t: _geri_sar(cur, changes, t) for t in tarihler}


def _geri_sar(cur: list[str], changes: pd.DataFrame, as_of: str) -> list[str]:
    """Güncel küme + değişiklik tablosundan `as_of` üyeliği (ağ YOK)."""
    members = set(cur)
    cutoff = pd.Timestamp(as_of)
    # D'den sonraki değişiklikleri geri-sar — en YENİ değişiklikten en eskiye (tarih
    # AZALAN). Aksi hâlde aynı ticker'ın ardışık add/remove'u yanlış sırada işlenip
    # üyelik bozulur (Wikipedia scrape satır sırası garantisiz). Aynı-gün bağı için
    # ikincil deterministik anahtar (added/removed) → sonuç scrape sırasından bağımsız.
    future = changes[changes["date"] > cutoff]
    _sort_cols = ["date"] + [c for c in ("added", "removed") if c in future.columns]
    future = future.sort_values(by=_sort_cols, ascending=False, kind="mergesort")
    for _, row in future.iterrows():
        added = _norm(row["added"]) if "added" in row and pd.notna(row.get("added")) else ""
        removed = _norm(row["removed"]) if "removed" in row and pd.notna(row.get("removed")) else ""
        if added and added in members:
            members.discard(added)      # sonradan eklendi → o tarihte yoktu
        if removed:
            members.add(removed)        # sonradan çıkarıldı → o tarihte vardı
    members = {m for m in members if m and m != "NAN" and len(m) <= 6}
    return sorted(members)


def removed_since(as_of: str) -> list[str]:
    """`as_of`'ta endekste olup BUGÜN olmayan semboller (survivorship-düzeltme isimleri)."""
    cur, _ = _current_and_changes()
    return sorted(set(sp500_as_of(as_of)) - set(_norm(s) for s in cur))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Point-in-time S&P 500 üyeliği")
    ap.add_argument("--as-of", default="2022-01-01", help="YYYY-MM-DD")
    args = ap.parse_args()
    syms = sp500_as_of(args.as_of)
    cur, _ = _current_and_changes()
    gone = sorted(set(syms) - set(_norm(s) for s in cur))
    print(f"\nS&P500 @ {args.as_of}: {len(syms)} sembol")
    print(f"Bugün artık endekste OLMAYAN ({len(gone)}): {', '.join(gone)}")
