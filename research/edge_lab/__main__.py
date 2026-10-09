"""`python -m research.edge_lab` — kenar-keşif programının güncel durumu.

CLI neden `ledger.py`'nin İÇİNDE değil: `__init__.py` → `harness` → `ledger` zinciri
yüzünden `python -m research.edge_lab.ledger` modülü İKİ KEZ yükler (runpy önce paketi
import eder, sonra modülü `__main__` olarak çalıştırır) ve `RuntimeWarning` basar.
Watchdog/zamanlanmış görev çıktısında bu gürültüdür; paket giriş noktası temiz yoldur.
"""
from __future__ import annotations

from nihai.console import ensure_utf8_stdout
from research.edge_lab import ledger

if __name__ == "__main__":
    ensure_utf8_stdout()
    print(ledger.summary())
