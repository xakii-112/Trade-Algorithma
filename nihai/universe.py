"""universe.py — paylaşılan sembol evrenleri (tek doğruluk kaynağı)."""
from __future__ import annotations

# Sabit, likit, sektör-çeşitli sepet (sonuç tekrarlanabilir olsun diye sabit).
# Eskiden bench.py'de tanımlıydı; production da kullandığı için buraya taşındı.
BASKET = [
    "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "GOOGL", "META",
    "SPY", "QQQ", "JPM", "XOM", "GE", "AVGO", "NFLX", "COST",
]
