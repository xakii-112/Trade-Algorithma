"""
concurrency.py — Paylaşılan portföy kilidi.

Sorun: web_dashboard ayrı bir thread'de (uvicorn) çalışırken flatten/close ile
portfolio.open_positions'ı değiştirebilir; aynı anda ana döngü update_all_* ile
aynı listeyi değiştirir. Tek bir RLock ile tüm portföy mutasyonları ve web
okumaları serileştirilir.

PositionManager mutasyon metodları ve web_dashboard bu kilidi kullanır.
RLock — aynı thread iç içe (örn. flatten → _close) çağırabilsin diye.
"""
from __future__ import annotations

import threading

PORTFOLIO_LOCK = threading.RLock()
