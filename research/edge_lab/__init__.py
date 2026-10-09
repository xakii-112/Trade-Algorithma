"""
research.edge_lab — KENAR-KEŞİF PROGRAMI (Tur-16 WP-E, docs/AI_V3_DERIN_BEYIN_PLAN §WP-E).

Dürüstlük çerçevesi (pazarlık edilemez, plan §WP-E'den): mevcut ölçümler "BU veri + BU
sinyal ailelerinde kenar yok" dedi — "hiçbir yerde kenar yok" demedi. Bu paket kenarı
GARANTİ ETMEZ; ettiği şey aramayı sistematikleştirmek ve her adayı AYNI sert kapılardan
(purged CV → time+symbol çift-bölme kapı → DSR → exec-lag-1 duyarlılık) geçirmektir.
`harness.run_candidate` HİÇBİR aday için "kenar var" YAZMAZ — en iyi ihtimalle
"TÜM KAPILAR GEÇİLDİ — canlı gölge doğrulaması gerekli" der (ACI DERS #1-#7 disiplini).

KARAR-DIŞI/ARAŞTIRMA: bu paket `nihai.signals`/`nihai.strategy`/META_GATE/execution'a
İMPORT BAĞI ALMAZ; yalnız araştırma-düzeyinde bağımsız CSV veri setleri üzerinde çalışır.
"""
from __future__ import annotations

from research.edge_lab.harness import GateResult, run_candidate

__all__ = ["GateResult", "run_candidate"]
