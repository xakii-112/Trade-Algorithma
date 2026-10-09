"""console.py — CLI çıktısının kodlama güvenliği (TEK doğruluk kaynağı, Tur-24).

Neden: Windows'ta `python -m ...` çıktısı bir dosyaya/pipe'a yönlendirildiğinde stdout
konsol kod sayfasına düşer (bu makinede **cp1254**). Kaynakta gayet masum duran `→ ⚠ Δ ↓
λ σ ≈ 📈` gibi karakterler cp1254'te YOK → ilk yazdırmada `UnicodeEncodeError` ve süreç
ÇÖKER. Türkçe harfler (ğ ş ı İ ç ö ü) cp1254'te var, bu yüzden sorun yıllarca görünmez
kalır ve yalnız ok/emoji/matematik sembolü basan satırda patlar.

Bu tam olarak izleme altındaki servislerin çalışma biçimidir (watchdog/zamanlanmış görev
çıktıyı `logs\\*.log`'a yönlendirir), yani hata yolu ETKİLEŞİMLİ kullanımda değil, tam da
gözetimsiz üretim yolunda tetiklenir.

Tur-24'te ÖLÇÜLDÜ (varsayılmadı): `python -m nihai.ai.selfcheck --report > dosya` gerçekten
`UnicodeEncodeError: 'charmap' codec can't encode character '\\u26a0'` ile çöküyordu.

Proje zaten 59 dosyada aynı 5 satırı elle kopyalamıştı; bu modül o deseni tek yerde toplar
(`atomic_io.py` ile aynı gerekçe). Yeni giriş noktaları `ensure_utf8_stdout()` çağırmalı —
`tests/test_console_encoding.py` bunu her `__main__` modülü için kilitler.
"""
from __future__ import annotations

import sys


def ensure_utf8_stdout() -> None:
    """stdout/stderr'i UTF-8'e çevir (yalnız Windows'ta gerekli, her yerde zararsız).

    Fail-soft: `reconfigure` olmayan akışlarda (pytest `capsys`, bazı pipe sarmalayıcıları)
    sessizce atlanır — kodlama güvenliği bir KOLAYLIKTIR, çağıranı çökertmemelidir."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass  # capsys/pipe: reconfigure yoksa da çalış
