"""Borsa takvimi testleri — ET duvar-saati (timezone tutarlılığı)."""
from datetime import datetime

from nihai.data.market_calendar import now_et, now_et_naive


def test_now_et_naive_is_naive_et_walltime():
    # BUG fix: tarih-bazlı mantık (wash-sale/gün-içi/T+1/günlük reset) ET tarihine göre
    # olmalı; yerel UTC+3'te gece-yarısı sonrası tarih kaymasın diye now_et_naive kullanılır.
    aware = now_et()
    naive = now_et_naive()
    assert naive.tzinfo is None                      # naive (sistem geneliyle uyumlu)
    assert isinstance(naive, datetime)
    # Aynı ET duvar-saati değeri (saniye toleransı), yerel saat DEĞİL
    assert abs((naive - aware.replace(tzinfo=None)).total_seconds()) < 2
