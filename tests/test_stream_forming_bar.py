"""A1 regresyon — oluşmakta olan (yarı) barın atılması TZ-BAĞIMSIZ olmalı (look-ahead yok).

Madde 3: get_bars son (henüz kapanmamış) barı atmalı. Hata (düzeltildi): naive-LOCAL
`datetime.now()` ile naive-UTC bar ts kıyaslanıyordu → UTC+3'te yarı-dolu bar ATILMAZDI
(gelecek verisi sızardı = look-ahead). Düzeltme: `datetime.now(UTC).replace(tzinfo=None)`.

Bu test sağlayıcıyı DÜZENLEMEZ; yalnız kilitler. Modülün `datetime`'ı sahteyle değiştirilir:
`.now(tz)` doğru UTC anı verir; naked `.now()` YEREL offset'li naive an verir (yanlış-kullanım
simülasyonu). Düzeltilmiş kod naked dalı ASLA çağırmadığından sonuç yerel offset'ten bağımsız
kalır → hem UTC+3 hem UTC-4'te oluşan bar atılır, kapalı bar tutulur. Regresyonda (naked now)
UTC+3 senaryosu patlar."""
import datetime as _dt
from collections import deque

import pandas as pd

import nihai.data.alpaca_stream_provider as asp

# Dondurulmuş "şimdi" = 2025-06-02 14:30 UTC. 1h bar konvansiyonu (label=left):
#   14:00 barı [14:00,15:00) → 14:30'da HÂLÂ oluşuyor (atılmalı).
#   13:00 barı [13:00,14:00) → kapandı (tutulmalı).
_FROZEN_UTC = _dt.datetime(2025, 6, 2, 14, 30, tzinfo=_dt.UTC)

# Canlı 1m tampon → 1h'a toplanınca 13:00 ve 14:00 barlarını verir.
_MINUTES = [
    (_dt.datetime(2025, 6, 2, 13, 0), 100.0, 101.0, 99.0, 100.5, 1000),
    (_dt.datetime(2025, 6, 2, 13, 30), 100.5, 101.5, 99.5, 100.8, 1200),
    (_dt.datetime(2025, 6, 2, 14, 0), 100.8, 102.0, 100.0, 101.5, 1500),
    (_dt.datetime(2025, 6, 2, 14, 15), 101.5, 102.5, 101.0, 101.9, 900),
]


class _FakeRest:
    """REST seed — yalnız KAPALI 13:00 barını verir (oluşan 14:00 canlı stream'den gelir)."""

    def get_bars(self, symbol, interval, lookback):
        idx = pd.to_datetime(["2025-06-02 13:00:00"])
        return pd.DataFrame({"open": [100.0], "high": [101.0], "low": [99.0],
                             "close": [100.5], "volume": [1000]}, index=idx)

    def get_quote(self, *a, **k):
        return None

    def get_option_chain(self, *a, **k):
        return {}


def _fake_datetime(offset_hours: int):
    """Yerel offset'i offset_hours olan bir makineyi taklit eden sahte datetime sınıfı."""
    off = _dt.timedelta(hours=offset_hours)

    class _FakeDatetime:
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return _FROZEN_UTC.astimezone(tz)          # doğru: UTC-farkında an
            # naked now() = yerel duvar-saati (naive). Düzeltilmiş kod bunu ÇAĞIRMAZ;
            # burada olması regresyonda (naked now'a dönüş) testin patlamasını sağlar.
            return (_FROZEN_UTC + off).replace(tzinfo=None)

    return _FakeDatetime


def _run(monkeypatch, offset_hours: int) -> pd.DataFrame:
    monkeypatch.setattr(asp, "datetime", _fake_datetime(offset_hours))
    prov = asp.AlpacaStreamProvider(symbols=[], rest=_FakeRest())
    prov._minutes["AAPL"] = deque(_MINUTES)
    return prov.get_bars("AAPL", "1h")


def test_forming_bar_dropped_closed_kept_utc_plus_3(monkeypatch):
    res = _run(monkeypatch, 3)     # yerel = UTC+3 (naked now İLERİ → eski bug burada sızardı)
    assert pd.Timestamp("2025-06-02 14:00:00") not in res.index   # oluşan bar ATILDI
    assert pd.Timestamp("2025-06-02 13:00:00") in res.index       # kapalı bar TUTULDU


def test_forming_bar_dropped_closed_kept_utc_minus_4(monkeypatch):
    res = _run(monkeypatch, -4)    # yerel = UTC-4
    assert pd.Timestamp("2025-06-02 14:00:00") not in res.index
    assert pd.Timestamp("2025-06-02 13:00:00") in res.index


def test_forming_bar_drop_is_tz_independent(monkeypatch):
    # Aynı UTC anı; yalnız yerel offset (naked now) değişiyor → sonuç ÖZDEŞ olmalı.
    r_plus = _run(monkeypatch, 3)
    r_minus = _run(monkeypatch, -4)
    assert list(r_plus.index) == list(r_minus.index)
    assert pd.Timestamp("2025-06-02 14:00:00") not in r_plus.index
