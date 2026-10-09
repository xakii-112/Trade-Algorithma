"""pit_universe point-in-time S&P500 yeniden-kurulum testleri (survivorship düzeltmesi).

Kritik: as_of tarihinden SONRA eklenen sembol o tarihte YOKTU (çıkarılmalı); sonradan
çıkarılan sembol o tarihte VARDI (eklenmeli). Yanlışsa survivorship-şişkin evren →
backtest kenarı abartılır. _current_and_changes monkeypatch'lenir → ağ/Wikipedia yok."""
import pandas as pd

import nihai.data.pit_universe as pit
from nihai.data.pit_universe import _norm, removed_since, sp500_as_of


def test_norm_normalizes_ticker():
    assert _norm("  brk.b ") == "BRK-B"
    assert _norm("aapl") == "AAPL"


def _patch(monkeypatch, current, changes_rows):
    changes = pd.DataFrame(changes_rows)
    changes["date"] = pd.to_datetime(changes["date"])
    monkeypatch.setattr(pit, "_current_and_changes", lambda: (current, changes))


def test_symbol_added_after_asof_excluded(monkeypatch):
    _patch(monkeypatch, ["AAPL", "MSFT", "NEWCO"],
           [{"date": "2023-06-01", "added": "NEWCO", "removed": "OLDCO"}])
    members = sp500_as_of("2022-01-01")
    assert "NEWCO" not in members       # 2023'te eklendi → 2022'de yoktu


def test_symbol_removed_after_asof_included(monkeypatch):
    _patch(monkeypatch, ["AAPL", "MSFT", "NEWCO"],
           [{"date": "2023-06-01", "added": "NEWCO", "removed": "OLDCO"}])
    members = sp500_as_of("2022-01-01")
    assert "OLDCO" in members           # 2023'te çıkarıldı → 2022'de vardı


def test_change_before_asof_has_no_effect(monkeypatch):
    _patch(monkeypatch, ["AAPL", "MSFT"],
           [{"date": "2021-01-01", "added": "FOO", "removed": "BAR"}])
    members = sp500_as_of("2022-01-01")
    # as_of'tan ÖNCEKİ değişiklik geri-sarılmaz → mevcut liste aynen
    assert members == ["AAPL", "MSFT"]
    assert "BAR" not in members


def test_removed_since_lists_survivorship_names(monkeypatch):
    _patch(monkeypatch, ["AAPL", "MSFT", "NEWCO"],
           [{"date": "2023-06-01", "added": "NEWCO", "removed": "OLDCO"}])
    assert removed_since("2022-01-01") == ["OLDCO"]


def test_same_ticker_add_then_remove_after_asof_nonsorted_rows(monkeypatch):
    """A3: aynı ticker as_of'tan SONRA önce eklenip sonra çıkarıldı; satırlar tarih-AZALAN
    DEĞİL. Geri-sarma tarihe göre sıralanmazsa yanlış üyelik verir.

    Timeline (cutoff=2022-01-01): ZZZ 2023-01'de eklendi, 2023-06'da çıkarıldı → bugün
    endekste YOK, 2022'de de YOK. Naive satır-sırası (artan tarih) işlerse: 'added' no-op
    (ZZZ üyelikte değil) → 'removed' ZZZ'yi geri EKLER = yanlış. Azalan sıra: önce 'removed'
    ekler, sonra 'added' çıkarır → doğru (ZZZ yok). Satırlar bilinçli artan tarih sırasında."""
    _patch(monkeypatch, ["AAPL", "MSFT"],
           [{"date": "2023-01-01", "added": "ZZZ", "removed": ""},
            {"date": "2023-06-01", "added": "", "removed": "ZZZ"}])
    members = sp500_as_of("2022-01-01")
    assert "ZZZ" not in members
    assert members == ["AAPL", "MSFT"]
