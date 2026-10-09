"""alpaca_history testleri — ağ YOK (fetch monkeypatch'lenir), cache geçici dizinde.

_clamp_end (SIP-güvenli kırpma), cache yaz/oku turu, pencere kırpma ve
load_bars 'alpaca_hist' dalının period→start çevirisi doğrulanır.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import nihai.data.alpaca_history as ah


@pytest.fixture(autouse=True)
def _tmp_hist_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ah, "HIST_DIR", tmp_path / "alpaca_hist")
    yield


def _bars(start: str = "2021-07-01", n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    close = 50 + np.cumsum(rng.normal(0, 0.4, n))
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1e6}, index=idx)


# ------------------------------------------------------------- _clamp_end
def test_clamp_end_sip_guvenli():
    # None → şimdi-16dk civarı (asla gelecekte değil)
    e = ah._clamp_end(None)
    assert e <= datetime.now(UTC) - timedelta(minutes=15)
    # Gelecek tarih → kırpılır
    e2 = ah._clamp_end(datetime.now(UTC) + timedelta(days=1))
    assert e2 <= datetime.now(UTC) - timedelta(minutes=15)
    # Geçmiş tarih → dokunulmaz
    past = datetime(2024, 1, 1, tzinfo=UTC)
    assert ah._clamp_end(past) == past


# ------------------------------------------------------- cache + pencere
def test_get_history_cache_turu(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(symbol, interval, start, end, feed):
        calls["n"] += 1
        return _bars()

    monkeypatch.setattr(ah, "_fetch", fake_fetch)
    df1 = ah.get_history("AAPL", "1d", start="2021-07-01")
    assert not df1.empty and calls["n"] == 1
    # İkinci çağrı cache'ten gelmeli (bugün yazıldı + start'ı kapsıyor)
    df2 = ah.get_history("AAPL", "1d", start="2021-07-01")
    assert calls["n"] == 1
    assert len(df2) == len(df1)
    # Daha ESKİ start istenirse cache kapsamıyor → yeniden çekilir
    ah.get_history("AAPL", "1d", start="2019-01-01")
    assert calls["n"] == 2


def test_get_history_pencere_kirpma(monkeypatch):
    monkeypatch.setattr(ah, "_fetch", lambda *a, **k: _bars(start="2020-01-01", n=1500))
    df = ah.get_history("MSFT", "1d", start="2022-01-01", end="2023-01-01", use_cache=False)
    assert not df.empty
    assert df.index.min() >= pd.Timestamp("2022-01-01")
    assert df.index.max() <= pd.Timestamp("2023-01-01")


def test_get_history_bos_negatif_cache(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(*a, **k):
        calls["n"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(ah, "_fetch", fake_fetch)
    assert ah.get_history("DELISTX", "1d", start="2021-07-01").empty
    # Boş sonuç da cache'lenir → aynı gün ikinci deneme Alpaca'ya gitmez
    assert ah.get_history("DELISTX", "1d", start="2021-07-01").empty
    assert calls["n"] == 1


def test_get_history_bos_cache_pencere_kapsami(monkeypatch):
    """Denetim Tur-11: boş-negatif cache yalnız KAPSADIĞI pencereye servis edilir.
    start=2022 için boş çıkan sembol, start=2016 isteyen PIT koşusunda yeniden
    çekilmeli (aksi hâlde delist sembolleri sessizce düşer → survivorship sapması)."""
    calls = {"n": 0}

    def fake_fetch(*a, **k):
        calls["n"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(ah, "_fetch", fake_fetch)
    assert ah.get_history("DELISTY", "1d", start="2022-01-01").empty
    assert calls["n"] == 1
    # Daha GENİŞ (erken) pencere → boş cache kapsamıyor → yeniden çekim
    assert ah.get_history("DELISTY", "1d", start="2016-01-01").empty
    assert calls["n"] == 2
    # İkinci boş çekim start=2016 metasını yazdı → aynı/dar pencere artık cache'ten
    assert ah.get_history("DELISTY", "1d", start="2023-06-01").empty
    assert calls["n"] == 2
    # Eski-format boş cache (metasız) kapsamı kanıtlayamaz → bir kez yeniden çekilir
    ah._empty_meta_path("DELISTY", "1d").unlink()
    assert ah.get_history("DELISTY", "1d", start="2023-06-01").empty
    assert calls["n"] == 3


def test_get_history_hata_cachelenmez(monkeypatch):
    """BULGU-14: geçici hata (None) negatif-cache'e yazılMAZ — sonraki çağrı yeniden dener,
    başarılı sonuç normal cache turuna döner."""
    calls = {"n": 0}

    def fake_fetch(*a, **k):
        calls["n"] += 1
        return None if calls["n"] == 1 else _bars()

    monkeypatch.setattr(ah, "_fetch", fake_fetch)
    # 1. çağrı: hata → boş df döner ama cache dosyası OLUŞMAZ
    assert ah.get_history("AAPL", "1d", start="2021-07-01").empty
    assert not ah._cache_path("AAPL", "1d").exists()
    # 2. çağrı: yeniden dener, veri gelir ve cache'lenir
    df = ah.get_history("AAPL", "1d", start="2021-07-01")
    assert not df.empty and calls["n"] == 2
    # 3. çağrı: artık cache'ten
    ah.get_history("AAPL", "1d", start="2021-07-01")
    assert calls["n"] == 2


def test_get_history_hata_cachesiz_yol(monkeypatch):
    """use_cache=False yolunda da hata (None) boş df olarak döner, çökmez."""
    monkeypatch.setattr(ah, "_fetch", lambda *a, **k: None)
    assert ah.get_history("AAPL", "1d", start="2021-07-01", use_cache=False).empty


# --------------------------------------------------- load_bars entegrasyonu
def test_load_bars_alpaca_hist_period_cevirisi(monkeypatch):
    seen = {}

    def fake_get_history(symbol, interval, *, start, end=None, **kw):
        seen["start"] = start
        return _bars()

    monkeypatch.setattr("nihai.data.alpaca_history.get_history", fake_get_history)
    from nihai.execution.backtest import load_bars
    df = load_bars("AAPL", "1d", period="5y", provider_name="alpaca_hist")
    assert not df.empty
    # period=5y → start ~5 yıl önce (string YYYY-MM-DD)
    start_ts = pd.Timestamp(seen["start"])
    yil_farki = (pd.Timestamp.now() - start_ts).days / 365
    assert 4.8 <= yil_farki <= 5.3


# ------------------------------------------------- BULGU-19: bölünme düzeltmesi
def test_adjustment_varsayilan_all_ve_env_override(monkeypatch):
    """Varsayılan 'all' (düzeltilmiş); env override edilir; geçersiz değer 'all'e düşer."""
    monkeypatch.delenv("ALPACA_HIST_ADJUSTMENT", raising=False)
    assert ah._adjustment() == "all"
    for kip in ("raw", "split", "dividend", "all"):
        monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", kip.upper())   # büyük/küçük harf duyarsız
        assert ah._adjustment() == kip
    monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", "sacma")
    assert ah._adjustment() == "all"


def test_cache_yolu_duzeltme_kipini_ayirir(monkeypatch):
    """Ham ve düzeltilmiş barlar AYNI dosyaya yazılmamalı — karışırsa sessizce
    bozuk fiyat serisi servis edilir. 'raw' geriye-uyum için eski adı korur."""
    monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", "raw")
    ham = ah._cache_path("AAPL", "1d")
    monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", "all")
    duz = ah._cache_path("AAPL", "1d")
    assert ham.name == "AAPL_1d.csv"          # eski format adı DEĞİŞMEDİ
    assert duz.name == "AAPL_1d_all.csv"
    assert ham != duz
    # boş-cache metaları da ayrışır (biri diğerinin kapsamını iddia edemez)
    monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", "raw")
    assert ah._empty_meta_path("AAPL", "1d").name == "AAPL_1d.empty.json"
    monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", "all")
    assert ah._empty_meta_path("AAPL", "1d").name == "AAPL_1d_all.empty.json"


def test_fetch_istege_adjustment_yazar(monkeypatch):
    """Asıl regresyon: StockBarsRequest'e adjustment GİDİYOR mu?
    Gitmezse SDK None gönderir → API 'raw' döner → bölünmeler düzeltilmez."""
    pytest.importorskip("alpaca.data.requests")
    from alpaca.data.enums import Adjustment
    gorulen = {}

    class _FakeClient:
        def get_stock_bars(self, req):
            gorulen["adjustment"] = req.adjustment
            raise RuntimeError("dur")          # normalize'a gitmeden çık

    monkeypatch.setattr(ah, "_get_client", lambda: _FakeClient())
    monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", "all")
    ah._fetch("AAPL", "1d", datetime(2021, 7, 1, tzinfo=UTC),
              datetime(2022, 1, 1, tzinfo=UTC), "sip")
    assert gorulen["adjustment"] == Adjustment.ALL

    monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", "raw")
    ah._fetch("AAPL", "1d", datetime(2021, 7, 1, tzinfo=UTC),
              datetime(2022, 1, 1, tzinfo=UTC), "sip")
    assert gorulen["adjustment"] == Adjustment.RAW
