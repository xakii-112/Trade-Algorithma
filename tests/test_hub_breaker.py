"""HubProvider devre kesici (Ek Tur 21) — davranış guard'ları.

ÖLÇÜM (kesicinin gerekçesi): hub KAPALIYKEN 10 ardışık `get_bars` = 20,30 sn
(2,02–2,06 sn/çağrı, her seferinde). Kesiciyle 2,03 sn (ilk çağrı yine sondalar).
Hub'ın kapalı olması istisna değil — watchdog kalkana kadar ~30 sn, 2026-07-18'de
10+ saat sürmüştü; o sürede taramanın HER sembolü bu bedeli ayrı ödüyordu.

Bu testler ağa çıkmaz: `_client` sahte nesneyle değiştirilir (conftest ağ guard'ı da
zaten dış bağlantıya izin vermez).
"""
from __future__ import annotations

import httpx
import pytest

from nihai.data.data_provider import HubProvider


@pytest.fixture(autouse=True)
def _temiz_kesici():
    """Kesici SINIF düzeyinde (süreç-geneli) — testler birbirine sızmasın."""
    HubProvider._down_until.clear()
    yield
    HubProvider._down_until.clear()


class _SahteIstemci:
    """Sayaçlı sahte httpx.Client — `hata` None ise 200 döner."""

    def __init__(self, hata: Exception | None = None, durum_hatasi: bool = False):
        self.hata = hata
        self.durum_hatasi = durum_hatasi
        self.cagri = 0

    def get(self, url, params=None):
        self.cagri += 1
        if self.hata is not None:
            raise self.hata
        return _SahteYanit(self.durum_hatasi)


class _SahteYanit:
    text = '{"columns":[],"index":[],"data":[]}'

    def __init__(self, durum_hatasi: bool):
        self._kotu = durum_hatasi

    def raise_for_status(self):
        if self._kotu:
            raise httpx.HTTPStatusError(
                "404", request=httpx.Request("GET", "http://127.0.0.1:8799/bars"),
                response=httpx.Response(404))


def _hub(client) -> HubProvider:
    h = HubProvider(base_url="http://127.0.0.1:8799")
    h._client = client
    return h


def test_baglanti_hatasi_kesiciyi_acar_ikinci_cagri_denemez():
    c = _SahteIstemci(hata=httpx.ConnectError("baglanti reddedildi"))
    h = _hub(c)
    assert h._get("/bars", {}) is None
    assert c.cagri == 1
    for _ in range(5):
        assert h._get("/bars", {}) is None
    assert c.cagri == 1, "kesici açıkken hub'a bir daha gidilmemeli"


def test_http_durum_hatasi_kesiciyi_ACMAZ():
    """404/500 tek bir ucun sorunudur — ayakta olan hub'ın tamamı kapatılmamalı."""
    c = _SahteIstemci(durum_hatasi=True)
    h = _hub(c)
    assert h._get("/options", {}) is None
    assert h._get("/options", {}) is None
    assert c.cagri == 2, "durum hatası kesiciyi açmamalı"


def test_basarili_cagri_kesiciyi_sifirlar():
    h = _hub(_SahteIstemci(hata=httpx.ConnectError("kapali")))
    h._get("/bars", {})
    assert h._breaker_open() is True
    h._client = _SahteIstemci()                 # hub geri geldi
    HubProvider._down_until.clear()             # kesici süresi doldu (zaman ilerledi)
    assert h._get("/bars", {}) is not None
    assert h._breaker_open() is False


def test_kesici_suresi_dolunca_yeniden_sondalar(monkeypatch):
    c = _SahteIstemci(hata=httpx.ConnectError("kapali"))
    h = _hub(c)
    h._get("/bars", {})
    assert c.cagri == 1 and h._breaker_open()
    # süreyi geçmişe al = pencere doldu
    HubProvider._down_until[h.base_url] = 0.0
    h._get("/bars", {})
    assert c.cagri == 2, "pencere dolunca hub yeniden denenmeli (kalıcı kapanma YOK)"


def test_breaker_sec_sifir_eski_davranisi_korur(monkeypatch):
    monkeypatch.setenv("HUB_BREAKER_SEC", "0")
    c = _SahteIstemci(hata=httpx.ConnectError("kapali"))
    h = _hub(c)
    for _ in range(3):
        h._get("/bars", {})
    assert c.cagri == 3, "HUB_BREAKER_SEC=0 → her çağrı hub'ı dener (eski davranış)"


def test_kesici_acikken_get_bars_bos_doner_failover_calisir():
    """Sözleşme (Madde 45): kesici açıkken de dönüş BOŞ DataFrame — dıştaki
    FailoverProvider bunu görüp yedeğe geçer. Kesici yalnız HIZLANDIRIR."""
    h = _hub(_SahteIstemci(hata=httpx.ConnectError("kapali")))
    ilk = h.get_bars("SPY", "1d", 220)
    assert ilk.empty
    ikinci = h.get_bars("SPY", "1d", 220)       # kesici açık
    assert ikinci.empty
    assert type(ikinci) is type(ilk)
