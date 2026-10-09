"""A2 — hub /options → HubProvider round-trip: all_expiries hub sınırında düşmemeli.

DATA_PROVIDER=hub'ta opsiyon vade seçimi (main.py:227 `chain['all_expiries']`) hub'ın
opsiyon uç-noktasından beslenir. all_expiries iki yarıdan birinde düşerse — hub_server
serileştirmesi VEYA HubProvider ayrıştırması — opsiyon seçimi sessizce tek vadeye çöker
(silent break). Bu test her iki yarıyı tek round-trip'te alıştırır (ağ yok: sahte
upstream + sahte _get). Regresyonda all_expiries None döner → assert patlar."""
import pandas as pd

from nihai.data import hub_server
from nihai.data.data_provider import HubProvider

_ALL_EXP = ["2025-01-17", "2025-02-21", "2025-03-21"]


class _FakeUpstream:
    """hub_server.provider() yerine — all_expiries'li tam bir opsiyon zinciri döndürür."""

    def get_option_chain(self, symbol, expiry=None):
        calls = pd.DataFrame({"strike": [100.0, 105.0], "bid": [1.2, 0.6]})
        puts = pd.DataFrame({"strike": [100.0, 95.0], "bid": [1.1, 0.5]})
        return {"expiry": "2025-01-17", "all_expiries": _ALL_EXP,
                "calls": calls, "puts": puts, "underlying_price": 101.25}


class _FakeResp:
    """httpx.Response yerine — HubProvider._get sonucundan yalnız .json() kullanılır."""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_all_expiries_round_trips_hub_to_hubprovider(monkeypatch):
    # 1) hub_server.options() GERÇEK serileştirmesini çalıştır (sahte upstream ile).
    monkeypatch.setattr(hub_server, "provider", lambda: _FakeUpstream())
    hub_json = hub_server.options("AAPL", None)
    assert hub_json.get("all_expiries") == _ALL_EXP        # hub sınırı düşürmedi

    # 2) Aynı JSON'u HubProvider.get_option_chain GERÇEK ayrıştırmasına ver (sahte _get).
    hp = HubProvider(base_url="http://test-hub")
    monkeypatch.setattr(hp, "_get", lambda path, params: _FakeResp(hub_json))
    chain = hp.get_option_chain("AAPL")

    assert chain["all_expiries"] == _ALL_EXP               # HubProvider tarafı da korudu
    # sanity: zincirin geri kalanı da round-trip'te bozulmadan geldi
    assert chain["expiry"] == "2025-01-17"
    assert chain["underlying_price"] == 101.25
    assert not chain["calls"].empty and not chain["puts"].empty


def test_options_empty_chain_returns_empty(monkeypatch):
    # Zincir boşsa {} döner (all_expiries eklenmesi bu davranışı bozmamalı).
    class _Empty:
        def get_option_chain(self, symbol, expiry=None):
            return {}

    monkeypatch.setattr(hub_server, "provider", lambda: _Empty())
    assert hub_server.options("AAPL", None) == {}
