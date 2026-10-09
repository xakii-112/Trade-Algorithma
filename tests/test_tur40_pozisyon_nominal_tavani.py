"""Tur-40 — MAX_POSITION_NOTIONAL: pozisyon başına nominal tavanı.

Tavan bir BÜYÜKLÜK tavanıdır: cap-aware boyutlama pozisyonu reddetmez, sığacak
şekilde küçültür. 0 = kapalı (eski davranış).
"""
import importlib
import os

import pytest


def _fresh(monkeypatch, value: str):
    monkeypatch.setenv("MAX_POSITION_NOTIONAL", value)
    import nihai.config as cfg
    importlib.reload(cfg)
    import nihai.risk.risk_manager as rm
    importlib.reload(rm)
    return rm


@pytest.fixture(autouse=True)
def _restore():
    yield
    os.environ.pop("MAX_POSITION_NOTIONAL", None)
    import nihai.config as cfg
    importlib.reload(cfg)
    import nihai.risk.risk_manager as rm
    importlib.reload(rm)


def _size(rm, equity=1_000_000.0, entry=100.0, stop=99.0):
    st = rm.PortfolioState(equity=equity, ath_equity=equity, cash_balance=equity)
    return rm.RiskManager(st).size_stock(entry, stop)


def test_tavan_kapaliyken_buyuk_pozisyon(monkeypatch):
    r = _size(_fresh(monkeypatch, "0"))
    assert r.approved
    assert r.sizing["notional"] > 3000, "tavan kapalı → risk-hedef boyut korunmalı"


def test_tavan_3000_kucultur_reddetmez(monkeypatch):
    r = _size(_fresh(monkeypatch, "3000"))
    assert r.approved, f"tavan reddetmemeli: {r.reason}"
    assert r.sizing["notional"] <= 3000
    assert r.sizing["shares"] == 30          # 3000 / 100
    assert r.sizing["downsized"] is True


def test_tavan_birim_fiyatindan_kucukse_sifir(monkeypatch):
    """1 hissenin nominali tavandan büyükse pozisyon açılamaz (0 birim → red)."""
    r = _size(_fresh(monkeypatch, "3000"), entry=5000.0, stop=4900.0)
    assert not r.approved
    assert "MAX_POSITION_NOTIONAL" in r.reason
