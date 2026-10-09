"""research.option_surface_probe testleri — opsiyon yüzeyi ölçüm hattı (Tur-32).

AĞ YOK: `dolt_options.option_chain` monkeypatch'lenir ve yerine BİLİNEN parametrelerden
(F, r, σ) Black-Scholes ile üretilmiş SENTETİK bir zincir konur. Hat doğruysa ölçüm o
parametreleri geri vermelidir — canlı koşumun (2026-08-13 SPY/AAPL) yakaladığı davranışın
determinist kilidi budur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.ai import quant_math as qm
from research import option_surface_probe as osp

_F, _R, _SIG = 100.0, 0.03, 0.20
_GUN = "2026-01-01"
_VADELER = {"2026-04-02": 91 / 365.0, "2026-07-02": 182 / 365.0}


def _sentetik_zincir() -> pd.DataFrame:
    satir = []
    for exp, t in _VADELER.items():
        s_eff = _F * np.exp(-_R * t)                      # forward-tutarlı "spot"
        for k in range(85, 121, 5):
            for kind, etiket in (("c", "Call"), ("p", "Put")):
                px = qm.bs_price(s_eff, float(k), t, _R, _SIG, kind)
                if px is None or px < 0.05:
                    continue
                satir.append({"date": pd.Timestamp(_GUN), "act_symbol": "TEST",
                              "expiration": pd.Timestamp(exp), "strike": float(k),
                              "call_put": etiket, "bid": px * 0.999, "ask": px * 1.001,
                              "vol": _SIG})
    return pd.DataFrame(satir)


@pytest.fixture()
def dilimler(monkeypatch):
    from nihai.data import dolt_options
    monkeypatch.setattr(dolt_options, "option_chain",
                        lambda symbol, *, date, **kw: _sentetik_zincir())
    return osp.probe("TEST", _GUN)


def test_parite_regresyonu_forward_ve_faizi_geri_verir(dilimler):
    """Zincirin KENDİSİNDEN çıkarılan F ve r, üretici parametrelere eşit olmalı; dışarıdan
    spot/faiz varsayımı GİRMEZ. Canlı koşumda parite-R² 0,9999 ölçüldü."""
    assert len(dilimler) == 2
    for d in dilimler:
        assert d["forward"] == pytest.approx(_F, rel=1e-6)
        assert d["r"] == pytest.approx(_R, abs=1e-6)
        assert d["parity_r2"] == pytest.approx(1.0, abs=1e-9)


def test_iv_cozucusu_uretici_sigmayi_geri_verir(dilimler):
    """OTM kotasyonlarda kendi IV'miz satıcının kolonuyla (burada gerçek σ) eşleşmeli.
    Canlı SPY'da bu fark 0,0004–0,0021 ölçüldü."""
    for d in dilimler:
        assert d["n_karsilastirilan"] >= 5
        assert d["iv_fark_max"] < 1e-5


def test_duz_gulumsemede_svi_yassidir_ve_arbitrajsizdir(dilimler):
    """Sabit σ üreticide SVI eğrisi düz olmalı (b≈0), Durrleman g(k) ≥ 0 ve RR/BF ≈ 0."""
    for d in dilimler:
        assert d["svi"] is not None and d["svi"]["rmse"] < 1e-6
        assert d["svi"]["b"] == pytest.approx(0.0, abs=1e-4)
        assert d["arb"]["butterfly_free"] is True
        assert d["smile"]["atm_vol"] == pytest.approx(_SIG, abs=1e-4)
        assert d["smile"]["rr25"] == pytest.approx(0.0, abs=1e-4)


def test_risk_notr_yogunluk_kutlesi_bire_yakin(dilimler):
    """BS zincirinden Breeden-Litzenberger yoğunluğu: ızgara dar olduğu için kütle 1'in
    ALTINDA kalır ama negatif nokta OLMAMALI (arbitrajsız üretici)."""
    for d in dilimler:
        assert d["rnd"] is not None
        assert d["rnd"]["n_negative"] == 0
        assert 0.5 < d["rnd"]["mass"] <= 1.001


def test_r_guvenilirlik_bayragi_kisa_vadeyi_isaretler(dilimler):
    """t ≥ 0,08y ve |r| ≤ 0,15 ise güvenilir. Canlı koşumda kısa vadelerde r kötü-koşullu
    çıktı (SPY t=0,041y'de r=−0,16) — bayrak bunu okuyucuya söyler."""
    assert all(d["r_guvenilir"] for d in dilimler)      # ikisi de t ≥ 0,08


def test_rapor_metni_kenar_iddiasi_icermez(dilimler):
    metin = osp.report("TEST", _GUN, dilimler)
    assert "kenar kanıtı DEĞİLDİR" in metin
    assert "TEST @ 2026-01-01" in metin


def test_bos_zincir_savunmaci(monkeypatch):
    from nihai.data import dolt_options
    monkeypatch.setattr(dolt_options, "option_chain",
                        lambda symbol, *, date, **kw: pd.DataFrame())
    assert osp.probe("YOK", _GUN) == []
    assert "kullanılabilir zincir YOK" in osp.report("YOK", _GUN, [])


def test_eslesmeyen_strike_paritesi_none(monkeypatch):
    """Yalnız call olan bir dilimde parite denklemi kurulamaz → dilim raporlanmaz
    (uydurma forward üretilmez)."""
    yalniz_call = _sentetik_zincir()
    yalniz_call = yalniz_call[yalniz_call["call_put"] == "Call"]
    from nihai.data import dolt_options
    monkeypatch.setattr(dolt_options, "option_chain",
                        lambda symbol, *, date, **kw: yalniz_call)
    assert osp.probe("TEST", _GUN) == []
