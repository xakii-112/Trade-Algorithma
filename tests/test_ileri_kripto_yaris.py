"""Kripto ileri yarışı (`research.ileri_kripto_yaris`) — ön-kayıt kilidinin sözleşmeleri.

Sınama ancak üç şey tutarsa anlamlıdır: kural geleceği görmez, deftere yalnız kayıt gününden SONRAKİ tam
günler girer, ve geçmişi/defteri değiştiren her şey (kod, veri, belge) yakalanır. Ağ gerektirmez.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research import ileri_kripto_yaris as Y


def _R(tarihler, coinler=Y.COINLER, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"tarih": t, "coin": c, "strateji": s, "getiri": float(rng.normal(0, 0.02))}
                         for t in tarihler for c in coinler for s in Y.STRATEJILER])


def test_ma10_50_gelecege_bakmaz():
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.03, 300))),
                      pd.date_range("2020-01-01", periods=300))
    oyna = close.copy()
    oyna.iloc[200:] *= 1.5                       # 200. günden sonrası değişir
    for g in (0, 1):
        a, b = Y.ma10_50(close, g), Y.ma10_50(oyna, g)
        pd.testing.assert_series_equal(a.iloc[:200], b.iloc[:200])
        assert not a.iloc[200:].equals(b.iloc[200:])


def test_defter_yalniz_kayittan_sonraki_tam_gunleri_alir():
    gunler = pd.date_range(Y.KAYIT_GUNU - pd.Timedelta(days=2), periods=6)   # 09-23 … 09-28
    R = _R(gunler)
    R = R[~((R.tarih == gunler[-1]) & (R.coin == "SOL-USD"))]              # son gün bir coin eksik
    y = Y.yeni_satirlar(R, R.iloc[:0])
    assert sorted(y.tarih.unique()) == list(gunler[3:5])                  # 09-26, 09-27
    defter = y[y.tarih == gunler[3]]
    assert list(Y.yeni_satirlar(R, defter).tarih.unique()) == [gunler[4]]


def test_dogrula_kod_veri_belge_ve_defter_bozulmasini_yakalar():
    R = _R(pd.date_range("2018-06-01", Y.KAYIT_GUNU - pd.Timedelta(days=1), freq="30D"), coinler=("BTC-USD",))
    kilit = {"onkayit_sha256": "abc", "parmak_izi": Y._gecmis(R)}
    defter = _R([Y.KAYIT_GUNU + pd.Timedelta(days=1)], coinler=("BTC-USD",), seed=5)
    R2 = pd.concat([R, defter], ignore_index=True)
    assert Y.dogrula(R2, kilit, defter, "abc") == []
    assert any("belgesi" in s for s in Y.dogrula(R2, kilit, defter, "xyz"))
    bozuk = R2.copy()
    bozuk.loc[(bozuk.tarih.dt.year == 2019) & (bozuk.strateji == "event_breakout"), "getiri"] += 1e-6
    sorun = Y.dogrula(bozuk, kilit, defter, "abc")
    assert len(sorun) == 1 and "event_breakout" in sorun[0] and "2019" in sorun[0]
    defter2 = defter.assign(getiri=defter.getiri + 1e-6)
    assert any("defterdeki" in s for s in Y.dogrula(R2, kilit, defter2, "abc"))


def test_rapor_ara_donemde_karar_vermez_son_gunde_verir():
    defter = _R(pd.date_range(Y.KAYIT_GUNU + pd.Timedelta(days=1), periods=120))
    ara = Y.rapor(defter, Y.SON_KARAR - pd.Timedelta(days=1))
    assert "ARA RAPOR" in ara and "KARAR H1" not in ara
    son = Y.rapor(defter, Y.SON_KARAR)
    assert "KARAR H1" in son and son.count("KARAR H2") == 2


def test_onkayit_belgesi_kilitle_ayni():
    """Ön-kayıt belgesi kilitten sonra düzenlenirse test paketi de kırmızıya döner (ek belge yazılmalı)."""
    kilit = json.loads(Y.KILIT.read_text(encoding="utf-8"))
    assert kilit["kayit_gunu"] == "2026-09-25"
    assert Y._sha(Y.ONKAYIT) == kilit["onkayit_sha256"]
    assert set(kilit["parmak_izi"]) == set(Y.COINLER)
