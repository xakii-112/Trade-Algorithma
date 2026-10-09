"""cboe_data testleri — TAMAMI ÇEVRİMDIŞI (pd.read_csv / get_index monkeypatch'lenir).

En kritik testi `test_vix_pctile_look_ahead_icermez`: yüzdelik dilim tüm örneklem
üzerinden hesaplanırsa geleceği sızdırır ve bu projenin 1 numaralı kuralını çiğner.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nihai.data import cboe_data as cb


def _ham_ohlc():
    """CBOE'nin gerçek şeması: DATE,OPEN,HIGH,LOW,CLOSE ve MM/DD/YYYY tarih."""
    return pd.DataFrame({
        "DATE": ["01/03/2020", "01/02/2020", "01/06/2020"],   # kasten sırasız
        "OPEN": [12.0, 13.0, 14.0],
        "HIGH": [12.5, 13.5, 14.5],
        "LOW": [11.5, 12.5, 13.5],
        "CLOSE": [12.2, 13.2, 14.2],
    })


def _ham_tek_kolon(ad="VVIX"):
    return pd.DataFrame({"DATE": ["01/02/2020", "01/03/2020"], ad: [90.0, 95.0]})


def test_bilinmeyen_seri_reddedilir():
    with pytest.raises(ValueError):
        cb.get_index("YOKBOYLE")


def test_ohlc_normalize_ve_siralama(monkeypatch):
    monkeypatch.setattr(cb.pd, "read_csv", lambda *a, **k: _ham_ohlc())
    monkeypatch.setattr(cb.pd.DataFrame, "to_csv", lambda *a, **k: None)
    df = cb.get_index("VIX", use_cache=False)
    assert list(df.columns) == ["open", "high", "low", "close"]
    assert df.index.is_monotonic_increasing, "tarihe göre sıralanmamış"
    assert str(df.index[0].date()) == "2020-01-02"


def test_tek_kolonlu_seri_close_a_normalize_edilir(monkeypatch):
    monkeypatch.setattr(cb.pd, "read_csv", lambda *a, **k: _ham_tek_kolon("VVIX"))
    monkeypatch.setattr(cb.pd.DataFrame, "to_csv", lambda *a, **k: None)
    df = cb.get_index("VVIX", use_cache=False)
    assert "close" in df.columns, "VVIX/SKEW tek kolonu 'close'a çevrilmedi"
    assert df["close"].iloc[-1] == 95.0


def test_indirme_hatasinda_bos_doner_cokmez(monkeypatch):
    def patla(*a, **k):
        raise OSError("ağ yok")

    monkeypatch.setattr(cb.pd, "read_csv", patla)
    monkeypatch.setattr(cb, "_read_cache", lambda name: None)
    monkeypatch.setattr(cb, "_cache_path", lambda name: cb.Path("yok_boyle_dosya.csv"))
    assert cb.get_index("VIX", use_cache=False).empty


def _sahte_seriler(n=400):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return {
        "VIX": pd.DataFrame({"close": [15.0 + (i % 10) for i in range(n)]}, index=idx),
        "VIX3M": pd.DataFrame({"close": [18.0] * n}, index=idx),
        "VIX9D": pd.DataFrame({"close": [13.0] * n}, index=idx),
        "VVIX": pd.DataFrame({"close": [90.0] * n}, index=idx),
        "SKEW": pd.DataFrame({"close": [130.0] * n}, index=idx),
    }


def test_term_structure_matematigi(monkeypatch):
    seriler = _sahte_seriler()
    monkeypatch.setattr(cb, "get_index", lambda n, **k: seriler.get(n, pd.DataFrame()))
    ts = cb.term_structure()
    # VIX=15 iken ts_slope = 18/15 = 1.2 → contango → backwardation=0
    ilk = ts.iloc[0]
    assert ilk["ts_slope"] == pytest.approx(18.0 / 15.0)
    assert ilk["backwardation"] == 0
    assert ilk["near_slope"] == pytest.approx(15.0 / 13.0)
    assert ilk["vvix_ratio"] == pytest.approx(0.90)


def test_backwardation_bayragi_stres_gununde_yanar(monkeypatch):
    seriler = _sahte_seriler(300)
    seriler["VIX"] = pd.DataFrame({"close": [40.0] * 300}, index=seriler["VIX"].index)
    monkeypatch.setattr(cb, "get_index", lambda n, **k: seriler.get(n, pd.DataFrame()))
    ts = cb.term_structure()
    # VIX 40 > VIX3M 18 → ts_slope < 1 → backwardation
    assert (ts["backwardation"] == 1).all()


def test_vix_pctile_look_ahead_icermez(monkeypatch):
    """Yüzdelik dilim YALNIZ geçmişe bakmalı (genişleyen pencere).

    Tüm örneklem üzerinden rank alınsaydı, serinin en yüksek VIX'i hangi gün olursa
    olsun o gün 100 olurdu — yani gelecekteki bilgi bugünkü özelliğe sızardı.
    Test: seriyi ortasından kesip yeniden hesaplayınca ORTAK günlerin değeri
    DEĞİŞMEMELİ. Değişiyorsa geleceğe bakıyor demektir.
    """
    seriler = _sahte_seriler(600)
    monkeypatch.setattr(cb, "get_index", lambda n, **k: seriler.get(n, pd.DataFrame()))
    tam = cb.term_structure()

    kisa_seriler = {k: v.iloc[:400] for k, v in seriler.items()}
    monkeypatch.setattr(cb, "get_index", lambda n, **k: kisa_seriler.get(n, pd.DataFrame()))
    kisa = cb.term_structure()

    ortak = kisa.index.intersection(tam.index)
    assert len(ortak) > 100
    pd.testing.assert_series_equal(
        tam.loc[ortak, "vix_pctile"], kisa.loc[ortak, "vix_pctile"],
        check_names=False,
    )


def test_vix_pctile_ilk_252_gunde_uretilmez(monkeypatch):
    """252 gözlemden az veriyle yüzdelik dilim anlamsızdır → NaN kalmalı."""
    seriler = _sahte_seriler(300)
    monkeypatch.setattr(cb, "get_index", lambda n, **k: seriler.get(n, pd.DataFrame()))
    ts = cb.term_structure()
    assert ts["vix_pctile"].iloc[:251].isna().all()
    assert ts["vix_pctile"].iloc[252:].notna().any()
