"""research.realized_vol_probe testleri — gerçekleşen oynaklık ölçüm hattı (Tur-32).

AĞ YOK: `alpaca_history.get_history` monkeypatch'lenir. Barlar BİLİNEN bir oynaklık
sürecinden üretilir (rejim değiştiren günlük varyans) — hat doğruysa HAR triviyal tabanları
GEÇMELİ; sabit-varyanslı (öngörülemez) süreçte ise GEÇMEMELİ. İkinci test birincisinden
daha önemlidir: harness'ın olmayan beceriyi VAR göstermediğini kilitler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research import realized_vol_probe as rvp

_BAR_PER_DAY = 7


def _bars(gun_vol: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Günlük vol dizisinden saatlik bar serisi (gün içi sabit vol, günler arası değişken)."""
    rng = np.random.default_rng(seed)
    ts, px = [], []
    fiyat = 100.0
    baslangic = pd.Timestamp("2024-01-01 10:00")
    for g, vol in enumerate(gun_vol):
        for b in range(_BAR_PER_DAY):
            fiyat *= float(np.exp(rng.normal(0.0, vol)))
            ts.append(baslangic + pd.Timedelta(days=g, hours=b))
            px.append(fiyat)
    arr = np.asarray(px)
    return pd.DataFrame({"close": arr, "high": arr * 1.001, "low": arr * 0.999,
                         "volume": np.full(arr.size, 1e6)}, index=pd.DatetimeIndex(ts))


def _kalici_vol(n: int, seed: int = 1) -> np.ndarray:
    """Kalıcı (öngörülebilir) günlük vol: log-AR(1), φ=0.97."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.97 * x[i - 1] + rng.normal(0.0, 0.25)
    return 0.004 * np.exp(x)


# ---------------------------------------------------------------------------
def test_daily_rv_tanim_ve_sicrama_ayrismasi():
    bars = _bars(np.full(40, 0.005), seed=3)
    gunluk = rvp.daily_rv(bars)
    assert len(gunluk) == 40
    assert set(gunluk.columns) >= {"rv", "bv", "jump", "n_bar"}
    # İLK gün bir bar eksiktir: `.diff()` serinin ilk gözlemini düşürür. Bu KASITLI ve
    # doğrudur — o düşen fark GECELİK getiridir, gün-içi varyansa ait değildir.
    assert gunluk["n_bar"].iloc[0] == _BAR_PER_DAY - 1
    assert (gunluk["n_bar"].iloc[1:] == _BAR_PER_DAY).all()
    # RV = Σr²; sabit vol'da beklenen ≈ bar sayısı · σ²
    assert gunluk["rv"].iloc[1:].mean() == pytest.approx(_BAR_PER_DAY * 0.005 ** 2, rel=0.5)
    assert (gunluk["jump"] >= 0).all() and (gunluk["jump"] <= gunluk["rv"] + 1e-18).all()


def test_daily_rv_yetersiz_bar_gunu_duser():
    bars = _bars(np.full(5, 0.005), seed=4)
    kirpik = bars.iloc[:-(_BAR_PER_DAY - 1)]          # son gün yalnız 1 bar
    assert len(rvp.daily_rv(kirpik)) == 4
    assert rvp.daily_rv(pd.DataFrame()).empty


def test_qlike_kusursuz_tahminde_sifir_ve_pozitif_tanimli():
    y = np.array([1e-4, 4e-4, 9e-4])
    assert rvp._qlike(y, y) == pytest.approx(0.0, abs=1e-12)
    assert rvp._qlike(y, y * 2.0) > 0
    assert rvp._qlike(y, y * 0.5) > 0


def test_har_kalici_volde_tabanlari_gecer():
    """Öngörülebilir (kalıcı) oynaklıkta HAR en iyi triviyal tabanı GEÇMELİ."""
    gunluk = rvp.daily_rv(_bars(_kalici_vol(500), seed=5))
    wf = rvp.walk_forward(gunluk["rv"].to_numpy(dtype=float), refit=20)
    assert wf is not None and wf["n_oos"] > 200
    assert set(wf["skor"]) == {"HAR", "RW", "MA22", "SABIT"}
    assert wf["beceri_qlike"] > 0.0
    assert wf["skor"]["HAR"]["qlike"] < wf["skor"]["RW"]["qlike"]


def test_har_sabit_volde_tabani_ezmez():
    """Öngörülemez (sabit vol) süreçte HAR sabit tabana karşı KAYDA DEĞER kazanç
    üretmemeli — harness olmayan beceriyi var göstermemeli (ACI DERS #9)."""
    gunluk = rvp.daily_rv(_bars(np.full(500, 0.005), seed=6))
    wf = rvp.walk_forward(gunluk["rv"].to_numpy(dtype=float), refit=20)
    assert wf is not None
    assert wf["beceri_qlike"] < 0.05


def test_walk_forward_yetersiz_orneklemde_none():
    assert rvp.walk_forward(np.full(50, 1e-4)) is None


def test_micro_diagnostics_ve_rapor_kenar_iddiasi_icermez():
    bars = _bars(_kalici_vol(200), seed=7)
    gunluk = rvp.daily_rv(bars)
    mikro = rvp.micro_diagnostics(bars)
    assert {"imza", "roll", "tsrv", "amihud", "corwin_schultz"} <= set(mikro)
    metin = rvp.report("TEST", gunluk, None, mikro)
    assert "KENAR DEĞİLDİR" in metin
    assert "YETERSİZ örneklem" in metin          # wf=None dürüstçe raporlanır


def test_probe_bos_bar_savunmaci(monkeypatch):
    from nihai.data import alpaca_history
    monkeypatch.setattr(alpaca_history, "get_history",
                        lambda symbol, **kw: pd.DataFrame())
    gunluk, wf, mikro = rvp.probe("YOK", "2024-01-01")
    assert gunluk.empty and wf is None and mikro == {}


def test_probe_uctan_uca_sentetik_barlarla(monkeypatch):
    from nihai.data import alpaca_history
    bars = _bars(_kalici_vol(400), seed=8)
    monkeypatch.setattr(alpaca_history, "get_history", lambda symbol, **kw: bars)
    gunluk, wf, mikro = rvp.probe("TEST", "2024-01-01")
    assert len(gunluk) == 400 and wf is not None
    assert "HAR becerisi" in rvp.report("TEST", gunluk, wf, mikro)
