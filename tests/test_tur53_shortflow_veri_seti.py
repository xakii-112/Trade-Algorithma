"""
test_tur53_shortflow_veri_seti.py — BACKLOG #8 (FINRA günlük açığa satış akışı) kilitleri.

Bu ailenin sessizce "kenar" uydurabileceği beş yol var; beşi de burada çapalanır:

* **feature-lag 1 (ACI DERS #7).** FINRA D gününün dosyasını D kapandıktan SONRA yayımlar.
  Giriş barı D'nin kendisi olursa ölçülen şey sinyal değil o günün kendi hareketidir.
* **Yön kuralı.** z>0 (olağandışı YOĞUN açığa satış) → SHORT bacak. Yön ters bağlanırsa
  literatürün işaretini test etmiş olmayız, tersini test etmiş oluruz.
* **Puanlama FAZLALIK üstünden (§6).** Boğa piyasasında her short bacağı ham getiride
  kaybeder, her long bacağı kazanır; `pnl` `fwd_ret`e değil `excess`e bakmalı.
* **Örtüşmesizlik.** Aynı sembolde `horizon` bardan yakın satırlar aynı ileri pencereyi
  paylaşır; sayılırsa t-istatistiği şişer (Tur-27: bağımsızlık birimi gözlem değildir).
* **Yarım ay önbelleği kabul EDİLEMEZ.** Tur-51'in 20 aylık sessiz deliği tam olarak
  budur: eksik günleri olan bir önbellek "tamam" sayılırsa kapsam sessizce daralır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.edge_lab import build_shortflow_dataset as B


def _px(n: int = 300, start: str = "2021-01-04") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    # hacim SABİT olamaz: log-dolar-hacim z-skorunun std'si 0 olur ve TÜM satırlar
    # elenir — testler sessizce boş küme üstünde geçerdi.
    hacim = 1_000_000.0 * np.exp(rng.normal(0, 0.2, n))
    return pd.DataFrame({"date": idx, "close": close, "volume": hacim})


def _svr(px: pd.DataFrame, sicrama_idx: int | None = None, sicrama: float = 0.9) -> pd.Series:
    """Sabit ~0,45 taban + istenirse tek günlük sıçrama (z>0 üretir)."""
    rng = np.random.default_rng(1)
    v = 0.45 + rng.normal(0, 0.01, len(px))
    if sicrama_idx is not None:
        v[sicrama_idx] = sicrama
    return pd.Series(v, index=pd.DatetimeIndex(px["date"]))


def _ornek_karar_gunu(px: pd.DataFrame, horizon: int = 5) -> int:
    """Örtüşmezlik ızgarasına DÜŞEN bir karar günü (D) indeksi seç.

    Satırlar kronolojik-açgözlü seçilir (her `horizon` barda bir); rastgele bir güne
    sıçrama koymak satırın hiç üretilmemesine yol açar — testin kendi kusuru olurdu.
    """
    rows = B.rows_for("T", px, _svr(px), horizon=horizon)
    ts = rows[len(rows) // 2]["ts"]
    return int(pd.DatetimeIndex(px["date"]).get_loc(ts)) - 1


def test_giris_bari_FINRA_gununden_SONRAKI_bardir():
    """feature-lag 1: D gününün SVR'si D+1 kapanışına girer, D'ye değil."""
    px = _px()
    i = _ornek_karar_gunu(px)
    ser = _svr(px, sicrama_idx=i)
    rows = B.rows_for("TEST", px, ser, horizon=5)
    hit = [r for r in rows if r["f_svr_absz"] > 5]
    assert hit, "sıçrama satırı üretilmedi"
    r = hit[0]
    assert r["ts"] == px["date"].iloc[i + 1], "giriş barı D+1 olmalı"
    beklenen = px["close"].iloc[i + 1 + 5] / px["close"].iloc[i + 1] - 1.0
    assert r["fwd_ret"] == pytest.approx(beklenen)


def test_yon_kurali_yogun_aciga_satis_SHORT():
    px = _px()
    i = _ornek_karar_gunu(px)
    yukari = B.rows_for("T", px, _svr(px, sicrama_idx=i, sicrama=0.90), horizon=5)
    asagi = B.rows_for("T", px, _svr(px, sicrama_idx=i, sicrama=0.05), horizon=5)
    ust = [r for r in yukari if r["f_svr_absz"] > 5][0]
    alt = [r for r in asagi if r["f_svr_absz"] > 5][0]
    assert ust["f_svr_z"] > 0 and ust["f_is_short"] == 1.0
    assert alt["f_svr_z"] < 0 and alt["f_is_short"] == 0.0


def test_ortusmezlik_horizon_kadar():
    px = _px()
    rows = B.rows_for("T", px, _svr(px), horizon=5)
    assert len(rows) > 20, "boş küme üstünde örtüşmezlik iddiası anlamsız"
    ts = pd.DatetimeIndex([r["ts"] for r in rows])
    idx = pd.DatetimeIndex(px["date"]).get_indexer(ts)
    assert (np.diff(idx) >= 5).all(), "ardışık satırlar horizon'dan yakın olamaz"


def test_puanlama_HAM_getiriye_degil_FAZLALIGA_bakar():
    """§6 beta tuzağı: piyasa +%3 iken short bacak ham getiride kaybeder ama
    fazlalıkta kazanır; `pnl` fazlalığı görmeli."""
    df = pd.DataFrame({
        "f_is_short": [1.0, 0.0],
        "fwd_ret": [0.010, 0.010],     # ikisi de ham POZİTİF
        "mkt_ret": [0.030, 0.030],     # ama piyasa daha çok yükselmiş
        "excess": [-0.020, -0.020],
    })
    out = B.puanla(df.copy(), horizon=5, cost_pct=0.0, borrow_bps_yr=0.0)
    assert out["pnl"].iloc[0] == pytest.approx(0.020)    # short: −(−0,02) = +0,02
    assert out["pnl"].iloc[1] == pytest.approx(-0.020)   # long: −0,02
    assert out["label"].tolist() == [1, 0]


def test_odunc_maliyeti_YALNIZ_short_bacakta():
    df = pd.DataFrame({"f_is_short": [1.0, 0.0], "excess": [0.0, 0.0]})
    out = B.puanla(df.copy(), horizon=252, cost_pct=0.0, borrow_bps_yr=100.0)
    assert out["pnl"].iloc[0] == pytest.approx(-0.01)    # 1 yıl × %1
    assert out["pnl"].iloc[1] == pytest.approx(0.0)


def test_yarim_ay_onbellegi_YENIDEN_cekilir(tmp_path, monkeypatch):
    """Tur-51 dersi: eksik günü olan önbellek 'tamam' sayılamaz."""
    monkeypatch.setattr(B, "CACHE_DIR", tmp_path)
    gunler = pd.DatetimeIndex(pd.bdate_range("2021-03-01", periods=5))
    (tmp_path / "2021-03.csv").write_text(
        "date,symbol,short_vol,total_vol\n2021-03-01,AAA,10,100\n", encoding="utf-8")

    cagrilan: list[pd.Timestamp] = []

    def sahte(gun, evren):
        cagrilan.append(gun)
        return pd.DataFrame({"date": [gun], "symbol": ["AAA"],
                             "short_vol": [10.0], "total_vol": [100.0]})

    monkeypatch.setattr(B, "_fetch_day", sahte)
    out = B.load_shortvol(gunler, {"AAA"}, workers=2)
    assert len(cagrilan) == 5, "yarım ay önbelleği kabul edilmiş"
    assert len(out) == 5

    cagrilan.clear()
    B.load_shortvol(gunler, {"AAA"}, workers=2)
    assert cagrilan == [], "tam ay önbelleği yeniden çekilmemeli"
