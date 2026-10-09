"""
test_tur54_optskew_veri_seti.py — BACKLOG #9 (opsiyon yüzeyi yön sinyali) kilitleri.

Bu ailenin sessizce yanlış bir şey ölçmesi altı yoldan mümkündü:

* **Vade seçimi.** DTE penceresi dışındaki bir vade seçilirse ölçülen şey başka bir
  enstrümandır; "30'a en yakın" kuralı ön-kayıtlıdır ve taranmamıştır.
* **Uydurma yüzey.** İstenen deltaya yakın kotasyon YOKSA en yakını almak sessizce
  başka bir opsiyonu ATM/OTM sanmaktır → ölçüm yokluğu `None` olmalı (Tur-31 kuralı).
* **Yön kuralı.** VS yüksek (call'lar zengin) = LONG. Ters bağlanırsa literatürün
  işaretini değil tersini test etmiş oluruz.
* **feature-lag 1 (ACI DERS #7).** Zincir gün sonu verisidir; giriş D+1 kapanışıdır.
* **Kotasyon kalitesi.** Sıfır-bid opsiyonun IV'si ölçüm değil gürültüdür.
* **PARÇA SINIRI.** Dosya belleğe sığmadığı için akıtılır; bir (gün, sembol) grubu iki
  parçaya bölünürse yarım ölçülürdü. Devretme mekanizması bunu engellemeli — bu test,
  bölünmüş okumanın bölünmemiş okumayla AYNI sonucu vermesini kilitler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.edge_lab import build_optskew_dataset as B

STRIKES = [90.0, 95.0, 100.0, 105.0, 110.0]
CALL_D = [0.80, 0.65, 0.50, 0.35, 0.20]
PUT_D = [-0.20, -0.35, -0.50, -0.65, -0.80]


def _zincir(gun: str, sym: str = "TEST", dte: int = 30, iv_call: float = 0.30,
            iv_put: float = 0.28, bid: float = 1.0) -> pd.DataFrame:
    """Tek gün×sembol için tam bir zincir dilimi (5 kullanım fiyatı × call/put)."""
    d = pd.Timestamp(gun)
    exp = d + pd.Timedelta(days=dte)
    rows = []
    for k, dc, dp in zip(STRIKES, CALL_D, PUT_D, strict=True):
        rows.append({"date": d, "act_symbol": sym, "expiration": exp, "strike": k,
                     "call_put": "Call", "bid": bid, "ask": bid + 0.2,
                     "vol": iv_call, "delta": dc})
        rows.append({"date": d, "act_symbol": sym, "expiration": exp, "strike": k,
                     "call_put": "Put", "bid": bid, "ask": bid + 0.2,
                     "vol": iv_put, "delta": dp})
    return pd.DataFrame(rows)


def test_yuzey_olculeri_beklenen_degerler():
    r = B._yuzey_olc(_zincir("2021-06-01"))
    assert r is not None
    assert r["vs"] == pytest.approx(0.02)        # 0,30 − 0,28, beş eşleşen çift
    assert r["skew"] == pytest.approx(-0.02)     # IV(OTM put) − IV(ATM call)
    assert r["iv_atm"] == pytest.approx(0.30)
    assert r["n_cift"] == 5


def test_vade_penceresi_disi_OLCULMEZ():
    """DTE ∈ [20,60] ön-kayıtlıdır; 7 günlük vade başka bir enstrümandır."""
    assert B._yuzey_olc(_zincir("2021-06-01", dte=7)) is None
    assert B._yuzey_olc(_zincir("2021-06-01", dte=200)) is None


def test_30a_EN_YAKIN_vade_secilir():
    yakin = _zincir("2021-06-01", dte=28, iv_call=0.30, iv_put=0.28)   # VS +0,02
    uzak = _zincir("2021-06-01", dte=55, iv_call=0.30, iv_put=0.20)    # VS +0,10
    r = B._yuzey_olc(pd.concat([yakin, uzak], ignore_index=True))
    assert r is not None
    assert r["vs"] == pytest.approx(0.02), "30'a uzak vade seçilmiş"


def test_delta_toleransi_disinda_UYDURMAZ():
    """ATM'ye yakın kotasyon yoksa en yakınını ATM saymak sessiz bir yalandır."""
    g = _zincir("2021-06-01")
    g = g[~((g["call_put"] == "Call") & (g["delta"].between(0.30, 0.70)))]
    assert B._yuzey_olc(g) is None


def test_yon_kurali_VS_yuksek_LONG():
    px = _px()
    yuzey_yuksek = _yuzey_serisi(px, vs_son=+0.10)
    yuzey_dusuk = _yuzey_serisi(px, vs_son=-0.10)
    r_up = [r for r in B.rows_for("T", px, yuzey_yuksek, 5) if r["f_vs_absz"] > 3]
    r_dn = [r for r in B.rows_for("T", px, yuzey_dusuk, 5) if r["f_vs_absz"] > 3]
    assert r_up and r_dn
    assert r_up[0]["f_vs_z"] > 0 and r_up[0]["f_is_short"] == 0.0
    assert r_dn[0]["f_vs_z"] < 0 and r_dn[0]["f_is_short"] == 1.0


def test_giris_bari_karar_gununden_SONRAKI_bardir():
    px = _px()
    i = _karar_gunu(px)
    yuzey = _yuzey_serisi(px, vs_son=+0.10, sicrama_idx=i)
    hit = [r for r in B.rows_for("T", px, yuzey, 5) if r["f_vs_absz"] > 3]
    assert hit
    assert hit[0]["ts"] == px["date"].iloc[i + 1]
    beklenen = px["close"].iloc[i + 1 + 5] / px["close"].iloc[i + 1] - 1.0
    assert hit[0]["fwd_ret"] == pytest.approx(beklenen)


def test_sifir_bid_kotasyonu_ELENIR(tmp_path):
    yol = tmp_path / "chain.csv"
    iyi = _zincir("2021-06-01", sym="AAA", bid=1.0)
    kotu = _zincir("2021-06-01", sym="BBB", bid=0.0)
    pd.concat([iyi, kotu], ignore_index=True).to_csv(yol, index=False)
    out = B.yuzey_serisi(yol, {"AAA", "BBB"}, "2021-01-01", "2026-12-31")
    assert set(out["symbol"]) == {"AAA"}


def test_PARCA_SINIRI_bolunmus_okuma_bolunmemisle_AYNI(tmp_path):
    """Bir gün×sembol grubu iki parçaya bölünse bile ölçü TAM kalmalı."""
    yol = tmp_path / "chain.csv"
    parcalar = [_zincir(g, sym=s)
                for g in ("2021-06-01", "2021-06-02", "2021-06-03")
                for s in ("AAA", "BBB")]
    pd.concat(parcalar, ignore_index=True).to_csv(yol, index=False)
    evren = {"AAA", "BBB"}
    tam = B.yuzey_serisi(yol, evren, "2021-01-01", "2026-12-31", chunksize=10_000)
    bolunmus = B.yuzey_serisi(yol, evren, "2021-01-01", "2026-12-31", chunksize=7)
    assert len(tam) == 6
    pd.testing.assert_frame_equal(tam, bolunmus)


# --- yardımcılar ------------------------------------------------------------------------
def _px(n: int = 240, start: str = "2021-01-04") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    return pd.DataFrame({"date": idx, "close": close,
                         "volume": 1e6 * np.exp(rng.normal(0, 0.2, n))})


def _yuzey_serisi(px: pd.DataFrame, vs_son: float,
                  sicrama_idx: int | None = None) -> pd.DataFrame:
    """Sabit ~0,00 taban + tek günlük sıçrama → o günde büyük |z|."""
    rng = np.random.default_rng(1)
    vs = rng.normal(0, 0.005, len(px))
    i = _karar_gunu(px) if sicrama_idx is None else sicrama_idx
    vs[i] = vs_son
    return pd.DataFrame({"date": px["date"], "vs": vs,
                         "skew": rng.normal(-0.02, 0.005, len(px)),
                         "iv_atm": 0.30, "n_cift": 5})


def _karar_gunu(px: pd.DataFrame, horizon: int = 5) -> int:
    """Örtüşmezlik ızgarasına DÜŞEN bir karar günü indeksi (testin kendi kusuru olmasın)."""
    # skew SABİT olamaz: z-skorunun std'si 0 olur ve TÜM satırlar elenir — yardımcı
    # sessizce boş liste döndürür ve testin kendisi kusurlu olur.
    rng = np.random.default_rng(2)
    duz = pd.DataFrame({"date": px["date"], "vs": np.linspace(0, 1, len(px)),
                        "skew": rng.normal(-0.02, 0.005, len(px)),
                        "iv_atm": 0.30, "n_cift": 5})
    rows = B.rows_for("T", px, duz, horizon)
    ts = rows[len(rows) // 2]["ts"]
    return int(pd.DatetimeIndex(px["date"]).get_loc(ts)) - 1


def _zincir_ozel(gun: str, sym: str, satirlar: list[tuple], dte: int = 30) -> pd.DataFrame:
    """(strike, call_put, delta, vol) dörtlülerinden elle zincir — kenar durumlar için."""
    d = pd.Timestamp(gun)
    return pd.DataFrame([
        {"date": d, "act_symbol": sym, "expiration": d + pd.Timedelta(days=dte),
         "strike": k, "call_put": cp, "bid": 1.0, "ask": 1.2, "vol": iv, "delta": dl}
        for k, cp, dl, iv in satirlar])


def test_vektor_referansla_AYNI():
    """`_parca_olc` (vektör) ile `_yuzey_olc` (referans) AYNI kararı vermeli.

    Referans gövde üretimde çağrılmaz; bu testin orakldır. Kenar durumlar kasten
    içeride: iki vade, ATM'siz zincir, pencere dışı vade, tek eşleşen çift ve
    |dte−30| EŞİT iki vade (eşitlik bozucusu).
    """
    parcalar = [
        _zincir("2021-06-01", sym="AAA"),                                   # normal
        _zincir("2021-06-01", sym="BBB", dte=28),                           # 30'a yakın
        _zincir("2021-06-01", sym="BBB", dte=55, iv_put=0.20),              # 30'a uzak
        _zincir("2021-06-02", sym="DDD", dte=7),                            # pencere dışı
        _zincir("2021-06-03", sym="FFF", dte=25),                           # |dte−30|=5
        _zincir("2021-06-03", sym="FFF", dte=35, iv_put=0.10),              # |dte−30|=5
    ]
    ccc = _zincir("2021-06-02", sym="CCC")
    parcalar.append(ccc[~((ccc["call_put"] == "Call") & ccc["delta"].between(0.30, 0.70))])
    parcalar.append(_zincir_ozel("2021-06-03", "EEE", [                     # tek çift
        (90.0, "Call", 0.90, 0.30), (90.0, "Put", -0.20, 0.28),
        (100.0, "Call", 0.50, 0.30), (100.0, "Put", -0.50, 0.28)]))
    d = pd.concat(parcalar, ignore_index=True)

    ref = [r for _, g in d.groupby(["date", "act_symbol"], sort=False)
           if (r := B._yuzey_olc(g)) is not None]
    ds = d.assign(date=d["date"].dt.strftime("%Y-%m-%d"),
                  expiration=d["expiration"].dt.strftime("%Y-%m-%d"))
    vek = B._parca_olc(ds)

    kol = ["date", "symbol", "vs", "skew", "iv_atm", "n_cift"]
    a = pd.DataFrame(ref)[kol].sort_values(["symbol", "date"], ignore_index=True)
    b = pd.DataFrame(vek)[kol].sort_values(["symbol", "date"], ignore_index=True)
    assert set(a["symbol"]) == {"AAA", "BBB", "FFF"}, "kenar durumlar elenmemiş"
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
