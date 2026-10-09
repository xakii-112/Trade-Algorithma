"""Tur-78: eksik ölçüm, satır bazında gamma ve günlük zaman sınırı regresyonları."""
import math

import numpy as np
import pandas as pd
import pytest

from nihai.ai import option_flow as flow
from nihai.ai import options_math as om
from nihai.data import dolt_options as dolt


def _taraf(oi=100):
    return {"strike": [100], "openInterest": [oi], "impliedVolatility": [.3]}


def _gamma(spot=100, strike=100, t=.1, iv=.3):
    # r=0 Black-Scholes kapalı formu; üretim yardımcısını referans almıyoruz.
    d1 = (math.log(spot / strike) + iv * iv * t / 2) / (iv * math.sqrt(t))
    return math.exp(-d1 * d1 / 2) / (math.sqrt(2 * math.pi) * spot * iv * math.sqrt(t))


@pytest.mark.parametrize("oi", [None, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("ters", [False, True])
def test_o1_olculmemis_taraf_net_isareti_uretemez(oi, ters):
    eksik = _taraf(oi)
    if oi is None:
        del eksik["openInterest"]
    calls, puts = (eksik, _taraf()) if ters else (_taraf(), eksik)
    assert flow.dealer_gamma_exposure(calls, puts, 100, .1) is None
    tam = flow.dealer_gamma_exposure(_taraf(), _taraf(200), 100, .1)
    assert tam["net_gex"] == pytest.approx(-_gamma() * 100 * 100 * 100**2 * .01)
    assert tam["net_gex"] < 0 and tam["regime"] == "negatif_gamma"


def test_o1_kismi_oi_sayilir_verilmeyen_taraf_korunur():
    calls = {"strike": [100, 105, -1], "openInterest": [100, np.nan, np.nan],
             "impliedVolatility": [.3, .3, .3]}
    puts = {"strike": [100, 110], "openInterest": [200, np.nan], "impliedVolatility": [.3, .3]}
    sonuc = flow.dealer_gamma_exposure(calls, puts, 100, .1)
    assert sonuc["n_oi_eksik"] == 2 and type(sonuc["n_oi_eksik"]) is int
    assert sonuc["net_gex"] == pytest.approx(-_gamma() * 1e6)
    tek = flow.dealer_gamma_exposure(_taraf(), None, 100, .1)
    assert tek["net_gex"] == pytest.approx(_gamma() * 1e6)
    assert tek["n_oi_eksik"] == tek["n_gamma_eksik"] == 0
    gecersiz = flow.dealer_gamma_exposure(_taraf(), {"strike": [0, np.nan]}, 100, .1)
    assert gecersiz["net_gex"] == tek["net_gex"]


def test_o1b_iv_yoksa_olculmus_gamma_korunur():
    calls = {"strike": [100, 105], "openInterest": [100, 200],
             "impliedVolatility": [.3, np.nan], "gamma": [_gamma(), .04]}
    sonuc = flow.dealer_gamma_exposure(calls, None, 100, .1)
    assert sonuc["call_gex"] == pytest.approx([_gamma() * 1e6, 80000])
    assert sonuc["net_gex"] == pytest.approx(_gamma() * 1e6 + 80000)
    assert sonuc["gamma_source"] == "karisik" and sonuc["gamma_flip"] is None
    assert sonuc["n_gamma_eksik"] == 0
    # Geçerli IV zincir gammasına üstün gelir; sıfır gamma da ölçülmüş değerdir.
    calls["gamma"] = [.9, 0.0]
    sonuc = flow.dealer_gamma_exposure(calls, None, 100, .1)
    assert sonuc["call_gex"] == pytest.approx([_gamma() * 1e6, 0])
    assert sonuc["n_gamma_eksik"] == 0


def test_o1b_gamma_eksigi_sayilir_ve_flip_engellenir():
    # İlk iki kontrat tek başına 90–110 arasında kök verir; üçüncü OI ölçülüdür.
    calls = {"strike": [110, 120], "openInterest": [100, 100],
             "impliedVolatility": [.3, np.nan]}
    puts = {"strike": [90], "openInterest": [100], "impliedVolatility": [.3]}
    sonuc = flow.dealer_gamma_exposure(calls, puts, 100, .1, span=.3)
    assert sonuc["n_gamma_eksik"] == 1
    assert sonuc["gamma_flip"] is None
    assert sonuc["net_gex"] == pytest.approx((_gamma(strike=110) - _gamma(strike=90)) * 1e6)
    # Tamamen gammasız taraf nete sıfır yazılamaz (O1 ile aynı işaret hatası) → ölçüm yok.
    assert flow.dealer_gamma_exposure(_taraf(), {"strike": [100], "openInterest": [200]}, 100, .1) is None
    assert flow.dealer_gamma_exposure({"strike": [100], "openInterest": [200]}, _taraf(), 100, .1) is None


@pytest.mark.parametrize("iv_var", [False, True])
def test_o1b_tam_olcum_izgarasi_kapali_formla_ayni(iv_var):
    calls, puts = _taraf(), _taraf(200)
    for side in (calls, puts):
        side["gamma"] = [_gamma()]
        if not iv_var:
            del side["impliedVolatility"]
    sonuc = flow.dealer_gamma_exposure(calls, puts, 100, .1)
    beklenen = [-(_gamma(spot=s) if iv_var else _gamma()) * 100 * s * s
                for s in sonuc["grid_spot"]]
    assert sonuc["grid_net_gex"] == pytest.approx(beklenen, rel=1e-12)
    assert sonuc["gamma_source"] == ("iv_turetildi" if iv_var else "zincir")
    assert sonuc["n_gamma_eksik"] == 0


@pytest.mark.parametrize("oi", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("kismi", [False, True])
def test_o2_saat_eksik_oi_ile_akis_uydurmaz(oi, kismi):
    puts = _taraf(oi)
    if kismi:
        puts = {"strike": [100, 105], "openInterest": [200, oi], "impliedVolatility": [.3, .3]}
    assert flow.dealer_hedging_clock(_taraf(), puts, 100, .1, hours=1) is None
    # ATM, r=0: charm = -phi(d1)*sigma/(4*sqrt(T)); net OI = -100.
    phi = math.exp(-(.3 * math.sqrt(.1) / 2)**2 / 2) / math.sqrt(2 * math.pi)
    charm = -phi * .3 / (4 * math.sqrt(.1))
    beklenen = charm * 100 * 100 * 100 / (365 * 24)
    sonuc = flow.dealer_hedging_clock(_taraf(), _taraf(200), 100, .1, hours=1)
    assert sonuc["rows"][0]["net_usd"] == pytest.approx(beklenen)
    assert sonuc["rows"][0]["net_usd"] < 0


def test_o2_vanna_charm_eksik_satirlari_bir_kez_sayar():
    calls = {"strike": [100, 105, 110, 115, -1], "openInterest": [100, np.nan, 100, np.nan, np.nan],
             "impliedVolatility": [.3, .3, np.nan, np.nan, np.nan]}
    puts = {"strike": [100, 105], "openInterest": [200, np.nan], "impliedVolatility": [.3, .3]}
    sonuc = flow.dealer_vanna_charm(calls, puts, 100, .1)
    assert sonuc["n_eksik"] == 4 and type(sonuc["n_eksik"]) is int
    assert sonuc["n_contracts"] == 2
    phi = math.exp(-(.3 * math.sqrt(.1) / 2)**2 / 2) / math.sqrt(2 * math.pi)
    # ATM, r=0: vanna = φ√T/2, charm = −φσ/(4√T); net OI·100 = (100 − 200)·100.
    assert sonuc["vanna_exposure"] == pytest.approx(phi * math.sqrt(.1) / 2 * -10000)
    assert sonuc["charm_exposure"] == pytest.approx(-phi * .3 / (4 * math.sqrt(.1)) * -10000)
    # Verilmiş ama hiç ölçülemeyen taraf: tek taraflı toplam net diye sunulamaz.
    assert flow.dealer_vanna_charm(calls, {"strike": [100, 105]}, 100, .1) is None
    assert flow.dealer_vanna_charm(calls, {"strike": [0, np.nan]}, 100, .1)["n_contracts"] == 1


@pytest.mark.parametrize("ts,beklenen", [([.1, .2, 1.0], [.07, .08, .16]),
                                        ([.1, .2, .3], [.07, .08, .09])])
def test_o3_dupire_esit_ve_esit_olmayan_aralik(ts, beklenen):
    slices = [dict(t=t, params=dict(a=.04*t + .1*t*t, b=0, rho=0, m=0, sigma=.1)) for t in ts]
    sonuc = om.dupire_local_vol_svi(slices, n_k=5)
    # Strike-düz yüzeyde payda 1; iç noktada w'(T)=.04+.2T, uçlarda komşu eğimi.
    np.testing.assert_allclose(sonuc["local_vol"]**2, np.repeat(np.array(beklenen)[:, None], 5, axis=1))
    assert sonuc["n_calendar_ihlal"] == 0
    if ts[-1] == .3:
        w = np.array([.04*t + .1*t*t for t in ts])
        eski_merkezi = (w[2] - w[0]) / (ts[2] - ts[0])
        assert sonuc["local_vol"][1]**2 == pytest.approx(eski_merkezi)


def _zincir():
    return pd.DataFrame({"date": [pd.Timestamp("2024-02-05"), pd.Timestamp("2024-02-05 23:59")],
                         "act_symbol": ["AAA", "AAA"],
                         "expiration": ["2024-03-15", "2024-03-15"], "strike": [100, 105],
                         "call_put": ["Call", "Call"], "bid": [2., 2.], "ask": [2.5, 2.5]})


@pytest.mark.parametrize("saat", ["23:59", "00:00:01", "00:00:00.000000001"])
def test_o4_gun_ici_kotasyon_karar_gunune_sizamaz(saat):
    chain = _zincir()
    chain["date"] = chain["date"].astype("datetime64[ns]")
    chain.loc[1, "date"] = pd.Timestamp(f"2024-02-05 {saat}")
    history = dolt.prepare_option_history(chain, "AAA")
    snapshot = dolt.option_snapshot(history, as_of="2024-02-05")
    assert snapshot.strike.tolist() == [100]  # 105 kapanış kararına giremez.
    assert history.attrs["rejected_rows"] == 1
    chain.loc[1, "date"] = pd.Timestamp("2024-02-05 00:00:00")
    tam = dolt.prepare_option_history(chain, "AAA")
    assert tam.strike.tolist() == [100, 105] and tam.attrs["rejected_rows"] == 0


@pytest.mark.parametrize("history", [None, {}, [], "x", pd.DataFrame(), pd.DataFrame({"x": [1]}),
                                    pd.DataFrame({"date": ["bozuk"]})])
def test_o5_gecersiz_history_bos_cerceve(history):
    sonuc = dolt.option_snapshot(history, as_of="2024-02-05")
    assert isinstance(sonuc, pd.DataFrame) and sonuc.empty


@pytest.mark.parametrize("as_of", [None, np.nan, pd.NaT, "", "bozuk", [], {}, 123,
                                  "2024-02-05 12:00", "2024-02-05T00:00:00Z"])
def test_o5_gecersiz_as_of_bos_cerceve(as_of):
    for history in (pd.DataFrame(), dolt.prepare_option_history(_zincir(), "AAA")):
        sonuc = dolt.option_snapshot(history, as_of=as_of)
        assert isinstance(sonuc, pd.DataFrame) and sonuc.empty


@pytest.mark.parametrize("symbol", [None, np.nan, 123, [], {}, "", " "])
def test_o5_gecersiz_sembol_bos_cerceve(symbol):
    sonuc = dolt.prepare_option_history(_zincir(), symbol)
    assert isinstance(sonuc, pd.DataFrame) and sonuc.empty


@pytest.mark.parametrize("frame", [None, [], {}, "x", pd.DataFrame()])
def test_o5_gecersiz_frame_bos_cerceve(frame):
    sonuc = dolt.prepare_option_history(frame, "AAA")
    assert isinstance(sonuc, pd.DataFrame) and sonuc.empty


@pytest.mark.parametrize("as_of", [pd.Timestamp("2024-02-05"), pd.Timestamp("2024-02-05").date(),
                                  np.datetime64("2024-02-05")])
def test_o5_gecerli_gunluk_tarih_tipleri_korunur(as_of):
    history = dolt.prepare_option_history(_zincir(), "AAA")
    pd.testing.assert_frame_equal(dolt.option_snapshot(history, as_of=as_of), history)
