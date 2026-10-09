"""Tur-77: nedensel U-eğrisi, tamamlanmış slot ve likidite çekilmesi sözleşmeleri."""

import json

import numpy as np
import pandas as pd
import pytest

from nihai.ai import microstructure as ms


def _bars(days=25, start="2024-06-03", noise=0.05):
    """ET seanslarından naive-UTC barlar; tatilsiz hafta içi ızgarası ve sabit seed."""
    dates = pd.bdate_range(start, periods=days)
    stamps = pd.DatetimeIndex([
        stamp for day in dates
        for stamp in pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=390, freq="min")
    ]).tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)
    minute = np.arange(390)
    curve = 1000 * (1 + 3 * ((minute - 195) / 195) ** 2)
    values = np.tile(curve, days) * np.random.default_rng(77).lognormal(0, noise, days * 390)
    return stamps, values


def test_ucurve_geri_kazanim_ve_python_tipleri():
    index, values = _bars()
    result = ms.intraday_profile(index, values, index_tz="UTC")
    assert result is not None
    assert result["n_sessions"] == 24
    assert result["n_days"] == [24] * 13
    assert result["slots"][0] == "09:30" and result["slots"][-1] == "15:30"
    assert result["median"][0] / result["median"][result["slots"].index("12:30")] > 3
    # Σm ve Σm² kapalı formlarıyla her otuz dakikalık slotun gürültüsüz toplamı.
    expected = []
    for first in range(0, 390, 30):
        last = first + 29
        sum_m = (first + last) * 30 / 2
        sum_m2 = (last * (last + 1) * (2 * last + 1)
                  - (first - 1) * first * (2 * first - 1)) / 6
        expected.append(1000 * (30 + 3 * (sum_m2 - 390 * sum_m + 30 * 195**2) / 195**2))
    np.testing.assert_allclose(result["median"], expected, rtol=0.05)
    assert all(lo <= med <= hi for lo, med, hi in zip(
        result["q25"], result["median"], result["q75"], strict=True,
    ))
    assert all(type(v) is float for key in ("median", "q25", "q75") for v in result[key])
    assert all(type(v) is int for v in result["n_days"])
    json.dumps(result, allow_nan=False)


def test_lookahead_ve_determinizm():
    index, values = _bars()
    original = values.copy()
    before = ms.intraday_profile(index, values, index_tz="UTC")
    assert before == ms.intraday_profile(index, values, index_tz="UTC")
    values[-390:] *= 100
    assert before == ms.intraday_profile(index, values, index_tz="UTC")
    assert ms.intraday_profile(index, values, index_tz="UTC", exclude_last_session=False)["n_sessions"] == 25
    values[-390:] = np.nan
    assert before == ms.intraday_profile(index, values, index_tz="UTC")
    assert ms.intraday_anomaly(index, values, index_tz="UTC") is None
    result = ms.intraday_anomaly(index, original, index_tz="UTC")
    assert result == ms.intraday_anomaly(index, original, index_tz="UTC")


def test_saat_dilimi_esdegerligi_ve_zorunlu_parametre():
    index, values = _bars()
    aware = index.tz_localize("UTC").tz_convert("America/New_York")
    result = ms.intraday_profile(index, values, index_tz="UTC")
    assert result == ms.intraday_profile(aware, values, index_tz="America/New_York")
    assert result == ms.intraday_profile(aware.tz_localize(None), values, index_tz="America/New_York")
    assert ms.intraday_anomaly(index, values, index_tz="UTC") == ms.intraday_anomaly(
        aware, values, index_tz="America/New_York",
    )
    with pytest.raises(TypeError):
        ms.intraday_profile(index, values)
    with pytest.raises(TypeError):
        ms.intraday_anomaly(index, values)


def test_dst_gecisinde_acilis_slotu():
    index, values = _bars(days=10, start="2024-03-04", noise=0)
    assert index[0].strftime("%H:%M") == "14:30"
    assert index[5 * 390].strftime("%H:%M") == "13:30"
    result = ms.intraday_profile(index, values, index_tz="UTC")
    assert result["slots"][0] == "09:30" and result["n_days"] == [9] * 13
    assert result["median"][0] == pytest.approx(values[:30].sum())


def test_yarim_gun_ve_yetersiz_slot_sifirla_doldurulmaz():
    index, values = _bars(days=6)
    keep = np.ones(values.size, dtype=bool)
    keep[210:390] = False  # İlk seans 13:00'te biter; geçmişte beş seans var.
    result = ms.intraday_profile(index[keep], values[keep], index_tz="UTC")
    assert result["n_sessions"] == 5
    assert result["n_days"] == [5] * 7 + [4] * 6
    for key in ("median", "q25", "q75"):
        assert all(v > 0 for v in result[key][:7])
        assert result[key][7:] == [None] * 6
    anomaly = ms.intraday_anomaly(index[keep], values[keep], index_tz="UTC")
    assert anomaly["median"] is None and anomaly["ratio"] is None
    assert anomaly["z_robust"] is None and anomaly["anormal"] is False


@pytest.mark.parametrize("index,values", [([], []), (["2024-06-03"], [np.nan]),
                                         (["2024-06-03"], []), (["bozuk"], [1]),
                                         (["2024-06-03 13:30"], ["bozuk"])])
def test_dejenere_veri(index, values):
    assert ms.intraday_profile(index, values, index_tz="UTC") is None
    assert ms.intraday_anomaly(index, values, index_tz="UTC") is None


def test_tek_seans_sabit_hacim_ve_tamamlanmamis_seans():
    index, values = _bars(days=6)
    assert ms.intraday_profile(index[:390], values[:390], index_tz="UTC") is None
    assert ms.intraday_profile(index, np.full(values.size, np.nan), index_tz="UTC") is None
    result = ms.intraday_anomaly(index, np.ones(values.size), index_tz="UTC")
    assert result["z_robust"] is None and result["anormal"] is False
    assert result["ratio"] == 1 and result["bar_minutes"] == 1
    for extra in (1, 7):
        assert ms.intraday_anomaly(index[:1950 + extra], values[:1950 + extra], index_tz="UTC") is None
    result = ms.intraday_anomaly(index, np.zeros(values.size), index_tz="UTC")
    assert result["median"] == 0 and result["ratio"] is None and result["z_robust"] is None


def test_anomali_tamamlanmis_ve_kismi_slot():
    index, values = _bars()
    today = 24 * 390
    values[today + 30:today + 60] *= 10
    for offset in (60, 61):  # Son bar 10:29 veya 10:30: 10:00 slotu tamamdır.
        result = ms.intraday_anomaly(index[:today + offset], values[:today + offset], index_tz="UTC")
        assert result["slot"] == "10:00" and result["bar_minutes"] == 1
        assert result["z_robust"] >= 3 and result["anormal"] is True
        history = values[:today].reshape(24, 390)[:, 30:60].sum(axis=1)
        logs = np.log(history)
        expected = (np.log(result["value"]) - np.median(logs)) / (
            1.4826 * np.median(np.abs(logs - np.median(logs)))
        )
        assert result["z_robust"] == pytest.approx(expected)
        assert result["ratio"] == pytest.approx(result["value"] / np.median(history))
    partial = ms.intraday_anomaly(index[:today + 38], values[:today + 38], index_tz="UTC")
    assert partial["slot"] == "09:30"  # Son bar 10:07: ikinci slot kullanılmaz.
    values[today + 30:today + 60] = np.nan
    assert ms.intraday_anomaly(index[:today + 60], values[:today + 60], index_tz="UTC") is None


@pytest.mark.parametrize("agg", ["sum", "median", "mean"])
def test_toplama_nan_seans_siniri_ve_sirasiz_girdi(agg):
    index, values = _bars(days=6)
    values[0], values[1] = np.nan, np.inf
    index = index.append(pd.DatetimeIndex(["2024-06-03 13:29", "2024-06-03 20:00"]))
    values = np.concatenate([values, [1e20, 1e20]])
    result = ms.intraday_profile(index, values, index_tz="UTC", agg=agg)
    reduce = {"sum": np.sum, "median": np.median, "mean": np.mean}[agg]
    history = [reduce(values[2:30])] + [reduce(values[d * 390:d * 390 + 30]) for d in range(1, 5)]
    assert result["median"][0] == pytest.approx(np.median(history))
    assert result["q25"][0] == pytest.approx(np.quantile(history, 0.25))
    assert result["q75"][0] == pytest.approx(np.quantile(history, 0.75))
    assert result["n_days"][0] == 5
    assert result == ms.intraday_profile(index[::-1], values[::-1], index_tz="UTC", agg=agg)


@pytest.mark.parametrize("kwargs", [{"slot_minutes": 0}, {"slot_minutes": 1.5}, {"min_days": 0},
                                    {"session_open": "25:00"}, {"session_close": "09:00"},
                                    {"index_tz": "bozuk"}, {"session_tz": "bozuk"}])
def test_gecersiz_ayarlar(kwargs):
    index, values = _bars(days=6)
    options = {"index_tz": "UTC", **kwargs}
    assert ms.intraday_profile(index, values, **options) is None
    assert ms.intraday_anomaly(index, values, **options) is None


def test_bilinmeyen_agg_api_ihlali():
    for function in (ms.intraday_profile, ms.intraday_anomaly):
        with pytest.raises(ValueError):
            function([], [], index_tz="UTC", agg="bozuk")


def test_likidite_cekilmesi_ve_olcum_yoklugu():
    volume = {"session": "2024-06-03", "slot": "10:00", "z_robust": -4.0}
    spread = {**volume, "z_robust": 4.0}
    result = ms.liquidity_withdrawal(volume, spread)
    assert result == {"cekilme": True, "volume_z": -4.0, "spread_z": 4.0,
                      "session": "2024-06-03", "slot": "10:00"}
    assert ms.liquidity_withdrawal(volume, spread, z_threshold=4)["cekilme"] is True
    assert ms.liquidity_withdrawal(volume, spread, z_threshold=5)["cekilme"] is False
    assert ms.liquidity_withdrawal(spread, volume)["cekilme"] is False
    for invalid in (None, {}, {**spread, "z_robust": None}, {**spread, "z_robust": np.nan},
                    {**spread, "slot": "10:30"}, {**spread, "session": "2024-06-04"}):
        assert ms.liquidity_withdrawal(volume, invalid) is None
        assert ms.liquidity_withdrawal(invalid, volume) is None
