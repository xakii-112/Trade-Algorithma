"""Tur-80: opsiyon önbelleği ve Greek aday maskesi için birebir eski-yol kanıtı."""
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from nihai.execution import backtest as bt
from nihai.risk.options_analyzer import GreekSet
from nihai.signals import strategy


def gunluk_barlar(n=900):
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(.12, 1, n))
    op = close + rng.normal(0, .3, n)
    return pd.DataFrame(dict(
        open=op, high=np.maximum(op, close) + rng.uniform(.1, 1, n),
        low=np.minimum(op, close) - rng.uniform(.1, 1, n), close=close,
        volume=rng.integers(1_000_000, 5_000_000, n).astype(float),
    ), index=pd.bdate_range("2020-01-01", periods=n))


class Onbelleksiz:
    def __init__(self, df):
        pass

    def upto(self, n):
        return None


def eski_measured_greeks(row):
    # Üretimdeki eski satır yolu: hata türü ve GreekSet dönüşümü de referansa dahil.
    values = [row[k] for k in ("delta", "gamma", "theta", "vega", "vol")]
    if not np.isfinite(values).all() or not -1 <= values[0] <= 1 or min(values[1], values[3]) < 0 or values[4] <= 0:
        return None
    return GreekSet(*map(float, values))


def eski_maske(frame, measured_greeks=eski_measured_greeks):
    return frame.apply(lambda row: measured_greeks(row) is not None, axis=1)


@pytest.fixture(autouse=True)
def yerel_modelden_bagimsiz(monkeypatch):
    # Sinyal/risk gerçek; yalnız bu optimizasyonla ilgisiz yerel model ve VIX kapalı.
    cfg = replace(strategy.CONFIG, strategy=replace(
        strategy.CONFIG.strategy, meta_gate_mode="off", enable_vix_regime_gate=False))
    monkeypatch.setattr(strategy, "CONFIG", cfg)


def denk_sonuc(monkeypatch, runner, **kwargs):
    yeni = runner("AAA", strategy_mode="event_breakout", initial_equity=10_000, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(bt, "_SharedIndicators", Onbelleksiz)
        patch.setattr(bt, "_measured_greeks_mask", eski_maske)
        eski = runner("AAA", strategy_mode="event_breakout", initial_equity=10_000, **kwargs)
    # Tolerans yok; yeni eklenen alanlar da otomatik olarak karşılaştırılır.
    assert asdict(yeni) == asdict(eski)
    return yeni


def kucuk_zincir(bars):
    # Aylık vadeler; bir kontrat sonraki günde de mevcut. Eşit uzaklıklı strike'lar
    # seçim sırasını, sıfır IV ve kısa DTE ise aday elemesini sınar.
    expirations = pd.date_range(bars.index[0], bars.index[-1] + pd.Timedelta(days=60), freq="MS")
    return pd.DataFrame([
        dict(date=day.normalize().tz_localize(None), act_symbol="AAA", expiration=expiry,
             strike=strike, call_put=side, bid=1.4, ask=1.5, vol=vol,
             delta=.5 if side == "c" else -.5, gamma=.02, theta=-.03, vega=.1)
        for day in bars.index
        for expiry in expirations if 0 <= (expiry - day.tz_localize(None).normalize()).days <= 60
        for side in ("c", "p")
        for strike, vol in ((95., 0.), (100., .3), (105., .3))
    ])


def test_synth_bit_denklik(monkeypatch):
    r = denk_sonuc(monkeypatch, bt.backtest_option_synth, df=gunluk_barlar())
    assert r.trades_count > 0
    print(f"synth: bar={r.bars_total}, islem={r.trades_count}, alan={len(asdict(r))}")


def test_chain_bit_denklik(monkeypatch):
    bars = gunluk_barlar(220)
    chain = kucuk_zincir(bars)
    # Önbellek normalize ve NYSE tatil süzgecinden SONRA kurulmalı.
    bars.index = (bars.index + pd.Timedelta(hours=16)).tz_localize("America/New_York")
    r = denk_sonuc(monkeypatch, bt.backtest_option_chain, df=bars, chain_df=chain)
    assert r.bars_total < len(bars)
    assert r.trades_count > 0
    assert any(f["side"] == "buy" for f in r.option_data["fills"])
    assert all(f["strike"] == 100 for f in r.option_data["fills"])
    print(f"chain: ham_bar={len(bars)}, seans={r.bars_total}, islem={r.trades_count}, "
          f"dolum={len(r.option_data['fills'])}, zincir={len(chain)}, alan={len(asdict(r))}")


@pytest.mark.parametrize("n,duz", [(0, False), (12, False), (900, True)])
def test_dejenere(monkeypatch, n, duz):
    bars = gunluk_barlar(n)
    if duz:
        bars.loc[:, ["open", "high", "low", "close"]] = 100.
    r = denk_sonuc(monkeypatch, bt.backtest_option_synth, df=bars)
    assert r.trades_count == 0


@pytest.mark.parametrize("n,warmup", [(12, 0), (220, 60)])
def test_chain_dejenere(monkeypatch, n, warmup):
    bars = gunluk_barlar(n)
    bars.loc[:, ["open", "high", "low", "close"]] = 100.
    r = denk_sonuc(monkeypatch, bt.backtest_option_chain, df=bars,
                   chain_df=kucuk_zincir(bars), warmup_bars=warmup)
    assert r.trades_count == 0 and r.option_data["fills"] == []


def maske_izgarasi():
    rng = np.random.default_rng(80)
    columns = ["delta", "gamma", "theta", "vega", "vol"]
    random = rng.uniform(-1.5, 1.5, (1000, 5))
    edges = []
    for col in range(5):
        for value in (np.nan, np.inf, -np.inf, -1., 1., 0., -0., -.01,
                      np.nextafter(0., 1.), np.nextafter(1., 2.), np.nextafter(-1., -2.)):
            row = [.5, .02, -.03, .1, .3]
            row[col] = value
            edges.append(row)
    frame = pd.DataFrame(np.vstack([random, edges]), columns=columns)
    # Tekil/sıralı indeks varsayımı yok; seçimde sıra ve tekrarlı etiketler korunur.
    frame.index = np.arange(len(frame))[::-1] // 2
    return frame


@pytest.mark.parametrize("dtype", ["float64", "float32", "object"])
def test_maske_denklik(dtype):
    frame = maske_izgarasi().astype(dtype)
    frame["expiration"] = pd.Timestamp("2024-03-15")
    frame["call_put"] = "c"
    pd.testing.assert_series_equal(bt._measured_greeks_mask(frame, eski_measured_greeks),
                                   eski_maske(frame), check_exact=True)


def test_sayisal_maskede_satir_cagrisi_yok():
    def satir_yolu(row):
        pytest.fail("Sayısal aday başına Python çağrısı yapılmamalı")

    frame = maske_izgarasi()
    pd.testing.assert_series_equal(bt._measured_greeks_mask(frame, satir_yolu), eski_maske(frame))


@pytest.mark.parametrize("value", ["0.3", "bozuk", None, pd.NA])
@pytest.mark.parametrize("dtype", ["object", "string"])
def test_maske_donusum_hatasi_aynen_kalir(value, dtype):
    frame = pd.DataFrame([[.5, .02, -.03, .1, .3]], columns=["delta", "gamma", "theta", "vega", "vol"])
    frame["vol"] = pd.Series([value], dtype=dtype)
    with pytest.raises(TypeError) as eski:
        eski_maske(frame)
    with pytest.raises(TypeError) as yeni:
        bt._measured_greeks_mask(frame, eski_measured_greeks)
    assert str(yeni.value) == str(eski.value)


def test_maske_bos_ve_tamsayi():
    frame = pd.DataFrame([[1, 0, -1, 0, 1], [-1, 0, -1, 0, 1], [0, 0, 0, 0, 0]],
                         columns=["delta", "gamma", "theta", "vega", "vol"])
    pd.testing.assert_series_equal(bt._measured_greeks_mask(frame, eski_measured_greeks), eski_maske(frame))
    pd.testing.assert_series_equal(bt._measured_greeks_mask(frame.iloc[:0], eski_measured_greeks),
                                   eski_maske(frame.iloc[:0]))
