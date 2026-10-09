"""Teknik gösterge doğruluk testleri — Wilder yumuşatması (RMA), RSI/ATR/ADX."""
import numpy as np
import pandas as pd
import pytest

import nihai.signals.strategy as S


def _ohlc(closes):
    c = pd.Series(closes, dtype="float64")
    return pd.DataFrame({"open": c, "high": c + 1.0, "low": c - 1.0, "close": c, "volume": 1e6})


def test_rma_equals_wilder_recursion():
    # _rma, Wilder rekürsiyonunun ta kendisi olmalı: y_t = ((n-1)y_{t-1} + x_t)/n, seed=x_0
    s = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    n = 3
    got = S._rma(s, n).to_numpy()
    ref = np.full(len(s), np.nan)
    prev = s.iloc[0]
    ref[0] = prev
    for i in range(1, len(s)):
        prev = ((n - 1) * prev + s.iloc[i]) / n
        ref[i] = prev
    # min_periods=n → ilk n-1 NaN; geri kalan birebir eşleşmeli
    np.testing.assert_allclose(got[n - 1:], ref[n - 1:], rtol=1e-9)


def test_rsi_bounds_and_extremes():
    up = S.rsi(pd.Series(np.arange(1, 60, dtype="float64")))      # sürekli artan
    dn = S.rsi(pd.Series(np.arange(60, 1, -1, dtype="float64")))  # sürekli azalan
    assert up.dropna().between(0, 100).all()
    assert dn.dropna().between(0, 100).all()
    assert up.iloc[-1] > 99      # hiç kayıp yok → ~100
    assert dn.iloc[-1] < 1       # hiç kazanç yok → ~0


def test_rsi_no_loss_is_100():
    # Tüm hareketler yukarı → roll_dn=0 → RSI=100 (NaN/inf değil)
    r = S.rsi(pd.Series([10.0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]))
    assert r.iloc[-1] == 100.0


def test_rsi_duz_seri_notr_50_dir_100_degil():
    """Tur-28: roll_dn==0 VE roll_up==0 (tamamen DÜZ pencere: halt / illikit isim /
    seans-dışı bar) → RS=0/0 BELİRSİZ, doğru cevap NÖTR 50.

    Eskiden `out.where(roll_dn != 0, 100.0)` koşulsuz 100 yazıyordu = "azami aşırı-alım".
    Bu, _bias_and_confidence'a düz bir seriden +1.0 (azami boğa) bileşeni sokuyordu
    (clip((100−50)/25)=+1.0) ve meta_features'ın rsi14/rsi_slope'unu kirletiyordu."""
    r = S.rsi(pd.Series([100.0] * 40), 14)
    assert r.iloc[-1] == 50.0
    assert (r.dropna() == 50.0).all()        # düz pencerede HER bar nötr
    assert np.isnan(r.iloc[0])               # ısınma penceresi NaN kalmalı (maskelenmedi)


def test_rsi_duz_sonra_hareketli_gecis():
    """Düz bölüm 50, sonra gerçek hareket başlayınca normal RSI: maske yalnız DÜZ
    pencereye uygulanmalı, sonrasını dondurmamalı."""
    c = pd.Series([100.0] * 20 + [100.0 - i for i in range(1, 21)])
    r = S.rsi(c, 14)
    assert r.iloc[19] == 50.0                # hâlâ düz
    assert r.iloc[-1] < 20.0                 # 20 bar düşüş → aşırı-satım


def test_rsi_uses_wilder_not_sma():
    # Wilder RMA ile SMA-tabanlı RSI farklı olmalı (düzeltmenin etkili olduğunu kanıtla)
    np.random.seed(3)
    c = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 200)))
    wilder = S.rsi(c, 14).iloc[-1]
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    sma_rsi = (100 - 100 / (1 + up / dn.replace(0, np.nan))).iloc[-1]
    assert abs(wilder - sma_rsi) > 1e-6


def test_atr_positive_and_wilder():
    df = _ohlc(100 + np.cumsum(np.random.RandomState(1).normal(0, 1, 100)))
    a = S.atr(df, 14)
    assert (a.dropna() > 0).all()
    # ATR = RMA(TR); referansla eşleş
    tr = S._true_range(df)
    np.testing.assert_allclose(a.dropna().to_numpy(),
                               S._rma(tr, 14).dropna().to_numpy(), rtol=1e-9)


def test_adx_bounds():
    np.random.seed(5)
    df = _ohlc(100 + np.cumsum(np.random.normal(0, 1, 200)))
    a = S.adx(df, 14).dropna()
    assert a.between(0, 100).all()


def test_adx_high_in_strong_trend():
    # Güçlü tek-yön trend → ADX yüksek olmalı
    df = _ohlc(np.arange(100, 300, dtype="float64"))   # düz yukarı rampa
    assert S.adx(df, 14).iloc[-1] > 40


def test_bollinger_uses_population_std():
    c = pd.Series(100 + np.cumsum(np.random.RandomState(2).normal(0, 1, 50)))
    ma, up, lo = S.bollinger_bands(c, 20, 2.0)
    expected_sd = c.rolling(20).std(ddof=0)
    np.testing.assert_allclose((up - ma).dropna().to_numpy(),
                               (2.0 * expected_sd).dropna().to_numpy(), rtol=1e-9)
