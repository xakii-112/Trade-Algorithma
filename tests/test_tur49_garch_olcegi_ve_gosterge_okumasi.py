"""
test_tur49_garch_olcegi_ve_gosterge_okumasi.py — Tur-49: ÖLÇEK ve OKUMA.

Bu tur iki bağımsız kusur + bir optimizasyon ailesi kapatır.

KUSUR-49-1 · `garch11_fit` MLE'yi GETİRİ ÖLÇEĞİNDE koşuyordu. Günlük getiride
  var ~ 1e-4'tür; SLSQP'nin varsayılan sonlu-fark adımı (~1,5e-8) ve `ftol` bu
  ölçekte omega yönünde anlamsız kalır. ÖLÇÜLDÜ (296 gerçek S&P sembolü, 1000
  günlük log-getiri): %8,4'ünde `uncond_var` örneklem varyansının 5 katından
  BÜYÜK, %1,4'ünde 0,2 katından KÜÇÜK, %1,7'sinde fit tamamen None; en kötü
  sapmalar 18.068x ve 1,3e-06x. GARCH(1,1) ölçek-EŞDEĞİŞKENDİR (r->c*r ise
  omega->c^2*omega, alpha/beta değişmez) — artıklar birim varyansa normalize
  edilip omega geri ölçeklenince: fit None %0,0 · 0,2x altı %0,0 · 5x üstü 1/296
  (FICO, persistence 0,996 — köşede değil, gerçek duyarlılık). Çoklu-başlangıç
  TEK BAŞINA yetmiyordu: beş başlangıç da AYNI bozuk yüzeyde arıyordu.

KUSUR-49-2 · `alpha+beta` durağanlık KISITININ kenarına çakıldığında (IGARCH)
  `uncond_var` yine de SAYI basıyordu — ama o sayı bir tahmin değil, kısıttan
  türeyen bir yapaydır (ölçüldü: örneklem varyansının 34-528 katı). Kusuru ifade
  edemeyen bir alan onu açığa da çıkaramaz (Tur-47 dersi) → köşede None.

KUSUR-49-3 · Modül sözleşmesi "hiçbir girdi İSTİSNA fırlatmaz" diyordu; CPython'da
  float `**` TAŞMADA `OverflowError` fırlatır (çarpma ise numpy gibi `inf` verir).
  `ewma_vol` ve `bocpd` iç döngülerinde `x ** 2` kullanıyordu → sözleşme ihlali.

OPT-49 · Göstergeler Tur-7/43/45'te önbelleğe alınmıştı ama OKUNMALARI hâlâ
  pandas'tı. `Series.iloc[int]` ÖLÇÜLDÜ 2,573 µs, `ndarray[int]` 0,126 µs (20,4x).
  Beş detektör + `extract_features` numpy konumsal okumaya geçti; 20.000 barlık
  backtest 5 sembolde 163,70 s -> 66,36 s (**2,47x**), TÜM metrikler birebir aynı.

Bu dosya kusurların KENDİ İFADELERİNİ kaynakta arar (regresyon kapısı): eski
ölçeksiz çağrı, eski `** 2` taşma yolu ve eski pandas okumaları geri gelirse KIRILIR.
"""
from __future__ import annotations

import io
import math
import tokenize
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nihai.ai import advanced_math, quant_math
from nihai.signals import strategy as S
from nihai.signals.meta_features import extract_features

_QM_SRC = Path(quant_math.__file__).read_text(encoding="utf-8")
_AM_SRC = Path(advanced_math.__file__).read_text(encoding="utf-8")
_ST_SRC = Path(S.__file__).read_text(encoding="utf-8")
_MF_SRC = (Path(S.__file__).parent / "meta_features.py").read_text(encoding="utf-8")


def _kod(src: str, bas: str, son: str) -> str:
    """[bas, son) aralığındaki gövdenin YALNIZ KOD kısmı — yorum ve dizgiler atılır.

    Bu ayrım kapının kendisi kadar önemlidir: bu turun onarım YORUMLARI kusurun eski
    ifadesini (`.iloc[-30:]`) bilerek ALINTILAR. Ham metinde arayan bir kapı kendi
    açıklamasına takılır ve "kusur geri geldi" der — yanlış alarm bir kapıyı işe
    yaramaz hâle getirir (Tur-48: her metinde ateşlenen uyarı hiç ateşlenmiyor demektir).
    """
    i = src.index(bas)
    govde = src[i:src.index(son, i + len(bas))]
    out: list[str] = []
    okuyucu = io.StringIO(govde).readline
    try:
        for tok in tokenize.generate_tokens(okuyucu):
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                out.append(tok.string)
    except (tokenize.TokenError, IndentationError):   # kırpık gövde → satır bazlı yedek
        return " ".join(satir.split("#", 1)[0] for satir in govde.splitlines())
    return " ".join(out)


def _sim_garch(omega: float, alpha: float, beta: float, n: int, seed: int) -> np.ndarray:
    """Test-yardımcısı (üretim kodu DEĞİL): bilinen parametrelerle GARCH(1,1) simülasyonu."""
    rng = np.random.default_rng(seed)
    s2 = omega / (1.0 - alpha - beta)
    out = np.empty(n)
    for t in range(n):
        e = rng.normal(0.0, math.sqrt(s2))
        out[t] = e
        s2 = omega + alpha * e * e + beta * s2
    return out


def _bars(n: int = 400, *, seed: int = 49) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1.0, n))
    return pd.DataFrame(
        {"open": c, "high": c + np.abs(rng.normal(0, 0.5, n)),
         "low": c - np.abs(rng.normal(0, 0.5, n)), "close": c,
         "volume": rng.integers(1e4, 1e6, n).astype(float)},
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )


# ---------------------------------------------------------------------------
# KUSUR-49-1 — ölçek
# ---------------------------------------------------------------------------
def test_garch11_fit_olcek_esdegisken():
    """r -> 100*r: alpha/beta DEĞİŞMEZ, omega 10.000 katına çıkar."""
    r = _sim_garch(1e-6, 0.08, 0.90, 1500, seed=7)
    a, b = quant_math.garch11_fit(r), quant_math.garch11_fit(r * 100.0)
    assert a is not None and b is not None
    assert b["alpha"] == pytest.approx(a["alpha"], abs=5e-3)
    assert b["beta"] == pytest.approx(a["beta"], abs=5e-3)
    assert b["omega"] / a["omega"] == pytest.approx(1e4, rel=0.05)


@pytest.mark.parametrize("seed", range(12))
def test_garch11_fit_uncond_var_ornek_varyanstan_kopmaz(seed: int):
    """ASIL KUSURUN KAPISI. Gerçek GARCH verisinde koşulsuz varyans örneklem
    varyansının makul bandında kalmalı; onarımdan ÖNCE 10^1-10^4 kat sapabiliyordu."""
    r = _sim_garch(1e-6, 0.06, 0.91, 900, seed=seed)
    fit = quant_math.garch11_fit(r)
    assert fit is not None, "fit sessizce None dönmemeli"
    if fit["uncond_var"] is None:      # IGARCH köşesi — dürüst None, sayı basılmıyor
        assert fit["persistence"] >= quant_math._GARCH_PERSIST_MAX - 1e-9
        return
    oran = fit["uncond_var"] / float(np.var(r))
    assert 0.2 <= oran <= 5.0, f"uncond_var/örneklem = {oran:.4g}"


@pytest.mark.parametrize("n", [200, 500, 1000, 3000])
def test_garch11_fit_arch_etkisi_yokken_de_saglam(n: int):
    """i.i.d. normal getiri (ARCH etkisi YOK) GARCH-MLE'nin en kaygan yüzeyidir.
    Onarımdan önce bu girdiler ya None ya da 10^2-10^5 kat sapmış uncond_var veriyordu."""
    rng = np.random.default_rng(11)
    r = rng.normal(0.0004, 0.011, n)
    fit = quant_math.garch11_fit(r)
    assert fit is not None
    if fit["uncond_var"] is not None:
        assert 0.2 <= fit["uncond_var"] / float(np.var(r)) <= 5.0


def test_garch11_fit_sozlesme_alanlari():
    fit = quant_math.garch11_fit(_sim_garch(1e-6, 0.08, 0.90, 800, seed=3))
    assert fit is not None
    assert set(fit) == {"omega", "alpha", "beta", "mean", "loglik", "persistence",
                        "uncond_var", "last_sigma2", "last_resid2", "n"}
    assert fit["persistence"] == pytest.approx(fit["alpha"] + fit["beta"])


def test_garch11_fit_loglik_jacobian_duzeltmesi_uygulanmis():
    """Olabilirlik ÖLÇEKLİ artıkların değil ORİJİNAL getirilerin olabilirliğidir.
    Düzeltme unutulsaydı değer 0,5*n*ln(var0) kadar KÜÇÜK kalırdı (günlük ölçekte
    bu ~ -3.700'dür, yani işaret bile ters dönerdi)."""
    r = _sim_garch(1e-6, 0.08, 0.90, 800, seed=5)
    fit = quant_math.garch11_fit(r)
    assert fit is not None
    sabit_varyans_loglik = -0.5 * fit["n"] * (math.log(2 * math.pi * float(np.var(r))) + 1.0)
    assert fit["loglik"] > sabit_varyans_loglik     # GARCH sabit-varyanstan İYİ olmalı
    assert fit["loglik"] > 0.0                      # günlük ölçekte yoğunluk > 1


def test_garch11_sigma2_yolu_python_dongusuyle_bit_denk():
    """lfilter (C) yinelemesi Python döngüsüyle BİREBİR aynı sayıları vermeli —
    hız kazancı sayıları değiştirmeyi haklı çıkarmaz."""
    rng = np.random.default_rng(2)
    resid2 = rng.normal(0, 0.012, 800) ** 2
    var0 = float(resid2.mean())
    for omega, alpha, beta in ((1e-6, 0.08, 0.90), (1e-12, 0.0, 0.999),
                               (5e-5, 0.5, 0.3), (1e-6, 0.0, 0.0)):
        beklenen = np.empty(resid2.size)
        beklenen[0] = var0
        for t in range(1, resid2.size):
            beklenen[t] = omega + alpha * resid2[t - 1] + beta * beklenen[t - 1]
        alinan = quant_math._garch11_sigma2_path(omega, alpha, beta, resid2, var0)
        assert np.array_equal(alinan, beklenen), (omega, alpha, beta)


# ---------------------------------------------------------------------------
# KUSUR-49-2 — IGARCH köşesinde SAYI BASMA
# ---------------------------------------------------------------------------
def test_igarch_kosesinde_uncond_var_none():
    """Kalıcılığı ~1 olan bir seride koşulsuz varyans TANIMSIZDIR. Sayı basmak,
    kısıtın artığını tahmin gibi göstermek olurdu."""
    r = _sim_garch(1e-8, 0.10, 0.8999, 1200, seed=17)     # kalıcılık ~0,9999
    fit = quant_math.garch11_fit(r)
    assert fit is not None
    if fit["persistence"] >= quant_math._GARCH_PERSIST_MAX - 1e-9:
        assert fit["uncond_var"] is None
    # köşede olmayan fit sayı basmalı (alan her koşulda None DEĞİL)
    saglikli = quant_math.garch11_fit(_sim_garch(2e-6, 0.15, 0.70, 1200, seed=4))
    assert saglikli is not None and saglikli["uncond_var"] is not None


def test_igarch_kosesinde_kisa_ufuk_ongoru_yine_calisir():
    """Tanımsız olan uzun-ufuk SINIRIdır; koşullu varyans (h=1) tanımlıdır."""
    params = {"omega": 1e-8, "alpha": 0.5, "beta": 0.499, "last_sigma2": 1e-4,
              "last_resid2": 2e-4}
    v = quant_math.garch11_forecast(params, horizon=1)
    assert v is not None and math.isfinite(v) and v > 0


# ---------------------------------------------------------------------------
# KUSUR-49-3 — taşmada İSTİSNA YOK sözleşmesi
# ---------------------------------------------------------------------------
def test_ewma_vol_tasmada_istisna_degil_none():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert quant_math.ewma_vol(np.full(60, 1e300)) is None
    assert quant_math.ewma_vol(np.array([0.01, -0.02, 0.015])) is not None


def test_bocpd_tasmada_istisna_degil_sonlu():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = advanced_math.bocpd(np.full(60, 1e300))
    assert out["n"] == 60
    assert np.all(np.isfinite(out["changepoint_prob"]))


def test_saf_matematik_tasma_sinirinda_istisna_atmaz():
    """Sözleşme taraması: taşma sınırındaki girdi hiçbir giriş noktasında patlamamalı."""
    buyuk = np.full(60, 1e300)
    for fn in (quant_math.ewma_vol, quant_math.vol_close_to_close, quant_math.sharpe,
               quant_math.sortino, quant_math.var_historical, quant_math.cvar_historical,
               quant_math.ulcer_index, quant_math.downside_deviation):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(buyuk)          # İSTİSNA = başarısızlık


# ---------------------------------------------------------------------------
# OPT-49 — numpy okuma PARİTESİ (sayılar değişmedi)
# ---------------------------------------------------------------------------
def test_np_erisimciler_pandas_serileriyle_ayni():
    df = _bars()
    p = S._SharedIndicators(df)
    assert np.array_equal(p.rsi14_np(), p.rsi14().to_numpy(dtype=float), equal_nan=True)
    assert np.array_equal(p.adx_np(), p.adx_series().to_numpy(dtype=float), equal_nan=True)
    assert np.array_equal(p.ema_np(20), p.ema(20).to_numpy(dtype=float), equal_nan=True)
    assert np.array_equal(p.bb_width20_np(), p.bb_width20().to_numpy(dtype=float),
                          equal_nan=True)
    for got, exp in zip(p.macd_np(), p.macd(), strict=True):
        assert np.array_equal(got, exp.to_numpy(dtype=float), equal_nan=True)
    for got, exp in zip(p.bb20_np(), p.bb20(), strict=True):
        assert np.array_equal(got, exp.to_numpy(dtype=float), equal_nan=True)
    assert np.array_equal(p.col_np("high"), df["high"].to_numpy(dtype=float))
    assert p.col_np("yok_boyle_sutun") is None


def test_slice_np_gorunumleri_gelecegi_icermez():
    """NEDENSELLİK KAPISI: kırpılmış numpy görünümü, dilim üzerinde SIFIRDAN
    hesaplananla aynı olmalı ve uzunluğu n'i AŞMAMALI."""
    df = _bars(360)
    p = S._SharedIndicators(df)
    for n in (120, 250, 359):
        sl = p.upto(n)
        ref = S._SharedIndicators(df.iloc[:n])
        assert sl.rsi14_np().size == n
        assert np.array_equal(sl.rsi14_np(), ref.rsi14_np(), equal_nan=True)
        assert np.array_equal(sl.closes(), ref.closes(), equal_nan=True)
        assert np.array_equal(sl.bb_width20_np(), ref.bb_width20_np(), equal_nan=True)
        assert np.array_equal(sl.col_np("high"), ref.col_np("high"))
        assert sl.ema_at(20, -1) == ref.ema_at(20, -1)
        assert sl.breakout_at(-1, 20) == ref.breakout_at(-1, 20)


def test_donchian_penceresi_breakout_at_ile_denk():
    """`.iloc[-period-1:-1].max()` ile `rolling(period).max().shift(1)` son konumu
    BİT-DENK — `detect_donchian` artık kendi penceresini kurmak yerine bunu paylaşıyor."""
    df = _bars(300)
    p = S._SharedIndicators(df)
    for period in (10, 20, 55):
        hi, lo = p.breakout_at(-1, period)
        assert hi == float(df["high"].iloc[-period - 1:-1].max())
        assert lo == float(df["low"].iloc[-period - 1:-1].min())


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_detektorler_pandas_referansiyla_ayni_karar(seed: int):
    """Numpy gövdeleri, aynı seriden okuyan pandas referansıyla AYNI kararı vermeli."""
    df = _bars(320, seed=seed)
    p = S._SharedIndicators(df)
    for n in range(200, 320, 7):
        sl, sub = p.upto(n), df.iloc[:n]
        m, _, hist = sl.macd()
        a, r = S.detect_macd(sub, shared=sl)
        if not (np.isnan(hist.iloc[-2]) or np.isnan(hist.iloc[-1])):
            bekle = (S.SignalAction.LONG if (hist.iloc[-2] <= 0 < hist.iloc[-1])
                     else S.SignalAction.SHORT if (hist.iloc[-2] >= 0 > hist.iloc[-1])
                     else S.SignalAction.WAIT)
            assert a == bekle
            if a != S.SignalAction.WAIT:
                assert f"macd_{m.iloc[-1]:+.3f}" in r
        upper = sub["high"].iloc[-21:-1].max()
        lower = sub["low"].iloc[-21:-1].min()
        price = float(sub["close"].iloc[-1])
        adx_son = sl.adx_last()
        a2, _ = S.detect_donchian(sub, shared=sl)
        if not np.isnan(adx_son) and adx_son < S.CONFIG.strategy.adx_trend_threshold:
            assert a2 == S.SignalAction.WAIT
        elif price > upper:
            assert a2 == S.SignalAction.LONG
        elif price < lower:
            assert a2 == S.SignalAction.SHORT


def test_extract_features_pandas_referansiyla_ayni():
    df = _bars(320)
    p = S._SharedIndicators(df)

    class _Sig:
        action = S.SignalAction.LONG
        rr_ratio = 2.0
        quality_score = 3.5
        reasons = ["CONFLUENCE_2"]

    for n in (250, 300, 319):
        sl, sub = p.upto(n), df.iloc[:n]
        f = extract_features(sub, _Sig(), shared=sl)
        price = float(sub["close"].iloc[-1])
        ma, upper, lower = sl.bb20()
        rsi_s = sl.rsi14()
        assert f["dist_hi20"] == pytest.approx(
            (price - float(sub["high"].iloc[-21:-1].max())) / price)
        assert f["dist_lo20"] == pytest.approx(
            (price - float(sub["low"].iloc[-21:-1].min())) / price)
        assert f["bb_width"] == pytest.approx(
            (upper.iloc[-1] - lower.iloc[-1]) / ma.iloc[-1])
        assert f["rsi14"] == pytest.approx(float(rsi_s.iloc[-1]))
        assert f["rsi_slope"] == pytest.approx(float(rsi_s.iloc[-1] - rsi_s.iloc[-6]))
        avg_v = float(sub["volume"].iloc[-21:-1].mean())
        assert f["rvol"] == pytest.approx(float(sub["volume"].iloc[-1]) / avg_v)
        assert f["ret_5"] == pytest.approx(price / float(sub["close"].iloc[-6]) - 1)


def test_onbelleksiz_ve_onbellekli_yol_ayni_karari_verir():
    """`shared=None` yolu (dış çağıranlar, canlı) ile paylaşımlı yol AYNI kalmalı."""
    df = _bars(340)
    p = S._SharedIndicators(df)
    for n in range(220, 340, 11):
        sl, sub = p.upto(n), df.iloc[:n]
        for fn in (S.detect_macd, S.detect_donchian, S.detect_mean_reversion,
                   S.detect_squeeze, S.detect_rsi_divergence):
            assert fn(sub) == fn(sub, shared=sl), fn.__name__


# ---------------------------------------------------------------------------
# REGRESYON KAPILARI — kusurun KENDİ İFADESİ kaynakta ARANIR
# ---------------------------------------------------------------------------
def test_regresyon_garch_mle_olceksiz_kosmuyor():
    assert "args=(resid, var0)" not in _QM_SRC, "MLE yine ölçeksiz artıkla koşuyor"
    assert "args=(resid2_scaled, 1.0)" in _QM_SRC
    assert "_GARCH_PERSIST_MAX" in _QM_SRC


def test_regresyon_float_kare_tasma_yolu_geri_gelmemis():
    assert "float(r) ** 2" not in _QM_SRC, "CPython float ** taşmada OverflowError atar"
    assert "(xt - m_i) ** 2" not in _AM_SRC


def test_regresyon_detektorler_pandas_okumaya_donmemis():
    assert ".iloc[" not in _kod(_ST_SRC, "def detect_rsi_divergence", "def macd(series"),         "RSI-divergence yine pandas dilimliyor"
    assert ".iloc[" not in _kod(_ST_SRC, "def detect_mean_reversion",
                                "def detect_rsi_divergence")
    assert ".iloc[" not in _kod(_ST_SRC, "def detect_macd", "def detect_donchian")
    assert ".iloc[" not in _kod(_ST_SRC, "def detect_donchian", "def detect_squeeze")
    assert ".iloc[" not in _kod(_ST_SRC, "def detect_squeeze", "def detect_trend_pullback")


def test_regresyon_extract_features_ham_sutun_dilimlemiyor():
    govde = _kod(_MF_SRC, "def extract_features", "def ensure_feature_columns")
    for yasak in ('df_slice["high"]', 'df_slice["low"]', 'df_slice["open"]',
                  'df_slice["volume"]', "c.iloc[", "shared.rsi14()", "shared.bb20()"):
        assert yasak not in govde, yasak
