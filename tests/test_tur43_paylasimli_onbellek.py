"""
test_tur43_paylasimli_onbellek.py — Tur-43: Tur-7 önbelleğinin ULAŞMADIĞI yerler.

ÖLÇÜLEN KUSUR. Tur-7 gösterge hesabını O(n²)→O(n) indirmişti; Tur-43 profili aynı
O(n²)'nin ÜÇ ayrı biçimde hayatta kaldığını gösterdi (ACI DERS: "bir ders MODÜLE
değil DESENE uygulanır"):

  BULGU-43-1 · `meta_features.extract_features` önbelleği HİÇ kullanmıyordu —
    EMA20/50/200 + ATR14 + BB20 + RSI14 + ADX14'ü büyüyen dilim üzerinde sıfırdan
    hesaplıyordu. 4000 barlık backtest süresinin **%49'u** oradaydı; canlı yolda
    aynı beş gösterge sembol başına **üç kez** (pano: dört kez) hesaplanıyordu.
  BULGU-43-2 · `dropna()` yalnız SAYMAK/BOŞ MU diye sormak için çağrılıyordu
    (`volatility_explosion_filter`, `extract_features`, pano `bb_mid`) — her çağrıda
    yeni bir Series+Index tahsisi. `count()` aynı cevabı tahsissiz verir.
  BULGU-43-3 · `_SharedSlice` hesabı değil ama KIRPMAYI tekrarlıyordu: `.iloc[:n]`
    her erişimde yeni Series+DatetimeIndex kurar. `detect_event` bar başına iki
    görünüm kurup ondan yalnız DÖRT sayı okuyordu.

Bu dosya üç şeyi kilitler: (a) önbellekli yol ile önbelleksiz yol BİT-DENK, (b)
hizasız bir görünüm sessizce LOOK-AHEAD sızdıramaz, (c) `rsi`/`adx` vektörleştirmesi
ve `breakout_at` bit-denk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.signals import strategy as S
from nihai.signals.meta_features import FEATURE_NAMES, extract_features


def _df(n: int = 300, *, seed: int = 43, kind: str = "normal") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if kind == "duz":
        c = np.full(n, 100.0)
    elif kind == "nanli":
        c = 100 + np.cumsum(rng.normal(0, 1, n))
        c[rng.choice(n, max(1, n // 12), replace=False)] = np.nan
    else:
        c = 100 + np.cumsum(rng.normal(0, 1, n))
    hi = c if kind == "duz" else c + np.abs(rng.normal(0, 0.5, n))
    lo = c if kind == "duz" else c - np.abs(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {"open": c, "high": hi, "low": lo, "close": c,
         "volume": rng.integers(1e4, 1e6, n).astype(float)},
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )


class _Sig:
    action = S.SignalAction.LONG
    rr_ratio = 2.0
    quality_score = 3.5
    reasons = ["CONFLUENCE_2"]


def _ayni(a: dict, b: dict) -> list[str]:
    return [k for k in FEATURE_NAMES
            if not (np.isnan(a[k]) and np.isnan(b[k])) and a[k] != b[k]]


# ---------------------------------------------------------------------------
# BULGU-43-1 — önbellekli ↔ önbelleksiz özellik vektörü BİT-DENK
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["normal", "duz", "nanli"])
def test_extract_features_onbellekli_yol_bit_denk(kind):
    df = _df(300, kind=kind)
    shared = S._SharedIndicators(df)
    assert _ayni(extract_features(df, _Sig()),
                 extract_features(df, _Sig(), shared=shared)) == []


def test_extract_features_kirpilmis_gorunum_bit_denk():
    """Backtest yolu: tam df üzerinde kurulan önbelleğin `upto(n)` görünümü,
    df.iloc[:n] üzerinde sıfırdan hesaplananla AYNI vektörü vermelidir."""
    df = _df(400)
    tam = S._SharedIndicators(df)
    for n in (120, 250, 399, 400):
        assert _ayni(extract_features(df.iloc[:n], _Sig()),
                     extract_features(df.iloc[:n], _Sig(), shared=tam.upto(n))) == [], f"n={n}"


def test_hizasiz_gorunum_look_ahead_sizdiramaz():
    """SESSİZ LOOK-AHEAD KAPISI: bar sayısı uyuşmayan bir görünüm verilirse
    önbellek YOK SAYILMALI — aksi halde 400. bara ait EMA 200. barın vektörüne
    sızar ve hiçbir hata görünmez."""
    df = _df(400)
    gelecegi_goren = S._SharedIndicators(df)          # nbars=400
    dilim = df.iloc[:200]
    assert _ayni(extract_features(dilim, _Sig()),
                 extract_features(dilim, _Sig(), shared=gelecegi_goren)) == []
    # ve görünüm gerçekten hizasız (test kendi ön koşulunu doğrular)
    assert gelecegi_goren.nbars != len(dilim)


# ---------------------------------------------------------------------------
# BULGU-43-2 — `dropna()` yerine `count()`: verdikt değişmez
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["normal", "duz", "nanli"])
def test_volatility_filtresi_verdikti_degismez(kind):
    df = _df(200, kind=kind)
    a = S.atr(df, 14)
    if a.dropna().empty:
        beklenen = (True, "ATR_NA")
    else:
        cur, avg = a.iloc[-1], a.iloc[-20:].mean()
        beklenen = ((False, f"ATR_EXPLOSION_{cur/avg:.2f}x")
                    if cur > S.CONFIG.strategy.atr_explosion_multiplier * avg
                    else (True, "ATR_NORMAL"))
    assert S.volatility_explosion_filter(df, atr_series=a) == beklenen


def test_volatility_filtresi_tam_nan_seride_ATR_NA():
    """`count()==0` yolu: hepsi NaN olan seri hâlâ ATR_NA vermeli (eski `dropna().empty`)."""
    df = _df(100)
    assert S.volatility_explosion_filter(df, atr_series=pd.Series([np.nan] * 100)) == (True, "ATR_NA")


def test_volatility_filtresi_son_bar_nan_ama_seri_dolu():
    """Son bar NaN + öncesi geçerli → ATR_NA DEĞİL (kısa devre yanlış dallanmamalı)."""
    df = _df(100)
    a = S.atr(df, 14).copy()
    a.iloc[-1] = np.nan
    assert S.volatility_explosion_filter(df, atr_series=a)[1] != "ATR_NA"


# ---------------------------------------------------------------------------
# BULGU-43-3 — kırpma yerine skaler okuma; detect_event verdikti değişmez
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["normal", "duz", "nanli"])
def test_detect_event_uc_yolda_ayni_verdikt(kind):
    """shared=None · tam önbellek · kırpılmış görünüm — üçü de aynı sonucu vermeli."""
    df = _df(300, kind=kind)
    tam = S._SharedIndicators(df)
    a = S.detect_event(df)
    assert a == S.detect_event(df, shared=tam)
    assert a == S.detect_event(df, shared=tam.upto(len(df)))


def test_breakout_at_karar_barini_disarida_birakir():
    """`breakout_at` = önceki 20 barın uç değeri; KARAR barı pencereye GİRMEZ
    (girseydi kırılım kendi barıyla kıyaslanır ve asla tetiklenmezdi)."""
    df = _df(120)
    tam = S._SharedIndicators(df)
    for n in (60, 90, 120):
        hh, ll = tam.upto(n).breakout_at(-1, 20)
        dilim = df.iloc[:n]
        assert hh == pytest.approx(float(dilim["high"].iloc[-21:-1].max()))
        assert ll == pytest.approx(float(dilim["low"].iloc[-21:-1].min()))
        # karar barı hariç: son barın high'ı pencerenin üstündeyse hh ondan KÜÇÜK kalır
        assert hh != dilim["high"].iloc[-1] or dilim["high"].iloc[-1] in dilim["high"].iloc[-21:-1].values


def test_kirpma_ayni_bar_icinde_tek_kez_kurulur():
    """`_SharedSlice` aynı anahtarı ikinci kez isteyene AYNI nesneyi vermeli —
    aksi halde bar başına O(n) kırpma sayısı çağıran sayısıyla çarpılır."""
    view = S._SharedIndicators(_df(200)).upto(150)
    assert view.ema(50) is view.ema(50)
    assert view.rsi14() is view.rsi14()
    assert view.atr14() is view.atr14()


def test_nbars_gorunumun_kapsamini_bildirir():
    tam = S._SharedIndicators(_df(200))
    assert tam.nbars == 200
    assert tam.upto(77).nbars == 77


# ---------------------------------------------------------------------------
# rsi/adx vektörleştirmesi — bit-denklik (pandas gövdesi referans)
# ---------------------------------------------------------------------------
def _rsi_pandas(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0.0).fillna(0.0)
    dn = (-delta.clip(upper=0.0)).fillna(0.0)
    roll_up, roll_dn = S._rma(up, period), S._rma(dn, period)
    rs = roll_up / roll_dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    duz = (roll_dn == 0) & (roll_up == 0)
    return out.where(roll_dn != 0, 100.0).mask(duz, 50.0).where(roll_up.notna(), np.nan)


def _adx_pandas(df, period=14):
    h, l = df["high"], df["low"]
    up_move = h.diff().fillna(0.0)
    dn_move = (-l.diff()).fillna(0.0)
    p_dm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=df.index)
    m_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=df.index)
    atr_v = S._rma(S._true_range(df), period)
    p_di = 100 * S._rma(p_dm, period) / atr_v
    m_di = 100 * S._rma(m_dm, period) / atr_v
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di).replace(0, np.nan)
    return S._rma(dx, period)


def _bit_denk(a, b) -> bool:
    av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.isnan(av), np.isnan(bv)
    return bool(np.array_equal(na, nb) and (av[~na] == bv[~nb]).all())


@pytest.mark.parametrize("kind", ["normal", "duz", "nanli"])
@pytest.mark.parametrize("period", [7, 14, 21])
def test_rsi_adx_vektorlestirme_bit_denk(kind, period):
    df = _df(250, kind=kind)
    assert _bit_denk(_rsi_pandas(df["close"], period), S.rsi(df["close"], period))
    assert _bit_denk(_adx_pandas(df, period), S.adx(df, period))


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_rsi_adx_uc_uzunluklar(n):
    """Boş/tek/iki barlık seri: vektörleştirme IndexError doğurmamalı."""
    df = _df(5).iloc[:n]
    assert _bit_denk(_rsi_pandas(df["close"]), S.rsi(df["close"]))
    assert _bit_denk(_adx_pandas(df), S.adx(df))


def test_rsi_seri_adini_korur():
    """Pano `_series_payload` seri adına bakabilir — vektörleştirme onu düşürmemeli."""
    df = _df(80)
    assert S.rsi(df["close"]).name == df["close"].name


def test_rsi_duz_pencerede_notr_kalir():
    """Tur-28 dersi vektörleştirmeden SONRA da geçerli: 0/0 → 100 değil 50."""
    duz = pd.Series([7.0] * 60)
    assert S.rsi(duz).iloc[-1] == pytest.approx(50.0)
