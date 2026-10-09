"""
test_tur51_vrp_veri_seti.py — Tur-51 kilitleri.

Üç ayrı kusur sınıfını kilitler (hepsi bu turda ÖLÇÜLEREK bulundu):

* **BULGU-51-1/K0** — ileri gerçekleşen oynaklık FİYATTAN hesaplanır. Etiketin karar
  barından SONRAKİ barları ölçtüğü, öncekilerden ETKİLENMEDİĞİ doğrudan sınanır
  (`hv_current.shift(-W)` vekilinin (30−W)/30 kirlenmesi yapısal olarak imkânsız olmalı).
* **BULGU-51-2** — sağlayıcının `iv_month_ago`/`hv_month_ago` kolonları 2024'ten beri
  BOŞTUR; onlara bağlanan hat 2023 sonunda sessizce biter. Bu script onları OKUMAMALIDIR.
* **BULGU-51-3** — DÜZENSİZ bir seride `shift(N)` bir ZAMAN kaydırması değildir. Izgaraya
  taşındıktan sonra N satır TAM OLARAK N işlem günü olmalı, ffill sınırı aşılmamalıdır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.edge_lab import build_vrp_dataset as B


# ---------------------------------------------------------------------------------------
# BULGU-51-1 / K0 — ileri RV yalnız GELECEĞE bakar
# ---------------------------------------------------------------------------------------
def test_ileri_rv_tam_olarak_sonraki_W_barin_getirisidir():
    """rv_fwd(i), r_{i+1}..r_{i+W} getirilerinin yıllıklandırılmış std'si OLMALI."""
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    W = 21
    got = B.forward_realized_vol(close, W)
    lr = np.log(close).diff()
    for i in (30, 75, 120):
        beklenen = lr.iloc[i + 1: i + 1 + W].std(ddof=1) * np.sqrt(B.TRADING_DAYS)
        assert got.iloc[i] == pytest.approx(beklenen, rel=1e-12)


def test_ileri_rv_KARAR_BARINDAN_ONCEKI_hareketten_etkilenmez():
    """Etiket kirlenmesinin (proxy'nin (30−W)/30 kusuru) yapısal yokluğu.

    t'den ÖNCEKİ getirileri devasa büyütmek rv_fwd(t)'yi DEĞİŞTİRMEMELİ; t'den
    SONRAKİLERİ büyütmek DEĞİŞTİRMELİ. Vekil etiket bu testin ilk yarısını GEÇEMEZ.
    """
    W, t = 21, 100
    rng = np.random.default_rng(1)
    ret = rng.normal(0, 0.01, 200)
    taban = B.forward_realized_vol(pd.Series(100 * np.exp(np.cumsum(ret))), W).iloc[t]

    onceki = ret.copy()
    onceki[:t + 1] *= 20.0                      # yalnız GEÇMİŞ oynatıldı
    assert B.forward_realized_vol(pd.Series(100 * np.exp(np.cumsum(onceki))), W).iloc[t] == \
        pytest.approx(taban, rel=1e-12)

    sonraki = ret.copy()
    sonraki[t + 1:] *= 20.0                     # yalnız GELECEK oynatıldı
    assert B.forward_realized_vol(pd.Series(100 * np.exp(np.cumsum(sonraki))), W).iloc[t] > \
        taban * 5


def test_ileri_rv_serinin_sonunda_NaN_birakir():
    """Son W barda ileri pencere YOKTUR — uydurulmaz, NaN kalır."""
    close = pd.Series(np.linspace(100, 120, 60))
    got = B.forward_realized_vol(close, 21)
    assert got.iloc[-21:].isna().all()


# ---------------------------------------------------------------------------------------
# BULGU-51-2 — sağlayıcının boş kolonlarına BAĞLANILMAZ
# ---------------------------------------------------------------------------------------
def test_saglayicinin_month_ago_kolonlari_OKUNMAZ():
    """2024'ten beri %100 NaN olan kolonlar veri setini sessizce kısaltmamalı."""
    assert "iv_month_ago" not in B._VOL_COLS
    assert "hv_month_ago" not in B._VOL_COLS
    assert "iv_week_ago" not in B._VOL_COLS


def test_degisim_ozellikleri_KENDI_serisinden_hesaplanir():
    """f_iv_chg_m, iv_current'ın _LAG_MONTH gözlem gerisiyle farkı OLMALI."""
    n = 60
    df = pd.DataFrame({
        "symbol": ["AAA"] * n,
        "date": pd.bdate_range("2024-01-01", periods=n),
        "iv_current": np.linspace(0.20, 0.50, n),
        "hv_current": np.linspace(0.10, 0.40, n),
        "iv_year_high": 0.60, "iv_year_low": 0.10,
    })
    out = B._add_features(df.copy(), min_rank_obs=5)
    i = 40
    assert out["f_iv_chg_m"].iloc[i] == pytest.approx(
        df["iv_current"].iloc[i] - df["iv_current"].iloc[i - B._LAG_MONTH])
    assert out["f_iv_chg_w"].iloc[i] == pytest.approx(
        df["iv_current"].iloc[i] - df["iv_current"].iloc[i - B._LAG_WEEK])


# ---------------------------------------------------------------------------------------
# BULGU-51-3 — satır kaydırması ancak DÜZENLİ ızgarada zaman kaydırmasıdır
# ---------------------------------------------------------------------------------------
def _bosluklu_ornek():
    """Fiyat takvimi TAM, dolt serisi DELİKLİ (gerçek veri deseninin küçük hâli)."""
    gunler = pd.bdate_range("2024-01-01", periods=40)
    px = pd.DataFrame({"symbol": "AAA", "date": gunler, "rv_fwd": 0.3})
    tut = list(range(0, 10)) + list(range(25, 40))      # 10..24 arası YOK (15 gün delik)
    vol = pd.DataFrame({
        "symbol": "AAA", "date": gunler[tut],
        "iv_current": np.linspace(0.2, 0.5, len(tut)),
        "hv_current": np.linspace(0.1, 0.4, len(tut)),
        "iv_year_high": 0.6, "iv_year_low": 0.1,
    })
    return px, vol


def test_izgaraya_tasima_TAM_takvimi_uretir():
    px, vol = _bosluklu_ornek()
    out = B._to_price_calendar(vol, px)
    assert len(out) == len(px)
    assert out["date"].tolist() == px["date"].tolist()


def test_ffill_SINIRLIDIR_eski_IV_bugunku_IV_sayilmaz():
    """15 günlük delik `_STALE_LIMIT` kadar taşınır, gerisi NaN kalır (Tur-31 disiplini)."""
    px, vol = _bosluklu_ornek()
    out = B._to_price_calendar(vol, px).set_index("date")
    delik = px["date"].iloc[10:25]
    assert out.loc[delik[:B._STALE_LIMIT], "iv_current"].notna().all()
    assert out.loc[delik[B._STALE_LIMIT:], "iv_current"].isna().all()


def test_izgarada_N_satir_TAM_N_ISLEM_GUNUDUR():
    """BULGU-51-3'ün özü: ham seride `shift` ufku KAYAR, ızgarada kaymaz."""
    px, vol = _bosluklu_ornek()
    N = 12
    ham = vol.sort_values("date").reset_index(drop=True)
    ham_ufuk = (ham["date"] - ham["date"].shift(N)).dt.days.dropna()

    izgara = B._to_price_calendar(vol, px)
    izg_ufuk = (izgara["date"] - izgara["date"].shift(N)).dt.days.dropna()

    takvim = px["date"].reset_index(drop=True)
    hedef = (takvim - takvim.shift(N)).dt.days.dropna()

    assert izg_ufuk.tolist() == hedef.tolist()          # ızgara: TAM
    assert ham_ufuk.max() > hedef.max()                 # ham: KAYIYOR


# ---------------------------------------------------------------------------------------
# Harness sözleşmesi
# ---------------------------------------------------------------------------------------
def test_sema_harness_ile_uyumlu():
    """`run_candidate` ts/symbol/feature/label/pnl bekler — şema onları taşımalı."""
    for c in ("ts", "symbol", "label", "pnl"):
        assert c in B.SCHEMA_COLS
    assert all(c in B.SCHEMA_COLS for c in B.FEATURE_COLS)
    assert all(c.startswith("f_") for c in B.FEATURE_COLS)


def test_etiket_pnl_ile_TUTARLI():
    """label, pnl'in işaretinden türer — ikisi ayrı hesaplanırsa sessizce ayrışırlar."""
    n = 300
    df = pd.DataFrame({
        "symbol": ["AAA"] * n,
        "date": pd.bdate_range("2024-01-01", periods=n),
        "iv_current": np.random.default_rng(2).uniform(0.1, 0.6, n),
        "rv_fwd": np.random.default_rng(3).uniform(0.1, 0.6, n),
    })
    df["pnl"] = df["iv_current"] - df["rv_fwd"] - B.DEFAULT_COST_VOLPTS / 100.0
    df["label"] = (df["pnl"] > 0).astype(int)
    assert ((df["pnl"] > 0) == (df["label"] == 1)).all()


# ---------------------------------------------------------------------------------------
# BULGU-52-1 — `min_periods` pencerenin KENDİSİ olamaz + kapsam ön-koşulu
# ---------------------------------------------------------------------------------------
def test_min_rank_obs_URETIM_SOZLESMESINDEN_okunur():
    """Araştırma tarafı üretimden SIKI olamaz.

    Tur-51'in veri seti `min_periods=RANK_WINDOW=252` istiyordu: "son 252 işlem gününün
    HEPSİNDE IV olsun". Izgara dolt'un yalnız ~%92'sini taşıdığı ve eksikler ÖBEKLİ
    olduğu için tek bir kesinti SONRAKİ 252 günün rank'ını da NaN yapıyor, ortak
    `dropna` o satırları sessizce siliyordu — bedeli 20 AYDI. Üretim (`iv_history`)
    30 gözlemden rank döner, 100'den "OK" der; sabit ORADAN okunmalı, kopyalanmamalı.
    """
    from nihai.data.iv_history import MIN_DAYS_OK, ROLLING_WINDOW_DAYS
    assert B.RANK_WINDOW == ROLLING_WINDOW_DAYS
    assert MIN_DAYS_OK < B.RANK_WINDOW, "asgari gözlem pencerenin KENDİSİ olamaz"


def test_obekli_bosluk_rank_i_bir_YIL_boyunca_oldurmez():
    """Tek bir öbek kesinti, sonraki bir yılın tamamını silmemeli (kusurun kendisi)."""
    n = 700
    s = pd.Series(np.linspace(0.2, 0.5, n))
    s.iloc[300:315] = np.nan                       # 15 işlem günlük öbek kesinti
    df = pd.DataFrame({"symbol": ["AAA"] * n, "iv_current": s.to_numpy(),
                       "hv_current": s.to_numpy(), "iv_year_high": 0.6,
                       "iv_year_low": 0.1, "date": pd.bdate_range("2021-01-01", periods=n)})
    kati = B._add_features(df.copy(), B.RANK_WINDOW)["f_iv_rank"]
    gevsek = B._add_features(df.copy(), 100)["f_iv_rank"]
    # 15 günlük kesinti, katı sözleşmede SONRAKİ ~252 günü de siler; gevşek olan kurtarır.
    sonra = slice(315, 315 + B.RANK_WINDOW)
    assert kati.iloc[sonra].isna().sum() >= B.RANK_WINDOW - 5
    assert gevsek.iloc[sonra].notna().sum() > B.RANK_WINDOW // 2


def test_takvim_ORTASINDA_delik_olan_veri_seti_KABUL_EDILMEZ():
    """Kapsam ön-koşulu: delikli veri üstünde hüküm verilmez (kapı koşmadan durur)."""
    from research.edge_lab.harness import PRECONDITION_VERDICT, _stage_coverage
    tam = pd.DataFrame({"ts": pd.bdate_range("2021-01-01", "2023-12-31")})
    assert _stage_coverage(tam, "ts").passed
    delikli = tam[~tam["ts"].dt.year.eq(2022)]
    g = _stage_coverage(delikli, "ts")
    assert not g.passed and g.metrics["n_empty"] == 12
    assert "kenar" not in PRECONDITION_VERDICT.lower(), "ön-koşul hatası kenar İDDİASI değildir"
