"""Lead-lag aday veri seti testleri (edge_lab BACKLOG #3).

Ağırlık merkezi: **leak-free** olduğunun ve **lider seçiminde hindsight bulunmadığının**
kanıtı. Bu iki şey bu ailenin en kolay sessizce bozulacağı yerler (CLAUDE.md: look-ahead
yasak; ACI DERS #1 survivorship/hindsight kenar uydurur).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.edge_lab.build_leadlag_dataset import (
    COST_THRESHOLD,
    FEATURE_COLS,
    SCHEMA_COLS,
    build_dataset,
    build_features_for_sector,
)


# Saatlik ızgara: N gün × 8 bar (14:00–21:00) — gerçek seans şekline yakın.
def _grid(days: int = 14) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([
        pd.Timestamp("2026-01-05") + pd.Timedelta(days=d, hours=h)
        for d in range(days) for h in range(14, 22)
    ])


def _bars(idx: pd.DatetimeIndex, *, vol_level: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.004, len(idx))
    close = 100 * np.exp(np.cumsum(ret))
    volume = vol_level * 1e6 * (1 + rng.normal(0, 0.2, len(idx)))
    return pd.DataFrame({"close": close, "volume": np.abs(volume)}, index=idx)


@pytest.fixture
def frames() -> dict[str, pd.DataFrame]:
    idx = _grid()
    return {
        "BIG": _bars(idx, vol_level=10.0, seed=1),
        "MID": _bars(idx, vol_level=3.0, seed=2),
        "SMALL": _bars(idx, vol_level=1.0, seed=3),
    }


# ---------------------------------------------------------------------------------------
# Şema / temel sözleşme
# ---------------------------------------------------------------------------------------
def test_sema_harness_sozlesmesiyle_birebir(frames):
    out = build_features_for_sector(frames, "test")
    assert list(out.columns) == SCHEMA_COLS
    assert out["label"].isin([0, 1]).all()
    assert out[FEATURE_COLS].isna().sum().sum() == 0, "özelliklerde NaN modele sızmamalı"


def test_iki_uyeden_az_sektor_none(frames):
    assert build_features_for_sector({"BIG": frames["BIG"]}, "tek") is None
    assert build_features_for_sector({}, "bos") is None


def test_yetersiz_gecmis_none():
    idx = _grid(days=1)  # 8 bar < MIN_HISTORY_BARS
    frames = {"A": _bars(idx, vol_level=5.0, seed=1), "B": _bars(idx, vol_level=1.0, seed=2)}
    assert build_features_for_sector(frames, "kisa") is None


# ---------------------------------------------------------------------------------------
# Lider mantığı
# ---------------------------------------------------------------------------------------
def test_lider_kendisi_icin_satir_uretmez(frames):
    out = build_features_for_sector(frames, "test")
    assert not (out["symbol"] == out["leader"]).any(), "bir sembol kendini lead edemez"


def test_lider_en_yuksek_dolar_hacimli_uye(frames):
    """BIG'in dolar-hacmi diğerlerinin ~3–10 katı → lider daima BIG olmalı."""
    out = build_features_for_sector(frames, "test")
    assert set(out["leader"].unique()) == {"BIG"}
    assert set(out["symbol"].unique()) == {"MID", "SMALL"}


def test_lider_zamanla_degisebilir_ve_geriye_bakar():
    """Hacim liderliği el değiştirince lider kolonu da DEĞİŞMELİ — lider sabit/elle değil.

    Ayrıca hindsight yok: liderlik SON 20 barın medyan dolar-hacmine göre, yani devir
    ancak yeni rejimin penceresi dolduktan SONRA görünür (anında değil)."""
    idx = _grid(days=20)
    n = len(idx)
    half = n // 2
    rng = np.random.default_rng(7)

    def with_volume(levels: np.ndarray, seed: int) -> pd.DataFrame:
        ret = rng.normal(0, 0.004, n)
        close = 100 * np.exp(np.cumsum(ret))
        return pd.DataFrame({"close": close, "volume": levels}, index=idx)

    # A önce yüksek hacimli, sonra düşük; B tam tersi.
    a_vol = np.r_[np.full(half, 10e6), np.full(n - half, 1e6)] * (1 + rng.normal(0, 0.05, n))
    b_vol = np.r_[np.full(half, 1e6), np.full(n - half, 10e6)] * (1 + rng.normal(0, 0.05, n))
    frames = {"A": with_volume(np.abs(a_vol), 1), "B": with_volume(np.abs(b_vol), 2)}

    out = build_features_for_sector(frames, "devir")
    leaders = out.sort_values("ts")["leader"]
    assert set(leaders.unique()) == {"A", "B"}, "liderlik el değiştirmeli"

    # Devir ANI, rejim değişiminden SONRA olmalı (geriye-bakan medyan penceresi):
    devir_ts = out.sort_values("ts").loc[out.sort_values("ts")["leader"].eq("B"), "ts"].min()
    assert devir_ts > idx[half], "lider rejim değişmeden ÖNCE devredemez (bu look-ahead olurdu)"


# ---------------------------------------------------------------------------------------
# LEAK-FREE — bu dosyanın asıl işi
# ---------------------------------------------------------------------------------------
def test_gelecek_bar_yalniz_etiketi_etkiler(frames):
    """t+1 barının fiyatını DEĞİŞTİR: hiçbir ÖZELLİK değişmemeli, yalnız label değişebilir.

    Bu, look-ahead'in doğrudan testidir — bir özellik gelecek bardan besleniyorsa burada
    yakalanır."""
    base = build_features_for_sector(frames, "test")

    # Son barı değil, ORTA bir barı bozuyoruz ki hem özellik hem etiket tarafı canlı olsun.
    bozuk = {k: v.copy() for k, v in frames.items()}
    hedef_ts = base["ts"].iloc[len(base) // 2]
    idx = bozuk["SMALL"].index
    sonraki = idx[idx.get_loc(hedef_ts) + 1]
    bozuk["SMALL"].loc[sonraki, "close"] *= 1.05     # yalnız GELECEK bar

    yeni = build_features_for_sector(bozuk, "test")

    ortak = ["ts", "symbol"]
    b = base.set_index(ortak).sort_index()
    y = yeni.set_index(ortak).sort_index()
    ortak_idx = b.index.intersection(y.index)

    # hedef_ts'teki SMALL satırının ÖZELLİKLERİ birebir aynı kalmalı
    anahtar = (hedef_ts, "SMALL")
    assert anahtar in ortak_idx
    for col in FEATURE_COLS:
        assert b.loc[anahtar, col] == pytest.approx(y.loc[anahtar, col]), (
            f"{col} gelecek bardan etkilendi — LOOK-AHEAD"
        )
    # ...ama etiketi değişmiş OLMALI (+%5 maliyet eşiğini kesin aşar)
    assert y.loc[anahtar, "label"] == 1


def test_etiket_maliyet_esigini_kullanir():
    """Getiri eşiğin ALTINDA → 0, ÜSTÜNDE → 1 (sıfır getiri kâr sayılmaz)."""
    idx = _grid(days=14)
    n = len(idx)
    rng = np.random.default_rng(3)
    lider = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n))),
                          "volume": np.abs(10e6 * (1 + rng.normal(0, 0.1, n)))}, index=idx)
    # Takipçi tamamen SABİT fiyat → her bar getirisi 0 → eşiğin altında → tüm etiketler 0
    takipci = pd.DataFrame({"close": np.full(n, 50.0),
                            "volume": np.abs(1e6 * (1 + rng.normal(0, 0.1, n)))}, index=idx)
    out = build_features_for_sector({"L": lider, "F": takipci}, "esik")
    sabit = out[out["symbol"] == "F"]
    assert len(sabit) > 0
    assert (sabit["label"] == 0).all(), f"0 getiri (< {COST_THRESHOLD}) kâr sayılamaz"


def test_ayni_gun_kisiti_gece_boslugunu_disarida_birakir(frames):
    """Gün SONU barı (t+1 ertesi güne düşer) satır üretmemeli — aksi halde etiket overnight
    getirisini içerir ve BACKLOG #1 hipoteziyle karışır."""
    out = build_features_for_sector(frames, "test")
    saatler = pd.DatetimeIndex(out["ts"]).hour
    assert 21 not in set(saatler.tolist()), "günün son barı etikete gece boşluğu taşır"


# ---------------------------------------------------------------------------------------
# Çoklu-sektör sürücüsü
# ---------------------------------------------------------------------------------------
def test_build_dataset_bozuk_sektoru_atlar_koşu_durmaz():
    idx = _grid()
    ok = {"P": _bars(idx, vol_level=9.0, seed=1), "Q": _bars(idx, vol_level=1.0, seed=2)}

    def fake_fetch(sym: str, years: int, interval: str):
        return ok.get(sym)          # "YOK" sembolü için None → o sektör elenir

    df = build_dataset(
        {"iyi": ["P", "Q"], "bozuk": ["YOK", "YOK2"]},
        fetch_fn=fake_fetch,
    )
    assert not df.empty
    assert set(df["sector"].unique()) == {"iyi"}
    assert list(df.columns) == SCHEMA_COLS


def test_build_dataset_sembolu_bir_kez_ceker():
    """Aynı sembol iki sektörde geçse bile ağdan/diskten BİR KEZ çekilmeli (cache)."""
    idx = _grid()
    cagri: list[str] = []
    veri = {"X": _bars(idx, vol_level=9.0, seed=1), "Y": _bars(idx, vol_level=1.0, seed=2)}

    def sayan_fetch(sym: str, years: int, interval: str):
        cagri.append(sym)
        return veri.get(sym)

    build_dataset({"a": ["X", "Y"], "b": ["X", "Y"]}, fetch_fn=sayan_fetch)
    assert sorted(cagri) == ["X", "Y"], f"sembol tekrar çekildi: {cagri}"
