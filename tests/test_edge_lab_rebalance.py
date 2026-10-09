"""research.edge_lab.build_rebalance_dataset testleri (edge_lab BACKLOG #4).

AĞIRLIK MERKEZİ = **sızıntı kanıtı**. Bu turun tüm güvenilirliği iki teste dayanıyor:

  * `test_gelecek_barlar_bozulunca_ozellikler_degismez` (K3) — gelecek barlar BOZULUNCA
    hiçbir özellik değişmemeli, YALNIZ label/pnl değişmeli.
  * `test_gelecekteki_endeks_olayi_onceki_barlarin_ozelliklerini_degistirmez` (K1) —
    ileri-bakan olay özelliği YOK. Wikipedia tablosu yürürlük tarihi verir, ANONS tarihini
    değil; "kaç gün SONRA değişiklik olacak" özelliği sinsi look-ahead olurdu.

Tüm veri BU dosyada sentetik üretilir (gerçek piyasa verisi/ağ YOK — kök conftest'in ağ
guard'ı zaten dışarı çıkışı engeller).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from research.edge_lab import build_rebalance_dataset as brd
from research.edge_lab.build_rebalance_dataset import (
    EVENT_CAP_DAYS,
    FEATURE_COLS,
    SCHEMA_COLS,
    IndexTimeline,
    build_dataset,
    build_symbol_rows,
    build_timeline,
    load_timeline,
)

BASE_DATE = pd.Timestamp("2021-01-01")


# ---------------------------------------------------------------------------
# Sentetik veri üreticileri
# ---------------------------------------------------------------------------
def _bars(n: int = 200, seed: int = 0, start: str = "2021-01-04") -> pd.DataFrame:
    """Deterministik günlük OHLCV (yalnız iş günleri — gerçek bar ızgarası gibi)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, n)))
    volume = rng.uniform(1e6, 5e6, n)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": volume}, index=idx)


def _timeline(base: set[str] | None = None,
              adds: dict[str, list[str]] | None = None,
              rems: dict[str, list[str]] | None = None) -> IndexTimeline:
    return IndexTimeline(
        base_date=BASE_DATE,
        base_members=frozenset(base if base is not None else {"AAA"}),
        add_dates={k: [pd.Timestamp(d) for d in v] for k, v in (adds or {}).items()},
        remove_dates={k: [pd.Timestamp(d) for d in v] for k, v in (rems or {}).items()},
    )


# ---------------------------------------------------------------------------
# K0 — düzeltilmiş veri env kilidi (import anında kurulur)
# ---------------------------------------------------------------------------
def test_modul_import_edilince_duzeltme_kipi_all_kilitlenir():
    """`raw` dosyalara düşmek bölünme günlerinde sahte %50 getiri üretir (BULGU-19) —
    ölçüm baştan çöp olurdu."""
    assert os.environ["ALPACA_HIST_ADJUSTMENT"] == "all"
    assert int(os.environ["ALPACA_HIST_CACHE_MAX_AGE_DAYS"]) > 30


def test_cache_yolu_all_dosyasini_gosterir():
    """Env kilidinin GERÇEK etkisi: okunacak cache dosyası `*_1d_all.csv` olmalı."""
    from nihai.data.alpaca_history import _cache_path
    assert _cache_path("AAPL", "1d").name == "AAPL_1d_all.csv"


def test_norm_ticker_pit_universe_ile_paritetli():
    from nihai.data.pit_universe import _norm
    for t in ("brk.b", " BF.B ", "aapl", "BRK-B", "nan"):
        assert brd._norm_ticker(t) == _norm(t)


# ---------------------------------------------------------------------------
# K3 — SIZINTI KANITI (bu turun taşıyıcı testi)
# ---------------------------------------------------------------------------
def test_gelecek_barlar_bozulunca_ozellikler_degismez():
    """Karar barından SONRAKİ barlar bozulduğunda hiçbir ÖZELLİK değişmemeli; yalnız
    label/pnl (gelecekten gelen tek şeyler) değişmeli."""
    bars = _bars(n=220, seed=1)
    tl = _timeline()
    temiz = build_symbol_rows("AAA", bars, tl)
    assert temiz is not None

    kesim = bars.index[140]
    bozuk_bars = bars.copy()
    ileri = bozuk_bars.index > kesim
    bozuk_bars.loc[ileri, "close"] *= 2.5      # gelecek fiyatları BOZ
    bozuk_bars.loc[ileri, "volume"] *= 7.0     # gelecek hacimlerini de BOZ
    bozuk = build_symbol_rows("AAA", bozuk_bars, tl)
    assert bozuk is not None

    m = temiz.merge(bozuk, on=["ts", "symbol"], suffixes=("_a", "_b"))
    onceki = m[m["ts"] <= kesim.normalize()]
    assert len(onceki) > 50, "karşılaştırılacak yeterli 'öncesi' satır olmalı"
    for c in FEATURE_COLS:
        assert np.allclose(onceki[f"{c}_a"], onceki[f"{c}_b"], equal_nan=True), \
            f"{c} gelecek barlardan etkilendi = LOOK-AHEAD"
    # Bozulma etikete GERÇEKTEN ulaştı mı (test kendi kendini kandırmasın)?
    assert not np.allclose(onceki["pnl_a"], onceki["pnl_b"]), \
        "gelecek bozulması label/pnl'e hiç ulaşmadıysa test bir şey kanıtlamıyor"


def test_etiket_ve_pnl_tutarli_ve_gercek_getiri():
    """K4: `pnl` sözde değil GERÇEK (yüzde) getiridir ve `label == 1` ⟺ `pnl > 0`."""
    rows = build_symbol_rows("AAA", _bars(n=200, seed=2), _timeline())
    assert rows is not None
    assert set(rows.columns) == set(SCHEMA_COLS)
    assert ((rows["label"] == 1) == (rows["pnl"] > 0)).all()
    assert rows["pnl"].abs().max() > 0.1          # yüzde ölçeği (±1 sözde-pnl DEĞİL)
    assert rows["pnl"].nunique() > len(rows) // 2  # sürekli değişken


def test_pnl_ufuk_getirisiyle_elle_dogrulanir():
    """PnL formülünün bağımsız (elle) yeniden hesabı — 5-bar ileri getiri, maliyet düşülmüş."""
    bars = _bars(n=120, seed=3)
    rows = build_symbol_rows("AAA", bars, _timeline(), horizon=5)
    assert rows is not None
    close = bars["close"]
    ts0 = rows["ts"].iloc[10]
    i = list(pd.DatetimeIndex(bars.index).normalize()).index(ts0)
    beklenen = 100.0 * ((close.iloc[i + 5] / close.iloc[i] - 1.0) - brd.COST_THRESHOLD)
    assert rows["pnl"].iloc[10] == pytest.approx(beklenen, rel=1e-9)


# ---------------------------------------------------------------------------
# K1 — İLERİ-bakan olay özelliği YOK
# ---------------------------------------------------------------------------
def test_gelecekteki_endeks_olayi_onceki_barlarin_ozelliklerini_degistirmez():
    """Aynı fiyat verisi, tek farkı GELECEKTE bir çıkarma olayı olan iki zaman çizgisi:
    olaydan ÖNCEKİ satırların özellikleri BİREBİR aynı olmalı. Aksi hâlde özellik seti
    henüz kamuya açık olmayan (anons tarihi elimizde yok) bilgiyi taşıyordu demektir."""
    bars = _bars(n=240, seed=4)
    olay = pd.Timestamp("2021-09-01")
    olaysiz = _timeline()
    olayli = _timeline(rems={"AAA": ["2021-09-01"]})

    a = build_symbol_rows("AAA", bars, olaysiz)
    b = build_symbol_rows("AAA", bars, olayli)
    assert a is not None and b is not None
    m = a.merge(b, on=["ts", "symbol"], suffixes=("_a", "_b"))

    onceki = m[m["ts"] < olay]
    assert len(onceki) > 50
    for c in FEATURE_COLS:
        assert np.allclose(onceki[f"{c}_a"], onceki[f"{c}_b"]), \
            f"{c} GELECEKTEKİ endeks olayından etkilendi = ileri-bakan olay özelliği (K1 ihlali)"

    sonraki = m[m["ts"] >= olay]
    assert len(sonraki) > 0
    assert not np.allclose(sonraki["endeks_uyesi_a"], sonraki["endeks_uyesi_b"])
    assert not np.allclose(sonraki["gun_since_cikarildi_a"], sonraki["gun_since_cikarildi_b"])


def test_olay_mesafesi_geriye_bakar_ve_ust_sinirlanir():
    d = pd.DatetimeIndex(["2021-05-31", "2021-06-01", "2021-06-11", "2021-09-30"])
    tl = _timeline(adds={"AAA": ["2021-06-01"]})
    gun = tl.days_since("AAA", d, kind="eklendi")
    assert gun[0] == EVENT_CAP_DAYS      # olaydan ÖNCE → sınır değeri (mesafe NEGATİF olamaz)
    assert gun[1] == 0.0                 # yürürlük gününde 0 (o gün kamuya açık)
    assert gun[2] == 10.0
    assert gun[3] == EVENT_CAP_DAYS      # 121 gün sonra → üst sınıra oturur


def test_olay_yoksa_mesafe_sinir_degerinde():
    d = pd.bdate_range("2021-01-04", periods=5)
    tl = _timeline()
    assert (tl.days_since("AAA", d, kind="eklendi") == EVENT_CAP_DAYS).all()
    assert (tl.days_since("AAA", d, kind="cikarildi") == EVENT_CAP_DAYS).all()


# ---------------------------------------------------------------------------
# K2 / K2b — PIT üyelik ve PIT EVREN
# ---------------------------------------------------------------------------
def test_uyelik_pit_cikarmadan_sonra_sifirlanir():
    bars = _bars(n=240, seed=5)
    cikarma = pd.Timestamp("2021-09-01")
    rows = build_symbol_rows("AAA", bars, _timeline(rems={"AAA": ["2021-09-01"]}))
    assert rows is not None
    assert (rows.loc[rows["ts"] < cikarma, "endeks_uyesi"] == 1.0).all()
    assert (rows.loc[rows["ts"] >= cikarma, "endeks_uyesi"] == 0.0).all()
    # Çıkarılan isim panelde KALIR (survivorship düzeltmesinin ta kendisi)
    assert (rows["ts"] >= cikarma).sum() > 0


def test_endekse_girmeden_onceki_barlar_panele_alinmaz():
    """K2b: sembol panele ancak GİRDİĞİ tarihten itibaren alınır. "Bu hisse ileride
    endekse alınacak" bilgisi karar anında YOKTUR — giriş öncesi barları eklemek evren
    seçimine look-ahead sokar."""
    bars = _bars(n=240, seed=6)
    giris = pd.Timestamp("2021-06-01")
    tl = _timeline(base={"ZZZ"}, adds={"AAA": ["2021-06-01"]})
    rows = build_symbol_rows("AAA", bars, tl)
    assert rows is not None
    assert rows["ts"].min() >= giris
    assert (rows["endeks_uyesi"] == 1.0).all()
    assert tl.panel_start("AAA") == giris


def test_hic_uye_olmamis_sembol_satir_uretmez():
    tl = _timeline(base={"ZZZ"})
    assert tl.panel_start("AAA") is None
    assert build_symbol_rows("AAA", _bars(), tl) is None


def test_ileri_sarma_sp500_as_of_ile_ayni_uyelik(monkeypatch):
    """K2 eşdeğerlik kilidi: `sp500_as_of(start)` + değişikliklerin İLERİ sarılması,
    her tarih için `sp500_as_of(o_tarih)` ile AYNI üyeliği vermeli. (Aksi hâlde
    performans için kurulan önbellek PIT semantiğini bozmuş olurdu.)"""
    from nihai.data import pit_universe

    guncel = ["AAA", "BBB", "CCC"]
    degisiklikler = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-01", "2022-06-15", "2021-09-10"]),
        "added": ["CCC", "BBB", "AAA"],
        "removed": ["DDD", "EEE", "FFF"],
    })
    monkeypatch.setattr(pit_universe, "_current_and_changes",
                        lambda: (list(guncel), degisiklikler.copy()))

    tl = load_timeline("2021-01-01", "2026-07-31",
                       as_of_fn=pit_universe.sp500_as_of,
                       changes_fn=pit_universe._current_and_changes)
    assert set(tl.base_members) == {"DDD", "EEE", "FFF"}

    for gun in ("2021-01-01", "2021-09-09", "2021-09-10", "2022-06-14",
                "2022-06-15", "2023-01-01", "2024-03-01", "2026-07-01"):
        d = pd.DatetimeIndex([pd.Timestamp(gun)])
        beklenen = set(pit_universe.sp500_as_of(gun))
        olculen = {s for s in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
                   if tl.membership(s, d)[0] > 0.5}
        assert olculen == beklenen, f"{gun}: ileri-sarma ≠ sp500_as_of"


def test_evren_bugun_endekste_olmayanlari_icerir():
    """Survivorship-temizlik kanıtı: evren, pencerede ÇIKARILMIŞ isimleri de taşımalı."""
    degisiklikler = pd.DataFrame({
        "date": pd.to_datetime(["2022-06-15"]), "added": ["YENI"], "removed": ["ESKI"]})
    tl = build_timeline(degisiklikler, ["ESKI", "KALICI"], BASE_DATE,
                        pd.Timestamp("2026-07-31"))
    assert set(tl.universe()) == {"ESKI", "KALICI", "YENI"}
    assert tl.panel_start("ESKI") == BASE_DATE       # çıkarılan isim panelde
    assert tl.panel_start("YENI") == pd.Timestamp("2022-06-15")


def test_build_timeline_temel_tarihe_kadarki_degisiklikleri_tekrar_uygulamaz():
    """`base_members` zaten `sp500_as_of(base_date)`'ten geldi; base_date'e kadarki
    değişiklikleri yeniden uygulamak üyeliği bozardı."""
    degisiklikler = pd.DataFrame({
        "date": pd.to_datetime(["2020-05-05", "2021-01-01", "2022-01-01"]),
        "added": ["ESKI1", "ESKI2", "YENI"], "removed": [None, None, "ESKI1"]})
    tl = build_timeline(degisiklikler, ["ESKI1", "ESKI2"], BASE_DATE, pd.Timestamp("2026-01-01"))
    assert set(tl.add_dates) == {"YENI"}             # 2020/2021-01-01 olayları alınmadı
    assert set(tl.remove_dates) == {"ESKI1"}


def test_bozuk_ticker_satirlari_ayiklanir():
    degisiklikler = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"]),
        "added": ["TOOLONGX", "nan", "iyi"], "removed": [None, None, None]})
    tl = build_timeline(degisiklikler, ["AAA"], BASE_DATE, pd.Timestamp("2026-01-01"))
    assert set(tl.add_dates) == {"IYI"}


# ---------------------------------------------------------------------------
# Takvim özellikleri (K1 muafiyeti) + ızgara bütünlüğü
# ---------------------------------------------------------------------------
def test_takvim_ozellikleri_bagimsiz_hesapla_ayni():
    rows = build_symbol_rows("AAA", _bars(n=260, seed=7), _timeline())
    assert rows is not None
    ts = pd.DatetimeIndex(rows["ts"])
    ay_sonu = ts.to_period("M").to_timestamp(how="end").normalize()
    ceyrek_sonu = ts.to_period("Q").to_timestamp(how="end").normalize()
    assert np.array_equal(rows["ay_sonuna_gun"].to_numpy(),
                          (ay_sonu - ts).days.to_numpy(dtype=float))
    assert np.array_equal(rows["ceyrek_sonuna_gun"].to_numpy(),
                          (ceyrek_sonu - ts).days.to_numpy(dtype=float))
    assert rows["ay_sonuna_gun"].min() >= 0
    assert rows["ceyrek_sonuna_gun"].max() <= 92


def test_takvim_ozelligi_ay_sonu_barinda_sifir():
    bars = _bars(n=200, seed=8)
    rows = build_symbol_rows("AAA", bars, _timeline())
    assert rows is not None
    ay_sonlari = rows[rows["ay_sonuna_gun"] == 0.0]
    # Ay sonu HAFTA SONUNA denk geldiğinde 0'lı bar olmayabilir; en az bir tane bulunmalı
    assert len(ay_sonlari) >= 1
    for t in pd.DatetimeIndex(ay_sonlari["ts"]):
        assert t == t.to_period("M").to_timestamp(how="end").normalize()


def test_uretilen_barlar_girdi_izgarasinin_alt_kumesi_hafta_sonu_yok():
    """Builder bar UYDURMAZ: çıktı tarihleri girdi bar ızgarasının alt kümesidir ve
    hafta sonu/tatil barı içermez (girdi zaten iş günü ızgarasında)."""
    bars = _bars(n=200, seed=9)
    rows = build_symbol_rows("AAA", bars, _timeline())
    assert rows is not None
    girdi = set(pd.DatetimeIndex(bars.index).normalize())
    assert set(pd.DatetimeIndex(rows["ts"])).issubset(girdi)
    assert (pd.DatetimeIndex(rows["ts"]).dayofweek < 5).all()


def test_isinma_ve_ufuk_kuyrugu_dusurulur():
    """İlk 20 bar (yuvarlanan pencere ısınması) ve son `horizon` bar (etiket yok) düşer."""
    n, h = 120, 5
    bars = _bars(n=n, seed=10)
    stats: dict[str, int] = {}
    rows = build_symbol_rows("AAA", bars, _timeline(), horizon=h, stats=stats)
    assert rows is not None
    assert len(rows) == n - brd.VOL_WINDOW - h
    assert stats["dusen_isinma"] == brd.VOL_WINDOW
    assert stats["dusen_ufuk_kuyrugu"] == h
    assert rows["ts"].min() == pd.Timestamp(bars.index[brd.VOL_WINDOW]).normalize()


# ---------------------------------------------------------------------------
# Savunmacı davranış — eksik/bozuk veri ÇÖKME üretmez
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bozuk", ["bos", "none", "kisa", "hacimsiz"])
def test_yetersiz_veya_bozuk_veri_none_doner_cokmez(bozuk):
    tl = _timeline()
    if bozuk == "bos":
        bars: pd.DataFrame | None = pd.DataFrame()
    elif bozuk == "none":
        bars = None
    elif bozuk == "kisa":
        bars = _bars(n=brd.MIN_HISTORY_BARS + brd.DEFAULT_HORIZON - 1)
    else:
        bars = _bars(n=100).drop(columns=["volume"])
    assert build_symbol_rows("AAA", bars, tl) is None  # type: ignore[arg-type]


def test_sabit_hacim_sifir_std_cokmez_ve_satir_uretmez():
    """Hacim std'si 0 → z-skoru payda 0; NaN'a çevrilip satır düşürülmeli (bölme hatası yok)."""
    bars = _bars(n=120, seed=11)
    bars["volume"] = 1_000_000.0
    bars["close"] = 100.0            # dolar hacim sabit → std 0
    rows = build_symbol_rows("AAA", bars, _timeline())
    assert rows is None


def test_tekrarli_bar_indeksi_ayiklanir():
    bars = _bars(n=120, seed=12)
    bars = pd.concat([bars, bars.iloc[[50]]]).sort_index()
    rows = build_symbol_rows("AAA", bars, _timeline())
    assert rows is not None
    assert rows["ts"].is_unique


# ---------------------------------------------------------------------------
# build_dataset — uçtan uca (ağ YOK: timeline + fetch enjekte)
# ---------------------------------------------------------------------------
def _fetch_stub(veri: dict[str, pd.DataFrame | None]):
    def _f(sym: str, start: str, end: str) -> pd.DataFrame | None:
        return veri.get(sym)
    return _f


def test_build_dataset_uctan_uca_sayaclarla():
    tl = _timeline(base={"AAA", "BBB"}, adds={"CCC": ["2021-06-01"]},
                   rems={"BBB": ["2021-09-01"]})
    veri = {"AAA": _bars(n=240, seed=20), "BBB": _bars(n=240, seed=21),
            "CCC": _bars(n=240, seed=22), }
    df, stats = build_dataset(timeline=tl, fetch_fn=_fetch_stub(veri))
    assert list(df.columns) == SCHEMA_COLS
    assert stats["evren_sembol"] == 3
    assert stats["uretilen_sembol"] == 3
    assert stats["satir"] == len(df) > 0
    assert stats["ekleme_olayi"] == 1 and stats["cikarma_olayi"] == 1
    assert stats["uye_satir"] + stats["uye_disi_satir"] == len(df)
    assert stats["uye_disi_satir"] > 0            # BBB çıkarma sonrası satırları
    assert stats["dusen_panel_oncesi"] > 0        # CCC'nin giriş öncesi barları
    # sembol-içi kronolojik sıra (harness'ın exec_lag kaydırması buna güvenir)
    for _, g in df.groupby("symbol"):
        assert g["ts"].is_monotonic_increasing


def test_build_dataset_veri_alinamayan_sembolu_sayar_ve_devam_eder():
    tl = _timeline(base={"AAA", "YOK"})
    veri: dict[str, pd.DataFrame | None] = {"AAA": _bars(n=200, seed=23), "YOK": None}
    df, stats = build_dataset(timeline=tl, fetch_fn=_fetch_stub(veri))
    assert stats["veri_alinamadi"] == 1
    assert stats["uretilen_sembol"] == 1
    assert set(df["symbol"]) == {"AAA"}


def test_build_dataset_hicbir_veri_yoksa_bos_sema_doner():
    tl = _timeline(base={"AAA"})
    df, stats = build_dataset(timeline=tl, fetch_fn=_fetch_stub({}))
    assert df.empty and list(df.columns) == SCHEMA_COLS
    assert stats["satir"] == 0 and stats["uretilen_sembol"] == 0


def test_build_dataset_deterministik_ayni_girdi_ayni_csv():
    """Determinizm: aynı girdi → BAYT-BAYT aynı CSV (yeniden üretilebilirlik kilidi)."""
    tl = _timeline(base={"AAA", "BBB"}, rems={"BBB": ["2021-09-01"]})
    veri = {"AAA": _bars(n=200, seed=30), "BBB": _bars(n=200, seed=31)}
    a, _ = build_dataset(timeline=tl, fetch_fn=_fetch_stub(veri))
    b, _ = build_dataset(timeline=tl, fetch_fn=_fetch_stub(veri))
    assert a.to_csv(index=False) == b.to_csv(index=False)


def test_max_symbols_deterministik_ve_alfabeye_yayilir():
    items = [f"S{i:03d}" for i in range(100)]
    a = brd._even_slice(items, 5)
    assert a == brd._even_slice(items, 5)          # deterministik
    assert len(a) == 5
    assert a[0] == items[0] and a[-1] == items[-1]  # uçlar dahil = yayılmış
    assert brd._even_slice(items, None) == items
    assert brd._even_slice(items, 500) == items


def test_max_symbols_evreni_kisitlar():
    tl = _timeline(base={"AAA", "BBB", "CCC"})
    veri = {s: _bars(n=200, seed=40 + i) for i, s in enumerate(("AAA", "BBB", "CCC"))}
    df, stats = build_dataset(timeline=tl, fetch_fn=_fetch_stub(veri), max_symbols=2)
    assert stats["evren_sembol"] == 2
    assert df["symbol"].nunique() == 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_csv_yazar(tmp_path, monkeypatch):
    tl = _timeline(base={"AAA", "BBB"})
    veri = {"AAA": _bars(n=200, seed=50), "BBB": _bars(n=200, seed=51)}
    monkeypatch.setattr(brd, "load_timeline", lambda *a, **k: tl)
    monkeypatch.setattr(brd, "fetch_bars", _fetch_stub(veri))
    out = tmp_path / "cikti.csv"
    assert brd.main(["--out", str(out)]) == 0
    okunan = pd.read_csv(out)
    assert list(okunan.columns) == SCHEMA_COLS
    assert len(okunan) > 0
