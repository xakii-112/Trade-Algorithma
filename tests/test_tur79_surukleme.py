"""Tur-79 D: bağımsız formüller, örneklem geri kazanımı ve savunmacı sözleşme."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nihai.ai import drift_monitor as d
from nihai.signals.meta_features import FEATURE_NAMES


def test_ozdes_orneklem() -> None:
    x = np.random.default_rng(79).normal(size=1000)
    assert d.psi(x, x)["psi"] == 0.0
    assert d.ks_drift(x, x)["stat"] == 0.0
    assert d.wasserstein_drift(x, x)["w1"] == 0.0
    assert d.fisher_rao_drift(x, x)["fr"] == 0.0
    assert d.psi(x, x) == d.psi(x, x)


@pytest.mark.parametrize("seed", [7, 79, 2026])
def test_normal_kayma_geri_kazanim(seed: int) -> None:
    rng = np.random.default_rng(seed)
    r, c = rng.normal(size=200000), rng.normal(0.5, 1, 200000)
    # Ön ölçüm (3 tohum) azami hatalar: PSI göreli .019161; W1 .007100; FR .006506.
    assert abs(d.psi(r, c, bins=50)["psi"] / 0.25 - 1) < 0.04
    assert d.wasserstein_drift(r, c)["w1"] == pytest.approx(0.5, abs=0.015)
    expected = math.sqrt(2) * math.acosh(1 + 0.5**2 / 4)
    assert d.fisher_rao_drift(r, c)["fr"] == pytest.approx(expected, abs=0.014)


def test_fisher_kapali_form_olcek_ve_kayma() -> None:
    assert d.fisher_rao_normal(0, 1, 0, 2) == pytest.approx(math.sqrt(2) * math.log(2), abs=1e-12)
    assert d.fisher_rao_normal(0, 1, 0.5, 1) == pytest.approx(
        math.sqrt(2) * math.acosh(1 + 0.5**2 / 4), abs=1e-12)
    # Yakın dağılımda arccosh(1+x) yuvarlamasına karşı yerel Fisher metriği.
    assert d.fisher_rao_normal(0, 1, 1e-10, 1) == pytest.approx(1e-10, rel=1e-12, abs=0)
    assert d.fisher_rao_normal(0, 1e200, 0, 2e200) == pytest.approx(math.sqrt(2)*math.log(2), abs=1e-12)


def test_fisher_ucgen_simetri_200_uclu() -> None:
    rng = np.random.default_rng(79)
    triples = np.stack((rng.normal(size=(200, 3)), np.exp(rng.normal(size=(200, 3)))), axis=-1)
    for a, b, c in triples:
        ab = d.fisher_rao_normal(*a, *b)
        assert ab == pytest.approx(d.fisher_rao_normal(*b, *a), abs=1e-12)
        assert d.fisher_rao_normal(*a, *c) <= ab + d.fisher_rao_normal(*b, *c) + 1e-12
        expected = math.sqrt(2) * math.acosh(
            1 + ((a[0]-b[0])**2 / 2 + (a[1]-b[1])**2) / (2*a[1]*b[1]))
        assert ab == pytest.approx(expected, abs=1e-12)


def test_sabit_kenar_simetri_ve_kaba_sayim() -> None:
    a = np.arange(100, dtype=float)
    b = np.arange(100, dtype=float) + 21
    edges = np.array([-np.inf, 25, 50, 75, np.inf])
    original = edges.copy()
    ab, ba = d.psi(a, b, edges=edges), d.psi(b, a, edges=edges)
    assert ab["psi"] == pytest.approx(ba["psi"], abs=1e-12)
    expected = sum((c-r) * math.log(c/r) for r, c in zip([.25]*4, [.04, .25, .25, .46], strict=True))
    assert ab["psi"] == pytest.approx(expected, abs=1e-12)
    assert ab["kategorik"] is False
    np.testing.assert_array_equal(edges, original)


def test_kategorik_elle_psi_ve_scipy_chi2() -> None:
    r = np.repeat([0, 1, 2], [60, 30, 10])
    c = np.repeat([0, 1, 2], [20, 50, 30])
    result = d.psi(r, c)
    expected = (-.4)*math.log(1/3) + .2*math.log(5/3) + .2*math.log(3)
    assert result["psi"] == pytest.approx(expected, abs=1e-12)
    assert result["edges"] is None and result["bins_used"] == 3 and result["kategorik"]
    chi = stats.chi2_contingency([[60, 30, 10], [20, 50, 30]], correction=False)
    test = d.ks_drift(r, c)
    assert test["stat"] == pytest.approx(chi.statistic, abs=1e-12)
    assert test["p"] == pytest.approx(chi.pvalue, abs=1e-12)
    assert test["cramer_v"] == pytest.approx(math.sqrt(chi.statistic/200), abs=1e-12)
    assert test["reliable"] and test["test"] == "chi2"


def test_chi2_seyrek_ve_yuzde20_siniri() -> None:
    sparse = d.ks_drift([0, 0, 1], [0, 1, 2])
    assert sparse["p"] is None and not sparse["reliable"]
    # 10 hücrenin tam ikisi <5: %20 sınırı güvenilir tarafta.
    r = np.repeat(np.arange(5), [2, 20, 20, 20, 20])
    c = np.repeat(np.arange(5), [2, 10, 30, 20, 20])
    assert d.ks_drift(r, c)["reliable"] is True


def test_ks_ve_w1_bagimsiz_referans() -> None:
    r = np.arange(50, dtype=float)
    c = r * 1.25 + 2
    result = d.ks_drift(r, c)
    expected = stats.ks_2samp(r, c)
    assert result == {"stat": float(expected.statistic), "p": float(expected.pvalue), "test": "ks"}
    # Eşit ağırlıklı tek boyutlu taşıma = sıralanmış örnekler arası ortalama mutlak uzaklık.
    w1 = np.mean(np.abs(np.sort(r) - np.sort(c)))
    result = d.wasserstein_drift(r, c)
    assert result["w1"] == pytest.approx(w1, abs=1e-12)
    assert result["w1_std"] == pytest.approx(w1 / np.std(r, ddof=1), abs=1e-12)
    fr = d.fisher_rao_drift(r, c)
    expected_fr = math.sqrt(2) * math.acosh(
        1 + ((r.mean()-c.mean())**2/2 + (r.std(ddof=1)-c.std(ddof=1))**2)
        / (2*r.std(ddof=1)*c.std(ddof=1)))
    assert fr["fr"] == pytest.approx(expected_fr, abs=1e-12)
    assert fr["mu_ref"] == r.mean() and fr["sd_cur"] == c.std(ddof=1)
    assert fr["normal_varsayimi"] is True


def test_kova_uclari_taban_ve_nan_sayimi() -> None:
    r = np.arange(100, dtype=float)
    c = np.array([-1e6, -1e5, 1e5, 1e6])
    result = d.psi(r, c, bins=10)
    assert result["edges"][0] == -np.inf and result["edges"][-1] == np.inf
    assert result["n_bos_kova"] == 8 and result["bins_used"] == 10
    expected = 2*(.5-.1)*math.log(.5/.1) + 8*(1e-4-.1)*math.log(1e-4/.1)
    assert result["psi"] == pytest.approx(expected, abs=1e-12)
    assert sum(result["katki"]) == pytest.approx(expected, abs=1e-12)
    partial = d.psi([0, 1, np.nan, np.nan], [0, 1, np.nan])
    assert partial["eksik_ref"] == .5 and partial["eksik_cur"] == pytest.approx(1/3)
    assert partial["n_ref"] == partial["n_cur"] == 2 and partial["psi"] == 0
    # İki tarafı da boş kova bir kez sayılır; sıfır olmayan ama eps altı oran da sayılır.
    assert d.psi([0, 0], [0, 0], edges=[-np.inf, 1, 2, np.inf])["n_bos_kova"] == 2
    assert d.psi([0]*99+[1], [0]*99+[1], eps=.02)["n_bos_kova"] == 1


def test_referans_kuantil_kenarlari_ve_kategori_secimi() -> None:
    r = np.repeat(np.arange(30, dtype=float), np.arange(1, 31))
    result = d.psi(r, r+100, bins=50)
    expected = np.unique(np.quantile(r, np.linspace(0, 1, 51)))
    expected[0], expected[-1] = -np.inf, np.inf
    assert result["edges"] == expected.tolist()
    assert result["bins_used"] == len(expected)-1
    assert d.psi([0, 1], np.arange(100))["kategorik"] is True
    assert d.ks_drift([0, 1], np.arange(100))["test"] == "chi2"
    assert d.psi([0, 1], [0, 1], kategorik_max=0)["kategorik"] is False


def test_sabit_seri_uydurma_olcek_yok() -> None:
    assert d.psi([3]*20, [3]*20)["psi"] == 0
    assert d.psi([3]*20, [4]*20)["durum"] == "BUYUK"
    assert d.wasserstein_drift([3]*20, [4]*20) == {"w1": 1.0, "w1_std": None}
    assert d.fisher_rao_drift([3]*20, [4]*20) is None
    assert d.ks_drift([3]*20, [3]*20) == {
        "stat": 0.0, "p": None, "cramer_v": None, "test": "chi2", "reliable": False}


def test_rapor_bonferroni_siralama_ve_yetersiz() -> None:
    r = np.linspace(-2, 2, 400)
    ref = pd.DataFrame({"orta": r, "buyuk": r, "stabil": r, "az": np.r_[r[:199], [np.nan]*201],
                        "sabit": np.ones(400), "metin": ["x"]*400})
    cur = pd.DataFrame({"orta": r+.4, "buyuk": r+2, "stabil": r, "az": r, "sabit": np.ones(400),
                        "metin": ["y"]*400})
    report = d.feature_drift_report(ref, cur)
    rows = report["ozellikler"]
    assert report["m_test"] == 3 and report["n_yetersiz"] == 1
    assert set(rows) == {"orta", "buyuk", "stabil", "az", "sabit"}
    assert rows["az"]["yetersiz"] and rows["az"]["neden"]
    assert all(rows["az"].get(key) is None for key in ("psi", "stat", "w1", "fr", "p_bonf"))
    assert rows["sabit"]["p_bonf"] is None and rows["sabit"]["anlamli"] is None
    assert 0 < rows["orta"]["p"] < 1/3  # Mutantın doygun p=0/1 ile saklanmasını engeller.
    for name in ("orta", "buyuk", "stabil"):
        expected = min(1, stats.ks_2samp(ref[name], cur[name]).pvalue * 3)
        assert rows[name]["p_bonf"] == pytest.approx(expected, abs=1e-15)
        assert rows[name]["anlamli"] == (expected < .05)
    ranking = report["siralama"]
    assert "az" not in ranking and ranking[0] == "buyuk"
    assert [rows[name]["psi"] for name in ranking] == sorted(
        [rows[name]["psi"] for name in ranking], reverse=True)
    for key, label in (("n_buyuk", "BUYUK"), ("n_orta", "ORTA"), ("n_stabil", "STABIL")):
        assert report[key] == sum(rows[name]["durum"] == label for name in ranking)
    assert report["alpha"] == .05 and report["min_n"] == 200
    selected = d.feature_drift_report(ref, cur, features=["stabil", "eksik"])
    assert selected["siralama"] == ["stabil"] and selected["n_yetersiz"] == 1


def test_csv_sema_naive_zaman_ve_yari_acik_suzgec(tmp_path: Path) -> None:
    # ornek_baslik.txt şeması; üretim özellik adları TEK kaynaktan alınır.
    extras = ["cand_side", "br_above_sma50", "br_adv_ratio", "br_mcclellan", "br_n",
              "rs_bench_20", "xa_bond_20", "xa_dollar_20", "xa_oil_20"]
    columns = ["logged_at", "symbol", "interval", "bar_ts", *FEATURE_NAMES, *extras]
    frame = pd.DataFrame(np.nan, index=range(5), columns=FEATURE_NAMES + extras)
    frame = frame.assign(logged_at="2026-07-06T19:02:16", symbol=["A", "B", "C", "D", "E"],
                         interval=["1m", "1m", "5m", "1m", "1m"],
                         bar_ts=["2026-07-06 16:00:00", "2026-07-06 16:01:00", "2026-07-06 16:02:00",
                                 "2026-07-06 16:03:00", "2026-07-06 16:04:00"])[columns]
    path = tmp_path / "live.csv"
    frame.to_csv(path, index=False)
    result = d.load_live_features(path, since="2026-07-06 16:01", until="2026-07-06 16:04", interval="1m")
    assert result["symbol"].tolist() == ["B", "D"]
    assert result["bar_ts"].dt.tz is None
    assert result["bar_ts"].dt.hour.tolist() == [16, 16]
    assert result.columns.tolist() == columns and result[FEATURE_NAMES].isna().all().all()
    assert len(d.load_live_features(path)) == 5
    assert d.load_live_features(path, interval="1h").empty
    assert d.load_live_features(tmp_path / "yok.csv") is None
    assert d.load_live_features(path, since="bozuk") is None
    assert d.load_live_features(path, since="2026-07-06T00:00:00Z") is None
    assert d.load_live_features(path, since="2026-08-01", until="2026-07-01") is None
    # tz-aware satır REDDEDİLMEZ, KANONİKLEŞİR: UTC'ye çevrilip tz düşürülür
    # (`prediction_tracker._canon_ts` sözleşmesi). Üretim logu %37 tz-aware'dir;
    # reddetmek logun TAMAMINI reddetmek olurdu (2026-09-16 ölçümü).
    path.write_text("bar_ts,interval\n2026-07-06T16:00:00Z,1m\n", encoding="utf-8")
    kanonik = d.load_live_features(path)
    assert kanonik is not None and kanonik["bar_ts"].dt.tz is None
    assert kanonik["bar_ts"].iloc[0] == pd.Timestamp("2026-07-06 16:00:00")
    # KARIŞIK tz (üretimdeki GERÇEK durum): naive ZATEN UTC sayılır, ET satır UTC'ye
    # çevrilir → aynı an, aynı naive değer.
    path.write_text("bar_ts,interval\n2026-07-06 16:00:00,1m\n"
                    "2026-07-06T12:00:00-04:00,1m\n", encoding="utf-8")
    karisik = d.load_live_features(path)
    assert karisik["bar_ts"].tolist() == [pd.Timestamp("2026-07-06 16:00:00")] * 2
    path.write_text("bar_ts,interval\nbozuk,1m\n", encoding="utf-8")
    assert d.load_live_features(path) is None


def test_haftalik_kayma_ve_gelecek_degismezligi(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = np.linspace(-2, 2, 1000)
    times = np.repeat(pd.to_datetime(["2026-07-06", "2026-07-13", "2026-07-20"]), 1000)
    current = np.tile(ref, 3)
    current[-1000:] += 2
    calls = []
    quantile = np.quantile

    def counted(*args, **kwargs):
        calls.append(1)
        return quantile(*args, **kwargs)

    monkeypatch.setattr(d.np, "quantile", counted)
    result = d.time_sliced_psi(ref, current, times)
    assert len(calls) == 1  # Kenarlar her hafta yeniden fit edilmez.
    assert result["n"] == [1000]*3 and result["psi"][:2] == [0.0, 0.0]
    assert result["psi"][2] > .25
    assert result["donem"] == ["2026-07-12 00:00:00", "2026-07-19 00:00:00", "2026-07-26 00:00:00"]
    changed = current.copy()
    changed[-1000:] = np.arange(1000) * 100
    future = d.time_sliced_psi(ref, changed, times)
    assert future["psi"][:2] == result["psi"][:2]
    prefix = d.time_sliced_psi(ref, current[:2000], times[:2000])
    assert prefix["psi"] == result["psi"][:2]


def test_haftalik_kategoriler_az_gozlem_ve_bos_donem() -> None:
    ref = np.tile([0, 1], 100)
    values = np.r_[ref, [np.nan, 1], ref]
    times = np.repeat(pd.to_datetime(["2026-07-06", "2026-07-13", "2026-07-27"]), [200, 2, 200])
    result = d.time_sliced_psi(ref, values, times)
    assert result["n"] == [200, 1, 0, 200] and result["psi"] == [0.0, None, None, 0.0]
    values[-200:] = np.arange(200)
    assert d.time_sliced_psi(ref, values, times)["psi"][:3] == result["psi"][:3]
    # Sıralanmamış girdi aynı döneme yerleştirilir.
    reverse = d.time_sliced_psi(ref, values[::-1], times[::-1])
    assert reverse == d.time_sliced_psi(ref, values, times)


@pytest.mark.parametrize("bad", [None, [], [1], [np.nan]*3, [1, np.nan], [0, np.inf],
                                  [[1, 2], [3, 4]], "yanlis", {"x": 1}, [1, "x"], [1+2j, 3],
                                  [10**400, 1], [1, -np.inf]])
def test_gecersiz_orneklem_none(bad: object) -> None:
    for function in (d.psi, d.ks_drift, d.wasserstein_drift, d.fisher_rao_drift):
        assert function(bad, [1, 2, 3]) is None
        assert function([1, 2, 3], bad) is None
    assert d.time_sliced_psi(bad, [1, 2, 3], ["2026-01-01"]*3) is None
    assert d.time_sliced_psi([1, 2, 3], bad, ["2026-01-01"]*3) is None
    assert d.feature_drift_report(bad, pd.DataFrame({"x": [1, 2]})) is None


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf"), [], [1], "x", 1j])
def test_fisher_gecersiz_skaler(bad: object) -> None:
    for i in range(4):
        values = [0, 1, 0, 1]
        values[i] = bad
        assert d.fisher_rao_normal(*values) is None
    assert d.fisher_rao_normal(0, 0, 0, 1) is None
    assert d.fisher_rao_normal(0, 1, 0, -1) is None


@pytest.mark.parametrize("kwargs", [{"bins": 0}, {"bins": 1.5}, {"bins": None}, {"eps": 0},
                                     {"eps": 1}, {"eps": np.nan}, {"eps": []}, {"kategorik_max": -1},
                                     {"edges": []}, {"edges": [[0, 1]]}, {"edges": [-np.inf, 1, 1, np.inf]},
                                     {"edges": [-np.inf, np.nan, np.inf]}, {"edges": [0, 1]},
                                     {"edges": [0, 1j]}, {"edges": "x"}])
def test_gecersiz_psi_parametreleri(kwargs: dict) -> None:
    assert d.psi([0, 1, 2], [0, 1, 2], **kwargs) is None


def test_diger_gecersiz_parametreler_ve_sekiller(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [1, 2, 3]})
    for kwargs in ({"min_n": 1}, {"min_n": np.inf}, {"alpha": np.nan}, {"alpha": 1},
                   {"bins": "x"}, {"features": 1}, {"features": "x"}, {"features": [[1]]}):
        assert d.feature_drift_report(frame, frame, **kwargs) is None
    assert d.feature_drift_report(pd.DataFrame(), pd.DataFrame()) is None
    assert d.feature_drift_report(pd.DataFrame([[1, 2]], columns=["x", "x"]), frame) is None
    for times in (None, [], [["2026-01-01"]]*3, ["x"]*3, ["2026-01-01"]*2, [None]*3):
        assert d.time_sliced_psi([1, 2, 3], [1, 2, 3], times) is None
    for kwargs in ({"freq": "bozuk"}, {"freq": None}, {"min_n": 1}, {"bins": 0}):
        assert d.time_sliced_psi([1, 2, 3], [1, 2, 3], ["2026-01-01"]*3, **kwargs) is None
    assert d.load_live_features(None) is None
    assert d.load_live_features(tmp_path) is None
    path = tmp_path / "bad.csv"
    for content in ("", "foo\n1\n", "bar_ts\n\n", "bar_ts,interval\n,,fazla,kolon\n"):
        path.write_text(content, encoding="utf-8")
        assert d.load_live_features(path) is None
