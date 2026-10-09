"""Tur-86 kilitleri — bir dış (Gemini/agy) benchmark sohbetinin önerilerini ölçerek uygulamak.

Dört soru:
  * λ süpürmesi DÜRÜST mü? — CLI exec-lag kapısını sessizce kapatmıyor (BULGU-86-1) ve boş
    kümeden "ABSTAIN doğru" hükmü kurmuyor (BULGU-86-2); ölçüm koşulu selfcheck'e taşınıyor.
  * PIT evren KURULABİLİYOR mu? — Wikipedia değişiklik tablosu taşındı (BULGU-86-3); tablo
    konumla değil imzayla aranıyor, bulunamazsa survivorship'e SESSİZCE düşülmüyor.
  * Kapasite süpürmesi GPU'da koşabiliyor mu? — `--device` reçeteye gidiyor, istenen ↔
    gerçekleşen aygıt ayrı alanlarda.
  * Kalıntı momentumu ailesi (BACKLOG #11) ön-kayıtlı kurala sadık mı? — skor formülü, son ay
    hariç oluşum, PIT üyelik süzgeci (BULGU-86-4), ay boşluğu, look-ahead yokluğu.
"""
from __future__ import annotations

import ast
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _kaynak(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# ─────────────────────────── λ süpürmesi (BULGU-86-1/2) ───────────────────────────

def _sdb():
    import research.shadow_decision_backtest as sdb
    return sdb


def test_cli_exec_lag_varsayilani_fonksiyona_birakiliyor(monkeypatch, capsys):
    sdb = _sdb()
    alinan: list[tuple[str, dict]] = []
    monkeypatch.setattr(sdb, "lambda_sweep", lambda **kw: alinan.append(("sweep", kw)) or {})
    monkeypatch.setattr(sdb, "run", lambda **kw: alinan.append(("run", kw)) or {})
    monkeypatch.setattr(sdb, "_write_kpi", lambda *a, **k: None)

    sdb.main(["--lambda-sweep"])
    sdb.main(["--lambda-sweep", "--exec-lag", "0"])
    sdb.main([])
    capsys.readouterr()

    assert "exec_lag" not in alinan[0][1]            # fonksiyonun KENDİ varsayılanı geçerli
    assert alinan[1][1]["exec_lag"] == 0              # açık değer hâlâ geçiyor
    assert alinan[2][0] == "run" and "exec_lag" not in alinan[2][1]


def test_varsayilanlar_durustluk_kapisiyla_hizali():
    import research.shadow_decision_backtest as sdb
    # Süpürme dürüstlük kapısı #1 ile (sinyalden 1 bar SONRA giriş) ölçülür; `run` eski 0'da.
    assert inspect.signature(sdb.lambda_sweep).parameters["exec_lag"].default == 1
    assert inspect.signature(sdb.run).parameters["exec_lag"].default == 0
    assert 'ap.add_argument("--exec-lag", type=int, default=0)' not in \
        _kaynak("research/shadow_decision_backtest.py")


def _sahte_supurme(monkeypatch, secim: dict, dogrulama: dict):
    """_collect 10 bağlam döndürür; _eval_lambda bölmeye göre sabit sözlük (λ'dan bağımsız)."""
    sdb = _sdb()
    monkeypatch.setattr(sdb, "_collect",
                        lambda *a, **k: [{"i": i, "symbol": "A"} for i in range(10)])

    def _ev(ctxs, lam, cost):
        taban = secim if ctxs[0]["i"] == 0 else dogrulama
        return {"lambda": lam, **taban}
    monkeypatch.setattr(sdb, "_eval_lambda", _ev)
    return sdb.lambda_sweep(days=10, horizon=1, lambdas=[0.5, 0.05, 0.0])


def test_hic_lambda_konusmuyorsa_hukum_KONUSAN_LAMBDA_YOK(monkeypatch):
    r = _sahte_supurme(monkeypatch, {"n_yonlu": 0}, {"n_yonlu": 0})
    assert r["VERDIKT_KOD"] == "KONUSAN_LAMBDA_YOK"
    assert "ölçülmüş doğru davranıştır" not in r["VERDIKT"]    # boş kümeden hüküm YOK
    assert "ÖLÇÜLEMEDİ" in r["VERDIKT"]


def test_secimde_konusup_dogrulamada_susan_da_olculemez(monkeypatch):
    r = _sahte_supurme(monkeypatch, {"n_yonlu": 5, "FAZLALIK_vs_hep_long": 1.0},
                       {"n_yonlu": 0})
    assert r["VERDIKT_KOD"] == "KONUSAN_LAMBDA_YOK"


def test_konusan_ama_negatif_fazlalik_KAL(monkeypatch):
    r = _sahte_supurme(monkeypatch, {"n_yonlu": 5, "FAZLALIK_vs_hep_long": 1.0},
                       {"n_yonlu": 5, "FAZLALIK_vs_hep_long": -0.2})
    assert r["VERDIKT_KOD"] == "KAL"
    assert "ölçülmüş doğru davranıştır" in r["VERDIKT"]


def test_konusan_pozitif_fazlalik_DOGRULANDI(monkeypatch):
    r = _sahte_supurme(monkeypatch, {"n_yonlu": 5, "FAZLALIK_vs_hep_long": 1.0},
                       {"n_yonlu": 5, "FAZLALIK_vs_hep_long": 0.3})
    assert r["VERDIKT_KOD"] == "DOGRULANDI"


def test_kpi_olcum_kosulunu_ve_hukum_kodunu_tasir(tmp_path, monkeypatch):
    import nihai.config as cfg
    sdb = _sdb()
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    par = {"gun": 180, "horizon": 5, "exec_lag": 1, "n_baglam": 10}
    sdb._write_kpi({"tablo": [], "VERDIKT_KOD": "KONUSAN_LAMBDA_YOK", "VERDIKT": "x",
                    "parametre": par}, True)
    kpi = json.loads((tmp_path / "uretim_kpi.json").read_text(encoding="utf-8"))
    assert kpi["verdikt_kod"] == "KONUSAN_LAMBDA_YOK" and kpi["parametre"] == par


def _selfcheck_kpi(tmp_path, monkeypatch, kpi: dict):
    import nihai.ai.decision as dm
    from nihai.ai import selfcheck
    (tmp_path / "uretim_kpi.json").write_text(json.dumps(kpi), encoding="utf-8")
    monkeypatch.setattr(selfcheck, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dm, "DECISIONS_FILE", tmp_path / "yok.jsonl")
    return selfcheck.exam_production()


def test_selfcheck_bos_kumeden_abstain_dogru_demez(tmp_path, monkeypatch):
    r = _selfcheck_kpi(tmp_path, monkeypatch, {
        "ts": datetime.now(UTC).isoformat(), "n_yonlu": 0, "en_iyi_konusan_lambda": None,
        "verdikt_kod": "KONUSAN_LAMBDA_YOK",
        "parametre": {"gun": 180, "horizon": 5, "exec_lag": 1, "n_baglam": 5400}})
    assert any("HİÇBİR λ (0 dahil)" in u for u in r["uyarilar"])
    assert not any("kapı gevşetilse" in u for u in r["uyarilar"])    # eski "doğru" hükmü YOK
    assert r["fazlalik"]["olcum_kosulu"] == {"gun": 180, "horizon": 5, "exec_lag": 1,
                                             "n_baglam": 5400}


def test_selfcheck_eski_artefakt_geriye_uyumlu_ve_kosul_uydurmaz(tmp_path, monkeypatch):
    r = _selfcheck_kpi(tmp_path, monkeypatch, {
        "ts": datetime.now(UTC).isoformat(), "n_yonlu": 0,
        "en_iyi_konusan_lambda": {"lambda": 0.05, "FAZLALIK_vs_hep_long": -4.3881}})
    assert any("ABSTAIN ölçülmüş doğru davranış" in u for u in r["uyarilar"])
    assert r["fazlalik"]["olcum_kosulu"] is None and r["fazlalik"]["verdikt_kod"] is None


# ─────────────────────────── PIT evren (BULGU-86-3) ───────────────────────────

def _tablolar():
    guncel = pd.DataFrame({"Symbol": ["AAA", "BBB", "NEW"], "Security": ["a", "b", "n"]})
    navbox = pd.DataFrame({"vteS&P 500 companies": ["Energy"], "vteS&P 500 companies.1": ["x"]})
    cols = pd.MultiIndex.from_tuples([("Effective Date", "Effective Date"), ("Added", "Ticker"),
                                      ("Added", "Security"), ("Removed", "Ticker"),
                                      ("Removed", "Security")])
    degisim = pd.DataFrame([["August 18, 2026", "NEW", "n", "OLD", "o"]], columns=cols)
    return guncel, navbox, degisim


def test_degisiklik_tablosu_konumla_degil_imzayla_bulunur():
    from nihai.data import pit_universe as pit
    guncel, navbox, degisim = _tablolar()
    assert pit._degisiklik_tablosu([guncel, navbox]) is None          # navbox ≠ değişiklik
    assert pit._degisiklik_tablosu([guncel, navbox, degisim]) is degisim
    assert pit._degisiklik_tablosu([degisim, navbox]) is degisim
    assert "raw = tabs[1]" not in _kaynak("nihai/data/pit_universe.py")


def test_ana_maddede_tablo_yoksa_tasindigi_maddeden_okunur(monkeypatch):
    from nihai.data import pit_universe as pit
    guncel, navbox, degisim = _tablolar()
    cagri: list[str] = []

    def _sahte(url):
        cagri.append(url)
        return [guncel, navbox] if url == pit._SP500_URL else [degisim, navbox]
    monkeypatch.setattr(pit, "_read_html_tables", _sahte)
    cur, ch = pit._current_and_changes()
    assert cur == ["AAA", "BBB", "NEW"] and cagri == [pit._SP500_URL, pit._SP500_CHANGES_URL]
    assert pit._geri_sar(cur, ch, "2026-08-01") == ["AAA", "BBB", "OLD"]   # NEW sonra girdi


def test_degisiklik_tablosu_hic_yoksa_sessizce_survivorshipe_dusulmez(monkeypatch):
    from nihai.data import pit_universe as pit
    guncel, navbox, _ = _tablolar()
    monkeypatch.setattr(pit, "_read_html_tables", lambda url: [guncel, navbox])
    with pytest.raises(RuntimeError, match="PIT üyelik KURULAMAZ"):
        pit._current_and_changes()


def test_alt_baslik_degisirse_guncel_kume_gecmis_diye_donmez(monkeypatch):
    """Gemini incelemesi (doğrulandı): "Ticker" → "Symbol" olursa added/removed hiç kurulmaz ve
    geri-sarma boş döner = güncel küme geçmiş üyelik diye SESSİZCE dönerdi (hata YOK)."""
    from nihai.data import pit_universe as pit
    guncel, navbox, _ = _tablolar()
    cols = pd.MultiIndex.from_tuples([("Effective Date", "Effective Date"), ("Added", "Symbol"),
                                      ("Added", "Security"), ("Removed", "Symbol"),
                                      ("Removed", "Security")])
    degisim = pd.DataFrame([["August 18, 2026", "NEW", "n", "OLD", "o"]], columns=cols)
    monkeypatch.setattr(pit, "_read_html_tables", lambda url: [guncel, degisim])
    with pytest.raises(RuntimeError, match="PIT üyelik KURULAMAZ"):
        pit._current_and_changes()


def test_cok_tarihli_uyelik_tek_cekim_ve_sp500_as_of_ile_ayni(monkeypatch):
    from nihai.data import pit_universe as pit
    guncel, _, degisim = _tablolar()
    ch = pd.DataFrame({"date": pd.to_datetime(["2026-08-18"]), "added": ["NEW"],
                       "removed": ["OLD"]})
    sayac = {"n": 0}

    def _sahte():
        sayac["n"] += 1
        return ["AAA", "BBB", "NEW"], ch
    monkeypatch.setattr(pit, "_current_and_changes", _sahte)
    tarihler = ["2026-01-31", "2026-08-18", "2026-09-01"]
    u = pit.sp500_uyelik(tarihler)
    assert sayac["n"] == 1
    for t in tarihler:
        assert u[t] == pit.sp500_as_of(t)


# ─────────────────────────── kapasite süpürmesi --device ───────────────────────────

def _sahte_sweep(monkeypatch, tmp_path, argv: list[str]) -> tuple[dict, list[dict]]:
    import research.capacity_data_sweep as cds
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 3))
    y = (rng.random(3000) < 0.4).astype(int)
    monkeypatch.setattr(cds, "load_real_dataset", lambda p: (X, y, ["a", "b", "c"], 10))
    monkeypatch.setattr(cds, "neutralize_degenerate", lambda X, fn: X)
    alinan: list[dict] = []

    def _cv(Xi, yi, fn, **kw):
        alinan.append(kw)
        return {"auc_mean": 0.5, "auc_std": 0.01, "n_folds_done": 3,
                "dsr_report": {"dsr": 0.0, "psr": 0.0, "sharpe": 0.0}}
    monkeypatch.setattr(cds, "run_cv", _cv)
    out = tmp_path / "o.json"
    cds.main(["--dataset", "x.csv", "--points", "1000,2000,3000", "--json", str(out), *argv])
    return json.loads(out.read_text(encoding="utf-8")), alinan


def test_sweep_aygit_recete_olarak_fite_gider_ve_iki_alan_ayridir(monkeypatch, tmp_path, capsys):
    from nihai.ai.neural import _resolve_device
    res, alinan = _sahte_sweep(monkeypatch, tmp_path, ["--device", "auto"])
    capsys.readouterr()
    assert [kw["recipe"] for kw in alinan] == [
        {"device": "auto", "optimizer_name": "adamw", "amp": False}] * 3
    assert res["device"] == "auto"
    assert res["device_resolved"] == str(_resolve_device("auto"))   # "oldu" ≠ "istendi" alanı


def test_sweep_varsayilani_fitin_varsayilaniyla_ayni(monkeypatch, tmp_path, capsys):
    """Varsayılan reçete `fit` imzasıyla BİT-DENK olmalı; aksi halde eski koşular kıyaslanamaz."""
    import inspect

    from nihai.ai.neural import NeuralCapacity
    res, alinan = _sahte_sweep(monkeypatch, tmp_path, [])
    capsys.readouterr()
    imza = inspect.signature(NeuralCapacity.fit).parameters
    assert all(kw["recipe"] == {"device": "cpu", "optimizer_name": "adamw", "amp": False}
               for kw in alinan)
    assert (imza["device"].default, imza["optimizer_name"].default, imza["amp"].default) == \
        ("cpu", "adamw", False)
    assert "batch_size" not in alinan[0]["recipe"]        # verilmeyen değer İLETİLMEZ
    assert res["device"] == "cpu" and res["device_resolved"] == "cpu"


def test_sweep_bellek_recetesi_ve_maliyet_alanlari(monkeypatch, tmp_path, capsys):
    """Üst basamaklar 8 GB'ta yalnız adafactor + bf16 + küçük toplu-işle eğitilebilir; ve bir
    basamağın "daha iyi" olup olmadığı MALİYETİ bilinmeden okunamaz."""
    res, alinan = _sahte_sweep(monkeypatch, tmp_path,
                               ["--optimizer", "adafactor", "--amp", "--batch-size", "32"])
    capsys.readouterr()
    assert all(kw["recipe"] == {"device": "cpu", "optimizer_name": "adafactor", "amp": True,
                                "batch_size": 32} for kw in alinan)
    assert res["recete"]["optimizer_name"] == "adafactor"
    n0 = res["noktalar"][0]
    assert isinstance(n0["sure_s"], float) and n0["sure_s"] >= 0.0
    assert n0["zirve_vram_mb"] is None                    # CPU'da 0.0 UYDURULMAZ


def test_sweep_hukmu_olculemeyen_dsryi_esikle_kiyaslanmis_gibi_basmaz():
    """quant_transformer PIT-97k koşusu: model hiçbir örnekte çoğunluk kuralından ayrılmadı →
    `dsr_report` None. Eski metin "max DSR=None ≤ 0.05" basıyordu."""
    from research.capacity_data_sweep import _verdikt
    v = _verdikt([{"n": n, "auc_mean": a, "auc_std": 0.01, "dsr": None}
                  for n, a in [(12_000, 0.529), (34_000, 0.499), (97_000, 0.506)]])
    assert v["label"] == "KENAR_YOK_HER_N" and v["dsr_olculen_nokta"] == 0
    assert "HESAPLANAMADI" in v["text"] and "DSR=None" not in v["text"]


def test_sweep_run_cv_cagrisi_receteyi_tasir():
    agac = ast.parse(_kaynak("research/capacity_data_sweep.py"))
    cagrilar = [n for n in ast.walk(agac) if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "run_cv"]
    assert cagrilar and all(any(k.arg == "recipe" for k in c.keywords) for c in cagrilar)


# ─────────────────────────── kalıntı momentumu (BACKLOG #11) ───────────────────────────

def _rm():
    import research.edge_lab.build_rmom_dataset as rm
    return rm


def test_skor_formulu_onkayitli_kurala_esit():
    rm = _rm()
    rng = np.random.default_rng(1)
    F = rng.normal(0, 0.04, size=(36, 3))
    y = 0.002 + F @ np.array([1.2, 0.3, -0.4]) + rng.normal(0, 0.03, 36)
    skor, beta, ivol = rm.kalinti_skoru(y, F)
    X = np.column_stack([np.ones(36), F])
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    olusum = e[24:35]                                  # t−12…t−2: son 12 ayın ilk 11'i
    assert skor == pytest.approx(olusum.sum() / olusum.std(ddof=1), rel=1e-12)
    assert ivol == pytest.approx(e.std(ddof=1), rel=1e-12)
    assert abs(beta - 1.2) < 0.4


def test_tam_aciklanan_seri_skor_uydurmaz():
    rm = _rm()
    F = np.random.default_rng(2).normal(0, 0.04, size=(36, 3))
    y = 0.001 + F @ np.array([1.0, 0.5, 0.2])          # kalıntı ~1e-17 (kayan nokta)
    assert rm.kalinti_skoru(y, F) is None


def _sentetik_px(bas="2016-01-01", son="2023-12-29", tohum=3, bosluk_ay: str | None = None):
    gun = pd.bdate_range(bas, son)
    rng = np.random.default_rng(tohum)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(gun))))
    px = pd.DataFrame({"date": gun, "close": close, "volume": 1e6 + rng.random(len(gun))})
    if bosluk_ay:
        px = px[px["date"].dt.to_period("M") != pd.Period(bosluk_ay)].reset_index(drop=True)
    return px


def _fa(bas="2015-01", son="2023-12", tohum=4):
    idx = pd.period_range(bas, son, freq="M")
    rng = np.random.default_rng(tohum)
    return pd.DataFrame({"mkt_rf": rng.normal(0.008, 0.04, len(idx)),
                         "smb": rng.normal(0, 0.02, len(idx)),
                         "hml": rng.normal(0, 0.02, len(idx)), "rf": 0.001}, index=idx)


def _karar(px):
    return _rm()._ay_sonu_gunleri(pd.DatetimeIndex(px["date"]))


def test_pit_uyelik_suzgeci_karar_gunu_uye_olmayani_disarida_birakir():
    rm = _rm()
    px = _sentetik_px()
    karar = _karar(px)
    hep = {d: {"A"} for d in karar}
    tam = rm.sembol_satirlari("A", px, karar, _fa(), 21, hep)
    assert len(tam) > 20
    dislanan = tam[5]["karar_gunu"]
    kismi = rm.sembol_satirlari("A", px, karar, _fa(), 21, {**hep, dislanan: set()})
    assert [r["karar_gunu"] for r in kismi] == [r["karar_gunu"] for r in tam if
                                                r["karar_gunu"] != dislanan]


def test_toplam_momentum_ayni_olusum_penceresinden_son_ay_haric():
    rm = _rm()
    px = _sentetik_px()
    karar = _karar(px)
    satir = rm.sembol_satirlari("A", px, karar, _fa(), 21, {d: {"A"} for d in karar})
    ay_kap = px.set_index("date")["close"].groupby(lambda t: t.to_period("M")).last()
    for r in satir[::7]:
        m = r["karar_gunu"].to_period("M")
        beklenen = ay_kap[m - 1] / ay_kap[m - 12] - 1.0     # t−12…t−2 = (m−11 … m−1) getirileri
        assert r["f_tmom"] == pytest.approx(beklenen, rel=1e-10)


def test_ay_boslugu_iki_aylik_getiriyi_tek_ay_saymaz():
    rm = _rm()
    px = _sentetik_px(bosluk_ay="2019-06")
    karar = _karar(px)
    satir = rm.sembol_satirlari("A", px, karar, _fa(), 21, {d: {"A"} for d in karar})
    aylar = {r["karar_gunu"].to_period("M") for r in satir}
    # boşluktan sonraki ay (2019-07) NaN → onu içeren her pencere (karar 2019-07…2022-06) düşer
    assert not any(pd.Period("2019-06") <= a <= pd.Period("2022-06") for a in aylar)
    assert any(a < pd.Period("2019-06") for a in aylar)
    assert any(a >= pd.Period("2022-07") for a in aylar)


def test_ozellikler_karar_gununden_sonraki_fiyati_gormez():
    rm = _rm()
    px = _sentetik_px()
    karar = _karar(px)
    uy = {d: {"A"} for d in karar}
    once = {r["karar_gunu"]: r for r in rm.sembol_satirlari("A", px, karar, _fa(), 21, uy)}
    assert all(r["ts"] > d for d, r in once.items())      # feature-lag 1: giriş D+1
    D = sorted(once)[10]
    px2 = px.copy()
    sonra_mi = px2["date"] > D                          # D'den SONRAKİ her fiyat değişti
    # Rampa: sabit çarpan oranı (ileri getiriyi) DEĞİŞTİRMEZDİ ve etiket kontrolü boşa düşerdi
    px2.loc[sonra_mi, "close"] *= np.linspace(1.1, 1.9, int(sonra_mi.sum()))
    sonra = {r["karar_gunu"]: r for r in rm.sembol_satirlari("A", px2, karar, _fa(), 21, uy)}
    for k in ("f_rmom", "f_beta", "f_ivol", "f_tmom", "f_ret_21", "f_vol_60", "f_dvol_z"):
        assert sonra[D][k] == once[D][k], k
    assert sonra[D]["fwd_ret"] != once[D]["fwd_ret"]    # yalnız ETİKET değişir


def test_build_karar_penceresi_uyelik_ve_kesit_fazlaligi(tmp_path, monkeypatch):
    rm = _rm()
    import nihai.data.factors as fac
    import nihai.data.pit_universe as pit
    semboller = [f"S{i}" for i in range(8)]
    panel = {s: _sentetik_px(tohum=10 + i) for i, s in enumerate(semboller)}
    gun = pd.bdate_range("2015-01-01", "2023-12-29")
    ff = pd.DataFrame({"mkt_rf": 0.0004, "smb": 0.0, "hml": 0.0, "rf": 0.00004}, index=gun)
    ff["mkt_rf"] += np.random.default_rng(5).normal(0, 0.01, len(gun))
    monkeypatch.setattr(rm, "load_panel", lambda since, until: panel)
    monkeypatch.setattr(fac, "french_factors", lambda *a, **k: ff)
    # S7 HİÇBİR karar günü üye değil (sonradan eklenen isim gibi)
    monkeypatch.setattr(pit, "sp500_uyelik", lambda t: {x: semboller[:7] for x in t})
    out = tmp_path / "rmom.csv"
    df = rm.build(21, 0.1, 100.0, "2016-01-01", "2023-06-30", str(out))
    assert list(df.columns) == rm.SCHEMA_COLS and not df[rm.FEATURE_COLS].isna().any().any()
    assert "S7" not in set(df["symbol"])
    assert df["ts"].max() <= pd.Timestamp("2023-07-03")          # son karar ≤ until, giriş D+1
    g = df.groupby("ts")["excess"].mean()
    assert np.allclose(g.to_numpy(), 0.0, atol=1e-12)             # kıyas = AYNI PIT kesiti
