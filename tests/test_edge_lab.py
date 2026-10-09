"""research.edge_lab.harness testleri — sentetik BİLGİSİZ veri kapıları FAIL etmeli
(yanlış-pozitif üretmemeli), sentetik AYRIŞAN/KALICI veri kapıları PASS edebilmeli AMA
verdikt metni ASLA 'kenar var' içermemeli (WP-E dürüstlük çerçevesi). Tüm veri setleri
BU dosyada üretilir (gerçek piyasa verisi YOK — sızıntı/harici bağımlılık riski yok).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.edge_lab import harness
from research.edge_lab.harness import run_candidate


# ---------------------------------------------------------------------------
# Sentetik veri üreticileri (deterministik seed'ler)
# ---------------------------------------------------------------------------
def _noise_dataset(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Etiket özelliklerden TAMAMEN bağımsız (rastgele) — 'bilgisiz' aday."""
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i % 20}" for i in range(n)]
    ts = pd.date_range("2022-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "ts": ts, "symbol": symbols,
        "f1": rng.normal(size=n), "f2": rng.normal(size=n),
        "label": rng.integers(0, 2, size=n),
    })


def _strong_same_bar_dataset(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Etiket AYNI bardaki f1'den neredeyse belirlenimli — CV/kapı kolayca PASS eder ama
    exec-lag-1 (ertesi bar için kaydırılmış etiket) sinyali YOK EDER (gerçekçi: aynı-bar
    'sızıntı benzeri' bir ayrışma, kalıcı bir kenar DEĞİL)."""
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i % 20}" for i in range(n)]
    ts = pd.date_range("2022-01-01", periods=n, freq="h")
    f1 = rng.normal(size=n)
    label = (f1 + rng.normal(scale=0.05, size=n) > 0).astype(int)
    return pd.DataFrame({"ts": ts, "symbol": symbols, "f1": f1,
                        "f2": rng.normal(size=n), "label": label})


def _persistent_dataset(n_symbols: int = 15, bars: int = 60, seed: int = 3) -> pd.DataFrame:
    """Sembol-içi YAVAŞ değişen (random-walk) rejim değişkeni → etiket bu değişkenden türer;
    komşu barlar arasında YÜKSEK korelasyon olduğundan 1-bar kaydırma sonrası da (exec-lag)
    kısmen HAYATTA kalır — dört kapının de PASS edebildiği tek senaryo bu dosyada budur."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2022-01-01")
    rows = []
    for s in range(n_symbols):
        walk = np.cumsum(rng.normal(scale=0.05, size=bars))
        for i in range(bars):
            ts = t0 + pd.Timedelta(hours=s * bars + i)
            rows.append((ts, f"SYM{s}", walk[i], rng.normal(), 1 if walk[i] > 0 else 0))
    return pd.DataFrame(rows, columns=["ts", "symbol", "f1", "f2", "label"])


@pytest.fixture()
def noise_csv(tmp_path):
    p = tmp_path / "noise.csv"
    _noise_dataset().to_csv(p, index=False)
    return p


@pytest.fixture()
def strong_csv(tmp_path):
    p = tmp_path / "strong.csv"
    _strong_same_bar_dataset().to_csv(p, index=False)
    return p


@pytest.fixture()
def persistent_csv(tmp_path):
    p = tmp_path / "persistent.csv"
    _persistent_dataset().to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Veri yükleme / savunmacı hatalar
# ---------------------------------------------------------------------------
def test_dataset_bulunamadi_fail_veri_yukleme(tmp_path):
    v = run_candidate("x", tmp_path / "yok.csv", ["f1"], "label")
    assert v["verdict"] == harness.FAIL_VERDICT
    assert v["gates"][0]["name"] == "veri_yukleme"
    assert v["gates"][0]["passed"] is False


def test_eksik_kolon_fail_veri_yukleme(tmp_path, noise_csv):
    v = run_candidate("x", noise_csv, ["f1", "hic_yok_boyle_kolon"], "label")
    assert v["verdict"] == harness.FAIL_VERDICT
    assert "hic_yok_boyle_kolon" in v["gates"][0]["detail"]


# ---------------------------------------------------------------------------
# Bilgisiz (noise) veri → yanlış-pozitif ÜRETMEZ (genel verdikt FAIL)
# ---------------------------------------------------------------------------
def test_gurultu_verisi_genel_verdikt_fail(noise_csv):
    v = run_candidate("noise_test", noise_csv, ["f1", "f2"], "label", n_folds=5)
    assert v["verdict"] == harness.FAIL_VERDICT
    purged = next(g for g in v["gates"] if g["name"] == "purged_cv")
    assert purged["passed"] is False
    assert purged["metrics"]["mean_auc"] < 0.60   # gürültüde AUC şansa yakın kalmalı


def test_gurultu_verisinde_kenar_var_ifadesi_gecmez(noise_csv):
    v = run_candidate("noise_test", noise_csv, ["f1", "f2"], "label", n_folds=5)
    assert "kenar var" not in v["verdict"].lower()


# ---------------------------------------------------------------------------
# Güçlü-ayrık (aynı-bar) veri → CV/kapı PASS edebilir AMA verdikt "kenar var" demez
# ---------------------------------------------------------------------------
def test_guclu_ayrisan_veri_cv_ve_kapi_pass(strong_csv):
    v = run_candidate("strong_test", strong_csv, ["f1", "f2"], "label", n_folds=5)
    purged = next(g for g in v["gates"] if g["name"] == "purged_cv")
    gate_time = next(g for g in v["gates"] if g["name"] == "gate_time")
    gate_symbol = next(g for g in v["gates"] if g["name"] == "gate_symbol")
    assert purged["passed"] is True
    assert purged["metrics"]["mean_auc"] > 0.9
    assert gate_time["passed"] is True
    assert gate_symbol["passed"] is True


def test_guclu_ayrisan_veride_de_kenar_var_ifadesi_gecmez(strong_csv):
    v = run_candidate("strong_test", strong_csv, ["f1", "f2"], "label", n_folds=5)
    assert "kenar var" not in v["verdict"].lower()


def test_ayni_bar_sinyali_exec_lag_sonrasinda_zayiflar(strong_csv):
    """ACI-7 tanısı: aynı-bar 'sızıntı-benzeri' ayrışma, 1-bar kaydırma sonrası kapıyı
    ya kaybeder ya da belirgin biçimde zayıflar (zamanlama artefaktı şüphesi doğru okunur)."""
    v = run_candidate("strong_test", strong_csv, ["f1", "f2"], "label", n_folds=5)
    lag_gate = next(g for g in v["gates"] if g["name"] == "exec_lag_1")
    gate_time = next(g for g in v["gates"] if g["name"] == "gate_time")
    # ya kapıyı hiç geçemez (gate_e=None) ya da geçiyorsa bile lag0'dan gözle görülür düşük
    if lag_gate.get("metrics", {}).get("gate_e") is not None and gate_time["metrics"]["gate_e"] is not None:
        assert lag_gate["metrics"]["gate_e"] < gate_time["metrics"]["gate_e"]
    else:
        assert lag_gate["passed"] is False


# ---------------------------------------------------------------------------
# Kalıcı (yavaş-değişen rejim) sinyal → TÜM kapılar PASS edebilir (dört kapı da)
# ---------------------------------------------------------------------------
def test_kalici_sinyal_tum_kapilar_pass(persistent_csv):
    v = run_candidate("persist_test", persistent_csv, ["f1", "f2"], "label",
                      n_folds=4, thr=0.5)
    assert v["verdict"] == harness.PASS_VERDICT
    assert all(g["passed"] for g in v["gates"])
    assert "kenar var" not in v["verdict"].lower()


def test_kalici_sinyal_exec_lag_etkisi_olculur(persistent_csv):
    v = run_candidate("persist_test", persistent_csv, ["f1", "f2"], "label",
                      n_folds=4, thr=0.5)
    assert v["exec_lag_effect"] is not None
    assert isinstance(v["exec_lag_effect"], float)


# ---------------------------------------------------------------------------
# Sembol kolonu yok → gate_symbol dürüstçe FAIL (çökme yok)
# ---------------------------------------------------------------------------
def test_sembol_kolonu_yoksa_gate_symbol_fail_ama_cokmez(tmp_path):
    df = _noise_dataset()
    df = df.drop(columns=["symbol"])
    p = tmp_path / "no_symbol.csv"
    df.to_csv(p, index=False)
    v = run_candidate("no_symbol_test", p, ["f1", "f2"], "label", n_folds=5)
    gate_symbol = next(g for g in v["gates"] if g["name"] == "gate_symbol")
    assert gate_symbol["passed"] is False
    assert "symbol" in gate_symbol["detail"]


# ---------------------------------------------------------------------------
# purged_cv import-fallback (research.purged_cv erişilemezse yerel purge kullanılır)
# ---------------------------------------------------------------------------
def test_purged_folds_fallback_yerel_uygulama_calisir(monkeypatch, strong_csv):
    monkeypatch.setattr(harness, "_purged_folds_impl", None)
    v = run_candidate("strong_test_fallback", strong_csv, ["f1", "f2"], "label", n_folds=5)
    purged = next(g for g in v["gates"] if g["name"] == "purged_cv")
    assert purged["passed"] is True


def test_local_purged_folds_sozlesmesi_ayni():
    """_local_purged_folds ile research.purged_cv.purged_folds AYNI (train_idx, test_idx)
    bölmelerini üretir (n küçükken) — fallback'in gerçek uygulamayla PARİTELİ olduğunu doğrular."""
    from research.purged_cv import purged_folds as real_impl

    n, k, embargo = 100, 5, 2
    local = list(harness._local_purged_folds(n, k, embargo))
    real = list(real_impl(n, k, embargo))
    assert len(local) == len(real)
    for (ltr, lte), (rtr, rte) in zip(local, real, strict=True):
        assert np.array_equal(ltr, rtr)
        assert np.array_equal(lte, rte)


# ---------------------------------------------------------------------------
# Tur-24: ORANSAL seçicilik tabanı (§26 — "kapı PASS dese bile n_kept okunmalı")
# ---------------------------------------------------------------------------
def test_min_kept_required_kucuk_veride_mutlak_buyukte_oransal_baglar():
    """Küçük test diliminde mutlak taban (5), büyükte oran (%0,5) bağlar."""
    assert harness._min_kept_required(0) == harness._MIN_KEPT
    assert harness._min_kept_required(100) == harness._MIN_KEPT      # 0.5 → tavan 1 < 5
    assert harness._min_kept_required(1000) == 5                     # 5.0 → tam sınır
    assert harness._min_kept_required(33212) == 167                  # §26 örneklemi
    assert harness._min_kept_required(33212) > 29                    # §26'nın 29'u ELENİR


def test_oransal_taban_buyuk_dilimde_seyrek_kapiyi_eler(tmp_path):
    """§26 deseni: kapı büyük bir test diliminden yalnız birkaç işlem tutuyorsa, kapı-E
    base-E'yi geçse BİLE sonuç okunabilir değildir → FAIL, gerekçesi metinde."""
    n = 4000
    rng = np.random.default_rng(11)
    f1 = rng.normal(size=n)
    # Etiket neredeyse tamamen gürültü; yalnız ÇOK uç f1'lerde bilgi var → kapı seyrek tutar
    label = (rng.random(size=n) < np.where(f1 > 3.0, 0.95, 0.5)).astype(int)
    df = pd.DataFrame({"ts": pd.date_range("2022-01-01", periods=n, freq="h"),
                       "symbol": [f"SYM{i % 25}" for i in range(n)],
                       "f1": f1, "f2": rng.normal(size=n), "label": label})
    p = tmp_path / "seyrek.csv"
    df.to_csv(p, index=False)
    v = run_candidate("seyrek", p, ["f1", "f2"], "label", n_folds=5, thr=0.95)
    gate_time = next(g for g in v["gates"] if g["name"] == "gate_time")
    assert gate_time["passed"] is False
    assert gate_time["metrics"]["n_kept"] < gate_time["metrics"]["min_kept_required"]
    assert "gerekli=" in gate_time["detail"]
    assert v["verdict"] == harness.FAIL_VERDICT


# ---------------------------------------------------------------------------
# Tur-24: örneklem-koşullu anlamlılık (Welch t) — "kapı-E > base-E" TEK BAŞINA yetmez
# ---------------------------------------------------------------------------
def test_welch_t_yetersiz_orneklem_veya_sifir_varyansta_none():
    assert harness._welch_t(np.array([1.0]), np.array([1.0, -1.0])) is None
    assert harness._welch_t(np.array([1.0, -1.0]), np.array([1.0])) is None
    # iki tarafta da sıfır varyans → payda 0 → dürüstçe None (varsayılan PASS yok)
    assert harness._welch_t(np.ones(10), np.ones(10)) is None


def test_welch_t_isareti_ve_buyuklugu_beklendigi_gibi():
    kept = np.concatenate([np.ones(50), -np.ones(10)])       # ort +0,667
    dropped = np.concatenate([np.ones(10), -np.ones(50)])    # ort −0,667
    t = harness._welch_t(kept, dropped)
    assert t is not None and t > 5.0
    assert harness._welch_t(dropped, kept) == pytest.approx(-t)


def test_kapi_e_base_e_gecse_de_gurultu_icindeyse_fail(tmp_path, monkeypatch):
    """Kapı-E base-E'yi geçiyor AMA fark örneklem gürültüsünün içinde → FAIL.
    Oransal taban devre dışı bırakılır ki ÖLÇÜLEN şey yalnız anlamlılık olsun."""
    monkeypatch.setattr(harness, "_MIN_KEPT_FRAC", 0.0)
    n = 600
    rng = np.random.default_rng(5)
    f1 = rng.normal(size=n)
    # etiket f1'e ÇOK zayıf bağlı → kapı hafif pozitif seçer ama fark gürültüde kalır
    label = (rng.random(size=n) < (0.5 + 0.02 * np.sign(f1))).astype(int)
    df = pd.DataFrame({"ts": pd.date_range("2022-01-01", periods=n, freq="h"),
                       "symbol": [f"SYM{i % 20}" for i in range(n)],
                       "f1": f1, "f2": rng.normal(size=n), "label": label})
    p = tmp_path / "zayif.csv"
    df.to_csv(p, index=False)
    v = run_candidate("zayif", p, ["f1", "f2"], "label", n_folds=5, thr=0.5)
    gate_time = next(g for g in v["gates"] if g["name"] == "gate_time")
    m = gate_time["metrics"]
    if m["gate_e"] is not None and m["gate_e"] > m["base_e"]:
        assert m["t_stat"] is not None
        assert m["t_stat"] <= harness._GATE_T_THRESHOLD
        assert gate_time["passed"] is False
        assert "gürültüsünü AŞMIYOR" in gate_time["detail"]


def test_kapi_metrikleri_secicilik_ve_t_iceriyor(persistent_csv):
    """Kapı sonucu artık seçiciliği ve t'yi RAPORLUYOR (okuma kuralı makinede)."""
    v = run_candidate("persist_test", persistent_csv, ["f1", "f2"], "label", n_folds=4, thr=0.5)
    gate_time = next(g for g in v["gates"] if g["name"] == "gate_time")
    m = gate_time["metrics"]
    assert set(m) >= {"base_e", "gate_e", "n_kept", "n_test", "keep_frac",
                      "min_kept_required", "t_stat"}
    assert m["keep_frac"] == pytest.approx(m["n_kept"] / m["n_test"], abs=1e-5)
    assert m["t_stat"] is not None and m["t_stat"] > harness._GATE_T_THRESHOLD


# ---------------------------------------------------------------------------
# Tur-24 BULGU: exec_lag_1 kaydırması gerçek pnl_col'u da kaydırmalı
# ---------------------------------------------------------------------------
def test_shift_1bar_pnl_kolonunu_da_kaydirir_ozellikler_kalir():
    df = pd.DataFrame({
        "symbol": ["A", "A", "A", "B", "B"],
        "label": [1.0, 0.0, 1.0, 0.0, 1.0],
        "pnl": [10.0, -20.0, 30.0, -1.0, 2.0],
        "f1": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    out = harness._shift_label_1bar(df, "label", "symbol", "pnl")
    assert list(out["symbol"]) == ["A", "A", "B"]      # her grubun SON satırı düşer
    assert list(out["label"]) == [0.0, 1.0, 1.0]       # etiket t+1'den
    assert list(out["pnl"]) == [-20.0, 30.0, 2.0]      # PnL de t+1'den (BULGU)
    assert list(out["f1"]) == [0.1, 0.2, 0.4]          # ÖZELLİKLER kaydırılmaz


def test_shift_1bar_pnl_verilmezse_yalniz_etiket_kayar():
    df = pd.DataFrame({"symbol": ["A", "A", "A"], "label": [1.0, 0.0, 1.0],
                       "pnl": [10.0, -20.0, 30.0]})
    out = harness._shift_label_1bar(df, "label", "symbol")
    assert list(out["label"]) == [0.0, 1.0]
    assert list(out["pnl"]) == [10.0, -20.0]           # dokunulmadı


def test_shift_1bar_sembol_kolonu_yoksa_ve_tek_satirda():
    df = pd.DataFrame({"label": [1.0, 0.0, 1.0], "pnl": [10.0, -20.0, 30.0]})
    out = harness._shift_label_1bar(df, "label", "symbol", "pnl")
    assert list(out["label"]) == [0.0, 1.0]
    assert list(out["pnl"]) == [-20.0, 30.0]
    tek = pd.DataFrame({"label": [1.0], "pnl": [10.0]})
    assert len(harness._shift_label_1bar(tek, "label", "symbol", "pnl")) == 0


def test_exec_lag_gercek_pnl_ile_gecikmeyi_gercekten_olcer(tmp_path, monkeypatch):
    """Regresyon: PnL yalnız KARAR barında pozitif, sonraki barda negatif. Kaydırma PnL'i
    de taşımazsa exec_lag_1 kapısı gecikmesiz PnL'i ölçer ve sahte PASS verirdi."""
    monkeypatch.setattr(harness, "_MIN_KEPT_FRAC", 0.0)
    rng = np.random.default_rng(7)
    rows = []
    for s in range(20):
        for i in range(40):
            f1 = rng.normal()
            good = f1 > 0
            rows.append((pd.Timestamp("2022-01-01") + pd.Timedelta(hours=s * 40 + i),
                         f"SYM{s}", f1, rng.normal(),
                         1 if good else 0,
                         (5.0 if good else -5.0)))
    df = pd.DataFrame(rows, columns=["ts", "symbol", "f1", "f2", "label", "pnl"])
    # bir sonraki barın PnL'i ile karar barınınki İLİŞKİSİZ olsun diye pnl'i etiketten
    # türettik; kaydırma doğruysa exec_lag kapı-E'si lag0'ınkinin ÇOK altına düşer
    p = tmp_path / "pnl.csv"
    df.to_csv(p, index=False)
    v = run_candidate("pnl_lag", p, ["f1", "f2"], "label", pnl_col="pnl", n_folds=4, thr=0.5)
    gate_time = next(g for g in v["gates"] if g["name"] == "gate_time")
    lag = next(g for g in v["gates"] if g["name"] == "exec_lag_1")
    assert gate_time["metrics"]["gate_e"] is not None
    assert gate_time["metrics"]["gate_e"] > 3.0        # lag0: PnL etiketle hizalı → yüksek
    assert lag["metrics"]["gate_e"] is not None
    assert lag["metrics"]["gate_e"] < 1.0             # lag1: bilgi yok → çöker
    assert lag["passed"] is False


# ---------------------------------------------------------------------------
# BULGU-25 — dsr kapısı piyasa sürüklenmesini beceri sanmamalı
# ---------------------------------------------------------------------------
def _surukleme_dataset(n_symbols: int = 20, bars: int = 40, seed: int = 11) -> pd.DataFrame:
    """BULGU-25'in tam senaryosu: etiket BİLGİSİZ (özelliklerden bağımsız) ama PnL güçlü
    POZİTİF sürüklenme taşıyor (yükselen piyasa, `base_E≈+0,25`). Her alt küme — kapının
    tuttukları, ELEDİKLERİ, rastgele olanı — pozitif ortalamalıdır; sıfıra karşı ölçülen
    Sharpe hepsinde yüksektir. Beceri YOKTUR, dolayısıyla DSR kapısı PASS ETMEMELİDİR."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_symbols):
        for i in range(bars):
            rows.append((pd.Timestamp("2022-01-01") + pd.Timedelta(hours=s * bars + i),
                         f"SYM{s}", rng.normal(), rng.normal(),
                         int(rng.integers(0, 2)),            # etiket = yazı-tura
                         0.25 + rng.normal(scale=0.20)))     # PnL = saf sürüklenme
    return pd.DataFrame(rows, columns=["ts", "symbol", "f1", "f2", "label", "pnl"])


def test_dsr_saf_piyasa_suruklenmesinde_pass_ETMEZ(tmp_path, monkeypatch):
    """Kapı-sonrası HAM PnL sıfıra karşı ölçülürse bu veri DSR≈1,0 verir (BULGU-25).
    Fazlalık serisinde ölçülünce beceri yokluğu görünür ve kapı FAIL eder."""
    monkeypatch.setattr(harness, "_MIN_KEPT_FRAC", 0.0)
    p = tmp_path / "surukleme.csv"
    _surukleme_dataset().to_csv(p, index=False)
    v = run_candidate("surukleme", p, ["f1", "f2"], "label", pnl_col="pnl", n_folds=4, thr=0.5)
    dsr_gate = next(g for g in v["gates"] if g["name"] == "dsr")
    assert dsr_gate["passed"] is False, f"sürüklenme beceri sanıldı: {dsr_gate['detail']}"
    # ...ve taban gerçekten ÇIKARILDI (rapor okunabilirliği + regresyon kilidi)
    assert dsr_gate["metrics"]["taban"] > 0.15
    assert abs(dsr_gate["metrics"]["sr_fazlalik"]) < 0.05      # beceri ~0


def test_dsr_taban_DONEM_farkini_da_temizler():
    """Taban eğitim diliminden alınırsa dönem farkı sahte fazlalık olarak kalır.

    Gerçek rebalans setinde ölçüldü: eğitim tabanı +0,1751 ↔ test tabanı +0,2514; aradaki
    +0,076/işlem tek başına n=27.003'te DSR=1,0 üretiyordu. Burada fark abartılır (0 ↔ +10):
    çıkarılan taban TEST diliminin tabanı olmalı, yoksa hiç bilgisiz kapı bile PASS eder."""
    n = 400
    ts = pd.date_range("2022-01-01", periods=n, freq="h")
    rng = np.random.default_rng(5)
    pnl = np.where(np.arange(n) < 200, rng.normal(scale=1.0, size=n),
                   10.0 + rng.normal(scale=1.0, size=n))
    df = pd.DataFrame({"ts": ts, "symbol": [f"S{i % 10}" for i in range(n)],
                       "f1": rng.normal(size=n), "f2": rng.normal(size=n),
                       "label": rng.integers(0, 2, size=n), "pnl": pnl})
    out = harness._kept_pnl_from_time_gate(df, ["f1", "f2"], "label", "pnl", 0.5, 0.5)
    assert out is not None
    fazlalik, taban = out
    assert taban == pytest.approx(10.0, abs=0.5), f"dönem farkı temizlenmedi: {taban}"
    assert abs(float(np.mean(fazlalik))) < 0.5      # bilgisiz kapı → fazlalık ~0


def test_dsr_kapinin_ELEDIGI_ve_RASTGELE_alt_kume_PASS_ETMEZ(tmp_path, monkeypatch):
    """BULGU-25'in ayırt-edicilik ölçütü, birebir.

    Teşhis şuydu: DSR=1,0'ı kapının ELEDİĞİ satırlar ve RASTGELE alt küme de üretiyordu.
    Gerçek rebalans setinde düzeltme sonrası ölçüldü — kapı 1,0 (fazlalık gerçek,
    Welch t=8,51 ile tutarlı), ELENEN 0,0000, RASTGELE 0,2649. Bu test aynı ayrımı
    sentetik veride kilitler: sürüklenmeli havuzda hiçbir bilgisiz alt küme geçmemeli."""
    monkeypatch.setattr(harness, "_MIN_KEPT_FRAC", 0.0)
    df = _surukleme_dataset()
    pnl = df["pnl"].to_numpy(dtype=float)
    taban = float(np.mean(pnl))
    rng = np.random.default_rng(0)
    rastgele = rng.choice(pnl, size=len(pnl) // 8, replace=False)
    for ad, seri in [("RASTGELE", rastgele), ("TUMU", pnl)]:
        gate, _ = harness._stage_dsr((seri - taban, taban))
        assert gate.passed is False, f"{ad} bilgisiz alt küme PASS etti: {gate.detail}"
        # ...ham seride (eski davranış) aynı küme geçerdi — kıyas kaydı
        ham_gate, _ = harness._stage_dsr((seri, 0.0))
        assert ham_gate.passed is True, f"{ad} ham seride zaten fail — test ayrımı ölçmüyor"


# ---------------------------------------------------------------------------
# Verdikt dict şeması + sabitler
# ---------------------------------------------------------------------------
def test_verdikt_dict_semasi(noise_csv):
    v = run_candidate("x", noise_csv, ["f1", "f2"], "label", n_folds=5)
    assert set(v) == {"candidate", "dataset", "n_rows", "generated_at", "gates", "dsr",
                      "exec_lag_effect", "prior_runs", "verdict"}
    assert v["candidate"] == "x"
    assert v["n_rows"] == 600
    for g in v["gates"]:
        assert set(g) == {"name", "passed", "detail", "metrics"}


def test_pass_ve_fail_verdict_sabitleri_kenar_var_icermez():
    assert "kenar var" not in harness.PASS_VERDICT.lower()
    assert "kenar var" not in harness.FAIL_VERDICT.lower()
