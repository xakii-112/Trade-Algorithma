"""research.train_neural testleri — sentetik müfredat üretici, DSR raporu, CV döngüsü, CLI.

Torch'lu testler `@pytest.mark.skipif(not neural.TORCH_AVAILABLE, ...)` ile (SPEC §4).
`--save` verilmeyen koşularda dosya yazılmadığı ayrıca doğrulanır (tmp_path izlenir).
Ağır testler küçük tutulur (mini --synthetic, --max-epochs<=2, deep_mlp tercih edilir —
quant_transformer yalnız bir smoke testte, tiny n ile).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nihai.ai import neural
from research import train_neural

TORCH = pytest.mark.skipif(not neural.TORCH_AVAILABLE, reason="torch yok")


@TORCH
def test_progress_measures_epochs_without_changing_training():
    X = np.random.default_rng(8).normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(float)
    models = [train_neural.build_model("deep_mlp", ["a", "b", "c"], 42, 1) for _ in range(2)]
    events = []
    models[0].fit(X, y, max_epochs=2)
    models[1].fit(X, y, max_epochs=2, progress=events.append)
    assert [e["epoch"] for e in events] == [1, 2]
    assert all(np.isfinite(e["train_loss_last_batch"]) and np.isfinite(e["val_loss"]) for e in events)
    np.testing.assert_array_equal(models[0].predict_proba(X), models[1].predict_proba(X))


@TORCH
def test_cli_progress_and_json_report(tmp_path, capsys):
    path = tmp_path / "training.json"
    assert train_neural.main(["--tier", "deep_mlp", "--synthetic", "120", "--max-epochs", "2",
                              "--folds", "2", "--progress", "--json", str(path)]) == 0
    assert "EPOCH " in capsys.readouterr().out
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["n_folds_done"] == 2 and report["auc_mean"] is not None
    assert "progress" not in report["recipe"]


# ---------------------------------------------------------------------------
# Sentetik müfredat üretici (BÖLÜM F)
# ---------------------------------------------------------------------------
def test_build_synthetic_dataset_sekiller():
    X, y, feature_names, aux = train_neural.build_synthetic_dataset(120, seed=1)
    assert X.shape == (120, len(train_neural.SYNTHETIC_FEATURES))
    assert y.shape == (120,)
    assert feature_names == train_neural.SYNTHETIC_FEATURES
    assert set(aux.keys()) == {"magnitude", "vol", "regime"}
    for v in aux.values():
        assert v.shape == (120,)


def test_build_synthetic_dataset_etiket_ikili():
    _, y, _, _ = train_neural.build_synthetic_dataset(80, seed=2)
    assert set(np.unique(y).tolist()) <= {0.0, 1.0}


def test_build_synthetic_dataset_aux_araliklari():
    _, _, _, aux = train_neural.build_synthetic_dataset(200, seed=3)
    assert aux["magnitude"].min() >= 0 and aux["magnitude"].max() <= 4
    assert aux["regime"].min() >= 0 and aux["regime"].max() <= 5
    assert np.all(np.isfinite(aux["vol"]))
    assert np.all(aux["vol"] >= 0)


def test_build_synthetic_dataset_nan_yok():
    """leak-free pencereleme sırasında oluşabilecek ısınma-dönemi NaN'ları temizlenir —
    modele ASLA NaN sızmaz (savunmacı ikinci kontrol, bkz. fonksiyon docstring'i)."""
    X, y, _, aux = train_neural.build_synthetic_dataset(150, seed=4)
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y))
    for v in aux.values():
        assert np.all(np.isfinite(v))


def test_build_synthetic_dataset_determinizm_ayni_seed():
    X1, y1, _, aux1 = train_neural.build_synthetic_dataset(100, seed=7)
    X2, y2, _, aux2 = train_neural.build_synthetic_dataset(100, seed=7)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    for k in aux1:
        assert np.array_equal(aux1[k], aux2[k])


def test_build_synthetic_dataset_farkli_seed_farkli_veri():
    X1, _, _, _ = train_neural.build_synthetic_dataset(100, seed=1)
    X2, _, _, _ = train_neural.build_synthetic_dataset(100, seed=2)
    assert not np.array_equal(X1, X2)


def test_synthetic_features_coklu_olcek_onekleri_kapsiyor():
    prefixes = {neural._feature_scale_id(n) for n in train_neural.SYNTHETIC_FEATURES}
    assert prefixes == {0, 1, 2, 3}   # öneksiz + h_/d_/w_ hepsi temsil edilir


def test_generate_synthetic_series_determinizm():
    s1, g1 = train_neural._generate_synthetic_series(500, seed=9)
    s2, g2 = train_neural._generate_synthetic_series(500, seed=9)
    assert np.array_equal(s1, s2)
    assert np.array_equal(g1, g2)
    assert set(np.unique(g1).tolist()) <= {0, 1, 2}


# ---------------------------------------------------------------------------
# DSR raporu (deflated.py yeniden kullanımı)
# ---------------------------------------------------------------------------
def test_dsr_report_yetersiz_veri_none_alanlar():
    res = train_neural.dsr_report([1.0], n_trials=1)
    assert res["sharpe"] is None and res["dsr"] is None and res["psr"] is None
    assert res["n"] == 1


def test_dsr_report_bos_liste():
    res = train_neural.dsr_report([], n_trials=1)
    assert res["n"] == 0
    assert res["dsr"] is None


def test_dsr_report_gecerli_veri_alanlari():
    rng = np.random.default_rng(0)
    pseudo = (rng.random(200) > 0.48).astype(float) * 2 - 1   # hafif pozitif yanlı ±1 serisi
    res = train_neural.dsr_report(pseudo.tolist(), n_trials=4)
    for key in ("n", "sharpe", "psr", "dsr", "n_trials"):
        assert key in res
    assert res["n"] == 200
    assert res["n_trials"] == 4
    assert res["sharpe"] is not None


def test_dsr_report_n_trials_arttikca_dsr_azalir_ya_da_esit():
    """Çoklu-test deflasyonu: AYNI Sharpe/n için daha çok deneme (n_trials) → DSR ≥ deflasyonu
    (E[max SR] artar) → DSR AYNI ya da DAHA KÜÇÜK olmalı (deflated.py'nin kendi monotonluk
    sözleşmesi; burada CLI köprüsünün doğru argüman sırasıyla çağırdığının kanıtı)."""
    rng = np.random.default_rng(1)
    pseudo = (rng.random(300) > 0.45).astype(float) * 2 - 1
    az_deneme = train_neural.dsr_report(pseudo.tolist(), n_trials=1)
    cok_deneme = train_neural.dsr_report(pseudo.tolist(), n_trials=50)
    assert az_deneme["dsr"] is not None and cok_deneme["dsr"] is not None
    assert cok_deneme["dsr"] <= az_deneme["dsr"] + 1e-9


# ---------------------------------------------------------------------------
# run_cv (purged_cv.py bölme deseni + neural fit/predict)
# ---------------------------------------------------------------------------
@TORCH
def test_run_cv_kucuk_sentetik_deep_mlp():
    X, y, feature_names, aux = train_neural.build_synthetic_dataset(300, seed=5)
    result = train_neural.run_cv(X, y, feature_names, tier="deep_mlp", folds=3,
                                 max_epochs=2, aux=aux, purge=train_neural._HORIZON + 1)
    assert result["n_folds_done"] > 0
    assert len(result["fold_aucs"]) == result["n_folds_done"]
    assert result["auc_mean"] is not None
    assert 0.0 <= result["auc_mean"] <= 1.0
    assert result["dsr_report"]["n_trials"] == result["n_folds_done"]


@TORCH
def test_run_cv_purge_none_calisir():
    X, y, feature_names, aux = train_neural.build_synthetic_dataset(200, seed=6)
    result = train_neural.run_cv(X, y, feature_names, tier="deep_mlp", folds=2,
                                 max_epochs=1, aux=aux, purge=None)
    assert result["n_folds_done"] >= 0   # embargo-simetrik purge ile de çökmeden tamamlanır


@TORCH
def test_run_cv_quant_transformer_smoke():
    """quant_transformer'ın CV döngüsünde de çalıştığının minimal smoke testi (tiny n,
    tek epoch) — ağır-test disiplini: param sözleşmesi ayrı ve fit'siz test edilir."""
    X, y, feature_names, aux = train_neural.build_synthetic_dataset(120, seed=8)
    result = train_neural.run_cv(X, y, feature_names, tier="quant_transformer", folds=2,
                                 max_epochs=1, aux=aux, purge=train_neural._HORIZON + 1)
    assert result["tier"] == "quant_transformer"
    assert result["n_folds_done"] > 0


# ---------------------------------------------------------------------------
# load_real_dataset
# ---------------------------------------------------------------------------
def test_load_real_dataset_donus_bicimi(tmp_path):
    from nihai.signals.meta_features import FEATURE_NAMES
    n = 40
    rng = np.random.default_rng(0)
    df = pd.DataFrame({name: rng.normal(size=n) for name in FEATURE_NAMES[:3]})
    df["label"] = (rng.random(n) > 0.5).astype(int)
    df["entry_time"] = pd.date_range("2024-01-01", periods=n, freq="h").astype(str)
    df["bars_held"] = rng.integers(1, 5, size=n)
    csv_path = tmp_path / "mini_dataset.csv"
    df.to_csv(csv_path, index=False)

    X, y, feature_names, purge = train_neural.load_real_dataset(str(csv_path))
    assert X.shape == (n, len(FEATURE_NAMES))
    assert y.shape == (n,)
    assert feature_names == list(FEATURE_NAMES)
    assert purge is None or isinstance(purge, int)


# ---------------------------------------------------------------------------
# CLI (main()) — smoke + --save yazma disiplini + hata yolları
# ---------------------------------------------------------------------------
@TORCH
def test_cli_smoke_synthetic_deep_mlp(capsys):
    rc = train_neural.main(["--tier", "deep_mlp", "--folds", "2", "--synthetic", "150",
                            "--max-epochs", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OOS-AUC" in out
    assert "DSR raporu" in out
    assert "kenar" in out   # "'kenar' iddiası içermez" notu HER ZAMAN basılır
    assert "hiçbir dosya yazılmadı" in out


@TORCH
def test_cli_save_yokken_dosya_yazilmaz(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = train_neural.main(["--tier", "deep_mlp", "--folds", "2", "--synthetic", "120",
                            "--max-epochs", "1"])
    assert rc == 0
    assert list(tmp_path.iterdir()) == []   # --save verilmedi → HİÇBİR dosya yazılmadı


@TORCH
def test_cli_save_verilince_dosya_yazilir(tmp_path, capsys):
    base = tmp_path / "alt" / "cli_model"
    rc = train_neural.main(["--tier", "deep_mlp", "--folds", "2", "--synthetic", "150",
                            "--max-epochs", "1", "--save", str(base)])
    assert rc == 0
    assert (tmp_path / "alt" / "cli_model.json").exists()
    assert (tmp_path / "alt" / "cli_model.pt").exists()
    out = capsys.readouterr().out
    assert "kaydedildi" in out
    loaded = neural.NeuralCapacity.load(base)
    assert loaded is not None
    assert loaded.param_count() > 0


@TORCH   # torch'suz ortamda CLI daha erken fail-soft torch mesajıyla çıkar (ayrı kod yolu)
def test_cli_veri_kaynagi_yoksa_hata_kodu(capsys):
    rc = train_neural.main(["--tier", "deep_mlp", "--folds", "2"])   # ne --dataset ne --synthetic
    assert rc == 1
    assert "Veri kaynağı yok" in capsys.readouterr().out


def test_cli_gecersiz_tier_argparse_hatasi():
    with pytest.raises(SystemExit):
        train_neural.main(["--tier", "bilinmeyen_tier", "--synthetic", "100"])


# ---------------------------------------------------------------------------
# Sözde-getiri = çoğunluk-kuralına göre FAZLALIK (Tur-17 kanıt-kapısı düzeltmesi)
# ---------------------------------------------------------------------------
def test_excess_pseudo_returns_trivial_kural_sifir_uretir():
    """REGRESYON: dengesiz veride (taban %30) 'hep çoğunluk sınıfını söyle' kuralı
    SIFIR fazlalık üretmeli. Ham isabet (2·hit−1) kullanılsaydı bu seri +0.4 ortalama
    ve büyük n'de DSR≈1.0 verirdi — sınıf dengesizliğini beceri sanan yanlış-pozitif."""
    rng = np.random.default_rng(0)
    y_train = (rng.random(1000) < 0.3).astype(float)
    y_test = (rng.random(1000) < 0.3).astype(float)
    p_const = np.zeros(1000)                       # her satırda p<0.5 → hep sınıf 0 (çoğunluk)
    ex = train_neural.excess_pseudo_returns(p_const, y_test, y_train)
    assert np.all(ex == 0.0)
    assert train_neural.dsr_report(ex.tolist(), n_trials=5)["dsr"] is None


def test_excess_pseudo_returns_gercek_beceri_pozitif():
    y_train = np.array([0.0, 0.0, 0.0, 1.0])       # çoğunluk = 0
    y_test = np.array([1.0, 1.0, 0.0, 0.0])
    p_perfect = np.array([0.9, 0.9, 0.1, 0.1])     # kusursuz tahmin
    ex = train_neural.excess_pseudo_returns(p_perfect, y_test, y_train)
    assert ex.tolist() == [1.0, 1.0, 0.0, 0.0]     # yalnız çoğunluğun ıskaladığında kazanır
    assert float(ex.mean()) > 0


def test_excess_pseudo_returns_cogunluk_egitimden_hesaplanir():
    """Sızıntı yok: çoğunluk sınıfı TEST etiketlerinden değil EĞİTİM fold'undan gelir."""
    y_train = np.array([1.0, 1.0, 1.0, 0.0])       # eğitimde çoğunluk = 1
    y_test = np.array([0.0, 0.0, 0.0, 0.0])        # testte her şey 0 (dağılım kaymış)
    p_const_zero = np.zeros(4)
    ex = train_neural.excess_pseudo_returns(p_const_zero, y_test, y_train)
    # Model hep 0 der (4/4 isabet); eğitim-çoğunluğu 1 olduğundan trivial kural 0/4 isabet
    assert ex.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_dsr_report_sifir_varyans_none_doner():
    rapor = train_neural.dsr_report([0.0] * 50, n_trials=3)
    assert rapor["dsr"] is None and rapor["psr"] is None and rapor["sharpe"] is None
    assert rapor["n"] == 50


# ---------------------------------------------------------------------------
# Model kartı (Tur-17) — registry köprüsü. build_model_card SAF: torch gerektirmez.
# ---------------------------------------------------------------------------
def _ornek_cv_sonucu() -> dict:
    return {"auc_mean": 0.503, "auc_std": 0.021, "n_folds_done": 3,
            "dsr_report": {"dsr": 0.11, "psr": 0.42, "sharpe": 0.08, "n_trials": 3, "n": 120}}


def test_build_model_card_alanlari():
    card = train_neural.build_model_card(
        tier="deep_mlp", feature_names=["a", "b"], param_count=1234,
        data_source="synthetic:3000", n_rows=3000, positive_rate=0.4977, purge=6,
        folds=3, max_epochs=40, seed=42, cv_result=_ornek_cv_sonucu(), save_path="data/x/m")
    assert card["kind"] == "neural" and card["tier"] == "deep_mlp"
    assert card["param_count"] == 1234 and card["n_features"] == 2
    assert card["data"] == {"source": "synthetic:3000", "n_rows": 3000,
                            "positive_rate": 0.4977, "purge": 6}
    assert card["oos"]["auc_mean"] == 0.503 and card["oos"]["n_folds_done"] == 3
    assert card["dsr"]["dsr"] == 0.11 and card["dsr"]["n_trials"] == 3
    assert card["artifacts"]["files"] == ["data/x/m.json", "data/x/m.pt"]
    assert card["env"]["python"]


def test_build_model_card_kenar_iddiasi_icermez():
    """ACI DERSLER: kart ölçümü kaydeder, YORUMLAMAZ — pozitif kenar dili yasak."""
    metin = json.dumps(train_meta_card := train_neural.build_model_card(
        tier="deep_mlp", feature_names=["a"], param_count=10, data_source="synthetic:100",
        n_rows=100, positive_rate=0.5, purge=None, folds=2, max_epochs=1,
        seed=42, cv_result=_ornek_cv_sonucu(), save_path="m"), ensure_ascii=False).lower()
    for yasak in ("kenar var", "kenar bulundu", "dayanıklı kenar", "kârlı"):
        assert yasak not in metin
    # NOT: karşılaştırma büyük/küçük harf DUYARLI — Python'da "DEĞİLDİR".lower() Türkçe
    # noktalı-İ'yi birleşen noktayla üretir ("deği̇ldi̇r"), casefold kıyası yanıltıcı olur.
    assert "kenar iddiası DEĞİLDİR" in train_meta_card["not"]


def test_build_model_card_ts_icermez_hash_kararli():
    """Kart içeriğinde zaman damgası YOK: aynı eğitim → aynı registry hash'i (idempotent
    dizin). `registered_at`'i registry ekler, hash'e girmez."""
    from nihai.ai import registry
    kw = dict(tier="deep_mlp", feature_names=["a"], param_count=10, data_source="synthetic:100",
              n_rows=100, positive_rate=0.5, purge=None, folds=2, max_epochs=1, seed=42,
              cv_result=_ornek_cv_sonucu(), save_path="m")
    c1 = train_neural.build_model_card(**kw)
    c2 = train_neural.build_model_card(**kw)
    assert c1 == c2
    assert registry._content_hash(c1) == registry._content_hash(c2)


@TORCH
def test_cli_register_save_ile_kart_yazar(tmp_path, capsys):
    base = tmp_path / "model" / "m"
    reg = tmp_path / "kayit"
    rc = train_neural.main(["--tier", "deep_mlp", "--folds", "2", "--synthetic", "150",
                            "--max-epochs", "1", "--save", str(base),
                            "--register", "--registry-dir", str(reg)])
    assert rc == 0
    from nihai.ai import registry
    kartlar = registry.list_models(artifacts_base_path=reg)
    assert len(kartlar) == 1
    kart = kartlar[0]
    assert kart["tier"] == "deep_mlp" and kart["kind"] == "neural"
    assert kart["data"]["source"] == "synthetic:150"
    assert kart["param_count"] > 0
    assert kart["hash"] and kart["registered_at"]
    assert "model kartı yazıldı" in capsys.readouterr().out


@TORCH
def test_cli_register_save_olmadan_yazmaz(tmp_path, capsys, monkeypatch):
    """--register tek başına HİÇBİR ŞEY yazmaz (artefaktsız kart yanıltıcı olurdu)."""
    monkeypatch.chdir(tmp_path)
    reg = tmp_path / "kayit"
    rc = train_neural.main(["--tier", "deep_mlp", "--folds", "2", "--synthetic", "120",
                            "--max-epochs", "1", "--register", "--registry-dir", str(reg)])
    assert rc == 0
    assert not reg.exists()
    out = capsys.readouterr().out
    assert "YOK SAYILDI" in out


@TORCH
def test_cli_dataset_oncelikli_synthetic_yoksayilir(tmp_path, capsys):
    """Hem --dataset hem --synthetic verilirse gerçek veri ÖNCELİKLİDİR (modül docstring'i)."""
    from nihai.signals.meta_features import FEATURE_NAMES
    n = 30
    rng = np.random.default_rng(0)
    df = pd.DataFrame({name: rng.normal(size=n) for name in FEATURE_NAMES[:2]})
    df["label"] = (rng.random(n) > 0.5).astype(int)
    csv_path = tmp_path / "gercek.csv"
    df.to_csv(csv_path, index=False)

    rc = train_neural.main(["--tier", "deep_mlp", "--folds", "2", "--dataset", str(csv_path),
                            "--synthetic", "999999", "--max-epochs", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dataset (gerçek)" in out
    assert "SENTETİK" not in out
