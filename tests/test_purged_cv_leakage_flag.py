"""KALEM 2(d) — sızıntı bayrağı: permütasyon-önem sıralamasında top-1 düşüş ≥ 0.08 ise
purged_cv.main() 'SIZINTI ŞÜPHESİ' satırını basmalı; aksi halde basmamalı.

Bu bayrak yalnız main()'in yazdırma bloğunda yaşıyor (permutation_importance zaten test
edilmiş bir çekirdek fonksiyon — bkz test_purged_cv_leakage.py); bu yüzden gerçek bir CSV
üzerinden main()'i CLI gibi çağırıp (sys.argv monkeypatch) stdout'u (capsys) doğruluyoruz.
DATA_DIR'a dokunmuyoruz (--csv ile açık dosya yolu veriyoruz); conftest zaten NIHAI_DATA_DIR'ı
geçiciye yönlendiriyor ama açık --csv otomatik-keşfi (glob) devre dışı bırakır."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from nihai.signals.meta_features import FEATURE_NAMES
from research import purged_cv


def _write_csv(tmp_path, df: pd.DataFrame) -> str:
    p = tmp_path / "ds.csv"
    df.to_csv(p, index=False)
    return str(p)


def _leaky_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """İlk özellik (adx14) ETİKETİ NEREDEYSE BİREBİR kodluyor (look-ahead/etiket-koreli
    sızıntı simülasyonu) — diğer tüm özellikler saf gürültü. Bu tek özellik permüte
    edildiğinde AUC çökmeli (büyük düşüş → sızıntı bayrağı beklenir)."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    X[:, 0] = label.astype(float) * 10.0 - 5.0 + 0.01 * rng.normal(size=n)   # neredeyse-belirleyici
    pnl = np.where(label == 1, 5.0, -5.0)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = label
    df["pnl"] = pnl
    df["symbol"] = [f"SYM{i % 10}" for i in range(n)]
    df["entry_time"] = pd.date_range("2026-01-01", periods=n, freq="1h").astype(str)
    return df


def _clean_df(n: int = 300, seed: int = 1) -> pd.DataFrame:
    """Hiçbir özellik etiketle ilişkili değil (saf gürültü) — hiçbiri sızıntı eşiğini geçmemeli."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    pnl = np.where(label == 1, 5.0, -5.0)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = label
    df["pnl"] = pnl
    df["symbol"] = [f"SYM{i % 10}" for i in range(n)]
    df["entry_time"] = pd.date_range("2026-01-01", periods=n, freq="1h").astype(str)
    return df


def test_leaky_feature_triggers_suspicion_flag(tmp_path, monkeypatch, capsys):
    csv_path = _write_csv(tmp_path, _leaky_df())
    monkeypatch.setattr(sys, "argv", [
        "purged_cv.py", "--csv", csv_path, "--model", "logistic",
        "--folds", "3", "--null-reps", "0"])
    purged_cv.main()
    out = capsys.readouterr().out
    assert "SIZINTI ŞÜPHESİ" in out
    assert "adx14" in out   # sızdıran özellik ismiyle raporlanmalı


def test_clean_features_do_not_trigger_suspicion_flag(tmp_path, monkeypatch, capsys):
    csv_path = _write_csv(tmp_path, _clean_df())
    monkeypatch.setattr(sys, "argv", [
        "purged_cv.py", "--csv", csv_path, "--model", "logistic",
        "--folds", "3", "--null-reps", "0"])
    purged_cv.main()
    out = capsys.readouterr().out
    assert "SIZINTI ŞÜPHESİ" not in out


def test_leakage_threshold_is_top1_drop_at_least_0_08():
    """Bayrak eşiği (main() içindeki `ranked[0][0] >= 0.08` şartı) doğrudan permutation_importance
    çıktısı üzerinden — sızdıran özelliğin düşüşü belirgin biçimde 0.08'i AŞMALI (aksi halde bu
    test veri kurgusunun yeterince güçlü olmadığını gösterir, eşik mantığını değil)."""
    from research.purged_cv import _build_model, permutation_importance
    from research.train_meta import _auc

    df = _leaky_df()
    # .copy(): permutation_importance sütunu YERİNDE permüte edip geri yükler (Tur-10 D3
    # optimizasyonu) → yazılabilir bir dizi ister. Üretim çağrı yerleri (purged_cv.main,
    # live_retrain._purged_gate) her zaman X_all[te_idx] (fancy-index → otomatik yazılabilir
    # kopya) üzerinden çağırır; burada doğrudan .to_numpy(float) bazı pandas sürümlerinde
    # salt-okunur görünüm döndürebildiğinden açık kopya şart (testin kendi sorunu, üretim kodu değil).
    X = df[FEATURE_NAMES].to_numpy(float).copy()
    y = df["label"].to_numpy(float)
    model = _build_model("logistic", X, y, l2=1.0)
    p = model.predict_proba(X)
    base_auc = _auc(y, p)
    drops = permutation_importance(model, X, y, base_auc=base_auc, seed=0)
    assert drops["adx14"] >= 0.08


@pytest.mark.parametrize("null_reps", [0, 1])
def test_flag_independent_of_null_reps_setting(tmp_path, monkeypatch, capsys, null_reps):
    """Sızıntı bayrağı permütasyon-önemine dayanır, NULL tabana değil — --null-reps 0/1
    ikisinde de aynı şekilde tetiklenmeli (iki mekanizma bağımsız)."""
    csv_path = _write_csv(tmp_path, _leaky_df())
    monkeypatch.setattr(sys, "argv", [
        "purged_cv.py", "--csv", csv_path, "--model", "logistic",
        "--folds", "3", "--null-reps", str(null_reps)])
    purged_cv.main()
    out = capsys.readouterr().out
    assert "SIZINTI ŞÜPHESİ" in out
