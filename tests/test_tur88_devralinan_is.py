"""Tur-88 kilitleri — yarım bırakılmış otomatik aday eğitiminin devralınması.

Dört soru:
  * Karışık tarih biçimi kronolojiyi bozuyor mu? — pandas 3'te `utc=True` tek başına ilk
    elemandan biçim çıkarır; canlı veri setinin %21'i NaT olup zaman bölmesinin SONUNA düşüyordu.
  * Her gece yeniden sınanan aday eğitimi kaçıncı sınav olduğunu taşıyor mu? (Tur-85 dersi)
  * Uyuyan makine gece görevini kaçırınca araç bir çare sunuyor mu? (`-WakeToRun`) Görevin
    yokluğunu görevden bağımsız biri görüyor mu? (deadman `_DAILY`) Günlük dosyası kilitliyken
    gece işi yine koşuyor mu? (`Write-Log`)
  * Belge ölçülmemiş bir "kazanç" iddiasını geri getiriyor mu? (ACI DERS #9)
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Canlı defterdeki gerçek biçimler: 19 karakter naive UTC + 25 karakter ofsetli yerel saat.
_KARISIK = ["2026-07-06 16:01:00", "2026-07-06 12:02:00-04:00", "2026-07-06 16:00:00",
            "2026-07-07 10:00:00-04:00"]


def test_karisik_bicim_kontrolu_eski_cagri_gercekten_NaT_uretiyor():
    # Kontrol: guard kargo-kült değil — düzeltmesiz çağrı bu pandas'ta satır düşürür.
    eski = pd.to_datetime(pd.Series(_KARISIK), errors="coerce", utc=True)
    assert eski.isna().sum() == 2


def test_chrono_key_karisik_bicimde_hic_satir_dusurmez_ve_sirayi_korur():
    from research.train_meta import _chrono_key, chrono_sort

    ts = _chrono_key(pd.Series(_KARISIK))
    assert ts.notna().all()
    df = pd.DataFrame({"entry_time": _KARISIK, "id": [1, 2, 0, 3]})
    assert chrono_sort(df)["id"].tolist() == [0, 1, 2, 3]  # 16:00 < 16:01 < 16:02 (UTC) < ertesi gün


def test_span_days_karisik_bicimde_sifira_dusmez():
    from research.live_retrain import _span_days

    assert _span_days(pd.DataFrame({"entry_time": _KARISIK})) > 0.7


def test_regresyon_kapisi_chrono_key_format_mixed():
    from research import train_meta

    assert 'format="mixed"' in inspect.getsource(train_meta._chrono_key)


def _veri(n, start="2025-01-01"):
    from nihai.signals.meta_features import FEATURE_NAMES

    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = (x[:, 0] + .3 * rng.normal(size=n) > 0).astype(int)
    df = pd.DataFrame(x, columns=FEATURE_NAMES)
    df["label"], df["pnl"] = y, np.where(y == 1, 1., -1.)
    df["symbol"] = [f"S{i % 20}" for i in range(n)]
    df["entry_time"] = pd.date_range(start, periods=n, freq="4h").astype(str)
    df.attrs["cohorts"] = [("1m", "5")]
    return df


def test_aday_egitimi_sinav_numarasini_tasir_ve_tekrarda_artirmaz(tmp_path, monkeypatch):
    from nihai.ai import learning
    from research import auto_retrain as auto
    from research import train_meta as tm

    monkeypatch.setattr(auto, "RUNS", tmp_path / "retraining")
    monkeypatch.setattr(auto, "KINDS", ("logistic",))
    monkeypatch.setattr(tm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(learning, "DATA_DIR", tmp_path)
    data = [_veri(600)]
    monkeypatch.setattr(auto.build_live_dataset, "build", lambda: data[0])
    ilk = auto.run()
    assert ilk["sinav_no"] == 1
    assert auto.run()["status"] == "UNCHANGED"  # aynı veri → sınav sayılmaz
    data[0] = _veri(900)
    ikinci = auto.run()
    assert ikinci["status"] == "COMPLETED" and ikinci["sinav_no"] == 2
    kayit = json.loads((auto.RUNS / "latest.json").read_text(encoding="utf-8"))
    assert kayit["sinav_no"] == 2
    assert ikinci["verdict"] == "DURABLE"  # öğrenilebilir DGP: aday yazılır, sınav no'yu taşımalı
    aday = json.loads(Path(ikinci["candidate_path"]).read_text(encoding="utf-8"))  # lojistik → .json
    assert aday["provenance"]["sinav_no"] == 2
    assert "sınav #2" in learning.learning_report({})


def test_gece_gorevi_uyandirma_secenegi_opt_in():
    s = (ROOT / "register-forward-harness.ps1").read_text(encoding="ascii")
    assert "[switch]$WakeToRun" in s
    assert "if ($WakeToRun) { $settings.WakeToRun = $true }" in s


def test_gece_gorevi_gunluk_yazimi_isi_durduramaz():
    # BULGU-88-4: Add-Content -Path $log ... (ErrorActionPreference=Stop) kilitli günlükte
    # tüm koşuyu başlamadan öldürüyordu (2026-09-22 23:30, sonuç 1).
    s = (ROOT / "forward-harness.ps1").read_text(encoding="utf-8-sig")
    assert "Add-Content -Path $log -Value" not in s
    assert s.count("Write-Log ") >= 9
    govde = s.split("function Write-Log", 1)[1].split("\n}\n", 1)[0]
    assert "-ErrorAction Stop" in govde and "catch" in govde and "-ErrorAction SilentlyContinue" in govde
    assert "Clear-Content -Path $log -ErrorAction SilentlyContinue" in s


def _yas(path: Path, saat: float) -> None:
    import os
    import time

    path.write_text("{}", encoding="utf-8")
    t = time.time() - saat * 3600
    os.utime(path, (t, t))


def test_deadman_gunluk_ogrenme_hafizasi_bayatsa_seans_kapaliyken_de_alarm(monkeypatch, tmp_path):
    from nihai.notifier import deadman as DM

    monkeypatch.setattr(DM, "_market_open", lambda: False)
    monkeypatch.setattr(DM, "_proje_kapali", lambda log_dir=None: False)
    _yas(tmp_path / "ai_learning_state.json", 25)  # tek tetik kaçtı → yedek kapatır, alarm yok
    assert DM.check(data_dir=tmp_path)["verdict"] == "SEANS_KAPALI"
    _yas(tmp_path / "ai_learning_state.json", 28)  # 2026-09-21/22 olayının yaşı
    res = DM.check(data_dir=tmp_path)
    assert res["verdict"] == "BAYAT" and res["stale"] == ["ai_learning_state"]
    monkeypatch.setattr(DM, "_market_open", lambda: None)  # takvim okunamasa da günlük ölçüm geçerli
    assert DM.check(data_dir=tmp_path)["verdict"] == "BAYAT"
    monkeypatch.setattr(DM, "_proje_kapali", lambda log_dir=None: True)
    assert DM.check(data_dir=tmp_path)["verdict"] == "PROJE_KAPALI"


def test_belge_sabit_tabana_kiyaslanmamis_kalibrasyon_kazanci_iddia_etmez():
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Ölçülen tek kazanç KALİBRASYON" not in claude
    assert "BULGU-87-1" in claude and "Tur-88" in claude
