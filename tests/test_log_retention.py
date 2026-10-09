"""BULGU B-7 — logs/ saklama politikası.

Bu modül DOSYA SİLER; testlerin ağırlığı "doğru dosyayı sildi mi"den çok
**yanlış dosyaya dokunmadığından emin olmak** üzerinedir: araştırma verisi
(data/), taze loglar ve tanınmayan adlar hiçbir koşulda silinmemeli.
"""
from __future__ import annotations

from datetime import date

import pytest

from nihai import log_retention as lr

BUGUN = date(2026, 7, 26)


def _yaz(kok, ad: str, kb: int = 1):
    p = kok / ad
    p.write_text("x" * (kb * 1024), encoding="utf-8")
    return p


# --- Seçim doğruluğu ------------------------------------------------------------


def test_eski_karar_logu_aday(tmp_path):
    _yaz(tmp_path, "decision_log_2026_06_01.txt")
    ad = lr.adaylar(tmp_path, days=30, today=BUGUN)
    assert [a.yol.name for a in ad] == ["decision_log_2026_06_01.txt"]


def test_taze_log_aday_degil(tmp_path):
    _yaz(tmp_path, "decision_log_2026_07_20.txt")     # 6 gün önce
    assert lr.adaylar(tmp_path, days=30, today=BUGUN) == []


def test_bugun_ve_dun_dokunulmaz(tmp_path):
    """Canlı süreç bu dosyaları açık tutuyor olabilir — days=0 verilse bile korunur."""
    _yaz(tmp_path, "decision_log_2026_07_26.txt")
    _yaz(tmp_path, "decision_log_2026_07_25.txt")
    assert lr.adaylar(tmp_path, days=0, today=BUGUN) == []


@pytest.mark.parametrize("ad", [
    "predictions_log.csv",          # ARAŞTIRMA VERİSİ — asla
    "live_features_log.csv",        # ARAŞTIRMA VERİSİ — asla
    "meta_dataset_pit_sp500.csv",
    "system.log",                   # aktif log (döndürülmüş .1 değil)
    "hub.log",
    "rastgele_dosya.txt",
    "notlar.md",
])
def test_hedef_disi_dosyalar_asla_secilmez(tmp_path, ad):
    _yaz(tmp_path, ad)
    assert lr.adaylar(tmp_path, days=0, today=BUGUN) == []


def test_tarama_ciktisi_addaki_tarihten_secilir(tmp_path):
    _yaz(tmp_path, "auto_screen_20260601_120000.csv")
    adlar = {a.yol.name for a in lr.adaylar(tmp_path, days=30, today=BUGUN)}
    assert adlar == {"auto_screen_20260601_120000.csv"}


def test_tarihsiz_donmus_log_mtime_ile_degerlendirilir(tmp_path):
    """system.log.3'te ad tarihi yok → mtime'a düşülür. TAZE mtime = korunur;
    yalnız gerçekten eski olan silinir (yanlış-pozitif silme olmaz)."""
    import os
    import time

    taze = _yaz(tmp_path, "system.log.3")
    assert lr.adaylar(tmp_path, days=30, today=BUGUN) == []      # mtime bugün → korunur

    eski = _yaz(tmp_path, "system.log.4")
    eski_ts = time.time() - 120 * 86400
    os.utime(eski, (eski_ts, eski_ts))
    adlar = {a.yol.name for a in lr.adaylar(tmp_path, days=30, today=date.today())}
    assert adlar == {"system.log.4"}
    assert taze.exists()


def test_tarihi_cozulemeyen_ad_atlanir(tmp_path):
    """Silme kararı ASLA tahmine dayanmaz: ad tarihsiz + mtime taze → atla."""
    _yaz(tmp_path, "decision_log_yyyy_mm_dd.txt")     # desene uymuyor zaten
    assert lr.adaylar(tmp_path, days=0, today=BUGUN) == []


def test_olmayan_dizin_bos_doner(tmp_path):
    assert lr.adaylar(tmp_path / "yok", days=30, today=BUGUN) == []


# --- Silme davranışı ------------------------------------------------------------


def test_varsayilan_kuru_kosudur(tmp_path):
    """EN ÖNEMLİ TEST: apply verilmedikçe hiçbir dosya kaybolmaz."""
    p = _yaz(tmp_path, "decision_log_2026_06_01.txt")
    ozet = lr.purge(tmp_path, days=30, today=BUGUN)          # apply YOK
    assert ozet["apply"] is False
    assert ozet["aday"] == 1 and ozet["silinen"] == 0
    assert p.exists()                                         # dosya duruyor


def test_apply_ile_silinir(tmp_path):
    p = _yaz(tmp_path, "decision_log_2026_06_01.txt", kb=4)
    korunan = _yaz(tmp_path, "decision_log_2026_07_26.txt")
    ozet = lr.purge(tmp_path, days=30, apply=True, today=BUGUN)
    assert ozet["silinen"] == 1 and ozet["bayt"] >= 4096
    assert not p.exists()
    assert korunan.exists()


def test_kilitli_dosya_islemi_durdurmaz(tmp_path, monkeypatch):
    """OneDrive/AV kilidi bir dosyayı tutabilir; kalanlar yine temizlenmeli."""
    _yaz(tmp_path, "decision_log_2026_06_01.txt")
    _yaz(tmp_path, "decision_log_2026_06_02.txt")

    gercek = lr.Path.unlink

    def kilitli(self, *a, **k):
        if "06_01" in self.name:
            raise OSError("dosya kullanımda")
        return gercek(self, *a, **k)

    monkeypatch.setattr(lr.Path, "unlink", kilitli)
    ozet = lr.purge(tmp_path, days=30, apply=True, today=BUGUN)
    assert ozet["silinen"] == 1
    assert len(ozet["hatalar"]) == 1 and "06_01" in ozet["hatalar"][0]
