"""dir_bar / conf_bar GÖLGE yön hipotezi testleri — Tur-20 (Tur-19 §8 maddesi 2).

Tur-19 ölçtü: giriş `price` (canlı quote) ile çıkış `fwd_price` (bar kapanışı) farklı
kaynaklardan geliyor, fark sistematik olarak aleyhte (+1,85 bps, t=21,2); aynı tahminler
bar kapanışında girilseydi isabet 0,4654 → 0,4937 olurdu.

O ölçüm GİRİŞ FİYATINI değiştirdi, YÖNÜ sabit tuttu. "Yönü de bar kapanışından hesapla"
AYRI ve sınanmamış bir hipotezdir → varsayılan olarak yalnız LOGLANIR. Bu testler o
sözleşmeyi sabitler: gölge hesaplanır, diske yazılır, panoya SIZMAZ, varsayılanı
değiştirmez.

Tur-83: hipotez biriken gölgeyle SINANDI ve varsayılan açıldı (`_yon_kipi`); karşı
(canlı) varyant artık `dir_live`/`conf_live`ta aynı sözleşmeyle yazılır.
"""
from __future__ import annotations

import csv

from nihai.execution.prediction_tracker import FIELDS, _dirbar_cells


# ---------------------------------------------------------------------------
# hücre dönüşümü — savunmacı sözleşme (_neural_cells ile aynı)
# ---------------------------------------------------------------------------
def test_dirbar_cells_gecerli():
    assert _dirbar_cells({"_dirbar": {"direction": "+", "confidence": 0.7312}}) == ("+", "0.7312")
    assert _dirbar_cells({"_dirbar": {"direction": "-", "confidence": 0.5}}) == ("-", "0.5")


def test_dirbar_cells_yoksa_bos():
    assert _dirbar_cells({}) == ("", "")
    assert _dirbar_cells({"_dirbar": None}) == ("", "")
    assert _dirbar_cells({"_dirbar": "bozuk"}) == ("", "")


def test_dirbar_cells_gecersiz_yon_bos():
    """Yön '+'/'-' değilse hiçbir şey yazılmaz (sessiz çöp kolona girmesin)."""
    assert _dirbar_cells({"_dirbar": {"direction": "?", "confidence": 0.5}}) == ("", "")
    assert _dirbar_cells({"_dirbar": {"confidence": 0.5}}) == ("", "")


def test_dirbar_cells_bozuk_guven_yonu_korur():
    """Güven bozuksa yön yine yazılır — yön hipotezi güvenden bağımsız ölçülebilir."""
    assert _dirbar_cells({"_dirbar": {"direction": "+", "confidence": "abc"}}) == ("+", "")
    assert _dirbar_cells({"_dirbar": {"direction": "+"}}) == ("+", "")


def test_kolonlar_semada_ve_sonda():
    """Yeni kolonlar şemaya EKLENEREK girmeli — eski satırların hizası bozulmasın.

    Tur-20'de bu iddia `FIELDS[-2:] == [...]` snapshot'ıydı; Tur-37 gölge ÖLÇÜM kolonlarını
    (mkt_health/har_rv_fc/… ) sona eklediğinde snapshot kırıldı ama KORUNAN ŞEY kırılmadı.
    Kilit artık niyeti ifade ediyor: `dir_bar`/`conf_bar` BİTİŞİK durur, nöral bloğun HEMEN
    ardından gelir ve kendinden ÖNCEKİ hiçbir kolonun sırası değişmez. Sonraki turların
    kolonu SONA eklemesi serbesttir — araya sokması değil (o, eski CSV satırlarını kaydırır)."""
    i = FIELDS.index("dir_bar")
    assert FIELDS[i:i + 2] == ["dir_bar", "conf_bar"]
    assert FIELDS[i - 3:i] == ["neural_pwin", "neural_pwin_side", "neural_pwin_std"]


# ---------------------------------------------------------------------------
# record() uçtan uca — gölge diske yazılır
# ---------------------------------------------------------------------------
def _satir(symbol="AAPL", bar_ts="2026-07-30T14:00:00", **extra):
    row = {"symbol": symbol, "bar_ts": bar_ts, "direction": "+", "action": "BUY",
           "confidence": 0.61, "prob_cal": 0.55, "price": 101.5, "bar_close": 101.0,
           "pwin": None, "pwin_side": None}
    row.update(extra)
    return row


def _oku(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_record_dirbar_kolonlarini_yazar():
    from nihai.execution import prediction_tracker as pt

    n = pt.record([_satir(_dirbar={"direction": "-", "confidence": 0.42})], "1m")
    assert n == 1
    row = _oku(pt.PRED_LOG)[-1]
    assert row["dir_bar"] == "-" and row["conf_bar"] == "0.42"
    # yürürlükteki yön DEĞİŞMEDİ — gölge varsayılanı etkilemez
    assert row["direction"] == "+" and row["confidence"] == "0.61"


def test_record_dirbar_yoksa_bos_kolon():
    """Gölge yoksa kayıt yine de yazılmalı (gölge katman tahmin satırından önemsizdir)."""
    from nihai.execution import prediction_tracker as pt

    assert pt.record([_satir(symbol="MSFT")], "1m") == 1
    row = _oku(pt.PRED_LOG)[-1]
    assert row["dir_bar"] == "" and row["conf_bar"] == ""


def test_gecersiz_dirbar_kaydi_engellemez():
    from nihai.execution import prediction_tracker as pt

    assert pt.record([_satir(symbol="NVDA", _dirbar={"direction": "?"})], "1m") == 1
    assert _oku(pt.PRED_LOG)[-1]["dir_bar"] == ""


# ---------------------------------------------------------------------------
# ANAYASA: gölge panoya sızmaz
# ---------------------------------------------------------------------------
def test_dirbar_web_satirindan_dusurulur():
    """`_dirbar` `_feats`/`_neural` ile aynı desende: kayıttan sonra satırdan düşer.

    Panoda ikinci bir 'yön' sütunu göstermek onu sinyal sanma riski taşır (nöral p_win
    köprüsüyle aynı gerekçe) — bu yüzden alt-çizgili anahtar web/state'e taşınmaz."""
    import inspect

    from nihai.execution import live_predict_service as lps

    src = inspect.getsource(lps)
    assert 'r.pop("_dirbar", None)' in src
    # ve alt-çizgili anahtar sözleşmesi korunuyor (web JSON'u yalnız çizgisiz alanları alır)
    assert '"_dirbar"' in src


def test_varsayilan_bayrak_acik_tur83(monkeypatch):
    """SIGNAL_ON_BAR_CLOSE varsayılanı Tur-83'te AÇILDI — hipotez artık SINANDI.

    Tur-20 bu testi "sınanmamış hipotez üretime girmez" diye KAPALI kilitlemişti ve doğru
    olan buydu. Kanıt birikti: 43.195 satır / 18 gün, AYNI satırlarda AYNI referanstan
    (canlı fiyat) canlı-yön %48,20 ↔ bar-yön %50,12 (gün-kümeli t=+12,8). Kilidin ÖZÜ
    değişmedi: varsayılanı KANIT belirler; kanıt değişince kilit onunla değişir."""
    from nihai.execution.live_predict_service import PredictionService

    monkeypatch.delenv("SIGNAL_ON_BAR_CLOSE", raising=False)
    assert PredictionService()._signal_on_bar_close is True


def test_bayrak_acilabilir(monkeypatch):
    """Geri dönüşlü olmalı: kanıt birikince operatör açabilmeli (env, kod değişikliği değil)."""
    from nihai.execution.live_predict_service import PredictionService

    monkeypatch.setenv("SIGNAL_ON_BAR_CLOSE", "true")
    assert PredictionService()._signal_on_bar_close is True
    monkeypatch.setenv("SIGNAL_ON_BAR_CLOSE", "off")
    assert PredictionService()._signal_on_bar_close is False
