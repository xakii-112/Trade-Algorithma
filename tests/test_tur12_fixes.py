"""Tur-12 denetim düzeltmeleri — regresyon testleri.

Kapsam:
- nihai/atomic_io: OneDrive/AV kilidine dayanıklı atomik yazım (state.json + watchlist +
  prediction_tracker'ın TEK ortak kaynağı; ham os.replace çöküşü = 2026-07-11 olayı).
- prediction_tracker._existing_keys: okuma hatasında None → record() o döngüde YAZMAZ
  (kısmi dedup kümesiyle çift-kayıt = isabet/AUC şişmesi guard'ı).
- purged_cv.purged_folds: iki-TARAFLI (simetrik) horizon-purge (B2 yalnız sol'u düzeltmişti).
"""
from __future__ import annotations

import json
import threading

import pytest

from nihai import atomic_io


def test_atomic_write_text_roundtrip(tmp_path):
    p = tmp_path / "x.json"
    atomic_io.atomic_write_text(p, json.dumps({"a": 1}))
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}
    assert not (tmp_path / "x.json.tmp").exists()  # tmp taşındı, artık yok


def test_atomic_replace_retries_transient(tmp_path, monkeypatch):
    """Geçici OSError (OneDrive kilidi) → yeniden dener, sonunda yazar."""
    real_replace = atomic_io.os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:          # ilk 2 deneme "kilitli"
            raise OSError("locked")
        return real_replace(src, dst)

    monkeypatch.setattr(atomic_io.time, "sleep", lambda *_: None)  # testi hızlandır
    monkeypatch.setattr(atomic_io.os, "replace", flaky)
    p = tmp_path / "s.json"
    atomic_io.atomic_write_text(p, "ok")
    assert p.read_text(encoding="utf-8") == "ok"
    assert calls["n"] == 3          # 2 başarısız + 1 başarılı


def test_atomic_replace_raises_on_permanent(tmp_path, monkeypatch):
    """Kalıcı kilit → istisna YÜKSELİR (sessizce yutulmaz)."""
    def always_fail(*_):
        raise OSError("locked forever")

    monkeypatch.setattr(atomic_io.time, "sleep", lambda *_: None)
    monkeypatch.setattr(atomic_io.os, "replace", always_fail)
    with pytest.raises(OSError):
        atomic_io.atomic_write_text(tmp_path / "s.json", "x", retries=3)
    # Tur-28: kalıcı hatada yarım tmp diskte KALMAZ (ad artık yazana özel; kimse üzerine yazmaz).
    assert list(tmp_path.glob("*.tmp")) == []


def test_tmp_path_yazana_ozel_es_zamanli_yazan_cakismaz(tmp_path):
    """Tur-28: iki YAZAN aynı hedefe yazarken AYNI ara dosyayı paylaşmamalı.

    Eski sabit `<dosya>.tmp` adında sıra `A yaz → B KIRP+yaz → A replace` hedefe B'nin
    yarım içeriğini atomik olarak yayınlıyordu (atomik ama BOZUK). Burada aynı hedef için
    iki farklı thread'in ara yolu FARKLI olmalı."""
    hedef = tmp_path / "state.json"
    yollar: list = []
    t = threading.Thread(target=lambda: yollar.append(atomic_io.tmp_path(hedef)))
    t.start(); t.join()
    benim = atomic_io.tmp_path(hedef)
    assert benim != yollar[0]                 # yazan başına ayrı tmp
    assert benim.parent == hedef.parent       # aynı volüm (os.replace atomikliği)
    assert benim.name.endswith(".tmp")        # .gitignore deseni tutsun


def test_atomic_write_text_es_zamanli_yazanlar_yarim_dosya_yayinlamaz(tmp_path):
    """Uçtan uca: 6 yazan aynı hedefe FARKLI uzunlukta içerik yazarken YAYINLANAN dosya her
    an TAM bir yazımdır (yarım/karışık ASLA). Sabit tmp adıyla bu invariant kırılırdı:
    `A tmp'ye yaz → B tmp'yi KIRPIP yazmaya başla → A replace` hedefe B'nin yarısını atomik
    olarak yayınlar.

    ÖLÇÜLEN ŞEY YALNIZ BÜTÜNLÜKTÜR. Windows'ta yazma/okuma birbirini iter: açık bir okuyucu
    `os.replace`'i düşürebilir (yazan `OSError` alır — atomic_io'nun BELGELENMİŞ sözleşmesi:
    kalıcı kilitte sessizce yutma, yükselt) ve `os.replace` sırasında okuyucu
    `PermissionError` alabilir. İkisi de platform gerçeği, bu düzeltmenin konusu değil →
    ayrı sayılır ve testi düşürmez. Düşüren tek şey TORN (geçersiz) içeriktir."""
    hedef = tmp_path / "state.json"
    icerikler = [str(i) * (400 * (i + 1)) for i in range(6)]
    gecerli = set(icerikler)
    torn: list[str] = []
    kilit_sayisi = {"yaz": 0, "oku": 0}

    def oku() -> None:
        try:
            gorulen = hedef.read_text(encoding="utf-8")
        except (PermissionError, FileNotFoundError):
            kilit_sayisi["oku"] += 1
            return
        if gorulen not in gecerli:
            torn.append(f"{len(gorulen)} bayt: {gorulen[:20]!r}…")

    def yaz(s: str) -> None:
        for _ in range(15):
            try:
                atomic_io.atomic_write_text(hedef, s)
            except OSError:
                kilit_sayisi["yaz"] += 1      # kalıcı kilit → belgelenmiş davranış
            oku()

    tds = [threading.Thread(target=yaz, args=(s,)) for s in icerikler]
    for td in tds:
        td.start()
    for td in tds:
        td.join()
    assert torn == [], f"YARIM/KARIŞIK dosya yayınlandı: {torn[:3]}"
    assert hedef.read_text(encoding="utf-8") in gecerli
    assert list(tmp_path.glob("*.tmp")) == []   # ne başarı ne hata yolu ara dosya bırakır


def test_existing_keys_read_error_skips_record(tmp_path, monkeypatch):
    """_existing_keys None dönerse record() YAZMAZ (kısmi dedup → çift-kayıt guard)."""
    from nihai.execution import prediction_tracker as pt

    monkeypatch.setattr(pt, "PRED_LOG", tmp_path / "predictions_log.csv")
    monkeypatch.setattr(pt, "_SEEN_KEYS", None)
    monkeypatch.setattr(pt, "_SEEN_SIG", None)
    monkeypatch.setattr(pt, "_existing_keys", lambda: None)  # okuma hatasını taklit et

    rows = [{"symbol": "AAA", "action": "LONG", "direction": "+", "price": 10.0,
             "bar_ts": "2026-01-02T15:00:00", "confidence": 0.6}]
    assert pt.record(rows, "1h") == 0        # yazılmadı
    assert not pt.PRED_LOG.exists()          # dosya hiç oluşmadı


def test_existing_keys_ok_still_records(tmp_path, monkeypatch):
    """Kontrol: _existing_keys normal küme dönerse record() yazar (guard yalnız hata yolunda)."""
    from nihai.execution import prediction_tracker as pt

    monkeypatch.setattr(pt, "PRED_LOG", tmp_path / "predictions_log.csv")
    monkeypatch.setattr(pt, "_SEEN_KEYS", None)
    monkeypatch.setattr(pt, "_SEEN_SIG", None)

    rows = [{"symbol": "AAA", "action": "LONG", "direction": "+", "price": 10.0,
             "bar_ts": "2026-01-02T15:00:00", "confidence": 0.6}]
    assert pt.record(rows, "1h") == 1
    assert pt.PRED_LOG.exists()


def test_resolve_marks_aged_out_unresolvable(tmp_path, monkeypatch):
    """400-bar penceresi dışına düşen (delist/uzun kesinti) tahmin UNRESOLVABLE olur, sonsuza
    pending kalmaz, terminal (yeniden denenmez) ve report/coverage onu 'çözülmüş' saymaz."""
    import pandas as pd

    from nihai.execution import prediction_tracker as pt

    monkeypatch.setattr(pt, "PRED_LOG", tmp_path / "predictions_log.csv")
    monkeypatch.setattr(pt, "FEATS_LOG", tmp_path / "live_features_log.csv")
    monkeypatch.setattr(pt, "_SEEN_KEYS", None)
    monkeypatch.setattr(pt, "_SEEN_SIG", None)

    rows = [{"symbol": "OLD", "action": "LONG", "direction": "+", "price": 10.0,
             "bar_ts": "2020-01-02T15:00:00", "pwin_side": "LONG", "confidence": 0.6}]
    assert pt.record(rows, "1h") == 1

    idx = pd.date_range("2026-06-19 10:00:00", periods=10, freq="h")  # bar_ts bunlardan ESKİ
    df = pd.DataFrame({"close": range(10)}, index=idx)

    class _P:
        def get_bars(self, s, i, lookback_bars=400):
            return df

    res = pt.resolve(provider=_P())
    assert res["unresolvable"] == 1 and res["resolved"] == 0 and res["pending"] == 0
    res2 = pt.resolve(provider=_P())          # terminal: bir daha denenmez
    assert res2["unresolvable"] == 0 and res2["pending"] == 0
    assert pt.report()["resolved"] == 0        # isabet istatistiğine girmez
    assert pt.coverage()["unresolvable"] == 1 and pt.coverage()["pending"] == 0


def test_purged_folds_symmetric_two_sided_purge():
    """Tur-12: purge>embargo iken SOL ve SAĞ purge horizon kadar (eskiden sağ=ham embargo)."""
    from research.purged_cv import purged_folds

    n, k, embargo, purge = 100, 5, 2, 10
    saw_middle = False
    for tr, te in purged_folds(n, k, embargo, purge=purge):
        lo, hi = int(te[0]), int(te[-1]) + 1
        trs = {int(x) for x in tr}
        assert not (trs & set(range(max(0, lo - purge), lo)))       # SOL purge
        assert not (trs & set(range(hi, min(n, hi + purge))))       # SAĞ purge (Tur-12)
        if 0 < lo and hi < n:
            saw_middle = True
    assert saw_middle  # en az bir iç fold iki tarafı da sınadı
