"""Tur-64 — `dolt_options.read_bulk` parquet yan dosyası.

Üç soru sorulur:

1. **EŞDEĞER Mİ?** Parquet yolu ile CSV yolu AYNI çerçeveyi vermeli — yalnız
   satır sayısı değil: kolon SIRASI (pandas `usecols` DOSYA sırasını korur,
   pyarrow `columns=` verilen sırayı), dtype'lar ve satır sırası. Bir hızlanma,
   çağıranın gördüğü çerçeveyi değiştiriyorsa hızlanma değil sessiz bir göçtür.
2. **GERİ ALINABİLİR Mİ?** pyarrow yokken davranış eskisiyle birebir aynı
   kalmalı (Tur-26: opsiyonel bağımlılık tüm kapıyı düşürmemeli) ve yarım kalan
   bir yazım ASLA yayımlanmamalı.
3. **BAYAT VERİ DÖNDÜRÜR MÜ?** `bulk_csv` dosyayı 7 günde bir tazeler; CSV'den
   eski bir yan dosya sessizce eski veriyi döndürürdü — bu depodaki en pahalı
   hata sınıfı budur.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from nihai.data import dolt_options as D

pq = pytest.importorskip("pyarrow.parquet", reason="parquet yolu yalnız pyarrow ile ölçülebilir")

COLS = ["date", "act_symbol", "iv_current", "hv_current"]


def _yaz_csv(tmp_path, satir=1200):
    """Gerçek `volatility_history` şemasına benzer küçük bir toplu CSV."""
    syms = ["AAPL", "MSFT", "SPY", "NVDA"]
    rows = []
    for i in range(satir):
        rows.append({
            "date": f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}",
            "act_symbol": syms[i % len(syms)],
            "iv_current": 0.10 + (i % 50) / 100.0,
            "iv_year_high": 0.90,
            "iv_year_low": 0.05,
            "hv_current": 0.12 + (i % 40) / 100.0,
            "hv_month_ago": None if i % 3 == 0 else 0.2,   # gerçek veri gibi NaN'lı kolon
        })
    p = tmp_path / "volatility_history__master.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _csv_yolu(path, cols, symbols, chunksize=500):
    """Değiştirilmeden ÖNCEKİ okuma: parça parça CSV + sembol süzmesi."""
    syms = None if symbols is None else sorted({s.upper() for s in symbols})
    frames = []
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        if syms is not None:
            chunk = chunk[chunk["act_symbol"].str.upper().isin(syms)]
        if not chunk.empty:
            frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(cols))


# --------------------------------------------------------------------------- 1. eşdeğerlik
@pytest.mark.parametrize("symbols", [None, ["AAPL"], ["AAPL", "SPY"], ["aapl"]])
def test_parquet_yolu_csv_yoluyla_esdeger(tmp_path, symbols):
    p = _yaz_csv(tmp_path)
    beklenen = _csv_yolu(p, COLS, symbols)
    alinan = D.read_bulk(p, cols=COLS, symbols=symbols, chunksize=500)
    assert D._sidecar_path(p).exists(), "yan dosya kurulmadı"
    pd.testing.assert_frame_equal(alinan, beklenen)   # değer + dtype + KOLON SIRASI + satır sırası


def test_kolonsuz_okuma_tum_tabloyu_verir(tmp_path):
    p = _yaz_csv(tmp_path)
    alinan = D.read_bulk(p, chunksize=500)
    beklenen = pd.read_csv(p)
    pd.testing.assert_frame_equal(alinan, beklenen)


def test_eslesmeyen_sembol_bos_ama_kolonlu_doner(tmp_path):
    p = _yaz_csv(tmp_path)
    out = D.read_bulk(p, cols=COLS, symbols=["YOKBOYLEBIRSEY"], chunksize=500)
    assert out.empty and list(out.columns) == COLS


def test_olmayan_dosya_bos_doner(tmp_path):
    assert D.read_bulk(tmp_path / "yok.csv", cols=COLS).empty
    assert D.read_bulk(None).empty


# --------------------------------------------------------------------------- 2. geri alınabilirlik
def test_pyarrow_yokken_csv_yolu_birebir_ayni(tmp_path, monkeypatch):
    """pyarrow import edilemiyorsa yan dosya KURULMAZ ve sonuç değişmez."""
    p = _yaz_csv(tmp_path)
    beklenen = _csv_yolu(p, COLS, ["AAPL"])
    gercek_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def _engelle(name, *a, **kw):
        if name.startswith("pyarrow"):
            raise ImportError("test: pyarrow yok")
        return gercek_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", _engelle)
    alinan = D.read_bulk(p, cols=COLS, symbols=["AAPL"], chunksize=500)
    pd.testing.assert_frame_equal(alinan, beklenen)
    assert not D._sidecar_path(p).exists()


def test_yarim_yazim_yayimlanmaz(tmp_path):
    """Parçalar arası dtype çatışması → ne yan dosya ne geçici dosya kalır; CSV'ye düşülür.

    Kurgu değil GERÇEK arıza yolu: bir kolon ilk parçada tamamen boşsa pandas onu
    float64 çıkarır, sonraki parçada metin gelirse cast patlar. Yan dosya
    mekanizmasının var olma nedeni tam olarak budur.
    """
    p = tmp_path / "volatility_history__master.csv"
    rows = [{"date": "2024-01-01", "act_symbol": "AAPL", "iv_current": 0.2,
             "hv_current": 0.3, "hv_month_ago": None} for _ in range(500)]
    rows += [{"date": "2024-02-01", "act_symbol": "AAPL", "iv_current": 0.2,
              "hv_current": 0.3, "hv_month_ago": "yok"} for _ in range(200)]
    pd.DataFrame(rows).to_csv(p, index=False)

    alinan = D.read_bulk(p, cols=COLS, symbols=["AAPL"], chunksize=500)
    pd.testing.assert_frame_equal(alinan, _csv_yolu(p, COLS, ["AAPL"]))
    assert not D._sidecar_path(p).exists()
    assert not list(tmp_path.glob("*.yaziliyor-*")), "geçici dosya sızdı"


def test_bozuk_yan_dosya_csv_ye_duser(tmp_path):
    p = _yaz_csv(tmp_path)
    side = D._sidecar_path(p)
    side.write_bytes(b"bu bir parquet degil")
    os.utime(side, (p.stat().st_atime + 10, p.stat().st_mtime + 10))   # "taze" görünsün
    pd.testing.assert_frame_equal(D.read_bulk(p, cols=COLS, symbols=["AAPL"], chunksize=500),
                                  _csv_yolu(p, COLS, ["AAPL"]))


# --------------------------------------------------------------------------- 3. tazelik
def test_csv_tazelendiginde_yan_dosya_yeniden_kurulur(tmp_path):
    p = _yaz_csv(tmp_path, satir=400)
    D.read_bulk(p, cols=COLS, chunksize=500)
    side = D._sidecar_path(p)
    assert side.exists()

    # bulk_csv yeni bir kopya indirdi: CSV değişti ve mtime ilerledi
    yeni = _yaz_csv(tmp_path, satir=800)
    assert yeni == p
    os.utime(p, (side.stat().st_atime + 100, side.stat().st_mtime + 100))

    alinan = D.read_bulk(p, cols=COLS, chunksize=500)
    assert len(alinan) == 800, "bayat yan dosya eski veriyi döndürdü"
    pd.testing.assert_frame_equal(alinan, _csv_yolu(p, COLS, None))


# --------------------------------------------------------------------------- 4. bağlandı mı
def test_uc_cagiran_da_kopya_dongu_tasimiyor():
    """Aynı desenin dördüncü kopyası doğmasın: çağıranlar `read_bulk` kullanmalı."""
    from pathlib import Path
    kok = Path(__file__).resolve().parent.parent
    for rel in ("research/edge_lab/build_vrp_dataset.py", "research/iv_rank_threshold_test.py"):
        src = (kok / rel).read_text(encoding="utf-8")
        assert "read_bulk(" in src, f"{rel}: tek kaynağa bağlanmamış"
        assert "chunksize=500_000" not in src, f"{rel}: kopya parça döngüsü hâlâ duruyor"
