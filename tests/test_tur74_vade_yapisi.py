"""Tur-74 — CBOE vade yapısı KARAR-DIŞI yüzey olarak bağlandı (dış liste Madde 9).

Üç soru sorulur:

1. **BAĞLANDI MI?** `cboe_data` Tur-57'den beri depodaydı ve `nihai/` altında TEK bir
   üretim çağıranı yoktu (ölü köprü — Tur-18/Tur-37/Tur-61'in tekrar eden sınıfı).
   Artık `/api/quant/regime` onu okuyor ve pano basıyor.
2. **ANAYASA KORUNDU MU?** Madde 9 "backwardation tüm swing alımlarını KİLİTLESİN"
   diyordu; bu bir kenar iddiasıdır ve ölçülmemiş bileşen karara giremez (ACI DERS #5).
   Vade yapısı hiçbir karar/sinyal yolundan okunmamalıdır.
3. **UÇ AĞA ÇIKIYOR MU?** Çıkmamalı (BULGU-59-8: bir okuma yoluna saniyelik ağ çağrısı
   koymak, ölçüm bütçesini okuyana ödetmenin ta kendisidir).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nihai.data import cboe_data
from nihai.web import api_quant

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1) ÖNBELLEK-YALNIZ SÖZLEŞMESİ — uç ağa çıkamaz
# ---------------------------------------------------------------------------
def test_fetch_false_AGA_CIKMAZ_onbellek_yoksa_bos_doner(tmp_path, monkeypatch):
    """`fetch=False` ile ağ çağrısı DENENMEZ; önbellek yoksa dürüstçe boş df."""
    monkeypatch.setattr(cboe_data, "CBOE_CACHE_DIR", tmp_path / "yok")

    # BAYRAK ile ölç, İSTİSNA ile DEĞİL: `get_index` fail-soft'tur ve geniş bir
    # `except Exception` istisnayı YUTAR → istisna atan bir sahte, guard silinse bile
    # testi YEŞİL bırakırdı (bu tam olarak ölçüldü: mutasyon KAÇTI).
    denendi: list[str] = []

    def _kaydet(*a, **k):
        denendi.append(str(a[0]) if a else "?")
        raise OSError("ağ yok")

    monkeypatch.setattr(cboe_data.pd, "read_csv", _kaydet)
    assert cboe_data.get_index("VIX", fetch=False).empty
    assert not denendi, f"fetch=False iken ağ çağrısı DENENDİ: {denendi}"


def test_fetch_false_BAYAT_onbellegi_KABUL_eder(tmp_path, monkeypatch):
    """Bayat önbellek > veri yok. Aksi halde uç her gün BOŞ kalırdı; bayatlık
    gizlenmez, `_vade_yapisi` onu `bayat_gun` ile RAPORLAR."""
    import os
    import time

    cdir = tmp_path / "cboe"
    cdir.mkdir()
    p = cdir / "VIX.csv"
    p.write_text("date,close\n2020-01-02,14.0\n2020-01-03,13.5\n", encoding="utf-8")
    eski = time.time() - 90 * 86400          # 90 gün önce yazılmış
    os.utime(p, (eski, eski))
    monkeypatch.setattr(cboe_data, "CBOE_CACHE_DIR", cdir)

    assert cboe_data._read_cache("VIX") is None                    # yaş kapısı reddeder
    df = cboe_data.get_index("VIX", fetch=False)
    assert not df.empty and len(df) == 2                           # fetch=False KABUL eder


def test_uc_TERM_STRUCTURE_i_fetch_False_ile_cagiriyor():
    """Kaynak denetimi: uç `term_structure`ı ÖNBELLEK-YALNIZ çağırmalı. Bu bir yorum
    değil bir sözleşmedir — `fetch=False` düşerse uç HTTP isteği içinde ağa çıkar."""
    src = (REPO / "nihai" / "web" / "api_quant.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_vade_yapisi")
    cagri = [c for c in ast.walk(fn)
             if isinstance(c, ast.Call)
             and any(isinstance(a, ast.Attribute) and a.attr == "term_structure"
                     for a in ast.walk(c))]
    assert cagri, "_vade_yapisi term_structure'ı hiç çağırmıyor"
    assert any(k.arg == "fetch" and k.value.value is False
               for c in cagri for k in c.keywords), "fetch=False kaybolmuş → uç ağa çıkar"


# ---------------------------------------------------------------------------
# 2) ANAYASA — vade yapısı KARAR yolundan okunamaz
# ---------------------------------------------------------------------------
def test_cboe_data_KARAR_yollarindan_import_EDILMIYOR():
    """Madde 9'un önerdiği kapı BİLEREK kurulmadı. `signals`/`risk`/`execution` altında
    `cboe_data` importu belirirse, ölçülmemiş bir bileşen karara girmiş demektir."""
    suclu = []
    for alt in ("signals", "risk", "execution"):
        for f in (REPO / "nihai" / alt).rglob("*.py"):
            src = f.read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.ImportFrom) and "cboe_data" in (node.module or ""):
                    suclu.append(str(f.relative_to(REPO)))
                elif isinstance(node, ast.Import):
                    if any("cboe_data" in a.name for a in node.names):
                        suclu.append(str(f.relative_to(REPO)))
    assert not suclu, (
        f"cboe_data KARAR yoluna sızmış: {suclu}. Vade yapısı ölçülmemiş bir rejim "
        "hipotezidir; kapı yapmak için önce research/edge_lab kanıt kapısı gerekir.")


@pytest.fixture()
def sahte_ts(monkeypatch):
    """Gerçekçi bir vade yapısı paneli — ATLANAN test hiçbir şey kilitlemez."""
    idx = pd.date_range("2026-01-01", periods=60, freq="B")
    wide = pd.DataFrame({"VIX": [14.0 + (i % 5) for i in range(60)],
                         "VIX3M": [25.0] * 60, "VIX9D": [13.0] * 60,
                         "VVIX": [85.0] * 60, "SKEW": [150.0] * 60}, index=idx)
    monkeypatch.setattr(cboe_data, "get_all", lambda *a, **k: wide)
    return wide


def test_blok_KARAR_DISI_bayragini_tasiyor(sahte_ts):
    d = api_quant._vade_yapisi(history=3)
    assert d is not None and d["karar_disi"] is True
    # VIX3M sabit 25, VIX 14-18 → ts_slope HER barda >1 = kesin contango (belirsizlik yok)
    assert d["backwardation"] is False
    assert d["ts_slope"] > 1.0


# ---------------------------------------------------------------------------
# 3) ÖLÇÜM DOĞRULUĞU — bilinen özdeşlikler
# ---------------------------------------------------------------------------
def test_ts_slope_ozdeslik_VIX3M_bolu_VIX(monkeypatch):
    """ts_slope tanımı VIX3M/VIX'tir ve backwardation SADECE <1'de doğrudur.
    İşareti ters çevirmek stresi sakinlik diye okumak olurdu."""
    idx = pd.to_datetime(["2026-01-02", "2026-01-05"])
    wide = pd.DataFrame({"VIX": [20.0, 30.0], "VIX3M": [24.0, 27.0],
                         "VIX9D": [16.0, 33.0], "VVIX": [90.0, 120.0],
                         "SKEW": [140.0, 150.0]}, index=idx)
    monkeypatch.setattr(cboe_data, "get_all", lambda *a, **k: wide)
    ts = cboe_data.term_structure(fetch=False)
    assert ts["ts_slope"].tolist() == [24.0 / 20.0, 27.0 / 30.0]
    assert ts["backwardation"].tolist() == [0, 1]          # 1,2 contango · 0,9 backwardation
    assert ts["near_slope"].tolist() == [20.0 / 16.0, 30.0 / 33.0]


def test_vix_pctile_GENISLEYEN_pencere_look_ahead_yok(monkeypatch):
    """Yüzdelik yalnız GEÇMİŞE bakmalı: son satırın yüzdeliği, seriye SONRADAN
    eklenen bir gözlemden ETKİLENMEMELİ (aksi halde look-ahead)."""
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    base = pd.DataFrame({"VIX": [15.0 + (i % 7) for i in range(n)],
                         "VIX3M": [17.0] * n}, index=idx)
    monkeypatch.setattr(cboe_data, "get_all", lambda *a, **k: base)
    once = cboe_data.term_structure(fetch=False)["vix_pctile"].iloc[-1]

    uzun = pd.concat([base, pd.DataFrame({"VIX": [99.0], "VIX3M": [17.0]},
                                         index=pd.date_range("2025-06-01", periods=1, freq="B"))])
    monkeypatch.setattr(cboe_data, "get_all", lambda *a, **k: uzun)
    sonra = cboe_data.term_structure(fetch=False)["vix_pctile"].iloc[-2]
    assert once == sonra, "gelecekteki gözlem geçmişin yüzdeliğini değiştirdi = look-ahead"


def test_bayatlik_RAPORLANIYOR(sahte_ts):
    """Bayat önbelleği taze gibi sunmak Tur-31 kusurudur; yaş taşınmalı ve GERÇEK olmalı."""
    import datetime as _dt
    d = api_quant._vade_yapisi(history=3)
    assert d is not None
    beklenen = (_dt.date.today() - sahte_ts.index[-1].date()).days
    assert d["bayat_gun"] == beklenen, "bayatlık uydurulmuş ya da sabitlenmiş"


def test_uc_dusmez_vade_yapisi_olcumu_yoksa(monkeypatch):
    """Fail-soft: vade yapısı ölçülemezse `/api/quant/regime` 500 atmamalı, blok None olmalı."""
    monkeypatch.setattr(cboe_data, "term_structure", lambda **k: pd.DataFrame())
    assert api_quant._vade_yapisi() is None


# ---------------------------------------------------------------------------
# 4) UZUN UÇ (VIX6M) — Madde 9 bunu AYRICA istiyordu
# ---------------------------------------------------------------------------
def test_far_slope_ozdeslik_ve_FFILL_YOK(monkeypatch):
    """`far_slope = VIX6M/VIX`. CBOE VIX6M'i düzensiz yayımlar (ölçüldü: VIX 2026-09-04'e,
    VIX6M yalnız 2026-08-12'ye kadar) → EKSİK gün eksik KALIR. Son bilinen uzun ucu ileri
    taşımak, üç haftalık bir vade yapısını "bugünkü" diye sunmak olurdu (BULGU-51-2)."""
    idx = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    wide = pd.DataFrame({"VIX": [20.0, 25.0, 20.0], "VIX3M": [24.0, 24.0, 24.0],
                         "VIX9D": [18.0, 18.0, 18.0],
                         "VIX6M": [26.0, np.nan, 30.0],      # ORTA gün eksik
                         "VVIX": [90.0] * 3, "SKEW": [140.0] * 3}, index=idx)
    monkeypatch.setattr(cboe_data, "get_all", lambda *a, **k: wide)
    ts = cboe_data.term_structure(fetch=False)
    assert ts["far_slope"].iloc[0] == 26.0 / 20.0
    assert pd.isna(ts["far_slope"].iloc[1]), "eksik VIX6M SESSİZCE dolduruldu (ffill)"
    assert ts["far_slope"].iloc[2] == 30.0 / 20.0


def test_far_slope_VIX6M_yoksa_KOLON_HIC_URETILMEZ(monkeypatch):
    """Sağlayıcı seriyi hiç vermezse (404) kolon YOK; sıfır/NaN uydurulmaz."""
    idx = pd.to_datetime(["2026-01-02", "2026-01-05"])
    wide = pd.DataFrame({"VIX": [20.0, 22.0], "VIX3M": [24.0, 24.0],
                         "VIX9D": [18.0, 18.0]}, index=idx)
    monkeypatch.setattr(cboe_data, "get_all", lambda *a, **k: wide)
    assert "far_slope" not in cboe_data.term_structure(fetch=False).columns


def test_uc_far_slope_TARIHINI_ayrica_tasiyor(sahte_ts):
    """`far_slope` bugün ölçülemese bile SON GEÇERLİ günü raporlanmalı — yoksa operatör
    uzun ucun ne kadar bayat olduğunu bilemez."""
    d = api_quant._vade_yapisi()
    assert d is not None and "far_slope_tarih" in d


# ---------------------------------------------------------------------------
# 5) KANIT KAPISI AİLESİ (Madde 9'un KAPI iddiası) — tasarım değişmezleri
# ---------------------------------------------------------------------------
def test_kirilma_KENDI_barini_dislar():
    """20-bar en yüksek `shift(1)` ile alınmalı. Alınmazsa tanım kendine referans verir
    (close her zaman kendi dahil maksimumun içindedir) ve "kırılma" anlamsızlaşır."""
    from research.edge_lab import build_vixts_dataset as B

    src = (REPO / "research" / "edge_lab" / "build_vixts_dataset.py").read_text(encoding="utf-8")
    assert ".rolling(BREAKOUT_WINDOW).max().shift(1)" in src, "kırılma kendi barını DIŞLAMIYOR"
    assert B.BREAKOUT_WINDOW == 20 and B.ISINMA_GUN >= 60


def test_olcut_HAM_getiri_piyasaya_gore_DEGIL():
    """Madde 9 bir PİYASA ZAMANLAMASI iddiasıdır; piyasaya göre fazlalık almak ölçülmek
    istenen etkiyi TANIM GEREĞİ görünmez kılardı. `excess` HAM, `excess_duz` piyasaya göre."""
    src = (REPO / "research" / "edge_lab" / "build_vixts_dataset.py").read_text(encoding="utf-8")
    assert 'df["excess"] = df["fwd_ret"]' in src
    assert 'df["excess_duz"] = df["fwd_ret"] - df["mkt_ret"]' in src


def test_edge_lab_URETIM_SINYALINE_import_bagi_ALMAZ():
    """Paket docstring'inin sözleşmesi: `edge_lab` nihai.signals/strategy/execution'a
    bağlanmaz. Kırılma kuralı bu yüzden dosyanın KENDİSİNDE tanımlıdır."""
    src = (REPO / "research" / "edge_lab" / "build_vixts_dataset.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        mod = ""
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
        elif isinstance(node, ast.Import):
            mod = ",".join(a.name for a in node.names)
        assert not any(y in mod for y in ("nihai.signals", "nihai.execution", "nihai.risk")), \
            f"edge_lab üretim sinyaline bağlandı: {mod}"
