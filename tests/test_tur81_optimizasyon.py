"""Tur-81: ÜRETİMDE ölçülmüş hızlandırmaların DENKLİK ve BAĞLANTI kilitleri.

Bu turun bulguları canlı süreçten çıktı (kıyastan değil — Tur-80 dersi): `/api/predictions/
accuracy` 43,5 sn, `/api/quant/tda` ~25 sn, `/api/quant/curve` 8,5 sn. Her düzeltme ESKİ
yolun çıktısını BİREBİR vermek zorundadır; bu dosya eski yolu referans olarak taşır ve
yeniyi ona karşı ölçer (Tur-36: denklik varsayılmaz, ızgarada ölçülür).
"""
from __future__ import annotations

import ast
import csv
import tomllib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nihai.ai import quant_math as qm
from nihai.ai import tda as td
from nihai.data.data_provider import CachingProvider, DataProvider
from nihai.execution import prediction_tracker as pt

ROOT = Path(__file__).resolve().parents[1]


# ===========================================================================
# BULGU-81-1 — resolve() bar çekme fan-out'u (43,5 sn → havuz)
# ===========================================================================
class _SayanSaglayici:
    """Çağrıları (sym, iv, lookback) olarak sayar; deterministik çerçeve döner."""

    def __init__(self, bar_sayisi: int = 400, patlat: set[str] | None = None):
        self.calls: list[tuple[str, str, int]] = []
        self.bar_sayisi = bar_sayisi
        self.patlat = patlat or set()

    def get_bars(self, sym, iv, lookback_bars=400):
        self.calls.append((sym, iv, lookback_bars))
        if sym in self.patlat:
            raise RuntimeError("sağlayıcı arızası")
        idx = pd.date_range("2026-09-17 09:30", periods=self.bar_sayisi, freq="1min",
                            tz="US/Eastern")
        return pd.DataFrame({"close": np.linspace(100.0, 101.0, len(idx))}, index=idx)


def _defter(yol: Path, satirlar: list[dict]) -> None:
    with yol.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pt.FIELDS)
        w.writeheader()
        for s in satirlar:
            w.writerow({k: s.get(k, "") for k in pt.FIELDS})


def _satir(sym: str, bar_ts: str, *, horizon: str = "5", price: str = "100.0") -> dict:
    return {"logged_at": "2026-09-17T13:00:00", "symbol": sym, "interval": "1m",
            "bar_ts": bar_ts, "direction": "+", "action": "LONG", "confidence": "0.6",
            "prob_cal": "0.52", "price": price, "bar_close": price,
            "horizon_bars": horizon, "resolved_at": ""}


_BAR_TS = [f"2026-09-17 09:{dk:02d}:00-04:00" for dk in (30, 35, 40, 45, 50, 55)]


def _kur(tmp_path: Path, monkeypatch, semboller: list[str], **kw) -> Path:
    p = tmp_path / "predictions_log.csv"
    satirlar = [_satir(s, ts, **kw) for s in semboller for ts in _BAR_TS[:3]]
    _defter(p, satirlar)
    monkeypatch.setattr(pt, "PRED_LOG", p)
    return p


def test_havuz_ve_sirali_AYNI_sayaci_ve_AYNI_dosyayi_uretir(tmp_path, monkeypatch):
    """Tek değişken thread sayısı; sonuç ve yazılan defter birebir aynı olmalı."""
    semboller = [f"S{i:02d}" for i in range(12)]
    okunan: dict[int, tuple[dict, str]] = {}
    for workers in (1, 8):
        d = tmp_path / f"w{workers}"
        d.mkdir()
        p = d / "predictions_log.csv"
        _defter(p, [_satir(s, ts) for s in semboller for ts in _BAR_TS[:3]])
        monkeypatch.setattr(pt, "PRED_LOG", p)
        monkeypatch.setattr(pt, "RESOLVE_FETCH_WORKERS", workers)
        prov = _SayanSaglayici()
        res = pt.resolve(prov)
        # `resolved_at` saat damgası iki koşuda farklı olabilir → normalize et.
        govde = [
            [("<T>" if (k == "resolved_at" and v) else v) for k, v in zip(r.keys(), r.values(),
                                                                          strict=True)]
            for r in csv.DictReader(p.open(encoding="utf-8", newline=""))
        ]
        okunan[workers] = (res, repr(govde))
        assert len(prov.calls) == len(semboller), "anahtar başına TAM bir çağrı olmalı"

    assert okunan[1][0] == okunan[8][0], "sayaçlar ayrıştı"
    assert okunan[1][1] == okunan[8][1], "yazılan defter ayrıştı"


def test_havuz_anahtar_basina_TEK_cagri_yapar(tmp_path, monkeypatch):
    """Aynı sembolün üç bekleyen satırı TEK bar çekimi tetiklemeli (dedup korunur)."""
    _kur(tmp_path, monkeypatch, ["AAA", "BBB", "CCC"])
    monkeypatch.setattr(pt, "RESOLVE_FETCH_WORKERS", 8)
    prov = _SayanSaglayici()
    pt.resolve(prov)
    assert sorted(c[0] for c in prov.calls) == ["AAA", "BBB", "CCC"]
    assert all(c[1] == "1m" and c[2] == 400 for c in prov.calls)


def test_bozuk_horizon_satiri_ON_CEKIMI_TETIKLEMEZ(tmp_path, monkeypatch):
    """BULGU-80-1 dalı: `horizon_bars` ayrıştırılamayan satır eski kodda da fetch
    tetiklemiyordu. Ön-çekim süzgeci döngününkiyle BİREBİR aynı olmalı — aksi hâlde
    havuz, eski yolun HİÇ çekmediği bir sembol için ağa çıkardı."""
    p = tmp_path / "predictions_log.csv"
    _defter(p, [_satir("IYI", _BAR_TS[0]), _satir("BOZUK", _BAR_TS[0], horizon="abc")])
    monkeypatch.setattr(pt, "PRED_LOG", p)
    monkeypatch.setattr(pt, "RESOLVE_FETCH_WORKERS", 8)
    prov = _SayanSaglayici()
    res = pt.resolve(prov)
    assert [c[0] for c in prov.calls] == ["IYI"], "bozuk satırın sembolü çekilmemeli"
    assert res["pending"] >= 1


def test_saglayici_ISTISNASI_havuzda_da_fail_soft(tmp_path, monkeypatch):
    """Bir anahtarın çekimi patlarsa YALNIZ o satırlar pending kalır; çözüm sürer."""
    _kur(tmp_path, monkeypatch, ["IYI", "OLU"])
    monkeypatch.setattr(pt, "RESOLVE_FETCH_WORKERS", 8)
    prov = _SayanSaglayici(patlat={"OLU"})
    res = pt.resolve(prov)
    assert res["resolved"] > 0, "sağlam sembol yine çözülmeli"
    assert res["pending"] >= 3, "patlayan sembolün satırları pending kalmalı"


def test_workers_env_ile_ayarlanir_ve_en_az_1():
    """`RESOLVE_FETCH_WORKERS=1` eski (sıralı) davranışa döndürür — geri alma yolu."""
    assert pt.RESOLVE_FETCH_WORKERS >= 1
    kaynak = (ROOT / "nihai" / "execution" / "prediction_tracker.py").read_text(encoding="utf-8")
    assert 'os.getenv("RESOLVE_FETCH_WORKERS"' in kaynak


def test_resolve_govdesinde_HAVUZ_GERCEKTEN_kurulu(tmp_path):
    """Bağlandı mı? (Tur-37 deseni) — `ThreadPoolExecutor` `resolve` gövdesinde olmalı."""
    agac = ast.parse((ROOT / "nihai" / "execution" / "prediction_tracker.py").read_text(
        encoding="utf-8"))
    fn = next(n for n in agac.body if isinstance(n, ast.FunctionDef) and n.name == "resolve")
    adlar = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "ThreadPoolExecutor" in adlar
    assert "RESOLVE_FETCH_WORKERS" in adlar


# ===========================================================================
# BULGU-81-2 — CachingProvider ÜST-KÜME yeniden kullanımı
# ===========================================================================
class _Sayac(DataProvider):
    name = "sayac"

    def __init__(self, toplam: int = 5000):
        self.calls: list[tuple[str, str, int]] = []
        idx = pd.date_range("2026-01-01", periods=toplam, freq="1min")
        self._full = pd.DataFrame({"open": np.arange(toplam, dtype=float),
                                   "high": np.arange(toplam, dtype=float),
                                   "low": np.arange(toplam, dtype=float),
                                   "close": np.arange(toplam, dtype=float),
                                   "volume": np.arange(toplam, dtype=float)}, index=idx)

    def get_bars(self, symbol, interval, lookback_bars=200):
        self.calls.append((symbol, interval, lookback_bars))
        return self._full.tail(lookback_bars).copy()

    def get_quote(self, symbol):
        return None

    def get_option_chain(self, symbol, expiry=None):
        return {}


def test_buyuk_sonra_kucuk_UPSTREAM_ikinci_kez_cagrilmaz():
    ic = _Sayac()
    cp = CachingProvider(ic, ttl_sec=45.0)
    buyuk = cp.get_bars("AAPL", "1m", 400)
    kucuk = cp.get_bars("AAPL", "1m", 220)
    assert [c[2] for c in ic.calls] == [400], "220 isteği üst kümeden karşılanmalı"
    assert len(kucuk) == 220
    pd.testing.assert_frame_equal(kucuk, buyuk.tail(220), check_exact=True)


def test_ust_kume_sonucu_ONBELLEKSIZ_cagriyla_BIREBIR():
    """Izgara: canlı karışımın her adımı, taze bir çağrının döndüreceğiyle aynı olmalı."""
    karisim = [220, 60, 400, 220, 60, 220, 400, 220, 60, 400]
    ic = _Sayac()
    cp = CachingProvider(ic, ttl_sec=45.0)
    alinan = [cp.get_bars("AAPL", "1m", n) for n in karisim]
    ref_ic = _Sayac()
    for got, n in zip(alinan, karisim, strict=True):
        pd.testing.assert_frame_equal(got, ref_ic.get_bars("AAPL", "1m", n), check_exact=True)
    assert len(ic.calls) == 2, f"canlı karışımda 2 upstream çağrı beklenir, {len(ic.calls)}"


def test_kucuk_sonra_buyuk_UPSTREAM_yeniden_cagrilir():
    """Üst küme İDDİASI tek yönlüdür: 220'lik girdi 400'lük isteği KARŞILAYAMAZ."""
    ic = _Sayac()
    cp = CachingProvider(ic, ttl_sec=45.0)
    cp.get_bars("AAPL", "1m", 220)
    buyuk = cp.get_bars("AAPL", "1m", 400)
    assert [c[2] for c in ic.calls] == [220, 400]
    assert len(buyuk) == 400


def test_kucuk_istek_girdiyi_DARALTMAZ():
    """Yarışta küçük istek büyüğün üstüne yazarsa sonraki büyük istek gereksiz yere ıskalar."""
    ic = _Sayac()
    cp = CachingProvider(ic, ttl_sec=45.0)
    cp.get_bars("AAPL", "1m", 400)
    cp.get_bars("AAPL", "1m", 60)
    cp.get_bars("AAPL", "1m", 400)
    assert [c[2] for c in ic.calls] == [400], "girdi daraltıldı → 400 yeniden çekildi"


def test_daraltma_korumasi_YARISTA_calisir():
    """Koruma YALNIZ yarışta ateşlenir ve bunu bir KAÇAN MUTASYON gösterdi (Tur-73):
    tek thread'de küçük istek zaten önbellekten karşılanır ve SAKLAMA yoluna hiç ulaşmaz,
    yani `test_kucuk_istek_girdiyi_DARALTMAZ` kurgusu dejenereydi. Gerçek senaryo: upstream
    çağrısı KİLİT DIŞINDA koşar; tam o sırada başka bir thread daha GENİŞ bir girdi yazar.
    Araya giren yazıcı burada deterministik olarak simüle edilir."""
    import time as _t

    ic = _Sayac()
    cp = CachingProvider(ic, ttl_sec=45.0)
    orijinal = ic.get_bars

    def araya_gir(symbol, interval, lookback_bars=200):
        df = orijinal(symbol, interval, lookback_bars)
        if lookback_bars == 60:                       # yalnız KÜÇÜK istek koşarken
            with cp._lock:
                cp._cache[("bars", symbol, interval)] = (
                    _t.monotonic(), ic._full.tail(400).copy(), 400)
        return df

    ic.get_bars = araya_gir                           # type: ignore[method-assign]
    kucuk = cp.get_bars("AAPL", "1m", 60)             # ıska → upstream → araya giren 400 yazar
    assert len(kucuk) == 60
    buyuk = cp.get_bars("AAPL", "1m", 400)            # daraltılmadıysa İSABET
    assert [c[2] for c in ic.calls] == [60], "60'lık istek 400'lük girdiyi DARALTTI"
    assert len(buyuk) == 400


def test_farkli_interval_ve_sembol_KARISMAZ():
    ic = _Sayac()
    cp = CachingProvider(ic, ttl_sec=45.0)
    cp.get_bars("AAPL", "1m", 400)
    cp.get_bars("AAPL", "1h", 60)
    cp.get_bars("MSFT", "1m", 60)
    assert [(c[0], c[1]) for c in ic.calls] == [("AAPL", "1m"), ("AAPL", "1h"), ("MSFT", "1m")]


def test_ttl_sifirda_onbellek_TAMAMEN_devre_disi():
    ic = _Sayac()
    cp = CachingProvider(ic, ttl_sec=0.0)
    cp.get_bars("AAPL", "1m", 400)
    cp.get_bars("AAPL", "1m", 220)
    assert [c[2] for c in ic.calls] == [400, 220]


def test_ttl_dolunca_yeniden_cekilir(monkeypatch):
    import time as _t
    saat = {"t": 1000.0}
    monkeypatch.setattr(_t, "monotonic", lambda: saat["t"])
    ic = _Sayac()
    cp = CachingProvider(ic, ttl_sec=45.0)
    cp.get_bars("AAPL", "1m", 400)
    saat["t"] += 10.0
    cp.get_bars("AAPL", "1m", 220)
    saat["t"] += 40.0                      # toplam 50 sn > 45 sn TTL
    cp.get_bars("AAPL", "1m", 220)
    assert [c[2] for c in ic.calls] == [400, 220]


def test_bos_sonuc_ONBELLEGE_girmez():
    class _Bos(_Sayac):
        def get_bars(self, symbol, interval, lookback_bars=200):
            self.calls.append((symbol, interval, lookback_bars))
            return pd.DataFrame()

    ic = _Bos()
    cp = CachingProvider(ic, ttl_sec=45.0)
    cp.get_bars("AAPL", "1m", 400)
    cp.get_bars("AAPL", "1m", 400)
    assert len(ic.calls) == 2, "geçici hata TTL boyunca kilitlenmemeli (_is_cacheable)"


# ===========================================================================
# BULGU-81-3 — lint kapısı `nihai/data/`yi HİÇ görmüyordu
# ===========================================================================
def test_ruff_exclude_desenleri_PROJE_KOKUNE_cipalanmis():
    """`exclude=["data"]` gitignore biçiminde HER derinlikteki `data/`yi eler → `nihai/data/`
    (29 modül: tüm sağlayıcılar, hub_server, EDGAR/FRED köprüleri) lint kapısının DIŞINDAYDI.
    Kural MAKİNEYE çevrildi: her desen `/` ile başlamalı (yalnız kök eşleşir)."""
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    desenler = cfg["tool"]["ruff"]["exclude"]
    assert desenler, "exclude listesi boş olamaz"
    for d in desenler:
        assert d.startswith("/"), f"çıpasız desen her derinlikte eler: {d!r}"


def test_hicbir_uretim_paketi_lint_disinda_degil():
    """Çıpalı desenler `nihai/`, `research/`, `tools/`, `tests/` altındaki hiçbir dizini
    elememelidir (kuralın VARLIK denetimi — Tur-50: biçim değil varlık)."""
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    elenen = {d.lstrip("/").rstrip("/") for d in cfg["tool"]["ruff"]["exclude"]}
    for paket in ("nihai", "research", "tools", "tests"):
        kok = ROOT / paket
        if not kok.is_dir():
            continue
        for alt in kok.rglob("*"):
            if alt.is_dir() and alt.name != "__pycache__":
                bagil = alt.relative_to(ROOT).as_posix()
                assert bagil not in elenen, f"{bagil} lint dışında"


def test_nihai_data_paketi_gercekten_var_ve_kalabalik():
    """Kapının koruduğu şeyin BOŞ olmadığını ölç (dersin öznesi 29 modüldü)."""
    moduller = list((ROOT / "nihai" / "data").glob("*.py"))
    assert len(moduller) >= 25, f"nihai/data beklenenden küçük: {len(moduller)}"


# ===========================================================================
# BULGU-81-5 — NSS tasarım matrisi önbelleği (`/api/quant/curve`)
# ===========================================================================
_VADELER = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
_GETIRI = np.array([5.30, 5.25, 5.10, 4.80, 4.35, 4.15, 4.05, 4.10, 4.25, 4.70, 4.90])


def _nss_dict_esit(a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return a is b
    if set(a) != set(b):
        return False
    for k, va in a.items():
        vb = b[k]
        if isinstance(va, list):
            if len(va) != len(vb) or any(x != y for x, y in zip(va, vb, strict=True)):
                return False
        elif va != vb:
            return False
    return True


def test_nss_onbellekli_ve_onbelleksiz_BIT_DENK():
    qm._nss_design.cache_clear()
    rng = np.random.default_rng(3)
    for _ in range(25):
        y = _GETIRI + rng.normal(0, 0.15, _GETIRI.size)
        qm._nss_design.cache_clear()
        soguk = qm.nss_fit(_VADELER, y)
        sicak = qm.nss_fit(_VADELER, y)           # önbellek dolu
        assert _nss_dict_esit(soguk, sicak)


def test_nss_onbellek_ISABET_ediyor():
    qm._nss_design.cache_clear()
    qm.nss_fit(_VADELER, _GETIRI)
    ilk = qm._nss_design.cache_info()
    qm.nss_fit(_VADELER, _GETIRI * 1.01)
    ikinci = qm._nss_design.cache_info()
    assert ikinci.misses == ilk.misses, "ikinci fit yeni matris kurmamalı"
    assert ikinci.hits > ilk.hits, "ikinci fit önbellekten okumalı"


def test_nss_FARKLI_vadelerde_matris_KARISMAZ():
    """Anahtar vade baytlarını taşımazsa iki farklı eğri aynı tasarımı paylaşır (sessiz
    yanlış sayı). Mutasyon kapısı: kısa vadeli ızgara ile uzun vadeli ızgara ayrışmalı."""
    qm._nss_design.cache_clear()
    kisa = np.array([0.25, 0.5, 1.0, 2.0, 3.0])
    uzun = np.array([5.0, 7.0, 10.0, 20.0, 30.0])
    y = np.array([4.0, 4.1, 4.2, 4.3, 4.4])
    a = qm.nss_fit(kisa, y)
    b = qm.nss_fit(uzun, y)
    assert a is not None and b is not None
    assert a["fit"] == pytest.approx(y.tolist(), abs=1e-3)
    assert not _nss_dict_esit(a, b), "farklı vade vektörleri aynı sonucu veremez"


def test_nss_tasarim_matrisi_SALT_OKUNUR():
    d = qm._nss_design(_VADELER.tobytes(), 2.0, 5.0)
    assert d.shape == (_VADELER.size, 4)
    with pytest.raises(ValueError):
        d[0, 0] = 1.0


# ===========================================================================
# BULGU-81-6 — Rips indirgemesi (liste pivot) + üçgen köşe önbelleği
# ===========================================================================
def _cember(n: int, yaricap: float = 1.0) -> np.ndarray:
    aci = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((yaricap * np.cos(aci), yaricap * np.sin(aci)))


def test_cemberde_TEK_uzun_omurlu_H1_cubugu():
    """Bilinen cevap: düzgün çemberin bir deliği vardır. İndirgeme döngüsü bozulursa
    (pivot nöbetçisi yanlış, liste indeksi kayık) bu çubuk kaybolur ya da çoğalır."""
    p = td.rips_persistence(_cember(24))
    assert p is not None
    h1 = np.asarray(p["h1"]).reshape(-1, 2)
    assert len(h1) >= 1
    omur = h1[:, 1] - h1[:, 0]
    en_uzun = float(np.max(omur))
    assert en_uzun > 1.0, f"çemberin deliği bulunamadı (en uzun ömür {en_uzun})"
    kalan = np.sort(omur)[:-1]
    assert kalan.size == 0 or float(np.max(kalan)) < en_uzun / 2.0


def test_sabit_bulutta_H1_YOK_ve_istisna_YOK():
    """Dejenere girdi (tüm noktalar aynı) — pivot listesi boyutu 0 kenara düşebilir."""
    p = td.rips_persistence(np.ones((8, 3)))
    assert p is not None
    assert np.asarray(p["h1"]).reshape(-1, 2).shape[0] == 0


def test_ucgen_koseleri_combinations_ile_BIREBIR():
    for n in (3, 7, 12, 40):
        beklenen = np.array(list(combinations(range(n), 3)), dtype=int)
        alinan = td._ucgen_koseleri(n)
        assert np.array_equal(alinan, beklenen)


def test_ucgen_koseleri_SALT_OKUNUR_ve_tavan_uzeri_ONBELLEGE_girmez():
    v = td._ucgen_koseleri(10)
    with pytest.raises(ValueError):
        v[0, 0] = 99
    assert td._TRI_CACHE_MAX_N >= 50, "kayan pencere varsayılanı (50) tavanın altında kalmalı"
    kaynak = (ROOT / "nihai" / "ai" / "tda.py").read_text(encoding="utf-8")
    assert "_TRI_CACHE_MAX_N" in kaynak and "n <= _TRI_CACHE_MAX_N" in kaynak


def test_tavan_USTUNDE_ve_ALTINDA_ayni_sonuc():
    """Tavanın iki yanındaki yol AYNI üçgen kümesini kurmalı (şubeleşme sessiz olmamalı)."""
    rng = np.random.default_rng(11)
    bulut = rng.normal(0, 1.0, (20, 3))
    a = td.rips_persistence(bulut)
    eski_tavan = td._TRI_CACHE_MAX_N
    try:
        td._TRI_CACHE_MAX_N = 3          # 20 > 3 → önbelleksiz dal
        b = td.rips_persistence(bulut)
    finally:
        td._TRI_CACHE_MAX_N = eski_tavan
    assert a is not None and b is not None
    assert np.array_equal(np.asarray(a["h1"]), np.asarray(b["h1"]))
    assert np.array_equal(np.asarray(a["h0"]), np.asarray(b["h0"]))
    assert a["n_triangles"] == b["n_triangles"]


def test_pivot_SOZLUK_olarak_geri_gelmemeli():
    """Regresyon kapısı — `ast` üzerinde (Tur-49: kapı ANLATIYA değil KODA bakar)."""
    agac = ast.parse((ROOT / "nihai" / "ai" / "tda.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(agac)
              if isinstance(n, ast.FunctionDef) and n.name == "rips_persistence")
    cagrilar = {n.func.attr for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "get" not in cagrilar, "pivot araması yeniden sözlüğe döndü"


# ===========================================================================
# Codex triyajı madde B — `annual_frame` aynı istekte iki kez
# ===========================================================================
def _edgar_ham(ef) -> pd.DataFrame:
    """`facts` şemasına uygun sentetik ham kayıt — kalem adları ve etiketler MODÜLÜN
    KENDİ `CONCEPTS` tablosundan okunur (elle yazılan ad, tabloyla sessizce ayrışırdı)."""
    donemler = [("2023-09-30", "2022-10-01", "2023-11-03"),
                ("2024-09-28", "2023-10-01", "2024-11-01"),
                ("2025-09-27", "2024-09-29", "2025-10-31")]
    satirlar = []
    for i, (kalem, zincir) in enumerate(list(ef.CONCEPTS.items())[:6]):
        for j, (bitis, baslangic, beyan) in enumerate(donemler):
            satirlar.append({
                "kalem": kalem, "etiket": zincir[0], "end": bitis,
                "start": (bitis if kalem in ef.ANLIK else baslangic),
                "val": 1000.0 * (i + 1) + 10.0 * j, "filed": beyan,
                "form": "10-K", "fy": 2023 + j, "fp": "FY",
            })
    d = pd.DataFrame(satirlar)
    for k in ("end", "start", "filed"):
        d[k] = pd.to_datetime(d[k])
    d.attrs["kaynak"] = "kurgu"
    return d


def test_son_iki_yil_annual_parametresi_AYNI_sonucu_verir():
    from nihai.data import edgar_facts as ef

    ham = _edgar_ham(ef)
    yillik = ef.annual_frame(ham, fetch=False)
    assert not yillik.empty, "kurgu ham kayıt yıllık çerçeve üretmeli (şema kaymış olabilir)"
    ikili_eski = ef.son_iki_yil(ham, fetch=False)
    ikili_yeni = ef.son_iki_yil(ham, fetch=False, annual=yillik)
    assert ikili_eski == ikili_yeni
    assert ikili_yeni[0].get("_donem_sonu") == "2025-09-27", "cari dönem yanlış"


def test_fundamentals_ucu_annual_parametresini_GERCEKTEN_geciriyor():
    """Bağlandı mı? — yazılıp çağrılmayan parametre bu depoda kusurdur (Tur-38/42)."""
    agac = ast.parse((ROOT / "nihai" / "web" / "api_quant.py").read_text(encoding="utf-8"))
    bulundu = False
    for n in ast.walk(agac):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_guard"
                and n.args and isinstance(n.args[0], ast.Attribute)
                and n.args[0].attr == "son_iki_yil"):
            bulundu = any(kw.arg == "annual" for kw in n.keywords)
    assert bulundu, "/api/quant/fundamentals hazır çerçeveyi geçirmiyor"


# ===========================================================================
# Codex triyajı madde C — /api/logs kuadratik tampon
# ===========================================================================
def _eski_tail(raw: bytes) -> list[str]:
    """Tur-80'e kadarki gövde — REFERANS."""
    import io
    f = io.BytesIO(raw)
    f.seek(0, 2)
    pointer = f.tell()
    buffer = bytearray()
    lines_list: list[bytes] = []
    while pointer > 0 and len(lines_list) <= 150:
        step = min(4096, pointer)
        pointer -= step
        f.seek(pointer)
        buffer = f.read(step) + buffer
        lines_list = buffer.split(b"\n")
    cozulen = [ln.decode("utf-8", errors="ignore").strip() for ln in lines_list]
    return [ln for ln in cozulen if ln][-100:]


@pytest.mark.parametrize("kurgu", ["kisa", "uzun", "son_satirsiz", "bozuk_utf8", "bos_satirli"])
def test_log_kuyrugu_ESKI_govdeyle_BIREBIR(kurgu):
    if kurgu == "kisa":
        raw = b"\n".join(f"satir {i} ".encode() + b"y" * 40 for i in range(6000)) + b"\n"
    elif kurgu == "uzun":
        raw = b"\n".join((f"{i}-" + "x" * 900).encode() for i in range(800)) + b"\n"
    elif kurgu == "son_satirsiz":
        raw = b"\n".join(f"satir {i} ".encode() + b"y" * 40 for i in range(6000))
    elif kurgu == "bozuk_utf8":
        raw = b"\n".join(f"satir {i} ".encode() + b"\xff\xfe" * 20 for i in range(6000)) + b"\n"
    else:
        raw = b"\n".join((b"" if i % 3 else f"satir {i} ".encode() + b"z" * 60)
                         for i in range(12000)) + b"\n"
    assert len(raw) > 256 * 1024

    # ÜRETİM gövdesi koşulur, kopyası DEĞİL: kaçan bir mutasyon (durma koşulu off-by-one)
    # testin kendi kopyasını değiştirmediği için yeşil kalıyordu (Tur-73: kaçan mutasyon
    # ya eksik test ya ölü guard ya dejenere kurgu der — buradaki BİRİNCİSİYDİ).
    from nihai.web import web_dashboard as wd

    yol = wd.LOG_DIR / "system.log"
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_bytes(raw)
    try:
        assert wd.get_logs() == _eski_tail(raw)
    finally:
        yol.unlink(missing_ok=True)


def test_log_kuyrugu_150_esigi_PAY_degil_SINIRDIR():
    """İlk mutasyon koşusunda `newlines < 150` → `< 149` KAÇTI ve ilk okuma *"150 bir
    PAYDIR, çıktıyı değiştirmez"* oldu. Öncül ÖLÇÜLDÜ, varsayılmadı: 56 elle kurgulanmış
    ve rastgele girdide arandı ve fark ÜRETEN bir sınıf bulundu — BOŞ satır yoğun loglar.
    Orada 150 satır sonu okumak 87, 149 okumak 83 satır bırakır (`[-100:]` ikisini de
    kırpmaz). Yani eşik bir sözleşmedir ve pinlenir. Üretim gövdesi koşulur."""
    import random

    r = random.Random(0)
    parcalar: list[bytes] = []
    while sum(len(p) for p in parcalar) < 300 * 1024:
        parcalar.append(b"\n" if r.random() < 0.45
                        else b"L" * r.randrange(1, 4200) + b"\n")
    raw = b"".join(parcalar)

    from nihai.web import web_dashboard as wd

    yol = wd.LOG_DIR / "system.log"
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_bytes(raw)
    try:
        alinan = wd.get_logs()
    finally:
        yol.unlink(missing_ok=True)
    assert alinan == _eski_tail(raw)
    assert len(alinan) == 87, f"150 eşiği kaymış olabilir (dönen satır {len(alinan)})"


def test_logs_govdesinde_KUADRATIK_desen_geri_gelmemeli():
    kaynak = (ROOT / "nihai" / "web" / "web_dashboard.py").read_text(encoding="utf-8")
    agac = ast.parse(kaynak)
    fn = next((n for n in ast.walk(agac)
               if isinstance(n, ast.FunctionDef) and n.name == "get_logs"), None)
    assert fn is not None, "get_logs bulunamadı"
    govde = ast.dump(fn)
    assert "chunk" in govde and "join" in govde, "parça listesi + tek birleştirme kalkmış"
