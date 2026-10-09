"""dolt_options testleri — TAMAMI ÇEVRİMDIŞI (_api_get monkeypatch'lenir).

Odak: sessizce yanlış veri üretebilecek yollar. Ağ davranışı burada test EDİLMEZ
(conftest ağ guard'ı zaten kapatır); test edilen şey istemci mantığıdır.
"""
from __future__ import annotations

import pytest

from nihai.data import dolt_options as do


def _payload(rows, status="Success", msg=""):
    return {"query_execution_status": status, "query_execution_message": msg, "rows": rows}


def _vh_row(date, sym="AAPL", iv="0.20", hi="0.40", lo="0.10"):
    return {"date": date, "act_symbol": sym, "iv_current": iv,
            "iv_year_high": hi, "iv_year_low": lo,
            "hv_current": "0.30", "hv_year_high": "0.5", "hv_year_low": "0.1"}


# --- AS OF / SQL kurulumu: PIT iddiasının doğruluğu buna bağlı ---------------

def test_as_of_clause_bicimleri():
    assert do._as_of("volatility_history", None) == "volatility_history"
    assert do._as_of("t", "HEAD~50") == "t AS OF 'HEAD~50'"
    assert do._as_of("t", "abc123") == "t AS OF 'abc123'"
    # TIMESTAMP(...) tırnak İÇİNE alınmamalı, yoksa string literal olur ve PIT çalışmaz
    ts = "TIMESTAMP('2024-01-02')"
    assert do._as_of("t", ts) == f"t AS OF {ts}"


def test_as_of_sorguya_gercekten_giriyor(monkeypatch):
    gorulen = []

    def sahte(sql, **kw):
        gorulen.append(sql)
        return _payload([])

    monkeypatch.setattr(do, "_api_get", sahte)
    do.volatility_history("AAPL", as_of="HEAD~50")
    assert "AS OF 'HEAD~50'" in gorulen[0]


def test_sembol_tirnak_enjeksiyonu_kirpilir():
    assert do._quote_sym("aa'pl") == "'AAPL'"


# --- Sessiz kırpma: bu projenin en pahalı hata sınıfı ------------------------

def test_kismi_sonuc_hata_ile_gelirse_KULLANILMAZ(monkeypatch):
    """Deadline'a takılan sorgu Error + kısmi satır döndürür. Kabul edilmemeli."""
    monkeypatch.setattr(do.time, "sleep", lambda *_: None)

    def sahte_get(sql, **kw):
        return _payload([_vh_row("2025-01-02")], status="Error",
                        msg="query error: context deadline exceeded")

    # gerçek _api_get'i kullan (retry/durum mantığı test ediliyor), HTTP'yi taklit et
    class SahteResp:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return sahte_get("")

    monkeypatch.setattr(do, "requests", type("R", (), {"get": staticmethod(lambda *a, **k: SahteResp)}))
    with pytest.raises(do.DoltQueryError):
        do.query("SELECT 1 FROM volatility_history", paginate=False)


def test_kalici_sql_hatasi_yeniden_denenmez(monkeypatch):
    cagri = {"n": 0}

    class SahteResp:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            cagri["n"] += 1
            return _payload([], status="Error", msg="column 'yok' not found")

    monkeypatch.setattr(do.time, "sleep", lambda *_: None)
    monkeypatch.setattr(do, "requests", type("R", (), {"get": staticmethod(lambda *a, **k: SahteResp)}))
    with pytest.raises(do.DoltQueryError):
        do.query("SELECT yok FROM t", paginate=False)
    assert cagri["n"] == 1, "kalıcı hata için boşuna yeniden denendi"


def test_hiz_siniri_uzun_bekler_ve_hizli_denemez(monkeypatch):
    uykular = []
    monkeypatch.setattr(do.time, "sleep", lambda s: uykular.append(s))

    class SahteResp:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return _payload([], status="Error", msg="query error: http status: 403")

    monkeypatch.setattr(do, "requests", type("R", (), {"get": staticmethod(lambda *a, **k: SahteResp)}))
    with pytest.raises(do.DoltQueryError):
        do.query("SELECT 1 FROM t", paginate=False)
    assert uykular, "hız sınırında hiç beklenmedi"
    assert max(uykular) >= do.RATE_LIMIT_SLEEP_SEC, "hız sınırında kısa bekleme = durumu kötüleştirir"


# --- Sayfalama: satır düşürmemeli, sonsuza dönmemeli ------------------------

def test_sayfalama_tum_satirlari_toplar(monkeypatch):
    do_page = do.PAGE_SIZE
    sayfa1 = [_vh_row(f"2025-01-{i:02d}") for i in range(1, do_page + 1)]  # tam sayfa
    sayfa2 = [_vh_row("2025-02-01"), _vh_row("2025-02-02")]                # kısa → son
    sayfalar = [sayfa1, sayfa2]

    def sahte(sql, **kw):
        return _payload(sayfalar.pop(0) if sayfalar else [])

    monkeypatch.setattr(do, "_api_get", sahte)
    df = do.query("SELECT * FROM volatility_history WHERE act_symbol='AAPL'")
    assert len(df) == do_page + 2


def test_kendi_LIMITini_yazan_sorgu_sayfalanmaz(monkeypatch):
    gorulen = []

    def sahte(sql, **kw):
        gorulen.append(sql)
        return _payload([_vh_row("2025-01-02")])

    monkeypatch.setattr(do, "_api_get", sahte)
    do.query("SELECT * FROM t LIMIT 5")
    assert len(gorulen) == 1
    assert "OFFSET" not in gorulen[0], "kullanıcının LIMIT'i ezildi"


# --- Tip dönüşümü ve IV rank matematiği ------------------------------------

def test_rows_to_df_tipleri_donusturur():
    df = do._rows_to_df([_vh_row("2025-01-02")])
    assert str(df["date"].dtype).startswith("datetime64")
    assert df["iv_current"].iloc[0] == pytest.approx(0.20)
    assert df["iv_year_high"].dtype.kind == "f"


def test_iv_rank_hesabi(monkeypatch):
    monkeypatch.setattr(do, "_api_get",
                        lambda sql, **kw: _payload([_vh_row("2025-01-02", iv="0.25",
                                                            hi="0.40", lo="0.10")]))
    df = do.iv_rank_series("AAPL")
    # (0.25-0.10)/(0.40-0.10)*100 = 50
    assert df["iv_rank"].iloc[0] == pytest.approx(50.0)


def test_iv_rank_aralik_cokunce_NaN_birakir(monkeypatch):
    """hi==lo iken 50.0 uydurmak, ölçüm yokluğunu ölçüm gibi göstermek olurdu."""
    monkeypatch.setattr(do, "_api_get",
                        lambda sql, **kw: _payload([_vh_row("2025-01-02", iv="0.25",
                                                            hi="0.20", lo="0.20")]))
    df = do.iv_rank_series("AAPL")
    assert df["iv_rank"].isna().all()


def test_cok_sembol_tek_tek_sorulur(monkeypatch):
    """Çok-sembollü IN(...) sunucuda zaman aşımına giriyor (docstring'deki ölçüm)."""
    gorulen = []

    def sahte(sql, **kw):
        gorulen.append(sql)
        return _payload([_vh_row("2025-01-02", sym="X")])

    monkeypatch.setattr(do, "_api_get", sahte)
    monkeypatch.setattr(do.time, "sleep", lambda *_: None)
    do.volatility_history(["AAPL", "MSFT", "SPY"])
    assert len(gorulen) == 3
    assert all(" IN (" not in s for s in gorulen)


# --- Toplu CSV yolu: asıl üretim yolu, SQL API'ye dokunmaz --------------------

_CSV_GOVDE = (
    b"date,act_symbol,hv_current,hv_year_high,hv_year_low,"
    b"iv_current,iv_year_high,iv_year_low\n"
    b"2025-01-02,AAPL,0.30,0.5,0.1,0.20,0.40,0.10\n"
    b"2025-01-02,MSFT,0.28,0.5,0.1,0.22,0.44,0.11\n"
    b"2025-01-03,AAPL,0.31,0.5,0.1,0.25,0.40,0.10\n"
)


class _SahteStream:
    """requests.get(..., stream=True) context manager taklidi."""

    def __init__(self, govde=_CSV_GOVDE, patlat=False):
        self._govde = govde
        self._patlat = patlat

    def __enter__(self):
        if self._patlat:
            raise OSError("ağ koptu")
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1):
        yield self._govde


def _sahte_requests(stream):
    return type("R", (), {"get": staticmethod(lambda *a, **k: stream)})


def test_bulk_csv_indirir_ve_yol_dondurur(monkeypatch, tmp_path):
    monkeypatch.setattr(do, "DOLT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(do, "requests", _sahte_requests(_SahteStream()))
    p = do.bulk_csv("volatility_history")
    assert p is not None and p.exists()
    assert p.read_bytes() == _CSV_GOVDE


def test_bulk_csv_taze_kopyayi_yeniden_indirmez(monkeypatch, tmp_path):
    monkeypatch.setattr(do, "DOLT_CACHE_DIR", tmp_path)
    hedef = tmp_path / "volatility_history__master.csv"
    hedef.write_bytes(b"eski")

    def patla(*a, **k):
        raise AssertionError("taze önbellek varken indirme yapıldı")

    monkeypatch.setattr(do, "requests", type("R", (), {"get": staticmethod(patla)}))
    p = do.bulk_csv("volatility_history", max_age_days=7)
    assert p == hedef and p.read_bytes() == b"eski"


def test_bulk_csv_yarim_indirmeyi_yayimlamaz(monkeypatch, tmp_path):
    """Ağ ortada koparsa hedef dosya ASLA yarım içerikle değişmemeli."""
    monkeypatch.setattr(do, "DOLT_CACHE_DIR", tmp_path)
    hedef = tmp_path / "volatility_history__master.csv"
    hedef.write_bytes(b"saglam-eski-kopya")
    monkeypatch.setattr(do, "requests", _sahte_requests(_SahteStream(patlat=True)))
    p = do.bulk_csv("volatility_history", force=True)
    assert hedef.read_bytes() == b"saglam-eski-kopya", "yarım/boş içerik yayımlandı"
    assert p == hedef, "bayat kopya varken None döndü (bayat kopya > kopya yok)"
    assert not list(tmp_path.glob("*.indiriliyor-*")), "geçici dosya temizlenmedi"


def test_goreli_ref_commit_hashine_cozulur(monkeypatch):
    """CSV uç noktası HEAD~50'yi 500 ile reddeder; hash'e çevrilmeli."""
    monkeypatch.setattr(do, "_api_get", lambda sql, **kw: _payload(
        [{"commit_hash": f"h{i}"} for i in range(51)]))
    assert do._resolve_ref("HEAD~50") == "h50"
    assert do._resolve_ref("HEAD") == "h0"
    assert do._resolve_ref("HEAD^^") == "h2"
    # hash/branch adı olduğu gibi geçer, boşuna sorgu yapılmaz
    monkeypatch.setattr(do, "_api_get", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("hash için dolt_log sorgulandı")))
    assert do._resolve_ref("abc123") == "abc123"
    assert do._resolve_ref("master") == "master"


def test_bulk_csv_goreli_refi_cozup_url_e_koyar(monkeypatch, tmp_path):
    monkeypatch.setattr(do, "DOLT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(do, "_resolve_ref", lambda r: "COZULMUS")
    gorulen = {}

    def sahte_get(url, **kw):
        gorulen["url"] = url
        return _SahteStream()

    monkeypatch.setattr(do, "requests", type("R", (), {"get": staticmethod(sahte_get)}))
    p = do.bulk_csv("volatility_history", ref="HEAD~50")
    assert "/COZULMUS/" in gorulen["url"], "göreli ref URL'e ham girdi (500 verirdi)"
    assert p is not None and "COZULMUS" in p.name, "PIT kopyası HEAD ile aynı dosyaya yazıldı"


def test_volatility_history_bulk_yalniz_istenen_sembolu_suzer(monkeypatch, tmp_path):
    monkeypatch.setattr(do, "DOLT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(do, "requests", _sahte_requests(_SahteStream()))

    def patla(*a, **k):
        raise AssertionError("toplu kipte SQL API'ye gidildi")

    monkeypatch.setattr(do, "_api_get", patla)
    df = do.volatility_history_bulk(["AAPL"])
    assert set(df["act_symbol"]) == {"AAPL"}
    assert len(df) == 2
    assert str(df["date"].dtype).startswith("datetime64")


def test_sunucu_siralamasi_PK_onegi_disina_cikmaz(monkeypatch):
    """ORDER BY act_symbol,date ölçüldü: 51.522 ms ve zaman aşımı. Bir daha girmesin."""
    gorulen = []
    monkeypatch.setattr(do, "_api_get", lambda sql, **kw: gorulen.append(sql) or _payload([]))
    monkeypatch.setattr(do.time, "sleep", lambda *_: None)
    do.volatility_history(["AAPL"])
    do.option_chain("AAPL", date="2025-01-02")
    for sql in gorulen:
        assert "ORDER BY act_symbol" not in sql
        if "ORDER BY" in sql:
            assert "ORDER BY date" in sql
