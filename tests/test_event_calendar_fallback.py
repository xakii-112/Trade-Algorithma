"""event_calendar İKİNCİ kaynak (Nasdaq) fallback + BMO/AMC + hata-yapışması testleri.

Tamamı AĞSIZ: HTTP taşıması monkeypatch'lenir, fixture'lar gerçek Nasdaq
/api/calendar/earnings yanıtından kırpılmıştır. event_calendar tarafı
_fetch_yfinance_events / _nasdaq_next_earnings monkeypatch'i ile birincil/ikincil
dalları ağ olmadan izole eder.

Kapsanan senaryolar (görev şartı):
(a) birincil OK → fallback HİÇ çağrılmaz
(b) birincil boş/hata → fallback doldurur
(c) ikisi de hata → zarif boş + kısa-TTL negatif cache + TTL dolunca yeniden deneme
(d) BMO/AMC/None normalizasyonu
(e) header'sız 403/asılı-kalma (timeout) senaryosu → zarif None
(f) modül-içi tarih cache süre mantığı
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest

import nihai.data.event_calendar as ec
import nihai.data.nasdaq_calendar as nc


def _patch_get(monkeypatch, fn):
    """HTTP taşımasını yamala.

    Tur-65 (katalog #27): `nasdaq_calendar` artık modül düzeyi `httpx.get` ÇAĞIRMIYOR —
    her çağrı yeni bir `Client` kuruyordu ve bu ölçülen 224,66 ms'lik bir vergiydi
    (`_http_fetch_date` GÜN GÜN döndüğü için pencere başına ~3,1 sn). Taşıma
    `nihai.net.shared_client()` altına indi, yama noktası da onunla birlikte taşındı.
    `fn` imzası DEĞİŞMEDİ: `(url, params=..., headers=..., timeout=...)`.
    """
    monkeypatch.setattr(nc.net, "shared_client", lambda: SimpleNamespace(get=fn))


# --------------------------------------------------------------- sahte HTTP katmanı
class _FakeResp:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPError(f"HTTP {self.status_code}")


def _rows_payload(rows: list) -> dict:
    """Gerçek /api/calendar/earnings şemasından kırpılmış minimal gövde."""
    return {"data": {"asOf": None, "headers": {}, "rows": rows},
            "message": None, "status": {"rCode": 200}}


def _age_cache(events_dir, sym: str, *, minutes: int) -> None:
    """Diskteki cache dosyasının fetched_at'ini geriye alır (TTL testi)."""
    p = events_dir / f"{sym.upper()}.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw["fetched_at"] = (datetime.now() - timedelta(minutes=minutes)).isoformat()
    p.write_text(json.dumps(raw), encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_nasdaq_cache():
    nc.clear_cache()
    yield
    nc.clear_cache()


# ============================================================ nasdaq_calendar
def test_timing_normalizasyonu(monkeypatch):
    """(d) time-pre-market→BMO, time-after-hours→AMC, diğer/eksik→None; tarih=sorgu tarihi."""
    rows = [
        {"symbol": "AAA", "time": "time-pre-market"},
        {"symbol": "BBB", "time": "time-after-hours"},
        {"symbol": "CCC", "time": "time-not-supplied"},
        {"symbol": "DDD"},                              # time alanı yok
        {"symbol": "eee", "time": "time-pre-market"},   # küçük harf → normalize
    ]
    _patch_get(monkeypatch, lambda *a, **k: _FakeResp(_rows_payload(rows)))
    d = date(2026, 7, 20)
    out = nc.fetch_earnings_for_date(d)
    assert out is not None
    m = {r.symbol: r.timing for r in out}
    assert m == {"AAA": "BMO", "BBB": "AMC", "CCC": None, "DDD": None, "EEE": "BMO"}
    assert all(r.date == d for r in out)   # look-ahead güvenli: sağlayıcının tarihi


def test_tarayici_headerlari_gonderilir(monkeypatch):
    """(e-önkoşul) İstek User-Agent + Accept ile api.nasdaq.com'a gider."""
    seen: dict = {}

    def cap(url, **k):
        seen["url"] = url
        seen.update(k)
        return _FakeResp(_rows_payload([]))

    _patch_get(monkeypatch, cap)
    nc.fetch_earnings_for_date(date(2026, 7, 20))
    h = seen["headers"]
    assert "Mozilla" in h["User-Agent"]
    assert "application/json" in h["Accept"]
    assert seen["url"].startswith("https://api.nasdaq.com")


def test_http_403_none_doner_cachelenmez(monkeypatch):
    """(e) 403 → None; None cache'lenMEZ (hata yapışmaz) → sonraki çağrı yeniden dener."""
    calls = {"n": 0}

    def forbidden(*a, **k):
        calls["n"] += 1
        return _FakeResp({}, status=403)

    _patch_get(monkeypatch, forbidden)
    assert nc.fetch_earnings_for_date(date(2026, 7, 20)) is None
    assert calls["n"] == 2                              # retries=1 → 2 deneme
    assert nc.fetch_earnings_for_date(date(2026, 7, 20)) is None
    assert calls["n"] == 4                              # None cache'lenmedi → yeniden denedi


def test_headersiz_asili_kalma_timeout(monkeypatch):
    """(e) Gerçek gözlem: header'sız istek ReadTimeout verir → zarifçe None."""
    def hang(*a, **k):
        raise httpx.ReadTimeout("read timed out")

    _patch_get(monkeypatch, hang)
    assert nc.fetch_earnings_for_date(date(2026, 7, 20)) is None


def test_bos_gun_success_cachelenir(monkeypatch):
    """Boş AMA başarılı (ulaşıldı) yanıt cache'lenir → ikinci çağrı ağ atmaz."""
    calls = {"n": 0}

    def empty(*a, **k):
        calls["n"] += 1
        return _FakeResp(_rows_payload([]))

    _patch_get(monkeypatch, empty)
    d = date(2026, 7, 18)
    assert nc.fetch_earnings_for_date(d) == []
    assert nc.fetch_earnings_for_date(d) == []
    assert calls["n"] == 1


def test_date_cache_ttl_dolunca_yeniden_ceker(monkeypatch):
    """(f) Modül-içi tarih cache TTL: içinde ağ yok, dolunca yeniden çekim."""
    calls = {"n": 0}

    def once(*a, **k):
        calls["n"] += 1
        return _FakeResp(_rows_payload([{"symbol": "AAA", "time": "time-pre-market"}]))

    _patch_get(monkeypatch, once)
    d = date(2026, 7, 20)
    nc.fetch_earnings_for_date(d)
    nc.fetch_earnings_for_date(d)
    assert calls["n"] == 1                              # cache hit
    ts, rows = nc._DATE_CACHE[d.isoformat()]
    nc._DATE_CACHE[d.isoformat()] = (ts - timedelta(seconds=nc._DATE_CACHE_TTL_SEC + 1), rows)
    nc.fetch_earnings_for_date(d)
    assert calls["n"] == 2                              # TTL doldu → yeniden çekti


def test_get_next_earnings_sweep_ilk_eslesme(monkeypatch):
    """Sweep artan tarama; ilk eşleşme = sıradaki earnings; cache sembol arası paylaşılır."""
    target = date.today() + timedelta(days=3)
    while target.weekday() >= 5:                        # sweep hafta sonu atlar → hafta içi seç
        target += timedelta(days=1)

    def fake(url, params=None, **k):
        rows = [{"symbol": "AAA", "time": "time-after-hours"}] \
            if params and params["date"] == target.isoformat() else []
        return _FakeResp(_rows_payload(rows))

    _patch_get(monkeypatch, fake)
    r = nc.get_next_earnings("AAA", window_days=25)
    assert r is not None and r.date == target and r.timing == "AMC"
    assert nc.get_next_earnings("ZZZ", window_days=25) is None


def test_get_next_earnings_pencere_disinda_none(monkeypatch):
    """Pencere dışındaki earnings görülmez (guard'lar zaten yakın-pencere ile ilgilenir)."""
    far = date.today() + timedelta(days=40)
    while far.weekday() >= 5:
        far += timedelta(days=1)

    def fake(url, params=None, **k):
        rows = [{"symbol": "AAA", "time": "time-pre-market"}] \
            if params and params["date"] == far.isoformat() else []
        return _FakeResp(_rows_payload(rows))

    _patch_get(monkeypatch, fake)
    assert nc.get_next_earnings("AAA", window_days=14) is None


# ============================================================ event_calendar
@pytest.fixture
def _tmp_events(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "EVENTS_DIR", tmp_path)
    ec._MEM.clear()
    yield tmp_path
    ec._MEM.clear()


def _yf(ne=None, nx=None, ls=None, ok=True, div=None):
    # Tur-58: dönüş 5'li oldu (son alan = son ödenen temettü $/hisse; Madde 25).
    return lambda sym: (ne, nx, ls, ok, div)


def test_birincil_ok_fallback_cagrilmaz(_tmp_events, monkeypatch):
    """(a) yfinance earnings dolduysa Nasdaq fallback HİÇ çağrılmaz."""
    ne = date.today() + timedelta(days=10)
    monkeypatch.setattr(ec, "_fetch_yfinance_events", _yf(ne=ne))
    calls = {"n": 0}

    def spy(sym, **k):
        calls["n"] += 1
        return None

    monkeypatch.setattr(ec, "_nasdaq_next_earnings", spy)
    e = ec.get_company_events("AAA", refresh=True)
    assert e.next_earnings == ne
    assert e.source == "yfinance"
    assert e.timing is None
    assert calls["n"] == 0


def test_birincil_bos_fallback_doldurur(_tmp_events, monkeypatch):
    """(b) yfinance earnings boş (hatasız) → Nasdaq doldurur; kaynak yfinance+nasdaq."""
    monkeypatch.setattr(ec, "_fetch_yfinance_events", _yf(ne=None, ok=True))
    ne = date.today() + timedelta(days=3)
    monkeypatch.setattr(ec, "_nasdaq_next_earnings",
                        lambda sym, **k: nc.NasdaqEarnings(symbol=sym, date=ne, timing="AMC"))
    e = ec.get_company_events("BBB", refresh=True)
    assert e.next_earnings == ne
    assert e.timing == "AMC"
    assert e.source == "yfinance+nasdaq"


def test_birincil_hata_fallback_doldurur(_tmp_events, monkeypatch):
    """(b) yfinance earnings HATA → Nasdaq doldurur; kaynak yalnız nasdaq."""
    monkeypatch.setattr(ec, "_fetch_yfinance_events", _yf(ne=None, ok=False))
    ne = date.today() + timedelta(days=1)
    monkeypatch.setattr(ec, "_nasdaq_next_earnings",
                        lambda sym, **k: nc.NasdaqEarnings(symbol=sym, date=ne, timing="BMO"))
    e = ec.get_company_events("CCC", refresh=True)
    assert e.next_earnings == ne
    assert e.timing == "BMO"
    assert e.source == "nasdaq"


def test_ikisi_de_hata_negatif_cache_ve_yeniden_deneme(_tmp_events, monkeypatch):
    """(c) İkisi de hata → negatif cache; TTL içinde ağ yok; TTL dolunca yeniden dener."""
    calls = {"yf": 0, "nq": 0}

    def yf(sym):
        calls["yf"] += 1
        return (None, None, None, False, None)

    def nq(sym, **k):
        calls["nq"] += 1
        return None

    monkeypatch.setattr(ec, "_fetch_yfinance_events", yf)
    monkeypatch.setattr(ec, "_nasdaq_next_earnings", nq)

    e1 = ec.get_company_events("DDD")
    assert e1.next_earnings is None
    assert e1.source == ec._NEG_SOURCE
    assert calls["yf"] == 1 and calls["nq"] == 1

    e2 = ec.get_company_events("DDD")                  # TTL içinde → negatif cache servis
    assert e2.source == ec._NEG_SOURCE
    assert calls["yf"] == 1 and calls["nq"] == 1

    _age_cache(_tmp_events, "DDD", minutes=ec.NEG_CACHE_TTL_MIN + 1)
    e3 = ec.get_company_events("DDD")                  # TTL doldu → yeniden dener
    assert e3.source == ec._NEG_SOURCE
    assert calls["yf"] == 2


def test_hard_failure_24h_pozitif_cache_yazmaz(_tmp_events, monkeypatch):
    """Hard-failure 24h cache yazmamalı: geçici hata earnings guard'ını 24 saat kapatmasın."""
    monkeypatch.setattr(ec, "_fetch_yfinance_events", lambda sym: (None, None, None, False, None))
    monkeypatch.setattr(ec, "_nasdaq_next_earnings", lambda sym, **k: None)
    ec.get_company_events("EEE")
    raw = json.loads((_tmp_events / "EEE.json").read_text(encoding="utf-8"))
    assert raw["source"] == ec._NEG_SOURCE             # 24h değil, kısa-TTL negatif


def test_timing_cache_persist_turu(_tmp_events, monkeypatch):
    """timing alanı diske yazılır ve cache'ten geri okunur (round-trip)."""
    ne = date.today() + timedelta(days=4)
    monkeypatch.setattr(ec, "_fetch_yfinance_events", lambda sym: (None, None, None, True, None))
    monkeypatch.setattr(ec, "_nasdaq_next_earnings",
                        lambda sym, **k: nc.NasdaqEarnings(symbol=sym, date=ne, timing="AMC"))
    ec.get_company_events("FFF", refresh=True)
    ec._MEM.clear()
    e = ec.get_company_events("FFF")
    assert e.timing == "AMC" and e.source == "yfinance+nasdaq" and e.next_earnings == ne


def test_yfinance_bos_nasdaq_bos_pozitif_cachelenir(_tmp_events, monkeypatch):
    """yfinance ULAŞTI ama boş + Nasdaq boş → hata YOK → 24h pozitif cache (negatif değil)."""
    monkeypatch.setattr(ec, "_fetch_yfinance_events", lambda sym: (None, None, None, True, None))
    monkeypatch.setattr(ec, "_nasdaq_next_earnings", lambda sym, **k: None)
    e = ec.get_company_events("GGG", refresh=True)
    assert e.next_earnings is None
    assert e.source == "yfinance"                      # negatif-cache DEĞİL
    raw = json.loads((_tmp_events / "GGG.json").read_text(encoding="utf-8"))
    assert raw["source"] == "yfinance"
