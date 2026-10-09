"""Ağ guard'ı (BULGU-19) kalıcı guard testleri — guard'ın NO-OP OLMADIĞINI kanıtlar.

Bu dosya DIŞ AĞA TEK PAKET GÖNDERMEZ: hedef olarak RFC 5737 belgeleme aralığından
192.0.2.1 kullanılır ve guard zaten `connect` çağrılmadan ÖNCE yükselir (DNS de yok,
çünkü hedef ham IP). Yani guard bozulursa test "yavaşlayıp geçmez" — hedef
yönlendirilemez olduğu için davranış yine gözlenebilir kalır.

Neden gerekli: `test_main_llm_falls_back_deterministic` docstring'inde "ağ çağrısı YOK"
yazarken her tam koşuda gerçek bir Gemini isteği atıyordu (12,9 sn + ücretsiz kotadan
bir istek). Guard'ın kendisi test edilmezse aynı sızıntı sessizce geri gelir.
"""
from __future__ import annotations

import socket

import pytest

import conftest as root_conftest

# MODÜL DÜZEYİNDE import edilir (Tur-26): kök conftest'in `_vix_onbellegi_agsiz` fixture'ı
# yalnız `nihai.signals.strategy` ZATEN yüklüyse yamalar (ilgisiz testlere ~1 sn import
# maliyeti ödetmemek için). Test gövdesinde import edilseydi fixture'ı ıskalardı.
from nihai.signals import strategy as strategy_mod

_TESTNET = ("192.0.2.1", 443)      # RFC 5737 — yönlendirilemez, DNS gerektirmez


def test_dis_hedef_dis_sayilir_loopback_sayilmaz():
    assert root_conftest._is_external(_TESTNET) is True
    for yerel in ("127.0.0.1", "::1", "localhost"):
        assert root_conftest._is_external((yerel, 8765)) is False


def test_af_unix_benzeri_adres_dokunulmaz():
    """(host, port) OLMAYAN adresler (AF_UNIX yolu vb.) guard'ın konusu değil."""
    assert root_conftest._is_external("/tmp/soket") is False
    assert root_conftest._is_external(()) is False


def test_connect_engellenir_ve_deftere_yazilir():
    """Guard hem YÜKSELİR hem KAYDEDER — kayıt, fail-soft yutulan çağrıyı da yakalar."""
    root_conftest._net_attempts.clear()
    s = socket.socket()
    try:
        with pytest.raises(OSError, match="ağ guard"):
            s.connect(_TESTNET)
        # Üretim kodu bu OSError'ı yutabilir (router.complete sözleşmesi); defterdeki
        # kayıt tam da bu yüzden var — çağrı fazı kancası ihlali yine görür.
        assert any("192.0.2.1" in a for a in root_conftest._net_attempts)
    finally:
        # Bilerek tetiklenen ihlal: defteri temizle, yoksa kanca BU testi düşürür.
        root_conftest._net_attempts.clear()
        s.close()


def test_connect_ex_de_engellenir():
    root_conftest._net_attempts.clear()
    s = socket.socket()
    try:
        with pytest.raises(OSError, match="ağ guard"):
            s.connect_ex(_TESTNET)
        assert root_conftest._net_attempts
    finally:
        root_conftest._net_attempts.clear()
        s.close()


# ---------------------------------------------------------------------------
# BULGU-26 (Tur-26): curl_cffi socket katmanını ATLAR — ayrı yama, ayrı test.
# ---------------------------------------------------------------------------
def test_curl_cffi_de_engellenir():
    """yfinance 1.x'in taşıması curl_cffi'dir; libcurl'ü cffi ile çağırdığı için
    `socket.socket.connect` yaması onu HİÇ görmez.

    ÖLÇÜLDÜ (2026-08-11, yama ÖNCESİ): guard etkinken `curl_cffi.requests.get(
    query2.finance.yahoo.com/...)` **status=429** döndürdü — istek gerçek Yahoo
    sunucusuna ULAŞMIŞTI ve defterde iz YOKTU. Bu test o deliğin geri gelmesini
    engeller. Hedef yine RFC 5737 belgeleme IP'si: guard bozuksa bile paket gitmez."""
    curl_requests = pytest.importorskip("curl_cffi.requests")
    root_conftest._net_attempts.clear()
    try:
        with pytest.raises(OSError, match="ağ guard"):
            curl_requests.get("https://192.0.2.1/test", timeout=5)
        assert any("192.0.2.1" in a for a in root_conftest._net_attempts)
    finally:
        root_conftest._net_attempts.clear()


def test_curl_cffi_loopback_serbest():
    """Kural socket guard'ıyla AYNI olmalı: loopback engellenmez (yerel hub testleri)."""
    pytest.importorskip("curl_cffi.requests")
    root_conftest._net_attempts.clear()
    root_conftest._guard_url("http://127.0.0.1:8765/bars")   # yükselmemeli
    root_conftest._guard_url("http://localhost:8799/health")
    assert root_conftest._net_attempts == []


def test_vix_yukleyicisi_testlerde_etkisiz():
    """`strategy._load_vix()` 10 YILLIK canlı `yf.download("^VIX")` yapıyordu ve sonucu
    modül-global sakladığı için ihlali HANGİ testin önce koştuğu belirliyordu (sıra-bağımlı).
    Kök conftest'teki autouse fixture bunu etkisizleştirir; bu test onu kilitler."""
    import datetime as _dt
    assert strategy_mod._VIX_LOADED is True      # "yüklendi" → indirme denenmez
    assert strategy_mod._VIX_DATES is None
    assert strategy_mod._vix_close_before(_dt.date(2026, 1, 2)) is None


@pytest.mark.allow_network
def test_marker_guardi_kapatir():
    """@allow_network işaretli testte guard karar vermez (bağlantı DENENMEZ)."""
    assert root_conftest._is_external(_TESTNET) is False


def test_loopback_gercekten_calisir():
    """Guard yerel dinleyicileri kırmamalı — _port_open testlerinin dayandığı davranış."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        with socket.create_connection(("127.0.0.1", srv.getsockname()[1]), timeout=2):
            pass
    finally:
        srv.close()


def test_daily_summary_llm_yolu_hermetik(tmp_path, monkeypatch, capsys):
    """Regresyon: --llm yolu sağlayıcı anahtarı olsa BİLE ağa çıkmamalı.

    Anahtarlar test ortamında SET EDİLİR (gerçek .env senaryosu); testin kendisi
    router.complete'i monkeypatch'lediği için hiçbir SDK çağrılmaz. Guard defteri
    boş kalmalı."""
    import nihai.ai.router as router
    import nihai.notifier.daily_summary as ds

    monkeypatch.setenv("GEMINI_API_KEY", "sahte-anahtar")
    monkeypatch.setattr(ds, "SUMMARY_TXT", tmp_path / "s.txt")
    monkeypatch.setattr(ds, "collect", lambda: {
        "today": "2026-07-31", "history_last": None, "history_prev": None,
        "gate_sim": None, "coverage": None,
        "services": {"hub": False, "kayitci": False}, "live_features_today": None})
    monkeypatch.setattr(router, "complete", lambda task, prompt, **kw: None)
    assert ds.main(["--llm"]) == 0
    capsys.readouterr()
    assert not root_conftest._net_attempts
