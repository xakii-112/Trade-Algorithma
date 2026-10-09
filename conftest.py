"""Kök conftest — pytest'in proje kökünü sys.path'e eklemesini garantiler.

Ayrıca test sürecinde dosya-log sink'ini kapatır (NIHAI_DISABLE_FILE_LOG):
main.py'yi (dolaylı: research.duo) import eden bir test, loguru dosya sink'ini
test sürecine eklerdi → tüm test logları üretim logs/system.log'una akar,
main'in import-anı prune_logs()'u gerçek log/CSV dosyalarını silebilirdi.

BULGU-8: NIHAI_DISABLE_FILE_LOG yalnız system.log sink'ini koruyordu; testler
signal_engine/position_manager/auto_screener üzerinden ÜRETİM trade_log.txt,
decision_log_*.txt, auto_screen_*.csv ve data/trades.csv dosyalarına sahte
işlem yazıyordu (gözlem: trades.csv'de 8k+ AAA/TEST_CLOSE satırı). Bu yüzden
LOG_DIR ve DATA_DIR test sürecinde geçici dizine yönlendirilir — nihai.config
env'i import anında okur; kök conftest her nihai importundan önce koşar."""
import atexit
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("NIHAI_DISABLE_FILE_LOG", "1")
_tmp = Path(tempfile.mkdtemp(prefix="nihai-pytest-"))
os.environ.setdefault("NIHAI_LOG_DIR", str(_tmp / "logs"))
os.environ.setdefault("NIHAI_DATA_DIR", str(_tmp / "data"))
# Tur-97: testlerin VARSAYILAN stratejisi operatörün `.env`'ini izlememeli. `load_dotenv()` var olan env'i
# ezmez; `.env` canlıda `quant_trend`e geçince stratejisiz `generate_signal` çağıran testler 1h veride
# `QUANT_GUNLUK_BAR_GEREKLI` aldı. Kod varsayılanıyla (config.py) aynı değer sabitlenir.
os.environ["ACTIVE_STRATEGY"] = "event_breakout"
# Tur-102: gölge kasalar tahmin servisinin bar yolunda koşar (model çağrıları + disk). Tahmin servisini
# süren testler onları yan etki olarak tetiklemesin; kasa testleri kipi kendisi açar.
os.environ["GOLGE_KASA"] = "off"


@atexit.register
def _temizle_gecici_dizin() -> None:
    """Koşu sonunda mkdtemp dizinini sil (Tur-26).

    Dizin oluşturuluyor ama HİÇ silinmiyordu: ölçüm anında %TEMP%'te **479** birikmiş
    `nihai-pytest-*` dizini vardı (her pytest koşusu bir tane). `atexit` seçildi çünkü
    dizin pytest'ten ÖNCE, modül import anında yaratılıyor (nihai.config env'i import
    anında okur) — bir pytest fixture/hook'u onun ömrünü kapsayamaz.
    Silme fail-soft: Windows'ta bir tanıtıcı hâlâ açıksa koşuyu düşürmemeli."""
    shutil.rmtree(_tmp, ignore_errors=True)

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# AĞ GUARD'I (BULGU-19, 2026-07-31) — BULGU-8/9'un ağ karşılığı.
#
# BULGU-8/9 testlerin ÜRETİM DOSYALARINA yazmasını kapatmıştı; testler DIŞ AĞA
# çıkmaya devam ediyordu. Kanıt: `test_main_llm_falls_back_deterministic` docstring'i
# "ağ çağrısı YOK" dediği hâlde her tam koşuda generativelanguage.googleapis.com'a
# GERÇEK bir Gemini isteği atıyordu (12,9 sn + ücretsiz kotadan 1 istek). O kota
# 2026-07-13'te FAZ AI-1 ölçümünü zaten bloke etmişti (OPTIMIZASYON_LOG Ek Tur 11).
#
# Guard iki katmanlı, çünkü tek başına engellemek YETMEZ: üretim kodu LLM hatasını
# bilerek yutar (router.complete sözleşmesi) → engellenen çağrı testi yeşil bırakır,
# ihlal görünmez kalır. Bu yüzden deneme AYRICA kaydedilir ve teardown'da test
# ERROR'a düşer.
#   * loopback (127.0.0.1/::1/localhost) SERBEST — _port_open testleri gerçek yerel
#     dinleyici açar.
#   * istisna gereken test: @pytest.mark.allow_network
#   * tüm paketi canlıya açmak: NIHAI_ALLOW_NETWORK=1
# ---------------------------------------------------------------------------
_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}
_ALLOW_ALL = os.environ.get("NIHAI_ALLOW_NETWORK") == "1"
_net_attempts: list[str] = []
_net_allowed = {"on": False}

_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex


def _is_external(address: object) -> bool:
    """Yalnız (host, port) TCP/UDP hedefleri denetlenir; AF_UNIX vb. dokunulmaz."""
    if _ALLOW_ALL or _net_allowed["on"] or not isinstance(address, tuple) or not address:
        return False
    return str(address[0]) not in _LOOPBACK


def _guard_connect(self, address, *args, **kwargs):
    if _is_external(address):
        _net_attempts.append(str(address))
        raise OSError(f"[ağ guard'ı] testte dış ağ çağrısı engellendi: {address}")
    return _orig_connect(self, address, *args, **kwargs)


def _guard_connect_ex(self, address, *args, **kwargs):
    if _is_external(address):
        _net_attempts.append(str(address))
        raise OSError(f"[ağ guard'ı] testte dış ağ çağrısı engellendi: {address}")
    return _orig_connect_ex(self, address, *args, **kwargs)


socket.socket.connect = _guard_connect          # type: ignore[method-assign]
socket.socket.connect_ex = _guard_connect_ex    # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# BULGU-26 (Tur-26) — GUARD'IN DELİĞİ: curl_cffi, socket katmanını ATLAR.
#
# BULGU-19'un socket yaması yalnız Python'ın `socket` modülünü kullanan istemcileri
# (requests/urllib3/httpx) yakalar. yfinance 1.x HTTP taşımasını **curl_cffi**'ye taşıdı;
# o ise libcurl'ü cffi üzerinden çağırır → `socket.socket.connect` HİÇ çalışmaz.
# ÖLÇÜLDÜ (2026-08-11, guard etkinken): `curl_cffi.requests.get(query2.finance.yahoo.com)`
# **status=429 döndü** — yani istek gerçek Yahoo sunucusuna ULAŞTI ve guard ne engelledi
# ne bildirdi. Aynı koşuda `socket.create_connection` doğru şekilde engellendi.
# Sonuç: yfinance yoluna dokunan her test sessizce canlı ağa çıkıyordu (kota yakar,
# çevrimdışı geliştirmeyi bozar, süreyi şişirir: `test_position_size_gercek_hesap` 9,80 sn).
#
# Yama noktası `Session.request`: modül düzeyi `get/post/...` yardımcıları da içeriden
# Session kullanır, yani tek nokta iki yüzeyi birden kapatır.
# ---------------------------------------------------------------------------
def _guard_url(url: object) -> None:
    """URL'nin host'unu socket guard'ıyla AYNI kurala sokar (loopback serbest)."""
    from urllib.parse import urlparse
    try:
        host = urlparse(str(url)).hostname or ""
    except ValueError:            # ayrıştırılamayan URL → dış kabul et
        host = str(url)
    if _is_external((host, 0)):
        _net_attempts.append(str(url))
        raise OSError(f"[ağ guard'ı] testte dış ağ çağrısı engellendi: {url}")


try:                                             # pragma: no cover - opsiyonel bağımlılık
    from curl_cffi import requests as _curl_requests
except Exception:                                # pragma: no cover
    _curl_requests = None                        # type: ignore[assignment]

if _curl_requests is not None:                   # pragma: no cover - ortam bağımlı
    _orig_curl_request = _curl_requests.Session.request

    def _guard_curl_request(self, method, url, *args, **kwargs):
        _guard_url(url)
        return _orig_curl_request(self, method, url, *args, **kwargs)

    _curl_requests.Session.request = _guard_curl_request        # type: ignore[method-assign]

    if hasattr(_curl_requests, "AsyncSession"):
        _orig_curl_arequest = _curl_requests.AsyncSession.request

        async def _guard_curl_arequest(self, method, url, *args, **kwargs):
            _guard_url(url)
            return await _orig_curl_arequest(self, method, url, *args, **kwargs)

        _curl_requests.AsyncSession.request = _guard_curl_arequest   # type: ignore[method-assign]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "allow_network: bu test DIŞ ağa çıkabilir (ağ guard'ı devre dışı)")


@pytest.fixture(autouse=True)
def _izole_edge_defteri(tmp_path, monkeypatch):
    """Her teste AYRI kenar-keşif defteri (`research.edge_lab.ledger`).

    Defter, DSR deflasyonunun `n_trials`'ını belirler (program aradıkça kanıt eşiği
    sıkışır). DATA_DIR tüm oturum için TEK geçici dizin olduğundan defter test'ten
    test'e birikir ve `run_candidate` çağıran her test bir SONRAKİNİN eşiğini yükseltirdi:
    kapı sıra-bağımlı olur, tam koşu ile tek-dosya koşusu FARKLI verdikt verir. Bu, kanıt
    kapısının kendisinin bozulmasıdır — üretimde istenen birikim, testte izolasyon."""
    monkeypatch.setenv("NIHAI_EDGE_LEDGER", str(tmp_path / "edge_lab_ledger.jsonl"))


@pytest.fixture(autouse=True)
def _vix_onbellegi_agsiz(request, monkeypatch):
    """`strategy._load_vix()`'i testlerde ETKİSİZLEŞTİR (Tur-26, BULGU-26'nın devamı).

    `meta_features.extract_features` → `_vix_close_before` → `_load_vix()` zinciri
    `yf.download("^VIX", period="10y")` ile CANLI ağa çıkıyordu. Sonuç modül-global
    (`_VIX_LOADED`) olduğu için maliyeti ve —curl_cffi deliği kapandıktan sonra— guard
    ihlalini HANGİ testin önce koştuğu belirliyordu: sıra-bağımlı, gerçek bir kırılganlık.
    `test_ai_shadow_runner.py` bunu zaten yerel olarak çözmüştü; burada genelleştiriliyor.

    VIX yokluğu güvenli: `_vix_or_neutral` nötr 18.0'a düşer (F6, 2026-07-26 denetimi).
    `allow_network` işaretli testler muaf; monkeypatch olduğu için testin kendi
    setattr'ı bunu EZEBİLİR (shadow_runner testi tam olarak bunu yapar)."""
    if request.node.get_closest_marker("allow_network"):
        return
    strategy = sys.modules.get("nihai.signals.strategy")
    if strategy is None:
        # Test bu modüle hiç dokunmuyor → ilgisiz testlere ~1 sn import maliyeti ödetme.
        # Kalan boşluk (modülü test GÖVDESİNDE import eden bir test) artık SESSİZ değil:
        # curl_cffi yaması indirmeyi engeller ve defter ihlali testin kendi çağrı fazında
        # yükseltir — yani gözden kaçmaz, gürültülü şekilde kırılır.
        return
    monkeypatch.setattr(strategy, "_VIX_LOADED", True, raising=False)
    monkeypatch.setattr(strategy, "_VIX_DATES", None, raising=False)
    monkeypatch.setattr(strategy, "_VIX_CLOSES", None, raising=False)


def pytest_runtest_setup(item):
    _net_allowed["on"] = item.get_closest_marker("allow_network") is not None
    _net_attempts.clear()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """İhlali TEST FAZINDA bildir — teardown'da DEĞİL.

    İlk sürüm teardown'da yükseliyordu; pytest o istisnayı SetupState'e taşıyıp
    SONRAKİ testlerin setup'ını da düşürüyordu (tek ihlal → zincirleme ERROR).
    `wrapper=True` ile ihlal, testin kendi çağrı fazında normal bir hataya döner.
    Test zaten başarısızsa `yield` yükselir ve asıl hata MASKELENMEZ."""
    res = yield
    if _net_attempts:
        hedefler = ", ".join(sorted(set(_net_attempts)))
        _net_attempts.clear()
        raise AssertionError(
            f"[ağ guard'ı] test DIŞ ağa çıkmayı denedi: {hedefler}\n"
            "Testler hermetik olmalı (kota yakar, çevrimdışı geliştirmeyi bozar, "
            "kırılganlık üretir). Sağlayıcıyı monkeypatch'le ya da anahtar env'lerini "
            "delenv ile temizle; gerçekten gerekiyorsa @pytest.mark.allow_network.")
    return res
