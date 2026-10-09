"""
net.py — SÜREÇ ÖMRÜ BOYUNCA TEK `httpx.Client` (dış katalog maddesi #27).

NEDEN: `httpx.get(...)` / `httpx.post(...)` modül düzeyi kısayolları her çağrıda YENİ bir
`Client` kurar. Bedel ağ DEĞİL, istemcinin kendisidir: httpx SSL bağlamını + CA demetini
İSTEMCİ KURULURKEN (tembel değil) yükler. ÖLÇÜLDÜ (ağsız, 15 yineleme, bu makine):

    httpx.Client()            → p50 **224,66 ms**
    requests.Session()        → p50 **  0,02 ms**   (SSL bağlamını bağlantıda, tembel kurar)
    ssl.create_default_context() → p50 14,39 ms

Yani bu vergi httpx'e ÖZGÜDÜR — `requests` çağıranları aynı sorunu yaşamaz ve bu yüzden
BİLEREK dokunulmadı (bkz. Tur-64 dersi: önerinin dayandığı koşulu ölç, varsayma).

KÖK NEDEN, TEK ÇAĞIRAN DEĞİL (Tur-65 ikinci geçiş): Tur-65'in ilk geçişi bu düzeltmeyi
YALNIZCA `notifier/telegram_bot.py`ye uygulamıştı. Aynı desendeki kardeş çağıranlar açıktı
ve ikisi de DÖNGÜ içindeydi:

  · `data/nasdaq_calendar._http_fetch_date` — `fetch_next_earnings` **gün gün** döner
    (`for off in range(window_days + 1)`), her gün için ayrı `httpx.get` + retry.
    14 günlük pencere = 14 istemci = ~3,1 sn saf kurulum.
  · `execution/screen._read_html_tables` — evren tazelemesinde Wikipedia'ya iki çağrı.

Üç modül de artık BU istemciyi paylaşır: süreçte tek SSL bağlamı, tek bağlantı havuzu
(keep-alive de bedavaya gelir).

TEMBELLİK ZORUNLU: istemci import zamanında kurulursa `import nihai.notifier.telegram_bot`
tek başına 225 ms'ye çıkar — Tur-26'nın `import nihai.ai` 3854 → 11 ms dersinin aynısı.
Bu yüzden global, ilk `shared_client()` çağrısında kurulur.

ZAMAN AŞIMI / YÖNLENDİRME BURADA AYARLANMAZ: `timeout` ve `follow_redirects` httpx'te
İSTEK BAŞINA parametrelerdir ve çağıranların gerçek ihtiyacı farklıdır (`sendPhoto` 20 sn,
`getUpdates` long-poll `timeout + 10`, Wikipedia `follow_redirects=True`). İstemciye ortak
bir varsayılan dayatmak bu farkları sessizce ezerdi.
"""
from __future__ import annotations

import os
import re
import threading

import httpx

_client: httpx.Client | None = None
_lock = threading.Lock()


def shared_client() -> httpx.Client:
    """Paylaşılan `httpx.Client` (tembel kurulur, thread-safe).

    httpx `Client`'ı eşzamanlı kullanım için belgelenmiş biçimde güvenlidir; çağıranlar
    daemon thread'lerden (Telegram dinleyicisi, tarayıcı havuzu) aynı anda kullanır.
    """
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = httpx.Client()
    return _client


_ua_uyarildi = False
# Kendi kilidi: `_lock` İSTEMCİ kilididir ve `shared_client()` onu ~225 ms'lik
# `httpx.Client()` kurulumu boyunca tutar (yukarıdaki ölçüm). UA dizesi üretmek o
# kuyruğa girmemeli — çağıranlar `contact_ua()`yu `shared_client()`ten ÖNCE çağırır.
_ua_lock = threading.Lock()

# "Dolu" YETMEZ, ULAŞILABİLİR ADRES gerekir. `"@" in contact` testi yetersizdi: `"@"`,
# `"NihaiBot@"` ve `"Ad Soyad @nihaibot" ` ondan sessizce geçiyordu (403 riski aynen
# duruyor, tek uyarı çıkmıyordu) — üstelik operatörün yapmaya en yatkın olduğu hata
# tam olarak budur. Karşı yön de yanlıştı: Wikimedia'nın UA politikası e-posta VEYA
# iletişim URL'si kabul eder, "NihaiAlgo/1.0 (+https://github.com/x/y)" geçerli bir
# UA'yken uyarı alıyordu. Desen ikisini de tanır.
_ADRES = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}|https?://[^\s)]+")


def contact_ua(product: str) -> str:
    """"Kendini tanit" politikali uclar (SEC EDGAR, FINRA, Wikimedia) icin User-Agent.

    NEDEN ORTAK: dort cagiran (`data/insider.py`, `execution/screen.py`, edge_lab'in iki
    veri kurucusu) ayni dizeyi KOPYALAMISTI ve dizenin icinde gelistiricinin kisisel
    e-postasi vardi. Depo herkese acildiginda o adres dort dosyada birden yayinlanir ve
    tek dosyayi duzeltmek digerlerini acikta birakir. Tek kaynak: `.env`deki
    `SEC_USER_AGENT` (kimlik/iletisim bilgisi kodda DEGIL, `.env`de).

    `.env` BURADA yuklenir cunku burasi `SEC_USER_AGENT`i okuyan TEK yerdir. edge_lab'in
    iki kurucusu `nihai.config`i hic import etmez; env'e bugun yalnizca TESADUFI bir
    zincirle (`research/edge_lab/__init__` -> harness -> ledger -> `nihai.config`)
    ulasiyor. O zincir koptugu gun UA iletisim bilgisini SESSIZCE kaybederdi.

    POLITIKA ADRES ISTER, "dolu" YETMEZ: SEC ve Wikimedia UA'da ulasilabilir bir adres
    (e-posta ya da iletisim URL'si) arar — bkz. `_ADRES`. `SEC_USER_AGENT` BOS ise de
    "MyBot" gibi ADRESSIZ ise de 403 riski aynidir, bu yuzden ikisi de uyarir. Adressiz
    deger yine de KULLANILIR — operator onu bilerek set etmistir; karari onun, ama
    sessizce degil.
    """
    global _ua_uyarildi
    from nihai import config as _config  # noqa: F401 — import'un yan etkisi: load_dotenv()

    contact = os.getenv("SEC_USER_AGENT", "").strip()
    # Kilit ZORUNLU: `build_shortflow_dataset.load_shortvol` bu yolu 6 thread'lik havuzdan
    # cagiriyor; korumasiz read-modify-write'ta altisi da bayragi bos gorup ayni uyariyi
    # basardi ("surecte BIR KEZ" sozlesmesi tek thread'de dogru, havuzda yanlis olurdu).
    # Cift kontrol `shared_client()`in desenidir: uyari basildiktan sonra kilit alinmaz.
    if not _ADRES.search(contact) and not _ua_uyarildi:
        with _ua_lock:
            ilk_kez = not _ua_uyarildi
            _ua_uyarildi = True
        if ilk_kez:
            from loguru import logger
            logger.warning(
                ("SEC_USER_AGENT tanimsiz" if not contact
                 else f"SEC_USER_AGENT iletisim adresi tasimiyor ({contact!r})")
                + f" — '{product}' istekleri SEC/FINRA/Wikimedia'dan 403 alabilir"
                " (Wikipedia evren tazelemesinde S&P 500 listesi BOSALIR). .env'e"
                ' SEC_USER_AGENT="Ad Soyad eposta@ornek.com" ekleyin.'
            )
    return (f"{product} ({contact})" if contact
            else f"{product} (iletisim: SEC_USER_AGENT tanimsiz)")


def close_shared_client() -> None:
    """İstemciyi kapat ve global'i sıfırla.

    ÜRETİM ÇAĞIRANI YOKTUR ve bu bilinçlidir — istemci süreç ömrü boyunca yaşar, soketleri
    çıkışta işletim sistemi toplar. Tek çağıran test paketidir: süreç-genelindeki bir
    global'i sıfırlamadan "tembel kurulur" iddiası sınanamaz ve bir test canlı istemciyi
    diğerine sızdırır. Gerekçe burada YAZILIDIR çünkü bu depoda çağıranı olmayan public
    tanım varsayılan olarak kusur sayılır (Tur-50/62).
    """
    global _client
    with _lock:
        if _client is not None:
            _client.close()
            _client = None
