"""
nasdaq_calendar.py — Bilanço takvimi İKİNCİ kaynak (keysiz Nasdaq API).

PDF Referansları:
- Madde 26 : Olay/bilanço takvimi farkındalığı (event_calendar.py'nin fallback'i)
- Madde 17 : IV crush koruması (bilanço öncesi long opsiyon yasağı)

yfinance bilanço tarihleri zaman zaman gecikmeli/eksik/rate-limited döner. Bu modül
``api.nasdaq.com``'un keysiz (ücretsiz, anahtarsız) takvim ucundan ikinci bir teyit
sağlar ve — yfinance'te olmayan — BMO/AMC (piyasa-öncesi / kapanış-sonrası) zamanlamasını
normalize eder.

Uçlar (2026-07 gerçek istekle doğrulandı):
- Tarih-bazlı: ``GET /api/calendar/earnings?date=YYYY-MM-DD`` → ``data.rows[]`` içinde
  ``{symbol, time, name, epsForecast, ...}``. ``time`` alanı: ``time-pre-market`` → BMO,
  ``time-after-hours`` → AMC, ``time-not-supplied``/eksik → None. Satır TARİHİ sorgu
  tarihidir (look-ahead güvenli: sağlayıcının verdiği tarih olduğu gibi kullanılır).
- Sembol-bazlı ``GET /api/analyst/{symbol}/earnings-date`` de mevcuttur AMA serbest-metin
  ("... on 07/30/2026 after market close ...") döndürür; ayrıştırması kırılgan olduğundan
  BURADA TARİH-BAZLI uç tercih edildi. Bir sembolün sıradaki bilançosu, bugünden +N gün
  penceresi TEK taramayla çekilip (modül-içi kısa cache) çözülür → sembol başına ayrı ayrı
  N istek atılmaz; tüm izleme listesi aynı taramayı paylaşır.

api.nasdaq.com tarayıcı-benzeri header ister (User-Agent + Accept). Header'sız istek 403
verir ya da asılı kalır (gözlem: header'sız istek ReadTimeout). Bu yüzden _HEADERS zorunlu.

Hata-yapışması yasağı (BULGU-14): GEÇİCİ ağ hatası (None) modül cache'ine YAZILMAZ —
yalnız başarılı (ulaşılan) yanıt, boş olsa bile, kısa-TTL cache'lenir → sonraki tarama
yeniden dener. KENAR İDDİASI YOK — bu modül yalnız takvim verisi taşır.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from loguru import logger

from nihai import net

_BASE_URL = "https://api.nasdaq.com"

# api.nasdaq.com header'sız 403/asılı kalma verir → tarayıcı-benzeri başlıklar zorunlu.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}

# Nasdaq "time" alanı → normalize zamanlama etiketi (diğer/eksik → None).
_TIMING_MAP = {
    "time-pre-market": "BMO",   # Before Market Open — bilanço piyasa açılışından önce
    "time-after-hours": "AMC",  # After Market Close — bilanço kapanıştan sonra
}

DEFAULT_TIMEOUT = 10.0        # saniye — task: ≤10s
DEFAULT_RETRIES = 1           # 1 yeniden deneme (toplam 2 deneme)
DEFAULT_WINDOW_DAYS = 14      # sembolün sıradaki bilançosu için taranan gün penceresi

# Modül-içi tarih cache'i: iso-tarih → (çekim_zamanı, satırlar). Kısa TTL: canlı-taze
# kalır ama gün-içi tekrar taramada (izleme listesi genelinde) aynı tarihe istek atmaz.
_DATE_CACHE_TTL_SEC = 900     # 15 dk
_DATE_CACHE: dict[str, tuple[datetime, list[NasdaqEarnings]]] = {}


@dataclass
class NasdaqEarnings:
    """Normalize bilanço kaydı. ``date`` = sağlayıcının verdiği event-tarihi (kaydırma yok)."""
    symbol: str
    date: date
    timing: str | None = None   # "BMO" | "AMC" | None


def _parse_rows(rows: list, d: date) -> list[NasdaqEarnings]:
    """Ham ``rows`` listesini normalize NasdaqEarnings listesine çevirir (tarih = ``d``)."""
    out: list[NasdaqEarnings] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        timing = _TIMING_MAP.get(str(row.get("time") or "").strip())
        out.append(NasdaqEarnings(symbol=sym, date=d, timing=timing))
    return out


def _http_fetch_date(d: date, *, timeout: float, retries: int) -> list | None:
    """Tek tarihin ham ``rows``'unu çeker. Ulaşıldı → list (boş olabilir); AĞ HATASI → None.

    None dönüşü 'geçici hata' anlamına gelir ve ÇAĞIRAN tarafından cache'lenMEZ (BULGU-14).
    """
    url = f"{_BASE_URL}/api/calendar/earnings"
    params = {"date": d.isoformat()}
    attempts = max(1, retries + 1)
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            r = net.shared_client().get(url, params=params, headers=_HEADERS,
                                       timeout=timeout)
            r.raise_for_status()
            data = r.json()
            block = data.get("data") if isinstance(data, dict) else None
            if not isinstance(block, dict):
                return []
            rows = block.get("rows")
            return rows if isinstance(rows, list) else []
        except Exception as e:   # ağ/HTTP/parse hatası tek noktada yutulur (retry + None)
            last_err = e
            logger.debug(f"[nasdaq] {d} deneme {i + 1}/{attempts}: {e}")
    logger.warning(f"[nasdaq] {d} takvim çekilemedi ({attempts} deneme): {last_err}")
    return None


def fetch_earnings_for_date(
    d: date, *, timeout: float = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES,
    use_cache: bool = True,
) -> list[NasdaqEarnings] | None:
    """Bir günün bilanço kayıtlarını (normalize) döndürür.

    Ulaşıldı → list (boş olabilir, cache'lenir); geçici ağ hatası → None (cache'lenMEZ).
    """
    key = d.isoformat()
    if use_cache:
        hit = _DATE_CACHE.get(key)
        if hit and (datetime.now() - hit[0]).total_seconds() < _DATE_CACHE_TTL_SEC:
            return hit[1]
    rows = _http_fetch_date(d, timeout=timeout, retries=retries)
    if rows is None:
        return None   # geçici hata → cache YOK → sonraki çağrı yeniden dener
    parsed = _parse_rows(rows, d)
    if use_cache:
        _DATE_CACHE[key] = (datetime.now(), parsed)
    return parsed


def get_next_earnings(
    symbol: str, *, window_days: int = DEFAULT_WINDOW_DAYS,
    timeout: float = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES,
) -> NasdaqEarnings | None:
    """Sembolün bugünden +``window_days`` gün içindeki SIRADAKI bilançosu (yoksa None).

    Tarih-bazlı ucu artan sırayla tarar; ilk eşleşme = en erken (sıradaki) bilanço →
    hemen döner. Hafta sonları atlanır (Nasdaq takvimi boş). Her tarih modül-içi
    cache'lenir → aynı taramayı çoklu sembol paylaşır (sembol başına N istek atılmaz).
    """
    sym = symbol.upper()
    today = date.today()
    for off in range(window_days + 1):
        d = today + timedelta(days=off)
        if d.weekday() >= 5:   # Cmt/Pazar → US bilanço yok, isteği atla
            continue
        rows = fetch_earnings_for_date(d, timeout=timeout, retries=retries)
        if rows is None:
            continue           # bu tarih geçici hata → diğer günlere devam
        for rec in rows:
            if rec.symbol == sym:
                return rec
    return None


def clear_cache() -> None:
    """Modül-içi tarih cache'ini temizler (test/manuel refresh)."""
    _DATE_CACHE.clear()
