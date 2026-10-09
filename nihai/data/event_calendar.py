"""
event_calendar.py — Olay/bilanço/temettü takvimi.

PDF Referansları:
- Madde 26 : Olay/bilanço takvimi farkındalığı
- Madde 17 : IV crush koruması (planlı olay öncesi şişmiş IV → long opsiyon yasak)
- Madde 28 : Ex-dividend yakını short call atama riski

Veri kaynağı (Madde 26): BİRİNCİL yfinance (Ticker.earnings_dates + Ticker.calendar),
İKİNCİL keysiz Nasdaq takvimi (nasdaq_calendar.py). yfinance earnings boş dönerse VEYA
hata verirse Nasdaq fallback devreye girer + BMO/AMC zamanlamasını (opsiyonel `timing`)
ekler. Hangi kaynağın kullanıldığı `source` alanına yazılır ve loglanır.

Persistent cache: data/events/<SYMBOL>.json — başarı → 24 saat TTL. Hata-yapışması yasağı
(BULGU-14): her iki kaynak da başarısızsa 24h cache'e YAZILMAZ; bunun yerine 15 dk'lık
NEGATİF cache yazılır (source="hata") → süresi dolunca yeniden denenir (geçici ağ hatası
earnings guard'ını 24 saat sessizce kapatmaz).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import yfinance as yf
from loguru import logger

from nihai.config import DATA_DIR
from nihai.data.nasdaq_calendar import get_next_earnings as _nasdaq_next_earnings

EVENTS_DIR = DATA_DIR / "events"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 24
NEG_CACHE_TTL_MIN = 15   # Hata-yapışması yasağı (BULGU-14): geçici hata → kısa-TTL negatif cache
_NEG_SOURCE = "hata"     # source alanı bu değerse kayıt negatif-cache'tir (15 dk TTL)
# Süreç-içi memo: aynı tarama turunda event_guard + corporate_action_guard +
# iv_crush_guard aynı sembol için get_company_events'i tekrar tekrar çağırıp her
# seferinde JSON dosyasını okuyordu. 1 saatlik in-proc memo dosya IO'sunu keser.
_MEM: dict[str, tuple[datetime, CompanyEvents]] = {}
_MEM_TTL_SEC = 3600


@dataclass
class CompanyEvents:
    symbol: str
    next_earnings: date | None = None
    next_ex_dividend: date | None = None
    last_split: date | None = None        # Madde 25 — son hisse bölünmesi tarihi
    fetched_at: datetime | None = None
    source: str = "yfinance"
    # OPSİYONEL (geriye-uyumlu additive alan): bilanço zamanlaması "BMO"|"AMC"|None.
    # Yalnız Nasdaq ikinci kaynağı doldurur; yfinance-yalnız yolda None kalır.
    timing: str | None = None
    # Tur-58 (Madde 25) — SON ÖDENEN temettü tutarı ($/hisse). YAKLAŞMAKTA OLAN temettünün
    # tutarı DEĞİLDİR: yfinance takvimi tutar vermez, `dividends` serisi yalnız ÖDENMİŞLERİ
    # taşır. Vekil olduğu için ad da öyle; erken-kullanım eşiğinde `measured` bayrağıyla
    # birlikte taşınır (Tur-34: ölçüm KOŞULU ölçümle birlikte gider). Eski cache'lerde
    # alan yoktur → None = "bilinmiyor" (0.0 DEĞİL; sıfır "temettü yok" demek olurdu).
    last_dividend_amount: float | None = None


def _cache_path(symbol: str) -> Path:
    return EVENTS_DIR / f"{symbol.upper()}.json"


def _load_cache(symbol: str) -> CompanyEvents | None:
    p = _cache_path(symbol)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(data["fetched_at"])
        src = data.get("source", "yfinance")
        # Negatif-cache kayıtları kısa (15 dk) TTL; başarı kayıtları 24 saat.
        ttl = timedelta(minutes=NEG_CACHE_TTL_MIN) if src == _NEG_SOURCE \
            else timedelta(hours=CACHE_TTL_HOURS)
        if datetime.now() - fetched > ttl:
            return None
        return CompanyEvents(
            symbol=data["symbol"],
            next_earnings=date.fromisoformat(data["next_earnings"]) if data.get("next_earnings") else None,
            next_ex_dividend=date.fromisoformat(data["next_ex_dividend"]) if data.get("next_ex_dividend") else None,
            last_split=date.fromisoformat(data["last_split"]) if data.get("last_split") else None,
            fetched_at=fetched,
            source=src,
            timing=data.get("timing"),
            last_dividend_amount=data.get("last_dividend_amount"),
        )
    except Exception as e:
        logger.debug(f"[event] cache okuma hatasi {symbol}: {e}")
        return None


def _save_cache(events: CompanyEvents) -> None:
    p = _cache_path(events.symbol)
    payload = {
        "symbol": events.symbol,
        "next_earnings": events.next_earnings.isoformat() if events.next_earnings else None,
        "next_ex_dividend": events.next_ex_dividend.isoformat() if events.next_ex_dividend else None,
        "last_split": events.last_split.isoformat() if events.last_split else None,
        "fetched_at": (events.fetched_at or datetime.now()).isoformat(),
        "source": events.source,
        "timing": events.timing,
        "last_dividend_amount": events.last_dividend_amount,
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fetch_yfinance_events(sym: str) -> tuple[date | None, date | None, date | None, bool, float | None]:
    """BİRİNCİL kaynak. Dönüş: (next_earnings, next_ex_div, last_split, earnings_ok, last_div).

    ``earnings_ok`` = earnings sorgusu AĞ HATASI vermeden tamamlandı (boş sonuç de OK).
    False ise earnings kaynağı erişilemedi → ikinci-kaynak boşsa hata-yapışması yasağı
    (BULGU-14) devreye girer. Ex-div/split hataları earnings_ok'u düşürmez (Madde 26'da
    earnings kritik alandır)."""
    next_earnings: date | None = None
    next_ex_div: date | None = None
    last_split: date | None = None
    last_div: float | None = None
    earnings_ok = True

    try:
        tkr = yf.Ticker(sym)

        # Earnings
        try:
            edates = tkr.earnings_dates
            if edates is not None and not edates.empty:
                today = date.today()
                future_idx = [d.date() if hasattr(d, "date") else d for d in edates.index]
                future = sorted([d for d in future_idx if d >= today])
                if future:
                    next_earnings = future[0]
        except Exception as e:
            earnings_ok = False   # earnings erişilemedi (rate-limit/ağ) → fallback + BULGU-14
            logger.debug(f"[event] earnings_dates {sym}: {e}")

        # Ex-dividend
        try:
            cal = tkr.calendar
            ex_key_candidates = ["Ex-Dividend Date", "Ex Dividend Date", "exDividendDate"]
            if isinstance(cal, dict):
                for k in ex_key_candidates:
                    if k in cal and cal[k]:
                        v = cal[k]
                        if isinstance(v, list) and v:
                            v = v[0]
                        if isinstance(v, datetime):
                            next_ex_div = v.date()
                        elif isinstance(v, date):
                            next_ex_div = v
                        break
        except Exception as e:
            logger.debug(f"[event] calendar {sym}: {e}")

        # Madde 25 — Son hisse bölünmesi (split)
        try:
            splits = tkr.splits
            if splits is not None and not splits.empty:
                last_idx = splits.index[-1]
                last_split = last_idx.date() if hasattr(last_idx, "date") else None
        except Exception as e:
            logger.debug(f"[event] splits {sym}: {e}")

        # Tur-58 (Madde 25) — SON ÖDENEN temettü tutarı. Erken-kullanım eşiği bir TUTAR
        # ister; ex-div TARİHİ tek başına eşiği kuramaz. Hata/boş seri → None kalır ve
        # eşik "ölçülmedi" dalına düşer (uydurma 0.0 yazılmaz).
        try:
            divs = tkr.dividends
            if divs is not None and not divs.empty:
                v = float(divs.iloc[-1])
                last_div = v if v > 0 else None
        except Exception as e:
            logger.debug(f"[event] dividends {sym}: {e}")

    except Exception as e:
        logger.warning(f"[event] yfinance fetch hatasi {sym}: {e}")
        return None, None, None, False, None

    return next_earnings, next_ex_div, last_split, earnings_ok, last_div


def _nasdaq_fallback(sym: str) -> tuple[date | None, str | None]:
    """İKİNCİL kaynak (Madde 26). Nasdaq keysiz takviminden (next_earnings, timing).

    Ağ hatası/eşleşme yok → (None, None). KENAR eklemez; yalnız earnings teyidi + zamanlama."""
    try:
        nq = _nasdaq_next_earnings(sym)
    except Exception as e:   # nasdaq istemcisi kendi içinde yutuyor; bu ekstra güvenlik
        logger.debug(f"[event] nasdaq fallback {sym}: {e}")
        return None, None
    if nq is None:
        return None, None
    return nq.date, nq.timing


def get_company_events(symbol: str, *, refresh: bool = False) -> CompanyEvents:
    """BİRİNCİL yfinance + (boş/hata → ) İKİNCİL Nasdaq fallback; başarı 24h, hata 15dk cache."""
    sym = symbol.upper()
    if not refresh:
        memo = _MEM.get(sym)
        if memo and (datetime.now() - memo[0]).total_seconds() < _MEM_TTL_SEC:
            return memo[1]
        cached = _load_cache(sym)
        if cached is not None:
            # Negatif-cache kaydını _MEM'e KOYMA: 1h memo, 15dk negatif TTL'i uzatırdı.
            if cached.source != _NEG_SOURCE:
                _MEM[sym] = (datetime.now(), cached)
            return cached

    next_earnings, next_ex_div, last_split, earnings_ok, last_div = _fetch_yfinance_events(sym)
    source = "yfinance"
    timing: str | None = None

    # Madde 26 — İKİNCİ KAYNAK: yfinance earnings boş VEYA hata → Nasdaq keysiz takvim.
    # (yfinance earnings dolduysa fallback ÇAĞRILMAZ — gereksiz istek yok.)
    if next_earnings is None:
        nq_date, nq_timing = _nasdaq_fallback(sym)
        if nq_date is not None:
            next_earnings = nq_date
            timing = nq_timing
            source = "nasdaq" if not earnings_ok else "yfinance+nasdaq"
            logger.info(
                f"[event] {sym} earnings ikinci kaynak (nasdaq): {next_earnings} timing={timing}")

    now = datetime.now()

    # Hata-yapışması yasağı (BULGU-14): earnings kaynağı HATA verdi ve ikinci kaynak da
    # doldurmadıysa uzun-ömürlü 24h cache'e YAZMA (geçici ağ hatası earnings guard'ını 24
    # saat sessizce kapatmasın). 15 dk negatif cache yaz → süresi dolunca yeniden dene.
    if not earnings_ok and next_earnings is None:
        neg = CompanyEvents(symbol=sym, fetched_at=now, source=_NEG_SOURCE)
        _save_cache(neg)
        logger.warning(f"[event] {sym} earnings her iki kaynakta başarısız → {NEG_CACHE_TTL_MIN}dk negatif cache")
        return neg   # _MEM'e KOYMA (kısa ömürlü)

    events = CompanyEvents(
        symbol=sym, next_earnings=next_earnings, next_ex_dividend=next_ex_div,
        last_split=last_split, fetched_at=now, source=source, timing=timing,
        last_dividend_amount=last_div,
    )
    _save_cache(events)
    _MEM[sym] = (now, events)
    return events


def days_to_next_earnings(symbol: str) -> int | None:
    e = get_company_events(symbol)
    if e.next_earnings is None:
        return None
    return (e.next_earnings - date.today()).days


def days_to_next_ex_dividend(symbol: str) -> int | None:
    e = get_company_events(symbol)
    if e.next_ex_dividend is None:
        return None
    return (e.next_ex_dividend - date.today()).days


def ex_dividend_context(symbol: str) -> tuple[int | None, float | None]:
    """(ex-güne kalan gün, son ödenen temettü $/hisse) — TEK çağrıda (Tur-58).

    İki ayrı çağrı iki ayrı `get_company_events` demek olurdu; memo TTL sınırında ikisi
    farklı anlık görüntüye düşebilir ve erken-kullanım eşiği tutarsız bir çift okurdu."""
    e = get_company_events(symbol)
    days = (e.next_ex_dividend - date.today()).days if e.next_ex_dividend else None
    return days, e.last_dividend_amount


# ---------------------------------------------------------------------------
# Guard fonksiyonları
# ---------------------------------------------------------------------------
def event_guard(symbol: str, *, earnings_buffer_days: int = 2,
                ex_div_buffer_days: int = 1) -> tuple[bool, str]:
    """
    Sinyal-anı kontrolu. (True, reason) = guard aktif (yeni giriş yapma).

    Madde 26 — Earnings <= buffer gün → tüm yön sinyali bloke.
    """
    e = get_company_events(symbol)
    if e.next_earnings:
        delta = (e.next_earnings - date.today()).days
        if 0 <= delta <= earnings_buffer_days:
            return True, f"EARNINGS_IN_{delta}_DAYS_{e.next_earnings}"
    if e.next_ex_dividend:
        delta = (e.next_ex_dividend - date.today()).days
        if 0 <= delta <= ex_div_buffer_days:
            return True, f"EX_DIV_IN_{delta}_DAYS_{e.next_ex_dividend}"
    return False, "CLEAR"


def corporate_action_guard(symbol: str, *, lookback_days: int | None = None) -> tuple[bool, str]:
    """
    Madde 25 — Son N iş günü içinde hisse bölünmesi olduysa fiyat sıçraması
    gerçek hareket değildir; bu pencerede yeni sinyal şüpheyle karşılanır.

    (True, reason) = guard aktif (split yakını, dikkat).
    """
    from nihai.config import CONFIG
    days = lookback_days if lookback_days is not None else \
        CONFIG.market_structure.corporate_action_lookback_days
    e = get_company_events(symbol)
    if e.last_split:
        delta = (date.today() - e.last_split).days
        if 0 <= delta <= days:
            return True, f"RECENT_SPLIT_{delta}_DAYS_AGO_{e.last_split}"
    return False, "CLEAR"


def iv_crush_guard_for_option(symbol: str, *, earnings_proximity_days: int = 7) -> tuple[bool, str]:
    """
    Madde 17 — Bilanço öncesi şişmiş IV ile uzun opsiyon alma.

    Earnings <= proximity gün → long opsiyon bloke (IV crush riski).
    """
    e = get_company_events(symbol)
    if e.next_earnings:
        delta = (e.next_earnings - date.today()).days
        if 0 <= delta <= earnings_proximity_days:
            return True, f"IV_CRUSH_RISK_EARNINGS_IN_{delta}_DAYS"
    return False, "CLEAR"
