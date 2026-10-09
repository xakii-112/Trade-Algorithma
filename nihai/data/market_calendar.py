"""
market_calendar.py — Borsa takvimi, RTH, DST, tatil yönetimi.

PDF Referansları:
- Madde 22 : Seans saati / oturum yönetimi (RTH 09:30–16:00 ET)
- Madde 43 : Timezone & DST (zoneinfo ile ET'ye sabitleme)
- Madde 26 : Olay takvimi farkındalığı (placeholder — bilanço API'si sonra eklenir)

exchange-calendars kullanılır (NYSE/XNYS).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

ET = ZoneInfo("America/New_York")
NYSE = xcals.get_calendar("XNYS")

# Madde 5 — açılış ilk 5 dk anormal etiketlenir
OPENING_VOLATILITY_WINDOW = timedelta(minutes=5)
# Madde 33 — MOC kesim ~15:50 ET
MOC_CUTOFF = time(15, 50)


def now_et() -> datetime:
    return datetime.now(tz=ET)


def now_et_naive() -> datetime:
    """ET duvar-saati, NAIVE (tzinfo yok). Sistem geneli naive datetime kullanır;
    ABD borsasını izlediğimiz için TARİH-bazlı mantık (wash-sale 30g, gün-içi işlem,
    T+1 takas, günlük reset) yerel saat yerine ET tarihine göre olmalı — yoksa
    Türkiye gibi UTC+3'te gece-yarısı sonrası tarih 1 gün kayar."""
    return now_et().replace(tzinfo=None)


def _et(dt: datetime | None) -> datetime:
    """`dt` → ET. Saat dilimsiz zaman ET duvar saatidir (`now_et_naive` sözleşmesi).

    Tur-106 (BULGU-106-8): her fonksiyon `dt.astimezone(ET)` yapıyordu; Python saat dilimsiz zamanı orada
    makinenin YEREL saati (TR, UTC+3) sayar → ET-naive 15:58 "08:58 ET, seans kapalı" okunurdu."""
    if dt is None:
        return now_et()
    return dt.replace(tzinfo=ET) if dt.tzinfo is None else dt.astimezone(ET)


def is_market_open(dt: datetime | None = None) -> bool:
    """Madde 22 — RTH içinde mi? Tatil değil mi? DST hesaba katılır."""
    dt_et = _et(dt)
    try:
        return bool(NYSE.is_open_on_minute(dt_et.replace(second=0, microsecond=0)))
    except Exception:
        return False


def kapanmis_gunluk(df, dt: datetime | None = None):
    """Günlük bar serisinden seans AÇIKKEN oluşmakta olan BUGÜN barını at (Madde 3'ün 1d hâli).

    Tur-106 (BULGU-106-17): akış sağlayıcısı 1d'de bugünün yarım barını bilerek ekler (Madde 3 dışlaması yalnız
    gün içi aralıklarda). Canlı quant günlük seriyi ET günü başına BİR kez önbelleğe aldığı için günün kararı ilk
    çekim anındaki fiyata donuyordu — süreç başlatma saatine bağlı — ve ölçülen kuraldan (karar T−1 kapanışında,
    giriş T'de; Tur-101) sembol-günlerin %5,35'inde ayrışıyordu. Kapanıştan sonra bugünün barı kapanmıştır, korunur.
    Etiket günün kendisidir (Alpaca 04:00 UTC = 00:00 ET; hub saat dilimsiz verir)."""
    dt_et = _et(dt)
    if df is None or len(df) == 0 or not is_market_open(dt_et):
        return df
    return df.iloc[:-1] if df.index[-1].date() >= dt_et.date() else df


def ilk_bar_ani(hedef):
    """`hedef` andan SONRA bar oluşabilecek ilk seans dakikası (UTC `pd.Timestamp`): seans içindeyse hedef + 1 dk,
    değilse sonraki açılış. Saat dilimsiz `hedef` bu modülün sözleşmesiyle ET'dir. Tur-106 (BULGU-106-20): 1m
    tahmin bu andan önce çözülemez (`prediction_tracker._forward_price` hedef sonrası bar ister, IEX 1m yalnız
    seansta bar üretir) — gece/hafta sonu bekleyen kapanış kuyruğu için bar çekmek boşunadır."""
    import pandas as pd

    h = pd.Timestamp(hedef)
    h = (h.tz_localize(ET) if h.tzinfo is None else h).tz_convert("UTC").floor("min")
    ilk = h + pd.Timedelta(minutes=1)
    return ilk if NYSE.is_open_on_minute(ilk) else NYSE.next_open(h)


def in_opening_volatility_window(dt: datetime | None = None) -> bool:
    """Madde 5, 22 — açılışın ilk 5 dakikası (yeni giriş için 'anormal' bant)."""
    dt_et = _et(dt)
    if not is_market_open(dt_et):
        return False
    sched = NYSE.schedule.loc[dt_et.date().isoformat()]
    open_et = sched["open"].to_pydatetime().astimezone(ET)
    return open_et <= dt_et < open_et + OPENING_VOLATILITY_WINDOW


def past_moc_cutoff(dt: datetime | None = None) -> bool:
    """Madde 33 — kapanışa yakın yeni giriş yapılmaz."""
    dt_et = _et(dt)
    return dt_et.time() >= MOC_CUTOFF


def can_enter_new_position(dt: datetime | None = None) -> tuple[bool, str]:
    """Madde 22 — yeni giriş şartı: RTH + açılış 5 dk geçmiş + MOC öncesi."""
    dt = _et(dt)
    if not is_market_open(dt):
        return False, "MARKET_CLOSED"
    if in_opening_volatility_window(dt):
        return False, "OPENING_VOLATILITY_5MIN"
    if past_moc_cutoff(dt):
        return False, "PAST_MOC_CUTOFF"
    return True, "OK"


def session_remaining(dt: datetime | None = None) -> timedelta | None:
    """Şu anki seansın kapanışına ne kadar var? Kapalıysa None."""
    dt_et = _et(dt)
    if not is_market_open(dt_et):
        return None
    sched = NYSE.schedule.loc[dt_et.date().isoformat()]
    close_et = sched["close"].to_pydatetime().astimezone(ET)
    return close_et - dt_et


def son_seans_acilisi_yerel(dt: datetime | None = None, pay_dk: int = 30) -> str:
    """`dt`den en az `pay_dk` dakika önce AÇILMIŞ son seansın açılış anı: makinenin YEREL duvar saati, saat dilimsiz
    ISO (saniye). Yerel saatle yazılan günlük damgalarıyla (`logged_at`) sözlük sırasıyla kıyaslanır.

    Tur-106 (BULGU-106-36): gece özeti canlı özellik satırlarını yerel TAKVİM gününden sayıyordu. Seans açılmadan
    (12:00 koşusu, hafta sonu) ya da yerel gece yarısından sonra (kış saatinde kapanış TR 00:00) koşan özet için
    "bugün" boş bir gündür: satır sayısı 0 görünür, "artmadı" uyarısı hiç konuşamaz. Sorunun birimi gün değil
    SEANSTIR. `pay_dk`: yeni açılan seans henüz satır üretmemiş olabilir; o süre boyunca önceki seans raporlanır."""
    import pandas as pd

    an = pd.Timestamp(_et(dt)).tz_convert("UTC").floor("min") - pd.Timedelta(minutes=pay_dk)
    acilis = NYSE.previous_open(an).to_pydatetime()
    return acilis.astimezone().replace(tzinfo=None).isoformat(timespec="seconds")


def local_session_window(dt: datetime | None = None) -> dict | None:
    """ABD seansının YEREL (makine saati) açılış/kapanışı + DST kayması (Tur-58, Madde 43).

    Dış rapor §G'nin gerçek kalıntısı budur: karar yolu zaten ET'ye sabitlenmiştir
    (`is_market_open`/`can_enter_new_position` zoneinfo kullanır), ama **Windows Task
    Scheduler tetikleri yerel duvar-saatidir**. ABD ve TR yaz saatine farklı tarihlerde
    geçtiği için "09:30'da başlat" gibi sabit bir tetik yılda iki kez bir saat kayar.
    Kayan şey KARAR değil ZAMANLAMADIR; yine de bir saatlik seans dilimi kaybettirir.

    Onarım kayan tetiği tahmin etmek değil, tetiği ÖLÇÜLEN seanstan TÜRETMEKTİR:

        python -m nihai.data.market_calendar          # bugünün yerel penceresi
        python -m nihai.data.market_calendar --at     # yalnız "HH:MM" (script'ler için)

    Piyasa o gün kapalıysa (hafta sonu/tatil) **None** döner — uydurma bir pencere
    üretmez; çağıran bir sonraki işlem gününe bakmalıdır."""
    dt_et = _et(dt)
    try:
        sched = NYSE.schedule.loc[dt_et.date().isoformat()]
    except Exception:
        return None
    open_et = sched["open"].to_pydatetime().astimezone(ET)
    close_et = sched["close"].to_pydatetime().astimezone(ET)
    open_local = open_et.astimezone()
    close_local = close_et.astimezone()
    return {
        "date": dt_et.date().isoformat(),
        "open_et": open_et.strftime("%H:%M"), "close_et": close_et.strftime("%H:%M"),
        "open_local": open_local.strftime("%H:%M"), "close_local": close_local.strftime("%H:%M"),
        "local_tz": str(open_local.tzinfo),
        # ET ile yerel saat arasındaki fark (saat). DST geçişlerinde 7 ↔ 8 arasında oynar.
        "offset_hours": (open_local.utcoffset().total_seconds()
                         - open_et.utcoffset().total_seconds()) / 3600.0,
        "duration_hours": (close_et - open_et).total_seconds() / 3600.0,
    }


def main() -> None:
    import argparse

    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="ABD seansının yerel penceresi (Madde 43)")
    ap.add_argument("--at", action="store_true", help="yalnız yerel açılış saatini bas (HH:MM)")
    ap.add_argument("--duration", action="store_true", help="yalnız seans uzunluğunu bas (saat)")
    ap.add_argument("--kalan-sn", action="store_true",
                    help="seans açıksa kapanışa kalan saniye (tam sayı), kapalıysa -1 (gece görevinin seans kapısı)")
    a = ap.parse_args()
    if a.kalan_sn:
        kalan = session_remaining()
        print(-1 if kalan is None else max(1, int(kalan.total_seconds())))
        return
    w = local_session_window()
    if w is None:
        if a.at or a.duration:
            raise SystemExit(1)          # script'ler sessiz bir varsayılana düşmesin
        print("Bugün ABD piyasası KAPALI (hafta sonu/tatil) — pencere yok.")
        return
    if a.at:
        print(w["open_local"])
        return
    if a.duration:
        print(f"{w['duration_hours']:.2f}")
        return
    print(f"{w['date']}  ET {w['open_et']}-{w['close_et']}  →  "
          f"YEREL {w['open_local']}-{w['close_local']} ({w['local_tz']}, "
          f"fark {w['offset_hours']:+.0f} sa, {w['duration_hours']:.1f} sa seans)")
    print("Zamanlanmış görev tetiği bu YEREL saatten türetilmelidir; ABD ve TR yaz "
          "saatine farklı tarihlerde geçer (yılda 2 kez 1 saatlik kayma).")


if __name__ == "__main__":
    main()
