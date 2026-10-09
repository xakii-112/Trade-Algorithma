"""
macro_regime.py — MAKRO REJİM PANELİ (Tur-75). KARAR-DIŞI.

Dış geliştirme listesinin BÖLÜM 8'i (Madde 151-175) sistemde en büyük boşluktu: FRED
köprüsü (`fred_data.py`) Tur-51'den beri VARDI ama yalnız İKİ seri tüketiliyordu
(`T10Y2Y`, `NFCI` — `edge_lab/build_babmacro_dataset.py`). Bu modül o köprünün ÜSTÜNE
listenin ölçülebilir makro göstergelerini kurar.

ANAYASA: KARAR-DIŞI. Buradaki hiçbir seri sinyale/eşiğe/META_GATE'e bağlanmaz. Listenin
her maddesi bir KURAL öneriyor ("net likidite düşerken long tavanı kısılmalı" — Madde 151);
o kurallar BURAYA YAZILMAZ. Kural bir kenar iddiasıdır ve kenar iddiası `edge_lab`
kapısından geçmeden koda giremez (ACI DERS #3/#5). Bu modül yalnız SAYIYI üretir.

## İki tuzak — ikisi de ölçüldü, ikisi de burada kapatıldı

**(1) BİRİM TUZAĞI (Madde 151).** Net likidite = WALCL − WTREGEN − RRPONTSYD. FRED'de
WALCL ve WTREGEN **milyon** dolar, RRPONTSYD **milyar** dolardır. Naif çıkarma ters repoyu
1000 kat küçük sayar: 2026-09-02'de ham fark 5.769.269 (milyon $) çıkar, doğrusu
5.769.269 − 675 = 5.768.594. Fark bugün küçük ama RRP 2023'te ~2.500 milyar $'dı; o
dönemde naif hesap likiditeyi **2,5 TRİLYON dolar fazla** gösterirdi. Bu bir yuvarlama
değil, işaretin yönünü değiştiren bir hatadır.

**(2) YAYIN GECİKMESİ.** FRED'in `observation_date`'i verinin AİT OLDUĞU gündür, BİLİNDİĞİ
gün değildir. WALCL çarşamba gözlemi perşembe 16:30 ET'de yayımlanır; SAHMREALTIME'ın
"2026-08-01" satırı 2026-09-05'te doğar. Gözlem tarihini bilinme tarihi saymak, ACI DERS
#7'nin (exec-lag artefaktı) makro versiyonudur — model geleceği okur. `PUBLICATION_LAG`
her seriyi kendi gecikmesi kadar İLERİ kaydırır; `panel()` yalnız kaydırılmış seriyi
işlem günü ızgarasına taşır. Gecikmeyi kapatmak isteyen `lag=False` demek zorundadır ve
o zaman ne yaptığını bilir.

Üçüncü bir küçük tuzak: `NROU`/`GDPPOT` CBO **projeksiyonlarıdır** ve 2036'ya kadar satır
taşırlar. Bugünden sonraki satırlar `panel()`e girmez (`_bugune_kadar`).

## Kapsanan maddeler

| Madde | Fonksiyon | Seri |
|---|---|---|
| 151 Fed net likidite | `net_liquidity` | WALCL, WTREGEN, RRPONTSYD |
| 153 USD/JPY carry çözülmesi | `usdjpy_shock` | DEXJPUS |
| 154 ACM vade primi | `term_premium` | THREEFYTP10 |
| 155 NFCI finansal koşullar | `financial_conditions` | NFCI |
| 158 WTI şoku | `oil_shock` | DCOILWTICO |
| 160 Sahm kuralı | `sahm_gap` | SAHMREALTIME |
| 161 Taylor kuralı sapması | `taylor_rule_gap` | DFF, PCEPILFE, UNRATE, NROU |
| 164 DXY baskısı | `dollar_momentum` | DTWEXBGS |
| 166 Breakeven enflasyon | `breakeven_inflation` | T10YIE |
| 167 Kredi riski makası | `credit_spread` | BAMLH0A0HYM2, BAMLC0A0CM |
| 169 Getiri eğrisi dikleşmesi | `curve_slope` | T10Y2Y |
| 170 CP − T-Bill makası | `cp_tbill_spread` | RIFSPPFAAD90NB, DTB3 |
| 174 SOFR − EFFR basisi | `sofr_effr_basis` | SOFR, EFFR |

Madde 152/157/159/162/163/165/168/171/172/173/175 için gerekçeli ret: `docs/TUR75_RAPORU`.

Sözleşme `fred_data`/`cboe_data` ile aynı: yalnız veri taşır, karar vermez, ağ hatasında
fail-soft (boş seri/None) döner, istisna fırlatmaz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from nihai.data import fred_data

# Serinin AİT OLDUĞU günden BİLİNDİĞİ güne kadar geçen TAKVİM günü (muhafazakâr üst sınır).
# Kaynak: FRED sürüm takvimleri. Bilinmeyen seri için varsayılan 1 gün (günlük seri).
PUBLICATION_LAG: dict[str, int] = {
    # Haftalık Fed bilançosu: çarşamba gözlemi, perşembe 16:30 ET yayın.
    "WALCL": 1,
    "WTREGEN": 1,
    # Günlük ama T+1 yayın (NY Fed operasyon verisi, ertesi sabah).
    "RRPONTSYD": 1,
    "SOFR": 1,
    "EFFR": 1,
    # Hazine/piyasa günlük serileri: aynı gün kapanışından sonra → T+1'de güvenli.
    "DGS2": 1, "DGS10": 1, "T10Y2Y": 1, "T10YIE": 1, "DTB3": 1, "DFF": 1,
    "DTWEXBGS": 1, "DEXJPUS": 1, "DCOILWTICO": 1, "RIFSPPFAAD90NB": 1,
    "THREEFYTP10": 1, "BAMLH0A0HYM2": 1, "BAMLC0A0CM": 1,
    # NFCI: cuma biten hafta, ertesi ÇARŞAMBA yayın.
    "NFCI": 5,
    # Aylık istihdam: ayın 1'i gözlemi, ertesi ayın ilk cuması yayın (≈35 gün).
    "UNRATE": 35,
    "SAHMREALTIME": 35,
    # Aylık PCE: ay sonundan ~30 gün sonra → gözlem tarihinden ≈60 gün.
    "PCEPILFE": 60,
    # Çeyreklik CBO projeksiyonu: çeyrek başı gözlemi, ~1 çeyrek sonra yayın.
    "NROU": 100,
    "GDPPOT": 100,
}
DEFAULT_LAG_DAYS = 1

# Ileri-doldurma sınırı: seri FREKANSINDAN uzun boşluk doldurulmaz. Sınırsız ffill,
# aylık bir seriyi günlük gibi göstererek "her gün taze veri var" yalanını üretir.
FFILL_LIMIT: dict[str, int] = {
    "WALCL": 10, "WTREGEN": 10, "NFCI": 10,
    "UNRATE": 45, "SAHMREALTIME": 45, "PCEPILFE": 70,
    "NROU": 130, "GDPPOT": 130,
}
DEFAULT_FFILL_LIMIT = 5

# Taylor (1993) katsayıları — ön-kayıtlı, TARANMADI.
TAYLOR_R_STAR = 2.0        # denge reel faizi (%)
TAYLOR_PI_STAR = 2.0       # Fed enflasyon hedefi (%)
TAYLOR_PI_COEF = 0.5       # enflasyon açığı katsayısı
TAYLOR_GAP_COEF = 0.5      # çıktı açığı katsayısı
OKUN_COEF = 2.0            # Okun yasası: 1 puan işsizlik açığı ≈ 2 puan çıktı açığı


def _bugune_kadar(s: pd.Series, *, today: pd.Timestamp | None = None) -> pd.Series:
    """Bugünden SONRAKİ satırları at (CBO projeksiyonları 2036'ya kadar satır taşır).

    `pd.Timestamp.utcnow()` TZ-BİLİNÇLİdir, FRED indeksi ise naive'dir; ikisini
    doğrudan kıyaslamak TypeError verir. Sınır her iki tarafta da naive'e indirilir.
    """
    if s.empty:
        return s
    ham = pd.Timestamp(today) if today is not None else pd.Timestamp.now("UTC")
    sinir = (ham.tz_localize(None) if ham.tzinfo is not None else ham).normalize()
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
        s = pd.Series(s.to_numpy(), index=idx, name=s.name)
    return s[idx <= sinir]


def _lagged(series: str, *, since: str, lag: bool = True,
            today: pd.Timestamp | None = None, fetch: bool = True) -> pd.Series:
    """Seriyi getir, gelecek satırlarını at, YAYIN GECİKMESİ kadar ileri kaydır.

    Kaydırma indeksin KENDİSİNE uygulanır (değere değil): 2026-08-01 gözlemi
    35 gün gecikmeyle 2026-09-05'te "bilinir" hale gelir.
    """
    s = fred_data.get(series, since=since, fetch=fetch)
    if s is None or len(s) == 0:
        return pd.Series(dtype=float, name=series)
    s = _bugune_kadar(pd.Series(s).dropna(), today=today)
    if s.empty:
        return pd.Series(dtype=float, name=series)
    if lag:
        gun = PUBLICATION_LAG.get(series, DEFAULT_LAG_DAYS)
        s.index = pd.DatetimeIndex(s.index) + pd.Timedelta(days=gun)
    s.name = series
    return _bugune_kadar(s, today=today)


def _causal_ffill(s: pd.Series, grid: pd.DatetimeIndex, max_days: int) -> pd.Series:
    """Nedensel ileri-doldurma — sınır SATIR değil TAKVİM GÜNÜdür.

    `pandas`ın `ffill(limit=N)`'i ardışık N **satırı** doldurur. Izgara seyrekse
    (aylık kontrol noktaları, tatil delikleri) N satır aylara karşılık gelir ve sınır
    HİÇ ISIRMAZ: tek bir temmuz gözlemi ağustos satırını doldurur, panel "her gün
    taze makro veri var" yalanını üretir. Burada her hedef gün için kendisinden
    ÖNCEKİ (veya aynı gündeki) son gözlem bulunur ve YAŞI `max_days`i aşarsa NaN kalır.
    """
    if s.empty or len(grid) == 0:
        return pd.Series(np.nan, index=grid, dtype=float)
    src = s.sort_index()
    src_idx = pd.DatetimeIndex(src.index)
    pos = src_idx.searchsorted(grid, side="right") - 1
    out = np.full(len(grid), np.nan)
    gecerli = pos >= 0
    if gecerli.any():
        yerler = np.where(gecerli)[0]
        idx = pos[gecerli]
        yas = (grid[gecerli] - src_idx[idx]).days
        taze = np.asarray(yas) <= max_days
        out[yerler[taze]] = src.to_numpy(dtype=float)[idx[taze]]
    return pd.Series(out, index=grid, dtype=float)


def _to_grid(s: pd.Series, grid: pd.DatetimeIndex, *, series: str | None = None) -> pd.Series:
    """Seriyi işlem günü ızgarasına taşı — SINIRLI nedensel doldurma (bkz. FFILL_LIMIT)."""
    limit = FFILL_LIMIT.get(series or str(s.name), DEFAULT_FFILL_LIMIT)
    return _causal_ffill(s, grid, limit)


def _zscore(s: pd.Series, window: int) -> pd.Series:
    """Yürüyen z-skoru — pencere DOLMADAN değer üretmez (min_periods=window)."""
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=1)
    return (s - mu) / sd.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# Madde 151 — Fed net likidite
# ---------------------------------------------------------------------------
def net_liquidity(*, since: str = "2015-01-01", lag: bool = True,
                  today: pd.Timestamp | None = None, fetch: bool = True) -> pd.DataFrame:
    """WALCL − WTREGEN − RRPONTSYD, HEPSİ MİLYON DOLARA çevrilerek (Madde 151).

    RRPONTSYD FRED'de MİLYAR dolardır; 1000 ile çarpılmadan çıkarılırsa ters repo
    1000 kat küçük sayılır (modül docstring'indeki ölçüm). Dönen kolonlar:
    `walcl`, `tga`, `rrp`, `net_liquidity` (milyon $) ve `net_liq_chg_20` (20 günlük fark).
    Seri yoksa BOŞ DataFrame (fail-soft) — sıfır DEĞİL, çünkü sıfır bir ölçümdür.
    """
    walcl = _lagged("WALCL", since=since, lag=lag, today=today, fetch=fetch)
    tga = _lagged("WTREGEN", since=since, lag=lag, today=today, fetch=fetch)
    rrp = _lagged("RRPONTSYD", since=since, lag=lag, today=today, fetch=fetch)
    if walcl.empty:
        logger.warning("[makro] WALCL boş — net likidite hesaplanamadı")
        return pd.DataFrame()
    grid = pd.DatetimeIndex(sorted(set(walcl.index) | set(tga.index) | set(rrp.index)))
    d = pd.DataFrame(index=grid)
    d["walcl"] = _to_grid(walcl, grid, series="WALCL")
    d["tga"] = _to_grid(tga, grid, series="WTREGEN")
    d["rrp"] = _to_grid(rrp, grid, series="RRPONTSYD") * 1000.0   # milyar → milyon
    d["net_liquidity"] = d["walcl"] - d["tga"] - d["rrp"]
    d["net_liq_chg_20"] = d["net_liquidity"] - d["net_liquidity"].shift(20)
    return d.dropna(subset=["net_liquidity"])


# ---------------------------------------------------------------------------
# Madde 161 — Taylor kuralı sapması
# ---------------------------------------------------------------------------
def taylor_rule_gap(*, since: str = "2015-01-01", lag: bool = True,
                    today: pd.Timestamp | None = None, fetch: bool = True) -> pd.DataFrame:
    """Taylor (1993) kuralının önerdiği faiz ile fiili EFFR arasındaki sapma (Madde 161).

        i* = r* + π + 0.5·(π − π*) + 0.5·(çıktı açığı)
        çıktı açığı ≈ Okun · (u* − u)          [u = UNRATE, u* = NROU]
        π = çekirdek PCE'nin 12 aylık yüzde değişimi

    `gap = policy_rate − taylor_rate`: NEGATİF ise Fed kuralın işaret ettiğinden
    GEVŞEK, pozitif ise SIKI politikadadır. Katsayılar ön-kayıtlıdır (§TAYLOR_*),
    taranmamıştır — taranmış bir katsayı bu modülü kenar arayıcıya çevirirdi.
    """
    pce = _lagged("PCEPILFE", since="2010-01-01", lag=lag, today=today, fetch=fetch)
    unrate = _lagged("UNRATE", since=since, lag=lag, today=today, fetch=fetch)
    nrou = _lagged("NROU", since=since, lag=lag, today=today, fetch=fetch)
    dff = _lagged("DFF", since=since, lag=lag, today=today, fetch=fetch)
    if pce.empty or unrate.empty or dff.empty:
        logger.warning("[makro] Taylor girdileri eksik — hesaplanamadı")
        return pd.DataFrame()
    # Enflasyon: 12 AYLIK yüzde değişim. pct_change(12) aylık seride 12 ayı verir;
    # gecikme kaydırması indeksi ötelediği için satır SIRASI korunur, aralık bozulmaz.
    enf = pce.pct_change(12, fill_method=None) * 100.0
    grid = pd.DatetimeIndex(sorted(set(dff.index)))
    d = pd.DataFrame(index=grid)
    d["policy_rate"] = _to_grid(dff, grid, series="DFF")
    d["inflation"] = _to_grid(enf.dropna(), grid, series="PCEPILFE")
    d["unrate"] = _to_grid(unrate, grid, series="UNRATE")
    d["nrou"] = _to_grid(nrou, grid, series="NROU") if not nrou.empty else np.nan
    d["output_gap"] = OKUN_COEF * (d["nrou"] - d["unrate"])
    d["taylor_rate"] = (TAYLOR_R_STAR + d["inflation"]
                        + TAYLOR_PI_COEF * (d["inflation"] - TAYLOR_PI_STAR)
                        + TAYLOR_GAP_COEF * d["output_gap"])
    d["gap"] = d["policy_rate"] - d["taylor_rate"]
    return d.dropna(subset=["gap"])


# ---------------------------------------------------------------------------
# Tek serilik göstergeler (Madde 153 · 154 · 155 · 158 · 160 · 164 · 166 · 169)
# ---------------------------------------------------------------------------
def _tek_seri(series: str, *, since: str, lag: bool, today: pd.Timestamp | None,
              fetch: bool = True,
              chg: int | None = None, z: int | None = None,
              ret: int | None = None) -> pd.DataFrame:
    """Bir FRED serisini panelleştir: seviye + istenen türev kolonlar."""
    s = _lagged(series, since=since, lag=lag, today=today, fetch=fetch)
    if s.empty:
        return pd.DataFrame()
    d = pd.DataFrame({"level": s})
    if chg is not None:
        d[f"chg_{chg}"] = s - s.shift(chg)
    if ret is not None:
        d[f"ret_{ret}"] = s.pct_change(ret, fill_method=None)
    if z is not None:
        base = d[f"ret_{ret}"] if ret is not None else d["level"]
        d["z"] = _zscore(base, z)
    return d


def term_premium(*, since: str = "2015-01-01", lag: bool = True,
                 today: pd.Timestamp | None = None, fetch: bool = True) -> pd.DataFrame:
    """ACM 10 yıllık vade primi (Madde 154) — seviye + 20 günlük değişim."""
    return _tek_seri("THREEFYTP10", since=since, lag=lag, today=today, fetch=fetch, chg=20)


def financial_conditions(*, since: str = "2015-01-01", lag: bool = True,
                         today: pd.Timestamp | None = None, fetch: bool = True) -> pd.DataFrame:
    """Chicago Fed NFCI (Madde 155). >0 = tarihsel ortalamadan SIKI koşullar."""
    return _tek_seri("NFCI", since=since, lag=lag, today=today, fetch=fetch, chg=20)


def sahm_gap(*, since: str = "2015-01-01", lag: bool = True,
             today: pd.Timestamp | None = None, fetch: bool = True) -> pd.DataFrame:
    """Sahm kuralı gerçek-zamanlı göstergesi (Madde 160). ≥0,50 = resesyon tetiği."""
    return _tek_seri("SAHMREALTIME", since=since, lag=lag, today=today, fetch=fetch, chg=3)


def breakeven_inflation(*, since: str = "2015-01-01", lag: bool = True,
                        today: pd.Timestamp | None = None, fetch: bool = True) -> pd.DataFrame:
    """10 yıllık TIPS breakeven enflasyon beklentisi (Madde 166) — seviye + 250g z."""
    return _tek_seri("T10YIE", since=since, lag=lag, today=today, fetch=fetch, chg=20, z=250)


def curve_slope(*, since: str = "2015-01-01", lag: bool = True,
                today: pd.Timestamp | None = None, fetch: bool = True) -> pd.DataFrame:
    """2s10s getiri eğrisi eğimi (Madde 169) + `uninversion`: negatiften pozitife GEÇİŞ.

    `uninversion` yalnız GEÇİŞ barında 1'dir (seviye değil OLAY) — pozitif kaldığı
    her günü işaretlemek, olayı 200 gün boyunca tekrar eden bir seviyeye çevirirdi.
    """
    d = _tek_seri("T10Y2Y", since=since, lag=lag, today=today, fetch=fetch, chg=20)
    if d.empty:
        return d
    onceki = d["level"].shift(1)
    d["uninversion"] = ((onceki < 0.0) & (d["level"] >= 0.0)).astype(int)
    return d


def dollar_momentum(*, since: str = "2015-01-01", lag: bool = True,
                    today: pd.Timestamp | None = None, fetch: bool = True) -> pd.DataFrame:
    """Geniş dolar endeksi (DTWEXBGS) 20 günlük momentumu (Madde 164)."""
    return _tek_seri("DTWEXBGS", since=since, lag=lag, today=today, fetch=fetch, ret=20, z=250)


def usdjpy_shock(*, since: str = "2015-01-01", lag: bool = True,
                 today: pd.Timestamp | None = None, fetch: bool = True) -> pd.DataFrame:
    """USD/JPY 5 günlük getirisi ve z-skoru (Madde 153 — carry çözülmesi).

    DEXJPUS "1 dolar kaç yen" kotasyonudur: paritenin DÜŞMESİ yenin GÜÇLENMESİdir,
    yani carry çözülmesi NEGATİF `ret_5`'tir. Listenin `−2,5σ` eşiği bu yöndedir.
    """
    return _tek_seri("DEXJPUS", since=since, lag=lag, today=today, fetch=fetch, ret=5, z=250)


def oil_shock(*, since: str = "2015-01-01", lag: bool = True,
              today: pd.Timestamp | None = None, fetch: bool = True) -> pd.DataFrame:
    """WTI ham petrol 3 günlük getirisi (Madde 158)."""
    return _tek_seri("DCOILWTICO", since=since, lag=lag, today=today, fetch=fetch, ret=3, z=250)


# ---------------------------------------------------------------------------
# İki serilik makaslar (Madde 167 · 170 · 174)
# ---------------------------------------------------------------------------
def _makas(a: str, b: str, *, since: str, lag: bool, today: pd.Timestamp | None,
           fetch: bool = True, z: int | None = 250) -> pd.DataFrame:
    """a − b makası, ORTAK ızgarada. Biri boşsa boş DataFrame (yarım makas ölçüm değildir)."""
    sa = _lagged(a, since=since, lag=lag, today=today, fetch=fetch)
    sb = _lagged(b, since=since, lag=lag, today=today, fetch=fetch)
    if sa.empty or sb.empty:
        return pd.DataFrame()
    grid = pd.DatetimeIndex(sorted(set(sa.index) & set(sb.index)))
    if len(grid) == 0:
        return pd.DataFrame()
    d = pd.DataFrame(index=grid)
    d[a.lower()] = _to_grid(sa, grid, series=a)
    d[b.lower()] = _to_grid(sb, grid, series=b)
    d["spread"] = d[a.lower()] - d[b.lower()]
    if z is not None:
        d["z"] = _zscore(d["spread"], z)
    return d.dropna(subset=["spread"])


def credit_spread(*, since: str = "2015-01-01", lag: bool = True,
                  today: pd.Timestamp | None = None, fetch: bool = True) -> pd.DataFrame:
    """Yüksek getirili − yatırım yapılabilir OAS makası (Madde 167).

    Liste HYG/LQD ETF ORANINI önerir; burada onun yerine ICE BofA OAS serileri
    kullanılır çünkü ETF oranı süre (duration) ve temettü farkını da taşır — yani
    kredi riskinin YANINDA faiz riskini de ölçer. OAS saf kredi primi.
    UYARI: her iki OAS serisi de FRED grafik ucunda ~3 yılla KIRPIKTIR (`fred_data`
    `attrs["kirpildi"]` ile işaretler); 2023 öncesi için ölçüm YOKTUR.
    """
    return _makas("BAMLH0A0HYM2", "BAMLC0A0CM", since=since, lag=lag, today=today, fetch=fetch)


def cp_tbill_spread(*, since: str = "2015-01-01", lag: bool = True,
                    today: pd.Timestamp | None = None, fetch: bool = True) -> pd.DataFrame:
    """3 aylık AA finans-dışı ticari senet − 3 aylık Hazine bonosu (Madde 170)."""
    return _makas("RIFSPPFAAD90NB", "DTB3", since=since, lag=lag, today=today, fetch=fetch)


def sofr_effr_basis(*, since: str = "2018-04-01", lag: bool = True,
                    today: pd.Timestamp | None = None, fetch: bool = True) -> pd.DataFrame:
    """SOFR − EFFR basisi (Madde 174). SOFR 2018-04'te başlar; öncesi ölçüm YOK."""
    return _makas("SOFR", "EFFR", since=since, lag=lag, today=today, fetch=fetch)


# ---------------------------------------------------------------------------
# ETF vekilleri — aynı Alpaca temettü/bölünme düzeltilmiş tarihçesi kullanılır.
# HYG/LQD oranı duration/fon etkileri taşır; mevcut FRED OAS makasının yerine geçmez.
# BIL çıktısı geçmiş toplam getiridir; ileriye dönük risksiz faiz değildir.
# ---------------------------------------------------------------------------
ETF_SYMBOLS = ("GLD", "USO", "CPER", "HYG", "LQD", "BIL")


def etf_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Seans sıralı düzeltilmiş kapanışlar → 3 ETF oranı + BIL 20 seans toplam getiri vekili."""
    if (not isinstance(prices, pd.DataFrame) or prices.empty
            or not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans
            or prices.index.has_duplicates or not prices.index.is_monotonic_increasing):
        return pd.DataFrame()
    px = prices.apply(pd.to_numeric, errors="coerce")
    px = px.where(np.isfinite(px) & (px > 0))
    out = pd.DataFrame(index=px.index)
    for name, a, b in (("copper_gold_etf", "CPER", "GLD"), ("gold_oil_etf", "GLD", "USO"),
                       ("hy_ig_etf", "HYG", "LQD")):
        if a in px and b in px:
            out[name] = px[a] / px[b]
    if "BIL" in px:
        with np.errstate(all="ignore"):
            out["bil_return20_ann"] = (px["BIL"] / px["BIL"].shift(20)) ** (252 / 20) - 1
    return out.replace([np.inf, -np.inf], np.nan)


def etf_proxies(*, since: str = "2015-01-01", lag: bool = True,
                today: pd.Timestamp | None = None, fetch: bool = True) -> pd.DataFrame:
    """Alpaca tarihçesinin cache-only okuma sözleşmesiyle, tamamlanmış seanslar.

    Güncel düzeltilmiş fiyat serisidir; tarihsel revizyon/vintage kanıtı değildir.
    Varsayılan T+1 kullanılabilirlik gecikmesi diğer makro serileriyle uyumludur.
    """
    from nihai.data import alpaca_history as ah
    from nihai.data.market_calendar import NYSE

    if ah._adjustment() != "all":
        logger.warning("[makro ETF] toplam getiri için ALPACA_HIST_ADJUSTMENT=all gerekli")
        return pd.DataFrame()
    cutoff = pd.Timestamp(today if today is not None else pd.Timestamp.now("UTC")).tz_localize(None).normalize()
    columns = {}
    for symbol in ETF_SYMBOLS:
        bars = ah.get_history(symbol, "1d", start=since, end=cutoff.date().isoformat(), fetch=fetch)
        if bars.empty or "close" not in bars:
            continue
        series = pd.to_numeric(bars["close"], errors="coerce").copy()
        series.index = series.index.tz_localize(None).normalize()
        series = series.loc[(series.index < cutoff) & ~series.index.duplicated(keep=False)]
        columns[symbol] = series
    if not columns:
        return pd.DataFrame()
    prices = pd.DataFrame(columns).sort_index()
    sessions = NYSE.sessions.tz_localize(None)
    grid = sessions[(sessions >= prices.index.min()) & (sessions <= prices.index.max())]
    out = etf_indicators(prices.reindex(grid))
    if lag:
        out.index = out.index + pd.Timedelta(days=1)
    return out.loc[out.index <= cutoff]


# ---------------------------------------------------------------------------
# Birleşik panel
# ---------------------------------------------------------------------------
# (üretici, kolon, ileri-doldurma bütçesi TAKVİM GÜNÜ). Bütçe kaynağın FREKANSINDAN
# gelir: günlük seri 5, haftalık 10, aylık 45. `uninversion` bütçesi 0'dır çünkü o bir
# OLAYdır — ileri doldurulan bir olay, 200 gün süren bir seviyeye dönüşürdü.
PANEL_BUILDERS: dict[str, tuple[object, str, int]] = {
    "net_liq": (net_liquidity, "net_liquidity", 10),
    "net_liq_chg20": (net_liquidity, "net_liq_chg_20", 10),
    "taylor_gap": (taylor_rule_gap, "gap", 5),
    "term_premium": (term_premium, "level", 5),
    "nfci": (financial_conditions, "level", 10),
    "sahm": (sahm_gap, "level", 45),
    "breakeven": (breakeven_inflation, "level", 5),
    "breakeven_z": (breakeven_inflation, "z", 5),
    "curve_2s10s": (curve_slope, "level", 5),
    "uninversion": (curve_slope, "uninversion", 0),
    "dxy_ret20": (dollar_momentum, "ret_20", 5),
    "usdjpy_ret5_z": (usdjpy_shock, "z", 5),
    "wti_ret3": (oil_shock, "ret_3", 5),
    "credit_spread": (credit_spread, "spread", 5),
    "cp_tbill": (cp_tbill_spread, "spread", 5),
    "sofr_effr": (sofr_effr_basis, "spread", 5),
    "copper_gold_etf": (etf_proxies, "copper_gold_etf", 5),
    "gold_oil_etf": (etf_proxies, "gold_oil_etf", 5),
    "hy_ig_etf": (etf_proxies, "hy_ig_etf", 5),
    "bil_return20_ann": (etf_proxies, "bil_return20_ann", 5),
}


def panel(*, since: str = "2015-01-01", lag: bool = True,
          today: pd.Timestamp | None = None, fetch: bool = True,
          grid: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Tüm makro göstergeleri TEK DataFrame'de topla (KARAR-DIŞI özellik yüzeyi).

    `grid` verilirse (tipik olarak fiyat panelinin işlem günleri) her kolon o ızgaraya
    SINIRLI nedensel ffill ile taşınır; verilmezse kolonların tarih birleşimi kullanılır.
    Bir gösterge üretilemezse (ağ yok / seri kırpık) o KOLON düşer — panel boş dönmez;
    hangi kolonun düştüğü `attrs["eksik"]`te durur.
    """
    parcalar: dict[str, tuple[pd.Series, int]] = {}
    eksik: list[str] = []
    onbellek: dict[object, pd.DataFrame] = {}
    for ad, (fn, kolon, butce) in PANEL_BUILDERS.items():
        if fn not in onbellek:
            try:
                onbellek[fn] = fn(since=since, lag=lag, today=today,  # type: ignore[operator]
                              fetch=fetch)
            except Exception as e:                       # pragma: no cover - fail-soft
                logger.warning(f"[makro] {ad} üretilemedi: {type(e).__name__}: {e}")
                onbellek[fn] = pd.DataFrame()
        d = onbellek[fn]
        if d.empty or kolon not in d.columns:
            eksik.append(ad)
            continue
        parcalar[ad] = (d[kolon].dropna(), butce)
    if not parcalar:
        out = pd.DataFrame()
        out.attrs["eksik"] = eksik
        out.attrs["lag"] = lag
        return out
    hedef = grid if grid is not None else pd.DatetimeIndex(
        sorted(set().union(*(set(s.index) for s, _ in parcalar.values()))))
    out = pd.DataFrame(index=hedef)
    for ad, (s, butce) in parcalar.items():
        out[ad] = _causal_ffill(s, hedef, butce)
    out.attrs["eksik"] = eksik
    out.attrs["lag"] = lag
    out.attrs["source_dates"] = {
        ad: str(s.loc[s.index <= hedef.max()].index[-1].date())
        if len(s.loc[s.index <= hedef.max()]) else None for ad, (s, _) in parcalar.items()
    }
    return out


def summary(*, since: str = "2015-01-01", lag: bool = True,
            fetch: bool = True) -> dict:
    """Panonun/raporun okuduğu SON durum özeti — her gösterge için son değer + tarih."""
    d = panel(since=since, lag=lag, fetch=fetch)
    if d.empty:
        return {"ok": False, "eksik": list(d.attrs.get("eksik") or []), "n": 0}
    son = d.iloc[-1]
    return {
        "ok": True,
        "as_of": str(d.index[-1].date()),
        "n": int(len(d)),
        "eksik": list(d.attrs.get("eksik") or []),
        "lag_uygulandi": bool(lag),
        "source_dates": d.attrs.get("source_dates", {}),
        "degerler": {k: (None if pd.isna(v) else float(v)) for k, v in son.items()},
    }


def _cli() -> None:  # pragma: no cover - operatör aracı
    """Önbelleği DOLDUR + panelin son durumunu bas.

    `/api/quant/macro` önbellek-YALNIZ okur (BULGU-59-8); doldurmak bu adımdır ve
    uç hata mesajı doğrudan buraya yönlendirir. Belgede geçen her `python -m <modül>`
    çözümlenebilmelidir (Tur-61 kuralı) — bu giriş noktası o kuralın gereğidir.
    """
    import argparse

    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Makro rejim paneli (FRED, anahtarsız)")
    ap.add_argument("--since", default="2015-01-01")
    ap.add_argument("--no-lag", action="store_true",
                    help="yayın gecikmesini UYGULAMA (look-ahead riski — bilerek seç)")
    ap.add_argument("--no-fetch", action="store_true", help="yalnız önbellekten oku")
    ap.add_argument("--etf-only", action="store_true", help="yalnız GLD/USO/CPER/HYG/LQD/BIL vekillerini hazırla")
    a = ap.parse_args()
    if a.etf_only:
        d = etf_proxies(since=a.since, lag=not a.no_lag, fetch=not a.no_fetch)
        if d.empty:
            print("ETF ölçümü yok; Alpaca anahtarı ve adjustment=all önkoşulunu kontrol edin.")
        else:
            print("ETF vekilleri (güncel düzeltilmiş seri, BIL geçmiş toplam getirisi):")
            print(d.tail(5).to_string())
        return
    d = summary(since=a.since, lag=not a.no_lag, fetch=not a.no_fetch)
    if not d.get("ok"):
        print(f"panel kurulamadı — eksik: {', '.join(d.get('eksik') or []) or '?'}")
        return
    print(f"makro paneli: {d['n']} gün · son {d['as_of']} · "
          f"yayın gecikmesi {'UYGULANDI' if d['lag_uygulandi'] else 'KAPALI'}")
    if d.get("eksik"):
        print(f"ÖLÇÜLEMEYEN gösterge: {', '.join(d['eksik'])}")
    for ad, v in d["degerler"].items():
        print(f"  {ad:<16} {'—' if v is None else f'{v:,.4f}'}")


if __name__ == "__main__":  # pragma: no cover
    _cli()
