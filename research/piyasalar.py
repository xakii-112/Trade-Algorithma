"""piyasalar.py — ÇOKLU PİYASA kataloğu + veri kaynakları + önbellekli yükleyici (Tur-96).

Sistem ABD hisse/opsiyon için yazıldı; bu modül aynı AI'ları ve algoritmaları dünyanın geri kalanında
ölçebilmek için gereken üç şeyi TEK yerde tutar:

  1. `PIYASALAR`      — piyasa grubu → temsilci enstrümanlar (endeks + likit hisse/ETF/vadeli/parite).
  2. `VERI_KAYNAKLARI` — hangi kaynağın hangi piyasayı, hangi koşulla verdiği; `durum` alanı
                        2026-09-25'te ÇAĞRILARAK ölçüldü (varsayılmadı).
  3. `yukle()`        — yfinance günlük bar → temizlenmiş, tz-naive OHLCV; önbellek OneDrive DIŞINDA
                        (`PIYASA_CACHE_DIR`, varsayılan `~/nihai_hist/piyasalar`; Tur-94 ile aynı ilke:
                        araştırma önbelleği senkronlanacak bir şey değildir).

Sözleşmeler:
  · Fiyatlar YEREL para birimindedir (kur çevrimi YOK) — her ölçü getiri bazlıdır, kur riski enstrümanın
    kendi getirisine dahil değildir. Kur etkisini ölçmek isteyen FX grubuna bakar.
  · `olcekle()` fiyatı ilk kapanış = 100 olacak şekilde SABİT bir çarpanla ölçekler: getiriler aynen korunur,
    ama backtest motoru TAM SAYI hisse alır (`risk_manager.size_stock`) ve ₩70.000'lik bir hisse ya da
    1,08'lik bir parite ölçeksiz koşunca yuvarlama boyutu bozardı.
  · Temizlik yalnız mekanik tutarsızlığı düzeltir (NaN/≤0 satır atılır; high/low OHLC zarfına çekilir) ve
    SAYILIR (`kalite()`); şüpheli sıçramalar SİLİNMEZ, raporlanır — veriyi sonuca göre ayıklamak ölçümü
    kirletirdi.

CLI:  python -m research.piyasalar [--indir] [--yenile]   → kapsama + kalite tablosu
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(os.getenv("PIYASA_CACHE_DIR") or (Path.home() / "nihai_hist" / "piyasalar"))

# ad → {bolge, semboller, gun (yıllık işlem günü; Sharpe ölçeği), not}
PIYASALAR: dict[str, dict] = {
    "ABD": {"bolge": "Kuzey Amerika", "gun": 252,
            "semboller": ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "AMZN", "JPM", "XOM", "JNJ"],
            "not": "sistemin kendi piyasası (canlı evren ABD)"},
    "Avrupa": {"bolge": "Avrupa", "gun": 252,
               "semboller": ["^STOXX50E", "^GDAXI", "^FTSE", "^FCHI", "^IBEX", "^SSMI",
                             "ASML.AS", "SAP.DE", "NESN.SW", "MC.PA", "SHEL.L", "AZN.L"]},
    "Japonya": {"bolge": "Asya", "gun": 245,
                "semboller": ["^N225", "7203.T", "6758.T", "8306.T", "9984.T", "6861.T", "9432.T"]},
    "Cin_HK": {"bolge": "Asya", "gun": 245,
               "semboller": ["^HSI", "000001.SS", "399001.SZ", "0700.HK", "0005.HK", "0941.HK",
                             "600519.SS", "1299.HK"]},
    "Kore_Tayvan_Sing": {"bolge": "Asya", "gun": 248,
                         "semboller": ["^KS11", "005930.KS", "000660.KS", "^TWII", "2330.TW", "2317.TW",
                                       "^STI", "D05.SI"]},
    "Hindistan": {"bolge": "Asya", "gun": 248,
                  "semboller": ["^NSEI", "^BSESN", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
                                "ICICIBANK.NS"]},
    "Turkiye": {"bolge": "Avrupa/Asya", "gun": 252,
                "semboller": ["XU100.IS", "XU030.IS", "THYAO.IS", "GARAN.IS", "AKBNK.IS", "ASELS.IS",
                              "BIMAS.IS", "KCHOL.IS", "SISE.IS", "TUPRS.IS", "EREGL.IS"]},
    "Kanada_Avustralya": {"bolge": "Diğer gelişmiş", "gun": 252,
                          "semboller": ["^GSPTSE", "RY.TO", "TD.TO", "ENB.TO", "^AXJO", "CBA.AX", "BHP.AX",
                                        "CSL.AX"]},
    "Latin_Amerika": {"bolge": "Latin Amerika", "gun": 248,
                      "semboller": ["^BVSP", "^MXX", "VALE3.SA", "PETR4.SA", "ITUB4.SA", "WALMEX.MX",
                                    "GFNORTEO.MX"]},
    "Orta_Dogu_Afrika": {"bolge": "Orta Doğu/Afrika", "gun": 250,
                         "semboller": ["^TA125.TA", "TEVA.TA", "LUMI.TA", "1120.SR", "2010.SR", "NPN.JO",
                                       "SBK.JO"]},
    "Kripto": {"bolge": "Küresel (7/24)", "gun": 365,
               "semboller": ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BNB-USD", "ADA-USD", "DOGE-USD",
                             "SOL-USD", "TRX-USD", "XLM-USD"],
               "not": "7/24 işlem; yfinance günü UTC"},
    "Forex": {"bolge": "Küresel (24/5)", "gun": 260,
              "semboller": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X",
                            "NZDUSD=X", "EURJPY=X", "USDTRY=X", "USDMXN=X"],
              "not": "HACİM YOK (yfinance 0 yazar) → hacim koşullu algoritmalar tetiklenmez"},
    "Emtia": {"bolge": "Küresel (vadeli)", "gun": 252,
              "semboller": ["GC=F", "SI=F", "CL=F", "BZ=F", "NG=F", "HG=F", "ZC=F", "ZW=F", "ZS=F", "KC=F"],
              "not": "sürekli vadeli seri, DEVİR (roll) boşlukları düzeltilmemiş"},
    "Tahvil": {"bolge": "ABD sabit getirili (ETF)", "gun": 252,
               "semboller": ["TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "EMB", "AGG", "MUB"]},
}

# Ölçüldü 2026-09-25 (bu makineden, anahtarsız): `durum` = çağrının gerçek sonucu.
VERI_KAYNAKLARI: list[dict] = [
    {"ad": "yfinance (Yahoo Finance)", "kapsam": "tüm gruplar: endeks, hisse, ETF, vadeli, parite, kripto",
     "anahtar": "yok", "cozunurluk": "1d tam geçmiş; 1h son 730 gün; 1m son 7-30 gün",
     "durum": "ÇALIŞIYOR (projede kurulu; bu turun birincil kaynağı)",
     "not": "FX hacmi 0; vadeli seri devir-düzeltmesiz; resmî olmayan uç (hız sınırı/biçim değişebilir)"},
    {"ad": "Binance açık REST (api.binance.com/api/v3/klines)", "kapsam": "kripto spot (USDT pariteleri)",
     "anahtar": "yok", "cozunurluk": "1m…1M, 2017'den",
     "durum": "ÇALIŞIYOR (HTTP 200, BTCUSDT 1h)",
     "not": "kripto için yfinance'ten derin gün-içi geçmiş; bazı ülkelerden erişim kısıtlı olabilir"},
    {"ad": "Alpaca Crypto Data API", "kapsam": "kripto (BTC/USD, ETH/USD …)", "anahtar": "projede mevcut",
     "cozunurluk": "1m…1d", "durum": "KURULU (alpaca-py), bu turda kullanılmadı",
     "not": "canlı hub'ın hisse akışıyla aynı hesap; kripto 7/24"},
    {"ad": "Stooq (stooq.com/q/d/l)", "kapsam": "küresel EOD endeks/hisse/FX", "anahtar": "—",
     "cozunurluk": "1d", "durum": "KULLANILAMAZ (HTTP 404 — anahtarsız CSV ucu kapandı)", "not": ""},
    {"ad": "FRED (St. Louis Fed)", "kapsam": "faiz/getiri eğrisi/makro", "anahtar": "yok (CSV)",
     "cozunurluk": "1d/aylık", "durum": "ÇALIŞIYOR (projede `nihai/data/fred_data.py`)",
     "not": "fiyat serisi değil; tahvil grubunun bağlamı"},
    {"ad": "CBOE (cdn.cboe.com)", "kapsam": "VIX ailesi", "anahtar": "yok", "cozunurluk": "1d",
     "durum": "ÇALIŞIYOR (projede `nihai/data/cboe_data.py`)", "not": "ABD oynaklık bağlamı"},
    {"ad": "Borsa İstanbul (yfinance `.IS`)", "kapsam": "BIST endeks + hisse", "anahtar": "yok",
     "cozunurluk": "1d", "durum": "ÇALIŞIYOR (XU100.IS 1997'den)", "not": "resmî BIST veri paketi ücretli"},
    {"ad": "Ücretli/anahtarlı (Polygon, Tiingo, EODHD, Twelve Data, Alpha Vantage, J-Quants, Refinitiv)",
     "kapsam": "küresel; PIT/düzeltilmiş/tick", "anahtar": "gerekli", "cozunurluk": "tick…1d",
     "durum": "DENENMEDİ (anahtar yok)",
     "not": "survivorship-temiz PIT evren için gerçek yol (ACI DERS #5); bu turun ölçümleri bugünkü listeyle"},
]


def _dosya(sym: str, interval: str) -> Path:
    guvenli = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in sym)
    return CACHE_DIR / f"{guvenli}_{interval}.csv"


def temizle(df: pd.DataFrame) -> pd.DataFrame:
    """Mekanik tutarsızlığı düzelt: sütun adları küçük harf, NaN/≤0 OHLC satırı at, high/low'u
    OHLC zarfına çek, indeks tekil + sıralı + tz-naive. Hacim yoksa 0."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    ohlc = ["open", "high", "low", "close"]
    df = df.dropna(subset=ohlc)
    df = df[(df[ohlc] > 0).all(axis=1)]
    df["high"] = df[ohlc].max(axis=1)
    df["low"] = df[ohlc].min(axis=1)
    df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


def indir(sym: str, interval: str = "1d") -> pd.DataFrame:
    """yfinance'ten tam geçmiş (1d: period=max; 1h: 730 gün) → temizle → önbelleğe yaz."""
    import yfinance as yf
    period = "max" if interval == "1d" else "730d"
    raw = yf.download(sym, period=period, interval=interval, auto_adjust=True, progress=False,
                      threads=False)
    df = temizle(raw)
    if not df.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_dosya(sym, interval))
    return df


def yukle(sym: str, interval: str = "1d", *, fetch: bool = False) -> pd.DataFrame:
    """Önbellekten oku; yoksa ve `fetch` açıksa indir. Yoksa boş df (istisna yok)."""
    p = _dosya(sym, interval)
    if p.exists():
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return temizle(df)
    return indir(sym, interval) if fetch else temizle(pd.DataFrame())


def aralik_onar(df: pd.DataFrame, esik: float = 0.5) -> tuple[pd.DataFrame, int]:
    """Tek-bar YANLIŞ BASKI onarımı: log(high/low) > `esik` olan barın high/low'u açılış-kapanış zarfına
    indirilir; onarılan bar SAYISI döner. Kripto için çağrılmaz (orada %65'lik gün-içi aralık gerçektir).
    Neden: SHEL.L 2019'da ~2.300'lük fiyatta low=3/23/26 yazıyor — ATR/aralık özellikleri ve HAR'ın RV
    vekili o barda patlar (HAR vekili/r² oranı 56); kapanış bazlı kusur kuralı bunu görmez."""
    if df.empty:
        return df, 0
    bozuk = np.log(df["high"] / df["low"]) > esik
    n = int(bozuk.sum())
    if n:
        df = df.copy()
        df.loc[bozuk, "high"] = df.loc[bozuk, ["open", "close"]].max(axis=1)
        df.loc[bozuk, "low"] = df.loc[bozuk, ["open", "close"]].min(axis=1)
    return df, n


def bayat_bar_maskesi(df: pd.DataFrame) -> np.ndarray:
    """İşlem OLMAYAN gün satırı (Tur-97, BULGU-97-5): aralık sıfır (high == low) VE kapanış bir öncekiyle
    aynı. yfinance yerel tatillerde fiyatı kopyalayıp satır verir (Japonya 2018-01-03; ölçüldü: 121
    enstrümanda 2.197 satır, Orta Doğu %3,6, Türkiye %1,7, ABD/kripto 0). Böyle bir satır oynaklığı SIFIR
    gösterir: oynaklık öngörüsü ertesi işlem günü için ~1e-8 varyans basıyordu (QLIKE patlaması)."""
    if df.empty:
        return np.zeros(0, dtype=bool)
    h, lo, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    ayni = np.zeros(c.size, dtype=bool)
    ayni[1:] = c[1:] == c[:-1]
    return (h == lo) & ayni


def olcekle(df: pd.DataFrame, hedef: float = 100.0) -> pd.DataFrame:
    """OHLC'yi ilk kapanış = `hedef` olacak şekilde SABİT çarpanla ölçekle (getiriler aynen)."""
    if df.empty:
        return df
    k = hedef / float(df["close"].iloc[0])
    out = df.copy()
    out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]] * k
    return out


def kalite(df: pd.DataFrame) -> dict:
    """Veri sağlığı SAYIMI (düzeltme değil): kapsam, hacim yokluğu, donuk bar, şüpheli sıçrama."""
    if df.empty:
        return {"n": 0}
    r = np.log(df["close"]).diff().to_numpy()
    r1 = np.nan_to_num(r[1:], nan=0.0)
    r2 = np.nan_to_num(r[2:], nan=0.0)
    # sıçra-geri-dön: |r_t| > 0,25 ve r_{t+1} ters işaretli ve |r_{t+1}| > 0,2 (tipik yanlış baskı imzası)
    geri = int(np.sum((np.abs(r1[:-1]) > 0.25) & (np.abs(r2) > 0.2) & (np.sign(r1[:-1]) != np.sign(r2))))
    return {"n": int(len(df)), "ilk": str(df.index[0].date()), "son": str(df.index[-1].date()),
            "hacim_yok_pay": round(float((df["volume"] <= 0).mean()), 4),
            "donuk_bar_pay": round(float((df["high"] <= df["low"]).mean()), 4),
            "max_abs_log_getiri": round(float(np.nanmax(np.abs(r))), 4) if len(r) > 1 else 0.0,
            "sicra_geri_don": geri}


def tum_semboller() -> list[tuple[str, str]]:
    return [(p, s) for p, v in PIYASALAR.items() for s in v["semboller"]]


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Çoklu piyasa kapsama + kalite")
    ap.add_argument("--indir", action="store_true", help="önbellekte olmayanları indir")
    ap.add_argument("--yenile", action="store_true", help="hepsini yeniden indir")
    ap.add_argument("--interval", default="1d")
    a = ap.parse_args()
    for piyasa, sym in tum_semboller():
        df = (indir(sym, a.interval) if a.yenile
              else yukle(sym, a.interval, fetch=a.indir))
        k = kalite(df)
        print(f"{piyasa:18s} {sym:12s} {k}")
    print(f"önbellek: {CACHE_DIR}")


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    _cli()
