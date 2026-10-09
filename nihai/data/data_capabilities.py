"""
data_capabilities.py — "ELİMDE HANGİ VERİ VAR?" sorusunun ÖLÇÜLMÜŞ cevabı (Tur-57).

Proje boyunca veri kaynağı tartışmaları hep VARSAYIMLA yürüdü ("Alpaca opsiyonu paylı",
"ücretsiz plan SIP vermez", "OI yok"). Tur-57'de bunlar tek tek ÇAĞRILARAK ölçüldü ve
üçü de yanlış çıktı. Bu modül o ölçümü OTOMATİKLEŞTİRİR: aynı komut bugün ücretsiz
anahtarla, yarın paylı planla koşulur ve FARKI gösterir — plan yükseltmesinin ne
kazandırdığı iddia değil ÖLÇÜM olur.

    python -m nihai.data.data_capabilities            # tam tablo (ağ kullanır)
    python -m nihai.data.data_capabilities --json     # makine okunur
    python -m nihai.data.data_capabilities --offline  # yalnız anahtar envanteri

Proje dersi (Tur-12): "baseline'ı rapordan değil YERİNDE ölç". Bu dosya o dersin
veri katmanındaki karşılığıdır.

Her yetenek üç alan döndürür:
    ok      : True (ölçüldü, çalışıyor) | False (ölçüldü, çalışmıyor) | None (ölçülemedi)
    detay   : ne görüldüğü — sayıyla
    kilit   : çalışmıyorsa neyin açacağı (hangi anahtar / hangi plan)

ANAYASA: salt-okunur teşhis. Hiçbir sinyal/eşik/emir bu modüle bağlanmaz.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import nihai.config  # noqa: F401 — .env'i yükler (tek doğruluk kaynağı)
from nihai.console import ensure_utf8_stdout

# Ölçüm sembolleri: biri likit hisse, biri en likit ETF (opsiyon zinciri her zaman dolu).
DEFAULT_SYMBOL = "AAPL"
DEFAULT_OPTION_SYMBOL = "SPY"


def _res(ok: bool | None, detay: str, kilit: str = "") -> dict[str, Any]:
    return {"ok": ok, "detay": detay, "kilit": kilit}


def _keys() -> dict[str, bool]:
    """Hangi API anahtarı DOLU (değer basılmaz — yalnız var/yok)."""
    names = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
             "FRED_API_KEY", "NASDAQ_DATA_LINK_API_KEY", "REDDIT_CLIENT_ID",
             "FINNHUB_API_KEY", "POLYGON_API_KEY")
    return {n: bool((os.getenv(n) or "").strip()) for n in names}


def _clients() -> tuple[Any, Any, Any, Any]:
    """(stock, option, trading, news) istemcileri — SDK/anahtar yoksa None."""
    key, sec = os.getenv("ALPACA_API_KEY", ""), os.getenv("ALPACA_SECRET_KEY", "")
    if not (key and sec):
        return None, None, None, None
    stock = option = trading = news = None
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        stock = StockHistoricalDataClient(key, sec)
    except Exception:
        pass
    try:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        option = OptionHistoricalDataClient(key, sec)
    except Exception:
        pass
    try:
        from alpaca.trading.client import TradingClient
        trading = TradingClient(key, sec, paper=True)
    except Exception:
        pass
    try:
        from alpaca.data.historical.news import NewsClient
        news = NewsClient(key, sec)
    except Exception:
        pass
    return stock, option, trading, news


def _bars(stock: Any, symbol: str, feed_name: str, days: int = 8):
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                           start=datetime.now(UTC) - timedelta(days=days),
                           feed=DataFeed(feed_name))
    df = stock.get_stock_bars(req).df
    return df.droplevel(0) if getattr(df.index, "nlevels", 1) > 1 else df


def probe(symbol: str = DEFAULT_SYMBOL, option_symbol: str = DEFAULT_OPTION_SYMBOL,
          *, network: bool = True) -> dict[str, Any]:
    """Yetenekleri ÖLÇ. `network=False` ise yalnız anahtar envanteri (test/CI yolu)."""
    out: dict[str, Any] = {
        "olcum_zamani": datetime.now(UTC).isoformat(timespec="seconds"),
        "anahtarlar": _keys(),
        "ayar": {"ALPACA_FEED": os.getenv("ALPACA_FEED", "iex"),
                 "ALPACA_BARS_FEED": os.getenv("ALPACA_BARS_FEED", "(ALPACA_FEED ile aynı)"),
                 "ALPACA_OPTIONS": os.getenv("ALPACA_OPTIONS", "off"),
                 "DATA_PROVIDER": os.getenv("DATA_PROVIDER", "yfinance")},
        "yetenekler": {},
    }
    caps: dict[str, dict] = out["yetenekler"]
    if not network:
        caps["ag"] = _res(None, "ağ ölçümü atlandı (--offline)", "")
        return out

    stock, option, trading, news = _clients()
    if stock is None:
        caps["alpaca"] = _res(False, "alpaca-py veya ALPACA_API_KEY/SECRET yok",
                              ".env'e anahtar ekle + pip install alpaca-py")
        return out

    # 1) Konsolide hacim: IEX ne kadarını görüyor?
    try:
        iex, sip = _bars(stock, symbol, "iex"), _bars(stock, symbol, "sip")
        j = iex.join(sip, lsuffix="_iex", rsuffix="_sip", how="inner").tail(3)
        pay = float((j["volume_iex"] / j["volume_sip"].replace(0, float("nan"))).mean() * 100.0)
        caps["sip_tarihsel_bar"] = _res(
            True, f"SIP tarihsel bar YETKİLİ · IEX konsolide hacmin %{pay:.1f}'ini görüyor "
                  f"({symbol}, son {len(j)} gün)",
            "ALPACA_BARS_FEED=sip (uzatılmış seans barları da gelir — önce A/B ölç)")
    except Exception as e:
        caps["sip_tarihsel_bar"] = _res(False, f"{type(e).__name__}: {str(e)[:120]}",
                                        "Alpaca Market Data aboneliği")

    # 2) Bar kolonları: trade_count / vwap
    try:
        cols = list(_bars(stock, symbol, os.getenv("ALPACA_FEED", "iex")).columns)
        var = [c for c in ("trade_count", "vwap") if c in cols]
        caps["bar_akis_kolonlari"] = _res(
            len(var) == 2, f"kolonlar: {cols}",
            "" if len(var) == 2 else "sağlayıcı bu kolonları vermiyor")
    except Exception as e:
        caps["bar_akis_kolonlari"] = _res(None, f"{type(e).__name__}: {str(e)[:120]}", "")

    # 3) Gerçek-zamanlı SIP (son 15 dk) — paylı planın ASIL sattığı şey
    try:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestQuoteRequest
        stock.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.SIP))
        caps["sip_gercek_zamanli"] = _res(True, "SIP anlık kotasyon YETKİLİ", "")
    except Exception as e:
        msg = str(e)
        blocked = "subscription" in msg.lower()
        caps["sip_gercek_zamanli"] = _res(
            False, "abonelik son 15 dk SIP'i kapatıyor" if blocked else msg[:120],
            "Alpaca Algo Trader Plus (~$99/ay) — TEK gerçek eksik budur")

    # 4) OPRA opsiyon zinciri (IV + greekler + NBBO)
    if option is not None:
        try:
            from alpaca.data.requests import OptionChainRequest
            ch = option.get_option_chain(OptionChainRequest(underlying_symbol=option_symbol))
            n = len(ch or {})
            iv_var = sum(1 for s in (ch or {}).values()
                         if getattr(s, "implied_volatility", None))
            caps["opra_zincir"] = _res(
                n > 0, f"{option_symbol}: {n} kontrat, {iv_var} tanesinde IV+greek",
                "" if n else "OPRA aboneliği")
        except Exception as e:
            caps["opra_zincir"] = _res(False, f"{type(e).__name__}: {str(e)[:120]}",
                                       "OPRA opsiyon veri aboneliği")

    # 5) Açık pozisyon (OI) — trading API kontratlarından
    if trading is not None:
        try:
            from alpaca.trading.requests import GetOptionContractsRequest
            r = trading.get_option_contracts(
                GetOptionContractsRequest(underlying_symbols=[option_symbol], limit=100))
            cs = list(getattr(r, "option_contracts", None) or [])
            dolu = [c for c in cs if getattr(c, "open_interest", None) is not None]
            tarih = str(getattr(dolu[0], "open_interest_date", "")) if dolu else "-"
            caps["opsiyon_oi"] = _res(
                bool(dolu), f"{len(dolu)}/{len(cs)} kontratta OI (tarih {tarih})",
                "" if dolu else "OCC OI beslemesi bu hesapta yok")
        except Exception as e:
            caps["opsiyon_oi"] = _res(False, f"{type(e).__name__}: {str(e)[:120]}", "")

        # 6) Açığa satış bayrakları
        try:
            a = trading.get_asset(symbol)
            caps["acik_satis_bayraklari"] = _res(
                True, f"{symbol}: shortable={a.shortable} easy_to_borrow={a.easy_to_borrow}", "")
        except Exception as e:
            caps["acik_satis_bayraklari"] = _res(False, f"{type(e).__name__}: {str(e)[:120]}", "")

    # 7) Haber akışı (Benzinga)
    if news is not None:
        try:
            from alpaca.data.requests import NewsRequest
            n = news.get_news(NewsRequest(symbols=symbol, limit=3))
            items = (getattr(n, "data", {}) or {}).get("news", [])
            caps["haber_akisi"] = _res(bool(items), f"{len(items)} başlık çekildi "
                                       f"(kaynak: {getattr(items[0], 'source', '?') if items else '-'})",
                                       "" if items else "haber aboneliği")
        except Exception as e:
            caps["haber_akisi"] = _res(False, f"{type(e).__name__}: {str(e)[:120]}", "")

    return out


_BASLIK = {
    "sip_tarihsel_bar": "Konsolide (SIP) tarihsel bar",
    "bar_akis_kolonlari": "Bar akış kolonları (trade_count/vwap)",
    "sip_gercek_zamanli": "SIP gerçek-zamanlı (son 15 dk)",
    "opra_zincir": "OPRA opsiyon zinciri (IV + greek)",
    "opsiyon_oi": "Opsiyon açık pozisyon (OI)",
    "acik_satis_bayraklari": "Açığa satış bayrakları",
    "haber_akisi": "Haber akışı (Benzinga)",
    "alpaca": "Alpaca bağlantısı",
    "ag": "Ağ ölçümü",
}


def rapor(d: dict[str, Any]) -> str:
    """İnsan okunur tablo. Karakter kümesi bilinçli olarak ASCII+Türkçe (cp1254 güvenli)."""
    satir = ["", f"VERİ YETENEK ÖLÇÜMÜ — {d['olcum_zamani']}", "=" * 78]
    ayar = d.get("ayar", {})
    satir.append("Ayar: " + " · ".join(f"{k}={v}" for k, v in ayar.items()))
    dolu = [k for k, v in d.get("anahtarlar", {}).items() if v]
    bos = [k for k, v in d.get("anahtarlar", {}).items() if not v]
    satir += [f"Anahtar DOLU : {', '.join(dolu) if dolu else '(yok)'}",
              f"Anahtar BOŞ  : {', '.join(bos) if bos else '(yok)'}", "-" * 78]
    for key, r in d.get("yetenekler", {}).items():
        isim = _BASLIK.get(key, key)
        durum = {True: "[VAR]", False: "[YOK]", None: "[?]"}[r["ok"]]
        satir.append(f"{durum:6} {isim:38} {r['detay']}")
        if r.get("kilit"):
            satir.append(f"       {'':38} → açan: {r['kilit']}")
    satir += ["-" * 78,
              "Not: '[VAR]' bir KENAR iddiası değildir; yalnız verinin ELDE olduğunu söyler.", ""]
    return "\n".join(satir)


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Veri yeteneklerini ÖLÇ (salt-okunur)")
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL)
    ap.add_argument("--option-symbol", default=DEFAULT_OPTION_SYMBOL)
    ap.add_argument("--json", action="store_true", help="ham JSON bas")
    ap.add_argument("--offline", action="store_true", help="ağ çağrısı yapma")
    a = ap.parse_args(argv)
    d = probe(a.symbol, a.option_symbol, network=not a.offline)
    print(json.dumps(d, ensure_ascii=False, indent=2) if a.json else rapor(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
