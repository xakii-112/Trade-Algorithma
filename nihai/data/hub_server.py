"""
hub_server.py — Process-dışı veri hub'ı (market-data gateway).

Tek bir upstream sağlayıcıya (varsayılan: alpaca_stream = 1 WebSocket bağlantısı)
bağlanır, bar/quote/opsiyon verisini yerel HTTP'den sunar. Birden çok proje/varyant
(ör. ana proje + HighWR70 kopyası) `DATA_PROVIDER=hub` ile buradan okur:

    • 1 Alpaca bağlantısı (eşzamanlı-bağlantı limiti sorunu biter)
    • iki projeye ÖZDEŞ veri (adil canlı A/B)
    • ~yarı API çağrısı (hub CachingProvider ile cache'ler)

Upstream ASLA 'hub' olamaz (kendine referans) — HUB_UPSTREAM_PROVIDER ile ayarlanır,
varsayılan alpaca_stream. Hub düşerse projelerin HubProvider'ı yfinance'e düşer (çökmez).

Kullanım:
    .venv\\Scripts\\python.exe hub_server.py
    # → http://127.0.0.1:8799  (HUB_HOST/HUB_PORT ile değişir)

.env:
    HUB_HOST=127.0.0.1   HUB_PORT=8799   HUB_UPSTREAM_PROVIDER=alpaca_stream
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from nihai.console import ensure_utf8_stdout

load_dotenv()

# Tur-106 (BULGU-106-3): elle blok yalnız stdout'u çeviriyordu; loguru'nun varsayılan çıkışı
# stderr'dir ve bekçi onu logs/hub_server.err.log'a yönlendirir → dosyanın satırlarının ~%26'sı
# cp1254 baytı taşıyordu ("�ekme hatas�"). Ortak yardımcı iki akışı da çevirir.
ensure_utf8_stdout()

# NOT: importlar load_dotenv()'den SONRA — config modülü import anında env okur,
# bu yüzden .env önce yüklenmeli (E402 bilinçli olarak susturuldu).
from fastapi import FastAPI, Response  # noqa: E402
from loguru import logger  # noqa: E402

from nihai.config import HUB_HOST, HUB_PORT, HUB_UPSTREAM_PROVIDER, LOG_DIR  # noqa: E402
from nihai.data import data_provider as _dp  # noqa: E402
from nihai.data.data_provider import get_provider  # noqa: E402

# Operasyon/failover loglarını kalıcı dosyaya da yaz (konsola ek olarak). '[failover]
# secondary ... kullanildi: SYM.method' uyarıları burada birikir → hangi sembolün/
# metodun neden yedeğe düştüğü sonradan incelenebilir. Rotasyon+retention disk şişirmez.
# Konum = config.LOG_DIR (proje kökü/logs): eski os.getcwd()/logs hangi klasörden
# başlatılırsa oraya 'logs/' saçıyordu (CWD'ye bağımlı, MainHub kopyası dosya-köklü).
logger.add(LOG_DIR / "hub.log", rotation="10 MB", retention=5,
           encoding="utf-8", level="INFO", enqueue=True, backtrace=False, diagnose=False)

if HUB_UPSTREAM_PROVIDER == "hub":
    raise SystemExit("[hub] HUB_UPSTREAM_PROVIDER 'hub' olamaz (kendine referans). "
                     "alpaca_stream / alpaca / yfinance kullan.")

app = FastAPI(title="NihaiAlgoritma Data Hub", version="1.0")
_provider = None


def provider():
    """Upstream sağlayıcıyı tek sefer kur (lazy). get_provider failover+cache sarmalar;
    alpaca_stream ise WS thread'i burada (tek process'te) başlar."""
    global _provider
    if _provider is None:
        _provider = get_provider(HUB_UPSTREAM_PROVIDER, with_failover=True, with_cache=True)
        logger.info(f"[hub] upstream hazır: {_provider.name}")
    return _provider


@app.get("/health")
def health():
    p = provider()
    stats = getattr(p, "source_stats", lambda: {})()
    return {"status": "ok", "upstream": HUB_UPSTREAM_PROVIDER, "provider": p.name, "source": stats,
            "pid": os.getpid(), "service_root": str(Path(__file__).resolve().parents[2])}


@app.get("/streams")
def streams():
    return {"streams": _dp.stream_status(provider()), "upstream": HUB_UPSTREAM_PROVIDER}


@app.get("/bars")
def bars(symbol: str, interval: str = "1h", lookback: int = 200):
    df = provider().get_bars(symbol.upper(), interval, lookback)
    if df is None or df.empty:
        return Response('{"columns":[],"index":[],"data":[]}', media_type="application/json")
    return Response(df.to_json(orient="split", date_format="iso"), media_type="application/json")


@app.get("/quote")
def quote(symbol: str):
    q = provider().get_quote(symbol.upper())
    if q is None:
        return {}
    ts = q.timestamp
    return {"symbol": q.symbol, "bid": q.bid, "ask": q.ask, "last": q.last,
            "spread_measured": q.spread_measured,
            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts)}


@app.get("/options")
def options(symbol: str, expiry: str | None = None):
    ch = provider().get_option_chain(symbol.upper(), expiry)
    if not ch:
        return {}
    out: dict = {"expiry": ch.get("expiry"), "underlying_price": ch.get("underlying_price"),
                 "all_expiries": ch.get("all_expiries")}
    for side in ("calls", "puts"):
        d = ch.get(side)
        out[side] = (d.to_json(orient="split", date_format="iso")
                     if (d is not None and hasattr(d, "empty") and not d.empty) else "")
    return out


@app.get("/asset_flags")
def asset_flags(symbol: str):
    """`shortable` / `easy_to_borrow` / `marginable` (Madde 30 — SHORT ön koşulu).

    Tur-58: hub bu bayrakları SUNMUYORDU → `DATA_PROVIDER=hub` koşan her süreç,
    hub'ın zaten kurulu Alpaca bağlantısını kullanmak yerine KENDİ trading istemcisini
    kurmak zorunda kalıyordu (hub'ın "tek upstream" amacı tam da bunu önlemek).

    Bayrak okunamazsa `{}` döner — istemci tarafında **None**'a çevrilir. "Bilinmiyor"
    ile "satılabilir" aynı şey DEĞİLDİR (Tur-31 disiplini); bu yüzden burada asla
    varsayılan bir bayrak uydurulmaz."""
    flags = _dp.asset_flags(symbol.upper(), provider())
    return flags or {}


def main():
    import uvicorn
    logger.info(f"[hub] başlıyor → http://{HUB_HOST}:{HUB_PORT}  (upstream={HUB_UPSTREAM_PROVIDER})")
    provider()   # uvicorn bloke etmeden önce upstream'i (ve WS thread'ini) ayağa kaldır
    uvicorn.run(app, host=HUB_HOST, port=HUB_PORT, log_level="warning")


if __name__ == "__main__":
    main()
