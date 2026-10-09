"""ib_smoke.py — IBKR canlı veri bağlantı testi (tek dosya, projeye dokunmaz).

Ön koşul:
  1) TWS veya IB Gateway AÇIK ve giriş yapılmış (API yerel sokettir, bulut REST değil).
  2) TWS → Global Configuration → API → Settings:
       [x] Enable ActiveX and Socket Clients
       [x] 127.0.0.1 → Trusted IPs   (Read-Only API: sadece veri çekecekseniz açık bırakın)
  3) pip install ib_async          (ib_insync artık bakımsız; ib_async onun devamı)

Çalıştır:  python ib_smoke.py AAPL
Çıktı canlı bid/ask basıyorsa market-data aboneliğiniz çalışıyor demektir.
"""
from __future__ import annotations

import math
import sys

from ib_async import IB, Stock, util

# TWS canlı/paper ve Gateway canlı/paper varsayılan portları — sırayla denenir.
PORTS = {"TWS paper": 7497, "TWS live": 7496, "Gateway paper": 4002, "Gateway live": 4001}

# Gürültü: bağlantı/veri-çiftliği bilgilendirmeleri, hata değil.
_QUIET = {2104, 2106, 2107, 2119, 2158}


def _is_price(v: float | None) -> bool:
    return v is not None and not math.isnan(v) and v > 0


def _on_error(reqId, code, msg, contract=None) -> None:
    if code not in _QUIET:
        print(f"  [IB {code}] {msg}")


def _connect(ib: IB) -> str:
    for label, port in PORTS.items():
        try:
            ib.connect("127.0.0.1", port, clientId=17, timeout=4)
            return f"{label} (port {port})"
        except Exception:
            continue
    sys.exit("TWS/IB Gateway'e bağlanılamadı → açık mı, giriş yapılmış mı, API etkin mi?")


def main(symbol: str = "AAPL") -> None:
    ib = IB()
    ib.errorEvent += _on_error
    print(f"bağlı: {_connect(ib)}")

    contract = Stock(symbol, "SMART", "USD")
    if not ib.qualifyContracts(contract):
        sys.exit(f"{symbol} sözleşmesi çözülemedi")

    # 1 = CANLI (abonelik ister), 3 = 15 dk GECİKMELİ (ücretsiz, abonelik istemez)
    for mdtype, label in ((1, "CANLI"), (3, "GECİKMELİ")):
        ib.reqMarketDataType(mdtype)
        tkr = ib.reqMktData(contract, "", snapshot=False)
        ib.sleep(4)
        if _is_price(tkr.bid) or _is_price(tkr.last):
            print(f"{label}: bid={tkr.bid} ask={tkr.ask} last={tkr.last} ts={tkr.time}")
            ib.cancelMktData(contract)
            break
        ib.cancelMktData(contract)
        print(f"{label}: veri gelmedi (abonelik yok / piyasa kapalı)")

    bars = ib.reqHistoricalData(contract, "", "1 D", "5 mins", "TRADES",
                                useRTH=True, formatDate=1)
    df = util.df(bars)
    print(df.tail(3).to_string() if df is not None and not df.empty else "bar gelmedi")
    ib.disconnect()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
