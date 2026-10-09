"""
build_extra_universe.py — PIT-S&P500 + watchlist DIŞINDA kalan ABD hisse evrenini
likidite-sıralı listeler (ek eğitim verisi çekimi için).

- Aktifler: NYSE/NASDAQ/AMEX (ARCA/BATS hariç → ETF'lerin çoğu elenir), sembol
  tamamen harf, ≤5 karakter; 30-günlük medyan dolar-hacmine göre AZALAN sıralanır
  (çekim bütçesi likit isimlere önce harcansın).
- İnaktifler (delist): aynı borsalar, ≤4 karakter; her 4 aktife 1 delist serpiştirilir.
  DÜRÜSTLÜK: yalnız-aktif evren survivorship-şişkindir; delist serpiştirme bunu
  kısmen dengeler ama bu ek set PIT-üyelik temelli DEĞİLDİR — rapora not düş.

Çıktı: data/extra_universe_ranked.csv (symbol,status,dollar_vol)
"""
from __future__ import annotations

import os
import random
import sys
from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from nihai.config import DATA_DIR  # noqa: E402


def main() -> None:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    from nihai.data.pit_universe import sp500_as_of
    from nihai.execution import watchlist as wl

    key, sec = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    tc = TradingClient(key, sec, paper=True)
    dc = StockHistoricalDataClient(key, sec)

    exclude = set(sp500_as_of("2016-01-01")) | set(wl.current())
    exclude |= {s.replace("-", ".") for s in exclude}

    # get_all_assets dönüşü `list[Asset] | RawData` olarak tiplenmiş; raw=False varsayılanında
    # daima Asset nesneleri gelir (cast çalışma zamanında no-op).
    act = cast(list[Any], tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY,
                                                             status=AssetStatus.ACTIVE)))
    ina = cast(list[Any], tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY,
                                                             status=AssetStatus.INACTIVE)))
    actives = sorted({a.symbol for a in act
                      if a.exchange and a.exchange.value in ("NYSE", "NASDAQ", "AMEX")
                      and a.symbol.isalpha() and len(a.symbol) <= 5} - exclude)
    inactives = sorted({a.symbol for a in ina
                        if a.exchange and a.exchange.value in ("NYSE", "NASDAQ", "AMEX")
                        and a.symbol.isalpha() and len(a.symbol) <= 4} - exclude)
    print(f"aday: aktif {len(actives)} | delist {len(inactives)}", flush=True)

    # Aktifleri 30g medyan dolar-hacmine göre sırala (500'lük çoklu-sembol istekler).
    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=30)
    vol: dict[str, float] = {}
    for i in range(0, len(actives), 500):
        chunk = actives[i:i + 500]
        try:
            resp = dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                start=start, end=end, adjustment=Adjustment.ALL, feed=DataFeed.SIP))
            for sym, bars in (getattr(resp, "data", {}) or {}).items():
                if bars:
                    vol[sym] = float(np.median([b.close * b.volume for b in bars]))
        except Exception as e:
            print(f"  hacim batch {i//500}: HATA {type(e).__name__}: {e}", flush=True)
        print(f"  hacim: {min(i+500, len(actives))}/{len(actives)}", flush=True)
    ranked_act = sorted(vol, key=lambda s: vol[s], reverse=True)
    print(f"hacim verisi olan aktif: {len(ranked_act)}", flush=True)

    rng = random.Random(42)
    ina_shuf = inactives[:]
    rng.shuffle(ina_shuf)

    rows, ai, ii = [], 0, 0
    while ai < len(ranked_act) or ii < len(ina_shuf):
        for _ in range(4):                       # 4 aktif : 1 delist
            if ai < len(ranked_act):
                s = ranked_act[ai]
                ai += 1
                rows.append({"symbol": s, "status": "active", "dollar_vol": vol[s]})
        if ii < len(ina_shuf):
            rows.append({"symbol": ina_shuf[ii], "status": "inactive", "dollar_vol": 0.0})
            ii += 1

    out = DATA_DIR / "extra_universe_ranked.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out} ({len(rows)} sembol)", flush=True)


if __name__ == "__main__":
    main()
