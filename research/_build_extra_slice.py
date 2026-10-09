"""Geçici: EK evren dataset üretimini dilimle (paralel süreçler). Cache'te 1h verisi
OLAN ek-evren sembollerinin i::n dilimini yearly-windows backtest'ler.
Kullanım: python -m research._build_extra_slice <i> <n>  → data/_extra_part_<i>.csv"""
import sys

import pandas as pd

from nihai.config import DATA_DIR
from nihai.data.alpaca_history import HIST_DIR
from research.build_meta_dataset import build

i, n = int(sys.argv[1]), int(sys.argv[2])
uni = pd.read_csv(DATA_DIR / "extra_universe_ranked.csv")
have = [(r.symbol, r.status) for r in uni.itertuples()
        if (HIST_DIR / f"{r.symbol}_1h.csv").exists()]
have = have[i::n]
syms = [s for s, _ in have]
removed = {s for s, st in have if st == "inactive"}
total = build(syms, interval="1h", period="2016-01-01", mode="event_breakout",
              out=str(DATA_DIR / f"_extra_part_{i}.csv"), start="2016-01-01",
              removed=removed, provider="alpaca_hist", throttle=0, yearly_windows=True)
print(f"DILIM {i} TAMAM: {total} satır ({len(syms)} sembol)", flush=True)
