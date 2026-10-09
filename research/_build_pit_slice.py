"""Geçici: PIT dataset üretimini dilimle (symbols[i::n]) — paralel süreçler için.
Kullanım: python -m research._build_pit_slice <i> <n>  → data/_pit_part_<i>.csv"""
import sys

from nihai.config import DATA_DIR
from nihai.data.pit_universe import _current_and_changes, _norm, sp500_as_of
from research.build_meta_dataset import build

i, n = int(sys.argv[1]), int(sys.argv[2])
uni = sorted(sp500_as_of("2016-01-01"))
cur, _ = _current_and_changes()
removed = set(uni) - {_norm(s) for s in cur}
syms = uni[i::n]
total = build(syms, interval="1h", period="2016-01-01", mode="event_breakout",
              out=str(DATA_DIR / f"_pit_part_{i}.csv"), start="2016-01-01",
              removed=removed, provider="alpaca_hist", throttle=0, yearly_windows=True)
print(f"DILIM {i} TAMAM: {total} satır", flush=True)
