"""Geçici paralel ön-çekim (EK evren): extra_universe_ranked.csv'nin i::n dilimini
likidite sırasıyla indirir; SON TARİHE kadar (deadline) çalışır, sonra kendini durdurur.
Kullanım: python -m research._prefetch_extra <i> <n> <deadline "YYYY-MM-DD HH:MM">"""
import sys
import time
from datetime import datetime

import pandas as pd

from nihai.config import DATA_DIR
from nihai.data.alpaca_history import HIST_DIR, get_history

i, n = int(sys.argv[1]), int(sys.argv[2])
deadline = datetime.fromisoformat(sys.argv[3])
uni = pd.read_csv(DATA_DIR / "extra_universe_ranked.csv")
syms = list(uni["symbol"])[i::n]
ok = miss = skip = 0
for j, s in enumerate(syms, 1):
    if datetime.now() >= deadline:
        print(f"dilim {i}: SON TARİH ({deadline:%H:%M}) — duruldu", flush=True)
        break
    if (HIST_DIR / f"{s}_1h.csv").exists():
        skip += 1
        continue
    got = False
    for attempt in range(2):
        try:
            df = get_history(s, "1h", start="2016-01-01")
            if len(df) or attempt:   # boş dönüş 2. denemede de boşsa: gerçekten veri yok say
                got = len(df) > 0
                break
        except Exception as e:
            print(f"{s}: HATA {type(e).__name__}: {e}", flush=True)
        time.sleep(20)
    ok += 1 if got else 0
    miss += 0 if got else 1
    if j % 25 == 0:
        print(f"dilim {i}: {j}/{len(syms)} (dolu {ok} boş {miss} atlanan {skip})", flush=True)
    time.sleep(0.3)
print(f"dilim {i} BİTTİ: dolu {ok} / boş {miss} / atlanan {skip}", flush=True)
