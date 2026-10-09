"""Geçici paralel ön-çekim yardımcı: sembol listesinin i::n dilimini indirir.
429/ağ hatasında sembol başına 3 deneme (30sn ara) — sembol kaybolmaz.
Kullanım: python -m research._prefetch_slice <i> <n>"""
import sys
import time

from nihai.data.alpaca_history import HIST_DIR, get_history
from nihai.data.pit_universe import sp500_as_of

i, n = int(sys.argv[1]), int(sys.argv[2])
syms = sorted(sp500_as_of("2016-01-01"))[i::n]
ok = miss = skip = 0
for j, s in enumerate(syms, 1):
    if (HIST_DIR / f"{s}_1h.csv").exists() or (HIST_DIR / f"{s.replace('-', '.')}_1h.csv").exists():
        skip += 1
        continue
    got = False
    for _attempt in range(3):
        try:
            df = get_history(s, "1h", start="2016-01-01")
            if len(df):
                got = True
                break
            # boş dönüş = hata (cache yazılmadı) VEYA gerçekten veri yok;
            # hata ihtimaline karşı bekle-yeniden-dene, son denemede bırak.
        except Exception as e:
            print(f"{s}: HATA {type(e).__name__}: {e}", flush=True)
        time.sleep(30)
    ok += 1 if got else 0
    miss += 0 if got else 1
    if j % 20 == 0:
        print(f"dilim {i}: {j}/{len(syms)} (dolu {ok} boş {miss} atlanan {skip})", flush=True)
    time.sleep(0.5)
print(f"dilim {i} BİTTİ: dolu {ok} / boş {miss} / atlanan {skip} / dilim {len(syms)}", flush=True)
