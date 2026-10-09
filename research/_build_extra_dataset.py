"""Geçici: EK evren (S&P-dışı ABD) dataset üretimi — çekim bittikten sonra koşulur.
Cache'te 1h verisi OLAN ek-evren sembollerinden yearly-windows meta-dataset üretir.
DÜRÜSTLÜK: bu set PIT-üyelik temelli DEĞİL (aktifler bugün-hayatta olanlar =
survivorship-şişkin; delist serpiştirmesi kısmen dengeler). Eğitim sonuçları
raporlanırken pit_sp500 setiyle AYNI kefeye konmaz; removed_since=1 delist demektir."""
import sys

import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.config import DATA_DIR
from nihai.data.alpaca_history import HIST_DIR
from research.build_meta_dataset import build

uni = pd.read_csv(DATA_DIR / "extra_universe_ranked.csv")
have = [(r.symbol, r.status) for r in uni.itertuples()
        if (HIST_DIR / f"{r.symbol}_1h.csv").exists()]
syms = [s for s, _ in have]
removed = {s for s, st in have if st == "inactive"}
print(f"cache'te veri olan ek-evren sembolü: {len(syms)} (delist: {len(removed)})", flush=True)

out = str(DATA_DIR / "meta_dataset_extra_us_event_breakout_1h_2016-01-01.csv")
n = build(syms, interval="1h", period="2016-01-01", mode="event_breakout",
          out=out, start="2016-01-01", removed=removed,
          provider="alpaca_hist", throttle=0, yearly_windows=True)
print(f"TOPLAM {n} satır -> {out}", flush=True)

df = pd.read_csv(out)
et = pd.to_datetime(df["entry_time"], errors="coerce")
print(f"DOĞRULAMA: satır {len(df)} | sembol {df['symbol'].nunique()} | "
      f"tarih {et.min().date()} - {et.max().date()}", flush=True)
print(f"kazanma %{100*df['label'].mean():.1f} | delist satır %{100*df['removed_since'].mean():.1f} | "
      f"NaN {df.isna().sum().sum()}", flush=True)
print("yıl dağılımı:", {int(k): int(v) for k, v in et.dt.year.value_counts().sort_index().items()}, flush=True)
