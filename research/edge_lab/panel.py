"""
panel.py — aday veri seti kurucularının PAYLAŞTIĞI üç yardımcı (Tur-54).

`build_shortflow_dataset` (Tur-53) ve `build_optskew_dataset` (Tur-54) aynı üç şeye
ihtiyaç duyuyor: PIT fiyat paneli · aynı evrenin eşit-ağırlık ileri getirisi (§6 beta
kontrolü) · yön × FAZLALIK puanlaması. İkinci kurucuyu yazarken bunları kopyalamak,
bu deponun defalarca cezalandırdığı desenin ta kendisi olurdu (*"bir kusuru düzeltirken
grep ile AYNI DESENİ tüm repoda ara"*).

**Eski kurucular (overnight/leadlag/rebalance/vrp/insider) BİLEREK DOKUNULMADI.** Onların
kendi kopyaları YAYIMLANMIŞ sonuçların yanında donmuştur; ortak bir gövdeye taşımak
`EDGE_FINDINGS`'te sayısı basılı beş verdikti yeniden üretilebilir olmaktan çıkarma riski
taşır. Kopya borcu kayda geçti, sessizce kapatılmadı.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

PRICE_GLOB = "data/alpaca_hist/*_1d_all.csv"
TRADING_DAYS = 252


def load_panel(since: str, until: str, price_glob: str = PRICE_GLOB) -> dict[str, pd.DataFrame]:
    """PIT/survivorship-TEMİZ günlük panel: sembol → (date, close, volume).

    `*_1d_all.csv` düzeltilmiş kapanış taşır ve bugün endekste OLMAYAN isimleri de
    içerir (ACI DERS #1). 300 bardan kısa seriler atılır — z-skor pencereleri kurulamaz.

    ⚠️ BULGU-86-4: bu evren "pencerede HERHANGİ bir an üye" kümesidir (2021-01 üyeleri ∪
    pencerede EKLENENLER), karar gününün üyeliği DEĞİL. Süzülmezse sonradan eklenen isim
    geçmiş kesite girer (endekse eklenme çoğu zaman yükselişin SONUCUDUR). Ölçüldü: sekiz
    ailenin satırlarının %5,6–15,9'u karar günü üye olmayan isimden; vixts'te o satırların
    pnl'i anlamlı yüksek (t=+4,23), babmacro'da düşük (t=−2,18). Yeni aile karar günü
    üyeliğini `nihai.data.pit_universe.sp500_uyelik` ile süzmelidir (örnek:
    `build_rmom_dataset.sembol_satirlari`).
    """
    panel: dict[str, pd.DataFrame] = {}
    for f in glob.glob(price_glob):
        sym = os.path.basename(f).split("_1d_all")[0]
        try:
            d = pd.read_csv(f, usecols=["timestamp", "close", "volume"])
        except (OSError, ValueError):
            continue
        ts = pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
        d["date"] = ts.dt.tz_localize(None).dt.normalize()
        d = d.dropna(subset=["date", "close"]).sort_values("date", ignore_index=True)
        d = d[(d["date"] >= pd.Timestamp(since)) & (d["date"] <= pd.Timestamp(until))]
        if len(d) > 300:
            panel[sym] = d[["date", "close", "volume"]].reset_index(drop=True)
    return panel


def market_forward(panel: dict[str, pd.DataFrame], horizon: int) -> pd.Series:
    """AYNI evrenin eşit-ağırlık `horizon`-bar ileri getirisi (§6 beta tuzağı kontrolü).

    Ham ileri getiri bir kenar iddiası DEĞİLDİR: 2021-2026 boğasında her LONG ailesi
    kendiliğinden pozitif, her SHORT ailesi kendiliğinden negatif görünür.
    """
    wide = pd.DataFrame({s: d.set_index("date")["close"].astype(float)
                         for s, d in panel.items()}).sort_index()
    return (wide.shift(-horizon) / wide - 1.0).mean(axis=1)


def puanla(df: pd.DataFrame, horizon: int, cost_pct: float,
           borrow_bps_yr: float) -> pd.DataFrame:
    """Yön × FAZLALIK − maliyet; short bacağa ayrıca ödünç maliyeti.

    `f_is_short` 1 ise yön −1'dir. Puanlanan şey `fwd_ret` DEĞİL `excess`tir — kolon
    şemada olduğu için kontrol unutulamaz (Tur-52 dersi).
    """
    yon = 1.0 - 2.0 * df["f_is_short"]                          # long +1, short −1
    borrow = (borrow_bps_yr / 10_000.0) * (horizon / TRADING_DAYS) * df["f_is_short"]
    df["pnl"] = yon * df["excess"] - cost_pct / 100.0 - borrow
    df["label"] = (df["pnl"] > 0).astype(int)
    return df
