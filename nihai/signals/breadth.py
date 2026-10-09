"""
breadth.py — PİYASA BAĞLAMI özellikleri: genişlik, sektör göreceli gücü, çapraz varlık.

NEDEN (Tur-59, dış rapor F1-F4): `meta_features.FEATURE_NAMES` 25 boyutun tamamını TEK
hissenin teknik göstergelerinden türetir. Sistem, taradığı ismin piyasanın geri kalanına
göre nerede durduğunu HİÇ görmez: kaç isim SMA50 üstünde, yükselen/düşen oranı ne, sembol
benchmark'a göre ayrışıyor mu, tahvil/dolar/petrol ne yapıyor.

BU MODÜL KARAR ÜRETMEZ. Ölçtüğü sayılar `live_features_log.csv`ye EK KOLON olarak yazılır
(FEATURE_NAMES'e GİRMEZ → model girdisi ve mevcut ölçümler DEĞİŞMEZ). Gerekçe projenin
kendi disiplinidir: kanıtsız bir bileşeni skora eklemek yasaktır (Tur-51), ama ölçülmemiş
bir bilgiyi hiç biriktirmemek de darboğazın ta kendisidir (ACI DERS #5). Önce veri birikir,
sonra `research/edge_lab` bunu bir AİLE olarak sınar.

NEDENSELLİK: tüm fonksiyonlar yalnız verilen serilerin GEÇMİŞİNİ kullanır. Canlı hatta
değerler bir ÖNCEKİ tarama turundan gelir (`MarketContext.snapshot`), yani karar barından
kesinlikle önceki bilgidir.

ÖLÇÜM KOŞULU TAŞINIR (Tur-34): ölçülemeyen her alan None döner, uydurma nötr değer YOK.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

# Ek kolon adları — TEK kaynak (prediction_tracker.FEAT_FIELDS bunu okur).
CONTEXT_FIELDS = [
    "br_above_sma50",   # SMA50 üstündeki sembol payı [0,1]
    "br_adv_ratio",     # son barda yükselen sembol payı [0,1]
    "br_mcclellan",     # McClellan osilatörü (EMA19-EMA39, net yükseliş oranı üzerinde)
    "br_n",             # ölçüme giren sembol sayısı (PAYDA — Tur-33: oran paydasız okunmaz)
    "rs_bench_20",      # sembolün 20-bar getirisi eksi benchmark'ın 20-bar getirisi
    "xa_bond_20",       # çapraz varlık: tahvil vekili 20-bar getirisi
    "xa_dollar_20",     # çapraz varlık: dolar vekili 20-bar getirisi
    "xa_oil_20",        # çapraz varlık: petrol vekili 20-bar getirisi
]

# Çapraz varlık VEKİLLERİ. ^TNX/DXY endeksleri hisse sağlayıcılarında yoktur; ETF vekilleri
# aynı riski TAŞIR ama AYNI ŞEY DEĞİLDİR (TLT uzun-vade faiz duyarlılığı, UUP DXY sepetini
# yaklaşık izler, USO vadeli-roll taşır). Vekil olduğu alan adında görünür kalsın diye
# adlar "bond/dollar/oil"dur, "TNX/DXY/WTI" değil.
CROSS_ASSET_PROXIES = {"bond": "TLT", "dollar": "UUP", "oil": "USO"}

# SAYIM alanları (oran değil) — CSV'ye tam sayı yazılır. `br_n` bir PAYDADIR ve
# "12.0" diye yazılması onu bir orana benzetir (Tur-33: oran paydasız okunmaz →
# payda da oran gibi görünmemelidir).
COUNT_FIELDS = {"br_n"}

_MIN_BREADTH_SYMBOLS = 10   # Tur-33: yönlü sayım 10'un altındaysa oran yayımlanmaz


def _sma(closes: np.ndarray, n: int) -> float | None:
    if closes.size < n:
        return None
    v = float(np.mean(closes[-n:]))
    return v if np.isfinite(v) else None


def symbol_state(close) -> dict | None:
    """Bir sembolün genişlik katkısı: {'above_sma50': bool, 'advanced': bool} — yoksa None.

    `close` en az 51 barlık kapanış serisi/dizisi olmalı (SMA50 + bir önceki bar)."""
    try:
        c = np.asarray(pd.Series(close).astype(float).dropna().to_numpy(), dtype=float)
    except (TypeError, ValueError):
        return None
    if c.size < 51:
        return None
    sma50 = _sma(c, 50)
    if sma50 is None or not np.isfinite(c[-1]) or not np.isfinite(c[-2]):
        return None
    return {"above_sma50": bool(c[-1] > sma50), "advanced": bool(c[-1] > c[-2])}


def breadth(states: list[dict]) -> dict:
    """Sembol durumlarından genişlik oranları. n < 10 ise oranlar None (payda kuralı)."""
    n = len(states)
    if n < _MIN_BREADTH_SYMBOLS:
        return {"br_above_sma50": None, "br_adv_ratio": None, "br_n": n}
    return {
        "br_above_sma50": sum(1 for s in states if s["above_sma50"]) / n,
        "br_adv_ratio": sum(1 for s in states if s["advanced"]) / n,
        "br_n": n,
    }


def mcclellan(adv_ratio_history, *, fast: int = 19, slow: int = 39) -> float | None:
    """McClellan osilatörü: EMA(fast) - EMA(slow), NET yükseliş oranı (2*adv_ratio-1) üzerinde.

    Klasik tanım (adv-dec)/(adv+dec) oranıdır; `2*adv_ratio - 1` bunun cebirsel eşitidir
    (adv+dec = n). En az `slow` gözlem yoksa None — kısa geçmişten osilatör uydurulmaz."""
    h = [float(x) for x in adv_ratio_history if x is not None and np.isfinite(float(x))]
    if len(h) < slow:
        return None
    s = pd.Series([2.0 * x - 1.0 for x in h])
    v = float(s.ewm(span=fast, adjust=False).mean().iloc[-1]
              - s.ewm(span=slow, adjust=False).mean().iloc[-1])
    return v if np.isfinite(v) else None


def rel_return(close, window: int = 20) -> float | None:
    """Son `window` barın basit getirisi — ölçülemezse None."""
    try:
        c = pd.Series(close).astype(float).dropna().to_numpy(dtype=float)
    except (TypeError, ValueError):
        return None
    if c.size < window + 1 or c[-window - 1] <= 0:
        return None
    v = float(c[-1] / c[-window - 1] - 1.0)
    return v if np.isfinite(v) else None


def relative_strength(sym_close, bench_close, window: int = 20) -> float | None:
    """Sembol getirisi eksi benchmark getirisi (aynı pencere). Biri ölçülemezse None.

    Beta-ayarlı DEĞİLDİR: bu ham ayrışmadır. Beta düzeltmesi için
    `market_regime.beta_vs_benchmark` ayrı bir ölçümdür ve ayrı raporlanır — iki sayıyı
    tek alanda birleştirmek hangi kısmın piyasa hangi kısmın isim olduğunu gizlerdi."""
    a, b = rel_return(sym_close, window), rel_return(bench_close, window)
    if a is None or b is None:
        return None
    return a - b


class MarketContext:
    """Tarama turları arasında piyasa bağlamını biriktiren hafif toplayıcı.

    Kullanım: her turda gezilen sembol için `observe(symbol, closes)`, tur sonunda
    `roll()`. `snapshot()` BİR ÖNCEKİ turun bitmiş ölçümünü döndürür → o turda yazılan
    her özellik satırı için değer KARAR BARINDAN ÖNCEKİ bilgidir (look-ahead yok) ve tur
    içinde SABİTTİR (aynı turdaki tüm semboller aynı bağlamı görür — `_load_calibration`
    ile aynı sözleşme, Tur-28)."""

    def __init__(self, history: int = 120) -> None:
        self._cur: dict[str, dict] = {}
        self._adv_hist: deque = deque(maxlen=history)
        self._snap: dict = dict.fromkeys(CONTEXT_FIELDS)
        self._snap["br_n"] = 0

    def pending(self) -> int:
        """Bu turda gözlenen sembol sayısı. `roll()` yalnız bu > 0 iken çağrılmalıdır:
        canlı serviste bar başına birden çok TUR koşar ama gösterge önbelleği yüzünden
        yalnız BAR DEĞİŞTİĞİNDE gözlem düşer → McClellan geçmişi bar başına TEK kez
        ilerler (aksi halde aynı barı 20 kez sayıp osilatörü düzleştirirdi)."""
        return len(self._cur)

    def observe(self, symbol: str, close) -> None:
        st = symbol_state(close)
        if st is not None:
            self._cur[symbol] = st

    def roll(self) -> dict:
        """Turu kapat: genişliği hesapla, McClellan geçmişini ilerlet, snapshot'ı yenile."""
        b = breadth(list(self._cur.values()))
        if b["br_adv_ratio"] is not None:
            self._adv_hist.append(b["br_adv_ratio"])
        snap: dict = dict.fromkeys(CONTEXT_FIELDS)
        snap.update(b)
        snap["br_mcclellan"] = mcclellan(self._adv_hist)
        self._snap = snap
        self._cur = {}
        return snap

    def snapshot(self) -> dict:
        """Son TAMAMLANMIŞ turun bağlamı (kopya)."""
        return dict(self._snap)


def demo() -> None:
    """assert tabanlı öz-denetim: `python -m nihai.signals.breadth`."""
    idx = pd.date_range("2026-01-01", periods=80, freq="D")
    up = pd.Series(np.linspace(100, 140, 80), index=idx)
    down = pd.Series(np.linspace(140, 100, 80), index=idx)
    assert symbol_state(up) == {"above_sma50": True, "advanced": True}
    assert symbol_state(down) == {"above_sma50": False, "advanced": False}
    assert symbol_state(up.head(10)) is None                    # yetersiz bar → ölçüm yok

    # Payda kuralı: 10'un altında oran YAYIMLANMAZ (Tur-33)
    few = [{"above_sma50": True, "advanced": True}] * 9
    assert breadth(few)["br_above_sma50"] is None and breadth(few)["br_n"] == 9
    mixed = ([{"above_sma50": True, "advanced": True}] * 6
             + [{"above_sma50": False, "advanced": False}] * 4)
    assert abs(breadth(mixed)["br_above_sma50"] - 0.6) < 1e-12

    def _mc(h: list[float]) -> float:
        v = mcclellan(h)
        assert v is not None
        return v

    assert mcclellan([0.5] * 38) is None                        # geçmiş kısa → None
    assert abs(_mc([0.5] * 60)) < 1e-12                         # nötr piyasa → 0
    # Osilatör SEVİYEYİ değil DEĞİŞİMİ ölçer: sabit "hep yükseliş" serisinde iki EMA de
    # 1.0'a yakınsar ve fark ~0 çıkar (doğru davranış). İşaret ancak genişlik DÖNDÜĞÜNDE
    # doğar; test tam da bunu kilitler.
    assert abs(_mc([1.0] * 200)) < 1e-3                         # doymuş → fark ~0
    assert _mc([0.0] * 40 + [1.0] * 15) > 0                     # düşüşten dönüş → pozitif
    assert _mc([1.0] * 40 + [0.0] * 15) < 0                     # yükselişten dönüş → negatif

    assert relative_strength(up, up) == 0.0                     # aynı seri → ayrışma yok
    rs = relative_strength(up, down)
    assert rs is not None and rs > 0
    assert rel_return(up.head(3)) is None

    mc = MarketContext()
    assert mc.snapshot()["br_above_sma50"] is None              # tur yok → ölçüm yok
    for i in range(12):
        mc.observe(f"S{i}", up if i < 7 else down)
    snap = mc.roll()
    assert snap["br_n"] == 12 and abs(snap["br_above_sma50"] - 7 / 12) < 1e-12
    assert mc.snapshot()["br_adv_ratio"] == snap["br_adv_ratio"]
    print("breadth demo OK")


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout

    ensure_utf8_stdout()
    demo()
