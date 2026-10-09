"""Tur-80 — canlı döngü 1m barlarını PORTFOLIO_LOCK DIŞINDA çeker + BULGU-80-5.

Eskiden `run_loop` kilidi açık hisse pozisyonu başına bir `get_bars` ağ çağrısı boyunca
tutuyordu; aynı kilidi 14 pano ucu ve tarama thread'leri bekler. Artık ağ `fetch_1m_bars`
ile kilit dışında, güncelleme kilit altında.

BULGU-80-5: eski döngü POZİSYON başınaydı ve her turda `update_all_with_bars`ı o sembolün
TÜM pozisyonlarına uyguluyordu → aynı sembolde iki pozisyon (küme sınırı 2) her barı iki
kez görüyordu: `bars_held` 2×, breakeven açıkken aynı barın taşıdığı stop aynı barda
tetiklenip ikisini de kapatıyordu (tek pozisyon açık kalıyor).

Denklik referansı `_eski_1m`: değişiklik öncesi gövdenin birebir kopyası.
"""
import ast
import threading
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nihai.config import CONFIG
from nihai.execution import main as canli
from nihai.execution.concurrency import PORTFOLIO_LOCK
from nihai.risk.position_manager import PositionManager, SlippageModel, liquidity_inputs
from nihai.risk.risk_manager import PortfolioState
from nihai.signals.strategy import Signal, SignalAction

T0 = datetime(2024, 1, 2, 9, 31)


def _eski_1m(self, provider, now):
    """Değişiklik öncesi `update_all_with_1m_bars` (yalnız denklik referansı)."""
    closed = []
    for pos in list(self.portfolio.open_positions):
        if pos.asset_type != "stock":
            continue
        try:
            lookback = max(3, CONFIG.backtest.impact_adv_bars + 16) if self.slip.impact_coef else 3
            df = provider.get_bars(pos.symbol, "1m", lookback_bars=lookback)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        bars = {pos.symbol: (float(last["open"]), float(last["high"]),
                             float(last["low"]), float(last["close"]))}
        inputs = liquidity_inputs(df.iloc[:-1]) if self.slip.impact_coef else {}
        inputs["spread_pct"] = None
        closed.extend(self.update_all_with_bars(bars, now=now, liquidity_by_symbol={pos.symbol: inputs}))
    return closed


class _Saglayici:
    """Sembol başına sabit rastgele yol; `adim` penceresinin son barı 'şimdiki' bardır."""

    def __init__(self, seed, bozuk=()):
        self.seed, self.adim, self.cagri, self.bozuk, self._yol = seed, 0, [], dict(bozuk), {}

    def get_bars(self, symbol, interval, lookback_bars):
        self.cagri.append(symbol)
        tur = self.bozuk.get((symbol, self.adim))
        if tur == "hata":
            raise RuntimeError("ağ")
        if tur == "yok":
            return None
        if tur == "bos":
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if symbol not in self._yol:
            rng = np.random.default_rng([self.seed, int(symbol[1:])])
            n = 400
            c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
            o = np.r_[100.0, c[:-1]] * np.exp(rng.normal(0, 0.002, n))
            h = np.maximum(o, c) * (1 + rng.uniform(0, 0.006, n))
            lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.006, n))
            self._yol[symbol] = pd.DataFrame(dict(open=o, high=h, low=lo, close=c,
                                                  volume=rng.uniform(1e4, 1e5, n)))
        son = 100 + self.adim
        return self._yol[symbol].iloc[max(0, son - lookback_bars + 1):son + 1].copy()


def _yonetici(impact=0.0):
    pf = PortfolioState(equity=100_000, cash_balance=100_000, ath_equity=100_000)
    return pf, PositionManager(pf, SlippageModel(impact_coef=impact), record_logs=False)


def _ac(m, symbol, uzun=True, risk=1.0):
    sig = Signal(SignalAction.LONG if uzun else SignalAction.SHORT, symbol, 100.0,
                 100 - risk if uzun else 100 + risk, 100 + 2 * risk if uzun else 100 - 2 * risk)
    return m.open_from_signal(sig, {"shares": 10}, now=datetime(2024, 1, 2, 9, 30))


def _kos(yol, seed, n_poz, impact, bozuk=(), adimlar=60):
    pf, m = _yonetici(impact)
    rng = np.random.default_rng(seed)
    for i in range(n_poz):
        _ac(m, f"S{i}", uzun=bool(rng.integers(2)), risk=float(rng.uniform(0.3, 1.5)))
    prov = _Saglayici(seed, bozuk)
    olaylar = []
    for adim in range(adimlar):
        prov.adim = adim
        olaylar += yol(m, prov, T0 + timedelta(minutes=adim))
    # repr: float'lar bit düzeyinde, NaN'lar eşit karşılaştırılır.
    durum = repr(([(vars(e.pos), e.exit_trigger, e.exit_price, e.exit_time, e.realized_pnl,
                    e.mae_pct, e.mfe_pct, e.bars_held, e.cost_attribution) for e in olaylar],
                  [vars(p) for p in pf.open_positions],
                  [vars(t) for t in m._trackers.values()],
                  (pf.equity, pf.cash_balance, pf.daily_pnl)))
    return durum, prov.cagri, len(olaylar)


_YOLLAR = {
    "eski": _eski_1m,
    "yeni": lambda m, p, now: m.update_all_with_1m_bars(p, now),
    "on_cekimli": lambda m, p, now: m.update_all_with_1m_bars(p, now, prefetched=m.fetch_1m_bars(p)),
}
_BOZUK = ((("S1", 5), "hata"), (("S2", 7), "yok"), (("S0", 9), "bos"), (("S3", 11), "hata"))


@pytest.mark.parametrize("be_trail", [False, True])
@pytest.mark.parametrize("impact", [0.0, 1.0])
def test_farkli_semboller_eski_yolla_birebir(be_trail, impact):
    kapanis = 0
    with patch.dict(vars(CONFIG.risk), enable_breakeven=be_trail, enable_trailing=be_trail):
        for seed in range(12):
            n_poz = 1 + seed % 5
            eski = _kos(_YOLLAR["eski"], seed, n_poz, impact, _BOZUK)
            for ad in ("yeni", "on_cekimli"):
                assert _kos(_YOLLAR[ad], seed, n_poz, impact, _BOZUK) == eski, (ad, seed)
            kapanis += eski[2]
    assert kapanis >= 12  # ızgara gerçekten çıkış üretiyor (boş denklik değil)


def test_ayni_sembolde_iki_pozisyon_bari_bir_kez_gorur():
    bar = pd.DataFrame(dict(open=[100.], high=[111.], low=[99.], close=[100.5], volume=[1e5]))
    cagri = []
    prov = SimpleNamespace(get_bars=lambda symbol, interval, lookback_bars: cagri.append(symbol) or bar)
    with patch.dict(vars(CONFIG.risk), enable_breakeven=True):
        (tek_pf, tek), (cift_pf, cift), (_, eski) = _yonetici(), _yonetici(), _yonetici()
        for m in (tek, cift, cift, eski, eski):  # hedef 130: bar yalnız breakeven'ı tetikler
            m.open_from_signal(Signal(SignalAction.LONG, "AAA", 100, 95, 130), {"shares": 10},
                               now=datetime(2024, 1, 2, 9, 30))
        assert tek.update_all_with_1m_bars(prov, T0) == []
        cagri.clear()
        assert cift.update_all_with_1m_bars(prov, T0) == []
        assert cagri == ["AAA"]
        assert [t.bars_held for t in cift._trackers.values()] == [1, 1]
        assert [p.stop for p in cift_pf.open_positions] == [tek_pf.open_positions[0].stop] * 2
        # Kusurun kaydı: eski gövde aynı barda taşınan stopla ikisini de kapatıyordu.
        assert [e.bars_held for e in _eski_1m(eski, prov, T0)] == [2, 2]


def test_on_bellek_disi_pozisyon_cekilir_hatali_sembol_yeniden_cekilmez():
    pf, m = _yonetici()
    _ac(m, "S0", risk=20)
    _ac(m, "S1", risk=20)
    prov = _Saglayici(3)
    frames = m.fetch_1m_bars(prov)
    assert prov.cagri == ["S0", "S1"]
    m.force_close(pf.open_positions[1], 100.0, "TEST", T0)  # arada kapanan → yok sayılır
    _ac(m, "S2", risk=20)                                   # arada açılan → çekilir
    frames["S0"] = None                                     # başarısız çekim → atlanır
    m.update_all_with_1m_bars(prov, T0, prefetched=frames)
    assert prov.cagri == ["S0", "S1", "S2"]
    assert {t.pos.symbol: t.bars_held for t in m._trackers.values()} == {"S0": 0, "S2": 1}


def test_fetch_kilidi_tutmaz():
    _, m = _yonetici()
    _ac(m, "S0")
    serbest = []

    def get_bars(symbol, interval, lookback_bars):
        def dene():
            if PORTFOLIO_LOCK.acquire(blocking=False):
                PORTFOLIO_LOCK.release()
                serbest.append(True)
        t = threading.Thread(target=dene)
        t.start()
        t.join()
        return None

    m.fetch_1m_bars(SimpleNamespace(get_bars=get_bars))
    assert serbest == [True]


def test_canli_dongu_bar_cekimini_kilit_disinda_yapar():
    cagrilar: dict[str, list[tuple[bool, ast.Call]]] = {}

    def gez(dugum, kilitli):
        if isinstance(dugum, ast.With) and any(
                "PORTFOLIO_LOCK" in ast.unparse(i.context_expr) for i in dugum.items):
            kilitli = True
        if isinstance(dugum, ast.Call) and isinstance(dugum.func, ast.Attribute):
            cagrilar.setdefault(dugum.func.attr, []).append((kilitli, dugum))
        for cocuk in ast.iter_child_nodes(dugum):
            gez(cocuk, kilitli)

    gez(ast.parse(Path(canli.__file__).read_text(encoding="utf-8")), False)
    (cekim_kilitli, _), = cagrilar["fetch_1m_bars"]
    (guncelleme_kilitli, guncelleme), = cagrilar["update_all_with_1m_bars"]
    assert not cekim_kilitli
    assert guncelleme_kilitli
    assert [k.arg for k in guncelleme.keywords] == ["prefetched"]
