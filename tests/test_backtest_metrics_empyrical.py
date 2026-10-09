"""P0-4 — backtest metriklerinin empyrical-reloaded ile ÇAPRAZ DOĞRULAMASI (test-only).

empyrical-reloaded (Stefan Jansen bakımı) RUNTIME bağımlılığı DEĞİLDİR — yalnız bu dosya
bağımsız bir referans implementasyona karşı parite kilitler (proje parite kültürünün devamı:
tests/test_greeks_parity.py, tests/test_perf_parity_round10.py). Kurulu değilse dosya atlanır
(kurulum: pip install empyrical-reloaded).

Kilitlenen sözleşmeler:
- ``sortino_annualized`` ≡ ``empyrical.sortino_ratio(annualization=ppy)`` — BİREBİR
  (downside deviation'ın TÜM-N'e bölünmesi konvansiyonu dahil; backtest docstring'i bu
  özdeşliği iddia ediyor, burada sayısal olarak kanıtlanır).
- ``sharpe_annualized`` = ``empyrical.sharpe_ratio`` × sqrt((n-1)/n) — BİLİNEN TEK fark:
  bizim payda pstdev (popülasyon, ddof=0), empyrical ddof=1. Fark çarpanı burada AÇIKÇA
  kilitlenir → iki yönde de sessiz konvansiyon kayması yakalanır (çarpan kaybolursa da,
  başka bir fark sızarsa da test kırılır).
- ``max_drawdown(equity_curve)`` ≡ −``empyrical.max_drawdown(returns)``×100 — empyrical de
  seriye başlangıç-değeri noktası ekler (start=100), bizim equity_curve[0] tabanıyla eşdeğer.
- Ayrışma sözleşmeleri: tüm-pozitif seride sortino=0.0 (empyrical inf/nan üretirdi; JSON/log
  güvenliği kararı — backtest docstring) ve ``profit_factor``'ın None sözleşmesi (empyrical'de
  karşılığı yok → elle hesaplanmış brüt kâr/brüt zararla doğrulanır).
"""
import math

import numpy as np
import pytest

empyrical = pytest.importorskip("empyrical", reason="empyrical-reloaded kurulu değil (test-only bağımlılık)")

from nihai.execution.backtest import (  # noqa: E402 — importorskip'ten sonra kasıtlı
    max_drawdown,
    profit_factor,
    sharpe_annualized,
    sortino_annualized,
)

# Seriler kasıtlı çeşitli: karışık işaret, eğilimli, kayıp-ağırlıklı, gürültülü (seed'li).
_RNG = np.random.default_rng(42)
_SERIES = {
    "karisik": [0.010, -0.005, 0.020, -0.015, 0.007, -0.002, 0.012, -0.008],
    "egilimli": [0.004, 0.006, -0.001, 0.009, 0.003, -0.002, 0.008, 0.005, 0.001, -0.003],
    "kayip_agirlikli": [-0.010, 0.004, -0.020, -0.006, 0.010, -0.015, 0.002, -0.007],
    "gurultulu_120bar": list(_RNG.normal(0.0005, 0.01, 120)),
}
# Günlük (252) + saatlik (252*6.5 → int 1638; backtest interval haritasıyla aynı) ölçek.
_PPY = [252, 1638]


def _cases():
    return [(name, ppy) for name in _SERIES for ppy in _PPY]


# ---------------------------------------------------------------------------
# Sortino — empyrical ile BİREBİR parite
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,ppy", _cases())
def test_sortino_birebir_empyrical_paritesi(name, ppy):
    r = _SERIES[name]
    ours = sortino_annualized(r, ppy)
    ref = float(empyrical.sortino_ratio(np.asarray(r, dtype=float), required_return=0.0,
                                        annualization=ppy))
    assert math.isfinite(ours)
    assert ours == pytest.approx(ref, rel=1e-9)


def test_sortino_hedefli_target_paritesi():
    """target!=0 yolu da empyrical required_return ile örtüşmeli."""
    r = _SERIES["karisik"]
    ours = sortino_annualized(r, 252, target=0.001)
    ref = float(empyrical.sortino_ratio(np.asarray(r), required_return=0.001, annualization=252))
    assert ours == pytest.approx(ref, rel=1e-9)


# ---------------------------------------------------------------------------
# Sharpe — bilinen ddof farkı AÇIKÇA kilitlenir
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,ppy", _cases())
def test_sharpe_ddof_carpaniyla_empyrical_paritesi(name, ppy):
    r = _SERIES[name]
    n = len(r)
    ours = sharpe_annualized(r, ppy)
    ref = float(empyrical.sharpe_ratio(np.asarray(r, dtype=float), risk_free=0.0,
                                       annualization=ppy))
    # ours = mean/pstdev(ddof=0) · √ppy ; empyrical = mean/std(ddof=1) · √ppy
    # → ours = empyrical · √(n/(n-1)). Çarpan kaybolursa DA başka fark sızarsa DA kırılır.
    assert ours == pytest.approx(ref * math.sqrt(n / (n - 1)), rel=1e-9)


# ---------------------------------------------------------------------------
# Max drawdown — equity eğrisi ↔ getiri serisi eşdeğerliği
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(_SERIES))
def test_max_drawdown_empyrical_paritesi(name):
    r = np.asarray(_SERIES[name], dtype=float)
    base = 25_000.0
    equity_curve = [base] + list(base * np.cumprod(1.0 + r))
    ours_pct = max_drawdown(equity_curve)                       # pozitif yüzde
    ref = float(empyrical.max_drawdown(r))                      # negatif ondalık
    assert ours_pct == pytest.approx(-ref * 100.0, rel=1e-9)


def test_max_drawdown_hep_yukari_sifir():
    r = np.asarray([0.01, 0.02, 0.005], dtype=float)
    equity_curve = [100.0] + list(100.0 * np.cumprod(1.0 + r))
    assert max_drawdown(equity_curve) == 0.0
    assert float(empyrical.max_drawdown(r)) == 0.0


# ---------------------------------------------------------------------------
# Ayrışma sözleşmeleri — kasıtlı farklar (JSON güvenliği)
# ---------------------------------------------------------------------------
def test_sortino_tum_pozitifte_sifir_sozlesmesi():
    """Hedefin altına hiç inilmemiş seri: empyrical inf/nan üretir; biz 0.0 (sonlu, JSON-güvenli).
    Bu KASITLI ayrışmadır — backtest.sortino_annualized docstring'inde belgeli."""
    assert sortino_annualized([0.01, 0.02, 0.005, 0.03], 252) == 0.0


def test_kisa_seri_sifir_sozlesmesi():
    assert sharpe_annualized([], 252) == 0.0
    assert sharpe_annualized([0.01], 252) == 0.0
    assert sortino_annualized([], 252) == 0.0
    assert sortino_annualized([0.01], 252) == 0.0


def test_profit_factor_elle_hesap():
    """empyrical'de trade-PnL profit factor yok → elle hesaplanmış referansla kilitle."""
    assert profit_factor([10.0, -5.0, 20.0, -10.0]) == pytest.approx(2.0)
    assert profit_factor([-5.0, 0.0, 10.0]) == pytest.approx(2.0)   # 0-PnL iki tarafa da girmez
    assert profit_factor([-3.0, -7.0]) == pytest.approx(0.0)        # hiç kazanç yok → 0.0 (tanımlı)


def test_profit_factor_none_sozlesmesi():
    """Brüt zarar 0 → None (inf DEĞİL): RFC 8259 JSON'da Infinity yok; tüketiciler null bekler."""
    assert profit_factor([]) is None
    assert profit_factor([5.0, 12.0]) is None
    assert profit_factor([0.0, 0.0]) is None
