"""option_strategy._score_contract saf skorlama mantığı testleri.

Bu fonksiyon hangi opsiyon kontratının alınacağını sıralar (yüksek=iyi). Üç bileşenin
yön/monotonluğu davranış-kritik: hedef-delta yakınlığı (↑ iyi), likidite/OI (↑ iyi),
makas (↑ kötü). Yanlış işaret/ölçek yanlış kontratı seçtirir."""
import numpy as np
import pytest

from nihai.risk.option_strategy import _score_contract


def test_perfect_delta_match_yields_max_delta_component():
    # delta_dist=0 → delta_score 5.0; spread=0, OI=500 → 5 + log1p(500)
    s = _score_contract(target_delta=0.5, contract_delta=0.5,
                        open_interest=500, spread_pct=0.0)
    assert s == np.log1p(500) + 5.0


def test_closer_delta_scores_higher():
    near = _score_contract(target_delta=0.5, contract_delta=0.50,
                           open_interest=1000, spread_pct=0.0)
    far = _score_contract(target_delta=0.5, contract_delta=0.30,
                          open_interest=1000, spread_pct=0.0)
    assert near > far


def test_uses_absolute_delta_for_puts():
    # Put delta negatif; |delta| ile kıyaslanmalı → -0.5 ile +0.5 aynı skor
    call = _score_contract(target_delta=0.5, contract_delta=0.5,
                           open_interest=1000, spread_pct=0.01)
    put = _score_contract(target_delta=0.5, contract_delta=-0.5,
                          open_interest=1000, spread_pct=0.01)
    assert call == put


def test_more_open_interest_scores_higher():
    lo = _score_contract(target_delta=0.5, contract_delta=0.5,
                         open_interest=100, spread_pct=0.0)
    hi = _score_contract(target_delta=0.5, contract_delta=0.5,
                         open_interest=10000, spread_pct=0.0)
    assert hi > lo


def test_wider_spread_penalized():
    tight = _score_contract(target_delta=0.5, contract_delta=0.5,
                            open_interest=1000, spread_pct=0.01)
    wide = _score_contract(target_delta=0.5, contract_delta=0.5,
                           open_interest=1000, spread_pct=0.10)
    # %1 → %10 makas farkı tam (0.10-0.01)*50 = 4.5 puan ceza
    assert tight - wide == pytest.approx((0.10 - 0.01) * 50.0)


def test_delta_score_floors_at_zero():
    # Çok uzak delta (dist=0.5) → 5 - 0.5*20 = -5 → 0'a kırpılır, negatif olmaz
    s = _score_contract(target_delta=0.1, contract_delta=0.6,
                        open_interest=0, spread_pct=0.0)
    # OI=0 → log1p(0)=0; delta_score=0; spread=0 → toplam tam 0.0
    assert s == 0.0
