"""B2 — purged_folds SOL PURGE (sızıntı) değişmezleri.

Bağlam: eskiden purged_folds yalnız SAĞ embargo uyguluyor, test bloğunun SOLUNDAKİ tüm
`[0, lo)` satırlarını 'test'ten önce → güvenli' sayıp eğitime alıyordu. Ama finansal etiket
ileriye bakar (giriş t → sonuç t+horizon): test bloğuna BİTİŞİK bir eğitim satırının etiket
penceresi test dönemine taşar → gelecek-bilgisi eğitime sızar, OOS-AUC şişer. Sol purge bunu
keser. Bu testler yalnız indeks aritmetiğini doğrular (model/veri sürümünden bağımsız)."""
from __future__ import annotations

import pandas as pd

from research.purged_cv import purged_folds, suggest_purge


def test_no_train_label_window_crosses_test_block():
    """horizon uzunluğunda etiket penceresi olan hiçbir SOL eğitim satırı test bloğuna
    taşmamalı: purge=embargo=horizon ile [j, j+horizon] test [lo, hi)'ye değmez."""
    n, k, horizon = 200, 5, 6
    for tr, te in purged_folds(n, k, embargo=horizon):
        lo = int(te.min())
        for j in tr[tr < lo]:            # yalnız test'ten ÖNCEKI eğitim satırları
            assert j + horizon < lo, f"sol sızıntı: {j}+{horizon} test başı {lo}'ye taşıyor"


def test_left_purge_removes_rows_adjacent_to_test_block():
    """Sol purge, test bloğuna bitişik `purge` satırı eğitimden çıkarır (regresyon: eski
    naive davranış [0, lo)'yu tümüyle alıyordu → o bant sızıntıydı)."""
    n, k, embargo = 100, 5, 4
    folds = list(purged_folds(n, k, embargo))
    tr, te = folds[1]                    # bounds=[0,20,40,60,80,100] → 2. fold lo=20
    lo = int(te.min())
    left = tr[tr < lo]
    assert left.max() == lo - embargo - 1                      # en son sol satır = 15
    assert set(range(lo - embargo, lo)).isdisjoint(tr.tolist())  # 16..19 purge edildi


def test_right_embargo_band_still_excluded():
    """Sağ embargo korunur: test'ten hemen sonraki embargo bandı eğitimde OLMAMALI."""
    n, k, embargo = 200, 5, 6
    for tr, te in purged_folds(n, k, embargo):
        hi = int(te.max()) + 1
        band = set(range(hi, min(hi + embargo, n)))
        assert band.isdisjoint(tr.tolist())


def test_folds_disjoint_and_test_covers_all():
    """Test blokları ayrık + [0, n)'i tam kaplar; eğitim/test kesişmez."""
    n, k, embargo = 200, 5, 6
    covered: list[int] = []
    for tr, te in purged_folds(n, k, embargo):
        assert set(tr.tolist()).isdisjoint(te.tolist())
        assert len(te) > 0
        covered.extend(te.tolist())
    assert sorted(covered) == list(range(n))   # her satır tam bir test bloğunda


def test_explicit_purge_param_overrides_symmetric_default():
    """purge açıkça verilince sol bant o kadar (embargo'dan bağımsız) kırpılır."""
    n, k, embargo, purge = 100, 5, 2, 10
    tr, te = list(purged_folds(n, k, embargo, purge=purge))[1]
    lo = int(te.min())                   # lo=20
    left = tr[tr < lo]
    assert left.max() == lo - purge - 1  # 9 (embargo=2 değil, purge=10 baz alınır)


# --- P2-1: suggest_purge (bars_held → sol purge satırı) ---

def test_suggest_purge_single_symbol_rows_equal_bars():
    """Satır=bar (her bar bir satır) durumunda purge = horizon (q95 bars_held)."""
    df = pd.DataFrame({"bars_held": [6] * 100,
                       "entry_time": [f"t{i:03d}" for i in range(100)]})
    assert suggest_purge(df) == 6


def test_suggest_purge_scales_with_row_density():
    """Aynı bara 4 satır (4 sembol) düşüyorsa purge 4 katına çıkar — bar cinsinden
    horizon'u satır cinsine çevirmeden purge yetersiz kalır (sızıntı yönü)."""
    ts = [f"t{i:03d}" for i in range(25) for _ in range(4)]
    df = pd.DataFrame({"bars_held": [6] * 100, "entry_time": ts})
    assert suggest_purge(df) == 24


def test_suggest_purge_missing_or_invalid_bars_held_returns_none():
    """bars_held yok / hepsi ≤0 ya da parse edilemez → None (embargo-simetrik davranış)."""
    assert suggest_purge(pd.DataFrame({"x": [1, 2]})) is None
    df = pd.DataFrame({"bars_held": [0, -1, float("nan"), ""],
                       "entry_time": ["a", "b", "c", "d"]})
    assert suggest_purge(df) is None


def test_suggest_purge_outlier_robust_and_capped():
    """q95: tek aykırı bars_held purge'ü şişirmez; tavan cap_frac*n dejenere fold önler."""
    df = pd.DataFrame({"bars_held": [3] * 99 + [1000],
                       "entry_time": [f"t{i:03d}" for i in range(100)]})
    assert suggest_purge(df) == 3        # q95 → 3; 1000'lük aykırı yok sayılır
    df2 = pd.DataFrame({"bars_held": [90] * 30,
                        "entry_time": [f"t{i:03d}" for i in range(30)]})
    assert suggest_purge(df2) == 10      # tavan: int(30/3)
