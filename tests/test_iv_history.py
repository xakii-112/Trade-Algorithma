"""iv_history.get_iv_rank güven-katmanı + rank matematiği testleri (Madde 17 destek).

IV rank, long-opsiyon girişini kapılar; güven katmanı (NO_HISTORY/LOW/PARTIAL/OK) ve
0-100 ölçeği yanlışsa şişmiş IV'de alım yapılabilir. IV_HIST_DIR tmp_path'e
yönlendirilir → diske gerçek CSV yazılır, ağ/global cache yok."""
import pandas as pd
import pytest

import nihai.data.iv_history as ivh
from nihai.data.iv_history import get_iv_rank


@pytest.fixture(autouse=True)
def _isolate_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ivh, "IV_HIST_DIR", tmp_path)
    return tmp_path


def _write(symbol, ivs):
    df = pd.DataFrame({"date": [f"2026-01-{i+1:02d}" for i in range(len(ivs))], "iv": ivs})
    df.to_csv(ivh.IV_HIST_DIR / f"{symbol}.csv", index=False)


def test_no_file_returns_no_history():
    r = get_iv_rank("ZZZ", 0.3)
    assert r.confidence == "NO_HISTORY" and r.iv_rank == 50.0 and r.sample_size == 0


def test_too_few_days_low_confidence():
    _write("AAA", [0.2, 0.25, 0.3])           # 3 < MIN_DAYS_PARTIAL
    r = get_iv_rank("AAA", 0.25)
    assert r.confidence.startswith("LOW_CONFIDENCE") and r.iv_rank == 50.0


def test_flat_iv_no_range():
    _write("BBB", [0.30] * 40)                 # aralık yok
    r = get_iv_rank("BBB", 0.30)
    assert r.confidence == "NO_RANGE" and r.iv_rank == 50.0


def test_rank_scales_0_to_100():
    ivs = [0.10 + 0.01 * i for i in range(40)]  # 0.10 .. 0.49, n=40 → PARTIAL
    _write("CCC", ivs)
    lo, hi = min(ivs), max(ivs)
    assert get_iv_rank("CCC", lo).iv_rank == pytest.approx(0.0)
    assert get_iv_rank("CCC", hi).iv_rank == pytest.approx(100.0)
    assert get_iv_rank("CCC", (lo + hi) / 2).iv_rank == pytest.approx(50.0)


def test_rank_clipped_beyond_range():
    _write("DDD", [0.10 + 0.01 * i for i in range(40)])
    assert get_iv_rank("DDD", 0.01).iv_rank == 0.0      # min'in altı → 0
    assert get_iv_rank("DDD", 9.99).iv_rank == 100.0    # max'ın üstü → 100


def test_partial_vs_ok_confidence():
    _write("EEE", [0.10 + 0.001 * i for i in range(40)])     # 40 → PARTIAL
    assert get_iv_rank("EEE", 0.12).confidence.startswith("PARTIAL")
    _write("FFF", [0.10 + 0.001 * i for i in range(120)])    # 120 → OK
    assert get_iv_rank("FFF", 0.15).confidence == "OK"


def test_measured_ayirt_edici_uydurulan_50_den():
    """UYDURULAN 50.0, gerçek 50.0'dan ayırt edilebilmeli.

    Üç ölçümsüz dal da 50.0 döner ve bu değer `signal_quality_score`'un bonus
    bandının ([30,50]) TAM İÇİNDEDİR → bayrak olmadan ölçüm yokluğu +0.5 ödül alır.
    """
    assert get_iv_rank("ZZZ", 0.3).measured is False              # NO_HISTORY
    _write("AAA", [0.2, 0.25, 0.3])
    assert get_iv_rank("AAA", 0.25).measured is False             # LOW_CONFIDENCE
    _write("BBB", [0.30] * 40)
    assert get_iv_rank("BBB", 0.30).measured is False             # NO_RANGE
    # ...ama GERÇEKTEN 50.0 ölçülen rank ölçülmüş sayılır
    ivs = [0.10 + 0.01 * i for i in range(40)]
    _write("CCC", ivs)
    gercek = get_iv_rank("CCC", (min(ivs) + max(ivs)) / 2)
    assert gercek.iv_rank == pytest.approx(50.0) and gercek.measured is True


def test_olcumsuz_iv_rank_skora_bonus_VERMEZ():
    """Uçtan uca: ölçümsüz IV rank `signal_quality_score`'a girmemeli.

    `main.py` ölçümsüzü None'a çevirir; bu test bonusun gerçekten kaybolduğunu
    (0.5 fark) kilitler — bandın kendisi doğru/yanlış olsun, ölçüm yokluğu
    ödüllendirilmemeli.

    **TUR-51 NOTU — kip neden ELLE `band`a sabitleniyor.** Varsayılan `off`a çekildi
    (bant hiçbir yönde bilgi taşımıyor; EDGE_FINDINGS §28) ve `off` altında bonus HİÇ
    verilmediği için 0.5'lik fark gözlenemez hâle gelir — test sessizce `0.0 == 0.0`
    doğrular ve **Tur-31'in güvencesini artık ÖLÇMEZ**. Güvence gevşetilmedi, yeniden
    ÇAPALANDI: iddia "ölçüm yokluğu ödüllendirilmez"dir ve ancak bonusun AÇIK olduğu
    kipte sınanabilir. (Tur-44: bir bayrak iki soruyu birden yanıtlıyorsa biri sessizce
    yanlıştır — burada "kip kapalı" ile "yokluk ödüllendirilmiyor" ayrı tutuluyor.)"""
    from nihai.config import CONFIG
    from nihai.signals.strategy import SignalAction, signal_quality_score

    olcumsuz = get_iv_rank("ZZZ", 0.3)                    # NO_HISTORY → 50.0, measured=False
    df = pd.DataFrame({"close": [100.0 + i for i in range(30)],
                       "high": [101.0 + i for i in range(30)],
                       "low": [99.0 + i for i in range(30)]})
    kw = dict(htf_trend_aligned=False, chain_health_ok=False, confluence=1)
    onceki = CONFIG.strategy.iv_regime_bonus_mode
    try:
        object.__setattr__(CONFIG.strategy, "iv_regime_bonus_mode", "band")
        ham, _ = signal_quality_score(df, SignalAction.LONG,
                                      iv_rank_val=olcumsuz.iv_rank, **kw)   # eski davranış
        yeni, bd = signal_quality_score(
            df, SignalAction.LONG,
            iv_rank_val=(olcumsuz.iv_rank if olcumsuz.measured else None), **kw)
    finally:
        object.__setattr__(CONFIG.strategy, "iv_regime_bonus_mode", onceki)
    assert ham - yeni == pytest.approx(0.5)
    assert "iv_regime" not in bd


# ----------------------------------------------------- Tur-31: IV bonus rejimi
def test_iv_regime_bonus_kipleri_ve_VARSAYILAN_bit_denk():
    """Ölçüm bandın ters olduğunu söylüyor; bayrak bunu İFADE EDEBİLMELİ ama
    VARSAYILAN yürürlükteki davranışla BİT-DENK kalmalı (ACI DERS #3: ölçülen
    IV−RV vekilidir, P&L değil → canlı skorlama kanıtsız çevrilmez)."""
    from nihai.config import CONFIG
    from nihai.signals.strategy import SignalAction, signal_quality_score

    df = pd.DataFrame({"close": [100.0 + i for i in range(30)],
                       "high": [101.0 + i for i in range(30)],
                       "low": [99.0 + i for i in range(30)]})
    kw = dict(htf_trend_aligned=False, chain_health_ok=False, confluence=1)

    def _bonus(rank: float) -> float:
        return signal_quality_score(df, SignalAction.LONG, iv_rank_val=rank,
                                    **kw)[1].get("iv_regime", 0.0)

    # TUR-51: varsayılan "band" → "off". Bant hiçbir yönde bilgi TAŞIMIYOR (gün-bazlı
    # t=−0,17, n=20.910 / 581 PIT sembol; EDGE_FINDINGS §28) → kanıtsız bileşen skordan
    # çıkarıldı. Bu bir "ters kenar" iddiası DEĞİL, iddiasızlıktır.
    assert CONFIG.strategy.iv_regime_bonus_mode == "off"
    assert (_bonus(5.0), _bonus(40.0), _bonus(95.0)) == (0.0, 0.0, 0.0)

    # Üç kipin DAVRANIŞI aynen kilitli kalır (varsayılan değişti, mekanizma değil).
    # CONFIG dondurulmuş dataclass — test_config_validation.py'nin deseni
    onceki = CONFIG.strategy.iv_regime_bonus_mode
    try:
        object.__setattr__(CONFIG.strategy, "iv_regime_bonus_mode", "band")
        assert (_bonus(5.0), _bonus(40.0), _bonus(95.0)) == (0.0, 0.5, 0.0)
        object.__setattr__(CONFIG.strategy, "iv_regime_bonus_mode", "low")
        assert (_bonus(5.0), _bonus(40.0), _bonus(95.0)) == (0.5, 0.0, 0.0)
        object.__setattr__(CONFIG.strategy, "iv_regime_bonus_mode", "off")
        assert (_bonus(5.0), _bonus(40.0), _bonus(95.0)) == (0.0, 0.0, 0.0)
    finally:
        object.__setattr__(CONFIG.strategy, "iv_regime_bonus_mode", onceki)
