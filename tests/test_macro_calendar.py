"""B1 — FRED makro risk-off kapısı testleri (saf eşik mantığı + devre-dışı snapshot).

Ağ İSTEĞİ YAPILMAZ: compute_macro_multiplier saf; get_macro_snapshot kapı kapalı (varsayılan)
olduğundan FRED'e gitmeden nötr döner."""
from nihai.config import CONFIG
from nihai.data.macro_calendar import compute_macro_multiplier, get_macro_snapshot


def test_risk_on_no_triggers():
    mult, label, reasons = compute_macro_multiplier(vix=15, t10y2y=0.5, hy_spread=3.0)
    assert mult == 1.0 and label == "RISK_ON" and reasons == []


def test_single_trigger_caution():
    # Sadece VIX yüksek → 1 tetik → temkin
    mult, label, reasons = compute_macro_multiplier(vix=35, t10y2y=0.5, hy_spread=3.0)
    assert label == "CAUTION"
    assert mult == CONFIG.macro.one_trigger_mult
    assert len(reasons) == 1


def test_multi_trigger_risk_off():
    # VIX yüksek + ters eğri + geniş HY spread → ≥2 tetik → risk-off
    mult, label, reasons = compute_macro_multiplier(vix=40, t10y2y=-0.3, hy_spread=8.0)
    assert label == "RISK_OFF"
    assert mult == CONFIG.macro.multi_trigger_mult
    assert len(reasons) == 3


def test_none_values_ignored():
    # Tüm göstergeler None → tetik yok → risk-on (çekim başarısızlığına dayanıklı)
    mult, label, reasons = compute_macro_multiplier(None, None, None)
    assert mult == 1.0 and label == "RISK_ON"


def test_threshold_boundary_inclusive():
    # vix == eşik (>=) tetik sayılır; t10y2y == sınır (<) tetik DEĞİL
    m1, _, r1 = compute_macro_multiplier(vix=CONFIG.macro.vix_high, t10y2y=1.0, hy_spread=1.0)
    assert len(r1) == 1
    m2, _, r2 = compute_macro_multiplier(vix=10, t10y2y=CONFIG.macro.yield_inverted_below, hy_spread=1.0)
    assert r2 == []


def test_snapshot_disabled_is_neutral_no_network():
    # Varsayılan MACRO_GATE kapalı → nötr, ağ isteği yok
    snap = get_macro_snapshot()
    assert snap.multiplier == 1.0
    assert snap.enabled is False
    assert "kapalı" in snap.reasons[0] or "yok" in snap.reasons[0]
