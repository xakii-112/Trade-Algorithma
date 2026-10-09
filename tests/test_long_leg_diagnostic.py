"""long_leg_diagnostic (F4) testleri — 2026-07-27.

Ağ YOK: normalizasyon denetimi geçici CSV'den, eşleştirilmiş test sentetik
çerçevelerden okur. Testler bir KENAR iddiasını değil, teşhisin bilinen girdide
doğru hipotezi seçtiğini doğrular."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import long_leg_diagnostic as lld


def _log_yaz(path, satirlar: list[dict]) -> str:
    """predictions_log.csv şemasının teşhis için gereken alt kümesi."""
    pd.DataFrame(satirlar).to_csv(path, index=False)
    return str(path)


def _satir(direction: str, price: float, fwd: float, hit: float | None = None) -> dict:
    ham = fwd / price - 1.0
    sgn = 1.0 if direction == "+" else -1.0
    return {"symbol": "X", "bar_ts": "2026-06-23 15:59:00-04:00", "direction": direction,
            "confidence": 0.5, "price": price, "fwd_price": fwd,
            "hit": (1.0 if ham * sgn > 0 else 0.0) if hit is None else hit,
            "fwd_ret_pct": ham * sgn * 100}


def test_normalizasyon_tutarli_logda_bug_bulmaz(tmp_path):
    p = _log_yaz(tmp_path / "log.csv",
                 [_satir("+", 100.0, 101.0), _satir("+", 100.0, 99.0),
                  _satir("-", 100.0, 99.0), _satir("-", 100.0, 101.0)])
    r = lld.normalizasyon_denetimi(p)
    assert r["durum"] == "OLCULDU" and r["H1_BUG"] is False
    assert r["taraf"]["LONG"]["hit_yeniden_uretim_orani"] == 1.0
    assert r["fwd_ret_pct_isaretli_mi"]["sozlesme"].startswith("işaretli")


def test_normalizasyon_yalniz_LONG_ters_ise_bug_der(tmp_path):
    """Tur-15 dersinin regresyon testi: yön normalizasyonu TEK tarafta bozulursa yakalanmalı."""
    satirlar = [_satir("+", 100.0, 101.0, hit=0.0),      # LONG kazandı ama 0 yazılmış
                _satir("+", 100.0, 99.0, hit=1.0),
                _satir("-", 100.0, 99.0), _satir("-", 100.0, 101.0)]
    r = lld.normalizasyon_denetimi(_log_yaz(tmp_path / "log.csv", satirlar))
    assert r["H1_BUG"] is True
    assert r["taraf"]["LONG"]["hit_yeniden_uretim_orani"] == 0.0
    assert r["taraf"]["SHORT"]["hit_yeniden_uretim_orani"] == 1.0
    assert r["taraf_ayrismasi"] == 1.0


def _taraf(gunler: list[str], mu: list[float], n: list[int], sapma: float = 0.0) -> pd.DataFrame:
    """Sembol×gün kümelerine dağılmış işaretli getiriler (deterministik)."""
    rows = []
    for i, (g, m, k) in enumerate(zip(gunler, mu, n, strict=True)):
        kayma = sapma * (1.0 if i % 2 == 0 else -1.0)     # gün farkları ± → ortalaması 0
        for j in range(k):
            rows.append({"symbol": "X", "ts": pd.Timestamp(g) + pd.Timedelta(minutes=j),
                         "signed": m + kayma})
    return pd.DataFrame(rows)


def test_eslestirilmis_fark_gun_kompozisyonunu_dusurur():
    """Ham ortalamalar taraf kompozisyonu yüzünden çok farklı, gün-içi fark ise SIFIR.

    H3'ün (rejim) çekirdek iddiası budur: LONG çoğunlukla kötü günlere, SHORT iyi günlere
    düşmüşse ham karşılaştırma elmayla armuttur."""
    gunler = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-08"]
    mu = [-0.005, -0.005, -0.005, 0.005, 0.005, 0.005]
    a = _taraf(gunler, mu, [200, 200, 200, 5, 5, 5], sapma=1e-4)      # LONG: kötü günlerde yoğun
    b = _taraf(gunler, mu, [5, 5, 5, 200, 200, 200])                  # SHORT: iyi günlerde yoğun

    ham_fark = float(a["signed"].mean() - b["signed"].mean())
    assert ham_fark < -0.008                       # ham karşılaştırma DEVASA bir açık gösterir

    r = lld.eslestirilmis_fark(a, b)
    assert r["n_kume"] == 6 and r["kume_kapsami"] == 1.0
    assert abs(r["ort_fark_pct"]) < 1e-9           # gün eşleştirilince fark yok olur
    assert abs(r["t"]) < 1e-6


def test_eslestirilmis_fark_gercek_farki_korur():
    """Gün etkisi düşülünce GERÇEK bir taraf farkı hâlâ görünmeli (test kör değil)."""
    gunler = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"]
    mu = [-0.004, 0.006, -0.002, 0.003]
    a = _taraf(gunler, [m - 0.001 for m in mu], [50] * 4, sapma=1e-5)  # LONG her gün 10 bps geride
    b = _taraf(gunler, mu, [50] * 4)
    r = lld.eslestirilmis_fark(a, b)
    assert r["ort_fark_pct"] < -0.09 and r["t"] is not None and r["t"] < -20


def test_eslestirilmis_fark_ortak_kume_yoksa_hukum_vermez():
    a = _taraf(["2026-06-01", "2026-06-02"], [0.001, 0.001], [10, 10])
    b = _taraf(["2026-06-09", "2026-06-10"], [0.001, 0.001], [10, 10])
    assert lld.eslestirilmis_fark(a, b)["n_kume"] == 0


def test_gunluk_piyasa_barlardan_hesaplanir():
    """Rejim vekili gün-kapanışı getirisidir (tahminlerin mikro-getirileri DEĞİL)."""
    idx = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f"2026-06-0{g} 13:30:00", periods=390, freq="1min").to_numpy()
        for g in (1, 2, 3)]))
    close = np.concatenate([np.full(390, 100.0), np.full(390, 102.0), np.full(390, 101.0)])
    bars = {"X": pd.DataFrame({"open": close, "high": close, "low": close,
                               "close": close, "volume": 1.0}, index=idx)}
    s = lld.gunluk_piyasa(bars)
    assert len(s) == 3
    assert np.isnan(s.iloc[0])                       # ilk günün öncesi yok
    assert s.iloc[1] > 0 and s.iloc[2] < 0


_NORM_TEMIZ = {"H1_BUG": False, "taraf_ayrismasi": 0.0001}


def _acik(t_canli: float, t_bar: float, ham_canli: float, ham_bar: float,
          n_kume: int = 400) -> dict:
    return {
        "canli_lag0": {"ham_isabet_acigi_SHORT_eksi_LONG": ham_canli,
                       "gun_eslestirilmis_getiri_farki_LONG_eksi_SHORT":
                           {"n_kume": n_kume, "t": t_canli}},
        "bar_lag0": {"ham_isabet_acigi_SHORT_eksi_LONG": ham_bar,
                     "gun_eslestirilmis_getiri_farki_LONG_eksi_SHORT":
                         {"n_kume": n_kume, "t": t_bar}},
    }


def test_verdikt_bug_varsa_once_onu_soyler():
    v, _ = lld._verdikt({"H1_BUG": True, "taraf_ayrismasi": 0.06},
                        _acik(-3.0, -3.0, 0.05, 0.05), {}, "canli_lag0", "bar_lag0")
    assert v.startswith("H1 BUG")


def test_verdikt_canlida_var_barda_yoksa_kayma_der():
    v, gerekce = lld._verdikt(_NORM_TEMIZ, _acik(-2.8, -1.2, 0.0129, 0.0002),
                              {"LONG": {"ort_bps": 2.694}, "SHORT": {"ort_bps": 1.069}},
                              "canli_lag0", "bar_lag0")
    assert v.startswith("H2 KAYMA")
    assert any("H1 (bug) ELENDİ" in g for g in gerekce)
    assert any("1.625" in g for g in gerekce)        # taraflar arası kayma farkı raporlanır


def test_verdikt_eslestirilmiste_fark_yoksa_rejim_der():
    v, _ = lld._verdikt(_NORM_TEMIZ, _acik(-0.4, -0.3, 0.02, 0.02), {}, "canli_lag0", "bar_lag0")
    assert v.startswith("H3 REJİM")


def test_verdikt_iki_giriste_de_duruyorsa_yapisal_der():
    v, _ = lld._verdikt(_NORM_TEMIZ, _acik(-4.0, -3.5, 0.05, 0.045), {}, "canli_lag0", "bar_lag0")
    assert v.startswith("H4 YAPISAL")


def test_verdikt_kume_yetersizse_hukum_vermez():
    v, _ = lld._verdikt(_NORM_TEMIZ, _acik(-4.0, -3.5, 0.05, 0.045, n_kume=5), {},
                        "canli_lag0", "bar_lag0")
    assert v.startswith("HÜKÜM YOK")
