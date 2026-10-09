"""adaptive_calibration.py testleri — yürüyen (walk-forward) kalibrasyon + taban-oran kapısı.

Ağırlık merkezi DÖRT şey (modülün sessizce bozulabileceği yerler):
  1. **Nedensellik/sızıntısızlık** — bir bloğu puanlayan kalibratör YALNIZ o bloktan ÖNCEKİ
     örneklerden fit edilmiş olmalı (CLAUDE.md: look-ahead yasak).
  2. **Taban-oran kapısı ÇALIŞIYOR** — bilgisiz ama kötü-kalibre bir kaynak ham'ı yener ama
     SABİT tabanı yenemez → `kazandi=False` (ACI DERS #9 artık makinede).
  3. **Kapı ölü değil** — gerçekten bilgi taşıyan bir kaynak kapıyı GEÇEBİLMELİ. Hiç geçmeyen
     kapı işe yaramaz kapıdır.
  4. **Yeterlilik kapısı** — az veri → YETERSIZ, asla 'iyileşti' iddiası.

Yardımcılar `test_ai_calibration_loop.py`'den KOPYALANDI (import EDİLMEDİ — test modülleri
arası import kırılgan; projenin kendi "kopyalandı, import edilmedi" deseni).
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta

import numpy as np
import pytest

from nihai.ai import adaptive_calibration as ac
from nihai.ai import calibration_loop, learning

FIELDS = ["logged_at", "symbol", "interval", "bar_ts", "direction", "action", "confidence",
          "prob_cal", "pwin", "pwin_side", "price", "bar_close", "horizon_bars",
          "resolved_at", "fwd_price", "hit", "fwd_ret_pct", "hit_bar", "slip_bps",
          "neural_pwin", "neural_pwin_side", "neural_pwin_std", "dir_bar", "conf_bar"]

_BASE = datetime(2026, 1, 1, 0, 0, 0)


@pytest.fixture(autouse=True)
def _tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_loop, "PRED_LOG", tmp_path / "predictions_log.csv")
    monkeypatch.setattr(ac, "ARTIFACT_FILE", tmp_path / "ai_adaptive_calibration.json")
    return tmp_path


def _row(i: int, *, pwin: str, fwd: str) -> dict:
    bar_ts = (_BASE + timedelta(minutes=i)).isoformat()
    return {"logged_at": bar_ts, "symbol": "TST", "interval": "1m", "bar_ts": bar_ts,
            "direction": "LONG", "action": "LONG", "pwin": pwin, "pwin_side": "LONG",
            "fwd_ret_pct": fwd,
            "resolved_at": (_BASE + timedelta(minutes=i, hours=2)).isoformat()}


def _write_log(rows: list[dict]) -> None:
    with calibration_loop.PRED_LOG.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: "" for k in FIELDS}, **r})


def _bilgisiz_rows(n: int = 2000, seed: int = 7, taban: float = 0.47) -> list[dict]:
    """Ham p geniş yayılır AMA sonuçla İLİŞKİSİZDİR (y ~ Bernoulli(taban), p'den bağımsız).
    Canlı `pwin`'in ölçülen durumu tam olarak budur (ACI DERS #2/#4/#9). Kalibrasyon burada
    yalnız taban orana BÜZÜLEBİLİR — tabanı YENEMEZ."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = float(rng.uniform(0.25, 0.75))
        y = 1 if rng.random() < taban else 0
        rows.append(_row(i, pwin=f"{p:.6f}", fwd=("1.0" if y == 1 else "-1.0")))
    return rows


def _bilgili_rows(n: int = 2000, seed: int = 42, overconf: float = 1.8) -> list[dict]:
    """GERÇEK bilgi taşıyan ama sistematik FAZLA-GÜVENLİ ham olasılıklar: gerçek q
    0.30–0.70 arasında değişir, raporlanan p ondan daha uçtur. Kalibrasyon uç-güveni
    düzeltir ve AYIRT ETME korunduğu için sabit tabanı da yenmelidir."""
    rng = np.random.default_rng(seed)
    q_levels = [0.30, 0.40, 0.50, 0.60, 0.70]
    rows = []
    for i in range(n):
        q = q_levels[i % len(q_levels)]
        p = float(min(max(0.5 + overconf * (q - 0.5), 0.02), 0.98))
        y = 1 if rng.random() < q else 0
        rows.append(_row(i, pwin=f"{p:.6f}", fwd=("1.0" if y == 1 else "-1.0")))
    return rows


# ---------------------------------------------------------------------------
# 1 — nedensellik: hiçbir fit kendi puanladığı (veya sonraki) örneği GÖRMEZ
# ---------------------------------------------------------------------------
def test_fit_asla_gelecegi_gormez(monkeypatch):
    _write_log(_bilgisiz_rows(1200))
    triples = calibration_loop._load_pairs("pwin", 60_000)
    triples.sort(key=lambda t: t[0])
    beklenen = [(p, y) for _, p, y in triples]

    gorulen: list[list[tuple[float, int]]] = []
    gercek_fit = learning.fit_calibrator
    monkeypatch.setattr(learning, "fit_calibrator",
                        lambda pairs, method="platt": (gorulen.append(list(pairs))
                                                       or gercek_fit(pairs, method=method)))

    ac.prequential_eval("pwin", "platt", warmup=200, refit_every=100, min_eval=100)

    # Son çağrı `calibrator` alanı için TÜM veriden fit'tir (walk-forward puanına dahil DEĞİL).
    yuruyen = gorulen[:-1]
    assert yuruyen, "en az bir yürüyen fit olmalı"
    for k, pairs in enumerate(yuruyen):
        beklenen_uzunluk = 200 + k * 100
        assert len(pairs) == beklenen_uzunluk, f"{k}. fit yanlış boyda geçmiş gördü"
        # Gördüğü şey KESİNLİKLE kronolojik ÖN EK olmalı — tek bir gelecek örnek bile sızmamalı.
        assert pairs == beklenen[:beklenen_uzunluk]


# ---------------------------------------------------------------------------
# 2 — ACI DERS #9 kapıda: bilgisiz kaynak ham'ı yener, TABANI yenemez
# ---------------------------------------------------------------------------
def test_bilgisiz_kaynak_tabani_yenemez_ve_kapiyi_gecemez():
    _write_log(_bilgisiz_rows(2000))
    r = ac.prequential_eval("pwin", "platt", warmup=500, refit_every=100, min_eval=100)
    assert r["durum"] == "OLCULDU"
    assert r["ham_yener"] is True, "kötü-kalibre ham p kalibrasyonla düzelmeliydi"
    assert r["taban_yener"] is False, "bilgisiz kaynak SABİT tabanı yenemez"
    assert r["kazandi"] is False
    # Büzülme izi: kalibre olasılıkların yayılımı çökmeli (ACI DERS #9 teşhisi).
    assert r["adaptif"]["p_std"] < r["ham"]["p_std"] / 2
    assert "ACI DERS #9" in r["verdict"]


# ---------------------------------------------------------------------------
# 3 — kapı ÖLÜ DEĞİL: gerçekten bilgili kaynak geçebilmeli
# ---------------------------------------------------------------------------
def test_bilgili_kaynak_kapiyi_gecer():
    _write_log(_bilgili_rows(2000))
    r = ac.prequential_eval("pwin", "platt", warmup=500, refit_every=100, min_eval=100)
    assert r["durum"] == "OLCULDU"
    assert r["taban_yener"] is True, "ayırt etme korunduğunda sabit taban YENİLEBİLMELİ"
    assert r["kazandi"] is True
    assert r["adaptif"]["brier"] < r["ham"]["brier"]


# ---------------------------------------------------------------------------
# 4 — yeterlilik + kalıcılık + geçiş-şeffaflık
# ---------------------------------------------------------------------------
def test_yetersiz_orneklem_verdikt_vermez():
    _write_log(_bilgisiz_rows(120))
    r = ac.prequential_eval("pwin", "platt", warmup=100, refit_every=50)
    assert r["durum"] == "YETERSIZ"
    assert "calibrator" not in r
    assert "iddia" in r["verdict"]


def test_kapiyi_gecmeyen_kaydedilmez_ve_calibrate_gecis_seffaf():
    _write_log(_bilgisiz_rows(2000))
    art = ac.refresh(sources=("pwin",), methods=("platt",), warmup=500, refit_every=100,
                     min_eval=100)
    assert art["pwin"]["durum"] == "KAPI_GECILEMEDI"
    assert "calibrator" not in art["pwin"]
    assert ac.calibrate("pwin", 0.73) == 0.73          # kayıt yok → p AYNEN


def test_kapiyi_gecen_kaydedilir_ve_calibrate_uygular():
    _write_log(_bilgili_rows(2000))
    art = ac.refresh(sources=("pwin",), methods=("platt",), warmup=500, refit_every=100,
                     min_eval=100)
    assert art["pwin"]["durum"] == "KAYITLI"
    assert ac.calibrate("pwin", 0.86) != 0.86          # kayıtlı kalibratör uygulanır
    assert "pwin: KAYITLI" in ac.report()


def test_gecersiz_parametreler_reddedilir():
    _write_log(_bilgisiz_rows(600))
    with pytest.raises(ValueError):
        ac.prequential_eval("pwin", "platt", refit_every=0)
    with pytest.raises(ValueError):
        ac.prequential_eval("pwin", "platt", window=1)
