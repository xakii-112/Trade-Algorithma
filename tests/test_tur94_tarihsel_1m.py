"""Tur-94 — tarihsel 1m aday satırları canlı yolun eşi mi?

Sorular: pencere canlıyla aynı mı (son 220 bar, karar barında biter → look-ahead yok) · etiket aday tarafına
göre mi işaretli (SHORT çevrilir, beraberlik ıska) · `is_long` aday tarafı mı · örnekleme adımı · seans
süzgeci · çıktı adı kart aralığını taşıyor mu.
"""
from __future__ import annotations

import pandas as pd
import pytest

import nihai.signals.meta_features as mf
import nihai.signals.strategy as st
from nihai.signals.strategy import Signal, SignalAction
from research import build_hist_1m_dataset as b


def _barlar(n: int, adim_fiyat: float = 0.01) -> pd.DataFrame:
    # 2026-09-22 (Pazartesi) 13:30 UTC = 09:30 ET'den itibaren tek seans, dakikalık.
    idx = pd.date_range("2026-09-22 13:30", periods=n, freq="1min")
    c = pd.Series([100.0 + adim_fiyat * i for i in range(n)], index=idx)
    return pd.DataFrame({"open": c, "high": c + 0.05, "low": c - 0.05, "close": c, "volume": 1000.0})


@pytest.fixture()
def sahte(monkeypatch):
    goren: list[pd.DataFrame] = []
    taraf = {"v": SignalAction.LONG}

    def gs(sym, w, **kw):
        goren.append(w)
        return Signal(taraf["v"], sym, float(w["close"].iloc[-1]), 0.0, 0.0)

    monkeypatch.setattr(st, "generate_signal", gs)
    monkeypatch.setattr(mf, "extract_features", lambda w, sig, shared=None: {"rsi14": 55.0})
    return goren, taraf


def test_pencere_220_ve_karar_barinda_biter(sahte):
    goren, _ = sahte
    df = _barlar(240)
    rows = b.sembol_satirlari("X", df, adim=5)
    assert goren and all(len(w) == b.PENCERE for w in goren)
    for w, r in zip(goren, rows, strict=True):
        assert str(w.index[-1]) == r["entry_time"]          # pencere karar barını aşmaz


def test_etiket_tarafa_gore_ve_ufuk_5dk(sahte):
    _, taraf = sahte
    df = _barlar(240)                                           # fiyat her dakika +0,01
    long_ = b.sembol_satirlari("X", df, adim=5, capa="kapanis_i")
    assert long_ and all(r["label"] == 1 and r["is_long"] == 1.0 for r in long_)
    i = df.index.get_loc(pd.Timestamp(long_[0]["entry_time"]))
    beklenen = (df["close"].iloc[i + 5] / df["close"].iloc[i] - 1) * 100
    assert long_[0]["pnl"] == pytest.approx(beklenen)
    taraf["v"] = SignalAction.SHORT
    short = b.sembol_satirlari("X", df, adim=5, capa="kapanis_i")
    assert all(r["label"] == 0 and r["is_long"] == 0.0 and r["pnl"] < 0 for r in short)


def test_varsayilan_capa_sonraki_barin_ortasi(sahte):
    df = _barlar(240)
    # −0,02'de orta(i+1) = kapanış(i) olurdu (adım 0,01) → kurgu çapaları AYIRT ETMEZDİ (mutasyon kaçtı).
    df["open"] = df["close"] - 0.05
    r = b.sembol_satirlari("X", df, adim=5)[0]
    i = df.index.get_loc(pd.Timestamp(r["entry_time"]))
    c0 = (df["open"].iloc[i + 1] + df["close"].iloc[i + 1]) / 2
    assert r["pnl"] == pytest.approx((df["close"].iloc[i + 5] / c0 - 1) * 100)


def test_orta_capa_seans_sinirinda_satir_yok():
    df = pd.concat([_barlar(3), _barlar(3).set_axis(pd.date_range("2026-09-23 13:30", periods=3,
                                                                  freq="1min"))])
    from nihai.execution.prediction_tracker import _pos_map
    pm = _pos_map(df)
    assert b.taraf_getirisi(df, pm, 2, "LONG") is None                  # sonraki bar ertesi gün
    assert b.taraf_getirisi(df, pm, 2, "LONG", capa="kapanis_i") is not None
    with pytest.raises(ValueError):
        b.taraf_getirisi(df, pm, 0, "LONG", capa="kotasyon")


def test_yeniden_etiketle_dogrudan_kurulumla_ayni(sahte, tmp_path):
    df = _barlar(260)
    df["open"] = df["close"] - 0.02
    df.to_csv(tmp_path / "X_1m_split_iex.csv")
    _, taraf = sahte
    satir = b.sembol_satirlari("X", df, adim=5)
    taraf["v"] = SignalAction.SHORT                             # iki taraf da: işaret yeniden kurulmalı
    satir += b.sembol_satirlari("X", df, adim=5)
    zaman = [r["entry_time"] for r in satir]
    _, pnl = b._etiketle_isci("X", zaman, [r["is_long"] for r in satir], str(tmp_path), "orta_i1")
    assert pnl == pytest.approx([r["pnl"] for r in satir])


def test_beraberlik_iska(sahte):
    rows = b.sembol_satirlari("X", _barlar(240, adim_fiyat=0.0), adim=5)
    assert rows and all(r["label"] == 0 for r in rows)


def test_aday_yoksa_satir_yok_ve_adim(sahte):
    goren, taraf = sahte
    df = _barlar(260)
    b.sembol_satirlari("X", df, adim=7)
    sonlar = [df.index.get_loc(w.index[-1]) for w in goren]
    assert sonlar[0] == b.PENCERE - 1 and all(y - x == 7 for x, y in zip(sonlar, sonlar[1:], strict=False))
    taraf["v"] = SignalAction.WAIT
    assert b.sembol_satirlari("X", df, adim=5) == []


def test_rth_barlar_seans_disi_ve_tekrari_atar():
    idx = pd.to_datetime(["2026-09-22 12:00", "2026-09-22 13:30", "2026-09-22 13:30",
                          "2026-09-22 19:59", "2026-09-22 20:00"])
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    out = b.rth_barlar(df)
    assert list(out["close"]) == [3.0, 4.0]


def test_ogrenme_egrisi_egitim_canli_donemden_once_ve_test_c_canli_semboller(tmp_path):
    import json

    import numpy as np

    from nihai.signals.meta_features import FEATURE_NAMES
    from research import hist_learning_curve as lc
    rng = np.random.default_rng(0)

    def csv(ad, zaman, sembol):
        d = pd.DataFrame(rng.normal(size=(len(zaman), len(FEATURE_NAMES))), columns=FEATURE_NAMES)
        d = d.assign(label=rng.integers(0, 2, len(zaman)), symbol=sembol, entry_time=zaman.astype(str))
        d.to_csv(tmp_path / ad, index=False)
        return tmp_path / ad

    hz = pd.Series(pd.date_range("2026-06-01 14:00", "2026-07-20 19:00", freq="3h"))
    hist = csv("h_1m_.csv", hz, np.where(np.arange(len(hz)) % 2, "A", "B"))
    live = csv("c.csv", pd.Series(pd.date_range("2026-07-06 14:00", periods=40, freq="3h")), "A")
    out = tmp_path / "o.json"
    assert lc.main(["--hist", str(hist), "--live", str(live), "--points", "100,all", "--max-epochs", "1",
                    "--eval-hist-max", "20", "--json", str(out)]) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert pd.Timestamp(r["egitim_araligi"][1]) < pd.Timestamp(r["kesim_utc"]) <= pd.Timestamp("2026-07-05", tz="UTC")
    beklenen_c = int(((hz >= "2026-07-06") & (np.arange(len(hz)) % 2 == 1)).sum())
    assert [p["n"] for p in r["noktalar"]] == [100, r["egitim_havuzu"]]
    # n satır = kesime EN YAKIN son n satır (ilk n değil); tüm havuz ilk satırdan başlar
    assert pd.Timestamp(r["noktalar"][0]["egitim_ilk"]) > pd.Timestamp(r["egitim_araligi"][0])
    assert r["noktalar"][1]["egitim_ilk"] == r["egitim_araligi"][0]
    assert all(p["tarihsel_canli_sembol"]["n"] == beklenen_c and p["tarihsel"]["n"] == 20
               and p["canli"]["n"] == 40 for p in r["noktalar"])


def _kucuk_fit(monkeypatch, parca: int):
    import numpy as np

    from nihai.ai import neural
    monkeypatch.setattr(neural, "_INFER_BATCH", parca)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 6))
    y = (X[:, 0] + rng.normal(scale=2.0, size=400) > 0).astype(float)
    m = neural.NeuralCapacity("deep_mlp", feature_names=[f"f{i}" for i in range(6)], seed=7)
    m.fit(X, y, max_epochs=6, patience=2)
    return m, X


def test_dogrulama_parcali_gecer_ve_tek_gecisle_ayni_sonuc(monkeypatch):
    """BULGU-94-2: val tek geçişte 282M'de VRAM'i taşırdı. Eval modunda hiçbir ileri geçiş
    parçadan büyük olmamalı; parçalı kayıp aynı erken-durma kararını vermeli."""
    import numpy as np
    import torch
    goren: list[int] = []

    def kanca(mod, args):
        if not mod.training and args and hasattr(args[0], "shape"):
            goren.append(int(args[0].shape[0]))

    h = torch.nn.modules.module.register_module_forward_pre_hook(kanca)
    try:
        parcali, X = _kucuk_fit(monkeypatch, 7)
    finally:
        h.remove()
    assert goren and max(goren) <= 7                    # val = 60 satır → 9 parça
    tek, _ = _kucuk_fit(monkeypatch, 4096)
    np.testing.assert_allclose(parcali.predict_proba(X), tek.predict_proba(X), atol=1e-6)
    assert parcali._fit_info["n_epochs_trained"] == tek._fit_info["n_epochs_trained"]


def test_cikti_adi_1m_tasimali(tmp_path):
    assert b.main(["--hist-dir", str(tmp_path), "--out", str(tmp_path / "x.csv"),
                   "--symbols", "AAPL"]) == 1
