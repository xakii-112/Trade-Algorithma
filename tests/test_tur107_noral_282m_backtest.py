"""Tur-107 — 282M sinir ağının 10 yıllık backtest düzeneği (`research/noral_282m_backtest.py`).

Üç soru: ölçü DOĞRU mu (nedensel taban, nedensel kapı, bilgiliyi bilgisizden ayırıyor mu), canlı yolla
AYNI mı (kasa kuralı, besleme kipleri), sessiz boşluk var mı (strateji verilmeyen kurucu, eksik yıl).
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import research.noral_282m_backtest as M


def _sentetik(n: int = 12000, gun: int = 80, seed: int = 0) -> pd.DataFrame:
    """Bilgili (`p_iyi` = gerçek kazanma olasılığı) ve bilgisiz (`p_sabit`, `p_gurultu`) tahminciler."""
    rng = np.random.default_rng(seed)
    q = 0.45 + 0.2 * rng.choice([-1.0, 1.0], n)
    y = (rng.random(n) < q).astype(int)
    zaman = (pd.Timestamp("2025-01-06 14:30") + pd.to_timedelta(np.sort(rng.integers(0, gun, n)), unit="D")
             + pd.to_timedelta(rng.integers(0, 380, n), unit="m"))
    return M.hazirla(pd.DataFrame({
        "entry_time": zaman.astype(str), "symbol": rng.choice(list("ABCD"), n), "label": y,
        "pnl": (2 * y - 1) * 0.1, "p_iyi": q, "p_sabit": 0.47, "p_gurultu": rng.normal(0.486, 0.001, n)}))


def test_kasa_kurali_canli_kasayla_ayni_ve_nedensel():
    p = pd.Series(np.random.default_rng(1).normal(0.486, 0.001, 500))
    t = M.kasa_kurali(p)
    assert (t[:30] == -1).all() and (t[30:] >= 0).all()                     # 30 ölçüm ısınma
    for j in range(30, 500):                                                # eşik = ÖNCEKİ ≤200 ölçümün medyanı
        assert t[j] == int(p.iloc[j] >= p.iloc[max(0, j - 200):j].median()), j
    assert (M.kasa_kurali(pd.Series([0.48554] * 100))[30:] == 1).all()      # eşitlik GEÇER (canlı: p < eşik → veto)
    p2 = p.copy()
    p2.iloc[300:] += 1.0                                                    # gelecek değişir, geçmiş karar değişmez
    assert (M.kasa_kurali(p2)[:300] == t[:300]).all()
    p3 = p.copy()
    p3.iloc[250] = 0.0                                                      # aday kendi eşiğine girmez
    assert M.kasa_kurali(p3)[250] == 0 and (M.kasa_kurali(p3)[:250] == t[:250]).all()


def test_hazirla_taban_yalniz_gecmisten_ve_donemler():
    h = M.hazirla(pd.DataFrame({
        "entry_time": ["2026-07-02 14:00:00", "2026-07-02 15:00:00+00:00", "2026-07-06 14:00:00",
                       "2026-07-07 14:00:00", "2026-09-23 19:00:00", "2026-09-24 14:00:00"],
        "symbol": list("ABABAB"), "label": [1, 0, 1, 1, 0, 1]}))
    assert list(h["donem"]) == ["oncesi", "oncesi", "egitim", "egitim", "egitim", "ileri"]
    assert list(h["taban"]) == pytest.approx([0.5, 0.5, 0.5, 2 / 3, 3 / 4, 3 / 5])   # aynı günün etiketi girmez


def test_olc_bilgiliyi_bilgisizden_ayirir():
    h = _sentetik()
    iyi, sabit = M.olc(h, "p_iyi"), M.olc(h, "p_sabit")
    assert iyi["bss"] > 0.1 and iyi["t_gun"] < -1.645 and iyi["auc"] > 0.65 and iyi["auc_sembol_ici"] > 0.65
    assert sabit["auc"] == pytest.approx(0.5) and sabit["p_std"] < 1e-12 and not sabit["t_gun"] < -1.645
    assert M.olc(h.iloc[:50], "p_iyi")["hukum"] == "YETERSIZ"


def test_kapi_bilgiliyi_gecirir_gurultuyu_gecirmez():
    h, rng = _sentetik(), np.random.default_rng(7)
    iyi = M.kapi_ozet(h, M.kasa_kurali(h["p_iyi"]), "pnl", rng)
    assert iyi["fazlalik"] > 1.0 and iyi["t_gun"] > 1.645 and iyi["rastgele_yuzdelik"] == 100.0
    assert iyi["tutulan"] > iyi["taban"] > iyi["elenen"]
    gurultu = M.kapi_ozet(h, M.kasa_kurali(h["p_gurultu"]), "pnl", rng)
    assert abs(gurultu["t_gun"]) < 2.5 and abs(gurultu["fazlalik"]) < 0.5
    assert M.kapi_ozet(h, np.full(len(h), -1), "pnl", rng)["hukum"] == "YETERSIZ"


def test_hukum_uc_kosulun_hepsini_ister():
    iyi = {"t_gun": -3.0, "p_std": 0.01, "kapi": {"t_gun": 3.0}, "kapi_g1": {"fazlalik": 0.1}}
    assert M._hukum(iyi)["hukum"] == "BECERI_VAR"
    for bozuk in ({"t_gun": -1.0}, {"t_gun": None}, {"kapi": {"t_gun": 1.0}}, {"kapi_g1": {"fazlalik": -0.1}}, {"kapi_g1": {}}):
        assert M._hukum({**iyi, **bozuk})["hukum"] == "BECERI_YOK"
    assert M._hukum({**iyi, "p_std": 1e-5})["cikti_cokmus"] and not M._hukum(iyi)["cikti_cokmus"]


class _SahteModel:
    """`a`: μ≠0 · `b`: σ≠1 → nötr DEĞİL; yalnız μ=0 VE σ=1 olan iki kolon eğitimde nötrlenmiştir."""
    feature_names = ["a", "rr_ratio", "confluence", "b"]
    _feat_mu = np.array([0.3, 0.0, 0.0, 0.0])
    _feat_sigma = np.array([1.0, 1.0, 1.0, 2.0])

    def ood_zmax(self, X):
        return np.abs((X - self._feat_mu) / self._feat_sigma).max(axis=1)

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-X.sum(axis=1)))


def test_besleme_kipleri_canli_ham_egitim_notr():
    df = pd.DataFrame({"a": [0.1, 0.2], "rr_ratio": [2.0, 2.0], "confluence": [1.0, 1.0], "b": [0.5, -0.5],
                       "symbol": ["X", "X"], "entry_time": ["2025-01-02 15:00:00"] * 2, "label": [1, 0], "pnl": [0.1, -0.1]})
    m = _SahteModel()
    assert M.notr_kolonlar(m) == [1, 2]
    out = M.tahmin_kolonlari(df, {"x": m})
    sig = lambda v: 1.0 / (1.0 + np.exp(-np.asarray(v)))  # noqa: E731
    assert out["p_x_canli"].to_numpy() == pytest.approx(sig([3.6, 2.7]))      # canlı yol ham besler (z=2,0 ve 1,0)
    assert out["p_x_egitim"].to_numpy() == pytest.approx(sig([0.6, -0.3]))    # eğitim: iki kolon 0
    assert out["z_x"].to_numpy() == pytest.approx([2.0, 2.0])                 # OOD ölçüsü canlıdaki gibi HAM vektörde
    assert "pnl_g1" not in out.columns and list(out["label"]) == [1, 0]


def test_govde_katkisi_gammayi_gecici_sifirlar_ve_geri_yazar():
    """Teşhis kayıtlı modeli DEĞİŞTİRMEZ: γ ölçüm süresince 0, sonra eski değerinde."""
    torch = pytest.importorskip("torch")

    class Ag(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.ls = torch.nn.Module()
            self.ls.gamma = torch.nn.Parameter(torch.full((2,), 0.5))
            self.head = torch.nn.Linear(2, 1)

    class Model:
        def __init__(self, ag):
            self._model = ag

        def param_count(self):
            return sum(p.numel() for p in self._model.parameters())

        def predict_proba(self, X):
            n = self._model
            with torch.no_grad():
                x = torch.as_tensor(X)
                govde = n.ls.gamma * n.blocks[0](x) if hasattr(n, "ls") else 0.0
                return torch.sigmoid(n.head(x + govde)).squeeze(1).numpy()

    torch.manual_seed(0)
    m = Model(Ag())
    X = np.random.default_rng(0).normal(size=(50, 2)).astype(np.float32)
    once = m.predict_proba(X)
    g = M.govde_katkisi(m, X)
    assert g["gamma_ort_mutlak"] == pytest.approx(0.5) and g["blok_parametre_payi"] == pytest.approx(6 / 11)
    assert g["govdesiz_maks_fark"] > 1e-3                                      # gövde açıkken katkısı ÖLÇÜLÜR
    assert torch.equal(m._model.ls.gamma.detach(), torch.full((2,), 0.5))      # γ geri yazıldı
    assert np.array_equal(m.predict_proba(X), once)
    assert M.govde_katkisi(Model(torch.nn.Linear(2, 1)), X) is None            # LayerScale'siz basamak


def test_satir_kurucu_stratejiyi_acikca_alir(monkeypatch):
    """Kusur: canlı strateji `quant_trend`e geçince kurucu sessizce 0 satır üretiyordu (strateji varsayılandan)."""
    import nihai.signals.strategy as S
    import research.build_hist_1m_dataset as B

    gorulen: list = []

    def sahte(sym, w, **kw):
        gorulen.append(kw.get("strategy_mode", "VERILMEDI"))
        return S.Signal(S.SignalAction.WAIT, sym, 0, 0, 0)

    monkeypatch.setattr(S, "generate_signal", sahte)
    ix = pd.date_range("2025-01-02 14:30", periods=232, freq="min")
    df = pd.DataFrame({"open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 1000.0}, index=ix)
    B.sembol_satirlari("X", df, adim=5, strateji="event_breakout")
    n = len(gorulen)
    B.sembol_satirlari("X", df, adim=5)
    assert n > 0 and set(gorulen[:n]) == {"event_breakout"} and set(gorulen[n:]) == {None}   # None = eski davranış
    assert M.ADAY_STRATEJISI == "event_breakout" and "strateji=ADAY_STRATEJISI" in inspect.getsource(M._veri_isci)


def test_eksik_yil_sessizce_gecilmez(tmp_path, monkeypatch):
    """İndirici hatada BOŞ çerçeve döndürür; ortadaki boş yıl veri deliğidir, baştaki boş yıllar geç halka arzdır."""
    ix = pd.date_range("2020-01-02 14:30", periods=5, freq="min")
    dolu = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=ix)
    pencereler = M._yil_pencereleri()
    assert pencereler[0][0] == M.BASLANGIC and pencereler[-1][1] == M.BITIS
    assert all(a[1] == b[0] for a, b in zip(pencereler, pencereler[1:], strict=False))
    (tmp_path / "sip_rth").mkdir()
    for sym, bos in (("DELIK", {3}), ("GEC", {0, 1, 2})):
        for k, (a, _) in enumerate(pencereler):
            (pd.DataFrame() if k in bos else dolu).to_parquet(tmp_path / "sip_rth" / f"{sym}_{a[:4]}.parquet")
    sym, n, _, not_ = M._veri_isci(str(tmp_path), "DELIK", 10)
    assert n == -1 and "EKSİK YIL" in not_ and not (tmp_path / "satir" / "DELIK.parquet").exists()
    sym, n, _, not_ = M._veri_isci(str(tmp_path), "GEC", 10)
    assert n == 0 and not_ == "" and (tmp_path / "satir" / "GEC.parquet").exists()
    assert M._yil_sayimi(tmp_path, "YOK") is None                              # indirmesi bitmemiş sembol beklenir
    assert M.BILINEN_BOSLUK == {"NBIS": (2023,)}                               # doğrulanmış tek işlem durdurması
    monkeypatch.setattr(M, "BILINEN_BOSLUK", {"DELIK": (int(pencereler[3][0][:4]),)})
    assert M._veri_isci(str(tmp_path), "DELIK", 10)[1] == 0                    # beyan edilen boşluk delik sayılmaz


def test_gunluk_seri_maliyet_ve_bilinen_degerler():
    s = M.seri_ozet(np.array([10.0, -50.0, 10.0]))
    assert s["toplam_pct"] == pytest.approx((1.1 * 0.5 * 1.1 - 1) * 100) and s["mdd_pct"] == pytest.approx(-50.0)
    gun = pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-06"])
    d = pd.DataFrame({"gun": gun, "pnl": [0.3, 0.5, 0.3, 0.3]})
    spy = pd.Series([1.0, 1.0, 1.0], index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]))
    g = M.gunluk_seriler(d, np.array([1, 0, -1, 1]), spy)
    # taban = ısınma dışı tüm adaylar (gün ortalaması: 0,4 · 0 · 0,3); kapı = tutulanlar (0,3 · 0 · 0,3)
    assert g["taban_tum_adaylar_brut"]["toplam_pct"] == pytest.approx((1.004 * 1.0 * 1.003 - 1) * 100)
    assert g["kapi_brut"]["toplam_pct"] == pytest.approx((1.003 * 1.0 * 1.003 - 1) * 100)
    assert g["kapi_net"]["toplam_pct"] == pytest.approx((1.001 * 1.0 * 1.001 - 1) * 100)   # işlem başı %0,2
    assert g["spy_al_tut"]["toplam_pct"] == pytest.approx((1.01 ** 3 - 1) * 100)
    # sürekli kip: adaysız seans da sayılır (strateji 0, SPY kendi getirisi) — al-tut yalnız aday günlerinden kurulmaz
    spy2 = pd.Series([1.0, -20.0, 1.0, 1.0], index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]))
    d2 = d[d["gun"] != pd.Timestamp("2025-01-03")].reset_index(drop=True)
    s2 = M.gunluk_seriler(d2, np.array([1, 0, 1]), spy2, surekli=True)
    assert s2["spy_al_tut"]["gun"] == 3 and s2["spy_al_tut"]["mdd_pct"] == pytest.approx(-20.0)
    assert M.gunluk_seriler(d2, np.array([1, 0, 1]), spy2)["spy_al_tut"]["mdd_pct"] == pytest.approx(0.0)
