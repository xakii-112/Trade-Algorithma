"""quant_ml.py — Tur-97: makine öğrenimini quant çekirdeğin HİZMETİNE sokmak (ölçüm düzeneği).

Tur-96 iki şeyi ölçtü: (1) ertesi gün YÖNÜ tahmin eden ML'in ekonomik değeri yok; (2) OYNAKLIK
öngörüsünde gerçek, piyasalar arası tutarlı beceri var. Tur-97 karar yolunu kural tabanlı TA'dan
`quant_trend`e (nihai/signals/quant.py) kaydırdı. Bu modül ML'i o çekirdeğin iki girdisine yarıştırır:

  A) TREND DURUMU — kural: 21/63/252 günlük getiri işaretlerinin ortalaması > 0 (canlı quant_trend).
     ML: oynaklıkla normalize çok ufuklu momentum özelliklerinden İLERİ 21 GÜNLÜK normalize getiriyi
     tahmin eden havuzlanmış model (Lim-Zohren-Roberts 2019 yaklaşımının sade hâli).
  B) OYNAKLIK — kural: EWMA(0,94) (canlı quant_trend'in σ'sı). ML: log-RV özelliklerinden ertesi gün
     log r² regresyonu + eğitimde kestirilen yanlılık düzeltmesi. Kıyas: EWMA, Tur-96'nın HAR ve GARCH
     öngörüleri (AYNI hedef, AYNI günler), kayıp QLIKE = ln f + r²/f.

ÖN-KAYIT (2026-09-26, sonuçlara bakılmadan sabitlendi):
  · Evren: Tur-96'nın 14 piyasa grubu (kusur kuralından geçen 121 enstrüman), HAVUZLANMIŞ panel.
  · Yürüyen eğitim: her test yılı Y için eğitim = etiketi Y'nin ilk gününden ÖNCE bilinen satırlar
    (A: t+22 < Y başı; B: t+1 < Y başı). Model yıllık yeniden eğitilir.
  · A özellikleri (hepsi nedensel, ölçeksiz): m_h = ln(C_t/C_{t−h})/(σ_t√h), h ∈ {5,21,63,126,252};
    σ_t/σ̄_t (σ̄ genişleyen ortalama); σ5/σ63 (kısa/uzun gerçekleşen oynaklık); r_t/σ_t;
    ln(C_t / max_252)/σ_t (52 hafta zirvesine uzaklık). Hedef y = ln(O_{t+22}/O_{t+1})/(σ_t√21), [−3,3].
    Modeller: lojistik (sign y) ve HistGradientBoosting regresyonu (y). Karar: uzun iff p>0,5 / ŷ>0.
  · A değerlendirmesi (vektörel, kesif97 ile aynı): karar t kapanışı, getiri open[t+1]→open[t+2],
    ağırlık = 1{uzun} × min(σ̄/σ, 2), maliyet %0,1 × |Δw|; piyasa = eşit ağırlıklı enstrüman ortalaması.
    Birincil ölçü: piyasa başına günlük FARK (ML − kural) ortalamasının NW t'si. Çoklu test: 2 model ×
    14 piyasa = 28 hücre → Bonferroni |t| ≥ 3,20. Ek: enstrüman-içi AUC (sign y) — kural skoru dahil.
  · B özellikleri: ln EWMA v_t; ln ort(r²) 5/22/63 gün; ln r_t²; 1{r_t<0}; ln (ln H/L)²_t; ln ort((ln H/L)²)
    5 gün. Hedef ln(r²_{t+1} + 1e-10). Öngörü f = exp(ŷ)·c, c = ort(r²)/ort(exp(ŷ)) EĞİTİM kümesinde.
    Birincil ölçü: piyasa başına günlük QLIKE farkı (ML − EWMA) ortalamasının NW t'si (DM); ek HAR/GARCH.

EK (ilk koşu GÖRÜLDÜKTEN SONRA eklendi; her biri bir MEKANİZMA teşhisine dayanır, kapı gevşetmez):
  1. SABİT-UZUN TABAN (ACI DERS #9'un pozisyon uzayı): ilk koşuda lojistiğin Sharpe'ı 13/14 piyasada
     "her gün uzun × σ ölçeği" politikasının Sharpe'ıyla BİREBİR aynı çıktı — uzun kalma payı %100.
     Model zamanlama değil SÜRÜKLENME öğrenmişti. Her ML politikası artık `sabit`e karşı da (fark t)
     ve uzun kalma payıyla raporlanır. Sürüklenmesi çıkarılmış hedefle (y − eğitim medyanı) eğitilen
     `lojistik_zamanlama` eklendi: yalnız ZAMANLAMA öğrenebilir; kıyası kurala (ikisi de zamanlar).
  2. BAYAT BAR (BULGU-97-5): oynaklık ML'inin QLIKE patlamalarının (Kore/Türkiye/Orta Doğu/Japonya)
     hepsi open=high=low=close VE kapanış değişmemiş yerel TATİL satırlarındandı (yfinance işlem
     olmayan günü satır olarak verir; Japonya 2018-01-03). Model sıfır aralığı "sıfır oynaklık" okuyup
     ertesi işlem günü için ~1e-8 varyans öngörüyordu. Bu satırlar İŞLEM GÜNÜ DEĞİLDİR → panelden atılır
     (tüm modeller için aynı; EWMA da ölçülür). Tur-96 düzeneği (`hazirla`) bu satırları hâlâ taşır —
     açık kalem.
  3. ÖRNEKLEM-DIŞI PENCERE 2011-2017: ön-kayıtlı pencere 2018-2026 idi ve yukarıdaki iki ek o pencerenin
     sonucuna BAKILARAK geldi. Bu yüzden aynı düzenek hiç bakılmamış 2011-2017'de de koşulur (yürüyen
     eğitim 2008'den); bir iddia İKİ pencerede de aynı yönde değilse kurulmaz.

CLI:  python -m research.quant_ml      → data/quant_ml_karnesi.json + tablolar
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research import coklu_piyasa as C

ROOT = Path(__file__).resolve().parents[1]
KARNE = ROOT / "data" / "quant_ml_karnesi.json"
UFUK_OZ = (5, 21, 63, 126, 252)
ILERI = 21
LAM = 0.94
EPS = 1e-10
TEST_YILLARI = tuple(range(2011, 2027))
PENCERELER = {"oos_2011_17": ("2011-01-01", "2017-12-31"), "test_2018_26": ("2018-01-01", "2026-12-31")}
A_OZ = [f"m{h}" for h in UFUK_OZ] + ["sig_rejim", "sig_kisa_uzun", "r_z", "zirve_z"]
B_OZ = ["ln_ewma", "ln_rv5", "ln_rv22", "ln_rv63", "ln_r2", "neg", "ln_hl2", "ln_hl2_5"]
POLITIKALAR = ("kural", "sabit", "lojistik", "lojistik_zamanlama", "gbm")


def _ewma_var(lr: np.ndarray) -> np.ndarray:
    return pd.Series(lr * lr).ewm(alpha=1 - LAM, adjust=False, ignore_na=True).mean().to_numpy()


def panel(sym: str, piyasa: str) -> pd.DataFrame:
    """Enstrüman başı günlük özellik + hedef tablosu (tüm satırlar nedensel özellik taşır)."""
    df = C.hazirla(sym)                                         # bayat barlar hazirla'da atılır
    o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = c.size
    lc = np.log(c)
    lr = np.full(n, np.nan)
    lr[1:] = np.diff(lc)
    v = _ewma_var(lr)
    sig = np.sqrt(v)
    sbar = pd.Series(sig).expanding(min_periods=63).mean().to_numpy()
    r2 = pd.Series(lr * lr)
    out = pd.DataFrame(index=df.index)
    out["piyasa"], out["sembol"] = piyasa, sym
    for hh in UFUK_OZ:
        m = np.full(n, np.nan)
        m[hh:] = (lc[hh:] - lc[:-hh]) / (sig[hh:] * math.sqrt(hh))
        out[f"m{hh}"] = m
    with np.errstate(divide="ignore", invalid="ignore"):
        out["sig_rejim"] = sig / sbar
        out["sig_kisa_uzun"] = np.sqrt(r2.rolling(5).mean().to_numpy() / r2.rolling(63).mean().to_numpy())
        out["r_z"] = lr / sig
        out["zirve_z"] = (lc - pd.Series(lc).rolling(252).max().to_numpy()) / sig
        # A hedefi: t+1 açılışı → t+22 açılışı, σ_t√21 ile normalize
        y = np.full(n, np.nan)
        y[: n - ILERI - 1] = np.log(o[ILERI + 1:] / o[1: n - ILERI]) / (sig[: n - ILERI - 1] * math.sqrt(ILERI))
    out["yA"] = np.clip(y, -3, 3)
    # Vektörel değerlendirme: open[t+1]→open[t+2]
    ileri = np.full(n, np.nan)
    ileri[: n - 2] = o[2:] / o[1:-1] - 1.0
    out["ileri"] = ileri
    out["olcek"] = np.minimum(sbar / sig, 2.0)
    isaret = np.vstack([np.sign(np.r_[np.full(k, np.nan), lc[k:] - lc[:-k]]) for k in (21, 63, 252)])
    out["kural_skor"] = isaret.mean(axis=0)
    # B özellikleri + hedef (t+1 kapanış→kapanış log getirinin karesi)
    hl2 = np.log(h / lo) ** 2
    out["ln_ewma"] = np.log(v + EPS)
    out["ln_rv5"] = np.log(r2.rolling(5).mean().to_numpy() + EPS)
    out["ln_rv22"] = np.log(r2.rolling(22).mean().to_numpy() + EPS)
    out["ln_rv63"] = np.log(r2.rolling(63).mean().to_numpy() + EPS)
    out["ln_r2"] = np.log(lr * lr + EPS)
    out["neg"] = (lr < 0).astype(float)
    out["ln_hl2"] = np.log(hl2 + EPS)
    out["ln_hl2_5"] = np.log(pd.Series(hl2).rolling(5).mean().to_numpy() + EPS)
    r2_ileri = np.full(n, np.nan)
    r2_ileri[:-1] = lr[1:] ** 2
    out["r2_ileri"] = r2_ileri
    out["ewma_f"] = v                                           # EWMA öngörüsü: v_t → r²_{t+1}
    return out


def tum_panel() -> pd.DataFrame:
    kull, _ = C.enstrumanlar()
    return pd.concat([panel(s, pz) for pz, s in kull])


def _gun(piyasa: str) -> int:
    from research import piyasalar as P
    return P.PIYASALAR.get(piyasa, {}).get("gun", 252)


def egitim_maskesi(tarih: pd.DatetimeIndex, yil: int, etiket_gun: int) -> np.ndarray:
    """Etiketi `yil` başından ÖNCE bilinen satırlar: t + etiket_gun iş günü < yıl başı (takvim
    yaklaşığı: etiket_gun × 7/5 + 3 gün tampon; tatiller etiketi yalnız GECİKTİRİR → güvenli taraf)."""
    sinir = pd.Timestamp(f"{yil}-01-01") - pd.Timedelta(days=int(etiket_gun * 7 / 5) + 3)
    return np.asarray(tarih < sinir)


# ---------------------------------------------------------------------------------------------------
# A) Trend durumu
# ---------------------------------------------------------------------------------------------------
def trend_tahmin(P: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    P = P.copy()
    for k in ("p_lojistik", "p_zamanlama", "yhat_gbm"):
        P[k] = np.nan
    ok = (P[A_OZ].notna().all(axis=1) & np.isfinite(P[A_OZ]).all(axis=1)).to_numpy()
    for yil in TEST_YILLARI:
        tr = ok & P["yA"].notna().to_numpy() & egitim_maskesi(P.index, yil, ILERI + 1)
        te = ok & np.asarray(P.index.year == yil)
        if tr.sum() < 5000 or te.sum() == 0:
            continue
        X, Xt = P.loc[tr, A_OZ].to_numpy(), P.loc[te, A_OZ].to_numpy()
        sc = StandardScaler().fit(X)
        ya = P.loc[tr, "yA"].to_numpy()
        lg = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), (ya > 0).astype(int))
        P.loc[te, "p_lojistik"] = lg.predict_proba(sc.transform(Xt))[:, 1]
        # EK-1: sürüklenmesi çıkarılmış hedef → model yalnız ZAMANLAMA öğrenebilir
        lz = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), (ya > np.median(ya)).astype(int))
        P.loc[te, "p_zamanlama"] = lz.predict_proba(sc.transform(Xt))[:, 1]
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=200, l2_regularization=1.0, random_state=42)
        gb.fit(X, ya)
        P.loc[te, "yhat_gbm"] = gb.predict(Xt)
    return P


def _uzun(g: pd.DataFrame, politika: str) -> np.ndarray:
    return {"kural": lambda: g["kural_skor"].to_numpy() > 0,
            "sabit": lambda: np.ones(len(g), dtype=bool),
            "lojistik": lambda: g["p_lojistik"].to_numpy() > 0.5,
            "lojistik_zamanlama": lambda: g["p_zamanlama"].to_numpy() > 0.5,
            "gbm": lambda: g["yhat_gbm"].to_numpy() > 0}[politika]()


def _poz_getiri(g: pd.DataFrame, uzun: np.ndarray, mal: float) -> pd.Series:
    w = np.nan_to_num(np.where(uzun, g["olcek"].to_numpy(), 0.0), nan=0.0)
    return pd.Series(C.pozisyon_getirisi(w, g["ileri"].to_numpy(), mal), index=g.index)


def _fark(a: pd.Series, b: pd.Series, gun: int) -> dict:
    ortak = a.index.intersection(b.index)
    d = (a.loc[ortak] - b.loc[ortak]).to_numpy()
    return {"t": C.nw_ortalama_t(d), "yillik_pp": round(float(np.mean(d)) * gun * 100, 3) if d.size else None}


def trend_degerlendir(P: pd.DataFrame, bas: str, son: str) -> dict:
    mal = C.maliyet_orani()
    pen = P[(P.index >= pd.Timestamp(bas)) & (P.index <= pd.Timestamp(son))]
    pen = pen[pen["p_lojistik"].notna()]                         # yalnız modelin tahmin ettiği satırlar
    skor_sutunu = {"kural": "kural_skor", "lojistik": "p_lojistik", "lojistik_zamanlama": "p_zamanlama",
                   "gbm": "yhat_gbm"}
    sonuc: dict = {}
    for pz, gp in pen.groupby("piyasa", sort=False):
        gun = _gun(pz)
        seri: dict[str, list[pd.Series]] = {k: [] for k in POLITIKALAR}
        uzun_payi: dict[str, list[float]] = {k: [] for k in POLITIKALAR}
        auc_ic: dict[str, list[float]] = {k: [] for k in skor_sutunu}
        for _s, g in gp.groupby("sembol", sort=False):
            for k in POLITIKALAR:
                u = _uzun(g, k)
                seri[k].append(_poz_getiri(g, u, mal))
                uzun_payi[k].append(float(np.mean(u)))
            yb = np.where(g["yA"].notna(), (g["yA"] > 0).astype(int), -1)
            for k, col in skor_sutunu.items():
                a = C.auc(yb, g[col].to_numpy())
                if a is not None:
                    auc_ic[k].append(a)
        ew = {k: C._ew(v) for k, v in seri.items()}
        blok = {}
        for k in POLITIKALAR:
            m = C.seri_metrik(ew[k].to_numpy(), gun)
            m["uzun_payi"] = round(float(np.mean(uzun_payi[k])), 4)
            if k in auc_ic:
                m["auc_ic"] = round(float(np.mean(auc_ic[k])), 4) if auc_ic[k] else None
            if k != "kural":
                m["fark_kural"] = _fark(ew[k], ew["kural"], gun)
            if k != "sabit":
                m["fark_sabit"] = _fark(ew[k], ew["sabit"], gun)
            blok[k] = m
        sonuc[pz] = blok
    return sonuc


# ---------------------------------------------------------------------------------------------------
# B) Oynaklık
# ---------------------------------------------------------------------------------------------------
def ml_f_uygula(P: pd.DataFrame, modeller: dict) -> np.ndarray:
    """Yıllık modelleri (yıl → (model, düzeltme)) satırın YILINA göre uygular; modelsiz yıl/eksik
    özellik → NaN. Havuz dışı enstrümanlar (ABD izleme listesi) için de aynı fonksiyon."""
    f = np.full(len(P), np.nan)
    ok = P[B_OZ].notna().all(axis=1).to_numpy() & np.isfinite(P[B_OZ].to_numpy()).all(axis=1)
    yil = np.asarray(P.index.year)
    for y, (gb, duzeltme) in modeller.items():
        te = ok & (yil == y)
        if te.any():
            f[te] = np.exp(gb.predict(P.loc[te, B_OZ].to_numpy())) * duzeltme
    return f


def vol_tahmin(P: pd.DataFrame, modeller: dict | None = None) -> pd.DataFrame:
    """Yürüyen yıllık eğitim; `modeller` verilirse yıl → (model, düzeltme) ile doldurulur."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    P = P.copy()
    P["ml_f"] = np.nan
    hedef = np.log(P["r2_ileri"].to_numpy() + EPS)
    ok = P[B_OZ].notna().all(axis=1).to_numpy() & np.isfinite(P[B_OZ].to_numpy()).all(axis=1)
    for yil in TEST_YILLARI:
        tr = ok & np.isfinite(hedef) & egitim_maskesi(P.index, yil, 1)
        te = ok & np.asarray(P.index.year == yil)
        if tr.sum() < 5000 or te.sum() == 0:
            continue
        X = P.loc[tr, B_OZ].to_numpy()
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=200, l2_regularization=1.0, random_state=42)
        gb.fit(X, hedef[tr])
        duzeltme = float(np.mean(P.loc[tr, "r2_ileri"].to_numpy()) / np.mean(np.exp(gb.predict(X))))
        P.loc[te, "ml_f"] = np.exp(gb.predict(P.loc[te, B_OZ].to_numpy())) * duzeltme
        if modeller is not None:
            modeller[yil] = (gb, duzeltme)
    return P


def qlike(f: np.ndarray, r2: np.ndarray) -> np.ndarray:
    f = np.where(f > 0, f, np.nan)
    return np.log(f) + r2 / f


def vol_degerlendir(P: pd.DataFrame, bas: str, son: str, tur96: bool) -> dict:
    """ML ↔ EWMA her pencerede; HAR/GARCH yalnız Tur-96 önbelleğinin kapsadığı pencerede (tur96=True).
    Tur-96 öngörüsü HEDEF gününe göre indekslidir; satır t'nin hedefi bu panelde bir sonraki satırdır
    (bayat barlar atıldığı için Tur-96'nın 'sonraki satır'ı ile her zaman aynı gün değildir → yalnız
    tarihleri eşleşen günler kıyaslanır)."""
    pen = P[(P.index >= pd.Timestamp(bas)) & (P.index <= pd.Timestamp(son))]
    sonuc: dict = {}
    for pz, gp in pen.groupby("piyasa", sort=False):
        kay: dict[str, list[pd.Series]] = {"ewma": [], "ml": []}
        if tur96:
            kay.update({"har": [], "garch": []})
        for s, g in gp.groupby("sembol", sort=False):
            r2 = g["r2_ileri"].to_numpy()
            kay["ewma"].append(pd.Series(qlike(g["ewma_f"].to_numpy(), r2), index=g.index))
            kay["ml"].append(pd.Series(qlike(g["ml_f"].to_numpy(), r2), index=g.index))
            if tur96:
                e = C.ens_oku(s)["vol"]
                sonraki = g.index[1:].append(pd.DatetimeIndex([pd.NaT]))
                for k, anahtar in (("har", "har_olcekli"), ("garch", "garch")):
                    f = pd.Series(np.asarray(e["tahmin"][anahtar], float), index=pd.DatetimeIndex(e["tarih"]))
                    kay[k].append(pd.Series(qlike(f.reindex(sonraki).to_numpy(), r2), index=g.index))
        with np.errstate(all="ignore"):
            ort = {k: pd.concat(v, axis=1).mean(axis=1, skipna=True) for k, v in kay.items()}
        ortak = ort["ewma"].dropna().index
        for k in ort:
            ortak = ortak.intersection(ort[k].dropna().index)
        blok: dict = {"n_gun": int(len(ortak)),
                      "qlike": {k: round(float(ort[k].loc[ortak].mean()), 5) for k in ort},
                      "dm_t_ml_ewma": C.nw_ortalama_t((ort["ml"].loc[ortak] - ort["ewma"].loc[ortak]).to_numpy())}
        if tur96:
            for k in ("har", "garch"):
                blok[f"dm_t_{k}_ewma"] = C.nw_ortalama_t((ort[k].loc[ortak] - ort["ewma"].loc[ortak]).to_numpy())
            blok["dm_t_ml_har"] = C.nw_ortalama_t((ort["ml"].loc[ortak] - ort["har"].loc[ortak]).to_numpy())
        sonuc[pz] = blok
    return sonuc


# ---------------------------------------------------------------------------------------------------
# C) Ekonomik test: σ_ML ile oynaklık hedefleme ↔ σ_EWMA (EK-4, ön-kayıt: aşağıdaki SIGMA_KARAR)
# ---------------------------------------------------------------------------------------------------
# σ_ML quant çekirdeğine YALNIZ şu koşulda bağlanır (QLIKE sonucu görüldükten SONRA ama bu testin
# sonucundan ÖNCE sabitlendi): hem `kural` hem `sabit` politikasında, İKİ pencerede de, fark t > 0 olan
# piyasa sayısı ≥ 10/14 VE ABD'de fark t > 1,645.
SIGMA_KARAR = {"min_piyasa": 10, "abd_t": 1.645}


def sigma_degerlendir(P: pd.DataFrame, bas: str, son: str) -> dict:
    mal = C.maliyet_orani()
    pen = P[(P.index >= pd.Timestamp(bas)) & (P.index <= pd.Timestamp(son))]
    sonuc: dict = {}
    for pz, gp in pen.groupby("piyasa", sort=False):
        gun = _gun(pz)
        seri: dict[str, list[pd.Series]] = {k: [] for k in ("kural_ewma", "kural_ml", "sabit_ewma", "sabit_ml")}
        for _s, g0 in gp.groupby("sembol", sort=False):
            sig_ml = np.sqrt(P.loc[P["sembol"] == _s, "ml_f"])      # σ̄ tüm ÖNCEKİ ML öngörülerinden
            sbar_ml = sig_ml.expanding(min_periods=63).mean()
            olcek_ml = np.minimum(sbar_ml / sig_ml, 2.0).reindex(g0.index)
            g = g0[np.isfinite(olcek_ml.to_numpy()) & np.isfinite(g0["olcek"].to_numpy())]
            if len(g) < 60:
                continue
            om = olcek_ml.loc[g.index].to_numpy()
            for pol, uzun in (("kural", g["kural_skor"].to_numpy() > 0), ("sabit", np.ones(len(g), dtype=bool))):
                seri[f"{pol}_ewma"].append(_poz_getiri(g, uzun, mal))
                w = np.where(uzun, om, 0.0)
                seri[f"{pol}_ml"].append(pd.Series(C.pozisyon_getirisi(w, g["ileri"].to_numpy(), mal), index=g.index))
        ew = {k: C._ew(v) for k, v in seri.items()}
        blok = {k: {"sharpe": C.seri_metrik(v.to_numpy(), gun).get("sharpe"),
                    "mdd_pct": C.seri_metrik(v.to_numpy(), gun).get("mdd_pct")} for k, v in ew.items()}
        for pol in ("kural", "sabit"):
            blok[f"{pol}_fark"] = _fark(ew[f"{pol}_ml"], ew[f"{pol}_ewma"], gun)
        sonuc[pz] = blok
    return sonuc


def sigma_hukmu(karne: dict) -> dict:
    """SIGMA_KARAR'ı karneye uygular (makine-okunur hüküm)."""
    kosullar = {}
    for pen, bloklar in karne["sigma"].items():
        for pol in ("kural", "sabit"):
            olumlu = sum(1 for b in bloklar.values() if (b[f"{pol}_fark"].get("t") or 0) > 0)
            abd_t = (bloklar.get("ABD", {}).get(f"{pol}_fark") or {}).get("t")
            kosullar[f"{pen}/{pol}"] = {"olumlu_piyasa": olumlu, "n_piyasa": len(bloklar), "abd_t": abd_t,
                                        "gecti": olumlu >= SIGMA_KARAR["min_piyasa"]
                                        and abd_t is not None and abd_t > SIGMA_KARAR["abd_t"]}
    return {"kosullar": kosullar, "baglanir": all(k["gecti"] for k in kosullar.values())}


def ozet(karne: dict) -> str:
    def f(x, nd=2):
        return "—" if x is None else f"{x:.{nd}f}"
    s = []
    for pen in PENCERELER:
        s += [f"### Trend — {pen}", "",
              "| piyasa | kural S (uzun %) | sabit S | lojistik S (uzun %) | zamanlama S (fark_kural t) | "
              "GBM S (uzun %) | AUC_ic kural/zaman/GBM |", "|---|---|---|---|---|---|---|"]
        for pz, b in karne["trend"][pen].items():
            s.append(f"| {pz} | {f(b['kural'].get('sharpe'))} ({f(100 * b['kural']['uzun_payi'], 0)}) | "
                     f"{f(b['sabit'].get('sharpe'))} | {f(b['lojistik'].get('sharpe'))} "
                     f"({f(100 * b['lojistik']['uzun_payi'], 0)}) | {f(b['lojistik_zamanlama'].get('sharpe'))} "
                     f"({f(b['lojistik_zamanlama']['fark_kural'].get('t'))}) | {f(b['gbm'].get('sharpe'))} "
                     f"({f(100 * b['gbm']['uzun_payi'], 0)}) | {f(b['kural'].get('auc_ic'), 3)}/"
                     f"{f(b['lojistik_zamanlama'].get('auc_ic'), 3)}/{f(b['gbm'].get('auc_ic'), 3)} |")
        s += ["", f"### Oynaklık — {pen}", "",
              "| piyasa | QLIKE EWMA | ML | HAR | GARCH | DM t ML−EWMA | DM t HAR−EWMA | DM t ML−HAR |",
              "|---|---|---|---|---|---|---|---|"]
        for pz, b in karne["vol"][pen].items():
            q = b["qlike"]
            s.append(f"| {pz} | {f(q['ewma'], 4)} | {f(q['ml'], 4)} | {f(q.get('har'), 4)} | "
                     f"{f(q.get('garch'), 4)} | {f(b['dm_t_ml_ewma'])} | {f(b.get('dm_t_har_ewma'))} | "
                     f"{f(b.get('dm_t_ml_har'))} |")
        s += ["", f"### σ kaynağı (ekonomik) — {pen}", "",
              "| piyasa | kural σEWMA S | kural σML S (t) | sabit σEWMA S | sabit σML S (t) |", "|---|---|---|---|---|"]
        for pz, b in karne.get("sigma", {}).get(pen, {}).items():
            s.append(f"| {pz} | {f(b['kural_ewma']['sharpe'])} | {f(b['kural_ml']['sharpe'])} "
                     f"({f(b['kural_fark'].get('t'))}) | {f(b['sabit_ewma']['sharpe'])} | "
                     f"{f(b['sabit_ml']['sharpe'])} ({f(b['sabit_fark'].get('t'))}) |")
        s.append("")
    if "sigma_hukmu" in karne:
        s.append(f"σ_ML bağlanır mı: {karne['sigma_hukmu']['baglanir']} — {karne['sigma_hukmu']['kosullar']}")
    return "\n".join(s)


def _cli() -> None:
    C._isci_hazirla()
    t = time.time()
    P = tum_panel()
    print(f"panel {len(P)} satır, {P['sembol'].nunique()} enstrüman, {round(time.time() - t, 1)} sn", flush=True)
    P = trend_tahmin(P)
    print(f"trend modelleri {round(time.time() - t, 1)} sn", flush=True)
    P = vol_tahmin(P)
    print(f"oynaklık modeli {round(time.time() - t, 1)} sn", flush=True)
    karne = {"uretim": time.strftime("%Y-%m-%d %H:%M"), "pencereler": PENCERELER,
             "maliyet_taraf": C.maliyet_orani(), "a_ozellik": A_OZ, "b_ozellik": B_OZ, "n_satir": int(len(P)),
             "trend": {p: trend_degerlendir(P, *PENCERELER[p]) for p in PENCERELER},
             "vol": {p: vol_degerlendir(P, *PENCERELER[p], tur96=p == "test_2018_26") for p in PENCERELER},
             "sigma": {p: sigma_degerlendir(P, *PENCERELER[p]) for p in PENCERELER}}
    karne["sigma_hukmu"] = sigma_hukmu(karne)
    karne["parmak_izi"] = C.girdi_parmak_izi(sorted({s for _, s in C.enstrumanlar()[0]}))   # Tur-99
    KARNE.write_text(json.dumps(karne, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(ozet(karne))
    print(f"karne → {KARNE}")


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    _cli()
