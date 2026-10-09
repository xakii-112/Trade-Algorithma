"""quant_ml_motor.py — Tur-98: Tur-97'nin iki ML kusurunu ONARIP karar mekanizmasında (motor) sınamak.

Tur-97 teşhisleri (docs/TUR97_RAPORU §4):
  K1 — σ_ML QLIKE'ta EWMA'yı yendi ama motorda yazı-tura çıktı. Kök neden: model ERTESİ GÜNÜN r²'sini
       öngörüyordu; `quant_trend` σ'yı yalnız GİRİŞTE, haftalarca taşınan bir stop/boyut için kullanır.
       Günlük hedef gürültülüdür (tek günün r²'si) ve kararın ufkuyla eşleşmez.
  K2 — Yön modeli sürüklenme öğrendi (uzun payı %99-100): "yarın yükselir mi?" hedefi, yükselen
       piyasalarda "hep evet"le çözülür. Kök neden: ML kuralın YERİNE yön tahmin ediyordu; kararın
       mekanizmasında ML'in yapabileceği tek şey kuralın girişlerini ELEMEKTİR (meta-etiketleme,
       López de Prado 2018 bölüm 3) — taban "hep uzun" değil, kuralın kendisidir.

ÖN-KAYIT (2026-09-26, bu modülün hiçbir çıktısı görülmeden sabitlendi):
  Ortak: Tur-96/97 motoru (`backtest_symbol` 1d, kill-switch kapalı, AI kapısı kapalı), Tur-97 σ
  motor testinin pencereleri (2011-17, 2018-26) ve evreni (14 piyasa havuzu + ABD_izleme). Modeller
  YALNIZ 14 piyasa havuzunda, yıllık yürüyen eğitimle kurulur (etiketi yıl başından önce bilinen
  satırlar); ABD_izleme satırları eğitime girmez (7 sembolü — SPY, QQQ, AAPL, MSFT, AMZN, JPM, XOM —
  ABD havuzunda da var: onların GEÇMİŞİ yürüyen eğitimde görülür, geleceği görülmez). Piyasa = eşit ağırlıklı enstrüman günlük getirisi.
  Birincil ölçü: yıllık Sharpe FARKI (aday − canlı). Çıkarım: Memmel (2003) düzeltmeli
  Jobson-Korkie z'si. Neden Sharpe: σ bir BOYUTLAMA girdisidir; Tur-97'nin ortalama getiri farkı
  daha küçük maruziyeti ceza sayıyordu (BULGU-97-7). Ortalama fark t'si, MDD, işlem sayısı ek ölçüdür.

  KAPI (her hipotez için ayrı; İKİ pencerede de): ΔSharpe > 0 olan piyasa ≥ 10 VE ABD_izleme
  Memmel z > 1,645. Geçen aday canlı quant çekirdeğine bağlanır; geçmeyen bağlanmaz.

  H1 — σ UFUK EŞLEME: σ_ML21 = √(öngörülen ORTALAMA günlük r², t+1..t+21). HistGradientBoosting,
       Tur-97 oynaklık özellikleri (B_OZ), hedef ln RV21, eğitimde kestirilen yanlılık düzeltmesi
       c = ort(RV21)/ort(exp ŷ). Aday − canlı = σ_ML21 − σ_EWMA(0,94).
  H2 — META KAPI: birincil = kural (skor > 0). İkincil model = HistGradientBoosting regresyonu,
       hedef Tur-97'nin yA'sı (t+1→t+22 açılış getirisi / σ√21), YALNIZ skor > 0 satırlarında
       eğitilir. Özellikler: Tur-97 trend özellikleri (A_OZ) + kural skoru + PİYASA REJİMİ (aynı
       piyasada skoru > 0 olan enstrüman payı, piyasa ort. m63, piyasa ort. σ rejimi). Politika:
       giriş YALNIZ ŷ > 0 ise (model "bu girişin beklenen getirisi negatif" demedikçe kural geçer).
       Aday − canlı = meta kapılı kural − kural. Yozlaşma bayrağı: skor > 0 günlerinin < %5'i
       elenirse model "etkisiz" sayılır (kapıdan geçse de bağlanmaz — kuralın kopyasıdır).

  BİLGİ KOLLARI (kapısız; karar vermez): σ_HAR21 (aynı hedef, doğrusal: ln RV5/22/63 → "ML gerekli
  mi?"), σ_ML1 (Tur-97 modeli — ΔSharpe'ı Tur-97'de GÖRÜLDÜ, burada yalnız sonradan-keşif satırıdır).
  Öngörü kaybı: 21 günlük hedefte QLIKE (EWMA v_t, ML21, HAR21), DM t'si NW gecikmesi ≥ 21 (örtüşen
  hedef).

CLI:  python -m research.quant_ml_motor --asama model          # paneller + yürüyen modeller → seriler
      python -m research.quant_ml_motor --asama motor --isci 8  # enstrüman başı motor kolları
      python -m research.quant_ml_motor --asama rapor           # data/quant_ml_motor_karnesi.json
"""
from __future__ import annotations

import contextlib
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research import coklu_piyasa as C
from research import quant_kayma as QK
from research import quant_ml as QM

ROOT = Path(__file__).resolve().parents[1]
SONUC_DIR = C.SONUC_DIR / "quant98"
SERI_YOLU = SONUC_DIR / "seriler98.pkl"
KARNE = ROOT / "data" / "quant_ml_motor_karnesi.json"
ILERI = QM.ILERI
HAR_OZ = ["ln_rv5", "ln_rv22", "ln_rv63"]
REJIM_OZ = ["gen_pay", "gen_m63", "gen_sig"]
META_OZ = QM.A_OZ + ["kural_skor"] + REJIM_OZ
KOLLAR = ("ewma", "ml21", "har21", "ml1", "meta")
# TEŞHİS (SONRADAN, kapısız): ilk koşudan sonra σ_ML21'in medyanda σ_EWMA'nın 1,091 katı olduğu görüldü
# (sakin rejimde 1,158). Motor farkı "daha iyi zamanlama" mı yoksa yalnız "daha geniş stop" mu? EWMA'yı
# aynı sabit katla genişleten kol ikisini ayırır. Sabit tüm panelden seçildi → hüküm vermez, yalnız okutur.
TESHIS_KOLLARI = ("ewma_k",)
TESHIS_OLCEK = 1.091
KARAR = {"min_piyasa": 10, "izleme_z": 1.645, "min_eleme": 0.05}


# ---------------------------------------------------------------------------------------------------
# Panel + modeller
# ---------------------------------------------------------------------------------------------------
def _panel(sym: str, piyasa: str) -> pd.DataFrame:
    p = QM.panel(sym, piyasa)
    # r2_ileri[t] = lr[t+1]² → 21 satırlık İLERİ ortalama = ort(lr[t+1..t+21]²)
    p["rv21"] = p["r2_ileri"][::-1].rolling(ILERI).mean()[::-1].to_numpy()
    return p


def rejim_ekle(P: pd.DataFrame) -> pd.DataFrame:
    """Aynı piyasa + aynı gün kesitinden rejim özellikleri (hepsi t kapanışında bilinir)."""
    P = P.copy()
    P["_t"] = P.index
    P["_uzun"] = np.where(P["kural_skor"].notna(), (P["kural_skor"] > 0).astype(float), np.nan)
    g = P.groupby(["piyasa", "_t"], sort=False)
    P["gen_pay"] = g["_uzun"].transform("mean")
    P["gen_m63"] = g["m63"].transform("mean")
    P["gen_sig"] = g["sig_rejim"].transform("mean")
    return P.drop(columns=["_t", "_uzun"])


def tum_paneller() -> pd.DataFrame:
    kull, _ = QK.evren()
    return rejim_ekle(pd.concat([_panel(s, pz) for pz, s in kull]))


def _ok(P: pd.DataFrame, oz: list[str]) -> np.ndarray:
    X = P[oz].to_numpy(float)
    return np.isfinite(X).all(axis=1)


def _gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=200, l2_regularization=1.0, random_state=42)


def sigma21_tahmin(P: pd.DataFrame, havuz: np.ndarray) -> pd.DataFrame:
    """ML21 ve HAR21: ln RV21 regresyonu, yıllık yürüyen; eğitim yalnız `havuz` satırları."""
    from sklearn.linear_model import LinearRegression
    P = P.copy()
    P["f_ml21"], P["f_har21"] = np.nan, np.nan
    hedef = np.log(P["rv21"].to_numpy(float) + QM.EPS)
    ok_b, ok_h = _ok(P, QM.B_OZ), _ok(P, HAR_OZ)
    yil = np.asarray(P.index.year)
    for y in QM.TEST_YILLARI:
        tr = havuz & np.isfinite(hedef) & QM.egitim_maskesi(P.index, y, ILERI + 1)
        for ad, oz, ok, model in (("f_ml21", QM.B_OZ, ok_b, _gbm()), ("f_har21", HAR_OZ, ok_h, LinearRegression())):
            m = tr & ok
            te = ok & (yil == y)
            if m.sum() < 5000 or not te.any():
                continue
            X = P.loc[m, oz].to_numpy(float)
            model.fit(X, hedef[m])
            c = float(np.mean(P.loc[m, "rv21"].to_numpy(float)) / np.mean(np.exp(model.predict(X))))
            P.loc[te, ad] = np.exp(model.predict(P.loc[te, oz].to_numpy(float))) * c
    return P


def meta_tahmin(P: pd.DataFrame, havuz: np.ndarray) -> pd.DataFrame:
    """Meta kapı: skor > 0 satırlarında yA regresyonu; `meta_ok` = 1 (gir) / 0 (ele) / NaN (tahmin yok)."""
    P = P.copy()
    P["meta_yhat"] = np.nan
    uzun = (P["kural_skor"] > 0).to_numpy()
    ok = _ok(P, META_OZ) & uzun
    yil = np.asarray(P.index.year)
    for y in QM.TEST_YILLARI:
        tr = havuz & ok & P["yA"].notna().to_numpy() & QM.egitim_maskesi(P.index, y, ILERI + 1)
        te = ok & (yil == y)
        if tr.sum() < 5000 or not te.any():
            continue
        gb = _gbm().fit(P.loc[tr, META_OZ].to_numpy(float), P.loc[tr, "yA"].to_numpy(float))
        P.loc[te, "meta_yhat"] = gb.predict(P.loc[te, META_OZ].to_numpy(float))
    P["meta_ok"] = np.where(P["meta_yhat"].notna(), (P["meta_yhat"] > 0).astype(float), np.nan)
    return P


def model_asamasi() -> pd.DataFrame:
    t = time.time()
    P = tum_paneller()
    havuz = (P["piyasa"] != QK.IZLEME).to_numpy()
    print(f"panel {len(P)} satır, {P['sembol'].nunique()} enstrüman ({round(time.time() - t)} sn)", flush=True)
    P = sigma21_tahmin(P, havuz)
    print(f"σ21 modelleri ({round(time.time() - t)} sn)", flush=True)
    # Tur-97 ML1: aynı eğitim kuralı; havuz dışı satırlar tahmin alır ama eğitime girmez
    modeller: dict = {}
    QM.vol_tahmin(P[havuz], modeller)
    P["f_ml1"] = QM.ml_f_uygula(P, modeller)
    print(f"ML1 modeli ({round(time.time() - t)} sn)", flush=True)
    P = meta_tahmin(P, havuz)
    print(f"meta modeli ({round(time.time() - t)} sn)", flush=True)
    SONUC_DIR.mkdir(parents=True, exist_ok=True)
    seriler = {k: g[["f_ml21", "f_har21", "f_ml1", "meta_ok"]].copy()
               for k, g in P.groupby(["piyasa", "sembol"], sort=False)}
    with open(SERI_YOLU, "wb") as f:
        pickle.dump(seriler, f)
    sut = ["piyasa", "sembol", "rv21", "ewma_f", "f_ml21", "f_har21", "f_ml1", "kural_skor", "meta_ok", "meta_yhat"]
    P[sut].to_pickle(SONUC_DIR / "panel98.pkl")
    return P


# ---------------------------------------------------------------------------------------------------
# Motor kolları
# ---------------------------------------------------------------------------------------------------
_SERI: dict = {"v": None, "kol": None}


@contextlib.contextmanager
def yama(sigma: pd.Series | None = None, meta: pd.Series | None = None, olcek: float = 1.0):
    """Motorun quant (skor, σ) çiftini geçici değiştirir: σ → `sigma` (eksik gün EWMA'ya düşer);
    `meta` == 0 olan günlerde skor > 0 → 0 (giriş yok, QUANT_NOTR). Orijinal Tur-97 ile ORTAK
    sözlükte bir kez saklanır (BULGU-96-5: sarmalar zincirlenmez)."""
    from nihai.signals import quant as Q
    from nihai.signals import strategy as S
    orj = QK._ORIJINAL.setdefault("quant_np", S._SharedIndicators.quant_np)

    def quant_np(self):
        def _kur():
            skor, sig = Q.seriler(self.closes())
            kol = _SERI["kol"]
            if kol["sigma"] is not None:
                s2 = kol["sigma"].reindex(self.df.index).to_numpy(dtype=float)
                sig = np.where(np.isfinite(s2) & (s2 > 0), s2, sig)
            if kol["meta"] is not None:
                m = kol["meta"].reindex(self.df.index).to_numpy(dtype=float)
                skor = np.where((m == 0) & (skor > 0), 0.0, skor)
            return skor, sig * kol["olcek"]
        return self._get("quant_np_98", _kur)

    _SERI["kol"] = {"sigma": sigma, "meta": meta, "olcek": float(olcek)}
    setattr(S._SharedIndicators, "quant_np", quant_np)      # noqa: B010 — metot ataması (mypy)
    try:
        yield
    finally:
        setattr(S._SharedIndicators, "quant_np", orj)     # noqa: B010
        _SERI["kol"] = None


def _seriler() -> dict:
    if _SERI["v"] is None:
        with open(SERI_YOLU, "rb") as f:
            _SERI["v"] = pickle.load(f)
    return _SERI["v"]


def kol_argumanlari(s: pd.DataFrame) -> dict[str, dict]:
    def sig(k: str) -> pd.Series:
        return np.sqrt(s[k])
    return {"ewma": {}, "ml21": {"sigma": sig("f_ml21")}, "har21": {"sigma": sig("f_har21")},
            "ml1": {"sigma": sig("f_ml1")}, "meta": {"meta": s["meta_ok"]}, "ewma_k": {"olcek": TESHIS_OLCEK}}


def _m98(piyasa: str, sym: str) -> Path:
    return SONUC_DIR / QK.dosya(piyasa, sym, "m98_").name


def enstruman_kos(arg: tuple[str, str]) -> str:
    piyasa, sym = arg
    C._isci_hazirla()
    t = time.time()
    df = C.hazirla(sym)
    kol = kol_argumanlari(_seriler()[(piyasa, sym)])
    out: dict = {"piyasa": piyasa, "sembol": sym, "kos": {}}
    for pen, (bas, son) in QK.SIGMA_PENCERE.items():
        d = df if son is None else df[df.index < pd.Timestamp(son)]
        t0 = int((d.index < pd.Timestamp(bas)).sum())
        if len(d) - t0 < 250 or t0 < 60:
            continue
        for ad in KOLLAR + TESHIS_KOLLARI:
            if kol[ad]:
                with yama(**kol[ad]):
                    out["kos"][f"{ad}_{pen}"] = QK._kos(sym, d, t0, "quant_trend")
            else:
                out["kos"][f"{ad}_{pen}"] = QK._kos(sym, d, t0, "quant_trend")
    with open(_m98(piyasa, sym), "wb") as f:
        pickle.dump(out, f)
    return f"{piyasa}/{sym}: {len(out['kos'])} kol, {round(time.time() - t, 1)} sn"


# ---------------------------------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------------------------------
def memmel_z(a: pd.Series, b: pd.Series) -> dict:
    """Sharpe farkı (a − b) için Memmel (2003) düzeltmeli Jobson-Korkie z'si + günlük Sharpe'lar."""
    ortak = a.index.intersection(b.index)
    x, y = a.loc[ortak].to_numpy(float), b.loc[ortak].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 60 or x.std() == 0 or y.std() == 0:
        return {"z": None, "n": int(x.size)}
    sa, sb = x.mean() / x.std(ddof=1), y.mean() / y.std(ddof=1)
    rho = float(np.corrcoef(x, y)[0, 1])
    v = (2 - 2 * rho + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / x.size
    return {"z": round(float((sa - sb) / math.sqrt(v)), 3) if v > 0 else None, "rho": round(rho, 4),
            "n": int(x.size)}


def nw_t(d: np.ndarray, gecikme: int) -> float | None:
    """Ortalamanın NW t'si, gecikme en az `gecikme` (örtüşen çok günlük hedefler için)."""
    from nihai.data.factors import _hac_se
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 60:
        return None
    L = max(int(4 * (d.size / 100.0) ** (2.0 / 9.0)), gecikme)
    se = _hac_se(np.ones((d.size, 1)), d - d.mean(), np.array([[1.0 / d.size]]), L)
    return round(float(d.mean() / se[0]), 3) if se[0] > 0 else None


def ongoru_kaybi(Pn: pd.DataFrame) -> dict:
    """21 günlük hedefte QLIKE: EWMA v_t · ML21 · HAR21 (piyasa = enstrüman kayıplarının günlük ortalaması)."""
    sonuc: dict = {}
    for pen, (bas, son) in QK.SIGMA_PENCERE.items():
        p = Pn[(Pn.index >= pd.Timestamp(bas)) & ((Pn.index < pd.Timestamp(son)) if son else True)]
        blok: dict = {}
        for pz, g in p.groupby("piyasa", sort=False):
            rv = g["rv21"].to_numpy(float)
            kay = {k: QM.qlike(g[c].to_numpy(float), rv)
                   for k, c in (("ewma", "ewma_f"), ("ml21", "f_ml21"), ("har21", "f_har21"))}
            df = pd.DataFrame(kay, index=g.index).replace([np.inf, -np.inf], np.nan).dropna()
            ort = df.groupby(level=0).mean()
            blok[pz] = {"n_gun": int(len(ort)), "qlike": {k: round(float(ort[k].mean()), 5) for k in ort},
                        "dm_t_ml21_ewma": nw_t((ort["ml21"] - ort["ewma"]).to_numpy(), ILERI),
                        "dm_t_har21_ewma": nw_t((ort["har21"] - ort["ewma"]).to_numpy(), ILERI),
                        "dm_t_ml21_har21": nw_t((ort["ml21"] - ort["har21"]).to_numpy(), ILERI)}
        sonuc[pen] = blok
    return sonuc


def meta_eleme(Pn: pd.DataFrame) -> dict:
    """Pencere × piyasa: skor > 0 VE tahminli günlerin elenen payı (yozlaşma bayrağı)."""
    sonuc: dict = {}
    for pen, (bas, son) in QK.SIGMA_PENCERE.items():
        p = Pn[(Pn.index >= pd.Timestamp(bas)) & ((Pn.index < pd.Timestamp(son)) if son else True)]
        p = p[(p["kural_skor"] > 0) & p["meta_ok"].notna()]
        sonuc[pen] = {pz: round(float((g["meta_ok"] == 0).mean()), 4) for pz, g in p.groupby("piyasa", sort=False)}
        sonuc[pen]["_hepsi"] = round(float((p["meta_ok"] == 0).mean()), 4) if len(p) else None
    return sonuc


def rapor(kull: list[tuple[str, str]]) -> dict:
    piyasalar: dict[str, list[str]] = {}
    for pz, s in kull:
        piyasalar.setdefault(pz, []).append(s)
    karne: dict = {"uretim": time.strftime("%Y-%m-%d %H:%M"), "pencereler": QK.SIGMA_PENCERE, "karar": KARAR,
                   "piyasalar": {}}
    for pz, semboller in piyasalar.items():
        gun = QK._gun(pz)
        seri: dict[str, list[pd.Series]] = {}
        islem: dict[str, int] = {}
        for s in semboller:
            p = _m98(pz, s)
            if not p.exists():
                continue
            with open(p, "rb") as f:
                q = pickle.load(f)
            for ad, k in q["kos"].items():
                seri.setdefault(ad, []).append(k["s"])
                islem[ad] = islem.get(ad, 0) + int(k["islem"])
        ew = {ad: C._ew(v) for ad, v in seri.items()}
        blok: dict = {}
        for pen in QK.SIGMA_PENCERE:
            taban = ew.get(f"ewma_{pen}")
            if taban is None:
                continue
            b: dict = {"ewma": {**C.seri_metrik(taban.to_numpy(), gun), "islem": islem.get(f"ewma_{pen}")}}
            for ad in KOLLAR[1:] + TESHIS_KOLLARI:
                a = ew.get(f"{ad}_{pen}")
                if a is None:
                    continue
                m = {**C.seri_metrik(a.to_numpy(), gun), "islem": islem.get(f"{ad}_{pen}")}
                sa, sb = m.get("sharpe"), b["ewma"].get("sharpe")
                m["d_sharpe"] = round(sa - sb, 3) if sa is not None and sb is not None else None
                m["memmel"] = memmel_z(a, taban)
                m["fark_ort"] = QK.fark_t(a, taban)
                b[ad] = m
            blok[pen] = b
        karne["piyasalar"][pz] = blok
    karne["hukum"] = hukum(karne)
    return karne


def hukum(karne: dict) -> dict:
    out: dict = {}
    for ad in ("ml21", "meta", "har21", "ml1", *TESHIS_KOLLARI):
        kos: dict = {}
        for pen in QK.SIGMA_PENCERE:
            bl = [b[pen][ad] for b in karne["piyasalar"].values() if pen in b and ad in b[pen]]
            olumlu = sum(1 for m in bl if (m.get("d_sharpe") or 0) > 0)
            izl = ((karne["piyasalar"].get(QK.IZLEME, {}).get(pen) or {}).get(ad) or {}).get("memmel", {}).get("z")
            kos[pen] = {"olumlu_piyasa": olumlu, "n_piyasa": len(bl), "izleme_z": izl,
                        "gecti": olumlu >= KARAR["min_piyasa"] and izl is not None and izl > KARAR["izleme_z"]}
        out[ad] = {"kosullar": kos, "gecti": all(k["gecti"] for k in kos.values()),
                   "kapili": ad in ("ml21", "meta")}
    return out


def meta_yozlasma(karne: dict) -> dict:
    """Ön-kayıtlı yozlaşma bayrağı: bir pencerede skor > 0 günlerinin < %5'i elenirse meta kapı kuralın
    kopyasıdır → kapıdan geçse de bağlanmaz. Hüküm her pencerede ayrıca raporlanır."""
    h = karne["hukum"]["meta"]
    paylar = [pen.get("_hepsi") for pen in karne.get("meta_eleme", {}).values()]
    h["eleme_payi"] = paylar
    h["yozlasma"] = any(v is None or v < KARAR["min_eleme"] for v in paylar) if paylar else True
    if h["yozlasma"]:
        h["gecti"] = False
    return karne


def ozet_md(karne: dict) -> str:
    def f(x, nd=2):
        return "—" if x is None else f"{x:.{nd}f}"
    s = []
    for pen in QK.SIGMA_PENCERE:
        s += [f"### Motor — {pen}", "",
              "| piyasa | EWMA S | ML21 ΔS (z) | HAR21 ΔS (z) | ML1 ΔS (z) | META ΔS (z) | EWMA×k ΔS (z) | MDD EWMA/ML21/META |",
              "|---|---|---|---|---|---|---|---|"]
        for pz, b in karne["piyasalar"].items():
            if pen not in b:
                continue
            x = b[pen]

            def h(ad, x=x):
                m = x.get(ad) or {}
                return f"{f(m.get('d_sharpe'))} ({f((m.get('memmel') or {}).get('z'))})"
            s.append(f"| {pz} | {f(x['ewma'].get('sharpe'))} | {h('ml21')} | {h('har21')} | {h('ml1')} | {h('meta')} | {h('ewma_k')} | "
                     f"{f(x['ewma'].get('mdd_pct'), 1)}/{f((x.get('ml21') or {}).get('mdd_pct'), 1)}/"
                     f"{f((x.get('meta') or {}).get('mdd_pct'), 1)} |")
        s.append("")
    for ad, hk in karne["hukum"].items():
        s.append(f"- {ad} ({'KAPILI' if hk['kapili'] else 'bilgi'}): geçti={hk['gecti']} {hk['kosullar']}")
    if "meta_eleme" in karne:
        s.append(f"- meta eleme payı: {karne['meta_eleme']}")
    return "\n".join(s)


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Tur-98 ML onarımı — motor sınaması")
    ap.add_argument("--asama", required=True, choices=["model", "motor", "rapor"])
    ap.add_argument("--isci", type=int, default=8)
    a = ap.parse_args()
    C._isci_hazirla()
    kull, _ = QK.evren()
    if a.asama == "model":
        model_asamasi()
    elif a.asama == "motor":
        C._havuz(enstruman_kos, kull, a.isci)
    else:
        karne = rapor(kull)
        Pn = pd.read_pickle(SONUC_DIR / "panel98.pkl")
        karne["ongoru_21g"] = ongoru_kaybi(Pn)
        karne["meta_eleme"] = meta_eleme(Pn)
        meta_yozlasma(karne)
        karne["parmak_izi"] = C.girdi_parmak_izi(sorted({s for _, s in kull}))   # Tur-99
        KARNE.write_text(json.dumps(karne, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(ozet_md(karne))
        print(f"karne → {KARNE}")


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    _cli()
