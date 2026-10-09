"""Ön-kayıtlı ileri sınama: σ_ML21 ↔ EWMA — `quant_trend`in σ kaynağı (Tur-98/99'un açık adayı).

Kurallar, evren, model tarifi, kapılar ve karar günleri `docs/ONKAYIT_SIGMA_ML21_2026-09-26.md`'de DONDURULDU.
Bu dosya yalnız ölçer; hiçbir parametre sonuç görüldükten sonra değişmez.

    python -m research.ileri_sigma_ml21 kilitle    # BİR KEZ (2026-09-27 UTC'ye kadar): veri, evren, 2026 modeli, kanarya
    python -m research.ileri_sigma_ml21 guncelle   # veri yenile → kilidi doğrula → defterlere yeni ileri günler → rapor
    python -m research.ileri_sigma_ml21 rapor      # yalnız defterlerden rapor (önbellekteki veriyle)

Kilit DAVRANIŞSALDIR: her piyasanın ilk sembolünün kilit günü verisi KOPYALANIR (kanarya) ve iki kolun 2026-01-01 →
kayıt günü getirileri + σ_ML21 serisi o kopyayla yeniden hesaplanır. Kod, ayar ya da kütüphane davranışı değiştiyse
kanarya tutmaz ve deftere yazılmaz. yfinance revizyonu kanaryayı etkilemez (kopya donmuş); taze veride yazılmış
satırın yeniden hesabı farklıysa satır DEĞİŞMEZ, yalnız sayısı raporlanır.
"""
from __future__ import annotations

import json
import math
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research import coklu_piyasa as C
from research import quant_kayma as QK
from research import quant_ml as QM
from research.ileri_kripto_yaris import _sha, parmak_izi

ROOT = Path(__file__).resolve().parents[1]
KOK = ROOT / "data" / "ileri_sigma_ml21"
DEFTER_R = KOK / "defter_getiri.csv"
DEFTER_T = KOK / "defter_ongoru.csv"
MODEL_DIR = KOK / "modeller"
MODEL_KAYIT = KOK / "modeller.json"
KANARYA_VERI = KOK / "kanarya_veri"
KILIT = ROOT / "research" / "ileri_sigma_ml21_kilit.json"
ONKAYIT = ROOT / "docs" / "ONKAYIT_SIGMA_ML21_2026-09-26.md"
VERI = C.SONUC_DIR.parent / "ileri_sigma_ml21_veri"      # sınamaya ÖZEL önbellek, OneDrive DIŞI (BULGU-98-6)

KAYIT_GUNU = pd.Timestamp("2026-09-27")     # ileri = bu günden SONRA gerçekleşen
MOTOR_BAS = pd.Timestamp("2026-01-01")      # iki kolun ortak başlangıcı (ısınma geçmişi 2008'den)
H1_SON, H1_KARAR = pd.Timestamp("2027-08-31"), pd.Timestamp("2027-09-30")
H2_KARAR = pd.Timestamp("2031-09-30")
KOLLAR = ("ewma", "ml21")
KAPI = {"min_piyasa": 10, "z": 1.645, "min_ortak": 60}
NW_GECIKME = QM.ILERI                         # örtüşen 21 günlük hedef
# Tur-98 `quant_ml_motor._gbm` tarifi — burada DONDURULDU (oradaki değişiklik bu sınamayı değiştirmesin; test eşitliği kilitler)
GBM = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 200,
       "l2_regularization": 1.0, "random_state": 42}
TOL = 1e-9


# ── veri ───────────────────────────────────────────────────────────────────────────────────────────
def _veri_kaynak(yol: Path) -> None:
    """Önbellek yolunu değiştir ve VIX'i o yoldan yeniden yükle (motor VIX'i modül global'inden okur)."""
    from research import piyasalar as P
    P.CACHE_DIR = yol
    C._HAZIR["v"] = False
    C._isci_hazirla()


def indir(semboller: list[str]) -> list[str]:
    """VERI'ye tam geçmiş indir; indirilemeyenleri döndür (önbellekteki eski dosya kalır)."""
    from research import piyasalar as P
    P.CACHE_DIR = VERI
    eksik = []
    for s in semboller:
        try:
            if P.indir(s).empty:
                eksik.append(s)
        except Exception as e:  # noqa: BLE001 — bir sembolün ağ hatası güncellemeyi durdurmaz, YAZILIR
            eksik.append(f"{s} ({type(e).__name__})")
    return eksik


def panel(evren: list) -> pd.DataFrame:
    from research import quant_ml_motor as QMM
    return pd.concat([QMM._panel(s, pz) for pz, s in evren])


# ── model ──────────────────────────────────────────────────────────────────────────────────────────
def model_egit(P: pd.DataFrame, yil: int) -> dict:
    """Tur-98 σ_ML21: ln RV21 regresyonu; eğitim = havuz (ABD_izleme hariç) satırları, etiketi `yil` başından önce
    bilinen; yanlılık düzeltmesi c eğitim kümesinde."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    hedef = np.log(P["rv21"].to_numpy(float) + QM.EPS)
    X_hepsi = P[QM.B_OZ].to_numpy(float)
    m = ((P["piyasa"] != QK.IZLEME).to_numpy() & np.isfinite(X_hepsi).all(axis=1) & np.isfinite(hedef)
         & QM.egitim_maskesi(P.index, yil, QM.ILERI + 1))
    if m.sum() < 5000:
        raise RuntimeError(f"{yil} modeli için eğitim satırı yetersiz ({int(m.sum())})")
    model = HistGradientBoostingRegressor(**GBM).fit(X_hepsi[m], hedef[m])
    c = float(np.mean(P["rv21"].to_numpy(float)[m]) / np.mean(np.exp(model.predict(X_hepsi[m]))))
    return {"yil": yil, "model": model, "c": c, "n": int(m.sum())}


def model_kaydet(d: dict) -> str:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    p = MODEL_DIR / f"{d['yil']}.pkl"
    with open(p, "wb") as f:
        pickle.dump(d, f)
    return _sha(p)


def modeller_yukle() -> tuple[dict[int, dict], list[str]]:
    """Kayıtlı yıllık modeller + sha sorunları (dosya değiştiyse / kayıp)."""
    kayit = json.loads(MODEL_KAYIT.read_text(encoding="utf-8")) if MODEL_KAYIT.exists() else {}
    out, sorun = {}, []
    for y, bilgi in kayit.items():
        p = MODEL_DIR / f"{y}.pkl"
        if not p.exists() or _sha(p) != bilgi["sha"]:
            sorun.append(f"{y} modeli kayıp ya da değişti")
            continue
        with open(p, "rb") as f:
            out[int(y)] = pickle.load(f)
    return out, sorun


def _model_kayda_ekle(d: dict, sha: str) -> None:
    kayit = json.loads(MODEL_KAYIT.read_text(encoding="utf-8")) if MODEL_KAYIT.exists() else {}
    kayit[str(d["yil"])] = {"sha": sha, "n": d["n"], "c": d["c"],
                            "olusturma_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    MODEL_KAYIT.write_text(json.dumps(kayit, ensure_ascii=False, indent=1), encoding="utf-8")


def ongoru(P: pd.DataFrame, modeller: dict[int, dict]) -> np.ndarray:
    """σ_ML21 varyans öngörüsü: Y yılının satırı model Y ile. MOTOR_BAS sonrası bir yılın modeli yoksa DURUR
    (eksik öngörü motorda sessizce EWMA'ya düşer ve iki kol aynılaşırdı)."""
    f = np.full(len(P), np.nan)
    X = P[QM.B_OZ].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    yil = np.asarray(P.index.year)
    gerekli = {int(y) for y in np.unique(yil[np.asarray(P.index >= MOTOR_BAS)])}
    if gerekli - set(modeller):
        raise RuntimeError(f"model yok: {sorted(gerekli - set(modeller))}")
    for y in gerekli:
        te = ok & (yil == y)
        if te.any():
            d = modeller[y]
            f[te] = np.exp(d["model"].predict(X[te])) * d["c"]
    return f


# ── kollar ─────────────────────────────────────────────────────────────────────────────────────────
def kos(sym: str, df: pd.DataFrame, f_ml21: pd.Series) -> dict[str, pd.Series]:
    """İki kol, aynı motor, aynı başlangıç; tek değişken σ kaynağı (Tur-98 yama kuralı)."""
    from research import quant_ml_motor as QMM
    t0 = int((df.index < MOTOR_BAS).sum())
    out = {"ewma": QK._kos(sym, df, t0, "quant_trend")["s"]}
    with QMM.yama(sigma=np.sqrt(f_ml21)):
        out["ml21"] = QK._kos(sym, df, t0, "quant_trend")["s"]
    return {k: v.astype(float) for k, v in out.items()}


def getiriler(evren: list, P: pd.DataFrame, modeller: dict[int, dict], son: pd.Timestamp
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """→ (R [tarih, piyasa, sembol, kol, getiri], T [tarih, piyasa, sembol, ewma_f, f_ml21]); tarih < `son`."""
    P = P[P.index < son]                             # tamamlanmamış bugünkü bar (ve modeli olmayan yılı) girmez
    P = P.assign(f_ml21=ongoru(P, modeller))
    r_par, t_par = [], []
    for pz, s in evren:
        g = P[(P["piyasa"] == pz) & (P["sembol"] == s)]
        df = C.hazirla(s)
        df = df[df.index < son]
        for kol, r in kos(s, df, g["f_ml21"]).items():
            r_par.append(pd.DataFrame({"tarih": r.index, "piyasa": pz, "sembol": s, "kol": kol, "getiri": r.to_numpy()}))
        g = g[(g.index >= MOTOR_BAS) & (g.index < son)]
        t_par.append(pd.DataFrame({"tarih": g.index, "piyasa": pz, "sembol": s,
                                   "ewma_f": g["ewma_f"].to_numpy(), "f_ml21": g["f_ml21"].to_numpy()}))
    return pd.concat(r_par, ignore_index=True), pd.concat(t_par, ignore_index=True)


# ── kilit ──────────────────────────────────────────────────────────────────────────────────────────
def kanarya_sembolleri(evren: list) -> list:
    ilk: dict[str, str] = {}
    for pz, s in evren:
        ilk.setdefault(pz, s)
    return [[pz, s] for pz, s in ilk.items()]


def kanarya_izi(kanarya: list, modeller: dict[int, dict]) -> dict:
    """Donmuş kopyayla iki kolun MOTOR_BAS → KAYIT_GUNU getirileri + σ_ML21 parmak izi."""
    _veri_kaynak(KANARYA_VERI)
    try:
        P = panel(kanarya)
        P = P[P.index < KAYIT_GUNU + pd.Timedelta(days=1)]
        R, T = getiriler(kanarya, P, modeller, KAYIT_GUNU + pd.Timedelta(days=1))
    finally:
        _veri_kaynak(VERI)
    iz: dict = {}
    for (pz, s), g in R.groupby(["piyasa", "sembol"], sort=False):
        iz[f"{pz}/{s}"] = {k: parmak_izi(x.set_index("tarih")["getiri"]) for k, x in g.groupby("kol")}
    for (pz, s), g in T.groupby(["piyasa", "sembol"], sort=False):
        iz[f"{pz}/{s}"]["sigma"] = parmak_izi(g.set_index("tarih")["f_ml21"].fillna(0.0))
    return iz


def iz_farki(eski: dict, yeni: dict) -> list[str]:
    """Parmak izleri: sha eşitse tamam; değilse yıllık toplamlar (TOL) hakem (10. hanede yuvarlama gürültüsü)."""
    sorun = []
    for ad, kollar in eski.items():
        for k, iz in kollar.items():
            y = yeni.get(ad, {}).get(k)
            if y is not None and y["sha"] == iz["sha"]:
                continue
            yillar = [a for a, v in iz["yil"].items() if y is None or not abs(y["yil"].get(a, np.nan) - v) <= TOL]
            if y is None or yillar or y["n"] != iz["n"]:
                sorun.append(f"kanarya {ad}/{k}: {'yok' if y is None else 'değişti'} → kod, ayar ya da kütüphane")
    return sorun


def dogrula(kilit: dict, modeller_sorun: list[str]) -> list[str]:
    from research import piyasalar as P
    sorun = [] if _sha(ONKAYIT) == kilit["onkayit_sha256"] else ["ön-kayıt belgesi kilitten sonra DEĞİŞTİ"]
    sorun += modeller_sorun
    if sorun:
        return sorun
    modeller, _ = modeller_yukle()
    if modeller.get(2026) is None or _sha(MODEL_DIR / "2026.pkl") != kilit["model_2026_sha"]:
        return ["2026 modeli kilitteki değil"]
    for _pz, s in kilit["kanarya"]:
        if not (KANARYA_VERI / P._dosya(s, "1d").name).exists():
            return [f"kanarya verisi kayıp ({s})"]
    return iz_farki(kilit["kanarya_izi"], kanarya_izi(kilit["kanarya"], {2026: modeller[2026]}))


# ── defter ─────────────────────────────────────────────────────────────────────────────────────────
def defter_oku(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, parse_dates=["tarih"]) if p.exists() else pd.DataFrame()


def yeni_satirlar(Y: pd.DataFrame, defter: pd.DataFrame, anahtar: list[str]) -> pd.DataFrame:
    """Kayıt gününden SONRA ve o anahtarın (sembol, kol) deftere yazılmış son gününden SONRAKİ satırlar."""
    y = Y[Y.tarih > KAYIT_GUNU]
    if not defter.empty:
        son = defter.groupby(anahtar).tarih.max().rename("_son").reset_index()
        y = y.merge(son, on=anahtar, how="left")
        y = y[y._son.isna() | (y.tarih > y._son)].drop(columns="_son")
    return y.sort_values(["tarih", *anahtar]).reset_index(drop=True)


def uyusmayan(Y: pd.DataFrame, defter: pd.DataFrame, anahtar: list[str], deger: list[str]) -> int:
    """Yazılmış satırlardan yeniden hesabı farklı olanların sayısı (yfinance revizyonu; satır değişmez)."""
    if defter.empty:
        return 0
    k = defter.merge(Y, on=["tarih", *anahtar], how="left", suffixes=("", "_yeni"))
    esit = [((k[d] - k[f"{d}_yeni"]).abs() <= TOL * np.maximum(1.0, k[d].abs())) | (k[d].isna() & k[f"{d}_yeni"].isna())
            for d in deger]
    return int(sum((~e).sum() for e in esit))


def _ekle(p: Path, yeni: pd.DataFrame) -> None:
    if len(yeni):
        KOK.mkdir(parents=True, exist_ok=True)
        yeni.to_csv(p, mode="a", header=not p.exists(), index=False, float_format="%.12g")


# ── ölçüm ──────────────────────────────────────────────────────────────────────────────────────────
def h2_olcum(R: pd.DataFrame, son: pd.Timestamp = H2_KARAR) -> dict:
    """Piyasa ΔSharpe (ml21 − ewma) + Memmel z + birleşik z (R ileri dönemin fark serilerinden) + güncel MDE."""
    from research import quant_karar as QKR
    R = R[R.tarih <= son]
    piyasa: dict = {}
    fark: dict = {}
    for pz, g in R.groupby("piyasa", sort=False):
        ew = g.pivot_table(index="tarih", columns="kol", values="getiri", aggfunc="mean").dropna()
        if set(KOLLAR) - set(ew.columns):
            continue
        piyasa[pz] = QKR.sharpe_fark(ew["ml21"], ew["ewma"], QK._gun(pz))
        fark[pz] = ew["ml21"] - ew["ewma"]
    zler = {pz: m["z"] for pz, m in piyasa.items() if m["z"] is not None}
    ad, S = QKR.korelasyon_toplami({pz: fark[pz] for pz in zler}, min_ortak=int(KAPI["min_ortak"]))
    Z = round(sum(zler[a] for a in ad) / math.sqrt(S), 3) if len(ad) >= 2 and S > 0 else None
    olumlu = sum(1 for m in piyasa.values() if (m["ds"] or 0) > 0)
    izl = (piyasa.get(QK.IZLEME) or {}).get("ds")
    return {"piyasa": piyasa, "z": Z, "S": round(S, 3), "olumlu": olumlu, "n_piyasa": len(piyasa), "izleme_ds": izl,
            "mde": QKR.mde({pz: m["se"] for pz, m in piyasa.items() if pz in zler}, S),
            "gecti": bool(Z is not None and Z > KAPI["z"] and olumlu >= KAPI["min_piyasa"] and izl is not None and izl > 0)}


def h1_olcum(T: pd.DataFrame, rv: pd.DataFrame) -> dict:
    """QLIKE farkı (ml21 − ewma) — piyasa: günlük kesit ortalaması; izleme DM t'si NW gecikme ≥ 21."""
    from research import quant_ml_motor as QMM
    T = T[T.tarih <= H1_SON].merge(rv, on=["tarih", "piyasa", "sembol"], how="inner")
    T = T[np.isfinite(T.rv21)]
    T = T.assign(d=QM.qlike(T.f_ml21.to_numpy(float), T.rv21.to_numpy(float))
                 - QM.qlike(T.ewma_f.to_numpy(float), T.rv21.to_numpy(float)))
    T = T[np.isfinite(T.d)]
    piyasa: dict[str, dict] = {}
    for pz, g in T.groupby("piyasa", sort=False):
        gunluk = g.groupby("tarih").d.mean()
        piyasa[pz] = {"n_gun": int(len(gunluk)), "d_ort": round(float(gunluk.mean()), 5),
                      "dm_t": QMM.nw_t(gunluk.to_numpy(), NW_GECIKME)}
    olumlu = sum(1 for m in piyasa.values() if m["d_ort"] < 0)
    t = (piyasa.get(QK.IZLEME) or {}).get("dm_t")
    return {"piyasa": piyasa, "olumlu": olumlu, "n_piyasa": len(piyasa), "izleme_t": t,
            "gecti": bool(olumlu >= KAPI["min_piyasa"] and t is not None and t < -KAPI["z"])}


def rv_tablosu(P: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"tarih": P.index, "piyasa": P["piyasa"].to_numpy(), "sembol": P["sembol"].to_numpy(),
                         "rv21": P["rv21"].to_numpy(float)})


def hukum(h1: dict, h2: dict, bugun: pd.Timestamp) -> list[str]:
    if bugun < H1_KARAR:
        return [f"ARA RAPOR — ön-kayıt gereği hüküm YOK (H1 kararı {H1_KARAR.date()}, H2 kararı {H2_KARAR.date()})."]
    if not h1["gecti"]:
        return ["KARAR H1: GEÇMEDİ → σ_ML21 adaylıktan DÜŞER (öngörü üstünlüğü geleceğe taşınmadı); H2 yalnız bilgi."]
    if bugun < H2_KARAR:
        return ["KARAR H1: GEÇTİ (öngörü üstünlüğü sürüyor).",
                f"H2 ARA — hüküm {H2_KARAR.date()}'te."]
    return ["KARAR H1: GEÇTİ.", "KARAR H2: " + ("GEÇTİ → σ_ML21 canlı quant çekirdeğine ADAY (entegrasyon ayrı tur)"
                                                if h2["gecti"] else "GEÇMEDİ → öngörü sürüyor, karar değeri kanıtlanmadı")]


def rapor(R: pd.DataFrame, T: pd.DataFrame, rv: pd.DataFrame, bugun: pd.Timestamp) -> str:
    def f(x, nd=2):
        return "—" if x is None else f"{x:+.{nd}f}"
    if R.empty:
        return f"Defter boş: ilk ileri gün {KAYIT_GUNU.date()} sonrası ilk tamamlanan bar yazılınca başlar."
    gun = R.tarih.nunique()
    s = [f"İLERİ DÖNEM {R.tarih.min().date()} → {R.tarih.max().date()} · {gun} takvim günü · "
         f"{R.sembol.nunique()} sembol · kollar {'/'.join(KOLLAR)}"]
    h2 = h2_olcum(R)
    h1 = h1_olcum(T, rv)
    s.append(f"{'piyasa':<18}{'H1 ΔQLIKE':>11}{'DM t':>7}{'H2 ΔSharpe':>12}{'z':>7}")
    for pz in dict.fromkeys([*h1["piyasa"], *h2["piyasa"]]):
        a, b = h1["piyasa"].get(pz, {}), h2["piyasa"].get(pz, {})
        s.append(f"{pz:<18}{f(a.get('d_ort'), 4):>11}{f(a.get('dm_t')):>7}{f(b.get('ds'), 3):>12}{f(b.get('z')):>7}")
    s.append(f"H1: ΔQLIKE < 0 olan piyasa {h1['olumlu']}/{h1['n_piyasa']} · izleme DM t {f(h1['izleme_t'])} "
             f"(kapı ≥ {KAPI['min_piyasa']} ve t < −{KAPI['z']})")
    s.append(f"H2: birleşik z {f(h2['z'])} · ΔS > 0 piyasa {h2['olumlu']}/{h2['n_piyasa']} · izleme ΔS "
             f"{f(h2['izleme_ds'], 3)} · şu anki MDE {h2['mde']} Sharpe (kapı z > {KAPI['z']})")
    if gun < 60:
        s.append("(60 günden kısa: t/z değerleri anlamsız, yalnız betimsel)")
    return "\n".join(s + hukum(h1, h2, bugun))


# ── komutlar ───────────────────────────────────────────────────────────────────────────────────────
def _bugun() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)


def _semboller() -> list[str]:
    from research import piyasalar as P
    return sorted({s for _, s in P.tum_semboller()} | set(QK.izleme_listesi()) | {"^VIX"})


def kilitle() -> None:
    from research import piyasalar as P
    if KILIT.exists():
        raise SystemExit(f"kilit zaten var ({KILIT}); ön-kayıt YENİDEN kilitlenmez")
    if _bugun() > KAYIT_GUNU:
        raise SystemExit(f"kayıt günü geçti: kilit ancak {KAYIT_GUNU.date()} (UTC) içinde oluşturulabilir")
    eksik = indir(_semboller())
    if eksik:
        raise SystemExit(f"indirilemedi: {eksik} — kilit eksik veriyle kurulmaz")
    _veri_kaynak(VERI)
    evren = [[pz, s] for pz, s in QK.evren()[0]]
    d = model_egit(panel(evren), 2026)
    sha = model_kaydet(d)
    _model_kayda_ekle(d, sha)
    kanarya = kanarya_sembolleri(evren)
    KANARYA_VERI.mkdir(parents=True, exist_ok=True)
    for s in [s for _, s in kanarya] + ["^VIX"]:
        shutil.copy2(P._dosya(s, "1d"), KANARYA_VERI / P._dosya(s, "1d").name)
    modeller, _ = modeller_yukle()                  # diskten: pickle yolu da kanaryada sınanır
    kilit = {"kayit_gunu": str(KAYIT_GUNU.date()), "olusturma_utc": pd.Timestamp.now(tz="UTC").isoformat(),
             "onkayit": ONKAYIT.name, "onkayit_sha256": _sha(ONKAYIT), "evren": evren, "kanarya": kanarya,
             "model_2026_sha": sha, "model_2026_n": d["n"], "model_2026_c": d["c"],
             "kanarya_izi": kanarya_izi(kanarya, {2026: modeller[2026]}),
             "ortam": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                       "sklearn": __import__("sklearn").__version__}}
    KILIT.write_text(json.dumps(kilit, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"kilit yazıldı: {KILIT} · evren {len(evren)} · 2026 modeli {d['n']} satır · sha {sha[:16]}…")


def guncelle() -> int:
    kilit = json.loads(KILIT.read_text(encoding="utf-8"))
    evren = kilit["evren"]
    eksik = indir(sorted({s for _, s in evren} | {"^VIX"}))
    if eksik:
        print(f"uyarı: indirilemedi {eksik} — bu sembollerin yeni günleri sonraki güncellemede yazılır")
    _, msorun = modeller_yukle()
    sorun = dogrula(kilit, msorun)
    if sorun:
        print("KİLİT BOZUK: deftere yazılmadı\n  - " + "\n  - ".join(sorun))
        return 1
    _veri_kaynak(VERI)
    bugun = _bugun()
    P = panel(evren)
    modeller, _ = modeller_yukle()
    for y in range(2026, (bugun - pd.Timedelta(days=1)).year + 1):
        if y not in modeller:                        # ön-kayıt §3: Y modeli Y'deki ilk koşuda BİR KEZ eğitilir
            d = model_egit(P, y)
            _model_kayda_ekle(d, model_kaydet(d))
            modeller[y] = d
            print(f"{y} modeli eğitildi ve donduruldu ({d['n']} satır)")
    R, T = getiriler(evren, P, modeller, bugun)
    dR, dT = defter_oku(DEFTER_R), defter_oku(DEFTER_T)
    fark = (uyusmayan(R, dR, ["piyasa", "sembol", "kol"], ["getiri"])
            + uyusmayan(T, dT, ["piyasa", "sembol"], ["ewma_f", "f_ml21"]))
    yR, yT = yeni_satirlar(R, dR, ["piyasa", "sembol", "kol"]), yeni_satirlar(T, dT, ["piyasa", "sembol"])
    _ekle(DEFTER_R, yR)
    _ekle(DEFTER_T, yT)
    print(f"kilit doğrulandı · {yR.tarih.nunique()} yeni gün · revizyonla uyuşmayan yazılı satır {fark}")
    print(rapor(defter_oku(DEFTER_R), defter_oku(DEFTER_T), rv_tablosu(P), bugun))
    return 0


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    komut = sys.argv[1] if len(sys.argv) > 1 else "rapor"
    if komut == "kilitle":
        kilitle()
    elif komut == "guncelle":
        raise SystemExit(guncelle())
    else:
        _veri_kaynak(VERI)
        k = json.loads(KILIT.read_text(encoding="utf-8"))
        print(rapor(defter_oku(DEFTER_R), defter_oku(DEFTER_T), rv_tablosu(panel(k["evren"])), _bugun()))
