"""quant_ml_derin.py — Tur-100: ML oynaklık modelini (σ_ML21) derinleştirmek, öngörüden karara, ön-kayıtlı.

Tur-98/99: σ_ML21 (HistGradientBoosting, 8 öz-fiyat özelliği, hedef ln RV21, yıllık genişleyen eğitim) EWMA'yı
öngörüde iki pencerede de yendi; motorda Sharpe farkı küçük kaldı (birleşik Z 1,33 / 2,41). Tarif 2026-09-26'da
ileri sınamaya kilitlendi (docs/ONKAYIT_SIGMA_ML21_2026-09-26.md) — bu modül o tarife DOKUNMAZ; yalnız okur.

TEŞHİS (koşudan önce; kod ve YALNIZ 2011 öncesi eğitim verisi okunarak):
  D1 KAYIP HİZASI: model ln RV21'i karesel kayıpla öğrenir (koşullu GEOMETRİK ortalama) ve sabit c ile ölçeklenir;
     değerlendirme QLIKE'tır. Gamma sapması = 2·(QLIKE − ln y − 1) → loss="gamma" QLIKE'ı doğrudan küçültür.
     2011 öncesi eğitim kümesinde zamansal doğrulama QLIKE'ı: ln+c −7,723 (50 it.) · gamma −7,676 (100 it.) →
     ön-kayıtlı aday ln hedefini KORUR; gamma yalnız bilgi koludur.
  D2 SIZINTILI ERKEN DURDURMA: HistGBR varsayılanı early_stopping='auto' (n > 10.000 → açık) RASTGELE %10 doğrulama
     ayırır. Örtüşen 21 günlük hedefte komşu günler aynı hedefi taşır → doğrulama kaybı eğitimle birlikte düşer:
     kilitteki 2026 modeli n_iter_ = 300 (tavan; durdurma hiç tetiklenmedi) ve verinin %10'u sessizce atıldı.
     Aynı 2011 öncesi kümede zamansal doğrulama en iyi ≈ 50 iterasyon (300'de −7,710, 50'de −7,723).
  D3 DAR BİLGİ: özellikler yalnız enstrümanın kendi fiyatı. Oynaklık literatürünün iki güçlü ek bilgisi eksik:
     örtük oynaklık (VIX, vade yapısı) ve ORTAK oynaklık faktörü (piyasa ve küresel kesit).
  D4 ADAPTASYON ÖLÇÜLMEDİ: "yıllık genişleyen pencere" politikası hiç sınanmadı ("ML zamanla daha iyi olmaz mı?").

ÖN-KAYIT (2026-09-27, bu modülün HİÇBİR test-yılı çıktısı görülmeden sabitlendi):
  Ortak: Tur-98 paneli (`quant_ml_motor.tum_paneller`: 14 piyasa havuzu + ABD_izleme, bayat bar süzgeçli), hedef RV21,
  eğitim YALNIZ havuz satırları, etiketi dönem başından önce bilinen satırlar (`egitim_maskesi` formülü), yıllık
  yürüyen 2011-2026, pencereler Tur-98/99 ile aynı (2011-17, 2018-26). Satır kümesi tüm kollarda aynıdır (B_OZ sonlu).
  ÖNGÖRÜ KOLLARI (merdiven; her basamak tek değişiklik):
    ml21 — Tur-98 tarifi BİREBİR (parite: Tur-98 panelindeki f_ml21 ile göreli fark < 1e-9).
    k1   — ml21 + ZAMANSAL erken durdurma (D2): eğitim kümesinin son 365 günü doğrulama, öncesinde 31 gün arınma;
           ≤ 600 iterasyon içinde doğrulama QLIKE'ını (c eğitim alt örnekleminden) en küçükleyen sayı seçilir, sonra
           TÜM eğitim kümesiyle early_stopping=False ve o sayıyla yeniden eğitilir.
    k2   — k1 + loss="gamma", hedef RV21, c = 1 (D1; BİLGİ kolu).
    k3   — k1 + 5 yeni bilgi özelliği (D3) = ADAY. Hepsi satır tarihinden KESİN ÖNCEKİ son değerdir (aynı gün yasak:
           saat dilimi sızıntısı): pz_rv5 / pz_rv22 (aynı piyasanın önceki günkü kesit ort. ln RV5 / ln RV22),
           gl_rv22 (14 havuz piyasasının piyasa ortalamalarının eşit ağırlıklı ortalaması), ln_vix (ln (VIX/100)²/252),
           vix_ts (ln VIX/VIX3M). Eksik değer NaN kalır (HistGBR yerel işler); satır kümesi değişmez.
  G1 ÖNGÖRÜ KAPISI (k3 − ml21, İKİ pencerede de): 21 günlük QLIKE'ı düşük piyasa ≥ 10 (15'te) VE ABD_izleme DM t'si
     (NW gecikme ≥ 21) < −1,645.
  G2 KARAR KAPISI (motor, k3 − EWMA, İKİ pencerede de; ileri sınamanın H2 kapısı): korelasyon düzeltmeli birleşik Z
     > 1,645 VE ΔSharpe > 0 olan piyasa ≥ 10 VE ABD_izleme ΔSharpe > 0. Motor: `backtest_symbol` 1d quant_trend,
     σ yalnız girişte (Tur-98 `yama`). Bilgi: k3 − ml21, k1 − ml21 ve ml21 − EWMA aynı ölçüyle.
  HÜKÜM: G1 ve G2 geçerse k3 İLERİ SINAMA ADAYIdır (ek ön-kayıtla ikinci aday kol; canlı DEĞİŞMEZ — iki tarihsel
     pencere de Tur-97/98/99'da görüldü). G1 geçip G2 geçmezse: öngörü iyileşti, karar değeri kanıtlanmadı.
     G1 geçmezse: tarif değişmez.
  ADAPTASYON (ml21 tarifiyle, ileri sınamanın politikası sınanır; kapı G1 ile aynı, ml21'e karşı):
    ceyrek   — çeyreklik yeniden eğitim (genişleyen);   yuv2 / yuv5 — yalnız son 2 / 5 yılla eğitim (yuvarlanan);
    donmus11 / donmus18 — 2011 / 2018 başında BİR KEZ eğitilip hiç güncellenmeyen model (bayatlama eğrisi, yıl yıl);
    havuz25 / havuz50 — havuz enstrümanlarının rastgele %25 / %50'siyle eğitim (tohum 0; "daha çok veri" eğrisi).
  Bilgi: k3'ün yeni özelliklerinin permütasyon önemi (test yılında sütun karıştırılınca QLIKE artışı).

TEŞHİS (SONRADAN, kapısız; ön-kayıtlı koşu GÖRÜLDÜKTEN sonra eklendi, hüküm vermez — Tur-23):
  İlk koşuda k1 ve k3 ml21'den KÖTÜ çıktı (15 piyasanın 3/1 ve 0/1'inde iyi). Merdiven D3'ü D2 onarımına bağlamıştı;
  bu kollar etkileri ayırır: t_tam300 (ml21, erken durdurma KAPALI, 300 it., verinin %100'ü → D2'nin %10 kaybı) ·
  t_n100 / t_n600 (aynı, sabit 100 / 600 it. → iterasyon duyarlılığı; test yıllarında ölçülür, SEÇİM İÇİN DEĞİL) ·
  t_ml21_bilgi (ml21 tarifi + 5 bilgi özelliği → D3 iyi tarifin üstünde).

CLI:  python -m research.quant_ml_derin --asama model            # panel + tüm kollar (~20 dk)
      python -m research.quant_ml_derin --asama motor --isci 8   # motor: ewma, ml21, k1, k3
      python -m research.quant_ml_derin --asama teshis           # sonradan teşhis kolları (~5 dk), panele eklenir
      python -m research.quant_ml_derin --asama rapor            # data/quant_ml_derin_karnesi.json
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research import coklu_piyasa as C
from research import quant_karar as QKR
from research import quant_kayma as QK
from research import quant_ml as QM
from research import quant_ml_motor as QMM

ROOT = Path(__file__).resolve().parents[1]
SONUC_DIR = C.SONUC_DIR / "quant100"
KARNE = ROOT / "data" / "quant_ml_derin_karnesi.json"
ILERI = QM.ILERI
GBM = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
           random_state=42)                                     # Tur-98 _gbm() birebir
ES = {"dogrulama_gun": 365, "arinma_gun": 31, "max_iter": 600, "c_orneklem": 50_000}
YENI_OZ = ["pz_rv5", "pz_rv22", "gl_rv22", "ln_vix", "vix_ts"]
K3_OZ = QM.B_OZ + YENI_OZ
ONGORU_KOLLARI = {"ml21": (QM.B_OZ, "ml21"), "k1": (QM.B_OZ, "k1"), "k2": (QM.B_OZ, "k2"), "k3": (K3_OZ, "k1")}
ADAPTASYON: dict[str, dict[str, Any]] = {"ceyrek": {"donem": "ceyrek"}, "yuv2": {"geriye_yil": 2}, "yuv5": {"geriye_yil": 5},
              "donmus11": {"donem": ("donmus", 2011)}, "donmus18": {"donem": ("donmus", 2018)},
              "havuz25": {"havuz_payi": 0.25}, "havuz50": {"havuz_payi": 0.5}}
MOTOR_KOLLARI = ("ewma", "ml21", "k1", "k3")
TESHIS_KOLLARI: dict[str, dict[str, Any]] = {
    "t_tam300": {"oz": QM.B_OZ, "tarif": "sabit", "n_sabit": 300}, "t_n100": {"oz": QM.B_OZ, "tarif": "sabit", "n_sabit": 100},
    "t_n600": {"oz": QM.B_OZ, "tarif": "sabit", "n_sabit": 600}, "t_ml21_bilgi": {"oz": K3_OZ, "tarif": "ml21"}}
KAPI = {"min_piyasa": 10, "z": 1.645}


# ---------------------------------------------------------------------------------------------------
# Yeni bilgi özellikleri (D3) — hepsi KESİN ÖNCEKİ tarihten
# ---------------------------------------------------------------------------------------------------
def once(seri: pd.Series, tarih: pd.DatetimeIndex) -> np.ndarray:
    """`seri`nin her tarihten KESİN ÖNCEKİ son değeri; öncesi yoksa NaN (aynı gün değer ASLA kullanılmaz)."""
    s = seri.dropna().sort_index()
    i = s.index.searchsorted(tarih, side="left") - 1
    v = s.to_numpy(float)
    return np.where(i >= 0, v[np.clip(i, 0, None)] if v.size else np.nan, np.nan)


def bilgi_ekle(P: pd.DataFrame, vix: pd.Series, vix3m: pd.Series) -> pd.DataFrame:
    P = P.copy()
    for k in ("pz_rv5", "pz_rv22"):
        P[k] = np.nan
    for _pz, idx in P.groupby("piyasa", sort=False).indices.items():
        g = P.iloc[idx]
        for k, src in (("pz_rv5", "ln_rv5"), ("pz_rv22", "ln_rv22")):
            P.iloc[idx, P.columns.get_loc(k)] = once(g.groupby(g.index)[src].mean(), g.index)
    hv = P[P["piyasa"] != QK.IZLEME]
    gl = hv.groupby(["piyasa", hv.index])["ln_rv22"].mean().groupby(level=1).mean()
    P["gl_rv22"] = once(gl, P.index)
    P["ln_vix"] = once(np.log((vix / 100.0) ** 2 / 252.0), P.index)
    P["vix_ts"] = once(np.log(vix / vix3m.reindex(vix.index)), P.index)
    return P


# ---------------------------------------------------------------------------------------------------
# Eğitim tarifleri
# ---------------------------------------------------------------------------------------------------
def sinir_maskesi(tarih: pd.DatetimeIndex, bas: pd.Timestamp, etiket_gun: int) -> np.ndarray:
    """`quant_ml.egitim_maskesi` formülü, herhangi bir dönem başı için (yıl başında birebir aynı)."""
    return np.asarray(tarih < bas - pd.Timedelta(days=int(etiket_gun * 7 / 5) + 3))


def _model(tarif: str, n_iter: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    if tarif == "ml21":
        return HistGradientBoostingRegressor(**GBM)            # early_stopping='auto' DAHİL (kilitli tarif)
    kw: dict[str, Any] = {**GBM, "max_iter": int(n_iter), "early_stopping": False}
    if tarif == "k2":
        kw["loss"] = "gamma"
    return HistGradientBoostingRegressor(**kw)


def _ln(tarif: str) -> bool:
    return tarif != "k2"


def es_bolme(tarih: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Zamansal doğrulama bölmesi: son `dogrulama_gun` doğrulama; eğitim ondan `arinma_gun` önce biter (örtüşen
    21 günlük hedef doğrulama dönemine taşmaz)."""
    vb = tarih.max() - pd.Timedelta(days=ES["dogrulama_gun"])
    return np.asarray(tarih < vb - pd.Timedelta(days=ES["arinma_gun"])), np.asarray(tarih >= vb)


def zaman_es(X: np.ndarray, rv: np.ndarray, tarih: pd.DatetimeIndex, tarif: str) -> tuple[int, list[float]]:
    """D2 onarımı: `es_bolme` doğrulamasında QLIKE'ı en küçükleyen iterasyon sayısı. ln tarifinde c her
    aşamada eğitim alt örnekleminden."""
    tr, va = es_bolme(tarih)
    y = np.log(rv + QM.EPS) if _ln(tarif) else rv
    m = _model(tarif, ES["max_iter"]).fit(X[tr], y[tr])
    alt = np.random.default_rng(0).choice(np.flatnonzero(tr), size=min(ES["c_orneklem"], int(tr.sum())), replace=False)
    kayip = []
    for zt, zv in zip(m.staged_predict(X[alt]), m.staged_predict(X[va]), strict=True):
        f = np.exp(zv) * (rv[alt].mean() / np.exp(zt).mean()) if _ln(tarif) else zv
        kayip.append(float(np.mean(np.log(f) + rv[va] / f)))
    return int(np.argmin(kayip)) + 1, kayip


def egit(X: np.ndarray, rv: np.ndarray, tarih: pd.DatetimeIndex, tarif: str, n_sabit: int | None = None):
    """→ (model, c, n_iter). ml21: Tur-98 birebir; k1/k2: zamansal erken durdurma + tüm kümeyle yeniden eğitim;
    sabit (teşhis): ln hedef, erken durdurma kapalı, `n_sabit` iterasyon."""
    y = np.log(rv + QM.EPS) if _ln(tarif) else rv
    if tarif == "ml21":
        m = _model("ml21").fit(X, y)
    elif tarif == "sabit":
        m = _model("k1", int(n_sabit or 0)).fit(X, y)
    else:
        m = _model(tarif, zaman_es(X, rv, tarih, tarif)[0]).fit(X, y)
    c = float(np.mean(rv) / np.mean(np.exp(m.predict(X)))) if _ln(tarif) else 1.0
    return m, c, int(m.n_iter_)


def tahmin(m, c: float, X: np.ndarray, tarif: str) -> np.ndarray:
    return np.exp(m.predict(X)) * c if _ln(tarif) else m.predict(X)


def donemler(donem) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if donem == "yil":
        return [(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")) for y in QM.TEST_YILLARI]
    if donem == "ceyrek":
        b = pd.date_range("2011-01-01", f"{QM.TEST_YILLARI[-1]}-10-01", freq="QS")
        return [(x, x + pd.DateOffset(months=3)) for x in b]
    _, y0 = donem                                                # ("donmus", yıl): tek model, sonuna dek
    return [(pd.Timestamp(f"{y0}-01-01"), pd.Timestamp(f"{QM.TEST_YILLARI[-1] + 1}-01-01"))]


def havuz_alt_kumesi(P: pd.DataFrame, havuz: np.ndarray, pay: float, tohum: int = 0) -> np.ndarray:
    anahtar = P["piyasa"].astype(str) + "|" + P["sembol"].astype(str)
    hepsi = sorted(set(anahtar[havuz]))
    sec = set(np.random.default_rng(tohum).choice(hepsi, size=max(1, round(pay * len(hepsi))), replace=False))
    return anahtar.isin(sec).to_numpy()


def yuruyen(P: pd.DataFrame, havuz: np.ndarray, oz: list[str], tarif: str, donem="yil",
            geriye_yil: int | None = None, havuz_payi: float | None = None, onem: bool = False,
            n_sabit: int | None = None) -> tuple[np.ndarray, dict]:
    """Yürüyen öngörü; satır kümesi B_OZ-sonlu satırlar (tüm kollarda aynı). → (f, dönem bilgisi)."""
    ok = np.isfinite(P[QM.B_OZ].to_numpy(float)).all(axis=1)
    rv = P["rv21"].to_numpy(float)
    X = P[oz].to_numpy(float)
    egit_m = havuz & ok & np.isfinite(rv)
    if havuz_payi is not None:
        egit_m &= havuz_alt_kumesi(P, havuz, havuz_payi)
    f = np.full(len(P), np.nan)
    bilgi: dict = {}
    for bas, son in donemler(donem):
        tr = egit_m & sinir_maskesi(P.index, bas, ILERI + 1)
        if geriye_yil:
            tr &= np.asarray(P.index >= bas - pd.DateOffset(years=geriye_yil))
        te = ok & np.asarray((P.index >= bas) & (P.index < son))
        if tr.sum() < 5000 or not te.any():
            continue
        m, c, n = egit(X[tr], rv[tr], P.index[tr], tarif, n_sabit)
        f[te] = tahmin(m, c, X[te], tarif)
        b = {"n_egitim": int(tr.sum()), "n_iter": n, "c": round(c, 4)}
        if onem:
            b["onem"] = permutasyon_onemi(m, c, X[te], rv[te], tarif, oz)
        bilgi[str(bas.date())] = b
    return f, bilgi


def permutasyon_onemi(m, c: float, X: np.ndarray, rv: np.ndarray, tarif: str, oz: list[str],
                      n: int = 20_000) -> dict[str, float]:
    """Test döneminde sütun karıştırılınca ortalama QLIKE artışı (büyük = model o bilgiye dayanıyor)."""
    ok = np.isfinite(rv)
    X, rv = X[ok], rv[ok]
    rng = np.random.default_rng(0)
    if len(rv) > n:
        i = rng.choice(len(rv), size=n, replace=False)
        X, rv = X[i], rv[i]
    taban = float(np.mean(QM.qlike(tahmin(m, c, X, tarif), rv)))
    out = {}
    for j, ad in enumerate(oz):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        out[ad] = round(float(np.mean(QM.qlike(tahmin(m, c, Xp, tarif), rv))) - taban, 5)
    return out


# ---------------------------------------------------------------------------------------------------
# Model aşaması
# ---------------------------------------------------------------------------------------------------
def model_asamasi() -> None:
    from research import piyasalar as PY
    t = time.time()
    P = QMM.tum_paneller()
    P = bilgi_ekle(P, PY.yukle("^VIX")["close"], PY.yukle("^VIX3M")["close"])
    havuz = (P["piyasa"] != QK.IZLEME).to_numpy()
    print(f"panel {len(P)} satır, {P['sembol'].nunique()} enstrüman ({round(time.time() - t)} sn)", flush=True)
    bilgi: dict = {"yeni_oz_dolu": {k: round(float(P[k].notna().mean()), 4) for k in YENI_OZ}}
    for ad, (oz, tarif) in ONGORU_KOLLARI.items():
        P[f"f_{ad}"], bilgi[ad] = yuruyen(P, havuz, oz, tarif, onem=ad == "k3")
        print(f"{ad} ({round(time.time() - t)} sn)", flush=True)
    for ad, kw in ADAPTASYON.items():
        P[f"f_{ad}"], bilgi[ad] = yuruyen(P, havuz, QM.B_OZ, "ml21", **kw)
        print(f"{ad} ({round(time.time() - t)} sn)", flush=True)
    bilgi["parite_ml21"] = parite_ml21(P)
    SONUC_DIR.mkdir(parents=True, exist_ok=True)
    # HAR21 (bilgi): Tur-98 panelinden, yalnız satırlar birebir eşleşiyorsa
    P["f_har21"] = pd.read_pickle(QMM.SONUC_DIR / "panel98.pkl")["f_har21"].to_numpy()         if bilgi["parite_ml21"]["esit_satir"] else np.nan
    sut = ["piyasa", "sembol", "rv21", "ewma_f"] + [c for c in P.columns if c.startswith("f_")]
    P[sut].to_pickle(SONUC_DIR / "panel100.pkl")
    seriler = {k: g[[f"f_{a}" for a in MOTOR_KOLLARI if a != "ewma"]].copy()
               for k, g in P.groupby(["piyasa", "sembol"], sort=False)}
    with open(SONUC_DIR / "seriler100.pkl", "wb") as f:
        pickle.dump(seriler, f)
    (SONUC_DIR / "model100.json").write_text(json.dumps(bilgi, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"parite ml21: {bilgi['parite_ml21']} · toplam {round(time.time() - t)} sn", flush=True)


def teshis_asamasi() -> None:
    """Sonradan teşhis kolları: paneli yeniden kurar, kolları eğitir, panel100.pkl'e sütun olarak ekler."""
    from research import piyasalar as PY
    t = time.time()
    P = bilgi_ekle(QMM.tum_paneller(), PY.yukle("^VIX")["close"], PY.yukle("^VIX3M")["close"])
    havuz = (P["piyasa"] != QK.IZLEME).to_numpy()
    P100 = pd.read_pickle(SONUC_DIR / "panel100.pkl")
    if len(P100) != len(P) or not (P100["sembol"].to_numpy() == P["sembol"].to_numpy()).all():
        raise RuntimeError("panel100 satırları yeniden kurulan panelle eşleşmiyor")
    bilgi = json.loads((SONUC_DIR / "model100.json").read_text(encoding="utf-8"))
    for ad, kw in TESHIS_KOLLARI.items():
        P100[f"f_{ad}"], bilgi[ad] = yuruyen(P, havuz, kw["oz"], kw["tarif"], n_sabit=kw.get("n_sabit"))
        print(f"{ad} ({round(time.time() - t)} sn)", flush=True)
    P100.to_pickle(SONUC_DIR / "panel100.pkl")
    (SONUC_DIR / "model100.json").write_text(json.dumps(bilgi, ensure_ascii=False, indent=1), encoding="utf-8")


def parite_ml21(P: pd.DataFrame) -> dict:
    """Tur-98 panelinin f_ml21'i ile birebirlik (aynı satır sırası ve tarif)."""
    P98 = pd.read_pickle(QMM.SONUC_DIR / "panel98.pkl")
    if len(P98) != len(P) or not (P98["sembol"].to_numpy() == P["sembol"].to_numpy()).all():
        return {"esit_satir": False}
    a, b = P["f_ml21"].to_numpy(float), P98["f_ml21"].to_numpy(float)
    ikisi = np.isfinite(a) & np.isfinite(b)
    return {"esit_satir": True, "nan_uyumu": bool((np.isfinite(a) == np.isfinite(b)).all()),
            "max_goreli": float(np.max(np.abs(a[ikisi] / b[ikisi] - 1))) if ikisi.any() else None}


# ---------------------------------------------------------------------------------------------------
# Motor aşaması
# ---------------------------------------------------------------------------------------------------
_SERI: dict = {"v": None}


def _seriler() -> dict:
    if _SERI["v"] is None:
        with open(SONUC_DIR / "seriler100.pkl", "rb") as f:
            _SERI["v"] = pickle.load(f)
    return _SERI["v"]


def _m100(piyasa: str, sym: str) -> Path:
    return SONUC_DIR / QK.dosya(piyasa, sym, "m100_").name


def enstruman_kos(arg: tuple[str, str]) -> str:
    piyasa, sym = arg
    C._isci_hazirla()
    t = time.time()
    df = C.hazirla(sym)
    s = _seriler()[(piyasa, sym)]
    out: dict = {"piyasa": piyasa, "sembol": sym, "kos": {}}
    for pen, (bas, son) in QK.SIGMA_PENCERE.items():
        d = df if son is None else df[df.index < pd.Timestamp(son)]
        t0 = int((d.index < pd.Timestamp(bas)).sum())
        if len(d) - t0 < 250 or t0 < 60:
            continue
        for kol in MOTOR_KOLLARI:
            if kol == "ewma":
                out["kos"][f"{kol}_{pen}"] = QK._kos(sym, d, t0, "quant_trend")
                continue
            with QMM.yama(sigma=np.sqrt(s[f"f_{kol}"])):
                out["kos"][f"{kol}_{pen}"] = QK._kos(sym, d, t0, "quant_trend")
    with open(_m100(piyasa, sym), "wb") as f:
        pickle.dump(out, f)
    return f"{piyasa}/{sym}: {len(out['kos'])} kol, {round(time.time() - t, 1)} sn"


# ---------------------------------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------------------------------
def _pencere(P: pd.DataFrame, pen: str) -> pd.DataFrame:
    bas, son = QK.SIGMA_PENCERE[pen]
    return P[(P.index >= pd.Timestamp(bas)) & ((P.index < pd.Timestamp(son)) if son else True)]


def ongoru_farki(P: pd.DataFrame, a: str, b: str) -> dict:
    """QLIKE(a) − QLIKE(b), 21 günlük hedef; piyasa = günlük kesit ortalaması; aynı satırlar (ikisi de sonlu).
    `n_sonlu_disi`: iki öngörü de VAR ama kayıp sonlu değil (taşma/≤0) — bu satırlar atılır, sayısı görünür kalır."""
    rv = P["rv21"].to_numpy(float)
    fa, fb = P[a].to_numpy(float), P[b].to_numpy(float)
    d = QM.qlike(fa, rv) - QM.qlike(fb, rv)
    n_sonlu_disi = int((np.isfinite(rv) & ~np.isnan(fa) & ~np.isnan(fb) & ~np.isfinite(d)).sum())
    D = pd.DataFrame({"piyasa": P["piyasa"].to_numpy(), "d": d}, index=P.index)
    D = D[np.isfinite(D["d"])]
    piyasa: dict = {}
    for pz, g in D.groupby("piyasa", sort=False):
        gunluk = g.groupby(level=0)["d"].mean()
        piyasa[pz] = {"n_gun": int(len(gunluk)), "d_ort": round(float(gunluk.mean()), 5),
                      "dm_t": QMM.nw_t(gunluk.to_numpy(), ILERI)}
    olumlu = sum(1 for m in piyasa.values() if m["d_ort"] < 0)
    t = (piyasa.get(QK.IZLEME) or {}).get("dm_t")
    return {"piyasa": piyasa, "olumlu": olumlu, "n_piyasa": len(piyasa), "izleme_t": t, "n_sonlu_disi": n_sonlu_disi,
            "ort_d": round(float(np.mean([m["d_ort"] for m in piyasa.values()])), 5) if piyasa else None,
            "gecti": bool(olumlu >= KAPI["min_piyasa"] and t is not None and t < -KAPI["z"])}


def bayatlama(P: pd.DataFrame, kol: str, taban: str = "f_ml21") -> dict:
    """Yıl yıl QLIKE cezası (kol − taban): piyasa günlük ortalamalarının ortalaması + ABD_izleme."""
    rv = P["rv21"].to_numpy(float)
    D = pd.DataFrame({"piyasa": P["piyasa"].to_numpy(),
                      "d": QM.qlike(P[kol].to_numpy(float), rv) - QM.qlike(P[taban].to_numpy(float), rv)}, index=P.index)
    D = D[np.isfinite(D["d"])]
    out: dict = {}
    for y, g in D.groupby(D.index.year):
        pz = g.groupby("piyasa")["d"].mean()
        out[int(y)] = {"ort": round(float(pz.mean()), 5), "izleme": round(float(pz.get(QK.IZLEME, np.nan)), 5)}
    return out


def karar_olcumu(ew: dict[str, dict[str, pd.Series]], a: str, b: str, pen: str) -> dict:
    """Motor: piyasa ΔSharpe (a − b) + Memmel z; korelasyon düzeltmeli birleşik Z; MDE; G2 koşulları."""
    piyasa: dict = {}
    fark: dict = {}
    for pz, kol in ew.items():
        x, y = kol.get(f"{a}_{pen}"), kol.get(f"{b}_{pen}")
        if x is None or y is None:
            continue
        piyasa[pz] = QKR.sharpe_fark(x, y, QK._gun(pz))
        ortak = x.index.intersection(y.index)
        fark[pz] = x.loc[ortak] - y.loc[ortak]
    bz = QKR.birlesik_z({pz: m["z"] for pz, m in piyasa.items()}, fark)
    olumlu = sum(1 for m in piyasa.values() if (m["ds"] or 0) > 0)
    izl = (piyasa.get(QK.IZLEME) or {}).get("ds")
    return {"piyasa": piyasa, "z": bz.get("z"), "etkin_n": bz.get("etkin_n"), "olumlu": olumlu,
            "n_piyasa": len(piyasa), "izleme_ds": izl,
            "mde": QKR.mde({pz: m["se"] for pz, m in piyasa.items() if m["z"] is not None}, bz["S"]) if bz.get("S") else None,
            "gecti": bool(bz.get("z") is not None and bz["z"] > KAPI["z"] and olumlu >= KAPI["min_piyasa"]
                          and izl is not None and izl > 0)}


def motor_serileri(kull: list[tuple[str, str]]) -> tuple[dict, dict]:
    """→ ({piyasa: {kol_pencere: EW seri}}, parite: bu turun ewma/ml21 kolları ↔ Tur-98 m98 dosyaları)."""
    toplu: dict[str, dict[str, list[pd.Series]]] = {}
    fark_max = {"ewma": 0.0, "ml21": 0.0}
    n_kiyas = n_indeks_farkli = 0
    for pz, s in kull:
        p = _m100(pz, s)
        if not p.exists():
            continue
        with open(p, "rb") as f:
            q = pickle.load(f)
        for ad, k in q["kos"].items():
            toplu.setdefault(pz, {}).setdefault(ad, []).append(k["s"])
        p98 = QMM._m98(pz, s)
        if p98.exists():
            with open(p98, "rb") as f:
                q98 = pickle.load(f)
            for kol in fark_max:
                for pen in QK.SIGMA_PENCERE:
                    a, b = q["kos"].get(f"{kol}_{pen}"), q98["kos"].get(f"{kol}_{pen}")
                    if a is None or b is None:
                        continue
                    n_kiyas += 1
                    if not a["s"].index.equals(b["s"].index):
                        n_indeks_farkli += 1
                        continue
                    fark_max[kol] = max(fark_max[kol], float(np.max(np.abs(a["s"].to_numpy() - b["s"].to_numpy()))))
    ew = {pz: {ad: C._ew(v) for ad, v in kol.items()} for pz, kol in toplu.items()}
    return ew, {"max_mutlak_fark": fark_max, "n_kiyas": n_kiyas, "n_indeks_farkli": n_indeks_farkli}


def hukum(karne: dict) -> dict:
    g1 = all(karne["ongoru"][pen]["k3_ml21"]["gecti"] for pen in QK.SIGMA_PENCERE)
    g2 = all(karne["karar"][pen]["k3_ewma"]["gecti"] for pen in QK.SIGMA_PENCERE)
    if g1 and g2:
        sonuc = "k3 İLERİ SINAMA ADAYI (canlı değişmez)"
    elif g1:
        sonuc = "öngörü iyileşti, karar değeri kanıtlanmadı"
    else:
        sonuc = "tarif değişmez"
    adapt = {ad: all(karne["ongoru"][pen][f"{ad}_ml21"]["gecti"] for pen in QK.SIGMA_PENCERE
                     if f"{ad}_ml21" in karne["ongoru"][pen]) for ad in ADAPTASYON}
    return {"G1": g1, "G2": g2, "sonuc": sonuc, "adaptasyon_gecen": [k for k, v in adapt.items() if v]}


def rapor(kull: list[tuple[str, str]]) -> dict:
    P = pd.read_pickle(SONUC_DIR / "panel100.pkl")
    karne: dict = {"uretim": time.strftime("%Y-%m-%d %H:%M"), "pencereler": QK.SIGMA_PENCERE, "kapi": KAPI, "es": ES,
                   "yeni_oz": YENI_OZ, "model": json.loads((SONUC_DIR / "model100.json").read_text(encoding="utf-8")),
                   "ongoru": {}, "karar": {}}
    ciftler = [("ml21", "ewma_f"), ("k1", "f_ml21"), ("k2", "f_k1"), ("k3", "f_ml21"), ("k3", "f_k1"), ("k3", "ewma_f"),
               ("har21", "f_ml21")] + [(ad, "f_ml21") for ad in ADAPTASYON] + [(ad, "f_ml21") for ad in TESHIS_KOLLARI]
    for pen in QK.SIGMA_PENCERE:
        Pp = _pencere(P, pen)
        blok = {}
        for a, b in ciftler:
            if f"f_{a}" not in Pp or Pp[f"f_{a}"].notna().mean() < 0.5:
                continue                                          # kol bu pencerede yok (donmus18 → 2011-17)
            blok[f"{a}_{b.removeprefix('f_').removesuffix('_f')}"] = ongoru_farki(Pp, f"f_{a}", b)
        karne["ongoru"][pen] = blok
    karne["bayatlama"] = {k: bayatlama(P, f"f_{k}") for k in ("donmus11", "donmus18", "ceyrek", "yuv2")}
    ew, parite = motor_serileri(kull)
    karne["motor_parite"] = parite
    for pen in QK.SIGMA_PENCERE:
        karne["karar"][pen] = {f"{a}_{b}": karar_olcumu(ew, a, b, pen)
                               for a, b in (("k3", "ewma"), ("ml21", "ewma"), ("k1", "ewma"), ("k3", "ml21"), ("k1", "ml21"))}
    karne["hukum"] = hukum(karne)
    karne["parmak_izi"] = C.girdi_parmak_izi(sorted({s for _, s in kull}))
    return karne


def ozet_md(karne: dict) -> str:
    def f(x, nd=2):
        return "—" if x is None else f"{x:.{nd}f}"
    s = []
    for pen in QK.SIGMA_PENCERE:
        s += [f"### Öngörü — {pen} (QLIKE farkı; olumlu = farkı < 0 olan piyasa)", "",
              "| kıyas | ort. fark | olumlu/n | izleme DM t | sonlu dışı | kapı |", "|---|---|---|---|---|---|"]
        for ad, m in karne["ongoru"][pen].items():
            s.append(f"| {ad} | {f(m['ort_d'], 4)} | {m['olumlu']}/{m['n_piyasa']} | {f(m['izleme_t'])} | "
                     f"{m['n_sonlu_disi']} | {m['gecti']} |")
        s += ["", f"### Karar (motor) — {pen}", "",
              "| kıyas | birleşik Z | ΔS>0 piyasa | izleme ΔS | MDE | kapı |", "|---|---|---|---|---|---|"]
        for ad, m in karne["karar"][pen].items():
            s.append(f"| {ad} | {f(m['z'])} | {m['olumlu']}/{m['n_piyasa']} | {f(m['izleme_ds'], 3)} | "
                     f"{f(m['mde'], 3)} | {m['gecti']} |")
        s.append("")
    s.append(f"Bayatlama (donmus11 − ml21, yıl → ort/izleme): {karne['bayatlama']['donmus11']}")
    s.append(f"Motor paritesi (Tur-98 m98): {karne['motor_parite']}")
    s.append(f"HÜKÜM: {karne['hukum']}")
    return "\n".join(s)


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Tur-100 ML oynaklık modeli derinleştirme")
    ap.add_argument("--asama", required=True, choices=["model", "motor", "teshis", "rapor"])
    ap.add_argument("--isci", type=int, default=8)
    a = ap.parse_args()
    C._isci_hazirla()
    kull, _ = QK.evren()
    if a.asama == "model":
        model_asamasi()
    elif a.asama == "motor":
        C._havuz(enstruman_kos, kull, a.isci)
    elif a.asama == "teshis":
        teshis_asamasi()
    else:
        karne = rapor(kull)
        KARNE.write_text(json.dumps(karne, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(ozet_md(karne))
        print(f"karne → {KARNE}")


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    _cli()
