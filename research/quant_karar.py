"""quant_karar.py — Tur-99: ML'in KARAR mekanizması — öngörüyü tüketen boyut kararı + birleşik kapı.

Tur-98 teşhisi (BULGU-98-2/3): σ_ML21 21 günlük öngörüde EWMA'yı canlı evrende DM t −8 ile yendi; motorda
Sharpe farkı +0,05 (z 0,8). Neden: `quant_trend` σ'yı YALNIZ GİRİŞTE okur — adet (risk $ / 3σ) ve stop
mesafesi pozisyonun ömrü boyunca sabittir. Pozisyon haftalar-aylar sürer, oynaklık bu sürede değişir ve
kararın hiçbir parçası öngörüyü yeniden okumaz: öngörünün kalitesi karara yalnız giriş gününde dokunur.
İkinci kusur kapıdaydı: tek evrenin z'si (ABD_izleme) ≈ +0,10'dan küçük Sharpe farkını göremiyordu; 15
piyasanın kanıtı ise hiç birleştirilmiyordu.

ÖN-KAYIT (2026-09-26; aşağıdaki hipotezlerin hiçbir çıktısı görülmeden sabitlendi):
  Laboratuvar: Tur-98 motor kolları (`backtest_symbol` 1d, kill-switch/AI kapısı kapalı, pencereler
  2011-17 ve 2018-26, 14 piyasa + ABD_izleme) koşulur ve İŞLEM DEFTERİ yakalanır (giriş barı, dolum, adet,
  giriş özkaynağı, çıkış barı, çıkış fiyatı). Günlük getiri defterden yeniden kurulur. ÖN-KOŞUL: SABİT
  replika motorun MTM özkaynak eğrisini vermeli (her enstrümanda günlük getiri farkı ≤ 1e-8); tutmazsa HÜKÜM
  YOK.
  TEK DEĞİŞKEN — VY (oynaklık yönetimli adet): pozisyon açıkken her açılışta hedef adet
  u* = u_giriş · σ̂_giriş / σ̂_{t−1} (σ̂: o kolun σ serisi, t−1 kapanışında bilinir); |u*/u − 1| > BANT (0,25)
  ise o açılışta u*'a geçilir ve el değiştiren nominalin C.maliyet_orani()'sı (%0,1) maliyet yazılır;
  nominal/özkaynak ≤ F_TAVAN = 2,0 (motorun alım gücü tavanı). Çıkış günü yeniden boyutlama YOK. Giriş/çıkış
  günleri, dolum/çıkış fiyatları ve stop DEĞİŞMEZ (motorun kararı); sonraki girişin adedi kolun
  özkaynağıyla orantılı ölçeklenir (motorun risk-tabanlı boyutu gibi). Fiyat sürüklenmesi yeniden
  dengelenmez: tek değişen, oynaklık oranıdır.
  H3a (mekanizma, ML'siz): EWMA-VY − EWMA-SABİT.
  H3b (ML'in karar değeri): ML21-VY − EWMA-VY — aynı mekanizmada daha iyi öngörü değer katıyor mu?
  BİLGİ (kapısız): ML21-VY − EWMA-SABİT (toplam aday), HAR21-VY − EWMA-VY ("ML gerekli mi?").
  TEŞHİS (SONRADAN, kapısız): ML21-SABİT − EWMA-SABİT birleşik z'si — Tur-98 H1'in sonucu görüldü; "etki yok"
  mu "kapı göremedi" mi sorusunu okutur, hükmü DEĞİŞTİRMEZ (Tur-23).
  Birincil ölçü: yıllık Sharpe farkı; piyasa z'si Memmel (2003). Yan ölçü: ortalama maruziyet oranı
  (BULGU-98-3: maruziyet düzeyi değişirse MDD farkı beceri değildir), devir, MDD.
  KAPI (her H, İKİ pencerede de): (i) BİRLEŞİK z = Σ z_k / √(Σ_jk R_jk) > 1,645 — R, piyasaların günlük
  fark serilerinin korelasyon matrisi (ABD ↔ ABD_izleme 7 ortak sembol taşır; bağımsızlık VARSAYILMAZ);
  (ii) ΔSharpe > 0 olan piyasa ≥ 10; (iii) ABD_izleme ΔSharpe > 0 (işaret; bu evrende tek başına anlamlılık
  İSTENMEZ — MDE ≈ +0,10, BULGU-98-2).
  MDE (koşudan ÖNCE, `--asama mde`; Tur-98 SABİT kollarının standart hatası + piyasalar arası korelasyon, z'ler
  toplanmadan): %80 güç, tek yönlü %5 → birleşik kapı ortak ΔSharpe 0,072 (2011-17; 14 piyasa, etkin 9,9) ve
  0,065 (2018-26; 15 piyasa, etkin 11,9) görür; yalnız ABD_izleme 0,151 / 0,148. VY kollarının kendi MDE'si
  karnede ayrıca basılır (kollar arası korelasyon değişir).
  Bağlantı: H3a geçerse EWMA-VY, H3a + H3b geçerse ML21-VY canlı quant çekirdeğine ADAYDIR. Canlı pozisyon
  yöneticisinde adet güncelleme yolu yok; bağlantı (açık quant pozisyonu için hedef adet önerisi) ancak
  geçen aday için kurulur.

CLI:  python -m research.quant_karar --asama motor --isci 8   # kollar + işlem defterleri
      python -m research.quant_karar --asama rapor            # data/quant_karar_karnesi.json
      python -m research.quant_karar --asama mde              # birleşik kapının MDE'si (Tur-98 serileri)
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
from research import quant_ml_motor as QMM

ROOT = Path(__file__).resolve().parents[1]
SONUC_DIR = C.SONUC_DIR / "quant99"
KARNE = ROOT / "data" / "quant_karar_karnesi.json"
KOLLAR = ("ewma", "ml21", "har21")                  # motor kolu = σ kaynağı
# f_tavan: motorun alım gücü tavanı (Reg T 2×; ölçüldü: `size_stock` 0,01 stopta nominal/özkaynak = 2,0,
# bağlayıcı sınır INSUFFICIENT_BUYING_POWER). İşlem başı risk %1,5 → giriş maruziyeti = 0,005/σ.
KARAR = {"bant": 0.25, "f_tavan": 2.0, "min_piyasa": 10, "birlesik_z": 1.645, "parite": 1e-8}
HIPOTEZLER = {   # ad: (aday, taban, kapılı mı)
    "H3a": ("ewma_vy", "ewma_sabit", True),
    "H3b": ("ml21_vy", "ewma_vy", True),
    "aday_toplam": ("ml21_vy", "ewma_sabit", False),
    "har21_vy": ("har21_vy", "ewma_vy", False),
    "teshis_tur98_h1": ("ml21_sabit", "ewma_sabit", False),   # SONRADAN — Tur-98 sonucu görüldü
}


# ---------------------------------------------------------------------------------------------------
# İşlem defteri — motorun kendi pozisyon yöneticisinden
# ---------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def defter():
    """`backtest_symbol` süresince açılan/kapanan pozisyonları yakalar. Orijinal metotlar Tur-97/98 yamalarıyla
    ORTAK sözlükte bir kez saklanır (BULGU-96-5: havuz işçisinde sarmalar zincirlenmez)."""
    from nihai.risk.position_manager import PositionManager as PM
    o_ac = QK._ORIJINAL.setdefault("pm_open", PM.open_from_signal)
    o_gun = QK._ORIJINAL.setdefault("pm_update", PM.update_all_with_bars)
    o_zor = QK._ORIJINAL.setdefault("pm_force", PM.force_close)
    kay: dict = {"giris": [], "cikis": {}}

    def open_from_signal(self, signal, sizing, *, now, **kw):
        E = float(self.portfolio.equity)
        pos = o_ac(self, signal, sizing, now=now, **kw)
        kay["giris"].append({"pos": pos, "now": now, "E": E})
        return pos

    def update_all_with_bars(self, *a, **kw):
        ces = o_gun(self, *a, **kw)
        for ce in ces:
            kay["cikis"][id(ce.pos)] = ce
        return ces

    def force_close(self, pos, price, reason, now):
        ce = o_zor(self, pos, price, reason, now)
        kay["cikis"][id(ce.pos)] = ce
        return ce

    for ad, f in (("open_from_signal", open_from_signal), ("update_all_with_bars", update_all_with_bars),
                  ("force_close", force_close)):
        setattr(PM, ad, f)
    try:
        yield kay
    finally:
        for ad, f in (("open_from_signal", o_ac), ("update_all_with_bars", o_gun), ("force_close", o_zor)):
            setattr(PM, ad, f)


def islemler(kay: dict, index: pd.DatetimeIndex) -> list[dict]:
    """Yakalanan defter → [{a, b, u, F, X, E, neden}] (a: giriş barı = karar barı + 1, b: çıkış barı)."""
    konum = {ts: i for i, ts in enumerate(index)}
    out = []
    for g in kay["giris"]:
        pos = g["pos"]
        ce = kay["cikis"].get(id(pos))
        a = konum[pd.Timestamp(g["now"])] + 1
        # BT_END: motor koşu sonunda zorla kapatır ama o özkaynak noktası getiri serisine girmez (son nokta
        # kapanış MTM'idir) → replikada pozisyon sona kadar açık sayılır.
        if ce is not None and ce.exit_trigger == "BT_END":
            ce = None
        b = konum[pd.Timestamp(ce.exit_time)] if ce is not None else None
        out.append({"a": a, "b": b, "u": float(abs(pos.quantity)), "F": float(pos.entry_price),
                    "X": float(ce.exit_price) if ce is not None else None, "E": g["E"],
                    "neden": ce.exit_trigger if ce is not None else None,
                    "pnl": float(ce.realized_pnl) if ce is not None else None})
    return out


# ---------------------------------------------------------------------------------------------------
# Getiri yeniden kurulumu — SABİT (motor) ve VY (oynaklık yönetimli adet)
# ---------------------------------------------------------------------------------------------------
def simule(O: np.ndarray, Cl: np.ndarray, t0: int, isl: list[dict], E0: float, sig: np.ndarray | None = None,
           *, bant: float = 0.25, maliyet: float = 0.001, tavan: float = math.inf) -> dict:
    """Kapanış t0..n−1 özkaynak yolu. `sig` None → SABİT (motorun adedi), verilirse VY.

    Kesin nedensel: açılış t'deki karar yalnız σ̂[t−1] ve t−1 kapanış özkaynağıyla verilir.
    Motorun MTM konvansiyonu birebir: giriş barında işaret dolum fiyatında kalır (`last_mark=fill`), ilk
    kapanış işaretlemesi ertesi bardadır → giriş gününün getirisi bir gün sonra yazılır."""
    n = len(Cl)
    E = np.empty(n - t0)
    E[0] = eq = E0
    giris = {x["a"]: x for x in isl}
    u = u_g = s_g = isaret = 0.0
    acik: dict | None = None
    devir = maruz = 0.0
    for t in range(t0 + 1, n):
        pnl = 0.0
        if acik is None and t in giris:
            acik = giris[t]
            u = u_g = acik["u"] * eq / acik["E"]          # motorun risk-tabanlı boyutu özkaynakla ölçeklenir
            s_g = float(sig[t - 1]) if sig is not None else 0.0
            if acik["b"] == t:                            # motor giriş barında çıkış denetlemez; savunma
                raise AssertionError("giriş barında çıkış")
            isaret = acik["F"]
        elif acik is not None:
            if t == acik["b"]:
                pnl = u * (acik["X"] - isaret)
                acik, u = None, 0.0
            else:
                yeni = u
                if sig is not None and s_g > 0 and np.isfinite(sig[t - 1]) and sig[t - 1] > 0:
                    hedef = min(u_g * s_g / float(sig[t - 1]), tavan * eq / Cl[t - 1])
                    if abs(hedef / u - 1.0) > bant:
                        yeni = hedef
                m = maliyet * abs(yeni - u) * O[t]
                devir += abs(yeni - u) * O[t] / eq
                pnl = u * (O[t] - isaret) + yeni * (Cl[t] - O[t]) - m
                u, isaret = yeni, Cl[t]
        eq += pnl
        maruz += u * Cl[t] / eq if eq > 0 else 0.0
        E[t - t0] = eq
    return {"E": E, "devir_gunluk": devir / max(n - t0 - 1, 1), "maruziyet": maruz / max(n - t0 - 1, 1)}


def sigma_serisi(df: pd.DataFrame, kol: dict) -> np.ndarray:
    """Kolun σ serisi — motor yamasıyla AYNI kural: verilen σ sonlu ve pozitifse o, yoksa EWMA."""
    from nihai.signals import quant as Q
    _, sig = Q.seriler(df["close"].to_numpy(float))
    if kol.get("sigma") is not None:
        s2 = kol["sigma"].reindex(df.index).to_numpy(dtype=float)
        sig = np.where(np.isfinite(s2) & (s2 > 0), s2, sig)
    return sig * float(kol.get("olcek", 1.0))


# ---------------------------------------------------------------------------------------------------
# Motor aşaması
# ---------------------------------------------------------------------------------------------------
def _dosya(piyasa: str, sym: str) -> Path:
    return SONUC_DIR / QK.dosya(piyasa, sym, "k99_").name


def enstruman_kos(arg: tuple[str, str]) -> str:
    piyasa, sym = arg
    C._isci_hazirla()
    t = time.time()
    df = C.hazirla(sym)
    kol = QMM.kol_argumanlari(QMM._seriler()[(piyasa, sym)])
    out: dict = {"piyasa": piyasa, "sembol": sym, "kos": {}}
    for pen, (bas, son) in QK.SIGMA_PENCERE.items():
        d = df if son is None else df[df.index < pd.Timestamp(son)]
        t0 = int((d.index < pd.Timestamp(bas)).sum())
        if len(d) - t0 < 250 or t0 < 60:
            continue
        O, Cl = d["open"].to_numpy(float), d["close"].to_numpy(float)
        for ad in KOLLAR:
            k = kol[ad]
            with contextlib.ExitStack() as st:
                if k:
                    st.enter_context(QMM.yama(**k))
                kay = st.enter_context(defter())
                r = C._algoritma_kos(sym, d, t0, "quant_trend")
            isl = islemler(kay, d.index)
            motor = pd.Series(r["r"].astype(float), index=pd.DatetimeIndex(r["tarih"]))
            E0 = 1.0
            sab = simule(O, Cl, t0, isl, E0)
            sig = sigma_serisi(d, k)
            vy = simule(O, Cl, t0, isl, E0, sig, bant=KARAR["bant"], maliyet=C.maliyet_orani(),
                        tavan=KARAR["f_tavan"] or math.inf)
            idx = d.index[t0 + 1:]
            r_sab = pd.Series(sab["E"][1:] / sab["E"][:-1] - 1.0, index=idx)
            out["kos"][f"{ad}_{pen}"] = {
                "motor": motor, "sabit": r_sab,
                "vy": pd.Series(vy["E"][1:] / vy["E"][:-1] - 1.0, index=idx),
                "parite": float(np.nanmax(np.abs(r_sab.to_numpy() - motor.reindex(idx).to_numpy()))) if len(idx) else 0.0,
                "islem": len(isl), "f_giris": [x["u"] * x["F"] / x["E"] for x in isl],
                "devir": {"sabit": sab["devir_gunluk"], "vy": vy["devir_gunluk"]},
                "maruziyet": {"sabit": sab["maruziyet"], "vy": vy["maruziyet"]}}
    SONUC_DIR.mkdir(parents=True, exist_ok=True)
    with open(_dosya(piyasa, sym), "wb") as f:
        pickle.dump(out, f)
    return f"{piyasa}/{sym}: {len(out['kos'])} kol, {round(time.time() - t, 1)} sn"


# ---------------------------------------------------------------------------------------------------
# Çıkarım — piyasa Sharpe farkı + BİRLEŞİK (korelasyon düzeltmeli) z + MDE
# ---------------------------------------------------------------------------------------------------
def sharpe_fark(a: pd.Series, b: pd.Series, gun: int) -> dict:
    """Yıllık ΔSharpe (a − b), Memmel z'si ve ΔSharpe'ın yıllık standart hatası (MDE için)."""
    ortak = a.index.intersection(b.index)
    x, y = a.loc[ortak].to_numpy(float), b.loc[ortak].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 60 or x.std() == 0 or y.std() == 0:
        return {"ds": None, "z": None, "se": None, "n": int(x.size)}
    sa, sb = x.mean() / x.std(ddof=1), y.mean() / y.std(ddof=1)
    rho = float(np.corrcoef(x, y)[0, 1])
    v = (2 - 2 * rho + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / x.size
    ds = round(float((sa - sb) * math.sqrt(gun)), 4)
    if v <= 0:
        return {"ds": ds, "z": None, "se": None, "n": int(x.size)}
    return {"ds": ds, "z": round(float((sa - sb) / math.sqrt(v)), 3), "se": float(math.sqrt(v * gun)),
            "rho": round(rho, 4), "n": int(x.size)}


def korelasyon_toplami(farklar: dict[str, pd.Series], min_ortak: int = 250) -> tuple[list[str], float]:
    """Piyasaların günlük fark serileri → (adlar, Σ_jk R_jk). Ortak günü `min_ortak`tan az olan çiftte R = 0."""
    ad = list(farklar)
    R = np.eye(len(ad))
    for i in range(len(ad)):
        for j in range(i + 1, len(ad)):
            fi, fj = farklar[ad[i]], farklar[ad[j]]
            ortak = fi.index.intersection(fj.index)
            if len(ortak) < min_ortak:
                continue
            x, y = fi.loc[ortak].to_numpy(float), fj.loc[ortak].to_numpy(float)
            if x.std() > 0 and y.std() > 0:
                R[i, j] = R[j, i] = float(np.corrcoef(x, y)[0, 1])
    return ad, float(R.sum())


def birlesik_z(zler: dict[str, float | None], farklar: dict[str, pd.Series]) -> dict:
    """Korelasyon düzeltmeli Stouffer: Z = Σ z_k / √(Σ_jk R_jk). Bağımsızlıkta R = I (klasik Stouffer); ortak
    sembol taşıyan ya da aynı faktöre bağlı piyasalar pozitif korelasyonla ETKİN sayıyı düşürür."""
    kull = {k: z for k, z in zler.items() if z is not None and k in farklar}
    if len(kull) < 2:
        return {"z": None, "n": len(kull)}
    ad, S = korelasyon_toplami({k: farklar[k] for k in kull})
    if S <= 0:
        return {"z": None, "n": len(ad), "S": round(S, 3)}
    return {"z": round(float(sum(kull[k] for k in ad) / math.sqrt(S)), 3), "n": len(ad), "S": round(S, 3),
            "etkin_n": round(len(ad) ** 2 / S, 2)}


def mde(se: dict[str, float | None], S: float, alfa_z: float = 1.645, guc_z: float = 0.8416) -> float | None:
    """Birleşik kapının %80 güçle ayırt ettiği en küçük ORTAK yıllık ΔSharpe: her piyasada gerçek fark δ ise
    E[Z] = δ·Σ(1/se_k)/√S; E[Z] ≥ z_α + z_güç → δ ≥ (z_α + z_güç)·√S / Σ(1/se_k)."""
    ters = [1.0 / v for v in se.values() if v]
    if not ters or S <= 0:
        return None
    return round((alfa_z + guc_z) * math.sqrt(S) / sum(ters), 4)


# ---------------------------------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------------------------------
def _piyasa_serileri(pz: str, semboller: list[str]) -> tuple[dict, dict, dict]:
    """→ ({kol_politika_pen: EW seri}, {… : ort. maruziyet/devir}, parite)."""
    seri: dict[str, list[pd.Series]] = {}
    ek: dict[str, dict[str, list[float]]] = {}
    parite: dict = {"max": 0.0, "en_kotu": None}
    for s in semboller:
        p = _dosya(pz, s)
        if not p.exists():
            continue
        with open(p, "rb") as f:
            q = pickle.load(f)
        for anahtar, k in q["kos"].items():
            kol, pen = anahtar.split("_", 1)
            if k["parite"] > parite["max"]:
                parite = {"max": k["parite"], "en_kotu": f"{s}/{anahtar}"}
            for pol, r in (("sabit", k["motor"]), ("vy", k["vy"])):
                ad = f"{kol}_{pol}_{pen}"
                seri.setdefault(ad, []).append(r)
                e = ek.setdefault(ad, {"maruziyet": [], "devir": []})
                e["maruziyet"].append(k["maruziyet"][pol])
                e["devir"].append(k["devir"][pol])
    ew = {ad: C._ew(v) for ad, v in seri.items()}
    ort = {ad: {"maruziyet": round(float(np.mean(v["maruziyet"])), 4),
                "devir_yillik": round(float(np.mean(v["devir"])) * 252, 3)} for ad, v in ek.items()}
    return ew, ort, parite


def rapor(kull: list[tuple[str, str]]) -> dict:
    piyasalar: dict[str, list[str]] = {}
    for pz, s in kull:
        piyasalar.setdefault(pz, []).append(s)
    karne: dict = {"uretim": time.strftime("%Y-%m-%d %H:%M"), "pencereler": QK.SIGMA_PENCERE, "karar": KARAR,
                   "hipotezler": {h: {"aday": a, "taban": b, "kapili": k} for h, (a, b, k) in HIPOTEZLER.items()},
                   "maliyet_taraf": C.maliyet_orani(), "piyasalar": {}}
    fark: dict = {}          # (H, pencere) → {piyasa: günlük fark serisi}
    parite_max: dict = {"max": 0.0, "en_kotu": None}
    for pz, semboller in piyasalar.items():
        gun = QK._gun(pz)
        ew, ort, par = _piyasa_serileri(pz, semboller)
        if par["max"] >= parite_max["max"]:
            parite_max = {**par, "piyasa": pz}
        blok: dict = {}
        for pen in QK.SIGMA_PENCERE:
            kol = {ad[: -len(pen) - 1]: s for ad, s in ew.items() if ad.endswith("_" + pen)}
            if not kol:
                continue
            b: dict = {"metrik": {k: {**C.seri_metrik(s.to_numpy(), gun), **ort[f"{k}_{pen}"]}
                                  for k, s in kol.items()}}
            for h, (aday, taban, _) in HIPOTEZLER.items():
                if aday in kol and taban in kol:
                    b[h] = sharpe_fark(kol[aday], kol[taban], gun)
                    ortak = kol[aday].index.intersection(kol[taban].index)
                    fark.setdefault((h, pen), {})[pz] = kol[aday].loc[ortak] - kol[taban].loc[ortak]
            blok[pen] = b
        karne["piyasalar"][pz] = blok
    karne["parite"] = {**parite_max, "gecti": parite_max["max"] <= KARAR["parite"]}
    karne["birlesik"] = {}
    for h in HIPOTEZLER:
        karne["birlesik"][h] = {}
        for pen in QK.SIGMA_PENCERE:
            bl = {pz: b[pen][h] for pz, b in karne["piyasalar"].items() if pen in b and h in b[pen]}
            fk = fark.get((h, pen), {})
            bz = birlesik_z({pz: m["z"] for pz, m in bl.items()}, fk)
            _, S = korelasyon_toplami({pz: fk[pz] for pz in bl if bl[pz]["se"]})
            karne["birlesik"][h][pen] = {**bz, "olumlu_piyasa": sum(1 for m in bl.values() if (m["ds"] or 0) > 0),
                                         "n_piyasa": len(bl), "izleme_ds": (bl.get(QK.IZLEME) or {}).get("ds"),
                                         "mde": mde({pz: m["se"] for pz, m in bl.items()}, S)}
    karne["hukum"] = hukum(karne)
    return karne


def hukum(karne: dict) -> dict:
    out: dict = {}
    for h, (_, _, kapili) in HIPOTEZLER.items():
        kos = {pen: {"gecti": bool(b.get("z") is not None and b["z"] > KARAR["birlesik_z"]
                                   and b["olumlu_piyasa"] >= KARAR["min_piyasa"]
                                   and b.get("izleme_ds") is not None and b["izleme_ds"] > 0)}
               for pen, b in karne["birlesik"][h].items()}
        out[h] = {"kapili": kapili, "kosullar": kos,
                  "gecti": bool(kos) and all(k["gecti"] for k in kos.values()) and karne["parite"]["gecti"]}
    # ML21-VY'nin tabanı EWMA-VY'dir → H3b ancak H3a ile birlikte canlıya karşı bir aday üretir
    out["aday"] = ("ml21_vy" if out["H3a"]["gecti"] and out["H3b"]["gecti"]
                   else "ewma_vy" if out["H3a"]["gecti"] else None)
    return out


def mde_tur98() -> dict:
    """Koşudan ÖNCE: birleşik kapının MDE'si Tur-98'in SABİT kollarıyla (ML21 ↔ EWMA). Yalnız standart hatalar ve
    piyasalar arası korelasyon kullanılır; z'ler TOPLANMAZ (Tur-98 H1'in birleşik sonucu ön-kayda sızmasın)."""
    kull, _ = QK.evren()
    piyasalar: dict[str, list[str]] = {}
    for pz, s in kull:
        piyasalar.setdefault(pz, []).append(s)
    out: dict = {}
    for pen in QK.SIGMA_PENCERE:
        se: dict[str, float | None] = {}
        fk: dict[str, pd.Series] = {}
        se_izl = None
        for pz, semboller in piyasalar.items():
            seri: dict[str, list[pd.Series]] = {}
            for s in semboller:
                p = QMM._m98(pz, s)
                if not p.exists():
                    continue
                with open(p, "rb") as f:
                    q = pickle.load(f)
                for ad in ("ewma", "ml21"):
                    k = q["kos"].get(f"{ad}_{pen}")
                    if k is not None:
                        seri.setdefault(ad, []).append(k["s"])
            if len(seri) < 2:
                continue
            a, b = C._ew(seri["ml21"]), C._ew(seri["ewma"])
            m = sharpe_fark(a, b, QK._gun(pz))
            if m["se"]:
                se[pz] = m["se"]
                ortak = a.index.intersection(b.index)
                fk[pz] = a.loc[ortak] - b.loc[ortak]
                if pz == QK.IZLEME:
                    se_izl = m["se"]
        _, S = korelasyon_toplami(fk)
        out[pen] = {"mde_birlesik": mde(se, S), "S": round(S, 3), "n": len(se),
                    "etkin_n": round(len(se) ** 2 / S, 2) if S > 0 else None,
                    "mde_yalniz_izleme": round(2.4866 * se_izl, 4) if se_izl else None}
    return out


def ozet_md(karne: dict) -> str:
    def f(x, nd=2):
        return "—" if x is None else f"{x:.{nd}f}"
    par = karne["parite"]
    s = [f"parite: max {par['max']:.1e} ({par.get('en_kotu')}) → {'GEÇTİ' if par['gecti'] else 'HÜKÜM YOK'}", ""]
    for pen in QK.SIGMA_PENCERE:
        s += [f"### {pen}", "",
              "| piyasa | EWMA S | H3a ΔS (z) | H3b ΔS (z) | ML21-VY−EWMA (z) | HAR21-VY (z) | ML21−EWMA sabit (z) |"
              " maruz. sabit/VY |",
              "|---|---|---|---|---|---|---|---|"]
        for pz, b in karne["piyasalar"].items():
            if pen not in b:
                continue
            x = b[pen]

            def h(ad, x=x):
                m = x.get(ad) or {}
                return f"{f(m.get('ds'))} ({f(m.get('z'))})"
            mt = x["metrik"]
            s.append(f"| {pz} | {f(mt.get('ewma_sabit', {}).get('sharpe'))} | {h('H3a')} | {h('H3b')} | "
                     f"{h('aday_toplam')} | {h('har21_vy')} | {h('teshis_tur98_h1')} | "
                     f"{f(mt.get('ewma_sabit', {}).get('maruziyet'))}/{f(mt.get('ewma_vy', {}).get('maruziyet'))} |")
        s.append("")
    for h, bl in karne["birlesik"].items():
        s.append(f"- {h}: " + " · ".join(
            f"{pen} Z={f(b.get('z'))} (etkin n {f(b.get('etkin_n'), 1)}, olumlu {b['olumlu_piyasa']}/{b['n_piyasa']}, "
            f"izleme ΔS {f(b.get('izleme_ds'))}, MDE {f(b.get('mde'), 3)})" for pen, b in bl.items()))
    for h, hk in karne["hukum"].items():
        if h != "aday":
            s.append(f"- HÜKÜM {h} ({'KAPILI' if hk['kapili'] else 'bilgi'}): geçti={hk['gecti']} {hk['kosullar']}")
    s.append(f"- ADAY: {karne['hukum']['aday']}")
    return "\n".join(s)


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Tur-99 ML karar mekanizması — VY boyut kararı + birleşik kapı")
    ap.add_argument("--asama", required=True, choices=["mde", "motor", "rapor"])
    ap.add_argument("--isci", type=int, default=8)
    a = ap.parse_args()
    C._isci_hazirla()
    kull, _ = QK.evren()
    if a.asama == "mde":
        print(json.dumps(mde_tur98(), ensure_ascii=False, indent=1))
    elif a.asama == "motor":
        C._havuz(enstruman_kos, kull, a.isci)
    else:
        karne = rapor(kull)
        karne["parmak_izi"] = C.girdi_parmak_izi(sorted({s for _, s in kull}))
        KARNE.write_text(json.dumps(karne, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(ozet_md(karne))
        print(f"karne → {KARNE}")


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    _cli()
