"""quant_kayma.py — Tur-97: algoritmaların quant tarafına kaydırılmasının ÖLÇÜMÜ.

Soru: canlı karar yolunun kural tabanlı TA detektörlerinin (canlıda `event_breakout`) yerine
`quant_trend` (nihai/signals/quant.py: çok ufuklu zaman serisi momentumu + günlük σ ölçekli risk)
konulmalı mı? Ölçüt Tur-96 ile AYNI motor, AYNI veri, AYNI maliyet: `backtest_symbol` 1d, kill-switch
kapalı, AI kapısı kapalı (canlı paritesi), kapanış→kapanış MTM getirisi, piyasa = eşit ağırlıklı
enstrüman ortalaması, kıyas al-tut (kapanış→kapanış).

Pencereler:
  · TEST 2018-01-01 → son (Tur-96 penceresi; TA algoritmalarının sonuçları Tur-96 önbelleğinden okunur)
  · ÖRNEKLEM-DIŞI 2009-01-01 → 2017-12-31: quant tasarım kararları (özellikle uzun-yalnız) 2018-26
    vektörel keşfinden SONRA verildi; bu pencere o kararlara hiç dokunmamış veridir. Aynı pencerede
    event_breakout da koşulur (eşleştirilmiş kıyas).
  · +1 bar gecikme (TEST): karar bir bar eski veriyle (ACI DERS #7 — beceri ilk barda mı?).

Evren: Tur-96'nın 14 piyasa grubu (121 enstrüman) + `ABD_izleme` (canlı izleme listesinin 30 sembolü,
yfinance). ABD_izleme BUGÜNÜN listesidir → hayatta kalan yanlılığı taşır (ACI DERS #1): mutlak getiri
yorumlanmaz, yalnız AYNI evrende quant ↔ event_breakout farkı okunur.

Birincil ölçü (sonuçtan önce sabitlendi): piyasa başına günlük FARK serisi r_quant − r_event_breakout,
ortalamasının Newey-West t'si ("quant, canlı algoritmayı yeniyor mu?"). Çoklu test: 15 piyasa × 2
pencere = 30 hücre → Bonferroni |t| ≥ 3,14.

CLI:
  python -m research.quant_kayma --asama indir              # ABD_izleme yfinance önbelleği
  python -m research.quant_kayma --asama motor --isci 8     # enstrüman başı motor koşuları
  python -m research.quant_kayma --asama rapor              # data/quant_kayma_karnesi.json + tablo
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

ROOT = Path(__file__).resolve().parents[1]
SONUC_DIR = C.SONUC_DIR / "quant97"
KARNE = ROOT / "data" / "quant_kayma_karnesi.json"
OOS_BAS = "2009-01-01"
OOS_MIN_BAR = 500
IZLEME = "ABD_izleme"
TA_MODLARI = ("event_breakout", "mean_reversion", "rsi_divergence", "macd", "donchian", "squeeze",
              "trend_pullback", "vol_breakout", "ensemble", "confluence")


def izleme_listesi() -> list[str]:
    """Canlı izleme listesi (`.env` STREAM_SYMBOLS → yoksa WATCHLIST)."""
    import os

    from dotenv import dotenv_values
    env = dotenv_values(ROOT / ".env")
    ham = env.get("STREAM_SYMBOLS") or env.get("WATCHLIST") or os.getenv("WATCHLIST") or ""
    return [s.strip().upper() for s in ham.split("#")[0].split(",") if s.strip()]


def indir() -> list[str]:
    """ABD_izleme önbelleğini doldurur (ağ). Tur-96 havuzundaki semboller ÇEKİLMEZ (BULGU-98-6): 7 izleme sembolü
    (SPY, QQQ, AAPL…) havuzda da var ve yeniden çekmek havuzun dondurulmuş girdisini sessizce değiştirir — yeni gün
    ve düzeltilmiş geçmiş gelir, Tur-96/97 karneleri bit düzeyinde yeniden üretilemez olur. Dönen: çekilenler."""
    from research import piyasalar as P
    havuz = {s for _, s in P.tum_semboller()}
    cekilen = []
    for s in izleme_listesi():
        if s in havuz:
            print(s, "havuzda — önbellek dondurulmuş, çekilmedi", flush=True)
            continue
        df = P.yukle(s, fetch=True)
        cekilen.append(s)
        print(s, len(df), df.index[0].date() if len(df) else "—", flush=True)
    return cekilen


def evren() -> tuple[list[tuple[str, str]], list]:
    kull, dis = C.enstrumanlar()
    from research import piyasalar as P
    for s in izleme_listesi():
        neden = C.kusur_nedeni("ABD", P.yukle(s))
        (dis.append((IZLEME, s, neden)) if neden else kull.append((IZLEME, s)))
    return kull, dis


# ---------------------------------------------------------------------------------------------------
# Motor koşuları
# ---------------------------------------------------------------------------------------------------
def dosya(piyasa: str, sym: str, onek: str = "") -> Path:
    """Sonuç dosyası PİYASA + sembolle adlanır (BULGU-98-1): 7 sembol (SPY, QQQ, AAPL…) hem ABD havuzunda
    hem ABD_izleme'de; iki koşu farklı kol taşır ve yalnız sembolle adlanınca SON yazılan kazanıyordu."""
    return SONUC_DIR / f"{onek}{C._guvenli(piyasa)}__{C._guvenli(sym)}.pkl"


_ORIJINAL: dict = {}     # BULGU-96-5: orijinal fonksiyon süreç başına BİR KEZ saklanır (sarmalar zincirlenmez)


@contextlib.contextmanager
def gecikmeli(k: int):
    """Motorun `generate_signal`'ını k bar eski veriyle çağıran geçici sarmalayıcı."""
    import nihai.execution.backtest as B
    orj = _ORIJINAL.setdefault("gs", B.generate_signal)

    def gs(symbol, df, *a, shared=None, **kw):
        n = len(df) - k
        sh = shared._p.upto(n) if shared is not None else None
        return orj(symbol, df.iloc[:n], *a, shared=sh, **kw)

    setattr(B, "generate_signal", gs)                      # noqa: B010 — mypy imza denetimini aşmak için
    try:
        yield
    finally:
        setattr(B, "generate_signal", orj)                 # noqa: B010


def _bh(df: pd.DataFrame, t0: int) -> pd.Series:
    c = df["close"].to_numpy(float)
    return pd.Series(c[t0 + 1:] / c[t0:-1] - 1.0, index=df.index[t0 + 1:])


def _kos(sym: str, df: pd.DataFrame, t0: int, mod: str) -> dict:
    r = C._algoritma_kos(sym, df, t0, mod)
    return {"s": pd.Series(r["r"].astype(float), index=pd.DatetimeIndex(r["tarih"])),
            "islem": r["islem"], "kazanma": r["kazanma"], "pf": r["pf"]}


def enstruman_kos(arg: tuple[str, str]) -> str:
    piyasa, sym = arg
    C._isci_hazirla()
    t = time.time()
    df = C.hazirla(sym)
    t_test = int((df.index < pd.Timestamp(C.TEST_BAS)).sum())
    out: dict = {"piyasa": piyasa, "sembol": sym, "kos": {}, "bh": {"test": _bh(df, t_test)}}
    out["kos"]["quant_trend"] = _kos(sym, df, t_test, "quant_trend")
    with gecikmeli(1):
        out["kos"]["quant_trend_g1"] = _kos(sym, df, t_test, "quant_trend")
    if piyasa == IZLEME:                                     # Tur-96 önbelleğinde yok
        out["kos"]["event_breakout"] = _kos(sym, df, t_test, "event_breakout")
    oos = df[df.index < pd.Timestamp(C.TEST_BAS)]
    t_oos = int((oos.index < pd.Timestamp(OOS_BAS)).sum())
    t_oos = max(t_oos, 60)
    if len(oos) - t_oos >= OOS_MIN_BAR:
        out["bh"]["oos"] = _bh(oos, t_oos)
        out["kos"]["quant_trend_oos"] = _kos(sym, oos, t_oos, "quant_trend")
        out["kos"]["event_breakout_oos"] = _kos(sym, oos, t_oos, "event_breakout")
    SONUC_DIR.mkdir(parents=True, exist_ok=True)
    with open(dosya(piyasa, sym), "wb") as f:
        pickle.dump(out, f)
    q = out["kos"]["quant_trend"]
    return f"{piyasa}/{sym}: quant {q['islem']} işlem, {round(time.time() - t, 1)} sn"


# ---------------------------------------------------------------------------------------------------
# σ kaynağı MOTOR testi (Tur-97, ML): quant_trend'in σ'sı EWMA ↔ havuzlanmış ML öngörüsü
# ---------------------------------------------------------------------------------------------------
# Neden motor: `quant_ml`in vektörel testi her gün σ hedefine YENİDEN DENGELER; σ_ML'in günlük devri
# EWMA'nın 3-8 katı olduğu için maliyet kazancı yedi. Canlı quant_trend ise σ'yı yalnız GİRİŞTE (stop +
# boyut) kullanır — doğru soru motorun kendisinde sorulur.
# ÖN-KAYIT (motor sonucu görülmeden sabitlendi): σ_ML çekirdeğe YALNIZ iki pencerenin İKİSİNDE de
# (2011-17, 2018-26) fark t > 0 olan piyasa ≥ 10/15 VE ABD_izleme fark t > 1,645 ise bağlanır.
SIGMA_PENCERE = {"oos_2011_17": ("2011-01-01", "2018-01-01"), "test_2018_26": ("2018-01-01", None)}
SIGMA_KARAR = {"min_piyasa": 10, "izleme_t": 1.645}
_SIGMA_ML: dict = {"seri": None}


def _sigma_modelleri_yolu() -> Path:
    return SONUC_DIR / "vol_modeller.pkl"


@contextlib.contextmanager
def ml_sigma(seri: pd.Series):
    """Motorun quant σ'sını `seri`den (tarih → günlük σ) okutan geçici yama; eksik gün EWMA'ya düşer."""
    from nihai.signals import quant as Q
    from nihai.signals import strategy as S
    orj = _ORIJINAL.setdefault("quant_np", S._SharedIndicators.quant_np)

    def quant_np(self):
        def _kur():
            skor, sig = Q.seriler(self.closes())
            ml = _SIGMA_ML["seri"].reindex(self.df.index).to_numpy(dtype=float)
            return skor, np.where(np.isfinite(ml) & (ml > 0), ml, sig)
        return self._get("quant_np_ml", _kur)

    _SIGMA_ML["seri"] = seri
    setattr(S._SharedIndicators, "quant_np", quant_np)      # noqa: B010 — metot ataması (mypy)
    try:
        yield
    finally:
        setattr(S._SharedIndicators, "quant_np", orj)     # noqa: B010
        _SIGMA_ML["seri"] = None


def sigma_kos(arg: tuple[str, str]) -> str:
    from research import quant_ml as QM
    piyasa, sym = arg
    C._isci_hazirla()
    t = time.time()
    with open(_sigma_modelleri_yolu(), "rb") as f:
        modeller = pickle.load(f)
    df = C.hazirla(sym)
    pan = QM.panel(sym, piyasa)
    seri = pd.Series(np.sqrt(QM.ml_f_uygula(pan, modeller)), index=pan.index)
    out: dict = {"piyasa": piyasa, "sembol": sym, "kos": {}}
    for pen, (bas, son) in SIGMA_PENCERE.items():
        d = df if son is None else df[df.index < pd.Timestamp(son)]
        t0 = int((d.index < pd.Timestamp(bas)).sum())
        if len(d) - t0 < 250 or t0 < 60:
            continue
        out["kos"][f"ewma_{pen}"] = _kos(sym, d, t0, "quant_trend")
        with ml_sigma(seri):
            out["kos"][f"ml_{pen}"] = _kos(sym, d, t0, "quant_trend")
    with open(dosya(piyasa, sym, "sigma_"), "wb") as f:
        pickle.dump(out, f)
    return f"{piyasa}/{sym}: σ testi {round(time.time() - t, 1)} sn"


def sigma_rapor(kull: list[tuple[str, str]]) -> dict:
    piyasalar: dict[str, list[str]] = {}
    for pz, s in kull:
        piyasalar.setdefault(pz, []).append(s)
    sonuc: dict = {}
    for pz, semboller in piyasalar.items():
        gun = _gun(pz)
        seri: dict[str, list[pd.Series]] = {}
        islem: dict[str, int] = {}
        for s in semboller:
            p = dosya(pz, s, "sigma_")
            if not p.exists():
                continue
            with open(p, "rb") as f:
                q = pickle.load(f)
            for ad, k in q["kos"].items():
                seri.setdefault(ad, []).append(k["s"])
                islem[ad] = islem.get(ad, 0) + int(k["islem"])
        ew = {ad: C._ew(v) for ad, v in seri.items()}
        blok: dict = {}
        for pen in SIGMA_PENCERE:
            a, b = ew.get(f"ml_{pen}"), ew.get(f"ewma_{pen}")
            if a is None or b is None:
                continue
            blok[pen] = {"ewma": {**C.seri_metrik(b.to_numpy(), gun), "islem": islem.get(f"ewma_{pen}")},
                         "ml": {**C.seri_metrik(a.to_numpy(), gun), "islem": islem.get(f"ml_{pen}")},
                         "fark": fark_t(a, b)}
        sonuc[pz] = blok
    kosullar = {}
    for pen in SIGMA_PENCERE:
        olumlu = sum(1 for b in sonuc.values() if pen in b and (b[pen]["fark"].get("t") or 0) > 0)
        izl = (sonuc.get(IZLEME, {}).get(pen) or {}).get("fark", {}).get("t")
        kosullar[pen] = {"olumlu_piyasa": olumlu, "n_piyasa": sum(1 for b in sonuc.values() if pen in b),
                         "izleme_t": izl, "gecti": olumlu >= SIGMA_KARAR["min_piyasa"]
                         and izl is not None and izl > SIGMA_KARAR["izleme_t"]}
    return {"piyasalar": sonuc, "kosullar": kosullar, "baglanir": all(k["gecti"] for k in kosullar.values())}


def _oku(piyasa: str, sym: str) -> dict | None:
    p = dosya(piyasa, sym)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------------------------------
def _gun(piyasa: str) -> int:
    from research import piyasalar as P
    return P.PIYASALAR.get(piyasa, {}).get("gun", 252)


def fark_t(a: pd.Series, b: pd.Series) -> dict:
    """Eşleştirilmiş günlük fark (a − b): ortalamanın NW t'si + yıllık fark (yüzde puan)."""
    ortak = a.index.intersection(b.index)
    d = (a.loc[ortak] - b.loc[ortak]).to_numpy(float)
    return {"t": C.nw_ortalama_t(d), "yillik_pp": round(float(np.nanmean(d)) * 252 * 100, 3) if d.size else None,
            "n": int(d.size)}


def _satir(r: pd.Series, bh: pd.Series, gun: int, islem: int | None = None) -> dict:
    d = C._satir(r, bh, gun)
    if islem is not None:
        d["islem"] = islem
    return d


def rapor(kull: list[tuple[str, str]], dis: list) -> dict:
    piyasalar: dict[str, list[str]] = {}
    for pz, s in kull:
        piyasalar.setdefault(pz, []).append(s)
    karne: dict = {"uretim": time.strftime("%Y-%m-%d %H:%M"), "test_bas": C.TEST_BAS, "oos": [OOS_BAS, C.TEST_BAS],
                   "maliyet_taraf": C.maliyet_orani(), "dislanan": dis, "piyasalar": {}}
    for pz, semboller in piyasalar.items():
        gun = _gun(pz)
        seri: dict[str, list[pd.Series]] = {}
        islem: dict[str, int] = {}
        eksik = []
        for s in semboller:
            q = _oku(pz, s)
            if q is None:
                eksik.append(s)
                continue
            for ad, k in q["kos"].items():
                seri.setdefault(ad, []).append(k["s"])
                islem[ad] = islem.get(ad, 0) + int(k["islem"])
            for pen, b in q["bh"].items():
                seri.setdefault(f"bh_{pen}", []).append(b)
            if pz != IZLEME:
                e = C.ens_oku(s)
                for m in TA_MODLARI:
                    a = e["algo"].get(m)
                    if a is None:
                        continue
                    seri.setdefault(m, []).append(pd.Series(a["r"].astype(float),
                                                            index=pd.DatetimeIndex(a["tarih"])))
                    islem[m] = islem.get(m, 0) + int(a["islem"])
        ew = {ad: C._ew(lst) for ad, lst in seri.items()}
        bh_t = ew.get("bh_test", pd.Series(dtype=float))
        bh_o = ew.get("bh_oos")
        blok: dict = {"semboller": semboller, "eksik": eksik, "gun": gun, "test": {}, "oos": {}}
        blok["test"]["al_tut"] = C.seri_metrik(bh_t.to_numpy(), gun)
        for ad in ("quant_trend", "quant_trend_g1", *TA_MODLARI):
            if ad in ew:
                blok["test"][ad] = _satir(ew[ad], bh_t, gun, islem.get(ad))
        if "quant_trend" in ew and "event_breakout" in ew:
            blok["test"]["fark_quant_eb"] = fark_t(ew["quant_trend"], ew["event_breakout"])
            ta = [m for m in TA_MODLARI if m in ew]
            blok["test"]["quant_yendigi_ta"] = sum(
                1 for m in ta if (blok["test"]["quant_trend"].get("sharpe") or -9) > (blok["test"][m].get("sharpe") or -9))
            blok["test"]["ta_sayisi"] = len(ta)
        if bh_o is not None and "quant_trend_oos" in ew:
            blok["oos"]["al_tut"] = C.seri_metrik(bh_o.to_numpy(), gun)
            for ad in ("quant_trend_oos", "event_breakout_oos"):
                blok["oos"][ad.replace("_oos", "")] = _satir(ew[ad], bh_o, gun, islem.get(ad))
            blok["oos"]["fark_quant_eb"] = fark_t(ew["quant_trend_oos"], ew["event_breakout_oos"])
        karne["piyasalar"][pz] = blok
    return karne


def bonferroni(n: int) -> float:
    return C.bonferroni_esigi(n)


def ozet_md(karne: dict) -> str:
    def f(x, nd=2):
        return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"

    satir = ["| piyasa | al-tut S | quant S (MDD) | +1 bar | event_breakout S (MDD) | TA medyan S | quant > TA | "
             "fark t (yıllık pp) | ÖD: quant S | ÖD: EB S | ÖD fark t |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for pz, b in karne["piyasalar"].items():
        t, o = b["test"], b.get("oos") or {}
        q, eb = t.get("quant_trend", {}), t.get("event_breakout", {})
        ta = [t[m]["sharpe"] for m in TA_MODLARI if m in t and t[m].get("sharpe") is not None]
        fk, fo = t.get("fark_quant_eb", {}), o.get("fark_quant_eb", {})
        satir.append(
            f"| {pz} | {f(t['al_tut'].get('sharpe'))} | {f(q.get('sharpe'))} ({f(q.get('mdd_pct'), 1)}) | "
            f"{f(t.get('quant_trend_g1', {}).get('sharpe'))} | {f(eb.get('sharpe'))} ({f(eb.get('mdd_pct'), 1)}) | "
            f"{f(float(np.median(ta)) if ta else None)} | {t.get('quant_yendigi_ta', '—')}/{t.get('ta_sayisi', '—')} | "
            f"{f(fk.get('t'))} ({f(fk.get('yillik_pp'), 1)}) | {f(o.get('quant_trend', {}).get('sharpe'))} | "
            f"{f(o.get('event_breakout', {}).get('sharpe'))} | {f(fo.get('t'))} |")
    return "\n".join(satir)


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Tur-97 quant kayması ölçümü")
    ap.add_argument("--asama", required=True, choices=["indir", "motor", "sigma", "rapor", "ozet"])
    ap.add_argument("--isci", type=int, default=8)
    ap.add_argument("--eksik", action="store_true")
    a = ap.parse_args()
    from research import piyasalar as P
    if a.asama == "indir":
        indir()
        return
    P.yukle("^VIX", fetch=True)
    kull, dis = evren()
    if a.asama == "motor":
        gorev = [g for g in kull if not (a.eksik and dosya(*g).exists())]
        print(f"{len(gorev)} görev, dışlanan {len(dis)}", flush=True)
        C._havuz(enstruman_kos, gorev, a.isci)
    elif a.asama == "sigma":
        from research import quant_ml as QM
        modeller: dict = {}
        QM.vol_tahmin(QM.tum_panel(), modeller)             # 14 piyasa havuzu, yıllık yürüyen eğitim
        SONUC_DIR.mkdir(parents=True, exist_ok=True)
        with open(_sigma_modelleri_yolu(), "wb") as f:
            pickle.dump(modeller, f)
        print(f"{len(modeller)} yıllık oynaklık modeli → {_sigma_modelleri_yolu()}", flush=True)
        C._havuz(sigma_kos, kull, a.isci)
        sr = sigma_rapor(kull)
        print(json.dumps(sr["kosullar"], ensure_ascii=False), "bağlanır:", sr["baglanir"])
    elif a.asama == "rapor":
        karne = rapor(kull, dis)
        if any(SONUC_DIR.glob("sigma_*.pkl")):
            karne["sigma_motor"] = sigma_rapor(kull)
        karne["parmak_izi"] = C.girdi_parmak_izi(sorted({s for _, s in kull}))   # Tur-99
        KARNE.write_text(json.dumps(karne, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(ozet_md(karne))
        print(f"karne → {KARNE}")
    else:
        print(ozet_md(json.loads(KARNE.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    _cli()
