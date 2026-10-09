"""Ön-kayıtlı ileri sınama: kripto `ma10_50` (Quant AI alfa lab) ↔ `event_breakout` (Nihai, Tur-96 §9 madde 1).

Kurallar, evren, maliyet ve karar ölçütleri `docs/ONKAYIT_KRIPTO_YARIS_2026-09-25.md`'de DONDURULDU. Bu dosya
yalnız ölçer; hiçbir parametre sonuç görüldükten sonra değişmez.

    python -m research.ileri_kripto_yaris kilitle    # BİR KEZ: veri indir, geçmiş parmak izi + ön-kayıt özeti → kilit
    python -m research.ileri_kripto_yaris guncelle   # veri yenile → kilidi doğrula → deftere yeni ileri günleri EKLE → rapor
    python -m research.ileri_kripto_yaris rapor      # yalnız defterden rapor

Kilit DAVRANIŞSALDIR: motor kodu değişse bile event_breakout'un 2018-01-01 → 2026-09-24 getirileri birebir aynı
çıkıyorsa sınama sürer; tek bir gün farklıysa (kod ya da veri revizyonu) `guncelle` deftere YAZMAZ ve farkı basar.
Defter yalnız sona eklenir; yazılmış satır yeniden hesaplanıp karşılaştırılır, asla üzerine yazılmaz.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KOK = ROOT / "data" / "ileri_kripto_yaris"
DEFTER = KOK / "defter.csv"
KILIT = ROOT / "research" / "ileri_kripto_yaris_kilit.json"
ONKAYIT = ROOT / "docs" / "ONKAYIT_KRIPTO_YARIS_2026-09-25.md"

COINLER = ("BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BNB-USD", "ADA-USD", "DOGE-USD", "SOL-USD")
STRATEJILER = ("al_tut", "ma10_50", "ma10_50_g1", "event_breakout")
KAYIT_GUNU = pd.Timestamp("2026-09-25")      # ileri getiri = bu günden SONRA gerçekleşen (ilki 2026-09-26)
IZ_BAS = pd.Timestamp("2018-01-01")          # geçmiş parmak izi penceresi: IZ_BAS … KAYIT_GUNU − 1 gün
SON_KARAR = pd.Timestamp("2029-09-25")       # birincil karar günü (3 yıl); öncesi yalnız ARA rapor
MALIYET, GUN, TOL = 0.001, 365, 1e-9


# ── strateji getirileri ────────────────────────────────────────────────────────────────────────────
def ma10_50(close: pd.Series, gecikme: int = 0) -> pd.Series:
    """SMA10 > SMA50 → uzun, değilse nakit; t kapanışındaki sinyal (t+gecikme)→(t+gecikme+1) getirisini kazanır."""
    poz = (close.rolling(10).mean() > close.rolling(50).mean()).astype(float).shift(1 + gecikme).fillna(0.0)
    return poz * close.pct_change() - MALIYET * poz.diff().abs().fillna(0.0)


def _veri_hazirla(indir: bool) -> None:
    """Sınamaya ÖZEL önbellek (Tur-96 önbelleği değişmez); VIX de event_breakout motoru için yenilenir."""
    from research import coklu_piyasa as K
    from research import piyasalar as P
    P.CACHE_DIR = KOK / "veri"
    if indir:
        for s in (*COINLER, "^VIX"):
            if P.indir(s).empty:
                raise SystemExit(f"{s} indirilemedi — ağ/yfinance")
    K._isci_hazirla()


def getiriler(bugun: pd.Timestamp) -> pd.DataFrame:
    """Uzun biçim [tarih, coin, strateji, getiri]; yalnız TAMAMLANMIŞ barlar (tarih < bugün UTC)."""
    from research import coklu_piyasa as K
    parcalar = []
    for c in COINLER:
        df = K.hazirla(c)
        df = df[df.index < bugun]
        t0 = max(K.SICAK, int(np.searchsorted(df.index.values, np.datetime64(K.TEST_BAS))))
        eb = K._algoritma_kos(c, df, t0, "event_breakout")
        close = df["close"]
        g = pd.DataFrame({"al_tut": close.pct_change(), "ma10_50": ma10_50(close), "ma10_50_g1": ma10_50(close, 1),
                          "event_breakout": pd.Series(eb["r"].astype(float), pd.DatetimeIndex(eb["tarih"]))}).dropna()
        g = g[g.index >= IZ_BAS].rename_axis("tarih").reset_index().melt("tarih", var_name="strateji", value_name="getiri")
        parcalar.append(g.assign(coin=c))
    return pd.concat(parcalar, ignore_index=True)[["tarih", "coin", "strateji", "getiri"]]


# ── kilit ──────────────────────────────────────────────────────────────────────────────────────────
def parmak_izi(s: pd.Series) -> dict:
    """Tarih + 10 haneye yuvarlanmış getiri → sha256; farkın YILINI bulmak için yıllık toplamlar."""
    s = s.sort_index()
    h = hashlib.sha256(s.index.values.astype("datetime64[D]").astype(np.int64).tobytes()
                       + np.round(s.to_numpy(float), 10).tobytes()).hexdigest()
    return {"n": int(s.size), "sha": h, "yil": {str(y): round(float(v), 12) for y, v in s.groupby(s.index.year).sum().items()}}


def _gecmis(R: pd.DataFrame) -> dict:
    g = R[R.tarih < KAYIT_GUNU]
    return {c: {st: parmak_izi(x.set_index("tarih")["getiri"]) for st, x in g[g.coin == c].groupby("strateji")}
            for c in COINLER}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dogrula(R: pd.DataFrame, kilit: dict, defter: pd.DataFrame, onkayit_sha: str) -> list[str]:
    """Ön-kayıt belgesi, geçmiş parmak izi ve deftere yazılmış satırlar yeniden hesaplamayla birebir mi?"""
    sorun = [] if onkayit_sha == kilit["onkayit_sha256"] else ["ön-kayıt belgesi kilitten sonra DEĞİŞTİ"]
    simdi = _gecmis(R)
    for c, ss in kilit["parmak_izi"].items():
        for st, iz in ss.items():
            yeni = simdi.get(c, {}).get(st)
            if yeni is not None and yeni["sha"] == iz["sha"]:
                continue
            # sha, 10. hanede yuvarlama sınırına düşen kayan nokta gürültüsüyle de değişebilir → yıllık toplamlar hakem
            yillar = [y for y, v in iz["yil"].items() if yeni is None or not abs(yeni["yil"].get(y, np.nan) - v) <= TOL]
            if yeni is None or yillar or yeni["n"] != iz["n"]:
                sorun.append(f"{c}/{st}: geçmiş getiriler değişti (yıllar {yillar or 'yalnız n'}) → motor ya da veri revizyonu")
    if not defter.empty:
        k = defter.merge(R, on=["tarih", "coin", "strateji"], how="left", suffixes=("", "_yeni"))
        kotu = k[~((k.getiri - k.getiri_yeni).abs() <= TOL)]
        if len(kotu):
            sorun.append(f"defterdeki {len(kotu)} satır yeniden hesaplamayla uyuşmuyor (ilk: {kotu.iloc[0].tarih.date()} "
                         f"{kotu.iloc[0].coin}/{kotu.iloc[0].strateji})")
    return sorun


def yeni_satirlar(R: pd.DataFrame, defter: pd.DataFrame) -> pd.DataFrame:
    """Kayıt gününden SONRA ve defterin son gününden sonra gerçekleşen, 8 coinin hepsi için hazır günler."""
    son = max(KAYIT_GUNU, defter.tarih.max()) if not defter.empty else KAYIT_GUNU
    y = R[R.tarih > son]
    tam = y.groupby("tarih").coin.nunique()
    return y[y.tarih.isin(tam[tam == len(COINLER)].index)].sort_values(["tarih", "coin", "strateji"])


def defter_oku() -> pd.DataFrame:
    if not DEFTER.exists():
        return pd.DataFrame(columns=["tarih", "coin", "strateji", "getiri"]).astype({"tarih": "datetime64[ns]", "getiri": float})
    return pd.read_csv(DEFTER, parse_dates=["tarih"])


# ── ölçüm ──────────────────────────────────────────────────────────────────────────────────────────
def _sharpe(r: np.ndarray) -> np.ndarray:
    sd = r.std(axis=-1, ddof=1)
    return np.where(sd > 0, r.mean(axis=-1) / np.where(sd > 0, sd, 1) * np.sqrt(GUN), np.nan)


def sharpe_farki(a: np.ndarray, b: np.ndarray, blok: int = 20, n: int = 2000, seed: int = 0) -> dict:
    """Eşleştirilmiş dairesel blok bootstrap: Sharpe(a) − Sharpe(b), %90 aralık, P(fark ≤ 0)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    T = a.size
    bas = np.random.default_rng(seed).integers(0, T, size=(n, -(-T // blok)))
    idx = ((bas[:, :, None] + np.arange(blok)) % T).reshape(n, -1)[:, :T]
    d = _sharpe(a[idx]) - _sharpe(b[idx])
    return {"fark": float(_sharpe(a) - _sharpe(b)), "q05": float(np.nanquantile(d, 0.05)),
            "q95": float(np.nanquantile(d, 0.95)), "p_le0": float(np.nanmean(d <= 0))}


def ew(defter: pd.DataFrame) -> pd.DataFrame:
    """Eşit ağırlıklı 8 coin portföyü: günlük ortalama getiri, sütun = strateji."""
    return defter.pivot_table(index="tarih", columns="strateji", values="getiri", aggfunc="mean").sort_index()


def rapor(defter: pd.DataFrame, bugun: pd.Timestamp) -> str:
    from research.coklu_piyasa import nw_alfa
    if defter.empty:
        return "Defter boş: ilk ileri gün 2026-09-26 (karar 2026-09-25 kapanışı) tamamlanınca yazılır."
    E = ew(defter)
    yil = len(E) / GUN
    satir = [f"İLERİ DÖNEM {E.index[0].date()} → {E.index[-1].date()} · {len(E)} gün ({yil:.2f} yıl) · EW-8, %0,1/taraf",
             f"{'strateji':<16}{'Sharpe':>8}{'yıllık':>9}{'MDD':>7}{'alfa t':>8}"]
    alfa = {}
    for st in STRATEJILER:
        r = E[st]
        eq = (1 + r).cumprod()
        a = nw_alfa(r.to_numpy(), E["al_tut"].to_numpy(), GUN) if st != "al_tut" else None
        alfa[st] = a["alfa_t"] if a else None
        satir.append(f"{st:<16}{float(_sharpe(r.to_numpy())):>8.2f}{eq.iloc[-1] ** (1 / yil) - 1:>9.1%}"
                     f"{float((1 - eq / eq.cummax()).max()):>7.1%}{alfa[st] if alfa[st] is not None else float('nan'):>8.2f}")
    if len(E) < 60:
        return "\n".join(satir + ["(60 günden kısa: fark testi ve alfa anlamsız, yalnız betimsel)"])
    d = sharpe_farki(E["ma10_50"].to_numpy(), E["event_breakout"].to_numpy())
    satir.append(f"H1 Sharpe(ma10_50) − Sharpe(event_breakout) = {d['fark']:+.2f} [{d['q05']:+.2f}, {d['q95']:+.2f}] "
                 f"P(≤0)={d['p_le0']:.3f}")
    if bugun < SON_KARAR:
        satir.append(f"ARA RAPOR — ön-kayıt gereği karar YOK (birincil karar {SON_KARAR.date()}).")
        return "\n".join(satir)
    h1 = ("ma10_50 KAZANDI" if d["p_le0"] < 0.05 else "event_breakout KAZANDI" if d["p_le0"] > 0.95
          else "AYIRT EDİLEMEDİ")
    satir.append(f"KARAR H1: {h1}")
    for st in ("ma10_50", "event_breakout"):
        t = alfa[st]
        h2 = ("ölçülemedi" if t is None else "KENAR TEYİT (alfa t > 2)" if t > 2
              else "tutarlı, kanıtlanmadı (0 < t ≤ 2)" if t > 0 else "ÇÖKTÜ (alfa t ≤ 0)")
        satir.append(f"KARAR H2 {st}: {h2}")
    return "\n".join(satir)


# ── komutlar ───────────────────────────────────────────────────────────────────────────────────────
def _bugun() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)


def kilitle() -> None:
    if KILIT.exists():
        raise SystemExit(f"kilit zaten var ({KILIT}); ön-kayıt YENİDEN kilitlenmez")
    if _bugun() > KAYIT_GUNU:
        raise SystemExit("kayıt günü geçti: kilit ancak 2026-09-25 (UTC) içinde oluşturulabilir")
    _veri_hazirla(indir=True)
    R = getiriler(_bugun())
    kilit = {"kayit_gunu": str(KAYIT_GUNU.date()), "olusturma_utc": pd.Timestamp.now(tz="UTC").isoformat(),
             "onkayit": ONKAYIT.name, "onkayit_sha256": _sha(ONKAYIT), "parmak_izi": _gecmis(R),
             "ortam": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}}
    KILIT.write_text(json.dumps(kilit, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"kilit yazıldı: {KILIT} · ön-kayıt sha256 {kilit['onkayit_sha256'][:16]}…")


def guncelle() -> int:
    kilit = json.loads(KILIT.read_text(encoding="utf-8"))
    _veri_hazirla(indir=True)
    bugun = _bugun()
    R, defter = getiriler(bugun), defter_oku()
    sorun = dogrula(R, kilit, defter, _sha(ONKAYIT))
    if sorun:
        print("KİLİT BOZUK: deftere yazılmadı\n  - " + "\n  - ".join(sorun))
        return 1
    yeni = yeni_satirlar(R, defter)
    if len(yeni):
        KOK.mkdir(parents=True, exist_ok=True)
        yeni.to_csv(DEFTER, mode="a", header=not DEFTER.exists(), index=False, float_format="%.12g")
        defter = defter_oku()
    print(f"kilit doğrulandı · {yeni.tarih.nunique()} yeni gün eklendi\n{rapor(defter, bugun)}")
    return 0


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    komut =sys.argv[1] if len(sys.argv) > 1 else "rapor"
    if komut == "kilitle":
        kilitle()
    elif komut == "guncelle":
        raise SystemExit(guncelle())
    else:
        print(rapor(defter_oku(), _bugun()))
