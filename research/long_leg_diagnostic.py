"""
long_leg_diagnostic.py — LONG bacağı teşhisi (F4, 2026-07-26 planı).

SORU: canlı `predictions_log.csv`'de LONG isabeti SHORT'tan belirgin düşük
(denetim penceresinde 0,4215 vs 0,4963). Bu, sinyal ailesinin LONG tarafının
gerçekten kötü olduğu anlamına mı geliyor, yoksa ölçüm/uygulama artefaktı mı?

Dört hipotez BİRBİRİNİ DIŞLAR ve her biri AYRI kanıt ister:

  H1 BUG    — `is_long`/işaret normalizasyonu bozuk (Tur-15 dersi: yön `+/-`
              normalizasyonu bir kez sessizce ters dönmüştü). Kanıt: logdaki
              `hit`, `işaret(fwd_price − price) × yön` ile yeniden üretilebiliyor mu?
              Taraflar arasında yeniden-üretim oranı ayrışıyorsa BUG.
  H2 KAYMA  — giriş kayması LONG'ta daha pahalı. Canlı yön, canlı fiyattan
              hesaplanır; giriş de o fiyata yazılır, çıkış ise bar kapanışıdır
              → `hit` "yön becerisi" ile "kovalama maliyetini" karıştırır (F1,
              2026-07-27). Kanıt: açık CANLI girişte var, BAR girişinde yok.
  H3 REJİM  — ölçüm penceresinde piyasa yönü LONG aleyhineydi; taraflar farklı
              GÜNLERE düştüğü için karşılaştırma elmayla armut. Kanıt: ham açık
              büyük ama GÜN-EŞLEŞTİRMELİ açık ≈ 0.
  H4 YAPISAL— açık, gün eşleştirildikten ve kayma çıkarıldıktan sonra da duruyor.

YÖNTEM — gün-eşleştirmeli (paired) küme testi. Aynı sembol×gün kümesinde hem LONG
hem SHORT tahmin varsa, o kümenin LONG ortalaması ile SHORT ortalaması FARKI tek
gözlem sayılır. Böylece:
  * piyasa yönü/gün etkisi tasarım gereği düşer (H3 doğrudan test edilir),
  * 1m frekansta üst üste binen pencerelerin şişirdiği t engellenir (ACI DERS,
    `horizon_sweep._cluster_t` ile aynı gerekçe).

Puanlama F1 harness'ının AYNISIDIR (`horizon_sweep.sweep`) — taraflar ayrı ayrı
süpürülür, böylece giriş kayması da taraf bazında ölçülür. Ayrı bir puanlayıcı
yazılmaz (iki uygulamanın sessizce ayrışması bu projede daha önce yaşandı).

Kullanım (repo kökünden):
    python -m research.long_leg_diagnostic
    python -m research.long_leg_diagnostic --horizon 5 --exec-lags 0,1 \
        --spread-model corwin --json data/long_leg_2026-07-27.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nihai.config import DATA_DIR  # noqa: E402
from nihai.console import ensure_utf8_stdout
from research import horizon_sweep as hs  # noqa: E402

_MIN_KUME = 20          # eşleşmiş küme sayısı bunun altındaysa hüküm YOK
_ESIK_PUAN = 0.005      # 0,5 isabet puanı — "açık kapandı" sayılma sınırı


# ---------------------------------------------------------------------------
# H1 — normalizasyon denetimi (bar verisi GEREKTİRMEZ)
# ---------------------------------------------------------------------------
def normalizasyon_denetimi(log_path: str | None = None) -> dict:
    """Logdaki `hit`, kendi `price`/`fwd_price`/`direction` kolonlarından üretilebiliyor mu?

    Bu, bar verisinden BAĞIMSIZ bir iç tutarlılık testidir: `hit` alanını yazan kod
    ile yönü uygulayan kod aynı işaret sözleşmesini kullanıyor mu? Taraflar arasında
    uyum oranı ayrışıyorsa (örn. LONG %94, SHORT %100) hata yön normalizasyonundadır."""
    p = log_path or str(DATA_DIR / "predictions_log.csv")
    df = pd.read_csv(p)
    df = df[df["direction"].isin(["+", "-"])].copy()
    for c in ("price", "fwd_price", "hit", "fwd_ret_pct"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    df = df[df["hit"].notna() & df["fwd_price"].notna() & (df["price"] > 0)]
    if df.empty:
        return {"durum": "VERI_YOK"}
    sgn = np.where(df["direction"].to_numpy() == "+", 1.0, -1.0)
    ham = (df["fwd_price"].to_numpy(dtype=float) / df["price"].to_numpy(dtype=float)) - 1.0
    beklenen = (ham * sgn > 0).astype(float)
    uyum = beklenen == df["hit"].to_numpy(dtype=float)
    yon = np.where(sgn > 0, "LONG", "SHORT")

    out: dict[str, Any] = {"durum": "OLCULDU", "n": int(len(df)), "taraf": {}}
    for t in ("LONG", "SHORT"):
        m = yon == t
        out["taraf"][t] = {"n": int(m.sum()),
                           "hit_yeniden_uretim_orani": round(float(uyum[m].mean()), 5),
                           "kayitli_isabet": round(float(df["hit"].to_numpy()[m].mean()), 4)}
    # `fwd_ret_pct` işaret sözleşmesi: işaretli (yönle çarpılmış) olmalı
    if df["fwd_ret_pct"].notna().any():
        fark = np.abs(ham * sgn * 100 - df["fwd_ret_pct"].to_numpy(dtype=float))
        out["fwd_ret_pct_isaretli_mi"] = {
            "medyan_mutlak_fark_puan": round(float(np.nanmedian(fark)), 6),
            "sozlesme": "işaretli (yön ile çarpılmış)" if float(np.nanmedian(fark)) < 0.01
                        else "İŞARETSİZ/TUTARSIZ — incele"}
    oranlar = [v["hit_yeniden_uretim_orani"] for v in out["taraf"].values()]
    ayrisma = (max(oranlar) - min(oranlar)) if oranlar else 0.0
    out["taraf_ayrismasi"] = round(float(ayrisma), 5)
    out["H1_BUG"] = bool(min(oranlar) < 0.99 or ayrisma > 0.01) if oranlar else False
    out["NOT"] = ("Yeniden-üretim oranı ~1,0 ve taraflar arası ayrışma ~0 ise yön "
                  "normalizasyonu SAĞLAM; LONG açığı bir kodlama hatası DEĞİLDİR.")
    return out


# ---------------------------------------------------------------------------
# İstatistik — gün-eşleştirmeli küme farkı
# ---------------------------------------------------------------------------
def _kume_anahtari(sub: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(sub["ts"])
    return (sub["symbol"].astype(str) + "|" + ts.dt.date.astype(str)).to_numpy()


def eslestirilmis_fark(a: pd.DataFrame, b: pd.DataFrame, kolon: str = "signed") -> dict:
    """(a − b) farkının sembol×gün kümelerinde EŞLEŞTİRİLMİŞ t testi.

    Yalnız İKİ tarafta da gözlem bulunan kümeler kullanılır; her küme tek gözlemdir.
    Gün etkisi (piyasa rejimi) fark alınırken düşer → H3 bu testle elenir."""
    if a.empty or b.empty:
        return {"n_kume": 0}
    sa = pd.Series(a[kolon].to_numpy(dtype=float)).groupby(pd.Series(_kume_anahtari(a))).mean()
    sb = pd.Series(b[kolon].to_numpy(dtype=float)).groupby(pd.Series(_kume_anahtari(b))).mean()
    ortak = sa.index.intersection(sb.index)
    if len(ortak) < 3:
        return {"n_kume": int(len(ortak))}
    d = (sa.loc[ortak] - sb.loc[ortak]).to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    k = int(d.size)
    sd = float(d.std(ddof=1)) if k > 1 else 0.0
    return {"n_kume": k,
            "ort_fark_pct": round(float(d.mean()) * 100, 5),
            "t": round(float(d.mean() / (sd / np.sqrt(k))), 3) if sd > 0 else None,
            "kume_kapsami": round(float(len(ortak) / max(1, len(sa.index.union(sb.index)))), 3)}


def _karne(long: pd.DataFrame, giris: str, h: int, lag: int) -> dict:
    """Tek hücrenin karnesi (`horizon_sweep._stats` ile birebir aynı ölçüler)."""
    sub = long[(long["giris"] == giris) & (long["H"] == h) & (long["lag"] == lag)]
    if sub.empty:
        return {"n": 0}
    return hs._stats(sub["signed"].to_numpy(), sub["net"].to_numpy(), _kume_anahtari(sub))


def _hucre(long: pd.DataFrame, giris: str, h: int, lag: int) -> pd.DataFrame:
    return long[(long["giris"] == giris) & (long["H"] == h) & (long["lag"] == lag)]


# ---------------------------------------------------------------------------
# H3 — piyasa rejimi (barlardan; tahmin getirilerinden DEĞİL)
# ---------------------------------------------------------------------------
def gunluk_piyasa(bars: dict[str, pd.DataFrame]) -> pd.Series:
    """Gün → kesitsel ortalama günlük getiri (piyasa yönü vekili).

    Tahminlerin ileri getirilerinden DEĞİL, barların kendisinden hesaplanır: tahmin
    getirileri 5 dakikalık mikro-getiriler olduğu için ortalamaları neredeyse her gün
    negatiftir ve rejim vekili olarak kullanılamaz (yanıltıcı %94 'düşüş günü')."""
    seriler = []
    for df in bars.values():
        if df is None or df.empty:
            continue
        g = pd.Series(df["close"].to_numpy(dtype=float),
                      index=pd.to_datetime(pd.Series(df.index)).dt.date.to_numpy())
        son = g.groupby(level=0).last()
        seriler.append(son.pct_change())
    if not seriler:
        return pd.Series(dtype=float)
    return pd.concat(seriler, axis=1).mean(axis=1, skipna=True)


def rejim_bolmesi(a: pd.DataFrame, b: pd.DataFrame, piyasa: pd.Series) -> dict:
    """Yükseliş / düşüş günlerinde taraf karneleri (H3'ün betimleyici ayağı)."""
    out: dict[str, Any] = {}
    if piyasa.empty:
        return out
    for etiket, maske in (("YUKARI", piyasa > 0), ("ASAGI", piyasa <= 0)):
        gunler = set(piyasa.index[maske])
        aa = a[pd.to_datetime(a["ts"]).dt.date.isin(gunler)]
        bb = b[pd.to_datetime(b["ts"]).dt.date.isin(gunler)]
        if aa.empty or bb.empty:
            continue
        out[etiket] = {
            "n_gun": int(len(gunler)),
            "LONG": {"n": int(len(aa)), "isabet": round(float((aa["signed"] > 0).mean()), 4)},
            "SHORT": {"n": int(len(bb)), "isabet": round(float((bb["signed"] > 0).mean()), 4)},
            "eslestirilmis": eslestirilmis_fark(aa, bb),
        }
    return out


# ---------------------------------------------------------------------------
# Koşu
# ---------------------------------------------------------------------------
def run(horizon: int = 5, lags: list[int] | None = None, spread_model: str = "corwin",
        cost_bps: float = 0.0, log_path: str | None = None, max_symbols: int = 0,
        cache_days: int = 30, grid: str = "rth") -> dict:
    lags = lags or [0, 1]
    norm = normalizasyon_denetimi(log_path)

    preds = hs.load_predictions(log_path)
    if preds.empty:
        return {"normalizasyon": norm, "hata": "tahmin logu boş"}
    syms = sorted(preds["symbol"].astype(str).unique())
    if max_symbols:
        syms = syms[:max_symbols]
        preds = preds[preds["symbol"].astype(str).isin(syms)]

    from datetime import UTC, timedelta
    start = preds["ts"].min().to_pydatetime().replace(tzinfo=UTC) - timedelta(days=1)
    end = (preds["ts"].max().to_pydatetime().replace(tzinfo=UTC)
           + timedelta(days=max(3, int(horizon / 390) + 2)))
    bars = hs.fetch_bars(syms, start, end, "1m", cache_days, grid)
    if not bars:
        return {"normalizasyon": norm, "hata": "hiç bar çekilemedi (ALPACA anahtarı? SIP planı?)"}

    cost_frac: dict[str, float] = {"_default": 2.0 * cost_bps / 1e4}
    if spread_model in ("corwin", "roll", "max"):
        from research.spread_estimate import estimate
        for sym, df in bars.items():
            cost_frac[sym] = float(estimate(df, 1.0).get(f"{spread_model}_bps", 0.0)) / 1e4

    # Taraflar AYRI süpürülür → giriş kayması da taraf bazında ölçülür.
    parca: dict[str, pd.DataFrame] = {}
    kayma: dict[str, Any] = {}
    for etiket, m in (("LONG", preds["sign"] > 0), ("SHORT", preds["sign"] < 0)):
        long_i, diag_i = hs.sweep(preds[m], bars, [horizon], lags, cost_frac)
        parca[etiket] = long_i
        kayma[etiket] = diag_i.get("giris_kaymasi", {})
    if any(v.empty for v in parca.values()):
        return {"normalizasyon": norm, "hata": "bir tarafta hiç eşleşen tahmin yok"}

    piyasa = gunluk_piyasa(bars)
    taraflar: dict[str, Any] = {}
    aciklar: dict[str, Any] = {}
    for giris in ("canli", "bar"):
        for lag in lags:
            if giris == "canli" and lag != 0:
                continue                      # canlı fiyat yalnız karar anına aittir
            anahtar = f"{giris}_lag{lag}"
            a = _hucre(parca["LONG"], giris, horizon, lag)
            b = _hucre(parca["SHORT"], giris, horizon, lag)
            if a.empty or b.empty:
                continue
            taraflar[anahtar] = {"LONG": _karne(parca["LONG"], giris, horizon, lag),
                                 "SHORT": _karne(parca["SHORT"], giris, horizon, lag)}
            ham_acik = float((b["signed"] > 0).mean() - (a["signed"] > 0).mean())
            aciklar[anahtar] = {
                "ham_isabet_acigi_SHORT_eksi_LONG": round(ham_acik, 4),
                "gun_eslestirilmis_getiri_farki_LONG_eksi_SHORT": eslestirilmis_fark(a, b),
            }

    ana = "canli_lag0" if "canli_lag0" in aciklar else next(iter(aciklar), "")
    bar_anahtar = f"bar_lag{lags[0]}" if f"bar_lag{lags[0]}" in aciklar else ""
    verdikt, gerekce = _verdikt(norm, aciklar, kayma, ana, bar_anahtar)

    return {
        "parametre": {"horizon": horizon, "gecikmeler": lags, "spread_model": spread_model,
                      "sembol": len(bars), "n_tahmin": int(len(preds))},
        "H1_normalizasyon": norm,
        "H2_giris_kaymasi_bps": {k: {x: v.get(x) for x in ("ort_bps", "medyan_bps", "t", "n")}
                                 for k, v in kayma.items()},
        "taraf_karneleri": taraflar,
        "ACIKLAR": aciklar,
        "H3_rejim_bolmesi": rejim_bolmesi(_hucre(parca["LONG"], "bar", horizon, lags[0]),
                                          _hucre(parca["SHORT"], "bar", horizon, lags[0]), piyasa),
        "VERDIKT": verdikt,
        "GEREKCE": gerekce,
        "UYARI": ("Bu teşhis LONG ile SHORT'u KARŞILAŞTIRIR; hiçbir tarafın KÂRLI olduğunu "
                  "göstermez. Her iki taraf da başabaşın altındaysa 'açık yok' sonucu "
                  "'kenar var' anlamına GELMEZ (ACI DERS #3)."),
    }


def _verdikt(norm: dict, aciklar: dict, kayma: dict, ana: str, bar_anahtar: str,
             ) -> tuple[str, list[str]]:
    """Kanıtları hipotezlere bağla. Örneklem yetersizse hüküm verme (uydurma yok)."""
    g: list[str] = []
    if norm.get("H1_BUG"):
        g.append(f"Yön normalizasyonu tutarsız (taraf ayrışması {norm.get('taraf_ayrismasi')}).")
        return "H1 BUG — yön/işaret normalizasyonu bozuk, önce bunu düzelt", g
    g.append(f"`hit` yeniden-üretimi iki tarafta da ~1,0 (ayrışma {norm.get('taraf_ayrismasi')}) "
             "→ H1 (bug) ELENDİ.")

    a_ana = aciklar.get(ana, {})
    a_bar = aciklar.get(bar_anahtar, {}) if bar_anahtar else {}
    ham_ana = a_ana.get("ham_isabet_acigi_SHORT_eksi_LONG")
    ham_bar = a_bar.get("ham_isabet_acigi_SHORT_eksi_LONG")
    esl_ana = (a_ana.get("gun_eslestirilmis_getiri_farki_LONG_eksi_SHORT") or {})
    esl_bar = (a_bar.get("gun_eslestirilmis_getiri_farki_LONG_eksi_SHORT") or {})

    kl = (kayma.get("LONG") or {}).get("ort_bps")
    ks = (kayma.get("SHORT") or {}).get("ort_bps")
    if kl is not None and ks is not None:
        g.append(f"Giriş kayması LONG {kl} bps / SHORT {ks} bps "
                 f"(fark {round(float(kl) - float(ks), 3)} bps).")

    if esl_ana.get("n_kume", 0) < _MIN_KUME:
        return "HÜKÜM YOK — eşleşmiş küme sayısı yetersiz", g

    t_ana, t_bar = esl_ana.get("t"), esl_bar.get("t")
    if ham_ana is not None:
        g.append(f"Ham isabet açığı (SHORT−LONG): canlı girişte {ham_ana}"
                 + (f", bar girişinde {ham_bar}" if ham_bar is not None else ""))
    g.append(f"Gün-eşleştirilmiş getiri farkı (LONG−SHORT): canlı t={t_ana}"
             + (f", bar t={t_bar}" if t_bar is not None else "")
             + f" ({esl_ana.get('n_kume')} küme)")

    anlamli_ana = t_ana is not None and abs(float(t_ana)) > 2.0
    anlamli_bar = t_bar is not None and abs(float(t_bar)) > 2.0

    if not anlamli_ana:
        return ("H3 REJİM — ham açık gün kompozisyonundan geliyor; aynı sembol×günde "
                "taraflar arasında anlamlı fark YOK", g)
    if anlamli_ana and not anlamli_bar:
        return ("H2 KAYMA — açık yalnız CANLI girişte var, bar-kapanışı girişinde kayboluyor; "
                "LONG bacağının sorunu tahmin değil UYGULAMA maliyeti", g)
    if ham_bar is not None and abs(float(ham_bar)) < _ESIK_PUAN:
        return ("H2 KAYMA — bar girişinde isabet açığı 0,5 puanın altına iniyor "
                "(kalan fark maddi değil)", g)
    return ("H4 YAPISAL — açık gün eşleştirmesinden ve kayma düzeltmesinden SONRA da duruyor; "
            "LONG tarafı gerçekten daha kötü", g)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LONG bacağı teşhisi (F4)")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--exec-lags", default="0,1")
    ap.add_argument("--spread-model", default="corwin",
                    choices=["none", "corwin", "roll", "max", "fixed_bps"])
    ap.add_argument("--cost-bps", type=float, default=0.0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--cache-days", type=int, default=30)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = run(horizon=a.horizon, lags=[int(x) for x in a.exec_lags.split(",") if x.strip()],
              spread_model=a.spread_model, cost_bps=a.cost_bps, log_path=a.log,
              max_symbols=a.max_symbols, cache_days=a.cache_days)
    txt = json.dumps(res, ensure_ascii=False, indent=2, default=str)
    print(txt)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            f.write(txt)
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    raise SystemExit(main())
