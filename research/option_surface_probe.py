"""
option_surface_probe.py — `nihai/ai/options_math.py`'nin GERÇEK opsiyon zinciri üzerinde
İLK CANLI ÖLÇÜMÜ (Tur-32 açık kalemi #1). KARAR-DIŞI: yalnız ölçer ve raporlar.

Hat: DoltHub `option_chain` (gerçek bid/ask + satıcının IV'si + greeks)
  → parite regresyonuyla ÖRTÜK FORWARD ve ÖRTÜK İSKONTO
  → `implied_vol` ile kendi IV'miz  → SATICININ IV'siyle KARŞILAŞTIRMA (bağımsız doğrulama)
  → `svi_fit` + `svi_no_arbitrage` (Durrleman butterfly taraması)
  → `smile_metrics` (25Δ RR/BF)  → `risk_neutral_density`  → `vol_term_structure`

NEDEN PARİTE REGRESYONU: forward'ı spot'tan türetmek temettü/repo/borç maliyetini gözden
kaçırır ve TÜM gülümsemeyi kaydırır. Zincirin kendisi bunu söyler:
    C(K) − P(K) = e^{−rt}·(F − K)
Yani (C−P)'nin K'ya OLS regresyonunda eğim = −e^{−rt}, kesişim = e^{−rt}·F. Böylece F ve r
zincirden ÇIKARILIR; dışarıdan spot/faiz varsayımı GİRMEZ (ve bu, aynı anda paritenin ne
kadar tuttuğunun da ölçüsüdür: R² düşükse zincir tutarsızdır).

Kullanım:
    python -m research.option_surface_probe --symbol SPY --date 2026-08-13
    python -m research.option_surface_probe --symbol AAPL --date 2026-08-13 --min-quotes 8

DÜRÜSTLÜK: buradaki hiçbir çıktı kenar iddiası DEĞİLDİR (ACI DERS #3). Ölçülen şey
"kütüphane gerçek veride ne diyor" sorusudur; P&L ile ilgisi yoktur.
"""
from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np
import pandas as pd

from nihai.ai import options_math as om
from nihai.console import ensure_utf8_stdout

# Sıfır bid'li kotasyon (satılamaz kontrat) ve bid>ask (bozuk kotasyon) elenir. Eleme
# SESSİZ değildir: kaç satırın düştüğü rapora yazılır.
_MIN_BID = 0.01
# Parite regresyonu için K'nın F'ye yakın olması ŞART DEĞİL ama derin ITM/OTM çiftlerde
# makas gürültüsü baskındır; en dar makaslı N çift kullanılır.
_PARITY_PAIRS = 12


def _year_fraction(date: pd.Timestamp, expiry: pd.Timestamp) -> float:
    """ACT/365 (opsiyon piyasasının fiilî konvansiyonu). Aynı gün → 0."""
    return max(0.0, float((expiry - date).days) / 365.0)


def _implied_forward(sub: pd.DataFrame, t: float) -> dict[str, Any] | None:
    """Put-call paritesi REGRESYONU: (C−P) ~ a + b·K → F = −a/b, e^{−rt} = −b.

    Yalnız hem call hem put kotasyonu OLAN strike'lar kullanılır (eşleşmeyen strike parite
    denklemini kuramaz). <3 çift veya bozuk eğim (b ≥ 0) → None."""
    calls = sub[sub["call_put"] == "Call"].set_index("strike")
    puts = sub[sub["call_put"] == "Put"].set_index("strike")
    ortak = sorted(set(calls.index) & set(puts.index))
    if len(ortak) < 3:
        return None
    k = np.asarray(ortak, dtype=float)
    c = calls.loc[ortak, "mid"].to_numpy(dtype=float)
    p = puts.loc[ortak, "mid"].to_numpy(dtype=float)
    genislik = (calls.loc[ortak, "spread"].to_numpy(dtype=float)
                + puts.loc[ortak, "spread"].to_numpy(dtype=float))
    sec = np.argsort(genislik)[:_PARITY_PAIRS]      # en dar makaslı çiftler
    k_s, y_s = k[sec], (c - p)[sec]
    if k_s.size < 3 or np.ptp(k_s) <= 0:
        return None
    b, a = np.polyfit(k_s, y_s, 1)
    if not (math.isfinite(a) and math.isfinite(b)) or b >= 0:
        return None
    disc = -float(b)                                 # e^{−rt}
    fwd = -float(a) / float(b)
    tahmin = a + b * k_s
    ss_tot = float(np.sum((y_s - y_s.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y_s - tahmin) ** 2)) / ss_tot if ss_tot > 0 else None
    r = (-math.log(disc) / t) if (t > 0 and 0 < disc < 5) else None
    return {"forward": fwd, "discount": disc, "r": r, "parity_r2": r2, "n_pairs": int(k_s.size)}


def _slice_report(sub: pd.DataFrame, t: float) -> dict[str, Any] | None:
    """Tek vade dilimi: forward → IV → SVI → gülümseme → risk-nötr yoğunluk."""
    fwd_bilgi = _implied_forward(sub, t)
    if fwd_bilgi is None or t <= 0:
        return None
    fwd, disc = fwd_bilgi["forward"], fwd_bilgi["discount"]
    r = fwd_bilgi["r"] if fwd_bilgi["r"] is not None else 0.0
    # BS'e forward-tutarlı "spot" verilir: S = F·e^{−rt} (q=0 varsayımıyla denk).
    s_eff = fwd * disc

    satirlar = []
    for _, row in sub.iterrows():
        kind = "c" if row["call_put"] == "Call" else "p"
        iv = om.implied_vol(float(row["mid"]), s_eff, float(row["strike"]), t, r, kind)
        satirlar.append({"strike": float(row["strike"]), "kind": kind, "mid": float(row["mid"]),
                         "iv_bizim": iv, "iv_satici": float(row["vol"]) if pd.notna(row["vol"]) else None})
    df = pd.DataFrame(satirlar)
    cozulen = df["iv_bizim"].notna()

    # OTM tarafı kullanılır (piyasa konvansiyonu). GEREKÇE ÖLÇÜLDÜ (2026-08-14, SPY t=0.118y):
    # r'yi −0,07'den +0,04'e taşımak TÜM zincirin ortalama IV'sini 0,180→0,211 oynatıyor, ama
    # bu kayma tamamen DERİN ITM kotasyonlardan geliyor — onlarda vega≈0 olduğu için iskonto
    # faktöründeki %1'lik bir hata IV'de birkaç puanlık saçmalığa dönüşüyor. Yani derin ITM
    # kotasyonlar IV taşımaz; hem SVI hem de satıcı-karşılaştırması OTM'de yapılmalıdır.
    otm = df[cozulen & (((df["kind"] == "c") & (df["strike"] >= fwd))
                        | ((df["kind"] == "p") & (df["strike"] < fwd)))].copy()
    otm = otm.sort_values("strike").drop_duplicates("strike")

    # Bağımsız doğrulama: kendi çözücümüz ↔ veri satıcısının IV kolonu (YALNIZ OTM).
    kars = otm[otm["iv_satici"].notna()]
    fark = (kars["iv_bizim"] - kars["iv_satici"]).abs() if len(kars) else pd.Series(dtype=float)
    otm["k"] = np.log(otm["strike"] / fwd)
    otm["w"] = otm["iv_bizim"] ** 2 * t

    svi = om.svi_fit(otm["k"].to_numpy(), otm["w"].to_numpy()) if len(otm) >= 5 else None
    arb = None
    if svi is not None:
        arb = om.svi_no_arbitrage(svi, (float(otm["k"].min()), float(otm["k"].max())))
    smile = (om.smile_metrics(otm["k"].to_numpy(), otm["iv_bizim"].to_numpy(), fwd, t)
             if len(otm) >= 3 else None)

    # Risk-nötr yoğunluk: call mid'leri (strike'a göre tekil ve artan olmalı).
    cl = df[(df["kind"] == "c")].sort_values("strike").drop_duplicates("strike")
    rnd = (om.risk_neutral_density(cl["strike"].to_numpy(), cl["mid"].to_numpy(), t, r)
           if len(cl) >= 4 else None)

    # r KÖTÜ KOŞULLUDUR: iskonto faktöründeki %0,1'lik bir mid gürültüsü t=0,04y'de r'yi
    # ~2,5 puan oynatır (r = −ln(disc)/t). Negatif çıkması bir HATA değil, kısa vadede
    # ölçülemez olmasıdır — F ise aynı regresyondan çok sağlam gelir (parite-R² ≈ 0,9999).
    r_guvenilir = bool(t >= 0.08 and abs(fwd_bilgi["r"] or 0.0) <= 0.15)
    return {
        "t": t, **fwd_bilgi, "r_guvenilir": r_guvenilir,
        "n_kotasyon": int(len(df)), "n_cozulen": int(cozulen.sum()),
        "iv_fark_ort": float(fark.mean()) if len(fark) else None,
        "iv_fark_max": float(fark.max()) if len(fark) else None,
        "n_karsilastirilan": int(len(kars)),
        "n_otm": int(len(otm)), "svi": svi, "arb": arb, "smile": smile, "rnd": rnd,
        "atm_vol": (smile or {}).get("atm_vol"),
    }


def probe(symbol: str, date: str, min_quotes: int = 6) -> list[dict[str, Any]]:
    """Bir sembol-gün için tüm vade dilimlerinin ölçümü (ağ: DoltHub SQL API)."""
    from nihai.data import dolt_options

    ham = dolt_options.option_chain(symbol, date=date)
    if ham.empty:
        return []
    df = ham.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    onceki = len(df)
    df = df[(df["bid"] >= _MIN_BID) & (df["ask"] > df["bid"])]
    elenen = onceki - len(df)

    cikti = []
    gun = pd.Timestamp(date)
    for exp, sub in df.groupby("expiration"):
        if len(sub) < min_quotes:
            continue
        t = _year_fraction(gun, pd.Timestamp(exp))
        rapor = _slice_report(sub, t)
        if rapor is not None:
            rapor["expiration"] = str(pd.Timestamp(exp).date())
            rapor["elenen_kotasyon"] = elenen
            cikti.append(rapor)
    return sorted(cikti, key=lambda d: d["t"])


def _fmt(x: object, n: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{float(x):.{n}f}" if isinstance(x, (int, float)) else str(x)


def report(symbol: str, date: str, dilimler: list[dict[str, Any]]) -> str:
    if not dilimler:
        return f"[opsiyon yüzeyi] {symbol} @ {date}: kullanılabilir zincir YOK."
    satir = [f"[opsiyon yüzeyi] {symbol} @ {date} — {len(dilimler)} vade dilimi",
             f"  elenen kotasyon (bid<{_MIN_BID} veya bid≥ask): {dilimler[0]['elenen_kotasyon']}"]
    for d in dilimler:
        svi, arb, sm = d["svi"], d["arb"], d["smile"]
        r_not = "" if d["r_guvenilir"] else "  [r kötü-koşullu: kısa vade]"
        satir.append(
            f"  {d['expiration']} (t={d['t']:.4f}y) · kotasyon {d['n_cozulen']}/{d['n_kotasyon']} çözüldü"
            f" · F={_fmt(d['forward'], 2)} r={_fmt(d['r'])} parite-R²={_fmt(d['parity_r2'], 5)}{r_not}")
        satir.append(
            f"      IV bizim↔satıcı (OTM): n={d['n_karsilastirilan']}"
            f" ort|Δ|={_fmt(d['iv_fark_ort'], 5)} max|Δ|={_fmt(d['iv_fark_max'], 5)}")
        if svi:
            satir.append(
                f"      SVI: rmse={_fmt(svi['rmse'], 6)} n={svi['n']}"
                f" (a={_fmt(svi['a'], 4)} b={_fmt(svi['b'], 4)} ρ={_fmt(svi['rho'], 3)}"
                f" m={_fmt(svi['m'], 4)} σ={_fmt(svi['sigma'], 4)})")
        if arb:
            satir.append(
                f"      arbitraj: butterfly-serbest={arb['butterfly_free']}"
                f" · Durrleman-min={_fmt(arb['durrleman_min'], 5)} · kanat-ok={arb['wing_ok']}")
        if sm:
            satir.append(
                f"      gülümseme: ATM={_fmt(sm['atm_vol'])} RR25={_fmt(sm['rr25'])}"
                f" BF25={_fmt(sm['bf25'])} skew={_fmt(sm['skew'], 3)}")
        rnd = d["rnd"]
        if rnd:
            satir.append(
                f"      risk-nötr yoğunluk: kütle={_fmt(rnd['mass'], 4)}"
                f" ort={_fmt(rnd['mean'], 2)} std={_fmt(rnd['std'], 2)}"
                f" çarpıklık={_fmt(rnd['skew'], 3)} negatif-nokta={rnd['n_negative']}")
    ts = [d["t"] for d in dilimler if d["atm_vol"]]
    vs = [d["atm_vol"] for d in dilimler if d["atm_vol"]]
    if len(ts) >= 2:
        term = om.vol_term_structure(ts, vs)
        if term:
            satir.append(f"  vade yapısı: contango={term['contango']}"
                         f" · calendar-serbest={term['calendar_free']}"
                         f" · ihlal={term['n_violations']} · eğim={_fmt(term['slope'])}")
    satir.append("  Not: bu ölçüm yüzeyin TUTARLILIĞINI raporlar; kenar kanıtı DEĞİLDİR.")
    return "\n".join(satir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Opsiyon yüzeyi canlı ölçümü (KARAR-DIŞI)")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (zincir günü)")
    ap.add_argument("--min-quotes", type=int, default=6, help="vade başına asgari kotasyon")
    a = ap.parse_args(argv)
    dilimler = probe(a.symbol.upper(), a.date, min_quotes=a.min_quotes)
    print(report(a.symbol.upper(), a.date, dilimler))
    return 0 if dilimler else 1


if __name__ == "__main__":
    ensure_utf8_stdout()
    raise SystemExit(main())
