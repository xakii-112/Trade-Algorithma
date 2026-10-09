"""
realized_vol_probe.py — `nihai/ai/microstructure.py`'nin GERÇEK barlar üzerinde İLK CANLI
ÖLÇÜMÜ (Tur-32 açık kalemi #4). KARAR-DIŞI: yalnız ölçer ve raporlar.

Hat: Alpaca saatlik barlar → GÜNLÜK gerçekleşen varyans (RV) + sıçrama ayrışması
  → **YÜRÜYEN (walk-forward) HAR-RV** → ÜÇ TABANA karşı OOS beceri ölçümü
  → mikroyapı teşhisi (oynaklık imzası, Roll makası, Amihud)

KANIT DİSİPLİNİ (bu dosyanın var oluş sebebi): "HAR R²=0,98" cümlesi HİÇBİR ŞEY söylemez —
gerçekleşen oynaklık son derece kalıcıdır, "dünkü RV"yi tekrarlayan TRİVİYAL kural bile
yüksek R² alır. Bu yüzden model ÜÇ tabana karşı puanlanır:
  RW      : σ̂²_{t+1} = RV_t            (rastgele yürüyüş — asıl rakip budur)
  MA22    : son 22 günün ortalaması
  SABİT   : eğitim penceresinin ortalaması
Beceri = 1 − kayıp_model/kayıp_taban (pozitif = model tabanı geçiyor). ACI DERS #9'un
oynaklık eksenindeki karşılığı.

Kayıp ölçüsü **QLIKE** (Patton 2011): RV/σ̂² − ln(RV/σ̂²) − 1. MSE oynaklık tahmininde
yanıltıcıdır (birkaç kriz günü toplamı ele geçirir); QLIKE gürültülü RV vekiliyle bile
tutarlı sıralama verir. MSE de yan yana raporlanır.

DÜRÜSTLÜK: oynaklığın öngörülebilir olması bir KENAR DEĞİLDİR (ACI DERS #3). Oynaklık
finansın en öngörülebilir büyüklüğüdür ve bu, YÖN hakkında hiçbir şey söylemez. Bu ölçüm
boyutlandırma/risk tarafı içindir, sinyal tarafı için değil.

Kullanım:
    python -m research.realized_vol_probe --symbol SPY --start 2024-08-01
    python -m research.realized_vol_probe --symbol QQQ --start 2024-01-01 --refit 10
"""
from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np
import pandas as pd

from nihai.ai import microstructure as ms
from nihai.console import ensure_utf8_stdout

_MIN_BARS_PER_DAY = 3       # bu kadar bar görmeyen gün RV üretmez (yarım seans/tatil)
_MIN_TRAIN = 120            # HAR'ın 22-günlük penceresi + makul örneklem
_FLOOR = 1e-12              # QLIKE'ta log(0) koruması
# CLI'dan gelen serbest metin `alpaca_history.get_history`'nin Literal imzasına girmeden
# ÖNCE doğrulanır (yanlış aralık sessizce sağlayıcıya geçmesin).
_GECERLI_ARALIK = frozenset({"1m", "5m", "15m", "1h", "1d"})


def daily_rv(bars: pd.DataFrame) -> pd.DataFrame:
    """Bar-içi log getirilerden GÜNLÜK RV + BV + sıçrama bileşeni.

    RV = Σ r², BV = sıçramaya dayanıklı ikili-güç varyasyonu, J = max(RV−BV, 0).
    Look-ahead yok: her gün YALNIZ o günün barlarını kullanır."""
    if bars.empty or "close" not in bars:
        return pd.DataFrame()
    r = np.log(bars["close"]).diff().dropna()
    satir = []
    for gun, seri in r.groupby(pd.Grouper(freq="1D")):
        if len(seri) < _MIN_BARS_PER_DAY:
            continue
        arr = seri.to_numpy(dtype=float)
        rv = ms.realized_variance(arr)
        bv = ms.bipower_variation(arr)
        if rv is None or rv <= 0:
            continue
        jump = max(rv - bv, 0.0) if bv is not None else 0.0
        satir.append({"date": gun, "rv": rv, "bv": bv, "jump": jump, "n_bar": len(arr)})
    return pd.DataFrame(satir).set_index("date") if satir else pd.DataFrame()


def _qlike(gercek: np.ndarray, tahmin: np.ndarray) -> float:
    """QLIKE = ort(RV/σ̂² − ln(RV/σ̂²) − 1). Küçük = iyi; kusursuz tahminde 0."""
    f = np.maximum(tahmin, _FLOOR)
    oran = np.maximum(gercek, _FLOOR) / f
    return float(np.mean(oran - np.log(oran) - 1.0))


def walk_forward(rv: np.ndarray, min_train: int = _MIN_TRAIN, refit: int = 20,
                 log_target: bool = True) -> dict[str, Any] | None:
    """YÜRÜYEN OOS değerlendirme: her adımda model YALNIZ geçmişten fit edilir ve BİR
    SONRAKİ günü tahmin eder (kesin nedensel; `adaptive_calibration`'ın prequential
    disipliniyle aynı). `refit` gün başına bir kez yeniden fit edilir (maliyet freni;
    aradaki günlerde en son fit kullanılır — bu da nedenselliği bozmaz)."""
    n = int(rv.size)
    if n < min_train + 30:
        return None
    tahmin_har, tahmin_rw, tahmin_ma, tahmin_sabit, gercek = [], [], [], [], []
    model = None
    for t in range(min_train, n - 1):
        if model is None or (t - min_train) % max(1, refit) == 0:
            model = ms.har_rv_fit(rv[:t], log_target=log_target)
        if model is None:
            continue
        p = ms.har_rv_forecast(model, rv[:t], horizon=1)
        if p is None or not math.isfinite(p) or p <= 0:
            continue
        tahmin_har.append(p)
        tahmin_rw.append(float(rv[t - 1]))
        tahmin_ma.append(float(np.mean(rv[max(0, t - 22):t])))
        tahmin_sabit.append(float(np.mean(rv[:t])))
        gercek.append(float(rv[t]))
    if len(gercek) < 30:
        return None
    y = np.asarray(gercek)
    yarismacilar = {"HAR": np.asarray(tahmin_har), "RW": np.asarray(tahmin_rw),
                    "MA22": np.asarray(tahmin_ma), "SABIT": np.asarray(tahmin_sabit)}
    skor = {ad: {"qlike": _qlike(y, p), "mse": float(np.mean((y - p) ** 2))}
            for ad, p in yarismacilar.items()}
    en_iyi_taban = min(("RW", "MA22", "SABIT"), key=lambda a: skor[a]["qlike"])
    beceri = 1.0 - skor["HAR"]["qlike"] / skor[en_iyi_taban]["qlike"]
    return {"n_oos": int(y.size), "skor": skor, "en_iyi_taban": en_iyi_taban,
            "beceri_qlike": float(beceri),
            "beceri_mse": float(1.0 - skor["HAR"]["mse"] / skor[en_iyi_taban]["mse"]),
            "log_target": bool(log_target)}


def micro_diagnostics(bars: pd.DataFrame) -> dict[str, Any]:
    """Mikroyapı teşhisi: gürültü imzası + etkin makas + likidite (ACI DERS #7 takımı)."""
    kapanis = bars["close"].to_numpy(dtype=float)
    out: dict[str, Any] = {
        "imza": ms.volatility_signature(kapanis, 15),
        "roll": ms.roll_spread(kapanis),
        "tsrv": ms.two_scale_rv(kapanis, k=10),
    }
    if "volume" in bars:
        r = np.diff(np.log(kapanis))
        dolar = (bars["close"] * bars["volume"]).to_numpy(dtype=float)[1:]
        out["amihud"] = ms.amihud_illiquidity(r, dolar)
    if {"high", "low"} <= set(bars.columns):
        out["corwin_schultz"] = ms.corwin_schultz_spread(bars["high"].to_numpy(),
                                                          bars["low"].to_numpy())
    return out


def _fmt(x: object, n: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{float(x):.{n}f}" if isinstance(x, (int, float)) else str(x)


def report(symbol: str, gunluk: pd.DataFrame, wf: dict | None, mikro: dict) -> str:
    satir = [f"[gerçekleşen oynaklık] {symbol} — {len(gunluk)} gün"]
    if len(gunluk):
        rv = gunluk["rv"]
        pay = (gunluk["jump"] / gunluk["rv"]).mean()
        satir.append(f"  RV: ort={rv.mean():.3e} medyan={rv.median():.3e}"
                     f" → yıllık vol ort {math.sqrt(rv.mean() * 252) * 100:.1f}%"
                     f" · sıçrama payı ort {pay:.3f}")
    if wf is None:
        satir.append("  yürüyen OOS: YETERSİZ örneklem (uydurma skor yok).")
    else:
        satir.append(f"  yürüyen OOS (n={wf['n_oos']}, log_target={wf['log_target']}):")
        for ad in ("HAR", "RW", "MA22", "SABIT"):
            s = wf["skor"][ad]
            satir.append(f"      {ad:<6} QLIKE={s['qlike']:.5f}  MSE={s['mse']:.3e}")
        verdikt = "GEÇİYOR" if wf["beceri_qlike"] > 0 else "GEÇEMİYOR"
        satir.append(f"      en iyi taban = {wf['en_iyi_taban']} · HAR becerisi"
                     f" QLIKE {wf['beceri_qlike']:+.4f} / MSE {wf['beceri_mse']:+.4f}"
                     f" → tabanı {verdikt}")
    imza, roll, tsrv = mikro.get("imza"), mikro.get("roll"), mikro.get("tsrv")
    if imza:
        satir.append(f"  gürültü imzası: eğim={imza['noise_slope']:+.3e}"
                     f" · gürültü şüphesi={imza['noise_suspected']}")
    if tsrv:
        satir.append(f"  TSRV={tsrv['tsrv']:.3e} (ham RV={tsrv['rv_all']:.3e},"
                     f" gürültü var={_fmt(tsrv['noise_var'], 10)})")
    if roll:
        satir.append(f"  Roll etkin makas: {_fmt(roll['spread'], 5)}"
                     f" (pozitif-otokovaryans={roll['positive_autocov']})")
    cs, am = mikro.get("corwin_schultz"), mikro.get("amihud")
    if cs:
        # `spread` None ise tahminci alan dışındadır (BULGU-33-4): kırpılmış değeri
        # yine gösteriyoruz ama ARTEFAKT olduğunu söyleyerek — sessizce basmak, ölçüm
        # yokluğunu ölçüm gibi sunmak olurdu.
        satir.append(f"  Corwin-Schultz makas: {_fmt(cs['spread'], 5)}"
                     f" (ham {_fmt(cs['spread_raw'], 5)},"
                     f" negatif {cs['n_negative']}/{cs['n']} = %{cs['negative_share'] * 100:.0f})"
                     + ("" if cs["reliable"] else
                        f" ⚠ ALAN DIŞI: ham ortalama ≤ 0 → kırpılmış {_fmt(cs['spread_clipped'], 5)}"
                        " tamamen kırpma artefaktıdır (tahminci GÜNLÜK bar içindir)"))
    if am:
        satir.append(f"  Amihud illikidite (1M$ başına): {_fmt(am['illiq'], 8)}"
                     f" · kullanılan bar {am['n_used']}/{am['n_input']}")
    satir.append("  Not: oynaklık öngörülebilirliği KENAR DEĞİLDİR (ACI DERS #3);"
                 " bu ölçüm risk/boyutlandırma tarafı içindir.")
    return "\n".join(satir)


def probe(symbol: str, start: str, end: str | None = None, interval: str = "1h",
          refit: int = 20) -> tuple[pd.DataFrame, dict | None, dict]:
    from nihai.data import alpaca_history

    if interval not in _GECERLI_ARALIK:
        raise ValueError(f"bilinmeyen aralık: {interval!r} (geçerli: {sorted(_GECERLI_ARALIK)})")
    bars = alpaca_history.get_history(symbol, interval=interval, start=start, end=end)  # type: ignore[arg-type]
    if bars.empty:
        return pd.DataFrame(), None, {}
    gunluk = daily_rv(bars)
    wf = walk_forward(gunluk["rv"].to_numpy(dtype=float), refit=refit) if len(gunluk) else None
    return gunluk, wf, micro_diagnostics(bars)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gerçekleşen oynaklık canlı ölçümü (KARAR-DIŞI)")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--start", default="2024-08-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--refit", type=int, default=20, help="kaç günde bir yeniden fit")
    a = ap.parse_args(argv)
    gunluk, wf, mikro = probe(a.symbol.upper(), a.start, a.end, a.interval, a.refit)
    print(report(a.symbol.upper(), gunluk, wf, mikro))
    return 0 if len(gunluk) else 1


if __name__ == "__main__":
    ensure_utf8_stdout()
    raise SystemExit(main())
