"""
feed_ab_test.py — IEX ↔ SIP bar A/B'si: konsolide feed KARARLARI değiştiriyor mu?

SORU (dış rapor P3): `ALPACA_BARS_FEED=sip` varsayılan yapılmalı mı? Tur-57 ÖLÇTÜ ki
ücretsiz planda SIP **tarihsel** barlar yetkilidir ve IEX konsolide hacmin yalnız
%2,2-3,4'ünü görür. Ama "hacim farkı" tek başına bir karar gerekçesi DEĞİLDİR — bu sistem
hacmi doğrudan değil GÖSTERGELER üzerinden kullanır (rvol, ATR, RSI, sinyal). Sorulması
gereken şey farkın KARARA geçip geçmediğidir (Tur-31'in VIX A/B deseni: korelasyon 0,99997
ölçüldü, ayrışan karar 0/1/0/0 → göç ölçümleri geçersiz kılmıyor).

NE ÖLÇER (aynı sembol, aynı pencere, TEK değişken feed):
  · bar KÜMESİ farkı — SIP uzatılmış seansı da taşır, yani satır sayısı DEĞİŞİR;
    karşılaştırma ORTAK zaman damgalarında yapılır (aksi halde ölçülen şey feed değil
    TAKVİM olurdu).
  · ortak barlarda kapanış/hacim korelasyonu + bağıl fark dağılımı.
  · gösterge farkı: RSI14 · ATR14 · ADX14 (son bar) — panelin karar bileşenleri.
  · KARAR farkı: `generate_signal` aksiyonu ortak barların son N'inde kaç kez AYRIŞIYOR.

Bu bir kenar testi DEĞİLDİR ve bir kapı da değildir; bir GÖÇ KARARININ maliyetini ölçer.

Kullanım (ağ gerektirir, ALPACA anahtarları .env'de olmalı):
    python -m research.feed_ab_test --symbols AAPL,SPY --interval 1h --bars 400
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def _bars(symbol: str, interval: str, bars: int, feed: str) -> pd.DataFrame | None:
    """Tek değişken FEED: sağlayıcı env'i geçici olarak kurulur ve modül YENİDEN yüklenir.

    Aynı süreçte, art arda, aynı yapılandırmayla (Tur-49 A/B kuralı) — tek fark feed."""
    import importlib
    os.environ["ALPACA_BARS_FEED"] = feed
    import nihai.data.alpaca_provider as AP
    importlib.reload(AP)
    try:
        p = AP.AlpacaProvider()
        df = p.get_bars(symbol, interval, lookback_bars=bars)  # type: ignore[arg-type]
    except Exception as e:                                     # noqa: BLE001
        print(f"[feed-ab] {symbol} {feed}: {e}", file=sys.stderr)
        return None
    return df if (df is not None and not df.empty) else None


def _rel(a: np.ndarray, b: np.ndarray) -> dict:
    """Bağıl fark karnesi (payda ~0'a giderken bağıl hata RAPOR EDİLEMEZ — Tur-36)."""
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    if not ok.any():
        return {"n": 0, "max": None, "med": None}
    d = np.abs(a[ok] - b[ok]) / np.abs(b[ok])
    return {"n": int(ok.sum()), "max": float(d.max()), "med": float(np.median(d))}


def compare(symbol: str, interval: str, bars: int, decisions: int = 40) -> dict | None:
    """IEX ↔ SIP karnesi. Biri alınamazsa None (ölçüm yokluğu sonuç DEĞİLDİR)."""
    iex, sip = _bars(symbol, interval, bars, "iex"), _bars(symbol, interval, bars, "sip")
    if iex is None or sip is None:
        return None
    common = iex.index.intersection(sip.index)
    out: dict = {
        "symbol": symbol, "interval": interval,
        "n_iex": int(len(iex)), "n_sip": int(len(sip)), "n_common": int(len(common)),
        "sadece_sip": int(len(sip.index.difference(iex.index))),
        "sadece_iex": int(len(iex.index.difference(sip.index))),
    }
    if len(common) < 60:
        out["not"] = "ortak bar < 60 — gösterge/karar farkı ölçülmedi"
        return out
    a, b = iex.loc[common], sip.loc[common]
    ca, cb = a["close"].to_numpy(float), b["close"].to_numpy(float)
    out["close_kor"] = float(np.corrcoef(ca, cb)[0, 1])
    out["close_bagil"] = _rel(ca, cb)
    if "volume" in a.columns and "volume" in b.columns:
        va, vb = a["volume"].to_numpy(float), b["volume"].to_numpy(float)
        out["hacim_bagil"] = _rel(va, vb)
        tot = float(vb.sum())
        out["iex_hacim_payi"] = (float(va.sum()) / tot) if tot > 0 else None

    from nihai.signals.strategy import _SharedIndicators, generate_signal, last_valid
    sa, sb = _SharedIndicators(a), _SharedIndicators(b)
    out["gosterge"] = {
        "rsi14": [last_valid(sa.rsi14()), last_valid(sb.rsi14())],
        "atr14": [last_valid(sa.atr14()), last_valid(sb.atr14())],
        "adx14": [last_valid(sa.adx_series()), last_valid(sb.adx_series())],
    }
    # KARAR farkı — ortak barların son `decisions` tanesinde aksiyon ayrışması.
    n = len(common)
    ayrik = 0
    for i in range(max(60, n - decisions), n):
        act_a = generate_signal(symbol, a.iloc[: i + 1], shared=sa.upto(i + 1)).action
        act_b = generate_signal(symbol, b.iloc[: i + 1], shared=sb.upto(i + 1)).action
        ayrik += int(act_a != act_b)
    out["karar"] = {"denenen": n - max(60, n - decisions), "ayrisan": ayrik}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="IEX ↔ SIP bar feed A/B ölçümü")
    ap.add_argument("--symbols", default="AAPL,SPY")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--bars", type=int, default=400)
    ap.add_argument("--decisions", type=int, default=40)
    args = ap.parse_args()
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        r = compare(sym, args.interval, args.bars, args.decisions)
        if r is None:
            print(f"{sym}: ÖLÇÜLEMEDİ (feed'lerden biri veri vermedi)")
            continue
        print(f"\n=== {r['symbol']} {r['interval']} ===")
        print(f"bar: iex={r['n_iex']} sip={r['n_sip']} ortak={r['n_common']} "
              f"(yalnız-sip={r['sadece_sip']}, yalnız-iex={r['sadece_iex']})")
        if "close_kor" in r:
            cb = r["close_bagil"]
            print(f"kapanış: kor={r['close_kor']:.6f} bağıl-fark ort={cb['med']:.2e} "
                  f"maks={cb['max']:.2e}")
        if "hacim_bagil" in r:
            hb = r["hacim_bagil"]
            pay = r.get("iex_hacim_payi")
            print(f"hacim  : bağıl-fark ort={hb['med']:.3f} maks={hb['max']:.3f}"
                  + (f" | IEX payı={pay:.1%}" if pay else ""))
        if "gosterge" in r:
            for k, (x, y) in r["gosterge"].items():
                print(f"  {k}: iex={x:.4f} sip={y:.4f} Δ={abs(x - y):.4f}")
        if "karar" in r:
            k = r["karar"]
            print(f"KARAR: {k['ayrisan']}/{k['denenen']} barda aksiyon AYRIŞTI")
        if "not" in r:
            print("not:", r["not"])


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout

    ensure_utf8_stdout()
    main()
