"""
ext_indicator_eval.py — Madde 2: ek-göstergeler botun kararına KATKI sağlıyor mu?

Soru (kullanıcı): "Agresif ağırlıklı optimize edersek nasıl performans verir?"
Bu script DÜRÜST cevabı kullanıcının KENDİ verisiyle ölçer — laf değil, sayı.

Üç varyant, AYNI örneklemde, kronolojik IN-SAMPLE (ilk %70) vs OOS (son %30):
  1. baseline   : mevcut el-ayarı setup_score (ek-gösterge KAPALI)
  2. +ext        : ek göstergeler eklenmiş (el-ayarı ağırlık, SETUP_EXT_WEIGHT)
  3. aggressive  : TÜM bileşenlerin ağırlığı IN-SAMPLE'da en-küçük-kareler ile
                   forward-getiriye fit edilir, sonra OOS'a uygulanır

Metrik: IC (skor↔forward getiri Spearman korelasyonu) ve yön isabet oranı.
BEKLENEN DESEN (EDGE_FINDINGS): aggressive IN-SAMPLE'da en iyi, OOS'ta deflate olur
→ overfit kanıtı. Eğer +ext OOS'ta baseline'ı geçmezse, ek göstergeler edge KATMIYOR.

Look-ahead yok: skor t barına kadarki pencereden; getiri t→t+H.

    python -m research.ext_indicator_eval
    python -m research.ext_indicator_eval --interval 1h --horizon 5 --lookback 1200 --stride 2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.data.data_provider import get_provider


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Sıra-korelasyonu (skor↔forward getiri). NaN/sabit → 0."""
    if len(x) < 10:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _hit_rate(score: np.ndarray, fwd: np.ndarray) -> float:
    """Skor işareti forward getiri işaretini ne sıklıkla tutturuyor."""
    mask = (score != 0) & (fwd != 0)
    if mask.sum() == 0:
        return 0.0
    return float((np.sign(score[mask]) == np.sign(fwd[mask])).mean())


def _collect(provider, symbols, interval, horizon, lookback, stride):
    """Her sembol için her `stride` barda: (component_dict, forward_return).

    Look-ahead-güvenli: compute_setup_score yalnızca df.iloc[:t+1]'i görür."""
    from nihai.signals.setup_score import compute_setup_score

    rows: list[tuple[dict, float]] = []
    for sym in symbols:
        try:
            df = provider.get_bars(sym, interval, lookback_bars=lookback)
        except Exception as e:
            print(f"  {sym:6} hata: {e}")
            continue
        if df is None or df.empty or len(df) < 120 + horizon:
            print(f"  {sym:6} yetersiz veri ({0 if df is None else len(df)})")
            continue
        px = df["close"].astype(float).to_numpy()
        n = len(px)
        cnt = 0
        for t in range(80, n - horizon, stride):
            window = df.iloc[: t + 1]
            ss = compute_setup_score(window)
            fwd = (px[t + horizon] / px[t] - 1.0) * 100.0
            if fwd == 0 or not ss.components:
                continue
            rows.append((dict(ss.components), fwd))
            cnt += 1
        print(f"  {sym:6} {cnt} örnek")
    return rows


def _matrix(rows, keys):
    """component dict listesini (X[n,k], y[n]) matrisine çevir."""
    X = np.array([[float(c.get(k, 0.0)) for k in keys] for c, _ in rows])
    y = np.array([r for _, r in rows])
    return X, y


def evaluate(interval, horizon, lookback, stride, ext_weight):
    provider = get_provider()
    from nihai.execution.live_predict_service import _symbols
    symbols = _symbols()[:10]

    # 1) ek-gösterge KAPALI → baseline bileşenleri
    os.environ["SETUP_EXT_INDICATORS_ENABLED"] = "false"
    print(f"[baseline] {len(symbols)} sembol toplaniyor (ext OFF)...")
    rows_base = _collect(provider, symbols, interval, horizon, lookback, stride)

    # 2) ek-gösterge AÇIK → baseline + ext bileşenleri
    os.environ["SETUP_EXT_INDICATORS_ENABLED"] = "true"
    os.environ["SETUP_EXT_WEIGHT"] = str(ext_weight)
    print(f"[+ext] toplaniyor (ext ON, weight={ext_weight})...")
    rows_ext = _collect(provider, symbols, interval, horizon, lookback, stride)

    if len(rows_ext) < 50:
        raise SystemExit("Yetersiz örnek — lookback/stride/veri kaynağını kontrol et.")

    base_keys = sorted({k for c, _ in rows_base for k in c})
    ext_keys = sorted({k for c, _ in rows_ext for k in c})

    # kronolojik split (örnekler zaman sırasında toplandı)
    def split(rows):
        cut = int(len(rows) * 0.7)
        return rows[:cut], rows[cut:]

    base_is, base_oos = split(rows_base)
    ext_is, ext_oos = split(rows_ext)

    def score_sum(rows, keys):
        X, y = _matrix(rows, keys)
        return X.sum(axis=1), y

    # --- Varyant 1: baseline (el-ayarı toplam)
    s_is, y_is = score_sum(base_is, base_keys)
    s_oos, y_oos = score_sum(base_oos, base_keys)
    res_base = ("baseline", _spearman_ic(s_is, y_is), _hit_rate(s_is, y_is),
                _spearman_ic(s_oos, y_oos), _hit_rate(s_oos, y_oos))

    # --- Varyant 2: +ext (el-ayarı toplam, ek bileşenler dahil)
    s_is, y_is = score_sum(ext_is, ext_keys)
    s_oos, y_oos = score_sum(ext_oos, ext_keys)
    res_ext = ("+ext (el-ayari)", _spearman_ic(s_is, y_is), _hit_rate(s_is, y_is),
               _spearman_ic(s_oos, y_oos), _hit_rate(s_oos, y_oos))

    # --- Varyant 3: aggressive — IN-SAMPLE'da en-küçük-kareler ile ağırlık fit et
    Xi, yi = _matrix(ext_is, ext_keys)
    Xo, yo = _matrix(ext_oos, ext_keys)
    # z-normalize (sabit kolonları at) + bias
    mu, sd = Xi.mean(axis=0), Xi.std(axis=0)
    keep = sd > 1e-9
    Xi_n = (Xi[:, keep] - mu[keep]) / sd[keep]
    Xo_n = (Xo[:, keep] - mu[keep]) / sd[keep]
    Xi_b = np.hstack([Xi_n, np.ones((len(Xi_n), 1))])
    Xo_b = np.hstack([Xo_n, np.ones((len(Xo_n), 1))])
    w, *_ = np.linalg.lstsq(Xi_b, yi, rcond=None)
    pred_is = Xi_b @ w
    pred_oos = Xo_b @ w
    res_agg = ("aggressive (fit)", _spearman_ic(pred_is, yi), _hit_rate(pred_is, yi),
               _spearman_ic(pred_oos, yo), _hit_rate(pred_oos, yo))

    # --- rapor
    print("\n" + "=" * 74)
    print(f"EK-GÖSTERGE DEĞERLENDİRMESİ  (interval={interval} horizon={horizon} "
          f"örnek={len(rows_ext)})")
    print("=" * 74)
    print(f"{'varyant':18} | {'IS-IC':>7} {'IS-hit':>7} | {'OOS-IC':>7} {'OOS-hit':>7}")
    print("-" * 74)
    for name, ic_is, h_is, ic_oos, h_oos in (res_base, res_ext, res_agg):
        print(f"{name:18} | {ic_is:7.3f} {h_is*100:6.1f}% | "
              f"{ic_oos:7.3f} {h_oos*100:6.1f}%")
    print("-" * 74)
    print("YORUM:")
    print("• OOS-IC ~0 ve OOS-hit ~%50 → o varyantın öngörü gücü YOK.")
    print("• aggressive IS'te yüksek ama OOS'ta düşükse → OVERFIT (beklenen desen).")
    print("• +ext OOS'ta baseline'ı geçmiyorsa → ek göstergeler edge KATMIYOR;")
    print("  SETUP_EXT_INDICATORS_ENABLED'ı KAPALI tut (EDGE_FINDINGS disiplini).")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description="Madde 2 ek-gösterge OOS değerlendirmesi")
    ap.add_argument("--interval", default=os.getenv("PREDICT_INTERVAL", "1h"))
    ap.add_argument("--horizon", type=int, default=5, help="kaç bar ileri getiri")
    ap.add_argument("--lookback", type=int, default=1200)
    ap.add_argument("--stride", type=int, default=2, help="her N barda örnek (hız)")
    ap.add_argument("--ext-weight", type=float, default=1.0)
    args = ap.parse_args()
    evaluate(args.interval, args.horizon, args.lookback, args.stride, args.ext_weight)


if __name__ == "__main__":
    main()
