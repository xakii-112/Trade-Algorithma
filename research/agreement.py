"""
agreement.py — Bağımsız sinyal-TİPİ mutabakatı: precision yükselir mi? (EDGE_FINDINGS §13.4)

Soru (kullanıcı): "seçilen hisselerin doğru çıkma olasılığını aynı/yakın kâr oranıyla artır."
Tek bir sinyale güvenmek yerine, BAĞIMSIZ kaynaklar aynı işleme "evet" diyorsa o işlemin
isabeti artmalı — payoff'a (çıkış mantığı) dokunmadan.

Bu harness iki bağımsız kaynağı çaprazlar (ikisi de §13'te ayrı dayanıklı kenar):
  1. META — ikincil model kazanma olasılığı (meta_model.json; per-sembol ZAMANLAMA kalitesi).
  2. MOM  — kesitsel momentum percentile (xsect mantığı; PORTFÖY seçimi, farklı kaynak).

Girdi: event_breakout işlemleri (meta_dataset_*.csv — feature + pnl + entry_time + symbol).
Her işleme entry günündeki momentum-percentile'ı (look-ahead-güvenli) eklenir. Sonra:
  BASE / yalnız-META / yalnız-MOM / İKİSİ-BİRDEN (mutabakat) için PF/WR/E + kâr-korunan%.

Mutabakat (BOTH) BASE'i VE her tek filtreyi PF+WR'da geçerken makul kâr koruyorsa →
bağımsız onay precision'ı "bedavaya" yükseltiyor demektir. Geçmiyorsa kaynaklar bu
veride bağımsız bilgi katmıyor.

    python -m research.agreement
    python -m research.agreement --meta-thr 0.50 --mom-frac 0.50 --lookback 126 --skip 21
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import numpy as np
import pandas as pd

from nihai.config import DATA_DIR
from nihai.signals.meta_features import FEATURE_NAMES, ensure_feature_columns


def _map_mom_pct(df: pd.DataFrame, pct_ff: pd.DataFrame) -> np.ndarray:
    """Her işleme entry_dt anına kadarki (≤ entry, asof) momentum percentile'ını eşle.

    D8 (Tur-10): satır-başına iterrows+searchsorted → TEK vektörize searchsorted + konum
    lookup'ı. Bit-exact: sembol panelde yok / entry ilk tarihten önce → NaN (birebir kural;
    parite: test_perf_parity_round10)."""
    out = np.full(len(df), np.nan)
    if len(df) == 0 or pct_ff.empty:
        return out
    row_idx = pct_ff.index.searchsorted(pd.DatetimeIndex(df["entry_dt"]), side="right") - 1
    col_of = {c: j for j, c in enumerate(pct_ff.columns)}
    col_idx = np.array([col_of.get(s, -1) for s in df["symbol"]])
    vals = pct_ff.to_numpy(float)
    ok = (row_idx >= 0) & (col_idx >= 0)
    out[ok] = vals[row_idx[ok], col_idx[ok]]
    return out


def _logistic_pwin(df: pd.DataFrame) -> np.ndarray | None:
    """meta_model.json'daki lojistik ağırlıklarla her satırın kazanma olasılığı."""
    js = DATA_DIR / "meta_model.json"
    if not js.exists():
        return None
    d = json.loads(js.read_text(encoding="utf-8"))
    feats = d.get("features", FEATURE_NAMES)
    w, mu, sd = np.array(d["w"]), np.array(d["mu"]), np.array(d["sd"])
    # Tur-12: kardeş modüller (train_meta/purged_cv/live_retrain/train_pead_ml) gibi şemayı
    # yükselt — eksik özellik kolonu KeyError/sessiz-NaN yerine nötr-doldurulur (NaN modele sızıp
    # p_win'i NaN yapıp mutabakat ölçümünü sessizce aşağı çekmesin).
    df, _ = ensure_feature_columns(df)
    X = df[feats].to_numpy(float)
    xs = (X - mu) / sd
    xb = np.hstack([np.ones((len(X), 1)), xs])
    z = np.clip(xb @ w, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def _momentum_pct_panel(symbols: list[str], period: str, *, lookback: int,
                        skip: int, throttle: float = 0.15) -> pd.DataFrame:
    """Her gün, her sembolün kesitsel momentum PERCENTILE'ı (0..1). Look-ahead-güvenli:
    skor(t) = close(t-skip)/close(t-lookback)-1 → o günkü kesit içi rank pct."""
    import time

    from nihai.execution.backtest import load_bars
    series = {}
    n = len(symbols)
    for i, s in enumerate(symbols, 1):
        try:
            df = load_bars(s, "1d", period=period)
            if not df.empty and len(df) > lookback + 5:
                series[s] = df["close"]
        except Exception:
            pass
        if i % 25 == 0:
            print(f"  [mom panel {i}/{n}]", flush=True)
        if throttle > 0:
            time.sleep(throttle)
    panel = pd.DataFrame(series).sort_index()
    if panel.empty:
        return panel
    mom = panel.shift(skip) / panel.shift(lookback) - 1.0   # değer t indeksinde, geçmişi kullanır
    pct = mom.rank(axis=1, pct=True)                         # kesit içi percentile [0,1]
    return pct


def _metrics(pnls: np.ndarray) -> tuple[int, float, float, float]:
    """(n, WR%, PF, E)."""
    if len(pnls) == 0:
        return 0, 0.0, 0.0, 0.0
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gl = abs(losses.sum())
    pf = (wins.sum() / gl) if gl > 0 else float("inf")
    return len(pnls), 100 * len(wins) / len(pnls), pf, float(pnls.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Bağımsız sinyal mutabakatı (meta × momentum)")
    ap.add_argument("--csv", default=None, help="meta_dataset_*.csv (varsayılan: en büyük)")
    ap.add_argument("--period", default=None, help="momentum paneli periyodu (varsayılan: işlem span'ından)")
    ap.add_argument("--lookback", type=int, default=126)
    ap.add_argument("--skip", type=int, default=21)
    ap.add_argument("--meta-thr", type=float, default=None, help="varsayılan: modelin gate_threshold'u")
    ap.add_argument("--mom-frac", type=float, default=0.50, help="MOM-onay eşiği: percentile ≥ bu (0.5=üst yarı)")
    args = ap.parse_args()

    csv_path = args.csv
    if not csv_path:
        cands = sorted(glob.glob(str(DATA_DIR / "meta_dataset_*.csv")), key=os.path.getsize)
        if not cands:
            print("Dataset yok. Önce build_meta_dataset.py çalıştır.")
            return
        csv_path = cands[-1]
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["entry_time"]).copy()
    df["entry_dt"] = pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["entry_dt"])
    if df.empty:
        print("entry_time çözümlenemedi; mutabakat ölçülemiyor.")
        return
    print(f"Dataset: {csv_path}\nİşlem: {len(df)} | sembol: {df['symbol'].nunique()}")

    p_win = _logistic_pwin(df)
    if p_win is None:
        print("meta_model.json yok — önce train_meta.py --save (lojistik) çalıştır.")
        return
    df["p_win"] = p_win
    meta_thr = args.meta_thr
    if meta_thr is None:
        try:
            meta_thr = float(json.loads((DATA_DIR / "meta_model.json").read_text(encoding="utf-8"))
                             .get("gate_threshold", 0.5))
        except Exception:
            meta_thr = 0.5

    # Momentum paneli — periyot BUGÜNDEN en erken işlemin ~1 yıl ÖNCESİNE kadar uzamalı:
    # lookback (126 bar) momentum için ilk işlemden ÖNCE geçmiş gerekir, yoksa erken
    # işlemlerde percentile NaN olur (coverage çöker).
    if args.period:
        period = args.period
    else:
        now = pd.Timestamp.now(tz="UTC")
        back_days = (now - df["entry_dt"].min()).days
        period = f"{back_days // 365 + 2}y"   # +2y tampon: ~1y lookback-öncesi + marj
    symbols = sorted(df["symbol"].unique())
    print(f"Momentum paneli: {len(symbols)} sembol, {period}, "
          f"lookback={args.lookback} skip={args.skip}\n")
    pct = _momentum_pct_panel(symbols, period, lookback=args.lookback, skip=args.skip)
    if pct.empty:
        print("Momentum paneli boş.")
        return
    pct.index = pd.to_datetime(pct.index, utc=True)

    # Her işleme entry günündeki (≤ entry) momentum percentile'ı eşle (asof, ffill)
    pct_ff = pct.sort_index()
    df["mom_pct"] = _map_mom_pct(df, pct_ff)  # D8: vektörize asof eşleme (bit-exact)
    cov = df["mom_pct"].notna().mean()
    print(f"Momentum coverage: {100*cov:.0f}% işlemde percentile bulundu "
          f"(eşlenemeyen işlemler 'mutabakat yok' sayılır)\n")

    pnl = df["pnl"].to_numpy(float)
    base_total = float(pnl.sum())
    meta_ok = df["p_win"].to_numpy(float) >= meta_thr
    mom_ok = (df["mom_pct"].to_numpy(float) >= args.mom_frac)  # NaN ≥ x → False

    groups = {
        "BASE (tümü)":            np.ones(len(df), bool),
        f"yalnız META (p≥{meta_thr:.2f})": meta_ok,
        f"yalnız MOM (pct≥{args.mom_frac:.2f})": mom_ok,
        "İKİSİ (mutabakat)":      meta_ok & mom_ok,
        "HİÇBİRİ":                ~meta_ok & ~mom_ok,
    }

    print(f"{'='*74}")
    print(f"  {'grup':<26} {'işlem':>6} {'%al':>5} {'WR%':>6} {'PF':>6} {'E':>10} {'kâr-kor%':>9}")
    print(f"{'='*74}")
    base_wr = base_pf = None
    for name, mask in groups.items():
        sub = pnl[mask]
        n, wr, pf, e = _metrics(sub)
        ret_kept = (100 * sub.sum() / base_total) if base_total != 0 else float("nan")
        share = 100 * n / len(df)
        if name.startswith("BASE"):
            base_wr, base_pf = wr, pf
        flag = ""
        if name.startswith("İKİSİ") and base_wr is not None and base_pf is not None:
            flag = "  ← mutabakat üstün" if (wr > base_wr and pf > base_pf and pf > 1.0) else ""
        pf_s = f"{pf:>6.2f}" if pf != float("inf") else "   inf"
        print(f"  {name:<26} {n:>6} {share:>4.0f}% {wr:>6.1f} {pf_s} {e:>10.1f} {ret_kept:>8.1f}{flag}")
    print(f"{'='*74}")
    print("Yorum: 'İKİSİ' satırı BASE'i VE her tek filtreyi PF+WR'da geçip kâr-kor% makul")
    print("(işlem-payına yakın) ise → bağımsız onay precision'ı bedavaya yükseltiyor.")
    print("Tek filtreler geçip mutabakat geçmiyorsa kaynaklar bağımsız bilgi katmıyor.")


if __name__ == "__main__":
    main()
