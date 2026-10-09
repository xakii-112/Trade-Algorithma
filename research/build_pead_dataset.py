"""
build_pead_dataset.py — PEAD KESİT-ML eğitim verisi (survivorship-temiz, ücretsiz).

Amaç: PEAD'in sabit ±%2 eşiğini, HANGİ earnings olayına girileceğini öğrenen bir
kesit-seçim (meta-labeling) modeliyle değiştirebilmek için eğitim verisi üretmek.

Fark (build_meta_dataset'ten): birincil PEAD sinyalinin fire ettiği her olaya, o olayı
niteleyen KESİT özelliklerini ekler:
  - surprise        : bilanço sürpriz % (yönlü). PEAD'in tanımlayıcı değişkeni.
  - abs_surprise    : |surprise| — sürpriz büyüklüğü (drift'in gücüyle korele, §18.5).
  - dollar_vol_med  : giriş ÖNCESİ 60 barın medyan dolar-hacmi (ham). Trainer bunu
                      evren-içi YÜZDELİĞE (cap_tier ∈ [0,1]) çevirir → PIT market-cap
                      ücretsiz yok; likidite/büyüklük vekili. Survivorship-nötr (rank).

Disiplin (docs/EDGE_FINDINGS §16/§18, CLAUDE.md):
  - LOOK-AHEAD YOK: surprise = giriş barının BİR ÖNCEKİ (earnings) barında halka açık;
    dollar_vol_med yalnız giriş ÖNCESİ barlardan. Etiket (pnl) = gerçek backtest çıkışı.
  - Meta-labeling: birincil model (detect_pead, |surprise|≥2%) YÖN üretir; bu veri o
    olaylar üzerinde "kâr edecek mi?" ikincil modelini eğitmek içindir. Eşiği DÜŞÜRMEZ
    (birincil sinyali değiştirmek değil, onu FİLTRELEMEK).
  - removed_since etiketi: survivorship kapısı. Kenar iddiası removed-since'te de
    görülmeli (yoksa artefakt — §13.5).

    python -m research.build_pead_dataset --as-of 2022-01-01
    python -m research.build_pead_dataset --as-of 2020-01-01   # daha uzun drift penceresi
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time

import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.config import DATA_DIR
from nihai.execution.backtest import backtest_symbol, load_bars
from nihai.signals.meta_features import FEATURE_NAMES, TradeRecorder
from nihai.signals.strategy import _pead_earnings

# Kesit-özellik sütunları (generic 19 + bunlar) + etiket/iz sütunları.
# sue = Standardized Unexpected Earnings: sürprizi hissenin KENDİ geçmiş sürpriz
# dağılımına göre standardize eder (surprise - geçmiş-ort) / geçmiş-std. Ham surprise%
# gürültülüdür (küçük payda → aykırı); SUE literatürde PEAD drift'inin daha iyi öngörücüsü.
PEAD_COLS = ["surprise", "abs_surprise", "sue", "abs_sue", "dollar_vol_med"]
EXTRA_COLS = ["pnl", "label", "symbol", "entry_time", "bars_held",
              "mae_pct", "mfe_pct", "removed_since"]


def _enrich_row(row: dict, df: pd.DataFrame, earnings: pd.DataFrame | None) -> bool:
    """row'a surprise/abs_surprise/dollar_vol_med ekle. Look-ahead güvenli.

    Eşleşen earnings bulunamazsa False döner (satır düşürülür — güvenli taraf).
    """
    et = row.get("entry_time")
    if et in (None, ""):
        return False
    try:
        entry_ts = pd.to_datetime(et)
    except Exception:
        return False
    entry_date = entry_ts.date()

    # --- surprise: giriş barından ÖNCEKI en yakın earnings (≤5 takvim günü) ---
    # (index artan sıralı olmayabilir → tüm olayları gezip en yakın-öncesini seç.)
    if earnings is None or earnings.empty:
        return False
    best = None  # (label, delta_gün)
    for d in earnings.index:
        dd = d.date() if hasattr(d, "date") else d
        delta = (entry_date - dd).days
        if 0 < delta <= 5 and (best is None or delta < best[1]):
            best = (d, delta)
    if best is None:
        return False
    matched_date = best[0]
    try:
        surprise = float(earnings.loc[matched_date, "surprise"])
    except (TypeError, ValueError):
        return False  # B7 (Tur-10): duplike-tarih Series / sayı-dışı değer → güvenli tarafta düş
    if not math.isfinite(surprise):
        return False  # B7 (Tur-10): NaN/inf surprise KAYNAKTA elenir (prepare() savunmasına ek katman)

    # --- SUE: sürprizi hissenin GEÇMİŞ sürpriz dağılımına standardize et (look-ahead güvenli:
    # yalnız matched_date'ten ÖNCEKI earnings kullanılır) ---
    md = matched_date.date() if hasattr(matched_date, "date") else matched_date
    prior = [float(earnings.loc[d, "surprise"]) for d in earnings.index
             if (d.date() if hasattr(d, "date") else d) < md]
    sue = 0.0
    if len(prior) >= 4:
        ps = pd.Series(prior, dtype=float)
        sd = float(ps.std(ddof=0))
        if sd > 1e-9:
            sue = (surprise - float(ps.mean())) / sd

    # --- dollar_vol_med: giriş barına kadar 60 barın medyan dolar-hacmi ---
    dvm = 0.0
    try:
        idx = df.index.searchsorted(entry_ts)  # entry barının konumu
        lo = max(0, idx - 60)
        win = df.iloc[lo:idx]  # giriş barı HARİÇ → look-ahead yok
        if len(win) >= 5 and "volume" in win.columns:
            dv = (win["close"] * win["volume"]).median()
            dvm = float(dv) if pd.notna(dv) else 0.0
    except Exception:
        dvm = 0.0

    row["surprise"] = round(surprise, 3)
    row["abs_surprise"] = round(abs(surprise), 3)
    row["sue"] = round(sue, 3)
    row["abs_sue"] = round(abs(sue), 3)
    row["dollar_vol_med"] = dvm
    return True


def build(symbols: list[str], *, as_of: str, out: str, removed: set[str],
          throttle: float = 0.25) -> int:
    cols = FEATURE_NAMES + PEAD_COLS + EXTRA_COLS
    total = 0
    n = len(symbols)
    n_data = n_trades = n_removed_rows = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for i, s in enumerate(symbols, 1):
            try:
                df = load_bars(s, "1d", start=as_of)
                if df.empty or len(df) < 60:
                    continue
                n_data += 1
                earnings = _pead_earnings(s)
                if earnings is None or earnings.empty:
                    if i % 25 == 0:
                        print(f"[{i}/{n}] {s}: earnings yok", flush=True)
                    continue
                rec = TradeRecorder(symbol=s)
                backtest_symbol(s, interval="1d", df=df, strategy_mode="pead", recorder=rec)
                rs = 1 if s in removed else 0
                kept = []
                for row in rec.rows:
                    if _enrich_row(row, df, earnings):
                        row["removed_since"] = rs
                        kept.append(row)
                if kept:
                    w.writerows(kept)
                    f.flush()
                    n_trades += 1
                    total += len(kept)
                    n_removed_rows += rs * len(kept)
                if i % 25 == 0:
                    print(f"[{i}/{n}] veri={n_data} işlemli={n_trades} "
                          f"satır={total} (removed={n_removed_rows})", flush=True)
            except Exception as e:
                print(f"[{i}/{n}] {s}: hata {e}", flush=True)
            if throttle > 0:
                time.sleep(throttle)
    print(f"\n{'='*60}")
    print(f"COVERAGE: {n_data}/{n} sembolde veri | {n_trades} sembolde PEAD işlemi")
    print(f"Toplam {total} olay satırı ({n_removed_rows} removed-since = survivorship kontrolü)")
    print(f"→ {out}")
    print(f"{'='*60}")
    return total


def main():
    ap = argparse.ArgumentParser(description="PEAD kesit-ML eğitim verisi (PIT, survivorship-temiz)")
    ap.add_argument("--as-of", default="2022-01-01", help="S&P500 üyelik tarihi = test başlangıcı")
    ap.add_argument("--limit", type=int, default=0, help="ilk N sembol (deneme; 0=hepsi)")
    ap.add_argument("--throttle", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from nihai.data.pit_universe import _current_and_changes, _norm, sp500_as_of
    symbols = sp500_as_of(args.as_of)
    cur, _ = _current_and_changes()
    removed = set(symbols) - {_norm(s) for s in cur}
    if args.limit > 0:
        symbols = symbols[: args.limit]
    out = args.out or str(DATA_DIR / f"pead_ml_dataset_pit_sp500_{args.as_of}.csv")
    print(f"=== PEAD kesit-ML dataset | S&P500 @ {args.as_of} | {len(symbols)} sembol "
          f"({len(removed)} removed-since) ===")
    print(f"→ {out} (artımlı yazılıyor)\n")
    build(symbols, as_of=args.as_of, out=out, removed=removed, throttle=args.throttle)


if __name__ == "__main__":
    main()
