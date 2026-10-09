"""
build_live_dataset.py — CANLI loglardan meta-labeling eğitim CSV'si üretir.

live_features_log.csv (karar-barı özellik vektörleri) ile predictions_log.csv'nin
ÇÖZÜLMÜŞ satırlarını (symbol, bar_ts, interval) üzerinden birleştirir → train_meta.py /
purged_cv.py'nin doğrudan yiyebileceği dataset (data/meta_dataset_live_<tarih>.csv).

Bu, ACI DERS #5'in uygulamasıdır: darboğaz PIT veri yokluğu; canlı toplanan veri
yapısı gereği SURVIVORSHIP-TEMİZ (tahmin anında evrende olan semboller) ve
LEAK-FREE'dir (özellikler karar barında yazılır, etiket horizon bar sonra çözülür).

DÜRÜST SINIRLAR (okumadan eğitme):
- Etiket, backtest'teki gibi stop/target-yönetimli işlem PnL'i DEĞİL; sabit-horizon
  (varsayılan 5 bar) taraf-işaretli getiri işaretidir. train_meta çıktısındaki
  PF/expectancy bu yüzden "% getiri" birimindedir, dolar değil.
- Örneklem canlı biriktikçe büyür; birkaç yüz satırın altında hiçbir sonuç kenar
  kanıtı sayılmaz (train_meta zaten kapı simülasyonu + iki bölme ister).
- pwin_side'lı satırlar (giriş sinyali + kapı-engelli aday) dahil edilir → kapının
  elediği taraf da öğrenme setinde (seçilim yanlılığı yok).

Kullanım (repo kökünden):
    python -m research.build_live_dataset            # data/meta_dataset_live_<bugün>.csv
    python -m research.build_live_dataset --min-rows 200
    python -m research.train_meta --csv data/meta_dataset_live_<tarih>.csv --model both
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

import pandas as pd

from nihai.config import DATA_DIR
from nihai.execution.prediction_tracker import FEATS_LOG, PRED_LOG, _side_return
from nihai.signals.meta_features import FEATURE_NAMES, neutral_default

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

# train_meta/build_meta_dataset ile aynı kolon sözleşmesi (EXTRA_COLS karşılığı).
EXTRA_COLS = ["pnl", "label", "symbol", "period", "entry_time", "bars_held",
              "mae_pct", "mfe_pct", "removed_since"]


def build(pred_path=None, feats_path=None) -> pd.DataFrame:
    """Çözülmüş + pwin_side'lı tahminleri özellik vektörleriyle birleştir.

    Dönen DataFrame train_meta ile uyumlu: FEATURE_NAMES + EXTRA_COLS.
    pnl = taraf-işaretli % getiri (horizon-bazlı proxy; dolar değil — docstring'e bak)."""
    pred_path = pred_path or PRED_LOG
    feats_path = feats_path or FEATS_LOG
    if not pred_path.exists() or not feats_path.exists():
        return pd.DataFrame(columns=[*FEATURE_NAMES, *EXTRA_COLS])

    preds = pd.read_csv(pred_path, dtype=str).fillna("")
    # B9 (2026-07-09): join anahtar kolonlarını STR oku. preds zaten dtype=str; feats'te
    # symbol/bar_ts otomatik-çıkarımla int/float olursa (ör. sayısal-görünen bar_ts →
    # "123.0" vs "123") anahtar eşleşmez → SESSİZ satır kaybı. Str okuma iki tarafı hizalar.
    feats = pd.read_csv(feats_path, dtype={"symbol": str, "bar_ts": str, "cand_side": str})
    if preds.empty or feats.empty:
        return pd.DataFrame(columns=[*FEATURE_NAMES, *EXTRA_COLS])

    rows: list[dict] = []
    cohorts: set[tuple[str, str]] = set()
    intervals = feats.get("interval", pd.Series("", index=feats.index)).fillna("").astype(str)
    keys = zip(feats["symbol"].astype(str), feats["bar_ts"].astype(str), intervals, strict=True)
    fmap = dict(zip(keys, feats.index, strict=True))
    for r in preds.to_dict("records"):
        if (not r.get("resolved_at") or r.get("resolved_at") == "UNRESOLVABLE"
                or not r.get("pwin_side")):
            continue   # Tur-12: UNRESOLVABLE (400-bar dışı/delist) eğitim satırı olmaz
        ret = _side_return(r)   # taraf-işaretli % getiri (tracker ile TEK kaynak)
        if ret is None:
            continue
        fi = fmap.get((r["symbol"], r["bar_ts"], r.get("interval", "")))
        if fi is None:
            continue   # özellik logu bu satırdan önce başlamış olabilir → atla
        frow = feats.loc[fi]
        # NaN tuzağı: `x or 0.0` NaN'ı GEÇİRİR (NaN truthy) → eski başlık-yükseltmeli
        # satırların '' hücreleri NaN olarak dataset'e sızar ve modeli sessizce bozardı.
        # pd.notna + nötr varsayılan (extract_features guard'ıyla aynı tablo) kullan.
        out: dict[str, object] = {}
        for n in FEATURE_NAMES:
            v = frow.get(n)
            try:
                out[n] = float(v) if pd.notna(v) else neutral_default(n)
            except (TypeError, ValueError):
                out[n] = neutral_default(n)
        out.update({
            "pnl": ret,                      # % getiri proxy'si (bkz. modül docstring)
            "label": 1 if ret > 0 else 0,
            "symbol": r["symbol"],
            "period": "live",
            "entry_time": r["bar_ts"],
            "bars_held": r.get("horizon_bars", ""),
            "mae_pct": "", "mfe_pct": "", "removed_since": "",
        })
        rows.append(out)
        cohorts.add((r.get("interval", ""), r.get("horizon_bars", "")))
    dataset = pd.DataFrame(rows, columns=[*FEATURE_NAMES, *EXTRA_COLS])
    dataset.attrs["cohorts"] = sorted(cohorts)
    return dataset


def build_unlabeled(feats_path=None, pred_path=None, before: str | None = None) -> pd.DataFrame:
    """ETİKETSİZ özellik satırları (öz-denetimli ön-eğitim yakıtı — Tur-36).

    Dönen DataFrame YALNIZ `FEATURE_NAMES` + `symbol`/`bar_ts` taşır: etiket YOKTUR ve
    olmayacaktır (`neural.NeuralCapacity.pretrain` yalnız X ister).

    "Etiketsiz" = özellik logunda satırı olan ama `predictions_log`'da ÇÖZÜLMÜŞ **ve
    `pwin_side`'lı** karşılığı olmayan (symbol, bar_ts) çifti.

    **Tur-59'da bu kümenin BİLEŞİMİ değişti (BULGU-59-1).** Eskiden özellik logu yalnız
    ADAY barlarını taşıdığı için "etiketsiz" pratikte "henüz çözülmemiş" (= en TAZE =
    GELECEK) demekti; ölçülmüştü ki kullanılabilir kısım 69 satırdı. Politika açıldıktan
    sonra havuz iki farklı popülasyon taşır:
      (a) ADAY-DIŞI barlar — çözülmüş olabilirler, GEÇMİŞTEdirler, `cand_side` boştur;
          ön-eğitimin asıl yakıtı budur (sembol×bar ölçeği).
      (b) henüz çözülmemiş aday barları — hâlâ GELECEKTEdir.
    `before` (ISO ts) bu yüzden hâlâ ZORUNLUDUR: (b) sınıfı tek başına bile test
    dilimindeki özellik dağılımını modele sızdırır (transdüktif look-ahead). Nedensel bir
    ölçüm yapan çağıran `before=<eğitim fold'unun sonu>` vermek zorundadır."""
    pred_path = pred_path or PRED_LOG
    feats_path = feats_path or FEATS_LOG
    cols = ["symbol", "bar_ts", *FEATURE_NAMES]
    if not feats_path.exists():
        return pd.DataFrame(columns=cols)
    feats = pd.read_csv(feats_path, dtype={"symbol": str, "bar_ts": str, "cand_side": str})
    if feats.empty:
        return pd.DataFrame(columns=cols)

    labeled: set[str] = set()
    if pred_path.exists():
        preds = pd.read_csv(pred_path, dtype=str).fillna("")
        for r in preds.to_dict("records"):
            if r.get("resolved_at") and r["resolved_at"] != "UNRESOLVABLE" and r.get("pwin_side"):
                labeled.add(f"{r.get('symbol', '')}|{r.get('bar_ts', '')}")
    key = feats["symbol"].astype(str) + "|" + feats["bar_ts"].astype(str)
    out = feats.loc[~key.isin(labeled)].copy()
    if before:
        ts = pd.to_datetime(out["bar_ts"], errors="coerce", utc=True, format="mixed")
        out = out.loc[ts.notna() & (ts <= pd.Timestamp(before, tz="UTC"))]
    # Eksik/eski-şema kolonları nötr varsayılana doldur (build() ile AYNI sözleşme —
    # NaN ön-eğitime de sızmamalı; maskeli rekonstrüksiyon NaN hedefte tanımsızdır).
    for n in FEATURE_NAMES:
        if n not in out.columns:
            out[n] = neutral_default(n)
        else:
            v = pd.to_numeric(out[n], errors="coerce")
            out[n] = v.fillna(neutral_default(n))
    return out[cols].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Canlı loglardan meta-labeling eğitim CSV'si")
    ap.add_argument("--out", default=None, help="çıktı yolu (vars: data/meta_dataset_live_<bugün>.csv)")
    ap.add_argument("--min-rows", type=int, default=100,
                    help="bu satır sayısının altında CSV yazma (gürültüden model eğitme tuzağı)")
    ap.add_argument("--unlabeled", action="store_true",
                    help="etiketli set yerine ETİKETSİZ ön-eğitim CSV'si üret (Tur-36)")
    ap.add_argument("--before", default=None,
                    help="--unlabeled ile: yalnız bu ISO damgaya kadarki satırlar "
                         "(nedensellik; bkz. build_unlabeled docstring)")
    args = ap.parse_args()

    if args.unlabeled:
        u = build_unlabeled(before=args.before)
        print(f"=== Etiketsiz ön-eğitim seti | satır: {len(u)}"
              + (f" (before={args.before})" if args.before else ""))
        if u.empty:
            print("Etiketsiz satır yok — tüm özellik satırlarının etiketi çözülmüş demektir.")
            return
        out = args.out or (DATA_DIR / f"pretrain_unlabeled_{date.today().isoformat()}.csv")
        u.to_csv(out, index=False)
        print(f"→ {out}")
        print("Sonraki adım: python -m research.train_neural --pretrain --pretrain-dataset", out)
        return

    df = build()
    n = len(df)
    print(f"=== Canlı meta-dataset | çözülmüş+pwin_side'lı+özellikli satır: {n}")
    if n == 0:
        print("Henüz birleştirilebilir satır yok. PREDICT_TRACK=true ile özellik logu birikmeli,")
        print("sonra: python -m nihai.execution.prediction_tracker --resolve")
        return
    print(f"    sembol: {df['symbol'].nunique()} | kazanma %{100 * df['label'].mean():.1f} | "
          f"ort. taraf-getirisi %{df['pnl'].astype(float).mean():+.3f}")
    if n < args.min_rows:
        print(f"UYARI: {n} < --min-rows {args.min_rows} → CSV yazılmadı (örneklem yetersiz; "
              f"eğitim sonuçları gürültü olur). Biriktikçe tekrar çalıştır.")
        return
    out = args.out or (DATA_DIR / f"meta_dataset_live_{date.today().isoformat()}.csv")
    df.to_csv(out, index=False)
    print(f"→ {out}")
    print("Sonraki adım: python -m research.train_meta --csv", out)
    print("Hatırlatma: tek bölme yetmez — time+symbol kapı simülasyonu ikisinde de "
          "pozitifse VE örneklem büyükse anlamlı (ACI DERS #3).")


if __name__ == "__main__":
    main()
