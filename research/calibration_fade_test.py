"""
calibration_fade_test.py — Anti-kalibrasyon "fade" diagnostiği (Yol D).

GÖZLEM (EDGE_FINDINGS §17-C): canlı paper'da model GÜVENİ ile gerçek isabet TERS —
yüksek-güven kovaları (0.6-1.1) %40-44, düşük-güven (0.2-0.3) %56 isabet. Bu ya saf
gürültü (AUC~0.5 olduğundan en olası) ya da modelin ters öğrendiği bir yapı.

SORU: "Yüksek güvenli tahminleri FADE et (yönü ters çevir)" kuralı OOS'ta isabeti >%50 /
getiriyi pozitif yapar mı? Anlamı: hit_fade = 1−hit, ret_fade = −fwd_ret_pct.

DÜRÜST TEST (in-sample kandırmaca yok): kronolojik 70/30 train/test. Eşik (τ) yalnız
TRAIN'de seçilir, sonra dokunulmamış TEST'e uygulanır. Anti-kalibrasyon gerçek/dayanıklıysa
TEST'te de tutar; gürültüyse çöker (beklenen). Ek olarak per-sembol ÇAKIŞMASIZ alt-örneklemde
(her horizon'uncu tahmin) tekrarlanır — 1m bitişik barlar bağımsız değildir, gerçek N daha
küçüktür; çakışmasız sonuç gürültüye karşı daha sağlam.

    python -m research.calibration_fade_test
    python -m research.calibration_fade_test --train-frac 0.7 --min-bucket 50
"""
from __future__ import annotations

import argparse
import csv
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.config import DATA_DIR

PRED_LOG = DATA_DIR / "predictions_log.csv"


def _load() -> list[dict]:
    """Çözülmüş tahminleri (hit ∈ {0,1}) kronolojik sırada yükle."""
    if not PRED_LOG.exists():
        return []
    rows: list[dict] = []
    with PRED_LOG.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("hit") not in ("0", "1"):
                continue
            try:
                rows.append({
                    "logged_at": r.get("logged_at") or "",
                    "symbol": r.get("symbol") or "",
                    "conf": float(r["confidence"]),
                    "hit": int(r["hit"]),
                    "ret": float(r["fwd_ret_pct"]),
                    "horizon": int(r.get("horizon_bars") or 5),
                })
            except (ValueError, TypeError, KeyError):
                continue
    rows.sort(key=lambda x: x["logged_at"])
    return rows


def _nonoverlap(rows: list[dict]) -> list[dict]:
    """Her sembol için kronolojik her `horizon`'uncu tahmini al → çakışmayı azalt."""
    seen: dict[str, int] = {}
    out: list[dict] = []
    for r in rows:
        s = r["symbol"]
        h = max(r["horizon"], 1)
        idx = seen.get(s, 0)
        if idx % h == 0:
            out.append(r)
        seen[s] = idx + 1
    return out


def _stats(rows: list[dict], fade: bool) -> tuple[int, float, float]:
    """(n, isabet, ort_getiri%). fade=True → yönü ters çevir."""
    if not rows:
        return 0, float("nan"), float("nan")
    n = len(rows)
    hit = sum((1 - r["hit"]) if fade else r["hit"] for r in rows) / n
    ret = sum((-r["ret"]) if fade else r["ret"] for r in rows) / n
    return n, hit * 100, ret


def _best_threshold(train: list[dict]) -> float:
    """TRAIN'de 'conf≥τ olanı fade et' kuralının getirisini en üst yapan τ'yu seç.
    Aday eşikler: 0.3..1.0 arası 0.05 adım. Hiçbiri pozitif değilse en yükseği döner
    (test'te de muhtemelen başarısız olur — dürüst sonuç)."""
    best_tau, best_ret = 1.0, -1e9
    tau = 0.3
    while tau <= 1.0001:
        sel = [r for r in train if r["conf"] >= tau]
        if len(sel) >= 30:
            _, _, ret = _stats(sel, fade=True)
            if ret > best_ret:
                best_ret, best_tau = ret, tau
        tau += 0.05
    return round(best_tau, 2)


def _report(rows: list[dict], train_frac: float, min_bucket: int, label: str) -> bool:
    """Bir veri kümesi (tam veya çakışmasız) için tam diagnostik. Dönüş: OOS-fade kazandı mı."""
    n = len(rows)
    print(f"\n{'='*64}\n[{label}]  n={n} çözülmüş tahmin\n{'='*64}")
    if n < 200:
        print("  (yetersiz veri, ≥200 gerek)")
        return False

    cut = int(n * train_frac)
    train, test = rows[:cut], rows[cut:]

    # Kalibrasyon kararlılığı: güven kovası isabeti TRAIN vs TEST (anti-kalibrasyon stabil mi?)
    print(f"Güven kovası isabeti — TRAIN(n={len(train)}) vs TEST(n={len(test)}):")
    print(f"  {'kova':<10} {'tr_n':>6} {'tr_hit%':>8} {'te_n':>6} {'te_hit%':>8}")
    lo = 0.0
    stable_inversions = 0
    while lo < 1.1:
        tr = [r for r in train if lo <= r["conf"] < lo + 0.1]
        te = [r for r in test if lo <= r["conf"] < lo + 0.1]
        if len(tr) >= min_bucket and len(te) >= min_bucket:
            tr_h = sum(r["hit"] for r in tr) / len(tr) * 100
            te_h = sum(r["hit"] for r in te) / len(te) * 100
            mark = ""
            if lo >= 0.6:  # "yüksek güven" bölgesi
                if tr_h < 50 and te_h < 50:
                    stable_inversions += 1
                    mark = "  ← iki yarıda da <%50 (stabil ters)"
                elif (tr_h < 50) != (te_h < 50):
                    mark = "  ← yarılarda çelişiyor (gürültü)"
            print(f"  {lo:.1f}-{lo+0.1:.1f} {len(tr):>6} {tr_h:>8.1f} {len(te):>6} {te_h:>8.1f}{mark}")
        lo += 0.1

    # Strateji kıyası — hepsi TEST (OOS) üzerinde
    tau = _best_threshold(train)
    _, base_hit, base_ret = _stats(test, fade=False)              # sinyali izle
    _, fall_hit, fall_ret = _stats(test, fade=True)               # hepsini fade et
    sel = [r for r in test if r["conf"] >= tau]
    sn, sel_hit, sel_ret = _stats(sel, fade=True)                 # yalnız conf≥τ fade et

    print(f"\nOOS (TEST) strateji kıyası  [τ TRAIN'de seçildi = {tau}]:")
    print(f"  {'strateji':<28} {'n':>6} {'isabet%':>9} {'ort_getiri%':>12}")
    print(f"  {'sinyali izle (baseline)':<28} {len(test):>6} {base_hit:>9.1f} {base_ret:>12.4f}")
    print(f"  {'hepsini fade et':<28} {len(test):>6} {fall_hit:>9.1f} {fall_ret:>12.4f}")
    print(f"  {'conf≥τ fade et':<28} {sn:>6} {sel_hit:>9.1f} {sel_ret:>12.4f}")

    # Verdikt: OOS'ta conf≥τ fade hem isabet>%52 hem getiri>0 hem kalibrasyon stabil mi?
    won = (sn >= min_bucket and sel_hit > 52.0 and sel_ret > 0 and stable_inversions >= 1)
    print(f"\n  → VERDİKT [{label}]: ", end="")
    if won:
        print("OOS'ta conf≥τ fade KAZANDI + anti-kalibrasyon stabil → "
              "ADAY (örneklemi büyüt, ileri-doğrula; tek başına KANIT değil).")
    else:
        print("OOS'ta dayanıklı fade kenarı YOK → anti-kalibrasyon in-sample gürültü "
              "(beklenen; §17-C). Güven sinyali kullanma.")
    return won


def main() -> None:
    ap = argparse.ArgumentParser(description="Anti-kalibrasyon fade diagnostiği (canlı paper)")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--min-bucket", type=int, default=50, help="kova/strateji min örnek")
    args = ap.parse_args()

    rows = _load()
    if not rows:
        print("Çözülmüş tahmin yok. Önce: python -m nihai.execution.prediction_tracker --resolve")
        sys.exit(0)

    print(f"Yüklendi: {len(rows)} çözülmüş tahmin "
          f"({rows[0]['logged_at'][:10]} → {rows[-1]['logged_at'][:10]})")
    print("UYARI: 1m bitişik tahminler bağımsız değil; gerçek N daha küçük. Çakışmasız "
          "alt-örneklem de raporlanır — asıl ağırlık ona verilmeli.")

    won_full = _report(rows, args.train_frac, args.min_bucket, "TAM ÖRNEKLEM")
    won_no = _report(_nonoverlap(rows), args.train_frac, max(args.min_bucket // 3, 20),
                     "ÇAKIŞMASIZ (her horizon'uncu)")

    print(f"\n{'#'*64}")
    if won_full and won_no:
        print("GENEL: her iki örneklemde de fade kenarı → araştırmaya DEĞER (yine de canlı doğrula).")
    elif won_full or won_no:
        print("GENEL: yalnız bir örneklemde geçti → kırılgan, muhtemelen gürültü. Ship etme.")
    else:
        print("GENEL: hiçbir örneklemde dayanıklı fade kenarı YOK → güven anti-kalibrasyonu "
              "gürültü. Win-rate'i güven-filtresiyle artırma yolu KAPALI (§17-C doğrulandı).")
    print("#" * 64)


if __name__ == "__main__":
    main()
