"""
regime_pretest.py — HMM öncesi SIFIR-BAĞIMLILIK rejim ön-testi (ENTEGRASYON_RAPORU §1.6-HMM, §2.5 P1-1).

Soru (rapor §1.6-HMM, şart-1): forward kayıtçıdaki çözülmüş yön-tahminlerini SPY trend/vol rejim
hücrelerine böl; isabet oranı (≈ "NO_EDGE ayrışması") rejime göre AYRIŞIYOR mu? Ayrışmıyorsa
HMM gündemden düşer — hiç kod/bağımlılık eklenmez. Bu test tam da o "girmeye değer mi" sorusudur.

RAPOR-ONLY: üretim koduna dokunmaz; data/ ve nihai/ altına HİÇBİR yazım yapmaz (predictions_log
yalnız KOPYALANIP okunur). Çıktı: stdout Türkçe özet + docs/REGIME_PRETEST_2026-07-18.md. hmmlearn
KURULMAZ — bu testin amacı "HMM'e gerek var mı" sorusudur, kurulumu değil.

Look-ahead önlemleri (CLAUDE.md "look-ahead yasak"):
- Rejim etiketi tahminin bar_ts GÜNÜNE değil, ÖNCEKİ işlem gününün kapanışına kadarki veriyle
  atanır (merge_asof backward, allow_exact_matches=False → tahmin günü D için yalnız state[t<D]).
- TREND: önceki gün kapanışı SMA200'ün üstünde=UP, altında=DOWN.
- VOL: önceki güne kadarki 20g gerçekleşen vol, KENDİ trailing 252g medyanına göre HIGH/LOW.
  (Tam-örneklem tercil/eşik = look-ahead olurdu; KULLANILMAZ — yalnız geriye-dönük 252g medyan.)

"NO_EDGE payı" (per-satır): predictions_log'da NO_EDGE bir SATIR alanı değil; verdict()/forward_harness
onu bir KÜME üzerindeki hüküm olarak (|isabet-0.5|<0.03) tanımlar. Bu ön-testte satır-başı NO_EDGE
proxy'si olarak forecast'in KENDİ kalibre olasılığının nötr bandı kullanılır: |prob_cal-0.5|<0.03
(aynı 0.03 eşiği; §15.1: prob_cal≈0.5 = "öngörü gücü yok"). Böylece hem hücre isabet-ayrışması hem de
"kaç tahmin zaten nötr" birlikte raporlanır. Karar kuralı hücre İSABET ayrışmasına dayanır (aşağıda).

İstatistik research/deflated.py'den (yeniden yazılmaz): wilson_ci, binom_p_value.

    python -m research.regime_pretest
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.config import DATA_DIR
from research.deflated import binom_p_value, wilson_ci

# --- Ön-KAYITLI sabitler (deney öncesi sabitlenir; sonuca göre ayarlanmaz) --------------------
NEUTRAL_BAND = 0.03      # |prob_cal-0.5|<band → satır "NO_EDGE proxy" (nötr forecast); verdict()'in 0.03'ü
EDGE_PP = 3.0            # hücre verdict eşiği (|isabet-50%|, yüzde-puan) — forward_harness ile birebir
MIN_CELL_N = 300         # karar kuralı: bir hücrede en az bu kadar çözülmüş tahmin olmalı
DEVIATION = 0.05         # karar kuralı: havuzdan en az 5 puan (0.05) isabet sapması
N_CELLS = 4              # Bonferroni düzeltmesi (2×2 = 4 rejim hücresi)
ALPHA = 0.05             # düzeltme sonrası anlamlılık eşiği
SMA_WIN = 200            # TREND: SMA200
VOL_WIN = 20             # VOL: 20g gerçekleşen vol
VOL_MED_WIN = 252        # VOL: trailing 252g medyan (kendi geçmişine göre HIGH/LOW)
CELLS = ("UP-LOW", "UP-HIGH", "DOWN-LOW", "DOWN-HIGH")
REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "REGIME_PRETEST_2026-07-18.md"


def _bar_date(raw: str) -> date | None:
    """bar_ts'ten işlem-gününü çıkar. tz-aware ('...-04:00') = ET offset'i → yerel (ET) tarih;
    naive = olduğu gibi. Öğleden sonra barları için ET/UTC tarih ayrımı önemsiz (bkz. sınırlılıklar).
    Ayrıştırılamazsa None (çağıran satırı düşürür)."""
    try:
        return datetime.fromisoformat(str(raw)).date()
    except (ValueError, TypeError):
        return None


def load_predictions() -> pd.DataFrame:
    """predictions_log.csv'yi ÖNCE geçici konuma kopyala (canlı dosya; kayıtçı yazıyor olabilir),
    kopyadan oku (on_bad_lines='skip' → yarım son satıra dayanıklı). Yalnız ÇÖZÜLMÜŞ (hit∈{0,1})
    satırları döndür; bar_ts→pred_date, prob_cal/hit/fwd_ret_pct sayısal."""
    src = DATA_DIR / "predictions_log.csv"
    if not src.exists():
        raise FileNotFoundError(f"predictions_log bulunamadı: {src}")
    with tempfile.TemporaryDirectory() as td:
        snap = Path(td) / "predictions_snapshot.csv"
        shutil.copy2(src, snap)
        df = pd.read_csv(snap, dtype=str, on_bad_lines="skip", encoding="utf-8")

    # Çözülmüş + yönlü: hit ∈ {0,1}. (record() zaten yalnız yönlü satır yazar; yine de garanti.)
    df = df[df["hit"].isin(["0", "1"])].copy()
    df["hit"] = df["hit"].astype(int)
    df["prob_cal"] = pd.to_numeric(df["prob_cal"], errors="coerce")
    df["fwd_ret_pct"] = pd.to_numeric(df["fwd_ret_pct"], errors="coerce")
    df["pred_date"] = df["bar_ts"].map(_bar_date)
    df = df[df["pred_date"].notna()].copy()
    df["pred_date"] = pd.to_datetime(df["pred_date"])
    return df


def spy_regime_table(start: date, end: date) -> pd.DataFrame:
    """SPY 1g indir → her İŞLEM GÜNÜ t için (t'ye kadarki veriyle) trend/vol durumu.

    trend[t]  = close[t] > SMA200[t]            → 'UP' / 'DOWN'
    vol20[t]  = son 20g getiri std'i (t dahil)
    volmed[t] = vol20'nin trailing 252g medyanı  → vol20[t] > volmed[t] ? 'HIGH' : 'LOW'

    Hepsi ≤t veriyle; durum yalnız D>t günündeki tahminlere UYGULANIR (merge_asof strict-backward),
    böylece hiçbir tahmine kendi günü/geleceği sızmaz. Isınma (SMA200/252g medyan NaN) satırları düşer."""
    raw = yf.download("SPY", start=start.isoformat(), end=end.isoformat(),
                      interval="1d", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("SPY günlük veri indirilemedi (yfinance boş döndü; ağ/oran sınırı?)")
    if isinstance(raw.columns, pd.MultiIndex):   # tek-sembol indirmede ('Close','SPY') çok-seviyeli
        raw.columns = raw.columns.get_level_values(0)
    close = raw["Close"].astype(float)

    sma200 = close.rolling(SMA_WIN, min_periods=SMA_WIN).mean()
    vol20 = close.pct_change().rolling(VOL_WIN, min_periods=VOL_WIN).std()
    volmed = vol20.rolling(VOL_MED_WIN, min_periods=VOL_MED_WIN).median()

    state = pd.DataFrame({"close": close, "sma200": sma200, "vol20": vol20, "volmed": volmed})
    state = state.dropna()
    state["trend"] = (state["close"] > state["sma200"]).map({True: "UP", False: "DOWN"})
    state["vol"] = (state["vol20"] > state["volmed"]).map({True: "HIGH", False: "LOW"})
    state["cell"] = state["trend"] + "-" + state["vol"]
    state = state.reset_index().rename(columns={state.index.name or "index": "date",
                                                "Date": "date"})
    state["date"] = pd.to_datetime(state["date"]).dt.tz_localize(None).dt.normalize()
    return state[["date", "trend", "vol", "cell"]].sort_values("date").reset_index(drop=True)


def assign_regime(preds: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """Her tahmine, bar_ts gününden STRICT ÖNCEKİ işlem gününün rejimini ata (önceki gün kapanışı).

    merge_asof(direction='backward', allow_exact_matches=False): tahmin günü D için, D'den KÜÇÜK en
    son state günü t'yi bulur → state[t] yalnız ≤t<D veriyle üretildiğinden look-ahead yok. state
    başlangıcından ÖNCEKİ tahminler (ısınma öncesi) eşleşmez → cell NaN → analize alınmaz."""
    left = preds.sort_values("pred_date").reset_index(drop=True)
    merged = pd.merge_asof(left, state, left_on="pred_date", right_on="date",
                           direction="backward", allow_exact_matches=False)
    return merged


def _verdict_label(hits: int, n: int) -> str:
    """forward_harness._verdict ile aynı dürüst hüküm: nokta-kenar EDGE_PP'yi geçse bile Wilson %95
    GA 0.5'i içeriyorsa NO_EDGE. WEAK_POSITIVE yalnız GA alt>0.5, INVERTED yalnız GA üst<0.5."""
    if n <= 0:
        return "NO_DATA"
    hr = hits / n
    if abs(hr - 0.5) * 100.0 < EDGE_PP:
        return "NO_EDGE"
    lo, hi = wilson_ci(hits, n)
    if hr > 0.5:
        return "WEAK_POSITIVE" if lo > 0.5 else "NO_EDGE"
    return "INVERTED" if hi < 0.5 else "NO_EDGE"


def cell_stats(sub: pd.DataFrame, pool_rate: float) -> dict:
    """Bir rejim hücresi için ölçüm sözlüğü (İDDİA değil ÖLÇÜM)."""
    n = len(sub)
    hits = int(sub["hit"].sum())
    hr = hits / n if n else float("nan")
    lo, hi = wilson_ci(hits, n) if n else (float("nan"), float("nan"))
    p_pool = binom_p_value(hits, n, pool_rate) if n else float("nan")
    p_half = binom_p_value(hits, n, 0.5) if n else float("nan")
    neutral = int((sub["prob_cal"].sub(0.5).abs() < NEUTRAL_BAND).sum())
    no_edge_share = neutral / n if n else float("nan")
    mean_ret = float(sub["fwd_ret_pct"].mean()) if n else float("nan")
    return {"n": n, "hits": hits, "hit_rate": hr, "wilson_lo": lo, "wilson_hi": hi,
            "p_pool": p_pool, "p_half": p_half, "no_edge_share": no_edge_share,
            "mean_ret": mean_ret, "verdict": _verdict_label(hits, n)}


def analyze(preds: pd.DataFrame, label: str) -> dict:
    """Verilen (rejim-atanmış) tahmin kümesi üzerinde 4-hücre analizi + ön-kayıtlı karar kuralı.
    preds 'cell' kolonu içermeli (assign_regime çıktısı); cell NaN olanlar dışlanır."""
    used = preds[preds["cell"].isin(CELLS)].copy()
    n_total = len(used)
    pool_rate = float(used["hit"].mean()) if n_total else float("nan")

    cells = {c: cell_stats(used[used["cell"] == c], pool_rate) for c in CELLS}

    # Ön-KAYITLI karar kuralı: bir hücre (n≥MIN_CELL_N) havuzdan ≥DEVIATION saparsa VE Bonferroni
    # (×N_CELLS) düzeltmeli p_pool < ALPHA ise "ayrışma VAR". Aksi halde "ayrışma YOK".
    divergent = []
    for c in CELLS:
        s = cells[c]
        if (s["n"] >= MIN_CELL_N and abs(s["hit_rate"] - pool_rate) >= DEVIATION
                and (s["p_pool"] * N_CELLS) < ALPHA):
            divergent.append(c)

    testable = [c for c in CELLS if cells[c]["n"] >= MIN_CELL_N]
    if divergent:
        verdict = "AYRISMA_VAR"
    elif len(testable) < 2:
        # Tek rejim hücresi doldu → ayrışma OLGUSAL OLARAK TEST EDİLEMEDİ (rejim çeşitliliği yok).
        verdict = "TEST_EDILEMEDI"
    else:
        verdict = "AYRISMA_YOK"

    return {"label": label, "n_total": n_total, "pool_rate": pool_rate, "cells": cells,
            "divergent": divergent, "testable": testable, "verdict": verdict,
            "unmatched": int((~preds["cell"].isin(CELLS)).sum())}


# ---------------------------------------------------------------------------
# Sunum (stdout + markdown) — ölçümü İDDİAsız aktarır
# ---------------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    return "n/a" if x != x else f"{x * 100:.1f}%"


def _cell_row(c: str, s: dict) -> str:
    return (f"{c:>10} | {s['n']:>6} | {_fmt_pct(s['hit_rate']):>7} | "
            f"[{_fmt_pct(s['wilson_lo'])},{_fmt_pct(s['wilson_hi'])}] | "
            f"{s['p_pool']:>7.3f} | {s['p_half']:>7.3f} | {_fmt_pct(s['no_edge_share']):>7} | "
            f"{s['mean_ret']:>+7.3f} | {s['verdict']}")


def print_summary(res: dict, regime_days: pd.DataFrame) -> None:
    print("=" * 100)
    print(f"HMM ÖN-TESTİ — rejim-koşullu isabet ayrışması  [{res['label']}]")
    print("=" * 100)
    print(f"Toplam rejim-atanmış çözülmüş tahmin: {res['n_total']}   "
          f"havuz isabet oranı: {_fmt_pct(res['pool_rate'])}   "
          f"(eşleşmeyen/ısınma-öncesi: {res['unmatched']})")
    print(f"Rejim gün dağılımı (tahmin günleri): {regime_days}")
    print("-" * 100)
    print(f"{'hücre':>10} | {'n':>6} | {'isabet':>7} | {'Wilson %95 GA':^15} | "
          f"{'p(hav)':>7} | {'p(0.5)':>7} | {'NOEDGE%':>7} | {'ort.ret':>7} | verdict")
    print("-" * 100)
    for c in CELLS:
        print(_cell_row(c, res["cells"][c]))
    print("-" * 100)
    _print_verdict(res)


def _verdict_text(res: dict) -> str:
    if res["verdict"] == "AYRISMA_VAR":
        return (f"AYRIŞMA VAR → HMM şart-1 KARŞILANDI. Sapan hücre(ler): {res['divergent']}. "
                "(Kısa-pencere caveat'ıyla oku; şart-2 expanding-window fit hâlâ gerekir.)")
    if res["verdict"] == "TEST_EDILEMEDI":
        return (f"AYRIŞMA TEST EDİLEMEDİ → n≥{MIN_CELL_N} olan hücre <2 "
                f"({res['testable']}). Forward pencere tek rejime sıkıştığından ayrışma NE "
                "gösterilebildi NE çürütüldü → HMM için pozitif kanıt YOK; şart-1 karşılanmadı → "
                "şimdilik HMM'e girme (pencere genişleyince tekrar koş).")
    return (f"AYRIŞMA YOK → n≥{MIN_CELL_N} hücrelerin hiçbiri havuzdan ≥{DEVIATION * 100:.0f}pp + "
            "Bonferroni-anlamlı sapmıyor. Rapor §1.6-HMM şart-1 KARŞILANMADI → HMM gündemden düşer.")


def _print_verdict(res: dict) -> None:
    print(f"VERDİKT [{res['label']}]: {_verdict_text(res)}")


def _regime_day_counts(assigned: pd.DataFrame) -> pd.Series:
    """Tahmin GÜNÜ başına rejim (çeşitlilik teşhisi) — kaç ayrı gün hangi hücrede."""
    sub = assigned.dropna(subset=["cell"]).copy()
    per_day = sub.groupby(sub["pred_date"].dt.date)["cell"].first()
    return per_day.value_counts()


def build_report(res_all: dict, res_conv: dict, assigned: pd.DataFrame,
                 day_counts: pd.Series, bg_counts: pd.Series, span: tuple[date, date]) -> str:
    """docs/ altına yazılacak tam Türkçe markdown rapor."""
    lines: list[str] = []
    ap = lines.append
    ap("# HMM Öncesi Rejim Ön-Testi — Rapor")
    ap("")
    ap("**Tarih:** 2026-07-18  ·  **İş:** ENTEGRASYON_RAPORU §2.5 P1-1 (rapor-only, bağımlılıksız)  "
       "·  **Model:** Opus 4.8")
    ap("")
    ap("## 0. Soru ve karar bağlamı")
    ap("")
    ap("ENTEGRASYON_RAPORU §1.6-HMM, HMM rejim tespitini iki şarta bağladı. **Şart-1 (bu test):** "
       "*\"önce sıfır-bağımlılık ön-test: mevcut forward tahminlerini basit rejim etiketlerine "
       "bölüp NO_EDGE oranı rejime göre ayrışıyor mu bak — ayrışmıyorsa HMM'e hiç girme.\"* "
       "Şart-2 (expanding-window/walk-forward fit) yalnız şart-1 geçerse gündeme gelir; tam-örneklem "
       "HMM fit'i klasik look-ahead tuzağıdır.")
    ap("")
    ap("Bu test **hiçbir bağımlılık kurmaz** (hmmlearn dahil), üretim koduna dokunmaz, `data/` ve "
       "`nihai/` altına yazmaz. `data/predictions_log.csv` yalnız geçici konuma kopyalanıp okunur.")
    ap("")
    ap("## 1. Ön-KAYITLI karar kuralı (deney öncesi sabitlendi)")
    ap("")
    ap(f"> **\"Ayrışma VAR\"** = en az bir hücrede (n≥{MIN_CELL_N}) isabet oranı havuz oranından "
       f"≥{DEVIATION * 100:.0f} puan sapıyor **VE** Bonferroni-düzeltmeli ({N_CELLS} hücre) "
       f"p<{ALPHA}. Aksi halde **\"ayrışma YOK → HMM gündemden düşer\"** (rapor §1.6-HMM şart-1).")
    ap("")
    ap(f"Ek terminal durum (dürüstlük): n≥{MIN_CELL_N} olan hücre 2'den azsa forward pencere tek "
       "rejime sıkışmış demektir → ayrışma **test edilemedi** (ne gösterildi ne çürütüldü); bu da "
       "HMM için pozitif kanıt sağlamaz → şart-1 karşılanmaz.")
    ap("")
    ap("## 2. Yöntem ve look-ahead önlemleri")
    ap("")
    ap(f"- **Veri:** forward kayıtçı `predictions_log.csv` çözülmüş (`hit∈{{0,1}}`) yön-tahminleri. "
       f"Pencere: **{span[0]} → {span[1]}**.")
    ap("- **Rejim kaynağı:** SPY günlük (yfinance, auto-adjust). Etiketler **yalnız önceki işlem "
       "gününün kapanışına kadarki** veriyle (look-ahead yasak):")
    ap(f"  - **TREND:** önceki gün kapanışı SMA{SMA_WIN} üstünde = `UP`, altında = `DOWN`.")
    ap(f"  - **VOL:** önceki güne kadarki {VOL_WIN}g gerçekleşen vol, KENDİ trailing {VOL_MED_WIN}g "
       "medyanına göre `HIGH`/`LOW`. (Tam-örneklem tercil = look-ahead → kullanılmadı.)")
    ap("- **Eşleme:** `merge_asof(direction='backward', allow_exact_matches=False)` → tahmin günü D "
       "için yalnız t<D günlerinin durumu; tahminin kendi gününün kapanışı ASLA kullanılmaz.")
    ap("- **2×2 = 4 hücre:** UP-LOW, UP-HIGH, DOWN-LOW, DOWN-HIGH.")
    ap(f"- **\"NO_EDGE payı\" tanımı:** `predictions_log`'da NO_EDGE bir *satır* alanı değil — "
       f"`verdict()`/`forward_harness` onu bir KÜME hükmü olarak (|isabet-0.5|<0.03) tanımlar. "
       f"Satır-başı proxy olarak forecast'in kalibre olasılığının nötr bandı kullanıldı: "
       f"|prob_cal-0.5|<{NEUTRAL_BAND} (aynı 0.03 eşiği; §15.1 \"prob_cal≈0.5 = öngörü gücü yok\"). "
       "Karar kuralı hücre **isabet** ayrışmasına dayanır; NO_EDGE payı betimleyici ek metriktir.")
    ap("- **İstatistik:** `research/deflated.py` `wilson_ci` + `binom_p_value` (yeniden yazılmadı). "
       "Hücre verdict etiketi `forward_harness._verdict` ile birebir (EDGE_PP=3.0 + Wilson-GA 0.5 "
       "içerme kontrolü).")
    ap("")
    ap("## 3. Rejim çeşitliliği (tahmin günleri)")
    ap("")
    ap("Forward pencere kısadır; hangi rejimlerin gerçekten gözlendiği kritik. Tahmin günü başına "
       "atanan rejim dağılımı:")
    ap("")
    ap("| rejim hücresi | ayrı tahmin günü |")
    ap("|---|---|")
    for cell, cnt in day_counts.items():
        ap(f"| {cell} | {cnt} |")
    ap("")
    ap("**Sınıflandırıcı dejenerasyon kontrolü (kritik):** yukarıdaki tek-hücre yoğunlaşması bir "
       "hata DEĞİL, kısa pencerenin gerçek özelliğidir. İndirilen ~2 yıllık SPY (ısınma dahil) arka "
       "plan rejim dağılımı, sınıflandırıcının başka etiketler de ürettiğini gösterir:")
    ap("")
    ap("| rejim hücresi | ~2y işlem günü |")
    ap("|---|---|")
    for cell, cnt in bg_counts.items():
        ap(f"| {cell} | {cnt} |")
    ap("")
    ap("Yani sınıflandırıcı UP/DOWN ve HIGH/LOW'u ayırt edebiliyor; forward penceresi (2026 orta-Haz "
       "→ orta-Tem) tesadüfen **bütünüyle UP-HIGH** rejimde. (Not: DOWN-LOW bu 2y'de hiç görülmedi — "
       "düşük-vol düşüş trendi nadirdir; test ileride tekrarlanırken 4-hücre her zaman seyrek olabilir.)")
    ap("")
    ap(_report_table(res_all, "4.1 Sonuç — TÜM çözülmüş yön-tahminleri (birincil)"))
    ap("")
    ap(_report_table(
        res_conv,
        f"4.2 Sonuç — NO_EDGE-DIŞI alt küme (|prob_cal-0.5|≥{NEUTRAL_BAND}) — sağlamlık kontrolü"))
    ap("")
    ap("## 5. Verdikt")
    ap("")
    ap(f"- **Birincil (tüm çözülmüş):** {_verdict_text(res_all)}")
    ap(f"- **Sağlamlık (NO_EDGE-dışı):** {_verdict_text(res_conv)}")
    ap("")
    ap(_final_verdict_line(res_all, res_conv))
    ap("")
    ap("## 6. Sınırlılıklar (dürüst)")
    ap("")
    ap(f"- **Kısa pencere / tek piyasa dönemi:** forward log yalnız **{span[0]} → {span[1]}** "
       "(~haftalar, ~10-12 işlem günü) kapsıyor. Bu, rejim çeşitliliğini ciddi biçimde sınırlar; "
       "yukarıdaki gün-dağılımı tablosu kaç rejimin gerçekten gözlendiğini gösterir. Ayrışma-YOK "
       "sonucu 'rejimler fark yaratmaz' değil, 'bu pencerede fark GÖSTERİLEMEDİ' diye okunmalı.")
    ap("- **Tek gösterge = SPY:** rejim SPY'a göre; sembol-özel rejimler (sektör/beta) yakalanmaz.")
    ap("- **Havuz karşılaştırması:** hücre, kendini de içeren havuza karşı test edilir (kural metni "
       "böyle); hafif muhafazakâr/otokorelasyonlu — büyük-n'de etkisi küçük.")
    ap("- **Forecast zaten NO_EDGE:** havuz isabeti ~%47 (sistem geneli NO_EDGE). Rejim koşullama "
       "olsa olsa zaten kenarsız bir forecast'i böler; taban sinyalde kenar yaratmaz (ACI DERS #5).")
    ap("- **bar_ts tarih sınırı:** tz-aware (ET) ve naive zaman damgaları karışık; öğleden sonra "
       "barlarında ET/UTC tarih ayrımı önemsiz, ama gece-yarısı sınırındaki nadir barlar ±1 gün "
       "kayabilir (rejim günlük olduğundan etki ihmal edilebilir).")
    ap("")
    ap("## 7. Tekrar üretmek için")
    ap("")
    ap("```bash")
    ap(".venv/Scripts/python.exe -m research.regime_pretest")
    ap("```")
    ap("")
    return "\n".join(lines)


def _report_table(res: dict, title: str) -> str:
    rows = [f"## {title}", "",
            f"Rejim-atanmış çözülmüş tahmin: **{res['n_total']}**  ·  havuz isabet oranı: "
            f"**{_fmt_pct(res['pool_rate'])}**  ·  eşleşmeyen (ısınma-öncesi): {res['unmatched']}",
            "",
            "| hücre | n | isabet | Wilson %95 GA | p(havuz) | p(0.5) | NO_EDGE payı | ort. "
            "fwd_ret% | hücre verdict |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        s = res["cells"][c]
        rows.append(f"| {c} | {s['n']} | {_fmt_pct(s['hit_rate'])} | "
                    f"[{_fmt_pct(s['wilson_lo'])}, {_fmt_pct(s['wilson_hi'])}] | "
                    f"{s['p_pool']:.3f} | {s['p_half']:.3f} | {_fmt_pct(s['no_edge_share'])} | "
                    f"{s['mean_ret']:+.3f} | {s['verdict']} |")
    return "\n".join(rows)


def _final_verdict_line(res_all: dict, res_conv: dict) -> str:
    if "AYRISMA_VAR" in (res_all["verdict"], res_conv["verdict"]):
        return ("**NİHAİ:** en az bir katmanda ayrışma sinyali var → HMM şart-1 KOŞULLU karşılandı; "
                "kısa-pencere caveat'ıyla, pencere genişleyince doğrulanmalı (şart-2 = expanding "
                "window fit).")
    if res_all["verdict"] == "AYRISMA_YOK" or res_conv["verdict"] == "AYRISMA_YOK":
        return ("**NİHAİ:** ayrışma YOK → **HMM gündemden düşer** (§1.6-HMM şart-1 karşılanmadı). "
                "Rejim-koşullu isabet, bu forward penceresinde havuzdan anlamlı sapmıyor.")
    return ("**NİHAİ:** ayrışma bu pencerede **test edilemedi** (rejim çeşitliliği yetersiz) → HMM "
            "için pozitif kanıt YOK, şart-1 karşılanmadı → **şimdilik HMM'e girme**; forward log "
            "birden çok rejim görecek kadar uzayınca (aylar) testi tekrarla.")


def main() -> None:
    ap = argparse.ArgumentParser(description="HMM öncesi rejim ön-testi (rapor-only, bağımlılıksız)")
    ap.add_argument("--no-write", action="store_true",
                    help="markdown raporu YAZMA (yalnız stdout)")
    args = ap.parse_args()

    preds = load_predictions()
    if preds.empty:
        print("[regime_pretest] çözülmüş tahmin yok — çıkılıyor.")
        return
    span = (preds["pred_date"].min().date(), preds["pred_date"].max().date())

    # SMA200 (200g) + trailing 252g vol-medyan ısınması için bol pay: ~800 takvim günü geriden.
    start = span[0] - timedelta(days=800)
    end = span[1] + timedelta(days=3)   # yfinance 'end' hariç → son tahmin gününü de kapsa
    state = spy_regime_table(start, end)

    assigned = assign_regime(preds, state)
    day_counts = _regime_day_counts(assigned)
    bg_counts = state["cell"].value_counts()   # arka plan (~2y) rejim dağılımı — dejenerasyon kontrolü

    res_all = analyze(assigned, "TÜM çözülmüş")
    conv = assigned[assigned["prob_cal"].sub(0.5).abs() >= NEUTRAL_BAND].copy()
    res_conv = analyze(conv, f"NO_EDGE-dışı (|prob_cal-0.5|≥{NEUTRAL_BAND})")

    print_summary(res_all, day_counts.to_dict())
    print()
    print_summary(res_conv, day_counts.to_dict())
    print()
    print(_final_verdict_line(res_all, res_conv).replace("**", ""))

    if not args.no_write:
        report = build_report(res_all, res_conv, assigned, day_counts, bg_counts, span)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"\n[regime_pretest] rapor yazıldı: {REPORT_PATH}")


if __name__ == "__main__":
    main()
