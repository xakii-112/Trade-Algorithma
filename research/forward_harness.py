"""
forward_harness.py — Forward (paper) veri biriktirme + dürüst kenar izleme orkestratörü.

prediction_tracker canlı tahminleri loglar (record) ve vadesi geleni gerçekleşenle
etiketler (resolve). Bu harness onun ÜSTÜNE oturur ve asıl soruyu cevaplar: "Forward
veri biriktikçe GERÇEK bir kenar ortaya çıkıyor mu, yoksa hâlâ yazı-tura mı?"

report()'tan farkı — yalnızca yön-isabeti + güven kovaları değil, model-kalitesi
metrikleri: AUC (prob_cal/pwin vs isabet), Brier skoru, ECE (beklenen kalibrasyon
hatası) + örneklem-boyutu KAPISI (yetersiz veriyle kenar iddiası yok) + her çalıştırmayı
data/forward_eval/history.csv'ye EKLER → kenarın N büyüdükçe nasıl evrildiğini gösteren
zaman serisi. Hiçbir emir göndermez; saf, ileriye-bakmasız muhasebe (EDGE_FINDINGS ilkesi).
`--snapshot` aynı koşuda öğrenme hafızasını da yeniler (posterior, kalibrasyon, drift);
`--report` ve `--history` yalnız okur.

Kullanım:
    python -m research.forward_harness --resolve --snapshot --report
    python -m research.forward_harness --report           # sadece güncel özet
    python -m research.forward_harness --history          # birikim zaman serisi
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR
from nihai.execution import prediction_tracker as pt

HISTORY = DATA_DIR / "forward_eval" / "history.csv"

# BULGU B-1 (2026-07-25): daily_summary.py bu iki json'u okuyup "Meta-kapı canlı sim" ve
# "Kapsama" satırlarını yazıyordu AMA dosyaları hiçbir kod ÜRETMİYORDU (repo genelinde tek
# referans daily_summary'nin kendisi + testi). Sonuç: günlük özet her gün "veri yok" derken
# aynı anda /api/predictions/accuracy aynı veriyi hesaplayıp "GATE_HURTS, uplift -0.013pp"
# döndürüyordu — sistemin en önemli canlı kanıtı rapora hiç girmiyordu. Test yalnız
# "dosya yok → None" yolunu doğruladığı için bulgu yeşil testlerin arkasında saklandı.
#
# Üretici NEDEN burada: daily_summary bilinçli olarak prediction_tracker'ı İMPORT ETMEZ
# (notifier → execution bağımlılığı yok; bkz. modül docstring'i). Bu harness ise zaten
# pt'yi kullanan ve her gün 12:00'de koşan (tek sağlıklı zamanlanmış görev) taraf.
GATE_SIM_JSON = DATA_DIR / "forward_eval" / "gate_sim.json"
COVERAGE_JSON = DATA_DIR / "forward_eval" / "coverage.json"
# Tur-60: gate_sim META-KAPIYI, ai_gate_sim İÇ-AI'ı ölçüyordu; geri kalan GÖLGE yüzeyler
# (nöral p_win, dir_bar/conf_bar, gözlemevi kolonları) canlıda hiç değerlendirilmiyordu.
# "Gölgeyi asıl sisteme al" kararı bir TUR işi değil GÜNLÜK bir ölçümdür: yüzeylerin çoğu
# bugün örneklem yetersizliğinden hükümsüz ve tek çare kanıtın BİRİKMESİDİR. Bu yüzden
# terfi ölçümü de aynı günlük göreve bağlandı (üretici NEDEN burada: yukarıdaki gerekçe).
SHADOW_PROMOTE_JSON = DATA_DIR / "forward_eval" / "shadow_promote.json"

# Kenar iddiası için asgari kapı: bunun altında "yetersiz veri" denir (gürültüde
# desen görme tuzağını engeller). 2 günlük ilk veri yanıltıcıydı (rejim-spesifik).
MIN_RESOLVED = int(os.getenv("FORWARD_MIN_RESOLVED", "500"))
MIN_DAYS = float(os.getenv("FORWARD_MIN_DAYS", "10"))
EDGE_PP = float(os.getenv("FORWARD_EDGE_PP", "3.0"))  # |isabet-50%| eşiği (yüzde-puan)

HISTORY_FIELDS = [
    "run_at", "resolved", "days_span", "symbols", "hit_rate",
    "auc_prob_cal", "auc_pwin", "brier_pwin", "ece_pwin",
    "avg_signed_ret_pct", "verdict",
]


# ---------------------------------------------------------------------------
# Metrikler (saf numpy, bağımlılıksız)
# ---------------------------------------------------------------------------
def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    """ROC-AUC (tie-düzeltmeli rank). Tek sınıf varsa tanımsız → None."""
    pos_n = int((y == 1).sum())
    neg_n = int((y == 0).sum())
    if pos_n == 0 or neg_n == 0:
        return None
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    # D2 (Tur-10): saf-Python rank döngüsü → cumsum ile vektörize (train_meta._auc ile
    # aynı dönüşüm; bit-exact — parite: test_perf_parity_round10).
    cum = np.concatenate(([0], np.cumsum(counts)[:-1]))
    avg_rank = cum + (counts + 1) / 2.0
    ranks = avg_rank[inv]
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - pos_n * (pos_n + 1) / 2.0) / (pos_n * neg_n))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    """Brier skoru: mean((p-y)^2). Düşük=iyi; 0.25 = sabit 0.5 tahmini (referans)."""
    return float(np.mean((p - y) ** 2))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Beklenen Kalibrasyon Hatası: |tahmin-gerçekleşen| ortalaması, kova-ağırlıklı.
    0 = mükemmel kalibre; büyük = güven gerçekle uyumsuz."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    n = len(p)
    err = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not m.any():
            continue
        err += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(err)


def _pwin_label(row: dict) -> int | None:
    """pwin sütununun DOĞRU etiketi = TARAF (pwin_side) sonucu, yön-isabeti (hit) DEĞİL.

    B4 (2026-07-09): pwin, İŞLEM tarafının (pwin_side ∈ LONG/SHORT) kazanma olasılığıdır;
    `hit` ise sinyal YÖNÜNÜN (direction ±) isabetidir. SHORT adaylarda taraf sonucu yönün
    TERSİdir → pwin'i `hit` ile etiketlemek SHORT'ları ters ölçer (AUC/Brier/ECE bozulur).
    _side_return taraf-işaretli getiriyi verir; >0 → taraf kazandı. pwin_side yoksa None
    (eski şema satırı — taraf bilinemez, dürüstçe dışarıda bırakılır)."""
    ret = pt._side_return(row)
    if ret is None:
        return None
    return 1 if ret > 0 else 0


def _col(rows: list[dict], name: str, *, label_fn=None) -> tuple[np.ndarray, np.ndarray] | None:
    """Sayısal `name` sütunu + hizalı etiket dizisi; o sütunu numerik olan satırlar.
    label_fn: satır→etiket(0/1) VEYA None (satır atlanır). Verilmezse etiket = r['hit']
    (yön-isabeti). Yetersiz/eksik → None (metrik atlanır, çökmez)."""
    ps, ys = [], []
    for r in rows:
        v = r.get(name)
        if v is None or v in ("", "None"):
            continue
        try:
            p = float(v)
        except (ValueError, TypeError):
            continue
        if label_fn is None:
            try:
                y: int | None = int(r["hit"])
            except (ValueError, TypeError, KeyError):
                continue
        else:
            y = label_fn(r)
            if y is None:
                continue
        ps.append(p)
        ys.append(y)
    if len(ps) < 20:
        return None
    return np.array(ys, dtype=float), np.clip(np.array(ps, dtype=float), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Değerlendirme
# ---------------------------------------------------------------------------
def evaluate() -> dict:
    """Çözülmüş forward tahminlerden zengin metrik + örneklem-kapılı dürüst verdict."""
    # Tur-11 perf: sınırsız büyüyen predictions_log.csv koşu başına ÜÇ kez okunuyordu
    # (coverage + report + aşağıdaki resolved süzgeci) → D4'ün rows-paylaşım API'siyle BİR kez.
    # Dosya yokken None geçilir: coverage/report kendi dosya-yok dallarını birebir korur.
    log_rows = pt._read_log_rows() if pt.PRED_LOG.exists() else None
    cov = pt.coverage(rows=log_rows)
    base = pt.report(rows=log_rows)
    out: dict = {
        "resolved": cov["resolved"], "total": cov["total"], "pending": cov["pending"],
        "days_span": cov["days_span"], "symbols": cov["symbols"],
        "first_logged": cov["first_logged"], "last_logged": cov["last_logged"],
        "hit_rate": base.get("hit_rate"),
        # Tur-83: hüküm 0,5'e beraberlik HARİÇ isabeti kıyaslar (pt.report notu); tarihçe
        # sürekliliği için `hit_rate` (beraberlik = ıska) aynen yazılmaya devam eder.
        "hit_rate_excl_ties": base.get("hit_rate_excl_ties"),
        "tie_share": base.get("tie_share"),
        "avg_signed_ret_pct": base.get("avg_signed_ret_pct"),
        "buckets": base.get("buckets", {}),
        "meta_gate": base.get("meta_gate"),   # canlı kapı simülasyonu (ACI DERS #3)
        "auc": {}, "brier": {}, "ece": {},
    }
    if not pt.PRED_LOG.exists():
        out["verdict"] = {"label": "NO_DATA", "text": "Henüz tahmin yok."}
        return out
    resolved = [r for r in (log_rows or []) if r.get("hit") in ("0", "1")]

    for name in ("prob_cal", "pwin", "confidence"):
        # B4: pwin etiketi TARAF sonucudur (pwin_side), yön-isabeti (hit) değil.
        # prob_cal/confidence YÖN olasılıkları → hit ile etiketlenir (varsayılan).
        col = _col(resolved, name, label_fn=_pwin_label if name == "pwin" else None)
        if col is None:
            continue
        y, p = col
        a = _auc(y, p)
        if a is not None:
            out["auc"][name] = round(a, 4)
        out["brier"][name] = round(_brier(y, p), 4)
        out["ece"][name] = round(_ece(y, p), 4)

    out["verdict"] = _verdict(out)
    return out


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Binom oran için Wilson %95 güven aralığı (k başarı / n deneme). Normal-yaklaşımdan
    küçük/orta örneklemde daha doğru; asla [0,1] dışına taşmaz. Dönüş: (alt, üst)."""
    if n <= 0:
        return 0.0, 1.0
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = (z * float(np.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n)))) / denom
    return center - half, center + half


def _verdict(ev: dict) -> dict:
    """Örneklem-kapılı dürüst verdict. Kapı altında ASLA kenar iddia etme.

    label: NO_DATA | INSUFFICIENT | NO_EDGE | WEAK_POSITIVE | INVERTED.

    Nokta-tahmin kenarı (|isabet-50%| ≥ EDGE_PP) TEK BAŞINA yetmez: kenar binom %95
    güven aralığıyla da doğrulanmalı (honest-improvement 2026-07-09). GA 0.5'i içeriyorsa
    'zayıf-pozitif'/'ters' iddiası gürültüdür → NO_EDGE. WEAK_POSITIVE yalnız GA ALT sınırı,
    INVERTED yalnız GA ÜST sınırı 0.5'i net geçtiğinde döner (çoklu-örneklem dürüstlüğü)."""
    n = ev.get("resolved", 0)
    if not n or ev.get("hit_rate") is None:
        return {"label": "NO_DATA", "edge_pp": 0.0,
                "text": "Çözülmüş tahmin yok (önce --resolve / piyasa açık olmalı)."}
    if n < MIN_RESOLVED or (ev.get("days_span") or 0) < MIN_DAYS:
        return {"label": "INSUFFICIENT", "edge_pp": 0.0,
                "text": (f"Yetersiz forward örneklem: {n} çözülmüş / {ev.get('days_span')} gün "
                         f"(kapı ≥{MIN_RESOLVED} ve ≥{MIN_DAYS}g). Biriktirmeye devam; "
                         "bu eşik altında kenar İDDİA EDİLMEZ (gürültüde desen tuzağı).")}
    # Tur-83: beraberlik `hit`te ıska yazılır → yazı-tura tahmincinin beklenen isabeti 0,5
    # değil (1 − beraberlik)/2'dir. 0,5'e kıyaslanan oran da, Wilson aralığının n'i de
    # beraberlik HARİÇtir (`pt.verdict` ile aynı kural — kardeş hüküm sessizce ayrışmasın).
    hx = ev.get("hit_rate_excl_ties")
    hr = float(hx if hx is not None else ev["hit_rate"])
    n_hukum = (int(round(n * (1.0 - float(ev.get("tie_share") or 0.0)))) if hx is not None
               else int(n))
    edge = abs(hr - 0.5) * 100.0
    # Tur-106: rapor başlığı beraberlik DAHİL isabeti basar; hüküm oranı ayrıca adlandırılmazsa iki sayı çelişir görünür.
    hk = " beraberlik hariç" if hx is not None else ""
    if edge < EDGE_PP:
        return {"label": "NO_EDGE", "edge_pp": round(edge, 1),
                "text": f"İsabet ~%50 ({hr*100:.1f}%{hk}) → forward kenar YOK (offline OOS-AUC≈0.5 doğrulanır)."}
    lo95, hi95 = _wilson_ci(int(round(hr * n_hukum)), n_hukum)
    if hr > 0.5:
        if lo95 <= 0.5:
            return {"label": "NO_EDGE", "edge_pp": round(edge, 1),
                    "text": (f"İsabet %{hr*100:.1f}{hk} ama %95 GA [{lo95*100:.1f},{hi95*100:.1f}] "
                             "0.5'i içeriyor → kenar istatistiksel olarak doğrulanmadı (örneklemi büyüt).")}
        return {"label": "WEAK_POSITIVE", "edge_pp": round(edge, 1),
                "text": (f"İsabet %{hr*100:.1f}{hk} ({edge:.1f}pp üstün, %95 GA alt {lo95*100:.1f}>50) → "
                         "zayıf-pozitif; örneklemi büyütüp doğrula.")}
    if hi95 >= 0.5:
        return {"label": "NO_EDGE", "edge_pp": round(edge, 1),
                "text": (f"İsabet %{hr*100:.1f}{hk} ama %95 GA [{lo95*100:.1f},{hi95*100:.1f}] "
                         "0.5'i içeriyor → ters kenar doğrulanmadı.")}
    # TUR-47: bu satır *"fade/yön-çevirme test edilebilir"* diyordu — oysa o test 2026-06-25'te
    # KOŞULDU ve CEVAP VERDİ (EDGE_FINDINGS §17-D, 5.936 tahmin, kronolojik 70/30):
    # anti-kalibrasyon GERÇEK ve STABİL ama fade OOS'ta isabeti %54,6 yaparken ortalama
    # getiriyi −0,071% (NEGATİF) bırakıyor — isabet ile beklenti KOPUK. Cevabı TAŞIMAYAN bir
    # işaretçi, bir sonraki turu kapanmış bir kuyuyu yeniden kazmaya gönderir.
    return {"label": "INVERTED", "edge_pp": round(edge, 1),
            "text": (f"İsabet %{hr*100:.1f}{hk} (%95 GA üst {hi95*100:.1f}<50) → yön ölçümü ters "
                     "tarafta. ⛔ Fade/yön-çevirme ZATEN ÖLÇÜLDÜ ve KAPANDI (§17-D): fade "
                     "isabeti %54,6 çıkarıyor AMA ort. getiri −0,071% — isabet beklenti "
                     "değildir. Yeniden koşacaksan yeni bir SORU ile koş "
                     "(research/calibration_fade_test).")}


# ---------------------------------------------------------------------------
# Birikim zaman serisi (append-only)
# ---------------------------------------------------------------------------
def snapshot(ev: dict) -> None:
    """Bu çalıştırmanın metriklerini history.csv'ye EKLE → kenarın N büyüdükçe evrimi."""
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    fresh = not HISTORY.exists()
    with HISTORY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if fresh:
            w.writeheader()
        w.writerow({
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "resolved": ev["resolved"], "days_span": ev["days_span"],
            "symbols": ev["symbols"], "hit_rate": ev.get("hit_rate"),
            "auc_prob_cal": ev["auc"].get("prob_cal", ""),
            "auc_pwin": ev["auc"].get("pwin", ""),
            "brier_pwin": ev["brier"].get("pwin", ""),
            "ece_pwin": ev["ece"].get("pwin", ""),
            "avg_signed_ret_pct": ev.get("avg_signed_ret_pct"),
            "verdict": ev["verdict"]["label"],
        })


def _shadow_promote(rows: list[dict]) -> dict:
    """Gölge yüzey terfi ölçümü — özet için KÜÇÜLTÜLMÜŞ hâli (eşik taraması yazılmaz).

    Tam `sweep` listeleri dosyayı ~10× büyütür ve `daily_summary` yalnız etiketi okur;
    ayrıntı isteyen `python -m research.shadow_promote` koşar (aynı fonksiyon, aynı sayı)."""
    from research.shadow_promote import run as _run
    rep = _run(rows)
    for s in rep.get("surfaces", {}).values():
        s.pop("sweep", None)
    return rep


def publish_side_reports(rows: list[dict] | None = None) -> dict[str, bool]:
    """B-1: gate_sim + coverage'ı daily_summary'nin okuduğu json'lara YAZ (atomik).

    Salt-okunur hesap + atomik yazım; karar vermez, eşik değiştirmez. Her iki dosyaya
    'generated_at' ve 'source' eklenir ki özet bayat bir dosyayı taze sanmasın.
    Bir tarafın hesabı patlarsa DİĞERİ yine yazılır (fail-soft): bu bir gözlem hattı,
    onun yüzünden günlük harness çökmemeli. Dönen dict: hangi dosya yazıldı."""
    if rows is None:
        rows = pt._read_log_rows() if pt.PRED_LOG.exists() else []
    damga = datetime.now().isoformat(timespec="seconds")
    yazildi: dict[str, bool] = {}
    for ad, hesapla, yol in (
        ("gate_sim", lambda: pt.gate_sim([r for r in rows if r.get("hit") in ("0", "1")]),
         GATE_SIM_JSON),
        ("coverage", lambda: pt.coverage(rows), COVERAGE_JSON),
        # Geç import: `shadow_promote` numpy'ı ve pt'yi çeker; yan-rapor yolu dışında
        # harness'in geri kalanına maliyet bindirmesin.
        ("shadow_promote", lambda: _shadow_promote(rows), SHADOW_PROMOTE_JSON),
    ):
        try:
            veri = dict(hesapla())
            veri["generated_at"] = damga
            veri["source"] = "research.forward_harness"
            yol.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(yol, json.dumps(veri, ensure_ascii=False, indent=2, default=str))
            yazildi[ad] = True
        except Exception as e:  # noqa: BLE001 — gözlem hattı harness'i düşürmemeli
            print(f"[harness] {ad}.json yazılamadı: {e}")
            yazildi[ad] = False
    return yazildi


def read_history() -> list[dict]:
    if not HISTORY.exists():
        return []
    with HISTORY.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Raporlama
# ---------------------------------------------------------------------------
def _print_report(ev: dict) -> None:
    print("=" * 64)
    print("FORWARD VERİ BİRİKTİRME HARNESS — dürüst kenar izleme (paper)")
    print("=" * 64)
    print(f"Kapsama : {ev['resolved']} çözülmüş / {ev['total']} toplam "
          f"({ev['pending']} bekliyor) | {ev['symbols']} sembol | {ev['days_span']} gün")
    print(f"Aralık  : {ev['first_logged']} → {ev['last_logged']}")
    hr = ev.get("hit_rate")
    if hr is not None:
        print(f"Yön isabeti : {hr*100:.1f}%   (50% = yazı-tura) | "
              f"ort getiri {ev.get('avg_signed_ret_pct')}%/tahmin")
        # Tur-13 (RAPOR-only; verdikt makinesine dokunmaz): örneklem belirsizliği + şans testi.
        # "anlamlı" çıkması KENAR demek DEĞİLDİR — %50'den sapma isabetin ALTINDA da anlamlı
        # olabilir (canlıda %47.x böyle). Soru yalnız: "gözlem şansla açıklanabilir mi?"
        n_res = int(ev.get("resolved") or 0)
        if n_res >= 30:
            from research.deflated import binom_p_value, wilson_ci
            k = int(round(hr * n_res))
            lo, hi = wilson_ci(k, n_res)
            p = binom_p_value(k, n_res)
            p_s = "nan" if p != p else f"{p:.2g}"
            karar = "şanstan ayırt edilemez" if (p != p or p >= 0.05) else "%50'den sapma anlamlı"
            print(f"İsabet %95 CI : [{lo*100:.1f}%, {hi*100:.1f}%] | binom p(H0=%50)={p_s} → {karar}")
    if ev["auc"]:
        print("-" * 64)
        print("Model kalitesi (olasılık vs gerçekleşen isabet):")
        for name in ("prob_cal", "pwin", "confidence"):
            if name in ev["auc"] or name in ev["brier"]:
                a = ev["auc"].get(name, "—")
                b = ev["brier"].get(name, "—")
                e = ev["ece"].get(name, "—")
                print(f"  {name:9}: AUC={a}  Brier={b}  ECE={e}")
        print("  (AUC 0.5=öngörü yok; Brier<0.25 iyi; ECE 0=mükemmel kalibre)")
    gs = ev.get("meta_gate")
    if gs:
        print("-" * 64)
        print(f"META-KAPI canlı simülasyonu (eşik {gs['threshold']}, aday n={gs['n_candidates']}) "
              f"— AUC değil KAPI karar verir:")
        if gs["label"] != "INSUFFICIENT":
            k, c = gs["kept"], gs["cut"]
            print(f"  tutulan n={k['n']} ort={k['avg_ret_pct']:+.3f}%  |  "
                  f"elenen n={c['n']} ort={c['avg_ret_pct']:+.3f}%")
        print(f"  [{gs['label']}] {gs['text']}")
    print("-" * 64)
    v = ev["verdict"]
    print(f"VERDİKT [{v['label']}]: {v['text']}")


def _print_history(rows: list[dict]) -> None:
    if not rows:
        print("[history] henüz snapshot yok (--snapshot ile ekle)")
        return
    print("=" * 64)
    print("BİRİKİM ZAMAN SERİSİ (her satır bir çalıştırma; kenar N büyüdükçe evrilir)")
    print("=" * 64)
    print(f"{'run_at':19} {'çöz':>6} {'gün':>5} {'isabet':>7} {'AUC_pc':>7} {'AUC_pw':>7} {'verdict':>13}")
    for r in rows[-20:]:
        print(f"{r['run_at'][:19]:19} {r['resolved']:>6} {str(r['days_span']):>5} "
              f"{str(r['hit_rate']):>7} {str(r['auc_prob_cal'] or '—'):>7} "
              f"{str(r['auc_pwin'] or '—'):>7} {r['verdict']:>13}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward veri biriktirme + dürüst kenar izleme")
    ap.add_argument("--resolve", action="store_true", help="önce vadesi geleni gerçekleşenle etiketle")
    ap.add_argument("--snapshot", action="store_true", help="metrikleri history.csv'ye ekle")
    ap.add_argument("--report", action="store_true", help="güncel özet + verdict yazdır")
    ap.add_argument("--history", action="store_true", help="birikim zaman serisini yazdır")
    args = ap.parse_args()
    if not any((args.resolve, args.snapshot, args.report, args.history)):
        ap.print_help()
        return
    if args.resolve:
        res = pt.resolve()
        print(f"[harness] resolve: {res}")
    if args.history and not (args.report or args.snapshot):
        _print_history(read_history())
        return
    ev = evaluate()
    if args.snapshot:
        # Öğrenme, pano ziyaretinden bağımsız olarak çözülmüş sonuçlarla yenilenir.
        # Varsayılan pencere sabittir; rapor/önizleme çağrıları prior'ları yazmaz.
        try:
            from nihai.ai.learning import update_from_predictions
            state = update_from_predictions()
            if not state["saved"]:
                raise OSError("öğrenme durumu diske yazılamadı")
            print(f"[harness] öğrenme → n={state['strategies'].get('GLOBAL', {}).get('n', 0)}"
                  f" drift={state['retrain_candidate']}")
        except Exception as e:  # noqa: BLE001 — öğrenme hatası diğer ölçümleri durdurmaz
            print(f"[harness] ÖĞRENME GÜNCELLENEMEDİ: {e}")
        snapshot(ev)
        print(f"[harness] snapshot → {HISTORY}")
        # B-1: aynı koşuda daily_summary'nin beklediği yan raporları da tazele.
        durum = publish_side_reports()
        print(f"[harness] yan raporlar → {durum}")
    if args.report:
        _print_report(ev)
    if args.history:
        print()
        _print_history(read_history())
    if args.snapshot:
        try:
            from research.auto_retrain import run as retrain
            result = retrain()
            print(f"[harness] aday eğitim → {result['status']} "
                  f"{result.get('verdict', '')} n={result.get('rows', 0)}")
        except Exception as e:  # noqa: BLE001 — diğer günlük ölçümler tamamlandı
            print(f"[harness] ADAY EĞİTİMİ TAMAMLANAMADI: {e}")


if __name__ == "__main__":
    main()
