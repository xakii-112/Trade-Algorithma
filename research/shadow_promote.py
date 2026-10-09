"""
shadow_promote.py — GÖLGEDE KOŞAN her yüzeyi AYNI sözleşmeyle sınayan terfi ölçeri.

Neden: projede birden çok "gölge" yüzey birikiyor (meta-kapı p_win, nöral p_win, gölge
yön/güven `dir_bar`/`conf_bar`, gözlemevi kolonları `mkt_health`/`har_rv_fc`/`spread_bps`/
`hmm_*`, iç-AI verdikti). `prediction_tracker.gate_sim` YALNIZ meta-kapıyı, `ai_gate_sim`
YALNIZ AI vetosunu ölçüyordu; geri kalanı canlıda hiç değerlendirilmedi. "Gölgeyi asıl
sisteme al" kararı ancak HEPSİ aynı ölçekte ölçülünce verilebilir.

Soru her yüzey için AYNI: *bu gölge kolon bir KAPI olsaydı (eşiği geçmeyen adayı elesek),
ortalama getiri tabandan iyileşir miydi?* Bu, `gate_sim`in sorusunun genelleştirilmişidir
ve ACI DERS #3 gereği AUC değil KAPI SİMÜLASYONU karar verir.

DÜRÜSTLÜK KAPILARI:
  1. ÖRNEKLEM  — tutulan ve elenen tarafta >= `--min-side` çözülmüş satır yoksa hüküm YOK
                 (INSUFFICIENT). Kenar uydurulmaz.
  2. KÜMELENME — 1m barlarda 5-bar ufuklu satırlar ÖRTÜŞÜR; naif t sahte anlamlılık üretir.
                 t istatistiği (sembol, gün) kümesine göre KÜMELE-DAYANIKLI (CR0) hesaplanır.
  3. ÇOKLU TEST— her yüzeyde birden çok eşik denenir. Bonferroni'li p (p_adj = p x #eşik)
                 raporlanır ve TERFİ kararı p_adj'a bakar.
  4. EVREN     — her yüzey KENDİ eyleyeceği evrende ölçülür: işlem-tarafı olan yüzeyler
                 (pwin/neural/ai) taraf-işaretli getiride, yön yüzeyleri (dir_bar/gözlemevi)
                 yön-işaretli `fwd_ret_pct`te. Yanlış evrende ölçmek yanlı sonuç verir.
  5. KİRLİLİK  — bu ölçüm CANLI LOG üzerindedir ve İN-SAMPLE'dır (eşik burada seçilirse
                 aynı veride doğrulanmış olmaz). Rapor bunu her koşuda yazar.

Kullanım (repo kökünden):
    python -m research.shadow_promote
    python -m research.shadow_promote --min-side 50 --json data/forward_eval/shadow_promote.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nihai.console import ensure_utf8_stdout  # noqa: E402
from nihai.execution.prediction_tracker import (  # noqa: E402
    _bucket_stats,
    _read_log_rows,
    _side_return,
)

# Terfi eşiği: uplift POZİTİF + çoklu-teste düzeltilmiş p < bu değer.
PROMOTE_P = 0.05
MIN_SIDE_DEFAULT = 30


# --------------------------------------------------------------------------- yardımcı
def _f(row: dict, key: str) -> float | None:
    """Hücreyi float'a çevir; boş/bozuk/sonsuz → None (CSV'de eksik kolon "" gelir)."""
    v = row.get(key)
    if v is None or v == "" or v == "None":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _cluster(row: dict) -> tuple[str, str]:
    """(sembol, gün) kümesi — örtüşen ufuk yüzünden bağımsız olmayan satırları gruplar."""
    return (str(row.get("symbol") or "?"), str(row.get("logged_at") or "")[:10])


def _dir_return(row: dict) -> float | None:
    """YÖN-işaretli ileri getiri (%) — `fwd_ret_pct` zaten yönle işaretlidir.

    Referansı satırın KENDİ giriş baskısıdır (`price`); bu yüzden girişin bar kapanışından
    ne kadar uzakta olduğu (`slip_bps`) bu sayıya DOĞRUDAN girer. Bkz. `_bar_return`."""
    return _f(row, "fwd_ret_pct")


def _bar_return(row: dict) -> float | None:
    """Aynı ileri fiyat, ORTAK referanstan: (fwd − bar_close)/bar_close, yönle işaretli.

    KAYMA ARTEFAKTI KONTROLÜ. `_dir_return` her satırı KENDİ giriş baskısına böler ve
    kimlik olarak `ret_live ≈ ret_bar − slip` tutar: girişi bar kapanışının üstünde olan
    her satır, ödediği farkı ölçümde GERİ VERİR. Bu yüzden "girişi kovalanmış satırları
    ele" diyen HER kural, hiçbir bilgi taşımasa bile pozitif uplift üretir — ölçülen şey
    kuralın ayırt etmesi değil, bölenin kendisidir.

    Ortak referans o kimliği kırar: `fwd_price` DEĞİŞMEZ, yalnız payda tüm satırlarda
    aynı tanımdan gelir. Bir kuralın uplift'i burada İŞARET DEĞİŞTİRİYORSA kural
    bilgi değil giriş-fiyatı ofseti ölçüyordur (Tur-39 dersi: bir kaybı açıklarken
    piyasa hipotezinden ÖNCE ölçüm hipotezi elenmelidir)."""
    fwd, bc = _f(row, "fwd_price"), _f(row, "bar_close")
    if fwd is None or bc is None or bc <= 0:
        return None
    raw = (fwd - bc) / bc * 100.0
    d = row.get("direction")
    if d not in ("+", "-"):
        return None
    return raw if d == "+" else -raw


def _cluster_t(vals: np.ndarray, keep: np.ndarray, groups: list[Any]) -> tuple[float, float]:
    """(t, p) — tutulan/elenen ortalama farkının KÜMELE-DAYANIKLI (CR0) testi.

    r_i = a + b*keep_i + e_i modelinde b farkın kendisidir. Naif SE bu veride sahte
    anlamlılık üretir (aynı sembol-gününden onlarca örtüşen satır). Küme sayısı 2'den
    azsa (0.0, 1.0) döner — hüküm yok."""
    idx: dict[Any, int] = {}
    gid = np.empty(len(groups), dtype=int)
    for i, g in enumerate(groups):
        gid[i] = idx.setdefault(g, len(idx))
    n_g = len(idx)
    if n_g < 2 or len(vals) < 4:
        return 0.0, 1.0
    X = np.column_stack([np.ones(len(vals)), keep.astype(float)])
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta = XtX_inv @ (X.T @ vals)
        resid = vals - X @ beta
        meat = np.zeros((2, 2))
        for g in range(n_g):
            m = gid == g
            xg = X[m].T @ resid[m]
            meat += np.outer(xg, xg)
        # CR0 + küçük-küme düzeltmesi (Cameron-Miller): G/(G-1)
        cov = XtX_inv @ meat @ XtX_inv * (n_g / max(n_g - 1, 1))
        se = math.sqrt(max(float(cov[1, 1]), 0.0))
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 1.0
    if se <= 0.0:
        # TAM AYRIŞMA (her küme içinde artık sıfır) burada biter: SE=0 iken t tanımsızdır.
        # Dönen (0, 1) "anlamsız" DEĞİL "hüküm yok" demektir ve muhafazakâr yöndür — bir
        # kuralın kanıtı, artıkları sıfırlayan dejenere bir kurgudan gelemez. Gerçek
        # veride ateşlenmez; kurgu testlerde ateşlendiği için burada AÇIKÇA yazılıdır.
        return 0.0, 1.0
    t = float(beta[1]) / se
    # Normal yaklaşım; küme sayısı azken p muhafazakâr okunmalı (rapor n_clusters yazar).
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return round(t, 3), round(p, 5)


def _sim_one(scores: np.ndarray, rets: np.ndarray, groups: list[Any], thr: float,
             keep_high: bool, *, min_side: int) -> dict:
    """Tek eşik için tutulan/elenen/taban istatistiği + kümele-dayanıklı t."""
    keep = (scores >= thr) if keep_high else (scores <= thr)
    kept, cut = rets[keep].tolist(), rets[~keep].tolist()
    out: dict[str, Any] = {"thr": round(float(thr), 6), "keep": "high" if keep_high else "low",
                           "kept": _bucket_stats(kept), "cut": _bucket_stats(cut)}
    if len(kept) < min_side or len(cut) < min_side:
        out["uplift_pp"] = out["t"] = out["p"] = None
        return out
    base_avg = float(rets.mean())
    out["uplift_pp"] = round(out["kept"]["avg_ret_pct"] - base_avg, 4)
    out["t"], out["p"] = _cluster_t(rets, keep, groups)
    return out


def _thresholds(vals: np.ndarray) -> list[float]:
    """Aday eşikler: 0..1 aralığındaki kolonlarda sabit ızgara, diğerlerinde ÇEYREKLER.

    Sabit ızgara olasılık kolonlarının (p_win, hmm_prob, conf_bar) yorumunu korur;
    ölçek-serbest kolonlarda (har_rv_fc, spread_bps, mkt_health) veri kendi eşiğini
    verir — sabit bir sayı uydurmak ölçüm değil VARSAYIM olurdu."""
    if vals.size == 0:
        return []
    if float(vals.min()) >= 0.0 and float(vals.max()) <= 1.0:
        return [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    return sorted({round(float(q), 6) for q in np.quantile(vals, [0.2, 0.4, 0.6, 0.8])})


def evaluate(name: str, rows: list[dict], column: str,
             ret_fn: Callable[[dict], float | None], *,
             min_side: int = MIN_SIDE_DEFAULT, universe: str = "",
             higher_is_better: bool | None = None) -> dict:
    """Tek gölge kolonun KAPI olarak değeri. Eşik taraması x iki yön (yüksek/düşük tut).

    `higher_is_better=True` → kolon bir KAZANMA OLASILIĞIDIR (p_win, conf, hmm_mode_prob):
    orada "düşük tut" kazanırsa bu bir kenar DEĞİL, TERS KALİBRASYONdur ve terfi
    ETTİRİLMEZ (ACI DERS #3/#7 — `prediction_tracker._calibration_verdict` ile aynı
    yasak; iki hüküm fonksiyonu aynı veride çelişik konuşamaz, Tur-47 dersi). Ölçüm
    yine RAPORLANIR, yalnız INVERTED etiketiyle. `None` → yön agnostik kolon
    (oynaklık/makas/sağlık): iki yön de meşru bir hipotezdir."""
    scores: list[float] = []
    rets: list[float] = []
    groups: list[Any] = []
    for r in rows:
        x = _f(r, column)
        if x is None:
            continue
        ret = ret_fn(r)
        if ret is None:
            continue
        scores.append(x)
        rets.append(ret)
        groups.append(_cluster(r))

    res: dict[str, Any] = {"name": name, "column": column, "universe": universe,
                           "n": len(scores), "n_clusters": len(set(groups)),
                           "min_side": min_side}
    if len(scores) < 2 * min_side:
        res["label"] = "INSUFFICIENT"
        res["text"] = (f"Çözülmüş+kolonlu satır {len(scores)} < {2 * min_side} — "
                       "bu yüzey hakkında canlı hüküm YOK (gölgede kalmalı).")
        return res

    sc, rt = np.asarray(scores, dtype=float), np.asarray(rets, dtype=float)
    sweep = [_sim_one(sc, rt, groups, t, hi, min_side=min_side)
             for hi in (True, False) for t in _thresholds(sc)]
    res["sweep"] = sweep
    res["base"] = _bucket_stats(rets)

    scored = [s for s in sweep if s.get("uplift_pp") is not None]
    if not scored:
        res["label"] = "INSUFFICIENT"
        res["text"] = (f"Hiçbir eşik örneklem kapısını geçmedi (her iki tarafta >= "
                       f"{min_side} gerekli) — hüküm YOK.")
        return res

    n_tests = len(scored)
    for s in scored:
        s["p_adj"] = round(min(1.0, s["p"] * n_tests), 5)
    best = max(scored, key=lambda s: s["uplift_pp"])
    res["n_tests"] = n_tests
    res["best"] = best
    up, p_adj = best["uplift_pp"], best["p_adj"]

    # TERS KALİBRASYON KAPISI — olasılık kolonunda "düşük tut" kazanmışsa bu bir kenar
    # DEĞİLDİR; modelin sıralaması ters demektir ve o sıralamaya işlem kurmak yasaktır.
    if higher_is_better and best["keep"] == "low" and up > 0:
        res["label"] = "INVERTED"
        res["text"] = (f"En iyi kapı TERS yönde (düşük tut thr={best['thr']}, "
                       f"+{up:.4f}pp, t={best['t']}): model olasılıkları TERS sıralıyor. "
                       "Bu bir kenar DEĞİLDİR (tersine çevirme yasak, ACI DERS #3/#7) — "
                       "terfi YOK, modelin canlı olasılık üretimi güvenilmezdir.")
        return res

    if up > 0 and p_adj < PROMOTE_P:
        res["label"] = "PROMOTE"
        res["text"] = (f"En iyi kapı ({best['keep']} thr={best['thr']}) tabandan +{up:.4f}pp, "
                       f"kümele-dayanıklı t={best['t']} p_adj={p_adj} → canlıda AYIRIYOR. "
                       "İN-SAMPLE: eşik burada seçildi, ileri veriyle doğrulanmalı.")
    elif up > 0:
        res["label"] = "HOLD"
        res["text"] = (f"En iyi kapı +{up:.4f}pp AMA p_adj={p_adj} >= {PROMOTE_P} — "
                       "gürültüden ayrışmıyor. Gölgede kalır, örneklem büyüsün.")
    else:
        worst = min(scored, key=lambda s: s["uplift_pp"])
        res["label"] = "REJECT"
        res["text"] = (f"Hiçbir eşik tabanı geçemedi (en iyi {up:.4f}pp, en kötü "
                       f"{worst['uplift_pp']:.4f}pp) → kapı olarak değer katmıyor.")
    return res


def evaluate_binary(name: str, rows: list[dict], *, keep_if: Callable[[dict], bool | None],
                    ret_fn: Callable[[dict], float | None],
                    min_side: int = MIN_SIDE_DEFAULT, universe: str = "") -> dict:
    """Eşiksiz (ikili) kural — ör. `dir_bar == direction` uyuşması, `ai_veto == 0`."""
    vals: list[float] = []
    keep: list[bool] = []
    groups: list[Any] = []
    for r in rows:
        k = keep_if(r)
        if k is None:
            continue
        ret = ret_fn(r)
        if ret is None:
            continue
        vals.append(ret)
        keep.append(bool(k))
        groups.append(_cluster(r))

    arr, kp = np.asarray(vals, dtype=float), np.asarray(keep, dtype=bool)
    res: dict[str, Any] = {"name": name, "column": "-", "universe": universe,
                           "n": int(arr.size), "n_clusters": len(set(groups)),
                           "min_side": min_side}
    if int(kp.sum()) < min_side or int((~kp).sum()) < min_side:
        res["label"] = "INSUFFICIENT"
        res["text"] = (f"Tutulan {int(kp.sum())} / elenen {int((~kp).sum())} "
                       f"(kapı {min_side}+{min_side}) — hüküm YOK.")
        return res
    base = _bucket_stats(arr.tolist())
    kept_st, cut_st = _bucket_stats(arr[kp].tolist()), _bucket_stats(arr[~kp].tolist())
    t, p = _cluster_t(arr, kp, groups)
    up = round(kept_st["avg_ret_pct"] - base["avg_ret_pct"], 4)
    res.update({"base": base, "n_tests": 1,
                "best": {"thr": None, "keep": "rule", "kept": kept_st, "cut": cut_st,
                         "uplift_pp": up, "t": t, "p": p, "p_adj": p}})
    if up > 0 and p < PROMOTE_P:
        res["label"] = "PROMOTE"
        res["text"] = f"Kural tabandan +{up:.4f}pp (t={t}, p={p}) → canlıda AYIRIYOR."
    elif up > 0:
        res["label"] = "HOLD"
        res["text"] = f"Kural +{up:.4f}pp AMA p={p} >= {PROMOTE_P} — gürültüden ayrışmıyor."
    else:
        res["label"] = "REJECT"
        res["text"] = f"Kural tabanı geçemedi ({up:.4f}pp) → değer katmıyor."
    return res


# ------------------------------------------------------------------------ OOS onayı
def _apply_rule(rows: list[dict], spec: dict, rule: dict, ret_fn: Callable,
                *, min_side: int) -> dict:
    """IS'te seçilmiş kuralı BAŞKA bir satır kümesine uygula (eşik seçimi YOK).

    Terfinin tek dürüst sınavı budur: eşik bir veride seçilip AYNI veride ölçülürse
    "kenar" seçim sürecinin kendisidir (§PBO). Burada eşik SABİT gelir."""
    vals: list[float] = []
    keep: list[bool] = []
    groups: list[Any] = []
    for r in rows:
        if spec.get("keep_if"):
            k = spec["keep_if"](r)
            if k is None:
                continue
        else:
            x = _f(r, spec["column"])
            if x is None:
                continue
            k = (x >= rule["thr"]) if rule["keep"] == "high" else (x <= rule["thr"])
        ret = ret_fn(r)
        if ret is None:
            continue
        vals.append(ret)
        keep.append(bool(k))
        groups.append(_cluster(r))
    arr, kp = np.asarray(vals, dtype=float), np.asarray(keep, dtype=bool)
    out: dict[str, Any] = {"n": int(arr.size), "n_clusters": len(set(groups)),
                           "kept_n": int(kp.sum()), "cut_n": int((~kp).sum())}
    if out["kept_n"] < min_side or out["cut_n"] < min_side:
        out["label"] = "INSUFFICIENT"
        out["uplift_pp"] = out["t"] = out["p"] = None
        return out
    kept_st = _bucket_stats(arr[kp].tolist())
    out["kept"] = kept_st
    out["cut"] = _bucket_stats(arr[~kp].tolist())
    out["uplift_pp"] = round(kept_st["avg_ret_pct"] - float(arr.mean()), 4)
    out["t"], out["p"] = _cluster_t(arr, kp, groups)
    out["label"] = "CONFIRMED" if out["uplift_pp"] > 0 else "FAILED"
    return out


def _rows_with(spec: dict, rows: list[dict]) -> list[dict]:
    """Yüzeyin GERÇEKTEN ölçülebildiği satırlar. Bölmeyi tüm havuzda yapmak yanlıştır:
    gölge kolonlar farklı turlarda eklendi, bir kolonun tüm satırları zaman ekseninin
    SONUNDA toplanabilir → IS yarısı o kolondan tamamen boş çıkar (ilk koşuda dir_bar
    tam olarak böyle 'NO_RULE' vermişti)."""
    if spec.get("keep_if"):
        return [r for r in rows if spec["keep_if"](r) is not None]
    return [r for r in rows if _f(r, spec["column"]) is not None]


def _split_by_time(rows: list[dict], oos_frac: float) -> tuple[list[dict], list[dict]]:
    """Satırları `logged_at`e göre sırala → (IS, OOS). Zaman-bölmesi rastgele değil:
    örtüşen barlar ve rejim sürekliliği rastgele bölmede IS/OOS'u birbirine sızdırır."""
    ordered = sorted(rows, key=lambda r: str(r.get("logged_at") or ""))
    cut = int(len(ordered) * (1.0 - oos_frac))
    return ordered[:cut], ordered[cut:]


# --------------------------------------------------------------------------- yüzeyler
def _dirbar_agrees(r: dict) -> bool | None:
    # Tur-83: yayınlanan yön bar kapanışına geçti (`direction == dir_bar`); yüzeyin sorusu
    # "bar varyantı ↔ CANLI varyant" olarak kalsın diye canlı taraf `dir_live`dan okunur —
    # eski satırlarda (kolon boş) `direction` zaten canlı varyanttı.
    db, d = (r.get("dir_bar") or ""), (r.get("dir_live") or r.get("direction") or "")
    if db not in ("+", "-") or d not in ("+", "-"):
        return None
    return db == d


def _ai_no_veto(r: dict) -> bool | None:
    v = r.get("ai_veto")
    if v in (None, "", "None"):
        return None
    return str(v).strip() in ("0", "0.0", "False", "false")


# Yüzey kataloğu — TEK kaynak. `u`: evren ("side" | "dir"), `hib`: yüksek=iyi mi
# (None → yön agnostik), `keep_if`: eşiksiz ikili kural.
SURFACES: tuple[dict[str, Any], ...] = (
    {"key": "meta_pwin", "name": "Meta-kapı p_win", "column": "pwin",
     "u": "side", "hib": True},
    {"key": "neural_pwin", "name": "Nöral gölge p_win", "column": "neural_pwin",
     "u": "side", "hib": True},
    {"key": "conf_bar", "name": "Gölge güven (bar kapanışı)", "column": "conf_bar",
     "u": "dir", "hib": True},
    {"key": "mkt_health", "name": "Veri sağlığı skoru", "column": "mkt_health",
     "u": "dir", "hib": None},
    {"key": "har_rv_fc", "name": "HAR-RV oynaklık öngörüsü", "column": "har_rv_fc",
     "u": "dir", "hib": None},
    {"key": "spread_bps", "name": "Corwin-Schultz makas", "column": "spread_bps",
     "u": "dir", "hib": None},
    {"key": "hmm_mode_prob", "name": "HMM rejim olasılığı", "column": "hmm_mode_prob",
     "u": "dir", "hib": None},
    {"key": "hmm_switch", "name": "HMM rejim-değişim olasılığı", "column": "hmm_switch",
     "u": "dir", "hib": None},
    # Tur-76 — hacim profili ailesi. Üçünün de `hib=None`dur: bir POC uzaklığının ya da
    # değer-alanı konumunun "yüksek olanı iyidir" YÖNÜ ÖN-KAYITLI DEĞİLDİR ve olmadığı
    # için kapı ters kalibrasyon kilidini uygulamaz — yön VERİDEN çıkar. Defterin ayrık
    # yüzey sayısı 10 → 13 oldu ve Bonferroni çıtası herkes için o oranda YÜKSELDİ;
    # bu bir maliyet değil, çoklu-testin makinede olmasının ta kendisidir (Tur-46/55).
    {"key": "poc_dist_bps", "name": "POC uzaklığı (bps)", "column": "poc_dist_bps",
     "u": "dir", "hib": None},
    {"key": "value_area_pos", "name": "Değer alanı konumu", "column": "value_area_pos",
     "u": "dir", "hib": None},
    {"key": "cvd_divergence", "name": "CVD uyuşmazlığı", "column": "cvd_divergence",
     "u": "dir", "hib": None},
    {"key": "dir_bar_agree", "name": "Gölge yön <-> canlı yön uyuşması", "column": "-",
     "u": "dir", "hib": None, "keep_if": _dirbar_agrees},
    {"key": "ai_veto", "name": "İç-AI veto yok", "column": "-",
     "u": "side", "hib": None, "keep_if": _ai_no_veto},
)


def _eval_spec(spec: dict, rows: list[dict], ret_fn: Callable, *, min_side: int,
               universe: str) -> dict:
    """Katalog satırını uygun ölçere yönlendir (eşikli / ikili)."""
    if spec.get("keep_if"):
        return evaluate_binary(spec["name"], rows, keep_if=spec["keep_if"],
                               ret_fn=ret_fn, min_side=min_side, universe=universe)
    return evaluate(spec["name"], rows, spec["column"], ret_fn, min_side=min_side,
                    universe=universe, higher_is_better=spec["hib"])


def run(rows: list[dict] | None = None, *, min_side: int = MIN_SIDE_DEFAULT,
        oos_frac: float = 0.4) -> dict:
    """Tüm gölge yüzeyleri sına → {"surfaces": {...}}. Ağ/veri sağlayıcı GEREKTİRMEZ.

    Her yüzey İKİ kez ölçülür: (a) tüm veride tarama (hangi kural?), (b) kural IS'te
    seçilip OOS'ta uygulanır (kural gerçekten var mı?). TERFİ ikisini de ister —
    tek başına tarama, seçim sürecinin gürültüsünü kenar diye raporlar."""
    rows = _read_log_rows() if rows is None else rows
    resolved = [r for r in rows if str(r.get("hit") or "") in ("0", "1")]

    # U1: işlem TARAFI olan adaylar (taraf-işaretli getiri) — kapı buraya kurulur.
    side_rows = [r for r in resolved if _side_return(r) is not None]
    # U2: her yön tahmini (yön-işaretli getiri) — sembole ait yüzeyler burada eyler.
    dir_rows = [r for r in resolved if _dir_return(r) is not None]
    pools = {"side": (side_rows, _side_return, "U1 taraf"),
             "dir": (dir_rows, _dir_return, "U2 yön")}

    out: dict[str, Any] = {
        "rows": len(rows), "resolved": len(resolved),
        "u1_side_rows": len(side_rows), "u2_dir_rows": len(dir_rows),
        "promote_p": PROMOTE_P, "min_side": min_side, "oos_frac": oos_frac,
        "note": ("Tarama İN-SAMPLE'dır; TERFİ için zaman-bölmeli OOS onayı da şart. "
                 "t kümele-dayanıklı (sembol, gün)."),
        "surfaces": {},
    }
    for spec in SURFACES:
        pool, ret_fn, universe = pools[spec["u"]]
        res = _eval_spec(spec, pool, ret_fn, min_side=min_side, universe=universe)
        res["coverage_days"] = len({str(r.get("logged_at") or "")[:10]
                                    for r in _rows_with(spec, pool)} - {""})

        # KAYMA ARTEFAKTI KAPISI — OOS'tan ÖNCE, çünkü bir artefaktın OOS'ta da
        # doğrulanması onu kural yapmaz (kimlik her dilimde geçerlidir).
        if res["label"] == "PROMOTE" and res.get("best"):
            ctrl = _apply_rule(_rows_with(spec, pool), spec, res["best"], _bar_return,
                               min_side=min_side)
            res["slip_control"] = ctrl
            if ctrl["uplift_pp"] is None:
                res["label"] = "HOLD"
                res["text"] += (" Kayma kontrolü ÖLÇÜLEMEDİ (bar_close/fwd_price eksik) "
                                "→ terfi yok.")
            elif ctrl["uplift_pp"] <= 0:
                res["label"] = "SLIP_ARTIFACT"
                res["text"] = (f"Uplift ortak referansta ({res['best']['uplift_pp']:+.4f}pp "
                               f"→ {ctrl['uplift_pp']:+.4f}pp) İŞARET DEĞİŞTİRDİ: kural "
                               "bilgi değil GİRİŞ FİYATI OFSETİ ölçüyor (ret_live ≈ "
                               "ret_bar − slip kimliği). Terfi YOK.")

        if res["label"] == "PROMOTE" and oos_frac > 0:
            is_rows, oos_rows = _split_by_time(_rows_with(spec, pool), oos_frac)
            is_res = _eval_spec(spec, is_rows, ret_fn, min_side=min_side, universe=universe)
            rule = is_res.get("best")
            if rule is None or is_res["label"] in ("INSUFFICIENT", "INVERTED"):
                res["oos"] = {"label": "NO_RULE",
                              "text": f"IS yarısında kural seçilemedi ({is_res['label']})."}
                res["label"] = "HOLD"
                res["text"] += " OOS onayı YAPILAMADI → terfi yok."
            else:
                oos = _apply_rule(oos_rows, spec, rule, ret_fn, min_side=min_side)
                oos["is_rule"] = {"thr": rule["thr"], "keep": rule["keep"],
                                  "is_uplift_pp": rule["uplift_pp"]}
                res["oos"] = oos
                if oos["label"] != "CONFIRMED":
                    res["label"] = "HOLD"
                    res["text"] += (f" OOS onayı GEÇMEDİ ({oos['label']}, "
                                    f"uplift={oos['uplift_pp']}) → terfi yok.")
                else:
                    res["text"] += (f" OOS ONAYLI: aynı kural son %{int(oos_frac * 100)} "
                                    f"dilimde +{oos['uplift_pp']}pp (t={oos['t']}).")
        out["surfaces"][spec["key"]] = res
    out["promote"] = sorted(k for k, s in out["surfaces"].items() if s["label"] == "PROMOTE")
    return out


def _print(rep: dict) -> None:
    print("=" * 78)
    print("GÖLGE YÜZEY TERFİ ÖLÇÜMÜ (canlı log, ileriye-bakmasız)")
    print("=" * 78)
    print(f"Ham satır {rep['rows']} · çözülmüş {rep['resolved']} · "
          f"U1(taraf) {rep['u1_side_rows']} · U2(yön) {rep['u2_dir_rows']}")
    print(f"! {rep['note']}")
    print("-" * 78)
    order = {"PROMOTE": 0, "HOLD": 1, "SLIP_ARTIFACT": 2, "INVERTED": 3,
             "REJECT": 4, "INSUFFICIENT": 5}
    for _key, s in sorted(rep["surfaces"].items(),
                          key=lambda kv: order.get(kv[1]["label"], 9)):
        print(f"[{s['label']:13s}] {s['name']}  ({s['universe']}, n={s['n']}, "
              f"küme={s['n_clusters']}, gün={s.get('coverage_days', 0)})")
        b = s.get("best")
        if b:
            print(f"               kapı: {b['keep']} thr={b['thr']} -> "
                  f"tutulan n={b['kept']['n']} ort={b['kept']['avg_ret_pct']}% · "
                  f"elenen n={b['cut']['n']} ort={b['cut']['avg_ret_pct']}% · "
                  f"uplift={b['uplift_pp']}pp t={b['t']} p_adj={b.get('p_adj')}")
        sc = s.get("slip_control")
        if sc:
            print(f"               kayma-kontrol: ortak referansta uplift="
                  f"{sc['uplift_pp']}pp (t={sc['t']})")
        o = s.get("oos")
        if o:
            print(f"               OOS[{o['label']}] " + (
                f"kural(IS) {o['is_rule']['keep']}={o['is_rule']['thr']} -> "
                f"uplift={o['uplift_pp']}pp t={o['t']} (n={o['n']})"
                if "is_rule" in o else o.get("text", "")))
        print(f"               {s['text']}")
    print("-" * 78)
    promo = [rep["surfaces"][k]["name"] for k in rep.get("promote", [])]
    print("TERFİ EDİLEBİLİR: " + (", ".join(promo) if promo else "YOK — hepsi gölgede kalır"))


def main() -> None:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Gölge yüzeylerin terfi ölçümü")
    ap.add_argument("--min-side", type=int, default=MIN_SIDE_DEFAULT)
    ap.add_argument("--json", default="", help="sonucu bu dosyaya da yaz")
    args = ap.parse_args()
    rep = run(min_side=args.min_side)
    _print(rep)
    if args.json:
        from nihai.atomic_io import atomic_write_text
        atomic_write_text(Path(args.json), json.dumps(rep, ensure_ascii=False, indent=2))
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
