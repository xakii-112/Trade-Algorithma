"""
selfcheck.py — ÜÇ BÖLÜMLÜ ÖZ-DEĞERLENDİRME SINAVI (Tur-16 WP-A). KARAR-DIŞI.

AI'ın "kendi bozulmasını KENDİ raporlaması" için periyodik benchmark (haftalık koşum
düşünülür; WP-O'nun zamanlanmış görevine bağlanması bu dosyanın KAPSAMI DIŞINDadır):

  (a) `exam_knowledge`     — knowledge.py'deki kavramlardan DETERMİNİSTİK ~n soru üretir:
      "formül köprüsü çözünürlüğü" (c.formul_ref'teki her ad formül modüllerinden birinde GERÇEKTEN
      var mı — knowledge.py modül docstring'inin "uydurma yok" sözleşmesinin sınav hali) +
      "prereq DAG tutarlılığı" (c.prereqs CONCEPTS'te var mı VE knowledge.curriculum()'ın
      ürettiği topolojik sırada bu kavramdan ÖNCE mi geliyor). Aynı seed → aynı örneklem →
      aynı skor (random.Random(seed); CONCEPTS sabit tablo).
  (b) `exam_calibration`   — learning.load_state()'teki (son çözülmüş gölge kararlardan
      türetilmiş) Beta-posterior/Brier/drift özetini okur. Veri yoksa 'VERI_YOK' (uydurma
      skor YOK). NOT (bilinçli sapma): bu fonksiyon learning'in ZATEN aggregate ettiği
      durumu okur — kendi tail-N penceresi YENİDEN hesaplamaz (learning.update_from_predictions
      yan-etkili [state dosyasına yazar] ve tail-pencere semantiği zaten oradadır; burada
      tekrarlamak sızıntı/tutarsızlık riski taşır). `limit` parametresi ileri-uyumluluk için
      korunur ama bugün aggregate okumaya bir kırpma UYGULAMAZ.
  (c) `exam_replay`        — bilinen-rejim SENTETİK seriler (trend-up/trend-down/yüksek-vol;
      quant_math.mc_gbm_paths ile üretilir — GERÇEK veri YOK, sızıntı riski yok) üstünde
      brain.think() çalıştırıp ABSTAIN-oranı + rejim-tespit doğruluğunu ölçer. AI_DECISION_MODE
      geçici olarak 'off' zorlanır (brain._demo deseni) — hiçbir dosyaya yazılmaz.

`report_card()` üçünü birleştirir, `data/ai_selfcheck.jsonl`'e EKLER (append-only; memory.py/
episodic.py'nin basit-append deseni — düşük frekanslı, haftalık koşum, tam-yeniden-yazım
[consolidate.py deseni] burada gereksiz), önceki karneyle TREND satırı üretir. Çıktı metni
reasoner.CLOSING ile biter ve 'kenar' İDDİASI (pozitif) içermez — yalnız negatif/İZLEME
formunda geçebilir (ör. "kenar kanıtı değildir"), tüm codebase'deki CLOSING sözleşmesiyle
aynı disiplin (ACI DERS #3/#4).
"""
from __future__ import annotations

import json
import math
import os
import random
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from loguru import logger
from scipy.special import gammaln

from nihai.ai import (
    knowledge,
    learning,
    microstructure,
    online,
    options_math,
    quant_math,
    reasoner,
    regime_hmm,
)
from nihai.config import DATA_DIR
from nihai.console import ensure_utf8_stdout

# Modül-düzeyi yol: testler monkeypatch'ler (memory.py/episodic.py deseni).
SELFCHECK_FILE = DATA_DIR / "ai_selfcheck.jsonl"

# Formül köprüsünün ARAYACAĞI modüller (Tur-32; Tur-34'te regime_hmm, Tur-35'te online +
# learning eklendi). knowledge.formul_ref adları bu sırayla çözülür; tek kaynak burasıdır
# (test dosyası da aynı demeti kullanır — bir modül eklenirse İKİ yerde değil, burada değişir).
#
# `neural` BİLEREK DIŞARIDADIR: köprüye eklemek her kavram sorgusunda torch importunu
# tetiklerdi (Tur-26 "`import nihai.ai` UCUZ kalmalı" disiplini — ölçüldü: 3854 ms ↔ 11 ms).
# Bu yüzden sinir-ağı tarafındaki kavramlar (`maskeli_on_egitim`, `quantile_regresyon`)
# `formul_ref=[]` taşır ve tanımlarında hangi modülde yaşadıklarını METİNLE söyler.
_FORMULA_MODULES = (quant_math, options_math, microstructure, regime_hmm, online, learning)


def _formula_target(fn_name: str) -> Any:
    """formul_ref adı → çağrılabilir fonksiyon (hiçbir formül modülünde yoksa None)."""
    for mod in _FORMULA_MODULES:
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            return fn
    return None


_LOCK = threading.Lock()

# exam_replay sentetik senaryoları — Türkçe adlar, mc_gbm_paths parametreleri (yıllıklaştırılmış
# mu/sigma; dt=1/252 quant_math konvansiyonuyla tutarlı). yuksek_vol'da trend BEKLENMEZ (None).
_SCENARIOS: dict[str, dict[str, float]] = {
    "trend_up": {"mu": 0.35, "sigma": 0.15},
    "trend_down": {"mu": -0.35, "sigma": 0.15},
    "yuksek_vol": {"mu": 0.0, "sigma": 0.55},
}
_EXPECTED_TREND: dict[str, str | None] = {"trend_up": "up", "trend_down": "down", "yuksek_vol": None}
_REPLAY_DAYS = 252

# REJİM-GEÇİŞLİ senaryolar (denetim önerisi 10, 2026-07-26): tek-rejimli GBM serilerinde
# replay sınavı 1.00'a yapışıyordu — bozulmayı erken yakalayamayan "kolay" bir sınav.
# Geçiş serisi, dedektörün GÜNCEL rejimi mi yoksa dönem ORTALAMASINI mı raporladığını ayırt
# eder: yükseliş sonrası çöküşte doğru cevap "down"dur. Segmentler kesintisiz eklenir
# (ikinci parça birincinin SON değerinden başlar) — yapay sıçrama rejimi sahte-tetiklemesin.
_TRANSITIONS: dict[str, list[dict[str, float]]] = {
    # yükselen trend → sert çöküş (klasik rejim kırılması)
    "gecis_yukselis_cokus": [{"mu": 0.35, "sigma": 0.15, "days": 126},
                             {"mu": -0.90, "sigma": 0.45, "days": 126}],
    # çöküş → toparlanma (SİMETRİK kapı: dedektör volatil her seriye "down" diyemez)
    "gecis_cokus_toparlanma": [{"mu": -0.60, "sigma": 0.35, "days": 126},
                               {"mu": 0.60, "sigma": 0.20, "days": 126}],
}
_EXPECTED_TREND.update({"gecis_yukselis_cokus": "down", "gecis_cokus_toparlanma": "up"})


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# (a) Bilgi sınavı
# ---------------------------------------------------------------------------
def _sayisal_sorular() -> list[tuple[str, Any, float]]:
    """(etiket, hesaplanan_değer, beklenen_değer) — KAPALI FORMLA doğrulanabilir kimlikler.

    Denetim önerisi 10 (2026-07-26): bilgi sınavı yalnız İÇ TUTARLILIK ölçüyordu (formül
    adı quant_math'te var mı, prereq DAG'ı topolojik mi) ve bu yüzden 1.00'a yapışıktı —
    bir formülün MATEMATİĞİ bozulsa skor değişmezdi. Bu sorular bilinen girdi → beklenen
    çıktı eşleşmesini sınar, yani gerçek bozulmayı yakalar.

    Seçim ilkesi: yalnız konvansiyondan BAĞIMSIZ kimlikler (kapalı form ya da denklik).
    Örn. put-call paritesi `C − P = S − K·e^(−rT)`, hangi Black-Scholes parametrizasyonu
    kullanılırsa kullanılsın geçerlidir; bir çeyrek/yıllıklaştırma konvansiyonu tartışması
    doğurmaz. Fonksiyon None dönerse soru YANLIŞ sayılır (sessiz bozulma da bozulmadır).

    Hata fırlatan bir fonksiyon sınavı ÇÖKERTMEZ; ilgili soru yanlış sayılır (savunmacı
    sözleşme — selfcheck her koşulda bir karne üretmelidir)."""
    q = quant_math
    sorular: list[tuple[str, Any, float]] = []

    def _ekle(etiket: str, cagri: Any, beklenen: float) -> None:
        try:
            sorular.append((etiket, cagri(), beklenen))
        except Exception as e:  # noqa: BLE001 — sınav çökmemeli, soru yanlış sayılır
            logger.debug(f"[ai.selfcheck] sayısal soru hatası ({etiket}): {e}")
            sorular.append((etiket, None, beklenen))

    # Kelly ailesi — kapalı form
    _ekle("kelly_binary", lambda: q.kelly_binary(0.6, 1.0), 0.2)               # p−(1−p)/b
    _ekle("kelly_continuous", lambda: q.kelly_continuous(0.01, 0.04), 0.25)    # μ/σ²
    _ekle("fractional_kelly_cap", lambda: q.fractional_kelly(0.2, 0.25, 0.02), 0.02)  # cap bağlar
    # Yıllıklaştırma — √t ve bileşik büyüme
    _ekle("annualize_vol", lambda: q.annualize_vol(0.01, 252), 0.01 * math.sqrt(252))
    _ekle("annualize_return", lambda: q.annualize_return(0.001, 252), 1.001 ** 252 - 1.0)
    # Getiri dönüşümleri — birebir bilinen çıktı
    _ekle("simple_returns", lambda: float(q.simple_returns([100.0, 110.0, 121.0])[1]), 0.1)
    _ekle("log_returns", lambda: float(q.log_returns([1.0, math.e])[0]), 1.0)
    # Performans/risk — kapalı form
    _ekle("profit_factor", lambda: q.profit_factor([2.0, -1.0, 3.0, -1.0]), 2.5)   # 5/2
    def _mdd() -> float | None:
        # max_drawdown -> (dd, indeks) | None; None dönerse soru YANLIŞ sayılır (indekslemeye
        # kalkışmak sınavı çökertirdi — savunmacı sözleşme).
        r = q.max_drawdown([100.0, 120.0, 90.0, 150.0])
        return None if r is None else r[0]

    _ekle("max_drawdown", _mdd, 0.25)
    _ekle("omega_simetrik", lambda: q.omega([1.0, -1.0, 2.0, -2.0], 0.0), 1.0)
    _ekle("downside_deviation",
          lambda: q.downside_deviation([0.01, -0.02, 0.03, -0.04], 0.0),
          math.sqrt((0.02 ** 2 + 0.04 ** 2) / 4))
    # Put-call paritesi: C − P = S − K·e^(−rT) (parametrizasyondan bağımsız DENKLİK)
    def _parite() -> float | None:
        c = q.bs_price(100.0, 100.0, 1.0, 0.05, 0.2, "c")
        p = q.bs_price(100.0, 100.0, 1.0, 0.05, 0.2, "p")
        return None if (c is None or p is None) else c - p

    _ekle("put_call_paritesi", _parite, 100.0 - 100.0 * math.exp(-0.05))

    # --- Tur-32: yeni formül modülleri de sessiz bozulmaya karşı sınanır ---
    # Aynı ilke: yalnız konvansiyondan BAĞIMSIZ kimlikler.
    # (1) IV gidiş-dönüş: bs_price(σ) → implied_vol → σ (tanım gereği ters fonksiyonlar).
    def _iv_gidis_donus() -> float | None:
        px = q.bs_price(100.0, 105.0, 0.5, 0.02, 0.30, "c")
        return None if px is None else options_math.implied_vol(px, 100.0, 105.0, 0.5, 0.02, "c")

    _ekle("implied_vol_gidis_donus", _iv_gidis_donus, 0.30)
    # (2) SABR ATM limiti: f=k'da σ ≈ α/f^(1−β)·(1+corr·t); ν=0 & ρ=0 & β=1'de TAM α'dır.
    _ekle("sabr_atm_beta1",
          lambda: options_math.sabr_lognormal_vol(100.0, 100.0, 1.0, 0.25, 1.0, 0.0, 0.0), 0.25)
    # (3) SVI minimum: k=m'de w = a + b·σ (karekök terimi tam σ'ya eşittir).
    def _svi_minimum() -> float | None:
        w = options_math.svi_total_variance(
            [0.03], {"a": 0.04, "b": 0.4, "rho": -0.4, "m": 0.03, "sigma": 0.18})
        return None if w is None else float(w[0])

    _ekle("svi_minimum_noktasi", _svi_minimum, 0.04 + 0.4 * 0.18)
    # (4) Gerçekleşen varyans: Σr² (tanım).
    _ekle("realized_variance", lambda: microstructure.realized_variance([0.1, -0.2, 0.3]), 0.14)
    # (5) Roll makası: tam alternatif ±c sıçramasında Δp = ∓2c ve ardışık çarpımların
    #     toplamı 38·(−4c²)'dir; ÖRNEKLEM kovaryansı (ddof=1) 37'ye bölündüğü için
    #     S = 2·√(4c²·38/37). Beklenen değerin ddof çarpanını AÇIKÇA taşıması bilinçli:
    #     tahminci sessizce ddof=0'a kayarsa bu soru YANLIŞ olur (yakalanması istenen şey).
    def _roll_kapali_form() -> float | None:
        seri = [100.0 + (0.05 if i % 2 else -0.05) for i in range(40)]
        out = microstructure.roll_spread(seri)
        return None if out is None else out["spread"]

    _ekle("roll_spread_kapali_form", _roll_kapali_form, 2.0 * math.sqrt(0.01 * 38.0 / 37.0))

    # --- Tur-34: v6 modülleri (kopula / HMM / SSVI / PIN) ---
    # Aynı ilke sürüyor: yalnız ANALİTİK KİMLİKLER sorulur, uyum kalitesi değil.
    # (6) Clayton alt-kuyruk katsayısı: λ_L = 2^(−1/θ); θ=2 → 2^(−0.5).
    def _clayton_lambda() -> float | None:
        out = q.copula_tail_dependence("clayton", {"theta": 2.0})
        return None if out is None else out["lower"]

    _ekle("clayton_alt_kuyruk", _clayton_lambda, 2.0 ** -0.5)
    # (7) Gumbel üst-kuyruk katsayısı: λ_U = 2 − 2^(1/θ); θ=2 → 2 − √2.
    def _gumbel_lambda() -> float | None:
        out = q.copula_tail_dependence("gumbel", {"theta": 2.0})
        return None if out is None else out["upper"]

    _ekle("gumbel_ust_kuyruk", _gumbel_lambda, 2.0 - math.sqrt(2.0))
    # (8) SSVI ATM kimliği: w(k=0, θ) = θ (φ ve ρ'dan BAĞIMSIZ — parametrizasyonun tanımı).
    def _ssvi_atm() -> float | None:
        w = options_math.ssvi_total_variance([0.0], 0.04,
                                             {"rho": -0.4, "eta": 0.9, "gamma": 0.35})
        return None if w is None else float(w[0])

    _ekle("ssvi_atm_theta", _ssvi_atm, 0.04)
    # (9) HMM beklenen süre: P(i→i)=0.9 → geometrik bekleme ortalaması 1/(1−0.9) = 10.
    def _hmm_sure() -> float | None:
        out = regime_hmm.hmm_transition_summary(
            {"means": [0.0, 1.0], "stds": [1.0, 1.0], "start": [0.5, 0.5],
             "transition": [[0.9, 0.1], [0.2, 0.8]]})
        return None if out is None else out["expected_duration"][0]

    _ekle("hmm_beklenen_sure", _hmm_sure, 10.0)
    # (10) HMM durağan dağılım: [[0.9,0.1],[0.2,0.8]] için π = (2/3, 1/3) (π·P = π çözümü).
    def _hmm_duragan() -> float | None:
        out = regime_hmm.hmm_transition_summary(
            {"means": [0.0, 1.0], "stds": [1.0, 1.0], "start": [0.5, 0.5],
             "transition": [[0.9, 0.1], [0.2, 0.8]]})
        return None if out is None or out["stationary"] is None else out["stationary"][0]

    _ekle("hmm_duragan_dagilim", _hmm_duragan, 2.0 / 3.0)
    # (11) EKOP olabilirliğinin μ=0 LİMİTİ: bilgili akış yokken üç dal ÇAKIŞIR ve olabilirlik
    #      iki BAĞIMSIZ Poisson'a indirgenir — α/δ ne olursa olsun. Lin-Ke çarpanlaması
    #      bozulursa (bracket yanlış kurulursa) bu kimlik KIRILIR; ölçtüğü şey tam olarak
    #      sayısal-kararlılık için yapılan yeniden-düzenlemenin doğruluğudur.
    _b_soru = np.array([2.0, 3.0, 5.0])
    _s_soru = np.array([1.0, 4.0, 2.0])
    _eb_soru, _es_soru = 2.5, 2.0
    _pin_const = float(-np.sum(gammaln(_b_soru + 1.0) + gammaln(_s_soru + 1.0)))
    _pin_poisson = float(
        np.sum(-_eb_soru + _b_soru * math.log(_eb_soru) - gammaln(_b_soru + 1.0))
        + np.sum(-_es_soru + _s_soru * math.log(_es_soru) - gammaln(_s_soru + 1.0)))

    def _pin_mu_sifir_limiti() -> float | None:
        return -microstructure._pin_neg_loglik(
            np.array([0.5, 0.5, 0.0, _eb_soru, _es_soru]), _b_soru, _s_soru, _pin_const)

    _ekle("pin_mu_sifir_poisson_limiti", _pin_mu_sifir_limiti, _pin_poisson)
    return sorular


def exam_knowledge(n: int = 50, seed: int = 42) -> dict[str, Any]:
    """knowledge.CONCEPTS'ten deterministik ~n soru (formül köprüsü + prereq DAG) + sabit
    bir SAYISAL formül doğrulama seti; skor = doğru/toplam. `n<=0` veya CONCEPTS boşsa
    yapısal kısım ölçülemez (uydurma yok) ama sayısal sorular yine de koşar.

    Sayısal sorular karneye ayrıca `sayisal` altında dökülür (hangi formülün bozulduğu
    tek bakışta görünsün — toplam skorun içinde erimesin)."""
    # SAYISAL bölüm yapısal bölümden bağımsızdır: CONCEPTS boş olsa bile matematiğin
    # doğruluğu ölçülebilir (ve ölçülmelidir — sessiz bozulma da bozulmadır).
    sayisal = _sayisal_sorular()
    s_correct = 0
    s_failed: list[str] = []
    for etiket, got, beklenen in sayisal:
        ok = (got is not None and isinstance(got, (int, float))
              and math.isfinite(float(got))
              and abs(float(got) - beklenen) <= 1e-9 + 1e-6 * abs(beklenen))
        if ok:
            s_correct += 1
        else:
            s_failed.append(f"{etiket}(={got!r} bekl {beklenen:.6g})")
    sayisal_blok = {"correct": s_correct, "total": len(sayisal),
                    "score": round(s_correct / len(sayisal), 4) if sayisal else None,
                    "failed": s_failed}

    ids = knowledge.glossary()
    if not ids or n <= 0:
        # Sınav DEVRE DIŞI (n<=0) veya CONCEPTS boş → `score=None` ("ölçülemez, uydurma yok"
        # — modülün özgün sözleşmesi korunur). Sayısal blok yine de RAPORLANIR: bilgi
        # kaybedilmez, ama devre dışı bir sınavdan sahte bir toplam skor üretilmez.
        return {"n": 0, "seed": seed, "score": None, "correct": 0, "total": 0,
                "failed": [], "dag_tutarli": None, "sayisal": sayisal_blok}
    rng = random.Random(seed)
    sample = [ids[rng.randrange(len(ids))] for _ in range(int(n))]

    try:
        order = knowledge.curriculum()
        dag_tutarli = True
    except ValueError:
        order = []
        dag_tutarli = False
    order_index = {cid: i for i, cid in enumerate(order)}

    correct = 0
    total = 0
    failed: list[str] = []
    for cid in sample:
        c = knowledge.CONCEPTS.get(cid)
        if c is None:  # pragma: no cover - glossary() zaten CONCEPTS'ten türetilir
            continue
        # Soru 1: formül köprüsü — her formul_ref adı FORMÜL MODÜLLERİNDEN birinde GERÇEKTEN
        # var mı (Tur-32: köprü quant_math + options_math + microstructure'a genişledi;
        # tek modüle bakmak, uygulaması taşınan/yeni yazılan kavramı yanlışlıkla "kırık
        # köprü" sayardı — sınav matematiği değil, ADRESİ ölçmüş olurdu).
        total += 1
        bridge_ok = all(_formula_target(fn) is not None for fn in c.formul_ref)
        if bridge_ok:
            correct += 1
        else:
            failed.append(f"{cid}:formul")
        # Soru 2: prereq DAG tutarlılığı — prereq'ler CONCEPTS'te var mı VE curriculum
        # sırasında bu kavramdan ÖNCE mi geliyor (topolojik geçerlilik).
        total += 1
        prereqs_known = all(p in knowledge.CONCEPTS for p in c.prereqs)
        prereqs_ordered = dag_tutarli and all(
            order_index.get(p, -1) < order_index.get(cid, len(order)) for p in c.prereqs
        )
        if prereqs_known and (not c.prereqs or prereqs_ordered):
            correct += 1
        else:
            failed.append(f"{cid}:prereq")

    correct += s_correct
    total += len(sayisal)
    failed.extend(s_failed)
    score = round(correct / total, 4) if total else None
    return {"n": len(sample), "seed": seed, "score": score, "correct": correct,
            "total": total, "failed": failed, "dag_tutarli": dag_tutarli,
            "sayisal": sayisal_blok}


# ---------------------------------------------------------------------------
# (b) Kalibrasyon sınavı
# ---------------------------------------------------------------------------
def exam_calibration(limit: int = 2000) -> dict[str, Any]:
    """learning.load_state()'ten Brier/isabet TRENDİNİ oku (modül docstring'i: aggregate
    okuma, yeniden-hesap YOK). Hiç çözülmüş kayıt yoksa `durum='VERI_YOK'`."""
    state = learning.load_state()
    strategies = state.get("strategies") or {}
    pwin_cal = state.get("pwin_calibration")
    if not strategies and not pwin_cal:
        return {"durum": "VERI_YOK", "limit": int(limit),
                "detail": "öğrenme durumu boş — henüz çözülmüş gölge karar yok"}
    global_s = strategies.get("GLOBAL") or {}
    drift = state.get("drift")
    trend = "bilinmiyor"
    if isinstance(drift, dict):
        trend = "BOZULUYOR (drift tespit edildi)" if drift.get("drift") else "STABIL"
    return {
        "durum": "OLCULDU",
        "limit": int(limit),
        "n_karar": global_s.get("n"),
        "isabet_posterior": global_s.get("post_mean"),
        "isabet_ci95": ([global_s.get("lo95"), global_s.get("hi95")] if global_s else None),
        "pwin_brier": (pwin_cal or {}).get("brier"),
        # v4 (Tur-22): ECE, Brier'in ayrıştıramadığı ekseni gösterir — model ayırt
        # edemiyor olsa bile (AUC~0,50) olasılıkları dürüst mü? Karnede ikisi YAN YANA
        # durmalı; tek başına Brier "kalibrasyon iyi mi kötü mü" sorusuna cevap vermez.
        "pwin_ece": (pwin_cal or {}).get("ece"),
        "pwin_n": (pwin_cal or {}).get("n"),
        # Kaynak karnesi (prior/pwin/neural): hangi olasılık kaynağının daha dürüst
        # olduğu ancak kaynaklar KARŞILAŞTIRILARAK söylenebilir.
        "kaynaklar": state.get("sources"),
        "trend": trend,
        "retrain_candidate": bool(state.get("retrain_candidate")),
    }


# ---------------------------------------------------------------------------
# (c) Replay sınavı
# ---------------------------------------------------------------------------
def _synthetic_panel(seed: int = 42) -> dict[str, np.ndarray]:
    """Bilinen-rejim sentetik GBM serileri (gerçek veri YOK — sızıntı riski yok)."""
    panel: dict[str, np.ndarray] = {}
    for i, name in enumerate(sorted(_SCENARIOS)):
        params = _SCENARIOS[name]
        paths = quant_math.mc_gbm_paths(s0=100.0, mu=params["mu"], sigma=params["sigma"],
                                        days=_REPLAY_DAYS, n_paths=1, seed=seed + i)
        panel[name] = paths[0]
    # Rejim-GEÇİŞLİ seriler: segmentler kesintisiz birleştirilir (her parça bir öncekinin
    # SON fiyatından başlar, ilk örnek tekrar edilmez) — birleşim noktasında yapay gap yok.
    for j, name in enumerate(sorted(_TRANSITIONS)):
        s0 = 100.0
        parcalar: list[np.ndarray] = []
        for k, seg in enumerate(_TRANSITIONS[name]):
            path = quant_math.mc_gbm_paths(s0=s0, mu=seg["mu"], sigma=seg["sigma"],
                                           days=int(seg["days"]), n_paths=1,
                                           seed=seed + 100 + 10 * j + k)[0]
            parcalar.append(path if k == 0 else path[1:])
            s0 = float(path[-1])
        panel[name] = np.concatenate(parcalar) if parcalar else np.asarray([], dtype=float)
    return panel


def exam_replay(closes_panel: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Sentetik bilinen-rejim serilerinde brain.think() ABSTAIN-oranı + rejim-tespit
    doğruluğu. `closes_panel` verilmezse `_synthetic_panel()` üretir (deterministik,
    seed=42). Yazma YOK: AI_DECISION_MODE geçici 'off' zorlanır (brain._demo deseni)."""
    from nihai.ai import brain  # lazy — nihai.ai.__init__ lazy-yükleme deseniyle tutarlı

    panel = closes_panel if closes_panel is not None else _synthetic_panel()
    if not panel:
        return {"n_senaryo": 0, "abstain_orani": None, "rejim_dogruluk": None, "detay": []}

    prev_mode = os.environ.get("AI_DECISION_MODE")
    os.environ["AI_DECISION_MODE"] = "off"
    detay: list[dict[str, Any]] = []
    try:
        for name in sorted(panel):
            closes = panel[name]
            arr = np.asarray(closes, dtype=float).reshape(-1)
            try:
                regime = quant_math.classify_regime(arr) if arr.size else None
            except Exception as e:  # noqa: BLE001 — sınav çökmemeli
                logger.debug(f"[ai.selfcheck] rejim hesabı hatası ({name}): {e}")
                regime = None
            try:
                t = brain.think(symbol="SELFCHECK", strategy="selfcheck_replay",
                                features={"ret_5": 0.0, "rsi": 50.0}, closes=arr, p_win=0.5)
                action = t.decision.action if t.decision is not None else None
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[ai.selfcheck] brain.think hatası ({name}): {e}")
                action = None
            expected = _EXPECTED_TREND.get(name)
            got = (regime or {}).get("trend")
            trend_dogru = (got == expected) if expected is not None else None
            detay.append({"senaryo": name, "trend_tespit": got, "trend_dogru": trend_dogru,
                          "action": action, "abstain": action == "ABSTAIN"})
    finally:
        if prev_mode is None:
            os.environ.pop("AI_DECISION_MODE", None)
        else:
            os.environ["AI_DECISION_MODE"] = prev_mode

    n = len(detay)
    abstain_n = sum(1 for d in detay if d["abstain"])
    checked = [d for d in detay if d["trend_dogru"] is not None]
    rejim_dogruluk = (round(sum(1 for d in checked if d["trend_dogru"]) / len(checked), 4)
                      if checked else None)
    return {"n_senaryo": n, "abstain_orani": (round(abstain_n / n, 4) if n else None),
            "rejim_dogruluk": rejim_dogruluk, "detay": detay}


# ---------------------------------------------------------------------------
# Kalıcılık (append-only JSONL — memory.py/episodic.py deseni) + karne birleştirme
# ---------------------------------------------------------------------------
def _append(rec: dict) -> None:
    with _LOCK:
        SELFCHECK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with SELFCHECK_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_all() -> list[dict]:
    """Tüm geçerli karneler (bozuk satır atlanır, sayısı loglanır — memory.py deseni)."""
    if not SELFCHECK_FILE.exists():
        return []
    out: list[dict] = []
    corrupt = 0
    try:
        with SELFCHECK_FILE.open(encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except ValueError:
                    corrupt += 1
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
                else:
                    corrupt += 1
    except OSError as e:
        logger.warning(f"[ai.selfcheck] okunamadı ({SELFCHECK_FILE.name}): {e}")
        return out
    if corrupt:
        logger.warning(f"[ai.selfcheck] {corrupt} bozuk satır atlandı ({SELFCHECK_FILE.name})")
    return out


def last_card_line() -> str | None:
    """SON karnenin tek-satır özeti (Tur-17 raporlama köprüsü) — SALT-OKUNUR: yeni sınav
    KOŞULMAZ, `SELFCHECK_FILE`'a HİÇBİR ŞEY YAZILMAZ. Dosya yok/boş/tamamen bozuk → None
    (çağıran satırı hiç eklemez). Bu fonksiyon `report_card()`'ın PAHALI olmasının doğal
    sonucudur: günlük brif/özet gibi salt-okunur raporlayıcılar üç sınavı yeniden koşmadan
    en son ölçümü göstermek ister (bkz. brain.daily_brief, daily_summary._brain_block)."""
    records = _read_all()
    if not records:
        return None
    last = records[-1]
    bits = [f"ts={last.get('ts', '?')}"]
    know = last.get("knowledge") or {}
    if isinstance(know, dict) and isinstance(know.get("score"), (int, float)):
        bits.append(f"bilgi {float(know['score']):.2f}")
    cal = last.get("calibration") or {}
    if isinstance(cal, dict) and isinstance(cal.get("isabet_posterior"), (int, float)):
        bits.append(f"isabet {float(cal['isabet_posterior']):.3f}"
                    + (f" ({cal['trend']})" if cal.get("trend") else ""))
    rep = last.get("replay") or {}
    if isinstance(rep, dict) and isinstance(rep.get("rejim_dogruluk"), (int, float)):
        bits.append(f"replay-rejim {float(rep['rejim_dogruluk']):.2f}")
    return "[Karne] " + " · ".join(bits)


def _trend_text(rec: dict, prev: dict | None) -> str:
    """Önceki karneye göre TREND satırı (geriye dönük karşılaştırma; kenar iması YOK)."""
    if prev is None:
        return "Trend: karşılaştırılacak önceki karne yok (ilk koşu)."
    bits: list[str] = []
    ks, kp = rec["knowledge"].get("score"), (prev.get("knowledge") or {}).get("score")
    if ks is not None and kp is not None:
        bits.append(f"bilgi-sınavı {kp:.3f}→{ks:.3f} ({ks - kp:+.3f})")
    rs, rp = rec["replay"].get("rejim_dogruluk"), (prev.get("replay") or {}).get("rejim_dogruluk")
    if rs is not None and rp is not None:
        bits.append(f"replay-rejim-doğruluk {rp:.3f}→{rs:.3f} ({rs - rp:+.3f})")
    cal_now = rec["calibration"].get("durum")
    cal_prev = (prev.get("calibration") or {}).get("durum")
    if cal_now and cal_prev:
        bits.append(f"kalibrasyon durumu {cal_prev}→{cal_now}")
    if not bits:
        return "Trend: önceki karneyle karşılaştırılabilir alan yok."
    return "Trend: " + " · ".join(bits)


PRODUCTION_KPI_FILE = "uretim_kpi.json"
_PROD_MAX_AGE_DAYS = 14


def exam_production(max_age_days: int = _PROD_MAX_AGE_DAYS, gun: int = 30) -> dict[str, Any]:
    """O3 — ÜRETİM sınavı: "hiçbir şey üretmemenin" bir maliyeti olsun.

    NEDEN (2026-07-26 denetimi B-1/B-11): karar motoru aylarca %100 ABSTAIN üretti ve hiçbir
    metrik bunu cezalandırmadı; iki yapısal kilit fark edilmeden durdu. ABSTAIN sonsuza kadar
    "güvenli" görünür — ölçülmezse bozukluk sessizdir.

    İki bacak:
      * **canlı** — `data/ai_decisions.jsonl`'deki son `gun` günün gerçek aksiyon dağılımı
        (ucuz, her karnede koşar). `n_yonlu = 0` → AÇIK UYARI.
      * **ölçüm** — `data/uretim_kpi.json` artefaktı (fazlalık vs triviyal taban). Bunu
        `research.shadow_decision_backtest` koşusu yazar; artefakt yoksa/bayatsa sınav
        'OLCULMEDI' der ve **uydurmaz**.
    Skor YOK — bu bir tespit sınavıdır; kenar iddiası içermez."""
    from nihai.ai.decision import DECISIONS_FILE
    out: dict[str, Any] = {"gun": int(gun), "durum": "VERI_YOK", "uyarilar": []}

    # --- canlı bacak ---
    sinir = datetime.now(UTC) - timedelta(days=int(gun))
    sayac: dict[str, int] = {}
    try:
        with open(DECISIONS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ts = datetime.fromisoformat(str(rec.get("ts", "")))
                except (ValueError, TypeError):
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if ts < sinir:
                    continue
                act = str(rec.get("action", "?"))
                sayac[act] = sayac.get(act, 0) + 1
    except OSError:
        pass
    toplam = sum(sayac.values())
    if toplam:
        yonlu = sum(v for k, v in sayac.items() if k in ("LONG_BIAS", "SHORT_BIAS"))
        out.update({
            "durum": "OLCULDU", "n_karar": toplam, "n_yonlu": yonlu,
            "abstain_orani": round(sayac.get("ABSTAIN", 0) / toplam, 4),
            "aksiyon_dagilimi": sayac,
        })
        if yonlu == 0:
            out["uyarilar"].append(
                f"karar motoru {gun} gündür HİÇBİR yön üretmiyor (n={toplam}, %100 ABSTAIN) — "
                "kapılar ölçülmemiş/yapısal kilitli olabilir (bkz. B-1/B-11, O1 λ süpürmesi)")

    # --- ölçüm bacağı (artefakt) ---
    art = DATA_DIR / PRODUCTION_KPI_FILE
    try:
        kpi = json.loads(art.read_text(encoding="utf-8"))
        yas = (datetime.now(UTC) - datetime.fromisoformat(str(kpi["ts"]))).days
        par = kpi.get("parametre") or {}
        out["fazlalik"] = {
            "vs_hep_long": kpi.get("FAZLALIK_vs_hep_long"),
            "vs_momentum": kpi.get("FAZLALIK_vs_momentum"),
            "n_yonlu": kpi.get("n_yonlu"), "yas_gun": yas,
            "en_iyi_konusan_lambda": kpi.get("en_iyi_konusan_lambda"),
            # Tur-86: hüküm kodu + ölçüm KOŞULU (Tur-86 öncesi artefaktta yok → None, uydurulmaz).
            "verdikt_kod": kpi.get("verdikt_kod"),
            "olcum_kosulu": ({k: par.get(k) for k in ("gun", "horizon", "exec_lag", "n_baglam")}
                             if par else None),
        }
        eik = kpi.get("en_iyi_konusan_lambda") or {}
        if kpi.get("verdikt_kod") == "KONUSAN_LAMBDA_YOK":
            # BULGU-86-2: boş kümeden "ABSTAIN doğru" hükmü kurulamaz — kilidin yeri λ değil.
            out["uyarilar"].append(
                "λ süpürmesinde HİÇBİR λ (0 dahil) yön üretmedi → üretim fazlalığı ÖLÇÜLEMEZ; "
                "ABSTAIN'i λ değil prior güven kapısı (kapı-a: yön için CI tabanı > 0,5) "
                "belirliyor → 'ABSTAIN ölçülmüş doğru davranış' bu artefakttan KURULAMAZ")
        elif (kpi.get("n_yonlu") or 0) == 0 and eik:
            fz = eik.get("FAZLALIK_vs_hep_long") or 0.0
            # Eşik: 0,1 puanın altındaki "pozitif" fazlalık gürültüdür (ölçümde t≈0,03) —
            # sıfıra yakın bir sayıyı "kapıyı aç" alarmına çevirmek ACI DERS #3 ihlali olur.
            out["uyarilar"].append(
                f"üretim λ'sında hiç yön üretilmedi; kapı gevşetilse (λ={eik.get('lambda')}) "
                f"fazlalık {fz} puan olurdu → " + ("ABSTAIN'i sorgula (fazlalık pozitif)"
                                                  if fz > 0.1 else
                                                  "ABSTAIN ölçülmüş doğru davranış"))
        if yas > max_age_days:
            out["uyarilar"].append(
                f"üretim fazlalığı ölçümü {yas} gün eski (>{max_age_days}) — "
                "`python -m research.shadow_decision_backtest --lambda-sweep` ile tazele")
        elif (kpi.get("FAZLALIK_vs_hep_long") or 0) < 0:
            out["uyarilar"].append(
                f"motor triviyal 'hep LONG' tabanının ALTINDA "
                f"({kpi.get('FAZLALIK_vs_hep_long')} puan)")
    except (OSError, ValueError, KeyError, TypeError):
        out["fazlalik"] = {"durum": "OLCULMEDI"}
        out["uyarilar"].append(
            "üretim fazlalığı hiç ölçülmemiş (data/uretim_kpi.json yok) — ABSTAIN'in doğru mu "
            "yoksa bozuk mu olduğu BİLİNMİYOR")
    return out


def report_card() -> dict[str, Any]:
    """Üç sınavı birleştir, geçmişe EKLE (`SELFCHECK_FILE`), TREND üret. Çıktı:
    {'ts','knowledge','calibration','replay','trend','report_text'} — `report_text`
    reasoner.CLOSING ile biter, kenar iması içermez."""
    prev_records = _read_all()
    prev = prev_records[-1] if prev_records else None

    rec: dict[str, Any] = {"ts": _utc_now_iso(), "knowledge": exam_knowledge(),
                          "calibration": exam_calibration(), "replay": exam_replay(),
                          "production": exam_production()}
    try:
        _append(rec)
    except OSError as e:
        logger.warning(f"[ai.selfcheck] karne yazılamadı: {e}")

    trend = _trend_text(rec, prev)
    kw, cal, rep, pro = (rec["knowledge"], rec["calibration"], rec["replay"],
                         rec["production"])
    lines = [
        f"Öz-değerlendirme karnesi ({rec['ts']})",
        f"[Bilgi sınavı] n={kw['n']} skor={kw['score']} (DAG-tutarlı={kw['dag_tutarli']})",
        f"[Kalibrasyon sınavı] durum={cal['durum']}"
        + (f" isabet-posterior={cal.get('isabet_posterior')} trend={cal.get('trend')}"
           if cal["durum"] == "OLCULDU" else ""),
        f"[Replay sınavı] rejim-doğruluk={rep['rejim_dogruluk']} "
        f"abstain-oranı={rep['abstain_orani']} (n={rep['n_senaryo']})",
        f"[Üretim sınavı] durum={pro['durum']} n_karar={pro.get('n_karar', 0)} "
        f"n_yönlü={pro.get('n_yonlu', 0)} abstain={pro.get('abstain_orani')} "
        f"fazlalık={pro.get('fazlalik')}",
        *[f"  ⚠ {u}" for u in pro.get("uyarilar", [])],
        trend,
        "Not: bu karne modelin KENDİ tutarlılığını/kalibrasyonunu izler; kenar kanıtı değildir.",
        reasoner.CLOSING,
    ]
    return {**rec, "trend": trend, "report_text": "\n".join(lines)}


def main(argv: list[str] | None = None) -> int:
    """CLI: `--report` (üç sınavı koş + karne bas; manuel/tanılama amaçlı)."""
    import argparse
    ap = argparse.ArgumentParser(description="AI öz-değerlendirme sınavı (KARAR-DIŞI)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    if args.report:
        print(report_card()["report_text"])
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    raise SystemExit(main())
