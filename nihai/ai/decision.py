"""
decision.py — GÖLGE Bayes KARAR motoru (FAZ AI v2 "Quant Beyin", WP-C).

KARAR-DIŞI/GÖLGE: bu modülün çıktısı hiçbir üretim sinyalini üretmez, bastırmaz
veya boyutlandırmaz; strategy/execution/META_GATE'e bağlanamaz. enforce modu kasıtlı olarak yoktur.

Ne yapar: bir (symbol, strategy, features) üçlüsü için öğrenme-durumu (nihai.ai.learning)
tabanlı Bayesçi bir prior'dan, isteğe bağlı p_win harmanından, rejim (quant_math.classify_regime)
uyumundan ve geçmiş derslerden (nihai.ai.episodic.check_against_lessons) TÜRETİLMİŞ, tamamen
şeffaf bir GÖLGE karar üretir: LONG_BIAS / SHORT_BIAS / ABSTAIN + conviction + kelly önerisi.

Kip (AI_DECISION_MODE env — bkz. resolve_mode()):
  - "off"    (varsayılan) → saf hesap; HİÇBİR dosyaya yazılmaz.
  - "shadow"              → hesap + DECISIONS_FILE'a JSONL append + episodic'e bir epizot.
  - başka HERHANGİ bir değer (ör. yanlışlıkla "enforce" yazılması) → "off" muamelesi görür:
    dosya yazımı/epizot YALNIZCA env tam olarak "shadow" ise etkinleşir (güvenlik varsayılanı).

Karar akışı (7 adım — v2 ayrıntı: docs/AI_V2_BEYIN_SPEC.md §6; v3 farkları aşağıda ayrı
bölümde — `sources`/`neural_pwin`/`conformal_interval` verilmezse davranış BİREBİR aşağıdaki
gibidir, bu yüzden liste hâlâ "temel/degrade" akışın doğru özetidir):
  1) prior       : learning.load_state()'ten strateji-posterior (yoksa mean=0.5, geniş CI).
  2) p_win harmanı: post = 0.7*prior_mean + 0.3*clip(p_win, 0.01, 0.99) (p_win verildiyse;
                     v3'te bu, `sources` yokken hiyerarşik birleşimin SAYISAL DENK özel-hâlidir).
  3) rejim ayarı  : quant_math.classify_regime(closes) + statik strateji-rejim tablosu (±0.02).
  4) ders uyarısı : episodic.check_against_lessons(...) → şeffaflık amaçlı, CEZA değil.
  5) karar        : CI TABANI (adım-1'in lo95'i) 0.5'in üstünde değilse (belirsiz VEYA
                     aleyhte kanıt) VEYA maliyet-düzeltmeli kenar ≤ 0 → ABSTAIN; değilse
                     side_hint/momentum işaretine göre yön (v3'te İKİ kapı DAHA eklenir,
                     her ikisi de closes=None/conformal=None iken pasif/etkisiz kalır).
  6) kelly        : fractional_kelly(kelly_binary(post, 1.0)) (ABSTAIN → 0.0).
  7) kalıcılık    : yalnız mode=="shadow" (adım 1-6 HER kipte aynen çalışır — saf hesap ortak).

ÖNEMLİ: adım-5'teki CI testi HER ZAMAN adım-1'in (öğrenme durumundan gelen, ham) CI'ını
kullanır — p_win/rejim ayarlarıyla oynanan nokta-tahmin (`post`) bu CI'ı GENİŞLETMEZ/DARALTMAZ.
Amaç: tek bir p_win/rejim işaretinin, öğrenme geçmişinin gerçek istatistiksel belirsizliğini
"ezip geçmesini" önlemek (savunmacı tasarım).

Tüm dış çağrılar (learning/episodic/quant_math) tek tek korumalıdır: biri beklenmedik şekilde
patlarsa karar süreci çökmez, o adım güvenli varsayılana düşer.

--- v3 GENİŞLEMESİ (WP-D, docs/AI_V3_DERIN_BEYIN_SPEC.md §3.6) ---
Adım 2 artık İKİLİ (prior⊕p_win) değil HİYERARŞİK bir kaynak birleşimidir: prior (Beta) ⊕
p_win ⊕ neural_pwin (varsa; MC-dropout p±std, std büyükçe katkısı kısılır) ⊕ (adım-3'te
aynen) rejim ayarı. Her kaynağın ağırlığı learning.load_state()["sources"][k]["brier"]
varsa ondan MONOTON türetilir (`w = max(0.01, 0.25-brier)`; bkz. `_source_weight`); alan
yoksa/geçersizse o kaynak eski SABİT ağırlığına (_PRIOR_WEIGHT/_PWIN_WEIGHT/
_NEURAL_WEIGHT_FALLBACK) düşer — `sources` TAMAMEN yoksa/boşsa TÜM kaynaklar fallback
kullanır ve sonuç v2 ile SAYISAL DENKTİR (parite; bkz. testler).

Adım 5 artık ÜÇ KAPILIDIR (yalnız SIKILAŞMA — hiçbiri gevşetilmez):
  (a) v2'nin "prior CI tabanı>0.5 + maliyet-düzeltmeli kenar>0" kuralı AYNEN.
  (b) Fayda E = post·kazanç − (1−post)·(kayıp + λ·|CVaR95|) > 0 (`AI_UTILITY_LAMBDA` env,
      çağrı-anında okunur, varsayılan 0.5). closes verilmemişse kazanç=kayıp=1.0 ve CVaR
      terimi 0 → E=2·post−1: (a)'daki cost-edge'den DAHA GEVŞEK bir eşiktir, bu yüzden
      TEK BAŞINA v2 sonucunu değiştirmez (parite closes=None senaryosunda korunur);
      closes verilince CVaR cezası ek sıkılaştırma getirebilir (v3'ün amaçlanan davranışı).
      Tur-92 (BULGU-92-2): ceza eskiden KOŞULSUZ düşülüyordu (`… − λ·|CVaR95|`); kuyruk kaybı
      yalnız KAYBEDEN dalda yaşanır. Koşulsuz ceza tek-bar getirilerinde kazanç ortalamasını
      her zaman aştığı için E post=1'de bile negatifti (30 sembol×aralıkta gereken post
      1,05–1,48) → kapı HİÇBİR kanıtla geçilemiyordu. Kayıp dalına taşınınca eşik
      post > (kayıp+λ·CVaR)/(kazanç+kayıp+λ·CVaR) olur (10 sembol × 1m/1h/1d: 0,63–0,81).
  (c) `conformal_interval` verilmişse 0.5'i KAPSAMAMALI (kapsıyorsa ABSTAIN); None → pasif
      (Dalga-2'de WP-X bağlayacak).
Üçü de AND'lidir: herhangi biri geçilemezse ABSTAIN (v2'nin tek-koşullu ABSTAIN'i bunun
ALT-KÜMESİdir → v3 asla v2'nin ABSTAIN dediği yerde yön ÜRETMEZ, yalnızca v2'nin yön
ürettiği bazı durumları ABSTAIN'e çevirebilir — "ABSTAIN→yön" ASLA olmaz).

--- A2 (Tur-25/WP-A, 2026-08-10) — bayat prior'a karşı opt-in tüketim ---
`docs/AI_DENETIM_RAPORU_2026-07-26.md` §E.4: λ gevşetildiğinde motor bayat bir yan-koşullu
SHORT üstünlüğünü (geçmiş DOWN rejiminin artefaktı) prior alıp yükselen piyasada SHORT
girip kaybetti. `nihai.ai.learning`'in A1 EKİ zaman-ağırlıklı bir EK posterior üretir
(`state["strategies_decayed"]`); adım-1 (`_prior_from_state`) bunu YALNIZ `AI_PRIOR_DECAY`
env'i açıkken (1/true/on/yes) tüketir — kaynak etiketi "~decay" son-eki alır (ör.
"LONG~decay"). Env KAPALI (varsayılan) → bu kod yolu hiç çalışmaz, davranış ESKİSİYLE
BİT-DENKTİR. `_MIN_SIDE_PRIOR_N` eşiği ağırlıklı kipte de HAM (ağırlıksız) satır sayısına
bakar — etkin (float, decay'li) sayıma DEĞİL (bkz. `_lookup_prior_entry`) — aksi hâlde eşik
sessizce sıkılaşır. Etki yönü A1 ile aynı ilkeyi izler: daha az/daha eski kanıt → daha geniş
CI → kapı-a DAHA ZOR geçilir; bu değişiklik motoru daha çok susturabilir, daha çok
konuşturamaz.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike

from nihai.ai import episodic, learning, quant_math
from nihai.config import DATA_DIR

# Modül-düzeyi yol: testler monkeypatch'ler (memory.py/episodic.py deseni).
DECISIONS_FILE = DATA_DIR / "ai_decisions.jsonl"

_LOCK = threading.Lock()

# --- Sabit harman/ayar katsayıları (belgeli; spec §6 adım 2-3) ---
_PRIOR_WEIGHT = 0.7
_PWIN_WEIGHT = 0.3
_PWIN_CLIP_LO, _PWIN_CLIP_HI = 0.01, 0.99
_REGIME_ADJUST = 0.02

# --- v3 hiyerarşik kaynak ağırlıkları (spec §3.6 WP-D) ---
# sources[k]["brier"] geçerliyse: w = max(_BRIER_WEIGHT_FLOOR, _BRIER_WEIGHT_NEUTRAL - brier)
# (Brier 0 = mükemmel kalibrasyon → tavan ağırlık 0.25; Brier ≥0.25 = bilgisiz/kötü → taban 0.01).
# "neural" kaynağının v2-öncesi bir emsali yoktur; sources'ta henüz izlenmiyorsa pwin ile
# aynı sınıfta (nokta-tahmin + belirsizlik) muamele görsün diye _PWIN_WEIGHT'e eşitlendi.
_BRIER_WEIGHT_FLOOR = 0.01
_BRIER_WEIGHT_NEUTRAL = 0.25
_NEURAL_WEIGHT_FALLBACK = _PWIN_WEIGHT
# MC-dropout std'sinin neural katkısını kısma ölçeği: conf = 1/(1+(std/_NEURAL_STD_REF)^2)
# (std=0 → conf=1.0 tam katkı; std=_NEURAL_STD_REF'te conf=0.5; std→∞'de conf→0). 0.15,
# olasılık-çıktılı bir p_win std'si için makul orta-büyüklük referansıdır (belgeli sabit;
# WP-X/Dalga-2 kalibre edebilir).
_NEURAL_STD_REF = 0.15

# --- v3 fayda (utility) katmanı (spec §3.6 WP-D) ---
_UTILITY_LAMBDA_ENV = "AI_UTILITY_LAMBDA"
_UTILITY_LAMBDA_DEFAULT = 0.5
_UTILITY_CVAR_ALPHA = 0.05

# Statik strateji-rejim uyum tablosu (spec §6 adım 3). Adlar nihai.config.ACTIVE_STRATEGY
# dokümantasyonundaki modlarla eşleşir (event_breakout/mean_reversion/rsi_divergence/squeeze).
# "ensemble" kasıtlı olarak dışarıda bırakıldı: tek bir rejim-afinitesi tanımlı değil
# (ilk-tetik karışık mod).
_TREND_FOLLOWING_STRATEGIES = frozenset({"event_breakout", "squeeze", "rsi_divergence"})
_MEAN_REVERSION_STRATEGIES = frozenset({"mean_reversion"})

# Yön ipucu (side_hint) yoksa momentum işareti için sırayla denenecek
# nihai.signals.meta_features.FEATURE_NAMES anahtarları.
_MOMENTUM_FEATURE_KEYS = ("ret_20", "ema_gap", "ret_5", "streak")


@dataclass
class ShadowDecision:
    """Tek bir GÖLGE karar kaydı (spec §6). KARAR-DIŞI: hiçbir alanı sinyal/emir üretmez."""

    ts: str
    symbol: str
    strategy: str
    action: str                 # "LONG_BIAS" | "SHORT_BIAS" | "ABSTAIN"
    conviction: float           # 0..1
    prior_mean: float
    posterior: float
    evidence: list[str]         # Türkçe gerekçe zinciri (şeffaflık)
    kelly_suggest: float
    regime: dict | None
    lesson_warnings: list[str]
    mode: str                   # bu kararın ÜRETİLDİĞİ kip: "off" | "shadow"
    # --- v3 (WP-D) — YALNIZ SONA, default'lu (JSONL alan sırası/geriye-uyum korunur) ---
    utility: float | None = None   # fayda katmanı E (kapı-b); hesaplanamazsa None (savunmacı)


def _utc_now_iso() -> str:
    """UTC ts (tz-aware, saniye çözünürlüğü) → '...+00:00'. Yerel-saat karışıklığı yok."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def resolve_mode() -> str:
    """AI_DECISION_MODE env'ini kipe çevir: yalnızca TAM OLARAK 'shadow' (büyük/küçük harf
    duyarsız, kırpılmış) → 'shadow'; her şey (boş, 'off', yanlış yazım, 'enforce' dahil)
    → 'off'. Çağrı-anında okunur (import-anında DEĞİL) → testler monkeypatch.setenv ile
    güvenle değiştirebilir (env değişikliği sonraki decide() çağrısında hemen etkili olur)."""
    v = (os.getenv("AI_DECISION_MODE") or "").strip().lower()
    return "shadow" if v == "shadow" else "off"


# ---------------------------------------------------------------------------
# Adım 1 — prior
# ---------------------------------------------------------------------------
# Yan-koşullu (LONG/SHORT) prior için ASGARİ örneklem. Küçük örneklemde Beta posterior'un
# CI'ı dar görünüp kapı-a'yı şans eseri açabilir; bu eşik onu engeller.
_MIN_SIDE_PRIOR_N = 200

# --- A2 (Tur-25/WP-A, 2026-08-10): opt-in tüketim — nihai.ai.learning'in A1 EKİ olan
# zaman-ağırlıklı `state["strategies_decayed"]`'i okur. Env KAPALI (varsayılan) → bu kod
# yolu HİÇ çalışmaz, davranış v3 ile BİT-DENKTİR (bkz. testler). AÇMA/KAPAMA kümesi
# `resolve_mode()`'un "shadow" sözleşmesinden FARKLI (burada birden çok "açık" yazımı
# kabul edilir) — spec'in kendi ifadesi "1/true/on/yes" olduğundan bilerek geniş tutuldu.
_PRIOR_DECAY_ENV = "AI_PRIOR_DECAY"
_PRIOR_DECAY_TRUE_VALUES = frozenset({"1", "true", "on", "yes"})


def _resolve_prior_decay_enabled() -> bool:
    """AI_PRIOR_DECAY env'ini ÇAĞRI ANINDA oku (resolve_mode/_resolve_utility_lambda
    deseniyle AYNI felsefe: import-anında DEĞİL — research/shadow_decision_backtest.py gibi
    harness'lar env'i geçici değiştirerek AÇIK/KAPALI ölçümü yapar; import-anında okuma bu
    ölçümü bozar). AÇIK kümesi 1/true/on/yes (büyük/küçük harf duyarsız, kırpılmış);
    tanımsız/boş/bilinmeyen değer → False (varsayılan — mevcut davranış BİT-DENK kalır)."""
    return (os.getenv(_PRIOR_DECAY_ENV) or "").strip().lower() in _PRIOR_DECAY_TRUE_VALUES


def _side_key(side_hint: str | float | None, features: dict[str, float]) -> str | None:
    """Aday işlemin TARAFI ("LONG"/"SHORT") — yan-koşullu prior anahtarı. Yön çözülemezse
    None (GLOBAL'e düşülür). `AI_PRIOR_SIDE=off` ile eski GLOBAL-only davranışa dönülür."""
    if os.getenv("AI_PRIOR_SIDE", "").strip().lower() in ("0", "off", "false", "no"):
        return None
    sign = _direction_sign(side_hint, features)
    if sign > 0:
        return "LONG"
    if sign < 0:
        return "SHORT"
    return None


def _lookup_prior_entry(strategies: dict | None, decayed: dict | None, key: str,
                        decay_on: bool) -> tuple[dict | None, int, str]:
    """Tek bir prior anahtarı (strateji-adı/"LONG"/"SHORT"/"GLOBAL") için kullanılacak GİRİŞ
    → (entry, n_ham, kaynak_son_eki). `decay_on` AÇIK VE `decayed[key]` geçerliyse onu döner
    (son-ek "~decay", `n_ham` = entry'nin KENDİ `n_ham` alanı — A1'in HAM/ağırlıksız satır
    sayısı). Aksi hâlde (decay kapalı, VEYA bu anahtar için decayed girişi yok/bozuk) plain
    `strategies[key]`'e düşülür (son-ek "", `n_ham` = entry'nin `n` alanı — eski davranışla
    BİREBİR aynı okuma). İkisi de yoksa (None, 0, "").

    NEDEN n_ham (ham satır sayısı) HER İKİ dalda da eşik için kullanılan sayı: A1'in etkin
    (float, üstel-ağırlıklı) sayımı `n`, HAM sayıdan HER ZAMAN küçük-eşittir (her ağırlık
    ≤1.0) — eşik kontrolü etkin sayıma bakarsa `_MIN_SIDE_PRIOR_N` SESSİZCE sıkılaşır (aynı
    kanıt hacmi artık eşiği geçemez). Bu fonksiyon her iki dalda da HAM sayıyı döndürerek
    çağıranın eşiği her zaman doğru (kanıt hacmiyle orantılı) uygulamasını sağlar."""
    if decay_on and isinstance(decayed, dict):
        d_entry = decayed.get(key)
        if isinstance(d_entry, dict) and all(k in d_entry for k in ("post_mean", "lo95", "hi95")):
            try:
                n_ham = int(d_entry.get("n_ham") or 0)
            except (TypeError, ValueError):
                n_ham = 0
            return d_entry, n_ham, "~decay"
    entry = strategies.get(key) if isinstance(strategies, dict) else None
    if isinstance(entry, dict) and all(k in entry for k in ("post_mean", "lo95", "hi95")):
        try:
            n_ham = int(entry.get("n") or 0)
        except (TypeError, ValueError):
            n_ham = 0
        return entry, n_ham, ""
    return None, 0, ""


def _prior_from_state(strategy: str, side: str | None = None
                      ) -> tuple[float, float, float, str, dict | None]:
    """learning.load_state()'ten (mean, lo95, hi95, kaynak, decay_meta). Kaynak sırası:
    strateji-adı → YAN-KOŞULLU ("LONG"/"SHORT", yalnız n_ham≥`_MIN_SIDE_PRIOR_N` ise) →
    "GLOBAL" → "yok". Kayıt yok/bozuk/state okunamıyor → maksimum-belirsizlik prior
    (0.5, 0.0, 1.0, "yok", None) — CI [0,1] her zaman 0.5'i kapsar, bu da adım-5'te doğal
    olarak ABSTAIN'i tetikler (savunmacı davranış: veri yoksa karar verme).

    NEDEN yan-koşullu prior (2026-07-26 denetimi, B-1): canlı logda strateji kolonu yok,
    öğrenme durumu GLOBAL/LONG/SHORT anahtarlı. Yalnız (strateji, GLOBAL) denendiğinde
    kaynak HER ZAMAN GLOBAL'e düşüyordu; GLOBAL'in lo95'i 0.5'in altında olduğu için
    kapı-a (`lo95 > 0.5`) MATEMATİKSEL OLARAK hiç geçilemiyordu → %100 ABSTAIN, conviction
    daima 0.000, alt sistemin (fayda/CVaR/Kelly/conformal) tamamı ölü kod yolundaydı.
    Öğrenme katmanı yan-koşullu posterior'ları ZATEN üretiyordu; karar katmanı okumuyordu.

    DÜRÜSTLÜK NOTU: yan-koşullu bir üstünlük (ör. SHORT'un GLOBAL'den yüksek olması) büyük
    olasılıkla REJİM ARTEFAKTIDIR (ACI DERS #1/#7) — kenar değildir. Bu yüzden (a) kip
    yalnız `shadow`tır, enforce YOKTUR; (b) kaynak LONG/SHORT olduğunda `decide()` kanıt
    zincirine açık bir artefakt uyarısı yazar. Amaç kenar iddiası değil, kanıt döngüsünün
    YAŞAMASIDIR: %100 ABSTAIN üreten bir motor hakkında hiçbir şey ölçülemez (META_GATE'in
    `shadow` kipiyle aynı gerekçe).

    A2 (Tur-25/WP-A) — AI_PRIOR_DECAY opt-in zaman-ağırlıklı prior: env AÇIKSA (bkz.
    `_resolve_prior_decay_enabled`), her anahtar İÇİN ÖNCE `state["strategies_decayed"]`
    (learning.py A1'in üstel-unutmalı EK posterior'u) denenir, yoksa/boşsa AYNI anahtarın
    ağırlıksız `state["strategies"]` girişine düşülür (bkz. `_lookup_prior_entry`). Env
    KAPALI (varsayılan) → yalnızca ağırlıksız `strategies` okunur, davranış ESKİSİYLE
    BİT-DENKTİR (bkz. test_ai_prior_decay.py parite testleri). Kaynak etiketi ağırlıklı
    girişte "~decay" son-eki alır (ör. "LONG~decay") — `decide()`'ın kanıt zincirinde
    ayırt edilebilir olsun diye. `decay_meta` (yalnız "~decay" kullanıldığında dict, aksi
    hâlde None) `{'half_life_days', 'drift_shrink_uygulandi'}` taşır — çağıran bunu kanıt
    zincirine ayrı bir satır olarak yazabilir."""
    try:
        state = learning.load_state() or {}
    except Exception as e:  # noqa: BLE001 — öğrenme durumu okunamazsa geniş-belirsizliğe düş
        logger.debug(f"[ai.decision] learning.load_state hata: {e}")
        return 0.5, 0.0, 1.0, "yok", None
    decay_on = _resolve_prior_decay_enabled()
    strategies = state.get("strategies") if isinstance(state, dict) else None
    decayed = state.get("strategies_decayed") if isinstance(state, dict) else None
    keys = [strategy] + ([side] if side in ("LONG", "SHORT") else []) + ["GLOBAL"]
    for key in keys:
        entry, n_ham, src_suffix = _lookup_prior_entry(strategies, decayed, key, decay_on)
        if entry is None:
            continue
        if key in ("LONG", "SHORT") and n_ham < _MIN_SIDE_PRIOR_N:
            continue   # örneklem yetersiz (HAM satır sayısı) → bir sonraki kaynağa düş
        try:
            mean, lo, hi = float(entry["post_mean"]), float(entry["lo95"]), float(entry["hi95"])
        except (TypeError, ValueError):
            continue
        decay_meta = (
            {"half_life_days": entry.get("half_life_days"),
             "drift_shrink_uygulandi": bool(entry.get("drift_shrink_uygulandi"))}
            if src_suffix else None
        )
        return mean, lo, hi, key + src_suffix, decay_meta
    return 0.5, 0.0, 1.0, "yok", None


# ---------------------------------------------------------------------------
# Adım 2 — hiyerarşik kaynak birleşimi (v3, WP-D)
# ---------------------------------------------------------------------------
def _sources_from_state() -> dict | None:
    """learning.load_state()["sources"] savunmacı okuma. learning.py'A DOKUNULMAZ — yalnız
    dict'ten okur (spec: "sources alanını Sonnet-1 paralel ekliyor"). Alan yok/boş/bozuk/
    load_state patladı → None (çağıran o zaman TÜM kaynaklarda sabit fallback ağırlığa
    düşer — parite şartı budur)."""
    try:
        state = learning.load_state() or {}
    except Exception as e:  # noqa: BLE001 — kaynak-ağırlığı okunamazsa fallback'e düş
        logger.debug(f"[ai.decision] learning.load_state (sources) hata: {e}")
        return None
    src = state.get("sources") if isinstance(state, dict) else None
    return src if isinstance(src, dict) and src else None


def _source_weight(sources: dict | None, key: str, fallback: float) -> tuple[float, float | None]:
    """Kaynak `key`'in HAM (normalize edilmemiş) ağırlığı → (ağırlık, kullanılan_brier).

    `sources[key]["brier"]` sonlu bir sayıysa monoton formül: kötü kalibre (yüksek Brier)
    kaynak düşük ağırlık alır, taban `_BRIER_WEIGHT_FLOOR`'da durur (asla sıfırlanmaz — bir
    kaynağı TAMAMEN susturmak yerine küçük-ama-canlı katkı bırakılır, savunmacı tasarım).
    Alan yok/geçersiz → (fallback, None); `sources` parametresinin kendisi None/boşsa da
    aynı yol izlenir (tüm anahtarlar fallback alır → v2 sabit-ağırlık davranışına SAYISAL
    DENK degrade — parite testinin matematiksel temeli budur)."""
    entry = sources.get(key) if isinstance(sources, dict) else None
    if isinstance(entry, dict):
        raw_brier = entry.get("brier")
        if raw_brier is not None:   # doğrudan None-kontrolü (mypy daraltması için)
            try:
                b = float(raw_brier)
            except (TypeError, ValueError):
                b = None
            if b is not None and b == b:   # sonluluk (NaN != NaN) kontrolü
                return max(_BRIER_WEIGHT_FLOOR, _BRIER_WEIGHT_NEUTRAL - b), b
    return fallback, None


def _neural_std_confidence(std: float) -> float:
    """MC-dropout std'sinden [0,1] güven çarpanı: std=0→1.0 (tam katkı), std arttıkça
    monoton azalır (kare-ters formül; belgeli sabit `_NEURAL_STD_REF`). Negatif std
    (geçersiz girdi) 0.0'a kırpılır — savunmacı."""
    s = max(0.0, float(std))
    return 1.0 / (1.0 + (s / _NEURAL_STD_REF) ** 2)


# ---------------------------------------------------------------------------
# Adım 3 — rejim uyumu
# ---------------------------------------------------------------------------
def _regime_adjustment(strategy: str, regime: dict | None) -> float:
    """Statik strateji-rejim uyum tablosu → küçük ±0.02 posterior ayarı. Rejim yok/'unknown'
    veya strateji tabloda yoksa → 0.0 (nötr, ayar yapılmaz)."""
    if not regime or regime.get("label") == "unknown":
        return 0.0
    trend = regime.get("trend")
    is_mean_reverting = bool(regime.get("mean_reversion"))
    s = (strategy or "").strip().lower()
    if s in _MEAN_REVERSION_STRATEGIES:
        return _REGIME_ADJUST if is_mean_reverting else -_REGIME_ADJUST
    if s in _TREND_FOLLOWING_STRATEGIES:
        if trend in ("up", "down"):
            return _REGIME_ADJUST
        if trend == "flat":
            return -_REGIME_ADJUST
    return 0.0


# ---------------------------------------------------------------------------
# Adım 4 — ders uyarıları
# ---------------------------------------------------------------------------
def _lesson_context(strategy: str, regime: dict | None, used_p_win: bool) -> str:
    """episodic.check_against_lessons için anahtar-kelime bağlam metni (küçük-harf eşleşme
    episodic.py'nin işi — burada yalnız bağlam metni kurulur: strateji + rejim etiketi +
    p_win kullanıldıysa 'meta_labeling' ipucu)."""
    parts = [strategy]
    if regime and regime.get("label"):
        parts.append(str(regime["label"]))
    if used_p_win:
        parts.append("meta_labeling")
    return " ".join(parts)


def _collect_lesson_warnings(strategy: str, regime: dict | None, used_p_win: bool) -> list[str]:
    """İlgili ACI DERS'leri şeffaflık amaçlı Türkçe uyarı satırlarına çevirir (ceza değil).
    check_against_lessons patlarsa → [] (savunmacı; karar süreci asla düşmez)."""
    try:
        lessons = episodic.check_against_lessons(_lesson_context(strategy, regime, used_p_win))
    except Exception as e:  # noqa: BLE001 — ders kontrolü kararı asla düşürmemeli
        logger.debug(f"[ai.decision] check_against_lessons hata: {e}")
        return []
    return [f"[{les.id}] {les.claim} (guard: {les.guard})" for les in (lessons or [])]


# ---------------------------------------------------------------------------
# Fayda (utility) katmanı yardımcıları — kapı-b (v3, WP-D)
# ---------------------------------------------------------------------------
def _resolve_utility_lambda() -> float:
    """AI_UTILITY_LAMBDA env'ini ÇAĞRI ANINDA oku (resolve_mode deseniyle aynı felsefe:
    import-anında DEĞİL — testler monkeypatch.setenv ile güvenle değiştirebilir). Boş/bozuk/
    NaN → varsayılan `_UTILITY_LAMBDA_DEFAULT` (savunmacı; asla çökme/İstisna)."""
    raw = os.getenv(_UTILITY_LAMBDA_ENV)
    if raw is None or not raw.strip():
        return _UTILITY_LAMBDA_DEFAULT
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return _UTILITY_LAMBDA_DEFAULT
    return v if v == v else _UTILITY_LAMBDA_DEFAULT   # NaN guard


def _returns_from_closes(closes: ArrayLike | None) -> np.ndarray | None:
    """closes → sonlu log-getiri dizisi (savunmacı). None/kısa/dönüştürülemez/<2 sonlu
    getiri → None (çağıran fayda katmanını nötr 1:1/0 varsayılana düşürür)."""
    if closes is None:
        return None
    try:
        arr = np.asarray(closes, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr)):
        return None
    try:
        rets = quant_math.log_returns(arr)
    except Exception as e:  # noqa: BLE001 — getiri hesaplanamazsa fayda katmanı nötr kalır
        logger.debug(f"[ai.decision] log_returns hata: {e}")
        return None
    rets = rets[np.isfinite(rets)]
    return rets if rets.size >= 2 else None


def _estimate_gain_loss(rets: np.ndarray | None) -> tuple[float, float]:
    """Fayda E = post·kazanç−(1−post)·kayıp girdisi: pozitif/negatif getirilerin ORTALAMA
    BÜYÜKLÜĞÜ (kazanç/kayıp oranı). Veri yok/tek-yönlü/geçersiz → (1.0, 1.0) NÖTR — bu durumda
    E = post·1−(1−post)·1 = 2·post−1'e indirgenir (closes=None çağrılarıyla SAYISAL DENK;
    parite notu: spec §3.6/modül docstring'i)."""
    if rets is None or rets.size == 0:
        return 1.0, 1.0
    pos = rets[rets > 0]
    neg = rets[rets < 0]
    if pos.size == 0 or neg.size == 0:
        return 1.0, 1.0
    gain = float(np.mean(pos))
    loss = float(np.mean(np.abs(neg)))
    if not (np.isfinite(gain) and np.isfinite(loss)) or gain <= 0 or loss <= 0:
        return 1.0, 1.0
    return gain, loss


def _cvar_penalty(rets: np.ndarray | None) -> float:
    """CVaR95 BÜYÜKLÜĞÜ (≥0; quant_math.cvar_historical'a ince bir savunmacı sarmalayıcı).
    Veri yok/hesaplanamaz/None → 0.0 (ceza yok — closes verilmemiş çağrılarla NÖTR parite)."""
    if rets is None:
        return 0.0
    try:
        cv = quant_math.cvar_historical(rets, alpha=_UTILITY_CVAR_ALPHA)
    except Exception as e:  # noqa: BLE001 — CVaR hesabı fayda katmanını düşürmemeli
        logger.debug(f"[ai.decision] cvar_historical hata: {e}")
        return 0.0
    return float(cv) if cv is not None and cv == cv else 0.0


# ---------------------------------------------------------------------------
# Adım 5 — yön ipucu / momentum işareti
# ---------------------------------------------------------------------------
def _direction_sign(side_hint: str | float | None, features: dict[str, float]) -> int:
    """side_hint > features-momentum > 0 (bilinmiyor, çağıran posterior işaretine düşer).
    side_hint metinse LONG*/SHORT* eş anlamlıları, sayısalsa işareti kullanılır. side_hint
    yoksa/çözülemezse sırayla _MOMENTUM_FEATURE_KEYS'teki ilk sıfırdan-farklı değerin
    işareti kullanılır (ret_20 → ema_gap → ret_5 → streak)."""
    if isinstance(side_hint, str):
        s = side_hint.strip().upper()
        if s in ("LONG", "LONG_BIAS", "BUY", "UP"):
            return 1
        if s in ("SHORT", "SHORT_BIAS", "SELL", "DOWN"):
            return -1
    elif side_hint is not None:
        try:
            v = float(side_hint)
            if v != 0:
                return 1 if v > 0 else -1
        except (TypeError, ValueError):
            pass
    for key in _MOMENTUM_FEATURE_KEYS:
        try:
            fv = float(features.get(key, 0.0))
        except (TypeError, ValueError):
            continue
        if fv != 0:
            return 1 if fv > 0 else -1
    return 0


# ---------------------------------------------------------------------------
# Kalıcılık (yalnız mode == "shadow")
# ---------------------------------------------------------------------------
def _append_decision_file(decision: ShadowDecision) -> None:
    """DECISIONS_FILE'a tek satır JSONL ekle (thread-safe append; memory.py kilit deseni)."""
    line = json.dumps(asdict(decision), ensure_ascii=False)
    with _LOCK:
        DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with DECISIONS_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _record_shadow(decision: ShadowDecision, features: dict[str, float],
                   episode: bool = True) -> None:
    """mode=='shadow' iken çağrılır: dosya append + episodic'e bir epizot. Her adım kendi
    içinde korumalı — biri başarısız olsa da karar nesnesi çağırana zaten döndürülmüştür.

    `episode=False` (Tur-90, BULGU-90-7): anı yazılmaz. Epizodik hafızanın sonucu 5 GÜNLÜK bardan
    ölçülür (`episode_resolver`); gün-içi bar kararının anısı o ufka ait değildir."""
    try:
        _append_decision_file(decision)
    except OSError as e:
        logger.warning(f"[ai.decision] {DECISIONS_FILE.name} yazılamadı: {e}")
    if not episode:
        return
    try:
        episodic.append_episode(
            kind="ai_decision",
            symbol=decision.symbol,
            features=dict(features),
            summary=f"{decision.strategy}: {decision.action} (conviction={decision.conviction:.2f})",
            tags=[decision.strategy, decision.action.lower()],
            outcome=None,
            ayni_gun_tekil=True,       # Tur-77: aynı günlük bar gün içinde yeniden anı YAZMAZ
        )
    except Exception as e:  # noqa: BLE001 — epizot yazımı kararı asla düşürmemeli
        logger.debug(f"[ai.decision] epizot yazılamadı: {e}")


# ---------------------------------------------------------------------------
# Ana API
# ---------------------------------------------------------------------------
def decide(
    symbol: str,
    strategy: str,
    features: dict[str, float],
    p_win: float | None = None,
    closes: ArrayLike | None = None,
    side_hint: str | float | None = None,
    cost_bps: float = 5.0,
    neural_pwin: tuple[float, float] | None = None,
    conformal_interval: tuple[float, float] | None = None,
    record_episode: bool = True,
    persist: bool = True,
) -> ShadowDecision:
    """GÖLGE Bayes kararı üret (modül docstring'indeki 7 adım + v3 hiyerarşik birleşim/
    fayda/üçlü-kapı genişlemesi). KARAR-DIŞI: dönüş değeri hiçbir üretim sinyaline/eşiğine/
    META_GATE'e bağlanmaz; yalnızca mode=='shadow' iken kalıcılaştırılır (DECISIONS_FILE +
    episodic epizot).

    v3 YENİ parametreler (SONA eklendi, ikisi de default'lu — geriye uyumlu):
      neural_pwin: (p, std) — nöral kapasite katmanının MC-dropout p_win±std'si (varsa,
        WP-N/Dalga-2 bağlar); std büyüdükçe katkısı `_neural_std_confidence` ile kısılır.
      conformal_interval: (lo, hi) — split-conformal %90 garanti aralığı (varsa, WP-X/
        Dalga-2 bağlar); 0.5'i kapsıyorsa yön çıktısı ABSTAIN'e döner (kapı-c)."""
    mode = resolve_mode()
    features = dict(features) if features else {}
    evidence: list[str] = []

    # 1) prior (kaynak: strateji-adı → yan-koşullu LONG/SHORT → GLOBAL → yok; bkz.
    #    _prior_from_state / 2026-07-26 denetimi B-1)
    prior_mean, lo95, hi95, prior_src, decay_meta = _prior_from_state(
        strategy, _side_key(side_hint, features))
    evidence.append(
        f"Prior (kaynak={prior_src}): ortalama={prior_mean:.3f}, %95 CI=[{lo95:.3f}, {hi95:.3f}] "
        "(nihai.ai.learning öğrenme durumu)."
    )
    if decay_meta is not None:   # A2: yalnız AI_PRIOR_DECAY açıkken + decayed giriş kullanıldıysa
        evidence.append(
            f"Zaman-ağırlıklı prior (AI_PRIOR_DECAY): yarılanma ömrü="
            f"{decay_meta.get('half_life_days')} gün, drift-shrink uygulandı="
            f"{decay_meta.get('drift_shrink_uygulandi')}."
        )
    # "~decay" son-eki (A2) kaynak KİMLİĞİNİ değiştirmez, yalnız AĞIRLIKLANDIRMA yöntemini —
    # yan-koşullu artefakt uyarısı hem "LONG"/"SHORT" hem "LONG~decay"/"SHORT~decay" için geçerli.
    if prior_src.split("~", 1)[0] in ("LONG", "SHORT"):
        evidence.append(
            f"UYARI: yan-koşullu prior ({prior_src}) kullanıldı — yan-koşullu üstünlük büyük "
            "olasılıkla REJİM ARTEFAKTIDIR (ACI DERS #1/#7), kenar kanıtı DEĞİLDİR."
        )

    # 2) hiyerarşik kaynak birleşimi: prior ⊕ p_win ⊕ neural_pwin (v3, WP-D — bkz. modül
    # docstring'i). Kaynak ağırlıkları learning.load_state()["sources"] varsa Brier-monoton,
    # yoksa/boşsa v2'nin SABİT ağırlıklarına (_PRIOR_WEIGHT/_PWIN_WEIGHT) düşer → sources
    # yokken + neural_pwin=None iken matematiksel olarak v2 formülüyle AYNI (parite).
    sources_state = _sources_from_state()
    # (ad, ham_ağırlık, nokta_tahmin, kullanılan_brier)
    contributions: list[tuple[str, float, float, float | None]] = []

    prior_w, prior_brier = _source_weight(sources_state, "prior", _PRIOR_WEIGHT)
    contributions.append(("prior", prior_w, prior_mean, prior_brier))

    if p_win is not None:
        p_clipped = _clip(float(p_win), _PWIN_CLIP_LO, _PWIN_CLIP_HI)
        pwin_w, pwin_brier = _source_weight(sources_state, "pwin", _PWIN_WEIGHT)
        contributions.append(("pwin", pwin_w, p_clipped, pwin_brier))
        evidence.append(f"p_win={float(p_win):.3f} (kırpılmış {p_clipped:.3f}).")

    neural_std_conf: float | None = None
    if neural_pwin is not None:
        try:
            n_point = _clip(float(neural_pwin[0]), _PWIN_CLIP_LO, _PWIN_CLIP_HI)
            n_std = float(neural_pwin[1])
        except (TypeError, ValueError, IndexError) as e:
            logger.debug(f"[ai.decision] neural_pwin bozuk, kaynak atlanıyor: {e}")
        else:
            neural_w_raw, neural_brier = _source_weight(sources_state, "neural", _NEURAL_WEIGHT_FALLBACK)
            neural_std_conf = _neural_std_confidence(n_std)
            contributions.append(("neural", neural_w_raw * neural_std_conf, n_point, neural_brier))

    total_w = sum(w for _, w, _, _ in contributions)
    if total_w <= 0:   # matematiksel olarak imkansız (taban 0.01) ama sıfıra-bölme koruması
        total_w = 1.0
    post = sum(w * pt for _, w, pt, _ in contributions) / total_w
    post = _clip(post, 0.0, 1.0)

    for name, w, pt, brier in contributions:
        w_norm = w / total_w
        brier_txt = f"{brier:.4f}" if brier is not None else "yok→sabit ağırlık"
        std_txt = f", std-güven={neural_std_conf:.3f}" if name == "neural" and neural_std_conf is not None else ""
        evidence.append(f"Kaynak[{name}]: değer={pt:.3f}, ağırlık={w_norm:.3f} (brier={brier_txt}){std_txt}.")
    evidence.append(f"Hiyerarşik birleşim → posterior={post:.3f}.")

    # 3) rejim uyumu (yalnız closes verildiyse)
    regime: dict | None = None
    if closes is not None:
        try:
            regime = quant_math.classify_regime(closes)
        except Exception as e:  # noqa: BLE001 — rejim hesaplanamazsa karar süreci çökmemeli
            logger.debug(f"[ai.decision] classify_regime hata: {e}")
            regime = None
        adj = _regime_adjustment(strategy, regime)
        if adj != 0.0:
            post = _clip(post + adj, 0.0, 1.0)
            evidence.append(
                f"Rejim uyumu ({(regime or {}).get('label', '?')}): posterior {adj:+.2f} "
                f"ayarlandı → {post:.3f}."
            )
        elif regime is not None:
            evidence.append(
                f"Rejim ({regime.get('label', '?')}): strateji-uyum tablosunda ayar tanımlı değil."
            )

    # 4) ders uyarıları (ceza değil, şeffaflık)
    lesson_warnings = _collect_lesson_warnings(strategy, regime, p_win is not None)
    if lesson_warnings:
        evidence.append(
            f"{len(lesson_warnings)} ilgili ACI DERS bulundu (aşağıda; ceza değil, şeffaflık amaçlı)."
        )

    # 5) ABSTAIN / yön kararı — v3'te ÜÇ KAPI (yalnız SIKILAŞMA; bkz. modül docstring'i).
    # Kapı-a (v2 AYNEN): CI HER ZAMAN adım-1'in ham CI'ıdır. Yön için CI TABANI 0.5'in
    # ÜSTÜNDE olmalı: CI 0.5'i kapsıyorsa kanıt belirsizdir; CI tamamen 0.5'in ALTINDAYSA
    # kanıt izlenen tarafın ALEYHİNEDİR — ikisinde de ABSTAIN (aleyhte kanıttan "ters yöne
    # gir" kenarı TÜRETİLMEZ: ACI DERS #3/#7 disiplinine aykırı). Kapı-a'nın maliyet-kenarı
    # alt-koşulu da AYNEN korunur (yalnız isim/evidence etiketi netleştirildi).
    ci_supports_side = lo95 > 0.5
    # Tur-90 (BULGU-90-6): taraf çözüldüyse `post` O TARAFIN kazanma olasılığıdır (prior yan-koşullu,
    # p_win o işlemin) → kenar İŞARETLİDİR. `|2p−1|` post=0,398'i (LMT, 23 Eyl) +0,20 "kenar" sayıyordu:
    # kapı-b geçseydi aleyhte kanıtla LONG_BIAS (conviction 0,20) çıkacaktı. Mutlak değer yalnız
    # taraf BİLİNMEZKEN doğrudur (yön orada posterior işaretinden seçilir).
    tarafli = _direction_sign(side_hint, features) != 0
    edge = ((2 * post - 1) if tarafli else abs(2 * post - 1)) - 2 * cost_bps / 1e4
    kenar_txt = f"(2*{post:.3f}-1)" if tarafli else f"|2*{post:.3f}-1|"
    evidence.append(
        f"Kapı-a maliyet-düzeltmeli kenar: {kenar_txt} - 2*{cost_bps:.1f}bps/1e4 = {edge:.4f}."
    )
    gate_a = ci_supports_side and edge > 0

    # Kapı-b (YENİ, v3): fayda E = post·kazanç−(1−post)·(kayıp+λ·|CVaR95|). closes yoksa
    # kazanç=kayıp=1.0 ve CVaR terimi 0 → E=2·post−1 (kapı-a'nın kenarından DAHA GEVŞEK,
    # bu yüzden closes=None'da TEK BAŞINA yön değiştirmez — parite; closes varsa CVaR cezası
    # ek sıkılaştırma getirebilir — v3'ün amaçlanan davranışı). Kuyruk cezası yalnız KAYBEDEN
    # dalda: koşulsuz düşülünce kapı post=1'de bile geçilemiyordu (Tur-92, BULGU-92-2).
    rets = _returns_from_closes(closes)
    gain, loss = _estimate_gain_loss(rets)
    cvar_pen = _cvar_penalty(rets)
    util_lambda = _resolve_utility_lambda()
    utility = post * gain - (1.0 - post) * (loss + util_lambda * abs(cvar_pen))
    evidence.append(
        f"Kapı-b fayda: E = {post:.3f}*{gain:.4f} - {1 - post:.3f}*({loss:.4f} "
        f"+ {util_lambda:.2f}*{abs(cvar_pen):.4f}) = {utility:.4f}."
    )
    gate_b = utility > 0

    # Kapı-c (YENİ, v3): conformal_interval verilmemişse PASİF (her zaman geçer — Dalga-2
    # WP-X bağlayana dek nötr). Verilmişse 0.5'i kapsaması ABSTAIN'i tetikler.
    gate_c = True
    conformal_active = False
    if conformal_interval is not None:   # doğrudan None-kontrolü (mypy daraltması için)
        conformal_active = True
        try:
            c_lo, c_hi = float(conformal_interval[0]), float(conformal_interval[1])
        except (TypeError, ValueError, IndexError) as e:
            logger.debug(f"[ai.decision] conformal_interval bozuk, kapı-c pasif: {e}")
            conformal_active = False
        else:
            gate_c = not (c_lo <= 0.5 <= c_hi)
            evidence.append(
                f"Kapı-c conformal aralık [{c_lo:.3f}, {c_hi:.3f}]: "
                + ("0.5'i kapsamıyor, geçti." if gate_c else "0.5'i kapsıyor, ABSTAIN tetiklendi.")
            )

    if not (gate_a and gate_b and gate_c):
        action = "ABSTAIN"
        conviction = 0.0
        reasons: list[str] = []
        if not ci_supports_side:
            reasons.append("prior CI tabanı 0.5'in üstünde değil (belirsiz veya aleyhte kanıt)")
        elif edge <= 0:
            reasons.append("kapı-a maliyet-düzeltmeli kenar ≤ 0")
        if not gate_b:
            reasons.append("kapı-b fayda E ≤ 0")
        if conformal_active and not gate_c:
            reasons.append("kapı-c conformal aralık 0.5'i kapsıyor")
        evidence.append(f"Karar: ABSTAIN ({'; '.join(reasons)}).")
    else:
        sign = _direction_sign(side_hint, features)
        if sign == 0:
            sign = 1 if post > 0.5 else -1
        action = "LONG_BIAS" if sign > 0 else "SHORT_BIAS"
        conviction = abs(2 * post - 1)
        evidence.append(f"Karar: {action} (conviction={conviction:.3f}).")

    # 6) kelly önerisi (yalnız ABSTAIN değilse hesaplanır)
    if action == "ABSTAIN":
        kelly_suggest = 0.0
    else:
        try:
            kelly_suggest = quant_math.fractional_kelly(quant_math.kelly_binary(post, 1.0))
        except Exception as e:  # noqa: BLE001 — kelly hesap hatası kararı düşürmemeli
            logger.debug(f"[ai.decision] kelly hesap hata: {e}")
            kelly_suggest = 0.0

    decision = ShadowDecision(
        ts=_utc_now_iso(),
        symbol=str(symbol),
        strategy=str(strategy),
        action=action,
        conviction=float(conviction),
        prior_mean=float(prior_mean),
        posterior=float(post),
        evidence=evidence,
        kelly_suggest=float(kelly_suggest),
        regime=regime,
        lesson_warnings=lesson_warnings,
        mode=mode,
        utility=float(utility),
    )

    # 7) kalıcılık — YALNIZ shadow (off → hiçbir dosyaya dokunulmaz). `persist=False`: pano
    # görüntüleme sorgusu (Hisse Detay, Tur-92) — aynı hesap, karar defterine/anıya YAZILMAZ;
    # yoksa her "Analiz Et" tıklaması gölge karar raporunu şişirirdi.
    if mode == "shadow" and persist:
        _record_shadow(decision, features, episode=record_episode)

    return decision


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------
def _iter_decisions() -> list[dict]:
    """DECISIONS_FILE'daki geçerli kayıtlar. Bozuk/yarım JSON satırlar SESSİZCE ATLANMAZ:
    atlanır AMA sayısı logger.warning'e yazılır (memory.py ile aynı desen)."""
    if not DECISIONS_FILE.exists():
        return []
    records: list[dict] = []
    corrupt = 0
    try:
        with DECISIONS_FILE.open(encoding="utf-8") as f:
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
                    records.append(obj)
                else:
                    corrupt += 1
    except OSError as e:
        logger.warning(f"[ai.decision] okunamadı ({DECISIONS_FILE.name}): {e}")
        return records
    if corrupt:
        logger.warning(f"[ai.decision] {corrupt} bozuk satır atlandı ({DECISIONS_FILE.name})")
    return records


def _parse_ts(ts: object) -> datetime | None:
    """ISO ts → tz-aware datetime (naive ise UTC varsay). Ayrıştırılamazsa None."""
    if not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def decision_report(days: int = 7) -> str:
    """Son `days` gündeki GÖLGE kararların Türkçe özeti: n, abstain oranı, ortalama
    conviction, ders-uyarılı karar sayısı. KARAR-DIŞI: bu rapor bir kenar/performans iddiası
    içermez — yalnızca gölge-karar hacmini/dağılımını gözlemler (ACI DERS #3/#4 disiplini)."""
    cutoff = datetime.now(UTC) - timedelta(days=days)
    recs: list[dict] = []
    for rec in _iter_decisions():
        ts = _parse_ts(rec.get("ts"))
        if ts is not None and ts < cutoff:
            continue
        recs.append(rec)

    n = len(recs)
    if n == 0:
        return f"Gölge karar raporu (son {days} gün): kayıt yok."

    n_abstain = sum(1 for r in recs if r.get("action") == "ABSTAIN")
    convictions: list[float] = []
    for r in recs:
        try:
            convictions.append(float(r.get("conviction", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    avg_conv = sum(convictions) / len(convictions) if convictions else 0.0
    n_warn = sum(1 for r in recs if r.get("lesson_warnings"))
    abstain_pct = 100.0 * n_abstain / n

    return "\n".join([
        f"Gölge karar raporu (son {days} gün): n={n}, abstain %{abstain_pct:.1f}, "
        f"ortalama conviction={avg_conv:.3f}, ders-uyarılı {n_warn}/{n}.",
        "Bu rapor KARAR-DIŞI: gölge kararların hacmini/dağılımını gözlemler, "
        "kenar veya performans iddiası içermez.",
    ])
