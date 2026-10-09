"""
brain.py — Quant Beyin ORKESTRATÖRÜ (FAZ AI v2 · v3 Tur-16). KARAR-DIŞI/GÖLGE.

Düşünme döngüsü v3: ALGILA (çağıranın verdiği özellik/fiyat bağlamı) → HATIRLA (önce ucuz
ANLAMSAL özet [consolidate.semantic_recall], sonra ham epizodik benzerlik + LESSONS) →
TOPRAKLA (knowledge kavram bağları) → HESAPLA (quant_math: rejim/volatilite/risk) →
ÖNGÖR (duck-tipli neural modelden gölge p_win ± std; model yoksa adım pasif) →
KARAR (decision.decide — YALNIZ gölge; çift/üçlü kapı v3) → ÖZ-ELEŞTİR (deterministik
zayıf-halka raporu; LLM'siz) → KAYDET (gölge modda memory digest) → raporla (Türkçe zincir).

SÖZLEŞME: hiçbir çıktı üretim sinyaline/eşiğe/META_GATE'e bağlanmaz; "kenar bulundu" imalı
olamaz (ACI DERS #3/#4). `--demo` hiçbir dosyaya yazmaz (AI_DECISION_MODE=off zorlanır).
brain, `nihai.ai.neural`'e IMPORT BAĞI ALMAZ (torch'suz ortamda da aynen çalışır) —
neural model çağırandan duck-tipli gelir (is_fitted/feature_names/predict_with_uncertainty).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from nihai.ai import consolidate, data_health, episodic, knowledge, learning, memory, quant_math, reasoner
from nihai.ai import decision as decision_mod
from nihai.ai.decision import ShadowDecision

# Rapor kapanışı reasoner ile TEK kaynak (dürüstlük cümlesi her brifte).
CLOSING = reasoner.CLOSING


@dataclass
class Thought:
    """Tek bir düşünme döngüsünün izlenebilir çıktısı (yalnız rapor/hafıza; karar-dışı).
    v3 alanları SONDA ve default'lu (geriye-uyum: mevcut kurucular/tüketiciler kırılmaz)."""

    ts: str
    symbol: str
    strategy: str
    regime: dict | None
    risk: dict
    recalls: list[tuple[str, float]] = field(default_factory=list)
    lesson_warnings: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    decision: ShadowDecision | None = None
    narrative: str = ""
    neural_pwin: tuple[float, float] | None = None
    semantic: list[str] = field(default_factory=list)
    weak_links: list[str] = field(default_factory=list)
    health: dict | None = None
    conformal: tuple[float, float] | None = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _risk_block(closes: np.ndarray | None) -> dict:
    """Fiyat serisinden savunmacı risk özetleri (veri kısa/yok → boş alanlar)."""
    out: dict = {"vol_yillik": None, "cvar95_gunluk": None, "max_dd": None}
    if closes is None or len(closes) < 30:
        return out
    rets = quant_math.log_returns(np.asarray(closes, dtype=float))
    if rets is None or len(rets) < 20:
        return out
    v = quant_math.vol_close_to_close(np.asarray(closes, dtype=float))
    if v is not None:
        out["vol_yillik"] = round(float(quant_math.annualize_vol(v, 252)), 4)
    cv = quant_math.cvar_historical(rets, alpha=0.05)
    if cv is not None:
        out["cvar95_gunluk"] = round(float(cv), 5)
    dd = quant_math.max_drawdown(np.asarray(closes, dtype=float))
    if dd is not None:
        out["max_dd"] = round(float(dd[0]), 4)
    return out


def _recall(features: dict | None, strategy: str) -> tuple[list[tuple[str, float]], list[str]]:
    """Epizodik geri çağırma + ders taraması (hata → boş; düşünme asla çökmez)."""
    recalls: list[tuple[str, float]] = []
    warns: list[str] = []
    try:
        if features:
            num = {k: float(v) for k, v in features.items()
                   if isinstance(v, (int, float)) and v == v}
            for ep, dist in episodic.similar_episodes(num, k=3):
                recalls.append((ep.summary or ep.kind, round(float(dist), 3)))
    except Exception as e:  # noqa: BLE001 — hafıza arızası düşünmeyi durdurmasın
        logger.debug(f"[ai.brain] epizodik geri çağırma hatası: {e}")
    try:
        warns = [f"{les.id}: {les.claim}" for les in episodic.check_against_lessons(strategy)]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.brain] ders taraması hatası: {e}")
    return recalls, warns


def _semantic_lines(features: dict | None) -> list[str]:
    """HATIRLA'nın ucuz ilk basamağı (v3): konsolide anlamsal özetler. Kayıt yok/boş/hata →
    boş liste (ham epizot araması _recall her durumda AYRICA çalışır — sıra: önce özet,
    sonra ham; consolidate.py docstring sözleşmesi)."""
    if not features:
        return []
    try:
        num = {k: float(v) for k, v in features.items()
               if isinstance(v, (int, float)) and v == v}
        out: list[str] = []
        for rec in consolidate.semantic_recall(num, k=3):
            out.append(f"{rec.get('pattern', '?')} (n={rec.get('n')}, isabet={rec.get('isabet')})")
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.brain] anlamsal geri çağırma hatası: {e}")
        return []


def _neural_forecast(neural_model: Any | None,
                     features: dict | None) -> tuple[float, float] | None:
    """ÖNGÖR adımı (v3): duck-tipli neural modelden gölge p_win ± std. Model yok/fit'siz/
    hatalı → None (düşünme asla çökmez; fail-soft). Girdi satırı modelin feature_names
    sırasıyla kurulur; eksik/NaN özellik 0.0 (nötr) doldurulur — tam vektörü çağıran sağlar."""
    if neural_model is None or not features:
        return None
    try:
        if not getattr(neural_model, "is_fitted", False):
            return None
        names = list(getattr(neural_model, "feature_names", []) or [])
        if not names:
            return None
        row = []
        for nm in names:
            v = features.get(nm)
            row.append(float(v) if isinstance(v, (int, float)) and v == v else 0.0)
        x = np.asarray([row], dtype=float)
        # Tur-92 (BULGU-92-3): gölge koşucu GÜNLÜK barın özelliklerini 1dk'da eğitilmiş modele
        # veriyordu (max|z| 21–71, eğitim q99 3,15) → doymuş p (INTC 0,05) karar posteriorunu
        # %43 ağırlıkla aşağı çekiyordu. Canlı köprüyle (`neural_shadow`) AYNI eşik kuralı ve AYNI
        # env: sayı → sabit eşik (vars. 6,0), "auto" → max(4, 1,5·q99); ≤0 → kapı kapalı.
        raw = (os.getenv("NEURAL_SHADOW_MAX_Z") or "6.0").strip().lower()
        zq = getattr(neural_model, "train_zmax_q99", None)
        try:
            esik = max(4.0, 1.5 * float(zq)) if raw == "auto" and zq else float(raw)
        except ValueError:
            esik = 6.0
        if esik > 0 and hasattr(neural_model, "ood_zmax"):
            zmax = float(np.asarray(neural_model.ood_zmax(x)).ravel()[0])
            if not zmax <= esik:
                logger.debug(f"[ai.brain] neural girdi dağılım-dışı (max|z|={zmax:.1f}) — kaynak atlandı")
                return None
        if hasattr(neural_model, "predict_with_uncertainty"):
            p_arr, s_arr = neural_model.predict_with_uncertainty(x)
            return float(np.asarray(p_arr).ravel()[0]), float(np.asarray(s_arr).ravel()[0])
        p_arr = neural_model.predict_proba(x)
        return float(np.asarray(p_arr).ravel()[0]), 0.0
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.brain] neural öngörü hatası: {e}")
        return None


def _health_gate(closes_arr: np.ndarray | None, symbol: str) -> dict | None:
    """ALGILA başındaki veri-sağlık kapısı (v3/WP-O köprüsü): kapanış serisini
    data_health.health_check'e indirger. Veri yoksa/denetim çökerse None (kapı pasif —
    savunmacı; sağlık ölçülemedi diye düşünme durmaz, ama karantina da İLAN EDİLMEZ)."""
    if closes_arr is None or len(closes_arr) == 0:
        return None
    try:
        return data_health.health_check(pd.DataFrame({"close": closes_arr}), symbol=symbol)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.brain] veri-sağlık denetimi hatası: {e}")
        return None


def _quarantine_abstain(symbol: str, strategy: str, regime: dict | None,
                        lesson_warnings: list[str], health: dict) -> ShadowDecision:
    """Karantinalı veride ZORUNLU ABSTAIN kararı (WP-O sözleşmesi: 'veri güvenilmez' →
    karar motoru hiç çağrılmaz; posterior nötr 0.5, kelly 0). Gerekçe zinciri şeffaf."""
    issues = ", ".join(str(i.get("type")) for i in (health.get("issues") or [])[:4]) or "?"
    return ShadowDecision(
        ts=_utc_now_iso(), symbol=symbol, strategy=strategy, action="ABSTAIN",
        conviction=0.0, prior_mean=0.5, posterior=0.5,
        evidence=[f"Veri-sağlık kapısı: skor={health.get('score')} → KARANTİNA "
                  f"(sorunlar: {issues}) — veri güvenilmez, karar motoru çağrılmadı."],
        kelly_suggest=0.0, regime=regime, lesson_warnings=lesson_warnings,
        mode=decision_mod.resolve_mode())


def _conformal_from_calib(neural_pwin: tuple[float, float] | None,
                          conformal_calib: dict | None) -> tuple[float, float] | None:
    """Conformal 3. kapı köprüsü (WP-X): kalibrasyon sözlüğü + neural p_win varsa split-
    conformal aralık üretir; explain modülü/girdi yoksa None (kapı-c decision'da pasif kalır)."""
    if neural_pwin is None or conformal_calib is None:
        return None
    try:
        from nihai.ai import explain
        return explain.conformal_interval(float(neural_pwin[0]), conformal_calib)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.brain] conformal aralık hesabı hatası: {e}")
        return None


def _self_critique(t: Thought, features: dict | None) -> list[str]:
    """ÖZ-ELEŞTİR adımı (v3): bu düşüncenin EN ZAYIF halkaları — deterministik kural,
    LLM'siz. YALNIZ rapor: hiçbir eşiği/kararı değiştirmez (anayasa). En fazla 5 madde."""
    weak: list[str] = []
    try:
        srcs = (learning.load_state().get("sources") or {})
        briers: dict[str, float] = {}
        for k, v in srcs.items():
            if isinstance(v, dict) and isinstance(v.get("brier"), (int, float)):
                briers[k] = float(v["brier"])
        if briers:
            worst = max(briers, key=lambda k: briers[k])
            weak.append(f"en kötü kalibre kaynak: {worst} (Brier {briers[worst]:.3f})")
        else:
            weak.append("kaynak-Brier izlemi yok (sources boş) — kalibrasyon körlüğü")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.brain] öz-eleştiri kaynak okuması hatası: {e}")
    if not features:
        weak.append("özellik vektörü boş — hatırlama/öngörü zeminsiz")
    if t.regime is None:
        weak.append("fiyat serisi kısa/yok — rejim ve risk bloğu zayıf zeminde")
    if t.neural_pwin is None:
        weak.append("neural öngörü yok (model verilmedi/eğitilmedi) — ÖNGÖR adımı pasif")
    elif t.neural_pwin[1] >= 0.15:
        weak.append(f"neural belirsizlik yüksek (±{t.neural_pwin[1]:.2f}) — öngörü güveni düşük")
    if not t.semantic and not t.recalls:
        weak.append("hafızada benzer durum yok — karar yalnız prior/rejim zeminine dayanıyor")
    if t.decision is not None and t.decision.mode == "off":
        weak.append("kip=off — gölge kanıt birikmiyor")
    # LAB köprüsü (Tur-17): veri/hafıza zemini zayıfsa somut, YALNIZ-ANALİZ deney önerisi
    # (metin — otomatik çalıştırılmaz; lab.run_experiment insan/ayrı araç tetiklemesiyle).
    if features is None or t.regime is None:
        # Öneri son maddedir AMA 5-madde tavanına kurban gitmez: tetiklendiği durumda
        # (özellik ve/veya rejim zemini yok) üstteki tanılar tavanı zaten doldurur ve öneri
        # hiç GÖRÜNMEZDİ. Tavan tanılara uygulanır; öneri her hâlükârda listede kalır.
        return weak[:4] + ["öneri: nihai.ai.lab.run_experiment('dagilim_testi'/"
                           "'vol_model_karsilastirma') ile veri zeminini raporla (yalnız-analiz)"]
    return weak[:5]


def _ground(strategy: str) -> list[str]:
    """Stratejiyle ilişkili kavram kimlikleri (knowledge.search; hata → boş)."""
    try:
        return [c.id for c in knowledge.search(strategy)][:5]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.brain] kavram topraklama hatası: {e}")
        return []


def _narrative(t: Thought) -> str:
    """Türkçe, adım-adım düşünce zinciri (deterministik; LLM yok)."""
    lines = [f"🧠 Düşünce [{t.symbol} · {t.strategy}] {t.ts}"]
    if t.health is not None:
        tag = "KARANTİNA ⛔" if t.health.get("quarantine") else "geçti"
        lines.append(f"Veri sağlığı: skor={t.health.get('score')} → {tag}"
                     + (f" ({len(t.health.get('issues') or [])} sorun)" if t.health.get("issues") else ""))
    if t.regime:
        lines.append(f"Rejim: {t.regime.get('label', '?')} "
                     f"(trend={t.regime.get('trend')}, vol={t.regime.get('vol_bucket')})")
    rb = [f"{k}={v}" for k, v in t.risk.items() if v is not None]
    if rb:
        lines.append("Risk: " + " · ".join(rb))
    if t.semantic:
        lines.append("Anlamsal hafıza: " + " | ".join(t.semantic))
    if t.recalls:
        lines.append("Hatırlanan: " + " | ".join(f"{s} (uzaklık {d})" for s, d in t.recalls))
    if t.concepts:
        lines.append("Kavramlar: " + ", ".join(t.concepts))
    if t.lesson_warnings:
        lines.append("Ders uyarıları: " + " | ".join(t.lesson_warnings))
    if t.neural_pwin is not None:
        lines.append(f"ÖNGÖR (neural gölge): p_win={t.neural_pwin[0]:.3f} "
                     f"±{t.neural_pwin[1]:.3f} (MC-dropout)")
    if t.conformal is not None:
        lines.append(f"Conformal %90 aralık: [{t.conformal[0]:.3f}, {t.conformal[1]:.3f}]")
    d = t.decision
    if d is not None:
        lines.append(f"Gölge karar: {d.action} (kanaat {d.conviction:.2f}, "
                     f"posterior {d.posterior:.2f}, kelly-önerisi {d.kelly_suggest:.4f}, mod={d.mode})")
        lines.extend("  · " + e for e in d.evidence)
    if t.weak_links:
        lines.append("Öz-eleştiri (zayıf halkalar): " + " | ".join(t.weak_links))
    lines.append(CLOSING)
    return "\n".join(lines)


def think(symbol: str, strategy: str, features: dict | None = None,
          closes: np.ndarray | None = None, p_win: float | None = None,
          side_hint: str | None = None, neural_model: Any | None = None,
          conformal_calib: dict | None = None) -> Thought:
    """Tam düşünme döngüsü (v3 — modül docstring'indeki adımlar). Yazma davranışı
    decision.AI_DECISION_MODE'a tabidir (off → hiçbir dosyaya yazılmaz; shadow → gölge
    karar + epizot + memory digest). `neural_model` DUCK-TİPLİ ve opsiyoneldir
    (NeuralCapacity paritesi: is_fitted/feature_names/predict_with_uncertainty) —
    verilmezse ÖNGÖR adımı pasif kalır ve davranış v2 çağrısına denk düşer.
    `conformal_calib` (WP-X, explain.conformal_calibrate çıktısı) verilirse ve neural
    öngörü varsa kapı-c conformal aralığı decision'a aktarılır; yoksa kapı-c pasif.
    ALGILA başında veri-sağlık kapısı (WP-O): kapanış serisi KARANTİNAYA düşerse karar
    motoru hiç çağrılmaz, zorunlu ABSTAIN üretilir ('veri güvenilmez' gerekçesiyle)."""
    closes_arr = None if closes is None else np.asarray(closes, dtype=float)
    health = _health_gate(closes_arr, symbol)
    regime = None
    if closes_arr is not None:
        try:
            regime = quant_math.classify_regime(closes_arr)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[ai.brain] rejim hesabı hatası: {e}")
    recalls, lesson_warnings = _recall(features, strategy)
    t = Thought(ts=_utc_now_iso(), symbol=symbol, strategy=strategy, regime=regime,
                risk=_risk_block(closes_arr), recalls=recalls,
                lesson_warnings=lesson_warnings, concepts=_ground(strategy),
                neural_pwin=_neural_forecast(neural_model, features),
                semantic=_semantic_lines(features), health=health)
    if health is not None and health.get("quarantine"):
        t.decision = _quarantine_abstain(symbol, strategy, regime, lesson_warnings, health)
    else:
        t.conformal = _conformal_from_calib(t.neural_pwin, conformal_calib)
        t.decision = decision_mod.decide(symbol, strategy, features or {}, p_win=p_win,
                                         closes=closes_arr, side_hint=side_hint,
                                         neural_pwin=t.neural_pwin,
                                         conformal_interval=t.conformal)
    t.weak_links = _self_critique(t, features)
    t.narrative = _narrative(t)
    if t.decision is not None and t.decision.mode == "shadow":
        try:
            memory.append("thought_digest", {
                "symbol": symbol, "strategy": strategy,
                "action": t.decision.action, "conviction": t.decision.conviction,
                "regime": (regime or {}).get("label"),
            })
        except Exception as e:  # noqa: BLE001 — hafıza yazımı düşünmeyi düşürmesin
            logger.debug(f"[ai.brain] thought_digest yazılamadı: {e}")
    return t


def daily_brief() -> str:
    """Günlük beyin brifi: sistem durumu (reasoner facts) + öğrenme + kaynak kalibrasyonu
    (v3) + gölge kararlar. Salt-okunur (hiçbir durum dosyası güncellenmez); kenar iddiası
    içermez."""
    lines = [f"🧠 Quant Beyin — günlük brif ({_utc_now_iso()})"]
    try:
        digest = reasoner.digest_of(reasoner.gather_facts())
        sys_bits = []
        if digest.get("verdict"):
            sys_bits.append(f'verdikt "{digest["verdict"]}"')
        if digest.get("resolved") is not None:
            sys_bits.append(f"çözülen {digest['resolved']}")
        if digest.get("hit_rate") is not None:
            sys_bits.append(f"isabet %{digest['hit_rate'] * 100:.1f}")
        if digest.get("pred_pending_recent") is not None:
            sys_bits.append(f"bekleyen {digest['pred_pending_recent']}")
        lines.append("[Sistem] " + (" · ".join(sys_bits) if sys_bits else "veri yok"))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.brain] sistem durumu okunamadı: {e}")
        lines.append("[Sistem] okunamadı")
    try:
        lines.append(learning.learning_report())
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.brain] öğrenme raporu hatası: {e}")
        lines.append("[Öğrenme] rapor üretilemedi")
    try:
        srcs = (learning.load_state().get("sources") or {})
        bits = [f"{k} Brier={float(v['brier']):.3f} (n={v.get('n', 0)})"
                for k, v in srcs.items()
                if isinstance(v, dict) and isinstance(v.get("brier"), (int, float))]
        if bits:
            lines.append("[Kaynak kalibrasyonu] " + " · ".join(bits))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.brain] kaynak kalibrasyon satırı hatası: {e}")
    try:
        lines.append(decision_mod.decision_report(days=7))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.brain] karar raporu hatası: {e}")
        lines.append("[Gölge kararlar] rapor üretilemedi")
    karne = _last_selfcheck_line()
    if karne:
        lines.append(karne)
    lines.append(CLOSING)
    return "\n".join(lines)


def _last_selfcheck_line() -> str | None:
    """SON öz-değerlendirme karnesinin tek-satır özeti (Tur-17). SALT-OKUNUR: yeni sınav
    KOŞULMAZ (daily_brief'in salt-okunur sözleşmesi bozulmasın). Ayrıştırma karne dosyasının
    ve şemasının SAHİBİ olan `selfcheck`'te yaşar (aynı satırı daily_summary de kullanır);
    burada yalnız fail-soft sarmalayıcı var — hata → None (satır hiç eklenmez)."""
    try:
        from nihai.ai import selfcheck as selfcheck_mod
        return selfcheck_mod.last_card_line()
    except Exception as e:  # noqa: BLE001 — karne satırı brifi asla düşürmesin
        logger.debug(f"[ai.brain] karne satırı okunamadı: {e}")
        return None


def _demo(seed: int = 7) -> str:
    """Sentetik GBM fiyatı + örnek özelliklerle uçtan uca GÜVENLİ koşu: AI_DECISION_MODE=off
    zorlanır → hiçbir dosyaya yazılmaz. Çıktı: düşünce zinciri metni (v3: öz-eleştiri
    bölümü dahil; neural model kasıtlı verilmez — demo hızlı ve torch'suz kalır, ÖNGÖR
    adımının pasifliği zayıf-halka raporunda dürüstçe görünür)."""
    os.environ["AI_DECISION_MODE"] = "off"
    paths = quant_math.mc_gbm_paths(s0=100.0, mu=0.08, sigma=0.25, days=252, n_paths=1, seed=seed)
    closes = paths[0]
    feats = {"ret_5": 0.012, "rsi": 61.0, "vol_20": 0.018, "gap_pct": 0.001, "close_pos": 0.7}
    t = think("DEMO", "event_breakout", features=feats, closes=closes, p_win=0.55,
              side_hint="LONG")
    return t.narrative


def main(argv: list[str] | None = None) -> int:
    """CLI: --demo (sentetik, yazmasız uçtan uca) | --brief (günlük beyin brifi) |
    --ask "soru" (Türkçe soru-cevap, WP-A köprüsü; agent modülü yoksa dürüst hata)."""
    import argparse
    import sys
    try:  # Windows cp1254 konsolunda emoji/Türkçe çıktı güvencesi (O6 UTF-8 deseni)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.brain] stdout UTF-8 reconfigure edilemedi: {e}")
    ap = argparse.ArgumentParser(description="Quant Beyin (KARAR-DIŞI/GÖLGE) orkestratörü")
    ap.add_argument("--demo", action="store_true", help="sentetik veriyle güvenli uçtan uca koşu")
    ap.add_argument("--brief", action="store_true", help="günlük beyin brifi (salt-okunur)")
    ap.add_argument("--ask", type=str, default=None, metavar="SORU",
                    help="beyne Türkçe soru sor (kaynaklı cevap; salt-okunur)")
    args = ap.parse_args(argv)
    if args.demo:
        print(_demo())
        return 0
    if args.brief:
        print(daily_brief())
        return 0
    if args.ask:
        try:
            from nihai.ai import agent as agent_mod
        except Exception as e:  # noqa: BLE001
            print(f"agent modülü yüklenemedi: {e}")
            return 1
        resp = agent_mod.ask(args.ask)
        print(resp.get("answer", ""))
        srcs = resp.get("sources") or []
        if srcs:
            print("Kaynaklar: " + ", ".join(str(s) for s in srcs))
        print(resp.get("closing", CLOSING))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
