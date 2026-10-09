"""
shadow_bridge.py — GÖLGEDE koşan ölçüm yüzeylerinin ASIL tarama yoluna köprüsü (Tur-60).

SORUN: gözlemevi (`ai/observatory.bar_shadow`), nöral gölge p_win ve kovalama sapması
YALNIZ `execution/live_predict_service` içinde, yani AYRI bir süreçte hesaplanıyordu.
Canlı karar yolu (`execution/main.scan_symbol`) bunların hiçbirini GÖRMÜYORDU: gölge
gerçekten "gölgede", asıl koşucudan ayrı koşuyordu. Bu modül o ayrımı kapatır — aday
sinyal ÜRETEN sürecin kendisi de gölge ölçüleri görür, gerekçesine yazar ve karar
loguna düşürür.

KİP — `SHADOW_BRIDGE` env, ÇAĞRI ANINDA okunur (`ai_advisor.resolve_mode` deseni):

  "advise" (VARSAYILAN) → ölçüler HESAPLANIR, `sig.reasons`/`sig.metadata["shadow"]`e ve
      karar loguna yazılır. Sinyali NE susturur NE boyutlandırır: `veto=False`,
      `size_mult=1.0` YAPISAL olarak sabittir (aşağıdaki tek-yönlü kısıt).
  "off"              → tek satır bile çalışmaz; davranış Tur-59 ile BİT-DENK.
  "enforce"          → terfi etmiş kurallar KARARA girer. **Bugün terfi etmiş kural
      YOKTUR**; bu kipte köprü `advise` gibi davranır ve nedenini bir kez UYARI olarak
      loglar (sessizce enforce sanılmasın).

⚠️ NEDEN VARSAYILAN "advise" (ve neden "enforce" DEĞİL): `research/shadow_promote.py`
canlı logda on yüzeyi ölçtü ve HİÇBİRİ kanıt kapısını geçemedi — ikisi kapıda ELENDİ
(meta-kapı p_win TERS sıralıyor; gölge yön uyuşması bir kayma artefaktı), beşi örneklem
YETERSİZLİĞİNDEN hükümsüz kaldı. Ve o yetersizliğin nedeni yöntem değil KAPSAMA:
gözlemevi kolonları iki gün, makas bir gün kanıt taşıyor. `advise` tam olarak bu
darboğazı hedefler: yüzeyler artık asıl süreçte, GERÇEK aday sinyaller üzerinde koşar ve
görünür olur. Kararı değiştirmediği için bir KENAR İDDİASI taşımaz (ACI DERS #3/#5).

TEK YÖNLÜ KISIT — köprü yalnız SUSTURABİLİR ve KÜÇÜLTEBİLİR (enforce açılırsa):
  - Yön ÜRETMEZ, yön ÇEVİRMEZ ("aleyhte kanıttan ters yöne gir" ACI DERS #3/#7'ye aykırı).
  - Boyut BÜYÜTMEZ (`size_mult` tavanı 1.0 — `prob_sizing`/`ai_advisor` sözleşmesi).
  - ÇIKIŞ YOLUNA HİÇ BAĞLANMAZ: LONG/SHORT dışı sinyalde `evaluate()` None döner. Bir
    gölge ölçünün zarar-kesmeyi geciktirmesi bu sistemdeki en pahalı hata olurdu.

ÖLÇÜM AYRIMI: buradan yazılan ölçüler `predictions_log`a GİRMEZ. O defterin gölge
kolonları 1m tahmin yüzeyinde birikti; tarama yolu HTF barıyla çalışır ve ikisini aynı
havuza dökmek `shadow_promote`ın ölçtüğü dağılımı KİRLETİRDİ. Buradaki ölçüler sinyalin
gerekçesine, karar loguna ve `status()`e gider.

FAIL-SOFT: her yüzey kendi try/except'indedir; hepsi patlasa `evaluate()` yine bir
verdikt döner ve tarama SAĞLAM kalır. Sessiz ölüm YOK — neden `status()["last_error"]`
ile görünür (Tur-12/Tur-44 dersi).

.env:
    SHADOW_BRIDGE=advise|off|enforce   # varsayılan advise
    SHADOW_BRIDGE_NEURAL=0             # 1 → nöral gölge p_win de hesaplanır (torch!)

MALİYET (ölçüldü, gerçek AAPL 1h barları): çağrı başına **5,36 ms** (HMM önbellekli),
sembolün İLK adayında 65,7 ms (HMM fit). Yalnız LONG/SHORT adaylarda çalışır — bar
başına her sembol için değil. `AI_OBS_HMM_REFIT_BARS` (60) yeniden-fit sıklığını verir.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from loguru import logger

from nihai.signals.strategy import SignalAction

_MODE_ENFORCE = frozenset({"enforce", "veto", "zorla"})
_MODE_OFF = frozenset({"off", "0", "false", "no", "kapali", "kapalı"})

# ---------------------------------------------------------------------------
# TERFİ DEFTERİ — `research/shadow_promote.py`'ın 2026-08-31 ölçümü (canlı log,
# 63.909 çözülmüş satır). Bir kural KARARA ancak `PROMOTE` iken girebilir; başka her
# etiket `enforce` kipinde bile YALNIZ raporlanır.
#
# Defter neden KODDA ve neden `shadow_promote.json`dan OKUNMUYOR: canlı bir kapı,
# herhangi bir koşunun üzerine yazabildiği bir veri dosyasından kural almamalıdır.
# Ölçüm her gün tazelenir (günlük özet onu basar); DEFTERİ değiştirmek bilinçli bir
# tur işidir ve bu satırların düzenlenmesini gerektirir.
# ---------------------------------------------------------------------------
RULES: dict[str, tuple[str, str]] = {
    "meta_pwin": ("INVERTED", "canlıda TERS sıralıyor (t=3,31 'düşük tut' yönünde); "
                              "tersine çevirmek ACI DERS #3/#7 yasağı"),
    "dir_bar_agree": ("SLIP_ARTIFACT", "uplift ortak referansta işaret değiştiriyor "
                                       "(+0,0065 → −0,0015pp): giriş fiyatı ofseti"),
    "chase_drift": ("SLIP_ARTIFACT", "aynı kimliğin sapma-eşikli hâli (ret_live ≈ "
                                     "ret_bar − slip); kanıt DEĞİL"),
    "neural_pwin": ("HOLD", "+0,1233pp ama tutulan kova n=37, p_adj=1,0"),
    "conf_bar": ("HOLD", "+0,0137pp, p_adj=0,835"),
    "mkt_health": ("HOLD", "+0,0135pp, p_adj=1,0 (yalnız 2 gün kanıt)"),
    "har_rv_fc": ("INSUFFICIENT", "2 gün kanıt; hiçbir eşik örneklem kapısını geçmedi"),
    "spread_bps": ("INSUFFICIENT", "29 çözülmüş satır"),
    "hmm_mode_prob": ("INSUFFICIENT", "2 gün kanıt"),
    "hmm_switch": ("INSUFFICIENT", "2 gün kanıt"),
    "ai_veto": ("INSUFFICIENT", "karar motoru %100 ABSTAIN — 'geçen' kovası boş"),
}

PROMOTED = tuple(sorted(k for k, (lbl, _) in RULES.items() if lbl == "PROMOTE"))

_n_calls = 0
_n_marks = 0
_last_error: str | None = None
_enforce_uyarildi = False


def resolve_mode() -> str:
    """`SHADOW_BRIDGE` → "off" | "advise" | "enforce". Tanınmayan değer → "advise".

    Varsayılan `advise`tir çünkü bu kip KARARI DEĞİŞTİREMEZ (yapısal olarak
    `veto=False`/`size_mult=1.0`); yanlış yazılmış bir kip enforce'a DÜŞEMEZ."""
    v = (os.getenv("SHADOW_BRIDGE") or "").strip().lower()
    if v in _MODE_OFF:
        return "off"
    if v in _MODE_ENFORCE:
        return "enforce"
    return "advise"


def _envb(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ShadowVerdict:
    """Bir canlı aday sinyal için gölge ölçüler + (varsa) terfi etmiş kuralın hükmü."""

    mode: str                 # "advise" | "enforce"
    veto: bool                # True → giriş ve uyarı durur (YALNIZ terfi etmiş kural)
    size_mult: float          # (0, 1.0] — asla 1.0'ın ÜSTÜNE çıkmaz
    marks: dict[str, Any]     # ölçülen gölge değerler (kenar iddiası taşımaz)
    reason: str               # tek satır Türkçe özet (sig.reasons + karar logu)


def _observatory(df, symbol: str) -> dict[str, Any]:
    """Gözlemevi gölge paketi — mikroyapı + NEDENSEL rejim HMM + veri sağlığı.

    Geç import: `nihai.ai` zinciri (scipy) yalnız gerçekten aday varken ödenir.
    `AI_OBSERVATORY=off` ile kapatılabilir (modülün kendi anahtarı)."""
    try:
        from nihai.ai.observatory import bar_shadow
        out = bar_shadow(df, symbol)
    except Exception as exc:
        _note_error(exc, "observatory")
        return {}
    if not out:
        return {}
    # Yalnız SAYIYA dönüşen alanlar taşınır; `hmm_label` metindir ve `hmm_mode_prob`
    # OLMADAN yazılmaz (Tur-33: bir ORAN, paydası okunmadan okunamaz).
    keep: dict[str, Any] = {}
    for k in ("health", "har_rv_fc", "spread_bps", "hmm_mode_prob", "hmm_switch_prob"):
        v = out.get(k)
        if v is None:
            continue
        # `har_rv_fc` bar-başına VARYANSTIR (~1e-3) ve ham float gerekçe satırını 20
        # karakter uzatır; 6 anlamlı basamak ölçünün mertebesini korur (Telegram/karar
        # logu tek satır okunur). Diğerleri zaten observatory'de yuvarlanmış gelir.
        keep[k] = round(float(v), 6) if k == "har_rv_fc" else v
    if "hmm_mode_prob" in keep and out.get("hmm_label"):
        keep["hmm_label"] = out["hmm_label"]
    if out.get("quarantine"):
        keep["quarantine"] = True
    return keep


def _chase_drift_bps(sig, live_price: float | None) -> float | None:
    """Girişin sinyal fiyatından İŞLEM YÖNÜNDE sapması (bps). Ölçülemezse None.

    `prediction_tracker`ın çözümde yazdığı `slip_bps` ile AYNI tanımdır — orada
    (canlı fiyat / bar kapanışı − 1)·1e4·yön, burada karar anındaki hâli. Kovalama
    maliyetinin TEK ölçüsüdür ve `RULES["chase_drift"]` onun terfi ETMEDİĞİNİ taşır:
    bu sayı `fwd_ret_pct`in paydasına GİRDİĞİ için ona kurulan kapı bilgi değil
    aritmetik ölçer (Tur-60 BULGU-60-2)."""
    try:
        base = float(sig.price)
        px = float(live_price or 0.0)
    except (TypeError, ValueError):
        return None
    if base <= 0 or px <= 0:
        return None
    sgn = 1.0 if sig.action == SignalAction.LONG else -1.0
    return round((px / base - 1.0) * 1e4 * sgn, 2)


def _neural_pwin(df, sig, symbol: str) -> dict[str, Any]:
    """Nöral gölge p_win — YALNIZ `SHADOW_BRIDGE_NEURAL=1` iken.

    Ayrı bayrak ZORUNLU: bu çağrı canlı TARAYICI sürecinde `torch` importunu tetikler
    (Tur-26 import hijyeni; Tur-38 bu yüzden nöral gölgeyi bilerek bağlamamıştı). Ölçüm
    HOLD etiketlidir (+0,1233pp ama n=37) — yani açmanın gerekçesi kenar değil KANIT
    BİRİKTİRMEKTİR ve bedeli operatörün bilinçli tercihidir."""
    if not _envb("SHADOW_BRIDGE_NEURAL", False):
        return {}
    try:
        from nihai.execution.neural_shadow import get_shadow
        from nihai.signals.meta_features import extract_features
        side = str(sig.action.value)
        feats = dict(extract_features(df, sig))
        feats["is_long"] = 1.0 if side == "LONG" else 0.0
        res = get_shadow().pwin(feats, side)
    except Exception as exc:
        _note_error(exc, "neural")
        return {}
    if res is None:
        return {}
    return {"neural_pwin": round(float(res[0]), 4), "neural_std": round(float(res[1]), 4)}


def _note_error(exc: Exception, nerede: str) -> None:
    """İlk hatayı UYARI, gerisini debug olarak logla; son hata `status()`te durur.

    Zarif degrade doğru tasarımdır, SESSİZ degrade değildir (Tur-44 BULGU-44)."""
    global _last_error
    ilk = _last_error is None
    _last_error = f"{nerede}: {type(exc).__name__}: {exc}"[:160]
    (logger.warning if ilk else logger.debug)(f"[shadow_bridge] {_last_error}")


def _enforce_hukmu(marks: dict[str, Any]) -> tuple[bool, float, list[str]]:
    """Terfi etmiş kuralları uygula. Bugün PROMOTED boş → (False, 1.0, []).

    Kural burada BİR KEZ yazılır: `RULES` defterinde `PROMOTE` olmayan hiçbir yüzey
    karara giremez. Böylece "ölçüm ne diyor?" sorusu ile "canlı ne yapıyor?" sorusu
    ayrışamaz — Tur-51'in *kanıtsız bileşen skordan çıkarılır* ilkesinin kapı hâli."""
    global _enforce_uyarildi
    if not PROMOTED:
        if not _enforce_uyarildi:
            _enforce_uyarildi = True
            logger.warning(
                "[shadow_bridge] enforce istendi ama TERFİ ETMİŞ kural YOK "
                "(shadow_promote 2026-08-31: 2 aday kapıda elendi, 5'i örneklem yetersiz) "
                "→ köprü advise gibi davranıyor; hiçbir sinyal bastırılmayacak.")
        return False, 1.0, []
    # Terfi eden ilk kural buraya, KENDİ dalıyla ve kendi testiyle bağlanır.
    # Ölçülmemiş bir kuralı buraya eklemek defterin varlık nedenini yok eder.
    raise AssertionError(f"terfi defteri PROMOTE taşıyor ama uygulaması yok: {PROMOTED}")


def evaluate(symbol: str, df, sig, *, live_price: float | None = None) -> ShadowVerdict | None:
    """Aday sinyal için gölge ölçüleri üret (+ terfi etmişse hüküm). Kip `off` → None.

    YALNIZ yeni GİRİŞ adaylarında çalışır: LONG/SHORT dışı sinyalde None döner
    (çıkış yoluna bağlanmama sözleşmesi). Hiçbir aşama istisna sızdırmaz."""
    global _n_calls, _n_marks
    mode = resolve_mode()
    if mode == "off":
        return None
    if getattr(sig, "action", None) not in (SignalAction.LONG, SignalAction.SHORT):
        return None

    _n_calls += 1
    marks: dict[str, Any] = {}
    try:
        marks.update(_observatory(df, symbol))
        drift = _chase_drift_bps(sig, live_price)
        if drift is not None:
            marks["chase_drift_bps"] = drift
        marks.update(_neural_pwin(df, sig, symbol))
    except Exception as exc:            # savunma: yukarıdakiler kendi try'ında ama
        _note_error(exc, "evaluate")    # yeni bir yüzey eklenirse tarama düşmesin

    veto, size_mult = False, 1.0
    gerekceler: list[str] = []
    if mode == "enforce":
        veto, size_mult, gerekceler = _enforce_hukmu(marks)

    if marks:
        _n_marks += 1
        ozet = " ".join(f"{k}={v}" for k, v in sorted(marks.items()))
    else:
        ozet = "ölçüm yok"
    # Etiket ölçümün KİPİNİ taşır: gerekçeyi okuyan (Telegram metni dahil) bunun bir
    # sinyal değil GÖZLEM olduğunu görmeli.
    reason = f"SHADOW[{mode}] {ozet}" + (" | " + " ".join(gerekceler) if gerekceler else "")
    return ShadowVerdict(mode=mode, veto=veto, size_mult=size_mult, marks=marks,
                         reason=reason)


def status() -> dict[str, Any]:
    """Pano/günlük özet için tek bakışta durum (`ai_advisor.status()` deseni)."""
    return {"mode": resolve_mode(), "calls": _n_calls, "with_marks": _n_marks,
            "promoted": list(PROMOTED), "neural": _envb("SHADOW_BRIDGE_NEURAL", False),
            "last_error": _last_error,
            "rules": {k: lbl for k, (lbl, _) in RULES.items()}}
