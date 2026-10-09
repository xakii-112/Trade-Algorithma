"""
ai_advisor.py — İç AI'ın (`nihai.ai.decision`) CANLI KARAR/EMİR yoluna KÖPRÜSÜ (Tur-38).

Tur-37 "bağlandı ama karara GİRMEDİ" turuydu: gölge kolonlar, pano şeritleri, rapor
ekleri. Eksik kalan tek şey, iç AI'ın verdiği kararın canlı aday sinyale DOKUNABİLECEĞİ
yoldu — `scan_symbol` (tek canlı karar yolu) ile `decision.decide` arasında hiçbir bağ
yoktu. Bu modül o yolu kurar ve ANAHTARI OPERATÖRE VERİR: varsayılan hâlde tek satırı
bile çalışmaz.

KİP — `AI_LIVE_MODE` env, ÇAĞRI ANINDA okunur (`decision.resolve_mode` deseni; testler
ve A/B harness'ları monkeypatch ile kipi değiştirebilsin):

  "off" (VARSAYILAN) → `advise()` None döner. `scan_symbol` hiçbir AI çağrısı yapmaz,
      davranış Tur-37 ile BİT-DENKTİR; `nihai/ai` ANAYASASI (KARAR-DIŞI) yürürlüktedir.
  "advise"           → Karar HESAPLANIR; `sig.reasons`/karar loguna ve oradan Telegram
      "emir" metnine yazılır. Sinyali ne susturur ne boyutlandırır — ÖLÇÜM yolu.
  "enforce"          → ABSTAIN veya sinyalle TERS yön → VETO: uyarı da giriş de durur.
      Veto yoksa posterior pozisyon boyutunu KISABİLİR.

⚠️ NEDEN VARSAYILAN "off" (ACI DERS #2/#4/#5): bu sistemde OOS-kenarlı baz sinyal
kanıtlanmadı ve iç AI'ın prior'ı da aynı canlı loglardan besleniyor. `enforce` bir KENAR
İDDİASI DEĞİL, operatörün bilinçli tercihidir. Açmadan önce `ai.decision.decision_report()`
ve `ai.learning.learning_report()` okunmalı — kapı bu modülle açılmadı, yalnız önüne SIRA
kuruldu (Tur-37 ifadesiyle: sırayı kurduk, kapıyı değil).

TEK YÖNLÜ KISITLAMA — enforce yalnız SUSTURUR ve KÜÇÜLTÜR:
  - Yön ÜRETMEZ. AI "LONG_BIAS" derken sinyal SHORT ise sonuç VETO'dur; sinyalin yönü
    DEĞİŞMEZ ("aleyhte kanıttan ters yöne gir" ACI DERS #3/#7'ye aykırıdır).
  - Boyut BÜYÜTMEZ. `size_mult` tavanı 1.0'dır (`prob_sizing` sözleşmesi).
  - ÇIKIŞ YOLUNA HİÇ BAĞLANMAZ. LONG/SHORT dışı (EXIT/WAIT) sinyalde `advise()` None
    döner: bir AI vetosunun zarar-kesmeyi geciktirmesi bu sistemdeki en pahalı hata olurdu.

FAIL-SOFT: özellik çıkarımı/karar üretimi patlarsa None döner — tarama ve sinyal sağlam
kalır. Sessiz ölüm YOK: neden `status()["last_error"]` ile görünür (Tur-12 dersi).

.env:
    AI_LIVE_MODE=off|advise|enforce   # varsayılan off
    AI_LIVE_SIZE_P_LO=0.50            # posterior→boyut rampasının tabanı
    AI_LIVE_SIZE_P_HI=0.60            # tam risk için gereken posterior
    AI_LIVE_SIZE_FLOOR=0.25           # en düşük boyut çarpanı (0 = sinyali öldürmez)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from loguru import logger

from nihai.config import ACTIVE_STRATEGY
from nihai.risk.prob_sizing import prob_to_size_mult
from nihai.signals.meta_features import extract_features
from nihai.signals.strategy import is_quant_label

_MODE_ADVISE = frozenset({"advise", "advice", "tavsiye", "danisma", "danışma"})
_MODE_ENFORCE = frozenset({"enforce", "veto", "1", "true", "on", "yes"})

# Sayaçlar/son hata — pano ve günlük özet için (kenar iddiası taşımaz).
_n_calls = 0
_n_veto = 0
_last_error: str | None = None
# Tur-92: (kanal, sembol, taraf) → karar defterine en son YAZILAN bar. Tarama turu aynı barı ~6 sn'de bir
# yeniden sorar (advise açılınca QCOM tek barda onlarca satır yazdı); karar her turda HESAPLANIR, defter
# (`ai_decisions.jsonl`) bar başına BİR kez yazılır — satır sayısı ölçüm sayısı olsun (Tur-77 BULGU-77-1).
_son_kayit: dict[tuple[str, str, str], str] = {}
_n_kapsam_disi = 0

# Tur-99 (BULGU-99-1/2): karar motorunun kanıtı (öğrenme durumu GLOBAL/LONG/SHORT) canlı 1m tahmin
# servisinin 5 barlık YÖN isabetidir ve kapısı isabet sorar (lo95 > 0,5). quant_trend adayı günlük trend
# işlemidir (haftalar, izleyen stop) ve kenarı isabette değil kazanç/kayıp asimetrisindedir: motor
# koşularında isabet 30 piyasa×pencere hücresinin 29'unda %33-45, kâr faktörü çoğunda > 1. Ölçüldü (2026-09-26):
# canlı izleme listesindeki 23 quant LONG adayının 23'üne ABSTAIN (kaynak=LONG, 1m sonsal 0,476) → enforce'ta
# 23 veto. Motorun ne kanıtı ne kapısı bu soruya aittir; meta kapının BULGU-97-4 kararıyla aynı: kapsam
# dışı adaya görüş YAZILMAZ, veto VERİLMEZ, yalnız "kapsam dışı" olduğu yazılır.
KAPSAM_DISI = "KAPSAM_DISI_QUANT"
KAPSAM_NEDENI = ("karar motorunun kanıtı 1m/5 bar yön isabeti; quant işlemi günlük trend, kenarı isabette değil "
                 "kazanç/kayıp asimetrisinde")


def kapsam_disi(sig, strategy: str | None = None) -> bool:
    """Aday, karar motorunun kanıtının ait olmadığı bir strateji ailesinden mi? (canlı yol + pano TEK kural)"""
    return is_quant_label((getattr(sig, "metadata", None) or {}).get("strategy")) or is_quant_label(strategy)


def resolve_mode() -> str:
    """`AI_LIVE_MODE` → "off" | "advise" | "enforce". Tanınmayan/boş değer → "off"
    (güvenli varsayılan: yanlış yazılmış bir kip sessizce enforce'a düşemez)."""
    v = (os.getenv("AI_LIVE_MODE") or "").strip().lower()
    if v in _MODE_ADVISE:
        return "advise"
    if v in _MODE_ENFORCE:
        return "enforce"
    return "off"


def _envf(name: str, default: float) -> float:
    """Env'den float oku; bozuk değer varsayılana düşer (canlı taramayı düşürmez)."""
    try:
        return float(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AIAdvice:
    """Tek bir canlı aday sinyal için iç AI'ın görüşü. `veto`/`size_mult` YALNIZ
    enforce kipinde etkilidir; advise kipinde ikisi de nötrdür (False / 1.0)."""

    mode: str            # "advise" | "enforce"
    action: str          # "LONG_BIAS" | "SHORT_BIAS" | "ABSTAIN"
    conviction: float    # 0..1
    posterior: float     # 0..1
    size_mult: float     # (0, 1.0] — asla 1.0'ın ÜSTÜNE çıkmaz
    veto: bool           # True → giriş ve uyarı durur (yalnız enforce)
    reason: str          # tek satır Türkçe özet (sig.reasons + Telegram metni)


def _gunluk_bar(df) -> bool:
    """df günlük (≥ ~1 gün aralıklı) bar mı? Belirlenemezse True (eski davranış: anı yazılır)."""
    try:
        adim = df.index[-30:].to_series().diff().median()
        return bool(adim.total_seconds() >= 20 * 3600)
    except Exception:
        return True


def _expected(action: str) -> str:
    """Aday sinyalin tarafına KARŞILIK GELEN AI verdikti. Başka her şey (ABSTAIN dahil)
    uyuşmazlıktır."""
    return "LONG_BIAS" if action == "LONG" else "SHORT_BIAS"


def _judge(symbol: str, df, sig, strategy: str | None, side: str | None = None, *, kanal: str = "karar"):
    """ORTAK ÇEKİRDEK: aday sinyali karar motoruna sorar; KİPE BAKMAZ. → (karar, taraf)|None

    `advise()` (KARAR yolu) ve `shadow_verdict()` (ÖLÇÜM yolu) aynı sayıyı görsün diye
    tek yerdedir: ikisi ayrı hesaplasaydı `predictions_log`'a yazılan verdikt, canlıda
    UYGULANACAK verdikti ölçmezdi ve biriken kanıt sessizce yanlış soruyu yanıtlardı.

    p_win yeni bir çıkarımla DEĞİL, `strategy.py`'ın zaten yazdığı `meta_p_win`
    metadata'sından okunur (META_GATE shadow/enforce iken dolu) — ikinci bir model
    çağrısı ve ikinci bir "olasılık kaynağı" doğmaz."""
    global _n_calls, _last_error

    # ÇIKIŞ EMRİ AI'A SORULMAZ (modül docstring'i): yalnız yeni GİRİŞ adayları.
    # `side` verilmişse ONA bakılır: META_GATE=enforce aday sinyali WAIT'e çevirir ve
    # sig.action'a bakan bir okuyucu o adayı SESSİZCE atlar → ölçüm seçilim yanlı olur
    # (canlı kanıt döngüsünün `_signal_pwin`'de kırdığı yanlılığın tıpkısı).
    action = (str(side).upper() if side else
              getattr(getattr(sig, "action", None), "value", None))
    if action not in ("LONG", "SHORT"):
        return None

    # GEÇ İMPORT (Tur-26 "lazy nihai.ai" disiplini): `nihai.ai.decision` zinciri
    # (episodic/learning/quant_math + scipy) tarayıcı açılışına ~2,3 sn ekliyordu ve
    # VARSAYILAN kip bunu HİÇ kullanmıyor. Kip "off" iken yukarıda çıkıldığı için bu
    # satıra ancak operatör köprüyü açtığında gelinir.
    from nihai.ai import decision

    anahtar = (kanal, symbol, action)
    try:
        bar = str(df.index[-1]) if df is not None and len(df) else ""
    except Exception:
        bar = ""
    yaz = not bar or _son_kayit.get(anahtar) != bar
    try:
        feats = extract_features(df, sig)
        raw_pwin = (getattr(sig, "metadata", None) or {}).get("meta_p_win")
        closes = df["close"].to_numpy(dtype=float) if df is not None and "close" in df else None
        d = decision.decide(
            symbol,
            strategy or ACTIVE_STRATEGY,
            feats,
            p_win=float(raw_pwin) if raw_pwin is not None else None,
            closes=closes,
            side_hint=action,
            # Tur-90 (BULGU-90-7): epizodik hafıza sonucu 5 GÜNLÜK bardan ölçer; canlı tahmin
            # servisinin 1m kararları (anıların %80'i) o ufka ait değildi. Gün-içi → anı yazılmaz.
            record_episode=_gunluk_bar(df),
            persist=yaz,
        )
    except Exception as e:  # noqa: BLE001 — AI hatası canlı taramayı ASLA düşürmemeli
        _last_error = f"{type(e).__name__}: {e}"
        logger.debug(f"[ai_advisor] {symbol}: karar üretilemedi ({_last_error})")
        return None

    if yaz and bar:
        _son_kayit[anahtar] = bar
    _n_calls += 1
    return d, action


def enforce_boyut(posterior: float) -> float:
    """enforce'ta veto EDİLMEYEN adayın boyut çarpanı (posterior rampası). `advise()` ve gölge
    kasa (Tur-102) TEK kaynaktan okur: kasanın ölçtüğü boyut, canlıda uygulanacak boyutun kendisi."""
    return prob_to_size_mult(
        posterior,
        p_lo=_envf("AI_LIVE_SIZE_P_LO", 0.50),
        p_hi=_envf("AI_LIVE_SIZE_P_HI", 0.60),
        floor=_envf("AI_LIVE_SIZE_FLOOR", 0.25),
    )


def advise(symbol: str, df, sig, *, strategy: str | None = None) -> AIAdvice | None:
    """KARAR yolu: aday sinyal için iç AI görüşü; kip "off" ise (varsayılan) HİÇ çalışmaz."""
    global _n_veto, _n_kapsam_disi

    mode = resolve_mode()
    if mode == "off":
        return None
    if kapsam_disi(sig, strategy) and getattr(getattr(sig, "action", None), "value", None) in ("LONG", "SHORT"):
        _n_kapsam_disi += 1
        return AIAdvice(mode=mode, action=KAPSAM_DISI, conviction=0.0, posterior=float("nan"), size_mult=1.0,
                        veto=False, reason=f"AI_{mode.upper()}: {KAPSAM_DISI} ({KAPSAM_NEDENI})")
    sonuc = _judge(symbol, df, sig, strategy)
    if sonuc is None:
        return None
    d, action = sonuc

    # ABSTAIN de "uyuşmazlık"tır: AI yön için yeterli kanıt bulamadıysa enforce susar.
    uyusmazlik = d.action != _expected(action)

    size_mult = 1.0
    veto = False
    if mode == "enforce":
        veto = uyusmazlik
        if not veto:
            size_mult = enforce_boyut(d.posterior)

    # Gerekçe zincirinin SON satırı her zaman verdikttir ("Karar: ...") — ABSTAIN'in
    # nedeni de orada durur, bu yüzden operatöre giden tek satıra onu koyuyoruz.
    verdikt = d.evidence[-1] if d.evidence else ""
    reason = (f"AI_{mode.upper()}: {d.action} conv={d.conviction:.2f} "
              f"post={d.posterior:.3f}")
    if veto:
        _n_veto += 1
        reason += f" → VETO ({verdikt})"
    elif size_mult < 1.0:
        reason += f" → boyut×{size_mult:.2f}"

    return AIAdvice(mode=mode, action=d.action, conviction=float(d.conviction),
                    posterior=float(d.posterior), size_mult=float(size_mult),
                    veto=veto, reason=reason)


# --- ÖLÇÜM yolu (Tur-38B) — kanıt döngüsü kapı KAPALIYKEN de yaşamalı -------------
_VERDICT_LOG_OFF = ("0", "off", "false", "no")


def shadow_verdict(symbol: str, df, sig, *, strategy: str | None = None,
                   side: str | None = None) -> dict | None:
    """KİP-BAĞIMSIZ verdikt → `{"action","post","veto","side"}`; aday değilse/hata → None.

    `advise()`ten TEK farkı `AI_LIVE_MODE`'a BAKMAMASIDIR: `off` iken de hesaplanır ve
    `predictions_log.csv`'ye yazılır. Neden: "kapı açılmalı mı?" sorusunun tek dürüst
    cevabı, kapı KAPALIYKEN biriken kayıttır (META_GATE shadow dersi — kapı tamamen
    kapatılırsa kanıt döngüsü ölür ve soru bir daha ASLA yanıtlanamaz). `veto` alanı
    "enforce açık OLSAYDI bu sinyal susar mıydı?" sorusunun 0/1 cevabıdır; çözülen
    getiriyle join edilince `gate_sim` tarzı bir kapı simülasyonu mümkün olur.

    KARAR-DIŞI: dönen değer yalnız diske yazılır — hiçbir sinyali/boyutu değiştirmez.
    `AI_VERDICT_LOG=off` ölçümü de kapatır (maliyet: aday başına bir `decide()`)."""
    if (os.getenv("AI_VERDICT_LOG") or "").strip().lower() in _VERDICT_LOG_OFF:
        return None
    sonuc = _judge(symbol, df, sig, strategy, side, kanal="olcum")
    if sonuc is None:
        return None
    d, action = sonuc
    # `side` KAYDIN parçasıdır: getiriyi işaretlemek için taraf ŞART ve META_GATE kapalıyken
    # `pwin_side` kolonu BOŞ olur — tarafı oradan okumaya çalışan bir analiz sıfır satır bulur.
    return {"action": str(d.action),
            "post": round(float(d.posterior), 4),
            "veto": int(d.action != _expected(action)),
            "side": action}


def status() -> dict:
    """Pano/günlük özet için tek bakışta durum (`neural_shadow.status()` deseni)."""
    return {"mode": resolve_mode(), "n_calls": _n_calls, "n_veto": _n_veto,
            "n_kapsam_disi": _n_kapsam_disi, "last_error": _last_error}
