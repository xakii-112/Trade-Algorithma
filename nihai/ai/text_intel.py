"""
text_intel.py — METİN/HABER ZEKÂSI + TAKVİM FARKINDALIĞI (AI v3 WP-T, Tur-16). KARAR-DIŞI/GÖLGE.

Amaç: sayısal-olmayan bilgiyi (haber, kazanç metni, makro takvim) yapılandırılmış sinyale
çevirmek — İKİ KADEMELİ, LLM'siz de TAM çalışır (docs/AI_V3_DERIN_BEYIN_SPEC.md §3.13):

  1. Deterministik taban (HER ZAMAN çalışır): `tone_score` (Loughran-McDonald finans
     sözlüğü — modüle GÖMÜLÜ çekirdek alt-küme; kaynak/gerekçe aşağıda), `novelty_score`
     (BoW kosinüs — "aynı haberin 3. tekrarı" ≠ yeni bilgi), `headline_density` (sembol-başına
     haber-sayısı z-skoru).
  2. LLM katmanı (router üstünden, ANTHROPIC_API_KEY/GEMINI_API_KEY varsa): kazanç çağrısı/
     haber özeti → yapılandırılmış (ton/yön/belirsizlik/gerekçe) çıktı. DÜRÜSTLÜK GUARDI:
     LLM çıktısı ASLA tek başına özellik OLMAZ — `analyze_text` çıktısında deterministik
     skorla HER ZAMAN YAN YANA raporlanır (LLM None dönse de deterministik alanlar dolu kalır).

Loughran-McDonald notu (spec §3.13): genel-amaçlı sentiment sözlükleri (Harvard-IV/General
Inquirer gibi) finans metninde SİSTEMATİK YANILIR — ör. "liability", "tax", "cost", "cancer"
gibi nötr-finansal/tıbbi terimler genel sözlüklerde "negatif" sayılır ama 10-K/kazanç
metninde çoğu zaman nötr bağlamdadır. Loughran & McDonald (2011), "When Is a Liability Not
a Liability? Textual Analysis, Dictionaries, and 10-Ks," Journal of Finance 66(1), 35-65,
finans-özel bir sözlük inşa ederek bu sorunu çözdü ve alan-standardı haline geldi. Burada
GÖMÜLÜ olan `_LM_*` kümeleri TEMSİLİ bir ÇEKİRDEK ALT-KÜMEDİR (tam LM Master Dictionary'de
binlerce kelime var — özellikle Negative kategorisi ~2300+ kelime) — hafif/bağımlılıksız
gömülü skorlama için seçildi; üretim-kalite/kapsamlı analiz için tam listenin (LM'in resmi
sitesinden CSV) entegre edilmesi ayrı bir iyileştirme maddesidir.

Takvim farkındalığı (`market_calendar` alt-API'si): FOMC/CPI tarihleri STATİK, ELLE girilen
2026 tablolarıdır (periyodik doğrulama/güncelleme gerekir — aşağıda ayrı not); NFP (Non-Farm
Payrolls) ise KURAL-TABANLI türetilir ("ayın ilk Cuma günü" — BLS'in istikrarlı konvansiyonu,
kendiliğinden güncel kalır). `earnings_proximity` mevcut nasdaq-earnings köprüsünü
(`nihai.data.event_calendar`, Tur-14 P0'da eklendi) LAZY import + try/except ile kullanır;
köprü yok/hata verirse fail-soft `None`.

Hepsi tz-GÜVENLİ UTC: `date`/`datetime` girdileri `_to_utc_date` ile normalize edilir (naive
datetime zaten-UTC kabul edilir — episodic._parse_ts ile AYNI konvansiyon, OP-1 dersi).
Skor kayıtları `data/ai_text_scores.jsonl`'e ATOMİK (lock+append — episodic.py/decision.py
ile AYNI desen; büyüyen JSONL log için bu kod tabanının sanksiyonladığı "atomik" yöntemdir).
"""
from __future__ import annotations

import json
import math
import re
import threading
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

from loguru import logger

from nihai.ai import router
from nihai.config import DATA_DIR

# Modül-düzeyi yol: testler monkeypatch'ler (episodic.py/decision.py deseni).
TEXT_SCORES_FILE = DATA_DIR / "ai_text_scores.jsonl"
_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Loughran-McDonald ÇEKİRDEK ALT-KÜME (bkz. modül docstring'i — kaynak + gerekçe).
# Dört kategori KESİŞMEZ (LM Master Dictionary'nin gerçek kategorileri de birbirini
# dışlar): bir kelime yalnız TEK kümede yer alır.
# ---------------------------------------------------------------------------
_LM_POSITIVE: frozenset[str] = frozenset({
    "achieve", "achieved", "achieving", "beneficial", "benefit", "benefits", "boom",
    "delight", "delighted", "distinguished", "effective", "efficient", "efficiently",
    "enhance", "enhanced", "enhancement", "excellent", "exceptional", "favorable",
    "favorably", "gain", "gains", "good", "great", "greater", "growth", "improve",
    "improved", "improvement", "improving", "innovative", "leadership", "opportunities",
    "opportunity", "outperform", "positive", "profitable", "progress", "robust", "strong",
    "stronger", "strength", "succeed", "success", "successful", "successfully", "upturn",
    "valuable", "win", "winning",
})
_LM_NEGATIVE: frozenset[str] = frozenset({
    "adverse", "adversely", "bankrupt", "bankruptcy", "closure", "closures", "concern",
    "concerns", "crisis", "decline", "declined", "declines", "declining", "default",
    "defaults", "deficiency", "deficit", "delay", "delayed", "deteriorate", "deteriorated",
    "deterioration", "difficult", "difficulty", "disappointing", "discontinued",
    "disruption", "downgrade", "downturn", "failure", "failed", "fail", "fails", "fraud",
    "fraudulent", "harm", "harmful", "hurt", "impair", "impairment", "inability",
    "inadequate", "ineffective", "loss", "losses", "misconduct", "mislead", "misleading",
    "negative", "negatively", "penalty", "poor", "problem", "problems", "recession",
    "restructuring", "severe", "severely", "shortfall", "slowdown", "slump", "terminate",
    "termination", "threat", "unable", "unfavorable", "weak", "weakness", "weaker",
    "writedown",
})
_LM_UNCERTAINTY: frozenset[str] = frozenset({
    "uncertain", "uncertainty", "uncertainties", "approximate", "approximately",
    "appears", "contingency", "contingent", "depend", "depends", "dependent",
    "fluctuate", "fluctuation", "fluctuations", "indefinite", "likelihood", "may",
    "might", "anticipate", "anticipated", "exposure", "possibility", "possible",
    "possibly", "predict", "preliminary", "presumably", "unclear", "unknown",
    "variability", "vary", "volatile", "volatility",
})
_LM_LITIGIOUS: frozenset[str] = frozenset({
    "alleged", "allegation", "allegations", "arbitration", "attorney", "attorneys",
    "breach", "claimant", "claim", "claims", "complaint", "counsel", "court", "courts",
    "defendant", "defendants", "infringement", "injunction", "judgment", "judicial",
    "jurisdiction", "lawsuit", "lawsuits", "litigant", "litigate", "litigation",
    "plaintiff", "plaintiffs", "prosecute", "prosecution", "regulatory", "settlement",
    "sue", "sued", "sues", "suit", "suits", "tribunal", "unlawful", "violate",
    "violation", "violations", "whistleblower",
})

# net_tone bu eşiğin dışındaysa etiket pozitif/negatif; içindeyse nötr (gürültü tamponu).
_NET_TONE_NEUTRAL_BAND = 0.005

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    """Küçük-harf kelime tokenizasyonu (yalnız ASCII harf/kesme işareti dizileri —
    LM sözlüğü İngilizce metin içindir). Boş/str-olmayan girdi → []."""
    s = text if isinstance(text, str) else ""
    return _WORD_RE.findall(s.lower())


# ---------------------------------------------------------------------------
# Deterministik taban — 1) ton skoru
# ---------------------------------------------------------------------------
def tone_score(text: str) -> dict:
    """Loughran-McDonald tarzı finans ton skoru (bkz. modül docstring'i): pozitif/negatif/
    belirsizlik/dava-riski kelime sınıf SAYIMLARI + net skor `(pos-neg)/toplam_kelime`.

    Dönüş: `{'positive','negative','uncertainty','litigious','total_words','net_tone',
    'tone_label'}`. Boş/str-olmayan metin veya sıfır kelime → tüm sayaçlar 0, `net_tone=0.0`,
    `tone_label='nötr'` (savunmacı — asla None/istisna; SPEC bölüm 0 madde 7 desenine göre
    bu fonksiyonun dönüş tipi Optional DEĞİL, bu yüzden zararsız-değere düşer)."""
    words = _tokenize(text)
    total = len(words)
    pos = sum(1 for w in words if w in _LM_POSITIVE)
    neg = sum(1 for w in words if w in _LM_NEGATIVE)
    unc = sum(1 for w in words if w in _LM_UNCERTAINTY)
    lit = sum(1 for w in words if w in _LM_LITIGIOUS)
    net = ((pos - neg) / total) if total else 0.0
    if net > _NET_TONE_NEUTRAL_BAND:
        label = "pozitif"
    elif net < -_NET_TONE_NEUTRAL_BAND:
        label = "negatif"
    else:
        label = "nötr"
    return {"positive": pos, "negative": neg, "uncertainty": unc, "litigious": lit,
            "total_words": total, "net_tone": round(net, 6), "tone_label": label}


# ---------------------------------------------------------------------------
# Deterministik taban — 2) yenilik skoru (BoW kosinüs)
# ---------------------------------------------------------------------------
def _bow_vector(tokens: list[str]) -> dict[str, int]:
    vec: dict[str, int] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0) + 1
    return vec


def _cosine_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    """[0,1] kosinüs BENZERLİĞİ (kelime SAYIM vektörleri negatif olmadığından benzerlik
    doğal olarak [0,1] aralığındadır). Boş/sıfır-norm vektör → 0.0 (savunmacı)."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def novelty_score(text: str, prior_texts: list[str]) -> float:
    """BoW kosinüs YENİLİK skoru: `1 − max(kosinüs_benzerliği(text, önceki))` — "aynı
    haberin 3. tekrarı ≠ yeni bilgi" (spec §3.13): metin ÖNCEKİ metinlerden HERHANGİ birine
    çok benziyorsa (max benzerlik yüksek) yenilik DÜŞÜK olur.

    Dönüş [0,1]: 1.0 = önceki metinlerin HİÇBİRİNE benzemiyor (tamamen yeni İÇERİK — ama
    `prior_texts` BOŞSA da aynı değer döner, çünkü kıyaslanacak bir şey yok; bu KASITLI bir
    savunmacı-varsayım, "kanıt yok" ile "gerçekten yeni" farklı şeylerdir, çağıran bunu
    `prior_texts` uzunluğuyla ayırt edebilir). `text` boş/str-değilse → 0.0 (içeriksiz metin
    yeni BİLGİ taşımaz varsayımı)."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    priors = [p for p in (prior_texts or []) if isinstance(p, str) and p.strip()]
    if not priors:
        return 1.0
    vec = _bow_vector(tokens)
    max_sim = max(_cosine_similarity(vec, _bow_vector(_tokenize(p))) for p in priors)
    return round(1.0 - max_sim, 6)


# ---------------------------------------------------------------------------
# Deterministik taban — 3) başlık yoğunluğu (z-skor)
# ---------------------------------------------------------------------------
def headline_density(counts_by_symbol: dict[str, int]) -> dict:
    """Sembol-başına haber-sayısının popülasyon z-skoru: `(n_sembol − ortalama) / std`
    (episodic.py/consolidate.py'deki `_std` ile AYNI formül — ddof=0). Std=0 (tüm semboller
    eşit sayıda) → tüm z-skorlar 0.0 (bölme-sıfır koruması, savunmacı). Boş girdi → `{}`;
    sayısal-olmayan sayım değeri → o sembol için 0.0 (diğerleri etkilenmez)."""
    if not counts_by_symbol:
        return {}
    vals: list[float] = []
    for v in counts_by_symbol.values():
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return dict.fromkeys(counts_by_symbol, 0.0)
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    sd = math.sqrt(var)
    out: dict[str, float] = {}
    for sym, v in counts_by_symbol.items():
        try:
            vf = float(v)
        except (TypeError, ValueError):
            out[str(sym)] = 0.0
            continue
        out[str(sym)] = round((vf - m) / sd, 6) if sd > 1e-12 else 0.0
    return out


# ---------------------------------------------------------------------------
# LLM katmanı (router üstünden fail-soft) + birleşik analiz + kalıcılık
# ---------------------------------------------------------------------------
_LLM_FIELD_MAP = {"TON": "tone", "YÖN": "direction", "YON": "direction",
                   "BELIRSIZLIK": "uncertainty", "GEREKCE": "rationale", "GEREKÇE": "rationale"}


def _parse_llm_fields(out: str) -> dict | None:
    """LLM çıktısındaki `ETIKET: değer` satırlarını ayrıştırır (best-effort; şema uyulmazsa
    kısmi/boş sonuç). Hiçbir tanınan alan yoksa → None (çağıran fail-soft davranır)."""
    fields: dict[str, str] = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        key = _LLM_FIELD_MAP.get(k.strip().upper())
        val = v.strip()
        if key and val:
            fields[key] = val
    return fields or None


def _llm_summary(text: str, kind: str) -> dict | None:
    """router.complete üstünden YAPILANDIRILMIŞ özet (ton/yön/belirsizlik/gerekçe).
    Anahtar yok / router hatası / boş çıktı / ayrıştırılamaz çıktı → None (fail-soft;
    ÇAĞIRAN `analyze_text` deterministik alanları HER ZAMAN ayrı doldurur — dürüstlük
    guard'ı, spec §3.13: 'LLM çıktısı ASLA tek başına özellik olmaz')."""
    prompt = (
        f"Aşağıdaki finans/{kind} metnini analiz et. YALNIZ şu 4 satır Türkçe formatta "
        "cevap ver, başka hiçbir açıklama EKLEME:\n"
        "TON: <pozitif|negatif|nötr>\n"
        "YÖN: <olumlu|olumsuz|belirsiz>\n"
        "BELIRSIZLIK: <düşük|orta|yüksek>\n"
        "GEREKCE: <tek cümle Türkçe gerekçe>\n\n" + text[:4000]
    )
    try:
        out = router.complete("summary", prompt)
    except Exception as e:  # noqa: BLE001 — LLM hiçbir koşulda analizi düşürmesin
        logger.debug(f"[ai.text_intel] router hatası: {e}")
        return None
    if not out:
        return None
    parsed = _parse_llm_fields(out)
    if parsed is None:
        return None
    parsed["raw"] = out.strip()
    return parsed


def _append_score(rec: dict) -> None:
    """`TEXT_SCORES_FILE`'a tek satır JSONL ekle (kilitli append — episodic.py/decision.py
    ile AYNI 'atomik büyüyen log' deseni; bkz. modül docstring'i). Hata → sessiz-degrade+log
    (skor zaten çağırana döndürülmüştür, kalıcılık ikincil)."""
    line = json.dumps(rec, ensure_ascii=False)
    with _LOCK:
        try:
            TEXT_SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with TEXT_SCORES_FILE.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            logger.warning(f"[ai.text_intel] {TEXT_SCORES_FILE.name} yazılamadı: {e}")


def analyze_text(text: str, *, prior_texts: list[str] | None = None,
                  symbol: str | None = None, kind: str = "haber",
                  persist: bool = True) -> dict:
    """Bir metnin TAM analizi: deterministik taban (`tone_score`+`novelty_score`) HER ZAMAN
    hesaplanır; LLM katmanı (varsa) YAN YANA eklenir (asla tek başına özellik olmaz —
    dürüstlük guard'ı, test edilir). Sonuç `persist=True` (varsayılan) iken
    `TEXT_SCORES_FILE`'a atomik eklenir (gölge kayıt).

    Dönüş: `{'ts','symbol','kind','text_preview','deterministic':{'tone','novelty'},
    'llm': {...} | None}`. Boş/str-olmayan `text` → deterministik alanlar sıfır/nötr
    değerlerle dolu döner, LLM adımı hiç ÇAĞRILMAZ (boş metin için LLM'e gitmek anlamsız —
    maliyet tasarrufu), yine de kayıt YAZILIR (gölge iz sürekliliği için)."""
    text = text if isinstance(text, str) else ""
    det = {"tone": tone_score(text), "novelty": novelty_score(text, prior_texts or [])}
    llm = _llm_summary(text, kind) if text.strip() else None
    rec = {
        "ts": _utc_now_iso(),
        "symbol": (str(symbol).upper() if symbol else None),
        "kind": str(kind),
        "text_preview": text[:200],
        "deterministic": det,
        "llm": llm,
    }
    if persist:
        _append_score(rec)
    return rec


# ---------------------------------------------------------------------------
# market_calendar alt-API'si (spec §3.13 madde 3) — tz-güvenli UTC
# ---------------------------------------------------------------------------
def _to_utc_date(d: date | datetime) -> date:
    """`date`/`datetime` → tz-güvenli takvim günü. tz-aware `datetime` UTC'ye çevrilip
    `.date()`; NAIVE `datetime` zaten-UTC kabul edilir (episodic._parse_ts/decision.py ile
    AYNI proje-çapında konvansiyon — bu modülde 'forming bar' riski yok, yalnız takvim-günü
    karşılaştırması yapılır, OP-1 dersi ruhuna aykırı değil)."""
    if isinstance(d, datetime):
        dt = d if d.tzinfo is not None else d.replace(tzinfo=UTC)
        return dt.astimezone(UTC).date()
    return d


# FOMC 2026 toplantı KARAR günleri (2 günlük toplantının SON/duyuru günü — Fed konvansiyonu).
# ELLE girildi (federalreserve.gov yayınlanan takvimine göre); PERİYODİK DOĞRULAMA/GÜNCELLEME
# GEREKİR — bu KARAR-DIŞI/GÖLGE bağlamda kaba "bugün hangi olay rejimindeyiz" farkındalığı
# içindir, kesinlik iddiası taşımaz (spec §3.13: "statik tablo + elle güncelleme notu").
_FOMC_2026: tuple[date, ...] = (
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
)
# CPI (TÜFE) yayın tarihleri — BLS'in TİPİK takvim penceresi (ayın 2. haftası civarı);
# YAKLAŞIKTIR, gerçek BLS yıllık takvimine karşı periyodik doğrulanmalı (aynı elle-güncelleme
# notu). Kaynak: bls.gov yayın takvimi.
_CPI_2026: tuple[date, ...] = (
    date(2026, 1, 13), date(2026, 2, 11), date(2026, 3, 11), date(2026, 4, 10),
    date(2026, 5, 12), date(2026, 6, 10), date(2026, 7, 14), date(2026, 8, 12),
    date(2026, 9, 11), date(2026, 10, 13), date(2026, 11, 12), date(2026, 12, 10),
)


def _first_friday(year: int, month: int) -> date:
    """Ayın İLK Cuma günü — NFP (Non-Farm Payrolls / Employment Situation) raporunun
    BLS'in istikrarlı yayın kuralı (nadir istisnalar hariç); statik tablo yerine KURAL ile
    türetildiği için kendiliğinden güncel kalır (elle güncelleme GEREKMEZ)."""
    d = date(year, month, 1)
    offset = (4 - d.weekday()) % 7  # Monday=0 .. Sunday=6; Cuma=4
    return d + timedelta(days=offset)


def next_macro_events(as_of: date | datetime, n: int = 3) -> list[dict]:
    """`as_of`'tan (dahil) itibaren SIRADAKİ `n` makro takvim olayı (FOMC/CPI/NFP karışık,
    tarihe göre artan). NFP kural-tabanlı (bkz. `_first_friday`, `as_of` yılı ±1 yıl için
    üretilir — yıl-sınırı yakınında bile sürekli çalışır); FOMC/CPI yalnız 2026 STATİK
    tablosundan gelir (başka yıl için boş kalabilir — elle güncelleme notu, modül docstring'i).

    Dönüş: `[{'event','date' (ISO),'days_away'}, ...]`, `n<=0` → `[]`."""
    today = _to_utc_date(as_of)
    events: list[tuple[date, str]] = [(d, "FOMC") for d in _FOMC_2026]
    events += [(d, "CPI") for d in _CPI_2026]
    for yr in (today.year - 1, today.year, today.year + 1):
        events += [(_first_friday(yr, m), "NFP") for m in range(1, 13)]
    upcoming = sorted((d, kind) for d, kind in events if d >= today)
    return [{"event": kind, "date": d.isoformat(), "days_away": (d - today).days}
            for d, kind in upcoming[:max(0, int(n))]]


def opex_week(d: date | datetime) -> bool:
    """Standart AYLIK opsiyon vadesi (OPEX), ayın ÜÇÜNCÜ Cuma günüdür (OCC/borsa
    konvansiyonu). `d`, bu Cuma ile AYNI takvim haftasındaysa (Pazartesi–Pazar) `True`."""
    day = _to_utc_date(d)
    third_friday = _first_friday(day.year, day.month) + timedelta(days=14)
    week_start = third_friday - timedelta(days=third_friday.weekday())
    week_end = week_start + timedelta(days=6)
    return week_start <= day <= week_end


def quarter_end(d: date | datetime, window_days: int = 3) -> bool:
    """Çeyrek-SONU YAKINLIĞI: `d`, ilgili çeyreğin (Mar/Haz/Eyl/Ara) son takvim gününe
    `window_days` (varsayılan 3, dahil — rebalancing farkındalık penceresi) içindeyse
    `True`. `d`'nin ayı çeyrek-sonu ayı DEĞİLSE her zaman `False` (tek-gün eşleşmesi yerine
    kısa pencere: 'çeyrek-sonu rejimindeyiz' bağlamı için daha kullanışlı, spec §3.13)."""
    day = _to_utc_date(d)
    if day.month not in (3, 6, 9, 12):
        return False
    last_day = (date(day.year, 12, 31) if day.month == 12
                else date(day.year, day.month + 1, 1) - timedelta(days=1))
    return 0 <= (last_day - day).days < max(1, int(window_days))


def earnings_proximity(symbol: str, d: date | datetime) -> dict | None:
    """Sembolün SIRADAKİ bilanço tarihine yakınlığı — mevcut nasdaq-earnings köprüsünü
    (`nihai.data.event_calendar.get_company_events`, Tur-14 P0'da eklendi: BİRİNCİL
    yfinance + İKİNCİL keysiz Nasdaq takvimi) LAZY import ile kullanır (nihai.ai paket-
    importuna ağır/ağ-bağımlı bağımlılık EKLEMEMEK için). Köprü modülü bulunamaz/İÇE
    AKTARILAMAZ/çağrı herhangi bir istisna verirse (ağ hatası dahil) → `None` (fail-soft,
    spec §3.13: 'bulamazsan fail-soft None + docstring notu').

    Dönüş (bulunursa): `{'symbol','date','next_earnings','days_to_earnings','timing',
    'source'}` — `days_to_earnings`, köprünün döndürdüğü HAM `next_earnings` tarihi ile
    ÇAĞRILAN `d` (bugün DEĞİL) arasındaki farktır (tz-güvenli, look-ahead'siz: geçmiş bir
    `d` ile replay/test edilebilir)."""
    day = _to_utc_date(d)
    try:
        from nihai.data import event_calendar
        events = event_calendar.get_company_events(str(symbol))
    except Exception as e:  # noqa: BLE001 — köprü yok/ağ hatası: fail-soft None
        logger.debug(f"[ai.text_intel] earnings_proximity köprü hatası: {e}")
        return None
    next_earnings = getattr(events, "next_earnings", None)
    base = {"symbol": str(symbol).upper(), "date": day.isoformat(),
            "timing": getattr(events, "timing", None),
            "source": getattr(events, "source", None)}
    if next_earnings is None:
        return {**base, "next_earnings": None, "days_to_earnings": None}
    return {**base, "next_earnings": next_earnings.isoformat(),
            "days_to_earnings": (next_earnings - day).days}


market_calendar = SimpleNamespace(
    next_macro_events=next_macro_events, opex_week=opex_week,
    quarter_end=quarter_end, earnings_proximity=earnings_proximity,
)
