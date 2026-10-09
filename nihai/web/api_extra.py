"""api_extra.py — Backend'de VAR olup panoya AÇILMAMIŞ yeteneklerin HTTP yüzeyi (Tur-18).

Bu modül yeni bir yetenek EKLEMEZ; `nihai/ai`, `nihai/data`, `nihai/risk`, `nihai/signals`
ve `nihai/notifier` altında zaten çalışan ama web panosundan görünmeyen çıktıları
salt-okunur uçlara bağlar.

ANAYASA (CLAUDE.md "AI katmanı"): `nihai.ai` KARAR-DIŞI/GÖLGE'dir — sinyal/eşik/META_GATE/
execution'a ASLA bağlanmaz. Buradaki uçların HİÇBİRİ karar üretmez, yalnızca RAPORLAR.
Mutasyon yapan tek uçlar (`/api/ai/ask`, `/api/ai/selfcheck/run`, `/api/ai/portfolio-propose`)
kendi gölge JSONL'lerine yazar; portföye/emre dokunmaz.

DÜRÜSTLÜK: tüm uçlar fail-soft. Bir alt sistem yoksa (torch kurulu değil, FRED anahtarı
boş, henüz log birikmemiş) uç 500 atmaz — `{"ok": false, "error"/"reason": ...}` döner ve
pano "veri yok" der. Bir metriği İDDİA etmek yerine ÖLÇÜLMEDİ demek projenin kuralıdır
(bkz. ACI DERSLER).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field

from nihai.config import CONFIG, DATA_DIR, LOG_DIR

router = APIRouter()

# Port yoklaması önbelleği (Tur-31, `system_health` içinde açıklandı): (host, port) →
# (monotonic damga, ayakta mı). Modül düzeyinde ki uç yeniden çağrıldığında da yaşasın.
_PORT_TTL_SEC = 2.0
_PORT_CACHE: dict[tuple[str, int], tuple[float, bool]] = {}


# ---------------------------------------------------------------------------
# Ortak yardımcılar
# ---------------------------------------------------------------------------
def _safe(fn, *args, **kwargs) -> Any:
    """Çağrıyı fail-soft sarmala: istisna → {'ok': False, 'error': ...}.

    Pano tek bir alt sistemin (ör. FRED anahtarı yok) yokluğunda BÜTÜN sekmeyi
    kaybetmemeli; her kart kendi hatasını gösterir."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 — bilinçli geniş yakalama (rapor katmanı)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:220]}


def _provider():
    """Panonun paylaşılan veri sağlayıcısı (ekstra bağlantı açmaz).

    Geç import: `web_dashboard` bu modülü import ettiği için modül-düzeyi import
    döngü yaratırdı; fonksiyon gövdesinde çağrı anında güvenli."""
    from nihai.web.web_dashboard import _analysis_provider
    return _analysis_provider()


def _bars(symbol: str, interval: str = "1d", lookback: int = 220):
    try:
        return _provider().get_bars(symbol.upper().strip(), interval, lookback)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[api_extra] bar çekilemedi {symbol} {interval}: {e}")
        return None


def _tail_jsonl(path: Path, limit: int = 50) -> list[dict]:
    """JSONL dosyasının son `limit` GEÇERLİ satırı (bozuk satır atlanır, çökme yok).

    BULGU-61-5 (Tur-61): gövde `read_text().splitlines()` ile TÜM dosyayı belleğe alıyordu
    ve sorulan şey yalnız KUYRUKTU. Tur-35 bu deseni üç modülde birden bulup tek kaynağa
    (`atomic_io.tail_lines`, O(okunan bayt)) indirmişti — bu, aynı desenin dördüncü
    kopyasıydı ve tam da tur-35 dersinin uyardığı yerde, yeni bir modülde YENİDEN doğmuştu.
    Kuyruk okuyucu artık TEK kaynaktan gelir; `limit*2` payı bozuk/boş satır toleransı
    olarak KORUNDU (davranış aynı, yalnız okunan bayt düştü)."""
    if not path.exists():
        return []
    try:
        from nihai.atomic_io import tail_lines
        lines = tail_lines(path, max(1, limit) * 2)
    except Exception:  # noqa: BLE001
        return []
    out: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out[-limit:]


# ===========================================================================
# 1) QUANT BEYİN (nihai/ai) — panoda hiç yoktu
# ===========================================================================
@router.get("/api/ai/overview")
def ai_overview() -> dict:
    """AI katmanının tek-bakışta durumu: kip, LLM sağlayıcı, gölge-koşucu, karne, gündem.

    KARAR-DIŞI: buradaki hiçbir alan sinyal üretmez/etkilemez (CLAUDE.md anayasası)."""
    import os

    out: dict[str, Any] = {"ok": True}

    def _mode():
        from nihai.ai.decision import resolve_mode
        return resolve_mode()

    out["decision_mode"] = _safe(_mode)

    def _live():   # canlı karar yolunun kipi (Tur-92) — gölge kaydı kipinden AYRI bir sorudur
        from nihai.execution.ai_advisor import resolve_mode as live_mode
        return live_mode()

    out["live_mode"] = _safe(_live)
    out["meta_gate_mode"] = os.getenv("META_GATE", "") or "false (varsayılan)"
    out["predict_track"] = os.getenv("PREDICT_TRACK", "").lower() in ("1", "true", "yes", "on")

    def _prov():
        from nihai.ai.router import active_provider
        return {"active": active_provider()}

    out["llm"] = _safe(_prov)

    def _card():
        from nihai.ai.selfcheck import last_card_line
        return {"line": last_card_line()}

    out["selfcheck"] = _safe(_card)

    def _agenda():
        from nihai.ai.agent import agenda_status
        return agenda_status()

    out["agenda"] = _safe(_agenda)

    def _mem():
        from nihai.ai.memory import recent
        rows = recent(days=7, limit=5)
        return {"n_recent": len(rows), "last": rows[-1] if rows else None}

    out["memory"] = _safe(_mem)

    # Gölge-koşucu son turu (Tur-17): data/ai_shadow_runner_log.jsonl
    shadow = _tail_jsonl(DATA_DIR / "ai_shadow_runner_log.jsonl", limit=1)
    out["shadow_runner"] = {"last": shadow[-1] if shadow else None,
                            "log_exists": (DATA_DIR / "ai_shadow_runner_log.jsonl").exists()}

    # Nöral gölge p_win köprüsü (Tur-18): aktif mi, hangi model kartı, değilse NEDEN.
    # Bu görünürlük olmadan köprünün sessizce ölmesi fark edilmez (pasiflik = boş kolon).
    def _neural():
        from nihai.execution.neural_shadow import get_shadow
        return get_shadow().status()

    out["neural_shadow"] = _safe(_neural)

    # Kapasite merdiveni özeti (kaç parametreli katman öneriliyor)
    def _cap():
        # Tur-30: varsayılan `include_neural=False` idi → pano merdiveni v2'nin 4
        # satırında duruyor, HEMEN ALTINDAKİ model-kartı tablosunda listelenen
        # deep_mlp / quant_transformer / quant_transformer_xl satırları hiç
        # görünmüyordu (v3-v4'ün tüm kapasite tartışması panoda yok). Ek maliyet yok:
        # torch bu uçta zaten `neural_shadow.status()` ile içeri girmiş oluyor, sayım
        # teorik (fit yok); torch kurulu değilse satırlar param_count=None + note ile
        # fail-soft gelir.
        from nihai.ai.models import param_ladder_report
        return param_ladder_report(include_neural=True)

    out["capacity_ladder"] = _safe(_cap)
    return out


_BRIEF_CACHE: dict[str, Any] = {"ts": 0.0, "text": None}


@router.get("/api/ai/brief")
def ai_brief(refresh: bool = False) -> dict:
    """`brain.daily_brief()` — günlük beyin brifi (deterministik + opsiyonel LLM).

    10 dakika önbellekli: brif dosya okur ve (varsa) LLM çağırır; pano her 5sn'de
    yeniden üretmemeli."""
    now = time.time()
    if not refresh and _BRIEF_CACHE["text"] is not None and (now - _BRIEF_CACHE["ts"]) < 600:
        return {"ok": True, "cached": True, "age_sec": round(now - _BRIEF_CACHE["ts"]),
                "text": _BRIEF_CACHE["text"]}
    try:
        from nihai.ai.brain import daily_brief
        text = daily_brief()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:220]}
    _BRIEF_CACHE["ts"] = now
    _BRIEF_CACHE["text"] = text
    return {"ok": True, "cached": False, "age_sec": 0, "text": text}


@router.get("/api/ai/learning")
def ai_learning(tail: int = 5000) -> dict:
    """Öğrenme döngüsü: Beta posterior (GLOBAL/LONG/SHORT), Brier, güvenilirlik kovaları, drift.

    DÜRÜST: bu bir kenar İDDİASI değil, kalibrasyon ÖLÇÜMÜdür — ~%50 taban 'öngörü gücü
    yok' demektir."""
    out: dict[str, Any] = {"ok": True}

    def _state():
        from nihai.ai.learning import load_state
        return load_state()

    def _fresh():
        from nihai.ai.learning import update_from_predictions
        return update_from_predictions(tail_n=max(200, min(int(tail), 50000)), persist=False)

    # Salt-okunur önizleme; kalıcı öğrenme günlük harness'in sabit penceresinden gelir.
    fresh = _safe(_fresh)
    state = fresh if isinstance(fresh, dict) and fresh.get("ok") is not False else _safe(_state)
    out["state"] = state

    def _report():
        from nihai.ai.learning import learning_report
        return learning_report(state if isinstance(state, dict) else None)

    out["report"] = _safe(_report)
    return out


@router.get("/api/ai/decisions")
def ai_decisions(days: int = 14, limit: int = 40) -> dict:
    """GÖLGE Bayes kararlar (`nihai/ai/decision.py`): özet rapor + son kayıtlar.

    `AI_DECISION_MODE=off` iken kayıt üretilmez → boş liste dönmesi NORMALDİR."""
    days = max(1, min(int(days), 365))

    def _rep():
        from nihai.ai.decision import decision_report
        return decision_report(days=days)

    rows = _tail_jsonl(DATA_DIR / "ai_decisions.jsonl", limit=max(1, min(int(limit), 200)))
    rows.reverse()  # en yeni üstte
    return {"ok": True, "days": days, "report": _safe(_rep), "rows": rows}


@router.get("/api/golge_kasalar")
def golge_kasalar_durum() -> dict:
    """Tur-102 gölge kasalar: 5 gölge/danışman mekanizma + mekanizmasız kontrol, her biri AYRI sanal kasa.

    Aynı süreçte motor koşuyorsa bellekteki (canlı işaretli) durum, yoksa diskteki son kayıt. YAZMAZ."""
    def _durum():
        from nihai.execution import golge_kasalar
        return golge_kasalar.durum()

    d = _safe(_durum)
    return {"ok": True, **d} if "kasalar" in d else d   # hata: _safe'in {"ok": False, "error": …}


@router.get("/api/ai/episodes")
def ai_episodes(days: int = 30, limit: int = 40) -> dict:
    """Epizodik hafıza + DERS kataloğu (ACI DERSLER'in makine-okunur karşılığı)."""
    days = max(1, min(int(days), 365))
    limit = max(1, min(int(limit), 200))

    def _eps():
        from nihai.ai.episodic import recent_episodes
        eps = recent_episodes(days=days, limit=limit)
        out = []
        for e in eps:
            d = e.__dict__ if hasattr(e, "__dict__") else dict(e)
            out.append({k: v for k, v in d.items() if k in
                        ("id", "ts", "kind", "symbol", "summary", "tags", "outcome")})
        out.reverse()
        return out

    def _lessons():
        from nihai.ai import episodic as ep
        cat = getattr(ep, "LESSONS", None) or []
        out = []
        for lesson in cat:
            d = lesson.__dict__ if hasattr(lesson, "__dict__") else dict(lesson)
            out.append(d)
        return out

    return {"ok": True, "days": days, "episodes": _safe(_eps), "lessons": _safe(_lessons)}


@router.get("/api/ai/episodes/similar")
def ai_episodes_similar(symbol: str | None = None, k: int = 5, ayni_sembol: bool = False) -> dict:
    """Tur-77 · Madde 5.4 — K-NN tarihsel anı: sembolün (verilmezse defterin) EN SON gölge karar
    epizodunu sorgu yapar, en yakın ÇÖZÜLMÜŞ anıları ve havuzun koşulsuz ileri getirisini döndürür.
    KARAR-DIŞI; ölçülen sınırlar `episodic.analog_ozeti` docstring'indedir. Varsayılan olarak aynı
    sembolün geçmiş epizotları dışlanır (ardışık günlerin özellik vektörü neredeyse aynıdır — komşu
    listesi "dün"le dolardı)."""
    k = max(1, min(int(k), 20))

    def _hesap():
        from nihai.ai import episodic
        eps = episodic.recent_episodes(kind="ai_decision", days=365, limit=10_000)
        sym = symbol.upper().strip() if symbol else None
        sorgu = next((e for e in reversed(eps) if e.features and (sym is None or e.symbol == sym)), None)
        if sorgu is None:
            return {"ok": False, "reason": "EPIZOT_YOK", "symbol": sym}
        ozet = episodic.analog_ozeti(sorgu.features, k=k, haric_id=sorgu.id,
                                     haric_sembol=None if ayni_sembol else sorgu.symbol)
        return {"ok": True, "sorgu": {"id": sorgu.id, "ts": sorgu.ts, "symbol": sorgu.symbol,
                                      "karar": sorgu.summary, "n_ozellik": len(sorgu.features)},
                **ozet}

    return _safe(_hesap)


@router.get("/api/ai/knowledge")
def ai_knowledge(q: str = "", concept: str = "") -> dict:
    """Quant/finans bilgi çekirdeği: sözlük + arama + kavram açıklaması + öğrenme sırası."""
    out: dict[str, Any] = {"ok": True}

    def _gloss():
        from nihai.ai.knowledge import glossary
        return glossary()

    out["glossary"] = _safe(_gloss)

    if q.strip():
        def _search():
            from nihai.ai.knowledge import search
            return [{"id": c.id, "name": getattr(c, "name", c.id),
                     "definition": getattr(c, "definition", "")[:300]} for c in search(q)]
        out["results"] = _safe(_search)

    if concept.strip():
        def _explain():
            from nihai.ai.knowledge import explain, formula_for
            return {"id": concept, "text": explain(concept), "formulas": formula_for(concept)}
        out["concept"] = _safe(_explain)

    def _curr():
        from nihai.ai.knowledge import curriculum
        return curriculum()

    out["curriculum"] = _safe(_curr)
    return out


@router.get("/api/ai/registry")
def ai_registry() -> dict:
    """Model yaşam döngüsü: kayıtlı model KARTLARI (veri dönemi, OOS metrikler, kanıt kapıları).

    ACI DERS #8: bir kartta AUC~0.50 iken DSR~1.0 görüyorsan KAPI bozuktur, sonuç değil.

    Bu uç nokta EMEKLİ kartları da döner (`include_deprecated=True`) — pano bir DENETİM
    yüzeyidir, emekli kartı gizlemek kaydı görünmez kılardı (B-5). Emeklilik kartın kendi
    `deprecated` alanından okunur; seçim yapan taraflar (ör. `neural_shadow._pick`)
    varsayılan `list_models()` ile zaten yalnız güncel kartları görür."""
    def _list():
        from nihai.ai.registry import list_models
        return list(list_models(include_deprecated=True))

    cards = _safe(_list)
    ab = _tail_jsonl(DATA_DIR / "ai_models_artifacts" / "ab_log.jsonl", limit=20)
    n_dep = (sum(1 for c in cards if isinstance(c, dict) and c.get("deprecated"))
             if isinstance(cards, list) else 0)
    return {"ok": True, "cards": cards, "ab_log_tail": ab,
            "n": len(cards) if isinstance(cards, list) else 0, "n_emekli": n_dep}


# ---------------------------------------------------------------------------
# BULGU-61-6 (Tur-61) — İÇ-AI DEFTERLERİNİN YÜZEYİ
#
# `nihai/ai` yedi ayrı kalıcı defter yazıyordu ve HİÇBİRİNİN üretim okuyucusu yoktu:
# yazan modül kendi dosyasını okur, ama pano/özet/CLI hiçbirini göstermiyordu. Tur-38'in
# kuralı ("yazılıp okunmayan kolon kusurdur") burada DOSYA ölçeğinde ihlal ediliyordu ve
# sonucu şuydu: bir defter aylarca BOŞ kalsa (yazan yol hiç çalışmadıysa) bunu kimse
# göremezdi — nitekim `agent`/`lab`/`consolidate` defterleri tam olarak öyleydi.
#
# Yedi ayrı panel YAZILMADI. Her defterin sorusu AYNI: *"canlı mı, ne zaman yazıldı, en
# son ne yazıldı?"* → tek uç, tek kart. Ayrıntılı yorum defterin KENDİ sahibinin işidir;
# burada iddia yok, ÖLÇÜM var.
#
# Kayıt tek kaynaktır ve `tests/test_tur61_ai_depolari.py` onu koda karşı denetler: yeni
# bir defter doğar da buraya yazılmazsa test KIRMIZI olur (kural, tek seferlik düzeltme
# değil).
# ---------------------------------------------------------------------------
AI_STORES: dict[str, tuple[str, str]] = {
    "agenda": ("ai_agenda.jsonl", "Beyne sorulan sorular + üretilen görev/ajanda kayıtları"),
    "lab": ("ai_lab_log.jsonl", "Sandbox/deney koşuları (vol modeli, olay etüdü, özellik tarama)"),
    "lesson_candidates": ("ai_lesson_candidates.jsonl", "Epizodik hafızadan çıkan DERS ADAYLARI"),
    "semantic_memory": ("ai_semantic_memory.jsonl", "Bölüm kümelerinden damıtılmış desen özetleri"),
    "text_scores": ("ai_text_scores.jsonl", "Metin tonu/yenilik skorları (haber, başlık)"),
    "adaptive_calibration": ("ai_adaptive_calibration.json", "Yürüyen (prequential) kalibrasyon verdikti"),
    "uretim_kpi": ("uretim_kpi.json", "Gölge karar backtest'inin üretim KPI özeti"),
}

# Bu defterlerin ZATEN bir yüzeyi var — kayıt onları kapsamaz, denetim de aramaz.
# İki grup: (a) `nihai/ai`in sahibi olduğu ve kendi ucu bulunanlar, (b) `nihai/ai`in
# yalnız OKUDUĞU, sahibi başka paket olan defterler (onların yüzeyi de başka yerde).
AI_STORES_BASKA_UCTA = frozenset({
    # (a) kendi ucu var
    "ai_decisions.jsonl", "ai_episodes.jsonl", "ai_memory.jsonl", "ai_selfcheck.jsonl",
    "ai_shadow_runner_log.jsonl", "ai_portfolio_proposals.jsonl", "ai_learning_state.json",
    "ai_calibrators.json", "ai_router_state.json",
    # (b) sahibi `nihai/ai` DEĞİL — yalnız okunuyor (yüzeyleri: /api/system/health,
    # /api/predictions, /api/watchlist)
    "predictions_log.csv", "live_features_log.csv", "watchlist.json",
})


def _ozet_kayit(obj: Any, alan_siniri: int = 6, metin_siniri: int = 90) -> dict:
    """Bir JSON kaydını kart için KÜÇÜLT: ilk birkaç skaler alan, kırpılmış metin.

    Defter başına elle alan eşlemesi YAZILMADI — yedi defterin şeması birbirinden farklı
    ve hiçbirinin şeması sabit değil; sabitlemek, sahibinin bir alan eklediği gün kartı
    sessizce eskitirdi. Genel küçültme "ne yazıldı?" sorusunu şema bilgisi olmadan yanıtlar."""
    if not isinstance(obj, dict):
        return {"deger": str(obj)[:metin_siniri]}
    out: dict[str, Any] = {}
    for k, v in obj.items():
        if len(out) >= alan_siniri:
            break
        if isinstance(v, bool | int | float) or v is None:
            out[str(k)] = v
        elif isinstance(v, str):
            out[str(k)] = v[:metin_siniri]
        elif isinstance(v, list | dict):
            out[str(k)] = f"<{type(v).__name__} n={len(v)}>"
    return out


@router.get("/api/ai/stores")
def ai_stores(tail: int = 3) -> dict:
    """İç-AI defterlerinin canlılık karnesi: var mı · ne kadar · ne zaman · son kayıt.

    Satır SAYISI bilerek raporlanmaz: saymak dosyanın TAMAMINI taratır ve bu uç bir OKUMA
    yoludur (Tur-35/59: ölçüm bütçesini okuyan da öder). Canlılık için boyut + yaş yeter;
    içerik için `atomic_io.tail_lines` yalnız kuyruğu okur."""
    tail = max(0, min(int(tail), 20))
    now = time.time()
    out: dict[str, Any] = {}
    for ad, (dosya, aciklama) in AI_STORES.items():
        yol = DATA_DIR / dosya
        kayit: dict[str, Any] = {"file": dosya, "desc": aciklama, "exists": yol.exists()}
        if not yol.exists():
            # "Henüz yazılmadı" bir HATA değildir (yazan yol hiç çalışmamış olabilir);
            # kart bunu "YOK" diye gösterir, sessizce boş satır basmaz.
            out[ad] = kayit
            continue
        try:
            st = yol.stat()
            kayit["size_kb"] = round(st.st_size / 1024, 1)
            kayit["age_min"] = round((now - st.st_mtime) / 60, 1)
            if tail:
                if dosya.endswith(".jsonl"):
                    kayit["last"] = [_ozet_kayit(r) for r in _tail_jsonl(yol, tail)][-tail:]
                else:
                    obj = json.loads(yol.read_text(encoding="utf-8"))
                    kayit["last"] = [_ozet_kayit(obj)]
        except Exception as e:  # noqa: BLE001
            kayit["error"] = f"{type(e).__name__}: {e}"[:160]
        out[ad] = kayit
    n_bos = sum(1 for v in out.values() if not v.get("exists"))
    return {"ok": True, "stores": out, "n": len(out), "n_missing": n_bos}


@router.get("/api/ai/capacity")
def ai_capacity(n_features: int = 25, include_neural: bool = True) -> dict:
    """Parametre kapasite merdiveni + sinir ağı teorik parametre sayısı.

    'Kaç parametrelik model bu veri hacmine dürüst gelir?' sorusunun cevabı — az veriyle
    yüksek kapasite = ezberleme (ACI DERS #2/#5)."""
    n_features = max(1, min(int(n_features), 512))

    def _ladder():
        from nihai.ai.models import param_ladder_report
        return param_ladder_report(n_features=n_features, include_neural=include_neural)

    return {"ok": True, "n_features": n_features, "ladder": _safe(_ladder)}


class AskRequest(BaseModel):
    question: str


@router.post("/api/ai/ask")
def ai_ask(req: AskRequest) -> dict:
    """`agent.ask()` — kaynaklı Türkçe cevap. KARAR-DIŞI: yalnız mevcut log/hafızayı okur."""
    def _ask():
        from nihai.ai.agent import ask
        return ask(req.question)

    return {"ok": True, "answer": _safe(_ask)}


@router.get("/api/ai/selfcheck")
def ai_selfcheck_last(limit: int = 10) -> dict:
    """SON öz-değerlendirme karneleri (SALT-OKUNUR — yeni sınav KOŞMAZ)."""
    rows = _tail_jsonl(DATA_DIR / "ai_selfcheck.jsonl", limit=max(1, min(int(limit), 50)))
    rows.reverse()

    def _line():
        from nihai.ai.selfcheck import last_card_line
        return last_card_line()

    return {"ok": True, "last_line": _safe(_line), "cards": rows}


@router.post("/api/ai/selfcheck/run")
def ai_selfcheck_run() -> dict:
    """Üç bölümlü sınavı KOŞ ve karneyi geçmişe ekle (operatör tetikler; birkaç saniye)."""
    def _run():
        from nihai.ai.selfcheck import report_card
        return report_card()

    return {"ok": True, "card": _safe(_run)}


class PortfolioProposeRequest(BaseModel):
    symbols: list[str] = []
    lookback: int = 250
    interval: str = "1d"
    max_weight: float = 0.30
    kelly_fraction: float = 0.25
    covariance_model: Literal["ledoit_wolf", "rmt"] = "ledoit_wolf"
    herc_clusters: int = Field(default=0, ge=0, le=20)


@router.post("/api/ai/portfolio-propose")
def ai_portfolio_propose(req: PortfolioProposeRequest) -> dict:
    """GÖLGE portföy önerisi: min-var / maks-Sharpe / HRP / ERC / Black-Litterman /
    CVaR-LP / çok-varlıklı Kelly karşılaştırması.

    KARAR-DIŞI (spec §3.10): emir/pozisyon sistemine BAĞLANMAZ, yalnız `propose()`
    çıktısını raporlar ve gölge JSONL'e yazar. Sembol verilmezse canlı watchlist kullanılır."""
    syms = [s.upper().strip() for s in (req.symbols or []) if s and s.strip()]
    if not syms:
        try:
            import nihai.execution.watchlist as wl
            syms = list(wl.current())[:12]
        except Exception:  # noqa: BLE001
            syms = []
    syms = syms[:20]
    if len(syms) < 2:
        return {"ok": False, "reason": "En az 2 sembol gerekir (watchlist boş)."}

    lookback = max(60, min(int(req.lookback), 1000))
    closes: dict[str, Any] = {}
    missing: list[str] = []
    for s in syms:
        df = _bars(s, req.interval, lookback)
        if df is None or df.empty or "close" not in df.columns:
            missing.append(s)
            continue
        closes[s] = df["close"]
    if len(closes) < 2:
        return {"ok": False, "reason": f"Yeterli fiyat verisi yok (eksik: {', '.join(missing) or '—'})"}

    def _prop():
        import pandas as pd

        from nihai.ai.portfolio import propose
        panel = pd.DataFrame(closes).dropna()
        candidates = [{"symbol": s, "strategy": "watchlist"} for s in panel.columns]
        return propose(candidates, panel,
                       {"max_weight": float(req.max_weight),
                        "kelly_fraction": float(req.kelly_fraction),
                        "covariance_model": req.covariance_model, "herc_clusters": req.herc_clusters or None})

    return {"ok": True, "symbols": list(closes.keys()), "missing": missing,
            "proposal": _safe(_prop)}


@router.get("/api/ai/proposals")
def ai_proposals(limit: int = 10) -> dict:
    """Daha önce üretilmiş gölge portföy önerileri (JSONL kuyruğu)."""
    rows = _tail_jsonl(DATA_DIR / "ai_portfolio_proposals.jsonl", limit=max(1, min(int(limit), 50)))
    rows.reverse()
    return {"ok": True, "rows": rows}


# ===========================================================================
# 2) PİYASA İSTİHBARATI — event/macro/insider/sentiment/IV/yapı/kurulum skoru
# ===========================================================================
@router.get("/api/intel/symbol")
def intel_symbol(symbol: str, interval: str = "1d") -> dict:
    """TEK sembol için 'sinyal ötesi' bağlam: bilanço/temettü takvimi ve guard'lar,
    IV rank, insider (Form 4), Reddit sentiment, halt/LULD + SSR, kurulum skoru (B7/B8),
    piyasa rejimi ve VERİ SAĞLIĞI karnesi.

    Bunların hepsi motorda zaten çalışıyordu ama panoda görünmüyordu. Hiçbiri tek başına
    KANITLI kenar değildir (ADAY sinyaller — bkz. EDGE_FINDINGS); pano bunları 'bağlam'
    olarak göstermelidir."""
    sym = symbol.upper().strip()
    if not sym:
        return {"ok": False, "reason": "Sembol boş."}
    out: dict[str, Any] = {"ok": True, "symbol": sym, "interval": interval}

    df = _bars(sym, interval, 220)
    out["bars_count"] = 0 if df is None else int(len(df))

    # --- Olay takvimi + guard'lar
    def _events():
        from nihai.data.event_calendar import (
            corporate_action_guard,
            days_to_next_earnings,
            days_to_next_ex_dividend,
            event_guard,
            get_company_events,
        )
        ev = get_company_events(sym)
        blocked, reason = event_guard(sym)
        ca_blocked, ca_reason = corporate_action_guard(sym)
        return {
            "next_earnings": str(ev.next_earnings) if ev.next_earnings else None,
            "timing": ev.timing,
            "next_ex_dividend": str(ev.next_ex_dividend) if ev.next_ex_dividend else None,
            "last_split": str(ev.last_split) if ev.last_split else None,
            "source": ev.source,
            "days_to_earnings": days_to_next_earnings(sym),
            "days_to_ex_div": days_to_next_ex_dividend(sym),
            "event_guard": {"blocked": blocked, "reason": reason},
            "corporate_action_guard": {"blocked": ca_blocked, "reason": ca_reason},
        }

    out["events"] = _safe(_events)

    # --- IV rank (opsiyon tarafı bağlamı)
    def _iv():
        from nihai.data.iv_history import history_stats
        return history_stats(sym)

    out["iv_history"] = _safe(_iv)

    # --- Insider (SEC Form 4) ADAY sinyali
    def _ins():
        from nihai.data.insider import insider_events, insider_signal
        ev = insider_events(sym)
        if ev is None or getattr(ev, "empty", True):
            return {"n_events": 0, "signal": "WAIT", "reasons": ["kayıt yok / kapalı"]}
        last_idx = ev.index[-1]
        sig, reasons = insider_signal(ev, last_idx)
        return {"n_events": int(len(ev)), "last_filing": str(last_idx),
                "signal": sig, "reasons": reasons[:4]}

    out["insider"] = _safe(_ins)

    # --- Reddit sentiment ADAY sinyali (SALT geçmiş — canlı çekim yapmaz)
    def _sent():
        from nihai.data.sentiment import load_history, sentiment_signal
        hist = load_history(sym)
        if hist is None or getattr(hist, "empty", True):
            return {"n_days": 0, "signal": "WAIT", "reasons": ["geçmiş yok / kapalı"]}
        last_idx = hist.index[-1]
        sig, reasons = sentiment_signal(hist, last_idx)
        last_row = hist.iloc[-1].to_dict()
        return {"n_days": int(len(hist)), "last_date": str(last_idx),
                "last_score": round(float(last_row.get("score", 0.0)), 3),
                "signal": sig, "reasons": reasons[:4]}

    out["sentiment"] = _safe(_sent)

    # --- Piyasa yapısı (halt/LULD, SSR)
    def _struct():
        from nihai.data.market_structure import halt_or_luld_guard, ssr_active, ssr_short_guard
        if df is None or df.empty:
            return {"reason": "bar yok"}
        halted, hreason = halt_or_luld_guard(df)
        prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else float(df["close"].iloc[-1])
        price = float(df["close"].iloc[-1])
        ssr_block, ssr_reason = ssr_short_guard(prev_close, price)
        return {"halt_or_luld": {"blocked": halted, "reason": hreason},
                "ssr_active": bool(ssr_active(prev_close, price)),
                "ssr_short_guard": {"blocked": ssr_block, "reason": ssr_reason}}

    out["market_structure"] = _safe(_struct)

    # --- Kurulum skoru (B7) + benchmark'a görel güç (B8)
    def _setup():
        from nihai.signals.setup_score import benchmark_returns, compute_setup_score
        bench = _bars("SPY", interval, 220)
        bret = benchmark_returns(bench) if bench is not None else None
        sc = compute_setup_score(df, bret)
        return {"score": round(float(sc.score), 2), "components": sc.components,
                "rs_short": sc.rs_short, "rs_long": sc.rs_long}

    out["setup_score"] = _safe(_setup) if df is not None and not df.empty else {"reason": "bar yok"}

    # --- Piyasa rejimi (SPY): KESKİN etiket + OLASILIKSAL gölge okuma (Tur-37)
    def _regime():
        from nihai.signals.market_regime import compute_regime, regime_shadow
        bench = _bars("SPY", "1d", 220)
        r = compute_regime(bench)
        out_r = {"regime": r.regime, "multiplier": round(float(r.multiplier), 3),
                 "detail": r.detail}
        # `hmm`: KARAR-DIŞI ikinci okuma. `multiplier` bundan ETKİLENMEZ — kategorik etiketin
        # yanına "ne kadar eminiz / ne kadar sürer" sorusunun cevabı konur. `mode_prob`
        # okunmadan `label` okunmaz (regime_hmm sözleşmesi), bu yüzden sözlük BÜTÜN gelir.
        sh = regime_shadow(bench)
        if sh:
            hmm = {k: sh[k] for k in ("hmm_mode_prob", "hmm_switch_prob", "hmm_label")
                   if k in sh}
            if hmm:
                out_r["hmm"] = hmm
        return out_r

    out["regime"] = _safe(_regime)

    # --- Veri sağlığı karnesi (AI v3 WP-O) — 'bu barlara güvenilir mi?'
    def _health():
        from nihai.ai.data_health import health_check
        return health_check(df, symbol=sym)

    out["data_health"] = _safe(_health)

    # --- Metin/takvim zekâsı (OPEX haftası, çeyrek sonu, bilanço yakınlığı)
    def _text():
        from datetime import date

        from nihai.ai.text_intel import earnings_proximity, opex_week, quarter_end
        today = date.today()
        return {"opex_week": bool(opex_week(today)), "quarter_end": bool(quarter_end(today)),
                "earnings_proximity": earnings_proximity(sym, today)}

    out["calendar_intel"] = _safe(_text)
    return out


@router.get("/api/intel/macro")
def intel_macro() -> dict:
    """Makro risk-off kapısı (FRED: VIX / 10y-2y / HY spread) + sıradaki makro olaylar.

    Anahtar yoksa `enabled=false` + nötr çarpan döner (sistem davranışı değişmez)."""
    out: dict[str, Any] = {"ok": True}

    def _snap():
        from nihai.data.macro_calendar import get_macro_snapshot
        return get_macro_snapshot().as_dict()

    out["snapshot"] = _safe(_snap)

    def _events():
        from datetime import date

        from nihai.ai.text_intel import next_macro_events, opex_week, quarter_end
        today = date.today()
        return {"next": next_macro_events(today, n=6),
                "opex_week": bool(opex_week(today)), "quarter_end": bool(quarter_end(today))}

    out["calendar"] = _safe(_events)

    def _regime():
        from nihai.signals.market_regime import compute_regime
        r = compute_regime(_bars("SPY", "1d", 220))
        return {"regime": r.regime, "multiplier": round(float(r.multiplier), 3), "detail": r.detail}

    out["regime"] = _safe(_regime)
    return out


@router.get("/api/options/chain")
def options_chain(symbol: str, expiry: str | None = None, limit: int = 24) -> dict:
    """Opsiyon zinciri SAĞLIĞI + Greekler + önerilen kontrat.

    `options_analyzer` (likidite filtresi, ATM IV, IV rank, pin/atama riski, uzlaşma stili)
    ve `option_strategy.select_option_contract` motorda vardı ama panoda hiç görünmüyordu."""
    sym = symbol.upper().strip()
    try:
        prov = _provider()
        if not expiry:
            # yfinance vade listesi her zaman kronolojik gelmiyor; vadesi GEÇMİŞ bir zincir
            # DTE<0 → Greek/IV hesabını anlamsız kılar. İlk çağrıda en yakın GELECEK vadeyi seç.
            from datetime import date as _date
            probe = prov.get_option_chain(sym, None) or {}
            exps = sorted(str(e) for e in (probe.get("all_expiries") or []))
            today_iso = _date.today().isoformat()
            future = [e for e in exps if e >= today_iso]
            expiry = future[0] if future else (exps[-1] if exps else None)
        chain = prov.get_option_chain(sym, expiry)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "symbol": sym, "error": f"{type(e).__name__}: {e}"[:200]}
    if not chain or "calls" not in chain:
        return {"ok": False, "symbol": sym,
                "reason": "Zincir alınamadı (sağlayıcı opsiyon vermiyor olabilir)."}

    calls = chain.get("calls")
    puts = chain.get("puts")
    under = float(chain.get("underlying_price") or 0.0)
    exp = chain.get("expiry")
    out: dict[str, Any] = {"ok": True, "symbol": sym, "expiry": exp,
                           "all_expiries": chain.get("all_expiries", [])[:24],
                           "underlying_price": round(under, 2)}

    def _health(df, kind: str):
        from nihai.risk.options_analyzer import filter_liquid_chain
        f = CONFIG.options
        res = filter_liquid_chain(df, min_oi=f.min_open_interest,
                                  min_volume=f.min_daily_volume,
                                  max_spread_pct=f.max_spread_pct_of_mid)
        return {"kind": kind, "accepted": int(len(res.accepted)),
                "rejected": int(res.rejected_count), "reasons": res.reasons}

    out["health"] = {"calls": _safe(_health, calls, "calls"), "puts": _safe(_health, puts, "puts")}

    def _atm():
        from nihai.data.iv_history import get_iv_rank
        from nihai.risk.options_analyzer import days_to_expiry, estimate_atm_iv
        dte = days_to_expiry(exp)
        iv = estimate_atm_iv(calls, under, dte)
        rank = get_iv_rank(sym, iv) if iv else None
        return {"dte": dte, "atm_iv": round(float(iv), 4) if iv else None,
                "iv_rank": round(float(rank.iv_rank), 1) if rank else None,
                "iv_rank_confidence": rank.confidence if rank else None,
                "iv_rank_n": rank.sample_size if rank else None}

    out["atm"] = _safe(_atm)

    def _settle():
        from nihai.risk.options_analyzer import settlement_style
        return settlement_style(sym)

    out["settlement"] = _safe(_settle)

    def _exdiv():
        """Madde 25 (Tur-58) — ATM call için ex-temettü erken-kullanım eşiği.

        `option_strategy` bu eşiği SEÇİLEN kontrat üzerinde ölçüp gerekçeye yazıyor;
        operatör panoda bir sembolü İNCELERKEN de aynı soruyu sorabilmeli. Buradaki
        kontrat ATM'ye en yakın CALL'dır (zincir görünümünün çapası) — seçilen kontrat
        DEĞİL; ikisi farklı sorular olduğu için alan adı da ayrıdır."""
        from nihai.data.event_calendar import ex_dividend_context
        from nihai.risk.options_analyzer import early_exercise_risk
        days_div, div_amt = ex_dividend_context(sym)
        mid = None
        if calls is not None and not calls.empty and under > 0:
            row = calls.iloc[(calls["strike"] - under).abs().argsort().iloc[0]]
            k = float(row["strike"])
            mid = float(row.get("mid") or 0) or None
        else:
            k = 0.0
        ee = early_exercise_risk(option_type="c", strike=k, underlying=float(under),
                                 option_mid=mid, days_to_ex_div=days_div,
                                 dividend=div_amt, symbol=sym)
        ee["atm_strike"] = k or None
        return ee

    out["early_exercise"] = _safe(_exdiv)

    def _rows(df, kind: str):
        from nihai.risk.options_analyzer import compute_greeks, days_to_expiry, is_pin_risk
        dte = days_to_expiry(exp)
        t_years = max(dte, 0) / 365.0
        # ATM'ye en yakın `limit` strike (uçtaki ilgisiz strike'ları gösterme)
        d = df.copy()
        d["_dist"] = (d["strike"] - under).abs()
        d = d.nsmallest(max(4, min(int(limit), 60)), "_dist").sort_values("strike")
        rows = []
        for _, r in d.iterrows():
            strike = float(r.get("strike", 0.0))
            bid, ask = float(r.get("bid", 0.0) or 0), float(r.get("ask", 0.0) or 0)
            mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else float(r.get("lastPrice", 0.0) or 0)
            g = compute_greeks(option_type=("c" if kind == "calls" else "p"),
                               S=under, K=strike, T=max(t_years, 1e-6),
                               market_price=mid if mid > 0 else None)
            rows.append({
                "strike": round(strike, 2), "bid": round(bid, 2), "ask": round(ask, 2),
                "mid": round(mid, 2),
                "oi": int(r.get("openInterest", 0) or 0),
                "volume": int(r.get("volume", 0) or 0),
                "iv": round(float(r.get("impliedVolatility", 0.0) or 0), 4),
                "delta": round(g.delta, 3) if g else None,
                "gamma": round(g.gamma, 4) if g else None,
                "theta": round(g.theta, 3) if g else None,
                "vega": round(g.vega, 3) if g else None,
                "pin_risk": bool(is_pin_risk(strike, under, dte)),
            })
        return rows

    out["calls"] = _safe(_rows, calls, "calls")
    out["puts"] = _safe(_rows, puts, "puts")

    def _pick(direction: str):
        from nihai.risk.option_strategy import select_option_contract
        from nihai.signals.strategy import SignalAction
        act = SignalAction.LONG if direction == "LONG" else SignalAction.SHORT
        pick, reasons = select_option_contract(sym, act, _provider())
        if pick is None:
            return {"picked": None, "reasons": reasons[:5]}
        return {"picked": {"option_type": pick.option_type, "strike": pick.strike,
                           "expiry": pick.expiry, "mid": pick.mid, "iv": pick.iv,
                           "spread_pct": pick.spread_pct, "oi": pick.open_interest,
                           "volume": pick.volume,
                           "delta": getattr(pick.greeks, "delta", None)},
                "reasons": reasons[:5]}

    out["suggestion"] = {"long": _safe(_pick, "LONG"), "short": _safe(_pick, "SHORT")}
    return out


# ===========================================================================
# 3) ARAÇLAR — pozisyon boyutu hesaplayıcı + genişletilmiş gösterge paneli
# ===========================================================================
class SizeRequest(BaseModel):
    symbol: str = ""
    entry: float
    stop: float
    side: str = "LONG"
    sector: str = "UNKNOWN"
    confidence: float = 1.0
    p_win: float | None = None


@router.post("/api/tools/position-size")
def tools_position_size(req: SizeRequest) -> dict:
    """`RiskManager.size_stock` ile GERÇEK pozisyon boyutu hesabı — canlı portföy
    durumu (equity/ATH/günlük PnL/ardışık stop/rejim/makro) üzerinden.

    Motor bu hesabı her sinyalde yapıyordu ama operatör panodan SORAMIYORDU. SİNYAL-ONLY:
    emir göndermez, yalnız kaç adet ve niçin sorusunu yanıtlar."""
    if req.entry <= 0 or req.stop <= 0 or abs(req.entry - req.stop) < 1e-9:
        return {"ok": False, "reason": "entry/stop geçersiz (pozitif ve farklı olmalı)."}

    from nihai.config import INITIAL_EQUITY
    from nihai.execution.concurrency import PORTFOLIO_LOCK
    from nihai.execution.state import StateStore
    from nihai.risk.risk_manager import RiskManager

    with PORTFOLIO_LOCK:
        store = StateStore()
        portfolio = store.load_or_init(INITIAL_EQUITY)
        rm = RiskManager(portfolio)

        def _regime():
            from nihai.signals.market_regime import compute_regime
            r = compute_regime(_bars("SPY", "1d", 220))
            rm.regime_multiplier, rm.regime_label = float(r.multiplier), r.regime
            return {"regime": r.regime, "multiplier": round(float(r.multiplier), 3)}

        regime = _safe(_regime)

        conf = float(req.confidence)
        prob_mult = None
        if req.p_win is not None:
            def _pm():
                from nihai.risk.prob_sizing import prob_to_size_mult
                return float(prob_to_size_mult(float(req.p_win)))
            prob_mult = _safe(_pm)
            if isinstance(prob_mult, float):
                conf = max(0.0, min(1.0, conf * prob_mult))

        check = rm.size_stock(float(req.entry), float(req.stop),
                              sector=req.sector or "UNKNOWN",
                              symbol=(req.symbol or None), side=req.side.upper(),
                              confidence_multiplier=conf)

    return {
        "ok": True,
        "approved": bool(check.approved),
        "reason": check.reason,
        "sizing": check.sizing,
        "confidence_used": round(conf, 4),
        "prob_size_multiplier": prob_mult,
        "regime": regime,
        "equity": round(float(portfolio.equity), 2),
        "kill_switch_active": bool(store.kill_switch_active),
        "note": "SİNYAL-ONLY — hesap önerisidir, emir gönderilmez.",
    }


@router.get("/api/indicators/extended")
def indicators_extended(symbol: str, interval: str = "1h") -> dict:
    """`indicators_ext.py` (TradingView kapsamı: HMA/KAMA/TSI/WaveTrend/STC/Force Index...)
    — motorda 24 gösterge vardı, panoda hiçbiri yoktu. Son bar değerleri + yön okuması."""
    sym = symbol.upper().strip()
    df = _bars(sym, interval, 300)
    if df is None or df.empty or len(df) < 60:
        got = 0 if df is None else len(df)
        return {"ok": False, "symbol": sym, "reason": f"yeterli veri yok ({got}/60 bar)"}

    import numpy as np

    from nihai.signals import indicators_ext as ix

    close = df["close"]
    vol = df["volume"] if "volume" in df.columns else None

    def _last(s) -> float | None:
        try:
            v = float(s.dropna().iloc[-1])
            return None if (np.isnan(v) or np.isinf(v)) else round(v, 4)
        except Exception:  # noqa: BLE001
            return None

    price = float(close.iloc[-1])
    rows: list[dict] = []

    def add(name: str, value, note: str = "", signal: str | None = None):
        rows.append({"name": name, "value": value, "note": note, "signal": signal})

    def _ma_sig(v):
        if v is None:
            return None
        return "LONG" if price > v else ("SHORT" if price < v else "WAIT")

    for label, fn, n in (("SMA(20)", ix.sma, 20), ("WMA(20)", ix.wma, 20),
                         ("SMMA(20)", ix.smma, 20), ("DEMA(20)", ix.dema, 20),
                         ("TEMA(20)", ix.tema, 20), ("TMA(20)", ix.tma, 20),
                         ("HMA(20)", ix.hma, 20), ("KAMA(20)", ix.kama, 20)):
        v = _safe(lambda f=fn, k=n: _last(f(close, k)))
        v = v if not isinstance(v, dict) else None
        add(label, v, "fiyat > ortalama = boğa", _ma_sig(v))

    def osc(label, series, lo=None, hi=None, note=""):
        v = _safe(lambda: _last(series))
        v = v if not isinstance(v, dict) else None
        sig = None
        if v is not None and lo is not None and hi is not None:
            sig = "LONG" if v <= lo else ("SHORT" if v >= hi else "WAIT")
        add(label, v, note, sig)

    osc("CMO(14)", _safe(lambda: ix.cmo(close, 14)), -50, 50, "Chande momentum osilatörü")
    osc("DeMarker(14)", _safe(lambda: ix.demarker(df, 14)), 0.3, 0.7)
    osc("DPO(20)", _safe(lambda: ix.dpo(close, 20)), None, None, "trendden arındırılmış fiyat")
    osc("PPO(12,26)", _safe(lambda: ix.ppo(close)), None, None, "yüzde fiyat osilatörü")
    osc("TSI(25,13)", _safe(lambda: ix.tsi(close)), -25, 25, "gerçek güç endeksi")
    osc("RVI(10)", _safe(lambda: ix.rvi(df, 10)[0]), None, None, "göreli canlılık")
    osc("IMI(14)", _safe(lambda: ix.imi(df, 14)), 30, 70, "gün-içi momentum")
    osc("Kairi(20)", _safe(lambda: ix.kairi(close, 20)), -5, 5, "ortalamadan % sapma")
    osc("WaveTrend", _safe(lambda: ix.wavetrend(df)[0]), -60, 60, "WT1 çizgisi")
    osc("STC", _safe(lambda: ix.schaff_trend_cycle(close)), 25, 75, "Schaff trend döngüsü")
    osc("CoG(10)", _safe(lambda: ix.cog(close, 10)), None, None, "ağırlık merkezi")

    if vol is not None:
        osc("Force Index(13)", _safe(lambda: ix.force_index(df, 13)), None, None,
            "hacim × fiyat değişimi")
        osc("A/D Line", _safe(lambda: ix.adl(df)), None, None, "birikim/dağıtım")
        osc("Williams A/D", _safe(lambda: ix.williams_ad(df)), None, None)
        osc("PVT", _safe(lambda: ix.pvt(df)), None, None, "fiyat-hacim trendi")

    longs = sum(1 for r in rows if r["signal"] == "LONG")
    shorts = sum(1 for r in rows if r["signal"] == "SHORT")
    total = longs + shorts
    return {
        "ok": True, "symbol": sym, "interval": interval, "price": round(price, 2),
        "bars": int(len(df)), "rows": rows,
        "consensus": {"long": longs, "short": shorts,
                      "bias": ("LONG" if longs > shorts else ("SHORT" if shorts > longs else "WAIT")),
                      "strength_pct": round(abs(longs - shorts) / total * 100, 1) if total else 0.0},
        "note": "Konsensüs bir KENAR İDDİASI değildir — göstergeler aynı fiyat serisinden türer.",
    }


# ===========================================================================
# 4) SİSTEM — sağlık, günlük özet, kanıt notları
# ===========================================================================
@router.get("/api/system/health")
def system_health() -> dict:
    """Servis/dosya sağlığı: hub + kayıtçı portları, log tazeliği, veri dosyaları,
    forward-test kapsaması. Operatörün 'sistem gerçekten ayakta mı?' sorusu."""
    import os
    import socket

    # Tur-30: iki yoklama SERİ ve 0,6 sn zaman aşımlıydı. ÖLÇÜLDÜ: Windows'ta KAPALI bir
    # loopback portu zaman aşımının tamamını yiyor (624 + 605 ms) → uç 1,6 sn'nin 1,23
    # sn'sini burada geçiriyordu ve "servis kapalı" hâli (yani sorun anı) EN YAVAŞ hâldi.
    # Ayakta bir yerel uvicorn'u OS listen-backlog'u milisaniyede kabul eder; 0,25 sn
    # hâlâ ~50× pay bırakır. Yoklamalar ayrıca paralel.
    # Tur-31: yukarıdaki 0,25 sn'lik pay yalnız servis KAPALIYKEN ödenir ve tam da o an
    # pano en sık yenilenir → ÖLÇÜLDÜ: uç 269 ms'nin ~250 ms'sini iki kapalı portu
    # beklemekle geçiriyor (paralel oldukları için toplam değil, tek timeout kadar).
    # Sonuç 2 sn önbelleklenir: operatör için 2 sn gecikme görünmezdir, ama arka arkaya
    # gelen yoklamalar artık soketi hiç açmaz. Ayakta bir port zaten milisaniyede döner,
    # yani önbellek pratikte SADECE kapalı hâlin cezasını kaldırır.
    def _port(host: str, port: int, timeout: float = 0.25) -> bool:
        anahtar = (host, port)
        onbellek = _PORT_CACHE.get(anahtar)
        if onbellek is not None and (time.monotonic() - onbellek[0]) < _PORT_TTL_SEC:
            return onbellek[1]
        try:
            with socket.create_connection((host, port), timeout=timeout):
                sonuc = True
        except Exception:  # noqa: BLE001
            sonuc = False
        _PORT_CACHE[anahtar] = (time.monotonic(), sonuc)
        return sonuc

    from nihai.config import WEB_HOST, WEB_PORT

    hub_url = os.getenv("HUB_URL", "http://127.0.0.1:8799")
    try:
        hub_port = int(hub_url.rsplit(":", 1)[-1].strip("/"))
    except Exception:  # noqa: BLE001
        hub_port = 8799

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_web = ex.submit(_port, WEB_HOST or "127.0.0.1", int(WEB_PORT))
        f_hub = ex.submit(_port, "127.0.0.1", hub_port)
        services = {
            "web_panel": {"port": WEB_PORT, "up": f_web.result()},
            "data_hub": {"port": hub_port, "up": f_hub.result()},
        }

    def _age(p: Path) -> dict:
        if not p.exists():
            return {"exists": False}
        st = p.stat()
        return {"exists": True, "size_kb": round(st.st_size / 1024, 1),
                "age_min": round((time.time() - st.st_mtime) / 60, 1)}

    files = {
        "system.log": _age(LOG_DIR / "system.log"),
        "predictions_log.csv": _age(DATA_DIR / "predictions_log.csv"),
        "live_features_log.csv": _age(DATA_DIR / "live_features_log.csv"),
        "state.json": _age(DATA_DIR / "state.json"),
        "equity_curve.csv": _age(DATA_DIR / "equity_curve.csv"),
        "watchlist.json": _age(DATA_DIR / "watchlist.json"),
        "meta_model.json": _age(DATA_DIR / "meta_model.json"),
        "tearsheet.html": _age(DATA_DIR / "tearsheet.html"),
    }

    def _pred_service():
        # Tur-30: üç alanın ÜÇÜ de var olmayan anahtarları okuyordu. snapshot() sözleşmesi
        # {"meta", "predictions"} (+ Tur-30'da "running"); "rows"/"updated_at" hiç yoktu →
        # kart daima "durdu · 0 sembol · —" gösteriyordu. Sembol sayısı `predictions`
        # uzunluğu, zaman damgası `meta.updated`.
        from nihai.execution.live_predict_service import get_service
        snap = get_service().snapshot()
        return {"running": bool(snap.get("running", False)),
                "n_symbols": len(snap.get("predictions") or []),
                "updated_at": (snap.get("meta") or {}).get("updated")}

    def _cov():
        # Tur-28'in imza-önbellekli özeti buradan atlanmıştı: ham `coverage()` her
        # istekte 6,7 MB / 53k satırlık predictions_log'u yeniden ayrıştırıyordu
        # (ÖLÇÜLDÜ: ~400 ms, log büyüdükçe doğrusal). `summary()["coverage"]` AYNI
        # sözlüğü döner ama dosya imzası değişmedikçe tek `stat()` maliyetiyle.
        from nihai.execution import prediction_tracker as pt
        return pt.summary()["coverage"]

    def _bridges():
        """BULGU-61-2: canlı KARAR köprülerinin durumu — hiçbir üretim okuyucusu YOKTU.

        `ai_advisor.status()` (Tur-38) ve `shadow_bridge.status()` (Tur-60) docstring'leri
        "pano/günlük özet için" diyordu ama iki dosyada da tek çağıran TESTTİ; üçüncü
        kardeş (`neural_shadow.status()`) `/api/ai/overview`da zaten yayındaydı, yani
        desen vardı ve tam da KARAR yoluna bağlı iki köprü açıkta kalmıştı.

        Bu sayılar en çok `last_error` yüzünden önemli: köprüler fail-soft'tur ve İLK
        hatayı bir kez UYARI, gerisini debug olarak loglar. 7/24 koşan bir tarayıcının
        system.log'unda o tek satır pratikte görünmezdir — `status()` kalıcı yüzeydi ve
        OKUNMADIĞI için "sessiz ölüm yok" koruması kendisi sessizdi (Tur-38/44 dersi).

        Uç, tarayıcıyla AYNI süreçte koşar (`main.start_web_server` daemon thread) →
        sayaçlar GERÇEKTİR. Ayrı bir süreçten (ör. daily_summary) okunsaydı hep 0
        görünürdü; yüzeyin burası olmasının nedeni budur."""
        from nihai.execution import ai_advisor, shadow_bridge
        return {"ai_advisor": ai_advisor.status(), "shadow_bridge": shadow_bridge.status()}

    def _universe():
        """BULGU-69-1: evren DEGRADE'i yalnız bir log satırıydı, okuyan yüzey YOKTU.

        `sp500_tickers()` boş dönerse (Wikimedia 403 / adressiz `SEC_USER_AGENT`) tarama
        ~600 yerine bayat 101 isimlik Nasdaq yedeğiyle koşar; `nasdaq100_tickers()` de
        sessizce o yedeğe düşebilir. İkisi de fail-soft'tur ve durumu BİR KEZ loglar —
        7/24 koşan bir tarayıcının system.log'unda o satır görünmez, yani "sessiz evren
        çökmesi yok" koruması KENDİSİ sessizdi (Tur-61 BULGU-61-2 ile aynı sınıf).

        `canli=False` = degrade. `sources` boş ise bu süreçte evren henüz kurulmamıştır
        (hata DEĞİL): uç tarayan döngüyle AYNI süreçte koşar, ayrı süreçten okunsaydı
        sayılar daima boş görünürdü."""
        from nihai.config import AUTO_SCREEN_UNIVERSE
        from nihai.execution import screen
        return {"auto_screen": AUTO_SCREEN_UNIVERSE, "sources": screen.evren_durum()}

    def _zincir():
        """Madde 244 — karar zincirinin BÜTÜNLÜĞÜ (Tur-76).

        Tur-75 zinciri yazdı, doğruladı, CLI'ını verdi ama üretimde çağıranı YOKTU.
        Tur-76 `log_signal(accepted)` ve `log_trade_postmortem` yolundan yazmaya
        başladı; yazan bir defterin OKUYAN yüzeyi olmadan bırakılması bu depoda
        kusurdur (Tur-38/61). `verify()` zinciri baştan sona gezer — bu yüzden uç
        halka SAYISINI da basar: bir gün maliyet görünür hâle gelirse sayı ortada
        olur, tahmin edilmesi gerekmez."""
        from nihai import audit_chain
        d = audit_chain.verify()
        d["dosya"] = str(audit_chain.CHAIN_PATH)
        return d

    env_flags = {k: (os.getenv(k, "") or "—") for k in
                 ("META_GATE", "PREDICT_TRACK", "AI_DECISION_MODE", "DATA_PROVIDER",
                  "ACTIVE_STRATEGY", "OPTION_MODE", "SHADOW_BRIDGE", "AI_LIVE_MODE",
                  "AUDIT_CHAIN")}
    keys = {k: bool(os.getenv(k, "").strip()) for k in
            ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ALPACA_API_KEY",
             "FRED_API_KEY", "TELEGRAM_BOT_TOKEN", "NASDAQ_DATA_LINK_API_KEY")}

    return {"ok": True, "services": services, "files": files,
            "predict_service": _safe(_pred_service), "coverage": _safe(_cov),
            "bridges": _safe(_bridges), "universe": _safe(_universe),
            "audit_chain": _safe(_zincir),
            "env": env_flags, "api_keys_present": keys}


@router.get("/api/system/daily-summary")
def system_daily_summary(rebuild: bool = False) -> dict:
    """Günlük sistem-sağlığı ÖZETİ (`nihai/notifier/daily_summary.py`). Karar VERMEZ, özetler.

    Varsayılan: son üretilmiş özeti (data/daily_summary_last.txt) okur — ucuz.
    `rebuild=1` ile taze toplar (dosya IO + port yoklaması)."""
    path = DATA_DIR / "daily_summary_last.txt"
    if not rebuild and path.exists():
        try:
            return {"ok": True, "cached": True,
                    "age_min": round((time.time() - path.stat().st_mtime) / 60, 1),
                    "text": path.read_text(encoding="utf-8")}
        except Exception:  # noqa: BLE001
            pass

    def _build():
        from nihai.notifier.daily_summary import build_summary, collect
        return build_summary(collect())

    text = _safe(_build)
    if isinstance(text, dict):
        return {"ok": False, "error": text.get("error")}
    return {"ok": True, "cached": False, "age_min": 0.0, "text": text}


@router.get("/api/system/evidence-notes")
def evidence_notes() -> dict:
    """Projenin KANIT hafızası: docs/EDGE_FINDINGS.md başlıkları + CLAUDE.md ACI DERSLER.

    Panoda 'niçin META_GATE kapalı?' / 'bu kenar neden reddedildi?' sorusunun cevabı
    tek tıkla görünür olsun diye (yeni bir iddia ÜRETMEZ, mevcut kaydı okur)."""
    root = Path(__file__).resolve().parents[2]
    out: dict[str, Any] = {"ok": True}

    ef = root / "docs" / "EDGE_FINDINGS.md"
    if ef.exists():
        try:
            heads = [line.strip() for line in ef.read_text(encoding="utf-8").splitlines()
                     if line.startswith("#")]
            out["edge_findings"] = {"path": "docs/EDGE_FINDINGS.md", "headings": heads[:80],
                                    "count": len(heads)}
        except Exception as e:  # noqa: BLE001
            out["edge_findings"] = {"error": str(e)[:120]}
    else:
        out["edge_findings"] = {"error": "docs/EDGE_FINDINGS.md bulunamadı"}

    cm = root / "CLAUDE.md"
    lessons: list[str] = []
    if cm.exists():
        try:
            txt = cm.read_text(encoding="utf-8")
            if "ACI DERSLER" in txt:
                block = txt.split("ACI DERSLER", 1)[1]
                block = block.split("## Çalışma ilkeleri", 1)[0]
                cur = ""
                for line in block.splitlines():
                    s = line.strip()
                    if s[:2].rstrip(".").isdigit() and s[1:2] in (".", ")"):
                        if cur:
                            lessons.append(cur.strip())
                        cur = s
                    elif cur and s:
                        cur += " " + s
                    elif cur and not s:
                        lessons.append(cur.strip())
                        cur = ""
                if cur:
                    lessons.append(cur.strip())
        except Exception as e:  # noqa: BLE001
            out["lessons_error"] = str(e)[:120]
    out["lessons"] = [line[:600] for line in lessons if len(line) > 30][:12]
    return out
