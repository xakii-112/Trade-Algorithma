"""
shadow_runner.py — Quant Beyin GÖLGE-KOŞUCU (Tur-17). KARAR-DIŞI/GÖLGE.

NEDEN: `nihai.ai.brain.think()` (v3 "Quant Beyin" düşünme döngüsü) DECISION_MODE=shadow
iken bile hiçbir şey tarafından PERİYODİK çağrılmıyordu — gölge karar/epizot birikimi
olmadan kanıt döngüsü (learning.load_state kalibrasyonu, episodic benzer-durum hafızası)
asla beslenmez. Bu modül YALNIZCA o boşluğu doldurur: watchlist'teki sembolleri tarar,
her biri için son ~120 günlük kapanışı ÇEKER ve `brain.think()`'i çağırır.

ANAYASA (ihlal edilemez):
  - Bu koşucu sinyal/eşik/META_GATE/execution'a ASLA dokunmaz, hiçbir üretim kararını
    etkilemez/üretmez/bastırmaz.
  - YALNIZ OKUR: fiyat verisi (mevcut provider fabrikası üzerinden) + watchlist dosyası.
    Hiçbir strateji/sinyal/ÜRETİM dosyasına YAZMAZ. (Gölge kipte tur sonunda
    `nihai.ai.portfolio.propose` çağrılır — PORTFÖY BAKIŞI köprüsü, Tur-17; onun yazdığı
    tek şey kendi GÖLGE öneri-JSONL'idir ve decision.py'nin gölge kayıtlarıyla aynı
    disipline tabidir: rapor-only, execution'a bağlanmaz.)
  - Yazma davranışı TAMAMEN `nihai.ai.decision.resolve_mode()`'a tabidir: AI_DECISION_MODE
    != "shadow" iken brain/decision katmanı hiçbir dosyaya dokunmaz (ACI DERS #3/#4
    disiplini — "kenar bulundu" imalı hiçbir şey üretilmez). Bu modülün KENDİ telemetri
    özeti (`SHADOW_LOG_FILE`) koşunun kendisi hakkındadır (kaç sembol/başarı/ABSTAIN vb.)
    ve mod'dan BAĞIMSIZ her koşuda yazılır — kip=off iken de "koşu oldu ama kip kapalı"
    bilgisi kaybolmasın diye (özet içinde ayrıca UYARI notu taşır).
  - Özellik vektörü: `nihai.signals.meta_features.extract_features` gerçek strateji-kalite
    alanları (rr_ratio/quality_score/confluence) için bir `Signal` nesnesi BEKLER; burada
    duck-tipli minimal bir yer-tutucu (`_NullSignal`) kullanılır. Ayrıca extract_features
    ilk çağrıda `strategy._load_vix()` üzerinden BİR KEREYE MAHSUS ağ çağrısı (yfinance
    ^VIX) tetikler — bu koşucu SAATTE BİR yeni bir SÜREÇ olarak çalıştığından (zamanlanmış
    görev), bu "bir kereye mahsus" maliyeti her koşuda TEKRAR ödenir; bu artık "UCUZ"
    değildir ve Madde 48 (rate-limit) riskini gereksiz büyütür. Bu yüzden `_cheap_features`
    ASLA kendiliğinden ısıtmaz: yalnız önbellek ZATEN sıcaksa özellik döndürür.
    Isıtma artık AÇIK ve TEK NOKTADA yapılır (`_warm_vix`, koşu başında bir kez, `WARM_VIX`
    ile kapatılabilir). NEDEN gerekti: ısıtma olmadan bu süreçte önbellek hiç sıcak olmuyor,
    `features=None` geçiliyor ve ÖNGÖR adımı model yüklense bile PASİF kalıyordu — koşu
    başına tek ^VIX çağrısı, sembol başına ödenen bar çağrısının yanında ihmal edilebilir.
    Isıtma başarısız olursa VIX nötr 18.0'a düşer (F6) ve koşu aynen sürer.

  - GÖLGE SİNİR AĞI KÖPRÜSÜ (kablo): `brain.think()` bugüne dek `neural_model=None` ile
    çağrılıyordu, yani ÖNGÖR adımı ÜRETİMDE HİÇ ÇALIŞMAMIŞTI (diskteki eğitilmiş artefakt
    hiçbir yerden yüklenmiyordu). `neural.load_any`'nin docstring'inde adı geçen "canlı
    gölge köprüsü" ucu BURASIDIR. Model KARAR VERMEZ: `brain` onu yalnız gölge `p_win ± std`
    olarak taşır, yazma disiplini `decision.resolve_mode()`'a tabidir.

Kullanım:
    python -m nihai.ai.shadow_runner --once                  # watchlist, en fazla 30 sembol
    python -m nihai.ai.shadow_runner --once --symbols "AAPL,MSFT" --max 5
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from nihai.ai import brain
from nihai.ai import decision as decision_mod
from nihai.config import ACTIVE_STRATEGY, DATA_DIR
from nihai.data.data_provider import DataProvider, get_provider
from nihai.data.market_calendar import kapanmis_gunluk

# Modül-düzeyi yollar: testler monkeypatch'ler (decision.py/memory.py deseni).
WATCHLIST_FILE = DATA_DIR / "watchlist.json"
SHADOW_LOG_FILE = DATA_DIR / "ai_shadow_runner_log.jsonl"

# Koşu başında ^VIX önbelleğini bir kez ısıt (bkz. modül docstring'i). Testler bunu
# False'a çeker — `_cheap_features`'ın "ağı KENDİ tetiklemez" sözleşmesi ayrı ve korunur.
WARM_VIX = os.getenv("AI_SHADOW_WARM_VIX", "1") != "0"
# Gölge neural artefaktı: verilmezse DATA_DIR'daki en yeni UYUMLU `neural_*` manifesti.
NEURAL_MODEL_BASE = os.getenv("AI_NEURAL_MODEL") or None

_LOCK = threading.Lock()

_DEFAULT_MAX_SYMBOLS = 30
_LOOKBACK_BARS = 120


def _utc_now_iso() -> str:
    """UTC ts (tz-aware, saniye çözünürlüğü) → '...+00:00'. Yerel-saat karışıklığı yok."""
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Watchlist çözümü — data/watchlist.json > WATCHLIST env > boş (KENDİ dosyasına YAZMAZ;
# nihai.execution.watchlist import EDİLMEZ çünkü ilk yüklemede tohum dosyası YAZAR —
# bu koşucu salt-okunur kalmalı).
# ---------------------------------------------------------------------------
def _resolve_watchlist() -> list[str]:
    """Sembol listesi: WATCHLIST_FILE (data/watchlist.json, "symbols" alanı) varsa/doluysa
    ONU kullan; yoksa/boşsa/bozuksa WATCHLIST env'ini (virgüllü) kullan; o da yoksa boş liste
    (varsayılan sabit bir sepete SESSİZCE düşülmez — çağıran boş listeyi görüp fark eder)."""
    try:
        if WATCHLIST_FILE.exists():
            raw = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            syms = raw.get("symbols") if isinstance(raw, dict) else None
            if isinstance(syms, list) and syms:
                out = [str(s).strip().upper() for s in syms if str(s).strip()]
                if out:
                    return out
    except Exception as e:  # noqa: BLE001 — bozuk dosya koşuyu düşürmesin, env'e düş
        logger.debug(f"[ai.shadow_runner] {WATCHLIST_FILE} okunamadı: {e}")

    import os
    env_raw = os.getenv("WATCHLIST", "")
    if env_raw.strip():
        return [s.strip().upper() for s in env_raw.split(",") if s.strip()]
    return []


# ---------------------------------------------------------------------------
# Özellik çıkarımı — yalnız VIX önbelleği ZATEN sıcaksa (modül docstring'i: ağ
# tetiklememe sözleşmesi). Aksi halde None (brain.think closes-only modda çalışır).
# ---------------------------------------------------------------------------
class _NullSignal:
    """`meta_features.extract_features`'ın yalnız `getattr` ile okuduğu alanlar için
    minimal duck-type yer-tutucu. Gerçek `strategy.Signal` üretmek (generate_signal
    çağırmak) bu koşucu için gereksiz ağırlık/karmaşıklık katar — koşucu SİNYAL üretmez,
    yalnız fiyat-türevli teknik özellikleri okur; rr_ratio/quality_score/confluence/
    is_long bu bağlamda anlamsız olduğundan nötr değerde bırakılır."""

    reasons: list[str] = []
    rr_ratio: float = 0.0
    quality_score: float = 0.0
    action: Any = None


# ---------------------------------------------------------------------------
# Gölge sinir ağı köprüsü (kablo) + VIX ısıtma
# ---------------------------------------------------------------------------
_NEURAL_MODEL: Any = None
_NEURAL_LOADED = False
_MEMBER_RE = re.compile(r"\.m\d+$")   # ensemble ÜYE dosyaları (x.m0.json) tek başına yüklenmez


def _warm_vix() -> None:
    """^VIX önbelleğini koşu başında BİR KEZ ısıt. `_VIX_LOADED` zaten True ise `_load_vix`
    kendisi erken döner (idempotent). Hata/ağ yokluğu yutulur: VIX nötr 18.0'a düşer ve
    koşu sürer — ısıtma bir KOLAYLIKTIR, kapı değildir."""
    if not WARM_VIX:
        return
    try:
        from nihai.signals import strategy as strategy_mod
        strategy_mod._load_vix()
    except Exception as e:  # noqa: BLE001 — ısıtma hatası koşuyu asla düşürmemeli
        logger.debug(f"[ai.shadow_runner] VIX ısıtma atlandı: {e}")


def _neural_bases() -> list[Path]:
    """Aday artefakt tabanları (uzantısız), en YENİden eskiye. `NEURAL_MODEL_BASE`
    verilmişse yalnız o."""
    if NEURAL_MODEL_BASE:
        return [Path(NEURAL_MODEL_BASE)]
    try:
        cands = sorted(DATA_DIR.glob("neural_*.json"),
                       key=lambda q: q.stat().st_mtime, reverse=True)
    except OSError:
        return []
    return [q.with_suffix("") for q in cands if not _MEMBER_RE.search(q.stem)]


def _load_neural_model() -> Any | None:
    """Canlı özellik şemasıyla UYUMLU en yeni gölge modeli (yoksa None; fail-soft, süreç
    başına BİR kez). torch yoksa `neural.load_any` zaten None döner.

    ŞEMA KAPISI ZORUNLUDUR: `brain._neural_forecast` modelin feature_names'ini canlı
    özellik sözlüğünde arar ve BULAMADIĞINI 0.0 ile doldurur. Yani 12-özellikli sentetik
    bir artefakt (data/ai_models_artifacts/neural_deep_mlp_synthetic_*) sessizce SIFIR
    vektör görür ve gürültüyü `p_win` diye taşırdı — sessiz yanlış, gürültülü hatadan
    beterdir. Bu yüzden feature_names TAM eşleşmiyorsa artefakt ATLANIR."""
    global _NEURAL_MODEL, _NEURAL_LOADED
    if _NEURAL_LOADED:
        return _NEURAL_MODEL
    _NEURAL_LOADED = True
    try:
        from nihai.ai import neural
        from nihai.signals.meta_features import FEATURE_NAMES
    except Exception as e:  # noqa: BLE001 — torch/şema yoksa köprü sessizce pasif
        logger.debug(f"[ai.shadow_runner] neural köprü pasif: {e}")
        return None
    for base in _neural_bases():
        try:
            m = neural.load_any(base)
        except Exception as e:  # noqa: BLE001 — bozuk artefakt koşuyu düşürmemeli
            logger.debug(f"[ai.shadow_runner] {base.name} yüklenemedi: {e}")
            continue
        if m is None or not getattr(m, "is_fitted", False):
            continue
        if list(getattr(m, "feature_names", []) or []) != list(FEATURE_NAMES):
            logger.debug(f"[ai.shadow_runner] {base.name}: özellik şeması uyumsuz, atlandı")
            continue
        _NEURAL_MODEL = m
        logger.info(f"[ai.shadow_runner] gölge neural yüklendi: {base.name}")
        return m
    logger.debug("[ai.shadow_runner] uyumlu neural artefakt yok — ÖNGÖR adımı pasif")
    return None


def _cheap_features(df: pd.DataFrame) -> dict[str, float] | None:
    """Mümkünse GERÇEK teknik özellik vektörü; VIX önbelleği soğuksa (ilk ağ çağrısını
    TETİKLEMEMEK için) ya da herhangi bir hata durumunda None. brain.think closes-only
    (features=None) modda da tam çalıştığından None dönmesi güvenlidir."""
    try:
        from nihai.signals import meta_features
        from nihai.signals import strategy as strategy_mod
        if not getattr(strategy_mod, "_VIX_LOADED", False):
            return None   # ağ tetiklemeyecek şekilde yalnız önbellek sıcakken kullan
        return meta_features.extract_features(df, _NullSignal())
    except Exception as e:  # noqa: BLE001 — özellik çıkarımı koşuyu asla düşürmemeli
        logger.debug(f"[ai.shadow_runner] ucuz özellik çıkarımı atlandı: {e}")
        return None


# ---------------------------------------------------------------------------
# Fiyat çekimi — mevcut provider fabrikası (hub/yfinance failover + kısa TTL cache;
# nihai.data.data_provider.get_provider). Yeni bir veri yolu İCAT EDİLMEZ.
# ---------------------------------------------------------------------------
def _fetch_bars(provider: DataProvider, symbol: str,
                lookback_bars: int = _LOOKBACK_BARS) -> pd.DataFrame | None:
    """Son `lookback_bars` günlük OHLCV barı, yalnız KAPANMIŞ günler (Madde 3). Herhangi bir hata/boş/kolonsuz
    sonuç → None (sembol atlanır, koşu asla çökmez).

    Tur-106 (BULGU-106-17): docstring "sağlayıcı oluşmakta olan barı zaten dışlar" diyordu — akış sağlayıcısı 1d'de
    bugünün yarım barını EKLER. Seans içi saatlik koşular her saat farklı özellikle karar yazıyor, Tur-77'nin aynı-gün
    tekilleştirmesi (özellik eşitliği) tutmuyordu: 30 Eyl sembol başına 6 epizot."""
    try:
        df = kapanmis_gunluk(provider.get_bars(symbol, "1d", lookback_bars=lookback_bars))
    except Exception as e:  # noqa: BLE001 — sağlayıcı hatası koşuyu düşürmemeli
        logger.debug(f"[ai.shadow_runner] {symbol} bar çekimi hata: {e}")
        return None
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if not any(str(c).lower() == "close" for c in df.columns):
        return None
    return df


def _closes_from(df: pd.DataFrame, symbol: str) -> np.ndarray | None:
    """df'nin 'close' kolonunu (büyük/küçük harf duyarsız) sonlu float dizisine çevirir."""
    close_col = next((c for c in df.columns if str(c).lower() == "close"), None)
    if close_col is None:
        return None
    try:
        closes = df[close_col].to_numpy(dtype=float)
    except (TypeError, ValueError) as e:
        logger.debug(f"[ai.shadow_runner] {symbol} close dönüşüm hatası: {e}")
        return None
    return closes if closes.size > 0 else None


# ---------------------------------------------------------------------------
# Koşucunun KENDİ telemetrisi — gölge kararların kendisi decision.py içinde zaten
# yazılıyor; bu yalnız "koşu oldu, şöyle özetlendi" kaydı (mod'dan bağımsız).
# ---------------------------------------------------------------------------
def _append_run_log(summary: dict) -> None:
    """`summary`'i SHADOW_LOG_FILE'a tek satır JSONL ekle (thread-safe append; decision.py/
    memory.py/episodic.py ile AYNI kilit+append deseni — bu kod tabanında "atomik yazım"
    tek-satır append + kilit ile sağlanır; NIHAI_DATA_DIR'a saygılıdır çünkü SHADOW_LOG_FILE
    nihai.config.DATA_DIR'dan türetilir)."""
    line = json.dumps(summary, ensure_ascii=False)
    with _LOCK:
        try:
            SHADOW_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with SHADOW_LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            logger.warning(f"[ai.shadow_runner] {SHADOW_LOG_FILE.name} yazılamadı: {e}")


# ---------------------------------------------------------------------------
# Portföy bakışı köprüsü (Tur-17) — YALNIZ gölge kipte, tur sonunda bir kez.
# ---------------------------------------------------------------------------
_PORTFOLIO_MIN_SYMBOLS = 3
_PORTFOLIO_MIN_BARS = 60


def _shadow_portfolio_view(candidates: list[dict[str, Any]], panel: dict[str, pd.Series],
                           mode: str) -> dict | None:
    """Tur-sonu GÖLGE portföy önerisi: karantinasız sembollerin kapanış paneliyle
    `portfolio.propose` (HRP/BL/CVaR... yöntem-karşılaştırmalı; kendi gölge-JSONL'ine
    yazar). YALNIZ shadow kipte ve yeterli evrende (≥3 sembol, ≥60 ORTAK bar) çalışır;
    her hata → None (koşu asla düşmez). Özet: referans yöntemin (HRP) ilk-3 ağırlığı.

    Hizalama ZAMAN DAMGASIYLA yapılır (`join="inner"`), pozisyonel kuyrukla DEĞİL: bir
    sembolde eksik/fazla bar (tatil, halt, provider farkı) varsa kuyruk-hizalaması farklı
    günleri aynı satıra koyup kovaryansı sessizce bozardı. Yan fayda: ortak indeks
    DatetimeIndex kaldığından `portfolio._panel_dates` tarihleri okuyabilir ve tarihsel
    stres penceresi (replay) atlanmaz."""
    if mode != "shadow" or len(candidates) < _PORTFOLIO_MIN_SYMBOLS:
        return None
    try:
        clean = {s: v[~v.index.duplicated(keep="last")].sort_index()
                 for s, v in panel.items() if v is not None and len(v) > 0}
        if len(clean) < _PORTFOLIO_MIN_SYMBOLS:
            return None
        aligned = pd.concat(clean, axis=1, join="inner")
        if len(aligned) < _PORTFOLIO_MIN_BARS:
            return None
        from nihai.ai import portfolio as portfolio_mod
        res = portfolio_mod.propose(candidates, aligned)
        methods = res.get("methods") or {}
        hrp = methods.get("hrp") if isinstance(methods, dict) else None
        top3: list[list[Any]] = []
        if isinstance(hrp, dict) and isinstance(hrp.get("weights"), dict):
            top3 = [[k, round(float(v), 4)] for k, v in
                    sorted(hrp["weights"].items(), key=lambda kv: -abs(float(kv[1])))[:3]]
        # n_universe = ADAY sayısı değil, ORTAK tarihte gerçekten hizalanabilen sembol sayısı
        # (kesişim bir sembolü düşürdüyse özet bunu göstermeli).
        return {"n_universe": int(aligned.shape[1]), "n_bars": int(len(aligned)),
                "n_methods": len(methods), "hrp_top3": top3}
    except Exception as e:  # noqa: BLE001 — portföy bakışı koşuyu asla düşürmemeli
        logger.debug(f"[ai.shadow_runner] portföy bakışı atlandı: {e}")
        return None


# ---------------------------------------------------------------------------
# Ana API
# ---------------------------------------------------------------------------
def run_once(symbols: list[str] | None = None, max_symbols: int = _DEFAULT_MAX_SYMBOLS) -> dict:
    """Tek tur gölge-düşünme koşusu. KARAR-DIŞI: hiçbir sinyal/eşik/META_GATE/execution
    yoluna dokunmaz — yalnız fiyat OKUR ve `brain.think()` çağırır (yazmalar decision/
    episodic içindeki `AI_DECISION_MODE` yoluyla, bu modülün DIŞINDA gerçekleşir).

    symbols: verilmezse `_resolve_watchlist()` (data/watchlist.json > WATCHLIST env > boş).
    max_symbols: taranacak en fazla sembol sayısı (kaynak ne olursa olsun kırpılır).

    Dönüş — özet dict: {'ts','n_ok','n_fail','abstain','long_bias','short_bias',
    'quarantined','mode'} (+ mode != 'shadow' iken 'note': birikim olmayacağı uyarısı).
    Sembol-düzeyi HERHANGİ bir hata (bar çekimi/think çağrısı) o sembolü n_fail'e sayar
    ve koşuya devam eder — tek bir bozuk sembol tüm turu düşürmez."""
    mode = decision_mod.resolve_mode()
    strategy = (ACTIVE_STRATEGY or "event_breakout").strip() or "event_breakout"

    syms = [s.strip().upper() for s in (symbols if symbols is not None else _resolve_watchlist())
            if str(s).strip()]
    syms = syms[:max(0, max_symbols)]

    n_ok = n_fail = n_neural = 0
    abstain = long_bias = short_bias = quarantined = 0
    panel: dict[str, pd.Series] = {}
    candidates: list[dict[str, Any]] = []

    neural_model = _load_neural_model()
    if syms:
        _warm_vix()

    provider: DataProvider | None = None
    if syms:
        try:
            provider = get_provider()
        except Exception as e:  # noqa: BLE001 — provider kurulamazsa tüm semboller fail sayılır
            logger.debug(f"[ai.shadow_runner] provider kurulamadı: {e}")

    for sym in syms:
        if provider is None:
            n_fail += 1
            continue
        df = _fetch_bars(provider, sym)
        if df is None:
            n_fail += 1
            continue
        closes = _closes_from(df, sym)
        if closes is None:
            n_fail += 1
            continue
        feats = _cheap_features(df)
        try:
            thought = brain.think(sym, strategy, features=feats, closes=closes,
                                  neural_model=neural_model)
        except Exception as e:  # noqa: BLE001 — think() koşuyu asla düşürmemeli
            logger.debug(f"[ai.shadow_runner] {sym} think() hatası: {e}")
            n_fail += 1
            continue

        n_ok += 1
        if thought.neural_pwin is not None:
            n_neural += 1
        d = thought.decision
        if d is not None:
            if d.action == "ABSTAIN":
                abstain += 1
            elif d.action == "LONG_BIAS":
                long_bias += 1
            elif d.action == "SHORT_BIAS":
                short_bias += 1
        if thought.health is not None and thought.health.get("quarantine"):
            quarantined += 1
        elif d is not None:
            # PORTFÖY BAKIŞI köprüsü (Tur-17): karantinasız + kararlı semboller tur-sonu
            # gölge portföy önerisinin evrenini oluşturur (posterior = decision görüşü).
            # Kapanışlar KENDİ zaman indeksleriyle taşınır — hizalama tarihe göre yapılır.
            panel[sym] = pd.Series(closes, index=df.index)
            candidates.append({"symbol": sym, "strategy": strategy,
                               "posterior": float(d.posterior)})

    portfolio_summary = _shadow_portfolio_view(candidates, panel, mode)

    # SONUÇ KOLU (2026-07-26, B-3): ufku dolmuş gölge kararları ileri getiriyle bağla.
    # Aynı sağlayıcı yeniden kullanılır (yeni veri yolu icat edilmez). Hata koşuyu düşürmez.
    resolve_summary: dict | None = None
    if mode == "shadow":
        try:
            from nihai.ai import episode_resolver
            resolve_summary = episode_resolver.resolve_pending(provider=provider)
        except Exception as e:  # noqa: BLE001 — çözücü hatası gölge turu düşürmemeli
            logger.debug(f"[ai.shadow_runner] epizot çözümü atlandı: {e}")

    summary: dict = {
        "ts": _utc_now_iso(),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "abstain": abstain,
        "long_bias": long_bias,
        "short_bias": short_bias,
        "quarantined": quarantined,
        # Kablonun ÖLÇÜSÜ: kaç sembolde ÖNGÖR adımı gerçekten çalıştı. 0 ise model
        # yüklenmemiş ya da özellik vektörü gelmemiştir (sessiz pasiflik görünür olsun).
        "n_neural": n_neural,
        "mode": mode,
    }
    if resolve_summary is not None:
        summary["resolved"] = resolve_summary
    if portfolio_summary is not None:
        summary["portfolio"] = portfolio_summary
    if mode != "shadow":
        summary["note"] = "UYARI: kip off, birikim olmaz"

    _append_run_log(summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def repeat_report(days: int = 14) -> dict:
    """GÖLGE kararlarda TEKRAR oranı — O5/B-6'nın kabul kriterini ÖLÇÜLEBİLİR kılar.

    Bulgu (denetim 2026-07-26 B-6): tetik ABD seansından ÖNCE koştuğu için günlük barlar
    turlar arasında değişmiyordu; 248 kararın **%75.8'i** aynı (gün, sembol, posterior)
    üçlüsünün tekrarıydı — yani turların çoğu bilgi üretmiyordu. Plan O5'in kabul kriteri
    "tekrar oranı %10 altına insin"di ama bunu ölçen bir araç YOKTU (kriter yazılı olup
    ölçülemiyorsa kapı değildir — ACI DERS #8'in genel biçimi).

    Tekrar tanımı: aynı GÜN + SEMBOL + posterior (4 haneye yuvarlı) birden çok kez
    görülüyorsa, ilki hariç hepsi "fazlalık" sayılır. `oran = fazlalık / toplam`.

    Salt-okunur; hiçbir dosyaya yazmaz. `days<=0` → tüm geçmiş."""
    from nihai.ai import decision as _decision

    path = _decision.DECISIONS_FILE
    if not path.exists():
        return {"n": 0, "fazlalik": 0, "oran": None, "gun": days,
                "not": f"karar günlüğü yok: {path.name}"}
    try:
        raw = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        return {"n": 0, "fazlalik": 0, "oran": None, "gun": days, "not": f"okunamadı: {e}"}

    esik = None
    if days > 0:
        esik = (datetime.now(UTC) - timedelta(days=int(days))).isoformat()

    gorulen: dict[tuple[str, str, float], int] = {}
    n = 0
    for line in raw:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            ts = str(row["ts"])
            if esik is not None and ts < esik:
                continue
            key = (ts[:10], str(row["symbol"]), round(float(row["posterior"]), 4))
        except (ValueError, KeyError, TypeError):
            continue   # bozuk satır sayıma girmez (savunmacı okuma — modül geneli desen)
        n += 1
        gorulen[key] = gorulen.get(key, 0) + 1

    fazlalik = sum(c - 1 for c in gorulen.values() if c > 1)
    oran = round(fazlalik / n, 4) if n else None
    out: dict = {"n": n, "essiz": len(gorulen), "fazlalik": fazlalik, "oran": oran,
                 "gun": days, "esik_o5": 0.10}
    if oran is None:
        out["verdikt"] = "Ölçülecek karar yok (gölge koşucu hiç kaydetmemiş olabilir)."
    elif oran < 0.10:
        out["verdikt"] = (f"O5 KABUL: tekrar oranı %{oran * 100:.1f} < %10 — turlar farklı "
                          "girdi görüyor.")
    else:
        out["verdikt"] = (f"O5 GEÇİLMEDİ: tekrar oranı %{oran * 100:.1f} ≥ %10 — turların "
                          "büyük kısmı aynı girdiyi yeniden puanlıyor (tetik penceresi "
                          "piyasa seansıyla hizalı mı?).")
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: --once (tek tur; özet UTF-8 stdout'a JSON basılır) | --symbols "AAPL,MSFT"
    (watchlist'i override eder) | --max N (en fazla N sembol). Sonsuz döngü YOK —
    zamanlanmış görev (tur17-shadow-gorev.ps1) periyodik olarak --once çağırır; bu
    "basit ve çökme-dayanıklı" tasarımın gerekçesidir (uzun ömürlü süreç yok → bellek
    sızıntısı/kilitlenme birikmez, her koşu temiz bir Python süreci)."""
    import argparse
    import sys
    try:  # Windows cp1254 konsolunda emoji/Türkçe çıktı güvencesi (brain.main O6 deseni)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[ai.shadow_runner] stdout UTF-8 reconfigure edilemedi: {e}")

    ap = argparse.ArgumentParser(description="Quant Beyin GÖLGE-KOŞUCU (KARAR-DIŞI)")
    ap.add_argument("--once", action="store_true", help="tek gölge-düşünme turu çalıştır")
    ap.add_argument("--symbols", type=str, default=None, metavar="SEMBOLLER",
                    help='virgüllü override, örn. "AAPL,MSFT" (verilmezse watchlist)')
    ap.add_argument("--max", type=int, default=_DEFAULT_MAX_SYMBOLS, metavar="N",
                    help="en fazla N sembol")
    ap.add_argument("--repeat-report", action="store_true",
                    help="gölge kararlarda TEKRAR oranını ölç (O5/B-6 kabul kriteri; "
                         "salt-okunur, tur çalıştırmaz)")
    ap.add_argument("--days", type=int, default=14, metavar="N",
                    help="--repeat-report penceresi (gün; 0 = tüm geçmiş)")
    args = ap.parse_args(argv)

    if args.repeat_report:
        print(json.dumps(repeat_report(days=args.days), ensure_ascii=False, indent=2))
        return 0

    if not args.once:
        ap.print_help()
        return 0

    override = None
    if args.symbols:
        override = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    summary = run_once(symbols=override, max_symbols=args.max)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
