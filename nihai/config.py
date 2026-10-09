"""
config.py — Sabitler, watchlist, regülasyon eşikleri.

PDF Referansları:
- Madde 2  : İşlem başına risk yüzdesi
- Madde 9  : ATH-bazlı adaptif risk
- Madde 10 : Günlük max kayıp, PDT
- Madde 13 : Reg T (%50) / FINRA sürdürme (%25)
- Madde 14 : Portföy düzeyi maruziyet tavanları
- Madde 27 : Net Greek limitleri
- Madde 28 : DTE eşikleri (vade/0DTE)
- Madde 44 : Sırlar env'de (kodda DEĞİL)

NOT: Bu değerler örnek; canlıya geçmeden önce kendi backtest ve broker'ına göre kalibre et.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


_ENVB_TRUE = ("1", "true", "yes", "on")


def _envb(name: str, default: bool) -> bool:
    """Env'den bool oku ('1/true/yes/on' → True). A/B ölçüm için knob override'ı."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in _ENVB_TRUE


def _meta_gate_mode() -> str:
    """META_GATE env'ini ÜÇ değerli kipe çevir: "off" | "shadow" | "enforce".

    - "shadow"/"gölge"/"golge" (büyük-küçük duyarsız, trim) → "shadow": kapı p_win'i
      hesaplar + metadata/reason'a yazar ama sinyali ASLA bastırmaz. Amaç: canlı kanıt
      döngüsü (prediction_tracker pwin logu → gate_sim/pwin_calibration) kapı fiilen
      kapalıyken de yaşasın (kapı tamamen kapatılırsa bu döngü ölür).
    - _envb-doğru token'lar (1/true/yes/on) → "enforce": eski davranış (eşik altı WAIT).
    - diğer/boş → "off" (varsayılan; ACI DERS #4: OOS-kenarlı baz sinyal yok)."""
    v = (os.getenv("META_GATE") or "").strip().lower()
    if v in ("shadow", "gölge", "golge"):
        return "shadow"
    if v in _ENVB_TRUE:
        return "enforce"
    return "off"


def neural_device() -> str:
    """`NEURAL_DEVICE` env → `NeuralCapacity.fit/pretrain`'in `device` argümanı.

    VARSAYILAN **"cpu"** ve bu BİLEREK böyle: GPU (ve bf16 autocast) kayan-nokta
    sonucunu DEĞİŞTİRİR — aynı seed farklı donanımda farklı ağırlık üretir. Üretim
    varsayılanını çevirmek projenin bit-denklik disiplinini SESSİZCE kırardı, o yüzden
    kapı operatörde. Torch yoksa/CUDA yoksa "auto" zaten sessizce CPU'ya düşer
    (`neural._resolve_device`), yani yanlış ortamda açık kalması çökme üretmez.

    ÖLÇÜLDÜ (bu makine, RTX 4070 Laptop, n=4000 × f=96, 2 epoch, batch 256):
      · `deep_mlp`          0,22M param  cpu   0,21 s → gpu 0,14 s (**1,47×**)
      · `quant_transformer` 5,47M param  cpu 305,25 s → gpu 6,21 s (**49,1×**)

    Kabul edilen: "cpu" | "auto" | "cuda" | "cuda:N". Tanınmayan/boş → "cpu"
    (fail-soft: yanlış yazılmış env eğitimi çökertmez, sessizce eski davranışta kalır)."""
    v = (os.getenv("NEURAL_DEVICE") or "").strip().lower()
    if v in ("cpu", "auto", "cuda"):
        return v
    if v.startswith("cuda:") and v[5:].isdigit():
        return v
    return "cpu"


def neural_amp() -> bool:
    """`NEURAL_AMP` env → `fit(amp=)`. VARSAYILAN False (bkz. `neural_device` gerekçesi).

    `neural_device()` "cpu" iken amp'in CUDA'da zaten hükmü yoktur. AYRI bir anahtar
    olmasının nedeni ölçüm: aynı koşuda `quant_transformer` gpu+amp 3,36 s (cpu'ya göre
    **90,9×**, düz GPU'ya göre 1,85×) AMA `deep_mlp` gpu+amp 0,45 s = **0,48× — YAVAŞLADI**
    (autocast maliyeti 0,22M paramlık modelde kazancı yiyor). Bu yüzden çağıran ayrıca
    tier'a göre kısar; env tek başına küçük modele amp AÇTIRAMAZ."""
    return _envb("NEURAL_AMP", False)


def _parse_tiers(raw: str) -> tuple[tuple[float, float], ...]:
    """'profit_r:trail_r' çiftlerini (virgülle ayrık) parse et → kâr-eşiğine göre AZALAN
    sıralı tuple. Örn '1.5:2.0,3.0:1.5,5.0:1.0' → kâr 5R+ iken trail 1.0R (en sıkı),
    3R+ iken 1.5R, 1.5R+ iken 2.0R. Bozuk/boş girdi → boş tuple (tiered devre dışı)."""
    tiers: list[tuple[float, float]] = []
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            at_s, r_s = chunk.split(":")
            tiers.append((float(at_s), float(r_s)))
        except (ValueError, TypeError):
            continue
    tiers.sort(key=lambda t: -t[0])   # en yüksek kâr eşiği önce (lookup ilk eşleşmeyi alır)
    return tuple(tiers)


# Sinyal-only uyarı kipi: geçerli sinyal (RR+kalite filtrelerini geçen) üretilir
# üretilmez, RİSK ONAYINDAN BAĞIMSIZ olarak operatöre bildirilir ("X hisse LONG/SHORT,
# hedef, stop"). Risk kapıları (soğutma/caps/buying-power/PDT) yalnızca sanal portföyü
# etkiler, uyarıyı SUSTURMAZ. Kullanıcı emir değil sinyal istiyor → varsayılan AÇIK.
SIGNAL_ALERT_ALWAYS = _envb("SIGNAL_ALERT_ALWAYS", True)

# Kill-Switch (Emniyet Kilidi) Sistemi: Son N işlemin ortalama PnL beklentisi
# negatif olduğunda yeni girişleri durduran sistem. False -> Tamamen kapalı.
ENABLE_KILL_SWITCH = _envb("ENABLE_KILL_SWITCH", False)
# Kaç işlemden SONRA değerlendirmeye başla (rolling pencere maxlen=50).
KILL_SWITCH_MIN_TRADES = int(os.getenv("KILL_SWITCH_MIN_TRADES", "30"))
# OTOMATİK RESET (AI_trade'den): kill-switch aktifken rolling expectancy bu eşiğe
# (≥) toparlanırsa kilit kendiliğinden kalkar. Açık pozisyonlar kapanmaya devam
# ettiğinden (kilit yalnızca YENİ girişi bloke eder) pencere güncellenebilir; tümü
# flat olup negatif kalırsa kilit açık kalır (manuel müdahale gerekir — veri yoksa
# devam etmeyi haklı çıkaracak temel yok). 0.0 = başabaşa dönünce reset.
KILL_SWITCH_RESET_EXPECTANCY = float(os.getenv("KILL_SWITCH_RESET_EXPECTANCY", "0.0"))

# Ensemble seçim kipi: first_trigger (varsayılan) | confluence (oylama).
# NOT (ampirik): confluence oylaması backtest'te legacy first_trigger'dan daha kötü
# çıktı (1y weighted_E −43.5 vs −1.8) → varsayılan first_trigger. (docs/EDGE_FINDINGS.md)
ENSEMBLE_MODE = os.getenv("ENSEMBLE_MODE", "first_trigger").lower()

# Katı `confluence` modu (ACTIVE_STRATEGY=confluence): bir sinyal için en az kaç
# bağımsız detektörün hemfikir olması gerektiği. §16.2 (docs/EDGE_FINDINGS.md):
# bağımsız ONAY precision'ı keskinleştirdi (in-sample). 2 = makul taban; gürültü
# tek-detektör tetikleri elenir. KENAR İDDİASI DEĞİL — survivorship-temiz ölçüm şart.
CONFLUENCE_MIN_VOTES = int(os.getenv("CONFLUENCE_MIN_VOTES", "2"))

# ---------------------------------------------------------------------------
# Dizinler
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # repo kökü (nihai/'nin üstü)
# NIHAI_LOG_DIR / NIHAI_DATA_DIR: test süreci üretim logs/ ve data/ dosyalarını
# kirletmesin diye kök conftest.py bunları geçici dizine yönlendirir (BULGU-8).
# Üretimde env yokken davranış aynen eski: ROOT/logs ve ROOT/data.
LOG_DIR = Path(os.getenv("NIHAI_LOG_DIR") or (ROOT / "logs"))
DATA_DIR = Path(os.getenv("NIHAI_DATA_DIR") or (ROOT / "data"))
STATE_FILE = DATA_DIR / "state.json"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Çalışma kipi
# ---------------------------------------------------------------------------
RUN_MODE = os.getenv("RUN_MODE", "LIVE_SIGNALS")  # SIMULATION | BACKTEST | LIVE_SIGNALS
INITIAL_EQUITY = float(os.getenv("INITIAL_EQUITY", "25000"))
WATCHLIST = [s.strip().upper() for s in os.getenv("WATCHLIST", "AAPL,MSFT,SPY").split(",") if s.strip()]

# Madde 2, 16, 27 — Sinyal kipi: hisse mi opsiyon mu?
OPTION_MODE = os.getenv("OPTION_MODE", "stock").lower()  # "stock" | "option" | "both"

# Aktif strateji modu (Madde 1 — olay-bazlı + alternatif setup'lar)
# event_breakout: EMA crossover + 20-bar breakout
# mean_reversion: RSI aşırı + Bollinger Band touch (sideways piyasalar için)
# rsi_divergence: klasik RSI uyumsuzluğu (trend reversal yakalama)
# squeeze: volatilite sıkışması sonrası kırılım
# ensemble: hepsini dene, ilk tetik geçeni al
# VARSAYILAN (ampirik): event_breakout. Modlar 16-sembol sepetinde 1h/1y VE 1h/2y'de
# kıyaslandı (modes_compare.py): event_breakout HER İKİ periyotta da en kararlı ve
# ensemble'dan daha az kayıplıydı (1y −30 / 2y −27 vs ensemble −43 / −56). squeeze 1y'de
# +39 ile en iyiydi ama 2y'de −83'e döndü (overfit). Detay: docs/EDGE_FINDINGS.md.
ACTIVE_STRATEGY = os.getenv("ACTIVE_STRATEGY", "event_breakout").lower()

# Veri sağlayıcısı seçimi (Madde 45)
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yfinance").lower()   # yfinance | tradingview
# TradingView credentials (opsiyonel — boşsa anonim delayed; doluysa real-time)
TV_USERNAME = os.getenv("TV_USERNAME", "")
TV_PASSWORD = os.getenv("TV_PASSWORD", "")

# Alpaca Market Data (gerçek-zamanlı; IEX feed ücretsiz). DATA_PROVIDER=alpaca ile aktif.
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_FEED = os.getenv("ALPACA_FEED", "iex")        # iex (ücretsiz) | sip (paylı)

# --- Process-dışı veri hub'ı (hub_server.py) ---
# DATA_PROVIDER=hub: proje Alpaca'ya bağlanmaz, yerel hub'dan (tek upstream bağlantı) okur.
# Birden çok proje/varyant aynı hub'ı paylaşır → 1 Alpaca bağlantısı + özdeş veri + ~yarı API.
HUB_URL = os.getenv("HUB_URL", "http://127.0.0.1:8799")        # HubProvider buraya sorar
HUB_HOST = os.getenv("HUB_HOST", "127.0.0.1")                  # hub_server dinleme adresi
HUB_PORT = int(os.getenv("HUB_PORT", "8799"))
# Hub'ın GERÇEK kaynağı (asla 'hub' olamaz — kendine referans). Varsayılan canlı stream.
HUB_UPSTREAM_PROVIDER = os.getenv("HUB_UPSTREAM_PROVIDER", "alpaca_stream").lower()

# Telegram (Madde 47 — operatör bildirimi)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Web dashboard
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("WEB_PORT", "8765"))
# Mutasyon uçları (flatten/close/reset) için token. Boşsa yalnızca localhost güvenliği geçerli.
WEB_TOKEN = os.getenv("WEB_TOKEN", "")
# CORS izinli origin'ler. Panel yereldir; '*' + allow_credentials geçersiz/güvensiz kombinasyon
# olduğundan localhost varsayılanı kullanılır. Virgülle ayrılmış env ile genişletilebilir;
# WEB_CORS_ORIGINS="*" verilirse credentials otomatik kapatılır (web_dashboard.py).
WEB_CORS_ORIGINS = [
    o.strip() for o in os.getenv(
        "WEB_CORS_ORIGINS",
        f"http://{WEB_HOST}:{WEB_PORT},http://127.0.0.1:{WEB_PORT},http://localhost:{WEB_PORT}",
    ).split(",") if o.strip()
]

# Tarama yön bar aralığı (HTF). Bar UZUNLUĞU buradan TÜRETİLİR (INTERVAL_MINUTES),
# ayrıca bir yerde saklanmaz — ikinci kopya elle "uyumlu tutulmayı" gerektirirdi.
HTF_INTERVAL = os.getenv("HTF_INTERVAL", "1h")

# Madde 48 farkındalığı — yfinance/TV REST çağrılarını kısa TTL ile cache'le (rate-limit koruması)
DATA_CACHE_TTL_SEC = float(os.getenv("DATA_CACHE_TTL_SEC", "45"))

# Bar aralığı → dakika (cooldown ve Sharpe ölçeği için tek kaynak)
INTERVAL_MINUTES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": 390}


# ---------------------------------------------------------------------------
# OTO-TARAYICI (dinamik watchlist) — sistem evreni kendisi tarar, en iyi
# adayları canlı watchlist'e koyar. config WATCHLIST yalnızca başlangıç tohumu.
# ---------------------------------------------------------------------------
# Açık olduğunda main döngüsü arka planda periyodik tarama yapar ve watchlist'i
# tamamen taramanın en iyi N skoruyla yeniler (top-N dinamik mod).
AUTO_SCREEN_ENABLED = os.getenv("AUTO_SCREEN_ENABLED", "true").lower() in ("1", "true", "yes", "on")
# Evren: nasdaq100 | sp500 | both (birleşik). Birleşik ~500+ sembol → tarama uzun sürer.
AUTO_SCREEN_UNIVERSE = os.getenv("AUTO_SCREEN_UNIVERSE", "both").lower()
# Tarama BİTTİKTEN sonra bir sonraki taramaya kadar boşta beklenen süre (seans içinde
# arka plan thread'i). Gerçek periyot = tarama süresi + bu bekleme. Varsayılan 5 dk.
# AUTO_SCREEN_INTERVAL_MINUTES verilirse o öncelikli (dakika cinsinden daha okunaklı).
_screen_mins = os.getenv("AUTO_SCREEN_INTERVAL_MINUTES")
AUTO_SCREEN_INTERVAL_HOURS = (
    float(_screen_mins) / 60.0 if _screen_mins
    else float(os.getenv("AUTO_SCREEN_INTERVAL_HOURS", str(5.0 / 60.0)))
)
# Watchlist'e konacak en iyi N sembol (top-N). 0 veya negatif = HEPSİ modu:
# backtest sıralaması/filtresi yapılmaz, watchlist = tüm evren (herkes taranır).
AUTO_SCREEN_TOP_N = int(os.getenv("AUTO_SCREEN_TOP_N", "20"))
# Tarama backtest parametreleri
AUTO_SCREEN_INTERVAL = os.getenv("AUTO_SCREEN_INTERVAL", "1h")   # bar aralığı
AUTO_SCREEN_PERIOD = os.getenv("AUTO_SCREEN_PERIOD", "6mo")
# Sıralama metriği: expectancy | sharpe | total_return_pct | win_rate_pct
AUTO_SCREEN_SORT = os.getenv("AUTO_SCREEN_SORT", "expectancy").lower()
# Eleme filtreleri — gürültüyü/aşırı-uydurmayı dışla
AUTO_SCREEN_MIN_TRADES = int(os.getenv("AUTO_SCREEN_MIN_TRADES", "5"))
AUTO_SCREEN_MIN_EXPECTANCY = float(os.getenv("AUTO_SCREEN_MIN_EXPECTANCY", "0"))
# 0=filtre kapalı. Varsayılan 0: ruin filtresi MC>0 gerektirir (MC varsayılanı 0),
# tutarlı olsun diye kapalı başlar. Ruin filtresi için: AUTO_SCREEN_MC=1000 + bu>0.
AUTO_SCREEN_MAX_RUIN = float(os.getenv("AUTO_SCREEN_MAX_RUIN", "0"))
# A5 — DÜŞÜK-GÜVEN KADEMELİ BOYUTLAMA (AI_trade "AZ_VERI" deseni). Backtest geçmişi
# ince olan (az işlem) semboller binary elenmek yerine KÜÇÜK pozisyonla tutulur:
# filtre eşiğini (AUTO_SCREEN_MIN_TRADES) geçen ama "rahat" eşiğin altında kalan
# (trades < CONFIDENCE_COMFORT_TRADES) semboller risk × CONFIDENCE_LOW_MULT alır.
# risk_mult watchlist meta'sına HER ZAMAN yazılır (bilgi amaçlı); yalnızca boyutlamaya
# UYGULANMASI bu bayrakla kapılıdır → varsayılan kapalı = davranış değişmez.
CONFIDENCE_SIZING_ENABLED = _envb("CONFIDENCE_SIZING", False)
CONFIDENCE_COMFORT_TRADES = int(os.getenv("CONFIDENCE_COMFORT_TRADES", "10"))
CONFIDENCE_LOW_MULT = float(os.getenv("CONFIDENCE_LOW_MULT", "0.5"))
# Olasılık-bazlı boyutlama: meta_gate P(win)'i prob_sizing.prob_to_size_mult ile RİSK
# çarpanına çevirip kademeli boyutlar (ikili kapı yerine). ⚠️ VARSAYILAN KAPALI —
# yalnızca train_meta.py'ın OOS kapı simülasyonu modelin gerçek kenarını kanıtladıktan
# sonra açın (canlı veride confidence şu an isabetle TERS korele; körü körüne açmak
# kötü sinyalleri büyütür). META_GATE'ten bağımsız çalışır ama modeli yükler.
# CANLI GİRİŞ ÇAPASI (Tur-39) — sinyal fiyatı `df_htf["close"].iloc[-1]`'dir ve
# HTF_INTERVAL=1h iken ~59 dk BAYAT olabilir; pozisyon 1m/kotasyonla yönetilir. Ölçüldü
# (510 sanal işlem): işlemlerin %46'sında MFE tam 0,00 (girilen fiyat piyasada YOK) ve
# GAP_STOP sınıfı 68/68 aleyhte, −299.438 $ = sistemin tüm net zararından büyük.
# VARSAYILAN AÇIK: bu bir strateji tercihi değil, karar yolundaki bir ÖLÇÜM kusurunun
# onarımıdır (Tur-37'nin uydurma-Greek onarımıyla aynı sınıf). `=off` eski (bayat-fiyat)
# davranışa döndürür — A/B ölçümü için bilerek bırakıldı.
LIVE_ENTRY_ANCHOR = _envb("LIVE_ENTRY_ANCHOR", True)
PROB_SIZING_ENABLED = _envb("PROB_SIZING", False)
PROB_SIZING_P_LO = float(os.getenv("PROB_SIZING_P_LO", "0.45"))
PROB_SIZING_P_HI = float(os.getenv("PROB_SIZING_P_HI", "0.60"))
PROB_SIZING_FLOOR = float(os.getenv("PROB_SIZING_FLOOR", "0.25"))
# B7/B8 — HİBRİT TARAYICI SKORU: backtest metriği + canlı "kurulum kalitesi" skoru
# (setup_score.py; SPY'a görel güç dahil). Açıkken sıralama z(backtest)+w·z(setup)
# ile yapılır. ⚠️ kurulum-skoru ağırlıkları el-ayarı → overfit riski; açmadan önce
# walk-forward ile doğrula. VARSAYILAN KAPALI = sıralama yalnızca backtest (mevcut davranış).
SCREEN_HYBRID_ENABLED = _envb("SCREEN_HYBRID", False)
SCREEN_HYBRID_SETUP_WEIGHT = float(os.getenv("SCREEN_HYBRID_SETUP_WEIGHT", "1.0"))
SCREEN_RS_BENCHMARK = os.getenv("SCREEN_RS_BENCHMARK", "SPY")
# B8 — görel-güç ufukları İŞLEM GÜNÜ cinsinden (interval'den bağımsız anlamlı momentum).
# Klasik momentum 1–3 ay; auto_screener bunları AUTO_SCREEN_INTERVAL'e göre bar'a çevirir.
SCREEN_RS_SHORT_DAYS = int(os.getenv("SCREEN_RS_SHORT_DAYS", "21"))   # ~1 ay
SCREEN_RS_LONG_DAYS = int(os.getenv("SCREEN_RS_LONG_DAYS", "63"))     # ~3 ay
# Kısa "aşırı-uzama/geri-çekilme tazeliği" ufku (gün) — RS'ten ayrı, KISA tutulur.
SCREEN_RUNUP_DAYS = int(os.getenv("SCREEN_RUNUP_DAYS", "3"))
AUTO_SCREEN_MC = int(os.getenv("AUTO_SCREEN_MC", "0"))  # Monte Carlo iter (ruin filtresi için >0 gerekir)
# Semboller arası bekleme (sn) — yfinance ban önleme (Madde 48). PARALEL modda
# (AUTO_SCREEN_WORKERS>1) bu, her worker'ın bir sembolü bitirdikten sonra beklemesidir.
AUTO_SCREEN_THROTTLE = float(os.getenv("AUTO_SCREEN_THROTTLE", "0.5"))
# Paralel tarama worker sayısı. ÖLÇÜLDÜ: bir sembolün backtest'i ~1.2 sn ağ +
# ~7.5 sn HESAP (CPU). Yani iş CPU-AĞIRLIKLIDIR; gerçek hızlanma çoklu ÇEKİRDEK
# (process) ile gelir — thread GIL'e takılır (ölçümde 1.0x). ~600 sembollük tarama
# tek çekirdekte ~75 dk, 8 çekirdekte ~10 dk. 1 = eski seri davranış.
# Varsayılan: mevcut mantıksal çekirdek sayısı (en fazla 16).
AUTO_SCREEN_WORKERS = max(1, int(os.getenv("AUTO_SCREEN_WORKERS", str(min(16, (os.cpu_count() or 4))))))
# Paralellik motoru:
#   • "process" (VARSAYILAN): gerçek çoklu-çekirdek; CPU-ağırlıklı backtest için doğru
#     seçim. Windows'ta spawn → her worker stack'i bir kez import eder (~2 sn amorti).
#   • "thread": GIL nedeniyle CPU-bound işte hızlandırmaz; yalnızca saf I/O (örn.
#     sağlayıcı process-güvenli değilse) veya hata ayıklama için.
AUTO_SCREEN_EXECUTOR = os.getenv("AUTO_SCREEN_EXECUTOR", "process").lower()
# Tarama veri sağlayıcısı: toplu/paralel tarama için hub veya alpaca önerilir
# (eşzamanlı isteklere dayanıklı, düşük gecikme). yfinance de çalışır ama paralelde
# rate-limit'e girebilir; tradingview her sembolde yeniden bağlanır (yavaş, paralel
# kullanma). Canlı sinyal DATA_PROVIDER'dan bağımsızdır.
AUTO_SCREEN_PROVIDER = os.getenv("AUTO_SCREEN_PROVIDER", "yfinance").lower()
# Her taramada sıralamayı logs/auto_screen_<ts>.csv'ye yaz (operatör görünürlüğü)
AUTO_SCREEN_WRITE_CSV = os.getenv("AUTO_SCREEN_WRITE_CSV", "true").lower() in ("1", "true", "yes", "on")
# Dinamik watchlist kalıcılık dosyası
WATCHLIST_FILE = DATA_DIR / "watchlist.json"


# ---------------------------------------------------------------------------
# İKİ-KATMAN PARALEL TARAMA (dual-tier)
# En iyi N sembol "odak" katmanında TradingView ile, geri kalan tüm evren
# "geniş" katmanında yfinance ile PARALEL taranır. Ortak portföy/state, eşzamanlı
# yazıma karşı tek bir kilitle (PORTFOLIO_LOCK) korunur.
# Açıkken: oto-tarayıcı tüm evreni backtest'leyip SIRALAR (top-N kesmez); canlı
# döngü yeni-sinyal taramasını iki paralel thread'e devreder, ana thread yalnızca
# pozisyon yönetimi yapar.
# ---------------------------------------------------------------------------
DUAL_TIER_ENABLED = os.getenv("DUAL_TIER_ENABLED", "false").lower() in ("1", "true", "yes", "on")
# Odak katmanı: en iyi N sembol + hangi sağlayıcı
FOCUSED_TOP_N = int(os.getenv("FOCUSED_TOP_N", "100"))
FOCUSED_PROVIDER = os.getenv("FOCUSED_PROVIDER", "tradingview").lower()
# Geniş katman: geri kalan evren + hangi sağlayıcı
BROAD_PROVIDER = os.getenv("BROAD_PROVIDER", "yfinance").lower()
# Katman içi semboller arası bekleme (sn) — ban önleme (Madde 48)
FOCUSED_THROTTLE = float(os.getenv("FOCUSED_THROTTLE", "0"))
BROAD_THROTTLE = float(os.getenv("BROAD_THROTTLE", "0.3"))
# Bir tam tur bitince katmanın bir sonraki tura başlamadan beklemesi (sn)
TIER_CYCLE_GAP = float(os.getenv("TIER_CYCLE_GAP", "5"))

# ---------------------------------------------------------------------------
# CANLI TARAMA PARALELLİĞİ (tek-katman mod)
# DUAL_TIER kapalıyken canlı döngü watchlist'i tarar. Watchlist büyükse (örn.
# HEPSİ modu ~500+ sembol) SERİ tarama 60sn'lik döngüye yetişmez — ölçüldü:
# 520 sembol × ~215ms ≈ 112sn. İş I/O-AĞIRLIKLI (her sembol hub'dan bar+quote
# çeker), bu yüzden THREAD havuzu ağ beklerken GIL'i serbest bırakıp gerçekten
# paralelleşir (backtest'in CPU-bound process havuzunun tersine). Her worker thread
# KENDİ provider örneğini kullanır (thread-local); ortak portföy/state PORTFOLIO_LOCK
# ile korunur. 1 = eski seri davranış.
LIVE_SCAN_WORKERS = max(1, int(os.getenv("LIVE_SCAN_WORKERS", "8")))


# ---------------------------------------------------------------------------
# RİSK YAPISI — Madde 2, 9, 10
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskConfig:
    # Madde 2 — Tek işlem riski
    per_trade_risk_pct: float = field(
        default_factory=lambda: float(os.getenv("PER_TRADE_RISK_PCT", "0.02")))  # %2 (uzun opsiyonda primin tamamı)
    per_trade_risk_pct_directional: float = 0.015 # %1.5 tek bacak yönsel
    # Madde 9 — Kasa-bazlı adaptif risk
    drawdown_throttle_threshold: float = 0.20     # ATH'den %20 düşüşte
    drawdown_risk_multiplier: float = 0.50        # riski yarıya in
    # Madde 10 — Günlük freni
    daily_max_loss_pct: float = field(
        default_factory=lambda: float(os.getenv("DAILY_MAX_LOSS_PCT", "0.10")))  # %10 günlük kasa kaybı → STOP
    # 0/negatif = DEVRE DIŞI (consecutive_stops_cooldown ile aynı kural). Canlı kurulumda
    # .env içinde DAILY_MAX_LOSS_PCT=0 ile kapatıldı (kullanıcı kararı, 2026-08-31).
    # Üst üste N kayıp → soğutma (yeni giriş donar). 0/negatif = DEVRE DIŞI.
    # Sinyal-only kullanımda SIGNAL_ALERT_ALWAYS uyarıyı zaten risk-bağımsız gönderir;
    # bu yalnızca sanal portföyün poz açmasını etkiler. (CONSECUTIVE_STOPS_COOLDOWN env)
    consecutive_stops_cooldown: int = field(
        default_factory=lambda: int(os.getenv("CONSECUTIVE_STOPS_COOLDOWN", "3")))
    # Madde 11 — Zaman-bazlı stop
    time_stop_bars: int = 12                      # (YEDEK yol) zaman okunamazsa güncelleme sayısı
    # ZAMAN-bazlı stop (Tur-39 birim onarımı). `time_stop_bars` canlıda BAR değil
    # GÜNCELLEME sayıyordu (~60 sn'de bir) → 12 "bar" 12 DAKİKA ediyordu, oysa stop/target
    # geometrisi HTF_INTERVAL=1h barından türer ve 2R hedefi saatler ister. ÖLÇÜLDÜ
    # (510 sanal işlem): 405'i TIME_STOP ile ~17 dk'da kapandı, kazananları hedefin çok
    # altında +0,14R'de kesildi — hedef ile ufuk AYNI işlemi tarif etmiyordu.
    # Eşik artık AÇIK DAKİKADIR ve seans içi kalacak şekilde seçilmiştir.
    time_stop_minutes: int = field(
        default_factory=lambda: int(os.getenv("TIME_STOP_MINUTES", "90")))
    # SEANS SONU DÜZLEŞME: kapanışa bu kadar dakika kala açık pozisyon kapatılır →
    # gecelik taşıma ve GERÇEK gap riski YOK (sistem "seans içi" kimliğini korur).
    # 0 = kapalı (gecelik taşımaya izin verir; ayrı bir gecelik boyut/hedge kuralı ister).
    flat_before_close_min: int = field(
        default_factory=lambda: int(os.getenv("FLAT_BEFORE_CLOSE_MIN", "10")))
    # Breakeven stop: fiyat lehte +be_trigger_r × (başlangıç risk mesafesi) gidince
    # stop'u girişe (+ be_buffer_r tampon) çek. Look-ahead-güvenli (çıkış kontrolünden
    # SONRA, yalnızca sonraki barlar için). WR'ı kazananı KESMEDEN yükseltmenin tek yolu.
    enable_breakeven: bool = field(default_factory=lambda: _envb("BREAKEVEN", False))
    be_trigger_r: float = field(default_factory=lambda: float(os.getenv("BE_TRIGGER_R", "1.0")))
    be_buffer_r: float = field(default_factory=lambda: float(os.getenv("BE_BUFFER_R", "0.0")))
    # Trailing stop: stop'u en iyi-lehte fiyatın trail_r×(başlangıç risk) gerisinde sürükle.
    # trail_drop_target → sabit hedefi kaldır (kazananı koştur). Amaç: WR'ı sabit tutup
    # kazançları büyüterek payoff/PF'i yükseltmek. Look-ahead-güvenli (çıkıştan sonra).
    enable_trailing: bool = field(default_factory=lambda: _envb("TRAIL", False))
    trail_r: float = field(default_factory=lambda: float(os.getenv("TRAIL_R", "2.0")))
    trail_drop_target: bool = field(default_factory=lambda: _envb("TRAIL_DROP_TARGET", True))
    # A1 — KADEMELİ trailing + kâr-tabanı kilidi (AI_trade'den uyarlandı).
    # trail_tiered: kâr arttıkça trail mesafesini kâr-katmanına (R-cinsinden) göre SIKILAŞTIR.
    #   AI_trade ATR-çarpanını PnL%-katmanına bağlıyordu; burada R-uzayına taşındı (entry−initial_stop
    #   = 1R). Trail yalnızca lehe ratchet ettiği için katmanlı SIKILAŞMA gerçek davranış değiştirir
    #   (gevşeme zaten stop'u geri almaz). trail_tiers boşsa düz trail_r kullanılır.
    trail_tiered: bool = field(default_factory=lambda: _envb("TRAIL_TIERED", False))
    trail_tiers: tuple[tuple[float, float], ...] = field(
        default_factory=lambda: _parse_tiers(os.getenv("TRAIL_TIERS", "1.5:2.0,3.0:1.5,5.0:1.0")))
    # Kâr-tabanı kilidi (AI_trade Stage-3): kâr trail_lock_trigger_r'yi aşınca stop'u
    # EN AZ entry + trail_lock_floor_r×risk'e çek (kazanılmış kârı kilitle). 0 = kapalı.
    trail_lock_trigger_r: float = field(default_factory=lambda: float(os.getenv("TRAIL_LOCK_TRIGGER_R", "3.0")))
    trail_lock_floor_r: float = field(default_factory=lambda: float(os.getenv("TRAIL_LOCK_FLOOR_R", "1.0")))
    # A2 — MFE give-back (zafer-kaçırma) koruması (AI_trade'den uyarlandı, R-UZAYINDA).
    # AI_trade YÜZDE eşik kullanıyordu (kripto yüksek-vol); bu sistemin TÜM çıkışları
    # ATR/R-tabanlı (breakeven, trailing, A1) → tutarlılık ve volatiliteye-uyum için
    # give-back da R-cinsinden: MFE (tepe lehte hareket) arm_r'yi aştıktan SONRA tepeden
    # geri-veriş giveback_r'yi geçerse anında çık (1R = entry − başlangıç stop).
    # Look-ahead-güvenli: yalnızca o ana dek görülen high_water + kapanış fiyatı kullanılır.
    enable_mfe_giveback: bool = field(default_factory=lambda: _envb("MFE_GIVEBACK", False))
    mfe_giveback_arm_r: float = field(default_factory=lambda: float(os.getenv("MFE_GIVEBACK_ARM_R", "1.5")))
    mfe_giveback_r: float = field(default_factory=lambda: float(os.getenv("MFE_GIVEBACK_R", "1.0")))


# ---------------------------------------------------------------------------
# PORTFÖY TAVANLARI — Madde 14
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioCaps:
    max_open_positions: int = 50
    max_gross_risk_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_GROSS_RISK_PCT", "1.00")))              # toplam brüt risk tavanı (risk $ / equity)
    # Madde 14 — net yönsel maruziyet = (uzun − kısa) NOMİNAL / equity.
    # Birim NOMİNALDİR (risk değil): risk-bazlı boyutlamada dar stop büyük nominal
    # üretir, bu yüzden değer equity'nin katı olarak okunur. Opsiyonda nominal,
    # prim×100×kontrat ile yaklaşıklanır (delta-dolar değil; yön işareti doğrudur).
    max_net_directional_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_NET_DIRECTIONAL_PCT", "2.0")))          # net yönsel nominal ≤ 2.0×equity
    max_per_sector_positions: int = 3             # Madde 12 korelasyon
    # Cap-aware boyutlama: büyüklük tavanı (buying power / net yönsel / brüt risk)
    # aşılırsa pozisyonu REDDETMEK yerine tavana SIĞACAK şekilde küçült. Küçük hesapta
    # (örn. $25k) yüksek-fiyatlı sembollerde risk-bazlı nominal cap'i tek pozisyonda
    # doyurup tüm sonraki sinyalleri bloke ediyordu; bu, hedeften DÜŞÜK riskle (her
    # zaman güvenli yön) sığacak en büyük pozisyonu açtırır. False → eski reddet davranışı.
    cap_aware_sizing: bool = True
    # Küçültme sonrası nominal bu eşiğin altındaysa "toz" pozisyonu atla (0 = kapalı).
    min_position_notional: float = 0.0
    # Pozisyon BAŞINA nominal tavanı ($). 0 = kapalı. Büyüklük tavanı olduğu için
    # cap-aware boyutlama pozisyonu reddetmez, tavana sığacak şekilde küçültür →
    # aynı sermaye daha çok eşzamanlı pozisyona dağılır.
    max_position_notional: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_NOTIONAL", "0")))


# ---------------------------------------------------------------------------
# PORTFÖY GREEK LİMİTLERİ — Madde 27
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GreekCaps:
    max_net_delta: float = 200.0                  # ≈ 200 hisse eşdeğeri
    max_net_theta: float = -50.0                  # günlük max theta kaybı
    max_net_vega: float = 300.0                   # vega maruziyeti
    max_net_gamma: float = 30.0                   # gamma sınırı


# ---------------------------------------------------------------------------
# REGÜLASYON — Madde 13, 23
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Regulation:
    # Reg T (Fed) — GİRİŞ marjı; `risk_manager._magnitude_cap_units` (a) bunu uygular.
    reg_t_initial_margin: float = 0.50            # %50 başlangıç
    # FINRA Rule 4210 SÜRDÜRME marjı (%25) BURADA DEĞİL: bot onu uygulamıyor ve
    # uygulamadığı bir sayıyı taşımak, uygulanıyor sanılmasına yol açıyordu (Tur-50).
    # Sürdürme marjı PORTFÖY düzeyi brüt maruziyet ölçer; bu depoda brüt maruziyet
    # KATMAN düzeyinde bile birleştirilemiyor (DUAL_TIER ayrı RiskManager'lar kullanır
    # ve ortak `trades.csv`'den portföy brütü GERİ KURULAMAZ). Yani kural yalnız
    # uygulanmıyordu değil, İHLALİ DE ÖLÇÜLEMİYORDU. PDT notundaki ile aynı duruş:
    # hesap düzeyinde broker (Alpaca) uygular; botun ön-kontrolü Reg T ile sınırlıdır.
    # PDT (Madde 10, 23)
    # NOT: PDT ön-kontrolü yeniden AÇIK (kullanıcı kararı, 2026-08-31). Kapatmak:
    # PDT_ENABLED=0. Açıkken day_trades_count sayılır, panoda gösterilir VE (Tur-62)
    # `RiskManager._entry_blocked` üzerinden yeni pozisyonu ENGELLER. Fren yalnız
    # NLV < pdt_threshold_equity iken devrededir; canlı hesapta (294k$) atıldır.
    # Broker (Alpaca) PDT'yi hesap düzeyinde yine de uygulayabilir.
    pdt_enabled: bool = _envb("PDT_ENABLED", True)
    pdt_threshold_equity: float = field(
        default_factory=lambda: float(os.getenv("PDT_THRESHOLD_EQUITY", "25000.0")))
    pdt_lookback_days: int = field(
        default_factory=lambda: int(os.getenv("PDT_LOOKBACK_DAYS", "5")))
    # BULGU-62-4: varsayılan **10**du, ama kuralın adı ve `risk_manager` yorumu FINRA'nın
    # gerçek değerini söylüyordu ("5 iş günü içinde max 3"). Sabit, adını taşıdığı kuralla
    # çelişiyordu ve hiçbir yerde 10'un gerekçesi yoktu (Tur-61'in belge↔kod sınıfı).
    # Kural değeri bir TERCİH değildir → FINRA'nın 3'üne çekildi. Canlı hesapta BİT-DENK
    # (NLV eşiğin çok üstünde, kapı hiç ulaşılmıyor). Gevşetmek isteyen env ile yapar.
    pdt_max_day_trades_under_threshold: int = field(
        default_factory=lambda: int(os.getenv("PDT_MAX_DAY_TRADES_UNDER_THRESHOLD", "3")))
    # T+1 takas (28 Mayıs 2024'ten beri)
    settlement_days: int = field(
        default_factory=lambda: int(os.getenv("SETTLEMENT_DAYS", "1")))
    # Madde 30 — SSR (Reg SHO Rule 201): dayanak önceki kapanışa göre %10 düşerse tetiklenir
    ssr_trigger_drop_pct: float = field(
        default_factory=lambda: float(os.getenv("SSR_TRIGGER_DROP_PCT", "0.10")))
    # Madde 53 — Wash-sale: zararlı satıştan sonra 30 gün içinde aynı menkul
    wash_sale_days: int = field(
        default_factory=lambda: int(os.getenv("WASH_SALE_DAYS", "30")))


# ---------------------------------------------------------------------------
# OPSİYON FİLTRELERİ — Madde 21, 28
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OptionFilters:
    min_open_interest: int = 500
    min_daily_volume: int = 100
    max_spread_pct_of_mid: float = 0.10           # makas / mid < %10
    # DTE PENCERESİ BURADA DEĞİL: vade seçimi `OptionStrategyConfig.dte_min/dte_max`
    # ile yapılır (`risk/option_strategy.py`). Burada bir `min_dte/max_dte` çifti daha
    # DURUYORDU (5/60) ve hiçbir kod onu OKUMUYORDU — üstelik yürürlükteki pencereyle
    # (30/45) ÇELİŞİYORDU: operatör "0DTE'ye dokunma" yazan alanı sıkılaştırdığında
    # hiçbir şey değişmezdi. Tur-50'de silindi; bu satır işaretçi olarak kaldı.
    # Zincir filtresi (`filter_liquid_chain`) TEK bir vadenin zinciri üzerinde çalışır,
    # yani DTE kırpması oraya kavramsal olarak da ait değildir.
    # Madde 28 — vade yakınında kapat/roll
    close_before_expiry_dte: int = 5


# ---------------------------------------------------------------------------
# STRATEJİ FİLTRELERİ — Madde 8, 16, 17, 18
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyFilters:
    min_rr_ratio: float = field(default_factory=lambda: float(os.getenv("MIN_RR_RATIO", "1.2")))
    # Yalnızca-long: SHORT sinyalleri bastırılır. Sürekli-yükselen evrenlerde (mega-cap)
    # shortlar yapısal olarak para kaybeder; backtest'te doğrulanırsa açılır. (LONG_ONLY env)
    enable_long_only: bool = field(default_factory=lambda: _envb("LONG_ONLY", False))
    # Tur-97 quant_trend kısa bacağı. Varsayılan KAPALI: vektörel keşifte (14 piyasa, 2018-26) kısa
    # bacak hisse piyasalarında her yerde zarar yazdı (pozitif sürüklenme); FX/emtia araştırması açar.
    quant_allow_short: bool = field(default_factory=lambda: _envb("QUANT_KISA", False))
    # Trend-takip stop/hedef ATR çarpanları (env-ayarlanabilir, çıkış optimizasyonu için).
    trend_stop_atr: float = field(default_factory=lambda: float(os.getenv("TREND_STOP_ATR", "1.5")))
    trend_target_atr: float = field(default_factory=lambda: float(os.getenv("TREND_TARGET_ATR", "3.0")))
    # Madde 18 — kalite eşiği. NOT (ampirik): eşiği 2.5'e çekmek backtest'te (16-sembol
    # sepeti, 1h/1y) kâra DOKUNMADI veya kötüleştirdi → varsayılan 1.0'da bırakıldı.
    # A/B için: MIN_QUALITY env.  (bkz. docs/EDGE_FINDINGS.md)
    min_signal_quality_score: float = field(
        default_factory=lambda: float(os.getenv("MIN_QUALITY", "1.0")))
    adx_trend_threshold: float = 20.0             # Madde 16 (trend zayıfsa risk %50 ↓)
    # Edge filtreleri (trend-takip kurulumlarına). UYARI (ampirik): HER İKİSİ de
    # backtest'te weighted-expectancy'yi DÜŞÜRDÜ (trend-align −95, adx-floor −36 vs
    # legacy −1.8) → varsayılan KAPALI. Açmadan önce kendi sepetinde A/B yap.
    enable_trend_alignment_gate: bool = field(
        default_factory=lambda: _envb("GATE_TREND_ALIGN", False))
    enable_adx_floor_gate: bool = field(
        default_factory=lambda: _envb("GATE_ADX_FLOOR", False))
    rsi_chase_long: float = 78.0                  # RSI > X iken long kovalama yok (gate açıksa)
    rsi_chase_short: float = 22.0                 # RSI < X iken short kovalama yok (gate açıksa)
    # Çapraz-varlık VIX rejim kapısı (alternatif veri). VIX > eşik (panik/yüksek-korku)
    # iken yeni giriş bastırılır — tez: kırılım/trend sinyalleri yüksek-VIX'te başarısız.
    # Look-ahead-güvenli: karar gününden ÖNCEki VIX kapanışı kullanılır. UYARI: dayanıklı
    # doğrulanmadan açma (vix_test.py ile iki periyotta A/B). Varsayılan KAPALI.
    enable_vix_regime_gate: bool = field(default_factory=lambda: _envb("VIX_GATE", False))
    vix_high_threshold: float = field(default_factory=lambda: float(os.getenv("VIX_HIGH", "28")))
    # Meta-labeling kapısı (ikincil ML modeli — meta_gate.py). Birincil strateji yön üretir;
    # bu kapı 'kazanma olasılığı < eşik' sinyalleri değerlendirir. KİP (META_GATE env):
    #   "off"     → kapı hiç çağrılmaz (varsayılan; ACI DERS #4).
    #   "shadow"  → GÖLGE: p_win hesaplanır + metadata/reason yazılır ama sinyal ASLA
    #               bastırılmaz. Canlı gate_sim kanıtı GATE_HURTS derken enforce açılmamalı
    #               AMA kanıt döngüsü (pwin logu → gate_sim/pwin_calibration) yaşamalı.
    #   "enforce" → eski META_GATE=true davranışı: p_win < eşik sinyali WAIT'e çevirir.
    # Model yoksa kapı her kipte sessizce PASİF kalır. Eşik model dosyasından gelir;
    # META_GATE_THRESHOLD ile override edilebilir (0=model değeri).
    meta_gate_mode: str = field(default_factory=_meta_gate_mode)
    meta_gate_threshold: float = field(default_factory=lambda: float(os.getenv("META_GATE_THRESHOLD", "0")))
    atr_explosion_multiplier: float = 2.5         # Madde 17 (anlık ATR > X*ortalama → giriş yok)
    iv_rank_high_threshold: float = 50.0          # IV yüksek → prim sat favori
    iv_rank_low_threshold: float = 30.0           # IV düşük → prim al favori
    # IV bonusunun REJİMİ (Tur-31).
    # ⚠️ TUR-51: bu yorumun eski hâli 2026-08-13 ölçümünü aktarıyordu ("ilişki MONOTON,
    # en iyi dilim [0,10) t=+5,75, bant [30,50] tabandan KÖTÜ t=−7,35" → "kural TERS").
    # **O ÖLÇÜM ÇÜRÜTÜLDÜ** (`docs/EDGE_FINDINGS.md` §28): vekil etiket
    # `hv_current.shift(-W)` sabit-ufuklu DEĞİLDİ — `hv_current` geriye-bakan 30 günlük
    # RV'dir (kor. 0,99504 / 603 sembol) ve dolt serisi düzensiz olduğu için kaydırma
    # hedeflenen 42-44 takvim günü yerine ortanca 70 gün (std 34,0) ileri gidiyordu.
    # AYNI satırlarda tek değişken etiketken: vekil bant t=−2,29 (monoton) ↔ fiyattan
    # hesaplanan temiz ileri RV t=+0,61 (monotonluk YOK); iki etiketin korelasyonu 0,7177.
    # Düzenli ızgarada nihai ölçüm: bant gün-bazlı **t=−0,17** (n=20.910, 581 PIT sembol).
    #   "band"  → low<=rank<=high ise +0.5   (Tur-51 ÖNCESİ varsayılan)
    #   "low"   → rank <= low ise +0.5   ← 2026-08-13'te "ölçümün imlediği yön" diye
    #             yazılmıştı; **Tur-51 bu desteği GERİ ÇEKTİ** (temiz [0,10) t=+0,94, anlamsız)
    #   "off"   → bonus hiç verilmez   **(TUR-51'DEN İTİBAREN VARSAYILAN)**
    # ÜÇ kipin de ölçülmüş desteği YOKTUR: bant bilgi taşımıyor, ters de değil (t=−0,17).
    # Varsayılan `off`a çekildi çünkü bu bir "ters kenar" iddiası DEĞİL, KANITSIZ bir
    # bileşenin skordan çıkarılmasıdır — Tur-31'in "ölçüm yokluğu ödüllendirilmez"
    # disiplininin devamı. Ölçülen şey hâlâ IV−RV farkıdır, **P&L değildir** (ACI DERS #3):
    # yani `off` bir kenar iddiası değil, iddiasızlıktır. `IV_REGIME_BONUS=band` eski
    # davranışa döndürür (A/B için); kip DEĞİŞTİRMEK skorun tavanını 0.5 düşürür ve
    # eşik civarındaki adayları eler — etkisi `breakdown["iv_regime"]`den izlenebilir.
    iv_regime_bonus_mode: str = field(
        default_factory=lambda: (os.getenv("IV_REGIME_BONUS", "off") or "off").strip().lower())
    reentry_cooldown_bars: int = 2                # Madde 7  — gevşetildi 6→2
    # Cooldown'un "bar" uzunluğu BURADA DEĞİL: `execution/state.py` onu
    # `INTERVAL_MINUTES[HTF_INTERVAL]` ile TÜRETİR. Burada duran `htf_bar_minutes=60`
    # hiçbir yerden okunmuyordu ve HTF_INTERVAL env'den değiştiğinde sessizce
    # YANLIŞLAŞIYORDU; üstelik bir yorum operatörden ikisini elle "uyumlu tutmasını"
    # istiyordu. Türetilen bir değerin ikinci bir kopyası ayar değil, gelecekteki bir
    # tutarsızlıktır (Tur-50).

    @property
    def enable_meta_gate(self) -> bool:
        """Geriye uyum: kapı herhangi bir kipte etkin mi? (mode != "off").

        Eski okuyucular bool bekler; kaynak-of-truth artık meta_gate_mode. Alan yerine
        property: mode ile bayrak asla tutarsız düşemez (tek türetim noktası)."""
        return self.meta_gate_mode != "off"


# ---------------------------------------------------------------------------
# PİYASA YAPISI — Madde 24 (Halt/LULD), 30 (SSR)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketStructure:
    # Madde 24 — Halt/LULD sezgisel tespit (sinyal-only; gecikmeli veriyle yaklaşık)
    halt_frozen_bars: int = 3            # son N bar aynı fiyat + sıfır hacim → olası halt
    luld_move_pct: float = 0.10          # tek barda |getiri| > %10 → LULD/anormal etiketi
    # Madde 25 — Kurumsal aksiyon: son N iş günü içinde split olduysa fiyat sıçraması "kayıp" değil
    corporate_action_lookback_days: int = 5


# ---------------------------------------------------------------------------
# SAHA OPERASYONU — Madde 42
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SafetyLimits:
    max_signals_per_minute: int = 5
    max_signals_per_hour: int = 60
    max_signals_per_symbol_open: int = 2          # Madde 12 kümelenme


# ---------------------------------------------------------------------------
# BACKTEST — Madde 32, 50, 51 (komisyon, slipaj, sentetik opsiyon)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BacktestConfig:
    stock_commission_per_trade: float = 0.0      # ücretsiz broker varsayımı
    stock_slippage_pct: float = 0.001            # %0.1 hisse
    option_slippage_pct: float = 0.03            # %3 opsiyon (geniş makas)
    risk_free_rate: float = 0.05
    # Sentetik opsiyon backtest (tarihsel zincir yok → Black-Scholes ile yeniden fiyatla)
    option_bt_entry_delta: float = 0.50          # giriş hedef delta (ATM)
    option_bt_iv: float = 0.30                   # sabit IV (dinamik IV kapalıysa fallback)
    option_bt_dte_entry: int = 35                # giriş DTE'si
    option_bt_close_dte: int = 5                 # vade yakını kapat (Madde 28)
    # Dinamik IV: tarihsel zincir yokken IV'yi dayanağın GERÇEKLEŞEN volatilitesinden
    # türet ve her bar güncelle. Böylece IV crush/expansion vega P&L'ini etkiler (long
    # opsiyonun gerçek riski); sabit IV bunu hiç göstermiyordu. False → eski sabit IV.
    option_bt_dynamic_iv: bool = True
    option_bt_iv_lookback: int = 20              # gerçekleşen vol için bar penceresi
    option_bt_iv_risk_premium: float = 1.15      # IV ≈ realized × VRP (genelde >1)
    option_bt_iv_floor: float = 0.10             # alt sınır
    option_bt_iv_ceiling: float = 1.50           # üst sınır (şişmiş IV kırpması)
    # BULGU-59-2 — INTRA-BAR SIRA POLİTİKASI (`position_manager.check_exit` okur).
    #   "open_evidence" (varsayılan): bar AÇILIŞI iki yönde de sıra kanıtıdır. Açılış
    #       stop'un ötesindeyse GAP_STOP (eskiden de öyleydi), HEDEFİN ötesindeyse
    #       GAP_TARGET (yeni). İkincisi eskiden STOP yazıyordu; bu kötümserlik değil
    #       YANLIŞTI — sıra belirsiz değil, BELİRLENMİŞTİ ve ters okunuyordu.
    #   "stop_first": Tur-58 davranışı (yalnız stop gap'i okunur) — A/B ve geri alma için.
    # Açılış İKİSİNİN ARASINDAYKEN iki seviye de deliniyorsa sıra gerçekten belirsizdir;
    # her iki politikada da stop-önce (kötümser) varsayılır ve `intrabar_ambiguous` sayacı
    # artar (maruziyet ölçülür — 1m çözücü bağlıysa gerçek sıra sorulur).
    intrabar_policy: str = field(
        default_factory=lambda: (os.getenv("INTRABAR_POLICY") or "open_evidence").strip().lower())
    # BULGU-78-1 — ÇIKIŞ SONRASI YENİDEN GİRİŞ (`backtest_symbol` + `backtest_portfolio` okur).
    #   "next_bar" (varsayılan): bar i+1'de kapanan pozisyonun sembolünde AYNI iterasyonda
    #       giriş yok. Çıkış bar i+1'in İÇİNDE olur; bar i kapanışındaki karar o an açık
    #       pozisyonu görürdü. En erken gerçekçi yeniden giriş bar i+2 açılışıdır.
    #   "same_bar": eski döngü (çıkışı görüp aynı barın AÇILIŞINDA girer) — yalnız A/B için.
    # Ölçüldü (yerel Alpaca tarihçesi, aynı süreç): 1d 20 sembol 124 · 1h 17 sembol 82 aynı-bar
    # girişi; ortanca sembol getirisi +0,46 / +0,47 pp İYİMSER, 37 sembolün 24'ünde A>B.
    reentry_policy: str = field(
        default_factory=lambda: (os.getenv("REENTRY_POLICY") or "next_bar").strip().lower())
    # BULGU-59-3 — hacim-katılımına duyarlı piyasa etkisi katsayısı η
    # (`SlippageModel.slip_pct`: taban + η·σ·sqrt(Q/ADV)). **0.0 = KAPALI, varsayılan.**
    # Bu evrende ölçülmüş bir η YOK; ölçülmemiş bir katsayıyı varsayılan yapmak
    # "kanıtsız bileşen" yasağının maliyet tarafıdır (Tur-51/Tur-31 disiplini). Literatür
    # aralığı η≈0,3-1,0 (Almgren 2005). Duyarlılık taraması: `research/cost_attribution.py`.
    impact_coef: float = field(default_factory=lambda: float(os.getenv("IMPACT_COEF", "0")))
    impact_adv_bars: int = 20                    # ADV penceresi (nedensel, karar barına dek)


# ---------------------------------------------------------------------------
# OPSİYON STRATEJİ — Madde 2, 16, 17, 21, 27, 28
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OptionStrategyConfig:
    # Madde 28 — DTE penceresi
    target_delta: float = 0.50            # 0.30=OTM | 0.50=ATM | 0.70=ITM
    delta_tolerance: float = 0.10         # hedef delta'dan ± bu kadar uzaktaki strike'lar kabul
    dte_min: int = 30
    dte_max: int = 45
    # Madde 16 — IV rejimi guard'ları
    skip_if_iv_above: float = 1.0         # IV > %100 (genelde olay öncesi şişme) → atla
    # Madde 17 — IV crush proxy: yüksek IV'de long opsiyon kalitesi düşer
    # (gerçek IV crush filtresi olay takvimi gerektiriyor — sonraki tur)
    # Premium-based exit (Madde 2 ile uyumlu)
    premium_stop_loss_pct: float = 0.50   # primin %50'sini kaybettiğinde kapat
    premium_target_gain_pct: float = 1.00 # %100 kazanca kapat (2x premium)
    # Madde 28 — Vadeye kalan minimum. None → CONFIG.options.close_before_expiry_dte.
    # Zinciri OKUYAN taraf: `risk/position_manager.check_exit` (Tur-50'de bağlandı —
    # o güne dek bu yorum bir sözleşme tarif ediyordu, kod ise global alanı doğrudan
    # okuyup override'ı sessizce yok sayıyordu).
    close_before_expiry_dte_override: int | None = None


# ---------------------------------------------------------------------------
# PİYASA REJİMİ → RİSK ÇARPANI (A4 — AI_trade'den uyarlandı)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeConfig:
    """Piyasa-geneli rejim (benchmark trendi ± VIX) → risk çarpanı. Adaptif drawdown
    freni (Madde 9) ile ÇARPIMSAL istiflenir (AI_trade gibi). VARSAYILAN KAPALI:
    açmadan önce iki periyotta A/B (bench.py) yap — rejim filtresi de overfit edebilir."""
    enable_regime_risk: bool = field(default_factory=lambda: _envb("REGIME_RISK", False))
    benchmark: str = field(default_factory=lambda: os.getenv("REGIME_BENCHMARK", "SPY"))
    # Rejim GÜNLÜK bar konseptidir: SMA200 = ~200 işlem GÜNÜ (klasik boğa/ayı eşiği).
    # İşlem zaman dilimi (HTF=1h) ile hesaplanırsa SMA200 ≈ 8 gün olur → anlamsız.
    # Bu yüzden benchmark her zaman kendi (varsayılan günlük) aralığından çekilir ve
    # yavaş değiştiği için refresh_minutes'te bir yenilenir (her 60sn döngüde değil).
    benchmark_interval: str = field(default_factory=lambda: os.getenv("REGIME_INTERVAL", "1d"))
    refresh_minutes: float = field(default_factory=lambda: float(os.getenv("REGIME_REFRESH_MIN", "60")))
    sma_long: int = 200
    sma_short: int = 50
    recent_return_bars: int = 20
    # DEFENSIVE eşiği: benchmark son N-bar getirisi bu %'nin altındaysa savunmaya geç
    defensive_return_pct: float = field(default_factory=lambda: float(os.getenv("REGIME_DEFENSIVE_RET", "-5.0")))
    # Risk çarpanları (AI_trade: 0.5 / 0.8 / 1.0)
    defensive_mult: float = field(default_factory=lambda: float(os.getenv("REGIME_DEFENSIVE_MULT", "0.5")))
    neutral_mult: float = field(default_factory=lambda: float(os.getenv("REGIME_NEUTRAL_MULT", "0.8")))
    bullish_mult: float = field(default_factory=lambda: float(os.getenv("REGIME_BULLISH_MULT", "1.0")))
    # İsteğe bağlı VIX katkısı (çağıran VIX değerini geçerse). 0 = kullanma.
    vix_high: float = field(default_factory=lambda: float(os.getenv("REGIME_VIX_HIGH", "28")))
    vix_low: float = field(default_factory=lambda: float(os.getenv("REGIME_VIX_LOW", "16")))


# ---------------------------------------------------------------------------
# FRED MAKRO 'RISK-OFF' KAPISI (B1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MacroConfig:
    """Ücretsiz FRED göstergelerinden basit 'risk-off oy' sayımı → risk çarpanı.
    Rejim (A4) + drawdown (Madde 9) freniyle ÇARPIMSAL istiflenir. VARSAYILAN KAPALI
    ve ücretsiz FRED API key gerektirir; kapalıysa/keysizse/çekim başarısızsa çarpan
    1.0 (etkisiz) → sistem davranışı değişmez. Açmadan önce iki periyotta A/B yap
    (makro filtre de overfit/whipsaw üretebilir — docs/EDGE_FINDINGS.md disiplini)."""
    enable_macro_gate: bool = field(default_factory=lambda: _envb("MACRO_GATE", False))
    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))
    # Makro yavaş değişir → varsayılan 6 saatte bir yenile (her döngüde değil)
    refresh_minutes: float = field(default_factory=lambda: float(os.getenv("MACRO_REFRESH_MIN", "360")))
    # Tetik eşikleri (her tetik bir 'risk-off' oyu)
    vix_high: float = field(default_factory=lambda: float(os.getenv("MACRO_VIX_HIGH", "28")))
    yield_inverted_below: float = field(default_factory=lambda: float(os.getenv("MACRO_T10Y2Y_MIN", "0.0")))
    hy_spread_high: float = field(default_factory=lambda: float(os.getenv("MACRO_HY_SPREAD_HIGH", "5.0")))
    # Oy sayısına göre risk çarpanı (1 tetik = temkin, ≥2 = risk-off)
    one_trigger_mult: float = field(default_factory=lambda: float(os.getenv("MACRO_ONE_MULT", "0.8")))
    multi_trigger_mult: float = field(default_factory=lambda: float(os.getenv("MACRO_MULTI_MULT", "0.5")))


# ---------------------------------------------------------------------------
# Tek bir merkezi config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    risk: RiskConfig = field(default_factory=RiskConfig)
    portfolio: PortfolioCaps = field(default_factory=PortfolioCaps)
    greeks: GreekCaps = field(default_factory=GreekCaps)
    regulation: Regulation = field(default_factory=Regulation)
    options: OptionFilters = field(default_factory=OptionFilters)
    strategy: StrategyFilters = field(default_factory=StrategyFilters)
    safety: SafetyLimits = field(default_factory=SafetyLimits)
    market_structure: MarketStructure = field(default_factory=MarketStructure)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    option_strategy: OptionStrategyConfig = field(default_factory=OptionStrategyConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    macro: MacroConfig = field(default_factory=MacroConfig)


CONFIG = Config()


# ---------------------------------------------------------------------------
# Sektör eşlemesi — Madde 12 (korelasyonlu pozisyonların kümelenmesi)
# ---------------------------------------------------------------------------
SECTOR_MAP: dict[str, str] = {
    # Tech / IT
    "AAPL": "TECH", "MSFT": "TECH", "GOOGL": "TECH", "GOOG": "TECH",
    "NVDA": "TECH", "META": "TECH", "AVGO": "TECH", "AMD": "TECH",
    "INTC": "TECH", "ADBE": "TECH", "CSCO": "TECH", "ORCL": "TECH",
    "CRM": "TECH", "QCOM": "TECH", "IBM": "TECH", "PLTR": "TECH",
    # Communications
    "NFLX": "COMM", "DIS": "COMM", "T": "COMM", "VZ": "COMM",
    # Consumer Discretionary
    "AMZN": "DISCRETIONARY", "TSLA": "DISCRETIONARY", "HD": "DISCRETIONARY",
    "MCD": "DISCRETIONARY", "NKE": "DISCRETIONARY", "SBUX": "DISCRETIONARY",
    # Consumer Staples
    "WMT": "STAPLES", "COST": "STAPLES", "PEP": "STAPLES", "KO": "STAPLES",
    "PG": "STAPLES", "MO": "STAPLES",
    # Finance
    "JPM": "FINANCE", "BAC": "FINANCE", "WFC": "FINANCE", "GS": "FINANCE",
    "MS": "FINANCE", "C": "FINANCE", "V": "FINANCE", "MA": "FINANCE",
    "AXP": "FINANCE", "BRK-B": "FINANCE",
    # Healthcare
    "JNJ": "HEALTH", "PFE": "HEALTH", "MRK": "HEALTH", "UNH": "HEALTH",
    "ABBV": "HEALTH", "LLY": "HEALTH", "TMO": "HEALTH",
    # Energy
    "XOM": "ENERGY", "CVX": "ENERGY", "COP": "ENERGY", "SLB": "ENERGY",
    # ETFs - aynı kategorideyse aynı bahis kabul edilir
    "SPY": "ETF_US_BROAD", "VTI": "ETF_US_BROAD", "DIA": "ETF_US_BROAD",
    "QQQ": "ETF_NASDAQ", "IWM": "ETF_SMALLCAP",
    "TLT": "ETF_BONDS", "HYG": "ETF_BONDS",
    "GLD": "ETF_COMMODITY", "SLV": "ETF_COMMODITY", "USO": "ETF_COMMODITY",
    # Volatility
    "VIX": "VOL", "VXX": "VOL", "UVXY": "VOL",
    # Index opsiyon dayanağı
    "SPX": "INDEX", "NDX": "INDEX", "RUT": "INDEX",
    # S&P 500 dışı Nasdaq-100/ADR isimleri (Wikipedia S&P tablosunda yok → elle)
    "ARM": "TECH", "ASML": "TECH", "MRVL": "TECH", "MSTR": "TECH",
    "SHOP": "TECH", "ZS": "TECH",
    "ALNY": "HEALTH", "INSM": "HEALTH",
    "MELI": "DISCRETIONARY", "PDD": "DISCRETIONARY",
    "CCEP": "STAPLES", "FER": "INDUSTRIALS", "TRI": "INDUSTRIALS",
}


def _load_sector_file() -> dict[str, str]:
    """data/sector_map.json (build_sector_map.py ile üretilen tam evren GICS haritası).

    Dosya yoksa elle tanımlı SECTOR_MAP'e düşer. Modül-düzeyinde bir kez okunur;
    güncellemek için build_sector_map.py'yi yeniden çalıştır + süreci yeniden başlat.
    """
    f = DATA_DIR / "sector_map.json"
    try:
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                merged = {**data, **SECTOR_MAP}   # elle override her zaman öncelikli
                return {k.upper(): v for k, v in merged.items()}
    except Exception:
        pass
    return dict(SECTOR_MAP)


_SECTOR_LOOKUP: dict[str, str] = _load_sector_file()


def get_sector(symbol: str) -> str:
    """Madde 12 — Sembol için sektör/kategori. Bilinmiyorsa 'UNKNOWN'.

    Önce tam evren haritası (data/sector_map.json), sonra elle SECTOR_MAP.
    """
    return _SECTOR_LOOKUP.get(symbol.upper(), "UNKNOWN")


# ---------------------------------------------------------------------------
# Başlangıç config doğrulaması — yanlış yazılmış .env değerleri sessizce yanlış
# davranışa yol açmasın (örn. AUTO_SCREEN_SORT=expectancyy → None'a göre sıralama).
# ---------------------------------------------------------------------------
_VALID_PROVIDERS = {"yfinance", "tradingview", "tv", "alpaca", "alpaca_stream", "alpaca_ws", "hub"}
_VALID_INTERVALS = set(INTERVAL_MINUTES)               # 1m,5m,15m,1h,1d
_VALID_SORT = {"expectancy", "sharpe", "total_return_pct", "win_rate_pct"}
_VALID_UNIVERSE = {"nasdaq100", "sp500", "both"}
_VALID_OPTION_MODE = {"stock", "option", "both"}
_VALID_RUN_MODE = {"SIMULATION", "BACKTEST", "LIVE_SIGNALS"}


def validate_config() -> tuple[list[str], list[str]]:
    """(errors, warnings) döndürür. errors → uygulama başlamamalı; warnings → log.
    Modül sabitlerini okur (test/monkeypatch dostu)."""
    errors: list[str] = []
    warnings: list[str] = []

    def _enum(name, value, valid):
        if value not in valid:
            errors.append(f"{name}={value!r} geçersiz — geçerli: {sorted(valid)}")

    _enum("RUN_MODE", RUN_MODE, _VALID_RUN_MODE)
    _enum("OPTION_MODE", OPTION_MODE, _VALID_OPTION_MODE)
    # ACTIVE_STRATEGY — geçerli modlar strategy.py'da tek kaynak (geç import: döngü önleme)
    try:
        from nihai.signals.strategy import STRATEGY_MODES
        _enum("ACTIVE_STRATEGY", ACTIVE_STRATEGY, set(STRATEGY_MODES))
    except Exception:
        pass
    _enum("DATA_PROVIDER", DATA_PROVIDER, _VALID_PROVIDERS)
    _enum("AUTO_SCREEN_PROVIDER", AUTO_SCREEN_PROVIDER, _VALID_PROVIDERS)
    _enum("AUTO_SCREEN_SORT", AUTO_SCREEN_SORT, _VALID_SORT)
    _enum("AUTO_SCREEN_UNIVERSE", AUTO_SCREEN_UNIVERSE, _VALID_UNIVERSE)
    _enum("AUTO_SCREEN_INTERVAL", AUTO_SCREEN_INTERVAL, _VALID_INTERVALS)
    _enum("HTF_INTERVAL", HTF_INTERVAL, _VALID_INTERVALS)

    if DUAL_TIER_ENABLED:
        _enum("FOCUSED_PROVIDER", FOCUSED_PROVIDER, _VALID_PROVIDERS)
        _enum("BROAD_PROVIDER", BROAD_PROVIDER, _VALID_PROVIDERS)
        if FOCUSED_TOP_N < 0:
            errors.append(f"FOCUSED_TOP_N={FOCUSED_TOP_N} negatif olamaz")
        if FOCUSED_PROVIDER == BROAD_PROVIDER:
            warnings.append(f"FOCUSED_PROVIDER == BROAD_PROVIDER ({FOCUSED_PROVIDER}) — "
                            "iki-katmanın veri-kalitesi ayrımı yok; tek sağlayıcı taraması gibi")

    if AUTO_SCREEN_INTERVAL_HOURS <= 0:
        errors.append("AUTO_SCREEN_INTERVAL_(MINUTES|HOURS) pozitif olmalı")
    if INITIAL_EQUITY <= 0:
        errors.append(f"INITIAL_EQUITY={INITIAL_EQUITY} pozitif olmalı")
    if AUTO_SCREEN_MAX_RUIN > 0 and AUTO_SCREEN_MC <= 0:
        warnings.append("AUTO_SCREEN_MAX_RUIN>0 ama AUTO_SCREEN_MC=0 → ruin olasılığı "
                        "hesaplanmaz, ruin filtresi etkisiz")

    # --- A1–B8 geliştirme knob'ları (yalnızca açıkken doğrula) ---
    r = CONFIG.risk
    if r.enable_trailing and r.trail_tiered and not r.trail_tiers:
        warnings.append("TRAIL_TIERED açık ama TRAIL_TIERS boş/bozuk → düz TRAIL_R'ye düşülür")
    if r.enable_mfe_giveback and (r.mfe_giveback_arm_r <= 0 or r.mfe_giveback_r <= 0):
        errors.append("MFE_GIVEBACK açık ama MFE_GIVEBACK_ARM_R / MFE_GIVEBACK_R pozitif değil")

    reg = CONFIG.regime
    if reg.enable_regime_risk:
        _enum("REGIME_INTERVAL", reg.benchmark_interval, _VALID_INTERVALS)
        for nm, mv in (("DEFENSIVE", reg.defensive_mult), ("NEUTRAL", reg.neutral_mult),
                       ("BULLISH", reg.bullish_mult)):
            if mv <= 0:
                errors.append(f"REGIME_{nm}_MULT={mv} pozitif olmalı")
            elif mv > 1.0:
                warnings.append(f"REGIME_{nm}_MULT={mv} > 1.0 → bu rejimde riski ARTIRIR (amaç genelde azaltmak)")

    if CONFIDENCE_SIZING_ENABLED:
        if not (0 < CONFIDENCE_LOW_MULT <= 1.0):
            warnings.append(f"CONFIDENCE_LOW_MULT={CONFIDENCE_LOW_MULT} (0,1] dışında — sizing'i bozabilir")
        if CONFIDENCE_COMFORT_TRADES < AUTO_SCREEN_MIN_TRADES:
            warnings.append(f"CONFIDENCE_COMFORT_TRADES={CONFIDENCE_COMFORT_TRADES} < "
                            f"AUTO_SCREEN_MIN_TRADES={AUTO_SCREEN_MIN_TRADES} → düşük-güven katmanı hiç oluşmaz")

    if SCREEN_HYBRID_ENABLED and SCREEN_RS_SHORT_DAYS >= SCREEN_RS_LONG_DAYS:
        warnings.append(f"SCREEN_RS_SHORT_DAYS={SCREEN_RS_SHORT_DAYS} ≥ "
                        f"SCREEN_RS_LONG_DAYS={SCREEN_RS_LONG_DAYS} → kısa/uzun RS ayrımı anlamsız")
    return errors, warnings
