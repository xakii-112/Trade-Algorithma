"""coklu_piyasa.py — Projedeki AI'ları ve ALGORİTMALARI AYIRIR, hepsini tüm para piyasalarında koşturur (Tur-96).

AYRIM (dosyanın ilk yarısı, makine-okunur; testle kilitli):
  · ALGORİTMA = kural tabanlı, veriden ÖĞRENMEYEN karar üreticisi (`strategy.detect_*`, oylayıcılar).
  · AI        = veriden parametre ÖĞRENEN ya da akışta UYARLANAN model (ML, sinir ağı, çevrimiçi öğrenen,
                olasılıksal rejim/oynaklık modeli, Bayes karar, dil modeli).
  · ARAÇ      = karar üretmeyen deterministik matematik/ölçüm/altyapı (formül kütüphaneleri, göstergeler…).
  `MODUL_SINIFI` nihai/ai + nihai/signals altındaki HER modülü bu üç sınıftan birine koyar; yeni bir modül
  sınıfsız kalırsa test kırmızı verir (ayrım bir kez yazılıp bayatlamasın).

KOŞU (ikinci yarı) — `research.piyasalar` kataloğundaki 14 piyasa grubu:
  1. ALGORİTMALAR backtest motoruyla (`backtest_symbol`), AI bileşeni KAPALI (META_GATE=off), kill-switch
     KAPALI (canlı paritesi; Tur-95 BULGU-95-3: açıkken dönemin %93-96'sı işlemsiz geçiyor).
  2. AI'lar KENDİ öğrenme düzenekleriyle, algoritmadan BAĞIMSIZ birer "trader" olarak:
       ML (lojistik/GBM/XGB) + sinir ağı (deep_mlp, quant_transformer) → piyasa başı havuzlanmış YÜRÜYEN
       eğitim (her test yılından önce yeniden eğitim); FTRL ve Hedge → ÇEVRİMİÇİ (test-then-train);
       HMM → yıllık yeniden-fit + nedensel filtre; Avellaneda-Lee → kayan pencere kesitsel; GARCH/GJR/HAR →
       oynaklık ÖNGÖRÜSÜ (yön değil; EWMA tabanına karşı QLIKE).
     Canlı kayıtlı modeller (meta lojistik kapı, 1m sinir ağları) YENİDEN EĞİTİLMEDEN uygulanır.
  Ortak ölçüm penceresi `TEST_BAS`→bugün; ortak maliyet taraf başına `CONFIG.backtest.stock_slippage_pct`.

LOOK-AHEAD SÖZLEŞMESİ: karar bar i KAPANIŞINDA, infaz bar i+1 AÇILIŞINDA, getiri open[i+1]→open[i+2]. Etiket
y_i open[i+2]'de BİLİNİR → çevrimiçi öğrenenler i. karardan önce yalnız j ≤ i−2 etiketleriyle güncellenir;
yürüyen eğitimde eğitime yalnız etiketi test yılının ilk gününden ÖNCE bilinen satırlar girer.

CLI:
  python -m research.coklu_piyasa --asama 1 [--isci 8] [--piyasa ABD] [--sembol SPY]   # enstrüman başı
  python -m research.coklu_piyasa --asama 2 [--isci 8]                                  # piyasa başı ML
  python -m research.coklu_piyasa --asama 3                                             # GPU: transformer + canlı sinir ağı
  python -m research.coklu_piyasa --asama rapor                                         # karne JSON'u
"""
from __future__ import annotations

import json
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


# =====================================================================================================
# 1) AYRIM — algoritmalar ↔ AI'lar ↔ araçlar
# =====================================================================================================
@dataclass(frozen=True)
class Yontem:
    ad: str
    tur: str        # "algoritma" | "ai"
    aile: str
    kaynak: str     # "paket.modul.nesne" — test içe aktararak doğrular
    aciklama: str
    kosulur: bool   # bu turda çoklu piyasada koşuldu mu
    neden: str = ""  # koşulmadıysa NEDEN


_S = "nihai.signals.strategy."
ALGORITMALAR: dict[str, Yontem] = {y.ad: y for y in [
    Yontem("event_breakout", "algoritma", "kural", _S + "detect_event",
           "olay-bazlı giriş: 20 bar kırılımı/kesişim anı (Madde 1; canlıda ACTIVE_STRATEGY)", True),
    Yontem("mean_reversion", "algoritma", "kural", _S + "detect_mean_reversion",
           "ortalamaya dönüş: bant/RSI aşırılığından geri dönüş", True),
    Yontem("rsi_divergence", "algoritma", "kural", _S + "detect_rsi_divergence",
           "fiyat ↔ RSI uyumsuzluğu", True),
    Yontem("macd", "algoritma", "kural", _S + "detect_macd", "MACD sinyal çizgisi kesişimi", True),
    Yontem("donchian", "algoritma", "kural", _S + "detect_donchian", "Donchian kanal kırılımı (trend takibi)",
           True),
    Yontem("squeeze", "algoritma", "kural", _S + "detect_squeeze", "Bollinger-Keltner sıkışmasından çıkış",
           True),
    Yontem("trend_pullback", "algoritma", "kural", _S + "detect_trend_pullback", "trend yönünde geri çekilme",
           True),
    Yontem("vol_breakout", "algoritma", "kural", _S + "detect_vol_breakout", "oynaklık genişlemesi kırılımı",
           True),
    Yontem("ensemble", "algoritma", "oylayici", _S + "detect_setup",
           "8 baz detektörün oylaması (ENSEMBLE_MODE) — öğrenmez, sabit kural", True),
    Yontem("confluence", "algoritma", "oylayici", _S + "detect_setup",
           "≥N bağımsız detektör mutabakatı — öğrenmez", True),
    Yontem("quant_trend", "algoritma", "quant", _S + "detect_quant_trend",
           "Tur-97: 21/63/252 günlük getiri işaretlerinin ortalaması + EWMA σ ölçekli 3σ stop "
           "(research.quant_kayma ile aynı 14 piyasada ayrıca koşuldu)", True),
    Yontem("pead", "algoritma", "olay", _S + "detect_pead", "kazanç sonrası sürüklenme", False,
           "ABD kazanç takvimi/sürprizi — diğer piyasalar için ücretsiz kaynak yok"),
    Yontem("insider", "algoritma", "olay", _S + "detect_insider", "SEC Form 4 içeriden alım", False,
           "SEC EDGAR Form 4 yalnız ABD ihraççıları için"),
    Yontem("sentiment", "algoritma", "olay", _S + "detect_sentiment", "sosyal medya duyarlılığı", False,
           "tarihsel duyarlılık arşivi yok (canlı toplanan akış)"),
]}

AILER: dict[str, Yontem] = {y.ad: y for y in [
    Yontem("meta_lojistik", "ai", "ml", "research.train_meta.LogisticModel",
           "L2 lojistik regresyon, 20 teknik özellikten ertesi gün yönü", True),
    Yontem("meta_gbm", "ai", "ml", "research.train_meta.GBMModel", "gradyan artırmalı ağaçlar (sklearn)", True),
    Yontem("meta_xgb", "ai", "ml", "research.train_meta.XGBModel", "XGBoost", True),
    Yontem("sinir_mlp", "ai", "sinir_agi", "nihai.ai.neural.NeuralCapacity",
           "deep_mlp basamağı (canlı gölge köprünün tarihsel basamağı)", True),
    Yontem("sinir_transformer", "ai", "sinir_agi", "nihai.ai.neural.NeuralCapacity",
           "quant_transformer (~5M parametre, GPU)", True),
    Yontem("ftrl", "ai", "cevrimici", "nihai.ai.online.FTRLProximal",
           "akışta öğrenen lojistik (FTRL-Proximal), 2 bar gecikmeli güncelleme", True),
    Yontem("hedge_algoritma", "ai", "cevrimici", "nihai.ai.online.hedge_combine",
           "8 algoritmayı uzman sayıp ağırlığı akışta öğrenen birleştirici", True),
    Yontem("rejim_hmm", "ai", "olasiliksal", "nihai.ai.regime_hmm.gaussian_hmm_fit",
           "2 durumlu Gauss HMM; boğa rejimi olasılığı > 0,5 iken uzun", True),
    Yontem("statarb_ou", "ai", "olasiliksal", "nihai.ai.statarb.avellaneda_lee_scores",
           "öz-portföy kalıntısı + OU s-skoru (piyasa içi kesitsel, 60 gün)", True),
    Yontem("vol_garch", "ai", "oynaklik", "nihai.ai.quant_math.garch11_fit",
           "GARCH(1,1) ertesi gün varyans öngörüsü", True),
    Yontem("vol_gjr", "ai", "oynaklik", "nihai.ai.quant_math.gjr_garch_fit", "GJR-GARCH (kaldıraç asimetrisi)",
           True),
    Yontem("vol_har", "ai", "oynaklik", "nihai.ai.microstructure.har_rv_fit",
           "HAR-RV (Garman-Klass + gece boşluğu RV vekili)", True),
    Yontem("canli_meta_kapi", "ai", "canli_model", "nihai.signals.meta_gate.get_gate",
           "canlıdaki meta lojistik (data/meta_model.json) — YENİDEN EĞİTİLMEDEN event_breakout'a kapı", True),
    Yontem("canli_sinir_agi", "ai", "canli_model", "nihai.ai.neural.NeuralCapacity.load",
           "canlı gölge sinir ağları (1m kartları) — YENİDEN EĞİTİLMEDEN dağılım ve isabet", True),
    Yontem("quant_beyin", "ai", "bayes", "nihai.ai.decision.decide", "Bayes karar motoru", False,
           "önseli CANLI ABD tahmin defterinden öğrenir (ai_learning_state); başka piyasada öğrenme durumu "
           "yok ve Tur-95 süpürmesinde hiçbir λ yön üretmedi"),
    Yontem("ogrenme_kalibrasyon", "ai", "bayes", "nihai.ai.learning.update_from_predictions",
           "Beta posterior öğrenme + kalibratörler", False,
           "kendi başına yön üretmez; canlı kaynakların olasılığını kalibre eder"),
    Yontem("epizodik_hafiza", "ai", "bayes", "nihai.ai.episodic.analog_ozeti", "benzer-durum (k-NN) hafızası",
           False, "gölge Quant Beyin kararlarının defterinden beslenir (yukarıdaki nedenle yok)"),
    Yontem("dil_modeli", "ai", "llm", "nihai.ai.router.complete",
           "Claude yönlendirici/ajan/haber zekâsı + strateji kipi llm_event", False,
           "tarihsel koşuda model geçmişi zaten bilir (look-ahead) ve çağrı maliyeti ölçekte karşılanamaz"),
    Yontem("dcc_figarch", "ai", "oynaklik", "nihai.ai.econometrics", "DCC korelasyon / FIGARCH uzun hafıza",
           False, "yön/oynaklık öngörüsü GARCH ailesiyle ölçüldü; DCC bir korelasyon TEŞHİSİDİR"),
]}

# nihai/ai + nihai/signals altındaki her modül → (sınıf, gerekçe). Sınıf: "ai" | "algoritma" | "arac".
MODUL_SINIFI: dict[str, tuple[str, str]] = {
    # --- nihai/ai
    "nihai.ai.adaptive_calibration": ("ai", "yürüyen kalibratör öğrenir"),
    "nihai.ai.advanced_math": ("arac", "sıçrama/stokastik vol/opsiyon fiyatlama formülleri"),
    "nihai.ai.agent": ("ai", "LLM soru-cevap ajanı"),
    "nihai.ai.brain": ("ai", "Quant Beyin orkestratörü (hafıza+model+karar)"),
    "nihai.ai.calibration_loop": ("ai", "zaman-bölmeli kalibratör öğrenir"),
    "nihai.ai.consolidate": ("ai", "semantik hafıza konsolidasyonu"),
    "nihai.ai.data_health": ("arac", "veri sağlığı kapısı"),
    "nihai.ai.decision": ("ai", "Bayes karar motoru (posterior prior)"),
    "nihai.ai.drift_monitor": ("arac", "PSI/Wasserstein sürüklenme ölçümü"),
    "nihai.ai.econometrics": ("ai", "DCC/FIGARCH uydurur (olasılıksal model)"),
    "nihai.ai.econophysics": ("arac", "entropi/MFDFA/kaos ölçüleri"),
    "nihai.ai.episode_resolver": ("ai", "gölge kararların sonuç kolunu kapatır (öğrenme döngüsü)"),
    "nihai.ai.episodic": ("ai", "benzer-durum hafızası (k-NN)"),
    "nihai.ai.explain": ("arac", "açıklanabilirlik ölçümü"),
    "nihai.ai.fundamentals": ("arac", "temel analiz skorları (formül)"),
    "nihai.ai.knowledge": ("arac", "kavram kataloğu"),
    "nihai.ai.lab": ("ai", "LLM kod laboratuvarı"),
    "nihai.ai.learning": ("ai", "Beta posterior öğrenme + source_fusion"),
    "nihai.ai.memory": ("arac", "JSONL operatör hafızası (kayıt)"),
    "nihai.ai.microstructure": ("ai", "HAR-RV uydurur; ayrıca makas/VPIN ölçüleri"),
    "nihai.ai.ml_hygiene": ("arac", "etiketleme/aşırı-öğrenme karşıtı araçlar"),
    "nihai.ai.models": ("ai", "kapasite merdiveni (sklearn modelleri)"),
    "nihai.ai.neural": ("ai", "sinir ağı merdiveni (PyTorch)"),
    "nihai.ai.observatory": ("arac", "ölçüm modüllerini canlıya bağlayan gölge geçidi"),
    "nihai.ai.online": ("ai", "FTRL / Hedge / ADWIN çevrimiçi öğrenme"),
    "nihai.ai.option_flow": ("arac", "opsiyon OI/GEX toplamları"),
    "nihai.ai.options_math": ("arac", "IV/greek/SVI/SABR formülleri"),
    "nihai.ai.portfolio": ("arac", "portföy optimizasyonu (HRP/BL/CVaR) — öğrenmez, çözer"),
    "nihai.ai.quant_math": ("ai", "GARCH ailesi uydurur; ayrıca VaR/Kelly formülleri"),
    "nihai.ai.reasoner": ("ai", "LLM ile yapılandırılmış akıl yürütme"),
    "nihai.ai.regime_hmm": ("ai", "Gauss HMM (Baum-Welch)"),
    "nihai.ai.registry": ("arac", "model kartı yaşam döngüsü"),
    "nihai.ai.router": ("ai", "LLM yönlendirici"),
    "nihai.ai.selfcheck": ("arac", "öz-değerlendirme sınavı"),
    "nihai.ai.shadow_runner": ("ai", "Quant Beyin gölge koşucusu"),
    "nihai.ai.statarb": ("ai", "öz-portföy + OU uydurur"),
    "nihai.ai.tda": ("arac", "topolojik veri analizi ölçümü"),
    "nihai.ai.text_intel": ("ai", "LLM ile metin/haber zekâsı"),
    "nihai.ai.volume_profile": ("arac", "hacim profili ölçüleri"),
    # --- nihai/signals
    "nihai.signals.breadth": ("arac", "piyasa genişliği ölçüleri"),
    "nihai.signals.indicators": ("arac", "gösterge kütüphanesi + konsensüs paneli"),
    "nihai.signals.indicators_ext": ("arac", "genişletilmiş gösterge kütüphanesi"),
    "nihai.signals.market_regime": ("algoritma", "rejim → risk çarpanı kuralı"),
    "nihai.signals.meta_features": ("arac", "özellik çıkarımı"),
    "nihai.signals.meta_gate": ("ai", "eğitilmiş meta modelle sinyal kapısı"),
    "nihai.signals.quant": ("algoritma", "zaman serisi momentumu + σ ölçekli risk (Tur-97; sabit kural, öğrenmez)"),
    "nihai.signals.model_promotion": ("ai", "meta kapı modelinin kanıtlı terfisi"),
    "nihai.signals.setup_score": ("algoritma", "kurulum kalitesi skoru (kural)"),
    "nihai.signals.signal_engine": ("arac", "karar/işlem loglama"),
    "nihai.signals.strategy": ("algoritma", "14 strateji kipinin detektörleri + kalite kapıları"),
}

MODLAR_ALGO = ("event_breakout", "mean_reversion", "rsi_divergence", "macd", "donchian", "squeeze",
               "trend_pullback", "vol_breakout", "ensemble", "confluence")


# =====================================================================================================
# 2) Veri
# =====================================================================================================
BAS = "2008-01-01"          # veri başı: gösterge ısınması + AI eğitim geçmişi
TEST_BAS = "2018-01-01"     # ortak değerlendirme penceresi (algoritma + AI)
SICAK = 200                 # ilk SICAK bar özellik ısınması (EMA200)
KUSUR_ESIGI = 0.5           # kripto dışı |günlük log getiri| > 0,5 → veri kusuru (sonuçtan ÖNCE sabitlendi)
# Sinyale bağlı 4 özellik (nötr sinyalde sabit) + 1d'de her zaman 0 olan saat atılır.
ATILAN = ("rr_ratio", "quality_score", "confluence", "is_long", "hour")
SONUC_DIR = Path(os.getenv("PIYASA_SONUC_DIR") or (Path.home() / "nihai_hist" / "piyasa_sonuc"))
KARNE = ROOT / "data" / "coklu_piyasa_karnesi.json"


def ozellik_adlari() -> list[str]:
    from nihai.signals.meta_features import FEATURE_NAMES
    return [f for f in FEATURE_NAMES if f not in ATILAN]


def kusur_nedeni(piyasa: str, df: pd.DataFrame) -> str | None:
    """Pencere içi veri kusuru → neden (dışlama); temizse None. Kural sonuçlara bakılmadan sabitlendi."""
    w = df[df.index >= BAS]
    if len(w) < SICAK + 300:
        return f"yetersiz geçmiş ({len(w)} bar)"
    if w.index[-1] < pd.Timestamp(TEST_BAS) + pd.Timedelta(days=365):
        return "test penceresinde veri yok"
    if piyasa != "Kripto":
        r = np.abs(np.diff(np.log(w["close"].to_numpy())))
        if r.size and float(r.max()) > KUSUR_ESIGI:
            i = int(np.argmax(r))
            return f"|log getiri| {r[i]:.2f} ({w.index[i + 1].date()}) > {KUSUR_ESIGI}"
    return None


def enstrumanlar(piyasa: str | None = None, sembol: str | None = None) -> tuple[list, list]:
    """(kullanılan [(piyasa, sembol)], dışlanan [(piyasa, sembol, neden)])."""
    from research import piyasalar as P
    ok, dis = [], []
    for pz, s in P.tum_semboller():
        if (piyasa and pz != piyasa) or (sembol and s != sembol):
            continue
        neden = kusur_nedeni(pz, P.yukle(s))
        (dis.append((pz, s, neden)) if neden else ok.append((pz, s)))
    return ok, dis


def hazirla(sym: str) -> pd.DataFrame:
    """Önbellek → BAS sonrası → işlem olmayan (bayat) günler atılır → (kripto dışı) tek-bar aralık onarımı
    → ilk kapanış = 100 ölçeği. Bayat bar süzgeci Tur-97'de eklendi (BULGU-97-5); Tur-96 karnesi o
    satırlarla kuruldu (2.197 / 552.434 satır, çoğu Orta Doğu ve Türkiye)."""
    from research import piyasalar as P
    df = P.yukle(sym)
    df = df[df.index >= BAS]
    df = df[~P.bayat_bar_maskesi(df)]
    if sym not in P.PIYASALAR["Kripto"]["semboller"]:
        df, _ = P.aralik_onar(df, KUSUR_ESIGI)
    return P.olcekle(df)


def girdi_parmak_izi(semboller: list[str]) -> dict:
    """Karnenin GİRDİ parmak izi: `hazirla` çıktısının sembol başına satır sayısı + son tarih + SHA-256'sı ve
    hepsinin birleşik özeti (Tur-98 §9 #4). BULGU-98-6: paylaşılan bir sembolün önbelleği başka bir
    düzeneğin indirmesiyle tazelenince "aynı" komut başka sayı üretiyordu ve bunu hiçbir çıktı söylemiyordu;
    iki karnenin özeti farklıysa sayılar karşılaştırılmadan önce girdi farkı aranır."""
    import hashlib
    toplam = hashlib.sha256()
    sembol: dict[str, dict] = {}
    for s in sorted(semboller):
        df = hazirla(s)
        h = hashlib.sha256(np.ascontiguousarray(df.index.as_unit("ns").asi8).tobytes())   # birimden bağımsız
        h.update(np.ascontiguousarray(df[["open", "high", "low", "close"]].to_numpy(np.float64)).tobytes())
        ozet = h.hexdigest()[:16]
        toplam.update(f"{s}:{ozet};".encode())
        sembol[s] = {"n": int(len(df)), "son": str(df.index[-1].date()) if len(df) else None, "sha": ozet}
    return {"ozet": toplam.hexdigest()[:16], "n_sembol": len(sembol), "sembol": sembol}


# =====================================================================================================
# 3) Ölçüm yardımcıları
# =====================================================================================================
def seri_metrik(r: np.ndarray, gun: int) -> dict:
    """Günlük (net) getiri serisi → toplam getiri, CAGR, yıllık vol, Sharpe (rf=0), MDD."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if r.size < 20:
        return {"n": int(r.size)}
    eq = np.cumprod(1.0 + r)
    tepe = np.maximum.accumulate(eq)
    sd = float(r.std(ddof=1))
    yil = r.size / gun
    son = float(eq[-1])
    return {"n": int(r.size),
            "toplam_pct": round((son - 1) * 100, 2),
            "cagr_pct": round(((son ** (1 / yil)) - 1) * 100, 2) if son > 0 and yil > 0 else None,
            "vol_pct": round(sd * math.sqrt(gun) * 100, 2),
            "sharpe": round(float(r.mean()) / sd * math.sqrt(gun), 3) if sd > 0 else None,
            "mdd_pct": round(float(((tepe - eq) / tepe).max()) * 100, 2)}


def nw_alfa(y: np.ndarray, x: np.ndarray, gun: int) -> dict | None:
    """y = a + b·x (OLS), a'nın Newey-West t'si (günlük getiriler bağımsız değildir)."""
    from nihai.data.factors import _hac_se
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = y.size
    if n < 60:
        return None
    X = np.column_stack([np.ones(n), x])
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    b = xtx_inv @ (X.T @ y)
    L = int(4 * (n / 100.0) ** (2.0 / 9.0))
    se = _hac_se(X, y - X @ b, xtx_inv, L)
    return {"alfa_yillik_pct": round(float(b[0]) * gun * 100, 3), "beta": round(float(b[1]), 4),
            "alfa_t": round(float(b[0] / se[0]), 3) if se[0] > 0 else None, "n": int(n)}


def nw_ortalama_t(d: np.ndarray) -> float | None:
    """Bir fark serisinin ortalamasının Newey-West t'si (Diebold-Mariano)."""
    from nihai.data.factors import _hac_se
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 60:
        return None
    X = np.ones((d.size, 1))
    xtx_inv = np.array([[1.0 / d.size]])
    L = int(4 * (d.size / 100.0) ** (2.0 / 9.0))
    se = _hac_se(X, d - d.mean(), xtx_inv, L)
    return round(float(d.mean() / se[0]), 3) if se[0] > 0 else None


def auc(y: np.ndarray, p: np.ndarray) -> float | None:
    from research.train_meta import _auc
    y = np.asarray(y)
    p = np.asarray(p, float)
    ok = np.isfinite(p) & (y >= 0)
    y, p = y[ok].astype(int), p[ok]
    if y.size < 50 or y.min() == y.max():
        return None
    return round(float(_auc(y, p)), 4)


def pozisyon_getirisi(poz: np.ndarray, ileri: np.ndarray, maliyet: float) -> np.ndarray:
    """poz_i (bar i kapanışında karar) × open[i+1]→open[i+2] − maliyet × |Δpoz|."""
    poz = np.nan_to_num(np.asarray(poz, float), nan=0.0)
    onceki = np.concatenate([[0.0], poz[:-1]])
    return poz * np.nan_to_num(ileri, nan=0.0) - maliyet * np.abs(poz - onceki)


def maliyet_orani() -> float:
    from nihai.config import CONFIG
    return float(CONFIG.backtest.stock_slippage_pct)


# =====================================================================================================
# 4) AŞAMA 1 — enstrüman başı: algoritmalar + enstrüman-yerel AI'lar + özellik paneli
# =====================================================================================================
_HAZIR = {"v": False}


def _isci_hazirla() -> None:
    """İşçi süreç kurulumu: .env proje kökünden, AI kapısı KAPALI, VIX önbellekten (ağ yok)."""
    if _HAZIR["v"]:
        return
    _HAZIR["v"] = True
    os.chdir(ROOT)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from nihai.config import CONFIG
    object.__setattr__(CONFIG.strategy, "meta_gate_mode", "off")
    from nihai.signals import strategy as st
    from research import piyasalar as P
    vix = P.yukle("^VIX")
    if not vix.empty:
        st._VIX_DATES = [d.date() for d in vix.index]
        st._VIX_CLOSES = [float(x) for x in vix["close"].to_numpy()]
        st._VIX_LOADED = True


def _algoritma_kos(sym: str, df: pd.DataFrame, t0: int, mod: str, kapi: str = "off") -> dict:
    from nihai.config import CONFIG
    from nihai.execution.backtest import backtest_symbol
    onceki = CONFIG.strategy.meta_gate_mode
    object.__setattr__(CONFIG.strategy, "meta_gate_mode", kapi)
    try:
        r = backtest_symbol(sym, "1d", df=df, strategy_mode=mod, warmup_bars=t0, disable_kill_switch=True)
    finally:
        object.__setattr__(CONFIG.strategy, "meta_gate_mode", onceki)
    n = len(df)
    eq = np.asarray(r.equity_curve, float)[: n - t0]            # [kapanış t0, …, kapanış n-1]
    ret = eq[1:] / eq[:-1] - 1.0
    return {"tarih": df.index[t0 + 1: t0 + 1 + ret.size].to_numpy(), "r": ret.astype(np.float32),
            "islem": int(r.trades_count), "kazanma": round(float(r.win_rate), 4),
            # işlemsiz koşuda profit_factor None, kayıpsızda inf → ikisi de "ölçülemedi"
            "pf": (round(float(r.profit_factor), 3)
                   if r.profit_factor is not None and math.isfinite(r.profit_factor) else None)}


def _ozellik_ve_eylem(df: pd.DataFrame, sym: str) -> tuple[np.ndarray, np.ndarray]:
    """Her bar için 20 özellik (nötr sinyal) + 8 baz algoritmanın eylemi (+1/−1/0)."""
    from nihai.signals.meta_features import extract_features
    from nihai.signals.strategy import BASE_STRATEGY_MODES, SignalAction, _SharedIndicators, detect_setup
    adlar = ozellik_adlari()
    notr = SimpleNamespace(rr_ratio=0.0, quality_score=0.0, reasons=[], action=None)
    sh = _SharedIndicators(df)
    n = len(df)
    X = np.zeros((n, len(adlar)), dtype=np.float32)
    E = np.zeros((n, len(BASE_STRATEGY_MODES)), dtype=np.int8)
    for i in range(SICAK - 1, n):
        dilim = df.iloc[: i + 1]
        g = sh.upto(i + 1)
        f = extract_features(dilim, notr, shared=g)
        X[i] = [f[a] for a in adlar]
        for k, mod in enumerate(BASE_STRATEGY_MODES):
            a, _r, _l = detect_setup(dilim, mod, symbol=sym, shared=g)
            E[i, k] = 1 if a == SignalAction.LONG else (-1 if a == SignalAction.SHORT else 0)
    return X, E


def _hmm_kos(lr: np.ndarray, tarih: pd.DatetimeIndex, t0: int) -> np.ndarray:
    """Her test yılından önce yalnız geçmişle 2 durumlu HMM; nedensel filtreyle P(boğa)."""
    from nihai.ai.regime_hmm import gaussian_hmm_fit, hmm_filter
    n = lr.size
    p = np.full(n, np.nan)
    yillar = sorted(set(tarih[t0:].year))
    for y in yillar:
        idx = np.where(tarih.year == y)[0]
        idx = idx[idx >= t0]
        if idx.size == 0:
            continue
        bas, son = int(idx[0]), int(idx[-1])
        egitim = lr[1:bas]                                      # lr[j] bar j kapanışında bilinir
        if egitim.size < 500:
            continue
        m = gaussian_hmm_fit(egitim, n_states=2)
        if m is None:
            continue
        f = hmm_filter(lr[1: son + 1], m)
        if f is None:
            continue
        boga = int(np.argmax(np.asarray(m["means"], float)))
        filt = f["filtered"][:, boga]                           # filt[k] ↔ bar k+1
        p[bas: son + 1] = filt[bas - 1: son]
    return p


def _ftrl_kos(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """FTRL-Proximal; özellikler genişleyen (nedensel) ortalama/std ile ölçeklenir; güncelleme 2 bar gecikmeli."""
    from nihai.ai.online import FTRLProximal
    n, f = X.shape
    Xd = X.astype(float)
    cs = np.cumsum(Xd, axis=0)
    cs2 = np.cumsum(Xd * Xd, axis=0)
    k = np.arange(1, n + 1)[:, None]
    mu = cs / k
    sd = np.sqrt(np.maximum(cs2 / k - mu * mu, 1e-12))
    Z = np.clip((Xd - mu) / sd, -5, 5)
    m = FTRLProximal(n_features=f)
    p = np.full(n, np.nan)
    for i in range(SICAK + 50, n):
        j = i - 2
        if j >= SICAK + 50 and y[j] >= 0:
            m.update_one(Z[j], float(y[j]))
        p[i] = m.predict_one(Z[i])
    return p


def _hedge_kos(E: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    """8 algoritma + gecikmeli taban oranı uzmanı; Hedge ağırlığı yalnız etiketi bilinen olaylardan (gecikme 2)."""
    from nihai.ai.online import hedge_combine
    from nihai.signals.strategy import BASE_STRATEGY_MODES
    n = E.shape[0]
    p = np.full(n, np.nan)
    s = SICAK
    son = n - 2                                                  # y tanımlı son satır n-3
    if son - s < 100:
        return p, {}
    yy = y[s:son].astype(float)
    uz = 0.5 + 0.05 * E[s:son].astype(float)
    kum = np.concatenate([[0.0, 0.0], np.cumsum(yy)[:-2]])
    say = np.concatenate([[0.0, 0.0], np.arange(1, yy.size - 1, dtype=float)])
    taban = np.where(say > 0, kum / np.maximum(say, 1), 0.5)
    uz = np.column_stack([uz, np.clip(taban, 0.01, 0.99)])
    res = hedge_combine(uz, yy, include_base=False)
    if res is None:
        return p, {}
    w = res["weights"]                                          # w[t]: 0..t-1 olayları görmüş
    w_gec = np.vstack([np.full((1, w.shape[1]), 1.0 / w.shape[1]), w[:-1]])   # 0..t-2 → gecikme 2
    p[s:son] = np.sum(w_gec * uz, axis=1)
    adlar = list(BASE_STRATEGY_MODES) + ["taban_orani"]
    return p, {a: round(float(v), 4) for a, v in zip(adlar, res["final_weights"], strict=True)}


def _vol_kos(df: pd.DataFrame, t0: int, yeniden: int = 63) -> dict:
    """Ertesi gün varyans öngörüsü: EWMA(0,94) tabanı ↔ GARCH(1,1) ↔ GJR ↔ HAR; kayıp QLIKE = ln f + r²/f."""
    from nihai.ai.microstructure import har_rv_fit, har_rv_forecast
    from nihai.ai.quant_math import garch11_fit, gjr_garch_fit
    o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = c.size
    lr = np.zeros(n)
    lr[1:] = np.log(c[1:] / c[:-1])
    gece = np.zeros(n)
    gece[1:] = np.log(o[1:] / c[:-1]) ** 2
    rv = gece + 0.5 * np.log(h / lo) ** 2 - (2 * math.log(2) - 1) * np.log(c / o) ** 2
    tahmin = {k: np.full(n, np.nan) for k in ("ewma", "garch", "gjr", "har", "har_olcekli")}
    v = float(np.var(lr[1:251]))
    for j in range(1, n):                                       # EWMA: tahmin[j] = lr[j+1] için
        v = 0.94 * v + 0.06 * lr[j] ** 2
        tahmin["ewma"][j] = v
    def _gjr_adim(p: dict, e: float, s2: float) -> float:
        return p["omega"] + (p["alpha"] + (p["gamma"] if e < 0 else 0.0)) * e * e + p["beta"] * s2

    pg = pj = ph = None
    sg = sj = 0.0
    for j in range(t0, n - 1):
        # Yeniden-fit yalnız GEÇMİŞLE (lr[1..j]); BAŞARISIZ yeniden-fit o çeyreği NaN bırakmaz, önceki
        # parametrelerle özyineleme sürer (ilk sürüm GJR'de sentetik serinin %63'ünü NaN bırakıyordu).
        yg = yj = None
        if (j - t0) % yeniden == 0:
            gecmis = lr[1: j + 1]
            yg, yj = garch11_fit(gecmis), gjr_garch_fit(gecmis)
            ph = har_rv_fit(rv[1: j + 1]) or ph
            # TEŞHİS: RV vekilinin r²'ye oranı yalnız GEÇMİŞTEN (vekil düzey yanlılığını ayırmak için)
            oran = float(np.mean(lr[1: j + 1] ** 2) / max(np.mean(rv[1: j + 1]), 1e-300))
        if yg is not None:
            pg = yg
            sg = pg["omega"] + pg["alpha"] * pg["last_resid2"] + pg["beta"] * pg["last_sigma2"]
        elif pg is not None:
            sg = pg["omega"] + pg["alpha"] * (lr[j] - pg["mean"]) ** 2 + pg["beta"] * sg
        if yj is not None:
            pj = yj
            sj = _gjr_adim(pj, pj["last_resid"], pj["last_sigma2"])
        elif pj is not None:
            sj = _gjr_adim(pj, lr[j] - pj["mean"], sj)
        if pg is not None:
            tahmin["garch"][j] = sg
        if pj is not None:
            tahmin["gjr"][j] = sj
        if ph is not None:
            f = har_rv_forecast(ph, rv[1: j + 1])
            tahmin["har"][j] = f if f is not None else np.nan
            tahmin["har_olcekli"][j] = tahmin["har"][j] * oran
    hedef = np.full(n, np.nan)
    hedef[:-1] = lr[1:] ** 2
    kayip, negatif = {}, {}
    for k, seri in tahmin.items():
        ff = seri[t0: n - 1]
        negatif[k] = int(np.sum(~(ff > 0) & np.isfinite(ff)))
        ff = np.where(ff > 0, ff, np.nan)
        kayip[k] = (np.log(ff) + hedef[t0: n - 1] / ff).astype(np.float32)
    return {"tarih": df.index[t0 + 1: n].to_numpy(), "L": kayip, "negatif_tahmin": negatif,
            "rv_vekil_orani": round(float(np.mean(rv[t0:]) / max(np.mean(lr[t0:] ** 2), 1e-300)), 4),
            "tahmin": {k: seri[t0: n - 1].astype(np.float32) for k, seri in tahmin.items()}}


def enstruman_kos(arg: tuple[str, str]) -> str:
    """Aşama 1 görevi; sonucu SONUC_DIR/ens/<sembol>.pkl'e yazar, özet satırı döndürür."""
    piyasa, sym = arg
    _isci_hazirla()
    t_bas = time.perf_counter()
    df = hazirla(sym)
    n = len(df)
    # geç listelenen enstrüman (ör. SOL 2020): ilk karar gösterge ısınmasından sonra
    t0 = max(SICAK, int(np.searchsorted(df.index.values, np.datetime64(TEST_BAS))))
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    ileri = np.full(n, np.nan)
    ileri[: n - 2] = o[2:] / o[1:-1] - 1.0
    y = np.full(n, -1, dtype=np.int8)
    y[: n - 2] = (ileri[: n - 2] > 0).astype(np.int8)
    lr = np.zeros(n)
    lr[1:] = np.log(c[1:] / c[:-1])

    X, E = _ozellik_ve_eylem(df, sym)
    out: dict = {"piyasa": piyasa, "sembol": sym, "n": n, "t0": t0,
                 "tarih": df.index.to_numpy(), "ileri": ileri, "y": y, "X": X, "E": E}
    out["ai"] = {"rejim_hmm": _hmm_kos(lr, df.index, t0), "ftrl": _ftrl_kos(X, y)}
    out["ai"]["hedge_algoritma"], out["hedge_agirlik"] = _hedge_kos(E, y)
    out["vol"] = _vol_kos(df, t0)
    out["algo"] = {m: _algoritma_kos(sym, df, t0, m) for m in MODLAR_ALGO}
    out["algo"]["canli_meta_kapi"] = _algoritma_kos(sym, df, t0, "event_breakout", kapi="enforce")
    out["bh_kk"] = (c[t0 + 1:] / c[t0:-1] - 1.0).astype(np.float32)   # algoritmaların kıyası (kapanış→kapanış)
    out["sure_sn"] = round(time.perf_counter() - t_bas, 1)
    (SONUC_DIR / "ens").mkdir(parents=True, exist_ok=True)
    with open(SONUC_DIR / "ens" / f"{_guvenli(sym)}.pkl", "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return f"{piyasa} {sym} n={n} {out['sure_sn']}s"


def vol_yenile(arg: tuple[str, str]) -> str:
    """Var olan enstrüman sonucunun YALNIZ oynaklık bölümünü güncel `_vol_kos` ile yeniden hesaplar
    (atomik yazım: aşama 2 aynı dosyayı okurken yarım dosya görmesin)."""
    _piyasa, sym = arg
    _isci_hazirla()
    yol = SONUC_DIR / "ens" / f"{_guvenli(sym)}.pkl"
    e = ens_oku(sym)
    e["vol"] = _vol_kos(hazirla(sym), e["t0"])
    gecici = yol.with_suffix(f".{os.getpid()}.tmp")
    with open(gecici, "wb") as fh:
        pickle.dump(e, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(gecici, yol)
    return f"vol {sym}"


def _guvenli(sym: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in sym)


def ens_oku(sym: str) -> dict:
    with open(SONUC_DIR / "ens" / f"{_guvenli(sym)}.pkl", "rb") as fh:
        return pickle.load(fh)


# =====================================================================================================
# 5) AŞAMA 2/3 — piyasa başı havuzlanmış YÜRÜYEN eğitim (ML + sinir ağı) ve kesitsel statarb
# =====================================================================================================
ML_MODELLERI = ("meta_lojistik", "meta_gbm", "meta_xgb", "sinir_mlp")


def _model_kur(ad: str, adlar: list[str], seed: int = 42):
    from research.train_meta import GBMModel, LogisticModel, XGBModel
    if ad == "meta_lojistik":
        return lambda X, y: LogisticModel.fit(X, y)
    if ad == "meta_gbm":
        return lambda X, y: GBMModel.fit(X, y, seed=seed)
    if ad == "meta_xgb":
        return lambda X, y: XGBModel.fit(X, y, seed=seed)
    from nihai.ai.neural import NeuralCapacity
    tier, cihaz = ("deep_mlp", "cpu") if ad == "sinir_mlp" else ("quant_transformer", "cuda")

    def _fit(X, y):
        m = NeuralCapacity(tier, adlar, seed=seed)
        kw = {"amp": True, "batch_size": 512} if cihaz == "cuda" else {}
        return m.fit(X, y, max_epochs=30, patience=5, device=cihaz, **kw)
    return _fit


def egitim_test_bolmesi(P: pd.DataFrame, yil: int) -> tuple[pd.Series, pd.Series]:
    """Test = kararı `yil` içinde olan satırlar; eğitim = ETİKETİ yıl başından ÖNCE bilinen satırlar
    (`bilinir` = bar i+2 tarihi). Yalnız `karar < yıl başı` demek, Aralık'ın son iki kararını (etiketi Ocak'ta
    doğan) eğitime sokardı."""
    ilk = pd.Timestamp(f"{yil}-01-01")
    te = (P["karar"] >= ilk) & (P["karar"] < pd.Timestamp(f"{yil + 1}-01-01"))
    return P["bilinir"] < ilk, te


def piyasa_ml_kos(arg: tuple[str, str, list[str]]) -> str:
    """Bir piyasanın enstrümanlarını havuzla; her test yılı için yalnız geçmişle eğit, o yılı tahmin et."""
    piyasa, model_ad, semboller = arg
    _isci_hazirla()
    adlar = ozellik_adlari()
    parca = []
    for s in semboller:
        e = ens_oku(s)
        n = e["n"]
        idx = np.arange(SICAK, n - 2)
        tarih = pd.DatetimeIndex(e["tarih"])
        parca.append(pd.concat([pd.DataFrame({"sembol": s, "i": idx, "karar": tarih[idx],
                                              "bilinir": tarih[idx + 2], "y": e["y"][idx]}),
                                pd.DataFrame(e["X"][idx].astype(float), columns=adlar)], axis=1))
    P = pd.concat(parca, ignore_index=True).sort_values(["karar", "sembol"], kind="stable")
    fit = _model_kur(model_ad, adlar)
    p_all = pd.Series(np.nan, index=P.index)
    auc_yil, cokme = {}, {}
    t_bas = time.perf_counter()
    for yil in range(int(TEST_BAS[:4]), int(P["karar"].max().year) + 1):
        tr_m, te_m = egitim_test_bolmesi(P, yil)
        tr, te = P[tr_m], P[te_m]
        if te.empty or len(tr) < 1000 or tr["y"].nunique() < 2:
            continue
        m = fit(tr[adlar].to_numpy(float), tr["y"].to_numpy(float))
        pr = np.asarray(m.predict_proba(te[adlar].to_numpy(float)), float)
        p_all.loc[te.index] = pr
        auc_yil[yil] = auc(te["y"].to_numpy(), pr)
        cokme[yil] = round(float(np.std(pr)), 6)
    tahmin = {s: np.full(ens_oku(s)["n"], np.nan) for s in semboller}
    for s, g in P.assign(p=p_all).groupby("sembol"):
        tahmin[s][g["i"].to_numpy()] = g["p"].to_numpy()
    out = {"piyasa": piyasa, "model": model_ad, "tahmin": tahmin, "auc_yil": auc_yil,
           "tahmin_std_yil": cokme, "sure_sn": round(time.perf_counter() - t_bas, 1)}
    (SONUC_DIR / "ml").mkdir(parents=True, exist_ok=True)
    with open(SONUC_DIR / "ml" / f"{piyasa}__{model_ad}.pkl", "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return f"{piyasa} {model_ad} {out['sure_sn']}s auc={auc_yil}"


def piyasa_statarb_kos(arg: tuple[str, list[str]]) -> str:
    """Avellaneda-Lee: her gün son 60 günün getiri paneliyle s-skor; s<−1,25 uzun, s>1,25 kısa (1 faktör)."""
    piyasa, semboller = arg
    _isci_hazirla()
    from nihai.ai.statarb import avellaneda_lee_scores
    kap, acl = {}, {}
    for s in semboller:
        df = hazirla(s)
        kap[s] = df["close"]
        acl[s] = df["open"]
    # Farklı borsa tatilleri: kapalı gün = önceki kapanış (getiri 0). Doldurulmasa her 60 günlük
    # pencerede bir tatil o enstrümanı panelden düşürürdü. Listelenmeden önce doldurma YOK.
    C = pd.DataFrame(kap).sort_index().ffill()
    O = pd.DataFrame(acl).reindex(C.index).ffill()
    R = np.log(C).diff()
    ileri = O.shift(-2) / O.shift(-1) - 1.0                      # karar t kapanışı → open[t+1]→open[t+2]
    tarih = C.index
    poz = pd.DataFrame(0.0, index=tarih, columns=C.columns)
    for k in range(60, len(tarih)):
        if tarih[k] < pd.Timestamp(TEST_BAS):
            continue
        pen = R.iloc[k - 59: k + 1]
        kol = [c for c in pen.columns if pen[c].notna().all() and C[c].iloc[k] == C[c].iloc[k]]
        if len(kol) < 3:
            continue
        sonuc = avellaneda_lee_scores(pen[kol].to_numpy(float), n_factors=1, symbols=kol)
        if sonuc is None:
            continue
        for a in sonuc["assets"]:
            poz.at[tarih[k], a["symbol"]] = 1.0 if a["yon"] == "LONG" else (-1.0 if a["yon"] == "SHORT" else 0.0)
    out = {"piyasa": piyasa, "tarih": tarih.to_numpy(), "poz": poz.to_numpy(np.float32),
           "ileri": ileri.to_numpy(np.float32), "semboller": list(C.columns)}
    (SONUC_DIR / "statarb").mkdir(parents=True, exist_ok=True)
    with open(SONUC_DIR / "statarb" / f"{piyasa}.pkl", "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return f"{piyasa} statarb ok"


def canli_sinir_agi_kos(kullanilan: list[tuple[str, str]]) -> dict:
    """Canlı gölge sinir ağı kartlarını (1m'de eğitilmiş) YENİDEN EĞİTMEDEN günlük barlara uygula:
    dağılım-içi pay (NEURAL_SHADOW_MAX_Z=auto kuralı: max(4, 1,5·q99)) ve dağılım-içi satırlarda AUC."""
    from nihai.ai.neural import NeuralCapacity
    adlar = ozellik_adlari()
    sonuc: dict = {}
    for kart in ("neural_deep_mlp_live1m_2026-09-23", "neural_moe_xl_live1m_2026-09-23"):
        m = NeuralCapacity.load(ROOT / "data" / kart)
        if m is None:
            sonuc[kart] = {"neden": "yüklenemedi"}
            continue
        if hasattr(m, "to_device"):
            try:
                m.to_device("cuda")
            except Exception:
                pass
        esik = max(4.0, 1.5 * float(m.train_zmax_q99 or 0.0)) if m.train_zmax_q99 else 6.0
        pz_sonuc: dict = {}
        for piyasa in sorted({p for p, _ in kullanilan}):
            ys, ps, iceride, toplam = [], [], 0, 0
            for pz, s in kullanilan:
                if pz != piyasa:
                    continue
                e = ens_oku(s)
                idx = np.arange(max(SICAK, e["t0"]), e["n"] - 2)
                if idx.size == 0:
                    continue
                # Sinyale bağlı özellikler (rr/kalite/konfluens) + saat modelin KENDİ eğitim
                # ortalamasına konur (z=0): dağılım-dışılık yalnız PİYASA özelliklerinden ölçülsün.
                mu = np.asarray(m._feat_mu, float)
                tam = np.tile(mu, (idx.size, 1))
                for k, ad in enumerate(m.feature_names):
                    if ad in adlar:
                        tam[:, k] = e["X"][idx, adlar.index(ad)]
                    elif ad == "is_long":
                        tam[:, k] = 1.0                           # aday LONG: p = P(uzun kazanır)
                z = m.ood_zmax(tam)
                ic = z <= esik
                toplam += idx.size
                iceride += int(ic.sum())
                if ic.any():
                    ps.append(np.asarray(m.predict_proba(tam[ic]), float))
                    ys.append(e["y"][idx][ic])
            pz_sonuc[piyasa] = {"dagilim_ici_pay": round(iceride / toplam, 4) if toplam else None,
                                "n": toplam,
                                "auc_dagilim_ici": auc(np.concatenate(ys), np.concatenate(ps)) if ys else None}
        sonuc[kart] = {"tier": m.tier, "esik_z": round(esik, 3), "piyasalar": pz_sonuc}
        del m
    return sonuc


# =====================================================================================================
# 6) RAPOR — piyasa × yöntem karnesi
# =====================================================================================================
def _ew(seriler: list[pd.Series]) -> pd.Series:
    """Enstrüman günlük getirilerinin EŞİT AĞIRLIKLI ortalaması (o gün verisi olanlar)."""
    if not seriler:
        return pd.Series(dtype=float)
    return pd.concat(seriler, axis=1).mean(axis=1, skipna=True).dropna()


def _satir(r: pd.Series, bh: pd.Series, gun: int) -> dict:
    ortak = r.index.intersection(bh.index)
    d = seri_metrik(r.loc[ortak].to_numpy(), gun)
    d["alfa"] = nw_alfa(r.loc[ortak].to_numpy(), bh.loc[ortak].to_numpy(), gun)
    return d


def _ai_satiri(E: dict, p_by_sym: dict, bh: pd.Series, gun: int, mal: float, uzun_duz: bool = False) -> dict:
    """Yön olasılığı → pozisyon (p>0,5 uzun, p<0,5 kısa; `uzun_duz` ise kısa yerine nakit) → EW piyasa satırı + AUC."""
    seriler, brut, ys, ps, devir, ic, uzun = [], [], [], [], [], [], []
    for s, e in E.items():
        p = p_by_sym.get(s)
        if p is None:
            continue
        idx = np.arange(e["t0"], e["n"] - 2)
        pp = p[idx]
        ok = np.isfinite(pp)
        if uzun_duz:
            poz = np.where(pp > 0.5, 1.0, 0.0)
        else:
            poz = np.where(pp > 0.5, 1.0, np.where(pp < 0.5, -1.0, 0.0))
        poz[~ok] = 0.0
        tarih = pd.DatetimeIndex(e["tarih"][idx + 1])
        seriler.append(pd.Series(pozisyon_getirisi(poz, e["ileri"][idx], mal), index=tarih))
        brut.append(pd.Series(pozisyon_getirisi(poz, e["ileri"][idx], 0.0), index=tarih))
        devir.append(float(np.abs(np.diff(np.r_[0.0, poz])).mean()))
        ys.append(e["y"][idx][ok])
        ps.append(pp[ok])
        if ok.any():
            uzun.append(float(np.mean(pp[ok] > 0.5)))
        a_ic = auc(ys[-1], ps[-1])
        if a_ic is not None:
            ic.append(a_ic)
    d = _satir(_ew(seriler), bh, gun)
    d["sharpe_brut"] = seri_metrik(_ew(brut).to_numpy(), gun).get("sharpe")
    d["gunluk_devir"] = round(float(np.mean(devir)), 4) if devir else None
    d["auc"] = auc(np.concatenate(ys), np.concatenate(ps)) if ys else None
    # Havuz AUC'si enstrümanlar arası TABAN ORAN farkını da "beceri" sayar (taban kuralının Forex AUC'si
    # 0,522 çıktı: USDTRY neredeyse hep yükselir). ZAMANLAMA becerisi enstrüman-İÇİ AUC'dir.
    d["auc_ic"] = round(float(np.mean(ic)), 4) if ic else None
    d["auc_ic_t"] = (round(float((np.mean(ic) - 0.5) / (np.std(ic, ddof=1) / math.sqrt(len(ic)))), 3)
                     if len(ic) >= 3 and np.std(ic, ddof=1) > 0 else None)
    # Tur-97 (BULGU-97-6): olasılığı sabit OLMAYAN ama hep aynı tarafta kalan model de zamanlamaz —
    # sürüklenmeyi (taban oranı) öğrenmiştir; Sharpe'ı "hep uzun" politikasınınkiyle aynıdır. `cokmus`
    # (std≈0) bunu yakalamaz. Uzun payı ≥ %95 ya da ≤ %5 → tek yönlü; "en iyi AI" seçiminden çıkar.
    d["uzun_payi"] = round(float(np.mean(uzun)), 4) if uzun else None
    d["tek_yonlu"] = bool(d["uzun_payi"] is not None and not 0.05 < d["uzun_payi"] < 0.95)
    d["tur"] = "ai"
    return d


def rapor(kullanilan: list[tuple[str, str]], dislanan: list, canli_sinir: dict | None = None) -> dict:
    from research import piyasalar as P
    mal = maliyet_orani()
    karne: dict = {"uretim": time.strftime("%Y-%m-%d %H:%M"), "test_bas": TEST_BAS, "maliyet_taraf": mal,
                   "ayrim": {"algoritmalar": {k: asdict(v) for k, v in ALGORITMALAR.items()},
                             "ailer": {k: asdict(v) for k, v in AILER.items()},
                             "modul_sinifi": {k: list(v) for k, v in MODUL_SINIFI.items()}},
                   "veri_kaynaklari": P.VERI_KAYNAKLARI, "dislanan": [list(x) for x in dislanan],
                   "piyasalar": {}}
    for piyasa in P.PIYASALAR:
        semboller = [s for pz, s in kullanilan if pz == piyasa]
        if not semboller:
            continue
        gun = int(P.PIYASALAR[piyasa]["gun"])
        E = {s: ens_oku(s) for s in semboller}
        blok: dict = {"semboller": semboller, "yontem": {}, "vol": {}, "hedge_agirlik": {}}
        # --- al-tut kıyasları (iki saat: algoritmalar kapanış→kapanış, AI'lar açılış→açılış)
        bh_kk = _ew([pd.Series(e["bh_kk"], index=pd.DatetimeIndex(e["algo"]["macd"]["tarih"])) for e in E.values()])
        ai_bh = []
        for e in E.values():
            idx = np.arange(e["t0"], e["n"] - 2)
            ai_bh.append(pd.Series(e["ileri"][idx], index=pd.DatetimeIndex(e["tarih"][idx + 1])))
        bh_aa = _ew(ai_bh)
        blok["al_tut"] = seri_metrik(bh_aa.to_numpy(), gun)
        # --- algoritmalar (+ canlı meta kapı)
        for m in (*MODLAR_ALGO, "canli_meta_kapi"):
            seriler = [pd.Series(e["algo"][m]["r"], index=pd.DatetimeIndex(e["algo"][m]["tarih"])) for e in E.values()]
            d = _satir(_ew(seriler), bh_kk, gun)
            d["islem"] = int(sum(e["algo"][m]["islem"] for e in E.values()))
            d["tur"] = "ai" if m == "canli_meta_kapi" else "algoritma"
            blok["yontem"][m] = d

        # --- AI (yön): enstrüman-yerel + havuzlanmış ML
        for ad in ("rejim_hmm", "ftrl", "hedge_algoritma"):
            blok["yontem"][ad] = _ai_satiri(E, {s: e["ai"][ad] for s, e in E.items()}, bh_aa, gun, mal,
                                             uzun_duz=(ad == "rejim_hmm"))
        for ad in (*ML_MODELLERI, "sinir_transformer"):
            f = SONUC_DIR / "ml" / f"{piyasa}__{ad}.pkl"
            if not f.exists():
                continue
            with open(f, "rb") as fh:
                ml = pickle.load(fh)
            d = _ai_satiri(E, ml["tahmin"], bh_aa, gun, mal)
            yil = [v for v in ml["auc_yil"].values() if v is not None]
            d["auc_yil_ort"] = round(float(np.mean(yil)), 4) if yil else None
            d["auc_yil_t"] = (round(float((np.mean(yil) - 0.5) / (np.std(yil, ddof=1) / math.sqrt(len(yil)))), 3)
                              if len(yil) >= 3 and np.std(yil, ddof=1) > 0 else None)
            d["tahmin_std_medyan"] = float(np.median(list(ml["tahmin_std_yil"].values()))) if ml["tahmin_std_yil"] else None
            # ÇÖKME (Tur-94): her satıra ~aynı olasılık → pozisyon sabit (≈ al-tut ya da tersi), AUC yuvarlama
            # gürültüsü. Çökmüş model "en iyi AI" seçilemez; tabloda ÇÖKTÜ yazar.
            d["cokmus"] = bool(d["tahmin_std_medyan"] is not None and d["tahmin_std_medyan"] < 1e-3)
            blok["yontem"][ad] = d
        # --- taban kuralı (AI DEĞİL; AI'ların geçmesi gereken çıta): eğitim taban oranı > 0,5 ise uzun, değilse kısa
        taban = {}
        for s, e in E.items():
            p = np.full(e["n"], np.nan)
            tarih = pd.DatetimeIndex(e["tarih"])
            for yil in sorted(set(tarih[e["t0"]:].year)):
                ilk = pd.Timestamp(f"{yil}-01-01")
                gecmis = e["y"][(tarih < ilk - pd.Timedelta(days=5)) & (e["y"] >= 0)][SICAK:]
                if gecmis.size:
                    p[(tarih.year == yil)] = 0.51 if gecmis.mean() > 0.5 else 0.49
            taban[s] = p
        blok["yontem"]["taban_kurali"] = _ai_satiri(E, taban, bh_aa, gun, mal)
        blok["yontem"]["taban_kurali"]["tur"] = "referans"
        # --- statarb
        f = SONUC_DIR / "statarb" / f"{piyasa}.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                sa = pickle.load(fh)
            poz = np.nan_to_num(sa["poz"].astype(float))
            il = np.nan_to_num(sa["ileri"].astype(float))
            onceki = np.vstack([np.zeros((1, poz.shape[1])), poz[:-1]])
            gr = (poz * il - mal * np.abs(poz - onceki)).mean(axis=1)
            tarih = pd.DatetimeIndex(sa["tarih"])
            maske = tarih >= pd.Timestamp(TEST_BAS)
            # getiri open[t+1]→open[t+2] bir sonraki tarihe aittir
            r = pd.Series(gr[maske][:-2], index=tarih[maske][1:-1])
            d = _satir(r, bh_aa, gun)
            d["tur"] = "ai"
            d["maruz_pay"] = round(float((np.abs(poz[maske]) > 0).mean()), 4)
            blok["yontem"]["statarb_ou"] = d
        # --- oynaklık AI'ları: EWMA tabanına karşı QLIKE farkı (negatif = model daha iyi)
        for ad in ("garch", "gjr", "har", "har_olcekli"):
            if any(ad not in e["vol"]["L"] for e in E.values()):
                continue
            farklar, kazanan = [], 0
            for e in E.values():
                L = e["vol"]["L"]
                dd = pd.Series(L[ad] - L["ewma"], index=pd.DatetimeIndex(e["vol"]["tarih"]))
                t = nw_ortalama_t(dd.to_numpy())
                if t is not None and t < -1.96:
                    kazanan += 1
                farklar.append(dd)
            ort = _ew(farklar)
            blok["vol"][f"vol_{ad}"] = {"qlike_fark_ort": round(float(ort.mean()), 5), "dm_t": nw_ortalama_t(ort.to_numpy()),
                                        "ewma_yi_yenen_enstruman": kazanan, "enstruman": len(E),
                                        "negatif_tahmin": int(sum(e["vol"]["negatif_tahmin"][ad] for e in E.values()))}
        blok["hedge_agirlik"] = {s: e["hedge_agirlik"] for s, e in E.items()}
        blok["rv_vekil_orani"] = {s: e["vol"].get("rv_vekil_orani") for s, e in E.items()}
        karne["piyasalar"][piyasa] = blok
    if canli_sinir is not None:
        karne["canli_sinir_agi"] = canli_sinir
    return karne


YON_AILERI = ("meta_lojistik", "meta_gbm", "meta_xgb", "sinir_mlp", "sinir_transformer", "ftrl",
              "hedge_algoritma", "rejim_hmm", "statarb_ou", "canli_meta_kapi")


def _f(x, nd: int = 2) -> str:
    return "—" if x is None else (f"{x:+.{nd}f}" if isinstance(x, float) else str(x))


def bonferroni_esigi(n_test: int, alfa: float = 0.05) -> float:
    """İki yönlü Bonferroni |t| eşiği (normal yaklaşım)."""
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - alfa / (2 * max(1, n_test)))


def ozet_md(karne: dict) -> str:
    """Karne JSON'u → rapora giren markdown tabloları."""
    pz = karne["piyasalar"]
    satir: list[str] = []
    # --- çoklu test: yön yöntemleri × piyasa hücreleri (algoritma + AI)
    hucre = [(p, m, d) for p, b in pz.items() for m, d in b["yontem"].items() if d.get("tur") in ("ai", "algoritma")]
    esik = bonferroni_esigi(len(hucre))
    gecen = [(p, m, d["alfa"]["alfa_t"]) for p, m, d in hucre
             if d.get("alfa") and d["alfa"].get("alfa_t") is not None and abs(d["alfa"]["alfa_t"]) >= esik]
    satir += [f"Yön hücresi: {len(hucre)} · Bonferroni |t| eşiği {esik:.2f} · eşiği geçen: "
              + (", ".join(f"{p}/{m} (t={t:+.2f})" for p, m, t in gecen) or "YOK"), ""]
    # --- piyasa özeti
    satir += ["| Piyasa | enstr. | Al-tut Sharpe | En iyi algoritma (Sharpe · alfa t) | En iyi yön AI'ı "
              "(Sharpe · AUC · alfa t) | En iyi oynaklık AI'ı (ΔQLIKE · DM t) |",
              "|---|---|---|---|---|---|"]
    for p, b in pz.items():
        algo = {m: d for m, d in b["yontem"].items() if d.get("tur") == "algoritma" and d.get("sharpe") is not None}
        ai = {m: d for m, d in b["yontem"].items()
              if m in YON_AILERI and d.get("sharpe") is not None and not d.get("cokmus")
              and not d.get("tek_yonlu")}
        ba = max(algo.items(), key=lambda kv: kv[1]["sharpe"]) if algo else None
        bi = max(ai.items(), key=lambda kv: kv[1]["sharpe"]) if ai else None
        bv = min(b["vol"].items(), key=lambda kv: kv[1]["qlike_fark_ort"]) if b["vol"] else None
        satir.append(
            f"| {p} | {len(b['semboller'])} | {_f(b['al_tut'].get('sharpe'))} | "
            + (f"{ba[0]} ({_f(ba[1]['sharpe'])} · {_f((ba[1].get('alfa') or {}).get('alfa_t'))})" if ba else "—")
            + " | "
            + (f"{bi[0]} ({_f(bi[1]['sharpe'])} · {_f(bi[1].get('auc'), 3)} · "
               f"{_f((bi[1].get('alfa') or {}).get('alfa_t'))})" if bi else "—")
            + " | "
            + (f"{bv[0]} ({_f(bv[1]['qlike_fark_ort'], 4)} · {_f(bv[1]['dm_t'])})" if bv else "—") + " |")
    # --- yön AI × piyasa: Sharpe (net) / AUC
    for baslik, alan, nd in (("Yön AI'ları — net Sharpe", "sharpe", 2), ("Yön AI'ları — maliyet ÖNCESİ Sharpe",
                                                                         "sharpe_brut", 2),
                             ("Yön AI'ları — havuz AUC (ertesi gün yönü)", "auc", 3),
                             ("Yön AI'ları — enstrüman-İÇİ AUC (zamanlama becerisi)", "auc_ic", 3),
                             ("Yön AI'ları — al-tuta karşı alfa NW t", "alfa_t", 2)):
        adlar = [a for a in (*YON_AILERI, "taban_kurali") if any(a in b["yontem"] for b in pz.values())]
        satir += ["", f"**{baslik}**", "", "| Piyasa | " + " | ".join(adlar) + " |",
                  "|---|" + "---|" * len(adlar)]
        for p, b in pz.items():
            h = []
            for a in adlar:
                d = b["yontem"].get(a) or {}
                v = (d.get("alfa") or {}).get("alfa_t") if alan == "alfa_t" else d.get(alan)
                h.append("ÇÖKTÜ" if d.get("cokmus") and alan.startswith("auc") else _f(v, nd))
            satir.append(f"| {p} | " + " | ".join(h) + " |")
    # --- algoritmalar × piyasa
    satir += ["", "**Algoritmalar — net Sharpe (AI bileşeni kapalı)**", "",
              "| Piyasa | " + " | ".join(MODLAR_ALGO) + " |", "|---|" + "---|" * len(MODLAR_ALGO)]
    for p, b in pz.items():
        satir.append(f"| {p} | " + " | ".join(_f(b["yontem"].get(m, {}).get("sharpe")) for m in MODLAR_ALGO) + " |")
    # --- oynaklık AI'ları
    satir += ["", "**Oynaklık AI'ları — EWMA tabanına karşı QLIKE farkı (negatif = daha iyi) · DM t · "
              "EWMA'yı anlamlı yenen enstrüman**", "", "| Piyasa | GARCH | GJR | HAR | HAR (seviye düzeltmeli, teşhis) |",
              "|---|---|---|---|---|"]
    for p, b in pz.items():
        h = [f"{_f(v['qlike_fark_ort'], 4)} · {_f(v['dm_t'])} · {v['ewma_yi_yenen_enstruman']}/{v['enstruman']}"
             for v in (b["vol"].get(f"vol_{a}", {}) for a in ("garch", "gjr", "har", "har_olcekli")) if v]
        satir.append(f"| {p} | " + " | ".join(h) + " |")
    # --- Hedge: AI hangi algoritmaya güveniyor
    satir += ["", "**Hedge AI'ının son ağırlıkları (piyasa ortalaması)**", ""]
    for p, b in pz.items():
        w = pd.DataFrame(list(b["hedge_agirlik"].values())).mean()
        satir.append(f"- {p}: " + ", ".join(f"{k} {v:.3f}" for k, v in w.sort_values(ascending=False).items()
                                             if v >= 0.005))
    cs = karne.get("canli_sinir_agi")
    if cs:
        satir += ["", "**Canlı sinir ağları (1m kartları, yeniden eğitilmeden) — dağılım-içi pay · AUC**", ""]
        for kart, v in cs.items():
            if "piyasalar" not in v:
                satir.append(f"- {kart}: {v}")
                continue
            satir.append(f"- {kart} ({v['tier']}, z eşiği {v['esik_z']}): "
                         + "; ".join(f"{p} {_f(d['dagilim_ici_pay'], 3)} · {_f(d['auc_dagilim_ici'], 3)}"
                                     for p, d in v["piyasalar"].items()))
    return "\n".join(satir)


# =====================================================================================================
# CLI
# =====================================================================================================
def _havuz(fn, gorevler: list, isci: int) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t = time.time()
    if isci <= 1:
        for g in gorevler:
            print(fn(g), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=isci) as ex:
            fut = {ex.submit(fn, g): g for g in gorevler}
            for f in as_completed(fut):
                try:
                    print(f.result(), flush=True)
                except Exception as e:  # noqa: BLE001 — bir enstrümanın hatası koşuyu durdurmaz, YAZILIR
                    print(f"HATA {fut[f]}: {type(e).__name__}: {e}", flush=True)
    print(f"bitti {round(time.time() - t, 1)} sn", flush=True)


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="AI'lar ve algoritmalar — çoklu piyasa koşusu (Tur-96)")
    ap.add_argument("--asama", required=True, choices=["1", "vol", "2", "3", "rapor", "ozet"])
    ap.add_argument("--isci", type=int, default=8)
    ap.add_argument("--piyasa")
    ap.add_argument("--sembol")
    ap.add_argument("--eksik", action="store_true", help="aşama 1: yalnız sonucu olmayan enstrümanlar")
    a = ap.parse_args()
    from research import piyasalar as P
    P.yukle("^VIX", fetch=True)
    kullanilan, dislanan = enstrumanlar(a.piyasa, a.sembol)
    if a.asama == "1":
        print(f"{len(kullanilan)} enstrüman, dışlanan {len(dislanan)}: {dislanan}", flush=True)
        gorev = [g for g in kullanilan
                 if not (a.eksik and (SONUC_DIR / "ens" / f"{_guvenli(g[1])}.pkl").exists())]
        _havuz(enstruman_kos, gorev, a.isci)
    elif a.asama == "vol":
        _havuz(vol_yenile, kullanilan, a.isci)
    elif a.asama == "2":
        gorev = []
        for piyasa in P.PIYASALAR:
            ss = [s for pz, s in kullanilan if pz == piyasa]
            if ss:
                gorev += [(piyasa, m, ss) for m in ML_MODELLERI
                          if not (a.eksik and (SONUC_DIR / "ml" / f"{piyasa}__{m}.pkl").exists())]
        gorev.sort(key=lambda g: g[1] != "meta_gbm")          # en uzun işler (GBM) önce
        _havuz(piyasa_ml_kos, gorev, a.isci)
        _havuz(piyasa_statarb_kos, [(pz, [s for p2, s in kullanilan if p2 == pz]) for pz in P.PIYASALAR
                                    if any(p2 == pz for p2, _ in kullanilan)
                                    and not (a.eksik and (SONUC_DIR / "statarb" / f"{pz}.pkl").exists())],
               a.isci)
    elif a.asama == "3":
        gorev = [(pz, "sinir_transformer", [s for p2, s in kullanilan if p2 == pz]) for pz in P.PIYASALAR
                 if any(p2 == pz for p2, _ in kullanilan)
                 and not (a.eksik and (SONUC_DIR / "ml" / f"{pz}__sinir_transformer.pkl").exists())]
        _havuz(piyasa_ml_kos, gorev, 1)
        cs = canli_sinir_agi_kos(kullanilan)
        (SONUC_DIR / "canli_sinir.json").write_text(json.dumps(cs, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(cs, ensure_ascii=False)[:2000])
    elif a.asama == "ozet":
        print(ozet_md(json.loads(KARNE.read_text(encoding="utf-8"))))
    else:
        cs_yol = SONUC_DIR / "canli_sinir.json"
        cs_kayit = json.loads(cs_yol.read_text(encoding="utf-8")) if cs_yol.exists() else None
        karne = rapor(kullanilan, dislanan, cs_kayit)
        karne["parmak_izi"] = girdi_parmak_izi(sorted({s for _, s in kullanilan}))   # Tur-99
        KARNE.write_text(json.dumps(karne, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"karne → {KARNE}")


if __name__ == "__main__":
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    _cli()
