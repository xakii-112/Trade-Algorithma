"""api_quant.py — QUANT MODELLERİ'nin GÖRSEL yüzeyi (Tur-56).

`nihai/ai/quant_math.py`, `options_math.py`, `advanced_math.py`, `microstructure.py`,
`portfolio.py`, `regime_hmm.py` ve `nihai/risk/*` içinde ÇALIŞAN hesaplayıcılar vardı;
panoda ya hiç görünmüyorlardı ya da TEK BİR SAYI olarak görünüyorlardı. Bir volatilite
modeli tek sayıya indirgenince "anlaşılır" olmaz — GÖRÜLMESİ gerekir: koşullu vol yolu,
dağılımın kuyruğu, gülümsemenin eğimi, ağırlıkların dağılımı.

Bu modül YENİ MATEMATİK EKLEMEZ. Var olan fonksiyonları çağırır ve çıktılarını
GRAFİĞE HAZIR seri/ızgara biçiminde döndürür (Chart.js doğrudan çizebilsin diye).

ANAYASA (CLAUDE.md "AI katmanı"): buradaki hiçbir uç sinyal/eşik/META_GATE/execution'a
bağlanmaz — hepsi SALT-OKUNUR RAPOR. Uçlar FAIL-SOFT: veri yoksa 500 atmaz,
`{"ok": false, "reason": ...}` döner ve pano "ölçülmedi" der (ACI DERSLER: ölçüm
yokluğu bir ölçüm gibi raporlanmaz — uydurulmuş varsayılan basmaktansa boş bırakılır).

Uçlar:
  /api/quant/catalog     — model dizini (ne anlatır + hangi fonksiyon + hangi grafik)
  /api/quant/risk        — getiri dağılımı + VaR/CVaR/EVT + düşüş eğrisi + oranlar + hafıza
  /api/quant/vol         — 5 vol tahmincisi + vol konisi + GARCH(1,1) koşullu yol/öngörü
  /api/quant/regime      — kural-tabanlı rejim + Gauss HMM olasılıkları + BOCPD kırılmaları
  /api/quant/mc          — Monte Carlo GBM yelpazesi + terminal dağılım + blok-bootstrap
  /api/quant/options     — Black-Scholes fiyat/Greek eğrileri + vade-sonu ödeme profili
  /api/quant/smile       — canlı zincirden IV gülümsemesi + SVI uyumu + RND + vade yapısı
  /api/quant/portfolio   — 7 tahsis yöntemi + risk katkısı + korelasyon + PCA + etkin sınır
  /api/quant/micro       — imza grafiği + sıçrama ayrışımı + makas/etki/VPIN + HAR-RV
  /api/quant/execution   — Almgren-Chriss optimal tasfiye yörüngesi
  /api/quant/kelly       — Kelly büyüme eğrisi (log-servet) + kesirli Kelly noktası
  /api/quant/drift       — SR 11-7 özellik sürüklenmesi (PSI/KS/Wasserstein/Fisher-Rao)
  /api/quant/econometrics— DCC-GARCH korelasyon yolu + FIGARCH uzun hafıza + MODWT
  /api/quant/tda         — Vietoris-Rips H0/H1 kalıcılığı + manzara normları
  /api/quant/dispersion  — ETF sepetinden örtük korelasyon + Cboe COR1M + ileri vol
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field, FiniteFloat

router = APIRouter()

_MAX_LOOKBACK = 1500
_MIN_LOOKBACK = 60


# ---------------------------------------------------------------------------
# Ortak yardımcılar
# ---------------------------------------------------------------------------
def _provider():
    """Panonun paylaşılan sağlayıcısı (geç import — web_dashboard bu modülü içe aktarır,
    modül düzeyinde import döngü olurdu; çağrı anında güvenli)."""
    from nihai.web.web_dashboard import _analysis_provider
    return _analysis_provider()


def _f(v: Any, nd: int = 6) -> float | None:
    """Sonlu float'a yuvarla; NaN/inf/None/dönüştürülemez → None (JSON'da null).

    Grafik katmanı null'ı ÇİZMEZ (boşluk bırakır) — bu bilinçli: ölçülmemiş bir noktayı
    sıfır veya son değerle doldurmak, olmayan bir ölçümü var gibi gösterirdi."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _arr(a: Any, nd: int = 6) -> list[float | None]:
    """numpy dizisi / liste → JSON listesi (NaN → null)."""
    if a is None:
        return []
    return [_f(v, nd) for v in np.asarray(a, dtype=float).ravel()]


def _nums(d: Any, nd: int = 6) -> dict | None:
    """Sonuç sözlüğünün skaler alanlarını JSON'a hazırla; dizi alanları düşer.

    `None` alanlar KORUNUR (düşürülmez): `roll_spread`/`corwin_schultz` "ölçülemedi"yi
    tam da `spread: None` ile söyler — anahtarı atmak, pano açısından alanın hiç
    olmamasıyla aynı görünür ve "ölçüm yok" mesajı kaybolurdu. `bool` de bool kalır
    (`reliable: false`, `1.0/0.0` değil)."""
    if not isinstance(d, dict):
        return None
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None or isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = _f(v, nd)
    return out


def _ppy(interval: str) -> int:
    """Bar aralığı → yıllık bar sayısı. Backtest'teki tabloyu YENİDEN YAZMAZ, çağırır."""
    from nihai.execution.backtest import _periods_per_year
    return int(_periods_per_year(interval))


def _ohlc(symbol: str, interval: str, lookback: int) -> dict | None:
    """Sağlayıcıdan bar çek → {'dates','open','high','low','close','volume'} (np dizileri).
    Veri yoksa/alınamazsa None — çağıran uç bunu `reason` ile bildirir."""
    lb = max(_MIN_LOOKBACK, min(int(lookback), _MAX_LOOKBACK))
    try:
        df = _provider().get_bars(symbol.upper().strip(), interval, lb)
    except Exception as e:  # noqa: BLE001 — rapor katmanı; sağlayıcı hatası sekmeyi düşürmesin
        logger.warning(f"[api_quant] bar çekilemedi {symbol} {interval}: {e}")
        return None
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return None
    col = lambda c: df[c].to_numpy(dtype=float) if c in df.columns else None  # noqa: E731
    return {
        "dates": [str(ix)[:19] for ix in df.index],
        "open": col("open"), "high": col("high"), "low": col("low"),
        "close": df["close"].to_numpy(dtype=float), "volume": col("volume"),
    }


def _bars_raw(symbol: str, interval: str, lookback: int):
    """Ham bar DataFrame'i (index + varsa `trade_count`/`vwap`). `_ohlc` yalnız numpy
    dizisi döndürür; akış ölçüleri ZAMAN DAMGASINA (gün-içi dilim) ve Alpaca'nın ek
    kolonlarına ihtiyaç duyar."""
    lb = max(_MIN_LOOKBACK, min(int(lookback), _MAX_LOOKBACK))
    try:
        df = _provider().get_bars(symbol.upper().strip(), interval, lb)
    except Exception as e:  # noqa: BLE001 — rapor katmanı
        logger.warning(f"[api_quant] bar çekilemedi {symbol} {interval}: {e}")
        return None
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return None
    return df


def _asset_flags(symbol: str) -> dict | None:
    """Açığa satış bayrakları — okuyucu `data_provider.asset_flags` (TEK kaynak)."""
    from nihai.data.data_provider import asset_flags as _flags
    try:
        return _flags(symbol, _provider())
    except Exception:  # noqa: BLE001 — pano katmanı, sekmeyi düşürmesin
        return None


def _slots(index, interval: str):
    """Bar zaman damgalarını gün-içi DİLİM etiketine çevir (RVOL kıyas anahtarı).

    Gün-içi barda dakika-of-day, günlük barda hafta günü. Kıyas AYNI dilimle yapılır:
    açılışın ilk 15 dakikası öğlene göre zaten kat kat hacimlidir."""
    import pandas as _pd
    idx = _pd.DatetimeIndex(index)
    if interval.endswith("d"):
        return idx.dayofweek.to_numpy()
    return (idx.hour * 60 + idx.minute).to_numpy()


def _histogram(x: Any, bins: int = 40) -> dict:
    """Yoğunluk histogramı → {'centers','counts','density','bin_width'}."""
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {"centers": [], "counts": [], "density": [], "bin_width": None}
    counts, edges = np.histogram(arr, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = float(edges[1] - edges[0]) if edges.size > 1 else 1.0
    dens = counts / max(1e-12, counts.sum() * width)
    return {"centers": _arr(centers), "counts": [int(c) for c in counts],
            "density": _arr(dens, 4), "bin_width": _f(width)}


def _fail(reason: str, **extra: Any) -> dict:
    return {"ok": False, "reason": reason, **extra}


def _guard(fn, *args, **kwargs):
    """Tek bir alt-hesap patlarsa TÜM kartı düşürme: None döndür, ayrıntıyı log'a yaz."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 — bilinçli geniş yakalama (rapor katmanı)
        logger.debug(f"[api_quant] {getattr(fn, '__name__', fn)} hata: {e}")
        return None


# ===========================================================================
# 0) KATALOG — hangi model neyi anlatır (pano kart başlıklarını buradan okur)
# ===========================================================================
_CATALOG: list[dict[str, Any]] = [
    {
        "id": "risk", "title": "Getiri Dağılımı & Kuyruk Riski",
        "endpoint": "/api/quant/risk", "chart": "histogram + su-altı (düşüş) eğrisi",
        "soru": "Kötü gün ne kadar kötü olabilir?",
        "anlatim": "Getirilerin histogramı çizilir; üstüne VaR (o güven düzeyinde kayıp "
                   "EŞİĞİ) ve CVaR (eşiğin ÖTESİNDEKİ ortalama kayıp) dikey çizgi olarak "
                   "düşer. Aynı olay üç FARKLI varsayımla ölçülür: tarihsel (ham veri), "
                   "Cornish-Fisher (çarpıklık/basıklık düzeltmeli) ve EVT-GPD (yalnız "
                   "kuyruğu modelleyen uç-değer teorisi). Üçü ayrışıyorsa dağılım normal "
                   "DEĞİLDİR ve normal varsayan her hesap kuyruğu hafife alır.",
        "oku": "EVT şekil parametresi (ξ) > 0 ise kuyruk KALIN — normal dağılımın "
               "öngördüğünden daha sık ve daha derin kayıp beklenir.",
        "formul": ["quant_math.var_historical", "quant_math.cvar_historical",
                   "quant_math.var_cornish_fisher", "quant_math.evt_gpd_var_es",
                   "quant_math.max_drawdown", "quant_math.ulcer_index",
                   "quant_math.sharpe", "quant_math.sortino", "quant_math.calmar",
                   "quant_math.omega", "quant_math.hurst_exponent",
                   "quant_math.variance_ratio", "quant_math.ou_half_life",
                   "quant_math.adf_test", "quant_math.permutation_entropy",
                   "quant_math.sample_entropy"],
    },
    {
        "id": "vol", "title": "Volatilite Modelleri (GARCH + koni)",
        "endpoint": "/api/quant/vol", "chart": "tahminci çubukları + koni + GARCH yolu",
        "soru": "Oynaklık şu an ne, yarın ne olur?",
        "anlatim": "Beş tahminci AYNI veriye uygulanır: kapanış-kapanış, Parkinson, "
                   "Garman-Klass, Yang-Zhang (üçü OHLC menzilini kullanır = aynı veriden "
                   "daha verimli tahmin) ve EWMA. Vol KONİSİ farklı pencerelerde geçmişte "
                   "görülen aralığı gösterir — bugünkü vol koninin neresinde? GARCH(1,1) "
                   "koşullu vol YOLUNU çizer ve ufka doğru koşulsuz varyansa dönüşü "
                   "tahmin eder (volatilite kümelenmesi + ortalamaya dönüş).",
        "oku": "α+β (kalıcılık) 1'e ne kadar yakınsa şok o kadar uzun yaşar; yarı-ömür "
               "kaç barda etkinin yarıya indiğini söyler. HABER-ETKİSİ eğrisi simetrikse "
               "asimetri yoktur; GJR/EGARCH eğrisi sola yatıksa kötü haber oynaklığı "
               "daha çok büyütür (LR > 3,84 ise fark χ²(1) %95'te anlamlıdır).",
        "formul": ["quant_math.vol_close_to_close", "quant_math.vol_parkinson",
                   "quant_math.vol_garman_klass", "quant_math.vol_yang_zhang", "quant_math.volatility_cone",
                   "quant_math.ewma_vol", "quant_math.garch11_fit",
                   "quant_math.garch11_forecast", "quant_math.gjr_garch_fit",
                   "quant_math.egarch_fit", "quant_math.asym_garch_forecast",
                   "quant_math.haar_dwt_vol", "quant_math.annualize_vol"],
    },
    {
        "id": "regime", "title": "Rejim & Kırılma Noktaları",
        "endpoint": "/api/quant/regime", "chart": "olasılık şeridi + kırılma serisi",
        "soru": "Hangi rejimdeyiz, ne kadar eminiz, ne zaman değişti?",
        "anlatim": "Kural-tabanlı etiket (trend × vol kovası) KESKİNDİR — 'up_high'sın ya "
                   "da değilsin. Gauss HMM aynı seriye rejim OLASILIĞI verir (nedensel "
                   "filtre: yalnız o ana kadarki veri), geçiş matrisi + beklenen rejim "
                   "SÜRESİ çıkarır. BOCPD ise kırılma-noktası posteriorunu çizer: MAP "
                   "run-length zirveye tırmanır, kırılmada 1'e düşer, yeniden tırmanır.",
        "oku": "Rejim olasılığı %50'lerde geziyorsa rejim BELİRSİZDİR; keskin etiket bu "
               "belirsizliği sessizce sıfıra yuvarlar — en çok da rejim sınırında yanılır. "
               "VADE YAPISI kartı piyasa GENELİDİR (sembolden bağımsız) ve bilerek yalnız "
               "RAPORDUR: dış liste (Madde 9) backwardation'ın swing alımlarını KİLİTLEMESİNİ "
               "öneriyordu, ama bu bir kenar iddiasıdır ve ölçülmemiş hiçbir bileşen bu "
               "sistemde karara giremez (ACI DERS #5). Kaynak göçü de YAPILMADI ve nedeni "
               "ölçüldü (Tur-74): VIX rejim KAPISI kararı yfinance ↔ CBOE arasında "
               "2515 günün 0'ında ayrışıyor, ama `meta_features` VIX'i modelin ÖZELLİK "
               "vektörüne sokar ve günlerin yalnız %3,9'u bit-denktir → kayıtlı modelde "
               "p_win en fazla 0,0506 oynar, eşiğe yakın bir kararı çevirmeye yeter.",
        "formul": ["quant_math.classify_regime", "regime_hmm.hmm_select_states",
                   "regime_hmm.gaussian_hmm_fit", "regime_hmm.hmm_filter",
                   "regime_hmm.hmm_transition_summary", "regime_hmm.hmm_state_labels",
                   "regime_hmm.hmm_next_state_probs", "advanced_math.bocpd",
                   "quant_math.critical_slowing_down", "quant_math.mfdfa",
                   "cboe_data.term_structure"],
    },
    {
        "id": "mc", "title": "Monte Carlo Simülasyonu",
        "endpoint": "/api/quant/mc", "chart": "yelpaze + terminal histogram",
        "soru": "Ufuk sonunda fiyat nerede olabilir?",
        "anlatim": "GBM ile binlerce fiyat yolu üretilir; yelpaze grafiği yüzdelik "
                   "dilimleri (p5/p25/p50/p75/p95) gösterir — TEK bir tahmin değil, "
                   "olasılık huzmesi. Terminal histogram ufuk sonundaki sonuçların "
                   "dağılımıdır. Blok-bootstrap ise GBM'in normallik varsayımını "
                   "KULLANMAZ: gerçek getiri bloklarını yeniden örnekler; ikisinin "
                   "ayrışması varsayımın maliyetini gösterir.",
        "oku": "Yelpaze zamanla √t ile açılır. p5 çizgisi stop seviyenizin altındaysa "
               "senaryo 'olağan' aralıktadır — sürpriz değildir.",
        "formul": ["quant_math.mc_gbm_paths", "quant_math.mc_bootstrap_terminal",
                   "quant_math.annualize_return", "quant_math.annualize_vol"],
    },
    {
        "id": "options", "title": "Black-Scholes Hesaplayıcı & Greekler",
        "endpoint": "/api/quant/options", "chart": "fiyat/Greek eğrileri + ödeme profili",
        "soru": "Bu opsiyon ne eder, girdiler değişince ne olur?",
        "anlatim": "Spot ızgarası boyunca opsiyon değeri, vade-sonu ödeme profili ve tüm "
                   "Greekler (delta/gamma/theta/vega/rho + vanna/volga/ultima/charm/speed/zomma/"
                   "color) EĞRİ olarak çizilir. Greek'i sayı olarak görmek 'gamma 0,03' "
                   "der; eğri olarak görmek 'gamma ATM'de tepe yapar, kanatlarda "
                   "sıfırlanır, vadeye yaklaştıkça sivrilir' der. Prim girilirse implied "
                   "vol de geri çözülür.",
        "oku": "Fiyat eğrisi ile ödeme profili arasındaki dikey mesafe ZAMAN DEĞERİdir; "
               "vade yaklaştıkça bu mesafe kapanır (theta).",
        "formul": ["quant_math.bs_price", "quant_math.bs_greeks", "quant_math.greek_neutral_hedge",
                   "options_math.bs_greeks_extended", "options_math.bs_no_arb_bounds",
                   "options_math.implied_vol", "risk.options_analyzer.compute_greeks"],
    },
    {
        "id": "smile", "title": "Volatilite Gülümsemesi (SVI) & Risk-Nötr Yoğunluk",
        "endpoint": "/api/quant/smile", "chart": "gülümseme + SVI uyumu + yoğunluk",
        "soru": "Piyasa hangi dağılımı fiyatlıyor?",
        "anlatim": "Canlı zincirden implied vol'ler çekilir, log-moneyness'a karşı çizilir "
                   "ve SVI parametrik eğrisi uydurulur. Durrleman g fonksiyonu NEGATİFE "
                   "düşerse kelebek arbitrajı vardır — uyum kabul edilemez. "
                   "Breeden-Litzenberger ile call fiyatlarının strike'a göre ikinci "
                   "türevinden RİSK-NÖTR YOĞUNLUK çıkarılır: piyasanın fiilen fiyatladığı "
                   "dağılım budur. Vade yapısı takvim arbitrajını kontrol eder.",
        "oku": "Sol kanat sağdan dikse (negatif RR25) piyasa çöküş sigortasına prim "
               "ödüyor demektir. Negatif yoğunluk = zincirde arbitraj/kirli veri. "
               "Varyans swap K_vol'ü ATM IV'nin üstündeyse fark doğrudan gülümsemenin "
               "katkısıdır; kanat kırpımı 1'e yakınsa kuyruk örneklenmemiştir.",
        "formul": ["options_math.implied_vol", "options_math.svi_fit",
                   "options_math.svi_total_variance", "options_math.svi_durrleman_g",
                   "options_math.svi_no_arbitrage", "options_math.smile_metrics",
                   "options_math.risk_neutral_density", "options_math.vol_term_structure",
                   "options_math.put_call_parity", "options_math.dupire_local_vol_svi",
                   "options_math.variance_swap_strike"],
    },
    {
        "id": "portfolio", "title": "Portföy Tahsis Yöntemleri",
        "endpoint": "/api/quant/portfolio", "chart": "ağırlık çubukları + sınır + ısı haritası",
        "soru": "Sermaye nasıl dağıtılmalı — ve yöntemler ne kadar ayrışıyor?",
        "anlatim": "Tahsis yöntemleri seçilen Ledoit–Wolf veya RMT kovaryansıyla koşulur (min-var, maks-Sharpe, HRP, ERC, "
                   "Black-Litterman, CVaR-LP, çok-varlıklı Kelly) ve ağırlıkları yan yana "
                   "çizilir. Risk KATKISI ağırlıktan farklıdır: %10 ağırlık %40 risk "
                   "taşıyabilir. Etkin sınır risk-getiri ödünleşimini, korelasyon ısı "
                   "haritası ve PCA çeşitlendirmenin GERÇEK olup olmadığını gösterir.",
        "oku": "İlk PCA bileşeni varyansın %80'ini açıklıyorsa portföy aslında TEK "
               "pozisyondur; çeşitlendirme isimde vardır, riskte yoktur. λ⁺ ALTINDA "
               "kalan özdeğer sayısı ne kadar çoksa korelasyon matrisi o kadar "
               "gürültüdür; MST'nin merkez düğümü şokun yayıldığı yerdir. "
               "BLACK-LITTERMAN satırı okunmadan önce `BL piyasa önseli` kartına bakın: "
               "π = δΣw_mkt kimliği w_mkt'in GERÇEK kapitalizasyon ağırlığı olmasını "
               "varsayar. Kart 'ters-vol PROXY' diyorsa o satır piyasa dengesini DEĞİL "
               "bir DÜŞÜK-OYNAKLIK önselini anlatır — ölçüldü (2026-09-16, 8 SPY "
               "bileşeni): iki önsel arasında ağırlık farkı en fazla 8,65 puan "
               "(NVDA %15,3 → %23,9). Panelde sepette olmayan tek bir sembol bile varsa "
               "proxy'ye düşülür, çünkü eksik ismi ters-vol ile doldurmak İKİ FARKLI "
               "önseli tek vektörde karıştırmak olurdu.",
        "formul": ["factors.french_factors", "factors.factor_attribution",
                   "portfolio.propose", "portfolio.compare_methods",
                   "etf_holdings.holdings",
                   "portfolio.min_variance_weights", "portfolio.max_sharpe_weights",
                   "portfolio.hrp_weights", "portfolio.erc_weights", "portfolio.herc_weights",
                   "portfolio.black_litterman", "portfolio.cvar_lp_weights", "portfolio.cvar_risk_parity",
                   "portfolio.multi_asset_kelly", "quant_math.ledoit_wolf_cov",
                   "quant_math.pca_risk", "quant_math.denoise_corr_mp",
                   "quant_math.mp_bounds", "quant_math.mst_edges"],
    },
    {
        "id": "statarb", "title": "İstatistiksel Arbitraj (öz-portföy & kointegrasyon)",
        "endpoint": "/api/quant/statarb", "chart": "s-skor çubukları + kalıntı yolu + trace testi",
        "soru": "Bu isim, kendi risk faktörlerine göre pahalı mı ucuz mu?",
        "anlatim": "Avellaneda-Lee (2010): evrenin korelasyon matrisi öz-ayrıştırılır, en "
                   "büyük öz-vektörler PİYASA ve SEKTÖR faktörleridir. Her varlığın getirisi "
                   "bu faktörlere regres edilir; geriye kalan KALINTI o ismin kendine özgü "
                   "hareketidir. Kalıntının kümülatifi X(t) bir Ornstein-Uhlenbeck süreci "
                   "olarak modellenir ve s-skor, X'in denge dağılımındaki z-skorudur. "
                   "Johansen testi ise aynı soruyu SEVİYELERDE sorar: bu fiyat serilerinin "
                   "kaç tane durağan doğrusal bileşimi var? Rank ≥ 1 çıkarsa VECM hata "
                   "düzeltme terimini ve dengeye dönüş hızını verir.",
        "oku": "s-skor eşiği aşsa bile YARI-ÖMRE bak: dönüş penceren içinde bitmiyorsa "
               "ölçüm gürültüdür (`reliable=false` bunu söyler). Bu bir sinyal DEĞİL, "
               "örneklem-içi bir tanımdır; kenar iddiası için ileri test gerekir.",
        "formul": ["statarb.eigenportfolio_returns", "statarb.residual_ou_fit",
                   "statarb.avellaneda_lee_scores", "statarb.johansen_test",
                   "statarb.vecm_fit"],
    },
    {
        "id": "micro", "title": "Mikroyapı & Likidite",
        "endpoint": "/api/quant/micro", "chart": "imza eğrisi + VPIN + makas",
        "soru": "Bu barlarda gürültü mü var, bilgi mi? Girmek ne kadar pahalı?",
        "anlatim": "İmza grafiği, örnekleme adımı büyüdükçe bar-başına RV'nin düşüşünü "
                   "gösterir — düşüş = bid-ask sıçraması (mikroyapı gürültüsü). Sıçrama "
                   "ayrışımı sürekli oynaklığı zıplamalardan ayırır. Roll ve "
                   "Corwin-Schultz makası, Amihud likidite-dışılığı ve Kyle λ 'birim "
                   "işlem fiyatı ne kadar iter' sorusunu, VPIN tek-yönlü akış riskini "
                   "ölçer. HAR-RV yarının gerçekleşen oynaklığını tahmin eder.",
        "oku": "Makas tahmincisi negatif ham değer veriyorsa ÖLÇÜM YOKTUR (kırpılmış "
               "pozitif ortalama makas UYDURUR). Uç `reliable=false` der, sayı basmaz. "
               "U-eğrisi (Tur-77): hacim ve yüksek-düşük aralığı AYNI yarım saatlik dilimin "
               "geçmiş seanslarıyla kıyaslanır; aralık bir makas DEĞİLDİR.",
        "formul": ["microstructure.volatility_signature", "microstructure.jump_split",
                   "microstructure.two_scale_rv", "microstructure.roll_spread",
                   "microstructure.corwin_schultz_spread",
                   "microstructure.amihud_illiquidity", "microstructure.kyle_lambda",
                   "microstructure.bulk_volume_classification", "microstructure.vpin",
                   "microstructure.har_rv_fit", "microstructure.har_rv_forecast",
                   "microstructure.intraday_profile", "microstructure.intraday_anomaly"],
    },
    {
        "id": "execution", "title": "Optimal Tasfiye (Almgren-Chriss)",
        "endpoint": "/api/quant/execution", "chart": "yörünge + ödünleşim eğrisi",
        "soru": "Pozisyonu ne kadar hızlı kapatmalı?",
        "anlatim": "Hızlı çıkış piyasa ETKİSİ ödetir, yavaş çıkış fiyat RİSKİ taşıtır. "
                   "Almgren-Chriss bu ödünleşimin ortalama-varyans optimalini verir: risk "
                   "kaçınması (λ) arttıkça yörünge öne yığılır (hızlı çıkış), sıfıra "
                   "giderken doğrusal TWAP'a iner. Grafik hem yörüngeyi hem λ "
                   "taramasında etki-maliyeti ↔ tutma-riski ödünleşimini çizer.",
        "oku": "Yörünge TWAP çizgisinin ne kadar üstündeyse o kadar 'acele' ediyorsunuz; "
               "bunun bedeli etki maliyetidir.",
        "formul": ["quant_math.almgren_chriss_cost"],
    },
    {
        "id": "kelly", "title": "Kelly Boyutlama Eğrisi",
        "endpoint": "/api/quant/kelly", "chart": "log-büyüme eğrisi",
        "soru": "Sermayenin yüzde kaçını riske atmalı?",
        "anlatim": "Kelly, uzun vadeli log-servet büyümesini maksimize eden bahis "
                   "oranıdır. Eğrinin TEPE noktası tam Kelly'dir; sağına geçmek büyümeyi "
                   "DÜŞÜRÜR ve iflas riskini artırır. Projede kullanılan kesirli Kelly "
                   "(varsayılan ¼) tepenin solunda kalır: büyümenin küçük bir kısmından "
                   "vazgeçip düşüş derinliğini büyük ölçüde azaltır. Kazanma olasılığı "
                   "taraması, kenarın hangi p'de bittiğini (başabaş) gösterir.",
        "oku": "p_win başabaş noktasının (1/(1+b)) altındaysa Kelly NEGATİFtir = bahis "
               "yok. Az örneklemle tahmin edilen p, Kelly'yi sistematik ŞİŞİRİR. CPT "
               "eğrisi sıfırı kestiği λ'da bahis öznel olarak reddedilir — pozitif "
               "beklenen değere rağmen.",
        "formul": ["quant_math.kelly_binary", "quant_math.kelly_continuous",
                   "quant_math.fractional_kelly", "risk.prob_sizing.prob_to_size_mult",
                   "quant_math.cpt_value", "quant_math.prelec_weight",
                   "quant_math.cpt_prospect"],
    },
    {
        "id": "flow", "title": "Opsiyon Akışı & Piyasa Yapıcı Konumu",
        "endpoint": "/api/quant/flow", "chart": "strike başına $GEX + ızgara + acı eğrisi",
        "soru": "Piyasa yapıcı bu fiyatta ne yapmak ZORUNDA?",
        "anlatim": "Opsiyon satıcısı hedge etmek zorundadır; ne kadar hedge edeceğini "
                   "AÇIK POZİSYON (OI) ve gamma belirler. Strike başına $GEX, dayanağın "
                   "%1'lik hareketinde piyasa yapıcının almak/satmak zorunda kalacağı "
                   "dolar tutarıdır. Net GEX pozitifken hedge akışı hareketi SÖNÜMLER "
                   "(bant içi), negatifken BÜYÜTÜR (trend/çöküş). Gamma flip ikisinin "
                   "sınırıdır ve burada kestirme kümülatif toplamla değil, net GEX spot "
                   "ızgarasında YENİDEN fiyatlanarak (gamma spot'a bağlıdır) bulunur. "
                   "Yanına üç ölçü daha gelir: put/call oranı, max pain (vade günü "
                   "yazar-ödemesini en aza indiren spot) ve hacim>OI olan kontratlar. "
                   "Bar tarafında Alpaca'nın trade_count/vwap kolonları göreceli hacmi "
                   "(RVOL, aynı gün-içi dilimine göre), ortalama işlem büyüklüğünü "
                   "(kalabalık mı blok mu) ve VWAP sapmasını verir.",
        "oku": "Net GEX NEGATİF olduğunda, varsayılan dealer konumuyla hedge akışı hareketi büyütür "
               "— aynı haber daha büyük bir çubuk üretir. Bu bir yön iddiası DEĞİLDİR: "
               "dealer'ın call'da uzun/put'ta kısa olduğu VARSAYIMDIR (işlem tarafı "
               "verisi yok), yanlışsa net GEX'in işareti ters döner.",
        "formul": ["option_flow.dealer_gamma_exposure", "option_flow.put_call_ratio",
                   "option_flow.max_pain", "option_flow.unusual_activity",
                   "option_flow.dealer_vanna_charm", "option_flow.dealer_hedging_clock", "option_flow.bs_gamma_vec",
                   "microstructure.relative_volume", "microstructure.trade_size_profile",
                   "microstructure.vwap_deviation", "microstructure.turnover_rate"],
    },
    {
        "id": "curve", "title": "ABD Hazine Getiri Eğrisi (NSS)",
        "endpoint": "/api/quant/curve", "chart": "eğri + NSS uyumu + faktör serisi + farklar",
        "soru": "Tahvil piyasası büyüme ve enflasyon hakkında ne fiyatlıyor?",
        "anlatim": "11 sabit vade (1 ay → 30 yıl) FRED'den ANAHTARSIZ çekilir ve üstüne "
                   "Nelson-Siegel-Svensson uydurulur. NSS eğriyi ÜÇ okunabilir faktöre "
                   "indirger: SEVİYE (β₀, uzun vade asimptotu), EĞİM (−β₁, kısa↔uzun farkı) "
                   "ve EĞRİLİK (β₂, ortanın kamburu). Yanına FRED'in kendi yayımladığı iki "
                   "fark (10Y−2Y, 10Y−3M) ve TIPS başabaş enflasyonu (5Y/10Y) gelir — "
                   "farkları elle çıkarmak yerine kaynağından okumak, iki serinin farklı "
                   "günlerde güncellenmesinden doğan hayalet inversiyonları önler.",
        "oku": "NSS aşırı parametrelenmiştir: salt en-küçük-kareler eğriyi mükemmel uydurup "
               "β₀'ı anlamsız bir yere koyabilir. ÖLÇÜLDÜ (son 500 işlem günü): kısıtsız "
               "uyum günlerin %98,0'ında β₀'ı gözlenen getiri aralığının, %81,0'ında "
               "toleranslı bandın bile DIŞINA atıyordu "
               "(2026-09-03'te β₀ = −2,08 — 30Y %5,25 iken). Bu yüzden panodaki uyum, β₀'ı "
               "[min(y), max(y)] içinde tutan en iyi adaydır; kısıtsız uyumun ne diyeceği "
               "`beta0_kisitsiz` ile AYRICA gösterilir. Ters eğri (10Y−2Y < 0) tarihsel bir "
               "resesyon ÖNCÜSÜ olarak izlenir — bu modül bir tahmin ÜRETMEZ, fiyatı okur.",
        "formul": ["quant_math.nss_fit", "fred_data.treasury_curve",
                   "fred_data.curve_spreads"],
    },
    {
        "id": "macro", "title": "Makro Rejim Paneli (13 gösterge)",
        "endpoint": "/api/quant/macro", "chart": "gösterge kartları + seçili serinin yolu",
        "soru": "Likidite, para politikası ve kredi koşulları bugün nerede duruyor?",
        "anlatim": "20 FRED serisinden 13 gösterge kurulur: Fed NET LİKİDİTESİ "
                   "(WALCL − TGA − ters repo), TAYLOR kuralı sapması, ACM vade primi, "
                   "NFCI finansal koşullar, WTI şoku, SAHM resesyon kuralı, DXY momentumu, "
                   "TIPS başabaş enflasyonu, HY−IG kredi makası, 2s10s eğim, CP−T-Bill ve "
                   "SOFR−EFFR basisi. Üç ölçüm kuralı panelin İÇİNDEDİR: (1) her seri "
                   "kendi YAYIN GECİKMESİ kadar ileri kaydırılır — FRED'in gözlem tarihi "
                   "BİLİNME tarihi değildir ve onu bilinme tarihi saymak modele geleceği "
                   "okutur; (2) ileri doldurma sınırı SATIR değil TAKVİM GÜNÜdür (seyrek "
                   "ızgarada satır sınırı hiç ısırmaz ve panel 'her gün taze makro veri "
                   "var' yalanını üretir); (3) birimler çevrilir — RRPONTSYD FRED'de "
                   "MİLYAR, WALCL MİLYON dolardır ve çevirmeyen naif çıkarma net "
                   "likiditeyi 2022 sonunda 2,55 TRİLYON dolar fazla gösteriyordu.",
        "oku": "Bu panel bir KURAL İÇERMEZ. Listenin önerdiği eşikler ('net likidite "
               "düşerken long tavanı kısılsın', 'NFCI > 0 ise nakde geç') KODA YAZILMADI: "
               "bir eşik bir kenar iddiasıdır ve `edge_lab` kanıt kapısından geçmeden "
               "sinyal/risk katmanına giremez (ACI DERS #3/#5). Kırpık seri uyarısı ciddiye "
               "alınmalıdır — HY/IG OAS serileri FRED grafik ucunda ~3 yılla kırpıktır, "
               "yani 2023 öncesi için kredi makası ÖLÇÜLMEMİŞTİR, 'sıfır' değildir.",
        "formul": ["macro_regime.net_liquidity", "macro_regime.taylor_rule_gap",
                   "macro_regime.term_premium", "macro_regime.financial_conditions",
                   "macro_regime.sahm_gap", "macro_regime.breakeven_inflation",
                   "macro_regime.credit_spread", "macro_regime.curve_slope",
                   "macro_regime.dollar_momentum", "macro_regime.usdjpy_shock",
                   "macro_regime.oil_shock", "macro_regime.cp_tbill_spread",
                   "macro_regime.sofr_effr_basis", "macro_regime.panel"],
    },
    {
        "id": "fundamentals", "title": "Şirket Temelleri (SEC EDGAR XBRL)",
        "endpoint": "/api/quant/fundamentals", "chart": "skor kartları + kalem serileri",
        "soru": "Bu şirketin bilançosu ne söylüyor — ve rakam ne zaman BİLİNİR oldu?",
        "anlatim": "SEC'in `companyfacts` ucundan (anahtarsız) 10-K kalemleri çekilir ve "
                   "sekiz klasik skor kurulur: PIOTROSKI F (9 ikili sinyal), ALTMAN Z "
                   "(iflas mesafesi), BENEISH M (kazanç manipülasyonu), MOHANRAM G, "
                   "SLOAN tahakkuku, DuPont ayrıştırması, nakit dönüşüm süresi ve "
                   "hissedar getirisi. İki ölçüm kuralı taşınır: (1) bir rakam dönem "
                   "sonunda OLUŞUR ama DOSYALANDIĞINDA bilinir — her okuma `filed <= "
                   "as_of` ile sınırlıdır, yani panel geçmişe bakarken de o gün gerçekten "
                   "bilinen rakamı kullanır; (2) her kalem bir ETİKET ZİNCİRİDİR — tek "
                   "us-gaap etiketi okuyan bir okuyucu AAPL için 2018'de biten `Revenues` "
                   "serisini 2026 kârıyla kıyaslar ve HATA VERMEZ.",
        "oku": "Skorların YANINDAKİ `n_sinyal` okunmadan skor okunamaz: 9 yerine 6 "
               "sinyalden kurulan bir F-Score 'zayıf şirket' demek DEĞİLDİR. Altman'ın "
               "`varyant` alanı da zorunludur — klasik Z (piyasa değerli) ile Z' (defter "
               "değerli) EŞİKLERİ farklıdır. Beneish M eksik endeksle HİÇ hesaplanmaz: "
               "−1,78 eşiği tam sekiz endeksle kalibre edilmiştir. Bu panel de bir kural "
               "içermez; 'F ≥ 7 ise al' bir kenar iddiasıdır ve kanıt kapısı ister.",
        "formul": ["edgar_facts.facts", "edgar_facts.annual_frame",
                   "edgar_facts.son_iki_yil", "edgar_facts.filings",
                   "edgar_facts.item_filings", "fundamentals.piotroski_f_score",
                   "fundamentals.altman_z_score", "fundamentals.beneish_m_score",
                   "fundamentals.mohanram_g_score", "fundamentals.accruals_ratio",
                   "fundamentals.dupont", "fundamentals.cash_conversion_cycle",
                   "fundamentals.shareholder_yield", "fundamentals.net_cash_runway",
                   "fundamentals.goodwill_risk", "fundamentals.pension_gap",
                   "fundamentals.rd_intensity", "fundamentals.lease_burden",
                   "fundamentals.fcf_vs_dividend", "fundamentals.earnings_yield",
                   "fundamentals.rapor"],
    },
    {
        "id": "drift", "title": "Model Sürüklenmesi (SR 11-7)",
        "endpoint": "/api/quant/drift", "chart": "PSI çubukları + haftalık PSI zaman serisi",
        "soru": "Modelin BUGÜN gördüğü dünya, öğrendiği dünyayla aynı mı?",
        "anlatim": "Canlı özellik logu iki pencereye bölünür: REFERANS (eski) ve CARİ "
                   "(son). Her özellik için dört ölçü alınır — PSI (kova oranlarının "
                   "simetrik KL benzeri toplamı), iki-örneklem testi (sayısalda KS, "
                   "≤20 kategoride χ²), Wasserstein-1 (dağılımı taşımanın maliyeti, "
                   "referans std'sine oranlanır) ve normal ailesinde Fisher–Rao "
                   "jeodezik mesafesi. Referans penceresi cari pencereden ÖNCE BİTER; "
                   "çakışan pencereler ölçümü kendi kendisiyle kıyaslardı. p değerleri "
                   "Bonferroni ile düzeltilir — 30 özelliği ayrı ayrı sınamak %5 "
                   "düzeyinde ortalama 1,5 sahte alarm üretir.",
        "oku": "n büyüdükçe önemsiz kaymalar da 'anlamlı' çıkar: ÖNCE ETKİ BÜYÜKLÜĞÜNÜ "
               "(PSI, w1_std, fr) oku, p'yi sonra. PSI'nin 0,10 / 0,25 eşikleri kredi "
               "skorlama geleneğidir, DÜZENLEYİCİ ZORUNLULUK DEĞİLDİR. Testler bağımsız "
               "gözlem varsayar; bar verisi otokorelasyonludur → p'ler iyimserdir. "
               "Sürüklenme bir KENAR İDDİASI da değildir: 'PSI yüksek → modeli değiştir' "
               "kararı bu panelden ÇIKMAZ.",
        "formul": ["drift_monitor.psi", "drift_monitor.ks_drift",
                   "drift_monitor.wasserstein_drift", "drift_monitor.fisher_rao_normal",
                   "drift_monitor.fisher_rao_drift", "drift_monitor.feature_drift_report",
                   "drift_monitor.load_live_features", "drift_monitor.time_sliced_psi"],
    },
    {
        "id": "econometrics", "title": "DCC · FIGARCH · Dalgacık",
        "endpoint": "/api/quant/econometrics",
        "chart": "dinamik korelasyon yolu + ölçek-varyans çubukları + MODWT detayları",
        "soru": "Korelasyon ve oynaklık ZAMANLA nasıl değişiyor, hangi ÖLÇEKTE yaşıyor?",
        "anlatim": "Üç ayrı soru tek panelde: (1) DCC-GARCH (Engle 2002) her varlığa "
                   "ayrı GARCH takar, standartlaşmış artıklardan ZAMANLA DEĞİŞEN "
                   "korelasyon matrisi kurar (H=D·R·D) — sabit korelasyon (CCC) "
                   "varsayımı log-olabilirlikle kıyaslanır. (2) FIGARCH (Baillie ve ark. "
                   "1996) oynaklığın hafızasını KESİRLİ d ile ölçer: d=0 sıradan "
                   "GARCH'tır, d→1 entegre (IGARCH) demektir; aradaki değer şokun ne "
                   "üstel ne kalıcı, HİPERBOLİK söndüğünü söyler. (3) Haar MODWT "
                   "getiriyi 2, 4, 8, 16… barlık ölçeklere ayırır; hangi ölçeğin "
                   "varyansı taşıdığı log-log eğimden okunur.",
        "oku": "DCC'nin a+b'si 1'e yakınsa korelasyon şoku KALICIDIR — çeşitlendirme "
               "tam ihtiyaç anında kaybolur. MODWT DÖNGÜSELDİR (sıfır faz): serinin "
               "SONU BAŞINA sarılır, yani bir detay katsayısı GELECEĞİ görebilir — "
               "`causal: false` bu yüzden yazılıdır ve bu blok canlı özellik OLARAK "
               "KULLANILAMAZ. Üç uyumun tamamı ÖRNEKLEM İÇİDİR; 'FIGARCH GARCH'ı yendi' "
               "demek için yürüyen OOS kıyası gerekir.",
        "formul": ["econometrics.dcc_garch_fit", "econometrics.dcc_correlation_path",
                   "econometrics.dcc_forecast", "econometrics.figarch_fit",
                   "econometrics.modwt_haar", "econometrics.modwt_mra",
                   "econometrics.wavelet_variance", "quant_math.garch11_fit"],
    },
    {
        "id": "tda", "title": "Topolojik Veri Analizi (Kalıcı Homoloji)",
        "endpoint": "/api/quant/tda",
        "chart": "kalıcılık diyagramı + manzara eğrileri + kayan pencere norm serisi",
        "soru": "Getiri bulutunun ŞEKLİ ne — kaç parça, kaç delik, ne kadar kalıcı?",
        "anlatim": "Her gün bir nokta (N sembolün o günkü getirisi) olarak N-boyutlu "
                   "uzaya konur. Yarıçap ε büyütülürken noktalar önce kenarlarla, sonra "
                   "üçgenlerle bağlanır (Vietoris–Rips). H0 çubukları BİLEŞENLERİN "
                   "birleşmesini, H1 çubukları kapanan DÖNGÜLERİ (delikleri) izler; bir "
                   "çubuğun ÖMRÜ (ölüm−doğum) o yapının gürültüye karşı DAYANIKLILIĞIDIR. "
                   "Kalıcılık manzarası (Bubenik 2015) diyagramı çadır fonksiyonlarına "
                   "çevirir ve L1/L2 normuyla TEK SAYIYA indirger — böylece şekil bir "
                   "zaman serisi olarak izlenebilir. Kayan pencere NEDENSELDİR: her "
                   "nokta yalnız pencere sonuna kadarki getirilerden kurulur.",
        "oku": "Uzun H1 çubuğu = korelasyon yapısında KAPANMAYAN bir döngü; kısa "
               "çubukların hepsi gürültüdür. Gidea–Katz (2018) çöküş öncesinde L1/L2'nin "
               "yükseldiğini bildirir — BU SİSTEMDE ÖLÇÜLMEMİŞTİR, panel iddiayı değil "
               "SERİYİ gösterir. Rips hesabı nokta sayısında kübiktir: pencere 200 nokta "
               "ile sınırlıdır ve `kesildi` alanı `max_edge` yüzünden atılan çubukları "
               "sayar — sıfır değilse diyagram EKSİKTİR.",
        "formul": ["tda.rips_persistence", "tda.persistence_landscape",
                   "tda.landscape_norms", "tda.sliding_window_tda"],
    },
    {
        "id": "dispersion", "title": "Dispersiyon & Örtük Korelasyon",
        "endpoint": "/api/quant/dispersion",
        "chart": "örtük korelasyon serisi + Cboe COR1M kıyası + ileri vol eğrisi",
        "soru": "Endeks opsiyonu, bileşenlerinin toplamına göre PAHALI mı?",
        "anlatim": "Dispersiyon ticaretinin fiyatladığı büyüklük ÖRTÜK KORELASYONDUR: "
                   "endeksin varyansı, bileşen varyanslarının ağırlıklı toplamı ve "
                   "ikili korelasyonlarla bağlıdır → ρ_imp = (σ_endeks² − Σwᵢ²σᵢ²) / "
                   "(ΣΣ_{i≠j} wᵢwⱼσᵢσⱼ). Bileşen örtük vol'leri DoltHub günlük "
                   "arşivinden, ağırlıklar SPDR'ın günlük sepet dosyasından gelir; "
                   "sonuç Cboe'nin resmî COR1M endeksiyle KIYASLANIR (aynı kimlik, aynı "
                   "evren: ilk 50 bileşen). İkinci blok VIX ailesinden İLERİ VOL "
                   "eğrisini çıkarır: f² = (T₂σ₂² − T₁σ₁²)/(T₂ − T₁).",
        "oku": "COR1M kıyasındaki iki sayıyı AYRI oku: SEVİYE korelasyonu ağırlıkların "
               "doğruluğunu, DEĞİŞİM korelasyonu serinin günlük sinyalini ölçer. Sepet "
               "ağırlıkları BUGÜNÜNDÜR (`agirlik_pit_acigi: true`) — geçmiş günlerin ρ'su "
               "o günün sepetiyle DEĞİL bugünkü sepetle kurulmuştur, bu yüzden erken "
               "yıllarda SEVİYE aşağı yanlıdır. İleri varyansın NEGATİF çıkması bir "
               "arbitraj sinyali değil, çoğunlukla iki ucun aynı gün yayımlanmamış "
               "olmasının izidir; `n_negatif_gun` bunu sayar.",
        "formul": ["etf_holdings.holdings", "etf_holdings.parse_xlsx",
                   "etf_holdings.implied_correlation_frame",
                   "options_math.implied_correlation",
                   "options_math.implied_correlation_series",
                   "options_math.forward_variance_curve", "options_math.variance_swap_strike",
                   "cboe_data.forward_variance", "cboe_data.get_index"],
    },
]


def _sepet_agirliklari(etf: str) -> dict[str, float] | None:
    """SPDR sepetinden {sembol: ağırlık} — ÖNBELLEK-YALNIZ (ağa çıkmaz, BULGU-59-8).

    Önbellek yoksa `None` (ölçüm yokluğu); doldurmak ayrı adımdır
    (`python -m nihai.data.etf_holdings --warm SPY`). Aynı sembol iki satırda gelirse
    (sınıf hisseleri) ağırlıklar TOPLANIR."""
    from nihai.data import etf_holdings as eh

    d = eh.holdings(etf.upper(), fetch=False)
    if d is None or getattr(d, "empty", True) or "weight" not in d.columns:
        return None
    seri = d.groupby(d["ticker"].astype(str).str.upper())["weight"].sum()
    return {str(k): float(v) for k, v in seri.items() if np.isfinite(v) and v > 0}


def _symbol_panel(symbols: str, interval: str, lookback: int,
                  min_rows: int = 30) -> tuple[dict | None, dict | None]:
    """Çok-sembollü ORTAK TARİHLİ kapanış paneli (portföy ve statarb aynı paneli ister).

    Semboller boşsa izleme listesinden okunur (en fazla 15). Ortak tarihe göre `dropna`
    edilir — hizalanmamış tarihleri ileri-doldurmak look-ahead üretirdi. Dönüş
    `(veri, hata)` çiftidir; hata `_fail` sözlüğüdür (uç FAIL-SOFT kalır)."""
    import pandas as pd

    syms = [x.upper().strip() for x in symbols.split(",") if x.strip()]
    if not syms:
        try:
            import nihai.execution.watchlist as wl
            syms = list(wl.current())[:10]
        except Exception:  # noqa: BLE001
            syms = []
    syms = syms[:15]
    if len(syms) < 2:
        return None, _fail("En az 2 sembol gerekir (izleme listesi boş olabilir).")
    closes, missing = {}, []
    for sm in syms:
        d = _ohlc(sm, interval, lookback)
        if d is None:
            missing.append(sm)
            continue
        closes[sm] = pd.Series(d["close"], index=pd.Index(d["dates"]))
    if len(closes) < 2:
        return None, _fail(f"Yeterli fiyat verisi yok (eksik: {', '.join(missing) or '—'}).")
    panel = pd.DataFrame(closes).dropna()
    if len(panel) < min_rows:
        return None, _fail(f"Ortak tarih sayısı çok az ({len(panel)}); "
                           f"en az {min_rows} gerekir.")
    px = panel.to_numpy(dtype=float)
    return {"panel": panel, "names": list(panel.columns), "px": px,
            "rets": np.diff(np.log(px), axis=0), "missing": missing,
            "dates": [str(x) for x in panel.index.tolist()],
            "ppy": _ppy(interval)}, None


def _turnover_cells(symbol: str, volume) -> dict:
    """Dönüşüm hızı hücreleri (BULGU-59-8). Float ölçülemezse `turnover=None` + neden.

    Float ANLIKTIR (point-in-time değil) → yalnız SON barın dönüşüm hızı raporlanır;
    tarihsel bir seri kurmak bugünkü float'ı geçmişe uygulamak olurdu (look-ahead)."""
    from nihai.ai import microstructure as ms
    from nihai.data.market_structure import float_shares
    fs = float_shares(symbol)          # ÖNBELLEK-OKUMA (fetch=False; ağ çağrısı YOK)
    if fs is None:
        return {"turnover": None,
                "turnover_neden": "float önbellekte yok — "
                                  "`python -m nihai.data.market_structure --warm <SEM>`"}
    tr = _guard(ms.turnover_rate, float(volume[-1]), fs["shares"])
    if tr is None:
        return {"turnover": None, "turnover_neden": "hacim/float geçersiz"}
    return {"turnover": {"rate": _f(tr, 8), "float_shares": _f(fs["shares"], 0),
                         "source": fs["source"], "asof": fs["asof"],
                         "last_volume": _f(float(volume[-1]), 0)},
            "turnover_neden": None}


@router.get("/api/quant/catalog")
def quant_catalog() -> dict:
    """Model dizini — pano kart başlıklarını/açıklamalarını TEK kaynaktan okur."""
    return {"ok": True, "models": _CATALOG, "count": len(_CATALOG)}


# ===========================================================================
# 1) RİSK — getiri dağılımı, VaR/CVaR/EVT, düşüş, oranlar, hafıza teşhisi
# ===========================================================================
@router.get("/api/quant/risk")
def quant_risk(symbol: str = "SPY", interval: str = "1d", lookback: int = 500,
               alpha: float = 0.05) -> dict:
    """Getiri dağılımı + üç VaR ailesi + düşüş eğrisi + performans oranları + hafıza."""
    from scipy import stats as _st

    from nihai.ai import quant_math as qm

    data = _ohlc(symbol, interval, lookback)
    if data is None:
        return _fail(f"{symbol} için bar verisi alınamadı.", symbol=symbol)
    close = data["close"]
    if close.size < 60:
        return _fail(f"Yetersiz bar ({close.size}); en az 60 gerekir.", symbol=symbol)

    rets = qm.log_returns(close)
    a = float(alpha) if 0.0 < float(alpha) < 0.5 else 0.05
    ppy = _ppy(interval)

    evt = _guard(qm.evt_gpd_var_es, rets, 0.01, 0.95)
    tail = {
        "alpha": a,
        "hist_var": _f(_guard(qm.var_historical, rets, a)),
        "hist_cvar": _f(_guard(qm.cvar_historical, rets, a)),
        "cf_var": _f(_guard(qm.var_cornish_fisher, rets, a)),
        "hist_var_99": _f(_guard(qm.var_historical, rets, 0.01)),
        "hist_cvar_99": _f(_guard(qm.cvar_historical, rets, 0.01)),
        "cf_var_99": _f(_guard(qm.var_cornish_fisher, rets, 0.01)),
        "evt": None if evt is None else {
            "var_99": _f(evt["var"]), "es_99": _f(evt["es"]),
            "threshold": _f(evt["threshold"]), "shape": _f(evt["shape"], 4),
            "scale": _f(evt["scale"]), "n_exceed": int(evt["n_exceed"]),
            "fat_tail": bool(evt["shape"] > 0.0),
        },
    }

    # Düşüş (su-altı) eğrisi: max_drawdown SAYIyı verir, bu seri ŞEKLİ verir.
    equity = close / float(close[0])
    peak = np.maximum.accumulate(equity)
    underwater = (equity - peak) / peak
    mdd = _guard(qm.max_drawdown, close)
    cc = _guard(qm.vol_close_to_close, close)

    ratios = {
        "sharpe": _f(_guard(qm.sharpe, rets, 0.0, ppy), 3),
        "sortino": _f(_guard(qm.sortino, rets, 0.0, ppy), 3),
        "calmar": _f(_guard(qm.calmar, rets, ppy), 3),
        "omega": _f(_guard(qm.omega, rets, 0.0), 3),
        "profit_factor": _f(_guard(qm.profit_factor, rets), 3),
        "ulcer_index": _f(_guard(qm.ulcer_index, equity), 4),
        "downside_dev": _f(_guard(qm.downside_deviation, rets)),
        "max_drawdown": _f(mdd[0], 4) if mdd else None,
        "max_dd_bars": int(mdd[1]) if mdd else None,
        "ann_vol": _f(_guard(qm.annualize_vol, cc, ppy), 4) if cc is not None else None,
        "ann_return": _f(_guard(qm.annualize_return, float(np.mean(rets)), ppy), 4),
        # Tur-73 (Madde 249/250): mevcut oranların yanına ASİMETRİ ve SÜRE ölçüleri.
        # ulcer_index düşüşün DERİNLİĞİ+süresini birlikte cezalandırır ama "ne kadar
        # sürdü"yü SAYI olarak vermez; time_underwater tam onu verir.
        "gain_to_pain": _f(_guard(qm.gain_to_pain, rets), 3),
        "tail_ratio": _f(_guard(qm.tail_ratio, rets, 0.95), 3),
    }
    tuw = _guard(qm.time_underwater, equity)
    ratios["tuw_max_bar"] = int(tuw["max_bar"]) if tuw else None
    ratios["tuw_mevcut_bar"] = int(tuw["mevcut_bar"]) if tuw else None
    ratios["tuw_orani"] = _f(tuw["sualti_orani"], 4) if tuw else None

    # Hafıza/öngörülebilirlik teşhisi: trend mi, ortalamaya dönüş mü, saf gürültü mü?
    memory = {
        "hurst": _f(_guard(qm.hurst_exponent, close), 3),
        "ou_half_life": _f(_guard(qm.ou_half_life, close), 2),
        "vr_2": _f(_guard(qm.variance_ratio, close, 2), 3),
        "vr_5": _f(_guard(qm.variance_ratio, close, 5), 3),
        "vr_10": _f(_guard(qm.variance_ratio, close, 10), 3),
        "perm_entropy": _f(_guard(qm.permutation_entropy, rets, 3, 1), 3),
        "sample_entropy": _f(_guard(qm.sample_entropy, rets, 2, 0.2), 3),
        # Tur-73 (Madde 203): aynı seriye ÜÇ entropi. Shannon düzensizliğin tamamını,
        # Rényi(α=2) yoğun bölgeyi, Tsallis(q=2) ise kuyruğu ağırlıklandırır. Üçü birlikte
        # okunur: yalnız Shannon'a bakmak dağılımın NERESİNİN çöktüğünü gizler.
        "shannon_entropy": _f(_guard(qm.shannon_entropy, rets, 20), 4),
        "renyi_entropy_2": _f(_guard(qm.renyi_entropy, rets, 2.0, 20), 4),
        "tsallis_entropy_2": _f(_guard(qm.tsallis_entropy, rets, 2.0, 20), 4),
    }
    adf = _guard(qm.adf_test, close)
    if adf:
        memory["adf_stat"], memory["adf_pvalue"] = _f(adf[0], 3), _f(adf[1], 4)

    return {
        "ok": True, "symbol": symbol.upper().strip(), "interval": interval,
        "n_bars": int(close.size), "ppy": ppy, "dates": data["dates"],
        "hist": _histogram(rets, bins=45),
        "tail": tail, "ratios": ratios, "memory": memory,
        "underwater": _arr(underwater, 5), "equity": _arr(equity, 5),
        "returns_stats": {
            "mean": _f(float(np.mean(rets))), "std": _f(float(np.std(rets, ddof=1))),
            "skew": _f(float(_st.skew(rets)), 3),
            "kurtosis": _f(float(_st.kurtosis(rets)), 3),
            "min": _f(float(np.min(rets))), "max": _f(float(np.max(rets))),
        },
    }


# ===========================================================================
# 2) VOLATİLİTE — tahminci karşılaştırması, koni, GARCH(1,1), dalgacık
# ===========================================================================
@router.get("/api/quant/vol")
def quant_vol(symbol: str = "SPY", interval: str = "1d", lookback: int = 500,
              horizon: int = 20, cone_estimator: str = "yang_zhang") -> dict:
    """Beş vol tahmincisi + vol konisi + GARCH koşullu yol/öngörü + çok-ölçekli ayrışım."""
    from nihai.ai import quant_math as qm

    data = _ohlc(symbol, interval, lookback)
    if data is None:
        return _fail(f"{symbol} için bar verisi alınamadı.", symbol=symbol)
    close, o, h, low = data["close"], data["open"], data["high"], data["low"]
    if close.size < 60:
        return _fail(f"Yetersiz bar ({close.size}).", symbol=symbol)
    ppy = _ppy(interval)
    rets = qm.log_returns(close)
    hz = max(1, min(int(horizon), 120))

    def _ann(v):
        return None if v is None else _f(qm.annualize_vol(v, ppy), 4)

    has_ohlc = all(x is not None for x in (o, h, low))
    est = {
        "close_to_close": _ann(_guard(qm.vol_close_to_close, close)),
        "parkinson": _ann(_guard(qm.vol_parkinson, h, low)) if has_ohlc else None,
        "garman_klass": _ann(_guard(qm.vol_garman_klass, o, h, low, close)) if has_ohlc else None,
        "yang_zhang": _ann(_guard(qm.vol_yang_zhang, o, h, low, close)) if has_ohlc else None,
        "ewma_094": _ann(_guard(qm.ewma_vol, rets, 0.94)),
    }

    cone = _cone_report(data, symbol, ppy, cone_estimator)

    # Rolling gerçekleşen vol serisi — GARCH koşullu yolunun karşılaştırma tabanı.
    w = min(21, max(5, rets.size // 10))
    realized = np.full(rets.size, np.nan)
    for i in range(w, rets.size + 1):
        realized[i - 1] = float(np.std(rets[i - w:i], ddof=1))

    # GARCH(1,1). `garch11_fit` sigma YOLUNU döndürmez (yalnız son değeri); yolu aynı
    # modülün kendi yineleme fonksiyonuyla YENİDEN KURARIZ — formül burada tekrarlanmaz.
    garch: dict | None = None
    fit = _guard(qm.garch11_fit, rets)
    if fit:
        resid = rets - float(fit["mean"])
        var0 = float(np.var(resid))
        sig2 = _guard(qm._garch11_sigma2_path, float(fit["omega"]), float(fit["alpha"]),
                      float(fit["beta"]), resid * resid, var0)
        fc: list[float | None] = []
        for k in range(1, hz + 1):
            v = _guard(qm.garch11_forecast, fit, k)
            fc.append(_f(qm.annualize_vol(v, ppy), 4) if v is not None else None)
        pers = float(fit["persistence"])
        garch = {
            "omega": _f(fit["omega"], 12), "alpha": _f(fit["alpha"], 4),
            "beta": _f(fit["beta"], 4), "persistence": _f(pers, 4),
            "loglik": _f(fit["loglik"], 2), "n": int(fit["n"]),
            "uncond_vol": (None if fit["uncond_var"] is None else
                           _f(qm.annualize_vol(math.sqrt(fit["uncond_var"]), ppy), 4)),
            "igarch_boundary": fit["uncond_var"] is None,
            "cond_vol": ([] if sig2 is None else
                         [_f(qm.annualize_vol(math.sqrt(max(float(v), 0.0)), ppy), 4)
                          for v in np.asarray(sig2).ravel()]),
            "forecast": fc,
            "half_life_bars": (_f(math.log(0.5) / math.log(pers), 1)
                               if 0.0 < pers < 1.0 else None),
        }

    # ASİMETRİK GARCH (Tur-72, yol haritası §6.1). GARCH(1,1) şoku KARESİYLE ölçer:
    # +%3 ile −%3 aynı vol tepkisini üretir. Hisse getirilerinde bu ampirik olarak
    # yanlıştır. GJR eşikle (1[eps<0]), EGARCH sürekli işaretle (z) asimetriyi ölçer.
    # HABER-ETKİSİ EĞRİSİ (Engle-Ng 1993) asimetrinin GÖRSEL kanıtıdır: σ²_{t-1}
    # koşulsuz düzeyde sabitlenip σ²_t şoka karşı çizilir; GARCH parabolü SİMETRİK,
    # GJR/EGARCH eğrileri sola YATIKtır.
    asym: dict | None = None
    gjr = _guard(qm.gjr_garch_fit, rets)
    egf = _guard(qm.egarch_fit, rets)
    if gjr or egf:
        eps_grid = np.linspace(-4.0, 4.0, 81) * float(np.std(rets))
        asym = {"eps_grid": _arr(eps_grid, 6), "garch_loglik": (None if not fit else
                                                               _f(fit["loglik"], 2))}

        def _annv(v2):
            return _f(qm.annualize_vol(math.sqrt(max(float(v2), 0.0)), ppy), 4)

        if fit and fit.get("uncond_var"):
            v_bar = float(fit["uncond_var"])
            asym["news_garch"] = [_annv(fit["omega"] + fit["alpha"] * e * e
                                        + fit["beta"] * v_bar) for e in eps_grid]
        else:
            asym["news_garch"] = None
        if gjr:
            resid_g = rets - float(gjr["mean"])
            s2g = _guard(qm._gjr_sigma2_path, float(gjr["omega"]), float(gjr["alpha"]),
                         float(gjr["gamma"]), float(gjr["beta"]), resid_g,
                         float(np.var(resid_g)))
            v_bar = gjr["uncond_var"] or gjr["last_sigma2"]
            asym["gjr"] = {
                "omega": _f(gjr["omega"], 12), "alpha": _f(gjr["alpha"], 4),
                "gamma": _f(gjr["gamma"], 4), "beta": _f(gjr["beta"], 4),
                "persistence": _f(gjr["persistence"], 4), "loglik": _f(gjr["loglik"], 2),
                "leverage": bool(gjr["leverage"]),
                "lr_vs_garch": (None if not fit else
                                _f(2.0 * (gjr["loglik"] - fit["loglik"]), 3)),
                "uncond_vol": (None if gjr["uncond_var"] is None else
                               _annv(gjr["uncond_var"])),
                "cond_vol": ([] if s2g is None else
                             [_annv(v) for v in np.asarray(s2g).ravel()]),
                "forecast": [_f(qm.annualize_vol(v, ppy), 4) if v is not None else None
                             for v in (_guard(qm.asym_garch_forecast, gjr, k)
                                       for k in range(1, hz + 1))],
                "news": [_annv(gjr["omega"]
                               + (gjr["alpha"] + gjr["gamma"] * (1.0 if e < 0 else 0.0))
                               * e * e + gjr["beta"] * float(v_bar)) for e in eps_grid],
            }
        if egf:
            resid_e = rets - float(egf["mean"])
            var0e = float(np.var(resid_e))
            ls2 = _guard(qm._egarch_logs2_path, float(egf["omega"]), float(egf["alpha"]),
                         float(egf["gamma"]), float(egf["beta"]), resid_e,
                         math.log(var0e) if var0e > 0 else 0.0)
            lv_bar = float(egf["uncond_log_var"])
            sig_bar = math.exp(0.5 * lv_bar)
            asym["egarch"] = {
                "omega": _f(egf["omega"], 6), "alpha": _f(egf["alpha"], 4),
                "gamma": _f(egf["gamma"], 4), "beta": _f(egf["beta"], 4),
                "persistence": _f(egf["persistence"], 4), "loglik": _f(egf["loglik"], 2),
                "leverage": bool(egf["leverage"]),
                "lr_vs_garch": (None if not fit else
                                _f(2.0 * (egf["loglik"] - fit["loglik"]), 3)),
                "uncond_vol_median": (None if egf["uncond_var_median"] is None else
                                      _annv(egf["uncond_var_median"])),
                "cond_vol": ([] if ls2 is None else
                             [_annv(math.exp(min(float(v), 50.0)))
                              for v in np.asarray(ls2).ravel()]),
                "forecast": [_f(qm.annualize_vol(v, ppy), 4) if v is not None else None
                             for v in (_guard(qm.asym_garch_forecast, egf, k)
                                       for k in range(1, hz + 1))],
                "news": [_annv(math.exp(min(
                    egf["omega"] + egf["alpha"] * (abs(e / sig_bar) - math.sqrt(2 / math.pi))
                    + egf["gamma"] * (e / sig_bar) + egf["beta"] * lv_bar, 50.0)))
                    for e in eps_grid] if sig_bar > 0 else None,
            }

    wav = _guard(qm.haar_dwt_vol, rets, 3)
    wavelet = None
    if wav:
        ratio = wav.get("energy_ratio") or {}
        wavelet = {
            "labels": list(ratio.keys()) + ["yaklaşıklık"],
            "share": [_f(v, 4) for v in ratio.values()] +
                     [_f(max(0.0, 1.0 - sum(float(v) for v in ratio.values())), 4)],
            "levels": int(wav.get("levels") or 0), "n_used": int(wav.get("n_used") or 0),
        }

    return {
        "ok": True, "symbol": symbol.upper().strip(), "interval": interval,
        "n_bars": int(close.size), "ppy": ppy, "horizon": hz,
        "dates": data["dates"][1:],       # getiri serisiyle hizalı (ilk bar düşer)
        "estimators": est, "cone": cone,
        "realized_vol": [_f(qm.annualize_vol(v, ppy), 4) if math.isfinite(v) else None
                         for v in realized],
        "garch": garch, "asym": asym, "wavelet": wavelet,
    }


def _current_atm_iv(symbol: str) -> tuple[float | None, str | None]:
    """En yakın canlı vadenin ATM call/put IV ortalaması; veri yoksa None.

    ATM = spot'u çevreleyen İKİ strike (strike'ta doğrusal enterpolasyon; yalnız biri
    ölçülüyse o). Tur-78 (R1-B7): eskiden IV'si olan EN YAKIN strike alınıyordu — ATM
    strike'ların IV'si boşken spot 100'de 70 strike'ın 0,9'u "ATM IV" oldu ve koninin yedi
    penceresinde de yüzdelik 100 basıldı. Çevreleyen strike'larda IV yoksa ölçüm yoktur."""
    from datetime import date

    from nihai.ai.option_flow import chain_side

    try:
        provider = _provider()
        chain = provider.get_option_chain(symbol.upper(), None) or {}
        expiries = sorted(str(e) for e in chain.get("all_expiries", []) if str(e) > date.today().isoformat())
        if expiries:
            chain = provider.get_option_chain(symbol.upper(), expiries[0]) or {}
        expiry = str(chain.get("expiry") or "")
        spot = _f(chain.get("underlying_price"))
        if not spot or spot <= 0 or expiry <= date.today().isoformat():
            return None, None
        values = []
        for side in ("calls", "puts"):
            data = chain_side(chain.get(side))
            if data is None or "iv" not in data:
                continue
            strikes, ivs = data["strike"], data["iv"]
            fin = np.isfinite(strikes) & (strikes > 0)
            below, above = strikes[fin & (strikes <= spot)], strikes[fin & (strikes >= spot)]
            pts = []
            for k0 in ([below.max()] if below.size else []) + ([above.min()] if above.size else []):
                sel = fin & (strikes == k0) & np.isfinite(ivs) & (ivs > 0)
                if sel.any():
                    pts.append((float(k0), float(np.mean(ivs[sel]))))
            if len(pts) == 2 and pts[1][0] > pts[0][0]:
                (k_lo, iv_lo), (k_hi, iv_hi) = pts
                values.append(iv_lo + (iv_hi - iv_lo) * (spot - k_lo) / (k_hi - k_lo))
            elif pts:
                values.append(pts[0][1])
        return (float(np.mean(values)) if values else None), expiry
    except Exception as exc:
        logger.debug(f"[api_quant] ATM IV ölçülemedi: {exc}")
        return None, None


def _cone_report(data: dict, symbol: str, ppy: int, estimator: str,
                  iv_current: float | None = None) -> dict | None:
    from nihai.ai.quant_math import volatility_cone

    iv, expiry = (iv_current, None) if iv_current is not None else _current_atm_iv(symbol)
    result = volatility_cone(data["open"], data["high"], data["low"], data["close"],
                             estimator=estimator, iv_current=iv, ppy=ppy)
    if result is not None:
        result.update({"iv_expiry": expiry, "iv_source": "girdi" if iv_current is not None else "canli_zincir",
                       "rv_as_of": data["dates"][-1]})
    return result


@router.get("/api/quant/vol_cone")
def quant_vol_cone(symbol: str = "SPY", lookback: int = 1000,
                   estimator: str = "yang_zhang", iv_current: float | None = None) -> dict:
    """10/20/30/60/90/180/252 işlem günlük OHLC konisi ve güncel IV sırası."""
    data = _ohlc(symbol, "1d", lookback)
    if data is None:
        return _fail("OHLC_YOK", symbol=symbol.upper())
    cone = _cone_report(data, symbol, 252, estimator, iv_current)
    if cone is None:
        return _fail("KONI_OLCULEMEDI", symbol=symbol.upper())
    return {"ok": True, "symbol": symbol.upper(), "interval": "1d", "cone": cone}


# ===========================================================================
# 3) REJİM — kural etiketi + Gauss HMM + BOCPD
# ===========================================================================
def _bocpd_series(rets: np.ndarray) -> dict | None:
    """BOCPD çıktısını grafiğe hazırla: posterior serisi + kırılma indisleri."""
    from nihai.ai import advanced_math as am

    res = _guard(am.bocpd, rets)
    if not res:
        return None
    cp = np.asarray(res.get("changepoint_prob", []), dtype=float)
    mrl = np.asarray(res.get("map_run_length", []), dtype=float)
    if cp.size == 0:
        return None
    # Kırılma İŞARETİ: MAP run-length'in DÜŞTÜĞÜ barlar (docstring'in "birincil sinyal"i:
    # run-length zirveye tırmanır, kırılmada 1'e düşer).
    drops = (np.where(np.diff(mrl) < -1.0)[0] + 1) if mrl.size > 1 else np.array([], dtype=int)
    return {"changepoint_prob": _arr(cp, 4), "map_run_length": _arr(mrl, 1),
            "changepoints": [int(i) for i in drops[-40:]],
            "n_changepoints": int(drops.size)}


def _vade_yapisi(history: int = 250) -> dict | None:
    """CBOE volatilite VADE YAPISI — piyasa geneli, SEMBOLDEN BAĞIMSIZ, KARAR-DIŞI.

    ts_slope = VIX3M/VIX (>1 contango/sakin, <1 backwardation/stres) · near_slope =
    VIX/VIX9D · VVIX · SKEW · VIX'in genişleyen-pencere yüzdeliği. Kaynak `cboe_data`
    (CBOE CDN, ANAHTARSIZ, günlük önbellek).

    **NEDEN YALNIZ RAPOR, NEDEN KAPI DEĞİL (Tur-74'te ölçüldü).** Dış liste (Madde 9)
    "backwardation tüm swing alımlarını kilitlesin" öneriyordu. Bu bir KENAR iddiasıdır
    ve bu depoda ölçülmemiş hiçbir bileşen karara giremez (ACI DERS #5, Tur-51
    `IV_REGIME_BONUS=off` ilkesi). Blok burada yalnız GÖSTERİLİR.

    **KAYNAK DEĞİŞİMİ DE YAPILMADI, ÇÜNKÜ BEDAVA DEĞİL.** Aynı listenin ikinci yarısı
    `strategy._load_vix()`'i yfinance'ten CBOE'ye taşımayı istiyordu. Ölçüldü
    (2016-09-06 → 2026-09-04, 2515 ortak gün):
      · VIX rejim KAPISI (`VIX > 28`) kararı **0/2515 günde ayrışıyor** — kapı için
        kaynak değişimi tamamen etkisiz;
      · ama `_vix_close_before` tek tüketici DEĞİL: `meta_features` VIX'i meta-modelin
        ÖZELLİK vektörüne sokar ve günlerin yalnız %3,9'u bit-denktir (std 0,052,
        max |Δ| 2,61). `data/meta_model.json`'ın standartlaştırılmış katsayısıyla
        (w=+0,3076, sd=3,9646) bu **p_win'de en fazla 0,0506** oynama demektir — eşiğe
        yakın bir kararı ÇEVİRMEYE yeter.
    Yani kaynağı sessizce değiştirmek, yfinance serisiyle EĞİTİLMİŞ bir modele
    train/serve kayması sokardı. `cboe_data`'nın kendi docstring'i tam da bunu şart
    koşuyordu ("canlıya taşınacaksa A/B ölçümüyle taşınmalı"); A/B yapıldı ve cevap
    **taşıma** değil. `_load_vix` yfinance'te KALDI.

    Dönüş fail-soft: ağ/önbellek yoksa None (uç düşmez)."""
    from nihai.data import cboe_data as cb

    ts = _guard(cb.term_structure, fetch=False)   # ÖNBELLEK-YALNIZ: uç ağa çıkmaz (BULGU-59-8)
    if ts is None or getattr(ts, "empty", True):
        return None
    tail = ts.tail(max(2, min(int(history), 2000)))
    son = tail.iloc[-1]

    def _col(name: str) -> list[float | None] | None:
        return _arr(tail[name].to_numpy(), 4) if name in tail.columns else None

    ts_slope = _f(son.get("ts_slope"), 4)
    # Önbellek-yalnız okuma BAYAT olabilir; bayatlığı SAKLAMAK, ölçüm yokluğunu ölçüm gibi
    # sunmanın takvim tarafı olurdu (Tur-31). Yaş AÇIKÇA taşınır, pano onu basar.
    son_tarih = tail.index[-1]
    try:
        import datetime as _dt
        bayat = int((_dt.date.today() - son_tarih.date()).days)
    except Exception:  # noqa: BLE001
        bayat = None

    def _gun(x: Any) -> str:
        d = getattr(x, "date", None)
        return str(d() if callable(d) else x)

    return {
        "tarih": _gun(son_tarih),
        "bayat_gun": bayat,
        "n": int(len(tail)),
        "dates": [_gun(i) for i in tail.index],
        "vix": _f(son.get("vix"), 2), "ts_slope": ts_slope,
        "near_slope": _f(son.get("near_slope"), 4),
        # Uzun uç AYRI raporlanır: CBOE VIX6M'i düzensiz yayımlar, o yüzden `far_slope`
        # son GEÇERLİ gününü de taşır — bayat bir uzun ucu bugünkü gibi basmak
        # (ya da ffill'lemek) vade yapısını yanlış gösterirdi.
        "far_slope": _f(son.get("far_slope"), 4),
        "far_slope_tarih": (None if "far_slope" not in tail.columns
                            or tail["far_slope"].last_valid_index() is None
                            else _gun(tail["far_slope"].last_valid_index())),
        "vvix": _f((son.get("vvix_ratio") or 0) * 100.0, 2) if "vvix_ratio" in tail.columns else None,
        "skew": _f(son.get("skew"), 2),
        "vix_pctile": _f(son.get("vix_pctile"), 1),
        "backwardation": (None if ts_slope is None else bool(ts_slope < 1.0)),
        "seri": {"ts_slope": _col("ts_slope"), "vix": _col("vix"),
                 "near_slope": _col("near_slope"), "far_slope": _col("far_slope"),
                 "vix_pctile": _col("vix_pctile")},
        "karar_disi": True,
    }


@router.get("/api/quant/regime")
def quant_regime(symbol: str = "SPY", interval: str = "1d", lookback: int = 500,
                 n_states: int = 0) -> dict:
    """Kural-tabanlı rejim + HMM olasılık şeridi/geçiş matrisi + BOCPD kırılma serisi."""
    from nihai.ai import quant_math as qm
    from nihai.ai import regime_hmm as rh

    data = _ohlc(symbol, interval, lookback)
    if data is None:
        return _fail(f"{symbol} için bar verisi alınamadı.", symbol=symbol)
    close = data["close"]
    if close.size < 80:
        return _fail(f"Yetersiz bar ({close.size}); HMM için en az 80 gerekir.", symbol=symbol)
    rets = qm.log_returns(close)

    rule = _guard(qm.classify_regime, close, data["high"], data["low"]) or {}

    # Durum SAYISI bir MODEL KARARIDIR: verilmezse BIC ile seçtirilir (elle 2 seçmek
    # "iki rejim vardır" varsayımını gizli bırakırdı).
    req = int(n_states) if int(n_states) in (2, 3, 4) else 0
    sel = _guard(rh.hmm_select_states, rets, (2, 3, 4)) if req == 0 else None
    model, ns, selection = None, req or 2, None
    if sel and sel.get("fits"):
        ns = int(sel["best"])
        model = (sel["fits"] or {}).get(ns)
        selection = {"n_states": ns, "decisive": bool(sel.get("decisive")),
                     "delta_bic": _f(sel.get("delta_bic"), 2),
                     "bic": {str(k): _f((v or {}).get("bic"), 1)
                             for k, v in (sel.get("fits") or {}).items()}}
    if model is None:
        model = _guard(rh.gaussian_hmm_fit, rets, ns)

    # Tur-73 (Madde 204/317): HMM rejimi KESİKLİ etiket verir; bu iki ölçü SÜREKLİ.
    # `critical_slowing_down` ÖRTÜŞMEYEN pencerelerle koşar (varsayılan `step=window`) —
    # örtüşen pencerelerde Kendall p'si durağan gürültüde bile %19,5 yanlış-pozitif
    # veriyordu (Tur-73'te ölçüldü). MFDFA aynı seriye ölçek-bağımlı bakar: `genislik`
    # sıfıra yakınsa TEK üstel yeter (tek-fraktal), genişse ölçeğe göre farklı davranış var.
    win = max(20, min(120, int(rets.size // 8)))
    csd = _guard(qm.critical_slowing_down, rets, win)
    mf = _guard(qm.mfdfa, rets)
    yapisal = {
        "csd": (None if csd is None else {
            "ar1_tau": _f(csd["ar1_tau"], 4), "ar1_p": _f(csd["ar1_p"], 5),
            "var_tau": _f(csd["var_tau"], 4), "var_p": _f(csd["var_p"], 5),
            "n_pencere": int(csd["n_pencere"]), "adim": int(csd["adim"]),
            "ortusen": bool(csd["ortusen"]), "devrilme": bool(csd["devrilme"]),
            "pencere": win}),
        "mfdfa": (None if mf is None else {
            "q": _arr(mf["q"], 2), "h": _arr(mf["h"], 4),
            "alpha": _arr(mf["alpha"], 4), "f_alpha": _arr(mf["f_alpha"], 4),
            "genislik": _f(mf["genislik"], 4), "h2": _f(mf["h2"], 4)}),
    }

    hmm_out = None
    if model:
        filt = _guard(rh.hmm_filter, rets, model)
        trans = _guard(rh.hmm_transition_summary, model)
        labels = _guard(rh.hmm_state_labels, model)
        nxt = _guard(rh.hmm_next_state_probs, rets, model, 1)
        probs = None
        if filt is not None and filt.get("filtered") is not None:
            p = np.asarray(filt["filtered"], dtype=float)
            if p.ndim == 2:
                probs = [_arr(p[:, j], 4) for j in range(p.shape[1])]
        hmm_out = {
            "n_states": int(model.get("n_states") or ns),
            "means": _arr(model.get("means"), 8),
            "stds": _arr(model.get("stds"), 8),
            "labels": labels,
            "bic": _f(model.get("bic"), 1), "converged": bool(model.get("converged")),
            "state_probs": probs,
            "current": None if probs is None else [row[-1] for row in probs],
            "matrix": [[_f(v, 4) for v in row]
                       for row in (model.get("transition") or [])],
            "persistence": _arr((trans or {}).get("persistence"), 4) if trans else None,
            "expected_duration": ([_f(v, 1) for v in (trans or {}).get("expected_duration", [])]
                                  if trans else None),
            "stationary": _arr((trans or {}).get("stationary"), 4) if trans else None,
            "next": (None if not nxt else {
                "probs": _arr(nxt.get("next"), 4),
                "mode": int(nxt.get("mode")) if nxt.get("mode") is not None else None,
                "mode_prob": _f(nxt.get("mode_prob"), 4),
                "switch_prob": _f(nxt.get("switch_prob"), 4)}),
            "selection": selection,
        }

    return {
        "ok": True, "symbol": symbol.upper().strip(), "interval": interval,
        "n_bars": int(close.size), "dates": data["dates"][1:],
        "close": _arr(close[1:], 4),
        "rule": rule, "hmm": hmm_out, "yapisal": yapisal, "bocpd": _bocpd_series(rets),
        "vade_yapisi": _vade_yapisi(),
    }


# ===========================================================================
# 4) MONTE CARLO — GBM yelpazesi + terminal dağılım + blok-bootstrap
# ===========================================================================
@router.get("/api/quant/mc")
def quant_mc(symbol: str = "SPY", interval: str = "1d", lookback: int = 500,
             horizon: int = 21, paths: int = 2000, seed: int = 42) -> dict:
    """GBM yol yelpazesi (yüzdelik dilimler) + terminal histogram + bootstrap karşılaştırma."""
    from nihai.ai import quant_math as qm

    data = _ohlc(symbol, interval, lookback)
    if data is None:
        return _fail(f"{symbol} için bar verisi alınamadı.", symbol=symbol)
    close = data["close"]
    if close.size < 60:
        return _fail(f"Yetersiz bar ({close.size}).", symbol=symbol)

    rets = qm.log_returns(close)
    ppy = _ppy(interval)
    hz = max(1, min(int(horizon), 252))
    npaths = max(200, min(int(paths), 20000))
    s0 = float(close[-1])
    sigma_bar = _guard(qm.vol_close_to_close, close) or float(np.std(rets, ddof=1))
    sigma = float(qm.annualize_vol(sigma_bar, ppy))
    # Tur-78: `mc_gbm_paths` artımı (μ − ½σ²)dt'dir, yani μ ARİTMETİK sürüklenmedir. Log getiri
    # ortalamasını μ diye vermek İtô düzeltmesini İKİ KEZ düşer ve medyan yolu yılda ½σ² aşağı
    # kaydırırdı (σ=0,6'da 18 pp). E[ln S_T/S_0] = m·ppy·T olacak μ: m·ppy + ½σ².
    mu = float(np.mean(rets)) * ppy + 0.5 * sigma ** 2

    p = qm.mc_gbm_paths(s0=s0, mu=mu, sigma=sigma, days=hz, n_paths=npaths, seed=int(seed))
    if p.size == 0 or p.shape[1] < 2:
        return _fail("Monte Carlo yolu üretilemedi (geçersiz parametre).", symbol=symbol)

    fan = {f"p{q}": _arr(np.percentile(p, q, axis=0), 4) for q in (5, 25, 50, 75, 95)}
    terminal = p[:, -1]
    term_ret = terminal / s0 - 1.0
    boot = _guard(qm.mc_bootstrap_terminal, rets, hz, min(npaths, 4000), int(seed), 5) or {}

    return {
        "ok": True, "symbol": symbol.upper().strip(), "interval": interval,
        "s0": _f(s0, 4), "horizon": hz, "paths": npaths, "seed": int(seed),
        "mu_annual": _f(mu, 4), "sigma_annual": _f(sigma, 4), "ppy": ppy,
        "steps": list(range(hz + 1)), "fan": fan,
        "sample_paths": [_arr(p[i], 4) for i in range(min(20, p.shape[0]))],
        "terminal_hist": _histogram(terminal, bins=45),
        "terminal": {
            "p5": _f(float(np.percentile(term_ret, 5)), 4),
            "p50": _f(float(np.percentile(term_ret, 50)), 4),
            "p95": _f(float(np.percentile(term_ret, 95)), 4),
            "prob_loss": _f(float(np.mean(term_ret < 0.0)), 4),
            "expected": _f(float(np.mean(term_ret)), 4),
        },
        "bootstrap": {k: _f(v, 4) for k, v in boot.items()},
    }


# ===========================================================================
# 5) OPSİYON — Black-Scholes eğrileri (SAF hesaplayıcı, ağ gerektirmez)
# ===========================================================================
@router.get("/api/quant/options")
def quant_options(s: float = 100.0, k: float = 100.0, t: float = 0.25, r: float = 0.04,
                  sigma: float = 0.25, kind: str = "c", contracts: int = 1,
                  premium: float | None = None) -> dict:
    """Spot ızgarası boyunca fiyat + tüm Greek eğrileri + vade-sonu ödeme profili.

    `premium` verilirse implied vol de çözülür (piyasa fiyatından σ geri çıkarma)."""
    from nihai.ai import options_math as om
    from nihai.ai import quant_math as qm

    kd = "c" if str(kind).lower().startswith("c") else "p"
    try:
        sf, kf, tf, rf, sg = float(s), float(k), float(t), float(r), float(sigma)
    except (TypeError, ValueError):
        return _fail("Geçersiz sayısal girdi.")
    if sf <= 0 or kf <= 0 or tf <= 0 or sg <= 0:
        return _fail("S, K, T ve σ pozitif olmalı.")

    grid = np.linspace(max(0.35 * kf, 0.01), 1.65 * kf, 121)
    price_curve: list[float | None] = []
    greek_curves: dict[str, list] = {g: [] for g in
                                     ("delta", "gamma", "theta", "vega", "rho")}
    ext_curves: dict[str, list] = {g: [] for g in
                                   ("vanna", "volga", "ultima", "charm", "speed",
                                    "zomma", "color")}
    for sp in grid:
        price_curve.append(_f(qm.bs_price(float(sp), kf, tf, rf, sg, kd), 4))
        g = qm.bs_greeks(float(sp), kf, tf, rf, sg, kd) or {}
        for key in greek_curves:
            value = g.get(key)
            scale = 365 if key == "theta" else 100 if key == "vega" else 1
            greek_curves[key].append(_f(value / scale, 6) if value is not None else None)
        ge = om.bs_greeks_extended(float(sp), kf, tf, rf, sg, kd) or {}
        for key in ext_curves:
            ext_curves[key].append(_f(ge.get(key), 8))

    # Vade-sonu ödeme (intrinsik). Fiyat eğrisiyle arasındaki fark = ZAMAN DEĞERİ.
    payoff = [_f(max(float(sp) - kf, 0.0) if kd == "c" else max(kf - float(sp), 0.0), 4)
              for sp in grid]

    # Zaman aşınması: T→0 giderken değerin erimesi (theta'nın integrali).
    t_grid = np.linspace(max(tf, 1e-4), 1e-4, 60)
    theta_decay = [_f(qm.bs_price(sf, kf, float(tt), rf, sg, kd), 4) for tt in t_grid]

    # Vol duyarlılığı: σ ızgarasında fiyat (vega eğrisinin integrali).
    v_grid = np.linspace(max(0.02, 0.2 * sg), 2.5 * sg, 60)
    vol_curve = [_f(qm.bs_price(sf, kf, tf, rf, float(vv), kd), 4) for vv in v_grid]

    now = qm.bs_greeks(sf, kf, tf, rf, sg, kd) or {}
    now = {key: value / (365 if key == "theta" else 100 if key == "vega" else 1)
           for key, value in now.items()}
    now_ext = om.bs_greeks_extended(sf, kf, tf, rf, sg, kd) or {}
    bounds = om.bs_no_arb_bounds(sf, kf, tf, rf, kd)
    mult = 100 * max(1, int(contracts))
    iv_solved = None
    if premium is not None:
        iv_solved = _f(_guard(om.implied_vol, float(premium), sf, kf, tf, rf, kd), 4)
    from nihai.ai.portfolio import tims_stress
    stress = tims_stress([{"kind": kd, "spot": sf, "strike": kf, "t": tf, "sigma": sg,
                           "r": rf, "quantity": max(1, int(contracts)), "multiplier": 100}])

    return {
        "ok": True,
        "inputs": {"s": _f(sf, 4), "k": _f(kf, 4), "t": _f(tf, 6), "r": _f(rf, 4),
                   "sigma": _f(sg, 4), "kind": kd, "contracts": max(1, int(contracts))},
        "spot_grid": _arr(grid, 4),
        "price_curve": price_curve, "payoff": payoff,
        "greeks_curve": greek_curves, "greeks_ext_curve": ext_curves,
        "t_grid": _arr(t_grid, 6), "theta_decay": theta_decay,
        "vol_grid": _arr(v_grid, 4), "vol_curve": vol_curve,
        "tims": stress,
        "greek_units": {"theta": "per_calendar_day", "vega": "per_1_percentage_point_iv"},
        "now": {
            "price": _f(qm.bs_price(sf, kf, tf, rf, sg, kd), 4),
            "moneyness": _f(sf / kf, 4),
            "intrinsic": _f(max(sf - kf, 0.0) if kd == "c" else max(kf - sf, 0.0), 4),
            "greeks": {kk: _f(vv, 6) for kk, vv in now.items()},
            "greeks_ext": {kk: _f(vv, 8) for kk, vv in now_ext.items()},
            "position_greeks": {kk: _f((vv or 0.0) * mult, 4) for kk, vv in now.items()},
            "no_arb_bounds": (None if bounds is None else
                              {"lower": _f(bounds[0], 4), "upper": _f(bounds[1], 4)}),
            "implied_vol_from_premium": iv_solved,
        },
    }


# ===========================================================================
# 6) GÜLÜMSEME — canlı zincir → IV smile + SVI + Durrleman + RND + vade yapısı
# ===========================================================================
@router.get("/api/quant/smile")
def quant_smile(symbol: str = "SPY", expiry: str | None = None, r: float = 0.04,
                term: bool = True) -> dict:
    """Piyasa IV gülümsemesi, SVI uyumu, kelebek-arbitraj kontrolü, risk-nötr yoğunluk."""
    from datetime import date as _date

    from nihai.ai import options_math as om

    sym = symbol.upper().strip()
    try:
        prov = _provider()
        if not expiry:
            probe = prov.get_option_chain(sym, None) or {}
            exps = sorted(str(e) for e in (probe.get("all_expiries") or []))
            today = _date.today().isoformat()
            future = [e for e in exps if e > today]
            expiry = future[0] if future else (exps[-1] if exps else None)
        chain = prov.get_option_chain(sym, expiry)
    except Exception as e:  # noqa: BLE001
        return _fail(f"Zincir alınamadı: {type(e).__name__}: {e}"[:180], symbol=sym)
    if not chain or "calls" not in chain:
        return _fail("Sağlayıcı opsiyon zinciri vermedi.", symbol=sym)

    from nihai.risk.options_analyzer import days_to_expiry

    calls, puts = chain.get("calls"), chain.get("puts")
    under = float(chain.get("underlying_price") or 0.0)
    exp = str(chain.get("expiry"))
    dte = days_to_expiry(exp)
    tt = max(dte, 1) / 365.0
    if under <= 0 or calls is None or len(calls) < 5:
        return _fail("Zincir çok sığ (5'ten az strike) veya dayanak fiyatı yok.", symbol=sym)
    fwd = under * math.exp(float(r) * tt)

    def _side(df) -> dict:
        d = df.copy()
        d = d[d["strike"] > 0].sort_values("strike")
        strikes = d["strike"].to_numpy(dtype=float)
        ivs = (d["impliedVolatility"].to_numpy(dtype=float)
               if "impliedVolatility" in d.columns else np.zeros(len(d)))
        bid = d["bid"].to_numpy(dtype=float) if "bid" in d.columns else np.zeros(len(d))
        ask = d["ask"].to_numpy(dtype=float) if "ask" in d.columns else np.zeros(len(d))
        last = (d["lastPrice"].to_numpy(dtype=float) if "lastPrice" in d.columns
                else np.zeros(len(d)))
        mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0, last)
        return {"strikes": strikes, "iv": ivs, "mid": mid}

    c = _side(calls)
    p = _side(puts) if puts is not None and len(puts) else None

    # Gülümseme CALL tarafından kurulur. Ekstrapolasyon YOK: piyasa ne verdiyse o
    # (np.interp uçları sabitler ve sahte bir düz kanat üretirdi).
    k_log = np.log(c["strikes"] / fwd)
    mask = (np.isfinite(k_log) & np.isfinite(c["iv"]) & (c["iv"] > 0.001)
            & (np.abs(k_log) < 1.2))
    k_m, iv_m, strikes_m, mid_m = k_log[mask], c["iv"][mask], c["strikes"][mask], c["mid"][mask]
    if k_m.size < 5:
        return _fail(f"Uyum için yeterli geçerli IV yok ({int(k_m.size)} nokta).",
                     symbol=sym, expiry=exp)

    svi = _guard(om.svi_fit, k_m, iv_m * iv_m * tt)
    svi_curve, durrleman = None, None
    if svi:
        kk = np.linspace(float(np.min(k_m)) - 0.05, float(np.max(k_m)) + 0.05, 120)
        w = _guard(om.svi_total_variance, kk, svi)
        if w is not None:
            svi_curve = {"k": _arr(kk, 5),
                         "iv": _arr(np.sqrt(np.maximum(np.asarray(w), 1e-12) / tt), 5)}
            g = _guard(om.svi_durrleman_g, kk, svi)
            if g is not None:
                ga = np.asarray(g, dtype=float)
                durrleman = {"k": _arr(kk, 5), "g": _arr(ga, 6),
                             "min_g": _f(float(np.nanmin(ga)), 6),
                             "butterfly_free": bool(np.nanmin(ga) >= -1e-9)}
    arb = _guard(om.svi_no_arbitrage, svi) if svi else None
    metrics = _guard(om.smile_metrics, k_m, iv_m, fwd, tt)
    rnd = _guard(om.risk_neutral_density, strikes_m, mid_m, tt, float(r))

    # Vade yapısı: birkaç vadenin ATM vol'ü (vade başına 1 zincir isteği; en fazla 6).
    term_out = None
    dupire = None
    if term:
        try:
            exps = sorted(str(x) for x in (chain.get("all_expiries") or []))
            today = _date.today().isoformat()
            ts, atms, dslices = [], [], []
            for exp_i in [x for x in exps if x > today][:6]:
                ch = _guard(prov.get_option_chain, sym, exp_i)
                cc = ch.get("calls") if ch else None
                if cc is None or len(cc) < 3 or "impliedVolatility" not in cc.columns:
                    continue
                u = float(ch.get("underlying_price") or under)
                cd = cc.copy()
                cd["_d"] = (cd["strike"] - u).abs()
                iv0 = float(cd.nsmallest(1, "_d").iloc[0].get("impliedVolatility") or 0.0)
                d_e = days_to_expiry(str(ch.get("expiry")))
                if iv0 > 0.001 and d_e > 0:
                    t_i = d_e / 365.0
                    ts.append(t_i)
                    atms.append(iv0)
                    # AYNI zincirden SVI dilimi de uydurulur (EK AĞ İSTEĞİ YOK) —
                    # Dupire yerel vol yüzeyi için ∂w/∂T buradan gelir.
                    f_i = u * math.exp(float(r) * t_i)
                    kd = np.log(cd["strike"].to_numpy(dtype=float) / f_i)
                    ivd = cd["impliedVolatility"].to_numpy(dtype=float)
                    md = np.isfinite(kd) & np.isfinite(ivd) & (ivd > 0.001) & (np.abs(kd) < 1.2)
                    if int(md.sum()) >= 5:
                        sv_i = _guard(om.svi_fit, kd[md], ivd[md] ** 2 * t_i)
                        if sv_i:
                            dslices.append({"t": t_i, "params": sv_i})
            if len(ts) >= 2:
                vts = _guard(om.vol_term_structure, ts, atms)
                if vts:
                    order = np.argsort(np.asarray(ts))
                    term_out = {
                        "expiries_years": _arr(vts["expiries"], 5),
                        "atm_vols": _arr(np.asarray(atms)[order], 5),
                        "forward_vols": _arr(vts["forward_vols"], 5),
                        "calendar_free": bool(vts["calendar_free"]),
                        "n_violations": int(vts["n_violations"]),
                        "slope": _f(vts["slope"], 5), "contango": bool(vts["contango"]),
                    }
            if len(dslices) >= 2:
                dl = _guard(om.dupire_local_vol_svi, dslices, None, 31)
                if dl:
                    dupire = {
                        "k": _arr(dl["k"], 5), "t": [_f(x, 5) for x in dl["t"]],
                        "local_vol": [[_f(v, 5) for v in row]
                                      for row in np.asarray(dl["local_vol"]).tolist()],
                        "implied_vol": [[_f(v, 5) for v in row]
                                        for row in np.asarray(dl["implied_vol"]).tolist()],
                        "n_calendar_ihlal": int(dl["n_calendar_ihlal"]),
                        "n_butterfly_ihlal": int(dl["n_butterfly_ihlal"]),
                        "n_slices": int(dl["n_slices"]),
                        "gecerli_oran": _f(dl["gecerli_oran"], 4),
                        "arbitrage_heatmap": {
                            "butterfly_g": [[float(v) if np.isfinite(v) else None for v in row] for row in dl["durrleman_g"]],
                            "calendar_slope": [[float(v) if np.isfinite(v) else None for v in row] for row in dl["calendar_slope"]],
                            "tolerance": dl["arbitrage_tolerance"],
                            "scope": "sampled_k_grid_not_global_arbitrage_proof",
                        },
                    }
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[api_quant] vade yapısı atlandı: {e}")

    # VARYANS SWAP adil kullanım fiyatı (Tur-72, §7.5): CBOE'nin VIX ayrık log-kontrat
    # replikasyonu. MODELDEN BAĞIMSIZ — tek bir σ'ya indirgemez, gülümsemenin TAMAMINI
    # kullanır. K_vol > ATM IV farkı doğrudan skew/kurtoz katkısıdır.
    varswap = None
    if p is not None:
        vs = _guard(om.variance_swap_strike, c["strikes"], c["mid"], p["strikes"],
                    p["mid"], fwd, tt, float(r))
        if vs:
            atm_i = int(np.argmin(np.abs(k_m)))
            atm_iv = float(iv_m[atm_i]) if iv_m.size else None
            varswap = {
                "k_var": _f(vs["k_var"], 6), "k_vol": _f(vs["k_vol"], 5),
                "variance_term": _f(vs["variance_term"], 6),
                "forward_term": _f(vs["forward_term"], 8),
                "k0": _f(vs["k0"], 2), "n_strikes": int(vs["n_strikes"]),
                "strikes": _arr(vs["strikes"], 2), "contrib": _arr(vs["contrib"], 8),
                "otm_tip": vs["otm_tip"],
                "kanat_alt": _f(vs["kanat_alt"], 4), "kanat_ust": _f(vs["kanat_ust"], 4),
                "kanat_kirpimi": _f(vs["kanat_kirpimi"], 4),
                "atm_iv": _f(atm_iv, 5),
                "skew_katkisi": (_f(vs["k_vol"] - atm_iv, 5)
                                 if (vs["k_vol"] is not None and atm_iv) else None),
            }

    # Put-call parite sapması (ATM civarı): arbitraj iddiası DEĞİL, veri-sağlığı teşhisi.
    parity = None
    if p is not None:
        common = np.intersect1d(c["strikes"], p["strikes"])
        if common.size:
            k_atm = float(common[int(np.argmin(np.abs(common - under)))])
            ci = int(np.where(c["strikes"] == k_atm)[0][0])
            pi = int(np.where(p["strikes"] == k_atm)[0][0])
            raw = _guard(om.put_call_parity, float(c["mid"][ci]), float(p["mid"][pi]),
                         under, k_atm, tt, float(r))
            if raw:
                parity = _nums(raw, 5) or {}
                parity["strike"] = _f(k_atm, 2)

    return {
        "ok": True, "symbol": sym, "expiry": exp, "dte": int(dte),
        "underlying": _f(under, 2), "forward": _f(fwd, 2), "t_years": _f(tt, 5),
        "all_expiries": [str(e) for e in (chain.get("all_expiries") or [])][:24],
        "market": {"k": _arr(k_m, 5), "iv": _arr(iv_m, 5),
                   "strikes": _arr(strikes_m, 2), "mid": _arr(mid_m, 3)},
        "svi": _nums(svi, 6),
        "svi_curve": svi_curve, "durrleman": durrleman,
        "svi_arb": _nums(arb, 6),
        "metrics": _nums(metrics, 5),
        "rnd": (None if not rnd else {
            "strikes": _arr(rnd["strikes"], 2), "density": _arr(rnd["density"], 8),
            "mass": _f(rnd["mass"], 4), "n_negative": int(rnd["n_negative"]),
            "mean": _f(rnd["mean"], 3), "std": _f(rnd["std"], 3),
            "skew": _f(rnd["skew"], 3), "kurtosis": _f(rnd["kurtosis"], 3)}),
        "term_structure": term_out, "dupire": dupire, "varswap": varswap,
        "parity": parity,
    }


# ===========================================================================
# 7) PORTFÖY — 7 yöntem + risk katkısı + korelasyon + PCA + etkin sınır
# ===========================================================================
class StressPosition(BaseModel):
    kind: Literal["stock", "c", "p"]
    quantity: FiniteFloat
    multiplier: FiniteFloat = Field(gt=0)
    spot: FiniteFloat = Field(gt=0)
    strike: FiniteFloat | None = None
    t: FiniteFloat | None = None
    sigma: FiniteFloat | None = None
    r: FiniteFloat | None = None
    q: FiniteFloat = 0.


class StressRequest(BaseModel):
    positions: list[StressPosition] = Field(min_length=1, max_length=32)
    available_margin: FiniteFloat | None = None


@router.post("/api/quant/tims")
def quant_tims(req: StressRequest) -> dict:
    from nihai.ai.portfolio import tims_stress
    out = tims_stress([p.model_dump() for p in req.positions], req.available_margin)
    return {"ok": True, **out} if out is not None else _fail("Stres girdileri eksik/geçersiz; portföy ölçülemedi.")


@router.get("/api/quant/insurance")
def quant_insurance(initial_floor: float, max_drawdown: float = .10, multiplier: float = 3.) -> dict:
    """Kayıtlı sanal NLV/ATH üzerinde CPPI ve TIPP; state oluşturmaz, güncellemez."""
    import json

    from nihai.ai.portfolio import cppi_allocation, tipp_allocation
    from nihai.config import STATE_FILE
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        equity, peak = (float(state["portfolio"][key]) for key in ("equity", "ath_equity"))
        tipp = tipp_allocation(equity, peak, initial_floor, max_drawdown, multiplier)
        if tipp is None:
            return _fail("Taban/sermaye/tepe parametreleri geçersiz.")
        cppi = cppi_allocation(equity, multiplier=multiplier, floor_value=initial_floor)
        return {"ok": True, "equity": equity, "asof": state.get("saved_at"), "cppi": cppi, "tipp": tipp,
                "note": "Kayıtlı sanal NLV ve ATH; başlangıç tabanı operatör girdisidir. Kesikli dengeleme/fiyat boşlukları tabanı delebilir."}
    except (OSError, ValueError, TypeError, KeyError):
        return _fail("Kayıtlı sanal portföy NLV/ATH ölçümü yok.")


class GreekExposure(BaseModel):
    delta: FiniteFloat
    gamma: FiniteFloat
    vega: FiniteFloat


class HedgeInstrument(GreekExposure):
    name: str = Field(min_length=1, max_length=60)
    lower: FiniteFloat
    upper: FiniteFloat


class GreekHedgeRequest(BaseModel):
    current: GreekExposure
    instruments: list[HedgeInstrument] = Field(min_length=1, max_length=32)
    neutralize: list[Literal["delta", "gamma", "vega"]] = Field(
        default=["delta", "gamma", "vega"], min_length=1, max_length=3)


@router.post("/api/quant/greek_hedge")
def quant_greek_hedge(req: GreekHedgeRequest) -> dict:
    """Çağıranın pay/kontrat başına Greekleriyle, emir üretmeyen hedge hesabı."""
    from nihai.ai.quant_math import greek_neutral_hedge

    selected = list(dict.fromkeys(req.neutralize))
    matrix = [[getattr(leg, greek) for leg in req.instruments] for greek in selected]
    out = greek_neutral_hedge(matrix, [getattr(req.current, greek) for greek in selected],
                              [leg.lower for leg in req.instruments], [leg.upper for leg in req.instruments])
    if out is None:
        return _fail("Bu Greekler ve adet sınırlarıyla nötrleme ölçülemedi.")
    weights = out["weights"]
    all_greeks = ("delta", "gamma", "vega")
    after = {g: getattr(req.current, g) + sum(getattr(leg, g) * w
             for leg, w in zip(req.instruments, weights, strict=True)) for g in all_greeks}
    return {"ok": True, "neutralize": selected, "continuous": True,
            "instruments": [{"name": leg.name, "quantity": float(w)}
                            for leg, w in zip(req.instruments, weights, strict=True)],
            "before": req.current.model_dump(), "after": {g: _f(v, 10) for g, v in after.items()}, "rank": out["rank"],
            "objective": out["objective"], "max_residual": out["max_residual"],
            "note": "Amaç adetlerin kareleri toplamıdır; para maliyeti/tam sayı icrası değildir. Vega birimi her girdide aynı olmalıdır."}


@router.get("/api/quant/portfolio")
def quant_portfolio(symbols: str = "", interval: str = "1d", lookback: int = 250,
                    max_weight: float = 0.30, kelly_fraction: float = 0.25,
                    target_vol: float = 0.10,
                    covariance_model: Literal["ledoit_wolf", "rmt"] = "ledoit_wolf",
                    herc_clusters: int = 0, mkt_etf: str = "SPY") -> dict:
    """Tahsis yöntemi karşılaştırması + risk katkıları + korelasyon/PCA + etkin sınır.

    `mkt_etf`: Black-Litterman piyasa önselinin okunacağı SPDR sepeti (önbellek-yalnız)."""
    from nihai.ai import quant_math as qm
    from nihai.ai.portfolio import max_sharpe_weights, min_variance_weights, propose

    got, err = _symbol_panel(symbols, interval, lookback)
    if got is None:
        return err or _fail("Panel kurulamadı.")
    panel, names = got["panel"], got["names"]
    missing, rets, ppy = got["missing"], got["rets"], got["ppy"]
    if not 0 <= herc_clusters <= len(names):
        return _fail("HERC küme sayısı 0 (otomatik) ile varlık sayısı arasında olmalı.")

    candidates = [{"symbol": sm, "strategy": "panel"} for sm in names]
    # Tur-79 (§4.1): Black-Litterman'ın π = δΣw_mkt önseli GERÇEK kapitalizasyon ağırlığı
    # ister. Sepet ÖNBELLEK-YALNIZ okunur (ağa çıkmaz, BULGU-59-8) ve ancak panelin
    # TAMAMI sepette bulunursa geçilir — eksik ismi ters-vol ile doldurmak İKİ FARKLI
    # önseli tek vektörde karıştırmak olurdu. Kaynak çıktıda beyan edilir.
    mkt_w, mkt_kaynak, mkt_eksik = None, "ters_vol_proxy", []
    sepet = _guard(_sepet_agirliklari, mkt_etf)
    if sepet:
        mkt_eksik = [s for s in names if s not in sepet]
        if not mkt_eksik:
            mkt_w = [sepet[s] for s in names]
            mkt_kaynak = f"{mkt_etf.upper()}_sepeti"
    prop = _guard(propose, candidates, panel,
                  {"max_weight": float(max_weight), "kelly_fraction": float(kelly_fraction),
                   "covariance_model": covariance_model, "herc_clusters": herc_clusters or None,
                   "market_weights": mkt_w})
    if prop is None or not prop.get("methods") or prop.get("covariance") is None:
        return _fail("Yöntem karşılaştırması koşulamadı (çözücü/veri).")

    cov = np.asarray(prop["covariance"], dtype=float)
    mu = rets.mean(axis=0)

    # Risk KATKISI: ağırlık ≠ risk. RC_i = w_i·(Σw)_i / (wᵀΣw), toplamı 1.
    methods_out: dict[str, Any] = {}
    for mname, m in (prop.get("methods") or {}).items():
        wd = m.get("weights") or {}
        w = np.array([float(wd.get(sm, 0.0)) for sm in names])
        pvar = float(w @ cov @ w)
        rc = (w * (cov @ w) / pvar) if pvar > 0 else np.zeros_like(w)
        ss = float(np.sum(w * w))
        methods_out[mname] = {
            "weights": [_f(x, 4) for x in w],
            "risk_contrib": [_f(x, 4) for x in rc],
            "vol_annual": _f(qm.annualize_vol(m["vol_per_bar"], ppy), 4),
            # `rets` log fark (`_symbol_panel`) → log yıllıklandırma (Tur-78, R2-B2).
            "return_annual": _f(qm.annualize_return(m["expected_return_per_bar"], ppy, log=True), 4),
            "sharpe_bar": _f(m.get("sharpe_per_bar"), 4),
            "effective_n": _f(1.0 / ss, 2) if ss > 0 else None,
            "max_weight": _f(float(np.max(np.abs(w))) if w.size else None, 4),
            "tail_risk": m.get("tail_risk"),
        }

    # ETKİN SINIR: min-var ↔ maks-Sharpe arasında kısıtlı karışım taraması.
    frontier: dict[str, list] = {"vol": [], "ret": [], "sharpe": []}
    n = len(names)
    lo, hi = np.zeros(n), np.full(n, float(max_weight))
    w_mv = _guard(min_variance_weights, cov, (lo, hi), 1.0)
    w_ms = _guard(max_sharpe_weights, mu, cov, (lo, hi), 1.0)
    if w_mv is not None and w_ms is not None:
        for lam in np.linspace(0.0, 1.0, 25):
            w = (1.0 - lam) * np.asarray(w_mv, dtype=float) + lam * np.asarray(w_ms, dtype=float)
            # Tur-78 (R2-B1): yeniden normalize ETME. N·tavan < 1 iken çözücüler bütçeyi
            # ulaşılabilir toplama kırpar (kalan nakit); w/Σw tavanı deliyordu (3 varlık, %30
            # → %33,3) ve yöntem tablosuyla çelişiyordu. Bütçe ulaşılabilirse Σw zaten 1'dir.
            if float(np.sum(w)) <= 0:
                continue
            v = math.sqrt(max(float(w @ cov @ w), 0.0))
            rr = float(w @ mu)
            frontier["vol"].append(_f(qm.annualize_vol(v, ppy), 4))
            frontier["ret"].append(_f(qm.annualize_return(rr, ppy, log=True), 4))
            frontier["sharpe"].append(_f(rr / v * math.sqrt(ppy), 3) if v > 0 else None)

    sd = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    corr = cov / np.outer(sd, sd)

    # Tur-73: her yönteme ÜÇ ek ölçü. Ağırlık tablosu "ne kadar aldık"ı söyler; bunlar
    # "ne kadar çeşitlendi · ne kadar çekildi · hedefe göre ne kadar büyük" sorularını
    # yanıtlar. HİÇBİRİ bir öneri değildir (modül anayasası: KARAR-DIŞI).
    from nihai.ai.portfolio import cdar as _cdar
    from nihai.ai.portfolio import diversification_ratio as _dr
    from nihai.ai.portfolio import vol_target_scale as _vts
    for mo in methods_out.values():
        w = np.array([float(x or 0.0) for x in mo["weights"]])
        mo["dr"] = _f(_guard(_dr, w, cov), 3)
        vt = _guard(_vts, w, cov, float(target_vol), ppy)
        mo["vt_olcek"] = _f(vt["olcek"], 3) if vt else None
        mo["vt_nakit"] = _f(vt["nakit_orani"], 3) if vt else None
        dd = _guard(_cdar, rets, w, 0.05)
        mo["cdar"] = _f(dd["cdar"], 5) if dd else None
        mo["max_dd_kum"] = _f(dd["max_dd"], 5) if dd else None

    # GERÇEKLEŞEN ortalama korelasyon — `options_math.implied_correlation` KİMLİĞİ,
    # örtük vol yerine ÖLÇÜLEN vol ile. Aynı payda (Σwσ)² − Σ(wσ)²; yani opsiyon
    # yüzeyinden gelen ρ_imp ile DOĞRUDAN karşılaştırılabilir bir taban verir.
    from nihai.ai.options_math import implied_correlation as _ic
    w_eq = np.full(len(names), 1.0 / max(1, len(names)))
    sig_bar = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    sigma_basket = math.sqrt(max(float(w_eq @ cov @ w_eq), 0.0))
    rho_out = _guard(_ic, sigma_basket, w_eq, sig_bar, corr)
    korelasyon_ozeti = None if rho_out is None else {
        "rho_realized": _f(rho_out["rho_realized"], 4),
        "rho_kimlik": _f(rho_out["rho_imp"], 4),
        "sigma_sepet": _f(qm.annualize_vol(sigma_basket, ppy), 4),
        "sigma_bagimsiz": _f(qm.annualize_vol(rho_out["sigma_basket_indep"], ppy), 4),
        "n": int(rho_out["n"]),
    }
    pca = _guard(qm.pca_risk, rets, min(3, n))

    # RMT GÜRÜLTÜ TEMİZLEME (Tur-72, §10.3): T/N küçüldükçe örneklem korelasyonunun
    # özdeğerlerinin çoğu SADECE gürültüdür; optimizasyon kovaryansı TERSLEDİĞİ için
    # tam da o en gürültülü yönlere yaslanır. Marčenko-Pastur eşiği gürültü alt-uzayını
    # düzler. Ledoit-Wolf ile RAKİP değil, farklı önseldir (o büzer, bu filtreler).
    rmt = _guard(qm.denoise_corr_mp, rets)
    rmt_out = None
    if rmt:
        rmt_out = {
            "eigenvalues": _arr(rmt["eigenvalues"], 5),
            "eigenvalues_denoised": _arr(rmt["eigenvalues_denoised"], 5),
            "lambda_plus": _f(rmt["lambda_plus"], 4),
            "lambda_minus": _f(rmt["lambda_minus"], 4),
            "n_signal": int(rmt["n_signal"]), "n_noise": int(rmt["n_noise"]),
            "q": _f(rmt["q"], 3),
            "cond_before": _f(rmt["cond_before"], 2), "cond_after": _f(rmt["cond_after"], 2),
            "corr_denoised": [[_f(v, 3) for v in row]
                              for row in np.asarray(rmt["corr_denoised"]).tolist()],
        }

    # PİYASA AĞI (Mantegna MST): korelasyon uzaklığının en kısa iskeleti. Yüksek
    # dereceli düğüm yapının merkezindedir (şok oradan yayılır); yaprak düğümler
    # çeşitlendirme adayıdır. Ağırlık ÖNERMEZ — topolojiyi gösterir.
    mst = _guard(qm.mst_edges, corr)
    mst_out = None
    if mst:
        mst_out = {"edges": [{"i": e["i"], "j": e["j"], "dist": _f(e["dist"], 4),
                              "corr": _f(e["corr"], 3)} for e in mst["edges"]],
                   "degree": list(mst["degree"]),
                   "total_length": _f(mst["total_length"], 4),
                   "merkez": names[int(np.argmax(mst["degree"]))] if mst["degree"] else None}

    return {
        "ok": True, "symbols": names, "missing": missing, "interval": interval,
        "n_obs": int(rets.shape[0]), "ppy": ppy, "methods": methods_out,
        "asset_stats": {
            "vol_annual": [_f(qm.annualize_vol(float(x), ppy), 4) for x in sd],
            "return_annual": [_f(qm.annualize_return(float(x), ppy, log=True), 4) for x in mu],
        },
        "correlation": [[_f(v, 3) for v in row] for row in corr.tolist()],
        "frontier": frontier,
        "pca": (None if pca is None else
                {"explained": _arr(pca["explained"], 4),
                 "loadings": [[_f(v, 4) for v in row] for row in pca["loadings"]]}),
        "condition_number": _f(prop.get("condition_number"), 2),
        "covariance_model": prop.get("cov_source"),
        "rmt": rmt_out, "mst": mst_out, "korelasyon": korelasyon_ozeti,
        "target_vol": _f(target_vol, 4),
        "faktor": _faktor_bloku(panel, names, interval),
        "stress": prop.get("stress"), "notes": prop.get("notes"),
        # ÖLÇÜM TAŞINMALIDIR (Tur-63): hangi piyasa önselinin kullanıldığı sayının
        # ANLAMINI değiştirir — "Black-Litterman" satırı ters-vol proxy'yle kurulduysa
        # o satır piyasa dengesini DEĞİL bir düşük-oynaklık önselini anlatır.
        "market_prior": {
            "kaynak": mkt_kaynak, "etf": mkt_etf.upper(),
            "sepet_okundu": sepet is not None,
            "eksik_semboller": mkt_eksik[:20], "n_eksik": len(mkt_eksik),
            "ipucu": (None if sepet is not None else
                      f"python -m nihai.data.etf_holdings --warm {mkt_etf.upper()}"),
        },
    }


def _faktor_bloku(panel: Any, names: list[str], interval: str) -> dict | None:
    """Madde 224/221 — EŞİT AĞIRLIKLI sepetin Fama-French 5 + momentum ayrıştırması.

    Neden eşit ağırlık: paneldeki hangi ağırlığın seçileceği bir KARARDIR ve bu blok
    karar-dışıdır; eşit ağırlık hiçbir yöntemi kayırmaz ve tekrarlanabilir. Yalnız
    GÜNLÜK barda çalışır — Ken French serisi günlüktür ve saatlik getiriyi günlük
    faktörle regres etmek iki farklı frekansı aynı denklemde toplamak olurdu.

    ÖNBELLEK-YALNIZ (`fetch=False`): uç Dartmouth'a çıkmaz. Doldurmak ayrı adımdır
    (`python -m nihai.data.factors`).
    """
    if str(interval).lower() not in ("1d", "1day", "d"):
        return {"ok": False, "reason": "FREKANS_UYUSMUYOR",
                "not": "Ken French serisi GÜNLÜKTÜR; saatlik getiri regres edilemez"}
    try:
        from nihai.data.factors import factor_attribution, french_factors
        f = french_factors(fetch=False, with_momentum=True)
        if f is None or f.empty:
            return {"ok": False, "reason": "FAKTOR_ONBELLEK_YOK",
                    "ipucu": "python -m nihai.data.factors --momentum"}
        sepet = panel[names].pct_change(fill_method=None).mean(axis=1).dropna()
        d = factor_attribution(sepet, f)
        if d is None:
            return {"ok": False, "reason": "ORTAK_GUN_YETERSIZ",
                    "not": "faktör serisi bir ay gecikmelidir; kesişim 60 günün altında"}
        return {"ok": True, "alpha_gunluk": _f(d["alpha"], 8),
                "alpha_yillik": _f(d["alpha_yillik"], 5),
                "alpha_t": _f(d["alpha_t"], 3), "r2": _f(d["r2"], 4),
                "n": d["n"], "lag": d["lag"],
                "yuklemeler": {k: _f(v, 4) for k, v in d["yuklemeler"].items()},
                "t": {k: _f(v, 2) for k, v in d["t"].items()},
                "son_faktor_tarihi": d["son_faktor_tarihi"]}
    except Exception as e:  # noqa: BLE001 — rapor katmanı, kart düşmesin
        logger.debug(f"[api_quant] faktör bloğu: {type(e).__name__}: {e}")
        return None


# ===========================================================================
# 8) MİKROYAPI — imza, sıçrama, makas, etki, VPIN, HAR-RV
# ===========================================================================
@router.get("/api/quant/statarb")
def quant_statarb(symbols: str = "", interval: str = "1d", lookback: int = 300,
                  n_factors: int = 3, esik: float = 1.25, rank: int = 1) -> dict:
    """Avellaneda-Lee öz-portföy s-skoru + Johansen kointegrasyon rank testi + VECM."""
    from nihai.ai import statarb as sa

    got, err = _symbol_panel(symbols, interval, lookback, min_rows=70)
    if got is None:
        return err or _fail("Panel kurulamadı.")
    names, rets, px = got["names"], got["rets"], got["px"]
    nf = max(1, min(int(n_factors), max(1, len(names) - 1)))

    al = _guard(sa.avellaneda_lee_scores, rets, nf, list(names), float(esik))
    if al is None:
        return _fail(f"Öz-portföy çözülemedi ({rets.shape[0]} bar × {len(names)} varlık; "
                     "en az 60 bar ve 3 varlık gerekir).", symbols=names)
    fac = al["factors"]
    xs = np.asarray(al["residual_cumsum"], dtype=float)

    assets = []
    for a in al["assets"]:
        assets.append({
            "symbol": a["symbol"], "index": int(a["index"]),
            "s_score": _f(a["s_score"], 4), "yon": a["yon"],
            "half_life": _f(a["half_life"], 2), "kappa": _f(a["kappa"], 5),
            "sigma_eq": _f(a["sigma_eq"], 6), "m": _f(a["m"], 6),
            "x_last": _f(a["x_last"], 6), "r2": _f(a["r2"], 4),
            "beta": [_f(b, 5) for b in a["beta"]],
            "reliable": bool(a["reliable"]), "neden": a["neden"],
        })

    # JOHANSEN kritik değer tablosu k−r ≤ 5 için gömülüdür; daha geniş evrende
    # istatistik hesaplanır ama karar VERİLMEZ (crit=None) — bu bilinçlidir.
    jo_syms = names[:5]
    log_px = np.log(px[:, :len(jo_syms)])
    jo = _guard(sa.johansen_test, log_px, 0, 1)
    johansen = None
    if jo:
        johansen = {
            "symbols": jo_syms, "kaynak": jo["kaynak"], "neden": jo["neden"],
            "eigenvalues": _arr(jo["eigenvalues"], 6),
            "trace_stat": _arr(jo["trace_stat"], 4),
            "max_eig_stat": _arr(jo["max_eig_stat"], 4),
            "trace_crit95": (None if jo["trace_crit"] is None else
                             _arr(np.asarray(jo["trace_crit"])[:, 1], 3)),
            "max_crit95": (None if jo["max_crit"] is None else
                           _arr(np.asarray(jo["max_crit"])[:, 1], 3)),
            "rank_trace": jo["rank_trace"], "rank_maxeig": jo["rank_maxeig"],
            "k": int(jo["k"]), "n": int(jo["n"]),
        }

    vecm = None
    rk = jo["rank_trace"] if (jo and jo["rank_trace"]) else 0
    want = max(1, min(int(rank), len(jo_syms) - 1)) if len(jo_syms) > 1 else 1
    if rk >= 1:
        vf = _guard(sa.vecm_fit, log_px, min(want, rk), 1)
        if vf:
            ect = np.asarray(vf["ect_series"], dtype=float)
            vecm = {
                "rank": int(vf["rank"]),
                "beta": [[_f(v, 5) for v in row] for row in np.asarray(vf["beta"]).tolist()],
                "alpha": [[_f(v, 5) for v in row] for row in np.asarray(vf["alpha"]).tolist()],
                "ect_last": _arr(np.asarray(vf["ect_last"]).ravel(), 5),
                "ect_series": [_arr(ect[:, j], 5) for j in range(ect.shape[1])],
                "half_life": [_f(v, 2) for v in vf["half_life"]],
            }

    step = max(1, xs.shape[0] // 400)      # pano için seyreltme (çizim maliyeti)
    return {
        "ok": True, "symbols": names, "missing": got["missing"], "interval": interval,
        "n_obs": int(rets.shape[0]), "esik": _f(float(esik), 3), "n_factors": int(nf),
        "dates": got["dates"][1:][::step],
        "assets": assets,
        "factors": {"explained": _arr(fac["explained"], 5),
                    "eigenvalues": _arr(fac["eigenvalues"], 5),
                    "weights": [[_f(v, 5) for v in row]
                                for row in np.asarray(fac["weights"]).tolist()]},
        "residual_cumsum": [_arr(xs[::step, j], 6) for j in range(xs.shape[1])],
        "johansen": johansen, "vecm": vecm,
        "vecm_neden": (None if rk >= 1 else
                       "Johansen %95 trace testi kointegrasyon bulmadı (rank 0) — "
                       "VECM kurmak yok olan bir denge ilişkisini varsaymak olurdu"),
    }


def _naive_index_tz(provider_name: str) -> str | None:
    """Naive bar indeksinin saat dilimi — TAHMİN EDİLMEZ, doğrulanmış sağlayıcıya göre eşlenir.
    Alpaca `_normalize_alpaca_bars` ve hub köprüsü naive-UTC üretir (Tur-77'de okundu);
    yfinance tz-farkında ET döndürür (bu dala hiç girmez). Doğrulanmamış sağlayıcı → None."""
    ad = (provider_name or "").lower()
    return "UTC" if ("alpaca" in ad or "hub" in ad) else None


def _ucurve_block(symbol: str, interval: str = "15m", lookback: int = _MAX_LOOKBACK) -> dict:
    """Tur-77 · Madde 4.5 — gün içi U-eğrisi (hacim + yüksek-düşük ARALIĞI) ve son tamamlanmış
    slotun anomalisi. Barlarda bid/ask yoktur: aralık bir MAKAS DEĞİLDİR ve likidite çekilmesi
    (makas↑ + hacim↓) bu kaynaktan ölçülemez → `cekilme` None + neden. KARAR-DIŞI."""
    from nihai.ai import microstructure as ms
    df = _bars_raw(symbol, interval, lookback)
    if df is None or getattr(df, "empty", True) or "volume" not in df.columns:
        return {"ok": False, "reason": "BAR_YOK"}
    idx = df.index
    tz = None
    if getattr(idx, "tz", None) is None:
        try:
            tz = _naive_index_tz(str(_provider().name))
        except Exception:  # noqa: BLE001 — rapor katmanı
            tz = None
        if tz is None:
            return {"ok": False, "reason": "SAAT_DILIMI_DOGRULANAMADI"}
    kw = {"index_tz": tz or "UTC", "slot_minutes": 30}
    vol = df["volume"].to_numpy(dtype=float)
    hacim = _guard(ms.intraday_profile, idx, vol, agg="sum", **kw)
    if hacim is None:
        return {"ok": False, "reason": "YETERSIZ_SEANS"}
    aralik = None
    if {"high", "low", "close"} <= set(df.columns):
        c = df["close"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rng = (df["high"].to_numpy(dtype=float) - df["low"].to_numpy(dtype=float)) / c * 1e4
        aralik = _guard(ms.intraday_profile, idx, rng, agg="median", **kw)
    # Rapor alanı KAYNAĞI söyler: tz-farkında indekste `index_tz` parametresi kullanılmaz (ölçüldü:
    # hub kapalıyken failover yfinance'ten tz-farkında bar geldi ve alan yanlışlıkla "UTC" yazıyordu).
    kaynak_tz = str(idx.tz) if getattr(idx, "tz", None) is not None else f"naive→{tz}"
    return {"ok": True, "interval": interval, "index_tz": kaynak_tz,
            "hacim": hacim, "aralik_bps": aralik,
            "hacim_anomali": _guard(ms.intraday_anomaly, idx, vol, agg="sum", **kw),
            "cekilme": None, "cekilme_reason": "MAKAS_VERISI_YOK (OHLCV'de bid/ask yok; uydurulmaz)"}


@router.get("/api/quant/micro")
def quant_micro(symbol: str = "SPY", interval: str = "1h", lookback: int = 600) -> dict:
    """Gürültü/likidite teşhisi: imza grafiği, sıçrama ayrışımı, makas, Kyle λ, VPIN, HAR."""
    from nihai.ai import microstructure as ms
    from nihai.ai import quant_math as qm

    data = _ohlc(symbol, interval, lookback)
    if data is None:
        return _fail(f"{symbol} için bar verisi alınamadı.", symbol=symbol)
    close, h, low, vol = data["close"], data["high"], data["low"], data["volume"]
    if close.size < 80:
        return _fail(f"Yetersiz bar ({close.size}).", symbol=symbol)
    rets = qm.log_returns(close)
    ppy = _ppy(interval)

    sig = _guard(ms.volatility_signature, close, 20)
    jump = _guard(ms.jump_split, rets)
    tsrv = _guard(ms.two_scale_rv, close, 5)
    roll = _guard(ms.roll_spread, close)
    cs = (_guard(ms.corwin_schultz_spread, h, low)
          if (h is not None and low is not None) else None)

    amihud = kyle = vpin_res = None
    if vol is not None and vol.size == close.size:
        amihud = _guard(ms.amihud_illiquidity, rets, close[1:] * vol[1:])
        bvc = _guard(ms.bulk_volume_classification, close, vol)
        if bvc is not None:
            signed = np.asarray(bvc["signed"], dtype=float)
            dp = np.diff(close)
            m = min(signed.size, dp.size)
            kyle = _guard(ms.kyle_lambda, dp[-m:], signed[-m:])
        vpin_res = _guard(ms.vpin, close, vol, 50)

    # HAR-RV: bar serisinden RV serisi kur (her nokta = w barlık pencere), fit + öngörü.
    har = None
    w = max(5, min(21, rets.size // 20))
    rv_series = np.array([float(np.dot(rets[i - w:i], rets[i - w:i]))
                          for i in range(w, rets.size + 1)])
    if rv_series.size >= 30:
        model = _guard(ms.har_rv_fit, rv_series, True)
        if model:
            f1 = _guard(ms.har_rv_forecast, model, rv_series, 1)
            f5 = _guard(ms.har_rv_forecast, model, rv_series, 5)

            def _annv(x):
                return (_f(qm.annualize_vol(math.sqrt(max(float(x), 0.0)) / math.sqrt(w), ppy), 4)
                        if x is not None else None)

            har = {
                "coef": _arr(model["coef"], 4), "intercept": _f(model["intercept"], 8),
                "r2": _f(model["r2"], 4), "persistence": _f(model["persistence"], 4),
                "n": int(model["n"]), "window_bars": int(w),
                "rv_vol": [_annv(v) for v in rv_series],
                "forecast_1": _annv(f1), "forecast_5": _annv(f5),
            }

    return {
        "ok": True, "symbol": symbol.upper().strip(), "interval": interval,
        "n_bars": int(close.size), "ppy": ppy,
        "signature": (None if not sig else {
            "scales": [int(x) for x in np.asarray(sig["scales"]).ravel()],
            "rv_per_bar": _arr(sig["rv_per_bar"], 12),
            "noise_slope": _f(sig["noise_slope"], 12),
            "noise_suspected": bool(sig["noise_suspected"])}),
        "jump": _nums(jump, 12), "two_scale_rv": _nums(tsrv, 12),
        "roll_spread": _nums(roll, 8), "corwin_schultz": _nums(cs, 8),
        "amihud": _nums(amihud, 12), "kyle": _nums(kyle, 12),
        "vpin": (None if not vpin_res else {
            "vpin": _f(vpin_res["vpin"], 4), "buckets": _arr(vpin_res["buckets"], 4),
            "n_buckets": int(vpin_res["n_buckets"]), "causal": bool(vpin_res.get("causal"))}),
        "har": har,
        "ucurve": _ucurve_block(symbol),
    }


# ===========================================================================
# 9) UYGULAMA — Almgren-Chriss optimal tasfiye
# ===========================================================================
@router.get("/api/quant/execution")
def quant_execution(q: float = 10000.0, sigma: float = 0.02, eta: float = 1e-6,
                    gamma_risk: float = 1e-6, t: float = 1.0, n: int = 20) -> dict:
    """Optimal tasfiye yörüngesi + risk-kaçınma taramasında etki↔risk ödünleşimi."""
    from nihai.ai import quant_math as qm

    ni = max(1, min(int(n), 200))
    base = _guard(qm.almgren_chriss_cost, q, sigma, eta, gamma_risk, t, ni)
    if base is None:
        return _fail("Geçersiz parametre (eta>0, t>0, n>=1, sigma>=0, gamma>=0).")
    twap = _guard(qm.almgren_chriss_cost, q, sigma, eta, 0.0, t, ni)

    sweep: dict[str, list] = {"gamma": [], "impact": [], "risk": [], "total": [], "kappa": []}
    for g in np.logspace(-9, -3, 25):
        rr = _guard(qm.almgren_chriss_cost, q, sigma, eta, float(g), t, ni)
        if not rr:
            continue
        sweep["gamma"].append(_f(float(g), 12))
        sweep["impact"].append(_f(rr["expected_temp_impact_cost"], 4))
        sweep["risk"].append(_f(rr["cost_variance"], 4))
        sweep["total"].append(_f(rr["risk_adjusted_cost"], 4))
        sweep["kappa"].append(_f(rr["kappa"], 4))

    return {
        "ok": True,
        "inputs": {"q": _f(q, 2), "sigma": _f(sigma, 6), "eta": _f(eta, 12),
                   "gamma_risk": _f(gamma_risk, 12), "t": _f(t, 4), "n": ni},
        "steps": list(range(ni + 1)),
        "trajectory": _arr(base["trajectory"], 2),
        "twap": None if not twap else _arr(twap["trajectory"], 2),
        "trades": _arr(base["trades"], 2),
        "cost": {"impact": _f(base["expected_temp_impact_cost"], 4),
                 "risk_variance": _f(base["cost_variance"], 4),
                 "risk_adjusted": _f(base["risk_adjusted_cost"], 4),
                 "kappa": _f(base["kappa"], 6),
                 "twap_impact": (_f(twap["expected_temp_impact_cost"], 4)
                                 if twap else None)},
        "sweep": sweep,
    }


# ===========================================================================
# 10) KELLY — büyüme eğrisi (log-servet) + kesirli Kelly noktası
# ===========================================================================
@router.get("/api/quant/kelly")
def quant_kelly(p_win: float = 0.55, win_loss: float = 1.5, fraction: float = 0.25,
                cap: float = 0.02) -> dict:
    """Bahis oranı → beklenen log-büyüme eğrisi; tam/kesirli Kelly noktaları işaretlenir."""
    from nihai.ai import quant_math as qm

    try:
        p, b = float(p_win), float(win_loss)
    except (TypeError, ValueError):
        return _fail("Geçersiz sayısal girdi.")
    if not (0.0 < p < 1.0) or b <= 0:
        return _fail("p_win (0,1) aralığında, win_loss > 0 olmalı.")

    k_full = float(qm.kelly_binary(p, b))
    k_frac = float(qm.fractional_kelly(k_full, float(fraction), float(cap)))

    # E[log servet] = p·ln(1+f·b) + (1−p)·ln(1−f); f ∈ [0,1) tanımlı bölgede taranır.
    f_grid = np.linspace(0.0, 0.99, 200)
    growth = np.array([
        0.0 if f <= 0 else (p * math.log(1.0 + f * b) + (1 - p) * math.log(1.0 - f))
        for f in f_grid
    ], dtype=float)

    # KÜMÜLATİF BEKLENTİ TEORİSİ (Tur-72, §9.1): Kelly log-fayda MAKSİMİZE eder ve
    # operatörün zarardan kaçınmasını yok sayar. CPT aynı bahsi insan karar vericinin
    # gördüğü gibi fiyatlar (kayıp λ=2,25 kat ağır, küçük olasılıklar şişer). İkisinin
    # AYRIŞMASI "matematiksel optimum" ile "katlanılabilir" arasındaki mesafedir.
    # NOT: bu bir boyutlama ÖNERİSİ değildir — sistem Kelly'yi kullanır, CPT teşhistir.
    cpt = _guard(qm.cpt_prospect, [b, -1.0], [p, 1.0 - p])
    cpt_out = None
    if cpt:
        lam_grid = np.linspace(1.0, 4.0, 31)
        ce_by_lam = []
        for lm in lam_grid:
            cc = _guard(qm.cpt_prospect, [b, -1.0], [p, 1.0 - p], 0.88, 0.88, float(lm))
            ce_by_lam.append(_f(cc["certainty_equivalent"], 5) if cc else None)
        cpt_out = {
            "cpt_value": _f(cpt["cpt_value"], 5),
            "certainty_equivalent": _f(cpt["certainty_equivalent"], 5),
            "ev": _f(cpt["ev"], 5), "premium": _f(cpt["premium"], 5),
            "weights": _arr(cpt["weights"], 5),
            "outcomes": _arr(cpt["outcomes_sorted"], 4),
            "loss_aversion": _f(cpt["loss_aversion"], 3),
            "kabul": bool(cpt["cpt_value"] > 0.0),
            "lam_grid": _arr(lam_grid, 3), "ce_by_lam": ce_by_lam,
            "lam_break": next((_f(float(lm), 3) for lm, ce in zip(lam_grid, ce_by_lam,
                                                                 strict=True)
                               if ce is not None and ce <= 0.0), None),
        }

    # Kazanma olasılığı taraması: "p kaça düşerse kenar biter?"
    p_grid = np.linspace(0.05, 0.95, 91)
    k_by_p = [_f(float(qm.kelly_binary(float(pp), b)), 4) for pp in p_grid]

    return {
        "ok": True,
        "inputs": {"p_win": _f(p, 4), "win_loss": _f(b, 4),
                   "fraction": _f(float(fraction), 4), "cap": _f(float(cap), 4)},
        "f_grid": _arr(f_grid, 4), "growth": _arr(growth, 6),
        "kelly_full": _f(k_full, 4), "kelly_fractional": _f(k_frac, 4),
        "growth_at_full": _f(float(np.interp(k_full, f_grid, growth)), 6),
        "growth_at_frac": _f(float(np.interp(k_frac, f_grid, growth)), 6),
        "breakeven_p": _f(1.0 / (1.0 + b), 4),
        "p_grid": _arr(p_grid, 4), "kelly_by_p": k_by_p,
        "edge": _f(p * b - (1.0 - p), 4), "cpt": cpt_out,
    }


def _expiry_years(expiry: str) -> float | None:
    """Hisse/ETF PM vadesi: NYSE erken kapanışları ve ET yaz saati dahil."""
    from nihai.data.market_calendar import NYSE, now_et

    try:
        close = NYSE.schedule.loc[expiry, "close"].to_pydatetime()
        return max(0.0, (close - now_et()).total_seconds() / (365 * 24 * 3600))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


@router.get("/api/quant/flow")
def quant_flow(symbol: str = "SPY", expiry: str | None = None, interval: str = "1d",
               lookback: int = 500, r: float = 0.04,
               clock_hours: int = 6, iv_change_per_hour: float = 0.0) -> dict:
    """Opsiyon akışı (GEX/PCR/max pain/UOA/vanna-charm) + bar akışı + açığa satış bayrakları.

    Her blok BAĞIMSIZ fail-soft: zincir yoksa bar tarafı yine döner, bar yoksa zincir
    tarafı döner. Ölçülemeyen blok `null` olur — sıfırla doldurulmaz."""
    from nihai.ai import microstructure as ms
    from nihai.ai import option_flow as of

    sym = symbol.upper().strip()
    out: dict[str, Any] = {"ok": True, "symbol": sym, "options": None, "bar": None,
                           "short": None, "notlar": []}

    # --- 1) Opsiyon zinciri tarafı -------------------------------------------------
    chain: dict = {}
    try:
        prov = _provider()
        if not expiry:
            probe = prov.get_option_chain(sym, None) or {}
            exps = sorted(str(e) for e in (probe.get("all_expiries") or []))
            future = [e for e in exps if (_expiry_years(e) or 0) > 0]
            expiry = future[0] if future else (exps[-1] if exps else None)
        chain = prov.get_option_chain(sym, expiry) or {}
    except Exception as e:  # noqa: BLE001
        out["notlar"].append(f"Zincir alınamadı: {type(e).__name__}")
    if chain and "calls" in chain:
        from nihai.risk.options_analyzer import days_to_expiry
        calls, puts = chain.get("calls"), chain.get("puts")
        under = float(chain.get("underlying_price") or 0.0)
        exp = str(chain.get("expiry"))
        dte = days_to_expiry(exp)
        tt = _expiry_years(exp)
        if under > 0 and tt is not None and tt > 0:
            gex = _guard(of.dealer_gamma_exposure, calls, puts, under, tt, float(r))
            pcr = _guard(of.put_call_ratio, calls, puts)
            mp = _guard(of.max_pain, calls, puts)
            uoa = _guard(of.unusual_activity, calls, puts)
            vc = _guard(of.dealer_vanna_charm, calls, puts, under, tt, float(r))
            out["options"] = {
                "expiry": exp, "dte": int(dte), "spot": _f(under, 4),
                "t_years": tt, "expiry_convention": "NYSE_PM",
                "source": chain.get("option_source", "yfinance"),
                "gex": None if gex is None else {
                    "strikes": _arr(gex["strikes"], 4),
                    "call_gex": _arr(gex["call_gex"], 2),
                    "put_gex": _arr(gex["put_gex"], 2),
                    "net_by_strike": _arr(gex["net_gex_by_strike"], 2),
                    "net_gex": _f(gex["net_gex"], 2),
                    "net_gex_full_move": _f(gex["net_gex_full_move"], 2),
                    "gex_unit": gex["gex_unit"],
                    "net_gex_bn": _f(gex["net_gex_bn"], 4),
                    "gamma_flip": _f(gex["gamma_flip"], 4),
                    "spot_vs_flip": _f(gex["spot_vs_flip"], 4),
                    "regime": gex["regime"], "call_wall": _f(gex["call_wall"], 4),
                    "put_wall": _f(gex["put_wall"], 4),
                    "grid_spot": _arr(gex["grid_spot"], 4),
                    "grid_net_gex": _arr(gex["grid_net_gex"], 2),
                    "gamma_source": gex["gamma_source"],
                    "dealer_convention": gex["dealer_convention"],
                    "n_strikes": gex["n_strikes"],
                    # Tur-78: kısmi ölçüm kapsamı — eksik OI/gamma satırı toplama girmez, sayılır.
                    "n_oi_eksik": gex["n_oi_eksik"], "n_gamma_eksik": gex["n_gamma_eksik"]},
                "pcr": None if pcr is None else {
                    "oi_pcr": _f(pcr["oi_pcr"], 4), "volume_pcr": _f(pcr["volume_pcr"], 4),
                    "measured": pcr["measured"],
                    "oi_call": _f(pcr.get("oi_call"), 0), "oi_put": _f(pcr.get("oi_put"), 0),
                    "volume_call": _f(pcr.get("volume_call"), 0),
                    "volume_put": _f(pcr.get("volume_put"), 0)},
                "max_pain": None if mp is None else {
                    "level": _f(mp["max_pain"], 4), "strikes": _arr(mp["strikes"], 4),
                    "pain": _arr(mp["pain"], 0), "n_strikes": mp["n_strikes"]},
                "uoa": None if uoa is None else {
                    "rows": [{k: (_f(v, 4) if isinstance(v, float) else v)
                              for k, v in row.items()} for row in uoa["rows"]],
                    "n_hit": uoa["n_hit"]},
                "vanna_charm": _nums(vc),
                "hedging_clock": _guard(of.dealer_hedging_clock, calls, puts, under, tt, float(r),
                                         hours=clock_hours, iv_change_per_hour=iv_change_per_hour),
            }
        else:
            out["notlar"].append("Dayanak fiyatı veya geçerli PM vadesi yok — opsiyon tarafı ölçülemedi.")
    elif not out["notlar"]:
        out["notlar"].append("Sağlayıcı opsiyon zinciri vermedi.")

    # --- 2) Bar akışı --------------------------------------------------------------
    df = _bars_raw(sym, interval, lookback)
    if df is not None and "volume" in df.columns:
        import numpy as _np
        vol = df["volume"].to_numpy(dtype=float)
        rv = _guard(ms.relative_volume, vol, _slots(df.index, interval))
        ts = (_guard(ms.trade_size_profile, vol, df["trade_count"].to_numpy(dtype=float))
              if "trade_count" in df.columns else None)
        vw = (_guard(ms.vwap_deviation, df["close"].to_numpy(dtype=float),
                     df["vwap"].to_numpy(dtype=float))
              if "vwap" in df.columns else None)
        tail = 120
        out["bar"] = {
            "interval": interval, "n_bars": int(len(df)),
            "dates": [str(ix)[:19] for ix in df.index[-tail:]],
            "volume": _arr(vol[-tail:], 0),
            "rvol": None if rv is None else {
                "rvol": _f(rv["rvol"], 3), "rvol_mean": _f(rv["rvol_mean"], 3),
                "volume": _f(rv["volume"], 0), "slot_median": _f(rv["slot_median"], 0),
                "n_history": rv["n_history"], "slot": rv["slot"]},
            "trade_size": None if ts is None else {
                "avg_trade_size": _f(ts["avg_trade_size"], 2), "z": _f(ts["z"], 3),
                "baseline": _f(ts["baseline"], 2), "whale": ts["whale"],
                "series": _arr(_np.asarray(ts["series"])[-tail:], 2),
                "n_history": ts["n_history"]},
            "vwap_dev": None if vw is None else {
                "dev_bps": _f(vw["dev_bps"], 2), "z": _f(vw["z"], 3),
                "vwap": _f(vw["vwap"], 4), "close": _f(vw["close"], 4),
                "series": _arr(_np.asarray(vw["series"])[-tail:], 2)},
            # BULGU-59-8: float kaynağı BAĞLANDI (`market_structure.float_shares`,
            # yfinance + 7 günlük disk önbelleği). Ölçülemezse alan None KALIR ve
            # `turnover_neden` nedenini söyler — 0 basılmaz (Tur-31/56).
            **_turnover_cells(sym, vol),
        }
        if "trade_count" not in df.columns:
            out["notlar"].append("Bu sağlayıcı `trade_count` vermiyor "
                                 "(ortalama işlem büyüklüğü ölçülemedi).")
        if "vwap" not in df.columns:
            out["notlar"].append("Bu sağlayıcı `vwap` vermiyor (VWAP sapması ölçülemedi).")
    else:
        out["notlar"].append("Bar verisi alınamadı.")

    # --- 3) Açığa satış bayrakları --------------------------------------------------
    out["short"] = _asset_flags(sym)
    if out["short"] is None:
        out["notlar"].append("Açığa satış bayrakları okunamadı (Alpaca anahtarı/SDK yok).")

    if out["options"] is None and out["bar"] is None:
        return _fail(" · ".join(out["notlar"]) or "Ölçülebilir veri yok.", symbol=sym)
    return out


# ===========================================================================
# 13) HAZİNE GETİRİ EĞRİSİ — NSS uyumu + inversiyon + başabaş enflasyon (Tur-73)
# ===========================================================================
@router.get("/api/quant/curve")
def quant_curve(since: str = "2015-01-01", history: int = 250) -> dict:
    """ABD Hazine sabit-vadeli eğrisi: NSS uyumu (seviye/eğim/eğrilik) + 10Y-2Y / 10Y-3M
    inversiyonu + TIPS başabaş enflasyonu. Kaynak FRED (ANAHTARSIZ, 7 günlük önbellek)."""
    from nihai.ai import quant_math as qm
    from nihai.data import fred_data as fd

    panel = _guard(fd.treasury_curve, since=since)
    if panel is None or getattr(panel, "empty", True):
        return _fail("FRED Hazine paneli çekilemedi (ağ/önbellek).", kaynak="FRED")
    vadeler = panel.attrs.get("vadeler") or fd.TREASURY_MATURITIES
    tam = panel.dropna()
    if tam.empty:
        return _fail("Hiçbir günde TÜM vadeler yayımlanmamış — eğri uydurulamaz.",
                     kolonlar=list(panel.columns), n_gun=int(panel.shape[0]))
    son = tam.iloc[-1]
    mats = np.array([float(vadeler[c]) for c in tam.columns], dtype=float)
    nss = _guard(qm.nss_fit, mats, son.to_numpy(dtype=float))

    # Seviye/eğim/eğrilik TARİHİ: her günü ayrı uydurmak pahalı; son `history` gün yeter.
    hist = tam.tail(max(20, min(int(history), 1000)))
    seri: dict[str, list] = {"tarih": [], "seviye": [], "egim": [], "egrilik": [], "rmse": []}
    for ts, satir in hist.iterrows():
        f = _guard(qm.nss_fit, mats, satir.to_numpy(dtype=float))
        if f is None:
            continue
        seri["tarih"].append(str(ts.date()))
        seri["seviye"].append(_f(f["seviye"], 4))
        seri["egim"].append(_f(f["egim"], 4))
        seri["egrilik"].append(_f(f["egrilik"], 4))
        seri["rmse"].append(_f(f["rmse"], 5))

    sp = _guard(fd.curve_spreads, since=since)
    spread_out: dict[str, Any] = {}
    if sp is not None and not getattr(sp, "empty", True):
        for kol in sp.columns:
            s = sp[kol].dropna()
            if s.empty:
                continue
            spread_out[kol] = {"son": _f(float(s.iloc[-1]), 3), "tarih": str(s.index[-1].date()),
                               "min_1y": _f(float(s.tail(252).min()), 3),
                               "max_1y": _f(float(s.tail(252).max()), 3),
                               "ters_gun_1y": int((s.tail(252) < 0).sum())}
    ters = {k: (v["son"] is not None and v["son"] < 0)
            for k, v in spread_out.items() if k in fd.CURVE_SPREADS}

    return {
        "ok": True, "asof": str(tam.index[-1].date()), "n_gun": int(tam.shape[0]),
        "kirpik": list(panel.attrs.get("kirpik") or []),
        "vadeler": [_f(m, 5) for m in mats],
        "vade_adlari": list(tam.columns),
        "getiriler": [_f(float(v), 3) for v in son.to_numpy(dtype=float)],
        "nss": (None if nss is None else {
            "beta0": _f(nss["beta0"], 4), "beta1": _f(nss["beta1"], 4),
            "beta2": _f(nss["beta2"], 4), "beta3": _f(nss["beta3"], 4),
            "tau1": _f(nss["tau1"], 4), "tau2": _f(nss["tau2"], 4),
            "rmse": _f(nss["rmse"], 5), "seviye": _f(nss["seviye"], 4),
            "egim": _f(nss["egim"], 4), "egrilik": _f(nss["egrilik"], 4),
            "fit": _arr(nss["fit"], 4)}),
        "faktor_serisi": seri, "spreads": spread_out, "ters_egri": ters,
    }


# ===========================================================================
# 14) MAKRO REJİM PANELİ (Tur-76 — Madde 151-174)
# ===========================================================================
@router.get("/api/quant/macro")
def quant_macro(since: str = "2015-01-01", history: int = 500,
                lag: bool = True) -> dict:
    """Makro göstergeler ve dört ETF vekilinin son durumu/serileri (KARAR-DIŞI).

    **ÖNBELLEK-YALNIZ okur** (`fetch=False`): uç 20 FRED serisini indirmez, diskteki
    kopyayı okur (BULGU-59-8 — "ölçüm bütçesini OKUYAN da öder"). Önbelleği doldurmak
    ayrı ve bilinçli bir adımdır: `python -m nihai.data.macro_regime`.
    """
    from nihai.data import macro_regime as mr

    d = _guard(mr.panel, since=since, lag=lag, fetch=False)
    if d is None or getattr(d, "empty", True):
        return _fail("MAKRO_ONBELLEK_YOK",
                     ipucu="python -m nihai.data.macro_regime ile önbelleği doldurun",
                     eksik=list(getattr(d, "attrs", {}).get("eksik") or []))
    tail = d.tail(max(20, int(history)))
    son = d.iloc[-1]
    # Her gösterge için SON DEĞER + o değerin gerçekten ölçüldüğü SON GÜN. İkisi
    # ayrı raporlanır: ffill edilmiş bir kolonun son değeri "bugünkü ölçüm" DEĞİLDİR.
    hucreler: dict[str, Any] = {}
    for kol in d.columns:
        s = d[kol].dropna()
        hucreler[kol] = {
            "son": _f(float(son[kol]), 4) if son[kol] == son[kol] else None,
            "tarih": d.attrs.get("source_dates", {}).get(kol, str(s.index[-1].date()) if len(s) else None),
            "n": int(len(s)),
            "min_1y": _f(float(s.tail(252).min()), 4) if len(s) else None,
            "max_1y": _f(float(s.tail(252).max()), 4) if len(s) else None,
        }
    return {
        "ok": True, "asof": str(d.index[-1].date()), "n_gun": int(d.shape[0]),
        "lag_uygulandi": bool(lag),
        "eksik": list(d.attrs.get("eksik") or []),
        "gostergeler": hucreler,
        "seri": {
            "tarih": [str(t.date()) for t in tail.index],
            **{k: _arr(tail[k].to_numpy(dtype=float), 4)
               for k in ("net_liq", "taylor_gap", "nfci", "curve_2s10s",
                         "breakeven", "term_premium", "copper_gold_etf", "gold_oil_etf", "hy_ig_etf", "bil_return20_ann") if k in tail.columns},
        },
    }


# ===========================================================================
# 15) ŞİRKET TEMELLERİ — SEC EDGAR XBRL (Tur-76 — Bölüm 9)
# ===========================================================================
def _market_cap(symbol: str, hisse_adedi: float | None,
                as_of: str | None = None) -> tuple[float | None, str]:
    """Piyasa değeri = son kapanış × hisse adedi. Adet yoksa **None** (uydurma yok)."""
    if hisse_adedi is None or hisse_adedi <= 0:
        return None, "hisse_adedi_yok"
    bars = _guard(_bars_raw, symbol, "1d", _MAX_LOOKBACK if as_of else 5)
    if bars is None or getattr(bars, "empty", True):
        return None, "fiyat_yok"
    try:
        if as_of:
            import pandas as pd
            cutoff = pd.Timestamp(as_of)
            index = pd.DatetimeIndex(bars.index)
            if cutoff.tzinfo is not None:
                cutoff = cutoff.tz_convert("UTC").tz_localize(None)
            if index.tz is not None:
                index = index.tz_convert("UTC").tz_localize(None)
            bars = bars[index <= cutoff]
        px = float(bars["close"].dropna().iloc[-1])
    except (KeyError, IndexError, ValueError):
        return None, "fiyat_yok"
    return px * float(hisse_adedi), "kapanis_x_10K_hisse_adedi"


@router.get("/api/quant/fundamentals")
def quant_fundamentals(symbol: str = "AAPL", as_of: str | None = None,
                       fetch: bool = False) -> dict:
    """Sembolün 10-K temelleri + sekiz skor + olay dosyalamaları (KARAR-DIŞI).

    `fetch` VARSAYILAN FALSE: uç EDGAR'a çıkmaz, damıtılmış önbelleği okur. Doldurmak
    ayrı adımdır (`python -m nihai.data.edgar_facts AAPL --refresh`). `as_of` verilirse
    panel O GÜN BİLİNEN rakamları gösterir (`filed <= as_of`).
    """
    from nihai.ai import fundamentals as fu
    from nihai.data import edgar_facts as ef

    ham = _guard(ef.facts, symbol.upper(), fetch=fetch)
    if ham is None or getattr(ham, "empty", True):
        return _fail("EDGAR_ONBELLEK_YOK", symbol=symbol.upper(),
                     ipucu=f"python -m nihai.data.edgar_facts {symbol.upper()} --refresh")
    yillik = _guard(ef.annual_frame, ham, as_of=as_of)
    if yillik is None or getattr(yillik, "empty", True):
        return _fail("YILLIK_KALEM_YOK", symbol=symbol.upper(), as_of=as_of)
    # Tur-81: `yillik` bir satır yukarıda AYNI girdilerle kuruldu — yeniden kurma.
    ikili: tuple[dict, dict] = _guard(ef.son_iki_yil, ham, as_of=as_of,
                                      annual=yillik) or ({}, {})
    cari, onceki = ikili
    if not cari:
        return _fail("DONEM_YOK", symbol=symbol.upper(), as_of=as_of)

    mcap, mcap_kaynak = _market_cap(symbol.upper(), cari.get("hisse_adedi"), as_of)
    median = ef.median_snapshot(as_of=as_of)
    history = [{"_donem_sonu": str(t.date()), **row.to_dict()}
               for t, row in yillik.iterrows()]
    rap = _guard(fu.rapor, cari, onceki, market_cap=mcap,
                 medyanlar=(median or {}).get("medyanlar"), history=history) or {}

    seri: dict[str, Any] = {"donem": [str(t.date()) for t in yillik.index]}
    for kol in ("hasilat", "net_kar", "faaliyet_nakit", "ozkaynak", "varliklar"):
        if kol in yillik.columns:
            seri[kol] = _arr(yillik[kol].to_numpy(dtype=float), 2)

    olaylar: dict[str, Any] = {"okundu": False}
    if fetch:                                    # dosyalama listesi AĞ ister
        d = _guard(ef.filings, symbol.upper())
        if d is not None and not getattr(d, "empty", True):
            if as_of:
                d = ef.bilinen(d, as_of=as_of)
            sekiz = d[d["form"].astype(str).str.startswith("8-K")]
            sahiplik = d[d["form"].astype(str).str.contains("13D|13G", regex=True,
                                                            na=False)]
            olaylar = {
                "okundu": True,
                "yeniden_beyan_4_02": _olay_ozet(ef.item_filings(sekiz, "4.02")),
                "yonetici_5_02": _olay_ozet(ef.item_filings(sekiz, "5.02")),
                "sahiplik_13d_g": _olay_ozet(sahiplik),
                "formlar": list(d.attrs.get("formlar") or [])[:40],
            }

    return {
        "ok": True, "symbol": symbol.upper(),
        "donem": cari.get("_donem_sonu"), "onceki_donem": onceki.get("_donem_sonu"),
        "as_of": as_of, "n_mali_yil": int(yillik.shape[0]),
        "n_kalem": int(yillik.shape[1]),
        "market_cap": _f(mcap, 0), "market_cap_kaynak": mcap_kaynak,
        "karisik_kalemler": list(yillik.attrs.get("karisik") or []),
        "skorlar": _nums_derin(rap),
        "seri": seri, "olaylar": olaylar,
        "evren_medyanlari": median,
    }


def _olay_ozet(d: Any) -> dict:
    """Dosyalama alt kümesini say + son üç tarihi taşı (tam liste panoya gitmez)."""
    if d is None or getattr(d, "empty", True):
        return {"n": 0, "son": []}
    return {"n": int(len(d)),
            "son": [str(t.date()) for t in d["filed"].head(3)]}


def _nums_derin(d: Any) -> Any:
    """İç içe skor sözlüklerini JSON'a hazırla — `None` KORUNUR (BULGU-56-1).

    `_nums` düz sözlük içindir; skorlar iki katman derindir (`piotroski.sinyaller`).
    Ölçülemeyen alanı düşürmek "ölçüm yok" mesajını sessizce yok ederdi.
    """
    if isinstance(d, dict):
        return {k: _nums_derin(v) for k, v in d.items()}
    # numpy dizisi JSON'a SERİLEŞTİRİLEMEZ (pydantic `PydanticSerializationError` atar) —
    # Tur-79 modülleri (tda/econometrics) ölçüleri dizi olarak döndürür. `tolist()` numpy
    # skalerlerini de Python tipine indirir, `None` yine KORUNUR (BULGU-56-1).
    if isinstance(d, np.ndarray):
        return _nums_derin(d.tolist())
    if isinstance(d, np.generic):
        return _nums_derin(d.item())
    if isinstance(d, (list, tuple)):
        return [_nums_derin(v) for v in d]
    if isinstance(d, bool) or d is None or isinstance(d, str):
        return d
    if isinstance(d, (int, float)):
        return _f(d, 6)
    return d


# ===========================================================================
# 16) SÜRÜKLENME KARNESİ — SR 11-7 model izleme (yol haritası §12.3)
# ===========================================================================
@router.get("/api/quant/drift")
def quant_drift(interval: str = "1m", ref_gun: int = 60, cur_gun: int = 30,
                bins: int = 10, min_n: int = 200, alpha: float = 0.05,
                populasyon: Literal["aday", "hepsi", "aday_disi"] = "aday",
                feature: str | None = None) -> dict:
    """Canlı özellik logunun REFERANS ↔ CARİ pencere sürüklenmesi (KARAR-DIŞI).

    Referans penceresi cari pencereden ÖNCE BİTER (`drift_monitor` sözleşmesi); çakışan
    iki pencere ölçümü kendi kendisiyle kıyaslardı. Uç AĞA ÇIKMAZ — yalnız
    `data/live_features_log.csv` okunur; log yoksa `ok:false` + doldurma yönergesi döner.
    PSI eşikleri (0,10 / 0,25) sektör geleneğidir, düzenleyici zorunluluk DEĞİLDİR.

    `populasyon` NEDEN VAR ve NEDEN VARSAYILANI `aday`: bu logda iki FARKLI satır türü
    yaşar — ADAY barlar (`cand_side` dolu; `quality_score`/`rr_ratio` gerçek değer taşır)
    ve ADAY-DIŞI barlar (sinyal yok; aynı kolonlar 0 yazılır). Ölçüldü (2026-09-16):
    aday-dışı payı Temmuz'da **%0**, Eylül'de **%85** — yani iki pencereyi ham hâlde
    kıyaslamak modelin dünyasını değil KAYIT EDİCİNİN ÖRNEKLEMESİNİ ölçer
    (`quality_score` PSI 2,85 · `rr_ratio` PSI 8,82 tamamen bu karışımın izidir).
    Karışık popülasyonda alınan bir PSI, sürüklenme ölçümü DEĞİLDİR. `populasyon_karnesi`
    seçim ne olursa olsun iki penceredeki payı taşır ki artefakt gizlenmesin.
    """
    import pandas as pd

    from nihai.ai import drift_monitor as dm
    from nihai.config import DATA_DIR

    yol = DATA_DIR / "live_features_log.csv"
    if not yol.exists():
        return _fail("CANLI_OZELLIK_LOGU_YOK", yol=str(yol),
                     ipucu="kayıt servisi (live_predict_service) hiç koşmamış olabilir")
    ham = _guard(dm.load_live_features, yol, interval=interval)
    if ham is None or getattr(ham, "empty", True) or "bar_ts" not in ham.columns:
        return _fail("LOG_BOS_VEYA_ARALIK_YOK", interval=interval, yol=str(yol))

    son = pd.to_datetime(ham["bar_ts"]).max()
    cur_bas = son - pd.Timedelta(days=int(max(1, cur_gun)))
    ref_bas = cur_bas - pd.Timedelta(days=int(max(1, ref_gun)))
    ref = ham[(ham["bar_ts"] >= ref_bas) & (ham["bar_ts"] < cur_bas)]
    cur = ham[ham["bar_ts"] >= cur_bas]

    # Popülasyon karnesi ÖNCE ölçülür: süzgeç ne olursa olsun operatör karışımı GÖRÜR.
    # AYRAÇ `cand_side` DEĞİL `quality_score > 0`: `cand_side` Tur-59'da (BULGU-59-1)
    # DOĞDU, ondan önceki satırların hepsi zaten adaydı ve kolon BOŞTUR — yani `cand_side`
    # ile süzmek referans penceresini SIFIRA indirir. Ölçüldü (2026-09-16): `cand_side`
    # dolu ∧ `quality_score == 0` → 0 satır, yani `quality_score > 0` her iki loglama
    # döneminde de aday kümesini verir (`rr_ratio > 0` ile 16.066/16.066 ÖZDEŞ).
    aday_maske = (ham["quality_score"] > 0 if "quality_score" in ham.columns
                  else pd.Series(True, index=ham.index))
    cs = ham["cand_side"].notna() if "cand_side" in ham.columns else None
    karne: dict[str, Any] = {"ayrac": "quality_score>0",
             "cand_side_dolu_ref": int(cs.loc[ref.index].sum()) if cs is not None else None,
             "cand_side_dolu_cur": int(cs.loc[cur.index].sum()) if cs is not None else None,
             "ref_aday": int(aday_maske.loc[ref.index].sum()), "ref_n": int(len(ref)),
             "cur_aday": int(aday_maske.loc[cur.index].sum()), "cur_n": int(len(cur))}
    karne["ref_aday_pay"] = _f(karne["ref_aday"] / karne["ref_n"], 4) if karne["ref_n"] else None
    karne["cur_aday_pay"] = _f(karne["cur_aday"] / karne["cur_n"], 4) if karne["cur_n"] else None
    if populasyon == "aday":
        ref, cur = ref[aday_maske.loc[ref.index]], cur[aday_maske.loc[cur.index]]
    elif populasyon == "aday_disi":
        ref, cur = ref[~aday_maske.loc[ref.index]], cur[~aday_maske.loc[cur.index]]

    if len(ref) < min_n or len(cur) < min_n:
        return _fail(f"PENCERE_YETERSIZ (ref={len(ref)}, cari={len(cur)}, "
                     f"gereken={min_n})", n_ref=int(len(ref)), n_cur=int(len(cur)),
                     populasyon=populasyon, populasyon_karnesi=karne,
                     ilk_bar=str(ham["bar_ts"].min())[:19], son_bar=str(son)[:19])

    rapor = _guard(dm.feature_drift_report, ref, cur, min_n=min_n, alpha=alpha, bins=bins)
    if rapor is None:
        return _fail("RAPOR_URETILEMEDI", n_ref=int(len(ref)), n_cur=int(len(cur)))

    # En çok sürüklenen özellik (ya da operatörün seçtiği) HAFTALIK dilimlenir: tek bir
    # PSI sayısı "ne zaman kaydı" sorusunu yanıtlamaz.
    sirali = list(rapor.get("siralama") or [])
    hedef = feature if feature in (rapor.get("ozellikler") or {}) else (
        sirali[0] if sirali else None)
    dilim = None
    if hedef is not None and hedef in ham.columns:
        dilim = _guard(dm.time_sliced_psi, ref[hedef].to_numpy(dtype=float),
                       cur[hedef].to_numpy(dtype=float),
                       cur["bar_ts"].to_numpy(), freq="W", bins=bins,
                       min_n=max(30, min_n // 4))

    ozet = {k: rapor.get(k) for k in
            ("n_buyuk", "n_orta", "n_stabil", "n_yetersiz", "m_test", "alpha", "min_n")}
    return {
        "ok": True, "interval": interval, "yol": str(yol),
        "populasyon": populasyon, "populasyon_karnesi": karne,
        "pencere": {"ref_bas": str(ref_bas)[:19], "ref_son": str(cur_bas)[:19],
                    "cur_bas": str(cur_bas)[:19], "cur_son": str(son)[:19],
                    "n_ref": int(len(ref)), "n_cur": int(len(cur)),
                    "n_sembol": int(ham["symbol"].nunique()) if "symbol" in ham else None},
        "ozet": _nums_derin(ozet),
        "siralama": sirali,
        "ozellikler": _nums_derin(rapor.get("ozellikler")),
        "dilim": ({"feature": hedef, **_nums_derin(dilim)} if dilim else None),
        "dilim_neden": None if dilim else "haftalık dilim ölçülemedi (dönem başına gözlem az)",
    }


# ===========================================================================
# 17) ÇOK DEĞİŞKENLİ OYNAKLIK — DCC · FIGARCH · MODWT (§6.1 · §6.2 · §6.5)
# ===========================================================================
@router.get("/api/quant/econometrics")
def quant_econometrics(symbols: str = "", symbol: str = "SPY", interval: str = "1d",
                       lookback: int = 900, horizon: int = 5) -> dict:
    """DCC-GARCH korelasyon yolu + FIGARCH uzun hafızası + Haar MODWT ölçek ayrışımı.

    `symbols` ≥2 sembol içerirse DCC bloğu kurulur (yoksa izleme listesinden okunur);
    `symbol` tek seri FIGARCH ve dalgacık bloklarını besler. MODWT DÖNGÜSELDİR (sıfır
    faz) → NEDENSEL DEĞİLDİR; canlı özellik olamaz, blok bunu `causal: false` ile taşır.
    DCC/FIGARCH parametreleri ÖRNEKLEM İÇİdir; buradan kenar iddiası çıkmaz.
    """
    from nihai.ai import econometrics as ec
    from nihai.ai import quant_math as qm

    out: dict[str, Any] = {"ok": True, "interval": interval, "horizon": int(horizon)}

    dcc = None
    veri, hata = _symbol_panel(symbols, interval, lookback, min_rows=260)
    if veri is not None:
        fit = _guard(ec.dcc_garch_fit, veri["rets"])
        if fit is not None:
            ileri = _guard(ec.dcc_forecast, fit, int(horizon))
            yollar = np.asarray(fit["corr_path"], dtype=float)   # (T, ikili_sayısı)
            R_son = np.asarray(fit["R_last"], dtype=float)
            dcc = {
                "names": veri["names"],
                "dates": veri["dates"][-yollar.shape[0]:],
                "pairs": [f"{veri['names'][i]}–{veri['names'][j]}"
                          for i, j in (fit.get("pairs") or [])],
                "a": _f(fit.get("a")), "b": _f(fit.get("b")),
                "persistence": _f(fit.get("persistence")),
                "loglik_corr": _f(fit.get("loglik_corr"), 3),
                "loglik_ccc": _f(fit.get("loglik_ccc"), 3),
                "lr_vs_ccc": _f(fit.get("lr_vs_ccc"), 3),
                "sinirda": _nums_derin(fit.get("sinirda")),
                "ort_korelasyon": _arr(yollar.mean(axis=1), 6),
                "ikili_yollar": [_arr(yollar[:, k], 6)
                                 for k in range(min(yollar.shape[1], 10))],
                "R_son": [[_f(v) for v in row] for row in R_son],
                "R_ileri": ([[_f(v) for v in row]
                             for row in np.asarray(ileri["R"], dtype=float)]
                            if ileri else None),
                "n": int(fit.get("n") or 0), "N": int(fit.get("N") or 0),
            }
    out["dcc"] = dcc
    out["dcc_neden"] = None if dcc else ((hata or {}).get("reason")
                                         or "DCC uyumu kurulamadı (T≥250 ve N≥2 gerekir)")

    tek = _ohlc(symbol, interval, lookback)
    if tek is None or tek["close"].size < 300:
        out.update(symbol=symbol.upper(), figarch=None, dalgacik=None,
                   tek_neden=f"{symbol.upper()} için ≥300 bar gerekir")
        return out

    rets = qm.log_returns(tek["close"])
    fig = _guard(ec.figarch_fit, rets)
    out["figarch"] = ({"d": _f(fig.get("d")), "omega": _f(fig.get("omega"), 12),
                       "phi": _f(fig.get("phi")), "beta": _f(fig.get("beta")),
                       "loglik": _f(fig.get("loglik"), 3), "aic": _f(fig.get("aic"), 3),
                       "bic": _f(fig.get("bic"), 3), "n": int(fig.get("n") or 0),
                       "trunc": int(fig.get("trunc") or 0),
                       "sinirda": _nums_derin(fig.get("sinirda")),
                       "kisit_ihlali": bool(fig.get("kisit_ihlali")),
                       "converged": bool(fig.get("converged")),
                       "uncond_var": _f(fig.get("uncond_var")),
                       "uncond_neden": fig.get("uncond_neden"),
                       "garch11_loglik": _f((fig.get("garch11") or {}).get("loglik"), 3),
                       "lr_vs_garch11": _f((fig.get("garch11") or {}).get("lr"), 3)} if fig else None)

    wv = _guard(ec.wavelet_variance, rets)
    mra = _guard(ec.modwt_mra, rets)
    dal = None
    if wv is not None:
        D = np.asarray(mra["D"], dtype=float) if mra else None
        dal = {"levels": int(len(wv["olcek"])), "olcek": [int(o) for o in wv["olcek"]],
               "nu2": _arr(wv["nu2"], 12), "nu2_yanli": _arr(wv["nu2_yanli"], 12),
               "enerji_payi": _arr(wv["enerji_payi"], 6),
               "M": [int(m) for m in wv["M"]], "log_egim": _f(wv.get("log_egim")),
               "detay_son": ([_arr(D[j][-250:], 8) for j in range(D.shape[0])]
                             if D is not None else None),
               "trend_son": (_arr(np.asarray(mra["S"], dtype=float)[-250:], 8)
                             if mra else None),
               "dates": tek["dates"][-250:], "causal": False}
    out["dalgacik"] = dal
    out.update(symbol=symbol.upper(), n_bar=int(tek["close"].size), tek_neden=None)
    return out


# ===========================================================================
# 18) TOPOLOJİK VERİ ANALİZİ — kalıcı homoloji H0/H1 (§10.1)
# ===========================================================================
@router.get("/api/quant/tda")
def quant_tda(symbols: str = "", interval: str = "1d", lookback: int = 500,
              window: int = 50, step: int = 2, k_max: int = 5,
              standardize: bool = False) -> dict:
    """Getiri bulutunun Vietoris–Rips H0/H1 çubukları + kayan pencere manzara normları.

    Kayan pencere NEDENSELDİR: her nokta yalnız pencere sonuna kadarki getirilerden
    kurulur. Gidea–Katz'ın *"çöküş öncesi manzara normu yükselir"* iddiası BU SİSTEMDE
    ÖLÇÜLMEMİŞTİR — panel iddiayı değil SERİYİ gösterir; kenar iddiası kanıt kapısı ister.
    """
    from nihai.ai import tda as td

    veri, hata = _symbol_panel(symbols, interval, lookback, min_rows=int(window) + 20)
    if veri is None:
        return hata or _fail("panel kurulamadı")
    rets = veri["rets"]
    w = int(window)
    kayan = _guard(td.sliding_window_tda, rets, window=w, step=int(step),
                   k_max=int(k_max), standardize=bool(standardize))
    diag = _guard(td.rips_persistence, rets[-w:])
    norms = _guard(td.landscape_norms, (diag or {}).get("h1"), k_max=int(k_max))
    manzara = _guard(td.persistence_landscape, (diag or {}).get("h1"), k_max=int(k_max))

    def _cubuk(bars: Any) -> list[dict]:
        arr = np.asarray(bars, dtype=float) if bars is not None else np.empty((0, 2))
        if arr.ndim != 2 or arr.size == 0:
            return []
        omur = np.where(np.isfinite(arr[:, 1]), arr[:, 1] - arr[:, 0], np.inf)
        return [{"dogum": _f(arr[k, 0]),
                 "olum": _f(arr[k, 1]) if np.isfinite(arr[k, 1]) else None,
                 "omur": _f(omur[k]) if np.isfinite(omur[k]) else None}
                for k in np.argsort(-omur)[:40]]

    kayan_blok = None
    if kayan is not None:
        uclar = np.asarray(kayan["end_index"], dtype=int)
        tarihler = veri["dates"][1:]          # getiri i, tarih i+1'e aittir
        kayan_blok = {
            "dates": [tarihler[i] if 0 <= i < len(tarihler) else None for i in uclar],
            "l1": _arr(kayan["l1"], 8), "l2": _arr(kayan["l2"], 8),
            "n_h1": _nums_derin(kayan["n_h1"]),
            "max_omur": _arr(kayan["max_omur"], 8),
            "h0_toplam_omur": _arr(kayan["h0_toplam_omur"], 8),
            "causal": True,
        }

    return {
        "ok": True, "names": veri["names"], "interval": interval,
        "window": w, "step": int(step), "n_obs": int(rets.shape[0]),
        "boyut": int(rets.shape[1]), "missing": veri["missing"],
        "H0": _cubuk((diag or {}).get("h0")), "H1": _cubuk((diag or {}).get("h1")),
        "n_H0": int(np.asarray((diag or {}).get("h0", [])).reshape(-1, 2).shape[0]),
        "n_H1": int(np.asarray((diag or {}).get("h1", [])).reshape(-1, 2).shape[0]),
        "max_edge": _f((diag or {}).get("max_edge")),
        "n_ucgen": int((diag or {}).get("n_triangles") or 0),
        "kesildi": int((diag or {}).get("kesildi") or 0),
        "normlar": _nums_derin(norms),
        "manzara": ({"grid": _arr(manzara["grid"], 6),
                     "lambda": [_arr(row, 6)
                                for row in np.asarray(manzara["landscapes"], dtype=float)]}
                    if manzara else None),
        "kayan": kayan_blok,
        "kayan_neden": None if kayan_blok else "kayan pencere ölçülemedi (pencere/adım?)",
        "uyari": "Kalıcılık çöküş ÖNGÖRMEZ; bu panel yalnız topolojik yapıyı ÖLÇER.",
    }


# ===========================================================================
# 19) DİSPERSİYON — örtük korelasyon + ileri varyans eğrisi (§7.5)
# ===========================================================================
@router.get("/api/quant/dispersion")
def quant_dispersion(etf: str = "SPY", n_top: int = 50, fetch: bool = False,
                     history: int = 750) -> dict:
    """ETF sepetinden kurulan günlük ÖRTÜK KORELASYON serisi + Cboe COR1M kıyası.

    `fetch` VARSAYILAN FALSE (BULGU-59-8): uç AĞA ÇIKMAZ, sepet ve IV önbelleğini okur.
    Sepet AĞIRLIKLARI BUGÜNÜNDÜR — geçmiş günlerin ρ'su o günün sepetiyle DEĞİL bugünün
    sepetiyle hesaplanır (PIT AĞIRLIK AÇIĞI); seri erken yıllarda seviyede AŞAĞI YANLIDIR,
    güvenilir olan son dönem ve DEĞİŞİM YÖNÜDÜR. `agirlik_pit_acigi` bunu taşır.
    """
    from nihai.data import cboe_data as cb
    from nihai.data import etf_holdings as eh

    sepet = _guard(eh.holdings, etf.upper(), fetch=fetch)
    if sepet is None or getattr(sepet, "empty", True):
        return _fail("SEPET_ONBELLEKTE_YOK", etf=etf.upper(),
                     ipucu=f"python -m nihai.data.etf_holdings --warm {etf.upper()}")
    kare = _guard(eh.implied_correlation_frame, etf.upper(), int(n_top), fetch=fetch,
                  holdings_df=sepet)
    if kare is None or getattr(kare, "empty", True):
        return _fail("ORTUK_KORELASYON_SERISI_BOS", etf=etf.upper(), n_top=int(n_top),
                     ipucu="DoltHub örtük vol toplu kopyası eksik olabilir "
                           "(`python -c \"from nihai.data import dolt_options as D; "
                           "D.bulk_csv('volatility_history')\"`)")
    kare = kare.tail(max(30, int(history)))
    rho = kare["rho_imp"].to_numpy(dtype=float)

    cor1m = None
    ref = _guard(cb.get_index, "COR1M", fetch=fetch)
    if ref is not None and not getattr(ref, "empty", True):
        ort = ref[ref.columns[0]].reindex(kare.index)
        ortak = ort.notna().to_numpy() & np.isfinite(rho)
        if int(ortak.sum()) >= 30:
            a, b = rho[ortak], ort.to_numpy(dtype=float)[ortak] / 100.0
            cor1m = {"dates": [str(x)[:10] for x in kare.index[ortak]],
                     "values": _arr(b, 6), "n_ortak": int(ortak.sum()),
                     "korelasyon_seviye": _f(float(np.corrcoef(a, b)[0, 1])),
                     "korelasyon_degisim": (_f(float(np.corrcoef(np.diff(a),
                                                                np.diff(b))[0, 1]))
                                            if ortak.sum() > 31 else None),
                     "ort_bizim": _f(float(np.mean(a))), "ort_cboe": _f(float(np.mean(b)))}

    fwd = _guard(cb.forward_variance, fetch=fetch)
    ileri = None
    if fwd is not None and not getattr(fwd, "empty", True):
        kolonlar = [c for c in fwd.columns if c.startswith("fwd_")]
        sonsatir = fwd[kolonlar].dropna(how="all").tail(1)
        if not sonsatir.empty and kolonlar:
            ileri = {"asof": str(sonsatir.index[-1])[:10],
                     "segment": [c[4:] for c in kolonlar],
                     "fwd_vol": [_f(v) for v in sonsatir.iloc[0].to_numpy(dtype=float)],
                     "n_negatif_gun": {c[4:]: int(fwd[f"neg_{c[4:]}"].sum())
                                       for c in kolonlar if f"neg_{c[4:]}" in fwd.columns},
                     "vade_gun": fwd.attrs.get("vade_gun")}

    ilk = sepet.head(int(n_top))
    return {
        "ok": True, "etf": etf.upper(), "n_top": int(n_top),
        "sepet_asof": sepet.attrs.get("as_of"), "sepet_kaynak": sepet.attrs.get("kaynak"),
        "sepet_bayat_gun": sepet.attrs.get("bayat_gun"),
        "n_bilesen": int(len(sepet)),
        "agirlik_toplam": _f(float(sepet["weight"].sum())),
        "eksik_semboller": list(kare.attrs.get("eksik_semboller") or [])[:40],
        "iv_son_tarih": str(kare.attrs.get("iv_son_tarih") or "")[:10] or None,
        "ilk_bilesenler": [{"ticker": str(t), "weight": _f(float(w), 6)}
                           for t, w in zip(ilk["ticker"], ilk["weight"], strict=False)],
        "dates": [str(x)[:10] for x in kare.index],
        "rho_imp": _arr(rho, 6),
        "kapsama": (_arr(kare["kapsama"].to_numpy(dtype=float), 4)
                    if "kapsama" in kare.columns else None),
        "n_gecerli": (int(np.count_nonzero(kare["gecerli"].to_numpy()))
                      if "gecerli" in kare.columns else None),
        "cor1m": cor1m, "ileri_varyans": ileri,
        "agirlik_pit_acigi": bool(kare.attrs.get("agirlik_sabit", True)),
    }
