"""
live_predict_service.py — Web panosu için SÜREKLİ fiyat-tahmin servisi (arka plan).

Seçili (≤30) hisse için her N saniyede generate_signal + meta-gate P(win) hesaplar ve
sonuçları thread-safe bir önbellekte tutar. Web endpoint (/api/predictions) bu önbelleği
okur → istek başına ağır hesap YOK. Tek paylaşılan provider (DATA_PROVIDER, varsayılan
hub) kullanılır → ekstra WS bağlantısı açılmaz (ücretsiz IEX tek-bağlantı limiti korunur).

DÜRÜST ÇERÇEVE (live_predict.py ile aynı):
- SİNYAL-ONLY: emir GÖNDERMEZ. P(win) "kesin tahmin" değil, meta-model güvenidir.
- P(win) yalnızca GERÇEK giriş sinyalinde (LONG/SHORT) anlamlıdır; WAIT'te None.
- "Hedef fiyat" stratejinin RR hedefidir, bir regresyon fiyat-tahmini değildir.

.env:
    PREDICT_SYMBOLS=AAPL,MSFT,...   # boşsa STREAM_SYMBOLS, o da boşsa bench.BASKET
    PREDICT_INTERVAL=1h             # boşsa HTF_INTERVAL
    PREDICT_REFRESH_SEC=20          # tur arası saniye (min 5)
"""
from __future__ import annotations

import contextlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import pandas as pd
from loguru import logger

from nihai.config import HTF_INTERVAL
from nihai.data.data_provider import CachingProvider, get_provider
from nihai.data.market_calendar import is_market_open
from nihai.signals.breadth import CONTEXT_FIELDS, CROSS_ASSET_PROXIES, MarketContext, relative_strength
from nihai.signals.market_regime import compute_regime
from nihai.signals.meta_features import extract_features
from nihai.signals.meta_gate import gate_health, get_gate
from nihai.signals.strategy import (
    SignalAction,
    _SharedIndicators,
    ema,
    generate_signal,
    last_valid,
    vote,
)

if TYPE_CHECKING:
    from nihai.data.data_provider import DataProvider, Interval
    from nihai.signals.meta_gate import _MetaGate


def _last_valid(series) -> float:
    """Serinin son NaN-olmayan değeri — Tur-43'te `strategy.last_valid`e indi (tek kaynak)."""
    return last_valid(series)


def _bias_and_confidence(price, e20, e50, r, adx_v, htf_dir=0, regime_tilt=1.0):
    """SAF çekirdek (hem canlı _forecast hem calibrate_forecast.py aynısını kullanır).

    Yön: EMA20/50 + RSI(14) + fiyat-EMA20 bileşkesi. Güven: |bias|·0.6 + ADX/40·0.4,
    sonra üst-zaman-dilimi onayı (htf_dir) ve rejim eğilimi (regime_tilt) ile modüle.
    Döner: (direction '+'/'-', bias [-1,1], confidence [0,1])."""
    # EŞİTLİK = BİLGİ YOK → 0.0 (Tur-28). Eskiden her iki karşılaştırma da `>` idi, yani
    # eşitlik SESSİZCE −1.0 (tam SHORT) sayılıyordu; oysa 4 satır aşağıdaki nötr dalı aynı
    # eşitliği `>=` ile LONG'a veriyor — kendi içinde çelişik. Somut zarar: tamamen DÜZ bir
    # seride (halt / illikit / seans-dışı) üç bileşen [−1, 0, −1] olup bias=−0,667 ve
    # **conf=0,60 ile SHORT** üretiyordu; yani SIFIR bilgili girdiye yüksek güven. Ties→0.0
    # ile aynı girdi bias=0 → mevcut nötr dalına düşer → conf≈0 (dürüst cevap).
    # Eşitlik DIŞINDAKİ her girdide davranış BİT-DENK (yalnız `==` dalı yeni).
    # Tur-61: yerel `_yon` kopyası `strategy.vote`a indi (Tur-33 TEK KAYNAK kuralı —
    # "ikinci bir kopya aynı hatayı yeniden doğurur"). `vote` ek olarak NaN/±inf'i de 0
    # sayar; buradaki tek KORUNMASIZ girdi `price`tı (`sig.price` NaN gelirse `_yon`
    # onu −1 = tam SHORT sayıyordu). Sonlu girdilerde BİT-DENK.
    comps: list[float] = []
    if not (np.isnan(e20) or np.isnan(e50)):
        comps.append(float(vote(e20, e50)))
    if not np.isnan(r):
        comps.append(float(np.clip((r - 50.0) / 25.0, -1.0, 1.0)))
    if not np.isnan(e20):
        comps.append(float(vote(price, e20)))
    bias = float(np.mean(comps)) if comps else 0.0
    if abs(bias) < 1e-9:   # nötr bırakma → fiyat-EMA50 ile kesinleştir
        bias = 1e-3 if (not np.isnan(e50) and price >= e50) else -1e-3
    direction = "+" if bias >= 0 else "-"

    adx_norm = 0.0 if np.isnan(adx_v) else float(np.clip(adx_v / 40.0, 0.0, 1.0))
    conf = abs(bias) * 0.6 + adx_norm * 0.4
    # MTF: üst zaman dilimi trendi aynı yöndeyse güven artar, çelişirse düşer (whipsaw filtresi).
    if htf_dir != 0:
        agree = (htf_dir > 0 and direction == "+") or (htf_dir < 0 and direction == "-")
        conf *= 1.15 if agree else 0.65
    conf *= regime_tilt   # rejim: trend-yönlü pozisyonlar daha güvenilir
    return direction, bias, float(np.clip(conf, 0.0, 1.0))


def _htf_direction(df_htf) -> int:
    """Üst zaman dilimi trend yönü: EMA20>EMA50 → +1, < → -1, belirsiz → 0.

    BULGU-61-1 (Tur-61): son satır `1 if e20 > e50 else -1` idi — yani TAM EŞİTLİK
    (e20 == e50) sessizce **−1 (ayı HTF onayı)** sayılıyordu ve bu, fonksiyonun KENDİ
    docstring'iyle ("belirsiz → 0") çelişiyordu. Ders bu dosyada zaten uygulanmıştı:
    20 satır YUKARIDA `_bias_and_confidence` Tur-28'de tam bu nedenle üç durumlu
    `_yon`a geçmişti, Tur-33 kuralı `strategy.vote`ta TEK KAYNAĞA indirmişti — kardeş
    fonksiyon açıkta kalmıştı (aynı dosyada üçüncü tekrar).

    Somut zarar: tamamen DÜZ bir HTF penceresinde (halt / illikit isim / seans-dışı)
    htf_dir −1 döner ve `_bias_and_confidence` `conf *= 0.65` uygular — SIFIR bilgiden
    %35 güven cezası. Ölçüldü: yükselen bir sinyalde conf 0,7760 → 0,5044.
    Gerçek veride nadir ama VAR: 400 sembolün 7.531.642 geçerli 1h barında tam eşitlik
    406 kez (%0,0054); eşitlik DIŞINDAKİ her girdide davranış BİT-DENKtir."""
    if df_htf is None or getattr(df_htf, "empty", True) or len(df_htf) < 50:
        return 0
    c = df_htf["close"]
    # `vote` NaN'ı da 0 sayar (ölçüm yokluğu ceza değildir) → ayrı isnan dalı gereksiz.
    return vote(_last_valid(ema(c, 20)), _last_valid(ema(c, 50)))


def _regime_tilt(regime: str | None, direction: str) -> float:
    """Rejim → yön bazlı güven çarpanı. BULLISH long'u, DEFENSIVE short'u destekler."""
    if not regime:
        return 1.0
    if regime == "BULLISH":
        return 1.10 if direction == "+" else 0.85
    if regime == "DEFENSIVE":
        return 0.80 if direction == "+" else 1.10
    return 1.0


def _yon_kipi() -> str:
    """Yayınlanan yön/güvenin FİYAT GİRDİSİ: 'bar' (karar barının kapanışı) | 'canli' (kotasyon).

    TEK KAYNAK — tahmin servisi, hisse-detay ucu (`/api/analyze`) ve kalibrasyon kurucusu
    (`research/calibrate_forecast.py`) aynı kuralı buradan okur; iki kopya sessizce ayrışırdı.

    Tur-83 (BULGU-83-1) varsayılanı 'bar' yaptı. Tur-20 `dir_bar`/`conf_bar` gölgesini "kanıt
    birikince karar verilsin" diye kurmuştu; 43.195 satır / 18 gün birikti ve AYNI satırlarda,
    AYNI referans fiyattan (tahminin kullanılabildiği tek fiyat: canlı kotasyon) ölçüldü:
    isabet canlı-yön %48,20 ↔ bar-yön %50,12 (+1,92 puan, gün-kümeli t=+12,8; 18 günün 15'inde
    bar önde). Fark YALNIZ yönün ayrıştığı %4,05'lik dilimden gelir ve orada canlı-yön %26,2
    isabet eder: kotasyon bar kapanışından sapıp EMA20'nin öbür yanına geçtiğinde (medyan
    |sapma| 15 bps) sapma 5 dakikada GERİ DÖNER. Yani canlı-yön kovaladığı gürültüyü tahmin
    ediyordu. Bu bir kenar DEĞİLDİR (bar-yön de yazı-turadır) — yazı-turanın ALTINDA kalan
    sistematik bir kusurun kaldırılmasıdır.
    `SIGNAL_ON_BAR_CLOSE=false|0|off|no` eski davranış ('canli'); başka HER değer 'bar' —
    yanlış yazım sessizce kovalayan varyanta düşemez."""
    raw = os.getenv("SIGNAL_ON_BAR_CLOSE", "true").strip().lower()
    return "canli" if raw in ("0", "false", "off", "no") else "bar"


# (mtime_ns, size, yön kipi) — kip de anahtardadır: kabul kararı ona bağlıdır.
_CALIB_SIG: tuple | None = None
_CALIB_PTS: list | None = None
_CALIB_META: dict = {}
_CALIB_SEEN = False


def _load_calibration(interval: str | None = None) -> list | None:
    """#2 — data/forecast_calibration.json: [[güven, gerçekleşen_isabet], ...] artan güven.
    `research/calibrate_forecast.py` üretir; yoksa/kabul edilmezse None (prob_cal gösterilmez).

    TAZE OKUR (Tur-28): dosya imzası değişince yeniden ayrıştırılır, değişmediyse maliyet tek
    `stat()` (prediction_tracker._file_sig deseni).

    ŞEMA KAPISI (Tur-83, BULGU-83-3). Eğri yalnız KENDİ ALANINDA uygulanır: `meta.girdi ==
    "yayinlanan_guven"` (canlı defterden, yayınlanan güven üzerinde kuruldu) VE `meta.yon`
    bugünkü yön kipine eşit — ayrıca `interval` verilirse `meta.interval` ona eşit olmalıdır.
    Eşleşmeyen eğri REDDEDİLİR (None), kırpılıp uydurulmaz. Neden: 23 Haziran'dan beri
    kullanılan tarihsel eğri (1) örneklem-İÇİ 19 ham kovaydı (0,4775 → 0,5772 gibi gürültü
    sıçramalı), (2) canlı `hit`ten FARKLI bir hedefe (bar→bar, beraberlik hariç) kurulmuştu,
    (3) HTF/rejim HARİÇ bir güven bekliyordu ve 1m eğrisi 1h detay analizine de uygulanıyordu.
    Ölçüldü (93.825 çözülmüş tahmin): Brier 0,25002 ↔ nedensel sabit taban 0,24940 — eğri
    SABİT TABAN ORANDAN KÖTÜYDÜ (gün-kümeli t=+4,38). Nöral gölgenin şema kapısıyla aynı
    ilke (Tur-41): alanı eşleşmeyen artefakt yüklenmez."""
    global _CALIB_SIG, _CALIB_PTS, _CALIB_META, _CALIB_SEEN
    import json

    from nihai.config import DATA_DIR
    p = DATA_DIR / "forecast_calibration.json"
    kip = _yon_kipi()
    try:
        st = p.stat()
        sig: tuple | None = (st.st_mtime_ns, st.st_size, kip)
    except OSError:
        sig = None                                  # dosya yok/erişilemez
    if not (_CALIB_SEEN and sig == _CALIB_SIG):
        pts: list | None = None
        meta: dict = {}
        if sig is not None:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                meta = d.get("meta") or {}
                if meta.get("girdi") == "yayinlanan_guven" and meta.get("yon") == kip:
                    parsed = sorted([(float(c), float(pr)) for c, pr in d["points"]])
                    pts = parsed if parsed else None
            except Exception:
                # Tur-84: GEÇİCİ okuma hatası (OneDrive/AV kilidi — `atomic_io`nun var olma nedeni)
                # imzayla önbelleğe YAZILMAZ; yazılsaydı eğri dosya bir sonraki gece değişene kadar
                # kapalı kalırdı. Bu çağrı "—" döner, sonraki çağrı yeniden dener (Gemini incelemesi).
                return None
        _CALIB_SIG, _CALIB_PTS, _CALIB_META, _CALIB_SEEN = sig, pts, meta, True
    if interval is not None and _CALIB_META.get("interval") != interval:
        return None
    return _CALIB_PTS


def _apply_calibration(points: list | None, confidence: float) -> float | None:
    """Ham güveni gerçekleşen-isabet olasılığına eşle (parçalı-doğrusal interpolasyon)."""
    if not points:
        return None
    if confidence <= points[0][0]:
        return points[0][1]
    if confidence >= points[-1][0]:
        return points[-1][1]
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        if confidence <= x1:
            t = (confidence - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)
    return points[-1][1]


# (dosya imzası, ayrıştırılmış içerik | None) — `_load_calibration` ile aynı tazelik sözleşmesi.
_BAND_ONBELLEK: tuple | None = None


def _load_band(interval: str | None) -> dict | None:
    """Tur-84 — data/forecast_band.json: yayınlanan ARALIĞIN kalibre z-kuantilleri.

    Gece görevi (`research/calibrate_forecast.py`) defterin `atr_pct` kolonundan kurar ve yürüyen
    kapıdan geçirir. Dosya yoksa / kapıdan geçmediyse / alanı eşleşmiyorsa None → pano aralığı "—"
    gösterir; kırpılıp uydurulmaz. ŞEMA KAPISI (Tur-83 eğrisi, Tur-41 nöral gölge ile aynı ilke):
    `meta.olcek == "atr14_yuzde_us"` + geçerli bir `us` (Tur-85: ölçek ATR14^us; üssü taşımayan
    Tur-84 dosyası ATR'yi ham ölçek sayar — üssü bilmeyen kod yeni dosyayı yanlış ölçekte uygulamasın
    diye ad değişti), `meta.durum == "GECTI"`, `meta.interval == interval`, `meta.horizon` = defterin
    çözüm ufku, `meta.referans == "bar_kapanisi"` (çapa; kotasyona çapalı kuantiller bar kapanışı
    çapasıyla birlikte kullanılamaz). TAZE OKUR (Tur-28): imza değişmediyse tek `stat()`."""
    global _BAND_ONBELLEK
    import json

    from nihai.config import DATA_DIR
    from nihai.execution.prediction_tracker import HORIZON_BARS
    p = DATA_DIR / "forecast_band.json"
    try:
        st = p.stat()
        sig: tuple | None = (st.st_mtime_ns, st.st_size)
    except OSError:
        sig = None
    if _BAND_ONBELLEK is None or _BAND_ONBELLEK[0] != sig:
        icerik = None
        if sig is not None:
            try:
                icerik = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None          # geçici okuma hatası önbelleğe YAZILMAZ (bkz. `_load_calibration`)
        _BAND_ONBELLEK = (sig, icerik)
    band = _BAND_ONBELLEK[1]
    if not isinstance(band, dict):
        return None
    meta = band.get("meta") or {}
    if (meta.get("olcek") != "atr14_yuzde_us" or "us" not in band or _band_us(band) is None
            or meta.get("durum") != "GECTI" or meta.get("referans") != "bar_kapanisi"
            or meta.get("interval") != interval or meta.get("horizon") != HORIZON_BARS
            or not isinstance(band.get("havuz"), dict)):
        return None
    return band


def _band_us(band: dict) -> float | None:
    """Aralığın ölçek üssü (Tur-85): yoksa 1,0 = ATR14'ün kendisi; sayı değilse / (0, 2] dışıysa None."""
    try:
        us = float(band.get("us", 1.0))
    except (TypeError, ValueError):
        return None
    return us if np.isfinite(us) and 0.0 < us <= 2.0 else None


def _apply_band(band: dict | None, symbol: str,
                atr_pct: float | None) -> tuple[float, float, float] | None:
    """(alt %, üst %, tipik |hareket| %) — ufuk sonundaki fiyatın SON BAR KAPANIŞINA göre aralığı.

    SAF ve TEK KAYNAK: servis, hisse-detay ucu ve gece görevinin yürüyen değerlendirmesi aynı
    fonksiyonu çağırır (puanlanan şey yayınlanan temsilin kendisidir — Tur-83 ilkesi).
    Ölçek oynaklıktır: ATR14 %'nin `us` üssü; z-kuantilleri sembolün KENDİSİNDEN (yeterli örnekte),
    yoksa havuzdan. Aralık yön İDDİA ETMEZ: yön tahmini yazı-turadır (Tur-83), oynaklık ise ölçülerek
    öngörülebilir (ATR ↔ |hareket| Spearman 0,46). `tipik` = |hareketin| MEDYANI (yarı yarıya).

    Tur-85 (BULGU-85-1): us < 1 ölçeği sembolün kendi düzeyine BÜZER (sembol kuantilleriyle
    ATR^us ≡ ATR^us · düzey^(1−us)). Ham ATR14 (us=1) aralığı sembol-içi ATR ondalığına göre
    kapsamayı %70,6 (en sakin) → %86,4 (en oynak) arasına yayıyordu: 14 barlık ATR anlık oynaklığın
    gürültülü tahminidir ve sapması 5 dakikada kısmen geri döner. `us` gece görevinde seçilir ve
    yalnız sabit ATR'yi yürüyen sınavda yenerse 1'den farklı yayınlanır (örneklem-dışı Mart–Mayıs
    tekrarında 60/60 gün 0,8, t=−8,46). Kapanış yarım saati için ek çarpan ÖLÇÜLDÜ ve örneklem-dışı
    tekrarlanmadığı için KALDIRILDI (BULGU-85-2, rapor §4/§8.1)."""
    if not band or atr_pct is None or not (np.isfinite(atr_pct) and atr_pct > 0):
        return None
    q = (band.get("sembol") or {}).get(symbol) or band.get("havuz")
    us = _band_us(band)
    if not isinstance(q, dict) or us is None:
        return None
    s = atr_pct ** us
    try:
        return (float(q["z_lo"]) * s, float(q["z_hi"]) * s, float(q["z_med_abs"]) * s)
    except (KeyError, TypeError, ValueError):
        return None


_ACILIS_NEG_TTL = 300.0   # sn — başarısız seans-açılışı çekiminin en sık yeniden deneme aralığı


def _et_gunu(ts) -> date:
    """Bar zamanının (naive = UTC, sistem sözleşmesi) ABD Doğu işlem günü."""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return t.tz_convert("America/New_York").date()


def _seans_acilisi(provider, symbol: str, et_gun: date) -> float | None:
    """`et_gun` işlem gününün SEANS açılışı — günlük bardan; yoksa None.

    Tur-84 (BULGU-84-3): "Değişim" / "% (gün)" `df[df.index >= df.index[-1].normalize()]`in ilk
    açılışından hesaplanıyordu. 1m'de bu, 220 barlık pencerenin UTC GECE YARISINDAN sonraki ilk
    barıdır: seans öncesi bir IEX işlemi ya da 13:10 ET'den sonra pencerenin başı. Ölçüldü (93.464
    satır): gerçek seans açılışı satırların yalnız %3,5'inde kullanılmış; |hata| medyan 0,32 puan,
    p90 1,92; satırların %16,5'inde gün-içi değişimin İŞARETİ yanlış. Alpaca günlük barı SEANS
    barıdır (IEX 1d açılışı = ilk seans-içi 1m açılışı, 4/4 sembolde birebir) ve oluşan bugünkü
    barı da döner. Bugünün barı yoksa (seans öncesi / veri yok) None: pano "—" gösterir."""
    try:
        d = provider.get_bars(symbol, "1d", lookback_bars=3)
        if d is None or getattr(d, "empty", True) or pd.Timestamp(d.index[-1]).date() != et_gun:
            return None
        o = float(d["open"].iloc[-1])
    except Exception:
        return None
    return o if np.isfinite(o) and o > 0 else None


# BULGU-59-1 (Tur-59) — ETİKETSİZ HAVUZUN KÖK NEDENİ: özellik logu politikası.
# Tur-36 ölçmüştü: `live_features_log.csv`in 7.708 satırının yalnız 391'i etiketsizdi ve
# NEDENSEL olarak kullanılabilen kısmı 69 (+%1,5) — yani öz-denetimli ön-eğitimin YAKITI
# yoktu. Kök neden yöntem değil POLİTİKAYDI: özellik vektörü yalnız ADAY sinyal barlarında
# yazılıyordu (aşağıdaki `side is not None` kapısı). Taranan HER bar zaten
# `predictions_log`a yönlü bir satır yazıyor; özellik vektörünü de yazmak havuzu
# sembol×bar ölçeğine çıkarır ve satırlar `(symbol, bar_ts)` ile ETİKETLENEBİLİR kalır.
# Kapatmak için FEATURE_LOG_ALL=false. Aday-DIŞI satırlarda `is_long` sinyalden gelir
# (WAIT → 0) ve bu bir ETİKET değil BAĞLAMDIR; ayırt etmek için özellik logu `cand_side`
# kolonu taşır (boş = aday değil) → eski davranışın satırları `cand_side != ""` ile
# BİREBİR geri alınabilir.
_FEATURE_LOG_ALL = (os.getenv("FEATURE_LOG_ALL", "true").strip().lower()
                    not in ("0", "false", "off", "no"))


def _candidate_side(sig) -> str | None:
    """Aday İŞLEM tarafı ('LONG'/'SHORT') — yoksa None (aday değil).

    TEK kaynak: hem loglanacak p_win'in tarafı (_signal_pwin), hem özellik vektörünün
    `is_long`'u, hem nöral gölge çıkarımı bunu kullanır. Ayrı ayrı türetilirse
    (eskiden öyleydi) enforce kipinde sessizce ayrışırlar."""
    md = getattr(sig, "metadata", None) or {}
    if sig.action in (SignalAction.LONG, SignalAction.SHORT):
        return str(sig.action.value)
    if md.get("meta_p_win") is not None:
        side = md.get("meta_gated")     # enforce kipinde elenen yön
        return str(side) if side else None
    return None


def _neural_shadow_pwin(feats: dict, side: str) -> dict | None:
    """Nöral GÖLGE p_win'i ({'pwin','std','side'}) — pasif/hatalıysa None.

    Geç import: torch yükü yalnız gerçekten aday varken (ve PREDICT_TRACK açıkken)
    ödenir. KARAR-DIŞI — dönen değer hiçbir sinyal/kapı/boyut kararına girmez."""
    try:
        from nihai.execution.neural_shadow import get_shadow
        res = get_shadow().pwin(feats, side)
    except Exception:
        return None
    if res is None:
        return None
    return {"pwin": res[0], "std": res[1], "side": side}


def _ai_verdict(symbol: str, df, sig, side: str) -> dict | None:
    """İç AI'ın canlı aday hakkındaki verdikti ({'action','post','veto'}) — yoksa None.

    `ai_advisor.shadow_verdict` KİP-BAĞIMSIZDIR: `AI_LIVE_MODE=off` (varsayılan) iken de
    ölçülür, çünkü "kapı açılmalı mı?" sorusunun tek dürüst cevabı kapı KAPALIYKEN
    biriken kayıttır (META_GATE shadow dersi). Karar yolundaki `advise()` ile AYNI
    çekirdeği çağırır → loglanan verdikt, uygulanacak verdiktin ta kendisidir.
    Geç import: `nihai.ai` zinciri yalnız PREDICT_TRACK açıkken ve aday varken ödenir.
    KARAR-DIŞI — değer `_ai` altında disk-loga gider, kayıttan sonra satırdan düşürülür."""
    try:
        from nihai.execution.ai_advisor import shadow_verdict
        return shadow_verdict(symbol, df, sig, side=side)
    except Exception:
        return None


def _bar_observatory(df, symbol: str) -> dict | None:
    """Gölge ÖLÇÜM paketi (mikroyapı + rejim-HMM + veri sağlığı) — pasif/hatalıysa None.

    Geç import: `nihai.ai` zinciri (scipy/pandas) yalnız PREDICT_TRACK açıkken ödenir.
    KARAR-DIŞI — dönen hiçbir değer sinyale/kapıya/boyuta/rejim çarpanına girmez; yalnız
    `_obs` altında disk-loga taşınır ve kayıttan sonra satırdan düşürülür (`_neural`
    deseninin birebir aynısı). `_neural`'dan farkı: ADAY sinyal şartı YOKTUR — sağlık,
    makas ve rejim adaya değil SEMBOLE aittir; yalnız adaylarda ölçmek gölge kolonu
    seçilim yanlı yapardı (canlı kanıt döngüsünün `_signal_pwin`'de kırdığı yanlılığın
    tıpkısı)."""
    try:
        from nihai.ai.observatory import bar_shadow
        return bar_shadow(df, symbol)
    except Exception:
        return None


def _signal_pwin(sig, gate, df) -> tuple[float | None, str | None]:
    """SAF: satıra yazılacak (pwin, pwin_side) — meta modelin CANLI kanıt döngüsü.

    - LONG/SHORT sinyal: p = metadata['meta_p_win'] (kapı zaten hesapladıysa aynı değer,
      ikinci model çağrısı yok) yoksa gate.probability. Taraf = sinyal yönü.
    - Kapı-engelli WAIT (metadata'da meta_p_win var): p + elenen taraf (meta_gated) yine
      raporlanır → eşik-ALTI adaylar da loglanır. Bu olmadan META_GATE açıkken canlı log
      yalnız eşik-üstünü görür (seçilim yanlılığı) ve kapı canlıda hiç değerlendirilemez
      (bugünkü log: 11.8k çözülmüş tahmin, pwin'li satır 0).
    - Sıradan WAIT (giriş kurulumu yok): (None, None) — model dejenere, yanıltma.
    """
    md = getattr(sig, "metadata", None) or {}
    side = _candidate_side(sig)
    if sig.action in (SignalAction.LONG, SignalAction.SHORT):
        p = md.get("meta_p_win")
        if p is None:
            try:
                p = gate.probability(df, sig)
            except Exception:
                p = None
        return (None if p is None else float(p)), side
    p = md.get("meta_p_win")
    if p is not None:
        return float(p), side
    return None, None


def _symbols() -> list[str]:
    """PREDICT_SYMBOLS → STREAM_SYMBOLS → bench.BASKET sırasıyla sembol listesi."""
    for env_key in ("PREDICT_SYMBOLS", "STREAM_SYMBOLS"):
        raw = [s.strip().upper() for s in os.getenv(env_key, "").split(",") if s.strip()]
        if raw:
            return raw[:30]
    from nihai.universe import BASKET
    return BASKET


def _find_stream_status(provider) -> dict | None:
    """Tur-58: gezinti `data_provider.stream_status`a taşındı (tek kaynak).

    Ad burada KORUNDU — testler ve `snapshot()` bu adı çağırıyor; ikinci bir gezinti
    yazmak Tur-35'in "aynı desen birden çok kopyada" kusurunu doğururdu."""
    from nihai.data.data_provider import stream_status
    return stream_status(provider)


class PredictionService:
    """Arka plan thread'inde sürekli tahmin üreten singleton servis."""

    def __init__(self) -> None:
        self.interval = cast("Interval", os.getenv("PREDICT_INTERVAL") or HTF_INTERVAL)
        self.refresh = max(5.0, float(os.getenv("PREDICT_REFRESH_SEC", "20")))
        # Quote sağlık eşikleri: geniş spread / son işlemden büyük sapma → quote reddet.
        self._max_spread = float(os.getenv("PREDICT_QUOTE_MAX_SPREAD", "0.005"))  # %0.5
        self._max_dev = float(os.getenv("PREDICT_QUOTE_MAX_DEV", "0.02"))         # %2
        # #3 çok-zaman-dilimli onay (üst interval) ve #2 kalibrasyon.
        self._htf_interval = cast("Interval", os.getenv("PREDICT_HTF_INTERVAL", "15m"))
        # Döngü BAŞINDA tazelenir (_run). Döngü İÇİNDE sabit tutulur ki aynı turdaki 30
        # sembol AYNI eğriyi kullansın (yarı-tazelenmiş tur = kıyaslanamaz satırlar).
        self._calib = _load_calibration(self.interval)
        # Tur-84: yayınlanan aralığın kuantilleri — eğriyle aynı yaşam döngüsü (döngü başında tazelenir).
        self._band = _load_band(self.interval)
        # Tur-84: sembol → (ET işlem günü, seans açılışı | None, çekim anı). Açılış gün içinde SABİTTİR
        # → sembol başına günde bir 1d çekimi; başarısız çekim kısa süre (`_ACILIS_NEG_TTL`) bekletilir.
        self._acilis_memo: dict[str, tuple[date, float | None, float]] = {}
        # #4 hesaplama: ağır işi (generate_signal + HTF + gösterge) bar başına bir kez yap.
        self._sig_cache: dict[str, dict] = {}   # symbol -> {ts, sig, e20..adx, htf_dir, day_open}
        # Döngü-başı bağlam (tüm sembollerde ortak): piyasa-geneli rejim + açık/kapalı.
        self._cycle_regime: str | None = None
        self._cycle_market_open: bool | None = None
        self._bench = os.getenv("PREDICT_REGIME_BENCH", "SPY")
        # BULGU-59-7 — piyasa BAĞLAMI (genişlik / göreceli güç / çapraz varlık).
        # `_cycle_ctx` tur BAŞINDA bir ÖNCEKİ turun snapshot'ıyla doldurulur ve tur boyunca
        # SABİT kalır → aynı turdaki her sembol aynı bağlamı görür ve değer karar barından
        # kesinlikle öncedir (`_load_calibration` ile aynı sözleşme, Tur-28).
        self._mkt_ctx = MarketContext()
        self._cycle_ctx: dict = dict.fromkeys(CONTEXT_FIELDS)
        self._cycle_bench_close = None
        # Çapraz varlık vekilleri PAHALI DEĞİL ama BEDAVA da değil (3 ek sembol/tur).
        # TTL ile seyreltilir; ölçülemezse alanlar None kalır.
        self._xa_ttl = float(os.getenv("CROSS_ASSET_TTL_SEC", "900"))
        self._xa_at = 0.0
        self._xa: dict = {}
        # #3 — kapalı-döngü doğrulama: her bar'ın forecast'ini diske kaydet (opt-in).
        self._track = os.getenv("PREDICT_TRACK", "").lower() in ("1", "true", "yes", "on")
        # Tur-20 — yön/güveni CANLI fiyat yerine KARAR BARININ KAPANIŞINDAN hesapla. Tur-20
        # bunu kanıt birikene kadar KAPALI tuttu ve iki varyantı da logladı; Tur-83 biriken
        # 43.195 satırı ölçüp varsayılanı AÇTI (gerekçe ve sayılar: `_yon_kipi`). İki varyant
        # yine her barda hesaplanır ve ikisi de loglanır (`dir_bar`/`conf_bar` ↔
        # `dir_live`/`conf_live`) → karar canlı veriyle yeniden sınanabilir kalır.
        self._signal_on_bar_close = _yon_kipi() == "bar"
        self._provider: DataProvider | None = None
        self._gate: _MetaGate | None = None
        self._lock = threading.RLock()
        self._rows: list[dict] = []
        self._meta: dict = {"updated": None, "source": None, "interval": self.interval,
                            "refresh_sec": self.refresh, "meta_model": None, "symbols": 0,
                            "actionable": 0, "ups": 0, "downs": 0, "avg_confidence": 0.0,
                            "avg_prob_cal": None,
                            "cycle_ms": 0, "market_open": None, "regime": None,
                            "calibrated": False, "band_coverage": None, "htf_interval": None,
                            "stream": None, "error": None}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ----------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Servisi başlat (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="predict-svc", daemon=True)
        self._thread.start()
        logger.info(f"[predict] tahmin servisi başladı (interval={self.interval}, "
                    f"yenileme={self.refresh:.0f}s)")

    def stop(self) -> None:
        self._stop.set()

    # ----------------------------------------------------------------- compute
    def _gun_acilisi(self, symbol: str, bar_ts) -> float | None:
        """Karar barının ET işlem gününün seans açılışı (bkz. `_seans_acilisi`), gün başına bir çekim.

        Başarısız çekim (seans öncesi: bugünün günlük barı henüz yok) en fazla `_ACILIS_NEG_TTL`
        saniyede bir yeniden denenir — her yeni seans-öncesi barda 30 sembolün 1d çekimi hub'ın
        Alpaca bütçesine binerdi; gün boyu önbelleklemek ise 09:30'dan sonra da "—" bırakırdı."""
        try:
            gun = _et_gunu(bar_ts)
        except Exception:
            return None
        hit = self._acilis_memo.get(symbol)
        if hit and hit[0] == gun and (hit[1] is not None
                                      or time.monotonic() - hit[2] < _ACILIS_NEG_TTL):
            return hit[1]
        o = _seans_acilisi(self._provider, symbol, gun)
        self._acilis_memo[symbol] = (gun, o, time.monotonic())
        return o

    def _bar_context(self, symbol: str, df) -> dict:
        """#4 — bar başına BİR kez hesaplanan ağır bağlam (generate_signal + HTF + gösterge).
        last_bar_ts değişene kadar cache'ten servis edilir; her döngüde yeniden hesaplanmaz."""
        assert self._provider is not None   # _run() bunu set ettikten sonra çağrılır
        ts = str(df.index[-1])
        hit = self._sig_cache.get(symbol)
        if hit and hit.get("ts") == ts:
            return hit
        close = df["close"]
        # HTF (üst zaman dilimi) trend yönü — #3. Ağ çağrısı; bar başına bir kez.
        htf_dir = 0
        try:
            htf_df = self._provider.get_bars(symbol, self._htf_interval, lookback_bars=80)
            htf_dir = _htf_direction(htf_df)
        except Exception:
            htf_dir = 0
        day_open = self._gun_acilisi(symbol, df.index[-1])
        # Tur-43: EMA20/EMA50/RSI14/ATR14/ADX14 bu blokta ÜÇ KEZ hesaplanıyordu —
        # (1) generate_signal kendi içinde `_SharedIndicators` kurup ATIYORDU,
        # (2) aşağıdaki ctx satırları sıfırdan, (3) extract_features bir kez daha.
        # Tur-7 önbelleği bar başına BİR kez kurulur ve üç tüketiciye de o geçer;
        # `_SharedIndicators` aynı fonksiyonları aynı argümanlarla çağırır → bit-denk.
        shared = _SharedIndicators(df)
        sig = generate_signal(symbol, df, shared=shared, live_interval=self.interval)
        adx_series = shared.adx_series()
        ctx = {
            "ts": ts,
            "sig": sig,
            "last_close": float(close.iloc[-1]),
            "e20": _last_valid(shared.ema(20)), "e50": _last_valid(shared.ema(50)),
            "r": _last_valid(shared.rsi14()), "a": _last_valid(shared.atr14()),
            "adx_v": _last_valid(adx_series), "htf_dir": htf_dir, "day_open": day_open,
        }
        # Eğitim-verisi birikimi: track açıkken ADAY sinyallerin (giriş VEYA kapı-engelli)
        # karar-barı özellik vektörü de loglanır → canlı, survivorship-temiz meta-dataset
        # (ACI DERS #5: darboğaz veri; bu döngü onu her gün büyütür). Bar başına bir kez.
        if self._track:
            # Gölge ölçüm (Tur-37) — bar başına BİR kez, ADAY şartı YOK (bkz.
            # _bar_observatory docstring'i: sağlık/makas/rejim sembole aittir).
            ctx["obs"] = _bar_observatory(df, symbol)
            # BULGU-59-7: piyasa GENİŞLİĞİ gözlemi — bar başına bir kez, aday şartı YOK
            # (genişlik sembole değil PİYASAYA aittir; `_bar_observatory` ile aynı gerekçe).
            # `pool.map` 8 thread'den çağırır; tek `dict[str]=...` ataması GIL altında
            # atomiktir ve her sembol KENDİ anahtarını yazar (yarış yok).
            self._mkt_ctx.observe(symbol, df["close"])
            side = _candidate_side(sig)
            # BULGU-59-1: özellik çıkarımı ADAY şartından AYRILDI (bkz. _FEATURE_LOG_ALL).
            # Gölge çıkarım (nöral/AI) aday şartında KALIR — onlar model çağrısıdır,
            # bar başına her sembol için ödenmemelidir; özellik çıkarımı ise zaten
            # kurulmuş `shared` önbelleğinden okur (Tur-43: sembol-başı blok 2,97×).
            if side is not None or _FEATURE_LOG_ALL:
                try:
                    # Tip notu: `extract_features` dict[str, float] döndürür; aşağıda
                    # '_' önekli META alanlar (cand_side/ctx) eklenir. Bunlar MODEL
                    # GİRDİSİ DEĞİLDİR (`features_to_vector` FEATURE_NAMES'e göre filtreler),
                    # yalnız `_write_features` okur → kap tipi genişletilir.
                    feats: dict[str, Any] = dict(extract_features(df, sig, shared=shared))
                    # enforce kipinde kapı sinyali WAIT'e çevirir → extract_features
                    # `is_long`i 0 yazar (aday LONG olsa bile). Aday tarafını geri koy:
                    # yanlış is_long hem canlı eğitim setini (build_live_dataset) hem
                    # gölge çıkarımı sessizce bozardı. Shadow/off kipinde no-op.
                    # Aday YOKSA dokunulmaz: orada `is_long` bir işlem tarafı değil,
                    # "giriş kurulumu yok" bağlamıdır (`cand_side` boş kalır).
                    if side is not None:
                        feats["is_long"] = 1.0 if side == "LONG" else 0.0
                    # Aday tarafı özellik logunun kolonuna taşınır ('_' önekli anahtar
                    # `features_to_vector`/FEATURE_NAMES filtresinden geçmez → model
                    # girdisi DEĞİŞMEZ; yalnız _write_features okur).
                    feats["_cand_side"] = side or ""
                    # BULGU-59-7 — PİYASA BAĞLAMI ek kolonları. `_` önekli olduğu için
                    # FEATURE_NAMES filtresinden geçmez (model girdisi DEĞİŞMEZ); yalnız
                    # `_write_features` okur. Değerler bir ÖNCEKİ turun snapshot'ından
                    # gelir → karar barından kesinlikle önceki bilgi (look-ahead yok).
                    feats["_ctx"] = self._context_cells(df)
                    ctx["feats"] = feats
                except Exception:
                    ctx["feats"] = None
            # Tur-90 (BULGU-90-7): gölge çıkarım yalnız piyasa AÇIKKEN. `_ai_verdict` kaydı defterden
            # BAĞIMSIZ yazar (`decision._record_shadow` → ai_decisions.jsonl + epizot): kapalı piyasada
            # aynı bayat bar, hub↔yfinance her geçişte yeniden karar oluyordu — 15.038 kararın %27'si,
            # 4.229 anının %20'si seans dışıydı. Tahmin defteri zaten yalnız açıkken yazar (record()).
            if side is not None and self._cycle_market_open is not False:
                # Nöral GÖLGE p_win (Tur-18) — bar başına BİR kez, aday varsa.
                cached_feats = ctx.get("feats")
                if isinstance(cached_feats, dict) and cached_feats:
                    ctx["neural"] = _neural_shadow_pwin(cached_feats, side)
                # İç AI verdikti (Tur-38B) — aday başına BİR kez. Kip-bağımsız ölçüm:
                # `enforce` açılsaydı bu sinyal susar mıydı? sorusunun 0/1 kaydı.
                ctx["ai"] = _ai_verdict(symbol, df, sig, side)
        # Tur-102: gölge kasalar — 5 gölge/danışman mekanizmanın her biri kendi sanal kasasında karar verir.
        # KARAR-DIŞI (ctx'e/tahmine dokunmaz), bar başına bir kez, yalnız seans AÇIK bilindiğinde (BULGU-90-7).
        # `conf_bar` deftere yazılanla AYNI sayıdır (`_yon_guven` tek kaynak): köprü kasasının dondurulmuş kuralı onu okur.
        if self._cycle_market_open is True:
            from nihai.execution import golge_kasalar
            golge_kasalar.isle(symbol, df, shared,
                               conf_bar=self._yon_guven(ctx, float(close.iloc[-1]))[1])
        self._sig_cache[symbol] = ctx
        return ctx

    def _yon_guven(self, ctx: dict, px: float) -> tuple[str, float]:
        """Yön + güven (HTF + rejim dahil) — `_row`un iki varyantı ve gölge kasa TEK kaynaktan okur."""
        d, _b, cb = _bias_and_confidence(
            px, ctx["e20"], ctx["e50"], ctx["r"], ctx["adx_v"],
            htf_dir=ctx["htf_dir"], regime_tilt=1.0)
        return d, float(np.clip(cb * _regime_tilt(self._cycle_regime, d), 0.0, 1.0))

    def _bench_series(self):
        """Benchmark kapanışları TAHMİN ARALIĞINDA (göreceli güç için). Hata → None.

        `compute_regime`in 1d serisi BURADA KULLANILAMAZ: 20 barlık getiri farkı iki tarafta
        AYNI aralığı ister (1d'lik 20 bar ≠ 5m'lik 20 bar). Sağlayıcı önbelleği turda tek
        çağrıya indirir."""
        try:
            assert self._provider is not None       # _run() set ettikten sonra çağrılır
            b = self._provider.get_bars(self._bench, self.interval, lookback_bars=60)
        except Exception:
            return None
        if b is None or getattr(b, "empty", True) or "close" not in b.columns:
            return None
        return b["close"]

    def _cross_asset(self) -> dict:
        """Çapraz varlık vekillerinin 20-bar getirileri (TTL'li). Ölçülemeyen → None."""
        now = time.monotonic()
        if self._xa and (now - self._xa_at) < self._xa_ttl:
            return self._xa
        from nihai.signals.breadth import rel_return
        out: dict = {}
        for ad, sym in CROSS_ASSET_PROXIES.items():
            val = None
            try:
                assert self._provider is not None   # _run() set ettikten sonra çağrılır
                d = self._provider.get_bars(sym, self.interval, lookback_bars=60)
                if d is not None and not getattr(d, "empty", True) and "close" in d.columns:
                    val = rel_return(d["close"])
            except Exception:
                val = None
            out[f"xa_{ad}_20"] = val
        self._xa, self._xa_at = out, now
        return out

    def _context_cells(self, df) -> dict:
        """Bu satıra yazılacak piyasa-bağlamı hücreleri (BULGU-59-7).

        Genişlik + McClellan: bir önceki turun tamamlanmış ölçümü (tur içinde SABİT).
        `rs_bench_20`: sembol ile benchmark'ın AYNI aralıktaki 20-bar getiri farkı —
        benchmark serisi yoksa None (uydurma yok)."""
        cells = dict(self._cycle_ctx)
        try:
            cells["rs_bench_20"] = relative_strength(df["close"], self._cycle_bench_close)
        except Exception:
            cells["rs_bench_20"] = None
        return cells

    def _row(self, symbol: str) -> dict:
        assert self._provider is not None and self._gate is not None  # _run() set eder
        df = self._provider.get_bars(symbol, self.interval, lookback_bars=220)
        if df is None or df.empty or len(df) < 60:
            return {"symbol": symbol, "action": "NODATA", "direction": "•",
                    "price": 0.0, "chg_pct": None, "stop": 0.0, "target": 0.0, "rr": 0.0,
                    "quality": 0.0, "pwin": None, "pwin_side": None, "confidence": 0.0,
                    "prob_cal": None, "exp_move_pct": None, "band_lo": None, "band_hi": None,
                    "why": "veri yok"}

        ctx = self._bar_context(symbol, df)
        sig = ctx["sig"]
        last_close = ctx["last_close"]

        # Canlı fiyat — quote orta noktası, SAĞLIK KONTROLÜ ile (geniş spread / büyük
        # sapma → reddet, son gerçek işlem fiyatına düş). Piyasa kapalıysa quote sorma.
        live_price = bid = ask = 0.0
        if self._cycle_market_open is not False:
            try:
                q = self._provider.get_quote(symbol)
                if q:
                    live_price = float(getattr(q, "last", 0) or 0)
                    bid = float(getattr(q, "bid", 0) or 0)
                    ask = float(getattr(q, "ask", 0) or 0)
            except Exception:
                live_price = bid = ask = 0.0

        price = float(sig.price or last_close)
        if live_price > 0:
            spread = (ask - bid) / live_price if (bid > 0 and ask > 0) else 1.0
            dev = abs(live_price - last_close) / last_close if last_close > 0 else 1.0
            if spread <= self._max_spread and dev <= self._max_dev:
                price = live_price

        day_open = ctx["day_open"]
        chg_pct = round((price - day_open) / day_open * 100.0, 2) if day_open else None

        # Yön + güven (cache'li gösterge + HTF + rejim) — #3. İki varyant: KARAR BARININ
        # KAPANIŞIYLA (Tur-83'ten beri yayınlanan) ve CANLI kotasyonla (eski varsayılan).
        # İkisi de her barda hesaplanır ve ikisi de deftere yazılır; hangisinin
        # YAYINLANACAĞI `_yon_kipi` kararıdır (ölçüm ve gerekçe orada).
        dir_live, conf_live = self._yon_guven(ctx, price)
        bar_px = float(last_close or 0.0)
        # Bar kapanışı yoksa/bozuksa gölge varyant hesaplanmaz (uydurma yok).
        dir_bar, conf_bar = self._yon_guven(ctx, bar_px) if bar_px > 0 else ("", 0.0)

        if self._signal_on_bar_close and bar_px > 0:
            direction, confidence = dir_bar, conf_bar
        else:
            direction, confidence = dir_live, conf_live
        # #2 — Tur-83: eğri CANLI defterde YAYINLANAN güven üzerinde kurulur (HTF/rejim DAHİL;
        # defterde tam da o sayı var) → girdi yayınlanan güvendir. Alanı eşleşmeyen eğri
        # `_load_calibration`da reddedilir; burada kırpılıp uydurulmaz.
        prob_cal = _apply_calibration(self._calib, confidence)

        # Tur-84 (BULGU-84-2): "Öngörü" eskiden `price ± ATR`di (yön işaretli nokta tahmini). Yön
        # yazı-tura olduğu için bu nokta "fiyat değişmez"den ÖLÇÜLEREK kötüydü (93.464 satır: MAE
        # 24,47 ↔ 21,29 bps, gün-kümeli t=+8,7; yürüyen en iyi yön ofseti ≈ 0). Yerini kalibre bir
        # ARALIK aldı (ölçek ATR14, kuantiller gece görevinden); aralık yoksa uydurulmaz → "—".
        # ÇAPA = son BAR KAPANIŞI, kotasyon değil (BULGU-84-4): aynı 83.147 satırda aralık skoru
        # 74,4 ↔ 96,9 bps (eşleştirilmiş t=−10,7; 32 günün 30'unda) — kotasyon şu anki fiyatın
        # gürültülü bir tahmini (bayat son işlem, makas; sapması 5 dk'da geri döner — Tur-83).
        a = ctx["a"]
        atr_pct = (a / bar_px * 100.0) if (bar_px > 0 and np.isfinite(a) and a > 0) else None
        # Kuantiller AÇIK seansın 5 dakikalık hareketinden kuruldu (defter yalnız piyasa açıkken
        # yazar); kapalı piyasada "5 dk sonraki fiyatın aralığı" kalibrasyon alanının dışıdır → "—".
        bant = (_apply_band(self._band, symbol, atr_pct)
                if self._cycle_market_open is not False else None)
        band_lo = round(bar_px * (1.0 + bant[0] / 100.0), 2) if bant else None
        band_hi = round(bar_px * (1.0 + bant[1] / 100.0), 2) if bant else None
        exp_move = round(bant[2], 2) if bant else None

        # P(win): giriş sinyali VEYA kapı-engelli aday (meta_p_win metadata'da) için;
        # sıradan WAIT'te None (model dejenere → ≈%100, yanıltır). Kapı-engelli tarafın
        # da loglanması canlı kanıt döngüsünün seçilim yanlılığını kırar (_signal_pwin).
        pwin, pwin_side = _signal_pwin(sig, self._gate, df)
        # '_feats' yalnız kayıt anına taşınır; record() sonrası satırdan düşürülür
        # (web JSON'una sızmaz). BULGU-59-1'e kadar buradaki şart `pwin_side is not None`
        # idi ve etiketsiz havuzu 8× küçültüyordu: satırın ETİKETİ pwin'den değil
        # `predictions_log`un `hit`/`fwd_ret_pct` kolonundan gelir ve o KAYIT ZATEN
        # YAZILIYOR (`_is_directional`). Yani "etiket yok" gerekçesi yanlıştı — eksik olan
        # yalnız kapı olasılığıydı, etiketin kendisi değil.
        feats = (ctx.get("feats")
                 if (self._track and (pwin_side is not None or _FEATURE_LOG_ALL))
                 else None)
        # Nöral gölge: özellikle AYNI koşula bağlı (aday + track). '_neural' de yalnız
        # disk-loga gider, kayıttan sonra satırdan düşürülür → web JSON'una sızmaz.
        neural = ctx.get("neural") if feats else None

        obs = ctx.get("obs") if self._track else None
        # İç AI verdikti: '_neural' ile AYNI sözleşme (yalnız disk-log, sonra düşürülür).
        ai_v = ctx.get("ai") if self._track else None

        return {
            **({"_feats": feats} if feats else {}),
            **({"_neural": neural} if neural else {}),
            # Gölge ölçüm paketi — '_neural' ile aynı sözleşme: yalnız disk-loga gider,
            # record() sonrası satırdan DÜŞÜRÜLÜR (web JSON'una/pano tablosuna girmez).
            **({"_obs": obs} if obs else {}),
            **({"_ai": ai_v} if ai_v else {}),
            # Gölge yön hipotezi: yalnız disk-loga (dir_bar/conf_bar), kayıttan sonra
            # satırdan DÜŞÜRÜLÜR. Panoda ikinci bir "yön" sütunu göstermek onu sinyal
            # sanma riski taşır — nöral p_win köprüsüyle aynı gerekçe.
            **({"_dirbar": {"direction": dir_bar, "confidence": round(conf_bar, 4)}}
               if dir_bar else {}),
            # Tur-83: KARŞI varyant da loglanır (canlı kotasyon tabanlı) — bar kipi yayındayken
            # A/B aynı satırlarda sürer; `_dirbar` ile aynı sözleşme (kayıttan sonra düşer).
            "_dirlive": {"direction": dir_live, "confidence": round(conf_live, 4)},
            "symbol": symbol,
            "action": sig.action.value,
            "direction": direction,
            "bar_ts": ctx["ts"],   # #3 kapalı-döngü kaydı için (bar başına tek tahmin)
            "confidence": round(confidence, 2),
            "prob_cal": (round(prob_cal, 3) if prob_cal is not None else None),
            # Tipik |hareket| (ufuk sonunda, %; YÖNSÜZ) + ~%80 aralık. Aralığın ölçeği defterin
            # `atr_pct` kolonuna yazılır → kuantilleri gece görevi oradan kurar ve kapıdan geçirir.
            "exp_move_pct": exp_move,
            "band_lo": band_lo,
            "band_hi": band_hi,
            "atr_pct": (round(atr_pct, 6) if atr_pct is not None else None),
            "price": price,
            # Karar barının KAPANIŞI — `price` (sinyal anındaki canlı fiyat) ile arasındaki
            # fark kovalama/gecikme maliyetidir. prediction_tracker bunu `slip_bps` +
            # `hit_bar` olarak ayrıştırır; yön becerisi ile uygulama maliyeti karışmasın.
            "bar_close": float(last_close or 0.0),
            "chg_pct": chg_pct,
            "stop": float(sig.stop or 0.0),
            "target": float(sig.target or 0.0),
            "rr": float(sig.rr_ratio or 0.0),
            "quality": float(sig.quality_score or 0.0),
            "pwin": pwin,
            "pwin_side": pwin_side,
            "meta_model_id": (sig.metadata.get("meta_model_id") or getattr(self._gate, "model_id", None)) if pwin is not None else None,
            "meta_strategy": sig.metadata.get("meta_strategy") if pwin is not None else None,
            "why": (sig.reasons[0] if sig.reasons else ""),
        }

    _ORDER: ClassVar[dict[str, int]] = {"LONG": 0, "SHORT": 1, "WAIT": 2, "NODATA": 3}

    def _run(self) -> None:
        try:
            # Sekmeye ÖZEL kısa önbellek (global DATA_CACHE_TTL_SEC'i ve ana taramayı
            # bozmadan canlı tut). Hub yerel olduğundan sık okuma maliyetsiz.
            base = get_provider(with_cache=False)
            pttl = float(os.getenv("PREDICT_CACHE_TTL_SEC", "8"))
            self._provider = CachingProvider(base, pttl) if pttl > 0 else base
            self._gate = get_gate()
        except Exception as e:
            logger.error(f"[predict] başlatma hatası: {e}")
            with self._lock:
                self._meta["error"] = str(e)
            return

        src = getattr(self._provider, "name", "?")
        meta_kind = self._gate.kind if self._gate.active else None
        workers = max(1, int(os.getenv("PREDICT_WORKERS", "8")))
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="predict-w")

        def _safe_row(s: str) -> dict:
            try:
                return self._row(s)
            except Exception as e:
                logger.debug(f"[predict] {s}: {e}")
                return {"symbol": s, "action": "ERROR", "direction": "•",
                        "price": 0.0, "chg_pct": None, "stop": 0.0, "target": 0.0,
                        "rr": 0.0, "quality": 0.0, "pwin": None, "pwin_side": None,
                        "confidence": 0.0, "prob_cal": None, "exp_move_pct": None,
                        "band_lo": None, "band_hi": None, "why": str(e)[:40]}

        try:
            while not self._stop.is_set():
                from nihai.signals.model_promotion import policy_revision
                gate = get_gate()
                revision = policy_revision()
                if gate is not self._gate or revision != getattr(self, "_policy_revision", None):
                    self._gate = gate
                    self._policy_revision = revision
                    self._sig_cache.clear()
                    meta_kind = gate.kind if gate.active else None
                symbols = _symbols()
                t0 = time.monotonic()
                # Kalibrasyon eğrisi diskte tazelendiyse bu turdan itibaren geçerli olsun
                # (Tur-28: 7/24 süreçte bayat eğri). Değişmediyse maliyet tek stat().
                self._calib = _load_calibration(self.interval)
                self._band = _load_band(self.interval)          # Tur-84: aynı tazelik sözleşmesi
                # Döngü-başı ortak bağlam: piyasa açık/kapalı + piyasa-geneli rejim (#3).
                try:
                    self._cycle_market_open = bool(is_market_open())
                except Exception:
                    self._cycle_market_open = None
                try:
                    bench_df = self._provider.get_bars(self._bench, "1d", lookback_bars=220)
                    self._cycle_regime = compute_regime(bench_df).regime
                except Exception:
                    self._cycle_regime = None
                # BULGU-59-7: tur bağlamını SABİTLE (bir önceki turun tamamlanmış ölçümü
                # + TTL'li çapraz varlık). `_track` kapalıyken hiç ödenmez.
                if self._track:
                    self._cycle_ctx = self._mkt_ctx.snapshot()
                    self._cycle_ctx.update(self._cross_asset())
                    self._cycle_bench_close = self._bench_series()
                # PARALEL: 30 sembolü eşzamanlı hesapla (sıralı ~30×gecikme yerine ~1×).
                rows: list[dict] = list(pool.map(_safe_row, symbols))
                # BULGU-59-7: turu KAPAT — yalnız bu turda gerçekten yeni gözlem düştüyse
                # (gösterge önbelleği yüzünden bu bar başına BİR kez olur; her ~20 sn'lik
                # turda ilerletmek McClellan geçmişini sahte biçimde düzleştirirdi).
                if self._track and self._mkt_ctx.pending():
                    self._mkt_ctx.roll()
                cycle_ms = int((time.monotonic() - t0) * 1000)

                # #3 — kapalı-döngü kaydı: yönlü tahminleri bar başına bir kez diske yaz.
                # Sadece piyasa açıkken (kapalıyken bar değişmez → dedup zaten engeller).
                if self._track and self._cycle_market_open is not False:
                    try:
                        from nihai.execution.prediction_tracker import record
                        record(rows, self.interval)
                    except Exception as e:
                        logger.debug(f"[predict] track kaydı atlandı: {e}")
                for r in rows:
                    r.pop("_feats", None)   # yalnız disk-log içindi; web/state'e taşınmaz
                    r.pop("_neural", None)  # aynısı gölge p_win için (ANAYASA: karar yüzeyi değil)
                    r.pop("_dirbar", None)  # aynısı gölge yön hipotezi için (Tur-20)
                    r.pop("_dirlive", None)  # aynısı karşı (canlı) yön varyantı için (Tur-83)
                    r.pop("_obs", None)     # aynısı gölge ölçüm paketi için (Tur-37)
                    r.pop("_ai", None)      # aynısı iç AI verdikti için (Tur-38B)

                # Sırala: aksiyon (LONG/SHORT üstte) → P(win) → SEMBOL.
                # TUR-47: üçüncü anahtar `-confidence` İDİ ve bu bir İDDİADIR ("yüksek güven
                # önce bakılmayı hak eder"). Canlı defter tersini ölçtü: conf>=0.66 sınıfı
                # isabet %43,95 ↔ conf<0.40 sınıfı %50,94 (n=63.237, z=+14,4; hem zaman hem
                # sembol bölmesinde). Anahtar TERSİNE ÇEVRİLMEDİ (ACI DERS #3/#7: aleyhte
                # kanıt kenar üretmez) — iddia KALDIRILDI. Sembol, sıralamayı deterministik
                # tutan ama hiçbir şey iddia etmeyen tek eşit-güçlü anahtardır (boş bırakmak
                # eşitlikte tabloyu her döngüde oynatırdı). SATIR KÜMESİ AYNI; yalnız
                # pwin'i eşit/None olan satırların kendi arasındaki sırası değişir.
                rows.sort(key=lambda r: (self._ORDER.get(r["action"], 9),
                                         -(r["pwin"] if r["pwin"] is not None else -1.0),
                                         r["symbol"]))
                actionable = sum(1 for r in rows if r["action"] in ("LONG", "SHORT"))
                ups = sum(1 for r in rows if r["direction"] == "+")
                downs = sum(1 for r in rows if r["direction"] == "-")
                confs = [r["confidence"] for r in rows if r["action"] not in ("NODATA", "ERROR")]
                avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0
                # Kalibre olasılık ortalaması = DÜRÜST başlık metriği (ham güven yanıltıcı).
                # ~0.50'ye yakınsa pano bunu nötr/gri gösterir = "öngörü gücü yok".
                cals = [r["prob_cal"] for r in rows if r.get("prob_cal") is not None]
                avg_prob_cal = round(sum(cals) / len(cals), 3) if cals else None
                market_open = self._cycle_market_open

                with self._lock:
                    self._rows = rows
                    self._meta = {
                        "updated": datetime.now().isoformat(timespec="seconds"),
                        "source": src,
                        "interval": self.interval,
                        "refresh_sec": self.refresh,
                        "meta_model": meta_kind,
                        # Tur-44: `meta_model` yalnız MODEL ADIdır; kapının fiilen
                        # üretip üretmediğini `meta_gate.durum` söyler (SUSUYOR =
                        # model yüklü ama sorulan her soruda hata → pano bunu görmeli).
                        "meta_gate": gate_health(),
                        "symbols": len(rows),
                        "actionable": actionable,
                        "ups": ups,
                        "downs": downs,
                        "avg_confidence": avg_conf,
                        "avg_prob_cal": avg_prob_cal,
                        "cycle_ms": cycle_ms,
                        "market_open": market_open,
                        "regime": self._cycle_regime,
                        "calibrated": self._calib is not None,
                        # Aralığın YÜRÜYEN (örneklem-dışı) kapsaması — pano başlığı nominal %80'i
                        # değil ölçülen sayıyı gösterir; aralık yoksa None.
                        "band_coverage": ((self._band or {}).get("meta", {})
                                          .get("yuruyen", {}).get("kapsama")),
                        "htf_interval": self._htf_interval,
                        "stream": _find_stream_status(self._provider),
                        "error": None,
                    }

                # Periyot ≈ refresh olsun: bekleme = refresh - geçen hesap süresi.
                elapsed = time.monotonic() - t0
                # Tur-58: aynı sayı `/metrics`e de gitsin — tahmin döngüsü gecikmesi
                # yalnız `snapshot()`ta duruyordu ve dışarıdan izlenemiyordu.
                with contextlib.suppress(Exception):
                    from nihai.observability import record_loop
                    record_loop("predict", elapsed)
                self._stop.wait(max(0.5, self.refresh - elapsed))
        finally:
            pool.shutdown(wait=False)

    # ----------------------------------------------------------------- read
    @property
    def running(self) -> bool:
        """Tahmin döngüsü gerçekten AYAKTA mı (thread canlı mı)."""
        return bool(self._thread and self._thread.is_alive())

    def snapshot(self) -> dict:
        """Anlık görüntü. `running` Tur-30'da eklendi: `/api/system/health` bunu
        `snap.get("running", False)` ile okuyordu ama böyle bir anahtar YOKTU → pano
        "Tahmin Servisi" kartı servis çalışırken bile HER ZAMAN 'durdu' diyordu.
        Servislerin sessizce ölmesi bu projenin bilinen arızası olduğu için, hep-yanlış
        bir sağlık kartı operatörü kartı görmezden gelmeye alıştırır."""
        alive = self.running          # kilit dışında: thread durumu kendi kilidini tutar
        with self._lock:
            return {"meta": dict(self._meta), "predictions": list(self._rows),
                    "running": alive}


_SERVICE: PredictionService | None = None


def get_service() -> PredictionService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = PredictionService()
    return _SERVICE
