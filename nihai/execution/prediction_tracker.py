"""
prediction_tracker.py — Canlı forecast'in KAPALI DÖNGÜ doğrulaması (#3).

Backtest survivorship'ten muzdarip; bu modül gerçek-zamanlı, ileriye-bakmayan kanıt
üretir: her bar'da forecast'i (yön + güven + kalibre olasılık) kaydeder, `horizon`
bar sonra gerçekleşen yönü ölçer, isabet/getiri ile etiketler. Rapor → canlı
kalibrasyon eğrisi + gerçekleşen beklenti.

DÜRÜST ÇERÇEVE: bu, offline `forecast_calibration.json`'un canlı sınamasıdır. Offline
ölçüm yön tahminini ~%50 (yazı-tura) buldu; bu döngü onu canlıda DOĞRULAR ya da ÇÜRÜTÜR.
Hiçbir emir göndermez — saf tahmin-vs-gerçekleşen muhasebesi (paper-trade eşdeğeri).

Kullanım:
    # canlı kayıt: predict servisinde PREDICT_TRACK=true (otomatik record())
    python -m nihai.execution.prediction_tracker --resolve   # vadesi gelenleri gerçekleşenle etiketle
    python -m nihai.execution.prediction_tracker --report    # canlı isabet + kalibrasyon özeti
    python -m nihai.execution.prediction_tracker --resolve --report
"""
from __future__ import annotations

import csv
import math
import os
import sys
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from nihai.atomic_io import atomic_replace as _atomic_replace
from nihai.atomic_io import tmp_path as _tmp_path
from nihai.config import DATA_DIR, INTERVAL_MINUTES
from nihai.signals.breadth import CONTEXT_FIELDS, COUNT_FIELDS
from nihai.signals.meta_features import FEATURE_NAMES

PRED_LOG = DATA_DIR / "predictions_log.csv"
FIELDS = [
    "logged_at", "symbol", "interval", "bar_ts", "direction", "action", "confidence",
    "prob_cal", "pwin", "pwin_side", "price", "bar_close", "horizon_bars",
    "resolved_at", "fwd_price", "hit", "fwd_ret_pct", "hit_bar", "slip_bps",
    "neural_pwin", "neural_pwin_side", "neural_pwin_std",
    "dir_bar", "conf_bar",
    "mkt_health", "har_rv_fc", "spread_bps", "hmm_mode_prob", "hmm_switch", "hmm_label",
    "poc_dist_bps", "value_area_pos", "cvd_divergence",
    "ai_action", "ai_post", "ai_veto", "ai_side",
    "meta_model_id", "meta_strategy",
    "dir_live", "conf_live",
    "atr_pct",
]
# atr_pct (Tur-84) — YAYINLANAN aralığın ÖLÇEĞİ: karar barındaki ATR14, o barın KAPANIŞININ
#   (`bar_close`, aralığın çapası — kotasyon değil, BULGU-84-4) %'si.
#   Pano "Öngörü" sütununda `fiyat ± ATR`yi yön renginde gösteriyordu; yön yazı-tura olduğu için bu
#   nokta tahmini "fiyat değişmez"den ÖLÇÜLEREK kötüydü (93.464 satır: MAE 24,47 ↔ 21,29 bps,
#   gün-kümeli t=+8,7). Yerini ölçek × kalibre z-kuantilleri olan bir ARALIK aldı
#   (`live_predict_service._apply_band`); kuantilleri gece görevi BU kolondan kurar
#   (`research/calibrate_forecast.py`). Kolon olmadan aralığın doğruluğu defterden ölçülemezdi
#   (Tur-37: öngörü ↔ gerçekleşen farkı ancak loglanırsa ölçülebilir). Eski satırlarda boş kalır —
#   ve bu bir özelliktir: kolonu taşıyan her satır zaman-tabanlı ufukla çözülür (`_forward_price`).
# dir_live / conf_live (Tur-83) — yönün CANLI KOTASYON varyantı, `dir_bar`ın AYNASI:
#   Tur-83 yayınlanan yönü bar kapanışına çevirdi (`live_predict_service._yon_kipi`). O andan
#   sonra `direction == dir_bar` olur ve eski varyant hiçbir kolonda kalmazdı → kararın A/B'si
#   AYNI SATIRLARDA sürdürülemezdi (bugünkü ölçümü mümkün kılan tam da ikisinin birlikte
#   loglanmasıydı). Kolon adı İÇERİĞİ söyler, kipi değil: `dir_bar` her zaman bar-kapanışı,
#   `dir_live` her zaman canlı-kotasyon varyantıdır. Eski satırlarda boş kalır — o satırlarda
#   `direction` zaten canlı varyanttı (varsayılan kapalıydı, .env de açmıyordu).
# ai_action / ai_post / ai_veto (Tur-38B) — İÇ AI VERDİKTİ, KARAR-DIŞI KAYIT:
#   Kaynak `execution/ai_advisor.shadow_verdict`; `_neural`/`_obs` ile BİREBİR aynı
#   sözleşme (satırda '_ai' altında taşınır, record() sonrası düşürülür, web JSON'una
#   ve pano tablosuna ASLA girmez).
#   NEDEN: Tur-38 iç AI'ı canlı karar yoluna BAĞLANABİLİR yaptı (`AI_LIVE_MODE`), ama
#   "kapı açılmalı mı?" sorusu ancak kapı KAPALIYKEN biriken kayıtla yanıtlanabilir
#   (META_GATE shadow dersi). Bu kolonlar olmadan verdikt yalnız metin loguna düşüyordu
#   ve çözülen getiriyle JOIN EDİLEMİYORDU → soru ölçülemezdi.
#     ai_action : "LONG_BIAS" | "SHORT_BIAS" | "ABSTAIN" (motorun ham verdikti).
#     ai_post   : birleşik posterior 0..1 — EŞİK TARAMASI için sürekli skor
#                 (ikili veto tek bir eşiği ölçer, posterior tüm eşikleri ölçtürür).
#     ai_side   : verdiktin ait olduğu İŞLEM tarafı (LONG/SHORT). Getiriyi işaretlemek
#                 için ŞARTTIR ve `pwin_side` META_GATE kapalıyken BOŞ kalır — tarafı
#                 oradan okumaya çalışan analiz sıfır satır bulurdu (`neural_pwin_side`
#                 kolonunun var olma gerekçesiyle birebir aynı).
#     ai_veto   : 1 = `AI_LIVE_MODE=enforce` AÇIK OLSAYDI bu sinyal susardı, 0 = geçerdi.
#                 `hit`/`fwd_ret_pct` ile join → kapı simülasyonu: vetolananlar gerçekten
#                 daha mı kötüydü? Bu sütun olmadan enforce'a geçmek KANITSIZ olurdu.
#   `AI_VERDICT_LOG=off` ölçümü kapatır. Eski satırlarda boş kalır (şema göçü kayıpsız).
# mkt_health / har_rv_fc / spread_bps / hmm_* (Tur-37) — GÖLGE ÖLÇÜM, KARAR-DIŞI:
#   Kaynak `nihai/ai/observatory.bar_shadow`; `_neural`/`_dirbar` ile BİREBİR aynı sözleşme
#   (satırda '_obs' altında taşınır, record() sonrası düşürülür, web JSON'una/panoya girmez).
#   NEDEN bu kolonlar: `microstructure`, `regime_hmm` ve `data_health` yazılmış, test edilmiş
#   ve HİÇ ÇAĞRILMAMIŞ modüllerdi. Çağrılmayan modül ÖLÇÜLMEZ, ölçülmeyen modül kanıt
#   kapısına HİÇ GELEMEZ — kapının önünde sıra bile yoktu. Bu kolonlar o sırayı kurar.
#     mkt_health    : bar veri-sağlığı skoru 0..100 (Madde 34'ün BAR tarafı; kotasyon
#                     tarafını `is_quote_healthy` zaten denetliyordu, barları HİÇBİR ŞEY
#                     denetlemiyordu). Hiçbir barı REDDETMEZ — yalnız ölçer.
#     har_rv_fc     : HAR-RV (Corsi 2009) 1-adım ileri gerçekleşen-varyans ÖNGÖRÜSÜ, `rv`
#                     ile AYNI birimde (Σr²). Projede oynaklık hep ÖLÇÜLÜYOR, hiç TAHMİN
#                     EDİLMİYORDU; öngörü ↔ gerçekleşen farkı ancak loglanırsa ölçülebilir.
#     spread_bps    : Corwin-Schultz etkin makas (bps) — ACI DERS #7'nin ("kenar ilk bardaki
#                     makas sıçramasıydı") teşhis aracı. Tahminci yansız değilse BOŞ kalır
#                     (Tur-31: ölçüm YOKLUĞU bir ölçüm gibi yazılmaz).
#     hmm_mode_prob : bir sonraki barın en olası rejiminin OLASILIĞI. `hmm_label` bu kolon
#                     olmadan ASLA yazılmaz — 0,34'lük bir tepe rejim ilan etmez.
#     hmm_switch    : rejim DEĞİŞİM olasılığı (1 − P(aynı durumda kal)). `knowledge.py`'nin
#                     kendi kataloğunda "geçiş olasılığı mevcut modülde yoktur" yazan boşluk.
#     hmm_label     : durumun okunabilir adı (`classify_regime` sözlüğüyle uyumlu) — bir
#                     YORUMDUR, ölçüm değildir; hiçbir karar ada bakmaz.
#   Eski satırlarda boş kalır (şema göçü kayıpsız).
# dir_bar / conf_bar (2026-07-30, Tur-20) — GÖLGE yön hipotezi, KARAR-DIŞI:
#   Tur-19 ölçtü ki giriş `price` (canlı quote) ile çıkış `fwd_price` (bar kapanışı) FARKLI
#   kaynaklardan geliyor ve fark sistematik olarak aleyhte (+1,85 bps, t=21,2). Ölçüm,
#   aynı tahminler BAR KAPANIŞINDA girilseydi isabetin 0,4654 → 0,4937 olacağını gösterdi.
#   O ölçüm GİRİŞ FİYATINI değiştirdi, YÖNÜ sabit tuttu. "Yönü de bar kapanışından hesapla"
#   ise AYRI ve HENÜZ SINANMAMIŞ bir hipotezdir — doğrudan sinyal yoluna konamaz (ACI DERS
#   #3/#4: sınanmamış hipotez üretime girmez).
#   Bu yüzden yön/güven her barda İKİ KEZ hesaplanır: `direction`/`confidence` yürürlükteki
#   (canlı fiyat tabanlı), `dir_bar`/`conf_bar` ise bar-kapanışı tabanlı GÖLGE. Kolonlar
#   biriktikçe "hangisi daha iyi" sorusu CANLI veriyle yanıtlanabilir; ancak o zaman
#   SIGNAL_ON_BAR_CLOSE=true ile varsayılan çevrilmelidir.
#   Tur-83: ÇEVRİLDİ (43.195 satır, aynı referansta +1,92 puan, gün-kümeli t=+12,8) — ölçüm
#   `live_predict_service._yon_kipi` docstring'inde; karşı varyant artık `dir_live`dadır.
#   Eski satırlarda boş kalır (şema göçü kayıpsız).
# bar_close / hit_bar / slip_bps (2026-07-27, F1 ölçümü) — NEDEN:
#   `price` sinyal ANINDAKİ canlı fiyattır (quote.last, sağlık kontrolünden geçerse) ve YÖN
#   de o fiyattan hesaplanır; `fwd_price` ise +horizon barın KAPANIŞIDIR. Yani `hit` iki
#   farklı fiyat kaynağını karşılaştırır ve içinde iki şey birden vardır:
#     (a) yön becerisi, (b) bar kapanışı ile sinyal anı arasındaki KOVALAMA maliyeti.
#   Ölçüldü (research/horizon_sweep.py, 50k tahmin): kayma sinyal yönünde ORTALAMA pozitif
#   (t≈3-5, 6/6 sembolde) → canlı isabet 0,4664 iken aynı tahminler bar kapanışında
#   girilseydi 0,5033 olurdu (3,9 puan). Bu bir ölçüm hatası değil GERÇEK maliyettir; ama
#   ayrıştırılmadan "tahmin gücü" hakkında hiçbir şey söylenemez.
#   `hit` DEĞİŞMEDİ (geriye uyum + gerçekçi olan o); `hit_bar` saf yön becerisini, `slip_bps`
#   aradaki farkı ölçer. Eski satırlarda bu kolonlar boş kalır (şema göçü kayıpsız).
#   ⚠️ Tur-83 (BULGU-83-2): `hit_bar` YALNIZ yön bar kapanışından hesaplandığında "saf"tır.
#   Yön canlı kotasyondan geliyorsa (Tur-83 öncesi varsayılan; kotasyon bar kapanışından
#   medyan ~39 sn SONRA okunur) `hit_bar` İLERİ-BAKIŞ taşır: yön, ölçümün başladığı fiyattan
#   sonra olmuş hareketi zaten görmüştür. Tur-60'ın "uyuşmazlıkta yön DAHA DOĞRU (%60,9)"
#   okuması bu kirlenmiş ölçüydü; AYNI referansta (canlı fiyat) o dilimde canlı-yön %26,2,
#   bar-yön %73,5 isabet eder. İki yönü kıyaslarken ikisi de AYNI referanstan ölçülmelidir.
# Canlı ÖZELLİK logu — meta modelin gelecekteki eğitim verisi (survivorship-temiz,
# leak-free: özellikler karar barında, etiket horizon sonra predictions_log'dan).
# (symbol, bar_ts) anahtarı predictions_log ile birebir join sağlar.
FEATS_LOG = DATA_DIR / "live_features_log.csv"
# cand_side (Tur-59, BULGU-59-1): satırın ADAY sinyal barı olup olmadığı ("LONG"/"SHORT"/"").
# Politika değişmeden ÖNCE bu log yalnız aday barlarını taşıyordu; artık taranan HER bar
# yazılır ve `cand_side != ""` filtresi eski davranışın satır kümesini BİREBİR verir.
# FEATURE_NAMES'in DIŞINDADIR: model girdisi değişmez, yalnız tüketici ayırt edebilir.
# CONTEXT_FIELDS (Tur-59, BULGU-59-7): piyasa GENİŞLİĞİ / göreceli güç / çapraz varlık.
# FEATURE_NAMES'in DIŞINDADIR — model girdisi ve tüm mevcut ölçümler DEĞİŞMEZ; bu kolonlar
# yalnız BİRİKİR ki ileride `research/edge_lab` onları bir AİLE olarak sınayabilsin
# (kanıtsız bileşen skora eklenmez — Tur-51).
FEAT_FIELDS = ["logged_at", "symbol", "interval", "bar_ts", *FEATURE_NAMES,
               "cand_side", *CONTEXT_FIELDS]
# action    : satır anındaki sinyal aksiyonu (LONG/SHORT/WAIT) — pwin'li WAIT = kapı-engelli aday.
# pwin_side : pwin'in ait olduğu İŞLEM tarafı (LONG/SHORT). Yön-forecast'ten ('direction')
#             bağımsızdır; kapı simülasyonu getiriyi BU tarafa göre işaretler.
# neural_*  : nöral kapasite katmanının GÖLGE p_win'i (Tur-18, nihai/execution/neural_shadow).
#             pwin ile AYNI şema + MC-dropout std'si. KARAR-DIŞI: hiçbir sinyali/kapıyı
#             etkilemez, yalnız ai/learning.py'nin `sources.neural` Brier'ini besler
#             (kalibrasyonun canlı ölçümü). Model/torch yoksa kolon boş kalır.

# Forecast intervalında kaç bar ileriye bakılır (gerçekleşen yön penceresi).
HORIZON_BARS = int(os.getenv("PREDICT_TRACK_HORIZON_BARS", "5"))

# Tur-81: resolve()'un bar ÇEKME fan-out'u kaç thread'e dağılır. 1 = eski (sıralı) davranış.
# 8: canlı defterdeki 30 anahtar için ~4 dalga; hub tek süreçtir ve kendi API hız sınırı
# (200/dk) vardır — daha geniş havuz kuyruğu hub'a taşır, kazanç getirmez.
RESOLVE_FETCH_WORKERS = max(1, int(os.getenv("RESOLVE_FETCH_WORKERS", "8")))

_WRITE_LOCK = threading.Lock()
# _atomic_replace: OneDrive/AV kilidine dayanıklı yeniden-denemeli os.replace. Tur-11'de burada
# yerel tanımlıydı; Tur-12'de nihai/atomic_io'ya taşındı (state.json/watchlist de aynı kilit
# riskindeydi — TEK doğruluk kaynağı). Ad korunuyor (mevcut çağrılar + testler değişmesin).


def _is_directional(row: dict) -> bool:
    """Yönlü tahmin mi? (NODATA/ERROR ve yönsüz satırlar kaydedilmez)."""
    return (row.get("action") not in (None, "NODATA", "ERROR")
            and row.get("direction") in ("+", "-")
            and float(row.get("price") or 0) > 0
            and bool(row.get("bar_ts")))


def _col(header: list[str], name: str) -> int:
    """`name`in başlıktaki SON konumu (csv.DictReader'da tekrar eden adda sonuncu kazanır); yoksa -1."""
    for j in range(len(header) - 1, -1, -1):
        if header[j] == name:
            return j
    return -1


def _dict_row(header: list[str], row: list[str]) -> dict:
    """csv.DictReader'ın ürettiği sözlüğün BİREBİR aynısı (kısa satır → None, fazla alan → None anahtarı)."""
    d: dict = dict(zip(header, row, strict=False))
    lf, lr = len(header), len(row)
    if lf < lr:
        d[None] = row[lf:]
    elif lf > lr:
        for key in header[lr:]:
            d[key] = None
    return d


def _existing_keys() -> set[tuple[str, str]] | None:
    """Zaten kaydedilmiş (symbol, kanonik-bar_ts) anahtarları — bar başına tek kayıt (dedup).

    Denetim Tur-11: eskiden HAM `r["bar_ts"]` string'i anahtardı. Canlı sağlayıcı failover'ı
    (hub↔yfinance) aynı barı FARKLI gösterimle (naive-UTC vs tz-aware-ET) döndürebilir →
    ham eşleşme aynı barı iki kez kaydeder (isabet/AUC istatistiklerini şişirir). `_canon_ts`
    ile aynı sözleşme (`_forward_price`/`_pos_map`, C1) burada da kullanılır — TEK anahtar.

    Tur-12: okuma HATA verirse (ör. OneDrive kilidi) None döner. Kısmi/boş anahtar kümesiyle
    dedup yapmak zaten-kayıtlı barları YENİ sanıp ÇİFT kaydeder (isabet/AUC şişer) → çağıran
    None görünce o döngüde YAZMAZ (imza uyuşmadığından sonraki döngü yeniden tohumlar)."""
    if not PRED_LOG.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    try:
        with PRED_LOG.open(encoding="utf-8", newline="") as f:
            # Tur-80: satır başına sözlük kurmadan iki kolon okunur (85.576 satırda 670 → 442 ms).
            # DictReader anlambilimi korunur: boş satır atlanır, kısa satırda eksik alan None,
            # başlıkta ad yoksa (veri satırı varken) KeyError → aşağıdaki None dönüşü.
            reader = csv.reader(f)
            header = next(reader, None) or []
            si, bi = _col(header, "symbol"), _col(header, "bar_ts")
            for row in reader:
                if not row:
                    continue
                if si < 0 or bi < 0:
                    raise KeyError("symbol" if si < 0 else "bar_ts")
                n = len(row)
                keys.add((row[si] if si < n else None,   # type: ignore[arg-type]
                          _canon_ts(row[bi] if bi < n else None)))
    except Exception as e:
        print(f"[tracker] _existing_keys okunamadı ({e}); bu döngüde kayıt atlanıyor "
              "(çift-kayıt guard, Tur-12)", file=sys.stderr)
        return None
    return keys


# D1 — record() her ~20s döngüde TÜM predictions_log.csv'yi yeniden okumasın (dosya
# sınırsız büyür, _WRITE_LOCK altında). Modül-cache: bir kez diskten tohumla, yazımda
# ekle. Dosya imzası (mtime_ns, size) beklenmedik değişirse (başka süreç/işlem yazdı
# VEYA _ensure_schema yeniden yazdı) diskten yeniden tohumla → dedup davranışı BİREBİR.
_SEEN_KEYS: set[tuple[str, str]] | None = None
_SEEN_SIG: tuple[int, int] | None = None


def _file_sig() -> tuple[int, int] | None:
    """PRED_LOG imzası (mtime_ns, size) — dış değişiklik tespiti. Dosya yoksa None."""
    try:
        st = PRED_LOG.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _is_fresh(path) -> bool:
    """Dosya yok VEYA 0-bayt (boş) → 'taze' say → başlık yazılmalı. Var-ama-boş CSV'ye
    başlıksız append şemayı bozardı (C5)."""
    try:
        return path.stat().st_size == 0
    except OSError:
        return True


def _ensure_schema() -> None:
    """predictions_log başlığını güncel FIELDS'e yükselt (yeni sütun, örn. pwin, eklenince).

    Eski başlıklı dosyaya append başlık↔veri kayması yaratır (13-sütun başlık altına 14
    değer). Başlık eşleşmiyorsa dosyayı atomik yeniden yaz: eksik yeni sütunlar '' olur,
    bilinmeyen eski sütunlar düşürülür. Çağıran _WRITE_LOCK tutmalı."""
    if not PRED_LOG.exists():
        return
    with PRED_LOG.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return
    if header == FIELDS:
        return
    with PRED_LOG.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    tmp = _tmp_path(PRED_LOG)
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)  # eksik FIELDS (pwin) → '' (restval); eski fazlalık → ignore
    _atomic_replace(tmp, PRED_LOG)


def _ctx_cell(name: str, v) -> str | float | int:
    """Bağlam hücresi: ölçüm yoksa BOŞ, sayım ise TAM SAYI, oran ise yuvarlanmış float."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if f != f:                       # NaN — ölçüm yokluğunun ikinci biçimi
        return ""
    return int(f) if name in COUNT_FIELDS else round(f, 6)


def _num_cell(v, nd: int) -> str | float:
    """Pozitif sonlu ölçü → yuvarlanmış float; yok/bozuk/≤0 → BOŞ (ölçüm yokluğu 0 yazılmaz)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return round(f, nd) if math.isfinite(f) and f > 0 else ""


def _write_features(new_rows: list[dict], interval: str, now: str) -> int:
    """Yeni kaydedilen tahmin satırlarının '_feats' vektörünü FEATS_LOG'a ekle.

    Yalnız record()'un YENİ yazdığı satırlarla çağrılır → dedup predictions_log'dan
    miras (bar başına tek). Başlık FEATURE_NAMES değişirse atomik yükseltilir
    (_ensure_schema ile aynı desen). Çağıran _WRITE_LOCK tutmalı. Yazılan satır sayısı."""
    cand = [r for r in new_rows if isinstance(r.get("_feats"), dict)]
    if not cand:
        return 0
    if FEATS_LOG.exists():
        with FEATS_LOG.open(encoding="utf-8", newline="") as f:
            try:
                header = next(csv.reader(f))
            except StopIteration:
                header = FEAT_FIELDS
        if header != FEAT_FIELDS:
            with FEATS_LOG.open(encoding="utf-8", newline="") as f:
                old = list(csv.DictReader(f))
            tmp = _tmp_path(FEATS_LOG)
            with tmp.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FEAT_FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(old)   # eksik yeni özellik → '' (restval); eski fazlalık → düşer
            _atomic_replace(tmp, FEATS_LOG)
    fresh = _is_fresh(FEATS_LOG)   # C5: yok VEYA boş → başlık yaz (başlıksız append önle)
    with FEATS_LOG.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FEAT_FIELDS, extrasaction="ignore")
        if fresh:
            w.writeheader()
        for r in cand:
            feats = r["_feats"]
            row = {"logged_at": now, "symbol": r["symbol"], "interval": interval,
                   "bar_ts": r["bar_ts"],
                   "cand_side": str(feats.get("_cand_side") or "")}
            row.update({k: round(float(v), 6) for k, v in feats.items() if k in FEATURE_NAMES})
            # Ölçülemeyen bağlam alanı BOŞ kalır (0.0 yazmak "ölçüm yokluğunu ölçüm gibi
            # sunmak" olurdu — Tur-31/33/56). `br_n` bir SAYIMDIR, 0 gerçek bir değerdir.
            ctx = feats.get("_ctx") or {}
            row.update({k: _ctx_cell(k, ctx.get(k)) for k in CONTEXT_FIELDS})
            w.writerow(row)
    return len(cand)


def _dirbar_cells(row: dict, key: str = "_dirbar") -> tuple[str, str]:
    """Satırdaki yön varyantını hücrelere çevir: (yön, güven) — `_dirbar` → dir_bar/conf_bar,
    `_dirlive` → dir_live/conf_live (Tur-83; iki varyant AYNI sözleşmeyle yazılır).

    Beklenen biçim: {'direction': '+'|'-', 'confidence': float}. Yok/bozuk → boş hücre
    (kayıt ASLA engellenmez — gölge katman tahmin satırından daha az önemlidir;
    `_neural_cells` ile aynı sözleşme). Güven bozuksa yön yine de yazılır: yön hipotezi
    güvenden bağımsız olarak karşılaştırılabilir."""
    dv = row.get(key)
    if not isinstance(dv, dict):
        return "", ""
    d = str(dv.get("direction") or "")
    if d not in ("+", "-"):
        return "", ""
    try:
        return d, str(round(float(dv["confidence"]), 4))
    except (KeyError, TypeError, ValueError):
        return d, ""


def _neural_cells(row: dict) -> tuple[str, str, str]:
    """Satırdaki '_neural' gölge çıkarımını CSV hücrelerine çevir: (pwin, side, std).

    Beklenen biçim: {'pwin': float, 'side': 'LONG'|'SHORT', 'std': float}. Yok/bozuk/
    eksik → üç boş hücre (kayıt ASLA engellenmez — gölge katman tahmin satırından
    daha az önemlidir). Taraf 'pwin_side'dan bağımsız yazılır: gölge kaynağın kendi
    tarafı hangisiyse learning.py getiriyi ONA göre işaretlemeli."""
    nv = row.get("_neural")
    if not isinstance(nv, dict):
        return "", "", ""
    side = str(nv.get("side") or "").upper()
    if side not in ("LONG", "SHORT"):
        return "", "", ""
    try:
        p = round(float(nv["pwin"]), 4)
    except (KeyError, TypeError, ValueError):
        return "", "", ""
    try:
        std = str(round(float(nv["std"]), 4))
    except (KeyError, TypeError, ValueError):
        std = ""   # std opsiyonel — p_win'siz bırakmaz
    return str(p), side, std


def _ai_cells(row: dict) -> tuple[str, str, str, str]:
    """Satırdaki '_ai' verdiktini hücrelere çevir: (ai_action, ai_post, ai_veto, ai_side).

    `_neural_cells` ile AYNI sözleşme: yok/bozuk → üç boş hücre, kayıt ASLA engellenmez.
    ÜÇÜ BİRLİKTE yazılır: `ai_veto` tek başına okunamaz (hangi verdiktten ve hangi
    posteriordan geldiği bilinmeden bir kapı simülasyonu kurulamaz — Tur-33'ün "bir ORAN,
    paydası okunmadan okunamaz" kuralının buradaki karşılığı)."""
    av = row.get("_ai")
    if not isinstance(av, dict):
        return "", "", "", ""
    action = str(av.get("action") or "").upper()
    side = str(av.get("side") or "").upper()
    if action not in ("LONG_BIAS", "SHORT_BIAS", "ABSTAIN") or side not in ("LONG", "SHORT"):
        return "", "", "", ""
    try:
        post = round(float(av["post"]), 4)
        veto = int(av["veto"])
    except (KeyError, TypeError, ValueError):
        return "", "", "", ""
    if veto not in (0, 1):
        return "", "", "", ""
    return action, str(post), str(veto), side


_OBS_NUMERIC = (("mkt_health", "health", 1), ("har_rv_fc", "har_rv_fc", 9),
                ("spread_bps", "spread_bps", 2), ("hmm_mode_prob", "hmm_mode_prob", 4),
                ("hmm_switch", "hmm_switch_prob", 4),
                # Tur-76 (Madde 069/070/105): hacim profili ve net alıcı baskısı.
                # Üçü de SEMBOLLER ARASI kıyaslanabilir olacak şekilde seçildi —
                # ham CVD hacim ölçeğindedir ve bir kapı adayı olarak kıyaslanamaz,
                # bu yüzden deftere UYUŞMAZLIK işareti (−1/0/+1) girer, seviye değil.
                ("poc_dist_bps", "poc_dist_bps", 2),
                ("value_area_pos", "value_area_pos", 0),
                ("cvd_divergence", "cvd_divergence", 0))


def _obs_cells(row: dict) -> dict[str, str]:
    """Satırdaki '_obs' gölge ölçümünü (observatory.bar_shadow) CSV hücrelerine çevir.

    `_neural_cells`/`_dirbar_cells` ile AYNI sözleşme: yok/bozuk/eksik alan → o hücre BOŞ,
    kayıt ASLA engellenmez. Her alan BAĞIMSIZ değerlendirilir — makas ölçülemediyse rejim
    olasılığı yine yazılır (gölge katman kısmi de olsa değerlidir).

    TEK bağımlılık kuralı: `hmm_label` yalnız `hmm_mode_prob` ile BİRLİKTE yazılır
    (regime_hmm sözleşmesi: "mode_prob okunmadan mode okunamaz" — paydasız bir tepe
    rejim ilan edemez)."""
    ov = row.get("_obs")
    if not isinstance(ov, dict):
        return {}
    cells: dict[str, str] = {}
    for col, key, nd in _OBS_NUMERIC:
        try:
            v = float(ov[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(v):
            cells[col] = str(round(v, nd))
    label = str(ov.get("hmm_label") or "")
    if label and "hmm_mode_prob" in cells:
        cells["hmm_label"] = label
    return cells


def record(rows: list[dict], interval: str, horizon: int = HORIZON_BARS) -> int:
    """Yönlü tahminleri predictions_log.csv'ye ekle (bar başına tek, dedup). Eklenen sayı.

    `rows` predict servisi satırları (symbol/action/direction/confidence/prob_cal/price/bar_ts).
    Thread-safe; servis döngüsünden her bar çağrılabilir, yalnız YENİ barlar yazılır."""
    cand = [r for r in rows if _is_directional(r)]
    if not cand:
        return 0
    with _WRITE_LOCK:
        _ensure_schema()  # eski başlıklı dosyayı pwin'li şemaya yükselt (append kayması önle)
        global _SEEN_KEYS, _SEEN_SIG
        # D1: dedup anahtarlarını modül-cache'ten oku; dosya imzası beklenmedik değiştiyse
        # (dış yazım / _ensure_schema yeniden-yazımı / test dosya değişimi) yeniden tohumla.
        if _SEEN_KEYS is None or _SEEN_SIG is None or _file_sig() != _SEEN_SIG:
            seeded = _existing_keys()
            if seeded is None:
                return 0  # Tur-12: okuma hatası → kısmi dedup kümesiyle YAZMA (çift-kayıt
                          # guard); imza hâlâ uyuşmadığından sonraki döngü yeniden tohumlar.
            _SEEN_KEYS, _SEEN_SIG = seeded, _file_sig()
        seen = _SEEN_KEYS
        assert seen is not None  # yukarıdaki blok None bırakmaz (mypy daraltması)
        new = [r for r in cand if (r["symbol"], _canon_ts(r["bar_ts"])) not in seen]
        if not new:
            return 0
        fresh_file = _is_fresh(PRED_LOG)   # C5: yok VEYA boş → başlık yaz
        now = datetime.now().isoformat(timespec="seconds")
        with PRED_LOG.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if fresh_file:
                w.writeheader()
            for r in new:
                pw = r.get("pwin")
                npw, nside, nstd = _neural_cells(r)
                dbar, cbar = _dirbar_cells(r)
                dlive, clive = _dirbar_cells(r, "_dirlive")
                aiact, aipost, aiveto, aiside = _ai_cells(r)
                w.writerow({
                    "logged_at": now, "symbol": r["symbol"], "interval": interval,
                    "bar_ts": r["bar_ts"], "direction": r["direction"],
                    "action": r.get("action", ""),
                    "confidence": r.get("confidence", ""), "prob_cal": r.get("prob_cal", ""),
                    "pwin": ("" if pw is None else round(float(pw), 4)),
                    "pwin_side": (r.get("pwin_side") or ""),
                    "meta_model_id": r.get("meta_model_id") or "",
                    "meta_strategy": r.get("meta_strategy") or "",
                    "price": r["price"], "bar_close": r.get("bar_close", ""),
                    "horizon_bars": horizon,
                    "resolved_at": "", "fwd_price": "", "hit": "", "fwd_ret_pct": "",
                    "hit_bar": "", "slip_bps": "",
                    "neural_pwin": npw, "neural_pwin_side": nside, "neural_pwin_std": nstd,
                    "dir_bar": dbar, "conf_bar": cbar,
                    "dir_live": dlive, "conf_live": clive,
                    "atr_pct": _num_cell(r.get("atr_pct"), 6),
                    "ai_action": aiact, "ai_post": aipost, "ai_veto": aiveto,
                    "ai_side": aiside,
                    # Gölge ölçüm (mikroyapı/rejim-HMM/veri-sağlığı) — eksik alanlar
                    # DictWriter'ın restval'ı ile boş kalır, yazılmayan kolon yoktur.
                    **_obs_cells(r),
                })
        # D1: yeni anahtarları cache'e ekle + imzayı yenile (sonraki çağrı diski okumaz).
        for r in new:
            _SEEN_KEYS.add((r["symbol"], _canon_ts(r["bar_ts"])))
        _SEEN_SIG = _file_sig()
        try:
            _write_features(new, interval, now)
        except Exception as e:
            # özellik logu opsiyonel — tahmin kaydını asla engellemesin; ama SESSİZ kaybı
            # (build_live_dataset'in eğitim satırı) görünür kıl (Tur-12).
            print(f"[tracker] _write_features hatası ({e}); {len(new)} özellik satırı bu "
                  "döngüde yazılamadı (tahmin kaydı sağlam)", file=sys.stderr)
    return len(new)


def _canon_ts(ts) -> str:
    """bar_ts'i kanonik anahtara indir: tz-aware ise UTC'ye çevirip tz düşür, naive ise
    olduğu gibi. tz-aware KAYIT (canlı hub) vs tz-naive ÇÖZÜM (stripped df) aynı barı
    eşlesin diye — ham str eşleşmesi her satırı kalıcı 'pending' bırakıyordu (C1, ACI
    DERS #5: forward-kanıt döngüsü sessizce ölür). Ayrıştırılamazsa ham str döner (eski
    satırlarla geriye-uyumlu); aynı gösterimde idempotent → hâlihazırda eşleşenleri bozmaz."""
    s = str(ts)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return s
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt.isoformat()


def _pos_map(df) -> dict[str, int]:
    """df.index → {kanonik-ts: konum} (D5: cache'li df başına BİR kez kurulur; satır başına
    str-index listesi + lineer arama yerine O(1) sözlük araması). Yinelenen ts'te İLK konum
    tutulur (eski idx.index() ilk-eşleşme davranışıyla birebir)."""
    m: dict[str, int] = {}
    for j, t in enumerate(df.index):
        k = _canon_ts(t)
        if k not in m:
            m[k] = j
    return m


def _forward_price(df, pos_map: dict[str, int], bar_ts: str, horizon: int,
                   interval: str | None = None) -> float | None:
    """bar_ts'ten `horizon` ARALIK SÜRESİ sonraki fiyat (henüz belirlenmediyse None).

    Look-ahead yok: yalnız bar_ts'ten sonra GERÇEKLEŞMİŞ barlara bakar. Eşleşme kanonik-UTC
    anahtarla (tz-aware/naive dayanıklı, C1); pos_map cache'li df başına bir kez kurulur (D5).

    Tur-84 (BULGU-84-1): ufuk ZAMANLA ölçülür, KONUMLA değil. Eskiden `iloc[pos + horizon]`
    okunuyordu; IEX'te işlem olmayan dakika bar ÜRETMEZ, yani "5 bar" 5 dakika değildi.
    Ölçüldü (95.576 çözülmüş satır, tarihsel IEX 1m ile yeniden kurulum, bar_close paritesi
    medyan 0 bps): satırların %18,6'sında gerçek ufuk >5 dk, %2,3'ünde >10 dk; LMT'de %88,3;
    411 satırda ufuk geceyi/hafta sonunu AŞIYORDU (en uzun 3950 dk). Konum-getirilerinin
    varyansı zaman-getirilerininkinden %39 fazla (IBM'de std 2,4×). Kural: hedef an
    t0 + horizon·aralık; o anda BAŞLAYAN bar varsa onun kapanışı (düzenli ızgarada eski
    kuralla BİREBİR), yoksa hedeften önceki son barın kapanışı — ama yalnız hedeften SONRA
    bir bar oluşmuşsa (an geçmiştir; seans sonuna 5 dk'dan az kala verilen tahmin kapanış
    fiyatıyla, ertesi seansın ilk barı gelince çözülür). "1d"/bilinmeyen aralıkta seri işlem
    günü başına düzenlidir → eski konum kuralı."""
    pos = pos_map.get(_canon_ts(bar_ts))
    if pos is None:
        return None
    dk = INTERVAL_MINUTES.get(str(interval)) if interval not in (None, "1d") else None
    try:
        if dk is None:
            tgt = pos + horizon
            if tgt >= len(df):
                return None
            return float(df["close"].iloc[tgt])
        idx = df.index
        hedef = idx[pos] + timedelta(minutes=horizon * dk)
        j = int(idx.searchsorted(hedef, side="right")) - 1
        if not (idx[j] == hedef or j + 1 < len(idx)):
            return None                       # hedef an henüz geçmedi (arkasında bar yok)
        return float(df["close"].iloc[j])
    except Exception:
        return None


def _cozulebilir(bar_ts, horizon: int, interval, simdi: datetime) -> bool:
    """Hedef andan (bar + ufuk) sonra bir seans dakikası geçti mi? Geçmediyse `_forward_price` None döner ve bar
    çekmek boşunadır. Tur-106 (BULGU-106-20): gözcü `resolve`u 5 dk'da bir tetikler; seans dışında bekleyen ~120
    satırın hepsi ertesi açılışı bekler → gece/hafta sonu her turda ~30 hub isteği hiçbir şey çözmüyordu.
    Ölçülemeyen satır ("1d", bozuk damga, takvim dışı) → True (eski davranış: çek)."""
    dk = INTERVAL_MINUTES.get(str(interval)) if interval not in (None, "1d") else None
    if dk is None:
        return True
    try:
        from nihai.data.market_calendar import ilk_bar_ani

        t = datetime.fromisoformat(_canon_ts(bar_ts)).replace(tzinfo=UTC)   # kanonik bar_ts = UTC
        return simdi >= ilk_bar_ani(t + timedelta(minutes=horizon * dk))
    except Exception:
        return True


def _is_aged_out(df, bar_ts) -> bool:
    """bar_ts, çekilen pencerenin (lookback_bars) EN ESKİ barından daha ESKİ mi? Öyleyse bu
    tasarımla artık ASLA çözülemez (uzun kesinti / delist) → resolve() UNRESOLVABLE işaretler.
    Çok-YENİ (horizon dolmamış) satırları YANLIŞLIKLA işaretlememek için yalnız kesin-ESKİ
    durumu döndürür (kanonik-UTC lexical karşılaştırma = kronolojik; _canon_ts ile TEK sözleşme)."""
    try:
        return _canon_ts(bar_ts) < _canon_ts(df.index[0])
    except Exception:
        return False


def resolve(provider=None) -> dict:
    """Vadesi gelen (resolved_at boş, horizon bar geçmiş) tahminleri gerçekleşenle etiketle.
    Döner: {'checked', 'resolved', 'pending'}.

    ⚠️ BİLİNEN SINIR (Tur-11 denetimi, PLAUSIBLE, düzeltilMEDİ — bilinçli erteleme):
    `lookback_bars=400` sabittir; uzun servis kesintisi (>400 bar) veya delist edilmiş
    sembolde bar_ts artık dönen pencerede yoksa `_forward_price` sürekli None döner →
    satır SONSUZA dek 'pending' kalır (asla UNRESOLVABLE diye ayrılmaz). Bu, canlı eğitim
    setinin (build_live_dataset) "yapısı gereği survivorship-temiz" iddiasını hafifçe
    aşındırır: tam delist-öncesi sonuçlar sessizce dışlanabilir. Doğru fix (schema
    değişikliği + report()/gate_sim()/coverage()/forward_harness/dashboard'ın 'pending'
    sayımını güncellemesi gerektirir — canlı 2M+ satırlık predictions_log.csv'ye dokunan
    orta-büyük bir değişiklik) BİLİNÇLİ ERTELENDİ; ayrı bir oturumda ele alınmalı. Bkz.
    DENETIM_BULGULARI_2026-07-09.md Tur-11 eki."""
    if not PRED_LOG.exists():
        return {"checked": 0, "resolved": 0, "pending": 0}
    if provider is None:
        from nihai.data.data_provider import get_provider
        provider = get_provider()

    # Tur-80: bu okuma pano yoklamasında (≤20 sn TTL) defterin TAMAMINI sözlüğe çeviriyordu
    # (85.576 satır / 14,2 MB → çözülecek satır 0 iken bile 739 ms). Yalnız ÇÖZÜLMEMİŞ satır
    # sözlüğe çevrilir; sözlükler DictReader'ınkiyle birebir, `checked` hâlâ tüm satır sayısıdır.
    rows: list[dict] = []
    n_rows = 0
    with _WRITE_LOCK, PRED_LOG.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None) or []
        ri = _col(header, "resolved_at")
        for row in reader:
            if not row:
                continue
            n_rows += 1
            if 0 <= ri < len(row) and row[ri]:
                continue
            rows.append(_dict_row(header, row))

    # Sembol+interval başına bir kez bar çek (tekrar ağ çağrısını önle).
    bars_cache: dict[tuple[str, str], object] = {}
    posmap_cache: dict[tuple[str, str], dict[str, int]] = {}   # D5: df başına tek pos_map
    # Çözümleri key'le topla (satır nesnesini DEĞİL). Ağ-çözümü uzun sürer; bu sırada
    # record() yeni satır ekleyebilir VEYA ayrı resolver süreci dosyayı değiştirebilir.
    # Eski snapshot'ı geri yazmak yerine, sona dosyayı YENİDEN okuyup çözümleri eşleştir
    # ve atomik yaz → boşlukta eklenen tahminler kaybolmaz, çökmede dosya bozulmaz.
    resolutions: dict[tuple[str, str, str], dict[str, str]] = {}
    pending = 0
    unresolvable = 0

    # Tur-81 (BULGU-81-1): bu döngü bekleyen satırların HER (sembol, interval) anahtarı için
    # `provider.get_bars`i SIRAYLA çağırıyordu. ÖLÇÜLDÜ (canlı defter, 2026-09-17): 30 anahtar,
    # hub yanıtı 3 ms (önbellek isabeti) ile 6,9 sn (soğuk seed) arasında → toplam ~41 sn, ve
    # `/api/predictions/accuracy` canlı süreçte 43,5 sn sürüyordu (aynı defterin izole
    # kopyasında 13,8 sn). Tur-80 bu döngünün CSV ayrıştırmasını hızlandırmıştı (0,2 sn);
    # süreyi belirleyen o değil, AĞ FAN-OUT'uydu ve izole kıyas onu hiç görmedi.
    # Çağrılar birbirinden BAĞIMSIZ ve ağ-bekleme ağırlıklıdır (GIL'i bırakırlar) → havuz.
    # Anahtar kümesi aşağıdaki döngünün fetch'e ULAŞTIĞI satırlarla BİREBİR aynı süzgeçten
    # geçer: `horizon` ayrıştırılamayan satır bugün de fetch tetiklemiyordu (BULGU-80-1 dalı),
    # bu yüzden ÇEKİLEN ANAHTAR KÜMESİ ve dönen sözlükler değişmez — yalnız sıra değişir.
    # "Ufku dolmamış satırı atla" Tur-81'de SEANS İÇİNDE ölçülüp reddedilmişti (30 sembolün 30'unun da dolmuş
    # satırı vardı → 0 tasarruf); seans DIŞINDA bekleyenlerin hepsi ertesi açılışı bekler → Tur-106 (BULGU-106-20)
    # `_cozulebilir`: henüz çözülemeyecek satır bar çektirmez (iki döngüde aynı süzgeç → anahtar kümesi eşit).
    simdi = datetime.now(UTC)
    fetch_keys: list[tuple[str, str]] = []
    _gorulen: set[tuple[str, str]] = set()
    for r in rows:
        if r.get("resolved_at"):
            continue
        try:
            h0 = int(r.get("horizon_bars") or HORIZON_BARS)
        except (TypeError, ValueError):
            continue
        if not _cozulebilir(r["bar_ts"], h0, r["interval"], simdi):
            continue
        k = (r["symbol"], r["interval"])
        if k not in _gorulen:
            _gorulen.add(k)
            fetch_keys.append(k)

    def _cek(key: tuple[str, str]) -> tuple[tuple[str, str], Any, dict[str, int]]:
        sym0, iv0 = key
        try:
            df0 = provider.get_bars(sym0, iv0, lookback_bars=400)
        except Exception:
            df0 = None
        pmap = (_pos_map(df0) if df0 is not None and not getattr(df0, "empty", True) else {})
        return key, df0, pmap

    if len(fetch_keys) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(RESOLVE_FETCH_WORKERS, len(fetch_keys)),
                                thread_name_prefix="pt-resolve") as _havuz:
            for key, df0, pmap in _havuz.map(_cek, fetch_keys):
                bars_cache[key] = df0
                posmap_cache[key] = pmap

    for r in rows:
        if r.get("resolved_at"):
            continue
        sym, iv = r["symbol"], r["interval"]
        try:
            horizon = int(r.get("horizon_bars") or HORIZON_BARS)
        except (TypeError, ValueError):
            # BULGU-80-1: yarım yazılmış/bozuk TEK satır `int()`/`float()` istisnasıyla TÜM
            # çözümü düşürüyordu → kanıt döngüsü o satır elle onarılana kadar dururdu.
            pending += 1
            continue
        if not _cozulebilir(r["bar_ts"], horizon, iv, simdi):
            pending += 1
            continue
        key = (sym, iv)
        if key not in bars_cache:
            try:
                bars_cache[key] = provider.get_bars(sym, iv, lookback_bars=400)
            except Exception:
                bars_cache[key] = None
            df0 = bars_cache[key]
            posmap_cache[key] = (_pos_map(df0)
                                 if df0 is not None and not getattr(df0, "empty", True)
                                 else {})
        df = bars_cache[key]
        if df is None or getattr(df, "empty", True):
            pending += 1
            continue
        fwd = _forward_price(df, posmap_cache[key], r["bar_ts"], horizon, iv)
        if fwd is None:
            # Tur-12: bar_ts, çekilen 400-bar penceresinin EN ESKİ barından daha eskiyse artık
            # ASLA çözülemez (uzun kesinti / delist) → UNRESOLVABLE terminal durumu; aksi halde
            # SONSUZA 'pending' kalıp kanıt döngüsünü sessizce aşındırır (ACI DERS #5). Çok-YENİ
            # (horizon dolmamış) satır hâlâ pending.
            if _is_aged_out(df, r["bar_ts"]):
                resolutions[(sym, iv, str(r["bar_ts"]))] = {
                    "resolved_at": "UNRESOLVABLE", "fwd_price": "", "hit": "", "fwd_ret_pct": "",
                }
                unresolvable += 1
            else:
                pending += 1
            continue
        try:
            price = float(r["price"])
        except (TypeError, ValueError):
            price = 0.0            # BULGU-80-1: sayı olmayan fiyat → aşağıdaki 'çözülmedi' dalı
        if price <= 0:
            pending += 1   # Tur-12: bozuk/yönsüz fiyat da 'çözülmedi' say (sessiz düşme yok)
            continue
        ret_pct = (fwd - price) / price * 100.0
        signed = ret_pct if r["direction"] == "+" else -ret_pct  # yön-ayarlı getiri
        res_upd = {
            "resolved_at": datetime.now().isoformat(timespec="seconds"),
            "fwd_price": f"{fwd:.4f}", "hit": "1" if signed > 0 else "0",
            "fwd_ret_pct": f"{signed:.4f}",
        }
        # Saf yön becerisi (aynı fiyat kaynağı: bar kapanışı → bar kapanışı) + kovalama
        # maliyeti. `bar_close` yoksa (eski satır/eksik alan) sessizce atlanır.
        try:
            bc = float(r.get("bar_close") or 0.0)
        except (TypeError, ValueError):
            bc = 0.0
        if bc > 0:
            sgn = 1.0 if r["direction"] == "+" else -1.0
            res_upd["hit_bar"] = "1" if (fwd - bc) * sgn > 0 else "0"
            res_upd["slip_bps"] = f"{(price / bc - 1.0) * 1e4 * sgn:.2f}"
        resolutions[(sym, iv, str(r["bar_ts"]))] = res_upd

    resolved_count = sum(1 for v in resolutions.values()
                         if v.get("resolved_at") != "UNRESOLVABLE")
    if resolutions:
        global _ROWS_HANDOFF
        with _WRITE_LOCK:
            tmp = _tmp_path(PRED_LOG)
            # Sona kadar dosya değişmiş olabilir → güncel halini oku, çözümleri uygula.
            yazilan = _rewrite_resolved_fast(tmp, resolutions)
            if yazilan is None:
                with PRED_LOG.open(encoding="utf-8", newline="") as f:
                    cur = list(csv.DictReader(f))
                for r in cur:
                    if r.get("resolved_at"):
                        continue
                    upd = resolutions.get((r["symbol"], r["interval"], str(r["bar_ts"])))
                    if upd:
                        r.update(upd)
                with tmp.open("w", encoding="utf-8", newline="") as f:
                    # extrasaction="ignore": FIELDS-dışı legacy kolon ValueError ile resolve'u
                    # düşürmesin (record yolundaki _ensure_schema yeniden-yazımıyla aynı sözleşme).
                    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(cur)
            # Pano bu çağrının hemen ardından `summary()` ister ve az önce yazılan içeriği
            # baştan ayrıştırırdı; yazılan satırlar imzayla birlikte TEK SEFERLİK devredilir.
            # İmza ara dosyadan, değiştirmeden ÖNCE alınır (yeniden adlandırma mtime/boyutu korur):
            # sonradan alınsaydı araya giren başka SÜREÇ eklemesi eski satırlarla "eşleşirdi"
            # (Codex incelemesi; bağımsız doğrulandı: devir 160 satır ↔ disk 161).
            imza = None
            if yazilan is not None:
                st = tmp.stat()
                imza = (st.st_mtime_ns, st.st_size)
            _atomic_replace(tmp, PRED_LOG)
            _ROWS_HANDOFF = (imza, yazilan) if yazilan is not None else None
        return {"checked": n_rows, "resolved": resolved_count, "pending": pending,
                "unresolvable": unresolvable}
    return {"checked": n_rows, "resolved": 0, "pending": pending,
            "unresolvable": unresolvable}


_FI = {k: i for i, k in enumerate(FIELDS)}
# resolve() → summary() tek seferlik satır devri: (yazım sonrası dosya imzası, yazılan satırlar).
_ROWS_HANDOFF: tuple[tuple[int, int] | None, list[list[str]]] | None = None


def _rewrite_resolved_fast(tmp, resolutions: dict[tuple[str, str, str], dict[str, str]]
                           ) -> list[list[str]] | None:
    """`resolve()`un yeniden yazımı — başlık FIELDS ise liste tabanlı, BAYT-DENK (Tur-80).

    DictReader→DictWriter yolu her satırı sözlüğe çevirip geri serileştiriyordu (85.576 satırda
    1330 → 942 ms). Aynı çıktı: kısa satırın eksik alanı DictReader'da None'dır ve '' yazılır;
    fazla alan DictWriter'da `extrasaction="ignore"` ile düşer; eşleşme anahtarı None/"None"
    anlambilimini korur. Yazılan satırları döndürür; başlık FIELDS değilse None → çağıran eski
    (ad-eşlemeli) yola düşer."""
    with PRED_LOG.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        if next(reader, None) != FIELDS:
            return None
        n = len(FIELDS)
        si, ii, bi, ri = _FI["symbol"], _FI["interval"], _FI["bar_ts"], _FI["resolved_at"]
        cur: list[list[str]] = []
        for row in reader:
            if not row:
                continue
            lr = len(row)
            upd = None
            if not (ri < lr and row[ri]):
                upd = resolutions.get((row[si] if si < lr else None,        # type: ignore[arg-type]
                                       row[ii] if ii < lr else None,
                                       row[bi] if bi < lr else "None"))
            if lr != n:
                row = row[:n] if lr > n else row + [""] * (n - lr)
            if upd:
                for k, v in upd.items():
                    j = _FI.get(k)
                    if j is not None:
                        row[j] = v
            cur.append(row)
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(cur)
    return cur


# Kapı simülasyonu örneklem kapısı: iki tarafta da (tutulan/elenen) bundan az çözülmüş
# aday varsa sonuç 'INSUFFICIENT' etiketlenir — gürültüde desen görme tuzağına karşı.
GATE_SIM_MIN_SIDE = int(os.getenv("GATE_SIM_MIN_SIDE", "30"))


def _side_return(row: dict, side: str | None = None) -> float | None:
    """Çözülmüş satırın İŞLEM TARAFINA göre işaretli getirisi (%). Taraf verilmezse
    `pwin_side` okunur (eski davranış birebir).

    fwd_ret_pct YÖN-işaretlidir (direction ±); pwin ise İŞLEM tarafının (pwin_side)
    kazanma olasılığıdır ve yönle aynı olmak zorunda değildir. Önce ham getiri geri
    çözülür, sonra tarafa göre işaretlenir. pwin_side yoksa None (eski şema satırı —
    tarafı bilinemeyeceği için kapı simülasyonuna DAHİL EDİLMEZ; yaklaşıklık uydurulmaz)."""
    side = (side or row.get("pwin_side") or "").upper()
    if side not in ("LONG", "SHORT"):
        return None
    try:
        fwd = float(row["fwd_ret_pct"])
    except (TypeError, ValueError, KeyError):
        return None
    raw = fwd if row.get("direction") == "+" else -fwd
    return raw if side == "LONG" else -raw


def _pwin_pairs(rows: list[dict]) -> list[tuple[float, float]]:
    """Çözülmüş+pwin'li satırlardan (p, taraf-işaretli getiri %) çiftleri (gate_sim/sweep
    /pwin-kalibrasyon TEK kaynağı)."""
    pairs: list[tuple[float, float]] = []
    for r in rows:
        pw = r.get("pwin")
        if pw is None or pw in ("", "None"):
            continue
        ret = _side_return(r)
        if ret is None:
            continue
        try:
            pairs.append((float(pw), ret))
        except (TypeError, ValueError):
            continue
    return pairs


# Eşik taramasının sabit ızgarası (aktif eşikten bağımsız "ne olurdu" sorusu).
SWEEP_THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def _threshold_sweep(pairs: list[tuple[float, float]], *, min_side: int) -> list[dict]:
    """Her aday eşik için tutulan/elenen istatistiği. İki taraf da >= min_side ise
    uplift_pp yazılır; değilse None (o eşik için canlı hüküm YOK — kenar uydurma yok).
    'Aktif eşik doğru mu, kaç olmalı?' sorusunun veri-bazlı cevabını hazırlar."""
    if not pairs:
        return []
    base_avg = sum(ret for _, ret in pairs) / len(pairs)
    out = []
    for thr in SWEEP_THRESHOLDS:
        kept = [ret for p, ret in pairs if p >= thr]
        cut = [ret for p, ret in pairs if p < thr]
        ks = _bucket_stats(kept)
        uplift = (round(ks["avg_ret_pct"] - base_avg, 4)
                  if len(kept) >= min_side and len(cut) >= min_side else None)
        out.append({"thr": thr, "kept": ks, "cut": _bucket_stats(cut),
                    "uplift_pp": uplift})
    return out


def _bucket_stats(vals: list[float]) -> dict:
    n = len(vals)
    if n == 0:
        return {"n": 0, "hit_rate": None, "avg_ret_pct": None}
    wins = sum(1 for v in vals if v > 0)
    return {"n": n, "hit_rate": round(wins / n, 4),
            "avg_ret_pct": round(sum(vals) / n, 4)}


def gate_sim(rows: list[dict], threshold: float | None = None,
             *, min_side: int = GATE_SIM_MIN_SIDE,
             pairs: list[tuple[float, float]] | None = None) -> dict:
    """CANLI kapı simülasyonu — ACI DERS #3: 'AUC değil, KAPI SİMÜLASYONU karar verir'.

    Çözülmüş + pwin'li satırlarda (sinyal adayları: eşiği GEÇEN LONG/SHORT + kapı-engelli
    WAIT) eşiği uygular: tutulan (p>=thr) / elenen (p<thr) / taban (hepsi) isabet & ort.
    getiri — getiri pwin TARAFINA göre işaretli. Bu, offline train_meta kapı simülasyonunun
    CANLI karşılığıdır: META_GATE açık kalmalı mı sorusunun tek dürüst canlı cevabı.

    KENAR İDDİA ETMEZ: iki tarafta da >= min_side çözülmüş aday yoksa INSUFFICIENT der.
    threshold=None → aktif eşik (META_GATE_THRESHOLD || model eşiği || 0.5)."""
    if threshold is None:
        try:
            from nihai.config import CONFIG
            thr = float(CONFIG.strategy.meta_gate_threshold or 0.0)
            if thr <= 0.0:
                from nihai.signals.meta_gate import get_gate
                thr = float(get_gate().threshold or 0.5)
            threshold = thr
        except Exception:
            threshold = 0.5

    # `pairs` D4 paylaşımıdır (`_read_log_rows` deseninin aynısı): `report()` çiftleri
    # pwin kalibrasyonu için ZATEN kuruyordu ve buraya `rows` geçince AYNI 63k satır
    # ikinci kez taranıyordu (profilin %24'ü). Verilmezse eski davranış BİREBİR.
    if pairs is None:
        pairs = _pwin_pairs(rows)      # [(p, taraf-işaretli getiri %), ...]
    kept = [ret for p, ret in pairs if p >= threshold]
    cut = [ret for p, ret in pairs if p < threshold]

    base = _bucket_stats(kept + cut)
    out = {"threshold": round(float(threshold), 3), "min_side": min_side,
           "n_candidates": base["n"],
           "kept": _bucket_stats(kept), "cut": _bucket_stats(cut), "base": base,
           # Eşik taraması: "aktif eşik yerine X olsaydı?" — her eşik kendi örneklem
           # kapısından geçer; geçemeyen eşiğe uplift YAZILMAZ (None = hüküm yok).
           "sweep": _threshold_sweep(pairs, min_side=min_side)}
    if len(kept) < min_side or len(cut) < min_side:
        out["label"] = "INSUFFICIENT"
        out["text"] = (f"Yetersiz canlı aday: tutulan {len(kept)} / elenen {len(cut)} "
                       f"(kapı ≥{min_side}+{min_side}). Kapı-engelli adaylar da loglandıkça "
                       "birikir; bu eşik altında kapı hakkında canlı hüküm YOK.")
        return out
    uplift = out["kept"]["avg_ret_pct"] - base["avg_ret_pct"]
    out["uplift_pp"] = round(uplift, 4)
    if uplift > 0 and out["kept"]["avg_ret_pct"] > out["cut"]["avg_ret_pct"]:
        out["label"] = "GATE_HELPS"
        out["text"] = (f"Tutulanlar tabandan +{uplift:.3f}pp iyi → kapı canlıda AYIRIYOR "
                       "(örneklemi büyütüp doğrula; offline PIT bulgusuyla birlikte oku).")
    elif uplift < 0:
        out["label"] = "GATE_HURTS"
        # Tur-106: eski öneri "META_GATE'i kapatmayı değerlendir" idi; bu hükmün belgelenmiş cevabı `shadow`
        # kipidir — kapıyı tamamen kapatmak (`false`) bu ölçümün kendisini (pwin kanıt döngüsünü) öldürür.
        out["text"] = (f"Tutulanlar tabandan {uplift:.3f}pp KÖTÜ → kapı canlıda değer "
                       "katmıyor/zarar veriyor (enforce AÇILMAMALI; META_GATE=shadow kanıt döngüsünü yaşatır).")
    else:
        out["label"] = "NEUTRAL"
        out["text"] = "Tutulan ≈ taban → kapı canlıda fark yaratmıyor (gürültü)."
    return out


def _ai_pairs(rows: list[dict]) -> list[tuple[int, float]]:
    """Çözülmüş + AI-verdiktli satırlardan (veto 0/1, taraf-işaretli getiri %) çiftleri.

    Taraf `ai_side`ten okunur (`pwin_side` DEĞİL): META_GATE kapalıyken ikincisi boştur
    ve tarafı oradan okuyan bir analiz sıfır satır bulurdu."""
    pairs: list[tuple[int, float]] = []
    for r in rows:
        v: Any = r.get("ai_veto")     # CSV alanı: serbest metin; dönüşüm try ile korunur
        if v in (None, "", "None"):
            continue
        try:
            veto = int(float(v))
        except (TypeError, ValueError):
            continue
        if veto not in (0, 1):
            continue
        ret = _side_return(r, r.get("ai_side"))
        if ret is None:
            continue
        pairs.append((veto, ret))
    return pairs


def ai_gate_sim(rows: list[dict], *, min_side: int = GATE_SIM_MIN_SIDE) -> dict:
    """İÇ AI KAPI SİMÜLASYONU — "AI_LIVE_MODE=enforce açılmalı mı?" sorusunun tek dürüst
    canlı cevabı (`gate_sim`'in meta-kapı için yaptığının iç AI karşılığı, ACI DERS #3).

    `ai_veto=0` (AI'ın geçireceği) ile `ai_veto=1` (susturacağı) adayların ÇÖZÜLMÜŞ
    taraf-getirilerini ve tabanı (hepsi) karşılaştırır. Kenar İDDİA ETMEZ: iki tarafta da
    >= min_side çözülmüş aday yoksa INSUFFICIENT der.

    DİKKAT — bir uyarı bu kapıya YAPIŞIKTIR: motor bugün %100 ABSTAIN veriyor, yani
    `ai_veto=0` kovası BOŞ kalır ve kapı yapısı gereği INSUFFICIENT döner. Bu bir kusur
    değil ölçümdür: seçici olmayan bir kapı ("her şeyi ele") hiçbir bilgi taşımaz ve
    "tümünü susturmak tabandan iyiydi" cümlesi bir KENAR değil, taban sinyalin kârsız
    olduğunun ifadesidir (ACI DERS #5: darboğaz baz sinyal, model değil)."""
    pairs = _ai_pairs(rows)
    kept = [ret for veto, ret in pairs if veto == 0]
    cut = [ret for veto, ret in pairs if veto == 1]
    base = _bucket_stats(kept + cut)
    out = {"min_side": min_side, "n_candidates": base["n"],
           "kept": _bucket_stats(kept), "cut": _bucket_stats(cut), "base": base}
    if len(kept) < min_side or len(cut) < min_side:
        out["label"] = "INSUFFICIENT"
        out["text"] = (f"Yetersiz canlı aday: AI'ın geçireceği {len(kept)} / susturacağı "
                       f"{len(cut)} (kapı ≥{min_side}+{min_side}). Motor her adaya ABSTAIN "
                       "diyorsa 'geçen' kovası hiç dolmaz — enforce hakkında canlı hüküm YOK.")
        return out
    uplift = out["kept"]["avg_ret_pct"] - base["avg_ret_pct"]
    out["uplift_pp"] = round(uplift, 4)
    if uplift > 0 and out["kept"]["avg_ret_pct"] > out["cut"]["avg_ret_pct"]:
        out["label"] = "AI_GATE_HELPS"
        out["text"] = (f"AI'ın geçirdikleri tabandan +{uplift:.3f}pp iyi → iç AI canlıda "
                       "AYIRIYOR (örneklemi büyüt; zaman VE sembol bölmesinde doğrula).")
    elif uplift < 0:
        out["label"] = "AI_GATE_HURTS"
        out["text"] = (f"AI'ın geçirdikleri tabandan {uplift:.3f}pp KÖTÜ → enforce değer "
                       "katmıyor/zarar veriyor; AI_LIVE_MODE=off/advise'da kalmalı.")
    else:
        out["label"] = "NEUTRAL"
        out["text"] = "Geçen ≈ taban → iç AI canlıda fark yaratmıyor (gürültü)."
    return out


def _calibration_verdict(pairs: list[tuple[float, float]], pw_buckets: dict[str, list[int]], *,
                         min_bucket_n: int = 30, min_buckets: int = 3,
                         min_total: int = 100, konu: str = "olasilik") -> dict:
    """META `pwin` eğrisinin canlı kalibrasyon HÜKMÜ — 2026-07-26 denetimi B-4.

    `konu` (Tur-47) — HÜKMÜN NEYİ YARGILADIĞI. İki ayrı soru vardır ve karıştırılamaz:
      * `"olasilik"` : yargılanan şey bir OLASILIKTIR (pwin/prob_cal). Sıralama SORULUR ve
        ayrıca Brier ile ÖLÇEK sorulur (`brier` ↔ `brier_bilgisiz` 0.25).
      * `"skor"`     : yargılanan şey bir SKORDUR (yön güveni). Yalnız SIRALAMA sorulur;
        **Brier HESAPLANMAZ.** Bir skora Brier uygulamak, onu olasılık sanmanın ta
        kendisidir — Tur-36 §5 bunu ölçtü: `confidence` uzmanının log-kaybı **1,282**,
        taban 0,694 (**1,85 katı**). Yani "yanlış kalibre olasılık" değil, farklı bir
        büyüklük. Etiketler de bu yüzden ayrıdır (`SIRALIYOR` ≠ `TUTARLI`): bir skor
        doğru sıralarken kalibre OLMAYABİLİR ve rapor bunu iddia etmemelidir.

    Neden gerekli: kalibrasyon tablosu ham kova sayılarını basıyordu ama TERSLİĞİ
    isimlendirmiyordu. Ölçülen vaka: Brier **0.2853** (bilgisiz taban 0.25'ten KÖTÜ) ve
    kovalar ters yönlü — model 0.268 derken gerçekleşen %56.1, 0.4375 derken %44.3.
    Bu "kenar yok"un ötesinde bir **ters işaret** sinyalidir ve raporda görünmeliydi.

    Ölçüt (denetim önerisi 8): kova ORTA NOKTASI ile GERÇEKLEŞEN oran arasındaki Pearson
    korelasyonu. Negatifse model olasılıkları ters sıralıyor demektir.

    Örneklem-kapılı (ACI DERS #3 ruhu): yalnız n >= `min_bucket_n` kovalar sayılır; en az
    `min_buckets` kova ve toplam `min_total` gözlem yoksa hüküm VERİLMEZ (`YETERSIZ_ORNEK`)
    — az veriden "ters kalibre" demek yeni bir uydurma olurdu.

    ⛔ Bu hüküm bir İŞLEM ÖNERİSİ DEĞİLDİR: "ters, o hâlde tersine çevir ve kâr et"
    çıkarımı ACI DERS #3/#7 ihlalidir (aleyhte kanıttan kenar türetme). Hüküm yalnız
    ölçüm hijyenini raporlar."""
    skor = konu == "skor"
    usable = [(k, v) for k, v in sorted(pw_buckets.items()) if len(v) >= min_bucket_n]
    n_total = sum(len(v) for _, v in usable)
    out: dict = {"label": "YETERSIZ_ORNEK", "n": n_total,
                 "n_kova": len(usable), "min_kova_n": min_bucket_n, "konu": konu}

    # Brier: kova eşiğinden bağımsız, TÜM pwin'li gözlemler üstünde (bilgisiz taban 0.25).
    # SKOR kipinde HİÇ hesaplanmaz (yukarıdaki `konu` notu) — alan yokluğu bilinçlidir.
    if pairs and not skor:
        ys = [1.0 if ret > 0 else 0.0 for _, ret in pairs]
        out["brier"] = round(sum((p - y) ** 2 for (p, _), y in zip(pairs, ys, strict=True))
                             / len(pairs), 4)
        base_rate = sum(ys) / len(ys)
        # "hep taban oranını söyle" kuralının Brier'i — bilgisiz 0.25'ten daha adil bir taban
        out["brier_taban_orani"] = round(base_rate * (1.0 - base_rate), 4)
        out["brier_bilgisiz"] = 0.25

    if len(usable) < min_buckets or n_total < min_total:
        out["text"] = (f"Kalibrasyon hükmü YOK: {len(usable)} kova / {n_total} gözlem "
                       f"(kapı ≥{min_buckets} kova ve ≥{min_total} gözlem). Biriktikçe ölçülür.")
        return out

    # Kova ORTA NOKTASI — anahtarın ayrıştırılması. Tur-47: eskiden `k.split("-")[0..1]`
    # doğrudan `float()`'a veriliyordu ve NEGATİF anahtarda ("-0.3--0.2" → ['','0.3','','0.2'])
    # `float('')` ile ÇÖKÜYORDU; `/api/predictions/accuracy` 500 verirdi. Bozuk bir kova
    # hükme GİRMEZ (sıralama sorusu onun için tanımsızdır) ama tüm hükmü de DÜŞÜREMEZ.
    orta = [(k, _bucket_mid(k), v) for k, v in usable]
    usable = [(k, v) for k, m, v in orta if m is not None]
    xs = [m for _, m, _ in orta if m is not None]
    if len(usable) != len(orta):           # hükme GİREN kova/gözlem sayısı raporlanır
        out["n_kova"] = len(usable)
        out["n"] = sum(len(v) for _, v in usable)
    if len(usable) < min_buckets:
        out["text"] = (f"Kalibrasyon hükmü YOK: ayrıştırılabilir {len(usable)} kova "
                       f"(kapı ≥{min_buckets}). Bozuk kova anahtarı hükme girmez.")
        return out
    ys_b = [sum(v) / len(v) for _, v in usable]
    mx, my = sum(xs) / len(xs), sum(ys_b) / len(ys_b)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys_b, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys_b)
    corr = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0
    out["korelasyon"] = round(corr, 3)
    # YAYILIM kapısı: Pearson ölçek-bağımsızdır — gerçekleşen oranlar 0.48/0.50/0.49/0.50
    # gibi DÜZ olsa bile ufacık tutarlı bir eğim ρ≈+0.9 verir ve "tutarlı" görünür.
    # Kalibre bir eğrinin oranları büyüklük olarak da AYRIŞMALI. (Bu kapı, kendi testinde
    # yakalandı — bir metriğin yeşil görünürken gerçeğin düz olması ACI DERS #8'in şeklidir.)
    yayilim = max(ys_b) - min(ys_b)
    out["yayilim"] = round(yayilim, 3)

    brier = out.get("brier")
    kotu_brier = isinstance(brier, float) and brier > out.get("brier_bilgisiz", 0.25)
    brier_eki = f"; Brier {brier} > bilgisiz taban 0.25." if kotu_brier else "."
    if yayilim < 0.05 and corr > -0.3:
        out["label"] = "SIRALAMIYOR" if skor else "ZAYIF"
        out["text"] = (f"{'Skor' if skor else 'Kalibrasyon'} DÜZ: gerçekleşen oranların kovalar "
                       f"arası yayılımı yalnız {yayilim:.3f} (< 0.05) — "
                       f"{'skor' if skor else 'olasılıklar'} sonucu AYIRMIYOR (ρ={corr:+.2f} "
                       f"bu ölçekte anlamsız)" + brier_eki)
    elif corr <= -0.3:
        out["label"] = "TERS_SIRALAMA" if skor else "TERS_KALIBRASYON"
        out["text"] = (
            (f"TERS SIRALAMA: kova orta noktası ↔ gerçekleşen oran korelasyonu {corr:+.2f} "
             f"(negatif) — skor YÜKSELDİKÇE isabet DÜŞÜYOR. Skoru 'yüksek = iyi' diye "
             f"gösteren her yüzey (renk, sıralama, eşik) operatörü ölçüme AYKIRI yönlendirir."
             if skor else
             f"TERS KALİBRASYON: kova orta noktası ↔ gerçekleşen oran korelasyonu {corr:+.2f} "
             f"(negatif) — model olasılıkları TERS sıralıyor" + brier_eki)
            + " Bu bir kenar DEĞİLDİR (tersine çevirme yasak, ACI DERS #3/#7);"
            + (" skorun yönü güvenilmezdir, çaresi TERSİNE ÇEVİRMEK değil İDDİA ETMEMEKTİR."
               if skor else " modelin canlı olasılık üretimi güvenilmezdir."))
    elif corr < 0.3:
        out["label"] = "SIRALAMIYOR" if skor else "ZAYIF"
        out["text"] = (f"{'Skor' if skor else 'Kalibrasyon'} ZAYIF: korelasyon {corr:+.2f} "
                       f"(|ρ| < 0.30) — {'skor' if skor else 'olasılıklar'} gerçekleşen oranı "
                       f"sıralamıyor" + brier_eki)
    elif skor:
        # ⚠️ "SIRALIYOR" ≠ "kalibre": bir skor doğru sıralarken ölçeği tamamen bozuk olabilir
        # (olasılık gibi okunamaz). Bu ayrım etikete yazılıdır ki pano onu %'ye çevirmesin.
        out["label"] = "SIRALIYOR"
        out["text"] = (f"Skor SIRALIYOR: korelasyon {corr:+.2f} — yüksek skor daha yüksek "
                       f"isabetle gidiyor. Bu bir OLASILIK olduğu anlamına GELMEZ (ölçek "
                       f"ölçülmedi); %'ye çevrilip kalibre gibi sunulamaz.")
    else:
        out["label"] = "TUTARLI"
        out["text"] = (f"Kalibrasyon tutarlı: korelasyon {corr:+.2f}"
                       + (f" AMA Brier {brier} > bilgisiz taban 0.25 (sıralama doğru, "
                          f"ölçek bozuk)." if kotu_brier else "."))
    return out


def _read_log_rows() -> list[dict]:
    """predictions_log.csv'yi BİR KEZ ham oku (D4, Tur-10): web uçları report()+coverage()
    arasında paylaşır — istek başına 2-3 tam-CSV okuması 1'e iner. Dosya yoksa [].

    Tur-80: `resolve()` az önce yazdığı satırları devrettiyse ve dosya imzası o yazımdan beri
    DEĞİŞMEDİYSE ayrıştırma yapılmaz — yazılan dosya FIELDS başlığı + tam genişlikte satırlardır,
    DictReader'ın döndüreceği sözlük `dict(zip(FIELDS, satır))`ın ta kendisidir. Devir tek seferliktir."""
    global _ROWS_HANDOFF
    devir, _ROWS_HANDOFF = _ROWS_HANDOFF, None
    if not PRED_LOG.exists():
        return []
    if devir is not None and devir[0] is not None and devir[0] == _file_sig():
        return [dict(zip(FIELDS, r, strict=True)) for r in devir[1]]
    with PRED_LOG.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# 0.0-0.1 … 0.9-1.0 — kova anahtarları ÖNCEDEN kurulur. Eskiden her satırda iki `:.1f`
# biçimlendirme + birleştirme yapılıyordu: 63k satırda 63k biçimlendirme, oysa ayrık
# anahtar YALNIZ 10 tane. (Tur-33 "aynı ağırlığı iki kez hesaplama" dersinin string hâli.)
_BUCKET_KEYS = tuple(f"{i / 10:.1f}-{i / 10 + 0.1:.1f}" for i in range(10))


def _bucket_key(x: float) -> str:
    """`x` → "0.5-0.6" biçiminde 0.1'lik kova anahtarı (kalibrasyon tabloları TEK kaynağı).

    x=1.0 → "0.9-1.0" (hayalet '1.0-1.1' kovası olmasın). NEGATİF/bozuk girdide eski
    biçimlendirme yolu KORUNUR: `_BUCKET_KEYS[-3]` sessizce "0.7-0.8" verirdi ve dejenere
    bir satır geçerli bir kovaya karışırdı (Tur-33: dejenere girdi sessiz yanlı çıktı
    üretir). Bu dal bir hız kaybı değil bir DENKLİK koşuludur."""
    i = int(x * 10)
    if i < 0:
        lo = i / 10.0
        return f"{lo:.1f}-{lo + 0.1:.1f}"
    return _BUCKET_KEYS[i if i < 9 else 9]


_BUCKET_MIDS = {k: (i / 10 + 0.05) for i, k in enumerate(_BUCKET_KEYS)}


def _bucket_mid(key: str) -> float | None:
    """Kova anahtarı → ORTA NOKTA; TANINMAYAN anahtarda **None** (`_bucket_key`'in tersi).

    Tur-47: hüküm fonksiyonu bunu `float(k.split("-")[0])` ile yapıyordu ve NEGATİF bir
    anahtarda ("-0.3--0.2" → `['', '0.3', '', '0.2']`) `float('')` ile ÇÖKÜYORDU — bozuk
    tek bir CSV satırı `/api/predictions/accuracy` ucunu 500'e düşürebilirdi. Ayrıştırma
    bir tahmin DEĞİL bir TERS FONKSİYONDUR: anahtarı KURAN tablo onu ÇÖZER de. Bu modülün
    üretmediği bir anahtarın orta noktası YOKTUR (uydurulmaz) ve o kova hükme girmez —
    dejenere bir kova için "sıralama" sorusu zaten tanımsızdır."""
    return _BUCKET_MIDS.get(key)


def report(rows: list[dict] | None = None) -> dict:
    """Çözülmüş tahminleri özetle: canlı isabet oranı + güven-kovası kalibrasyon + beklenti
    + meta-kapı canlı simülasyonu ('meta_gate' anahtarı; pwin'li aday satırlar üzerinde).

    rows: önceden okunmuş HAM log satırları (D4 — çağıran tek okuyup coverage() ile
    paylaşabilir). None → dosyadan okur (CLI/eski davranış birebir)."""
    if rows is None:
        if not PRED_LOG.exists():
            return {"resolved": 0, "msg": "henüz kayıt yok"}
        rows = _read_log_rows()
    rows = [r for r in rows if r.get("hit") in ("0", "1")]
    if not rows:
        return {"resolved": 0, "msg": "henüz çözülmüş tahmin yok (önce --resolve)"}

    n = len(rows)
    hits = sum(int(r["hit"]) for r in rows)
    rets = [float(r["fwd_ret_pct"]) for r in rows]
    hit_rate = hits / n
    avg_ret = sum(rets) / n
    # Tur-83: BERABERLİK (fiyat ufukta hiç kıpırdamadı) `hit`te 0 yazılır — yön ne doğru ne
    # yanlıştır, ama ıska sayılır. Yazı-tura tahmincinin beklenen `hit_rate`i bu yüzden 0,5
    # değil (1 − beraberlik)/2'dir; 0,5'e kıyaslayan hüküm yazı-turayı "ters" okumaya kayar.
    # Ölçüldü: defterde %0,90 (ONDS'ta %7,9). `hit_rate` DEĞİŞMEZ (pano + tarihçe sürekliliği);
    # 0,5 boş hipotezine kıyaslanacak sayı beraberlik HARİÇ olandır.
    ties = sum(1 for v in rets if v == 0.0)

    # Güven kovaları → canlı kalibrasyon: tahmin güveni gerçek isabeti öngörüyor mu?
    buckets: dict[str, list[int]] = {}
    for r in rows:
        try:
            c = float(r.get("confidence"))     # type: ignore[arg-type]  # None → TypeError
        except (ValueError, TypeError):
            continue
        buckets.setdefault(_bucket_key(c), []).append(int(r["hit"]))

    # pwin kalibrasyonu: meta model 0.6 diyorsa TARAF gerçekte ~%60 mı kazanıyor?
    # (yukarıdaki 'buckets' YÖN-forecast güvenini ölçer; bu, META modelin canlı eğrisi.)
    pw_buckets: dict[str, list[int]] = {}
    pw_pairs = _pwin_pairs(rows)
    for p, ret in pw_pairs:
        pw_buckets.setdefault(_bucket_key(p), []).append(1 if ret > 0 else 0)

    return {
        "resolved": n, "hit_rate": round(hit_rate, 4),
        "tie_share": round(ties / n, 4),
        "hit_rate_excl_ties": (round(hits / (n - ties), 4) if n > ties else None),
        "avg_signed_ret_pct": round(avg_ret, 4),
        "buckets": {k: {"n": len(v), "hit_rate": round(sum(v) / len(v), 3)}
                    for k, v in sorted(buckets.items())},
        "pwin_calibration": {k: {"n": len(v), "win_rate": round(sum(v) / len(v), 3)}
                             for k, v in sorted(pw_buckets.items())},
        # B-4: kalibrasyon eğrisinin HÜKMÜ (ters/zayıf/tutarlı) — tablo tersliği
        # isimlendirmiyordu, denetimde bu yüzden gözden kaçmıştı.
        "pwin_verdict": _calibration_verdict(pw_pairs, pw_buckets),
        # Tur-47: AYNI kusur KARDEŞ tabloda 10 tur boyunca AÇIK kaldı. `buckets` (yön
        # güveni → isabet) B-4'ten beri hükümsüz basılıyordu; oysa terslik `config.py`
        # yorumunda YAZILIYDI ("canlı veride confidence şu an isabetle TERS korele").
        # Bir kod YORUMU çalışma-zamanı hükmü DEĞİLDİR: operatör panoyu okur, yorumu değil.
        # `pairs=[]` bilinçli: güven bir SKORDUR, Brier'i hesaplanmaz (bkz. `konu` notu).
        "confidence_verdict": _calibration_verdict([], buckets, konu="skor"),
        "meta_gate": gate_sim(rows, pairs=pw_pairs),
    }


def coverage(rows: list[dict] | None = None) -> dict:
    """Doğrulama harness'inin OPERASYONEL durumu (salt-okunur): log büyüyor mu, ne kadar
    veri birikti, hangi zaman aralığını kapsıyor. 'report()' İSABETİ özetler; bu fonksiyon
    KAPSAMA'yı (kaç tahmin, çözülmüş/bekleyen, gün aralığı) verir → operatör forward-test'in
    gerçekten ilerlediğini görür.

    rows: önceden okunmuş HAM log satırları (D4 — report() ile paylaşılan tek okuma).
    None → dosyadan okur (eski davranış birebir)."""
    if rows is None:
        if not PRED_LOG.exists():
            return {"tracking_file": False, "total": 0, "resolved": 0, "pending": 0,
                    "first_logged": None, "last_logged": None, "days_span": 0.0, "symbols": 0}
        rows = _read_log_rows()
    total = len(rows)
    resolved = sum(1 for r in rows if r.get("hit") in ("0", "1"))
    unresolvable = sum(1 for r in rows if r.get("resolved_at") == "UNRESOLVABLE")  # Tur-12
    logged = [r["logged_at"] for r in rows if r.get("logged_at")]
    first = min(logged) if logged else None
    last = max(logged) if logged else None
    days = 0.0
    if first and last:
        try:
            days = round(
                (datetime.fromisoformat(last) - datetime.fromisoformat(first))
                .total_seconds() / 86400.0, 2)
        except ValueError:
            days = 0.0
    symbols = len({r["symbol"] for r in rows if r.get("symbol")})
    return {"tracking_file": True, "total": total, "resolved": resolved,
            "pending": total - resolved - unresolvable, "unresolvable": unresolvable,
            "first_logged": first, "last_logged": last,
            "days_span": days, "symbols": symbols}


# D4b (Tur-28) — özet ÖNBELLEĞİ. Web uçları her istekte 6,7 MB / 53k satırı yeniden
# ayrıştırıyordu (ölçüldü: _read_log_rows 163 ms + report 142 ms + coverage 49 ms = ~354 ms,
# ve log büyüdükçe DOĞRUSAL artıyor). Anahtar = PRED_LOG imzası (mtime_ns, size): dosya
# değişmediyse aynı satırlar aynı özeti verir, değiştiyse (record/resolve yazdı) yeniden
# hesaplanır. HAM SATIRLAR SAKLANMAZ — 53k satır 70,3 MB tutuyor (ölçüldü); saklanan yalnız
# ~3 KB'lık türetilmiş özet. Aynı imza-cache deseni: _SEEN_KEYS / _load_calibration.
_SUMMARY: tuple[tuple[int, int] | None, dict[str, Any]] | None = None


def summary(*, force: bool = False) -> dict[str, Any]:
    """{'report','coverage','verdict'} — TEK okumadan, PRED_LOG imzasıyla önbellekli.

    Dosya değişmediği sürece tekrar çağırmak bedavaya yakındır (tek `stat()`); değiştiği anda
    kendiliğinden tazelenir, yani bayat özet DÖNMEZ. `force=True` önbelleği atlar.
    Not: dönen dict PAYLAŞILANDIR — çağıran üstüne alan ekleyecekse `dict(...)` ile kopyalasın."""
    global _SUMMARY, _ROWS_HANDOFF
    sig = _file_sig()
    if not force and _SUMMARY is not None and _SUMMARY[0] == sig:
        # Özet bu imzada zaten var → bekleyen satır devri gereksizdir ve BELLEKTE TUTULMAZ
        # (Codex incelemesi: araya giren bir okuyucu önbelleği doldurunca devir tüketilmiyordu).
        _ROWS_HANDOFF = None
        return _SUMMARY[1]
    rows = _read_log_rows() if PRED_LOG.exists() else None
    rep = report(rows)
    out: dict[str, Any] = {"report": rep, "coverage": coverage(rows), "verdict": verdict(rep)}
    _SUMMARY = (sig, out)
    return out


def verdict(rep: dict) -> dict:
    """report() özetinden makine-okunur DÜRÜST verdict (CLI + web ortak kullanır).

    label: NO_DATA | NO_EDGE | WEAK_POSITIVE | INVERTED. Eşik: |isabet-0.5|<0.03 → öngörü
    gücü yok (offline ~%50 bulgusunu doğrular). İsabeti İDDİA etmez, ÖLÇER.
    Tur-83: 0,5'e kıyaslanan isabet BERABERLİK HARİÇ olandır (bkz. report() notu); dönen
    `hit_rate` hükmün kullandığı sayıdır, `tie_share` onunla birlikte taşınır."""
    n = rep.get("resolved", 0)
    if not n:
        return {"label": "NO_DATA", "edge_pp": 0.0, "hit_rate": None,
                "text": "Henüz çözülmüş tahmin yok (önce --resolve / piyasa açık olmalı)."}
    hx = rep.get("hit_rate_excl_ties")
    hr = float(hx if hx is not None else rep["hit_rate"])
    edge = abs(hr - 0.5)
    if edge < 0.03:
        label = "NO_EDGE"
        text = "İsabet ~%50 → canlıda da öngörü gücü YOK (offline bulguyu doğrular)."
    elif hr > 0.5:
        label = "WEAK_POSITIVE"
        text = (f"İsabet >%50 ({edge * 100:.1f}pp üstün) → canlıda zayıf-pozitif sinyal "
                "(örneklemi büyüt, doğrula).")
    else:
        label = "INVERTED"
        # TUR-47: bu dal eskiden *"sinyali tersine çevirmek bile düşünülebilir"* diyordu —
        # yani AYNI MODÜLDEKİ `_calibration_verdict`in AÇIKÇA YASAKLADIĞI çıkarımı ÖNERİYORDU
        # ("Bu bir kenar DEĞİLDİR, tersine çevirme yasak, ACI DERS #3/#7"). İki hüküm
        # fonksiyonu aynı veride ÇELİŞİK konuşuyordu ve yanlış olanı okuyan operatör tuzağa
        # düşerdi — ACI DERS #9'un kendi ifadesiyle: *"bir kural iki kapıdan yalnız birine
        # konursa..."*. Etiket (ÖLÇÜM) kalır, ÖNERİ kalkar.
        text = ("İsabet <%50 → yön ölçümü ters tarafta. ⛔ Bu bir 'tersine çevir ve kâr et' "
                "sinyali DEĞİLDİR (ACI DERS #3/#7). Bu kural bu projede ZATEN ÖLÇÜLDÜ "
                "(EDGE_FINDINGS §17-D): fade OOS'ta isabeti %54,6'ya çıkardı AMA ortalama "
                "getiri −0,071% kaldı — isabet beklenti DEĞİLDİR. Okunacak şey ölçüm "
                "hijyenidir, işlem yönü değil.")
    return {"label": label, "edge_pp": round(edge * 100, 1), "hit_rate": round(hr, 4),
            "tie_share": rep.get("tie_share"), "text": text}


def _print_report(rep: dict) -> None:
    if rep.get("resolved", 0) == 0:
        print(f"[track] {rep.get('msg', 'veri yok')}")
        return
    print("=" * 60)
    print("CANLI FORECAST DOĞRULAMA (paper, ileriye-bakmasız)")
    print("=" * 60)
    print(f"Çözülmüş tahmin   : {rep['resolved']}")
    print(f"Yön isabet oranı  : {rep['hit_rate'] * 100:.1f}%   (50% = yazı-tura)")
    print(f"Ort. yön-ayarlı getiri: {rep['avg_signed_ret_pct']:+.3f}%/tahmin")
    print("-" * 60)
    print("Güven kovası → gerçek isabet (kalibrasyon):")
    for k, v in rep["buckets"].items():
        print(f"  güven {k}: n={v['n']:4}  isabet={v['hit_rate'] * 100:5.1f}%")
    gs = rep.get("meta_gate")
    if gs:
        print("-" * 60)
        print(f"META-KAPI canlı simülasyonu (eşik {gs['threshold']}, aday n={gs['n_candidates']}):")
        if gs["label"] == "INSUFFICIENT":
            print(f"  [{gs['label']}] {gs['text']}")
        else:
            for name in ("kept", "cut", "base"):
                b = gs[name]
                tag = {"kept": "tutulan (p>=thr)", "cut": "elenen (p<thr) ",
                       "base": "taban (hepsi)  "}[name]
                print(f"  {tag}: n={b['n']:4}  isabet={b['hit_rate'] * 100:5.1f}%  "
                      f"ort={b['avg_ret_pct']:+.3f}%")
            print(f"  [{gs['label']}] {gs['text']}")
        sweep = [s for s in gs.get("sweep", []) if s["uplift_pp"] is not None]
        if sweep:
            print("  Eşik taraması (örneklem-kapılı; '+' = tutulan taban üstünde):")
            for s in sweep:
                print(f"    thr={s['thr']:.2f}: tut n={s['kept']['n']:4} "
                      f"ort={s['kept']['avg_ret_pct']:+.3f}%  uplift={s['uplift_pp']:+.3f}pp")
        pc = rep.get("pwin_calibration")
        if pc:
            print("-" * 60)
            print("META P(win) kovası → gerçek taraf-kazanma (canlı model kalibrasyonu):")
            for k, v in pc.items():
                print(f"  pwin {k}: n={v['n']:4}  kazanma={v['win_rate'] * 100:5.1f}%")
            pv = rep.get("pwin_verdict") or {}
            if pv.get("text"):
                brier = pv.get("brier")
                if brier is not None:
                    print(f"  Brier={brier} (bilgisiz taban 0.25 · taban-oranı kuralı "
                          f"{pv.get('brier_taban_orani')})")
                print(f"  → {pv.get('label')}: {pv['text']}")
    print("-" * 60)
    print(f"VERDİKT: {verdict(rep)['text']}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Canlı forecast kapalı-döngü doğrulama")
    ap.add_argument("--resolve", action="store_true", help="vadesi gelen tahminleri etiketle")
    ap.add_argument("--report", action="store_true", help="canlı isabet/kalibrasyon özeti")
    args = ap.parse_args()
    if not (args.resolve or args.report):
        ap.print_help()
        return
    if args.resolve:
        res = resolve()
        print(f"[track] resolve: {res}")
    if args.report:
        _print_report(report())


if __name__ == "__main__":
    main()
