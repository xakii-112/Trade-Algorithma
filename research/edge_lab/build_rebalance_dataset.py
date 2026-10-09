"""
build_rebalance_dataset.py — Endeks rebalans / akış olayları aday veri seti
(edge_lab BACKLOG.md #4). KARAR-DIŞI/ARAŞTIRMA.

Bu script KENAR İDDİASI YAPMAZ — yalnız veri taşır; hüküm `harness.run_candidate`'ın
beş kapısınındır (purged_cv → gate_time → gate_symbol → dsr → exec_lag_1).

## Hipotez

**H1 (olay):** S&P 500'e EKLENME / ÇIKARILMA çevresinde endeks-fonu akışı geçici fiyat
baskısı yaratır (Shleifer 1986 "downward sloping demand curve"; Harris & Gurel 1986).
**H2 (takvim):** ay-sonu / çeyrek-sonu yeniden dengeleme akışı kesitte kısa vadeli getiri
asimetrisi üretir.

**Koşudan ÖNCE yazılmış dürüst ön-beklenti: DÜŞÜK.** Petajisto (2011) ve sonrası literatür
2000'lerden itibaren ekleme etkisinin büyük ölçüde söndüğünü ölçüyor (arbitrajcıların
olayı önceden fiyatlaması + endeks fonlarının uygulama esnekliği). H2 için de takvim
etkileri kamuya en açık, dolayısıyla en çok arbitrajlanmış anomali sınıfıdır. Bu beklenti
SONUC raporuna da yazılmıştır (koşudan önce yazılmış beklenti, sonradan rasyonalizasyondan
daha değerlidir — lead-lag turunda bu disiplin işe yaradı).

## Veri

* Fiyat: `nihai.data.alpaca_history.get_history(sym, "1d")` — Alpaca SIP GÜNLÜK barlar,
  **düzeltilmiş** (`ALPACA_HIST_ADJUSTMENT=all`, aşağıdaki kilide bak).
* Üyelik: `nihai.data.pit_universe.sp500_as_of` (point-in-time) + Wikipedia değişiklik
  tablosu (Added/Removed/Effective Date).
* Evren: pencerenin BAŞINDAKİ PIT üyelik ∪ pencere içinde EKLENEN her sembol. Bu, günlük
  PIT anlık görüntülerinin birleşimine EŞİTTİR ve bugün endekste olmayan (satın alınan/
  iflas eden/çıkarılan) isimleri İÇERİR → survivorship-temiz (ACI DERS #1).

## TASARIM KİLİTLERİ (ihlal = ölçüm çöp)

**K0 — DÜZELTİLMİŞ BARLAR ZORUNLU.** `alpaca_history._cache_path` düzeltme kipini dosya
adına koyar (`all` → `SYM_1d_all.csv`, `raw` → `SYM_1d.csv`). `raw` dosyalara düşmek
bölünme günlerinde sahte ~%50 getiri üretir (BULGU-19: GE 1:8 ters bölünme; PIT evrenin
%8,5'i etkileniyordu) ve ölçüm baştan çöp olur. Bu yüzden env burada AÇIKÇA kilitlenir
(`_lock_adjusted_data_env`), varsayılana güvenilmez.

**K1 — ANONS ≠ EFFECTIVE DATE (en kritik).** Elimizdeki değişiklik tablosu **yürürlük
(effective) tarihi** verir, ANONS tarihini DEĞİL; anons tipik olarak günler öncedir ve o
veri elimizde YOKTUR. Dolayısıyla **"kaç gün SONRA endeks değişikliği olacak" türü
İLERİ-bakan olay özelliği YASAKTIR** — anons henüz kamuya açık değilken bilgi sızdırır
(lead-lag turundaki "lider hindsight'sız" kilidinin birebir aynısı). Yalnız GERİYE-bakan
olay mesafesi kullanılır: `gun_since_eklendi`, `gun_since_cikarildi`, üst sınır
`EVENT_CAP_DAYS`. Takvim özellikleri (`ay_sonuna_gun`, `ceyrek_sonuna_gun`) bu yasaktan
MUAFTIR: takvim önceden ve kamuya bilinir, ileri-bakan takvim mesafesi meşrudur.

**K2 — PIT üyelik.** `endeks_uyesi` her barda o barın TARİHİNDEKİ üyeliktir. "Bugün
endekste olanlar" listesi geçmişe taşınmaz. Uygulama: `sp500_as_of(start)` ile TEK bir
temel anlık görüntü alınır, sonra değişiklik tablosu İLERİ sarılır (aynı semantik, ~1400
ağ çağrısı yerine 2). `test_ileri_sarma_sp500_as_of_ile_ayni_uyelik` bu eşdeğerliği kilitler.

**K2b — EVREN de point-in-time (bu turun ek kilidi).** Bir sembol panele ancak endekse
GİRDİĞİ tarihten itibaren alınır; girişinden ÖNCEKİ barları panele KONMAZ. Gerekçe:
"bu hisse ileride S&P 500'e alınacak" bilgisi karar anında YOKTUR — giriş öncesi barları
eklemek, evren seçimine gizli bir look-ahead sokar (ekleme öncesi ralli literatürün en
bilinen etkisidir; tam da uydurulmuş kenar üretecek yer). Çıkarılan isimler ise panelde
KALIR — çıkarma karar anında bilinen bir olaydır ve bu isimler survivorship düzeltmesinin
ta kendisidir. NOT: şartname bu turda "girişten önceki barlarda `endeks_uyesi=0`" testi
istemişti; koşudan ÖNCE alınan bu tasarım kararı gereği o barlar hiç üretilmez, testin
yerini daha SIKI bir PIT kilidi alır (girişten önce SATIR YOK + çıkarmadan sonra
`endeks_uyesi=0`). Sapma SONUC raporunda da yazılıdır.

**K3 — Look-ahead yok.** Tüm özellikler yalnız karar barının KAPANIŞINA kadarki veriden.
Yuvarlanan istatistikler (`dolar_hacim_z`, `vol_20b`, `ret_5b`) mevcut barı İÇERİR — bu
meşrudur (kapanış karar anında bilinir) ve etikete karışmaz. Etiket/PnL YALNIZ
`shift(-horizon)` ile gelecekten gelir ve hiçbir özellik kolonuna girmez.
`test_gelecek_barlar_bozulunca_ozellikler_degismez` bunu kanıtlar.

**K4 — GERÇEK `pnl_col`.** Dataset ikili `label`'ın YANINDA gerçek getiri kolonu (`pnl`,
yüzde) taşır ve `run_candidate`'a `pnl_col="pnl"` ile verilir. Tur-24'e kadar
`harness._shift_label_1bar` yalnız etiketi kaydırıyordu; gerçek `pnl_col` verilen ilk
adayda sahte PASS üretecekti. Hata düzeltildi — bu, gerçek `pnl_col` veren İLK adaydır,
yani ACI DERS #7 kapısı (exec_lag_1) ilk kez tam anlamıyla ısırır.

**K5 — Tek ufuk, tek eşik.** Birincil ufuk 5 işlem barı (projenin `horizon_bars=5`
konvansiyonu). Ufuk süpürmesi YAPILMAZ (Tur-19 zaten süpürdü, hiçbir ufukta kenar yok;
süpürme yalnız çoklu-test yükü ekler). Eşik taraması da yapılmaz — `harness` DSR'yi
`n_trials` ile zaten deflate eder.

**K6 — Özellik sayısı KÜÇÜK (8).** ~800k satırlık panelde bile özellik enflasyonu overfit
üretir. Çekirdek liste `FEATURE_COLS`'tur ve genişletilmemiştir.

**K7 — Örneklem gerçekçiliği.** Endeks ekleme/çıkarma olayı bu pencerede ~100+100
civarındadır; olay-pencereli alt küme DSR kapısını geçemeyecek kadar küçük olurdu, bu
yüzden TAM PANEL üstünde çalışılır (olay özellikleri çoğu satırda sınır değerini alır).
Gerçek olay sayısı yine de sayılır ve rapora yazılır (§26 dersi: kapı PASS dese bile
`n_kept` okunur).

## Özellikler (8 — tümü `t` barının KAPANIŞINA kadarki veriden)

  gun_since_eklendi  — sembolün EN SON endekse eklenme yürürlük tarihinden bu yana geçen
                       gün; olay yoksa/`EVENT_CAP_DAYS`'i aşarsa sınır değeri (GERİYE-bakan)
  gun_since_cikarildi— aynısı, çıkarma olayı için (GERİYE-bakan)
  endeks_uyesi       — o barın tarihindeki PIT üyelik (1/0)
  ay_sonuna_gun      — ay sonuna kalan TAKVİM günü (takvim → ileri-bakan meşru, K1 muafiyeti)
  ceyrek_sonuna_gun  — çeyrek sonuna kalan takvim günü (aynı muafiyet)
  dolar_hacim_z      — dolar hacminin (close×volume) son 20 barlık z-skoru (geriye-bakan)
  ret_5b             — son 5 barlık getiri
  vol_20b            — son 20 barlık getiri std'si

## Etiket ve PnL

  fwd   = close[t+H]/close[t] − 1            (H = --horizon, varsayılan 5)
  label = 1 if fwd > COST_THRESHOLD else 0   (5 bps maliyet eşiği)
  pnl   = 100 × (fwd − COST_THRESHOLD)       (YÜZDE, maliyet düşülmüş LONG getirisi)

`label == 1` ⟺ `pnl > 0` (test kilitler). PnL sözde değil GERÇEK getiridir (K4).

CSV şeması: ts,symbol,<8 özellik>,label,pnl

Kullanım:
    python -m research.edge_lab.build_rebalance_dataset --out data/edge_lab_rebalance_v1.csv
    python -m research.edge_lab.build_rebalance_dataset --max-symbols 60   # boru hattı dumanı
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from nihai.console import ensure_utf8_stdout

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Sabitler --------------------------------------------------------------------------
DEFAULT_START = "2021-01-01"
DEFAULT_END = "2026-07-31"
DEFAULT_HORIZON = 5          # K5: TEK ufuk (projenin horizon_bars konvansiyonu)
COST_BPS = 5.0
COST_THRESHOLD = COST_BPS / 10_000.0
EVENT_CAP_DAYS = 60          # K1: geriye-bakan olay mesafesi üst sınırı (gün)
VOL_WINDOW = 20              # hacim z / oynaklık penceresi (bar)
RET_WINDOW = 5               # ret_5b penceresi (bar)
MIN_HISTORY_BARS = VOL_WINDOW + 1   # en geniş pencere + 1 bar getiri ısınması

FEATURE_COLS: list[str] = [
    "gun_since_eklendi", "gun_since_cikarildi", "endeks_uyesi",
    "ay_sonuna_gun", "ceyrek_sonuna_gun", "dolar_hacim_z", "ret_5b", "vol_20b",
]
SCHEMA_COLS: list[str] = ["ts", "symbol", *FEATURE_COLS, "label", "pnl"]

DEFAULT_OUT = ROOT / "data" / "edge_lab_rebalance_v1.csv"

_NS_PER_DAY = 86_400_000_000_000


def _lock_adjusted_data_env() -> None:
    """K0 — bölünme/temettü DÜZELTİLMİŞ barları ve ön-koşul cache'ini kilitle.

    NEDEN `ALPACA_HIST_ADJUSTMENT=all`: `alpaca_history._cache_path` düzeltme kipini
    dosya adına yazar. Kip `raw` olursa okunacak dosyalar `SYM_1d.csv` (DÜZELTMESİZ)
    olur ve bölünme günlerinde sahte ~%50 getiri ölçülür — ölçümün tamamı çöp olur
    (BULGU-19). Varsayılan bugün zaten 'all' ama bu script'in geçerliliği buna
    BAĞIMLI olduğu için varsayıma bırakılmaz.

    NEDEN `ALPACA_HIST_CACHE_MAX_AGE_DAYS`: ön-koşul verisi (604 × `*_1d_all.csv`)
    bilerek bir kez çekildi. Varsayılan tazelik 1 GÜN'dür; gece yarısını geçen bir
    koşuda tüm cache "bayat" sayılır ve 600+ sembol ağdan yeniden çekilir (429
    fırtınası → sessizce sembol düşer = evren daralır). Tarihsel bir pencere için
    tazelik anlamsızdır; `alpaca_history` docstring'i de bunu açıkça tavsiye eder.

    Modül seviyesinde import'lardan SONRA çağrılması güvenlidir: `alpaca_history`
    her iki değişkeni de ÇAĞRI anında okur (`_adjustment()` / `_read_cache`);
    import anında okunan tek env `ALPACA_HIST_DIR`'dir ve ona dokunulmaz.
    """
    os.environ.setdefault("ALPACA_HIST_ADJUSTMENT", "all")
    os.environ.setdefault("ALPACA_HIST_CACHE_MAX_AGE_DAYS", "3650")


_lock_adjusted_data_env()


def _norm_ticker(t: object) -> str:
    """Ticker normalizasyonu — `nihai.data.pit_universe._norm` ile BİREBİR aynı sözleşme
    (nokta → tire; Alpaca'ya giderken `_to_alpaca_symbol` geri çevirir). Kopyalanmasının
    nedeni özel bir isme bağlanmamak; `test_norm_ticker_pit_universe_ile_paritetli`
    parite bozulursa kırılır."""
    return str(t).strip().upper().replace(".", "-")


def _valid_ticker(t: str) -> bool:
    """`sp500_as_of`'un uyguladığı ayıklama ile aynı: boş / 'NAN' / 6 karakterden uzun
    değerler Wikipedia tablosunun artığıdır, ticker değildir."""
    return bool(t) and t != "NAN" and len(t) <= 6


# ---------------------------------------------------------------------------------------
# Endeks takvimi — PIT üyelik + GERİYE-bakan olay tarihleri (K1/K2/K2b)
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class IndexTimeline:
    """Pencere boyunca S&P 500 üyeliğinin point-in-time gösterimi.

    `base_members` = `base_date`'teki üyelik (sp500_as_of'tan). `add_dates`/`remove_dates`
    = sembol → yürürlük tarihleri (ARTAN sıralı, yalnız pencere içi). Üyelik herhangi bir
    `d` tarihinde temel kümeye `d`'ye KADARKİ (dahil) olayların ileri sarılmasıyla bulunur —
    `sp500_as_of(d)` ile aynı semantik (o da `date > d` olanları geri sarar)."""

    base_date: pd.Timestamp
    base_members: frozenset[str]
    add_dates: dict[str, list[pd.Timestamp]] = field(default_factory=dict)
    remove_dates: dict[str, list[pd.Timestamp]] = field(default_factory=dict)

    # --- evren ---------------------------------------------------------------------
    def universe(self) -> list[str]:
        """Pencerede bir kez bile endekste bulunmuş TÜM semboller (bugün endekste
        olmayanlar DAHİL = survivorship-temiz)."""
        return sorted(set(self.base_members) | set(self.add_dates))

    def panel_start(self, symbol: str) -> pd.Timestamp | None:
        """K2b — sembolün panele girdiği tarih: temel kümedeyse pencere başı, değilse
        pencere içindeki İLK eklenme tarihi. Hiç üye olmamışsa None (satır üretilmez)."""
        if symbol in self.base_members:
            return self.base_date
        adds = self.add_dates.get(symbol)
        return adds[0] if adds else None

    # --- PIT üyelik ----------------------------------------------------------------
    def membership(self, symbol: str, dates: pd.DatetimeIndex) -> np.ndarray:
        """`dates`'in her elemanı için PIT üyelik (1/0). Vektörel: olay tarihleri üstünde
        `searchsorted` ile "son olayın durumu", olay yoksa temel durum."""
        events = [(d, 0) for d in self.remove_dates.get(symbol, [])]
        events += [(d, 1) for d in self.add_dates.get(symbol, [])]
        base = 1 if symbol in self.base_members else 0
        if not events:
            return np.full(len(dates), base, dtype=float)
        # Aynı gün hem çıkarma hem ekleme (pratikte görülmez) → deterministik sıra:
        # önce çıkarma, sonra ekleme (net sonuç 'üye').
        events.sort(key=lambda e: (e[0], e[1]))
        ev_ns = np.array([e[0].value for e in events], dtype="int64")
        ev_state = np.array([e[1] for e in events], dtype=float)
        pos = np.searchsorted(ev_ns, _as_ns(dates), side="right") - 1
        return np.where(pos >= 0, ev_state[np.clip(pos, 0, None)], float(base))

    # --- GERİYE-bakan olay mesafesi (K1) -------------------------------------------
    def days_since(self, symbol: str, dates: pd.DatetimeIndex, *, kind: str,
                   cap: int = EVENT_CAP_DAYS) -> np.ndarray:
        """`dates`'in her elemanı için "en son <kind> olayından bu yana geçen gün", `cap`
        ile üstten sınırlanmış. Olay YOKSA veya olay `cap` günden eskiyse `cap` döner —
        yani özellik "yakında olay yok" nötr değerine oturur.

        **İLERİ-bakan varyantı YASAK (K1):** gelecekteki bir yürürlük tarihine olan mesafe,
        anons tarihi elimizde olmadığı için henüz kamuya açık olmayan bilgi olurdu."""
        table = self.add_dates if kind == "eklendi" else self.remove_dates
        dts = table.get(symbol)
        if not dts:
            return np.full(len(dates), float(cap))
        ev_ns = np.array([d.value for d in dts], dtype="int64")
        d_ns = _as_ns(dates)
        pos = np.searchsorted(ev_ns, d_ns, side="right") - 1
        gun = (d_ns - ev_ns[np.clip(pos, 0, None)]) / _NS_PER_DAY
        out = np.where(pos >= 0, gun, float(cap))
        return np.minimum(out, float(cap)).astype(float)

    # --- olay sayaçları (rapor için) ------------------------------------------------
    def event_counts(self) -> dict[str, int]:
        return {
            "ekleme_olayi": sum(len(v) for v in self.add_dates.values()),
            "cikarma_olayi": sum(len(v) for v in self.remove_dates.values()),
            "ekleme_sembolu": len(self.add_dates),
            "cikarma_sembolu": len(self.remove_dates),
        }


def _as_ns(dates: pd.DatetimeIndex) -> np.ndarray:
    """DatetimeIndex → int64 nanosaniye (birim-bağımsız; Tur-10 dersi: `asi8` birim
    bağımlıdır, `as_unit('ns')` ile sabitlenir)."""
    return np.asarray(pd.DatetimeIndex(dates).as_unit("ns").asi8, dtype="int64")


def build_timeline(changes: pd.DataFrame, base_members: list[str], base_date: pd.Timestamp,
                   end: pd.Timestamp) -> IndexTimeline:
    """Değişiklik tablosundan (`date`/`added`/`removed`) pencere-içi olay sözlüklerini kur.

    Yalnız `base_date < date <= end` aralığındaki değişiklikler alınır: `base_date`'e KADAR
    olanlar zaten `base_members`'ın içinde (o küme `sp500_as_of(base_date)`'ten geldi),
    tekrar uygulanırsa üyelik bozulur."""
    adds: dict[str, list[pd.Timestamp]] = {}
    rems: dict[str, list[pd.Timestamp]] = {}
    if not changes.empty:
        win = changes[(changes["date"] > base_date) & (changes["date"] <= end)]
        win = win.sort_values("date", kind="mergesort")
        for _, row in win.iterrows():
            d = pd.Timestamp(row["date"]).normalize()
            for col, table in (("added", adds), ("removed", rems)):
                if col not in row or pd.isna(row.get(col)):
                    continue
                sym = _norm_ticker(row[col])
                if _valid_ticker(sym):
                    table.setdefault(sym, []).append(d)
    members = frozenset(s for s in (_norm_ticker(m) for m in base_members) if _valid_ticker(s))
    return IndexTimeline(base_date=pd.Timestamp(base_date).normalize(), base_members=members,
                         add_dates=adds, remove_dates=rems)


def load_timeline(start: str | pd.Timestamp, end: str | pd.Timestamp, *,
                  as_of_fn: Callable[[str], list[str]] | None = None,
                  changes_fn: Callable[[], tuple[list[str], pd.DataFrame]] | None = None,
                  ) -> IndexTimeline:
    """Canlı kaynaklardan zaman çizgisi (2 ağ çağrısı: temel anlık görüntü + değişiklikler).

    `pit_universe` LAZY import edilir (modül import'u ağ/ağır zincir tetiklemesin) ve
    `_current_and_changes` bilerek kullanılır: paket, olay YÜRÜRLÜK TARİHLERİNİ veren
    başka bir genel arayüz sunmuyor (`removed_since` yalnız isim döndürür). Testler bu
    iki fonksiyonu enjekte eder → ağ'a çıkılmaz."""
    if as_of_fn is None or changes_fn is None:
        from nihai.data.pit_universe import _current_and_changes, sp500_as_of
        as_of_fn = as_of_fn or sp500_as_of
        changes_fn = changes_fn or _current_and_changes
    base_date = pd.Timestamp(start).normalize()
    base_members = as_of_fn(base_date.strftime("%Y-%m-%d"))
    _, changes = changes_fn()
    return build_timeline(changes, base_members, base_date, pd.Timestamp(end).normalize())


# ---------------------------------------------------------------------------------------
# Veri çekimi — alpaca_history (savunmacı: hata → None, koşu durmaz)
# ---------------------------------------------------------------------------------------
def fetch_bars(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Sembolün GÜNLÜK OHLCV barları (`ALPACA_HIST_ADJUSTMENT=all` kilitli, K0).
    Hata/boş → None (çağıran sembolü sayarak atlar)."""
    _lock_adjusted_data_env()   # doğrudan çağrı yolunda da garanti (K0)
    try:
        from nihai.data.alpaca_history import get_history
    except Exception as e:  # pragma: no cover - import-zamanı arıza
        logger.debug(f"[rebalance] alpaca_history import edilemedi: {e}")
        return None
    try:
        df = get_history(symbol, "1d", start=start, end=end)
    except Exception as e:
        logger.warning(f"[rebalance] {symbol} çekim hatası: {e}")
        return None
    if df is None or df.empty:
        return None
    return df


# ---------------------------------------------------------------------------------------
# Sembol-başı satır üretimi (leak-free)
# ---------------------------------------------------------------------------------------
def build_symbol_rows(symbol: str, bars: pd.DataFrame, timeline: IndexTimeline, *,
                      horizon: int = DEFAULT_HORIZON,
                      cost_threshold: float = COST_THRESHOLD,
                      stats: dict[str, int] | None = None) -> pd.DataFrame | None:
    """Bir sembolün barlarından aday satırları üret. Yetersiz veri → None.

    Özelliklerin TAMAMI `t` barının kapanışına kadarki veriden (K3); etiket/PnL yalnız
    `shift(-horizon)` ile gelecekten gelir ve hiçbir özelliğe karışmaz."""
    if bars is None or bars.empty or not {"close", "volume"}.issubset(bars.columns):
        return None
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    if len(bars) < MIN_HISTORY_BARS + horizon:
        return None

    panel_start = timeline.panel_start(symbol)
    if panel_start is None:
        return None

    d_norm = pd.DatetimeIndex(bars.index).normalize()
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)

    # --- fiyat/hacim özellikleri (GERİYE-bakan; mevcut bar DAHİL = kapanışta bilinir) ---
    ret1 = close.pct_change()
    ret_5b = close.pct_change(RET_WINDOW)
    vol_20b = ret1.rolling(VOL_WINDOW).std()
    dv = close * volume
    dv_std = dv.rolling(VOL_WINDOW).std().replace(0.0, np.nan)
    dolar_hacim_z = (dv - dv.rolling(VOL_WINDOW).mean()) / dv_std

    # --- takvim özellikleri (K1 muafiyeti: takvim önceden ve kamuya bilinir) ---
    ay_sonu = d_norm.to_period("M").to_timestamp(how="end").normalize()
    ceyrek_sonu = d_norm.to_period("Q").to_timestamp(how="end").normalize()
    ay_sonuna_gun = (ay_sonu - d_norm).days.to_numpy(dtype=float)
    ceyrek_sonuna_gun = (ceyrek_sonu - d_norm).days.to_numpy(dtype=float)

    # --- olay özellikleri (YALNIZ geriye-bakan; K1) ---
    gun_ekle = timeline.days_since(symbol, d_norm, kind="eklendi")
    gun_cikar = timeline.days_since(symbol, d_norm, kind="cikarildi")
    uye = timeline.membership(symbol, d_norm)

    # --- etiket + GERÇEK pnl (K4) ---
    fwd = close.shift(-horizon) / close - 1.0
    fwd_np = fwd.to_numpy(dtype=float)
    # `NaN > eşik` False döner (NaN değil) → son barlarda sahte label=0 sızardı; NaN'ı
    # AÇIKÇA koru (build_leadlag/build_overnight'taki aynı tuzak).
    label = np.where(np.isnan(fwd_np), np.nan, (fwd_np > cost_threshold).astype(float))
    pnl = 100.0 * (fwd_np - cost_threshold)

    out = pd.DataFrame({
        "ts": d_norm,
        "symbol": symbol,
        "gun_since_eklendi": gun_ekle,
        "gun_since_cikarildi": gun_cikar,
        "endeks_uyesi": uye,
        "ay_sonuna_gun": ay_sonuna_gun,
        "ceyrek_sonuna_gun": ceyrek_sonuna_gun,
        "dolar_hacim_z": dolar_hacim_z.to_numpy(dtype=float),
        "ret_5b": ret_5b.to_numpy(dtype=float),
        "vol_20b": vol_20b.to_numpy(dtype=float),
        "label": label,
        "pnl": pnl,
    })

    n_bar = len(out)
    # K2b: endekse girmeden ÖNCEKİ barlar panele KONMAZ (evren look-ahead'i).
    out = out[out["ts"] >= panel_start]
    n_pre = n_bar - len(out)
    before_na = len(out)
    ozellik_na = out[FEATURE_COLS].isna().any(axis=1)
    sonuc_na = out[["label", "pnl"]].isna().any(axis=1)
    n_isinma = int((ozellik_na & ~sonuc_na).sum())
    n_ufuk = int(sonuc_na.sum())
    out = out.dropna(subset=[*FEATURE_COLS, "label", "pnl"]).reset_index(drop=True)

    if stats is not None:
        stats["dusen_panel_oncesi"] = stats.get("dusen_panel_oncesi", 0) + n_pre
        stats["dusen_isinma"] = stats.get("dusen_isinma", 0) + n_isinma
        stats["dusen_ufuk_kuyrugu"] = stats.get("dusen_ufuk_kuyrugu", 0) + n_ufuk
        stats["dusen_toplam"] = stats.get("dusen_toplam", 0) + (n_bar - len(out))
        stats["_okunan_bar"] = stats.get("_okunan_bar", 0) + n_bar
        stats["_panel_ici_bar"] = stats.get("_panel_ici_bar", 0) + before_na

    if out.empty:
        return None
    out["label"] = out["label"].astype(int)
    return out[SCHEMA_COLS]


def _even_slice(items: list[str], k: int | None) -> list[str]:
    """Deterministik, ALFABE BOYUNCA yayılmış alt küme (`--max-symbols`; yalnız boru hattı
    dumanı içindir). `items[:k]` alfabenin başına yığılırdı."""
    if k is None or k <= 0 or k >= len(items):
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in dict.fromkeys(idx.tolist())]


# ---------------------------------------------------------------------------------------
# Panel inşası
# ---------------------------------------------------------------------------------------
def build_dataset(*, start: str = DEFAULT_START, end: str = DEFAULT_END,
                  horizon: int = DEFAULT_HORIZON, max_symbols: int | None = None,
                  timeline: IndexTimeline | None = None,
                  fetch_fn: Callable[[str, str, str], pd.DataFrame | None] | None = None,
                  ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Tam paneli kur. `timeline`/`fetch_fn` testlerde ağ'a çıkmadan enjekte edilir.

    Dönüş: (dataset, sayaçlar). Sayaçlar rapora ham sayı taşır — düşürülen satırların
    NEDENİ sayılarak raporlanır (şartname C1)."""
    tl = timeline if timeline is not None else load_timeline(start, end)
    fetch = fetch_fn or fetch_bars
    universe = _even_slice(tl.universe(), max_symbols)

    stats: dict[str, int] = {"evren_sembol": len(universe), **tl.event_counts()}
    frames: list[pd.DataFrame] = []
    for i, sym in enumerate(universe, 1):
        bars = fetch(sym, start, end)
        if bars is None or bars.empty:
            stats["veri_alinamadi"] = stats.get("veri_alinamadi", 0) + 1
            logger.debug(f"[rebalance] {sym}: veri alınamadı — atlanıyor")
            continue
        rows = build_symbol_rows(sym, bars, tl, horizon=horizon, stats=stats)
        if rows is None or rows.empty:
            stats["yetersiz_gecmis"] = stats.get("yetersiz_gecmis", 0) + 1
            logger.debug(f"[rebalance] {sym}: yetersiz geçmiş/panel-dışı — atlanıyor")
            continue
        frames.append(rows)
        if i % 100 == 0:
            logger.info(f"[rebalance] {i}/{len(universe)} sembol işlendi "
                        f"({sum(len(f) for f in frames)} satır)")

    if not frames:
        stats["uretilen_sembol"] = 0
        stats["satir"] = 0
        return pd.DataFrame(columns=SCHEMA_COLS), stats

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)
    stats["uretilen_sembol"] = int(out["symbol"].nunique())
    stats["satir"] = len(out)
    stats["uye_satir"] = int((out["endeks_uyesi"] > 0.5).sum())
    stats["uye_disi_satir"] = int((out["endeks_uyesi"] <= 0.5).sum())
    stats["ekleme_penceresi_satir"] = int((out["gun_since_eklendi"] < EVENT_CAP_DAYS).sum())
    stats["cikarma_penceresi_satir"] = int((out["gun_since_cikarildi"] < EVENT_CAP_DAYS).sum())
    return out, stats


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Endeks rebalans/akış olayları aday veri seti (edge_lab #4)")
    ap.add_argument("--start", default=DEFAULT_START, help="pencere başı (YYYY-MM-DD)")
    ap.add_argument("--end", default=DEFAULT_END, help="pencere sonu (YYYY-MM-DD)")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                    help="etiket ufku (işlem barı) — K5: süpürme YAPMA")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max-symbols", type=int, default=None,
                    help="yalnız boru hattı dumanı için deterministik alt küme")
    args = ap.parse_args(argv)

    t0 = datetime.now(UTC)
    logger.info(f"[rebalance] pencere {args.start} → {args.end}, ufuk={args.horizon} bar, "
                f"düzeltme={os.environ.get('ALPACA_HIST_ADJUSTMENT')}")
    df, stats = build_dataset(start=args.start, end=args.end, horizon=args.horizon,
                              max_symbols=args.max_symbols)

    logger.info(f"[rebalance] evren {stats.get('evren_sembol', 0)} sembol | "
                f"veri yok {stats.get('veri_alinamadi', 0)} | "
                f"yetersiz geçmiş {stats.get('yetersiz_gecmis', 0)} | "
                f"üretilen {stats.get('uretilen_sembol', 0)}")
    logger.info(f"[rebalance] olay: {stats.get('ekleme_olayi', 0)} ekleme "
                f"({stats.get('ekleme_sembolu', 0)} sembol) / "
                f"{stats.get('cikarma_olayi', 0)} çıkarma "
                f"({stats.get('cikarma_sembolu', 0)} sembol)")
    logger.info(f"[rebalance] düşen satır: panel-öncesi {stats.get('dusen_panel_oncesi', 0)} | "
                f"ısınma {stats.get('dusen_isinma', 0)} | "
                f"ufuk kuyruğu {stats.get('dusen_ufuk_kuyrugu', 0)}")
    logger.info(f"[rebalance] satır {stats.get('satir', 0)} | "
                f"üye {stats.get('uye_satir', 0)} / üye-dışı {stats.get('uye_disi_satir', 0)} | "
                f"ekleme penceresi {stats.get('ekleme_penceresi_satir', 0)} | "
                f"çıkarma penceresi {stats.get('cikarma_penceresi_satir', 0)}")
    if not df.empty:
        logger.info(f"[rebalance] taban oran (label=1): {df['label'].mean():.4f} | "
                    f"ort pnl {df['pnl'].mean():+.4f}% | "
                    f"tarih {df['ts'].min()} → {df['ts'].max()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"[rebalance] yazıldı: {out_path} "
                f"({(datetime.now(UTC) - t0).total_seconds():.1f} sn)")
    return 0


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    raise SystemExit(main())
