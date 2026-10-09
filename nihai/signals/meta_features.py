"""
meta_features.py — Meta-labeling için özellik çıkarımı + işlem kaydedici.

Fikir (López de Prado, "Advances in Financial ML", meta-labeling): birincil model
(strategy.generate_signal) YÖN üretir; ikincil bir model "bu sinyal kâr edecek mi?"
olasılığını tahmin eder ve düşük-olasılıkları eler. Bu modül ikincil modelin EĞİTİM
VERİSİNİ üretir: her girişte look-ahead-güvenli özellik vektörü + nihai sonuç (kâr/zarar).

Tüm özellikler KARAR barına (df_slice'ın son barı) kadarki veriden hesaplanır →
look-ahead yok (backtest motoru zaten df.iloc[:i+1] dilimiyle çağırır).

Kullanım backtest motoruyla: TradeRecorder örneğini backtest_symbol(recorder=...)'a ver;
giriş anında on_open, kapanışta on_close çağrılır ve eşleştirilmiş satır birikir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nihai.signals.strategy import (
    SignalAction,
    _SharedIndicators,
    _vix_close_before,
)

# Sıralı, sabit özellik isimleri — eğitim/çıkarım arası tutarlılık için TEK kaynak.
# 2026-07-03 genişletme (19→25): canlı özellik logu henüz doğmadan eklendi → geriye dönük
# boş-kolon kirliliği yok. Eski modeller kendi kayıtlı 'features' listesiyle çalışmaya
# devam eder (features_to_vector names param — train/serve hizası bozulmaz); eski
# dataset CSV'leri ensure_feature_columns ile nötr-doldurularak okunabilir.
FEATURE_NAMES = [
    "adx14", "rsi14", "atr_pct", "atr_regime", "rvol",
    "ema_gap", "px_vs_ema50", "px_vs_ema200",
    "ret_5", "ret_20", "dist_hi20", "dist_lo20", "bb_width",
    "gap_pct", "range_pct", "close_pos", "streak", "rsi_slope",
    "rr_ratio", "quality_score", "confluence", "is_long",
    "vix", "dow", "hour",
]

# Özellik başına NÖTR varsayılan (eksik gösterge/eksik kolon = "bilgi yok" değeri).
# TEK kaynak: extract_features'ın boş-df guard'ı ve ensure_feature_columns aynı tabloyu
# kullanır → iki yol asla ıraksamaz.
NEUTRAL_DEFAULTS: dict[str, float] = {
    "rsi14": 50.0, "atr_regime": 1.0, "rvol": 1.0, "confluence": 1.0, "close_pos": 0.5,
}


def neutral_default(name: str) -> float:
    return NEUTRAL_DEFAULTS.get(name, 0.0)


# VIX yokluk-değeri (F6, 2026-07-26 denetimi P1-6): canlıda VIX önbelleği boşken özellik
# 0.0 yazılıyordu ve model bunu GEÇERLİ bir VIX gibi görüyordu — 0 volatilite fiziksel
# olarak imkânsız, üstelik eğitim setindeki en uç değerin de dışında (dağılım-dışı girdi).
# Eksik = "bilgi yok" → uzun-dönem tipik seviyeye (≈18) düş. Sabit: canlı/eğitim aynı yolu
# kullanır, sadece 0/None vakasını kapsar; gerçek VIX değerleri hiç dokunulmadan geçer.
_VIX_NEUTRAL = 18.0


def _vix_or_neutral(vix: float | None) -> float:
    """VIX yoksa/0/negatif/NaN ise nötr seviye; aksi halde değerin kendisi."""
    v = _safe(vix if vix is not None else np.nan, 0.0)
    return v if v > 0 else _VIX_NEUTRAL


def _safe(x: float, default: float = 0.0) -> float:
    """NaN/inf → default (model NaN yiyemez; eksik gösterge nötr sayılır)."""
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def extract_features(df_slice: pd.DataFrame, signal, *, shared=None) -> dict[str, float]:
    """Karar barındaki bağlamı sayısal vektöre çevir. df_slice son barı = karar barı.

    signal: strategy.Signal (rr_ratio, quality_score, action, reasons içerir).
    Look-ahead-güvenli: yalnızca df_slice (i. bara kadar) kullanılır.

    shared: opsiyonel `strategy._SharedIndicators`/`_SharedSlice` (Tur-7 önbelleği).
        BULGU-43-1: bu fonksiyon Tur-7'nin O(n²)→O(n) önbelleğini kullanan TEK
        kardeş çağırandı ki KULLANMIYORDU — EMA/ATR/BB/RSI/ADX'i büyüyen dilim
        üzerinde SIFIRDAN hesaplıyordu. Ölçüldü: 4000 barlık backtest süresinin
        **%49'u** buradaydı. Önbellek verilirse aynı seriler yeniden kullanılır
        (Tur-7 parite argümanı: nedensel gösterge + kırpılmış görünüm = bit-denk).
        None ise dilimin KENDİSİ üzerinde önbellek kurulur → eski davranış birebir
        (aynı fonksiyonlar, aynı argümanlar) ve tek çağrı içinde de tekrar-hesap yok.

        SESSİZ LOOK-AHEAD KORUMASI: verilen görünümün bar sayısı df_slice'ınkinden
        FARKLIYSA önbellek YOK SAYILIR ve dilim üzerinde yeniden kurulur. Yanlış
        hizalanmış bir görünüm geleceği sızdırırdı; bu kapı onu imkânsız kılar
        (`tests/test_tur43_paylasimli_onbellek.py` kilitler)."""
    # Savunmacı guard: boş/sütunsuz df → nötr varsayılan vektör (IndexError yerine).
    # Üretimde meta_gate çağrıyı try/except'le sarar; bu, doğrudan çağıranları da korur.
    if df_slice is None or df_slice.empty or "close" not in df_slice.columns:
        return {n: neutral_default(n) for n in FEATURE_NAMES}
    if shared is None or getattr(shared, "nbars", -1) != len(df_slice):
        shared = _SharedIndicators(df_slice)

    # Tur-49: gövde bar başına ~20 pandas skaler/pencere okuması yapıyordu ve profilde
    # (Tur-43+45 sonrası) backtest süresinin **%33'ü** buradaydı. Tur-43 HESABI önbelleğe
    # aldı; bu tur OKUMAYI numpy'a indiriyor. Okunan seriler AYNI seriler, konumlar aynı.
    # `mean()`/`max()`/`min()`/`count()` yerine `~np.isnan` maskesi kullanılır — pandas'ın
    # skipna=True varsayılanının BİREBİR karşılığı.
    c = shared.closes()
    n = c.size
    price = _safe(c[-1], 1.0) or 1.0

    e20 = shared.ema_at(20, -1)
    e50 = shared.ema_at(50, -1)
    e200 = shared.ema_at(200, -1) if n >= 200 else np.nan
    atr_np = shared.atr14_np()
    atr_now = atr_np[-1]
    atr_ok = ~np.isnan(atr_np)
    if int(atr_ok.sum()) >= 5:
        w = atr_np[-50:]
        w = w[~np.isnan(w)]
        atr_avg = float(w.mean()) if w.size else np.nan
    else:
        atr_avg = atr_now

    vol = shared.col_np("volume")
    if vol is not None and vol.size >= 21:
        w = vol[-21:-1]
        w = w[~np.isnan(w)]
        avg_v = float(w.mean()) if w.size else 0.0
        rvol = float(vol[-1]) / avg_v if avg_v > 0 else 1.0
    else:
        rvol = 1.0

    if n >= 21:
        hh20, ll20 = shared.breakout_at(-1, 20)   # == high.iloc[-21:-1].max() / low.min()
    else:
        hh20 = ll20 = price
    ma, upper, lower = shared.bb20_np()
    bb_w = (upper[-1] - lower[-1]) / ma[-1] if ma[-1] else 0.0

    ret_5 = price / _safe(c[-6], price) - 1 if n >= 6 else 0.0
    ret_20 = price / _safe(c[-21], price) - 1 if n >= 21 else 0.0

    # --- Karar-barı mikro-yapısı (2026-07-03 genişletmesi; hepsi yalnız df_slice'tan) ---
    _hi_np, _lo_np, _op_np = (shared.col_np("high"), shared.col_np("low"),
                              shared.col_np("open"))
    hi = _safe(_hi_np[-1], price) if _hi_np is not None else price
    lo = _safe(_lo_np[-1], price) if _lo_np is not None else price
    op = _safe(_op_np[-1], price) if _op_np is not None else price
    prev_c = _safe(c[-2], price) if n >= 2 else price
    gap_pct = (op - prev_c) / prev_c if prev_c else 0.0          # bar açılış boşluğu
    range_pct = (hi - lo) / price if price else 0.0              # bar genişliği (oynaklık)
    close_pos = (price - lo) / (hi - lo) if hi > lo else 0.5     # kapanış bar içinde nerede [0,1]
    # Ardışık aynı-yön kapanış serisi (işaretli, ±10 kırpma): momentum tükenmişliği sinyali.
    # Döngü |streak|>=10'da kırılır → en çok 11 yineleme, yani en çok SON 12 kapanış
    # okunur. Eski gövde bunu `c.iloc[k]` ile yapıyordu: yineleme başına iki pandas
    # skaler getitem (Tur-33 "sıkı döngüde genel-amaçlı sarmalayıcı" dersi). Kuyruk bir
    # kez numpy'a alınır; `_safe` NaN/inf→0.0 eşlemesi burada np.isfinite ile birebir.
    _tail = np.where(np.isfinite(c[-12:]), c[-12:], 0.0)
    streak = 0
    for k in range(_tail.size - 1, 0, -1):
        d = float(_tail[k]) - float(_tail[k - 1])
        if d == 0 or (streak != 0 and (d > 0) != (streak > 0)) or abs(streak) >= 10:
            break
        streak += 1 if d > 0 else -1
    rsi_np = shared.rsi14_np()
    rsi_now = _safe(rsi_np[-1], 50.0)
    rsi_prev5 = _safe(rsi_np[-6], rsi_now) if rsi_np.size >= 6 else rsi_now
    rsi_slope = rsi_now - rsi_prev5                              # momentumun momentumu

    idx = df_slice.index[-1]
    dow = float(idx.dayofweek) if hasattr(idx, "dayofweek") else 0.0
    hour = float(getattr(idx, "hour", 0) or 0)                   # gün-içi saat etkisi (1d→0)
    vix = _vix_close_before(idx.date() if hasattr(idx, "date") else None)

    confluence = 1
    for r in getattr(signal, "reasons", []) or []:
        if isinstance(r, str) and r.startswith("CONFLUENCE_"):
            try:
                confluence = int(r.split("_", 1)[1])
            except ValueError:
                pass

    feats = {
        "adx14": _safe(shared.adx_last()),
        "rsi14": rsi_now,
        "atr_pct": _safe(atr_now / price),
        "atr_regime": _safe(atr_now / atr_avg, 1.0) if atr_avg else 1.0,
        "rvol": _safe(rvol, 1.0),
        "ema_gap": _safe((e20 - e50) / price),
        "px_vs_ema50": _safe((price - e50) / price),
        "px_vs_ema200": _safe((price - e200) / price) if np.isfinite(e200) else 0.0,
        "ret_5": _safe(ret_5),
        "ret_20": _safe(ret_20),
        "dist_hi20": _safe((price - hh20) / price),
        "dist_lo20": _safe((price - ll20) / price),
        "bb_width": _safe(bb_w),
        "gap_pct": _safe(gap_pct),
        "range_pct": _safe(range_pct),
        "close_pos": _safe(close_pos, 0.5),
        "streak": float(streak),
        "rsi_slope": _safe(rsi_slope),
        "rr_ratio": _safe(getattr(signal, "rr_ratio", 0.0)),
        "quality_score": _safe(getattr(signal, "quality_score", 0.0)),
        "confluence": float(confluence),
        "is_long": 1.0 if getattr(signal, "action", None) == SignalAction.LONG else 0.0,
        "vix": _vix_or_neutral(vix),
        "dow": dow,
        "hour": hour,
    }
    return feats


def ensure_feature_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Eski dataset CSV'lerini güncel FEATURE_NAMES şemasına yükselt (okuma anında).

    FEATURE_NAMES genişleyince (örn. 19→25) eski CSV'ler eksik kolonda KeyError verirdi.
    Eksik özellik kolonları NÖTR varsayılanla eklenir, mevcut kolonlardaki NaN'lar da
    nötrle doldurulur (model NaN yiyemez). Dönüş: (df, eklenen_kolonlar) — çağıran eklenen
    varsa uyarı basar (eski veride yeni özellik sinyal taşımaz; dürüst raporlama)."""
    added: list[str] = []
    for n in FEATURE_NAMES:
        if n not in df.columns:
            df[n] = neutral_default(n)
            added.append(n)
        else:
            df[n] = pd.to_numeric(df[n], errors="coerce").fillna(neutral_default(n))
    return df, added


def features_to_vector(feats: dict[str, float], names: list[str] | None = None) -> list[float]:
    """Özellik sözlüğünü sabit SIRAYA göre düzleştir.

    `names` verilirse o sıra kullanılır — bu, modelle birlikte kaydedilen EĞİTİM-ANI
    feature listesidir (train_meta.py `{"features": FEATURE_NAMES}` yazar). Çıkarımda
    bunu geçmek train/serve hizasını garanti eder: kod `FEATURE_NAMES`'i sonradan
    yeniden sıralasa/değiştirse bile vektör modelin beklediği sırada kurulur (aynı
    uzunlukta yeniden-sıralama sklearn'de hata vermez → sessiz yanlış olasılık riski).
    `names=None` ise güncel `FEATURE_NAMES` (eğitim verisi üretimi için tek kaynak)."""
    return [_safe(feats.get(n, 0.0)) for n in (names or FEATURE_NAMES)]


# ---------------------------------------------------------------------------
# İşlem kaydedici — backtest motorunun giriş/çıkışını eşleştirir
# ---------------------------------------------------------------------------
class TradeRecorder:
    """backtest_symbol(recorder=...) ile kullanılır.

    on_open(pos, df_slice, signal): giriş anında özellikleri pozisyon kimliğiyle stash'le.
    on_close(ce): kapanışta nihai PnL ile birleştir, satırı (özellikler+etiket) ekle.

    Pozisyon nesnesi açılıştan kapanışa aynı kalır (CloseEvent.pos == aynı nesne) →
    id(pos) ile güvenli eşleşme."""

    def __init__(self, symbol: str = "", period: str = ""):
        self.symbol = symbol
        self.period = period
        self.rows: list[dict] = []
        self._pending: dict[int, dict | None] = {}  # id(pos) -> feats

    def on_open(self, pos, df_slice: pd.DataFrame, signal) -> None:
        try:
            self._pending[id(pos)] = extract_features(df_slice, signal)
        except Exception:
            self._pending[id(pos)] = None  # bozuk bar → atlanır (on_close'da düşer)

    def on_close(self, ce) -> None:
        feats = self._pending.pop(id(ce.pos), None)
        if not feats:
            return
        row = dict(feats)
        row["pnl"] = float(ce.realized_pnl)
        row["label"] = 1 if ce.realized_pnl > 0 else 0
        row["symbol"] = self.symbol or ce.pos.symbol
        row["period"] = self.period
        row["entry_time"] = getattr(ce.pos, "entry_time", "")
        row["bars_held"] = ce.bars_held
        row["mae_pct"] = ce.mae_pct
        row["mfe_pct"] = ce.mfe_pct
        self.rows.append(row)
