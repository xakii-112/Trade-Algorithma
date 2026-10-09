"""
harness.py — Kenar-keşif ana motoru (Tur-16 WP-E). KARAR-DIŞI/ARAŞTIRMA.

`run_candidate(name, dataset_csv, feature_cols, label_col, ...)` bir aday sinyal-ailesinin
CSV veri setini SIRAYLA dört sert kapıdan geçirir (CLAUDE.md ACI DERSLER'in doğrudan
uygulaması):

  1. **purged_cv**    — purge+embargolu K-fold OOS-AUC (research.purged_cv.purged_folds
     UYGUNSA import edilir; değilse aynı sözleşmeli minimal yerel purge-split'e düşülür).
     Ortalama OOS-AUC eşiği aşmalı (ACI-2/ACI-6: AUC~0.50 = kenar yok).
  2. **gate_time / gate_symbol** — train_meta.evaluate deseninin GENELLEŞTİRİLMİŞ hali:
     lojistik regresyon eğitilir, olasılık-eşiği filtresi (kapı) UYGULANMAMIŞ (BASE) ile
     UYGULANMIŞ (kapı) expectancy'i KIYASLANIR. Hem ZAMAN-bölmesinde hem SEMBOL-bölmesinde
     (hiç görülmemiş semboller) kapı > base olmalı — tek bölmede iyi görünmek yetmez
     (ACI-3: dayanıklılık = ikisinde de çıkmak). **Tur-24'ten beri iki EK koşul** (§26'nın
     "kapı PASS dese bile n_kept okunmalı" kuralının makineye çevrilmesi, bkz.
     `_min_kept_required` / `_welch_t`): kapı-sonrası işlem sayısı test diliminin ORANSAL
     tabanını da geçmeli ve kapı-E ile elenen-E farkı örneklem gürültüsünü (Welch t) aşmalı.
  3. **dsr**           — research.deflated.dsr (Bailey & López de Prado): kapı-sonrası getiri
     serisinin Deflated Sharpe'ı, eşik-taramasının imlediği çoklu-test sayısıyla düzeltilerek.
  4. **exec_lag_1**    — ACI-7'nin doğrudan uygulaması: etiketler 1 bar (sembol-içi kronolojik)
     KAYDIRILIP kapı YENİDEN ölçülür. Kenar 1-bar sonra da PASS kalmıyorsa muhtemelen
     zamanlama artefaktıdır (bkz. pairs_test.py --exec-lag deseni, EDGE_FINDINGS §22).

Verdikt METNİ hiçbir koşulda "kenar var" YAZMAZ (`_FORBIDDEN_PHRASE` — testlerle korunur);
TÜM kapılar geçilse bile en iyi ihtimalle `PASS_VERDICT` ("canlı gölge doğrulaması gerekli")
döner. Herhangi bir kapı FAIL ise `FAIL_VERDICT`. Bu, "yanlış-pozitif ÜRETMEME" ilkesinin
kod-düzeyinde garantisidir (WP-E dürüstlük çerçevesi).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from research.deflated import dsr, sharpe_estimator_var, sharpe_moments
from research.edge_lab import ledger

_PurgedFoldsFn = Callable[..., Iterator[tuple[np.ndarray, np.ndarray]]]
_purged_folds_impl: _PurgedFoldsFn | None = None
try:
    from research.purged_cv import purged_folds as _purged_folds_impl
except ImportError:  # pragma: no cover - purged_cv ağır zincirde bir bağımlılık kaybederse
    pass

# Verdikt metninde ASLA geçmemesi gereken ifade (dürüstlük guard'ı — testlerle korunur).
_FORBIDDEN_PHRASE = "kenar var"
PASS_VERDICT = "TÜM KAPILAR GEÇİLDİ — canlı gölge doğrulaması gerekli"
FAIL_VERDICT = "KAPI(LAR) GEÇİLEMEDİ — kenar YOK / dayanıklı değil (CLAUDE.md ACI DERSLER)"
# Ön-koşul hatası bir SONUÇ değildir: kusurlu/delikli veri "kenar yok" kanıtı üretmez,
# hiçbir şey üretmez. İkisini aynı cümleyle raporlamak, ölçüm yokluğunu ölçüm gibi
# sunmak olurdu (Tur-31 ilkesi) — bu yüzden ayrı metin.
PRECONDITION_VERDICT = "ÖN-KOŞUL SAĞLANMADI — veri kabul edilmedi, HÜKÜM YOK (ölçüm yapılmadı)"
# NOT: `veri_yukleme` bilerek DIŞARIDA — mevcut kilitler (test_edge_lab.py) onu
# FAIL_VERDICT ile çapalıyor. "Dosya bulunamadı" da bir kenar iddiası değildir ve
# aynı metne taşınmalı, ama bu AYRI bir karardır: kural sonuca göre değiştirilmez.
_PRECONDITION_GATES = ("veri_kapsami",)

_CV_AUC_THRESHOLD = 0.53
_DSR_THRESHOLD = 0.95
# Eşik-taraması ima eden çoklu-test sayısı (DSR deflasyonu için — train_meta.evaluate'in
# thr_candidates'ıyla AYNI sözleşme: gerçek bir tarama yapılmasa da TEK eşiğin seçilme
# SÜRECİ genelde birkaç aday arasından olur; muhafazakâr sabit sayı, dürüst tarafta kalır).
_THR_CANDIDATES: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60)
_DSR_N_TRIALS = len(_THR_CANDIDATES)
# Bu sabit artık YALNIZ adayın KENDİ içindeki taramadır. Programın TAMAMINDA kaç ayrık
# aday denendiği `research.edge_lab.ledger`'dan okunur ve deflasyon onunla ÇARPILIR
# (BACKLOG.md'nin elle tutulan "aileler arası çoklu-test" sayacının makine karşılığı).
# Boş defterde çarpan 1'dir → bugüne kadarki koşularla bit-denk.

_MIN_TRAIN = 20
_MIN_TEST = 10
_MIN_KEPT = 5
# --- Tur-24: §26'nın (lead-lag) doğurduğu iki EK yeterlilik ölçütü -------------------
# §26'da kapı simülasyonu İKİ bölmede de PASS etti ama yalnız **29/33.212** işlem üstünde
# (%0,087 seçicilik): 29 gözlemle ölçülen +0,10 expectancy'nin standart hatası farkın
# kendisi mertebesindeydi, yani "PASS" istatistiksel olarak boştu. O turda kural BİLEREK
# değiştirilmedi (bir aday FAIL ettikten SONRA eşik sıkıştırmak, sonuca göre kural
# değiştirmektir); adayı olmayan bu turda makineye çevriliyor.
#
# (a) ORANSAL seçicilik tabanı: mutlak `_MIN_KEPT=5` 33k'lık bir test diliminde anlamını
#     yitirir. Bu oranın ALTINDA kalan kapı, lojistik modelin en kötü kalibre olduğu uç
#     kuyruğu örnekliyordur — kenar değil, kuyruk artefaktı şüphesi.
_MIN_KEPT_FRAC = 0.005
# (b) ÖRNEKLEM-KOŞULLU anlamlılık: kapının TUTTUĞU ile ELEDİĞİ (ayrık iki örneklem)
#     ortalamaları arasındaki fark Welch t ile gürültüye kıyaslanır; tek-taraflı ~%5.
#     §26 bu ölçütte t≈1,15 verirdi (eşiğin altı) — yani kural o adayı da elerdi.
_GATE_T_THRESHOLD = 1.645


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class GateResult:
    """Tek bir kapının sonucu: ad + geçti/geçmedi + Türkçe açıklama + sayısal ayrıntılar."""

    name: str
    passed: bool
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Purge+embargo K-fold bölücü — research.purged_cv'den UYGUNSA import edilir
# ---------------------------------------------------------------------------
def _local_purged_folds(n: int, k: int, embargo: int, *, purge: int | None = None):
    """`research.purged_cv.purged_folds` erişilemezse minimal yerel purge+embargo K-fold
    üretici (AYNI sözleşme: sıralı [0..n) indeksini k bloğa böl, sol-purge + sağ-embargo)."""
    left_purge = embargo if purge is None else max(int(purge), 0)
    bounds = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        lo, hi = bounds[i], bounds[i + 1]
        test_idx = np.arange(lo, hi)
        left = np.arange(0, max(lo - left_purge, 0))
        right = np.arange(min(hi + max(embargo, left_purge), n), n)
        train_idx = np.concatenate([left, right])
        yield train_idx, test_idx


def _folds(n: int, k: int, embargo: int):
    fn = _purged_folds_impl if _purged_folds_impl is not None else _local_purged_folds
    yield from fn(n, k, embargo)


# ---------------------------------------------------------------------------
# Veri yükleme + savunmacı doğrulama
# ---------------------------------------------------------------------------
def _load_dataset(dataset_csv: str | Path, feature_cols: list[str], label_col: str,
                  ts_col: str) -> tuple[pd.DataFrame | None, str | None]:
    path = Path(dataset_csv)
    if not path.exists():
        return None, f"dataset bulunamadı: {path}"
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError) as e:
        return None, f"CSV okunamadı: {e}"
    missing = [c for c in [*feature_cols, label_col] if c not in df.columns]
    if missing:
        return None, f"eksik kolon(lar): {missing}"
    if ts_col in df.columns:
        df = (df.assign(_ord=pd.to_datetime(df[ts_col], errors="coerce", utc=True))
                .sort_values("_ord", kind="stable").drop(columns="_ord").reset_index(drop=True))
    return df, None


def _pnl_array(df: pd.DataFrame, label_col: str, pnl_col: str | None) -> np.ndarray:
    """Gerçek pnl_col verilmişse onu kullan; yoksa etiketten ikili sözde-pnl türet
    (win=+1/loss=-1) — kapı simülasyonunun dolar-yoksunu genel hali."""
    if pnl_col and pnl_col in df.columns:
        return df[pnl_col].to_numpy(dtype=float)
    y = df[label_col].to_numpy(dtype=float)
    return np.where(y >= 0.5, 1.0, -1.0)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, p))


def _min_kept_required(n_test: int) -> int:
    """Kapı sonucunun okunabilir sayılması için gereken en az işlem sayısı: mutlak taban
    (`_MIN_KEPT`) VE test diliminin oranı (`_MIN_KEPT_FRAC`) — hangisi büyükse. Küçük
    veri setlerinde mutlak taban, büyüklerde oran bağlar (§26 dersi)."""
    return max(_MIN_KEPT, math.ceil(_MIN_KEPT_FRAC * max(n_test, 0)))


def _welch_t(kept: np.ndarray, dropped: np.ndarray) -> float | None:
    """Kapının TUTTUĞU ve ELEDİĞİ getirilerin ortalamaları arasındaki Welch t.

    Neden base yerine ELENEN ile kıyas: `kept ⊂ base` olduğundan kapı-E ile base-E
    bağımlı örneklemlerdir; tutulan↔elenen ise AYRIKTIR, dolayısıyla "bu filtre iyiyi
    kötüden ayırıyor mu" sorusunun dürüst istatistiksel biçimi budur. Eşit varyans
    varsayılmaz (kuyruk örneklemi genelde farklı yayılımdadır). Örneklem/varyans
    yetersizse None (kapı dürüstçe FAIL eder — asla varsayılan PASS yok)."""
    if len(kept) < 2 or len(dropped) < 2:
        return None
    var_k = float(np.var(kept, ddof=1))
    var_d = float(np.var(dropped, ddof=1))
    se = math.sqrt(var_k / len(kept) + var_d / len(dropped))
    if not math.isfinite(se) or se <= 0.0:
        return None
    diff = float(np.mean(kept)) - float(np.mean(dropped))
    return float(diff / se) if math.isfinite(diff) else None


# ---------------------------------------------------------------------------
# 0) VERİ KAPSAMI — ön-koşul (BECERİ kapısı DEĞİL)
# ---------------------------------------------------------------------------
# Tur-51'in VRP verdikti "2021-06 → 2026-07" kapsamıyla raporlandı; veri setinde
# 2023-12 → 2025-07 arası **20 ay tamamen boştu** ve hiçbir uyarı çıkmamıştı. Kaynakta
# o sembollerin 2024 için 96.369, 2025 için 138.431 satırı DURUYORDU: hüküm, kapsamadığı
# bir dönem hakkında verilmiş oldu. Sebep builder'daydı (bkz. BULGU-52-1) ama yakalanması
# gereken yer BURASI: hangi aile olursa olsun, takvimin ORTASINDA deliği olan bir veri
# seti üstünde "kenar yok" demek o dönem için kanıt üretmez.
#
# Bu bir KABUL şartıdır, bir beceri ölçüsü değil → FAIL ederse kapılar hiç koşmaz ve
# koşu DEFTERE YAZILMAZ (`veri_yukleme` ile aynı sözleşme: kusurlu bir veri seti bir
# ARAMA denemesi değildir, sonraki adayın DSR eşiğini haksız yere yükseltmemeli).
# Tolerans sıfır değil: tatil/kapanış ayları ve gerçek veri-sonu için küçük bir pay
# bırakılır; delik ORANLA ölçülür, mutlak sayıyla değil.
_MAX_EMPTY_MONTH_FRAC = 0.02

# --- Tur-63: BULGU-54-1'in kapıya çevrilmesi (İSTENEN pencere ≠ DOĞAN pencere) --------
# Yukarıdaki delik ölçümü veri setinin KENDİ [min,max] aralığına bakar. Tur-54'te opsiyon
# yüzeyi 2021-01'den istendi ama ilk satır 2024-10-14'te DOĞDU: aralığın İÇİ deliksiz
# olduğu için kapı PASS dedi, verdikt "2021-2026" diye okundu, ölçülen 1,7 yıldı. Kesilen
# uç kapının baktığı yerin DIŞINDAYDI — delik ARAMAK, uç KESİLMESİNİ görmez.
#
# İstenen pencere veri setinden ÇIKARILAMAZ (hiç doğmamış bir yılın satırda izi yoktur) →
# kurucu BİLDİRMEK zorundadır. Bildirmezse kapı "kesilme ÖLÇÜLEMEDİ" der ve bunu metne
# yazar: ölçüm yokluğu ihlal yokluğu DEĞİLDİR (Tur-60/62 deseni). Sessizce PASS demek,
# tam da bu turun kapattığı kusuru kapının kendi içinde yeniden üretirdi.
#
# `warmup_days` ailenin ilk satırı üretebilmesi için gereken geçmiştir ve bir MAZERET
# değil BEYANDIR: kurucu kendi penceresini yazar, kapı beyanı gerçekle kıyaslar. Beyan
# gerçekten küçükse (optskew: 40 barlık z-penceresi beyan edilir, veri 3,7 yıl geç doğar)
# kapı FAIL eder; beyan büyükse "bu aile o dönemi ZATEN yanıtlayamaz" kaydı düşer.
_MAX_TRUNCATION_FRAC = 0.05


def _kesilme(ts_min: pd.Timestamp, ts_max: pd.Timestamp, expected_start: str | None,
             expected_end: str | None, warmup_days: int) -> dict[str, Any] | None:
    """İstenen pencere ile GERÇEKLEŞEN pencerenin uç farkları (gün + oran).

    Dönüş None = pencere BİLDİRİLMEDİ (ölçüm yapılamaz). Beyan geçersizse (etkin başlangıç
    bitişten sonra) `gecersiz` bayrağıyla döner — sessizce 0 kesilme raporlamak, yanlış
    beyanı DOĞRU ölçüm gibi göstermek olurdu."""
    if not expected_start or not expected_end:
        return None
    bas = pd.to_datetime(expected_start, errors="coerce")
    son = pd.to_datetime(expected_end, errors="coerce")
    if pd.isna(bas) or pd.isna(son):
        return {"gecersiz": "istenen pencere çözümlenemedi"}
    etkin = bas + pd.Timedelta(days=max(int(warmup_days), 0))
    span = (son - etkin).days
    if span <= 0:
        return {"gecersiz": f"etkin başlangıç ({etkin.date()}) bitişten ({son.date()}) sonra"}
    bas_gun = max((ts_min - etkin).days, 0)
    son_gun = max((son - ts_max).days, 0)
    return {
        "expected_start": str(bas.date()), "expected_end": str(son.date()),
        "warmup_days": int(max(warmup_days, 0)), "effective_start": str(etkin.date()),
        "span_days": int(span), "head_trunc_days": int(bas_gun), "tail_trunc_days": int(son_gun),
        "head_frac": round(bas_gun / span, 4), "tail_frac": round(son_gun / span, 4),
    }


def _stage_coverage(df: pd.DataFrame, ts_col: str, *, expected_start: str | None = None,
                    expected_end: str | None = None, warmup_days: int = 0) -> GateResult:
    """İki ön-koşul: (a) veri setinin kendi [min,max] aralığında BOŞ ay var mı,
    (b) İSTENEN pencerenin uçları kesilmiş mi (Tur-63/BULGU-54-1)."""
    if ts_col not in df.columns:
        return GateResult("veri_kapsami", True, f"'{ts_col}' kolonu yok — kapsam ölçülemedi")
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dropna()
    if ts.empty:
        return GateResult("veri_kapsami", False, f"'{ts_col}' hiçbir satırda çözümlenemedi")
    ts = ts.dt.tz_localize(None)                      # Period tz taşımaz — sessiz uyarıyı kes
    aylik = ts.dt.to_period("M").value_counts()
    tam = pd.period_range(ts.min().to_period("M"), ts.max().to_period("M"), freq="M")
    bos = [str(p) for p in tam if int(aylik.get(p, 0)) == 0]
    tolerans = math.floor(_MAX_EMPTY_MONTH_FRAC * len(tam))
    passed = len(bos) <= tolerans
    ornek = (": " + ", ".join(bos[:4]) + ("…" if len(bos) > 4 else "")) if bos else ""
    detail = (f"{len(tam)} aylık aralıkta {len(bos)} ay BOŞ "
              f"(tolerans {tolerans} = %{_MAX_EMPTY_MONTH_FRAC * 100:g}){ornek}")
    # Ölçülen aralık HER koşuda metriğe yazılır: verdiktin hangi dönemi kapsadığı
    # defterde kalmalı, yalnız elle yazılan markdown satırında değil (BULGU-63-2).
    metrics: dict[str, Any] = {"n_months": len(tam), "n_empty": len(bos), "empty_months": bos[:24],
                               "data_start": str(ts.min().date()), "data_end": str(ts.max().date())}

    kes = _kesilme(ts.min(), ts.max(), expected_start, expected_end, warmup_days)
    if kes is None:
        metrics["expected_declared"] = False
        detail += " | istenen pencere BİLDİRİLMEDİ — uç kesilmesi ÖLÇÜLEMEDİ"
    elif "gecersiz" in kes:
        metrics["expected_declared"] = False
        passed = False
        detail += f" | istenen pencere GEÇERSİZ: {kes['gecersiz']}"
    else:
        metrics.update(kes, expected_declared=True)
        en_buyuk = max(kes["head_frac"], kes["tail_frac"])
        if en_buyuk > _MAX_TRUNCATION_FRAC:
            passed = False
        detail += (f" | istenen {kes['expected_start']}→{kes['expected_end']} "
                   f"(+{kes['warmup_days']}g ısınma → {kes['effective_start']}), "
                   f"doğan {metrics['data_start']}→{metrics['data_end']}: "
                   f"baş kesme {kes['head_trunc_days']}g (%{100 * kes['head_frac']:.1f}), "
                   f"son kesme {kes['tail_trunc_days']}g (%{100 * kes['tail_frac']:.1f}) "
                   f"— tolerans %{_MAX_TRUNCATION_FRAC * 100:g}")
    return GateResult("veri_kapsami", passed, detail, metrics)


# ---------------------------------------------------------------------------
# 1) purged CV
# ---------------------------------------------------------------------------
def _stage_purged_cv(df: pd.DataFrame, feature_cols: list[str], label_col: str,
                     n_folds: int, embargo_frac: float) -> GateResult:
    n = len(df)
    min_n = max(n_folds * 20, 40)
    if n < min_n:
        return GateResult("purged_cv", False, f"yetersiz veri (n={n} < gerekli {min_n})")
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy(dtype=float)
    embargo = max(1, int(round(n * embargo_frac)))
    aucs: list[float] = []
    for tr_idx, te_idx in _folds(n, n_folds, embargo):
        if len(te_idx) == 0:
            continue
        ytr = y[tr_idx]
        if len(np.unique(ytr)) < 2:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr_idx], ytr)
        p = clf.predict_proba(X[te_idx])[:, 1]
        aucs.append(_auc(y[te_idx], p))
    if not aucs:
        return GateResult("purged_cv", False, "hiçbir fold değerlendirilemedi (tek sınıf/yetersiz veri)")
    mean_auc = float(np.mean(aucs))
    passed = mean_auc > _CV_AUC_THRESHOLD
    detail = f"ortalama OOS-AUC {mean_auc:.3f} ({len(aucs)} fold; eşik>{_CV_AUC_THRESHOLD})"
    return GateResult("purged_cv", passed, detail, {"mean_auc": round(mean_auc, 4), "n_folds": len(aucs)})


# ---------------------------------------------------------------------------
# 2) time + symbol çift-bölme kapı simülasyonu (train_meta.evaluate deseni)
# ---------------------------------------------------------------------------
def _time_split(df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(n * (1.0 - frac))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def _symbol_split(df: pd.DataFrame, symbol_col: str, frac: float,
                  seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = sorted(df[symbol_col].astype(str).unique())
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(len(symbols) * frac)))
    n_test = min(n_test, len(symbols))
    test_syms = set(rng.choice(np.array(symbols, dtype=object), size=n_test, replace=False).tolist())
    test_mask = df[symbol_col].astype(str).isin(test_syms)
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def _fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str],
                 label_col: str) -> np.ndarray | None:
    ytr = train_df[label_col].to_numpy(dtype=float)
    if len(np.unique(ytr)) < 2:
        return None
    Xtr = train_df[feature_cols].to_numpy(dtype=float)
    Xte = test_df[feature_cols].to_numpy(dtype=float)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def _one_split_gate(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str],
                    label_col: str, pnl_col: str | None, thr: float,
                    split_name: str) -> GateResult:
    gate_name = f"gate_{split_name}"
    if len(train_df) < _MIN_TRAIN or len(test_df) < _MIN_TEST:
        return GateResult(gate_name, False, f"yetersiz veri ({split_name} bölmesi)")
    p = _fit_predict(train_df, test_df, feature_cols, label_col)
    if p is None:
        return GateResult(gate_name, False, "eğitimde tek sınıf — model kurulamadı")
    pnl_te = _pnl_array(test_df, label_col, pnl_col)
    n_test = len(pnl_te)
    base_e = float(np.mean(pnl_te))
    kept_mask = p >= thr
    n_kept = int(kept_mask.sum())
    min_kept = _min_kept_required(n_test)
    enough = n_kept >= min_kept
    gate_e = float(np.mean(pnl_te[kept_mask])) if enough else None
    t_stat = _welch_t(pnl_te[kept_mask], pnl_te[~kept_mask]) if enough else None
    beats_base = gate_e is not None and gate_e > base_e
    significant = t_stat is not None and t_stat > _GATE_T_THRESHOLD
    passed = beats_base and significant
    metrics = {"base_e": round(base_e, 4),
              "gate_e": (round(gate_e, 4) if gate_e is not None else None),
              "n_kept": n_kept, "n_test": n_test,
              "keep_frac": (round(n_kept / n_test, 5) if n_test else None),
              "min_kept_required": min_kept,
              "t_stat": (round(t_stat, 3) if t_stat is not None else None)}
    if not enough:
        detail = (f"kapıyı geçen yeterli işlem yok (alınan={n_kept} < gerekli={min_kept}; "
                  f"mutlak taban {_MIN_KEPT}, oransal taban %{_MIN_KEPT_FRAC * 100:g} × {n_test})")
    else:
        assert gate_e is not None  # okuyucu/mypy: enough ise gate_e her zaman dolu
        t_txt = f"{t_stat:.2f}" if t_stat is not None else "hesaplanamadı"
        detail = (f"base E={base_e:+.4f} | kapı(≥{thr}) E={gate_e:+.4f} "
                  f"(alınan={n_kept}/{n_test} = %{100 * n_kept / n_test:.3f}) | "
                  f"tutulan↔elenen Welch t={t_txt} (eşik>{_GATE_T_THRESHOLD})")
        if beats_base and not significant:
            detail += " → fark örneklem gürültüsünü AŞMIYOR (§26 dersi)"
    return GateResult(gate_name, passed, detail, metrics)


def _stage_dual_gate(df: pd.DataFrame, feature_cols: list[str], label_col: str,
                     pnl_col: str | None, symbol_col: str, thr: float, time_frac: float,
                     symbol_frac: float, seed: int) -> tuple[GateResult, GateResult]:
    tr_t, te_t = _time_split(df, time_frac)
    gate_time = _one_split_gate(tr_t, te_t, feature_cols, label_col, pnl_col, thr, "time")

    if symbol_col in df.columns:
        tr_s, te_s = _symbol_split(df, symbol_col, symbol_frac, seed)
        gate_symbol = _one_split_gate(tr_s, te_s, feature_cols, label_col, pnl_col, thr, "symbol")
    else:
        gate_symbol = GateResult(
            "gate_symbol", False,
            f"'{symbol_col}' kolonu yok — sembol-bölmesi (hiç görülmemiş sembol genellemesi) ölçülemedi",
        )
    return gate_time, gate_symbol


# ---------------------------------------------------------------------------
# 3) DSR — kapı-sonrası (time bölmesi) getiri serisi üstünde
# ---------------------------------------------------------------------------
def _kept_pnl_from_time_gate(df: pd.DataFrame, feature_cols: list[str], label_col: str,
                             pnl_col: str | None, thr: float,
                             time_frac: float) -> tuple[np.ndarray, float] | None:
    """Kapı-sonrası **FAZLALIK** getiri serisi + çıkarılan taban (BULGU-25).

    Eskiden HAM PnL dönüyordu ve `_stage_dsr` Sharpe'ı SIFIRA karşı ölçüyordu. Yükselen
    piyasada (`base_E=+0,2514`) her alt küme pozitiftir → DSR=1,0'ı kapı-sonrası satırlar
    kadar kapının ELEDİĞİ satırlar (SR 0,042) ve RASTGELE alt küme (0,050) de veriyordu:
    kapı beceriyi değil piyasa sürüklenmesini ölçüyordu. ACI DERS #8'in (sınıf dengesizliği
    DSR'yi şişirir) farklı yerdeki AYNI yapısal hatası.

    Düzeltme, Tur-17'nin `excess_pseudo_returns` desenidir: karşılaştırma tabanı SIFIR değil
    **TRİVİYAL KURAL** — burada "kapıyı hiç uygulama, her işlemi al". Sabit çıkarmak std'yi
    değiştirmez, yani SR_fazlalık = (ort_kept − taban) / std_kept = kapının KAPISIZLIĞA karşı
    bilgi oranıdır. Bu, `_stage_dual_gate`'in "kapı-E > base-E" iddiasının deflasyona tabi hâli.

    **Taban AYNI TEST diliminden ölçülür — eğitim diliminden DEĞİL (ölçülerek seçildi).**
    Eğitim tabanı ilk denenendi (Tur-17'nin sızıntı kaygısıyla) ve gerçek veride YETMEDİ:
    rebalans setinde eğitim tabanı +0,1751 iken test tabanı +0,2514'tür; aradaki +0,076/işlem
    DÖNEM FARKI test diliminde sahte fazlalık olarak kalır ve n=27.003'te tek başına DSR=1,0
    üretir. Test tabanı sızıntı DEĞİLDİR: "her işlemi al" kuralı hiçbir şey öğrenmez, geleceği
    bilmez ve tahminleri değiştirmez — kıyas ölçütüdür (bilgi oranının tanımı), üstelik ikili
    kapının zaten kullandığı `base_E`'nin ta kendisidir. Yan fayda: neredeyse her şeyi tutan
    bir kapının fazlalığı yapısı gereği 0'a gider → beceriye değil bolluğa PASS verilemez.
    """
    tr_t, te_t = _time_split(df, time_frac)
    if len(tr_t) < _MIN_TRAIN or len(te_t) < _MIN_TEST:
        return None
    p = _fit_predict(tr_t, te_t, feature_cols, label_col)
    if p is None:
        return None
    pnl_te = _pnl_array(te_t, label_col, pnl_col)
    kept = pnl_te[p >= thr]
    # Kapı simülasyonuyla AYNI yeterlilik tabanı — DSR'yi 29 gözlem üstünde hesaplayıp
    # "n yeterli" saymak §26'nın tam olarak elediği şeydir (Tur-24).
    if len(kept) < _min_kept_required(len(pnl_te)):
        return None
    taban = float(np.mean(pnl_te))          # = _stage_dual_gate'in base_E'si (aynı dilim)
    return kept - taban, taban


def _stage_dsr(kept: tuple[np.ndarray, float] | None,
               n_trials: int = _DSR_N_TRIALS) -> tuple[GateResult, float | None]:
    if kept is None or len(kept[0]) < 5:
        return GateResult("dsr", False, "DSR için kapı-sonrası yeterli işlem yok (time bölmesi)"), None
    excess, taban = kept
    mom = sharpe_moments(np.asarray(excess, dtype=float))
    if mom["n"] < 5:
        return GateResult("dsr", False, "DSR için yeterli gözlem yok"), None
    var_trials = sharpe_estimator_var(mom["sr"], mom["n"], skew=mom["skew"], kurt=mom["kurt"])
    var_trials = var_trials if math.isfinite(var_trials) else 0.0
    n_trials = max(1, int(n_trials))
    value = dsr(mom["sr"], mom["n"], n_trials, var_trials, skew=mom["skew"], kurt=mom["kurt"])
    value_ok = math.isfinite(value)
    passed = value_ok and value > _DSR_THRESHOLD
    rounded = round(value, 4) if value_ok else None
    # Taban raporda GÖRÜNÜR: okuyucu kapının ne kadarının piyasa sürüklenmesi olduğunu
    # tek bakışta görebilmeli (BULGU-25'i teşhis etmek tam bir tur sürmüştü).
    detail = (f"DSR={rounded} FAZLALIK serisinde (n={mom['n']}, kapısız taban={taban:+.4f}/işlem "
              f"çıkarıldı, SR_fazlalık={mom['sr']:+.4f}, n_trials={n_trials} "
              f"(program-düzeyi çoklu-test), eşik>{_DSR_THRESHOLD})"
             if value_ok else "DSR hesaplanamadı (NaN — payda<=0)")
    return (GateResult("dsr", passed, detail,
                       {"dsr": rounded, "n": mom["n"], "taban": round(taban, 6),
                        "sr_fazlalik": round(mom["sr"], 6), "n_trials": n_trials}),
            rounded)


# ---------------------------------------------------------------------------
# 4) exec-lag-1 duyarlılık (ACI-7)
# ---------------------------------------------------------------------------
def _shift_label_1bar(df: pd.DataFrame, label_col: str, symbol_col: str,
                      pnl_col: str | None = None) -> pd.DataFrame:
    """Etiketi — VE verilmişse `pnl_col`'u — 1 bar (sembol-içi kronolojik sırayla; df zaten
    ts_col'a göre sıralı) kaydır: satır i'nin ÖZELLİKLERİ kalır, SONUCU bir sonraki barınkiyle
    değiştirilir (uygulamanın kararı 1 bar GECİKMİŞ verildiği senaryosu). Grubun son satırı
    (kaydıracak sonraki barı olmadığından) düşürülür.

    **PnL de kaydırılmalı (Tur-24 bulgusu).** Önceki sürüm yalnız etiketi kaydırıyordu;
    gerçek bir `pnl_col` verildiğinde kapı, KAYDIRILMIŞ etiketle eğitilip KAYDIRILMAMIŞ
    (gecikmesiz) PnL üstünde ölçülüyordu — yani ACI-7 kapısı tam da ölçtüğünü sandığı şeyi
    ölçmüyordu. İlk iki aday sözde-pnl kullandığı için (etiketten türetilir, kaydırılır)
    ısırmamıştı; gerçek PnL'li ilk adayda sessizce yanlış-pozitife dönerdi."""
    shift_cols = [label_col]
    if pnl_col and pnl_col in df.columns and pnl_col != label_col:
        shift_cols.append(pnl_col)
    if symbol_col in df.columns:
        out = df.copy()
        out[shift_cols] = out.groupby(symbol_col, sort=False)[shift_cols].shift(-1)
        return out.dropna(subset=shift_cols).reset_index(drop=True)
    if len(df) < 2:
        return df.iloc[0:0].copy()
    out = df.copy()
    out[shift_cols] = out[shift_cols].shift(-1)
    return out.dropna(subset=shift_cols).reset_index(drop=True)


def _stage_exec_lag(df: pd.DataFrame, feature_cols: list[str], label_col: str,
                    pnl_col: str | None, symbol_col: str, thr: float, time_frac: float,
                    gate_time: GateResult) -> GateResult:
    shifted = _shift_label_1bar(df, label_col, symbol_col, pnl_col)
    if len(shifted) < (_MIN_TRAIN + _MIN_TEST):
        return GateResult("exec_lag_1", False, "1-bar kaydırma sonrası yetersiz veri")
    tr, te = _time_split(shifted, time_frac)
    gate_lag = _one_split_gate(tr, te, feature_cols, label_col, pnl_col, thr, "exec_lag_1_inner")

    lag0_e = gate_time.metrics.get("gate_e")
    lag1_e = gate_lag.metrics.get("gate_e")
    effect = round(lag1_e - lag0_e, 4) if (lag0_e is not None and lag1_e is not None) else None
    metrics = dict(gate_lag.metrics)
    metrics["lag0_gate_e"] = lag0_e
    metrics["effect_vs_lag0"] = effect
    detail = (f"1-bar kaydırma sonrası kapı: {gate_lag.detail} | lag0 kapı-E={lag0_e} → "
             f"etki={effect} (ACI-7: kenar 1-bar sonra da kalıcı olmalı; aksi halde "
             "zamanlama artefaktı şüphesi — bkz. pairs_test.py --exec-lag)")
    return GateResult("exec_lag_1", gate_lag.passed, detail, metrics)


# ---------------------------------------------------------------------------
# Ana API
# ---------------------------------------------------------------------------
def run_candidate(name: str, dataset_csv: str | Path, feature_cols: list[str], label_col: str,
                  ts_col: str = "ts", symbol_col: str = "symbol", *, pnl_col: str | None = None,
                  n_folds: int = 5, thr: float = 0.55, embargo_frac: float = 0.02,
                  time_test_frac: float = 0.3, symbol_test_frac: float = 0.3,
                  seed: int = 42, ledger_path: str | Path | None = None,
                  record_run: bool = True, expected_start: str | None = None,
                  expected_end: str | None = None, warmup_days: int = 0) -> dict[str, Any]:
    """Bir aday sinyal ailesini modül docstring'indeki DÖRT kapıdan sırayla geçir.

    `label_col` ikili (0/1) beklenir. `pnl_col` verilmezse dolar-yoksunu sözde-pnl
    (win=+1/loss=-1) kullanılır — ölçüm YÖNÜ değişmez, yalnız ölçeği. Herhangi bir kapı
    öncekinin verisine bağımlıysa (dsr/exec_lag → gate_time) ve o veri eksikse ilgili kapı
    dürüstçe FAIL döner (asla varsayılan PASS yok — "yanlış-pozitif üretmeme" ilkesi).

    Dönüş: tek-sayfa verdikt dict — {'candidate','dataset','n_rows','generated_at','gates'
    [{'name','passed','detail','metrics'}, ...], 'dsr', 'exec_lag_effect', 'verdict'}.
    `verdict` ASLA "kenar var" içermez (bu modülün `_FORBIDDEN_PHRASE` guard'ı; test bunu
    doğrular) — en iyi ihtimalle `PASS_VERDICT` döner.

    **Defter (`research.edge_lab.ledger`).** Kapılara ULAŞAN her koşu deftere eklenir ve
    DSR deflasyonu defterdeki AYRIK aday sayısıyla çarpılır: program aradıkça kanıt eşiği
    SIKIŞIR (BACKLOG.md'nin elle tutulan çoklu-test sayacının makine karşılığı). Veri
    yükleme hatası deftere YAZILMAZ — yanlış yazılmış bir dosya yolu bir deneme değildir
    ve sonraki adayın eşiğini haksız yere yükseltmemeli. `record_run=False` yalnız
    keşif/geliştirme koşuları içindir (yazmaz ama okur: eşik yine program-düzeyidir).

    **`expected_start`/`expected_end`/`warmup_days` (Tur-63).** Kurucunun İSTEDİĞİ pencere
    ve ailenin ilk satırı üretmesi için gereken ısınma. Verilmezse uç kesilmesi ÖLÇÜLEMEZ
    ve kapı bunu METNE yazar — sessiz PASS yok. Verilirse doğan aralık beyanla kıyaslanır;
    tolerans aşılırsa ön-koşul DÜŞER, çünkü hüküm hiç doğmamış bir dönemi kapsayamaz
    (BULGU-54-1: 5,5 yıl istendi, 1,7 yıl ölçüldü, tek uyarı çıkmadı)."""
    generated_at = _utc_now_iso()
    df, err = _load_dataset(dataset_csv, feature_cols, label_col, ts_col)
    if err:
        gate = GateResult("veri_yukleme", False, err)
        return _finalize(name, dataset_csv, 0, [gate], generated_at)

    assert df is not None  # mypy/okuyucu için — err None ise df her zaman dolu

    # Kapılardan ÖNCE: kapsam ön-koşulu. Delikli veri üstünde verilen hüküm, delik olan
    # dönemi kapsamaz — kapıları koşturmak yanlış bir kesinlik üretirdi (bkz. BULGU-52-1).
    kapsam = _stage_coverage(df, ts_col, expected_start=expected_start,
                             expected_end=expected_end, warmup_days=warmup_days)
    if not kapsam.passed:
        return _finalize(name, dataset_csv, len(df), [kapsam], generated_at)

    n_rows = len(df)
    # GEÇEN kapsam ölçümü de verdiktte durur (Tur-63/BULGU-63-2): eskiden yalnız FAIL
    # ederse görünüyordu, yani "bu hüküm hangi dönemi kapsıyor" sorusunun yanıtı ne
    # kurucunun çıktısında ne de defterde vardı — yalnız elle yazılmış markdown satırında.
    gates: list[GateResult] = [kapsam,
                               _stage_purged_cv(df, feature_cols, label_col, n_folds, embargo_frac)]

    gate_time, gate_symbol = _stage_dual_gate(df, feature_cols, label_col, pnl_col, symbol_col,
                                              thr, time_test_frac, symbol_test_frac, seed)
    gates.extend([gate_time, gate_symbol])

    kept_pnl = _kept_pnl_from_time_gate(df, feature_cols, label_col, pnl_col, thr, time_test_frac)
    n_trials = ledger.n_trials_for(name, len(_THR_CANDIDATES), ledger_path)
    dsr_gate, dsr_value = _stage_dsr(kept_pnl, n_trials)
    gates.append(dsr_gate)

    lag_gate = _stage_exec_lag(df, feature_cols, label_col, pnl_col, symbol_col, thr,
                               time_test_frac, gate_time)
    gates.append(lag_gate)

    verdikt = _finalize(name, dataset_csv, n_rows, gates, generated_at, dsr_value=dsr_value,
                        exec_lag_effect=lag_gate.metrics.get("effect_vs_lag0"),
                        ledger_path=ledger_path)
    if record_run:
        ledger.record(verdikt, feature_cols=list(feature_cols), path=ledger_path)
    return verdikt


def _finalize(name: str, dataset_csv: str | Path, n_rows: int, gates: list[GateResult],
             generated_at: str, *, dsr_value: float | None = None,
             exec_lag_effect: float | None = None,
             ledger_path: str | Path | None = None) -> dict[str, Any]:
    all_passed = bool(gates) and all(g.passed for g in gates)
    if not all_passed and all(g.name in _PRECONDITION_GATES for g in gates):
        verdict = PRECONDITION_VERDICT
    else:
        verdict = PASS_VERDICT if all_passed else FAIL_VERDICT
    assert _FORBIDDEN_PHRASE not in verdict.lower(), "verdikt asla 'kenar var' içeremez (WP-E anayasası)"
    return {
        "candidate": name,
        "dataset": str(dataset_csv),
        "n_rows": n_rows,
        "generated_at": generated_at,
        "gates": [asdict(g) for g in gates],
        "dsr": dsr_value,
        "exec_lag_effect": exec_lag_effect,
        # BU koşu deftere yazılmadan ÖNCEKİ geçmiş: "bu aileyi zaten çürütmüştük" bilgisi
        # verdiktin yanında durmalı (ACI-7 dört kez tekrarlandı — hatırlatıcı ücretsiz).
        "prior_runs": ledger.prior_runs(name, ledger_path),
        "verdict": verdict,
    }
