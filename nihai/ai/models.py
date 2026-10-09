"""
models.py — Parametre KAPASİTE MERDİVENİ (AI v2 §2 Tur-13 WP-A · v3 Tur-16 WP-I). KARAR-DIŞI.

Yedi katmanlı ikili-sınıflandırma kapasitesi (logistic → elasticnet → gbm_small → mlp →
deep_mlp → quant_transformer → quant_transformer_xl), GERÇEK öğrenilmiş parametre sayısıyla
birlikte (v4/Tur-22: XL basamağı ≈30M — merdivenin üst ucu ≈10⁷ mertebesine çıktı). Bu modül
YALNIZ kapasiteyi sağlar; hangi katmanın ÜRETİME alınacağına karar vermez (promotion =
research/purged_cv + research/live_retrain --promote, kanıt-kapılı manuel yol — ACI DERS
#2/#5 bağlamı: az veriyle yüksek kapasite = overfit riski, "kapasite hazır" ≠ "kullanılmalı").

v3 notu (Tur-16): sinir-ağı katmanları `nihai/ai/neural.py`'dan LAZY sarılır (bkz.
`_neural_mod`); torch kurulu değilse fail-soft — neural tier'lar kullanılamaz, merdiven
mlp'de durur, v2 API'lerinin varsayılan davranışı BİT-DENK korunur (`recommend_tier` /
`param_ladder_report` genişlemeleri opt-in parametrelidir; eski çağıranlar etkilenmez).

Kayıt yolu SÖZLEŞMESİ: `save`/`load` YALNIZ çağıranın verdiği `path_base`'e okur/yazar.
ÜRETİM `data/meta_model.*` dosyalarına bu modülden HİÇBİR koşulda yazılmaz — modül-düzeyi
bir veri-dizini sabiti YOKTUR, DATA_DIR'a hiç referans verilmez (kasıtlı; anayasa madde 2/5).

API: PARAM_TIERS, NEURAL_TIERS, ALL_TIERS, CapacityModel(tier, feature_names, seed)
.fit/predict_proba/param_count/meta/save/load, recommend_tier(n_samples, n_features,
allow_neural=False), param_ladder_report(n_features, include_neural=False).
"""
from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from nihai import config as _cfg
from nihai.atomic_io import atomic_replace, atomic_write_text, tmp_path

PARAM_TIERS: list[str] = ["logistic", "elasticnet", "gbm_small", "mlp"]

# v3 (Tur-16): sinir-ağı katmanları — nihai/ai/neural.py'daki NEURAL_TIERS'ın LİTERAL
# kopyası. Buradan import EDİLMEZ çünkü modül-düzeyi `import nihai.ai.neural` torch
# importunu tetiklerdi: `import nihai.ai` her çağıranda saniyeler yer, torch'suz ortamda
# da paket importu ucuz kalmalıdır (fail-soft). Eşitlik test sözleşmesiyle garantilidir
# (tests/test_ai_models.py::test_neural_tiers_literal_kopya_eslesir).
NEURAL_TIERS: list[str] = ["deep_mlp", "quant_transformer", "quant_transformer_xl",
                           "quant_transformer_moe", "quant_transformer_moe_xl",
                           "quant_transformer_ultra", "quant_transformer_mega"]
ALL_TIERS: list[str] = PARAM_TIERS + NEURAL_TIERS


def _neural_mod() -> Any:
    """nihai.ai.neural LAZY import — torch importunu yalnız neural-tier kod yoluna
    sınırlar. Modülün kendisi fail-soft olduğundan bu import torch yokken de BAŞARILIDIR;
    torch yokluğu `neural.TORCH_AVAILABLE` bayrağıyla raporlanır (ImportError üretmez)."""
    from nihai.ai import neural
    return neural


def _utc_now_iso() -> str:
    """UTC ts (tz-aware, saniye çözünürlüğü) — memory.py/router.py ile aynı konvansiyon."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _build_estimator(tier: str, seed: int) -> Any:
    """(tier, seed) → fit-edilmemiş sklearn tahmin edici/pipeline (spec §2 hiperparametreleri
    BİREBİR). `saga` çözücü stokastik olduğundan elasticnet'te de random_state=seed verilir
    (determinizm — anayasa madde 6). YALNIZ sklearn katmanları (PARAM_TIERS); neural
    katmanlar CapacityModel.fit içindeki delegasyonla kurulur."""
    if tier == "logistic":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if tier == "elasticnet":
        # sklearn ≥1.8: penalty="elasticnet" deprecated — float l1_ratio tek başına elasticnet demek.
        return Pipeline([
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("clf", LogisticRegression(solver="saga", l1_ratio=0.5,
                                        max_iter=3000, random_state=seed)),
        ])
    if tier == "gbm_small":
        return HistGradientBoostingClassifier(max_iter=100, max_depth=4, early_stopping=True,
                                               random_state=seed)
    if tier == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500,
                              random_state=seed)
    raise ValueError(f"bilinmeyen tier: {tier!r} (geçerli: {PARAM_TIERS})")


def _gbm_param_count(model: HistGradientBoostingClassifier) -> int:
    """Tüm ağaçlardaki TOPLAM düğüm sayısı (iç+yaprak birlikte; her TreePredictor.nodes düz
    yapısal dizisinin uzunluğu). sklearn'ün `_predictors` iç API'sine dayanır (sürüm-kırılgan
    olabilir) — erişilemezse 0 döner (çökme yok, gerçek sayı yerine muhafazakâr alt-sınır)."""
    total = 0
    try:
        for iteration in model._predictors:
            for predictor in iteration:
                if predictor is not None:
                    total += int(predictor.nodes.shape[0])
    except AttributeError:
        return 0
    return total


def _mlp_param_count(model: MLPClassifier) -> int:
    """Σ ağırlık-matrisi eleman sayısı + Σ bias eleman sayısı (coefs_/intercepts_)."""
    total = sum(int(c.size) for c in model.coefs_)
    total += sum(int(b.size) for b in model.intercepts_)
    return int(total)


@dataclass
class CapacityModel:
    """Tek bir kapasite katmanını (tier) saran ince sarmalayıcı.

    Kurucu imzası spec ile birebir: `CapacityModel(tier, feature_names, seed=42)`
    (n_features_/trained_at/_model dataclass alanları init dışıdır, fit() doldurur).
    fit() ÇAĞRILMADAN predict_proba/param_count/save → RuntimeError (programcı hatası,
    veri-kalitesi sorunu değil — savunmacı-girdi sözleşmesi yalnız DATA'yı kapsar).

    v3: NEURAL_TIERS için ince FACADE — fit/predict/param_count/meta/save/load çağrıları
    `neural.NeuralCapacity`'ye delege edilir (API paritesi orada test sözleşmesidir).
    torch kurulu değilken neural tier kurucusu RuntimeError fırlatır (ortam eksikliği;
    ValueError = programcı hatası ayrımı korunur)."""
    tier: str
    feature_names: list[str]
    seed: int = 42
    n_features_: int | None = field(default=None, init=False)
    trained_at: str | None = field(default=None, init=False)
    _model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tier not in ALL_TIERS:
            raise ValueError(f"bilinmeyen tier: {self.tier!r} (geçerli: {ALL_TIERS})")
        if self.tier in NEURAL_TIERS and not _neural_mod().TORCH_AVAILABLE:
            raise RuntimeError(
                f"torch kurulu değil — {self.tier!r} kullanılamaz "
                "(fail-soft: merdiven mlp'de durur, v2 davranışı değişmez)")
        self.feature_names = list(self.feature_names)

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, X: ArrayLike, y: ArrayLike) -> CapacityModel:
        """X:(n,f) float64, y:(n,) {0,1}. Uyumsuz/yetersiz (n<2) girdi → ValueError (programcı
        hatası — API sözleşmesi ihlali, savunmacı-degrade değil). Neural tier'lar
        NeuralCapacity'ye delege edilir (aux başlıkları facade'dan verilmez — çok-görevli
        eğitim isteyen çağıran NeuralCapacity'yi doğrudan kullanır)."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2 or X_arr.shape[0] < 2 or X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X/y boyutları uyumsuz veya yetersiz örnek (fit için ≥2 satır gerekir)")
        if self.tier in NEURAL_TIERS:
            wrapped = _neural_mod().NeuralCapacity(self.tier, self.feature_names, seed=self.seed)
            # Cihaz/amp OPT-IN (env; varsayılan cpu + amp kapalı → sayısal sonuç
            # DEĞİŞMEZ, bit-denklik testleri dokunulmadan geçer). Gerekçe ve ölçüm:
            # `config.neural_device` / `config.neural_amp`.
            # amp'i tier'a göre KISIYORUZ: `deep_mlp`te ölçüldü ve ZARARLI
            # (gpu 0,14 s → gpu+amp 0,45 s = 0,48×); token'lı tier'larda faydalı
            # (quant_transformer 6,21 → 3,36 s). Env tek başına küçüğe amp açtıramaz.
            device = _cfg.neural_device()
            amp = _cfg.neural_amp() and device != "cpu" and self.tier != "deep_mlp"
            wrapped.fit(X_arr, y_arr, device=device, amp=amp)
            self._model = wrapped
        else:
            model = _build_estimator(self.tier, self.seed)
            model.fit(X_arr, y_arr)
            self._model = model
        self.n_features_ = int(X_arr.shape[1])
        self.trained_at = _utc_now_iso()
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """p(y=1), şekil (n,). fit edilmemişse RuntimeError."""
        if not self.is_fitted:
            raise RuntimeError("CapacityModel henüz fit edilmedi")
        X_arr = np.asarray(X, dtype=float)
        if self.tier in NEURAL_TIERS:
            return np.asarray(self._model.predict_proba(X_arr), dtype=float)
        proba = self._model.predict_proba(X_arr)
        return np.asarray(proba[:, 1], dtype=float)

    def param_count(self) -> int:
        """GERÇEK öğrenilmiş parametre sayısı (tier'e göre yöntem — bkz. modül-düzeyi
        _gbm_param_count/_mlp_param_count docstring'leri; neural'da numel toplamı).
        fit edilmemişse RuntimeError."""
        if not self.is_fitted:
            raise RuntimeError("CapacityModel henüz fit edilmedi")
        if self.tier in NEURAL_TIERS:
            return int(self._model.param_count())
        if self.tier == "logistic":
            clf = self._model
            return int(clf.coef_.size + clf.intercept_.size)
        if self.tier == "elasticnet":
            clf = self._model.named_steps["clf"]
            return int(clf.coef_.size + clf.intercept_.size)
        if self.tier == "gbm_small":
            return _gbm_param_count(self._model)
        if self.tier == "mlp":
            return _mlp_param_count(self._model)
        return 0  # erişilemez (tier __post_init__'te doğrulanır)

    def meta(self) -> dict:
        """{'tier','n_features','param_count','seed','trained_at'} (spec §2 birebir; neural
        tier fit edilmişse NeuralCapacity.meta() — aynı 5 anahtar + opsiyonel eğitim-bilgisi
        anahtarları, tüketiciler fazla anahtarı yok sayar). fit edilmemişse param_count None
        (henüz hesaplanamaz — RuntimeError fırlatmaz, meta() salt-okunur özet olduğundan
        degrade eder)."""
        if self.tier in NEURAL_TIERS and self.is_fitted:
            return dict(self._model.meta())
        return {
            "tier": self.tier,
            "n_features": self.n_features_,
            "param_count": self.param_count() if self.is_fitted else None,
            "seed": self.seed,
            "trained_at": self.trained_at,
        }

    def save(self, path_base: Path) -> None:
        """<path_base>.json (meta + feature_names, atomik METİN) + model gövdesi: sklearn
        tier'larda <path_base>.pkl (atomik İKİLİ), neural tier'larda <path_base>.pt
        (NeuralCapacity.save delegasyonu — aynı atomik disiplin). YALNIZ verilen path_base'e
        yazar; üretim data/meta_model.* dosyalarına bu metottan ASLA yazılmaz (çağıranın
        sorumluluğudur — bkz. modül docstring'i). fit edilmemişse RuntimeError."""
        if not self.is_fitted:
            raise RuntimeError("fit edilmemiş CapacityModel kaydedilemez")
        if self.tier in NEURAL_TIERS:
            self._model.save(Path(path_base))
            return
        base = Path(path_base)
        base.parent.mkdir(parents=True, exist_ok=True)
        json_path = Path(f"{base}.json")
        pkl_path = Path(f"{base}.pkl")
        payload = {**self.meta(), "feature_names": list(self.feature_names)}
        atomic_write_text(json_path, json.dumps(payload, ensure_ascii=False, indent=2))
        pkl_tmp = tmp_path(pkl_path)   # yazana özel ara dosya (eşzamanlı yazan çakışmasın)
        pkl_tmp.write_bytes(pickle.dumps(self._model))
        atomic_replace(pkl_tmp, pkl_path)

    @classmethod
    def load(cls, path_base: Path) -> CapacityModel | None:
        """Savunmacı yükleme (memory.py/router.py ile aynı desen): dosya yok/bozuk/eksik-alan
        → None (çökme yok). `feature_names` JSON'da yoksa jenerik f0..f{n-1} üretilir.
        Neural tier artefaktı (JSON'daki tier NEURAL_TIERS'taysa) NeuralCapacity.load'a
        delege edilir; torch kurulu değilse None (savunmacı — torch'suz ortamda neural
        artefaktı OKUNAMAZ ama süreç çökmez)."""
        base = Path(path_base)
        json_path = Path(f"{base}.json")
        pkl_path = Path(f"{base}.pkl")
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict) or "tier" not in raw:
            return None
        if str(raw["tier"]) in NEURAL_TIERS:
            neural = _neural_mod()
            if not neural.TORCH_AVAILABLE:
                return None
            wrapped = neural.NeuralCapacity.load(base)
            if wrapped is None:
                return None
            obj = cls(tier=str(wrapped.tier), feature_names=list(wrapped.feature_names),
                      seed=int(wrapped.seed))
            obj._model = wrapped
            obj.n_features_ = getattr(wrapped, "n_features_", None)
            obj.trained_at = getattr(wrapped, "trained_at", None)
            return obj
        try:
            model_obj = pickle.loads(pkl_path.read_bytes())
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ValueError):
            return None
        n_features = raw.get("n_features")
        names = raw.get("feature_names")
        if not isinstance(names, list) or not names:
            try:
                names = [f"f{i}" for i in range(int(n_features or 0))]
            except (TypeError, ValueError):
                names = []
        try:
            seed = int(raw.get("seed", 42))
        except (TypeError, ValueError):
            seed = 42
        try:
            obj = cls(tier=str(raw["tier"]), feature_names=[str(nm) for nm in names], seed=seed)
        except ValueError:
            return None
        obj._model = model_obj
        try:
            obj.n_features_ = int(n_features) if n_features is not None else None
        except (TypeError, ValueError):
            obj.n_features_ = None
        trained_at = raw.get("trained_at")
        obj.trained_at = str(trained_at) if trained_at is not None else None
        return obj


def recommend_tier(n_samples: int, n_features: int, allow_neural: bool = False) -> str:
    """Örnek/param oranı disiplini (spec §2, ACI DERS #2/#5 bağlamı — az veriyle yüksek
    kapasite = overfit riski):
      n < 50*f  → logistic
      n < 400*f → elasticnet
      n < 2000*f → gbm_small
      aksi      → mlp
    n_features<=0 veya dönüştürülemez girdi → en muhafazakâr katman ("logistic").

    v3 (allow_neural=True; varsayılan False → v2 davranışı BİT-DENK, eski çağıranlar
    etkilenmez) — plan BÖLÜM D eşikleri, YALNIZ torch kuruluyken (yoksa dürüstçe mlp'de
    durur, fail-soft):
      n ≥ 26.000.000  → quant_transformer_mega   (v8/Tur-36)
      n ≥ 10.000.000  → quant_transformer_ultra  (v7/Tur-35)
      n ≥ 3.600.000   → quant_transformer_moe_xl (v6/Tur-34)
      n ≥ 1.200.000   → quant_transformer_moe  (v5/Tur-32)
      n ≥ 400.000     → quant_transformer_xl   (v4/Tur-22)
      n ≥ 75.000      → quant_transformer
      n ≥ 8.000·f/25  → deep_mlp
    NOT: deep_mlp eşiği (320·f) ham param/örnek disiplinine göre mlp eşiğinden (2000·f)
    GEVŞEKTİR — bilinçli plan kararı: gerekçe Bölüm F üç-aşamalı reçetedir (sentetik
    müfredat ön-eğitimi az-veri rejimini taşır); promotion zaten kanıt-kapılı ayrı yoldur.

    v4 XL eşiği (400k) quant_transformer eşiğinin ≈5,5 katıdır — parametre sayısının da
    ≈5,5 katı olması nedeniyle (örnek/parametre oranı korunur). Bu eşik BİR ÖNERİDİR, izin
    değil: 400k satır biriktiğinde bile promotion kanıt-kapılı ayrı yolda kalır (ACI DERS
    #2/#5 — "kapasite hazır" ≠ "kullanılmalı").

    v5 MoE eşiği (1,2M) AYNI oran kuralının devamıdır: ≈93M parametre / ≈30M (XL) ≈ 3,1 →
    400k × 3 ≈ 1,2M. Eşik TOPLAM parametreye göre belirlenir, aktif parametreye (≈36M) göre
    DEĞİL — aşırı-uyum riskini taşıyan şey modelin ezberleyebileceği kapasitedir, tek bir
    ileri geçişte kaç ağırlığın çarpıldığı değil.

    v6 MoE-XL eşiği (3,6M) AYNI oranın üçüncü uygulamasıdır: ÖLÇÜLDÜ (f=96) 283.639.571 /
    93.228.307 = 3,04 → 1,2M × 3 = 3,6M. Bu eşik bugünkü canlı logdan (≈6k satır) ÜÇ
    MERTEBE uzaktır; yani basamak pratikte ERİŞİLEMEZDİR ve bu BİLİNÇLİDİR — kapasite
    merdiveni veri disiplinini gevşetmek için değil, gevşetmediğini GÖSTERMEK için uzar.

    v7 ULTRA eşiği (10M) AYNI oranın dördüncü uygulamasıdır: ÖLÇÜLDÜ (f=96) 771.874.072 /
    283.639.571 = 2,72 → 3,6M × 2,72 ≈ 9,8M → 10M. Canlı logdan (≈6k satır) DÖRT mertebe
    uzak; bu basamağın canlı modeli YOKTUR ve planlanmıyor.

    v8 MEGA eşiği (26M) AYNI oranın beşinci uygulamasıdır: ÖLÇÜLDÜ (f=96) 2.014.146.906 /
    771.874.072 = 2,61 → 10M × 2,61 ≈ 26M. Bu basamağın canlı modeli YOKTUR ve
    PLANLANMIYOR (2,01 milyar float32 ≈ 8 GB ağırlık + ≈24 GB AdamW durumu — bu donanımda
    eğitim imkânsız; ACI DERS #5)."""
    try:
        n, f = int(n_samples), int(n_features)
    except (TypeError, ValueError):
        return "logistic"
    if f <= 0:
        return "logistic"
    if allow_neural and _neural_mod().TORCH_AVAILABLE:
        if n >= 26_000_000:
            return "quant_transformer_mega"
        if n >= 10_000_000:
            return "quant_transformer_ultra"
        if n >= 3_600_000:
            return "quant_transformer_moe_xl"
        if n >= 1_200_000:
            return "quant_transformer_moe"
        if n >= 400_000:
            return "quant_transformer_xl"
        if n >= 75_000:
            return "quant_transformer"
        if n >= 8_000 * f / 25:
            return "deep_mlp"
    if n < 50 * f:
        return "logistic"
    if n < 400 * f:
        return "elasticnet"
    if n < 2000 * f:
        return "gbm_small"
    return "mlp"


@lru_cache(maxsize=32)
def _gbm_representative_param_count(n_features: int, seed: int = 42) -> int | None:
    """gbm_small için KAPALI-FORM parametre formülü YOKTUR (düğüm sayısı veriye bağımlı —
    ağaç büyümesi split kalitesine/early-stopping'e göre durur). Sabit-seed TEMSİLİ sentetik
    ayrılabilir veri üzerinde GERÇEK bir fit ile ölçülür (deterministik: aynı n_features/seed
    → aynı sayı). Sentetik fit başarısızsa (ör. çok küçük n_features) → None (çökme yok).

    lru_cache (Tur-30): fonksiyon skaler girdilerin SAF fonksiyonu ve docstring'i gereği
    deterministik; sonuç int|None (paylaşım riski yok). Web panosu bu raporu her Beyin
    sekmesi açılışında istiyordu ve sentetik fit ÖLÇÜLDÜ: 186 ms — her seferinde AYNI
    sayıyı yeniden üretmek için."""
    f = max(1, int(n_features))
    n_informative = max(1, min(f, math.ceil(f / 2)))
    try:
        X, y = make_classification(
            n_samples=2000, n_features=f, n_informative=n_informative, n_redundant=0,
            n_clusters_per_class=1, class_sep=1.5, random_state=seed)
    except ValueError:
        return None
    model = HistGradientBoostingClassifier(max_iter=100, max_depth=4, early_stopping=True,
                                            random_state=seed)
    model.fit(X, y)
    return _gbm_param_count(model)


def param_ladder_report(n_features: int = 25, include_neural: bool = False) -> list[dict]:
    """Katman-başına parametre sayısı raporu (entegrasyon raporunun kaynağı — bkz.
    AI_V2_BEYIN_SPEC.md 'Parametre artışı raporu'). logistic/elasticnet/mlp KAPALI-FORM
    (fit gerekmez, anında); gbm_small sabit-seed TEMSİLİ fit ile (bkz.
    _gbm_representative_param_count — None olabilir, veri-bağımlılığı belgelenmiş).

    v3 (include_neural=True; varsayılan False → v2 ile aynı 4 satır): deep_mlp ve
    quant_transformer satırları eklenir — torch varsa GERÇEK numel sayımı
    (neural.theoretical_param_count, fit'siz), yoksa param_count=None + 'note' degrade
    satırı (fail-soft görünürlüğü: merdiven raporu torch'suz ortamda da üretilir).

    Dönüş: [{'tier','param_count'}, ...] — PARAM_TIERS (+NEURAL_TIERS) sırasıyla."""
    f = max(1, int(n_features))
    logistic_p = f + 1
    elasticnet_p = math.comb(f, 2) + f + 1
    mlp_p = 64 * f + 64 + 64 * 32 + 32 + 32 * 1 + 1
    gbm_p = _gbm_representative_param_count(f)
    rows: list[dict] = [
        {"tier": "logistic", "param_count": logistic_p},
        {"tier": "elasticnet", "param_count": elasticnet_p},
        {"tier": "gbm_small", "param_count": gbm_p},
        {"tier": "mlp", "param_count": mlp_p},
    ]
    if include_neural:
        neural = _neural_mod()
        for tier in NEURAL_TIERS:
            if neural.TORCH_AVAILABLE:
                rows.append({"tier": tier,
                             "param_count": neural.theoretical_param_count(tier, f)})
            else:
                rows.append({"tier": tier, "param_count": None,
                             "note": "torch kurulu değil (fail-soft — merdiven mlp'de durur)"})
    return rows
