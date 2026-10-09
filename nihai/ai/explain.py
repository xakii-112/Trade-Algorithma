"""
explain.py — Açıklanabilirlik + BELİRSİZLİK GARANTİSİ (AI v3 WP-X, Tur-16). KARAR-DIŞI/GÖLGE.

Amaç: 5.5M parametrelik ağ (nihai/ai/neural.py) kara kutu KALMASIN — her gölge kararın
"neden"i (özellik atıfları) ve "ne kadar eminiz"i (dağılım-varsayımsız garanti aralığı)
raporlanabilsin (docs/AI_V3_DERIN_BEYIN_SPEC.md §3.14).

Beş yetenek:
  - `permutation_importance` : model-BAĞIMSIZ özellik önemi (yalnız `predict_proba` duck-tipi
    ister — CapacityModel/NeuralCapacity API paritesiyle uyumlu; research/purged_cv.py'deki
    "sütunu karıştır, OOS-AUC düşüşünü ölç" deseninin genellemesi).
  - `integrated_gradients` / `attention_rollout` : TORCH'lu, KENDİ try-import'u ile —
    `nihai.ai.neural`'a MODÜL-DÜZEYİ import bağı YOK (anayasa/spec §3.14: "model objesi
    parametre olarak gelir"). `neural_model` DUCK-TİPLİDİR: `._model` özniteliği varsa
    (NeuralCapacity sarmalayıcı paritesi) o kullanılır, yoksa nesnenin KENDİSİ çağrılabilir
    (nn.Module) kabul edilir. attention_rollout `nn.MultiheadAttention` alt-modülüne forward
    hook takıp normal bir ileri-geçiş çalıştırır — transformer mimarisinin iç yapısı
    (tokenizer/blok sırası) TEKRAR YAZILMAZ, bu yüzden neural.py değişse bile kırılmaz.
  - `conformal_calibrate` / `conformal_interval` : split-conformal (Vovk/Lei), YALNIZ verilen
    kalibrasyon diliminden p_win için dağılım-varsayımsız garanti aralığı. Çıktısı
    `decision.decide(conformal_interval=...)` girdisine DOĞRUDAN uygundur (Dalga-2 köprüsü —
    bu modül decision.py'a import bağı ALMAZ; bağlama orkestratörün işidir).
  - `decision_card` / `global_model_report` : yukarıdakileri TEK Türkçe şablonda birleştiren
    rapor üretici brif'ler (reasoner.CLOSING ile biter — anayasa madde 3).

Savunmacı sözleşme: torch yok / model çağrılamaz / veri kısa-bozuk-NaN → None (fail-soft;
hiçbir fonksiyon istisna fırlatmaz — bkz. SPEC bölüm 0 madde 7-8). Determinizm: permütasyon
`seed` alır (rng tüketimi purged_cv.permutation_importance ile aynı desen); integrated_gradients
STOKASTİK DEĞİLDİR (kapalı-form yol-integrali, model `eval()`'a zorlanır — dropout kapalı).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike

from nihai.ai import reasoner

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - CI/geliştirme ortamında torch kurulu
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Permütasyon önemi — model-bağımsız (predict_proba duck-tipi)
# ---------------------------------------------------------------------------
def _auc(y: ArrayLike, p: ArrayLike) -> float | None:
    """AUC (ROC eğrisi altı alan); sklearn'in olgun/test-edilmiş implementasyonu (proje
    genelinde zaten bağımlılık — models.py/learning.py). Tek-sınıf/geçersiz girdi → None
    (çökme yok; savunmacı)."""
    y_arr = np.asarray(y)
    p_arr = np.asarray(p, dtype=float)
    if y_arr.size == 0 or y_arr.size != p_arr.size or len(set(y_arr.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_arr, p_arr))
    except ValueError:
        return None


def permutation_importance(model: Any, X: ArrayLike, y: ArrayLike, seed: int = 42,
                            feature_names: list[str] | None = None) -> list[dict]:
    """Model-BAĞIMSIZ permütasyon önemi: her özellik sütununu `seed`'li RNG ile karıştırıp
    OOS-AUC düşüşünü ölçer (research/purged_cv.permutation_importance ile AYNI fikir; burada
    `model` yalnız `predict_proba(X) -> (n,)` p(y=1) ister — CapacityModel/NeuralCapacity API
    paritesiyle uyumlu HERHANGİ bir nesne, sınıfa import bağı YOK).

    Dönüş: `[{'feature','importance','rank'}, ...]`, importance'a göre AZALAN sıralı (en
    önemli ilk; rank 1-tabanlı). `feature_names` verilmez/uzunluk uyuşmazsa "f0".."fn-1"
    kullanılır (savunmacı — asla çökme).

    Savunmacı: X 2B değil/çok kısa, X/y satır sayısı uyuşmuyor, y tek-sınıf veya base-AUC
    hesaplanamıyorsa → `[]` (SPEC bölüm 0 madde 7: bozuk VERİ → zararsız değer, istisna yok).
    Determinizm: aynı `seed` + aynı veri → bit-bit aynı sıralama (SPEC bölüm 0 madde 6)."""
    try:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
    except (TypeError, ValueError):
        return []
    if X_arr.ndim != 2 or X_arr.shape[0] < 2 or X_arr.shape[0] != y_arr.shape[0]:
        return []
    try:
        base_p = model.predict_proba(X_arr)
    except Exception as e:  # noqa: BLE001 — duck-tipli çağıran modelin her hatası tolere edilir
        logger.debug(f"[ai.explain] permutation_importance base predict_proba hata: {e}")
        return []
    base_auc = _auc(y_arr, base_p)
    if base_auc is None:
        return []

    n_feat = X_arr.shape[1]
    names = (list(feature_names) if feature_names and len(feature_names) == n_feat
             else [f"f{i}" for i in range(n_feat)])
    rng = np.random.default_rng(seed)
    drops: list[tuple[str, float]] = []
    for j, name in enumerate(names):
        orig = X_arr[:, j].copy()
        try:
            X_arr[:, j] = rng.permutation(orig)
            try:
                perm_p = model.predict_proba(X_arr)
                perm_auc = _auc(y_arr, perm_p)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[ai.explain] permutation_importance {name} predict_proba hata: {e}")
                perm_auc = None
            drop = (base_auc - perm_auc) if perm_auc is not None else 0.0
        finally:
            X_arr[:, j] = orig
        drops.append((name, float(drop)))

    drops.sort(key=lambda t: -t[1])
    return [{"feature": name, "importance": round(val, 6), "rank": i + 1}
            for i, (name, val) in enumerate(drops)]


# ---------------------------------------------------------------------------
# Torch'lu açıklanabilirlik — KENDİ try-import'u (nihai.ai.neural'a import bağı YOK)
# ---------------------------------------------------------------------------
_IG_STEPS_DEFAULT = 50


def _resolve_torch_model(neural_model: Any) -> Any:
    """`neural_model` duck-tipi çözümü: `._model` özniteliği varsa (NeuralCapacity
    sarmalayıcı paritesi — bkz. nihai/ai/neural.py, buradan İMPORT edilmez) İÇ torch
    modülü kullanılır; yoksa nesnenin KENDİSİ (ör. çıplak nn.Module) kullanılır.

    TOPLULUK (v4/Tur-22 `NeuralEnsemble`): `members` listesi varsa İLK üyenin torch modeli
    çözülür. Gerekçe: gradyan-tabanlı atıf tek bir hesap grafiğine ihtiyaç duyar; K üyenin
    atıflarını ortalamak MATEMATİKSEL olarak farklı bir nicelik olurdu (ve tamlık aksiyomu
    üye-başına tanımlıdır). Temsilî üye açıklanır — bu, sessizce None dönmekten (açıklama
    yeteneğini topluluk seçince kaybetmekten) dürüst biçimde daha iyidir."""
    members = getattr(neural_model, "members", None)
    if isinstance(members, list) and members:
        return _resolve_torch_model(members[0])
    inner = getattr(neural_model, "_model", None)
    return inner if inner is not None else neural_model


def _torch_forward_scalar(model: Any, x: Any) -> Any:
    """model(x) çıktısını TEK skaler-hedef tensöre indirger: dict çıktı ise 'direction'
    anahtarı (neural.py DeepMLP/QuantTransformer çok-görevli sözleşmesi — buraya import
    edilmeden, yalnız anahtar adı VARSAYIMI olarak bilinir) yoksa sözlükteki ilk değer;
    2-sınıflı logit ise softmax sonrası p(sınıf=1); 1-kolonlu çıktı olduğu gibi kullanılır."""
    out = model(x)
    if isinstance(out, dict):
        out = out.get("direction", next(iter(out.values())))
    if out.ndim == 2 and out.shape[-1] >= 2:
        out = torch.softmax(out, dim=-1)[:, 1]
    elif out.ndim == 2 and out.shape[-1] == 1:
        out = out[:, 0]
    return out.reshape(-1)


def integrated_gradients(neural_model: Any, x_row: ArrayLike, baseline: ArrayLike | None = None,
                          *, steps: int = _IG_STEPS_DEFAULT) -> dict | None:
    """Integrated Gradients (Sundararajan, Taly, Yan — ICML 2017): baseline→x_row doğrusal
    yolunda `steps` ara noktada gradyan örnekler; yol-integralini trapez kuralıyla yaklaşıp
    `(x-baseline) * ortalama_gradyan` ile özellik-başına ATIF üretir.

    `neural_model` DUCK-TİPLİ (bkz. `_resolve_torch_model`); `baseline` verilmezse NÖTR
    (sıfır) özellik vektörü kullanılır (spec §3.14). Model gradyan hesabı sırasında `eval()`
    zorlanır (dropout KAPALI → sonuç DETERMİNİSTİK, MC-dropout'un aksine — anayasa madde 6).

    Dönüş None: torch kurulu değil / model çağrılamaz-uyumsuz / `x_row`/`baseline` kısa-NaN-
    şekil-uyumsuz (fail-soft; hiçbir durumda istisna fırlatmaz).

    Tamlık aksiyomu (test kanıtı): Σ atıf ≈ f(x_row) − f(baseline); sapma `completeness_delta`
    alanında raporlanır (tolerans testte belgelenir — sayısal integral yaklaşıklığı)."""
    if not TORCH_AVAILABLE:
        return None
    torch_model = _resolve_torch_model(neural_model)
    if not callable(torch_model):
        return None
    try:
        x_arr = np.asarray(x_row, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if x_arr.size == 0 or not np.all(np.isfinite(x_arr)):
        return None
    base_arr = (np.zeros_like(x_arr) if baseline is None
                else np.asarray(baseline, dtype=float).reshape(-1))
    if base_arr.shape != x_arr.shape or not np.all(np.isfinite(base_arr)):
        return None

    feature_names = list(getattr(neural_model, "feature_names", []) or [])
    if len(feature_names) != x_arr.size:
        feature_names = [f"f{i}" for i in range(x_arr.size)]

    n_steps = max(1, int(steps))
    was_training = bool(getattr(torch_model, "training", False))
    try:
        if hasattr(torch_model, "eval"):
            torch_model.eval()
        x_t = torch.tensor(x_arr, dtype=torch.float32)
        b_t = torch.tensor(base_arr, dtype=torch.float32)
        alphas = torch.linspace(0.0, 1.0, n_steps + 1).reshape(-1, 1)
        # interp YENİ bir LEAF tensör olmalı (detach + requires_grad_) — b_t/x_t'nin kendisi
        # grad izlemiyor; autograd.grad bu yüzden doğrudan interp'e göre türev alabilsin diye.
        interp = (b_t + alphas * (x_t - b_t)).detach().requires_grad_(True)  # (steps+1, n_f)
        out = _torch_forward_scalar(torch_model, interp)  # (steps+1,) — satır-bağımsız model varsayımı
        grads = torch.autograd.grad(out.sum(), interp)[0]  # (steps+1, n_f)
        avg_grads = ((grads[:-1] + grads[1:]) / 2.0).mean(dim=0)  # trapez ortalaması
        attributions = ((x_t - b_t) * avg_grads).detach().numpy()
        with torch.no_grad():
            base_out = float(_torch_forward_scalar(torch_model, b_t.reshape(1, -1))[0])
            input_out = float(_torch_forward_scalar(torch_model, x_t.reshape(1, -1))[0])
    except Exception as e:  # noqa: BLE001 — herhangi bir torch/şekil hatası fail-soft None
        logger.debug(f"[ai.explain] integrated_gradients hata: {e}")
        return None
    finally:
        if was_training and hasattr(torch_model, "train"):
            torch_model.train()

    attr_map = {name: round(float(v), 6)
                for name, v in zip(feature_names, attributions, strict=True)}
    total_attr = float(attributions.sum())
    completeness_delta = abs(total_attr - (input_out - base_out))
    top5 = sorted(attr_map.items(), key=lambda kv: -abs(kv[1]))[:5]
    return {
        "feature_names": feature_names,
        "attributions": attr_map,
        "top5": [{"feature": k, "attribution": v} for k, v in top5],
        "baseline_output": round(base_out, 6),
        "input_output": round(input_out, 6),
        "total_attribution": round(total_attr, 6),
        "completeness_delta": round(completeness_delta, 6),
        "n_steps": n_steps,
    }


def attention_rollout(neural_model: Any, x_row: ArrayLike) -> dict | None:
    """Attention rollout (Abnar & Zuidema, ACL 2020): modelin TÜM `nn.MultiheadAttention`
    alt-modüllerine geçici forward hook takar (pre-hook `need_weights=True` ZORLAR, forward-
    hook başlık-ortalamalı dikkat matrisini yakalar), NORMAL bir ileri-geçiş çalıştırır,
    katman matrislerini residual-eklenmiş + satır-normalize ederek soldan çarpar (rollout).

    Bilinçli tasarım: transformer'ın tokenizer/blok İÇ YAPISI burada TEKRAR YAZILMAZ — model
    yalnız `model(x)` ile ÇAĞRILIR (QuantTransformer.forward(x) sözleşmesi: x şekli
    (batch, n_features), ham özellik vektörü — hazır token dizisi DEĞİL) ve dikkat ağırlıkları
    hook'larla dıştan gözlenir. Bu yüzden neural.py'nin iç mimarisi değişse bile kırılmaz.

    Yalnız GERÇEK transformer mimarisinde (en az bir `nn.MultiheadAttention` alt-modülü)
    anlamlı sonuç üretir — böyle bir alt-modül YOKSA (ör. DeepMLP) → None (spec §3.14:
    "yalnız transformer; değilse None"). Rollout matrisi satır-stokastiktir (her satır
    toplamı ≈1.0 — iki satır-stokastik matrisin çarpımı yine satır-stokastiktir); dönüşteki
    `cls_attention` bu özelliği miras alır (test kanıtı).

    `neural_model` DUCK-TİPLİ (bkz. `_resolve_torch_model`/`integrated_gradients`). torch
    kurulu değil / model çağrılamaz / MHA alt-modülü yok / şekil uyumsuz → None (fail-soft)."""
    if not TORCH_AVAILABLE:
        return None
    torch_model = _resolve_torch_model(neural_model)
    if not callable(torch_model):
        return None
    try:
        x_arr = np.asarray(x_row, dtype=float).reshape(1, -1)
    except (TypeError, ValueError):
        return None
    if x_arr.size == 0 or not np.all(np.isfinite(x_arr)):
        return None

    captured: list[Any] = []
    handles: list[Any] = []

    def _pre_hook(module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        kw = dict(kwargs)
        kw["need_weights"] = True
        kw["average_attn_weights"] = True
        return args, kw

    def _fwd_hook(module: Any, args: tuple, output: Any) -> None:
        if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
            captured.append(output[1].detach())

    was_training = bool(getattr(torch_model, "training", False))
    try:
        for m in torch_model.modules():
            if isinstance(m, torch.nn.MultiheadAttention):
                handles.append(m.register_forward_pre_hook(_pre_hook, with_kwargs=True))
                handles.append(m.register_forward_hook(_fwd_hook))
        if not handles:
            return None  # transformer değil (MultiheadAttention alt-modülü yok)
        if hasattr(torch_model, "eval"):
            torch_model.eval()
        with torch.no_grad():
            torch_model(torch.tensor(x_arr, dtype=torch.float32))
    except Exception as e:  # noqa: BLE001 — hook/forward hatası fail-soft None
        logger.debug(f"[ai.explain] attention_rollout hata: {e}")
        return None
    finally:
        for h in handles:
            h.remove()
        if was_training and hasattr(torch_model, "train"):
            torch_model.train()

    if not captured:
        return None
    seq_len = captured[0].shape[-1]
    eye = torch.eye(seq_len).unsqueeze(0)
    rollout = None
    for a in captured:  # forward çağrı sırası = katman 1→L (hook'lar bu sırada tetiklenir)
        a_hat = a + eye
        a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = a_hat if rollout is None else torch.bmm(a_hat, rollout)
    assert rollout is not None
    cls_row = rollout[0, 0, :].detach().numpy()  # (seq_len,) — pozisyon-0'ın aldığı toplam ağırlık

    feature_names = list(getattr(neural_model, "feature_names", []) or [])
    n_extra = seq_len - len(feature_names)
    if feature_names and n_extra >= 1:
        # QuantTransformer sözleşmesi (neural.py): [CLS] + 3×[SCALE] + n_features.
        labels = ["CLS"] + [f"scale_{i}" for i in range(n_extra - 1)] + feature_names
    else:
        labels = [f"pos{i}" for i in range(seq_len)]
    if len(labels) != seq_len:
        labels = [f"pos{i}" for i in range(seq_len)]

    weights = {lbl: round(float(w), 6) for lbl, w in zip(labels, cls_row, strict=True)}
    top5 = sorted(weights.items(), key=lambda kv: -kv[1])[:5]
    return {"n_layers": len(captured), "seq_len": seq_len, "cls_attention": weights,
            "top5": [{"position": k, "weight": v} for k, v in top5]}


# ---------------------------------------------------------------------------
# Conformal prediction — split-conformal (dağılım-varsayımsız garanti aralığı)
# ---------------------------------------------------------------------------
def conformal_calibrate(p_cal: ArrayLike, y_cal: ArrayLike, alpha: float = 0.1) -> dict | None:
    """Split-conformal kalibrasyon (Vovk/Lei; regresyon-tarzı MUTLAK-KALINTI nonconformity
    skoru p_win noktasal tahminine uygulanır): s_i = |y_i − p_i|. (1−alpha) garantili kota
    `q_hat` = s'nin sıralı ceil((n+1)(1−alpha))-inci istatistiği (sonlu-örneklem düzeltmesi —
    klasik split-conformal sonucu: test noktasının GERÇEK etiketi [p−q_hat, p+q_hat] aralığını
    en az (1−alpha) olasılıkla MARJİNAL olarak kapsar). Sıra n'i aşarsa `q_hat=1.0` (s∈[0,1]
    olduğundan pratikte "aralık [0,1]'in tamamı" ile eşdeğer — savunmacı üst sınır).

    YALNIZ verilen (p_cal, y_cal) kalibrasyon DİLİMİNDEN fit edilir — train/kalibrasyon
    ayrımı ÇAĞIRANIN sorumluluğudur (nihai.ai.learning.fit_calibrator ile AYNI felsefe:
    "YALNIZ verilen OOS çiftlerde fit"). `y_cal` ikili (0/1) olmalı.

    Geçersiz girdi (alpha∉(0,1), boş/dönüştürülemez çiftler, <1 geçerli çift) → None
    (savunmacı — çökme yok)."""
    try:
        a = float(alpha)
    except (TypeError, ValueError):
        return None
    if not (0.0 < a < 1.0):
        return None
    try:
        p_arr = np.asarray(p_cal, dtype=float).reshape(-1)
        y_arr = np.asarray(y_cal, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if p_arr.shape[0] != y_arr.shape[0]:
        return None
    pairs: list[tuple[float, float]] = []
    for pf, yf in zip(p_arr.tolist(), y_arr.tolist(), strict=True):
        if pf == pf and yf in (0.0, 1.0):  # NaN guard + ikili etiket şartı
            pairs.append((float(pf), float(yf)))
    n = len(pairs)
    if n < 1:
        return None
    scores = sorted(abs(y - p) for p, y in pairs)
    k = math.ceil((n + 1) * (1.0 - a))
    q_hat = 1.0 if k > n else scores[k - 1]
    return {"method": "split_conformal_abs_residual", "alpha": a, "n": n,
            "q_hat": round(float(q_hat), 6)}


def conformal_interval(p: float, calib: dict | None) -> tuple[float, float] | None:
    """`conformal_calibrate` çıktısını TEK bir p_win noktasına uygular: `[p−q_hat, p+q_hat]
    ∩ [0,1]`. Bu çıktı `nihai.ai.decision.decide(conformal_interval=...)` girdisine DOĞRUDAN
    uygundur (Dalga-2 köprüsü — decision.py bu modüle import bağı ALMAZ; bağlama orkestratörün
    işidir, bkz. modül docstring'i).

    `calib` None/dict-değil/`q_hat` eksik veya `p` sayısal değilse (NaN dahil) → None
    (savunmacı; decision.py'nin kapı-c'si zaten `None`'ı "pasif" sayar — hatalı/eksik
    kalibrasyon KAPIYI SESSİZCE DEVRE DIŞI bırakır, yanlış-güvenli sıkılaştırma üretmez)."""
    if not isinstance(calib, dict):
        return None
    try:
        pf = float(p)
        qf = float(calib.get("q_hat"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (pf == pf and qf == qf):  # NaN guard
        return None
    lo = max(0.0, pf - qf)
    hi = min(1.0, pf + qf)
    return (round(lo, 6), round(hi, 6))


# ---------------------------------------------------------------------------
# Rapor şablonları — brif-tipi çıktı (reasoner.CLOSING ile biter; anayasa madde 3)
# ---------------------------------------------------------------------------
_MAX_CARD_ATTRIBUTIONS = 5


def _top_attributions(attributions: list[dict] | dict | None) -> list[dict]:
    """`permutation_importance`/`integrated_gradients` çıktılarının HERHANGİ birini kabul
    eden esnek normalize edici: list[dict] (rank'lı) OLDUĞU GİBİ ilk-N alınır; ham
    {özellik: değer} sözlüğü mutlak-değere göre azalan sıralanır. Geçersiz/boş → []."""
    if not attributions:
        return []
    if isinstance(attributions, dict):
        try:
            items = sorted(attributions.items(), key=lambda kv: -abs(float(kv[1])))
        except (TypeError, ValueError):
            return []
        return [{"feature": k, "value": round(float(v), 6)}
                for k, v in items[:_MAX_CARD_ATTRIBUTIONS]]
    if isinstance(attributions, list):
        return [dict(a) for a in attributions[:_MAX_CARD_ATTRIBUTIONS] if isinstance(a, dict)]
    return []


def decision_card(decision_dict: dict, attributions: list[dict] | dict | None = None,
                   conformal: tuple[float, float] | None = None,
                   extras: dict | None = None) -> dict:
    """Bir GÖLGE kararın (ör. `dataclasses.asdict(ShadowDecision(...))`) tek şablonda AÇIKLAMA
    KARTI: karar özeti + ilk-5 özellik atfı (permutation_importance VEYA integrated_gradients
    çıktısından, İKİSİ de kabul edilir — bkz. `_top_attributions`) + conformal aralık yorumu +
    (varsa) extras (ör. text_intel takvim/ton bağlamı).

    `decision.py`'a İMPORT BAĞI YOK — `decision_dict` tamamen DUCK-TİPLİDİR (`.get` ile
    okunur; ShadowDecision alan adlarıyla eşleşir ama eksik/fazladan alan çökme üretmez).

    Dönüş: `{'card': {...yapısal alanlar...}, 'text': 'Türkçe açıklama...'}`; `text`
    reasoner.CLOSING ile biter (brif-tipi çıktı; anayasa madde 3)."""
    d = decision_dict if isinstance(decision_dict, dict) else {}
    symbol = d.get("symbol", "?")
    strategy = d.get("strategy", "?")
    action = d.get("action", "?")
    posterior = d.get("posterior")
    conviction = d.get("conviction")
    kelly = d.get("kelly_suggest")
    utility = d.get("utility")

    top_attrs = _top_attributions(attributions)

    conformal_flag: bool | None = None
    conformal_note = "conformal aralık verilmedi (kapı-c pasif — Dalga-2 köprüsü henüz bağlanmamış)."
    if conformal is not None:
        try:
            lo, hi = float(conformal[0]), float(conformal[1])
        except (TypeError, ValueError, IndexError):
            conformal_note = "conformal aralık okunamadı (bozuk girdi)."
        else:
            conformal_flag = lo <= 0.5 <= hi
            conformal_note = (
                f"conformal aralık [{lo:.3f}, {hi:.3f}]: " +
                ("0.5'i KAPSIYOR — kararsız." if conformal_flag
                 else "0.5'i kapsamıyor — yön ayrımı istatistiksel olarak destekli."))

    card = {
        "symbol": symbol, "strategy": strategy, "action": action,
        "posterior": posterior, "conviction": conviction, "kelly_suggest": kelly,
        "utility": utility, "top_attributions": top_attrs,
        "conformal_interval": (list(conformal) if conformal is not None else None),
        "conformal_straddles_half": conformal_flag,
        "extras": dict(extras) if extras else {},
    }

    lines = [f"Açıklama kartı — {symbol}/{strategy}: {action}"]
    if posterior is not None:
        lines.append(f"Posterior={posterior}, kanaat={conviction}, "
                     f"kelly-önerisi={kelly}, fayda={utility}.")
    if top_attrs:
        etiketler = []
        for a in top_attrs:
            val = a.get("value", a.get("importance", a.get("attribution")))
            etiketler.append(f"{a.get('feature', '?')}({val})")
        lines.append("En etkili özellikler: " + ", ".join(etiketler) + ".")
    lines.append(conformal_note)
    if extras:
        lines.append("Ek bağlam: " + ", ".join(f"{k}={v}" for k, v in extras.items()) + ".")
    lines.append(reasoner.CLOSING)
    return {"card": card, "text": "\n".join(lines)}


def global_model_report(model: Any, X: ArrayLike, y: ArrayLike, *,
                         feature_names: list[str] | None = None,
                         seed: int = 42, pdp_grid: int = 5) -> str:
    """Eğitim-sonrası KÜRESEL model-davranış raporu (Türkçe): permütasyon önem sıralaması +
    ilk-10 özelliğin kısmi bağımlılık (PDP) özeti. PDP: her özellik için `pdp_grid` noktalık
    ızgara (o sütunun min/25/50/75/max yüzdelikleri, varsayılan) — ızgara noktasında İLGİLİ
    sütun TÜM satırlarda o değere SABİTLENİR (diğer sütunlar DEĞİŞMEZ), ortalama p(y=1)
    ölçülür (klasik PDP tanımı, Friedman 2001). `research/train_neural.py` çıktısına
    model-davranış bölümü olarak eklenmek üzere tasarlanmıştır (spec §3.14).

    Brif-tipi çıktı: reasoner.CLOSING ile biter (anayasa madde 3). Savunmacı: `model`/`X`/`y`
    işlenemezse ilgili bölüm "hesaplanamadı" notuyla atlanır — fonksiyon asla çökmez."""
    lines = ["[Küresel model raporu]"]
    imp = permutation_importance(model, X, y, seed=seed, feature_names=feature_names)
    if not imp:
        lines.append("Permütasyon önemi hesaplanamadı (tek-sınıf veri veya yetersiz örnek).")
    else:
        lines.append("Permütasyon önemi (AUC düşüşü, azalan sıralı):")
        for row in imp[:15]:
            lines.append(f"  {row['rank']}. {row['feature']}: {row['importance']:+.4f}")

    try:
        X_arr = np.asarray(X, dtype=float)
    except (TypeError, ValueError):
        X_arr = np.empty((0, 0))
    if X_arr.ndim == 2 and X_arr.shape[0] >= 2 and X_arr.shape[1] >= 1:
        n_feat = X_arr.shape[1]
        names = (list(feature_names) if feature_names and len(feature_names) == n_feat
                 else [f"f{i}" for i in range(n_feat)])
        n_show = min(10, n_feat)
        lines.append(f"Kısmi bağımlılık (PDP), ilk {n_show} özellik "
                     f"({max(2, int(pdp_grid))} noktalık ızgara):")
        pct = np.linspace(0, 100, max(2, int(pdp_grid)))
        for j in range(n_show):
            grid_vals = np.percentile(X_arr[:, j], pct)
            pd_vals: list[float] = []
            for gv in grid_vals:
                Xg = X_arr.copy()
                Xg[:, j] = gv
                try:
                    p = model.predict_proba(Xg)
                    pd_vals.append(float(np.mean(p)))
                except Exception as e:  # noqa: BLE001 — PDP tek başına raporu düşürmemeli
                    logger.debug(f"[ai.explain] PDP {names[j]} hata: {e}")
                    pd_vals = []
                    break
            if not pd_vals:
                lines.append(f"  {names[j]}: hesaplanamadı.")
                continue
            trend = ("artan" if pd_vals[-1] > pd_vals[0] + 1e-6 else
                     "azalan" if pd_vals[-1] < pd_vals[0] - 1e-6 else "düz")
            degerler = ", ".join(f"{v:.3f}" for v in pd_vals)
            lines.append(f"  {names[j]}: [{degerler}] ({trend})")
    else:
        lines.append("Kısmi bağımlılık hesaplanamadı (X uygun şekilde değil).")
    lines.append(reasoner.CLOSING)
    return "\n".join(lines)
