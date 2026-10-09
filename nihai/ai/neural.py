"""
neural.py — Sinir ağı çekirdeği (AI v3 Tur-16 · v4 Tur-22 · v5 Tur-32 · v6 Tur-34 ·
v7 Tur-35 · v8 Tur-36 · v9 Tur-42). KARAR-DIŞI/GÖLGE.

Parametre kapasitesini GERÇEK, ölçülebilir, kaydedilebilir PyTorch modelleriyle sağlar.
YEDİ basamak (`NEURAL_TIERS`) — ölçülen sayılar f=96 üretim özellik sayısında:
  · `deep_mlp`                 — ara basamak, (512,256,128,64) MLP (223k).
  · `quant_transformer`        — v3, 5,47M (per-feature tokenizasyon + 6 pre-LN blok).
  · `quant_transformer_xl`     — v4, 31,8M (12 blok · d=384 · SwiGLU · RMSNorm ·
                                 LayerScale · stokastik derinlik · PLR sayısal gömme).
  · `quant_transformer_moe`    — v5, 93,2M (blok başına 6 uzman · top-2).
  · `quant_transformer_moe_xl` — v6, 283,6M (16 ince-taneli uzman + paylaşımlı uzman).
  · `quant_transformer_ultra`  — v7, 771,9M (40 ince-taneli uzman + paylaşımlı uzman +
                                 DAĞILIMSAL oynaklık başlığı `vol_q`, pinball kaybı).
  · `quant_transformer_mega`   — v8, 2,01 MİLYAR (80 ince-taneli uzman + KANIT temelli yön
                                 başlığı `dir_evid`, Dirichlet/EDL).

v9 EKİ — BELLEK REÇETESİ (Tur-42), İKİSİ DE OPT-IN ve varsayılan yol v8 ile BİT-DENK:
`fit(optimizer_name="adafactor")` optimizatör durumunu O(n)→O(√n) indirir (ölçüldü: 8,00 →
0,0156 bayt/parametre) — ULTRA/MEGA'yı 8 GB'lık bir kartta eğitilebilir kılan tek şey budur.
`fit(amp=True)` ileri geçişi bf16 autocast'e alır (YALNIZ CUDA'da; ağırlıklar float32).
Adafactor FARKLI bir algoritmadır (AdamW ile bit-denk DEĞİL) → seçim `meta()["recipe"]`e
ve model kartına yazılır. CLI karşılıkları: `--optimizer` / `--amp`.

v7 EKİ — ÖZ-DENETİMLİ ÖN-EĞİTİM (`NeuralCapacity.pretrain`): maskeli özellik modelleme,
ETİKET GEREKTİRMEZ. ACI DERS #5 darboğazı "veri yokluğu" diye adlandırır; bu kol etiketsiz
veriyi kullanan tek koldur. `is_fitted`'i AÇMAZ (etiket görmemiş gövdeden p_win okunamaz).
Ayrıca `NeuralEnsemble` (derin topluluk, Lakshminarayanan ve ark. 2017): K üyeli, üye
başına farklı seed → kapasite K katına çıkar VE belirsizlik ayrışır (üye-içi MC-dropout =
aleatorik, üye-arası yayılım = epistemik; toplam varyans yasası ile birleşir).

Bu modül YALNIZ kapasiteyi sağlar; hangi katmanın ÜRETİME alınacağına karar VERMEZ
(promotion = kanıt-kapılı ayrı süreç). ACI DERS #2/#5: yüksek kapasite + az veri = overfit
riski; "kapasite hazır" ≠ "kullanılmalı". v4'ün hiçbir eklentisi bu kuralı gevşetmez —
büyük katman da, topluluk da, kalibrasyon da GÖLGE kalır.

FAIL-SOFT SÖZLEŞMESİ: torch kurulu değilse `TORCH_AVAILABLE=False`; modül YİNE DE import
edilebilir (mevcut hiçbir akış/test kırılmaz — router.py'nin opsiyonel-sağlayıcı deseninin
sinir-ağı karşılığı). `NeuralCapacity`/`NeuralEnsemble` kurucuları bu durumda RuntimeError
fırlatır; nn.Module sınıfları da yalnız torch varken tanımlanır (modül seviyesinde `None`'a
düşer — çağrılırlarsa TypeError, ama zaten hiçbir kod yolu torch yokken oraya ulaşmaz).

CapacityModel (nihai/ai/models.py) API PARİTESİ: fit/predict_proba/param_count/meta/save/
load/is_fitted — aynı hata sözleşmesi (fit-öncesi RuntimeError, API ihlali ValueError).
save/load YALNIZ çağıranın verdiği path_base'e yazar/okur; üretim `data/meta_model.*`
dosyalarına bu modülden HİÇBİR koşulda yazılmaz (modül-düzeyi DATA_DIR sabiti YOKTUR —
anayasa madde 2). `neural.py` `models.py`'ı import ETMEZ (orkestratör lazy-import ile bağlar).

Eğitim reçetesi (v3 tabanı, spec §3.1): AdamW, dropout 0.2, weight_decay 1e-4, grad-clip
1.0, val = kronolojik SON %15 (shuffle YOK — zaman serisi, look-ahead yasağı; çağıran X/y'yi
zaman sırasıyla vermelidir), early-stop patience 10, `torch.manual_seed(seed)` determinizmi.
Çok-görevli kayıp: ana görev yön (CE, 2 sınıf) + yardımcı görevler (büyüklük-kovası CE, vol
MSE, rejim CE) — yalnız `aux`'ta VERİLEN başlıklar eğitilir.

v4 REÇETE EKLERİ — HEPSİ OPT-IN, VARSAYILANLAR v3 İLE BİT-DENK (eski çağıran/kayıtlı
artefakt etkilenmez; `models.recommend_tier(allow_neural=...)` deseniyle aynı disiplin):
  · `scheduler="cosine"`      — ısınmalı kosinüs LR (sabit LR yerine; adım-başı).
  · `ema_decay=0.999`         — ağırlık EMA'sı; val ölçümü ve nihai model EMA kopyasından.
  · `class_weight="balanced"` — sınıf-dengesi ağırlıkları YALNIZ eğitim dilimindeki orandan
                                (ACI DERS #8'in eğitim tarafındaki karşılığı: dengesizliği
                                beceri sanmamak için kaybı da dengele).
  · `label_smoothing=0.05`    — aşırı-güvenli logit bastırma.
  · `early_stop_metric="val_auc"` — kayıp yerine ayırt-etme üstünden erken durma.
  · `calibrate=True`          — fit SONRASI val diliminde sıcaklık ölçekleme (Guo ve ark.
                                2017); kanonik kalibrasyon YİNE DE learning.fit_calibrator'
                                dadır — bu, modelin kendi çıktısını dürüstleştiren ek kat.
  · `device="auto"`           — CUDA varsa GPU (yoksa sessizce CPU; ortam-bağımlı, fail-soft).

Çoklu zaman-ölçeği: `feature_names` önek konvansiyonu — `h_*` saatlik, `d_*` günlük,
`w_*` haftalık, öneksiz = ölçeksiz (tekil-ölçek, savunmacı — yine çalışır). Her özellik
kendi [SCALE] tip gömmesiyle işaretlenir; 3 ayrı [SCALE] özet-token'ı (h/d/w) transformer
çıktısından füzyon başlığına beslenir. Dizi düzeni ([CLS] + 3×[SCALE] + n_features) TÜM
transformer katmanlarında AYNIDIR — `nihai/ai/explain.py::attention_rollout` bu düzene ve
`nn.MultiheadAttention` alt-modülünün varlığına dayanır; v4 bloğu modernleşirken (RMSNorm/
SwiGLU/LayerScale) dikkat modülü bilinçli olarak `nn.MultiheadAttention` bırakılmıştır.
"""
from __future__ import annotations

import contextlib
import inspect
import json
import math
import threading
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from nihai.atomic_io import atomic_replace, atomic_write_text, tmp_path

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - CI/geliştirme ortamında torch kurulu
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

# Tur-80 (BULGU-80-3): çıkarım yolları PAYLAŞILAN modelin train()/eval() kipini değiştirir ve
# MC-dropout süreç-genel `torch.manual_seed` kullanır. Canlı tahmin servisi bunları 8 thread'den
# EŞZAMANLI çağırıyordu → ölçüldü (gerçek deep_mlp artefaktı, MC=20, 8 thread): 2160 çağrının
# 2100'ü seri (deterministik) sonuçtan FARKLIYDI; kilitle 0/2160. RLock: topluluk üyeleri iç içe girer.
INFER_LOCK = threading.RLock()


@contextlib.contextmanager
def inference_threads(n: int):
    """Kilit altında torch iç-işlem thread sayısını geçici olarak `n` yapar, çıkışta geri alır.

    Tek satırlık canlı çıkarım içindir: varsayılan 8 OpenMP thread'i küçük modelde iş yapmadan
    CPU yakar. Toplu (araştırma) çıkarımı etkilemez — ayar yalnız bu blok süresincedir."""
    with INFER_LOCK:
        if torch is None:
            yield
            return
        onceki = torch.get_num_threads()
        torch.set_num_threads(n)
        try:
            yield
        finally:
            torch.set_num_threads(onceki)


NEURAL_TIERS: list[str] = ["deep_mlp", "quant_transformer", "quant_transformer_xl",
                           "quant_transformer_moe", "quant_transformer_moe_xl",
                           "quant_transformer_ultra", "quant_transformer_mega"]

# Tokenizer TABANLI basamaklar: dizi düzeni [CLS]+3×[SCALE]+özellikler olanlar. Maskeli
# ön-eğitim (Tur-35) ve dikkat-rollout YALNIZ bunlarda anlamlıdır (`deep_mlp` düz MLP'dir,
# token'ı yoktur → `pretrain` orada dürüstçe ValueError verir, sessizce atlamaz).
TOKENIZED_TIERS: list[str] = [t for t in NEURAL_TIERS if t != "deep_mlp"]

# Çok-görevli yardımcı başlıklar: büyüklük 5 kovaya (ör. güçlü-aşağı..güçlü-yukarı),
# rejim 6 sınıfa (çağıranın etiket şeması — quant_math.classify_regime'e sıkı bağ YOK,
# üretici (train_neural sentetik müfredatı / gerçek etiketleyici) semantiği tanımlar).
MAGNITUDE_BUCKETS = 5
REGIME_CLASSES = 6

# Sabit eğitim reçetesi (spec §3.1 — v3 tabanı; v4 ekleri fit() argümanlarıyla opt-in).
_GRAD_CLIP_NORM = 1.0
_DROPOUT = 0.2
_AUX_LOSS_WEIGHT = 0.3   # yardımcı görev kayıplarının ana-görev kaybına göre ağırlığı

# v4 (Tur-22) XL mimarisi sabitleri — plan BÖLÜM D rev.4.
_XL_D_MODEL = 384
_XL_BLOCKS = 12
_XL_HEADS = 6
_XL_FFN = 1536
_XL_HEAD_HIDDEN = 128
_XL_DROP_PATH_MAX = 0.1     # stokastik derinlik: son blokta 0.1'e doğrusal artar
_XL_PLR_FREQS = 24          # PLR sayısal gömme: özellik başına periyodik frekans sayısı
_XL_PLR_SIGMA = 0.05        # frekans ilklendirme ölçeği (Gorishniy ve ark. 2022 §3.3)
_XL_LAYERSCALE_INIT = 1e-4  # LayerScale γ ilklendirmesi (Touvron ve ark. 2021)

# v5 (Tur-32) SEYREK UZMAN KARIŞIMI (MoE) basamağı — parametre sayısı XL'in ≈3 katı,
# TOKEN BAŞINA HESAP ise ≈aynı (yalnız top_k uzman çalışır). Merdivenin üst ucunu
# "daha çok hesap" ile değil "daha çok KAPASİTE" ile büyütmenin tek dürüst yolu budur.
_MOE_D_MODEL = 384
_MOE_BLOCKS = 8
_MOE_HEADS = 6
_MOE_FFN = 1536
_MOE_EXPERTS = 6
_MOE_TOP_K = 2
_MOE_HEAD_HIDDEN = 128
_MOE_DROP_PATH_MAX = 0.1
# Yük dengeleme kaybının ağırlığı (Switch Transformer, Fedus ve ark. 2021 §2.2). Dengeleme
# OLMADAN yönlendirici tek uzmana çöker ve model fiilen tek-FFN'li bir transformer olur —
# yani kapasite artışı KÂĞIT ÜSTÜNDE kalır. Bu terim kapasiteyi gerçek kılan şeydir.
_MOE_BALANCE_WEIGHT = 0.01

# v6 (Tur-34) İNCE-TANELİ + PAYLAŞIMLI uzman karışımı — merdivenin YENİ üst basamağı.
# İki fark Tur-32 MoE'sinden (DeepSeekMoE, Dai ve ark. 2024 §3):
#   (1) İNCE TANE: uzman başına gizli genişlik YARIYA iner (1536→1024) ama uzman SAYISI
#       2,7 katına çıkar (6→16). Aynı aktif bütçeyle seçilebilecek uzman KOMBİNASYONU
#       C(6,2)=15'ten C(16,2)=120'ye çıkar → yönlendiricinin ifade gücü artar.
#   (2) PAYLAŞIMLI UZMAN: her token'da KOŞULSUZ çalışan bir uzman, tüm rejimlerde ortak
#       olan bilgiyi taşır; yönlendirilen uzmanlar böylece ARTIĞI (rejime özgü kısmı)
#       öğrenmekte serbest kalır — aksi hâlde her uzman ortak bilgiyi ayrı ayrı ezberler.
_MOEXL_D_MODEL = 512
_MOEXL_BLOCKS = 10
_MOEXL_HEADS = 8
_MOEXL_EXPERT_FFN = 1024      # yönlendirilen uzman başına gizli genişlik (ince tane)
_MOEXL_ROUTED_EXPERTS = 16
_MOEXL_TOP_K = 2
_MOEXL_SHARED_FFN = 1024      # paylaşımlı uzman (her token'da her zaman çalışır)
_MOEXL_HEAD_HIDDEN = 192
_MOEXL_DROP_PATH_MAX = 0.1
# Yönlendirici z-kaybı (ST-MoE, Zoph ve ark. 2022 §3.2): logsumexp(router_logits)² cezası.
# NEDEN: 16 uzmanlı yönlendiricide logitler serbestçe büyür, softmax doyar ve gradyan ölür
# (yük dengeleme kaybı bunu GÖRMEZ — o yalnız dağılımın düzlüğüne bakar). z-kaybı logit
# ÖLÇEĞİNİ sınırlar; iki terim farklı patolojileri hedefler, biri diğerinin yerine geçmez.
_MOE_ZLOSS_WEIGHT = 1e-3

# v7 (Tur-35) ULTRA basamağı — merdivenin YENİ üst ucu. İki eksende birden büyür:
#   (1) KAPASİTE: uzman sayısı 16→40 ve d_model 512→640 (ince tane KORUNUR: uzman gizli
#       genişliği 1024→768'e İNER). Seçilebilen uzman kombinasyonu C(16,2)=120 → C(40,2)=780.
#   (2) YETENEK: yalnız büyüklük değil — `vol` görevi artık NOKTA tahmini yanında bir
#       QUANTILE demeti de üretir (pinball kaybı). Oynaklık tahmininde nokta değeri
#       dağılımı gizler; bu basamak dağılımı da söyler (çapraz-geçiş YAPISI GEREĞİ yok,
#       bkz. `_monotone_quantiles`).
_ULTRA_D_MODEL = 640
_ULTRA_BLOCKS = 12
_ULTRA_HEADS = 8
_ULTRA_EXPERT_FFN = 768
_ULTRA_ROUTED_EXPERTS = 40
_ULTRA_TOP_K = 2
_ULTRA_SHARED_FFN = 1536
_ULTRA_HEAD_HIDDEN = 256
_ULTRA_DROP_PATH_MAX = 0.1
# Oynaklık quantile'ları — medyan DAHİL (tek sayı isteyen çağıran onu okur; nokta `vol`
# başlığı ayrıca durur ve BİT-DENK kalır). Artan sırada olmak ZORUNDA (sözleşme).
VOL_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

# v8 (Tur-36) MEGA basamağı — merdivenin YENİ üst ucu. ULTRA'dan iki farkı var:
#   (1) KAPASİTE: uzman 40→80, d_model 640→768, blok 12→16; uzman gizli genişliği yine
#       İNER (768→640) — ince tane KORUNUR, sayı katlanır. C(40,2)=780 → C(80,2)=3160.
#   (2) SEYREKLİK: toplam parametre ≈2,6× artarken token başına AKTİF parametre ≈1,6×
#       artar; yani basamak "daha çok hesap" değil "daha çok KAPASİTE" ekler (bkz. §3).
_MEGA_D_MODEL = 768
_MEGA_BLOCKS = 16
_MEGA_HEADS = 12
_MEGA_EXPERT_FFN = 640
_MEGA_ROUTED_EXPERTS = 80
_MEGA_TOP_K = 2
_MEGA_SHARED_FFN = 2048
_MEGA_HEAD_HIDDEN = 320
_MEGA_DROP_PATH_MAX = 0.1

# Maskeli özellik ön-eğitimi (Tur-35) — bkz. `NeuralCapacity.pretrain`.
_PRETRAIN_MASK_RATIO = 0.25
_PRETRAIN_HEAD_HIDDEN = 128

# --------------------------- v8 (Tur-36) OPT-IN EK BAŞLIKLAR ---------------------------
# Tur-35'te `vol_q` YALNIZ ULTRA'da vardı ve açık kalem #2 şunu söylüyordu: alt basamaklara
# taşımak `state_dict` şemasını değiştirir → kayıtlı canlı artefaktlar bozulur. Çözüm başlığı
# basamağa GÖMMEK değil, OPT-IN yapmaktır: `extra_heads=()` varsayılanında hiçbir modül
# kurulmaz → RNG tüketimi, ağırlıklar ve state_dict anahtarları v7 ile BİT-DENK kalır;
# isteyen çağıran (ve yeni MEGA basamağı) aynı yeteneği HER basamakta açabilir.
#   'vol_q'    : dağılımsal oynaklık (Tur-35; ULTRA'da varsayılan AÇIK — geriye uyum)
#   'dir_evid' : kanıt-temelli (Dirichlet) yön başlığı — TEK geçişte epistemik belirsizlik
EXTRA_HEADS: tuple[str, ...] = ("vol_q", "dir_evid")

# Kanıt-temelli derin öğrenme (Sensoy ve ark. 2018, "Evidential Deep Learning"): başlık
# softmax OLASILIK değil, Dirichlet KANITI üretir (α = softplus(raw) + 1). Belirsizlik
# u = K/S (S = Σα) Dempster-Shafer boşluğudur ve TEK ileri geçişte okunur — MC-dropout'un
# T geçişine karşılık. KL cezası "yanıltıcı kanıtı" düz Dirichlet'e çeker; katsayı ilk
# epokta 0'dan başlayıp `_EVID_ANNEAL_EPOCHS`'ta 1'e tavlanır (kanıt daha oluşmadan
# cezalandırılırsa model hiç kanıt üretmemeyi öğrenir — makalenin §4 uyarısı).
_EVID_LOSS_WEIGHT = 0.3
_EVID_KL_WEIGHT = 0.1
_EVID_ANNEAL_EPOCHS = 10

# Çıkarım toplu-iş boyutu: 30M parametreli model + 200k satırlık X tek forward'da belleğe
# sığmaz. Sonuç toplu-iş boyutundan BAĞIMSIZDIR (batch-norm yok) — yalnız bellek disiplini.
_INFER_BATCH = 4096

# [SCALE] tip kimlikleri — feature_names önekinden. 0 = öneksiz/ölçeksiz (savunmacı taban).
_SCALE_PREFIX_IDS: dict[str, int] = {"h_": 1, "d_": 2, "w_": 3}
_N_SCALE_TYPES = 4

# Kalibrasyon sıcaklığı arama bandı (log-uzayda kaba tarama + altın-oran incelemesi —
# optimizer yerine DETERMİNİSTİK 1-B arama; anayasa madde 6).
_TEMP_LOG_LO, _TEMP_LOG_HI = -2.0, 2.0
_TEMP_GRID = 81
_TEMP_REFINE_ITERS = 40


def _utc_now_iso() -> str:
    """UTC ts (tz-aware, saniye çözünürlüğü) — models.py/learning.py ile aynı konvansiyon."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _feature_scale_id(name: str) -> int:
    """Özellik adı önekinden [SCALE] tip kimliği: h_→1 (saatlik) d_→2 (günlük) w_→3 (haftalık);
    önek yoksa 0 (ölçeksiz — tekil-ölçek girdi de savunmacı biçimde çalışır, spec §3.1)."""
    for prefix, sid in _SCALE_PREFIX_IDS.items():
        if name.startswith(prefix):
            return sid
    return 0


# ---------------------------------------------------------------------------
# Ölçüm yardımcıları — SAF NUMPY (sklearn bağımlılığı YOK: neural.py'nin import maliyeti
# torch ile zaten yüksek, ikinci ağır bağımlılık eklenmez). learning.py'deki karşılıklarıyla
# aynı tanımlar; oradan import edilmez çünkü learning.py bu modülü tüketen üst katmandır.
# ---------------------------------------------------------------------------
def _auc_np(y: np.ndarray, p: np.ndarray) -> float | None:
    """ROC-AUC, sıra-toplamı (Mann-Whitney U) ile; BERABERLİKLER ortalama sıra alır.
    Tek sınıflı y → None (AUC tanımsız; çökme yok).

    Tur-35 OPTİMİZASYONU — ACI DERSİ'nin ("bir ders MODÜLE değil DESENE uygulanır")
    üçüncü tekrarı: Tur-10 tam bu beraberlik döngüsünü `research/train_meta._auc` ve
    `research/forward_harness._auc`'ta cumsum ile vektörleştirmişti; `neural.py` Tur-16'da
    yazılırken SAF-PYTHON hâli yeniden doğdu ve `evaluation_metrics`in %85'ini yiyordu
    (n=200k'da 319 ms / 377 ms). Dönüşüm BİT-DENKTİR:
      grup [i..j] için  (i+j)/2 + 1  ≡  i + (c+1)/2   (c = j−i+1, ikisi de tam temsil edilir)
    ve grup sınırı `sorted_p[k] != sorted_p[k-1]` ile bulunur — NaN != NaN olduğundan her
    NaN özgün döngüdeki gibi KENDİ grubunu kurar (semantik birebir korunur)."""
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    if y_arr.size != p_arr.size or y_arr.size == 0:
        return None
    pos = y_arr >= 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    n = p_arr.size
    order = np.argsort(p_arr, kind="mergesort")
    sorted_p = p_arr[order]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_p[1:] != sorted_p[:-1]
    starts = np.flatnonzero(new_group)                       # her grubun başlangıç indeksi
    counts = np.diff(np.append(starts, n))
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.repeat(starts + (counts + 1) / 2.0, counts)   # 1-tabanlı ortalama sıra
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _brier_np(y: np.ndarray, p: np.ndarray) -> float | None:
    """Brier skoru = ortalama (p − y)². Boş girdi → None."""
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    if y_arr.size == 0 or y_arr.size != p_arr.size:
        return None
    return float(np.mean((p_arr - y_arr) ** 2))


def _ece_np(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float | None:
    """Beklenen Kalibrasyon Hatası (ECE, Naeini ve ark. 2015): eşit-genişlikli kovalarda
    |ortalama tahmin − gerçekleşen oran|'ın örneklem-ağırlıklı ortalaması. BOŞ kovalar
    katkı vermez. Brier'den farkı: ayrıştırma değil, DOĞRUDAN kalibrasyon sapması —
    "model 0.6 diyorsa gerçekten %60 mı?" sorusunun tek sayılı cevabı."""
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    if y_arr.size == 0 or y_arr.size != p_arr.size:
        return None
    edges = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    idx = np.clip(np.digitize(p_arr, edges[1:-1], right=False), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        sel = idx == b
        k = int(sel.sum())
        if k == 0:
            continue
        total += (k / y_arr.size) * abs(float(p_arr[sel].mean()) - float(y_arr[sel].mean()))
    return float(total)


def evaluation_metrics(y: ArrayLike, p: ArrayLike, n_bins: int = 10) -> dict:
    """{'n','auc','brier','ece','accuracy','pos_rate','pred_mean'} — tek çağrıda dürüstlük
    seti. AUC tek-sınıflı y'de None (diğerleri hesaplanır). Bu fonksiyon KENAR İDDİASI
    ÜRETMEZ; yalnız ölçer (CLAUDE.md ACI DERS #3: metriğin yeşilliği kenar demek değildir)."""
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    n = int(min(y_arr.size, p_arr.size))
    if n == 0:
        return {"n": 0, "auc": None, "brier": None, "ece": None, "accuracy": None,
                "pos_rate": None, "pred_mean": None}
    y_arr, p_arr = y_arr[:n], p_arr[:n]
    acc = float(np.mean((p_arr >= 0.5).astype(float) == (y_arr >= 0.5).astype(float)))
    return {
        "n": n,
        "auc": _auc_np(y_arr, p_arr),
        "brier": _brier_np(y_arr, p_arr),
        "ece": _ece_np(y_arr, p_arr, n_bins=n_bins),
        "accuracy": acc,
        "pos_rate": float(np.mean(y_arr >= 0.5)),
        "pred_mean": float(np.mean(p_arr)),
    }


# ---------------------------------------------------------------------------
# BELİRSİZLİK AYRIŞMASI (Tur-77, Madde 5.1) + KANIT İNANCI (Madde 5.2) — saf numpy, torch
# GEREKTİRMEZ. KARAR-DIŞI: çıktılar rapor/teşhistir, hiçbir karar/eşik yoluna bağlanmaz.
# ---------------------------------------------------------------------------
_P_EPS = 1e-12


def _bernoulli_entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, _P_EPS, 1.0 - _P_EPS)
    return -(q * np.log(q) + (1.0 - q) * np.log1p(-q))


def decompose_uncertainty(prob_samples: ArrayLike) -> dict[str, Any] | None:
    """(S, n) olasılık örneklemi → aleatorik/epistemik ayrışma, İKİ eş biçimde.

    S = parametre örneği (MC-dropout geçişi × topluluk üyesi), n = satır; her örnek
    p_s = P(y=1 | x, θ_s). Etiket Y ~ Bernoulli(p_θ) olduğundan:

      VARYANS:  Var[Y] = E_θ[p(1−p)]  +  Var_θ[p]      (toplam varyans yasası)
                         ALEATORİK       EPİSTEMİK
                ve özdeşlik gereği toplam = p̄(1−p̄) — yalnız ortalamaya bağlıdır.
      ENTROPİ:  H[p̄] = E_θ[H(p)]  +  I(Y; θ)           (Depeweg ve ark. 2018; BALD)
                       ALEATORİK     EPİSTEMİK (karşılıklı bilgi; Jensen gereği ≥ 0)

    ADLANDIRMA DÜZELTMESİ (Tur-77): depo belgeleri üye-içi MC-dropout yayılımını
    "aleatorik-benzeri" diye anıyordu. Bu YANLIŞTIR — dropout ağırlık belirsizliğini örnekler
    (Gal ve Ghahramani 2016), yani EPİSTEMİKTİR. Aleatorik belirsizlik tek bir parametre
    örneğinin kendi tahminindeki gürültüdür; ikili hedefte p(1−p).

    Geçersiz girdi (2B değil, boş, sonlu değil, [0,1] dışı) → None."""
    try:
        arr = np.asarray(prob_samples, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        return None
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0) or np.any(arr > 1.0):
        return None
    p_mean = arr.mean(axis=0)
    aleatorik_var = (arr * (1.0 - arr)).mean(axis=0)
    epistemik_var = arr.var(axis=0)
    toplam_h = _bernoulli_entropy(p_mean)
    aleatorik_h = _bernoulli_entropy(arr).mean(axis=0)
    return {"p": p_mean, "aleatorik_var": aleatorik_var, "epistemik_var": epistemik_var,
            "toplam_var": aleatorik_var + epistemik_var, "toplam_entropi": toplam_h,
            "aleatorik_entropi": aleatorik_h,
            "karsilikli_bilgi": np.maximum(toplam_h - aleatorik_h, 0.0),
            "n_ornek": int(arr.shape[0])}


def epistemic_flag(skor: ArrayLike, referans: ArrayLike, q: float = 0.9,
                   min_ref: int = 30) -> dict[str, Any] | None:
    """"Bunu daha önce görmedim" RAPOR bayrağı: satırın epistemik ölçüsü (ör.
    `karsilikli_bilgi`), REFERANS dağılımın (ör. doğrulama dilimi) q-yüzdeliğini aşıyor mu?

    Eşik sabit sayı DEĞİLDİR, modelin kendi referansından türetilir (`ood_zmax` deseni);
    <`min_ref` geçerli referansta eşik UYDURULMAZ → None. Bayrak bir KARAR değildir ve
    YÖNÜ VARSAYILAMAZ (Tur-36: MC-dropout std'si canlıda isabeti TERS sıralamıştı) — yüksek
    epistemik belirsizliğin gerçekten düşük isabetle birlikte gittiği ölçülmeden hiçbir
    yere bağlanamaz. Ölçüm: `docs/TUR77_RAPORU_2026-09-14.md`."""
    if not (0.0 < float(q) < 1.0):
        return None
    try:
        s = np.asarray(skor, dtype=float).reshape(-1)
        r = np.asarray(referans, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    r = r[np.isfinite(r)]
    if r.size < int(min_ref) or s.size == 0 or not np.all(np.isfinite(s)):
        return None
    esik = float(np.quantile(r, float(q)))
    bayrak = s > esik
    return {"esik": esik, "bayrak": bayrak, "pay": float(np.mean(bayrak)), "q": float(q),
            "n_ref": int(r.size)}


def evidential_belief(alpha: ArrayLike) -> dict[str, Any] | None:
    """Dirichlet α (n, K) → öznel mantık görüşü (Jøsang 2016; Sensoy ve ark. 2018).

    Kanıt e_k = α_k − 1, güç S = Σα, İNANÇ b_k = e_k / S, BOŞLUK u = K / S ve yapısı gereği
    Σ_k b_k + u = 1. İkili yönde b[:,1] = "yukarı" inancı, b[:,0] = "aşağı" inancı; u→1
    "hiç kanıt yok". `predict_evidential`'ın (p, u) çiftinin kütle ayrışmış hâlidir.
    α < 1 (negatif kanıt), K < 2 ya da sonlu olmayan değer → None."""
    try:
        a = np.asarray(alpha, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 2:
        return None
    if not np.all(np.isfinite(a)) or np.any(a < 1.0 - 1e-9):
        return None
    guc = a.sum(axis=1)
    return {"inanc": (a - 1.0) / guc[:, None], "bosluk": a.shape[1] / guc,
            "p": a / guc[:, None], "guc": guc}


# ---------------------------------------------------------------------------
# nn.Module mimarileri — YALNIZ torch kuruluyken tanımlanır (fail-soft; modül importu
# torch yokken bu bloğu hiç ÇALIŞTIRMAZ, bu yüzden `nn.Module` referansı asla patlamaz).
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:

    class _FeatureTokenizer(nn.Module):
        """Sayısal özellik → d-boyutlu token (FT-Transformer deseni, Gorishniy ve ark. 2021):
        token_j = x_j·W_j + b_j + kimlik_gömme_j + [SCALE]_tip_gömme(öneği).

        Her özelliğin KENDİ W_j/b_j'si vardır (paylaşılan tek Linear DEĞİL) — plan BÖLÜM D
        parametre tablosundaki "her özellik → d=256 token (per-feature linear + tip
        gömmesi)" satırının birebir karşılığı. `feature_id_embed` özelliğin kimliğini
        (permütasyon-değişmez dikkat mekanizmasına "hangi özellik bu" bilgisini) taşır;
        `scale_type_embed` ayrıca [SCALE] ÖNEK kategorisini (h_/d_/w_/öneksiz) taşır —
        ikisi birlikte hem kimlik hem çoklu-ölçek bağlamını kodlar."""

        def __init__(self, n_features: int, d_model: int, scale_ids: list[int]) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_features, d_model))
            self.bias = nn.Parameter(torch.zeros(n_features, d_model))
            self.feature_id_embed = nn.Parameter(torch.zeros(n_features, d_model))
            self.scale_type_embed = nn.Embedding(_N_SCALE_TYPES, d_model)
            self.register_buffer("scale_ids", torch.tensor(scale_ids, dtype=torch.long))
            nn.init.normal_(self.weight, std=0.02)
            nn.init.normal_(self.bias, std=0.02)
            nn.init.normal_(self.feature_id_embed, std=0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, n_features) → (B, n_features, d_model)
            tok = x.unsqueeze(-1) * self.weight + self.bias + self.feature_id_embed
            return tok + self.scale_type_embed(self.scale_ids)

    class _TransformerBlock(nn.Module):
        """Pre-LN transformer bloğu: LN→MHSA→artık + LN→FFN(GELU)→artık (spec §3.1:
        d=256, 4 başlık, FFN 1024, pre-LN, 6 blok — bu sınıf tek bir bloktur)."""

        def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.ln1(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + attn_out
            return x + self.ffn(self.ln2(x))

    class _Head(nn.Module):
        """Görev başlığı: d_model→d_hidden(GELU+Dropout)→d_out (spec tablosundaki
        '256→64→çıkış' deseni; hem çok-görevli hem probe başlıklarında aynı şekil)."""

        def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_hidden, d_out),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # ---------------------- v4 (Tur-22) yapı taşları ----------------------
    class _RMSNorm(nn.Module):
        """RMSNorm (Zhang & Sennrich 2019): x·γ/RMS(x). LayerNorm'dan farkı ORTALAMA
        ÇIKARMAZ — merkezleme adımı yoktur. Neden v4'te tercih edildi: derin yığınlarda
        LayerNorm ile denk doğruluk, daha az işlem ve daha kararlı gradyan ölçeği; ayrıca
        β (kaydırma) parametresi olmadığından blok başına parametre bütçesi dikkat/FFN'e
        kalır. `quant_transformer` (v3) bilinçli olarak LayerNorm'da BIRAKILDI — kayıtlı
        artefaktların bit-denkliği korunur."""

        def __init__(self, d_model: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d_model))
            self.eps = float(eps)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x * rms * self.weight

    class _SwiGLU(nn.Module):
        """SwiGLU FFN (Shazeer 2020): (SiLU(xW_g) ⊙ xW_u)W_d. Kapı (gate) dalı, GELU'lu
        düz FFN'e göre aynı gizli genişlikte daha ifadeli bir fonksiyon ailesi verir —
        modern LLM/tabular yığınlarının fiilî standardı. Parametre maliyeti 3 matris
        (2 yukarı + 1 aşağı) — v4 parametre artışının en büyük tek kalemi."""

        def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
            super().__init__()
            self.gate = nn.Linear(d_model, ffn_dim)
            self.up = nn.Linear(d_model, ffn_dim)
            self.down = nn.Linear(ffn_dim, d_model)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.drop(self.down(self.drop(nn.functional.silu(self.gate(x)) * self.up(x))))

    class _LayerScale(nn.Module):
        """LayerScale (Touvron ve ark. 2021, CaiT): artık dalın çıktısını öğrenilebilir
        kanal-başı γ ile ölçekler (γ₀=1e-4). Derin (12 blok) yığında eğitimin ilk
        adımlarında artık dalları fiilen KAPALI başlatır → kimlik-yakını ağ; bloklar
        gerektiği ölçüde 'açılır'. Az veri rejiminde derinliğin maliyetini düşürür."""

        def __init__(self, d_model: int, init: float = _XL_LAYERSCALE_INIT) -> None:
            super().__init__()
            self.gamma = nn.Parameter(torch.full((d_model,), float(init)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.gamma

    class _DropPath(nn.Module):
        """Stokastik derinlik (Huang ve ark. 2016): eğitimde artık dalı ÖRNEK BAZINDA
        p olasılıkla tamamen düşürür (ölçek 1/(1−p) ile korunur). Değerlendirmede kimlik.
        Parametresizdir; 12 bloklu XL'de düzenlileştirmenin ana kaynağıdır. MC-dropout
        geçişlerinde (model.train()) aktif kalır — belirsizlik örneklemesine katkı verir."""

        def __init__(self, p: float = 0.0) -> None:
            super().__init__()
            self.p = float(p)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.p <= 0.0 or not self.training:
                return x
            keep = 1.0 - self.p
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)   # örnek-başı maske (yayınlanır)
            mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
            return x * mask / keep

    class _PeriodicTokenizer(nn.Module):
        """PLR sayısal gömme (Gorishniy ve ark. 2022, "On Embeddings for Numerical
        Features"): x_j → [sin(2πc_j x_j), cos(2πc_j x_j)] → Linear_j → ReLU → token.

        NEDEN: düz `x·W` tokenizasyonu tek bir doğrusal yönle sınırlıdır; periyodik taban
        aynı özelliğin FARKLI ölçeklerdeki (eşik-benzeri) etkilerini tek katmanda temsil
        eder ve tabular kıyaslarda tutarlı kazanç verir. `c_j` öğrenilebilirdir (N(0,σ)
        ile ilklendirilir). Kimlik + [SCALE] gömmeleri v3 tokenizer'ıyla AYNI semantikte
        korunur — dizi düzeni sözleşmesi (explain.attention_rollout) bozulmaz."""

        def __init__(self, n_features: int, d_model: int, scale_ids: list[int],
                     n_freqs: int = _XL_PLR_FREQS, sigma: float = _XL_PLR_SIGMA) -> None:
            super().__init__()
            self.n_features = int(n_features)
            self.n_freqs = int(n_freqs)
            self.coeffs = nn.Parameter(torch.empty(n_features, n_freqs))
            self.proj_weight = nn.Parameter(torch.empty(n_features, 2 * n_freqs, d_model))
            self.proj_bias = nn.Parameter(torch.zeros(n_features, d_model))
            self.feature_id_embed = nn.Parameter(torch.zeros(n_features, d_model))
            self.scale_type_embed = nn.Embedding(_N_SCALE_TYPES, d_model)
            self.register_buffer("scale_ids", torch.tensor(scale_ids, dtype=torch.long))
            nn.init.normal_(self.coeffs, std=float(sigma))
            nn.init.normal_(self.proj_weight, std=0.02)
            nn.init.normal_(self.feature_id_embed, std=0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, F) → periyodik taban (B, F, 2k) → özellik-başı projeksiyon (B, F, d)
            ang = 2.0 * math.pi * x.unsqueeze(-1) * self.coeffs        # (B, F, k)
            basis = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, F, 2k)
            tok = torch.einsum("bfk,fkd->bfd", basis, self.proj_weight) + self.proj_bias
            tok = nn.functional.relu(tok)
            return tok + self.feature_id_embed + self.scale_type_embed(self.scale_ids)

    class _TransformerBlockXL(nn.Module):
        """v4 bloğu: RMSNorm→MHSA→LayerScale→DropPath→artık, RMSNorm→SwiGLU→LayerScale→
        DropPath→artık. Dikkat modülü BİLİNÇLİ olarak `nn.MultiheadAttention`'dır —
        `explain.attention_rollout` hook'ları bu tipe bakar (özel dikkat implementasyonu
        açıklanabilirliği sessizce öldürürdü)."""

        def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float,
                     drop_path: float = 0.0) -> None:
            super().__init__()
            self.norm1 = _RMSNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.ls1 = _LayerScale(d_model)
            self.dp1 = _DropPath(drop_path)
            self.norm2 = _RMSNorm(d_model)
            self.ffn = _SwiGLU(d_model, ffn_dim, dropout)
            self.ls2 = _LayerScale(d_model)
            self.dp2 = _DropPath(drop_path)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm1(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + self.dp1(self.ls1(attn_out))
            return x + self.dp2(self.ls2(self.ffn(self.norm2(x))))

    # ---------------------- v5 (Tur-32) seyrek uzman karışımı ----------------------
    class _MoEFFN(nn.Module):
        """Seyrek Uzman Karışımı FFN (Shazeer ve ark. 2017 · Fedus ve ark. 2021 "Switch").

        Tek bir SwiGLU yerine E adet UZMAN SwiGLU tutar; her token için yönlendirici
        (router) en iyi `top_k` uzmanı seçer ve çıktıları normalize edilmiş kapı
        olasılıklarıyla harmanlar. SONUÇ: parametre sayısı E/top_k katına çıkarken TOKEN
        BAŞINA hesap sabit kalır.

        NEDEN BU PROJEDE ANLAMLI: kullanıcı isteği "parametre sayısını artır" iken yoğun
        (dense) genişletme hesabı da aynı oranda büyütür ve bu donanımda eğitim İMKÂNSIZ
        hâle gelir (Tur-22 §7-1'de ÖLÇÜLDÜ: XL süpürmesi = 21 gün). MoE kapasiteyi hesaptan
        AYIRIR — merdivenin üst basamağını gerçekten tırmanılabilir tutan tek yapı budur.
        Ayrıca finansal veride uzmanların rejime göre uzmanlaşması doğal bir tümevarım
        önyargısıdır (yüksek-vol / düşük-vol / trend rejimleri farklı fonksiyonlar ister).

        DÜRÜSTLÜK: kapasite artışı bir KENAR iddiası değildir (ACI DERS #5). Yönlendirmenin
        rejime göre uzmanlaştığı bu turda ÖLÇÜLMEMİŞTİR — mimari olanak sağlar, kanıt ayrı
        bir iştir.

        DETERMİNİZM: gürültülü (noisy) kapı KULLANILMAZ — yönlendirme argmax tabanlıdır ve
        rastgelelik içermez (anayasa madde 6: aynı seed → bit-bit aynı çıktı).

        `balance_loss`: Switch yük-dengeleme yardımcı kaybı  E·Σ_e f_e·P_e  (f_e = e'ye
        yönlenen token oranı, P_e = e'nin ortalama kapı olasılığı). Dengeli dağılımda
        değeri 1'e yakındır, çöküşte E'ye yaklaşır. forward() sırasında yazılır ve
        `QuantTransformerMoE` tarafından toplanıp çıktı sözlüğüne konur."""

        def __init__(self, d_model: int, ffn_dim: int, n_experts: int, top_k: int,
                     dropout: float) -> None:
            super().__init__()
            if n_experts < 2 or not 1 <= top_k <= n_experts:
                raise ValueError("n_experts >= 2 ve 1 <= top_k <= n_experts olmalı")
            self.n_experts = int(n_experts)
            self.top_k = int(top_k)
            self.router = nn.Linear(d_model, n_experts, bias=False)
            self.experts = nn.ModuleList(
                [_SwiGLU(d_model, ffn_dim, dropout) for _ in range(n_experts)])
            nn.init.normal_(self.router.weight, std=0.02)
            self.balance_loss: torch.Tensor | None = None
            # Yönlendirme TEŞHİSİ (Tur-34): açıkken her forward'da top-1 atamalarını saklar.
            # VARSAYILAN KAPALI — 4096×100 token × blok başına int64 tamponu ölçülebilir
            # bellektir ve eğitim/çıkarım yolunun ona ihtiyacı yoktur (`moe_routing_stats`
            # bayrağı geçici olarak açar, ölçer, kapatır).
            self.record_routing = False
            self.last_top1: torch.Tensor | None = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            probs = torch.softmax(self.router(flat), dim=-1)          # (N, E)
            top_p, top_i = torch.topk(probs, self.top_k, dim=-1)      # (N, k)
            top_p = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            out = torch.zeros_like(flat)
            for e in range(self.n_experts):
                hit = top_i == e                                       # (N, k)
                if not bool(hit.any()):
                    continue
                idx, kpos = hit.nonzero(as_tuple=True)
                # index_add_: aynı token birden çok kez seçilemez (topk tekildir), yine de
                # toplama kullanılır — uzmanlar arası harman doğal olarak toplamdır.
                # `.to(out.dtype)`: autocast (amp=True) altında uzman çıktısı bf16, kapı
                # olasılığı float32 kalır → çarpım float32 olur ve index_add_ tip uyuşmazlığı
                # verir. amp kapalıyken tipler zaten aynıdır, cast NO-OP'tur.
                out.index_add_(0, idx, (self.experts[e](flat[idx])
                                        * top_p[idx, kpos].unsqueeze(-1)).to(out.dtype))
            # Yük dengeleme: f_e top-1 seçim oranı, P_e ortalama kapı olasılığı.
            top1 = top_i[:, 0]
            f_e = torch.zeros(self.n_experts, device=flat.device, dtype=probs.dtype)
            f_e.index_add_(0, top1, torch.ones_like(top1, dtype=probs.dtype))
            f_e = f_e / max(1, top1.numel())
            self.balance_loss = self.n_experts * torch.sum(f_e * probs.mean(dim=0))
            if self.record_routing:
                # (N,) düz atama — çağıran (B, seq) olarak yeniden şekillendirir.
                self.last_top1 = top1.detach().clone()
            return out.reshape(shape)

    class _TransformerBlockMoE(nn.Module):
        """XL bloğunun MoE varyantı: yoğun SwiGLU yerine `_MoEFFN`. Dikkat/RMSNorm/
        LayerScale/DropPath katmanları XL ile AYNI (yalnız FFN değişti) — böylece iki
        basamak arasındaki fark TEK değişkene indirgenir ve kıyas anlamlı olur."""

        def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float,
                     n_experts: int, top_k: int, drop_path: float = 0.0) -> None:
            super().__init__()
            self.norm1 = _RMSNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.ls1 = _LayerScale(d_model)
            self.dp1 = _DropPath(drop_path)
            self.norm2 = _RMSNorm(d_model)
            self.ffn = _MoEFFN(d_model, ffn_dim, n_experts, top_k, dropout)
            self.ls2 = _LayerScale(d_model)
            self.dp2 = _DropPath(drop_path)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm1(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + self.dp1(self.ls1(attn_out))
            return x + self.dp2(self.ls2(self.ffn(self.norm2(x))))

    # ---------------------- v6 (Tur-34) ince-taneli + paylaşımlı MoE ----------------------
    class _FineGrainedMoEFFN(nn.Module):
        """DeepSeekMoE tarzı FFN (Dai ve ark. 2024): İNCE TANELİ yönlendirilen uzmanlar +
        her token'da KOŞULSUZ çalışan PAYLAŞIMLI uzman.

        `_MoEFFN`'den (Tur-32) iki yapısal fark:

        1. **İnce tane.** Uzman gizli genişliği yarıya iner, uzman sayısı katlanır. Aktif
           bütçe (top_k·h) sabit tutulurken seçilebilecek uzman KOMBİNASYONU C(E,k) ile
           patlar: 6 uzmandan 2 seçmek 15 farklı harman verir, 16'dan 2 seçmek **120**.
           Yönlendiricinin ifade gücü buradan gelir — parametreden değil.
        2. **Paylaşımlı uzman.** Tüm rejimlerde ortak olan dönüşümü ayrı bir daima-açık
           uzman taşır. Onsuz her yönlendirilen uzman aynı ortak bilgiyi ayrı ayrı
           öğrenmek zorundadır (parametre israfı) ve uzmanlaşma gerçekleşmez.

        `balance_loss` Switch yardımcı kaybıdır (`_MoEFFN` ile AYNI tanım — YALNIZ
        yönlendirilen uzmanlar üzerinden; paylaşımlı uzman dengelemeye girmez çünkü onun
        yükü tanımı gereği %100'dür). `z_loss` ST-MoE (Zoph ve ark. 2022) yönlendirici
        logit-ölçeği cezasıdır: logsumexp(logits)². İki terim FARKLI patolojileri hedefler
        — denge dağılımın DÜZLÜĞÜNE, z-kaybı logitlerin BÜYÜKLÜĞÜNE bakar.

        DETERMİNİZM: yönlendirme argmax tabanlıdır, gürültülü kapı YOKTUR (anayasa madde 6).
        """

        def __init__(self, d_model: int, expert_ffn: int, n_routed: int, top_k: int,
                     shared_ffn: int, dropout: float) -> None:
            super().__init__()
            if n_routed < 2 or not 1 <= top_k <= n_routed:
                raise ValueError("n_routed >= 2 ve 1 <= top_k <= n_routed olmalı")
            if shared_ffn < 1:
                raise ValueError("shared_ffn >= 1 olmalı")
            self.n_routed = int(n_routed)
            self.top_k = int(top_k)
            self.router = nn.Linear(d_model, n_routed, bias=False)
            self.experts = nn.ModuleList(
                [_SwiGLU(d_model, expert_ffn, dropout) for _ in range(n_routed)])
            self.shared = _SwiGLU(d_model, shared_ffn, dropout)
            nn.init.normal_(self.router.weight, std=0.02)
            self.balance_loss: torch.Tensor | None = None
            self.z_loss: torch.Tensor | None = None
            self.record_routing = False
            self.last_top1: torch.Tensor | None = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            logits = self.router(flat)                                 # (N, E)
            probs = torch.softmax(logits, dim=-1)
            top_p, top_i = torch.topk(probs, self.top_k, dim=-1)       # (N, k)
            top_p = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            out = self.shared(flat)          # paylaşımlı dal: KOŞULSUZ, yönlendirme yok
            for e in range(self.n_routed):
                hit = top_i == e
                if not bool(hit.any()):
                    continue
                idx, kpos = hit.nonzero(as_tuple=True)
                # `.to(out.dtype)`: autocast (amp=True) altında uzman çıktısı bf16, kapı
                # olasılığı float32 kalır → çarpım float32 olur ve index_add_ tip uyuşmazlığı
                # verir. amp kapalıyken tipler zaten aynıdır, cast NO-OP'tur.
                out.index_add_(0, idx, (self.experts[e](flat[idx])
                                        * top_p[idx, kpos].unsqueeze(-1)).to(out.dtype))
            top1 = top_i[:, 0]
            f_e = torch.zeros(self.n_routed, device=flat.device, dtype=probs.dtype)
            f_e.index_add_(0, top1, torch.ones_like(top1, dtype=probs.dtype))
            f_e = f_e / max(1, top1.numel())
            self.balance_loss = self.n_routed * torch.sum(f_e * probs.mean(dim=0))
            self.z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
            if self.record_routing:
                self.last_top1 = top1.detach().clone()
            return out.reshape(shape)

    class _TransformerBlockMoEXL(nn.Module):
        """MoE bloğunun ince-taneli varyantı: `_MoEFFN` yerine `_FineGrainedMoEFFN`.
        Dikkat/RMSNorm/LayerScale/DropPath katmanları XL/MoE ile AYNI — böylece basamaklar
        arasındaki fark yine TEK değişkene (FFN) iner ve kıyas anlamlı kalır."""

        def __init__(self, d_model: int, n_heads: int, expert_ffn: int, dropout: float,
                     n_routed: int, top_k: int, shared_ffn: int,
                     drop_path: float = 0.0) -> None:
            super().__init__()
            self.norm1 = _RMSNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.ls1 = _LayerScale(d_model)
            self.dp1 = _DropPath(drop_path)
            self.norm2 = _RMSNorm(d_model)
            self.ffn = _FineGrainedMoEFFN(d_model, expert_ffn, n_routed, top_k,
                                          shared_ffn, dropout)
            self.ls2 = _LayerScale(d_model)
            self.dp2 = _DropPath(drop_path)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm1(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + self.dp1(self.ls1(attn_out))
            return x + self.dp2(self.ls2(self.ffn(self.norm2(x))))
            return x + self.dp2(self.ls2(self.ffn(self.norm2(x))))

    # ---------------------- v7 (Tur-35) ortak kodlayıcı + quantile ----------------------
    def _monotone_quantiles(raw: torch.Tensor) -> torch.Tensor:
        """Ham başlık çıktısı (B,Q) → AZALMAYAN quantile demeti. q₀ = raw₀,
        q_i = q_{i−1} + softplus(raw_i).

        NEDEN kümülatif: bağımsız Q çıkış nöronu eğitilirse quantile'lar birbirini KESER
        (q₀.₉ < q₀.₁ gibi anlamsız çıktı üretir) ve bu kayıp fonksiyonuyla cezalandırılmaz
        — pinball kaybı her quantile'ı AYRI AYRI ölçer. Kümülatif kurulum çaprazlamayı
        yasaklamaz, YAPISI GEREĞİ imkânsız kılar; kısıt kayba değil mimariye yazılır.

        İDDİANIN SINIRI (ölçüldü, Tur-35): garanti "AZALMAYAN"dır, "kesin artan" DEĞİL.
        softplus(x) çok negatif x'te float32'de tam 0.0'a alt-taşar → iki komşu quantile
        EŞİTLENEBİLİR. Bu bir çapraz geçiş değildir (sıra bozulmaz) ve dejenere/aşırı
        girdide beklenen davranıştır; ama "kesin artan" demek ölçülemeyen bir iddia olurdu
        (Tur-33'ün "hangi iddianın kurulabileceğini algoritma belirler" kuralı)."""
        first = raw[:, :1]
        steps = nn.functional.softplus(raw[:, 1:])
        return torch.cat([first, first + torch.cumsum(steps, dim=1)], dim=1)

    def _pinball_loss(q_pred: torch.Tensor, target: torch.Tensor,
                      taus: tuple[float, ...]) -> torch.Tensor:
        """Pinball (quantile/çek-defteri) kaybı: Σ_τ max(τ·e, (τ−1)·e), e = y − q̂_τ.
        Minimize eden q̂_τ, koşullu dağılımın τ-quantile'ıdır (Koenker & Bassett 1978) —
        MSE'nin ORTALAMAyı vermesi gibi. Ortalama alınır (Q ve B üzerinden)."""
        t = torch.as_tensor(taus, dtype=q_pred.dtype, device=q_pred.device)
        e = target.reshape(-1, 1) - q_pred
        return torch.mean(torch.maximum(t * e, (t - 1.0) * e))

    # ------------------- v8 (Tur-36) kanıt-temelli yön + opt-in başlıklar -------------------
    def _dirichlet_alpha(raw: torch.Tensor) -> torch.Tensor:
        """Ham başlık çıktısı → Dirichlet konsantrasyonu α = softplus(raw) + 1 (α ≥ 1).

        NEDEN softplus+1: Dirichlet parametresi POZİTİF olmak zorundadır ve α = 1 "kanıt
        yok" (düz/uniform) noktasıdır. exp() de pozitiftir ama üstel büyür ve eğitimde
        taşar; softplus doğrusal büyür. Kanıt e = α − 1 ≥ 0 doğal olarak "sıfır kanıt"
        tabanına oturur — Sensoy ve ark. 2018 §3."""
        return nn.functional.softplus(raw) + 1.0

    def _evidential_loss(alpha: torch.Tensor, y: torch.Tensor,
                         anneal: float = 1.0) -> torch.Tensor:
        """EDL kaybı: Bayes-riski CE (Tip-II ML) + tavlanan KL cezası (Sensoy ve ark. 2018).

        Terim 1 — `Σ_k y_k (ψ(S) − ψ(α_k))`: Dirichlet ALTINDA beklenen çapraz-entropi.
        Softmax-CE'den farkı, olasılığı nokta değil DAĞILIM sayması; model "emin değilim"
        derse (α→1) ceza sınırlıdır, yanlış tarafta EMİN olursa ceza büyür.

        Terim 2 — yanlış sınıfların kanıtı düz Dirichlet'e çekilir: `KL(D(α̃) ‖ D(1))`,
        α̃ = y + (1−y)⊙α (doğru sınıfın kanıtı KORUNUR, yalnız YANILTICI kanıt cezalanır).
        `anneal` ∈ [0,1] katsayısı ilk epoklarda 0'a yakındır: kanıt daha oluşmadan
        cezalandırılırsa model hiç kanıt üretmemeyi öğrenir (makalenin §4 uyarısı).

        DETERMİNİZM: rastgelelik yok; yalnız digamma/lgamma (anayasa madde 6)."""
        s = alpha.sum(dim=1, keepdim=True)
        y1h = nn.functional.one_hot(y.long(), num_classes=alpha.shape[1]).to(alpha.dtype)
        ce = torch.sum(y1h * (torch.digamma(s) - torch.digamma(alpha)), dim=1)
        at = y1h + (1.0 - y1h) * alpha           # doğru sınıfın kanıtı sökülür (α̃_k = 1)
        st = at.sum(dim=1, keepdim=True)
        k = float(alpha.shape[1])
        kl = (torch.lgamma(st).squeeze(1) - torch.lgamma(at).sum(dim=1)
              - math.lgamma(k)
              + torch.sum((at - 1.0) * (torch.digamma(at) - torch.digamma(st)), dim=1))
        return torch.mean(ce + _EVID_KL_WEIGHT * float(anneal) * kl)

    def _init_extra_heads(module: Any, extra: tuple[str, ...], make: Any,
                          n_vol_q: int) -> None:
        """Opt-in ek başlıkları KANONİK sırada kurar (`make(n_out) -> nn.Module`).

        Sıra çağıranın verdiği sıradan BAĞIMSIZDIR ve bu ÖNEMLİDİR: modül kurulum sırası
        `torch.manual_seed` akışından hangi ağırlığın hangi rastgele sayıyı alacağını
        belirler — aynı KÜME hep aynı ağırlıkları vermelidir.

        `extra` BOŞ ise hiçbir şey kurulmaz → RNG akışı, parametre sayısı ve state_dict
        anahtarları v7 ile BİT-DENK kalır (bu, özelliğin geriye uyumunun tek dayanağıdır).
        Çağıran bu fonksiyonu `vol_head` ile `regime_head` ARASINDA çağırmalıdır — ULTRA'nın
        Tur-35'teki kurulum sırası orasıydı."""
        unknown = [e for e in extra if e not in EXTRA_HEADS]
        if unknown:
            raise ValueError(f"bilinmeyen extra_head: {unknown} (geçerli: {list(EXTRA_HEADS)})")
        if "vol_q" in extra:
            module.vol_quantile_head = make(int(n_vol_q))
        if "dir_evid" in extra:
            module.dir_evidence_head = make(2)

    class _SeqEncoderMixin:
        """DÖRT+ transformer basamağının ORTAK gövdesi (Tur-35'te tekilleştirildi).

        Önceden her sınıf `forward`'ında aynı 12 satırı kopyalıyordu (tokenize → [CLS] +
        3×[SCALE] + özellik dizisi → blok yığını → son norm → füzyon → 6 başlık). Kopya,
        maskeli ön-eğitimin (aşağıda) beşinci bir kopya daha doğurmasını gerektirirdi;
        onun yerine gövde TEK yere alındı ve ön-eğitim AYNI gövdeyi `mask` argümanıyla
        çağırır — "ön-eğitim ile eğitim aynı hesabı görür" bir yorum değil, YAPISAL sonuç.

        BİT-DENKLİK: işlem sırası/tipi birebir korundu; Tur-34 çıktılarıyla 62 tensörde
        fark 0 ölçüldü (bkz. Tur-35 raporu §2). `final_norm` (v4+) ile `final_ln` (v3)
        ayrımı `getattr` ile çözülür — state_dict anahtarları DEĞİŞMEZ."""

        # Yalnız ANNOTATION (değer atanmaz) → çalışma zamanında sınıf özniteliği YARATMAZ,
        # `nn.Module.__setattr__` davranışı ve state_dict anahtarları etkilenmez. Amaç mypy'a
        # "bunları mixin'i kullanan sınıfın __init__'i kurar" demek.
        tokenizer: Any
        cls_token: Any
        scale_tokens: Any
        pos_embed: Any
        blocks: Any
        fusion: Any
        log_temperature: Any
        direction_head: Any
        magnitude_head: Any
        vol_head: Any
        regime_head: Any
        probe_head1: Any
        probe_head2: Any
        final_ln: Any

        def _encode(self, x: torch.Tensor, *, mask: torch.Tensor | None = None,
                    mask_token: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            """x:(B,F) → (seq:(B,4+F,d), moe_balance|None, moe_zloss|None).

            `mask`:(B,F) bool — True olan özelliğin token'ı `mask_token` ile DEĞİŞTİRİLİR
            (sıfırlanmaz: sıfır da geçerli bir normalize değerdir, ağ 'maskelendi'yi
            'değeri 0' sanardı). Yalnız ön-eğitim yolu kullanır; None → v6 ile bit-denk."""
            b = x.shape[0]
            tok = self.tokenizer(x)
            if mask is not None and mask_token is not None:
                tok = torch.where(mask.unsqueeze(-1), mask_token.to(tok.dtype), tok)
            seq = torch.cat([self.cls_token.expand(b, -1, -1),
                             self.scale_tokens.expand(b, -1, -1), tok], dim=1) + self.pos_embed
            balance = torch.zeros((), device=x.device, dtype=seq.dtype)
            zloss = torch.zeros((), device=x.device, dtype=seq.dtype)
            n_bal = n_z = 0
            for block in self.blocks:
                seq = block(seq)
                # getattr: ModuleList yinelemesi mypy'da `Tensor | Module` görünür; dinamik
                # erişim hem tip denetimini hem "MoE olmayan blok" durumunu birlikte çözer.
                ffn = getattr(block, "ffn", None)
                bal = getattr(ffn, "balance_loss", None)
                if bal is not None:
                    balance = balance + bal
                    n_bal += 1
                zl = getattr(ffn, "z_loss", None)
                if zl is not None:
                    zloss = zloss + zl
                    n_z += 1
            final = getattr(self, "final_norm", None)
            if final is None:
                final = self.final_ln
            return final(seq), (balance if n_bal else None), (zloss if n_z else None)

        def _decode(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
            """Kodlanmış diziden görev çıktıları: [CLS] + füzyon([SCALE]×3) → 6 başlık
            (+ varsa `vol_q` quantile demeti) ve öğrenilebilir sıcaklıkla ölçeklenmiş yön."""
            b = seq.shape[0]
            rep = seq[:, 0] + self.fusion(seq[:, 1:4].reshape(b, -1))
            temperature = torch.exp(self.log_temperature)
            out = {
                "direction": self.direction_head(rep) / temperature,
                "magnitude": self.magnitude_head(rep),
                "vol": self.vol_head(rep),
                "regime": self.regime_head(rep),
                "probe1": self.probe_head1(rep),
                "probe2": self.probe_head2(rep),
            }
            qhead = getattr(self, "vol_quantile_head", None)
            if qhead is not None:
                out["vol_q"] = _monotone_quantiles(qhead(rep))
            ehead = getattr(self, "dir_evidence_head", None)
            if ehead is not None:
                # Kanıt başlığı SICAKLIKLA ölçeklenmez: sıcaklık `direction` logitlerinin
                # kalibrasyon düzeltmesidir, Dirichlet kanıtının değil (α'yı bölmek kanıt
                # BÜYÜKLÜĞÜNÜ = belirsizliği bozardı; ikisi ayrı ölçeklerdir).
                out["dir_evid"] = _dirichlet_alpha(ehead(rep))
            return out

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            seq, balance, zloss = self._encode(x)
            out = self._decode(seq)
            n_blocks = max(1, len(self.blocks))
            if balance is not None:
                out["moe_balance"] = balance / n_blocks
            if zloss is not None:
                out["moe_zloss"] = zloss / n_blocks
            return out

        def encode_masked(self, x: torch.Tensor, mask: torch.Tensor,
                          mask_token: torch.Tensor) -> torch.Tensor:
            """Ön-eğitim yolu: maskeli dizinin YALNIZ ÖZELLİK token'larını döndürür
            (ilk 4 konum [CLS]+[SCALE]×3 rekonstrüksiyona girmez — onların 'gerçek
            değeri' yoktur). Şekil (B, F, d)."""
            seq, _, _ = self._encode(x, mask=mask, mask_token=mask_token)
            return seq[:, 4:]

    class _MaskedFeatureHead(nn.Module):
        """Maskelenmiş özelliğin normalize SKALER değerini geri kuran ön-eğitim başlığı.

        Özellikler arasında PAYLAŞILIR — hangi özellik olduğu bilgisi zaten token'ın
        kimlik/[SCALE] gömmesinde taşınır, ikinci kez parametreye yazmak israftır.
        Ön-eğitim bitince ATILIR: `state_dict` artefaktına girmez, bu yüzden ön-eğitilmiş
        bir gövde ile sıfırdan eğitilmiş bir gövde AYNI dosya şemasını paylaşır."""

        def __init__(self, d_model: int, hidden: int = _PRETRAIN_HEAD_HIDDEN) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

        def forward(self, seq_feat: torch.Tensor) -> torch.Tensor:
            return self.net(seq_feat).squeeze(-1)     # (B, F, d) → (B, F)

    class DeepMLP(nn.Module):
        """Ara basamak (plan BÖLÜM D: (512,256,128,64) MLP, ~×100 merdiven sürekliliği).
        Düz ileri-beslemeli gövde + 4 çok-görevli başlık (yön/büyüklük/vol/rejim);
        gerçek parametre sayısı `feature_names` uzunluğuna göre değişir (kapalı-form
        sabit değildir — models.py'nin gbm_small'ı gibi veri/girdi-bağımlı, dürüstçe
        `param_count()` ile ölçülür)."""

        def __init__(self, feature_names: list[str], hidden: tuple[int, ...] = (512, 256, 128, 64),
                     dropout: float = _DROPOUT, magnitude_buckets: int = MAGNITUDE_BUCKETS,
                     regime_classes: int = REGIME_CLASSES,
                     extra_heads: tuple[str, ...] = (),
                     vol_quantiles: tuple[float, ...] = VOL_QUANTILES) -> None:
            super().__init__()
            n_features = len(feature_names)
            if n_features < 1:
                raise ValueError("feature_names boş olamaz")
            layers: list[nn.Module] = []
            prev = n_features
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
                prev = h
            self.backbone = nn.Sequential(*layers)
            self.vol_quantiles = tuple(float(q) for q in vol_quantiles)
            self.direction_head = nn.Linear(prev, 2)
            self.magnitude_head = nn.Linear(prev, magnitude_buckets)
            self.vol_head = nn.Linear(prev, 1)
            # Ek başlıklar gövdenin geri kalanıyla AYNI stilde (düz Linear) kurulur; boş
            # `extra_heads` hiçbir modül yaratmaz → canlı deep_mlp artefaktı bit-denk.
            _init_extra_heads(self, tuple(extra_heads), lambda k: nn.Linear(prev, k),
                              len(self.vol_quantiles))
            self.regime_head = nn.Linear(prev, regime_classes)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            h = self.backbone(x)
            out = {
                "direction": self.direction_head(h),
                "magnitude": self.magnitude_head(h),
                "vol": self.vol_head(h),
                "regime": self.regime_head(h),
            }
            qhead = getattr(self, "vol_quantile_head", None)
            if qhead is not None:
                out["vol_q"] = _monotone_quantiles(qhead(h))
            ehead = getattr(self, "dir_evidence_head", None)
            if ehead is not None:
                out["dir_evid"] = _dirichlet_alpha(ehead(h))
            return out

    class QuantTransformer(_SeqEncoderMixin, nn.Module):
        """Hedef mimari (plan BÖLÜM D, rev.3): ≈5.5M parametre. Sıra = [CLS] + 3×[SCALE]
        (h/d/w özet-çapaları) + n_features token; 6× pre-LN transformer bloğu (d=256,
        4 başlık, FFN=1024); çıkışta [CLS] temsili + [SCALE] üçlüsünün füzyon başlığı
        (768→512→256, GELU) toplanır → 4 çok-görevli başlık (yön/büyüklük/vol/rejim) +
        2 transfer-probe başlığı (WP-E için mimari yer tutucu — bu turda hedefsiz, yine
        de parametreleri sayılır) + öğrenilebilir sıcaklık (yön logitlerini ölçekler;
        kanonik kalibrasyon YİNE DE learning.fit_calibrator/apply_calibration'dadır —
        bu yalnız mimari kanca, ana görev kaybı üzerinden dolaylı gradyan alır).

        TEST SÖZLEŞMESİ: `sum(p.numel() for p in QuantTransformer(feature_names).parameters())
        >= 3_777_000` — blok+füzyon+başlık parametreleri `n_features`'tan BAĞIMSIZDIR
        (self-attention'da token-başı ağırlık yok), bu yüzden sözleşme herhangi bir
        makul özellik sayısında güvenle sağlanır (tokenizer katkısı ek/marjinal).

        v4 NOTU: bu sınıf DEĞİŞTİRİLMEDİ — canlı gölge köprüsündeki kayıtlı artefaktlar
        (`state_dict` anahtarları) ve determinizm testleri bit-denk kalsın diye. Modern
        yapı taşları ayrı bir sınıfta (`QuantTransformerXL`) toplanmıştır."""

        def __init__(self, feature_names: list[str], d_model: int = 256, n_blocks: int = 6,
                     n_heads: int = 4, ffn_dim: int = 1024, dropout: float = _DROPOUT,
                     magnitude_buckets: int = MAGNITUDE_BUCKETS,
                     regime_classes: int = REGIME_CLASSES,
                     extra_heads: tuple[str, ...] = (),
                     vol_quantiles: tuple[float, ...] = VOL_QUANTILES) -> None:
            super().__init__()
            n_features = len(feature_names)
            if n_features < 1:
                raise ValueError("feature_names boş olamaz")
            scale_ids = [_feature_scale_id(n) for n in feature_names]
            self.n_features = n_features
            self.d_model = d_model
            self.tokenizer = _FeatureTokenizer(n_features, d_model, scale_ids)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.scale_tokens = nn.Parameter(torch.zeros(1, 3, d_model))
            seq_len = n_features + 1 + 3
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.scale_tokens, std=0.02)
            nn.init.normal_(self.pos_embed, std=0.02)
            self.blocks = nn.ModuleList(
                [_TransformerBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_blocks)])
            self.final_ln = nn.LayerNorm(d_model)
            self.fusion = nn.Sequential(
                nn.Linear(3 * d_model, 512), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(512, d_model),
            )
            self.vol_quantiles = tuple(float(q) for q in vol_quantiles)
            self.direction_head = _Head(d_model, 64, 2, dropout)
            self.magnitude_head = _Head(d_model, 64, magnitude_buckets, dropout)
            self.vol_head = _Head(d_model, 64, 1, dropout)
            _init_extra_heads(self, tuple(extra_heads),
                              lambda k: _Head(d_model, 64, k, dropout),
                              len(self.vol_quantiles))
            self.regime_head = _Head(d_model, 64, regime_classes, dropout)
            self.probe_head1 = _Head(d_model, 64, 2, dropout)
            self.probe_head2 = _Head(d_model, 64, 2, dropout)
            self.log_temperature = nn.Parameter(torch.zeros(1))

        # forward: `_SeqEncoderMixin` (Tur-35'te tekilleştirildi — bit-denk).

    class QuantTransformerXL(_SeqEncoderMixin, nn.Module):
        """v4 hedef mimarisi (Tur-22, plan BÖLÜM D rev.4): ≈30M parametre — v3'ün ≈5.5×'i.

        Sıra düzeni v3 ile AYNI ([CLS] + 3×[SCALE] + n_features) → `explain.attention_rollout`
        etiketleme sözleşmesi korunur. Farklar:
          · Tokenizasyon: PLR periyodik gömme (`_PeriodicTokenizer`) — düz doğrusal yerine.
          · Blok: 12× `_TransformerBlockXL` (d=384, 6 başlık, SwiGLU FFN=1536, RMSNorm,
            LayerScale, doğrusal artan stokastik derinlik 0→0.1).
          · Füzyon: 3d→768→d, SwiGLU-benzeri değil, GELU (v3 ile simetrik okunabilirlik).
          · Başlıklar: d→128→çıkış (v3'te 64) — 4 görev + 2 probe + öğrenilebilir sıcaklık.

        TEST SÖZLEŞMESİ: parametre sayısı ≥ 25.000.000 ve `n_features`'tan BAĞIMSIZ taban
        (blok yığını tek başına ≈28.4M) — tek özellikli listeyle bile eşik geçilir.

        MALİYET UYARISI (dürüstlük): bu katman CPU'da eğitim için ağırdır (≈30M parametre,
        blok başına SwiGLU). Kapasite MERDİVENİNİN üst basamağıdır; `recommend_tier` onu
        yalnız çok büyük n'de önerir ve promotion yine kanıt-kapılıdır."""

        def __init__(self, feature_names: list[str], d_model: int = _XL_D_MODEL,
                     n_blocks: int = _XL_BLOCKS, n_heads: int = _XL_HEADS,
                     ffn_dim: int = _XL_FFN, dropout: float = _DROPOUT,
                     drop_path_max: float = _XL_DROP_PATH_MAX,
                     magnitude_buckets: int = MAGNITUDE_BUCKETS,
                     regime_classes: int = REGIME_CLASSES,
                     extra_heads: tuple[str, ...] = (),
                     vol_quantiles: tuple[float, ...] = VOL_QUANTILES) -> None:
            super().__init__()
            n_features = len(feature_names)
            if n_features < 1:
                raise ValueError("feature_names boş olamaz")
            scale_ids = [_feature_scale_id(n) for n in feature_names]
            self.n_features = n_features
            self.d_model = d_model
            self.tokenizer = _PeriodicTokenizer(n_features, d_model, scale_ids)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.scale_tokens = nn.Parameter(torch.zeros(1, 3, d_model))
            seq_len = n_features + 1 + 3
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.scale_tokens, std=0.02)
            nn.init.normal_(self.pos_embed, std=0.02)
            # Stokastik derinlik doğrusal artar: ilk blok 0, son blok drop_path_max
            # (derin bloklar daha çok düzenlileştirilir — standart "linear decay" kuralı).
            dps = ([0.0] if n_blocks == 1 else
                   [drop_path_max * i / (n_blocks - 1) for i in range(n_blocks)])
            self.blocks = nn.ModuleList([
                _TransformerBlockXL(d_model, n_heads, ffn_dim, dropout, dps[i])
                for i in range(n_blocks)])
            self.final_norm = _RMSNorm(d_model)
            self.fusion = nn.Sequential(
                nn.Linear(3 * d_model, 768), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(768, d_model),
            )
            h = _XL_HEAD_HIDDEN
            self.vol_quantiles = tuple(float(q) for q in vol_quantiles)
            self.direction_head = _Head(d_model, h, 2, dropout)
            self.magnitude_head = _Head(d_model, h, magnitude_buckets, dropout)
            self.vol_head = _Head(d_model, h, 1, dropout)
            _init_extra_heads(self, tuple(extra_heads),
                              lambda k: _Head(d_model, h, k, dropout),
                              len(self.vol_quantiles))
            self.regime_head = _Head(d_model, h, regime_classes, dropout)
            self.probe_head1 = _Head(d_model, h, 2, dropout)
            self.probe_head2 = _Head(d_model, h, 2, dropout)
            self.log_temperature = nn.Parameter(torch.zeros(1))

        # forward: `_SeqEncoderMixin` (Tur-35'te tekilleştirildi — bit-denk).

    class QuantTransformerMoE(_SeqEncoderMixin, nn.Module):
        """v5 hedef mimarisi (Tur-32): ≈93M parametre — XL'in ≈3 katı, AKTİF parametre
        (token başına gerçekten çalışan) ise ≈36M, yani XL ile aynı mertebede.

        Sıra düzeni v3/v4 ile AYNI ([CLS] + 3×[SCALE] + n_features) → `explain.
        attention_rollout` sözleşmesi korunur; dikkat modülü yine `nn.MultiheadAttention`.
        Fark TEK kalemdir: her bloğun FFN'i 6 uzmanlı, top-2 yönlendirmeli `_MoEFFN`.

        Çıktı sözlüğü XL ile aynı 6 anahtarı taşır + `moe_balance` (yük dengeleme kaybı,
        blokların toplamı). `_compute_loss` bu anahtarı GÖRÜRSE `_MOE_BALANCE_WEIGHT` ile
        kayba ekler; görmezse (diğer tier'lar) davranış BİT-DENKTİR.

        TEST SÖZLEŞMESİ: parametre sayısı ≥ 85.000.000 ve tabanı `n_features`'tan
        BAĞIMSIZDIR (blok yığını tek başına ≈90M).

        MALİYET UYARISI (dürüstlük — Tur-22'nin ölçülmüş dersi tekrarlanmasın): 93M
        parametre AdamW ile ≈1,1 GB bellek ister ve CPU'da eğitim saatler sürer. Bu basamak
        merdivenin ÜST UCUDUR; `recommend_tier` onu yalnız çok büyük n'de önerir ve
        promotion yine kanıt-kapılı ayrı yoldur ("kapasite hazır" ≠ "kullanılmalı")."""

        def __init__(self, feature_names: list[str], d_model: int = _MOE_D_MODEL,
                     n_blocks: int = _MOE_BLOCKS, n_heads: int = _MOE_HEADS,
                     ffn_dim: int = _MOE_FFN, n_experts: int = _MOE_EXPERTS,
                     top_k: int = _MOE_TOP_K, dropout: float = _DROPOUT,
                     drop_path_max: float = _MOE_DROP_PATH_MAX,
                     magnitude_buckets: int = MAGNITUDE_BUCKETS,
                     regime_classes: int = REGIME_CLASSES,
                     extra_heads: tuple[str, ...] = (),
                     vol_quantiles: tuple[float, ...] = VOL_QUANTILES) -> None:
            super().__init__()
            n_features = len(feature_names)
            if n_features < 1:
                raise ValueError("feature_names boş olamaz")
            scale_ids = [_feature_scale_id(n) for n in feature_names]
            self.n_features = n_features
            self.d_model = d_model
            self.tokenizer = _PeriodicTokenizer(n_features, d_model, scale_ids)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.scale_tokens = nn.Parameter(torch.zeros(1, 3, d_model))
            seq_len = n_features + 1 + 3
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.scale_tokens, std=0.02)
            nn.init.normal_(self.pos_embed, std=0.02)
            dps = ([0.0] if n_blocks == 1 else
                   [drop_path_max * i / (n_blocks - 1) for i in range(n_blocks)])
            self.blocks = nn.ModuleList([
                _TransformerBlockMoE(d_model, n_heads, ffn_dim, dropout, n_experts, top_k, dps[i])
                for i in range(n_blocks)])
            self.final_norm = _RMSNorm(d_model)
            self.fusion = nn.Sequential(
                nn.Linear(3 * d_model, 768), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(768, d_model),
            )
            h = _MOE_HEAD_HIDDEN
            self.vol_quantiles = tuple(float(q) for q in vol_quantiles)
            self.direction_head = _Head(d_model, h, 2, dropout)
            self.magnitude_head = _Head(d_model, h, magnitude_buckets, dropout)
            self.vol_head = _Head(d_model, h, 1, dropout)
            _init_extra_heads(self, tuple(extra_heads),
                              lambda k: _Head(d_model, h, k, dropout),
                              len(self.vol_quantiles))
            self.regime_head = _Head(d_model, h, regime_classes, dropout)
            self.probe_head1 = _Head(d_model, h, 2, dropout)
            self.probe_head2 = _Head(d_model, h, 2, dropout)
            self.log_temperature = nn.Parameter(torch.zeros(1))

        # forward: `_SeqEncoderMixin` (Tur-35'te tekilleştirildi — bit-denk; `moe_balance`
        # anahtarı bloklar dengeleme kaybı ÜRETTİĞİ için otomatik eklenir).

    class QuantTransformerMoEXL(_SeqEncoderMixin, nn.Module):
        """v6 hedef mimarisi (Tur-34): ≈281M parametre — MoE basamağının ≈3 katı, XL'in
        ≈8,9 katı. AKTİF parametre (token başına gerçekten çarpılan) ≈58M.

        Sıra düzeni v3/v4/v5 ile AYNI ([CLS] + 3×[SCALE] + n_features) ve dikkat modülü yine
        `nn.MultiheadAttention` → `explain.attention_rollout` sözleşmesi KORUNUR. Fark yine
        TEK kalemdir: her bloğun FFN'i 16 ince-taneli uzman (top-2) + 1 PAYLAŞIMLI uzman.

        Çıktı sözlüğü MoE ile aynı anahtarları taşır + `moe_zloss`. `_compute_loss` her iki
        anahtarı da YALNIZ VARSA ekler → diğer basamaklar bit-denk kalır.

        TEST SÖZLEŞMESİ: parametre sayısı ≥ 270.000.000 ve tabanı `n_features`'tan
        BAĞIMSIZDIR (blok yığını tek başına ≈278M).

        MALİYET UYARISI (Tur-22/Tur-32 dersinin üçüncü tekrarı): 281M parametre AdamW ile
        ≈3,4 GB optimizer durumu ister; CPU'da eğitim bu donanımda GÜNLER sürer. Bu basamak
        merdivenin YENİ üst ucudur ve **canlı modeli YOKTUR**. Kapasite artışı bir KENAR
        iddiası DEĞİLDİR (ACI DERS #5) — `recommend_tier` onu yalnız n ≥ 3.600.000'de önerir
        ve promotion yine kanıt-kapılı ayrı yoldur."""

        def __init__(self, feature_names: list[str], d_model: int = _MOEXL_D_MODEL,
                     n_blocks: int = _MOEXL_BLOCKS, n_heads: int = _MOEXL_HEADS,
                     expert_ffn: int = _MOEXL_EXPERT_FFN,
                     n_routed: int = _MOEXL_ROUTED_EXPERTS, top_k: int = _MOEXL_TOP_K,
                     shared_ffn: int = _MOEXL_SHARED_FFN, dropout: float = _DROPOUT,
                     drop_path_max: float = _MOEXL_DROP_PATH_MAX,
                     magnitude_buckets: int = MAGNITUDE_BUCKETS,
                     regime_classes: int = REGIME_CLASSES,
                     extra_heads: tuple[str, ...] = (),
                     vol_quantiles: tuple[float, ...] = VOL_QUANTILES) -> None:
            super().__init__()
            n_features = len(feature_names)
            if n_features < 1:
                raise ValueError("feature_names boş olamaz")
            scale_ids = [_feature_scale_id(n) for n in feature_names]
            self.n_features = n_features
            self.d_model = d_model
            self.tokenizer = _PeriodicTokenizer(n_features, d_model, scale_ids)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.scale_tokens = nn.Parameter(torch.zeros(1, 3, d_model))
            seq_len = n_features + 1 + 3
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.scale_tokens, std=0.02)
            nn.init.normal_(self.pos_embed, std=0.02)
            dps = ([0.0] if n_blocks == 1 else
                   [drop_path_max * i / (n_blocks - 1) for i in range(n_blocks)])
            self.blocks = nn.ModuleList([
                _TransformerBlockMoEXL(d_model, n_heads, expert_ffn, dropout, n_routed,
                                       top_k, shared_ffn, dps[i])
                for i in range(n_blocks)])
            self.final_norm = _RMSNorm(d_model)
            self.fusion = nn.Sequential(
                nn.Linear(3 * d_model, 1024), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(1024, d_model),
            )
            h = _MOEXL_HEAD_HIDDEN
            self.vol_quantiles = tuple(float(q) for q in vol_quantiles)
            self.direction_head = _Head(d_model, h, 2, dropout)
            self.magnitude_head = _Head(d_model, h, magnitude_buckets, dropout)
            self.vol_head = _Head(d_model, h, 1, dropout)
            _init_extra_heads(self, tuple(extra_heads),
                              lambda k: _Head(d_model, h, k, dropout),
                              len(self.vol_quantiles))
            self.regime_head = _Head(d_model, h, regime_classes, dropout)
            self.probe_head1 = _Head(d_model, h, 2, dropout)
            self.probe_head2 = _Head(d_model, h, 2, dropout)
            self.log_temperature = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            b = x.shape[0]
            tok = self.tokenizer(x)
            cls = self.cls_token.expand(b, -1, -1)
            scale = self.scale_tokens.expand(b, -1, -1)
            seq = torch.cat([cls, scale, tok], dim=1) + self.pos_embed
            balance = torch.zeros((), device=x.device, dtype=seq.dtype)
            zloss = torch.zeros((), device=x.device, dtype=seq.dtype)
            for block in self.blocks:
                seq = block(seq)
                ffn = getattr(block, "ffn", None)
                bal = getattr(ffn, "balance_loss", None)
                if bal is not None:
                    balance = balance + bal
                zl = getattr(ffn, "z_loss", None)
                if zl is not None:
                    zloss = zloss + zl
            seq = self.final_norm(seq)
            cls_out = seq[:, 0]
            scale_out = seq[:, 1:4].reshape(b, -1)
            rep = cls_out + self.fusion(scale_out)
            temperature = torch.exp(self.log_temperature)
            n_blocks = max(1, len(self.blocks))
            return {
                "direction": self.direction_head(rep) / temperature,
                "magnitude": self.magnitude_head(rep),
                "vol": self.vol_head(rep),
                "regime": self.regime_head(rep),
                "probe1": self.probe_head1(rep),
                "probe2": self.probe_head2(rep),
                "moe_balance": balance / n_blocks,
                "moe_zloss": zloss / n_blocks,
            }

    class QuantTransformerUltra(_SeqEncoderMixin, nn.Module):
        """v7 hedef mimarisi (Tur-35): ≈772M parametre — MoE-XL'in ≈2,7 katı, XL'in ≈24 katı.
        ÖLÇÜLDÜ (f=96): toplam **771.874.072**, AKTİF (token başına gerçekten çarpılan)
        **98.482.456** → seyreklik oranı **7,84×**.

        Sıra düzeni v3..v6 ile AYNI ([CLS] + 3×[SCALE] + n_features) ve dikkat modülü yine
        `nn.MultiheadAttention` → `explain.attention_rollout` sözleşmesi KORUNUR.

        MoE-XL'den İKİ fark:

        1. **Daha ince tane, daha çok uzman.** Yönlendirilen uzman 16 → **40**, uzman gizli
           genişliği 1024 → 768 (yani uzman BAŞINA küçülür, sayı katlanır); d_model 512 →
           640. Seçilebilen uzman kombinasyonu C(16,2)=120 → **C(40,2)=780**. İnce tanenin
           kazancı parametre değil İFADE GÜCÜDÜR (Dai ve ark. 2024 §3) — parametre yalnız
           yan üründür.
        2. **Dağılımsal oynaklık başlığı (`vol_q`).** Nokta `vol` başlığı DURUYOR (MSE) ve
           yanına 5 quantile'lık bir demet geldi (pinball kaybı, çapraz-geçişsiz kümülatif
           kurulum). Sistemin oynaklık tahmini bugüne kadar hep TEK SAYIydı; oysa karar
           için gereken şey aralıktır. Bu, "kapasiteyi büyütürken yalnız büyüklük değil
           YETENEK de ekle" kuralının bu turdaki karşılığıdır.

        Çıktı sözlüğü MoE-XL'in 8 anahtarını taşır + `vol_q` (B, 5). `_compute_loss` pinball
        terimini YALNIZ `vol_q` VE `aux['vol']` birlikte varken ekler → diğer basamaklar
        v6 ile BİT-DENK kalır.

        TEST SÖZLEŞMESİ: parametre sayısı ≥ 740.000.000 ve tabanı `n_features`'tan
        BAĞIMSIZDIR (blok yığını tek başına ≈764M).

        MALİYET UYARISI (Tur-22/32/34 dersinin dördüncü tekrarı — dürüstlük): 771M parametre
        float32'de ≈3,1 GB ağırlık + AdamW ile ≈9,2 GB optimizer durumu ister; CPU'da eğitim
        bu donanımda GÜNLER sürer. Bu basamağın **canlı modeli YOKTUR** ve olması da
        planlanmıyor. Kapasite artışı bir KENAR iddiası DEĞİLDİR (ACI DERS #5) —
        `recommend_tier` onu yalnız n ≥ 10.000.000'de önerir (bugünkü canlı logdan DÖRT
        mertebe uzak) ve promotion yine kanıt-kapılı ayrı yoldur."""

        def __init__(self, feature_names: list[str], d_model: int = _ULTRA_D_MODEL,
                     n_blocks: int = _ULTRA_BLOCKS, n_heads: int = _ULTRA_HEADS,
                     expert_ffn: int = _ULTRA_EXPERT_FFN,
                     n_routed: int = _ULTRA_ROUTED_EXPERTS, top_k: int = _ULTRA_TOP_K,
                     shared_ffn: int = _ULTRA_SHARED_FFN, dropout: float = _DROPOUT,
                     drop_path_max: float = _ULTRA_DROP_PATH_MAX,
                     magnitude_buckets: int = MAGNITUDE_BUCKETS,
                     regime_classes: int = REGIME_CLASSES,
                     extra_heads: tuple[str, ...] = ("vol_q",),
                     vol_quantiles: tuple[float, ...] = VOL_QUANTILES,
                     head_hidden: int = _ULTRA_HEAD_HIDDEN) -> None:
            super().__init__()
            n_features = len(feature_names)
            if n_features < 1:
                raise ValueError("feature_names boş olamaz")
            if len(vol_quantiles) < 2 or any(
                    vol_quantiles[i] >= vol_quantiles[i + 1] for i in range(len(vol_quantiles) - 1)):
                raise ValueError("vol_quantiles KESİN ARTAN ve en az 2 elemanlı olmalı")
            scale_ids = [_feature_scale_id(n) for n in feature_names]
            self.n_features = n_features
            self.d_model = d_model
            self.vol_quantiles = tuple(float(q) for q in vol_quantiles)
            self.tokenizer = _PeriodicTokenizer(n_features, d_model, scale_ids)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.scale_tokens = nn.Parameter(torch.zeros(1, 3, d_model))
            seq_len = n_features + 1 + 3
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.scale_tokens, std=0.02)
            nn.init.normal_(self.pos_embed, std=0.02)
            dps = ([0.0] if n_blocks == 1 else
                   [drop_path_max * i / (n_blocks - 1) for i in range(n_blocks)])
            self.blocks = nn.ModuleList([
                _TransformerBlockMoEXL(d_model, n_heads, expert_ffn, dropout, n_routed,
                                       top_k, shared_ffn, dps[i])
                for i in range(n_blocks)])
            self.final_norm = _RMSNorm(d_model)
            self.fusion = nn.Sequential(
                nn.Linear(3 * d_model, 1280), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(1280, d_model),
            )
            h = int(head_hidden)
            self.direction_head = _Head(d_model, h, 2, dropout)
            self.magnitude_head = _Head(d_model, h, magnitude_buckets, dropout)
            self.vol_head = _Head(d_model, h, 1, dropout)
            # Tur-36: `vol_q` artık gömülü değil OPT-IN — ama ULTRA'nın VARSAYILANI açık ve
            # kurulum SIRASI Tur-35'teki yerinde kaldı → ağırlıklar/state_dict bit-denk.
            _init_extra_heads(self, tuple(extra_heads),
                              lambda k: _Head(d_model, h, k, dropout),
                              len(self.vol_quantiles))
            self.regime_head = _Head(d_model, h, regime_classes, dropout)
            self.probe_head1 = _Head(d_model, h, 2, dropout)
            self.probe_head2 = _Head(d_model, h, 2, dropout)
            self.log_temperature = nn.Parameter(torch.zeros(1))

        # forward: `_SeqEncoderMixin` (`vol_q` anahtarı `vol_quantile_head` VARLIĞINDAN
        # türer — mixin hiçbir basamağı özel-kılıflamaz).

    class QuantTransformerMega(QuantTransformerUltra):
        """v8 hedef mimarisi (Tur-36): ≈2,0 MİLYAR parametre — merdivenin YENİ üst ucu.

        **ULTRA'nın ALT SINIFIDIR ve bu bilinçlidir:** mimari İDENTİK (aynı dizi düzeni,
        aynı `_TransformerBlockMoEXL`, aynı `nn.MultiheadAttention` → `attention_rollout`
        sözleşmesi ayakta), fark YALNIZ ölçek sabitleri ve varsayılan ek başlıklardır.
        70 satırlık bir kopya, basamaklar arası farkı "tek değişken" tutma kuralını
        (Tur-32'den beri) görünmez kılardı.

        ULTRA'dan iki fark:

        1. **Kapasite + seyreklik.** Yönlendirilen uzman 40 → **80**, d_model 640 → 768,
           blok 12 → 16; uzman gizli genişliği yine İNER (768 → 640) — ince tane KORUNUR,
           sayı katlanır. Seçilebilen kombinasyon C(40,2)=780 → **C(80,2)=3160**. Toplam
           parametre ≈2,6× artarken token başına AKTİF parametre yalnız ≈1,6× artar; yani
           basamak "daha çok hesap" değil "daha çok KAPASİTE" ekler (MoE'nin varlık sebebi).
        2. **Kanıt-temelli yön başlığı varsayılan AÇIK** (`extra_heads=("vol_q","dir_evid")`).
           MC-dropout T geçiş harcar; Dirichlet kanıtı epistemik belirsizliği TEK geçişte
           verir. Bu basamak hem en büyük hem de belirsizliği en UCUZ okuyandır.

        TEST SÖZLEŞMESİ: parametre sayısı ≥ 1.900.000.000 ve tabanı `n_features`'tan
        BAĞIMSIZDIR (blok yığını tek başına ≈2,0 milyar).

        MALİYET UYARISI (Tur-22/32/34/35 dersinin BEŞİNCİ tekrarı — dürüstlük): 2,0 milyar
        parametre float32'de ≈8,0 GB ağırlık + AdamW ile ≈24 GB optimizer durumu ister; bu
        donanımda eğitim İMKÂNSIZDIR. Bu basamağın **canlı modeli YOKTUR** ve olması da
        planlanmıyor. `recommend_tier` onu yalnız n ≥ 26.000.000'de önerir (bugünkü canlı
        logdan DÖRT-BUÇUK mertebe uzak). Kapasite artışı bir KENAR iddiası DEĞİLDİR
        (ACI DERS #5) — merdiven veri disiplinini gevşetmek için değil, gevşetmediğini
        GÖSTERMEK için uzar."""

        def __init__(self, feature_names: list[str], d_model: int = _MEGA_D_MODEL,
                     n_blocks: int = _MEGA_BLOCKS, n_heads: int = _MEGA_HEADS,
                     expert_ffn: int = _MEGA_EXPERT_FFN,
                     n_routed: int = _MEGA_ROUTED_EXPERTS, top_k: int = _MEGA_TOP_K,
                     shared_ffn: int = _MEGA_SHARED_FFN, dropout: float = _DROPOUT,
                     drop_path_max: float = _MEGA_DROP_PATH_MAX,
                     magnitude_buckets: int = MAGNITUDE_BUCKETS,
                     regime_classes: int = REGIME_CLASSES,
                     extra_heads: tuple[str, ...] = ("vol_q", "dir_evid"),
                     vol_quantiles: tuple[float, ...] = VOL_QUANTILES) -> None:
            super().__init__(feature_names, d_model=d_model, n_blocks=n_blocks,
                             n_heads=n_heads, expert_ffn=expert_ffn, n_routed=n_routed,
                             top_k=top_k, shared_ffn=shared_ffn, dropout=dropout,
                             drop_path_max=drop_path_max,
                             magnitude_buckets=magnitude_buckets,
                             regime_classes=regime_classes, extra_heads=extra_heads,
                             vol_quantiles=vol_quantiles,
                             head_hidden=_MEGA_HEAD_HIDDEN)

else:  # pragma: no cover - torch kurulu geliştirme/CI ortamında bu dal koşmaz
    DeepMLP = None  # type: ignore[assignment,misc]
    QuantTransformer = None  # type: ignore[assignment,misc]
    QuantTransformerXL = None  # type: ignore[assignment,misc]
    QuantTransformerMoE = None  # type: ignore[assignment,misc]
    QuantTransformerMoEXL = None  # type: ignore[assignment,misc]
    QuantTransformerUltra = None  # type: ignore[assignment,misc]
    QuantTransformerMega = None  # type: ignore[assignment,misc]


def _build_torch_model(tier: str, feature_names: list[str], seed: int,
                       extra_heads: tuple[str, ...] | None = None,
                       **overrides: Any) -> Any:
    """(tier, feature_names, seed[, extra_heads]) → fit-edilmemiş nn.Module.
    `torch.manual_seed(seed)` ÇAĞRIDAN HEMEN ÖNCE uygulanır → ağırlık ilklendirmesi
    deterministik (anayasa madde 6). torch yokken RuntimeError (çağıran NeuralCapacity
    zaten bunu __init__'te elemiş olur; bu fonksiyon yine de savunmacı).

    `extra_heads=None` → basamağın KENDİ varsayılanı (ULTRA `('vol_q',)`, MEGA ikisi de,
    diğerleri boş) → Tur-35 ile bit-denk. Boş tuple ise açıkça "hiç ek başlık yok"tur;
    None ile () farkı bu yüzden korunur."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch kurulu değil — neural katman devre dışı (fail-soft)")
    if tier not in NEURAL_TIERS:
        raise ValueError(f"bilinmeyen tier: {tier!r} (geçerli: {NEURAL_TIERS})")
    kw: dict[str, Any] = {} if extra_heads is None else {"extra_heads": tuple(extra_heads)}
    # Tur-80 (Codex incelemesi): kurulum da süreç-genel RNG'yi tohumlayıp tüketir; kilitsizken
    # başka thread'deki MC-dropout örneklemini ortasından bozuyordu (ölçüldü: load araya girince fark).
    with INFER_LOCK:
        torch.manual_seed(int(seed))
        return _tier_class(tier)(feature_names, **kw, **overrides)


def _tier_class(tier: str) -> Any:
    """tier → mimari sınıfı. Sözlük (if-zinciri DEĞİL) çünkü `_numel_of_built_model` de
    sınıfın KURUCU VARSAYILANLARINI okumak zorunda (bkz. orada) ve iki yerde iki eşleme
    tutmak sapma kaynağıdır."""
    return {
        "deep_mlp": DeepMLP,
        "quant_transformer": QuantTransformer,
        "quant_transformer_xl": QuantTransformerXL,
        "quant_transformer_moe": QuantTransformerMoE,
        "quant_transformer_moe_xl": QuantTransformerMoEXL,
        "quant_transformer_ultra": QuantTransformerUltra,
        "quant_transformer_mega": QuantTransformerMega,
    }[tier]


def _resolve_device(device: str) -> Any:
    """'auto'|'cpu'|'cuda[:i]' → torch.device. 'auto' CUDA varsa GPU, yoksa SESSİZCE CPU
    (ortam eksikliği hata değildir — fail-soft). Bilinmeyen/kurulamayan aygıt → CPU."""
    name = (device or "cpu").strip().lower()
    if name == "auto":
        name = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    if name.startswith("cuda") and not (TORCH_AVAILABLE and torch.cuda.is_available()):
        name = "cpu"
    try:
        return torch.device(name)
    except (RuntimeError, ValueError, TypeError):
        return torch.device("cpu")


def _compute_loss(out: dict, y_dir: Any, aux_batch: dict, *, weight: Any = None,
                  label_smoothing: float = 0.0,
                  quantiles: tuple[float, ...] | None = None,
                  evid_anneal: float = 1.0) -> Any:
    """Çok-görevli ağırlıklı kayıp: ana görev yön (CE) + yalnız BATCH'te VERİLEN yardımcı
    hedefler (magnitude/vol/regime), her biri `_AUX_LOSS_WEIGHT` ile ölçeklenir. Verilmeyen
    aux anahtarı katkı vermez (o başlık bu adımda eğitimsiz kalır — spec §3.1).

    v4: `weight` (sınıf ağırlıkları) ve `label_smoothing` YALNIZ ana göreve uygulanır —
    yardımcı görevler yön dengesizliğinden bağımsızdır ve düzeltilmeleri istenmez.
    İkisi de varsayılan-nötr (None/0.0) olduğunda kayıp v3 ile BİT-DENKTİR."""
    loss = nn.functional.cross_entropy(out["direction"], y_dir, weight=weight,
                                       label_smoothing=float(label_smoothing))
    if "magnitude" in aux_batch:
        loss = loss + _AUX_LOSS_WEIGHT * nn.functional.cross_entropy(
            out["magnitude"], aux_batch["magnitude"])
    if "vol" in aux_batch:
        loss = loss + _AUX_LOSS_WEIGHT * nn.functional.mse_loss(
            out["vol"].squeeze(-1), aux_batch["vol"])
        # v7: DAĞILIMSAL oynaklık — yalnız `vol_q` başlığı OLAN basamakta (ultra) ve yalnız
        # vol hedefi verilmişken. Nokta tahmini (MSE) yerine geçmez, YANINA gelir: biri
        # koşullu ortalamayı, diğeri koşullu quantile'ları öğrenir.
        if "vol_q" in out:
            loss = loss + _AUX_LOSS_WEIGHT * _pinball_loss(
                out["vol_q"], aux_batch["vol"], quantiles or VOL_QUANTILES)
    if "regime" in aux_batch:
        loss = loss + _AUX_LOSS_WEIGHT * nn.functional.cross_entropy(
            out["regime"], aux_batch["regime"])
    # v8: kanıt-temelli yön — YALNIZ `dir_evid` başlığı OLAN modelde. AYNI hedefi (y_dir)
    # görür ama ana CE başlığının YERİNE GEÇMEZ: `predict_proba` yine `direction`'ı okur ve
    # sıcaklıkla kalibre edilir. Bu terim ikinci bir GÖRÜŞ değil, ikinci bir ÖLÇÜdür
    # (belirsizlik). Anahtar yoksa kayıp v7 ile BİT-DENK.
    if "dir_evid" in out:
        loss = loss + _EVID_LOSS_WEIGHT * _evidential_loss(out["dir_evid"], y_dir,
                                                           anneal=evid_anneal)
    # v5: MoE yük dengeleme — YALNIZ modelin çıktısında varsa (yalnız quant_transformer_moe).
    # Diğer tier'larda anahtar yoktur → kayıp v4 ile BİT-DENK kalır.
    if "moe_balance" in out:
        loss = loss + _MOE_BALANCE_WEIGHT * out["moe_balance"]
    # v6: yönlendirici z-kaybı — YALNIZ quant_transformer_moe_xl çıktısında vardır.
    if "moe_zloss" in out:
        loss = loss + _MOE_ZLOSS_WEIGHT * out["moe_zloss"]
    return loss


def _cosine_lr(step: int, total_steps: int, base_lr: float, warmup_frac: float,
               min_lr_frac: float) -> float:
    """Isınmalı kosinüs LR (adım-başı): ilk %`warmup_frac` adımda 0→base_lr doğrusal,
    sonra base_lr→base_lr·min_lr_frac kosinüs. Sabit LR'ye göre kazanç: erken adımlarda
    büyük gradyanların ağırlıkları savurmasını engeller (LayerScale'li derin yığında
    kritik), sonda küçük adım = daha iyi minimum. Deterministik (rastgelelik yok)."""
    total = max(1, int(total_steps))
    warm = int(max(0.0, min(0.5, warmup_frac)) * total)
    if warm > 0 and step < warm:
        return base_lr * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    prog = min(1.0, max(0.0, prog))
    floor = base_lr * float(min_lr_frac)
    return floor + (base_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))


def _fit_temperature(logits: Any, targets: Any) -> float:
    """Sıcaklık ölçekleme (Guo ve ark. 2017): val diliminde NLL'i minimize eden tek skaler T.
    Yöntem DETERMİNİSTİK 1-B arama (kaba log-ızgara + altın-oran incelemesi) — LBFGS gibi
    bir optimizer yerine, çünkü sonucun ortam/sürüm bağımsız yeniden üretilebilir olması
    gerekir (anayasa madde 6). T>1 = model aşırı-güvenli (logitler yumuşatılır), T<1 = aşırı
    temkinli. Ölçüm başarısızsa 1.0 (nötr — kalibrasyon uygulanmamış gibi)."""
    def _nll(t: float) -> float:
        with torch.no_grad():
            return float(nn.functional.cross_entropy(logits / max(1e-3, t), targets).item())

    grid = [math.exp(_TEMP_LOG_LO + (_TEMP_LOG_HI - _TEMP_LOG_LO) * i / (_TEMP_GRID - 1))
            for i in range(_TEMP_GRID)]
    try:
        best_i = min(range(len(grid)), key=lambda i: _nll(grid[i]))
        lo = grid[max(0, best_i - 1)]
        hi = grid[min(len(grid) - 1, best_i + 1)]
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = _nll(c), _nll(d)
        for _ in range(_TEMP_REFINE_ITERS):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = _nll(c)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = _nll(d)
        return float(max(1e-3, (a + b) / 2.0))
    except (RuntimeError, ValueError):   # pragma: no cover - savunmacı
        return 1.0


class NeuralCapacity:
    """`nihai/ai/models.py`'nin `CapacityModel`'ine BİREBİR API paritesi sağlayan sinir-ağı
    sarmalayıcısı (bkz. modül docstring'i). torch yoksa kurucu RuntimeError fırlatır (modül
    yine de import edilebilir kalır — fail-soft). `neural.py` `models.py`'ı import ETMEZ;
    orkestratör models.py merdivenine bu sınıfı lazy-import ile bağlar.

    Kurucu: `NeuralCapacity(tier, feature_names, seed=42)`. `feature_names` SIRASI X'in
    sütun sırasıyla eşleşmelidir (çoklu-ölçek [SCALE] tip kimliği önekten türetilir)."""

    def __init__(self, tier: str, feature_names: list[str], seed: int = 42,
                 extra_heads: tuple[str, ...] | None = None) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch kurulu değil — neural katman devre dışı (fail-soft)")
        if tier not in NEURAL_TIERS:
            raise ValueError(f"bilinmeyen tier: {tier!r} (geçerli: {NEURAL_TIERS})")
        if extra_heads is not None:
            bad = [e for e in extra_heads if e not in EXTRA_HEADS]
            if bad:
                raise ValueError(f"bilinmeyen extra_head: {bad} (geçerli: {list(EXTRA_HEADS)})")
        self.tier = tier
        self.feature_names: list[str] = list(feature_names)
        self.seed = int(seed)
        # None = basamağın kendi varsayılanı (v7 ile bit-denk); tuple = açık seçim.
        self.extra_heads: tuple[str, ...] | None = (
            None if extra_heads is None else tuple(extra_heads))
        self.n_features_: int | None = None
        self.trained_at: str | None = None
        self._model: Any = None
        self._feat_mu: np.ndarray | None = None
        self._feat_sigma: np.ndarray | None = None
        self._fit_info: dict = {}
        self._device: Any = torch.device("cpu")
        self.temperature: float = 1.0        # fit-sonrası kalibrasyon (1.0 = uygulanmadı)
        self.train_zmax_q99: float | None = None   # eğitim dağılımının |z| üst kuyruğu (OOD eşiği)
        # Tur-35: `pretrain` de `_model`'i doldurur ama ETİKET GÖRMEMİŞ bir gövdedir.
        # `is_fitted` bu yüzden artık "model var mı" DEĞİL "etiketli eğitim tamamlandı mı"
        # sorusunu yanıtlar — ön-eğitilmiş gövdeden p_win okumak sessiz bir yalan olurdu.
        self._fitted: bool = False
        self._pretrain_info: dict = {}

    @property
    def is_fitted(self) -> bool:
        return self._model is not None and self._fitted

    @property
    def is_pretrained(self) -> bool:
        """Maskeli özellik ön-eğitiminden geçmiş bir GÖVDE tutuluyor mu (etiketli eğitim
        yapılmış olsun ya da olmasın)."""
        return bool(self._pretrain_info)

    # ------------------------------------------------------------------
    # Ön-eğitim (Tur-35) — ETİKETSİZ veriyle temsil öğrenme
    # ------------------------------------------------------------------
    def pretrain(self, X: ArrayLike, *, mask_ratio: float = _PRETRAIN_MASK_RATIO,
                 max_epochs: int = 30, patience: int = 5, lr: float = 1e-3,
                 weight_decay: float = 1e-4, val_frac: float = 0.15,
                 batch_size: int = 256, scheduler: str = "cosine",
                 warmup_frac: float = 0.05, min_lr_frac: float = 0.01,
                 device: str = "cpu") -> NeuralCapacity:
        """MASKELİ ÖZELLİK MODELLEME ile öz-denetimli ön-eğitim (VIME/SAINT deseninin
        tablosal karşılığı; Devlin ve ark. 2019'un maskeleme fikri, hedef sınıf değil
        özelliğin KENDİ normalize değeri).

        **NEDEN BU PROJEDE:** ACI DERS #5 gerçek darboğazın *"veri ve OOS-kenarlı baz sinyal
        yokluğu, model mimarisi değil"* olduğunu söylüyor ve *"yeni bir sınıflandırıcı
        eklemek bunu çözmez"* diye bitiyor. Bu yöntem yeni bir sınıflandırıcı DEĞİLDİR:
        ETİKET GEREKTİRMEZ. `data/live_features_log.csv` her aday sinyalde bir özellik
        vektörü biriktirir; bunların yalnız ÇÖZÜLMÜŞ olanları etiketlidir. Ön-eğitim
        etiketsiz çoğunluğu da kullanır — darboğazın veri tarafına dokunan tek koldur.

        **NASIL:** her satırda özelliklerin `mask_ratio` kadarı seçilir, o özelliklerin
        token'ı öğrenilebilir `[MASK]` gömmesiyle DEĞİŞTİRİLİR (sıfırlanmaz — sıfır geçerli
        bir normalize değerdir, ağ 'maskelendi'yi 'değeri 0' sanardı) ve ağ maskelenen
        özelliğin gerçek değerini geri kurmaya çalışır. Kayıp YALNIZ maskeli konumlardan
        alınır. Ağ böylece özellikler arası koşullu yapıyı (ör. "yüksek vol + düşük hacim
        genelde birlikte gelir") etiket olmadan öğrenir.

        **DÜRÜSTLÜK SÖZLEŞMESİ:**
          · Ön-eğitim `is_fitted`'i AÇMAZ. `predict_proba` yine RuntimeError verir — etiket
            görmemiş gövdeden p_win okumak sessiz bir yalan olurdu.
          · Ön-eğitim bir KENAR iddiası değildir (ACI DERS #5). "Rekonstrüksiyon kaybı
            düştü" ⇒ "yön tahmini iyileşir" ÇIKMAZ; ikisi ayrı ölçülür.
          · Rekonstrüksiyon başlığı ve `[MASK]` gömmesi ön-eğitim sonunda ATILIR → artefakt
            şeması sıfırdan eğitilmiş modelle AYNI kalır.
          · Val bölmesi yine KRONOLOJİK son %`val_frac`'tır (shuffle yok; look-ahead yasağı
            etiketsiz veride de geçerlidir — maske deseni geçmişten geleceğe sızmamalı).

        `deep_mlp` token'sızdır → ValueError (sessizce atlamak, çağıranın ön-eğitim
        yaptığını sanmasına yol açardı). Kullanım: `pretrain(X_unlabeled)` →
        `fit(X, y, warm_start=True)`."""
        if self.tier not in TOKENIZED_TIERS:
            raise ValueError(
                f"maskeli ön-eğitim yalnız token tabanlı basamaklarda anlamlıdır "
                f"(geçerli: {TOKENIZED_TIERS}) — {self.tier!r} düz MLP'dir, özellik token'ı yoktur")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2 or X_arr.shape[0] < 2:
            raise ValueError("pretrain için X (n>=2, f) 2-B olmalı")
        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X sütun sayısı ({X_arr.shape[1]}) feature_names uzunluğuyla "
                f"({len(self.feature_names)}) uyuşmuyor")
        if not 0.0 < float(mask_ratio) < 1.0:
            raise ValueError(f"mask_ratio (0,1) aralığında olmalı — {mask_ratio!r}")
        if scheduler not in ("none", "cosine"):
            raise ValueError(f"scheduler 'none'|'cosine' olmalı — {scheduler!r}")

        t_start = time.perf_counter()
        n, f = X_arr.shape
        dev = _resolve_device(device)
        model = _build_torch_model(self.tier, self.feature_names, self.seed,
                                   self.extra_heads).to(dev)
        head = _MaskedFeatureHead(int(model.d_model)).to(dev)
        mask_token = nn.Parameter(torch.zeros(1, 1, int(model.d_model), device=dev))
        torch.manual_seed(self.seed)
        nn.init.normal_(mask_token, std=0.02)

        mu = X_arr.mean(axis=0)
        sigma = X_arr.std(axis=0)
        sigma[sigma == 0] = 1.0
        Xn = (X_arr - mu) / sigma

        n_val = int(round(n * val_frac))
        if n_val < 1 or (n - n_val) < 2:
            n_val = 0
        has_val = n_val > 0
        n_tr = n - n_val
        Xtr_t = torch.tensor(Xn[:n_tr], dtype=torch.float32, device=dev)
        Xval_t = (torch.tensor(Xn[n_tr:], dtype=torch.float32, device=dev)
                  if has_val else None)

        # Her satırda TAM `k` özellik maskelenir (Bernoulli değil): oran satır başına sabit
        # kalır, yoksa bazı satırlar hiç maskelenmez ve kayıp tanımsız olur.
        k = max(1, min(f - 1, int(round(float(mask_ratio) * f)))) if f > 1 else 1
        gen = torch.Generator().manual_seed(self.seed)

        def _sample_mask(rows: int) -> torch.Tensor:
            """Satır başına tam k rastgele konum (deterministik: `gen` seed'li)."""
            scores = torch.rand(rows, f, generator=gen)
            idx = torch.topk(scores, k, dim=1).indices
            m = torch.zeros(rows, f, dtype=torch.bool)
            m.scatter_(1, idx, True)
            return m.to(dev)

        val_mask = _sample_mask(n_val) if has_val else None   # SABİT → val ölçümü kıyaslanabilir

        params = list(model.parameters()) + list(head.parameters()) + [mask_token]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        bs = max(1, int(batch_size))
        steps_per_epoch = max(1, math.ceil(n_tr / bs))
        total_steps = steps_per_epoch * max(1, int(max_epochs))

        def _recon_loss(xb: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            pred = head(model.encode_masked(xb, m, mask_token))
            mf = m.to(pred.dtype)
            return ((pred - xb) ** 2 * mf).sum() / mf.sum().clamp_min(1.0)

        best = float("inf")
        best_state: dict | None = None
        patience_left = int(patience)
        epochs_run = 0
        global_step = 0
        first_val: float | None = None
        for epoch in range(int(max_epochs)):
            model.train()
            head.train()
            perm = torch.randperm(n_tr, generator=gen)
            for start in range(0, n_tr, bs):
                idx = perm[start:start + bs].to(dev)
                if scheduler == "cosine":
                    lr_t = _cosine_lr(global_step, total_steps, lr, warmup_frac, min_lr_frac)
                    for group in optimizer.param_groups:
                        group["lr"] = lr_t
                optimizer.zero_grad()
                xb = Xtr_t[idx]
                loss = _recon_loss(xb, _sample_mask(int(idx.numel())))
                loss.backward()
                nn.utils.clip_grad_norm_(params, _GRAD_CLIP_NORM)
                optimizer.step()
                global_step += 1
            epochs_run = epoch + 1
            if has_val and Xval_t is not None and val_mask is not None:
                model.eval()
                head.eval()
                with torch.no_grad():
                    v = float(_recon_loss(Xval_t, val_mask).item())
                if first_val is None:
                    first_val = v
                if v < best - 1e-9:
                    best = v
                    best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
                    patience_left = int(patience)
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        self._model = model
        self._fitted = False           # gövde hazır, ETİKET görmedi (bkz. sözleşme)
        self._device = dev
        self._feat_mu = mu
        self._feat_sigma = sigma
        self.n_features_ = int(f)
        self._pretrain_info = {
            "n_rows": int(n),
            "mask_ratio": round(float(k) / float(f), 6),
            "masked_per_row": int(k),
            "n_epochs": epochs_run,
            "early_stopped": bool(has_val and patience_left <= 0),
            "has_val_split": has_val,
            "val_recon_mse_first": (round(first_val, 6) if first_val is not None else None),
            "val_recon_mse_best": (round(float(best), 6) if best_state is not None else None),
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "scheduler": scheduler,
            "device": str(dev),
            "seconds": round(time.perf_counter() - t_start, 3),
        }
        return self

    # ------------------------------------------------------------------
    # Eğitim
    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike, y: ArrayLike, aux: dict[str, np.ndarray] | None = None, *,
            max_epochs: int = 100, patience: int = 10, lr: float = 1e-3,
            weight_decay: float = 1e-4, val_frac: float = 0.15,
            batch_size: int = 256, scheduler: str = "none", warmup_frac: float = 0.05,
            min_lr_frac: float = 0.01, ema_decay: float | None = None,
            class_weight: str | None = None, label_smoothing: float = 0.0,
            early_stop_metric: str = "val_loss", calibrate: bool = False,
            device: str = "cpu", warm_start: bool = False,
            optimizer_name: str = "adamw", amp: bool = False, progress=None) -> NeuralCapacity:
        """X:(n,f) float, y:(n,) {0,1}. `aux` opsiyonel: {'magnitude':(n,) int[0..4],
        'vol':(n,) float, 'regime':(n,) int[0..5]} — yalnız VERİLEN anahtarlar kayba girer.

        Val bölmesi KRONOLOJİK SON %`val_frac` dilimidir (shuffle YOK — çağıran X/y'yi
        zaman sırasıyla vermelidir, look-ahead yasağı). n çok küçükse (val'e yer kalmazsa)
        val bölünmez, erken-durma devre dışı kalır, tüm veri eğitimde kullanılır (çökme yok).

        v4 opt-in argümanları (varsayılanları v3 ile BİT-DENK — bkz. modül docstring'i):
          scheduler          "none" | "cosine" (ısınmalı, adım-başı)
          ema_decay          None | 0<d<1 — ağırlık EMA'sı; val ölçümü ve nihai ağırlıklar
                             EMA kopyasından alınır (Polyak ortalamasının pratik biçimi)
          class_weight       None | "balanced" — YALNIZ eğitim diliminin sınıf oranından
          label_smoothing    0.0..0.3 — ana görev CE'sinde
          early_stop_metric  "val_loss" | "val_auc" (AUC'de büyük daha iyi)
          calibrate          True → fit sonrası val diliminde sıcaklık ölçekleme
          device             "cpu" | "cuda" | "auto"

        v7 (Tur-35): `warm_start=True` → elde bir gövde VARSA (bkz. `pretrain`) sıfırdan
        ilklendirme YAPILMAZ, o ağırlıklardan devam edilir. Gövde yoksa sessizce sıfırdan
        kurar (hata değil — "ön-eğitim opsiyoneldir" sözleşmesi). Varsayılan False →
        v6 ile BİT-DENK. UYARI: normalizasyon istatistikleri (mu/sigma) HER ZAMAN bu
        çağrının X'inden yeniden hesaplanır; ön-eğitim verisi farklı bir dağılımdan
        geliyorsa gövde biraz kaymış bir girdi görür — bu ÇAĞIRANIN sorumluluğudur ve
        `recipe.warm_start` ile artefakta yazılır.

        v9 (Tur-42) BELLEK REÇETESİ — ikisi de opt-in, varsayılanları v8 ile BİT-DENK:
          optimizer_name     "adamw" (varsayılan) | "adafactor". Adafactor optimizatör
                             DURUMUNU O(n)→O(√n) indirir (ölçüldü: 8,00 → 0,0156
                             bayt/parametre) → ULTRA/MEGA 8 GB'lık kartta eğitilebilir olur.
                             FARKLI bir algoritmadır: ağırlıklar AdamW ile bit-denk DEĞİL,
                             bu yüzden seçim `recipe.optimizer`e yazılır. Ad boşluk/büyük
                             harfe dayanıklıdır; tanınmayan ad SESSİZCE düşmez, ValueError.
          amp                True → ileri geçiş bf16 autocast. YALNIZ CUDA'da etkindir;
                             CPU'da sessizce yok sayılır ve `recipe.amp` FALSE yazılır
                             ("istendi" ile "oldu" aynı alana yazılamaz). GradScaler YOK —
                             bf16'nın dinamik aralığı float32 ile aynıdır.

        Uyumsuz/yetersiz girdi → ValueError (programcı hatası — CapacityModel deseniyle
        aynı: savunmacı-degrade DEĞİL, API sözleşmesi ihlali)."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if X_arr.ndim != 2 or X_arr.shape[0] < 2 or X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X/y boyutları uyumsuz veya yetersiz örnek (fit için >=2 satır gerekir)")
        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X sütun sayısı ({X_arr.shape[1]}) feature_names uzunluğuyla "
                f"({len(self.feature_names)}) uyuşmuyor")
        if early_stop_metric not in ("val_loss", "val_auc"):
            raise ValueError(f"early_stop_metric 'val_loss'|'val_auc' olmalı — {early_stop_metric!r}")
        if scheduler not in ("none", "cosine"):
            raise ValueError(f"scheduler 'none'|'cosine' olmalı — {scheduler!r}")
        if class_weight not in (None, "balanced"):
            raise ValueError(f"class_weight None|'balanced' olmalı — {class_weight!r}")
        opt_name = (optimizer_name or "adamw").strip().lower()
        if opt_name not in ("adamw", "adafactor"):
            raise ValueError(f"bilinmeyen optimizer: {optimizer_name!r} (adamw|adafactor)")
        n = X_arr.shape[0]
        t_start = time.perf_counter()

        aux_arrs: dict[str, np.ndarray] = {}
        for key in ("magnitude", "vol", "regime"):
            v = (aux or {}).get(key)
            if v is not None:
                a = np.asarray(v)
                if a.shape[0] != n:
                    raise ValueError(f"aux[{key!r}] uzunluğu X ile uyuşmuyor")
                aux_arrs[key] = a

        dev = _resolve_device(device)
        warm_used = bool(warm_start and self._model is not None
                         and self.n_features_ == X_arr.shape[1])
        model = (self._model.to(dev) if warm_used
                 else _build_torch_model(self.tier, self.feature_names, self.seed,
                                         self.extra_heads).to(dev))
        q_taus = getattr(model, "vol_quantiles", None)

        mu = X_arr.mean(axis=0)
        sigma = X_arr.std(axis=0)
        sigma[sigma == 0] = 1.0
        Xn = (X_arr - mu) / sigma

        n_val = int(round(n * val_frac))
        if n_val < 1 or (n - n_val) < 2:
            n_val = 0
        has_val = n_val > 0
        n_tr = n - n_val

        def _split(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
            if has_val:
                return arr[:n_tr], arr[n_tr:]
            return arr, None

        Xtr_n, Xval_n = _split(Xn)
        ytr, yval = _split(y_arr)
        aux_tr: dict[str, torch.Tensor] = {}
        aux_val: dict[str, torch.Tensor] = {}
        def _to_tensor(arr: np.ndarray, dt: Any) -> torch.Tensor:
            data = arr if dt == torch.float32 else np.round(arr).astype(np.int64)
            return torch.tensor(data, dtype=dt, device=dev)

        for k, a in aux_arrs.items():
            tr_a, val_a = _split(a)
            dtype = torch.float32 if k == "vol" else torch.long
            aux_tr[k] = _to_tensor(tr_a, dtype)
            if val_a is not None:
                aux_val[k] = _to_tensor(val_a, dtype)

        Xtr_t = torch.tensor(Xtr_n, dtype=torch.float32, device=dev)
        ytr_t = torch.tensor(np.round(ytr).astype(np.int64), dtype=torch.long, device=dev)
        if has_val:
            assert Xval_n is not None and yval is not None
            Xval_t = torch.tensor(Xval_n, dtype=torch.float32, device=dev)
            yval_t = torch.tensor(np.round(yval).astype(np.int64), dtype=torch.long, device=dev)

        # Sınıf ağırlıkları YALNIZ eğitim diliminden (val'e/teste bakılmaz → sızıntı yok).
        cls_w: Any = None
        cls_w_list: list[float] | None = None
        if class_weight == "balanced":
            pos = float(np.mean(np.round(ytr)))
            if 0.0 < pos < 1.0:
                w0, w1 = 1.0 / (2.0 * (1.0 - pos)), 1.0 / (2.0 * pos)
                cls_w_list = [round(w0, 6), round(w1, 6)]
                cls_w = torch.tensor([w0, w1], dtype=torch.float32, device=dev)

        # BELLEK REÇETESİ (opt-in; ikisi de varsayılan-kapalı → v8 ile BİT-DENK).
        #
        # `optimizer_name="adafactor"` (Shazeer ve Stern 2018): ikinci momenti SATIR×SÜTUN
        # çarpanlarına ayırır ve birinci momenti hiç tutmaz → durum O(n) yerine O(√n).
        # ÖLÇÜLDÜ (ULTRA, 771.874.072 parametre, float32): AdamW durumu ≈6,17 GB, Adafactor
        # ≈0,01 GB. 8 GB'lık bir kartta ULTRA'yı AdamW ile eğitmek İMKÂNSIZDIR (ölçüldü:
        # 16,94 GiB ayırıp OOM); Adafactor ile mümkün hâle gelir.
        # BEDELİ SAKLI DEĞİL: bu FARKLI bir optimizasyon algoritmasıdır, aynı reçetenin
        # ucuz hâli değil. Ağırlıklar AdamW koşusuyla bit-denk OLMAZ; bu yüzden seçim
        # `meta()["recipe"]["optimizer"]`e YAZILIR — bir artefakt hangi reçeteyle
        # eğitildiğini taşımadan karşılaştırmaya sokulamaz.
        #
        # `amp=True`: ileri geçiş bf16 autocast (yalnız CUDA'da; ağırlıklar float32 kalır).
        # Aktivasyon belleğini ≈yarıya indirir. GradScaler YOK: bf16'nın dinamik aralığı
        # float32 ile aynıdır (fp16 değil) → ölçekleme gereksizdir.
        optimizer: torch.optim.Optimizer
        if opt_name == "adafactor":
            optimizer = torch.optim.Adafactor(model.parameters(), lr=lr,
                                              weight_decay=weight_decay)
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        else:   # ULAŞILAMAZ (ad yukarıda doğrulandı) — listeye ad eklenip gönderimin
            # güncellenmemesine karşı kasıtlı guard: sessizce AdamW'ye düşmektense patla.
            raise ValueError(f"bilinmeyen optimizer: {optimizer_name!r} (adamw|adafactor)")
        use_amp = bool(amp) and dev.type == "cuda"
        # Bağlam NESNESİ döngü DIŞINDA kurulur (Tur-42). Devre DIŞI bir `torch.autocast`
        # bile toplu-iş başına 7,04 µs harcar (ölçüldü, 20.000 yineleme); yeniden kullanılan
        # `nullcontext` 0,30 µs — 23×. AMP KAPALI olan VARSAYILAN yol bu vergiyi ödememeli.
        # Ayrıca CUDA'sız kurulumda `autocast("cuda")` KURMAK bile uyarı bastırır; burada
        # hiç kurulmaz. Aynı nesnenin ardışık yeniden kullanımı güvenlidir (iç içe DEĞİL).
        amp_ctx: Any = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if use_amp else contextlib.nullcontext())
        gen = torch.Generator().manual_seed(self.seed)
        bs = max(1, int(batch_size))
        steps_per_epoch = max(1, math.ceil(n_tr / bs))
        total_steps = steps_per_epoch * max(1, int(max_epochs))

        use_ema = ema_decay is not None and 0.0 < float(ema_decay) < 1.0
        ema_val: float = float(ema_decay) if use_ema and ema_decay is not None else 0.0
        ema_state: dict[str, torch.Tensor] | None = None
        if use_ema:
            ema_state = {k: v.detach().clone().float()
                         for k, v in model.state_dict().items() if v.is_floating_point()}

        def _eval_state() -> dict[str, torch.Tensor] | None:
            """Değerlendirmede kullanılacak ağırlıklar: EMA varsa onun kopyası, yoksa None
            (canlı ağırlıklar). Geri-yükleme çağıranın sorumluluğundadır."""
            if not use_ema or ema_state is None:
                return None
            live = {k: v.detach().clone() for k, v in model.state_dict().items()
                    if k in ema_state}
            model.load_state_dict({**model.state_dict(), **ema_state}, strict=True)
            return live

        best_score = float("inf")   # her zaman "küçük daha iyi" ölçeğinde tutulur
        best_state: dict | None = None
        patience_left = int(patience)
        epochs_run = 0
        global_step = 0
        for epoch in range(int(max_epochs)):
            model.train()
            # EDL KL katsayısı epokla 0→1 tavlanır (Sensoy §4): kanıt daha oluşmadan
            # cezalandırılırsa model hiç kanıt üretmemeyi öğrenir. `dir_evid` başlığı
            # yoksa değer hiç kullanılmaz (kayıp bit-denk).
            evid_anneal = min(1.0, (epoch + 1) / _EVID_ANNEAL_EPOCHS)
            perm = torch.randperm(n_tr, generator=gen)
            for start in range(0, n_tr, bs):
                idx = perm[start:start + bs].to(dev)
                if scheduler == "cosine":
                    lr_t = _cosine_lr(global_step, total_steps, lr, warmup_frac, min_lr_frac)
                    for group in optimizer.param_groups:
                        group["lr"] = lr_t
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx:
                    out = model(Xtr_t[idx])
                    aux_batch = {k: v[idx] for k, v in aux_tr.items()}
                    loss = _compute_loss(out, ytr_t[idx], aux_batch, weight=cls_w,
                                         label_smoothing=label_smoothing, quantiles=q_taus,
                                         evid_anneal=evid_anneal)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
                optimizer.step()
                global_step += 1
                if use_ema and ema_state is not None:
                    with torch.no_grad():
                        sd = model.state_dict()
                        for k in ema_state:
                            ema_state[k].mul_(ema_val).add_(sd[k].detach().float(),
                                                            alpha=1.0 - ema_val)
            epochs_run = epoch + 1
            val_loss = None
            stop_training = False
            if has_val:
                live = _eval_state()
                model.eval()
                with torch.no_grad():
                    # Tur-94 (BULGU-94-2): val kümesi `_INFER_BATCH` parçalarıyla geçer. Tek geçiş
                    # 282M basamağında 9,6 bin satırda VRAM'i sistem RAM'ine taşırdı (paylaşımlı
                    # 7,9 GB, süreç 18 GB). Val ≤ parça → tek geçiş, eskisiyle BİT-DENK; çok parçada
                    # kayıp parça kayıplarının satır-ağırlıklı ortalamasıdır (MoE yük terimi ve
                    # sınıf ağırlığı parça başına — yaklaşık).
                    parcalar, yon = [], []
                    for s in range(0, int(yval_t.shape[0]), _INFER_BATCH):
                        sl = slice(s, s + _INFER_BATCH)
                        out_val = model(Xval_t[sl])
                        lp = float(_compute_loss(out_val, yval_t[sl], {k: v[sl] for k, v in aux_val.items()},
                                                 weight=cls_w, label_smoothing=label_smoothing,
                                                 quantiles=q_taus).item())
                        parcalar.append((lp, int(out_val["direction"].shape[0])))
                        yon.append(out_val["direction"])
                    val_loss = (parcalar[0][0] if len(parcalar) == 1 else
                                sum(lp * k for lp, k in parcalar) / sum(k for _, k in parcalar))
                    if early_stop_metric == "val_auc":
                        p_val = torch.softmax(torch.cat(yon), dim=-1)[:, 1].cpu().numpy()
                        auc = _auc_np(np.asarray(yval, dtype=float), p_val)
                        # AUC "büyük daha iyi" → ortak ölçeğe çevir; ölçülemezse kayba düş.
                        score = (1.0 - float(auc)) if auc is not None else val_loss
                    else:
                        score = val_loss
                    # En iyi durum kopyası CPU'da (282M'de GPU'da 1,13 GB daha tutardı).
                    snapshot = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
                if live is not None:
                    model.load_state_dict({**model.state_dict(), **live}, strict=True)
                if score < best_score - 1e-6:
                    best_score = score
                    best_state = snapshot
                    patience_left = int(patience)
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        stop_training = True
            if progress is not None:
                # Ölçüm son minibatch'e aittir; epoch ortalaması gibi sunulmaz.
                progress({"epoch": epochs_run, "max_epochs": int(max_epochs),
                          "train_loss_last_batch": float(loss.detach().item()), "val_loss": val_loss})
            if stop_training:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        elif use_ema and ema_state is not None:
            # val yok → erken-durma da yok; EMA yine de gürültüyü azaltır (son adım yerine
            # ortalama ağırlık). v3 davranışı ema_decay=None ile bit-denk korunur.
            model.load_state_dict({**model.state_dict(), **ema_state}, strict=True)
        model.eval()

        self._model = model
        self._fitted = True
        self._device = dev
        self._feat_mu = mu
        self._feat_sigma = sigma
        self.n_features_ = int(X_arr.shape[1])
        self.trained_at = _utc_now_iso()
        self.temperature = 1.0

        # Eğitim dağılımının |z| üst kuyruğu — canlı OOD guard'ı için VERİ-TÜRETİLMİŞ eşik
        # (sabit 6.0 yerine). σ<1e-9 kolonlar dışlanır (orada z tanımsız — neural_shadow ile
        # aynı kural). Model kartı/artefaktla birlikte taşınır.
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = np.abs(Xn[:, sigma > 1e-9]) if np.any(sigma > 1e-9) else np.zeros((n, 1))
        zz = zz[np.isfinite(zz)]
        self.train_zmax_q99 = float(np.quantile(zz, 0.99)) if zz.size else None

        val_metrics: dict = {}
        if has_val:
            p_val = self.predict_proba(X_arr[n_tr:])
            if calibrate:
                with torch.no_grad():
                    logits_val = torch.cat([model(Xval_t[s:s + _INFER_BATCH])["direction"].detach()
                                            for s in range(0, int(Xval_t.shape[0]), _INFER_BATCH)])
                self.temperature = _fit_temperature(logits_val, yval_t)
                p_val = self.predict_proba(X_arr[n_tr:])
            val_metrics = evaluation_metrics(y_arr[n_tr:], p_val)

        self._fit_info = {
            "n_epochs_trained": epochs_run,
            "early_stopped": bool(has_val and patience_left <= 0),
            "val_loss": (round(best_score, 6) if (has_val and best_state is not None
                                                  and early_stop_metric == "val_loss") else None),
            "aux_heads_trained": sorted(aux_arrs.keys()),
            "has_val_split": has_val,
            # --- v4 ek gözlemler (v3 tüketicileri fazla anahtarı yok sayar) ---
            "recipe": {"scheduler": scheduler, "ema_decay": (ema_val if use_ema else None),
                       "class_weight": class_weight, "class_weights": cls_w_list,
                       "label_smoothing": float(label_smoothing),
                       "early_stop_metric": early_stop_metric, "calibrated": bool(calibrate),
                       "device": str(dev), "optimizer": opt_name, "amp": bool(use_amp),
                       "batch_size": bs, "lr": float(lr),
                       "weight_decay": float(weight_decay),
                       "warm_start": bool(warm_start), "warm_start_used": warm_used,
                       "pretrained": bool(self._pretrain_info)},
            "pretrain": (dict(self._pretrain_info) if self._pretrain_info else None),
            "best_score": (round(float(best_score), 6) if best_state is not None else None),
            "temperature": round(float(self.temperature), 4),
            "train_zmax_q99": (round(float(self.train_zmax_q99), 4)
                               if self.train_zmax_q99 is not None else None),
            "train_pos_rate": round(float(np.mean(np.round(y_arr))), 6),
            "val_metrics": val_metrics or None,
            "train_seconds": round(time.perf_counter() - t_start, 3),
        }
        return self

    # ------------------------------------------------------------------
    # Tahmin
    # ------------------------------------------------------------------
    def _normalize(self, X: ArrayLike) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2 or X_arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X şekli beklenenle uyuşmuyor (n,{len(self.feature_names)}) — alınan {X_arr.shape}")
        assert self._feat_mu is not None and self._feat_sigma is not None
        return (X_arr - self._feat_mu) / self._feat_sigma

    def _direction_logits(self, Xn: np.ndarray) -> Any:
        """Normalize edilmiş X → yön logitleri (toplu-iş bölerek; bkz. `_INFER_BATCH`)."""
        chunks = []
        with torch.no_grad():
            for s in range(0, Xn.shape[0], _INFER_BATCH):
                xt = torch.tensor(Xn[s:s + _INFER_BATCH], dtype=torch.float32, device=self._device)
                chunks.append(self._model(xt)["direction"].detach().cpu())
        return torch.cat(chunks, dim=0) if chunks else torch.zeros((0, 2))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """p(y=1), şekil (n,). fit edilmemişse RuntimeError. `temperature` 1.0'dan farklıysa
        (fit(calibrate=True)) logitler ona bölünür — sıcaklık ölçekleme sıralamayı DEĞİŞTİRMEZ
        (AUC aynı kalır), yalnız kalibrasyonu düzeltir."""
        if not self.is_fitted:
            raise RuntimeError("NeuralCapacity henüz fit edilmedi")
        Xn = self._normalize(X)
        with INFER_LOCK:
            self._model.eval()
            logits = self._direction_logits(Xn)
        t = float(self.temperature) if self.temperature else 1.0
        probs = torch.softmax(logits / max(1e-3, t), dim=-1)[:, 1]
        return probs.numpy().astype(float)

    def predict_vol_quantiles(self, X: ArrayLike) -> np.ndarray | None:
        """DAĞILIMSAL oynaklık tahmini (Tur-35): (n, Q) artan quantile'lar — `VOL_QUANTILES`
        sırasıyla. Bu başlığı OLMAYAN basamaklarda dürüstçe **None** döner ("oynaklık
        belirsizliği yok" DEĞİL, "bu mimaride ölçülmüyor" — Tur-34'ün ölçüm-koşulu kuralı).

        Ölçek uyarısı: değerler `aux['vol']` hedefinin HAM ölçeğindedir (fit'te vol hedefi
        normalize EDİLMEZ); çağıran hangi büyüklüğü hedef verdiyse onu geri okur.
        fit edilmemişse RuntimeError."""
        if not self.is_fitted:
            raise RuntimeError("NeuralCapacity henüz fit edilmedi")
        if getattr(self._model, "vol_quantile_head", None) is None:
            return None
        Xn = self._normalize(X)
        chunks = []
        with INFER_LOCK, torch.no_grad():
            self._model.eval()
            for s in range(0, Xn.shape[0], _INFER_BATCH):
                xt = torch.tensor(Xn[s:s + _INFER_BATCH], dtype=torch.float32, device=self._device)
                chunks.append(self._model(xt)["vol_q"].detach().cpu())
        if not chunks:
            return np.zeros((0, len(getattr(self._model, "vol_quantiles", VOL_QUANTILES))))
        return torch.cat(chunks, dim=0).numpy().astype(float)

    def predict_with_uncertainty(self, X: ArrayLike, T: int = 20, *,
                                 batched: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """MC-dropout (T geçiş, dropout AKTİF tutulur): (ortalama_p, std_p), ikisi de (n,)
        şeklinde. `torch.manual_seed(seed)` ile örnekleme deterministik (anayasa madde 6).
        XL katmanında stokastik derinlik de aktif kalır → belirsizlik hem birim hem KATMAN
        düzeyinde örneklenir. `batched` → bkz. `_mc_samples`. fit edilmemişse RuntimeError."""
        if not self.is_fitted:
            raise RuntimeError("NeuralCapacity henüz fit edilmedi")
        arr = self._mc_samples(X, T, batched=batched)
        return arr.mean(axis=0).astype(float), arr.std(axis=0).astype(float)

    def _mc_samples(self, X: ArrayLike, T: int, *, batched: bool = False) -> np.ndarray:
        """MC-dropout örneklem matrisi (T, n) — `predict_with_uncertainty` ile ayrışmanın
        ORTAK kaynağı (aynı tohum, aynı işlem sırası → iki yol aynı örnekleri görür).

        Tur-93 `batched=True`: T kopyayı TEK ileri geçişte örnekler. Dropout ve stokastik
        derinlik maskesi örnek başınadır, katmanlarda toplu-iş istatistiği ve uzman kapasite
        sınırı yoktur → her kopya bağımsız bir MC örneğidir; RNG akışı farklı olduğundan
        döngüyle BİT-DENK DEĞİLDİR (T=1'de aynıdır). Ölçüldü (282M MoE-XL, tek aday, T=20):
        GPU döngü 1.825 ms → toplu 102 ms; CPU tek thread 2.924 → 1.040 ms."""
        Xn = self._normalize(X)
        n, reps = Xn.shape[0], max(1, int(T))
        Xb = np.tile(Xn, (reps, 1)) if batched else Xn
        t = float(self.temperature) if self.temperature else 1.0
        samples = []
        with INFER_LOCK:   # tohum + train/eval kipi süreç/model geneli (BULGU-80-3)
            torch.manual_seed(self.seed)
            self._model.train()   # dropout AKTİF kalsın — MC-dropout'un kalbi
            try:
                with torch.no_grad():
                    for _ in range(1 if batched else reps):
                        chunks = []
                        for s in range(0, Xb.shape[0], _INFER_BATCH):
                            xt = torch.tensor(Xb[s:s + _INFER_BATCH], dtype=torch.float32,
                                              device=self._device)
                            logits = self._model(xt)["direction"].detach().cpu()
                            chunks.append(torch.softmax(logits / max(1e-3, t), dim=-1)[:, 1])
                        samples.append(torch.cat(chunks, dim=0).numpy() if chunks else np.zeros(0))
            finally:
                self._model.eval()   # istisnada da paylaşılan model train() kipinde kalmasın
        return samples[0].reshape(reps, n) if batched else np.stack(samples, axis=0)

    def to_device(self, device: str) -> Any:
        """Tur-93: yüklü modeli çıkarım için taşı ('cpu'|'cuda'|'auto'; kurulamayan aygıt
        sessizce CPU — `_resolve_device`). Çözülen torch.device döner."""
        if not self.is_fitted:
            raise RuntimeError("NeuralCapacity henüz fit edilmedi")
        dev = _resolve_device(device)
        self._model.to(dev)
        self._device = dev
        return dev

    def predict_uncertainty_decomposition(self, X: ArrayLike, T: int = 20) -> dict[str, Any] | None:
        """Tur-77 · Madde 5.1: `decompose_uncertainty` MC-dropout örneklemi üzerinde. Tek
        modelde epistemik bileşen yalnız dropout'un ağırlık belirsizliğidir (topluluk ayrıca
        üyeler arası ayrışmayı ekler). `p` ve `sqrt(epistemik_var)`, `predict_with_uncertainty`
        ile AYNI örneklerden gelir; fark yalnız float32 örneklemin burada float64'te
        ortalanmasındandır (ölçüldü ~1e-8, test ≤1e-6). fit edilmemişse RuntimeError."""
        if not self.is_fitted:
            raise RuntimeError("NeuralCapacity henüz fit edilmedi")
        return decompose_uncertainty(self._mc_samples(X, T))

    def predict_evidential(self, X: ArrayLike) -> tuple[np.ndarray, np.ndarray] | None:
        """KANIT-temelli yön tahmini (Tur-36): `(p, u)` — ikisi de (n,).

        `p` = α₁/S (Dirichlet ortalaması), `u` = K/S ∈ (0,1] = Dempster-Shafer BOŞLUĞU
        (S = Σα, K = 2). u→1 "hiç kanıt yok", u→0 "bol kanıt".

        MC-dropout'tan FARKI ve BENZERİ:
          · `predict_with_uncertainty` T ileri geçiş harcar ve std'si p'nin ÖRNEKLEME
            saçılımıdır; burada TEK geçiş yeter ve u doğrudan KANIT MİKTARIdır.
          · İkisi de EPİSTEMİK belirsizliği hedefler ama AYNI SAYI DEĞİLDİR: u p'den
            bağımsız olabilir (p=0,5 + bol kanıt = "gerçekten kararsız", p=0,5 + kanıt
            yok = "bilmiyorum"). MC-dropout bu ikisini ayırmakta zorlanır.

        Bu başlığı OLMAYAN modelde dürüstçe **None** ("belirsizlik yok" DEĞİL, "bu
        mimaride ölçülmüyor"). fit edilmemişse RuntimeError.

        Sıcaklık UYGULANMAZ: `temperature` `direction` logitlerinin kalibrasyonudur;
        Dirichlet kanıtı ayrı bir ölçektir (bkz. `_decode`)."""
        if not self.is_fitted:
            raise RuntimeError("NeuralCapacity henüz fit edilmedi")
        alpha = self._dir_alpha(X)
        if alpha is None:
            return None
        if alpha.shape[0] == 0:
            return np.zeros(0), np.zeros(0)
        strength = alpha.sum(axis=1)
        return (alpha[:, 1] / strength), (float(alpha.shape[1]) / strength)

    def _dir_alpha(self, X: ArrayLike) -> np.ndarray | None:
        """Dirichlet konsantrasyonları (n, K); `dir_evid` başlığı yoksa None."""
        if getattr(self._model, "dir_evidence_head", None) is None:
            return None
        Xn = self._normalize(X)
        chunks = []
        with INFER_LOCK, torch.no_grad():
            self._model.eval()
            for s in range(0, Xn.shape[0], _INFER_BATCH):
                xt = torch.tensor(Xn[s:s + _INFER_BATCH], dtype=torch.float32, device=self._device)
                chunks.append(self._model(xt)["dir_evid"].detach().cpu())
        if not chunks:
            return np.zeros((0, 2))
        return torch.cat(chunks, dim=0).numpy().astype(float)

    def predict_belief(self, X: ArrayLike) -> dict[str, Any] | None:
        """Tur-77 · Madde 5.2: kanıt başlığının İNANÇ KÜTLELERİ (`evidential_belief`):
        `inanc` (n, K) · `bosluk` u · `p` · `guc` S. Başlık yoksa dürüstçe None ("belirsizlik
        yok" DEĞİL, "bu mimaride ölçülmüyor"). fit edilmemişse RuntimeError."""
        if not self.is_fitted:
            raise RuntimeError("NeuralCapacity henüz fit edilmedi")
        alpha = self._dir_alpha(X)
        return None if alpha is None or alpha.shape[0] == 0 else evidential_belief(alpha)

    # ------------------------------------------------------------------
    # Dağılım-dışı (OOD) ölçüsü
    # ------------------------------------------------------------------
    def ood_zmax(self, X: ArrayLike) -> np.ndarray:
        """Satır-başına max|z| (modelin KENDİ eğitim istatistiklerine göre). σ<1e-9 olan
        (eğitimde fiilen sabit) kolonlar DIŞLANIR — orada z tanımsızdır, 1e-15'lik bölen
        sahte 10¹⁵ mertebesinde değer üretirdi. `train_zmax_q99` ile karşılaştırılınca
        "bu girdi eğitim dağılımının neresinde?" sorusunun sayısal cevabıdır; canlı gölge
        köprüsü (nihai/execution/neural_shadow.py) susma kararını buna dayandırabilir.

        fit edilmemişse RuntimeError; şekil uyuşmazlığı ValueError (API sözleşmesi)."""
        if not self.is_fitted:
            raise RuntimeError("NeuralCapacity henüz fit edilmedi")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2 or X_arr.shape[1] != len(self.feature_names):
            raise ValueError(f"X şekli beklenenle uyuşmuyor — alınan {X_arr.shape}")
        assert self._feat_mu is not None and self._feat_sigma is not None
        live = self._feat_sigma > 1e-9
        if not bool(live.any()):
            return np.zeros(X_arr.shape[0], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.abs((X_arr[:, live] - self._feat_mu[live]) / self._feat_sigma[live])
        z = np.where(np.isfinite(z), z, 0.0)
        return z.max(axis=1).astype(float)

    def evaluate(self, X: ArrayLike, y: ArrayLike) -> dict:
        """{'n','auc','brier','ece','accuracy','pos_rate','pred_mean'} — dürüstlük seti
        (bkz. `evaluation_metrics`). KENAR İDDİASI ÜRETMEZ; yalnız ölçer."""
        return evaluation_metrics(y, self.predict_proba(X))

    # ------------------------------------------------------------------
    # Meta / kalıcılık
    # ------------------------------------------------------------------
    def effective_extra_heads(self) -> list[str] | None:
        """Modelde GERÇEKTEN kurulu ek başlıklar (`EXTRA_HEADS` sırasında).

        Model henüz kurulmamışsa (fit/pretrain öncesi) ve çağıran açık bir seçim
        vermediyse **None** döner: "hiç yok" DEĞİL, *"basamağın varsayılanı, henüz
        ölçülmedi"* (Tur-34'ün ölçüm-koşulu kuralı). Kurulmuşsa liste ÖLÇÜMDÜR —
        modüllerin varlığından okunur, çağıranın niyetinden değil."""
        model = self._model
        if model is None:
            return None if self.extra_heads is None else list(self.extra_heads)
        attrs = {"vol_q": "vol_quantile_head", "dir_evid": "dir_evidence_head"}
        return [n for n in EXTRA_HEADS if getattr(model, attrs[n], None) is not None]

    def param_count(self) -> int:
        """GERÇEK `sum(p.numel())`. fit edilmemişse RuntimeError."""
        if not self.is_fitted:
            raise RuntimeError("NeuralCapacity henüz fit edilmedi")
        return int(sum(p.numel() for p in self._model.parameters()))

    def meta(self) -> dict:
        """{'tier','n_features','param_count','seed','trained_at'} (models.py ile birebir
        anahtar seti) + fit-sonrası EK alanlar (n_epochs_trained/early_stopped/val_loss/
        aux_heads_trained/has_val_split + v4: recipe/temperature/val_metrics/train_zmax_q99/
        train_pos_rate/train_seconds — orkestratör entegrasyon notu: bunlar CapacityModel
        meta()'sında YOK, yalnız neural.py'ye özgü ek gözlemlerdir)."""
        base = {
            "tier": self.tier,
            "n_features": self.n_features_,
            "param_count": self.param_count() if self.is_fitted else None,
            "seed": self.seed,
            "trained_at": self.trained_at,
            "extra_heads": self.effective_extra_heads(),
        }
        if self.is_fitted:
            base.update(self._fit_info)
        return base

    def save(self, path_base: Path) -> None:
        """`<path_base>.json` (meta+feature_names+özellik-normalizasyon istatistikleri,
        atomic_write_text) + `<path_base>.pt` (state_dict, tmp'ye yaz → atomic_replace).
        YALNIZ verilen path_base'e yazar. fit edilmemişse RuntimeError."""
        if not self.is_fitted:
            raise RuntimeError("fit edilmemiş NeuralCapacity kaydedilemez")
        base = Path(path_base)
        base.parent.mkdir(parents=True, exist_ok=True)
        json_path = Path(f"{base}.json")
        pt_path = Path(f"{base}.pt")
        assert self._feat_mu is not None and self._feat_sigma is not None
        payload = {
            **self.meta(),
            "feature_names": list(self.feature_names),
            "feat_mu": self._feat_mu.tolist(),
            "feat_sigma": self._feat_sigma.tolist(),
        }
        atomic_write_text(json_path, json.dumps(payload, ensure_ascii=False, indent=2))
        pt_tmp = tmp_path(pt_path)     # yazana özel ara dosya (eşzamanlı yazan çakışmasın)
        # CPU'ya taşıyarak kaydet: GPU'da eğitilmiş bir artefakt CPU-only bir canlı süreçte
        # de yüklenebilmelidir (load zaten map_location="cpu" kullanır; simetri için).
        torch.save({k: v.detach().cpu() for k, v in self._model.state_dict().items()}, pt_tmp)
        atomic_replace(pt_tmp, pt_path)

    @classmethod
    def load(cls, path_base: Path) -> NeuralCapacity | None:
        """Savunmacı yükleme (models.py/memory.py deseniyle aynı): dosya yok/bozuk/eksik
        alan → None (çökme yok). torch yoksa da None (fail-soft — yükleme mümkün değil).
        v3 artefaktları (temperature/train_zmax_q99 alanı OLMAYAN JSON'lar) aynen okunur —
        eksik alanlar nötr varsayılana düşer (geriye dönük uyum sözleşmesi)."""
        if not TORCH_AVAILABLE:
            return None
        base = Path(path_base)
        json_path = Path(f"{base}.json")
        pt_path = Path(f"{base}.pt")
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict) or "tier" not in raw:
            return None
        names = raw.get("feature_names")
        if not isinstance(names, list) or not names:
            return None
        # Tur-36: `extra_heads` YOKSA (v7 ve öncesi artefaktlar) None = basamak varsayılanı
        # → eski state_dict birebir yüklenir. Varsa artefaktın ÖLÇÜLEN başlık listesidir.
        eh = raw.get("extra_heads")
        eh_t = tuple(str(e) for e in eh) if isinstance(eh, list) else None
        try:
            seed = int(raw.get("seed", 42))
            obj = cls(tier=str(raw["tier"]), feature_names=[str(n) for n in names], seed=seed,
                      extra_heads=eh_t)
        except (ValueError, TypeError):
            return None
        try:
            model = _build_torch_model(obj.tier, obj.feature_names, obj.seed, obj.extra_heads)
            state = torch.load(pt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
        except (OSError, RuntimeError, EOFError, KeyError, ValueError, TypeError):
            return None
        obj._model = model
        n_features = raw.get("n_features")
        try:
            obj.n_features_ = int(n_features) if n_features is not None else None
        except (TypeError, ValueError):
            obj.n_features_ = None
        trained_at = raw.get("trained_at")
        obj.trained_at = str(trained_at) if trained_at is not None else None
        mu, sigma = raw.get("feat_mu"), raw.get("feat_sigma")
        try:
            obj._feat_mu = np.asarray(mu, dtype=float) if mu is not None else np.zeros(len(names))
            obj._feat_sigma = (np.asarray(sigma, dtype=float) if sigma is not None
                               else np.ones(len(names)))
        except (TypeError, ValueError):
            obj._feat_mu = np.zeros(len(names))
            obj._feat_sigma = np.ones(len(names))
        try:
            obj.temperature = float(raw.get("temperature", 1.0) or 1.0)
        except (TypeError, ValueError):
            obj.temperature = 1.0
        try:
            zq = raw.get("train_zmax_q99")
            obj.train_zmax_q99 = float(zq) if zq is not None else None
        except (TypeError, ValueError):
            obj.train_zmax_q99 = None
        obj._fit_info = {k: raw[k] for k in
                         ("n_epochs_trained", "early_stopped", "val_loss",
                          "aux_heads_trained", "has_val_split", "recipe", "best_score",
                          "temperature", "train_zmax_q99", "train_pos_rate", "val_metrics",
                          "train_seconds", "pretrain") if k in raw}
        pre = raw.get("pretrain")
        obj._pretrain_info = dict(pre) if isinstance(pre, dict) else {}
        obj._fitted = True     # diskteki artefakt TANIM GEREĞİ etiketli eğitimden geçmiştir
        return obj


class NeuralEnsemble:
    """Derin topluluk (Lakshminarayanan ve ark. 2017, "Deep Ensembles"): K bağımsız üye,
    üye i seed = `seed + i`. `NeuralCapacity` ile AYNI API (fit/predict_proba/
    predict_with_uncertainty/param_count/meta/save/load/is_fitted/ood_zmax/evaluate) —
    çağıran hangi nesneyle çalıştığını bilmek zorunda değildir (`load_any` bunu kullanır).

    NEDEN v4'te var:
      1) KAPASİTE: param_count üye toplamıdır → aynı mimariyle K katı kapasite.
      2) BELİRSİZLİK: `predict_with_uncertainty` p'nin örneklem varyansını TOPLAM VARYANS
         YASASI ile birleştirir: Var[p] = E[Var[p|üye]] + Var[E[p|üye]]. İKİ terim de
         EPİSTEMİKTİR (Tur-77 düzeltmesi — önceki metin üye-içi MC-dropout yayılımını
         "aleatorik-benzeri" diye anıyordu; dropout ağırlık belirsizliğini örnekler). Gerçek
         aleatorik/epistemik ayrışma: `predict_uncertainty_decomposition` (p(1−p) · BALD).
      3) KALİBRASYON: topluluk ortalaması tek modelden sistematik olarak daha iyi kalibre
         olur (aşırı-güven kırılır) — literatürdeki en tekrarlanabilir kazanç.

    ANAYASA DEĞİŞMEZ: topluluk da KARAR-DIŞI/GÖLGE'dir. "K model aynı şeyi söylüyor" bir
    kenar kanıtı DEĞİLDİR (hepsi aynı veriye bakar; ACI DERS #5)."""

    def __init__(self, tier: str, feature_names: list[str], seed: int = 42,
                 n_members: int = 5,
                 extra_heads: tuple[str, ...] | None = None) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch kurulu değil — neural katman devre dışı (fail-soft)")
        if tier not in NEURAL_TIERS:
            raise ValueError(f"bilinmeyen tier: {tier!r} (geçerli: {NEURAL_TIERS})")
        if int(n_members) < 1:
            raise ValueError("n_members >= 1 olmalı")
        self.tier = tier
        self.feature_names: list[str] = list(feature_names)
        self.seed = int(seed)
        self.n_members = int(n_members)
        self.members: list[NeuralCapacity] = []
        # Üyeler AYNI ek başlık kümesini alır: farklı üyelerin farklı başlığı olsaydı
        # `predict_evidential` bir kısmında None döner, topluluk sessizce yarım kalırdı.
        # None = basamağın kendi varsayılanı (üye artefaktları kendi seçimini taşır).
        self.extra_heads: tuple[str, ...] | None = (
            None if extra_heads is None else tuple(extra_heads))
        self.n_features_: int | None = None
        self.trained_at: str | None = None

    @property
    def is_fitted(self) -> bool:
        return bool(self.members) and all(m.is_fitted for m in self.members)

    @property
    def temperature(self) -> float:
        """Üye sıcaklıklarının ortalaması (yalnız raporlama; olasılıklar üye düzeyinde
        zaten kalibre edilmiş olarak ortalanır)."""
        return (float(np.mean([m.temperature for m in self.members])) if self.members else 1.0)

    @property
    def _feat_mu(self) -> np.ndarray | None:
        return self.members[0]._feat_mu if self.members else None

    @property
    def _feat_sigma(self) -> np.ndarray | None:
        return self.members[0]._feat_sigma if self.members else None

    @property
    def train_zmax_q99(self) -> float | None:
        vals = [m.train_zmax_q99 for m in self.members if m.train_zmax_q99 is not None]
        return float(np.mean(vals)) if vals else None

    def pretrain(self, X: ArrayLike, **kwargs: Any) -> NeuralEnsemble:
        """Her üyeyi AYRI seed ile maskeli-özellik ön-eğitiminden geçirir (bkz.
        `NeuralCapacity.pretrain`). Üyeler farklı ilklendirme VE farklı maske deseni görür
        → topluluk çeşitliliği ön-eğitim aşamasında da korunur. Ardından
        `fit(..., warm_start=True)` ile etiketli eğitime devam edilir.

        `is_fitted` yine False kalır (etiket görülmedi) — tek üye sözleşmesiyle aynı."""
        members: list[NeuralCapacity] = []
        for i in range(self.n_members):
            m = NeuralCapacity(self.tier, self.feature_names, seed=self.seed + i,
                               extra_heads=self.extra_heads)
            m.pretrain(X, **kwargs)
            members.append(m)
        self.members = members
        self.n_features_ = members[0].n_features_
        return self

    def fit(self, X: ArrayLike, y: ArrayLike, aux: dict[str, np.ndarray] | None = None,
            **kwargs: Any) -> NeuralEnsemble:
        """Her üyeyi AYNI veriyle, FARKLI seed ile eğitir (bagging YOK — deep ensembles
        deseni: çeşitlilik ilklendirme + toplu-iş sırasından gelir; az-veri rejiminde
        bootstrap ayrıca veri kaybettirirdi). `kwargs` doğrudan `NeuralCapacity.fit`'e geçer.

        Tur-35: `warm_start=True` ve elde ÖN-EĞİTİLMİŞ üyeler varsa onlar yeniden kurulmaz
        (yoksa `pretrain` çıktısı sessizce çöpe giderdi ve çağıran ön-eğitim yaptığını
        sanmaya devam ederdi). Varsayılan davranış değişmedi."""
        warm = bool(kwargs.get("warm_start")) and len(self.members) == self.n_members
        members: list[NeuralCapacity] = []
        for i in range(self.n_members):
            m = (self.members[i] if warm
                 else NeuralCapacity(self.tier, self.feature_names, seed=self.seed + i,
                                     extra_heads=self.extra_heads))
            m.fit(X, y, aux=aux, **kwargs)
            members.append(m)
        self.members = members
        self.n_features_ = members[0].n_features_
        self.trained_at = _utc_now_iso()
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("NeuralEnsemble henüz fit edilmedi")
        return np.mean([m.predict_proba(X) for m in self.members], axis=0).astype(float)

    def to_device(self, device: str) -> Any:
        """Tüm üyeleri aynı aygıta taşır (bkz. `NeuralCapacity.to_device`)."""
        devs = [m.to_device(device) for m in self.members]
        return devs[0] if devs else _resolve_device("cpu")

    def predict_with_uncertainty(self, X: ArrayLike, T: int = 20, *,
                                 batched: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(ortalama_p, toplam_std). Toplam varyans yasası: üye-içi MC varyansının ortalaması
        + üye ortalamalarının varyansı. Tek üyede `NeuralCapacity` ile aynı sonucu verir
        (üye-arası varyans 0)."""
        if not self.is_fitted:
            raise RuntimeError("NeuralEnsemble henüz fit edilmedi")
        means, varis = [], []
        for m in self.members:
            mu_m, sd_m = m.predict_with_uncertainty(X, T=T, batched=batched)
            means.append(mu_m)
            varis.append(sd_m ** 2)
        mean_arr = np.stack(means, axis=0)
        total_var = np.mean(np.stack(varis, axis=0), axis=0) + mean_arr.var(axis=0)
        return mean_arr.mean(axis=0).astype(float), np.sqrt(total_var).astype(float)

    def predict_vol_quantiles(self, X: ArrayLike) -> np.ndarray | None:
        """Üye quantile demetlerinin ortalaması (n, Q); başlık yoksa None. Quantile
        ortalaması ARTANLIK'ı korur (artan dizilerin ortalaması artandır) — çapraz-geçiş
        toplulukta da yapısı gereği imkânsızdır."""
        if not self.is_fitted:
            raise RuntimeError("NeuralEnsemble henüz fit edilmedi")
        parts = [m.predict_vol_quantiles(X) for m in self.members]
        ok = [p for p in parts if p is not None]
        if len(ok) != len(parts):
            return None
        return np.mean(np.stack(ok, axis=0), axis=0).astype(float)

    def predict_evidential(self, X: ArrayLike) -> tuple[np.ndarray, np.ndarray] | None:
        """Üyelerin Dirichlet kanıtı KONSANTRASYON düzeyinde birleştirilir (α ortalaması),
        p/u oradan türetilir — p ve u'yu ayrı ayrı ortalamak tutarsız olurdu (u = K/Σα
        doğrusal DEĞİLDİR). Üyelerden biri bile başlığı taşımıyorsa None."""
        if not self.is_fitted:
            raise RuntimeError("NeuralEnsemble henüz fit edilmedi")
        parts = [m.predict_evidential(X) for m in self.members]
        if any(p is None for p in parts):
            return None
        # α = (K/u) ile geri çözülür: p ve u'dan konsantrasyon tek anlamlıdır.
        alphas = []
        for p_m, u_m in parts:                    # type: ignore[misc]
            with np.errstate(divide="ignore", invalid="ignore"):
                strength = np.where(u_m > 0, 2.0 / u_m, np.inf)
            alphas.append(np.stack([(1.0 - p_m) * strength, p_m * strength], axis=1))
        alpha = np.mean(np.stack(alphas, axis=0), axis=0)
        s = alpha.sum(axis=1)
        return (alpha[:, 1] / s).astype(float), (2.0 / s).astype(float)

    def predict_uncertainty_decomposition(self, X: ArrayLike, T: int = 20) -> dict[str, Any] | None:
        """Tur-77 · Madde 5.1: tüm üyelerin MC örneklemleri (K·T, n) üzerinde
        `decompose_uncertainty` + epistemik varyansın iki kaynağı AYRI:
          `uye_ici_mc_var`  = E_üye[Var_MC p]  (dropout'un ağırlık belirsizliği)
          `uye_arasi_var`   = Var_üye[E_MC p]  (modeller arası ayrışma — "model cehaleti")
        ve eşit T'de `epistemik_var = uye_ici_mc_var + uye_arasi_var` (testte kilitli).
        `p` ve `sqrt(epistemik_var)`, `predict_with_uncertainty` ile float32 yuvarlaması
        düzeyinde (≤1e-6) aynıdır."""
        if not self.is_fitted:
            raise RuntimeError("NeuralEnsemble henüz fit edilmedi")
        parts = [m._mc_samples(X, T) for m in self.members]
        out = decompose_uncertainty(np.concatenate(parts, axis=0))
        if out is None:
            return None
        out["uye_ici_mc_var"] = np.mean(np.stack([s.var(axis=0) for s in parts], axis=0), axis=0)
        out["uye_arasi_var"] = np.stack([s.mean(axis=0) for s in parts], axis=0).var(axis=0)
        return out

    def predict_belief(self, X: ArrayLike) -> dict[str, Any] | None:
        """Tur-77 · Madde 5.2: üyelerin Dirichlet α'ları KONSANTRASYON düzeyinde ortalanır
        (`predict_evidential` ile aynı ilke) ve inanç kütleleri oradan türetilir. Bir üye bile
        başlığı taşımıyorsa None."""
        if not self.is_fitted:
            raise RuntimeError("NeuralEnsemble henüz fit edilmedi")
        alphas = [m._dir_alpha(X) for m in self.members]
        if any(a is None for a in alphas):
            return None
        alpha = np.mean(np.stack(alphas, axis=0), axis=0)            # type: ignore[arg-type]
        return None if alpha.shape[0] == 0 else evidential_belief(alpha)

    def member_spread(self, X: ArrayLike) -> np.ndarray:
        """Üyeler arası standart sapma (EPİSTEMİK belirsizlik tek başına). MC-dropout'suz,
        ucuz; "modeller birbirine katılıyor mu?" sorusunun doğrudan ölçüsü."""
        if not self.is_fitted:
            raise RuntimeError("NeuralEnsemble henüz fit edilmedi")
        return np.stack([m.predict_proba(X) for m in self.members], axis=0).std(axis=0).astype(float)

    def ood_zmax(self, X: ArrayLike) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("NeuralEnsemble henüz fit edilmedi")
        return self.members[0].ood_zmax(X)

    def evaluate(self, X: ArrayLike, y: ArrayLike) -> dict:
        return evaluation_metrics(y, self.predict_proba(X))

    def param_count(self) -> int:
        """Üye parametrelerinin TOPLAMI — topluluk gerçekten K× kapasitedir (ortalama
        DEĞİL: her üyenin ağırlıkları diskte ayrı ayrı durur ve çıkarımda hepsi koşar)."""
        if not self.is_fitted:
            raise RuntimeError("NeuralEnsemble henüz fit edilmedi")
        return int(sum(m.param_count() for m in self.members))

    def meta(self) -> dict:
        base: dict[str, Any] = {
            "tier": self.tier,
            "n_features": self.n_features_,
            "param_count": self.param_count() if self.is_fitted else None,
            "seed": self.seed,
            "trained_at": self.trained_at,
            "ensemble": self.n_members,
        }
        if self.is_fitted:
            first = dict(self.members[0].meta())
            for k in ("aux_heads_trained", "has_val_split", "recipe", "train_pos_rate"):
                if k in first:
                    base[k] = first[k]
            base["member_param_count"] = self.members[0].param_count()
            pre = self.members[0].meta().get("pretrain")
            if pre:
                base["pretrain"] = pre
            base["member_val_metrics"] = [m.meta().get("val_metrics") for m in self.members]
            base["temperature"] = round(self.temperature, 4)
            zq = self.train_zmax_q99
            base["train_zmax_q99"] = (round(zq, 4) if zq is not None else None)
        return base

    def save(self, path_base: Path) -> None:
        """`<base>.json` = topluluk manifestosu (ensemble=K + meta + feature_names +
        normalizasyon istatistikleri — tek üyeli artefaktla AYNI anahtar seti, böylece
        `neural_shadow`/`registry` tarafındaki okuyucular değişmeden çalışır);
        üyeler `<base>.m{i}.json` / `<base>.m{i}.pt` olarak yan yana. fit yoksa RuntimeError."""
        if not self.is_fitted:
            raise RuntimeError("fit edilmemiş NeuralEnsemble kaydedilemez")
        base = Path(path_base)
        base.parent.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.members):
            m.save(Path(f"{base}.m{i}"))
        mu, sigma = self._feat_mu, self._feat_sigma
        assert mu is not None and sigma is not None
        payload = {
            **self.meta(),
            "feature_names": list(self.feature_names),
            "feat_mu": mu.tolist(),
            "feat_sigma": sigma.tolist(),
            "members": [f"{base.name}.m{i}" for i in range(len(self.members))],
        }
        atomic_write_text(Path(f"{base}.json"),
                          json.dumps(payload, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path_base: Path) -> NeuralEnsemble | None:
        """Savunmacı yükleme: manifest yok/bozuk, `ensemble` alanı yok, ÜYE dosyalarından
        biri bile okunamıyorsa → None (yarım topluluk sessizce farklı bir modeldir; bunu
        kabul etmek 'sessiz yanlış-model' riskidir)."""
        if not TORCH_AVAILABLE:
            return None
        base = Path(path_base)
        try:
            raw = json.loads(Path(f"{base}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict) or "tier" not in raw or "ensemble" not in raw:
            return None
        names = raw.get("feature_names")
        if not isinstance(names, list) or not names:
            return None
        try:
            k = int(raw["ensemble"])
            obj = cls(tier=str(raw["tier"]), feature_names=[str(n) for n in names],
                      seed=int(raw.get("seed", 42)), n_members=max(1, k))
        except (ValueError, TypeError, KeyError):
            return None
        members: list[NeuralCapacity] = []
        for i in range(obj.n_members):
            m = NeuralCapacity.load(Path(f"{base}.m{i}"))
            if m is None:
                return None
            members.append(m)
        obj.members = members
        obj.n_features_ = members[0].n_features_
        trained_at = raw.get("trained_at")
        obj.trained_at = str(trained_at) if trained_at is not None else members[0].trained_at
        return obj


def load_any(path_base: Path) -> NeuralCapacity | NeuralEnsemble | None:
    """Artefakt tipinden BAĞIMSIZ yükleyici: manifestte `ensemble` varsa `NeuralEnsemble`,
    yoksa `NeuralCapacity`. Tüketiciler (canlı gölge köprüsü, açıklayıcılar) tek çağrıyla
    her iki biçimi de okuyabilsin diye. Okunamazsa None (fail-soft)."""
    if not TORCH_AVAILABLE:
        return None
    base = Path(path_base)
    try:
        raw = json.loads(Path(f"{base}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(raw, dict) and raw.get("ensemble"):
        return NeuralEnsemble.load(base)
    return NeuralCapacity.load(base)


@lru_cache(maxsize=32)
def _numel_of_built_model(tier: str, n_features: int, seed: int) -> int:
    """Modeli kurup parametre sayar. lru_cache YALNIZ burada: bu fonksiyon skalerlerin
    SAF fonksiyonudur. `theoretical_param_count`'un tamamını önbelleğe almak YANLIŞTI —
    o, modül-global `TORCH_AVAILABLE`'ı da okuyor, yani argümanlarının saf fonksiyonu
    değil; testler bayrağı monkeypatch'leyince önbellek `None`'u sonraki çağrılara
    sızdırıyordu (yakalandı: test_theoretical_param_count_sozlesme).

    Tur-35: model `meta` AYGITINDA kurulur — parametreler ŞEKLE sahiptir ama DEPOLAMA
    AYIRMAZ. Parametre SAYISI depolamadan bağımsız olduğundan sonuç birebir aynıdır
    (altı basamakta da doğrulandı) ama maliyet çöker: `quant_transformer_ultra` 772M
    parametreyi kurmak için 3,1 GB ayırıp 5,2 sn harcıyordu — ve `param_ladder_report`
    bunu pano her Beyin sekmesi açtığında istiyor. Ölçüldü (f=8, tüm merdiven):
    7,79 s / ~4,5 GB → 0,68 s / ~0 GB.

    Tur-36: `meta` aygıtı DEPOLAMAYI kaldırdı ama Python NESNE maliyetini kaldırmaz. MEGA
    basamağı blok başına 80 uzman × 16 blok = **1280 uzman modülü** (≈3840 `nn.Linear`)
    kuruyor ve bu tek başına **1,62 s** (ULTRA 494 ms). Tur-35'in kendi dersi ("ölçüm
    bütçesini OKUYAN da öder; yeni bir basamak eklerken 'bu sayıyı kim, ne sıklıkta
    soruyor?' sorusu mimarinin parçasıdır") bu turda kendi eklediğimiz basamakta ihlal
    edildi → KISA YOL eklendi:

      Yönlendirilen uzmanlar ŞEKİL OLARAK BİREBİR AYNIDIR. Model `n_routed = top_k` ile
      kurulur, eksik uzmanların parametresi ÖLÇÜLEN uzman boyutuyla çarpılarak eklenir
      (+ yönlendirici satırları: router `Linear(d_model, n_routed, bias=False)`).
      Bu bir KAPALI-FORM TABLO DEĞİLDİR — çarpan modelin KENDİSİNDEN okunur, uzman
      mimarisi değişse de sayı otomatik izler. `n_routed`/`top_k`'nın GERÇEK değeri
      sınıfın KURUCU VARSAYILANINDAN (`inspect.signature`) alınır: tek doğruluk kaynağı
      yine sınıfın kendisidir, ikinci bir sabit tablosu doğmaz.
      `tests/test_tur36_ai.py::test_param_sayimi_kisa_yol_tam_kurulumla_ayni` kilitler.

    Fail-soft: `meta` yolu herhangi bir nedenle patlarsa (gelecekteki bir mimari kurulumda
    gerçek veri okursa) sessizce GERÇEK kuruluma düşer — sayı yine doğru üretilir."""
    names = [f"f{i}" for i in range(n_features)]
    over, n_routed, top_k = _routed_expert_shortcut(tier)
    try:
        with torch.device("meta"):
            model = _build_torch_model(tier, names, seed, None, **over)
    except (RuntimeError, NotImplementedError, TypeError, ValueError):
        model = _build_torch_model(tier, names, seed, None, **over)
    total = int(sum(p.numel() for p in model.parameters()))
    if over:
        ffns = [f for f in _moe_ffn_modules(model) if getattr(f, "n_routed", None)]
        if ffns:
            per_expert = int(sum(p.numel() for p in ffns[0].experts[0].parameters()))
            d_model = int(getattr(model, "d_model", 0))
            total += len(ffns) * (n_routed - top_k) * (per_expert + d_model)
    return total


def _routed_expert_shortcut(tier: str) -> tuple[dict[str, Any], int, int]:
    """İNCE-TANELİ MoE basamakları için (`{'n_routed': top_k}`, n_routed, top_k) döndürür;
    diğerleri için ({}, 0, 0). Değerler sınıfın KURUCU VARSAYILANINDAN okunur (model
    KURULMADAN) — bkz. `_numel_of_built_model` docstring'i."""
    try:
        params = inspect.signature(_tier_class(tier).__init__).parameters
    except (KeyError, TypeError, ValueError):
        return {}, 0, 0
    n_routed = params["n_routed"].default if "n_routed" in params else None
    top_k = params["top_k"].default if "top_k" in params else None
    if isinstance(n_routed, int) and isinstance(top_k, int) and n_routed > top_k >= 1:
        return {"n_routed": top_k}, n_routed, top_k
    return {}, 0, 0


def theoretical_param_count(tier: str, n_features: int, seed: int = 42) -> int | None:
    """Fit GEREKTİRMEDEN gerçek mimari parametre sayısı: model kurulur, numel toplanır
    (kapalı-form tablo DEĞİL — mimari değişirse sayı otomatik izler, tablo-kod sapması
    imkânsız). torch yoksa None (fail-soft — models.param_ladder_report degrade satırı
    için). Bilinmeyen tier / n_features<1 → ValueError (API ihlali, programcı hatası).
    Parametre SAYISI ağırlık değerlerinden bağımsızdır; seed yalnız kurulum determinizmi.

    Sayım `_numel_of_built_model` ile önbelleklenir (Tur-30) — modelin GERÇEKTEN kurulması
    gerekiyor ve ÖLÇÜLDÜ: quant_transformer_xl 171 ms; pano bu raporu her Beyin sekmesi
    açılışında istiyor. Doğrulamalar ve TORCH_AVAILABLE dalı önbellek DIŞINDA kalır."""
    if tier not in NEURAL_TIERS:
        raise ValueError(f"bilinmeyen neural tier: {tier!r} (geçerli: {NEURAL_TIERS})")
    if int(n_features) < 1:
        raise ValueError("n_features >= 1 olmalı")
    if not TORCH_AVAILABLE:
        return None
    return _numel_of_built_model(tier, int(n_features), seed)


# ---------------------------------------------------------------------------
# Yönlendirme TEŞHİSİ (Tur-34) — Tur-32 açık kalem #3'ün kapanışı
# ---------------------------------------------------------------------------
def _moe_ffn_modules(module: Any) -> list[Any]:
    """Bir nn.Module ağacındaki tüm MoE FFN'lerini blok sırasıyla döndürür (MoE ve
    ince-taneli MoE-XL varyantlarının ikisini de yakalar; MoE olmayan modelde boş liste)."""
    if not TORCH_AVAILABLE or not hasattr(module, "modules"):
        return []
    return [m for m in module.modules()
            if hasattr(m, "router") and hasattr(m, "last_top1")
            and hasattr(m, "record_routing")]


def _chi2_independence(table: np.ndarray) -> dict | None:
    """Grup×uzman olumsallık tablosunda bağımsızlık ki-kare testi + Cramér V etki büyüklüğü.

    scipy LAZY import edilir — `neural.py` canlı gölge yolunda import edilir ve modül
    düzeyinde scipy çekmek o yolu pahalılaştırırdı (Tur-26'nın "`import nihai.ai` ucuz
    kalmalı" disiplini).

    Beklenen sayımların %20'den fazlası 5'in altındaysa ki-kare yaklaşımı GEÇERSİZDİR →
    `reliable=False` ve p DÖNDÜRÜLMEZ (uydurma anlamlılık yok — Tur-31/33 dersi)."""
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
        return None
    total = float(table.sum())
    if total <= 0.0:
        return None
    row = table.sum(axis=1, keepdims=True).astype(float)
    col = table.sum(axis=0, keepdims=True).astype(float)
    expected = row @ col / total
    if not np.all(expected > 0.0):
        # Hiç gözlenmemiş satır/sütun → tabloyu daralt (sıfır beklenen sayım tanımsız).
        keep_r = row.ravel() > 0.0
        keep_c = col.ravel() > 0.0
        if int(keep_r.sum()) < 2 or int(keep_c.sum()) < 2:
            return None
        return _chi2_independence(table[np.ix_(keep_r, keep_c)])
    chi2 = float(np.sum((table - expected) ** 2 / expected))
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    small = float(np.mean(expected < 5.0))
    reliable = small <= 0.20
    k = min(table.shape) - 1
    cramer_v = float(math.sqrt(chi2 / (total * k))) if k > 0 else None
    p_value: float | None = None
    if reliable:
        from scipy import stats as _stats
        p_value = float(_stats.chi2.sf(chi2, dof))
    return {"chi2": chi2, "dof": int(dof), "cramer_v": cramer_v, "n": int(total),
            "small_expected_share": small, "reliable": reliable, "p_value": p_value}


def moe_routing_stats(model: Any, X: ArrayLike, *, groups: ArrayLike | None = None,
                      batch_size: int = _INFER_BATCH) -> dict | None:
    """MoE yönlendiricisinin DAVRANIŞINI ölçer — Tur-32 açık kalem #3'ün kapanışı.

    İki ayrı soru, iki ayrı ölçüm:

    1. **ÇÖKME var mı?** (tüm token'lar üzerinden) `usage` = uzman başına top-1 payı,
       `entropy_norm` = kullanım dağılımının normalize Shannon entropisi (1.0 = tam düz,
       0.0 = tek uzmana çökmüş), `max_share`. Yük dengeleme kaybının İŞE YARAYIP
       yaramadığının tek dürüst kanıtı budur — kayıp DEĞERİNİ raporlamak yetmez.
    2. **UZMANLAŞMA var mı?** (`groups` verilirse; ör. rejim etiketi) [CLS] token'ının
       hangi uzmana yönlendiğine bakılır — karar temsili odur — ve grup×uzman tablosunda
       bağımsızlık ki-kare testi + Cramér V uygulanır. p küçük + V büyük = yönlendirme
       gruba göre farklılaşıyor. **Bu bir KENAR iddiası DEĞİLDİR** (ACI DERS #5): uzmanlar
       rejime göre ayrışsa bile modelin TAHMİN GÜCÜ hakkında hiçbir şey söylemez.

    `model` NeuralCapacity | NeuralEnsemble | nn.Module olabilir. MoE olmayan basamakta
    (ör. quant_transformer_xl) dürüstçe **None** döner — "uzmanlaşma yok" DEĞİL, "ölçülecek
    yönlendirici yok". Ölçüm `torch.no_grad()` + `eval()` altında yapılır (DropPath/dropout
    kapalı → deterministik atama) ve `record_routing` bayrağı ölçüm sonunda GERİ ALINIR.

    Dönüş: {'n_blocks','n_experts','n_samples','groups','blocks':[...],'specialization':[...]|None}
    """
    if not TORCH_AVAILABLE:
        return None
    inner = getattr(model, "_model", None)
    if inner is None and getattr(model, "members", None):
        inner = getattr(model.members[0], "_model", None)
    if inner is None:
        inner = model
    ffns = _moe_ffn_modules(inner)
    if not ffns:
        return None
    xa = np.asarray(X, dtype=float)
    if xa.ndim == 1:
        xa = xa.reshape(1, -1)
    if xa.ndim != 2 or xa.shape[0] < 1:
        return None
    norm = getattr(model, "_normalize", None)
    xn = norm(xa) if callable(norm) else xa
    n = int(xn.shape[0])
    bs = max(1, int(batch_size))

    gvals: np.ndarray | None = None
    if groups is not None:
        ga = np.asarray(groups).ravel()
        if ga.size != n:
            return None
        gvals = ga

    prev_flags = [bool(f.record_routing) for f in ffns]
    n_experts = int(getattr(ffns[0], "n_routed", 0) or getattr(ffns[0], "n_experts", 0))
    counts = np.zeros((len(ffns), max(1, n_experts)), dtype=np.int64)
    cls_assign = np.zeros((len(ffns), n), dtype=np.int64)
    was_training = bool(getattr(inner, "training", False))
    for f in ffns:
        f.record_routing = True
    try:
        inner.eval()
        with torch.no_grad():
            for start in range(0, n, bs):
                chunk = np.ascontiguousarray(xn[start:start + bs], dtype=np.float32)
                inner(torch.from_numpy(chunk))
                b = chunk.shape[0]
                for bi, f in enumerate(ffns):
                    top1 = f.last_top1
                    if top1 is None:
                        continue
                    flat = top1.cpu().numpy().reshape(b, -1)
                    counts[bi] += np.bincount(flat.ravel(), minlength=counts.shape[1])
                    cls_assign[bi, start:start + b] = flat[:, 0]
    finally:
        for f, flag in zip(ffns, prev_flags, strict=False):
            f.record_routing = flag
            f.last_top1 = None
        if was_training:
            inner.train()

    blocks: list[dict] = []
    n_cols = int(counts.shape[1])
    for bi in range(len(ffns)):
        tot = float(counts[bi].sum())
        if tot <= 0.0:
            blocks.append({"block": bi, "usage": None, "entropy_norm": None,
                           "max_share": None, "collapsed": None})
            continue
        share = counts[bi] / tot
        nz = share[share > 0.0]
        ent = float(-np.sum(nz * np.log(nz)))
        ent_norm = float(ent / math.log(n_cols)) if n_cols > 1 else 0.0
        blocks.append({
            "block": bi,
            "usage": [float(v) for v in share],
            "entropy_norm": ent_norm,
            "max_share": float(share.max()),
            # Çökme eşiği: tek uzman token'ların YARISINDAN fazlasını alıyorsa (düz dağılımda
            # pay 1/E olurdu) yük dengeleme fiilen başarısızdır ve kapasite kâğıt üstünde kalır.
            "collapsed": bool(share.max() > 0.5),
        })

    spec: list[dict] | None = None
    if gvals is not None:
        uniq = np.unique(gvals)
        spec = []
        for bi in range(len(ffns)):
            table = np.zeros((uniq.size, n_cols), dtype=np.int64)
            for gi, gv in enumerate(uniq):
                sel = cls_assign[bi][gvals == gv]
                if sel.size:
                    table[gi] = np.bincount(sel, minlength=n_cols)
            res = _chi2_independence(table)
            used_cols = int((table.sum(axis=0) > 0).sum())
            # ÖLÇÜM YOKLUĞU ÖLÇÜM DEĞİLDİR (Tur-31 IV Rank dersi): [CLS] tek uzmana
            # gidiyorsa "uzmanlaşma yok" denemez — karşılaştırılacak ikinci hücre yoktur.
            reason = None if res is not None else (
                "CLS_TEK_UZMAN" if used_cols < 2 else "TEK_GRUP_VEYA_BOS")
            spec.append({"block": bi, "test": res, "measured": res is not None,
                         "reason": reason, "cls_experts_used": used_cols,
                         "table": [[int(v) for v in row] for row in table]})

    return {"n_blocks": len(ffns), "n_experts": n_cols, "n_samples": n,
            "groups": None if gvals is None else [str(v) for v in np.unique(gvals)],
            "blocks": blocks, "specialization": spec}


