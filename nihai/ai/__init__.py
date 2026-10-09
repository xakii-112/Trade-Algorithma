"""
nihai.ai — KARAR-DIŞI/GÖLGE "Quant Beyin" katmanı (FAZ AI v2 + v3: routing + hafıza + akıl
yürütme + finans matematiği + öğrenme + GÖLGE karar + kapasite merdiveni + sinir ağı çekirdeği).

ANAYASA: Bu paket sinyal üretimine, META_GATE'e, eşiklere veya strategy/execution'a ASLA
bağlanmaz. Tüm çıktısı rapor/özet/hafıza/gölge-kayıttır; hiçbir çıktı "kenar bulundu"
imalı olamaz (ACI DERS #3/#4). Karar motoru yalnız off|shadow kiplerine sahiptir —
enforce kipi kasıtlı olarak YOKTUR.

Tur-13 çekirdeği:
- `router`     : görev-profiline göre LLM sağlayıcı/model seçer (cooldown'lı; yoksa None).
- `memory`     : kalıcı, append-güvenli JSONL operatör hafızası (günlük digest vb.).
- `reasoner`   : loglardan FACTS toplar, deterministik Türkçe değerlendirme (dürüstlük guardı).

v2 "Quant Beyin" modülleri (2026-07-19, spec: docs/AI_V2_BEYIN_SPEC.md):
- `quant_math` : derin finans matematiği (v3'te GARCH/EVT/kopula/FFD/spektral/entropi/
                 yüksek-moment aileleriyle genişledi).
- `models`     : parametre KAPASİTE merdiveni (v3: logistic→…→mlp→deep_mlp→quant_transformer;
                 gerçek param_count; promotion kanıt-kapılı manuel yolda kalır).
- `episodic`   : epizodik hafıza (v3: kosinüs recall + unutma eğrisi + ders-adayı) + LESSONS.
- `learning`   : çözülen tahminlerden kalibrasyon/Beta-posterior/drift (v3: kaynak-Brier
                 `sources` + Platt/isotonic kalibratör + retrain_candidate bayrağı).
- `knowledge`  : borsa/ekonomi/quant kavram çekirdeği (v3: 100+ kavram + curriculum DAG'ı).
- `decision`   : GÖLGE Bayes karar motoru (v3: hiyerarşik kaynak birleşimi + fayda +
                 çift/üçlü kapı; AI_DECISION_MODE=off|shadow; enforce YOK).
- `brain`      : düşünme döngüsü orkestratörü (v3: +ÖNGÖR +ÖZ-ELEŞTİR + anlamsal HATIRLA).

v3 "Derin Beyin" modülleri (2026-07-23, spec: docs/AI_V3_DERIN_BEYIN_SPEC.md — LAZY yüklenir):
- `neural`     : PyTorch çekirdeği (deep_mlp ~380k · quant_transformer ≈5.4M · v4/Tur-22
                 quant_transformer_xl ≈30M parametre + `NeuralEnsemble` K× kapasite;
                 torch yoksa fail-soft — TORCH_AVAILABLE=False, hiçbir akış kırılmaz).

v4 "Kapasite ve Kalibrasyon" turu (2026-08-01, Tur-22): `neural` katmanına 12 bloklu
SwiGLU/RMSNorm/LayerScale/stokastik-derinlik + PLR gömmeli XL mimarisi, derin topluluk,
opt-in eğitim reçetesi (kosinüs LR, EMA, sınıf ağırlığı, label smoothing, val-AUC erken
durma, sıcaklık kalibrasyonu, cihaz seçimi), veri-türetilmiş OOD eşiği ve ECE ölçüsü
eklendi. Hepsi GÖLGE kalır — kapasite artışı bir kenar iddiası DEĞİLDİR (ACI DERS #5).
- `consolidate`: epizot konsolidasyonu → anlamsal hafıza (gece görevi; semantic_recall).

v5 "Derinleşme" turu (2026-08-14, Tur-32) — üç YENİ modül + iki mevcut modülde genişleme:
- `options_math`  : opsiyon & volatilite YÜZEYİ (implied vol kök bulma, yüksek mertebeden
                    greekler, put-call paritesi/örtük forward, raw SVI + Durrleman
                    arbitraj kontrolü, SABR/Hagan, 25Δ gülümseme ölçüleri,
                    Breeden-Litzenberger risk-nötr yoğunluk, calendar arbitraj).
                    `knowledge.py`'nin KENDİ kataloğunda "uygulaması YOK" yazan iki boşluğu
                    (implied_vol · vol_smile) kapatır.
- `microstructure`: gerçekleşen oynaklık ÖNGÖRÜSÜ (HAR-RV/HAR-CJ), sıçrama ayrıştırması,
                    gürültü-dayanıklı TSRV + oynaklık imzası, likidite/etki ölçüleri
                    (Amihud, Roll, Corwin-Schultz, Kyle λ, BVC-VPIN). ACI DERS #7'nin
                    ("kenar ilk bardaki makas sıçramasıydı") TEŞHİS aracı.
- `online`        : çevrimiçi uyarlama — ADWIN sürüklenme dedektörü, Hedge/fixed-share
                    uzman birleştirme (koşan taban oranı OTOMATİK yarışmacı = ACI DERS #9
                    makinede), FTRL-Proximal akış-içi lojistik regresyon.
- `learning`      : + `log_loss_score`, `brier_skill_score` (BSS), `murphy_decomposition`
                    (güvenilirlik/ÇÖZÜNÜRLÜK/belirsizlik). Kaynak karnesi artık BSS ve
                    RES taşır: "Brier düştü" satırı tek başına artık okunamaz.
- `neural`        : + `quant_transformer_moe` (≈93M parametre, 6 uzmanlı top-2 seyrek
                    yönlendirme). Kapasite XL'in ≈3 katı, TOKEN BAŞINA hesap ≈aynı —
                    merdivenin üst ucunu bu donanımda tırmanılabilir tutan yapı budur.
                    Kapasite artışı bir KENAR iddiası DEĞİLDİR (ACI DERS #5).

v6 "Belirsizlik ve Kapasite" turu (2026-08-15, Tur-34) — bir YENİ modül + dört genişleme.
Hepsi KARAR-DIŞI; hiçbiri kenar iddiası taşımaz. Ortak seçim ilkesi: `knowledge.py`'nin
KENDİ kataloğunda "bu modülde YOKTUR" yazan boşluklar ve Tur-32'nin açık kalemleri.
- `regime_hmm`    : YENİ. Rejim GEÇİŞ OLASILIKLARI (Gauss saklı Markov modeli, Baum-Welch).
                    Katalog "rejim etiketi KATEGORİKTİR — geçiş OLASILIĞI bilgisi yoktur"
                    diyordu. `hmm_filter` NEDENSEL (canlı yolun tek geçerli çıktısı),
                    `hmm_smooth`/`hmm_viterbi` look-ahead içerir ve bunu `causal=False`
                    bayrağıyla MAKİNE-OKUNUR biçimde taşır.
- `quant_math`    : + ASİMETRİK (arşimedyen) kopulalar — Clayton (alt kuyruk) / Gumbel (üst
                    kuyruk) + `copula_tail_dependence` (kapalı form λ) + `copula_select`
                    (dört aile, AIC, ΔAIC<2'de `decisive=False`). Gauss "kuyruk yok",
                    t "simetrik kuyruk" varsayardı; asimetri artık VERİYE sorulur.
- `microstructure`: + PIN (Easley-Kiefer-O'Hara-Paperman yapısal MLE'si, Lin-Ke sayısal
                    çarpanlamasıyla) + `pin_from_bars` (OHLCV vekili, `proxy=True` taşır).
                    Makasın BÜYÜKLÜĞÜNÜ ölçen araçların yanına NEDENİNİ (ters seçim payı)
                    ölçen araç eklendi.
- `options_math`  : + SSVI (Gatheral-Jacquier arbitrajsız yüzey) + `ssvi_diagnose` —
                    Tur-32 açık kalem #5'in kapanışı: bir kelebek ihlali UYUM ARTEFAKTI mı
                    yoksa VERİDE mi, iki bağımsız kalibrasyonun kıyasıyla ayrılır.
- `neural`        : + `quant_transformer_moe_xl` (≈284M parametre; 16 İNCE-TANELİ uzman
                    top-2 + her token'da çalışan PAYLAŞIMLI uzman, ST-MoE router z-kaybı)
                    ve `moe_routing_stats` — Tur-32 açık kalem #3'ün kapanışı: yönlendirici
                    çöküyor mu, uzmanlar gruba göre AYRIŞIYOR mu (ki-kare + Cramér V).

v7 "Etiketsiz Veri, Dağılım ve Ölçek" turu (2026-08-16, Tur-35):
- `neural`        : + `quant_transformer_ultra` (**771.874.072** parametre; 40 İNCE-TANELİ
                    uzman top-2 + paylaşımlı uzman; aktif 98.482.456 = seyreklik 7,84×),
                    + DAĞILIMSAL oynaklık başlığı `vol_q` (pinball kaybı; çapraz geçiş
                    kayba değil MİMARİYE yasaklandı) ve
                    + `NeuralCapacity.pretrain` — maskeli özellik ÖN-EĞİTİMİ. **Etiket
                    gerektirmez** (ACI DERS #5'in veri tarafına dokunan ilk kol) ve
                    `is_fitted`'i AÇMAZ: etiket görmemiş gövdeden p_win okunamaz.
- `learning`      : + `source_fusion` — Tur-32 açık kalem #4'ün kapanışı. `pwin`/`neural`/
                    koşan-taban üçlüsü ÇEVRİMİÇİ (Hedge + fixed-share) birleştirilir; kapı
                    `vs_base > 0` VE eşleştirilmiş Brier t < −1,645. ÖLÇÜLDÜ: canlı logda
                    **TABANI_GECILEMEDI** (ACI DERS #9'un üçüncü bağımsız doğrulaması).

v8 "Belirsizlik, Opt-in Başlık ve Ölçek" turu (2026-08-17, Tur-36):
- `neural`        : + `quant_transformer_mega` (**2.014.146.906** parametre; 80 İNCE-TANELİ
                    uzman top-2, d=768, 16 blok — C(80,2)=3160 uzman kombinasyonu),
                    + **OPT-IN ek başlıklar** (`extra_heads`): `vol_q` artık HER basamakta
                    açılabilir (Tur-35 açık kalem #2) ve varsayılan boş → v7 artefaktları
                    BİT-DENK yüklenir,
                    + **kanıt-temelli yön** `dir_evid` (Dirichlet/EDL) + `predict_evidential`
                    → epistemik belirsizlik TEK geçişte. ÖLÇÜLDÜ (canlı 1m, 5 seed): kanıt
                    boşluğu `u` güvenilirliği **5/5 seed'de DOĞRU** sıralar (düşük-u dilimi
                    isabeti yüksek-u diliminden ort. **+0,098**), MC-dropout std'si ise
                    **0/5** (ters sıralar) ve 124× daha pahalıdır.
- `learning`      : `source_fusion` artık **K uzman** alır (`extra=`). ÖLÇÜLDÜ (canlı log,
                    n=1307): `prob_cal` sistemin TABANI geçen ilk TEK uzmanı (0,693865 ↔
                    taban 0,694327) ama BİRLEŞİM yine geçemedi → **TABANI_GECILEMEDI**.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "brain",
    "consolidate",
    "decision",
    "episodic",
    "knowledge",
    "learning",
    "memory",
    "microstructure",
    "models",
    "neural",
    "observatory",
    "online",
    "options_math",
    "quant_math",
    "reasoner",
    "regime_hmm",
    "router",
]

# TÜM alt modüller lazy (Tur-26). Öncesinde `decision`/`quant_math`/`models`/… modül
# düzeyinde EAGER import ediliyordu; `decision → quant_math → sklearn.covariance +
# scipy.stats` zinciri yüzünden ÖLÇÜLDÜ: `import nihai.ai` = 3854 ms. Bunu `from nihai.ai
# import registry` diyen CANLI yol (execution/neural_shadow) ve her `web/api_extra` AI ucu
# ödüyordu — oysa bu dosyanın kendi gerekçesi zaten "`import nihai.ai` her çağıran için ucuz
# kalmalıdır" diyor. Eager blok API için GEREKSİZDİ: `from nihai.ai import X` deseni zaten
# import sisteminin alt-modül çözümüyle çalışır (bu yüzden `registry`/`selfcheck`/`agent`
# hiç listede olmadan çalışıyordu); eager import yalnız `import nihai.ai` + `nihai.ai.X`
# ÖZNİTELİK erişimini destekliyordu — onu da aşağıdaki PEP 562 kancası karşılıyor.
# LİSTE DİZİNLE EŞİT OLMALIDIR ve bu artık MAKİNEDE denetlenir
# (`tests/test_tur76_kopruler.py::test_nihai_ai_LAZY_listesi_dizinle_ESIT`). Elle
# bakılan bir liste sessizce eskir: Tur-76'da DÖRT modül (`option_flow` Tur-57'den,
# `econophysics`/`ml_hygiene`/`volume_profile` Tur-75'ten) listede yoktu ve
# `import nihai.ai; nihai.ai.volume_profile` AttributeError veriyordu — `from nihai.ai
# import X` deseni çalıştığı için kusur hiçbir yerde görünmemişti.
_LAZY_SUBMODULES = frozenset({
    "adaptive_calibration", "advanced_math", "agent", "brain", "calibration_loop",
    "consolidate", "data_health",
    "decision", "drift_monitor", "econometrics", "econophysics", "episode_resolver", "episodic", "explain",
    "fundamentals", "knowledge", "lab", "learning",
    "memory", "microstructure", "ml_hygiene", "models", "neural", "observatory",
    "online", "option_flow", "options_math", "portfolio",
    "quant_math", "reasoner", "regime_hmm", "registry", "router", "selfcheck",
    "shadow_runner", "statarb", "tda", "text_intel", "volume_profile",
})


def __getattr__(name: str) -> Any:
    """Alt modüller LAZY yüklenir (PEP 562) — `import nihai.ai` maliyeti ~0 kalır.

    Tarihçe: kanca önce yalnız `brain`/`neural`/`consolidate` içindi (brain: runpy
    çift-import uyarısı; neural: torch importu saniyeler sürer; consolidate: brain'in
    bağımlılığı). Tur-26'da tüm alt modüllere genişletildi — gerekçe yukarıdaki blokta,
    ölçümle. importlib kullanılır — `from nihai.ai import X` deseni bu kancanın içinde
    yeniden __getattr__'ı tetikleyip sonsuz özyineleme yapardı."""
    if name in _LAZY_SUBMODULES:
        import importlib
        return importlib.import_module(f"nihai.ai.{name}")
    raise AttributeError(f"module 'nihai.ai' has no attribute {name!r}")
