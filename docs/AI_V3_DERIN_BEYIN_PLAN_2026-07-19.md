# AI v3 "DERİN QUANT BEYİN" — DETAYLI UYGULAMA RAPORU/PLANI (rev.2)
**Tarih:** 2026-07-19 (rev.2 aynı gün) · **Tur:** 16 (aday) · **Durum:** ⏸ **ONAY BEKLİYOR — KOD YAZILMADI**
**rev.2 ekleri:** WP-M2 ekstra ileri matematik · WP-C kod yazma/çalıştırma laboratuvarı
(+dil kararı: Python ana, Numba→C++ merdiveni) · WP-P portföy yönetimi · WP-E kenar-keşif programı
**rev.3 ekleri (2026-07-20):** WP-T metin/haber zekâsı + takvim farkındalığı · WP-X
açıklanabilirlik + conformal belirsizlik · WP-A soru-cevap/ajanlık + öz-değerlendirme +
karşı-olgusal motor · WP-O veri sağlığı + MLOps + operasyon/otonom döngü · çoklu
zaman-ölçeği füzyonu · hiperparametre disiplini

> Kullanıcı talebi: `nihai/ai` katmanını borsa/ekonomi/finans/trade/quant finance'e
> özelleşmiş **tam teşekküllü bir AI modeline** geliştir. Eksenler: karar verme, düşünme,
> hafıza, öğrenme, verileri matematiksel formüllerle işleme, derin/zor matematik,
> **parametre sayısı ×1000**. Planlama/yönetim en gelişmiş model (Fable 5), kodlama uygun
> model (Sonnet ajanları). **Kod, kullanıcı "başla" diyene kadar YAZILMAZ.**

---

## BÖLÜM A — YÖNETİCİ ÖZETİ

**Ne yapılacak:** Mevcut AI v2 "Quant Beyin" (zirve kapasitesi mlp = 3.777 parametre) üstüne,
**≈ 5.5 milyon parametreli** (rev.3; sözleşme tabanı ≥3.777M = ×1000) gerçek bir sinir ağı
çekirdeği (QuantTransformer, PyTorch CPU) +
6 bilişsel yükseltme (derin matematik, hiyerarşik Bayes karar, vektörel/konsolidasyonlu hafıza,
çevrimiçi öğrenme, 100+ kavramlık bilgi tabanı, öz-eleştirili düşünme döngüsü) inşa edilecek.

**Nasıl yapılacak:** Fable 5 orkestratör (bu plan + brifler + entegrasyon + denetim),
3 paralel Sonnet ajanı (örtüşmesiz iş paketleri WP-N/M/D/H/L/K), tek-thread entegrasyon (WP-I).
Tur-15'te kanıtlanmış desen.

**Ne değişmeyecek:** Proje anayasası. AI katmanı KARAR-DIŞI/GÖLGE kalır; sinyal/eşik/
META_GATE/execution'a bağlanmaz; ×1000 parametre bir **kapasite** teslimatıdır, üretime
alınması (promotion) kanıt-kapılı ayrı bir süreçtir (ACI DERS #2/#5).

**Genişletme (2026-07-19 rev.2, kullanıcı isteği):** 4 yeni eksen eklendi —
**WP-M2** ekstra ileri matematik (sıçrama-difüzyon, Heston/FFT fiyatlama, Hawkes, BOCPD,
rough vol, path signatures...), **WP-C** kod yazma/çalıştırma laboratuvarı (dil kararı
analizi dahil: Python + kademeli Numba/C++ hızlandırma merdiveni), **WP-P** portföy yönetimi
(HRP, Black-Litterman, CVaR-optimizasyon, çok-varlık Kelly, stres testi, atribüsyon),
**WP-E** sistematik kenar-keşif programı (yeni veri kaynakları + test edilmemiş sinyal
aileleri + yöntem iyileştirmeleri; hepsi kanıt-kapılı).

**Genişletme (2026-07-20 rev.3, eksik-analizi):** Planın kapsamadığı 10 AI yetkinlik alanı
tespit edilip 4 yeni iş paketiyle kapatıldı — **WP-T** metin/haber zekâsı (finans sözlüğü +
LLM fail-soft, makro/kazanç takvim farkındalığı), **WP-X** açıklanabilirlik + conformal
belirsizlik garantisi (5.5M paramlık ağ kara kutu KALMAZ), **WP-A** soru-cevap/ajanlık
(brain'e Türkçe soru sorma, hipotez kuyruğu, öz-değerlendirme sınavı, karşı-olgusal replay),
**WP-O** veri sağlığı kapısı + model yaşam döngüsü (registry/model kartı/gölge A/B) +
otonom operasyon döngüsü. Ayrıca WP-N'e çoklu zaman-ölçeği füzyonu, WP-L'ye hiperparametre
disiplini eklendi.

**Tahmini kapsam (rev.3):** ~9 yeni modül + 7 modül genişletmesi + research/edge_lab;
+430–520 yeni test (hedef toplam ~1.400+); 1 yeni bağımlılık (torch CPU, fail-soft;
portföy/matematik için scipy yeterli — cvxpy bilinçli dışarıda).

---

## BÖLÜM B — MEVCUT DURUM ANALİZİ (v2 zemini)

| Modül | Satır | v2'de ne var | v3'te ne olacak |
|---|---:|---|---|
| `quant_math.py` | 770 | 40+ fonksiyon: 4 OHLC vol tahmincisi, VaR/CVaR/Cornish-Fisher, OU/Hurst/varyans-oranı, Engle-Granger, Kalman beta, Ledoit-Wolf, PCA, MC (GBM+bootstrap), Black-Scholes+greekler, rejim sınıflandırıcı | **GENİŞLER** (WP-M): +GARCH, EVT, kopula, FFD, spektral, Almgren-Chriss, entropi, yüksek momentler |
| `models.py` | 250 | Kapasite merdiveni logistic 26 → elasticnet 326 → gbm_small ~2.660 → mlp 3.777; `recommend_tier` örnek/param disiplini | **GENİŞLER** (WP-I): +deep_mlp ~380k, +quant_transformer ≈5.5M (rev.3) |
| `decision.py` | 384 | Gölge Bayes: prior×p_win×rejim → LONG_BIAS/SHORT_BIAS/ABSTAIN; CI-tabanı>0.5 kuralı; fractional-Kelly | **GENİŞLER** (WP-D): hiyerarşik birleşim + fayda katmanı |
| `episodic.py` | 298 | z-norm RMS benzer-epizot, LESSONS (13 ders), 5MB/2000-satır koruma | **GENİŞLER** (WP-H): kosinüs recall, konsolidasyon, unutma eğrisi, ders-adayı |
| `learning.py` | 257 | Beta posterior GLOBAL/LONG/SHORT, Brier, Page-Hinkley drift | **GENİŞLER** (WP-L): replay buffer, kalibrasyon, drift-tetikli aday eğitimi |
| `knowledge.py` | 715 | 45 kavram + formül köprüsü + tuzaklar + ders bağları | **GENİŞLER** (WP-K): 100+ kavram + curriculum |
| `brain.py` | 186 | ALGILA→HATIRLA→TOPRAKLA→HESAPLA→KARAR→KAYDET | **GENİŞLER** (WP-I): +ÖNGÖR +ÖZ-ELEŞTİR adımları |
| `memory.py` / `reasoner.py` / `router.py` | 111/271/155 | Tur-13 çekirdeği | Değişmez (arayüzleri kullanılır) |
| — | — | — | **YENİ:** `neural.py` (WP-N), `consolidate.py` (WP-H), `research/train_neural.py` (WP-L) |

**Ortam tespitleri (2026-07-19 ölçüldü):**
- venv: Python 3.12.10; sklearn ✓, scipy ✓, **torch ✗ (kurulacak)**
- 938 test yeşil, ruff + mypy temiz (v2 teslim durumu)
- Çalışılan kopya: `C:\Users\Kagan\OneDrive\Desktop\NihaiAlgoritma toplu\NihaiAlgoritma`
  (⚠ başlarken teyit: hedef bu kopya mı, ana `Desktop\NihaiAlgoritma` mı?)

---

## BÖLÜM C — HEDEF MİMARİ

```
                          brain.think()  —  v3 düşünme döngüsü
  ┌─────────────────────────────────────────────────────────────────────┐
  │ ALGILA → HATIRLA → TOPRAKLA → HESAPLA → ÖNGÖR → KARAR → ÖZ-ELEŞTİR │
  │   │        │          │          │        │       │         │       │
  └───┼────────┼──────────┼──────────┼────────┼───────┼─────────┼──────┘
      │        │          │          │        │       │         └─ zayıf-halka raporu
      │        │          │          │        │       └─ decision v3 (çift kapı:
      │        │          │          │        │          CI-tabanı>0.5 VE fayda>0)
      │        │          │          │        └─ neural.py (YENİ):
      │        │          │          │           QuantTransformer ≈5.5M param (gölge p_win)
      │        │          │          └─ quant_math v3 (+GARCH/EVT/kopula/FFD/entropi...)
      │        │          └─ knowledge v3 (100+ kavram, curriculum)
      │        └─ episodic v3 (kosinüs recall) + consolidate.py (YENİ: anlamsal özet)
      └─ features/closes (çağıran sağlar — üretim hattına BAĞLANMAZ)

  learning v3 (replay + kalibrasyon + drift) ──→ decision & neural (prior/eğitim beslemesi)
  models.py merdiveni: logistic 26 → elasticnet 326 → gbm ~2.6k → mlp 3.8k
                       → deep_mlp ~380k (YENİ) → quant_transformer ≈5.5M (YENİ, rev.3)

  ── rev.2 katmanları ─────────────────────────────────────────────────────
  advanced_math.py (YENİ): Merton/Heston/FFT, Hawkes, BOCPD, rough vol,
                           signatures, PDE/LSMC, qp_solve  → HESAPLA + portföy besler
  lab.py (YENİ):           soru → kod üret → sandbox koş → yorumla → gerekçeye ekle
                           (LLM'siz şablon-deney modu da var)  → ÖZ-ELEŞTİR köprüsü
  portfolio.py (YENİ):     HRP · Black-Litterman(decision görüşleri) · CVaR-LP ·
                           çok-varlık Kelly · stres/atribüsyon  → GÖLGE ağırlık raporu
  research/edge_lab (YENİ): aday-aile harness (CV→çift-bölme→DSR/PBO→exec-lag→PIT)

  ── rev.3 katmanları ─────────────────────────────────────────────────────
  text_intel.py (YENİ):  LM-sözlük ton + LLM özet (fail-soft) + makro/kazanç takvimi
                         → ALGILA'ya bağlam + ağa metin/takvim özellikleri
  explain.py (YENİ):     integrated gradients + attention rollout + CONFORMAL aralık
                         → her karara "açıklama kartı"; conformal = 3. ABSTAIN kapısı
  agent.py+selfcheck.py: brain --ask (Türkçe RAG soru-cevap) · hipotez kuyruğu ·
                         öz-değerlendirme karnesi · karşı-olgusal replay (rapor-only)
  data_health.py+registry.py: veri-sağlık kapısı (karantina→ABSTAIN) · model kartı/
                         sürümleme · gölge A/B · otonom görev döngüsü (gece/sabah/hafta)
```

---

## BÖLÜM D — PARAMETRE ×1000: KESİN HESAP (rev.2 güncellendi)

Referans: mevcut zirve **mlp = 3.777** parametre → sözleşme tabanı **≥ 3.777.000 (×1000)**.

**rev.2 etkisi — girdi uzayı büyümesi:** rev.2 eklerinin çoğu (portföy optimizasyonu,
lab, kapalı-form matematik) öğrenilmiş parametre EKLEMEZ (HRP/BL/CVaR çözücüdür, model
değil; GARCH/Heston/Hawkes fit'leri 3–5 parametrelik istatistiksel modellerdir — dürüst
sayımda ihmal düzeyi). Ancak WP-M2 çıktıları ağın GİRDİSİNİ zenginleştirir: mevcut 25
özelliğe path-signature bileşenleri, BOCPD rejim-olasılıkları, rough-vol/Hurst, sıçrama
yoğunluğu, VRP-tipi türetilmiş özellikler eklenir → **girdi ~25 → ~60 özellik**. Daha
zengin girdi + çok-görevli başlıklar 6. transformer katmanını haklı çıkarır:

**rev.3 etkisi (2026-07-20):** WP-T metin/takvim özellikleri (+~10) ve çoklu zaman-ölçeği
füzyonu (saatlik/günlük/haftalık üç blok) girdiyi **~60 → ~96 token'a** çıkarır. Token
artışı transformer bloklarına parametre EKLEMEZ (hesap yükü ekler); parametre ekleyen iki
şey: büyüyen gömme tablosu ve **çok-ölçekli füzyon başlığı** (üç ölçek-özetini birleştiren
öğrenilmiş katman — rev.3'ün mimari gereği). WP-X (conformal/IG) ve WP-A/O parametre eklemez.

**QuantTransformer rev.3 mimarisi** (~96 giriş token'ı: 3 ölçek bloğu + metin/takvim):

| Bileşen | Konfigürasyon | Parametre hesabı | Yaklaşık |
|---|---|---|---:|
| Özellik gömme | her özellik → d=256 token (per-feature linear + tip gömmesi) | 96×(1×256+256) + 96×256 | ~74k |
| [CLS]+[SCALE] token + pozisyon | 97×256 + 3 ölçek işareti | | ~25k |
| Transformer blokları ×**6** | d=256, 4 başlık, FFN=1024, pre-LN | 6 × [4·(256²+256) + 2·(256·1024+~640) + 4·256] | ~4.74M |
| **Çok-ölçekli füzyon başlığı (rev.3)** | 3 ölçek-özeti concat (768) → 512 → 256 | 768·512+512 + 512·256+256 | ~525k |
| Çok-görevli başlıklar ×4 | yön(2) + büyüklük-kovası(5) + vol(1) + rejim(6); her biri 256→64→çıkış | ~4×(256·64+64+…) | ~70k |
| Kalibrasyon + transfer probe başlıkları (WP-L/WP-E) | sıcaklık + 2 probe head (256→64→2) | | ~35k |
| **TOPLAM (rev.3)** | | | **≈ 5.5M ✅ (~×1.450)** |

- Tahmin evrimi: rev.1 (25 özellik / 5 katman) ≈4.04M → rev.2 (60 özellik / 6 katman +
  probe) ≈4.9M → **rev.3 (96 token + füzyon başlığı) ≈5.5M**.
- Kesin sayı kodda `sum(p.numel() for p in model.parameters())` ile ölçülür;
  **test sözleşmesi taban olarak kalır: `assert param_count >= 3_777_000`** (×1000 garanti),
  teslimat raporunda gerçek sayı (~5.5M beklenir) ayrıca yazılır. Özellik/signature seçimi
  Faz 1'de kesinleşince sayı ±%10 oynayabilir; taban sözleşmesi etkilenmez.
- Ara basamak **deep_mlp**: (512,256,128,64) MLP ≈ 380k param (~×100) — merdiven sürekliliği.
- `recommend_tier` genişletmesi (muhafazakâr): `n ≥ 8.000·f/25` → deep_mlp;
  `n ≥ 75.000` → quant_transformer. Canlı log (~12k) neural önermez → dürüst raporlanır;
  Alpaca 10y (~97k) transformer eşiğini karşılar.
- Bilişsel "parametreler" ayrıca büyür: kavram 45→100+, ders 13→13+adaylar,
  anlamsal hafıza kayıtları (yeni).

---

## BÖLÜM E — İŞ PAKETLERİ (dosya ve fonksiyon düzeyi)

### WP-N · Sinir ağı çekirdeği — YENİ `nihai/ai/neural.py` (Sonnet-1)
**Amaç:** ×1000 parametre kapasitesini gerçek, ölçülebilir, kaydedilebilir bir modelle sağlamak.

Yapılacaklar:
1. `torch>=2.4` CPU wheel bağımlılığı (`requirements`/`pyproject` güncellenir).
2. **Fail-soft import sözleşmesi:** modül başında `try: import torch except ImportError:
   TORCH_AVAILABLE=False` — torch yoksa `neural` katmanları merdivende "unavailable"
   görünür, mlp'de durur, **hiçbir mevcut test/akış kırılmaz** (router LLM deseni).
3. `class DeepMLP` ve `class QuantTransformer` (nn.Module) + bunları `CapacityModel`
   API'sine BİREBİR uyduran `NeuralCapacity` sarmalayıcı:
   - `fit(X, y)` — AdamW, dropout 0.2, weight-decay 1e-4, grad-clip 1.0,
     early-stopping (%15 val split, patience 10), `torch.manual_seed(seed)` determinizm
   - `predict_proba(X)` — p(y=1), şekil (n,)
   - `param_count()` — gerçek `numel` toplamı
   - `meta()` / `save(path_base)` / `load(path_base)` — state_dict (`.pt`) + JSON meta,
     **atomic_io ile** (OneDrive dersi); yalnız verilen path_base'e yazar
4. **Çok-görevli kayıp:** ana görev yön (BCE) + yardımcı görevler (büyüklük-kovası CE,
   vol MSE, rejim CE) ağırlıklı toplam — az-veri rejiminde düzenlileştirici etki.
5. Belirsizlik çıktısı: MC-dropout (T=20 geçiş) ile p_win ± std → decision'a "güven" sinyali
   (üstüne WP-X conformal garanti katmanı gelir).
6. **Çoklu zaman-ölçeği füzyonu (rev.3):** girişe üç ufuk bloğu — saatlik (son 64 bar),
   günlük (son 32 gün) ve haftalık (son 26 hafta) özet özellikleri; her blok kendi tip
   gömmesiyle işaretlenir, transformer dikkat mekanizması ölçekler-arası ilişkiyi öğrenir
   (ör. "günlük trend yukarı ama saatlik momentum kırılıyor"). Özellik çıkarımı leak-free
   kuralla (yalnız karar barına kadar); ~60 özellik bütçesinin içinde ufuk-başına pay.

Testler (~45): param sayısı ≥3.777M; determinizm (aynı seed→aynı ağırlık/çıktı);
fail-soft (torch mock-yok); save/load gidiş-dönüş; erken-durma; MC-dropout şekilleri;
fit-öncesi RuntimeError; API parite (CapacityModel ile aynı sözleşme).

### WP-M · Derin matematik — `quant_math.py` v3 genişlemesi (Sonnet-2)
**Amaç:** "verileri matematiksel formüllerle işleme + derin/zor matematik" eksenini kapatmak.
Hepsi numpy/scipy kapalı-form, deterministik, savunmacı (kısa/bozuk veri → None, çökme yok).

| Fonksiyon | İçerik |
|---|---|
| `garch11_fit/forecast` | GARCH(1,1) MLE (scipy.optimize.minimize, kısıtlı), σ² öngörüsü; EWMA/RiskMetrics köprüsü |
| `evt_hill_tail_index`, `evt_gpd_var_es` | Hill tahmincisi + POT/GPD kuyruk VaR & Expected Shortfall (Cornish-Fisher'in kuyruk-ötesi eşi) |
| `copula_fit_gaussian/_t`, `tail_dependence` | Kopula uyumu + alt/üst kuyruk-bağımlılık katsayısı |
| `frac_diff_ffd` | López de Prado fractional differentiation (durağanlık↔hafıza dengesi) |
| `welch_psd`, `haar_dwt_vol` | Spektral güç yoğunluğu + çok-ölçek dalgacık vol ayrışımı |
| `almgren_chriss_cost` | Optimal yürütme maliyet eğrisi (YALNIZ rapor; execution'a bağlanmaz) |
| `permutation_entropy`, `sample_entropy`, `mutual_info` | Bilgi-kuramsal desen/özellik ölçüleri |
| `coskewness`, `cokurtosis`, `omega_ratio` | Yüksek-moment risk matrisleri + Omega oranı |

NOT: HMM bilinçli dışarıda (Tur-14 kararı: tek-rejim penceresi). Testler (~55): bilinen-değer
(sentetik GARCH parametre geri-bulma, GPD kuyruk analitik karşılaştırma, FFD birim testleri,
entropi sıralama özellikleri), savunmacı-girdi, determinizm.

### WP-D · Karar v3 — `decision.py` genişlemesi (Sonnet-3)
**Amaç:** karar vermeyi çok-kaynaklı, kalibrasyona duyarlı, fayda-tabanlı hale getirmek.

1. **Hiyerarşik Bayes birleşimi:** kaynaklar = Beta prior (learning) ⊕ neural p_win±std
   (varsa, gölge) ⊕ rejim posterior'u. Kaynak ağırlığı = geçmiş Brier'ine göre
   (kötü kalibre kaynak otomatik cezalanır; ağırlıklar gerekçe zincirinde yazılır).
2. **Fayda katmanı:** E[fayda] = posterior·kazanç − (1−posterior)·kayıp − λ·CVaR₉₅ cezası
   (λ konfig, varsayılan muhafazakâr). ABSTAIN ⟺ E[fayda] ≤ 0.
3. **ÇİFT KAPI (sıkılaşma, gevşeme değil):** mevcut "prior CI tabanı > 0.5" kuralı AYNEN
   kalır; yön çıktısı için **hem** CI kapısı **hem** fayda kapısı geçilmeli.
4. Gerekçe zinciri: kaynak-başına katkı satırı + kalibrasyon notu + hangi kapının
   neden geçilemediği. Mod sözleşmesi aynen: `off|shadow`, enforce KODDA YOK.

Testler (~30): kapı kombinasyonları (CI✓fayda✗ → ABSTAIN vb.), Brier-ağırlık monotonluğu,
neural-yokken degrade (v2 davranışına denk), gerekçe metni sözleşmesi.

### WP-H · Hafıza v3 — `episodic.py` genişlemesi + YENİ `nihai/ai/consolidate.py` (Sonnet-3)
**Amaç:** hatırlamayı vektörel, hafızayı kendi kendini örgütleyen hale getirmek.

1. **Vektörel geri çağırma:** epizot vektörü = z-norm özellikler ⊕ rejim one-hot ⊕ sonuç;
   kosinüs benzerliği (mevcut RMS korunur, `metric="rms"|"cosine"` parametresi).
2. **Konsolidasyon** (`consolidate.py`, gece görevi, salt-append): benzer epizotlar
   küme-özetine indirgenir → `data/ai_semantic_memory.jsonl` (atomic_io):
   `{"pattern": "down_high rejiminde event_breakout", "n": 42, "isabet": 0.44, ...}`.
   Brain HATIRLA adımı önce anlamsal özete, sonra ham epizota bakar.
3. **Unutma eğrisi:** skor = f(yaş, erişim sıklığı, bilgi değeri); 5MB/2000-satır koruması
   korunur ama kırpma artık düşük-skorludan başlar (rastgele değil).
4. **Ders-adayı üretici:** drift/kalibrasyon olayından otomatik LESSON adayı
   (`status=candidate`); KATALOĞA TERFİ YALNIZ İNSAN ONAYIYLA (dürüstlük guard'ı).

Testler (~30): kosinüs vs RMS tutarlılığı, konsolidasyon determinizmi, unutma sıralaması,
aday-ders asla otomatik terfi etmez, atomik yazım.

### WP-L · Öğrenme v3 — `learning.py` genişlemesi + YENİ `research/train_neural.py` (Sonnet-1)
**Amaç:** öğrenmeyi tek-seferlik rapordan sürekli (ama kanıt-kapılı) döngüye çevirmek.

1. **Replay buffer:** `live_features_log.csv` ⋈ çözülen tahminler → eğitim örneği akışı
   (`build_live_dataset` altyapısı yeniden kullanılır; leak-free/PIT-temiz miras).
2. **Üç aşamalı eğitim reçetesi** (Bölüm F'te detay): sentetik müfredat → Alpaca 97k →
   canlı ince-ayar.
3. **Kalibrasyon:** isotonic/Platt YALNIZ OOS diliminde fit; Brier + güven-kovası raporu
   `learning_report`'a eklenir.
4. **Drift-tetikli yeniden-eğitim ADAYI:** Page-Hinkley tetiklenirse yeni aday model
   eğitilir + raporlanır; **otomatik promotion ASLA** (Bölüm G kapıları).
5. `research/train_neural.py`: purged+embargolu K-fold CV ile neural katman ölçümü —
   `purged_cv.py` desenine birebir (fold-başı kapı simülasyonu + DSR raporu).
6. **Hiperparametre disiplini (rev.3):** arama YALNIZ bütçe-sınırlı rastgele tarama
   (≤30 deneme) + **iç-içe (nested) purged CV** — hiperparametre seçimi dış-OOS dilimini
   ASLA görmez; denenen konfigürasyon sayısı raporlanır ve DSR hesabına girer (çoklu-test
   dürüstlüğü). Grid/BO çerçevesi bilinçli dışarıda (aşırı-arama = overfit daveti).

Testler (~35): replay join doğruluğu (leak testi: karar-barı sonrası kolon yok),
kalibrasyon yalnız-OOS, drift→aday-ama-promotion-yok, train_neural CLI smoke (mini veri).

### WP-K · Bilgi v3 — `knowledge.py` genişlemesi (Sonnet-2)
**Amaç:** alan bilgisini 45 → **100+ kavrama** çıkarmak; her kavramda tanım + formül köprüsü
(yeni quant_math fonksiyonları dahil) + tuzaklar + ACI DERS bağları.

Yeni kavram grupları: volatilite ailesi (GARCH/EWMA/gerçekleşen), kuyruk riski (EVT/GPD/ES),
bağımlılık (kopula/kuyruk-bağımlılık), mikroyapı (spread, kayma, fiyat etkisi, PIN),
makro/ekonomi (getiri eğrisi, enflasyon rejimi, likidite döngüsü, faiz-hisse kanalı),
opsiyon yüzeyi (smile/skew/vade yapısı), yürütme (Almgren-Chriss, katılım oranı),
istatistiksel öğrenme tuzakları (overfit/leakage/survivorship — mevcut derslerle bağlı).
YENİ `curriculum()`: önkoşul grafiğiyle sıralı öğrenme yolu raporu.

Testler (~25): kavram sayısı ≥100, her kavramda zorunlu alanlar, formül köprüleri gerçek
fonksiyonlara çözünür, curriculum çevrimsiz (DAG).

### WP-M2 · EKSTRA İLERİ MATEMATİK — `quant_math.py` ikinci dalga + YENİ `nihai/ai/advanced_math.py` (Sonnet-2)
**Amaç (rev.2):** WP-M'in üstüne, piyasa modellemesinin araştırma-düzeyi matematiğini eklemek.
Hacim büyüdüğü için ayrı modül: `advanced_math.py` (quant_math ile aynı savunmacı sözleşme).

| Grup | Fonksiyonlar | Ne işe yarar |
|---|---|---|
| Sıçrama-difüzyon | `merton_jump_fit` (MLE), `jump_detect_bipower` (bipower varyasyon sıçrama testi) | Fiyat sıçramalarını ayrıştırma; haber-şoku tespiti |
| Stokastik volatilite | `heston_charfn`, `carr_madan_fft_price` (Avrupa opsiyon FFT fiyatlama), `heston_calibrate` (kısıtlı en-küçük-kareler) | Opsiyon yüzeyinden piyasa beklentisi okuma |
| Amerikan opsiyon | `american_crank_nicolson` (PDE), `longstaff_schwartz_lsmc` (Monte Carlo regresyon) | Erken kullanım değeri; gerçek fiyat/teorik sapma raporu |
| Olay kümelenmesi | `hawkes_fit` (üstel çekirdek MLE), `hawkes_intensity` | Hacim/haber patlamalarının öz-tetikleme yapısı |
| Rejim kırılması | `bocpd` (Bayesian Online Changepoint Detection, Student-t öngörü) | Rejim değişimini GECİKMESİZ olasılıkla izleme (Page-Hinkley'in olasılıksal üstü) |
| Durum-uzay | `kalman_regime_switching` (2-rejim), `particle_filter_light` (bootstrap PF, ≤1000 parçacık) | Gizli durum takibi; dinamik beta'nın genelleştirilmesi |
| Rough volatility | `realized_vol_hurst` (gerçekleşen vol Hurst'ü), `rough_vol_forecast` | Vol'ün kaba-yol (H<0.5) davranışından öngörü |
| Path signatures | `signature_level2/3` (yol imzaları) | Model-bağımsız zaman-serisi özellik üretimi (neural girişine ek) |
| Motif/anomali | `matrix_profile_light` (STOMP basitleştirmesi) | Tekrar eden desen + anomali keşfi |
| Numerik altyapı | `qp_solve` (scipy SLSQP/quadprog sarmalı), `adaptive_quad`, `logsumexp_stable`, `fft_convolve` | Portföy optimizasyonu + sayısal kararlılık temeli |

Testler (~50): analitik bilinen-değer (Heston→BS limiti, Merton λ→0 limiti, CN-PDE vs BS,
LSMC vs CN paritesi), sentetik geri-bulma (Hawkes/BOCPD/Hurst), determinizm, savunmacı-girdi.

### WP-C · KOD YAZMA/ÇALIŞTIRMA LABORATUVARI — YENİ `nihai/ai/lab.py` (Sonnet-4)
**Amaç (rev.2):** AI'ın "uygun dilde kod yazıp çalıştırıp sonucu yorumlayıp karara
bağlayabilmesi" yetkinliği.

**1. Dil kararı — gerekçeli analiz (bu planın kararı):**
- Piyasa gerçeği: C++ HFT/execution-latency dünyasının dilidir (mikrosaniye). Bu sistem
  **sinyal-only, saatlik/günlük bar** çalışır → darboğaz latency DEĞİL, doğruluk ve
  araştırma hızıdır. Quant araştırmanın endüstri dili Python'dur (numpy/scipy/sklearn/torch).
- **Karar: Python ana dil KALIR; performans için KADEMELİ HIZLANDIRMA MERDİVENİ:**
  1. numpy vektörizasyon (varsayılan) → 2. **Numba JIT** (`@njit`; MC/backtest sıcak
  döngüleri, tek dekoratör, dağıtım riski yok) → 3. **Cython/pybind11 C++ çekirdeği**
  (YALNIZ profil kanıtı "Numba yetmedi" derse; ör. 10⁶+ yol MC, Hawkes MLE iç döngüsü).
- Tam C++ geçişi REDDEDİLDİ (gerekçe: sıfır latency-kazancı ihtiyacı, MSVC derleme
  zinciri + bakım maliyeti, 938-testlik Python ekosistemi). Merdiven adım 3 için iskelet
  (`native/` dizini + pybind11 CMake şablonu + fallback-saf-Python) hazırlanır ama
  aktivasyonu profil kanıtına bağlıdır.

**2. Analiz döngüsü (LLM-destekli, fail-soft):**
```
soru/hipotez → [router: LLM Python analiz kodu üretir] → sandbox'ta çalıştır
            → sonuç (sayı/tablo) → [router: LLM yorumlar] → gerekçe zincirine eklenir
```
- ANTHROPIC_API_KEY yoksa: LLM adımları devre dışı, **şablon-tabanlı deterministik
  deneyler** çalışır (aşağıda) — lab hiç LLM'siz de değerlidir.

**3. Sandbox güvenliği (anayasa uzantısı):**
- Kısıtlı namespace (yalnız numpy/scipy/pandas + quant_math/advanced_math); `open` yok,
  `__import__` beyaz-listeli; **data/ salt-okunur**, üretim dosyalarına yazım İMKÂNSIZ
- Süre limiti (varsayılan 30 sn) + bellek tavanı + ağ erişimi yok
- Her koşu JSONL'e loglanır (kod + çıktı + yorum) — denetlenebilirlik

**4. Hazır deney şablonları** (`lab.run_experiment(name, ...)`):
`vol_model_karsilastirma` (4 tahminci + GARCH OOS yarışı), `korelasyon_rejimi`
(Ledoit-Wolf pencere analizi), `olay_etudu` (CAR/AAR şablonu), `dagilim_testi`
(normallik/kuyruk/sıçrama raporu), `ozellik_tarama` (MI + permütasyon önemi).

Testler (~35): sandbox kaçış denemeleri (dosya yazma/import/ağ → hepsi engellenir),
süre limiti, şablonların deterministik çıktıları, LLM-yok fail-soft, log bütünlüğü.

### WP-P · PORTFÖY YÖNETİMİ — YENİ `nihai/ai/portfolio.py` (Sonnet-4, GÖLGE)
**Amaç (rev.2):** tekil-sinyal dünyasından portföy-düzeyi düşünmeye geçiş. TÜM çıktılar
öneri-raporu (JSONL, gölge); execution'a/emir sistemine BAĞLANMAZ (sistem zaten sinyal-only).

| Yetenek | İçerik |
|---|---|
| Kovaryans tahmini | Ledoit-Wolf (mevcut köprü) + EWMA kovaryans + tek-faktör (piyasa) modeli; kondisyon-sayısı raporu |
| Klasik optimizasyon | min-varyans, kısıtlı max-Sharpe (`qp_solve` ile; kısa-yasak/uzun-yalnız seçeneği) |
| **HRP** | Hierarchical Risk Parity (López de Prado): korelasyon-kümeleme + özyinelemeli risk paylaşımı — kovaryans-ters-çevirme kırılganlığına dayanıklı |
| Risk parity | ERC (eşit risk katkısı, Newton yinelemesi) |
| **Black-Litterman** | Piyasa-denge prior + "görüşler" = decision v3 gölge posterior'ları (güven = CI genişliği) → karma beklenti; AI'ın kanaatlerini portföye disiplinli taşıma yolu |
| Çok-varlık Kelly | Kovaryans-ayarlı fractional Kelly vektörü (tekil-Kelly'nin genellemesi); tavan kısıtlı |
| CVaR optimizasyonu | Rockafellar-Uryasev LP formülasyonu (`scipy.optimize.linprog`) — kuyruk-riski minimize eden ağırlıklar |
| Kısıtlar | pozisyon tavanı, **net-directional cap** (mevcut WIP ile uyumlu), sektör/tekil maruziyet, ciro cezası + işlem maliyeti farkındalığı (bps parametrik) |
| Drawdown kontrolü | hedef-volatilite ölçekleme + çizgi-aşımında risk-azalt önerisi (kill-switch köprüsü: yalnız RAPOR) |
| Stres/senaryo | tarihsel senaryo replay (2008/2020/2022 pencereleri) + MC stres (WP-N/M altyapısı); component-VaR risk ayrıştırma |
| Atribüsyon | strateji/sembol bazlı katkı ayrıştırma (Brinson-benzeri) + mevcut tearsheet'e portföy sekmesi |

Akış: `portfolio.propose(candidates, closes_panel, constraints)` → ağırlık vektörü +
yöntem-karşılaştırma tablosu (HRP vs BL vs CVaR aynı raporda — tek yönteme körlük yok) +
gerekçe. `brain --brief`'e opsiyonel portföy bölümü.

Testler (~45): HRP bilinen-küçük-örnek, ERC risk-katkı eşitliği, BL limit durumları
(görüş-yok → piyasa ağırlığı), CVaR-LP küçük analitik örnek, kısıt ihlali imkânsızlığı,
ciro cezası monotonluğu, determinizm.

### WP-E · KENAR-KEŞİF PROGRAMI — YENİ `research/edge_lab/` (tasarım: Fable 5, kodlama: Sonnet-4)
**Amaç (rev.2):** "EDGE YOK" ölçülen yerde kenar ARAMAK için sistematik program.

**Dürüstlük çerçevesi (pazarlık edilemez):** Mevcut ölçümler (§13.5/§16/§19/§20/§21/§22)
"BU veri + BU sinyal ailelerinde kenar yok" dedi — "hiçbir yerde kenar yok" demedi.
Kenar GARANTİ EDİLEMEZ; edilebilecek şey aramayı sistematikleştirmek, aday-uzayını
büyütmek ve her adayı aynı sert kapılardan geçirmektir. ACI DERS #5 teşhisi geçerli:
darboğaz model değil, **veri ve sinyal-ailesi çeşitliliği**. Program bu ikisine saldırır:

**1. Veri genişletme (en yüksek beklenen değer, öncelik sırasıyla):**
| Kaynak | Ne getirir | Durum/maliyet |
|---|---|---|
| Sharadar PIT fundamentals | Survivorship-dertsiz temel veriler → PEAD/değer/kalite aileleri GERÇEK testi | Ücretli; karar kullanıcıda (bilinen açık madde) |
| Opsiyon zinciri (IV) | IV skew, put/call oranı, **VRP** (volatilite risk primi), vade yapısı | Alpaca/ücretsiz kaynak araştırılacak |
| FRED makro | Rejim-koşullu aktivasyon değişkenleri (faiz, kredi spreadi, VIX yapısı) | Ücretsiz; anahtar açık madde |
| Short interest + insider | Kalabalık/bilgili-taraf sinyalleri | Ücretsiz gecikmeli sürümler var |
| Kazanç çağrısı metinleri | LLM-tabanlı ton/da belirsizlik skoru (router altyapısı hazır) | Ücretsiz transkript kaynakları |

**2. Test edilmemiş sinyal aileleri (aday sıralaması = literatür desteği × veri hazırlığı):**
- **VRP** (IV − gerçekleşen vol satımı yönlü değil, zamanlama sinyali olarak) — literatürde en dayanıklı primlerden
- Endeks rebalans/akış olayları (S&P ekleme-çıkarma, ay-sonu akışları)
- Overnight/intraday getiri ayrışması (saatlik veri MEVCUT — hemen test edilebilir)
- Lead-lag ağları (sektör-içi büyük→küçük gecikmeli yayılım; saatlik veriyle)
- Buyback/split/insider olay etütleri (lab.olay_etudu şablonuyla)
- 13F kurumsal kümelenme (çeyreklik)
- Mevsimsellik ailesi (yalnız çoklu-test-düzeltmeli: DSR zorunlu)

**3. Yöntem iyileştirmeleri:**
- **CPCV** (kombinatoryal purged CV) — purged_cv'nin üstü: çok daha fazla OOS yolu → PBO'nun doğal girdisi
- Rejim-koşullu aktivasyon: sinyal ailesi × BOCPD rejim-olasılığı etkileşimi (sinyal "her zaman" değil "şu rejimde" çalışıyor olabilir — bugüne dek koşulsuz test edildi)
- Triple-barrier etiketlemenin bariyer-taraması (mevcut horizon-bazlı etiketin genellemesi)
- Ensemble meta-strateji: mutabakat harness'ının (agreement.py) N-sinyal genellemesi
- **Neural transfer:** WP-N ön-eğitimli gövde + küçük "probe head" → az-veri aday ailelerde hızlı tarama (tam eğitim maliyeti olmadan sinyal-vaadi ölçümü)

**4. Canlı kanıt hattı:** prediction_tracker paper-trade birikimi (akıyor) — her yeni aday
aile gölge-modda canlıya bağlanır; aylar içinde PIT-dertsiz canlı OOS birikir.

**5. Altyapı teslimatı:** `research/edge_lab/harness.py` — tek komutla aday-aile testi:
veri → özellik → purged CV/CPCV → çift-bölme kapı → DSR/PBO → **exec-lag 1** → PIT →
tek-sayfa verdikt raporu (EDGE_FINDINGS formatında). + `edge_lab/BACKLOG.md`: aday
sıralaması ve durum takibi. **Hiçbir aday kapıları geçmeden "kenar" ilan edilemez.**

### WP-T · METİN/HABER ZEKÂSI + TAKVİM FARKINDALIĞI — YENİ `nihai/ai/text_intel.py` (Sonnet-5)
**Amaç (rev.3):** AI'ın sayısal olmayan bilgiyi (haber, kazanç metni, makro takvim) okuyup
yapılandırılmış sinyale çevirmesi. İKİ KADEMELİ, LLM'siz de çalışır:

1. **Deterministik taban (her zaman çalışır):** Loughran-McDonald finans sözlüğü tabanlı
   ton skoru (pozitif/negatif/belirsizlik/dava-riski kelime sınıfları — genel sentiment
   sözlükleri finans metninde YANILIR, LM standarttır), başlık-yoğunluğu (sembol-başına
   haber sayısı z-skoru), yenilik skoru (önceki metinlere kosinüs benzerliği — "aynı haberin
   3. tekrarı" ≠ yeni bilgi).
2. **LLM katmanı (ANTHROPIC_API_KEY varsa, router üstünden):** kazanç çağrısı/haber özeti →
   yapılandırılmış çıktı (ton, guidance-yönü, belirsizlik, tek-cümle gerekçe); dürüstlük
   guard'ı: LLM çıktısı ASLA tek başına özellik olmaz, deterministik skorla YAN YANA raporlanır.
3. **Takvim farkındalığı** (`market_calendar` alt-modülü): FOMC/CPI/NFP tarihleri (statik
   yıllık tablo + elle güncelleme), sembol kazanç-tarihi yakınlığı (MainHub/nasdaq-earnings
   köprüsü MEVCUT — Tur-14 P0'da eklendi), opsiyon vade (OPEX) haftası, yıl-sonu/çeyrek-sonu
   bayrakları → brain'e "bugün hangi olay rejimindeyiz" bağlamı + ağa 3-4 takvim özelliği.
4. Çıktılar: `data/ai_text_scores.jsonl` (atomic, gölge) + brain HATIRLA/ALGILA adımına
   bağlam satırı + (WP-E ile) kazanç-çağrısı ton ailesi aday-sinyal harness'ı.

Testler (~35): LM sözlük bilinen-cümle skorları, yenilik-skoru monotonluğu, LLM-yok
fail-soft, takvim yakınlık hesabı (tz-güvenli), JSONL atomiklik.

### WP-X · AÇIKLANABİLİRLİK + BELİRSİZLİK GARANTİSİ — YENİ `nihai/ai/explain.py` (Sonnet-5)
**Amaç (rev.3):** 5.5M parametrelik ağ kara kutu KALMASIN; her gölge kararın "neden"i ve
"ne kadar eminiz"i istatistiksel garanti ile raporlansın.

| Yetenek | Yöntem | Çıktı |
|---|---|---|
| Özellik atıfları | Permütasyon önemi (model-bağımsız, mevcut purged_cv deseni) + **integrated gradients** (torch varsa; baseline=nötr özellik vektörü) | "Bu kararda en çok X, Y, Z etkiledi" ilk-5 listesi |
| Dikkat haritası | Transformer attention rollout ([CLS]→özellik toplam ağırlığı) | Özellik-grubu ısı özeti (fiyat/vol/metin/takvim grupları) |
| **Conformal prediction** | Split-conformal: kalibrasyon OOS dilimiyle p_win için **dağılım-varsayımsız garanti aralığı** (%90 kapsam garantili) — MC-dropout'un sezgisel std'sinin üstüne İSPATLI katman | "p_win 0.54, %90 garanti aralığı [0.41, 0.66] → aralık 0.5'i kapsıyor = kararsız" |
| Karar açıklama kartı | Yukarıdakiler + decision v3 kaynak-katkıları tek şablonda | Her gölge karara ekli "açıklama kartı" (JSONL + brif metni) |
| Küresel model raporu | Eğitim sonrası: özellik önem sıralaması + kısmi bağımlılık (PDP, ilk-10 özellik) | `train_neural` çıktısına model-davranış bölümü |

Kural: decision v3'ün ABSTAIN mantığına üçüncü kapı eklenir — **conformal aralık 0.5'i
kapsıyorsa yön çıktısı verilmez** (CI + fayda + conformal = ÜÇLÜ kapı; yine sıkılaşma).

Testler (~30): conformal kapsam simülasyonu (sentetikte nominal %90 tutuyor mu),
IG tamlık aksiyomu (atıf toplamı ≈ çıktı farkı), permütasyon determinizmi, kart şablonu.

### WP-A · SORU-CEVAP, AJANLIK VE ÖZ-DEĞERLENDİRME — YENİ `nihai/ai/agent.py` + `selfcheck.py` (Sonnet-5)
**Amaç (rev.3):** AI'ı "çağrılan fonksiyon"dan "muhatap alınabilen, kendini sınayan sistem"e
taşımak.

1. **`brain --ask "soru"` (Türkçe soru-cevap, RAG):** soru → knowledge(100+ kavram) +
   episodic/anlamsal hafıza + EDGE_FINDINGS/docs araması + gerekirse lab deney koşusu →
   kaynaklı cevap ("kaynak: knowledge/vrp, EDGE_FINDINGS §22"). LLM varsa akıcılaştırma,
   yoksa şablon-cevap (fail-soft). Cevaplar da reasoner kapanışıyla biter.
2. **Hipotez kuyruğu (çok-adımlı ajanlık):** `agent.plan(hedef)` → hedefi alt-görevlere
   böler (ör. "VRP ailesini değerlendir" → veri-kontrol → lab ön-analiz → edge_lab harness →
   verdikt-özeti), her adımın çıktısı sonrakine akar, ilerleme `data/ai_agenda.jsonl`'de
   (atomic; kaldığı yerden devam edebilir). SINIR: kuyruk yalnız analiz/rapor görevleri
   üretebilir — üretim dosyası yazan görev tipi YOK (anayasa).
3. **Öz-değerlendirme sınavı (`selfcheck.py`):** üç bölümlü periyodik benchmark:
   (a) bilgi sınavı — knowledge'dan üretilen bilinen-cevaplı ~50 soru (formül köprüleri
   sayısal doğrulanır); (b) kalibrasyon sınavı — son N gölge kararın Brier/isabet trendi;
   (c) tarihsel replay sınavı — bilinen dönemlerde (ör. 2022 ayı piyasası) brain kararları
   makul mü (ABSTAIN oranı, rejim tespiti doğruluğu). Çıktı: karne (`--brief`e bölüm) +
   geriye dönük trend. AI kendi bozulmasını KENDİ raporlar.
4. **Karşı-olgusal replay motoru:** çözülen tahminler üstünde "decision v3 şu eşikle/şu
   kaynak-ağırlığıyla çalışsaydı gölge-PnL ne olurdu" ızgara taraması — mevcut gate_sim
   deseninin karar-motoru genellemesi. SINIR: sonuçlar SADECE rapor; eşik otomatik
   değişmez (değişiklik önerisi insan onayına sunulur).

Testler (~40): RAG kaynak-atfı doğruluğu, ajanda kalıcılık/devam, sınav determinizmi,
replay ızgarasının gate_sim ile tutarlılığı, yazma-yasağı sınırları.

### WP-O · VERİ SAĞLIĞI + MODEL YAŞAM DÖNGÜSÜ + OPERASYON — YENİ `nihai/ai/data_health.py` + `registry.py` (Sonnet-4'e ek)
**Amaç (rev.3):** AI'ın girdisini, modellerini ve günlük işleyişini kurumsal disipline almak.

1. **Veri sağlığı kapısı (`data_health.py`):** brain/eğitim girdisi ÖNCE buradan geçer:
   eksik-bar tespiti (takvime karşı), donmuş-fiyat (n-bar aynı değer), sıçrama-outlier
   (bipower testi WP-M2'den — gerçek sıçrama mı veri hatası mı ayrımı split-faktör
   kontrolüyle), hacim anomalisi, NaN/tip denetimi → sağlık skoru + karantina listesi.
   Skor eşik-altıysa brain o sembol için otomatik ABSTAIN + neden satırı ("veri güvenilmez").
2. **Model registry (`registry.py`):** her eğitilen model → içerik-hash + **model kartı**
   (JSON: veri dönemi/boyutu, hiperparametreler, OOS metrikleri, kalibrasyon, DSR, eğitim
   ortamı, üst-model hash'i) + `data/ai_models/` altında sürümlü saklama (atomic).
   **Gölge A/B çerçevesi:** registry'deki iki model aynı canlı akışta paralel gölge-tahmin
   üretir, learning ikisini ayrı izler → "aday model eskisinden İSTATİSTİKSEL olarak iyi mi"
   sorusuna canlı cevap (promotion kapılarının canlı ayağı; yine öneri-only).
3. **Otonom operasyon döngüsü (zamanlanmış, mevcut watchdog desenine ek görevler):**
   gece: konsolidasyon (WP-H) + veri-sağlık taraması; sabah: takvim brifi (WP-T) + günlük
   brif; hafta-sonu: öz-değerlendirme sınavı (WP-A) + edge_lab BACKLOG ilerletme önerisi +
   registry bakım raporu. Hepsi mevcut `tur13-servis-devri` interactive-user modeline uyar
   (SYSTEM-zombi dersi); görev tanımları hazırlanır, KAYIT kullanıcı onayıyla.
4. **Web/Telegram yüzeyi:** mevcut web paneline salt-okunur "Beyin" sekmesi (son kararlar +
   açıklama kartları + karne + drift) ve `daily_summary`'ye karne satırı — mevcut fail-soft
   desenle.

Testler (~35): sağlık kapısı senaryoları (split vs hata ayrımı dahil), karantina→ABSTAIN
zinciri, registry hash/kart bütünlüğü, A/B izolasyonu (iki model birbirinin durumunu bozamaz),
zamanlanmış görev kuru-koşusu.

### WP-I · Entegrasyon — orkestratör Fable 5, TEK THREAD
1. `models.py`: merdivene `deep_mlp` + `quant_transformer` (neural.NeuralCapacity sarımı);
   `PARAM_TIERS` genişler; `param_ladder_report` yeni satırlar; torch-yok degrade satırı;
   `recommend_tier` yeni eşikler.
2. `brain.py` v3 döngüsü: +ÖNGÖR (neural gölge p_win, varsa) +ÖZ-ELEŞTİR (karar
   gerekçesindeki en zayıf halka: hangi kaynak en kötü kalibre / hangi veri eksik —
   deterministik kural, LLM'siz; router varsa opsiyonel akıcılaştırma) + **PORTFÖY bakışı**
   (tekil karar sonrası: "bu sembol mevcut gölge-portföyde ne değiştirir" — WP-P köprüsü)
   + **LAB köprüsü** (öz-eleştiri bir veri-eksiği tespit ederse lab deney-önerisi üretir).
   **rev.3 köprüleri:** ALGILA başında veri-sağlık kapısı (karantina→ABSTAIN); ALGILA'ya
   metin/takvim bağlamı (WP-T); KARAR'a conformal 3. kapı (WP-X); her karara açıklama
   kartı; `--ask` alt-komutu (WP-A) ve karne bölümü `--brief`e.
3. `--demo`/`--brief` genişletme; `daily_summary --brain` uyum kontrolü (fail-soft korunur).
4. Dokümanlar: `AI_V3_DERIN_BEYIN_SPEC.md` (ajan brifleri buradan) + teslimat raporu.
5. Tüm ajan çıktılarının anayasa denetimi + mypy/ruff/pytest entegrasyonu + çakışma çözümü.

---

## BÖLÜM F — VERİ VE EĞİTİM STRATEJİSİ (dürüstlük çerçevesi)

**Sorun:** 5.5M parametre / ~12k canlı örneklem ≈ 460 param/örnek → tek başına canlı
veriyle eğitim **overfit garantisi**. Çözüm üç aşama:

| Aşama | Veri | Amaç | Koruma |
|---|---|---|---|
| 1. Sentetik müfredat ön-eğitimi | MC üretimi (≤500k örnek): GBM + blok-bootstrap + OU + rejim karışımları, etiketli yön/vol/rejim | Ağa finans zaman-serisi "gramerini" öğretmek | Deterministik seed; gerçek veri sızıntısı imkânsız |
| 2. Tarihsel ince-ayar | Alpaca 10y saatlik PIT dataset (~97k işlem, survivorship-temiz) | Gerçek dağılıma adaptasyon | purged+embargolu CV ile ölçüm; bilinen sonuç KENAR YOK (§21) — neural sonucu ne çıkarsa o raporlanır |
| 3. Canlı adaptasyon | replay buffer (canlı log, büyüdükçe) | Güncel rejime hafif uyum | Düşük lr, early-stop, drift-tetikli yalnız-ADAY |

**Rapor dili sözleşmesi:** her çıktıda "kapasite hazır ve ölçüldü" — "kenar bulundu" ASLA
(reasoner kapanış cümlesi her brife eklenmeye devam eder).

---

## BÖLÜM G — KANIT KAPILARI (değişmedi; v3 hiçbirini gevşetmez)

Herhangi bir katmanın üretim önerisi için SIRAYLA: purged+embargolu K-fold OOS →
time+symbol çift-bölme kapı simülasyonu → DSR/PBO → **exec-lag 1 duyarlılığı (ACI DERS #7)**
→ `live_retrain --promote` insan-onaylı yol. Promotion bu planın kapsamı DIŞINDA.

---

## BÖLÜM H — MODEL YÖNLENDİRME VE YÜRÜTME SIRASI

**Roller:**

| Rol | Model | Sorumluluk |
|---|---|---|
| Orkestratör | **Fable 5** | Plan, SPEC, ajan brifleri, **WP-E tasarımı**, anayasa/sandbox denetimi, entegrasyon, mypy/ruff/test, dokümantasyon |
| Sonnet-1 | Sonnet | WP-N (neural.py) + WP-L (learning v3 + train_neural) |
| Sonnet-2 | Sonnet | WP-M (quant_math v3) + **WP-M2 (advanced_math.py)** + WP-K (knowledge v3) |
| Sonnet-3 | Sonnet | WP-D (decision v3) + WP-H (episodic v3 + consolidate) |
| **Sonnet-4** | Sonnet | **WP-C (lab.py) + WP-P (portfolio.py) + WP-E harness + WP-O (data_health/registry)** |
| **Sonnet-5** | Sonnet | **WP-T (text_intel) + WP-X (explain) + WP-A (agent/selfcheck)** — rev.3 paketi |

Partisyonlar örtüşmesiz (Tur-9/15 dersi); ajan oturum-limitine takılırsa orkestratör
tamamlar (Tur-15 WP-B'de kanıtlanmış geri-düşüş). Yük dengelemesi: Faz 1 İKİ DALGA koşulur —
Dalga-1 çekirdek (WP-N/M/D/H/L/K: v3 beyni ayağa kaldırır), Dalga-2 rev.2+rev.3 katmanları
(WP-M2/C/P/E/T/X/A/O: çekirdek API'leri sabitlendikten sonra üstüne inşa) — böylece
Dalga-2 ajanları kararlı arayüzlere karşı kodlar, çakışma penceresi küçülür.

**Fazlar (onay sonrası):**

- **Faz 0 — Hazırlık (orkestratör):** hedef kopya teyidi (toplu vs ana) → paralel
  oturum/servis/yazım-zamanı kontrolü (Tur-8 dersi) → git temiz-commit noktası →
  `pip install torch` (CPU index) + import smoke → SPEC dosyası yazımı (ajan brifleri).
- **Faz 1 — Paralel kodlama (İKİ DALGA, rev.3):** Dalga-1 çekirdek (WP-N/M/D/H/L/K,
  3 ajan) → orkestratör ara-entegrasyonu (API'ler sabitlenir) → Dalga-2 katmanlar
  (WP-M2/C/P/E/T/X/A/O, Sonnet-2/4/5). Her ajan kendi test dosyalarını da yazar;
  ajanlar birbirinin dosyasına DOKUNMAZ.
- **Faz 2 — Entegrasyon (tek-thread, orkestratör):** models.py + brain.py bağlama
  (portföy/lab köprüleri dahil), anayasa + **sandbox-güvenlik denetimi** (kaçış testleri
  bizzat koşulur), mypy/ruff, tam pytest, çakışma/deprecation düzeltmeleri.
- **Faz 3 — Gerçek-veri dürüstlük koşusu:** `learning.update_from_predictions` +
  `brain --demo/--brief` + (veri hazırsa) `train_neural` mini-ölçüm + **portföy önerisi
  demo koşusu** (mevcut watchlist kapanışlarıyla HRP/BL/CVaR karşılaştırma raporu) +
  **edge_lab ilk aday koşusu** (overnight/intraday ayrışması — saatlik veri hazır olduğu
  için ilk aday bu); sonuç NE ÇIKARSA o raporlanır.
- **Faz 4 — Teslimat:** `AI_V3_DERIN_BEYIN_RAPORU.md` + `edge_lab/BACKLOG.md` başlangıç
  sıralaması + commit + memory güncellemesi.

**Geri alma (rollback):** Faz 0'daki temiz-commit noktasına `git reset`; torch kurulumu
venv-lokal olduğundan sistem etkisi yok; fail-soft sayesinde neural katman silinse bile
v2 davranışı aynen çalışır.

---

## BÖLÜM I — KALİTE HEDEFLERİ VE KABUL KRİTERLERİ

1. `NeuralCapacity("quant_transformer").param_count() ≥ 3.777.000` — **testle kanıtlı ×1000**.
2. Yeni test: +430–520 (dağılım: WP-N ~45, WP-M ~55, WP-M2 ~50, WP-D ~30, WP-H ~30,
   WP-L ~35, WP-K ~25, WP-C ~35, WP-P ~45, WP-E harness ~20, **WP-T ~35, WP-X ~30,
   WP-A ~40, WP-O ~35** [rev.3]); hedef toplam ~1.400+; TÜMÜ yeşil.
   Yeni kabul kriterleri (rev.3): conformal kapsam sentetikte nominali tutar; veri-sağlık
   karantinası → ABSTAIN zinciri uçtan uca; selfcheck karnesi deterministik; agent yazma
   sınırları (üretim dosyası yazan görev tipi yok) testle kanıtlı.
3. **Torch'suz ortamda eski 938 testin tamamı yeşil** (fail-soft kanıtı — ayrıca koşulur).
4. mypy(nihai) temiz + ruff temiz.
5. Anayasa denetimi: `nihai/ai` → sinyal/eşik/META_GATE/execution yönünde SIFIR import;
   üretim `data/meta_model.*`'a sıfır yazım; tüm ts UTC; tüm durum yazımları atomic_io.
6. Rapor dilinde "kenar" iması sıfır; her brif reasoner kapanışıyla biter.

---

## BÖLÜM J — RİSKLER VE AÇIK SORULAR

| # | Risk/Soru | Etki | Önlem |
|---|---|---|---|
| 1 | **Hedef kopya belirsiz** ("toplu" vs ana Desktop) | Yanlış repoya teslim | Faz 0'da kullanıcı teyidi — onay mesajında belirtilebilir |
| 2 | torch CPU wheel ~200MB indirme | Kurulum süresi | Faz 0'da tek sefer; venv OneDrive dışı değilse boyut notu |
| 3 | 5.5M param CPU eğitim süresi (96-token dizi = ek dikkat-hesabı yükü) | Faz 3 uzayabilir | 97k örneklemde ~dakikalar–on dakikalar mertebesi; sentetik ≤500k bütçeli; gerekirse batch/epoch bütçesi kısılır (taban sözleşmesi etkilenmez) |
| 4 | OneDrive dosya kilidi | Bozuk kayıt | Tüm yazımlar atomic_io (OP-2 dersi zaten kurumsal) |
| 5 | Ajan oturum-limiti | Eksik WP | Orkestratör geri-düşüşü (Tur-15'te kanıtlı) |
| 6 | Az veriyle overfit algısı | Yanlış "kenar" iddiası | Bölüm F çerçevesi + rapor dili sözleşmesi + Bölüm G kapıları |
| 7 | Lab sandbox kaçışı (kod çalıştırma yeteneği) | Üretim dosyası bozulması | Beyaz-liste namespace, salt-okunur data/, süre/bellek limiti, kaçış-denemesi testleri Faz 2'de bizzat koşulur |
| 8 | Edge-lab çoklu-test şişmesi (çok aday → şans eseri "kenar") | Sahte pozitif | Her adayda DSR (deflated Sharpe) zorunlu + BACKLOG'da denenen-aday sayacı (deneme sayısı DSR'ye girer) |
| 9 | Portföy önerisinin "tavsiye" sanılması | Yanlış kullanım | Çıktı başlığında sabit GÖLGE/rapor ibaresi + reasoner kapanışı; execution bağlantısı kodda yok |
| 10 | WP-M2/Heston-PDE sayısal kararsızlık | Yanlış değer raporu | Analitik-limit testleri (BS paritesi) + kondisyon uyarıları + savunmacı None dönüşü |
| 11 | LLM maliyeti/kotası (WP-T/WP-A router çağrıları) | Maliyet sürprizi / kesinti | Tüm LLM adımları fail-soft + çağrı-başına bütçe sayacı + deterministik yedek her yerde |
| 12 | Metin verisi yanlılığı (haber kaynağı seçimi) | Çarpık ton skoru | LM-sözlük deterministik taban her zaman yan yana; kaynak adı skor kaydında saklanır |
| 13 | Ajanlık kapsam-kaçağı (hipotez kuyruğu kendi kendine büyür) | Kaynak israfı | Kuyruk derinlik limiti + yalnız-analiz görev tipleri + günlük görev tavanı |
| 14 | rev.3 kapsam büyümesi (13 WP tek turda) | Teslimat riski | İki-dalga fazlaması; Dalga-2 paketleri bağımsız — gecikirse çekirdek teslim edilir, kalan Tur-17'ye devreder |

---

## BÖLÜM K — BAŞLAMA PROTOKOLÜ

Kullanıcı **"limit sıfırlandı, başla"** (+ tercihen hedef kopya bilgisi) dediğinde:
1. Bu plan yeniden okunur (tek gerçek kaynak).
2. Faz 0 çalıştırılır (teyit + temiz-commit + torch + SPEC).
3. Faz 1 ajanları başlatılır.

**O ana kadar: HİÇBİR kod/kurulum/commit yapılmaz.** ⏸
