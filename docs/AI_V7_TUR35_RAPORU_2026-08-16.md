# AI v7 — "Etiketsiz Veri, Dağılım ve Ölçek" (Tur-35, 2026-08-16)

**Kapsam:** kapasite merdiveninin yeni üst basamağı + iki YENİ yetenek + iki tur ertelenmiş
bir açık kalemin kapanışı + dört ölçülmüş optimizasyon. Hepsi **KARAR-DIŞI**; hiçbiri kenar
iddiası taşımaz.

**Doğrulama:** **2595 test yeşil** (Tur-34 sonu 2459 → **+136**) · `ruff check .` temiz ·
`mypy nihai` 96 dosya temiz. Ağ erişimi yok; canlı `data/`, `logs/` dosyalarına YAZILMADI
(tüm canlı ölçümler kopya üzerinde koştu).

---

## 1. Neden bu tur

Kullanıcı isteği üç kalemliydi: **AI'ı geniş kapsamlı geliştir · parametre sayısını artır ·
sonunda detaylı optimizasyon yap.** Turun seçim ilkesi Tur-32/34'ün aynısı: yeni yetenek
İCAT ETMEK yerine sistemin KENDİ belgelerinde işaretli boşluklar kapatıldı.

| Kaynak | Satır | Kapanış |
|---|---|---|
| ACI DERS #5 | *"Gerçek darboğaz = veri ... yeni bir sınıflandırıcı eklemek bunu çözmez"* | `NeuralCapacity.pretrain` — **etiket gerektirmeyen** öz-denetimli ön-eğitim (§5) |
| Tur-32 açık kalem #4 | *"`hedge_combine` × canlı `sources` A/B"* — iki turdur erteleniyor | `learning.source_fusion` (§6) |
| `neural.py` (tüm basamaklar) | oynaklık hep TEK SAYI olarak tahmin ediliyordu | `vol_q` quantile başlığı + pinball kaybı (§4) |
| Kullanıcı isteği | kapasite | `quant_transformer_ultra` — **771.874.072 parametre** (§3) |
| Profil (bu tur) | üretim yolunda kuadratik kuyruk okuma | `atomic_io.tail_lines` (§7.3) |

---

## 2. Önce temizlik: ortak dizi kodlayıcı (`_SeqEncoderMixin`)

Dört transformer sınıfı `forward()` gövdesinin AYNI 12 satırını kopyalıyordu (tokenize →
`[CLS]`+3×`[SCALE]`+özellik dizisi → blok yığını → son norm → füzyon → 6 başlık). Maskeli
ön-eğitim bu kopyayı BEŞE çıkarırdı; onun yerine gövde tek yere alındı ve ön-eğitim AYNI
gövdeyi `mask` argümanıyla çağırıyor — *"ön-eğitim ile eğitim aynı hesabı görür"* bir yorum
değil, **yapısal sonuç**.

`state_dict` anahtarları DEĞİŞMEDİ (v3'ün `final_ln` ↔ v4+'in `final_norm` ayrımı `getattr`
ile çözüldü), yani kayıtlı canlı artefaktlar aynen yüklenir.

**BİT-DENKLİK KANITI.** Refactor ÖNCESİ beş basamağın tüm çıktı tensörleri (eval + train
kipi, sabit seed) diske yazıldı; refactor SONRASI aynı script karşılaştırma kipinde koştu:

```
karşılaştırılan 62 tensör · fark 0
```

---

## 3. Kapasite: `quant_transformer_ultra` (≈772M parametre)

Merdivenin yeni üst ucu. MoE-XL'den iki farkı var ve **ikisi de yalnız büyüklük değil**:

1. **Daha ince tane, daha çok uzman.** Yönlendirilen uzman 16 → **40**, uzman gizli genişliği
   1024 → **768** (uzman BAŞINA küçülür, sayı katlanır), d_model 512 → 640, blok 10 → 12.
   Seçilebilen uzman kombinasyonu C(16,2)=120 → **C(40,2)=780**. İnce tanenin kazancı
   parametre değil **ifade gücüdür** (DeepSeekMoE, Dai ve ark. 2024 §3); parametre yan üründür.
2. **Dağılımsal oynaklık başlığı** (§4) — merdivenin bu basamağı yeni bir GÖREV de öğrenir.

Paylaşımlı uzman (her token'da koşulsuz), Switch yük-dengeleme kaybı ve ST-MoE router
z-kaybı MoE-XL ile aynen korundu. Dizi düzeni ve `nn.MultiheadAttention` DEĞİŞMEDİ →
`explain.attention_rollout` sözleşmesi ayakta.

**ÖLÇÜLDÜ (f=96, CPU):**

| basamak | toplam parametre | önceki basamağa oran |
|---|---:|---:|
| `deep_mlp` | 223.054 | — |
| `quant_transformer` | 5.465.363 | 24,5× |
| `quant_transformer_xl` | 31.756.051 | 5,8× |
| `quant_transformer_moe` | 93.228.307 | 2,9× |
| `quant_transformer_moe_xl` | 283.639.571 | 3,0× |
| **`quant_transformer_ultra`** | **771.874.072** | **2,72×** |

**Seyreklik ÖLÇÜLDÜ:** toplam 771.874.072, AKTİF (token başına gerçekten çarpılan)
**98.482.456** → **7,84×**. Yani kapasite hesaptan gerçekten ayrılmıştır; MoE'nin varlık
sebebi budur (yoğun genişletme bu donanımda eğitimi imkânsız kılardı — Tur-22 §7-1).

`recommend_tier` eşiği **n ≥ 10.000.000** (aynı oran kuralı: 3,6M × 2,72 ≈ 9,8M). Bugünkü
canlı log ≈6k satır — eşik **dört mertebe** uzakta ve bu BİLİNÇLİDİR.

> **Kapasite artışı bir KENAR iddiası DEĞİLDİR (ACI DERS #5).** Bu basamağın canlı modeli
> YOKTUR ve planlanmıyor. 772M parametre float32'de ≈3,1 GB ağırlık + AdamW ile ≈9,2 GB
> optimizer durumu ister; bu donanımda eğitim GÜNLER sürer (Tur-22 §7-1'in ölçülmüş dersinin
> dördüncü tekrarı). Merdiven veri disiplinini gevşetmek için değil, **gevşetmediğini
> göstermek için** uzuyor.

---

## 4. Dağılımsal oynaklık: `vol_q` + pinball kaybı

Sistem oynaklığı bugüne kadar hep TEK SAYI olarak tahmin etti (`vol` başlığı, MSE = koşullu
ORTALAMA). Karar için gereken şey aralıktır. ULTRA basamağı nokta tahminini KORUYUP yanına
5 quantile'lık bir demet ekler (τ = 0,1 / 0,25 / 0,5 / 0,75 / 0,9), pinball kaybıyla
(Koenker & Bassett 1978) eğitilir.

**Çapraz geçiş kayba değil MİMARİYE yasaklandı.** Bağımsız Q çıkış nöronu eğitilirse
quantile'lar birbirini keser (q₀.₉ < q₀.₁) ve pinball kaybı bunu GÖRMEZ — her quantile'ı
ayrı ayrı ölçer. Kurulum kümülatiftir: `q₀ = raw₀`, `qᵢ = qᵢ₋₁ + softplus(rawᵢ)`.

**İDDİANIN SINIRI (ölçüldü).** Garanti **AZALMAYAN**dır, *"kesin artan"* DEĞİL: softplus çok
negatif girdide float32'de tam 0.0'a alt-taşar ve iki komşu quantile eşitlenebilir. Bu bir
çapraz geçiş değildir (sıra bozulmaz). Test tam olarak kurulabilen iddiayı kilitler — Tur-33'ün
*"hangi iddianın kurulabileceğini algoritma belirler, umut değil"* kuralı.

`_compute_loss` pinball terimini YALNIZ `vol_q` VE `aux['vol']` birlikte varken ekler →
diğer beş basamak v6 ile **bit-denk**.

### 4.1 ÖLÇÜLDÜ — başlık gerçekten kalibre quantile öğreniyor mu?

*"Çapraz geçiş imkânsız"* mimari bir garantidir; *"doğru quantile"* DEĞİLDİR. Bilinen koşullu
dağılımlı sentetik veride (ölçek `0,5+|x₀|`, çarpımsal log-normal gürültü → **asimetrik**;
simetrik olsaydı nokta tahmini + sabit bant yeterdi) **ampirik kapsama** ölçüldü — τ-quantile'ın
altında kalan gözlem oranı τ'ya yakın olmalı. Küçük yapılandırma, 4.500 eğitim / 1.500 test:

| τ | 0,10 | 0,25 | 0,50 | 0,75 | 0,90 |
|---|---:|---:|---:|---:|---:|
| **ampirik kapsama** | 0,080 | 0,235 | 0,459 | 0,724 | 0,891 |

Beşi de nominalin 0,02–0,04 içinde ve hepsi hafif MUHAFAZAKÂR yönde. Çapraz geçiş **0**. Nokta
`vol` başlığı da bozulmadı (ort. 1,343 ↔ gerçek 1,388) — iki başlık birbirini yemiyor.

> Bu bir KENAR iddiası değildir: sentetik veride bilinen bir dağılımın geri kazanılması, canlı
> oynaklığın tahmin edilebildiği anlamına GELMEZ. Ölçülen şey mekanizmanın doğru çalıştığıdır.

---

## 5. Maskeli öz-denetimli ön-eğitim (`pretrain`)

**Neden bu, ACI DERS #5'e verilen ilk cevap:** ders *"gerçek darboğaz veri ve OOS-kenarlı baz
sinyal yokluğu, model mimarisi değil"* diyor ve *"yeni bir sınıflandırıcı eklemek bunu
çözmez"* diye bitiyor. Maskeli özellik modelleme yeni bir sınıflandırıcı DEĞİLDİR:
**etiket gerektirmez.** `data/live_features_log.csv` her aday sinyalde bir özellik vektörü
biriktirir; bunların yalnız ÇÖZÜLMÜŞ olanları etiketlidir. Ön-eğitim etiketsiz çoğunluğu da
kullanır — darboğazın VERİ tarafına dokunan tek koldur.

**Nasıl:** her satırda özelliklerin `mask_ratio` kadarı seçilir, o özelliklerin token'ı
öğrenilebilir `[MASK]` gömmesiyle **değiştirilir** (sıfırlanmaz — sıfır geçerli bir normalize
değerdir, ağ *"maskelendi"*yi *"değeri 0"* sanardı) ve ağ maskelenen özelliğin gerçek değerini
geri kurar. Kayıp YALNIZ maskeli konumlardan alınır.

**Dürüstlük sözleşmesi (dördü de testle kilitli):**

| Kural | Neden |
|---|---|
| `pretrain` `is_fitted`'i **AÇMAZ** | Etiket görmemiş gövdeden `p_win` okumak sessiz bir yalan olurdu. `predict_proba` RuntimeError verir. |
| Rekonstrüksiyon başlığı + `[MASK]` gömmesi **atılır** | Artefakt şeması sıfırdan eğitilmiş modelle AYNI kalır (`state_dict`e sızmaz). |
| Val bölmesi **kronolojik** son %15 | Look-ahead yasağı etiketsiz veride de geçerlidir. |
| `deep_mlp` → **ValueError** | Token'ı yoktur; sessizce atlamak çağıranın "ön-eğitim yaptım" sanmasına yol açardı. |

CLI: `python -m research.train_neural --pretrain --pretrain-epochs 30 [--pretrain-dataset X.csv]`.
Ayrı (daha büyük, etiketsiz) bir dataset verilebilir; şema uyuşmazlığı hata verir.

### 5.1 ÖLÇÜLDÜ — canlı 1m verisinde A/B: KAZANÇ YOK

Canlı 1m dataset (`data/meta_dataset_live_1m_2026-07-30.csv`, 5.996 satır, 25 özellik, taban
oranı 0,4681), **purged 5-fold** (purge=35, embargo=60), İKİ KOL TAM AYNI bölmeleri gördü:

- **A** — v4 reçetesi, sıfırdan (cosine + EMA + balanced + label smoothing + val_auc + calibrate)
- **B** — aynı reçete + maskeli ön-eğitim (10 epoch, YALNIZ eğitim diliminin X'i → sızıntı yok) + `warm_start`

| metrik | A (sıfırdan) | B (ön-eğitim + warm start) | Δ (B−A) |
|---|---:|---:|---:|
| OOS-AUC | 0,4896 ± 0,0167 | 0,4881 ± 0,0135 | **−0,0015** |
| Brier | 0,2491 ± 0,0010 | 0,2490 ± 0,0009 | −0,0001 |
| ECE | 0,0178 ± 0,0125 | 0,0147 ± 0,0124 | −0,0031 |
| doğruluk | 0,5234 ± 0,0288 | 0,5319 ± 0,0190 | +0,0085 |

B, AUC'de A'yı **3/5** fold'da geçti — yani yazı-tura. **Hipotez DOĞRULANMADI:** farkların
hepsi kendi fold-arası standart sapmalarının çok altında. (Tek gözlenebilir örüntü B'nin
fold-arası YAYILIMININ daha dar olması: AUC std 0,0167 → 0,0135, doğruluk std 0,0288 → 0,0190.
Bu tek başına bir kazanç iddiası DEĞİLDİR; 5 fold'da std farkı gürültüdür.)

**NEDEN — ve mekanizmanın çalıştığının kanıtı.** Rekonstrüksiyon kaybı canlı sette neredeyse
hiç düşmedi: fold 1 ve 2'de **tam sıfır** iyileşme (erken durma ilk epoch'ta en iyiyi buldu),
en iyi durumda fold 4'te 0,6761 → 0,6568 (%2,9). Karşılaştırma: AYNI kod sentetik müfredat
üzerinde 2 epoch'ta 0,8365 → 0,6764 (**%19**) verdi. Yani gövde öğrenebiliyor; **canlı 25
özellikte öğrenilecek çapraz-özellik yapısı kalmamış.** Bu şaşırtıcı değil: `meta_features.py`
vektörü zaten elle mühendislik edilmiş, büyük ölçüde dikleştirilmiş göstergelerden oluşuyor —
maskeli modelleme tam da bu tür artıklıktan beslenir.

**Dürüst sonuç:** ön-eğitim bu veri üzerinde yardımcı olmadı. Yöntemin asıl hedefi zaten bu
değildi: kazanç `data/live_features_log.csv`'nin **ETİKETSİZ çoğunluğunda** aranmalı (bu turda
ölçüm etiketli setin kendi X'iyle yapıldı, yani **ek veri KULLANILMADI** — açık kalem #3).
Rekonstrüksiyon iyileşmesi ⇒ yön tahmini iyileşmesi çıkarımı da zaten yasaklı (ACI DERS #5).

---

## 6. Çevrimiçi kaynak füzyonu — Tur-32 açık kalem #4 KAPANDI

`decision.py` iki olasılık kaynağını (`pwin` meta-model, `neural` gölge ağı) Brier'den türeyen
**sabit** ağırlıklarla harmanlıyor. Sabit ağırlık iki şeyi varsayar: (1) kaynakların göreli
kalitesi zamanla değişmez, (2) kalite ölçümü tüm geçmişe eşit bakabilir. İkisi de rejimli bir
piyasada şüphelidir. `online.hedge_combine` yazılmış ve test edilmişti ama **canlı `sources`
üçlüsüne hiç bağlanmamıştı** — iki tur ertelendi.

**Hizalama sözleşmesi (bu kısım en kolay yanlış yapılan yer):** üç uzman TEK olayı yanıtlamak
zorundadır — *"canlı sinyalin YÖNÜ kazandı mı"*. Kaynak karşı tarafı söylüyorsa olasılığı
**1−p** ile çevrilir. Aksi hâlde uzmanlar farklı sorulara cevap verir ve birleşim anlamsızlaşır.
`update_from_predictions` yalnız İKİ kaynağın da olasılık ürettiği satırları kullanır.

**Kapı iki katmanlı** ve ikisi de geçilmelidir: `vs_base > 0` (yön) **ve**
`paired_brier_t(birleşim, taban) < −1,645` (gürültü mü). t-testi Tur-24/27/28 ile AYNI tek
kaynaktan gelir (`learning.paired_brier_t`).

**ÖLÇÜLDÜ (canlı `predictions_log.csv` kopyası, 62.462 satır → 1.286 hizalı üçlü):**

```
log-kayıp   birleşim 0,694924  ↔  pwin 0,741877 · neural 0,699335 · TABAN 0,694189
en iyi tek uzman : base
pişmanlık        : +0,000735      (birleşim en iyi TEK uzmanın gerisinde)
vs_base          : −0,000735
eşleştirilmiş Brier farkı (birleşim − taban) : +0,000373   (t = +0,329)
son ağırlıklar   : pwin 0,142 · neural 0,429 · base 0,429
durum            : TABANI_GECILEMEDI
```

Okunuşu: **hiçbir kaynak koşan taban oranının ötesinde bilgi taşımıyor** — ACI DERS #2/#4/#9'un
üçüncü bağımsız doğrulaması (tek-bölme kalibrasyon kapısı, yürüyen kalibrasyon kapısı ve şimdi
çevrimiçi füzyon aynı şeyi söylüyor). Mekanizma DOĞRU çalıştı: Hedge en kötü uzmanı (`pwin`,
log-kayıp 0,742) 1/3'ten **0,142**'ye indirdi. `pişmanlık > 0` satırı rapora bilerek yazılıyor
— Hedge'in teorik garantisi *"en iyi uzmana yakın kalmak"*tır, *"onu geçmek"* DEĞİLDİR.

---

## 6.1 Kavram kataloğu: 107 → 110

Bu turda eklenen üç yetenek `knowledge.py`'ye de yazıldı — sistemin kendi bilgi tabanı
yeteneklerini yansıtmazsa `reasoner`/`brain` onları hiç bilmez: `maskeli_on_egitim`,
`quantile_regresyon`, `cevrimici_uzman_birlestirme`. Üçü de **tuzak** listeleriyle geldi
(ör. *"rekonstrüksiyon kaybının düşmesi yön tahmininin iyileşeceği anlamına GELMEZ"*,
*"birleşim en iyi tek uzmandan iyi olacak beklentisi YANLIŞTIR — pişmanlık pozitif olabilir
ve bu Hedge'in kusuru değil TANIMIDIR"*).

Formül köprüsü (`selfcheck._FORMULA_MODULES`) `online` + `learning` ile genişledi (Tur-32'de
3, Tur-34'te 4, şimdi 6 modül). **`neural` bilerek DIŞARIDA**: köprüye eklemek her kavram
sorgusunda torch importunu tetiklerdi (Tur-26'nın ölçülmüş disiplini: 3854 ms ↔ 11 ms). O
kavramlar `formul_ref=[]` taşır ve hangi modülde yaşadıklarını metinle söyler.

`tests/test_ai_knowledge.py::test_ders_ref_format_aci_n` deseni `^ACI-[1-7]$` idi ve
**bayattı** — ACI DERSLER #8 ve #9 CLAUDE.md'ye 2026-07/08'de eklenmişti. Desen 1-9'a
hizalandı: bu bir kural gevşetmesi değil, testin belgeyle yeniden hizalanmasıdır.

---

## 7. Optimizasyon — ölçülmüş, denkliği kanıtlanmış

Profil önce alındı, sonra kod değişti. Dört kalem; her biri için denklik iddiası koda
gömüldü (`tests/test_tur35_parity.py`).

### 7.1 `neural._auc_np` — beraberlik döngüsü (BİT-DENK, 3,7–12,1×)

**ACI DERSİ üçüncü kez tekrarladı: bir ders MODÜLE değil DESENE uygulanır.** Tur-10 tam bu
beraberlik döngüsünü `research/train_meta._auc` ve `research/forward_harness._auc`'ta cumsum
ile vektörleştirmişti. `neural.py` Tur-16'da yazılırken **saf-Python hâli yeniden doğdu** ve
`evaluation_metrics`in %85'ini yiyordu.

Dönüşüm: grup [i..j] için `(i+j)/2 + 1 ≡ i + (c+1)/2` (ikisi de float64'te tam temsil edilir),
grup sınırı `sorted_p[k] != sorted_p[k−1]` ile bulunur — **NaN != NaN** olduğundan her NaN
özgün döngüdeki gibi kendi grubunu kurar (semantik birebir korunur).

```
n=  5.000 sürekli:  3,87 →  0,32 ms  (12,08×)      n=  5.000 bağlı:  1,16 → 0,26 ms  (4,43×)
n= 50.000 sürekli: 39,58 →  3,98 ms  ( 9,96×)      n= 50.000 bağlı: 11,36 → 2,69 ms  (4,22×)
n=200.000 sürekli:166,97 → 21,10 ms  ( 7,91×)      n=200.000 bağlı: 46,42 →12,49 ms  (3,72×)

evaluation_metrics(n=200k):  377,3 → 35,0 ms  (10,8×)
DENKLİK: 30 senaryo (sürekli/çok-bağlı/hepsi-aynı/NaN/±Inf/negatif × 6 boyut) · bit-fark 0
```

### 7.2 `online.hedge_combine` — adım-başı kayıp (BİT-DENK, 2,6–2,9×)

Uzman kayıpları her adımda yeniden hesaplanıyordu. Kayıp yalnız `(pᵢ,ₜ, yₜ)`'ye bağlıdır —
ağırlıklara DEĞİL — yani (n,k) matrisi döngü ÖNCESİ hesaplamak nedenselliği bozmaz; ağırlık
güncellemesi hâlâ yalnız o adımın satırını okur. `_log_loss` eleman-bazlı olduğu için (indirgeme
yok) dönüşüm bit-denktir.

```
n=  1.286:   22,95 →  8,80 ms  (2,61×)     ← canlı füzyonun bugünkü boyutu
n=  6.000:  109,32 → 37,91 ms  (2,88×)
n= 60.000: 1075,36 →382,21 ms  (2,81×)
DENKLİK: 5 senaryo (k=2..5, share=0..1) · combined/weights/final_weights maksimum fark 0,0
```

### 7.3 `atomic_io.tail_lines` — kuadratik kuyruk okuma (ÇIKTI-DENK, 9,7–86×)

Profil üretim yolunda tek bir suçlu gösterdi: `learning.update_from_predictions` süresinin
**%67'si** `bytes.count` içindeydi. Neden: kuyruk okuyucu her 8 KB'lik adımda **biriken
tamponun TAMAMINI** yeniden sayıyor ve `f.read(step) + data` ile baştan kopyalıyordu. 8,4 MB'lik
canlı logda 1020 yineleme = **4,3 GB tarama, 2,10 s**.

**Aynı kusur ÜÇ modülde birebir kopya hâlinde yaşıyordu** (`ai/learning`, `ai/calibration_loop`,
`ai/reasoner`) — ve doğru artımlı sayım deseni zaten `reasoner._count_data_lines`'ta, **20 satır
yukarıda** duruyordu. Ders yine MODÜLE değil DESENE aitti. Üç kopya `nihai/atomic_io.py`'deki
tek kaynağa indirildi (blok geometrik büyür: 16 KB → 1 MB).

```
n=    500:    0,4 →  0,40 ms  ( 1,1×)     ← reasoner'ın kullandığı boyut (regresyon YOK)
n=  5.000:   26,5 →  2,73 ms  ( 9,7×)     ← learning'in üretim varsayılanı
n=100.000: 2666,6 → 31,12 ms  (85,7×)
DENKLİK: 12 senaryo (n=1..10⁷, boş dosya, olmayan dosya, yarım ilk satır) · fark 0

ÜRETİM YOLU: learning.update_from_predictions(62k satır) 3151 → 544 ms (5,8×)
```

**Yan ürün — sessiz bir kusur kapandı.** `n=0`'da `lst[-0:]` Python'da `lst[0:]`dir: *"sıfır
satır iste"* TÜM okunan bloğu döndürüyordu ve satır sayısı BLOK BOYUTUNA bağlıydı (8 KB'de 70,
256 KB'de 1693) — tanımsız davranış. `AI_REASONER_PRED_TAIL=0` ile tetiklenebilirdi. Artık
`n <= 0 → []`.

### 7.4 `neural.theoretical_param_count` — `meta` aygıtı (SAYI-DENK, 2,5× ve ~0 bellek)

Bu turda eklenen 772M'lik basamak `param_ladder_report`'u zehirliyordu: pano her Beyin sekmesi
açtığında merdiveni istiyor ve ULTRA'yı GERÇEKTEN kurmak **3,1 GB ayırıp 5,2 sn** harcıyordu.
Model artık `meta` aygıtında kurulur — parametreler ŞEKLE sahiptir ama DEPOLAMA AYIRMAZ.
Parametre sayısı depolamadan bağımsız olduğundan sonuç birebir aynıdır.

```
altı basamağın hepsinde meta == gerçek sayım (deep_mlp 177.998 … ultra 768.999.640, f=8)
merdiven kurulumu: 7,58 s / ~4,5 GB  →  3,02 s / ~0 GB
```

Fail-soft: `meta` yolu patlarsa (gelecekteki bir mimari kurulumda gerçek veri okursa) sessizce
gerçek kuruluma düşer — sayı yine doğru üretilir.

### 7.5 Gösterge hattı: `pd.concat(...).max(axis=1)` → `np.fmax` (BİT-DENK, 1,8–13,5×)

**Aynı ders, aynı turda ÜÇÜNCÜ kez.** Tur-6 tam bu deseni `strategy._true_range`'ten
çıkarmıştı ve kodda hâlâ şu yorum duruyor: *"concat+max backtest sıcak yolunda ~%40 CPU
yiyordu"*. Desen gösterge modüllerinde **dört yerde** yaşamaya devam ediyordu — üstelik
Tur-33/34'ün optimize ettiği aynı dosyalarda. `ultimate_oscillator` ayrıca AYNI ifadeyi
(`pd.concat([low, prev_close]).min(axis=1)`) **iki kez** hesaplıyordu; `swing_index` de iki
terimi iki kez birleştiriyordu.

```
ultimate_oscillator (panelin içinde):  2,833 → 1,581 ms  ( 1,79×)
williams_ad         (pano osilatörü):  2,066 → 0,153 ms  (13,48×)
swing_index                         :  1,730 → 0,348 ms  ( 4,98×)
indicator_panel (n=1500, bütün)     :   85,4 →  70,0 ms  ( 1,22×)

DENKLİK: 300 rastgele df (normal / TAMAMEN DÜZ / %5 NaN'lı × dört uzunluk)
         NaN deseni farkı 0 · maksimum mutlak değer farkı 0,000e+00
```

`np.fmax/fmin` NaN'ı yok sayar = pandas'ın `skipna=True`'su; ikisi de "her ikisi de NaN ise
NaN" der → dönüşüm bit-denktir ve dejenere (düz/NaN'lı) girdide de doğrulandı.

### 7.6 Ölçüldü ama DEĞİŞTİRİLMEDİ (dürüstlük kalemi)

Optimizasyon raporu yalnız kazanılanları yazarsa, sonraki tur aynı yerleri yeniden kazar.

- **MoE FFN ileri geçişi.** `_FineGrainedMoEFFN` (16 uzman, top-2, N=4096): 111,4 ms. Tek bir
  `_SwiGLU` 35,3 ms → top-2 + paylaşımlı = 3 uzman-eşdeğeri ≈ 105,9 ms. Yönlendirme yükü
  (`nonzero` 3 ms + `index_add_` 9 ms + `any` 1 ms) toplamın **%3,4'ü**. Kazanılacak yer YOK;
  gruplu matmul'a geçmek toplama sırasını değiştirir ve bit-denkliği %3,4 için satmaya değmez.
- **`predict_with_uncertainty`.** T=20 geçişte sürenin **%84'ü** `torch.dropout`'un kendisinde
  (Philox RNG). Tensör kurulumunu döngü dışına almak <%3 kazandırır; T geçişi tek toplu-işe
  yığmak RNG tüketim SIRASINI değiştirir → bit-denklik kaybolur. Yapılmadı.
- **`neural._ece_np`.** 200k'da 11,5 ms; kova döngüsünü `bincount`la vektörleştirmek toplama
  SIRASINI değiştirir (bit-denk olmaz) ve `_auc_np` düzeltmesinden sonra zaten baskın değil.

---

## 8. Doğrulama

- **2595 test yeşil** (Tur-34 sonu 2459 → +136), `ruff check .` temiz, `mypy nihai` 96 dosyada
  sorun yok.
- Yeni test dosyaları: `tests/test_tur35_parity.py` (optimizasyon paritesi) +
  `tests/test_tur35_ai.py` (yetenek sözleşmeleri).
- Bit-denklik referansı 62 tensörde fark 0 (§2).
- **CANLI ARTEFAKT GERİYE UYUMU ölçüldü.** Tur-20'nin üretimdeki gölge modeli
  (`data/neural_deep_mlp_live1m_2026-07-30`) yeni kodla yüklendi: `is_fitted=True`,
  `param_count=186.702`, `predict_proba` ve MC-dropout çalışıyor, `predict_vol_quantiles`
  dürüstçe `None`. `is_fitted` sözleşmesi değiştiği hâlde diskteki artefakt etkilenmedi
  (`load` onu tanımı gereği eğitilmiş sayar).
- Canlı `data/` ve `logs/` dosyalarına YAZILMADI; tüm canlı ölçümler kopya üzerinde koştu.

---

## 9. Açık kalemler

1. `quant_transformer_ultra`'nın canlı modeli YOK ve olması planlanmıyor (eşik 10M satır,
   bugün ≈6k). Basamak KAPASİTE beyanıdır, kullanım önerisi değil.
2. `vol_q` başlığı YALNIZ ULTRA'da. Alt basamaklara taşımak `state_dict` şemasını değiştirir →
   kayıtlı artefaktların bit-denkliği bozulur; ayrı ve bilinçli bir adım olmalıdır.
3. Ön-eğitimin `live_features_log.csv`'nin ETİKETSİZ çoğunluğuyla koşturulması — asıl kazanç
   oradadır (bu turda ölçüm etiketli setin X'iyle yapıldı, yani ek veri KULLANILMADI).
4. Kaynak füzyonu bugün iki uzmanla koşuyor (`pwin`, `neural`). `prob_cal`/`confidence` de
   uzman olarak eklenebilir; hizalama sözleşmesi (ortak olaya çevirme) her yenisinde tekrar
   kurulmalıdır.
5. Tur-34 açık kalemleri 1–4 bu turda da kapanmadı (MoE uzmanlaşmasının gerçek rejim
   etiketleriyle ölçümü, `regime_hmm`'in canlı loga bağlanması, `copula_select`in gerçek
   sembol çiftlerinde koşturulması, `pin_from_bars`in tick verisiyle çapraz-doğrulanması).
