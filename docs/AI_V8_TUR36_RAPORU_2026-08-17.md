# AI v8 — "Belirsizlik, Opt-in Başlık ve Ölçek" (Tur-36, 2026-08-17)

**Kapsam:** kapasite merdiveninin yeni üst basamağı (**2,01 milyar** parametre) + bir YENİ
yetenek (kanıt-temelli belirsizlik) + Tur-35'in **dört açık kaleminden üçünün** kapanışı +
**üç** ölçülmüş optimizasyon (biri bu turun kendi getirdiği regresyonun onarımı). Hepsi **KARAR-DIŞI**; hiçbiri kenar iddiası taşımaz.

**Doğrulama:** **2600 test yeşil** (bu turda **+30**: `tests/test_tur36_ai.py` 22 +
`tests/test_tur36_parity.py` 8) · `ruff check .` temiz · `mypy nihai` 96 dosya temiz.
Ağ erişimi yok; canlı `data/`, `logs/` dosyalarına **YAZILMADI** (tüm canlı ölçümler kopya
üzerinde koştu).

---

## 1. Turun seçim ilkesi

Tur-32/34/35'in aynısı: yeni yetenek İCAT ETMEK yerine sistemin **kendi belgelerinde
işaretli** boşluklar kapatıldı.

| Kaynak | Satır | Bu turdaki kapanış |
|---|---|---|
| Tur-35 açık kalem **#2** | *"`vol_q` YALNIZ ULTRA'da; alt basamaklara taşımak state_dict şemasını bozar"* | `extra_heads` opt-in mekanizması (§3) |
| Tur-35 açık kalem **#3** | *"asıl kazanç etiketsiz çoğunluktadır"* | `build_unlabeled` + ÖLÇÜM: çoğunluk **YOK** (§6) |
| Tur-35 açık kalem **#4** | *"`prob_cal`/`confidence` de uzman olarak eklenebilir"* | `source_fusion(extra=…)` (§5) |
| Kullanıcı isteği | kapasite / parametre sayısı | `quant_transformer_mega` — **2.014.146.906** (§2) |
| `neural_shadow.py` üretim yolu | her tahminde **T=20** MC-dropout geçişi | `dir_evid` — TEK geçişte belirsizlik (§4) |
| Profil (bu tur) | `stats.norm.cdf` skaler çağrısı **39,3 µs** | stdlib `NormalDist` (§7.2) |

---

## 2. Kapasite: `quant_transformer_mega` (2,01 MİLYAR parametre)

Merdivenin yeni üst ucu. **ULTRA'nın ALT SINIFIDIR** ve bu bilinçli: mimari identik (aynı
dizi düzeni `[CLS]+3×[SCALE]+özellikler`, aynı `_TransformerBlockMoEXL`, aynı
`nn.MultiheadAttention` → `explain.attention_rollout` sözleşmesi ayakta), fark YALNIZ ölçek
sabitleri ve varsayılan ek başlıklardır. 70 satırlık bir kopya, "basamaklar arası fark tek
değişken olsun" kuralını görünmez kılardı.

**ÖLÇÜLDÜ (f=96, `meta` aygıtı):**

| basamak | toplam parametre | önceki basamağa oran |
|---|---:|---:|
| `deep_mlp` | 223.054 | — |
| `quant_transformer` | 5.465.363 | 24,5× |
| `quant_transformer_xl` | 31.756.051 | 5,8× |
| `quant_transformer_moe` | 93.228.307 | 2,9× |
| `quant_transformer_moe_xl` | 283.639.571 | 3,0× |
| `quant_transformer_ultra` | 771.874.072 | 2,72× |
| **`quant_transformer_mega`** | **2.014.146.906** | **2,61×** |

Alt basamakların altı sayısı da Tur-35 raporundaki değerlerin **birebir aynısıdır** —
`extra_heads` mekanizmasının (§3) geriye uyumunun ilk kanıtı budur.

**Seyreklik ÖLÇÜLDÜ:** toplam 2.014.146.906, token başına AKTİF **171.340.122** →
**11,76×** (ULTRA'da 7,84×). Yani toplam parametre 2,61× artarken hesap yalnız **1,74×**
arttı: basamak "daha çok hesap" değil **daha çok KAPASİTE** ekliyor. Seçilebilen uzman
kombinasyonu C(40,2)=780 → **C(80,2)=3160**.

`recommend_tier` eşiği **n ≥ 26.000.000** (aynı oran kuralının beşinci uygulaması:
10M × 2,61 ≈ 26M). Bugünkü canlı log ≈7,7k satır — eşik **dört-buçuk mertebe** uzakta.

> **Kapasite artışı bir KENAR iddiası DEĞİLDİR (ACI DERS #5).** Bu basamağın canlı modeli
> YOKTUR ve planlanmıyor: 2,01 milyar parametre float32'de ≈8,0 GB ağırlık + AdamW ile
> ≈24 GB optimizer durumu ister; bu donanımda eğitim İMKÂNSIZDIR (Tur-22 §7-1 dersinin
> beşinci tekrarı). Merdiven veri disiplinini gevşetmek için değil, **gevşetmediğini
> göstermek için** uzuyor.

---

## 3. `extra_heads` — Tur-35 açık kalem #2'nin kapanışı

Tur-35 `vol_q` başlığını ULTRA'ya **gömmüştü** ve açık kalem şunu söylüyordu: *"alt
basamaklara taşımak `state_dict` şemasını değiştirir → kayıtlı artefaktların bit-denkliği
bozulur; ayrı ve bilinçli bir adım olmalıdır."*

Bu tur o adımı attı ve **başlığı basamağa gömmek yerine OPT-IN yaptı**:

```python
NeuralCapacity("deep_mlp", feats, extra_heads=("vol_q", "dir_evid"))   # açıkça iste
NeuralCapacity("deep_mlp", feats)                                      # varsayılan: v7 ile BİT-DENK
```

Üç şey bu tasarımı taşıyor:

1. **Boş küme hiçbir modül kurmaz.** `_init_extra_heads` `extra=()` ile hiç `nn.Module`
   yaratmaz → `torch.manual_seed` akışından tek bir rastgele sayı bile tüketilmez →
   ağırlıklar, parametre sayısı ve `state_dict` anahtarları v7 ile **birebir aynıdır**.
   Merdivenin altı basamağının §2'deki sayıları bunun ölçülmüş kanıtıdır.
2. **Kurulum sırası KANONİKTİR.** Çağıran `("dir_evid","vol_q")` da yazsa
   `("vol_q","dir_evid")` da yazsa modüller AYNI sırada kurulur → aynı küme = aynı
   ağırlıklar (`test_extra_head_sirasi_onemsiz_agirliklar_ayni` kilitler). Sıra RNG
   tüketimini belirlediği için bu bir kozmetik detay DEĞİLDİR.
3. **Çağrı yeri ULTRA'nın Tur-35'teki yeriyle AYNI** (`vol_head` ile `regime_head` arasında)
   → ULTRA varsayılanı (`("vol_q",)`) altında ağırlıklar v7 ile bit-denk kalır.

`extra_heads` artefaktla taşınır: `meta()["extra_heads"]` **ÖLÇÜMDÜR** — çağıranın
niyetinden değil, modelde kurulu modüllerin varlığından okunur. Alan taşımayan v7 ve öncesi
JSON'lar basamak varsayılanına düşer (`test_extra_heads_alani_olmayan_eski_artefakt_yuklenir`).

**CANLI ARTEFAKT GERİYE UYUMU ölçüldü.** Üretimdeki gölge model
(`data/neural_deep_mlp_live1m_2026-07-30`) yeni kodla yüklendi: `is_fitted=True`,
`param_count=**186.702**` (Tur-35 raporundaki sayının aynısı), `predict_proba` çalışıyor
(5.996 satırda ort. 0,4543), `effective_extra_heads()` dürüstçe `[]`, `predict_vol_quantiles`
ve `predict_evidential` dürüstçe **None**.

---

## 4. Kanıt-temelli yön (`dir_evid`) — belirsizlik TEK geçişte

### 4.1 Neden

`nihai/execution/neural_shadow.py` her canlı aday için `predict_with_uncertainty(x, T=20)`
çağırıyor: **20 ileri geçiş**, üstelik dropout RNG'siyle (Tur-35 §7.6: sürenin %84'ü
`torch.dropout`ta). Sonuç `neural_pwin_std` olarak loglanıyor.

Kanıt-temelli derin öğrenme (Sensoy ve ark. 2018) aynı soruyu tek geçişte yanıtlar: başlık
softmax OLASILIK değil **Dirichlet KANITI** üretir (α = softplus(raw) + 1, kanıt e = α−1 ≥ 0)
ve belirsizlik u = K/S (S = Σα) doğrudan okunur.

Kayıp iki terimlidir: Bayes-riski CE (`Σ y_k(ψ(S) − ψ(α_k))`) + YANILTICI kanıtı düz
Dirichlet'e çeken KL cezası. KL katsayısı ilk 10 epokta 0→1 **tavlanır** — kanıt daha
oluşmadan cezalandırılırsa model hiç kanıt üretmemeyi öğrenir (makalenin §4 uyarısı).

Başlık ana görevi ELE GEÇİRMEZ: `predict_proba` yine `direction` başlığını okur ve
sıcaklıkla kalibre edilir; `dir_evid` ikinci bir **GÖRÜŞ** değil ikinci bir **ÖLÇÜ**dür.
(Sıcaklık kanıta uygulanmaz: α'yı bölmek kanıt BÜYÜKLÜĞÜNÜ = belirsizliği bozardı.)

### 4.2 ÖLÇÜLDÜ — ve sonuç beklenenden ilginç çıktı

Canlı 1m seti (5.996 satır), **kronolojik** %75/%25 bölme, `deep_mlp`, 5 seed:

```
seed  AUC     u: düşük-çeyrek ↔ yüksek-çeyrek     sd(MC): düşük ↔ yüksek
   7  0,5369   0,6000 ↔ 0,4960   fark +0,1040     0,4987 ↔ 0,5520   fark −0,0533
  13  0,5346   0,5440 ↔ 0,4400   fark +0,1040     0,4480 ↔ 0,5413   fark −0,0933
  21  0,5289   0,5280 ↔ 0,4800   fark +0,0480     0,5227 ↔ 0,5413   fark −0,0187
  42  0,5273   0,5760 ↔ 0,4907   fark +0,0853     0,4933 ↔ 0,5813   fark −0,0880
  99  0,5333   0,5947 ↔ 0,4480   fark +0,1467     0,4800 ↔ 0,5733   fark −0,0933

u  farkı: ort +0,0976 · işaret pozitif 5/5   ← DÜŞÜK belirsizlikte model DAHA İSABETLİ
sd farkı: ort −0,0693 · işaret pozitif 0/5   ← TERS sıralıyor
```

**Okunuşu.** `u` "modelin ne kadar kanıtı var" sorusunu yanıtlıyor ve güvenilirliği
**doğru** sıralıyor: kanıtın bol olduğu çeyrekte isabet ~%57-60, kıt olduğu çeyrekte
~%44-50. MC-dropout std'si ise **beş seedin beşinde ters** sıralıyor. Teşhis de ölçüldü:

```
spearman(sd, |p−0,5|)  = +0,504     ← sd, modelin EMİN olduğu yerde BÜYÜYOR
spearman(sd, ood_zmax) = +0,534
spearman(u,  |p−0,5|)  = −0,300
spearman(u,  sd_MC)    = −0,386     ← ikisi AYNI SAYI DEĞİL
```

Yani MC-dropout std'si logit ölçeğinin bir artefaktını taşıyor (sigmoid'in eğimi p=0,5
civarında en büyüktür → std orada büyür), kanıt boşluğu ise doğrudan kanıt miktarını.

**Maliyet ÖLÇÜLDÜ (n=1.499):** `predict_evidential` **3,0 ms** ↔
`predict_with_uncertainty(T=20)` **439 ms** → **144×**. Ek başlık ana görevi bozmadı
(AUC 0,5334 → 0,5369; ECE 0,0349 → 0,0328 — ikisi de gürültü mertebesinde) ve
`deep_mlp`'de yalnız **130 parametre** ekliyor (186.702 → 186.832).

> **Bu bir KENAR iddiası DEĞİLDİR.** Ölçülen şey p'nin ayırt etme gücü değil, BELİRSİZLİK
> SIRALAMASININ doğruluğudur; AUC hâlâ ~0,53 (yazı-turaya yakın, ACI DERS #2). Kazanç
> "hangi tahminlere güvenilmez" sorusunun ucuz ve **doğru yönlü** bir cevabıdır.
> Üretime bağlanması ayrı ve bilinçli bir adımdır (bkz. §9 açık kalem 1).

---

## 5. K-uzmanlı kaynak füzyonu — Tur-35 açık kalem #4

`source_fusion` artık `extra={ad: [p,…]}` ile keyfi sayıda uzman alır. Hizalama sözleşmesi
her yeni uzman için AYNEN geçerlidir (hepsi TEK olayı yanıtlar: *"canlı sinyalin YÖNÜ
kazandı mı"*); hizasız uzunlukta uzman → sessiz kırpma değil **None**.

`update_from_predictions` iki yeni uzman topluyor: `prob_cal` (kalibre edilmiş yön
olasılığı) ve `confidence` (sinyal güveni). İkisi de zaten `direction`'a hizalıdır, çeviri
gerekmez. Bir uzmanın TEK bir eksik satırı varsa o **uzman** düşer (satır değil) — satır
kırpmak diğer uzmanların örneklemini değiştirir ve karşılaştırmayı bozardı.

**ÖLÇÜLDÜ (canlı log kopyası, 60.088 çözülmüş satır, 1.307 hizalı beşli):**

```
uzmanlar : pwin      neural    prob_cal  confidence   TABAN
log-kayıp: 0,742858  0,700429  0,693865  1,282181     0,694327
birleşim : 0,695116  ·  en iyi TEK uzman = prob_cal  ·  pişmanlık +0,00125
son ağırlıklar: pwin=0,0525  neural=0,1541  prob_cal=0,3583  confidence=0,0297  taban=0,4055
tabana karşı eşleştirilmiş Brier farkı = +0,000305 (t=+0,192, eşik −1,645)
→ TABANI_GECILEMEDI
```

Üç okuma, üçü de dürüstlük kalemi:

1. **`prob_cal` bu sistemin TABANI geçen ilk TEK uzmanıdır** (0,693865 ↔ 0,694327). Fark
   0,00046 log-kayıp = gürültü mertebesinde ve **kenar DEĞİLDİR**; ama Hedge onu fark etti
   ve ağırlığını 1/5 → 0,358'e çıkardı. Mekanizmanın çalıştığının kanıtı.
2. **`confidence` bir OLASILIK DEĞİLDİR** ve öyle okunursa felakettir: log-kayıp 1,282
   (tabanın **1,85 katı**). Bu, `confidence` alanının kusuru değil, ona olasılık muamelesi
   yapmanın kusurudur — Hedge ağırlığını 0,03'e indirerek bunu kendiliğinden söyledi.
   Pano/karar yolunda `confidence`'ın olasılık gibi okunmadığı ayrıca kontrol edilmelidir.
3. **BİRLEŞİM yine tabanı geçemedi** (t=+0,192). ACI DERS #2/#4/#9'un **dördüncü** bağımsız
   doğrulaması: kaynaklar taban oranın ötesinde kullanılabilir bilgi taşımıyor.

---

## 6. Etiketsiz ön-eğitim — Tur-35 açık kalem #3 KAPANDI (hipotez YANLIŞ çıktı)

Tur-35 şöyle yazmıştı: *"Asıl hedef `live_features_log.csv`'nin ETİKETSİZ çoğunluğudur."*
Bu tur o çoğunluğu **aradı ve bulamadı**.

`research/build_live_dataset.build_unlabeled(before=…)` eklendi (`--unlabeled` CLI'siyle):
özellik logunda satırı olan ama çözülmüş etiketi olmayan satırları döndürür. `before`
argümanı **süsleme değildir**: çözülmemiş satırlar tanımı gereği eğitim penceresinin
GELECEĞİNDEDİR; onları ön-eğitimde kullanmak, etiketsiz bile olsa, test dilimindeki özellik
dağılımını modele sızdırır (transdüktif look-ahead).

**ÖLÇÜLDÜ:**

```
özellik logu satırı        : 7.708 (benzersiz symbol|bar_ts)
çözülmüş (etiketli)        : 7.317
ETİKETSİZ toplam           :   391   (+%5,3)
ETİKETSİZ ve NEDENSEL      :    69   (+%1,5 — eğitim penceresi içinde kalanlar)
```

Nedensel A/B (canlı 1m, kronolojik bölme, `quant_transformer`, 2 seed):

```
A sıfırdan                     : OOS-AUC 0,5283 · ECE 0,0436 · Brier 0,25163
B ön-eğitim(eğitim X'i)        : OOS-AUC 0,4728 · ECE 0,0436 · Brier 0,25163
C ön-eğitim(eğitim X + 69 satır): OOS-AUC 0,4649 · ECE 0,0435 · Brier 0,25163
```

**Yapısal teşhis — asıl bulgu budur.** Etiketsiz "çoğunluk" yok, çünkü `prediction_tracker`
özellik vektörünü **YALNIZ aday sinyal barlarında** yazıyor ve adayların neredeyse tamamı
er ya da geç çözülüyor. Geriye kalan tek şey "henüz çözülmemiş en taze kuyruk"tur ve o da
tanımı gereği geleceğe bakar. Yani öz-denetimli ön-eğitimin bu sistemde işe yaramamasının
nedeni **yöntem ya da ayar değil, LOGLAMA POLİTİKASIDIR**.

Gerçek etiketsiz rezervuar, aday ÜRETİLMEYEN barlardır (30 sembol × ~390 bar/gün — bugün
biriktirilenin iki mertebe üstü). Onu toplamak canlı yazma hacmini değiştiren bir karardır
ve bu turda YAPILMADI; §9'da açık kalem olarak duruyor.

---

## 7. Optimizasyon (ölçülmüş)

### 7.1 `learning.update_from_predictions` satır okuma (ÇIKTI-DENK, 1,29× / e2e 1,10×)

Profil suçluyu net gösterdi: satır başına 11 kez çağrılan `cell(row, name)` kapanışı —
60k satırda **664.524 çağrı**, profilin **%28'i**. Her çağrı bir sözlük araması + sınır
kontrolü + `.strip()` yapıyordu. Üçü de gereksizdi:

- sütun indeksleri döngü DIŞINDA sabittir → bir kez yerel int'e çözülür,
- satır `n_col + 1` hücreye BİR KEZ normalize edilir; son hücre daima boştur → **olmayan
  sütun** dal olmadan `""` verir,
- tüketicilerin ikisi de zaten kırpar (`_f` → `float(" x ")` aynı; `_norm_dir` gövdesinde
  `.strip().upper()` var) → yalnız literal karşılaştırılan `resolved_at` kırpılır.

```
PARİTE: 62.465 satır (dejenere ekler dahil: KISA / UZUN / boşluklu / tamamen boş)
        çıkarılan alan üçlüleri → 59.949 satır, FARK 0
        AYNI donmuş logda tam `state` sözlüğü → "AYNI"
HIZ   : satır okuma 253,5 → 196,9 ms (1,29×)
        uçtan uca update_from_predictions 523,2 → 474,7 ms (1,10×)
```

> **Ölçüm hijyeni notu (dürüstlük kalemi).** İlk "önce" ölçümü 1112 ms çıkmıştı ve 2,3×'lik
> bir kazanç gibi görünüyordu — ama o ölçüm ARKA PLANDA bir eğitim işi koşarken alınmıştı.
> Geçerli sayı, iki sürümün AYNI süreçte art arda koşturulmasıyla elde edilen **1,10×**tir.
> Yük altında alınmış bir "önce" değeri, optimizasyonu olduğundan büyük gösterir.

### 7.2 Skaler normal dağılım: scipy → stdlib (53× / 29×)

**Tur-34 dersinin ÜÇÜNCÜ tekrarı:** *"sıkı bir döngüde çağrılan her GENEL-AMAÇLI yardımcı
şüphelidir."* Bu kez suçlu `scipy.stats.norm`:

```
stats.norm.cdf(skaler): 39,3 µs   ↔  NormalDist().cdf: 0,23 µs  → 171×
stats.norm.pdf(skaler): 40,8 µs   ↔  NormalDist().pdf: 0,08 µs  → 510×
stats.norm.ppf(skaler): 50,9 µs   ↔  NormalDist().inv_cdf: 0,10 µs → 509×
```

`bs_price` çağrı başına İKİ `cdf` çağırıyor ve `options_math.implied_vol` onu Brent
döngüsünde ~10 kez çağırıyordu → tek bir örtük-oynaklık hesabı yalnız **sarmalayıcıda**
~0,8 ms yakıyordu. Çözüm ek bağımlılık değil **stdlib**: `statistics.NormalDist`.

```
bs_price     :  85,5 → 1,6 µs   (53,4×)
bs_greeks    :         2,2 µs
implied_vol  : 864,1 → 29,4 µs  (29,4×)
```

**DENKLİK İDDİASI ÜÇE AYRILDI — ve bunu ölçüm belirledi, umut değil.** İlk ölçüm YALNIZ
p=0,75'te yapılmış ve *"ppf bit-denk"* sanılmıştı; ızgaraya yayılınca iddia düzeltildi:

| fonksiyon | maks. mutlak fark | tam eşit oranı | iddia |
|---|---:|---:|---|
| `pdf` | **0,000e+00** | %100 | **BİT-DENK** |
| `cdf` | 2,220e-16 | %42,2 | ≤ 1 ULP |
| `ppf` | 1,776e-15 | %31,3 | ≤ 4 ULP |

Bit-denk olmayan bir dönüşümde doğru cevap vazgeçmek değil, **iddiayı karar düzeyine
taşımaktır** (Tur-33 kuralı). Ölçüldü (4.004 nokta, iki opsiyon tipi):

```
fiyat > 1 sent      (n=3.879): maks bağıl fark 8,5e-13
fiyat > 1e-4        (n=3.917): maks bağıl fark 9,5e-11
fiyat > 1e-8        (n=3.953): maks bağıl fark 5,7e-07
fiyat > 1e-11       (n=3.963): maks bağıl fark 1,4e-04   ← "değersiz opsiyon" bölgesi
maks MUTLAK fark (tüm ızgara): 2,3e-13
implied_vol gidiş-dönüş (n=786): maks |Δσ| 9,1e-08 · ort 1,2e-10
```

Bağıl fark yalnız fiyatın kendisi ~0'a giderken büyür; orada mutlak fark 1e-13
mertebesindedir. İnsanın işlem yapabileceği her fiyatta (≥1 sent) anlaşma 8,5e-13'tür.
**DİZİ girdilerinde scipy KALDI** (`stats.norm.ppf(dizi)` orada vektörel ve hızlıdır);
değişen YALNIZ skaler çağrılardır.

### 7.3 `theoretical_param_count` kısa yolu — KENDİ eklediğimiz regresyonun onarımı (18×)

Tur-35'in dersi şuydu: *"Ölçüm bütçesini yalnız YAZAN değil OKUYAN da öder... yeni bir
kapasite basamağı eklerken 'bu sayıyı kim, ne sıklıkta soruyor?' sorusu mimarinin
parçasıdır."* Bu tur o dersi **kendi eklediği basamakta ihlal etti** ve profil yakaladı:

```
param_ladder_report(include_neural=True), ilk çağrı: 8,37 s   ← MEGA eklendikten SONRA
  ...ve MEGA satırı tek başına 1622,6 ms (ULTRA 493,7 · MoE-XL 186,3)
```

`meta` aygıtı DEPOLAMAYI kaldırdı ama Python **NESNE** maliyetini kaldırmaz: MEGA blok
başına 80 uzman × 16 blok = **1280 uzman modülü** (≈3840 `nn.Linear`) kuruyor.

Çözüm: yönlendirilen uzmanlar **şekil olarak birebir aynıdır**. Model `n_routed = top_k`
ile kurulur, eksik uzmanların parametresi **ÖLÇÜLEN** uzman boyutuyla çarpılarak eklenir
(+ router satırları). Bu bir **kapalı-form tablo DEĞİLDİR** — çarpan modelin kendisinden
okunur, uzman mimarisi değişse de sayı otomatik izler. `n_routed`/`top_k`'nın gerçek değeri
sınıfın **kurucu varsayılanından** (`inspect.signature`, model KURULMADAN) alınır → ikinci
bir sabit tablosu doğmaz.

```
DENKLİK: yedi basamağın YEDİSİNDE de sayı TAM AYNI
         (moe_xl/ultra tam kurulumla karşılaştırıldı; mega DONDURULMUŞ sayıyla)
HIZ    : mega    1622,6 → 90,2 ms  (18,0×)
         ultra    493,7 → 70,6 ms  ( 7,0×)
         moe_xl   186,3 → 67,1 ms  ( 2,8×)
         merdiven bütünü 8,37 → 6,89 s (kalan sürenin ≈3,7 s'i torch'un ilk model
         kurulumundaki tek-seferlik ısınmasıdır — bu turdan bağımsız)
```

Yani MEGA'nın panoya getirdiği maliyet **+1,6 s değil +90 ms**. Yan ürün: `_build_torch_model`
içindeki yedi dallı `if` zinciri tek bir `_tier_class` sözlüğüne indi (kısa yolun sınıfın
kurucu varsayılanlarını okuması zaten bunu gerektiriyordu; iki yerde iki eşleme tutmak
sapma kaynağı olurdu).

### 7.4 Ölçüldü ama DEĞİŞTİRİLMEDİ (dürüstlük kalemi)

Optimizasyon raporu yalnız kazanılanları yazarsa, sonraki tur aynı yerleri yeniden kazar.

- **`_source_card` (profilin %31'i).** Dört metrik (`brier_score` · `expected_calibration_error`
  → `reliability_bins` · `brier_skill_score` · `murphy_decomposition`) AYNI çift listesini
  **8-9 kez** dolaşıyor ve her elemanda `_f` çağırıyor. Tek geçişe indirmek ya dört genel
  API'ye özel bir "zaten temiz" bayrağı eklemeyi ya da ECE/Murphy formüllerini `_source_card`
  içinde KOPYALAMAYI gerektirir. İkincisi bu projenin en önemli tek-doğruluk-kaynağı
  kuralını (panodaki eğri ile özet sayı birbirini tutmalı) kırardı; kazanç ~0,1 s.
  **Yapılmadı** — kural, hızdan pahalıdır.
- **`indicator_panel`.** n=1500'de **51,9 ms** (Tur-35 sonu 70,0 ms; makine farkı dahil).
  Profilde tek bir baskın nokta YOK — süre 300'den fazla küçük pandas işlemine yayılmış
  durumda; en büyük blok `extended_rows` (%55) ve o da homojen. Tur-33/34/35 bu dosyayı üç
  kez taradı; dördüncü tarama getiri sağlamıyor.
- **`knowledge` / `selfcheck`.** `search` 0,1 ms · `glossary` 0,04 ms · `exam_knowledge(50)`
  1,9 ms. Optimize edilecek bir şey yok; ölçüldüğü buraya yazılıyor ki bir daha bakılmasın.

---

## 8. Doğrulama

- **2600 test yeşil** · `ruff check .` temiz · `mypy nihai` 96 dosyada sorun yok.
- Yeni test dosyaları: `tests/test_tur36_ai.py` (22 test — yetenek sözleşmeleri) ve
  `tests/test_tur36_parity.py` (8 test — optimizasyon paritesi + denklik SINIRI).
- Merdivenin alt altı basamağının parametre sayıları Tur-35 raporuyla **birebir aynı**.
- Canlı `deep_mlp` artefaktı yeni kodla yüklendi (`param_count=186.702`, tahmin çalışıyor).
- Canlı `data/` ve `logs/` dosyalarına YAZILMADI; tüm canlı ölçümler kopya üzerinde koştu.

---

## 9. Açık kalemler

1. **`dir_evid` üretime bağlanmadı.** `neural_shadow` hâlâ `predict_with_uncertainty(T=20)`
   kullanıyor. §4'ün ölçümü (u 5/5 doğru, sd 0/5) bir DEĞİŞİKLİK ÖNERİSİDİR, değişikliğin
   kendisi değil: canlı modelin `dir_evid` başlığıyla YENİDEN EĞİTİLMESİ gerekir ve o ayrı,
   kanıt-kapılı bir adımdır.
2. **Etiketsiz rezervuar toplanmıyor** (§6). Aday üretilmeyen barların özellik vektörünü
   loglamak, öz-denetimli ön-eğitimin tek gerçek yakıtıdır — ama canlı yazma hacmini iki
   mertebe artırır. Karar kullanıcınındır.
3. **`confidence` alanının olasılık gibi okunduğu yerler taranmalı** (§5.2): log-kaybı
   tabanın 1,85 katı. Alan bir güven SKORUdur; nerede olasılık muamelesi görüyorsa orası
   sessiz bir hatadır.
4. `quant_transformer_mega`'nın canlı modeli YOK ve planlanmıyor (eşik 26M satır, bugün
   ≈7,7k). Basamak KAPASİTE beyanıdır, kullanım önerisi değil.
5. Tur-34/35 açık kalemleri: MoE uzmanlaşmasının gerçek rejim etiketleriyle ölçümü,
   `regime_hmm`'in canlı loga bağlanması, `copula_select`in gerçek sembol çiftlerinde
   koşturulması, `pin_from_bars`in tick verisiyle çapraz-doğrulanması — bu turda da
   kapanmadı (hepsi harici veri ister).
