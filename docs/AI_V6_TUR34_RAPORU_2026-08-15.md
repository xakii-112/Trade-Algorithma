# AI v6 — "Belirsizlik ve Kapasite" (Tur-34, 2026-08-15)

**Kapsam:** bir YENİ modül + dört mevcut modülde genişleme + ÜRETİM yolunda ölçülmüş
optimizasyon. Hepsi **KARAR-DIŞI**; hiçbiri kenar iddiası taşımaz.

**Doğrulama:** 2459 test yeşil (Tur-33 sonu 2427 → +32 yeni) · `ruff check .` temiz · `mypy nihai` 96 dosya temiz.
Ağ erişimi yok; canlı `data/`, `logs/` dosyalarına yazılmadı.

---

## 1. Neden bu tur — kapatılan boşluklar

Tur-32'nin yöntemi aynen sürdürüldü: **yeni yetenek icat etmek yerine, sistemin KENDİ
belgelerinde "burada YOKTUR" yazan satırlar** kapatıldı. Dört kaynak:

| Kaynak | Satır | Kapanış |
|---|---|---|
| `knowledge.py` → `student_t_copula.tuzaklar` | *"asimetrik kopulalar (ör. Clayton/Gumbel) bu modülde YOKTUR"* | `quant_math` arşimedyen aile |
| `knowledge.py` → `vpin.tuzaklar` | *"Bu modülde doğrudan bir PIN tahmincisi YOKTUR"* | `microstructure.pin_ekop` |
| `knowledge.py` → `rejim.tuzaklar` | *"geçiş OLASILIĞI ... bilgisi mevcut modülde yoktur"* | `nihai/ai/regime_hmm.py` |
| Tur-32 raporu açık kalem #3 ve #5 | MoE uzmanlaşması ölçülmedi · SVI arbitrajının kaynağı ayrıştırılmadı | `neural.moe_routing_stats` · `options_math.ssvi_diagnose` |

Ayrıca kullanıcı isteği gereği kapasite merdiveninin üst basamağı büyütüldü (§5).

---

## 2. `nihai/ai/regime_hmm.py` — rejim GEÇİŞ OLASILIKLARI (YENİ)

Sistem rejimi bugüne kadar hep KESKİN bir etiketle okudu (`classify_regime` → `"up_high"`).
Oysa rejim gözlenmez, **çıkarsanır**; keskin etiket belirsizliği sessizce sıfıra yuvarlar ve
en çok rejim SINIRLARINDA yanılır — tam da riskin arttığı yerde.

Gauss saklı Markov modeli (Baum-Welch) üç şeyi birden verir: bar başına rejim OLASILIK
DAĞILIMI, rejimler arası GEÇİŞ matrisi, rejim başına beklenen SÜRE.

### 2.1 Look-ahead sınırı — modülün en kritik sözleşmesi

| Fonksiyon | Nedensel mi | Kullanım |
|---|---|---|
| `hmm_filter` | ✅ `causal=True` | **Canlı yolda kullanılabilecek TEK çıktı** |
| `hmm_smooth` | ❌ `causal=False` | Geçmiş etiketleme/analiz |
| `hmm_viterbi` | ❌ `causal=False` | En olası yol (tüm seriye bakar) |
| `hmm_next_state_probs` | ✅ `causal=True` | P(S_{T+h} | x_1..T) — **boşluğu kapatan çıktı** |

Bayrak makine-okunurdur; çağıran yanlışlıkla nedensel sanamaz. `gaussian_hmm_fit`
parametreleri tüm örneklemden tahmin eder (iç-örneklem) ve bu docstring'de açıkça yazılıdır.

### 2.2 Ölçüldü — bilinen parametreleri geri buluyor

Sentetik iki-rejim (T=2000, A = [[0,98, 0,02], [0,05, 0,95]], σ = 0,006 / 0,020):

```
σ geri kazanımı      : 0,00593 ↔ 0,006   ·  0,02014 ↔ 0,020
kalıcılık (köşegen)  : 0,9799 ↔ 0,98     ·  0,9372 ↔ 0,95
durağan dağılım      : 0,757  ↔ 0,714 (teorik)
beklenen süre        : 49,7 bar · 15,9 bar
nedensel FİLTRE isabeti  : %94,5
Viterbi (look-ahead) isab: %97,05     ← geleceği gören daha isabetli olmalı ✓
BIC durum seçimi     : K=2, decisive=True
```

### 2.3 Devralınan okuma kuralları

- **`mode_prob` okunmadan `mode` okunamaz** — 0,34'lük bir tepe rejim ilan etmez
  (Tur-33'ün "bir ORAN, paydası okunmadan okunamaz" kuralı).
- **ΔBIC < 2 → `decisive=False`** — bu mertebede BIC modelleri ayırt EDEMEZ.
- **σ tabanı ZORUNLUDUR** — bir durum tek gözlemin üstüne çökerse olabilirlik sonsuza gider
  ve EM o sahte optimuma kilitlenir (Gauss karışımlarının klasik dejenere çözümü).
- **İlklendirme quantile tabanlı, `seed` argümanı YOK** — anayasa madde 6.

---

## 3. `quant_math` — ASİMETRİK (arşimedyen) kopulalar

Gauss kopulası iki kuyrukta da bağımlılığı SIFIR, t kopulası SİMETRİK varsayar. Gerçek hisse
getirilerinde düşüş kuyruğu daha kalındır: ortak çöküş ortak ralliden olasıdır.

- `copula_fit_clayton` — λ_L = 2^(−1/θ) > 0, λ_U = 0 (ALT kuyruk)
- `copula_fit_gumbel` — λ_U = 2 − 2^(1/θ) > 0, λ_L = 0 (ÜST kuyruk)
- `copula_tail_dependence` — dört aile için KAPALI FORM λ (ampirik `tail_dependence`in model karşılığı)
- `copula_select` — dört aile, AIC ile sıralama

θ Kendall tau tersinmesiyle başlatılır, 1B sınırlı MLE ile inceltilir (deterministik Brent).

**Ölçüldü** (θ=2 ile üretilmiş 3000 nokta):

```
θ̂ = 2,0046   (gerçek 2,0)        τ̂ = 0,5081  (gerçek 0,5)
λ̂_L = 0,7077 (teorik 0,70711)    ampirik λ_L = 0,713
AIC:  clayton −2605  ·  t −2004  ·  gaussian −1925  ·  gumbel −1365
seçim: clayton, decisive=True, ΔAIC = 600,8
```

**Dürüstlük kapıları:** (a) τ ≤ 0'da arşimedyen aile UYDURULMAZ, tabloya `None` girer —
"uymadı" ile "denenmedi" ayrımı korunur; (b) ΔAIC < 2 → `decisive=False`; (c) ampirik λ bir
KONTROLdür, kapı değil (sonlu örneklemde kuyruk tahmini çok gürültülüdür).

---

## 4. `microstructure` — PIN + `options_math` — SSVI

### 4.1 PIN (Easley-Kiefer-O'Hara-Paperman 1996)

`roll_spread`/`corwin_schultz_spread` makasın BÜYÜKLÜĞÜNÜ ölçer; PIN o makasın NEDENİNİ
(ters seçim payını) ölçer — ACI DERS #7'nin teşhis takımının eksik parçası.

Olabilirlik **Lin & Ke (2011) çarpanlamasıyla** kurulur: naif form ε^B çarpanları yüzünden
B,S birkaç yüzü geçince taşar (literatürde belgelenmiş sayısal çöküş).

**Ölçüldü** (α=0,4 · δ=0,5 · μ=60 · ε_b=40 · ε_s=45, n=400):

```
α̂ 0,3926 · δ̂ 0,4649 · μ̂ 58,58 · ε̂_b 40,44 · ε̂_s 44,87
PIN 0,2123 ↔ gerçek 0,2202
ölçek değişmezliği: ×3 sayımda PIN 0,2123 (fark < 1e-6)
bilgisiz akışta (μ=0) PIN < 0,15 ✓
```

`pin_from_bars` OHLCV vekilidir ve **`proxy=True` bayrağı taşır**: Poisson varsayımı işlem
SAYISI üzerinedir, hacim aşırı-dağılımlıdır. Gerçek tick PIN'i ile aynı şey değildir.

### 4.2 SSVI — Tur-32 açık kalem #5 KAPANDI

Soru şuydu: SPY'ın dilimlerinde bulunan kelebek arbitrajı **uyum artefaktı mı, veride mi?**

`ssvi_fit` (Gatheral-Jacquier 2014) yüzeyi arbitrajsız bölgeye KISITLAR (γ ≤ ½,
η(1+|ρ|) ≤ 2, θ_t artan) — arbitrajsızlık bir SONUÇ değil bir KISITTIR. `ssvi_diagnose` iki
bağımsız kalibrasyonu aynı veriye uygular ve verdikt üretir:

| Durum | Verdikt |
|---|---|
| raw SVI ihlal bulmuyor | `ARBITRAJ_YOK` |
| SVI buluyor, SSVI benzer hatayla uyduruyor (oran ≤ 2) | `UYUM_ARTEFAKTI` |
| SVI buluyor, SSVI belirgin daha kötü | `VERIDE_ARBITRAJ` |
| kısıtlı fit çözüm bulamıyor | `BELIRSIZ` |

**Ölçüldü — dört senaryo da doğru ayrıldı:**

```
temiz SSVI yüzeyi   : ARBITRAJ_YOK    (parametre geri kazanımı rmse 1,17e-08)
gürültülü (σ=0,006) : UYUM_ARTEFAKTI  (1 ihlal, rmse oranı 1,16)
dik kanat (b(1+|ρ|)>2): VERIDE_ARBITRAJ (3 ihlal, rmse oranı 2,7e+06)
takvim ihlali       : BELIRSIZ        (yanlış verdikt yerine "ölçemedim")
```

Verdikt kaynağı SÖYLEMEZ: asenkron EOD kotasyonu, Amerikan erken-kullanım primi ve gerçek
arbitraj birbirinden ayrılmaz — ama hepsi **"fit hatası" şıkkından ayrılır.** Sorulan soru
buydu.

---

## 5. Kapasite: `quant_transformer_moe_xl` (≈284M parametre)

Merdivenin yeni üst basamağı. Tur-32 MoE'sinden iki YAPISAL fark (DeepSeekMoE, Dai ve ark. 2024):

1. **İnce tane** — uzman gizli genişliği yarıya iner (1536→1024), uzman sayısı 6→16 çıkar.
   Aktif bütçe sabitken seçilebilen uzman KOMBİNASYONU C(6,2)=15'ten **C(16,2)=120**'ye
   çıkar. Yönlendiricinin ifade gücü buradan gelir — parametreden değil.
2. **Paylaşımlı uzman** — her token'da KOŞULSUZ çalışan bir uzman ortak bilgiyi taşır;
   yönlendirilen uzmanlar artığı öğrenmekte serbest kalır. Onsuz her uzman aynı ortak
   bilgiyi ayrı ayrı ezberler ve uzmanlaşma gerçekleşmez.

Ek olarak **ST-MoE router z-kaybı** (Zoph ve ark. 2022): yük dengeleme dağılımın DÜZLÜĞÜNE,
z-kaybı logitlerin BÜYÜKLÜĞÜNE bakar — farklı patolojiler, biri diğerinin yerine geçmez.

**ÖLÇÜLDÜ (f=96):**

| basamak | parametre | önceki basamağa oran |
|---|---:|---:|
| `deep_mlp` | 223.054 | — |
| `quant_transformer` | 5.465.363 | 24,5× |
| `quant_transformer_xl` | 31.756.051 | 5,81× |
| `quant_transformer_moe` | 93.228.307 | 2,94× |
| **`quant_transformer_moe_xl`** | **283.639.571** | **3,04×** |

`recommend_tier` eşiği aynı oran kuralının üçüncü uygulaması: 1,2M × 3 = **3.600.000**.
Bu eşik bugünkü canlı logdan (~6k satır) **üç mertebe** uzaktır ve bu BİLİNÇLİDİR —
kapasite merdiveni veri disiplinini gevşetmek için değil, gevşetmediğini GÖSTERMEK için uzar.

> **Kapasite artışı bir KENAR iddiası DEĞİLDİR (ACI DERS #5). Bu basamağın canlı modeli YOKTUR.**

### 5.1 `moe_routing_stats` — Tur-32 açık kalem #3 KAPANDI

Yük dengeleme kaybının DEĞERİNİ raporlamak, işe yarayıp yaramadığının kanıtı değildir. Yeni
fonksiyon iki ayrı soruyu ayrı ayrı ölçer:

- **Çökme var mı?** (tüm token'lar) uzman kullanım payı · normalize Shannon entropisi ·
  `max_share` · `collapsed` (tek uzman > %50).
- **Uzmanlaşma var mı?** (`groups` verilirse) [CLS] token'ının yönlendiği uzman ×
  grup tablosunda ki-kare + Cramér V.

Ölçüm koşulu TAŞINIR: beklenen sayımların >%20'si 5'in altındaysa ki-kare geçersizdir →
`reliable=False`, **p döndürülmez**. [CLS] tek uzmana gidiyorsa `measured=False` +
`reason="CLS_TEK_UZMAN"` — "uzmanlaşma yok" DEĞİL, "karşılaştırılacak ikinci hücre yok".
MoE olmayan basamakta dürüstçe `None`.

Teşhis bayrağı varsayılan KAPALIdır ve ölçüm sonunda geri alınır (açık kalsa her canlı
çıkarım blok başına int64 tamponu biriktirirdi).

---

## 6. Optimizasyon — ölçülmüş, denkliği kanıtlanmış

### 6.1 Yeni modüller

Profil tek bir suçlu gösterdi: **`scipy.special.logsumexp`in çağrı başına maliyeti.**
`gaussian_hmm_fit` T=2000'de **86.021 kez** çağırıyordu ve sürenin neredeyse tamamı
array-API gönderimi + dtype yükseltme sarmalayıcısındaydı — matematikte DEĞİL.

```
(2,2) dizide çağrı başına:  scipy 151,3 µs · yerel _lse 20,9 µs · açılmış 11,2 µs
```

Üç adımda çözüldü: (1) yerel `_lse`, (2) skaler kaydırma + matris-vektör çarpımı,
(3) Rabiner (1989 §V.A) ÖLÇEKLEME şeması — döngü içinde `max`/`exp` KALMAZ.

| fonksiyon | ÖZGÜN | YENİ | kazanç |
|---|---:|---:|---:|
| `_forward` (T=2000) | 294,7 ms | 15,6 ms | **18,9×** |
| `_backward` (T=2000) | 294,0 ms | 15,6 ms | **18,8×** |
| `hmm_filter` (T=2000) | 893,7 ms | 24,9 ms | **35,9×** |
| `gaussian_hmm_fit` | — | 743 ms | (2034 ms → 743 ms, uçtan uca 2,7×) |
| `pin_ekop` (n=400) | 600,9 ms | 200,6 ms | **3,00×** |

`pin_ekop` ayrıca **kaba eleme → ince çözüm**e geçti: 18 başlangıcın tamamı kısa bütçeyle
taranır, yalnız en iyi 3'ü tam bütçeyle çözülür. Determinizm korunur (ızgara sabit, sıralama
sabit). Üç bağımsız sentetik sette **PIN farkı ≤ 4,4e-06** (optimizatör yakınsama gürültüsü).

### 6.2 ÜRETİM yolu — `indicator_panel`

Profil `nvi`'yi panelin en pahalı tek göstergesi olarak gösterdi: bar başına `pct.iloc[i]` +
`vol_dn.iloc[i]` (pandas skaler indeksleme). Özyineleme bir ÇARPIM birikimidir → `np.cumprod`
ile **aynı sırada**. `supertrend`/`kama` özyinelemelidir, vektörleştirilemez; döngü GÖVDESİ
ucuzlatıldı (`np.isnan(x)` → IEEE denkliği `x != x`, tekrar indeksleme yerine yerel değişken).

| fonksiyon | ÖZGÜN | YENİ | kazanç |
|---|---:|---:|---:|
| `nvi` (1500 bar) | 9,61 ms | 0,37 ms | **26,0×** |
| `supertrend` | 12,01 ms | 2,43 ms | **4,93×** |
| `kama` | 3,08 ms | 1,33 ms | **2,31×** |
| **`indicator_panel`** | **139,7 ms** | **85,1 ms** | **1,64×** |

### 6.3 Denklik iddiaları — ÖLÇÜLDÜ

Tur-33 kuralı: *"bit-denk" iddiası ölçülebilir olmalı; olmuyorsa iddia değiştirilir.*

**Üretim yolu — BİT-DENK** (300 rastgele df; üç fiyat tabanı × üç oynaklık × üç sürüklenme):

```
nvi · supertrend · direction · kama   maks mutlak fark : 0,000e+00
NaN deseni farkı                      : 0
```

**HMM — operasyonel denk** (toplama sırası değişti, bit-denk OLAMAZ):

```
log_alpha bağıl fark      : 1,5e-14
loglik    bağıl fark      : 1,5e-14
filtreli olasılık farkı   : 5,6e-16
KARAR (argmax) farkı      : 0 / 2000
```

İkisi de `tests/test_ai_regime_hmm.py` ve mevcut panel testleriyle kilitli.

---

## 7. Yeni dersler

1. **Bir sarmalayıcının maliyeti, sardığı matematiğin 13 katı olabilir.** `scipy.logsumexp`
   (2,2) bir dizide 151 µs harcıyor; aynı aritmetik elle yazılınca 11,2 µs. Tur-33 bunu
   `spearman`'da (pencere başına `pd.Series`) öğrenmişti; ders MODÜLE değil DESENE aittir —
   **sıkı bir döngüde çağrılan her genel-amaçlı yardımcı fonksiyon şüphelidir.**
2. **Özyineleme "vektörleştirilemez" demek "optimize edilemez" demek değildir.** `supertrend`
   ve `kama` sıralı kalmak zorundaydı; gövdedeki `np.isnan` → `x != x` ve yerel değişken
   ile 4,9× ve 2,3× kazanıldı, tek bir aritmetik işlem değişmeden.
3. **Bir çarpım birikimi cumprod'dur ve BİT-DENKTİR; bir toplam birikimi değildir.**
   `nvi` 26× hızlandı ve farkı TAM SIFIR çıktı çünkü soldan sağa çarpım sırası korundu.
   Tur-33'ün `wma`/`_linreg` durumunda sıra değişiyordu ve iddia "operasyonel denk"e
   çekilmişti. **Hangi iddianın kurulabileceğini algoritma belirler, umut değil.**
4. **Ölçüm KOŞULU, ölçümün kendisi kadar taşınmalıdır.** `causal=False` · `proxy=True` ·
   `reliable=False` · `measured=False` + `reason` · `decisive=False` — bu turda eklenen beş
   bayrağın hepsi aynı işi yapar: sayıyı okuyanın onu ne sanabileceğini engeller.
5. **Bir aile seçimi bir MODEL KARARIDIR.** "Düşüş kuyruğu daha kalın, o hâlde Clayton"
   demek varsayımı veriye sormadan dayatmaktır. `copula_select` ve `hmm_select_states`
   sorar — ve ΔAIC/ΔBIC < 2'de "ayırt edemedim" der.

---

## 8. Açık kalemler

1. MoE yönlendirmesinin gerçek rejim etiketleriyle uzmanlaşıp uzmanlaşmadığı — ARAÇ artık
   var (`moe_routing_stats`), ama EĞİTİLMİŞ bir MoE modeli üzerinde canlı veriyle
   koşturulmadı (bu basamağın canlı modeli yok).
2. `regime_hmm`'in canlı `prediction_tracker` verisine bağlanması — eğitim penceresinde fit
   + `hmm_filter` ile ileri uygulama. Yapılırsa ölçüm KARAR-DIŞI kalmalı.
3. `copula_select`in gerçek sembol çiftlerinde koşturulması (sektör içi ↔ sektör arası
   kuyruk asimetrisi). Portföy tarafına bağlanması ayrı ve bilinçli bir adımdır.
4. `pin_from_bars`in gerçek tick verisiyle çapraz-doğrulanması — vekil ile yapısal PIN
   arasındaki sapma ÖLÇÜLMEDİ (bayrak var, ölçüm yok).
5. Tur-32 açık kalem #4 (`online.hedge_combine` × canlı `sources` üçlüsü A/B) bu turda da
   YAPILMADI.
6. `hmm_select_states` (2,3,4) T=2000'de 5,8 s — K=3/4 fitleri EM iterasyonlarının
   çoğunluğunu tüketiyor. Kabul edildi: model seçimi seyrek bir işlemdir.
