# AI v5 — "Derinleşme" turu (Tur-32, 2026-08-14)

**Kapsam:** `nihai/ai` katmanının quant finance / finans / uyarlama / öğrenme eksenlerinde
genişletilmesi + kapasite merdiveninin üst basamağının büyütülmesi.
**Anayasa değişmedi:** eklenen HİÇBİR şey sinyale/eşiğe/META_GATE'e/execution'a bağlanmaz.
**Kenar iddiası YOKTUR** — bu tur ölçüm ve KAPASİTE turudur, kenar arama turu değildir
(ACI DERS #3/#5).

Baseline: `af399ea` (Tur-31, 2259 test). Sonuç: **2372 test yeşil** (+113; 2353 kütüphane
turu + 19 canlı-ölçüm harness'ı), ruff temiz, mypy 158 dosyada temiz.

---

## 1. Neden bu tur — kapatılan boşluklar

Boşlukların üçü sistemin KENDİ kayıtlarında zaten yazılıydı:

| Boşluk | Kanıt (tur öncesi) | Kapatan |
|---|---|---|
| Örtük volatilite çözücüsü yok | `knowledge.py`: *"quant_math'te doğrudan bir 'IV' fonksiyonu yoktur"* | `options_math.implied_vol` |
| Gülümseme (smile) fit yok | `knowledge.py`: *"doğrudan bir smile-fit fonksiyonu YOKTUR"* | `options_math.svi_fit` / `sabr_calibrate` |
| Oynaklık hep ÖLÇÜLÜYOR, hiç ÖNGÖRÜLMÜYOR | `quant_math` 6 vol tahmincisi + GARCH; HAR yok | `microstructure.har_rv_fit` |
| Makas/etki ölçüsü yok | ACI DERS #7 üç kez "kenar ilk bardaki makas sıçramasıydı" dedi, ama etkin makası ÖLÇEN bir fonksiyon yoktu | `microstructure.roll_spread` / `corwin_schultz_spread` / `kyle_lambda` |
| Uyarlama tek eksende (kalibrasyon eşlemesi) | `adaptive_calibration` var, sürüklenme-pencere/uzman-ağırlığı/akış-öğrenimi yok | `online.adwin` / `hedge_combine` / `FTRLProximal` |
| "Brier düştü" satırı tek başına okunuyordu | Tur-25/27/28 aynı tuzağı üç kez elle çözdü | `learning.brier_skill_score` + `murphy_decomposition` (kaynak karnesinde) |

---

## 2. Yeni modüller

### 2.1 `nihai/ai/options_math.py` — opsiyon & volatilite yüzeyi

- `implied_vol` — Brent kök bulma. Newton DEĞİL: BS fiyatı σ'da kesin artan olduğu için
  parantezleme güvenlidir, Newton ise derin OTM'de vega≈0 yüzünden ıraksar.
  Arbitrajsızlık bandının **dışındaki** (ve sınırına tam oturan) kotasyonda `None` —
  uydurma σ döndürmez.
- `bs_greeks_extended` — vanna · volga · charm · veta · speed · zomma · color ·
  dual_delta · dual_gamma. **Hepsi sonlu-farkla doğrulandı** (rel < 1e-4).
  Sözleşme: zamana bağlı greekler `theta` ile AYNI eksende, ∂/∂t (takvim zamanı).
  Literatürdeki `veta`/`color` kapalı formları ∂/∂τ yönündedir; burada işaret çevrildi
  ve test bunu kilitler.
- `put_call_parity` — gap + **örtük forward** + örtük temettü. Gülümsemenin doğru çapası
  spot değil bu forward'dır.
- `svi_total_variance` / `svi_fit` / `svi_durrleman_g` / `svi_no_arbitrage` — raw SVI
  (Gatheral 2004). Fit tek başına yeterli sayılmaz: **butterfly arbitrajı Durrleman
  g(k) ≥ 0 ile sayısal olarak taranır** (kanatlar veri aralığının 1,5 katına kadar —
  arbitraj tipik olarak ekstrapole bölgede doğar). Kanat sınırı `b(1+|ρ|) ≤ 2`
  (Lee 2004 moment formülü) GEREKLİ koşul olarak ayrıca raporlanır.
- `sabr_lognormal_vol` / `sabr_calibrate` — Hagan (2002). ATM dalı ayrı ele alınır
  (z/x(z) → 1 belirsizliği; SABR uygulamalarının klasik NaN hatası). β SABİT tutulur:
  (α, β) neredeyse özdeşleşmez, ikisini birden optimize etmek kalibrasyonu keyfileştirir.
- `smile_metrics` — ATM vol, 25Δ risk-reversal, 25Δ butterfly, skew, curvature.
  25Δ strike'lar veri aralığının dışına düşerse `None` (sessiz ekstrapolasyon yok).
- `risk_neutral_density` — Breeden-Litzenberger. Düzgün OLMAYAN strike ızgarasında
  üç-nokta bölünmüş fark. **Negatif yoğunluk bastırılmaz, `n_negative` ile raporlanır.**
- `vol_term_structure` — ileri vol + **calendar arbitraj** (toplam varyans azalamaz).

**Doğrulama:** IV gidiş-dönüş 18 (parite × moneyness × vol) kombinasyonunda 1e-6;
py_vollib ile çapraz kontrol; SVI/SABR parametre geri kazanımı (rmse < 1e-5);
BL yoğunluğu BS'in kendi lognormal'ini 1e-5 hatayla geri veriyor, kütle 0,9996,
ortalama = ileri fiyat.

### 2.2 `nihai/ai/microstructure.py` — gerçekleşen oynaklık + mikroyapı

- `har_rv_fit` / `har_rv_forecast` / `har_cj_fit` — Corsi (2009) HAR-RV ve
  Andersen-Bollerslev-Diebold (2007) HAR-CJ. `log_target=True` seçildiğinde geri
  dönüşümde **yarı-varyans düzeltmesi** uygulanır (düz `exp(μ)` sistematik olarak düşük
  tahmin verir).
- `realized_variance` / `bipower_variation` / `jump_split` — sürekli↔sıçrama ayrışması.
- `two_scale_rv` — Zhang-Mykland-Aït-Sahalia (2005) gürültüden arındırılmış RV +
  gürültü varyansı tahmini.
- `volatility_signature` — RV'nin örnekleme adımına göre imzası; düşen imza = mikroyapı
  gürültüsü.
- `amihud_illiquidity` · `roll_spread` · `corwin_schultz_spread` · `kyle_lambda` ·
  `bulk_volume_classification` · `vpin`.

**Doğrulama (sentetik, bilinen parametre → geri kazanım):**

| Ölçü | Gerçek | Ölçülen |
|---|---|---|
| HAR katsayıları | (0,35 · 0,35 · 0,25) | (0,376 · 0,374 · 0,208), R² 0,988 |
| Sıçrama payı (saf difüzyon → tek dev getiri) | 0 → yüksek | 0,000 → 0,584 |
| Entegre varyans (gürültülü seri) | 6,40e-4 | ham RV 5,49e-3 (8,6× şişik) → **TSRV 6,25e-4** |
| Roll etkin makas | 0,0500 | 0,0510 |
| Kyle λ | 5,00e-4 | 4,99e-4 (R² 1,00) |
| VPIN (dengeli ↔ tek yönlü akış) | ayrışmalı | 0,12 ↔ 0,68 |

`roll_spread` pozitif otokovaryansta **`None` + `positive_autocov=True`** döner
(literatürdeki "negatifi sıfırla" alışkanlığı, ölçüm yokluğunu "makas yok" diye
raporlamak olurdu — Tur-31'in IV Rank dersi).

### 2.3 `nihai/ai/online.py` — çevrimiçi uyarlama

- `adwin` — Bifet & Gavaldà (2007). Pencereyi VERİDEN seçer; yanlış-alarm oranı δ ile
  sınırlıdır. Ölçüldü: durgun seride 0 alarm (pencere 1500'de kalıyor); t=700'de
  ortalama 0→2,5 kayan seride **değişim 707'de yakalandı**, pencere 700'e daraldı,
  pencere ortalaması 2,55.
- `hedge_combine` — üstel ağırlıklı uzman birleştirme + **fixed-share** (Herbster &
  Warmuth 1998). `include_base=True` **koşan taban oranını otomatik bir uzman olarak
  ekler** → ACI DERS #9 makinede.
  Ölçüldü: bilgili+gürültü+sabit üçlüsünde birleşim en iyi uzmana yakınsıyor
  (regret +0,0005), tabana karşı +0,406 kazanç. Rejim değişimli (t=1500'de roller yer
  değiştiren) senaryoda saf Hedge log-kayıp **0,9165**, fixed-share (α=0,02) **0,2347**.
- `FTRLProximal` / `ftrl_prequential` — McMahan ve ark. (2013). Tek geçişli, koordinat-başı
  uyarlanır, L1 GERÇEK sıfır üretir. `uplift` alanı koşan tabana karşı farkı verir.
  Ölçüldü: öğrenilebilir veride uplift **+0,206** (ağırlıklar (1,72 · −0,81 · ~0 · ~0 · 0,34)
  ↔ gerçek (1,5 · −1 · 0 · 0 · 0,5)); **saf gürültüde uplift −0,014** (bilgi icat etmiyor).

---

## 3. `learning.py` — beceri skorları

Eklenenler: `log_loss_score`, `brier_skill_score` (BSS), `murphy_decomposition`
(güvenilirlik / **çözünürlük** / belirsizlik).

Kaynak karnesi (`sources.*`) artık **beş** alan taşır: `brier · ece · bss · resolution · n`,
ve rapor satırının altına şu uyarı basılır:

> (BSS ≤ 0 = sabit taban-orandan iyi DEĞİL · RES ≈ 0 = taban ötesi bilgi YOK — ACI DERS #9)

Bu, Tur-25/27/28'de ÜÇ KEZ elle yapılan kontrolün kalıcı hâlidir: "Brier 0,25'ten 0,249'a
düştü" satırı artık tek başına okunamaz.

---

## 4. Kapasite: `quant_transformer_moe` (≈93M parametre)

Merdivenin yeni üst basamağı **seyrek uzman karışımıdır** (MoE; Shazeer 2017 / Fedus 2021):
her blokta 6 uzman SwiGLU, token başına **top-2** yönlendirme. Dizi düzeni ([CLS] + 3×[SCALE]
+ özellikler) ve `nn.MultiheadAttention` XL ile AYNI — `explain.attention_rollout` sözleşmesi
korunur; TEK değişken FFN'dir.

**Neden yoğun (dense) genişletme değil:** Tur-22 §7-1'de ÖLÇÜLDÜ — XL süpürmesi bu donanımda
21 gün. Yoğun büyütme parametreyle birlikte hesabı da büyütür ve merdivenin üst basamağını
fiilen erişilemez kılar. MoE kapasiteyi hesaptan ayırır.

**ÖLÇÜLDÜ** (f=96, 256×96 ileri geçiş, CPU):

| | toplam parametre | aktif parametre | ileri geçiş |
|---|---|---|---|
| `quant_transformer_xl` | 31.756.051 | 31.756.051 | 4.709 ms |
| `quant_transformer_moe` | **93.228.307** | 36.494.611 | 5.883 ms |
| oran | **2,94×** | 1,15× | **1,25×** |

Yani ≈3 kat kapasite, ≈1,25 kat maliyet. Aradaki fark (1,15 → 1,25) yönlendirme/dağıtma
ek yüküdür — teorik oran değil ÖLÇÜLEN oran raporlanır.

Yük dengeleme (`_MOE_BALANCE_WEIGHT = 0.01`, Switch aux loss) **zorunludur**: onsuz
yönlendirici tek uzmana çöker ve kapasite artışı kâğıt üstünde kalır. Kayıp terimi yalnız
model çıktısında `moe_balance` anahtarı VARSA eklenir → diğer basamaklarda davranış v4 ile
bit-denk. Yönlendirme argmax tabanlıdır (gürültülü kapı YOK) → determinizm korunur.

Merdivenin tamamı (f=96): 97 → 4.657 → 2.810 → 8.321 → 223.054 → 5.465.363 → 31.756.051 →
**93.228.307**.

`recommend_tier` eşiği **n ≥ 1.200.000** (400k × 93/30; oran kuralının devamı). Eşik TOPLAM
parametreye göredir, aktife göre değil: aşırı-uyum riskini taşıyan şey ezberlenebilir
kapasitedir.

> **KAPASİTE ARTIŞI BİR KENAR İDDİASI DEĞİLDİR (ACI DERS #5).** Bu basamağın canlı bir
> modeli YOKTUR, promotion yine kanıt-kapılı ayrı yoldadır ve bu turda hiçbir kenar
> ölçümü yapılmamıştır.

---

## 5. Bağlantılar (yeni yetenek nereye dokundu)

- `nihai/ai/__init__.py` — üç yeni modül lazy listeye eklendi; `import nihai.ai` ucuz kaldı.
- `knowledge.py` — "uygulaması YOK" diyen üç tuzak satırı gerçeğe göre düzeltildi;
  `implied_vol` → `implied_vol`, `vol_smile` → `svi_fit` köprüsü kuruldu.
- **Formül köprüsü genişledi**: `formul_ref` adları artık üç modülde aranıyor
  (`quant_math` · `options_math` · `microstructure`). `selfcheck.exam_knowledge` ve
  `tests/test_ai_knowledge.py` aynı üçlüyü kullanıyor — tek doğruluk kaynağı
  `selfcheck._FORMULA_MODULES`.
- `selfcheck._sayisal_sorular` 12 → **17 soru**: IV gidiş-dönüş, SABR ATM (β=1) limiti,
  SVI minimum noktası, gerçekleşen varyans, Roll makası kapalı formu (ddof=1 çarpanı
  açıkça beklenen değerde — tahminci sessizce ddof=0'a kayarsa soru yanlış olur).
- `models.NEURAL_TIERS` / `param_ladder_report` / web panosunun Beyin sekmesi yeni
  basamağı otomatik gösterir (rapor mimariden ÖLÇER, tablo tutmaz).

## 6. Test

| dosya | yeni test |
|---|---|
| `tests/test_ai_options_math.py` | 47 |
| `tests/test_ai_microstructure.py` | 18 |
| `tests/test_ai_online.py` | 17 |
| `tests/test_ai_learning.py` (ek) | 7 |
| `tests/test_ai_neural.py` (ek) | 6 |

Her mekanizma İKİ senaryoda sınandı: yakalaması gereken durum **ve** yakalamaMASI gereken
durum. İkincisi bu projede daha önemlidir — `test_hedge_tum_uzmanlar_gurultuyse_taban_orani_gecemez`
ve `test_ftrl_gurultude_taban_orani_gecemez` uyarlama mekanizmalarının olmayan kenarı VAR
göstermediğini kilitler.

## 7. İLK CANLI ÖLÇÜM (2026-08-14, aynı gün eklendi)

İki yeniden-koşulabilir harness yazıldı ve GERÇEK veriyle çalıştırıldı. İkisi de KARAR-DIŞI.

### 7.1 `research/option_surface_probe.py` — opsiyon yüzeyi

Hat: DoltHub `option_chain` (gerçek bid/ask) → **parite regresyonuyla** örtük forward
(`C−P = e^{−rt}(F−K)`; F ve r zincirin KENDİSİNDEN çıkarılır, dışarıdan spot/faiz varsayımı
GİRMEZ) → `implied_vol` → `svi_fit` + Durrleman → `smile_metrics` → `risk_neutral_density`.

**SPY @ 2026-08-13 (4 vade, 196 kotasyon):**

| vade | t (yıl) | parite R² | F | IV bizim↔satıcı (OTM) ort\|Δ\| | SVI rmse | Durrleman min | ATM | RR25 | skew |
|---|---|---|---|---|---|---|---|---|---|
| 08-28 | 0,0411 | 0,99990 | 778,64 | **0,00212** | 8,1e-5 | +0,0035 | 0,1076 | −0,0167 | −0,561 |
| 09-11 | 0,0795 | 0,99994 | 780,06 | **0,00076** | 8,1e-5 | −0,0082 | 0,1185 | −0,0258 | −0,538 |
| 09-25 | 0,1178 | 0,99997 | 779,77 | **0,00046** | 5,9e-5 | −0,0105 | 0,1272 | −0,0332 | −0,523 |
| 09-30 | 0,1315 | 0,99997 | 779,98 | **0,00036** | 3,3e-5 | −0,0044 | 0,1281 | −0,0342 | −0,510 |

Vade yapısı: contango, **calendar arbitrajı YOK** (0 ihlal), eğim 0,2274.

**Ne öğrenildi (üç bulgu, ikisi harness'ı DEĞİŞTİRDİ):**

1. **Kendi IV çözücümüz satıcının IV'siyle 3,6–21 baz puan içinde örtüşüyor** (OTM'de).
   Bu, sentetik değil GERÇEK piyasa kotasyonlarında bağımsız doğrulamadır.
2. **Derin ITM kotasyonlar IV taşımaz.** İlk koşumda fark 0,0055–0,0099 görünüyordu.
   Sebep ölçüldü: r'yi −0,07'den +0,04'e taşımak tüm zincirin ortalama IV'sini
   0,180→0,211 oynatıyor, ama kayma tamamen derin ITM'den geliyor (orada vega≈0, iskonto
   faktöründeki %1'lik hata IV'de puanlarca saçmalığa dönüşüyor). Karşılaştırma OTM'e
   kısıtlandı → fark **0,00036–0,0021**'e düştü.
3. **r kısa vadede KÖTÜ KOŞULLU, F değil.** Parite regresyonu F'yi R²≈0,9999 ile veriyor
   ama r negatif çıkıyor (SPY t=0,041y'de −0,16). Bu bir hata değil: `r = −ln(disc)/t`
   olduğu için mid'deki ‰1'lik gürültü kısa vadede r'yi puanlarca savurur. Rapora
   `r_guvenilir` bayrağı eklendi (t ≥ 0,08y ve |r| ≤ 0,15).

**AAPL @ 2026-08-13** aynı hattan geçirildi ve **beklenen yapısal farkı gösterdi**:
skew −0,008…+0,159 (SPY: −0,51…−0,56). Endeks-skew'ün tek-isim skew'den çok daha dik
olması opsiyon piyasasının en bilinen düzenliliğidir; kütüphane onu bağımsız olarak
yeniden üretiyor. OTM IV farkı 0,0036–0,021 (SPY'dan gevşek — daha geniş makas, temettü,
Amerikan erken kullanım). SPY'da 4 dilimden 3'ü butterfly-arbitrajlı çıkarken AAPL'de
3 dilimin 3'ü de arbitrajsız.

### 7.2 `research/realized_vol_probe.py` — gerçekleşen oynaklık

Hat: Alpaca saatlik bar → günlük RV/BV/sıçrama → **yürüyen (walk-forward) HAR-RV** →
üç tabana karşı OOS beceri. Kayıp ölçüsü **QLIKE** (Patton 2011).

**SPY, 2024-08-01→2026-08-12, 8.142 saatlik bar → 509 gün, n_OOS = 388:**

| tahminci | QLIKE | MSE |
|---|---|---|
| **HAR** | **0,44919** | 1,065e-7 |
| MA22 (en iyi taban) | 0,56252 | 1,274e-7 |
| RW | 0,74232 | 1,609e-7 |
| SABİT | 0,82661 | 1,313e-7 |

**HAR becerisi: QLIKE +0,2015 · MSE +0,1645 → en iyi triviyal tabanı GEÇİYOR.**
RV ort 1,104e-4 (yıllık ≈%16,7), sıçrama payı ort 0,224.

> Bu **KENAR DEĞİLDİR** (ACI DERS #3). Oynaklık finansın en öngörülebilir büyüklüğüdür ve
> bu ölçüm YÖN hakkında hiçbir şey söylemez. Değeri risk/boyutlandırma tarafındadır.

**Likidite tahmincileri 4 sembolde doğru sıralıyor** (günlük bar, 510 gün):

| sembol | Amihud (1M$ başına) | Corwin-Schultz | Roll |
|---|---|---|---|
| SPY | 1,7e-7 | 0,00237 | 3,4866 |
| AAPL | 8,9e-7 | 0,00474 | tanımsız (pozitif otokov.) |
| F | 2,0e-5 | 0,00533 | tanımsız |
| PLUG | 3,4e-4 | 0,02087 | 0,04231 |

Amihud SPY→PLUG arasında **2000×** artıyor; sıralama monotone ve beklenen likidite
sırasıyla birebir. Roll iki sembolde `None` döndü (pozitif otokovaryans) — uydurma sıfır
YERİNE ölçüm yokluğunu söyledi, tasarım gereği.

### 7.3 Ölçüm sırasında bulunan durum değişikliği

`ALPACA_API_KEY` artık **DOLU** ve çalışıyor (2026-08-13 notu "eksik → sessizce yfinance'e
düşüyor" diyordu). Alpaca saatlik/günlük geçmiş doğrudan erişilebilir durumda.

## 8. Açık kalemler

1. ~~Opsiyon hattının gerçek zincirde koşulması~~ → **§7.1'de YAPILDI.**
2. ~~HAR-RV'nin canlı barlarla OOS değerlendirmesi~~ → **§7.2'de YAPILDI.**
3. MoE yönlendirmesinin rejime göre uzmanlaşıp uzmanlaşmadığı ÖLÇÜLMEDİ (mimari olanak
   sağlıyor; kanıt ayrı iş).
4. `online.hedge_combine`'ın canlı `sources` (pwin/neural/prior) üçlüsü üzerinde
   koşturulması — `decision.py`'nin sabit Brier-ağırlıklarına karşı A/B. Yapılmadı;
   yapılırsa ölçüm KARAR-DIŞI kalmalı.
5. SPY'ın 4 diliminden 3'ünde bulunan butterfly arbitrajının kaynağı ayrıştırılmadı:
   (a) EOD kotasyonların asenkronluğu, (b) Amerikan erken-kullanım primi, (c) gerçek
   arbitraj. Ayırt etmenin yolu gün-içi kotasyonlar veya arbitrajsız-SVI (SSVI) kalibrasyonu.
6. ~~Corwin-Schultz saatlik barda anlamsız (tahminci GÜNLÜK bar için tasarlandı; saatlikte
   negatif oran %36). Kullanım notu koda değil rapora yazıldı; istenirse guard eklenebilir.~~
   → **Tur-33'te KAPANDI (BULGU-33-4).** Guard eklendi çünkü bu bir kullanım notu değil,
   Tur-31 IV Rank dersinin tekrarıydı: sıfır-kırpma simetrik gürültüden bile POZİTİF sayı
   üretiyor (gerçek makası 0 olan seride `spread_raw = −0,000607` iken kırpılmış
   **+0,000140**). Artık `roll_spread` ile aynı sözleşme: yansız tahmin ≤ 0 → `spread`
   **None** + `reliable=False`; `spread_clipped`/`negative_share` ayrıca raporlanır.
   Ayrıntı: `docs/TUR33_RAPORU_2026-08-15.md`.
7. **Commit edilmedi** — çalışma ağacı `af399ea` üstünde duruyor.
