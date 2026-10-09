# Tur-25 — AI Katmanı Derinleştirme: Tasarım Kaydı ve Ölçülmüş Taban

**Tarih:** 2026-08-10 · **Taban commit:** `81c9d01` (Tur-24) · **Planlama:** Opus 5 (orkestratör)
**Kodlama:** paralel ajanlar (WP-A/WP-B: Sonnet 5 · WP-C: Opus)

Bu belge turun **kararlarını ve gerekçelerini** kaydeder. Sonuçlar `OPTIMIZASYON_LOG.md`
"Ek Tur 25" bölümüne ve ilgili SONUC raporlarına yazılır.

---

## 0. Taban YERİNDE ölçüldü (rapordan okunmadı)

Tur-12'nin dersi gereği ("baseline'ı rapordan değil YERİNDE ölç") turun başında her sayı
komut koşularak alındı:

| Ölçüm | Sonuç | Nasıl |
|---|---|---|
| Çalışma ağacı | temiz, `81c9d01` | `git status --short` → 0 satır |
| Test paketi | **exit 0 / YEŞİL** | `pytest -q` |
| Ortam | Python 3.12.10 · torch 2.13.0+cpu · **CUDA yok** | `torch.cuda.is_available()=False` |
| Servisler | **KAPALI** (`logs/project.stopped` var) | dosya kontrolü |
| `predictions_log.csv` | 53.019 satır · **50.660 çözülmüş** · 2026-06-24 → **2026-07-31** | CSV taraması |
| `pwin` dolu + çözülmüş | **6.002** | CSV taraması |
| `neural_pwin` dolu + çözülmüş | **6** | CSV taraması |
| AI öz-karnesi | bilgi 1.0 · kalibrasyon **0.5028, trend BOZULUYOR (drift)** · üretim **338 karar, %100 ABSTAIN** | `python -m nihai.ai.selfcheck --report` |

### 0.0 Öğrenme durumunun içi (`data/ai_learning_state.json`, `updated_at=2026-07-31T21:06Z`)

WP-A'nın hedeflediği kusur burada çıplak duruyor:

| anahtar | n | post_mean | lo95 | hi95 |
|---|---|---|---|---|
| GLOBAL | 3.546 | 0,5028 | 0,4864 | 0,5193 |
| LONG | 1.598 | 0,4456 | 0,4213 | 0,4700 |
| **SHORT** | 1.948 | **0,5497** | **0,5276** | 0,5718 |

`drift: {"drift": true, "index": 16, "window": 50, "n_resolved": 3546}` · `sources.neural.n = 6`

Üç şey aynı anda doğru: (a) SHORT'un `lo95`'i 0,5'in ÜSTÜNDE → kapı-a SHORT'a **AÇIK**
(motor tek yönlü makine — §E.4'ün ölçtüğü tam bu), (b) LONG'un `lo95`'i altında → kapı-a
LONG'a kapalı, (c) **drift ZATEN `true`** ve hiçbir şey onu tüketmiyor. Bu üçlü, WP-A'nın
neden kozmetik değil ölçülmüş bir kusur onarımı olduğunun kanıtıdır. Bu dosyanın kopyası
karşılaştırmanın ADİL olması için alındı (aynı prior state ile önce/sonra ölçümü).

### 0.1 Ölçümden çıkan YENİ bulgu — nöral köprünün canlı kanıtı 6 satır

`neural_shadow --selfcheck` bugün **598/600 üretiyor** ve "KÖPRÜ CANLI ve kalibrasyon taban
oranıyla tutarlı" diyor. Ama bu **sentetik** bir öz-sınavdır. Canlı log'da tablo şudur:

| gün | log satırı | `pwin` dolu | `neural_pwin` dolu |
|---|---|---|---|
| 2026-07-24 | 4.444 | 697 | **0** |
| 2026-07-30 | 732 | 113 | **0** |
| 2026-07-31 | 30 | 6 | **6** |

Yani Tur-20'nin "köprü CANLI" verdikti **doğru ama boş**: köprü çağrıldığında çalışıyor,
canlı yolda ise yalnız **6 kez** çağrıldı. Neden Tur-18'de zaten yazılmıştı — *"kolonlar
yalnız kayıtçı süreç YENİDEN BAŞLADIKTAN sonra dolmaya başlar (şema süreç belleğinde)"* —
ve kayıtçı 31 Temmuz'da geç saatte yeniden başlayıp ertesi gün kapandı.

**Sonuç: `sources.neural` Brier'i n=6 üstündedir, yani ölçüm DEĞİLDİR.** Bu bir kod hatası
değil, bir **veri-birikimi boşluğudur** ve yalnız servislerin açılmasıyla kapanır
(§4 operatör maddesi 1). Bu tur bunu kod ile "çözmeye" ÇALIŞMAZ — çözülemez.

### 0.2 WP-A'nın "ÖNCE" ölçümü (bugün koşuldu, decay KAPALI)

`python -m research.shadow_decision_backtest --days 180 --horizon 5 --lambda-sweep
--lambdas 0.5,0.05` · parametre: 180 gün · ufuk 5 · exec-lag 0 · maliyet 5 bps ·
n_bağlam 5.400 → seçim 3.240 / doğrulama 2.160.

| λ | bölme | yönlü karar | abstain | net getiri | **FAZLALIK vs hep-LONG** | t |
|---|---|---|---|---|---|---|
| **0,5 (üretim)** | her ikisi | **0** | %100 | — | — | — |
| 0,05 | SEÇİM | 820 | %74,7 | **−1,37%** | **−2,29 puan** | −4,14 |
| 0,05 | DOĞRULAMA | 532 | %75,4 | **−2,57%** | **−4,22 puan** | −4,75 |

Harness'ın kendi verdikti: *"Hiçbir λ doğrulama bölmesinde pozitif fazlalık vermedi →
λ=0.5'te KAL."* Not: açık ölçüldüğünde motor Temmuz'daki −1,75 puandan **DAHA KÖTÜ**
(−2,29 / −4,22) ve t istatistikleri artık anlamlı derecede NEGATİF — yani bayat prior'ın
zararı zamanla derinleşmiş. Bu, WP-A'nın "önce" tarafıdır; aynı `ai_learning_state.json`
kopyası saklandı ki "sonra" ölçümü adil olsun.

**Yorum kilidi (sonucu görmeden yazıldı):** decay AÇIKken yönlü karar sayısı sıfıra
giderse fazlalık açığı "kapanmış" görünür — ama bu kâr değil, yalnız MARUZİYETİN
kalkmasıdır. Dürüst okuma şudur: kenar yokken doğru cevap zaten susmaktır; WP-A'nın
katkısı bu cevaba **λ'nın elle sıkılığından değil, verinin kendi güncellik yapısından**
varılmasıdır. "Kâr arttı" diye raporlanamaz.

---

## 1. Teşhis: AI katmanının gerçek darboğazı nedir

Denetim (2026-07-26) düzeyleri: **kod sağlığı 10/10, ölçülen tahmin gücü 1.5/10.** Bugünkü
öz-karne bunu doğruluyor: motor 338 kararın **%100'ünde ABSTAIN** ediyor ve λ süpürmesi
kapı gevşetilse fazlalığın **0.0008 puan** olacağını söylüyor — yani susmak ÖLÇÜLMÜŞ doğru
davranış.

Bu durumda "AI'ı geliştirmek" iki yanlış biçimde anlaşılabilir ve ikisi de bu projede
açıkça yasaklıdır:

- ❌ **Daha büyük model / yeni sınıflandırıcı.** ACI DERS #5: *"Gerçek darboğaz veri ve
  OOS-kenarlı baz sinyal yokluğu, model mimarisi değil. Yeni bir sınıflandırıcı eklemek
  bunu çözmez."* Tur-22 zaten 30,4M→158,8M parametreye çıktı: DSR 0,33, kenar yok.
- ❌ **Kapıyı gevşetip motoru konuşturmak.** Ölçüldü (§E.4): λ=0,05'te motor 192 SHORT
  üretti, net **−0,50%**, hep-LONG tabanına göre **−1,75 puan**. Konuşan motor para
  kaybediyor.

Geriye üç meşru eksen kalıyor ve bu turun üç iş paketi tam olarak bunlardır:

1. **Öğrenme→karar döngüsündeki ÖLÇÜLMÜŞ bir kusuru onarmak** (bayat prior).
2. **Projenin kendi ilan ettiği tek dürüst ilerleme eksenini işletmek** (kalibrasyon).
3. **AI'ın asıl işini yapmak: bir sonraki aday aileyi dürüstçe ölçüp öldürmek** (edge_lab #4).

---

## 2. İş paketleri

### WP-A — Prior'ın hafızası: zaman-ağırlıklı + drift-koşullu posterior

**Ölçülmüş problem.** `learning.update_from_predictions` sayımları **zaman-ağırlıksızdır**:
5.000 satırlık kuyruktaki en eski kayıt ile dünkü kayıt aynı ağırlıktadır.
`page_hinkley` drift'i tespit edip `drift`/`retrain_candidate` yazar ama **bu bilgiyi
hiçbir şey tüketmez**. Bu soyut değil: §E.4'te motor, geçmiş DOWN rejiminden kalan SHORT
üstünlüğünü (0,5579) prior alıp yükselen piyasada zarar etti — raporun kendi teşhisiyle
*"ACI DERS #1'in doğrudan ampirik doğrulaması"*.

**Tasarım.** Üstel unutma (yarılanma ömrü 30 gün, env ile taşınır) + drift tespit edilirse
etkin sayımları `DRIFT_SHRINK=2.0`'a bölme. Yeni state anahtarı `strategies_decayed`;
mevcut `strategies` **dokunulmadan** kalır. `decision._prior_from_state` bunu **yalnız
`AI_PRIOR_DECAY` açıkken** okur → varsayılan davranış **bit-denk**.

**Kilitler.** (a) Referans zaman = kuyruktaki en son damga, "bugün" DEĞİL → aynı log her
zaman aynı posterior'u verir (determinizm). (b) Tüm damgalar **naive-UTC**'ye kanonikleşir
— BULGU-16 (naive-LOCAL/naive-UTC karışımı look-ahead'i) tekrarlanmaz. (c) Yan-koşullu
prior'ın `n≥200` kapısı **ham** satır sayısına bakar, etkin float sayıma değil (yoksa eşik
sessizce sıkılaşır). (d) Drift tepkisi **her zaman daha AZ güven** yönündedir.

**Neden güvenli.** Etki yönü tektir: daha az kanıt → daha geniş CI → `lo95 > 0.5` kapısı
DAHA ZOR geçilir. Yani bu değişiklik motoru daha çok susturabilir, daha çok konuşturamaz.

**Kabul kapısı.** Varsayılan bit-denklik testi + yarılanma matematiği kapalı formla +
drift'in CI'yı GENİŞLETTİĞİ testi + determinizm testi. Ardından **ölçüm**:
`research/shadow_decision_backtest.py` ile decay AÇIK/KAPALI, λ ∈ {0,5 · 0,05}, exec-lag
duyarlılığı ve triviyal taban (hep-LONG / momentum) karşılaştırması.

**Falsifiable öngörü (koşudan ÖNCE yazıldı):** bayat-prior hipotezi doğruysa decay AÇIK
iken λ=0,05'teki **−1,75 puanlık fazlalık açığı KÜÇÜLMELİDİR**. Küçülmezse hipotez yanlıştır
ve öyle raporlanır. Bu ölçüm İN-SAMPLE'dır (prior pencerenin kendi sonuçlarından türer) →
OOS iddiası ASLA yapılmaz.

### WP-B — Kalibrasyon döngüsünün kapatılması (yalnız-OOS)

**Ölçülmüş problem.** `learning.fit_calibrator` / `apply_calibration` (Platt + izotonik)
yazılmış ve test edilmiş ama **üretimde hiçbir çağıranı yok** — repo taraması: yalnız
`tests/test_ai_learning.py`. AI katmanı kalibrasyonunu **ölçüyor** (`sources.*.ece`) ama
hiçbir zaman **düzeltmiyor**. Oysa Tur-22 kaydı aynen şöyle diyor: *"AUC ~0,50 iken tek
dürüst ilerleme ekseni budur."* Denetimin B-4 bulgusu da açık: meta `pwin` bilgisiz
tabandan kötü ve **ters** kalibre.

**Tasarım.** Yeni `nihai/ai/calibration_loop.py`: kaynak başına (`pwin`/`neural`/`prior`)
**kronolojik** bölme → kalibratör YALNIZ fit diliminde öğrenilir → OOS diliminde ham vs
kalibre Brier/ECE karşılaştırılır. Artefakt `data/ai_calibrators.json` **yalnızca Brier VE
ECE'nin İKİSİ birden OOS'ta iyileşirse** yazılır (gürültü freni — kanıt kapısı değil, öyle
sunulmayacak). Anayasa gereği çıktı yalnız rapor/gölgedir; canlı karar yolundan çağrılmaz.

**Bugünkü veri gerçeği:** `pwin` için 6.002 çift → ölçülebilir. `neural` için 6 çift →
modül dürüstçe `YETERSIZ` demek ZORUNDA (bu yol bugün gerçekten tetiklenecek, teorik değil).

**Kabul kapısı.** Sızıntı testi (OOS dilimini bozmak fit sonucunu değiştirmez), kronolojik
bölme testi, yetersiz-örneklem testi, kalıcılık kapısı testi, bozuk artefakt dayanıklılığı.
Ardından **gerçek log üstünde koşu** — sonuç negatifse negatif raporlanır.

### WP-C — edge_lab BACKLOG #4: endeks rebalans / akış ailesi (PIT)

**Neden şimdi.** BACKLOG'daki tek `HAZIR` aday. #1 ve #3'ün aksine survivorship-ŞİŞKİN
olmak ZORUNDA DEĞİL: `sp500_as_of` geçmiş üyeliği verir ve çıkarılmış isimler önbellekte.

**Ön-koşul KAPANDI (bu turda, 148 sn).** Tur-24 engeli — *"516 `*_1d.csv` `raw` kipiyle
yazılmış, bölünme günlerinde sahte %50 getiri"* — çözüldü: 2021-01 → 2026-07 arası
çeyreklik `sp500_as_of` birleşimi = **604 sembol**, hepsi `adjustment=all` ile çekildi
(`*_1d_all.csv`, sembol başına ~1.555 bar, **0 hata**). Mevcut `raw` dosyalar EZİLMEDİ
(düzeltilmiş veri ayrı ada gider) → geri dönüş mümkün.

**Tasarım kilitleri.** (K1) Wikipedia *effective date* verir, ANONS tarihini değil →
**ileri-bakan olay özelliği YASAK**, yalnız geriye-bakan olay mesafesi; takvim özellikleri
(ay/çeyrek sonu) muaf çünkü takvim kamuya açıktır. (K2) Üyelik her barda PIT çözülür.
(K4) **Gerçek `pnl_col` zorunlu** — bu, Tur-24'te düzeltilen `_shift_label_1bar` yolunu
gerçek anlamda ısıracak **ilk aday**dır; sözde-pnl ile kaçılamaz. (K5) Tek ufuk (5 bar),
ufuk süpürmesi yok (Tur-19 zaten süpürdü; süpürme yalnız çoklu-test yükü ekler).
(K6) 6-8 özellik tavanı. (K7) `n_kept` ve gerçek olay sayısı raporlanır.

**Dürüst ön-beklenti (koşudan ÖNCE yazıldı):** DÜŞÜK. Literatür 2000 sonrası endeks-ekleme
etkisinin büyük ölçüde söndüğünü söylüyor (Petajisto 2011 ve sonrası). Beklenti FAIL'dir;
PASS çıkarsa ekstra şüphe gerekir — denenen-aday sayacı **3**'e çıkıyor ve sayaç büyüdükçe
ilk PASS'in kanıt değeri DÜŞER.

---

## 3. Bu turda BİLEREK YAPILMAYANLAR

| Yapılmadı | Neden |
|---|---|
| Yeni/daha büyük nöral mimari | ACI DERS #5; Tur-22'de 158,8M parametre denendi, DSR 0,33 |
| `AI_UTILITY_LAMBDA`'yı düşürmek | §E.4'te ölçüldü: konuşan motor para kaybediyor |
| Kanıt kapısı eşiklerini oynatmak | Tur-23/24 kuralı: eşik yalnız ADAYI OLMAYAN turda değişir; bu turda aday VAR (WP-C) |
| Nöral modeli yeniden eğitmek | Canlı veri 1 Ağustos'tan beri BÜYÜMÜYOR (servisler kapalı) → yeni eğitim aynı 5.996 satır |
| XL kapasite süpürmesi | Tur-24'te ÖLÇÜLDÜ: bu donanımda PIT 97k için 21 gün — imkânsız, GPU gerekir |
| FAZ AI-1 (LLM akıl yürütme) | `ANTHROPIC_API_KEY` boş — operatör kilidi |

---

## 3.5 SONUÇLAR (ölçüldü — bu bölüm koşulardan SONRA yazıldı)

### WP-A sonucu — ÖN-KAYITLI ÖNGÖRÜ ÇÜRÜTÜLDÜ

§0.2'de koşudan ÖNCE şunu yazmıştım: *"bayat-prior hipotezi doğruysa decay AÇIK iken
λ=0,05'teki fazlalık açığı KÜÇÜLMELİDİR. Küçülmezse hipotez yanlıştır ve öyle raporlanır."*

Öğrenme durumu yeni kodla yeniden üretildi (aynı state ile A/B, adil karşılaştırma):

| anahtar | ağırlıksız `lo95` | decay+shrink `lo95` | `post_mean` |
|---|---|---|---|
| SHORT | 0,5290 | **0,5165** (hâlâ >0,5) | 0,5508 → 0,5498 |
| LONG | 0,4210 | 0,4123 | 0,4451 → 0,4484 |

Backtest (180 gün · ufuk 5 · maliyet 5 bps · aynı state):

| λ=0,05 | decay KAPALI | decay AÇIK |
|---|---|---|
| SEÇİM: yönlü / FAZLALIK vs hep-LONG | 823 / **−2,4236** | 806 / **−2,3973** |
| DOĞRULAMA: yönlü / FAZLALIK vs hep-LONG | 553 / **−4,2328** | 531 / **−4,3881** |
| λ=0,5 (üretim) her iki kip | 0 yönlü / %100 ABSTAIN | 0 yönlü / %100 ABSTAIN |

Seçim bölmesinde ihmal edilebilir iyileşme, **doğrulama bölmesinde kötüleşme**; fark her iki
yönde de gürültü mertebesinde. **Hipotez DOĞRULANMADI.**

**Neden — ölçüldü, uydurulmadı:** canlı log yalnız **2026-06-24 → 07-31** arasını kapsıyor
(~5 hafta). 30 günlük yarılanmada en eski satır bile 0,43 ağırlık alır, bu yüzden decay
prior'ı kımıldatamıyor (SHORT `post_mean` 0,5508 → 0,5498). Yani **SHORT yanlılığı bu
pencerede bayat değil, baştan sona mevcut** — zararın nedeni prior'ın ESKİMESİ değil,
prior'ın hiç tahmin gücü OLMAMASIDIR.

**Yine de ne kazanıldı:** (a) tespit edilip **tüketilmeyen** bir drift sinyali artık
tüketiliyor (mekanizma doğru ve testlerle kilitli), (b) mekanizmanın etkisi ancak log
BİRDEN FAZLA REJİMİ kapsadığında anlamlı olur — bu da doğrudan operatör maddesi 1'e bağlı,
(c) `AI_PRIOR_DECAY` **varsayılan KAPALI kalır**: ölçüm onu açmayı desteklemiyor. Bir kez
daha: ölçüm kendi düzeltmemizi reddetti ve varsayılan değişmedi.


### WP-B sonucu — kalibrasyon kazancının kaynağı ÖLÇÜLDÜ

`calibration_loop --refresh --tail 60000`: `pwin` kaynağında OOS (2.401 nokta, kronolojik
bölme, fit 3.601) **Brier 0,2723 → 0,2490 · ECE 0,1325 → 0,0045**. Kapı geçildi, artefakt
`data/ai_calibrators.json`'a yazıldı. `neural`: **YETERSIZ** (n_oos=3 < 200) — §0.1'in
öngördüğü dürüst yol tetiklendi.

Kalibratörün katsayısı NEGATİF (`coef=−0,9134`). Bu "ham pwin ters bilgi taşıyor" gibi
okunabilir ve ilk raporlamada öyle okundu. **Kontrol koşuldu:**

| tahminci | Brier | ECE |
|---|---|---|
| HAM `pwin` | 0,2723 | 0,1325 |
| KALİBRE (platt) | 0,2490 | 0,0045 |
| **SABİT = fit taban oranı (0,4646)** | **0,2494** | **0,0090** |
| SABİT 0,5 | 0,2500 | 0,0264 |

Kalibre sürüm sabit tabanı yalnız **0,0004 Brier** geçiyor; kalibre olasılıkların std'si
**0,0156** (ham 0,0686), ortalaması 0,4704 ≈ taban oran. Yani kazancın tamamına yakını
**taban orana BÜZÜLMEDEN** geliyor, ters çevirmeden değil. Dürüst ifade: *meta `pwin`
çıktısı ağır fazla-güvenlidir ve kullanılabilir bilgi taşımaz; dürüst kalibre biçimi taban
oranın kendisidir.* Bu, ACI DERS #2/#4'ün (META_GATE varsayılan KAPALI) bağımsız ve taze
bir doğrulamasıdır — ve bir kenar iddiası DEĞİLDİR (ACI DERS #3).

### WP-C sonucu — aday #4 FAIL, ama iki yan bulgu

Dataset **728.978 satır / 605 sembol / 2021-02-02 → 2026-07-23**, PIT (evrenin 90 üyesi
bugün endekste YOK), 101 ekleme / 103 çıkarma olayı. Verdikt: **KAPI GEÇİLEMEDİ**
(purged_cv OOS-AUC **0,5066** < 0,53). Denenen-aday sayacı **2 → 3**.

**BULGU-25 — `dsr` kapısı gerçek `pnl_col` altında ayırt edici DEĞİL.** Kapı DSR=1,0 verdi
(AUC 0,507 iken). Ajan bunu ölçtü: DSR=1,0'ı kapı-sonrası (SR 0,091), kapısız base (0,048),
kapının **ELEDİĞİ** satırlar (0,042) ve **rastgele** alt küme (0,050) de üretiyor. Neden:
`_kept_pnl_from_time_gate` HAM PnL döndürüyor ve `_stage_dsr` Sharpe'ı **sıfıra karşı**
ölçüyor; `base_E=+0,2514` (yükselen piyasa) olduğunda her alt küme pozitiftir. Bu, ACI
DERS #8'in (sınıf dengesizliği DSR'yi şişirir) **aynı yapısal hatasının farklı yerdeki
tekrarıdır** ve gerçek `pnl_col` veren ilk adayla görünür oldu.
**Kapı bu turda BİLEREK DÜZELTİLMEDİ**: Tur-23 kuralı — aday FAIL ettikten sonra kanıt
kapısını elden geçirmek sonuca göre kural değiştirmektir; adayı olmayan ayrı bir turda
yapılmalı. Ayrıca doğru taban seçimi önemsiz değil: sözde-pnl'de taban oranı %50'nin
altındayken taban çıkarmak kapıyı **gevşetir** (ters yön). Tasarım sorusu açık bırakıldı.

**Şartname hatası (benim):** WP-C brief'imde "giriş öncesi barlarda `endeks_uyesi=0`" testi
istemiştim. Ajan bunu koşudan ÖNCE reddetti: bu, sembolün endekse girmeden önceki barlarını
panele koymayı gerektirir = **evren-seçimi look-ahead'i** ("bu hisse ileride S&P 500'e
alınacak") ve ekleme-öncesi ralli literatürün en bilinen etkisi olduğu için tam da uydurulmuş
kenar üretecek yerdir. Yerine daha sıkı bir kilit kondu (giriş öncesi satır ÜRETİLMEZ,
çıkarılan isimler panelde KALIR). Sapma haklıydı ve SONUC raporunda belgelendi.

---

## 4. Operatöre açık maddeler (kod ile kapanmaz)

1. **Servisler 1 Ağustos 00:09'dan beri KAPALI** (`logs/project.stopped`). 10 gündür canlı
   tahmin/özellik birikmiyor. §0.1 bunun somut bedelini gösteriyor: nöral köprünün canlı
   kanıtı **6 satırda** donmuş durumda. ACI DERS #5'e göre kenar sorusunun tek dürüst yolu
   canlı paper kanıtıdır → birikim durduğu sürece proje ilerlemiyor. Açmak: `Nihai - Projeyi Ac`.
2. Tur-18/20'den devreden: `tur18-servis-tetik.ps1` + `tur20-shadow-gorev.ps1` YÖNETİCİ
   PowerShell'de bir kez; `NihaiAlgo-ShadowRunner-Hourly` hâlâ `Disabled`.
3. `ANTHROPIC_API_KEY` boş (FAZ AI-1 kilidi).
4. GPU yok → XL/topluluk kapasite çalışmaları askıda (`--device auto` hazır).
