# AI Katmanı — Bağımsız Test / Benchmark / Doğruluk Denetimi

**Tarih:** 2026-07-26 (Pazar) · **Kapsam:** `nihai/ai/` (23 modül) + `nihai/execution/neural_shadow.py` + canlı kanıt döngüsü
**Yöntem:** salt-okunur ölçüm — testler, mypy/ruff, dahili öz-sınav (`selfcheck --report`), canlı log analizi, uç-nokta duman testi.
**Not:** Bu denetim yalnız ölçer; hiçbir kenar iddiası içermez (ACI DERS #3/#4).

---

## 0. Yönetici özeti

| Boyut | Not | Gerekçe |
|---|---|---|
| Yazılım mühendisliği | **9.0 / 10** | 11.350 satır, 960 AI testi %100 yeşil, mypy 0 hata / 86 dosya, ruff temiz, fail-soft + provenance disiplini |
| Quant/ML metodoloji | **8.5 / 10** | purged+embargo CV, PBO/CSCV, DSR/PSR (düzeltilmiş), HRP, MC-dropout belirsizliği, exec-lag kapısı — literatür seviyesinde |
| Dürüstlük / anti-overfit | **9.5 / 10** | Sektörde nadir. Kendi modelini "KENAR YOK" diye kaydediyor, kendi kanıt kapısının bozuk olduğunu bulup düzeltiyor |
| **Ölçülen tahmin gücü** | **1.5 / 10** | OOS-AUC 0.500 · canlı isabet %46.8 · pwin Brier 0.285 (0.25 bilgisiz tabanın ALTINDA değil, ÜSTÜNDE = daha kötü) |
| **Operasyonel karar üretimi** | **1.0 / 10** | 248/248 karar ABSTAIN, conviction her zaman 0.000 — **yapısal olarak** yön üretemez (bkz. B-1) |
| Öğrenme döngüsünün kapalılığı | **4.0 / 10** | 4 geri-besleme kolundan yalnız 1'i çalışıyor |

**Tek cümle:** Mühendisliği ve bilimsel dürüstlüğü üst düzey, ölçüm altyapısı sağlam; ancak **hâlâ hiçbir tahmin gücü ölçülmemiş** ve karar motoru bir yapılandırma hatası nedeniyle **hiçbir koşulda yön üretemiyor**.

---

## 1. Ölçüm sonuçları

### 1.1 Kod sağlığı
| Ölçüm | Sonuç |
|---|---|
| `nihai/ai/` üretim kodu | 23 modül · **11.350 satır** |
| AI test kodu | 25 dosya · 6.769 satır · **960 test — hepsi GEÇTİ** |
| Tam paket (`pytest`) | **exit 0 — tamamı yeşil** |
| `ruff check nihai/ai research tests` | **All checks passed** |
| `mypy nihai` | **Success: no issues found in 86 source files** |
| Web AI uç noktaları | **13/13 → HTTP 200** (127.0.0.1:8765) |

İçerik envanteri: `knowledge` **103 kavram**, `quant_math` **54 fonksiyon**, `advanced_math` **22**, `portfolio` **9**, `episodic.LESSONS` **13 ders**.
Kapasite merdiveni: logistic 26 → elasticnet 326 → gbm_small 2.660 → mlp 3.777 → deep_mlp 186.702 → quant_transformer **5.392.659** parametre.

### 1.2 Dahili öz-sınav — `python -m nihai.ai.selfcheck --report` (bugün koşuldu)
```
[Bilgi sınavı]       n=50  skor=1.00   (DAG-tutarlı=True)
[Kalibrasyon sınavı] isabet-posterior=0.5028   trend=BOZULUYOR (drift tespit edildi)
[Replay sınavı]      rejim-doğruluk=1.00   abstain-oranı=1.00 (n=3)
```
> Bilgi sınavı 1.00 gerçek "bilgi" değil, **iç tutarlılık**tır (formül köprüsü + prereq DAG). Replay 1.00 da sentetik GBM serisinde trend etiketi doğruluğudur — kolay bir sınav.

### 1.3 Canlı doğruluk (`data/predictions_log.csv`)
| Ölçüm | Değer |
|---|---|
| Toplam tahmin / çözülmüş | 52.257 / **50.015** |
| **Canlı isabet oranı** | **%46.83** (verdikt: `INVERTED`) |
| Haftalık trend | %47.1 → %47.8 → %46.5 → **%46.2** (düşüyor) |
| Yön bazlı (ham log) | LONG `+` %45.7 · SHORT `−` %47.9 |
| Bekleyen | 15 |

### 1.4 Öğrenme durumu (`nihai/ai/learning.py`, pwin'li alt küme, tail=4000)
| Anahtar | n | isabet | %95 CI |
|---|---|---|---|
| GLOBAL | 2.987 | **0.5028** | [0.4849, 0.5208] |
| LONG | 1.114 | **0.4104** | [0.3817, 0.4394] ← 0.5'in **anlamlı altında** |
| SHORT | 1.873 | **0.5579** | [0.5353, 0.5803] ← 0.5'in **anlamlı üstünde** |

- `pwin` kalibrasyonu: n=471, **Brier 0.2853** — bilgisiz taban 0.25'ten **daha kötü**.
  Kovalar ters: model 0.268 derken gerçek kazanma %56.1; 0.4375 derken %44.3.
- `drift = true`, `retrain_candidate = true`
- `sources.neural`: **n = 0** (hiç kayıt yok)

### 1.5 Gölge karar motoru
| Ölçüm | Değer |
|---|---|
| Toplam gölge karar | 248 (**yalnız 2026-07-25, Cumartesi**) |
| Aksiyon dağılımı | **ABSTAIN 248 / 248 (%100)** |
| Conviction | min 0.000 · max **0.000** |
| Ders uyarısı tetiklenen | 0 / 248 |
| Aynı gün+sembol+posterior tekrarı | **188 fazlalık (%75.8)** |
| Epizodik hafıza | 248 epizot — **`outcome` hepsinde `None`** |
| shadow_runner turları | 8 tur, 218 abstain, 0 long/short bias, 0 hata |
| HRP portföy sonucu | 8 turda **yalnız 2 farklı** çıktı |

---

## 2. Bulgular

### 🔴 B-1 — Karar motoru YAPISAL olarak yön üretemez (KRİTİK)
`decision.py:_prior_from_state()` prior'ı `(strateji_adı, "GLOBAL")` sırasıyla arar. Canlı strateji `event_breakout`, öğrenme durumu ise `GLOBAL/LONG/SHORT` anahtarlı → **her zaman GLOBAL'e düşer**.
Kapı-a şartı `ci_supports_side = lo95 > 0.5`; GLOBAL'in lo95 = **0.4849** → koşul **hiçbir zaman** sağlanamaz → `%100 ABSTAIN` matematiksel olarak garantidir.

Aynı state dosyasında **SHORT'un lo95 = 0.5353 > 0.5** — yani öğrenme katmanı kapıyı geçecek kanıtı üretiyor, karar katmanı onu **hiç okumuyor**. Yan-koşullu prior'ın dışarıda bırakılması modül docstring'inde gerekçelendirilmemiş.

**Sonuç:** conviction/Kelly/fayda/CVaR/conformal — tüm alt sistem sadece ölü kod yolu üzerinde çalışıyor; hiçbiri fiilen test edilmiş değil.
**Uyarı:** Bu "SHORT'u aç" demek DEĞİLDİR. SHORT'un %55.8'i büyük olasılıkla rejim artefaktıdır (kararların rejim etiketi `down_mid`) — ACI DERS #1/#7 aynen geçerli. Bulgu, kenar değil **mimari kopukluk**tur.

### 🔴 B-2 — Nöral gölge köprüsü bozuk kalibrasyonla canlıya bağlı (KRİTİK)
`neural_shadow.selfcheck()` gerçek canlı özellik vektörlerinde: **600/600 üretildi, 0 susturuldu, ortalama p_win = 0.0585**.
400 vektörde doğrulandı: ort **0.0546**, medyan 0.0458, aralık 0.033–0.162, MC-std 0.034.
Eğitim taban oranı **0.357** → kalibre bir model ~0.357 üretmeliydi. Model pratikte sabit "%5 kazanır" diyor.

**Kök neden — zaman-dilimi uyumsuzluğu:** model **1g (günlük)** barlarda eğitildi (`meta_dataset_pit_sp500_..._1d_...`), canlı tahminlerin **%100'ü `interval=1m`**.

| Özellik | Canlı ort | Eğitim μ | Eğitim σ | \|z\|max |
|---|---|---|---|---|
| **hour** | 16.84 | 4.38 | 0.485 | **30.2** ← her satırda ~+26σ sabit itki |
| **rvol** | 1.36 | 1.32 | 0.897 | **13.1** |
| atr_pct | 0.00092 | 0.0275 | — | 30× küçük |
| bb_width | 0.0057 | 0.1217 | — | 21× küçük |
| range_pct | 0.00094 | 0.0336 | — | 36× küçük |

Doygunluk guard'ı (BULGU-17) bunu **yakalamıyor**: guard `std == 0` şartı arıyor, buradaki std ≈ 0.034.
Beklenen sonuç: `sources.neural` Brier ≈ **0.42** (bilgisiz taban 0.25). Pazartesi kolonlar dolmaya başlayınca sistemin en kötü kaynağı görünecek — ama nedeni "model kenarsız" değil, **boru hattı uyumsuz**. Yanlış teşhis riski yüksek.
İyi haber: köprü log-only, karara sızmıyor (anayasa korunuyor).

### 🟠 B-3 — Epizodik hafızanın geri-besleme kolu ölü
248 epizodun **hepsinde `outcome = None`**. Sonuç yazılmadığı için "benzer durum hatırla" çağrısı sonuçsuz epizotlar döndürüyor; `consolidate` → anlamsal hafıza → ders üretimi zinciri bilgi taşımıyor. Epizodik hafıza şu an bir **arşiv**, öğrenme organı değil.

### 🟠 B-4 — Meta `pwin` bilgisiz tabandan kötü ve ters kalibre
Brier **0.2853** > 0.25. 0.2–0.3 kovasında model 0.268 derken gerçekleşen %56.1. Bu, "kenar yok"un ötesinde **ters işaret** sinyalidir. `META_GATE=shadow` sayesinde zarar vermiyor ama `pwin_calibration` raporu bu tersliği ayrıca isimlendirmiyor.

### 🟠 B-5 — Kayıt defterinde bozuk-DSR'li kart hâlâ duruyor
`ac505353…` kartı: aynı PIT verisi, **DSR 1.0 / PSR 1.0 / sharpe_pseudo 0.2908** — ACI DERS #8'in düzelttiği sınıf-dengesizliği hatasının ürünü. Düzeltilmiş kart (`1f072998…`, DSR **0.0**) yanında duruyor ve `oos.auc_mean` alanı yok. `list_models()` üçünü de eşit statüde döndürüyor → gelecekte yanlış kartın seçilmesi/alıntılanması riski.

### 🟡 B-6 — Gölge tur zamanlaması piyasayla uyumsuz + %76 tekrar
`NihaiAlgo-ShadowRunner-Hourly`: günlük 09:30 (TR) + 1s aralık × 7s = **06:30–13:30 UTC**. ABD açılışı 13:30 UTC → **AI, piyasa açılmadan düşünüyor**. Günlük barlar gün içinde değişmediği için 8 turun 6'sı birebir aynı sonucu üretiyor (HRP: 8 turda 2 farklı çıktı; kararların %75.8'i tekrar). Cumartesi de koştu.

### 🟡 B-7 — Ölü ve bozuk özellikler
- `rr_ratio` sabit 2.0 (eğitim σ = **3.2e-15**), `confluence` sabit 1.0 → 25 özelliğin 2'si sıfır bilgi. σ≈0 normalizasyonda mayın: değer bir gün değişirse z-skor 10¹⁵ mertebesine fırlar.
- Canlı `vix` yalnız **9 eşsiz değer** ve içinde **0.0** var (eksik → 0 dolgusu; 0 geçerli bir VIX değil).
- 4 eski meta-dataset'te 6 kolon (`gap_pct/range_pct/close_pos/streak/rsi_slope/hour`) hiç yok → nötr-dolgu ile okunuyor.

### 🟡 B-8 — Kullanılmayan modüller
`agent.py` (413 satır): gündem **0/0/0**, üretimde yalnız pano uç noktasından erişilebiliyor. `lab.py` (460 satır): üretimde çağıran yok — `brain.py:222`'de sadece **öneri metni** olarak adı geçiyor.

### 🟡 B-9 — Commitsiz iş
8 değişik + 7 izlenmeyen dosya commit edilmemiş — aralarında **`nihai/execution/neural_shadow.py`** ve `tests/test_neural_bridge_selfcheck.py` var. Tur-18 sonrası düzeltmeler versiyonlanmamış.

### 🟡 B-10 — 4 zamanlanmış görevin hepsi `0xC000013A` ile bitiyor
`STATUS_CONTROL_C_EXIT` — görev sonlandırılarak bitmiş. Servisler ayakta olduğu için sonuç şu an zararsız, ama görev sağlığı bu koddan okunamıyor (gerçek hatayı maskeler).

---

## 3. Artılar

1. **Bilimsel dürüstlük** — sektörde nadir. Model kendi kartına "KENAR YOK" yazıyor; kanıt kapısının kendisi bozukken (DSR 1.0 ↔ AUC 0.50 çelişkisi) bunu tespit edip düzeltmiş.
2. **Anayasa gerçekten uygulanmış.** `nihai/ai` hiçbir yoldan sinyale/eşiğe/emre bağlı değil; `enforce` kipi kodda **yok**. Nöral köprü bile web JSON'undan düşürülüyor.
3. **Metodolojik olgunluk** — purged+embargo K-fold, PBO/CSCV, DSR/PSR, permütasyon önemi, exec-lag duyarlılığı, survivorship/PIT ayrımı, HRP, MC-dropout belirsizliği.
4. **Kod kalitesi** — mypy 0 hata (86 dosya), ruff temiz, 960 AI testi yeşil, atomik/kilitli yazımlar, fail-soft her katmanda, lazy torch importu.
5. **Gözlemlenebilirlik** — her karar tam kanıt zinciriyle loglanıyor (prior → kaynak → rejim → kapı-a → kapı-b → sonuç). Bu denetimin bu kadar hızlı yapılabilmesinin sebebi bu.
6. **Öz-değerlendirme altyapısı** — `selfcheck` üç bölümlü sınav + trend satırı; `neural_shadow.selfcheck()` "köprü sessiz mi ölü mü" ayrımını piyasa beklemeden ölçüyor. Bu denetimdeki en değerli iki bulgu bu araçlardan çıktı.
7. **Provenance** — model kartlarında veri kaynağı, satır sayısı, purge, hiperparametre, ortam (python/torch/platform), hash.
8. **Dokümantasyon** — CLAUDE.md'deki 8 "ACI DERS" gerçek kurumsal bilgi; her biri ölçümle desteklenmiş.

---

## 4. Eksiler / eksikler

1. **Hiçbir tahmin gücü ölçülmemiş.** Tüm modellerde OOS-AUC ≈ 0.50; canlı isabet %46.8 ve düşüyor. 5.4M parametrelik QuantTransformer kapasitesi 0.50 AUC'lik veriyle karşı karşıya.
2. **Karar motoru hiç karar vermedi** (B-1) — dolayısıyla conviction/Kelly/fayda/CVaR yolu üretimde **hiç yürütülmedi**.
3. **Öğrenme döngüsünün 4 kolundan 3'ü kopuk:** learning ✅ · episodic outcome ❌ (B-3) · neural kaynağı ❌ (B-2) · decision→prior ❌ (B-1).
4. **Operasyonel geçmiş yok** — gölge kip yalnız **1 gün** (ve o gün Cumartesi) veri üretti. Karar motoru hakkında istatistiksel hiçbir şey söylenemez.
5. **Karne sınavları fazla kolay** — bilgi 1.00 (iç tutarlılık) ve replay 1.00 (sentetik GBM'de trend etiketi) tavana yapışık; bozulmayı erken yakalayamaz.
6. **Zaman-dilimi tutarsızlığı sistemik:** canlı 1m, meta modeller 1d/1h karışık. Tek bir "hangi ufuk" kararı yok.
7. **Ölçek/veri dengesizliği** — 234k ve 97k satırlık dataset'ler diskte dururken deep_mlp 12k satırlık sette eğitilmiş.
8. **`quant_transformer` (5.4M) hiç kaydedilmemiş** — merdivende tanımlı ama defterde kartı yok.

---

## 5. Geliştirme önerileri (öncelik sırasıyla)

**P0 — kanıt döngüsünü dürüst hale getir (bunlar yapılmadan başka hiçbir ölçüm anlamlı değil)**
1. **B-2'yi kapat.** Ya `NEURAL_SHADOW=off` yap, ya 1m barlarda eğitilmiş bir model kaydet, ya da köprüye zaman-dilimi kilidi ekle: kart `data.source`'undaki interval ile canlı `interval` eşleşmiyorsa yükleme reddedilsin + `status().reason` yazsın. (`feature_names` kilidinin yanına **`interval` kilidi**.)
2. **Doygunluk guard'ını genişlet.** `std == 0` yerine: eğitim taban oranına göre `|p̄ − pos_rate| > 0.2` **veya** `|z|max > 6` → sustur + nedeni logla. Bugünkü guard bu vakayı kaçırdı.
3. **B-1'i karara bağla.** İki seçenekten birini seç ve **docstring'e yaz**: (a) `side_hint` verildiğinde prior kaynağı `LONG`/`SHORT` olsun (kapı-a ilk kez çalışabilir); (b) yan-koşullu prior kasıtlı olarak dışarıda kalsın ve `_prior_from_state` bunu açıkça belgeleyip evidence satırına "yan-koşullu prior kullanılmadı (rejim artefaktı riski)" eklesin. **Sessiz ölü kod yolu üçüncü seçenek değil.**
4. **B-3'ü kapat.** `prediction_tracker.resolve()` bir tahmini çözdüğünde ilgili epizodun `outcome`'unu güncellesin (episodic'e `update_outcome(id, outcome)`). Bu olmadan epizodik hafıza öğrenmiyor.

**P1 — ölçüm hijyeni**
5. `rr_ratio` ve `confluence`'ı FEATURE_NAMES'ten çıkar veya σ<1e-6 olan özelliği eğitimde otomatik **drop** et (σ≈0 bölme mayını).
6. Canlı `vix == 0` durumunu eksik-değer say (nötr-dolgu), 0'ı geçerli VIX gibi modele verme.
7. B-5: `registry`'ye `deprecated` bayrağı ekle; bozuk-DSR'li kartı işaretle. `list_models()` varsayılan olarak yalnız güncel kartları dönsün.
8. `pwin_calibration` raporuna **ters-kalibrasyon uyarısı** ekle: kova ortalaması ile gerçek oran arasındaki korelasyon negatifse açıkça yaz.

**P2 — verim ve kapsam**
9. B-6: gölge turu ABD seansına kaydır (13:30–20:00 UTC) veya günde 1 kez koş; hafta sonu tetiklemesini kapat. Şu anki 8 turun 6'sı çöp.
10. Karne sınavlarını zorlaştır: bilgi sınavına formül **sayısal doğrulaması** (bilinen girdi → beklenen çıktı), replay'e rejim-geçişli (trend→çöküş) seri ekle.
11. Deep model'i 97k/234k satırlık dataset'lerde yeniden eğit; kapasite merdiveninin gerçekten veriyle mi yoksa gürültüyle mi sınırlı olduğunu ayır.
12. B-8: `lab.py` ve `agent.py` için ya gerçek bir üretim çağrısı ekle ya da `docs/`'ta "manuel/CLI-only" olarak etiketle.
13. B-9: commitsiz işi commit'le; B-10: görev sonlanma kodunu araştır (`ExecutionTimeLimit` PT20M ile ilişkili olabilir).

---

## 6. Sonuç

Bu AI katmanı **bir tahmin sistemi olarak değil, bir ölçüm ve dürüstlük sistemi olarak** üst düzey. En büyük değeri, "kenar bulduk" demeyi neredeyse imkânsız kılan mimarisi; bugünkü iki kritik bulgu (B-1, B-2) da zaten **kendi öz-denetim araçlarıyla** ortaya çıktı — bu, tasarımın çalıştığının kanıtı.

Buna karşılık: **karar motoru hiç karar vermedi, nöral köprü bozuk kalibrasyonla canlıya bağlandı, epizodik hafıza sonuç görmüyor ve canlı isabet %46.8'e geriliyor.** Sistem "düşünme" iskeletini kurmuş, ama iskeletin içinden henüz hiçbir sinyal geçmemiş.

Darboğaz değişmedi ve bu denetim onu bir kez daha doğruluyor: **mimari değil, OOS-kenarlı baz sinyal ve tutarlı zaman-dilimli veri.** P0 maddeleri o gerçeği daha net görmeye yarar; kenar üretmez.

---

# EK — P0 UYGULAMASI VE ÖLÇÜM (aynı gün, 2026-07-26)

## E.1 Yapılanlar

| # | İş | Dosya | Durum |
|---|---|---|---|
| P0-1 | Nöral köprüye **OOD guard** (`max|z| > NEURAL_SHADOW_MAX_Z`, vars. 6.0) + eğitim aralığı/taban oranı görünürlüğü + sayaçlar | `nihai/execution/neural_shadow.py` | ✅ |
| P0-2 | Doygunluk guard'ı genişletildi (`std==0` → **+ dağılım-dışı z**), `selfcheck()` verdikt üretiyor | `nihai/execution/neural_shadow.py` | ✅ |
| P0-3 | **Yan-koşullu prior** (strateji → LONG/SHORT (n≥200) → GLOBAL → yok) + artefakt uyarısı + `AI_PRIOR_SIDE=off` geri dönüşü | `nihai/ai/decision.py` | ✅ |
| P0-4 | **Epizot sonuç çözücüsü** (yeni modül) + `shadow_runner` her turda çağırıyor + 11 test | `nihai/ai/episode_resolver.py`, `tests/test_ai_episode_resolver.py` | ✅ |
| Ölçüm | **Gölge karar motoru backtest harness'ı** (taban kuralı karşılaştırması + exec-lag + maliyet) | `research/shadow_decision_backtest.py` | ✅ |

Kod sağlığı: **ruff temiz · mypy 0 hata / 87 dosya · tam paket exit 0** (yeni 11 test dahil).

## E.2 🔴 YENİ BULGU B-11 — Kapı-b de yapısal olarak geçilemez

B-1 düzeltildikten sonra prior kaynağı doğru okunuyor (backtest'te LONG 322 / SHORT 275 / GLOBAL 3)
ama motor **hâlâ 600/600 ABSTAIN** verdi. Neden: kapı-b

`E = post·kazanç − (1−post)·kayıp − λ·|CVaR95| > 0`, `λ` varsayılan **0.5**.

Tipik günlük seride kazanç≈kayıp≈0.04, |CVaR95|≈0.09 → E>0 için `(2·post−1)·0.04 > 0.045`,
yani **post > 1.06** gerekir. Matematiksel olarak imkânsız. Yani B-1 tek başına ABSTAIN kilidini
açmıyor; **iki bağımsız yapısal kilit** vardı.

`λ=0.05` ile kapı geçilebilir hale geliyor (ölçüm için kullanıldı). **Üretim varsayılanı
DEĞİŞTİRİLMEDİ** — aşağıdaki ölçüm bunun neden doğru olduğunu gösteriyor.

## E.3 Operasyonel karar üretimi — ölçüm (15 sembol · 120 gün · ufuk 5 bar · 600 karar)

| λ | ABSTAIN | LONG_BIAS | SHORT_BIAS |
|---|---|---|---|
| **0.5 (üretim)** | 600 (%100) | 0 | 0 |
| 0.05 (ölçüm) | 408 (%68) | **0** | 192 (%32) |

LONG hiç üretilemiyor: LONG prior'ın lo95'i (0.3817) 0.5'in altında → kapı-a LONG'a kapalı.
Motor tek yönlü (yalnız SHORT) bir makine.

## E.4 🔴 Tahmin gücü — motor triviyal kuralların ALTINDA

| Kural | n | İsabet | Ort. getiri | Net getiri | PF (net) | t |
|---|---|---|---|---|---|---|
| **Gölge karar motoru** | 192 | **0.5104** | **−0.40%** | **−0.50%** | **0.887** | −0.50 |
| Taban: hep LONG | 600 | 0.5133 | **+1.34%** | +1.24% | 1.388 | 3.05 |
| Taban: hep SHORT | 600 | 0.4867 | −1.34% | −1.44% | 0.683 | −3.05 |
| Taban: momentum işareti | 600 | 0.5283 | +1.22% | +1.12% | 1.344 | 2.77 |

- **FAZLALIK vs hep-LONG = −1.75 puan** · FAZLALIK vs hep-SHORT = +0.94 puan
- Motor, "hep LONG" ve "momentum" triviyal kurallarının **ikisini de kaybediyor**; yalnız
  "hep SHORT"u geçiyor — o da beceri değil, %68 ABSTAIN sayesinde **daha az maruziyet**.
- t = −0.50 → istatistiksel anlamlılık yok.

**Yorum:** Ölçüm penceresi YÜKSELİŞ eğilimliydi (5-bar ileri getiri ortalaması +1.34%). Motor,
öğrenme durumundaki SHORT üstünlüğünü (0.5579) prior olarak alıp yükselen piyasaya SHORT girdi
ve kaybetti. Bu, **ACI DERS #1'in doğrudan ampirik doğrulamasıdır**: yan-koşullu SHORT üstünlüğü
bir kenar değil, geçmiş DOWN rejiminin artefaktıydı — motor artefaktı alıp zarar etti.

> UYARI: Bu ölçüm **İN-SAMPLE**'dır (prior, pencerenin kendi canlı sonuçlarından türemiştir).
> Gerçek performans bundan daha KÖTÜ olma eğilimindedir. Sonuç negatif olduğu için exec-lag
> duyarlılığına gerek kalmadı (ACI DERS #7 kapısı zaten geçilmemiş sayılır).

## E.5 Nöral köprü — ölçüm sonrası

| | Önce | Sonra |
|---|---|---|
| Denenen | 600 | 600 |
| Üretilen | **600** (p̄=0.0585) | **0** |
| OOD susturulan | — | **600** (max\|z\|=**30.16** > 6.0) |
| Verdikt | (yoktu) | `DAĞILIM-DIŞI: … eğitim aralığı=1d. Köprü bilinçli SUSUYOR.` |

Köprü artık `sources.neural` Brier'ini ~0.42'lik çöple kirletmiyor; bunun yerine **nedenini
söyleyerek susuyor**. Kolonların dolması için 1m barlarda eğitilmiş bir model gerekiyor.

## E.6 Öğrenme döngüsünün durumu

| Kol | Önce | Sonra |
|---|---|---|
| learning (Beta/Brier/drift) | ✅ | ✅ |
| decision → prior | ❌ GLOBAL'e kilitli | ✅ yan-koşullu okunuyor |
| episodic → outcome | ❌ 248/248 `None` | ✅ çözücü kuruldu + `shadow_runner`'a bağlandı (11 test) |
| neural → sources | ❌ n=0, sessiz bozuk | ✅ guard'lı; dağılım-dışıyken **gerekçeli** susuyor |

Not: canlı epizotlar (278 adet, 2026-07-25) **hâlâ `bekleyen`** — 5 barlık ufuk dolmadı
(son işlem günü 24 Temmuz Cuma). Bu doğru davranıştır: çözücü geleceği uydurmaz. İlk gerçek
çözümler 31 Temmuz civarında birikmeye başlar.

## E.7 Güncellenmiş düzey

| Boyut | Önce | Sonra | Not |
|---|---|---|---|
| Öğrenme döngüsünün kapalılığı | 4.0 | **8.0** | 4 koldan 4'ü bağlı (biri veri bekliyor) |
| **Ölçülen tahmin gücü** | 1.5 | **1.5** | Değişmedi — artık **ölçüldü**: motor triviyal tabanın altında |
| **Operasyonel karar üretimi** | 1.0 | **3.0** | Kilitler açıklandı ve ölçüldü; üretimde bilinçli ABSTAIN |
| Dürüstlük / anti-overfit | 9.5 | **9.5** | Ölçüm kendi düzeltmemi reddetti — sistem çalışıyor |

**Nihai karar:** `AI_UTILITY_LAMBDA` **0.5'te bırakıldı** (üretimde %100 ABSTAIN sürüyor).
Denetim öncesi bu davranış bir hatanın yan etkisiydi; artık **ölçülmüş bir gerekçesi var**:
motor λ gevşetildiğinde para kaybediyor. Kapıyı açmak için önce OOS-kenarlı bir baz sinyal
gerekir — mimari değil.

*Bu rapor karar vermez ve kenar iddiası içermez (ACI DERS #3/#4).*
