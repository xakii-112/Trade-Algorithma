# Geliştirme Raporu — Tahmin Gücü ve Karar Üretimi

**Tarih:** 2026-07-26 · **Girdi:** `docs/AI_DENETIM_RAPORU_2026-07-26.md` (denetim + P0 uygulaması + ölçüm)
**Amaç:** "ölçülen tahmin gücü" ve "operasyonel karar üretimi" notlarını yükseltmek.
**Disiplin:** Bu belge hiçbir kenar iddiası içermez. Her iş paketinin bir **kabul kriteri** ve bir
**red kriteri** vardır; red kriteri tetiklenirse paket "başarısız" değil, **ölçülmüş negatif sonuç**
sayılır ve EDGE_FINDINGS'e yazılır (ACI DERS #3/#4).

---

## 0. TL;DR — üç cümle

1. **"Doğruluk yüzdesi" yanlış amaç fonksiyonu.** Bu veride isabet ile kârlılık *ters* korelasyonlu
   (`corr(confidence, hit) = −0.060` ama `corr(confidence, yönlü getiri) = +0.017`); isabeti
   kovalamak sistemi "kuruş toplayıcı"ya çevirir. Hedef **beklenti/PF** olmalı.
2. **Tek gerçek bilinmeyen ufuk.** 50.015 tahminin tamamı tek ufukta (`horizon_bars=5` → 1m barda
   **5 dakika**). Bu parametre hiç süpürülmedi. %46.8'in bir *ölçek artefaktı* mı yoksa gerçek
   sinyal kalitesi mi olduğunu yalnız **F1** cevaplayabilir — ve **geriye dönük** cevaplanabilir,
   haftalarca beklemeye gerek yok.
3. Karar motorunun notu düşük çünkü **hiçbir şey üretmemenin bir maliyeti yok**. Kapılar ölçülmüş
   değil, tahmin edilmiş parametrelerle kurulu (λ=0.5 keyfî ve geçilemez).

---

## 1. Kanıt tabanı (bu oturumda ÖLÇÜLDÜ — plan bunlara dayanıyor)

### 1.1 Canlı taban
| Ölçüm | Değer |
|---|---|
| Çözülmüş tahmin | **50.015** |
| İsabet | **0.4683** (SE 0.0022 → 0.5'ten **z = −14.2**) |
| Yönlü ortalama getiri | **−0.0072%** · PF **0.9428** |
| Kazanan / kaybeden | n=24.341 ort **+0.2427%** / n=25.302 ort **−0.2477%** |
| Ödül/risk | **0.980** (simetrik) → **başabaş isabet = 0.5050** |
| Ufuk çeşitliliği | **YOK** — 50.015/50.015 satır `horizon_bars=5` |
| Medyan mutlak hareket | **~0.135%** (≈13.5 bps) |

### 1.2 Güven skoru ters işaretli (5 dilim, n≈10k/dilim)
| Dilim | İsabet | Yönlü getiri | Mutlak hareket |
|---|---|---|---|
| 1 (en düşük güven) | **0.5222** | **−0.0239%** | 0.2325% |
| 2 | 0.4860 | −0.0133% | 0.2487% |
| 3 | 0.4601 | −0.0031% | 0.2461% |
| 4 | **0.4285** | **+0.0092%** | 0.2523% |
| 5 (en yüksek) | 0.4413 | −0.0037% | 0.2382% |

- Dilim-sırası ↔ isabet korelasyonu **−0.927** (monoton)
- `corr(conf, hit) = −0.0603` (z≈−13.5) · `corr(conf, yönlü getiri) = **+0.0165**` (z≈+3.7)
- **Ayırt edici test sonucu:** mutlak hareket dilimler arası **sabit** (0.2325→0.2382, **1.02×**)
  → bu bir *reversal/mikroyapı* deseni **DEĞİL**, skorun **yön tahmininde ters** olmasıdır.
- Dilim-1 zamanda kararlı: ilk yarı 0.5184, ikinci yarı 0.5260.
- **Tuzak:** Dilim-1 isabet 0.5222 > başabaş 0.5050 AMA **PF 0.8134** — yine para kaybediyor
  (küçük kazanç / büyük kayıp). İsabeti hedeflemenin neden yanlış olduğunun kanıtı budur.
- `pwin` rank korelasyonu −0.849 · `prob_cal` −0.623 (aynı yönde bozukluk).

### 1.3 Diğer dilimler
- **Taraf:** LONG aksiyonu isabet **0.4215** (n=1566) · SHORT **0.4963** (n=1368).
  Öğrenme durumu: LONG n=1114 **0.4104** [0.3817, 0.4394] · SHORT n=1873 **0.5579** [0.5353, 0.5803].
- **Saat (UTC):** 20:00 → 0.4537 (en kötü) … 22:00 → 0.4902 (en iyi). Aralık dar, hepsi <0.50.
- **Sembol:** isabet std **0.0208** vs şans beklentisi 0.0124 → **1.68×** (gerçek kesitsel dağılım).
  En kötü: CRWD 0.433 / AVGO 0.435 / ONDS 0.439 / TSM 0.441 / QCOM 0.449.
  En iyi: IBM 0.521 / AMZN 0.502 / DELL 0.496 / LMT 0.496 / NOW 0.486.

### 1.4 Model ve motor
- Kayıtlı nöral model `1f072998`: **OOS-AUC 0.5001 ± 0.0185**, DSR 0.0, 186.702 param, **1d barda eğitilmiş**.
- Nöral köprü: canlı vektörlerde `max|z| = 30.16` → yeni OOD guard 600/600 susturuyor (doğru davranış).
- Gölge karar motoru backtest (15 sembol, 120 gün, ufuk 5, 600 karar):

| Kural | n | İsabet | Ort. getiri | Net | PF | t |
|---|---|---|---|---|---|---|
| Motor (λ=0.05) | 192 | 0.5104 | **−0.40%** | −0.50% | 0.887 | −0.50 |
| Taban: hep LONG | 600 | 0.5133 | **+1.34%** | +1.24% | 1.388 | 3.05 |
| Taban: momentum | 600 | 0.5283 | +1.22% | +1.12% | 1.344 | 2.77 |

  → **FAZLALIK vs hep-LONG = −1.75 puan.** λ=0.5 (üretim) → 600/600 ABSTAIN.
- Kapı-b matematiği: kazanç≈kayıp≈0.04, |CVaR95|≈0.09, λ=0.5 → E>0 için **post > 1.06** gerekir = imkânsız.

### 1.5 Bilinen veri kusurları (önkoşul)
`rr_ratio` sabit 2.0 (eğitim σ = 3.2e-15) · `confluence` sabit 1.0 · canlı `vix` yalnız 9 eşsiz
değer ve içinde **0.0** · `hour` canlı 9–20 vs eğitim μ=4.38/σ=0.485 · 4 eski dataset'te 6 kolon yok.

---

## 2. İş paketleri — TAHMİN GÜCÜ

> Sıralama ölçülmüş kanıta göre. **F1 diğer her şeyden önce gelir**: sonucu, kalan paketlerin
> anlamlı olup olmadığını belirler.

---

### F1 — Ufuk süpürmesi (GERİYE DÖNÜK) ⭐⭐⭐
**Boyut:** M · **Bağımlılık:** yok · **Süre:** saatler (bekleme YOK)

**Neden:** 50.015 tahminin tamamı `horizon_bars=5`. 1m barda bu **5 dakika**. `event_breakout`
tezi (kırılım devam eder) 5 dakikalık ufukta mikroyapı gürültüsüyle yarışır. Bu parametre hiç
denenmedi ve sistemin en büyük ölçülmemiş serbestlik derecesi.

**Kritik nokta:** Bunu beklemeden ölçebilirsin. `nihai/data/alpaca_history.py` ücretsiz SIP planında
5 yıllık 1m tarihsel bar veriyor → mevcut 50k tahminin **hepsi** geriye dönük yeniden puanlanabilir.

**Yapılacak — yeni dosya `research/horizon_sweep.py`:**
1. `data/predictions_log.csv` oku (symbol, bar_ts, direction, confidence, price).
2. Her sembol için `alpaca_history` ile ilgili tarih aralığında 1m bar çek (sembol başına tek çağrı, cache'le).
3. Her tahmin için H ∈ **{1, 5, 15, 30, 60, 120, 390}** barda ileri getiri + `hit` hesapla.
   - Karar barı = `bar_ts`; giriş `bar_ts+1` kapanışı (look-ahead yok, `--exec-lag` ile ayarlanabilir).
4. Her H için raporla: `n`, `isabet`, `yönlü ort getiri`, `PF`, `ödül/risk`, `başabaş isabet`,
   `t-stat`, ve **güven dilimi × H** çapraz tablosu.
5. `--exec-lag {0,1,2,5}` duyarlılığı (ACI DERS #7 kapısı) her H için.

**Kabul kriteri:** En az bir (H, exec-lag≥1) kombinasyonunda yönlü ortalama getiri **maliyet
sonrası pozitif** ve `t > 2`, **ve** bu H hem ilk yarı hem ikinci yarı alt-örnekleminde pozitif.
**Red kriteri:** Tüm H'lerde net getiri ≤ 0 → **ölçüm penceresi/sinyal ailesi bu ufuklarda ölü**;
F2–F5 ertelenir, doğrudan yeni sinyal ailesine geçilir. EDGE_FINDINGS §23 olarak yazılır.

**Not:** Süpürme sonucu H* bulunursa, **F1b** olarak `nihai/execution/prediction_tracker.py`'ye
çoklu-ufuk kolonları eklenir (`fwd_ret_h{H}` / `hit_h{H}` merdiveni; mevcut `hit`/`fwd_ret_pct`
geriye uyumluluk için h=5'te kalır) ki kanıt döngüsü ileriye doğru da çoklu ufuk biriktirsin.

---

### F2 — `confidence` skorunun yeniden kalibrasyonu ⭐⭐⭐
**Boyut:** M · **Bağımlılık:** F1 (H* bilinmeli) · **Süre:** 1 gün

**Neden:** 5 dilimde monoton ters ilişki (rank ρ = −0.927), hareket büyüklüğü sabit (1.02×).
Bu bir mikroyapı artefaktı değil; skor yönü sistematik olarak yanlış tahmin ediyor. Aynı anda
skor **getiri** ile pozitif korelasyonlu (+0.0165) — yani içinde bilgi var, işareti/ölçeği bozuk.

**Yapılacak:**
1. **Önce hata ara.** `nihai/signals/strategy.py` içinde `confidence`'ı üreten toplama zincirini
   çıkar; her bileşenin (`quality_score`, `confluence`, `rr_ratio`, momentum terimleri) gerçekleşen
   **yönlü getiri** ile tek tek korelasyonunu ölç. Bileşen bazında bir işaret hatası varsa orada görünür.
2. Hata yoksa: `research/confidence_calibration.py` — `confidence → E[yönlü getiri | H*]`
   izotonik regresyon; zaman-bölmeli (ilk %60 fit, son %40 OOS).
3. Kalibre skoru **yeni bir kolon olarak** (`confidence_cal`) logla; eskisini DEĞİŞTİRME
   (parite + geri dönüş).

**Kabul kriteri:** OOS bölmede `confidence_cal` dilimleri ile yönlü getiri arasında **monoton
artan** ilişki ve en üst dilimin net getirisi pozitif.
**Red kriteri:** OOS'ta monotonluk kaybolursa skor gürültüdür → `confidence` karar zincirinden
tamamen çıkarılır (bugün zaten kullanılmıyor sayılır).

> ⛔ **YAPMA:** "Skoru ters çevir ve kâr et." Aleyhte kanıttan ters-yön kenarı türetmek
> ACI DERS #3/#7 ihlalidir ve `decision.py` bunu kod düzeyinde de yasaklar.

---

### F3 — Gerçekçi doldurma (fill) modeli ⭐⭐
**Boyut:** M · **Bağımlılık:** yok (F1 ile paralel) · **Süre:** 1 gün

**Neden:** Medyan hareket **~13.5 bps**. Bu isimlerde etkin spread 2–5 bps → hareketin
**%15–40'ı**. Şu an hiç modellenmiyor; dolayısıyla bugünkü %46.8 bile **iyimser**.

**Yapılacak:**
1. Geriye dönük ucuz tahmin: 1m bar `high/low`'dan **Corwin–Schultz** etkin spread tahmincisi
   (veya getiri otokovaryansından **Roll** tahmincisi). Yeni dosya `research/spread_estimate.py`.
2. F1'in ufuk süpürmesine `--spread-model {none,corwin,roll,fixed_bps}` bayrağı ekle; net getiri
   bu maliyetle yeniden hesaplansın.
3. İleriye dönük: sinyal anında bid/ask'i `predictions_log`'a logla (`bid`, `ask` kolonları) —
   `nihai/execution/live_predict_service.py`.

**Kabul kriteri:** F1'in H* bulgusu spread modeli **açıkken de** hayatta kalır.
**Red kriteri:** H* yalnız sıfır-maliyet varsayımıyla pozitifse → kenar yok, kapat.

---

### F4 — LONG bacağının teşhisi ⭐⭐
**Boyut:** S · **Bağımlılık:** yok · **Süre:** yarım gün

**Neden:** LONG isabet 0.4215 vs SHORT 0.4963; LONG prior lo95 = 0.3817. Tüm ortalamayı LONG
bacağı aşağı çekiyor ve karar motorunun LONG kapısını yapısal olarak kilitliyor.

**Yapılacak — `research/long_leg_diagnostic.py`:**
1. SPY (veya `nihai/signals/market_regime.py` çıktısı) ile her tahmini **rejim etiketiyle** eşle.
2. `isabet ~ taraf × rejim` tablosunu çıkar.
3. Ayrıca `is_long` özelliğinin sinyal üretiminde ve `meta_features`'ta **aynı** anlama geldiğini
   doğrula (Tur-15'te `direction=+/-` normalizasyon dersi vardı — aynı sınıftan bir hata olabilir).

**Kabul kriteri:** LONG'un zayıflığı yalnız düşüş rejiminde görülüyorsa → **rejim**, aksiyon:
rejim-koşullu prior (O2). Her rejimde görülüyorsa → **bug**, aksiyon: sinyal zincirini düzelt.
**Red kriteri:** yok — bu bir teşhis paketidir, her iki sonuç da bilgidir.

---

### F5 — Evren filtresi (ex-ante likidite) ⭐
**Boyut:** S · **Bağımlılık:** F3 · **Süre:** yarım gün

**Neden:** Semboller arası isabet std 0.0208 vs şans 0.0124 → **1.68×**, gerçek dağılım var.
En kötüler (CRWD/AVGO/TSM/QCOM) yüksek volatilite + geniş spread isimleri.

**Yapılacak:** `nihai/execution/watchlist.py`'ye **ex-ante** filtre: medyan spread (F3'ten) ve ADV
eşiği. Filtre **gerçekleşen isabete göre KURULMAZ** — bu in-sample seçim olur.

**Kabul kriteri:** Ex-ante spread eşiği ile daralan evrende F1'in H* bulgusu güçlenir.
**Red kriteri:** Fark yoksa filtre eklenmez (gereksiz karmaşıklık).

---

### F6 — Veri hijyeni (ÖNKOŞUL, tek başına kazanç getirmez) ⭐⭐
**Boyut:** S · **Bağımlılık:** yok · **Süre:** yarım gün — **F1'den ÖNCE yapılmalı**

1. `nihai/signals/meta_features.py`: `rr_ratio` (sabit 2.0) ve `confluence` (sabit 1.0)
   FEATURE_NAMES'ten çıkar **veya** eğitimde σ<1e-6 özellikleri otomatik düşür
   (`research/train_neural.py` + `train_meta.py`). σ=3.2e-15 bölme mayınıdır.
2. Canlı `vix == 0` → **eksik değer** say (nötr dolgu), 0'ı geçerli VIX gibi modele verme.
   Kaynak: `nihai/signals/strategy.py` VIX önbelleği.
3. `hour` özelliğini ya kaldır ya da **bar aralığına göre normalize** et (1d'de anlamsız, 1m'de
   seans-içi konum olarak yeniden tanımla).
4. Eski 4 dataset'i (6 eksik kolonlu) ya yeniden üret ya da `deprecated` işaretle.

**Kabul kriteri:** `research/train_neural.py` çıktısındaki `feat_sigma` içinde <1e-6 kalmaz;
canlı `vix` dağılımında 0.0 yok.

---

## 3. İş paketleri — OPERASYONEL KARAR ÜRETİMİ

---

### O1 — λ'yı ölçümle seç (tahminle değil) ⭐⭐⭐
**Boyut:** S · **Bağımlılık:** yok · **Süre:** yarım gün

**Neden:** `AI_UTILITY_LAMBDA=0.5` keyfî ve matematiksel olarak geçilemez (post>1.06 gerekir);
0.05 karar üretiyor ama para kaybediyor. Aradaki hiçbir değer denenmedi.

**Yapılacak:** `research/shadow_decision_backtest.py`'ye `--lambda-sweep` ekle: λ ∈
{0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0} için (ABSTAIN oranı, FAZLALIK_vs_hep_long,
FAZLALIK_vs_momentum) eğrisini çıkar. **Zaman-bölmeli**: λ ilk %60'ta seçilir, son %40'ta doğrulanır.

**Kabul kriteri:** OOS bölmede fazlalığı pozitif yapan bir λ* bulunursa `.env`'e yazılır.
**Red kriteri:** Hiçbir λ pozitif fazlalık vermiyorsa → **λ=0.5'te kal ve bunu belgeye yaz**.
Bugünkü %100 ABSTAIN o zaman *ölçülmüş* doğru davranıştır, kaza değil.

---

### O2 — Koşullu prior tablosu ⭐⭐⭐
**Boyut:** L · **Bağımlılık:** F4 · **Süre:** 1–2 gün

**Neden:** GLOBAL/LONG/SHORT dönem-toplamları rejimleri karıştırıyor. GLOBAL'in CI tabanı
tanım gereği 0.5'i geçemiyor; LONG'unki 0.3817. Bu prior'la kapı-a hiçbir zaman anlamlı olamaz.

**Yapılacak — `nihai/ai/learning.py`:**
1. Posterior tablosunu **(rejim × taraf × güven dilimi)** anahtarlı hale getir. 50.015 çözülmüş
   tahminle hücre başına ~500–3000 örneklem düşer (yeterli).
2. `decision._prior_from_state()`'in arama sırasını genişlet:
   `(rejim|taraf|dilim) → (rejim|taraf) → taraf → GLOBAL → yok`, her seviyede
   `_MIN_SIDE_PRIOR_N` benzeri asgari örneklem kapısı.
3. **Kritik:** hücre seçimi karar anında bilinen bilgiyle yapılmalı (look-ahead yok) ve
   posterior **zamanda kayan pencereyle** güncellenmeli, tüm geçmişle değil.

**Kabul kriteri:** O1'in λ süpürmesi koşullu prior ile tekrarlandığında OOS fazlalık iyileşir.
**Red kriteri:** Hücre sayısı arttıkça OOS fazlalık düşerse (çoklu-test şişmesi) → geri al,
`research/pbo.py` (CSCV) ile aşırı-uyum olasılığını raporla.

---

### O3 — Üretim KPI'sı: "hiçbir şey üretmeme" ölçülsün ⭐⭐
**Boyut:** S · **Bağımlılık:** yok · **Süre:** yarım gün

**Neden:** Bugün %100 ABSTAIN sonsuza kadar "güvenli" görünüyor; hiçbir metrik bunu cezalandırmıyor.
Sistem ölçülmeden bozuk kalabilir (nitekim 2 yapısal kilit aylarca fark edilmedi).

**Yapılacak:**
1. `nihai/ai/selfcheck.py`'ye **dördüncü sınav**: `exam_production` —
   `research/shadow_decision_backtest.run()` çağırır, `FAZLALIK_vs_hep_long`,
   `abstain_orani`, `n_yonlu` alanlarını karneye yazar.
2. `report_card()` çıktısına ve `daily_summary --brain` bloğuna ekle.
3. Karne trendi: fazlalık üst üste N hafta ölçülemiyorsa (n_yonlu=0) **açık uyarı** üret:
   "karar motoru hiçbir yön üretmiyor — kapılar ölçülmemiş olabilir".

**Kabul kriteri:** Karnede `üretim` satırı görünür ve `n_yonlu=0` durumu uyarı üretir.

---

### O4 — Ufuk/ölçek uyumu ⭐⭐
**Boyut:** M · **Bağımlılık:** F1 · **Süre:** 1 gün

**Neden:** `shadow_runner` **günlük** barlarda karar veriyor; canlı sinyal **1m**. İki farklı
sistem tek `ai_learning_state.json` prior'ıyla puanlanıyor. Nöral model de 1d'de eğitilmiş
(OOD guard'ın 30σ yakalamasının kökü bu).

**Yapılacak:** F1'in H* sonucuna göre **tek ölçek** seç ve üçünü hizala:
(a) `shadow_runner._fetch_bars` aralığı, (b) `prediction_tracker` ufku, (c) nöral eğitim dataset'i.
Alternatif: prior'ı ölçek-anahtarlı yap (`GLOBAL@1m`, `GLOBAL@1d`) ve karışmasınlar.

**Kabul kriteri:** `neural_shadow.selfcheck()` `max|z| < 6` ile geçer ve `n_ok > 0` olur.

---

### O5 — Gölge tur zamanlaması ⭐
**Boyut:** XS · **Bağımlılık:** yok · **Süre:** 15 dk

`NihaiAlgo-ShadowRunner-Hourly` şu an 06:30–13:30 UTC (ABD açılışı 13:30) → AI piyasa açılmadan
düşünüyor, günlük barlar değişmediği için 8 turun 6'sı birebir aynı (kararların %75.8'i tekrar,
HRP 8 turda 2 farklı sonuç). Cumartesi de koşuyor.

**Yapılacak:** Tetiği ABD seansına kaydır (13:30–20:00 UTC) veya günde 1 kez koş; hafta sonunu kapat.
`register-*.ps1` desenini izleyerek yeni bir kayıt script'i.

**Kabul kriteri:** `ai_decisions.jsonl`'de aynı gün+sembol+posterior tekrarı **%10 altına** iner.

---

## 4. Uygulama sırası

```
FAZ 0 (yarım gün)  F6 veri hijyeni  ·  O5 zamanlama        ← ucuz, her şeyi temizler
        │
FAZ 1 (1-2 gün)    F1 UFUK SÜPÜRMESİ (geriye dönük)        ← KARAR NOKTASI
        │          F3 spread modeli (paralel) · F4 LONG teşhisi (paralel)
        │
        ├── H* BULUNDU ──→ FAZ 2: F1b çoklu-ufuk logu · F2 confidence kalibrasyonu
        │                          F5 evren filtresi · O4 ölçek hizalama
        │                          → sonra O1 λ süpürmesi · O2 koşullu prior
        │
        └── H* YOK ──────→ FAZ 2': EDGE_FINDINGS §23 "bu sinyal ailesi bu ufuklarda ölü"
                                   O1 λ süpürmesini yine koş (λ=0.5 gerekçesini belgelemek için)
                                   O3 üretim KPI'sı · sonra YENİ SİNYAL AİLESİ ara
FAZ 3 (her durumda) O3 üretim KPI'sı → karneye bağla
```

**En kritik karar noktası FAZ 1'in sonu.** F1 negatif çıkarsa F2/F5/O2'ye harcanacak emek boşa
gider — bu yüzden hiçbiri F1'den önce başlamamalı.

---

## 5. Disiplin kapıları — bunları YAPMA

1. **Ters çevirme kenarı türetme.** `confidence` ters korelasyonlu diye işaretini çevirip
   "kenar bulduk" deme. Önce hata ara, sonra kalibre et (ACI DERS #3/#7; `decision.py` kod
   düzeyinde de yasaklıyor).
2. **Gerçekleşen isabete göre sembol/saat seçme.** IBM 0.521, CRWD 0.433 — bu in-sample seçimdir.
   Filtre yalnız **ex-ante** metrikle (spread, ADV) kurulur.
3. **İsabet oranını optimize etme.** Ölçüldü: dilim-1 isabet 0.5222 ama PF 0.8134. İsabet ve
   kârlılık bu veride ters yönde hareket ediyor.
4. **Tek bölmeye güvenme.** Her kabul kriteri **zaman-bölmeli OOS** ister; `research/purged_cv.py`
   ve `research/pbo.py` (CSCV) hazır, kullan.
5. **exec-lag testini atlama.** ACI DERS #7 (§22): tüm metrikler yeşilken bile 1 bar gecikme
   kenarı öldürebilir. F1'in kabul kriteri `exec-lag ≥ 1` şartını içeriyor — gevşetme.
6. **`AI_DECISION_MODE=enforce` arama.** Yok, kasıtlı. Bu planın hiçbir maddesi onu getirmez.
7. **Overfit korumalarını "iyileştirme" diye gevşetme** (CLAUDE.md çalışma ilkesi).

---

## 6. Bu oturumda ZATEN yapılanlar (referans)

| İş | Dosya | Durum |
|---|---|---|
| Nöral köprü OOD guard + verdikt + sayaçlar | `nihai/execution/neural_shadow.py` | ✅ |
| Yan-koşullu prior (LONG/SHORT, n≥200) + artefakt uyarısı | `nihai/ai/decision.py` | ✅ |
| Epizot sonuç çözücüsü (öğrenme döngüsü 4. kolu) | `nihai/ai/episode_resolver.py` **(yeni)** | ✅ |
| Çözücünün `shadow_runner`'a bağlanması | `nihai/ai/shadow_runner.py` | ✅ |
| Çözücü testleri (11 adet) | `tests/test_ai_episode_resolver.py` **(yeni)** | ✅ |
| Gölge karar backtest harness'ı (taban karşılaştırması + exec-lag) | `research/shadow_decision_backtest.py` **(yeni)** | ✅ |
| Denetim raporu | `docs/AI_DENETIM_RAPORU_2026-07-26.md` **(yeni)** | ✅ |

Kod sağlığı: **ruff temiz · mypy 0 hata / 87 dosya · tam paket exit 0**.
⚠️ **Hiçbiri commit edilmedi** (denetim öncesinde de 15 dosya commitsizdi). İlk iş: temiz commit.

---

## 7. Gerçekçi beklenti

Genel yönlü getiri **−0.0072%**, PF **0.943**, ödül/risk 0.980, başabaş isabet 0.5050.
Bu veride, bu ufukta kenar **yok**.

Bu planın hedefi "%46.8 → %55" değil. Hedef: **%46.8'in bir ölçek/ölçüm artefaktı mı yoksa
gerçek sinyal kalitesi mi olduğunu kesin olarak öğrenmek.** F1 bunu birkaç saatte cevaplıyor.
Cevap "artefakt" ise F2–F5 anlamlı; "gerçek" ise sinyal ailesi değişmeli ve F2–F5'e harcanan
her saat kayıp olurdu.

Projenin darboğazı bir kez daha aynı: **mimari değil, OOS-kenarlı baz sinyal.**

*Bu rapor karar vermez ve kenar iddiası içermez (ACI DERS #3/#4).*
