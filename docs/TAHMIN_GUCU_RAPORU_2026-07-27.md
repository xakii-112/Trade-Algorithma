# Tahmin Gücü ve Karar Üretimi — Ölçüm Raporu (2026-07-27)

**Girdi:** `docs/GELISTIRME_RAPORU_2026-07-26.md` (plan) + `docs/AI_DENETIM_RAPORU_2026-07-26.md` (denetim)
**Amaç:** AI katmanının **canlı tahmin gücünü** ve **karar (ABSTAIN) mekanizmasını** geliştirmek.
**Disiplin:** Bu belge hiçbir kenar iddiası içermez. Ölçülen her şey kapı testleriyle birlikte raporlanır.
**Kalıcı kayıt:** `docs/EDGE_FINDINGS.md` **§23**.

> **Not:** bu oturum 2026-07-27 19:52'de cihazın beklenmedik yeniden başlatılmasıyla kesildi;
> kaldığı yerden (F4 + kalite kapısı + raporlama) tamamlandı. Kesinti anında hiçbir dosya
> bozulmamıştı, hiçbir ölçüm kaybedilmedi.

---

## 1. Yapılan işin özeti

| # | İş | Çıktı | Hüküm |
|---|---|---|---|
| F1 | Ufuk süpürmesi | `research/horizon_sweep.py` | **H\* YOK** — 28 kombinasyonun hiçbiri kapıları geçmedi |
| F3 | Gerçekçi spread | `research/spread_estimate.py` | Corwin–Schultz medyan **2,11 bps** |
| F4 | LONG bacağı teşhisi | `research/long_leg_diagnostic.py` | **H2 KAYMA** — bug değil, uygulama maliyeti |
| O1 | λ süpürmesi | `research/shadow_decision_backtest.py --lambda-sweep` | **λ=0,5'te KAL** |
| O3 | Üretim KPI'sı | `nihai/ai/selfcheck.py::exam_production` | %100 ABSTAIN artık **açık uyarı** üretiyor |
| F6 | Veri hijyeni | `nihai/signals/meta_features.py`, `research/train_neural.py` | σ≈0 özellik + vix=0 mayınları etkisiz |
| — | **Giriş kayması** (plansız bulgu) | `bar_close`/`hit_bar`/`slip_bps` kolonları | Canlı `hit` iki bileşene ayrıldı |

---

## 2. F1 — "%46,8 isabet yanlış ölçekte mi ölçüldü?" sorusunun cevabı: HAYIR

50.015 çözülmüş tahmin, 36 sembol, **H ∈ {1,5,15,30,60,120,390} × exec-lag ∈ {0,1,2,5}**
(28 kombinasyon) ile geriye dönük yeniden puanlandı. Kapılar: maliyet sonrası pozitif net getiri
**+** sembol×gün kümelenmiş t > Šidák kritik (3,116) **+** her iki zaman yarısında pozitif.

**`KABUL_ADAYLARI` = [] → H\* YOK.** 1 dakikadan tam seansa kadar hiçbir ufukta kenar yok.
Tek pozitif net getiri H390'da, t_küme ≤ 0,58 (gürültü). Ayrıntı: EDGE_FINDINGS §23.

Harness'ın dürüstlük ayakları: kümelenmiş t (1m'de örtüşen pencereler ham t'yi şişirir),
Šidák çoklu-test düzeltmesi, kesintisiz RTH dakika ızgarası (H=390 gece boşluğuna düşmesin),
zaman-bölmeli doğrulama ve **parite kapısı** (aşağıya bak).

---

## 3. Asıl bulgu — GİRİŞ KAYMASI (plansız, en değerli çıktı)

Parite kapısı kurulurken ortaya çıktı: logdaki `fwd_price`, +5 barın SIP kapanışıyla **birebir**
eşleşiyor (0,14–0,31 bps), ama giriş `price`'ı karar barının kapanışıyla eşleşmiyor. Giriş ve
çıkış **farklı kaynaklardan** geliyordu ve fark **sistematik olarak aleyhte**:

* ortalama kayma **+1,85 bps** sinyal yönünde (t = **21,2**, n = 52.210; 6/6 sembolde pozitif),
* medyan hareket ~13,5 bps → kaymanın payı hareketin **%7–22**'si,
* aynı tahminler bar kapanışında girilseydi isabet 0,4654 → **0,4937** (**+2,83 puan**),
* parite: canlı-giriş varyantı üretimi **%95,1** yeniden üretiyor → ölçüm hizalı, bulgu gerçek.

**Mekanizma:** yön canlı fiyattan hesaplanıyor (`_bias_and_confidence(price, …)`), giriş de o
fiyata yazılıyor, çıkış ise bar kapanışı. Canlı `hit` kolonu bu yüzden **yön becerisi** ile
**kovalama maliyetini** tek sayıda karıştırıyordu.

**Üretimde ayrıştırıldı** (`live_predict_service.py` + `prediction_tracker.py`):

| Kolon | Anlamı |
|---|---|
| `bar_close` | karar barının kapanışı (kayıt anında yazılır) |
| `hit_bar` | **saf yön becerisi** — bar kapanışında girilseydi isabet |
| `slip_bps` | **uygulama maliyeti** — canlı girişin sinyal yönündeki kayması |

`hit` bilerek **DEĞİŞMEDİ** (geriye uyum + gerçekleşen sonuç odur). Kolonlar yalnız kayıtçı
süreç yeniden başladıktan sonra dolmaya başlar (şema süreç belleğinde).

---

## 4. F4 — LONG bacağı: bug DEĞİL, kayma

`research/long_leg_diagnostic.py` dört hipotezi birbirinden ayırdı. Yöntem: **gün-eşleştirmeli
(paired) küme testi** — aynı sembol×gün kümesinde LONG ile SHORT ortalamalarının farkı tek
gözlem sayılır; böylece piyasa yönü/gün etkisi tasarım gereği düşer.

| Hipotez | Kanıt | Sonuç |
|---|---|---|
| H1 BUG (`is_long` normalizasyonu) | `hit` yeniden-üretimi LONG %99,94 / SHORT %99,95, ayrışma 0,0001 | **ELENDİ** |
| **H2 KAYMA** | kayma LONG **2,69** bps vs SHORT **1,07** bps; ham açık canlıda **0,0129** → barda **0,0002**; eşleştirilmiş t −2,78 → **−1,23** | **DOĞRULANDI** |
| H3 REJİM | eşleştirilmiş fark hem yükseliş (t=−1,36) hem düşüş (t=−0,50) günlerinde anlamsız | destek yok |
| H4 YAPISAL | bar girişinde LONG 0,494 ≈ SHORT 0,494 | destek yok |

**Uyarı:** "açık yok" ≠ "kenar var". Bar girişinde bile her iki tarafın net getirisi maliyet
sonrası negatif.

---

## 5. O1/O3 — karar mekanizması: %100 ABSTAIN kaza değil, ölçülmüş doğru davranış

λ ∈ {0,5 … 0} süpürmesi, zaman-bölmeli (ilk %60 seçim / son %40 doğrulama, 1800 bağlam):

* λ = 0,5 / 0,3 / 0,2 → motor hiç konuşmuyor (abstain 1,0),
* konuştuğu her λ'da **doğrulama fazlalığı ≤ 0** (en iyisi λ=0,02'de +0,0008 puan, PF 0,98),
* → **λ=0,5'te KAL.** Eşiği düşürmek zarar üretir; sessizlik doğru cevaptır.

`selfcheck.exam_production` (4. sınav) artık `FAZLALIK_vs_hep_long`, `abstain_orani`, `n_yonlu`
üçlüsünü karneye yazıyor ve `n_yonlu == 0` durumunda **"karar motoru HİÇBİR yön üretmiyor"**
uyarısını açıkça basıyor — denetimdeki B-1 bulgusunun sessiz kalmasının önüne geçer.

**Sınır:** prior `data/ai_learning_state.json`'dan gelir ve pencerenin canlı sonuçlarından
türer → SEÇİM bölmesi in-sample, DOĞRULAMA da prior kirliliğinden tam arınmış değildir
(ACI DERS #3/#4).

---

## 6. F6 — veri hijyeni (eğitimde bölme mayınları)

* `rr_ratio` (σ = 3,2e−15) ve `confluence` (sabit 1,0) gibi **dejenere kolonlar** eğitimde
  nötrleştirilir (`neutralize_degenerate`) — kolon **düşürülmez**, sıfırlanır: train/serve
  özellik hizası (`FEATURE_NAMES` tek kaynak) korunur.
* canlı `vix == 0` artık **eksik değer** sayılır ve nötr dolgulanır (`_vix_or_neutral`, 18.0);
  sıfır bir "sakin piyasa" sinyali gibi öğrenilmiyor.

---

## 7. Ne DEĞİŞMEDİ / ne iddia EDİLMİYOR

* Hiçbir kenar bulunmadı. `META_GATE` varsayılan KAPALI, `AI_DECISION_MODE` gölge, λ=0,5.
* Sinyal/eşik/execution mantığına dokunulmadı; eklenen her şey **ölçüm** veya **log**.
* `neural_shadow` KARAR-DIŞI kalmaya devam ediyor.
* F1 bir **keşif** süpürmesidir; aday çıksaydı bile OOS kanıtı sayılmazdı.

## 8. Sıradaki adımlar (öncelik sırasıyla)

1. **Kayıtçıyı yeniden başlat** — `bar_close`/`hit_bar`/`slip_bps` ancak o zaman dolmaya başlar.
   Birkaç gün sonra `hit_bar` ile `hit` arasındaki fark, kovalama maliyetinin canlı ölçüsü olur.
2. **Giriş kaymasını azalt** (tek somut iyileştirme yolu): yön hesabını karar barının
   KAPANIŞINDAN yap, canlı fiyatı yalnız uygulama simülasyonunda kullan. Bu, ölçülen +2,83
   isabet puanının bir kısmını geri kazanabilir — ama kenar üretmez, sadece maliyeti düşürür.
3. **Darboğaz değişmedi:** OOS-kenarlı baz sinyal yok (ACI DERS #5). Ufuk artık elenmiş bir
   hipotez olduğuna göre sıra yeni sinyal ailesinde veya Sharadar/CRSP PIT verisinde.
