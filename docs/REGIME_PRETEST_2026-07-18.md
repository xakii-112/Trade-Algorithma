# HMM Öncesi Rejim Ön-Testi — Rapor

**Tarih:** 2026-07-18  ·  **İş:** ENTEGRASYON_RAPORU §2.5 P1-1 (rapor-only, bağımlılıksız)  ·  **Model:** Opus 4.8

## 0. Soru ve karar bağlamı

ENTEGRASYON_RAPORU §1.6-HMM, HMM rejim tespitini iki şarta bağladı. **Şart-1 (bu test):** *"önce sıfır-bağımlılık ön-test: mevcut forward tahminlerini basit rejim etiketlerine bölüp NO_EDGE oranı rejime göre ayrışıyor mu bak — ayrışmıyorsa HMM'e hiç girme."* Şart-2 (expanding-window/walk-forward fit) yalnız şart-1 geçerse gündeme gelir; tam-örneklem HMM fit'i klasik look-ahead tuzağıdır.

Bu test **hiçbir bağımlılık kurmaz** (hmmlearn dahil), üretim koduna dokunmaz, `data/` ve `nihai/` altına yazmaz. `data/predictions_log.csv` yalnız geçici konuma kopyalanıp okunur.

## 1. Ön-KAYITLI karar kuralı (deney öncesi sabitlendi)

> **"Ayrışma VAR"** = en az bir hücrede (n≥300) isabet oranı havuz oranından ≥5 puan sapıyor **VE** Bonferroni-düzeltmeli (4 hücre) p<0.05. Aksi halde **"ayrışma YOK → HMM gündemden düşer"** (rapor §1.6-HMM şart-1).

Ek terminal durum (dürüstlük): n≥300 olan hücre 2'den azsa forward pencere tek rejime sıkışmış demektir → ayrışma **test edilemedi** (ne gösterildi ne çürütüldü); bu da HMM için pozitif kanıt sağlamaz → şart-1 karşılanmaz.

## 2. Yöntem ve look-ahead önlemleri

- **Veri:** forward kayıtçı `predictions_log.csv` çözülmüş (`hit∈{0,1}`) yön-tahminleri. Pencere: **2026-06-23 → 2026-07-17**.
- **Rejim kaynağı:** SPY günlük (yfinance, auto-adjust). Etiketler **yalnız önceki işlem gününün kapanışına kadarki** veriyle (look-ahead yasak):
  - **TREND:** önceki gün kapanışı SMA200 üstünde = `UP`, altında = `DOWN`.
  - **VOL:** önceki güne kadarki 20g gerçekleşen vol, KENDİ trailing 252g medyanına göre `HIGH`/`LOW`. (Tam-örneklem tercil = look-ahead → kullanılmadı.)
- **Eşleme:** `merge_asof(direction='backward', allow_exact_matches=False)` → tahmin günü D için yalnız t<D günlerinin durumu; tahminin kendi gününün kapanışı ASLA kullanılmaz.
- **2×2 = 4 hücre:** UP-LOW, UP-HIGH, DOWN-LOW, DOWN-HIGH.
- **"NO_EDGE payı" tanımı:** `predictions_log`'da NO_EDGE bir *satır* alanı değil — `verdict()`/`forward_harness` onu bir KÜME hükmü olarak (|isabet-0.5|<0.03) tanımlar. Satır-başı proxy olarak forecast'in kalibre olasılığının nötr bandı kullanıldı: |prob_cal-0.5|<0.03 (aynı 0.03 eşiği; §15.1 "prob_cal≈0.5 = öngörü gücü yok"). Karar kuralı hücre **isabet** ayrışmasına dayanır; NO_EDGE payı betimleyici ek metriktir.
- **İstatistik:** `research/deflated.py` `wilson_ci` + `binom_p_value` (yeniden yazılmadı). Hücre verdict etiketi `forward_harness._verdict` ile birebir (EDGE_PP=3.0 + Wilson-GA 0.5 içerme kontrolü).

## 3. Rejim çeşitliliği (tahmin günleri)

Forward pencere kısadır; hangi rejimlerin gerçekten gözlendiği kritik. Tahmin günü başına atanan rejim dağılımı:

| rejim hücresi | ayrı tahmin günü |
|---|---|
| UP-HIGH | 12 |

**Sınıflandırıcı dejenerasyon kontrolü (kritik):** yukarıdaki tek-hücre yoğunlaşması bir hata DEĞİL, kısa pencerenin gerçek özelliğidir. İndirilen ~2 yıllık SPY (ısınma dahil) arka plan rejim dağılımı, sınıflandırıcının başka etiketler de ürettiğini gösterir:

| rejim hücresi | ~2y işlem günü |
|---|---|
| UP-LOW | 163 |
| UP-HIGH | 120 |
| DOWN-HIGH | 12 |

Yani sınıflandırıcı UP/DOWN ve HIGH/LOW'u ayırt edebiliyor; forward penceresi (2026 orta-Haz → orta-Tem) tesadüfen **bütünüyle UP-HIGH** rejimde. (Not: DOWN-LOW bu 2y'de hiç görülmedi — düşük-vol düşüş trendi nadirdir; test ileride tekrarlanırken 4-hücre her zaman seyrek olabilir.)

## 4.1 Sonuç — TÜM çözülmüş yön-tahminleri (birincil)

Rejim-atanmış çözülmüş tahmin: **36544**  ·  havuz isabet oranı: **47.1%**  ·  eşleşmeyen (ısınma-öncesi): 0

| hücre | n | isabet | Wilson %95 GA | p(havuz) | p(0.5) | NO_EDGE payı | ort. fwd_ret% | hücre verdict |
|---|---|---|---|---|---|---|---|---|
| UP-LOW | 0 | n/a | [n/a, n/a] | nan | nan | n/a | +nan | NO_DATA |
| UP-HIGH | 36544 | 47.1% | [46.6%, 47.6%] | 1.000 | 0.000 | 79.0% | -0.021 | NO_EDGE |
| DOWN-LOW | 0 | n/a | [n/a, n/a] | nan | nan | n/a | +nan | NO_DATA |
| DOWN-HIGH | 0 | n/a | [n/a, n/a] | nan | nan | n/a | +nan | NO_DATA |

## 4.2 Sonuç — NO_EDGE-DIŞI alt küme (|prob_cal-0.5|≥0.03) — sağlamlık kontrolü

Rejim-atanmış çözülmüş tahmin: **7681**  ·  havuz isabet oranı: **47.0%**  ·  eşleşmeyen (ısınma-öncesi): 0

| hücre | n | isabet | Wilson %95 GA | p(havuz) | p(0.5) | NO_EDGE payı | ort. fwd_ret% | hücre verdict |
|---|---|---|---|---|---|---|---|---|
| UP-LOW | 0 | n/a | [n/a, n/a] | nan | nan | n/a | +nan | NO_DATA |
| UP-HIGH | 7681 | 47.0% | [45.9%, 48.1%] | 1.000 | 0.000 | 0.0% | -0.023 | NO_EDGE |
| DOWN-LOW | 0 | n/a | [n/a, n/a] | nan | nan | n/a | +nan | NO_DATA |
| DOWN-HIGH | 0 | n/a | [n/a, n/a] | nan | nan | n/a | +nan | NO_DATA |

## 5. Verdikt

- **Birincil (tüm çözülmüş):** AYRIŞMA TEST EDİLEMEDİ → n≥300 olan hücre <2 (['UP-HIGH']). Forward pencere tek rejime sıkıştığından ayrışma NE gösterilebildi NE çürütüldü → HMM için pozitif kanıt YOK; şart-1 karşılanmadı → şimdilik HMM'e girme (pencere genişleyince tekrar koş).
- **Sağlamlık (NO_EDGE-dışı):** AYRIŞMA TEST EDİLEMEDİ → n≥300 olan hücre <2 (['UP-HIGH']). Forward pencere tek rejime sıkıştığından ayrışma NE gösterilebildi NE çürütüldü → HMM için pozitif kanıt YOK; şart-1 karşılanmadı → şimdilik HMM'e girme (pencere genişleyince tekrar koş).

**NİHAİ:** ayrışma bu pencerede **test edilemedi** (rejim çeşitliliği yetersiz) → HMM için pozitif kanıt YOK, şart-1 karşılanmadı → **şimdilik HMM'e girme**; forward log birden çok rejim görecek kadar uzayınca (aylar) testi tekrarla.

## 6. Sınırlılıklar (dürüst)

- **Kısa pencere / tek piyasa dönemi:** forward log yalnız **2026-06-23 → 2026-07-17** (~haftalar, ~10-12 işlem günü) kapsıyor. Bu, rejim çeşitliliğini ciddi biçimde sınırlar; yukarıdaki gün-dağılımı tablosu kaç rejimin gerçekten gözlendiğini gösterir. Ayrışma-YOK sonucu 'rejimler fark yaratmaz' değil, 'bu pencerede fark GÖSTERİLEMEDİ' diye okunmalı.
- **Tek gösterge = SPY:** rejim SPY'a göre; sembol-özel rejimler (sektör/beta) yakalanmaz.
- **Havuz karşılaştırması:** hücre, kendini de içeren havuza karşı test edilir (kural metni böyle); hafif muhafazakâr/otokorelasyonlu — büyük-n'de etkisi küçük.
- **Forecast zaten NO_EDGE:** havuz isabeti ~%47 (sistem geneli NO_EDGE). Rejim koşullama olsa olsa zaten kenarsız bir forecast'i böler; taban sinyalde kenar yaratmaz (ACI DERS #5).
- **bar_ts tarih sınırı:** tz-aware (ET) ve naive zaman damgaları karışık; öğleden sonra barlarında ET/UTC tarih ayrımı önemsiz, ama gece-yarısı sınırındaki nadir barlar ±1 gün kayabilir (rejim günlük olduğundan etki ihmal edilebilir).

## 7. Tekrar üretmek için

```bash
.venv/Scripts/python.exe -m research.regime_pretest
```
