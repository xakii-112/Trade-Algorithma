# Tur-18 (2026-07-25) — Pano kapsamı · nöral gölge köprüsü · watchdog tetiği

> Bu belge iki oturumun çıktısını taşır: **§1–§5** pano/arayüz revizyonu,
> **§6–§8** Tur-17 raporu §6'daki açık kalemlerin kapatılması.

## Bölüm A — Pano Kapsam Analizi & Arayüz Revizyonu

**Soru:** Backend/kod olarak mevcut olup arayüzde olmayan neler var?
**Yanıt:** Aşağıdaki denetim + uygulanan revizyon. Yöntem: `nihai/` altındaki 84 modülün
public yüzeyi AST ile çıkarıldı, `web_dashboard.py`'deki 29 uç ve `web_dashboard.html`'deki
9 sekme ile karşılaştırıldı.

---

## 1. Denetim sonucu — revizyon ÖNCESİ durum

### 1.1 Rakamlar

| | Önce | Sonra |
|---|---|---|
| HTTP ucu | 29 | **50** |
| Pano sekmesi | 9 | **13** |
| Panodan erişilemeyen üretim modülü | ~35 | ~12 (bilinçli, §4) |
| Test | 1678 | **1698** |

### 1.2 Backend'de var, panoda yoktu

**A. Uç ZATEN vardı ama arayüz hiç çağırmıyordu** (en ucuz kayıp):

| Uç | Ne verir | Durum |
|---|---|---|
| `GET /api/forward-test` | Canlı kapalı-döngü doğrulama: kapsama, hedefe ilerleme, makine-okunur verdikt (`NO_DATA`/`NO_EDGE`/`WEAK_POSITIVE`/`INVERTED`) | HTML'de **0 referans** |
| `GET /api/tearsheet` | quantstats gerçekleşmiş-getiri tear-sheet'i (HTML) | HTML'de **0 referans** |

Projenin kendi tanımıyla **#1 önceliği** olan forward-test paneli panoda görünmüyordu.

**B. `nihai/ai` — Quant Beyin katmanının TAMAMI (22 modül, ~11.500 satır) panoda yoktu.**
Tur-13/15/16/17'de inşa edilen katman yalnız CLI + JSONL üzerinden erişilebilirdi:

- `brain` (günlük brif, düşünme döngüsü) · `agent` (soru-cevap + hipotez kuyruğu)
- `learning` (Beta posterior, Brier, güvenilirlik kovaları, Page-Hinkley drift)
- `decision` (gölge Bayes kararları) · `episodic` (epizodik hafıza + DERS kataloğu)
- `knowledge` (45 kavramlık quant bilgi çekirdeği + topolojik müfredat)
- `registry` (model kartları, A/B) · `selfcheck` (üç bölümlü öz-değerlendirme karnesi)
- `models`/`neural` (parametre kapasite merdiveni) · `portfolio` (HRP/ERC/BL/CVaR-LP/Kelly)
- `data_health` (veri sağlığı karnesi) · `text_intel` (ton/yenilik + takvim farkındalığı)
- `shadow_runner` · `router` · `memory` · `reasoner` · `consolidate` · `explain`
- `quant_math` (60+ fonksiyon) · `advanced_math` (Heston/Merton/Hawkes/BOCPD/imza)

**C. Piyasa istihbaratı — motorda çalışıyor, operatör göremiyordu:**

| Modül | Ne yapar | Panoda |
|---|---|---|
| `data/event_calendar` | Bilanço/temettü/bölünme + Madde 17/25 guard'ları | yok |
| `data/macro_calendar` | FRED risk-off kapısı (VIX/10y-2y/HY spread) | yok |
| `data/insider` | SEC Form 4 küme-alım ADAY sinyali | yok |
| `data/sentiment` | Reddit + VADER ADAY sinyali | yok |
| `data/iv_history` | Rolling 52h ATM IV rank | yok |
| `data/market_structure` | Halt/LULD + SSR short guard | yok |
| `signals/setup_score` | Kurulum kalitesi (B7) + benchmark'a görel güç (B8) | yok |
| `signals/market_regime` | Rejim + risk çarpanı | yalnız Hisse Detay'da tek kelime |
| `signals/indicators_ext` | 24 ek gösterge (HMA/KAMA/TSI/WaveTrend/STC/Force Index…) | **hiçbiri** |

**D. Opsiyon tarafı:** `risk/options_analyzer` (likidite filtresi, ATM IV, IV rank, pin/atama
riski, uzlaşma stili) ve `risk/option_strategy.select_option_contract` motorda vardı —
panoda yalnız **portföy toplam Greek'leri** görünüyordu, **zincir hiç yoktu**.

**E. Araçlar:** `risk/risk_manager.size_stock` (adaptif risk + rejim + makro + PDT +
sektör tavanı) her sinyalde çalışıyordu ama operatör panodan **soramıyordu**.
`risk/prob_sizing` de aynı şekilde erişilemezdi.

**F. Operasyon:** `notifier/daily_summary` (günlük sistem-sağlığı özeti) sadece Telegram/
stdout'a gidiyordu. Servis/dosya sağlığı ("hub ayakta mı, log taze mi, hangi API anahtarı
dolu") panoda hiç yoktu.

---

## 2. Uygulanan revizyon

### 2.1 Yeni modül: `nihai/web/api_extra.py` (21 uç)

Yeni **yetenek** eklemez; mevcut çıktıları HTTP'ye bağlar. Tasarım kuralları:

1. **Anayasa korundu** — `nihai.ai` KARAR-DIŞI/GÖLGE kalır. Hiçbir uç sinyal/eşik/
   `META_GATE`/execution'a dokunmaz. Mutasyon yapan 3 uç (`ai/ask`, `ai/selfcheck/run`,
   `ai/portfolio-propose`) yalnız kendi gölge JSONL'lerine yazar.
2. **Fail-soft** — her alt sistem `_safe()` ile sarmalı. FRED anahtarı yok / torch kurulu
   değil / log birikmemiş → uç **500 atmaz**, `{"ok": false, "error": …}` döner; pano
   yalnız o kartı "veri yok" gösterir. Test: `tests/test_api_extra.py` bunu 12 uç için doğrular.
3. **Paylaşılan sağlayıcı** — `_provider()` panonun mevcut bağlantısını yeniden kullanır
   (ekstra hub/WS bağlantısı açılmaz).

| Grup | Uçlar |
|---|---|
| Quant Beyin | `ai/overview` `ai/brief` `ai/learning` `ai/decisions` `ai/episodes` `ai/knowledge` `ai/registry` `ai/capacity` `ai/selfcheck` `ai/selfcheck/run` `ai/ask` `ai/portfolio-propose` `ai/proposals` |
| İstihbarat | `intel/symbol` `intel/macro` `options/chain` `indicators/extended` |
| Araçlar | `tools/position-size` |
| Sistem | `system/health` `system/daily-summary` `system/evidence-notes` |

### 2.2 Yeni sekmeler (9 → 13)

**① Quant Beyin** — karar kipi / LLM sağlayıcı / gölge-koşucu / gündem kartları · günlük
brif · öğrenme & kalibrasyon raporu · gölge Bayes kararlar · epizodik hafıza · ders
kataloğu · model kartları · bilgi çekirdeği araması · "beyne sor" · öz-değerlendirme
karnesi · gölge portföy önerisi · parametre kapasite merdiveni.

> Model kartları tablosu **ACI DERS #8**'i otomatik uygular: `AUC≈0.50` iken `DSR>0.9` olan
> satır kırmızıya boyanır — "kanıt kapısı raporun geri kalanıyla çelişiyorsa kapı bozuktur".

**② Kanıt & Doğrulama** — forward verdikt / çözülen tahmin / canlı isabet / hedefe ilerleme
kartları · kapsama tablosu · meta-kapı canlı simülasyonu (ACI DERS #3) · tear-sheet
(yeni sekme veya gömülü) · **ACI DERSLER** ve **EDGE_FINDINGS** başlıkları panoda.

**③ Piyasa İstihbaratı** — sembol istihbaratı (olay takvimi + guard'lar, insider,
sentiment, halt/LULD/SSR, IV geçmişi, kurulum skoru + görel güç, veri sağlığı karnesi,
OPEX/çeyrek-sonu) · makro rejim & risk-off kapısı · opsiyon zinciri (likidite sağlığı,
strike bazlı Greekler, pin riski, önerilen kontrat) · 23 satırlık genişletilmiş gösterge
paneli + konsensüs.

**④ Sistem & Araçlar** — hub/tahmin servisi/log tazeliği/tahmin logu kartları · veri
dosyaları yaşı · çalışma kipleri + API anahtarı varlığı · **pozisyon boyutu hesaplayıcı**
· günlük sistem özeti.

### 2.3 Performans/güvenlik notları

- Yeni sekmeler **tembel yüklenir** (`switchTab` kancası): 5 saniyelik ana yenileme
  döngüsüne hiçbir maliyet eklenmez. Beyin brifi ayrıca **10 dk önbellekli** (LLM çağırabilir).
- `POST` uçları mevcut `X-Auth-Token` middleware'inin arkasındadır (`WEB_TOKEN` doluysa).
- Her sekmenin başında **dürüstlük şeridi**: ADAY sinyal ≠ kanıtlı kenar; gölge katman
  emir üretmez; ~%50 isabet = öngörü gücü yok.

### 2.4 Doğrulama

```
pytest            → 1698 passed  (1678 + 20 yeni)
ruff check .      → All checks passed!
mypy nihai        → Success: no issues found in 84 source files
node --check      → pano JS sözdizimi geçerli
ID/rota eşleşmesi → eksik getElementById: 0 · bilinmeyen apiCall rotası: 0
canlı duman testi → 21 yeni uç 200 + gerçek veri (geçici sunucu :8791, kapatıldı)
```

---

## 3. Panoyu yeniden başlatma

HTML her istekte diskten okunur (`no-store`) → **sekmeler anında görünür**.
Yeni uçlar Python tarafındadır: `web_dashboard` süreci yeniden başlatılmadan
`/api/ai/*` vb. **404** döner. Servisler 7/24 çalıştığı için yeniden başlatma
zamanlaması operatörün kararıdır (forward-test kaydı kesintiye uğramasın).

---

## 4. BİLİNÇLİ olarak bağlanmayanlar (kalan boşluk, dürüst liste)

| Ne | Neden bağlanmadı |
|---|---|
| `execution/backtest` (walk-forward, Monte Carlo), `execution/screen`, `auto_screener.scan_and_update` | Dakikalar–saatler süren işler. HTTP isteğinden tetiklemek arka plan görev kuyruğu + ilerleme akışı ister; ayrı bir iş kalemi. |
| `research/*` (purged_cv, deflated/DSR, pbo/CSCV, live_retrain, edge_lab, pairs/pead testleri) | Araştırma katmanı — CLI'de kalması bilinçli. Sonuçları zaten model kartlarına ve EDGE_FINDINGS'e akıyor; pano bunları okuyor. |
| `execution/gate` (backtest→paper→canlı kapısı) | Girdisi backtest raporu; backtest tetikleme olmadan tek başına anlamsız. |
| `execution/cost_attribution` | Aynı — backtest sepeti gerektirir. |
| `ai/lab.run_sandbox` (kod çalıştırma) | Panodan `exec` yüzeyi açmak güvenlik riski. CLI'de kalmalı. |
| `ai/explain` (IG / attention rollout / conformal) | Canlı model nesnesi ister; model kartı + registry üzerinden kısmen görünüyor. |
| `notifier/chart_analyst` (`/yorum` AI grafik okuma) | Vision LLM kotası tüketir; Telegram tetikleyicisi yeterli. Panoya alınacaksa oran sınırı gerekir. |
| `advanced_math` (Heston/Merton/Hawkes/BOCPD/imza/LSMC) | Hesap kütüphanesi — doğal bir pano yüzeyi yok; `quant_math` bilgi çekirdeği üzerinden keşfedilebilir. |
| `data/sharadar`, `data/pit_universe`, `data/alpaca_history` | Araştırma veri adaptörleri (PIT/survivorship); canlı pano bağlamı yok. |

---

## 5. Değişen dosyalar

| Dosya | Değişiklik |
|---|---|
| `nihai/web/api_extra.py` | **YENİ** — 21 uç, ~850 satır |
| `nihai/web/web_dashboard.py` | +8 satır (router include) |
| `nihai/web/web_dashboard.html` | 3527 → 4451 satır (4 sekme + CSS + JS + tembel yükleme) |
| `tests/test_api_extra.py` | **YENİ** — 20 test (fail-soft sözleşmesi + sizing tutarlılığı) |

---

## Bölüm B — Tur-17 §6 açık kalemlerinin kapatılması

## 6. `neural_pwin` kolonları — ÖLÜ KÖPRÜ KAPANDI (Tur-17 §6 madde 1)

`ai/learning.py` `neural_pwin` + `neural_pwin_side` kolonlarını arıyor ve varsa
`sources.neural` Brier'ini otomatik dolduruyordu — ama bu kolonları **yazan taraf yoktu**.
Köprünün üretici ucu kuruldu.

### 6.1 Yeni modül: `nihai/execution/neural_shadow.py`

Kayıtlı `NeuralCapacity` modelini yükler, canlı aday sinyalin karar-barı özellik
vektörüne MC-dropout uygular, `(p_win, std)` döner.

**ANAYASA korundu — bu bir sinyal değil, bir ÖLÇÜMDÜR.** Değer `live_predict_service`
içinde `_neural` altında taşınır ve kayıttan sonra satırdan **düşürülür** (`_feats`
deseninin aynısı) → `/api/predictions` JSON'una ve pano işlem tablosuna girmez. Gerekçe:
kayıtlı modelin ölçülen OOS-AUC'si **0.500** (kart `1f072998…`). İşlem tablosunda ikinci
bir "P(win)" sütunu göstermek onu sinyal sanma riski taşır; amaç kenar değil, **canlı
kalibrasyon ölçümüdür**.

**Model seçimi — üç kilit** (sessiz yanlış-model riski):

1. `NEURAL_SHADOW_MODEL` env verilmişse yalnız o artefakt (operatör kontrolü).
2. Yoksa registry: `kind=="neural"` + artefaktı diskte olan + `feature_names` güncel
   `FEATURE_NAMES` ile **birebir** + `data.source` `synthetic` ile başlamayan kartların
   en yenisi. *(Sentetik müfredat modeli — kontrol koşusuydu — canlı bara sokulmaz.)*
3. Yükleme sonrası model nesnesinin kendi `feature_names`'i tekrar karşılaştırılır
   (yeniden-sıralama = sessizce yanlış olasılık).

**Fail-soft:** torch yok / kart yok / artefakt yok / şema uyuşmuyor / çıkarım hata verdi
→ `None`; kolon boş kalır, tahmin satırı sağlam yazılır. Pasiflik nedeni
`status()["reason"]`'da görünür (sessiz ölüm yasak).

### 6.2 BULGU — doygunluk artefaktı (canlı duman testinde yakalandı)

Gerçek modelle ilk duman testinde çoğu özelliği 0 olan bir vektör `p=0.0, std=0.0` verdi.
Bu "kesin tahmin" değil, **dağılım-dışı girdinin ağı doyurmasıdır**; loglanırsa
`learning.py`'nin Brier'ini sahte-kesin değerlerle kirletirdi. Dropout aktifken gerçek bir
tahminin std'si asla tam 0 olmaz → guard: `std==0` **ve** p uçta (≤1e-3 / ≥1−1e-3) →
kaydetme. (Uç-ama-varyanslı ve varyansız-ama-orta değerler KAYDEDİLİR; dejenere model de
ölçülmelidir.)

Aynı testin ikinci ayağı — **gerçek eğitim satırlarında** (PIT S&P500, 12.268 satır) model
p_win aralığı **0.24–0.44, ortalama 0.356** üretti; veri setinin taban oranı **0.357**.
Yani model AUC-0.500 karnesiyle tutarlı davranıyor: taban orana yakın, ayırt etmeden.
Beklenti buna göre kurulmalı — **canlı `sources.neural` Brier'i iyi ÇIKMAYACAK** (model
~0.36 derken canlı taraf-kazanma tabanı ~0.47–0.50). Bu bir kusur değil, ölçümün kendisidir.

Gecikme: **7,8 ms/aday** (MC=20, CPU), bar başına bir kez, ≤30 sembol → 5 sn'lik yenileme
döngüsüne anlamlı yük yok.

### 6.3 Yan düzeltme — enforce kipinde bozuk `is_long`

`extract_features` `is_long`i **sinyalin aksiyonundan** türetir. META_GATE **enforce**
kipinde kapı, aday LONG'u `WAIT`'e çevirip elenen yönü `metadata['meta_gated']`'e koyar →
`extract_features` `is_long=0` (yani SHORT) yazardı. Bu hem `build_live_dataset` eğitim
satırını hem gölge çıkarımı sessizce bozardı. Aday tarafı artık tek kaynaktan gelir
(`_candidate_side`) ve `is_long` ona göre düzeltilir. Üretim `shadow` kipinde olduğu için
**gizli/latent** bir hataydı; enforce açılsaydı ilk günden veri bozardı.

### 6.4 Şema

`FIELDS` += `neural_pwin`, `neural_pwin_side`, `neural_pwin_std`. Mevcut `_ensure_schema`
eski başlıklı `predictions_log.csv`'yi atomik yükseltir (eski 52.258 satır `''` alır, veri
kaymaz — test edildi). Tüm okuyucular isim-tabanlı (DictReader/başlık-indeksi) → pozisyonel
kırılma yok.

> **Kolonlar ne zaman dolar:** şema süreç belleğindedir → **kayıtçı servis yeniden
> başlayana kadar** yazılmaz. Yeniden başlatma zamanı operatörün kararıdır (forward-test
> kaydı kesilmesin). Aynı kısıt Bölüm A'daki 21 yeni uç için de geçerlidir.

### 6.5 Köprünün görünürlüğü — kendi dersimizi tekrarlamamak

`neural_shadow.status()` ilk yazımda hiçbir yerden çağrılmıyordu; yani **az kapattığım ölü
köprünün aynısını** (üretici var, tüketici yok) ters yönde kurmuş oluyordum. Bağlandı:
`/api/ai/overview` → `neural_shadow` alanı → Quant Beyin sekmesinde **"Nöral Gölge p_win"**
kartı. Pasifse kartta **nedeni** yazar (`torch kurulu değil` / `uygun nöral model kartı yok`
/ `NEURAL_SHADOW=off` …). Pasiflik yalnız "boş kolon" olarak görünseydi köprünün ölümü
haftalarca fark edilmezdi. Sözleşme testle kilitli (`test_neural_shadow_durumu_gorunur`).

## 7. Watchdog AtLogOn tetiği (Tur-14 açık madde 1) — kök neden düzeltildi

Kayıt scriptlerindeki yorum *"Daily 00:05 + StartWhenAvailable = AtLogOn ile aynı pratik
kapsama"* diyordu. **Yanlış:** `StartWhenAvailable` yalnız **kaçırılan** tetiği telafi eder
(makine o saatte kapalıysa). Gün içi reboot'ta 00:05 tetiği o gün **zaten koşmuş** sayılır →
telafi devreye girmez. Ölçülen sonuç 2026-07-18'de yaşandı: 03:00 reboot'undan sonra iki
servis de 13:46'ya kadar (10+ saat) ölü kaldı, forward kaydı o sürede hiç birikmedi.

- `register-watchdog-system.ps1` + `register-predict-watchdog.ps1`: tetik artık
  **Daily 00:05 + AtLogOn** (kaynak-doğruluk düzeltildi, yanlış yorum kaldırıldı).
- Canlı görevlere uygulama **yükselti istiyor** (`Set-ScheduledTask` → "Erişim engellendi";
  Tur-13'te ölçülen "güncelleme yükselti istemez" davranışı bu alan için geçerli değil) →
  tek komutluk **`tur18-servis-tetik.ps1`** yazıldı (idempotent, yönetici kontrolü + doğrulama).
- **Yükseltisiz eşdeğer KURULDU** (`logon-watchdog-tetik.ps1 -Install`): Başlangıç
  klasörüne kısayol → oturum açılışında **mevcut** görevleri `Start-ScheduledTask` ile
  tetikler (kullanıcının kendi görevi için yükselti istemez). Yeni watchdog döngüsü
  yaratmaz; görev çalışıyorsa `MultipleInstances=IgnoreNew` ikinci örneği engeller →
  projenin daha önce bilinçli olarak kaçındığı **çift-watchdog riski yok**. Yönetici
  penceresi bulununca `tur18-servis-tetik.ps1` gerçek tetiği ekler, kısayol gereksizleşir.

## 7.1 CANLI DEVREYE ALMA — yapıldı ve doğrulandı (2026-07-25 18:20–18:27)

Ölçüm sırasında ortaya çıkan asıl sorun planlanandan büyüktü: iki servis de
**2026-07-19 00:05'ten beri** çalışıyordu (6 gün) — yani **Tur-15/16/17 kodu hiç canlıya
geçmemişti** — ve her iki watchdog görevi `Ready` durumundaydı, yani servisler
**denetimsizdi** (ölseler kimse kaldırmayacaktı).

Sırayla yapıldı: (1) `predictions_log.csv` yedeklendi (`data/_backup_predictions_log_*`),
(2) watchdog görevleri tetiklendi → ikisi de `Running`, resolve turu döndü
(50.011 çözülmüş, isabet %46,8), (3) kayıtçı süreci durduruldu → watchdog **30 sn içinde
güncel kodla** ayağa kaldırdı (PID 13228, 18:22:22).

Canlı doğrulama:

| Kontrol | Sonuç |
|---|---|
| `/api/ai/overview` | **200** · `neural_shadow: active=true, tier=deep_mlp, MC=20, kart 1f072998…` |
| `/api/forward-test` · `/api/system/health` · `/api/ai/learning` · `/api/ai/registry` · `/api/indicators/extended` | hepsi **200** (yeniden başlatma öncesi 404'tü) |
| `predictions_log.csv` şema göçü | **canlıda gerçekleşti** 18:26 (watchdog'un resolve turu tetikledi) |
| Göç bütünlüğü | 52.257 satır → **52.257** · alan sayısı uyuşmayan satır **0** · eski kolonlar **birebir aynı** · yeni kolonlar eski satırlarda boş · 50.015 çözülmüş tahmin sağlam |

**Kolonlar ne zaman DEĞER alacak:** bugün Cumartesi, piyasa kapalı (`market_open=false`) →
aday sinyal kaydı yok. İlk `neural_pwin` değerleri **pazartesi açılışında** birikmeye başlar.

Bu turda ayrıca doğrulandı: `NihaiAlgo-ShadowRunner-Hourly` **kayıtlı ve koşuyor**
(Tur-17 §6 madde 2 kapalı) · Tur-17 §6 madde 3 (panoda AI/Beyin bölümü) Bölüm A ile kapandı.

## 8. Doğrulama (Bölüm B)

```
pytest            → tam paket YEŞİL (+22 test: tests/test_neural_shadow.py)
ruff check .      → All checks passed!
mypy nihai        → Success: no issues found in 85 source files
canlı duman testi → gerçek model yüklendi (kart 1f072998), 12.268 satırda p_win
                    aralığı 0.24–0.44 (taban 0.357 ile tutarlı), 7,8 ms/aday
```

### Kalan (öncelik sırasıyla)

1. **Yalnız yükselti bekleyen tek iş:** `tur18-servis-tetik.ps1` (yönetici PowerShell) —
   görevlere GERÇEK AtLogOn tetiğini ekler. Şu an yerine `logon-watchdog-tetik.ps1`
   kısayolu çalışıyor (§7), yani **acil değil**; kalıcı/temiz çözüm için yapılmalı.
2. edge_lab BACKLOG #2 (VRP) veya #3 (lead-lag) — kanıt-kapılı yeni aday aile.
3. Bölüm A §4'teki bilinçli boşluklar (backtest/screen tetikleme = arka plan görev kuyruğu).
4. `mypy research` 12 hata / 4 dosya — `mypy nihai` kapısının dışında, Tur-14'ten beri açık.

### İzlenecek (pazartesi)

- `neural_pwin` kolonları gerçekten doluyor mu (`/api/ai/overview` → `neural_shadow.active`
  ve logda dolu hücre). Dolmuyorsa doygunluk guard'ı hepsini eliyor olabilir → `reason`/
  debug logu bak.
- `sources.neural` Brier'i (`/api/ai/learning`) — **iyi çıkmayacak** (§6.2), beklenti bu.
