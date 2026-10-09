# GELİŞTİRME RAPORU — 2026-07-10 (Fable 5, max-effort taraması)

Kapsam: `NihaiAlgoritma toplu` (nihai + MainHub). Kullanıcının 4 maddesi tek raporda;
maddeler **aşamalı uygulanacak** şekilde checkbox'lı yazıldı. 5. madde (optimizasyon)
bu oturumda UYGULANDI → `OPTIMIZASYON_LOG.md → Ek Tur 10` + bu raporun Ek'i.

**Anlık durum (2026-07-10 akşamı):** ~528 test yeşil (exit 0) · mypy 56 dosya/0 hata ·
ruff temiz · ledger'daki 30+ bulgunun tamamı kapalı (yalnız "dürüst iyileştirme" backlog'u açık) ·
canlı servisler 7 Temmuz'dan beri KAPALI · forward-kanıt: 20.010 çözülmüş tahmin → **NO_EDGE**
(isabet %47.4, tüm AUC ≤ 0.5, kapı canlıda zarar veriyor) · git son commit 26 Haziran, **101 dosya commitsiz**.

> **TUR-11 GÜNCELLEMESİ (2026-07-13, Fable 5):** FAZ AI-3 "dürüst iyileştirmeler" backlog'u
> (aşağıdaki 3 madde) **TAMAMEN KAPANDI**; FAZ AI-4'ün ilk ve üçüncü maddesi (`daily_summary` +
> VaR kartı) **KAPANDI**; FAZ AI-1 denendi ama Gemini ücretsiz-katman kotasına takılıp
> **TAMAMLANAMADI** (kod değil, kota — kullanıcı kararı gerekiyor, FAZ AI-2'ye benzer). Ayrıca
> 2 paralel salt-okunur denetim (15 bulgu) + bu bulguların 15'i de kapatıldı; entegrasyon
> sırasında GERÇEK bir üretim çökmesi bulundu ve düzeltildi (2026-07-11 OneDrive kilit → forward-
> harness'in tüm günü kaybetmesi). nihai **582 test exit=0** · mypy **57/0** · ruff temiz ·
> MainHub **27 yeşil**. Detay: `OPTIMIZASYON_LOG.md → Ek Tur 11`, checkbox'lar aşağıda [x]
> işaretlendi; `DENETIM_BULGULARI_2026-07-09.md`'nin Tur-11 notu.

> **TUR-13 GÜNCELLEMESİ (2026-07-17, Fable 5 + Opus agent'ları):** FAZ AI-4 operatör katmanı
> KODLANDI: `nihai/ai/` (router = §1.2 model-yönlendirme tablosunun kodu + kota-cooldown'lu
> anthropic→gemini fallback; memory = kalıcı JSONL AI hafızası; reasoner = karar-DIŞI
> Durum/Değişim/Dikkat akıl-yürütmesi + LLM dürüstlük guardı) + `daily_summary`'ye bağlandı
> (AI-0 kabul satırı dahil). Dürüst-ölçüm: `research/deflated.py` (PSR/DSR, Bailey & LdP;
> §4.1'in "eksik: deflated Sharpe" kalemi) + forward raporuna İsabet %95 CI + binom-p satırı
> (canlı: %47.1, CI [%46.6-%47.6], p≈2e-26 → %50'den sapma anlamlı ama ALTINDA — kenar değil).
> OPS: META_GATE=true→**shadow** (.env; GATE_HURTS çelişkisi giderildi) · harness 07-14/15/16
> boşluğunun kök nedeni (uykuda kaçan 23:30 tetiği + 30dk görev-limiti kill'i) → limit 4sa +
> 12:00 yedek tetik + yarım-koşu tespiti · watchdog kayıtları: yeni görev oluşturma bu makinede
> yükseltisiz REDDEDİLİYOR (0x80070005) → `tur13-servis-devri.ps1` (**18:24'te UAC onayıyla
> KOŞULDU — devir TAMAM**: görevler interactive Running, hub+kayıtçı Kagan'da güncel kodla)
> · hub/kayıtçı/predict-watchdog'un 07-10'dan kalma SYSTEM ZOMBİLERİ olarak çalıştığı ve
> yükseltisiz oturuma görünmediği anlaşıldı (Tur-12 "kapalı/kayıtsız" teşhisi muhtemelen bu
> körlük). AI-0 kabulü YİNE verilemedi (3 günlük boşluk); izleme 07-17'den yeniden. +22 test
> (toplam suite exit=0) · mypy 62/0 · ruff temiz. Detay: `OPTIMIZASYON_LOG.md → Ek Tur 13`.

---

## BÖLÜM 1 — "Projedeki AI'ı geliştirme" yol haritası (Fable 5 + Opus 4.8 görev dağılımıyla)

### 1.0 Projedeki "AI" nedir? (üç katman)
1. **Meta-ML hattı**: meta_features (25 özellik) → train_meta/purged_cv → meta_gate (üretimde
   varsayılan KAPALI) + canlı eğitim hattı (live_features_log → build_live_dataset → live_retrain).
2. **LLM bileşenleri**: `llm_event.py` (Claude ile 8-K olay çıkarımı — ADAY sinyal, henüz hiç
   ölçülmedi), `chart_analyst` (/yorum, Gemini vision — karar-DIŞI yorum aracı), SHAP açıklama kartı.
3. **Geliştirme süreci AI'ı**: Claude Code denetim/fix turları (Tur 1-10), CLAUDE.md bağlam katmanı.

### 1.1 İlkeler (ACI DERSLER'e bağlı — bunlar pazarlık dışı)
- **Yeni sınıflandırıcı eklemek darboğazı çözmez** (ACI DERS #5). Darboğaz = OOS-kenarlı baz sinyal
  + PIT veri yokluğu. AI planının ağırlığı model değil; **veri, ölçüm ve kanıt döngüsü**.
- Her AI iddiası: purged CV + iki bölme + PIT + (mümkünse) forward'dan geçmeden "kenar" denmez.
- META_GATE, canlı gate_sim GATE_HURTS derken **kapalı kalır**.

### 1.2 Model seçim rehberi (hangi işe hangi Claude?)
| Görev tipi | Model | Neden |
|---|---|---|
| Kenar hükümleri, istatistik metodoloji tasarımı/denetimi, mimari kararlar, tz/concurrency gibi sinsi buglar, tur planlama | **Fable 5** (gerekirse max effort) | En derin muhakeme; pahalı → hüküm anlarına sakla |
| Toplu implementasyon, test yazımı, refactor, 3-4 paralel denetim/fix agent turları, veri boru hattı koşuları | **Opus 4.8** | Tur-9'da kanıtlanmış desen (partisyonlu domain + tek-thread entegrasyon) |
| Çalışma-zamanı içi çağrılar: 8-K çıkarımı toplu backfill, günlük özet üretimi | **Haiku 4.5** (`LLM_EVENT_MODEL=claude-haiku-4-5`) | Binlerce çağrı = maliyet; çıkarım (parse) işi Haiku'ya yeter |

### 1.3 Fazlar (sıralı; her fazın kabul kriteri var)

**FAZ AI-0 — Kanıt döngüsünü 7/24 yap (önkoşul; O4.8 + kullanıcı, ~yarım gün)**
Tüm canlı-eğitim AI'ı veri birikimine bağlı; servisler kapalıyken plan kağıtta kalır.
- [x] Hub + tarayıcıyı kalıcı başlat — 2026-07-10 21:15: watchdog yolu seçildi; iki SYSTEM görevi
      kayıtlı+Running (hub 8799 + kayıtçı 8765 UP). Not: register-watchdog-system.ps1 BOM'suz UTF-8 +
      em-dash yüzünden PS5.1'de parse patlıyordu → 5 ps1'e BOM eklendi.
- [x] Bekleyen tahminleri çöz — 2026-07-10 akşamı +245 çözüldü (20.337 toplam; bekleyen 424→405,
      kalan kısmı YENİ canlı tahminler + 3 günlük kapanmada penceresi kaçmış 1m'ler). Harness elle koşuldu.
- [ ] Kabul: 3 gün üst üste `forward_eval/history.csv`'ye satır ekleniyor + `live_features_log.csv`
      büyüyor. (İzleme başladı: 2026-07-10 akşamı; her iki dosya da o akşam yeniden büyümeye başladı.)
      **2026-07-13 GÜNCELLEME:** 3 günlük pencerede (07-10/11/12/13) bir GERÇEK kesinti bulundu —
      2026-07-11 02:04 forward-harness OneDrive dosya-kilidiyle çöktü (`os.replace` `PermissionError`),
      o gün hiçbir tahmin çözülmedi/history satırı eklenmedi → kabul kriteri HÂLÂ karşılanmadı.
      Kök neden düzeltildi (`_atomic_replace` yeniden-deneme, `OPTIMIZASYON_LOG → Ek Tur 11`); izleme
      sıfırdan devam ediyor (bu turdan sonraki 3 gün üst üste kesintisiz kabul verir).

**FAZ AI-1 — Ölçülmemiş AI adaylarını ölç (O4.8 koşar, F5 hüküm verir; 1-2 oturum)**
- [ ] `python -m research.llm_event_test` — projenin ilk gerçek LLM-sinyal ölçümü (8-K çıkarımı;
      EDGAR tarihsel+PIT olduğundan soğuk başlangıçta bile geriye dönük test edilebilir; Haiku ile
      maliyet düşür). Kabul: EDGE_FINDINGS'e yeni §, PIT + removed-since kapısıyla.
      **2026-07-13 DENENDİ, TAMAMLANAMADI:** `--pit --as-of 2022-01-01 --limit 30` koşuldu;
      `DATA_PROVIDER=alpaca_hist` ile fiyat tarafı (SIP, delist'ler dahil) sorunsuz çalıştı ama Gemini
      ÜCRETSİZ katman (günde ~20 istek) PIT taramasını bitirmeden 429 RESOURCE_EXHAUSTED'a düştü
      (coverage 1/30 sembol — hüküm verilemeyecek kadar küçük örneklem). Kod hazır ve doğru; gerçek
      ölçüm ANTHROPIC_API_KEY (Haiku, ucuz) veya Gemini ücretli katman gerektiriyor — kullanıcı kararı.
- [ ] `python -m research.insider_test --pit` (C1 kuruldu, gerçek ölçüm hiç koşulmadı).
- [ ] Sentiment birikimini başlat (AI-0 döngüsüne periyodik `update_sentiment(sym)` hookup'ı —
      tarihsel yok, ancak İLERİYE birikir; anlamlı ölçüm haftalar sonra).
- [ ] F5: üç sonucu tek oturumda yorumla → hangi aday (varsa) FAZ AI-3'ü hak ediyor?

**FAZ AI-2 — Veri kararı (KULLANICI kararı + F5 analizi)**
Ücretsiz yol tüketildi (§20-21: SIP-kalite PIT'te bile kenar yok). İki seçenek:
- [ ] **A: Sharadar (ücretli, ~$50+/ay)** → PIT fundamentals + delist-temiz evren; PEAD/SUE ve
      meta-labeling'i GERÇEK earnings sürprizleriyle yeniden ölç. F5'e "hangi abonelik hangi
      hipotezi test etmeye yeter" analizi yaptır, sonra karar ver.
- [ ] **B: Ücretsiz kal** → kenar arayışı canlı-birikimli veriye (AI-0/1) ve olay-bazlı ücretsiz
      kaynaklara (EDGAR) daralır; beklentiyi buna göre sabitle (aylar ölçeğinde sabır).

**FAZ AI-3 — Model disiplinini güçlendir (F5 tasarım + O4.8 kod; ledger "dürüst iyileştirmeler")**
Kenar UYDURMADAN modeli daha dürüst ölçen kalemler (kenar çıkarsa güvenilir olsun diye):
- [x] Eşik seçimini TEST'ten TRAIN/val'e taşı + denenen konfig sayısını raporla (çoklu-test deflasyonu).
      **2026-07-13:** `train_meta.evaluate()` artık eşiği TRAIN'de seçip yalnız O eşiği TEST'te
      doğruluyor; `thr_source`/`n_thr_tried` raporlanıyor. `OPTIMIZASYON_LOG → Ek Tur 11`.
- [x] purged_cv'ye label-shuffle null-AUC tabanı + leakage bayrağı; `purge=horizon` değerini
      çağıranlardan otomatik geçir (bars_held'den). **2026-07-13:** `--null-reps` + "SIZINTI ŞÜPHESİ"
      bayrağı eklendi; `suggest_purge` (Tur-10 P2-1'de tüm çağıranlara zaten bağlıydı) yerel-yoğunluk
      duyarlı hale getirildi (kümeli veride az-purge riski kapandı) + `live_retrain._purged_gate` artık
      SABİT 0.45 değil terfi edilecek eşikte CV'yi sınıyor.
- [x] `save_model` provenance: code_sha, dataset_sha256, n_rows, seed, embargo, min_gate_keep, ECE.
      **2026-07-13:** tüm alanlar + `threshold_selection`/`n_configs_tried`/`calibrate`/`saved_at`
      eklendi (embargo yerine daha bilgilendirici `min_gate_keep`+eşik-kaynağı tercih edildi).
- [ ] (Veri gelirse) kalibrasyonlu P(win) → `prob_sizing` bağlantısını conformal aralıklarla dene.
      (Veri koşulu hâlâ karşılanmıyor — ertelendi.)

**FAZ AI-4 — LLM operatör katmanı (O4.8; düşük risk, hemen değer)**
- [x] Günlük Telegram "sistem sağlığı + forward verdikt" özeti: `forward_eval/history.csv` +
      coverage + servis durumundan Haiku ile 5 satırlık Türkçe özet (karar vermez, ÖZETLER).
      **2026-07-13:** `nihai/notifier/daily_summary.py` — `forward-harness.ps1`'e günlük çağrı olarak
      bağlandı (harness başarısız olsa bile çalışır). `--llm` opsiyonel (varsayılan deterministik metin).
- [ ] chart_analyst'i karar-DIŞI tut (mevcut dürüst konum); sinyal üretimine bağlama.
      (Değişmedi — hâlâ doğru; bu turda dokunulmadı.)
- [x] (Ops.) `/api/var` + dashboard VaR kartı hookup'ı (A3'ün kalan tek ucu).
      **2026-07-13:** `web_dashboard.html`'e "Portföy VaR" kartı eklendi (`/api/var` zaten vardı).

**FAZ AI-5 — Süreç AI'ı: tur döngüsünü kurumsallaştır (kullanıcının ~15 tur planı)**
- [ ] Denetim turları: 4 paralel O4.8 salt-okunur denetçi → ledger → partisyonlu O4.8 fix →
      tek-thread entegrasyon (Tur-9 deseni; çakışma dersi: düzenlemeden önce süreç+mtime kontrolü).
- [ ] Tur SONUNDA git commit (bkz. §3 P0) — 101 commitsiz dosya bir daha birikmesin.
- [ ] F5'i turda bir kez kullan: bulguların önceliklendirmesi + "kenar/metodoloji" hükümleri.

---

## BÖLÜM 2 — Eksik taraması ve çözümler

### 2.1 Operasyonel eksikler (EN ACİL — kod değil, işletim)
| # | Eksik | Kanıt | Çözüm |
|---|---|---|---|
| O1 | Canlı servisler kapalı (7 Tem 20:06'dan beri) | `system.log`/`predictions_log.csv` mtime | AI-0: docker compose VEYA watchdog kayıtları |
| O2 | Watchdog görevleri kayıtsız (yalnız ForwardHarness-Daily kayıtlı) | `Get-ScheduledTask` | `register-watchdog-system.ps1` + `register-predict-watchdog.ps1` |
| O3 | 305 bekleyen tahmin çözülemiyor; resolver hub'a bağımlı, yfinance 1m fallback boş | bugünkü forward-harness.log (resolve 0/305) | Hub'ı aç → harness'ı elle koştur; 1m tahminlerde çözüm penceresi kaçmadan resolve şart |
| O4 | Git: son commit 26 Haziran; **101 dosya commitsiz** (Tur 4-10 tek kopyada, yalnız OneDrive) | `git status` | Hemen commit (bkz. §3 P0-1); sonrası: her tur sonunda commit |
| O5 | Canlı CSV'ler OneDrive-senkron klasörde (kilit/çakışma + senkron çakışması riski) | proje yolu | Kabul edilmiş risk olarak belgele; alternatif: data/'yı OneDrive-dışına taşı + junction |
| O6 | forward-harness.log'da Türkçe karakterler bozuk (konsol code-page) | log içeriği | `forward-harness.ps1`'e `[Console]::OutputEncoding=[Text.Encoding]::UTF8` (kozmetik) |
| O7 | Birden çok eski Claude oturumu açık duruyor (paralel-düzenleme riski, bilinen ders) | süreç listesi | Kullanmadıklarını kapat; turlarda tek yazıcı ilkesi |

### 2.2 Veri eksikleri (kenar arayışının gerçek kilidi)
- **PIT fundamentals yok** → PEAD/SUE/insider hipotezleri gerçek anlamda test edilemiyor
  (yfinance earnings ~%60 kapsama, survivorship'li). Çözüm: FAZ AI-2 kararı (Sharadar) veya bilinçli ücretsiz-kal.
- **Reddit tarihsel imkânsız** (Pushshift kapalı) → sentiment yalnız ileriye birikir; AI-0 şart.
- **Gerçek opsiyon zinciri tarihi yok** → opsiyon backtest'i BS-sentetik (smile/skew yok). Çözüm: Polygon
  (ücretli) veya mevcut yaklaşıklamayı sınırıyla belgeli tut (belgelenmiş durumda).
- **iv_history 52 hafta dolana kadar kısmi** (kendi kendine birikiyor; servisler kapalıyken BİRİKMİYOR → O1).
- **Extra-US dataset survivorship-şişkin** (bilinçli; PIT-üyelik temelli değil) — yeni ölçümlerde uyarıyla kullan.

### 2.3 Kod/altyapı eksikleri (küçük borçlar — büyükler Tur 9/10'da kapandı)
- 3 hazır-kod hookup anahtar bekliyor: **FRED** (`refresh_macro()` main-loop çağrısı), **Reddit**
  (periyodik `update_sentiment`), **/api/var UI kartı**. Hepsi tek-satır bağlama + ücretsiz hesap.
- `purged_cv` çağıranları `purge=horizon` geçmiyor (sol-purge embargo-simetrik varsayılanla çalışıyor;
  etiket horizonu uzunsa purge yetersiz kalabilir) → FAZ AI-3 maddesi.
- `llm_event` ölçümü hiç koşulmadı (ANTHROPIC_API_KEY gerek) → FAZ AI-1.
- MainHub `hub_server` `/options` `all_expiries` düzeltmesi (A2) Tur-9'da girdi — hub'lı opsiyon akışı
  bir kez canlıda smoke-test edilmeli (O1 sonrası 5 dk).
- Test kapsama ölçümü yok → `pytest-cov` ekle, %'yi CI/log'a yaz (hedef koymadan, görünürlük için).

### 2.4 Sürüm/süreç eksikleri
- CI yok (lokal disiplin güçlü ama otomasyon yok). Öneri (ops.): GitHub private repo + Actions'ta
  `pytest -q` + `ruff` + `mypy nihai` (push başına ~3 dk). Ön koşul: O4 (commit disiplini).
- `DENETIM_BULGULARI` madde-başı durum etiketleri Tur-9 uygulamasıyla eskimişti → bu turda başlık
  notuyla düzeltildi (gövde = tarihsel anlık görüntü; geçerlisi başlık notu + Ek Tur 9/10).

### 2.5 Spec (53 madde) kapsama durumu — eksik DEĞİL, bilinçli sınır
README "Madde Eşlemesi" 53 maddenin sinyal-only kapsamındaki tümünü eşliyor. Kapsam dışı bırakılanlar
(31/33/35/46/49 emir katmanı + 40 broker_client) **bilinçli ve belgeli** — sinyal-only sistemde
bulunmamaları doğru. Ek eksik tespit edilmedi.

---

## BÖLÜM 3 — Hata/eksik düzeltme maddeleri (öncelik sıralı iş listesi)

**P0 — Bu hafta (işletim + emniyet):**
- [x] P0-1: commit atıldı — f3ff692 (2026-07-10 20:31, önerilen mesajla; 101 dosya kurtarıldı).
- [x] P0-2: Hub (8799) + kayıtçı (8765) SYSTEM watchdog'larıyla ayakta; akşam +245 tahmin çözüldü (O3).
- [x] P0-3: NihaiAlgo-HubWatchdog-SYSTEM + NihaiAlgo-PredictWatchdog-SYSTEM kayıtlı (AtStartup, Running).

**P1 — Anahtarlar/hesaplar (10'ar dakika):**
- [ ] P1-1: FRED key → `.env` + main-loop `refresh_macro()` hookup'ı → `MACRO_GATE=true` (istenirse).
- [ ] P1-2: Reddit script app → `.env` + AI-0 döngüsüne `update_sentiment` hookup'ı.
- [x] P1-3: ANTHROPIC_API_KEY zaten .env'deydi; `LLM_EVENT_MODEL=claude-haiku-4-5` eklendi (2026-07-10).
      Not: LLM_EVENT_PROVIDER=gemini kaldı; model satırı yalnız anthropic sağlayıcıda okunur (çakışma yok).
      **2026-07-13 DÜZELTME:** satır .env'de var ama DEĞERİ BOŞ (`ANTHROPIC_API_KEY=`) — FAZ AI-1'in
      llm_event_test denemesi bu yüzden yalnız Gemini'ye (ücretsiz kota) düştü ve 429'a takıldı.
      Anahtar dolu değilse P1-3 fiilen açık sayılmalı; gerçek ölçüm için kullanıcı ya anahtarı doldurmalı
      ya da LLM_EVENT_PROVIDER=anthropic'e geçmeli.

**P2 — Küçük kod borçları (birer O4.8 oturumu):**
- [x] P2-1: `suggest_purge(df)` eklendi (purged_cv.py): q95(bars_held) × satır/bar yoğunluğu, tavan n/3;
      üç çağıran da (purged_cv CLI `--purge` ile ezilebilir, live_retrain._purged_gate, train_pead_ml)
      otomatik geçiyor + 4 kalıcı test (test_purged_cv_leakage.py). 2026-07-10.
- [x] P2-2: forward-harness.ps1: konsol UTF-8 + PYTHONIOENCODING=utf-8 + Add-Content -Encoding utf8
      (loguru stderr'i de düzeldi). 2026-07-10.
- [x] P2-3: pytest-cov kuruldu; pyproject `[tool.coverage.*]` + CLAUDE.md komutu. İlk ölçüm:
      **%40.1** (nihai+research, 532 test). Hedef değil, görünürlük. 2026-07-10.
- [x] P2-4: hub'lı opsiyon zinciri canlıda doğrulandı: /options AAPL → all_expiries 24 vade +
      canlı calls; expiry parametresi de doğru filtreliyor. 2026-07-10.

**P3 — Dürüst-iyileştirme backlog'u (FAZ AI-3 ile aynı; F5 tasarım + O4.8 kod):**
- [x] Eşik seçimi TRAIN/val'e + çoklu-test raporu · [x] null-AUC tabanı + leakage bayrağı ·
  [x] save_model provenance · [ ] (veri gelirse) conformal boyutlama.
  **TAMAMEN KAPANDI 2026-07-13** (ilk üç kalem) — bkz. FAZ AI-3 yukarıda + `OPTIMIZASYON_LOG → Ek Tur 11`.

**Kapalı olduğu bu turda doğrulananlar (yeniden yapma):** ledger A1-A5, B1-B4, B6, B8-B9, C1-C6,
D1-D10, B7; A1 tz regresyon testi mevcut (`test_stream_forming_bar.py`); live_retrain min-gate-keep
tutarlılığı (B1) girmiş durumda.

**Uygulama günlüğü — 2026-07-10 gecesi (F5 oturumu):** P0-1..3 + P1-3 + P2-1..4 kapatıldı (yukarıdaki
[x] notları). Doğrulama: 532 test yeşil (exit 0, +4 yeni suggest_purge testi) · mypy nihai 56/0 ·
ruff temiz · kapsama %40.1 (ilk ölçüm). YENİ BULGU (O6 ailesi): BOM'suz ps1 + tipografik karakter
(em-dash→0x94) PS 5.1'de string'i erken kapatıyor — register-watchdog-system.ps1 SYSTEM kaydı bu
yüzden hiç çalışamamıştı; 5 ps1'e BOM eklendi. İZLEMEDE: AI-0 kabulü (3 gün üst üste birikim,
başlangıç bu akşam) · mypy'ın research/'ü kapsamaması (live_retrain'de 2 gizli var-annotated hata
var; ayrı küçük kalem) · eski 1m bekleyenlerin bir kısmı 7-10 Tem kapalılığında çözüm penceresini
kaçırdı (kalıcı kayıp, yeniden oluşmaz — watchdog'lar açık kaldıkça).

**Uygulama günlüğü — 2026-07-13 (Fable 5, Tur-11):** P3'ün ilk üç kalemi + FAZ AI-4'ün iki kalemi
kapatıldı; 2 paralel salt-okunur denetim (research/ML+veri katmanı 7 bulgu, ops/web/MainHub 8 bulgu)
+ 15 bulgunun tamamı kapatıldı (detay `OPTIMIZASYON_LOG → Ek Tur 11`). **YENİ BULGU (canlı olay):**
2026-07-11 02:04 forward-harness OneDrive dosya-kilidiyle çöktü (`os.replace` `PermissionError`,
O5'in belgelenmiş riski gerçekleşti) → `_atomic_replace` yeniden-deneme + `daily_summary`'nin harness
başarısız olsa bile çalışması eklendi; AI-0 "3 gün üst üste" kabulü bu yüzden sıfırdan izleniyor.
P1-3 düzeltmesi: ANTHROPIC_API_KEY satırı .env'de ama DEĞERİ BOŞ (yukarıda not edildi) — FAZ AI-1
bu yüzden Gemini'nin ücretsiz kotasına (~20/gün) takılıp tamamlanamadı. Doğrulama: nihai **582 test
exit=0** · mypy **57/0** · ruff temiz (nihai+research+tests) · MainHub **27 yeşil** + ruff temiz.
İZLEMEDE: AI-0 kabulü (yeniden sıfırdan) · P1-1/P1-2 (FRED/Reddit anahtarı, kullanıcı kararı) ·
FAZ AI-2 (Sharadar/ücretsiz-kal, kullanıcı kararı) · train_pead_ml.prepare() transdüktif sızıntısı
ve prediction_tracker 400-bar lookback/UNRESOLVABLE sınırı bilinçli ertelendi (gerekçe kodda +
ledger'da) · git commit henüz atılmadı (bu turun sonunda P0 disiplini olarak atılacak).

---

## BÖLÜM 4 — Quant trading değerlendirmesi: proje ne seviyede, ne eksik?

### 4.1 Boyut boyut karne (0=yok · 5=kurumsal)
| Boyut | Puan | Gerekçe |
|---|---|---|
| **Alpha/kenar** | **0/5 (dürüstçe ölçülmüş)** | Kanıtlanmış kenar YOK: PEAD, meta-labeling, momentum, kesit-ML, 97k-234k işlemlik PIT ölçümleri ve 20k canlı tahmin hepsi ~yazı-tura. Bu bir başarısızlık değil DÜRÜST ölçüm — retail projelerin %99'u bu testleri hiç yapmaz ve sahte kenarla yaşar. |
| **Backtest/araştırma metodolojisi** | **4/5** | Purged+embargolu K-fold, sol-purge, iki-bölme kapı simülasyonu, PIT/survivorship kapıları, walk-forward, Monte Carlo, maliyet atfı, Wilson-CI, kill-switch farkındalığı, bit-exact parite testleri. Literatür (López de Prado) standardı uygulanmış. Eksik: deflated Sharpe, factor-nötr değerlendirme. |
| **Veri altyapısı** | **2.5/5** | Failover + cache disiplini + delist-dahil Alpaca SIP + PIT-evren iyi; ama PIT fundamentals, gerçek opsiyon zinciri, kurumsal kalite feed yok. |
| **Risk yönetimi** | **3/5** | VaR/CVaR, Greek tavanları, Reg-T/PDT/SSR/LULD/wash-sale kural motorları, kill-switch, korelasyon tavanları — retail üstü. Ama hiçbiri gerçek emirle sınanmadı; portföy-optimizasyon katmanı yok (tek-sinyal tavanları var, ağırlıklayıcı yok). |
| **Execution** | **—** | Sinyal-only, bilinçli kapsam dışı (Madde 31/33/35/46/49). Değerlendirilmez. |
| **Kod kalitesi/mühendislik** | **4/5** | ~528 test, mypy 0 hata, ruff temiz, leak-free feature disiplini, train/serve hizası, parite-test kültürü. Eksik: CI, kapsama ölçümü. |
| **Kanıt döngüsü (forward)** | **Tasarım 4/5 · İşletim 1/5** | prediction_tracker + gate_sim + günlük harness + verdikt makinesi retail'de NADİR görülür. Ama servisler kapalı → döngü ölü. En ucuz büyük kazanç: AI-0. |
| **Operasyon** | **1.5/5** | Tek makine, OneDrive'a emanet, 101 commitsiz dosya, watchdog kayıtsız. |

### 4.2 Genel hüküm
**"Alpha-öncesi aşamada, metodolojisi boutique-seviye, işletimi hobi-seviye bir araştırma platformu."**
Projenin en değerli varlığı kod değil, **negatif sonuç arşivi + ölçüm disiplini** (EDGE_FINDINGS §1-21):
neyin ÇALIŞMADIĞINI kanıtlarıyla biliyorsunuz. Gerçek para açısından mevcut durumda sistemin doğru
kullanımı: **sermaye koymamak** — sistemin kendisi de bunu söylüyor (META_GATE kapalı, NO_EDGE verdiktleri).

### 4.3 Seviye atlamak için eksik olan üç şey (sırayla)
1. **Kesintisiz canlı kanıt işletimi** (AI-0; ücretsiz, hemen): kenar sorusuna tek dürüst ücretsiz cevap yolu.
2. **Kurumsal PIT veri** (AI-2; ücretli karar): hipotez uzayını gerçekten test edilebilir yapar.
3. **Portföy katmanı** (kenar bulunursa): çoklu-sinyal ağırlıklama/optimizer, factor-maruziyet kontrolü,
   kapasite/likidite analizi. Bugün yazmak erken — kenar yokken portföy optimizasyonu gürültü optimizasyonudur.

---

## EK — Bu oturumda uygulanan optimizasyon (madde 5, TAMAM)
D2/D3/D4/D8/D9/D10 perf kalemleri + B7 kaynak-guard uygulandı — hepsi bit-exact, +8 kalıcı parite
testi (`tests/test_perf_parity_round10.py`). Doğrulama: tam suite ~528 test exit 0 · mypy 56/0 ·
ruff temiz. Somut kazançlar: purged_cv/forward koşularında fold başına ~6M Python iterasyonu →
numpy; permütasyon-önem ~47MB×25/fold kopya churn'ü → 0; `/api/forward-test` 2.4MB CSV okuma 3×→1×;
grafik payload'ları satır-Series kurulumundan kurtuldu. Detay: `OPTIMIZASYON_LOG.md → Ek Tur 10`.
