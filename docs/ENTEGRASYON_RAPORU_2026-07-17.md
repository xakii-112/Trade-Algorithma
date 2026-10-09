# ENTEGRASYON RAPORU — Dış Repo/Kütüphane Değerlendirmesi (2026-07-17)

**Amaç:** Kullanıcının verdiği 8 repo (tf-quant-finance, gs-quant, OpenBB, awesome-ai-in-finance,
500-AI-Agents-Projects, awesome-quant, machine-learning-for-trading, Financial-Models-Numerical-Methods)
içinden projede gerçekten kullanılabilecek parçaları belirlemek; Gemini'nin öneri listesini proje
gerçekliğine göre doğrulamak/düzeltmek ve ek önerilerle P0/P1/P2 yol haritasına bağlamak.

**Yöntem:** Kod tabanı envanteri + kütüphane güncellik/lisans doğrulaması Sonnet 5 ajanlarıyla yapıldı
(2026-07-17); sentez ve planlama Fable 5. Hiçbir kod değiştirilmedi — bu rapor salt değerlendirme + plan.

---

## 0. Çerçeve: Darboğaz model eksikliği DEĞİL

Karar verirken tek referans projenin kendi kanıt geçmişi:

- PEAD kenarı PIT evrende PF 1.57→1.05'e söndü (EDGE_FINDINGS §13.5).
- Meta-labeling (lojistik + **XGBoost + GBM** karşılaştırmalı, `research/train_meta.py`) PIT'te
  OOS-AUC 0.51–0.52 = yazı-tura (§16, §20); Alpaca 10y saatlik 97k işlemde de kenar YOK (§21).
- Forward kayıtçı ~20k tahminde NO_EDGE %47.4.

**Sonuç:** Aynı fiyat verisine daha sofistike model eklemek (Heston, XGBoost, FFT…) darboğazı çözmez.
Entegrasyonlar iki sınıfa ayrılır:

- **Sınıf A — Kanıt/ölçüm altyapısı ve operasyon:** düşük risk, hemen değer (tear-sheet, ikinci earnings
  kaynağı, PBO, RAG operatör komutu). Kapısız girebilir.
- **Sınıf B — Yeni sinyal/model aileleri:** her biri mevcut kanıt kapısından geçmek ZORUNDA
  (PIT evren + purged/embargo CV + deflated Sharpe/PBO + forward shadow, META_GATE deseni: default kapalı).

---

## 1. Gemini listesinin değerlendirmesi (madde madde)

Kısaltmalar: ✅ AL · 🟡 ŞARTLI (kanıt kapısı / veri koşulu) · ❌ ALMA · ♻️ ZATEN VAR

### 1.1 google/tf-quant-finance
| Öneri | Karar | Gerekçe |
|---|---|---|
| Heston → options_analyzer | ❌ | Kütüphane **Mart 2025'te resmen arşivlendi** ("no longer maintained"). Ayrıca TF bağımlılığı bu projeye ağır; kalibrasyon için OPRA kalitesinde zincir verisi de yok (Alpaca ücretsiz plan = indicative feed, geçmiş opsiyon verisi yalnız Şub-2024 sonrası). İleri fiyatlama gerekirse doğru araç QuantLib (aşağıda). |
| SABR (smile/skew) | ❌ | Aynı gerekçe; smile analizi model sorunundan önce **veri** sorunu. |
| Euler MC simülasyonu | ♻️ | `nihai/execution/backtest.py` Madde 15 gereği Monte Carlo drawdown persentilleri (p05/p50/p95) zaten üretiyor. Fiyat-patikası GBM simülasyonu sinyal-only sistemde ek karar değeri üretmiyor. |

### 1.2 goldmansachs/gs-quant
| Öneri | Karar | Gerekçe |
|---|---|---|
| instrument.VanillaOption fiyatlama + Greeks | ❌ | `.price()/.calc()` **GsSession (Marquee) kurumsal yetkisi ister** — bizde yok. Greeks zaten `nihai/risk/options_analyzer.py`'de py_vollib ile hesaplanıyor. |
| risk.RiskMeasure & Portfolio (Madde 27) | ❌ | Aynı gating. `PortfolioGreeks` (risk_manager, Madde 27: net Delta/Theta/Gamma/Vega tavanları) zaten mevcut. |
| (not) gs_quant.timeseries | ❌ | Oturumsuz çalışan tek kısım bu; ama `nihai/signals/indicators*.py` bit-exact parite testleriyle korunuyor — ikinci gösterge kütüphanesi sapma riski, değer yok. |

### 1.3 OpenBB (v4 Platform)
| Öneri | Karar | Gerekçe |
|---|---|---|
| equity.calendar.earnings (Madde 26) | ✅ **listenin en net kazancı** | README zaten `event_calendar` için ikinci kaynak istiyor (kritik canlı kullanım için yfinance tek başına yetersiz). İki uygulama seçeneği: (a) `pip install openbb[nasdaq]` — nasdaq sağlayıcısı **keysiz** çalışıyor; (b) tercihim: OpenBB'yi hiç taşımadan **hafif doğrudan Nasdaq API istemcisi** yazmak (tek endpoint için koca platform + AGPL-3.0 notu gereksiz). Fallback zinciri: yfinance → nasdaq. |
| derivatives.options.chains | 🟡 | yfinance sağlayıcısıyla keysiz çalışıyor; ama proje zaten zincir çekiyor (hub `/options`). Yalnız mevcut zincir kalitesinde sorun görülürse ikinci kaynak olarak gündeme alınmalı. |
| rolling sharpe/sortino (screener/backtest) | ♻️/✅ | Sharpe var; Sortino + profit_factor `backtest.py`'ye küçük ekleme. Bunu OpenBB'den değil quantstats/empyrical-reloaded'dan almak doğru (bkz. §2). |

**Lisans notu:** OpenBB v4 AGPL-3.0 — lokal araştırma kullanımı serbest; ama tek endpoint için taşımaya değmez.

### 1.4 AI önerileri (awesome-ai-in-finance + 500-AI-Agents-Projects)
| Öneri | Karar | Gerekçe |
|---|---|---|
| FinBERT haber duyarlılığı | 🟡 **güçlü şartlı — AI-1 kilidini açar** | GELISTIRME_RAPORU FAZ **AI-1 Gemini ücretsiz-kota engelinde takılıydı**; FinBERT (~110M, CPU'da rahat, kota yok) `nihai/data/llm_event.py`'nin yanına **lokal skorlayıcı** olarak bu ölçümü kotadan bağımsız kılar. ŞART: geriye dönük haber verisi PIT değil → geçmişe eğitim YASAK; skorlar yalnız **shadow-mode** ile `live_features_log`'a yazılır, 3–6 ay birikince `build_live_dataset` ile değerlendirilir. Alternatif model: `yiyanghkust/finbert-tone`. |
| XGBoost sinyal kalite skoru (Madde 18) | ❌ **zaten denendi ve elendi** | `research/train_meta.py` XGBClassifier'ı lojistik ve GBM ile karşılaştırdı; PIT S&P500'de AUC 0.51–0.52, hiçbir eşik kârlı değil (§16/§20/§21). Bu öneri projenin en pahalı öğrenilmiş dersini yok sayıyor. Aynı veriyle model değiştirmek değil, veri değiştirmek (AI-2 Sharadar kararı) veya canlı birikim gerekir. |
| RAG log ajanı (Telegram "neden?") | ✅ | AI-4/AI-5 ile uyumlu, sinyal üretimine dokunmaz. Altyapı hazır: `prediction_tracker` + decisions log + `nihai/ai/{memory,reasoner,router}` + `CommandListener` (mevcut komutlar: add/remove/watchlist/status/yorum/help). Yeni komut: `/neden <SYM> [tarih]`. |

### 1.5 awesome-quant (wilsonfreitas)
| Öneri | Karar | Gerekçe |
|---|---|---|
| pandas-ta / ta-lib ile gösterge motoru değişimi | ❌ | Wilder göstergeleri 53-madde spec'ine bağlı ve **bit-exact parite testleriyle** (Tur-10/12, B7) korunuyor; motor değişimi spec uyumunu riske atar, hız sorunu da yok. Ayrıca orijinal pandas-ta deposu kaldırılmış (fork: pandas-ta-classic). 🟡 İsteğe bağlı: TA-Lib artık resmi Windows wheel'li (v0.7.1, Tem-2026) → yalnız **test-oracle** (CI çapraz-doğrulama) olarak eklenebilir, runtime'a asla. |
| pyfolio & alphalens tear-sheet | ✅ ama **-reloaded** fork'ları | Orijinaller 2020'den beri ölü (Quantopian kapandı). Bakımlı olanlar: `pyfolio-reloaded`/`alphalens-reloaded` (Stefan Jansen — ML4T yazarı) ve pratikte en hızlı kazanç **quantstats** (aktif, Apache-2.0, tek çağrıda HTML tear-sheet). Panelde tear-sheet bugün YOK → gerçek boşluk dolduruyor. |

### 1.6 stefan-jansen/machine-learning-for-trading
| Öneri | Karar | Gerekçe |
|---|---|---|
| Triple-barrier etiketleme | 🟡 | Kavramsal olarak kısmen var (position_manager stop/target/time-stop; meta dataset etiketleri). Deney: `build_meta_dataset`'e ATR-ölçekli üçlü-bariyer etiket **varyantı** eklemek ucuz; ama §21 sonrası beklenti düşük — asıl anlamı YENİ veri (Sharadar) gelirse. |
| Kalman filtresi | 🟡 **ama farklı kullanımla** | Gemini'nin "gürültü temizleme" çerçevesi zayıf. ML4T'deki asıl değer: **pairs trading'de dinamik hedge-ratio**. Kointegrasyon/pairs projede hiç denenmemiş (grep: sıfır iz) → gerçekten YENİ kenar ailesi adayı. PIT evren + Alpaca 10y saatlik veri hazır → `research/pairs_test.py` (xsect.py deseninde), kanıt kapısından. |
| HMM rejim tespiti (Madde 16 rejim) | 🟡 | Doğal yuvası var: `nihai/signals/market_regime.py` (kural bazlı, default KAPALI). İki şart: (1) önce **sıfır-bağımlılık ön-test**: mevcut ~20k forward tahmini basit rejim etiketlerine bölüp NO_EDGE oranı rejime göre ayrışıyor mu bak — ayrışmıyorsa HMM'e hiç girme; (2) girilecekse yalnız expanding-window/walk-forward fit (tam-örneklem HMM fit'i klasik lookahead tuzağı). hmmlearn: wheel var ama "sınırlı bakım" modunda. |
| (ek) Purged CV / Deflated Sharpe | ♻️ | `research/purged_cv.py` (çift-taraflı purge) ve `research/deflated.py` zaten var — ML4T'nin bu bölümleri projede zaten uygulanmış durumda. |

### 1.7 cantaro86/Financial-Models-Numerical-Methods
| Öneri | Karar | Gerekçe |
|---|---|---|
| Binom/trinom ağaçlar (Madde 29 Amerikan) | 🟡 düşük öncelik | Madde 29 bugün settlement-style **metadata**'sı; erken-egzersiz primi hesaplanacaksa notebook kodu vendor'lamak yerine **QuantLib-Python** (v1.43, Tem-2026, BSD, resmi Windows wheel, CRR `BinomialVanillaEngine`) kullanılmalı. Kazanç: Amerikan putlarda BS'nin ıskaladığı erken-egzersiz primi → kontrat skoruna küçük düzeltme. |
| FFT ile opsiyon fiyatlama | ❌ | Heston/Merton kalibrasyonu yoksa FFT'nin konusu yok; BS analitik zaten mikrosaniye — çözülecek hız sorunu yok. |
| Sonlu farklar (BS PDE) | ❌ | Stres için mevcut MC + `/api/var` (VaR/CVaR) yeterli; PDE grid'i saf bakım yükü. |

---

## 2. Ek öneriler (Gemini listesinde olmayanlar)

1. **quantstats tear-sheet → web paneli** *(Sınıf A, P0)* — Sanal pozisyon kapanışlarından equity
   eğrisi üret, `/api/tearsheet` + günlük statik HTML. Panelde bugün hiç tear-sheet yok; Sharpe/Sortino/DD
   grafikleri tek pakete geliyor.
2. **`py_vollib` → `vollib` migrasyonu** *(P0 hijyen)* — py_vollib artık "geçiş paketi", `py_vollib.*`
   import'ları deprecated (Haz-2026). `options_analyzer.py` import'ları taşınmalı; öncesinde/sonrasında
   Greeks **parite testi** şart (proje kültürüne uygun).
3. **PBO/CSCV (Bailey–López de Prado)** *(Sınıf A, P1)* — `research/deflated.py`'nin yanına
   "backtest overfitting olasılığı" metriği; her aday-sinyal raporuna DSR ile birlikte girer. ML4T'de
   referans implementasyon var; mlfinlab KULLANMA (artık ticari lisans; gerekirse açık alternatif `mlfinpy`).
4. **Pairs/stat-arb deney hattı** *(Sınıf B, P1)* — Engle-Granger (+istenirse Johansen) kointegrasyon
   taraması + Kalman hedge-ratio; `research/pairs_test.py`. Bugüne dek test edilen üç aile (PEAD, meta,
   momentum) dışında ilk gerçekten YENİ aile; mevcut PIT altyapısıyla maliyeti düşük.
5. **HMM öncesi bedava ön-test** *(P1'in 0. adımı)* — Yeni bağımlılık/kod yok: forward kayıtçıdaki ~20k
   tahmini SPY trend/vol rejimlerine bölüp isabet/NO_EDGE ayrışmasına bak. Ayrışma yoksa HMM gündemden düşer.
6. **alphalens-reloaded ile aday-sinyal IC şablonu** *(P1, research-only)* — xsect/meta/pairs gibi tüm
   adaylar için ortak IC/quantile/turnover değerlendirme formatı; "kenar var/yok" kararını standartlaştırır.
7. **empyrical-reloaded çapraz-doğrulama testi** *(P0/P1, test-only)* — `backtest.py` metriklerini
   bağımsız kütüphaneyle karşılaştıran test; runtime bağımlılığı değil, parite kültürünün devamı.
8. **Earnings zaman damgası hijyeni (BMO/AMC)** *(Madde 26 + PEAD)* — İkinci kaynak eklenirken
   açıklamanın seans-öncesi/sonrası bilgisini de sakla; PEAD-tipi event çalışmalarında D+1 hizalaması
   buna bağlı ve mevcut tek-kaynakta zayıf.

**Zaten var, ekleme (Gemini'ye/kendime not):** exchange-calendars (requirements'ta + `market_calendar.py`),
Monte Carlo (backtest Madde 15), IV rank hattı (`iv_history.py`), Reddit/VADER sentiment (`sentiment.py`),
LLM 8-K sentiment (`llm_event.py`), purged CV + deflated Sharpe, Sharadar sağlayıcı iskeleti (`sharadar.py`).

---

## 3. Bilinçli dışarıda bırakılanlar

| Ne | Neden |
|---|---|
| tf-quant-finance (tümü) | Arşivlenmiş + TF bağımlılığı + veri önkoşulu yok |
| gs-quant pricing/risk | Kurumsal GS yetkisi ister |
| mlfinlab | Artık ticari/kısıtlı lisans |
| vectorbt / zipline / backtrader | İkinci backtest motoru = spec'li motorla sapma riski; ihtiyaç yok |
| riskfolio-lib / PyPortfolioOpt | Portföy inşası sinyal-only kapsamın dışında |
| pandas-ta (runtime) | Orijinal repo ölü; gösterge motoru zaten spec-bağlı ve parite-testli |
| FFT / sonlu farklar | Çözdükleri problem projede yok |

---

## 4. Yol haritası (öncelik + model ataması)

Kullanıcının istediği iş bölümü: **planlama/organizasyon = Fable 5**, kodlama = uygunluğa göre
**Sonnet 5** (kalıp-takip eden işler) veya **Opus 4.8** (metodolojik/istatistiksel hassas işler).
Her işte Tur-9 dersi geçerli: örtüşmeyen domain partisyonu + tek-thread entegrasyon; deney kodu yazan
ajana bağlam paketi (EDGE_FINDINGS ilgili §'ler + purged_cv/deflated kullanım zorunluluğu) verilmeden iş verilmez.

### P0 — Hijyen + hemen değer (1–2 oturum, kanıt kapısı gerektirmez)
> **Durum 2026-07-18 (Tur-14): P0-1..P0-4 UYGULANDI — commit `25b717e`, 716 test yeşil, ayrıntı OPTIMIZASYON_LOG §Ek Tur 14.**
| # | İş | Dosya/yer | Model |
|---|---|---|---|
| P0-1 | py_vollib→vollib migrasyonu + Greeks parite testi | `nihai/risk/options_analyzer.py`, `tests/` | Sonnet 5 |
| P0-2 | quantstats tear-sheet: `/api/tearsheet` + günlük HTML | `nihai/web/web_dashboard.py`, yeni `nihai/web/tearsheet.py` | Sonnet 5 |
| P0-3 | Madde 26 ikinci earnings kaynağı (hafif Nasdaq client; fallback yfinance→nasdaq; BMO/AMC alanı) | `nihai/data/event_calendar.py` + yeni client | Opus 4.8 (veri katmanı look-ahead korumalarına dokunuyor) |
| P0-4 | backtest'e Sortino + profit_factor + empyrical-reloaded çapraz test | `nihai/execution/backtest.py`, `tests/` | Sonnet 5 |

### P1 — Kanıt-kapılı deneyler (tasarım Fable 5, uygulama Opus 4.8)
| # | İş | Kapı | Model |
|---|---|---|---|
| P1-1 | HMM ön-testi: mevcut forward-log rejim-koşullu isabet analizi (bağımlılıksız) | rapor-only | Opus 4.8 |
| P1-2 | Pairs/kointegrasyon PIT deneyi `research/pairs_test.py` (+Kalman hedge) | PIT + purged CV + DSR/PBO + forward shadow | Opus 4.8 |
| P1-3 | PBO/CSCV modülü `research/pbo.py` (deflated.py yanına) | test + null-baseline | Opus 4.8 |
| P1-4 | Triple-barrier etiket varyantı `build_meta_dataset`'e (yalnız yeni veri gelirse anlamlı) | aynı kapı | Sonnet 5 |
| P1-5 | alphalens-reloaded IC şablonu (research-only) | — | Sonnet 5 |

### P2 — Veri-koşullu / birikim gerektiren
| # | İş | Koşul | Model |
|---|---|---|---|
| P2-1 | FinBERT shadow haber skoru → `live_features_log` kolonları (AI-1 kilidini kotasız açar) | yalnız ileriye dönük birikim; 3–6 ay sonra değerlendirme | Sonnet 5 |
| P2-2 | Telegram `/neden <SYM>` RAG komutu (ai/memory + prediction_tracker) | — | Sonnet 5 |
| P2-3 | QuantLib Amerikan erken-egzersiz primi (options skoruna opsiyonel alan) | opsiyon verisi kalitesi kabul edilirse | Opus 4.8 |
| P2-4 | OpenBB opsiyon zinciri ikinci kaynak | yalnız mevcut zincirde kalite sorunu görülürse | Sonnet 5 |

### Değişmez kurallar
- Sınıf B'de hiçbir sonuç "kenar" ilan edilmez: PIT evren + purged/embargo CV + DSR (ve P1-3 sonrası PBO)
  + forward shadow zorunlu; canlıya etki eden her kapı META_GATE deseninde default-KAPALI/shadow başlar.
- Bağımlılık disiplini: TensorFlow YASAK; her yeni paket için wheel-durumu/lisans satırı bu rapora eklenir.
- Yazımlar `nihai/atomic_io.py` üzerinden (OneDrive kilit dersi); kurulum scriptlerinde BOM'suz ps1 + em-dash tuzağı hatırlatması.
- Alpaca opsiyon verisi kısıtı unutulmasın: ücretsiz plan = indicative feed + 15dk gecikmeli trade,
  geçmiş yalnız Şub-2024 sonrası → IV tabanlı iddialar buna göre yorumlanır.

---

## 5. Kütüphane durum özeti (web doğrulaması, 2026-07-17)

| Paket | Durum | Lisans | Not |
|---|---|---|---|
| tf-quant-finance | **ARŞİVLİ** (Mar-2025) | Apache-2.0 | Kullanma |
| gs-quant | Aktif; pricing GS-oturum ister | Apache-2.0 | Yalnız timeseries oturumsuz |
| openbb v4 | Aktif | **AGPL-3.0** | earnings=nasdaq extra (keysiz), chains=yfinance dahili |
| quantstats | Aktif v0.0.81 (Oca-2026) | Apache-2.0 | pandas 2.x uyumu düzeltilmiş |
| alphalens-/pyfolio-/empyrical-reloaded | Aktif (Jansen) | Apache | Orijinalleri ölü |
| vollib (eski py_vollib) | Aktif; py_vollib=geçiş paketi | MIT | İmport migrasyonu gerekli (P0-1) |
| QuantLib-Python | Aktif v1.43 (Tem-2026) | BSD-3 | Resmi win_amd64 wheel; CRR binom |
| TA-Lib | Aktif v0.7.1 (Tem-2026) | BSD-2 | Artık resmi Windows wheel; yalnız test-oracle |
| pandas-ta | Orijinal ÖLÜ; fork pandas-ta-classic | belirsiz | Runtime'a alma |
| hmmlearn | Sınırlı bakım (0.3.3) | BSD | Wheel var; research-only |
| mlfinlab | **Ticari lisans** | kısıtlı | Alternatif: mlfinpy / ML4T kodu |
| FinBERT (ProsusAI) | Fiili standart, ~110M | — | CPU'da rahat; alternatif finbert-tone |
| statsmodels | Aktif 0.14.6 (venv'e kuruldu 2026-07-18) | BSD-3 | Yalnız research: P1-2 Engle-Granger/ADF; runtime'a girmez |

## 6. Gemini'nin yol düzeltmeleri (uygulama ajanları için)
- `event_calendar.py` → `nihai/signals/` değil **`nihai/data/`**
- `auto_screener.py` → **`nihai/execution/auto_screener.py`**
- `position_manager.py` → **`nihai/risk/`**
- `telegram_bot.py` → **`nihai/notifier/`**
- "signal_engine'e XGBoost ekle" hedefi geçersiz: bu deney `research/train_meta.py`'de yapıldı ve PIT'te elendi.

*Rapor: Fable 5 (sentez/plan) + 2× Sonnet 5 ajan (kod envanteri, web doğrulama). Kod değişikliği yapılmadı.*
