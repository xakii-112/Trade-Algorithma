# NihaiAlgoritma "toplu" — Hata Düzeltme + Optimizasyon Logu

> **Konum notu (2026-08-23).** Bu dosya Tur-43'e kadar depo KÖKÜNÜN DIŞINDA
> (`NihaiAlgoritma toplu/`) duruyordu: izlenen `CLAUDE.md` ona atıf yapıyor ama git
> onu hiç görmüyordu — yani projenin en pahalı kurumsal bilgisi sürüm kontrolü
> dışındaydı ve bir kaza onu geri alınamaz biçimde silebilirdi. Depo köküne
> (`CHANGELOG.md` ile kardeş) taşındı; kardeşi olan üç tarihli rapor
> (`DENETIM_BULGULARI_2026-07-09` · `ENTEGRASYON_RAPORU_2026-07-17` ·
> `GELISTIRME_RAPORU_2026-07-10`) `docs/` altındaki tarihli-rapor konvansiyonuna
> katıldı. Kod ve belgelerdeki atıflar ÇIPLAK DOSYA ADIYLA yapıldığı için hiçbiri
> kırılmadı; yalnız bu dosyanın içindeki dört `NihaiAlgoritma/docs/...` yolu
> depo-göreli `docs/...` biçimine indirildi.

Başlangıç baseline: **241 test geçti**, ruff temiz, tüm modüller import oluyor.
Odak (kullanıcı seçimi): **Kod kalitesi & sağlamlık · Performans & hız · MainHub & entegrasyon**
(Strateji/edge HARİÇ — gerçek PIT veri gerektirir.)

---

## Faz 1 — Hata & Eksik Bulma/Düzeltme

### BULGU-1 (YÜKSEK, entegrasyon+sağlamlık): `alpaca_stream_provider.py` iki kopya çift yönlü ıraksamış
İki kopya (`MainHub/` ve `nihai/data/`) farklı yönlerde gelişmiş; her biri diğerinin sağlamlık özelliğini kaybetmiş:

- **nihai/data** YENİ özelliklere sahip ama **`_run_forever` monkeypatch'i YOK** →
  bağlantı/DNS hatasında veya dinamik-sembol modunda (handler yokken) alpaca-py iç döngüsü
  `await asyncio.sleep(0)` ile **%100 CPU spin + log flood** yapar. Gerçek sağlamlık regresyonu.
- **MainHub** monkeypatch'e sahip ama şu YENİ özellikleri YOK:
  - `subscribe_trades` / `_on_trade` → gerçek son **işlem** fiyatı (Midas gibi), bid/ask orta noktası değil
  - lookback-duyarlı seed cache (daha büyük pencere istenince yeniden seed'ler) — kısa-geçmiş hatası
  - `data_provider.py`: quote için ayrı kısa-TTL cache (`QUOTE_CACHE_TTL_SEC`) — canlı fiyat bayatlamasın

**Düzeltme (TAMAM):** İki kopya birleşim olacak şekilde uzlaştırıldı.
- `nihai/data/alpaca_stream_provider.py`: monkeypatch eklendi (MainHub'dan port).
- `MainHub/alpaca_stream_provider.py`: nihai sürümüyle ÖZDEŞ yapıldı (trades/lookback-seed/
  gerçek-son-fiyat/stream_status) — yalnız import yolları farklı.
- `MainHub/data_provider.py`: `CachingProvider`'a quote kısa-TTL (`QUOTE_CACHE_TTL_SEC`) eklendi.
- İki dosya artık yalnız import-yolu farkıyla özdeş → gelecekte ıraksama riski düştü.

### BULGU-2 (DÜŞÜK, tip güvenliği): mypy hiç çalıştırılmamış; `ClassVar` eksikleri + monkeypatch atama
- mypy baseline'da kurulu değildi (`MainHub/.venv` içinde var) → ilk kez çalıştırıldı.
- nihai/data: `_yf_period`, `_calls` mutable class-default'ları `ClassVar` ile işaretlendi.
- monkeypatch atama satırına `# type: ignore[method-assign]` eklendi (kasıtlı).
- **Sonuç:** `mypy nihai` → *Success, 0 hata* (52 dosya); ruff temiz; **241 test geçiyor.**

### Sweep (temiz): `except:` (bare), `== None`, mutable default arg → proje kodunda YOK
(yalnız `.venv` üçüncü-parti kütüphanelerde — bizim kod temiz.)

---

## Faz 2 — 20 Döngülük Optimizasyon & İyileştirme

Her döngü: tek bir somut iyileştirme → `pytest` + `ruff` + (mümkünse) `mypy` ile doğrulanır.
Mevcut çalışan sistem korunur; her değişiklik geri-doğrulanır.

| # | Alan | İyileştirme | Durum |
|---|------|-------------|-------|
| 1 | MainHub/robustluk | MainHub'a test altyapısı + sağlayıcı testleri (Caching: quote kısa-TTL/0-TTL/boş-cache'lememe; Failover: primary/secondary/iki-fail/source_stats; Quote sağlık) — **13 test** | ✅ |
| 2 | MainHub/entegrasyon | `hub_server.py` HTTP endpoint testleri (FastAPI TestClient + fake upstream): /health /bars /quote /options + boş/None yolları — **7 test** | ✅ |
| 3 | MainHub/entegrasyon | `HubProvider` HTTP-istemci ayrıştırma + hub-düşük zarif degrade testleri — **7 test** | ✅ |
| 4 | Sağlamlık/regresyon | nihai `AlpacaStreamProvider` SAF-mantık testleri (BULGU-1 koruması: lookback-seed re-seed, gerçek-son-işlem quote, _on_trade, _aggregate_minutes, stream_status) — **11 test** (nihai 241→252) | ✅ |
| 5 | Entegrasyon/guard | **Kalıcı ıraksama guard'ı** (`test_stream_parity.py`): iki stream kopyasının (a) birleştirilmiş kritik özellikleri koruduğunu, (b) sınıf gövdesinin import dışı BİREBİR aynı olduğunu doğrular → BULGU-1 sessizce geri gelemez — **3 test** (nihai 252→255) | ✅ |
| 6 | MainHub/altyapı | MainHub'a `pyproject.toml` (ruff E/F/I/B/UP + pytest testpaths) → ilk-sınıf lin/test modülü. Sıkı lint 4 gizli nit yakaladı (import sırası + `.format`→f-string); İKİ stream kopyasında da özdeş auto-fix uygulandı (parity korundu) | ✅ |
| 7 | Sağlamlık/config | `config.py` saf parse yardımcıları (`_parse_tiers`, `_envb`) edge-case testleri: bozuk/boş/None girdi sessiz-atlama, tiered-trail AZALAN sıralama (yanlış basamak seçimi koruması), bool token tablosu — **10 test** | ✅ |
| 8 | Sağlamlık/go-no-go | `execution/gate.py` dağıtım kapısı kararları (Madde 52): `evaluate_backtest_dict` (min-trades/MDD sınır-içi, eksik-anahtar→BLOCK, MC ruin opsiyonel/yüksek/düşük, to_dict şekli) + `evaluate_parity` (E~0 özel dalı, tolerans-içi/dışı, WR sapması) — **14 test** | ✅ |
| 9 | Sağlamlık/seçim | `risk/option_strategy.py` `_score_contract` saf skorlama monotonluğu: hedef-delta yakınlığı (↑), |delta| put-simetrisi, OI likidite (↑), makas cezası (↓), delta-skor 0-tabanı — **6 test** | ✅ |
| 10 | Sağlamlık/gösterge | `signals/indicators_ext.py` MA ailesi (sma/wma/smma/dema/tema/tma/hma) kesin/değişmez + cmo sınır [-100,100] & monoton ±100 — **7 test** | ✅ |
| 11 | Sağlamlık/risk-guard | `data/event_calendar.py` guard'ları (Madde 17/25/26): earnings/ex-div/split tampon penceresi (0..buffer dahil sınır), geçmiş-olay bloke-etmez, IV-crush proximity — **10 test** | ✅ |
| 12 | Sağlamlık/IV | `data/iv_history.py` `get_iv_rank` güven katmanı (NO_HISTORY/LOW/PARTIAL/OK) + 0-100 rank ölçeği + aralık-dışı kırpma (tmp_path CSV) — **6 test** | ✅ |
| 13 | Sağlamlık/Greekler | `risk/options_analyzer.py` `compute_greeks` (geçersiz→None, call δ>0/put δ<0, ATM δ∈(0,1), iv geçişi; py_vollib-bağımsız) + `assignment_risk_short_call` ITM/ex-div sınırı — **8 test** | ✅ |
| 14 | Sağlamlık/portföy | `PortfolioGreeks` (Madde 27): **theta/vega ×100 ölçek-bug regresyon guard'ı** (4 Greek de contracts×multiplier ile), çok-pozisyon birikimi, delta/theta/gamma/vega cap ihlali — **7 test** | ✅ |
| 15 | Sağlamlık/fail-safe | `signals/meta_gate.py` opt-in disiplini: model yok/bozuk→PASİF (probability None, çökme yok), geçerli lojistik yükleme, eşik default, get_gate cache + reload (tmp_path) — **5 test** | ✅ |
| 16 | Sağlamlık/osilatör | `indicators_ext` oy-veren osilatörler: demarker [0,1] & monoton 1/0, dpo sabit→0, ppo trend-işareti & sabit→0, tsi [-100,100] & monoton ±100 — **7 test** | ✅ |
| 17 | Sağlamlık/sinyal | `data/sentiment.py` `sentiment_signal` saf karar: eksik-veri WAIT, düşük-hacim filtresi, boğa→LONG/ayı→SHORT, nötr WAIT, contrarian fade, score_min sınırı — **10 test** | ✅ |
| 18 | Sağlamlık/sinyal | `data/insider.py` `insider_signal` saf karar (Form 4 küme-alım): küme eşiği, alım>satım baskınlığı, satış varsayılan-KAPALI, allow_short→SHORT — **7 test** | ✅ |
| 19 | Sağlamlık/survivorship | `data/pit_universe.py` `sp500_as_of` PIT yeniden-kurulum: as_of-sonrası eklenen dışlanır / çıkarılan dahil, önceki değişiklik etkisiz, `removed_since` (monkeypatch — ağsız) — **5 test** | ✅ |
| 20 | Sağlamlık/None-yolu | `options_analyzer.estimate_atm_iv` guard'ları (Madde 16): None/boş df, geçersiz underlying/dte, eksik strike kolonu, sıfır-fiyat → güvenli None (çökme yok) — **7 test** | ✅ |

---

## Özet Durum (20 döngü TAMAMLANDI)

**Baseline → Final:**
- Testler: **241 → 391** (nihai 364 + MainHub 27). MainHub eskiden **0 testti**.
- Tip kontrolü: mypy hiç çalışmıyordu → **`mypy nihai` temiz (52 dosya, 0 hata)**, CI'ye eklenebilir.
- Lint: nihai **ve** MainHub `ruff` temiz (MainHub artık kendi pyproject'iyle sıkı lint'li).
- **1 gerçek sağlamlık bug'ı düzeltildi** (BULGU-1: stream sağlayıcı çift-yönlü ıraksama →
  CPU-spin/log-flood + bayat fiyat + kısa-geçmiş riski). Kalıcı guard ile kilitlendi.

**Yöntem:** Kod tabanı olgun (sofistike paralel tarama, atomik state, look-ahead korumaları
zaten mevcuttu) → optimizasyon, *spekülatif yeniden-yazım* yerine **gerçek bug düzeltme +
davranış-kritik saf fonksiyonların test ağına alınması** üzerine odaklandı. 20 döngüde
+150 test eklendi; HİÇBİRİ üretim davranışını değiştirmedi (yalnız BULGU-1 düzeltmesi +
güvenlik ağı) → çalışan sistemi bozma riski sıfır, regresyon koruması kalıcı.

**Kapsama alınan kritik karar yolları (önceden test-dışıydı):**
go/no-go dağıtım kapısı · opsiyon kontrat skorlama · Black-Scholes Greekler · portföy
Greek cap'leri (ölçek-bug guard'ı) · IV rank güven katmanı · earnings/split/ex-div risk
guard'ları · meta-gate fail-safe · sentiment/insider sinyal kararları · PIT survivorship
yeniden-kurulumu · config parse · gösterge ailesi & osilatörler.

### Artık güvenlik ağında olmayan / gelecek iş (isteğe bağlı)
- Gerçek profil-temelli perf (canlı taramada darboğaz ölçümü) — gerçek iş yükü gerektirir.
- `execution/` orkestrasyon yolları (`screen`/`main`/`live_predict`) — entegrasyon-ağır,
  fake-provider harness gerektirir; saf-mantık katmanı artık kapsanıyor.

---

## Ek Denetim — 2026-07-02

Baseline: nihai 382+1skip / MainHub 27 test yeşil, ruff+mypy iki projede temiz.
Tüm nihai+research modülleri smoke-import edildi → bir gerçek bug yakalandı:

### BULGU-3 (ORTA, yan-etki): `research/check_options_yf.py` import anında çalışıyordu
`__main__` guard'ı yoktu → modül İMPORT edilir edilmez 519 sembollük yfinance ağ taraması
başlatıp `data/options_availability_yf.json`'u üzerine yazıyordu (pytest toplama/IDE/araç
importu tetikleyebilirdi). **Düzeltme:** gövde `main()` içine alındı + guard eklendi;
davranış komut satırından değişmedi.

### Belge/kullanım düzeltmeleri (yanlış komutlar `ModuleNotFoundError` veriyordu)
- `python research/x.py` biçimindeki örnekler `python -m research.x`'e çevrildi:
  `.env.example` (insider/sentiment), `research/purged_cv.py` docstring, `docs/IMPROVEMENT_PLAN.md`.
- `python prediction_tracker.py` → `python -m nihai.execution.prediction_tracker`
  (`.env.example` + modül docstring'i; dosya kökte değil `nihai/execution/`te).
- CLAUDE.md paket listesi bayattı (`notifier/`, `trade/` kök paketleri artık yok) → düzeltildi.
- Kökteki yetim `notifier/` (yalnız `__pycache__`) silindi.

### Eksik yapılandırma belgeleri
- `.env.example`'a kodda okunan ama belgelenmemiş **40 anahtar** eklendi (STREAM_SYMBOLS,
  WEB_TOKEN, META_GATE, PROB_SIZING*, PDT_*, LLM_EVENT_*, FORWARD_*, LOG_TRUNCATE_MAX_BYTES,
  LONG_ONLY, VIX_GATE, GATE_*, SETUP_EXT_*, SENTIMENT_COLLECT vb.) — hepsi kod varsayılanında.
- MainHub'da hiç `.env.example` yoktu → oluşturuldu (sırlar boş; META_GATE=false disiplinli varsayılan).

### İşaretlendi, DEĞİŞTİRİLMEDİ (kullanıcı kararı gerekir)
1. Her iki `.env`'de `META_GATE=true` + `META_GATE_THRESHOLD=0.45` — CLAUDE.md "ACI DERSLER"
   §16 (PIT'te kenar YOK → varsayılan KAPALI kalmalı) ile çelişiyor; ama dosyadaki yorum
   bilinçli açıldığını söylüyor. Kapatmak istersen iki `.env`'de `META_GATE=false` yap.
2. `trade/` klasörü yanlışlıkla Obsidian vault'a dönmüş (`.obsidian` + "Hoş geldiniz.md") —
   kod yok; istersen silinebilir.
3. `.env`'deki `GEMINI_API_KEY` biçimi alışılmadık (`AQ.` öneki; Gemini anahtarları genelde
   `AIza` ile başlar) — LLM olay analizi kullanılacaksa anahtarı doğrula. (Değerin başındaki
   boşluk dotenv'ce kırpılıyor — zararsız, test edildi.)

### Yanlış-pozitif olduğu doğrulananlar (değişiklik yok)
`mypy --check-untyped-defs`'in 22 bulgusu incelendi: hepsi korumalı yollar (meta_gate
kind-guard'ları, backtest'te dal-içi `return`, web_dashboard bilinçli karışık-tip fallback,
bytes/bytearray split) → çalışma zamanı riski yok. `WATCHLIST` sonundaki sarkan virgül
parse'ta filtreleniyor. Son durum: **nihai 382+1skip, MainHub 27 test yeşil; ruff+mypy temiz.**

---

## Ek Denetim 2 + AI Canlı-Kanıt Döngüsü — 2026-07-02 (ikinci tur)

### Hata düzeltmeleri
- **BULGU-4 (ORTA, CLI tutarsızlığı):** `screen.py --provider` seçenekleri yalnız
  yfinance|tradingview idi; oysa kodun kendi ipucu paralelde "hub/alpaca düşünün" diyor ve
  `load_bars` hub/alpaca/alpaca_stream'i DESTEKLİYOR → tavsiye edilen bayrak argparse'ta
  reddediliyordu. Seçenekler genişletildi.
- **BULGU-5 (DÜŞÜK-ORTA, CWD bağımlılığı):** `nihai/data/hub_server.py` log dizini
  `os.getcwd()/logs` idi (başlatıldığı klasöre `logs/` saçar; MainHub kopyası dosya-köklü).
  `config.LOG_DIR`'a sabitlendi.
- **BULGU-6 (DÜŞÜK, sağlamlık paritesi):** `llm_event._extract_anthropic` geçici hatada
  (429/529/5xx/overloaded) retry'sızdı; Gemini yolunda 3-deneme backoff vardı. Parite sağlandı
  (toplu 8-K çıkarımında anlık yük hatası dosyalamayı günlerce atlatmasın).
- main.py makro-except yorumu davranışla hizalandı ("çarpan 1.0'da" değil "son bilinen
  değerde" — kasıtlı: FRED kesintisi risk-off korumasını sıfırlamasın; davranış değişmedi).

### AI geliştirmesi — meta-kapının CANLI kapı simülasyonu (ACI DERS #3 & #5 uygulaması)
Tespit: PREDICT_TRACK 8 gündür açık, 11.872 çözülmüş tahmin birikmiş ama **pwin'li satır 0**
— META_GATE açıkken kapı-engelli sinyaller WAIT'e dönüp P(win)'lerini kaybediyordu →
canlı log yalnız eşik-üstünü görür (seçilim yanlılığı) ve "kapı işe yarıyor mu?" canlıda
ASLA ölçülemezdi. Kapalı döngü tamamlandı:
1. `strategy.py`: kapı-engelli WAIT'in metadata'sına elenen yön (`meta_gated`) yazılır.
2. `live_predict_service._signal_pwin` (saf, test'li): aktif sinyal VE kapı-engelli aday
   için (pwin, pwin_side); sıradan WAIT'te (None, None) — dejenere-model yanıltması yok.
3. `prediction_tracker`: şemaya `action` + `pwin_side` kolonları (atomik header-upgrade
   mevcut `_ensure_schema` ile); `gate_sim()` saf canlı kapı simülasyonu — tutulan(p≥eşik)/
   elenen/taban isabet & taraf-işaretli ort. getiri, örneklem-kapılı (GATE_SIM_MIN_SIDE=30,
   iki tarafta da altındaysa INSUFFICIENT: hüküm YOK). `report()` → `meta_gate` bölümü;
   CLI `--report`, `research/forward_harness --report` ve web accuracy paneli gösterir.
4. Dürüstlük korunur: kenar İDDİA edilmez, ÖLÇÜLÜR; INSUFFICIENT altında etiket verilmez.
   Gerçek log smoke'u: eşik 0.45, aday n=0 → INSUFFICIENT (beklenen; adaylar şimdi birikecek).

**Doğrulama:** nihai 393 test (+7 yeni: şema/side-işaret/gate_sim/helper), MainHub 27;
ruff + mypy (54 dosya) iki projede temiz. Gerçek 11.9k-satır log üzerinde `--report` smoke OK.

### Not (kullanıcı kararı, değiştirilmedi)
`.env` META_GATE=true hâlâ CLAUDE.md varsayılan-KAPALI disipliniyle çelişiyor; artık
gate_sim canlı kanıt biriktirdiği için bu karar birkaç hafta içinde VERİYLE verilebilir
(GATE_HURTS/NEUTRAL çıkarsa kapat, GATE_HELPS çıkarsa örneklemi büyüt).

---

## Ek Tur 3 — Canlı EĞİTİM-VERİSİ hattı (AI'ı eğitilebilir yapma) — 2026-07-02 (üçüncü tur)

Baseline doğrulandı: nihai 389+1skip / MainHub 27 yeşil, ruff+mypy temiz.

### Tespit: canlı log kapıyı ÖLÇEBİLİYOR ama modeli EĞİTEMİYORDU
gate_sim (tur 2) pwin'i logluyor; fakat pwin'i üreten **özellik vektörü hiçbir yere
yazılmıyordu** → biriken canlı veri (11.9k+ çözülmüş tahmin) yeniden-eğitim için
kullanılamazdı. ACI DERS #5'in iki yolundan (Sharadar PIT parası VEYA canlı kanıt)
canlının eğitim-verisi ayağı eksikti.

### Yapılan (kapalı eğitim döngüsü)
1. `prediction_tracker`: `FEATS_LOG` (`data/live_features_log.csv`, başlık = FEATURE_NAMES,
   (symbol,bar_ts) anahtarlı) + `_write_features` — yalnız record()'un YENİ yazdığı
   satırlarla, atomik başlık-yükseltmeli, tahmin kaydını asla engellemeyen (try/except).
2. `live_predict_service`: track açıkken ADAY sinyallerin (giriş LONG/SHORT VEYA
   kapı-engelli `meta_p_win`'li WAIT) özellikleri `_bar_context`'te bar başına BİR kez
   `extract_features` ile çıkarılır; `_feats` yalnız disk-kaydına taşınır, record()
   sonrası satırdan düşürülür (web JSON'una sızmaz).
3. `research/build_live_dataset.py` (YENİ): çözülmüş+pwin_side'lı tahminler × özellik
   logu join → `data/meta_dataset_live_<tarih>.csv`, train_meta/purged_cv ile birebir
   uyumlu (FEATURE_NAMES + pnl/label/symbol/period/entry_time). Etiket = horizon-bazlı
   TARAF-işaretli getiri işareti (`_side_return` tek kaynak); dürüst sınırlar docstring'de
   (stop-yönetimli PnL değil; --min-rows 100 altında CSV yazmayı reddeder).
4. Testler: +5 (feature-log yazımı/dedup/başlık-yükseltme + join/etiket-işareti/boş-log).

**Doğrulama:** nihai **394+1skip**, MainHub 27; ruff + mypy (55 dosya) temiz.
Gerçek loglar üzerinde `python -m research.build_live_dataset` smoke OK (0 satır —
beklenen: özellik logu bu değişiklikle birikmeye BAŞLAYACAK; predict servisi yeniden
başlatılınca aktif olur, PREDICT_TRACK=true zaten açık).

**Anlam:** Sistem artık kendi canlı, survivorship-temiz, leak-free eğitim setini
biriktiriyor. Yeterli örneklem (yüzlerce çözülmüş aday) birikince:
`python -m research.build_live_dataset` → `python -m research.train_meta --csv ...`
→ kapı simülasyonu iki bölmede de pozitifse `--save` + `reload_gate()` ile canlıya al.

### BULGU-7 (ORTA, test hijyeni): pytest ÜRETİM loglarına yazıyor + dosya silebiliyordu
`test_duo.py` → `research/duo.py` → `from nihai.execution.main import ...` zinciri main'in
modül-seviyesi kodunu test sürecinde çalıştırıyordu: (a) loguru dosya sink'i eklenir → o
andan sonra TÜM test logları gerçek `logs/system.log`'a akar (318 sahte "boom" ERROR
birikmişti; izleme/watchdog'u yanıltır), (b) import-anı `prune_logs()` GERÇEK log/CSV
silebilir (canlıda görüldü: retention testte 1 screen_csv sildi). **Düzeltme:**
`NIHAI_DISABLE_FILE_LOG` env guard'ı (main.py) + kök `conftest.py` bunu test sürecinde
set eder. Üretim davranışı değişmedi; doğrulama: tam suite koşusunda system.log artık
büyümüyor (boom 318→318). **394+1skip / 27 test, ruff+mypy temiz.**

---

## Ek Tur 4 — Test/backtest veri-hijyeni + kapalı eğitim döngüsünün son halkası — 2026-07-02 (dördüncü tur)

Baseline: nihai 398 / MainHub 27 yeşil; ruff'ta 2 önemsiz import-sort (düzeltildi).

### BULGU-8 (YÜKSEK, veri hijyeni): pytest ÜRETİM trade/decision/screen/trades dosyalarına yazıyordu
BULGU-7 yalnız system.log sink'ini korumuştu; testler signal_engine/position_manager/
auto_screener yoluyla GERÇEK `logs/trade_log.txt`, `decision_log_*.txt`, `auto_screen_*.csv`
ve `data/trades.csv`'ye sahte işlem yazıyordu (gözlem: trades.csv 55k satırın 9.2k'sı
AAA/BBB/X-TEST_CLOSE; bugünkü decision_log %100 sahte; 31/33 auto_screen 50-baytlık artık).
**Düzeltme:** `nihai/config.py`'ye `NIHAI_LOG_DIR`/`NIHAI_DATA_DIR` env override; kök
`conftest.py` test sürecinde ikisini de geçici dizine yönlendirir. MainHub paritesi:
`MAINHUB_LOG_DIR`/`MAINHUB_DATA_DIR` + hub_server.py sink'i config.LOG_DIR'a bağlandı.
Doğrulama: tam suite sonrası üretim dosya hash'leri BİREBİR değişmedi.

### BULGU-9 (YÜKSEK, veri hijyeni): backtest de aynı üretim dosyalarına yazıyordu
`PositionManager` her aç/kapanışta koşulsuz log_decision/log_trade_postmortem/log_trade_csv
çağırıyordu → backtest koşuları trades.csv/trade_log/decision_log'u dolduruyor,
`research/postmortem.py` raporları + dashboard/VaR canlı-dışı veri okuyordu.
**Düzeltme:** `PositionManager(record_logs=...)` bayrağı (varsayılan True — canlı
main/web/duo değişmedi); backtest.py iki sitede `record_logs=False`.

### Temizlik (geri alınabilir — silinmedi, arşivlendi)
Kirli dosyalar `logs/contaminated_archive_20260702/` + `data/contaminated_archive_20260702/`
altına taşındı: trade_log.txt, trades.csv (55k karışık satır), bugünkü decision_log,
31 sahte auto_screen CSV. Temiz dosyalar bundan sonra YALNIZ canlı veriyle birikir.

### BULGU-10 (ORTA, operasyon): forward-harness görevi 2026-07-02 02:29'da sessiz öldü (sonuç 1)
PS 5.1 tuzağı: `$ErrorActionPreference=Stop` + native `2>&1` → python stderr'e TEK satır
yazsa NativeCommandError fırlar, çıktı kaybolur. Hub kapalıyken data_provider'ın failover
WARNING'leri stderr'e akar → görev her gece ölürdü. **Düzeltme:** forward-harness.ps1
çağrı süresince EAP=Continue + exit-code denetimi. Doğrulama: hub kapalıyken elle koşuldu,
exit=0, tam rapor loglandı.

### Operasyon tespiti (kod değil — kullanıcı aksiyonu gerekir)
predictions_log.csv 2026-06-29 22:25'ten beri BÜYÜMÜYOR ve live_features_log.csv hiç
oluşmadı: hub (8799) + predict recorder (8765) süreçleri çalışmıyor; Task Scheduler'da
yalnız `NihaiAlgo-ForwardHarness-Daily` kayıtlı, watchdog'lar KAYITLI DEĞİL. Canlı
eğitim-verisi birikimi bu yüzden durdu. Başlatma: `start-hub.ps1` + `run.ps1`/watchdog'lar
veya `register-watchdog-system.ps1` (kullanıcı kararı — bu turda hiçbir servis başlatılmadı).

### AI geliştirmesi — kapalı eğitim döngüsünün SON halkası (tek-komut yeniden-eğitim)
1. **train_meta `--save` disiplin açığı kapatıldı:** eskiden TEK bölmedeki kapı kenarı
   bile modeli kaydediyordu (özet "iki bölme ister" diyordu ama kod zorlamıyordu — overfit
   model üretime sızabilirdi). Artık AYNI model hem time hem symbol bölmesinde kenar
   göstermeden kayıt YOK; seçim kötü-bölge expectancy'sine göre (muhafazakâr). Kayıt
   `save_model()` ortak fonksiyonuna çıkarıldı; mevcut model `.bak-<tarih>` arşivlenir.
2. **`research/live_retrain.py` (YENİ):** build_live_dataset → guardrail (≥300 satır,
   ≥10 sembol, ≥10 gün) → time+symbol kapı simülasyonu → purged 5-fold CV → verdikt
   (INSUFFICIENT/NO_EDGE/DURABLE); `--promote` YALNIZ DURABLE'da tam-veri modeli kaydeder.
   Terfi asla otomatik değil. Smoke: boş canlı logda INSUFFICIENT + promote reddi (doğru).
3. Testler: +6 (`test_live_retrain.py` — guardrail/öğrenilebilir-sinyal→DURABLE/
   gürültü→NO_EDGE/süre-guardrail/eksik-kolon).

**Doğrulama:** nihai **404** test, MainHub 27; ruff iki projede temiz. CLAUDE.md
komutlar + çalışma ilkeleri güncellendi.

---

## Ek Tur 5 — Bug avı + AI özellik-uzayı ve canlı-kanıt derinleştirmesi — 2026-07-03

Baseline doğrulandı: nihai 400 / MainHub 27 yeşil, ruff + mypy (54 dosya) iki projede temiz.

### BULGU-11 (ORTA, veri hijyeni): `build_live_dataset.py` NaN'ı dataset'e sızdırıyordu
`float(frow.get(n, 0.0) or 0.0)` — **NaN truthy olduğundan `or` onu geçirir** → başlık-
yükseltmeli eski özellik-log satırlarının '' hücreleri NaN olarak eğitim CSV'sine akar,
lojistik model sessizce NaN üretirdi. **Düzeltme:** `pd.notna` + özellik-başına nötr
varsayılan (`neutral_default` — extract_features guard'ıyla TEK tablo). Regresyon testli.

### BULGU-12 (ORTA, yanlış verdikt): `purged_cv.py` kapı-vs-base fold hizasızlığı
`fold_gate_e` yalnız kept≥5 foldlarda, `fold_base_e` HER foldda birikiyordu → bir fold
eşiği geçemezse `ge > be` yanlış fold çiftlerini kıyaslar (veya boyut hatasıyla çöker) →
"kapı kazandığı fold" sayısı ve DAYANIKLI/DEĞİL verdikti bozulabilirdi. **Düzeltme:**
(kapı E, base E) çift olarak saklanır (live_retrain._purged_gate deseniyle aynı).

### AI geliştirmesi 1 — özellik uzayı 19→25 (canlı log doğmadan, sıfır-maliyet anı)
`live_features_log.csv` henüz hiç oluşmadığı için FEATURE_NAMES genişletmesi geriye dönük
boş-kolon kirliliği yaratmadan yapıldı: **gap_pct, range_pct, close_pos, streak,
rsi_slope, hour** (hepsi yalnız df_slice'tan — look-ahead yok; karar-barı mikro-yapısı +
momentum-ivmesi + gün-içi saat). Uyumluluk katmanı:
- Eski modeller kayıtlı `features` listesiyle çalışmaya devam eder (features_to_vector
  names param — train/serve hizası zaten garantiliydi).
- `ensure_feature_columns` (meta_features'ta, NEUTRAL_DEFAULTS tek kaynak): eski-şema
  dataset CSV'leri train_meta / purged_cv / live_retrain'de okuma anında nötr-doldurulur
  (KeyError yerine uyarı basar; NaN'lar da nötrlenir). Gerçek eski CSV ile smoke edildi.

### AI geliştirmesi 2 — canlı kanıt döngüsüne eşik-taraması + P(win) kalibrasyonu
- `gate_sim`'e `sweep`: 0.35–0.60 ızgarasında her eşik için tutulan/elenen istatistiği;
  **her eşik kendi örneklem kapısından geçer** (iki taraf ≥ min_side değilse uplift=None —
  hüküm yok). "Aktif eşik 0.45 doğru mu, kaç olmalı?" sorusu artık veriyle cevaplanacak.
- `report()`'a `pwin_calibration`: meta P(win) kovası → gerçek taraf-kazanma eğrisi
  (yön-güveni kovalarından AYRI; modelin canlı kalibrasyonunu ölçer). CLI çıktısında da.
- Kozmetik: confidence=1.0 hayalet "1.0-1.1" kovası 0.9-1.0'a kırpıldı.
- Ortak çekirdek `_pwin_pairs` (gate_sim/sweep/kalibrasyon tek kaynak).

### Operasyon durumu (kod değil — Tur-4 tespiti sürüyor)
Hub (8799) + predict (8765) süreçleri hâlâ ÇALIŞMIYOR (bugünkü python süreçleri MCP
araçları); predictions_log 29 Haziran'dan beri büyümüyor, live_features_log hiç oluşmadı.
Yeni 25-özellikli birikim, servisler başlatılınca temiz şemayla başlayacak (bilinçli
zamanlama). Başlatma kullanıcı kararı: `start-hub.ps1` + `run.ps1` / watchdog kayıtları.

**Doğrulama:** nihai **407** test (+7: mikro-özellikler/streak-gap/nötr-guard/
ensure_feature_columns/NaN-regresyonu/sweep/pwin-kalibrasyon), MainHub 27; ruff + mypy
(54 dosya) iki projede temiz. Gerçek 11.9k log üzerinde `--report` smoke OK
(aday n=0 → INSUFFICIENT, beklenen); eski-şema gerçek CSV'de train_meta + purged_cv smoke OK.

---

## Ek Tur 6 — Profil-temelli PERFORMANS optimizasyonu (davranış bit-bit korunarak) — 2026-07-03

Baseline doğrulandı: nihai 426 / MainHub 27 yeşil, ruff temiz. Bu tur, Faz-2 özetinde
"gelecek iş" bırakılan **gerçek profil-temelli perf**'i kapattı. Yöntem: cProfile +
sentetik OHLCV (ağsız) → sıcak yol kanıtı → değişiklik → **altın-referans bit-bit kıyas**.

### Profil kanıtı (2000 bar, tek sembol backtest)
`backtest_symbol` 5.27s → bunun **4.89s'i (%93) `generate_signal`**. Kırılım: `_true_range`
2.28s (pd.concat+max(axis=1) deseni), `atr` 1.82s / `adx` 1.46s / `rsi` 0.71s — hepsi her
barda TÜM dilim üzerinde ve AYNI çağrı içinde 2-3 kez tekrar hesaplanıyordu
(vol-filtre + stop/target + kalite skoru + rejim filtresi ayrı ayrı ATR/ADX/RSI çağırır).

### Yapılan (üç kontrollü değişiklik — tümü değer-eşdeğer)
1. **`_true_range` numpy'a indirildi** (`strategy.py`): `pd.concat([...]).max(axis=1)` →
   `np.fmax` zinciri. fmax NaN-atlar = eski skipna semantiği; ilk barda shift-NaN → h-l.
   Bit-bit aynı değerler (regresyon testi eski implementasyonu referans alarak doğrular).
2. **`generate_signal` içi tekrar-hesap eliminasyonu**: ATR(14) serisi çağrı başına BİR kez;
   kapı bloğunda hesaplanan ADX/RSI/EMA50 son değerleri saklanıp kalite skoru + HTF hizası +
   rejim filtresine aktarılır. `volatility_explosion_filter`/`regime_filter`/
   `signal_quality_score`/`compute_stops_targets` GERİYE-UYUMLU opsiyonel parametre aldı
   (`atr_series`/`adx_val`/`rsi_val`/`atr_val`; None = eski davranış, kendisi hesaplar).
3. **Backtest döngüsü bar-başı pandas yükü** (`backtest.py`): OHLC + zaman damgaları
   döngüden önce BİR kez numpy/list'e çıkarılır; her barda `df.iloc[i+1]` Series kurulumu
   ve tekrarlı `to_pydatetime()` elendi (datetime-dışı indeks için eski yol korunur).

### Ölçülen sonuç (aynı sentetik iş yükü, önce/sonra)
- 2000-bar tek-sembol backtest: **5.27s → 2.20s (2.4×)**; 4000-bar: 6.2k bar/s.
- 36-anahtarlı altın-referans anlık görüntüsü (3 seed × 10 strateji modu × 300 bar sinyal
  dizisi + 2 modda tam backtest, MC dahil): üretimi 84.4s → 46.5s (**1.8×**) ve içerik
  **BİREBİR AYNI ✔** (aksiyon/fiyat/stop/target/skor/RR/reasons + PnL/equity/MC repr-düzeyi).
- Kazanım canlı tarafta da geçerli: canlı tarama da her sembol/barda `generate_signal`
  çağırır → tarama turu başına CPU aynı oranda düşer.

### BULGU-13 (DÜŞÜK, tip güvenliği): Tur-5 sonrası eklenen dosyalarda 3 mypy hatası
`alpaca_history.py` + `chart_analyst.py` (yeni modüller) mypy'de 3 hata veriyordu:
2× `str|None`-üzerinde-`.lower()` kalıbı + 1 kullanılmayan `type: ignore`. Düzeltme:
`(x or os.getenv(K) or "default")` biçimine yeniden yapılandırma (boş-string env
dejenere durumunda ayrıca daha sağlam) + bayat ignore silindi. `mypy nihai` → 56 dosya, 0 hata.

### Kalıcı guard: `tests/test_strategy_perf_parity.py` (+16 test)
(a) numpy `_true_range` == eski concat/max referansı (rastgele + NaN-satırlı + tek-bar +
boş df), (b) 4 fonksiyonun önceden-hesaplanmış-parametre pariteleri, (c) 9 strateji
modunda `generate_signal` determinizmi (dilim-tekrarında aynı çıktı — memo sızıntısı yok).
→ Gelecek bir "optimizasyon" değerleri kaydırırsa test kırılır.

**Doğrulama:** nihai **442** test (426+16), MainHub 27; ruff + mypy (56 dosya) iki projede
temiz. Not: canlı servisler (hub 8799 / predict 8765) hâlâ KAPALI — Tur-4 operasyon
tespiti sürüyor, başlatma kullanıcı kararı.

---

## Ek Tur 7 — Paylaşımlı gösterge önbelleği: O(n²)→O(n) (davranış bit-bit korunarak) — 2026-07-03

Baseline: nihai 442 / MainHub 27 yeşil, ruff+mypy temiz. Tur-6 tek-strateji yolunu
hızlandırmıştı; ensemble yolu hâlâ 27.6s idi (2000 bar) — rsi 3408, adx 3506 çağrı:
mean_reversion + donchian + rsi_divergence her barda AYNI göstergeleri ayrı ayrı,
ÜSTELİK her seferinde tüm dilim üzerinde hesaplıyordu.

### Yapısal fikir: nedensellik → tam-df ön-hesap + bar-başına kırpılmış görünüm
Buradaki tüm göstergeler NEDENSELDİR (ewm adjust=False rekürsiyonu + rolling pencereler
yalnız geçmişe bakar) → tam df üzerinde BİR kez hesaplanan serinin ilk n bara kırpılmışı,
`df.iloc[:n]` üzerinde sıfırdan hesaplananla **bit-bit aynıdır**. Bu değişmez artık
`test_indicator_prefix_identity` ile (check_exact=True) kilitli — kırılırsa backtest
sinyalleri canlıdan sapardı, test buna izin vermez.

### Yapılan
1. **`_SharedIndicators` + `_SharedSlice`** (`strategy.py`): tembel-memoize ema/rsi14/
   atr14/adx/bb20/bb_width20/macd erişimcileri + `upto(n)` kırpılmış görünümü.
2. **8 fiyat-only detektör + `detect_setup` + iki ensemble yolu + `generate_signal`**
   geriye-uyumlu opsiyonel `shared` paramı aldı (None = eski davranış; dış çağıranlar
   etkilenmez). Ensemble artık 6 detektör arasında RSI/ADX/BB/MACD'yi BİR kez hesaplar;
   generate_signal kapıları da aynı önbelleği kullanır (Tur-6 yerel değişkenleri bununla
   sadeleşti).
3. **`backtest_symbol` + `backtest_portfolio`**: sembol başına `_SharedIndicators(df)`
   kurulur, her bara `upto(i+1)` verilir → gösterge maliyeti bar başına O(n)'den O(1)'e
   (toplam O(n²)→O(n)). Portföy döngüsünde ayrıca ts→konum haritası (yinelenen-ts atlama
   davranışı get_loc ile birebir korunarak) + OHLC/timestamp ön-çıkarımı.

### Ölçülen sonuç (aynı sentetik iş yükü, 2000 bar)
- **ensemble: 27.59s → 2.34s (11.8×)** · event_breakout: 2.20s → 0.37s
  (Tur-6 öncesine göre birleşik **14×**).
- 36-anahtarlı altın-referans: üretim 46.5s → 22.7s ve içerik **BİREBİR AYNI ✔**
  (3 kez doğrulandı: detektör edit'leri, backtest edit'leri ve tip-düzeltmeleri sonrası).
- Canlı tarama `shared=None` yoluyla çağrı-içi paylaşımdan kazanır (ensemble taramada
  detektör-arası tekrar-hesap elendi).

### Kalıcı guard'lar (+15 test → parite dosyası 31)
`test_indicator_prefix_identity` (7 gösterge ailesi × 3 n, bit-bit) ·
`test_detectors_shared_upto_parity` (7 detektör × 3 n: shared'sız == çağrı-içi ==
upto görünümü) · `test_generate_signal_shared_upto_parity` (9 mod × 3 dilim, tüm
sinyal alanları). Ayrıca mypy için `_ENSEMBLE_DETECTORS`/`single` Callable anotasyonu.

**Doğrulama:** nihai **457** test (442+15), MainHub 27; ruff + mypy (56 dosya) iki
projede temiz. Servisler hâlâ kapalı (kullanıcı kararı bekliyor).

---

## Ek Tur 8 — SIP PIT dataset dürüst değerlendirmesi + veri-hijyeni ve tip düzeltmeleri — 2026-07-03

Baseline: paralel oturumun Tur-7'si sonrası nihai 453 / MainHub 27 yeşil, ruff temiz
(not: Tur-7 özetindeki "457" sayımı fiilen 453'tü; mypy'de de 1 hata vardı — aşağıda).

### AI değerlendirmesi — yeni alpaca_hist SIP PIT dataset'i ölçüldü: kenar YOK (EDGE_FINDINGS §20)
Tur-7'nin kurduğu ama henüz hiç değerlendirmediği SIP-kalitesi dataset
(`meta_dataset_pit_sp500_event_breakout_1d_2021-07-01_alpaca_hist.csv`: 12.268 işlem,
505 sembol, 80 removed-since, NaN yok) train_meta (time+symbol) + purged 5-fold CV'den
geçirildi: logistic ORT OOS-AUC **0.487**, gbm **0.503** → her ikisi de KENAR YOK.
Tek-bölmede GBM'in gösterdiği "kenar" 7-49 işlemlik örneklemlerdeydi (overfit deseni).
§16'nın hükmü bağımsız, daha kaliteli veri kaynağında doğrulandı → "belki yfinance veri
kalitesiydi" hipotezi ELENDİ. META_GATE varsayılan-KAPALI kalır.

### BULGU-14 (ORTA, veri hijyeni): alpaca_history geçici hatayı "veri yok" diye negatif-cache'liyordu
`_fetch` hem gerçek veri-yokluğunda hem ağ/rate-limit HATASINDA boş df döndürüyor,
`get_history` ikisini de günlük negatif-cache'e yazıyordu → toplu indirme sırasında geçici
bir hata, sembolü o günkü TÜM koşulardan düşürürdü (dataset'te sessiz sembol kaybı).
**Düzeltme:** `_fetch` artık hata/istemci-yok durumunda `None` döndürür ve None asla
cache'lenmez (sonraki çağrı yeniden dener); gerçek boş sonuç için negatif-cache davranışı
korundu. +2 regresyon testi (`test_get_history_hata_cachelenmez` — cache dosyası oluşmaz +
ikinci çağrı yeniden dener; `use_cache=False` yolu).

### BULGU-15 (DÜŞÜK, tip güvenliği): chart_analyst gemini `contents` mypy hatası
Tur-6 "mypy temiz" doğrulamasından sonraki düzenlemelerde `contents=[Part, str]` listesi
`list[object]`'e çıkarsanıp mypy'de arg-type hatası veriyordu. **Düzeltme:**
`types.Content(role="user", parts=[Part.from_bytes(...), Part.from_text(...)])` sarmalayıcı
(çalışma zamanı davranışı birebir; google-genai'nin birincil kabul tipi). Ek doğrulama:
Anthropic yolu güncel API referansıyla karşılaştırıldı — `output_config.format=json_schema`
+ `claude-haiku-4-5` DOĞRU/güncel kullanım, değişiklik gerekmedi.

**Doğrulama:** nihai **455** test (453+2) + 1 skip, MainHub 27; ruff temiz; mypy nihai
**56 dosya 0 hata**. Servisler (hub 8799 / predict 8765) hâlâ KAPALI — kullanıcı kararı.

---

## Ek Tur 5 — Alpaca 10y saatlik PIT dataset + eğitim + kenar hükmü — 2026-07-03→07

### Yapılan (kronoloji)
1. **Alpaca kapsama sondası** (`research/alpaca_coverage_probe.py`): watchlist %99.8,
   removed-since (delist) **%100**, saatlik derinlik %100 → offline PIT eğitimi fizibıl.
2. **10y saatlik PIT dataset** (`build_meta_dataset --universe pit_sp500 --as-of 2016-01-01
   --interval 1h --provider alpaca_hist --yearly-windows`): 504 sembol, **97.143 işlem**,
   2016-2026, removed-since %27, 0 NaN. Bar cache ~650MB (`data/alpaca_hist/`).
3. **Eğitim + değerlendirme:** purged 5-fold CV + train_meta all/both. **VERDİKT: kenar YOK**
   (logistic/gbm OOS-AUC ~0.51, kapı base'i geçmiyor). Detay: EDGE_FINDINGS §21.

### Hata düzeltmeleri (bu tur ortaya çıkan)
- **Kill-switch sansürü (YÜKSEK):** Madde 20 + bileşik-iflas çok-yıllık koşuda örneklemi
  2016'ya sıkıştırıyordu (%95). `backtest_symbol(disable_kill_switch=)` +
  `build_meta_dataset --yearly-windows` (yıllık taze kasa). Canlı davranış değişmedi.
- **Cache tazelik (ORTA):** `_read_cache` yalnız "bugün yazılmış" cache'i geçerli sayıyordu
  → gece yarısı tarih dönünce tüm cache bayatladı, üretim ağa döndü + 429 fırtınası +
  sembol atlama. `ALPACA_HIST_CACHE_MAX_AGE_DAYS` (varsayılan 1 = eski davranış).
- **--save disiplin açığı (YÜKSEK):** train_meta 6-işlemlik "çift-bölme kenarını" dayanıklı
  sanıp modeli KAYDETTİ (purged CV kenar YOK derken). `--min-gate-keep` (varsayılan 50) +
  yanlış kayıt geri alındı + regresyon testi (`test_train_meta_save_discipline.py`).
- **xgboost eksik:** venv'de yoktu (train_meta --model all çöküyordu) → kuruldu +
  requirements.txt'e eklendi.

### Ek evren (S&P-dışı ABD) — veri HAZIR, dataset BEKLİYOR
`build_extra_universe.py`: PIT-S&P500+watchlist dışı 8.756 ABD sembolü (likidite-sıralı +
delist serpiştirme). 6 işçi 05:30'a kadar çekti → **~1.422 ek sembol cache'te**. Ama 05:30
orkestratörü makine uykusunda öldü → ek dataset HENÜZ KURULMADI (veri kayıp değil, cache'te).
Kurmak: `python -m research._build_extra_dataset` (survivorship-şişkin uyarısıyla).

### Not (tutarlılık, ertelendi)
`live_retrain.assess()` best_gate kontrolünde de min-kept şartı yok (train_meta ile aynı
açık); canlı yol purged-CV durable şartıyla kısmen korunuyor ama tutarlılık için
`--min-gate-keep` oraya da taşınmalı. Küçük borç.

**Test:** yeni +9 (save disiplini 4 + coverage probe akışı); ilgili suite yeşil.

### Ek evren (S&P-dışı ABD) dataset — KURULDU + EĞİTİLDİ (2026-07-07)
`_build_extra_slice.py` (6 paralel dilim) → `meta_dataset_extra_us_event_breakout_1h_2016-01-01.csv`:
**234.173 işlem, 1309 sembol**, 2016-2026 dengeli, delist %11, 0 NaN. Eğitim (purged CV +
train_meta all/both): **KENAR YOK** (logistic OOS-AUC 0.518, gbm 0.526; kapı 1/3 ve 1/1 fold,
tutma %0). train_meta'da gbm/xgb symbol-split PF 8.28/2.26 göründü ama gbm time 5-işlem,
xgb time kenar-yok → tek-bölme; --min-gate-keep disiplini modeli REDDETTİ (doğru). Bu set
PIT-üyelik temelli DEĞİL (survivorship-şişkin) → yine de kenar yok = sonuç daha da güçlü.
META_GATE kapalı kaldı, aktif model değişmedi.

---

## Ek Tur 9 — 4-agent paralel denetim + look-ahead düzeltmesi — 2026-07-09

Baseline: nihai 455+1skip / MainHub 27 yeşil. 4 salt-okunur Opus 4.8 agent (correctness / ML /
execution / perf) örtüşmeyen alanları denetledi → 30+ önceliklendirilmiş bulgu
`DENETIM_BULGULARI_2026-07-09.md`'ye yazıldı (çakışma-güvenli partisyon; hafızadaki paralel-düzenleme dersi).

### BULGU-16 (YÜKSEK, look-ahead — UYGULANDI + DOĞRULANDI): stream forming-bar tz-uyuşmazlığı
`alpaca_stream_provider.py` `get_bars` (nihai + MainHub parity kopyası) oluşan-bar atma kontrolü
naive-LOCAL `datetime.now()`'ı naive-UTC bar ts ile kıyaslıyordu → UTC+3'te (kullanıcının TZ'si) yarı-dolu
mevcut-interval barı ATILMIYOR, stratejiye "kapanmış son bar" gibi veriliyordu (Madde 3 ihlali, kararsız
canlı sinyal; 1h/15m/5m/1m). Düzeltme: `datetime.now(UTC).replace(tzinfo=None)` — İKİ kopyaya BİREBİR
(parity korundu). Doğrulama: tam suite **exit=0** (regresyon yok), stream_parity yeşil.
Kalan iş: A1 için tz-bağımsızlığını kilitleyen regresyon testi (ledger'da işaretli).

### Sonuç — TAMAMLANDI (2026-07-10): 3 paralel Opus Fix agent + tek-thread entegrasyon
Limit sıfırlanınca partisyonlu 3 Fix agent (data-infra / ML / execution, örtüşmeyen dosya kümeleri) tüm
Tier-1/2 + seçili Tier-3 bulgularını uyguladı (EXEC bir kez infra-stall yaşadı → odaklı yeniden koşuldu).

**Uygulanan:** A1 (önceden) + A2/A3/A4/A5 · B1/B2/B3/B4/B6/B7/B8/B9 + Wilson-CI dürüstlük kapısı ·
C1/C2/C4/C5/C6 + C3 konvansiyon-kilidi · D1/D5/D6/D7 perf (bit-exact). Her fix + regresyon/kapsama testli.

**Entegrasyon (tek-thread):** nihai tam suite **exit=0** (~520 test; 455+1skip → +~65 yeni test) · `mypy nihai`
**56 dosya 0 hata** · `ruff` iki projede **temiz** (+1 önceden-var B007 `_prefetch_slice.py` düzeltildi) ·
MainHub **27 test yeşil**. Kod-dışı yedek: `scratchpad/src_backup_20260709`.

**Dürüst kalıntılar:** B5 train_meta aslında bug DEĞİLmiş (yalnız live_retrain yolu düzeltildi; terfi modeli
artık time-train ~%60 alt-küme = model↔eşik tutarlılığının bilinçli bedeli, kenar yok → doğruluk öncelikli) ·
B7 `build_pead_dataset._enrich_row` finite-guard kapsam-dışıydı (prepare() sertleştirmesi NaN yolunu kapatır;
takip) · B2 purged sol-purge OOS-AUC'yi hafif DÜŞÜRÜR (daha SIKI, doğru; hiçbir verdikti çevirmez) · D6 yalnız
hub/alpaca dalı · D4/D9 web-perf ertelendi. META_GATE varsayılan-KAPALI korundu.
Tam bulgu listesi + öncelik: `DENETIM_BULGULARI_2026-07-09.md`.

---

## Ek Tur 10 — Kalan perf kalemleri + 4-bölümlük geliştirme raporu — 2026-07-10 (Fable 5)

Baseline: Tur-9 kapanışı (~520 test yeşil, mypy 56/0, ruff temiz). Bu tur Tur-9'un erteledigi/
takibe aldığı perf kalemlerini kapattı ve kullanıcının istediği 4-bölümlük raporu üretti:
**`GELISTIRME_RAPORU_2026-07-10.md`** (AI yol haritası Fable5/Opus4.8 · eksik taraması+çözümler ·
öncelikli düzeltme listesi · quant seviye değerlendirmesi — aşamalı uygulanacak).

### Uygulanan (hepsi bit-exact; kalıcı parite: `tests/test_perf_parity_round10.py`, +8 test)
- **D2** `train_meta._auc` + `forward_harness._auc`: saf-Python rank döngüsü → cumsum vektörizasyonu.
  234k-satır extra-US dataset'te fold başına ~6M Python iterasyonu numpy'a indi (purged_cv/forward hızlanır).
- **D3** `purged_cv.permutation_importance`: özellik başına tam `X.copy()` (~47MB × 25 özellik/fold
  alloc churn) → tek sütun yerinde permüte + `finally` ile geri yükleme. rng tüketimi aynı → bit-exact;
  çağıranın X'i istisna yolunda bile korunur (test kanıtlı).
- **D4** `prediction_tracker._read_log_rows()` + `report(rows=)`/`coverage(rows=)` paylaşımı:
  web `/api/forward-test` 2.4MB `predictions_log.csv`'yi istek başına 2-3 kez değil 1 kez okuyor.
  `rows=None` (CLI) yolu birebir eski davranış.
- **D8** `agreement._map_mom_pct` (yeni yardımcı): işlem başına iterrows+searchsorted → TEK vektörize
  searchsorted + konum lookup'ı (sembol-yok / geçmiş-yok → NaN kuralları birebir).
- **D9** `web_dashboard._bars/_series/_hist_payload`: iterrows/items (satır başına Series kurulumu)
  → toplu epoch + sütun listeleri; Python round / NaN-atlama / renk kuralları birebir.
  DERS: `DatetimeIndex.asi8` birim-BAĞIMLI (pandas us-çözünürlük üretebiliyor) — parite testi yakaladı
  → `as_unit("ns")` sabitlemesi. "Vektörize ederken epoch'u elle hesaplama" tuzağına kalıcı guard.
- **D10** `train_meta`: Xtr/ytr numpy dönüşümü model döngüsünün DIŞINA (--model both/all'da 2-3× tekrar).
- **B7 takibi** `build_pead_dataset._enrich_row`: surprise'a KAYNAK finite-guard (NaN/inf/duplike-tarih
  Series → satır güvenli tarafta düşer; `prepare()` savunmasına ek katman, davranış bilinçli değişti).

### Entegrasyon
nihai tam suite **exit=0** (~528 test; +8 parite) · `mypy nihai` **56 dosya 0 hata** · `ruff` temiz.
MainHub'a dokunulmadı (Tur-9 sabahı 27 yeşil). Kenar iddiası YOK; META_GATE varsayılan-KAPALI korundu.

### Operasyon tespiti (kod değil — kullanıcı aksiyonu, rapor §2/§3)
Forward-harness bugün 06:00 koştu: **20.010 çözülmüş tahmin, isabet %47.4, tüm AUC ≤0.5, gate_sim
GATE_HURTS, VERDİKT NO_EDGE** (offline bulgularla tutarlı). Hub KAPALI olduğundan resolve 0/305
(yfinance 1m fallback'i de boş döndü — QCOM/CRWD/LMT). Canlı servisler 7 Temmuz 20:06'dan beri kapalı;
watchdog görevleri hâlâ kayıtsız (yalnız ForwardHarness-Daily kayıtlı). Git: son commit 2026-06-26,
**101 dosya commitsiz** (Tur 4-10 emeği tek kopyada) → rapor P0 maddesi.

---

## Ek Tur 11 — 4 madde bir arada: AI geliştirme + denetim + düzeltme + optimizasyon — 2026-07-13 (Fable 5)

Baseline: Tur-10 kapanışı (~528 test, mypy 56/0, ruff temiz; servisler bu turun başında zaten
watchdog'larla ayaktaydı — hub 8799 + kayıtçı 8765). Kullanıcının 4 maddesi (AI geliştir /
eksik tara-düzelt / hata düzelt / optimize et) TEK turda, 2 paralel salt-okunur denetim +
2 paralel uygulama agent'ı + koordinatörün doğrudan düzeltmeleriyle işlendi.

### Yöntem
Tur-9 deseni tekrarlandı (partisyonlu domain + tek-thread entegrasyon), bu kez denetim ve
uygulama iki AYRI dalga: (1) **Denetim-A** (research/ML + prediction_tracker + veri katmanı,
13 dosya satır satır) ve **Denetim-B** (ps1'ler + MainHub + web_dashboard) paralel salt-okunur
koştu → 15 bulgu (7+8); (2) bulgular dosya-sahipliğine göre bölündü: **Fix-CORE** (nihai/config.py,
strategy.py, meta_gate.py docstring, YENİ daily_summary.py, web_dashboard.html/.py VaR kartı) ve
**Fix-ML** (research/train_meta.py, purged_cv.py, live_retrain.py) paralel uyguladı; kalan 8 bulgu
(alpaca_history, prediction_tracker dedup+DictWriter, tüm ps1'ler, MainHub data_provider.py,
web_dashboard.py update-config, forward_harness.py) hiçbir agent'a ait olmadığından koordinatör
(ben) doğrudan uyguladı. İki agent turu da altyapı-kaynaklı "stall" (600s ilerlemesizlik) yaşadı
(Tur-9'da da görülen bilinen desen) — `SendMessage` ile transkriptten devam ettirildi, iş kaybı olmadı.

### BÖLÜM 1 — "AI'ı geliştir" (FAZ AI-3 P3 + FAZ AI-4 + META_GATE mimarisi)

**META_GATE üç kipli oldu (off/shadow/enforce), META_GATE=shadow YENİ:** Kapı canlı gate_sim'de
GATE_HURTS derken tamamen kapatmak kanıt döngüsünü (prediction_tracker pwin logu → gate_sim/
pwin_calibration) öldürüyordu — "kapıyı kapat" ile "kanıt biriktirmeye devam et" çelişiyordu.
Gölge kip çözer: `p_win` hesaplanır + `meta_p_win`/`meta_shadow_gated` metadata'sına + reason'a
yazılır, sinyal (aksiyon/stop/hedef/skor) ASLA bastırılmaz. `true`→enforce (eski davranış,
regresyon testiyle kilitli), boş/geçersiz→off. `nihai/config.py` `_meta_gate_mode()` + geriye-uyum
`enable_meta_gate` property'si (field DEĞİL — `mode!="off"` türetimi, tek doğruluk kaynağı).

**FAZ AI-3 "DÜRÜST İYİLEŞTİRMELER" backlog'u TAMAMEN KAPANDI** (GELİŞTİRME_RAPORU P3, Tur-9'dan beri
açıktı):
- `train_meta.evaluate()`: eşik seçimi (0.45-0.60 taraması) artık TEST'ten **TRAIN**'e taşındı —
  TRAIN'de kenar gösteren eşik seçilir, yalnız O eşik TEST'te doğrulanır (çoklu-test/bakma-önyargısı
  kapatıldı). 4-eşikli tablo şeffaflık için basılmaya devam ediyor, seçilen satır işaretli. Dönüş
  `thr_source`/`n_thr_tried`/`ece_test` kazandı; `main()` "denenen konfig" sayısını raporluyor.
- `purged_cv.py`: **null-AUC tabanı** (`--null-reps`, varsayılan 1) — etiket-karıştırma ile fold
  OOS-AUC'nin şans altındaki dağılımı ölçülür; DAYANIKLI verdikt artık gözlenen AUC'nin null'dan
  AYRIŞMASINI da şart koşuyor. **Sızıntı bayrağı**: permütasyon-öneminde tek özellik AUC'yi
  tek başına ≥0.08 taşıyorsa "SIZINTI ŞÜPHESİ" uyarısı. `live_retrain._purged_gate` aynı null
  tabanını (1 tekrar) kullanıyor + `durable` artık ayrışma şartlı.
- `train_meta.save_model()`: **provenance** (code_sha, dataset_sha256, n_rows/features, seed,
  train_frac, min_gate_keep, threshold_selection, n_configs_tried, calibrate, ece_test, saved_at)
  hem `.json` hem `.pkl`+`meta_model_info.json` yollarına gömülüyor; `meta_gate._load` bilinmeyen
  anahtarı yok sayarak sözleşmeyi bozmadan yüklüyor (regresyon testli).
- **Ek 3 bulgu** (koordinatörün Denetim-A'dan Fix-ML'e ilettiği): `live_retrain._purged_gate` artık
  SABİT 0.45 yerine terfi edilecek eşikte CV'yi sınıyor (`assess()` `best_gate[0]`'dan türetiyor) —
  DAYANIKLI verdikt artık gerçekten deploy edilecek eşikte doğrulanmış oluyor; `suggest_purge` global
  ortalama yerine yerel-yoğunluk duyarlı (`max(ortalama, q95(bar-başı-satır), n/bar-sayısı)`) — kümeli
  (earnings-sezonu gibi) veride az-purge riskini kapattı; `entry_time` üç noktada (split_time, purged_cv,
  live_retrain._span_days) artık `pd.to_datetime(utc=True)` ile kronolojik sıralanıyor — karışık
  naive/aware format artık sıralamayı bozmuyor / `_span_days`'i sebepsiz sıfırlamıyor.

**FAZ AI-4 (LLM operatör katmanı, DÜŞÜK-risk hemen-değer) — YENİ `nihai/notifier/daily_summary.py`:**
history.csv son 2 satırından verdikt AYNEN + çözülen-delta, servis sağlığı (hub/kayıtçı socket),
bugünkü live_features satırı → 5-8 satır Türkçe özet. `prediction_tracker` İMPORT ETMEZ (execution
bağımlılığı yok). `--send` Telegram, `--llm` opsiyonel akıcılaştırma (her hatada deterministik metne
düşer). `forward-harness.ps1`'e günlük çağrı olarak bağlandı (bkz. BÖLÜM 3 — sıralama düzeltmesiyle
birlikte, harness başarısız olsa BİLE çalışır).

**FAZ AI-1 (ölçülmemiş AI adaylarını ölç) — llm_event_test denendi, TAMAMLANAMADI:** PIT modunda
(`--pit --as-of 2022-01-01 --limit 30`) fiyat tarafı önce toplu yfinance ile bot-korumasına takıldı
→ `DATA_PROVIDER=alpaca_hist` ile çözüldü (SIP, delist'ler dahil). LLM tarafı ise Gemini ÜCRETSİZ
katmanın günlük ~20 istek kotasına (429 RESOURCE_EXHAUSTED) birkaç dakikada çarptı — 30 sembollik PIT
taraması bitmeden durdu (tek tamamlanan filing: coverage 1/30 sembol, işlem=1, dürüst-küçük-örneklem
YORUM eşiğinin altında, hüküm YOK). Kod sorunu değil, kota. `llm_event_test.py` docstring'ine saha
notu + `DATA_PROVIDER=alpaca_hist` örnek komutu eklendi. Gerçek ölçüm için ANTHROPIC_API_KEY (Haiku)
veya Gemini ücretli katman gerekiyor — FAZ AI-2 kararına benzer bir kullanıcı kararı.

### BÖLÜM 2+3 — Eksik taraması + hata/eksik düzeltmeleri (denetim bulguları, 15 madde)

**Veri bütünlüğü (YÜKSEK/CONFIRMED):**
- `nihai/data/alpaca_history.py` `_read_cache`: boş-negatif cache artık yalnız KAPSADIĞI pencereye
  (yeni `.empty.json` meta dosyası, `{"start": iso}`) servis ediliyor. Eskiden `start=2022`'de boş
  çıkan bir sembol, `start=2016` isteyen PIT koşusuna da boş verilir, delist (removed-since) sembolleri
  sessizce dataset'ten düşerdi — tam da survivorship-temizliğin bozulduğu yön. +1 regresyon testi
  (geniş/dar pencere karışık senaryo + eski-format metasız cache geriye-uyumu).
- `nihai/execution/prediction_tracker.py` dedup: `_existing_keys`/`record()` artık HAM `bar_ts` string'i
  değil `_canon_ts` (C1'in fiyat-join'inde zaten kullandığı kanonikleştirme) ile anahtarlanıyor. Canlı
  sağlayıcı failover'ı (hub↔yfinance, doğrulanmış kod yolu) aynı barı naive-UTC/tz-aware farklı
  gösterimle döndürebilir → eskiden aynı bar İKİ kez kaydedilip isabet/AUC istatistiklerini şişirebilirdi.
  +1 regresyon testi (karışık format aynı bara indirgeniyor).
- `resolve()` yeniden-yazımı `csv.DictWriter(..., extrasaction="ignore")` aldı — FIELDS-dışı legacy
  kolonlu eski dosyada artık `ValueError` ile çökmüyor (record yoluyla aynı sözleşme).
- **Canlı olay + fix:** entegrasyon sırasında `logs/forward-harness.log`'da 2026-07-11 02:04 çöküşü
  bulundu — `os.replace(tmp, predictions_log.csv)` OneDrive senkron kilidiyle `PermissionError`
  fırlatıp TÜM harness'i (resolve+snapshot+rapor) düşürmüş (O5'in belgelenmiş riski gerçekleşti; o
  gün hiçbir tahmin çözülmedi, AI-0'ın "3 gün üst üste" kabulü hâlâ karşılanmıyor). Yeni
  `_atomic_replace()` (kısa üstel geri-çekilmeli yeniden deneme, ~6.3sn tavan, kalıcı hatada YÜKSELİR)
  üç yazım noktasında da (`_ensure_schema`, `_write_features`, `resolve`) kullanılıyor. +2 regresyon
  testi (geçici kilit → yeniden dener / kalıcı kilit → istisna yükselir).
- MainHub `data_provider.py` `HubProvider.get_option_chain`: eksik `all_expiries` alanı eklendi (A2
  paritesi — hub sunucusu zaten dönüyordu, istemci düşürüyordu; bugün etkisiz çünkü MainHub kendi
  hub'ına bağlanmıyor, ama kopya referans alınırsa opsiyon vade listesi sessizce boş kalırdı).

**Performans:**
- `research/forward_harness.py` `evaluate()`: `predictions_log.csv` istek/koşu başına 3× (coverage+
  report+kendi süzgeci) yerine D4'ün paylaşım API'siyle 1× okunuyor.

**Operasyon/PowerShell (ORTA, güvenilirlik):**
- `stop-hub.ps1`: artık `NihaiAlgo-HubWatchdog-SYSTEM` görevini de durdurup DEVRE DIŞI bırakıyor —
  eskiden yalnız bayrak dosyasını silip süreci öldürüyordu, SYSTEM görevi kayıtlıyken hub ~30sn içinde
  OTOMATİK geri geliyordu (durdurma vaadi fiilen boştu).
- `hub-watchdog.ps1` + `predict-watchdog.ps1`: `Start-Hub`/`Start-Recorder` artık yeniden başlatmadan
  ÖNCE port-kapalı-ama-süreç-YAŞIYOR durumunu (asılı/yavaş açılış) tespit edip eski süreci öldürüyor —
  eskiden her 30sn'lik döngüde bir yeni python süreci EKLENEBİLİRDİ (süreç birikimi riski).
  `Limit-LogSize`'daki sessiz `catch {}` artık `Write-WdLog` ile uyarı basıyor (truncate başarısızlığı
  artık görünür).
- `forward-harness.ps1`: `daily_summary` çağrısı artık harness'in çıkış koduna BAĞLI DEĞİL — eskiden
  `exit $code` erken dönüşü özet çağrısını tamamen atlıyordu, tam da 2026-07-11 gibi bir çöküş gününde
  operatör özeti/Telegram uyarısını GÖRMÜYORDU. Script'in nihai çıkış kodu yine harness'inki (Görev
  Zamanlayıcı doğru başarısızlık görsün diye).
- MainHub `start-hub.ps1`/`stop-hub.ps1` + `hub-watchdog.ps1`: eksik UTF-8 BOM eklendi (dünkü
  "BOM'suz+em-dash=PS5.1 parse bombası" dersinin MainHub kopyasında hâlâ açık olduğu görüldü;
  bugün yorum satırında olduğundan patlamadı ama risk kapatıldı). `MainHub/run-hub.ps1`'e, nihai
  kopyasından FARKLI (sonsuz auto-restart) davranışını açıklayan uyarı notu eklendi.

**Web/güvenlik:**
- `nihai/web/web_dashboard.py` `/api/update-config`: `.env` yazımı artık atomik (tmp+`os.replace`,
  `_atomic_replace` DEĞİL — ayrı, nadir-yazılan dosya) ve TÜM işlem `PORTFOLIO_LOCK` altında; eskiden
  read-modify-write kilitsizdi, eşzamanlı iki istek birbirinin anahtarını ezebilir veya kesintide
  dosya yarım kalabilirdi.

**Bilinçli ERTELENDİ (dokümante edildi, düzeltilMEDİ):**
- `research/train_pead_ml.py prepare()`: winsorize sınırları + `cap_tier` rank'i split'ten ÖNCE tüm
  df'te fit ediliyor → hafif transdüktif sızıntı (etiket sızıntısı DEĞİL). Mevcut verdikt zaten
  "kenar YOK" (§19) — sızıntı OOS-AUC'yi olduğundan YÜKSEK gösterir, yani hüküm sızıntıya RAĞMEN
  duruyor (güvende); düzeltme kenarı asla "var" yapamaz, yalnız daha muhafazakâr yapar. Doğru fix
  `prepare()`'i fit/apply ikilisine ayırmayı gerektiren orta-ölçekli refactor → docstring'e not
  düşülüp ertelendi.
- `prediction_tracker.resolve()` `lookback_bars=400` sabiti: uzun kesinti/delist sembolde satır
  SONSUZA dek 'pending' kalabilir (UNRESOLVABLE diye ayrılmıyor). Doğru fix şema değişikliği +
  report()/gate_sim()/coverage()/forward_harness/dashboard güncellemesi ister (canlı 2M+ satırlık
  dosyaya dokunan orta-büyük değişiklik) → docstring'e not düşülüp ertelendi.

### Entegrasyon
nihai tam suite **582 test, exit=0** (528→582; +54: shadow-mode 7, daily_summary 9, threshold-selection 5,
null-baseline 8, leakage-flag 5, provenance 10, alpaca_history 1, prediction_tracker dedup+atomic-replace
4, web VaR 2, +diğer küçük eklemeler) · `mypy nihai` **57 dosya 0 hata** · `ruff` nihai+research+tests
**temiz** · MainHub **27 test yeşil** + ruff temiz. Kenar iddiası YOK; META_GATE `.env`'lerde hâlâ
`shadow`'a GEÇİRİLMEDİ (bilinçli — bu kullanıcı kararı, kod hazır). Detay: `GELISTIRME_RAPORU_2026-07-10.md`
§Ek Tur-11, `DENETIM_BULGULARI_2026-07-09.md` Tur-11 notu.

## Ek Tur 12 — 3 paralel denetim + düzeltme + optimizasyon (Opus 4.8, max-effort) — 2026-07-13

Baseline (YERİNDE doğrulandı, rapora güvenilmedi): nihai 582 + MainHub 27 yeşil, git TEMİZ (4e483be
Tur-11 commit'liydi — "commit atılmadı" notu eskimiş), canlı servisler AYAKTA (predictions_log/
live_features 22:58'de yazılıyor → AI-0 birikiyor), ANTHROPIC_API_KEY BOŞ. Not: yalnız
ForwardHarness-Daily zamanlı görevi kayıtlı — rapordaki SYSTEM hub/predict watchdog'ları LİSTEDE YOK.

Yöntem: kullanıcının 6 maddesi için Tur-9/11 deseni — 3 salt-okunur Opus denetçisi örtüşmeyen
domenlerde (research/ML · execution/data · web/notifier/MainHub) hem eski ledger'ı güncel kodla
KIYASLADI hem yeni bug aradı. Sonuç: DENETIM_BULGULARI'nin B2-B9/A1-A5/C1-C6/D1-D10'unun TAMAMI
GERÇEKTEN kapalı doğrulandı (gövdedeki "BEKLİYOR" etiketleri Tur-9-öncesi snapshot'tı). Koordinatör
onaylı YENİ bulguları doğrudan düzeltti (kenar UYDURMADAN; hepsi çökme-emniyeti/dürüst-ölçüm).

Düzeltilen (commit 8bea48e, +8 regresyon testi):
1. [HIGH] nihai/atomic_io.py YENİ — OneDrive-kilidi dayanıklı atomik yazım TEK kaynağı; state.json
   (~60sn hot-loop, açık pozisyon+equity — EN kritik) + watchlist ham os.replace'i buradan geçiyor.
   2026-07-11 predictions_log'u çökerten kilit AYNEN state.json'da açıktı (Tur-11 yalnız tracker'ı
   sertleştirmişti; StateStore.save her ~60sn ham os.replace + yalnız KeyboardInterrupt guard'ı).
2. [MED] prediction_tracker _existing_keys okuma hatasında None → record() döngüyü ATLAR (kısmi
   dedup kümesiyle zaten-kayıtlı barı YENİ sanıp çift-kaydetme = isabet/AUC şişmesi guard'ı);
   _write_features sessiz kaybı stderr'e görünür.
3. [MED] purged_cv SAĞ tarafı da horizon-purge (B2 yalnız SOL'u yapmıştı; kümeli/çok-sembollü veride
   sağ ham-embargo OOS-AUC şişirip sahte-DAYANIKLI verdikt riski); --null-reps default 1→5.
4. [MED] web_dashboard /api/var + /api/predictions/accuracy async→def (senkron ağ I/O event-loop'u
   dondurmasın; predict-watchdog 5dk'da bir resolve=true çağırıyor → dashboard donuyordu).
5. [LOW] agreement ensure_feature_columns guard (kardeşlerle tutarlı); train_pead_ml chrono_sort.
Doğrulama: nihai 588 yeşil (exit 0) · mypy 58/0 · ruff temiz · MainHub 27 (dokunulmadı).

Bilinçli ERTELENDİ (belgeli; güvenli/büyük/veri-bekler): train_pead_ml prepare() transdüktif sızıntı
(verdikt "kenar yok" — sızıntı yalnız YÜKSEK gösterir, hüküm güvende), prediction_tracker 400-bar
UNRESOLVABLE (şema+tüm tüketiciler), resolve() tam-log yeniden-yazımı (perf), sağlayıcı-failover 1h
bar-sınır hizası (ampirik doğrulama gerek), cross-process resolve kilidi, data_provider CRLF/anahtar paritesi.

AÇIK — KULLANICI aksiyonu (kod değil):
- SYSTEM watchdog'lar HEM kayıtsız HEM mimari-yanlış: register-*.ps1 self-elevate→return (UAC reddi/
  hata SESSİZ yutuluyor) + SYSTEM kullanıcının OneDrive profiline erişemez (watchdog başlangıçta çöker)
  → hub/predict keep-alive fiilen YOK. ÇÖZÜM: ForwardHarness-Daily gibi interactive-current-user
  (-LogonType Interactive) modeli, SYSTEM'i bırak; predict-watchdog'a Start-ScheduledTask+doğrulama ekle.
- ANTHROPIC_API_KEY BOŞ → FAZ AI-1 (llm_event ölçümü) kilitli. FRED/Reddit anahtarı + FAZ AI-2 Sharadar
  (~$50/ay) kararı hâlâ kullanıcıda — kenar sorusunun tek gerçek kilidi (model değil, ACI DERS #5).

### Ek Tur 12 — Batch 2 (kullanici onayli: watchdog ops + ertelenen bug) — 2026-07-14
Kullanici Q sonucu: Sharadar ERTELENDI (aciklandi); siradaki = watchdog ops + ertelenen orta-buyuk bug.
Yapildi (commit e2ff3a3, nihai 589 yesil / mypy 58/0 / ruff temiz):
- WATCHDOG (HIGH): register-watchdog-system.ps1 + register-predict-watchdog.ps1 ForwardHarness desenine
  gore YENIDEN YAZILDI (interactive current-user + -LogonType Interactive + AtLogOn; SYSTEM ve
  UAC-self-elevate KALDIRILDI: SYSTEM kullanicinin OneDrive profiline erisemiyordu + self-elevate->return
  UAC reddini sessizce yutuyordu). Kayit sonrasi Start-ScheduledTask + Get-ScheduledTask dogrulama (yoksa
  LOUD hata). Saf-ASCII (parse-check 0 hata, non-ascii sadece BOM). MainHub/run-hub.ps1 sonsuz-restart ->
  hizli-cakilmada-bail. >>> KULLANICI RE-REGISTER ETMELI: .\register-watchdog-system.ps1 + .\register-predict-watchdog.ps1
- resolve() UNRESOLVABLE (ertelenen bug #1, en operasyonel — canli dongu ETKILENIR): 400-bar penceresi
  disi/delist tahmin _is_aged_out (bar_ts < en-eski cekilen bar) ile terminal UNRESOLVABLE (sonsuza pending
  kalmaz). Tuketiciler dogrulandi (report hit in {0,1} suzuyor; coverage ayri unresolvable sayip pending'den
  dusuyor; build_live_dataset UNRESOLVABLE'i acikca atliyor); +1 regresyon testi. price<=0 da pending sayiliyor.
HALA ERTELENDI (ayri odakli pass gerektirir): resolve() tam-log yeniden-yazimi -> aktif/arsiv split (perf;
TUM okuyuculari -report/coverage/build_live_dataset- etkiler), train_pead_ml.prepare() fit/apply refactor
(GUVENLI + dusuk-aciliyet: yalniz yeni PIT veriyle -FAZ AI-2- onemli, Sharadar karari kullanicida).

---

## Ek Tur 13 — 2026-07-17 (Fable 5 max + 3 Opus agent; agent'lar oturum limitinde yarım kaldı, F5 tamamladı)

**Hedefler (kullanıcı):** (1) eksik bul + geliştir, (2) projedeki AI'ı yükselt — ikisi paralel;
plan/hüküm/entegrasyon F5'te, kodlama Opus agent'larında (Tur-9 deseni).

### OPS (F5 doğrudan)
- **META_GATE=true → shadow (.env):** canlı gate_sim GATE_HURTS derken enforce AÇIKTI (kanıt-ilke
  çelişkisi). shadow = pwin loglanır + kanıt döngüsü yaşar, sinyal bastırılmaz. Canlıya geçiş
  servis devrini bekliyor (aşağıda).
- **Forward-harness 07-14/15/16 boşluğu kök nedeni:** 23:30 tetiğinde makine uykuda (StartWhenAvailable
  uykuda tetiklemez) + 07-16 19:53 catch-up koşusu 30dk ExecutionTimeLimit ile SESSİZCE öldürüldü
  (banner var, history satırı yok). Çare: limit 4 saat + 12:00 öğle yedek tetiği (görevde doğrulandı)
  + forward-harness.ps1'e BITTI işaretleyicisi ve yarım-koşu tespiti.
- **Görev kaydı gerçeği:** bu makinede YENİ scheduled-task oluşturma yükseltisiz REDDEDİLİYOR
  (0x80070005; Daily tetikle de). Mevcut görevi GÜNCELLEME çalışıyor. register-watchdog scriptleri
  AtLogOn→Daily 00:05+StartWhenAvailable'a çevrildi; **tur13-servis-devri.ps1** (YENİ) tek YÖNETİCİ
  adımında: eski SYSTEM görev/süreçleri temizler + interactive kayıtları yapar + doğrular.
- **SYSTEM zombi keşfi:** hub(8799)+kayıtçı(8765)+predict-watchdog 2026-07-10'dan beri SYSTEM
  bağlamında çalışıyor (predict-watchdog 5dk'da bir resolve atıyor — AKTİF). SYSTEM görev+süreçleri
  yükseltisiz oturuma GÖRÜNMEZ → Tur-12'nin "servisler kapalı/görevler kayıtsız" teşhisi muhtemelen
  bu körlüğün artefaktıydı. Zombiler ESKİ kod koşuyor (Tur-11/12 fix'leri + shadow canlıda değil).
- **AI-0 kabulü:** YİNE verilemedi (07-14/15/16 history boş; bugünkü 05:50 koşusu +5138 çözdü).
  İzleme 07-17'den yeniden; öğle tetiği + 4sa limit ile boşluk sınıfı kapatıldı.

### AI KATMANI (FAZ AI-4 genişleme; Opus başladı, F5 bitirdi) — hepsi KARAR-DIŞI
- `nihai/ai/router.py`: GELISTIRME_RAPORU §1.2 tablosunun kodu. Görev profili (parse/summary/
  judgment) → model (env override'lı); sağlayıcı zinciri ANTHROPIC→GEMINI→None; 429/kota →
  6 saat cooldown (süreç-içi + `ai_router_state.json` atomik kalıcılık); anlık hatalar cooldown'a
  SOKULMAZ (yalnız o çağrıda fallback); sağlayıcı başına tek deneme (retry fırtınası yok).
- `nihai/ai/memory.py`: kalıcı JSONL AI hafızası (UTC ts; thread-kilitli append; 5MB→son 2000
  satıra atomik kırpma; bozuk satır atlanır AMA sayısı loglanır).
- `nihai/ai/reasoner.py`: ucuz facts (history son/önceki satır, bekleyen kuyruk, bugünkü canlı-özellik)
  → deterministik Durum/Değişim/Dikkat metni + zorunlu ACI-DERS kapanışı; router varsa YALNIZ ifade
  akıcılaştırma, **DÜRÜSTLÜK GUARDI** ile: LLM çıktısı çözülen/isabet/verdikt/kapanışı AYNEN
  korumazsa ATILIR (test edildi). Her koşuda günün digest'i hafızaya yazılır (yarının deltası).
- `daily_summary`: Değişim/Dikkat bloğu + "AI-0 kabul: son 3 günün X'ünde koşu var" satırı kapanış
  notunun önüne bağlandı (fail-soft); `_llm_polish`'e de aynı dürüstlük guardı eklendi.

### DÜRÜST-ML ÖLÇÜM (Opus başladı, F5 bitirdi) — rapor-only, verdikt makinesine dokunulmadı
- `research/deflated.py` (YENİ): PSR + expected_max_sharpe + DSR (Bailey & López de Prado 2014;
  LdP 2018 Ch.8) + wilson_ci ([0,1] kelepçeli; float artığı düzeltildi) + binom_p_value.
  Raporun §4.1'de "eksik" dediği deflated-Sharpe kalemi kapandı (fonksiyon katmanı).
- `forward_harness` raporuna satır: **İsabet %95 CI + binom p** — canlı veride %47.1,
  CI [%46.6, %47.6], p≈2e-26 → %50'den sapma ANLAMLI ama isabet %50'nin ALTINDA (kenar değil;
  yorum satırı bunu açıkça söylüyor). postmortem PSR/DSR bağlama Tur-14'e bırakıldı.

### DOĞRULAMA
+22 kalıcı test (test_ai_memory 4 · test_ai_router 8 · test_ai_reasoner 5 · test_deflated 5);
nihai TAM suite **exit 0** · mypy **62 dosya / 0 hata** · ruff nihai+research+tests **temiz** ·
düzenlenen 5 ps1 PSParser ile **parse-temiz** · daily_summary + forward_harness gerçek veride
duman-testli. Commitler: 1be7d04 (ara, yeşil iskelet) + bu turun kapanış commiti.

### AÇIK / KULLANICIYA
1. **tur13-servis-devri.ps1'i YÖNETİCİ PowerShell'de çalıştır** (tek komut; watchdog kayıtları +
   zombi→interactive devri + shadow'un canlıya geçişi). Alternatif: yeniden başlatma sonrası
   normal oturumda register scriptlerini koşmak da OLMAZ (görev oluşturma yükseltisiz reddediliyor)
   — yönetici adımı şart.
2. ANTHROPIC_API_KEY hâlâ BOŞ → FAZ AI-1 ölçümü + router'ın anthropic bacağı kilitli (Gemini
   ücretsiz kota ~20/gün ile sınırlı).
3. Tur-12 değişikliklerinin TAM denetimi + postmortem PSR/DSR bağlama → Tur-14.

### DERSLER
- 3 paralel Opus agent aynı anda oturum-limitini tüketebiliyor → kritik yolda ya kademeli başlat
  ya F5-devam planıyla başlat; agent yarım bırakırsa ÖNCE duman testi + tam suite, SONRA karar.
- SYSTEM bağlamındaki görev/süreçler yükseltisiz oturuma görünmez → "servis kapalı" demeden önce
  PORTA bak (Get-NetTCPConnection), sürece/göreve değil.
- Scheduled-task politikası makineye özgü olabilir: oluşturma≠güncelleme izni ayrışıyor.

**Ek Tur 13 kapanışı (18:24):** `tur13-servis-devri.ps1` kullanıcının UAC onayıyla YÖNETİCİ olarak
koşuldu ve DOĞRULANDI: eski SYSTEM görevleri (principal=SYSTEM — gerçekten vardı, yükseltisiz
GÖRÜNMÜYORDU) kaldırıldı, 3 zombi python öldürüldü, interactive görevler Running, hub (PID 28336)
+ kayıtçı (PID 52436) Kagan kullanıcısında 18:24'te GÜNCEL kod + META_GATE=shadow ile başladı;
watchdog logları canlı ("recorder UP OK"). Açık kalanlar yalnız: ANTHROPIC_API_KEY (kullanıcı) +
Tur-14 (Tur-12 tam denetimi, postmortem PSR/DSR bağlama).

---

## Ek Tur 14 — ENTEGRASYON P0 paketi (2026-07-18)

**Kapsam:** ENTEGRASYON_RAPORU_2026-07-17 §4 P0 (4 iş). Kod yazımı bir önceki oturumda (17-07 23:05–23:18,
working-tree'de commitsiz bulundu); bu oturum = Tur-9 dersindeki **tek-thread entegrasyon**: doğrulama,
eksik testlerin tamamlanması, bağımlılık kayıtları, tam suite + commit. Süreç/yazım-zamanı kontrolüyle
paralel-oturum çakışması ELENDİ (Tur-8 dersi uygulandı).

- **P0-1 vollib migrasyonu:** `options_analyzer.py` + `backtest.py` importları `py_vollib`→`vollib`
  (py_vollib artık geçiş paketi); requirements + pyproject yorumları güncel. YENİ `tests/test_greeks_parity.py`
  — golden sabitsiz DEĞİŞMEZ testleri: IV round-trip ≤1e-6, put-call parite (Δc−Δp=1, γ/vega özdeş),
  ATM theta<0, compute_greeks iki dalı (sigma / market_price→IV çözümü).
- **P0-2 tear-sheet:** YENİ `nihai/web/tearsheet.py` (kaynak: data/trades.csv — gerekçeli seçim modül
  docstring'inde; nlv_after ya-hep-ya-hiç equity tabanı; işlemsiz güne yapay 0 yazılmaz; atomic_io yazımı;
  yetersiz-veri Türkçe HTML) + `/api/tearsheet` (6h dosya-cache, ?refresh=1, /api/var deseninde sync def,
  quantstats yoksa 503 degrade). YENİ `tests/test_tearsheet.py` (11 test; GERÇEK quantstats HTML üretimi
  canlı doğrulandı — pandas 2.x uyumu OK).
- **P0-3 ikinci earnings kaynağı (Madde 26):** YENİ `nihai/data/nasdaq_calendar.py` — keysiz api.nasdaq.com
  tarih-bazlı takvim (tarayıcı header ŞART; 15dk tarih-cache tüm izleme listesince paylaşılır; BMO/AMC
  normalize; geçici hata cache'lenmez=BULGU-14). `event_calendar.py`: yfinance→nasdaq fallback zinciri,
  `timing` alanı (additive), her iki kaynak da HATA verirse 24h yerine **15dk negatif cache** (source="hata";
  _MEM'e konmaz). YENİ `tests/test_event_calendar_fallback.py` (tamamı ağsız, 14 test).
- **P0-4 backtest metrikleri:** `sortino_annualized` (empyrical downside konvansiyonu; tüm-pozitifte 0.0
  JSON-güvenli sözleşme) + `profit_factor` (zarar yoksa None — RFC 8259/Infinity gerekçesi) BacktestResult'a
  ADDITIVE (varsayılanlı; eski çağrılar kırılmaz). YENİ `tests/test_backtest_metrics_empyrical.py` —
  bağımsız referansla çapraz kilit: sortino BİREBİR, sharpe'ta bilinen ddof farkı √(n/(n−1)) çarpanıyla
  AÇIKÇA kilitli, max_drawdown ≡ −empyrical×100, ayrışma sözleşmeleri; empyrical yoksa atlanır (test-only dep).

**Doğrulama:** pytest **716 collected, tam suite yeşil** (ilk koşuda `test_tier_exception_isolated`
zamanlama flake'i — 0.15s uyku penceresi, izole + yeniden koşumda yeşil, değişikliklerle ilgisiz; izlemeye
alındı) · ruff temiz · **mypy nihai: 64 dosya, 0 hata** (research'te 12 hata ÖNCEDEN VAR, 4 dosya —
`mypy nihai` kapısının dışında; Tur-15+ adayı). Bağımlılıklar: vollib 1.0.10 (MIT) + quantstats 0.0.81
(Apache-2.0) runtime; empyrical-reloaded 0.5.12 test-only (requirements'ta yorumlu satır).

**AÇIK:** P1 paketi (rapor §4: P1-1 HMM ön-testi [bağımlılıksız, rapor-only] → P1-2 pairs/kointegrasyon
PIT deneyi → P1-3 PBO/CSCV) — dağılım gereği Opus 4.8 ajanlarına bağlam paketiyle; P2 veri-koşullu.

**Ek Tur 14 (devam) — P1 paketi + servis restart (2026-07-18 öğleden sonra):**

- **P1 dağılımı uygulandı** (tasarım/entegrasyon Fable 5, uygulama 3× Opus 4.8 ajanı; kademeli başlatma —
  Tur-13 oturum-limiti dersi): P1-1 + P1-3 paralel, P1-2 onlar bitince (pbo.py'yi kapıda kullanabilsin diye).
- **P1-3 PBO/CSCV** (`research/pbo.py` + 12 test): Bailey ve ark. 2016 CSCV, saf numpy; DSR'nin tamamlayıcısı.
  Ajan bitişte API hatasıyla düştü ama teslimat tamdı (Tur-13 "önce duman testi" dersi uygulandı, kabul).
  İLK gerçek kullanımını P1-2'de yaptı.
- **P1-1 rejim ön-testi** (`research/regime_pretest.py` + `docs/REGIME_PRETEST_2026-07-18.md`): forward log
  yalnız 12 işlem günü + TEK rejim hücresi (UP-HIGH, n=36.544, isabet %47.1) → ayrışma TEST EDİLEMEDİ →
  **HMM şimdilik gündem DIŞI** (şart-1 karşılanmadı); aylar sonra loglar çok-rejimli olunca tekrarla.
  Yan bulgular (Tur-15 denetim adayı): predictions_log `interval` %100 "1m"; bar_ts karışık tz-aware/naive.
- **P1-2 pairs/kointegrasyon** (`research/pairs_test.py` + 8 test + `docs/PAIRS_TEST_2026-07-18.md` +
  EDGE_FINDINGS **§22**): BEŞİNCİ aile de kanıt kapısını GEÇEMEDİ ama YENİ başarısızlık modu belgelendi —
  Kalman varyantı PF 2.08 + survivorship-PARİTELİ + PBO 0.004 iken `--exec-lag 1`'de PF 1.03'e çöktü
  (kenar ilk bardaydı = reversal/bid-ask artefaktı) → **ACI DERS #7** CLAUDE.md'ye eklendi: exec-lag
  duyarlılığı artık kanıt kapısının parçası. Veri gerçeği: Alpaca ücretsiz SIP günlük ~2021-07'den → 5y/16 pencere.
- **Servisler restart edildi (kullanıcı istedi, yeni kod canlıya geçti):** 03:00 reboot'u servisleri öldürmüştü;
  KÖK NEDEN = watchdog görevleri yalnız GÜNLÜK 00:05 tetikli (boot/logon tetiği YOK, script başlıkları
  "at startup ile kaydet" derken) → reboot sonrası 10+ saat ölü kalmışlar. Watchdog görevleri elle tetiklendi
  (hub önce), 13:46'da ikisi de UP (hub PID 9324, kayıtçı 10156); KANIT: `/api/tearsheet` HTTP 200 gerçek
  QuantStats HTML (eski kodda 404 olurdu); resolve turu yeşil (36.544 çözülmüş, %47.1).
- Doğrulama: tam suite **736 test yeşil** · ruff repo temiz · mypy nihai 64/0. statsmodels 0.14.6 eklendi
  (yalnız research; requirements + rapor §5).

### AÇIK / KULLANICIYA (güncel)
1. **Watchdog kayıtlarına AtStartup/AtLogOn tetiği ekle** (bir kez YÖNETİCİ PowerShell; register-*.ps1
   güncellenmeli) — yoksa her gece-dışı reboot servisleri ertesi 00:05'e kadar öldürür.
2. P2 paketi (FinBERT shadow, /neden RAG komutu, QuantLib, OpenBB zinciri) + P1-4/P1-5 bekliyor.
3. ANTHROPIC_API_KEY hâlâ boş (AI-1 kilidi) — değişmedi.

---

## Ek Tur 15–17 — AI v2/v3 "Quant Beyin" (2026-07-19 → 2026-07-25) [özet; detay docs/]

Bu üç tur repo içindeki raporlarda tam kayıtlıdır; buraya yalnız iz düşürülüyor
(log Tur-14'te durmuştu, boşluk kapatıldı):

- **Tur-15 (`be047b4`, 2026-07-19):** AI v2 "Quant Beyin" — 7 modül + kapasite merdiveni
  (26→3.777 param), KARAR-DIŞI/GÖLGE. 938 test. Spec/rapor: `docs/AI_V2_BEYIN_*`.
- **Tur-16 (`34f3f52` + `f9ecd27`, 2026-07-24/25):** AI v3 Derin Beyin — `QuantTransformer`
  (5,44M param) + 9 modül + köprüler. 1615 test. `docs/AI_V3_DERIN_BEYIN_RAPORU_2026-07-25.md`.
- **Tur-17 (`045f512` + `9ff5c4a`, 2026-07-25):** gölge kipi açıldı (`shadow_runner`),
  overnight edge_lab pilotu **5/5 kapı FAIL**, ilk registry model kartları
  (PIT AUC 0.500 = veri boş) ve **ACI DERS #8**: sınıf dengesizliği DSR'yi şişiriyordu —
  sözde-getiri artık çoğunluk-kuralına göre FAZLALIK (`excess_pseudo_returns`); aynı model/
  aynı veride DSR 1.0 → **0.0**. 1678 test. `docs/AI_V3_TUR17_RAPORU_2026-07-25.md`.

---

## Ek Tur 18 — pano kapsamı + nöral gölge köprüsü + watchdog tetiği (2026-07-25)

**Kapsam:** Tur-17 raporu §6'daki açık kalemler + backend/arayüz kapsam denetimi.
Tam rapor: `docs/PANO_KAPSAM_ANALIZI_2026-07-25.md`.

- **Pano kapsamı (Bölüm A):** 84 modülün public yüzeyi AST ile çıkarılıp panoyla
  karşılaştırıldı. `nihai/ai` katmanının TAMAMI (22 modül) ve piyasa istihbaratı modülleri
  panoda yoktu; `/api/forward-test` ve `/api/tearsheet` uçları VARDI ama HTML hiç çağırmıyordu
  (projenin kendi #1 önceliği görünmüyordu). YENİ `nihai/web/api_extra.py` (21 uç, yeni
  yetenek yok — mevcut çıktıları HTTP'ye bağlar, hepsi `_safe()` fail-soft) + 4 yeni sekme
  (9→13, tembel yüklenir → 5 sn'lik döngüye maliyet yok). Model kartları tablosu **ACI DERS #8**'i
  otomatik uygular: `AUC≈0.50` iken `DSR>0.9` satırı kırmızıya boyanır.
- **`neural_pwin` ÖLÜ KÖPRÜSÜ KAPANDI (Tur-17 §6-1):** `learning.py` kolonları arıyordu ama
  yazan yoktu. YENİ `nihai/execution/neural_shadow.py` (kayıtlı `NeuralCapacity` + MC-dropout)
  → `prediction_tracker` `neural_pwin`/`_side`/`_std` kolonları → `sources.neural` Brier'i.
  ANAYASA korundu: değer `_neural` altında taşınır, kayıttan sonra satırdan DÜŞÜRÜLÜR → web
  JSON'una/işlem tablosuna girmez (model AUC'si 0.500; ikinci "P(win)" sütunu sinyal sanılır).
  Model seçimi üç kilitli (sentetik model canlıya alınmaz · `feature_names` birebir · artefakt
  diskte). Şema göçü `_ensure_schema` ile atomik (52.258 eski satır `''` alır, veri kaymaz).
- **BULGU-17 — doygunluk artefaktı:** canlı duman testinde dağılım-DIŞI vektör `p=0.0, std=0.0`
  verdi. Bu kesinlik değil, ağın doyması; loglansa Brier'i sahte-kesin değerlerle kirletirdi.
  Guard: `std==0` **ve** p uçta → kaydetme (dropout aktifken gerçek std asla tam 0 olmaz).
  Aynı testte gerçek PIT satırlarında p_win 0.24–0.44 / ort. **0.356**, taban oran **0.357** —
  yani model AUC-0.500 karnesiyle tutarlı. **Canlı `sources.neural` Brier'i İYİ ÇIKMAYACAK;
  bu kusur değil, ölçümün kendisi.** Gecikme 7,8 ms/aday.
- **BULGU-18 — enforce kipinde bozuk `is_long` (latent):** kapı adayı WAIT'e çevirince
  `extract_features` `is_long=0` yazıyordu (aday LONG olsa bile) → hem canlı eğitim seti hem
  gölge çıkarım sessizce bozulurdu. Aday tarafı tek kaynağa alındı (`_candidate_side`).
  Üretim `shadow` kipinde olduğu için tetiklenmemişti.
- **Watchdog AtLogOn tetiği (Tur-14 açık madde 1) — KÖK NEDEN:** kayıt scriptlerindeki
  *"Daily + StartWhenAvailable = AtLogOn eşdeğeri"* yorumu YANLIŞTI; StartWhenAvailable yalnız
  **kaçırılan** tetiği telafi eder, gün içi reboot'ta 00:05 zaten koşmuş sayılır. 2026-07-18'de
  10+ saatlik ölü servisin sebebi buydu. Kayıt scriptleri düzeltildi (Daily + AtLogOn); canlı
  görevlere uygulama yükselti istediğinden **`tur18-servis-tetik.ps1`** (idempotent, yönetici).

**Doğrulama:** tam paket YEŞİL (+22 test, `tests/test_neural_shadow.py`) · ruff temiz ·
mypy nihai **85 dosya 0 hata** · canlı duman testi (gerçek model + gerçek PIT verisi).

### AÇIK / KULLANICIYA (güncel)
1. **`tur18-servis-tetik.ps1`'i YÖNETİCİ PowerShell'de çalıştır** (bir kez) — reboot sonrası
   servislerin kendiliğinden kalkması için.
2. **Kayıtçı servisi yeniden başlat** — 21 yeni uç ve `neural_pwin` kolonları ancak o zaman
   canlıya geçer (şema/rota süreç belleğinde). Zamanlama operatörün: forward-test kaydı
   kesintiye uğramasın.
3. ANTHROPIC_API_KEY hâlâ boş (FAZ AI-1 kilidi) — değişmedi.
4. Sıradaki geliştirme: edge_lab BACKLOG #2 (VRP) / #3 (lead-lag) — kanıt-kapılı yeni aday aile.

---

## Ek Tur 21 — Hermetiklik + ölü-hub devre kesici + tip hijyeni (2026-07-31, Opus 5 max)

**Kapsam:** saf OPTİMİZASYON turu — kenar iddiası YOK, verdikt makinesine dokunulmadı.
Tur-19/20 bu log'a değil `docs/`'a yazılmıştı; bu tur log'a geri dönüyor.

**Yöntem:** her kalem ÖNCE ölçüldü, SONRA değiştirildi, SONRA yeniden ölçüldü (Ek Tur 6/7/10
deseni). Hiçbir kazanç tahmin edilmedi.

### Temel (değişiklik öncesi, yerinde ölçüldü — rapordan değil)
`pytest` **1884 test / 114 dosya / 106,2 sn / yeşil** · ruff temiz · mypy nihai 87/0 ·
mypy research **14 hata / 5 dosya**.

### BULGU-19 (YÜKSEK — test hijyeni + KOTA): testler DIŞ AĞA çıkıyordu

BULGU-8/9 testlerin ÜRETİM DOSYALARINA yazmasını 2026-07-02'de kapatmıştı; **ağ tarafı açık
kalmıştı**. Tüm pakete bir soket probu takıldı: 1884 testten **tam olarak biri** dışarı çıkıyor —
`test_main_llm_falls_back_deterministic` → `generativelanguage.googleapis.com:443`.

Mekanizma: test yalnız `ANTHROPIC_API_KEY`'i siliyordu; `router._provider_chain()` **GEMINI'ye
düşüp** `.env`'deki gerçek anahtarla CANLI istek atıyordu. Doğrulama: anahtarsız
`router.complete("summary","merhaba")` → `'Merhaba! Size nasıl yardımcı olabilirim?'` (2,98 sn).
Testin kendi docstring'i "**Ağ/LLM çağrısı YOK**" diyordu — belge yanlıştı.

Bedeli üç katmanlı: (1) her tam koşu ücretsiz Gemini kotasından **bir istek** yakıyordu — o kota
2026-07-13'te FAZ AI-1 ölçümünü zaten bloke etmişti (Ek Tur 11); (2) **12,88 sn** (suite'in %12'si);
(3) çevrimdışı/kotası dolu makinede kırılganlık.

**Düzeltme — iki katmanlı, çünkü engellemek TEK BAŞINA yetmez:** üretim kodu LLM hatasını
bilerek yutar (`router.complete` sözleşmesi) → engellenen çağrı testi yeşil bırakır, ihlal
GÖRÜNMEZ kalır. Bu yüzden kök `conftest.py`:
- `socket.connect`/`connect_ex` sarmalanır; loopback SERBEST (`_port_open` testleri gerçek
  yerel dinleyici açar), dış hedef **hem yükselir hem deftere yazılır**;
- ihlal, çağrı fazı kancasında (`@pytest.hookimpl(wrapper=True)`) teste hata olarak döner.
  **İlk sürüm teardown'da yükseliyordu ve pytest o istisnayı SetupState'e taşıyıp SONRAKİ
  testlerin setup'ını da düşürüyordu (tek ihlal → zincirleme ERROR); sonda testiyle yakalandı.**
- kaçış: `@pytest.mark.allow_network` / `NIHAI_ALLOW_NETWORK=1`.

Test hermetik hale getirildi (üç anahtar da silinir) ve **canlı çağrının tesadüfen kapsadığı
dallar gerçek testlere çevrildi**: sağlayıcı istisnası, dürüstlük-guard'ının RED dalı ve
—ilk kez— KABUL dalı. Yeni `tests/test_net_guard.py` guard'ın no-op olmadığını kanıtlar ve
**dışarı tek paket göndermez** (RFC 5737 `192.0.2.1`; guard `connect`'ten ÖNCE yükselir).

### BULGU-20 (ORTA — performans): ölü hub'a her çağrı 2,03 sn

Panonun `/api/tools/position-size` ucu profillendi: istek başına **7,30 sn** (ikinci istek 3,65 sn).
Kök neden `HubProvider`: hub KAPALIYKEN her çağrı bağlantı denemesini bekliyordu.
Ölçüm — 10 ardışık `get_bars`: **2,061 2,022 2,026 2,044 2,031 2,015 2,029 2,030 2,019 2,029 sn
= 20,30 sn** (ilk-çağrı artefaktı DEĞİL, her seferinde).

Hub'ın kapalı olması istisna değil: watchdog kalkana kadar ~30 sn, **2026-07-18'de 10+ saat**,
Tur-18'de 6 gün. O süre boyunca taramanın HER sembolü ve panonun HER isteği bu bedeli ayrı ayrı
ödüyor — sonunda zaten yfinance'e düşerek. 30 sembollük bir turda ≈60 sn boşa.

**Düzeltme:** `HubProvider`'a devre kesici (`router.py`'deki cooldown deseninin aynısı, sınıf
düzeyinde base_url anahtarlı). **Yalnız BAĞLANTI hatası** (`httpx.TransportError`/`OSError`)
kesiciyi açar — 404/500 AÇMAZ, yoksa tek desteklenmeyen uç ayakta olan hub'ın tamamını 30 sn
devre dışı bırakırdı. Başarılı çağrı kesiciyi sıfırlar; pencere dolunca yeniden sondalanır
(kalıcı kapanma yok). `HUB_BREAKER_SEC=0` eski davranışı birebir geri verir (ölçüldü: 2 çağrı
4,03 sn). Madde 45 sözleşmesi korunur: dönüş yine BOŞ DataFrame, failover yine yedeğe geçer —
kesici yalnız HIZLANDIRIR.

**Ölçülen:** 10 çağrı **20,30 sn → 2,03 sn (10×)** · uç testi **8,35 sn → 3,97 sn**.

### BULGU-21 (ORTA — operasyon): `proje-kapat.ps1` yetim işçileri bırakıyordu

Oturum başında canlı gözlem: proje 22:55'te kapatılmışken **16 python süreci hâlâ ayakta**,
ebeveyn (PID 25392) ölü, **1067 MB RAM** tutuluyor, 45+ dakikadır. Neden: Windows'ta havuz
ebeveyni öldürülünce `multiprocessing.spawn` çocukları hayatta kalır ve komut satırlarında
`nihai...` GEÇMEDİĞİ için betiğin (3). adımı onları hiç görmüyor.

**Düzeltme:** betiğe (4). adım — yalnız `multiprocessing.spawn` işçileri VE yalnız **ebeveyni
ÖLMÜŞ** olanlar süpürülür; canlı bir havuz ya da başka projenin python'u (graphify-mcp)
etkilenmez. Kuru koşu: 16/16 yetim bulundu, 4 meşru python'a dokunulmadı. Betik BOM'suz saf
ASCII kaldı (2026-07-10 "BOM'suz ps1 + ASCII-dışı = PS5.1 parse bombası" dersi; yazarken bir
`ğ` kaçtı, aynı turda yakalanıp temizlendi) ve `PSParser` ile doğrulandı.

### Tip hijyeni: `mypy research` 14 → 0 (Tur-14'ten beri açıktı)

Beş dosya, hepsi davranış-nötr: `forward_harness` `v in (None,...)` → `v is None or ...`
(daraltma); `capacity_data_sweep` `dict[str, Any]` + walrus'lu `dsrler` (çift `.get` de gitti);
`agreement` `base_pf is not None` (zaten birlikte atanıyorlar); `build_extra_universe`
`cast(list[Any], ...)` (raw=False'ta daima `Asset`) + `Adjustment.ALL`/`DataFeed.SIP`;
`alpaca_coverage_probe` aynı enum düzeltmesi (alpaca-py string'i zaten aynı değere çözüyordu).

### Doğrulama (değişiklik sonrası, tam koşu)
`pytest` **1900 test / 116 dosya / 79,4 sn / YEŞİL** (+16 test: 7 ağ guard'ı, 6 devre kesici,
3 net LLM dalı) · ruff temiz · mypy nihai 87/0 · **mypy research 57/0**.

**Toplam ölçülen kazanç:** suite **106,2 → 79,4 sn (−%25)** · ölü-hub çağrısı **10×** ·
tam koşu başına gerçek LLM isteği **1 → 0**.

### AÇIK / KULLANICIYA (güncel)
1. **Şu anda 16 yetim işçi (1067 MB) ayakta** — düzeltme betiğe girdi ama mevcut süreçler
   duruyor. Temizlemek için: `.\proje-kapat.ps1` (yeni (4). adım süpürür).
2. Tur-18/20'den devreden: `tur18-servis-tetik.ps1` + `tur20-shadow-gorev.ps1` YÖNETİCİ
   PowerShell'de bir kez; `NihaiAlgo-ShadowRunner-Hourly` hâlâ `Disabled`.
3. ANTHROPIC_API_KEY hâlâ boş (FAZ AI-1 kilidi) — değişmedi.
4. Sıradaki geliştirme (değişmedi): edge_lab BACKLOG #2 (VRP) / #3 (lead-lag).

---

## Ek Tur 22 — AI v4 "Kapasite ve Kalibrasyon" (2026-08-01/02, Opus 5 max)

Kapsam: `nihai/ai` katmanının her ekseninde geliştirme + parametre sayısının artırılması.
**Anayasa değişmedi** — hiçbir çıktı sinyale/eşiğe/META_GATE'e/execution'a bağlanmaz. Bu tur da
kenar peşinde DEĞİLDİR; aşağıdaki hiçbir sayı "kenar bulundu" anlamına gelmez.
Detay rapor: `docs/TUR22_RAPORU_2026-08-01.md`.

### Kapasite: yeni üst katman + derin topluluk

`quant_transformer_xl` (≈30,4M param; üretim f=96'da 31,76M): 12 blok, d=384, 6 başlık,
**SwiGLU** FFN, **RMSNorm**, **LayerScale** (γ₀=1e-4), doğrusal artan **stokastik derinlik**
(0→0,1), **PLR periyodik** sayısal gömme (Gorishniy 2022). v3'ün 5,39M'ine göre ≈5,6×.

**Bilerek DEĞİŞTİRİLMEYENLER** — kırılganlık burada: dizi düzeni `[CLS]+3×[SCALE]+özellikler`
(`explain.attention_rollout` etiketlemesi buna dayanır), `nn.MultiheadAttention` (özel bir
dikkat implementasyonu açıklanabilirliği SESSİZCE öldürürdü), `QuantTransformer` sınıfının
kendisi (kayıtlı canlı artefaktlar + determinizm testleri bit-denk kalsın diye).

`NeuralEnsemble` (K üye × farklı seed; bagging YOK — az-veri rejiminde bootstrap ayrıca veri
kaybettirirdi). Belirsizlik **toplam varyans yasasıyla** ayrışır: `Var[p] = E[Var[p|üye]] +
Var[E[p|üye]]` — birinci terim üye-içi MC-dropout, ikincisi **epistemik** (`member_spread()`).
`load_any()` tekil/topluluk artefaktını ayırt eder → canlı köprü kod yolu DEĞİŞMEDİ.
5 üyeli XL topluluğu üretim özellik sayısında 158,8M param.

### Eğitim reçetesi — hepsi OPT-IN, bayraksız koşu v3 ile BİT-DENK

`fit()`: `scheduler="cosine"` (ısınmalı, adım-başı), `ema_decay`, `class_weight="balanced"`,
`label_smoothing`, `early_stop_metric="val_auc"`, `calibrate=True`, `device="auto"`.
Bit-denklik testle kilitli (`test_varsayilan_recete_v3_ile_notr`) — kayıtlı canlı artefakt ve
önceki tur ölçümleri geçerliliğini korumalı.

İki nokta kritik:
- `class_weight="balanced"` **YALNIZ eğitim diliminin** sınıf oranından hesaplanır. Val/test'e
  bakmak sızıntı olurdu. Bu, ACI DERS #8'in eğitim tarafındaki karşılığıdır: dengesizliği
  yalnız ÖLÇÜMDE değil, KAYIPTA da dengele.
- `calibrate=True` **sıralamayı DEĞİŞTİRMEZ** — sıcaklık ölçekleme monotondur, AUC birebir aynı
  kalır, yalnız olasılıklar dürüstleşir (`test_kalibrasyon_sirayi_korur_olasiligi_degistirir`).
  Kalibrasyonun "kenar üretmediğinin" kod-düzeyi kanıtı. Sıcaklık araması **deterministik 1-B**
  (kaba log-ızgara + altın-oran), LBFGS değil — sonuç ortam/sürüm bağımsız yeniden üretilebilir
  olmalı (anayasa madde 6).

### OOD savunması: sabit eşikten VERİ-TÜRETİLMİŞ eşiğe

Tur-18/20'de guard sabit `max|z| > 6.0` kullanıyordu; 6,0 bir TAHMİNDİ. Artık `fit()` eğitim
setinin `|z|` %99 kuyruğunu ölçüp `train_zmax_q99` olarak artefakta yazar; `ood_zmax(X)` satır
başına max|z| döner (**σ<1e-9 kolonlar dışlanır** — orada z tanımsızdır; `rr_ratio` σ=3,2e-15
örneği sahte 10¹⁵ mertebesinde z üretirdi). `NEURAL_SHADOW_MAX_Z=auto` → `q99 × 1,5` (en az 4,0);
artefakt eskiyse **sessizce 6,0'a düşer ve nedenini loglar** — dürüst degrade, sessiz yanlış eşik değil.

### Ölçüm dürüstlüğü: ECE her yerde

**Neden Brier yetmiyor:** Brier = kalibrasyon + ayırt etme karışımıdır. Ayırt etmesi sıfır ama
kusursuz kalibre bir model (hep taban oranını söyleyen) DÜŞÜK Brier alabilir. **ECE yalnız
kalibrasyonu ölçer** — ölçülen OOS-AUC ~0,50 iken (ayırt etme yok) tek dürüst ilerleme ekseni budur.
Eklendiği yerler: `neural.evaluation_metrics()` (AUC beraberlik-doğru Mann-Whitney + Brier + ECE),
`learning.expected_calibration_error()` (`reliability_bins` ile **aynı kovalardan** — pano eğrisi
ile tek-sayılı özet birbirini tutar) + `sources.{prior,pwin,neural}.ece`, `selfcheck.exam_calibration()`,
`research/train_neural.py` fold-başı + kart `oos.brier_mean`/`oos.ece_mean`.

### ÖLÇÜM — canlı 1m seti, 5.996 satır, purged+embargolu 5-fold, AYNI fold'lar

| | OOS-AUC | Brier | ECE | DSR | param |
|---|---|---|---|---|---|
| v3 taban (bayraksız) | 0,525 ± 0,013 | 0,2491 | 0,0377 | 0,0006 | 186.702 |
| v4 reçetesi | 0,537 ± 0,012 | 0,2489 | 0,0312 | 0,1805 | 186.702 |
| v4 + 5 üyeli topluluk | **0,541 ± 0,010** | 0,2489 | **0,0280** | 0,3259 | 933.510 |

Reçete adımında **5/5 fold'da** AUC arttı (0,522→0,539 · 0,507→0,517 · 0,542→0,555 ·
0,537→0,544 · 0,518→0,533) — tek şanslı fold değil. ECE toplamda **%26** düştü.

**⚠️ Bu bir KENAR DEĞİLDİR.** AUC 0,541 hâlâ yazı-turaya yakın (ayırt etme fiilen yok);
**DSR 0,3259 hiçbir kanıt kapısını geçmez** (kapı ≈0,95). Kazanılan şey KALİBRASYON ve reçete
disiplinidir, ayırt etme değil (ACI DERS #3). Kapasite artışı bir kenar iddiası değildir:
**darboğaz veri, mimari değil** (ACI DERS #5) — 30M param'lı XL 6k satırda eğitilmemelidir,
merdiven onu 400k satırdan önce ÖNERMEZ ve promotion yine kanıt-kapılıdır.

### Doğrulama (bu turda YERİNDE ölçüldü, rapordan alınmadı)
`pytest` **1966 test / 116 dosya / 76,7 sn / YEŞİL** (rapor 1963 diyordu — yerinde sayım 1966;
Tur-12 dersi: baseline'ı rapordan değil yerinde ölç) · `ruff check .` temiz ·
`mypy nihai research` **144 dosyada 0 hata**.

### Hijyen
`.gitignore`'a `*.bak` — Tem-13'ten kalma `hub-watchdog.ps1.bak`/`predict-watchdog.ps1.bak`
(STOP-bayrağı öncesi sürümler, mevcut betiklerin gerçek alt kümesi) git durumunu kirletiyordu.
Dosyalar SİLİNMEDİ, yalnız izlemeden çıkarıldı.

### AÇIK / KULLANICIYA (güncel)
1. Tur-18/20'den devreden: `tur18-servis-tetik.ps1` + `tur20-shadow-gorev.ps1` YÖNETİCİ
   PowerShell'de bir kez; `NihaiAlgo-ShadowRunner-Hourly` hâlâ `Disabled`.
2. ANTHROPIC_API_KEY hâlâ boş (FAZ AI-1 kilidi) — değişmedi.
3. Tur-22 §7'den devreden operatör kararları: topluluğun canlı köprüye alınması (5× çıkarım
   maliyeti) ve `MAX_Z=auto`'nun varsayılan yapılması (q99 alanlı yeni model kaydı gerekir).

---

## Ek Tur 23 — edge_lab BACKLOG #3: sektör-içi LEAD-LAG ailesi ölçüldü (2026-08-03, Opus 5 max)

Üç turdur "sıradaki geliştirme" olarak bekleyen aday koşuldu. **Verdikt: kenar YOK**
(5 kapıdan 2'si FAIL). Detay: `research/edge_lab/SONUC_leadlag_pilot_2026-08-03.md`,
kalıcı kayıt: `docs/EDGE_FINDINGS.md §26`.

### Yeni: `research/edge_lab/build_leadlag_dataset.py`

Aile: aynı sektördeki likit bir hissenin `t` barındaki hareketi, küçük üyelerin `t+1` barına
gecikmeyle yayılır mı? (Lo & MacKinlay 1990; Hou 2007.) Veri: Alpaca SIP saatlik, 2 yıl,
28 sembol / 7 sektör → **110.704 satır**, taban oran 0,4414.

İki tasarım kilidi — sonucun okunabilmesi bunlara bağlı:

- **Lider HINDSIGHT'SIZ.** "NVDA yarıiletkenlerin lideridir" demek bugünün sıralamasını iki
  yıl öncesine taşımak olurdu (sinsi look-ahead). Lider her barda yeniden, **son 20 barın
  medyan dolar-hacmiyle** belirlenir (geriye-bakan; `t` kapanışında bilinen bilgi). Medyan
  seçildi çünkü ortalama, tek bir haber-günü hacim patlamasıyla liderliği devirirdi.
  Liderlik gerçekten el değiştirdi: 8 üyeli yarıiletken sektöründe **8 sembolün hepsi** bir
  noktada takipçi satırı üretti. Test kilitli: `test_lider_zamanla_degisebilir_ve_geriye_bakar`
  devrin rejim değişiminden ÖNCE olamayacağını doğrular.
- **Aynı-gün kısıtı.** `t` ve `t+1` farklı takvim günündeyse satır düşürülür; aksi halde etiket
  gece boşluğunu içerir ve BACKLOG #1'de zaten 5/5 kapıda FAIL etmiş overnight hipoteziyle
  karışırdı. Hipotez temiz tutuldu (saf gün-içi yayılım).

Sektör grupları elle (ETF'ler SPY/QQQ kasıtlı DIŞARIDA — piyasa vekilidir, sektör üyesi değil;
içeri alınsa "lider" kavramı piyasa-betasına dönerdi). **+11 test**, hepsi yeşil; ağırlık
merkezi leak-free kanıtı: `test_gelecek_bar_yalniz_etiketi_etkiler` t+1 barını bozar ve
HİÇBİR özelliğin değişmediğini, yalnız etiketin değiştiğini doğrular.

### Sonuç: 5 kapıdan 2'si FAIL

| kapı | sonuç | sayı |
|---|---|---|
| purged_cv | geçti (kıl payı) | OOS-AUC **0,532** (eşik >0,53) |
| gate_time | geçti | base E −0,1095 → kapı E +0,1034, **alınan 29/33.212** |
| gate_symbol | geçti | base E −0,1223 → kapı E +0,0667, **alınan 30/29.617** |
| **dsr** | **FAIL** | **0,3058** (n=29; eşik >0,95) |
| **exec_lag_1** | **FAIL** | 1-bar gecikmede kapıyı geçen işlem **29 → 3** |

### Neden kenar değil (üç bağımsız gerekçe)

1. **Seçicilik %0,087** — kapı 33.212 işlemden 29'unu tutuyor. 29 gözlemle hesaplanan +0,10
   expectancy'nin standart hatası farkın kendisi mertebesinde; "PASS" istatistiksel olarak
   bir şey söylemiyor.
2. **DSR bunu yakalıyor** (0,31 ≪ 0,95). ACI DERS #8'in AYNASI: orada kapı bozuktu ve rapora
   RAĞMEN geçiyordu; burada kapı raporun geri kalanıyla TUTARLI davranıyor — sistem çalışıyor.
3. **exec-lag-1 belirleyici darbe.** Bilgi tamamen sinyal-sonrası İLK barda. Saatlik barlarda
   bid-ask sıçraması / mikroyapı geri-dönüşü tam bu imzayı üretir.

BACKLOG.md bu satıra **koşudan ÖNCE** "exec-lag-1 testi burada ÖZELLİKLE kritik (ACI-7 riski
yüksek)" notunu düşmüştü — **not doğru çıktı.**

### Kalıcı kazanımlar (kod değil, KURAL)

- **ACI-7 üçüncü kez doğrulandı** (§22 pairs · §23 giriş kayması · §26 lead-lag). CLAUDE.md'de
  ders "artık VARSAYILAN ŞÜPHE" olarak yükseltildi.
- **YENİ okuma kuralı: kapı PASS dese bile `n_kept` okunmalı.** Harness'ın `_MIN_KEPT=5`
  mutlak tabanı 33k'lık bir test diliminde anlamını yitiriyor. **Bilerek DEĞİŞTİRİLMEDİ:**
  bir aday FAIL ettikten SONRA kanıt kapısının eşiğini sıkılaştırmak, sonuca göre kural
  değiştirmektir. Ayrı ve adayı olmayan bir turda yapılmalı (SONUC raporu §6'da öneri olarak duruyor).
- **Denenen-aday sayacı** BACKLOG.md'ye eklendi (şu an **2**: #1 overnight, #3 lead-lag —
  ikisi de FAIL). Aileler arası çoklu-test `run_candidate`'ın içindeki `n_trials` deflasyonuyla
  sayılmaz; insan tarafından izlenmeli. Sayaç büyüdükçe ilk PASS'in kanıt değeri DÜŞER.

**Öncelik notu:** evren survivorship-ŞİŞKİN, yani kenar LEHİNE yanlı. Şişkin evrende bile
geçemeyen aday temiz evrende geçmez → PIT tekrarının önceliği DÜŞÜK. **Aile kapandı.**

### Doğrulama
`pytest` **1977 test / 117 dosya / YEŞİL** (+11) · `ruff check .` temiz ·
`mypy nihai research` **145 dosyada 0**.

### AÇIK / KULLANICIYA (güncel)
1. Tur-18/20'den devreden: `tur18-servis-tetik.ps1` + `tur20-shadow-gorev.ps1` YÖNETİCİ
   PowerShell'de bir kez; `NihaiAlgo-ShadowRunner-Hourly` hâlâ `Disabled`.
2. ANTHROPIC_API_KEY hâlâ boş (FAZ AI-1 kilidi).
3. **Tur-22 §7'nin kalan kalemleri bu turda KAPANMADI ve nedeni donanım/karar:**
   - XL'in (30,4M param) 97k/234k sette kapasite süpürmesi: bu makinede **GPU YOK**
     (`torch.cuda.is_available()=False`, 8 CPU thread). **ÖLÇÜLDÜ** (varsayılmadı): XL 31.756.051 param, CPU maliyeti **92,6 ms/örnek/epoch**
     (n=400→37,1 sn ve n=800→74,1 sn — tam doğrusal). Buradan 40 epoch × 5 fold:
     canlı 1m seti (5.996) **30,8 saat**, PIT 97k **499 saat (21 gün)**, extra 234k
     **1.205 saat (50 gün)**. Yani "ağır" değil, bu donanımda **imkânsız**.
     Beklenti zaten "kenar" değil, "kapasite eğrisi nerede düzleşiyor" idi ve §25 (P2-11)
     öğrenme eğrisi zaten DÜZ çıkmıştı → bilgi değeri düşük, maliyeti yüksek. GPU gelirse
     `--device auto` hazır.
   - Topluluğun canlı köprüye alınması (5× çıkarım maliyeti) ve `MAX_Z=auto`'nun varsayılan
     yapılması: ikisi de **operatör kararı** — canlı davranışı değiştirir, kod hazır.
4. Sıradaki geliştirme: edge_lab BACKLOG **#2 (VRP)** — ama veri kaynağı (opsiyon zinciri/IV)
   henüz belirlenmedi; ondan önce **#4 (endeks rebalans/akış)** `pit_universe.py` ile
   PIT-uyumlu inşa edilebildiği için daha hazır bir adaydır.

---

## Ek Tur 24 — kanıt kapısının sertleştirilmesi + kodlama çökmesi + ekleme O(n²) (2026-08-10, Opus 5 max)

Bu tur **adayı olmayan** bir turdur ve bu KASITLIDIR: Tur-23, kanıt kapısını sıkılaştırma
önerisini "bir aday FAIL ettikten SONRA eşik değiştirmek, sonuca göre kural değiştirmektir;
ayrı ve adayı olmayan bir turda yapılmalı" diyerek ertelemişti. Kural burada, elde hiçbir
aday yokken sabitlendi.

### 1) Kanıt kapısı: iki EK yeterlilik ölçütü (`research/edge_lab/harness.py`)

§26'nın kalıcı dersi ("kapı PASS dese bile `n_kept` okunmalı") artık İNSAN okuma kuralı
değil, MAKİNE kuralı:

- **(a) Oransal seçicilik tabanı** — `_min_kept_required(n_test) = max(5, ceil(0,005·n_test))`.
  Mutlak `_MIN_KEPT=5` tabanı 33k'lık bir test diliminde anlamını yitiriyordu. Küçük veri
  setlerinde mutlak taban, büyüklerde oran bağlar → mevcut testler ETKİLENMEZ.
- **(b) Örneklem-koşullu anlamlılık** — kapının TUTTUĞU ile ELEDİĞİ getiriler arasında
  Welch t (tek-taraflı, eşik 1,645). Neden base yerine ELENEN ile kıyas: `kept ⊂ base`
  bağımlı örneklemlerdir, tutulan↔elenen AYRIKTIR — "bu filtre iyiyi kötüden ayırıyor mu"
  sorusunun dürüst istatistiksel biçimi budur.

Aynı taban DSR'yi besleyen `_kept_pnl_from_time_gate`'e de uygulandı (DSR'yi 29 gözlem
üstünde hesaplayıp "n yeterli" saymak §26'nın tam olarak elediği şeydi). Kapı metrikleri
artık `keep_frac` / `min_kept_required` / `t_stat` da RAPORLUYOR.

**GERİYE DÖNÜK DOĞRULAMA (yerinde koşuldu, varsayılmadı):** iki geçmiş aday yeni kapıdan
tekrar geçirildi —

| aday | eski kapı | YENİ kapı | verdikt |
|---|---|---|---|
| #3 lead-lag | 3 PASS / 2 FAIL | **1 PASS / 4 FAIL** | FAIL → FAIL (DEĞİŞMEDİ) |
| #1 overnight | 0 PASS / 5 FAIL | 0 PASS / 5 FAIL | FAIL → FAIL (DEĞİŞMEDİ) |

Kritik nokta: **hiçbir sonuç değişmedi.** Kural yalnız istatistiksel olarak BOŞ olan iki
"PASS"i (29/33.212 ve 30/29.617) kaldırdı. Bir kanıt kapısının sıkılaştırılmasından
beklenecek tam davranış budur — yeni sonuç ÜRETEMEZ, yalnız yanıltıcı olanı eler.
Ham çıktı: `data/verdict_leadlag_v1_tur24gate.json`.

### 2) BULGU — `exec_lag_1` kapısı gerçek PnL kolonunda ölçtüğünü sandığı şeyi ÖLÇMÜYORDU

`_shift_label_1bar` yalnız `label_col`'u kaydırıyordu. Gerçek bir `pnl_col` verildiğinde kapı,
KAYDIRILMIŞ etiketle eğitilip **KAYDIRILMAMIŞ (gecikmesiz) PnL** üstünde ölçülüyordu — yani
ACI-7 kapısı (projenin ÜÇ KEZ doğrulanmış en pahalı dersi) sessizce devre dışıydı. İlk iki
aday sözde-pnl kullandığı için (etiketten türetilir, birlikte kaydırılır) ısırmamıştı;
**gerçek PnL'li ilk adayda sahte PASS üretirdi.** Düzeltildi + regresyon testi
(`test_exec_lag_gercek_pnl_ile_gecikmeyi_gercekten_olcer`).

Yan kazanç: döngülü `.loc` yerine `groupby(...).shift(-1)` — çıktı BİT-DENK doğrulandı,
110.704 satırda 0,045 sn → 0,013 sn (**3,6×**).

### 3) BULGU — yönlendirilmiş stdout'ta cp1254 çökmesi (canlı yol)

**ÖLÇÜLDÜ:** `python -m nihai.ai.selfcheck --report > dosya` gerçekten
`UnicodeEncodeError: 'charmap' codec can't encode character '⚠'` ile ÇÖKÜYORDU.

Kök neden: Windows'ta çıktı dosyaya/pipe'a yönlendirilince stdout konsol kod sayfasına
(cp1254) düşer; `→ ⚠ Δ ↓ λ 📈` gibi karakterler orada YOKTUR. Türkçe harfler (ğ ş ı İ ç ö ü)
cp1254'te VAR — bu yüzden hata yıllarca görünmez kalır ve yalnız ok/emoji/matematik sembolü
basan satırda patlar. Watchdog ve zamanlanmış görevler çıktıyı tam da böyle yönlendirdiği
için hata yolu etkileşimli kullanım değil, **gözetimsiz üretim yoludur.**

Kapsam ölçüldü: 74 CLI giriş noktasının **16'sında** guard yoktu; AST taramasıyla üçünde
riskli karakterler docstring'de değil YAZDIRILABİLİR sabitlerde: `nihai/ai/selfcheck.py`
(doğrulanmış çökme), `nihai/execution/main.py` (canlı tarayıcı — `Δ` delta / `güven↓`),
`research/postmortem.py`. Kalan 13'ü bugün güvenli ama kırılgan.

Düzeltme projenin kendi idiomuyla: **`nihai/console.py`** (`ensure_utf8_stdout()`) —
`atomic_io.py` ile aynı gerekçe, 59 dosyada elle kopyalanmış 5 satırın TEK doğruluk
kaynağı. 16 giriş noktasına eklendi (stderr de kapsanır: traceback'ler de çöküyordu).
Mevcut 59 elle guard DEĞİŞTİRİLMEDİ (çalışıyorlar; gereksiz çalkantı yok).

Asıl kalıcı kazanım kod değil TEST: `tests/test_console_encoding.py` her `__main__`
modülünü tarar → guard'sız yeni bir CLI eklenirse suite kırılır (74 parametreli invariant).

### 4) OPTİMİZASYON — `ab_log.jsonl` eklemesi O(dosya) → O(satır)

`nihai/ai/registry.shadow_ab_log` her A/B gözleminde **dosyanın tamamını okuyup geçici
dosyayla yeniden yazıyordu**. Bu, prediction_tracker'da D1 olarak kapatılan hatanın
(canlı döngüde tüm log'u yeniden okuma) aynısıydı ve `ab_log.jsonl` yalnız-ekleme,
sınırsız büyüyen bir günlüktür.

**ÖLÇÜLDÜ:** 300 satır **3,385 → 0,352 sn (9,6×)** · 2.000 satır **22,935 → 2,263 sn (10,1×)**.
Suite'teki yansıması: `test_ab_report_*` en yavaş beş testti (4,70 sn), şimdi **0,39 sn (12×)**.

Dayanıklılık DÜŞMEDİ, ARTTI: `os.replace` her yazımda TÜM dosyayı riske atar; `write+flush+
fsync` en kötü ihtimalle son satırı yarım bırakır ve `_read_ab_rows` bozuk satırı zaten
atlıyordu. Test bunu kilitliyor (`test_yarim_kalan_son_satir_raporu_bozmaz`).

**Aynı desendeki diğer 4 yer BİLEREK değiştirilmedi** (`episodic`/`lab`/`portfolio`/
`consolidate`): hepsi belgelenmiş biçimde DÜŞÜK FREKANSLI (öneri/ders/konsolidasyon) —
orada tam-dosya atomikliği doğru tercihtir. Değişen tek yer gözlem-başına çağrılan yoldu.

### 5) edge_lab BACKLOG #4 — KOŞULMADI (bilinçli), ama iki engel ölçüldü

Aday çalıştırılmadı; bu turun bütçesi yukarıdaki kural/hata işine ve optimizasyona gitti.
Ancak sonraki tur iki bulguyla ÖNDEN başlıyor:

1. **İyi haber — PIT evren offline kurulabiliyor:** `sp500_as_of("2021-07-01")` = **506
   sembol, 84'ü bugün endekste YOK**, ve bunların **82'si zaten `data/alpaca_hist`'te
   önbellekli**. Yani #4, #1 ve #3'ün aksine "survivorship-ŞİŞKİN pilot" olmak zorunda
   DEĞİL — gerçekten PIT kurulabilir. Değişiklik tablosu 408 satır (1976→2026), yani
   endeks-olayı tarihleri de mevcut.
2. **ENGEL — günlük önbellek DÜZELTMESİZ:** `data/alpaca_hist`'teki 516 `*_1d.csv` dosyası
   `raw` kipiyle yazılmış (`_cache_path`: düzeltilmiş veri `_all` ekiyle AYRI dosyaya gider;
   diskte yalnız 28 adet `1h_all` var). Bölünme (split) günlerinde ham kapanış serisi sahte
   %50 getiri üretir → ölçüm baştan çöp olurdu. **Ölçümden önce
   `ALPACA_HIST_ADJUSTMENT=all` ile günlük veri yeniden çekilmelidir.**
3. **Tasarım kilidi (şimdiden):** Wikipedia değişiklik tablosu **effective date** verir,
   ANONS tarihini değil. "Kaç gün SONRA endeks değişikliği var" özelliği, anons henüz
   kamuya açık değilken bilgi sızdırır (sinsi look-ahead — lead-lag'daki "lider hindsight'sız"
   kilidinin aynısı). Ya yalnız GERİYE-bakan olay mesafesi kullanılmalı ya da açık bir
   anons-gecikmesi varsayımı yazılmalı. Takvim özellikleri (ay-sonu yakınlığı, çeyrek-sonu)
   bu sorundan muaftır — takvim kamuya açıktır.

### Doğrulama (YERİNDE ölçüldü)
Baseline: 1977 test / 117 dosya · ruff temiz · mypy 145 dosyada 0.
Tur sonu: **2069 test / 118 dosya / YEŞİL** (+92) · `ruff check .` temiz ·
`mypy nihai research` **146 dosyada 0** (+1 = `nihai/console.py`).

Süre: tam suite **63,3 sn**. Kıyas için Tur-21 kaydı 1900 testte 79,4 sn idi — bu turda
**+169 test eklenmesine RAĞMEN** süre düştü; fark ağırlıklı olarak §4'teki ekleme
optimizasyonundan geliyor (yalnız `test_ab_report_*` beşlisi ~21 sn → ~1,5 sn).

### AÇIK / KULLANICIYA (güncel)
1. **Servisler 1 Ağustos 00:09'dan beri KAPALI** (`logs/project.stopped` bayrağı; watchdog'lar
   ayakta ve bayrağa DOĞRU biçimde uyuyor). 9 gündür canlı tahmin/özellik birikmiyor.
   ACI DERS #5'e göre kenar sorusunun tek dürüst yolu canlı paper kanıtıdır → bu birikim
   durduğu sürece proje ilerlemiyor. Açmak operatör kararı: `Nihai - Projeyi Ac` kısayolu.
2. Tur-18/20'den devreden: `tur18-servis-tetik.ps1` + `tur20-shadow-gorev.ps1` YÖNETİCİ
   PowerShell'de bir kez; `NihaiAlgo-ShadowRunner-Hourly` hâlâ `Disabled`.
3. ANTHROPIC_API_KEY hâlâ boş (FAZ AI-1 kilidi).
4. Sıradaki geliştirme: edge_lab **#4** — §5'teki ön-koşul (düzeltilmiş günlük veri) ve
   tasarım kilidi (anons vs effective date) BACKLOG.md'ye yazıldı.

---

## Ek Tur 25 — AI katmanının derinleştirilmesi: prior hafızası + kalibrasyon döngüsü + aday #4 (2026-08-10, Opus 5 max planlama · 3 paralel kodlama ajanı, commit 4b393d8)

Bu tur, "AI'ı geliştir" isteğinin bu projede ne ANLAMA GELDİĞİ sorusuna verilmiş bir cevaptır.
İki yaygın okuma baştan reddedildi: (a) daha büyük model — ACI DERS #5 ve Tur-22'nin 158,8M
parametreli ölçümü bunu zaten kapattı; (b) λ'yı gevşetip motoru konuşturmak — denetimin §E.4
ölçümü konuşan motorun para kaybettiğini gösterdi. Geriye kalan üç meşru eksende çalışıldı.

**Model yönlendirmesi:** planlama/tasarım kilitleri/şartname Opus 5 (orkestratör); kodlama
üç paralel ajan — WP-A ve WP-B Sonnet 5, WP-C (araştırma doğruluğu kritik) Opus. Dosya
sahipliği ÖRTÜŞMEYECEK biçimde bölündü (Tur-9'un dersi), entegrasyon+ölçüm tek-thread
orkestratörde kaldı (Tur-12'nin "fix'i cold agent'a verme" dersi bu şekilde uygulandı).

### 0) Taban YERİNDE ölçüldü + iki yeni bulgu

`git status` temiz `81c9d01` · `pytest` yeşil · torch 2.13.0+cpu, **CUDA yok** · servisler
**1 Ağustos'tan beri KAPALI** · `predictions_log.csv` 53.019 satır / 50.660 çözülmüş /
2026-06-24 → **2026-07-31**.

**YENİ BULGU — nöral köprünün canlı kanıtı 6 satır.** `neural_shadow --selfcheck` bugün de
"598/600 üretiliyor, KÖPRÜ CANLI" diyor — ama bu **sentetik** öz-sınavdır. Canlı log'da
`neural_pwin` dolu satır: 24 Temmuz'da 4.444 satırda **0**, 30 Temmuz'da 732 satırda **0**,
31 Temmuz'un son partisinde **6**. Yani Tur-20'nin verdikti doğru ama BOŞ: köprü
çağrıldığında çalışıyor, canlı yolda yalnız 6 kez çağrıldı (Tur-18'in kendi notu: kolonlar
kayıtçı süreç YENİDEN BAŞLADIKTAN sonra dolar; kayıtçı 31 Temmuz gecesi başladı, ertesi gün
servisler kapandı). **`sources.neural` Brier'i n=6 üstündedir = ölçüm DEĞİLDİR.** Kod
hatası değil, veri-birikimi boşluğu — yalnız servisler açılınca kapanır.

**Öğrenme durumunun içi** (`ai_learning_state.json`): GLOBAL 0,5028 · LONG lo95 0,4213
(kapı-a LONG'a KAPALI) · **SHORT lo95 0,5276 (kapı-a SHORT'a AÇIK)** · **`drift: true`** —
ve drift'i hiçbir şey tüketmiyor. Motor tek yönlü bir makine ve bunun nedeni state'te çıplak.

### 1) WP-A — prior'ın hafızası (zaman-ağırlıklı + drift-koşullu Beta posterior)

`learning.py`: `strategies_decayed` (üstel unutma, yarılanma 30 gün, env
`AI_PRIOR_HALF_LIFE_DAYS`) + drift varsa etkin sayımlar `DRIFT_SHRINK=2.0`'a bölünür.
Mevcut `strategies` DOKUNULMADI. `decision.py`: yalnız `AI_PRIOR_DECAY` açıkken okur →
varsayılan **bit-denk**. Kilitler: referans zaman = kuyruğun son damgası ("bugün" değil,
determinizm), tüm damgalar naive-UTC (BULGU-16 tekrarlanmasın), yan-koşullu prior'ın n≥200
kapısı HAM satır sayısına bakar. **+51 test.**

**ÖN-KAYITLI ÖNGÖRÜ ÇÜRÜTÜLDÜ.** Koşudan ÖNCE yazılmıştı: *"bayat-prior hipotezi doğruysa
decay AÇIK iken fazlalık açığı KÜÇÜLMELİDİR; küçülmezse hipotez yanlıştır."* Aynı state
üstünde A/B (180 gün, ufuk 5, 5 bps):

| λ=0,05 | decay KAPALI | decay AÇIK |
|---|---|---|
| SEÇİM: yönlü / FAZLALIK vs hep-LONG | 823 / −2,4236 | 806 / −2,3973 |
| DOĞRULAMA: yönlü / FAZLALIK vs hep-LONG | 553 / **−4,2328** | 531 / **−4,3881** |
| λ=0,5 (üretim) | 0 yönlü / %100 ABSTAIN | 0 yönlü / %100 ABSTAIN |

Doğrulama bölmesinde KÖTÜLEŞTİ; fark gürültü mertebesinde. **Neden ölçüldü:** log yalnız ~5
hafta; 30 günlük yarılanmada en eski satır 0,43 ağırlık alıyor → SHORT `post_mean` 0,5508 →
0,5498, `lo95` 0,5290 → 0,5165 (hâlâ >0,5). **SHORT yanlılığı bu pencerede bayat değil,
baştan sona mevcut** → zararın nedeni prior'ın ESKİMESİ değil, hiç tahmin gücü OLMAMASI.
Kazanç: tespit edilip tüketilmeyen drift artık tüketiliyor (mekanizma testlerle kilitli) ve
mekanizma çok-rejimli log biriktiğinde anlam kazanır. **Varsayılan KAPALI kaldı — ölçüm
kendi düzeltmemizi reddetti.**

### 2) WP-B — yalnız-OOS kalibrasyon döngüsü (`nihai/ai/calibration_loop.py`, YENİ)

`fit_calibrator`/`apply_calibration` yazılmış+test edilmişti ama **üretimde tek çağıranı
yoktu** — repo taramasıyla doğrulandı. Yeni modül kaynak başına KRONOLOJİK bölmeyle
yalnız-OOS kalibrasyon ölçer; artefakt yalnız **Brier VE ECE'nin İKİSİ de** OOS'ta
iyileşirse yazılır. **+51 test** (sızıntı testi: OOS dilimini bozmak fit kalibratörünü
BYTE-DENK bırakıyor).

Gerçek log ölçümü: `pwin` → fit 3.601 / OOS 2.401, **Brier 0,2723→0,2490 · ECE
0,1325→0,0045** (KAYITLI) · `neural` → **YETERSIZ** (n_oos=3 < 200; §0'ın öngördüğü dürüst
yol tetiklendi).

**ÖNEMLİ DÜZELTME — kalibrasyon kazancı "ters bilgi" DEĞİL.** Platt katsayısı negatif
(−0,9134) çıkınca ilk okuma "ham pwin OOS'ta ters korelasyonlu" oldu. Kontrol koşuldu:

| tahminci | Brier | ECE |
|---|---|---|
| HAM `pwin` | 0,2723 | 0,1325 |
| KALİBRE (platt) | 0,2490 | 0,0045 |
| **SABİT = fit taban oranı (0,4646)** | **0,2494** | **0,0090** |
| SABİT 0,5 | 0,2500 | 0,0264 |

Kalibre sürüm sabit tabanı yalnız **0,0004 Brier** geçiyor; kalibre olasılıkların std'si
0,0156 (ham 0,0686), ortalaması 0,4704 ≈ taban oran. Kazancın tamamına yakını **taban orana
BÜZÜLMEDEN** geliyor. Dürüst ifade: *meta `pwin` ağır fazla-güvenlidir ve kullanılabilir
bilgi taşımaz* — ACI DERS #2/#4'ün (META_GATE varsayılan KAPALI) taze ve bağımsız
doğrulaması. **Yeni kural CLAUDE.md'ye ACI DERS #9 olarak yazıldı: bir kalibrasyon
iyileşmesi raporlanmadan önce SABİT-taban tahmincisiyle kıyaslanmalıdır.**

### 3) WP-C — edge_lab BACKLOG #4 (endeks rebalans/akış), PIT

**Ön-koşul kapatıldı (148 sn):** Tur-24'ün engeli (516 `*_1d.csv` `raw` kipinde → bölünme
günlerinde sahte %50 getiri) için 2021-01→2026-07 çeyreklik `sp500_as_of` birleşimi = **604
sembol** `adjustment=all` ile çekildi (0 hata). Mevcut `raw` dosyalar EZİLMEDİ.

Dataset **728.978 satır / 605 sembol / 2021-02-02 → 2026-07-23**, 8 özellik, 101 ekleme +
103 çıkarma olayı, evrenin **90 üyesi bugün endekste YOK** (PIT/survivorship-temiz).
**+33 test**, ağırlık merkezi leak-free kanıtı.

| kapı | sonuç | sayı |
|---|---|---|
| **purged_cv** | **FAIL** | OOS-AUC **0,5066** (eşik >0,53) |
| gate_time | geçti | +0,2514 → +0,5280, n_kept 27.003/218.694 (%12,3), t=8,51 |
| gate_symbol | geçti | +0,2079 → +0,6853, n_kept 18.188/223.163 (%8,2), t=11,51 |
| dsr | geçti | 1,0000 (n=27.003) — **ama bkz. BULGU-25** |
| exec_lag_1 | geçti | etki **−0,2009**; kapı fazlalığının yalnız %27,7'si kalıyor |

Verdikt: **KAPI GEÇİLEMEDİ — kenar YOK.** Denenen-aday sayacı **2 → 3**.
Dört kapı geçti ama okununca boş: kapı endeks olaylarına HİÇ bakmıyor (tutulan↔elenen
`gun_since_eklendi/cikarildi` farkı −0,09/+0,13; gördüğü şey takvim, en büyük standardize
katsayı `ceyrek_sonuna_gun` 0,063). H1 doğrudan test: fiyat baskısının olması gereken 0–10
gün penceresi BOŞ (t=−1,16/−1,73). H2 doğrudan test: satır düzeyi t=+5,75 → **tarih-kümelenmiş
t=+0,52** (11× şişme; günde 531 sembol tek ortak şoktur).

**BULGU-25 — `dsr` kapısı gerçek `pnl_col` altında ayırt edici DEĞİL.** DSR=1,0'ı kapı-sonrası
(SR 0,091), kapısız base (0,048), kapının **ELEDİĞİ** satırlar (0,042) ve **rastgele** alt
küme (0,050) de veriyor: `_stage_dsr` Sharpe'ı sıfıra karşı ölçüyor, `base_E=+0,2514`
(yükselen piyasa) olunca her alt küme pozitif. ACI DERS #8'in aynı yapısal hatasının farklı
yerdeki tekrarı; gerçek `pnl_col` veren İLK adayla görünür oldu. **Kapı bilerek
DÜZELTİLMEDİ** — Tur-23 kuralı (aday FAIL ettikten sonra kapı değiştirmek = sonuca göre
kural değiştirmek) ve doğru taban seçimi açık bir tasarım sorusu: sözde-pnl'de taban oranı
%50'nin altındayken taban çıkarmak kapıyı GEVŞETİR. Adayı olmayan bir turda ele alınmalı.

**ACI-7 dördüncü kez** (§22 pairs · §23 giriş kayması · §26 lead-lag · §27 rebalans). Bu kez
örneklem BÜYÜKTÜ (%12,3 seçicilik, t=8,51) — yani exec-lag kapısı yalnız uç-kuyruk
örneklemlerini değil, büyük ve "sağlam görünen" PASS'leri de eliyor.

**Orkestratör şartnamesindeki hata, ajan tarafından koşudan ÖNCE yakalandı:** brief "giriş
öncesi barlarda `endeks_uyesi=0`" testi istiyordu; bu, sembolün endekse GİRMEDEN önceki
barlarını panele koymayı gerektirir = **evren-seçimi look-ahead'i** ("bu hisse ileride S&P
500'e alınacak") ve ekleme-öncesi ralli literatürün en bilinen etkisi olduğu için tam da
uydurulmuş kenar üretecek yer. Yerine daha sıkı kilit kondu: giriş öncesi satır ÜRETİLMEZ,
çıkarılan isimler panelde KALIR. Sapma raporda belgelendi. **Ders: tasarım kilidini yazan
da yanılabilir; ajanın gerekçeli itirazı kabul edilmeli.**

### Doğrulama (YERİNDE ölçüldü)
Taban: 2069 test. Tur sonu: **2204 test / 0 hata / 0 skip / 61,3 sn** (+135 = WP-A 51 +
WP-B 51 + WP-C 33) · `ruff check .` **All checks passed** · `mypy nihai research`
**148 dosyada 0 hata**. Suite süresi Tur-24'ün 63,3 sn'sinden +135 teste RAĞMEN düşük kaldı.

### AÇIK / KULLANICIYA (güncel)
1. **Servisler 1 Ağustos 00:09'dan beri KAPALI** — 10 gündür canlı tahmin/özellik
   birikmiyor. §0'daki iki bulgu da (nöral köprü n=6, decay'in ölçülemeyişi) doğrudan buna
   bağlı: **AI katmanının bir sonraki adımı kod değil, VERİ bekliyor.** Açmak operatör
   kararı: `Nihai - Projeyi Ac`.
2. Tur-18/20'den devreden: `tur18-servis-tetik.ps1` + `tur20-shadow-gorev.ps1` YÖNETİCİ
   PowerShell'de bir kez; `NihaiAlgo-ShadowRunner-Hourly` hâlâ `Disabled`.
3. `ANTHROPIC_API_KEY` hâlâ boş (FAZ AI-1 kilidi).
4. Sıradaki tur önerisi: **adayı OLMAYAN** bir turda BULGU-25 (dsr kapısının gerçek pnl
   altındaki tabanı) ele alınmalı — geriye dönük doğrulamayla, Tur-24'ün deseniyle.
5. edge_lab sırada: #2 VRP (opsiyon zinciri/IV kaynağı hâlâ belirsiz) veya #5/#6.

---

## Ek Tur 26 — altyapı optimizasyonu: kör kalite kapısı + guard'ın deliği + lazy import (2026-08-11, Opus 5 max)

**Tur tipi:** adayı OLMAYAN optimizasyon turu. Yeni sinyal/kenar iddiası YOK; tamamı
altyapı, hermetiklik ve import maliyeti. Tüm sayılar YERİNDE ölçüldü (Tur-12 dersi:
baseline'ı rapordan okuma).

### §0 — Kapı KÖR ÇIKTI: taban ölçülemedi (KRİTİK, ilk 5 dakikada)

`pytest` **hiç koşmuyordu**: `statsmodels` kurulu değildi ve `tests/test_pairs_math.py`
onu import eden `research/pairs_test.py`'yi çektiği için pytest TOPLAMA fazında duruyordu
(`Interrupted: 1 error during collection`) → **2178 testin tamamı** koşmuyordu. `mypy` de
kurulu değildi, yani CLAUDE.md'de belgelenen tip kapısı da koşturulamıyordu.

Yani "1963/2204 test yeşil" bilgisi bu makinede DOĞRULANAMAZ durumdaydı. Kurulum sonrası
gerçek taban: **2178 test / 1 skip / 0 hata / 143 sn**, ruff temiz, mypy 148 dosyada 0.

Kalıcı düzeltmeler:
- `research/pairs_test.py` → statsmodels **fail-soft** (modül-düzeyi `try/except` + yönergeli
  yer tutucu). `coint` modül ADI olarak korundu, çünkü testler onu monkeypatch'liyor.
- `requirements.txt` → **`scikit-learn` bildirilmemişti** (Tur-26'da eklendi). `nihai/ai/models.py`,
  `nihai/ai/quant_math.py` ve `research/edge_lab/harness.py` onu MODÜL DÜZEYİNDE import ediyor;
  temiz bir `pip install -r requirements.txt` kurulumunda bu modüller ImportError veriyordu.

### §1 — BULGU-26: ağ guard'ı (BULGU-19) delinmiş — curl_cffi socket'i atlıyor (YÜKSEK)

Tur-21'de kurulan hermetiklik guard'ı `socket.socket.connect`'i yamalıyor. **yfinance 1.x
HTTP taşımasını `curl_cffi`'ye taşıdı**; o ise libcurl'ü cffi üzerinden çağırır → socket
yaması HİÇ çalışmaz.

**Kanıt (guard etkinken, aynı koşuda ölçüldü):**

| taşıma | sonuç |
|---|---|
| `socket.create_connection(example.com:80)` | `OSError` — doğru şekilde ENGELLENDİ |
| `curl_cffi.requests.get(query2.finance.yahoo.com/...)` | **`status=429`** — istek gerçek Yahoo sunucusuna ULAŞTI, defterde iz YOK |

Yani yfinance yoluna dokunan testler **sessizce canlı ağa çıkıyordu** — BULGU-19'un tam
olarak engellemek için yazıldığı şey. Guard `curl_cffi.requests.Session.request`
(+`AsyncSession.request`) için de yamalandı; `_guard_url` host'a socket ile AYNI loopback
kuralını uygular.

Delik kapanınca ağa çıkan **2 test** görünür oldu ve hermetikleştirildi:
1. `test_api_extra.py::test_position_size_gercek_hesap` — uç, rejim çarpanı için
   `_bars("SPY","1d",220)` çekiyordu. Dosyanın KENDİ docstring'i "ağ gerektiren uçlar burada
   DEĞİL" diyordu; ihlal guard görmediği için fark edilmemişti. Maliyeti: **9,80 sn** (tüm
   paketin en yavaş testi). `_bars` monkeypatch'lendi → rejim `_safe` sarmalında nötre düşer,
   testin ASIL iddiası (risk $ = |entry−stop| × adet) değişmez.
2. `test_meta_features.py::…covers_all_names…` — `extract_features` → `_vix_close_before` →
   `strategy._load_vix()` → **`yf.download("^VIX", period="10y")`**. Sonuç modül-global
   (`_VIX_LOADED`) olduğu için ihlali HANGİ testin önce koştuğu belirliyordu = sıra-bağımlı
   kırılganlık. Kök conftest'e autouse fixture eklendi (VIX yokluğu güvenli: `_vix_or_neutral`
   nötr 18.0'a düşer). `test_ai_shadow_runner.py` bu deseni zaten yerel olarak çözmüştü;
   genelleştirildi.

Kalan boşluk bilinçli ve BELGELİ: `strategy`'yi test GÖVDESİNDE import eden bir test
fixture'ı ıskalar — ama artık SESSİZ değil, guard gürültülü şekilde kırar.

### §2 — `import nihai.ai` 3854 ms → 11 ms (350×)

`nihai/ai/__init__.py`'nin kendi gerekçesi "`import nihai.ai` her çağıran için ucuz
kalmalıdır" diyordu ama kod `decision`/`quant_math`/`models`/… modüllerini EAGER import
ediyordu. `-X importtime` ile ölçülen zincir: `decision → quant_math → sklearn.covariance
(1,30 sn) + scipy.stats (1,39 sn)`.

Bunu ödeyen yerler: CANLI yol `execution/neural_shadow.py` (`from nihai.ai import registry`)
ve her `web/api_extra` AI ucu. Eager blok API için GEREKSİZDİ — `from nihai.ai import X`
zaten import sisteminin alt-modül çözümüyle çalışıyor (bu yüzden `registry`/`selfcheck`
hiç listede olmadan çalışıyordu); eager import yalnız ÖZNİTELİK erişimini destekliyordu,
onu da PEP 562 kancası karşılıyor.

- Tüm alt modüller lazy'ye çevrildi: **3854 → 11 ms**, ağır modül çekilmiyor (`sklearn`,
  `scipy`, `pandas`, `torch`, `statsmodels`, `matplotlib` → hiçbiri `sys.modules`'ta değil).
- `quant_math`'te `LedoitWolf`/`PCA` fonksiyon-içi lazy'ye alındı (yalnız 2 fonksiyonda
  kullanılıyor).
- `_adfuller` BİLEREK lazy YAPILMADI: kazanç yalnız 0,85 sn ve mevcut test sözleşmesini
  (`quant_math._adfuller is None` semantiği) kırma riski var.

### §3 — conftest geçici dizin sızıntısı

`conftest.py` her koşuda `mkdtemp` açıyor ama HİÇ silmiyordu: ölçüm anında %TEMP%'te
**479 birikmiş `nihai-pytest-*` dizini** (3,1 MB) vardı. `atexit` ile temizlik eklendi
(fixture olamaz: dizin pytest'ten ÖNCE, import anında yaratılıyor — `nihai.config` env'i
import anında okur). Birikmiş 485 dizin silindi; doğrulandı: yeni koşu 0 sızıntı.

### §4 — Denetim: temiz çıkanlar (uydurulmuş bulgu YOK)

`requests/httpx` timeout'suz çağrı: YOK (hepsi timeout'lu) · bare `except:`: YOK ·
değişebilir varsayılan argüman: YOK · `datetime.utcnow()`: YOK (BULGU-16 sonrası
kanonikleştirilmiş). Atomik olmayan 3 yazım var (`auto_screener`/`screen` CSV raporu,
`signal_engine` touch) ama hiçbiri Tur-12'de düzeltilen sıcak `state.json` yolunda değil
→ düşük öncelik, düzeltilmedi.

### Doğrulama (YERİNDE ölçüldü)

| | önce | sonra |
|---|---|---|
| pytest | **koşmuyordu** (toplama hatası) → kurulum sonrası 2178 test / 143 sn | **2186 geçti / 2 skip / 64,5 sn** |
| `import nihai.ai` | 3854 ms | **11 ms** |
| en yavaş test | 9,80 sn (`position_size`) | 4,12 sn (yeni hijyen testi, alt-süreçli) |
| %TEMP% sızıntısı | 485 dizin | **0** |
| ruff / mypy | temiz / (kurulu değildi) | **temiz / 148 dosyada 0 hata** |

Suite süresindeki **143 → 64,5 sn** düşüşün nedeni optimizasyon değil, **gerçek ağ
turlarının kalkması**; aynı zamanda suite artık Yahoo'nun keyfine göre dalgalanmıyor.

Yeni kalıcı testler (9): `tests/test_import_hygiene.py` (6 — lazy sözleşmesi + statsmodels
fail-soft) · `tests/test_net_guard.py` (3 — curl_cffi engellenir, loopback serbest, VIX
yükleyicisi etkisiz).

### AÇIK / KULLANICIYA

1. **Servisler HÂLÂ KAPALI** (bu turda da doğrulandı: koşan tek python süreci pytest'ti).
   Tur-25'in iki bulgusu (nöral köprü n=6, prior decay'in ölçülemeyişi) doğrudan buna bağlı.
   Açmak operatör kararı: `Nihai - Projeyi Ac`.
2. **BULGU-25 hâlâ AÇIK** ve bu tur onu ele almadı. Tur-25'in kendi önerisi "adayı OLMAYAN
   bir turda" idi — bu tur adaysızdı, ama kapsamı altyapıydı; `_stage_dsr`'ın doğru taban
   seçimi bir TASARIM kararıdır (sözde-pnl'de taban oranı %50 altındayken taban çıkarmak
   kapıyı GEVŞETİR) ve tek başına bir tur hak ediyor.
3. Devreden: `tur18-servis-tetik.ps1` + `tur20-shadow-gorev.ps1` YÖNETİCİ PowerShell'de bir
   kez; `ANTHROPIC_API_KEY` boş (FAZ AI-1 kilidi).
4. **Yeni ders:** bir bağımlılık yükseltmesi (yfinance → curl_cffi) mevcut bir korumayı
   SESSİZCE devre dışı bırakabilir. Guard'ın kendisi test edilmişti (BULGU-19'un dersi) ama
   test yalnız socket taşımasını kilitliyordu. **Koruma testi, korumanın KAPSADIĞI YÜZEYİ
   de kilitlemeli.**

---

## Ek Tur 28 — veri bütünlüğü + bayat kalibrasyon + kapı çelişkisi + uç nokta O(dosya) (2026-08-12, Opus 5 max)

**Baseline (yerinde ölçüldü, rapordan değil):** git temiz değil (Tur-27 commitsiz:
`nihai/ai/adaptive_calibration.py` + testi), ruff temiz, mypy 149 dosya temiz,
**2226 test yeşil / 72 sn**. Servisler KAPALI. Statik desen taramaları (mutable default,
bare except, TODO/FIXME) 27 turdan sonra BOŞ döndü — bu turda bulgular kodu OKUYARAK ve
ÇALIŞTIRARAK çıkarıldı.

### BULGU-27 (KRİTİK, veri bütünlüğü) — `atomic_io` sabit `.tmp` adı, eşzamanlı yazanda YARIM dosya yayınlıyor

`atomic_write_text` ara dosyayı hep `<dosya>.tmp` olarak açıyordu. Aynı hedefe yazan iki YAZAN
— canlı kayıtçı ↔ `research/forward_harness`, watchdog yeniden başlatırken üst üste binen iki
süreç, tek süreçte iki thread — AYNI ara dosyayı paylaşıyordu. Sıra
`A tmp'ye yaz → B tmp'yi KIRPIP yazmaya başla → A replace` olduğunda hedefe **B'nin yarım
içeriği ATOMİK olarak yayınlanır**: atomiklik korunur, içerik BOZUKTUR. Kapsam bu modülün
koruduğu en kritik dosyalar: `state.json` (portföy + açık pozisyonlar, ~60 sn'de bir),
`predictions_log.csv`, `ai_learning_state.json`, model kartları/artefaktları.

Bu, 2026-07-11'de bir günlük tahmin kaybettiren `os.replace` olayının KARDEŞ hatasıydı: o olay
kilidi, bu ise paylaşımı. Tur-11/Tur-12 birincisini kapatmış, ikincisi açık kalmıştı.

**KANITLANDI, varsayılmadı.** Eski davranış geçici bir fixture ile geri konup test koşuldu:
hedefe 2400 baytlık karışık bir dosya yayınlandı (geçerli içerik 1200 bayt olmalıydı), 4 ayrı
torn yayın yakalandı. Düzeltmeyle 0.

- Düzeltme: `atomic_io.tmp_path(dst)` ara dosyayı **pid + thread** ile ayırır; kalıcı hatada
  tmp SİLİNİR (benzersiz ad artık kendini üzerine yazmaz → Tur-26'daki tmp sızıntısı geri
  gelmesin).
- `atomic_replace`'i doğrudan çağıran 5 yerin hepsi (`prediction_tracker` x3, `tearsheet`,
  `models`, `neural`) elle tmp yolu kurmayı bıraktı → tek kaynak.
- Yeni testler: yazana-özel ad · 6 thread x 15 yazım altında **torn yayın YOK** · hata yolunda
  tmp kalmıyor.
- **Yan keşif (platform sözleşmesi, düzeltilmedi):** Windows'ta `os.replace` ile AÇIK bir
  okuyucu birbirini iter — okuyucu `PermissionError`, yazan tükenen yeniden-denemede `OSError`
  alır. İkisi de belgelenmiş davranış; test bunları AYRI sayar ve yalnız BÜTÜNLÜĞÜ ölçer.

### BULGU-28 — 7/24 süreçlerde kalibrasyon eğrisi BAYAT (CLAUDE.md'de açık madde olarak duruyordu)

`data/forecast_calibration.json` iki uzun-ömürlü süreçte de BİR KEZ okunuyordu:
`PredictionService.__init__` (`self._calib`) ve `web_dashboard._CALIB_CACHE`.
`calibrate_forecast.py` eğriyi yeniden ürettiğinde süreçler bunu ASLA görmüyordu — pano ve
canlı-tahmin sekmesi süreç yeniden başlayana kadar bayat "Olasılık (kal.)" gösteriyordu.
Tur-27 bunu docstring'inde yazmıştı ama düzeltmemişti.

- Düzeltme kök nedende: `_load_calibration` artık dosya imzasıyla (mtime_ns, size) kendi
  cache'ini tutar (`prediction_tracker._file_sig` deseni) → çağırmak hem TAZE hem ucuz.
  Panodaki yerel cache **SİLİNDİ** (net −10 satır); tahmin servisi döngü BAŞINDA tazeler
  (tur içinde sabit kalır ki aynı turdaki 30 sembol aynı eğriyi kullansın).
- Yeni testler: dosya değişince tazelenir · değişmediyse yeniden okumaz · yok/bozuk/boş → None.

### BULGU-29 — iki kalibrasyon kapısı AYNI veride ÇELİŞİK verdikt veriyordu (ACI DERS #9'un yarısı makinede kalmış)

`data/ai_calibrators.json` `pwin: KAYITLI` diyordu; Tur-27'nin taban-oran kontrollü kapısı ise
aynı veride `KAPI_GECILEMEDI`. Sebep: ACI DERS #9 ("bir kalibrasyon iyileşmesi raporlanmadan
önce SABİT-TABAN tahmincisiyle kıyaslanmalıdır") Tur-27'de yalnız `adaptive_calibration`'a
konmuştu; `calibration_loop.refresh` kapısı hâlâ kördü. CLAUDE.md bunu "yeniden-verdikt AÇIK"
diye işaretlemişti.

- Eşleştirilmiş Brier t-testi `learning.paired_brier_t`'ye taşındı (TEK kaynak; iki kapı da
  onu çağırıyor, `adaptive_calibration` kopyası silindi).
- `evaluate_source` artık ÜÇ yarışmacı puanlar: ham · kalibre · **sabit taban** (fit diliminin
  taban oranı). Kapı: ham'a karşı Brier VE ECE, tabana karşı eşleştirilmiş Brier farkı
  `t < −1,645` (Tur-24 Welch eşiğiyle aynı). Tabana karşı ECE ARANMAZ — sabit tahminci ECE'de
  yapısı gereği ~0'dır, kategori hatası olurdu.
- Gerekçe kuralına uygun: kapı sıkılaştırması **adayı OLMAYAN** bir turda yapıldı (Tur-23
  kuralı) ve yeni sonuç ÜRETEMEZ, yalnız yanıltıcıyı eler.

**ÖLÇÜLDÜ — gerçek artefakt yeniden verdiktlendi (`pwin`: KAYITLI → KAPI_GECILEMEDI):**

| yarışmacı | Brier | ECE | tabana karşı eşleştirilmiş fark |
|---|---|---|---|
| ham | 0,2951 | 0,1870 | — |
| **sabit TABAN** (0,4667) | **0,2530** | ~0 | — |
| platt | 0,2520 | 0,0603 | −0,00102 (**t=−1,42**, eşik −1,645) → GEÇEMEDİ |
| izotonik | 0,2725 | 0,0817 | +0,01950 (t=+2,27) → tabandan KÖTÜ |

n_oos=231. Yani platt ham'ı gerçekten düzeltiyor ama sabit tabanı **yenmiyor** — kazanç bilgi
değil, taban orana büzülme. İki bağımsız kapı (tek-bölme + yürüyen) artık AYNI şeyi söylüyor.
Eski artefakt `data/ai_calibrators.json.tur25-yedek`. `calibration_report()` de artık hangi
ayakta düşüldüğünü söylüyor ("iyileşmedi" ≠ "iyileşti ama tabanı geçemedi").
Yeni testler: bilgisiz-ama-aşırı-güvenli kaynak (y ⟂ p) ham'ı yener, tabanı yenemez,
KAYDEDİLMEZ · kaydedilen girdi taban kanıtını da taşır.

### OPT — `/api/forward-test` her istekte 6,7 MB / 53k satırı yeniden ayrıştırıyordu

Ölçüldü: `_read_log_rows` 163 ms + `report` 142 ms + `coverage` 49 ms ≈ **354 ms/istek**, ve
log büyüdükçe DOĞRUSAL artıyor. `pt.summary()` eklendi: report+coverage+verdict TEK okumadan,
anahtar = PRED_LOG imzası (mtime_ns, size). **Ham satırlar SAKLANMAZ** — 53k satır 70,3 MB
tutuyor (ölçüldü); saklanan yalnız ~3 KB'lık türetilmiş özet. Dosya değiştiği an kendiliğinden
tazelenir (bayat özet dönmez); eşdeğerlik birebir doğrulandı.

| ölçüm | önce | sonra |
|---|---|---|
| `report+coverage+verdict` tekrar çağrı | 385 ms | **0,163 ms** (~2360x) |
| **gerçek HTTP `/api/forward-test`** (TestClient) | 382 ms | **2,6 ms ort.** (10 istek; ~147x) |

`/api/predictions/accuracy` de aynı paylaşılan özete bağlandı (paylaşılan dict kopyalanarak —
önbellek kirletilmesin).

### BULGU-30 — düz seride `rsi()` 100 döndürüyor (doğrusu NÖTR 50)

`rsi()` son satırı `out.where(roll_dn != 0, 100.0)` idi: `roll_dn==0` görünce KOŞULSUZ 100
yazıyordu. Bu "hiç kayıp yok" durumunda DOĞRU, ama `roll_up` DA 0 ise (tamamen DÜZ pencere:
halt, illikit isim, seans-dışı bar) RS = 0/0 **belirsizdir** ve doğru cevap NÖTR 50'dir.
100 demek "azami aşırı-alım" demektir; `_bias_and_confidence`'a `clip((100−50)/25)=+1.0`
(azami boğa) bileşeni sokar ve `meta_features`'ın `rsi14`/`rsi_slope`'unu da kirletir —
yani ML eğitim verisine kadar gider.

**ETKİ ÖLÇÜLDÜ, ŞİŞİRİLMEDİ:** canlı özellik logunun **6374 satırının 0'ında** tetiklenmiş
(30 likit sembol her barda hareket ediyor). Yani latent bir hata; düzeltme mevcut ölçülen
veride BİT-DENK. Doğrulandı: düz → 50 · sadece artan → 100 (değişmedi) · sadece azalan → 0
(değişmedi) · ısınma penceresi NaN (maskelenmedi).

### BULGU-31 — eşitlik sessizce SHORT sayılıyordu: SIFIR bilgili girdi 0,60 güvenle SHORT üretiyordu

`_bias_and_confidence`'ın iki karşılaştırması da `>` idi (`e20 > e50`, `price > e20`), yani
EŞİTLİK −1.0 = tam SHORT sayılıyordu. Oysa dört satır aşağıdaki nötr dalı aynı eşitliği
`price >= e50` ile LONG'a veriyor — fonksiyon kendi içinde çelişikti.

Somut zarar (BULGU-30 düzeltilince ORTAYA ÇIKTI): tamamen düz bir seride bileşenler
`[−1, 0, −1]` olup bias = −0,667 ve **conf = 0,60 ile SHORT** üretiliyordu. Sıfır bilgi
taşıyan bir girdiye yüksek güvenli yönlü çağrı. (BULGU-30'dan ÖNCE sahte +1.0 RSI bileşeni
bunu kısmen maskeliyordu: bias −0,333 / conf 0,40. İki hata birbirini gizliyordu.)

Düzeltme: eşitlik → `0.0` (bilgi yok) → bias 0 → MEVCUT nötr dalına düşer.

| girdi (price=e20=e50, rsi=50) | önce | sonra |
|---|---|---|
| adx=20 | SHORT · bias −0,6667 · **conf 0,60** | bias +0,0010 · conf 0,20 (yalnız ADX'ten) |
| uçtan uca `_forecast(düz 80 bar)` | — | conf **0,00** (dürüst cevap) |

**BİT-DENKLİK KANITLANDI:** eşitlik içermeyen **20.000 rastgele girdide 0 fark**
(yön/bias/güven üçlüsü birebir). Yalnız `==` dalı yenidir → canlı log ve
`forecast_calibration.json`'ın alanı bozulmaz.

### ÖLÇÜLDÜ ve REDDEDİLDİ (dürüstlük kaydı)

- `_existing_keys()` `DictReader`→`csv.reader`+indeks: 214 → 152 ms (%29, çıktı birebir aynı).
  **Alınmadı:** kazanç günde birkaç kez 62 ms; karşılığında Tur-12'nin çift-kayıt guard'ına
  dokunmak gerekirdi. Bedeli faydasından büyük.
- `prediction_tracker` import 666 ms'nin 572'si `meta_features`→pandas. **Alınmadı:** süreç
  başına BİR KEZ ödeniyor, sıcak yolda değil (Tur-26'nın `nihai.ai` lazy'leştirmesi farklıydı:
  orası test paketinde ve her CLI'da tekrar tekrar ödeniyordu).
- `_sig_cache` sızıntısı ARANDI, YOK: `_symbols()` 30 ile sınırlı ve sabit küme.

### Kapı

ruff temiz · mypy 149 dosya temiz · **2235 test yeşil / 68 sn** (2226 → +9 yeni regresyon testi).

### Açık kalan (değişmedi)

BULGU-25 (`edge_lab/harness._stage_dsr` taban seçimi) hâlâ AÇIK — tasarım kararı, kendi turunu
hak ediyor. Servisler KAPALI; `ANTHROPIC_API_KEY` boş.

### Yeni dersler

1. **Bir kural iki kapıdan yalnız birine konursa, ikisi aynı veride çelişik verdikt verir** ve
   eski/yanlış olanı okuyan operatör aynı tuzağa düşer. Kuralı ORTAK bir yardımcıya koy.
2. **Atomik yazımda ara dosya adı YAZANA özel olmalı.** `os.replace` atomikliği tmp'nin
   tekliğini garanti etmez; paylaşılan tmp "atomik ama bozuk" yayın üretir.
3. **Uzun-ömürlü süreçte diskten okunan her şey TAZELİK sözleşmesi ister.** `__init__`'te bir
   kez okumak, dosyayı üreten CLI ile süreci sessizce ayrıştırır.
4. Bir düzeltmenin gerçekten hata yakaladığını kanıtlamak için **eski davranışı geçici olarak
   geri koy ve testin KIRILDIĞINI gör** — yoksa test boş olabilir.

---

## Ek Tur 30 — pano ↔ backend bütünlüğü: ölü panel, hep-yanlış sağlık kartı, atılan önbellek (2026-08-14, Opus 5 max)

**Baseline (yerinde ölçüldü, rapordan değil):** git kirli (Tur-27/28/29 hâlâ commitsiz),
**2272 test yeşil**, ruff temiz, mypy 92 dosya temiz. Servisler KAPALI (8765/8799 dinlemiyor).
Kapsam bu turda BİLEREK dar: yalnızca backend ile web panosu (`nihai/web/*` + panonun
okuduğu sözleşmeler) arasındaki uyumsuzluklar; sinyal/kenar iddiasına dokunulmadı.

**Yöntem.** 49 uç ile panonun 45 `apiCall` çağrısı makineyle eşleştirildi; 226 `getElementById`
kimliği ve 50 satır-içi `onclick` hedefi HTML'e karşı denetlendi (ikisi de EKSİKSİZ çıktı,
0 kopuk referans); sonra iki taraf da elle okundu. Uç gecikmeleri `TestClient` ile ölçüldü.
Panonun GERÇEK JS fonksiyonu, GERÇEK backend yanıtıyla Node'da koşturularak A/B alındı.

### BULGU-28 (KRİTİK) — "Kanıt & Doğrulama" sekmesinin forward-test paneli HİÇ dolmuyordu

`loadEvidence()` içindeki koşul `success` alanını KARŞILAŞTIRMADAN ÖNCE değilliyordu.
JS'te `!` `===`'den önce bağlanır: ifade `(!undefined) === false` → `true === false` →
**her zaman false**. Uç 200 dönüyor, konsola hata düşmüyor, panel sessizce boş kalıyordu.

Kaybedilen şey projenin kendi ifadesiyle **#1 önceliği**: `/api/forward-test` şu an
59.280 kayıt / 56.727 çözülmüş / 50,15 günlük kapsama ve `NO_EDGE` verdiktini üretiyor —
operatör bunların hiçbirini göremiyordu.

**A/B ile kanıtlandı (varsayılmadı).** Fonksiyon HTML'den çıkarılıp sahte-DOM ile canlı
uca karşı koşturuldu: eski koşulla DOM'a yazılan alan **2** (yalnız statik ders/başlık
listesi), yeni koşulla **15** (verdikt NO_EDGE · %47,1 isabet · 56.727 çözülmüş ·
meta-kapı GATE_HURTS n=927/5951/6878 · kapsama tablosu · ilerleme %100).

Regresyon kilidi: `test_html_karsilastirmadan_once_degilleme_yok` bütün pano JS'ini
`!x === y` deseni için tarar (aynı hatanın ikinci kopyası da yakalanır).

### BULGU-29 — "Tahmin Servisi" sağlık kartı, servis AYAKTAYKEN bile daima "durdu"

`/api/system/health` üç alanı da VAR OLMAYAN anahtarlardan okuyordu:
`snap.get("running")`, `snap.get("rows")`, `snap.get("updated_at")`. `snapshot()`
sözleşmesi ise `{"meta", "predictions"}`. Sonuç: kart sabit `durdu · 0 sembol · —`.

Servislerin sessizce ölmesi bu projenin bilinen arızası (Tur-13/18/25 kayıtları) olduğu
için hep-yanlış bir sağlık kartı en zararlı türden hatadır: operatörü kartı görmezden
gelmeye alıştırır. Düzeltme kaynağında: `PredictionService.running` (thread canlı mı)
eklendi, `snapshot()` onu yayımlıyor, uç doğru anahtarları okuyor. Test sahte-canlı
thread enjekte edip AYAKTA hâlini de doğruluyor (iki taraf da False iken eşitlik hiçbir
şey kanıtlamaz).

### BULGU-30 — kapasite merdiveni v2'de donmuştu (nöral katmanlar panoda yok)

`ai_overview` `param_ladder_report()`'u argümansız çağırıyordu; varsayılan
`include_neural=False`. Pano merdiveni `mlp` (3.777) satırında bitiyor, HEMEN ALTINDAKİ
model-kartı tablosu ise `deep_mlp`/`quant_transformer` kartlarını listeliyordu — yani
Tur-16→22'nin bütün kapasite tartışması panoda görünmüyordu. Artık 7 satır:
… `deep_mlp` 186.702 · `quant_transformer` 5.392.659 · `quant_transformer_xl` **30.363.883**
(CLAUDE.md'deki 30,4M ile birebir). torch yoksa satır `param_count=None` + `note` ile
fail-soft gelir; pano `0` yerine `—` basar (0 = "sıfır parametreli model" diye okunurdu).

### Ölçülen optimizasyonlar (hepsi ÖNCE/SONRA ölçüldü, tahmin yok)

| Uç / iş | Önce | Sonra | Not |
|---|---|---|---|
| `/api/var` (sıcak) | 2845 ms | **11,9 ms** | ~240× |
| `/api/system/health` | 1616 ms | **261 ms** | ~6,2× |
| `/api/ai/overview` | 194 ms | **9,4 ms** | ~20×, üstelik 3 satır DAHA fazla veriyle |
| pano 15 sn'lik ikincil blok | 5 seri gidiş-dönüş | **1** | `Promise.all` (ana döngüde zaten öyleydi) |

Kök nedenler:

1. **`_analysis_provider()` "paylaşılan" provider'ı her istekte YENİDEN kuruyordu.**
   Tahmin servisi kapalıyken yedek dal `get_provider(..., with_cache=True)` çağırıyor, bu
   da **boş sözlüklü yeni bir `CachingProvider`** üretiyordu → önbellek her HTTP isteğinde
   çöpe. `CachingProvider` docstring'i "thread-safe, web thread + ana döngü aynı provider'ı
   çağırır" diyor; yani paylaşılmak üzere yazılmıştı, paylaşılmıyordu. Kanıt: aynı süreçte
   ikinci `compute_portfolio_var` 7 ms, aynı isteği HTTP'den tekrarlamak 2845 ms.
   Etki alanı yalnız VaR değil: `/api/analyze`, `/api/bars`, `/api/quote`, `/api/intel/*`,
   `/api/options/chain`, `/api/indicators/extended` hepsi aynı yedek daldan geçiyor.
2. **Tur-28'in imza-önbellekli özeti `api_extra`'da atlanmıştı.** `system_health` ham
   `pt.coverage()` çağırıyor, her istekte 7,5 MB / 59k satırı yeniden ayrıştırıyordu
   (ölçüldü: ~400 ms, log büyüdükçe doğrusal). `pt.summary()["coverage"]` AYNI sözlüğü
   tek `stat()` ile veriyor. Test ham yolu patlatarak bunu kilitliyor.
3. **Kapalı bir loopback portu Windows'ta zaman aşımının TAMAMINI yiyor** (ölçüldü:
   624 + 605 ms). İki yoklama seriydi ve 0,6 sn zaman aşımlıydı → "servis kapalı" hâli,
   yani sorunun tam anı, ucun EN YAVAŞ hâliydi. Artık paralel + 0,25 sn (ayakta bir yerel
   uvicorn'u OS listen-backlog'u milisaniyede kabul eder; ~50× pay).
4. **`param_ladder_report` her çağrıda gerçek işi tekrar yapıyordu:** `gbm_small` için
   sabit-seed sentetik fit (186 ms) + XL için 30,4M parametrelik torch modülünün yeniden
   inşası (171 ms). İkisi de skalerlerin saf fonksiyonu → `lru_cache`.

### Küçük düzeltmeler

- `renderConfig` diğer render'ların guard desenine alındı: `apiCall` hatasında risk formunun
  TÜM alanları `NaN`'a dönüyordu (operatör kaydederse 422).
- `loadBrain`/`loadEvidence`/`loadMacro` tembel-yükleme bayrağını başarısızlıkta geri alıyor;
  tek ağ hıçkırığı sekmeyi oturum boyunca boş bırakmıyor.
- `filterSignals` artık `escapeHtml` kullanıyor (`renderLogs`/`renderTrades` zaten kullanıyordu):
  `<` içeren bir gerekçe innerHTML'de etiket başlatıp satırı yutuyordu.
- `loadSystem`'in ölü `_sysLoaded` dalı silindi (hiçbir şey yapmıyordu; sağlık zaten HER
  açılışta tazelenmeli — gerekçe yorumda).

### Değiştirilmedi (bilinçli)

- **`/api/bars` panodan hiç çağrılmıyor** (grafik barları `/api/analyze` yanıtına gömülü
  geliyor). Uç çalışıyor ve ucuz; canlı grafiğe yeni-bar ekleme bir ÖZELLİK olurdu, hata
  değil → rapor edildi, yazılmadı.
- `/api/ai/capacity` overview'daki merdivenle örtüşüyor ama `n_features` parametreli ve
  testli — bırakıldı.
- `ai/overview.selfcheck` ve `.memory` panoda kullanılmıyor; sözleşme testi bunları
  koruduğu için kaldırılmadı.
- Panoda gösterilmeyen ama hesaplanan küçük alanlar: `events.days_to_ex_div`,
  `atm.iv_rank_n`, `services.web_panel`. Zararsız; kart eklemek istenirse hazır.

### Doğrulama

**2275 test yeşil** (2272 + 3 yeni), ruff temiz, mypy 92 dosya temiz. Pano ayrıca canlı
verinin KOPYASIYLA (`NIHAI_DATA_DIR`/`NIHAI_LOG_DIR` sandbox'a yönlendirilmiş, tahmin
servisi başlatılmadan) 8766'da servis edilip 28 GET ucu tek tek yoklandı: hepsi 200,
hiçbirinde `ok:false` alt-blok yok. Canlı `data/`, `logs/` dosyalarına YAZILMADI.

### Yeni dersler

1. **Sessiz her-zaman-yanlış bir koşul, gürültülü bir hatadan kötüdür.** `!x === y` uç 200
   dönerken paneli öldürdü; ne konsolda iz vardı ne testte. Panonun DOM'a kaç alan yazdığını
   ölçen bir A/B, "sayfa açılıyor mu?" testinden çok daha keskin.
2. **Anahtar adı sözleşmedir.** `snapshot()`'ın döndürmediği bir anahtarı `.get(k, False)`
   ile okumak hatayı sessiz bir YANLIŞ DEĞERE çevirir. Sağlık yüzeylerinde `.get`
   varsayılanı, eksik anahtarı "sistem kapalı" diye raporlayan bir yalana dönüşebilir.
3. **"Cache'li kur" ≠ "cache'i paylaş".** Fabrika fonksiyonunu her istekte çağırmak
   önbelleği her istekte sıfırlar; sarmalayıcının docstring'i paylaşımı vaat ediyorsa
   nesnenin ÖMRÜ de o vaadi tutmalı.
4. **Saf olmayan fonksiyonu önbelleğe alma.** `theoretical_param_count` argümanlarının yanı
   sıra modül-global `TORCH_AVAILABLE`'ı da okuyor; tamamını `lru_cache`'lemek monkeypatch'li
   testte `None`'u sonraki çağrılara sızdırdı (test yakaladı). Önbellek yalnız saf ve pahalı
   çekirdeğe konuldu — doğrulamalar ve bayrak dalı dışarıda kaldı.

---

## Ek Tur 33 — dejenere girdi denetimi + gösterge hattı optimizasyonu (2026-08-15, Opus 5 max)

Tam rapor: `docs/TUR33_RAPORU_2026-08-15.md`.

**Baseline (yerinde ölçüldü).** Ağaç `af399ea` üstünde, Tur-32 hâlâ commit'siz.
**2399 test yeşil**, ruff temiz, mypy 95 dosya temiz. Yüzey tamamen yeşildi — bu turun
bulguları hiçbir mevcut kapının GÖRMEDİĞİ kusurlardır.

**Yöntem.** Projenin kendi Tur-28 dersi ("göstergeyi dejenere girdilerle SINA") bir denetim
aracına çevrilip o dersin uygulanmadığı iki yüzeye koşuldu: v5 modülleri ve gösterge paneli.

### BULGU-33-1 (KRİTİK) — pano SIFIR bilgiden "GÜÇLÜ SAT" üretiyordu

Tamamen düz 120 barlık pencere (halt / illikit / seans-dışı) → `indicator_panel`:

| | bull | bear | nötr | skor | etiket |
|---|---|---|---|---|---|
| **öncesi** | 11 | **46** | 17 | **−0,542** | **GÜÇLÜ SAT** |
| sonrası | 0 | 0 | 73 | +0,000 | YETERSİZ ÖLÇÜM |

Birbirini besleyen BEŞ kök neden: (1) iki modülde **51** karşılaştırma `1 if x > y else -1`
— eşitliği ve NaN'ı ayı sayar (Tur-28 aynı hatayı `_bias_and_confidence`'ta düzeltmiş, kardeş
çağıranlara bakmamıştı); (2) `mfi` 0/0 → 100 (`rsi` kusurunun birebir kardeşi); (3) `spearman`
tam-bağlı pencerede **+50** uyduruyor; (4) `supertrend` ATR=0'da −1 + yön dizisini 1.0 ile
ilklendiriyor; (5) `%B` span=0'da 0,5 uyduruyor.

Kural TEK KAYNAKTA: `strategy.vote(x, y)` (üç durumlu) + `strategy.note3`. Ek olarak
Tur-23'ün "`n_kept` oku" dersi panoya taşındı: `directional < 10` → "YETERSİZ ÖLÇÜM";
`directional` dönüş sözleşmesine ve pano kartına eklendi.

**Normal veride BİT-DENK:** `bull=39 · bear=26 · nötr=18 · skor +0,200` değişmedi.

### BULGU-33-4 — Corwin-Schultz dejenere rejimde MAKAS UYDURUYORDU (Tur-32 açık kalem #6)

Kırpma `E[max(S,0)]` simetrik gürültüden bile pozitif sayı üretir. Gerçek makası SIFIR olan
süreçte: `spread_raw = −0,000607` (ölçülebilir makas YOK) iken kırpılmış **+0,000140** =
"1,4 bps makas"; negatif oran **%61,7**. Sözleşme `roll_spread` ile eşitlendi: yansız tahmin
pozitif değilse `spread` **None** + `reliable=False`; `spread_clipped` ve `negative_share`
ayrıca raporlanır. Gerçek %2 makaslı seride `reliable=True` kalır (test kilitler).

### BULGU-33-5 — `roll_spread(n=3)` üç RuntimeWarning basıyordu

`np.cov(..., ddof=1)` tek gözlemli dilimle çağrılıyordu. Sonuç doğruydu (`None`) ama 7/24
süreçte defteri kirletiyor, `-W error` altında çökerdi. Alt sınır 3 → 4.

### Optimizasyon — profil temelli, oy-denkliği ÖLÇÜLEREK

Denetim `indicator_panel`'i profillemeye götürdü: sürenin **%25'i** `spearman`'daydı
(pencere başına `pd.Series` + `.rank()`). Aynı desen beş yerde daha vardı.

| birim | önce | sonra | kazanç | denklik |
|---|---:|---:|---:|---|
| `spearman(10)` n=4000 | 306,52 ms | **3,07 ms** | **100×** | **BİREBİR** (5 senaryo) |
| `_linreg(20)` n=5000 | 68,69 ms | **0,20 ms** | **343×** | ~1e-15 mutlak |
| `wma(20)` n=5000 | 13,77 ms | **0,14 ms** | **98×** | bağıl ≤ 4,9e-16 |
| `FTRL.fit_stream` 5000×96 | 296,2 ms | **133,2 ms** | **2,22×** | **BİT-DENK** |
| `adwin` n=6000 | 554,6 ms | **316,3 ms** | **1,75×** | **BİT-DENK** |
| **`indicator_panel` 300 bar** | 87,8 ms | **49,0 ms** | **1,79×** | 0 oy farkı |
| **`indicator_panel` 1000 bar** | 176,7 ms | **61,7 ms** | **2,86×** | 0 oy farkı |
| **`indicator_panel` 3000 bar** | 439,7 ms | **94,4 ms** | **4,66×** | 0 oy farkı |

`FTRL.fit_stream` örnek başına `predict_one`ı iki kez çağırıyor, `update_one` ayrıca kendi
`weights()`ini alıyordu → **3× `weights()`**; ortak `_step` ile 1×. `adwin` her gözlemde
`np.asarray(window)` çağırıyordu (profilin **%34'ü**) → numpy tamponunda bitişik görünüm.
`_aroon_osc`'un KOPYA uygulaması silinip çekirdeğe bağlandı.

**"Bit-denk" nerede İDDİA EDİLMEDİ:** `wma`/`_linreg`/`cog` toplama sırasını değiştirir
(bağıl ≤ 7,5e-16 ≈ 2-3 ULP). İddia "operasyonel denk"e taşındı ve ÖLÇÜLDÜ — 300 rastgele df
(3 fiyat tabanı × 3 oynaklık × 3 sürüklenme) · **24.900 panel satırı: 0 oy farkı, 0 konsensüs
farkı, 0 yuvarlanmış değer farkı**. Bu üç fonksiyon yalnız gösterim yolundadır (grep ile
doğrulandı), karar yolunda kullanılmaz.

### Değiştirilmedi (bilinçli)

- **IV Rank bandının yönü** — ölçülmüş ama kullanıcı kararı bekleyen kalem (ACI DERS #3).
- **`sabr_calibrate` sırasız strike** — `np.interp` sıralı `xp` bekler; ölçüldü, başlangıç
  tahmini `alpha` 2,03668 ↔ 2,03666 oynuyor, optimizasyon aynı noktaya yakınsıyor
  (rmse farkı < 1e-9) → gözlenen etkisi yok, raporlandı.
- **`vpin` kova taşması** ve **`adwin` O(W) karmaşıklığı** — ikisi de modül docstring'inde
  gerekçeli; sabit çarpan düşürüldü, karmaşıklık aynı.

### Doğrulama

**2427 test yeşil** (2399 + 28 yeni), ruff temiz, mypy 95 dosya temiz. Ağ yok; canlı
`data/`, `logs/` dosyalarına yazılmadı.

### Yeni dersler

1. **Bir ders MODÜLE değil DESENE uygulanır.** Tur-28 kuralı bulduğu tek fonksiyonda
   düzeltti; aynı desen iki komşu modülde 51 kez yaşadı. Kök neden bulununca `grep` ile
   DESENİN tüm örnekleri aranmalı.
2. **İki hata birbirini maskeler; birini düzeltmek tabloyu KÖTÜLEŞTİRMİŞ gösterir.** 51
   karşılaştırma düzeltilince düz seride tek ayı oy kaldı (SuperTrend) ve skor −0,542'den
   **−1,000**'e "kötüleşti" — payda küçüldüğü için. Skor tek başına değil PAYDASIYLA okunur.
3. **Kırpma bir ölçüm değil KARARDIR.** Simetrik gürültüyü pozitif sayıya çevirir; yansız
   tahmin negatifken kırpılmışı "ölçüm" diye raporlamak ölçüm yokluğunu ölçüm gibi sunmanın
   en sinsi biçimidir — çünkü sayı makul görünür.
4. **"Bit-denk" iddiası ölçülebilir olmalı; olmuyorsa iddia değiştirilir.** "24.900 satırda
   0 oy farkı" doğrulanabilir bir cümledir; "herhalde önemsizdir" değildir.
5. **Profil tahmini yener — hem de kendi kodunda.** 10 barlık bir sıra korelasyonu "ucuz"
   görünür; pahalı olan matematik değil, pencere başına kurulan `pd.Series`'ti.

---

## Tur-34 — AI v6 "Belirsizlik ve Kapasite" (2026-08-15)

Tam rapor: `docs/AI_V6_TUR34_RAPORU_2026-08-15.md`

### Ne eklendi (hepsi KARAR-DIŞI, kenar iddiası YOK)

| Ne | Nerede | Hangi boşluk |
|---|---|---|
| Gauss HMM — rejim GEÇİŞ olasılıkları | `nihai/ai/regime_hmm.py` (YENİ) | `knowledge.py`: "geçiş OLASILIĞI bilgisi yoktur" |
| Clayton/Gumbel kopulalar + `copula_select` | `nihai/ai/quant_math.py` | `knowledge.py`: "asimetrik kopulalar YOKTUR" |
| PIN (EKOP MLE, Lin-Ke çarpanlaması) | `nihai/ai/microstructure.py` | `knowledge.py`: "PIN tahmincisi YOKTUR" |
| SSVI + `ssvi_diagnose` | `nihai/ai/options_math.py` | Tur-32 açık kalem #5 |
| `quant_transformer_moe_xl` (≈284M) + `moe_routing_stats` | `nihai/ai/neural.py` | kullanıcı isteği · Tur-32 açık kalem #3 |

Kapasite merdiveni (f=96): 223.054 → 5.465.363 → 31.756.051 → 93.228.307 →
**283.639.571**. `recommend_tier` eşiği n ≥ 3.600.000 (aynı 3× oran kuralı).

### Optimizasyon — ÖLÇÜLDÜ

Kök neden TEK: `scipy.special.logsumexp` (2,2) dizide çağrı başına **151,3 µs**
(array-API gönderimi); elle yazılmış aynı aritmetik **11,2 µs**. `gaussian_hmm_fit`
T=2000'de 86.021 kez çağırıyordu.

| fonksiyon | ÖZGÜN | YENİ | kazanç | denklik |
|---|---:|---:|---:|---|
| `regime_hmm._forward` | 294,7 ms | 15,6 ms | **18,9×** | bağıl 1,5e-14 |
| `regime_hmm._backward` | 294,0 ms | 15,6 ms | **18,8×** | bağıl 1,5e-14 |
| `regime_hmm.hmm_filter` | 893,7 ms | 24,9 ms | **35,9×** | argmax farkı **0/2000** |
| `microstructure.pin_ekop` | 600,9 ms | 200,6 ms | **3,00×** | PIN farkı ≤ 4,4e-06 |
| `indicators_ext.nvi` | 9,61 ms | 0,37 ms | **26,0×** | **0,000e+00** |
| `indicators.supertrend` | 12,01 ms | 2,43 ms | **4,93×** | **0,000e+00** |
| `indicators_ext.kama` | 3,08 ms | 1,33 ms | **2,31×** | **0,000e+00** |
| **`indicators.indicator_panel`** | **139,7 ms** | **85,1 ms** | **1,64×** | (yukarıdakilerin bileşimi) |

Üretim yolu denkliği 300 rastgele df ile ölçüldü (üç fiyat tabanı × üç oynaklık × üç
sürüklenme); NaN deseni farkı 0. `import nihai.ai` 8,2 ms (Tur-26 bütçesi korundu).

### Doğrulama

**2459 test yeşil** (2427 + 32 yeni), `ruff check .` temiz, `mypy nihai` 96 dosya temiz.
Ağ yok; canlı `data/`, `logs/` dosyalarına yazılmadı.

### Yeni dersler

1. **Sarmalayıcının maliyeti, sardığı matematiğin 13 katı olabilir.** Sıkı bir döngüde
   çağrılan her genel-amaçlı yardımcı şüphelidir. Tur-33 bunu `pd.Series`'te öğrenmişti;
   ders MODÜLE değil DESENE aittir.
2. **"Vektörleştirilemez" ≠ "optimize edilemez".** Özyinelemeli `supertrend`/`kama` sıralı
   kaldı; `np.isnan(x)` → `x != x` ve yerel değişken ile 4,93× ve 2,31×, tek aritmetik
   işlem değişmeden.
3. **Bir ÇARPIM birikimi `cumprod`dur ve bit-denktir; bir TOPLAM birikimi değildir.** `nvi`
   26× hızlandı ve fark TAM SIFIR çıktı. Hangi denklik iddiasının kurulabileceğini
   ALGORİTMA belirler, umut değil.
4. **Ölçüm KOŞULU, ölçümün kendisi kadar taşınmalıdır.** `causal=False` · `proxy=True` ·
   `reliable=False` · `measured=False`+`reason` · `decisive=False` — beşi de sayıyı
   okuyanın onu ne sanabileceğini engeller.
5. **Aile/durum-sayısı seçimi bir MODEL KARARIDIR.** `copula_select` (AIC) ve
   `hmm_select_states` (BIC) veriye sorar ve Δ<2'de "ayırt edemedim" der.

---

## Tur-35 — AI v7 "Etiketsiz Veri, Dağılım ve Ölçek" (2026-08-16, Opus 5 max)

Rapor: `docs/AI_V7_TUR35_RAPORU_2026-08-16.md`.
Kullanıcı isteği üç kalemliydi: AI'ı geniş kapsamlı geliştir · parametre sayısını artır ·
sonunda detaylı optimizasyon yap.

### Yetenek

| Ne | Nerede | Not |
|---|---|---|
| Maskeli ÖZ-DENETİMLİ ön-eğitim | `neural.NeuralCapacity.pretrain` | **etiket gerektirmez** — ACI DERS #5'in veri tarafına dokunan ilk kol |
| ↳ ÖLÇÜLDÜ (canlı 1m, purged 5-fold) | | **hipotez doğrulanmadı**: AUC 0,4896±0,0167 → 0,4881±0,0135 (3/5 fold = yazı-tura). Rekonstrüksiyon canlıda ~hiç düşmedi (2 fold'da 0), sentetikte %19 → mekanizma çalışıyor, canlı vektörde yapı kalmamış |
| Dağılımsal oynaklık `vol_q` | `neural` (yalnız ultra) | pinball kaybı; çapraz geçiş MİMARİYLE imkânsız |
| `quant_transformer_ultra` | `neural` | 283.639.571 → **771.874.072** (2,72×), 40 ince-taneli uzman |
| Çevrimiçi kaynak füzyonu | `learning.source_fusion` | Tur-32 açık kalem #4 (iki tur ertelenmişti) |
| Ortak dizi kodlayıcı | `neural._SeqEncoderMixin` | dört `forward` kopyası tekilleşti · **62 tensörde fark 0** |

### Optimizasyon — profil ÖNCE, kod SONRA

Profil iki suçlu gösterdi ve ikisi de Tur-33'ün *"bir ders MODÜLE değil DESENE uygulanır"*
dersinin tekrarıydı: (a) Tur-10'un `research/`'te düzelttiği AUC beraberlik döngüsü
`nihai/ai/neural.py`'de yeniden doğmuştu; (b) doğru artımlı sayım deseni
`reasoner._count_data_lines`'ta dururken 20 satır aşağıdaki `_tail_lines` kuadratikti — ve
o kuadratik gövde ÜÇ modülde birebir kopyaydı.

| fonksiyon | ÖZGÜN | YENİ | kazanç | denklik |
|---|---:|---:|---:|---|
| `neural._auc_np` (n=200k, sürekli) | 166,97 ms | 21,10 ms | **7,91×** | **bit-fark 0** (30 senaryo) |
| `neural._auc_np` (n=5k, sürekli) | 3,87 ms | 0,32 ms | **12,08×** | **bit-fark 0** |
| `neural.evaluation_metrics` (n=200k) | 377,3 ms | 35,0 ms | **10,8×** | (bileşim) |
| `online.hedge_combine` (n=6000) | 109,3 ms | 37,9 ms | **2,88×** | **fark 0,0** (5 senaryo) |
| `atomic_io.tail_lines` (n=5000) | 26,5 ms | 2,73 ms | **9,7×** | çıktı-denk (12 senaryo) |
| `atomic_io.tail_lines` (n=100k) | 2666,6 ms | 31,1 ms | **85,7×** | çıktı-denk |
| **`learning.update_from_predictions`** (62k satır) | **3151 ms** | **544 ms** | **5,80×** | üretim yolu |
| `neural` param merdiveni (`meta` aygıtı) | 7,58 s / ~4,5 GB | 3,02 s / **~0 GB** | 2,51× | sayı birebir (6 basamak) |
| `indicators.ultimate_oscillator` | 2,833 ms | 1,581 ms | **1,79×** | **0,000e+00** (300 df) |
| `indicators_ext.williams_ad` | 2,066 ms | 0,153 ms | **13,48×** | **0,000e+00** |
| `indicators_ext.swing_index` | 1,730 ms | 0,348 ms | **4,98×** | **0,000e+00** |
| **`indicators.indicator_panel`** | **85,4 ms** | **70,0 ms** | **1,22×** | (yukarıdakilerin bileşimi) |

**Ölçüldü ama DEĞİŞTİRİLMEDİ** (dürüstlük kalemi — sonraki tur aynı yeri yeniden kazmasın):
MoE FFN yönlendirme yükü toplamın yalnız **%3,4'ü** (gruplu matmul bit-denkliği %3,4 için
satardı) · `predict_with_uncertainty` süresinin **%84'ü** `torch.dropout` RNG'sinde, T
geçişini yığmak RNG tüketim sırasını bozar · `_ece_np` kova döngüsünü `bincount`la
vektörleştirmek toplama sırasını değiştirir ve `_auc_np` düzeltmesinden sonra baskın değil.

### Doğrulama

**2595 test yeşil** (Tur-34 sonu 2459 → **+136**), `ruff check .` temiz, `mypy nihai`
96 dosya temiz. Ağ yok; canlı `data/`, `logs/` dosyalarına yazılmadı (canlı ölçümler
kopya üzerinde). Yeni dosyalar: `tests/test_tur35_parity.py`, `tests/test_tur35_ai.py`.

### Yeni dersler

1. **Bir kusuru düzeltirken `grep` ile AYNI DESENİ tüm repoda ara.** Düzelttiğin dosya
   genelde tek kopya değildir — bu turda AUC döngüsü 1 yerde, kuyruk okuyucu 3 yerde,
   Tur-6'nın `pd.concat(...).max(axis=1)` deseni 4 yerde çıktı (üçü de AYNI turda).
2. **Biriktirirken yeniden sayma; sonda bir kez birleştir.** `data = yeni + data` +
   `data.count(...)` deseni O(k²) bayt taratır (8,4 MB dosyada 4,3 GB).
3. **`lst[-0:]` TÜM listedir.** "Sıfır" sanılan sınır sessizce her şeyi döndürüyordu ve sonuç
   BLOK BOYUTUNA bağlıydı = tanımsız davranış. Sıfır/negatif sınırı AÇIKÇA ele al.
4. **Ölçüm bütçesini yalnız YAZAN değil OKUYAN da öder.** 772M'lik yeni basamak panonun her
   Beyin sekmesinde 3,1 GB + 5,2 sn demekti; `torch.device("meta")` sayıyı değiştirmeden
   maliyeti sıfırladı. Yeni kapasite eklerken "bu sayıyı kim, ne sıklıkta soruyor?" mimarinin
   parçasıdır.
5. **Ön-eğitilmiş gövde EĞİTİLMİŞ model değildir.** `pretrain` `is_fitted`'i açmaz; etiket
   görmemiş gövdeden `p_win` okumak, ölçüm yokluğunu ölçüm gibi sunmanın (Tur-31 IV Rank)
   sinir-ağı karşılığı olurdu.
6. **Kısıtı kayba değil MİMARİYE yaz** — ve kurulabilen iddiayı söyle: kümülatif softplus
   çapraz geçişi imkânsız kılar, ama garanti "azalmayan"dır, "kesin artan" değil (float32
   alt-taşması komşu quantile'ları eşitleyebilir).

## Tur-43 — Tur-7 önbelleğinin ULAŞMADIĞI üç yer + gösterge çekirdeğinin vektörleştirilmesi (2026-08-22/23, Opus 5 max)

Kullanıcı isteği tek kalemliydi: *"projeye detaylı optimizasyon yapar mısın"*. Kural gereği
**profil ÖNCE, kod SONRA** — hiçbir değişiklik tahminle yapılmadı.

### Profil: 4000 barlık gerçek 1h backtest (`data/alpaca_hist/A_1h.csv`, cProfile)

```
16,05 s toplam
 ├─ generate_signal              15,36 s (%96)   3792 çağrı
 │   ├─ meta_gate.probability     7,88 s (%49)    398 çağrı → çağrı başına 19,8 ms
 │   ├─ detect_setup→detect_event 3,77 s (%23)   3792 çağrı
 │   └─ volatility_explosion_f.   2,28 s (%14)   3792 çağrı
 └─ .iloc.__getitem__            4,75 s cum.   85.175 çağrı
```

398 kapı çağrısının 7,88 saniye yemesi tek başına teşhisti: kapı bar başına değil
**sinyal başına** çalışır, yani maliyeti dilim UZUNLUĞUNDAN gelmek zorundaydı.

### Bulgular — üçü de ACI DERS'in aynı satırının tekrarı: *"bir ders MODÜLE değil DESENE uygulanır"*

**BULGU-43-1 · Tur-7 önbelleğini kullanmayan TEK kardeş çağıran.** Tur-7 (2026-07-03)
göstergeleri sembol başına bir kez hesaplayıp her bara kırpılmış görünüm vererek O(n²)→O(n)
inmişti. `meta_features.extract_features` bu önbelleği **hiç görmedi**: EMA20/50/200 + ATR14
+ BB20 + RSI14 + ADX14'ü büyüyen dilim üzerinde SIFIRDAN hesaplıyordu. `generate_signal`
kapıyı `gate.probability(df_htf, final)` diye çağırıyor, elindeki `shared`'ı geçmiyordu.
Aynı kusur canlıda daha da pahalıydı: `live_predict_service._bar_context` bu beş göstergeyi
**üç kez** (generate_signal'in kurup ATTIĞI iç önbellek + ctx satırları + extract_features),
pano `/api/analyze` ise **dört kez** hesaplıyordu (`gate.explain` kendi içinde
`extract_features`'ı çalıştırıp sonra `self.probability`'yi çağırarak İKİNCİ kez çalıştırıyordu).

**BULGU-43-2 · `dropna()` yalnız SAYMAK için çağrılıyordu — altı yerde.** `x.dropna().empty`
ve `len(x.dropna()) >= k` yeni bir Series **ve** yeni bir Index tahsis eder; sorulan soru ise
sadece "kaç tane NA-olmayan var?"dır. `count()` aynı cevabı tahsissiz verir. Kardeş desen:
"son geçerli değeri al" için `s.dropna().iloc[-1]` yazan **dört ayrı `_last`/`_last_valid`
kopyası** (`indicators`, `indicators_ext`, `live_predict_service`, pano) — panel satır başına
en az bir kez çağırıyor, yani 1500 barlık bir df'de yüzlerce O(n) tahsis.

**BULGU-43-3 · Tur-7 HESABI tekilleştirmişti, KIRPMAYI değil.** `_SharedSlice.ema(p)` her
erişimde `parent.ema(p).iloc[:n]` yapıyordu — yeni Series + yeni DatetimeIndex (pandas'ta
indeks dilimlemesi freq çıkarımını da tetikler), yani **O(n)**. `detect_event` bar başına iki
görünüm kurup onlardan yalnız **dört sayı** okuyordu; `generate_signal` aynı `ema(50)`'yi üç,
`rsi14`'ü iki çağırana gönderiyordu. Kırpma makinesi profilin **%36'sıydı**.

**BULGU-43-4 · `rsi`/`adx` gövdesinin %43'ü sarmalayıcıydı.** Wilder yumuşatması (`_rma`→ewm)
pandas'ta kalmak zorunda; ama ETRAFINDAKİ elemanwise aritmetik beş `Series.where/mask/clip`
+ bir `replace` demekti ve profilde `generic._where` tek başına `rsi` süresinin **%43'ü** idi.
(Tur-34/36'nın "sıkı döngüdeki genel-amaçlı sarmalayıcı" dersinin dördüncü tekrarı.)

### Ölçümler — hepsi min-of-N, aynı süreçte art arda (Tur-36 dersi), sessiz makinede

| ölçüm (gerçek `A_1h`, 4000 bar) | ÖNCE | SONRA | kazanç |
|---|---:|---:|---:|
| **`backtest_symbol` 4000 bar** | **7,592 s** | **1,935 s** | **3,92×** |
| **`backtest_symbol` 1500 bar** | 2,508 s | 0,680 s | **3,69×** |
| **canlı sembol-başı ağır blok** | 15,998 ms | **5,386 ms** | **2,97×** |
| `gate.explain` (pano P(win) kartı) | 15,991 ms | 4,300 ms · **0,985 ms** (önbellekli) | 3,72× · **16,2×** |
| `gate.probability` | 7,627 ms | 4,418 ms · **0,928 ms** (önbellekli) | 1,73× · **8,22×** |
| `extract_features` | 7,575 ms | 4,099 ms · **0,789 ms** (önbellekli) | 1,85× · **9,60×** |
| `detect_event` (bar başına) | 0,2883 ms | **0,0320 ms** | **9,01×** |
| `volatility_explosion_filter` | 0,1996 ms | 0,0833 ms | 2,40× |
| `strategy.rsi(14)` | 2,658 ms | **0,452 ms** | **5,88×** |
| `strategy.adx(14)` | 2,170 ms | 0,977 ms | 2,22× |
| `strategy.last_valid` | 0,1182 ms | 0,0141 ms | **8,38×** |
| `indicators.indicator_panel` (1500 bar) | 88,65 ms | 73,18 ms | 1,21× |
| `indicators_ext.extended_rows` (1500 bar) | 52,61 ms | 42,42 ms | 1,24× |

`detect_event` kazancı **kırpma memo'su AÇIKKEN** ölçülmüştür (muhafazakâr okuma); Tur-43
öncesi gerçek hâlde her çağıran kendi kırpmasını kuruyordu.

### Denklik — iddia ÖLÇÜLDÜ, umut edilmedi

| ne | kanıt |
|---|---|
| `extract_features` (önbellekli ↔ önbelleksiz) | gerçek veri 40 dilim × 25 özellik × 2 yol = **2000 karşılaştırma → maks fark 0,000e+00** |
| ↳ rastgele/düz/NaN'lı df | 300 df × 25 özellik = **7500 karşılaştırma → 0,000e+00** |
| **hizasız görünüm koruması** | `nbars` uyuşmazlığında önbellek yok sayılır → **sızan özellik 0** |
| `detect_event` | gerçek veride **3941 bar**, `shared=None` yolunda 563 dilim, 400 rastgele/düz/NaN/tek-bar df × 3 yol = 1200 çağrı → **farklı verdikt 0** |
| `breakout_at` ↔ `.iloc[-21:-1].max()` | bar-bar ~30k karşılaştırma → **maks fark 0,000e+00** |
| `rsi` / `adx` vektörleştirme | gerçek **22.825 barlık** seri → **bit-denk**, NaN konum farkı 0; 600 df × 3 periyot (düz/NaN/tek-yön/uç n=0,1,2) → **bit-denk olmayan 0** |
| `volatility_explosion_filter` | 200 df → **farklı verdikt 0**; tam-NaN seri hâlâ `ATR_NA` |
| `last_valid` konsolidasyonu | 3000 seri (NaN/inf/tam-NaN/kuyruk-NaN) → farklı 0 |
| ↳ **panel bütünü** | 103 df · **13.734 satır** → **oy farkı 0, değer farkı 0, konsensüs farkı 0** |
| backtest davranışı | 1500 ve 4000 barda işlem sayısı **birebir aynı** (14 · 26) |

`rsi` seri ADI da korunur (pano `_series_payload` ona bakabilir) ve Tur-28'in "düz pencerede
RSI 50" dersi vektörleştirmeden sonra da geçerlidir — ikisi de teste bağlandı.

### Ölçüldü ama DEĞİŞTİRİLMEDİ (dürüstlük kalemi — sonraki tur burayı yeniden kazmasın)

- **`volatility_explosion_filter` hâlâ `atr14` kırpmasını zorluyor.** Kaldırmak için üç yeni
  erişimci (`atr_at` · `atr_mean` · `atr_valid_count`) gerekirdi — TEK çağıran için API
  büyütmek, kalan ~%10 için doğru takas değil.
- **`indicator_panel`'in kalan 73 ms'i.** Tur-33/34 bu dosyayı zaten kazımıştı; kalan süre 60+
  göstergeye ince yayılmış durumda, profilde **%5'i geçen tek bir suçlu yok**.
- **`live_predict_service._forecast` ÖLÜ KOD** (repoda hiç çağıran yok; pano aynı mantığın
  kendi kopyasını satır-içi taşıyor). Bir performans turunda silinmedi — kaldırma kararı
  davranış turuna aittir.
- **Kapının `except Exception: pass`'i imza kırılmasını GİZLİYOR.** `probability` imzası
  `shared=` ile genişleyince test ikizleri TypeError verdi; `generate_signal` bunu yutuyor ve
  kapı SESSİZCE pasifleşiyordu. Testler yakaladı, üretim yakalamazdı (bkz. ACI DERSLER).

### Doğrulama

**2801 test yeşil** (Tur-43 öncesi 2770 → **+31**), `ruff check nihai/` temiz, backtest işlem
sayıları birebir korundu. Değişen dosyalar: `nihai/signals/strategy.py`,
`nihai/signals/meta_features.py`, `nihai/signals/meta_gate.py`, `nihai/signals/indicators.py`,
`nihai/signals/indicators_ext.py`, `nihai/execution/live_predict_service.py`,
`nihai/web/web_dashboard.py` + `tests/test_tur43_paylasimli_onbellek.py` (yeni) ve iki test
ikizinin imzası.

## Tur-44 — kapının SESSİZ degradesi: `active` bir durum değil, bir dosya bayrağıydı (2026-08-23, Opus 5 max)

Tur-43'ün açık kalemi. Kusur bir performans kalemi değil bir **ÖLÇÜLEBİLİRLİK** kalemidir:
kapı bozulduğunda sistem bunu SÖYLEMİYORDU.

### Kusur (Tur-43 sırasında CANLI olarak yaşandı)

`_MetaGate.probability` imzası `shared=` ile genişleyince eski imzalı çağıranlar TypeError
verdi. İKİ katman birden yuttu: `probability` kendi gövdesini `except Exception: return None`,
`generate_signal` de kapı çağrısını `except Exception: pass` ile sarıyor. Sonuç: kapı
KONFİGÜRE AÇIK ve model YÜKLÜ iken sessizce pasifleşti. Testler yakaladı — **üretim
yakalamazdı**, çünkü tek dış gözlem `gate.active`'dir ve o hâlâ True döner.

Kök neden `except`'ten derindi: **`active` iki ayrı soruyu tek bayrakla yanıtlıyordu** —
"model dosyası yüklendi mi?" ve "P(win) üretiyor mu?". CLI `AKTİF (logistic)`, pano
`meta_model: logistic` yazmaya devam ediyordu.

Zarar teorik değil. Gölge kipin varlık nedeni CANLI KANIT DÖNGÜSÜDÜR
(`prediction_tracker` p_win logu → `gate_sim`). Kapı susarsa o log BOŞ birikir ve
"kapı işe yarıyor mu?" sorusu canlıda ASLA ölçülemez — ACI DERS #2/#5'in tam üstüne.

### Onarım — zarif degrade KORUNDU, sessizlik KALDIRILDI

| ne | nerede | not |
|---|---|---|
| Hata defteri `note_call`/`note_error` | `meta_gate` **modül düzeyi** | örnekte DEĞİL: `reload_gate()` örneği atar, hata geçmişi atılmamalı; ayrıca örneğe hiç ulaşamayan import/kurulum hataları da buradan geçer |
| Hız sınırlı UYARI | ilk hata + her 100. | backtest bar başına çağırır: **sayım TAM, log SEYREK** (300 hata → 4 satır) |
| `gate_health()` beş durum | `MODEL_YOK · HENUZ_SORULMADI · SAGLIKLI · KISMI_HATA · SUSUYOR` | `SUSUYOR` = sorulan HER soruda hata = fiilen ölü kapı |
| Sinyale İZ | `META_GATE_ERROR` gerekçesi + `meta_gate_error` metadata | sinyal **BLOKE EDİLMEZ** — yön/stop/hedef/kalite birebir aynı (testle kilitli) |
| Canlı yüzey | `snapshot()["meta"]["meta_gate"]` | yazılıp OKUNMAYAN teşhis kusurdur (Tur-38 kuralı) |
| CLI yüzeyi | `live_predict` döngü uyarısı | açılış satırı 0 çağrıda ölçüm yapamaz; uyarı DÖNGÜDE basılır |

`MODEL_YOK` bir hata DEĞİLDİR ve sayılmaz — model olmadan kapı pasif olmalıdır, bu
tasarımdır. Ayrım testle kilitli.

### Doğrulama

**2811 test yeşil** (Tur-43 sonu 2800 → **+11**), `ruff check .` ve `mypy nihai` temiz.
`tests/test_tur44_kapi_sessiz_hata.py` üç ekseni birden sorar: (a) hata görünür mü
(iz + sayım + log + hız sınırı), (b) zarif degrade korundu mu (sinyal bozulmuyor mu),
(c) durum ayrımı doğru mu (`MODEL_YOK` sayılmıyor, iç ve dış katman AYRI AYRI defterden
geçiyor, teşhis canlı meta'ya giriyor).

Test verisi notu: `generate_signal` rastgele bir seride çoğunlukla WAIT verir ve kapıya
HİÇ ulaşmaz — o durumda test yeşil görünüp hiçbir şey doğrulamaz. Seri (seed 7) aranarak
LONG üretecek şekilde seçildi ve bir ÖN KOŞUL TESTİ bunu kilitler: strateji eşikleri
değişip seri WAIT'e düşerse o test kırmızı olur. **Sessiz atlama, bu dosyanın konusu olan
sessiz yutmanın ta kendisidir.**

## Tur-45 — vol filtresinin ATR kırpması: nedenselliği parametreye değil VERİ YAPISINA yaz (2026-08-23, Opus 5 max)

Tur-43'ün son açık kalemi. O turda bilerek ertelenmişti (*"üç yeni erişimci TEK çağıran
için API büyütmek, kalan ~%10 için doğru takas değil"*); profil o tahmini düzeltti.

### Kusur

`generate_signal` her barda `shared.atr14()` çağırıyordu ve `_SharedSlice` bunu `.iloc[:n]`
ile yeni bir Series + yeni bir DatetimeIndex kurarak veriyordu — **bar başına O(n)**. Oysa
o seriden okunan yalnız ÜÇ şeydi: son değer · son 20 barın ortalaması · "hepsi NaN mi?".
Tur-44 sonrası profilde `volatility_explosion_filter` **%19**, kırpma makinesi **%13**'tü.

### Çözüm — ve neden `n` parametresi DEĞİL

`atr14_np()` ebeveynde diziyi BİR kez kurar; alt görünüm `[:n]` ile **sıfır-kopya numpy
görünümü** alır. İlk tasarım filtreye bir `n` (görünüm uzunluğu) geçirmekti; o tasarım
REDDEDİLDİ çünkü yanlış bir `n` sessizce GELECEĞİ okuturdu — Tur-43'ün `nbars` kapısıyla
aynı sınıf risk. `[:n]` dilimi ise geleceği taşıyamaz: **yanlış geçirilebilecek bir sayı
yoksa look-ahead de yoktur.** Nedensellik bir kontrole değil VERİ YAPISINA yazıldı.

Filtre gövdesi numpy'a indi ve `atr_series` hâlâ pandas Series kabul eder (dış çağıranlar
ve `test_strategy_perf_parity`); float64 seride `np.asarray` sıfır-kopyadır.

### Ölçümler — A/B AYNI SÜREÇTE (eski gövde `git show HEAD:` ile ayrı bir modüle yüklendi)

| ölçüm | ÖNCE | SONRA | kazanç |
|---|---:|---:|---:|
| bar döngüsü 4000 bar (backtest'in %87'si) | 1,036 s | 0,871 s | **1,19×** |
| bar döngüsü 1500 bar | 0,364 s | 0,310 s | 1,18× |
| filtre + ATR erişimi (4000 bar) | 0,0470 ms | 0,0162 ms | **2,90×** |

Aksiyonlu sinyal sayısı iki sürümde **birebir aynı** (1500 barda 146 ↔ 146, 4000 barda
426 ↔ 426).

**DÜRÜSTLÜK KALEMİ — Tur-36 dersi bu turda BANA çarptı.** İlk uçtan-uca okuma
`1,935 s → 0,899 s = 2,15×` gibi göründü; o "ÖNCE" sayısı bir tam test süiti koşusunun
hemen ardından, yük altında alınmıştı. Aynı süreçte art arda koşturulan A/B gerçek sayıyı
**1,19×** verdi. Kural işledi: *farklı zamanlarda alınmış iki sayıyı bölme.*

### Denklik

| ne | kanıt |
|---|---|
| numpy görünümü ↔ kırpılmış Series | 4 uzunlukta, NaN konumları dahil → **maks fark 0,000e+00** |
| filtre verdikti (gerçek veri) | **3941 bar**, her barda iki arayüz → **farklı verdikt 0** |
| filtre verdikti (dejenere) | 500 df (düz/NaN'lı/kuyruk-NaN/patlama) × 2 arayüz + 5 uç durum = **1005 çağrı → 0** |
| `.iloc[-20:].mean()` ↔ `np.nanmean` | ≤20 elemanlı **20.000 pencere** (NaN'lı/tam-NaN/1e8 ölçekli) → **0,000e+00**, üstelik `bottleneck` KURULU |
| görünüm nedenselliği | `upto(n).atr14_np()` ≡ `atr(df.iloc[:n], 14)` — sıfırdan hesapla ile bit-denk |

### Doğrulama

**2829 test yeşil** (Tur-44 sonu 2811 → **+18**), `ruff check .` ve `mypy nihai` temiz.
Kilit: `tests/test_tur45_atr_kirpmasiz.py` — bit-denklik, **sıfır-kopya olduğunun kanıtı**
(`dilim.base is taban`), nedensellik, iki arayüzün aynı verdikti, uç durumlar ve patlama
dalının ÖLÜ OLMADIĞI.

## Tur-46 — kenar-keşif defteri: aramanın kendisi kanıt eşiğini SIKIŞTIRSIN (2026-08-24, Opus 5 max)

Adayı OLMAYAN tur (Tur-24/27/31 deseni): elle uygulanan bir kural makineye taşınıyor.
Bir aday FAIL ettikten SONRA kapı değiştirmek sonuca göre kural değiştirmektir.

### Kusur

`BACKLOG.md` şunu YAZIYOR ama uygulamasını İNSANA bırakıyordu: *"aileler arası çoklu-test
bu sayaçla insan tarafından izlenir. Sayaç büyüdükçe tek bir PASS'in kanıt değeri DÜŞER."*
`harness._stage_dsr` ise DSR'yi HER koşuda sabit `_DSR_N_TRIALS=4` ile deflate ediyordu ve
bu sabit YALNIZ adayın kendi eşik taramasını sayıyordu. Sonuç: **1. adayla 10. adaya AYNI
çıta.** Program arttıkça şansla PASS alma olasılığı büyürken kanıt eşiği sabit kalıyordu —
üstelik elle tutulan sayaç unutulabilir ve geriye dönük düzenlenebilirdi. Kanıt kapısının
kendi disiplini içinde otomatikleşmemiş son halkası buydu.

### Çözüm

`research/edge_lab/ledger.py` (YENİ, 201 satır): yalnız-ekleme JSONL deney defteri.
`run_candidate` kapılara ULAŞAN her koşuyu deftere yazar ve DSR deflasyonunu defterden
okur:

    n_trials = (adayın kendi eşik taraması: 4) × (defterdeki AYRIK aday, bu koşu dahil)

* **Sayım birimi AİLEDİR, koşu değil.** Aynı adayı hata düzeltmek için yeniden koşmak yeni
  bir ARAMA değildir ve sayacı şişirmez (aksi halde geliştirme döngüsü kanıt eşiğini
  kaçırırdı); kapı sıkılaştırma yeniden-ölçümleri de (`*_TUR24GATE`, `*_bulgu25gate`) aynı
  aile adı altındadır. Defter yine de HER koşuyu kaydeder (denetim izi).
* **Veri yükleme hatası deftere YAZILMAZ.** Yanlış yazılmış bir dosya yolu bir deneme
  değildir ve sonraki adayın eşiğini haksız yere yükseltmemelidir.
* **TAVAN (bilinen):** ad seçimi araştırmacının elindedir; aynı adı FARKLI özellik kümesiyle
  yeniden kullanmak sayacı atlatır. `features`/`dataset` her satırda saklanır ki denetim
  bunu yakalayabilsin — otomatik tespit YOK.

### Defterin YERİ bir kusur düzeltmesidir

Defter `data/` altında DEĞİL, `research/edge_lab/kosu_defteri.jsonl`'de ve SÜRÜM
KONTROLÜNDE. `/data/` .gitignore'dadır: orada tutulsaydı taze bir klon (ya da silinen bir
data dizini) çoklu-test sayacını SIFIRLAR ve kanıt kapısı **sessizce gevşerdi** — tam da bu
modülün engellemek için var olduğu şey. Yeri artık `BACKLOG.md`/`SONUC_*.md` ile aynı:
kurumsal kayıt (3e6bc77'nin "kurumsal kayıt sürüm kontrolüne alındı" kararının devamı).

### Ölçüm — çıta ne kadar yükseliyor?

n = 20.000 kapı-sonrası işlem; `DSR = 0,95` eşiğini geçmek için gereken `SR_fazlalık`:

| n_trials | 4 (eski sabit) | 8 | 12 | **16** (3 aile + yeni aday) | 20 | 40 |
|---|---:|---:|---:|---:|---:|---:|
| gereken SR | 0,01907 | 0,02195 | 0,02341 | **0,02437** | 0,02508 | 0,02712 |

Bugünkü defterle (3 aile) sıradaki adayın çıtası **+%27,8**. Büyüme logaritmiktir
(`E[max SR] ~ sqrt(2 ln N)`): arama sürdükçe sıkışır, asla geçilemez hâle gelmez.

### Geriye dönük denklik

* **BOŞ defterde formül eski sabitin ta kendisidir** (1 aday × 4 eşik = 4). Yani bu
  değişiklik geçmişte koşulan üç ailenin verdiktini DEĞİŞTİREMEZ. Kanıt kapısı bu yolla
  yalnız SIKILAŞABİLİR, asla gevşeyemez (*"sıkılaştırma yeni sonuç ÜRETEMEZ, yanıltıcıyı
  eler"*).
* Defter, programın GERÇEK arama geçmişiyle dolduruldu: `overnight_intraday_v1` (SONUC
  belgesinden), `leadlag_sector_v1` ve `rebalance_flow_v1` (arşivlenmiş `data/verdict_*.json`
  artefaktlarından). Her satırda `not` alanı kaynağı söyler: ölçülmüş koşu ile belgeden
  aktarılmış kayıt defterde AYIRT EDİLEBİLİR. Sıradaki YENİ aday (#2 VRP) 4 ile değil
  **16** ile deflate edilecek.

### Test izolasyonu — kapıda sıra-bağımlılık, kapının kendisinin bozulmasıdır

DATA_DIR tüm pytest oturumu için TEK geçici dizindir; defter orada tutulsaydı test'ten
test'e birikir ve `run_candidate` çağıran her test bir SONRAKİNİN DSR eşiğini yükseltirdi
(tam süit ile tek-dosya koşusu FARKLI verdikt verirdi). Kök `conftest.py`'ye autouse
fixture eklendi: her teste ayrı defter (`NIHAI_EDGE_LEDGER`). Doğrulandı: tam süit
sonrası gerçek defter hâlâ **3 satır**.

### CLI — ve `python -m paket.modul` tuzağı

`python -m research.edge_lab` programın durumunu yazdırır (elle sayacın yerine). CLI
`ledger.py`'nin içine KONMADI: `__init__ → harness → ledger` zinciri yüzünden runpy modülü
İKİ KEZ yükler ve `RuntimeWarning` basar; watchdog/zamanlanmış görev çıktısında bu
gürültüdür. `__main__.py` temiz yoldur ve `ensure_utf8_stdout()` guard'ını taşır
(boru hattına yönlendirilmiş çıktıda Türkçe karakterler doğrulandı).

### Doğrulama

2842 passed / 1 skipped (+19 yeni test), ruff temiz, mypy 165 dosya temiz.
Kayıt: `research/edge_lab/BACKLOG.md` §çoklu-test muhasebesi yeniden yazıldı;
`CLAUDE.md` Komutlar bloğuna defter CLI'si eklendi.

## Tur-47 — yön güveni bir SKORDUR: pano ölçülen EN KÖTÜ alt kümeyi YEŞİL boyuyordu (2026-08-25, Opus 5 max)

Tur-36 §9 **açık kalem #3**'ün kapanışı — 10 tur açık kaldı. Kalem şöyle yazıyordu:
*"`confidence` alanının olasılık gibi okunduğu yerler taranmalı; nerede olasılık muamelesi
görüyorsa orası sessiz bir hatadır."* Tarama yapıldı; iki yüzeyde hata bulundu ve ikisi de
KARAR-YÜZEYİNDEYDİ.

### Kusur — bir RENK bir TAVSİYEDİR

`web_dashboard.html` canlı tahmin tablosunda ham yön güvenini trafik ışığıyla boyuyordu:

    conf >= 0.66 → YEŞİL · conf >= 0.40 → TURUNCU · altı → GRİ

ve sayıyı `%` ile sunuyordu. Canlı defter (`data/predictions_log.csv`, **63.237 çözülmüş
tahmin**) bunun TERSİNİ söylüyor — üstelik panonun KENDİ renk sınıflarıyla ölçüldüğünde:

| pano rengi | aralık | n | isabet |
|---|---|---:|---:|
| GRİ | conf < 0,40 | 24.999 | **0,5094** |
| TURUNCU | 0,40–0,66 | 19.653 | 0,4590 |
| **YEŞİL** | conf ≥ 0,66 | 18.585 | **0,4395** |

**Fark +6,98pp, iki-oran z = +14,43.** Kova düzeyinde Pearson ρ = **−0,889** (10 kova,
yayılım 0,097). Yani pano, ölçülen en KÖTÜ alt kümeyi "en iyi" diye boyayıp operatörü ona
yönlendiriyordu. Aynı iddia varsayılan SIRALAMADA da vardı (`rows.sort(..., -confidence)`).

**Neden 10 tur görünmedi:** kova renkleri her kovanın SEVİYESİNİ kodlar (`hit_rate >= 0,55
→ yeşil`), kovalar arası SIRAYI değil. TERS bir eğri "biraz yeşil biraz kırmızı" görünür =
sağlıklı bir eğriden yapısı gereği AYIRT EDİLEMEZ. **Kusuru ifade edemeyen bir gösterge onu
açığa da çıkaramaz.**

Terslik aslında `config.py`'de bir YORUMDA yazılıydı (*"canlı veride confidence şu an
isabetle TERS korele"*). **Bir kod yorumu çalışma-zamanı hükmü değildir**: operatör panoyu
okur, yorumu değil. Kardeş tablo (`pwin_calibration`) hükmünü denetim B-4'te almıştı;
`buckets` (güven→isabet) tablosu B-4'ten beri HÜKÜMSÜZ basılıyordu.

### Bölme testi — artefakt DEĞİL

ACI DERS #3'ün kendi çıtası (hem zaman hem sembol bölmesi) uygulandı:

| bölme | GRİ | YEŞİL | fark | z |
|---|---:|---:|---:|---:|
| hepsi | 0,5094 | 0,4395 | +6,98pp | +14,43 |
| zaman: ilk yarı (< 2026-07-16) | 0,5020 | 0,4405 | +6,15pp | +9,07 |
| zaman: ikinci yarı | 0,5158 | 0,4384 | +7,74pp | +11,13 |
| sembol A (alfabetik çift indeks) | 0,5190 | 0,4433 | +7,57pp | +11,47 |
| sembol B (tek indeks) | 0,4988 | 0,4349 | +6,39pp | +8,97 |
| 2026-06 / 07 / 08 | 0,5020 / 0,5071 / 0,5226 | 0,4321 / 0,4365 / 0,4576 | +7,00 / +7,05 / +6,50 | +6,38 / +11,38 / +5,93 |

### ⚠️ BU ÖLÇÜM YENİ DEĞİL — YENİ OLAN, YÜZEYE ULAŞMASI

Dürüstlük kalemi: terslik ilk kez burada bulunmadı. **EDGE_FINDINGS §17-C (2026-06-25)**
aynı şeyi yazmış: *"yüksek-güven kovaları (0.6-1.1) %40-44, düşük-güven (0.2-0.3) %56"*.
Bu turun katkısı ölçümü KEŞFETMEK değil, onu **karar yüzeyine taşımaktır**: iki ay boyunca
bulgu üç ayrı yerde yazılıydı — bir araştırma script'inin docstring'i, `EDGE_FINDINGS` §17-C
ve bir `config.py` YORUMU — ve üçü de operatörün KARAR ANINDA okuduğu yer değildi. Pano
o iki ay boyunca yüksek güveni yeşil boyamayı sürdürdü. **Kayıt ile YÜZEY arasındaki bu
boşluk kusurun kendisidir.** Tur-47 ayrıca ölçümü 5.936 → **63.237** tahminde (10×) ve
zaman+sembol bölmesinde bağımsız olarak yeniden üretti.

### ⛔ Çare TERSİNE ÇEVİRMEK DEĞİL, İDDİA ETMEMEKTİR

GRİ sınıf %50,94'tür = yazı-tura; defterin tabanı zaten %47,32. Yani **"düşük güveni al"
bir kenar DEĞİLDİR** — aleyhte kanıttan kenar türetmek ACI DERS #3/#7 ihlalidir. Ve bu
yasak burada yalnız DOKTRİN değil, **ÖLÇÜLMÜŞ** bir sonuçtur: `calibration_fade_test`
(2026-06-25, §17-D, 5.936 tahmin, kronolojik 70/30) tam bu kuralı sınadı — fade OOS'ta
isabeti **%54,6** (>%50!) yapıyor ama ortalama getiriyi **−0,071%** (NEGATİF) bırakıyor.
*İsabet beklenti değildir.* Yön çevirme yolu iki ay önce sayıyla kapatılmıştı. Ölçüm
yalnız şunu söyler: *panonun İDDİA ETTİĞİ sıralama veriyle çelişiyor.* Bu yüzden:

* ham güven sütunu **NÖTR** boyanır (`#94a3b8`), birim `%` değil **`sk`** olur (sayı ve
  çubuk KALIR — sinyalin öz-değerlendirmesi bilgidir, iyi/kötü anlamı kalkar);
* detay kartı `Yön güveni: %62` yerine `Yön güveni (skor): 62/100` yazar;
* varsayılan sıralamanın üçüncü anahtarı `-confidence` → **`symbol`** (deterministik ama
  iddiasız; boş bırakmak eşitlikte tabloyu her döngüde oynatırdı). SATIR KÜMESİ AYNI;
* `prob_cal` (GERÇEK kalibre olasılık) sütunu renkli KALIR — nötrleştirme ölçülmüş olanı
  soldurmamalıdır.

### Hüküm makineye taşındı — ve "skor" ile "olasılık" AYRILDI

`_calibration_verdict` artık `konu` alır: `"olasilik"` (varsayılan — B-4 davranışı
**bit-denk**) ve `"skor"`. Skor kipinde **Brier HESAPLANMAZ**; bir skora Brier uygulamak
onu olasılık sanmanın ta kendisidir (Tur-36 §5: `confidence` uzmanının log-kaybı **1,282**
↔ taban 0,694 = **1,85 katı**). Etiketler de ayrıdır — `SIRALIYOR ≠ TUTARLI`: doğru
sıralayan bir skor kalibre OLMAYABİLİR ve rapor bunu iddia etmemelidir.

`report()` → `confidence_verdict`; canlı defterde bugün:

    label=TERS_SIRALAMA  ρ=-0,889  yayılım=0,097  n=63.237  konu=skor

Pano bunu `güven→isabet` tablosunun ALTINA basar (B-4'ün aynı hatası tekrarlanmasın:
backend hüküm üretip pano basmazsa kusur kapanmamış olur).

### Taramanın ÜÇÜNCÜ bulgusu — kalibrasyonu FİT EDEN script de skora "Brier" diyordu

`research/calibrate_forecast.py` ham güvenin MSE'sini `meta["brier"]` diye kaydediyor ve
CLI'da `| Brier: 0.3261` diye basıyordu. Brier bir OLASILIK tahminini puanlar; `confs`
kalibrasyonun **GİRDİSİDİR** (skor, çıktı değil). Adı "brier" kaldıkça 0,3261 "kötü kalibre
olasılık" diye okunur — doğru okuma *"bu zaten olasılık değil"*dir. Üstelik sayı
REFERANSSIZDI: yanında ne bilgisiz taban (0,25) ne "hep taban oranını söyle" kuralının MSE'si
(p(1−p)) vardı, yani tek başına iyi/kötü diye okunamazdı (ACI DERS #9'un ölçüm tarafı).
Artık `skor_mse` + `mse_taban_orani` + `mse_bilgisiz` yazılır ve `note` alanı girdinin skor
olduğunu SÖYLER. Alanı okuyan kod YOKTU (`_load_calibration` yalnız `points` okur) — yani
bu aynı zamanda Tur-38'in "yazılıp okunmayan alan" kusurunun bir örneğiydi.

### DÖRDÜNCÜ bulgu — İKİ hüküm fonksiyonu AYNI veride ÇELİŞİYORDU

`prediction_tracker.verdict()`'in `INVERTED` dalı şunu yazıyordu: *"ters yön; **sinyali
tersine çevirmek bile düşünülebilir**"* — yani **AYNI MODÜLDEKİ** `_calibration_verdict`in
açıkça YASAKLADIĞI çıkarımı ÖNERİYORDU (*"tersine çevirme yasak, ACI DERS #3/#7"*). Bu,
ACI DERS #9'un kendi ifadesidir: *"bir kural iki kapıdan yalnız birine konursa, ikisi aynı
veride ÇELİŞİK verdikt verir ve eski/yanlış olanı okuyan operatör aynı tuzağa düşer."*
Bugün dal ateşlenmiyor (isabet 0,4732 → `NO_EDGE`, eşik 3pp) ama **0,3pp uzaktadır**.
Etiket (ÖLÇÜM) korundu, ÖNERİ kaldırıldı ve yerine §17-D'nin ölçümü kondu.

Kardeşi `research/forward_harness._verdict` ise *"fade/yön-çevirme test edilebilir
(calibration_fade_test)"* diyordu — **o test KOŞULDU ve CEVAP VERDİ** (§17-D). Cevabı
TAŞIMAYAN bir işaretçi, bir sonraki turu kapanmış bir kuyuyu yeniden kazmaya gönderir
(projenin kendi ritüel gerekçesi: *"rapor/ders yazılmazsa sonraki tur aynı yeri yeniden
kazıyor"*). İşaretçi korundu, yanına verdikt yazıldı.

### Optimizasyon (ölçülmüş, ÇIKTI-DENK)

`report()` canlı defterde profillendi: **860.850 `dict.get` çağrısı**, profilin %17,6'sı.
İki kök neden, ikisi de projenin kendi dersinin tekrarı:

1. **`_pwin_pairs` `report()` başına İKİ KEZ koşuyordu** — bir kez pwin kalibrasyonu için,
   bir kez de `gate_sim(rows)` içinden. 63k satır iki kez taranıyordu (tek geçiş **27,4 ms**).
   `gate_sim(..., pairs=)` D4 paylaşımıdır (`_read_log_rows` deseninin aynısı); verilmezse
   davranış BİREBİR. *(Tur-33: "aynı ağırlığı iki kez hesaplama".)*
2. **Kova anahtarı satır başına biçimlendiriliyordu** — 63k `f"{lo:.1f}-{lo+0.1:.1f}"`, oysa
   ayrık anahtar YALNIZ 10 tane. Önceden kurulan `_BUCKET_KEYS` tablosu: 65,6 → **13,6 ms
   (4,82×)**. **BİT-DENK** doğrulandı 2005 noktada (negatifler ve 1e-18 dahil): negatif
   girdide eski biçimlendirme yolu BİLEREK korundu, çünkü `_BUCKET_KEYS[-3]` sessizce
   "0.7-0.8" verirdi ve dejenere bir satır geçerli bir kovaya karışırdı (Tur-33 dersi).

3. **Hüküm fonksiyonu kova anahtarını `float(k.split("-")[0])` ile ayrıştırıyordu** ve
   NEGATİF bir anahtarda (`"-0.3--0.2"` → `['', '0.3', '', '0.2']`) `float('')` ile
   **ÇÖKÜYORDU** — bozuk tek bir CSV satırı `/api/predictions/accuracy` ucunu 500'e
   düşürebilirdi. Bu bir performans kalemi değil, optimizasyon sırasında ORTAYA ÇIKAN bir
   kusurdur: anahtarı KURAN tablo (`_BUCKET_KEYS`) yazılınca ÇÖZENİN de (`_bucket_mid`)
   onun TERSİ olması gerektiği görüldü. Tanınmayan anahtarın orta noktası UYDURULMAZ →
   o kova hükme girmez (dejenere bir kova için "sıralama" sorusu zaten tanımsızdır),
   ama tüm hükmü de düşürmez; `n_kova`/`n` hükme GİREN sayıyı raporlar.

**A/B AYNI SÜREÇTE, DÖNÜŞÜMLÜ** (eski gövde `git show HEAD:` ile ayrı modüle yüklendi;
11 tur boyunca ESKİ→YENİ art arda — Tur-36 kuralı: farklı zamanlarda alınmış iki sayı
bölünmez):

    report()   281,8 ms -> 183,4 ms   1,54x   (min'lere göre de 1,54x)
               (üstelik YENİ sürüm confidence_verdict'i de üretiyor = EK iş)

Çıktı denkliği: `pwin_verdict`'e eklenen `konu` alanı dışında **DEĞİŞEN ORTAK ALAN YOK**;
tek yeni anahtar `confidence_verdict`.

### ÖLÇÜLDÜ ama DEĞİŞTİRİLMEDİ (dürüstlük)

* **`hits` ve `rets` iki ayrı geçişte hesaplanıyor.** Tek döngüde birleştirmek 53,29 →
  48,12 ms (**1,11×**) — 183 ms'lik toplamda gürültü mertebesinde ve C düzeyindeki `sum`
  genexpr'ini elle döngüye çevirmek okunurluğu düşürür. Kazanç bedeli hak etmiyor.
* **`prob_cal` kovaları neredeyse DEJENERE:** 63.237 satırın **58.186'sı** tek kovada
  (0,4–0,5), isabet 0,4748. Kalibrasyon eğrisi her şeyi taban orana büzüyor — ACI DERS #9'un
  ("kazancın tamamına yakını taban orana BÜZÜLMEDEN geliyor") canlı yoldaki karşılığı.
  Eğri (`data/forecast_calibration.json`) 2026-06-23 fit'idir ve tazelenmedi: tazelemek ayrı
  bir karardır (`calibrate_forecast.py` koşusu), bu tur ÖLÇTÜ, DOKUNMADI.
* **Ters sıralamadan sinyal TÜRETİLMEDİ.** "Düşük güveni al"/fade kuralı bu turda
  denenmedi bile — ve denenmesine gerek de yoktu: **iki ay önce ÖLÇÜLDÜ ve KAPANDI**
  (§17-D, `calibration_fade_test`): fade OOS isabeti %54,6 ama ort. getiri −0,071%.
  GRİ sınıf zaten yazı-turadır; böyle bir kural ACI DERS #3/#7 ihlali olurdu.

### Doğrulama

`tests/test_tur47_guven_hukmu.py` (24 test): skor kipinde Brier YOK · etiketler ayrı ·
"tersine çevir" önerisi metinde YOK · olasılık kipi üç etikette regresyona kilitli ·
`report()` hükmü HER ZAMAN taşıyor · eksik `confidence` kolonu çökertmiyor · **pano
regresyon kapıları** (`conf >= 0.66 ? '#22c55e'` ifadesi dosyada OLMAMALI, `%` birimi
gitmiş, `confidence_verdict` basılıyor, `prob_cal` rengi KORUNMUŞ) · varsayılan sıralama
güvene bakmıyor · fit script'i skora 'Brier' demiyor · negatif kova anahtarı
hükmü çökertmiyor · `_bucket_key`↔`_bucket_mid` ters fonksiyon · `verdict()`
INVERTED dalı tersine çevirmeyi ÖNERMİYOR (diğer üç dal değişmedi).

Tam süit **2866 passed / 1 skipped** (Tur-46: 2842 → +24), `ruff check .` temiz, `mypy .` 166 dosya temiz.

## Tur-48 — ACI DERS denetimi: dersler DOĞRU çıktı, TAŞINMALARI kusurluydu (2026-08-25, Opus 5 max)

Kullanıcı isteği: *"projede adı acı ders ve benzeri geçenleri tekrar kontrol edip (doğru mu
değil mi) günceller misin"*. Dokuz dersin tamamı iddia-iddia koda ve veriye soruldu.

### Denetim kapsamı ve SONUÇ

| ne denetlendi | yöntem | sonuç |
|---|---|---|
| Derslerde adı geçen kod adları | 132 backtick'li ad kaynakta arandı | **0 kayıp** |
| Ölçülebilir davranış iddiaları | 12 iddia yeniden koşuldu | **12/12 TUTTU** |
| Parametre merdiveni çapaları | `theoretical_param_count` 6 basamak | **6/6 BİREBİR** |
| Manşet sayılar | 20 rakam kaynak belgede arandı | **20/20 izlenebilir** |
| Atıf ↔ bağlam uyumu | 229 kaynak atıfı otomatik tarandı | **doğrulanmış yanlış yönlendirme YOK** |

Ölçülen çapalardan bazıları: `deep_mlp` **186.702** (f=25), `quant_transformer_moe`
**93.228.307**, `moe_xl` **283.639.571**, `ultra` **771.874.072**, `mega`
**2.014.146.906** — belgedeki sayılarla TAM aynı. `rsi(düz)=50`, `mfi(düz)=50`,
`vote(eşit)=0`, düz panel `YETERSİZ ÖLÇÜM`, `NormalDist` pdf BİT-DENK / cdf 1,110e-16 /
ppf 1,776e-15, `last_valid` ±inf'i elemiyor NaN'ı eliyor — hepsi belgedeki hâliyle.

**DÜRÜSTLÜK NOTU — bir "başarısızlık" kendi girdimin hatasıydı.** İlk koşuda
`_bias_and_confidence(düz seri)` **conf=0,2006** verdi ve ders (*"conf 0,00"*) yanlış
göründü. Ama girdi yanlıştı: teste ADX=20 vermiştim, oysa DÜZ seride gerçek ADX **NaN**'dır
(`strategy.adx` ile ölçüldü) ve o değerle conf **0,0006 ≈ 0,00** çıkıyor. Ders DOĞRU.
Kayda geçsin: *bir dersi çürütmeden önce ölçümün girdisi dersin tarif ettiği durumu
gerçekten temsil ediyor mu diye sorulmalıdır* — Tur-39'un "piyasa hipotezinden önce ÖLÇÜM
hipotezini ele" kuralının denetim tarafı.

### Bulunan TEK kusur — içerikte değil, TAŞINMADA

`nihai/ai/episodic.py` `LESSONS` kataloğu **ACI-7'de duruyordu**. CLAUDE.md'de ders **#8**
(sınıf dengesizliği DSR'yi şişirir, 2026-07-25) ve **#9** (kalibrasyon kazancı sabit
tahminciyle kıyaslanmalı, 2026-08-10) aylardır vardı ve kaynak kodda sırasıyla **17** ve
**45** atıf almıştı — #9 en çok atıf alan üçüncü ders.

`check_against_lessons` `agent`/`brain`/`decision`'ın okuduğu **TEK** ders kapısıdır.
Yani AI'ın kendi karar döngüsü, projenin KANIT KAPILARININ bozulmasını anlatan iki dersi
için ASLA uyarı üretemiyordu. "DSR 1.0 çıktı" ya da "Brier düştü" diyen bir iddia kapıdan
sessizce geçerdi.

### Kök neden: BİÇİM genişletildi, VARLIK kurulmadı

Tur-35 boşluğun yarısını görmüştü — `test_ders_ref_format_aci_n` desenini `ACI-[1-7]` →
`ACI-[1-9]` genişletip *"belgeyle yeniden hizalandı"* dedi. Ama:

* desen genişledi, **katalog genişlemedi**;
* test yalnız BİÇİM denetliyordu (`^ACI-[1-9]$`), **VARLIK değil** → `knowledge.py`'deki
  `ders_ref=["ACI-9"]` **boşa işaret ederken yeşil kaldı**;
* `knowledge.py` docstring'i hâlâ *"ACI-1..ACI-7"* diyordu, yani modül KENDİ kodu ile
  çelişiyordu (Tur-42: "bir yorumdaki anahtar adı da API'dir").

### İkinci katman — dersin KONUSU olan kavram onu anmıyordu

`deflated_sharpe` kavramı **#8'in ta kendisi** hakkında ama `ders_ref=[]` idi; `brier`
kavramı **#9'un ta kendisi** hakkında ama `ders_ref=[]` idi. Katalog boş olduğu için bağ
hiç kurulmamıştı. İkisine hem `ders_ref` hem de KURALI SÖYLEYEN tuzak metni eklendi
(işaretçi tek başına yönlendirmez — Tur-42).

### Onarım

* `episodic.LESSONS` += **ACI-8**, **ACI-9** (iddia + guard + kaynak) ve `_LESSON_KEYWORDS`
  karşılıkları.
* **Anahtarlar AYIRT EDİCİ seçildi:** ECE metriği için çıplak `"ece"` eklemek cazipti ama
  Türkçe gövdelerin İÇİNDE eşleşir (`"gelecek"` → `"ece"`) ve kapıyı gürültüye boğardı.
  Her metinde ateşlenen bir uyarı hiç ateşlenmiyor demektir → regresyon kapısı yazıldı.
* `knowledge.py` docstring aralığı 1–7 → **1–9**; "gevşek çapraz referans"ın ne DEMEK
  OLDUĞU yazıldı: IMPORT bağımsızlığı, DENETİMSİZLİK değil.
* `tests/test_tur48_ders_katalogu.py` (**9 test**): katalog ↔ CLAUDE.md **eşitliği** (belge
  numaralarını ayrıştırıp karşılaştırır — bir sonraki ders eklendiğinde bu test KIRILIR ve
  katalog güncellemesini zorlar) · her `ders_ref`in VARLIĞI · her dersin anahtarı VAR ·
  yeni derslerin fiilen ATEŞLENDİĞİ · gömülü-parça eşleşmesi YOK · kavram↔ders bağı ·
  tuzak metninin kuralı taşıdığı.

Çalışma zamanı bağımlılığı DEĞİŞMEDİ: `knowledge` hâlâ `episodic`i import ETMEZ.

Tam süit **2873 passed / 1 skipped** (Tur-47: 2866 → +7), `ruff check .` temiz,
`mypy .` 166 dosya temiz.


## Tur-49 — ÖLÇEK ve OKUMA: MLE yanlış birimde arıyordu, detektörler doğru diziyi yanlış yoldan okuyordu (2026-08-25, Opus 5 max)

Kullanıcı isteği: *"projedeki hataları, eksiklikleri, bugları bulup düzeltir sonrasında
detaylı optimizasyon yapar mısın"*. Kusur avı statik araçlarla BAŞLAMADI (`ruff check .` +
`mypy .` tur başında zaten temizdi, 2873 test yeşildi) — üç bağımsız sorgu koşuldu:
(a) saf matematik kütüphanesi BAĞIMSIZ referanslara (scipy kapalı-form, sonlu-fark, bilinen
DGP) karşı çapraz-doğrulandı, (b) modülün kendi yazılı sözleşmesi (*"hiçbir girdi İSTİSNA
fırlatmaz"*) 1247 dejenere çağrıyla fuzz'landı, (c) backtest sıcak yolu profillendi.
Üçü de kusur buldu. **Statik araçların temiz olması kusur olmadığı anlamına gelmez;
yalnız o araçların BAKTIĞI kusur sınıfının olmadığı anlamına gelir.**

### KUSUR-49-1 — GARCH MLE'si GETİRİ ÖLÇEĞİNDE arıyordu

Bilinen parametreli sentetik GARCH(1,1) serilerinde `garch11_fit` bazı tohumlarda
`alpha=beta=0` (dejenere sabit-varyans) veya `alpha+beta=0,999` (kısıt kenarı) veriyordu.
İddia sentetikte kalmadı — **296 gerçek S&P sembolü**, sembol başına son 1000 günlük
log-getiri (canlı `data/` SALT-OKUNUR):

| ölçüm | ÖNCE | SONRA |
|---|---:|---:|
| fit tamamen `None` (sessiz başarısızlık) | 5 (**%1,7**) | **0** |
| `uncond_var` > 5× örneklem varyansı | 25 (**%8,4**) | **1** |
| `uncond_var` < 0,2× örneklem varyansı | 4 (**%1,4**) | **0** |
| en kötü sapma | **18.068×** ve **1,3e-06×** | 5,13× ve 0,81× |
| `uncond/örneklem` oranı p90 | 3,57 | **1,20** |
| süre / sembol | 658 ms | **81 ms (8,1×)** |

**Kök neden ÖLÇEKTİR.** Günlük getiride varyans ~1e-4'tür. SLSQP'nin varsayılan sonlu-fark
adımı ~1,5e-8 ve `ftol=1e-12`, omega yönünde bu ölçekte ANLAMSIZ kalır; arama omega'yı
1e-12 alt sınırına ya da alpha'yı 0,999'a çakar. GARCH(1,1) **ölçek-EŞDEĞİŞKENDİR**
(r→c·r ise omega→c²·omega, alpha/beta DEĞİŞMEZ), dolayısıyla artıkları birim varyansa
normalize edip omega'yı geri ölçeklemek modeli DEĞİŞTİRMEZ — yalnız optimize edilen yüzeyi
düzeltir. Olabilirlik Jacobian ile geri çevrilir (`loglik −= 0,5·n·ln(var0)`), yoksa alan
ölçekli artıkların olabilirliğini raporlardı.

**Çoklu-başlangıç TEK BAŞINA yetmiyordu.** Docstring beş başlangıç noktasını "GARCH-MLE
yerel-optimum zorluğuna" karşı önlem diye anlatıyordu; ama beş başlangıç da AYNI bozuk
yüzeyde arıyordu. Bir çözümün VAR OLMASI onun DOĞRU EKSENDE olduğunu göstermez.

### KUSUR-49-2 — IGARCH köşesinde `uncond_var` yine de SAYI basıyordu

Ölçek onarımından sonra kalan sapmaların tamamı tek sınıftı: `alpha+beta` durağanlık
KISITININ kenarına (0,999) oturan fitler. Orada `1−(alpha+beta)` tahmin edilmiş bir
büyüklük değil **kısıtın ARTIĞIdır** ve `omega/(1−a−b)` örneklem varyansının 34–528 katını
veriyordu. Docstring *"garch11_fit bu bölgeyi kısıtla eler"* diyordu — ELEMİYORDU,
KIRPIYORDU ve kırpılmış değeri tahmin gibi sunuyordu.

Artık `persistence` alanı açıkça raporlanır ve köşedeki fit `uncond_var=None` döner.
Gerçek veride bu **22/296 (%7,4)** sembolü kapsar — yani her 13 sembolden biri, iki tur
önce Tur-47'nin tarif ettiği hatayı yaşıyordu: *kusuru ifade edemeyen bir alan onu açığa
da çıkaramaz.* Kısa-ufuk `garch11_forecast` bu durumda da GEÇERLİDİR (koşullu varyans
tanımlıdır); tanımsız olan yalnız uzun-ufuk SINIRIdır.

### KUSUR-49-3 — `x ** 2` taşmada İSTİSNA atar, `x * x` atmaz

`quant_math.py` dosya başındaki sözleşme *"Hiçbir girdi İSTİSNA fırlatmaz"* diyor. 1247
dejenere çağrılık fuzz iki ihlal buldu: `ewma_vol` ve `advanced_math.bocpd`, taşma
sınırındaki girdide `OverflowError: (34, "Result too large")` atıyordu. Sebep CPython'a
özgüdür: **float `**` taşmada OverflowError FIRLATIR, float `*` ise numpy gibi `inf`
verir.** İki fonksiyon da etrafındaki numpy koduyla `inf`-tabanlı akış varsayıyor ama iç
döngüde Python `**` kullanıyordu. `ewma_vol` ayrıca taşan yolda `inf` DÖNDÜREBİLİYORDU →
artık `math.isfinite` kapısıyla `None`. Üretimde tetiklenmesi için |r| > 1,3e154 gerekir
(log-getiriden imkânsız) — yani **latent** bir yol; ama sözleşme YAZILIYSA DENETLENİR.
Depodaki 30 küsur `** 2` sahnesi tarandı; yalnız bu ikisi sıkı döngüde skaler-float
üzerinde çalışıyor, diğerleri numpy dizisi veya normalize büyüklük — dokunulmadı (YAGNI).

### OPT-49 — önbellek VARDI, OKUMA hâlâ pandas'tı

`AAPL_1h` 41.237 barlık backtest profili: `pandas.core.indexing.__getitem__` **87,0 s /
162,1 s = %54**. Tur-7 HESABI, Tur-43 KIRPMAYI, Tur-45 ATR ERİŞİMİNİ tekilleştirmişti;
geriye kalan şey OKUMA BİÇİMİYDİ. Ölçüldü: `Series.iloc[int]` **2,573 µs**,
`ndarray[int]` **0,126 µs** → **20,4×**.

Çağıran dağılımı (8.000 barlık koşuda `.iloc`/dilim çağrısı):

| çağıran | önce | sonra |
|---|---:|---:|
| `detect_rsi_divergence` | **74.102** | 0 |
| `meta_features.extract_features` | 35.319 | ~0 |
| `_SharedSlice.ema_at` | 31.448 | 0 |
| `_SharedSlice.bb20` kırpması | 23.019 | 0 |
| `_SharedSlice.macd` kırpması | 20.886 | 0 |
| `detect_donchian` | 19.365 | 0 |
| `_SharedSlice.adx_last` | 18.992 | 0 |

`detect_rsi_divergence` AÇIK ARA birinciydi: bar başına **9,3** pandas dilim/etiket-araması
(iki `.iloc[-30:]` görünümü + 8 yarı-pencere min/max/idx + 2 etiketli `.loc`) ve okuduğu
bilgi **SEKİZ SAYI**. Yeni gövde iki sıfır-kopya numpy penceresiyle çalışır; tek başına
**6,44×** (736,8 → 114,5 µs/bar, 55.160 gerçek barda FARK 0).

`detect_donchian`'ın penceresi zaten `breakout_levels(20)` önbelleğinde VARDI —
`detect_event` onu kullanıyordu, donchian kendi `.iloc[-21:-1].max()`'ını kuruyordu.
Yeni hesap EKLENMEDİ, var olan PAYLAŞILDI.

**Uçtan uca A/B (aynı süreç, aynı yapılandırma, profilsiz, 20.000 bar/sembol):**

| sembol | ÖNCE | SONRA | hızlanma | metrikler |
|---|---:|---:|---:|---|
| AAPL | 39,87 s | 15,43 s | **2,58×** | birebir aynı |
| MSFT | 40,78 s | 16,65 s | **2,45×** | birebir aynı |
| AMZN | 40,38 s | 16,41 s | **2,46×** | birebir aynı |
| GOOGL | 41,21 s | 16,67 s | **2,47×** | birebir aynı |
| NVDA | 1,46 s | 1,20 s | 1,22× | birebir aynı |
| **TOPLAM** | **163,70 s** | **66,36 s** | **2,47×** | — |

"Birebir aynı" = işlem sayısı, `total_return_pct`, `win_rate`, `max_drawdown_pct` ve
`sharpe_annualized` 8 ondalığa kadar EŞİT. Profilli tek-sembol koşusunda çağrı sayısı
**171,5M → 32,0M (5,4×)**, süre 162,18 → **57,52 s**.

**A/B'nin İLK denemesi GEÇERSİZDİ ve bu kayda geçmelidir.** Taban ölçümü `git worktree` ile
HEAD'de alındı; sonuçlar farklı çıktı (AAPL 7 işlem ↔ 9 işlem) ve bir an "optimizasyon
davranışı değiştirdi" gibi göründü. Sebep koddaydı DEĞİL ORTAMDAYDI: worktree'de `.env`
YOK (gitignore'da) → `CONFIG` başka eşiklerle yüklendi. **Bir A/B, iki tarafın
YAPILANDIRMASININ aynı olduğu kanıtlanmadan okunamaz.** Ölçüm aynı süreç içinde
monkeypatch ile yeniden kuruldu.

**Denklik AYRICA fonksiyon düzeyinde ölçüldü:** 5 detektör × 29.200 gerçek bar × iki yol
(`shared=_SharedSlice` ve `shared=None`) → **FARK 0**; dağılım boş değil (macd 2.149,
donchian 2.345, mean-rev 570, squeeze 145 tetikleme). `extract_features` HEAD sürümüne
karşı 3.416 çağrı × 25 özellik = **85.400 değer, maxdiff 0,0**.

### Bu turda ÖLÇÜLDÜ ama DEĞİŞTİRİLMEDİ (dürüstlük bölümü)

* **`ou_half_life` AR(1) yarı-ömrünü ~%5 sapmayla verir** (phi=0,9'da 6,93 ↔ teorik 6,579).
  `−ln2/b` (b = OLS eğimi ≈ phi−1) yerine `−ln2/ln(phi)` doğru olurdu; ama docstring
  yöntemi *"pairs-trading konvansiyonu"* diye AÇIKÇA belirtiyor ve mevcut kalibrasyonlar
  ona göre. Konvansiyon değiştirmek bir ONARIM değil bir KARAR — kullanıcınındır.
* **`check_exit` hedefte gap'i AÇILIŞTAN doldurmuyor** (stop'ta dolduruyor): LONG'ta
  `bar_open >= target` durumunda fill `target`'tır, `bar_open` değil. Yön MUHAFAZAKÂRDIR
  (kazancı EKSİK gösterir) — düzeltmek backtest getirisini ŞİŞİRİR, bu depoda ters yönde
  bir "iyileştirme". Dokunulmadı.
* **`garch11_fit`'in TEK üretim çağıranı `ai/lab.py`'dir** (`_run_vol_model_karsilastirma`)
  ve `uncond_var` alanını bugün OKUYAN üretim kodu YOKTUR — yani KUSUR-49-1/2 bugün karar
  yolunu bozmuyordu. Onarım yine de yapıldı: kütüphane yanlış sayı basıyordu ve Tur-38
  kuralının tersi de geçerlidir — bugün okunmayan alan yarın okunur.
* **FICO'nun `uncond_var`'ı hâlâ örneklem varyansının 5,13 katı** (persistence 0,996,
  köşede DEĞİL). Bu bir kusur değil, yüksek kalıcılıkta `1/(1−a−b)` teriminin GERÇEK
  duyarlılığıdır — kırpmak KUSUR-49-2'nin ta kendisini geri getirirdi.
* **Backtest süresinin kalan darboğazı `pandas.rolling`'dir** (7.109 çağrı, 4,85 s).
  Gösterge HESABININ kendisi; Tur-7 onu zaten sembol başına BİRE indirdi. Daha ileri
  gitmek pandas'ı gösterge hattından çıkarmak demektir — ayrı ve büyük bir karar.
* **`hurst_exponent`, `variance_ratio`, `evt_hill_tail_index`, Black-Scholes fiyat/greekler,
  `copula_fit_gaussian`, `frac_diff_ffd`, `permutation_entropy`, `mutual_info` bağımsız
  referanslara karşı DOĞRU çıktı** (BS fiyatı 4 senaryoda ≤3,6e-15; delta/gamma sonlu-farkla
  1,1e-09/2,3e-05; rastgele yürüyüşte H=0,5030, VR(2)=1,014; Pareto(α=3)'te Hill 2,992).
  Bu tur bu fonksiyonlarda kusur BULAMADI ve bu da bir ölçümdür.

### Kilit

`tests/test_tur49_garch_olcegi_ve_gosterge_okumasi.py` (**39 test**): ölçek eşdeğişkenliği ·
12 tohumda `uncond_var` bandı · ARCH etkisi olmayan dört uzunlukta sağlamlık · Jacobian
düzeltmesi · lfilter ↔ Python döngüsü BİT-DENKLİĞİ · IGARCH köşesinde None + kısa ufkun yine
çalışması · taşmada istisna yokluğu · numpy erişimcilerin seri karşılıklarıyla eşitliği ·
kırpılmış görünümün GELECEĞİ İÇERMEMESİ · donchian penceresinin `breakout_at` ile denkliği ·
detektörlerin pandas referansıyla aynı kararı · `extract_features` paritesi · önbellekli ↔
önbelleksiz yol eşitliği · **dört regresyon kapısı** (eski ölçeksiz MLE çağrısı,
`float(r) ** 2`, detektörlerde `.iloc[`, `extract_features`'ta ham sütun dilimleme).

**Kapının kendisinde bir tuzak vardı ve çözümü derse dönüştü:** bu turun onarım YORUMLARI
kusurun eski ifadesini (`.iloc[-30:]`) bilerek ALINTILIYOR. Ham metinde arayan bir kapı
kendi açıklamasına takılır ve "kusur geri geldi" der. `_kod()` yardımcısı `tokenize` ile
yorumları ve dizgi sabitlerini atar → kapı KODA bakar, ANLATIYA değil. Tur-48'in dersi:
her metinde ateşlenen bir uyarı hiç ateşlenmiyor demektir.

Tam süit **2912 passed / 1 skipped** (Tur-48: 2873 → **+39**), `ruff check .` temiz,
`mypy .` 166 dosya temiz.

## Tur-50 — ÖLÜ KAPILAR: geçici bir koşul KALICI bir mandal kuruyordu (2026-08-27, Opus 5 max)

Kullanıcı isteği: *"projeye detaylı optimizasyon yapar mısın"*. Tur başında üç kapı da
temizdi (2912 test yeşil, `ruff check .` temiz, `mypy .` 167 dosya temiz) — yani Tur-49'un
dersi bir kez daha sınandı: **statik araçlar BAKTIKLARI kusur sınıfının yokluğunu söyler.**
Bu turun sorgusu yeni bir eksendi: *bu depoda VAR OLAN, BELGELENEN, ama HİÇBİR KOD YOLUNDAN
OKUNMAYAN kaç kural var?* Tarama `ast` ile yapıldı (126 config alanı + `nihai/` altındaki
605 public tanım) ve **dokuz** ölü öğe buldu; biri yalnız ölü değil ZARARLIYDI.

**Yöntem notu — tarayıcı ilk denemede KENDİ dersine takıldı.** İlk sürüm regex'ti ve
`htf_bar_minutes`i "okunuyor" saydı: eşleşme bir YORUMDAN geliyordu (`…(HTF). htf_bar_minutes
(StrategyFilters) ile uyumlu tutulmalı`). Tur-49 tam bunu yazmıştı (*"kaynak metnini tarayan
bir kapı KODA bakmalı, ANLATIYA değil"*) ve bir tur sonra ilk ihlal eden ben oldum. `ast`'ye
geçildi: ayrıştırma sonucunda yorumlar ve dizgi sabitleri zaten YOKTUR.

### KUSUR-50-1 — hız sınırı GEÇİCİDİR, mandal KALICI; ikisi tek bayrağa binmişti

`CircuitBreaker.can_emit` hız sınırına takıldığında `_tripped = True` kuruyordu. Ama hız
sınırı bir PENCERE koşuludur ve pencere kendi kendini temizler (60 sn / 1 saat); mandal ise
yalnız `reset()` ile açılır — ve `reset()`in depoda **HİÇBİR ÇAĞIRANI YOKTU**. Sonuç: canlı
tarayıcı süreci `breaker`ı bir kez kurar (`main.py`, uzun ömürlü döngü, watchdog altında
günlerce), tek bir yoğun dakika mandalı indirir ve **süreç yeniden başlayana kadar hiçbir
sinyal yayılmaz.**

Bu latent bir yol DEĞİLDİ. Canlı karar loglarında (`logs/decision_log_*.txt`) ölçüldü:

| ölçüm | değer |
|---|---:|
| GERÇEK tetikleme (mandal ÖNCESİ: `RATE_LIMIT_MIN_5`×5, `RATE_LIMIT_HOUR_60`×1) | **6** |
| MANDALLI blok (`TRIPPED: …`) | **9.194** |
| tüm blokların mandal payı | **%99,93** |
| etkilenen gün | **10** |
| en uzun kesintisiz mandal penceresi (2026-07-24) | **275 dk**, 2.349 blok |
| 2026-07-07 penceresi | 17:39:56 → 19:47:25 (**128 dk**), 2.815 blok |

**Mandal takvim günlerini aştı:** 07-21, 07-22, 07-23 ve 07-24 loglarında **tek bir gerçek
tetikleme YOK** — yalnız `TRIPPED:` satırları var, yani mandalı indiren dakika ÖNCEKİ günde
yaşanmıştı ve süreç öyle devam etmişti.

Onarım yeni bir mekanizma değil, **birbirine yapışmış iki mekanizmanın AYRILMASIDIR**:
hız sınırı artık yalnız o çağrıyı reddeder (mandal kurmaz), `_tripped` yalnız Madde 47'nin
bilinçli "sinyalleri durdur" kararını taşır. `_tripped` okuması kilidin İÇİNE alındı
(eskiden kilit dışındaydı).

**Onarımın ALT SINIRI ölçüldü** (üst sınır değil): yeni davranışta bir çağrı ancak önceki
60 sn'de ≥5 YAYIN varsa reddedilir; yayın sayısı ≤ deneme sayısı olduğundan, önceki 60
sn'sinde 5'ten az DENEME bulunan her mandallı blok yeni kodda **kesinlikle** geçerdi:
**1.609 / 9.194 = %17,5**. Gerçek kurtarma bundan büyüktür (mandal sonrası yayın sayısı
sıfırdı, deneme değil) — ama %17,5 kanıtlanabilir tabandır.

### KUSUR-50-2 — Madde 47'nin acil müdahale kancası ERİŞİLEMEZDİ

`CircuitBreaker.trip()` ve `reset()` — modül docstring'inde *"Madde 47: Acil manuel müdahale
hook'u (flatten-all yerine 'sinyalleri durdur')"* diye yazılı olan kancanın ta kendisi —
`nihai/`, `research/` ve `tests/` genelinde **sıfır çağıran** taşıyordu. Yani operatörün
sinyalleri durdurma yolu YOKTU; sistemi durduran tek şey, onu YANLIŞLIKLA durduran hız
sınırıydı.

Kanca var olan yüzeye bağlandı — yeni soyutlama YOK: `execution/main.py`'deki mevcut
`tg_handlers` tablosuna `/dur` ve `/devam` eklendi (+`/help` metni). `/dur` yalnız YAYINI
durdurur; **açık pozisyonlara DOKUNMAZ** ve bu bir test ile kilitli: bir operatör freninin
zarar-kesmeyi geciktirmesi bu sistemdeki en pahalı hata olurdu (Tur-38'in `ai_advisor`
için kurduğu aynı sınır).

### KUSUR-50-3 — `RiskManager.margin_ok`: Madde 13'ün adı konmuş kapısı, çağıranı yok

`margin_ok(position_notional) -> bool` (Reg T: `notional × 0,50 ≤ 2 × equity`) depoda hiçbir
yerden çağrılmıyordu. Ama Reg T bağı **uygulanıyor** — `_magnitude_cap_units` (a) maddesinde,
üstelik FARKLI bir davranışla: `margin_ok` adayı REDDEDER, `_magnitude_cap_units` tavana
sığacak şekilde KÜÇÜLTÜR (`cap_aware_sizing`, Tur-40'ta bilinçle seçilmiş yön). Yani ölü kopya
yalnız fazlalık değil, yürürlükteki kararla ÇELİŞEN bir ikinci tanımdı. Silindi; kural tek
yerde yaşıyor.

### KUSUR-50-4 — okuyanı olmayan beş config alanı (126 alanın 5'i)

| alan | ne sanılıyordu | gerçek |
|---|---|---|
| `OptionFilters.min_dte=5` / `max_dte=60` | DTE penceresi ("0DTE'ye dokunma") | okunmuyor; yürürlükteki pencere `OptionStrategyConfig.dte_min=30`/`dte_max=45` ve **çelişiyor** |
| `OptionStrategyConfig.close_before_expiry_dte_override` | `None → CONFIG.options.close_before_expiry_dte` zinciri | zinciri hiçbir kod kurmuyordu; `check_exit` global alanı doğrudan okuyup override'ı **sessizce yok sayıyordu** |
| `StrategyFilters.htf_bar_minutes=60` | cooldown'un bar uzunluğu | `state.py` onu `INTERVAL_MINUTES[HTF_INTERVAL]` ile TÜRETİYOR; kopya okunmuyor |
| `Regulation.finra_maintenance_margin=0,25` | Madde 13'ün sürdürme yarısı | uygulanmıyor |

İki tanesi yalnız ölü değil YANILTICIYDI. `min_dte`/`max_dte` çifti 5–60 diyordu, gerçek
pencere 30–45'tir: `max_dte`yi 30'a çeken bir operatör sistemin hâlâ 45 günlük vade seçtiğini
göremezdi. `close_before_expiry_dte_override` ise bir SÖZLEŞME tarif ediyordu — Tur-42'nin
dersi (*"bir yorumdaki anahtar adı da API'dir"*) burada tersinden doğrulandı.

Kararlar ayrıştırıldı: **override BAĞLANDI** (`check_exit` artık zinciri kuruyor; varsayılan
`None` olduğu için davranış BİT-DENK ve bu dört senaryoyla üretim yolundan kilitlendi),
**diğer dördü SİLİNDİ** ve yerlerine yürürlükteki kaynağa işaret eden birer satır kondu.

`finra_maintenance_margin` için karar ölçümle verildi. Sürdürme marjı PORTFÖY düzeyi BRÜT
maruziyet ölçer; canlı defterden brütü kurmayı denedim (527 gerçek işlem, 2026-06-26→08-27,
`entry_price × |qty|` = kodun `notional` tanımıyla birebir) ve **brüt/özkaynak ortanca 5,80×,
azami 13,68×** çıktı — %25 sürdürme sınırı (brüt ≤ 4×) gözlemlerin %65'inde aşılmış görünüyordu.
**Bu okuma GEÇERSİZDİR ve kayda öyle geçmelidir:** `.env`'de `DUAL_TIER_ENABLED=true` ve
`trades.csv` iki katmanın (`event_breakout V1` 431 ↔ `EventBreakout V1` 96) işlemlerini TEK
deftere yazıyor; iki BAĞIMSIZ portföyün toplamı, tek portföyün tavanıyla kıyaslanamaz.
Tur-49'un dersi (*"bir A/B, iki tarafın aynı olduğu kanıtlanmadan okunamaz"*) burada
kıyaslamanın ÖZNESİNE uygulandı. Sonuç: kural yalnız uygulanmıyordu değil, **ihlali de
ölçülemiyordu** — bu yüzden yeni bir kapı EKLENMEDİ (ölçülemeyen bir kapı, bu turun
kapattığı kusurun ta kendisi olurdu); sabit silindi ve durum PDT notundaki duruşla aynı
biçimde yazıldı: hesap düzeyinde broker uygular.

### KUSUR-50-5 — tarayıcının kendi ortaya çıkardığı kusur: geçersiz escape

Yeni test dosyası depoyu `ast` ile ayrıştırınca bir `SyntaxWarning` düştü:
`nihai/console.py:10` docstring'inde `logs\*.log` yazıyor, ham dizgi değil. Python 3.12'de
uyarı, gelecek bir sürümde **SyntaxError**. Ne `ruff` (W605 bu projenin `select` listesinde
YOK: E,F,I,B,UP) ne `mypy` bakıyordu. 318 dosyada **tek** örnek — ve ironi kayda değer:
uyarıyı üreten modül tam da CLI çıktısının kodlama GÜVENLİĞİ için yazılmış olandır (Tur-24).

### Bu turda ÖLÇÜLDÜ ama DEĞİŞTİRİLMEDİ (dürüstlük bölümü)

* **`notifier/telegram_bot.heartbeat()` de ölü** (Madde 36) — `main.py` heartbeat'i
  `telegram_bot.send(...)` ile ELDE kuruyor ve NLV/pozisyon/PnL taşıyor; `heartbeat()`
  yalnız zaman damgası basar. Yani ölü olan, YERİNE geçen daha zengin bir çağrının
  bulunduğu bir kopyadır. Silmek doğru olurdu ama `nihai/risk` dışında ve bu turun
  kapısı (KAPI 2) bilinçli olarak YALNIZ para yolunu tarıyor: kapıyı tüm depoya açmak
  onlarca meşru genel-API tanımını (opsiyonel yardımcılar, CLI girişleri) yakalardı ve
  gerekçesiz büyüyen bir izin listesi kapıyı öldürürdü. Genişletme ayrı bir karardır.
* **`can_emit` mandalı ölçerken `record_emit` ile YARIŞ hâlindeydi** — `_tripped` okuması
  kilit dışındaydı. Kilide alındı; ama bu bir kusur RAPORU değil, onarımın yan etkisidir:
  eski kodda da yarış zararsızdı (mandal tek yönlüydü).
* **Hız sınırının DEĞERİ (`max_signals_per_minute=5`) sorgulanmadı.** 6 gerçek tetikleme,
  eşiğin canlı tarama hızına göre düşük olabileceğini düşündürüyor; ama eşik bir KARARDIR
  ve mandal kusuru giderildikten sonra üretilecek yeni defterle ölçülmelidir. Bu turda
  eşik DEĞİŞTİRİLMEDİ.
* **Brüt maruziyet ölçülemezliği açık kalemdir.** DUAL_TIER katmanları ortak `trades.csv`'ye
  katman etiketi YAZMIYOR (`strategy` alanındaki iki yazım biçimi bir tesadüftür, sözleşme
  değil). Portföy düzeyi risk soruları bugün bu depoda yanıtlanamaz; kolonu eklemek bir
  loglama politikası kararıdır (Tur-36'nın "etiketsiz çoğunluk" teşhisiyle aynı sınıf).
* **`config.py`'nin 126 alanının kalan 121'i OKUNUYOR** — tarama negatif sonuç da üretti ve
  bu da bir ölçümdür. Aynı şekilde `nihai/` altındaki 605 public tanımdan yalnız 4'ü
  çağrılmıyordu (`margin_ok`, `trip`, `reset`, `heartbeat`); üçü bu turda kapandı.

### Kilit

`tests/test_tur50_olu_kapilar.py` (**17 test**): config alanlarının hepsinin bir OKUYANI ·
silinen dörtlünün ALAN olarak geri gelmemesi · DTE penceresini uygulayan yerin TEK olması ·
`nihai/risk`'te çağıranı olmayan public tanım bulunmaması · hız sınırının mandallamaması
(dakika + saat, ikisinde de pencere daralınca yayının KENDİLİĞİNDEN geri gelmesi) ·
`can_emit` gövdesinde `_tripped` ATAMASININ olmaması (AST regresyon kapısı) · mandalın elle
kurulup yalnız `reset` ile temizlenmesi · `/dur`+`/devam`ın `tg_handlers`'ta KAYITLI olması ·
gövdelerinin gerçekten `trip`/`reset` ÇAĞIRMASI · `/dur`un çıkış yoluna DOKUNMAMASI ·
hiçbir kaynağın geçersiz escape taşımaması · DTE override zincirinin `check_exit`'te
GERÇEKTEN kurulu olması + dört senaryoda üretim yolundan ölçülmesi.

**İki kapı BİÇİM değil VARLIK denetler** (Tur-48'in dersi): "alan tanımlı mı" değil "alanı
OKUYAN var mı", "komut tabloda mı" değil "gövdesi doğru çağrıyı YAPIYOR mu". Ve tarama
`ast` üzerindedir — bu bölümdeki onarım yorumları silinen alanların adlarını bilerek
alıntılıyor; ham metinde arayan bir kapı kendi açıklamasına takılırdı.

**Bu depo aynı sınıfı ELLE iki kez kapatmıştı.** `risk_manager.portfolio_caps_breach`'in
kendi yorumları söylüyor: (d) *"daha önce config'de tanımlı ama uygulanmıyordu"*, (e)
*"config'de ölüydü"*. Kural o zaman MAKİNEYE çevrilmediği için geri büyüdü — Tur-24'ün
(*"kuralı makineye çevir"*) ve Tur-28'in (*"bir kural iki kapıdan yalnız birine konursa…"*)
dersi, bu kez kuralın KENDİSİNE uygulandı.

Tam süit **2929 passed / 1 skipped** (Tur-49: 2912 → **+17**), `ruff check .` temiz,
`mypy .` 167 dosya temiz.

