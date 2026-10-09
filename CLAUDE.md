# CLAUDE.md — Nihai Algoritma proje asistanı

Bu dosya, Claude'un bu projede çalışırken bağlamı her oturumda yeniden okumasını sağlar.
Projeye özel "AI"nın ilk katmanı budur: Claude'u bu trading sisteminin uzman geliştirici/analiz
asistanına dönüştürür.

## Sistem nedir

**Sinyal-only / raporlayıcı** hisse & opsiyon sistemi. **Emir GÖNDERMEZ** — tarar, kararı puanlar,
log + bildirim üretir. Mimari kaynağı: `Nihai_Algoritma_Mimarisi_Hisse_Opsiyon_v2_2.pdf` (53 madde,
6 bölüm). README'deki "Madde Eşlemesi" tablosu hangi PDF kuralının hangi modülde olduğunu söyler.

- Python `>=3.11`. Paketler: `nihai/` (üretim; `nihai/notifier`, `nihai/web` dahil), `research/` (deney/ML), `tests/`.
- Canlı iki süreç (`proje-ac.ps1` açar, `proje-kapat.ps1` kapatır): veri hub'ı `nihai.data.hub_server` (8799, Alpaca
  akışı) + kayıtçı/pano `nihai.execution.main --serve-web` (8765; tarama, tahmin servisi, gölge kasalar). `MainHub/`
  paylaşılan sağlayıcı dosyalarının türetilmiş kopyasıdır (`tools.sync_mainhub`); `run-duo.ps1` = `research.duo` A/B.

## Komutlar

```bash
pytest                              # tüm testler (testpaths=tests, -q); ~250 dosya
pytest tests/test_strategy.py       # tek dosya
ruff check .                        # lint (E,F,I,B,UP; line-length 110)
ruff check --fix .                  # otomatik düzelt
pytest --cov=nihai --cov=research --cov-report=term   # kapsama görünürlüğü (P2-3; hedef/kapı değil)
python -m nihai.execution.main      # tarayıcıyı çalıştır (PowerShell: run.ps1)
python -m nihai.notifier.daily_summary   # günlük sistem-sağlığı özeti (--send Telegram, --llm akıcılaştır)
python -m tools.sync_mainhub [--apply]      # MainHub paylaşılan dosyalarını nihai/'dan ÜRET (Tur-58)
python -m tools.pano_i18n [--hepsi]         # pano İngilizce sözlüğünde (pano_en.json) eksik metinler (Tur-108)
python -m nihai.data.market_calendar [--at] # ABD seansının YEREL karşılığı (zamanlayıcı tetiği)
python -m nihai.log_retention [--apply]     # logs/ sıkıştırma (×15,1 ölçüldü) + saklama
python -m nihai.notifier.deadman [--send --ping]   # ÖLÜ ADAM: kanıt akışı durduysa uyar (Tur-59)
#   .\deadman.ps1 [-Send -Ping]              # ^ sarmalayıcı (loglar; Tur-61)
#   .\register-deadman.ps1 [-Minutes 15]     # ^ 15 dk'lık görev (INTERACTIVE, admin gerekmez)
python -m nihai.data.option_archive --symbols SPY,QQQ  # OPRA zinciri GÜNLÜK arşivi (OI+hacim)
python -m nihai.data.market_structure --warm AAPL,MSFT # float önbelleğini doldur (turnover)
python -m nihai.data.edgar_facts AAPL --refresh  # SEC EDGAR XBRL temelleri (Tur-76, ANAHTARSIZ)
python -m nihai.data.factors --momentum      # Ken French faktör serisi (Tur-76, ANAHTARSIZ)
python -m nihai.data.macro_regime            # 13 makro göstergenin FRED önbelleğini doldur + bas
python -m nihai.execution.replay --symbols AAPL,MSFT --from 2026-09-11 --to 2026-09-11 --out <boş dizin> [--scan-every 5] [--fetch]
#   ^ Tur-77: canlı scan_symbol hattında İZOLE tarihsel replay (alt süreç + yazma çiti; üretim data/logs'a yazmaz)
python -m nihai.tax_report --lot-method hifo  # Form 8949 dökümü: position|fifo|hifo (Tur-77)
python -m nihai.data.fda_calendar --pdufa --since 2026-06-14 --until 2026-09-13  # İLERİ PDUFA takvimi (SEC 8-K/6-K; Tur-78)
python -m nihai.ai.adaptive_calibration --eval --method venn_abers   # Venn–Abers aralık teşhisi (Tur-77)
python -m research.calibrate_forecast        # "Olasılık (kal.)" eğrisi: CANLI defter, yürüyen + taban-oran kapısı (Tur-83; gece görevi koşar)
#   ^ + ~%80 fiyat ARALIĞI (Tur-84: ölçek ATR14, çapa bar kapanışı; Tur-85: ATR^us, üs yürüyen kapıyla)
#   python -m research.calibrate_forecast --bant-barlardan   # aralık önyüklemesi: atr_pct'siz satırlar IEX'ten
python -m research.uncertainty_slice_probe    # belirsizlik ölçüleri isabeti doğru yönde sıralıyor mu (Tur-77)
#   .\saat-onar.ps1 [-OlcumSadece]           # sistem saati kayması (YÖNETİCİ; ölçer + onarır)
python -m nihai.signals.breadth              # piyasa genişliği modülü öz-denetimi
curl localhost:8765/metrics                 # Prometheus metrikleri (uptime/RSS/çağrı-hızı/döngü)
```

ML / araştırma — repo kökünden **`-m` ile** çalıştır (paket olarak; `python research/x.py`
`ModuleNotFoundError: nihai` verir çünkü kök sys.path'te değil):

```bash
python -m research.build_meta_dataset                   # backtest'ten meta-labeling eğitim CSV'si
python -m research.train_meta --model all --split both  # logistic+gbm+xgb, time+symbol OOS (tek bölme)
python -m research.purged_cv --model both --folds 5     # purged+embargolu K-fold OOS (sağlam) + özellik önemi
python -m research.run_pit_validation                   # point-in-time (survivorship-temiz) doğrulama
python -m research.forward_harness                      # ileri/forward değerlendirme
python -m research.build_live_dataset                   # CANLI loglardan eğitim CSV'si (survivorship-temiz)
python -m research.live_retrain                         # TEK KOMUT: canlı veri → guardrail →
                                                        #   time+symbol kapı + purged CV → verdikt
python -m research.live_retrain --promote               # yalnız DAYANIKLI verdiktte modeli kaydet
python -m research.auto_retrain                         # günlük ADAY eğitimi (gece görevi koşar; Tur-88):
                                                        #   data/retraining/<run_id>/, canlı modele TERFİ ETMEZ
python -m research.backfill_iv_history --from-existing  # IV Rank geçmişini DoltHub'dan doldur
python -m research.edge_lab                           # kenar-keşif DEFTERİ: koşulan aday
                                                        #   aileleri + sıradaki DSR n_trials
python -m research.feed_ab_test --symbols AAPL,SPY      # IEX-SIP bar A/B (Tur-59; ağ ister)
python -m research.piyasalar --indir                   # 14 piyasa grubu önbelleği + kalite (Tur-96)
python -m research.coklu_piyasa --asama 1 --isci 8     # AI'lar ↔ algoritmalar çoklu piyasa koşusu;
                                                        #   sonra --asama vol|2|3|rapor|ozet (Tur-96)
python -m research.quant_kayma --asama motor --isci 8  # quant_trend ↔ TA aynı motorda, 14 piyasa + canlı 30 sembol,
                                                        #   ÖD 2009-17 + 1 bar gecikme; sonra --asama sigma|rapor|ozet (Tur-97)
python -m research.quant_ml                            # ML ↔ quant çekirdek: trend zamanlaması + oynaklık, 2 pencere (Tur-97)
python -m research.quant_ml_motor --asama model       # ML onarımı (Tur-98): σ_ML21 + meta kapı yürüyen modeller; sonra
                                                        #   --asama motor|rapor (motor kolları + ön-kayıtlı kapı, karne)
python -m research.quant_karar --asama mde           # ML karar mekanizması (Tur-99): birleşik kapının MDE'si; sonra
                                                        #   --asama motor|rapor (VY boyut laboratuvarı + birleşik z, karne)
python -m research.ileri_sigma_ml21 guncelle       # σ_ML21 ↔ EWMA ön-kayıtlı İLERİ sınama (kilit 2026-09-27; haftalık elle,
                                                        #   ~150 sembol indirir); ön-kayıt docs/ONKAYIT_SIGMA_ML21_2026-09-26.md DÜZENLENMEZ
python -m research.quant_ml_derin --asama model      # ML derinleştirme (Tur-100): kayıp/doğrulama/yeni bilgi/adaptasyon;
                                                        #   sonra --asama motor|teshis|rapor (karne data/quant_ml_derin_karnesi.json)
python -m research.quant_cekirdek --asama motor --isci 8  # quant çekirdeği (Tur-101): çıkış/skor boyutu/FX, sembol başı 3 pencere;
                                                        #   sonra --asama portfoy|parite|rapor (tek portföy, canlı tavanlar, Z doğrulaması)
python -m research.noral_282m_backtest --asama indir   # canlı gölge 282M sinir ağının 10 yıllık backtest'i + kıyası (Tur-107):
                                                        #   SIP 1m, 30 sembol (~1 sa); sonra --asama veri|tahmin|mc|kiyas|rapor
                                                        #   (mc = canlı köprünün kendi çağrısı, ~5 satır/sn; karne data/noral_282m_karne.json)

```

## Dış veri kaynakları (2026-08-13 eklendi)

Yeni bağımlılık yok, anahtar gerekmez, ikisi de ÜCRETSİZ. Rapor: `docs/VERI_KAYNAKLARI_2026-08-13.md`.

- **`nihai/data/dolt_options.py`** — DoltHub `post-no-preference/options`
  (2019-02-09→bugün, 2321 sembol, 1.898.469 satır). `volatility_history` (IV/HV +
  yıl yüksek/düşük) ve `option_chain` (GERÇEK bid/ask + tam greeks). Dolt **sürümlü**
  olduğu için `as_of="HEAD~50"` gerçek point-in-time okuma verir (doğrulandı: AAPL
  HEAD 1254 ↔ HEAD~50 1229 satır) — ACI DERS #1'e karşı yapısal koruma.
  **Üretim yolu toplu CSV'dir** (`bulk_csv`, 217 MB / 67 sn, tek istek); SQL API
  spot sorgu içindir ve hız sınırlıdır (403 görüldü). SORGU BİÇİMİ KURALI modül
  docstring'inde: PK öneki dışı `ORDER BY` 123× yavaşlatır ve çok-sembollü `IN(...)`
  zaman aşımına girer → tek sembol sor, sıralamayı istemcide yap.
- **`nihai/data/cboe_data.py`** — VIX/VIX9D/VIX3M/VIX6M/VVIX/SKEW (VIX 1990→bugün)
  + `term_structure()` rejim özellikleri (contango/backwardation, VIX yüzdelik dilimi
  GENİŞLEYEN pencereyle = look-ahead yok). Endeks düzeyi → survivorship yapısı gereği yok.
  **Canlı sinyale BAĞLANMADI ve bu bir KARARDIR**: `strategy._load_vix()` yfinance'te
  KALDI (Tur-74 ölçümü — aşağıya bak). Vade yapısı Tur-74'te `/api/quant/regime`
  (`vade_yapisi` bloğu) + pano üstünden KARAR-DIŞI rapor olarak bağlandı; uç
  `term_structure(fetch=False)` ile ÖNBELLEK-YALNIZ okur (BULGU-59-8), önbelleği
  doldurmak ayrı adımdır: `python -m nihai.data.cboe_data`. `far_slope` (VIX6M/VIX)
  Tur-74'te eklendi ve **ffill EDİLMEZ** (VIX6M düzensiz yayımlanır; ölçüldü 2026-09-06:
  VIX 09-04'e, VIX6M 08-12'ye kadar → uç `far_slope_tarih`i AYRICA taşır).
  **KAPI ÖLÇÜLDÜ ve KURULMADI (§32):** dış listenin *"backwardation swing alımlarını
  kilitlesin"* önerisi `vixts_v1` ailesiyle kanıt kapısına sokuldu (43.066 satır / 595
  sembol / 1277 gün; taban kural 20-bar kırılma, ufuk 5) → **purged_cv 0,505 · gate_time
  t=−1,39 · DSR 0,0005 · exec_lag t=−0,77 = KENAR YOK**. `signals`/`risk`/`execution`
  altında `cboe_data` importu testle YASAK.
  **A/B ÖLÇÜLDÜ (Tur-31):** 2.514 ortak günde korelasyon **0,99997**, \|fark\|<0,01 olan
  gün **%99,88**, rejim eşiklerinde (15/20/25/30) ayrışan karar **0/1/0/0**; 0,01'den
  fazla ayrışan yalnız 3 gün var (en büyüğü 2026-02-06: 20,37 ↔ 17,76). Yani göç ölçülmüş
  sonuçları geçersiz KILMAZ ve CBOE 1990'a kadar gider (yfinance yolu yalnız 2016'ya) =
  **26 yıl ek geçmiş**. Göç yine de ayrı ve bilinçli bir adımdır; bu tur YAPMADI.

**`option_chain` TOPLU CSV DİSKTE (Tur-54) — ve OneDrive DIŞINDA.** `volatility_history`
(230 MB) `data/dolt_cache/` altındadır; `option_chain` **~10 GB**tır ve oraya konulamaz:
`data/` OneDrive ile senkronlanır ve 10 GB'lık bir ARAŞTIRMA ÖNBELLEĞİ senkronlanacak bir
şey değildir (kotayı ve yükleme bandını yer). Modülün zaten `DOLT_CACHE_DIR` env değişkeni
vardır:

```bash
DOLT_CACHE_DIR='C:\Users\Kagan\dolt_cache' python -c "from nihai.data import dolt_options as D; D.bulk_csv('option_chain')"
```

Ölçüldü: ~3,2 MB/sn, tek istek, hız sınırı tetiklenmiyor. Dosya **birincil anahtar
sırasındadır** (2.000.000 satırda 57.178 bitişik grup, sonradan tekrar eden anahtar **0**,
tarih monoton artan) → akış hâlinde okuyan bir tüketici, parça sınırındaki son grubu bir
sonraki parçaya DEVREDEREK tam ölçüm yapabilir (`build_optskew_dataset.yuzey_serisi`).
Şema: `date · act_symbol · expiration · strike · call_put · bid · ask · vol(IV) · delta ·
gamma · theta · vega · rho`. **AÇIK POZİSYON (OI) ve HACİM KOLONU YOKTUR** → put/call
ORANI bu kaynaktan hesaplanamaz; yüzeyden çıkan yön sinyalleri (parite sapması, smirk)
hesaplanabilir.

**Madde 16 (IV Rank) ARTIK ÇALIŞIYOR.** `iv_history.py` günde 1 satır biriktirip
~252 günde rank kurmaya çalışıyordu (pratikte hep `NO_HISTORY`/50.0). Backfill ile
510/536 sembol ~1200 satıra çıktı, `get_iv_rank` `güven=OK` dönüyor. Mevcut kodun
tek satırı değişmedi — okuyucu yeniden yazılmadı, okuduğu dosya beslendi.

**~~Eşikler SINANDI (2026-08-13) ve bant TERS çıktı~~ — TUR-51'DE ÇÜRÜTÜLDÜ.**
`strategy.signal_quality_score` `30 <= iv_rank <= 50` bandına +0.5 verir ("uzun opsiyon
için ideal"). 2026-08-13'te vekil ölçüt `hv_ileri = hv_current.shift(-21)` ile ölçülmüş ve
"bant HER W'de tabandan KÖTÜ (t=−7,35), ilişki MONOTON" sonucu yazılmıştı. **Tur-51 o
sonucun VEKİLİN ARTEFAKTI olduğunu ölçtü** (§28): `hv_current` geriye-bakan 30 günlük
RV'dir (kor. 0,99504 / 603 sembol) ve dolt serisi DÜZENSİZ olduğu için `shift(-30)`
hedeflenen 42-44 takvim günü yerine ortanca **70** gün (std 34,0) ileri gidiyordu — sabit
ufuklu bir ölçüm DEĞİLDİ. AYNI satırlarda tek değişken etiketken: vekil t=−2,29 (monoton)
↔ fiyattan temiz ileri RV t=**+0,61** (monotonluk YOK); iki etiketin korelasyonu 0,7177.
Düzenli ızgarada nihai ölçüm **t=−0,17 = AYIRT EDİLEMEDİ** (n=20.910, 581 PIT sembol).
**Doğru okuma: bant hiçbir yönde bilgi taşımıyor — ne doğrudur ne terstir.** Bonusu
ÇEVİRMEK için kanıt YOKTUR (çevirme önerisi geri çekildi). **KARAR (Tur-51): varsayılan
`IV_REGIME_BONUS=off`** — kanıtsız bileşen skordan çıkarıldı (Tur-31'in "ölçüm yokluğu
ödüllendirilmez" disiplininin devamı). Bu bir kenar iddiası DEĞİL, iddiasızlıktır; ölçülen
şey hâlâ IV−RV farkıdır, **P&L değildir** (ACI DERS #3). `band` eski davranışa döndürür ve
üç kip de testte kilitlidir. Ayrıntı: `docs/EDGE_FINDINGS.md` §28 ·
`docs/TUR51_RAPORU_2026-08-27.md`.

**ÖLÇÜM YOKLUĞU ARTIK ÖDÜLLENDİRİLMİYOR (Tur-31).** Aynı ölçümün tartışmasız kusuru
kapatıldı: `get_iv_rank` üç ölçümsüz dalda da (`NO_HISTORY`/`LOW_CONFIDENCE_n`/`NO_RANGE`)
50.0 **uydurur** ve 50.0 bonus bandının TAM İÇİNDEDİR → IV geçmişi olmayan her sinyal
+0.5'i koşulsuz alıyordu (backfill öncesi bileşen sıfır bilgi taşıdı). `IVRankResult`
artık `measured: bool` taşır (varsayılan False = güvenli yön) ve `execution/main.py`
ölçümsüzü skorlamaya `None` olarak geçer. Bant tartışması bundan BAĞIMSIZDIR.

## ML hattı (meta-labeling) — mimari

- `nihai/signals/meta_features.py` — `FEATURE_NAMES` (TEK kaynak; 2026-07-03'te 19→25:
  gap_pct/range_pct/close_pos/streak/rsi_slope/hour — canlı özellik logu doğmadan genişletildi),
  `extract_features` (leak-free, yalnız karar barına kadarki veri), `neutral_default`/
  `ensure_feature_columns` (eski-şema CSV'leri nötr-doldurarak okur; NaN asla modele sızmaz),
  `TradeRecorder` (backtest giriş/çıkış → etiketli satır). Eski modeller kayıtlı `features`
  listesiyle çalışmayı sürdürür (train/serve hizası).
- `research/train_meta.py` — `LogisticModel` / `GBMModel` / `XGBModel`; `evaluate()` AUC + **kapı
  simülasyonu** + `_retention_sweep` (precision↔return). `--save` yalnız **dayanıklı kenar** varsa yazar.
- `research/purged_cv.py` — purged + embargolu **K-fold** CV (López de Prado Ch.7): tek 60/40 yerine
  K bağımsız OOS penceresi → şanslı-bölme kırılganlığını eler. Fold-başı kapı simülasyonu + ortalama
  OOS-AUC + permütasyon özellik-önemi (model neye bakıyor / sızıntı teşhisi). `train_meta`'nın model
  sınıflarını yeniden kullanır. PIT S&P500'de doğrulandı: OOS-AUC ~0.50, kapı BASE'i geçmiyor → kenar YOK.
- `nihai/signals/meta_gate.py` — üretimde modeli yükler, düşük-olasılıklı sinyalleri eler.
  `META_GATE` env değişkeni **varsayılan KAPALI** (aşağıdaki nedenle).
- **Gölge mod (2026-07):** `META_GATE=shadow` → kapı p_win'i hesaplar + `meta_shadow_gated`/
  reason yazar ama sinyali ASLA bastırmaz. Neden: canlı gate_sim GATE_HURTS diyor (enforce
  açılmamalı) AMA kanıt döngüsü (pwin logu → gate_sim/pwin_calibration) yaşamalı; kapı
  tamamen kapatılırsa bu döngü ölür. `true`=enforce (eski davranış), `false`=off.
- Modeller: `data/meta_model.json` (lojistik) / `data/meta_model.pkl` (gbm/xgb).
- Dataset'ler: `data/meta_dataset_*.csv` (en büyüğü = PIT S&P500, 11.874 işlem).
- **CANLI kanıt döngüsü (2026-07-02):** `prediction_tracker` artık sinyal adaylarının
  P(win)+tarafını da loglar — kapı-engelli WAIT dahil (`strategy.py` `meta_gated`
  metadata'sı → `live_predict_service._signal_pwin` → `action`/`pwin_side` kolonları).
  Bu olmadan META_GATE açıkken log yalnız eşik-üstünü görüyordu (seçilim yanlılığı;
  11.8k çözülmüş tahminde pwin'li satır 0'dı). `prediction_tracker.gate_sim()` =
  offline kapı simülasyonunun canlı karşılığı (tutulan/elenen/taban; örneklem-kapılı,
  iki tarafta ≥30 aday yoksa INSUFFICIENT). Raporda/`--report`/web accuracy panelinde
  `meta_gate` bölümü. "META_GATE açık kalmalı mı?" sorusunun tek dürüst canlı cevabı.
  2026-07-03: gate_sim'e örneklem-kapılı EŞİK TARAMASI (`sweep`: 0.35–0.60; kapıyı
  geçemeyen eşiğe uplift yazılmaz) + rapora `pwin_calibration` (meta P(win) kovası →
  gerçek taraf-kazanma; model 0.6 diyorsa canlıda ~%60 mı kazanıyor?) eklendi.

- **CANLI eğitim-verisi birikimi (2026-07-02B):** PREDICT_TRACK açıkken aday sinyallerin
  (giriş + kapı-engelli) karar-barı ÖZELLİK vektörü de `data/live_features_log.csv`'ye
  yazılır (`prediction_tracker._write_features`; (symbol,bar_ts) anahtarı). Çözülen
  tahminlerle join → `research/build_live_dataset.py` train_meta-uyumlu, yapısı gereği
  survivorship-temiz + leak-free eğitim CSV'si üretir (etiket = horizon-bazlı taraf-getirisi
  işareti; stop/target-yönetimli PnL DEĞİL — script docstring'ine bak). ACI DERS #5'in
  "canlı paper kanıtı" yolunun eğitim-verisi ayağı: aylar içinde PIT-dertsiz dataset birikir.

## AI katmanı (nihai/ai) — v2 "Quant Beyin" (2026-07-19, KARAR-DIŞI/GÖLGE)

Tur-13 çekirdeği (router/memory/reasoner) + v2 modülleri: `quant_math` (finans matematiği,
rejim), `models` (kapasite merdiveni 26→3.777 param; promotion kanıt-kapılı), `episodic`
(benzer-durum hafızası + LESSONS ders kataloğu), `learning` (Beta posterior GLOBAL/LONG/SHORT,
Brier, drift), `knowledge` (107 kavram; Tur-34'te 103→107), `decision` (GÖLGE Bayes; `AI_DECISION_MODE=off|shadow`,
enforce YOK; yön için prior CI tabanı>0.5 şart), `brain` (düşünme döngüsü; `--demo`/`--brief`).
Spec: `docs/AI_V2_BEYIN_SPEC.md` · Rapor: `docs/AI_V2_BEYIN_RAPORU_2026-07-19.md`.
ANAYASA (Tur-38'de KİPLENDİ): bu paket VARSAYILAN olarak sinyal/eşik/META_GATE/execution'a
bağlanmaz; çıktılar rapor/hafıza/gölge-kayıttır. Tek istisna `AI_LIVE_MODE` köprüsüdür
(`execution/ai_advisor.py`) ve o da varsayılan `off`'tur — aşağıdaki Tur-38 bölümüne bak.
`daily_summary --brain` opsiyonel Beyin bölümü ekler (fail-soft).

**Nöral GÖLGE p_win köprüsü (Tur-18):** `nihai/execution/neural_shadow.py` kayıtlı bir
`NeuralCapacity` modelini yükler ve her CANLI aday sinyal için MC-dropout (p_win, std)
üretir; `prediction_tracker` bunu `neural_pwin`/`neural_pwin_side`/`neural_pwin_std`
kolonlarına yazar, `ai/learning.py` de `sources.neural` Brier'ini otomatik doldurur.
YÖN: yalnız log — değer `_neural` altında taşınır ve kayıttan sonra satırdan düşürülür,
web JSON'una/pano tablosuna ASLA girmez (kayıtlı modelin OOS-AUC'si 0.500; işlem
tablosunda ikinci bir "P(win)" sütunu sinyal sanılır). Model seçimi üç kilitli: sentetik
müfredat modeli canlıya alınmaz, `feature_names` FEATURE_NAMES ile birebir eşleşmeli,
artefakt diskte olmalı. Doygunluk guard'ı: `std==0` + uç p (dağılım-dışı girdi ağı
doyurmuş) → kaydedilmez (sahte-kesinlik Brier'i kirletmesin). `NEURAL_SHADOW=off` kapatır.
Kolonlar yalnız kayıtçı süreç YENİDEN BAŞLADIKTAN sonra dolmaya başlar (şema süreç
belleğinde).

**ÖLÇEK HİZALAMASI (Tur-20, 2026-07-30) — köprü artık CANLI.** Denetimin kritik B-2
bulgusu kapandı: tüm nöral modeller 1d/1h barlarda eğitilmişti, canlı tahminlerin %100'ü
1m → `max|z|=30.16`, OOD guard 600/600 susturuyordu (guard DOĞRU davranıyordu; eksik olan
dağılımı eşleşen modeldi). `research/build_live_dataset` ile canlı loglardan
survivorship-temiz + leak-free **1m dataset** üretildi (5.996 satır/30 sembol/taban %46,8)
ve `deep_mlp` onunla eğitilip kaydedildi (kart `3473ecf3…`). Sonuç: **598/600 üretiliyor**,
`max|z|` kapsaması %99,67, p_win ort. 0.4582 ↔ eğitim tabanı 0.4681 (**sapma 0.0099**),
verdikt "KÖPRÜ CANLI ve kalibrasyon tutarlı". **Kenar DEĞİL:** OOS-AUC 0.525±0.013,
**DSR 0.0006**. Kazanç kalibrasyon hizasında; artık `sources.neural` Brier'i ÖLÇÜLEBİLİR.
Model eskidikçe (canlı veri büyüdükçe) yeniden eğitilmeli. Rapor: `docs/TUR20_RAPORU_2026-07-30.md`.

**Model kartı EMEKLİLİĞİ (Tur-20):** `registry.deprecate(hash, reason)` bir kartı SİLMEDEN
seçim dışına alır; `list_models()`/`latest()` varsayılan olarak emekli kartları eler
(`include_deprecated=True` tam döküm). Bozuk-DSR'li `ac505353…` kartı (ACI DERS #8
artefaktı) emekliye ayrıldı. `register()` içerik-özdeş yeniden-kayıtta bayrağı KORUR.

**AI v4 — KAPASİTE ve KALİBRASYON (Tur-22, 2026-08-01).** `nihai/ai/neural.py` üç katmanlı:
`deep_mlp` (~187k) · `quant_transformer` (≈5,39M) · **`quant_transformer_xl` (≈30,4M)** —
12 blok, d=384, SwiGLU FFN, RMSNorm, LayerScale, doğrusal artan stokastik derinlik, PLR
periyodik sayısal gömme (Gorishniy 2022). Dizi düzeni ([CLS]+3×[SCALE]+özellikler) ve
`nn.MultiheadAttention` BİLEREK korundu — `explain.attention_rollout` kırılmasın diye.
Ayrıca **`NeuralEnsemble`** (derin topluluk, K üye × farklı seed): kapasite K katı,
belirsizlik toplam varyans yasasıyla aleatorik (üye-içi MC-dropout) + epistemik (üye-arası)
olarak ayrışır. Üretim özellik sayısında (f=96) XL 31,76M, 5 üyeli XL topluluğu 158,8M.

Eğitim reçetesi ekleri **HEPSİ OPT-IN, varsayılan v3 ile BİT-DENK** (kayıtlı canlı artefakt
ve eski ölçümler geçerliliğini korur): `scheduler="cosine"` (ısınmalı), `ema_decay`,
`class_weight="balanced"` (YALNIZ eğitim diliminin oranından), `label_smoothing`,
`early_stop_metric="val_auc"`, `calibrate=True` (fit sonrası val diliminde sıcaklık
ölçekleme — sıralamayı DEĞİŞTİRMEZ, yalnız olasılığı dürüstleştirir), `device="auto"`.
`research/train_neural.py` bunları bayrak olarak sunar (`--ensemble K` dahil) ve raporda
AUC'nin yanına **Brier + ECE** yazar.

**ÖLÇÜLDÜ (canlı 1m seti, 5.996 satır, purged 5-fold, AYNI fold'lar):** v3 taban
OOS-AUC 0,525±0,013 / ECE 0,0377 / DSR 0,0006 → v4 reçetesi **0,537±0,012 / ECE 0,0312 /
DSR 0,1805** (5/5 fold'da AUC arttı, ECE %17 düştü). **KENAR DEĞİL:** AUC hâlâ yazı-turaya
yakın, DSR 0,18 hiçbir kanıt kapısını geçmez. Kazanılan şey ayırt etme değil, KALİBRASYON
ve reçete disiplinidir (ACI DERS #3: metrik iyileşmesi kenar demek değildir).

Ek: `NeuralCapacity.ood_zmax()` + fit'te ölçülen `train_zmax_q99` artefaktla taşınır →
`NEURAL_SHADOW_MAX_Z=auto` eşiği modelin KENDİ eğitim dağılımından türetir (sabit 6,0
yerine; q99×1,5, en az 4,0). `neural_shadow` artık `load_any` ile topluluk artefaktlarını
da yükler. `learning.py`'ye **ECE** eklendi (`sources.*.ece` + rapor "kaynak karnesi"
satırı): Brier kalibrasyon+ayırt etme karışımıdır, ECE yalnız kalibrasyonu ölçer — AUC
~0,50 iken tek dürüst ilerleme ekseni budur.

**Prior'ın HAFIZASI — `AI_PRIOR_DECAY` (Tur-25, 2026-08-10, varsayılan KAPALI).**
`learning.update_from_predictions` sayımları zaman-ağırlıksızdı ve `page_hinkley`'nin
bulduğu drift'i hiçbir şey TÜKETMİYORDU. Artık ek anahtar `strategies_decayed`: üstel
unutma (`DECAY_HALF_LIFE_DAYS=30`, env `AI_PRIOR_HALF_LIFE_DAYS`) + drift varsa etkin
sayımlar `DRIFT_SHRINK=2.0`'a bölünür (tepki HER ZAMAN daha AZ güven yönündedir). Mevcut
`strategies` DOKUNULMADI; `decision._prior_from_state` yeni anahtarı yalnız `AI_PRIOR_DECAY`
açıkken okur → varsayılan **bit-denk**. Referans zaman = kuyruğun son damgası ("bugün"
değil) → determinizm; tüm damgalar naive-UTC'ye kanonikleşir (BULGU-16 tekrarlanmasın).
Yan-koşullu prior'ın n≥200 kapısı HAM satır sayısına bakar.
**ÖLÇÜLDÜ ve varsayılan AÇILMADI:** aynı state ile A/B backtest'te λ=0,05'te fazlalık açığı
seçim bölmesinde −2,4236→−2,3973, **doğrulama bölmesinde −4,2328→−4,3881 (KÖTÜLEŞTİ)** —
fark gürültü mertebesinde, hipotez doğrulanmadı. Neden: canlı log yalnız ~5 hafta
(2026-06-24→07-31); 30 günlük yarılanma prior'ı kımıldatamıyor (SHORT post_mean
0,5508→0,5498, lo95 0,5290→0,5165 hâlâ >0,5). **Zararın nedeni prior'ın ESKİMESİ değil,
hiç tahmin gücü OLMAMASI.** Mekanizma birden fazla rejimi kapsayan log biriktiğinde anlam
kazanır → doğrudan "servisleri aç" maddesine bağlı.

**Kalibrasyon döngüsü `nihai/ai/calibration_loop.py` (Tur-25, KARAR-DIŞI).**
`learning.fit_calibrator`/`apply_calibration` yazılmış+test edilmişti ama üretimde HİÇ
çağıranı yoktu. Yeni modül kaynak başına (`pwin`/`neural`) KRONOLOJİK bölmeyle yalnız-OOS
kalibrasyon ölçer; artefakt `data/ai_calibrators.json`'a **yalnız Brier VE ECE'nin İKİSİ de
OOS'ta iyileşirse** yazılır (gürültü freni; kanıt kapısı DEĞİL). Yetersiz örneklemde
dürüstçe `YETERSIZ` der — `neural` bugün tam olarak orada (n_oos=3). `calibrate()` hiçbir
canlı karar yolundan çağrılmaz (anayasa). Sonuçların okunuşu için ACI DERS #9.

**YÜRÜYEN kalibrasyon `nihai/ai/adaptive_calibration.py` (Tur-27, KARAR-DIŞI).** Sistemdeki
her kalibratör TEK, STATİK bir fit'ti: `calibration_loop` bir kez %60/%40 böler;
`data/forecast_calibration.json` ise `live_predict_service.__init__`'te BİR KEZ okunur
(`self._calib`) — 7/24 serviste eğri tazelense bile süreç görmez. Yeni modül eşlemeyi zaman
içinde yeniden fit eder (prequential / "test-then-train"): ilk `warmup` yalnız fit, sonra her
`refit_every` örnekte kalibratör YALNIZ GEÇMİŞTEN öğrenilir ve BİR SONRAKİ bloğa uygulanır
(kesin nedensel; `window=N` kayan pencere = rejim adaptasyonu knob'u). ÜÇ yarışmacı TAM AYNI
örneklerde puanlanır: **ham · adaptif · TABAN** (koşan taban oranı = sabit tahminci).
Kapı: ham'a karşı Brier VE ECE, **tabana karşı EŞLEŞTİRİLMİŞ Brier farkı t<−1,645** (Tur-24
Welch eşiğiyle aynı). Tabana karşı ECE ARANMAZ — sabit tahminci ECE'de yapısı gereği ~0'dır,
onu ECE'de yenmesini istemek kategori hatası olurdu.
**ÖLÇÜLDÜ (canlı `pwin`, 6.002 üçlü, 2026-07-06→07-31, n_eval=5.002, 51 yeniden-fit):**
Brier ham 0,2680 · TABAN 0,2493 · adaptif 0,2493; eşleştirilmiş fark **+0,00000 (t=+0,01)**.
8 yapılandırmada (platt/izotonik × genişleyen/2000/1000/500) t **hiç** negatif-anlamlı olmadı;
kısa pencerede izotonik anlamlı ölçüde KÖTÜLEŞTİ (t=+4,65). p_std 0,0721→0,0161. Yani
**kalibrasyon adaptasyonu kazanç vermiyor; `pwin` taban oranın ötesinde bilgi taşımıyor**
(ACI DERS #2/#4/#9'un yürüyen-OOS'ta bağımsız doğrulaması). Kapı doğru davrandı:
`durum="KAPI_GECILEMEDI"`, kalibratör kaydedilmedi. `neural` YETERSİZ (n=6 — servisler kapalı).

**ÇELİŞKİ KAPANDI (Tur-28, 2026-08-12).** `calibration_loop`'un kapısı da artık sabit
taban-oran tahmincisini üçüncü yarışmacı olarak içerir; test t-testi TEK kaynaktan gelir
(`learning.paired_brier_t`, iki modül de onu çağırır). Gerçek artefakt yeniden verdiktlendi:
`pwin` **KAYITLI → KAPI_GECILEMEDI**. Ölçüm (n_oos=231, taban oran 0,4667): platt Brier
0,2951→**0,2520** ve ECE 0,1870→0,0603 ile ham'ı GERÇEKTEN düzeltiyor, ama sabit taban
**0,2530** — eşleştirilmiş fark yalnız −0,00102 (**t=−1,42**, eşik −1,645); izotonik tabandan
daha KÖTÜ (t=+2,27). İki bağımsız kapı (tek-bölme + yürüyen) artık AYNI şeyi söylüyor.
Eski artefakt `data/ai_calibrators.json.tur25-yedek` olarak duruyor.

**AI v5 — "DERİNLEŞME" (Tur-32, 2026-08-14).** Rapor: `docs/AI_V5_TUR32_RAPORU_2026-08-14.md`.
Üç YENİ modül + iki mevcut modülde genişleme; hepsi KARAR-DIŞI, hiçbir kenar iddiası YOK.

- **`nihai/ai/options_math.py`** — opsiyon & volatilite YÜZEYİ. `knowledge.py`'nin kendi
  kataloğunda *"quant_math'te doğrudan bir 'IV' fonksiyonu yoktur"* ve *"doğrudan bir
  smile-fit fonksiyonu YOKTUR"* diye yazılı olan iki boşluğu kapatır. `implied_vol`
  (Brent; arbitraj bandı dışında **None**, uydurma σ yok), `bs_greeks_extended` (vanna/
  volga/charm/veta/speed/zomma/color/dual_*, hepsi sonlu-farkla doğrulandı; zamana bağlı
  greekler theta ile AYNI eksende ∂/∂t), `put_call_parity` (→ **örtük forward**, gülümsemenin
  doğru çapası spot DEĞİL budur), raw **SVI** (`svi_fit` + `svi_durrleman_g` +
  `svi_no_arbitrage`: fit tek başına yetmez, butterfly arbitrajı g(k)≥0 ile taranır),
  **SABR**/Hagan (ATM dalı ayrı — z/x(z)→1 belirsizliği), `smile_metrics` (25Δ RR/BF;
  strike veri aralığı dışına düşerse None, sessiz ekstrapolasyon YOK),
  `risk_neutral_density` (Breeden-Litzenberger; **negatif yoğunluk bastırılmaz**,
  `n_negative` raporlanır), `vol_term_structure` (calendar arbitraj).
- **`nihai/ai/microstructure.py`** — gerçekleşen oynaklık ÖNGÖRÜSÜ + mikroyapı.
  **HAR-RV/HAR-CJ** (projede oynaklık hep ölçülüyordu, hiç tahmin edilmiyordu),
  `jump_split`, **TSRV** (gürültüden arındırılmış RV) + `volatility_signature`,
  ve ACI DERS #7'nin teşhis takımı: `roll_spread` · `corwin_schultz_spread` ·
  `kyle_lambda` · `amihud_illiquidity` · `vpin`. Üç kez "kenar ilk bardaki makas
  sıçramasıydı" dendi ama etkin makası ÖLÇEN fonksiyon yoktu; artık var.
  `roll_spread` pozitif otokovaryansta **None + bayrak** döner (ölçüm yokluğunu "makas yok"
  diye raporlamaz — Tur-31 IV Rank dersi).
- **`nihai/ai/online.py`** — ÇEVRİMİÇİ uyarlama. `adwin` (pencereyi VERİDEN seçen sürüklenme
  dedektörü; `page_hinkley`'in aksine eşik/pencere elde ayarlanmaz), `hedge_combine`
  (üstel ağırlıklı uzman birleştirme + **fixed-share** = rejim değişiminde ağırlığı geri
  dağıtma; ölçüldü: rol değiştiren iki uzmanda saf Hedge 0,9165 ↔ fixed-share **0,2347**),
  `FTRLProximal` (tek geçişli, L1 gerçek sıfır üreten akış-içi lojistik).
  **`include_base=True` koşan taban oranını OTOMATİK yarışmacı yapar → ACI DERS #9 makinede.**
  Ölçüldü: öğrenilebilir veride FTRL uplift **+0,206**, saf gürültüde **−0,014** (bilgi
  icat etmiyor).
- **`learning.py`** — `log_loss_score` · `brier_skill_score` (BSS) · `murphy_decomposition`
  (güvenilirlik/**ÇÖZÜNÜRLÜK**/belirsizlik). Kaynak karnesi artık `brier · ece · bss ·
  resolution · n` taşır ve raporda "BSS ≤ 0 = sabit taban-orandan iyi DEĞİL · RES ≈ 0 =
  taban ötesi bilgi YOK" satırı basılır. Tur-25/27/28'de ÜÇ KEZ elle yapılan kontrolün
  kalıcı hâli: "Brier düştü" satırı artık tek başına okunamaz.
- **`neural.py` → `quant_transformer_moe` (≈93M parametre).** Merdivenin yeni üst basamağı
  SEYREK uzman karışımıdır (blok başına 6 uzman SwiGLU, token başına **top-2**). Dizi
  düzeni ve `nn.MultiheadAttention` XL ile AYNI (attention_rollout korunur); tek değişken
  FFN'dir. **ÖLÇÜLDÜ** (f=96, 256×96 ileri geçiş, CPU): toplam **31,76M → 93,23M (2,94×)**,
  AKTİF parametre 31,76M → 36,49M (1,15×), süre 4.709 → 5.883 ms (**1,25×**). Yani ≈3 kat
  kapasite ≈1,25 kat maliyetle. Yoğun genişletme seçilmedi çünkü Tur-22 §7-1'de ölçüldü:
  XL süpürmesi bu donanımda 21 gün. Yük dengeleme (Switch aux loss, ağırlık 0,01)
  ZORUNLUDUR — onsuz yönlendirici tek uzmana çöker ve kapasite kâğıt üstünde kalır; terim
  yalnız çıktıda `moe_balance` VARSA eklenir → diğer basamaklar v4 ile bit-denk.
  Yönlendirme argmax tabanlı (gürültülü kapı yok) → determinizm korunur.
  `recommend_tier` eşiği n ≥ **1.200.000** (TOPLAM parametreye göre, aktife göre değil).
  **Kapasite artışı bir KENAR iddiası DEĞİLDİR (ACI DERS #5); bu basamağın canlı modeli YOK.**
- **Formül köprüsü genişledi:** `knowledge.formul_ref` adları artık üç modülde aranıyor
  (`quant_math` · `options_math` · `microstructure`); tek doğruluk kaynağı
  `selfcheck._FORMULA_MODULES`. `selfcheck` sayısal sınavı 12 → **17 soru** (IV gidiş-dönüş,
  SABR ATM β=1 limiti, SVI minimumu, RV, Roll kapalı formu).

**AI v6 — "BELİRSİZLİK ve KAPASİTE" (Tur-34, 2026-08-15).** Rapor:
`docs/AI_V6_TUR34_RAPORU_2026-08-15.md`. Bir YENİ modül + dört genişleme + ölçülmüş
optimizasyon; hepsi KARAR-DIŞI, kenar iddiası YOK. Seçim ilkesi Tur-32 ile aynı: sistemin
KENDİ belgelerinde "burada YOKTUR" yazan satırlar kapatıldı.

- **`nihai/ai/regime_hmm.py`** — rejim GEÇİŞ OLASILIKLARI (Gauss HMM, Baum-Welch).
  `knowledge.py` *"rejim etiketi KATEGORİKTİR — geçiş OLASILIĞI bilgisi yoktur"* diyordu.
  **LOOK-AHEAD SINIRI modülün en kritik sözleşmesidir:** `hmm_filter` NEDENSELDİR
  (`causal=True`) ve canlı yolda kullanılabilecek TEK çıktıdır; `hmm_smooth`/`hmm_viterbi`
  geleceği görür ve bunu `causal=False` ile MAKİNE-OKUNUR taşır. `gaussian_hmm_fit` tüm
  örneklemden fit eder = İÇ-ÖRNEKLEM; canlıda eğitim penceresinde fit + ileri uygulama
  gerekir. ÖLÇÜLDÜ (T=2000 sentetik): kalıcılık 0,9799↔0,98 · 0,9372↔0,95, durağan dağılım
  0,757↔0,714, nedensel filtre isabeti %94,5, Viterbi %97,05 (geleceği gören daha isabetli
  olmalı ✓), BIC K=2'yi decisive seçiyor. σ TABANI ZORUNLUDUR (bir durum tek gözleme
  çökerse olabilirlik sonsuza gider). İlklendirme quantile tabanlı, `seed` argümanı YOK.
- **`quant_math`** — ASİMETRİK (arşimedyen) kopulalar: `copula_fit_clayton` (λ_L>0, λ_U=0),
  `copula_fit_gumbel` (λ_U>0, λ_L=0), `copula_tail_dependence` (dört aile kapalı form λ),
  `copula_select` (AIC). Gauss "kuyruk yok", t "simetrik kuyruk" varsayıyordu; asimetri
  artık VERİYE sorulur. ÖLÇÜLDÜ (θ=2 üretilmiş 3000 nokta): θ̂=2,0046, λ̂_L=0,7077 ↔ teorik
  0,70711, seçim clayton ΔAIC=600,8. **τ ≤ 0'da arşimedyen aile UYDURULMAZ** → tabloya
  `None` girer ("uymadı" ≠ "denenmedi"); ΔAIC<2 → `decisive=False`.
- **`microstructure`** — **PIN** (EKOP yapısal MLE'si, Lin-Ke sayısal çarpanlamasıyla —
  naif form B,S birkaç yüzü geçince taşar). Makasın BÜYÜKLÜĞÜNÜ ölçen araçların yanına
  NEDENİNİ (ters seçim payı) ölçen araç geldi. ÖLÇÜLDÜ: α̂ 0,393↔0,4 · μ̂ 58,6↔60 ·
  PIN 0,2123↔0,2202; ölçek değişmezliği doğrulandı. `pin_from_bars` OHLCV VEKİLİDİR ve
  **`proxy=True` taşır** — Poisson varsayımı işlem SAYISI üzerinedir, hacim değil.
- **`options_math`** — **SSVI** (Gatheral-Jacquier arbitrajsız yüzey) + `ssvi_diagnose`:
  **Tur-32 açık kalem #5 KAPANDI.** SSVI'de arbitrajsızlık bir SONUÇ değil KISITTIR
  (γ≤½, η(1+|ρ|)≤2, θ_t artan). Verdikt dört senaryoda da doğru ayrıldı: temiz yüzey
  `ARBITRAJ_YOK`, gürültülü `UYUM_ARTEFAKTI` (oran 1,16), dik kanat `VERIDE_ARBITRAJ`
  (oran 2,7e6), takvim ihlali `BELIRSIZ`. Verdikt KAYNAĞI söylemez (asenkron kotasyon /
  Amerikan primi / gerçek arbitraj ayrılmaz) ama hepsini "fit hatası"ndan AYIRIR.
- **`neural` → `quant_transformer_moe_xl` (≈284M parametre).** Merdivenin yeni üst basamağı:
  16 **İNCE-TANELİ** uzman (top-2) + her token'da çalışan **PAYLAŞIMLI** uzman (DeepSeekMoE,
  Dai ve ark. 2024) + **ST-MoE router z-kaybı**. İnce tanenin anlamı parametre değil ifade
  gücüdür: seçilebilen kombinasyon C(6,2)=15 → **C(16,2)=120**. ÖLÇÜLDÜ (f=96):
  93.228.307 → **283.639.571 (3,04×)**. `recommend_tier` eşiği n ≥ **3.600.000** — bugünkü
  canlı logdan ÜÇ MERTEBE uzaktır ve bu bilinçlidir. **Kapasite artışı bir KENAR iddiası
  DEĞİLDİR (ACI DERS #5); bu basamağın canlı modeli YOKTUR.**
- **`neural.moe_routing_stats` — Tur-32 açık kalem #3 KAPANDI.** Yük dengeleme kaybının
  DEĞERİNİ raporlamak işe yaradığının kanıtı değildir. İki ayrı soru ayrı ölçülür: ÇÖKME
  (kullanım payı/entropi/`collapsed`) ve UZMANLAŞMA ([CLS] × grup ki-kare + Cramér V).
  Ölçüm koşulu taşınır: beklenen sayımın >%20'si 5'in altındaysa `reliable=False` ve **p
  DÖNDÜRÜLMEZ**; [CLS] tek uzmana gidiyorsa `measured=False` + `reason="CLS_TEK_UZMAN"`
  ("uzmanlaşma yok" DEĞİL, "karşılaştırılacak ikinci hücre yok"). MoE olmayan tier → None.

**AI v7 — "ETİKETSİZ VERİ, DAĞILIM ve ÖLÇEK" (Tur-35, 2026-08-16).** Rapor:
`docs/AI_V7_TUR35_RAPORU_2026-08-16.md`. İki YENİ yetenek + yeni kapasite basamağı + iki tur
ertelenmiş bir açık kalemin kapanışı + dört ölçülmüş optimizasyon. Hepsi KARAR-DIŞI, kenar
iddiası YOK.

- **`neural.NeuralCapacity.pretrain` — ÖZ-DENETİMLİ maskeli özellik ön-eğitimi.** ACI DERS #5
  *"yeni bir sınıflandırıcı eklemek bunu çözmez"* diyor; bu yöntem yeni bir sınıflandırıcı
  DEĞİLDİR — **etiket gerektirmez**. Her satırda özelliklerin %25'inin token'ı öğrenilebilir
  `[MASK]` gömmesiyle DEĞİŞTİRİLİR (sıfırlanmaz — sıfır geçerli bir normalize değerdir) ve ağ
  gerçek değeri geri kurar; kayıp YALNIZ maskeli konumlardan alınır. **`is_fitted`'i AÇMAZ**
  (etiket görmemiş gövdeden `p_win` okumak sessiz bir yalan olurdu → `predict_proba`
  RuntimeError). Rekonstrüksiyon başlığı + `[MASK]` gömmesi ön-eğitim sonunda ATILIR → artefakt
  şeması sıfırdan eğitilmiş modelle AYNI. Val bölmesi yine KRONOLOJİK (look-ahead yasağı
  etiketsiz veride de geçerlidir). `deep_mlp` token'sızdır → ValueError (sessiz atlama YOK).
  `fit(..., warm_start=True)` gövdeyi devralır; varsayılan False → v6 ile BİT-DENK.
  CLI: `python -m research.train_neural --pretrain [--pretrain-dataset X.csv]`.
  **ÖLÇÜLDÜ ve HİPOTEZ DOĞRULANMADI (canlı 1m seti, purged 5-fold, AYNI fold'lar):** sıfırdan
  OOS-AUC 0,4896±0,0167 / ECE 0,0178 → ön-eğitimli **0,4881±0,0135 / ECE 0,0147**; B, AUC'de
  3/5 fold'da önde = yazı-tura. **Neden:** rekonstrüksiyon kaybı canlı sette neredeyse hiç
  düşmedi (2 fold'da TAM SIFIR; en iyi %2,9) oysa AYNI kod sentetik müfredatta 2 epoch'ta
  **%19** düşürdü — yani mekanizma çalışıyor, canlı 25 özellikte öğrenilecek çapraz-özellik
  yapısı KALMAMIŞ (vektör zaten elle dikleştirilmiş). Asıl hedef `live_features_log.csv`'nin
  ETİKETSİZ çoğunluğudur; bu turda ek veri KULLANILMADI (açık kalem).
- **`neural` → `quant_transformer_ultra` (≈772M parametre).** 40 İNCE-TANELİ uzman (top-2,
  uzman genişliği 1024→768) + paylaşımlı uzman + z-kaybı; d=640, 12 blok. Seçilebilen
  kombinasyon C(16,2)=120 → **C(40,2)=780**. ÖLÇÜLDÜ (f=96): 283.639.571 → **771.874.072
  (2,72×)**. `recommend_tier` eşiği n ≥ **10.000.000** — canlı logdan DÖRT mertebe uzak.
  **Kapasite artışı bir KENAR iddiası DEĞİLDİR (ACI DERS #5); canlı modeli YOKTUR.**
- **DAĞILIMSAL oynaklık `vol_q` (YALNIZ ultra).** Oynaklık bugüne kadar hep TEK SAYIydı
  (`vol`, MSE = koşullu ortalama). Yanına 5 quantile'lık demet geldi (pinball kaybı, Koenker &
  Bassett 1978). Çapraz geçiş kayba değil MİMARİYE yasaklandı (kümülatif softplus).
  `_compute_loss` terimi YALNIZ `vol_q` VE `aux['vol']` birlikte varken ekler → diğer beş
  basamak bit-denk. `predict_vol_quantiles` başlığı olmayan basamakta dürüstçe **None**.
- **`learning.source_fusion` — Tur-32 açık kalem #4 KAPANDI (iki tur ertelenmişti).**
  `decision.py` iki kaynağı Brier'den türeyen SABİT ağırlıkla harmanlıyor; `hedge_combine`
  ağırlıkları her adımda o adıma KADARKİ kayıptan türetir (prequential) ve fixed-share rejim
  değişiminde ölü uzmanı diriltir. HİZALAMA SÖZLEŞMESİ kritik: üç uzman TEK olayı yanıtlar
  (*"canlı sinyalin YÖNÜ kazandı mı"*); kaynak karşı tarafı diyorsa olasılık **1−p** ile
  çevrilir. Kapı iki katmanlı: `vs_base > 0` VE `paired_brier_t < −1,645` (Tur-24/27/28 ile
  aynı tek kaynak). **ÖLÇÜLDÜ (canlı log, 1.286 hizalı üçlü):** log-kayıp birleşim 0,694924 ↔
  pwin 0,741877 · neural 0,699335 · **TABAN 0,694189**; t=+0,329 → **TABANI_GECILEMEDI**.
  Mekanizma doğru çalıştı (Hedge en kötü uzmanı 1/3 → **0,142**'ye indirdi) ama sonuç ACI DERS
  #2/#4/#9'un ÜÇÜNCÜ bağımsız doğrulaması: kaynaklar taban oranın ötesinde bilgi taşımıyor.
- **`_SeqEncoderMixin` — dört `forward` gövdesi tekilleştirildi.** `state_dict` anahtarları
  DEĞİŞMEDİ; refactor öncesi/sonrası **62 tensörde fark 0**.
- **`knowledge` 107 → 110 kavram** (`maskeli_on_egitim` · `quantile_regresyon` ·
  `cevrimici_uzman_birlestirme`); formül köprüsü `online` + `learning` ile genişledi.
  `neural` köprüye BİLEREK alınmadı — her kavram sorgusunda torch importunu tetiklerdi
  (Tur-26 disiplini), o kavramlar `formul_ref=[]` taşır ve modülü METİNLE söyler.
  `test_ders_ref_format_aci_n` deseni 1-7'den 1-9'a hizalandı (ders #8/#9 aylar önce
  eklenmişti, test bayattı — kural gevşetilmedi, belgeyle yeniden hizalandı).
- **Optimizasyon (§7):** `_auc_np` **7,9–12,1×** (bit-denk, 30 senaryo) · `hedge_combine`
  **2,8×** (fark 0) · `atomic_io.tail_lines` **9,7–86×** (çıktı-denk) → üretim yolu
  `update_from_predictions` 3151 → **544 ms (5,8×)** · `theoretical_param_count` `meta`
  aygıtında (merdiven 7,58 s / 4,5 GB → 3,02 s / ~0 GB) · gösterge hattında
  `pd.concat(...).max(axis=1)` → `np.fmax` (`williams_ad` **13,5×**, panel 85,4 → **70,0 ms**,
  300 df'de fark **0,000e+00**).

**AI v8 — "BELİRSİZLİK, OPT-IN BAŞLIK ve ÖLÇEK" (Tur-36, 2026-08-17).** Rapor:
`docs/AI_V8_TUR36_RAPORU_2026-08-17.md`. Yeni kapasite basamağı + bir yeni yetenek +
Tur-35'in dört açık kaleminden ÜÇÜNÜN kapanışı + iki ölçülmüş optimizasyon. Hepsi
KARAR-DIŞI, kenar iddiası YOK. **2600 test yeşil** (+30), ruff/mypy temiz.

- **`neural` → `quant_transformer_mega` (2.014.146.906 parametre).** 80 İNCE-TANELİ uzman
  (top-2, uzman genişliği 768→640) + paylaşımlı uzman; d=768, 16 blok. Kombinasyon
  C(40,2)=780 → **C(80,2)=3160**. ÖLÇÜLDÜ (f=96): toplam 2,01 milyar, AKTİF **171.340.122**
  → seyreklik **11,76×** (ULTRA 7,84×): toplam 2,61× artarken hesap yalnız 1,74× arttı.
  Sınıf ULTRA'nın ALT SINIFIDIR (mimari identik, yalnız ölçek sabitleri farklı) — 70
  satırlık kopya "basamaklar arası fark tek değişken" kuralını görünmez kılardı.
  `recommend_tier` eşiği n ≥ **26.000.000**. **Canlı modeli YOK ve planlanmıyor**
  (2,01 milyar float32 = ≈8 GB ağırlık + ≈24 GB AdamW durumu; bu donanımda eğitim
  İMKÂNSIZ — ACI DERS #5).
- **OPT-IN ek başlıklar `extra_heads` — Tur-35 açık kalem #2 KAPANDI.** `vol_q` artık
  ULTRA'ya GÖMÜLÜ değil; her basamakta açılabilir. Varsayılan boş küme HİÇBİR modül
  kurmaz → `torch.manual_seed` akışından tek rastgele sayı bile tüketilmez → ağırlıklar,
  parametre sayısı ve `state_dict` anahtarları v7 ile BİREBİR aynıdır (merdivenin alt altı
  basamağının sayıları Tur-35 raporuyla birebir eşleşti = ölçülmüş kanıt). Kurulum sırası
  KANONİKTİR (çağıranın verdiği sıra ağırlıkları DEĞİŞTİRMEZ). `meta()["extra_heads"]`
  bir ÖLÇÜMDÜR (kurulu modüllerden okunur); alan taşımayan eski JSON'lar basamak
  varsayılanına düşer. Canlı `deep_mlp` artefaktı yeni kodla yüklendi: `param_count`
  **186.702** (aynı), tahmin çalışıyor.
- **KANIT-temelli yön `dir_evid` (Dirichlet/EDL, Sensoy ve ark. 2018) + `predict_evidential`.**
  α = softplus(raw)+1, belirsizlik u = K/Σα — TEK ileri geçişte. Kayıp: Bayes-riski CE +
  KL cezası (katsayı ilk 10 epokta 0→1 TAVLANIR; kanıt oluşmadan cezalandırılırsa model hiç
  kanıt üretmemeyi öğrenir). `predict_proba` yine `direction` başlığını okur — kanıt ikinci
  bir GÖRÜŞ değil ikinci bir ÖLÇÜdür. **ÖLÇÜLDÜ (canlı 1m, kronolojik bölme, 5 seed):**
  düşük-u çeyreğinin isabeti yüksek-u çeyreğinden ort. **+0,098 yüksek, 5/5 seed**;
  MC-dropout std'si aynı testte **0/5** (ort. −0,069, yani TERS sıralıyor). Maliyet
  n=1499'da **3,0 ms ↔ 439 ms (144×)**. `deep_mlp`'de yalnız **130** parametre ekler.
- **`learning.source_fusion` K UZMAN — Tur-35 açık kalem #4 KAPANDI.** `extra={ad: [p,…]}`.
  **ÖLÇÜLDÜ (canlı log, n=1307):** log-kayıp pwin 0,742858 · neural 0,700429 ·
  **prob_cal 0,693865** · confidence 1,282181 · TABAN 0,694327; birleşim 0,695116,
  t=+0,192 → **TABANI_GECILEMEDI** (ACI DERS #2/#4/#9'un DÖRDÜNCÜ doğrulaması). İki yan
  bulgu: `prob_cal` tabanı geçen ilk TEK uzman (fark gürültü mertebesinde, kenar DEĞİL) ve
  **`confidence` bir OLASILIK DEĞİLDİR** — log-kaybı tabanın 1,85 katı; Hedge ağırlığını
  0,03'e indirerek bunu kendiliğinden söyledi.
- **Etiketsiz ön-eğitim — Tur-35 açık kalem #3 KAPANDI, HİPOTEZ YANLIŞ.**
  `research/build_live_dataset.build_unlabeled(before=…)` (+`--unlabeled`) eklendi.
  ÖLÇÜLDÜ: 7.708 özellik satırının 7.317'si zaten çözülmüş; etiketsiz **391 (+%5,3)**,
  NEDENSEL kullanılabilir olan yalnız **69 (+%1,5)**. A/B: sıfırdan 0,5283 ↔ ön-eğitimli
  0,4728 ↔ +69 satırla 0,4649. **Yapısal teşhis:** `prediction_tracker` özellik vektörünü
  YALNIZ aday sinyal barlarında yazıyor ve adaylar er geç çözülüyor → "etiketsiz çoğunluk"
  bu sistemde YOK. Sorun yöntem/ayar değil **LOGLAMA POLİTİKASI**.
- **`knowledge` 110 → 111 kavram** (`kanit_temelli_belirsizlik`); `neural` köprüye yine
  BİLEREK alınmadı (torch importunu tetiklerdi — Tur-26 disiplini), kavram `formul_ref=[]`
  taşır ve modülü METİNLE söyler.
- **Optimizasyon (§7):** `update_from_predictions` satır okuma — 664.524 kapanış çağrısı
  (profilin %28'i) önceden çözülmüş indekse indi: satır okuma **1,29×**, uçtan uca
  **1,10×**, `state` sözlüğü AYNI, 62.465 satırda (dejenere satırlar dahil) fark 0 ·
  skaler normal dağılım scipy → **stdlib `NormalDist`**: `bs_price` **53×** (85,5→1,6 µs),
  `implied_vol` **29×** (864,1→29,4 µs) · `theoretical_param_count` kısa yolu: MEGA satırı
  **1622,6 → 90,2 ms (18×)**, yedi basamakta sayı TAM AYNI (bu turun KENDİ getirdiği
  regresyonun onarımı — aşağıdaki derse bak).

**AI ÜRETİM KÖPRÜLERİ — "BAĞLI DEĞİL" ve "ÇAĞRILMIYOR" KAPANDI (Tur-37, 2026-08-20).**
Denetim iki ayrı boşluk ölçtü: (a) `nihai/ai` üretimde YALNIZ üç yerden import ediliyordu
(`execution/neural_shadow`, `notifier/daily_summary`, `web/api_extra`) — `signals/*`,
`risk/*`, `execution/*`, `data/*`, `telegram_bot`, `terminal_dashboard`, `web/tearsheet`
bağsızdı; (b) beş modül yazılmış, test edilmiş ve HİÇ ÇAĞRILMAMIŞTI (`microstructure`,
`regime_hmm`, `options_math`, `adaptive_calibration`, `calibration_loop`).
**(a)'nın bir kısmı TESADÜF DEĞİLDİR ve öyle KALDI** — kanıt kapısını geçmemiş bir MODEL
işlem hattına giremez. Ama "karara giremez" ile "hiç çağrılmaz" AYNI ŞEY DEĞİLDİR:
çağrılmayan modül ölçülmez, ölçülmeyen modül kanıt kapısına HİÇ gelemez. Kapının önünde
sıra bile yoktu; bu tur sırayı kurdu, kapıyı DEĞİL.

- **`nihai/ai/observatory.py` (YENİ) — tek gölge geçidi.** `bar_shadow(df, symbol)` mikroyapı
  (HAR-RV 1-adım RV ÖNGÖRÜSÜ · sıçrama payı · Corwin-Schultz etkin makas) + NEDENSEL rejim
  HMM (`hmm_filter` tabanlı; `hmm_smooth`/`hmm_viterbi` look-ahead içerir, BURADAN ASLA
  çağrılmaz) + veri sağlığı skorunu tek çağrıda üretir. `AI_OBSERVATORY=off` kapatır.
  **HMM yeniden-fit disiplini:** `gaussian_hmm_fit` İÇ-ÖRNEKLEM fit'tir (65 ms/sembol) →
  model sembol başına cache'lenir, `AI_OBS_HMM_REFIT_BARS` (60) barda bir yeniden fit
  edilir, aradaki her bar tek ileri geçişle (≈1,7 ms) puanlanır = modülün KENDİ sözleşmesi
  ("eğitim penceresinde fit et, ileri uygula"). **RV serisi sınırı bilinçli:** canlı hatta
  tek bar df'i vardır → RV KAYAN pencereyle (`AI_OBS_RV_WINDOW`, 20) kurulur, pencereler
  ÖRTÜŞÜR, bu yüzden HAR katsayılarının standart hataları OKUNMAZ (yalnız nokta öngörüsü).
  `rv` ve `har_rv_fc` AYNI tanımdadır (Σr², ortalama DEĞİL) — biri ortalama olsaydı öngörü
  20 kat düşük görünür ve "model bozuk" diye YANLIŞ teşhis edilirdi.
- **Gölge kolonlar `mkt_health`/`har_rv_fc`/`spread_bps`/`hmm_mode_prob`/`hmm_switch`/
  `hmm_label`.** `_neural`/`_dirbar` ile BİREBİR aynı sözleşme: satırda `_obs` altında
  taşınır, `record()` sonrası DÜŞÜRÜLÜR → web JSON'una ve pano tablosuna ASLA girmez.
  `_neural`'dan tek farkı ADAY şartının OLMAMASIDIR: sağlık/makas/rejim adaya değil
  SEMBOLE aittir, yalnız adaylarda ölçmek gölge kolonu seçilim yanlı yapardı.
  `hmm_label` `hmm_mode_prob` OLMADAN yazılmaz (Tur-33: "bir ORAN, paydası okunmadan
  okunamaz"); makas yansız tahmini pozitif değilse kolon BOŞ kalır (Tur-31/33: ölçüm
  YOKLUĞU ölçüm gibi raporlanamaz).
- **Yüzey bağlantıları.** `signals/market_regime.regime_shadow()` rejimin OLASILIKSAL
  ikinci okumasıdır ve `compute_regime`'in İÇİNDE DEĞİL YANINDADIR — çarpanı üreten
  fonksiyonun gövdesinde tek satır AI yoktur (bir AI importu, sonraki düzenleyici için
  "buradan besleyebilirim" davetidir). `data/data_provider.bars_health()` Madde 34'ün BAR
  tarafını kurar (`is_quote_healthy` kotasyonu denetliyordu, barları HİÇBİR ŞEY
  denetlemiyordu) ve `get_bars`'tan ÇAĞRILMAZ (bar başına bir kez, observatory üzerinden).
  `terminal_dashboard` 60 sn TTL'li kalibrasyon şeridi · `telegram_bot.ai_status_text()` +
  `/beyin` komutu · `web/tearsheet` AI kalibrasyon eki · `daily_summary --calib` gece
  görevi (`calibration_loop` + `adaptive_calibration` artık üretimde ÇAĞRILIYOR; `--calib`
  verilmedikçe yalnız kayıtlı artefakt OKUNUR — Tur-35: ölçüm bütçesini OKUYAN da öder).
- **`risk/options_analyzer` vollib'siz dalı UYDURMA SABİT döndürüyordu**
  (`delta=±0.5, gamma=0.02, theta=-0.05, vega=0.10`) — S/K/T/σ'ya HİÇ bakmıyor ve Madde 27
  portföy Greek limitleri o sabitlerin üstünde işliyordu; ölçüm YOKLUĞU ölçüm gibi
  raporlanıyordu, üstelik KARAR YOLUNDA. Artık `options_math.implied_vol` (Brent) +
  `quant_math.bs_greeks`. **Birim hizası ölçüldü:** `theta/365` ve `vega/100` ile vollib'e
  6 ondalığa kadar BİREBİR (atlanırsa iki yol 365 ve 100 kat ayrışırdı). Çözülemezse
  `None` — uydurma sabite GERİ DÖNÜLMEZ. Bu bir MODEL değil DETERMİNİST FORMÜLDÜR: kanıt
  kapısının koruduğu şey kanıtlanmamış modelin karara girmesidir, formülün uydurma sabitin
  yerini alması o kapıyla ilgili değildir, kapının ölçtüğü sayıyı DÜRÜSTLEŞTİRİR.
- **Kilit: `tests/test_tur37_ai_baglanti.py` (40 test).** İki soruyu birden sorar —
  "bağlandı mı?" (her köprünün ÜRETİM ÇAĞIRANI var mı; yoksa kusurun kendisini tekrar
  etmiş olurduk) ve "anayasa bozuldu mu?" (`_obs` düşürülüyor mu, `compute_regime`
  gövdesinde AI importu var mı, `regime_multiplier` gölge okumadan etkileniyor mu,
  `nihai.ai` yukarı doğru execution/signals/risk import ediyor mu).

**İÇ AI ARTIK KARAR/EMİR YOLUNA BAĞLANABİLİR — `AI_LIVE_MODE` (Tur-38, 2026-08-21).**
Tur-37 sırayı kurmuştu, kapıyı değil: iç AI ölçülüyordu ama canlı bir aday sinyale
DOKUNAMIYORDU. Bu tur yolu kurar ve anahtarı operatöre verir — kapıyı AÇMAZ.

- **`nihai/execution/ai_advisor.py` (YENİ) — tek karar geçidi.** `advise(symbol, df, sig)`
  `ai.decision.decide`'ı canlı aday için çağırır ve tek satırlık bir görüş döndürür.
  Kip ÇAĞRI ANINDA okunur (`decision.resolve_mode` deseni): `off` (VARSAYILAN — tek satır
  bile çalışmaz, davranış Tur-37 ile BİT-DENK) · `advise` (görüş yalnız `sig.reasons`/karar
  loguna ve oradan Telegram "emir" metnine yazılır; sonucu DEĞİŞTİRMEZ) · `enforce`
  (ABSTAIN veya ters yön → VETO: uyarı da giriş de durur; veto yoksa posterior boyutu kısar).
  Tanınmayan/yanlış yazılmış değer → `off` (yanlış yazım sessizce enforce'a düşemez).
- **Bağlandığı üç nokta (hepsi `execution/main.py:scan_symbol` — tek canlı karar yolu):**
  (1) KARAR: tüm determinist guard'lardan (kill-switch/seans/cooldown/olay/halt/kotasyon)
  SONRA, sinyal-only uyarıdan ÖNCE — AI kuralların YERİNE değil ARDINA konur, hiçbirini
  atlatamaz. (2) EMİR: veto uyarıyı da susturur (bu sistemde uyarı = emir), veto yoksa görüş
  `sig.reasons` üzerinden Telegram metnine girer. (3) BOYUT: `conf_mult *= size_mult`,
  `prob_sizing`'in yanında ve onunla AYNI sözleşmede.
- **TEK YÖNLÜ KISITLAMA — enforce yalnız SUSTURUR ve KÜÇÜLTÜR.** Yön ÜRETMEZ (LONG adayına
  SHORT_BIAS gelirse sonuç VETO'dur, sinyalin yönü değişmez — "aleyhte kanıttan ters yöne
  gir" ACI DERS #3/#7'ye aykırıdır). Boyut BÜYÜTMEZ (tavan 1.0). ÇIKIŞ YOLUNA HİÇ BAĞLANMAZ:
  LONG/SHORT dışı sinyalde (EXIT/WAIT) `advise()` None döner — bir AI vetosunun zarar-kesmeyi
  geciktirmesi bu sistemdeki en pahalı hata olurdu.
- **Yeni model/olasılık kaynağı DOĞMADI.** p_win, `strategy.py`'ın zaten yazdığı
  `meta_p_win` metadata'sından okunur (ikinci bir çıkarım yok). Nöral gölge bilerek
  bağlanmadı — canlı tarayıcı sürecinde torch importunu tetiklerdi (Tur-26 disiplini).
- **ÖLÇÜLDÜ (canlı öğrenme durumu, n=18.099): enforce BUGÜN HER SİNYALİ VETO EDER.**
  `ai_learning_state.json`: GLOBAL post=0,4861 (lo95 0,4788) · LONG lo95=0,4477 ·
  SHORT lo95=0,4990 — üçünün de %95 CI TABANI 0,5'in ALTINDA, yani kapı-a hiçbir yönde
  geçilemiyor ve motor ABSTAIN diyor. Bu bir kusur DEĞİL, ACI DERS #2/#4/#5'in kodda
  konuşmasıdır: 18 binden fazla çözülmüş canlı tahminde ölçülen şey "kenar yok"tur.
  Bugün kullanılabilir kip `advise`'dır; `enforce` ancak öğrenme durumu bir yönde
  CI-tabanı>0,5 üretirse anlamlı olur. `enforce` bir KENAR İDDİASI DEĞİL, operatör tercihidir.
- **KANIT DÖNGÜSÜ KAPANDI (Tur-38B).** Verdikt artık yalnız metin loguna değil,
  `predictions_log.csv`'ye MAKİNE-OKUNUR kolonlara yazılıyor: `ai_action` · `ai_post` ·
  `ai_veto` · `ai_side`. Kaynak `ai_advisor.shadow_verdict` ve o **KİP-BAĞIMSIZDIR** —
  `AI_LIVE_MODE=off` iken de ölçer, çünkü "kapı açılmalı mı?" sorusunun tek dürüst cevabı
  kapı KAPALIYKEN biriken kayıttır (META_GATE shadow dersi). `AI_VERDICT_LOG=off` kapatır.
  `advise()` ile AYNI çekirdeği (`_judge`) çağırır: loglanan verdikt, enforce'un
  uygulayacağı verdiktin ta kendisidir (ayrı hesaplasalardı kanıt yanlış soruyu yanıtlardı).
  `ai_side` ZORUNLU: getiriyi işaretlemek için taraf şart ve `pwin_side` META_GATE
  kapalıyken BOŞ kalır. `side` override'ı META_GATE=enforce'un WAIT'e çevirdiği adayı da
  ölçüme sokar (seçilim yanlılığı önlenir). `_ai` satırdan `record()` sonrası DÜŞÜRÜLÜR.
- **Kolonu OKUYAN taraf: `prediction_tracker.ai_gate_sim()`** — `gate_sim`'in iç AI
  karşılığı (AI'ın geçireceği ↔ susturacağı ↔ taban; iki tarafta <30 aday → INSUFFICIENT).
  Yazılıp okunmayan kolon bu projede kusurdur. **Bugün yapısı gereği INSUFFICIENT döner:**
  motor %100 ABSTAIN verdiği için "geçen" kovası hiç dolmuyor.
- **Kilit: `tests/test_tur38_ai_karar_koprusu.py` (48 test).** Üç soruyu birden sorar —
  "bağlandı mı?" (main.py köprüyü çağırıyor mu, veto girişi durduruyor mu, çarpan sizing'e
  giriyor mu), "varsayılan güvenli mi?" (off'ta `decide()` bir kez bile çağrılmıyor;
  6 yanlış-yazım enforce'a düşmüyor), "tek yönlü mü?" (yön çevrilemiyor, boyut
  büyütülemiyor, EXIT/WAIT'te `decide()` hiç çağrılmıyor, AI kapısı guard'lardan sonra).

**TABAN SİNYALİN İKİ ÖLÇÜM KUSURU KAPANDI (Tur-39, 2026-08-21).** Canlı sanal işlem
defteri (510 işlem, 2026-07-06→08-20) ilk kez UÇTAN UCA okundu: kazanma %32,5 ·
**PF 0,518** · beklenti −505 $/işlem · toplam **−257.634 $** · ort. R −0,209. Zarar
stratejinin YÖN başarısızlığı değil, İKİ BİRİM/FİYAT kusuruydu — ikisi de karar yolunda.

- **(1) Sinyal fiyatı ile İŞLEM fiyatı aynı şey değildi.** `generate_signal` fiyatı
  `df_htf["close"].iloc[-1]`'dir; `HTF_INTERVAL=1h` iken bu ~59 dk BAYAT olabilir, pozisyon
  ise 1m/kotasyonla yönetilir. ÖLÇÜLDÜ: işlemlerin **%46'sında MFE tam 0,00** (pozisyon
  giriş fiyatının lehine BİR KEZ bile geçmemiş = girilen fiyat piyasada YOKTU) ve `GAP_STOP`
  sınıfı **68/68 aleyhte**, ort. **−%2,45 / −1,497R**, toplam **−299.438 $** — sistemin TÜM
  net zararından BÜYÜK (bu sınıf olmadan kalan 442 işlem **+41.804 $ / PF 1,178**). 27'si
  girişten sonra 2 dakikadan kısa sürede tetiklenmiş. Gerçek bir gap dağılımı SİMETRİKTİR;
  68/68 aleyhte olmak piyasa değil ölçüm kusurudur (Tur-37'nin uydurma-Greek dersiyle aynı
  sınıf: karar yolundaki bir sayı gerçeği değil bir varsayımı raporluyordu).
  **Onarım `strategy.reanchor_to_live()`** (saf fonksiyon): giriş canlı kotasyona çapalanır,
  stop/target YAPISAL bırakılır (kaydırılmaz — kaydırmak RR'ı sahte biçimde korurdu), RR
  yeniden hesaplanır; fiyat stop'un ötesine geçmişse `STALE_PRICE_THROUGH_STOP`, RR eşiğin
  altına düşmüşse `STALE_RR` ile GİRİLMEZ. Bu YENİ bir filtre değil, mevcut `min_rr_ratio`
  kapısının GERÇEKTEN işlem görülebilecek fiyatta uygulanmasıdır. Kotasyon okunamazsa sinyal
  DEĞİŞMEDEN geçer (ölçüm yokluğu bir RED gerekçesi değildir). `LIVE_ENTRY_ANCHOR=off` eski
  davranışa döndürür (A/B için); **varsayılan AÇIK** çünkü bu bir strateji tercihi değil
  onarımdır.
- **(2) Zaman-stopu BAR değil GÜNCELLEME sayıyordu.** `time_stop_bars=12` canlı döngüde
  ~60 sn'lik quote yoklamalarını sayıyordu → ufuk **12 DAKİKA** oluyordu, oysa stop/target
  geometrisi 1 SAATLİK bardan türer ve 2R hedefi saatler ister. ÖLÇÜLDÜ: 510 işlemin
  **405'i TIME_STOP** ile (medyan ~17 dk) kapandı ve kazananları hedefin çok altında
  **+0,14R**'de kesildi (MFE ort. %0,75) — hedef ile ufuk AYNI işlemi tarif etmiyordu.
  Ayrıca ufuk tarama hızına bağlıydı: worker/uyku ayarı zaman-stopunu SESSİZCE kaydırırdı.
  Artık eşik AÇIK DAKİKADIR (`TIME_STOP_MINUTES`, varsayılan **90**) ve `bars_held` koşulu
  KORUNUR (AND): backtest'te her güncelleme gerçek bir bar olduğu için eşik ESKİSİYLE AYNI
  barda tetiklenir (parite ölçüldü), canlıda ise yalnız GECİKİR — kural asla eskisinden
  ERKEN tetiklenemez.
- **Gecelik taşıma YOK: `SESSION_FLAT`.** Kapanışa `FLAT_BEFORE_CLOSE_MIN` (vars. 10) dk
  kala açık pozisyon kâr/zarar durumundan BAĞIMSIZ kapatılır (gap riski pozisyonun yönünü
  sormaz). Stop/target'tan SONRA, zaman-stopundan ÖNCE. `session_remaining` kullanılır →
  yarım-gün kapanışlarını da bilir. Kapalı seansta None → kural uygulanmaz. `=0` ile
  kapatılabilir (gecelik taşımaya bilinçli dönüş; ayrı bir gecelik boyut/hedge kuralı ister).
- **Kilit: `tests/test_tur39_canli_giris_capasi.py` (27 test).** Çapa: stop'u geçmiş fiyat,
  kaçmış kurulum, sınır (risk=0), SHORT işareti, kotasyon yokluğu, üretim çağıranı ve
  "uyarıdan ÖNCE" sırası. Ufuk: 12 güncelleme/12 dk kapatmıyor, 90 dk'da kapatıyor,
  ESKİSİNDEN ERKEN tetiklenmiyor, backtest paritesi, seans-sonu düzleşme + öncelik sırası.
- **ÜÇÜNCÜ HİPOTEZ.** `research/gap_stop_diagnose.py` (2026-08-17) aynı kaybı bulmuş ama
  yalnız İKİ piyasa açıklaması kurmuştu: H1 mikroyapı (stop makasın içinde) · H2 gerçek
  sıçrama. Verinin **68/68 tek yönlü** olması ikisini de BİRİNCİL neden olmaktan çıkarır —
  ne makas ne gerçek sıçrama her seferinde aleyhte çıkar. Eksik olan H3'tü: giriş fiyatının
  kendisi piyasada yoktu. Ders: bir kaybı açıklarken piyasa hipotezlerinden önce ÖLÇÜM
  hipotezi elenmelidir; simetri kırılması (%100 tek yön) piyasayı değil boruyu gösterir.
- **⚠️ ÖLÇÜM SINIRI:** "GAP_STOP olmadan PF 1,178" bir KENAR İDDİASI DEĞİLDİR — kalan 442
  işlem de aynı bayat-fiyat yanlılığını taşıyor. Onarımın gerçek etkisi ancak ONARIMDAN
  SONRA biriken defterle ölçülebilir. Eski defterle yeni defter KARŞILAŞTIRILAMAZ.

**POZİSYON BAŞINA NOMİNAL TAVANI — `MAX_POSITION_NOTIONAL` (Tur-40, 2026-08-21).**
`PortfolioCaps.max_position_notional` ($, **0 = kapalı** → varsayılan davranış BİT-DENK)
pozisyon BAŞINA nominal tavanı koyar. Kritik ayrım: bu bir RED kapısı DEĞİL bir BÜYÜKLÜK
tavanıdır — cap-aware boyutlama adayı reddetmez, tavana sığacak şekilde KÜÇÜLTÜR. Sonuç:
aynı sermaye daha çok eşzamanlı pozisyona dağılır (tek isimde yoğunlaşma, adayı kaybetmeden
düşer). Bir tavanı "reddet" olarak kurmak ile "küçült" olarak kurmak AYNI risk hedefine
farklı bir portföy üretir; hangisi olduğu bir uygulama detayı değil bir KARARDIR.
- `risk_manager` ve `backtest` AYNI alanı okur — iki tavan tanımı olsaydı kâğıt ile canlı
  SESSİZCE ayrışırdı (tek kaynak kuralı; Tur-39 zaman-stopu paritesiyle aynı disiplin).
- Bağlayıcı limit ADIYLA raporlanır (`MAX_POSITION_NOTIONAL_<$>`) → pozisyonun NEDEN
  küçüldüğü loglardan okunur. Sessizce küçülen pozisyon bu projede kusurdur.
- Kilit: `tests/test_tur40_pozisyon_nominal_tavani.py`.

**KAYIP KOD TRANSKRİPTTEN GERİ KAZANILDI + GÖLGE NEURAL KABLOSU (Tur-41, 2026-08-21).**
Tur-35/36 işi bir önceki oturumun `git checkout --` komutuyla yok edilmiş, kırık hâl
`f3c3713` ile commit edilmişti (43 test kırmızı). Kod YENİDEN YAZILMADI:
`~/.claude/projects/<slug>/*.jsonl` oturum transkriptlerindeki `Edit` çağrılarının
`old_string`/`new_string` çiftleri zaman sırasına dizilip mevcut dosyalara uygulandı
(neural.py 65/91, learning.py 23/25, models.py 5/10). Doğrulama İDDİA değil ÖLÇÜMDÜ:
`quant_transformer_ultra` **771.874.072** ve `quant_transformer_mega` **2.014.146.906**
parametre — belgedeki sayılarla BİREBİR.
- **Replay'in tuzağı: uygulanamayan edit YİNELENEN tanım bırakır.** Aynı sınıf iki kez
  tanımlanınca İKİNCİSİ kazanır ve ilkinin forward'ı sessizce düşer — import hatası YOK,
  test kırmızısı belirsiz. Dört yinelenen blok ayıklandı, kalan altı boşluk elle bağlandı.
  Ders: bir replay'in başarı ölçütü "hata vermedi" değil, ÖLÇÜLEN bir değişmezdir
  (burada parametre sayısı); replay eden her akış böyle bir çapa taşımalıdır.
- **`brain.think()` üretimde HER ZAMAN `neural_model=None` ile çağrılıyordu** → ÖNGÖR adımı
  hiç çalışmamış, diskteki eğitilmiş artefaktı canlı akışa yükleyen kod HİÇ yazılmamıştı.
  Tur-37'nin "bağlı değil/çağrılmıyor" dersinin sinir-ağı tarafı: eğitmek ile BAĞLAMAK ayrı
  işlerdir ve ikincisi yapılmadan birincisi ölçülemez. `shadow_runner` artık:
  `_load_neural_model()` DATA_DIR'daki en yeni uyumlu `neural_*` manifestini yükler
  (`AI_NEURAL_MODEL` ile override) — **ŞEMA KAPISI ZORUNLU:** `feature_names` canlı şemayla
  eşleşmezse artefakt ATLANIR, çünkü brain eksik özelliği 0.0 ile doldurur ve uyumsuz model
  GÜRÜLTÜYÜ `p_win` diye taşırdı. `_warm_vix()` koşu başında bir kez ısıtır (`AI_SHADOW_WARM_VIX=0`
  kapatır); onsuz taze süreçte `features=None` kalıyor ve model yüklense bile ÖNGÖR PASİF
  kalıyordu. Özete `n_neural` eklendi: kablonun ÖLÇÜSÜ — sessiz pasiflik görünür olsun diye.
- Model KARAR VERMEZ: gölge `p_win ± std`, `decision.resolve_mode()` disiplininde.
  Ölçüldü (izole veri dizini, AAPL+MSFT): `n_ok=2 n_fail=0 n_neural=2`.
- Kilit: `tests/test_tur40_neural_kablo.py` (4 test) + `test_ai_shadow_runner`'a `WARM_VIX` pini.

**BELLEK REÇETESİ — `--optimizer adafactor` + `--amp` (Tur-42, 2026-08-22).** ULTRA/MEGA
basamakları merdivende VARDI ama 8 GB'lık bir kartta EĞİTİLEMİYORDU: ölçüldü, ULTRA
(771.874.072 parametre, float32) AdamW ile 16,94 GiB ayırıp OOM veriyor — durumun tek başına
≈6,17 GB'ı optimizatörün. İki OPT-IN bayrak eklendi; ikisi de **varsayılan KAPALI** ve
varsayılan yol v8 ile **BİT-DENK** (`tests/test_tur42_bellek_recetesi.py`, 20 test).
- **`optimizer_name="adafactor"` (Shazeer ve Stern 2018).** İkinci momenti SATIR×SÜTUN
  çarpanlarına ayırır, birinci momenti HİÇ tutmaz → durum O(n) yerine O(√n). Ölçüldü
  (512×512 = 262.144 parametre): AdamW **8,00 B/parametre**, Adafactor **0,0156 B/parametre**
  — **512×**. **BEDEL SAKLI DEĞİL:** bu FARKLI bir optimizasyon algoritmasıdır, aynı reçetenin
  ucuz hâli değil; ağırlıklar AdamW koşusuyla bit-denk OLMAZ. Bu yüzden seçim
  `meta()["recipe"]["optimizer"]`e ve oradan MODEL KARTINA yazılır — bir artefakt hangi
  reçeteyle eğitildiğini taşımadan karşılaştırmaya sokulamaz (Tur-22 kart kuralı).
  Yanlış yazım SESSİZCE AdamW'ye düşmez, `ValueError` verir.
- **`amp=True`: bf16 autocast (yalnız CUDA'da; ağırlıklar float32 kalır).** GradScaler YOK —
  bf16'nın dinamik aralığı float32 ile aynıdır (fp16 değil), ölçekleme gereksizdir.
  **"İSTENDİ" İLE "OLDU" AYNI ALANA YAZILAMAZ:** CPU'da `amp=True` bir hata değildir ama bir
  etki de değildir; kayda `amp: False` düşer. Aksi hâlde kart yalan söylerdi (Tur-31/34'ün
  "ölçüm yokluğu ölçüm gibi sunulamaz" kuralının reçete tarafı).
- **bf16 MoE'yi TİP YÜKSELTMESİYLE kırıyordu.** Autocast altında `out` bf16, kapı olasılığı
  (softmax fp32 listesindedir) float32 → çarpım float32'ye YÜKSELİR ve
  `index_add_(): self (BFloat16) and source (Float) must have the same scalar type` patlar.
  `.to(out.dtype)` kapatır; amp kapalıyken NO-OP'tur. Kilit CUDA GEREKTİRMEZ — aynı hata CPU
  autocast'inde bf16 girdiyle birebir üretilir; ayrıca bir KONTROL testi hatanın var olduğunu
  kanıtlar (cast'in kargo-kült olmadığı gösterilmeden eklenen guard, ölü koddur).
- **Kütüphane fonksiyonundaki bayrak, ÇAĞIRANI yoksa gönderilmemiştir.** `fit()` iki
  parametreyi almış ama `research/train_neural.py` reçetesine HİÇ girmemişti → CLI'dan
  erişilemiyordu, model kartına düşmüyordu, tek testi yoktu. "Yazılıp okunmayan kolon
  kusurdur" (Tur-38) kuralının eğitim tarafı: yazılıp ÇAĞRILMAYAN bayrak da kusurdur.
  `--optimizer {adamw,adafactor}` ve `--amp` eklendi; `adamw` bir SEÇİM değil VARSAYILAN
  olduğu için reçete ÖZET satırında listelenmez (yoksa her koşu reçete değiştirmiş görünürdü).
- **Devre DIŞI bir bağlam yöneticisi bedava değildir.** `torch.autocast(..., enabled=False)`
  toplu-iş başına **7,04 µs** (20.000 yineleme ölçümü); yeniden kullanılan `nullcontext`
  **0,30 µs** — **23×**. Bağlam artık döngü DIŞINDA bir kez kurulur ve yalnız amp açıkken
  `autocast` olur (CUDA'sız kurulumda `autocast("cuda")` KURMAK bile uyarı bastırır; artık
  hiç kurulmaz). Tur-34/36 dersinin dördüncü tekrarı: sıkı döngüdeki genel-amaçlı
  sarmalayıcı, sardığı işten pahalı olabilir.
- **Bir yorumdaki anahtar adı da API'dir.** Kod `meta()["training"]["optimizer"]` diyordu,
  gerçek anahtar `meta()["recipe"]["optimizer"]`. Var olmayan bir alana işaret eden belge,
  onu arayan bir sonraki turu sessizce yanlış yere gönderir.
- **`mypy .` yeniden TEMİZ (164 dosya, 0 hata).** Turlar "mypy temiz" diye raporlanırken
  gerçekte `mypy nihai` koşuluyordu; README'nin gate'i `mypy .`. Biriken 18 hatanın 16'sı
  TEK kök nedendi: `rows = []` literalinden çıkarılan `dict[str, str | float]` birleşimi
  `r["expectancy"] * r["trades"]` aritmetiğini imkânsız kılıyordu (`list[dict[str, Any]]`
  ile kapandı). Kalan ikisi tip DARALTMASI: `dict | None` dönen bir fit sonucunu önceden
  `dict` bağlanmış ada atamak (ayrı ad ile), ve serbest-metin CSV alanının `Any` olduğunun
  söylenmemesi. Ders: bir kalite kapısını RAPORLAMAK, onu tam kapsamıyla KOŞMAK değildir —
  kapının komutu belgede yazılıysa turda o komut çalışmalıdır.

**QUANT MODELLERİ PANODA GÖRÜNÜR OLDU (Tur-56, 2026-08-29).** Tur-37 modülleri
üretimde ÇAĞRILIR yapmıştı; operatör onları hâlâ GÖREMİYORDU. Panoda VaR 4 sayıydı,
Greekler 4 çubuk, portföy bir `<pre>` bloğuydu; GARCH · SVI · RND · HMM · BOCPD · HAR-RV ·
VPIN · Kyle λ · Almgren-Chriss · Kelly'nin panoda **hiçbir yüzeyi yoktu**. Bir volatilite
modeli tek sayıya indirgenince "anlaşılır" olmaz — koşullu vol yolu, dağılımın kuyruğu,
gülümsemenin eğimi ancak ÇİZİLİNCE okunur.

- **`nihai/web/api_quant.py` (YENİ) — 11 salt-okunur uç.** `catalog · risk · vol · regime ·
  mc · options · smile · portfolio · micro · execution · kelly`. **YENİ MATEMATİK YOK:**
  `quant_math`/`options_math`/`advanced_math`/`microstructure`/`portfolio`/`regime_hmm`
  fonksiyonları çağrılır, çıktıları GRAFİĞE HAZIR seri/ızgaraya çevrilir. Panoda yeni
  `Modeller → Quant Modelleri` sekmesi: 10 panel, **33 grafik + 84 ölçüm kartı**, tembel
  yüklenir (5 sn'lik ana döngüye maliyet EKLEMEZ).
- **Anayasa korunur:** hiçbir uç sinyal/eşik/META_GATE/execution'a bağlanmaz; hepsi
  RAPORLAR. Kapasite/görünürlük artışı bir KENAR iddiası DEĞİLDİR (ACI DERS #5).
- **Grafik tek başına yetmez — metin modelin PARÇASIDIR.** Her panel SORUyu, modeli ve
  **"Nasıl okunur"** notunu basar; her kutunun altında o kutuya özel yorum, panelin dibinde
  kaynak fonksiyon adları vardır. Metinler `_CATALOG`'da TEK kaynaktadır ve
  **68 formül referansının hepsi testte içe aktarılarak doğrulanır** — uydurma fonksiyon
  adı YOK (`knowledge.py` disiplininin web karşılığı).
- **`null` ÇİZİLMEZ.** Sonlu olmayan her değer `None`'a çevrilir ve grafik o noktayı boş
  bırakır (`spanGaps:false`). Ölçülmemiş noktayı sıfırla/son değerle doldurmak, ölçüm
  yokluğunu ölçüm gibi göstermek olurdu (Tur-31/33).
- **BULGU-56-1 — bir serileştirici `None` alanı DÜŞÜRÜNCE "ölçülemedi" mesajı KAYBOLUR.**
  `_nums()`'un ilk sürümü yalnız `int/float/bool` alanları geçiriyordu. `roll_spread` ve
  `corwin_schultz` "ölçüm yok"u tam da `spread: None` ile söyler (Tur-33 BULGU-33-4'ün
  sözleşmesi); anahtar düşünce pano açısından alan HİÇ YOKMUŞ gibi görünüyordu ve dürüst
  cevap sessizce sıfırlanıyordu. `None` artık KORUNUR, `bool` de bool kalır
  (`reliable: false`, `0.0` değil). Ders: ölçüm yokluğunu taşıyan sözleşmeler
  **serileştirme katmanında** kırılır — bir "yalnız sayıları geçir" filtresi, tam da
  sayı OLMAMASIYLA anlam taşıyan alanı yok eder.
- **IGARCH köşesi ve negatif Kelly kenarı YÜZEYE taşındı.** `uncond_vol` yalnız
  `igarch_boundary=False` iken sayıdır, aksi halde pano **"TANIMSIZ"** yazar (Tur-49);
  `kelly_binary` negatif kenarda bilinçle 0 döner ama `edge` İŞARETİ taşır ve kart onu
  kırmızı gösterir — "kenar yok" ile "tam başabaş" ayrı okunabilsin diye.
- **Kilit: `tests/test_tur56_quant_modelleri.py` (50 test).** Üç soru: *fail-soft mu*
  (sağlayıcı TAMAMEN ölüyken 11 uçtan hiçbiri 500 atmıyor, hepsi `reason` taşıyor),
  *grafik sözleşmesi tutuyor mu* (su-altı ≤ 0 · CVaR ≥ VaR · koni yüzdelikleri sıralı ·
  GARCH yolu tarihlerle hizalı · HMM olasılıkları barda 1'e toplanıyor · MC yelpazesi
  sıralı ve seed deterministik · BS değeri ödemenin altına düşmüyor · risk katkıları 1'e
  toplanıyor · tasfiye yörüngesi monoton azalan), *bağlandı mı* (`QM_MODELS`/`QM_RENDER`
  tabloları katalogla BİREBİR aynı küme; HTML'deki her `qmc-*` kanvası JS'te çiziliyor ve
  tersi — boş kanvas panoda sessiz bir deliktir). Rapor: `docs/TUR56_RAPORU_2026-08-29.md`.

**OPSİYON AKIŞI ve VERİ YETENEK ÖLÇÜMÜ (Tur-57, 2026-08-30).** Dış bir veri-kaynağı
önerisi triyaj edildi; iddialar VARSAYILMADI, **çağrılarak ölçüldü** ve üçü yanlış çıktı.
Rapor: `docs/TUR57_RAPORU_2026-08-30.md`.

- **ÖLÇÜM (ücretsiz Alpaca anahtarı, 2026-08-29):** SIP **tarihsel** barlar YETKİLİ ve IEX
  konsolide hacmin yalnız **%2,2-3,4**'ünü görüyor (AAPL/SPY/NVDA); kapalı olan tek şey
  **son 15 dakikalık** SIP penceresi. OPRA opsiyon zinciri (SPY 13.160 kontrat, 10.633'ünde
  IV+greek), **açık pozisyon (OI)** (trading API kontratları, SPY 91/100, T+1),
  `shortable`/`easy_to_borrow` ve Benzinga haber akışı da ÜCRETSİZ. Yani paylı planın
  ölçülmüş marjinal değeri son 15 dk penceresidir ve bu sistem Madde 3 gereği KAPANMIŞ barla
  karar verir. Ölçüm tekrarlanabilir: `python -m nihai.data.data_capabilities`.
- **`nihai/ai/option_flow.py` (YENİ, KARAR-DIŞI)** — `options_math` TEK kontratı taşıyordu,
  eksik olan **zincirin tamamı** üstündeki OI/hacim toplamlarıydı: strike başına $GEX, net
  GEX, **gamma flip**, call/put duvarları, PCR, max pain, UOA (hacim>OI), dealer vanna/charm.
  **Flip bir eşik değil KÖKTÜR:** yaygın kısayol strike üstünde kümülatif toplam alır, oysa
  gamma spot'a bağlıdır → net GEX spot ızgarasında YENİDEN fiyatlanır ve işaret değişimi
  ara-değerle bulunur. **Varsayım sayıdan ayrılmaz:** dealer'ın call'da uzun/put'ta kısa
  olduğu kabulü işlem TARAFI verisi olmadan doğrulanamaz ve yanlışsa net GEX'in İŞARETİ ters
  döner → her çıktı `dealer_convention` taşır (Tur-34 kuralı).
- **Gelen veri artık düşürülmüyor.** `_normalize_alpaca_bars` `trade_count`/`vwap` kolonlarını
  atıyordu; üç ölçü bu yüzden "ölçülemez" sanılıyordu (`microstructure.relative_volume` RVOL ·
  `trade_size_profile` balina imzası · `vwap_deviation`). **Eski test tam da düşürmeyi
  kilitliyordu** — bir sözleşme testi yazarken sor: *bu bir karar mı, yoksa bir kusurun
  fotoğrafı mı?*
- **Madde 30'un EKSİK YARISI kapandı.** SSR yıllardır kodluydu ama hissenin O GÜN açığa
  satılabilir OLUP OLMADIĞI hiç sorulmuyordu. `main.py` SHORT dalı `asset_flags`'i okur;
  SSR ile AYNI sözleşme: varsayılan İŞARETLER (`NOT_SHORTABLE`/`HARD_TO_BORROW`), bloke
  ETMEZ (bayrak brokerine özgüdür); `SHORT_BORROW_GUARD=block` kapıya çevirir. Bayrak
  okunamazsa hiçbir şey yapılmaz — "bilinmiyor" bir RED gerekçesi değildir (Tur-31).
- **Üç opt-in anahtar, üçü de varsayılan davranışı DEĞİŞTİRMEZ:** `ALPACA_BARS_FEED`
  (SIP barları uzatılmış seansı da getirir → bar KÜMESİ değişir, önce A/B ölç),
  `ALPACA_OPTIONS` (yfinance zincirinin üstüne OPRA NBBO/IV/greek işler; **yalnız
  sonlu-pozitif** değerler yazılır, yani veriyi bozamaz), `SHORT_BORROW_GUARD`.
- **Pano `Modeller → Opsiyon Akışı`** (11. quant modeli) + `/api/quant/flow`; ölçülemeyen
  blok `null` → panoda BOŞ kalır. **Kenar İDDİASI YOK:** bu aileden aday çıkarılacaksa
  `research/edge_lab` kanıt kapısından geçmeli ve defterin `n_trials` sayacını artırmalıdır.
- **Kilit: `tests/test_tur57_akis_ve_veri.py` (40 test).** Matematik pariteleri (gamma
  bit-denk, vanna/charm `options_math` ile aynı, tek-strike max pain), ölçüm-yokluğu
  sözleşmesi (OI yoksa GEX None, hacim yoksa UOA None), veri köprüleri (kolon korunuyor mu,
  zenginleştirme bozabiliyor mu, bayrak önbelleği), SHORT yolu kaynak denetimi ve pano bağı.

**DIŞ RAPOR TRİYAJI — köprüler, muafiyetler, ölçüm (Tur-58, 2026-08-30).** Dış bir
asistanın 9 başlıklı eksik listesi + 8 maddelik yol haritası KODA karşı denetlendi; üç iddia
yanlış, biri raporun söylediğinden BÜYÜK çıktı. Rapor: `docs/TUR58_RAPORU_2026-08-30.md`.
**3097 test** (+45), ruff/mypy temiz, MainHub 27 yeşil.

- **Sürüklenme kapatıldı ve BİR DAHA açılamaz (`tools/sync_mainhub.py`).** MainHub üç
  sağlayıcı dosyasını kopyala-yapıştır paylaşıyordu; Tur-57 düzeltmeleri (`trade_count`/
  `vwap` korunması, OPRA, `asset_flags`) oraya HİÇ ulaşmamıştı → `DATA_PROVIDER=hub` koşan
  süreç için o düzeltmeler YOKTU. Raporun önerdiği `nihai_core` paketi bu depoda fazla
  büyük (MainHub ayrı venv + düz modül düzeni); sürüklenmeyi bitiren asgari şey paket değil
  **tek kaynak + otomatik türetme + kapı**: `nihai/data/X.py` → import yeniden yazımı → 
  `MainHub/X.py`, ve `tests/test_tur58_hub_koprusu.py` sürüklenmede KIRMIZI verir. MainHub
  kopyası bir ARTEFAKTTIR, elle düzenlenmez. `/asset_flags` ucu iki hub_server'a da eklendi.
- **BULGU-58-1: bir uç EKLEMEK mevcut bir yolu SESSİZCE kapatabilir.** `asset_flags()`
  zincir kuralı *"metodu taşıyan İLK düğümün cevabını döndür"* idi; `HubProvider` metodu
  kazanınca hub KAPALIYKEN (dürüst `None`) zincir orada bitiyor ve modül-düzeyi Alpaca
  yedeği ERİŞİLEMEZ oluyordu. Kural üçe ayrıldı: **ölçüm YOKLUĞU (None) aramayı SÜRDÜRÜR ·
  ARIZA (istisna) zinciri BİTİRİR · ölçüm döner.** Arızayı atlayıp yedeğe gitmek arızayı
  GİZLERDİ (Tur-31/34'ün "ölçüm yokluğu ≠ ölçüm" kuralının zincir hâli).
- **Madde 25 — ölü fonksiyon canlandı ve YANLIŞ TARAFTAYDI.** `assignment_risk_short_call`
  yıllardır üretim çağıranı olmayan bir koddu ve bu sistemde uygulanamazdı (opsiyon YAZILMAZ).
  Uzun bacakta olay aynıdır, işareti terstir: ex-tarihte dayanak ≈D düşer, uzun call temettüyü
  ALMAZ → **ekstrinsik < D** olduğu anda "erken kullan ya da temettüyü yitir" ikilemi doğar.
  `options_analyzer.early_exercise_risk` bunu ölçer ve İKİ üretim yolundan çağrılır (kontrat
  seçimi + açık pozisyon). **Yalnız canlı kotasyon yolunda**; `check_exit` gövdesine
  KONULMADI (takvim BUGÜNÜ bilir, backtest barına uygulamak look-ahead olurdu — testle kilitli).
  Sözleşme Tur-57 borç bayrağıyla aynı: varsayılan İŞARETLER, `EXDIV_GUARD=block` kapıya çevirir;
  temettü TUTARI bilinmiyorsa `measured=False`; Avrupa/nakit uzlaşılıda risk YAPISAL olarak yok.
- **Madde 29/53 — raporun kaçırdığı gerçek kusur TERSİYDİ.** "Çapraz wash-sale yok" iddiasının
  yarısı yanlıştı (defter anahtarı zaten DAYANAK sembolü; eksik olan gerekçenin bunu
  SÖYLEYEBİLMESİYDİ → `asset_type` + `WASH_SALE_CAPRAZ`). Asıl kusur: `wash_sale_check`
  SPX/NDX/VIX/RUT/XSP için de işaretliyordu. Nakit uzlaşılı endeks opsiyonları **26 U.S.C.
  §1256** kapsamında piyasaya göre değerlenir ve wash-sale onlara UYGULANMAZ → sistem YANLIŞ
  vergi bilgisi üretiyordu. `settlement_style`'ın `tax_1256` alanı yıllardır vardı ve yalnız
  metadata olarak TAŞINIYORDU (okunmayan ölçüm, Tur-38). Artık okunuyor.
- **Madde 43 — DST doğruydu, kusur DAHA BÜYÜKTÜ.** `market_calendar.local_session_window()`
  (+ `python -m nihai.data.market_calendar [--at|--duration]`) ÖLÇTÜ: ABD seansı TR saatiyle
  **yaz 16:30-23:00, kış 17:30-00:00** (fark +7 ↔ +8 sa; yarım günler de doğru). Yani raporun
  "yılda 2 kez 1 saat" kayması gerçek. AMA `tur17-shadow-gorev.ps1` varsayılanı
  `-At "09:30" -DurationHours 7` idi ve yorumu "kabaca ABD piyasa saatleri" diyordu:
  **örtüşme SIFIR saat** — gölge koşucu her gün KAPALI piyasada koşuyordu. Script artık saati
  ve pencereyi borsanın O GÜNKÜ takviminden TÜRETİR; piyasa kapalıysa sessiz varsayılana
  DÜŞMEZ, hata verir. **Karar yolu zaten DST-güvenliydi** — kayan şey KARAR değil ZAMANLAMAYDI.
- **`nihai/observability.py` (YENİ) — "monitör yok" iddiasının yarısı yanlıştı.** Çağrı/dk
  ZATEN loglanıyordu; eksik olan o sayının MAKİNE tarafından OKUNMASI ve limite yaklaşınca
  taramanın KISILMASIYDI (yazılıp okunmayan ölçüm). `RateBudget` monotonik pencereyle ölçer,
  `throttle_sleep()` yumuşak tavanın (vars. limitin %80'i) ALTINDA **tam 0.0** döner → bugünkü
  rejimde bit-denk (testle kilitli), üstünde oransal fren uygular (üst sınır BİR döngü — fren,
  durdurma değil). Pano `/metrics` Prometheus düz-metni basar; **ölçülemeyen metrik BASILMAZ**
  (0 "sıfır"dır, "ölçüm yok" değil). Yeni bağımlılık YOK, `psutil` opsiyonel.
- **Log SIKIŞTIRMA (`log_retention.compress`).** Retention vardı, sıkıştırma yoktu. Orta bant
  (`compress_days`..`days`) gzip'lenir; silme bandındaki dosya sıkıştırılmaz, `.gz` yeniden
  sıkıştırılmaz, önce `.gz` yazılır SONRA kaynak silinir. **ÖLÇÜLDÜ (gerçek karar günlükleri):
  82,4 → 5,5 MB = ×15,1.** Kazanç yalnız disk değil OneDrive yükleme bandıdır.
- **Madde 44.** Token `!=` yerine `hmac.compare_digest` (zamanlama kanalı) + IP başına
  10 hatada 60 sn kilit. **2FA/oturum/RBAC bilinçle YAPILMADI** ve gerekçe koda yazıldı:
  tek operatörlü localhost aracında rol/oturum yoktur, doğru katman ters vekil + mTLS'tir.
- **Tamamlama: bağlanmamış uçlar kapatıldı.** İlk geçişte üretilip TÜKETİCİSİ olmayan
  yüzeyler bu depoda bitmemiş sayılır (Tur-38). (a) **WebSocket sağlığı `/metrics`e girdi**
  — rapor §H'nin üçüncü kalemiydi; `observability` KENDİ sağlayıcısını KURMAZ (ikinci bir WS
  bağlantısı açardı), sağlayıcıyı kuran iki yer `set_provider()` ile bildirir ve kayıt yoksa
  akış metrikleri hiç BASILMAZ ("0 bağlı" ≠ "sağlayıcı yok"). (b) **Zincir gezintisi TEK
  kaynağa indi** (`data_provider.stream_status`, `asset_flags`in kardeşi) — Tur-35'in "aynı
  desen üç kopyada" dersi bu kez ÖNCEDEN uygulandı. (c) **Tahmin döngüsü gecikmesi** artık
  `record_loop("predict", …)` ile ölçülüyor (eskiden yalnız `snapshot()`taydı). (d) **Pano
  yüzeyi**: `Sistem & Araçlar`ta dört metrik kartı (`parseProm` ile `/metrics` okunur;
  basılmayan metrik "—" kalır, 0 gösterilmez) + opsiyon panelinde **Ex-temettü erken
  kullanım** satırı. (e) `.env.example` dört yeni anahtarı belgeliyor — okunan ama
  belgelenmemiş bir anahtar, olmayan bir ayar kadar kötüdür (Tur-50).
  **Sürüklenme kapısı ilk gerçek işini yaptı:** `data_provider`e `stream_status` eklenince
  test KIRMIZI verdi ve MainHub yeniden üretilerek düzeldi — kapı teorik değil.
- **YAPILMAYANLAR ve gerekçeleri raporun §4'ünde tek tek yazılıdır** (OMS/canlı broker =
  anayasa ihlali + ACI DERS #5 · çok bacaklı opsiyon = göç, ekleme değil · §1256'yı skorda
  öne almak = kanıtsız bileşen, Tur-51 `IV_REGIME_BONUS=off` ilkesi · SQLite göçü = ölçülmüş
  değil VARSAYILAN sorun, pano darboğazı Tur-28/30/31'de zaten çözüldü · SSE = ölçülmemiş
  %80 iddiası, önce `/metrics` ile yük ölç · supervisord = Dockerfile zaten var).

**DIŞ RAPOR TRİYAJI #2 — LOGLAMA POLİTİKASI, İCRA GERÇEKÇİLİĞİ, PİYASA BAĞLAMI (Tur-59,
2026-08-30).** İkinci bir dış eksiklik listesi (7 başlık + 8 maddelik yol haritası) koda
karşı denetlendi; **üç iddia yanlış** çıktı (HRP ve Ledoit-Wolf ZATEN var ve gölgede
koşuyor; intra-bar sıra "varsayımsal" değil kötümser bir KONVANSİYON) ve raporun göremediği
gerçek kusur bulundu. **Bu turda hiçbir kenar iddiası yoktur** — eklenen her şey bir ölçüm
onarımı, bir veri biriktirme yolu ya da bir görünürlük yüzeyidir; skorlama fonksiyonuna tek
bileşen eklenmedi (Tur-51 ilkesi). Rapor: `docs/TUR59_RAPORU_2026-08-30.md`.

- **BULGU-59-1 — ETİKETSİZ HAVUZUN KÖK NEDENİ kapandı (P2).** Özellik vektörü yalnız ADAY
  barlarında yazılıyordu ve ikinci bir kapı (`pwin_side is not None`) onu daha da
  daraltıyordu. Kapının gerekçesi kodda YAZILIYDI (*"pwin_side'sız satıra özellik yazmak
  anlamsız (etiket yok)"*) ve **yanlıştı**: satırın etiketi `pwin`den değil
  `predictions_log`un `hit`/`fwd_ret_pct` kolonundan gelir ve o kayıt ZATEN her yönlü bar
  için yazılıyor. ÖLÇÜLDÜ: `predictions_log` **66.694** ↔ `live_features_log` **8.378**
  satır = **7,96× boşluk**; maliyet `extract_features` **0,084 ms/çağrı** (kurulu `shared`
  önbelleğinden) ve satır 273 bayt. Yeni `cand_side` kolonu ayrımı kayıtsız bırakmaz (eski
  satır kümesi `cand_side != ""` ile birebir geri alınır); `FEATURE_LOG_ALL=false` kapatır;
  `build_live_dataset.build()` (etiketli yol) `pwin_side` filtresini KORUR → mevcut hiçbir
  ölçüm değişmez.
- **BULGU-59-2 — bar AÇILIŞI iki yönde de sıra kanıtıdır (C1/P6).** Kod gap'i yalnız STOP
  tarafında okuyordu; LONG bir pozisyonda bar HEDEFİN ÜSTÜNDE açıldığında hedef ilk tikte
  doludur ve o bar `STOP` yazıyordu — bu kötümserlik değil YANLIŞLIK (sıra belirsiz değil
  BELİRLENMİŞTİ). A/B ÖLÇÜLDÜ (8 sembol, aynı süreçte art arda): **1d**'de 18 çıkış
  `GAP_TARGET` oldu, işlem 283→289, AAPL +10,11→+11,42 · TSLA +13,17→+16,86; **1h**'te
  **BİT-DENK**. Kalan GERÇEK belirsizlik (açılış ikisinin ARASINDA + ikisi de delindi)
  stop-önce kalır ve artık SAYILIR (`intrabar_ambiguous/resolved/flipped` →
  `BacktestResult` + `short_summary`). Ölçülen maruziyet **1/147 (1h) · 0/283 (1d)** olduğu
  için 1m alt-bar ÇÖZÜCÜSÜ YAZILMADI — enjekte edilebilir kanca (varsayılan `None` →
  bit-denk) duruyor. `INTRABAR_POLICY=stop_first` eski davranışa döner.
- **BULGU-59-3 — hacim katılımına duyarlı piyasa etkisi (C2).**
  `kayma = taban + eta*sigma*sqrt(Q/ADV)` (Almgren-Chriss geçici etki); sigma = ATR/fiyat,
  ADV NEDENSEL hareketli ortalama (`backtest._causal_adv`, pandas `rolling(min_periods=1)`
  ile birebir). **`impact_coef` varsayılan 0,0 = KAPALI → çıktı BİT-DENK** (289 işlem,
  sembol başı getiriler aynı); ölçülmemiş bir katsayıyı varsayılan yapmak "kanıtsız
  bileşen" yasağının maliyet tarafı olurdu. Duyarlılık ÖLÇÜLDÜ (1d): eta=0,3 → −1,27 pp ·
  eta=1,0 → −2,68 pp · eta=3,0 → **+1,61 pp** — işaret değişimi kazanç değil YOL
  BAĞIMLILIĞIDIR (maliyet arttıkça farklı işlemler açılır), yani tek bir eta'ya bakıp karar
  verilemez.
- **BULGU-59-4/5 — `tier` kolonu ve `trades.csv`'nin OLMAYAN başlık göçü.** İki katman tek
  deftere yazdığı için Tur-50 brüt maruziyeti ÖLÇEMEDİĞİNİ açık kalem bırakmıştı; katman
  artık pozisyonda taşınıp deftere yazılıyor (kaynak yeni bir bağlam zinciri değil ZATEN VAR
  OLAN thread adı: `TierScanner(name=...)` → `current_tier()`; AÇILIŞTA belirlenir çünkü
  çıkış başka thread'den gelebilir). Kolonu eklemek **latent bir kusuru tetikleyecekti**:
  `log_trade_csv` yalnız "dosya var mı?" diye soruyordu → eski başlığın altına yeni alan
  sırasıyla append = **tüm kolonlar kayar**. `prediction_tracker._ensure_schema` bu sınıfı
  aylar önce kapatmıştı, KARDEŞ defter açıktaydı (Tur-47 kuralı).
- **BULGU-59-6 — beta-ağırlıklı yönsel maruziyet (D4).** `max_net_directional_pct` NOMİNAL
  toplar (1 $ SPY ile 1 $ yüksek-beta isim aynı sayılır). `beta_vs_benchmark` +
  `PortfolioState.beta_exposure` + `RiskManager.beta_hedge_suggestion` eklendi. Üç disiplin:
  betası ÖLÇÜLMEMİŞ pozisyon toplama GİRMEZ ve `coverage` raporlanır (1.0 uydurmak portföyü
  sessizce "piyasa kadar" gösterirdi); eşik için İKİNCİ sabit tanımlanmadı (aynı
  `max_net_directional_pct` okunur); **bloke eden yeni kapı EKLENMEDİ** — dönen şey ters
  bacak ÖNERİSİDİR (Madde 47 duruşu).
- **BULGU-59-7 — piyasa BAĞLAMI kolonları (F1/F2/F3) — `nihai/signals/breadth.py` (YENİ).**
  25 özelliğin tamamı TEK hissenin teknik göstergesiydi. Genişlik (SMA50-üstü payı, yükselen
  payı, **McClellan**), göreceli güç (sembol − benchmark 20-bar getirisi) ve çapraz varlık
  vekilleri (TLT/UUP/USO) `live_features_log`a **EK KOLON** olarak yazılır; **`FEATURE_NAMES`
  DEĞİŞMEDİ** → model girdisi ve tüm mevcut ölçümler aynı. Değerler bir ÖNCEKİ turun
  snapshot'ından gelir (tur içinde SABİT, karar barından kesinlikle önce). Ölçülemeyen alan
  BOŞ; `br_n` bir SAYIMDIR (tam sayı) ve 10'un altında oran YAYIMLANMAZ.
- **BULGU-59-8 — float kaynağı bağlandı (A4).** `turnover_rate` yazılıydı, çağıranı yoktu,
  pano *"float kaynağı bağlı değil"* basıyordu. `market_structure.float_shares()` (yfinance
  + 7 günlük disk önbelleği) eklendi; `float` ile `outstanding` sessizce ikame EDİLMEZ ve
  **`fetch` varsayılan FALSE**tur — pano ucu önbelleği okur, doldurmak ayrı bir adımdır
  (`--warm`). Bunu testteki AĞ GUARD'ı yakaladı: bir HTTP isteğinin içinde saniyelik ağ
  çağrısı ödemek "ölçüm bütçesini OKUYAN da öder" kuralının ihlaliydi.
- **YENİ ARAÇLAR.** `nihai/data/option_archive.py` — OI/hacim taşıyan zincirin GÜNLÜK gzip
  arşivi (DoltHub'da OI/hacim YOK, OPRA yalnız ANLIK → tarihsel akış ancak BUGÜNDEN
  biriktirilerek doğar; gün içi ikinci kesit gzip ÇOKLU ÜYE olarak eklenir ve tekilleştirme
  YAPILMAZ çünkü kesitler arası FARK ölçülmek istenen şeydir).
  `nihai/notifier/deadman.py` — seans AÇIKKEN kanıt akışı bayatsa Telegram, SAĞLIKLIYSA dış
  snitch ping'i (mantık TERSTİR: makine ölürse yerel uyarı çıkamaz, ping'in YOKLUĞU
  tetikler); piyasa kapalıyken hiçbir iddia yok, dosya yokluğu bayatlık DEĞİL.
- **P3 ÖLÇÜLDÜ, VARSAYILAN DEĞİŞMEDİ (`research/feed_ab_test.py`).** IEX-SIP, aynı süreçte,
  tek değişken feed (1h/400 bar): kapanış korelasyonu **0,999836 / 0,999877**, IEX'in
  konsolide hacim payı **%3,9 / %3,2**, ama **bar KÜMESİ değişiyor** (400 barın yalnız
  ~%52'si ortak — SIP uzatılmış seansı taşır) ve **KARAR SPY'de 2/30 (%6,7) AYRIŞIYOR**
  (AAPL 0/30). Tur-31'in VIX göçünde kurulabilen *"ölçümleri geçersiz kılmaz"* cümlesi
  burada KURULAMAZ → `sip` bir veri iyileştirmesi değil bir **strateji değişikliğidir**;
  açılmadan önce backtest edilmelidir.
- **Kilit: `tests/test_tur59_dis_rapor.py` (49 test).** Üç soru: *onarım doğru mu* (gap
  sırası, karekök kapalı formu, nedensel ADV, beta DGP'si), *ölçüm koşulu taşınıyor mu*
  (resolver None → kötümser; float ölçülemezse neden; boş bağlam hücresi; beta kapsamı),
  *geri alma çalışıyor mu* (`stop_first`, `impact_coef=0` bit-denk, `cand_side`).

**GÖLGE YÜZEYLERİN TERFİ KAPISI — on yüzey, tek sözleşme (Tur-60, 2026-08-31).**
Hedef *"gölgede koşanları asıl sisteme al"*dı; kurulan şey bir ANAHTAR değil bir KAPI oldu:
**bugün hiçbir yüzey terfi etmedi** ve iki güçlü görünen aday kapıda ELENDİ. Depoda on
gölge yüzey vardı, yalnız ikisi ölçülüyordu (`gate_sim` meta-kapıyı, `ai_gate_sim` iç-AI'ı);
geri kalan sekizi canlıda hiç değerlendirilmemişti. Rapor: `docs/TUR60_RAPORU_2026-08-31.md`.

- **`research/shadow_promote.py` (YENİ)** — `gate_sim`in sorusunu genelleştirir (*bu kolon
  bir KAPI olsaydı ortalama getiri tabandan iyileşir miydi?*, ACI DERS #3). Altı dürüstlük
  kapısı: örneklem (≥30+30) · **kümele-dayanıklı t (sembol, gün — CR0)** · Bonferroni'li
  `p_adj` · doğru EVREN (taraf-işaretli ↔ yön-işaretli) · **ters kalibrasyon** ·
  **kayma artefaktı**. Terfi ayrıca **zaman-bölmeli OOS onayı** ister ve bölme yüzeyin
  KENDİ satırlarında yapılır (havuzda bölünürse geç eklenen kolonun tüm satırları OOS'a
  düşer — `dir_bar` ilk koşuda tam böyle `NO_RULE` verdi).
- **Entegrasyon: terfi kararı GÜNLÜK döngüye bağlandı.** `forward_harness.publish_side_reports`
  üçüncü yan raporu (`data/forward_eval/shadow_promote.json`) yazar, `daily_summary` onu
  okur ve *"Gölge terfi (10 yüzey): HOLD 3 · INSUFFICIENT 5 · INVERTED 1 · SLIP_ARTIFACT 1
  → terfiye hazır: YOK"* satırını basar. Boş liste sessizce atlanmaz, YAZILIR.
  **Hiçbir canlı kapı AÇILMADI** (`META_GATE`/`AI_LIVE_MODE`/`SIGNAL_ON_BAR_CLOSE`/
  `PROB_SIZING` varsayılanları aynı) — kanıt kapısı geçilmedi, anahtar çevrilmedi.
- **BULGU-60-1 — meta-kapı `pwin` canlıda TERS sıralıyor.** En iyi kapı "DÜŞÜK tut"
  yönünde: `pwin<=0,35` tutulanlar +0,0088% ↔ elenenler −0,0231% (uplift +0,0166pp,
  **t=3,311**, p_adj=0,0093). Bu bir kenar DEĞİLDİR; `pwin_calibration` (TERS_KALIBRASYON,
  ρ=−0,31) ve `gate_sim` (GATE_HURTS) ile ÜÇ bağımsız ölçüm aynı şeyi söylüyor. Kapı bu
  dalı MAKİNEDE tutar (`higher_is_better=True` kolonda "düşük tut" kazanırsa zorla
  `INVERTED`) — aksi halde harness projenin en pahalı yasağını "PROMOTE" diye raporlardı.
- **BULGU-60-2 — "gölge yön uyuşması" bir ÖLÇÜM ARTEFAKTIYDI.** Kural dört kapıyı da
  geçmişti: uplift +0,0065pp, **t=7,407**, elenen 522 satır ort. −0,1648%, ve **OOS bile
  ONAYLADI** (+0,0072pp, t=6,736). Mekanizma sorulunca çöktü: uyuşmaz grupta `slip_bps`
  ort. **+20,83** (uyuşan +0,40) ve bar→bar isabeti **%60,92** (uyuşan %50,25) — yani YÖN
  daha DOĞRU, giriş kovalanmış. Kimlik: `ret_live ≈ ret_bar − slip`. Aynı ileri fiyat,
  ortak referans (`bar_close`) paydasıyla ölçülünce uplift **+0,0065 → −0,0015pp**
  (t=−2,589) ve dilim deseni tamamen TERSİNE dönüyor. Sonuç: `SIGNAL_ON_BAR_CLOSE` bu
  kanıtla AÇILAMAZ. *(Tur-83 düzeltmesi: KAPI değerlendirmesi doğrudur; ama "yön DAHA DOĞRU
  %60,9" `hit_bar` ile ölçülmüştü ve canlı-yön için ileri-bakış taşır — aynı dilim aynı
  referanstan canlı %26,2 ↔ bar %73,5. Anahtar Tur-83'te açıldı; BULGU-83-2.)*
- **BULGU-60-3 — `shadow_runner`/iç-AI ölçülemiyor çünkü KARAR ÜRETMİYOR.** 40 koşu
  (2026-07-25→08-30): `n_ok 1150 · abstain 1150 · long_bias 0 · short_bias 0 · n_neural 839`.
  Nöral kablo canlı ama karar motoru beş haftada TEK yön üretmedi (`ai_gate_sim`in "geçen"
  kovasının boş olmasının nedeni de bu). Bu "kenar yok" değil "çıktı yok"tur; Tur-38'in
  ölçtüğü posterior CI-tabanı<0,5 durumu sürüyor.
- **BULGU-60-4 — asıl darboğaz KANIT KAPSAMASI.** Dolu GÜN sayısı: `pwin` 22 · `dir_bar`/
  `neural_pwin` 8 · gözlemevi ve `ai_*` kolonları **2** · `spread_bps` **1**. Beş yüzeyin
  `INSUFFICIENT` olmasının nedeni yöntem değil TAKVİMDİR — kayıtçı servis Tur-37/38'den
  beri iki gün koştu. `coverage_days` artık her yüzeyde raporlanır ve özete girer.
- **`nihai/execution/shadow_bridge.py` (YENİ) — gölge ARTIK asıl süreçte koşuyor.**
  Ölçümden BAĞIMSIZ, yapısal bir kusur vardı: gözlemevi ve kovalama sapması yalnız
  `live_predict_service` içinde, yani AYRI bir süreçte hesaplanıyordu; aday sinyali
  ÜRETEN yol (`main.scan_symbol`) onları HİÇ görmüyordu. Köprü `reanchor_to_live`
  bloğunun hemen ardına bağlandı (determinist guard'lardan SONRA, uyarıdan ÖNCE —
  `ai_advisor` ile aynı sıra) ve gerçek AAPL 1h barında yedi ölçüyü birden üretiyor:
  `health · har_rv_fc · spread_bps · hmm_mode_prob · hmm_switch_prob · hmm_label ·
  chase_drift_bps`. **Kip `SHADOW_BRIDGE`, VARSAYILAN `advise`:** ölçüler `sig.reasons` +
  `sig.metadata["shadow"]` + karar loguna yazılır, `veto=False`/`size_mult=1.0`
  YAPISALDIR → kip bir kenar iddiası TAŞIYAMAZ. `off` = Tur-59 ile bit-denk. `enforce` =
  yalnız TERFİ ETMİŞ kural karara girer.
  **Terfi defteri (`RULES`) KODDADIR ve `shadow_promote.json`dan OKUNMAZ** — canlı bir
  kapı, herhangi bir koşunun üzerine yazabildiği bir veri dosyasından kural alamaz.
  Bugün defterde tek `PROMOTE` yok → `enforce` de sinyal bastırmaz ve nedenini bir kez
  UYARI olarak loglar. `_enforce_hukmu` defterde `PROMOTE` görüp uygulaması olmayan
  kuralda `AssertionError` atar: bir sonraki tur bir yüzeyi terfi ettirirse onu KARARA
  bağlamadan yeşil geçemez. Tek yönlü kısıt `ai_advisor` ile aynı (yön ÜRETMEZ/ÇEVİRMEZ,
  boyut BÜYÜTMEZ, ÇIKIŞ yoluna HİÇ bağlanmaz). **Maliyet ölçüldü:** çağrı başına
  **5,36 ms** (HMM önbellekli), sembolün ilk adayında 65,7 ms; yalnız LONG/SHORT
  adaylarda çalışır. Nöral gölge AYRI bayrak ister (`SHADOW_BRIDGE_NEURAL=1`, varsayılan
  0) — canlı tarayıcıda `torch` importunu tetikler (Tur-26 hijyeni).
  **Köprünün ölçüleri `predictions_log`a GİRMEZ:** o defter 1m tahmin yüzeyinde birikti,
  tarama yolu HTF barıyla çalışır; ikisini aynı havuza dökmek `shadow_promote`ın ölçtüğü
  dağılımı kirletirdi (kilit `ast` ile denetlenir).
- **Kilit: `tests/test_tur60_golge_kopru.py` (34 test).** Dört soru: *BAĞLANDI MI*
  (`main.py` köprüyü `ast` ile TAM BİR yerden çağırıyor mu; ölçüler gerekçeye/metadata'ya
  düşüyor mu; `conf_mult *= sh_v.size_mult` var mı; sıra çapadan SONRA + uyarıdan ÖNCE
  mi), *VARSAYILAN GÜVENLİ Mİ* (7 yanlış-yazım enforce'a düşmüyor; `off`ta tek satır
  çalışmıyor), *TEK YÖNLÜ MÜ* (EXIT/WAIT'te hiç çalışmıyor; verdiktte yön alanı yok),
  *TERFİ DEFTERİ İŞLİYOR MU* (uygulamasız `PROMOTE` → `AssertionError`).
- **Kilit: `tests/test_tur60_golge_terfi.py` (11 test).** Üç soru: *kapı ELİYOR MU*
  (INVERTED/SLIP_ARTIFACT/INSUFFICIENT/küme-içi sahte anlamlılık), *kapı GEÇİRİYOR MU*
  (gürültülü ama gerçek bir ayrım PROMOTE + OOS CONFIRMED alıyor mu — yalnız reddeden bir
  kapı `return False`tir), *BAĞLANDI MI* (günlük görev yazıyor + özet okuyor mu).
  `test_forward_side_reports` yan rapor kümesini 2 → 3 olarak günceller (kilit TAM KÜME
  üzerinde: sessizce bir rapor düşerse kırmızı).

**DIŞ YOL HARİTASI TRİYAJI — altı quant ailesi uygulandı (Tur-72, 2026-09-06).**
Dış bir "tam kapsamlı quant finans yol haritası" (14 başlık, 39 KB) koda karşı denetlendi.
Başlıkların çoğu ZATEN karşılanmış çıktı (Heston/Merton/Hawkes/LSMC/Crank-Nicolson ·
SVI/SSVI/SABR · HAR-RV/VPIN/Kyle λ/Roll · Black-Litterman/HRP/CVaR-LP/çok-varlıklı Kelly ·
kopulalar/EVT/GARCH(1,1)/frac-diff/imzalar/parçacık filtresi/BOCPD). Kalanların büyük kısmı
BU SİSTEMİN VERİSİYLE ÖLÇÜLEMEZ (sabit getirili/FX/emtia/MBS/kripto piyasa verisi yok ·
L2/L3 emir defteri yok → OFI/VPIN-tick/Avellaneda-Stoikov/spoofing tespiti · XVA/CDO/13F
karşı taraf verisi yok · FIX/OMS anayasaya aykırı: bu sistem EMİR GÖNDERMEZ). Uygulanan
altı aile, ölçülebilir girdisi OLAN ve depoda karşılığı OLMAYAN kesişimdir. **Hiçbiri bir
kenar iddiası taşımaz**; hepsi KARAR-DIŞI ölçüm/rapor yüzeyidir (ACI DERS #5).
Rapor: `docs/TUR72_RAPORU_2026-09-06.md`. **3415 → 3475 test**, ruff/mypy temiz.

- **§6.1 ASİMETRİK GARCH — `quant_math.gjr_garch_fit` · `egarch_fit` · `asym_garch_forecast`.**
  GARCH(1,1) şoku KARESİYLE ölçer: +%3 ile −%3 aynı oynaklık tepkisini üretir. Hisse
  getirilerinde bu ampirik olarak yanlıştır. GJR (Glosten ve ark. 1993) eşikle
  (`1[eps<0]`), EGARCH (Nelson 1991) sürekli işaretle (`z`) asimetriyi ölçer. **İŞARET
  KONVANSİYONLARI TERSTİR ve karıştırmak kaldıracı ters okumaktır:** GJR'de kaldıraç
  `gamma > 0`, EGARCH'ta `gamma < 0`. GJR, GARCH(1,1)'in `gamma=0` kısıtlı hâlidir →
  log-olabilirlik farkı doğrudan bir χ²(1) olabilirlik-oranı testidir (panoda `LR` kartı,
  %95 eşiği 3,84). GJR `lfilter` kısayolunu MİRAS ALIR (aynı birinci-derece doğrusal fark
  denklemi); EGARCH ALAMAZ (girdi terimi durumun kendisine bağlı) → saf Python döngüsü
  bilinçlidir ve log-varyans [−50, 50]'ye kırpılır. Tur-49'un ÖLÇEK NORMALİZASYONU ikisinde
  de uygulanır. EGARCH'ta `uncond_var` DEĞİL **`uncond_var_median`** döndürülür: `exp(ω/(1−β))`
  E[σ²] değil MEDYAN seviyedir ve sayıyı yanlış adla basmak Tur-31 disiplinini bozardı.
- **§10.3 RASTGELE MATRİS TEORİSİ — `quant_math.mp_bounds` · `denoise_corr_mp` · `mst_edges`.**
  T/N küçüldükçe örneklem korelasyonunun özdeğerlerinin çoğu SADECE gürültüdür; optimizasyon
  kovaryansı TERSLEDİĞİ için tam da o en gürültülü yönlere yaslanır. Marčenko-Pastur eşiği
  (λ⁺ = (1+√(N/T))²) altındaki özdeğerler ORTALAMALARIYLA değiştirilir (sabit-artık yöntemi:
  iz korunur, sinyal özvektörlerine dokunulmaz). `ledoit_wolf_cov` ile RAKİP DEĞİL farklı
  önseldir (o tüm matrisi büzer, bu yalnız gürültü alt-uzayını düzler). **q ≤ 1'de MP desteği
  TANIMSIZDIR → None** (sayı basmak yerine ölçüm yokluğu). MST (Mantegna 1999) korelasyon
  uzaklığının (`d = √(2(1−ρ))`, gerçek metrik) en kısa iskeletidir; ağırlık ÖNERMEZ.
- **§2.1 DUPIRE YEREL VOLATİLİTE — `options_math.dupire_local_vol_svi`.** Örtük vol bir
  DİLİM ORTALAMASIDIR, yerel vol ANLIKTIR; ikisini karıştırmak klasik hatadır. Gatheral'in
  toplam-varyans biçimi kullanılır (fiyat türevleri yerine `w` türevleri — sayısal olarak
  çok daha kararlı); `∂w/∂k` ve `∂²w/∂k²` SVI'dan ANALİTİK (`_svi_derivatives`), yalnız
  `∂w/∂T` sonlu farktır. **İKİ ARBİTRAJ KAPISI SESSİZ GEÇİLMEZ:** pay ≤ 0 (takvim) veya
  payda ≤ 0 (kelebek) → o nokta **NaN** ve sayım raporlanır. Yerel varyans negatif çıkamaz;
  kırpıp pozitif bir sayı basmak olmayan bir ölçümü var göstermek olurdu (Tur-33).
  **Dilimler EK AĞ İSTEĞİ OLMADAN gelir:** `/api/quant/smile`in vade-yapısı döngüsü zaten
  6 zinciri çekiyordu, artık aynı zincirden SVI dilimi de uyduruyor.
- **§7.5 VARYANS SWAP — `options_math.variance_swap_strike`.** CBOE'nin VIX'i hesaplarken
  kullandığı ayrık log-kontrat replikasyonu (Demeterfi ve ark. 1999). MODELDEN BAĞIMSIZDIR:
  tek bir σ'ya indirgemez, gülümsemenin TAMAMINI 1/K² ağırlığıyla kullanır → `K_vol − ATM IV`
  farkı doğrudan skew/kurtoz katkısıdır. **ÖLÇÜMÜN SINIRI ÇIKTIDA TAŞINIR:** replikasyon
  SÜREKLİ strike sürekliliği varsayar, gerçek zincir SONLUdur → `kanat_kirpimi` 1'e yakınsa
  kuyruk örneklenmemiştir ve K_var AŞAĞI yanlıdır (ölçüldü: sabit-σ zincirde tam aralıkta
  K_vol 0,30005 ↔ σ=0,30, ±%15 kanatta 0,2575). Bu bir gerçekleşen-varyans TAHMİNİ değil,
  o varyansın PİYASA FİYATIdır.
- **§7.1/§6.3 İSTATİSTİKSEL ARBİTRAJ — `nihai/ai/statarb.py` (YENİ) + `/api/quant/statarb`.**
  `eigenportfolio_returns` (Avellaneda-Lee 2010 öz-portföyleri: korelasyon özayrışımı,
  ağırlık `v_j/σ_j`) · `residual_ou_fit` (kümülatif kalıntının AR(1)/OU uyumu) ·
  `avellaneda_lee_scores` (tam boru hattı → **s-skor** = kalıntının denge dağılımındaki
  z-skoru) · `johansen_test` (trace + max-eigen, Osterwald-Lenum 1992 kritik değerleri
  GÖMÜLÜ) · `vecm_fit` (β' S11 β = I normalizasyonu, α yüklemeleri, ECT serisi + yarı-ömür).
  **Johansen İKİ YOLLUDUR ve yerel yol DİFERANSİYEL DOĞRULANDI:** statsmodels varsa o
  kullanılır, yoksa modülün kendi indirgenmiş-rank regresyonu koşar; aynı panelde iki yolun
  trace istatistikleri **2,3e−11**, özdeğerleri **3,5e−14** farkla örtüşüyor (testte kilitli).
  **Rank 0'da VECM KURULMAZ** ve nedeni yazılır — olmayan bir denge ilişkisi için sayı
  basılmaz. Kritik değer tablosu k−r ≤ 5 içindir; daha geniş evrende istatistik hesaplanır
  ama KARAR VERİLMEZ (`crit=None`). s-skor **bir sinyal DEĞİL bir tanımdır**: eşiği aşsa
  bile `half_life` pencerenin yarısını aşıyorsa `reliable=false` yazılır.
- **§9.1 KÜMÜLATİF BEKLENTİ TEORİSİ — `quant_math.cpt_value` · `prelec_weight` ·
  `cpt_prospect`.** Kelly log-fayda maksimize eder ve operatörün zarardan kaçınmasını yok
  sayar; CPT (Tversky-Kahneman 1992) aynı bahsi insan karar vericinin gördüğü gibi fiyatlar
  (kayıp λ=2,25 kat ağır, küçük olasılıklar Prelec ağırlığıyla şişer). "Kümülatif" olan kısım
  rank-bağımlı KUYRUK ağırlıklandırmasıdır (aksi hâlde model birinci-derece stokastik
  baskınlığı ihlal eder). Panoda Kelly panelinin yanında: `p=0,55 · b=1,5` bahsi EV +0,375
  iken CPT değeri **−0,268**, kırılma λ **1,70**. **Bu bir boyutlama ÖNERİSİ DEĞİLDİR** —
  sistem Kelly kullanır, CPT teşhistir.
- **Pano: 12. quant modeli + dört mevcut panele ek.** `Modeller → Quant Modelleri` sekmesine
  `İstatistiksel Arbitraj` paneli (s-skor çubukları + kümülatif kalıntı yolları + Johansen
  trace/kritik + VECM ECT) eklendi; ayrıca **haber-etkisi eğrisi** (vol), **RMT özdeğer
  spektrumu + λ⁺ çizgisi ve MST derece grafiği** (portföy), **Dupire yüzeyi + varyans swap
  strike katkıları** (gülümseme), **CE(λ) eğrisi** (Kelly). Katalog `formul` referansları
  11 yeni fonksiyonla genişledi ve hepsi testte içe aktarılarak doğrulanır.
- **Paylaşılan panel yardımcısı `_symbol_panel`.** `/api/quant/portfolio` ile
  `/api/quant/statarb` AYNI ortak-tarihli kapanış panelini ister; ikinci bir kopya
  yazmak yerine tek kaynağa indirildi (Tur-35 "aynı desen üç kopyada" dersi ÖNCEDEN
  uygulandı).
- **Kilit: `tests/test_tur72_yol_haritasi.py` (60 test).** Her aile ya BİLİNEN BİR
  ÖZDEŞLİĞE (düz yüzeyde yerel vol = örtük vol, makine hassasiyetinde · sabit-BS zincirinde
  K_vol = σ, %0,016 · yerel Johansen = statsmodels, 1e−11) ya da DİFERANSİYEL bir
  karşılaştırmaya (kaldıraçlı seri ↔ simetrik seri · kointegre çift ↔ bağımsız yürüyüş)
  bağlanır. **11 MUTASYON KOŞTURULDU ve 11'i de YAKALANDI** (EGARCH işaret ters · MP eşiği
  λ⁺→λ⁻ · Prelec kimlik · MST guard kaldır · MST epsilon düşür · varyans swap ağırlığı
  1/K²→1/K · Dupire paydası 1/w işaret · OU denge std düzeltmesiz · statarb panelden
  düşürme · uç alanlarını None yapma ×2).

**320 MADDELİK DIŞ GELİŞTİRME LİSTESİNİN TRİYAJI — sekiz aile (Tur-73, 2026-09-06).**
`finans genel gelliştirme listesi.txt` (320 madde, 87 bölüm, 148 KB) koda karşı denetlendi.
**19 madde uygulandı**; geri kalanı dört kovaya ayrıldı ve gerekçeleri raporun §5'inde tek
tek yazıldı: **ZATEN VAR** (~70: Heston/Merton/Hawkes/LSMC · SVI/SSVI/SABR/Dupire ·
HAR-RV/VPIN/Kyle λ · HRP/BL/CVaR-LP · BOCPD/kopula/frac-diff/parçacık filtresi · RMT/MST/PCA
· DSR/purged-CV/PBO · Omega/Ulcer · §1256 · Vortex · SHAP/IG) · **VERİ YOK** (~150: L2/L3
emir defteri → Lee-Ready/efektif-makas/mikro-fiyat/Avellaneda-Stoikov/kuyruk önceliği;
MOO/MOC; ThetaData/Polygon/Sharadar/Ortex/Finnhub; uydu-patent-Glassdoor; 13F; kripto) ·
**ANAYASAYA AYKIRI** (~40: FIX/OMS/SOR/dark-pool/15c3-5/CAT — bu sistem EMİR GÖNDERMEZ) ·
**DONANIM** (~40: FPGA/DPDK/isolcpus/PTP/kuantum/VR/NY4). Madde 41 (OneDrive'dan çık)
Tur-64'ün ölçümüyle YENİDEN reddedildi. **Hiçbir aile kenar iddiası taşımaz** (ACI DERS #5).
Rapor: `docs/TUR73_RAPORU_2026-09-06.md`. **3479 → 3560 test**, ruff/mypy temiz.

- **§A PORTFÖY TAHSİSİ (Madde 116/117/118/119/120/281/283) — `nihai/ai/portfolio.py`.**
  `max_diversification_weights` + `diversification_ratio` (Choueifaty-Coignard; ρ=1'de DR
  **tam 1,0**, ρ=0'da kapalı form ters-vol ile fark **1,2e-08** ve DR=**√3**) ·
  `nco_weights` (López de Prado; HRP ile AYNI ağaç) · `vol_target_scale` (ASİMETRİK:
  `max_leverage=1` ile hedefin altında BÜYÜTMEZ) · `cdar_lp_weights`/`cdar`
  (Chekhlov-Uryasev LP; koşan tepe iki eşitsizlikle ALTTAN sınırlanır, tamsayı değişkeni
  YOK) · `cppi_allocation` · `bayesian_kelly` · `liquidity_adjusted_var`.
  **`nco` · `max_div` · `cdar_lp` doğrudan `compare_methods` tablosuna girdi** (7 → **10
  yöntem**) — ayrı bir yüzey açmak "tek yönteme körlük yok" kuralını ikiye bölerdi.
  **NCO'nun asıl iddiası ÖLÇÜLDÜ** (tahmini kovaryansla çöz, GERÇEK kovaryansta değerlendir;
  T=40/N=20 gürültü rejimi, 300 tekrar): **268/300** tekrarda MinVar'ı yendi, ortalama
  varyans **−%7,16** (0,009457 → 0,008780; oracle 0,008002 → boşluğun ~%46'sı kapandı),
  dağılım std 0,000874 → **0,000515**. TAM blok-köşegen kovaryansta ise ayrıştırma EXACT'tır:
  max|NCO − MinVar| = **2,2e-16**.
  **CDaR ≠ CVaR ölçüldü:** uzun çekilme dayatılmış 2-varlık panelinde CDaR-LP'nin CDaR'ı
  **0,0393**, CVaR-LP'ninki 0,0430, eşit-ağırlık 0,2346 — ve ağırlıklar da ayrışıyor.
  **CPPI'nin KAYAN TABAN TUZAĞI raporlanıyor:** `floor_pct` güncel sermayeden türetilirse
  taban sermayeyle DÜŞER ve riskli oran her düzeyde **sabit 0,3000** kalır = sigorta
  anlamını yitirir; `floor_value` (mutlak taban) ile 0,300 → 0,158 → 0,065 → 0. Bayrak
  `taban_kayan` bunu AÇIKÇA söyler.
- **BAYES KELLY'NİN BULGUSU (Madde 119) — "Bayesyen olmak tek başına ihtiyatlı olmak
  DEĞİLDİR".** İkili bahiste beklenen log-büyüme `p`'de DOĞRUSALDIR
  (E_p[p·ln(1+fb) + (1−p)·ln(1−f)] = E[p]·ln(1+fb) + (1−E[p])·ln(1−f)), dolayısıyla
  posterior üzerinden ORTALAMA ALMAK Kelly'yi HİÇ küçültmez: Bayes-optimal kesir, posterior
  ORTALAMASINDAKİ klasik Kelly'nin ta kendisidir. Sayısal maksimizasyonla doğrulandı
  (posterior-ağırlıklı E[log] `quad` ile doğrudan maksimize edildi): n=10 → fark **1,3e-07**,
  n=100 → 1,8e-05, n=1000 → 4,1e-05. Belirsizlik indirimi ORTALAMADAN değil **KUANTİLDEN**
  gelir (`f_robust`, p'nin alt kuantilinde) ve `kesinti` n ile eriyor: 0,1667 → 0,1612 →
  **0,0511**. Bu, "robust Kelly" diye ortalama alan her uygulamanın sessiz hatasıdır.
- **§B WHITE REALITY CHECK + HANSEN SPA (Madde 177) — `research/deflated.py`.**
  DSR çoklu-testi ANALİTİK ve normallik varsayarak deflate eder; RC/SPA aynı işi
  **bootstrap ile ve bağımlılık yapısını KORUYARAK** yapar. `stationary_bootstrap_indices`
  (Politis-Romano; ardışıklık oranı 0,8001 → ortalama blok **5,00**, hedef 5,0),
  `reality_check`, `hansen_spa` (üç merkezleme: lower/consistent/upper).
  **Hansen'in düzeltmesi ÖLÇÜLDÜ** (T=500, marjinal üstün aday t≈2,0, 40 koşu ortalaması;
  K = eklenen AÇIKÇA DEĞERSİZ aday sayısı): RC p **0,1829 → 0,4330 → 0,5865 → 0,7086**
  (K=0/20/60/150) iken SPA_c **basamak basamak 0,1744'te SABİT** kaldı; SPA_u (White
  merkezlemesi) RC'yi takip etti → mekanizma studentizasyon değil **YENİDEN MERKEZLEMEDİR**.
  Boyut kalibrasyonu (H0, 200 koşu, α=0,05): RC **0,045** · SPA_c **0,060**.
  **Okuma: bir aday havuzuna değersiz aday eklemek RC'yi sulandırır; edge_lab defterinin
  çoklu-test sayacı büyüdükçe SPA tercih edilmelidir.**
- **§C PERFORMANS KARNESİ (Madde 249/250).** `time_underwater` (max/ortalama/mevcut sualtı
  süresi + oran) · `gain_to_pain` · `tail_ratio`. Omega ve `ulcer_index` ZATEN vardı;
  eksik olan SÜRE ve ASİMETRİ eksenleriydi. `/api/quant/risk` oran bloğuna bağlandı.
- **§D EKONOFİZİK (Madde 203/204/317).** `shannon_entropy`/`renyi_entropy`/`tsallis_entropy`
  (ortak `_hist_probs`; α→1 ve q→1 Shannon LİMİTİNE düşer — düzgün dağılımda üçü de
  **ln 20 = 2,9957322736**) · `critical_slowing_down` · `mfdfa` (Kantelhardt 2002).
  **MFDFA ayırt ediyor** (n=2^14, aynı uzunluk): beyaz gürültü h₂=**0,5157**/genişlik
  **0,0437**, rastgele yürüyüş h₂=1,5485 (DFA'nın integre seride beklenen ≈1,5'i),
  iki-terimli çarpımsal kaskad (p=0,3) genişlik **1,2218 = 28 katı**.
- **§E GETİRİ EĞRİSİ (Madde 145/211/212) — 13. quant modeli.**
  `fred_data.treasury_curve()` (11 sabit vade, **ANAHTARSIZ**, ileri doldurma YOK) ·
  `curve_spreads()` (T10Y2Y · T10Y3M · T5YIE · T10YIE — farklar FRED'in KENDİ serisinden
  okunur; elle çıkarmak iki bacağın farklı günlerde güncellenmesinden hayalet inversiyon
  doğurur) · `quant_math.nss_fit` (τ ızgarasında KAPALI-FORM doğrusal EKK — NSS τ sabitken
  β'larda doğrusaldır, nonlineer optimizasyona gerek YOK) · **`/api/quant/curve`** +
  pano paneli (eğri+NSS · faktör serisi · farklar · başabaş). Canlı ölçüm (2026-09-03):
  1741 gün × 11 vade, kırpılma YOK; eğri 3,83 → 5,25; T10Y2Y +0,41 · T10Y3M +0,87;
  başabaş 5Y %2,37 · 10Y %2,35.
- **§F KALINTI MOMENTUM ve KALDIRAÇLI ETF (Madde 68/284).** `residual_momentum`
  (Blitz-Hanauer; faktörleri ÇAĞIRAN verir — `statarb.eigenportfolio_returns` zaten faktör
  üretiyor, **yeni veri kaynağı gerekmedi**) · `leveraged_etf_drag`.
- **§G İMA EDİLEN KORELASYON (Madde 279) — `options_math.implied_correlation`.**
  Portföy varyansı kimliğinin tek-ρ çözümü; payda O(n) hesaplanır (n² çift döngüsü YOK).
  **Üretilen ρ makine hassasiyetinde geri çıkarılıyor:** ρ ∈ {0 · 0,35 · 0,90 · 1,00} için
  hata ≤ **6,7e-16**. [−1,1] dışına çıkan değer **KIRPILMAZ**, `gecerli=False` ile raporlanır
  (tutarsız girdi kümesinin işareti; ρ=±1 sınırı için 1e-9 tolerans). `/api/quant/portfolio`
  aynı kimliği ÖLÇÜLEN vol'lerle koşup gerçekleşen ortalama korelasyonu basıyor.

- **BULGU-73-1 — `alpha` paket içinde İKİ ZIT ANLAMDA kullanılıyordu (KAÇAN BİR MUTASYON
  ORTAYA ÇIKARDI).** `quant_math.var_historical`/`cvar_historical`/`evt_gpd_var_es` alpha'yı
  KUYRUK OLASILIĞI sayar (0,05 = en kötü %5). Tur-16'nın `portfolio.cvar_lp_weights`'i ise
  Rockafellar-Uryasev'in GÜVEN DÜZEYİ biçimini kullanıyordu (payda **(1−α)·T**) → paketin
  varsayılanı `_DEFAULT_CVAR_ALPHA=0,05` ile en kötü **%95**'i ortalıyor, yani pratikte
  ORTALAMA KAYBI minimize ediyordu. Yöntem tablosundaki "CVaR-LP" satırı adının söylediği
  şeyi YAPMIYORDU. ÖLÇÜLDÜ (yavaş-kanama ↔ keskin-şok paneli, 201 noktalı kaba kuvvet
  ızgarasına karşı): eski payda w=[0,00; 1,00] verip CVaR@5%'i **0,017720**'de bırakıyordu;
  doğru payda (α·T) ile w=[0,7744; 0,2256] ve CVaR@5% **0,007225** (kaba kuvvet en iyisi
  0,007301 — LP sürekli uzayda daha iyisini buluyor). Tur-73'ün `cdar_lp_weights`'i bu
  konvansiyonu MİRAS ALMIŞTI ama ölçüm tarafı (`cdar()`) kuyruk-olasılığı okuyordu → aynı
  modülün iki fonksiyonu birbiriyle çelişiyordu. İkisi de `alpha·T`'ye çevrildi.
  **Kusuru bulan şey test değil, KAÇAN BİR MUTASYONDU:** CDaR LP'sinden koşan-tepe
  monotonluğunu silmek hiçbir testi kırmıyordu, çünkü test kurgusunda iki yöntem de AYNI
  köşe çözümünü veriyordu (w=[1,0]) ve "CDaR-LP daha düşük CDaR verir" iddiası bilgi
  taşımadan da doğruydu. Yeni kurgu ikisini AYIRIYOR (A yavaş kanar = kötü ÇEKİLME/iyi
  kuyruk; B keskin şok alır = kötü KUYRUK/iyi çekilme) ve **her yöntem KENDİ ölçütünde
  kazanıyor**: CDaR 0,0254 ↔ 0,2008 · CVaR 0,0168 ↔ 0,0072.

**TURUN DÖRT TEŞHİSİ (aşağıdaki derslere geçti):**
  1) `critical_slowing_down` ÖRTÜŞEN pencerede durağan gürültüde **%19,5 yanlış-pozitif**
     veriyordu (nominal %5) — varsayılan `step=window` (örtüşmeyen) oldu, güç kaybı YOK.
  2) `nss_fit`'in salt-SSR kazananı canlı Hazine eğrisinde günlerin **%98'inde** β0'ı
     gözlenen getiri aralığının DIŞINA atıyordu (β0 = −2,08, 30Y %5,25 iken).
  3) `leveraged_etf_drag`'in aritmetik sürüklenmesi Cheng-Madhavan yaklaşımıyla **14 kat**
     ayrışıyordu — ikisi FARKLI UZAYDA; `log_suruklenme` eklendi ve örtüşme ±%5.
  4) **BULGU-73-1**: `cvar_lp_weights` α'yı GÜVEN DÜZEYİ sayıyordu → varsayılan 0,05'te
     ortalama kaybı minimize ediyordu (yukarıda).

**GÖLGE yön hipotezi `dir_bar`/`conf_bar` (Tur-20):** yön/güven her barda iki kez
hesaplanır — yürürlükteki `direction` canlı fiyattan (DEĞİŞMEDİ), gölge varyant karar
barının KAPANIŞINDAN. Neden gölge: Tur-19 ölçümü giriş FİYATININ bar kapanışı olmasının
isabeti 0,4654→0,4937 yaptığını gösterdi ama YÖN girdisini değiştirmenin etkisi
ÖLÇÜLMEDİ (ayrı hipotez). `SIGNAL_ON_BAR_CLOSE=true` yürürlüğe alır — varsayılan KAPALI,
canlı karşılaştırma kanıtı birikene kadar açılmamalı. **Tur-83: kanıt birikti ve AYNI referanstan
ölçüldü → varsayılan AÇIK** (bar-yön +1,92 puan, t=+12,8; BULGU-83-1); karşı varyant `dir_live`da.

**BÖLÜM 9 AÇILDI + TUR-75'İN AÇIK KALEMLERİ KAPANDI (Tur-76, 2026-09-07).**
Tur-75, 255 maddelik listenin triyajını yapıp en yüksek getirili tek işi adıyla yazmıştı:
*"BÖLÜM 9'un 23 maddesi 'veri yok' DEĞİL, 'köprü yazılmadı'dır"*. Bu tur o köprüyü yazdı
ve Tur-75'in beş açık kaleminden dördünü kapattı. **21 madde uygulandı**, hiçbiri bir
kenar iddiası taşımaz. Rapor: `docs/TUR76_RAPORU_2026-09-07.md`. **3983 → 4076 test**,
ruff/mypy temiz (199 dosya).

- **`nihai/data/edgar_facts.py` (YENİ) — SEC EDGAR XBRL `companyfacts`, ANAHTARSIZ.**
  Ölçüldü: AAPL için 200 · 3,79 MB · 1,1 sn. Üç sözleşme: (1) **`filed` BİLİNME
  tarihidir** — her okuma `filed <= as_of` (doğrulandı: `as_of=2024-01-01`'de cari dönem
  FY2023'tür); yeniden beyanlar da bu kuralla çözülür. (2) Her kanonik kalem bir **ETİKET
  ZİNCİRİdir** ve zincir **DÖNEM BAŞINA** uygulanır (BULGU-76-1). (3) Önbellek ham değil
  **DAMITILMIŞ** tutulur (ham JSON 3,8 MB/şirket × 500 sembol ≈ 1,9 GB ve `data/`
  OneDrive'a senkronlanır). Olay dosyalamaları serbest metin ayrıştırmadan gelir:
  `submissions` JSON'unun `items` kolonu 8-K madde kodlarını taşır (ölçüldü, AAPL:
  **Item 5.02 → 12**, Item 4.02 → 0, **SC 13G → 14**) → 180·181·182 "veri yok" DEĞİLDİ.
- **`nihai/ai/fundamentals.py` (YENİ) — 15 temel skor** (Piotroski F · Altman Z/Z′ ·
  Beneish M · Mohanram G · Sloan tahakkuku · DuPont · nakit dönüşümü · şerefiye ·
  emeklilik açığı · Ar-Ge · nakit ömrü · hissedar getirisi · FCF↔temettü · kiralama ·
  E/P-ERP). Canlı ölçüm (AAPL FY2025, mcap 4,73 T$): **F 8/9 · Z 12,20 · M −2,29 ·
  ROE 1,7142 (çarpım kimliği doğrulandı) · CCC −71,8 gün · hissedar getirisi %3,03**.
  Üç kural koda yazılı: **eksik girdi UYDURULMAZ** (skor `None`, 0 değil; `n_sinyal`
  ayrıca döner), **her skor kendi VARYANTINI söyler** (Z ↔ Z′ eşikleri farklıdır),
  **kesitsel ölçüt ÇAĞIRANDAN gelir** (Mohanram medyanı uydurulmaz).
- **`nihai/data/factors.py` (YENİ) — Ken French faktörleri (224/221), ANAHTARSIZ.**
  15.876 gün, 1963-07-01 → **2026-07-31** (seri BİR AY GECİKMELİ ve bu `attrs`ta taşınır).
  `factor_attribution` **Newey-West (HAC)** standart hatası kullanır — günlük getiriler
  bağımsız değildir ve EKK alfanın t'sini şişirir. Doğrulamalar ÖLÇÜLDÜ: kendine
  regresyon β **1,000000** / α **2,7e-19** / R² **1,0**; bilinen DGP α 0,000441 ↔ 0,000500,
  β 1,3047 ↔ 1,30; SIFIR alfa kontrolü t **−0,03**. **Canlı ölçüm (AAPL+MSFT+SPY eşit
  ağırlık, 374 gün): yıllık alfa +%7,55 ama t=1,15 → AYIRT EDİLEMEDİ**, piyasa betası
  1,06, R² 0,79. Yani sepetin getirisi büyük ölçüde piyasa betasıdır.
- **Üretime BAĞLANDI (Tur-75 yazmış, çağıranı yoktu).** (a) **Madde 244 karar zinciri**:
  `log_signal(accepted=True)` ve kapanış otopsisi `data/audit_chain.jsonl`e halka ekler,
  bütünlük `/api/system/health` → `audit_chain` alanında OKUNUR; **her karar satırı
  YAZILMAZ** (tek günlük `decision_log` 35 MB — Tur-58). `AUDIT_CHAIN=off` kapatır.
  (b) **Hacim profili + CUSUM** `observatory.bar_shadow`a girdi (Madde 069/070/095/105);
  maliyet aynı süreçte A/B ölçüldü: **17,39 → 19,04 ms (+1,65 ms, 1,09×)**, eski
  anahtarların hepsi duruyor. Üç kolon (`poc_dist_bps`·`value_area_pos`·`cvd_divergence`)
  deftere ve **gölge terfi defterine** girdi → ayrık yüzey 10 → **13**, Bonferroni çıtası
  herkes için yükseldi (`hib=None`: yönleri ön-kayıtlı DEĞİL).
  (c) **Makro paneli** `/api/quant/macro` + pano (14. quant modeli); uç ÖNBELLEK-YALNIZ
  okur, doldurmak ayrı adımdır (`python -m nihai.data.macro_regime`). Canlı: 4229 gün,
  16 göstergenin 16'sı ölçülü. **`ml_hygiene` BİLEREK bağlanmadı** — o bir ARAŞTIRMA
  aracıdır (etiketleme/CPCV), bar başına çalıştırılacak bir ölçüm değil.
- **Madde 045 `ultima`** (∂³V/∂σ³) `bs_greeks_extended`e eklendi ve sonlu farkla
  kilitlendi (bağıl fark **3,9e-09**, üç strike'ta); uçta ve panoda YAYINDA.
  **Madde 049 Feller koşulu** (`2κθ > ξ²`) `heston_calibrate` çıktısında **ÖLÇÜLÜR,
  DAYATILMAZ** — kısıt olarak dayatmak, piyasanın gerçekten Feller-dışı fiyatladığı bir
  yüzeyi zorla içeri çekip uyum hatasını gizlerdi (Tur-49 dersi).
- **BULGU-76-7 — saat kayması KAPALI BİR HİZMETTİ, unutulmuş bir komut değil.** Tur-75
  −650 ms ölçüp *"operatörde `w32tm /resync`"* demişti; kayma **−965 → −980,5 ms**'ye
  BÜYÜDÜ. Kök neden: `W32Time` **Stopped/Manual** ve zaman kaynağı yerel CMOS saati —
  tek seferlik resync bu oturumu düzeltir, SORUNU düzeltmezdi. `saat-onar.ps1` hizmeti
  **Automatic** yapar, başlatır, kaynak yerelse NTP havuzunu yazar, senkron eder ve
  doğrulamayı projenin KENDİ aracıyla ÖNCE/SONRA ölçer. **Ölçüldü: −980,5 ms → +21,8 /
  +5,9 / −11,7 ms** (eşik ±200 ms). Ders: tekrarlanan bir ölçüm KÖTÜLEŞİYORSA önerilen
  onarım ile kök neden ayrışıyordur.
- **Pano:** iki YENİ quant modeli (14. Makro Rejim, 15. Şirket Temelleri; 8 kanvas,
  20 ölçüm kartı) + Portföy paneline **faktör ayrıştırma** kutusu + Opsiyon paneline
  **ultima**. `null` ÇİZİLMEZ; ölçülemeyen kart NEDENİNİ yazar.
- **ANAYASA:** `fundamentals`·`edgar_facts`·`macro_regime`·`data.factors` importu
  `signals`/`risk`/`execution` altında TESTLE YASAK. `edge_lab` `n_trials` ARTMADI
  (ölçüm yüzeyi eklemek arama değildir — Tur-55).

**"MEVCUT MODÜL GELİŞTİRME" LİSTESİNİN KALANI — ölçülerek uygulandı, iki kural ölçülerek REDDEDİLDİ
(Tur-77, 2026-09-14).** Önceki ajanın commitsiz 19 maddesi yerinde doğrulandı (süit 4131/2 kırmızı →
MainHub sürüklenmesi `tools.sync_mainhub --apply` ile kapandı); kalan 8 madde Claude + Codex (gpt-6-astra) +
Gemini ile yapıldı. Rapor: `docs/TUR77_RAPORU_2026-09-14.md`. **4134 → 4276 test**, ruff/`mypy .` temiz,
mutasyon 12/12.

- **4.5 U-eğrisi** `microstructure.intraday_profile/intraday_anomaly/liquidity_withdrawal` → `/api/quant/micro`
  `ucurve` + pano. Saat dilimi TAHMİN EDİLMEZ (Alpaca/hub naive-UTC ↔ yfinance tz-farkında ET). OHLCV'de
  bid/ask yok → makas/çekilme ölçülmez.
- **5.1 ayrışma** `neural.decompose_uncertainty` (p(1−p) + BALD) · `predict_uncertainty_decomposition` ·
  `epistemic_flag`. **"Epistemik yüksekse ABSTAIN" REDDEDİLDİ:** canlı 1m, 5 tohum — epistemik ölçüler isabeti
  TERS sıralıyor (topluluk üye-arası 0/5, karşılıklı bilgi 0/5); kural en isabetli dilimi atardı. Belgelerdeki
  "üye-içi MC-dropout = aleatorik-benzeri" ifadesi YANLIŞTI (dropout = epistemik).
- **5.2** `evidential_belief`/`predict_belief` (Σb+u=1); canlı artefaktta `dir_evid` başlığı yok.
- **5.3 Venn–Abers** `adaptive_calibration` (varsayılan yöntem kümesinde). Canlı pwin tabanı YENEMEDİ (t=+0,80).
  **"Aralık 0,5'i kapsıyorsa nötrle" REDDEDİLDİ:** genişlik bilgiyi değil kalibrasyon yoğunluğunu ölçer
  (n=100→0,10 · n=3000→0,009); canlıda 0,5'i kapsayan pay %0,25.
- **5.4 K-NN anı** `episodic.analog_ozeti` + `/api/ai/episodes/similar` + pano. **BULGU-77-1:** epizot defterinin
  %44,4'ü kopyaydı (1.000 çözümde 210 benzersiz karar barı; gölge koşucu aynı günlük barı gün içinde yeniden
  yazıyordu) → yazmada `append_episode(..., ayni_gun_tekil=True)` (yalnız `decision._record_shadow`), okumada
  karar barı başına tekil (taban havuzu 975 → 203).
- **6.2 HIFO** `tax_report.build(lot_method=...)`: 23 çakışan satış ölçüldü, bu defterde vergi etkisi 0.
- **6.3 Replay** `nihai/execution/replay.py` — `SimulatedClock` + `ReplayProvider` + izole alt süreç +
  `sys.addaudithook` yazma çiti; kapanış defteri canlı `_apply_close_bookkeeping`'in kendisi. Gerçek koşu
  (2026-09-11, AAPL+MSFT) 18 sn, üretim dosyaları değişmedi.
- **6.4 pano** tek betik → kayıt bloğu + 9 modül + başlatma bloğu (yürütülen 5 deyim en sonda, tek tek
  try/catch; eksik modül şeritte adıyla). Karar hunisi `signal_engine.karar_hunisi` (yazıcıyla aynı modül) +
  `/api/decision/funnel` + kart. **Pano JS'ine yeni kod eklerken ilgili `[pano modülü …]` bloğuna yaz; yeni
  modül açarsan başlatma bloğundaki `beklenen` listesine ekle** (`tests/test_tur77_pano_modul.py` kilitler).

**"MEVCUT MODÜL GELİŞTİRME" LİSTESİNİN BAĞIMSIZ DENETİMİ (Tur-78, 2026-09-14→15).** Tur-77'nin açık kalemi
("önceki ajanın 19 maddesinin sayısal iddiaları yeniden ölçülmedi") kapandı: Codex R1/R2/R3 salt-okunur denetimi
(17 bulgu, her biri çalıştırılabilir kanıt betiğiyle) + Gemini temiz-oda fark testi + Claude hakemliği. Rapor:
`docs/TUR78_RAPORU_2026-09-15.md`. **4362 → 4394 test**, ruff/`mypy .` temiz, mutasyon 26/26. Kenar iddiası YOK.

- **BULGU-78-1 — backtest aynı-bar yeniden girişi look-ahead'di.** Bar i+1 İÇİNDE kapanan pozisyonun sembolüne
  bar i kapanış kararıyla aynı iterasyonda giriliyordu. A/B: 1d 20 sembolde 124 giriş engellendi, sembol başı
  ortalama getiri **+%1,56 → +%0,40**. `REENTRY_POLICY=next_bar` (varsayılan; `same_bar` yalnız A/B), üç döngüde.
- **Opsiyon ailesi (Codex O1–O5 + inceleme):** eksik OI/gamma tarafı artık net GEX işareti UYDURMAZ (`None` +
  `n_oi_eksik`/`n_gamma_eksik`) · vanna/charm NaN OI → `None` · Dupire eşit olmayan aralık türevi (yerel varyans
  0,15 → 0,08) · Dolt gün içi damga reddi · dolum yalnız karar gününden SONRAKİ NYSE seansında · koni ATM IV'si
  yalnız spotu çevreleyen strike'lardan (eskiden 70 strike'ın IV'si "ATM" sayılıp yedi pencerede yüzdelik 100).
- **Portföy:** log getiri `annualize_return(..., log=True)` (günlük 0,005'te −1,10 pp) · MC GBM μ = m·ppy + ½σ²
  (İtô iki kez düşüyordu) · etkin sınır tavanı normalizasyonla delinmez · ERC konveks Spinu Newton'u (SLSQP 400
  kovaryansın 149'unda ağırlığı 0'a yapıştırıyordu → HERC `None`) · **CVaR risk paritesi kink'te minimaks LP**
  (300 problemin 40'ında çözüm varken `None`; artık 0) + `kink` bayrağı: optimum VaR bağındayken Euler katkısı
  TEK DEĞİLDİR, `shares` pariteye en yakın geçerli alt-türevdir (pano bunu yazar).
- **Savunmacı sözleşme:** `implementation_shortfall` geçersiz girdide `None` (canlı `_close` muhasebesinde pozisyon
  düşülmeden ÖNCE çağrılıyordu; teşhis kapanışı durduramaz), `cost_summary` sonlu olmayanı saymaz, CPPI/`warm`/
  `approval_events` istisna atmaz. `SlippageModel` geçersiz dolum fiyatında istisnayı KORUR (NaN özkaynağa girmez).
  VPIN/BVC `causal=False` taşır. SEC `Retry-After` HTTP tarihi okunur (89,9 sn erken istek). Mohanram tahakkuku
  doğrudan CFO > NI.
- **1.3a PDUFA ileri takvimi** `fda_calendar.pdufa_calendar` (SEC EDGAR tam-metin 8-K/6-K, anahtarsız; tarih ŞİRKET
  BEYANIDIR) + laboratuvar aracı. Ayrıştırıcı: kelime sınırlı "date", şirket kısaltması cümle sonu değil, tarihe
  BİTİŞİK etkinlik ("conference on <tarih>") reddedilir.

**YETİM ÖLÇÜM MODÜLLERİNİN PANO KÖPRÜSÜ + §4.1 PİYASA ÖNSELİ (Tur-79, 2026-09-16).**
Çalışma ağacında commit edilmemiş DÖRT modül vardı (`ai/drift_monitor` §12.3 SR 11-7 ·
`ai/econometrics` §6.1/6.2/6.5 · `ai/tda` §10.1 · `data/etf_holdings` §7.5), 258 testi
yeşildi ve **üretim çağıranı YOKTU**. Bu tur köprüyü kurdu; köprü kurulur kurulmaz üç
bulgu çıktı. Rapor: `docs/TUR79_RAPORU_2026-09-16.md`. Kenar iddiası YOK.
Dış katkı: Codex (gpt-6-astra) salt-okunur triyaj — üç iddiasının ÜÇÜ de koda karşı
doğrulandı. **Gemini CLI kullanılamadı** (`IneligibleTierError`, hesap katmanı
desteklenmiyor) → bu turda ikinci bağımsız denetçi YOKTU.

- **BULGU-79-1 — sürüklenme modülü YAZILDIĞI LOGU OKUYAMIYORDU.**
  `load_live_features` *"tz-aware reddedilir"* diyordu; üretim logunun (27.260 satır)
  **%37,2'si tz-aware** ve 30 sembolün 30'u da karışık (günlük oran %9→%50, kesme yok) →
  `format="mixed"` karışık tz'de `ValueError` atıyor, fonksiyon `None` dönüyor, modül
  üretimde HİÇ okuyamıyordu. Depoda bu sorunun DÖRT kopyalı kanonik cevabı vardı
  (`prediction_tracker._canon_ts` · `calibration_loop._parse_ts_utc` · `ai/memory` ·
  `ai/decision`): *tz-aware → UTC, tz düş; naive → ZATEN UTC*. Naive satırların gerçekten
  UTC olduğu VARSAYILMADI, ölçüldü: iki grup da aynı UTC seans bandında (13-20) ve
  `logged_at − bar_ts` medyanı **10.908 ↔ 10.889 sn** (p05/p95 örtüşüyor). Modül kanonik
  sözleşmeye çekildi; `since`/`until` sınırları hâlâ naive olmak ZORUNDA (sınırın tz'si
  ÇAĞIRANIN kararıdır).
- **BULGU-79-2 — canlı özellik logunda İKİ POPÜLASYON var; karıştırmak PSI'yi DÖRDE
  katlıyor.** `quality_score == 0` ⟺ `rr_ratio == 0` ⟺ aday-DIŞI bar (örtüşme
  16.066/16.066 TAM) ve payı **Temmuz %0 · Ağustos %70,2 · Eylül %84,9** (kök neden
  BULGU-59-1'in `FEATURE_LOG_ALL` politikası — bilinçli bir iyileştirme, ama iki dönemin
  satır kümesi artık AYNI ŞEY DEĞİL). Aynı veri, aynı pencere, aynı fonksiyon:
  **karışık popülasyonda 12 "BÜYÜK sürüklenme", homojen (`aday`) popülasyonda 3** —
  `rr_ratio` (PSI 8,82) ve `quality_score` (2,85) listeden TAMAMEN düşüyor.
  `/api/quant/drift` `populasyon` parametresi alır (varsayılan `aday`) ve
  `populasyon_karnesi`'ni SEÇİMDEN BAĞIMSIZ raporlar. Ayraç `cand_side` DEĞİL
  `quality_score > 0`: `cand_side` Tur-59'da doğdu, öncesinde BOŞTUR ve onunla süzmek
  referans penceresini SIFIRA indiriyordu. Yan bulgu: `vix == 0` (imkânsız) 451 satır,
  tek bir 2 saatlik pencerede — `meta_features._vix_or_neutral` BUGÜN doğru, kalıntı.
- **BULGU-79-3 (§4.1) — Black-Litterman'ın "piyasa ağırlığı" ters-volatiliteydi ve
  ÖNCÜLÜ BAYATLADI.** `_inverse_vol_weights` docstring'i dürüsttü (*"gerçek piyasa
  kapitalizasyonu verisi bu modülde YOK"*) ve yazıldığında doğruydu; `etf_holdings`
  doğunca YANLIŞ oldu. Proxy bir yaklaşıklık DEĞİL FARKLI BİR ÖNSELDİR — ölçüldü
  (8 SPY bileşeni): **maks |Δw| = 8,65 puan** (NVDA %15,3 → %23,9, XOM %6,9 → %3,2),
  L1 = 0,1885. `compare_methods(..., market_weights=…)` eklendi; **varsayılan `None` =
  eski davranış, BİT-DENK (maks |Δ| = 0,000e+00)**. `nihai/ai/portfolio` VERİ katmanını
  import ETMEZ — gerçek ağırlıklar uçtan geçer. Geçersiz vektör sessizce proxy'ye
  DÜŞMEZ (`ValueError`); panelin TEK sembolü bile sepette yoksa proxy'ye düşülür (eksik
  ismi ters-vol ile doldurmak iki önseli tek vektörde karıştırmak olurdu) ve kaynak
  `market_prior.kaynak` ile panoda BİR KARTTA yazılır.
- **Pano: dört yeni quant modeli (15 → 19), 13 grafik + 4 tablo.** Canlı ölçümler:
  DCC a+b = **0,9893** (korelasyon şoku neredeyse KALICI), LR (DCC↔CCC) = **306,7**
  (χ²(2) eşiği 5,99) · FIGARCH **d = 0,0 SINIRDA** (bu seride uzun hafıza YOK;
  `sinirda.d_sifir` taşınır — Tur-49) · MODWT varyansın **%52,3'ü** 1 barlık ölçekte,
  `causal:false` (döngüsel → canlı özellik OLAMAZ) · TDA H0=40 H1=10, `kesildi=0`,
  kayan pencere `causal:true`. Dördü de elle tetiklenen karttır (2,0-10,6 sn), 5 sn'lik
  ana döngüye maliyet EKLEMEZ.
- **Kilit: `tests/test_tur79_pano_uclari.py` (21 test).** Üç soru: BAĞLANDI MI (katalog ·
  panel kancaları · `QM_MODELS` · `QM_RENDER` · gönderici dalı + ANAYASA: dört modülün
  `signals`/`risk`/`execution` importu YASAK), FAIL-SOFT MU (sağlayıcı TAMAMEN ölüyken
  dördü de 200 + `reason`), SÖZLEŞME TUTUYOR MU (karışık-tz okunuyor · pencereler
  ÇAKIŞMIYOR · karışık popülasyon SAHTE sürüklenme üretiyor ve `aday` süzgeci eliyor ·
  `causal` bayrakları · numpy dizisi JSON'a serileşiyor · dispersiyon ucu AĞA ÇIKMIYOR
  (gözlem BAYRAKLA — Tur-74) · BL varsayılanı BİT-DENK). Mevcut kapılar bu turda İKİ
  gerçek kusur yakaladı: `nihai/ai/__init__` lazy listesi dizinle ayrıştı (BULGU-76-5)
  ve katalog formül referansı bilinmeyen modül öneki taşıdı (Tur-56).
- **AÇIK KALEMLER:** §2.2 Lévy ailesi (`carr_madan_fft_price` charfn ALMIYOR —
  `heston_params` alıyor, Codex doğruladı → FFT'yi charfn-parametrik yapmadan eklemek
  ölü köprü olurdu) · §7.3 `Cov(r)=B·Cov(f)·Bᵀ+Cov(e)` ayrıştırması (yeni veri GEREKMEZ) ·
  §4.6 OBPI koruyucu-put raporu · §8.4 Brinson **VERİ YOK** (SPY sepetinde sektör alanı
  504/504 `"-"` + PIT sepet yok) · dispersiyon serisinin PIT ağırlık açığı (SSGA geçmiş
  sepet yayımlamıyor) · Gemini denetçisi.

**CANLI SÜRECİN GERÇEK YÜKÜ — ÖLÇÜME DAYALI OPTİMİZASYON (Tur-80, 2026-09-16).** Rapor:
`docs/TUR80_RAPORU_2026-09-16.md`. Kenar iddiası YOK; karar yoluna bileşen eklenmedi.

- Canlı tarayıcı+pano süreci tek çekirdeğin **%45,7**'sini kullanıyordu. py-spy (`--nonblocking`, 180 sn):
  `/api/predictions/accuracy` aktif örneklerin **%42,8**'i (`resolve`+`summary` CSV döngüsü); OS thread
  CPU'sunun ~%35-40'ı py-spy'ın görmediği torch OpenMP thread'lerinde.
- Kazanımlar (hepsi denklik testli + mutasyonla sınandı; canlı süreçle aynı venv'de ölçüldü): accuracy
  döngüsü **2469 → 1625 ms CPU** (dosya baytları eşit) · nöral gölge MC-dropout çağrısı **84 → 11 ms CPU**
  (bit-denk; duvar 10,2 → 8,3 ms) · vergi raporu **747 → 132 ms** · `options_analyzer` importu
  **1666 → 667 ms** · opsiyon synth backtest 2000 barda **4,05 → 0,69 sn** (Codex uyguladı) · canlı
  döngünün `PORTFOLIO_LOCK` altındaki bar çekimi kilit DIŞINA alındı (`fetch_1m_bars` + `prefetched`;
  canlı ölçüm: sağlıklı hub'da 15 pozisyon için kilit altında 168,5 ms ağ, servis açılışından ~10 dk sonra
  tek çağrı 9,6 sn → artık kilit altında yalnız ~9 ms hesap).
- **BULGU-80-1** yarım yazılmış tek satır `resolve()`ı TÜMDEN düşürüyordu (gizli; bugün 0 bozuk satır) ·
  **BULGU-80-2** gözlemevi HMM'i sabit pencerede HİÇ yeniden fit edilmiyordu (300 yeni barda 1 fit) ·
  **BULGU-80-3** 8 thread'den eşzamanlı MC-dropout sonuçlarının **%97**'si seriden farklıydı (süreç-genel
  tohum + paylaşılan `train()/eval()`); kaydedilmiş `neural_pwin/std` geçmişi bu yarışın izini taşır ·
  **BULGU-80-4** aynı sembolün 1m (tahmin servisi) ve 1h (tarama köprüsü) barları TEK HMM önbellek anahtarını
  paylaşıyordu → bir aralığın getirisine fit edilen model ötekinin rejim olasılığında kullanılıyordu ·
  **BULGU-80-5** aynı sembolde iki hisse pozisyonu (küme sınırı 2) her 1m barı İKİ KEZ görüyordu
  (`bars_held` 2×; breakeven/trailing açıkken aynı barın taşıdığı stop aynı barda tetikleniyordu).
- Codex bağımsız incelemesi 7 kusur buldu (devir imzası yarışı, devrin bellekte kalması, `fromisoformat`
  fazlalığı, HMM sayacının pencere/gösterim bağımlılığı, model yüklemesi RNG yarışı, `HAS_VOLLIB` anlamı,
  test körlüğü); 7'si bağımsız yeniden üretildi, düzeltildi ve geri-alma mutasyonuyla kilitlendi.
- Gemini (`agy`, metin-yalnız) incelemesi 5 iddia: 2 geçerli, 3 yanlış pozitif; geçerli olanı doğrularken
  BULGU-80-4 çıktı.
- Kilitler: `tests/test_tur80_optimizasyon.py` (27) · `tests/test_tur80_opsiyon_backtest_onbellek.py` ·
  `tests/test_tur80_portfoy_kilidi.py` (8; değişiklik öncesi gövdeye karşı denklik, mutasyon 8/8).
- **Düzeltme:** raporun ilk sürümü "canlı süreç sistem Python'u" diyordu — YANLIŞ; watchdog temel
  `python.exe`'yi `__PYVENV_LAUNCHER__` ile başlatır, süreç venv paketlerini yükler. Sayılar venv'de yeniden ölçüldü.
- **Yeniden başlatma olayı (2026-09-17 20:14, seans açık):** `proje-ac.ps1` bayrağı servislerden ÖNCE
  kaldırıyordu → `predict-watchdog` 2 sn içinde kayıtçıyı başlattı, betiğin port kontrolü import aşamasındaki
  süreci göremedi → İKİ kayıtçı ~3 dk koştu. İkisi de ARE/HON'u ayrı ayrı kapattı: `state.json` tutarlı ama
  `data/trades.csv` (2 satır), `data/equity_curve.csv` (1 nokta), karar günlüğü ve `trade_log.txt` ÇİFT
  kayıt taşıyor; silme otomatik kipte reddedildi → operatör kararı. `audit_chain` bütün. Düzeltme:
  `proje-ac.ps1` bayrağı portlar açıldıktan SONRA kaldırır (bayrak varken watchdog'lar hiçbir şey yapmaz).
- **AÇIK:** yukarıdaki çift kayıtlar · opsiyon prim yolu (`update_all_with_quotes`) kilit altında ağ çağırır
  (`OPTION_MODE=stock`ta ölü dal) · IV geçmişi sembol başına çift okuma · `edgar_facts.annual_frame` çift
  kurulum · pytest-xdist yok.

**CANLI UÇLARIN GERÇEK MALİYETİ — kıyasın göremediği fan-out (Tur-81, 2026-09-17).** Rapor:
`docs/TUR81_RAPORU_2026-09-17.md`. Kenar iddiası YOK; karar yoluna bileşen eklenmedi.
Ortaklar: Codex (gpt-6-astra) salt-okunur triyaj · **Gemini 3.1 Pro `agy` ile** (`gemini` CLI
ÖLÜ: `IneligibleTierError`; çalışan yol `agy --mode plan --model gemini-3.1-pro-high --print=...`
ve prompt BAYRAĞA bağlanmalıdır).

- **Ölçüm canlı süreçte, 76 GET ucu ısındırılıp sırayla zamanlandı.** Tur-80'in pano kıyası yalnız
  UCUZ uçları yokluyordu; quant panelleri hiç ölçülmemişti: `/api/predictions/accuracy` **27,8-49,4 sn**,
  `/api/quant/tda` **24,8-37,6 sn**, `/api/quant/curve` **8,4-10,6 sn**. Tarama turu 87 → **100,75 sn**.
- **BULGU-81-1** — accuracy ucunun **%99'u 30 ardışık sağlayıcı çağrısıydı** (CSV okuma 0,189 sn;
  hub `/bars` 3 ms…6,9 sn, 30 sembol toplamı ≈41,5 sn). İkinci kat: web yolundaki TTL kapısı 20 sn,
  istek 43 sn → önbellek HİÇ isabet edemezdi. `resolve()` artık anahtarları ön geçişte toplayıp
  `ThreadPoolExecutor(RESOLVE_FETCH_WORKERS=8)` ile çeker; süzgeç döngününkiyle BİREBİR
  (`horizon` ayrıştırılamayan satır eskiden de fetch tetiklemiyordu) → çekilen anahtar KÜMESİ ve
  dönen sözlükler değişmez. **8,38 → 1,85 sn (4,52×)**, sayaç + yazılan defterin içeriği + çağrı
  sayısı eşit; gerçek hub'da 10,63/4,55 → 3,85/3,11 sn. `RESOLVE_FETCH_WORKERS=1` eski davranış.
  *Ölçülüp REDDEDİLEN:* "ufku dolmamış satırı atla" → 30 sembolün 30'unun da dolmuş satırı var → 0 kazanç.
- **BULGU-81-2** — `CachingProvider._cache` anahtarı `lookback_bars`ı İÇERİYORDU; aynı `(sembol,"1m")`
  için üç çağıran (tahmin servisi 220 · korelasyon 60 · resolve 400) ASLA aynı girdiyi paylaşmıyordu,
  400'lük çerçeve ötekilerin ÜST KÜMESİ olmasına rağmen. Artık anahtar `(sembol, interval)` ve girdide
  "kaç bar İSTENDİĞİ" saklanır; istenen ≥ yeni istek ise `tail(n)`. Canlı karışımda **3 → 2** upstream,
  ızgarada (3 lookback'in 6 sıralanışı) **18/18 → 11/18**; hepsi önbelleksiz referansla BİREBİR.
  Daha AZ bar istemek girdiyi DARALTMAZ. Sarmalayıcı hem tarayıcıda hem HUB'da kurulu → kazanç iki yerde.
- **BULGU-81-3** — `ruff check .` **`nihai/data/`yi HİÇ taramıyordu.** `exclude = ["data"]` gitignore
  biçimindedir ve ÇIPASIZ ad her derinlikteki `data/`yi eler: kapı 403 dosya görüyordu, **0'ı**
  nihai/data altındaydı (29 modül: tüm sağlayıcılar, `hub_server`, EDGAR/FRED/opsiyon köprüleri) ve
  orada 7 bulgu birikmişti (UP037×2 · I001×2 · F401 · F541 · **B905 `zip(strict=)`**). Desenler köke
  çipalandı (`"/data"`) → **432 dosya**, hepsi düzeltildi.
- **BULGU-81-4** — bir denklik testi ORTAMIN kod sayfasına bağlıydı (turdan ÖNCE de kırmızıydı):
  `python -c` alt süreci bir `__main__` GİRİŞ NOKTASI değildir → `ensure_utf8_stdout` hiç çalışmaz →
  loguru stderr'i cp1254'e düşer → `"bulunamadı"`nın `ı`sı U+FFFD olur. Alt sürece `PYTHONIOENCODING=utf-8`.
- **BULGU-81-5** — `/api/quant/curve`in **%98'i** 250 × `nss_fit`ti (izole: 3049 ms / 66 ms / 23 ms).
  `nss_fit` τ ızgarasında 240 tasarım matrisi kurar ve matris yalnız VADELERE + τ'ya bağlıdır →
  aynı 240 matris 250 kez, **60.000 kuruluş**. `_nss_design` (vade baytları, τ1, τ2) ile önbellekli
  ve salt-okunur: **3385 → 1489 ms (2,27×)**, 250/250 günde TÜM alanlar bit-denk.
- **BULGU-81-6** — `/api/quant/tda`: cProfile `int.bit_length`i **41.314.168**, `dict.get`i 41.315.568
  çağrıda gösterdi. Pivot sözlüğü KENAR-İNDEKSLİ listeye çevrildi (`0` = boş nöbetçi; saklanan sütun
  `while column`dan geçtiği için asla 0 olamaz) ve üçgen köşe listesi `n`e göre önbelleklendi
  (tavan n≤64 ≈1 MB; C(200,3) 31,5 MB'dır ve 7/24 süreçte tutulamaz). **9,04 → 6,21 sn (1,46×)**;
  kayan seri + 30-bulut ızgara imzası DEĞİŞMEDİ. Algoritma değişmedi.
- **Codex triyajı:** madde B `son_iki_yil(..., annual=…)` → `/api/quant/fundamentals` aynı yıllık
  tabloyu iki kez kuruyordu, **148,5 → 2,8 ms (52,5×)**; madde C `/api/logs` kuadratik tamponu
  parça-listesi + tek birleştirmeye geçti (durma koşulu ve UTF-8 `ignore` BİREBİR; 4 gerçek logda
  çıktı eşit, kazanç 1,07-1,23× ≈0,005 ms — düzeltilen SÜRE değil KARMAŞIKLIK).
- **Kilit:** `tests/test_tur81_optimizasyon.py` (**35 test**), **mutasyon 11/11**. Mutasyon koşucusu
  `-B` ile koşar ve HEDEFLİ ters değiştirmeyle geri alır (Tur-72 `.pyc` + Tur-74 `git checkout` dersleri).
- **AÇIK:** BULGU-81-1/2/5/6'nın CANLI karşılığı ancak servisler YENİDEN BAŞLATILDIKTAN sonra
  ölçülebilir (koşan süreçler Tur-80 kodunu taşıyor; yeniden başlatma operatör kararıdır) ·
  `/api/quant/tda` kalan ~17 sn gerçek matematiktir (O(n³) üçgen) · uç düzeyi sonuç önbelleği
  (`curve`/`drift`, Codex YENİ-2: 164 ms/istek) · Codex A (IV geçmişi çift okuma) · Codex D (opsiyon
  primi kilit altında; bugün 0 kazanç) · Codex YENİ-1 (beta hesabı kilit altında ağa çıkıyor —
  kaçırma sıklığı ÖLÇÜLMEDİ) · tarama turunun 87 → 100,75 sn artışı ayrıştırılmadı.

**CANLI VERİ YOLU — ölü akış, fazla geniş kesici, gereksiz zincir (Tur-82, 2026-09-18).** Rapor:
`docs/TUR82_RAPORU_2026-09-18.md`. Kenar iddiası YOK; karar yolunda kural değişmedi. Ortaklar: Codex (limite
takıldı, iki tespit) · Gemini (`agy`, Tur-81 + Tur-82 incelemeleri).

- **BULGU-82-1 — Alpaca IEX akışı HİÇ veri almıyordu.** 30 sembol × (bars+quotes+trades) istendi; Basic plan
  **trades+quotes TOPLAMINI 30 sembol-kanalla** sınırlar, **bars/updatedBars sayılmaz** (canlı hesapta ölçüldü:
  150 bars kabul, 30 quotes + 1 trades → 405). Sunucu isteği ATOMİK reddeder ve alpaca-py her abonelikte TÜM
  handler'ları yeniden gönderir → akış "bağlı ama 0 mesaj" kaldı. 405 yalnız stdlib log'una (stderr) düşüyordu;
  Tur-80 onu görüp `/health`'teki `active: alpaca_stream`i (SAĞLAYICI sayacı, REST dahil) akış sanmıştı.
  Onarım: `_channel_plan` (bars hepsi · quotes bütçe kadar · trades artandan) + `_dispatch` yaması →
  `stream_status()` `server_subscriptions` (sunucu ONAYI) ve `last_error` taşır, `hub.log`a yazar.
- **BULGU-82-2 — kesici okuma zaman aşımında da açılıyordu** (docstring "yalnız bağlantı"). 17 Eylül (155 dk):
  4447 zaman aşımı (28,7/dk) → 8439 atlanan çağrı → 12.873 yfinance yedeği (83/dk); pozisyon stop kontrolünün
  1m barları dahil. Artık yalnız `ConnectError`/`ConnectTimeout`/`ConnectionError`.
- **BULGU-82-3 — zincir her taramada iki kez çekiliyordu** (~6 Yahoo isteği/sembol); 6209 WAIT'in HİÇBİRİ
  zincirin girdiği tek yere (kalite adımı) ulaşmadı. `generate_signal(chain_eval=...)` zinciri yalnız o adımda
  ister; `scan_symbol` kotasyonu yalnız eyleme dönük sinyalde çeker → kararlar BİREBİR (240 çerçeve × 4 zincir
  durumu × 2 IV kipi). Değişen tek anlam IV geçmişi örneklemesi: WAIT'te en fazla `IV_COLLECT_MIN`=15 dk'da bir
  (17 Eylül replay'i: günün son değeri 13/510 sembolde değişir, ort. 0,03 puan).
- Ek: pozisyon 1m bar çekimi havuzda · **Tur-81 `.env.example` beyanını unutmuştu** (`RESOLVE_FETCH_WORKERS`
  → Tur-61 kapısı kırmızıydı) · MainHub Tur-81'den beri sürüklenmişti (`tools.sync_mainhub --apply`).
- **BULGU-82-6 — düzeltmenin açtığı gerileme, ölçülüp kapatıldı.** Eski kesici hub'ın 429 uykusunu
  GİZLİYORDU (her tarama işçisinin ilk zaman aşımı onu yfinance'e yolluyordu). Kesici daralınca açılış
  taraması (16 süreç × 516 sembol × 1h/982) 53-77 sn → **221 sn** ve 208 istemci zaman aşımı. Kök: alpaca-py
  429'da 3 sn × 3 uyur (kurucu açmaz) → `client._retry = 0` (alan adı testle kilitli): **86 sn, 0 zaman aşımı**,
  429 aynı istekte hub içindeki yfinance yedeğine düşer.
- **BULGU-82-7 — akış canlanınca devreye giren gizli birleştirme kusuru:** çakışan gün-içi barda canlı toplam
  seed'i eziyordu → WS kopması sonrası eksik dakikalı saat tamponda (2000 dk) kaldığı sürece REST'in tam barını
  ezerdi. Artık seed'in KAPANMIŞ barları otoritedir; canlı yalnız seed'den sonrasını ekler (1d değişmedi).
- **Dağıtıldı** (kullanıcı onayı, piyasa kapalı): 13:58 + 14:04, her seferinde tek hub + tek kayıtçı; akış
  sunucu onayı **bars 30 · quotes 30 · trades 0**, 405 yok.
- **Kilit:** `tests/test_tur82_canli_veri_yolu.py` (62 test), **mutasyon 16/16**; tam paket 4828/0.
- **AÇIK:** 1m seed TTL'i 60 sn (akış canlıyken ~30 REST/dk) · bars sınırsız → geniş katmana dakika barı
  akıtmak 1h barları tazeler ama SİNYAL VERİSİNİ değiştirir (ayrı tur) · açılış/4-saatlik tarama Alpaca
  bütçesini ~1,5 dk tüketir (`AUTO_SCREEN_PROVIDER` ya da disk önbelleği = operatör kararı) · başarısız seed
  yenilemesine kısa negatif önbellek (Gemini §7.1-4) · hub bar yanıtı başına istemcide 5-7 ms JSON ayrıştırma ·
  pytest-xdist (paket 642 sn, 4 parça).

**TAHMİN DOĞRULUĞU — yön girdisi, kalibre olasılık, beraberlik (Tur-83, 2026-09-18).** Rapor:
`docs/TUR83_RAPORU_2026-09-18.md`. Hedef: *"tahminlerin/hesaplamaların doğruluğu nasıl artırılır"*. Kenar iddiası
YOK; tarama/işlem karar yolunda kural değişmedi (değişen: tahmin sekmesi, `/api/analyze`, `predictions_log`,
raporlanan olasılık). Başlangıç: 93.825 çözülmüş tahminde isabet **%47,49** (yazı-turanın ALTI).

- **BULGU-83-1 — yayınlanan yön, bar kapanışından ~39 sn SONRA okunan kotasyondan hesaplanıyordu.** Aynı 43.195
  satırda, aynı referanstan (canlı kotasyon): canlı-yön %48,20 ↔ bar-yön %50,12 (+1,92 puan, gün-kümeli t=+12,8;
  18 günün 15'i). Fark yönlerin ayrıştığı %4,05'ten gelir; orada canlı-yön %26,2 isabet eder (kotasyonun EMA20'yi
  geçen sapması 5 dk'da geri döner). **"Yüksek güven → düşük isabet" anomalisi (§17-C/D, Tur-47) BUYDU:** bar
  kipinde güven kovaları düz (%49–51). `live_predict_service._yon_kipi()` TEK kaynak (servis + `/api/analyze` +
  kurucu), `SIGNAL_ON_BAR_CLOSE` varsayılanı **true**; karşı varyant `dir_live`/`conf_live` (FIELDS sonu) ile
  loglanır → A/B aynı satırlarda sürer. **Bar-yön de yazı-turadır:** kazanılan kenar değil, yazı-turanın altındaki
  kusurun kaldırılmasıdır.
- **BULGU-83-2 — `hit_bar` canlı-yön için ileri-bakış taşıyordu.** Tur-60'ın "uyuşmazlıkta yön daha doğru (%60,9)"
  okuması bu ölçüydü; gölge terfi yüzeyinin (`_dirbar_agrees`) canlı tarafı artık `dir_live`dır.
- **BULGU-83-3 — "Olasılık (kal.)" sabit taban orandan KÖTÜYDÜ** (Brier 0,25002 ↔ 0,24940, t=+4,38): 23 Haziran'dan
  beri tazelenmemiş, örneklem-içi 19 ham kova, başka hedef (bar→bar, beraberlik hariç), başka girdi (HTF/rejim
  hariç), 1h analize de uygulanıyordu. `research/calibrate_forecast.py` baştan: canlı defter → gün-gün yürüyen
  platt ↔ nedensel taban, gün-kümeli **CR0** t<−1,645 değilse eğri DÜZ = taban. Bugün bar kipi **TABAN (t=+0,39)
  → düz %50,12**; canlı kip BİLGİ VAR (t=−2,92, azalan eğri) — güvenin tek bilgisi kovalama iziymiş.
  `_load_calibration(interval)` ŞEMA KAPISI: `meta.girdi="yayinlanan_guven"` + `yon` + `interval` eşleşmeyen eğri
  REDDEDİLİR (23 Haziran biçimi dahil); `_calibration_input` silindi, girdi yayınlanan güvendir.
- **Beraberlik** `hit`te ıskadır (%0,90; ONDS %7,9) → `verdict()` ve kardeşi `forward_harness._verdict` 0,5'e
  beraberlik HARİÇ oranı (Wilson n'i dahil) kıyaslar; `report()` `tie_share` + `hit_rate_excl_ties` taşır,
  `hit_rate`/`history.csv` DEĞİŞMEDİ.
- **Gece görevi** (`forward-harness.ps1`): çözümden sonra `research.calibrate_forecast` + `daily_summary --calib`
  (docstring'de yazılı, satırda hiç verilmeyen bayrak; AI kalibrasyon artefaktları 12 Ağustos'tan beri bayattı —
  eğri dosyasını YAZMAZ, o iş `calibrate_forecast`ındır).
- **Önce/sonra (aynı 36.904 satır, 15 test günü):** isabet %48,14 → **%50,17** (t=+14,3) · yön-getiri
  −0,0099% → +0,0059%/tahmin (maliyet öncesi) · olasılık düzeltmesi tek başına Brier −0,00044 (t=−2,57).
- **Kilit:** `tests/test_tur83_tahmin_dogrulugu.py` (42 test), **mutasyon 24/24**; tam paket 4868/1 atlandı/0.
- **DAĞITILMADI** (servisler Tur-82 koduyla koşuyor, seans açıktı) — adımlar raporun §9'unda. Commit YOK.

**HESAPLAMALARIN DOĞRULUĞU — ufuk, öngörü aralığı, çapa, gün-içi değişim (Tur-84, 2026-09-19).** Rapor:
`docs/TUR84_RAPORU_2026-09-19.md`. Kenar iddiası YOK; tarama/işlem karar yolunda kural değişmedi. Tur-84 oturum
limitinde yarım kaldı (rapor `{KALITE}`, bu bölüm yoktu); Tur-85 tamamladı.

- **BULGU-84-1 — çözüm ufku KONUMLA seçiliyordu.** `prediction_tracker._forward_price` `iloc[pos + 5]` okuyordu;
  IEX'te işlem olmayan dakika bar ÜRETMEZ → satırların %18,6'sında ufuk >5 dk, 411 satır geceyi/hafta sonunu
  aşıyordu (LMT %88,3). İsabet iki tanımda da yazı-tura (%50,17 ↔ %50,01): kusur YÖNÜ değil OYNAKLIĞI bozuyordu
  (getiri varyansı oranı 1,391; IBM std 2,42×). Artık hedef an = t0 + ufuk·aralık: o anda başlayan bar, yoksa
  hedeften önceki son bar — yalnız an GEÇMİŞSE; düzenli ızgarada eski kuralla birebir; "1d"de konum kuralı kalır.
  Geçmiş satırlar yeniden yazılmadı.
- **BULGU-84-2 — "Öngörü" (fiyat ± ATR, yön renginde) "fiyat değişmez"den KÖTÜYDÜ** (MAE 24,47 ↔ 21,29 bps,
  gün-kümeli t=+8,69; yön yazı-turayken ±ATR hataya yalnız ATR² ekler). Yerine kalibre ARALIK: ufuk sonundaki
  fiyatın ~%80 aralığı = çapa × (1 + z·ölçek), z kuantilleri %10/%90 (sembolün ≥300 satırı, yoksa havuz);
  "Tipik Hareket" = medyan|z|·ölçek, yönsüz. Defterin YENİ `atr_pct` kolonu (FIELDS sonu) → gece görevi
  (`calibrate_forecast`) `data/forecast_band.json`ı kurar; yürüyen Gneiting-Raftery aralık skoru KOŞULSUZ aralığı
  gün-kümeli t<−1,645 ile yenmezse kuantil yayınlanmaz. `_load_band` şema kapısı; kapalı piyasada aralık yok.
- **BULGU-84-4 — çapa kotasyon değil son BAR KAPANIŞI:** aynı 83.147 satırda aralık skoru 74,4 ↔ 96,9 bps
  (eşleştirilmiş t=−10,7; 32 günün 30'u). Kotasyon çapası bayat son işlemin gürültüsünü de öğreniyordu (V z_lo −6,98).
- **BULGU-84-3 — "Değişim"/"% (gün)" seans açılışından DEĞİLDİ:** `normalize()` = UTC gece yarısı → gerçek açılış
  satırların yalnız %3,5'inde, gün-içi değişimin işareti %16,5 yanlış. `_seans_acilisi` günlük bardan (IEX 1d =
  seans barı, 4/4 birebir); servis günde bir çekim + 5 dk negatif TTL; bugünün barı yoksa "—".
- Yan: `alpaca_history` önbellek anahtarı feed taşımıyordu (IEX isteği SIP dosyasını okuyup ezerdi) · geçici
  okuma hatası aralık VE eğri yükleyicisinde önbelleğe yazılıyordu (Gemini incelemesi).
- **Kilit:** `tests/test_tur84_hesap_dogrulugu.py` (51 test), **mutasyon 43/43** (ilk koşu 42/43 — `/api/analyze`
  çapası tek uçtan bozulunca sıralama testi yeşil kalıyordu; Tur-85'te sıkılaştırıldı); tam paket 4919/1 atlandı/0.

**ARALIĞIN ÖLÇEĞİ — ATR^us, örneklem-dışı tekrarla (Tur-85, 2026-09-19).** Rapor:
`docs/TUR85_RAPORU_2026-09-19.md`. Kenar iddiası YOK; değişen yalnız aralık ve tipik hareket (tahmin sekmesi +
hisse detayı). Tur-84'ün yarım kalan kapanışı da bu turda (yukarıda).

- **BULGU-85-0 — "daha hızlı ölçek" (Tur-84 açık kalemi) ÇÜRÜDÜ.** 20 nedensel aday ölçek (gerçekleşen varyans
  10–120 dk, seans-içi RV, zaman-normalize TR², EWMA, önceki gün RV, seans-içi ATR) aynı 93.214 satırda yürüyen
  aralık skorunda: HİÇBİRİ ATR14'ü yenmedi (kısa pencereler gürültülü t=+8,6; uzunlar gün-içini kaçırıyor
  t=+4,6…+5,9; seans-içi ATR ayırt edilemedi t=−0,73). Harness üretim `bant_yuruyen`iyle birebir doğrulandı.
- **BULGU-85-1 — ATR aralığı oynaklık DÜZEYİNE göre kalibre değildi:** sembol-içi ATR ondalığında kapsama %70,6
  (en sakin) → %86,4 (en oynak) = ortalamaya dönüş. Ölçek artık `ATR14^us` (sembol kuantilleriyle bu, düzeye
  büzmenin ta kendisi; yeni veri yok). us her test günü YALNIZ geçmişten: −0,63 bps, t=−3,51, 29/32 gün, iki yarıda
  aynı işaret. **ÖRNEKLEM-DIŞI TEKRAR** (bulguların hiç bakılmadığı Mart–Mayıs 2026, aynı 30 sembolün IEX barları,
  686.864 satır / 60 test günü): **−0,53 bps, t=−8,46, 60/60 gün us=0,8**. Önyüklemede us=0,9 (t=−4,34); optimum
  0,8–0,9 arasında düz. Ondalık kapsaması %74,5 → %83,1.
- **BULGU-85-2 — kapanış çarpanı ÖLÇÜLDÜ, TEKRARLANMADI, EKLENMEDİ.** Haziran–Eylül'de son yarım saat kapsaması
  %67,7'ydi ve ≥15:30 ET çarpanı (c≈1,2–1,3) kapıyı geçti (t=−2,75; kapsama → %78,1). Sınır tablolara BAKILARAK
  seçilmişti (Gemini itirazı); örneklem-dışı dönemde çarpan 60 günün HİÇBİRİNDE seçilmedi (orada kapanış kapsaması
  zaten %74,6/%77,0). Kod yazılıp mutasyonla kilitlenmişti ve GERİ ALINDI: her gece yeniden sınanan boş bir özellik
  örtüşen veride er geç sahte bir "geçti" üretirdi. 7 dilimli gün-içi çarpan da üssün üstüne hiçbir şey eklemedi
  (t=+0,36). `test_ornek_disi_tekrarlanmayan_kapanis_carpani_yok` geri gelmesini kilitler. Açık: açılışın ilk 5 dk'sı
  (%62,9) · son yarım saat (%67,7–77) ölçülmüş ama düzeltilmemiş koşullu sapmalardır.
- `_apply_band(band, sym, atr_pct)` TEK kaynak (imza DEĞİŞMEDİ): ölçek `atr_pct ** us` (`us` yoksa 1,0). `_load_band`
  `meta.olcek == "atr14_yuzde_us"` + geçerli `us` ((0, 2]) ister — ESKİ ad `atr14_yuzde` REDDEDİLİR (üssü bilmeyen
  Tur-84 kodu yeni dosyayı ham ölçekte uygulamasın). Defter şeması DEĞİŞMEDİ (`atr_pct` ham ATR%).
- `calibrate_forecast`: iki aday AYNI satırlarda — sabit ATR ↔ her gün yalnız geçmişten seçilen üs; üs yalnız
  gün-kümeli t<−1,645 ile kazanırsa yayınlanır, seçim sonra TÜM satırlardan (Tur-83 platt↔taban ilkesi);
  `_eslestir` satırlar ayrışırsa DURUR; seçim ölçütü `_ic_skor` yayınlanan `_apply_band` temsiliyle 1e−12 içinde
  aynı. **Özet hafifçe iyimserdir** (yapı kararı tüm dönemden): tam nedensel prosedür −0,43 bps (t=−2,24) ↔ özet
  −0,78 (Gemini #1, ölçüldü). Hesap: 687 bin satır / 60 gün ≈ 48 sn (büyür; yükseltme yolu son N gün penceresi).
- Önyükleme (`--bant-barlardan`, 94.929 satır): skor 74,40 → **73,86** bps (koşulsuz 95,32), kapsama %79,0 →
  **%80,0**, sembol aralığı %77,3–81,4 → %78,4–82,5.
- **Süreç:** `NihaiAlgo-ForwardHarness-Daily` her gün **12:00 ve 23:30**'da ÇALIŞMA AĞACINDAKİ `forward-harness.ps1`i
  koşturur; bu tur 12:00'de ara kodla koştu (tutarlı durumdaydı, aralık dosyasına dokunmadı).
- **Kilit:** `tests/test_tur85_aralik_olcegi.py` (33 test), **mutasyon 20/20**; tam paket 4952/1 atlandı/0
  (222 dosya, 343 sn), `ruff check .` + `mypy .` (212 dosya) temiz. Bağımsız inceleme: Gemini 3.1 Pro (`agy`).
  DAĞITILMADI, commit YOK.

**DIŞ BENCHMARK SOHBETİNİN TRİYAJI — λ, PIT evren, kalıntı momentumu (Tur-86, 2026-09-19).** Rapor:
`docs/TUR86_RAPORU_2026-09-19.md`. Kenar iddiası YOK; tarama/işlem karar yolunda kural değişmedi. Kaynak: bir `agy`
(Gemini 3.8 Flash) sohbeti 5,4M→2B transformer'ların eğitilmemiş ağırlıklarla HIZINI/VRAM'ini ölçüp daha büyük model
önermişti; isabet hiç ölçülmemişti ve projenin kapasite süpürmesi (2026-07-31) "sınır sinyalin kendisi" diyordu.
Kayıtlı model kartlarının üçü de `deep_mlp`; hiçbir transformer aktife alınmadı.

- **BULGU-86-1 — `shadow_decision_backtest` CLI'ı `--exec-lag 0`'ı HER koşuda geçiriyordu**; `lambda_sweep()`in
  dürüstlük varsayılanı (1) hiç devreye girmiyor, selfcheck'in önerdiği komut ve `EDGE_FINDINGS`'in tekrar-üretim
  satırı süpürmeyi exec-lag kapısı KAPALI ölçüyordu. Varsayılan `None` = çağrılan fonksiyonun kendi varsayılanı.
- **BULGU-86-2 — λ süpürmesi boş kümeden "ABSTAIN ölçülmüş doğru" hükmü kuruyordu.** Tazelenmiş ölçümde (180 gün,
  exec_lag 1, 5.400 bağlam) **λ=0 dahil hiçbir λ tek yön üretmedi**: kapı-a (`lo95 > 0,5`, `decision.py:652`)
  λ'dan bağımsızdır ve prior'ın alt sınırları LONG **0,4929** · SHORT **0,4029**. `VERDIKT_KOD`
  (`DOGRULANDI`/`KONUSAN_LAMBDA_YOK`/`KAL`), KPI'da `verdikt_kod` + `parametre`; selfcheck `KONUSAN_LAMBDA_YOK`'ta
  "fazlalık ÖLÇÜLEMEZ, kilit prior kapısında" der. λ DEĞİŞMEDİ (0,5).
- **BULGU-86-3 — PIT üyelik kurulumu 2026-08-11'den beri KIRIKTI.** Wikipedia değişiklik tablosunu "Historical
  components of the S&P 500" maddesine taşıdı; `pit_universe` onu `tabs[1]` konumundan okuyordu (artık navbox) →
  `KeyError('date')`. Tablo artık kolon İMZASIYLA aranır (ana madde → yeni madde); tablo ya da `added`/`removed`
  kolonları kurulamazsa `RuntimeError` (güncel kümeyle devam = sessiz survivorship; ikinci dalı Gemini incelemesi
  buldu). `sp500_uyelik(tarihler)` tek çekim, ortak `_geri_sar`. Doğrulama: `sp500_as_of('2021-01-01')` = 506 / 90.
- **BULGU-86-4 — `panel.load_panel` evreni karar günü üyeliği DEĞİLDİR** ("pencerede herhangi bir an üye"). Sekiz
  edge_lab veri setinin satırlarının %5,6–15,9'u karar günü üye olmayan isimden; vixts'te o satırların pnl'i anlamlı
  yüksek (t=+4,23), babmacro'da düşük (t=−2,18). Hepsi FAIL → sahte PASS yayımlanmadı; yeniden koşum AÇIK KALEM.
- **Kalıntı momentumu (BACKLOG #11, ön-kayıtlı) — 5/5 kapı FAIL** (`research/edge_lab/build_rmom_dataset.py`):
  18.194 satır / 545 PIT sembol / 37 ay; purged_cv 0,517 · DSR 0,0006 (`n_trials` 44). Koşulsuz iz literatürün
  yönünde ama gün-t **1,63**; PIT süzgeci olmasa **2,23** olurdu. Defterde 11 aile, sıradaki adayın deflasyonu 48.
- **`capacity_data_sweep --device {cpu,cuda,auto}`** (varsayılan cpu = eski davranış) → `recipe` ile `fit`e; JSON
  `device` (istenen) + `device_resolved` (gerçekleşen). **quant_transformer (5,4M) PIT-97k GPU süpürmesi: AUC
  0,487–0,529, KENAR_YOK_HER_N; DSR hiçbir n'de hesaplanamadı çünkü model HİÇBİR test örneğinde çoğunluk sınıfından
  ayrılmadı** (deep_mlp'nin ~29 katı parametre, aynı hüküm). `_verdikt` ölçülemeyen DSR'yi artık eşikle kıyaslanmış
  gibi basmaz (`dsr_olculen_nokta`).
- **BASAMAK KIYASI (2026-09-20, §3.1) — "770M'i backtest edip canlı algoritmayla karşılaştır".** `fit`'in Tur-42
  bellek reçetesi CLI'ya çıkarıldı (`--optimizer`/`--amp`/`--batch-size`) ve her nokta artık SÜRE + ZİRVE VRAM
  yazıyor. **770M (769.554.928) 8 GB'a SIĞIYOR** (Adafactor+bf16+toplu-iş 32, ~7,0 GB) ama n=6.000'de 3 fold ×
  **1 epoch = 14 dk** → gerçek koşu n=12k'da ~6 sa, n=97k'da ~40 sa (yapılmadı, operatör kararı); sondanın kendi
  değeri AUC **0,4894**, DSR yok. AYNI veri/fold/reçete ile n=12.000: `deep_mlp` 186.702 → AUC **0,5133** (89 sn,
  34 MB) · 5,4M **0,5169** (848 sn, 1,6 GB) · 30,5M XL **0,5259** (1.779 sn, 3,3 GB) · 92M MoE **0,5235**
  (4.309 sn, 3,3 GB) · 282M MoE-XL **0,5333 ± 0,0202** (15.565 sn = 4 sa 19 dk, 5,4 GB). **DSR hiçbirinde
  hesaplanamadı** (sıfır varyans = hiçbir model çoğunluk kuralından ayrılmadı; canlı basamak dahil); AUC 1.509 kat
  parametre aralığında 0,513-0,533 bandında kaldı. 282M `purged_cv` eşiğini (>0,53) İLK KEZ geçti ama std'si
  ±0,0202 (eşik bandın içinde), DSR yok ve tek nokta `YETERSIZ_NOKTA` demektir → kenar iddiası KURULAMAZ; yol,
  aynı basamağı artan n'de koşturmaktır. ECE 0,051 → 0,027 düştü ama bu KAZANÇ DEĞİLDİR (Tur-88, ACI DERS #9):
  AYNI fold'larda sabit taban-oran tahmincisine karşı Brier beceri skoru 15 basamak×fold'un **14'ünde negatif**
  (282M −0,0016/−0,0040/−0,0037; tek pozitif deep_mlp fold 1 +0,0004) — hiçbir basamak sabit ~%34'ü yenmiyor. Maliyet: 282M =
  canlı basamağın **176 katı süre**, 162 katı bellek, karşılığı +0,0200 AUC.
  **SÜREÇ:** 282M koşusu DÖRT kez, makine kritik belleksizken (3,2-3,9 GB boş) oturum boşa geçtiği an sistem
  tarafından durduruldu; operatör uygulamaları kapatıp boş belleği **9,1 GB**'a çıkarınca aynı komut sorunsuz
  bitti. Uzun GPU koşusunu engelleyen şey VRAM değil ANA BELLEK olabilir; koşudan önce boş belleği ÖLÇ.
- **Kilit:** `tests/test_tur86_olcum_kapilari.py` (26 test), **mutasyon 26/26**; tam paket **4979/1 atlandı/0**
  (223 dosya, 4 parça, 340 sn); ruff + `mypy .` (213 dosya) temiz. 234k süpürmesi sistem belleği yüzünden
  DURDURULDU (artefakt yok, açık kalem); 282M sonradan bitti (yukarıda), 770M tam koşusu hiç bitmedi (2026-09-21
  02:13'te başladı, bellek korumasıyla kesildi; açık kalem) — boş bellek 3,6 GB / 23,7 GB ölçüldü. Bağımsız inceleme: Gemini 3.1
  Pro (`agy`) — 4 bulgunun 1'i gerçek (düzeltildi), 1'i yanlış pozitif ("KESİN" etiketliydi). Commit YOK.

**PANELDEN BAĞIMSIZ GÜNLÜK ÖĞRENME (Tur-87, 2026-09-21).** Rapor: `docs/TUR87_RAPORU_2026-09-21.md`. Yazan Codex
(hedef: "kendini sürekli geliştiren finans AI'ı"); bu bölüm Tur-88'de eklendi. Kenar iddiası YOK.

- **BULGU-87-1 — öğrenme hafızası yalnız pano açılınca yenileniyordu.** `learning.update_from_predictions`in tek
  üretim çağıranı `GET /api/ai/learning` idi: günlük harness prior'ları yenilemiyordu (21 Eylül'de kalıcı durum
  18 Eylül 21:52'deydi) ve panonun `tail` parametresi karar motorunun okuduğu prior'ları başka örneklemle ezebiliyordu.
  `forward_harness --snapshot` artık öğrenmeyi çözümden SONRA varsayılan 5.000 satırlık pencerede YAZAR; pano
  `persist=False` ile önizler; `_save` sonucu `saved` olarak döner (yazma hatası başarı diye raporlanmaz, harness
  bunu açıkça loglar ve diğer raporları sürdürür). Nöral ağırlık yeniden EĞİTİLMEZ; `retrain_candidate` bir bayraktır.
- Gerçek koşu: 4.831 çözülmüş kayıt → yazıldı, `drift=True`; tam değerlendirme 101.004 tahmin / 36 sembol →
  **NO_EDGE**, meta-kapı **GATE_HURTS**.
- **Tur-86 §9.2:** aynı 3.917 kayıtta eski yayınlanan yön ↔ bar yönü: LONG posterior 0,5144 → 0,5439, SHORT
  0,4255 → 0,4581 (SHORT %95 aralığı [0,4351; 0,4813]). Eski yön farkın bir parçasını açıklıyor, üstünlük üretmiyor;
  tarihsel etiketler değiştirilmedi (`data/learning_cohort_audit_2026-09-21.json`).
- **Canlı doğrulama (Tur-88):** pano 2026-09-22 16:35'te yeniden başladı; `ai_learning_state.json` 21 Eylül 18:51'deki
  elle koşudan beri DEĞİŞMEDİ → GET önizlemesi yazmıyor. Raporun "sonraki tetik 21 Eylül 23:30" beyanı ise TUTMADI
  (Tur-88: iki tetik uykuda kaçtı). Kilit: `tests/test_ai_learning.py` + `tests/test_forward_harness.py`.

**YARIM BIRAKILMIŞ İŞİN DEVRALINMASI — uyuyan gece görevi, karışık tarih, işaret-yalnız terfi (Tur-88, 2026-09-22).**
Rapor: `docs/TUR88_RAPORU_2026-09-22.md`. Hedef *"projedeki eksiklikleri bulup tamamlar mısın"*. Kenar iddiası YOK;
tarama/işlem karar yolunda kural değişmedi.

- **Devralınan iş `research/auto_retrain.py`** (Codex yazdı, raporsuz, izlenmeyen dosya; `forward_harness --snapshot`
  21 Eylül'den beri çağırıyordu): yeni canlı sonuçlarla günlük ADAY eğitimi — `build_live_dataset` → `live_retrain.assess`
  (lojistik+GBM, iki bölme + purged CV + null ayrışma) → `data/retraining/<run_id>/` (veri/kod/reçete karması), DURABLE
  ise aday yalnız o dizine (`save_model(output_dir=…)`); canlı `meta_model.*` ve `META_GATE` DEĞİŞMEZ. Aynı veri =
  `UNCHANGED`, <300 yeni satır ya da yeni gün yok = `WAITING_DATA`, tek süreç kilidi. Bu tur ekledi: **`sinav_no`**
  (tamamlanmış sınav sayısı; rapor + `latest.json` + aday kartı + öğrenme raporu satırı) — örtüşen veride her gece
  yeniden sınamak çoklu testtir (Tur-85), `DURABLE`ın DSR kapısı yoktur. **İlk gerçek koşu (sınav #1): NO_EDGE,
  15.453 satır / 30 sembol, 190 sn** (GBM purged AUC 0,5009 ↔ null 0,4922: ayrışma yok).
- **BULGU-88-1 — gece görevi iki tetiği uykuda kaçırdı** (21 Eylül 22:51 → 22 Eylül 14:24; `Missed=2`,
  `StartWhenAvailable` uyanışta koşmadı) ve öğrenme hafızası 28,5 sa bayat kaldı; kimse uyarmadı. `deadman._DAILY`:
  `ai_learning_state.json` (Tur-87'den beri yalnız gece görevi yazar) seanstan BAĞIMSIZ, eşik 26 sa; gerçek ölçüm
  1.712 dk → `BAYAT`. `register-forward-harness.ps1 -WakeToRun` (opt-in) ile görev kullanıcı onayıyla yeniden
  kaydedildi (`WakeToRun=True`), ama güç planında uyandırma zamanlayıcıları (RTCWAKE) AC/pilde KAPALI → bu sistem
  ayarı operatörce açılmadan makine uyandırılamaz.
- **Canlı gölge nöral model yeniden eğitildi** (kullanıcı onayı; Tur-20 yöntemi, doğru sıra, 15.454 satır):
  kart `06067576…`, OOS-AUC **0,495 ± 0,015**, DSR 0,0 — beceri yok; öz-denetim 1.200/1.200 çıktı, OOD 0,
  p_win ort. 0,4348 ↔ taban 0,481. Koşan süreçler eski kartı tutar, yeniden başlatmada devreye girer.
- **BULGU-88-4 — bu turun `tail -F` izleyicisi 23:30 koşusunu düşürdü** (PS 5.1 `Add-Content` "dosya kullanımda",
  `ErrorActionPreference=Stop`, günlükte tek satır yok; karalama kopyasında yeniden üretildi). `forward-harness.ps1`
  artık `Write-Log` (10×0,5 sn, sonra `forward-harness.log.kilitli`, asla fırlatmaz). Koşu `Start-ScheduledTask` ile
  telafi edildi: sonuç 0, öğrenme n=4.895 yazıldı.
- **BULGU-88-2 — `train_meta._chrono_key` `format="mixed"`** (Codex düzeltti, bu tur ölçtü): pandas 3.0.3'te
  `utc=True` tek başına canlı setin **3.276/15.453 (%21,2)** satırını NaT yapıp kronolojinin SONUNA atıyordu (%60
  zaman bölmesinde 4.508 satır farklı); `meta_dataset_live_1m_2026-07-30.csv`de 954/5.996. Tarihsel canlı-1m
  ölçümleri (Tur-20/22/35/36/77, canlı `deep_mlp` eğitimi) bu sırayla kuruldu. Temel ölçüm yeniden koşuldu (aynı
  kod/tohum/fold, tek değişken sıra): OOS-AUC **0,5356 → 0,5200**, Brier 0,2492 → 0,2546 (sabit ≈0,249'dan kötü) —
  eski sıra AUC'yi +0,016 şişiriyordu, hüküm ("kenar yok") değişmedi.
- **BULGU-88-3 — Tur-86'nın "tek kazanç kalibrasyon" cümlesi** sabit tahminciye karşı tutmadı (BSS 15 hücrenin
  14'ünde negatif); düzeltildi. 282M "durduruldu" çelişkisi kapandı; 770M tam koşusunun bitmediği yazıldı.
- **BULGU-88-5 — gölge terfi kapısının ilk `PROMOTE`u (`conf_bar`, `value_area_pos`) işaret-yalnız onaydır.**
  `conf_bar` 20 Eylül verisiyle SLIP_ARTIFACT (ortak referansta uplift +0,0000, t=0,02), iki gün sonra +0,0015
  (t=0,57) ile PROMOTE: `_apply_rule` onayı `uplift_pp > 0` ile verir. `value_area_pos` OOS t=1,80 (p=0,07), 9 gün.
  Canlıya bir şey GİRMEDİ (`shadow_bridge.RULES` kodda, `conf_bar: HOLD`); kapı sonuç görüldükten sonra DEĞİŞTİRİLMEDİ
  (Tur-23). **Ön-kayıt:** iki kural donduruldu, yalnız 2026-09-23 00:00 sonrası satırlarda canlı VE bar referansında
  kümele t > 1,645 ile sınanacak.
- **Tur-86 açık kalemi (BULGU-86-4) KAPANDI:** yedi edge_lab ailesinin mevcut satırları karar günü S&P 500
  üyeliğiyle süzülüp AYNI aile adıyla kapılara verildi (`data/edge_lab_<aile>_v1_pit.csv`; sayaç şişmedi, 12 ayrık
  aday). **Hiçbir hüküm değişmedi:** altısı FAIL, `optskew` kapsam ön-koşulundan HÜKÜM YOK; en yakını `seasonality`
  (purged 0,540 · gate_time PASS) süzgeçten sonra gate_symbol'ü de kaybetti. `rebalance` bilerek dışarıda (hipotezi
  endekse giriş/çıkıştır). Sınır: ücretsiz PIT kaynağı ad değişikliklerini (FLT→CPAY, RE→EG) bilmez.
- **Yan gözlem:** yeni pencerede LONG lo95 **0,5104 > 0,5** (SHORT 0,3871) → kapı-a LONG için ilk kez açık; iki yön
  birlikte piyasa sürüklenmesini ölçüyor (ACI DERS #8). `AI_LIVE_MODE=off` → canlı etkisi yok.
- **Kilit:** `tests/test_tur88_devralinan_is.py` (9 test), **mutasyon 8/8** (bayt düzeyi, `-B`, MD5 eşit);
  tam paket **4.996 geçti / 0 kaldı / 1 atlandı** (494 sn; ilk koşudaki tek kırmızı, canlı `state.json` yazıcısıyla yarışan kararsız bir kapıydı ve düzeltildi); `ruff check .` + `mypy .` (214 dosya) temiz. AGENTS.md CLAUDE.md'den yeniden türetildi. Tur-86..88 kullanıcı isteğiyle TEK commit (push YOK).

**ÇALIŞTIRINCA GÖRÜNEN MANTIKSIZLIKLAR — hayali kayma, 35× kaldıraç, boş tarama (Tur-89, 2026-09-23).**
Rapor: `docs/TUR89_RAPORU_2026-09-23.md`. Hedef *"projeyi çalıştırdığımda anlatması zor olan eksiklikler ve
hatalar var"*. Pano 15 sekme gezildi, her tuhaf sayı kaynağına izlendi. Kenar iddiası YOK; strateji kuralı değişmedi.

- **BULGU-89-1 — sanal hesabı eriten hata: IEX ince defteri "ölçülmüş makas" sayılıyordu.** Tur-77/78'in yarım-makas
  kayması `quote_spread_pct`ten %50'ye kadar her makası alıyordu; ücretsiz akış IEX'tir (NBBO değil). 2026-09-14→22'de
  45 girişte makas %9–12 → kayma %5–6, **35 giriş HEDEFİN ötesinde doldu** (PPG sinyal 106,49 → dolum 112,27, hedef
  109,67; −14,8R). Hesap 257 k$ → 19,5 k$. `MAX_CREDIBLE_STOCK_SPREAD=0,01` üstü ölçüm sayılmaz (taban kayma); canlı
  girişte `min_rr` DOLUM fiyatında (`FILL_SLIPPAGE`), boyut dolum − stop riskinden.
- **BULGU-89-2 — Reg T alım gücü pozisyon başına 4× ve toplam değildi** (brüt 7,85 M$ / 226 k$ = **35×**). Artık
  Σ|nominal| ≤ özkaynak / başlangıç marjı. Tur-50'nin "iki bağımsız katman → brüt ölçülemez" öncülü yanlıştı: iki
  katman AYNI `risk_mgr`/`store`/`pmgr`'yi paylaşır.
- **BULGU-89-3** SHORT kapanışı T+1 hasılatı sayılıyordu ("Bloke" nakdin 122 katı) → yalnız LONG satışı; etiket
  "Takasta". **BULGU-89-4** katmanlar seans dışında 5 sn'de bir 550 sembol reddediyordu (22 dk'da 31.404 satır) →
  `TierScanner(gate=can_enter_new_position)`; `[state] yüklendi` ve `tur … bitti` log seli kısıldı. **BULGU-89-5**
  taşınan pozisyon deftere `EventBreakout V1` + RR 0 ile düşüyordu → `VirtualPosition.strategy` + RR pozisyondan.
- Pano: karne `[object Object]` · `fmtTs` tz'li damgayı yerel saate çevirir, defter "ET" etiketli · beta boşken $0 ·
  PDT eşik/açılış günü · döngü "gecikmesi" artık iş süresi · Bid/Ask IEX ince defter işareti · `HubProvider.stream_status`.
- **Değiştirilmedi:** sanal hesap sıfırlanmadı (operatör kararı; eski defter yeni defterle KIYASLANAMAZ) · gölge
  koşucu görevi hâlâ 09:30 tetiğinde (kayıt operatör onayı ister).
- **Kilit:** `tests/test_tur89_calisma_mantigi.py` (16 test), **mutasyon 14/14**; tam paket **5.012/0/1 atlandı**;
  `ruff check .` + `mypy .` (214 dosya) temiz. Commit YOK.

**ÇALIŞAN SİSTEMİN HESAP BAĞLAMI — düzeltilmemiş bölünme, uzatılmış seans, çift gün, blok PDT (Tur-90, 2026-09-23).**
Rapor: `docs/TUR90_RAPORU_2026-09-23.md`. Hedef *"proje çalışırken algoritmasında/hesaplamasında hata/eksiklik/bağlam
eksikliği var"*. Kenar iddiası YOK; strateji kuralı değişmedi — değişen, kuralın OKUDUĞU barın içeriği.

- **BULGU-90-1 — canlı Alpaca barları düzeltilmemişti.** BULGU-19 `adjustment=all`ı yalnız `alpaca_history`e koydu
  ve *"canlıda da … sahte −%90 boşluk"* yazdı; canlı `get_bars` 7 hafta ham koştu. Ölçüldü: APH 1h −%49,8 (09-03,
  2:1), CRWD 1d −%74,9, KLAC −%89,4; evrende ~10 ayda 12 bölünme. `_adjustment` artık `alpaca_provider`da TEK kaynak.
- **BULGU-90-2 — Madde 22 canlıda uygulanmıyordu.** Alpaca IEX barları uzatılmış seansı taşır (odak katmanında
  200 barlık pencerenin %10–21'i; hacim 3,5 k ↔ 141 k) ve saat barı saate hizalıdır (09:00 barı 09:00–09:30'u içerir;
  31 günün 197'sinde, 56'sında tepe/dibi o yarım saat). Tek değişken süzgeç eyleme dönük kararların **%11,6**'sını,
  eski↔yeni canlı yol **%16,6**'sını değiştirir. `data_provider.rth_mask` (XNYS; tatil/yarım gün) + `resample_bars`;
  canlı 1h 30 dk alt bardan AYNI saat ızgarasında kurulur. Araştırma yolu `rth=False` (önbellekler süzülmemiş).
- **BULGU-90-3** akışın 1d serisinde her gün iki kez vardı (canlı UTC 00:00 + REST 04:00) → ET günü + REST etiketi,
  tohumun son günü birleşir. **BULGU-90-4** PDT takvim bloğuydu (Tur-62 ponytail koşulu doldu: NLV 19,5 k$) →
  `day_trade_dates` + yürüyen son 5 NYSE seansı (ET) + sunucuda `pdt_unlock_date` (pano Cumartesi basıyordu);
  gerçek: 178 day-trade, açılış 2026-09-29. **BULGU-90-5** yfinance yedeği ET-tz'li index → naive UTC.
  **BULGU-90-6** kapı-a kenarı `|2p−1|` aleyhte posterioru kenar sayıyordu (QQQ SHORT 0,297 → +0,405) → taraf
  çözüldüyse işaretli. **BULGU-90-7** gölge AI kapalı piyasada karar yazıyordu (%27) ve 1m kararları 5 GÜNLÜK ufuklu
  hafızaya giriyordu (anıların %80'i) → gölge çıkarım yalnız açıkken, gün-içi karar anı yazmaz.
- **Değiştirilmedi:** canlı saat ızgarası 09:00, yfinance/backtest 09:30 (göç = strateji değişikliği) · araştırma
  önbellekleri uzatılmış seansı taşır · geçmiş 1m anılar silinmedi · kill-switch kapalıyken −%98 (operatör kararı).
- **Kilit:** `tests/test_tur90_hesap_baglami.py` (20 test), **mutasyon 19/19**; Tur-62 PDT dürüstlük kapısı yeni
  tanıma çekildi; MainHub yeniden üretildi; tam paket (4 parça) **5.032/0/1 atlandı**, `ruff check .` + `mypy .`
  (214 dosya) temiz. Tek süreçte paket bir kez `test_tur80_optimizasyon`da takıldı (tek başına ve parçalı yeşil;
  neden açık kalem). Servisler yeni kodla başlatılıp doğrulandı, sonra kapatıldı. Commit YOK.

**CANLI BARIN SAĞLAYICIDAN BAĞIMSIZLIĞI — tek ızgara, tek gün-içi düzeltme kipi (Tur-91, 2026-09-23).**
Rapor: `docs/TUR91_RAPORU_2026-09-23.md`. Hedef *"algoritmayı detaylı geliştirir ve eksiklerini tamamlar mısın"*. Kenar
iddiası YOK; strateji kuralı değişmedi — değişen, kural hangi sağlayıcıdan beslenirse beslensin AYNI barı görmesi.

- **BULGU-91-1 — canlı 1h ızgarası sağlayıcıya bağlıydı.** Alpaca REST/akış saate (09:00, 10:00 …), yfinance yedeği
  seansa (09:30, 10:30 …) hizalı; hub 429'da aynı istekte yedeğe düşer (Tur-82) ve `hub.log`da seans içinde günde
  **471–1.446** 1h çekimi yedekten geldi → aynı hissenin serisi gün içinde iki ızgara arasında gidip geliyordu. Aynı IEX
  30 dk RTH barlarından iki ızgara (30 sembol, ~25 seansın her 30 dk'sı): eyleme dönük 1.846 anın **%64,4**'ünde karar
  ayrışıyor. `resample_bars` artık seans açılışına çapalı (`offset = 570 mod dk`; 5m/15m/30m'de 0); akış toplamı aynı
  fonksiyondan geçer. İki sağlayıcının da kurabildiği tek ızgara budur (yfinance 30 dk barı 60 günle sınırlı).
- **BULGU-91-2 — gün-içi temettü düzeltmesi sağlayıcıya bağlıydı.** Alpaca `all` gün içini temettü de düzeltir, yfinance
  gün içi yalnız bölünme (günlüğü ikisi de): her ex-tarihten geriye A/Y oranı bir temettü verimi kadar basamaklanıyordu
  (SPY 0,9975 · XOM 0,9936); tek değişken kararların **%3,9**'u. `_canli_duzeltme(interval)`: gün içi `all→split`,
  `dividend→raw`; günlük ve araştırma yolu `_adjustment()` (değişmedi).
- **Sonra:** yeni Alpaca 1h ↔ yfinance 1h — 5 sembolde 700/700 bar aynı damga, oran medyanı 1,00000; aynı anda karar
  farkı %64,4 → **%9,85** (kalan IEX↔konsolide besleme farkı, Tur-59 sınıfı). Backtest A/B (2 yıl, 30 sembol, ızgara
  tek değişken): −%19,32 ↔ −%18,82, t=+0,16 → **ayırt edilemez**; onarım performans değil tutarlılık düzeltmesidir.
- **BULGU-91-3** yfinance seans sonrası son günlük satırı OHLC'siz döndürür (25/25 sembol) → yedek artık atar (Alpaca
  yolu ve backtest zaten atıyordu). **BULGU-91-4** otomatik tarayıcı kapalı piyasada aynı barları 4 saatte bir ve her
  yeniden başlatmada yeniden tarıyordu (aynı veriden 63/64/63 geçen); `girdi_tazelenmedi()` seans açılışına erteler.
- **Replay** canlıyla aynı barı kurar: 1m yalnız seans, gün-içi HTF o dakikalardan 09:30 ızgarasında türetilir
  (araştırma önbelleğinin saat barı canlı bar değildi); beklenen-bar denetimi seans ızgarasında.
- **Değiştirilmedi:** besleme farkı (SIP = bar kümesi değişir) · 15:30 yarım barı `etiket + 60 dk` kuralıyla 16:30'da
  görünür (piyasa kapalı) · araştırma önbellekleri · açılışın ilk yoklamasında bir gereksiz tarama (`ponytail:`).
- **Kilit:** `tests/test_tur91_ortak_izgara.py` (14 test), **mutasyon 14/14**; saat ızgarasını kilitleyen üç eski test
  (Tur-90 normalize + istek, Tur-82 seed otoritesi, akış saatlik toplam) senaryoları korunarak yeni ızgaraya çekildi;
  MainHub yeniden üretildi; tam paket (4 parça) **5.046/0/1 atlandı**, tek süreçte de aynı (271 sn; Tur-90'ın takılması
  yinelenmedi), `ruff check .` + `mypy .` (214 dosya) temiz.
  Servisler kapalı bırakıldı; `proje-ac.ps1` ile açılınca devreye girer. Commit YOK.

**HİSSE DETAY TAHMİN FİYATI + GRAFİK; QUANT BEYİN CANLI YOLDA (Tur-92, 2026-09-23).** Rapor:
`docs/TUR92_RAPORU_2026-09-23.md`. Hedef *"hisse detayda tahmin fiyatı yok, grafik eksik/hatalı; Quant gölgedeyse canlıya al,
tam entegre ve sorunsuz olsun"*. Kenar iddiası YOK; tarama/işlem kuralı değişmedi.

- **BULGU-92-1 — detay sayfası hiçbir aralıkta tahmin basmıyordu:** Tur-84 aralığı yalnız `1m`'de açtı, seçicide `1m` yoktu.
  `web_dashboard._tahmin_bandi`: ufuk = defter ufku (5 bar), z = (C[t+5]/C[t]−1)·100/ATR14%, kuantiller sembolün 1000
  barından, çapa son bar kapanışı; YÜRÜYEN sınav (yalnız çözülmüş z, s+5 ≤ t) kapsama + MAE'yi karta basar. Merkez medyandır
  ancak sınavda "fiyat değişmez"i yenerse (yoksa son kapanış — BULGU-84-2). 8 sembolde yürüyen kapsama 1000 barda %77–79
  (220 barda saçılım %66,7–83,3); medyan kayması ortalamada hiçbir aralıkta değişmezi yenmiyor. Seçiciye `1m` eklendi.
- **BULGU-92-2 — Quant kapı-b hiçbir kanıtla geçilemiyordu:** `E = post·kazanç − (1−post)·kayıp − λ·|CVaR95|` cezayı
  post=1'de bile düşüyordu; 30 sembol×aralıkta E(post=1) > 0 olan 0/30 (gereken post 1,05–1,48). Ceza kaybeden dala
  taşındı: `−(1−post)·(kayıp + λ·|CVaR95|)` (`closes=None` bit-denk; kapı-a AND'li kaldığı için "ABSTAIN→yön ASLA"
  korunur). Bugünkü posteriorlar (0,39–0,59) yeni eşiğin (0,63–0,81) altında → canlı kararlar DEĞİŞMEDİ.
- **BULGU-92-3 — gölge koşucu 1dk modeline günlük özellik veriyordu** (max|z| 21–71 ↔ q99 3,15; doymuş p posteriorun
  %43'ü): `brain._neural_forecast` canlı `neural_shadow` ile AYNI OOD eşiğini ve env'i (`NEURAL_SHADOW_MAX_Z`) uygular.
- **BULGU-92-4 — grafik:** genişlik bir kez okunuyordu (gizli sekmede 0 px) → `autoSize`; alt grafikler mantıksal
  indeksle KAYIKTI → ana grafiğin zaman kümesine (barlar + 26 gelecek seans barı, XNYS) boşluk noktalarıyla hizalanır ve
  sağ eksenler zaman-ekseni boyut olayında en genişine eşitlenir; saatler yerel; ileri Ichimoku bulutu; tahmin konisi.
  `_gelecek_barlar`/`_bar_bitisi` takvimden (hafta sonu, tatil, yarım gün; takvim sonu kırpılır).
- **Quant canlı: `.env` `AI_LIVE_MODE=advise`** — `main.scan_symbol` her LONG/SHORT adayında görüşü gerekçeye/Telegram'a/
  karar günlüğüne yazar (aday başına ~5,6 ms). `enforce` AÇILMADI: 14 günde 7.711 canlı adayın 7.711'i ABSTAIN → bugün
  her sinyali susturur. Detay sayfasında "Quant Beyin" kartı aynı motoru sorar (`decide(persist=False)` — defter yazmaz).
  `/api/ai/overview.live_mode` + Beyin sekmesi "CANLI · ADVISE". Ana süreç bekçi üzerinden iki kez yeniden başlatıldı,
  tek süreç doğrulandı.
- **BULGU-92-6 — advise açılınca karar defteri aynı barı onlarca kez yazdı** (12 dk'da 123 satır / 54 tekil, bir karar 49
  kez; tarama turu aynı barı ~6 sn'de bir sorar): `ai_advisor._judge` (kanal, sembol, taraf) başına defterdeki son barı
  tutar → karar her turda hesaplanır, `ai_decisions.jsonl`e bar başına BİR kez yazılır (sonra 6 satır / 6 tekil).
- Yan: anthropic SDK 1.8 blok birleşimi `mypy .`'yi kırıyordu; aynı `b.text` deseni ÜÇ kopyada (`ai/router`,
  `notifier/chart_analyst`, `data/llm_event`) `getattr(b, "text", "")` oldu.
- **Kilit:** `tests/test_tur92_detay_tahmin_quant.py` (18 test), **mutasyon 18/18**; tam paket **5.064/0/1 atlandı**; `test_ai_decision` λ testi yeni
  formüle çekildi.

**CANLI GÖLGE SİNİR AĞI 282M `quant_transformer_moe_xl` İLE DEĞİŞTİ (Tur-93, 2026-09-23).** Rapor:
`docs/TUR93_RAPORU_2026-09-23.md`. Hedef *"mevcut sinir ağını 280M küsür parametreli ağ ile değiştir ve eğitimine
başla"*. Kenar iddiası YOK; model KARAR-DIŞI gölge p_win kaynağıdır (sinyal/Quant/boyut okumaz).

- Eğitim: canlı 1m seti (16.198 satır / 30 sembol / taban %48,1), 281.783.915 parametre, 3 fold purged CV + tüm
  veri, Adafactor + bf16 + toplu-iş 128 (256'da zirve VRAM 7,18 GB, 128'de 5,07 GB — `train_neural --batch-size`
  eklendi, verilmezse eski davranış). Claude Code bellek koruması yüzünden `Start-Process` ile ayrık süreç.
  Sonuç (4 sa 38 dk): OOS-AUC **0,503 ± 0,007**, Brier 0,2497 (sabit taban ≈0,2496), ECE 0,0121, DSR 0,0134 →
  beceri YOK; kart `c70acd68`. Canlı yol (ayrı süreçte, varsayılan ayarlar) 60/60 gerçek adayda üretti, OOD 0, ama
  p_win 0,4855–0,4874'e sıkışık: 16 bin satır bu basamağın `recommend_tier` alt sınırının (3,6 milyon) 225'te biri.
  Proje 22:43'te operatörce kapatılmıştı → değişim bir sonraki `proje-ac`ta.
- **BULGU-93-1 — 282M'i canlı döngüye olduğu gibi takmak tahmin döngüsünü tıkardı.** Tek aday MC=20: CPU döngü
  2.924 ms, GPU döngü 1.825 ms (20 sıralı geçiş, MoE çekirdek-başlatma sınırlı), CPU toplu 1.040 ms, **GPU toplu
  102 ms**; döngü dakikada ~30 aday sorar ve çıkarım kilit altında sıralıdır. `_mc_samples(batched=True)` T kopyayı
  tek ileri geçişte örnekler (maskeler örnek başına → bağımsız MC örneği; RNG akışı farklı, T=1'de bit-denk);
  `to_device()`; `NEURAL_SHADOW_DEVICE=auto` → `deep_mlp` CPU + döngü (geçmişle aynı), büyük basamak CUDA + toplu;
  aygıt hatası CPU'ya döner. `status()["device"]`.
- Değişim: `_pick` en yeni `_1m_` kartını, saatlik koşucu en yeni `neural_*.json`u seçer; koşan süreç modeli bellekte
  tutar → canlı değişim yeniden başlatmayla. Geri dönüş: `NEURAL_SHADOW_MODEL` + `AI_NEURAL_MODEL` ya da
  `registry.deprecate`. Saatlik koşucu CPU döngüsünde kaldı (~1,5 dk/koşu).
- **Kilit:** `tests/test_tur93_moe_xl_canli.py` (13 test), **mutasyon 12/12**; tam paket **5.077/0/1 atlandı**; `ruff check .` + `mypy .` (214 dosya) temiz.

**2 YILLIK TARİHSEL 1m PİLOTU — daha çok veri canlı tahmini iyileştirmiyor (Tur-94, 2026-09-24).** Rapor:
`docs/TUR94_RAPORU_2026-09-24.md`. Hedef *"canlı veri yerine son 10 yılla eğitsek? → önce 2 yıl, takip listesinin tümü"*.
Kenar iddiası YOK; kart kaydedilmedi, canlıya bir şey girmedi.

- Veri: 520 sembol × 2 yıl IEX 1m (`alpaca_history`, 58,6 M bar, `C:\Users\Kagan\nihai_hist`, OneDrive DIŞI) →
  `research/build_hist_1m_dataset.py` canlı yolun kendi fonksiyonlarıyla (RTH, son 220 bar, `generate_signal` →
  `_candidate_side` → `extract_features`), her 5. bar → 2,09 M aday satırı. Sınır: bugünün listesi → survivorship.
- **BULGU-94-1 — bar kapanışına çapalı etiket makas sıçramasını öğretiyordu.** Canlı etiket kapanıştan ~39 sn sonraki
  kotasyona çapalı; `deep_mlp`in tarihsel AUC'si (diğer semboller) 0,5215 → 1 bar gecikmede **0,5004** (ACI DERS #7,
  yedinci kez). Çapa ölçülerek seçildi (14.428 eşleşen canlı satır, etiket uyumu): kapanış %75,2 · **sonraki barın
  ortası %76,1**. `taraf_getirisi(capa="orta_i1")` varsayılan; `--yeniden-etiketle` özellikleri yeniden hesaplamadan
  etiketi kurar → 1.969.661 satır. Hiçbir bar çapası %76'yı geçmiyor: canlı etiket IEX kotasyon gürültüsü taşır.
- `research/hist_learning_curve.py`: eğitim = canlı dönemden önceki (< 2026-07-05) SON n satır; test A canlı 1m seti,
  B aynı dönemin tarihsel satırları, C yalnız canlı semboller (B↔C evren, C↔A etiket); AUC + Brier ↔ sabit taban +
  gün-kümeli t. `deep_mlp` 16 bin → 1,73 M: tarihsel **0,510 → 0,521** (t=−6,8) ama canlı sembollerde en fazla
  +0,007 AUC (BSS +0,0017), canlı etikette 0,50–0,51 düz. **282M 16 bin → 1 M: çıktı std'si her noktada < 5e-5** —
  taban orana çöküyor, küçük modelin bulduğu tarihsel deseni 1 M satırda da bulamıyor (reçete/optimizasyon sınırı:
  1 epoch, lr 1e-3, Adafactor). Hüküm: **10 yıla genişletmenin ölçülmüş gerekçesi YOK.**
- **BULGU-94-2 — `fit` doğrulama kümesini tek ileri geçişte GPU'ya veriyordu.** 282M'de val 9,6 bin satırda VRAM
  sistem RAM'ine taştı (paylaşımlı 7,9 GB, süreç 18 GB, boş RAM 0,4 GB); val artık `_INFER_BATCH` parçalarıyla
  (≤ 4.096 satırda eskisiyle BİT-DENK; çok parçada satır-ağırlıklı kayıp), en iyi durum kopyası CPU'da.
- Süreç: aynı GPU'da oyun (Palworld ~3 GB, %46) 282M'i 2× yavaşlattı; uzun GPU koşusundan önce GPU motor
  kullanımına bak.
- **Kilit:** `tests/test_tur94_tarihsel_1m.py` (11 test), **mutasyon 9/9**; tam paket **5.107/0/1 atlandı**; ruff + mypy
  (değişen dosyalar) temiz. Commit YOK.

**BACKTEST + BENCHMARK + TAM DOĞRULAMA KATMANI (Tur-95, 2026-09-24).** Rapor: `docs/TUR95_RAPORU_2026-09-24.md`.
Hedef *"projeye backtest ve benchmark ve yapılması gereken tüm testleri yapar mısın"*. Kenar iddiası YOK; strateji kuralı
ve canlı karar yolu değişmedi. Ortam: Tur-94'ün GPU eğitimi sürerken (proje kapalı, GPU kullanılmadı).

- **Koşulan katmanlar:** tam paket 5.090/0/1 atlandı (231 dosya, 401 sn) · ruff · `mypy .` 216 dosya · MainHub 27/27 +
  sürüklenme yok · 25 `.ps1` sözdizimi · AGENTS.md türetimi · `pip check` · AI öz-denetim · canlı hat replay'i
  (2026-09-23, AAPL+MSFT, 0 hata) · AI karar katmanı λ süpürmesi · performans benchmark ×3.
- **Backtest tekrar-üretimi:** dünkü kuralsız deneyin 2.640 görevi (22 konfig × 2 mod × 2 aralık × 29 sembol + portföy,
  yerel `data/alpaca_hist`) bugünkü kodla ALAN ALAN **2.640/2.640 bit-denk**; Tur-95 değişikliğinden sonra yine; üretim
  yazımı 0. Düzenek (CONFIG `object.__setattr__`, `research/ablation.py` deseni) sonraki motor değişikliklerinin tanığıdır.
- **BULGU-95-3 — varsayılan portföy backtest'i dönemin %93–96'sında işlem yapmıyordu.** Backtest kill-switch'i KALICI ve
  varsayılan AÇIK (canlıda `ENABLE_KILL_SWITCH` KAPALI): 1h'de verinin ilk ayında (35–50. işlem), 1d'de 2020'de
  tetikleniyor. Sıfır-kenarlı sentetik yürüyüşte 30+ işlemli 24 koşunun 21'inde, çoğu tam 30. işlemde; tarayıcı tipi
  koşuda (1h, 6 ay) 29 sembolün 23'ünde. `BacktestResult.kill_switch_at` + `kill_switch_idle_pct` (tek sembol ve
  portföyde AYNI tanım: tetiklendiği iterasyon) + `short_summary` `KILL_SWITCH@…` + CLI satırı. Sayılar DEĞİŞMEDİ;
  varsayılan AÇIK kaldı (spec Madde 20 ↔ canlı seçimi: operatör kararı).
- **BULGU-95-2 — Monte Carlo `ruin_probability` bir olasılık değildi.** Karıştırma toplamı değiştirmez → nihai sermaye
  tek değer, eski `P(nihai<=0)` 0 ya da 1; Madde 52 kapısı (`gate.MC_RUIN_LOW`) ve tarayıcı iflas filtresi bu alanı
  okuyordu. Artık YOL-İÇİ iflas (eskisinden asla küçük değil); laboratuvarın `ruin_during_path_probability` adı aynı
  değerle korunur; CLI nihai sermayeyi tek değer basar. Canlı pariteli 1h event_breakout'ta %1,2 (eski 0,0).
- **BULGU-95-1 — iki belgelenmiş komut çalışmıyordu:** `tools/sync_mainhub.py` docstring'i olmayan bir kontrol
  bayrağını ve olmayan bir test dosyasını; pano ipucu `dolt_options`'ın sahip olmadığı bir CLI bayrağını gösteriyordu
  (artık `bulk_csv('volatility_history')`). Tur-61 kapısı yalnız MODÜL varlığını soruyordu; yeni kapı bayrakları ast ile
  sınar ve sarmalayıcıların `& $venvPy -m …` / `@("-m", …)` biçimlerini de okur.
- **Canlı pariteli (kill-switch KAPALI):** dört portföy de zararda (1h ensemble −%85, alfa NW-t −3,81); aynı pencerede
  SPY al-tut +%45,6, eşit-ağırlık al-tut +%125,6. Maliyet stresi: yalnız 1h event_breakout maliyet öncesi pozitif
  (+%210, survivorship'li evren), başabaş kayma ≈ %0,05/taraf (varsayılan %0,1). Walk-forward (1d, 29 sembol, 581 fold):
  verimlilik medyanı 0,03, ROBUST 0/29, al-tutu geçen 2/29.
- **AI karar katmanı:** Tur-92 formülüyle ilk süpürmede de hiçbir λ yön üretmiyor; LONG/SHORT %95 ÜST sınırları
  0,4953 / 0,4888 → kapı-a iki yönde kapalı, ABSTAIN kanıtla tutarlı.
- **Benchmark:** makro sepet dünle birebir; mikro süreler aynı venv'li 19 Eylül'e göre %0–43 yavaş, koşular %23–35 arka
  plan yükü altında → kod gerilemesi iddiası kurulamaz. Thread ölçeklenmesi tüm koşularda < 1× (GIL; canlı tarama ağ
  ağırlıklıdır). Dünkü deneyin benchmark'ı sistem Python'uyla koşmuş (pandas 3.0.2/numpy 2.3.1 ↔ venv 3.0.3/2.4.6).
- **Kilit:** `tests/test_tur95_test_backtest_kapilari.py` (17 test: CLI kapısı + kendi kontrolü, yol-içi iflas, kill-switch
  anı kuraldan BAĞIMSIZ çapayla, sentetik ALTIN değer + belirlenimcilik), **mutasyon 13/13** (ilk geçişte portföy kayık
  indeks mutasyonu KAÇTI → öz-referanslı test düzeltildi). Tur-67 denklik tanığının iflas satırı yeni tanıma çekildi.
  Son doğrulama tam paket **5.107/0/1 atlandı** (232 dosya), kapsama toplam %71,1 (görünürlük); ruff + `mypy .` temiz.

**AI'LAR ↔ ALGORİTMALAR AYRILDI, 14 PİYASA GRUBUNDA KOŞULDU (Tur-96, 2026-09-25).** Rapor:
`docs/TUR96_RAPORU_2026-09-25.md`. Hedef *"AI'ları ve algoritmaları ayır, tüm para piyasalarında koştur, en iyi nerede;
veri kaynağı bul"*. Kenar iddiası YOK; canlı karar yolu, kural, eşik ve `.env` DEĞİŞMEDİ (yeni kod yalnız `research/`).

- **Ayrım `research/coklu_piyasa.py`** (makine-okunur, karnede `ayrim`): ALGORİTMA = öğrenmeyen kural (13; 10'u koşuldu —
  `pead`/`insider`/`sentiment` ABD'ye özel veri) · AI = veriden öğrenen/uyarlanan model (19; 14'ü koşuldu — Bayes karar,
  kalibrasyon, epizodik hafıza, LLM ve DCC koşulmadı, nedenleri katalogda) · `MODUL_SINIFI`: nihai/ai + nihai/signals'ın 49
  modülü = 24 AI · 22 araç · 3 algoritma. `llm_event` bir strateji kipi ama DİL MODELİ sinyali → AI tarafında.
- **Piyasalar `research/piyasalar.py`:** 14 grup, 124 enstrüman (121 kullanıldı), yfinance günlük, önbellek OneDrive DIŞI
  (`PIYASA_CACHE_DIR`), fiyat ilk kapanış = 100'e ölçekli (tam sayı hisse), `aralik_onar` tek-bar yanlış baskı onarımı
  (kripto hariç). Kaynak tablosu çağrılarak ölçüldü: yfinance + Binance açık REST çalışıyor, Stooq HTTP 404.
- **Koşu** (`--asama 1|vol|2|3|rapor|ozet`): algoritmalar backtest motoruyla (AI kapısı ve kill-switch KAPALI); AI'lar
  kendi düzeneğiyle — piyasa başı havuzlanmış YÜRÜYEN eğitim (lojistik/GBM/XGB/deep_mlp/quant_transformer), çevrimiçi
  FTRL + Hedge (2 bar gecikmeli güncelleme), HMM, Avellaneda-Lee, GARCH/GJR/HAR (EWMA'ya karşı QLIKE), canlı modeller
  yeniden eğitilmeden. Pencere 2018-01-02→2026-09-25, maliyet %0,1/taraf, 280 yön hücresi → Bonferroni |t| ≥ 3,75.
- **Sonuç:** tek gerçek beceri OYNAKLIK — en az bir oynaklık AI'ı 12/14 piyasada EWMA'yı anlamlı yeniyor (en güçlü ABD,
  Türkiye, Avrupa; kripto ve Forex anlamsız). Yön AI'larının enstrüman-içi AUC'si yalnız kripto/Çin-HK/Kore/Türkiye/
  Hindistan'da 0,52-0,54 (ABD/Japonya ~0,50), ama 1 gün gecikmede büyük ölçüde siliniyor ve günlük ~%70 devirle maliyet
  onu yiyor; Bonferroni'yi geçen 27 hücrenin HEPSİ negatif alfa. Algoritmalarda tek olumlu bölge kripto breakout
  (event_breakout 1,02 → +1 bar 0,73 → 2018 PIT evreni 0,73, 2022-26 0,53, t < 3,75) = ADAY. Canlı meta kapı rastgele
  kapıya karşı 8/14 (yazı-tura); canlı 1m sinir ağları günlük barda 11/14 piyasada %0 dağılım-içi.
- **BULGU-96-1..11** raporda: havuz AUC'si kesitsel taban oranını beceri sayıyordu (`auc_ic`) · HAR'ın başarısızlığı RV
  vekilinin (FX aralığı r²'nin 2,1 katı, SHEL.L 56 katı) · bar içi yanlış baskı kapanış kuralına görünmez · FX'te işlemsiz
  koşu koşucuyu düşürdü · havuz işçisinde monkeypatch zincirlendi ve gecikme testini tersine çevirdi · transformer 14
  piyasada sabite çöktü ve "en iyi AI" görünüyordu (`cokmus`) · yükselen gün oranı sürüklenme değildir.
- **Kilit:** `tests/test_tur96_coklu_piyasa.py` (49 test), **mutasyon 18/18**; `.env.example` `PIYASA_CACHE_DIR`/
  `PIYASA_SONUC_DIR` (Tur-61 kapısı yakaladı). Tam paket **5.170/0/1 atlandı**, `ruff check .` + `mypy .`
  (219 dosya) temiz. Karne `data/coklu_piyasa_karnesi.json`. Commit YOK.

**ALGORİTMALAR QUANT TARAFINA KAYDIRILDI; ML QUANT ÇEKİRDEĞE YARIŞTIRILDI (Tur-97, 2026-09-26).** Rapor:
`docs/TUR97_RAPORU_2026-09-26.md`. Hedef *"algoritmaları quant tarafına fazlaca kaydıralım, sonra makine öğrenimini
geliştir"*. Kenar iddiası YOK: `quant_trend` al-tutu geçmiyor; kanıtlanan, canlıdaki TA algoritmalarından iyi olması.

- **`nihai/signals/quant.py` (YENİ, algoritma sınıfı) + kip `quant_trend`.** Skor = 21/63/252 GÜNLÜK getiri
  işaretlerinin ortalaması (zaman serisi momentumu); σ = EWMA(0,94) günlük; stop 3σ; sabit hedef YOK, σ-mesafeli
  izleyen stop; uzun-yalnız varsayılan (`QUANT_KISA`). TA kapıları (EMA50/ADX/RSI freni), TA kalite eşiği ve opsiyon
  zinciri quant kipine uygulanmaz. Ufuklar GÜN cinsindendir: canlı 1h tarama günlük barları `generate_signal(df_daily=…)`
  ile ayrıca alır (`main._gunluk_barlar`, sembol × ET günü başına bir çekim); saatlik barla çağrılırsa
  `QUANT_GUNLUK_BAR_GEREKLI`. `BASE_STRATEGY_MODES`e ALINMADI (1h walk-forward + Tur-96 Hedge uzman listesi).
- **Pozisyon kuralları (`position_manager`, canlı + backtest ortak):** quant pozisyonu (etiket `is_quant_label`)
  sabit hedefle çıkmaz, izleyen stop = tepe − başlangıç riski; seans sonu düzleşme ve 90 dk zaman stopundan MUAF
  (günlük ufuklu pozisyon her gün kapanıp açılırdı). Backtest artık pozisyona açan stratejinin etiketini yazar.
- **Ölçüm (`research/quant_kayma.py`, karne `data/quant_kayma_karnesi.json`):** Tur-96 motoru/verisi + canlı 30 sembol
  (`ABD_izleme`). Eşleştirilmiş fark quant − event_breakout 2018-26'da 14/15, hiç bakılmamış 2009-17'de 15/15 piyasada
  olumlu; canlı evrende ÖD t=**3,84** (Bonferroni 3,14). MDD 14/15 piyasada küçük; 11 piyasada 10 TA algoritmasının
  hepsini geçiyor; +1 bar gecikmede dayanıklı (ABD 0,54→0,50). Al-tuta karşı alfa yok (beta ~0,13; yalnız kripto).
- **Canlı:** `.env` `ACTIVE_STRATEGY=quant_trend` (yedek `~/nihai_hist/env_yedek_tur97_*`); meta kapı quant sinyaline
  uygulanmaz (BULGU-97-4). **Ana süreç yeniden başlatılmadı** (izin reddi) → bir sonraki başlatmada devreye girer.
- **ML (`research/quant_ml.py`, karne `data/quant_ml_karnesi.json`):** havuzlanmış panel, yıllık yürüyen eğitim, iki
  pencere (2018-26 ön-kayıtlı · 2011-17 bakılmamış). Trend: lojistik günlerin %99-100'ünde "uzun" = sürüklenme öğrenmiş
  (BULGU-97-6); sürüklenmesi çıkarılmış zamanlama modeli kuralı 1/14 piyasada geçiyor → ML trend zamanlamasına beceri
  eklemiyor. Oynaklık: QLIKE'ta EWMA'yı 14/14 (2018-26; Tur-98 tazelenmiş önbellekte 13/14) ve 10/14 (2011-17), HAR'ı 11/14 yeniyor; ama ön-kayıtlı iki
  ekonomik kapıdan da geçmedi (günlük dengelemede devir 3-8×; `quant_trend` motorunda 6/14 ve 8/15) → çekirdeğe
  BAĞLANMADI, ADAY (risk-ayarlı ölçüyle yeniden ön-kayıt).
- **Veri/değerlendirme hijyeni:** `piyasalar.bayat_bar_maskesi` → `coklu_piyasa.hazirla` yfinance tatil kopya
  satırlarını atar (BULGU-97-5, 2.197 satır); `coklu_piyasa._ai_satiri` `uzun_payi` + `tek_yonlu` taşır ve tek yönlü
  modeli "en iyi AI"dan çıkarır (Tur-96 karnesinde 19 hücre; en iyi yön AI'ı sütunu değişmedi).
- **Canlı parite (BULGU-97-8):** quant bir DURUM sinyalidir; sembol başına küme sınırı (2) canlıda aynı bahsi ikiye
  katlıyordu (replay 2026-09-25: iki sembolde 3 pozisyon). `scan_symbol` quant kipinde açık pozisyonu olan sembolü bar
  çekiminden önce atlar (`QUANT_POZISYON_VAR`). **Test yalıtımı (BULGU-97-9):** `conftest.py` `ACTIVE_STRATEGY`yi kod
  varsayılanına sabitler — paket operatörün `.env`'ini izliyordu ve dört test kırmızıya düştü.
- **Kilit:** `tests/test_tur97_quant_kayma.py` (33) + `tests/test_tur97_ml_hizmet.py` (11), **mutasyon 33/33**. Commit YOK.

**ML SORUNLARI ONARILDI, KARAR MEKANİZMASINDA YENİDEN SINANDI — İKİ ADAY DA KAPIDA KALDI (Tur-98, 2026-09-26).**
Rapor: `docs/TUR98_RAPORU_2026-09-26.md`. Hedef *"makine öğrenmesinin sorunlarını çözer ve geliştirir misin"*. Kenar
iddiası YOK; canlı karar yolu, kural ve `.env` DEĞİŞMEDİ (yeni kod yalnız `research/`).

- **`research/quant_ml_motor.py` (YENİ; ön-kayıt docstring'de, çıktıdan ÖNCE yazıldı).** Tur-97'nin iki ML kusuru kök
  nedeninden onarıldı ve motorda sınandı. K1: σ modeli ertesi günün r²'sini öngörüyordu, kararın ufku haftalar →
  **σ_ML21** (hedef ln ort. r², t+1..t+21). K2: yön modeli kuralın YERİNE yön tahmin edip sürüklenme öğreniyordu →
  **meta kapı** (ML kuralın ARDINDA, yalnız girişleri eler; + piyasa rejimi özellikleri). Birincil ölçü Sharpe farkı +
  Memmel z; kapı: iki pencerede (2011-17, 2018-26) ≥ 10 piyasa ΔS > 0 VE ABD_izleme z > 1,645. Bilgi kolları HAR21/ML1,
  sonradan teşhis kolu EWMA×1,091 (kapısız).
- **Sonuç:** σ_ML21 21 günlük QLIKE'ta EWMA'yı 13/15 ve 15/15 (izleme DM t −7,45/−8,06), HAR'ı 13/15 ve 14/15 yeniyor;
  motorda ΔS > 0 10/14 · 11/15, ama izleme +0,05 Sharpe (z 0,82/0,81; ayırt edilebilir en küçük fark ≈ +0,10) →
  **bağlanmadı**. MDD kazancı (27/29) büyük ölçüde maruziyet: yalnız genişletilmiş EWMA 25/29 (BULGU-98-3). Meta kapı
  %14-24 eler (yozlaşmadı), ΔS > 0 5/14 · 10/15, izleme z 0,01/−1,30 → **bağlanmadı**.
- **BULGU-98-1 / 98-6 — ölçüm hijyeni.** Tur-97 `quant_kayma` sonuç dosyasını yalnız sembolle adlandırıyordu: 7 sembol
  hem ABD havuzunda hem izlemede → ABD event_breakout 7 sembolde çift sayıldı (2.246 ↔ 1.309 işlem; ABD fark t 1,11 →
  1,02, başka alan değişmedi) → `quant_kayma.dosya(piyasa, sembol, önek)`. Girdi tarafı: izleme `indir`i havuzun 7
  sembolünü yeniden çekip Tur-96/97 girdisini değiştirmişti → `quant_kayma.indir()` havuz sembolünü ATLAR.
- **Tur-96 karnesi bayat bar süzgeciyle yeniden kuruldu (Tur-97 §9 #4, 57 dk):** manşetler dayandı (oynaklık AI'ı
  12/14 piyasada anlamlı; Bonferroni'yi geçen 24 yön hücresinin hepsi negatif alfa). BULGU-96-9'un Türkiye yarısı
  ÇÖKTÜ: taban kuralı tatil satırlarıyla eşiğin altına itiliyordu (uzun payı 0,42 → 0,76, Sharpe −0,80 → +0,94;
  BULGU-98-7). Tur-97 karneleri aynı önbellekle yeniden üretildi (quant manşetleri aynı; ML σ QLIKE 14/14 → 13/14).
- **Kilit:** `tests/test_tur98_ml_onarim.py` (16), **mutasyon 20/20**. Tam paket 5.243/0/1 atlandı. Commit YOK.

**ML'İN KARAR MEKANİZMASI — BOYUT KARARI, BİRLEŞİK KAPI, AI KÖPRÜSÜNÜN KAPSAMI (Tur-99, 2026-09-26).** Rapor:
`docs/TUR99_RAPORU_2026-09-26.md`. Hedef *"makine öğrenmesinin karar verme mekanizmasını geliştir, sorunlarını çöz"*.
Kenar iddiası YOK; canlı kural/eşik/`.env` DEĞİŞMEDİ. Canlı kodda tek değişiklik AI köprüsünün kapsam kuralı.

- **`research/quant_karar.py` (YENİ; ön-kayıt docstring'de, MDE koşudan önce).** ML çıktısının karara girdiği yer
  sınandı: `quant_trend` σ'yı yalnız girişte okur (adet = risk $ / 3σ̂, pozisyon ömrü boyunca sabit). Laboratuvar
  motorun pozisyon yöneticisinden İŞLEM DEFTERİ yakalar (`defter()`), günlük getiriyi defterden kurar (`simule`);
  SABİT replika motorun MTM getirisini 3,7e-9 farkla veriyor (ön-koşul). Tek değişken VY: açık pozisyonda
  u* = u_giriş·σ̂_giriş/σ̂_{t−1}, bant 0,25, maliyet %0,1, tavan 2× (motorun alım gücü). H3a EWMA-VY − EWMA, H3b
  ML21-VY − EWMA-VY. Kapı: korelasyon düzeltmeli birleşik Z > 1,645 (15 piyasa) + ≥ 10 olumlu piyasa + izleme ΔS > 0,
  iki pencerede de.
- **Sonuç: aday yok.** H3a Z −1,62 (2011-17, 3/14) ↔ +2,80 (2018-26, 11/15; USDTRY sürüklüyor) → ters işaret. VY
  maruziyeti 29/29 hücrede artırıyor, MDD'yi 27/29 hücrede büyütüyor (BULGU-99-3). H3b Z 1,52 / 1,59 (10/14, 11/15,
  izleme iki pencerede olumlu): σ_ML21'in karar değeri üç ölçümde de aynı yönde, +0,03-0,06 Sharpe, eşiğin altında.
- **Birleşik kapı (BULGU-99-4).** MDE koşudan önce Tur-98 serilerinden: yalnız canlı evren ≈ 0,15, birleşik 0,065-0,072.
  Sonradan teşhis: Tur-98 σ_ML21 kolu birleşik Z 1,33 (2011-17) / 2,41 (2018-26); hüküm değişmedi.
- **BULGU-99-1/2 — canlı AI köprüsü.** `decision.decide`'ın önseli 1 dk tahmin servisinin 5 barlık yön isabeti
  (LONG n 2.348, ort. 0,476); strateji anahtarı yokken sessizce ona düşüyordu. Ölçüldü: 23/23 quant adayına ABSTAIN
  (enforce'ta 23 veto). Kapı da isabet soruyor; quant_trend isabeti 30 hücrenin 29'unda %33-45, PF medyanı 23/30 hücrede
  > 1. `ai_advisor.kapsam_disi(sig, strategy)`: quant adayında motor HİÇ çağrılmaz, `KAPSAM_DISI_QUANT` (veto yok,
  boyut 1,0), `status()["n_kapsam_disi"]`; pano "Quant Beyin" kartı aynı fonksiyonu kullanır. Quant dışı davranış aynı.
- **Girdi parmak izi (Tur-98 §9 #4 kapandı).** `coklu_piyasa.girdi_parmak_izi`: sembol başına satır/son tarih/SHA-256
  (indeks ns'ye çevrilerek) + özet. Beş karne yazıcısı taşıyor; dört eski karne yeniden üretildi, 14.566 alanda 0 fark.
- **Kilit:** `tests/test_tur99_karar_mekanizmasi.py` (21), **mutasyon 24/24** (ilk koşu 23/24: pandas 3 `bdate_range`
  zaten "us" üretiyordu, birim testi dejenereydi). Tam paket 5.265/0/1 atlandı. Commit YOK.
- **Sonrası — σ_ML21 İLERİ sınaması kilitlendi (2026-09-26).** `research/ileri_sigma_ml21.py` + ön-kayıt
  `docs/ONKAYIT_SIGMA_ML21_2026-09-26.md` (sha kilitte; değişiklik gerekirse `…_EK<N>_<tarih>.md`). Yalnız 2026-09-27
  sonrası veri; kollar `ewma` ↔ `ml21` (σ yalnız girişte, canlı mekanizma), 15 piyasa / 149 enstrüman dondurulmuş.
  H1 öngörü (QLIKE; karar 2027-09-30, güç ≈ %85) · H2 karar (birleşik Sharpe z; karar 2031-09-30, +0,05 etkide güç
  ≈ %43, %80 güç ≈ 14 yıl). H1 geçmezse aday düşer. Yıllık model Y'deki ilk `guncelle`de bir kez eğitilip dondurulur;
  KANARYA (her piyasanın ilk sembolü, kilit günü verisinin kopyası) kod/ayar/kütüphane değişikliğinde deftere yazmayı
  durdurur. Defter/model `data/ileri_sigma_ml21/`, veri `~/nihai_hist/ileri_sigma_ml21_veri` (gitignore'lu).

**ML OYNAKLIK MODELİ DERİNLEŞTİRİLDİ — DÖRT ONARIMIN DÖRDÜ DE KÖTÜLEŞTİRDİ, KİLİTLİ TARİF DOĞRULANDI (Tur-100, 2026-09-27).**
Rapor: `docs/TUR100_RAPORU_2026-09-27.md`. Hedef *"makine öğrenimini detaylı ve derin bir şekilde geliştir"*. Kenar iddiası
YOK; canlı kural/eşik/`.env` ve kilitli ileri sınama DEĞİŞMEDİ (yeni kod yalnız `research/quant_ml_derin.py`).

- **Ön-kayıt (docstring, test yılı görülmeden):** σ_ML21'in dört açığı merdivenle sınandı: ml21 (Tur-98 birebir) → k1
  (zamansal erken durdurma: son 365 gün doğrulama, 31 gün arınma, ≤ 600 it., tüm veriyle yeniden eğitim) → k3 = k1 + 5
  bilgi özelliği (aynı piyasanın ve 14 piyasanın önceki gün ort. ln RV5/RV22, ln VIX², ln VIX/VIX3M; hepsi KESİN ÖNCEKİ
  tarihten) = ADAY; k2 = k1 + gamma kaybı (bilgi). G1 öngörü (≥ 10/15 piyasa + izleme DM t < −1,645, iki pencere), G2
  motor (birleşik Z, ileri sınamanın H2 kapısı). Adaptasyon: çeyreklik · yuvarlanan 2/5 yıl · donmuş 2011/2018 · havuzun %25/%50'si.
- **Sonuç: tarif değişmez.** k3 σ_ML21'i 0/15 ve 1/15 piyasada yendi (G1 yok); motorda EWMA'ya Z 2,30 ↔ 1,09 (G2 yok).
  k1 3/15 · 1/15; gamma kolu dağıldı (5.879 sonsuz öngörü). ml21 Tur-98 ile göreli fark 0,0, motor 546 kıyasta 0,0;
  ileri sınamanın kanaryası tur sonunda SAĞLAM.
- **BULGU-100-1/2:** kilitli tarifin erken durdurması sızıntılı (16/16 yılda `n_iter_` 300 = tavan, %10 veri atılıyor)
  ama zararsız (%100 veri + sabit 300 ayırt edilemez; eğri 100-600 düz). "Sızıntısız" tek yıllık doğrulama seçimi 20-598
  arasında savurdu; COVID yılında doğrulanan 2021 modeli tek yılda +0,215 → 2018-26 kaybının neredeyse tamamı.
- **BULGU-100-3/4:** gamma (QLIKE'la hizalı) HistGBR'de Newton adımıyla taşıyor. Piyasa geneli bilgi iyi tarifin üstünde
  de zarar (0/15, izleme t +2,37 2011-17; permütasyon önemi ≤ 0,01): ceza eğitim günüyle eriyor (0,049 → 0,007), 2020
  krizinde yardım ediyor (−0,034).
- **BULGU-100-5 — "ML zamanla daha iyi olur mu?":** daha çok veri yardım ediyor (son 2/5 yıl ve havuzun %25/%50'si kötü,
  azalan getiri); güncellenmeyen model bayatlıyor (1 yılda +0,004, 7-15 yılda +0,01-0,035); çeyreklik eğitim yıllıktan
  iyi değil. İleri sınamanın politikası (yıllık, genişleyen) doğru.
- Sonradan teşhis kolları (kapısız): `t_tam300`, `t_n100`, `t_n600`, `t_ml21_bilgi`; ön-kayıtlı karne alanları teşhis
  eklenince değişmedi (alan alan kıyaslandı).
- **Kilit:** `tests/test_tur100_ml_derin.py` (20 test), **mutasyon 29/29** (ilk koşu 27/28: dejenere kurgu). Tam paket 5.298/1 atlandı; tek kırmızı
  canlı ana sürecin `state.json` yazımıyla yarışan replay hash kapısı (tek başına 14/14). Commit YOK.

**QUANT ÇEKİRDEĞİNİN KENDİ KARARLARI — ÇIKIŞ, SKOR BOYUTU, PORTFÖY; CANLI SADAKAT (Tur-101, 2026-09-27).** Rapor:
`docs/TUR101_RAPORU_2026-09-27.md`. Hedef *"projenin quant kısmını aşırı detaylı bir şekilde geliştir"*. Kenar iddiası YOK;
`quant_trend` çekirdeği (sinyal/stop/boyut) ve `.env` DEĞİŞMEDİ. Değişen: canlı yolun ölçülenle hizası + backtest seans sayaçları.

- **`research/quant_cekirdek.py` (YENİ; ön-kayıt docstring'de).** Teşhis (getiri görülmeden): D1 izleyen 3σ stopun beklenen ömrü
  (a/σ)² ≈ 9 gün ↔ sinyal ufku 21-252 · D2 skor büyüklüğü boyuta girmiyor (HOP 2017) · D3 FX'te "uzun" kotasyon yönü · D4 canlı
  kapasite · D5 sembol başı ölçüm ≠ tek portföy. Bölüm I (sembol başı motor, 15 piyasa, A 2001-08 YENİ · B 2009-17 · C 2018-26):
  q1 sinyal çıkışı (izleyen kapalı, 3σ sabit), q0s |skor| boyutu, q2 paket; bilgi q1n (stopsuz), FX simetrik; kapı G üç pencerede.
  Bölüm II (ABD_izleme tek portföy, canlı tavanlar, `backtest_portfolio` kopyası): liste sırası · 8 rastgele sıra · PLAN (skor sıralı,
  sektör tavanlı, m = min(1, (C − N_açık)/Σn_i)); kapı P. Ek ön-kayıt (A/B/C görüldükten SONRA): Z 1990-2000 doğrulaması.
- **Hüküm: çekirdek DEĞİŞMEZ, plan BAĞLANMADI.** H1 (sinyal çıkışı) A z 3,30 (12/14) · B z 2,04 ama izleme ΔS −0,01 · C z −0,10 (7/15);
  H2 (skor boyutu) hiçbir pencerede (B 4/15, z −1,47). P A/B/C'de geçti (plan − liste ΔS +0,24/+0,41/+0,15, 8 sıranın hepsinden iyi)
  ama Z'de MDD koşulu **0,13 puanla** kaldı → ek ön-kayıt gereği bağlanmadı. FX simetrik B/C'de kötü (ΔS −0,24/−0,26).
- **BULGU-101-2 (D5) — sembol başı ölçüm canlı portföyü temsil etmiyordu.** ABD_izleme q0: sembol başı EW Sharpe 0,97 (C) ↔ aynı sembollerin
  tek portföyü (canlı tavanlar) 8 sırada **0,38–0,57**; B 0,99 ↔ 0,50–0,67. q1'de B/C'de kayıp yok (0,92 ↔ 0,86–1,03). q1'in sıra
  aralığı q0'ınkiyle dört pencerenin dördünde kesişmiyor (Z: 1,21–1,42 ↔ 0,61–0,88; Z2: q1 plan − q0 plan z 1,84). Aday mekanizma:
  izleyen stopun kısa ömrü (BULGU-101-3: medyan tutma çoğunda 8-10 gün, çıkışların %62-91'i ≤ 2 barda yeniden giriş) paylaşılan
  kapasitede slotu başka sembole kaptırır (replay: JPM stoplandı, aynı dakikada NBIS/TSM/V girdi, JPM ertesi gün 1 hisse aldı).
  Sonradan gözlem → ileri sınama ÖNERİSİ (rapor §10), canlı değişmedi.
- **BULGU-101-1 (D4) — canlıda quant portföyünü tarama sırası dolduruyordu.** 520 sembol (odak 30 + geniş 490) ↔ tavan ~2× nominal:
  replay 2026-09-25 açılan 9 pozisyon A, AAPL, ABBV, ABNB, ACGL, AMD, AMZN, ARM, FCX (6'sı ölçülmemiş geniş katmandan, FCX 1 hisse),
  sonra 1.777 alım gücü reddi. `main.scan_symbol`: quant yalnız ODAK katmanında (`_quant_odak`, 60 sn önbellek; geniş katman bar
  çekilmeden durur, `QUANT_ODAK_DISI` sembol başına günde bir satır). Sonra 10 pozisyonun hepsi odaktan.
- **BULGU-101-4 — stoplanan quant pozisyonu canlıda aynı gün yeniden açılıyordu** (soğutma GİRİŞTEN 2 sa; backtest ertesi açılışta girer).
  `_apply_close_bookkeeping` quant çıkışını damgalar, `scan_symbol` aynı ET gününde `QUANT_AYNI_GUN` (TA soğutması aynı).
- **BULGU-101-5 — backtest günlük sayaçları hiç sıfırlamıyordu** (canlı her seans sıfırlar; varsayılan `last_session_date` BUGÜN).
  `.env.example` frenleriyle (soğutma 3, günlük %10) SPY quant 2018-02-07'den sonra 8,5 yıl tek işlem açmadı; operatör `.env`'i iki
  freni kapattığı için görünmüyordu. İki hisse backtest'i `last_session_date = date.min` + bar başına `roll_session_if_new_day`;
  operatör ayarında bit-denk (Tur-97 paritesi 0,0). Opsiyon backtest'leri `PositionManager` kullanmaz, frenler orada hiç çalışmaz.
- **Ölçüm hijyeni:** q0 ↔ Tur-97 dosyaları 149 enstrümanda 0,0 · portföy kopyası ↔ `backtest_portfolio` 0,0 $ (1.675 işlem) · 48
  portföy dosyası yeniden koşuda 0,0 · karne yeniden üretimi yalnız `uretim` farklı. Canlı odak (pin'ler) ↔ ölçülen ABD_izleme
  (`.env`) 30'da 24 ortak.
- **Canlı:** iki sadakat düzeltmesi CANLIDA: ana süreç 11:13'te eski kodla açılmıştı, operatör onayıyla 2026-09-27 13:23'te bekçi
  üzerinden yeniden başlatıldı (tek süreç, 0 açık pozisyon). İlk quant seansı 2026-09-28 TR 16:30.
- **Kilit:** `tests/test_tur101_quant_cekirdek.py` (21 test), **mutasyon 22/22** (ilk geçiş 18/22: iki CRLF deseni atlandı, iki dejenere
  kurgu kaçtı, bir "yakalama" sözdizimi hatasıydı). Tur-97 `scan_symbol` testi odak kümesi + `symbol_cooldowns` ile güncellendi.
  Tam paket 5.320 geçti / 1 atlandı / 1 kırmızı — kırmızı `test_tur77_replay` üretim dosyası hash kapısı: canlı ana süreç `data/state.json`ı paket sırasında yazdı (12:43:20), test tek başına 14/14 geçti (Tur-100'deki yarışın aynısı); `ruff check .` + `mypy .` (228 dosya) temiz. Commit YOK.

**GÖLGEDEKİ 5 KARAR MEKANİZMASINA AYRI KASA (Tur-102, 2026-09-28).** Rapor: `docs/TUR102_RAPORU_2026-09-28.md`. Hedef
*"gölgede/danışman çalışan 5 mekanizmaya ayrı kasa verelim, canlı veriyle test edeceğim (her kasa 1.000.000 $)"*. Kenar iddiası
YOK; canlı portföy/sinyal/kural/`.env` DEĞİŞMEDİ.

- **`nihai/execution/golge_kasalar.py` (YENİ, KARAR-DIŞI).** Altı kasa (`GOLGE_KASA_BAKIYE`, vars. 1.000.000 $): `kontrol`
  (mekanizmasız taban) · `meta_kapi` · `ai_karar` · `golge_kopru` · `noral` · `olasilik_boyut`. Her kasa mekanizmasının KENDİ
  enforce sözleşmesini uygular (saf `karar()`), kendi `PortfolioState`/`RiskManager`/`PositionManager(record_logs=False)`/
  `StateStore`'u vardır → boyut, tavan, fren, kayma, çıkış canlı motorun kendisi. Karar T kapanışı → dolum T+1 AÇILIŞI (backtest
  sözleşmesi); yalnız KAPANMIŞ bar; sembol başına tek pozisyon; 5 dk'da dolmayan emir iptal. Tahmin servisi `_bar_context`
  sonunda `golge_kasalar.isle(...)` çağırır — yalnız `_cycle_market_open is True`. Model çağrıları kilit dışında; altı kasa +
  `motor.json` TEK anlık görüntüde yazılır. Dosyalar `data/golge_kasalar/` (`<kasa>.json`, `islemler.csv`, `kararlar.csv`).
  Pano **Beyin → Gölge Kasalar** (`/api/golge_kasalar`), CLI `python -m nihai.execution.golge_kasalar`. `GOLGE_KASA=off`
  kapatır; `conftest.py` test sürecinde kapatır.
- **Kurallar:** meta p < `META_GATE_THRESHOLD` veto · AI verdikti ters/ABSTAIN veto, değilse `ai_advisor.enforce_boyut(post)`
  (`advise()` ile TEK kaynak) · köprü: gözlemevi karantinası + Tur-88'de DONDURULMUŞ iki terfi adayı (`KOPRU_KURALLARI`:
  `conf_bar` ≥ 0,60, `value_area_pos` ≥ 1,0; `conf_bar` `PredictionService._yon_guven` tek kaynağından) · nöral p < modelin son
  200 ölçümünün medyanı veto (30'dan önce ve ölçümsüz girmez) · olasılık boyutu = güven çarpanı × `prob_to_size_mult(meta p)`.
  Ölçüm yoksa (nöral hariç) aday geçer — canlı enforce dallarının fail-soft sözleşmesi.
- **BULGU-102-1 — quant geçişi beş mekanizmanın kanıt döngüsünü durdurmuştu.** `quant_trend` 1m barda hep WAIT → tahmin servisi
  2026-09-27'den beri TEK aday üretmedi; meta/nöral/AI ölçümleri ve `live_features_log` aday satırları kesildi, canlı süreç 282M
  modeli hiç yüklemedi (GPU 693 MiB). Kasalar adayı kendileri üretir (`GOLGE_KASA_STRATEJI`, vars. `event_breakout`); tahmin
  yüzeyinin kendi aday stratejisi DEĞİŞTİRİLMEDİ (operatör kararı).
- **Duman ölçümü (hub 1m, 8 sembol × 150 bar): 159 aday** → giriş kontrol 127 · olasılık 93 · nöral 29 · köprü 13 · meta 12 ·
  **AI 0** (159/159 ABSTAIN, BULGU-102-4); 31 ms/bar. Nöral p 0,484-0,487 (std 0,001) → sabit eşik dejenere (BULGU-102-2);
  köprünün kendi enforce'u boş → kasa kontrolün kopyası olurdu (BULGU-102-3).
- **Tek yazar + ayrı koşucu:** kasa dizinine yalnız `motor.lock`u alan süreç yazar; `python -m nihai.execution.golge_kasalar
  --kos` canlı ana sürece dokunmadan kendi tahmin servisini `PREDICT_TRACK=false` ile açar. **BULGU-102-6:** `-m` modülü
  `__main__` olarak yükler, kanca kanonik modülü İKİNCİ kez içe aktarır → aynı süreçte iki motor, kanca kasalara hiç dokunmadı;
  `--kos` kanonik modülde koşar. **BULGU-102-7:** 1m stop %0,1-0,3 → %1,5 risk kuralı hep alım gücü tavanına çarpar (canlı:
  IBM 9.104 hisse ≈ 2,0 M$, sonraki adaylar 1-4 hisse, ×0,25'lik kasa AYNI adet) → kasalarda adet = min(risk, özsermaye ×
  `GOLGE_KASA_POZ_PAY` 0,10 / fiyat) × çarpan (`kasa_adedi`; canlı ana yol değişmedi).
- **Canlı:** ana süreci durdurmam REDDEDİLDİ; operatör projeyi 18:31'de kapattı, "devam et" üzerine `proje-ac.ps1` ile 18:35'te
  yeni kodla açıldı → ilk 6 dk'da 36 aday, 10 pozisyon, hata yok. BULGU-102-7 düzeltmesi sonrasında geldi: operatör bir kez
  daha yeniden başlatıp `data/golge_kasalar/`i sıfırlamalı.
- **Kilit:** `tests/test_tur102_golge_kasalar.py` (31), **mutasyon 24/24** (ilk geçiş 19/20: dolum-kapanış mutasyonu %0,5'lik
  dejenere kurguda kaçtı). Tam paket 5.353/0/1 atlandı (son hâl, canlı süreç koşarken); ruff +
  `mypy .` (229) temiz. Commit YOK.

**EMİR AKIŞI — footprint · hacim profili · baloncuklar · likidite ısı haritası (Tur-103, 2026-09-28).** Rapor:
`docs/TUR103_RAPORU_2026-09-28.md`. Hedef *"projeye heatmap, footprint, volume profile, bubbles (deep charts) mümkünse
ekler misin"*. Kenar iddiası YOK; KARAR-DIŞI görselleştirme (sinyal/eşik/META_GATE/execution'a bağlanmaz — testle).

- **Mümkün olan (kaynaklar çağrılarak ölçüldü):** hisse işlemleri Alpaca — SIP tüm piyasa ama 15 dk gecikmeli, IEX
  canlı ama hacmin ~%3-4'ü; ABD borsaları agresör tarafını bildirmez → **tick kuralı (TAHMİN)**. Ücretsiz L2 YOK →
  **hissede ısı haritası YOK** (panoda nedeniyle yazılı). Kripto (…USDT) Binance anahtarsız: gerçek agresör tarafı +
  1000 kademe defter (BTC ±150-300 $); defter geçmişe dönük çekilemez → ısı haritası KAYITLA birikir.
- **`nihai/data/orderflow.py` (YENİ):** `tick_kurali` (ilk fiyat değişiminden önceki işlem 0 = sınıflanamadı) ·
  `footprint` (bar × kademe alış/satış, Δ/CVD/POC, çapraz dengesizlik ≥3× bit maskesi; karşı taraf SIFIRSA işaret yok;
  pencere profili + %70 değer alanı `volume_profile._genislet` konvansiyonuyla) · `baloncuklar` (ardışık aynı yönlü ≤1 ms
  baskı = tek agresif emir, VWAP) · `isi_haritasi` (kademe eşlemesi footprint'le AYNI `rint(fiyat/adım)`). Hisse: ham REST
  (alpaca-py DEĞİL: NVDA 60 dk 282 bin işlemde pydantic 13 sn) + 1 dk bar sayımıyla PLAN (en yeniden geriye 300 bin
  işlem, ~eşit ≤8 paralel parça; NVDA 60 dk 277 bin işlem **4,4 sn**) + (sembol, feed) başına ARTIMLI önbellek (≤10 dk
  artımda plan isteği yok) + kapalı seansta son işleme çapa. Kripto: `CryptoRecorder` (60 dk geri doldurma 16'şarlı
  paralel, 3 sn'de bir defter + yeni işlem, 45 dk görüntü, 10 dk boşta durur, en çok 2 kaydedici, ≈1.080 ağırlık/dk).
- **`nihai/web/api_orderflow.py` (YENİ)** `GET /api/orderflow` (fail-soft, `def`) + pano **Analiz → Emir Akışı**
  (`m11-emir-akisi`, Canvas 2D: ısı · footprint hücreleri · baloncuk · profil/POC/VAH/VAL · Δ/CVD · imleç ipucu ·
  tekerlek/sürükle yakınlaştırma) + Hisse Detay'da "Emir Akışı" düğmesi.
- **ÖLÇÜLDÜ — tick kuralı:** Binance'te gerçek tarafa karşı BTC %97,5 · ETH %97,2 · SOL %83,6 (hata sıfır-tik
  işlemlerinde); SIP'te Lee-Ready ile UYUM AAPL %84,3 · SPY %79,0 · NVDA %81,4 (hissede doğruluk ölçülemez). Lee-Ready
  EKLENMEDİ (≈3 puan ↔ 2× sayfa bütçesi).
- **BULGU-103-1** zaman ms'ye kırpılıp SONRA sıralanıyordu (API `sort=desc`): SIP işlemlerinin %33-56'sı aynı ms'yi
  paylaşıyor, eski sıra işlemlerin %3,8-8,6'sının tarafını değiştiriyordu (AAPL net Δ %22). **BULGU-103-2** eşit zamanlı
  paralel parçada en yeni parça bütçeyi doldurunca eski parçalar atılıyordu (NVDA 139 bin ↔ 277 bin) → sayımla plan.
  **BULGU-103-3** ısı penceresi son işlemde bitiyor, sonraki defter görüntüsü dışarıda kalıyordu. **BULGU-103-4**
  `parca_plani(hedef=_PARCA_HEDEF)` varsayılanı tanımda bağlandı → testin yaması etkisiz, paralel yol sınanmadı, iki
  mutasyon kaçtı. **BULGU-103-5** sınıflanamayan hacim farktan → `-0.0000`.
- **Değiştirilmedi:** hisse L2/ısı · kripto ısısının disk kaydı (süreç yeniden başlarsa sıfırlanır) · seans başından CVD.
  **Dağıtılmadı:** ana süreç yeniden başlatılınca devreye girer (test sunucusu 8777'de doğrulandı).
- **Kilit:** `tests/test_tur103_emir_akisi.py` (24 test; node ile gerçek biçimli JSON'u çizme dahil), **mutasyon 27/27**.
  Commit YOK.

**EMİR AKIŞINA NET GEX, DELTA/HACİM FOOTPRINT, BÜYÜK İŞLEMLER (Tur-104, 2026-10-01).** Rapor:
`docs/TUR104_RAPORU_2026-10-01.md`. Hedef *"devam et; heat map, volume profile, net gamma exp, delta, footprint, bid trades
eklemek mümkünse"*. Kenar iddiası YOK; KARAR-DIŞI (sinyal/eşik/META_GATE/execution'a bağlanmaz).

- Isı haritası · hacim profili · footprint · Δ/CVD Tur-103'te vardı. Bu tur eksikleri ekledi: **net GEX**, footprint
  **Delta / Hacim** kipi, **Büyük İşlemler** tablosu (+ `min_size` süzgeci). "bid traddes" DeepCharts *Big Trades*
  olarak okundu; taraf = agresör (ALIŞ ask'a vuran, SATIŞ bid'e vuran).
- **`orderflow.gex_toplu`** çok vadeli GEX: her vade `option_flow.dealer_gamma_exposure` ile AYNI spot ve ızgarada
  (±%15, 121 nokta) → strike ve ızgara eğrileri TOPLANIR; flip = toplam eğrinin spota en yakın kökü (doğrusal, %0,25
  ızgara; brentq'e fark 0,015 $). `gex_temizle` IV/OI/strike eksik satırı atar ve SAYAR (`dealer_gamma_exposure` tek
  IV'siz satırda flip ölçmez); tek taraflı vade ATLANIR. Hisse: panonun zinciri (OI günlük), ≤45 gün, en çok 8 vade,
  r=0,04, 10 dk önbellek. Kripto: **Bybit + OKX** anahtarsız (OI coin cinsinden → çarpan 1; vade 08:00 UTC; Deribit
  bu ağdan erişilemiyor; OKX'te SOL/XRP yok), tüm vadeler, 2 dk önbellek. Dealer konvansiyonu VARSAYIMDIR.
- **`GET /api/orderflow/gex`** (`def`, fail-soft, yalnız başarılı sonuç önbellekte) · `/api/orderflow` `min_size` → `en_az`.
  Pano: GEX sütunu + FLIP/CW/PW çizgileri (ekran dışı seviye eksende ▲/▼), GEX bilgi satırı, footprint kip seçici,
  büyük işlemler tablosu, en küçük boy girdisi, ipucunda en yakın strike.
- **ÖLÇÜLDÜ:** BTC (Bybit 726 + OKX 692 kontrat, 24 vade) net **+56,0 M$/%1**, flip 84.031 (spot 84.831), 1,5 sn ·
  ETH +11,0 M$ · SPY (8 vade 10-01→10-16, 0DTE dahil) net **−4,69 Mr$**, flip 766,09 > spot 763,83, 3,1 sn · AAPL +167 M$ ·
  NVDA +1,31 Mr$. Izgara orta noktası = net (birebir); tek vade paritesi net 8 anlamlı basamakta aynı.
- **BULGU-104-1** yükleme sürerken istenen sembol düşüyordu (Tur-103; `if (OF.yukleniyor) return;`): yeni sembol ancak
  sonraki zamanlayıcıda (SIP'te 30 sn) gelirdi → istek BİRİKİR (`OF.tekrar`), bayat yanıt atılır; GEX isteğinde de
  aynısı. Ölçüldü: 3,9 sn. **BULGU-104-2** tek IV'siz satır çok vadeli flip'i silerdi → `gex_temizle` (bugün atılan 0;
  önleyici). **BULGU-104-3** OKX OI listesi vadesi geçmiş kontratları taşır → eşleme canlı `opt-summary`den.
  **BULGU-104-4** `min_size` testi kötü değerleri sahte kurulmadan gönderiyordu; ağ guard'ı "ok: false" döndürdüğü için
  doğrulama silinse de yeşil kalırdı.
- **Kilit:** `tests/test_tur104_gex_ve_buyuk_islem.py` (16 test; node ile yarış + GEX + üç kip), **mutasyon 36/36 (ara geçişlerde iki dejenere kurgu yakalandı: en_az süzgeci hiçbir şeyi elemiyordu, +inf OI satırı yoktu)**.
  Dağıtılmadı: ana süreç yeniden başlatılınca devreye girer. Commit YOK.

**GÖLGE AI KASASINDA ABSTAIN ARTIK VETO DEĞİL (Tur-105, 2026-10-01).** Rapor: `docs/TUR105_RAPORU_2026-10-01.md`. Hedef
*"ABSTAIN veto sayılmasın, AI işlem açabilsin"*. Kenar iddiası YOK; canlı AI köprüsü (`ai_advisor.advise` enforce: ABSTAIN =
veto), canlı portföy, `.env` ve öbür beş kasa DEĞİŞMEDİ.

- **BULGU-105-1 — Tur-102 `ai_karar` kasası bir duvardı.** Canlı enforce sözleşmesini birebir uyguluyordu; prior lo95 LONG
  0,4229 · SHORT 0,4536 → 2026-09-28 → 10-01 arasında 5.531 adayın 5.531'i ABSTAIN, 0 işlem. "Kontrolle fark" (`fark_kontrol_pp`
  = kasa getirisi − kontrol getirisi, yüzde puan) +28,4 pp gösteriyordu: kaybeden akışa girmemek, beceri değil.
- **Kural:** `karar("ai_karar")` yalnız TERS yönü veto eder; ABSTAIN ve aynı yön `ai_advisor.enforce_boyut(post)` ile girer
  (yeni eşik yok). Defterin `ai_veto` kolonu canlı enforce'un cevabını yazmaya devam eder; kasanın hükmü ayrı kolonda.
- **BULGU-105-2 — ABSTAIN kalkınca veto kümesi BOŞALIR.** Kasa `shadow_verdict(side=taraf)` çağırır ve `decide` yönü
  `side_hint`ten alır → ters yön üretilemez. Kasa kontrolle aynı adaylara girer, posteriorun %99,4'ü < 0,50 → boyut çoğunlukla
  ×0,25. Geriye dönük kaba tahmin (kontrol K/Z × çarpan): −%7,2 ↔ kontrol −%28,5. Fark artık MARUZİYETTİR; AI'ın bilgisi
  ancak ×0,25 ölçekli kontrolle ya da rampa > 0,25 alan adayların ayrı K/Z'siyle okunur.
- **Canlı DEĞİL:** kasa motoru ana süreçte (2026-10-01 21:05'te başladı) eski kodla koşuyor; yeniden başlatma + (önerilen)
  `data/golge_kasalar/` yedekleyip altı kasayı sıfırlama operatör kararı.
- **Kilit:** `tests/test_tur105_ai_kasa_abstain.py` (12 test), Tur-102 testinde iki iddia güncellendi, **mutasyon 5/5**;
  tam paket 5.405/0/1 atlandı; `ruff check .` + `mypy .` (231 dosya) temiz. Commit YOK.

**PROJEDEKİ HATA VE TUTARSIZLIKLAR — canlı günlüklerden kök nedene (Tur-106, 2026-10-02).** Rapor:
`docs/TUR106_RAPORU_2026-10-02.md`. Hedef *"projedeki hata ve tutarsızlıkları düzelt"*. Taban temizdi (paket yeşil, ruff/mypy
temiz) → kusurlar canlı günlüklerde, gölge kasa defterlerinde ve 8777 test sunucusunda gezilen panoda arandı. Kenar iddiası
YOK; strateji kuralı/eşik değişmedi. Aynı hedefle kesilen önceki oturumun (`87564c22`) iki düzeltmesi devralındı (106-2/3).

- **BULGU-106-1 — hub keep-alive yarışı.** İstemci (httpx) ve hub (uvicorn) boşta bağlantıyı ikisi de 5 sn tutuyordu; ~5 sn
  ritimli döngüler sunucunun kapattığı bağlantıyı yeniden kullanıyordu (canlı saatte 50-98 "Server disconnected"/WinError
  10053-10054, her biri yfinance yedeği). Aynı süreçte boş uçla deney 0/360 ("çürüdü"); hub ayrı süreçte + CPU yükü ile
  **10-12/1020**, istemci ömrü 3 sn ile **0/990**. `HubProvider` `keepalive_expiry=3.0` (MainHub senkron).
- **BULGU-106-2/3 (devralındı)** — `[state] yüklendi` anahtarı dosya başına (6 kasa yüklemesi `system.log`un %31'iydi);
  hub `ensure_utf8_stdout()` (stderr cp1254 → `hub_server.err.log` ~%26 bozuk satır).
- **BULGU-106-4 — oto-tarayıcı quant'ı 1h barla backtest edip hep 0 işlem alıyordu.** 27 Eylül'den beri 16/16 tarama
  "513 sonuç, 0 filtreyi geçti" (AAPL: quant 0 ↔ event_breakout 30 işlem); 516 backtest Alpaca'yı 429'a sokup canlı sembolleri
  yedeğe düşürüyor, sonuç atılıyordu; boş sonuç tarama anını işaretlemediği için Tur-91 ertelemesi de tutmuyordu.
  `auto_screener.olculemeyen_strateji()` → tarama atlanır (watchlist aynı kalır, neden loglanır); strateji
  `runtime_strategy.settings()`ten (pano > `.env`, `generate_signal` ile aynı kaynak).
- **BULGU-106-5 — tohum yokken eksik seri.** Akış sağlayıcısı REST tohumu 429'da düşünce tampondan 1-8 günlük barı tam seri
  gibi döndürüyordu; boş olmadığı için yedek devreye girmedi, `main._gunluk_barlar` günü önbelleğe aldı → quant o sembolde gün
  boyu `QUANT_YETERSIZ_GECMIS_8<253` (28 Eyl 27.884 satır/6 sembol). İsteği karşılayamayan tampon artık boş döner.
- **BULGU-106-6** — "özellik logu artmadı" uyarısı yalnız bugünün seansı açıldıktan sonra (12:00 koşusunda her gün yanlış alarmdı).
- **BULGU-106-7 — pano NLV/nakit.** `nlv` gerçekleşmiş özkaynaktı, `cash` (`cash_balance`, alımda düşülmez) aynı sayıydı.
  Artık `nlv = equity + Σ açık K/Z`, `equity` ayrı (PDT etiketi ve günlük % onu okur), `cash` pozisyonlar sonrası (eksi = marj).
- **BULGU-106-8 — kasa seans kapısı + takvim saat dilimi.** Gölge kasalar canlı yolun `can_enter_new_position` kapısını
  sormuyordu (136 giriş 15:50 ET sonrası, 75 giriş ilk 5 dk; son dakika pozisyonları gece taşınıyor). `market_calendar`
  saat dilimsiz zamanı `astimezone` ile makinenin YEREL saati sayıyordu → `_et(dt)`: saat dilimsiz = ET.
- **BULGU-106-10 — kasa dolumunda canlı RR kapıları yoktu.** Canlı yol sinyali işlem görülebilir fiyata çapalar
  (Tur-39) ve `min_rr`'ı kaymalı dolumda yeniden sınar (Tur-89); kasa ikisini de atlıyordu. 1m sinyalde risk %0,108 iken
  açılış kovalaması + %0,10 kayma dolumda RR'ı 2,0 → **0,44**'e indiriyordu (%82'si RR<1); kapı kontrol girişlerinin
  %88'ini elerdi ve −268,6 k$ zararın −244,4 k$'ı onlardandı. Dolumda aynı iki kapı (`sayac["rr_kapisi"]`).
- **BULGU-106-11** — uyarı tekilleştirme anahtarı HTF (1h) barıydı: quant'ın günlük "LONG" durumu her saat yeniden
  "kabul edilen sinyal" (Telegram + karar zinciri halkası) oluyordu (1 Eki 76 halka, MU/SNDK 9'ar) → `main._uyari_bari`.
- **BULGU-106-12** — `backtest`/`screen` CLI'larında strateji seçenekleri elle yazılmış 7'lik bayat kopyaydı (15 kipten
  quant_trend, trend_pullback, vol_breakout … seçilemiyordu) ve varsayılan quant 1h'de sessizce 0 işlem basıyordu →
  `choices=list(STRATEGY_MODES)` + quant günlük olmayan aralıkta açık hata (BULGU-106-4'ün kardeşleri).
- **BULGU-106-13** — gece özetinin "Yürüyen kalibrasyon" bloğu 22-24 Eylül verisini (60 bin satırlık kuyruk; tahmin servisi
  28 Eylül'den beri aday üretmiyor, BULGU-102-1) her gece TARİHSİZ basıyordu, üstteki 5 bin satırlık blok aynı kaynağa
  "örneklem yok" diyordu → satır ölçülen dönemi (`eval_range`) yazar.
- **BULGU-106-14** — `/metrics` HELP/TYPE satırlarını etiketli her örnekte tekrarlıyordu (geçersiz sergi; katı ayrıştırıcı
  reddeder) ve `method="total"` aile toplamını ikiye katlıyordu; `[rate]` satırı hızı tüm sayaçlardan alıp yalnız
  bars/quote basıyordu ("5 çağrı/dk · bars=0 quote=0" — çağrılar Emir Akışı `get_trades`). Ad başına tek başlık, tüm sayaçlar.
- **BULGU-106-15** — karar hunisi Tur-89/97/101 guard'larını (`FILL_SLIPPAGE`, `QUANT_POZISYON_VAR`, `QUANT_ODAK_DISI`,
  `QUANT_AYNI_GUN`) tanımıyordu: günde ~42 bin deneme `taninmayan`a düştü ve pano onu çizmediği için "taranan deneme" 60.990
  gösterdi (gerçek 102.590); dolum reddi STRATEJI/LONG + tanınmayan olarak iki kez sayıldı. Aşamalar + çift yazım + panoda
  eşlenmemiş kod satırı; kilit `main.py`nin yazdığı her kodu huniye karşı sayar. `replay` özetinin `warnings` kümesi de huniye bağlandı.
- **BULGU-106-16** — `test_tur77_replay` pozitif kontrol testi üretim `state.json` hash'ini düz karşılaştırıyordu; canlı süreç
  seans açıkken dosyayı yazdığı için Tur-100/101 paketleri kırmızı verip "bilinen yarış" diye bırakıldı. Kardeş testin Tur-88
  içerik atfı (simüle gün izi) ortak yardımcıya alındı.
- **BULGU-106-17 — canlı quant oluşmakta olan bugün barıyla karar veriyordu.** Akış sağlayıcısı 1d'de bugünün yarım
  barını ekler; `main._gunluk_barlar` seriyi ET günü başına bir kez önbelleğe aldığı için günün kararı ilk çekim anının
  fiyatına donuyordu (süreç başlatma saatine bağlı). Kapanmış barlarla karar ↔ + açılış fiyatlı yarım bar: 7.500
  sembol-günün **%5,35**'inde LONG/değil farklı, σ medyan %2,8 kayık. Gölge koşucuda aynı yarım bar Tur-77 tekilleştirmesini
  bozdu (30 Eyl sembol başına 6 epizot). `market_calendar.kapanmis_gunluk`: seans açıkken bugünün barı atılır.
- **BULGU-106-18** — gece özetinin "bekleyen birikimi artıyor" uyarısı iki yapısal kapanış kuyruğunu (ertesi seansı bekleyen
  ~90-120 satır, Tur-84) kıyaslıyordu → 60 koşunun 10'unda yanlış alarm. Artık yalnız çözülmüş olması GEREKEN bekleyen
  sayılır (`reasoner._gecikmis`: hedef sonrası ilk seans dakikası + 30 dk geçmiş).
- **BULGU-106-19** — üç başlatıcı (iki gözcü + `proje-ac`) yeniden başlatmada `Start-Process` yönlendirmesiyle `*.err.log`u
  sıfırlıyordu; 2 Ekim 23:39'da asılı hub'ın (süreç canlı, port kapalı) nedeni böyle silindi. Artık eski günlük yeni süreçten
  önce `.prev.log`'a taşınır.
- **BULGU-106-20** — `resolve()` seans dışında ertesi açılışı bekleyen satırlar için de bar çekiyordu (gözcü 5 dk'da bir
  yoklar: gece ~6.300, hafta sonu ~23.600 boş hub isteği; Tur-81 alternatifi yalnız seans içinde ölçmüştü).
  `market_calendar.ilk_bar_ani` + `prediction_tracker._cozulebilir`; gece özetinin gecikme ölçüsü aynı yardımcıda.
- **BULGU-106-21** — README "Madde Eşlemesi" silinmiş `RiskManager.margin_ok`'u, hiç olmamış `/api/reset-breaker`'ı (gerçek
  yüzey Telegram `/dur`·`/devam`), bayat `time_stop_bars`'ı anıyordu; kilit tablonun her referansını koda karşı tarar.
- **BULGU-106-22** — kanonik gölge görev betiği `tur17` `-Daily` ile hafta sonunu da kuruyordu; "yerine geçen" `tur20`'nin
  dersi taşınmamıştı → `tur17` Pzt-Cum, `tur20` ESKİ. Kayıtlı görev hâlâ 2026-07-25 kaydı (her gün 09:30-16:30 TR): yeniden
  kayıt operatör işi; o zamana dek cuma barı hafta sonu üç kez epizot olur (okuma tarafı karar barı başına tekil).
- **BULGU-106-23** — Laboratuvar hub düğmeleri (`/api/hub/start|restart`) kanonik `nihai.data.hub_server` yerine
  `MainHub/hub_server.py`'yi açıyordu (`/quote`'ta `spread_measured` yok → kayma hep taban; gözcü/`proje-kapat` tanımıyor);
  panelden durdurma bayrağı (`hub.web-stopped`) `proje-ac`ta kalkmıyordu → gözcü kalıcı pasif. İkisi de düzeltildi.
- **BULGU-106-24** — ana süreçte tek-kopya kilidi yoktu: Tur-80 çift kayıtçı olayını yalnız başlatma sırası kapatmıştı,
  proje açıkken `run.ps1`/`-m nihai.execution.main` ikinci tarayıcıyı sessizce açıyordu. `atomic_io.tek_yazar_kilidi`
  (gölge kasaların `motor.lock` kodu ortaklaştı) → `main()` `data/main.lock`u alamazsa web ve döngüyü açmadan çıkar.
- **BULGU-106-25** — 106-7 pano NLV'sini düzeltti; Telegram `/status`, açılış/kapanış uyarıları, heartbeat ve terminal
  panosu gerçekleşmiş özkaynağı "NLV" diye basmayı sürdürdü (aynı anda iki sayı). `PortfolioState.open_pnl` tek tanım.
- **BULGU-106-26** — Emir Akışı hisse önbelleği kapalı seansta thrash ediyordu: boş canlı yoklama depoyu ezip çapalı
  pencereyi her yenilemede (IEX 5 sn) yeniden tam çektiriyordu; ayrıca canlı yenileme ilk geçici hatada kalıcı duruyordu.
  Boş çekim depoyu ezmez, çapalı yanıt `yenile_sn=60`; hatada 30 sn sonra yeniden dener.
- **BULGU-106-27** — Hisse Detay (`/api/analyze`) quant kipinde canlı kararı hiç göstermiyordu (`df_daily` geçilmiyordu:
  gün-içi aralıkta `QUANT_GUNLUK_BAR_GEREKLI`, 1d'de 220 barla `YETERSIZ_GECMIS`); Quant Beyin kartı stratejiyi sabit
  config adından okuyordu. Uç artık canlı taramayla aynı strateji kaynağını ve aynı kapanmış günlük barları kullanır
  (`quant.CANLI_GUNLUK_BAR` tek sabit); quant sinyaline meta P(win) basılmaz. Eski `live_predict` CLI'ı da aynı girdilere çekildi.
- **BULGU-106-28** — pano, sabit hedefte kapanmayan pozisyonun (quant; trailing kipi) hedefini çıkış seviyesi gibi
  gösteriyordu ve `trailing_enabled` quant'ı bilmiyordu. `position_manager.sabit_hedefle_cikar` tek kural (`check_exit` +
  pano); yük `hedef_nominal` taşır, etiket "HEDEF (nominal)"; Telegram sinyal uyarısı ve terminal satırı da aynı bayrağı
  basar (uyarı = emir: nominal hedef kâr-al seviyesi gibi okunmasın); emir planı nominalde TP bacağı önermez. Çıkış
  davranışı değişmedi.
- **BULGU-106-29** — `proje-kapat.ps1` ayrı gölge kasa koşucusunu (`golge_kasalar --kos`) eşleştirmiyordu: proje "kapalı"yken
  koşmaya ve `motor.lock`u tutmaya devam ederdi. Servis süzgeci onu da durdurur (rapor CLI'ına dokunmaz).
- **BULGU-106-30 — pano operatör eylemleri ana sürecin portföyüne ULAŞMIYORDU.** Kapat / Flatten All / stop-hedef düzenle /
  kill-switch sıfırla / elle aç uçları diskten ayrı bir kopya yükleyip ona yazıyordu; ana döngü belleğindeki portföyü her
  turda (~60 sn) geri yazar → kapatılan pozisyon geri gelir ve ikinci kez kapanır (çift defter), düzenleme silinir.
  `--flatten` CLI'ı ana süreç çalışırken aynıydı. Artık `run_loop` canlı `store`/`pmgr`'ı panoya bildirir
  (`canli_baglam_kaydet`), uçlar onlara uygulanır (defter pmgr kancasında, tek kaynak); pano tek başınayken `main.lock`
  tutuluyorsa 409; `--flatten` kilidi alır, ana süreç çalışıyorsa reddeder. Üretimde hiç kullanılmamıştı (defterde iz yok).
- **BULGU-106-31** — yeniden başlatmada taşınan quant pozisyonunun defter riski / planlanan RR'ı / tepesi o anki İZLEYEN
  stoptan kuruluyordu (canlı 12 pozisyonun 10'unda: risk 7–120 $ ↔ gerçek 118–145 $, RR 3,4–51 ↔ ~3) → her stop çıkışı
  deftere tam ±1R yazılırdı. `rehydrate_trackers` artık kalıcı `izleyen_mesafe`den kurar; tepe `stop ± mesafe` ile geri gelir.
- **BULGU-106-32** — pano Sinyal Akışı (`/api/signals`) bugünün yerine DÜNÜN son 200 karar satırını döndürüyordu (iki
  günlüğün kuyruğu `[bugün, dün]` birleştirilip tersten geziliyordu; gerçek günlüklerle 200/200 dün). Artık dosyalar en
  yeniden eskiye, dosya içi sondan başa okunur; gürültü-yalnız dosyalar (hafta sonu) atlanır.
- **BULGU-106-33** — "son sinyaller" (terminal paneli + pano akışının tepesi): iki-katman kipinde tarama iş parçacıkları
  listeye hiç yazmıyordu (`dashboard=None`), satırlar tarihsiz ve kalıcıydı (3 Ekim'de tepede 22 Eylül kapanışları), pano
  listeyi ikinci kez ters çevirip öne çıkanları en alta atıyordu. Pano nesnesi iş parçacıklarına verilir (kilitli liste),
  satır tarih+saat taşır, eski biçim ayıklanır, akış en yeni üstte çizilir.
- **BULGU-106-34** — quant kipinde odak katmanı aynı GÜNLÜK kararı ~6 sn'de bir yeniden üretip yazıyordu (2 Eki: 33.850
  "SKIP … RISK_REJECT" + AI_ADVICE + SHADOW, 40.620 `QUANT_POZISYON_VAR`; 188 bin satırlık günlüğün ~%75'i). `StateStore`
  geçici iz taşır; sembol başına ≥45 sn'de bir değerlendirme (ET saatiyle, replay etkilenmez), muhafız satırı günde bir.
- **BULGU-106-35** — tracker'ın en iyi/en kötü fiyatı ve bar sayısı yalnız bellekteydi: çok günlük pozisyonun MAE/MFE'si ve
  bar sayısı her gece yeniden başlatmada sıfırlanıyordu. Üçü de pozisyonda kalıcı (`en_iyi_fiyat`, `en_kotu_fiyat`,
  `bar_sayisi`); alanları taşımayan eski kayıtta 106-31'in geri kurulumu geçerli.
- **BULGU-106-36** — gece görevi seans açıkken de koşuyordu: kış saatinde (2 Kasım → Mart ortası, seans TR 17:30–00:00)
  23:30 tetiği 15:30 ET'dir (defter yeniden yazılırken canlı ek kaybı, seans içi aday eğitimi). `forward-harness.ps1` takvime
  sorar (`market_calendar --kalan-sn`): kapanışa ≤65 dk varsa kapanış + 5 dk bekler, çoksa atlar; görev yeniden kaydı gerekmez.
  Özetin canlı özellik sayımı yerel "bugün" yerine AÇILMIŞ son seansa bağlandı (`son_seans_acilisi_yerel`): öğle yedeği ve
  gece yarısı sonrası koşu da "artmadı" uyarısını verebilir, seans günü çıktıda.
- **BULGU-106-37** — gölge görev betiği kesirli seans süresini tam saate yuvarlıyordu (`New-TimeSpan -Hours` Int32 alır:
  6,5 → 6, yarım gün 3,5 → 4) ve "6.5 saat" yazıyordu; süre artık dakikayla kurulur. Görev 4 Ekim'de operatörce yeniden
  kaydedildi (Pzt-Cum 16:30, saatlik); 2 Kasım'da kış saati için yinelenmeli.
- **Ölçüldü, değiştirilmedi — quant izleyen stopu canlıda dakikalık güncelleniyor** (ölçülen kural günlük + giriş barı
  kontrolsüz): odak 30 sembol, ~1 yıl, aynı 1m kaynak → işlem 536 ↔ 573, tutma 11,3 ↔ 10,6 gün, sembol başı log getiri
  +0,199 ↔ +0,176 (t=−1,77). Hizalamak strateji kararı (ön-kayıt); rapor §4/§7.
- **BULGU-106-9** — `.env.example` `ACTIVE_STRATEGY=ensemble` → `quant_trend`; bu dosyada süreç yapısı, `/metrics` portu, dosya sayısı;
  pano: kasa açıklaması (giriş kapıları) ve kill-switch cümlesi (`ENABLE_KILL_SWITCH` canlıda varsayılan KAPALI); laboratuvar
  strateji formu quant'ın asgari oy/trend ATR alanlarını okumadığını söyler; forward hükmü oranı "beraberlik hariç" diye adlandırır;
  GATE_HURTS önerisi "kapıyı kapat" yerine "enforce açılmamalı; META_GATE=shadow kanıt döngüsünü yaşatır".
- **Kilit:** `tests/test_tur106_hata_tutarsizlik.py` (73 test), **mutasyon 170/170**; `test_tur90` ve `test_tur102` birer
  testi yeni sözleşmeye çekildi (amaç korunarak); `test_tur77_replay` iki testi ortak yardımcıya geçti (106-16); tam paket
  **5.478/0/1 atlandı**, `ruff check .` + `mypy .` (231) temiz. Canlı: operatör 17:22'de yeniden açtı (106-1..12); sonrasında
  istemci kopması/429/`QUANT_YETERSIZ` 0, RR kapısı 196 kasa girişini engelledi (rapor §6b). 106-13..18 bir sonraki yeniden
  başlatmada girer (gece görevi ve gölge koşucu çalışma ağacından hemen alır); 106-19..37 3-4 Ekim'de eklendi; operatör 3 Ekim 23:36'da yeniden açtı → 106-1..36 canlı (rapor §6). Commit YOK.

**282M CANLI GÖLGE SİNİR AĞI — 10 YILLIK BACKTEST + KIYAS: MODEL GİRDİYE BAKAMIYOR (Tur-107, 2026-10-06).** Rapor:
`docs/TUR107_RAPORU_2026-10-06.md`. Hedef *"projedeki 282m lik ai'a backtest (son 10 yıl) ve benchmark testlerini yap"*.
Kenar iddiası YOK; model yeniden eğitilmedi, canlı süreç/kart/`.env` DEĞİŞMEDİ (koşu canlı sistem açıkken, hub'da 429 = 0).

- **Düzenek `research/noral_282m_backtest.py`** (ön-kayıt + iki tarihli ek docstring'de). Modelin eğitim evreni olan 30
  sembol, 2016-10-06 → 2026-10-05, Alpaca **SIP** 1m (IEX 1m 2020 ortasından önce YOK), yalnız bölünme düzeltmesi, seans içi;
  Tur-94 kurucusuyla `event_breakout` adayları (her 10. bar) → **338.110 satır** (eğitim öncesi 328.491 · eğitim dönemi
  8.419 · ileri 1.200). Ölçüler: canlı OOD kapısı · AUC · Brier ↔ nedensel sabit taban (gün-kümeli t) · gölge `noral`
  kasasının kuralıyla kapı simülasyonu + rastgele kapı + 1 bar gecikme · günlük seri ↔ SPY. Ara dosyalar `~/nihai_hist/noral_282m/`.
- **BULGU-107-1 — 282M'in gövdesi kapalı.** Dropout kapalı çıktı 313 bin satırda **0,48697 ± 1,1e-7**; tüm özellikler
  ±3σ oynatılınca p 7e-7 değişir (`deep_mlp`: 0,467 ↔ 0,418). Neden: girdi [CLS]'e yalnız bloklardan ulaşır ve blok çıkışları
  LayerScale γ ile çarpılır; γ 1e-4'te başlar, eğitimden sonra ort. **7,3e-5** (Adafactor'ın adımı parametre ölçeğine
  görelidir → küçük başlayan katsayı yerinde kalır). γ=0 yapılınca çıktının yayılımı tam 0 olur. Parametrelerin **%98,8'i**
  bloklardadır ve çıktıya girdi-bağımlı katkı vermez. Tur-93 "sıkışık", Tur-94 "öğrenmedi" demiş, mekanizma ölçülmemişti.
- **BULGU-107-2 — canlının yazdığı p, MC-dropout'un yan ürünüdür.** Köprü dropout AÇIK 20 kopyanın ortalamasını yazar
  (her çağrıda aynı tohum): medyan 0,485542, ölçümlerin %94,5'i ±1e-5 içinde; canlı kasa defterinde (8.258 ölçüm) aynı
  dağılım. `noral` kasası kararı altıncı ondalıkta verir (canlı: 4.124 veto ↔ 3.236 giriş). MC ↔ eval korelasyonu 0,19 →
  eval kipiyle yapılan bir backtest canlıdaki değeri temsil etmez; birincil ölçüm köprünün kendi çağrısıyla
  (`get_shadow().pwin`, ~5 satır/sn) yıl başına 10 rastgele seansın tüm satırlarında + ileri dönemin tamamında yapıldı.
- **Backtest (canlı yol, 16.008 satır / 121 gün):** eğitim öncesi AUC **0,5051**, Brier tabandan kötü (gün-t +1,55), kapı
  fazlalığı **+0,16 bps** (t +1,00), 11 yılın hiçbirinde |t| ≥ 1,645; ileri dönem (8 seans) AUC 0,4870, fazlalık −0,11 bps.
  Ön-kayıtlı hüküm **BECERİ YOK**. OOD kapısı 2020 örneklem günlerinde adayların %54'ünü susturdu.
- **Kıyas modelleri (eval, 328.491 satır):** `deep_mlp` AUC 0,4951 ve sabit tabandan anlamlı KÖTÜ (t +10,7); iki modelin
  0,5 altı AUC'si taraf artefaktı (LONG 10 yılda daha sık kazanır, `deep_mlp` 2026 yazından SHORT'u kayırmayı öğrenmiş;
  taraf içi AUC ≈ 0,50). Taban (tüm adaylar, 5 dk) brüt −%0,51/yıl, işlem başı %0,2 ile −%39/yıl; SPY al-tut +%13,6/yıl.
- **Hız/bellek:** 282M 1.075 MB, canlı çağrı 190–270 ms (GPU), toplu 1.857 satır/sn (1,8 GB VRAM), CPU tek satır 266 ms ·
  `deep_mlp` canlı çağrı 18 ms (CPU), 720 bin satır/sn. Karne `data/noral_282m_kiyas.json`.
- **BULGU-107-3** eğitimde nötrlenen `rr_ratio`/`confluence` canlı köprüde HAM besleniyor (z=2,0/1,0); 282M'de etkisiz,
  `deep_mlp`te düzeyi 0,0043 kaydırır — düzeltilmedi (açık kalem). **BULGU-107-4** Tur-94 kurucusu canlı strateji
  `quant_trend` olunca sessizce 0 satır üretiyordu → `sembol_satirlari(..., strateji=)` (varsayılan eski davranış).
  **BULGU-107-5** indirici hata ile "veri yok"u ayırmaz → düzenek ortadaki boş yılda durur; NBIS 2023 doğrulanmış tek
  istisna (işlem durdurması).
- **Değiştirilmedi:** canlı gölge model 282M (dönüş operatör kararı) · reçete (γ + Adafactor; tek değişkenli deney açık
  kalem) · besleme kayması · `deep_mlp`in 1h işlem defterindeki AUC 0,52 yan gözlemi (kapıdan geçmedi).
- **Kilit:** `tests/test_tur107_noral_282m_backtest.py` (10 test), **mutasyon 20/20**; tam paket **5.489/0/1 atlandı**
  (526 sn, canlı süreç koşarken), `ruff check .` + `mypy .` (231 dosya) temiz. Commit YOK.

**PANONUN İNGİLİZCE ARAYÜZÜ — TR | EN, çalışma anı sözlük katmanı (Tur-108, 2026-10-09).** Rapor:
`docs/TUR108_RAPORU_2026-10-09.md`. Hedef *"arayüzün İngilizce versiyonu; istendiğinde Türkçe, istendiğinde İngilizce"*.
Kenar iddiası YOK; sinyal/karar/işlem yolu değişmedi.

- **Kaynak Türkçe KALIR.** ~40 test pano kaynağının Türkçe metnine bakar; İngilizce ikinci bir HTML değil bir katmandır:
  `nihai/web/pano_en.json` (3.222 giriş, 355'i kalıp; anahtar = ekrandaki Türkçe metnin boşlukları tekilleştirilmiş hâli,
  `{0}`… = değişken parça) + `nihai/web/pano_i18n.js` (DOM metni · `title`/`placeholder`/`aria-label` · canvas yazıları ·
  `confirm`; `tr-TR` sayı/tarih biçimi → `en-GB`). `GET /i18n.js` ikisini tek yanıtta verir, pano `<head>`'de eşzamanlı
  yükler. Seçim başlıktaki **TR | EN** düğmesi (`localStorage['pano.dil']`) ya da `?lang=en|tr`; Türkçe kipte motor yalnız
  düğmeyi ekler.
- **Pano metni ekleyen/değiştiren tur çevirisini de ekler:** `python -m tools.pano_i18n` eksikleri basar → `pano_en.json`.
  Kapı HTML'deki her Türkçe düğümü/özniteliği ister (JS sabitlerinde ≥ %99). Arka uçtan gelen görünür metin (model
  kataloğu, hüküm, not) kaynak taramasıyla görülmez; sözlüğe elle girer.
- **Motor sırası:** tam eşleşme → kalıp (en özgül önce; sabit kısmı kısa kalıp yalnız yalın parça yakalar) → kenardaki
  ayraç/sayı soyulur → bileşik metin cümle · `·` · `—` · `, ` · `: ` sırasıyla bölünür, parçalar ayrı çevrilir. Arka uç
  serbest metni (`pre`, `.log-line`, `.pre-box`, `#ev-lessons`, `#ev-findings`) yalnız TAM eşleşir; `translate="no"` atlanır.
- BULGU-108-1..6 raporda: çıkarıcının kod süzgeci düz yazıyı eliyordu (59 metin) · kimlik numaralı çeviri tablosu 454
  satır kaymıştı · `{0} gün` bileşik metni yutuyordu · `: ` ↔ `, ` sırası · serbest metinde yarım çeviri · `node -e` argv.
- **Ölçüm (8777, 17 sekme, canlı veri):** Türkçe-özel harfli çevrilmemiş metin 327 → 17. **Değiştirilmedi:** arka ucun düz
  yazısı (brif, özet, dersler, loglar) Türkçe · `bulten.html`, terminal panosu, Telegram kapsam dışı · koşan ana süreçte
  `/i18n.js` ucu yok (pano Türkçe sürer) → bir sonraki `proje-ac`'ta devreye girer.
- **Kilit:** `tests/test_tur108_pano_ingilizce.py` (8 test; motor node ile iki kipte), **mutasyon 14/14**; tam paket
  **5.498 geçti / 0 kaldı / 1 atlandı** (355 sn, canlı süreç koşarken); `ruff check .` + `mypy .` temiz. Commit YOK.

## ⚠️ ACI DERSLER — bunları tekrar etme (institutional knowledge)

Bu satırlar projenin en pahalı öğrendiği şeyler. Yeni "kenar" iddiası yapmadan önce oku:

1. **Survivorship bias kenarları uydurur.** PEAD: nasdaq100'de PF 1.57 → point-in-time S&P500'de
   PF **1.05** (başabaş); "removed-since" isimlerde PF 0.80 (kayıp). Kenar büyük ölçüde artefakttı.
2. **Meta-labeling PIT'te ölçüldü: kenar YOK.** Survivorship-temiz S&P500@2022 (11.874 işlem),
   OOS-AUC **0.51–0.52** (yazı-tura), hiçbir eşik kâra çevirmedi. nasdaq100'deki PF 1.07–1.38
   survivorship artefaktıydı.
3. **AUC değil, KAPI SİMÜLASYONU karar verir.** Model "düşük olasılıkları ele" filtresi olarak
   görülmemiş işlemlerde expectancy/PF'yi BASE'in üstüne çıkarıyor mu? **Hem time hem symbol**
   bölmesinde çıkmalı = dayanıklı. Tek bölmede çıkması overfit olabilir.
4. **Bu yüzden `META_GATE` varsayılan KAPALI.** OOS-kenarlı baz sinyal yokken kapı sadece gürültü ekler.
5. **Gerçek darboğaz = veri ve OOS-kenarlı baz sinyal yokluğu**, model mimarisi değil. İleriye giden
   yol: Sharadar/CRSP PIT verisi VEYA `prediction_tracker` ile canlı paper-trading kanıtı. Yeni bir
   sınıflandırıcı eklemek bunu çözmez.
6. **PEAD kesit-ML de ölçüldü: kenar YOK (§19).** Akademik desteği olan PEAD sinyaline survivorship-
   temiz PIT S&P500'de (3228 olay) kesit-seçim ML'i (surprise+cap-tier özellikleriyle) uygulandı;
   purged 5-fold CV OOS-AUC 0.50-0.52, removed-since kapısı 0-1/5 fold pozitif = artefakt. Tek-bölme
   PF 4.40 görünüyordu ama survivorship/şans idi. Model `surprise`'a değil `vix/ret`'e bakıyordu.
7. **TÜM metrikler yeşilken bile kenar zamanlama artefaktı olabilir — exec-lag testi ZORUNLU
   (§22 · §23 · §26 — ÜÇ KEZ doğrulandı, artık VARSAYILAN ŞÜPHE).**
   Pairs/kointegrasyon (Kalman) PF 2.08 + maliyet-dayanıklı + survivorship-PARİTELİ (rem-PF≈surv-PF,
   ilk kez!) + PBO 0.004 gösterdi; uygulamaya 1 bar gecikme eklenince PF 1.03 / +1.8 bps'e çöktü.
   Kenarın ~%80-95'i sinyal-sonrası İLK bardaydı = 1-günlük reversal/bid-ask sıçraması, kointegrasyon
   dönüşü değil. Yeni aday sinyalde `--exec-lag 1` duyarlılığı kanıt kapısının parçasıdır
   (`research/pairs_test.py` deseni); DSR/PBO/survivorship bunu YAKALAMIYOR.
   Harness (`build_pead_dataset.py`+`train_pead_ml.py`) survivorship-bölmeli; PIT veri gelince tekrar koş.
   **§26 (2026-08-03, lead-lag):** aynı desen üçüncü kez — kapı simülasyonu iki bölmede de
   PASS ederken 1-bar gecikme işlem sayısını 29→3'e düşürdü. Bu turda ayrıca **YENİ okuma
   kuralı** doğdu: *kapı PASS dese bile `n_kept` okunmalı* — 29/33.212 (%0,087) seçicilik
   PASS'i geçersiz kılar (uç-kuyruk örneklemi, kenar değil).
   **Tur-24 (adayı olmayan turda) bu kuralı MAKİNEYE çevirdi:** `harness` kapı simülasyonu
   artık `max(5, %0,5·n_test)` ORANSAL tabanını ve tutulan↔elenen **Welch t>1,645**
   anlamlılığını da arar — "kapı-E > base-E" tek başına YETMEZ. Geriye dönük koşuldu:
   iki geçmiş adayın da VERDİKTİ değişmedi, yalnız istatistiksel olarak boş iki "PASS"
   kalktı (kanıt kapısı sıkılaştırması yeni sonuç ÜRETEMEZ, yanıltıcıyı eler).
   **§27 (2026-08-10, endeks rebalans — DÖRDÜNCÜ kez):** kapı simülasyonu iki bölmede de
   PASS (t=8,51 ve 11,51; %12,3 ve %8,2 seçicilikle — bu kez örneklem BÜYÜKTÜ) ama 1-bar
   gecikmede kapı fazlalığının yalnız **%27,7'si** hayatta kaldı (+0,2766 → +0,0765).

8. **Sınıf dengesizliği DSR'yi şişirir — sözde-getiri TRİVİYAL kurala göre FAZLALIK olmalı
   (2026-07-25, Tur-17).** `train_neural` DSR'yi ham isabetten (`2·hit−1`) hesaplıyordu. Taban
   oranı %50'den uzak veri setinde "hep çoğunluk sınıfını söyle" kuralı bile pozitif ortalama
   verir: PIT S&P500'de (taban %35,7) **OOS-AUC 0.500 iken DSR 1.0** çıkıyordu — kapı, elemesi
   gereken şeyi onaylıyordu. Düzeltme: `excess_pseudo_returns` (modelin isabeti − çoğunluk
   kuralının isabeti; çoğunluk YALNIZ eğitim fold'undan). Sonra: DSR **0.0**. Ders: bir kanıt
   kapısı raporun geri kalanıyla ÇELİŞİYORSA (AUC~0.50 + DSR~1.0) kapı bozuktur, sonuç değil.
   **AYNI HATA YENİ YERDE (2026-08-10, Tur-25 — BULGU-25):** `edge_lab/harness._stage_dsr`
   Sharpe'ı **sıfıra karşı** ölçer ve `_kept_pnl_from_time_gate` HAM PnL döndürür. Gerçek
   `pnl_col` veren İLK adayda (endeks rebalans) `base_E=+0,2514` (yükselen piyasa) olduğu
   için **DSR=1,0 çıktı — OOS-AUC 0,507 iken**. Ölçüldü: aynı 1,0'ı kapının ELEDİĞİ satırlar
   (SR 0,042) ve RASTGELE alt küme (0,050) de veriyor → kapı beceriyi değil piyasa
   sürüklenmesini ölçüyor. **Kapı o turda BİLEREK düzeltilmedi** (aday FAIL ettikten SONRA
   kapı değiştirmek = sonuca göre kural değiştirmek; Tur-23 kuralı → adayı OLMAYAN bir turda).
   **Tur-31'de (adayı OLMAYAN tur) KAPATILDI.** `_kept_pnl_from_time_gate` artık kapı-sonrası
   **FAZLALIK** serisini döndürür: kıyas tabanı sıfır değil "kapıyı hiç uygulama, her işlemi
   al" triviyal kuralıdır (Tur-17 deseninin aynısı). Taban AYNI TEST diliminden alınır —
   eğitim tabanı ÖNCE denendi ve ÖLÇÜLEREK elendi: rebalans setinde eğitim +0,1751 ↔ test
   +0,2514, aradaki **+0,076/işlem DÖNEM FARKI** tek başına n=27.003'te DSR=1,0 üretiyordu.
   Test tabanı sızıntı DEĞİLDİR (triviyal kural hiçbir şey öğrenmez, geleceği bilmez,
   tahminleri değiştirmez) ve zaten ikili kapının kullandığı `base_E`'nin ta kendisidir.
   **Ayırt-edicilik ÖLÇÜLDÜ** (gerçek rebalans seti, BULGU-25'in kendi teşhis üçlüsüyle):

   | seri | ESKİ (sıfıra karşı) | YENİ (tabana karşı) |
   |---|---|---|
   | kapı | DSR 1,0 PASS | DSR 1,0 PASS (SR_fazlalık +0,0478 — Welch t=8,51 ile tutarlı) |
   | kapının ELEDİĞİ | DSR 1,0 **PASS** | DSR 0,0000 **fail** |
   | RASTGELE alt küme | DSR 1,0 **PASS** | DSR 0,2649 **fail** |

   Geriye dönük koşuldu: rebalans adayının VERDİKTİ DEĞİŞMEDİ (`purged_cv` AUC 0,507 → hâlâ
   FAIL) — kanıt kapısı sıkılaştırması yeni sonuç ÜRETEMEZ, yanıltıcıyı eler. Kapı artık
   sürüklenmeyi değil SEÇİCİLİĞİ ölçüyor; rapor satırı çıkarılan tabanı ve `SR_fazlalık`'ı
   yazar. Ayrıntı: `docs/AI_TUR25_PLAN.md` §3.5.

9. **Kalibrasyon kazancı "bilgi bulundu" demek değildir — SABİT tahminciyle kıyasla
   (2026-08-10, Tur-25).** `calibration_loop` meta `pwin` için OOS'ta Brier 0,2723→0,2490 ve
   **ECE 0,1325→0,0045** ölçtü; Platt katsayısı NEGATİF çıkınca ilk okuma "ham pwin ters
   bilgi taşıyor" oldu. Kontrol koşuldu: **sabit = fit-dilimi taban oranı** (0,4646) Brier
   0,2494 / ECE 0,0090 veriyor — kalibre sürüm sabiti yalnız **0,0004 Brier** geçiyor ve
   kalibre olasılıkların std'si 0,0156 (ham 0,0686). Yani kazancın tamamına yakını **taban
   orana BÜZÜLMEDEN** geliyor, ters çevirmeden değil. Dürüst ifade: pwin ağır fazla-güvenli
   ve kullanılabilir bilgi taşımıyor (ACI DERS #2/#4'ün taze doğrulaması). **Kural: bir
   kalibrasyon iyileşmesi raporlanmadan önce sabit-taban tahmincisiyle kıyaslanmalıdır.**
   **Tur-27'de bu kural MAKİNEYE geçti:** `nihai/ai/adaptive_calibration.py` sabit tabanı
   üçüncü yarışmacı olarak kapının içine koyar (eşleştirilmiş Brier farkı, t<−1,645).
   **Tur-28'de `calibration_loop.refresh` kapısına da kondu** (ortak kaynak:
   `learning.paired_brier_t`) ve gerçek artefakt yeniden verdiktlendi: `pwin` KAYITLI →
   **KAPI_GECILEMEDI** (t=−1,42, eşiği geçemiyor). Elle kontrol artık HİÇBİR yerde gerekmiyor.
   Ders: bir kural iki kapıdan yalnız birine konursa, ikisi aynı veride ÇELİŞİK verdikt verir
   ve eski/yanlış olanı okuyan operatör aynı tuzağa düşer.

- **ÖRTÜŞEN pencerelerden hesaplanan bir p-değeri BAĞIMSIZLIK varsayar ve durağan
  gürültüyü "devrilme" diye raporlar (Tur-73).** `critical_slowing_down`'un ilk sürümü
  pencereleri BİRER BAR kaydırıyordu; Kendall tau'nun p'si gözlemlerin bağımsız olduğunu
  varsayar, oysa komşu pencereler neredeyse aynı veriyi paylaşır. Ölçüldü (200 tekrar,
  DURAĞAN AR(1)=0,2, n=1200, window=100): step=1 → **%19,5 yanlış-pozitif** (p=9,7e−102 ile
  "devrilme"!), step=25 → %5,0, step=100 (örtüşmeyen) → **%0,0**; gerçek bir devrilmeyi
  yakalama gücü ÜÇ ayarda da **%100**. Yani örtüşmeyi bırakmanın GÜÇ MALİYETİ YOKTUR ve
  örtüşen varsayılan saf bir hataydı. Kural: kayan-pencere istatistiklerine bir HİPOTEZ
  TESTİ bağlarken pencereler arası bağımlılığı ÖNCE sor; "daha çok nokta" daha çok BİLGİ
  değildir. Eğriyi yoğun çizmek için örtüşme hâlâ seçilebilir ama sonuç `ortusen=True`
  taşır ve p yalnız betimleyicidir.
- **Aşırı parametreli bir uyum, MÜKEMMEL uyarken EKONOMİK OLARAK ANLAMSIZ parametre
  üretebilir (Tur-73).** NSS'in salt-en-küçük-kareler kazananı 2026-09-03 Hazine eğrisini
  rmse %0,029 ile uydururken **β0 = −2,08** verdi — 30Y getirisi %5,25 iken. Panoda
  "SEVİYE" diye gösterilecek sayı buydu. Tarihsel ölçüm daha da net: son 500 işlem gününün
  **%98,0'ında** β0 gözlenen getiri aralığının, **%81,0'ında** toleranslı bandın bile
  DIŞINDAYDI. Onarım KIRPMA değil SEÇİMDİR (Tur-33: "kırpma bir ölçüm değil KARARDIR"):
  arama iki aday tutar ve kabul edilebilir olan BİRİNCİL sonuç olur, kısıtsız uyumun ne
  diyeceği `beta0_kisitsiz` ile AYRICA taşınır. Bandın kendisi de körlemesine seçilmez —
  β0 bir ASİMPTOTTUR ve eğri ona ulaşmaz (sentetik eğride max(y)=4,43 iken gerçek β0=4,50),
  bu yüzden bant aralığın yarısı kadar taşırılır ve sentetik geri kazanım (rmse 9,47e-16)
  KORUNUR. Kural: bir uyumun `rmse`'si kabul kriteri DEĞİLDİR; parametrenin YORUMLANABİLİR
  olduğu da ayrıca sınanmalıdır.
- **İki "sürüklenme" farklı UZAYDA olabilir; birini diğerinin yaklaşığı sanmak 14 kat
  hata verir (Tur-73).** `leveraged_etf_drag` aritmetik sürüklenmeyi (bileşik farkı)
  hesaplıyor, yanına Cheng-Madhavan yaklaşımını basıyordu; ±%5 salınan seride ikisi −0,1317
  ↔ −1,8900 çıktı. Yaklaşım LOG uzayındadır: karşılaştırılacak büyüklük
  `ln(1+bileşik_lev) − β·ln(1+bileşik_idx)`'tir ve o eklenince örtüşme ±%5'e indi
  (−1,9212 ↔ −1,8900). Yan bulgu daha önemli: **"vol drag her zaman negatiftir" ARİTMETİK
  ölçüde YANLIŞTIR** — düz +%0,1'lik trendde aritmetik sürüklenme **+0,2680**, log
  sürüklenme −0,000754'tür (kaldıraçlı bileşik, tek yönlü seride naif katı GEÇER). Kural:
  bir yaklaşımı ölçümün yanına basmadan önce İKİSİNİN AYNI BÜYÜKLÜĞÜ ölçtüğünü doğrula.
- **KAÇAN bir mutasyon, KUSURU DEĞİL KURGUYU suçlamadan önce kovalanmalıdır (Tur-73).**
  CDaR LP'sinden koşan-tepe monotonluğunu silen mutasyon hiçbir testi kırmadı. İlk içgüdü
  "test zayıf, eşitliği sıkılaştır" idi; doğru hamle mutasyonun NE YAPTIĞINI ölçmekti.
  Ölçüm gösterdi ki mutasyonlu LP w=[1,0] veriyor ve DOĞRU kod da AYNI köşeyi veriyordu —
  yani test kurgusu iki yöntemi hiç ayırmıyordu. Kurguyu ayırt edici hâle getirmek ise
  ÜÇÜNCÜ bir şeyi açığa çıkardı: `cvar_lp_weights`in α konvansiyonu paketin geri kalanıyla
  TERS (BULGU-73-1). Kural: kaçan bir mutasyon üç şeyden birini söyler — eksik test, ölü
  guard (Tur-72) ya da **kurgunun dejenereliği**; üçüncüsü en sinsisidir çünkü test yeşil
  KALIR ve iddia "doğru ama boş"tur (Tur-33'ün "aradığım kusur olsaydı bu gösterge farklı
  görünür müydü?" sorusunun mutasyon tarafı).
- **Devredilen bir dosyaya, devir BİTENE KADAR dokunma (Tur-73, süreç dersi).** Bu turda
  ağır yazım işi `codex`e devredildi (`quant_math.py`, 11 fonksiyon + 281 test) ve ajan
  hâlâ koşarken aynı dosyaya iki düzeltme uygulandı. Ajan bitirirken dosyayı KENDİ tam
  sürümüyle yazdı ve **iki düzeltme sessizce geri alındı** — ne hata çıktı ne çakışma
  uyarısı. Kusur yalnız aşağı-akıştaki bir `KeyError: 'adim'` ile göründü ve ilk teşhis
  YANLIŞ oldu (Tur-72'nin bytecode-önbelleği dersi yanlış yere uygulandı). Kural: devrin
  bitişini bir **DURUM** olarak doğrula (süreç sayısı / bitiş bildirimi / dosya karması),
  bir VARSAYIM olarak değil; ve devir sonrası kendi düzeltmelerinin hâlâ yerinde olduğunu
  `grep` ile SAY.

- **Tek etiketli bir dış şemaya bağlanan okuyucu SESSİZCE yanlış sayı üretir; zincir
  KALEM başına değil DÖNEM başına uygulanmalıdır (Tur-76, BULGU-76-1).** SEC XBRL'de
  `Revenues` AAPL'de VARDIR ama **11 satırdır ve 2018'de biter**; gerçek kalem
  `RevenueFromContractWithCustomerExcludingAssessedTax`tır. Onu okuyan bir Piotroski
  skoru 2018 hasılatını 2026 kârıyla kıyaslar ve HATA VERMEZ. İlk onarım (kalem başına
  TEK etiket, zincir sırasına göre) da ölçümle düştü: `PaymentsOfDividendsCommonStock`
  zincirde öndedir ama 2017'de biter → 2018 sonrası tüm temettüler NaN oluyordu. **Sayı
  ÜRETMEMEK de yanlış sayı üretmenin bir biçimidir.** Kural: bir dış şemadan okurken
  alan adının VARLIĞI değil, o alanın İSTENEN DÖNEMİ kapsayıp kapsamadığı sorulmalı; ve
  hangi dönemin hangi etiketten geldiği çıktıyla TAŞINMALIDIR (Tur-34'ün "ölçüm KOŞULU
  taşınmalıdır" kuralının şema tarafı). Yan kural: **bir zincirin halkaları AYNI TARİH
  ANLAMINI taşımalıdır** — `dei:EntityCommonStockSharesOutstanding`ın `end`i KAPAK
  tarihidir ve zincire konunca AAPL çerçevesini 20 → 37 satıra çıkarıp 17 boş "mali yıl"
  üretiyordu; `son_iki_yil` cari yıl diye o boş satırı döndürüyor, TÜM skorlar sessizce
  `None` oluyordu (BULGU-76-2).
- **Bir modüle ÜRETİM ÇAĞIRANI bağlamadan önce onun MALİYETİNİ ölç (Tur-76,
  BULGU-76-3).** `audit_chain` Tur-75'te yazıldı, doğrulandı, CLI'ı verildi — ve
  `append` ekleme başına dosyayı İKİ KEZ baştan sona geziyordu (bir kez son hash, bir kez
  `seq` sayımı) → ekleme O(n), zincirin ömrü boyunca **O(n²)**. Kusur ancak CANLI KARAR
  yoluna bağlanacakken görüldü. Doğru desen zaten depodaydı (`atomic_io.tail_lines`,
  ölçüldü 9,7-86×) ve `seq` KAYDIN İÇİNDEYDİ, sayılması gerekmiyordu. Bu, Tur-35'in
  kuyruk-okuyucu dersinin dördüncü kopyasıdır. Kural: "yazıldı + test edildi + CLI'ı var"
  bir modülün ÜRETİMDE koşmaya hazır olduğunu göstermez; bağlamadan önce sıcak yolun
  karmaşıklığını SOR.
- **`numpy_dizisi or []` fail-soft bir yolda ölçümü SESSİZCE yok eder (Tur-76,
  BULGU-76-4).** `cus.get("events") or []` ifadesi `bool(dizi)` çağırır ve *"truth value
  of an array is ambiguous"* ile patlar; blok geniş bir `except` içindeyse istisna
  YUTULUR ve iki ölçü hiç yazılmaz — hiçbir yerde hata görünmez. Kusuru bulan şey bir
  istisna değil, ÖLÇÜLEN ALANIN YOKLUĞUYDU. (Tur-74'ün *"fail-soft bir yolda istisna
  hiçbir şey kanıtlamaz"* dersinin ÜRETİM tarafı; orada TEST tarafındaydı.) Kural:
  `or`/`and` ile varsayılan verirken solun `__bool__`u tanımlı mı diye bak; diziler,
  DataFrame'ler ve Series'ler için `is None` kullan.
- **Elle bakılan bir LİSTE sessizce eskir — dizinle eşitliğini MAKİNEYE sor (Tur-76,
  BULGU-76-5).** `nihai/ai/__init__.py`'nin PEP 562 lazy listesi dizinle eşit olmalıydı;
  DÖRT modül eksikti (`option_flow` Tur-57'den, `econophysics`/`ml_hygiene`/
  `volume_profile` Tur-75'ten) ve `import nihai.ai; nihai.ai.volume_profile`
  **AttributeError** veriyordu. Kusur görünmemişti çünkü `from nihai.ai import X` deseni
  import sisteminin KENDİ çözümüyle çalışır — yani liste yanlışken bile çoğu çağrı
  yolu doğru davranıyordu. Kural: bir yerde "bu liste şu dizinle aynı olmalı" yazıyorsa
  o eşitlik bir TESTTİR, bir yorum değil.

- **Bir karar kuralının YÖNÜ, kural bir şartnamede yazılı diye varsayılamaz (Tur-77).** Liste *"epistemik
  risk yüksekse ABSTAIN"* diyordu ve kural sezgisel olarak doğru görünüyordu. Ölçüldü (canlı 1m, kronolojik
  bölme, 5 tohum): topluluk üye-arası ayrışması ve karşılıklı bilgi isabeti **0/5** tohumda TERS sıralıyor —
  yüksek epistemik belirsizlikte model daha İSABETLİ. Kural bağlansaydı en iyi dilimi atardı. Tur-36'nın
  MC-dropout dersinin genellemesi: "belirsizlik" etiketli HER ölçü için dilim isabeti koşulmadan eşik kurulmaz.
- **Bir aralığın 0,5'i kapsaması bir BİLGİ ölçüsü değildir (Tur-77).** Venn–Abers aralığının genişliği
  kaynağın bilgisini değil kalibrasyon kümesinin YOĞUNLUĞUNU ölçer (aynı dağılımda n=100 → 0,10, n=3000 → 0,009).
  Büyük kümede aralık o kadar daralır ki bilgisiz kaynak da "nötr değil" görünür (canlı pwin: kapsayan pay
  %0,25). Bilgi sorusu yine eşleştirilmiş Brier kapısıyla sorulur (ACI DERS #9).
- **Bir defterin SATIR sayısı onun ÖLÇÜM sayısı değildir (Tur-77, BULGU-77-1).** Epizot defterinde 1.508 anının
  %44,4'ü kopyaydı; 1.000 çözümde yalnız 210 benzersiz karar vardı, biri 19 kez. Kopyalar hiçbir hata üretmez —
  yalnız her istatistiği (komşu listesi, taban ortalaması, unutma skoru) sessizce ağırlıklandırır. Kusur ancak
  bir YÜZEY ilk 5 komşunun beşini aynı karar olarak gösterince görüldü. Kural: append-only bir deftere yazan
  döngü tekrarlı koşuyorsa "aynı gözlemi ikinci kez mi yazıyorum?" sorusu yazma tarafında sorulmalıdır.
- **İki okuyucu aynı dosyayı FARKLI tarih kuralıyla adlandırırsa kusur yalnız bazı SAATLERDE görünür (Tur-77,
  BULGU-77-3).** Karar logunu yazıcı yerel tarihle, bir okuyucu ET tarihiyle adlandırıyordu; ABD seansında
  (TR 16:30–23:00) iki tarih aynı gün olduğu için test ve gündüz kullanımı yeşildi, TR 00:00–07:00 arası yanlış
  dosya okunuyordu. Dosya adı YAZICININ fonksiyonundan (`_today_log()`) alınır, ikinci kez kurulmaz.
- **Bir ajanın çıktısı, sınırı ve içeriği AYRI AYRI ölçülmeden kabul edilmez (Tur-77, süreç dersi).** Üç codex
  görevinde dosya sınırı MD5 anlık görüntüsüyle ölçüldü (ihlal 0) ama içerik incelemesi iki kusur buldu
  (tutarsız satırdan HAYALET lot; canlı kapanış defterinin opsiyon çarpanını atlayan KOPYASI) + bir mypy hatası
  (ajan yalnız ruff koşmuştu). Tersi de ölçüldü: Gemini'nin bağımsız incelemesindeki **3 bulgunun 3'ü yanlış
  pozitifti** (kesitte görünmeyen yardımcıyı "tanımsız" saydı, var olan bir scipy parametresini "yok" dedi).
  Bir inceleme bulgusu da bir iddiadır; test/kaynakla doğrulanmadan uygulanmaz.
- **Bir alanın tipini genişletmek (`float` → `float | None`) tüketicilerini sessizce kırar — tam kapıyla ölç
  (Tur-77).** Önceki ajan backtest sonuç alanlarını opsiyon yolu için `None`-olabilir yaptı ve "mypy temiz"
  yazdı; mypy'yi dosya bazında koşmuştu. README kapısı `mypy .` bir tüketicide 3 hata verdi (`round(None)`).
  Tur-42 dersinin ikinci tekrarı: kapının komutu belgede yazılıysa turda O komut koşulur.
- **İki uygulamanın ayrışması hangisinin doğru olduğunu söylemez — üçüncü yöntem hakemlik eder (Tur-78).** Gemini
  temiz-oda referansıyla fark testinde üç aile ayrıştı: HERC ve CVaR paritesinde ÜRETİM yanlıştı (tam çözücü
  gösterdi), VPIN'de 0/300 uyumsuzluğun yanlış tarafı REFERANSTI (kaba kuvvet hakemine üretim 2,1e-15, temiz oda
  0,44). Referans uygulama bir kanıt değil ikinci bir iddiadır.
- **"Test edilmedi" diye işaretlenen öneri de bir iddiadır — ve bir LP'de BANT, EŞİĞİ geçmez (Tur-78).** Codex'in
  CVaR önerisi tam eşitliği ±1e-5 bandına çevirmekti; ölçülünce `None` 40 → **202** oldu. Doğrusal programın
  çözümü köşededir, köşe bandın SINIRINDADIR ve aynı eşikteki kabul kontrolü kayan nokta gürültüsüyle düşer.
  Kabul eşiği olan bir uygunluk sorusu, ihlali EN KÜÇÜKLEYEN (minimaks) problemle sorulur.
- **Sözleşmeyi harfiyen izleyen düşman vakalar KODU sınar, KURALI değil (Tur-78).** Gemini'nin 24 PDUFA vakası
  üretimle 24/24 uyuştu çünkü istem beklentiyi yazılı kurala bağlıyordu; kuralın kendisi üç vakada anlamca
  yanlıştı ("up**date**", "Inc. (", konferans tarihi). Sezgiye aykırı sonuçlar ayrıca YARGILANMALIDIR.
- **Bir düzeltmenin gerçek veride "fark 0" vermesi yeni yanlış negatif doğurmadığını kanıtlamaz (Tur-78).** PDUFA
  etkinlik filtresi 221 gerçek bağlamda hiçbir satırı değiştirmedi, ama düşman cümle ("in the event of a major
  amendment, may be extended to <tarih>") gerçek tarihi düşürdü. Örnek o kalıbı içermiyorsa ölçemez; yeni
  kuralın yıkabileceği kalıp elle kurulur.
- **Kink'teki "risk katkısı" tek değerli değildir; süreksizliği HATA sanıp ağırlığı "düzeltme" (Tur-78).** CVaR
  paritesinde dönen 260 çözümün 187'si tam-bağ yorumunda pay hatası >1e-5 gösterdi (maks 0,62), oysa ağırlıklar
  tam çözüme ≤4,3e-7 idi: yuvarlama ölçeğinde sapma farklı kuyruk kümesi seçtiriyor. Doğru hamle bayrak (`kink`).
- **Durum değiştiren bir yolda, ondan ÖNCE çağrılan teşhis istisna atamaz (Tur-78, R3-B1).** Maliyet ayrıştırması
  canlı kapanışta pozisyon defterden düşülmeden önce koşuyordu; bozuk bir fiyat referansı kapanışı yarıda
  bırakabilirdi. Karşıt durum da ölçüldü: dolum FİYATI geçersizse istisna DOĞRUDUR, çünkü o sayı özkaynağa girer.
- **Kaçan mutasyon ölü koruma değil KÖR EŞİK olabilir (Tur-78).** ERC Armijo yuvarlama payını sıfırlamak hiçbir testi
  kırmadı; ölçülünce pay hatası 1,5e-11 → 1,6e-9 (5,4× yavaş, patolojik kümede `None` 21 → 43) çıktı. Test eşiği
  1e-8 olduğu için 100 katlık bozulma görünmüyordu. Kaçan mutasyonun etkisini ÖLÇ, sonra eşiği ya da korumayı seç.
- **Kıyas sıralaması üretim sıralaması değildir — canlı süreci örnekle (Tur-80).** Veri kopyasındaki uç kıyası
  en pahalıları elle açılan uçlar sanıyordu; canlı süreçte py-spy aktif örneklerin %42,8'ini 60 sn'de bir
  YOKLANAN tek bir ucun CSV döngüsüne yazdı. Kıyas maliyeti ölçer, üretim maliyet × SIKLIĞI.
- **Python örnekleyicisi yerel thread'lere KÖRDÜR (Tur-80).** py-spy tek çekirdeğin ~%18'ini açıklarken süreç
  ~%46 kullanıyordu; OS thread CPU'su py-spy thread kimlikleriyle eşlenince fark torch OpenMP'de çıktı. 0,2M
  parametreli ağda 8 iç thread çağrı başına 84 ms CPU yakıyordu (duvar 10,2 ms); tek thread 11 ms (duvar
  8,3 ms), bit-denk.
- **Paylaşılan model + süreç-genel RNG + thread havuzu = sessiz ölçüm bozulması (Tur-80, BULGU-80-3).** Seri
  çağrı deterministikti; 8 thread'de 2160 sonucun 2100'ü farklıydı ve hiçbir hata çıkmadı. Kilit kapsamı RNG'yi
  tüketen HER yolu kapsamalı — çıkarım kadar model KURULUMU da (ilk düzeltme yüklemeyi kaçırdı, Codex ölçtü).
- **Uzunluktan sayılan "N yeni bar" sabit pencerede hep sıfırdır (Tur-80, BULGU-80-2).** Sayım zaman ANINDAN
  (`asi8`) ve son görülen bardan biriktirilerek yapılır; `str()` karşılaştırması saat dilimi gösterimi değişince
  sahte "yeni bar", pencere içi arama ise eşik > pencere olduğunda sahte eşik üretir.
- **Bir imza önbelleğinin imzası YAZIMLA aynı anda alınmalıdır (Tur-80).** Değiştirmeden SONRA alınan imza,
  araya giren başka sürecin eklemesini eski satırlarla eşleştirir; doğru imza ara dosyanınkidir (yeniden
  adlandırma mtime/boyutu korur). Tek seferlik bir devirde "tüketilmezse ne olur?" da tasarımın parçasıdır.
- **"Normal" girdili denklik ızgarası hızlı yolun FAZLADAN kabul ettiğini göremez (Tur-80).** `fromisoformat`
  3024 ISO girdide pandas'la eşitti; pandas'ın REDDETTİĞİ hafta tarihini, `X` ayırıcıyı ve saniyeli ofseti
  kabul ediyordu → eskiden dışlanan işlem rapora giriyordu. Hızlı yol katı biçimle kapılanır, karşı örnekler
  ızgaraya girer.
- **Sürecin yorumlayıcısını yürütülebilir yoldan değil `sys.prefix`'ten oku (Tur-80, düzeltme).** Canlı süreç
  temel `Python312\python.exe` olarak görünüyordu ve "sistem Python'u" sonucuna varıldı; oysa watchdog onu
  `__PYVENV_LAUNCHER__=.venv\Scripts\python.exe` ile başlatır ve süreç venv paketlerini yükler. Yanlış öncülle
  dört A/B sistem yorumlayıcısında ölçülüp rapora "üretim yorumlayıcısı" diye yazıldı; venv'de yeniden
  ölçülünce hepsi değişti (MC-dropout 90,5 → 16,3 ms iken 84 → 11 ms). Ölçüm ortamını, başlatıcının kurduğu
  ortam değişkenleriyle aynı komutu koşup `sys.prefix` + paket sürümleriyle doğrula.
- **Pozisyon başına döngüde SEMBOL başına güncelleyici çağırmak ikiz pozisyonları çift işler (Tur-80,
  BULGU-80-5).** `update_all_with_1m_bars` pozisyon başına dönüp sembolün TÜM pozisyonlarını güncelleyen
  `update_all_with_bars`ı çağırıyordu; küme sınırı (2) aynı sembolde iki pozisyona izin verdiği için her bar iki
  kez işlendi (`bars_held` 2×, breakeven açıkken aynı-bar stop). Testlerin hepsi sembol başına tek pozisyon
  kuruyordu. Döngünün ve çağrılan işlemin anahtarı (pozisyon ↔ sembol) aynı olmalı; kurgu, sınırların İZİN
  VERDİĞİ en kalabalık durumu içermeli. Kusuru bulduran, kilit refaktörü için "eski gövdeye denklik" yazarken
  aynı sembolü iki kez açmayı denemekti.
- **"Port açık mı?" kontrolü BAŞLAMAKTA olan süreci göremez (Tur-80, yeniden başlatma olayı).** Kayıtçı portu
  import bitince bağlar (bu olayda 5 sn sonra); o pencerede ikinci bir başlatıcı (watchdog ↔ `proje-ac.ps1`) portu kapalı
  görüp ikinci süreci açar. Portsuz kalan süreç tarama döngüsünü yine koşturdu ve aynı pozisyonları ikinci kez
  kapatıp işlem defterine yazdı. Tekil bir kaynağa (bayrak, kilit dosyası, süreç listesi) bağlanmalı; buradaki
  en ucuz çözüm sıralamaydı: bayrak varken watchdog'lar pasif, bayrak servisler ayağa kalkınca kalkar. Yeniden
  başlatmadan sonra `Win32_Process` ile süreç SAYISINI doğrula; port yanıtı tek süreç olduğunu kanıtlamaz.
- **Kendi kilitlediğin diff'i de bağımsız incelet (Tur-80, süreç).** Denklik testleri ve mutasyonla sınanmış
  değişiklikte Codex 7 gerçek kusur buldu, 7'si bağımsız yeniden üretildi: benim ızgaralarım ve kurgularım
  üretimin "normal" şeklini taşıyordu, düşmanca girdiyi değil. Kanıt da iddiadır; ama incelemesiz kilit de.
- **Paylaşılan bir önbellek anahtarı, çağıranların FARKLILAŞTIĞI boyutu taşımalıdır (Tur-80, BULGU-80-4).**
  `bar_shadow` iki yoldan (1m tahmin · 1h tarama) aynı `sembol|durum` anahtarıyla çağrılıyordu; 1m getirisine
  fit edilmiş HMM 1h rejim olasılığını üretiyordu ve hata hiçbir yerde görünmüyordu. Önbelleğe anahtar yazarken
  `grep` ile TÜM çağıranları bul ve girdilerinin hangi boyutta ayrıştığını sor. Kusuru bulduran şey küçük bir
  inceleme iddiasını (kayıp güncelleme) doğrulamaktı — iddianın kendisi değil, doğrulamanın yolu.
- **"Bağlı" bir akış veri alan akış değildir; sağlığı SUNUCUNUN ONAYINDAN ve gelen mesajdan oku (Tur-82,
  BULGU-82-1).** Akış aylarca 0 mesajla çalıştı; `connected` (SDK döngüsü) ve `active: alpaca_stream`
  (sağlayıcı sayacı) ikisi de yeşildi. Asıl hata 405'ti ve SDK onu stdlib `logging`e yazdığı için loguru
  kullanan sürecin logunda görünmedi (tarih önekli satırları süzen her arama da kaçırdı). Bir aboneliği kurarken
  sunucunun "subscription" yanıtını durum alanına YAZ; niyet edilen abonelik bir ölçüm değildir.
- **Bir kesicinin kapsamı docstring'iyle değil istisna HİYERARŞİSİYLE sınanır (Tur-82, BULGU-82-2).** "Yalnız
  bağlantı hatası" yazıp `httpx.TransportError` yakalamak ReadTimeout'u da kapsar; ayakta ama yavaş bir hub'ın
  tek geç yanıtı sürecin TÜM çağrılarını 30 sn yedeğe yolladı ve yavaşlığın nedeni (429 uykusu) bu yüzden
  kendini BESLEDİ. Kesiciyi yaz → her alt sınıf için "bu hub KAPALI mı demek?" sorusunu testle sor.
- **Pahalı bir girdinin maliyeti, onu kullanan adıma ERİŞİM oranıyla çarpılır (Tur-82, BULGU-82-3).** Zincir
  her taramada çekiliyordu, kullanıldığı kalite adımına ise WAIT taramalarının %0'ı ulaşıyordu. Girdiyi
  tüketicisinin yanına taşımak (tembel değerlendirme) kararı değiştirmeden işi siler — ama önce girdinin
  BAŞKA bir yan etkisi var mı sor (burada IV geçmişi: günün son gözlemi → açık bir örnekleme aralığına
  dönüştü ve etkisi replay ile ölçüldü).
- **Bir koruma, başka bir kusuru GİZLİYOR olabilir; onu daraltan düzeltme gizleneni açığa çıkarır — dağıtımdan
  sonra PATLAMA yolunu da ölç (Tur-82, BULGU-82-6).** Fazla geniş kesici, hub'ın 429 uykusunu tarama
  işçilerinden saklıyordu; kesici doğru kapsama inince açılış taraması 4× uzadı. Kararlı durum ölçümü bunu
  göremezdi: kusur yalnız 4 saatte bir koşan toplu işte görünür.
- **Dosyaları değiştiren bir koşucu çalışırken o dosyalardan kanıt alma (Tur-82, süreç).** Gemini'ye giden
  diff, mutasyon koşucusu bir kusuru enjekte etmişken alınmıştı; inceleme o kusuru (doğru olarak) buldu ve
  "gerçek kusur" diye okunabilirdi. Diff/karma/kopya ancak koşucu bitip karma doğrulandıktan sonra üretilir
  (Tur-73'ün "devir bitene kadar dokunma" kuralının OKUMA tarafı).
- **İki tahmin varyantı AYNI referans fiyattan ölçülmeden kıyaslanamaz (Tur-83, BULGU-83-2).** Yayın anından
  SONRA gelen bilgiyi (bar kapanışından ~39 sn sonraki kotasyon) kullanan yön, yayından ÖNCEKİ fiyattan (bar
  kapanışı) ölçülürse ileri-bakışla kazanır: `hit_bar` canlı-yöne %60,9 verdi, aynı dilim aynı referanstan %26,2.
  Tur-60 bu ölçüyle kötü varyantı beş hafta daha yayında tuttu. Kural: A/B'de iki kolun ölçüm başlangıcı, kolların
  bilgisinin EN GEÇ olduğu andır.
- **Bir skorun "ters kalibrasyonu" piyasa özelliği sayılmadan önce GİRDİLERİNDEN biri gürültü mü diye sor (Tur-83,
  BULGU-83-1).** Güven↔isabet tersliği üç belgede (§17-C/D, Tur-47, `config.py` yorumu) "gerçek ve stabil" diye
  duruyordu; kotasyon yön girdisinden çıkınca kovalar düzleşti (%52,6→%44,6 ↔ %50,3→%49,2). Terslik, kovalanan
  gürültünün geri dönüşüydü — ve bir "fade" kuralı onu kenar sanabilirdi.
- **Bir kalibrasyon eğrisi bir ARTEFAKTTIR: alanını (hedef, girdi, aralık) taşımıyorsa yüklenmez, tazeleyeni yoksa
  bayatlar (Tur-83, BULGU-83-3).** 87 günlük, örneklem-içi, başka hedefe kurulmuş eğri sabit bir sayıdan KÖTÜYDÜ
  (t=+4,38) ve her panoda "kalibre olasılık" diye gösteriliyordu. Tur-41'in nöral şema kapısının olasılık tarafı.
- **Kümeli t'de küme AĞIRLIĞI bir karardır (Tur-83).** Gün ortalamalarını eşit tartan t (1 satırlık gün = 9.232
  satırlık gün) −1,40; satır ağırlıklı CR0 −2,92. Servisin kısa koştuğu günler bu defterde gerçektir.
- **Beraberlik ıska sayılıyorsa 0,5 yazı-turanın beklenen isabeti değildir (Tur-83).** %8 beraberlikli bir evrende
  yazı-tura `hit_rate` 0,46 verir ve 0,5'e kıyaslayan hüküm onu INVERTED okur.
- **Bir öneri doğru BAYRAĞI yanlış KUSUR için önerebilir (Tur-83, süreç).** Başka bir oturum "eğri bayat çünkü
  `--calib` verilmiyor" dedi: bayrak gerçekten eksikti ama eğriyi YAZMAZ. Öncül ölçülmeden uygulansaydı eğri bayat
  kalır, sorun kapandı sanılırdı (Tur-77: inceleme bulgusu da bir iddiadır).
- **Bir "N bar sonra" ufku, seri düzenli olmadıkça bir ZAMAN ufku değildir (Tur-84, BULGU-84-1).** Tur-51'in dolt
  dersi canlı defterde ikinci kez: işlem olmayan dakika bar üretmez; ufuk %18,6 satırda uzuyor, geceyi aşıyordu.
  İsabeti değil OYNAKLIĞI bozuyordu — yani kusur, yönle ölçülen her şeyde GÖRÜNMEZDİ.
- **Yön becerisi yokken yön-işaretli nokta tahmini "değişmez"den kötüdür (Tur-84, BULGU-84-2).** Hataya ATR² ekler
  (t=+8,7). Doğru hesap öngörülebilir olanın (oynaklık) üzerine kurulur: aralık.
- **Gösterilen fiyat en iyi çapa olmayabilir (Tur-84, BULGU-84-4).** Kotasyon şu anki fiyatın gürültülü bir tahmini;
  bar kapanışına çapalı aralık skoru %23 iyileşti (t=−10,7). Bir kuantilin "garip" çıkması (V z=−6,98) bir piyasa
  özelliği değil ölçüm girdisinin kusuru olabilir.
- **Takvim anlamı taşıyan bir referans (seans açılışı) bir pencere sınırından türetilemez (Tur-84, BULGU-84-3).**
  `normalize()` UTC gece yarısıdır; 1m penceresinde bu bir seans-öncesi işlem ya da pencerenin başıdır.
- **Başarısızlık da önbelleklenir mi? sorusu ayrıca sorulmalı (Tur-84).** Geçici okuma hatası imzayla yazılırsa
  kalıcılaşır (dosya değişene kadar "—"); kalıcı "yok" hiç yazılmazsa istek fırtınası doğar — cevap kısa negatif TTL.
- **Bir kilit yalnız SIRALAMAYA bakıyorsa iki uçtan birini bozan mutasyonu kaçırır (Tur-84/85).** `band_lo <
  last_close < band_hi`, yalnız `band_lo`yu kotasyona çapalayan mutasyonda da doğruydu (98,21 < 100 < 100,8).
  Bir hesabı kilitlerken her çıktıyı TAM DEĞERLE kilitle.
- **Toplam kapsama nominaldeyken aralık koşullu olarak bozuk olabilir (Tur-85, BULGU-85-1/2).** %79 toplam, ATR
  ondalığında %70,6 → %86,4'ü ve son yarım saatte %67,7'yi saklıyordu. Aralık skoru farkı küçük gösterir (−0,7 bps),
  koşullu tablo büyük gösterir; ikisi birlikte raporlanır.
- **Bir açığın önerilen ilacı ters yönde olabilir — önce KOŞULLU tabloyu ölç (Tur-85, BULGU-85-0).** "ATR yavaş →
  daha hızlı pencere" 20 adayda da kötüleştirdi; eksik olan HIZ değil ORTALAMAYA DÖNÜŞTÜ ve ilaç tersiydi (düzeye büzme).
- **Veriye bakılarak kurulan bir özelliği kapı değil ÖRNEKLEM-DIŞI TEKRAR temizler (Tur-85, BULGU-85-2).** Kapanış
  çarpanı yürüyen kapıyı geçmişti (t=−2,75); sınır tablolara bakılarak seçildiği için bağımsız incelemeci (Gemini)
  itiraz etti. Tartışmak yerine bulguların hiç bakılmadığı bir dönemin barlarıyla aynı defter satırları kuruldu:
  çarpan 60 günün HİÇBİRİNDE seçilmedi, üs ise 60/60 gün (t=−8,46) tekrarladı. Kod yazılmış ve mutasyonla
  kilitlenmişti; geri alındı. Aynı-dönem kapısı parametreyi dürüst seçer ama özelliği seçen gözü görmez.
- **Her gece yeniden sınanan boş bir özellik er geç sahte bir "geçti" üretir (Tur-85).** Gece görevi büyük ölçüde
  ÖRTÜŞEN veride her gece aynı t'yi yeniden hesaplar; etki yoksa bile eşik bir gece aşılır ve özellik sessizce
  yayına girer. Tekrarlanmayan bir özelliği "nasılsa kapı korur" diye kodda bırakma — çıkar ve geri gelmesini kilitle.
- **Parametre seçimi nedensel olsa da YAPI kararı tüm dönemden verilirse özet iyimserdir (Tur-85, Gemini #1).**
  Üs her gün yalnız geçmişten seçiliyordu ama "üs açık mı" kararı tüm yürüyen dönemin kapısından geliyordu. Tam
  nedensel prosedür (kapı kararı da her gün yalnız önceki test günlerinden) ölçüldü: −0,43 bps (t=−2,24) ↔ özetin
  −0,78'i. Kazanç gerçek ama rapor iki sayıyı birlikte verir; ilk ~17 test günü kapı kanıt biriktirirken sabit kalır.
- **Zamanlanmış görev çalışma ağacını koşturuyorsa her kaydedilen düzenleme bir dağıtımdır (Tur-85, süreç).**
  `NihaiAlgo-ForwardHarness-Daily` 12:00 ve 23:30'da `forward-harness.ps1`i çalışma ağacından koşturur; bu tur 12:00'de
  ara kodla koştu. Çok dosyalı bir değişikliği 12:00/23:30'a TUTARLI bir noktada gir; dosyaların tutarlılığı
  kalite kapısından önce de korunmalı.
- **Hız/VRAM ölçümü bir BECERİ ölçümü değildir (Tur-86).** Bir dış sohbet 5,4M→2B basamakların eğitilmemiş
  ağırlıklarla gecikme/bellek tablosunu kurup "en sağlıklı kararı XL üretir" dedi; hiçbir basamakta isabet
  ölçülmemişti. Aynı soru GPU'da ölçülünce (purged CV, aynı veri) 5,4M'lik transformer hiçbir test örneğinde
  çoğunluk sınıfından ayrılmadı. Bir model önerisinin kanıtı AYNI veride purged-CV + DSR'dir; gecikme yalnız
  "çalışabilir mi?" sorusunu yanıtlar.
- **Bir CLI varsayılanı, çağırdığı fonksiyonun dürüstlük varsayılanını sessizce ezebilir (Tur-86, BULGU-86-1).**
  `argparse` `default=0` her koşuda iletildiği için `lambda_sweep(exec_lag=1)` hiç devreye girmedi ve belgelenen
  tekrar-üretim komutu yayımlanmış artefaktı üretemiyordu. Farklı varsayılanlı iki fonksiyona tek bayrak
  bağlanıyorsa bayrak `None` olmalı ve verilmeyen değer İLETİLMEMELİ.
- **Boş kümeden kurulan hüküm doğru cümleyle yanlış bilgi verir (Tur-86, BULGU-86-2).** "Hiçbir λ pozitif fazlalık
  vermedi" hiçbir λ konuşmadığında da doğrudur; ardından gelen "ABSTAIN ölçülmüş doğru davranış" o koşuda
  ölçülmemiştir. Aynı sınıf aynı turda ikinci kez çıktı: kapasite hükmü "max DSR=None ≤ 0,05" basıyordu. Her hüküm
  dayandığı kümenin boş olmadığını ayrıca sormalı ve bunu makine-okunur bir KODLA taşımalıdır (Tur-60'ın "ölçülemedi
  üç ayrı şeydir" kuralının hüküm metnindeki hâli).
- **Dış bir sayfayı KONUMLA okuyan ayrıştırıcı sayfa yeniden düzenlenince yanlış tabloya geçer; kimse koşmazsa
  kırıklık haftalarca görünmez (Tur-86, BULGU-86-3).** Wikipedia tabloyu taşıdı, `tabs[1]` bir navbox oldu ve
  ayrıştırıcı `KeyError` ile düştü; bu şanstı. Alt başlığı değişen bir tablo ise HATA VERMEDEN boş geri-sarma üretip
  güncel kümeyi geçmiş üyelik diye döndürürdü (sessiz survivorship). Tabloyu İMZASIYLA ara; beklenen kolonların
  HEPSİ kurulamazsa yüksek sesle dur.
- **"Pencerede herhangi bir an üye" evreni, karar günü üyeliği değildir (Tur-86, BULGU-86-4).** Endekse eklenme
  yükselişin sonucudur; sonradan eklenen isim geçmiş kesite girince momentum benzeri bir aile sahte anlamlılık
  kazanır (kalıntı momentumunda gün-t 1,63 → 2,23; vixts'te kirli satırların pnl'i t=+4,23 yüksek). Kirlenmenin
  yönü aileye bağlıdır (babmacro'da t=−2,18), bu yüzden "zaten FAIL" demek hükmü korumaz. PIT evren, her karar
  gününde O GÜNÜN üyeleridir.
- **Bir ders yazılı olması UYGULANDIĞI anlamına gelmez; kendi raporundaki "kazanç" cümlesine de sor (Tur-88,
  BULGU-88-3).** ACI DERS #9 sekiz tur boyunca makineye geçirildi ve Tur-86 yine *"ölçülen tek kazanç
  KALİBRASYON (ECE 0,051 → 0,027)"* yazdı. Aynı fold'larda sabit %34 tahmincisine karşı Brier beceri skoru 15
  hücrenin 14'ünde NEGATİF çıktı. Kural: bir iyileşme cümlesi yazmadan önce "neye göre?" sorusunun cevabı aynı
  cümlede olmalı; ECE'nin düşmesi, taban orana yaklaşmanın da imzasıdır.
- **"Ready" bir görev koşuyor demek değildir; KAÇIRILAN tetik sayısını oku (Tur-88, BULGU-88-1).** Görev
  `Ready`/sonraki tetik doğru gösterirken makine 16 saat uyudu, iki tetik düştü (`Missed=2`), `StartWhenAvailable`
  uyanışta koşmadı ve öğrenme hafızası 28,5 saat bayat kaldı. Görevin ürettiği özet kendi YOKLUĞUNU raporlayamaz;
  tazelik, görevden BAĞIMSIZ bir izleyicide (deadman) ve ürünün kendisinde (dosya yaşı) ölçülür.
- **Bir ajan oturumu işi yarıda bırakıp başka projeye geçince iş ÜRETİM YOLUNDA kalır (Tur-88).** Codex aday
  eğitimini gece görevine bağlamış, bir hatayı düzeltirken başka projeye yönlendirilmişti; raporsuz ve izlenmeyen
  dosya, birkaç dakika sonraki 23:30 tetiğinde gerçek veride ilk kez koşacaktı. Devralan tur önce
  `git status`taki raporsuz dosyaların ÇAĞIRANINI arar (`grep` modül adı), sonra zamanlanmış görevin saatine
  kadar bağlı yolu test eder ve o saatten sonra o dosyalara dokunmaz.
- **`pd.to_datetime(..., utc=True)` karışık biçimde "hizalamaz", SİLER (Tur-88, BULGU-88-2).** pandas biçimi ilk
  elemandan çıkarır; ikinci biçim `errors="coerce"` ile NaT olur ve `chrono_sort` onu sona koyar. Canlı sette
  %21,2 satır kronolojinin sonuna düşüyordu ve üç docstring tersini iddia ediyordu. Bir tarih ayrıştırıcısının
  sözleşmesini gerçek defterin biçim DAĞILIMIYLA (`s.str.len().value_counts()`) sına; kontrol testi eski çağrının
  gerçekten satır düşürdüğünü de göstermeli (guard kargo-kült değil).
- **Bir günlüğü İZLEMEK onu kilitleyebilir; günlük yazımı da işi durduramamalı (Tur-88, BULGU-88-4).** Kaçan
  koşuyu gözlemek için `forward-harness.log`a Git-Bash `tail -F` bağlandı; 23:30 tetiği PS 5.1 `Add-Content`te
  "dosya kullanımda" aldı ve `ErrorActionPreference=Stop` yüzünden BAŞLAMADAN öldü (sonuç 1, günlükte tek satır
  yok) — izleyici, izlediği olayı yok etti. Karalama kopyasında birebir yeniden üretildi. İki ders: canlı bir
  görevin yazdığı dosyayı tutan okuyucu açma (görev durumunu `Get-ScheduledTaskInfo` ile yokla); ve bir işin
  günlüğü o işin ön koşulu olamaz — `forward-harness.ps1` artık `Write-Log` ile yeniden dener, olmazsa yan
  dosyaya yazar ve asla fırlatmaz.
- **İşaret-yalnız bir "onay", sıfır etrafındaki gürültüde hükmü çevirir (Tur-88, BULGU-88-5).** Gölge terfi
  kapısı kayma kontrolünü ve OOS onayını `uplift > 0` ile veriyordu. `conf_bar`ın ortak-referanslı uplift'i iki
  gece arasında +0,0000 (t=0,02) → +0,0015 (t=0,57) oldu ve etiket SLIP_ARTIFACT'tan PROMOTE'a döndü — bilgi
  değişmedi, işaret değişti. Her gece yeniden koşan bir kapıda anlamlılık şartı olmayan her adım, er geç
  gürültüyle geçilir (Tur-85). Sonuç görüldükten sonra kapı sıkılaştırılmaz: kural DONDURULUR ve yalnız sonraki
  verilerle, ön-kayıtlı anlamlılık eşiğiyle sınanır.

- **Yeni bir maliyet modeli canlıya girince GÜNLÜK maliyeti öncesiyle kıyasla; tek borsanın kotasyonu piyasanın
  dolum fiyatı değildir (Tur-89, BULGU-89-1).** Yarım-makas kayması "ölçülmüş" etiketiyle girdi ve kaynağın IEX
  (NBBO değil) olduğunu taşımadı; giriş kayması günde ≤18 k$'dan 58–92 k$'a sıçradı, 35 giriş hedefin ötesinde doldu
  ve sanal hesap altı işlem gününde %92 eridi — her satır "doğru loglandığı" için kimse fark etmedi. Kural: bir icra
  modelini açtıktan sonra `slip$`/gün ve "dolum hedefin ötesinde mi?" sayısını ÖLÇ; 5× sıçrama önce ölçüm kusurudur.
  RR kapısı ve boyut İŞLEM görülebilecek fiyatta (dolum) kurulur, sinyal fiyatında değil.
- **Bir kapıyı "ölçülemez" diye bırakan ÖNCÜL de ölçülmelidir (Tur-89, BULGU-89-2).** Tur-50 brüt maruziyeti
  "iki BAĞIMSIZ katman tek deftere yazıyor" diye ölçülemez saydı; `main._scan_tier` iki katmana AYNI `risk_mgr`/
  `store`/`pmgr`'yi verir → tek portföy. Öncül yanlış olunca eksik bırakılan kapı (toplam alım gücü) altı hafta
  boyunca 35× kaldıraca izin verdi. Bir formülde birim iki kez sayılıyorsa (`2×özkaynak / marj`) etiket ("Reg T")
  ile hesabın aynı şeyi söylediğini sayıyla sına.
- **Seans dışında dönen bir döngü, gerçek kararları kendi gürültüsüne gömer (Tur-89, BULGU-89-4).** Katmanlar kapalı
  piyasada 5 sn'de bir 550 sembolü reddediyordu; pano "Sinyal Akışı" son 1.000 satırı okuduğu için gerçek kapanışlar
  görünmez oldu ve konsol akışı `[state] yüklendi` seliyle doldu. Bir işi reddedecek kapı biliniyorsa işi hiç
  başlatma; bir log satırı yalnız DURUM DEĞİŞİNCE bilgi taşır.
- **Bir düzeltmenin "kardeş yolda da gerekir" cümlesi yazıldıysa, o yol ayrı bir iştir — yapılana kadar kusurdur
  (Tur-90, BULGU-90-1).** BULGU-19 araştırma yolunu düzeltirken canlı yolun AYNI kusuru taşıdığını kendi yorumuna
  yazdı; cümle yedi hafta yorumda kaldı ve canlı tarayıcı bölünmüş hisseleri −%50…−%89 sahte barla taradı. Tur-47'nin
  "yorum hüküm değildir" dersinin düzeltme tarafı: kardeşine uygulanmayan onarımın yeri yorum değil açık kalem/test.
- **Bir veri kuralının yazılı olduğu katman, kuralın uygulandığı katman değildir (Tur-90, BULGU-90-2).** Madde 22
  (`data_provider.py` başlığı) yfinance'te bir parametreydi (`prepost=False`); Alpaca'ya geçilince kural taşınmadı ve
  canlı sistemin ANA veri yolu seans-dışı barlarla koştu. Sağlayıcı değiştiren her göçte "önceki sağlayıcının SESSİZCE
  yaptığı" süzgeçler (seans, düzeltme, zaman dilimi) tek tek sorulur; üçü de bu turda eksik çıktı.
- **İki kaynağı birleştiren kod, AYNI şeyi AYNI adla etiketlediklerini varsayamaz (Tur-90, BULGU-90-3).** Canlı
  günlük toplam UTC gece yarısı, REST ET gece yarısı etiketliyordu; "tekilleştir" adımı yalnız aynı indeksi eledi ve
  her gün iki kez kaldı. Birleştirme testi iki kaynağın gerçek etiket biçimiyle kurulmalı; sentetik aynı-etiket
  kurgusu kusuru göremez.

- **Yedek sağlayıcı birincilin ÜRETMEDİĞİ biçimde veri veriyorsa her yedeğe düşüş sessiz bir strateji değişikliğidir
  (Tur-91, BULGU-91-1/2).** Failover "aynı veri başka yerden" sayılıyordu; yfinance 1h 09:30'a, Alpaca 09:00'a hizalıydı ve
  temettüyü farklı düzeltiyordu. 429 her gün yüzlerce çekimi yedeğe düşürdüğü için canlı sistem gün içinde iki ayrı barla
  koştu (karar %64 ayrışıyor) ve hiçbir log satırı bunu söylemedi. Failover kurarken iki kaynağın AYNI anda AYNI satırı
  ürettiği (damga, düzeltme, seans) ölçülmeli; ortak ızgara ikisinin de kurabildiği ızgaradır.
- **"Bu değişiklik strateji değişikliğidir, önce backtest" demeden önce BUGÜNKÜ durumun tek strateji olduğunu doğrula
  (Tur-91).** Tur-90 ızgara göçünü erteledi; oysa canlı yol zaten iki ızgaranın karışımıydı ve erteleme karışımı
  koruyordu. Ertelenen göç ölçülünce (backtest, tek değişken) ayırt edilemez çıktı (t=+0,16).
- **Deterministik bir hesap aynı girdiden farklı sonuç veriyorsa girdi aynı değildir (Tur-91, BULGU-91-4).** Kapalı
  piyasada değişmeyen barlarla koşan tarayıcı 63 → 64 → 63 "geçen" üretti; kaynağı sağlayıcı karışımıydı. Tekrarlanan
  bir işin çıktısındaki oynama, işi kısmadan önce SORULACAK ilk sayıdır.
- **Bir veri yolunu düzelten tur, o yolun TEKRARCILARINI da aynı sözleşmeye çekmeli (Tur-91).** Tur-90 canlı barı seansa
  süzdü; "canlı hattın izole tekrarı" olan replay araştırma önbelleğinin uzatılmış seanslı saat barını okumaya devam etti.
  Tur-90'ın "kardeşine uygulanmayan onarım" dersi bir tur sonra kendi kardeşinde tekrarlandı.
- **Bir kapının geçilebilir olduğunu EN İYİ girdide sına: "kanıt yok" ile "kapı duvar" aynı çıktıyı verir (Tur-92,
  BULGU-92-2).** Fayda kapısı post=1'de bile negatifti; 9.631 gölge karar ve 7.711 canlı verdiktin %100 ABSTAIN'i "kenar
  yok" diye okundu, oysa motor tanım gereği konuşamıyordu. Tur-86'nın λ süpürmesi kapı-a kapalıyken yapıldığı için
  göremedi. Her kapı için E(en iyi durum) > 0 sorusu sorulur; bir cezanın HANGİ DALDA ödendiği formülde yazılır.
- **Bir veri kapısı ile onu açan seçici ayrı yerde tanımlanıyorsa, kapının açıldığı değer seçicide var mı diye bak
  (Tur-92, BULGU-92-1).** Tur-84 aralığı yalnız `1m`'de açtı; detay seçicisinde `1m` yoktu → alan sekiz tur boyunca her
  durumda "—" bastı ve açıklaması ("yalnız 1dk") kullanıcıya olmayan bir seçeneği gösterdi.
- **Bir modelin dağılım kapısı modeli ÇAĞIRAN her yolda olmalı (Tur-92, BULGU-92-3).** Tur-20'nin OOD kapısı canlı
  köprüdeydi; gölge koşucu aynı modeli günlük özellikle kapısız çağırdı (max|z| 71). Kapı modele değil çağırana yazılmışsa
  her yeni çağıran onu yeniden yazmalıdır — `grep` ile model yükleyicisinin TÜM tüketicileri sayılır.
- **"Canlıya al" bir KİP kararıdır; önce o kipin bugün ne yapacağını ölç (Tur-92).** `enforce` açılsaydı `ABSTAIN = veto`
  sözleşmesiyle her sinyal susardı. Operatörün niyeti (Quant karara katılsın) ile kipin bugünkü etkisi (sistem susar)
  ayrışıyorsa, niyeti bozmayan en güçlü kip seçilir ve öbürünün koşulu (ölçülen kanıt) yazılır.
- **Grafik kütüphanesinde boyut ve senkron bir SÖZLEŞMEDİR (Tur-92, BULGU-92-4).** Oluşturma anında okunan genişlik gizli
  kapsayıcıda 0'dır; mantıksal indeksle senkron ancak zaman kümeleri VE eksen genişlikleri aynıysa hizalıdır (ısınma
  boşluğu 14 bar, eksen farkı 5 px kaydırdı). Doğrulama sayısal yapılabilir: `timeToCoordinate` üç grafikte aynı mı.
- **Bir modelin canlı maliyeti parametre sayısıyla değil İLERİ GEÇİŞ SAYISI × çağrı hızıyla ölçeklenir (Tur-93,
  BULGU-93-1).** 282M MoE GPU'da bile MC=20 döngüsünde 1,8 sn sürdü (20 sıralı geçiş, küçük çekirdekler); aynı 20
  örnek tek toplu geçişte 0,1 sn. Büyük bir modeli canlı döngüye almadan önce, aynı mimarinin eğitilmemiş bir sonda
  artefaktıyla duvar süresini ölç ve döngünün talep hızıyla (aday/dk, kilit altında sıralı mı) çarp — eğitimi bekleme.
- **Çıktısı aynı, maliyeti T katı bir değişikliği çıktı testleri göremez (Tur-93).** "Toplu yol yine T kez geçer"
  mutasyonu tüm doğruluk testlerinden kaçtı (ilk geçişin sonucu aynıydı). Bir hızlandırmanın iddiası (tek ileri geçiş)
  ayrıca SAYILARAK kilitlenir (ileri kancası).
- **Doyan bir test modeli MC yayılımını sıfırlar (Tur-93).** Güçlü sinyalle eğitilen küçük model bazı satırlarda p=1,
  std=0 verdi ve "kopyalar bağımsız" testi yanlış kırmızı oldu; satır eşlemesi ise zayıf modelde hiç ayırt
  edilemiyordu. MC testleri modelin KARARSIZ olduğu girdide, eşleme testleri AYIRT ettiği girdide kurulur.
- **Tarihsel bir eğitim seti canlının EŞİ sayılmadan önce ETİKETİNİN çapası canlıyla eşleştirilmeli (Tur-94,
  BULGU-94-1).** Özellikler canlı yolun kendi fonksiyonlarıyla kurulmuştu ve pariteyi geçti; etiket ise bar kapanışına
  çapalıydı, canlı etiket ~39 sn sonraki kotasyona. Tarihsel testte "veri arttıkça AUC yükseliyor" görüntüsü 1 bar
  gecikmede tamamen yok oldu (0,5215 → 0,5004). Kural: eşleşen satırlarda iki etiketin UYUMUNU ölç ve aday çapaları
  tek tek dene; öğrenme eğrisini ayrıca 1 bar gecikmeyle sına (ACI DERS #7).
- **Veriyle yükselen bir eğri, hedef EVRENDE yükseliyor mu diye ayrıca sor (Tur-94).** 520 sembolde AUC 0,510 →
  0,521 yükseldi; canlıda izlenen 30 sembolün aynı satırlarında 0,502 → 0,505 kaldı. Test kümesini hedef evrene
  (test C) ve hedef etikete (test A) ayrı ayrı daraltmayan bir pilot "10 yıla çıkalım" kararını yanlış verirdi.
- **Çıktı std'si sıfır olan bir modelin AUC'si ölçüm değil yuvarlama gürültüsüdür (Tur-94).** 282M dört noktada da
  her satıra aynı olasılığı verdi (std < 5e-5); AUC 0,493–0,513 arasında oynadı ve 0,5125 tek başına "gelişme" gibi
  okunabilirdi. Eğri raporuna tahmin std'si yazılmalı; sabit çıktıda veri miktarı hakkında hüküm kurulamaz.
- **Bir eğitim yardımcısının bellek maliyeti BÜYÜK n'de ayrıca sınanmalı (Tur-94, BULGU-94-2).** `fit` val kümesini
  tek geçişte değerlendiriyordu; bütün geçmiş eğitimler küçük setlerde koştuğu için görünmedi. 64 bin satırda 282M
  VRAM'i sistem RAM'ine taşırdı (boş RAM 0,4 GB) ve hata vermeden yavaşladı. Yeni bir ölçekte ilk koşuda süreç
  belleğini ve paylaşımlı GPU belleğini ölç.
- **Mutasyon geri alması global bir metin değişimi olamaz — orijinal içeriği yaz (Tur-94, süreç).**
  `replace(mutant, orijinal)` mutant metni (`"LONG"`) dosyanın başka satırlarında da geçtiği için üç satırı bozdu;
  koşucunun MD5 denetimi yakaladı. Tur-74'ün "hedefli ters değiştirme" kuralı yetmez: ters değişim de tekil olmalı,
  en güvenlisi dosyanın mutasyon öncesi içeriğini geri yazmaktır. Kaçan iki mutasyon yine dejenere kurgudan çıktı
  (sonraki barın ortası = karar barının kapanışı; yalnız LONG satır) — Tur-73 dersi.
- **Bir backtest'in içindeki KALICI bir anahtar tetiklendiğinde, metrik dönemi değil anahtarın açık kaldığı kısmı ölçer —
  sonuç bunu taşımıyorsa kimse görmez (Tur-95, BULGU-95-3).** Kill-switch 35–50. işlemde tetiklenip portföy koşularının
  %93–96'sını girişsiz bıraktı; dünkü deneyin tabanı ve benchmark'ın makro bölümü "2 yılda −%4,9" diye okundu. Sıfır-kenarlı
  bir stratejide "son 30+ işlemin ortalaması < 0" kuralı YAPISAL olarak erken tetiklenir (sentetik 24 koşunun 21'i). Bir
  backtest okunmadan önce maruziyet (girişsiz bar payı, işlemlerin tarih dağılımı) sorulur.
- **Karıştırma (permütasyon) Monte Carlo'su yalnız SIRAYA bağlı ölçüleri dağıtır (Tur-95, BULGU-95-2).** Nihai sermaye
  her yolda aynıdır; ondan türetilen "iflas olasılığı" 0 ya da 1'dir ve onu okuyan kapı hiç düşmez. Bir MC çıktısını
  tüketmeden önce sor: bu büyüklük örnekleme sırasına bağlı mı? Doğru ölçü depoda BAŞKA adla vardı (laboratuvar), karar
  tüketicileri eski adda kalmıştı — iki tanım yan yana yaşarken kök nedende tek tanıma çek.
- **Öz-referanslı bir test kayık bir indeksi göremez (Tur-95, mutasyon M5).** Kill-switch anı ile boşta payının
  tutarlılığını sınayan test anı raporlanan değerden türetiyordu; iki alanı birlikte kaydıran mutasyon kaçtı. Çapa, değeri
  üreten koddan BAĞIMSIZ kurulur (burada: kural kill-switch'siz koşunun kapanış sırasından yeniden hesaplandı). Kaçan
  mutasyon ayrıca tek sembol ↔ portföy tanımlarının bir iterasyon kayık olduğunu gösterdi.
- **Belgedeki komutun modülü var diye bayrağı var sayılamaz; tarayıcı da kodun GERÇEK çağrı biçimini okumalıdır (Tur-95,
  BULGU-95-1).** Tur-61 kapısı modül varlığını soruyordu; iki yüzey olmayan bayrak öğütlüyordu. Bu turun ilk tarayıcısı
  da yalnız `python -m` arıyordu ve gece görevinin `& $venvPy -m …` / dizi biçimli 23 çağrısını HİÇ görmüyordu. Bir
  tarayıcının kapsamı, bilinen kusurları YAKALAYIP doğru komutu GEÇİRDİĞİ bir kontrol testiyle sınanır.
- **Süreç içi CONFIG geçersiz kılma + yeniden kullanılan havuz işçisi = görevden göreve sızıntı (Tur-95, süreç).**
  `kural.uygula()` yalnız kendi listesindeki alanları tabana döndürür; stres koşusunun değiştirdiği `impact_coef` o
  listede yoktu ve aynı işçide sonraki göreve taşınacaktı (koşudan önce fark edildi). Tabana döndürülen alan kümesi,
  herhangi bir görevin değiştirdiği alanların ÜST KÜMESİ olmalıdır.

- **Havuzlanmış AUC enstrümanlar arası TABAN ORAN farkını "beceri" sayar (Tur-96, BULGU-96-1).** Yıl başına sabit
  olasılık veren taban kuralının Forex havuz AUC'si 0,522 çıktı (USDTRY neredeyse hep yükselir). Zamanlama becerisi
  enstrüman-İÇİ AUC'dir; birden çok enstrümanı havuzlayan her sınıflandırıcı karnesi ikisini ayrı basmalı ve AI'ı aynı
  havuzdaki sabit taban kuralının AUC'siyle kıyaslamalıdır (ACI DERS #8/#9'un kesit ekseni).
- **Bir vekilden kurulan ölçüm, vekilin kaynağını ölçer — modeli suçlamadan önce vekil/hedef SEVİYE oranına bak (Tur-96,
  BULGU-96-2/3).** HAR "EWMA'dan kötü" göründü; Garman-Klass RV vekili yfinance FX'te r²'nin 2,1 katı, SHEL.L'de 56 katıydı
  (2019'da low=3/23/26). Onarımdan sonra Avrupa'da HAR anlamlı İYİ (t −4,4), FX'te seviye düzeltmesiyle EWMA'ya eşit.
  Kapanıştan kurulan kusur kuralı bar içi yanlış baskıyı GÖRMEZ; aralık kullanan her özellik (ATR, range, RV) ayrıca
  denetlenmeli.
- **Havuz işçisi yeniden kullanıldığı için bir monkeypatch sarmalayıcısı görevden göreve ZİNCİRLENİR (Tur-96,
  BULGU-96-5).** Ölçüm betiğinde her görev `generate_signal`'ı öncekinin üstüne sardı: "+1 bar gecikme" +17 bara kadar
  büyüdü ve hükmü TERSİNE çevirdi (donchian "dayanıyor", event_breakout "çöküyor" — doğrusu tam tersi); rastgele kapı
  tüm işlemleri eledi ve bu ancak sıfır varyanslı Sharpe'tan görüldü. Orijinal fonksiyonu süreç başına BİR KEZ sakla;
  Tur-95'in CONFIG sızıntısı dersinin fonksiyon karşılığı.
- **Bir kapının "iyileştirmesi" aynı oranda RASTGELE eleyen kapıyla kıyaslanmadan okunamaz (Tur-96, BULGU-96-7).** Canlı
  meta kapı 14 piyasanın 8'inde event_breakout'u "iyileştirdi"; iyileştirdiği yerlerin çoğunda taban negatifti ve işlem
  kısmak |Sharpe|'ı sıfıra çeker. Rastgele kapıya karşı 8/14 = yazı-tura.
- **"Yükselen gün oranı" sürüklenme değildir (Tur-96, BULGU-96-9).** Türkiye ve kriptoda yükselen gün < %50 ama ortalama
  getiri > 0 (sağa çarpık): sınıf dengesini öğrenen yön sınıflandırıcısı/taban kuralı bu piyasalarda sürüklenmeyi TERS
  oynar (Sharpe −0,80 / −0,78). *(Tur-98 düzeltmesi: Türkiye yarısı TATİL SATIRI artefaktıydı — süzgeçle uzun payı
  0,42 → 0,76, Sharpe +0,94; kripto yarısı duruyor. BULGU-98-7.)*
- **Sabite çökmüş bir model sıralamada al-tut olarak "kazanır" (Tur-96, BULGU-96-10).** quant_transformer 14 piyasada
  std 0'a çöktü; pozisyonu "hep uzun" olduğu için Sharpe'ı al-tutla aynıydı ve özet onu 4 piyasada "en iyi AI" seçti.
  "En iyi" seçen her tablo tahmin std'sini (Tur-94) bir DIŞLAMA koşulu olarak taşımalıdır.
- **Kaçan mutasyon kurgunun dejenereliğini gösterebilir — ikinci kez (Tur-96).** GARCH'a ileri bakış eklemek testi
  geçti: sentetik seri i.i.d. idi, GARCH α sınırda 0 çıktı ve bugünkü getiri öngörüyü hiç etkilemedi. Oynaklık testleri
  oynaklık KÜMELENMESİ taşıyan veriyle kurulur (Tur-73 dersinin oynaklık hâli). Ayrıca `Path.write_text` ile yapılan
  düzenlemeler dosyayı sessizce CRLF'ye çevirdi ve mutasyon desenleri eşleşmedi — düzenleme de bayt düzeyinde yapılır.
- **Bir strateji değişikliğini "al-tuttan iyi mi?" diye değil "YERİNE geçtiği şeyden iyi mi?" diye sor — ikisini
  aynı tabloda ver (Tur-97).** `quant_trend` al-tutu 12/15 piyasada Sharpe'ta geçmedi ve bu tek başına okunsa kayma
  reddedilirdi; aynı motorda canlı algoritmaya (event_breakout) karşı fark 29/30 hücrede olumlu, örneklem-dışında
  15/15. Kıyas tabanı karar sorusundan türetilir: "canlıda neyi değiştiriyorum?" Al-tut satırı yine basılır, çünkü
  kenar iddiasını engelleyen odur.
- **Çıkışın mekanizması, girişin mekanizmasıyla aynı birimde kurulmalı: σ-MESAFELİ stop dolum fiyatına taşınır (Tur-97,
  BULGU-97-2).** Stop karar kapanışına göre seviye olarak kuruluyor, giriş sonraki açılıştan yapılıyordu; gece boşluğu ya
  da bir bar gecikme LONG stopunu girişin ÜSTÜNE taşıdı ve gecikme testi "beceri ilk barda" diye YANLIŞ hüküm verdi
  (ABD 0,48 → −0,19). Kırılım stopu bir SEVİYEDİR (yapısal kalır), oynaklık stopu bir MESAFEDİR (girişle yürür).
  Bir gecikme testinde çöküş görünce önce "gecikme sinyali mi bozuyor, geometriyi mi?" diye sor: işlem sayısı ve
  GAP_STOP sayısı artıyorsa geometridir.
- **Bellekte yaşayan durum, pozisyonun ömrü süreçten uzunsa kalıcı olmalıdır (Tur-97, BULGU-97-3).** Tracker'ın
  başlangıç stopu ve tepe fiyatı yeniden başlatmada pozisyonun O ANKİ stopundan kuruluyordu; seans içi TA
  pozisyonlarında bu hiç görünmedi, çok günlük quant pozisyonunda izleyen mesafeyi her sabah büzerdi. Bir kuralın
  ömrünü değiştiren (gün içi → çok gün) her değişiklik, "bu kural hangi durumu okuyor ve o durum yeniden başlatmada
  nereden geliyor?" sorusunu yeniden sorar.
- **Bir modelin kapsamı, eğitildiği sinyal ailesidir — yeni aileyi eski modele sokma (Tur-97, BULGU-97-4).** TA
  sinyalleriyle eğitilen meta kapı quant sinyallerinin HEPSİNE p_win=0,00 verdi (dağılım-dışı doyum); gölgede bile
  loglansaydı kapının kanıt defterini sessizce kirletirdi. Yeni bir strateji kipi eklerken, sinyali tüketen her modeli
  `grep` ile bul ve kapsam dışıysa açıkça atla (`KAPSAM_DISI_*`).
- **Veri kaynağının "işlem olmayan gün" satırı bir oynaklık gözlemi değildir (Tur-97, BULGU-97-5).** yfinance yerel
  tatillerde fiyatı kopyalayıp satır verir; EWMA bunu bir sıfır getiri olarak sindirir, ağaç tabanlı ML ise "sıfır
  aralık → sıfır oynaklık" kuralı öğrenip ertesi gün 1e-8 varyans basar. Kusuru gösteren ölçü QLIKE'ın uç değeriydi;
  ortalama bir metrik onu gizlerdi. Yeni bir öğrenen modeli eski veriyle beslerken dejenere satırlar (sıfır aralık,
  kopya kapanış) ayrıca sayılır — basit modelin sindirdiği şeyi esnek model sömürür.
- **ACI DERS #9 pozisyon uzayında: bir yön modeli "hep uzun" politikasına karşı sınanmadan zamanlama becerisi
  iddia edemez (Tur-97, BULGU-97-6).** Lojistik model kuralı 13/14 piyasada "geçti"; uzun kalma payı %100'dü ve
  Sharpe'ı sabit politikanınkiyle 14/14 aynıydı — sürüklenmeyi öğrenmişti. Tur-96'nın `cokmus` bayrağı (std≈0) bunu
  yakalamaz: olasılık oynar ama hep aynı taraftadır. Her yön karnesi uzun payını ve sabit tabana karşı farkı taşır;
  sürüklenmesi çıkarılmış hedefle eğitilen ikiz model zamanlamayı ayrıca ölçer.
- **Öngörü kaybını iyileştiren model, kararın mekanizmasında ölçülmeden bağlanmaz — ve ekonomik test kararın
  mekanizmasını taklit etmelidir (Tur-97, BULGU-97-7).** σ_ML QLIKE'ta EWMA'yı 14/14 yendi; günlük yeniden dengelemede
  3-8 kat devirle her yerde kaybetti; ama canlı quant σ'yı yalnız GİRİŞTE kullanır, yani o test yanlış mekanizmayı
  cezalandırıyordu. Motor testi (doğru mekanizma) ön-kayıtlı ölçüde yazı-tura çıktı. Ayrıca ölçü seçimi de bir
  karardır: boyutlama girdisinde ortalama getiri farkı daha küçük maruziyeti ceza sayar; risk-ayarlı fark önceden
  seçilmeliydi. Sonuç görüldükten sonra ölçü DEĞİŞTİRİLMEZ; aday bir sonraki ön-kayıtla sınanır.
- **Yan etkisini yakalamak istediğin mutasyonu iddianın DALINA kur (Tur-97).** "Quant dalı opsiyon zincirini çağırmaz"
  kilidi için kurulan ilk mutasyon TA dalını bozuyordu ve KAÇTI; kilit sağlamdı, mutasyon yanlış yerdeydi. Kaçan bir
  mutasyonun dördüncü anlamı: mutasyon, sınanan iddianın kod yolunda değil (Tur-72 ölü guard, Tur-73 dejenere kurgu,
  Tur-95 öz-referanslı test).
- **pandas 3 zaman indeksi ns olmayabilir: tamsayı damgaya (`asi8`) sabit birimle eşik koyma (Tur-97, BULGU-97-1).**
  `datetime64[us]` indekste ns eşiği günlük barı saatlik saydı ve yeni kip 14 piyasanın hepsinde sıfır işlem üretti;
  hata yok, yalnız boş sonuç. Zaman farkını `Timedelta` ile karşılaştır; bir kapının "hiçbir şey geçmedi" çıktısı ilk
  koşuda bir ÖLÇÜM değil bir alarmdır.
- **Olay sinyali için kurulan bir sınır, durum sinyalinde başka bir şeyi sınırlar (Tur-97, BULGU-97-8).** Sembol
  başına iki pozisyon sınırı, kırılım gibi seyrek olaylar için makuldü; her taramada yeniden ateşleyen bir trend
  durumunda aynı bahsin ikiye katlanmasına dönüştü ve backtest'in ölçtüğü şeyden (tek pozisyon) ayrıştı. Yeni bir
  sinyal TÜRÜ eklerken canlı hattı izole replay'le koştur: bu kusuru birim test değil, gerçek bir günün tekrarı buldu.
- **Test paketi operatörün yapılandırmasını izliyorsa, yapılandırma değişikliği test kırmızısı gibi görünür (Tur-97,
  BULGU-97-9).** `.env` aylarca kod varsayılanıyla aynı olduğu için bağımlılık görünmedi; canlı strateji değişince dört
  test düştü. Testler üretim DOSYALARINDAN yalıtıldığı gibi (NIHAI_LOG_DIR/DATA_DIR) üretim AYARLARINDAN da yalıtılır;
  karar yolunu seçen her env değişkeni conftest'te kod varsayılanına sabitlenir.
- **Bir sonuç dosyasının anahtarı, koşunun AYIRT EDİCİ boyutlarının tamamını taşımalıdır (Tur-98, BULGU-98-1).**
  Tur-97 düzeneği sonucu yalnız sembolle adlandırdı; 7 sembol hem ABD havuzunda hem canlı izleme listesindeydi ve iki
  koşu farklı kol taşıyordu. Son yazan kazandı: ABD grubunda event_breakout 7 sembolde iki kez sayıldı (2.246 ↔ gerçek
  1.309 işlem) ve izleme sonucunun doğru çıkması yalnız görev BİTİŞ SIRASININ şansıydı. Hiçbir hata çıkmaz; kusur ancak
  aynı evreni iki kez içeren bir karşılaştırmada görünür. Kural: dosya/önbellek anahtarı yazarken "aynı sembol iki
  bağlamda koşulabilir mi?" diye sor (Tur-80'in HMM önbellek anahtarı dersinin dosya tarafı).
- **Öngörü kaybındaki büyük üstünlük karar ölçeğinde küçük kalabilir — kapıyı kurmadan ÖNCE ayırt edebileceği en
  küçük farkı hesapla (Tur-98, BULGU-98-2).** σ_ML21 21 günlük öngörüde EWMA'yı canlı evrende DM t −8 ile yendi; aynı
  modelin motor Sharpe farkı z 0,8. Kararın mekanizması (girişte 3σ stop, risk boyutu, izleyen stop) öngörü hatasının
  çoğunu emer; iki σ kolunun getirileri ρ≈0,99 korelasyonludur ve ön-kayıtlı kapı bu evrende ancak ≈+0,10 Sharpe'ı
  görür (gözlenen +0,05). MDE hesaplanmadan kurulan kapı "kanıt yok" ile "kapı bu büyüklüğü göremez"i ayıramaz;
  hesaplanınca karar da değişir: ileri veriyle bile yıllar sürecek bir soru yerine birleşik (çok piyasalı) bir soru sorulur.
- **Bir boyutlama girdisinin DÜŞÜŞ iyileşmesi, girdinin DÜZEYİ değişiyorsa beceri değil maruziyettir — aynı düzeyi
  taşıyan aptal kolu koş (Tur-98, BULGU-98-3).** σ_ML21 MDD'yi 29 hücrenin 27'sinde küçülttü ve bu tek başına "risk
  azaltıyor" diye okunurdu; ML σ'sı medyanda EWMA'nın 1,091 katıydı (geniş stop → küçük pozisyon). EWMA'yı aynı katla
  genişleten kol 25/29 verdi. Zamanlama katkısı Sharpe'ta ölçülür (22/29) ve canlı evrende ayırt edilemez. Teşhis kolu
  sonuç görüldükten sonra eklendiği için KAPISIZ ve adıyla "sonradan" işaretlidir (Tur-23).
- **ML'i kuralın YERİNE değil ARDINA koymak sürüklenme tuzağını yapısal olarak kapatır, bilgi YARATMAZ (Tur-98,
  BULGU-98-4).** Meta kapının tabanı kuralın kendisidir, "hep uzun"a çökemez ve çökmedi (%14-24 eleme); yine de
  Sharpe'ı 29 hücrenin 15'inde artırdı = yazı-tura. Tur-97'nin lojistik (sürüklenme) ve sürüklenmesiz zamanlama (1/14)
  sonuçlarıyla birlikte ÜÇÜNCÜ bağımsız hüküm: günlük fiyat + rejim özellikleri trend girişini zamanlamıyor. Mimariyi
  düzeltmek doğru teşhisi ölçer, eksik bilgiyi üretmez (ACI DERS #5).
- **Bir evren için yapılan veri tazelemesi, o sembolleri paylaşan BAŞKA bir düzeneğin dondurulmuş girdisini sessizce
  değiştirir (Tur-98, BULGU-98-6).** İzleme listesinin indirmesi havuzun 7 sembolünü de yeniden çekti (yeni gün +
  düzeltilmiş geçmiş); sonraki her Tur-96/97 koşusu farklı girdiyle koştu ve havuzlanmış GBM başka ağaç kurdu (bir
  piyasanın QLIKE hükmü başabaşa döndü). Hiçbir hata yok, yalnız "aynı" komut artık aynı sayıyı üretmiyor. Kural: bir
  indirme adımı yalnız KENDİ evreninin eksik sembolünü çekmeli; paylaşılan sembol dondurulmuş sayılır.
- **Eşiğe yapışık bir oranın işareti ölçüm gürültüsüdür, piyasa özelliği değil (Tur-98, BULGU-98-7).** Tur-96 "Türkiye'de
  taban kuralı sürüklenmeyi ters oynar" dedi (Sharpe −0,80) ve bunu sağa çarpıklıkla açıkladı. Yükselen gün payı 0,503
  idi; 223 sıfır getirili tatil satırı onu eşiğin altına itiyordu, süzgeçle uzun payı 0,42 → 0,76 ve Sharpe +0,94.
  Bir kural "p > 0,5" gibi bir eşikten karar alıyorsa, girdinin eşiğe uzaklığı hükümle birlikte raporlanır; uzaklık
  gürültü mertebesindeyse açıklama (çarpıklık) kurulmaz.
- **Bir karar motorunun kanıtı hangi SORUYA aitse o soruya uygulanır; geri düşüş zinciri bunu sormuyorsa motor
  sessizce başka bir sorunun cevabını verir (Tur-99, BULGU-99-1).** `decide()`'ın önseli strateji → yan → GLOBAL diye
  düşüyordu ve öğrenme durumunda yalnız 1 dk tahmin servisinin 5 barlık yön isabeti vardı. Canlı strateji günlük trend
  takibine geçince aynı önsel her quant adayına ABSTAIN yazdı (23/23); enforce'ta hepsini veto ederdi. Hiçbir hata
  çıkmaz: motor kendi sorusu için doğru çalışır. Kural: bir modelin/karar motorunun kapsamı (hangi sinyal ailesi, hangi
  ufuk) çağıranın tarafında da denetlenir; yeni bir strateji eklerken sinyali TÜKETEN her karar katmanı `grep` ile
  bulunur (BULGU-97-4'ün meta kapı kararının AI köprüsündeki eşi).
- **İsabet kapısı bir işlem kuralının değil bir yön tahmincisinin ölçütüdür (Tur-99, BULGU-99-2).** quant_trend'in
  isabeti 30 hücrenin 29'unda %33-45, kâr faktörü çoğunda > 1: kenar kazanç/kayıp asimetrisindedir. "Alt sınır > 0,5"
  kapısı bu stratejiyi kendi kusursuz kanıtıyla bile hep veto ederdi. Bir işlemi onaylayan kapı beklenen değeri
  (işlem başı R'nin alt sınırı > 0) sorar; isabet yalnız kazanç ve kaybın simetrik olduğu soruda beklentinin yerine geçer.
- **Oynaklık yönetimi riski ancak taban sabit maruziyetliyse azaltır; taban zaten 1/σ̂ ile boyutlanmışsa kaldıraç
  ekler (Tur-99, BULGU-99-3).** Girişler büyük hareketlerden sonra, σ̂ yüksekken olur; pozisyon içinde σ̂ ortalamaya
  döndükçe VY adedi artırdı: maruziyet 29/29 hücrede yükseldi, MDD 27/29 hücrede büyüdü, Sharpe etkisi pencereye göre
  işaret değiştirdi. Literatürdeki sonucu bir mekanizmaya taşımadan önce o sonucun TABANINI sor (Moreira-Muir'in tabanı
  sabit ağırlıklı portföydür). Bir boyut kuralını değerlendirirken ortalama maruziyet oranı Sharpe'la birlikte basılır.
- **Tek evrenli kapı kanıtın çoğunu atar; çok piyasalı kanıt korelasyon düzeltilerek birleştirilir (Tur-99,
  BULGU-99-4).** Tur-98'in kapısı yalnız canlı evrenin z'sine bakıyordu (MDE ≈ 0,15 Sharpe) ve 15 piyasanın kanıtını
  yalnız "olumlu sayısı" olarak kullanıyordu. Korelasyon düzeltmeli Stouffer MDE'yi 0,065-0,072'ye indirdi; düzeltme
  zorunludur: VY hipotezinde 15 piyasa ortak oynaklık faktörü yüzünden yalnız 6,3-6,8 bağımsız piyasa değerindeydi.
  MDE koşudan ÖNCE, yalnız standart hata ve korelasyonla (z'ler toplanmadan) hesaplanır.
- **Bir ölçüm laboratuvarı, motoru yeniden yazmadan önce motorun KENDİ çıktısıyla birebir eşleşmelidir (Tur-99).**
  Boyut kararını sınamak için motora adet güncelleme eklemek yerine motorun işlem defteri yakalanıp getiri defterden
  yeniden kuruldu; ilk replika giriş gününü yanlış işaretliyordu (motor giriş barında MTM'yi dolum fiyatında bırakır)
  ve koşu sonu zorla kapanışını getiriye katıyordu. Parite 1,4e-2'den 3,7e-9'a ancak bu iki konvansiyon bulununca indi.
  Paritesi ölçülmemiş bir laboratuvarın "tek değişkenli" farkı, değişkenin değil replikanın farkıdır.
- **pandas 3'te `date_range`/`bdate_range` varsayılan olarak MİKROSANİYE üretir (Tur-99).** "Birimden bağımsız
  özet" testi indeksi "us"ye çevirip aynı özeti bekliyordu; indeks zaten "us" olduğu için dönüşüm boş işlemdi ve
  birim dönüşümünü silen mutasyon kaçtı. BULGU-97-1'in test tarafı: birim sınayan bir test iki birimi AÇIKÇA kurar.
- **Sızıntıyı kapatan onarım, varyansı büyük bir seçici getiriyorsa düzeltme değildir (Tur-100, BULGU-100-1/2).** Kilitli
  σ_ML21'in erken durdurması örtüşen hedefte rastgele doğrulamayla hiç tetiklenmiyordu; onarım (tek yıllık zamansal
  doğrulama) 15 piyasanın 1-3'ünde iyi çıktı, çünkü seçim kriz yılında 20 iterasyona savruldu. Kusuru onarmadan önce
  ZARARINI ölç (%100 veri + sabit 300 = ayırt edilemez) ve onarımı sabit-parametreli ikiziyle kıyasla; bir seçicinin
  değeri seçtiği seviyede değil seçiminin KARARLILIĞINDADIR.
- **Kaybı hizalamak teoride doğru, sayısal olarak dağılabilir (Tur-100, BULGU-100-3).** Gamma sapması QLIKE'ın ta
  kendisidir; HistGBR'de hessian y/μ küçükken Newton adımı büyüdü ve 5.879 öngörü taştı. 72 bin satırlık teşhis kümesinde
  görünmedi, 500 bin satırda çıktı: sayısal kararlılık küçük kümede sınanamaz.
- **Kesit-sabit bir özellik havuzlanmış ağaçta etkin örneklemi GÜN sayısına indirir (Tur-100, BULGU-100-4).** VIX ve
  piyasa ortalaması aynı gün her satırda aynıdır; 671 bin satırlık panel 2011'de 793 ayrık günle eğitiliyordu ve 200
  satırlık yaprak birkaç günle kurulabiliyordu → ezber, ceza (gün sayısı arttıkça eriyor). BULGU-53-1 / Tur-74'ün
  "bağımsızlık birimi gündür" dersi model EĞİTİMİNDE de geçerlidir.
- **Bir aday iki değişikliği birlikte taşıyorsa, biri kötüyse öbürü ölçülemez (Tur-100).** Merdiven yeni bilgiyi (D3)
  zamansal erken durdurmanın (D2) üstüne kurdu; D2 kötü çıkınca k3'ün kötülüğü ikisine birden yazılabiliyordu ve ayrım
  ancak SONRADAN bir teşhis koluyla yapıldı. Ön-kayıtta her bilgi değişikliği İYİ tabanın (kilitli tarifin) üstüne de konmalı.
- **"Zamanla daha iyi olur mu?" üç ayrı sorudur (Tur-100, BULGU-100-5).** Veri miktarı (evet, azalan getiri), bayatlama
  (güncellenmeyen modelin cezası yılda ≈0,001-0,002 birikir) ve yeniden eğitim sıklığı (yıllıktan sık = kazanç yok) ayrı
  kollarla ölçülür; biri ötekinin cevabı değildir.
- **Bir kararın kapısı, kararın uygulanacağı BİRİMDE kurulur (Tur-101, BULGU-101-2).** Tur-97 canlı evreni 30 BAĞIMSIZ hesap olarak
  ölçtü; canlı tek portföydür. Ön-kayıtlı Bölüm I de aynı birimi kullandı ve sinyal çıkışını 2018-26'da çürüttü; tek portföyde aynı
  kural dört pencerenin dördünde q0'ın her sırasından iyi. D5 teşhiste YAZILIYDI ama kapıya taşınmadı. Paylaşılan kapasite (tavan,
  sektör sınırı) çıkış kuralıyla etkileşir: kısa ömürlü pozisyon slotunu kaybeder. Sembol başı ölçüm yalnız bilgi kolu olabilir.
- **Bir strateji canlıya girerken çevresindeki TA-dönemi parametreleri o stratejinin ufkuna göre yeniden sorulur (Tur-101,
  BULGU-101-1/4).** Günde bir karar veren, çok günlük pozisyon tutan quant kipi 520 sembollük bir tarayıcı evrenine, girişten sayılan
  2 saatlik soğutmaya ve ~2× nominal tavana bağlandı; hiçbiri hata üretmedi. İzole replay açılan pozisyonların KİMLİĞİNİ gösterdi
  (alfabetik sıra, 6/9 ölçülmemiş geniş katman). Yeni bir sinyal türünde replay'in SAYISI değil açılan sembollerin LİSTESİ okunur.
- **Bir fren yalnız etkin olduğu yapılandırmada sınanır (Tur-101, BULGU-101-5).** Backtest günlük sayaçları hiç sıfırlamıyordu ve
  varsayılan seans tarihi bugündü; operatör iki freni kapattığı için yüzlerce ölçüm etkilenmedi, `.env.example` ile kurulan her ortamda
  backtest ilk 3 ardışık zararda kalıcı donuyordu. Bir canlı ↔ backtest parite iddiası frenler AÇIKKEN kurulur; "0 = kapalı" ayarıyla
  yeşil kalan test frenin kendisini sınamaz. Bir tarih alanının `date.today` varsayılanı, geçmiş veride "hiç yeni gün gelmedi" demektir.
- **Doğrulama penceresi ön-kayıtlı kararı çevirebilir — çevirdiğinde kural gevşetilmez (Tur-101, BULGU-101-7).** Portföy planı üç
  pencerede rahat geçti; eklenen bakılmamış 1990-2000'de MDD koşulu 0,13 puanla kaldı ve plan bağlanmadı. İnce kaçış "neredeyse geçti"
  diye okunmaz; yeniden sınama yeni bir ön-kayıt ister.
- **Mutasyonun "yakalandı"sı toplama hatasından geliyorsa yakalama değildir (Tur-101, süreç).** Desen satırın yalnız başını kapsadı,
  yorumun kalanı sözdizimini bozdu, pytest toplama hatasıyla kırmızı döndü ve koşucu bunu yakalama saydı. Desen tam satır olmalı,
  dosyanın satır sonu (bu depoda `backtest.py` CRLF, `main.py` LF) desene yazılmalı ve satır sonu Git Bash `grep`iyle değil baytla sayılmalı.
- **Bir stratejiyi değiştirmek, o stratejinin adaylarını TÜKETEN ölçüm döngülerini de değiştirir (Tur-102, BULGU-102-1).**
  Tur-97/99 meta kapıyı ve AI köprüsünü quant adayı için kapsam dışı yaptı; aynı değişikliğin 1m tahmin yüzeyinin aday akışını
  SIFIRLADIĞI sorulmadı. Meta/nöral/AI kanıtı ve canlı eğitim satırları bir gün boyunca hatasız durdu; ipucu GPU belleğiydi (282M
  model hiç yüklenmemişti). Kural: `ACTIVE_STRATEGY` değişince `generate_signal` çıktısını tüketen her yüzeyin (tahmin servisi,
  defterler) gün başına aday sayısı öncesi/sonrası ölçülür.
- **Bir mekanizmaya "yetki vermeden" önce enforce sözleşmesinin BOŞ ya da DUVAR olup olmadığını sor (Tur-102, BULGU-102-3/4).**
  Köprünün enforce'u terfi defteri boş olduğu için hiçbir şey yapmaz (kasası kontrolün kopyası), AI'ınki her adayı veto eder
  (kasası düz). İkisi de "ölçüldü" gibi görünen bir sonuç üretir ama bilgi taşımaz; kasa kurulmadan önce kuralın ateşlenme
  oranı ölçülür ve dejenere kural ya ön-kayıtlı bir kuralla değiştirilir ya da dejenere olduğu açıkça raporlanır.
- **Sabit eşikli bir kapı, çıktısı sıkışık bir modelde ya hep ya hiç çalışır (Tur-102, BULGU-102-2).** 282M nöral p 0,484-0,487:
  0,50 hiçbir adayı, taban oran (0,481) her adayı geçirir. Eşik ya modelin KENDİ dağılımına (nedensel kayan kuantil) bağlanır ya
  da kapının dejenere olduğu yazılır; ayrıca "ölçüm yoksa geçer" kuralı böyle bir kasada YALNIZ ölçülemeyen adaylarda işlem yaptırır.
- **Durum dosyası ile ilerleme işaretçisi aynı anlık görüntüde yazılır (Tur-102, BULGU-102-5).** İşaretçi (son işlenen bar)
  durumdan bağımsız ilerlerse yeniden başlatma, kaydedilmemiş güncellemelerin (izleyen stop) üstünden atlar ve hata vermez.
- **`python -m` ile çalıştırılan bir dosya İKİ modül kopyası doğurabilir (Tur-102, BULGU-102-6).** Giriş noktası `__main__`
  kopyasında süreç-tekil durum (motor, kilit) kurar; aynı dosyayı `import` eden bir kanca kanonik kopyayı görür ve o durum
  orada BOŞTUR. Hata yok: kanca kilidi "başka süreç" sanıp sessizce çekildi. Süreç-tekil durum taşıyan bir modülün CLI'ı işi
  kanonik modüle devreder (`from paket import modul; modul.kos()`); `runpy.run_module(..., run_name="__main__")` testi kilitler.
- **Bir boyut çarpanı, adedi kesen tavandan ÖNCE uygulanırsa tavan bağladığında sıfır etkilidir (Tur-102, BULGU-102-7).**
  Dar stop + yüzdesel risk kuralı hedef adedi özsermayenin onlarca katına çıkarır; tavan (alım gücü) her adayda bağlar, ×0,25
  ile ×1,0 aynı adede iner ve ilk aday bütün kapasiteyi alır. Bir boyut mekanizmasını ölçmeden önce "tavan kaç adayda
  bağlıyor?" sorulur; bağlıyorsa mekanizma ölçülemez, bir karşılaştırma da tek pozisyonun şansıdır.

- **Bir zaman damgasını SIRALAMADAN ÖNCE kırpmak, aynı birimdeki olayları kaynağın sırasına bırakır (Tur-103,
  BULGU-103-1).** Alpaca işlemleri `sort=desc` gelir; ms'ye kırpıp kararlı sıralamak aynı milisaniyedeki baskıları TERS
  sırada tuttu ve sıraya duyarlı tick kuralı işlemlerin %3,8-8,6'sına yanlış taraf yazdı — hata yok, yalnız delta %22 kaydı.
  Tam hassasiyette sırala, sonra kırp; testte aynı birimde TERS sırada gelen iki olay kur.
- **Paralel parçalı çekimde parça bütçesi eşit bölünürse, bitişiklik kuralı veriyi sessizce yarıya indirir (Tur-103,
  BULGU-103-2).** Eşit zamanlı 4 parçada en yoğun (en yeni) parça bütçeyi doldurunca eski üç parça atıldı: 139 bin ↔ 277
  bin işlem. Parçayı ZAMANA değil beklenen İŞ miktarına göre böl (ucuz bir sayım isteğiyle), tahmin şaşarsa bitişik en
  yeni kesiti koru ve kırpıldığını bayrakla söyle.
- **Bir fonksiyonun varsayılan argümanına bağlanan modül sabiti, sabiti yamalayan testi ETKİSİZ kılar (Tur-103,
  BULGU-103-4).** `def f(hedef=_SABIT)` tanım anında bağlanır; test `_SABIT`i yamaladı, yol tek parçaya düştü ve paralel
  dal hiç sınanmadı — iki mutasyon kaçtı. Üstelik "çağrı ≥ 8" iddiası sayfalamayla zaten sağlanıyordu. Ayar çağrı anında
  geçirilir; test MEKANİZMAYI ölçer (ayrık parça başlangıcı), başka bir mekanizmanın da şişirebileceği vekil sayıyı değil.
- **Anlık görüntü akışı ile olay akışı aynı pencereye konurken pencerenin sonu ikisinin EN GEÇİNE bağlanır (Tur-103,
  BULGU-103-3).** Isı penceresi son işlemde bitiyordu; defter görüntüsü işlemden sonra alındığı için ısı haritası boş
  çizildi ve bu yalnız tarayıcıda görüldü.
- **"TAHMİN" etiketi bir ölçümün yerine geçmez — gerçek etiketin bilindiği kardeş bir piyasada ölç (Tur-103).** Tick
  kuralı hissede doğrulanamaz; gerçek agresör tarafını bildiren Binance'te %97 (BTC/ETH) ↔ %84 (SOL) çıktı ve hatanın
  sıfır-tik işlemlerinde biriktiği görüldü. Hissede ölçülebilen tek şey başka bir tahminle UYUMDUR (%79-84) ve öyle adlandırılır.

- **Fail-soft bir uçta "reddedildi" çıktısı reddin NEDENİNİ kanıtlamaz (Tur-104, BULGU-104-4).** Doğrulama testi
  geçersiz girdiyi aşağı akış yamalanmadan gönderiyordu; test ortamının ağ guard'ı gerçek veri yolunu düşürüp
  `ok: false` döndürdüğü için doğrulama satırı silinse de test yeşil kalırdı. Bir reddi sınarken aşağı akış SAHTE BAŞARI
  döndürmeli ve hiç çağrılmadığı sayılmalı (Tur-74'ün "fail-soft yolu istisnayla problama" dersinin uç tarafı).
- **"Meşgulse dön" bayrağı kullanıcının SON isteğini düşürür (Tur-104, BULGU-104-1).** Asenkron yükleyicide
  `if (yukleniyor) return;` çift isteği önler ama yükleme sürerken verilen yeni komutu da siler; kusur yalnız
  sayfa açılışının kendi yüklemesiyle çakışan ilk tıklamada görünür. Meşgul bayrağı isteği BİRİKTİRİR (son isteneni
  hatırla, bitince bayat yanıtı at ve onu çek); aynı desenin kardeş istekleri (`grep` ile) aynı turda sorulur.
- **Bir süzgecin testi, süzgecin ELEDİĞİ bir girdiyle kurulmalı (Tur-104).** `en_az=400` testi rastgele 2000 işlemde
  yeşildi çünkü en büyük 150 emrin hepsi zaten 400'ün üstündeydi; `en_az` iletimini silen mutasyon KAÇTI. Kurgu, süzgeç
  açıkken ve kapalıyken FARKLI sonuç verdiğini ayrıca doğrulamalı (`hepsi["esik"] < 490 <= sec["esik"]`) — Tur-73
  dejenere-kurgu dersinin süzgeç hâli.

- **Bir duvarı kaldırmak mekanizmaya bilgi vermez; kaldırınca ne KALDIĞINI ölç (Tur-105, BULGU-105-1/2).** Gölge AI kasası
  ABSTAIN = veto ile 5.531 adayın hiçbirine girmedi; ABSTAIN geçince taraf verilmiş motor ters yön üretemediği için veto
  kümesi boşaldı ve kasa kontrolün ×0,25'lik kopyasına dönüştü. İki hâlde de "kontrolle fark" pozitif görünür (önce hep nakit,
  sonra dörtte bir maruziyet) ve ikisi de beceri değildir. Bir kuralı gevşetmeden önce gevşek hâlinin karar kümesini (kaç aday
  veto/geçer, hangi boyutla) defterden say; karşılaştırma tabanını o maruziyete göre kur (BULGU-98-3).
- **Bir hipotezi "çürüdü" saymadan önce düzeneğin kusuru ÜRETEBİLECEK koşulları taşıdığını sor (Tur-106, BULGU-106-1).**
  Keep-alive yarışı iki oturumda da "çürüdü" diye kaydedildi: sunucu ve istemci aynı süreçte, uç boş → 0/360. Canlıda hub ayrı
  süreçtir ve GIL tutan iş yapar; o koşullar kurulunca 10-12/1020 hata yeniden üretildi. Negatif sonuç veren bir deney ya
  pozitif bir kontrol (kusuru bilinçli üreten ayar) ya da canlıya benzerlik gerekçesi (süreç sınırı, yük, zamanlama) taşımalıdır.
- **Boş olmayan ama EKSİK bir cevap, boş cevaptan tehlikelidir: yedeği de atlatır (Tur-106, BULGU-106-5).** Akış sağlayıcısı
  tohum düşünce 8 günlük barı "veri" diye döndürdü; boş-sonuç kuralıyla kurulmuş iki failover katmanı da geçti ve tüketici günü
  önbelleğe aldı. İsteği karşılayamayan kaynak cevap vermemeli; önbellek yalnız TAM cevabı saklamalıdır.
- **Strateji değişikliğinin kör noktası: stratejiyi VARSAYILANDAN okuyan toplu işler (Tur-106, BULGU-106-4).** Tur-102 aday
  akışının durduğunu bulmuştu; aynı geçişte `backtest_symbol`ü strateji vermeden çağıran oto-tarayıcı 5 gün boyunca 516 boş
  backtest koştu ve yalnız bir "veri yok" uyarısı bıraktı. `ACTIVE_STRATEGY` değişince `grep "backtest_symbol("` ile
  `strategy_mode` geçmeyen her çağıranın o kipte işlem üretip üretemediği ölçülür.
- **Saat dilimsiz zaman `astimezone` içinde makinenin YEREL saatidir (Tur-106, BULGU-106-8).** Sistem sözleşmesi "saat
  dilimsiz = ET" iken takvimin altı fonksiyonu `dt.astimezone(ET)` yapıyordu; TR'de ET-naive 15:58 "08:58 ET, kapalı" okunur.
  Bir çağıranın elle `replace(tzinfo=ET)` koruması taşıması, PAYLAŞILAN fonksiyonun düzeltilmesi gerektiğinin işaretidir.
- **Beklenen sıfırı alarmdan ayıramayan bir uyarı her gün yanlış alarm verir (Tur-106, BULGU-106-6).** "Bugün +0 satır" 12:00
  koşusunda seans açılmadığı için hep doğruydu; kural ölçümün NE ZAMAN dolması gerektiğini (seans takvimi) sormalıdır.
- **"Canlı motorla aynı" diyen bir simülatör, canlının GİRİŞ kapılarını da taşımalıdır (Tur-106, BULGU-106-8/10).** Gölge
  kasalar boyut/çıkış/maliyette canlı motoru kullanıyordu ama girişte seans kapısını ve dolum RR kapısını atlıyordu; kasa
  zararının %91'i canlının asla girmeyeceği dolumlardandı. Bir yolun "aynı" olduğunu iddia etmeden önce canlı yolun
  karar zincirindeki HER `return` (red) noktasını say ve kopyada karşılığını ara.
- **Bir uyarının tekilleştirme anahtarı, kararın KENDİ barıdır (Tur-106, BULGU-106-11).** Saatlik barla anahtarlanan
  "bir kez uyar" kuralı, günlük karar veren quant'ta aynı durumu günde 8-9 kez "yeni sinyal" yaptı (BULGU-97-8'in uyarı
  tarafı: olay sinyali için kurulan sınır durum sinyalinde başka şeyi sınırlar).
- **İki kart aynı sayıyı farklı adla gösteriyorsa biri yanlıştır (Tur-106, BULGU-106-7).** "NLV" ve "Nakit Rezervi" aylarca
  aynı değeri bastı; `cash_balance` alımda hiç düşülmüyordu. Etiketin tanımını (NLV = nakit + pozisyon değeri) sayıyla sına.
- **Her gece yeniden hesaplanan bir ölçüm, girdisi kesilince ESKİ dilimi güncel gibi sunar (Tur-106, BULGU-106-13).** Yürüyen
  kalibrasyon 60 bin satırlık kuyruğu okuduğu için tahmin servisi aday üretmeyi bıraktıktan sonra da 22-24 Eylül'ü her gece
  yeniden ölçüp tarihsiz bastı; yanındaki 5 bin satırlık blok aynı kaynağa "örneklem yok" dedi. Bir ölçüm hükmü ölçtüğü
  DÖNEMLE birlikte yazılır (Tur-34'ün "ölçüm koşulu taşınmalıdır" kuralının zaman ekseni).
- **Yeni bir karar kodu eklemek, o kodu SAYAN her raporu da değiştirir — ve "sessizce yutulmaz" sözü yüzeyde de
  doğrulanmalıdır (Tur-106, BULGU-106-15).** Dört guard (Tur-89/97/101) karar hunisine eklenmedi; huni onları dürüstçe
  `taninmayan`a yazdı ama pano o alanı çizmediği için günde ~42 bin deneme operatörün gördüğü sayımdan düştü. Bir eylem kodu
  eklerken `grep` ile o günlüğü OKUYAN her ayrıştırıcıyı bul; ve bir API alanının "gösterilir" iddiasını yüzeyin çiziminde sına.
- **Günde bir kez önbelleğe alınan veri gün içinde DEĞİŞEBİLİYORSA, kararı önbelleğin dolduğu AN belirler (Tur-106,
  BULGU-106-17).** Canlı quant'ın günlük önbelleği "veri gün içinde değişmez" varsayımına dayanıyordu; sağlayıcı ise bugünün
  yarım barını ekliyordu. Sonuç: kararlar süreç başlatma saatine bağlıydı ve ölçülen kuraldan %5 ayrışıyordu. Bir önbelleğin
  anahtarı (gün) ile içeriğin değişme sıklığı aynı olmalı; değilse içerik o anahtarda SABİT olana (kapanmış barlar) daraltılır.
- **Bir çöküşü onaran bekçi, çöküşün kanıtını silmemelidir (Tur-106, BULGU-106-19).** Gözcü asılı hub'ı doğru yakalayıp
  yeniden başlattı; ama yeni süreç aynı stderr dosyasına yönlendirildiği için neden (uvicorn kapanışı, Traceback) o anda
  silindi ve geriye yalnız "port kapalı" satırı kaldı. Yeniden başlatan her yol, önceki sürecin çıktısını yeni süreçten ÖNCE
  saklamalıdır; aksi hâlde aynı arıza her seferinde kanıtsız tekrarlanır.
- **"Ölçüldü, reddedildi" hükmü ölçüldüğü KOŞULU taşır (Tur-106, BULGU-106-20).** Tur-81 "ufku dolmamış satırı atla"yı seans
  içinde ölçtü (0 tasarruf) ve reddetti; seans dışında aynı kural çekimlerin %100'ünü siler ve gözcü tam da o saatlerde
  yokluyordu. Bir alternatifi reddederken ölçümün koşulunu (saat, seans, yük) hükmün yanına yaz; koşul değişince hüküm
  yeniden sorulur.
- **Bir olayı SIRALAMAYLA önlemek, aynı olayın başka giriş yolunu açık bırakır (Tur-106, BULGU-106-24).** Tur-80 çift
  kayıtçıyı bayrağın kalkma sırasını değiştirerek kapattı; ders cümlesi "tekil bir kaynağa bağlanmalı" diyordu ama uygulanan
  çözüm tek yolu onardı. Belgelenmiş `run.ps1` aynı olayı proje açıkken yine üretiyordu. Kaynak tekilliği (kilit) bütün
  başlatıcıları birden kapatır; sıra düzeltmesi yalnız onarılan başlatıcıyı.
- **Bir etiketin tanımını değiştiren düzeltme, o etiketi BASAN her yüzeyi sayar (Tur-106, BULGU-106-25).** 106-7 "NLV"yi
  panoda düzeltti ve dersini yazdı; aynı turda Telegram, terminal ve uyarılar eski tanımı basmaya devam etti. Tanım tek bir
  özelliğe (`open_pnl`) indirilmeden her yüzey kendi toplamını yazar; `grep` etiketin kendisiyle ("NLV") yapılır.
- **Tek girdili bir önbelleği iki pencere sırayla soruyorsa her soru ötekinin verisini siler (Tur-106, BULGU-106-26).**
  Kusur yalnız BOŞ sonucun da önbelleğe yazıldığı durumda (kapalı seans) görünür ve hata vermez; yalnız her yenileme tam
  çekim olur. Bir önbellek testi "aynı soru iki kez" değil, yüzeyin GERÇEK soru dizisini (yoklama → çapa → yoklama) kurar.
- **Bir fonksiyona yeni bir girdi eklendiğinde, onu geçmeyen çağıran hata vermez, "bekle" der (Tur-106, BULGU-106-27).**
  Tur-97 `generate_signal`'a `df_daily` ekledi ve canlı taramayı bağladı; detay sayfası eski imzayla çağırmaya devam etti ve
  kart, canlının LONG tuttuğu sembolde günlerce `QUANT_GUNLUK_BAR_GEREKLI` gösterdi. Opsiyonel bir girdi eklerken
  `grep "<fonksiyon>("` ile HER çağıran sayılır; "verilmezse bekle" varsayılanı eksik bağlantıyı görünmez kılar.
- **Kırmızı tabanda koşan mutasyon her şeyi "yakalar" (Tur-106, süreç).** 106-28'in testi kendi hatasıyla kırmızıydı ve yedi
  mutasyonun yedisi "yakalandı" yazdı; sayı doğru görünüyordu, hiçbir şey ölçmüyordu. Koşucu artık her test kümesini önce
  mutasyonsuz koşar ve kırmızıysa durur. Bir yakalama oranı, tabanın yeşil olduğu gösterilmeden rapora yazılmaz.
- **Bir test dosyasının yeşili, testlerinin TEK BAŞINA yeşil olduğunu göstermez (Tur-106, süreç).** Kök `conftest`'in VIX
  yalıtımı yalnız modül önceden yüklüyse çalışır; `-k` ile tek koşulan test ağa çıkmayı denedi ve bunu dosya koşusu hiç
  göstermedi (önceki testler modülü yüklüyordu). Yeni testleri bir kez de tek tek, ayrı süreçte koştur.
- **Bir KİLİDİ paylaşmak, kilidin koruduğu NESNEYİ paylaşmak değildir (Tur-106, BULGU-106-30).** Pano ile ana döngü aynı
  `PORTFOLIO_LOCK`'u alıyordu ve bu, "aynı portföyü değiştiriyorlar" izlenimi veriyordu; pano ise diskten kendi kopyasını
  yükleyip ona yazıyordu. Yarış yoktu, hata yoktu, uç "kapatıldı" diyordu — ve bir tur sonra bellekteki sahip her şeyi geri
  yazıyordu. Bir durumu değiştiren her yol için "bu durumun SAHİBİ kim ve ben onun nesnesine mi yazıyorum?" sorulur; sahibi
  başka süreçteyse yazmak yerine reddetmek, başarı mesajıyla ezilecek bir yazım yapmaktan iyidir.
- **Değişen bir büyüklükten "başlangıç" değeri geri kurulamaz (Tur-106, BULGU-106-31).** Tur-89 yeniden başlatmada planlanan
  RR'ı ve riski `|giriş − stop|`'tan kurdu; stop sabitken doğruydu, izleyen stopla (quant, çok günlük) her yeniden başlatma
  riski küçültür ve R'yi ±1'e sabitler. Tur-97 aynı sınıfı mesafe için görüp `izleyen_mesafe`yi kalıcı yapmıştı (BULGU-97-3);
  defter tarafı eski kuralda kaldı. Bir alan kalıcı yapıldığında, aynı büyüklüğü BAŞKA yerde yeniden türeten kod aranır.
- **Bir listenin "en yeni N" kuralı, listenin NASIL birleştirildiğiyle birlikte sınanır (Tur-106, BULGU-106-32).** İki
  dosyanın kuyruğu doğru okunuyor, doğru süzülüyor, 200'de kesiliyordu; yalnız birleştirme sırası tersti ve panel ilk
  hâlinden (17 Haziran) beri seans içinde dünün kararlarını gösterdi. Tek dosyalı her test yeşil kalır. Çok kaynaklı bir "son N" ucunu sınarken iki kaynağı AYIRT
  EDİLEBİLİR damgalarla kur ve dönenin hangi kaynaktan geldiğini say.
- **Bir nesneyi "thread-safe değil" diye TÜMÜYLE esirgemek, güvenli yarısını da kapatır (Tur-106, BULGU-106-33).** Pano
  nesnesinin Live'a dokunan tek metodu `refresh`'ti; satır ekleyen metot düz bir listeydi. `dashboard=None` iki-katman
  kipinde sinyal satırlarını sessizce yuttu ve liste haftalarca yalnız kapanışları taşıdı. Bir nesneyi iş parçacığından
  saklamadan önce HANGİ metodunun güvensiz olduğunu sor; kalıcı bir satır da hangi GÜNE ait olduğunu taşımalıdır.
- **Bir işin ÖN KOŞULU duvar saatine gömülürse yılda iki kez kayar; koşulu işin başında TAKVİME sor (Tur-106,
  BULGU-106-36).** "23:30 = kapanıştan sonra" yazın doğruydu; ABD yaz saati bitince aynı tetik seansın son yarım saatine
  düşer ve hiçbir şey hata vermez. Tur-58 aynı dersi gölge koşucunun tetiği için yazmış, kardeş görev (gece
  değerlendirmesi) sabit saatte kalmıştı. Tetiği her mevsim yeniden kaydetmek yerine işin kendisi koşulu sınar (bekle ya da
  atla). Kardeşi: bir özetin "bugün"ü yerel takvim günüyse, gece yarısını aşan ya da henüz açılmamış bir seans için boş bir
  günü raporlar; sayımın birimi sorunun birimi (seans) olmalı ve hangi seansa ait olduğu çıktıyla taşınmalıdır.
- **Kaydettiğin şeyi, kaydedenin YAZDIĞI mesajdan değil kaydın KENDİSİNDEN doğrula (Tur-106, BULGU-106-37).** Betik
  "6.5 saat" yazdı, görev 6 saat kaydoldu: `New-TimeSpan -Hours` tam sayı alır ve PowerShell kesirli değeri hata vermeden
  yuvarlar (bankacı kuralı: 6,5 → 6, 3,5 → 4). Kusur Tur-58'den beri her kayıtta vardı ve ancak operatör kaydı yapıp
  tetik geri okunduğunda görüldü. Bir parametreyi başka birimde bekleyen bir komuta verirken tipini sor; sessiz dönüşüm
  yapan dilde kesirli girdiyi testte ayrıca kur.

- **"Eğitildi" demek gövdenin açık olduğunu göstermez — girdi duyarlılığını ölç (Tur-107, BULGU-107-1).** 282M iki tur
  boyunca "çıktısı sıkışık" ve "öğrenmedi" diye kayda geçti; kimse girdiyi oynatıp çıktıya bakmadı. Üç satırlık ölçüm
  (z=0 ↔ ±3σ; LayerScale γ ↔ başlangıç değeri; γ=0 müdahalesi) parametrelerin %98,8'inin çıktıya dokunmadığını gösterdi.
  Küçük başlatılan bir kapı katsayısı (γ₀=1e-4) ile adımı parametre ölçeğine göreli bir optimizatör (Adafactor) birlikte
  o kapıyı kapalı tutar. Yeni bir basamağı kaydetmeden önce girdi duyarlılığı kartın alanıdır; AUC ve kayıp bunu göstermez.
- **Backtest, canlının yazdığı değerin KİPİNDE yapılır (Tur-107, BULGU-107-2).** Canlı köprü MC-dropout ortalamasını yazar,
  araştırma `predict_proba` (dropout kapalı) kullanır; bu modelde ikisinin korelasyonu 0,19 ve eval çıktısı sabittir.
  Hızlı kipin canlı kipi temsil ettiği VARSAYILMAZ, birkaç yüz satırda ölçülür (`deep_mlp`te 0,9965, 282M'de 0,19).
- **Eğitim hattının sessizce yaptığı dönüşüm, servis hattında da yapılmalı (Tur-107, BULGU-107-3).** `neutralize_degenerate`
  iki sabit kolonu 0'a çekip manifestte yalnız μ=0/σ=1 izi bıraktı; canlı köprü aynı kolonları ham verdi. Dönüşüm bir
  veri yükleyicide yaşıyorsa modeli çağıran her yol onu yeniden yazmak zorundadır — dönüşüm artefakta yazılmalıdır.
- **Stratejiyi varsayılandan okuyan toplu iş, strateji değişince sessizce boş çıktı verir — üçüncü kez (Tur-107,
  BULGU-107-4).** BULGU-102-1 (aday akışı) ve 106-4 (oto-tarayıcı) aynı geçişin kardeşleriydi; Tur-94 kurucusu da 1.156
  pencereden 0 satır döndürdü. Geçmiş bir modelin aday popülasyonunu yeniden kuran her araç stratejiyi AÇIKÇA alır.
- **Bir yakalama sayacı, yakalamayı neyle tanıdığını da sınamalıdır (Tur-107, süreç).** Mutasyon koşucusu kırmızıyı çıktı
  metninden okuyordu; `-qq` özet satırını susturunca 15 gerçek yakalama "toplama hatası" sayıldı ve sonuç 5/20 göründü
  (Tur-106'daki "kırmızı tabanda her şey yakalanır"ın tersi). Sınıflama pytest ÇIKIŞ KODUNDAN yapılır (0 · 1 · diğer).
  Aynı turda bash `until` döngüsünün dönüş kodu (gövdenin son komutu) bir zinciri ilk adımda düşürdü: bekleme işlevi
  açıkça `return 0` der.
- **Yavaşlığı ortak kaynağa yazmadan önce yavaş olan birimin kimliğine bak (Tur-107, süreç).** İki indirici koşarken
  parça süresi 10 sn'den 60 sn'ye çıktı ve hız sınırına dayanıldığı sanılıp biri durduruldu; yavaş olan yalnız tek bir
  semboldü (V) ve hub'da 429 hiç görülmedi. Durdurmak zararsızdı ama teşhis yanlıştı.

- **Çok testin baktığı bir kaynağın ikinci dili KOPYA değil KATMAN olmalıdır (Tur-108).** Pano kaynağına ~40 test metin
  düzeyinde bakıyor; çevrilmiş bir ikinci HTML her pano değişikliğini iki dosyada yaptırır ve ilk unutulan düzenlemede iki
  dil ayrışırdı. Kaynak tek kalır, çeviri veri (sözlük) olur ve eksikliği bir araç sayar.
- **Kimlikle eşlenen bir tabloda tek atlama sonraki her satırı kaydırır ve hiçbir şey hata vermez (Tur-108, BULGU-108-2).**
  Çeviri `kimlik→metin` satırlarıyla yazıldı; dört satırlık bir atlama 454 çeviriyi bir önceki anahtara bağladı ve derleme
  temiz geçti. Kusuru gösteren BAĞIMSIZ bir sayaçtı: kapsama aracı çevrilmiş metinleri "eksik" bastı. Sıra numarasına
  dayanan her eşlemede rastgele örnek çiftleri göz önüne bas; iki uç da metin taşıyorsa anahtar metnin kendisi olsun.
- **Genel bir kural özgül bir karşılığın ÖNÜNDE denenirse onu ezer; ama özgüllük uzunlukla ölçülemez (Tur-108, BULGU-108-3/4).**
  `{0} gün` kalıbı içinde tam karşılığı olan bir listenin tamamına uydu; `: ` ayracı `, `'den önce denenince liste öğeleri
  etiket/değer diye bölündü. Kalıbın NEYİ yakalayabileceği (yalın parça ↔ bileşik metin) ve ayraçların iç içelik sırası
  kuralın parçasıdır; ikisi de gerçek ekran metniyle (hasat) sınanmadan görünmez.
- **Sözlükte olmayan düz yazıyı parça parça çevirmek, hiç çevirmemekten kötüdür (Tur-108, BULGU-108-5).** Arka ucun
  brifi `PF 0.80 (loss)`, `Note: bu ölçümler…` diye yarım çevrildi: okuyan iki dili birden çözmek zorunda kalır ve hangi
  kelimenin kaynağa ait olduğunu bilemez. Serbest metin kutusu ya tümüyle eşleşir ya dokunulmadan kalır.

Detaylı kayıt: `research/` içindeki test çıktıları, `docs/EDGE_FINDINGS.md` ve depo kökündeki
`OPTIMIZASYON_LOG.md` (tur-tur ölçüm + denklik kanıtları; tarihli tur raporları `docs/` altında).

## Çalışma ilkeleri

- **Look-ahead yasak.** Tüm özellikler/sinyaller yalnız karar barına kadarki veriden. `df.iloc[:i+1]`
  dilimi kuraldır; son bar bazen kasıtlı atılır (Madde 3).
- **Dürüst raporla.** Backtest "kâr" gösterdiğinde survivorship/PIT'i sorgula. Olumlu sonucu
  "kenar bulundu" diye sunmadan önce her iki bölme + PIT + (mümkünse) forward kontrolü iste.
- **Mevcut disipline saygı.** Bu kod overfitting'e karşı bilinçli yazılmış; "iyileştirme" diye
  bu korumaları gevşetme.
- Türkçe yorum/çıktı konvansiyonu korunur. `l` = low (OHLC), kasıtlı (ruff E741 ignore).
- **CLI çıktısı UTF-8 olmalı (Tur-24).** Windows'ta çıktı dosyaya/pipe'a yönlendirilince
  stdout cp1254'e düşer; `→ ⚠ Δ 📈` gibi karakterler orada YOK → süreç ÇÖKER (watchdog ve
  zamanlanmış görevler tam da böyle çalıştırır, yani hata yolu üretimdedir). Her yeni
  `__main__` giriş noktası `from nihai.console import ensure_utf8_stdout` + `__main__`
  bloğunun ilk satırında çağrı yapmalı; `tests/test_console_encoding.py` bunu 74 giriş
  noktası için kilitler.
- **Testler üretim dosyalarına dokunamaz (BULGU-8/9, 2026-07-02):** kök `conftest.py`
  `NIHAI_LOG_DIR`/`NIHAI_DATA_DIR` ile logs/ ve data/'yı geçici dizine yönlendirir;
  backtest `PositionManager(record_logs=False)` ile trade_log/trades.csv/decision_log'a
  yazmaz (bu dosyalar YALNIZ canlı). Bu korumaları gevşetme.
- **Ağ guard'ı İKİ taşımayı birden kapatır (BULGU-19 + BULGU-26, Tur-26).** Socket yaması
  yalnız `socket` kullanan istemcileri (requests/urllib3/httpx) yakalar. **yfinance 1.x
  taşımasını curl_cffi'ye taşıdı** → libcurl'ü cffi ile çağırır, `socket.socket.connect`
  hiç çalışmaz. Ölçüldü: guard etkinken `curl_cffi.requests.get(yahoo)` **429 döndü**
  (istek gerçekten gitti, defterde iz yok). Artık `curl_cffi.requests.Session.request`
  (+`AsyncSession`) da yamalı. Yeni bir HTTP istemcisi eklenirse guard'a da eklenmeli —
  `tests/test_net_guard.py` iki taşımayı da kilitler. Ayrıca `strategy._load_vix()`
  (10 yıllık canlı `yf.download("^VIX")`, modül-global → sıra-bağımlı) autouse fixture
  ile testlerde etkisizdir.
- **Opsiyonel bağımlılık TÜM kapıyı düşürmemeli (Tur-26).** `research/pairs_test.py`
  statsmodels'i katı import ediyordu ve `tests/test_pairs_math.py` onu import ettiği için
  statsmodels'siz ortamda pytest TOPLAMA fazında duruyor, **2178 testin tamamı**
  koşmuyordu. Opsiyonel bağımlılıkta desen: modül-düzeyi `try/except ImportError` +
  yönergeli yer tutucu (ya da testte `pytest.importorskip`).
- **Dejenere girdi SESSİZ yanlı çıktı üretebilir (Tur-28).** Tamamen DÜZ bir pencere (halt /
  illikit isim / seans-dışı bar) iki ayrı hatayı birden tetikliyordu: `rsi()` 0/0 belirsizliğinde
  100 ("azami aşırı-alım") döndürüyordu — doğrusu nötr 50 — ve `_bias_and_confidence`'ın `>`
  karşılaştırmaları eşitliği −1.0 (tam SHORT) sayıyordu. Sonuç: SIFIR bilgili girdiye
  **conf 0,60 ile SHORT**. İki hata birbirini kısmen maskeliyordu (sahte +1.0 RSI bileşeni
  |bias|'ı küçültüyordu), bu yüzden yalnız biri düzeltilince tablo KÖTÜLEŞMİŞ görünür.
  Şimdi: eşitlik → 0.0 (bilgi yok) → mevcut nötr dalı → `_bias_and_confidence(düz seri)`
  conf **0,00**. (Ölçüm o gün `_forecast` sarmalayıcısıyla alınmıştı; o sarmalayıcı
  Tur-43'te ÖLÜ KOD olarak silindi — dersin öznesi zaten `_bias_and_confidence`'tır ve
  pano onu doğrudan çağırır, yani iddia hâlâ koddan doğrulanabilir.)
  İkisi de ölçülen veride BİT-DENK (rsi14==100: canlı 6374 satırda 0 kez; eşitlik dışı 20.000
  rastgele girdide 0 fark). Ders: bir göstergeyi/skoru dejenere girdilerle (düz, tek-bar, NaN,
  sıfır) SINA — birim testler bunu yakalamıyordu.
- **`status()` YAZMAK yetmez, OKUNMASI gerekir (Tur-61, BULGU-61-2).** `ai_advisor.status()`
  (Tur-38) ve `shadow_bridge.status()` (Tur-60) docstring'lerinde "pano/günlük özet için"
  yazıyordu; ikisinin de tek çağıranı TESTTİ. Üçüncü kardeş `neural_shadow.status()`
  `/api/ai/overview`da zaten yayındaydı — yani desen vardı ve tam da CANLI KARAR yoluna
  bağlı iki köprü açıkta kalmıştı. Önemi `last_error`dadır: köprüler fail-soft'tur ve ilk
  hatayı BİR KEZ uyarı, gerisini debug olarak loglar; 7/24 koşan bir tarayıcının
  system.log'unda o satır görünmez. Yani "sessiz ölüm yok" koruması OKUNMADIĞI için
  kendisi sessizdi. Kural: bir durum yüzeyi yazan tur, onu okuyan yüzeyi de AYNI turda
  bağlamalı (`/api/system/health` → `bridges`), ve okuyucunun varlığı testle kilitlenmeli.
  Aynı denetimin iki kardeşi: kodun okuduğu her env `.env.example`te BEYANLI olmalı
  (23 ad belgesizdi — belgesiz düğme çevrilmeyen düğmedir) ve belgedeki her `python -m <modül>`
  çözümlenebilmeli (`CLAUDE.md` olmayan `nihai.main`'i belgeliyordu).
- **Yazılıp KOŞULMAYAN bir izleyici, izleme değildir (Tur-61, BULGU-61-7).** `deadman.py`
  Tur-59'da tam olarak *"servisler sessizce öldü"* hâlini yakalamak için yazıldı, test
  edildi ve belgelendi — ama onu çalıştıran bir sarmalayıcı ve bir görev HİÇ yazılmadı,
  yani modül üretimde bir kez bile koşmadı. Ölçüldü (2026-08-31): üç servis portu da
  KAPALI, `predictions_log` 35 saat / `live_features_log` 87 saat bayat, `history.csv`
  **37 gün** eski ve tek bir uyarı çıkmadı. Kök neden daha da acıydı:
  `NihaiAlgo-ForwardHarness-Daily` görevi 2026-07-25'te `0xC000013A` (oturum kapanışında
  öldürüldü) ile bitip **Disabled** kalmıştı ve o günün log'unda sistemin KENDİ uyarısı
  duruyordu (*"canlı özellik logu artmadı — kayıt servisi durmuş olabilir"*). Kural:
  bir izleyici eklerken KOŞUCUSUNU ve GÖREV KAYDINI aynı turda yaz (kayıt ADIMI
  operatörün, ama ARACI senin işin) — `tests/test_tur61_kopru_ve_belge.py` bunu kilitler.
- **Bir DEFTERİN boş olduğunu ancak onu OKUYAN bir yüzey söyleyebilir (Tur-61,
  BULGU-61-6).** `nihai/ai` yedi kalıcı defter yazıyordu ve hiçbirinin üretim okuyucusu
  yoktu. Yüzey eklenince ölçüm anında çıktı: **yedinin BEŞİ hiç yazılmamıştı** (yazan yol
  üretimde hiç koşmamış), kalan ikisi 19-20 gün bayattı. Tur-38'in kuralı burada DOSYA
  ölçeğinde ihlal ediliyordu. Yedi ayrı panel YAZILMADI — hepsinin sorusu aynıdır
  (*var mı · ne kadar · ne zaman · son kayıt*), o yüzden tek uç + tek kart + tek kayıt
  (`api_extra.AI_STORES`) ve kayıt `ast` ile koda karşı denetlenir.
- **Üç durumlu oy kuralı BEŞİNCİ kez (Tur-61).** `live_predict_service._htf_direction`
  `1 if e20 > e50 else -1` yazıyordu — eşitliği −1 sayıyor VE fonksiyonun kendi
  docstring'iyle ("belirsiz → 0") çelişiyordu. Kusur, dersin ilk uygulandığı dosyanın
  KENDİSİNDE, `_bias_and_confidence`ın 20 satır aşağısındaydı. Ölçüldü: düz HTF
  penceresinde `conf *= 0.65` → 0,7760'tan 0,5044'e; gerçek 1h verisinde tam eşitlik
  406/7.531.642 (%0,0054). Yerel `_yon` kopyası da `vote`a indirildi. `grep` kuralı
  (Tur-35) burada bir kez daha: düzelttiğin dosya genelde tek kopya değildir — bazen
  ikinci kopya AYNI dosyadadır.
- **DELİK aramak, KESİLMİŞ ucu bulmaz (Tur-63, BULGU-63-1).** Tur-52 `veri_kapsami`
  ön-koşulunu koydu: veri setinin kendi `[min,max]` aralığında BOŞ ay varsa hüküm yok.
  Kapı doğruydu, **baktığı yer eksikti** — hiç doğmamış bir yılın veri setinde İZİ YOKTUR;
  o yıl bir delik değil, aralığın DIŞIDIR. Kapı deliksiz 21 ay görüp PASS dedi, verdikt
  "5,5 yıl" diye okundu (opsiyon yüzeyi: istenen 2021-01, doğan **2024-10-14**). Ölçüm 7
  veri setine koşturuldu: üçünde istenen pencerenin **%13–69'u** hiç doğmamış, tek uyarı
  çıkmamıştı. Çare: istenen pencere veriden ÇIKARILAMAZ → **kurucu BEYAN eder**
  (`expected_start`/`expected_end` + `warmup_days`) ve kapı beyanı gerçekle kıyaslar.
  Beyanın ikinci faydası daha büyük çıktı: `seasonality`nin 2 yıllık kaybı bir arıza değil
  **ailenin yapısıydı** (pencere içinde 2 yıllık aynı-ay geçmişi ister → 2021-2022 hakkında
  hüküm ÜRETEMEZ) ve hiçbir yerde yazılı değildi. Kural: bir kapsama kapısı yazarken
  **"ne eksik" değil "ne İSTENMİŞTİ" diye sor**; kesilmiş veri `FAIL` ("kenar YOK") değil
  `PRECONDITION` ("HÜKÜM YOK") vermelidir — ölçülmemiş dönem çürütülmüş sayılamaz.
  Yan ders: **ısınma BİLEŞİKTİR** — bir rolling istatistiğin girdisi zaten rolling ise
  ısınmalar TOPLANIR (`babmacro`: 80+80 işlem günü; naif beyan 4 ay eksik gösterirdi).
- **Bir ölçüm, TAŞINMIYORSA yok sayılır (Tur-63, BULGU-63-2/63-4).** İki bağımsız yerde
  aynı biçimde: (a) kapsam kapısı YALNIZ FAIL ettiğinde `gates`e giriyordu → geçtiğinde
  ürettiği "bu hüküm hangi dönemi kapsıyor" ölçümü ne kurucunun çıktısına ne deftere
  ulaşıyordu (11 defter satırının **hiçbirinde** dönem yok); (b) `fred_data.get()`
  kırpılmayı ZATEN ölçüp `attrs["kirpildi"]`e yazıyordu ama `get_all()` Series→DataFrame
  geçişinde pandas attrs'ı DÜŞÜRÜYOR → bayrak hiçbir çağırana ulaşmıyor, `build_babmacro`
  aynı ölçümü **elle yeniden** yapıyordu. Kural: ölçümü ürettiğin yer değil, **okunacağı
  yer** onu taşıyor mu diye bak — taşımayan bayrak, ölçüm yokluğuyla aynı şeydir; ve
  taşınmayan bir ölçüm er geç ikinci kez, elle, başka bir yerde hesaplanır.
- **Bir performans önerisinin ÖNCÜLÜ bayatlar — koşulun HÂLÂ geçerli olduğunu ölç
  (Tur-64).** Dış rapor *"en kritik I/O adımı: projeyi OneDrive'dan çıkarın"* diyordu ve
  gerekçesi bu depoda TARİHSEL olarak doğruydu (`atomic_io.py` 2026-07-11'deki
  `predictions_log` çöküşü yüzünden yazıldı). Ölçüldü: **OneDrive süreci KOŞMUYOR**, otomatik
  başlatma kapalı (`StartupApproved` `0x03`), dosyalarda bulut yer tutucusu yok ve gerçek
  yazma deseniyle A/B'de `data/` ↔ OneDrive DIŞI SSD atomik p50 **1,74 ↔ 1,63 ms** = fark
  gürültüde. 2,7 GB'ı taşımanın ölçülmüş karşılığı YOKTU. Aynı turda ikinci öncül de düştü:
  *"tarama CPU-bound → `ProcessPoolExecutor`"* — canlı tarama CPU'su sembol başına
  **0,69 ms** (yanında iki ağ çağrısı), ve önerinin DOĞRU olduğu yerde (`screen.py`/
  `auto_screener.py`, ~7,5 sn hesap/sembol) process havuzu ZATEN vardı. Kural: bir
  optimizasyon önerisini uygulamadan önce **önerinin dayandığı koşulu bugün ölç** — geçmişte
  gerçek olan bir darboğaz, hâlâ orada olduğunun kanıtı değildir; ve bir öneriyi YANLIŞ
  katmana uygulamak (0,69 ms'lik işi Windows'ta spawn ile paralelleştirmek) düz bir
  regresyondur. Turun tek gerçek kazancı da formatta değil KOPYADAYDI: 230 MB'lık dolt
  önbelleğini ÜÇ çağıran birebir aynı döngüyle tarıyordu → `dolt_options.read_bulk` +
  parquet yan dosyası, **37,8–51,2×**, çıktı `assert_frame_equal` ile ÖZDEŞ.
- **SOĞUK ilk çağrı KARARLI DURUM DEĞİLDİR — ısındırmadan alınan profil yanlış darboğaz
  gösterir (Tur-65).** Panonun 55 GET ucu ısındırmasız ölçüldüğünde beş uç "yavaş" göründü:
  `/api/predictions/accuracy` 5007 ms · `/api/ai/capacity` 2899 ms · `/api/intel/macro`
  2857 ms. Aynı uçlar İKİNCİ çağrıda **10 / 7 / 7 ms**tir — ilk çağrı modül importunu,
  model yüklemesini ve sağlayıcı önbelleğinin doldurulmasını taşıyordu. Gerçek darboğaz
  bu üçünün hiçbiri değil, her çağrıda 2334,8 ms harcayan `/api/quant/regime`ti (panonun
  toplam maliyetinin **%75'i**). Isındırmasız ölçüme inanılsaydı emek üç ayrı hayalete
  giderdi. Kural: bir uzun-ömürlü süreçte p50 **ısındırma sonrası** ölçülür; ilk çağrının
  ölçtüğü şey işin kendisi değil, sürecin BAŞLATILMASIDIR. (Tur-36'nın *"ÖNCE ölçümü yük
  altında alınırsa optimizasyon şişer"* ve Tur-49'un *"iki tarafın yapılandırması aynı
  olmalı"* derslerinin üçüncü ekseni: orada ZAMAN ve ORTAM, burada SÜRECİN YAŞI kirletiyor.)
- **Bir hızlandırma önerisini uygulamadan önce onun BU DEPODAKİ payını ölç; "%10-20"
  bir ölçüm değil bir REKLAMDIR (Tur-65).** Dış katalog dört ayrı madde için somut kazanç
  vaat ediyordu; dördü de burada ölçüldü ve **hiçbiri tutmadı**: `torch.inference_mode()`
  (#50, vaat %10-20) → deep_mlp **1,05×**, quant_transformer **0,99×**; backtest'te GC'yi
  kapatmak (#190) → **1,028×** ve gen-2 toplama sayısı koşu boyunca yalnız 11; orjson /
  `ORJSONResponse` (#3/#67) → JSON serileştirme uç süresinin **%0,1–0,4'ü** (100 ms'lik bir
  uçta en fazla 0,2 ms kazanç, karşılığında YENİ BAĞIMLILIK). Öncüller bu depoda çürüktü
  çünkü Tur-43/45/49 sıcak yolu zaten numpy görünümlerine indirmişti — GC'nin toplayacağı
  milyonlarca küçük nesne artık yok. Kural: bir katalog maddesi "sıcak yolun %X'i" diyorsa
  o X'i ÖNCE ölç; ölçmeden uygulanan optimizasyon, kazanç değil yalnız DİFF üretir.
- **"Sıfır kod değişikliği" diye sunulan bir bayrak DOĞRULUĞU değiştirebilir (Tur-65).**
  Katalog `python -OO`yu *"bugün, 5 dk, sıfır kod değişikliği"* kutusuna koyuyor. Bu depoda
  `-O` **47 üretim `assert`ini sessizce kapatır** (ör. `neural.py:2139
  assert self._feat_mu is not None` — normalizasyon parametresi yoksa `predict_proba` hata
  vermez, YANLIŞ SAYI üretir) ve `-OO` ayrıca docstring'leri siler, oysa **51 test dosyası**
  `__doc__`/kaynak metni okur (formül köprüsü, ölü-kapı tarayıcısı, `ast` denetimleri) —
  yani sözleşmenin taşıyıcısı docstring'lerdir. Bir yorumlayıcı bayrağı "kod değişmiyor"
  diye zararsız sayılamaz: değişen şey kodun ÇALIŞMA ZAMANI SÖZLEŞMESİDİR.
- **Bir düzeltmeyi TEK çağırana uygulamak, düzeltme değil örnektir (Tur-65).** Turun ilk
  geçişi `httpx.Client()`in çağrı başına **224,66 ms**lik SSL/CA vergisini ölçüp yalnız
  `telegram_bot`a paylaşılan istemci koydu. `grep` ile bakılınca kardeş çağıranlar hem
  açıktı hem DAHA KÖTÜ durumdaydı: `nasdaq_calendar._http_fetch_date` bu vergiyi **DÖNGÜDE**
  ödüyordu (`fetch_next_earnings` gün gün ilerler → 14 günlük pencere ≈ 3,1 sn saf kurulum).
  Aynı ölçüm İKİNCİ bir karar da verdirdi: `requests.Session()` **0,02 ms**tir (SSL bağlamını
  bağlantıda tembel kurar), yani vergi httpx'e ÖZGÜDÜR ve 6 `requests` çağıranına DOKUNULMADI.
  Kural: bir performans kusurunu düzeltirken önce DESENİ ara (Tur-35), sonra her çağıranın
  aynı öncülü taşıyıp taşımadığını AYRICA ölç — desen aynı görünse de vergi aynı olmayabilir.
- **REDDEDİLEN bir madde bile DOĞRU YERİ gösteriyor olabilir — ilacı çürütmek öncülü
  çürütmek DEĞİLDİR (Tur-66).** Katalog #11 *"Windows `ProactorEventLoop`u yüksek eşzamanlı
  FastAPI'de darboğazdır → winloop kur, 2-3×"* diyordu. Tur-65 bunu ölçmeden "donanım/OS
  kümesi"ne atmıştı. Tur-66 ölçtü: **öncül DOĞRU** (`async def` + işsiz uçta 64 eşzamanlıda
  259,1 → 553,8 istek/sn = **2,14×**; Proactor eşzamanlılıkla %44'e çöküyor), ama **ilaç bu
  depoda YANLIŞ** — gerçek `hub_server` şeklinde (sync `def` uçlar, gerçek iş) winloop
  **0,92–0,99× yani DAHA YAVAŞ**. Asıl kazanç öncülü kovalayınca çıktı: panonun **23 ucu
  `async def`ti ve 23'ünde de `await` YOKTU** → FastAPI onları thread havuzuna göndermek
  yerine OLAY DÖNGÜSÜNDE koşturuyordu, dolayısıyla her bloklayan iş (ağ turu, disk,
  `PORTFOLIO_LOCK`) sıradaki TÜM istekleri donduruyordu. İlan `def` yapıldı: gerçek panoda
  **2,47×** (p50 277,7 → 111,7 ms), mekanizma yalıtılmışken **19,2×** (en kötü gecikme
  889,0 → 23,2 ms). Kural: bir katalog maddesini reddetmeden önce *"öncülü BU depoda nerede
  karşılığını buluyor?"* diye sor — reddedilen madde bir turdur, gözden kaçan bulgu kalıcıdır.
- **`async def` + sıfır `await` = uç, FastAPI'nin thread havuzunu ATLAR (Tur-66).** Bu
  depoda pahalıydı çünkü `main --serve-web` panoyu TARAYICIYLA AYNI SÜREÇTE kaldırır
  (`main.py:1048`; uvicorn daemon thread + `LIVE_SCAN_WORKERS=8` tarama thread'i) ve 23 ucun
  **14'ü** tarama thread'lerinin de tuttuğu `PORTFOLIO_LOCK`u alıyordu — kilidi olay döngüsü
  thread'inde beklemek, o ucu değil PANONUN TAMAMINI dondurur. Yeni uç yazarken: gövde
  `await` kullanmıyorsa ilan `def` olmalıdır. `websocket` handler'ları ve `await call_next`
  yapan middleware bu kuralın BİLİNÇLİ istisnasıdır (ikisi de testle ayrıca korunur).
- **Bir kütüphane KERNELDE kazanıp SİSTEMDE kaybedebilir — amortismanı ölç (Tur-67).**
  Numba bu deponun KENDİ vektörel NumPy'ına karşı adil sınavda gerçekten hızlıydı:
  `njit` **16,1–21,9×**, `prange` **62,4–167,3×**, üstelik çıktı bit-denk. Yine de
  reddedildi, çünkü görünmeyen maliyet ölçüldü: `import numba` **969,9 ms** + ilk çağrı
  (JIT) **1211,5 ms**, derlenmiş çağrı 0,133 ms → NumPy'ın 2,11 ms'ine karşı **613
  çağrıdan sonra** kara geçiyor; `/api/quant/mc` panoda elle açılan bir uçtur ve süreç
  yeniden başlamadan o sayıyı görmez. Üstelik 969,9 ms'lik import Tur-26'nın
  `import nihai.ai` 3854 → 11 ms dersini doğrudan geri alırdı. Kural: bir hızlandırıcının
  mikro-ölçümü değil, ÇAĞRI SAYISINA BÖLÜNMÜŞ toplam maliyeti karar verir.
- **Bir hızlanma `seed` sözünü bozuyorsa hızlanma değil, sessiz davranış değişikliğidir
  (Tur-67).** `monte_carlo_paths` ve `mc_bootstrap_terminal` vektörleştirildi (**2,55×** ve
  **69,2×**) ama BİT-DENKLİK ölçülerek korundu ve yolda iki tuzak çıktı: (1) `cumsum(pnl)`
  yapıp sermayeyi SONRA eklemek TOPLAMA SIRASINI değiştiriyor (maks |ΔMDD| **8,04e-14**) —
  sermaye diziye İLK ELEMAN konunca sıra oturdu; (2) `np.prod(..., axis=1)` eksende
  pairwise indirger, `np.multiply.accumulate(...)[:, -1]` ardışıktır. RNG akışına
  dokunulmadı: `shuffle`ın payı ölçüldü (%33), kalan %67 zaten vektörleşiyordu. Toplu
  `rng.integers(0,m,size=N)`in skaler akışla ÖZDEŞ olduğu da beş `m` değerinde ölçüldü ve
  testle kilitlendi. Kural: denklik VARSAYILMAZ, ızgarada ölçülür (Tur-36 dersinin devamı).
- **Bir katalog maddesinin ORANI doğru, PAYI sıfır olabilir (Tur-67).** `pandas
  rolling(14).mean()` **237,2 µs** ↔ ham `bn.move_mean` **12,2 µs** = **19,45×** ve fark
  GERÇEK (pandas 3.0 rolling'i bottleneck'e göndermiyor — Tur-65'in *"bottleneck kurulu →
  rolling hızlandı"* varsayımı yanlıştı). Ama uygulanmadı: 2000 barlık backtestte rolling
  **14 çağrı / 0,003 sn = %0,5**, çünkü `_SharedIndicators` göstergeleri zaten bir kez
  hesaplıyor. Aynı şekilde ciso8601 (*"50 kat"*) reddedildi: depo zaten
  `datetime.fromisoformat` **0,271 µs** kullanıyor, kataloğun eleştirdiği `pd.to_datetime`
  ise **578,2 µs**. Bir oranı doğrulamak, onu uygulamak için YETMEZ.
- **KÜMÜLATİF süre "buradan geçiliyor" der, "burada harcanıyor" DEMEZ — çağıranı ölç
  (Tur-68).** Tur-67 `indexing.py:__getitem__` kümülatifini (%39) görüp suçu
  `strategy.py`nin **50 `.iloc` bölgesine** yükledi ve *"generate_signal'ı (df, i)
  imzasına taşımak gerekir"* diye ayrı bir tur işi olarak yazdı. `print_callers` ile
  bakılınca tablo tersine döndü: `backtest_symbol`ün KENDİ tek satırı **1939 çağrı /
  0,087 sn**, `strategy.py`nin 50 bölgesinin TOPLAMI **~0,018 sn**. Planlanan refaktör,
  maliyetin BULUNMADIĞI yeri yeniden yazacaktı. Kural: bir profil satırını suçlamadan önce
  `print_callers` çalıştır — kümülatif sayı, çağrı zincirinin üstündeki herkese yazılır.
- **Pahalı nesneyi döngünün TEPESİNDE değil, KULLANILDIĞI dalda kur (Tur-68, katalog #34).**
  `backtest_symbol` her barda `df_slice = df.iloc[: i + 1]` kuruyordu ama dilim yalnız
  *"açık pozisyon yok"* dalında okunuyordu → 2000 barlık koşuda **1939 dilimin 1418'i (%73)
  kurulup ATILIYORDU**. `df.iloc[:n]` görünüm değildir: yeni `BlockManager` + indeks dilimi
  + `__finalize__`. Satır `if`in içine taşındı: **1,32×** (senaryoya göre 1,11–1,71×),
  `.iloc` çağrısı 1939 → 478, TAM `BacktestResult` **bit-denk**. Kazanç pozisyonun açık
  kaldığı süreyle orantılıdır. Bir backtest döngüsünde kod taşırken asıl risk hız değil
  LOOK-AHEAD'dir: dilimin ana çerçevenin ÖNEKİ olduğu (indeks + içerik) ve sınırın infaz
  barının gerisinde kaldığı ayrıca testle kilitlenir.
- **Bir KAPININ kendi testi, o kapıyı geçiren delil OLAMAZ (Tur-62, BULGU-62-3).** Tur-50
  bu depoya *"bir kural, onu çağıran yol yoksa kural DEĞİLDİR"* kapısını koydu ve dört ölü
  tanımı onunla yakaladı. Kapı doğruydu, **çağıran kümesi** yanlıştı: `_okunan_adlar()`
  `nihai` + `research` + **`tests`** tarıyordu, yani bir kuralın VAR olduğunu kanıtlamak
  için yazılan test, o kuralın ÇAĞRILDIĞININ da delili sayılıyordu. Kapı kendi kendini
  tatmin eder ve **ölü kapıyı arayan tarayıcı, ölü kapıyı GİZLEYEN şeye dönüşür.** İkinci
  körlük yanındaydı: yalnız SINIF METOTLARI taranıyor, modül düzeyi public fonksiyonlar
  hiç görülmüyordu. Üretim-yalnız kümeyle ölçüldü: `nihai/` genelinde çağıranı olmayan
  public tanım **5** (Tur-50 "onlarca meşru API" diye korkmuştu — varsayım ÇÜRÜDÜ), ikisi
  tam da PARA YOLUNDA. Aynı sıkı tarama `config.py`nin 125 alanında **0** verdi; negatif
  sonuç da ölçümdür. Kural: bir tarayıcı yazarken **DELİL KÜMESİNİ ayrıca gerekçelendir** —
  "nereye bakıyorum" kadar "neyi kanıt sayıyorum" da kapının tanımıdır; ve izin listesi
  varsa her girdi NEDENİNİ taşımalı (`_URETIMSIZ_IZIN`, test ediliyor).
- **Bir bayrağı AÇMAK, kuralı BAĞLAMAK değildir (Tur-62, BULGU-62-1).** Operatör
  `PDT_ENABLED`i açtı; sayaç işliyor, pano gösteriyor, terminal yazıyordu — ama
  `pdt_blocks_day_trade()` hiçbir üretim yolundan çağrılmıyordu, yani Madde 10/23
  hiçbir şeyi engellemiyordu (README onu uygulama olarak gösteriyor olmasına rağmen).
  Frenler ayrıca `size_stock` ve `size_long_option` gövdelerinde KOPYAYDI ve yeni fren
  ikisine de eklenmemişti → Tur-28'in dersi yapıya çevrildi: tek `_entry_blocked()`
  demeti, ve testi iki gövdenin de freni DOĞRUDAN çağırmasını YASAKLIYOR. Bağlamanın
  etkisi ÖLÇÜLDÜ ve iki yönlü okundu: canlı hesapta **bit-denk** (NLV 294.423$, eşiğin
  11,8 katı), küçük hesapta **belirleyici** (544 kapanışın %82,5'i aynı-gün; 14 işlem
  gününün **13'ünde** günlük 3 sınırı aşılmış). Kural: bir kuralı bağlarken etkisini
  BUGÜNKÜ kurulumda ÖLÇ **ve** kuralın ısırdığı kurulumu da yaz — "atıl" ile "zararsız"
  aynı şey değildir; `.env.example` `INITIAL_EQUITY=25000` ile geliyor.
- **Bir ders MODÜLE değil DESENE uygulanır — ÜÇÜNCÜ ve DÖRDÜNCÜ tekrar (Tur-35).** Aşağıdaki
  Tur-33 dersi bir kez daha, iki ayrı yerde doğrulandı:
  (a) **AUC beraberlik döngüsü.** Tur-10 tam bu saf-Python döngüsünü `research/train_meta._auc`
  ve `research/forward_harness._auc`'ta cumsum ile vektörleştirmişti. `nihai/ai/neural.py`
  Tur-16'da yazılırken kusur YENİDEN DOĞDU ve `evaluation_metrics`in **%85'ini** yiyordu
  (n=200k'da 319/377 ms). Vektörleştirme BİT-DENK: grup [i..j] için `(i+j)/2+1 ≡ i+(c+1)/2`;
  grup sınırı komşu karşılaştırmasıyla bulunur, **NaN != NaN** olduğundan her NaN özgün
  döngüdeki gibi kendi grubunu kurar. 30 senaryoda bit-fark 0; 3,7–12,1× hızlanma.
  (b) **Kuyruk okuyucu.** `_tail_lines` ÜÇ modülde birebir kopyaydı (`ai/learning`,
  `ai/calibration_loop`, `ai/reasoner`) ve üçünde de KUADRATİKTİ: her 8 KB adımda biriken
  tamponun TAMAMI yeniden `count(b"\n")` ediliyor, `f.read(step)+data` ile baştan
  kopyalanıyordu → 8,4 MB canlı logda **4,3 GB tarama, 2,10 s** (`update_from_predictions`
  süresinin %67'si). **Doğru desen zaten `reasoner._count_data_lines`'ta, 20 SATIR
  YUKARIDAYDI.** Üçü `nihai/atomic_io.tail_lines`e indi (9,7–86×; üretim yolu 5,8×).
  (c) **`pd.concat(...).max(axis=1)`.** Tur-6 bunu `strategy._true_range`'ten çıkarmıştı
  (kodda hâlâ *"backtest sıcak yolunda ~%40 CPU yiyordu"* yorumu duruyor) ama desen gösterge
  modüllerinde **dört yerde** yaşıyordu — üstelik Tur-33/34'ün optimize ettiği AYNI
  dosyalarda. `ultimate_oscillator` ayrıca aynı ifadeyi İKİ KEZ hesaplıyordu. `np.fmax/fmin`
  NaN'ı yok sayar = pandas `skipna` → bit-denk; 300 rastgele df'de (normal/DÜZ/NaN'lı) fark
  **0,000e+00**. `williams_ad` **13,5×**, `swing_index` **4,98×**, `ultimate_oscillator`
  **1,79×**, panel bütünü 85,4 → **70,0 ms**.
  Kural: bir kusuru düzeltirken `grep` ile AYNI DESENİ tüm repoda ara — düzelttiğin dosya
  genelde tek kopya değildir (bu turda üç ayrı desen, sırasıyla 1/3/4 kopya).
- **Biriktirirken yeniden sayma; sonda bir kez birleştir (Tur-35).** Yukarıdaki (b)'nin
  genel hâli: `data = yeni + data` ve ardından `data.count(...)` deseni O(k²) bayt taratır.
  Parçaları listede tut, YALNIZ yeni parçayı say, sonda `b"".join`. Aynı sınıf hata Python'da
  string/list birikiminde de doğar (`s += x` döngüsü).
- **`lst[-0:]` TÜM listedir — "sıfır" sanılan sınır sessizce her şeyi döndürür (Tur-35).**
  Eski `_tail_lines(path, 0)` okuduğu bloğun tamamını veriyordu ve satır sayısı BLOK BOYUTUNA
  bağlıydı (8 KB'de 70, 256 KB'de 1693) = tanımsız davranış; `AI_REASONER_PRED_TAIL=0` ile
  tetiklenebilirdi. Negatif/sıfır sınırı AÇIKÇA ele al.
- **Ölçüm bütçesini yalnız YAZAN değil OKUYAN da öder (Tur-35).** Bu turda eklenen 772M'lik
  basamak `param_ladder_report`'u zehirliyordu: pano her Beyin sekmesinde merdiveni istiyor ve
  ULTRA'yı gerçekten kurmak **3,1 GB + 5,2 sn** demekti. Çözüm `torch.device("meta")`:
  parametreler ŞEKLE sahip, DEPOLAMA yok; sayı depolamadan bağımsız olduğu için altı basamakta
  da birebir aynı (7,58 s / 4,5 GB → 3,02 s / ~0 GB). Yeni bir kapasite basamağı eklerken
  "bu sayıyı kim, ne sıklıkta soruyor?" sorusu mimarinin parçasıdır.
- **Ön-eğitilmiş gövde EĞİTİLMİŞ model değildir (Tur-35).** `pretrain` `_model`'i doldurur ama
  `is_fitted`'i AÇMAZ; `predict_proba`/`param_count` RuntimeError verir. Etiket görmemiş bir
  gövdeden `p_win` okumak, ölçüm yokluğunu ölçüm gibi sunmanın (Tur-31 IV Rank dersi) sinir-ağı
  karşılığı olurdu. Aynı disiplin: rekonstrüksiyon başlığı artefakta SIZMAZ, "ön-eğitildi mi?"
  sorusu `meta()["pretrain"]`'den yanıtlanır.
- **Kısıtı kayba değil MİMARİYE yaz (Tur-35).** Pinball kaybı her quantile'ı AYRI ölçer, çapraz
  geçişi (q₀.₉ < q₀.₁) GÖRMEZ. Kümülatif softplus kurulumu çaprazlamayı cezalandırmaz, imkânsız
  kılar. **İDDİANIN SINIRI ölçüldü:** garanti AZALMAYAN'dır, "kesin artan" DEĞİL — softplus
  float32'de 0.0'a alt-taşınca komşu quantile'lar eşitlenebilir (sıra bozulmaz). Test tam
  kurulabilen iddiayı kilitler (Tur-34: iddiayı algoritma belirler, umut değil).
- **Bir ders MODÜLE değil DESENE uygulanır (Tur-33, 2026-08-15).** Yukarıdaki Tur-28 dersi
  yalnız `_bias_and_confidence`'ta uygulanmıştı; AYNI desen (`1 if x > y else -1`, eşitliği
  ve NaN'ı −1 sayar) iki komşu modülde **51 kez** yaşamaya devam etti. Ölçüldü: tamamen DÜZ
  bir pencerede `indicators.indicator_panel` SIFIR bilgiden **"GÜÇLÜ SAT", skor −0,542
  (46 ayı ↔ 11 boğa)** üretiyordu — pano `/api/analyze`'da operatöre bunu gösteriyordu.
  Beş kök neden birbirini besliyordu: (1) eşitlik/NaN → −1, (2) `mfi` 0/0 → 100 (`rsi`
  kusurunun kardeşi), (3) `spearman` tam-bağlı pencerede +50 uyduruyor, (4) `supertrend`
  ATR=0'da −1 + ısınmayı 1.0 bırakıyor, (5) `%B` span=0'da 0,5 uyduruyor. Kural artık TEK
  KAYNAKTA: **`strategy.vote(x, y)`** (üç durumlu; eşit/NaN → 0) + `strategy.note3` (not oyla
  tutarlı). Normal veride BİT-DENK (`bull=39/bear=26/skor +0,200` değişmedi); 300 rastgele
  df · 24.900 panel satırında **0 oy farkı**. `tests/test_degenerate_panel.py` kilitler.
  Yeni bir yön oyu yazarken `vote()` kullan — ikinci bir kopya aynı hatayı yeniden doğurur.
- **Bir ORAN, paydası okunmadan okunamaz (Tur-33).** `vote` eşitliği nötr sayınca yönlü oy
  sayısı meşru olarak düşebilir hâle geldi ve 2 oydan 0 elde edilen skor **+1,000 = "GÜÇLÜ
  AL"** olurdu. `indicator_panel` artık `directional` (yönlü oy sayısı) döndürür ve 10'un
  altında etiket **"YETERSİZ ÖLÇÜM"**tür. Tur-23'ün "kapı PASS dese bile `n_kept` okunmalı"
  kuralının panodaki karşılığı.
- **Kırpma (`max(x, 0)`) bir ölçüm değil KARARDIR (Tur-33, BULGU-33-4).**
  `microstructure.corwin_schultz_spread` tahmincisi GÜNLÜK bar içindir; dakikalık seride
  negatif tahmin oranı **%61,7** ölçüldü ve sıfır-kırpılmış ortalama SİMETRİK gürültüden bile
  pozitif çıkıyor: gerçek makası SIFIR olan süreçte `spread_raw = −0,000607` iken kırpılmış
  değer **+0,000140** = "1,4 bps makas". Artık sözleşme `roll_spread` ile aynı: yansız tahmin
  pozitif değilse `spread` **None** + `reliable=False`; kırpılmış değer `spread_clipped`'ta
  durur, `negative_share` alan uyumunun teşhisidir. (Tur-32 açık kalem #6 kapandı.)
- **Ara dosya adı YAZANA özel olmalı (Tur-28).** `atomic_io` eskiden sabit `<dosya>.tmp`
  kullanıyordu; aynı hedefe yazan iki YAZAN (canlı kayıtçı ↔ `forward_harness`, watchdog
  yeniden başlatırken üst üste binen iki süreç, tek süreçte iki thread) AYNI ara dosyayı
  paylaşıyordu → `A yaz → B KIRP+yaz → A replace` sırasında hedefe YARIM içerik ATOMİK olarak
  yayınlanır (atomik ama BOZUK; kapsam: `state.json`, `predictions_log.csv`, öğrenme durumu,
  model kartları). Artık `atomic_io.tmp_path(dst)` pid+thread ile ayırır ve kalıcı hatada tmp
  silinir. Yeni bir atomik yazım eklenirse ara yolu ELLE kurma, `tmp_path()` kullan.
- **Uzun-ömürlü süreçte diskten okunan şey TAZE okunmalı (Tur-28).** `forecast_calibration.json`
  hem `PredictionService.__init__`'te hem web panosunun modül-global cache'inde BİR KEZ
  okunuyordu; `calibrate_forecast.py` eğriyi tazelediğinde 7/24 süreçler bunu asla görmüyordu.
  `_load_calibration` artık dosya imzasıyla (mtime_ns, size) kendi cache'ini tutar — çağırmak
  hem taze hem ucuz; tahmin servisi döngü BAŞINDA tazeler (tur içinde sabit kalır ki aynı
  turdaki 30 sembol aynı eğriyi kullansın).
- **`import nihai.ai` UCUZ kalmalı (Tur-26).** Tüm alt modüller PEP 562 ile LAZY;
  `__init__`'e eager import EKLEME. Ölçüm: eager hâlde 3854 ms (decision→quant_math→
  sklearn+scipy), lazy hâlde **11 ms**. `tests/test_import_hygiene.py` kilitler.
- **`rolling(p).apply(python_fn, raw=True)` gösterge hattında YASAK sayılmalı (Tur-33).**
  Pencere başına yorumlayıcıya dönüş O(n) Python geri-çağırımı demektir. Profilde
  `indicator_panel` süresinin **%25'i** tek bir göstergedeydi (`spearman`, pencere başına
  bir `pd.Series` kurup `.rank()` çağırıyordu). `sliding_window_view` ile vektörleştirildi:
  `spearman` **100×** (BİREBİR), `_linreg` **343×**, `wma` **98×**, `aroon` (argmax → tam
  sayı, yapısı gereği birebir). Panel bütünü 300 barda 1,79× · 3000 barda **4,66×** —
  kazanç seri uzunluğuyla BÜYÜR. **Denklik iddiası ölçülmelidir:** toplama sırasını
  değiştiren vektörleştirme bit-denk OLAMAZ (bağıl ≤ 7,5e-16). Doğru cevap vazgeçmek değil,
  iddiayı karara taşımaktır → 300 rastgele df · 24.900 satırda **0 oy / 0 konsensüs farkı**.
- **Sıkı bir döngüde çağrılan her GENEL-AMAÇLI yardımcı şüphelidir (Tur-34).** Tur-33 dersi
  (`spearman`'ın pencere başına `pd.Series` kurması) MODÜLE değil DESENE aitti ve yeni bir
  yerde aynen çıktı: `scipy.special.logsumexp` **(2,2) boyutlu** bir dizide çağrı başına
  **151,3 µs** harcıyor (array-API gönderimi + dtype yükseltme); aynı aritmetik elle
  yazılınca **11,2 µs**. `gaussian_hmm_fit` T=2000'de bu fonksiyonu 86.021 kez çağırıyordu.
  Rabiner (1989 §V.A) ölçekleme şemasına geçilerek döngüden `max`/`exp` tamamen çıkarıldı:
  `_forward` **18,9×**, `_backward` **18,8×**, `hmm_filter` **35,9×**. Sarmalayıcının
  maliyeti, sardığı matematiğin 13 katı olabilir — profil almadan tahmin etme.
- **Özyineleme "vektörleştirilemez" demek "optimize edilemez" demek DEĞİLDİR (Tur-34).**
  `supertrend` ve `kama` sıralı kalmak zorundaydı (her adım bir öncekine bağlı). Gövdedeki
  skaler `np.isnan(x)` IEEE denkliği `x != x` ile değiştirilip tekrar indeksleme yerel
  değişkene alınınca **4,93×** ve **2,31×** kazanıldı — tek bir aritmetik işlem değişmeden,
  300 rastgele df'de fark **0,000e+00**.
- **Bir ÇARPIM birikimi `cumprod`dur ve BİT-DENKTİR; bir TOPLAM birikimi değildir (Tur-34).**
  `nvi` bar başına `pct.iloc[i]`/`vol_dn.iloc[i]` çağırıyordu ve panelin en pahalı tek
  göstergesiydi; `np.cumprod` soldan sağa AYNI sırada çarptığı için **26,0×** hızlandı ve
  fark TAM SIFIR çıktı. Tur-33'ün `wma`/`_linreg` durumunda toplama sırası değişiyordu ve
  iddia "operasyonel denk"e çekilmişti. **Hangi iddianın kurulabileceğini algoritma
  belirler, umut değil.** `indicator_panel` bütünü 1500 barda 139,7 → **85,1 ms (1,64×)**.
- **Ölçüm KOŞULU, ölçümün kendisi kadar taşınmalıdır (Tur-34).** Bu turda eklenen beş bayrak
  aynı işi yapar: `causal=False` (look-ahead), `proxy=True` (vekil sayım), `reliable=False`
  (ki-kare geçersiz → p döndürülmez), `measured=False` + `reason` (karşılaştırılacak hücre
  yok), `decisive=False` (ΔAIC/ΔBIC < 2). Hepsi sayıyı okuyanın onu NE SANABİLECEĞİNİ
  engeller — Tur-31 IV Rank dersinin (ölçüm yokluğu ölçüm gibi sunulamaz) genellemesi.
- **Bir AİLE/DURUM SAYISI seçimi bir MODEL KARARIDIR (Tur-34).** "Düşüş kuyruğu daha kalın,
  o hâlde Clayton" demek varsayımı veriye SORMADAN dayatmaktır; "iki rejim vardır" demek de
  öyle. `copula_select` (AIC) ve `hmm_select_states` (BIC) sorar ve Δ < 2'de dürüstçe
  "ayırt edemedim" der.
- **Bir BELİRSİZLİK ölçüsünün YÖNÜ varsayılamaz — ölçülmelidir (Tur-36).** Sistem üç turdur
  MC-dropout std'sini (`neural_pwin_std`) "belirsizlik" diye loglıyordu. Canlı 1m setinde,
  kronolojik bölmede, 5 seed'de ölçüldü: **düşük-std çeyreğinin isabeti yüksek-std
  çeyreğinden DÜŞÜK (0/5 seed doğru yön, ort. −0,069)** — yani ölçü TERS sıralıyor. Teşhis:
  `spearman(sd, |p−0,5|) = +0,504` — std, modelin EMİN olduğu yerde büyüyor (sigmoid'in
  eğimi p=0,5'te en yüksektir; std logit ölçeğinin artefaktını taşıyor). Aynı testte
  Dirichlet kanıt boşluğu `u` **5/5 seed'de DOĞRU** sıralıyor (+0,098) ve **144× ucuz**.
  Kural: "belirsizlik" etiketli her sayı için *"yüksek olduğu yerde model gerçekten daha mı
  çok yanılıyor?"* testi koşulmalıdır; korelasyon değil DİLİM İSABETİ bakılmalıdır.
- **Bir yeteneği basamağa GÖMME, OPT-IN yap (Tur-36).** Tur-35 `vol_q`'yu ULTRA'ya gömdü ve
  açık kalem olarak *"alt basamaklara taşımak state_dict şemasını bozar"* diye yazdı. Doğru
  çözüm başlığı taşımak değil, **varsayılanı boş bırakmaktı**: `extra_heads=()` hiçbir
  `nn.Module` kurmaz → `torch.manual_seed` akışından TEK rastgele sayı bile tüketilmez →
  ağırlıklar/parametre sayısı/`state_dict` anahtarları bit-denk kalır; isteyen çağıran aynı
  yeteneği HER basamakta açar. **Modül kurulum SIRASI bir uygulama detayı DEĞİLDİR** —
  RNG tüketimini o belirler; ek başlıklar bu yüzden KANONİK sırada kurulur ve çağıranın
  verdiği sıra ağırlıkları değiştirmez.
- **Bir yöntemi denemeden önce YAKITININ VAR OLDUĞUNU ölç (Tur-36).** Tur-35 öz-denetimli
  ön-eğitimi yazdı ve *"asıl kazanç `live_features_log.csv`'nin etiketsiz ÇOĞUNLUĞUNDA"*
  dedi. Ölçüldü: 7.708 özellik satırının 7.317'si zaten çözülmüş; etiketsiz kalan **391
  (+%5,3)** ve bunların NEDENSEL olarak kullanılabilen kısmı **69 (+%1,5)** — çünkü
  çözülmemiş satırlar tanımı gereği eğitim penceresinin GELECEĞİNDEDİR (etiketsiz veri de
  look-ahead yapar: transdüktif sızıntı). Kök neden yöntem değil **loglama politikası**:
  `prediction_tracker` özellik vektörünü YALNIZ aday sinyal barlarında yazıyor. Bir sonraki
  turda "ön-eğitimi iyileştirelim" demek yanlış eksende çalışmak olurdu.
- **Sıkı döngüdeki genel-amaçlı sarmalayıcı — ÜÇÜNCÜ tekrar (Tur-36).** Tur-34 dersi
  (`scipy.special.logsumexp` sardığı matematiğin 13 katı) yeni bir yerde aynen çıktı:
  `scipy.stats.norm.cdf` TEK skaler için **39,3 µs** harcıyor (array-API gönderimi,
  argsreduce, dtype yükseltme); stdlib `statistics.NormalDist().cdf` **0,23 µs = 171×**.
  `ppf` **509×**. `bs_price` çağrı başına iki `cdf` çağırıyor ve `implied_vol` onu Brent
  döngüsünde ~10 kez çağırıyordu → **864 µs → 29 µs (29×)**, `bs_price` **53×**. Çözüm yeni
  bağımlılık değil merdivenin STDLIB basamağıydı. DİZİ girdilerinde scipy KALIR (orada
  vektörel ve hızlıdır) — değişen yalnız SKALER çağrılardır.
- **Bir denklik iddiası TEK NOKTADA ölçülmez (Tur-36).** Yukarıdaki geçişte `ppf` p=0,75'te
  scipy ile TAM eşit çıktı ve iddia *"bit-denk"* yazıldı; 20.011 noktalık ızgarada iddia
  ÇÖKTÜ (yalnız %31,3'ünde tam eşit, maks. fark 1,8e-15). Düzeltilmiş üç ayrı iddia:
  `pdf` **BİT-DENK (%100)** · `cdf` ≤2,2e-16 · `ppf` ≤1,8e-15. Bit-denk olmayan dönüşümde
  doğru cevap vazgeçmek değil iddiayı KARAR düzeyine taşımaktır: 1 sentin üstündeki her
  opsiyon fiyatında bağıl fark **8,5e-13**, `implied_vol` gidiş-dönüşünde maks |Δσ| **9e-8**.
  (Bağıl fark yalnız fiyat ~0'a giderken büyür — "değersiz opsiyon" bölgesi; orada mutlak
  fark 1e-13'tür. Bir bağıl hata metriği, paydası sıfıra giderken RAPOR EDİLEMEZ.)
- **"ÖNCE" ölçümü yük altında alınırsa optimizasyon şişer (Tur-36).** `update_from_predictions`
  ilk ölçümü 1112 ms çıktı ve değişiklikten sonra 482 ms → "2,3× kazanç" gibi göründü. O
  ölçüm ARKA PLANDA bir eğitim işi koşarken alınmıştı. İki sürüm AYNI süreçte art arda
  koşturulunca gerçek sayı **1,10×** çıktı (satır okuma aşamasında 1,29×). Kural: A/B'yi
  aynı süreçte, art arda ve tercihen aynı fonksiyondan koştur; farklı zamanlarda alınmış
  iki sayıyı bölme.
- **Kendi eklediğin basamak OKUMA maliyetini de büyütür — dersi KENDİNE uygula (Tur-36).**
  Tur-35 tam bu kuralı yazdı (*"ölçüm bütçesini OKUYAN da öder; yeni bir basamak eklerken
  'bu sayıyı kim, ne sıklıkta soruyor?' sorusu mimarinin parçasıdır"*) ve Tur-36 aynı hatayı
  bir tur sonra tekrarladı: MEGA basamağı `param_ladder_report`'u 3,02 → **8,37 s** yaptı
  (MEGA satırı tek başına **1,62 s**; pano bu raporu her Beyin sekmesinde istiyor).
  `meta` aygıtı DEPOLAMAYI kaldırır ama Python NESNE maliyetini kaldırmaz — 80 uzman × 16
  blok = **1280 modül**. Çözüm: yönlendirilen uzmanlar şekil olarak birebir AYNI olduğundan
  model `n_routed = top_k` ile kurulur ve eksik uzmanlar ÖLÇÜLEN uzman boyutuyla eklenir
  (**90,2 ms, 18×**; yedi basamakta sayı TAM AYNI). Çarpan modelin KENDİSİNDEN, `n_routed`
  ise sınıfın KURUCU VARSAYILANINDAN (`inspect.signature`) okunur → kapalı-form tablo YOK,
  ikinci sabit tablosu YOK, sessiz sapma imkânsız. Ders: bir ölçüm-maliyeti dersi yazdıktan
  sonra onu İHLAL EDEN ilk kişi genelde bir sonraki turdaki kendindir; yeni basamak eklerken
  merdiven raporunu ÖLÇEREK bitir.
- **Aynı ağırlığı iki kez hesaplama (Tur-33).** `online.FTRLProximal`'da `weights()`
  saklanmaz, her çağrıda L1-proximal kapalı formundan TÜRETİLİR; `fit_stream` örnek başına
  `predict_one`ı iki kez çağırıp `update_one` ayrıca kendi `weights()`ini alınca örnek
  başına **3× `weights()`** oluyordu. Ortak `_step` ile 1×: **2,22×**, çıktı bit-denk.
  `adwin` de her gözlemde `np.asarray(window)` çağırıyordu (profilin **%34'ü**) → numpy
  tamponunda bitişik görünüme geçildi: **1,75×**, bit-denk.
- **Bir ÖNBELLEK, onu kullanmayan KARDEŞ ÇAĞIRAN varken hâlâ O(n²)'dir (Tur-43).** Tur-7
  gösterge önbelleği (`_SharedIndicators`) O(n²)→O(n) inmişti; `meta_features.extract_features`
  bu önbelleği HİÇ görmeyen tek kardeş çağıran olarak kaldı ve EMA/ATR/BB/RSI/ADX'i büyüyen
  dilim üzerinde sıfırdan hesaplamayı sürdürdü. Ölçüldü: 4000 barlık backtest süresinin
  **%49'u** oradaydı (398 çağrı × 19,8 ms); canlıda aynı beş gösterge sembol başına ÜÇ,
  panoda DÖRT kez hesaplanıyordu (`gate.explain` kendi `extract_features`'ını koşup sonra
  `self.probability`'yi çağırarak ikinci kez koşturuyordu). Düzeltme yeni matematik değil
  `shared=` parametresini zincirin sonuna kadar geçirmekti: backtest **3,92×**, canlı
  sembol-başı blok **2,97×**, `gate.explain` **16,2×** — 2000 + 7500 karşılaştırmada
  **fark 0,000e+00**. Kural: bir optimizasyon eklendikten sonra `grep` ile o kaynağın TÜM
  çağıranları taranmalı; "önbellek var" demek "önbellek kullanılıyor" demek DEĞİLDİR.
- **Tur-7'nin ikinci yarısı: HESABI tekilleştirmek KIRPMAYI tekilleştirmez (Tur-43).**
  `_SharedSlice.ema(p)` her erişimde `parent.ema(p).iloc[:n]` kuruyordu — yeni Series + yeni
  DatetimeIndex (pandas'ta indeks dilimlemesi freq çıkarımı da tetikler) = **O(n)**.
  `detect_event` bar başına iki görünüm kurup onlardan yalnız DÖRT sayı okuyordu; kırpma
  makinesi profilin **%36'sı** idi. İki adımda kapandı: (a) görünüm başına anahtar başına
  tek kırpma (memo), (b) skaler erişimciler (`ema_at` · `close_at` · `breakout_at`) —
  kırpma da sütun çıkarma da yok. `detect_event` **9,01×**; 3941 gerçek bar + 1200 dejenere
  çağrıda **farklı verdikt 0**. Kural: "O(n) hesabı bir kereye indirdim" demek yetmez —
  o sonuca HER ERİŞİMİN maliyeti de mimarinin parçasıdır.
- **`dropna()` bir SORU değil bir KOPYADIR (Tur-43).** `x.dropna().empty` /
  `len(x.dropna()) >= k` / `x.dropna().iloc[-1]` desenlerinin hepsi yeni bir Series **ve**
  yeni bir Index tahsis eder; sorulan şey sırasıyla `count()==0`, `count()>=k` ve "kuyruktaki
  ilk geçerli değer"dir. Depoda ALTI kopya vardı; "son geçerli değer" yardımcısının DÖRT ayrı
  gövdesi (`indicators._last`, `indicators_ext._last`, `live_predict_service._last_valid`,
  pano) tek kaynağa (**`strategy.last_valid`**) indi: **8,38×**, panel bütünü 13.734 satırda
  **oy/değer/konsensüs farkı 0**. Dikkat: `dropna` NaN'ı atar ama **±inf'i ATMAZ** → kontrol
  `isnan` iledir, `isfinite` DEĞİL (aksi halde sonsuz bir gösterge değeri sessizce elenir).
- **Bir GÖRÜNÜM nesnesi hizasızsa sessizce LOOK-AHEAD sızdırır — kontrolü MAKİNEYE koy (Tur-43).**
  `extract_features(df_slice, shared=...)` yanlış pencereli bir görünüm alsaydı 400. barın
  EMA'sı 200. barın vektörüne girer ve HİÇBİR hata görünmezdi. Çözüm sözleşmeyi yorumla
  yazmak değil: görünüm `nbars` bildirir, uyuşmazlıkta önbellek YOK SAYILIP dilimin kendisi
  üzerinde kurulur (güvenli degrade, istisna yok). Look-ahead korumaları yoruma değil koda
  yazılmalıdır — Tur-7'nin nedensellik argümanı da bu yüzden testle kilitlidir.
  **Tur-45 bunu bir adım öteye taşıdı: nedenselliği KONTROLE değil VERİ YAPISINA yaz.**
  Vol filtresinin ATR kırpmasını kaldırırken ilk tasarım filtreye bir `n` (görünüm
  uzunluğu) geçirmekti; o tasarım REDDEDİLDİ çünkü yanlış bir `n` sessizce GELECEĞİ
  okuturdu. Seçilen çözüm ebeveyn dizisinin `[:n]` **sıfır-kopya numpy görünümü**dür:
  dilim geleceği TAŞIYAMAZ. Kural: *yanlış geçirilebilecek bir sayı yoksa look-ahead de
  yoktur* — bir güvenceyi çalışma-zamanı kontrolüyle kurmak zorunda kalıyorsan, önce onu
  temsil eden veriyi değiştirmeyi dene. (Kazanç: bar döngüsü **1,19×**, filtre+erişim
  **2,90×**; 3941 gerçek bar + 1005 dejenere çağrıda farklı verdikt 0.)
- **`except Exception: pass` bir İMZA kırılmasını da yutar (Tur-43).** `generate_signal`
  meta kapısını geniş bir try/except ile sarar ("kapı hatası sinyali bloke etmemeli").
  `probability` imzası `shared=` ile genişleyince test ikizleri TypeError verdi ve kapı
  SESSİZCE pasifleşti — testler yakaladı, üretim yakalamazdı (kapı kapalıyken sistem çalışır
  görünür). Zarif degrade doğru tasarımdır ama bedeli şudur: o bloğun içindeki her arayüz
  değişikliği ÇAĞRILIYOR MU diye ayrıca kanıtlanmalıdır.
  **Tur-44'te KAPANDI — ve kök neden `except`'ten DERİNDİ: `active` bir DURUM sanılıyordu.**
  Kapının tek dış gözlemi `gate.active`'dir ve o yalnız *"model dosyası yüklendi"* demektir;
  *"P(win) ÜRETİYOR"* demek değildir. İki kavram tek bayrağa binince her soruda patlayan bir
  kapı, canlı panoda ve CLI'da **"AKTİF (logistic)"** yazmaya devam eder. Onarım üç parçalı:
  (1) `meta_gate` modül düzeyinde bir HATA DEFTERİ tutar (`note_call`/`note_error`) — örnekte
  değil MODÜLDE, çünkü `reload_gate()` örneği atar ve hata geçmişi atılmamalıdır; (2) ilk hata
  ve her 100.'sü UYARI olarak loglanır (backtest bar başına çağırır — sayım TAM, log SEYREK:
  300 hata → 4 satır); (3) `gate_health()` beş durumu AYIRIR: `MODEL_YOK · HENUZ_SORULMADI ·
  SAGLIKLI · KISMI_HATA · **SUSUYOR**`. Sinyal hâlâ BLOKE EDİLMEZ (zarif degrade korundu) ama
  artık `META_GATE_ERROR` gerekçesi + `meta_gate_error` metadata'sı taşır ve durum canlı
  servisin döngü meta'sına (`snapshot()["meta"]["meta_gate"]`) girer — yazılıp OKUNMAYAN
  teşhis bu projede kusurdur (Tur-38 kuralı). Genel ders: **bir bayrak iki soruyu birden
  yanıtlıyorsa, ikisinden birinin cevabı sessizce yanlıştır.**
- **AYNI BOYUTLU bir kaynak düzenlemesi, AYNI SANİYE içinde geri alınırsa Python BAYAT
  BYTECODE'u kullanır (Tur-72).** Bu turun mutasyon koşusu `nihai/ai/options_math.py`'deki
  bir işareti (`-` → `+`, AYNI uzunluk) değiştirip testi koşturdu ve geri aldı; CPython'ın
  `.pyc` geçersizleştirmesi kaynağın `int(mtime)` **SANİYESİNE** ve `st_size`'ına bakar,
  ikisi de değişmediği için **mutasyonlu bytecode diskte kaldı ve sonraki tüm koşuları
  zehirledi**. Belirti sinsi: kaynakta doğru formül duruyor, `grep` doğruyu gösteriyor, ama
  modül yanlış sayı üretiyor — ve ilk teşhis "formül yanlış" oluyor. Ölçüldü: aynı ifade
  elle değerlendirildiğinde payda 0,5797, modül 0,6713 (tam olarak `+1/w` varyantı).
  Kural: mutasyon/geri-alma yapan her akış **`PYTHONDONTWRITEBYTECODE=1` (veya `-B`) ile
  koşmalı** ve bitiminde `__pycache__` temizlenmelidir; ayrıca `shutil.copy` mtime
  KORUMAZ, dolayısıyla "geri aldım" demek "önbellek geçersizleşti" demek DEĞİLDİR.
- **Bir GUARD, sardığı kütüphanenin EŞİĞİ ölçülmeden yazılamaz (Tur-72).** `mst_edges`
  mükemmel korele çiftin `d=0` uzaklığını `1e-12`'ye itiyordu, çünkü `scipy.sparse.csgraph`
  sıfırı "kenar YOK" sayar. Guard ETKİSİZDİ: ölçüldü, csgraph 1e-12 **ve 1e-9** ağırlıklı
  kenarı da sessizce düşürüyor, 1e-6'da alıyor. Yani "sıfırdan büyük olsun" yetmez — kenar
  ağaca girmiyordu ve MST mükemmel korele çiftin ETRAFINDAN dolaşıyordu (topoloji yanlış;
  o çift tek bağlantı olsaydı ağaç KOPARDI). Kusuru bulan şey testin kendisi değil,
  **mutasyonun KAÇMASIYDI**: guard'ı silmek hiçbir testi kırmıyordu, çünkü guard'ın
  ateşlendiği senaryo hiç sınanmamıştı. Kural: bir guard yazarken sardığı kütüphanenin
  eşiğini ÖLÇ ve guard'ın ATEŞLENDİĞİ senaryoyu ayrıca test et — kaçan bir mutasyon ya
  eksik bir test ya da ölü bir guard demektir.
- **Bir yol haritası maddesini "eksik" saymadan önce VERİSİNİN olup olmadığını sor (Tur-72).**
  39 KB'lik dış katalogun 14 başlığından yalnız altısı uygulanabildi ve elenenler
  "zor" oldukları için değil ÖLÇÜLEMEZ oldukları için elendi: L2/L3 emir defteri olmadan
  OFI/Avellaneda-Stoikov/spoofing yazmak, girdisi olmayan bir sayı basmaktır (Tur-31);
  FIX/OMS ise anayasa ihlalidir (bu sistem emir GÖNDERMEZ). Bir kapsam listesini
  değerlendirirken üç kova ayrılmalıdır: **zaten var · veri yok · anayasaya aykırı** —
  ve üçü de rapora yazılmalıdır, yoksa bir sonraki tur aynı elemeyi yeniden yapar.

- **DERSİN KENDİSİ de bayatlar: kataloğu belgeye karşı DENETLE (Tur-48).** Bu tur 9 acı
  dersin tamamı yeniden ölçüldü (132 kod adının hepsi kaynakta duruyor; 12 ölçülebilir
  iddia yeniden koşuldu ve hepsi TUTTU; merdivenin 6 parametre çapası — `deep_mlp` f=25'te
  **186.702**, MEGA **2.014.146.906** — birebir; 20 manşet sayısının hepsi kaynak belgeye
  kadar izlendi). **Bulunan tek kusur derslerin İÇERİĞİNDE değil TAŞINMASINDAYDI:**
  `episodic.LESSONS` **ACI-7'de duruyordu** — #8 (sınıf dengesizliği DSR'yi şişirir) ve
  #9 (kalibrasyon kazancı sabit tahminciyle kıyaslanmalı) kataloğa HİÇ girmemişti. Oysa
  `check_against_lessons` `agent`/`brain`/`decision`'ın okuduğu TEK ders kapısıdır; yani
  projenin kaynak kodda en çok atıf alan derslerinden ikisi (kodda #9 **45**, #8 **17**
  atıf) o kapıdan ASLA geçemiyordu — üstelik ikisi de tam olarak KANIT KAPILARININ
  bozulmasını anlatıyor.
- **Bir kuralın BİÇİMİNİ genişletmek VARLIĞINI kurmaz (Tur-48).** Tur-35 bu boşluğun
  yarısını görmüştü: `test_ders_ref_format_aci_n` desenini `ACI-[1-7]` → `ACI-[1-9]`
  genişletti *"belgeyle yeniden hizalandı"* diyerek. Desen genişledi, KATALOG genişlemedi
  ve test yalnız BİÇİM denetlediği için (`^ACI-[1-9]$`) `knowledge.ders_ref=["ACI-9"]`
  **boşa işaret ederken yeşil kaldı**. Bir kimliği doğrulayan test, o kimliğin KARŞILIĞI
  olduğunu da sormalıdır. **"Gevşek çapraz referans" IMPORT bağımsızlığıdır, DENETİMSİZLİK
  değil:** `knowledge` çalışma zamanında `episodic`i hâlâ import ETMEZ, bağ
  `tests/test_tur48_ders_katalogu.py` ile kilitlendi (katalog ↔ CLAUDE.md eşitliği + her
  `ders_ref`in VARLIĞI + yeni derslerin fiilen ATEŞLENMESİ).
- **Kusurun ikinci katmanı: dersin KONUSU olan kavram onu anmıyordu (Tur-48).**
  `deflated_sharpe` (#8'in ta kendisi) ve `brier` (#9'un ta kendisi) `ders_ref=[]` ile
  duruyordu — katalog boş olduğu için bağ hiç kurulmamıştı. Bir ders eklerken "bu dersi
  kim ANMALI?" sorusu eklemenin parçasıdır. Ayrıca işaretçi tek başına yetmez: kavramın
  kendi tuzak metni KURALI da söyler (Tur-42: cevabı taşımayan işaretçi yönlendirmez).
- **Anahtar kelime kapıları AYIRT EDİCİ olmalı (Tur-48).** `_LESSON_KEYWORDS`'e ECE
  metriği için çıplak `"ece"` eklemek cazipti; Türkçe gövdelerin İÇİNDE eşleşirdi
  (`"gelecek"` → `"ece"`) ve uyarı kapısını gürültüye boğardı. Bir uyarı her metinde
  ateşleniyorsa hiç ateşlenmiyor demektir. Kilit: `test_anahtarlar_GOMULU_parca_olarak_eslesmiyor`.
- **Bir RENK bir TAVSİYEDİR; ifade edemediği kusuru bir gösterge açığa ÇIKARAMAZ (Tur-47).**
  *(Tur-83: tersliğin KAYNAĞI bulundu — yönün kotasyon girdisi; bar kipinde kovalar düz, BULGU-83-1.)*
  Pano canlı tahmin tablosunda ham yön güvenini trafik ışığıyla boyuyordu
  (`conf>=0.66` YEŞİL) ve `%` ile sunuyordu. Canlı defterde ÖLÇÜLDÜ (63.237 çözülmüş,
  panonun KENDİ renk sınıflarıyla): YEŞİL isabet **%43,95** ↔ GRİ (`<0,40`) **%50,94** =
  **+6,98pp, z=+14,4**; kova düzeyinde ρ=**−0,889**. Hem zaman hem sembol bölmesinde ayrı
  ayrı (ACI DERS #3 çıtası). Yani pano ölçülen EN KÖTÜ alt kümeyi "en iyi" diye gösteriyor,
  operatörü oraya yönlendiriyordu — üstelik varsayılan SIRALAMA da aynı iddiayı taşıyordu.
  **10 tur görünmemesinin nedeni yapısaldır:** kova rengi her kovanın SEVİYESİNİ kodlar
  (`hit_rate>=0,55` → yeşil), kovalar arası SIRAYI değil → ters bir eğri sağlıklı bir
  eğriden AYIRT EDİLEMEZ. Bir yüzey tasarlarken sor: *"aradığım kusur olsaydı bu gösterge
  farklı görünür müydü?"* — cevap hayırsa gösterge kusuru gizler.
  **Çare TERSİNE ÇEVİRMEK DEĞİL, İDDİA ETMEMEKTİR** (ACI DERS #3/#7): GRİ sınıf %50,94 =
  yazı-tura, "düşük güveni al" bir kenar değildir. Renk nötrleşti (birim `%` → `sk`),
  sıralama anahtarı `-confidence` → `symbol` oldu, `prob_cal` (GERÇEK kalibre olasılık)
  renkli KALDI. Ve bu yasak DOKTRİN DEĞİL ÖLÇÜMDÜR: `calibration_fade_test` (2026-06-25,
  EDGE_FINDINGS **§17-D**) fade kuralını sınadı — OOS isabeti **%54,6** (>%50!) ama ort.
  getiri **−0,071%**; *isabet beklenti değildir.*
  **ÖLÇÜM YENİ DEĞİL, YÜZEYE ULAŞMASI YENİ:** terslik §17-C'de (2026-06-25) zaten yazılıydı
  ve iki ay boyunca ÜÇ ayrı yerde durdu — araştırma script'i docstring'i, `EDGE_FINDINGS`
  ve bir `config.py` yorumu — üçü de operatörün KARAR ANINDA okuduğu yer DEĞİLDİ. Kayıt ile
  YÜZEY arasındaki boşluk kusurun kendisidir. Kilit: `tests/test_tur47_guven_hukmu.py`.
- **Bir kod YORUMU çalışma-zamanı HÜKMÜ değildir (Tur-47).** Yukarıdaki terslik
  `config.py`'de bir yorumda ZATEN yazılıydı (*"canlı veride confidence şu an isabetle TERS
  korele"*) ve tam 10 tur orada kaldı: operatör panoyu okur, yorumu değil. Aynı sınıf kusur
  B-4'te `pwin` tablosu için kapatılmıştı; KARDEŞ tablo (`buckets`) hükümsüz basılmaya devam
  etti. `report()` artık `confidence_verdict` taşır ve pano onu basar. **Bir SKORA Brier
  uygulanmaz** — bu, onu olasılık sanmanın ta kendisidir (Tur-36 §5: `confidence` log-kaybı
  1,282 ↔ taban 0,694 = 1,85 katı); bu yüzden `_calibration_verdict(konu="skor")` Brier'i
  HİÇ hesaplamaz ve etiketleri ayrıdır (**`SIRALIYOR` ≠ `TUTARLI`**: doğru sıralayan bir skor
  kalibre OLMAYABİLİR). Kural: bir hüküm mekanizması yazdığında `grep` ile AYNI ŞEKİLDEKİ
  tabloları ara — kardeşine konmayan kural, orada sessizce yanlış okunmaya devam eder.
  Tarama ÜÇ yüzeyde kusur buldu: pano rengi/sıralaması, `report()`'un hükümsüz tablosu ve
  **kalibrasyonu FİT EDEN script'in kendisi** (`calibrate_forecast` ham güvenin MSE'sini
  `brier` diye kaydediyordu — Brier bir olasılığı puanlar, `confs` ise kalibrasyonun
  GİRDİSİDİR; artık `skor_mse` + `mse_taban_orani` + `mse_bilgisiz`). O alanı okuyan kod da
  yoktu = Tur-38'in "yazılıp okunmayan alan kusurdur" kuralının aynı yerde ikinci ihlali.
- **İKİ hüküm fonksiyonu AYNI veride ÇELİŞİYORDU (Tur-47) — ACI DERS #9'un kendi ifadesi.**
  `prediction_tracker.verdict()`'in `INVERTED` dalı *"sinyali tersine çevirmek bile
  düşünülebilir"* diyordu; AYNI MODÜLDEKİ `_calibration_verdict` ise tam bu çıkarımı açıkça
  YASAKLIYOR. Bugün dal ateşlenmiyor (isabet 0,4732 → NO_EDGE) ama eşiğe **0,3pp** uzakta.
  Etiket (ÖLÇÜM) korundu, ÖNERİ kaldırıldı. Kardeşi `forward_harness._verdict` de
  *"fade/yön-çevirme test edilebilir"* diye ZATEN KOŞULMUŞ bir teste yolluyordu (§17-D:
  isabet %54,6, getiri −0,071%): **cevabı taşımayan bir işaretçi, bir sonraki turu kapanmış
  bir kuyuyu yeniden kazmaya gönderir.** Kural: bir yasağı koda yazarken `grep` ile aynı
  KARARI veren tüm fonksiyonları tara — biri önerirken diğeri yasaklıyorsa yasak yoktur.
- **Bir OPTİMİZATÖR yanlış BİRİMDE arıyorsa çoklu-başlangıç onu KURTARMAZ (Tur-49).**
  `garch11_fit` MLE'yi getiri ölçeğinde koşuyordu (günlük varyans ~1e-4); SLSQP'nin
  varsayılan sonlu-fark adımı (~1,5e-8) ve `ftol` o ölçekte omega yönünde ANLAMSIZDIR ve
  arama köşeye çakılır. ÖLÇÜLDÜ (**296 gerçek S&P sembolü**, 1000 günlük log-getiri):
  `uncond_var` %8,4'ünde örneklem varyansının 5 katından BÜYÜK, %1,4'ünde 0,2 katından
  KÜÇÜK, %1,7'sinde fit tamamen `None`; en kötü sapmalar **18.068×** ve **1,3e-06×**.
  Onarım yeni bir algoritma değil bir DEĞİŞKEN DEĞİŞTİRMESİDİR: GARCH(1,1) ölçek-eşdeğişken
  olduğu için (r→c·r ⇒ omega→c²·omega, alpha/beta sabit) artıklar birim varyansa normalize
  edilir, omega geri ölçeklenir, olabilirlik Jacobian ile (`−0,5·n·ln(var0)`) düzeltilir →
  **%0,0 / %0,0 / %0,0** ve süre 658 → **81 ms/sembol**. Docstring beş başlangıç noktasını
  "yerel-optimum zorluğuna karşı önlem" diye anlatıyordu; beşi de AYNI bozuk yüzeyde
  arıyordu. **Bir çözümün VAR OLMASI onun DOĞRU EKSENDE olduğunu göstermez.** Sayısal bir
  yordamı kabul etmeden önce girdisinin BÜYÜKLÜK MERTEBESİNİ sor.
- **Bir KISITIN kenarı bir TAHMİN değildir (Tur-49).** `alpha+beta` durağanlık kısıtının
  0,999 kenarına oturduğunda `1−(alpha+beta)` tahmin edilmiş bir büyüklük değil KISITIN
  ARTIĞIdır; `omega/(1−a−b)` gerçek veride örneklem varyansının **34–528 katını** basıyordu
  ve docstring bunu *"bu bölgeyi kısıtla eler"* diye anlatıyordu — elemiyordu, KIRPIYOR ve
  kırpılmış değeri tahmin gibi sunuyordu (Tur-33'ün "kırpma bir ölçüm değil KARARDIR"
  dersinin optimizatör tarafı). Artık `persistence` açıkça raporlanır, köşedeki fit
  `uncond_var=None` döner (gerçek veride **22/296 = %7,4**) ve kısa-ufuk öngörü çalışmayı
  SÜRDÜRÜR — tanımsız olan yalnız uzun-ufuk SINIRIdır. Kural: bir optimizatör çıktısını
  raporlamadan önce "sınırda mı durdu?" sorusu SORULMALI ve cevabı çıktıyla TAŞINMALIDIR.
- **CPython'da `x ** 2` taşmada İSTİSNA atar, `x * x` atmaz (Tur-49).** `quant_math.py`
  başındaki sözleşme *"hiçbir girdi İSTİSNA fırlatmaz"* diyor; 1247 dejenere çağrılık fuzz
  iki ihlal buldu (`ewma_vol`, `advanced_math.bocpd`) ve ikisi de sıkı bir döngüde Python
  float `**` kullanıyordu — etraflarındaki numpy kodu ise `inf`-tabanlı akış varsayıyordu.
  Aynı ifadenin numpy'da `inf`, Python'da `OverflowError` vermesi bir "detay" değil bir
  SÖZLEŞME KIRILMASIDIR. Üretimde tetiklenmesi |r| > 1,3e154 ister (yani LATENT) ama bir
  sözleşme yazılmışsa DENETLENİR: fuzz, iddiayı test etmenin en ucuz biçimidir.
- **Bir ÖNBELLEK, ondan OKUMA biçimi pahalıysa hâlâ pahalıdır (Tur-49).** Tur-7 gösterge
  HESABINI, Tur-43 KIRPMAYI, Tur-45 ATR ERİŞİMİNİ tekilleştirmişti; geriye OKUMA BİÇİMİ
  kaldı ve backtest profilinin **%54'ünü** (87,0 s / 162,1 s) `pandas.indexing.__getitem__`
  yiyordu. ÖLÇÜLDÜ: `Series.iloc[int]` **2,573 µs**, `ndarray[int]` **0,126 µs** = **20,4×**.
  En büyük çağıran `detect_rsi_divergence`'tı: bar başına **9,3** pandas dilim/etiket-araması
  yapıp **SEKİZ SAYI** okuyordu (8.000 barlık koşuda 74.102 çağrı — ikincinin iki katı).
  Beş detektör + `extract_features` sıfır-kopya numpy pencerelerine geçti; uçtan uca
  backtest 5 sembol × 20.000 barda **163,70 → 66,36 s (2,47×)**, işlem sayısı/getiri/
  kazanma oranı/max-DD/Sharpe **8 ondalığa kadar AYNI**, fonksiyon düzeyinde 29.200 barda
  FARK 0. Kural: "önbellek var" → "önbellek kullanılıyor" (Tur-43) → **"önbellekten okumak
  ucuz"** üç AYRI iddiadır; üçü ayrı ayrı ölçülür.
- **Bir A/B, iki tarafın YAPILANDIRMASI aynı olduğu kanıtlanmadan okunamaz (Tur-49).**
  Taban ölçümü `git worktree` ile HEAD'de alındı ve sonuçlar ayrıştı (AAPL 7 ↔ 9 işlem) —
  bir an "optimizasyon davranışı değiştirdi" gibi göründü. Sebep kodda DEĞİL ORTAMDAYDI:
  worktree `.env` içermez (gitignore'da) → `CONFIG` başka eşiklerle yüklendi. Tur-36'nın
  "ÖNCE ölçümü yük altında alınırsa optimizasyon şişer" dersinin kardeşi: orada ZAMAN,
  burada ORTAM kirletti. Aynı süreçte, art arda, aynı yapılandırmayla koştur.
- **Kaynak metnini tarayan bir regresyon kapısı KODA bakmalı, ANLATIYA değil (Tur-49).**
  Bu turun onarım YORUMLARI kusurun eski ifadesini (`.iloc[-30:]`) bilerek ALINTILIYOR;
  ham metinde arayan kapı kendi açıklamasına takılıp "kusur geri geldi" dedi. Çözüm
  `tokenize` ile yorum ve dizgi sabitlerini atmaktır. Tur-48'in dersinin tersten hâli:
  her metinde ateşlenen bir uyarı hiç ateşlenmiyor demekse, kendi belgesinde ateşlenen
  bir kapı da hiç kapı değildir.
- **Statik araçların TEMİZ olması kusur YOK demek değildir (Tur-49).** `ruff check .` ve
  `mypy .` tur başında temizdi, 2873 test yeşildi — üç kusur yine de vardı. Onları bulan
  şey üç BAĞIMSIZ SORGUYDU: (a) saf matematiği DIŞ referansa (scipy kapalı-form, sonlu-fark,
  bilinen DGP, gerçek piyasa verisi) karşı çapraz-doğrulamak, (b) modülün KENDİ YAZILI
  sözleşmesini fuzz'lamak, (c) sıcak yolu profillemek. Statik araç, BAKTIĞI kusur sınıfının
  yokluğunu söyler; başka sınıfların yokluğunu değil.
- **GEÇİCİ bir koşul KALICI bir sonuç doğuruyorsa, o bir kusurdur (Tur-50).**
  `CircuitBreaker.can_emit` hız sınırına takıldığında `_tripped = True` kuruyordu. Hız
  sınırı bir PENCERE koşuludur ve pencere kendi kendini temizler (60 sn / 1 saat); mandal
  ise yalnız `reset()` ile açılır — ve `reset()`in depoda **hiçbir çağıranı YOKTU**. Canlı
  tarayıcı `breaker`ı bir kez kurar (uzun ömürlü döngü, watchdog altında günlerce), tek
  yoğun bir dakika mandalı indirir ve süreç yeniden başlayana kadar SİNYAL YAYILMAZ.
  ÖLÇÜLDÜ (canlı `logs/decision_log_*.txt`): **6 gerçek tetikleme ↔ 9.194 mandallı blok
  = %99,93**; en uzun kesintisiz pencere **275 dk / 2.349 blok** (2026-07-24) ve mandal
  TAKVİM GÜNLERİNİ AŞTI (07-21…07-24 loglarında tek bir gerçek tetikleme yok, yalnız
  `TRIPPED:` satırları). Onarım yeni mekanizma değil, birbirine yapışmış iki mekanizmanın
  AYRILMASIDIR: hız sınırı yalnız o çağrıyı reddeder, `_tripped` yalnız Madde 47'nin
  bilinçli kararını taşır. Onarımın **ALT SINIRI** ölçüldü (üst sınır değil): yayın sayısı
  ≤ deneme sayısı olduğundan, önceki 60 sn'sinde 5'ten az DENEME bulunan her mandallı blok
  yeni kodda kesinlikle geçerdi → **1.609/9.194 = %17,5** kanıtlanabilir taban. Kural: bir
  guard yazarken *"koşul geçtiğinde sonuç da geçiyor mu?"* sorusu tasarımın parçasıdır.
- **Bir bayrağı KURAN yol varsa, TEMİZLEYEN yol da olmalıdır (Tur-50).** Tur-44'ün dersi
  (*"bir bayrak iki soruyu birden yanıtlıyorsa, ikisinden birinin cevabı sessizce
  yanlıştır"*) burada zaman eksenine döndü: `_tripped`i kuran İKİ yol vardı (hız sınırı +
  `trip()`), temizleyen `reset()`in ise SIFIR çağıranı. Madde 47'nin *"flatten-all yerine
  sinyalleri durdur"* kancası böylece ERİŞİLEMEZDİ — sistemi durduran tek şey, onu
  yanlışlıkla durduran hız sınırıydı. Kanca var olan yüzeye bağlandı (mevcut `tg_handlers`
  tablosuna `/dur` + `/devam`; yeni soyutlama YOK) ve `/dur`un AÇIK POZİSYONLARA
  dokunmadığı testle kilitlendi: bir operatör freninin zarar-kesmeyi geciktirmesi bu
  sistemdeki en pahalı hata olurdu (Tur-38'in `ai_advisor` sınırıyla aynı).
- **Bir AYAR, onu okuyan kod yoksa ayar değil YANLIŞ BİLGİDİR (Tur-50).** `ast` taraması
  126 config alanının **5'ini**, `nihai/` altındaki 605 public tanımın **4'ünü** okuyansız
  buldu. İkisi yalnız ölü değil YANILTICIYDI: `OptionFilters.min_dte=5/max_dte=60`
  yürürlükteki pencereyle (`OptionStrategyConfig.dte_min=30/dte_max=45`) ÇELİŞİYORDU —
  `max_dte`yi 30'a çeken operatör sistemin hâlâ 45 gün seçtiğini göremezdi; ve
  `close_before_expiry_dte_override` kendi yorumunda bir yedekleme ZİNCİRİ tarif ederken
  `check_exit` global alanı doğrudan okuyup override'ı sessizce yok sayıyordu (Tur-42'nin
  *"bir yorumdaki anahtar adı da API'dir"* dersinin tersten hâli). Ayrıca `margin_ok`
  (Madde 13) yürürlükteki Reg T bağıyla ÇELİŞEN ikinci bir tanımdı: o REDDEDER,
  `_magnitude_cap_units` ise KÜÇÜLTÜR — ve küçültme Tur-40'ta bilinçle seçilmişti.
  **Bu depo aynı sınıfı ELLE iki kez kapatmıştı** (`portfolio_caps_breach` yorumları:
  *"daha önce config'de tanımlı ama uygulanmıyordu"*, *"config'de ölüydü"*) ve kural
  MAKİNEYE çevrilmediği için geri büyüdü. Artık `tests/test_tur50_olu_kapilar.py` her
  config alanının ve her `nihai/risk` public tanımının bir OKUYANI olduğunu denetler.
- **Uygulanamayan bir kuralı EKLEME; ÖLÇÜLEMEYEN bir kapı zaten kusurdur (Tur-50).**
  `Regulation.finra_maintenance_margin=0,25` (Madde 13'ün sürdürme yarısı) uygulanmıyordu
  ve ilk içgüdü kapıyı EKLEMEKTİ. Ölçüm başka şey söyledi: sürdürme marjı PORTFÖY düzeyi
  BRÜT maruziyet ister; canlı defterden brüt kurulmayı denedi ve **brüt/özkaynak ortanca
  5,80× / azami 13,68×** çıktı — ama okuma GEÇERSİZDİ, çünkü `DUAL_TIER_ENABLED=true` ve
  `trades.csv` İKİ BAĞIMSIZ katmanın işlemlerini tek deftere yazıyor; iki portföyün toplamı
  tek portföyün tavanıyla kıyaslanamaz (Tur-49'un A/B dersi, bu kez kıyaslamanın ÖZNESİNE).
  Yani kural yalnız uygulanmıyordu değil, **İHLALİ DE ÖLÇÜLEMİYORDU**. Ölçülemeyen bir
  kapı eklemek, bu turun kapattığı kusuru yeniden üretmek olurdu → sabit silindi, durum
  PDT notundaki duruşla yazıldı (hesap düzeyinde broker uygular) ve brüt ölçülemezliği
  AÇIK KALEM olarak kayda geçti.
- **Bir kapıyı yazarken önce KENDİ dersini ihlal edip etmediğine bak (Tur-50).** Bu turun
  tarayıcısının ilk sürümü regex'ti ve `htf_bar_minutes`i "okunuyor" saydı — eşleşme bir
  YORUMDAN geliyordu (`…(HTF). htf_bar_minutes … ile uyumlu tutulmalı`). Tur-49 tam bunu
  yazmıştı (*"kaynak metnini tarayan bir kapı KODA bakmalı, ANLATIYA değil"*) ve bir tur
  sonra ilk ihlal eden o dersi yazan taraf oldu. Çözüm `ast`tir: ayrıştırma sonucunda
  yorumlar ve dizgi sabitleri zaten YOKTUR. (Tur-36'nın *"bir ölçüm-maliyeti dersi
  yazdıktan sonra onu ihlal eden ilk kişi genelde bir sonraki turdaki kendindir"*
  gözleminin ikinci doğrulaması.)
- **Bir kalite kapısının SEÇTİĞİ kural kümesi, bakmadığı kusuru gizler (Tur-50).** Yeni
  test dosyası depoyu `ast` ile ayrıştırınca düşen `SyntaxWarning` bir kusur ortaya
  çıkardı: `nihai/console.py:10` docstring'i `logs\*.log` yazıyor (ham dizgi değil) →
  Python 3.12'de uyarı, gelecek sürümde **SyntaxError**. Ne `ruff` (W605 bu projenin
  `select` listesinde YOK: E,F,I,B,UP) ne `mypy` bakıyordu; 318 dosyada tek örnek. İroni
  kayda değer: uyarıyı üreten modül tam da CLI çıktısının kodlama GÜVENLİĞİ için yazılmış
  olandır (Tur-24). Tur-49'un dersinin devamı: statik araç, BAKTIĞI kusur sınıfının
  yokluğunu söyler — ve *neye baktığı* `select` listesinde bir KARARDIR.
- **Bir KAYDIRMA, serinin DÜZENLİ olduğu gösterilmeden bir ZAMAN kaydırması değildir (Tur-51).**
  İki katman birden kusurluydu ve birlikte iki ay boyunca sahte bir "monoton ilişki" ürettiler:
  (a) **Kaydırma MİKTARI, kaydırılan değişkenin PENCERESİ okunmadan seçilmişti.**
  `iv_rank_threshold_test` ileri oynaklığı `hv_current.shift(-W)` ile vekilliyor ve W'yi
  *"sağlayıcı belgelemiyor"* diyerek TARIYORDU. Pencere ÖLÇÜLDÜ (568.195 satır / 603 sembol,
  fiyattan geriye-bakan RV ile eşleştirme): `hv_current` = geriye-bakan **30 işlem günlük**
  RV, kor. **0,99504** (sembol ortancası 0,9998; <0,95 olan 8/600; likiditeden BAĞIMSIZ).
  Yani `hv_current(t+W)` `(t+W−30, t+W]` ölçer ve **W<30'da etiketin (30−W)/30'u KARAR
  BARINDAN ÖNCEDİR** — varsayılan W=21'de **%30**. "İleri" etiketin bir kısmı geçmişti.
  (b) **Serinin KENDİSİ düzensizdi**, yani N satır sabit bir ufuk bile değildi. Dolt serisi
  fiyat barlarının ~%93,4'ünü taşır ama boşluklar ÖBEKLİ (1713 günün 623'ünde kapsama <%50):
  `shift(-30)` hedeflenen 42-44 takvim günü yerine GERÇEKTE ortanca **70**, ortalama 65,2,
  **std 34,0** gün ileri gidiyor (%59,6'sı 60 günü aşıyor); geri yönde `shift(21)` ortanca
  **49**. Ufuk sembole ve döneme göre DEĞİŞİYOR.
  **SONUÇ ÇÜRÜTÜLDÜ.** CLAUDE.md 2026-08-13'ten beri *"IV Rank bandı TERS, ilişki MONOTON
  (t=−7,35), bonusu çevirmek kullanıcı kararıdır"* diyordu. İZOLE KONTROL (aynı satırlar,
  aynı evren/dönem/W=30/rank/exec-lag/örtüşmesizlik; **tek değişken etiket tahmincisi**,
  n=15.943): vekil bant t=**−2,29** + monoton dilim deseni ↔ fiyattan hesaplanan temiz ileri
  RV t=**+0,61**, monotonluk YOK; iki etiketin korelasyonu yalnız **0,7177**. Düzenli
  ızgarada nihai ölçüm bant gün-bazlı **t=−0,17 = AYIRT EDİLEMEDİ**. Bant hiçbir yönde bilgi
  taşımıyor — ne doğrudur ne terstir; "çevir" önerisi GERİ ÇEKİLDİ.
  **Ders kendine de uygulanır:** bu turun İLK veri seti aynı kusuru taşıyordu. Çözüm
  düzensiz seriyi TAM takvime taşımak + SINIRLI (5 işlem günü) nedensel ffill uygulamaktır
  (`build_vrp_dataset._to_price_calendar`) — sınırsız ffill bir haftadan eski IV'yi "bugünkü
  IV" diye sunardı (Tur-31). Kural: **bir ölçümün UFKU, ölçümün kendisi kadar taşınmalıdır**
  (Tur-34'ün "ölçüm KOŞULU taşınmalıdır" kuralının zaman eksenindeki hâli); bir `shift(N)`
  yazmadan önce *"bu serinin her satırı bir işlem günü mü?"* sorusu SORULMALIDIR.
- **Boş bir sağlayıcı kolonu HATA vermez, yalnız TARİH ARALIĞINI kısaltır (Tur-51).**
  DoltHub `volatility_history` `iv_month_ago`/`hv_month_ago` kolonlarını **2024'ten beri
  %100 NaN** bırakıyor (2019-22 ≈%0, 2023 %10). Onlara bağlanan hat 2023-11'de SESSİZCE
  biter: bu turun ilk veri setinde satırların **%55'i** düştü ve **2,7 yıl** kayboldu —
  istisna yok, uyarı yok, yalnız daha kısa bir `min/max` tarihi. Doldurulduğu yerde bile
  tanımı belirsizdi: `iv_month_ago ≈ iv_current(t−k)` eşleştirmesi k∈[18,23]'te MONOTON
  artan hata verir (en iyi k=18 = ARAMA ARALIĞININ SINIRI = uydurma sonuç — Tur-49'un
  "sınırda duran fit bir tahmin değildir" dersinin veri tarafı). Kural: bir dış kolona
  bağlanmadan önce NaN oranını **yıl-yıl** bak; ve bir veri setinin `min/max` tarihi
  beklenenden kısaysa bu bir ayrıntı değil BİRİNCİL teşhistir.
- **Sarmalayıcı vergisi göstergenin kendisinde de var (Tur-43, dördüncü tekrar).** Tur-34
  (`logsumexp`) ve Tur-36 (`norm.cdf`) dersinin gösterge hattındaki hâli: `rsi` gövdesinin
  **%43'ü** beş `Series.where/mask/clip` + bir `replace` çağrısının blok-yöneticisi yüküydü.
  Wilder yumuşatması (ewm) pandas'ta KALDI, yalnız etrafındaki elemanwise aritmetik numpy'a
  indi — işlem SIRASI değişmediği için iddia "operasyonel denk" değil **BİT-DENK** kurulabildi
  ve 22.825 barlık gerçek seride + 1800 dejenere senaryoda ölçüldü: `rsi` **5,88×**,
  `adx` **2,22×**. Vektörleştirirken serinin `name`'i gibi taşıyıcı alanlar da korunmalı
  (pano payload'ı ona bakar).
- **Satır düzeyi t, TARİH düzeyi t ile birlikte raporlanmadan okunamaz — birim değişimi
  t'nin yalnız BÜYÜKLÜĞÜNÜ değil YÖNÜNÜ de çevirebilir (Tur-53, BULGU-53-1).** Tur-27
  (endeks rebalans) bağımsızlık biriminin gözlem değil TARİH olduğunu ölçmüştü: satır-t
  +5,75 → tarih-kümelenmiş t +0,52 (**11× şişme**). Açığa satış akışı ailesinde aynı
  değişim daha sert davrandı: short bacak satır-t **+1,10 → gün-t −0,55**, long bacak
  **−0,92 → +1,82**. Sebep, satır ortalamasının çok-satırlı (kesitsel olarak korele)
  günlerce baskılanmasıdır. Yani "t küçüldü ama işaret aynı" varsayımı YANLIŞTIR; kesitsel
  bir olay ailesinde iki t AYNI tabloda basılmalıdır (`docs/EDGE_FINDINGS.md` §29).
- **Bir veri setinin İSTENEN aralığı ile GERÇEKLEŞEN aralığı ayrı ayrı raporlanmalıdır
  (Tur-54, BULGU-54-1).** Opsiyon yüzeyi ailesi 2021-01→2026-06 istedi, **2024-10→2026-06**
  aldı ve hiçbir istisna/uyarı üretmedi. Sebep sağlayıcı kapsamasıdır: DoltHub `option_chain`
  2021-23'te işlem günlerinin ~%55'ini, 2024 ortasından sonra ~%96'sını taşıyor; 60 günlük
  z-skor penceresi 40 gözlem istediği için ilk üç yılda **hiçbir sembol tek satır bile**
  üretemiyor (eşiği geçen sembol payı %0,0/%0,0/%0,0 → %79,2/%92,1/%94,6). Bu, BULGU-51-2'nin
  (*"boş bir sağlayıcı kolonu yalnız TARİH ARALIĞINI kısaltır"*) takvim tarafıdır. `harness`
  ön-koşulu veri setinin KENDİ [min,max] aralığında boş ay arar, "istenen aralığın 3,7 yılı
  hiç doğmadı" durumunu GÖREMEZ (açık kalem; kapı sonuca göre değiştirilmedi). Kural: bir
  veri seti kurulduğunda `min/max` beklenenle KIYASLANMALI, ve verdikt kapsadığı dönemle
  birlikte yazılmalıdır (`docs/EDGE_FINDINGS.md` §30).
- **Çoklu-test sayacı DEKORATİF DEĞİL: ilk kez bir kapının verdiktini çevirdi (Tur-55,
  BULGU-55-1).** Tur-46 DSR deflasyonunu defterdeki ayrık aile sayısıyla çarpmaya başlarken
  *"geçmişte koşulan ailelerin verdikti DEĞİŞMEZ"* diye kayda geçmişti; mekanizma bugüne
  kadar hiçbir şeyi çevirmemişti. Mevsimsellik ailesinde çevirdi (aynı seri, n=1412,
  SR_fazlalık +0,0882): DSR `n_trials=4`'te **0,9845 PASS**, 8'de 0,9600 PASS, 16'da 0,9206
  fail, **36'da 0,8558 fail**. Eski sabitle bu kapı geçilirdi. **İddianın SINIRI:** bir
  KAPININ verdikti değişti, AİLENİN verdikti değil — `exec_lag_1` zaten düşürüyordu.
  Okuma: arama sürdükçe çıta gerçekten sıkışıyor; 9 aileden sonra gelecek ilk PASS'e
  birinciye gösterilenden çok daha fazla şüpheyle bakılmalıdır.
- **ACI DERS #7 ALTINCI kez (Tur-55).** Mevsimsellik ailesi ÜÇ kapıyı geçti — `purged_cv`
  **0,534** (bu kapıyı geçen ilk aday), iki kapı simülasyonu da PASS, üstelik %21 seçicilik
  ve Welch t=3,73 ile BÜYÜK örneklemde — ve 1 bar gecikmede kapı fazlalığı **+0,0068 →
  −0,0014**, t **3,73 → 0,40** oldu. Desen artık altı ailede aynı (pairs §22 · giriş kayması
  §23 · lead-lag §26 · rebalans §27 · VRP §28 · mevsimsellik §31): sağlıklı görünen her
  fazlalık için önce `exec_lag_1` sorulmalıdır, en son değil.
- **Bir GEREKÇENİN yazılı olması onu DOĞRU yapmaz (Tur-59, BULGU-59-1).** Özellik logunun
  aday-şartı kodda gerekçesiyle duruyordu: *"pwin_side'sız satıra özellik yazmak anlamsız
  (etiket yok)"*. Cümle kendi içinde tutarlıydı ve aylarca kimse sorgulamadı; **öncülü
  yanlıştı** — satırın etiketi `pwin`den değil `predictions_log`un `hit`/`fwd_ret_pct`
  kolonundan gelir ve o kayıt zaten yazılıyordu. Bedeli ölçülebilir: etiketsiz havuz
  **7,96×** küçük kaldı (66.694 ↔ 8.378 satır) ve Tur-35/36 iki tur boyunca "ön-eğitim
  neden işe yaramıyor?" diye yanlış eksende çalıştı. Kural: bir yorumdaki gerekçe bir
  İDDİADIR; iddiayı taşıyan cümleyi değil, iddianın ÖNCÜLÜNÜ denetle (*"etiket gerçekten
  yok mu? nereden geliyor?"*). Tur-47'nin *"bir kod YORUMU çalışma-zamanı HÜKMÜ değildir"*
  dersinin tersi hâli: orada doğru bilgi yanlış yerdeydi, burada yanlış bilgi doğru yerde.
- **Tek yönde okunan bir kanıt YARIM kuraldır (Tur-59, BULGU-59-2).** `check_exit` bar
  açılışını STOP tarafında sıra kanıtı sayıyordu (`GAP_STOP`) ama HEDEF tarafında hiç
  sormuyordu. Sonuç "kötümser" değil YANLIŞTI: hedefin üstünde açılan bir barda hedef ilk
  tikte doludur, sıra BELİRSİZ değil BELİRLENMİŞTİR ve kod onu ters okuyordu. Ölçüldü:
  1d'de 18 çıkış yanlış sınıflanıyordu (283 → 289 işlem); 1h'te fark BİT-DENK, yani kusur
  yalnız barın hedefi aşacak kadar uzun olduğu çözünürlükte ısırıyordu. Kural: bir veriyi
  bir KARARIN kanıtı olarak kullanıyorsan, o kanıtın SİMETRİK durumunu da sor — asimetrik
  okunan bir kanıt, kötümser görünen bir yanlıştır.
- **Bir VARSAYIMI savunmak için önce MARUZİYETİNİ ölç (Tur-59).** "Aynı barda stop mu hedef
  mi önce?" sorusunun kötümser cevabı doğrudur; eksik olan, o cevabın KAÇ çıkışta
  kullanıldığıydı. Ölçüldü: **1/147 (1h) ve 0/283 (1d)**. Bu sayı olmadan 1m alt-bar
  çözücüsü ("gerçekçilik") yazmak savunulabilir görünüyordu; sayıyla birlikte açıkça
  YAGNI oldu — enjekte edilebilir 12 satırlık kanca bırakıldı, veri boru hattı KURULMADI.
  Tur-33'ün *"aradığım kusur olsaydı bu gösterge farklı görünür müydü?"* sorusunun
  maliyet tarafı: bir varsayımın kaç kez ateşlendiğini bilmeden onu düzeltmeye harcanan
  emek ölçülemez.
- **Bir KOLON eklemek, o defterin ŞEMA GÖÇÜNÜ test eder (Tur-59, BULGU-59-5).**
  `trades.csv`'ye `tier` eklemek latent bir kusuru tetikleyecekti: `log_trade_csv` yalnız
  *"dosya var mı?"* diye soruyor, eski başlıklı dosyaya YENİ alan sırasıyla append ediyordu
  → o satırdan sonra tüm kolonlar KAYAR (okuyanlar: pano Günlük Defteri, `tearsheet`,
  `postmortem`, VaR). Aynı sınıf `prediction_tracker._ensure_schema` ile aylar önce
  kapatılmıştı; KARDEŞ defter açıktaydı (Tur-47 kuralının üçüncü doğrulaması). Kural: bir
  append-only deftere kolon eklemeden önce *"başlık değişince ne olur?"* sorusu SORULMALI —
  ve cevabı yoksa göç ONARIMI eklemenin ÖN KOŞULUDUR.
- **Bir ölçümü YÜZEYE bağlamak, onun MALİYETİNİ de yüzeye bağlar (Tur-59, BULGU-59-8).**
  `turnover_rate`'i panoya bağlamak için float kaynağı gerekiyordu ve ilk sürüm
  `yfinance.Ticker.get_info()`'yu doğrudan uç içinden çağırdı. Testteki AĞ GUARD'ı bunu
  yakaladı — ama asıl kusur ağ değil MİMARİYDİ: saniyelik bir çağrıyı bir HTTP isteğinin
  içinde ödemek Tur-35/36'nın *"ölçüm bütçesini OKUYAN da öder"* kuralının ihlalidir.
  `fetch` varsayılan **False** oldu (okuyucu ÖNBELLEĞİ okur), doldurmak ayrı ve bilinçli
  bir adım (`--warm`). Kural: bir okuma yoluna ağ/disk çağrısı eklerken *"bunu kim, ne
  sıklıkta çağıracak?"* sorusu tasarımın parçasıdır; kalite kapısı bunu yakaladıysa şanslısın.
- **Bir GÖÇ "aynı verinin daha iyisi" olduğu KANITLANMADAN göç değildir (Tur-59, P3).**
  `ALPACA_BARS_FEED=sip` iki ölçümle savunuluyordu: IEX konsolide hacmin yalnız %3'ünü
  görüyor (doğru) ve SIP ücretsiz planda yetkili (doğru). Ama A/B koşulunca üçüncü bir şey
  çıktı: **bar KÜMESİ değişiyor** (400 barın yalnız ~%52'si ortak — SIP uzatılmış seansı
  taşır) ve ortak barlarda bile **karar SPY'de 2/30 ayrışıyor**. Tur-31'in VIX göçünde
  kurulabilen *"göç ölçülmüş sonuçları geçersiz KILMAZ"* cümlesi burada KURULAMAZ. Kural:
  bir veri kaynağı göçünde ölçülecek şey KAYNAĞIN kalitesi değil KARARIN değişip
  değişmediğidir; ve "daha çok veri" bir kalite iddiası değil bir ÖRNEKLEM değişikliğidir.
- **Bir OOS onayı, ARİTMETİK bir kimliği TEMİZLEMEZ (Tur-60, BULGU-60-2).** `dir_bar`
  uyuşma kuralı örneklem kapısını, kümele-dayanıklı t'yi (7,407), Bonferroni'yi ve
  zaman-bölmeli OOS'u (6,736) geçti; hepsini geçmesinin nedeni kuralın iyi olması değil,
  ölçülen büyüklüğün `ret_live ≈ ret_bar − slip` kimliğinin yeniden yazımı olmasıydı — bir
  kimlik HER dilimde geçerlidir, o yüzden kayma kontrolü OOS'tan ÖNCE koşar. Teşhis
  mekanizmayı sorunca geldi: uyuşmaz grupta bar→bar isabeti **%60,9 ↔ %50,3** (yön DAHA
  DOĞRU) ama `slip_bps` **+20,8 ↔ +0,4** (giriş kovalanmış). Kural: bir kapı adayının ÖNCE
  NE ÖLÇTÜĞÜ, sonra ne kadar anlamlı olduğu sorulmalıdır — ve "ne ölçtüğü"nün operasyonel
  hâli aynı sonucu FARKLI bir paydayla yeniden hesaplamaktır.
- **Getirinin PAYDASI bir tercih değil, ölçümün TANIMIDIR (Tur-60).** Her satırı kendi
  giriş baskısına bölmek doğru P&L sorusudur ama *"hangi satırları alsaydık"* sorusuna
  KARIŞTIRILAMAZ: ikinci soruda payda kuralın kendisiyle korelasyonludur ve hiçbir bilgi
  taşımayan bir kural bile pozitif uplift üretir. Bir alt küme karşılaştırması yaparken
  paydanın gruba göre değişip değişmediği SORULMALIDIR.
- **"Ölçülemedi" ÜÇ ayrı şeydir; tek etikete yıkmak düzeltilecek şeyi gizler (Tur-60).**
  Örneklem yok (`spread_bps`: 29 satır) · TAKVİM yok (gözlemevi kolonları: 2 gün, servis
  koşmadı) · ÇIKTI yok (`shadow_runner`: 1150/1150 abstain, karar motoru hiç yön üretmiyor).
  İlki beklemeyi, ikincisi servisi çalıştırmayı, üçüncüsü prior'ı gerektirir. Bu yüzden her
  yüzeyde `n`, `n_clusters` ve `coverage_days` AYRI raporlanır.
- **Kendi harness'ının İLK çıktısına inanma (Tur-60).** Bu turun ölçeri ilk koşuda iki
  PROMOTE bildirdi ve ikisi de yanlıştı: biri projenin en pahalı yasağını (ters
  kalibrasyona işlem kurmak, ACI DERS #3/#7), biri bir bölme kimliğini "kenar" diye
  raporluyordu. Kapılar sonuçtan SONRA eklendi — ama Tur-23 kuralını ihlal etmeden, çünkü
  eklenen şey ADAYI ELEYEN bir eşik değil adayın NE OLDUĞUNU söyleyen bir ölçümdür; ikisi
  de bugün `PROMOTE` alan sentetik pozitif kontrolü geçirmeye devam eder.
- **Yalnız reddeden bir kapı, kapı değil bir `return False`tir (Tur-60).** Her kanıt
  kapısının bir POZİTİF KONTROLÜ olmalı: gerçekten ayıran, gürültülü ve çok kümeye yayılı
  sentetik bir yüzey PROMOTE almalı. Ve o kontrol GÜRÜLTÜ taşımalıdır — tam ayrışmada
  artıklar sıfırlanır, kümele-dayanıklı SE de sıfır çıkar ve `_cluster_t` dürüstçe "hüküm
  yok" der (dejenere bir kurgu bir kapıyı doğrulayamaz).
- **PİYASA GENELİ bir özellikte satır sayısı örneklem DEĞİLDİR — ve işaret DÖNEBİLİR
  (Tur-74, §32).** Dış listenin Madde 9'u *"backwardation tüm swing alımlarını kilitlesin"*
  diyordu. `vixts_v1` ailesi kuruldu (20-bar kırılma tabanı, ufuk 5, PIT evren, 43.066 satır
  / 595 sembol / **1277 gün**) ve iddia İKİ BİRİMDE birden ölçüldü: **SATIR** düzeyinde
  contango +0,1294% ↔ backwardation **+0,0289%** (iddia doğrulanmış görünüyor, 4,5 kat fark)
  ama **GÜN** düzeyinde +0,1384% ↔ **+0,1730%** — **işaret DÖNÜYOR**. Sebep: vade yapısı
  özellikleri piyasa genelidir, aynı gün kırılan onlarca sembol AYNI değeri taşır, satır
  ortalaması çok-satırlı günlerce baskılanır. Etkin örneklem 43.066 değil **60 backwardation
  GÜNÜ**. Kapılar da aynı şeyi söyledi: purged_cv **0,505** · gate_time **t=−1,39** (kapı
  ZARAR VERİYOR) · DSR **0,0005** · exec_lag t=−0,77 → **KENAR YOK**; tek PASS (`gate_symbol`
  t=4,06) **%2,9 seçicilikle** geldi = Tur-23'ün `n_kept` kuralı. Kural: bir özelliğin
  KESİTTE DEĞİŞİP DEĞİŞMEDİĞİNİ sor — değişmiyorsa bağımsızlık birimi GÜNdür (BULGU-53-1'in
  üçüncü doğrulaması) ve sembol bölmesi o özellik üzerinde yapısal olarak ayırt edici değildir.
- **Bir GÖÇÜN A/B'si, girdinin TÜM tüketicileri sayılmadan tamamlanmış sayılamaz (Tur-74).**
  Dış liste (Madde 9) `strategy._load_vix`'i yfinance'ten CBOE'ye taşımayı istiyordu ve ilk
  ölçüm göçü destekliyordu: VIX rejim KAPISI kararı 2515 günün **0'ında** ayrışıyor (iki kaynak
  da 215 "OFF" günü), üstelik kapı zaten varsayılan KAPALI. Tur-31 de aynı A/B'yi yapmış ve
  aynı sonuca varmıştı. **İkisi de tüketici kümesini eksik saymıştı:** `_vix_close_before`'un
  ikinci çağıranı `meta_features.py`'dir ve orada önemli olan boolean değil DEĞERİN kendisidir
  — günlerin yalnız **%3,9'u bit-denk** (std 0,052, max \|Δ\| 2,61). `data/meta_model.json`'ın
  standartlaştırılmış katsayısıyla (w=+0,3076, sd=3,9646) bu **p_win'de en fazla 0,0506**
  oynama demektir; eşiğe yakın bir kararı çevirmeye yeter ve model yfinance serisiyle
  EĞİTİLMİŞTİR → train/serve kayması. Göç YAPILMADI. Kural: bir girdiyi başka kaynağa taşımadan
  önce `grep` ile o girdinin HER okuyucusunu say — bir KAPI (eşikli boolean) kaynak farkına
  kör olabilirken aynı sayıyı ÖZELLİK olarak kullanan bir model olmaz; "karar değişmiyor"
  cümlesi hangi karar olduğu söylenmeden kurulamaz.
- **`git checkout <dosya>` bir MUTASYONU değil DOSYAYI geri alır (Tur-74, süreç dersi).**
  Mutasyon koşusunda bir mutantı geri almak için `git checkout nihai/data/cboe_data.py`
  kullanıldı; o dosya turun KENDİ dört düzenlemesini de taşıyordu ve hepsi silindi. Belirti
  yalnız sonraki testin kırmızısıydı ve ilk teşhis YANLIŞ oldu ("mutasyon testi bozdu").
  Tur-73'ün *"devredilen dosyaya dokunma"* dersinin tekil-oturum hâli: kayıp eşzamanlı bir
  ajandan değil **kendi geri-alma komutundan** geldi. Kural: mutasyon geri alması **hedefli
  ters değiştirme** olmalıdır (`replace(mutant, orijinal)`); `checkout` yalnız çalışma ağacı
  o dosyada TEMİZKEN güvenlidir ve bunu varsaymak yerine ÖLÇMEK gerekir.
- **Fail-soft bir yolu İSTİSNAYLA problama — yuttuğu şey tam da odur (Tur-74).**
  `fetch=False` guard'ının ilk testi `pd.read_csv`'yi AssertionError atan bir sahteyle
  değiştiriyordu; `get_index` fail-soft'tur ve geniş `except Exception` taşır → istisna
  YUTULDU, boş df döndü, test YEŞİL kaldı ve guard'ı TAMAMEN SİLEN mutasyon **KAÇTI**.
  Sahte bir BAYRAĞA çevrilince (çağrıyı listeye yaz, sonra `assert not denendi`) mutasyon
  yakalandı. Kural: gözlemi yan etkiyle yap; bir istisna, onu yutmak için yazılmış bir
  yolda hiçbir şey kanıtlamaz.
- **Bir sözleşme, deponun ZATEN KANONİK olan sözleşmesiyle ÇELİŞİYORSA modül üretimde
  HİÇ koşmaz (Tur-79, BULGU-79-1).** `drift_monitor` tz-aware damgayı reddediyordu ve
  KENDİ sözleşmesi içinde doğru davranıyordu; üretim logunun %37,2'si tz-aware olduğu
  için modül bir kez bile okuyamadı ve bunu hiçbir istisna söylemedi (fonksiyon dürüstçe
  `None` dönüyordu). Aynı sorunun depoda DÖRT kopyalı, belgelenmiş ve ölçülmüş bir cevabı
  vardı (`_canon_ts`). Kural: yeni bir modül yazarken *"bu veriyi başka kim okuyor ve
  NASIL yorumluyor?"* sorusu sözleşmenin parçasıdır; bir girdi biçimini reddetmeden önce
  o biçimin üretimdeki PAYINI ölç. (Tur-35'in "aynı desen üç kopyada" dersinin sözleşme
  tarafı: orada kod kopyalanmıştı, burada kopyalanması gereken kod KOPYALANMAMIŞTI.)
- **Bir PSI, POPÜLASYONU sabit tutulmadan okunamaz (Tur-79, BULGU-79-2).** Aynı log,
  aynı pencere, aynı fonksiyon: karışık popülasyonda **12** "BÜYÜK sürüklenme", homojen
  popülasyonda **3**. Fark modelin dünyasından değil KAYIT EDİCİNİN örneklemesinden
  geliyordu — aday-dışı bar payı Temmuz %0'dan Eylül %85'e çıkmıştı (BULGU-59-1'in
  bilinçli politika değişikliği) ve `quality_score`/`rr_ratio` o satırlarda 0 yazılıyor.
  Hiçbir istisna çıkmaz, hiçbir kolon eksilmez; yalnız iki dönemin SATIR KÜMESİ artık
  aynı şey değildir. Kural: iki pencereyi kıyaslayan her ölçümde (drift, A/B, kapı
  simülasyonu) *"iki tarafta AYNI TÜR satır mı var?"* sorusu ÖNCE sorulur, ve karışım
  oranı seçilen süzgeçten BAĞIMSIZ raporlanır — artefaktı gizleyen panel kusurun
  kendisidir. (Tur-53'ün "bağımsızlık birimi satır değil TARİH olabilir" dersinin
  örneklem-bileşimi tarafı.)
- **Belgelenmiş bir BASİTLEŞTİRMENİN öncülü de bayatlar (Tur-79, BULGU-79-3).**
  `_inverse_vol_weights` docstring'i *"gerçek piyasa kapitalizasyonu verisi bu modülde
  YOK"* diyordu; dürüsttü ve yazıldığında DOĞRUYDU. `etf_holdings` doğunca yanlış oldu ve
  fark ölçüldü: BL önselinde **maks 8,65 puan** (NVDA %15,3 → %23,9). Kural: bir turda
  YENİ BİR VERİ yeteneği açtığında, o verinin YOKLUĞUNA dayanan belgeli basitleştirmeleri
  `grep` ile ara — Tur-64 "koşul HÂLÂ geçerli mi?" diye soruyordu, buradaki soru
  "koşul ARTIK geçerli değil mi?"dir ve ikincisi kendiliğinden görünmez, çünkü belge
  hâlâ tutarlı okunur. Onarımın şekli de bir karardır: varsayılan ESKİ davranış kalmalı
  (bit-denk), yeni önsel DIŞARIDAN verilmeli (katman kuralı), geçersiz girdi sessizce
  eskiye DÜŞMEMELİ ve hangi önselin kullanıldığı ÇIKTIYLA taşınmalıdır (Tur-63).
- **Bir aileyi ÖLÇEMEMEK, o ailede kenar olmadığı DEĞİLDİR (Tur-55, 13F).** SEC 13F toplu
  setleri var ama `INFOTABLE` şemasında ticker yok; evrene bağlamanın tek ücretsiz yolu isim
  eşlemesi ve ölçüldü: tam eşleşme %30,7, CUSIP'lerin %42,2'si birden fazla isimle geliyor,
  eşleşen ticker'ların %69,4'ü birden fazla CUSIP alıyor. Böyle bir harita BAĞIMSIZ
  DEĞİŞKENİN KENDİSİNİ yanlış ölçerdi (farklı menkul kıymetleri tek tickerda toplayarak) —
  gürültülü değil YANLIŞ. Defterde `VERI_ONKOSULU_YOK` ile ayrı durur ve çoklu-test sayacını
  ŞİŞİRMEZ: koşulmamış bir aile bir arama değildir.
