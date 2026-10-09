# Değişiklik Günlüğü

## [2.22.0] — İstenen pencere ile doğan pencere aynı şey değildir (Tur-63, 2026-08-31)

Denetim ekseni: **`veri_kapsami` ön-koşulu deliği arıyordu, KESİLMİŞ ucu değil.** Hiç
doğmamış bir yılın veri setinde izi yoktur — o yıl bir delik değil, aralığın DIŞIDIR;
kapı deliksiz 21 ay görüp PASS diyor, verdikt "5,5 yıl" diye okunuyordu (BULGU-54-1,
Tur-54'te ölçülmüş ve kapı bilerek değiştirilmemişti: aday FAIL ettikten sonra kapı
sıkıştırmak sonuca göre kural değiştirmektir). Adayı olmayan bu turda makineye çevrildi.
Tam süit **3289 passed / 1 skipped** (Tur-62: 3265 → +24, hepsi bu turun) · `ruff check .` temiz · `mypy .` 187 dosya temiz.
Rapor: `docs/TUR63_RAPORU_2026-08-31.md`.

### Düzeltildi
- **BULGU-63-1** `harness._stage_coverage` uç kesilmesini göremiyordu. Yedi veri seti
  ölçüldü: `optskew` istenen pencerenin **%69,2'sini**, `seasonality` **%40,9'unu**,
  `babmacro` **%13,3'ünü** hiç üretmemiş, tek uyarı çıkmamıştı. Kapı artık istenen
  pencereyi (`expected_start`/`expected_end`) ve ailenin ısınmasını (`warmup_days`)
  KURUCUNUN BEYANINDAN alır ve doğan aralıkla kıyaslar; tolerans (%5) yapıdan türetildi
  (yayın gecikmesi + takvim kayması), veriyi geçirmek için ayarlanmadı. Kesilmiş veri
  **FAIL_VERDICT ("kenar YOK") değil PRECONDITION_VERDICT ("HÜKÜM YOK")** verir — ölçülmemiş
  bir dönem çürütülmüş sayılamaz ve defter o aileyi denenmiş sayıp sonraki adayın DSR
  eşiğini haksız yere yükseltemez.
- **BULGU-63-1b** Isınmalar tahmin edilmedi, ÖLÇÜLDÜ: `babmacro=224g` (**bileşik ısınma**
  — `f_ivol` rolling'inin girdisi ZATEN beta rolling'ini bekler → 80+80 işlem günü; depoda
  hiçbir yerde yazılı değildi), `seasonality=730g` (aile pencere içinde 2 yıllık aynı-ay
  geçmişi ister → 2021-2022 hakkında hüküm ÜRETEMEZ), `shortflow=84g`, `optskew=56g`.
  Beyanla birlikte üç ailenin "kaybı" kendi YAPILARI çıktı; `optskew`inki gerçek kesilme.
- **BULGU-63-2** GEÇEN kapsam ölçümü hiçbir yere ulaşmıyordu (kapı yalnız FAIL ederse
  `gates`e giriyordu): defterin 11 satırının hiçbirinde kapsanan dönem yoktu. Kapsam kapısı
  artık PASS ederken de verdiktte durur ve defter satırı `kapsam` alanı taşır. Geçmiş
  satırlar geriye DOLDURULMADI (yalnız-ekleme defteri).
- **BULGU-63-4** `fred_data.get()` kırpılmayı ZATEN ölçüyordu (`attrs["kirpildi"]`, Tur-51)
  ama `get_all()` Series→DataFrame geçişinde attrs'ı düşürdüğü için bayrak hiçbir çağırana
  ulaşmıyordu; `build_babmacro_dataset` bu yüzden AYNI ölçümü elle yeniden yapıyordu.
  `get_all` artık `attrs["kirpik"]` ile taşıyor, kurucu kopyasını sildi (kural tek yerde).

### Eklendi
- `tests/test_tur63_kesilmis_pencere.py` — **21 fonksiyon / 24 test**, beş kapı:
  (A) kesilmiş uç ön-koşulu düşürür ve koşu deftere yazılmaz, (B) "ölçülemedi" ile "ihlal
  yok" ayrı cümledir (beyansız/yarım/geçersiz beyan sessizce PASS'e dönüşemez),
  (C) geçen ölçüm verdiktte + defterde durur ve kapsamsız ESKİ satırlar okunabilir kalır,
  (D) her `run_candidate` çağıranı pencereyi `ast` ile denetlenerek beyan eder ve beyan
  SABİT yazılamaz, (E) kırpılma bayrağı kaynaktan okunur. Regresyon kapısı: `BACKLOG.md`
  bir daha kapının bu durumu "göremediğini" yazamaz.
- **BULGU-63-3** `veri_kapsami` (kabul kapısı) Tur-52'den beri TESTSİZDİ (`grep` → 0
  sonuç): `_MAX_EMPTY_MONTH_FRAC` sessizce 1,0 yapılsa süit yeşil kalırdı. Artık tolerans
  sabitleri de kilitli.

### Ölçüldü, değiştirilmedi
- Üç eski kurucu (`vrp`, `insider`, `rebalance`) pencereyi hâlâ bildirmiyor → kapı
  "ÖLÇÜLEMEDİ" diyor, FAIL etmiyor (`panel.py`nin donmuş-kurucu kararı korundu).
- `seasonality` son kesmesi %4,7 — toleransın hemen altında. `vrp` istenen pencerenin
  DIŞINA taşıyor (kesilmenin tersi). Geçmiş 9 ailenin verdikti yeniden hesaplanmadı.

## [2.21.0] — Kendi testi çağırıyorsa kapı hâlâ ölüdür (Tur-62, 2026-08-31)

Denetim ekseni: **Tur-50'nin ölü-kapı tarayıcısı `tests`i çağıran sayıyordu** — bir kuralın
var olduğunu kanıtlamak için yazılan test, o kuralın ÇAĞRILDIĞININ da delili oluyordu.
Kapı kendi kendini tatmin ediyordu ve para yolunda İKİ kural bu yüzden bağlanmamış kaldı.
Tam süit **3265 passed / 1 skipped** (Tur-61: 3246 → +19; 18'i bu turun, 1'i
operatörün çalışma ağacındaki PDT testi) · `ruff check .` temiz · `mypy .` 187 dosya temiz. Rapor: `docs/TUR62_RAPORU_2026-08-31.md`.

### Düzeltildi
- **BULGU-62-1** `RiskManager.pdt_blocks_day_trade` (Madde 10, 23) hiçbir üretim yolundan
  ÇAĞRILMIYORDU; `README.md` onu Madde 10'un uygulaması olarak gösteriyor, sayaç işliyor,
  pano gösteriyordu — ama fren yoktu. Operatör 2026-08-31'de `PDT_ENABLED`i açmış,
  `config.py`ye de *"giriş yolundan ÇAĞRILMIYOR"* notunu bırakmıştı. Artık `size_stock` ve
  `size_long_option` yeni **`_entry_blocked()`** demetinden geçiyor (günlük kayıp · ardışık
  stop · PDT tek yerde; Tur-28'in dersi yapıya çevrildi — iki gövdede kopya fren yasak).
  **Fren GİRİŞİ kapatır, çıkışı değil:** her giriş bir STOP ile açılır, bütçe bittikten
  sonra açılan pozisyonun stop'u aynı gün tetiklenirse bot ya kuralı çiğner ya stop'u
  yutar; iki kuralı da sağlanabilir bırakan tek hamle pozisyonu açmamaktır.
  Etki ÖLÇÜLDÜ (`data/trades.csv`, 544 kapanış, 2026-07-06→08-17): kapanışların **%82,5'i
  (449) aynı-gün**, 14 işlem gününün **13'ünde** günlük 3 sınırı aşılmış (azami 89/gün).
  Canlı hesapta yine de **BİT-DENK**: `nlv_after` 294.296$–645.264$, eşiğin 11,8 katı →
  fren düşmez (bu bit-denklik bir testle çapalandı). Küçük hesapta ise belirleyicidir ve
  `.env.example` `INITIAL_EQUITY=25000` ile gelir — tek kayıp eşiğin altına düşürür.
- **BULGU-62-4** `pdt_max_day_trades_under_threshold` varsayılanı **10**du; kuralın adı
  ve `risk_manager`ın kendi yorumu FINRA'nın gerçek değerini yazıyordu (*"5 iş günü
  içinde max 3"*). 10'un gerekçesi depoda HİÇBİR yerde yoktu → **3**e çekildi
  (canlı hesapta bit-denk; gevşetmek env ile). Bir regülasyon sabitinin adını taşıdığı
  kuralla çelişmesi, uygulanmayan bir kuraldan daha tehlikelidir: uygulanıyor GİBİ
  görünür. Aynı ölçümde ikinci sapma KAYDA geçti (düzeltilmedi): `pdt_lookback_days`
  penceresi TAKVİM günü sayıyor, FINRA İŞ günü der → sayaç ~2 gün erken sıfırlanıyor;
  kapı canlı NLV'de atıl olduğu için takvim bağımlılığı açılmadı, sapma kaynağa yazıldı.
- **BULGU-62-3** `tests/test_tur50_olu_kapilar.py` tarayıcısı onarıldı: çağıran kümesi
  artık **yalnız üretim** (`nihai`, `research`), tarama **modül düzeyi public fonksiyonları
  da** görüyor, ve gerekçe TAŞIMAK ZORUNDA olan bir izin listesi (`_URETIMSIZ_IZIN`) var.
  Sözdizimi tarayan kardeş kapı (geçersiz escape) köklerini artık AÇIKÇA veriyor — o kapı
  çağrı değil sözdizimi denetler, `tests` oraya dahil olmalıdır.

### Eklendi
- **BULGU-62-2** `/api/status` → `beta_hedge` + pano **Beta Maruziyeti (Madde 14)** kartı.
  `RiskManager.beta_hedge_suggestion` Tur-59'da (BULGU-59-6) yazılmış ve kendi docstring'i
  *"operatöre gösterilecek bir ÖLÇÜM + öneri"* diyordu; depodaki tek çağıranı TESTTİ —
  Tur-61 BULGU-61-2'nin ("`status()` yazmak yetmez, okunması gerekir") üçüncü kardeşi.
  `suggestion=None` İKİ ayrı şeydir ve tek alana yıkılmadı (Tur-60 dersi): kart
  **ÖLÇÜLMEDİ** (betası ölçülmüş pozisyon yok) · **limit içinde** · **öneri** durumlarını
  ayrı gösterir, kapsam (`coverage`) öneriyle BİRLİKTE taşınır.
  **Kart ilk okumada bir şey buldu** (2026-08-31 19:16, 22 açık pozisyon): net nominal
  **+47.669$** (yürürlükteki tavan hiçbir ihlal görmüyor) iken net beta-dolar
  **−1.188.667$**, yani limitin (588.861$) **2,02 katı**; öneri `LONG SPY 599.807$`,
  kapsam %82 (4 pozisyonun betası ölçülmemiş, uydurulmadı). Hiçbir tavan
  DEĞİŞTİRİLMEDİ — nominal mi beta mı bağlayıcı olacağı operatörün kararıdır ve
  ölçümü görmek o kararın ön koşuluydu.
- `tests/test_tur62_test_kapisi_saymaz.py` (17 fonksiyon / 18 test) — kilit tek seferlik düzeltmeyi değil
  KURALI korur: tarayıcının varsayılan kökleri bir daha `tests` içeremez (`ast` ile) ·
  `_okunan_adlar` kök listesini `_py_dosyalari`'ndan almak zorunda · izin listesinin her
  girdisi gerekçeli · `_entry_blocked` üç freni de çağırıyor ve iki boyutlandırıcı da
  demetten geçiyor · eşik altında hisse+opsiyon girişi `PDT_DAY_TRADE_LIMIT` ile
  reddediliyor · eşik üstünde (canlı NLV) sayaç 50 kat dolu olsa bile onaylanıyor ·
  `PDT_ENABLED=0` gerçekten kapatıyor · panonun okuduğu her `beta_hedge` alanı uçta var ·
  ölçülmemiş beta "ihlal yok"tan ayrılıyor · `config.py` bir daha "çağrılmıyor" diyemez.

### Ölçüldü, DEĞİŞTİRİLMEDİ
- `nihai/ai`de üretim çağıranı olmayan üç tanım (`NeuralEnsemble.member_spread`,
  `FTRLProximal.predict_one` / `update_one`) — üçü de **kural değil kütüphane API'si**;
  üretim `fit_stream` (aynı `_step` çekirdeği) üzerinden koşar. Kapıyı `nihai/ai`ye açmak
  ayrı bir karardır. `config.py`nin 125 alanı için aynı sıkı tarama **0** verdi.
- `daily_max_loss_pct=0` (operatör kararı): `state.json` `daily_pnl=−30.464$` /
  `equity=294.423$` → o gün günlük kayıp **%10,3**, eski %10 eşiği TETİKLENİRDİ. Eşiğin
  değeri bir karardır; yalnız `0/negatif = devre dışı` sözleşmesi `consecutive_stops` ile
  aynı biçime getirildi.

## [2.20.0] — Köprülerin yüzeyi + belge↔kod kilidi (Tur-61, 2026-08-31)

Denetim ekseni: *yazılıp okunmayan (ya da hiç koşulmayan) yüzey kusurdur* (Tur-38) —
**altı ayrı yerde** tekrar etmişti. **3246 test** (+49; 22'si yeni, 27'si iki opsiyonel bağımlılık kurulunca İLK KEZ koşan) · ruff/mypy temiz (187 dosya).
Rapor: `docs/TUR61_RAPORU_2026-08-31.md`.

### Düzeltildi
- **BULGU-61-1** `live_predict_service._htf_direction` — `1 if e20 > e50 else -1`, yani TAM
  EŞİTLİK sessizce **−1 (ayı HTF onayı)** sayılıyordu ve fonksiyonun KENDİ docstring'iyle
  ("belirsiz → 0") çelişiyordu. Ders bu dosyada zaten uygulanmıştı (Tur-28 `_bias_and_confidence`,
  Tur-33 `strategy.vote` tek kaynağı) — kardeş fonksiyon 20 satır aşağıda açıkta kalmıştı.
  Ölçüldü: düz HTF penceresinde güven **0,7760 → 0,5044** (sıfır bilgi %35 ceza);
  gerçek veride 400 sembol · 7.531.642 barda tam eşitlik 406 kez (%0,0054). Yerel `_yon`
  kopyası da `vote`a indi → NaN fiyat artık SHORT oyu değil "bilgi yok".
- **BULGU-61-4** `CLAUDE.md` "Komutlar" `python -m nihai.main` diyordu; öyle bir modül YOK
  (`nihai.execution.main`). Her oturumun okuduğu sözleşmenin ilk çalıştırma komutuydu.

### Eklendi
- **BULGU-61-2** `/api/system/health` → `bridges`: `ai_advisor.status()` (Tur-38) ve
  `shadow_bridge.status()` (Tur-60) artık YAYINDA. İkisinin de tek çağıranı TESTTİ; üçüncü
  kardeş `neural_shadow.status()` `/api/ai/overview`da zaten yayındaydı. Önemi `last_error`da:
  köprüler ilk hatayı bir kez uyarı, gerisini debug loglar → "sessiz ölüm yok" koruması
  OKUNMADIĞI için kendisi sessizdi. Uç tarayıcıyla AYNI süreçte koşar, sayaçlar gerçektir.
  Pano `Sistem & Araçlar` sekmesine **Karar Köprüleri** kartı (kip · çağrı/veto · terfi
  defteri · son hata) eklendi.
- **BULGU-61-3** `.env.example`: üretim kodunun okuduğu **23 belgesiz ayar** yazıldı
  (`NEURAL_SHADOW`, `AI_OBSERVATORY`, `ALPACA_HIST_*`, `DOLT_OPTIONS_*`, `HUB_BREAKER_SEC`,
  `QUOTE_SANITY_PCT`, iç-AI budama sınırları …). Tur-60 raporu operatöre "NEURAL_SHADOW
  açık kalsın" diyordu ve o ad config şablonunda YOKTU.
- `/api/ai/stores` + pano **İç-AI Defterleri** kartı (**BULGU-61-6**): `nihai/ai` yedi kalıcı
  defter yazıyordu ve hiçbirinin üretim okuyucusu yoktu. Yüzey eklenince ölçüm anında çıktı:
  **yedinin BEŞİ hiç yazılmamış** (yazan yol üretimde bir kez bile koşmamış), kalan ikisi
  19-20 gün bayat. Yedi panel değil TEK uç + TEK kayıt (`AI_STORES`), kayıt `ast` ile koda
  karşı denetleniyor. Satır sayısı bilerek sayılmaz (okuma yolu dosyayı taratmamalı).
- `deadman.ps1` + `register-deadman.ps1` (**BULGU-61-7**): `nihai/notifier/deadman.py`
  Tur-59'da tam da "servisler sessizce öldü" hâli için yazılmıştı ama KOŞUCUSU ve GÖREVİ
  hiç yazılmamıştı — modül üretimde bir kez bile koşmadı. Görev kaydı INTERACTIVE
  kullanıcıyladır (Tur-12: SYSTEM, OneDrive profiline erişemez); kayıt adımı operatörde.
- `tests/test_tur61_kopru_ve_belge.py` (13 test) + `tests/test_tur61_ai_depolari.py` (7 test)
  — kilit tek seferlik düzeltmeyi değil KURALI korur: env okuması belgesiz kalırsa, belgeye
  (ya da bir `.ps1`e) olmayan modül yazılırsa, panonun okuduğu bir alan uçtan düşerse,
  `status()` yeniden yetim kalırsa, yeni bir defter kayıtsız doğarsa ya da ölü adam
  anahtarının koşucusu/kaydı kaybolursa test kırmızı olur.

### Ayrıca düzeltildi
- **BULGU-61-5** `api_extra._tail_jsonl` kuyruk yerine TÜM dosyayı okuyordu — Tur-35'in üç
  modülde birden kapattığı desenin DÖRDÜNCÜ kopyası. `atomic_io.tail_lines` tek kaynağına
  bağlandı (davranış aynı, okunan bayt düştü).
- **BULGU-61-8** `deadman` `logs/project.stopped` bayrağını bilmiyordu → operatörün BİLEREK
  kapattığı sistemde seans boyunca her 15 dk yanlış alarm üretecekti (bilinçli duruma
  bağıran izleyici susturulur; asıl arıza günü susturulmuş alarm hiç olmayandan kötüdür).
  Artık watchdog'larla AYNI sözleşme: verdikt `PROJE_KAPALI`, alarm yok, **ölçüm gizlenmez**.

### Ortam / işletim
- `quantstats` + `empyrical-reloaded` kuruldu (ikisi de `requirements.txt`te bildirilmişti).
  Etki ÖLÇÜLDÜ: `/api/tearsheet` artık 503 değil gerçek HTML üretiyor ve **hiç koşmamış 26
  test** (backtest metrik çapraz-doğrulaması, modül düzeyi `importorskip`) ilk kez yeşil —
  Tur-26'nın "kalite kapısı körse kapı değildir" dersinin aynısı.
- `NihaiAlgo-ForwardHarness-Daily` görevi **37 gündür Disabled**ti (son koşu 2026-07-25,
  sonuç `0xC000013A` = oturum kapanışında öldürüldü) → `history.csv` 37 gün bayattı.
  Yeniden etkinleştirildi **ve bir kez elle koşturularak doğrulandı**: kapsama 63.909/66.694,
  bekleyen 0, verdikt `NO_EDGE`; `history.csv`/`shadow_promote.json`/`gate_sim.json` taze.
- Servislerin kapalı olması ARIZA DEĞİL: `logs/project.stopped` (2026-08-30, `proje-kapat.ps1`)
  ve watchdog'lar bu kararı doğru uyguluyor. Açmak operatörün adımıdır (`.\proje-ac.ps1`).

## [2.19.0] — Gölgede koşanlar asıl sisteme bağlandı (Tur-60, 2026-08-31) — KARAR-DIŞI

`nihai/execution/shadow_bridge.py`: gözlemevi + kovalama sapması + (opsiyonel) nöral gölge
artık AYRI süreçte değil, aday sinyal ÜRETEN `main.scan_symbol` içinde koşuyor. Varsayılan
kip `advise` — ölçüler gerekçeye/karar loguna yazılır, **karar değişmez** (`veto=False`,
`size_mult=1.0` yapısal). `research/shadow_promote.py` on gölge yüzeyi TEK sözleşmede ölçer
(örneklem · kümele-dayanıklı t · Bonferroni · ters kalibrasyon · kayma artefaktı ·
zaman-bölmeli OOS) ve ölçüm günlük yan-rapor yoluna bağlandı.
**Hiçbir yüzey terfi etmedi**: `pwin` INVERTED (BULGU-60-1), `dir_bar` SLIP_ARTIFACT
(BULGU-60-2), beşi takvim yetersizliğinden hükümsüz (BULGU-60-4).
Rapor: `docs/TUR60_RAPORU_2026-08-31.md`.

## [2.18.0] — Dış rapor triyajı #2 (Tur-59, 2026-08-30)

Dış bir asistanın 7 başlıklı listesi koda karşı denetlendi (üç iddia yanlış). Bu turda
hiçbir kenar iddiası yoktur; eklenen her şey ölçüm kusuru onarımı, veri biriktirme yolu ya
da görünürlük yüzeyidir. **BULGU-59-1** özellik logu politikası (aday şartı kalktı →
etiketsiz havuz 7,96×), `GAP_TARGET`, karekök piyasa etkisi, `nihai/signals/breadth.py`,
opsiyon arşivi + ölü adam anahtarı. Rapor: `docs/TUR59_RAPORU_2026-08-30.md`.

## [2.17.0] — Dış rapor triyajı: köprüler, muafiyetler, ölçüm (Tur-58, 2026-08-30)

Dış bir asistanın 9 başlıklı eksik listesi + 8 maddelik yol haritası **koda karşı denetlendi**;
üç iddia yanlış, biri raporun söylediğinden BÜYÜK çıktı. Rapor: `docs/TUR58_RAPORU_2026-08-30.md`.
**3097 test** (+45) · ruff/mypy temiz (181 dosya) · MainHub 27 yeşil.

### Eklendi
- `tools/sync_mainhub.py` — MainHub'ın üç sağlayıcı dosyası artık `nihai/data/`'dan ÜRETİLİR
  (import yeniden yazımı + LF + isort). argümansız koşu (VARSAYILAN) sürüklenmeyi çıkış 1 ile bildirir, `--apply` yazar;
  `tests/test_tur58_hub_koprusu.py` kapıyı koşar. Raporun `nihai_core` paketi yerine asgari
  çözüm: tek kaynak + otomatik türetme + kapı.
- `/asset_flags` ucu (iki hub_server) + `HubProvider.asset_flags()` — `DATA_PROVIDER=hub`
  artık Madde 30 borç bayraklarını HUB'dan okur; her tüketici süreç ayrı trading istemcisi
  kurmuyor.
- `options_analyzer.early_exercise_risk()` — Madde 25 ex-temettü erken kullanım eşiği
  (LONG call), iki üretim yolundan çağrılır. `EXDIV_GUARD=block` kapıya çevirir.
- `event_calendar.ex_dividend_context()` + `CompanyEvents.last_dividend_amount` (additive).
- `nihai/observability.py` — `RateBudget` (monotonik pencere, yumuşak tavan freni) +
  Prometheus `prometheus_text()`; pano `/metrics` ucu.
- `log_retention.compress()` — gzip orta bandı. Ölçüldü: **82,4 → 5,5 MB (×15,1)**.
- `market_calendar.local_session_window()` + `python -m nihai.data.market_calendar
  [--at|--duration]` — ABD seansının YEREL karşılığı (Madde 43).
- `data_provider.stream_status()` — akış karnesi zincir gezintisi TEK kaynakta
  (`_find_stream_status` artık ona devrediyor); `/metrics` WebSocket sağlığını buradan okur.
- Pano `Sistem & Araçlar`: dört metrik kartı (kota · akış · döngü gecikmesi · RSS/CPU)
  + opsiyon panelinde **Ex-temettü erken kullanım** satırı; `.env.example` dört yeni anahtar.

### Düzeltildi
- **BULGU-58-1** — `asset_flags()` zincirinde ölçüm YOKLUĞU (None) artık aramayı bitirmiyor;
  ARIZA (istisna) bitiriyor. Bir uç eklemek mevcut bir yedek yolu sessizce kapatıyordu.
- **Section 1256 muafiyeti** — `wash_sale_check` SPX/NDX/VIX/RUT/XSP için de wash-sale
  işaretliyordu; nakit uzlaşılı endeks opsiyonları §1256 kapsamında piyasaya göre değerlenir
  ve wash-sale onlara UYGULANMAZ. `tax_1256` yıllardır yalnız metadata olarak taşınıyordu.
- **Çapraz wash-sale etiketi** — kayıt `asset_type` taşır, hisse↔opsiyon durumu
  `WASH_SALE_CAPRAZ` ile ayrılır (araç türü bilinmiyorsa iddia EDİLMEZ).
- **`tur17-shadow-gorev.ps1` seans hizası** — eski `-At "09:30" -DurationHours 7` varsayılanı
  ABD seansıyla **0 saat** örtüşüyordu (seans TR 16:30-23:00/17:30-00:00); saat ve pencere
  artık borsanın o günkü takviminden türetilir.
- **Madde 44** — token karşılaştırması `hmac.compare_digest` (zamanlama kanalı) + IP başına
  kaba-kuvvet kilidi (10 hata → 60 sn, 429).

### Değiştirilmedi (bilinçli — gerekçeler raporun §4'ünde)
OMS/canlı broker · çok bacaklı opsiyon · §1256'yı skorda önceliklendirme · SQLite/DuckDB
göçü · WebSocket/SSE pano · supervisord/Docker göçü · 2FA/RBAC.

## [2.16.0] — Opsiyon akışı ailesi + veri yetenek ÖLÇÜMÜ (Tur-57, 2026-08-30) — KARAR-DIŞI

Dış bir veri-kaynağı önerisi triyaj edildi: iddialar **çağrılarak ölçüldü** (üçü yanlış çıktı)
ve ölçüm sonucuna göre uygulanabilir olan her şey kodlanıp panoya bağlandı.
Rapor: `docs/TUR57_RAPORU_2026-08-30.md`.

### Ölçüldü (2026-08-29, ÜCRETSİZ Alpaca anahtarıyla)
- **SIP tarihsel bar YETKİLİ** — IEX konsolide hacmin yalnız **%2,2-3,4**'ünü görüyor
  (AAPL/SPY/NVDA, 3 gün). Kapalı olan tek şey **son 15 dakikalık** SIP penceresi.
- **OPRA opsiyon zinciri ÜCRETSİZ** (SPY 13.160 kontrat, 10.633'ünde IV+greek);
  **açık pozisyon (OI)** trading API kontratlarında **VAR** (SPY 91/100, T+1);
  `shortable`/`easy_to_borrow` ve Benzinga haber akışı da ücretsiz.
- Yani paylı planın ölçülmüş marjinal değeri = son 15 dk penceresi. Bu sistem Madde 3 gereği
  KAPANMIŞ barla karar verir → katkı yapısal olarak sınırlıdır.

### Eklendi
- **`nihai/ai/option_flow.py`** (KARAR-DIŞI) — `dealer_gamma_exposure` (strike başına $GEX,
  net GEX, **gamma flip**, duvarlar) · `put_call_ratio` · `max_pain` · `unusual_activity`
  (hacim>OI) · `dealer_vanna_charm` · `bs_gamma_vec` (`quant_math.bs_greeks` ile BİT-DENK).
  Flip kestirme kümülatif toplamla değil, net GEX spot ızgarasında YENİDEN fiyatlanarak
  bulunur (gamma spot'a bağlıdır). Dealer konvansiyonu her çıktıda adıyla taşınır.
- **`microstructure`** += `relative_volume` (RVOL, AYNI gün-içi dilimin medyanına göre) ·
  `trade_size_profile` (hacim/işlem sayısı + z + balina) · `vwap_deviation` · `turnover_rate`.
- **`nihai/data/data_capabilities.py`** — `python -m nihai.data.data_capabilities`
  yetenekleri O AN ölçer ve kapalı olanın NEYLE açılacağını yazar (anahtar DEĞERİ basmaz).
- **`/api/quant/flow` + pano `Modeller → Opsiyon Akışı` paneli** (11. quant modeli): 4 grafik,
  ~15 ölçüm kartı; ölçülemeyen blok `null` → panoda BOŞ kalır.
- **`tests/test_tur57_akis_ve_veri.py`** — 40 test (matematik pariteleri, ölçüm-yokluğu
  sözleşmesi, veri köprüleri, SHORT yolu kaynak denetimi, pano bağlantısı).

### Değişti
- **`_normalize_alpaca_bars` `trade_count` ve `vwap` kolonlarını artık DÜŞÜRMÜYOR.** Veri
  yıllardır geliyordu; normalize adımı atıyordu ve üç ölçü bu yüzden "ölçülemez" sanılıyordu.
  Eski test tam da düşürmeyi kilitliyordu (kusuru test koruyordu).
- **`main.py` SHORT dalı borç bayrağını okuyor** (Madde 30'un eksik yarısı): varsayılan
  İŞARETLER (`NOT_SHORTABLE`/`HARD_TO_BORROW`), `SHORT_BORROW_GUARD=block` kapıya çevirir.
  Bayrak okunamazsa hiçbir şey yapılmaz ("bilinmiyor" bir RED gerekçesi değildir).
- **Yeni opt-in anahtarlar** (üçü de varsayılan davranışı DEĞİŞTİRMEZ): `ALPACA_BARS_FEED`
  (bar feed'ini ayrı seç — SIP uzatılmış seansı da getirir, önce A/B ölç) · `ALPACA_OPTIONS`
  (yfinance zincirinin üstüne OPRA NBBO/IV/greek işle) · `SHORT_BORROW_GUARD`.
- **`data_provider.asset_flags`** — bayrak okuyucusu TEK kaynak; sarmalayıcı zincirini gezer,
  bulamazsa Alpaca trading API'sine düşer.

### Düzeltildi
- `test_tur56` PCA toleransı serileştirme yuvarlamasını hesaba katıyor (1e-6 eşiği matematiği
  değil `_f(v, 4)` hassasiyetini sınıyordu; ölçülen sapma 1,0001).

## [2.15.0] — Quant modellerinin görsel yüzeyi (Tur-56, 2026-08-29) — SALT-OKUNUR/KARAR-DIŞI

Motorda çalışan quant-finans hesaplayıcıları ve risk modelleyicileri ilk kez panoda
GÖRÜNÜR. Yeni matematik yok; mevcut fonksiyonların çıktısı grafiğe hazır hâle getirildi.
Rapor: `docs/TUR56_RAPORU_2026-08-29.md`.

### Eklendi
- **`nihai/web/api_quant.py`** — 11 salt-okunur uç (`/api/quant/{catalog,risk,vol,regime,
  mc,options,smile,portfolio,micro,execution,kelly}`). `quant_math` · `options_math` ·
  `advanced_math` · `microstructure` · `portfolio` · `regime_hmm` ve `nihai/risk/*`
  fonksiyonlarını çağırır, seri/ızgara döndürür. Hepsi fail-soft (`{"ok": false, "reason"}`).
- **Pano `Modeller → Quant Modelleri` sekmesi** — 10 model paneli, **33 grafik + 84 ölçüm
  kartı**: getiri dağılımı + VaR/CVaR/EVT kesimleri · su-altı eğrisi · 5 vol tahmincisi +
  vol konisi + GARCH(1,1) koşullu yol/öngörü · HMM rejim olasılık şeridi + geçiş matrisi +
  BOCPD kırılmaları · Monte Carlo yelpazesi + terminal dağılım · Black-Scholes fiyat/ödeme
  ve 11 Greek eğrisi · SVI gülümseme + Durrleman g + risk-nötr yoğunluk + vade yapısı ·
  7 tahsis yöntemi × ağırlık/risk katkısı + etkin sınır + korelasyon ısı haritası + PCA ·
  mikroyapı imza/VPIN/HAR-RV/sıçrama · Almgren-Chriss yörüngesi · Kelly büyüme eğrisi.
  Tembel yüklenir; 5 sn'lik ana döngüye maliyet EKLEMEZ.
- **`tests/test_tur56_quant_modelleri.py`** — 50 test: fail-soft sözleşmesi, grafik
  değişmezleri (CVaR ≥ VaR, HMM olasılıkları 1'e toplanıyor, MC seed determinizmi,
  risk katkıları 1'e toplanıyor, yörünge monoton azalan), pano bağlantısı ve
  **68 formül referansının** gerçek fonksiyonlara işaret ettiği.

### Düzeltildi
- **BULGU-56-1** — `_nums()` serileştiricisi `None` alanları düşürünce `roll_spread` /
  `corwin_schultz` "ölçülemedi" mesajı sessizce kayboluyordu. `None` korunur, `bool`
  bool kalır.
- `quant_smile` içinde `except ... as e` sonrası döngü değişkeni gölgeleme (`mypy`).

### Değişmedi
Sinyal, eşik, META_GATE, boyutlama, execution. Bu turun hiçbir çıktısı karar yoluna
bağlanmaz — kapasite/görünürlük artışı bir KENAR iddiası DEĞİLDİR (ACI DERS #5).

## [2.14.0] — Lead-lag aday ailesi ölçüldü (Tur-23, 2026-08-03) — ARAŞTIRMA/KARAR-DIŞI

edge_lab BACKLOG #3 koşuldu. **Verdikt: kenar YOK** — 5 kapıdan 2'si FAIL.
Kayıt: `docs/EDGE_FINDINGS.md §26` · `research/edge_lab/SONUC_leadlag_pilot_2026-08-03.md`.

### Eklendi
- **`research/edge_lab/build_leadlag_dataset.py`** — sektör-içi lead-lag aday veri seti
  (Alpaca SIP saatlik, 2y, 28 sembol / 7 sektör → 110.704 satır, taban 0,4414).
  Lider **hindsight'sız**: her barda sektörün son 20 barlık medyan dolar-hacmi en yüksek
  üyesi (geriye-bakan). Aynı-gün kısıtı gece boşluğunu etiketten çıkarır (BACKLOG #1
  hipoteziyle karışmasın diye). **+11 test** (`tests/test_edge_leadlag_dataset.py`),
  ağırlık merkezi leak-free ve hindsight-yok kanıtı.
- `research/edge_lab/BACKLOG.md` — **denenen-aday sayacı** (aileler arası çoklu-test
  muhasebesi; şu an 2, ikisi de FAIL).

### Ölçüldü (kenar YOK)
- purged_cv OOS-AUC 0,532 (kıl payı) · gate_time/gate_symbol PASS **ama yalnız 29/33.212
  işlem üstünde (%0,087)** · **DSR 0,3058** (kapı 0,95) · **exec_lag_1: 1 bar gecikmede
  kapıyı geçen işlem 29→3** = ACI-7 zamanlama artefaktı.

### Değişti (kural, kod değil)
- `CLAUDE.md` ACI DERS #7 → "**§22 · §23 · §26 — ÜÇ KEZ doğrulandı, artık VARSAYILAN ŞÜPHE**"
  + yeni okuma kuralı: *kapı PASS dese bile `n_kept` okunmalı*.
- Harness'ın `_MIN_KEPT=5` tabanı **bilerek değiştirilmedi** — bir aday FAIL ettikten SONRA
  kanıt kapısını sıkılaştırmak sonuca göre kural değiştirmek olur (öneri SONUC §6'da).

## [2.13.0] — AI v4 "Kapasite ve Kalibrasyon" (Tur-22, 2026-08-01) — KARAR-DIŞI/GÖLGE

Bu tur da **kenar peşinde değil**. AI katmanı her eksende genişletildi: kapasite, mimari,
eğitim reçetesi, kalibrasyon, belirsizlik, dağılım-dışı savunma ve ölçüm dürüstlüğü.
Anayasa değişmedi — hiçbir çıktı sinyale/eşiğe/META_GATE'e bağlanmaz.

### Kapasite (`nihai/ai/neural.py`)
- **Yeni katman `quant_transformer_xl` — ≈30,4M parametre** (v3'ün 5,6 katı; f=96'da 31,76M).
  12 pre-norm blok, d=384, 6 başlık, **SwiGLU** FFN (1536), **RMSNorm**, **LayerScale**
  (γ₀=1e-4 → derin yığın kimlik-yakını başlar), **doğrusal artan stokastik derinlik** (0→0,1),
  **PLR periyodik sayısal gömme** (Gorishniy ve ark. 2022). Dizi düzeni ([CLS]+3×[SCALE]+
  özellikler) ve `nn.MultiheadAttention` KORUNDU → `explain.attention_rollout` çalışmaya
  devam eder. `QuantTransformer` (v3) hiç DOKUNULMADI: kayıtlı canlı artefaktlar bit-denk.
- **`NeuralEnsemble`** — derin topluluk (Lakshminarayanan 2017): K üye × farklı seed,
  `param_count` üye toplamı (5 üyeli XL = 158,8M), belirsizlik **toplam varyans yasasıyla**
  aleatorik (üye-içi MC-dropout) + epistemik (üye-arası yayılım) olarak ayrışır.
  `member_spread()` epistemik bileşeni tek başına verir. `load_any()` iki artefakt biçimini
  de ayırt eder. Merdiven: 26 → 30.363.883 parametre (f=25).

### Eğitim reçetesi — hepsi OPT-IN, varsayılan v3 ile BİT-DENK
`fit()`: `scheduler="cosine"` (ısınmalı, adım-başı), `ema_decay`, `class_weight="balanced"`
(YALNIZ eğitim diliminin oranından — sızıntı yok), `label_smoothing`,
`early_stop_metric="val_auc"`, `calibrate=True` (fit sonrası val diliminde **sıcaklık
ölçekleme**, Guo ve ark. 2017; deterministik 1-B arama — optimizer değil), `device="auto"`.
Çıkarım artık toplu-iş bölerek yapılır (büyük X'te bellek disiplini).

### Kalibrasyon ve dürüstlük
- `neural.evaluation_metrics()` — AUC (beraberlik-doğru Mann-Whitney) + Brier + **ECE** +
  isabet + taban oranı, saf numpy.
- `learning.expected_calibration_error()` — `reliability_bins` ile AYNI kovalardan; state'te
  `pwin_calibration.ece` ve `sources.{prior,pwin,neural}.ece`; rapora "kaynak karnesi" satırı.
  Gerekçe: Brier kalibrasyon+ayırt etme karışımıdır, ECE yalnız kalibrasyonu ölçer.
- `research/train_neural.py`: yeni bayraklar (`--ensemble/--scheduler/--ema/--class-weight/
  --label-smoothing/--early-stop-metric/--calibrate/--device`), rapora Brier+ECE+kapasite,
  model kartına reçete + topluluk boyutu + doğru artefakt dosya listesi.

### Dağılım-dışı (OOD) savunması
- `NeuralCapacity.ood_zmax()` + fit'te ölçülen `train_zmax_q99` artefaktla taşınır.
- `NEURAL_SHADOW_MAX_Z=auto` → eşik modelin KENDİ eğitim dağılımından (q99×1,5, en az 4,0);
  artefakt eskiyse sessizce 6,0'a düşer (dürüst degrade). `neural_shadow` `load_any` ile
  topluluk artefaktlarını da yükler; `status()`'a `ensemble`/`max_z_mode`/`train_zmax_q99`.

### ÖLÇÜM (canlı 1m seti, 5.996 satır, purged 5-fold, AYNI fold'lar)
| | OOS-AUC | Brier | ECE | DSR | parametre |
|---|---|---|---|---|---|
| v3 taban | 0,525 ± 0,013 | 0,2491 | 0,0377 | 0,0006 | 186.702 |
| v4 reçetesi | 0,537 ± 0,012 | 0,2489 | 0,0312 | 0,1805 | 186.702 |
| v4 + 5 üyeli topluluk | **0,541 ± 0,010** | 0,2489 | **0,0280** | 0,3259 | **933.510** |

Üç eksende de (ayırt etme, kalibrasyon, DSR) her adımda tutarlı iyileşme; reçete adımında
5/5 fold'da AUC arttı, ECE toplamda %26 düştü, fold-arası sapma daraldı (±0,013→±0,010).
**KENAR DEĞİLDİR:** AUC hâlâ yazı-turaya yakın ve DSR 0,33 hiçbir kanıt kapısını geçmez
(kapı ≈0,95). Kazanılan şey ayırt etme değil kalibrasyon ve reçete disiplinidir
(ACI DERS #3). Kapasite artışı bir kenar iddiası DEĞİLDİR (ACI DERS #5: darboğaz veri,
mimari değil).

### Test
1900 → **1963 test yeşil**; ruff temiz; mypy 144 dosyada temiz.

## [2.8.0] — AI v2 "Quant Beyin": finans-özelleşmiş bilişsel katman (KARAR-DIŞI/GÖLGE)

Bu tur da **kenar peşinde değil** — Tur-13 AI çekirdeği (router/memory/reasoner), borsa/quant
alanına özelleşmiş tam bir bilişsel katmana genişletildi. ANAYASA korunuyor: hiçbir modül
sinyal/eşik/META_GATE'e bağlanmaz; karar motoru yalnız `off|shadow` kiplidir (enforce YOK).
Spec: `docs/AI_V2_BEYIN_SPEC.md` · Rapor: `docs/AI_V2_BEYIN_RAPORU_2026-07-19.md`.

### Yeni modüller (nihai/ai)
- **`quant_math`** — derin finans matematiği: 4 OHLC vol tahmincisi (Parkinson/GK/YZ/EWMA),
  VaR/CVaR/Cornish-Fisher, Kelly (binary/continuous/fractional), OU yarı-ömrü, Hurst,
  varyans-oranı, Engle-Granger + Kalman dinamik beta, Ledoit-Wolf/EWMA kovaryans, PCA riski,
  Monte Carlo (GBM + blok bootstrap), Black-Scholes fiyat+greekler, kural-tabanlı rejim
  sınıflandırıcı (`classify_regime`). Tümü NaN-savunmacı, seed'li, bilinen-değer testli.
- **`models`** — parametre KAPASİTE merdiveni: logistic (26) → elasticnet (326) →
  gbm_small (~2.6k düğüm) → mlp (3.777 ağırlık) [25 özellik için gerçek sayımlar].
  `recommend_tier` örnek/param oranı disiplini; promotion kanıt-kapılı manuel yolda
  (`live_retrain --promote`) kalır — "kapasite hazır" ≠ "kullanılmalı" (ACI DERS #2/#5).
- **`episodic`** — epizodik hafıza (z-normalize benzer-durum geri çağırma, JSONL append-only,
  çözüm-birleştirme) + **LESSONS**: ACI DERSLER #1-#7 + 6 operasyonel dersin makine-okur
  kataloğu; `check_against_lessons` öneri metnini derslerle çarpıştırır (uyarı, ceza değil).
- **`learning`** — çözülen canlı tahminlerden öğrenme: Beta-Binomial posterior (GLOBAL/LONG/
  SHORT; canlı logda strateji kolonu yok — OP-6), pwin Brier + güvenilirlik kovaları,
  yuvarlanan-isabet Page-Hinkley drift. Durum `ai_learning_state.json` (atomik).
- **`knowledge`** — 45 kavramlık quant/finans bilgi çekirdeği (tanım+formül köprüsü+tuzaklar+
  ders bağları; tüm `formul_ref`'ler quant_math'te getattr-doğrulanır).
- **`decision`** — GÖLGE Bayes karar motoru: prior(öğrenme) × p_win harmanı × rejim uyumu →
  LONG_BIAS/SHORT_BIAS/ABSTAIN + şeffaf gerekçe zinciri + fractional-Kelly önerisi.
  **Yön için prior CI tabanı > 0.5 şart** (belirsiz/aleyhte kanıtta ABSTAIN; ters-yön kenarı
  türetilmez). `AI_DECISION_MODE=off` (vars.) hiçbir dosyaya yazmaz; `shadow` → JSONL + epizot.
- **`brain`** — düşünme döngüsü orkestratörü: algıla→hatırla→topraklama→hesapla→karar→kaydet;
  `python -m nihai.ai.brain --demo` (yazmasız uçtan uca) / `--brief` (günlük beyin brifi).
- **`daily_summary --brain`** — opsiyonel Quant Beyin bölümü (öğrenme+gölge kararlar;
  bayraksız davranış birebir aynı, fail-soft).

### İlk gerçek-veri koşusu (dürüstlük kanıtı)
Son 5000 çözülmüş tahminden: GLOBAL isabet-posterior **0.458** (%95 CI 0.444–0.472, 0.5'in
altında — bilinen NO_EDGE ile tutarlı), pwin Brier **0.262** (bilgisizden kötü — meta modelin
kenarsızlığını bağımsız doğruluyor), drift uyarısı aktif. Gölge motor bu kanıtla doğru şekilde
**ABSTAIN** üretiyor. Testler 736 → **938** (AI katmanı 17 → 219); ruff + mypy(nihai) temiz.

Bu tur da **kenar peşinde değil** (§1/§6/§16: yeni fiyat-stratejisi eklemek dayanıklı kenar
getirmez). Operasyonel hata/tutarsızlık giderildi + araştırma disiplinindeki tek olumlu
mekanizma (§16.2 mutabakat) canlı moda + ölçüm harness'ına taşındı.

### Düzeltmeler
- **Log şişmesi (601MB → 132MB):** `hub_server.err.log` 495MB'a ulaşmıştı. Launcher
  redirect logları (`hub_server.{out,err}.log`, `_server_restart.log`) rotasyonsuzdu →
  `signal_engine.prune_logs()`'a boyut-tabanlı `_truncate_big()` + `system.*.log` arşiv
  kırpma; `hub-watchdog.ps1`'e her döngüde boyut-koruması (`-MaxLogMB`, vars. 50).
- **Walk-forward look-ahead/gürültü hatası:** `walk_forward` strateji-seçimi tüm
  `STRATEGY_MODES`'u aday alıyordu — `pead/insider/sentiment` (çalışma-anı harici veri →
  look-ahead) + `ensemble` (meta-mod) dahil. Yeni `BASE_STRATEGY_MODES` (8 fiyat-only mod)
  ile WF/oto-tarama yalnız backtest-güvenli adaylar arasından seçer.
- **Meta-kapısında sessiz train/serve skew:** `meta_gate` modelle kaydedilen eğitim-anı
  feature listesini (`self._features`) okuyordu ama çıkarım vektörünü global `FEATURE_NAMES`
  ile kuruyordu. Kod `FEATURE_NAMES`'i yeniden sıralarsa (aynı uzunluk) sklearn hata vermez →
  özellikler yanlış sırada → **sessizce yanlış P(win)** → sinyaller yanlış elenir; `explain`
  kartında da değer↔etiket kayması. `features_to_vector(feats, names=…)` eklenip `probability`/
  `explain` artık `self._features` geçiyor (model-anı sırası otoriter). 3 yeni birim testi.
- **`signal_quality_score` hacim KeyError koruması:** `detect_vol_breakout` `volume` yokluğuna
  savunmacıyken kalite skoru korumasız `df["volume"]` erişiyordu → hacimsiz df'de çökerdi.
  `if "volume" in df.columns` kapısı (volume varken davranış birebir aynı). 1 regresyon testi.
- **Doküman tutarlılığı:** README test sayısı 68→241; `.env.example`'a `CONFLUENCE_MIN_VOTES`
  + tam strateji listesi; junk dosya temizliği (`bağlantı oluştur.md`, `state.json.bak-*`).

### Yeni sinyal modu — `confluence` (katı mutabakat, §16.2)
≥`CONFLUENCE_MIN_VOTES` (vars. 2) bağımsız detektör hemfikir değilse WAIT — tek-detektör
gürültü tetikleri elenir. `_detect_ensemble(min_votes=…)` + config knob + dispatcher.
**KENAR İDDİASI DEĞİL** — false-positive azaltma kaldıracı; survivorship-temiz ölçüm için
`research/confluence_test.py` (BASKET 2-periyot + `--pit` survivorship: TÜM/hayatta-kalan/
removed-since, PEAD harness deseni). 3 yeni birim testi. Varsayılan davranışlar değişmedi.

## [2.7.0] — Altyapı + dürüstlük: log retention, canlı doğrulama, paket reorganizasyonu

Bu tur **kenar peşinde değil**; sistemi temiz/dürüst/doğrulanabilir/bakılabilir hale getirdi.
Kenar-ilgili tek bulgu (canlı forecast güveni ~%50 = öngörü gücü yok): `docs/EDGE_FINDINGS.md §15`.

### Altyapı
- **Log retention + rotasyon** (`signal_engine.prune_logs`): rotasyon arşivleri + günlük karar/tarama
  dosyaları sınırsız birikiyordu (`logs/` 580MB). Başlangıçta otomatik retention; trade arşivi her
  rotasyonda kırpılır. `LOG_KEEP_TRADE_ARCHIVES` / `LOG_KEEP_DECISION_LOGS` / `LOG_KEEP_SCREEN_CSV`
  env knob'ları (`.env.example`). Tek-seferlik temizlik: 580MB → 280MB.
- **Paket reorganizasyonu**: 78 düz kök modül → `nihai/{data,signals,risk,execution,web}` (canlı sistem)
  + `research/` (offline deney/araç). `bench.BASKET` → `nihai/universe.py` (tek kaynak; production da
  kullanıyor). Tüm import'lar dotted-path'e, `.ps1` launcher'lar `-m` modül formuna, ~52 doc repro
  komutu (README/CHANGELOG/EDGE_FINDINGS) güncellendi. `config.ROOT` + web asset (html/`.env`) yolları
  repo köküne sabitlendi. **Davranış değişmedi** (saf yapısal; 185 test yeşil, ruff temiz).
- README'ye "Dizin Yapısı" + modül-formu çalıştırma bölümü. `web_dashboard` B904 exception-chaining düzeltmesi.

### Canlı forecast: dürüstlük + doğrulama (sinyal-only ürün uyumu)
- **Dürüst başlık metriği**: pano "Ort. Yön Güveni" (ham, ≈yazı-tura, yanıltıcı) yerine **Ort. Kalibre
  Olasılık** gösteriyor (`avg_prob_cal`); ~%50'de gri = "öngörü gücü yok" (uydurma kenar görüntüsü yok).
- **Kapalı-döngü doğrulama** (`prediction_tracker.py`, opt-in `PREDICT_TRACK`): her bar forecast'ini
  kaydet → N bar sonra gerçekleşen yönle etiketle (look-ahead'siz) → canlı isabet + kalibrasyon raporla.
  Backtest survivorship'ine bağımlı olmayan gerçek OOS kanıtı (paper-trade eşdeğeri). `PREDICT_TRACK`,
  `PREDICT_TRACK_HORIZON_BARS` env (`.env.example`). 5 test (`tests/test_prediction_tracker.py`).
- **Kalibrasyon ölçümü** (`calibrate_forecast.py`, 40.240 örnek): güven %9-98 arası TÜM kovalarda
  gerçekleşen isabet düz ~%50 (genel %49.7) → forecast güveni geleceği öngörmüyor. §6'nın canlı-heuristik
  doğrulaması (`EDGE_FINDINGS §15`).

### Ampirik sonuç (ship değil)
- **Yeni kenar iddiası YOK.** Aksine §6 "fiyat-göstergesi kenarsız" canlı heuristikte doğrudan doğrulandı.
- Ship edilen: dürüst UI + doğrulama altyapısı + temiz paket yapısı. Asıl darboğaz değişmedi:
  survivorship-temiz PIT veri (Sharadar/CRSP) + biriken canlı paper kanıtı (`EDGE_FINDINGS §13.5/§15`).

## [2.6.0] — AI_trade'den çıkış/risk overlay'leri (kuruldu, ölçüldü, default-KAPALI)

Yan proje `AI_trade` (Binance botu) incelendi; varlık-bağımsız risk/çıkış fikirleri ana projeye
uyarlandı (kripto birimlerinden hisse **R/ATR-uzayına** taşınarak). Hepsi **env-kapılı, default-KAPALI**
ve `bench.py`/`exit_grid.py`/`wf_pullback.py` ile ölçülebilir. Tam analiz: `docs/EDGE_FINDINGS.md §14`.

### Yeni yetenek (opt-in, varsayılan davranış DEĞİŞMEDİ)
- **A1 — Kademeli trailing + kâr-tabanı kilidi** (`position_manager._apply_trailing`): kâr (R) arttıkça
  trail mesafesini kademeli SIKILAŞTIR + kâr eşiği aşılınca tabanı kilitle. `TRAIL_TIERED`, `TRAIL_TIERS`,
  `TRAIL_LOCK_TRIGGER_R`, `TRAIL_LOCK_FLOOR_R`. Kapalıyken eski düz `TRAIL_R` davranışı byte-aynı.
- **A2 — MFE give-back** (`check_exit`): tepe lehte hareket `MFE_GIVEBACK_ARM_R`'yi aştıktan sonra
  tepeden `MFE_GIVEBACK_R` kadar R geri verilince çık. **R-tabanlı** (AI_trade'in %-eşiği yerine →
  volatiliteye uyarlı, breakeven/trailing ile tutarlı).
- **A3 — Kill-switch otomatik reset** (`state.record_trade_pnl`): rolling expectancy `KILL_SWITCH_RESET_EXPECTANCY`'ye
  toparlayınca kilit kendiliğinden kalkar (eskiden yalnızca açılıyordu). `KILL_SWITCH_MIN_TRADES` eklendi.
- **A4 — Piyasa rejimi → risk çarpanı** (`market_regime.py`): benchmark GÜNLÜK trendi (SMA200) ± VIX →
  DEFENSIVE/NEUTRAL/BULLISH → ×0.5/0.8/1.0, drawdown freniyle çarpımsal istiflenir. `REGIME_RISK`.
  **Backtest'te de uygulanır** (`regime_multiplier_series`, 1-bar gecikmeli, look-ahead-güvenli) → A/B'lenebilir.
- **A5 — Düşük-güven kademeli boyutlama**: az backtest geçmişli sembol elenmek yerine risk × `CONFIDENCE_LOW_MULT`
  alır; risk_mult watchlist meta'sında taşınır. `CONFIDENCE_SIZING`, `CONFIDENCE_COMFORT_TRADES`.
- **A6 — Exit-grid trailing taraması** (`exit_grid.py --trailing`): trailing varyantlarını iki periyotta OOS tarar.
- **B7/B8 — Hibrit tarayıcı skoru + RS-vs-SPY** (`setup_score.py`): sıralama `z(backtest)+w·z(setup_score)`;
  görel güç İŞLEM-GÜNÜ bazlı (`SCREEN_RS_SHORT/LONG_DAYS`, runup'tan ayrı). `SCREEN_HYBRID`.
- **Karar-desteği**: canlı sinyal satırında rejim + düşük-güven bağlamı gösterilir (sinyal-only ürün uyumu).
- `.env.example`: ~30 yeni knob belgelendi (hepsi "önce A/B yap" uyarısıyla); `validate_config` sanity kontrolleri.

### Düzeltilen gerçek davranış hatası
- **A4 backtest sessiz no-op (tz/asof)**: yfinance intraday indeksi tz-aware, günlük benchmark tz-naive →
  `series.asof()` exception atıp yutuluyor, rejim backtest'te HİÇ uygulanmıyordu (özelliğin tüm amacı boşa).
  `_regime_mult_at` tz/normalize-güvenli arama + indeks normalizasyonu ile düzeltildi (+3 regresyon testi).
- **3 önceden-kırık test düzeltildi** (üretim bug'ı değil, test eksiği): auto_screener (process-havuzu lambda
  mock'u taşımaz → `WORKERS=1` pin); 2× net-directional (tunable default'a bağlıydı → bağlayıcı cap pin).

### Ampirik sonuç (ship değil)
- **Hiçbiri alfa olarak ship edilmedi.** In-sample iyileşmeler (özellikle çıkış paketi) **walk-forward'da
  ÇÜRÜDÜ** (event_breakout OOS: pozitif-sembol %37.5 → değişmedi, beklenti daha negatif). Overlay'ler
  **kayıp-azaltıcı, alfa değil**: zarar eden temelde iyileştiriyor, kazanan trend temelinde rejim zararlı.
  Asıl darboğaz: OOS kenarı olan baz sinyal yok. (Detay + tekrar-üretim: `EDGE_FINDINGS.md §14`.)

### Test
- +39 yeni test (138 → **180 geçti, 0 başarısız** — ilk kez tamamen yeşil suite). ruff 0 (dokunulan dosyalar).

## [2.5.6] — Walk-forward optimizasyon (overfitting teşhisi)

### Yeni yetenek
- **`walk_forward()` + `--walk-forward` CLI**: Yuvarlanan in-sample (IS) / out-of-sample (OOS)
  pencerelerinde strateji seçimini optimize eder. Her fold: önceki `is_bars` barda 7 adayı
  backtest'leyip metriğe (`expectancy`/`sharpe`/`return`) göre en iyiyi seçer, sonra takip
  eden `oos_bars` barda yalnızca o stratejiyi çalıştırır. OOS sonuçları compound equity ile
  birleştirilir. Look-ahead-safe: OOS her zaman IS'ten sonraki penceredir, OOS warmup'ı IS kuyruğu.
- **Walk-forward verimliliği** (`wf_efficiency = mean(OOS metrik)/mean(IS metrik)`): IS-seçim
  metriği OOS'ta tutmuyorsa <1 → parametre/strateji seçimi **overfit** (Madde 51/52). Rapor
  ROBUST/ZAYIF/OVERFIT verdict'i + fold-fold tablo (seçilen strateji, IS vs OOS metrik) verir.
- **CLI**: `python -m nihai.execution.backtest --symbol AAPL --walk-forward --is-bars 180 --oos-bars 60 --wf-metric sharpe`
- **Refactor**: `_periods_per_year()` ortak helper (Sharpe + realized-vol annualizasyonu).

### Test
- 3 yeni test (toplam 142→145): fold-başı seçim + OOS-IS'ten-sonra (look-ahead-safe),
  verimlilik sonlu, kısa IS penceresi reddi. mypy 0 · ruff 0.

## [2.5.5] — Sentetik opsiyon backtest: dinamik IV

### İyileştirme
- **Sabit IV → gerçekleşen-vol türevli dinamik IV**: Opsiyon backtest'i (`backtest_option_synth`)
  artık IV'yi dayanağın trailing gerçekleşen volatilitesinden türetiyor (`_realized_iv`:
  log-getiri std × √ppy × VRP, [floor, ceiling] kırpma) ve **her bar güncelliyor**. Eski
  sabit %30 IV, long opsiyonun en büyük gerçek riskini — **IV crush / vega P&L**'ini — hiç
  göstermiyordu: sakinde girilen opsiyon vol patlamasından kazanır, tepe IV'de girilen
  mean-reversion'la crush yaşar. Giriş IV'si karar barına kadarki veriden (look-ahead-safe),
  yeniden-fiyatlama IV'si o anki bara kadar.
- **Config** (`BacktestConfig`): `option_bt_dynamic_iv=True` (False → eski sabit IV),
  `option_bt_iv_lookback=20`, `option_bt_iv_risk_premium=1.15`, `option_bt_iv_floor=0.10`,
  `option_bt_iv_ceiling=1.50`.
- **Refactor**: Sharpe ve realized-vol annualizasyonu tek `_periods_per_year()` helper'ını paylaşır.
- **Hâlâ yaklaşık**: tek IV tüm strike'lara (smile/skew yok); tarihsel opsiyon zinciri hâlâ
  Polygon-bağımlı (gerçek bid/ask/OI için). Bu, zinciri olmadan yapılabilecek en gerçekçi adım.

### Test
- 4 yeni test (toplam 138→142): IV kapalıyken sabit, IV vol-rejimini izliyor + sınırlar içinde,
  aşırı volde ceiling kırpması, dinamik vs sabit equity ayrışması. mypy 0 · ruff 0.

## [2.5.4] — Cap-aware boyutlama (küçük hesap işlem-açma düzeltmesi)

### Düzeltilen gerçek davranış hatası
- **Küçük hesapta yüksek-fiyatlı sembol → hiç işlem açılmıyordu**: risk-bazlı boyutlama
  (Madde 2) dar stop + yüksek fiyatla *küçük risk ama devasa nominal* üretiyordu. Örn.
  $25k hesap, NVDA $900, 1.5 ATR stop → risk $375 ama nominal 37×$900 = $33,300 (equity'nin
  %133'ü). Tek pozisyon net-yönsel tavanı (Madde 14, 1.0×equity) anında doyuruyor; eski kod
  bunu **reddediyordu** → sistem sinyal üretip hepsini `NET_DIRECTIONAL_CAP`/
  `INSUFFICIENT_BUYING_POWER` ile pas geçiyordu (config değil, boyutlama mantığı sorunu).
- **Çözüm — cap-aware downsizing**: `RiskManager._magnitude_cap_units` büyüklük tavanlarının
  (buying power / net-yönsel / brüt risk) izin verdiği maksimum birimi hesaplar;
  `_apply_cap_aware` risk-hedefini bu tavana **sığacak şekilde küçültür** (reddetmek yerine).
  Küçültme her zaman hedeften DÜŞÜK risk demektir → güvenli yön. Sayı tavanları (max_open /
  sektör / sembol kümelenmesi) küçültülemez, hard-reject olarak kalır. NVDA artık 37→27 adetle
  AÇILIR. `sizing` dict'ine `downsized` + `target_shares/target_contracts` eklendi.
- **Config knob'ları** (`PortfolioCaps`): `cap_aware_sizing=True` (False → eski reddet davranışı),
  `min_position_notional=0.0` (>0 → küçültme sonrası toz pozisyonları atla).

### Test
- 4 yeni test (toplam 134→138): downsizing onayı + net-tavan aşılmaması, cap_aware kapalıyken
  eski reddet, orijinal yüksek-fiyat/küçük-hesap senaryosunun artık açması, min-pozisyon toz filtresi.
- mypy 0 · ruff 0 · 138 test geçti.

## [2.5.3] — Veri kaynağı gözlemlenebilirliği

- **"Alpaca'dan mı geliyor?" göstergesi**: `FailoverProvider` artık her sağlayıcının kaç
  kez servis ettiğini sayıyor (`source_stats()`: active/primary_hits/secondary_hits).
  Web panosunda yeni "Veri Kaynağı" satırı (🟢 alpaca / 🟡 yfinance + yedek sayısı) ve
  `/api/state` `data_source` alanı. Operatör artık canlının Alpaca'dan mı yoksa failover'la
  yfinance'den mi beslendiğini anlık görür. +2 test (toplam 132→134).

## [2.5.2] — Statik analiz temizliği (mypy 0, ruff 0)

Harici statik analiz (Gemini) bulgularının tek tek doğrulanıp giderilmesi.

### Gerçek düzeltmeler
- **`option_type.upper()` defansif guard** (terminal_dashboard + signal_engine): hisse/
  None option_type'ta `(p.option_type or '?').upper()` — AttributeError'a karşı.
- **mypy tip uyumsuzlukları (13 → 0)**: `load_or_init` dönüş garantisi (assert),
  `detect_setup(mode: str | None)` implicit-Optional, `os.getenv(...) or default` zinciri
  (data_provider/alpaca — runtime'da çökmüyordu ama mypy doğru flag'liyordu), Alpaca
  `feed` str → `DataFeed` enum'a çevrildi (`_feed_enum`; pydantic zaten coerce ediyordu),
  `opt_type: OptionType` Literal, `load_bars` interval cast, stdout.reconfigure type:ignore.
- **ruff lint (99 → 0)**: import sıralaması, `Optional[X]` → `X | None`, kullanılmayan
  importlar (tv_provider Literal dahil), gereksiz f-string'ler, noktalı-virgül çoklu ifadeler,
  ölü değişken (`bar_now`). `E741` (OHLC `l`=low) ve `UP042` (kasıtlı str-enum) bilinçli
  ignore'landı (pyproject).

### Doğrulama notu (Gemini bulgularının değerlendirmesi)
- **Doğru**: 13 mypy + 99 ruff bulgusu gerçekti, giderildi.
- **Şiddet abartılı**: `os.getenv().lower()` ve Alpaca `feed` "crash riski" değildi —
  default + pydantic coercion runtime'da güvenli (empirik canlı doğrulandı). Yine de mypy
  flag'leri gerçekti, tip-doğru hale getirildi.
- **Bilinen takaslar**: yfinance rate-limit (cache+throttle ile azaltılmış) ve iki-katman
  veri tazeliği (sinyal-only'de gerçek fill yok → milisaniye kayması etkisiz) — kasıtlı.

### Sonuç
mypy: 28 dosyada 0 hata · ruff: temiz · 132 test geçti · Alpaca canlı veri doğrulandı.

## [2.5.1] — Günlük sayaç deadlock'u (canlı çalıştırmada bulundu)

### Düzeltilen gerçek hatalar (sistemi kalıcı kilitliyordu)
- **`consecutive_stops` cooldown DEADLOCK'u**: 3 ardışık stop → cooldown → tüm girişler
  bloke → yeni pozisyon açılmaz → kapanış olmaz → `consecutive_stops` asla sıfırlanmaz
  (yalnızca kazançlı kapanış sıfırlıyor, o da olamıyor) → **sistem kalıcı olarak HER
  sinyali reddediyordu.** Canlı `--once` çalıştırmasında yakalandı (market açık, veri
  akıyor, ama 6/6 sembol `CONSECUTIVE_STOPS_COOLDOWN` ile SKIP).
- **`daily_pnl` hiç sıfırlanmıyordu**: `startup_reconciliation`'daki "günlük PnL sıfırla"
  yorumu yanlıştı — kod PDT sayacını sıfırlıyordu, daily_pnl'i değil. Ömür-boyu birikip
  günlük-max-kayıp frenini de kalıcı tetikleyebilirdi.
- **Kök neden + çözüm**: düzgün "yeni seans günü" reset'i yoktu. `PortfolioState`'e
  `last_session_date` + `roll_session_if_new_day()` eklendi; yeni günde daily_pnl ve
  consecutive_stops sıfırlanır (startup recon + canlı döngü market-açılış geçişinde).
  Eski state'ler (alan yok) ilk açılışta otomatik reset'lenir (migration güvenli).
  Strateji-düzeyi koruma kill-switch'tir; o rolling-50 işlemle günler-arası kalıcıdır.

### Test
- 2 yeni test (toplam 130→132): gün-dönümü sayaç reset'i, deadlock'un yeni günle kırılması.

## [2.5.0] — Alpaca gerçek-zamanlı veri sağlayıcısı

### Yeni özellik
- **`alpaca_provider.py` — Alpaca Market Data** (gerçek-zamanlı; IEX feed ücretsiz).
  yfinance ~15 dk gecikmeliyken Alpaca IEX feed gerçek-zamanlı + resmi API.
  `DataProvider` arayüzünü uygular (get_bars/get_quote); opsiyon zinciri Alpaca'da
  paylı olduğundan {} yerine yfinance fallback'e bırakılır. `DATA_PROVIDER=alpaca`
  (veya iki-katmanda `FOCUSED/BROAD_PROVIDER=alpaca`) ile aktif.
- **Lazy/dayanıklı tasarım:** `alpaca-py` SDK kurulu değilse veya anahtar yoksa
  sağlayıcı pasif kalır (boş döner) → `FailoverProvider` otomatik yfinance'e geçer.
  Hiçbir durumda import/başlangıç çökmesi yok. Normalize fonksiyonları saf → SDK'sız
  test edilebilir.
- **Failover yedeği akıllandı:** artık yfinance her zaman güvenli yedek (anahtarsız +
  opsiyon zinciri verir); primary yfinance ise yedek tradingview.

### Canlı doğrulamada bulunan + düzeltilen bug'lar
- **`start` parametresi yoktu**: Alpaca `start` verilmeyince yalnızca BUGÜNÜN barlarını
  döner (~3 bar) → strateji 60+ bar gerektirir, sinyal üretmezdi. `_start_for_lookback()`
  ile aralığa göre yeterli geçmiş hesaplanıp gönderiliyor.
- **`limit` + `start` en ESKİ barları döndürüyordu**: ikisi birlikte start'tan ileri N bar
  verince son ~3 hafta kesiliyordu (bayat veri). `limit` kaldırıldı → start penceresi çekilip
  `tail(lookback)` ile en GÜNCEL N bar alınıyor (bugüne kadar).

### Test
- 11 yeni test (toplam 119→130): bar normalize (MultiIndex→sistem formatı, oluşan-bar
  dışlama, tail, tz-naive), `_start_for_lookback` geçmiş kapsama, quote dönüşümü (sentetik
  makas), kimlik-yok pasiflik (env-izole), get_provider alpaca→yfinance failover.

## [2.4.3] — Crash-recovery (state yükleme) kritik düzeltmeleri

### Düzeltilen gerçek hatalar (restart sonrası çökme)
- **State'ten yüklenen pozisyonlar bozuk tiplerle geliyordu** (`VirtualPosition(**pos)`
  tip dönüşümü yapmaz): `entry_time` bir STRING, `greeks` bir DICT olarak kalıyordu.
  Sonuçlar (Madde 41 crash-recovery senaryosunda, taşınan pozisyon kapanınca):
  - post-mortem `exit_time - entry_time` → **TypeError** (datetime − str) → canlı döngü çöker
  - Greek toplama `greeks.delta` → **AttributeError** (dict) → risk kontrolü + web /api/state çöker
  Düzgün `_deserialize_position()` eklendi (entry_time→datetime, greeks→GreekSet, ileri
  uyumluluk için bilinmeyen anahtarlar elenir).
- **Post-mortem "Pozisyon Suresi" yanlıştı**: `entry_time` yalnızca tarih (gece-yarısı)
  olarak saklanıyordu → 10:30'da açılıp 14:00'te kapanan pozisyon "14 saat" gösteriyordu.
  Artık tam datetime saklanıyor (PDT/settle mantığı .date() ile çalışmaya devam eder).

### Test
- 2 yeni test (toplam 117→119): pozisyon save/load tip-korunumu, taşınan pozisyonun
  çökme olmadan kapanması.

### İncelendi, temiz çıktı
- market_calendar (exchange_calendars tz-aware UTC doğru), event_calendar, safety,
  iv_history, gate, backtest metrikleri (Sharpe bar-aralığına göre yıllıklaştırılıyor),
  signal_engine, option_strategy.

## [2.4.2] — Risk tavanı doğruluk düzeltmeleri

### Düzeltilen gerçek hatalar (risk yönetimini etkiliyordu)
- **SHORT pozisyonlar sıfır risk gösteriyordu** (`VirtualPosition.risk_dollars`):
  short'ta stop > entry olduğundan `max(entry-stop, 0)` negatifi 0'a kırpıyordu →
  short'lar brüt-risk tavanına (`gross_risk = Σ risk_dollars`, Madde 14) hiç katkı
  vermiyor, aşırı kaldıraca izin veriyordu. `abs(entry-stop)` ile düzeltildi.
- **Net theta/vega Greek tavanları asla tetiklenmiyordu** (`PortfolioGreeks.add_position`):
  py_vollib theta/vega'yı delta/gamma gibi HİSSE-BAŞI döner; kod delta/gamma'yı ×100
  (kontrat çarpanı) ile ölçeklerken theta/vega'yı ×100'süz topluyordu → net theta/vega
  100× küçük çıkıyor, cap'ler (Madde 27: -50 $/gün theta, 300 vega — pozisyon ölçeğinde
  kalibre) hiç çalışmıyordu. py_vollib konvansiyonu empirik doğrulanıp ×100 eklendi.

### İnceleme notu
- `check_exit` içindeki pin-riski kontrolü **ulaşılamaz ölü kod**: `DTE_GUARD`
  (dte ≤ 5) her zaman önce döner, pin-riski dte ≤ 1 gerektirir → asla çalışmaz
  (yanlış davranışa yol açmıyor, dokunulmadı).

### Test
- 4 yeni test (toplam 113→117): short risk_dollars, brüt-risk shortları sayıyor,
  theta/vega ×100 ölçekleme, theta tavanının artık tetiklenmesi.

## [2.4.1] — Log temizliği & sağlayıcı çift-fetch düzeltmesi

### Düzeltilen gerçek hata
- **TradingView çift yfinance çağrısı**: `TradingViewProvider` varsayılan
  `yf_fallback=True` ile İÇ yfinance fallback'i taşıyordu; ama `get_provider` zaten
  TV'yi bir `FailoverProvider(tv, yfinance)` içinde sarıyor. TV bir exception verince
  yfinance İKİ KEZ çağrılıyordu (iç fallback + dış failover) → gereksiz istek +
  system.log'da çift `[data] empty bars` satırları. Bars failover'ı artık tek otorite
  olan `FailoverProvider`'da; iç fallback yalnızca opsiyon zinciri için kaldı.

### Log hijyeni
- Kurtarılabilir ağ hataları (failover/None ile ele alınan) artık ERROR yerine
  WARNING: `tv_provider.get_bars` bar çekme hatası, `get_quote` (`'exchangeTimezoneName'`
  gibi geçici yfinance iç hataları dahil), `get_option_chain` hataları. Gerçek
  hatalar artık logda öne çıkıyor (offline iken yüzlerce sahte ERROR yok).
- **`tvDatafeed` kütüphane gürültüsü susturuldu**: kütüphane, bizim failover'la
  ele aldığımız bağlantı/boş-veri durumlarını kendi stdlib logger'ından stderr'e
  ERROR basıyordu ("Connection timed out", "no data...") → `tvDatafeed` logger'ı
  CRITICAL'e çekildi. İnternetli çalıştırmada log tamamen temiz (0 WARNING/ERROR).
- **`AUTO_SCREEN_MAX_RUIN` varsayılanı 0.10 → 0**: MC varsayılanı 0 olduğundan ruin
  filtresi zaten atıldı; tutarsız varsayılan her açılışta gereksiz config uyarısı
  üretiyordu. Artık varsayılan config 0 hata + 0 uyarı.
- _Not:_ system.log'daki uyarıların ~%95'i geçici ağ/çevrimdışı durumdu (`getaddrinfo
  failed` = o an DNS/internet yok, TV anonim boş dönüş) — kod bug'ı değil, beklenen
  ve zaten doğru ele alınan (failover) durumlar.

### Test
- 3 yeni test (toplam 110→113): TV çift-fetch yokluğu, log seviyesi, FailoverProvider
  tekil-secondary garantisi.

## [2.4.0] — Gösterge doğruluğu + 3 yeni strateji

### Doğruluk düzeltmeleri (gerçek hata)
- **RSI/ATR/ADX yanlış yumuşatma kullanıyordu**: basit hareketli ortalama (SMA)
  yerine artık **Wilder yumuşatması (RMA)** — bu göstergelerin evrensel standardı
  (TradingView/TA-Lib varsayılanı). SMA varyantı, RSI 30/70 ve ADX 20/25 eşiklerinin
  anlamını bozuyordu (özellikle ADX değerleri belirgin farklı). `_rma()` =
  `ewm(alpha=1/period, adjust=False)` Wilder rekürsiyonunun ta kendisi; `min_periods`
  ile ısınma boşluğu NaN kalır (yarı-oluşmuş göstergeden sinyal çıkmaz). Referansa
  karşı test edildi.
- **RSI sıfır-kayıp durumu**: hiç düşüş yokken RS=∞ → NaN/inf yerine artık RSI=100.
- **Bollinger Bands**: pandas varsayılanı örnek std (ddof=1) yerine standart
  popülasyon std (ddof=0).

### Yeni stratejiler (`strategy.py`)
- **`macd`** — MACD histogram sıfır kesişimi (momentum crossover).
- **`donchian`** — önceki 20-bar kanal kırılımı, yalnızca trend rejiminde (ADX≥eşik;
  whipsaw koruması).
- **`squeeze`** — kalıcı düşük-volatilite sıkışması (son N barın hepsi lookback
  medyanının belirgin altında) sonrası kanal kırılımı; mean_reversion'ın tersi
  (band kenarını fade etmez, patlamayı kovalar). `ensemble` üçünü de içerir.
- `STRATEGY_MODES` tek-kaynak sabiti; dispatcher + config doğrulama + CLI'ler ondan
  besleniyor. `validate_config` artık `ACTIVE_STRATEGY`'yi de doğruluyor.

### Test
- 18 yeni test (toplam 92→110): gösterge doğruluğu (Wilder rekürsiyonu referansa
  karşı, RSI sınırları/ekstremleri, ATR/ADX/Bollinger), 3 strateji + dispatcher/ensemble.

## [2.3.0] — Dinamik watchlist + iki-katman paralel tarama

### Yeni özellikler
- **Dinamik watchlist (`watchlist.py`)**: config `WATCHLIST` artık yalnızca tohum.
  Asıl canlı liste `data/watchlist.json`'da, çalışma anında değişir, thread-safe.
- **Oto-tarayıcı (`auto_screener.py`)**: sistem bir evreni (Nasdaq-100 / S&P 500 /
  birleşik) periyodik backtest'leyip watchlist'i kendisi günceller. Modlar:
  top-N (en iyi N), HEPSİ (tüm evren), ve iki-katman sıralaması.
- **İki-katman paralel tarama (`dual_tier.py`)**: en iyi N "odak" katmanı
  (TradingView) + kalan tüm evren "geniş" katmanı (yfinance) AYRI thread'lerde
  paralel taranır. Partisyon örtüşmesiz, tüm evreni kapsar. Ortak portföy/state
  tek `PORTFOLIO_LOCK` ile korunur; her thread kilidi yalnızca sinyal-kabul→
  pozisyon-açma kritik bölümünde tutar (veri çekme paralel kalır).
- **Manuel watchlist kontrolü (pin/block) — üç kanal**: operatör sembol ekler
  (📌 pin → oto-tarama silemez) / çıkarır (🚫 block → oto-tarama geri ekleyemez).
  Erişim: Web API + pano yönetim kartı, **Telegram komutları** (`/add /remove
  /watchlist /status`, yetkili chat_id), ve **CLI** (`python -m nihai.execution.watchlist
  add|remove|list`). Üçü de `data/watchlist.json`'a yazar (agent-native parite).
- **İki-katman gözlemlenebilirliği**: tier sağlığı (canlı mı, tur sayısı, tur
  süresi, sembol sayısı) `/api/state` + web panosunda "İki-Katman Tarama" kartı +
  terminal panosunda pin/block/son-tarama satırları.
- **Sağlamlık**: bir tier sağlayıcısı kurulamı (örn. tvDatafeed import hatası) tüm
  uygulamayı çökertmek yerine yalnızca o katmanı atlar, diğeri çalışmaya devam eder.
- **Başlangıç config doğrulaması** (`validate_config`): yanlış yazılmış `.env`
  değerleri (örn. `AUTO_SCREEN_SORT=expectancyy`, geçersiz sağlayıcı/aralık) sessiz
  yanlış davranış yerine başlangıçta net hatayla yakalanır; uyumsuzluklar (örn.
  iki-katmanda aynı sağlayıcı, MC'siz ruin filtresi) uyarı olarak loglanır.
- **Tarama sıralaması CSV'si**: her tarama `logs/auto_screen_<ts>.csv`'ye skorlu
  sıralamayı yazar (operatör hangi sembolün neden seçildiğini görebilir).

### Düzeltilen gerçek hatalar
- **tz-aware bar bug'ı (`backtest.py`)**: yfinance/TV intraday barları tz-aware
  gelince, kapanan işlemde `exit_time - entry_time` "offset-naive vs offset-aware"
  hatası veriyordu (screen.py'yi de sessizce bozuyordu). `_strip_tz()` ile index
  yükleme anında tz-naive'e normalize ediliyor.
- **Çift kilit yarış koşulu**: `dual_tier` ayrı bir `PORTFOLIO_LOCK` oluşturuyordu;
  `concurrency.PORTFOLIO_LOCK`'u (web/position_manager kullanıyor) yeniden ihraç
  edecek şekilde birleştirildi — yoksa web flatten ile tier thread'i çakışırdı.
- **`TierScanner._stop` ad çakışması**: `threading.Thread`'in iç `_stop()` metodunu
  eziyordu → `join()` bozuluyordu. `_stop_event` olarak yeniden adlandırıldı.

### Test
- 52 yeni test (toplam 38→90): watchlist (pin/block/kalıcılık/CLI), dual_tier
  (thread yaşam döngüsü + kilit yarış invariantı), auto_screener (filtre/sıralama/
  3 mod/CSV), web watchlist uçları + tier durumu, telegram komut ayrıştırma/dispatch,
  config doğrulama.

## [2.2.2] — Hata düzeltme & optimizasyon turu

### Düzeltilen gerçek hatalar
- **Crash-recovery (Madde 41) kritik bug**: `_trackers` yalnızca bellekteydi; restart sonrası
  `state.json`'dan yüklenen açık pozisyonların tracker'ı olmadığı için `update_all_*` onları
  atlıyordu → taşınan pozisyon hiç izlenmiyor/stop-target/çıkış uygulanmıyordu.
  `PositionManager.rehydrate_trackers()` eklendi, başlangıçta çağrılıyor (+ test).
- **Telegram bildirimi ana döngüyü bloke ediyordu**: senkron `asyncio.run` + 10 sn timeout;
  her sinyalde tarama döngüsü bekliyordu. Fire-and-forget daemon thread'e alındı
  (deprecated `get_event_loop` yolu da kaldırıldı).
- **Madde 16 IV Rank seri kirliliği**: ATM IV en yakın vadeden (çoğu zaman 1-2 DTE) alınıp
  52w seriye yazılıyordu. Artık ~30-45 DTE bandındaki vadeden alınıyor.

### Optimizasyonlar
- `CachingProvider` boş/None sonuçları artık cache'lemiyor (geçici hata TTL kilidi giderildi).
- `get_bars` look-ahead bar-süresi `INTERVAL_MINUTES` üzerinden (deprecated `"H"` + string hack kalktı).
- Opsiyon kontrat seçiminde BS implied-vol çözümünden önce moneyness ön-elemesi (±%30).
- `get_company_events` 1 saatlik süreç-içi memo (tarama başına 2-3 tekrarlı dosya IO kesildi).
- Equity eğrisi yalnızca seans açıkken yazılıyor (gece flat satır birikimi önlendi).
- `is_pin_risk` sıfıra bölme guard'ı.
- Canlı döngü: hisse `bars_held` çift sayımı giderildi + 1m granüler stop kapanışları
  artık bildirimlere ekleniyor.

## [2.2.1] — Sağlamlaştırma turu (A/B/C/D)

Bu tur PDF spec'in ötesinde, gerçek/sürdürülebilir bir sistem için gereken
mühendislik boşluklarını kapatır.

### Düzeltilen gerçek hatalar (bug)
- **`notifier/web_dashboard.py` yoktu** → `main.py` her modda `ImportError` ile çöküyordu.
  İşlevsel FastAPI panosu eklendi (Madde 39/47).
- **`size_stock` SHORT pozisyonlarda hatalı boyutlama**: `entry - stop` negatif olunca
  `risk_per_share` 0.01'e kırpılıp devasa pozisyon üretiyordu → `abs(entry - stop)`.
- **Madde 14 net yönsel tavan ölü config'di** → `portfolio_caps_breach`'e eklendi.
- **Madde 12 sembol-başı kümelenme tavanı ölü config'di** → eklendi.
- **Madde 16 canlı IV Rank `None` sabitti** → zincirden ATM IV ile besleniyor.
- **Madde 7 cooldown yanlış birimdeydi** (dk yerine bar; tek kaynak `HTF_INTERVAL`).
- **`symbol_cooldowns` diske yazılıp okunmuyordu** (restart'ta kayboluyordu) → yüklendi.

### A — Kritik / güvenlik
- `.gitignore` (sırlar/.venv/data/logs).
- `concurrency.PORTFOLIO_LOCK`: web thread ↔ ana döngü yarış koşulu giderildi.
- Web mutasyon uçlarında `WEB_TOKEN` kimlik doğrulaması (Madde 44).
- Test paketi (`tests/`, 37 test, offline) + `pyproject.toml` (ruff/mypy/pytest).

### B — Mühendislik
- `data_provider.CachingProvider`: kısa TTL'li REST cache (Madde 48 rate-limit).
- Canlı per-pozisyon mark/uPnL (`VirtualPosition.last_mark` + panoda kolon).
- `data/equity_curve.csv` canlı equity kalıcılığı; `recent_signals` diske.
- `requirements.txt` temizliği (ta, python-telegram-bot, jinja2, pydantic çıkarıldı).
- Web `/api/reset-breaker` (Madde 42 operatör reset).

### C — Spec tamamlama
- Madde 23: T+1 takas izleme + good-faith farkındalığı (`unsettled_lots`).
- Backtest: `--strategy`, `--portfolio` (paylaşılan equity, Madde 14), `--options`
  (sentetik Black-Scholes, sabit IV yaklaşıklaması), hisse komisyonu, `HTF_INTERVAL`.

### D — İyileştirme
- `trade_log.txt` boyut rotasyonu (5 MB).
- README mimari güncellemesi + bu CHANGELOG.

## [2.2.0] — İlk uyarlama
- PDF spec v2.2 (53 madde) sinyal-only uygulaması; broker/emir katmanı hariç.
