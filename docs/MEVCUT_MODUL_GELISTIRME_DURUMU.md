# Mevcut modül geliştirme listesi — uygulama ve doğrulama

Kaynak: Masaüstündeki `mevcut modül geliştirme .txt` (6 fazın tamamı kapsamda).
Başlangıç: 2026-09-08. Güncelleme: 2026-09-14 (Tur-77). Kalan 8 madde Tur-77'de uygulandı ve ölçüldü;
ayrıntı ve ölçümler: `docs/TUR77_RAPORU_2026-09-14.md`. Başlangıç çalışma ağacındaki diğer değişiklikler korunur.
**Güncelleme 2026-09-15 (Tur-78):** önceki ajanın 19 maddesi Codex R1/R2/R3 + Gemini temiz-oda fark testi +
Claude hakemliğiyle BAĞIMSIZ denetlendi; bulunan kusurlar düzeltildi (aşağıdaki "Tur-78" notları). Rapor:
`docs/TUR78_RAPORU_2026-09-15.md` · 4394 test · mutasyon 26/26.
Tur-77 yerinde ölçümü: bu belgenin "4133 geçti" koşusundan SONRA yapılan son veri-sağlayıcı düzenlemeleri
MainHub'a üretilmemişti (süit 4131 geçti · 2 kaldı) → `tools.sync_mainhub --apply` ile kapandı.

| Madde | Başlangıç kanıtı / yapılacak iş | Durum |
|---|---|---|
| 1.1 EDGAR + evren medyanları | Ortak SEC kuyruğu, `--warm`, tarihli medyan önbelleği, G-score ve pano bağlantısı; 503 sembolün 500'ü ölçüldü | Uygulandı; yıllık evren vekili · Tur-78: `Retry-After` HTTP tarihi, Mohanram tahakkuk taşması, `warm(None)` |
| 1.2 Gerçek opsiyon backtest | CLI ve laboratuvar yerel bid/ask kullanır; seçim/prim çıkışı 1 bar gecikmeli; veri boşluğunda MTM/risk oranı yok | Uygulandı; günlük veri sınırı · Tur-78: dolum yalnız karar gününden sonraki NYSE seansında, Dolt gün içi damga reddi, geçersiz girdi boş çerçeve |
| 1.3 FDA / Wikimedia / makro ETF | FDA onay geçmişi, Wikimedia 28 günlük ilgi ölçümü, 4 ETF vekili; laboratuvar/pano tüketir | Uygulandı · Tur-78: İLERİ PDUFA takvimi SEC 8-K/6-K bültenlerinden (`pdufa_calendar`, şirket beyanı) + lab aracı |
| 2.1 Volatilite konisi | Yang–Zhang/Parkinson, 7 pencere, ATM IV sırası, `/api/quant/vol_cone`, pano seçicisi/grafiği | Uygulandı ve test edildi · Tur-78: ATM IV yalnız spotu çevreleyen strike'lardan |
| 2.2 GEX / gamma flip | Yeniden fiyatlanan GEX üzerinde `brentq`; %1 hareket ve tam hareket birimleri açık | Uygulandı ve test edildi · Tur-78: eksik OI/gamma tarafı işaret uydurmaz (`None` + eksik sayıları) |
| 2.3 Vanna/charm saati | Saatlik Greek yenileme, IV senaryosu, 0DTE/erken kapanış, pano akış grafiği | Uygulandı ve test edildi · Tur-78: NaN OI → `None`; Dupire eşit olmayan aralık türevi |
| 2.4 SVI arbitraj taraması | Komşu vade azalışları ve Durrleman g(k) ısı haritası; eşitlik yalancı ihlali düzeltildi | Uygulandı ve test edildi |
| 2.5 Greek QP hedge | Sınırlandırılmış minimum norm, bağımlı eşitlikler, kalıntı denetimi; opsiyon panelinde form | Uygulandı; sürekli adet, para maliyeti değil |
| 3.1 RMT | Seçilen RMT matrisi optimizasyon, risk katkısı ve etkin sınıra aktarılır; pano seçicisi | Uygulandı ve test edildi · Tur-78: etkin sınır tavanı normalizasyonla delinmez; log getiri yıllıklandırması |
| 3.2 TIMS stres | Tam BS yeniden fiyatlama, 10 spot/IV çifti, senaryo içinde netleme ve teminat açığı; API/pano | Uygulandı; OCC/broker teminat modeli değil · Tur-78: temiz-oda referansıyla 400/400 |
| 3.3 HERC | Gerçek dendrogram dalları, terminal kümelerde tam kovaryansla ERC, küme sayısı seçicisi | Uygulandı · Tur-78: ERC konveks Newton (SLSQP 400'ün 149'unda ağırlığı 0'a yapıştırıyordu → 0) |
| 3.4 CVaR risk paritesi | Konveks log-risk bütçeleme, kesirli kuyruk kütlesi, Euler/alt-türev denetimi; pano katkı tablosu | Uygulandı · Tur-78: kink LP'si minimaks (300'de 40 yanlış `None` → 0); `kink` bayrağı + pano notu |
| 3.5 CPPI/TIPP | Sabit F0 ve kaydedilmiş sanal NLV/ATH; durum dosyası salt okunur, pano karşılaştırması | Uygulandı ve test edildi · Tur-78: dönüştürülemeyen girdi `None` |
| 4.1 Avellaneda–Lee | `statarb.avellaneda_lee_scores`, `/api/quant/statarb`, pano ve Tur-72 analitik testleri | Mevcut; doğrulandı |
| 4.2 Johansen sepetler | Çok değişkenli `johansen_test`, aynı uç/pano; 3–5 varlık analizi, Tur-72 testleri | Mevcut; doğrulandı |
| 4.3 Dinamik slipaj | Ortak modelde gözlenen yarım makas, nedensel hacim/ATR ve η; sanal döngü/backtest/pano bağlı | Uygulandı; Kyle kalibrasyonu veri bekliyor · Tur-78: geçersiz dolum fiyatında istisna BİLİNÇLİ korundu |
| 4.4 VPIN | Birden fazla kovaya taşan barın akışı artık orantılı bölünür; 1'i aşan kova hatası düzeltildi | Düzeltildi ve test edildi · Tur-78: BVC σ'sı tüm pencereden → `causal=False` taşınır; kaba kuvvet hakemiyle 2e-15 |
| 4.5 Gün içi U eğrisi | `intraday_profile/anomaly/liquidity_withdrawal` + `/api/quant/micro` `ucurve` + pano; saat dilimi tahmin edilmez; SPY 57 seans ölçüldü | Uygulandı (Tur-77); OHLCV'de makas yok → çekilme ölçülmez |
| 4.6 Kritik yavaşlama | Örtüşmeyen pencereler; 40 durağan örnekte yanlış pozitif ve 10 devrilmede güç testleri, rejim API/pano | Mevcut; doğrulandı |
| 5.1 Belirsizlik ayrıştırma | `decompose_uncertainty` (p(1−p) + BALD), `predict_uncertainty_decomposition`, `epistemic_flag`; canlı 1m, 5 tohum: epistemik ölçüler isabeti TERS sıralıyor (0–1/5) | Uygulandı; "epistemik yüksekse ABSTAIN" ölçülerek REDDEDİLDİ, bağlanmadı |
| 5.2 Dirichlet EDL | `evidential_belief` + `predict_belief` (Σb+u=1) | Uygulandı; canlı artefaktta `dir_evid` başlığı yok |
| 5.3 Venn–Abers | IVAP + yürüyen değerlendirme + günlük kalibrasyon kümesi; canlı pwin tabanı yenemedi (t=+0,80); 0,5-kapsama payı %0,25 | Uygulandı; nötrleme kuralı bilgi dedektörü değil (ölçüldü) |
| 5.4 K-NN epizot | `analog_ozeti` + `/api/ai/episodes/similar` + pano; defterde %44,4 kopya anı bulundu, iki uçta tekilleştirildi | Uygulandı; vektör 25 boyut, defter 2026-08-22'den, `win` hep boş |
| 6.1 Shortfall | Dört parçalı muhasebe, kalıcı fiyat referansları, CSV okuyucu ve backtest tablosu | Uygulandı; miktarı kayıtsız redler ölçülemez · Tur-78: geçersiz girdi `None`, canlı kapanışı durdurmaz; backtest aynı-bar yeniden girişi kapatıldı (BULGU-78-1) |
| 6.2 HIFO | `tax_report.build(lot_method=...)` + CLI; 23 çakışan satış ölçüldü, bu defterde vergi etkisi 0 | Uygulandı (codex + inceleme) |
| 6.3 Replay | `nihai/execution/replay.py`: SimulatedClock + ReplayProvider + izole alt süreç + yazma çiti; 2026-09-11 gerçek koşu 18 sn, üretim dosyaları değişmedi | Uygulandı (codex + inceleme); opsiyon/borç/rejim tazeleme kapsam dışı |
| 6.4 JS + karar hunisi | Pano betiği 9 modül + başlatma bloğuna bölündü (sabotaj benzetimi testli); `karar_hunisi` + `/api/decision/funnel` + kart | Uygulandı; 08-31: kabul 65 = pozisyon açılışı 65 |

## Değişmezler

- Canlı emir yok; yeni quant çıktıları yalnız rapor. AI modelleri karar paketlerine eklenmez.
- Gelecek veri yok; ölçülemeyen değer `None`. Yeni sinyal adayı eklenirse `exec_lag_1` zorunlu.
- Testler geçici veri/log dizinlerinde; üretim veri ve çalışan servisler korunur.
- Yeni bağımlılık eklenmez. Kaynak/veri sınırlamaları bu belgede açıkça kaydedilir.

## Kanıt ve veri sınırları

- **Dinamik icra maliyeti:** `SlippageModel` backtest ve sanal döngünün ortak kaynağıdır.
  Sağlıklı kote varsa sabit tabanın yerine yarım makas kullanılır; OHLCV'de makas uydurulmaz.
  Mevcut `IMPACT_COEF` artık sanal döngüde de okunur; ana tarayıcı girişte hacim/ATR taşır.
  Backtest çıkışı önceki karar barının, canlı 1m çıkışı tetik barından önceki tamamlanmış
  barların likiditesini kullanır. Tetik barının nihai hacmi geriye sızmaz. Quote/manüel çıkışta
  yeni bar verisi yoksa son ölçülen hacim/ATR taşınır. Eski giriş kotasyonu 1m çıkışına taşınmaz.
  Modelin kesir birimi `yarım_makas/fiyat + η × (ATR/fiyat) × √(Q/ortalama_bar_hacmi)`.
  Ham Kyle λ ($/adet) bu boyutsuz katsayıyla aynı değildir; planın ifadesine doğrudan
  yerleştirilmedi. İmzalı işlem akışı / gerçekleşmiş dolum kalibrasyonu olmadığından η
  ölçülmüş katsayı sayılmaz; varsayılan 0, pano alanı 0–5 arası duyarlılık senaryosudur.
  Günlük Dolt opsiyon testi gerçek bid/ask yolunu korur; hisse η alanını kullanmaz.
  [Boyut analizi ve piyasa etkisi](https://arxiv.org/abs/1702.05434).
- **Shortfall:** Sinyal→karar gecikmesi ve karar→dolum farkı yalnız dolan miktara,
  fırsat maliyeti dolmayan miktara yazılır; doğrudan ücret dolar cinsindedir. `BUY/SELL`
  emir yönüdür; opsiyon çarpanı ücreti ikinci kez çarpmaz. Fırsat maliyeti sinyal fiyatından
  sabit değerlendirme ufkuna ölçülür; kaçırılan zarar negatif kalır. Bilinmeyen ufuk fiyatı
  veya adet için kâr uydurulmaz. Dört bileşenin toplamı, aynı ufukta varsayımsal portföy
  PnL'i eksi gerçek/simüle PnL kimliğiyle testlidir.
  [CFA işlem maliyeti ayrıştırması](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/trade-strategy-execution).
  Fiyat referansları sanal pozisyonla kaydedilir ve yeniden başlatmada korunur. Eski pozisyon
  ve CSV satırlarında eksik ölçümler `None` kalır; rapor her sütunun ölçüm sayısını gösterir.
  `trades.csv` ve hisse backtest sonucu iki bacağın toplamını taşır. Bu sanal motor tam dolum
  ve çıkış tetik→karar arasında sıfır gecikme varsayar; broker dolum günlüğü değildir.
  Reddedilen adayların hedef adedi/ufuk fiyatı tarihsel kayıtlarda yok; bu adaylar toplamın
  dışındadır. `--orders-csv` miktarı/ufku kaydedilmiş kısmi veya hiç dolmayan emirleri de
  hesaplar. Dolt/sentetik opsiyon backtestlerinin mevcut makas/ücret raporu ayrı kalır.
  Maliyetler PnL'den ikinci kez düşülmez. Eski maliyet açık/kapalı karşılaştırması bir
  duyarlılık deneyidir; boyut/çıkışlar değişebildiğinden fark saf maliyet değildir.
  2026-09-10 gerçek CLI kontrolü mevcut **654** `data/trades.csv` satırını salt okunur
  işledi; geçmiş dosyada yeni alanlar olmadığından beş tutar `null`, ölçüm sayıları 0 çıktı.
- **TIMS tarzı stres:** Anlık spot/IV çiftleri sırasıyla `(-12,+20), (-8,+10), (-6,0),
  (-3,-10), (0,-20), (0,+20), (+3,-10), (+6,0), (+8,+10), (+12,+20)` yüzdedir.
  IV şoku göreli: %25 IV'nin +%20 şoku %30 IV olur. Her senaryoda tüm bacaklar önce
  yeniden fiyatlanır, sonra netlenir; en kötü bacak kayıpları ayrı ayrı toplanmaz.
  Teminat kapasitesi verilmezse açık `None` kalır. Avrupa BS, sabit vade, sürekli temettü
  varsayımı ve dayanaklarda ortak şok kullanılır. Bu, [OCC hesabının](https://www.theocc.com/risk-management/risk-based-haircuts)
  CRR fiyatlaması, ürün bazlı şokları ve ek taban ücretlerini uygulayan resmi teminat modeli değildir.
  `/api/quant/tims` çok bacaklı sepet alır; opsiyon hesaplayıcısı kendi kontratını, tahsis paneli
  referans hisse sepetini (birim sermaye başına) raporlar.
- **HERC:** Planın istediği tam kovaryanslı küme-içi ERC kullanılır; gerçek dendrogram
  dallarında iki kardeş portföyün risk katkıları eşitlenir. Tüm varlıkların **global** ERC'si
  iddia edilmez. Terminal kümede ERC sapması denetlenir; kutu sınırı nötr katkıyı bozacaksa
  sonradan kırpmak yerine yöntem ölçülemedi olarak kalır. Küme sayısı açık parametredir;
  0 seçimi mevcut NCO gibi yuvarlanmış sqrt(N) kesimidir, optimal küme sayısı taraması değildir.
- **CVaR risk paritesi:** `ES(x) − mean(log x)` destek düzlemleriyle çözülür. En kötü %5
  kuyrukta kesirli gözlem kütlesi korunur; VaR'da eşit kayıplar olduğunda Euler eşitliğini
  sağlayan geçerli alt-türev ayrıca LP ile denetlenir. Maksimum katkı payı hatası `1e-5`'i
  aşarsa/sınırlar bozulursa/pozitif minimum ES varsayımı sağlanmazsa `None` döner.
  Pano ES'yi bar başına log getiri kaybı olarak ve katkı paylarını ayrıca gösterir.
  [Risk bütçeleme çalışması](https://arxiv.org/abs/2302.01196).
- **CPPI/TIPP:** F0 operatörün girdiği sabit tabandır; TIPP tabanı `max(F0,(1−α)·ATH)` olur.
  `/api/quant/insurance` yalnız mevcut `state.json` NLV/ATH alanlarını ve kayıt zamanını okur;
  eksik dosyadan başlangıç sermayesi üretmez, dosya yazmaz. Aralıklı dengeleme ve fiyat
  sıçramaları tabanı delebilir. Sanal portföy yeniden fiyatlanmaz ve otomatik tahsis yapılmaz.
- **Opsiyon Greek birimleri:** Pano etiketleriyle uyum için API'de theta takvim günü başına,
  vega 1 IV yüzde puanı başına çevrilir; saf `bs_greeks` fonksiyonunun yıllık/ham birimleri korunur.
- **VPIN:** Eski döngü taşan barın tüm dengesizliğini ilk kovaya koyup sonraki kovaları
  sıfırlıyordu. Örneğin 90 alım + 10 satım, iki 50 hacimli kovada artık `[1.0, 0.6]` ve
  VPIN `0.8` verir. Kümülatif akış kova sınırlarında bölünür; bar içinde BVC oranı sabit
  varsayılır, tick bazlı işlem yönü ölçülmüş değildir.
- **Yeni veri köprüleri:** 2026-09-09 gerçek sorgularında FDA/PFIZER için 2025 yılına ait
  37 başvuru kaydı (tam sonuç), Wikimedia/Apple Inc. için 2026 Temmuz–Ağustos döneminde
  62/62 gün ve altı Alpaca ETF tarihçesi alındı. [openFDA](https://open.fda.gov/apis/drug/drugsfda/)
  onay ve ek başvuru geçmişidir; gelecekteki PDUFA takvimi sağlamaz. Sponsor/ticker eşlemesi yapılmaz.
  [Wikimedia](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html)
  karşılaştırması önceki 28 tam günü kullanır; eksik gün ve sıfır varyans ayrı raporlanır.
  GLD/USO, CPER/GLD, HYG/LQD oranları ve BIL'in yıllıklandırılmış geçmiş 20 seans getirisi,
  [Alpaca'nın tüm kurumsal işlem düzeltmesiyle](https://docs.alpaca.markets/us/reference/stockbars)
  hesaplanır. ETF oranları saf emtia fiyatı/kredi spreadi, BIL ölçümü ileriye dönük risksiz faiz değildir.
  Tamamlanmamış gün elenir; varsayılan kullanılabilirlik T+1'dir. Güncel düzeltilmiş seriler geçmiş
  revizyonların bilinme tarihini kanıtlamaz. Pano cache-only okur; ağ yalnız laboratuvar/CLI yenilemesinde açılır.
- **Makro okuma sözleşmesi:** PCE ve tek-seri yardımcılarında eksik `fetch` aktarımı düzeltildi.
  Yayın gecikmesi sonrasındaki gelecek tarihler de elenir. Özet/API, süresi dolan gözlemi ikinci
  kez ileri doldurmaz; son kaynak/kullanılabilirlik tarihi ayrı tutulur.
- **Greek nötrleme:** `quant_math.greek_neutral_hedge` mevcut `advanced_math.qp_solve` çözücüsünü
  kullanır. Bağımlı eşitlikler SVD ile indirgenir; çözümün özgün Greek kalıntıları ve adet sınırları
  yeniden denetlenir. Çelişkili hedef/olanaksız sınır `None` olur. Amaç `min ||w||²` olup parasal
  maliyet değildir; kontrat çarpanları girdide bulunur, kesirli adetler yuvarlanmaz. Kaynak:
  [SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).
  Rapor `/api/quant/greek_hedge` ve opsiyon panelindedir; risk/emir yoluna bağlanmaz.
- **RMT optimizasyonu:** Mevcut Marčenko–Pastur filtresi `covariance_model=rmt` ile seçilir.
  `propose` raporladığı kovaryansı API'ye verir; ağırlıklar, risk katkıları, korelasyon ve etkin
  sınır aynı matrisi kullanır. RMT ölçülemezse başka modele sessiz geçiş yapılmaz. Bu bağlantı
  bir OOS performans iddiası değildir. [RMT incelemesi](https://arxiv.org/abs/1610.08104).
- **EDGAR:** 2026-09-08 toplu koşusunda 503 istekten 500 sembolde yıllık veri bulundu;
  `ECHO`, `HONA`, `VMRK` eksik kaldı. Önbellek: `data/edgar_cache/_medians_2026-09-08.json`.
  Reklam gideri artık SG&A ile karıştırılmaz. Her kalemin geçerli örneklem sayısı ayrı tutulur.
  Özgün sektör/düşük B/M/çeyreklik karşılaştırması yerine **yıllık evren vekili** raporlanır;
  eksik kalemler puanlanmaz. [Mohanram çalışması](https://www-2.rotman.utoronto.ca/facbios/file/rast%20growth.pdf).
  SEC kuyruğu süreç içindeki çağrıları seri işler; 429 bekleme süresi paylaşılır. Birden fazla
  süreç için ortak IP hız limiti garantisi yoktur. [SEC hız sınırı](https://www.sec.gov/filergroup/announcements-old/new-rate-control-limits).
- **Dolt:** 10.69 GB yerel `C:\Users\Kagan\dolt_cache\option_chain__master.csv` salt okunur süzüldü;
  bütün dosyanın yeni parquet kopyası oluşturulmadı. AAPL, 2024-01-01–2024-06-30:
  **11.226 satır / 78 kayıt günü**, pazartesi/çarşamba/cuma (tatil kayıtları da mevcut).
  İşlem modeli yalnız NYSE seanslarını kullanır. Kesin kotasyon saati, işlem hacmi ve teslimat
  çarpanı kanıtı yok: günlük kapanış ve standart 100 pay çarpanı varsayılır.
  Geçmiş kayıt tarihlerinin süzülmesi testlidir; güncel CSV, tarihsel veri revizyonlarının o gün
  bilindiğini kanıtlamaz. Bunun için eski Dolt commit ihracı gerekir.
  [Veri kaynağı](https://www.dolthub.com/repositories/post-no-preference/options).
- **Opsiyon icrası:** t gününde kontrat seçilir; yalnız t+1 seansının ask'i ile alınır.
  Sonraki seansta kotasyon yoksa giriş iptal edilir. Prim stop/hedef kararı da sonraki seansta
  bid üzerinden icra edilir. Sıfır bid satış dolumu sayılmaz. Eksik açık pozisyon değerleri,
  bunlara bağlı Sharpe/Sortino/düşüş ve bilinmeyen son bakiye JSON'da `null` olur.
  Makas, orta fiyata göre giriş/çıkış maliyeti olarak ayrıca gösterilir; PnL'den ikinci kez düşülmez.
  Eski sentetik deney yalnız açık `--options-synthetic` bayrağıyla kullanılabilir.
- **SVI:** Plan metnindeki Durrleman formülünde eksik olan `w'²` doğru formülde korunur.
  Komşu vadelerdeki negatif toplam varyans eğimi kontrol edilir; merkezî farkın gizlediği yerel
  azalış yakalanır. Eşit varyans tek başına arbitraj değildir. Isı haritası sonlu strike ızgarası
  taramasıdır, küresel arbitrajsızlık kanıtı değildir. [Gatheral–Jacquier](https://arxiv.org/abs/1204.0646).
- **Dealer saati:** Sabit spot/OI, çağıranın IV değişimi ve saat başına birinci dereceden
  yaklaşım kullanılır. Call uzun/put kısa dealer varsayımı gözlenmiş işlem tarafı değildir.
  NYSE PM vadesi/erken kapanışı kullanılır; AM uzlaşmalı endeks kontratları kapsamda değildir.

## Doğrulama

- Gerçek zincir, mevcut backtest, parquet okuyucu ve laboratuvar: **52 test geçti**.
- SVI yüzeyi, volatilite konisi ve Tur-72 modelleri: **68 test geçti**.
- EDGAR/G-score ve dealer kontrolleri de geçti; yeni regresyonlar `tests/test_development_*.py` içinde.
- Değişen Python dosyalarında Ruff temiz; ilgili bağımlılıklarla mypy temiz.
- Laboratuvar JS sözleşmesi ve ana panonun tüm gömülü JS sözdizimi kontrolü geçti.
- Bu aşamanın tüm proje kontrolü: **4105 geçti, 1 atlandı**, 360 saniye; 2 bağımlılık eskime uyarısı.
  Ardından eklenen FDA/Wikimedia/ETF, makro, Alpaca ve laboratuvar kontrolleri: **192 test geçti**.
- QP/RMT, mevcut portföy/gölge akışları, quant API, veri köprüleri ve laboratuvar: **463 test geçti**;
  değişen 9 modülün mypy ve Ruff kontrolleri temiz.
- Önceki genel koşu (2026-09-10): **4121 geçti, 1 atlandı**, 314 saniye; 2 bağımlılık eskime uyarısı.
  Bu koşu başladıktan sonra bulunan VPIN kova düzeltmesi ve güçlendirilen bit-bit deterministik
  kontroller ayrıca bütün doğrudan tüketicileriyle doğrulandı: **94 test geçti**.
- Slipaj/shortfall sonrası genel koşu: **4133 geçti, 1 atlandı**, 304 saniye; aynı 2 bağımlılık
  uyarısı. Ardından ortak giriş/çıkış fiyat taşması koruması ve tüm ilgili yollar ayrıca
  **103 testle** geçti. Altı kaynak dosyasında mypy, değişen özellik dosyalarında Ruff ve
  laboratuvar JS kontrolü temiz. Sandbox içindeki genel koşu Windows geçici klasör izninde
  durdu; belirtilen başarılı genel koşu onaylı pytest kapsamıyla çalıştırıldı.
- Portföy/quant arayüzü ve laboratuvar JS çalıştırılabilir kontrolleri temiz. Tüm yeni matematik
  fonksiyonlarında aynı girdinin aynı sonucu üretmesi, kapalı form ve geçersiz girdi kontrolleri var.

## Kullanım

```powershell
.\.venv\Scripts\python.exe -m nihai.data.edgar_facts --warm --workers 4
.\.venv\Scripts\python.exe -m nihai.data.fda_calendar --sponsor PFIZER --since 2025-01-01 --until 2025-12-31
.\.venv\Scripts\python.exe -m nihai.data.wiki_trends --article "Apple Inc." --since 2026-07-01 --until 2026-08-31
.\.venv\Scripts\python.exe -m nihai.data.macro_regime --etf-only --since 2025-01-01
.\.venv\Scripts\python.exe -m nihai.execution.backtest --symbol AAPL --options --interval 1d --from 2024-01-01 --to 2024-06-30 --provider yfinance --save
.\.venv\Scripts\python.exe -m nihai.execution.cost_attribution --trades-csv data/trades.csv
.\.venv\Scripts\python.exe -m nihai.execution.cost_attribution --orders-csv data/order_costs.csv
```

Yerel zincir konumu `--options-file` ile verilebilir. Pano laboratuvarındaki
“Gerçek opsiyon (Dolt, günlük)” aynı motoru çağırır; boş kaynakla sentetik sonuca geçmez.

Laboratuvar veri araçları FDA/Wikimedia/ETF köprülerini yeniler. Quant opsiyon panelinde
Greek nötrleme ve 10 senaryo; tahsis panelinde RMT/HERC seçimi, CVaR katkı tablosu ve
kayıtlı sanal NLV üzerinden CPPI/TIPP formu bulunur.

`--orders-csv` girdi sütunları: `side,quantity,filled_quantity,signal_price,decision_price,fill_price,fees,benchmark_price,multiplier`.
`side` BUY/SELL, çarpan hissede 1/standart opsiyonda 100; fees toplam dolardır. Eksik fiyat/ücret
boş bırakılır. Dosya kullanıcı tarafından sağlanır; bu komut girdi oluşturmaz ve değiştirmez.
Laboratuvar hisse backtestinde η alanı ve ölçüm kapsamını gösteren dört maliyet tablosu vardır.

## Sıradaki çalışma

Listenin uygulanabilir maddelerinin tamamı Tur-77 itibarıyla uygulandı; Tur-78 önceki ajanın 19 maddesini bağımsız
denetleyip düzeltti (bkz. `docs/TUR78_RAPORU_2026-09-15.md` §8 açık kalemler). Veri bekleyenler: gerçek Kyle kalibrasyonu imzalı işlem akışı/dolum verisi; reddedilen
adayların parasal fırsat maliyeti hedef adet ve önceden seçilmiş değerlendirme ufku; 5.2'nin canlı yüzeyi
`dir_evid` başlıklı yeniden eğitilmiş bir model. Mevcut kayıttaki yokluk sıfır sayılmaz.
