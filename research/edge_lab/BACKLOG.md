# Kenar-Keşif Programı — Aday Sıralaması (Tur-16 WP-E)

Kaynak: `docs/AI_V3_DERIN_BEYIN_PLAN_2026-07-19.md` §WP-E. Sıralama = literatür desteği ×
veri hazırlığı (en yüksek beklenen değer önce). **Hiçbir aday `harness.run_candidate`'ın
dört kapısından (purged_cv → gate_time/gate_symbol → dsr → exec_lag_1) geçmeden "kenar"
ilan edilemez**. **Tur-24'ten beri kapı simülasyonunun İKİ EK koşulu vardır** (§26'nın
dersi makineye çevrildi): kapı-sonrası işlem sayısı `max(5, %0,5·n_test)` oransal tabanını
geçmeli VE tutulan↔elenen farkı Welch t>1,645 olmalı — "kapı-E > base-E" tek başına
YETMEZ — bu tablo yalnız SIRALAMA ve DURUM izlemidir, sonuç iddiası değildir.

Durum kolonu değerleri: `BEKLIYOR` (veri/harness hazır değil) · `HAZIR` (veri var, harness
koşulabilir, henüz koşulmadı) · `KOSULDU-FAIL` (kapı(lar) geçilemedi) · `KOSULDU-PASS`
(tüm kapılar geçildi — **yalnızca** canlı gölge doğrulaması gerektirir, kenar iddiası DEĞİL).

| # | Aday sinyal ailesi | Veri kaynağı | Literatür desteği | Durum | Not |
|---|---|---|---|---|---|
| 1 | **Overnight/intraday getiri ayrışması** (İLK ADAY) | Alpaca SIP günlük (30 sembol, 5y) | Orta-yüksek (bilinen overnight-drift anomalisi) | **KOSULDU-FAIL** (2026-07-25, pilot) | 5/5 kapı FAIL — purged_cv OOS-AUC 0.502 (yazı-tura), gate_time/gate_symbol/exec_lag_1 kapıyı geçen işlem YOK (0-2/kapı), DSR hesaplanamadı. **SURVIVORSHIP-ŞİŞKİN PİLOT** (mevcut-likit 30 sembol) — kenar YOK sonucu bu evrende bile net; PIT evrende tekrar denemenin önceliği düşük. Detay: `SONUC_overnight_pilot_2026-07-25.md` |
| 2 | **VRP (IV − gerçekleşen vol, zamanlama sinyali)** | DoltHub `volatility_history` (IV/HV) + Alpaca günlük **düzeltilmiş** (`_1d_all`), PIT evren | Yüksek (literatürün en dayanıklı primlerinden) | **KOSULDU-FAIL** (2026-08-27, Tur-51) | 5 kapıdan 3'ü FAIL. Veri kaynağı notu ESKİMİŞTİ: yakıt 2026-08-13'ten beri diskteydi. **20.910 satır / 581 PIT sembol / 2021-06→2026-07**, ileri RV FİYATTAN (vekil kaydırma YOK), örtüşmesiz, feature-lag 1. **Prim VAR** (taban +0,00867 oynaklık puanı, %60,5 kazanan — 1,0 puan maliyet DÜŞÜLMÜŞ) ama SEÇİLEBİLİR DEĞİL. Belirleyici: `purged_cv` **0,514** · `dsr` **0,083** · ve asıl olan **`exec_lag_1` t=−2,61** — sembol bölmesinde t=+7,32 olan kapı fazlalığı 1 bar gecikmede **İŞARET DEĞİŞTİRİYOR** = **ACI-7 BEŞİNCİ kez** (öncekilerde eriyordu, burada TERSİNE DÖNÜYOR). Seçicilik sorunu YOK (%85-91 tutuluyor) → §27 gibi, büyük örneklemli PASS de kenar değil. Yan ürün: BULGU-51-1/2/3/4 (`hv_current`=30g RV ÖLÇÜLDÜ · sağlayıcı `*_month_ago` 2024'ten beri BOŞ · düzensiz seride `shift` ufku ortanca 70 gün KAYIYOR · **"IV Rank bandı ters" bulgusu vekilin ARTEFAKTIYDI, çürütüldü**). Detay: `docs/EDGE_FINDINGS.md` §28 |
| 3 | **Lead-lag ağları** (sektör-içi büyük→küçük gecikmeli yayılım) | Alpaca SIP saatlik, 2y, 28 sembol | Orta | **KOSULDU-FAIL** (2026-08-03, pilot) | 5 kapıdan 2'si FAIL. purged_cv 0,532 (eşiğin 0,002 üstü, kıl payı) ve her iki kapı simülasyonu PASS **ama yalnız 29/33.212 işlem üstünde (%0,087 seçicilik)**; DSR **0,3058** (kapı 0,95). Belirleyici: **exec_lag_1** — 1 bar gecikmede kapıyı geçen işlem 29→**3**, ölçülemez hale geliyor = ACI-7 zamanlama artefaktı, tıpkı pairs §22'deki gibi. Bu satırdaki "ACI-7 riski yüksek" ön-notu koşudan ÖNCE yazılmıştı ve DOĞRU çıktı. Lider seçimi hindsight'sız (her barda geriye-bakan 20-bar medyan dolar-hacmi). Detay: `SONUC_leadlag_pilot_2026-08-03.md` |
| 4 | **Endeks rebalans/akış olayları** (S&P ekleme-çıkarma, ay-sonu) | Alpaca günlük **düzeltilmiş** (`_1d_all`), PIT evren (pit_universe.py) | Orta-yüksek | **KOSULDU-FAIL** (2026-08-10, pilot) | 5 kapıdan 1'i FAIL → verdikt FAIL. **PIT/survivorship-TEMİZ** (ilk aday): 607 sembol = `sp500_as_of('2021-01-01')` (506, **90'ı bugün endekste YOK**) ∪ pencerede eklenen 101; **728.978 satır** / 605 sembol / 2021-02-02→2026-07-23, ufuk 5 bar, 8 özellik. Belirleyici: **`purged_cv` OOS-AUC 0,5066** (yazı-tura, eşik >0,53). Diğer dört kapı GEÇTİ ama okunduğunda boş: (a) **`dsr` bu veri türünde AYIRT EDİCİ DEĞİL** — DSR=1,0'ı kapı-sonrası (SR 0,091), kapısız base (0,048), kapının **ELEDİĞİ** satırlar (0,042) ve **rastgele** alt küme (0,050) de veriyor = ölçtüğü şey piyasa sürüklenmesi (ACI-8'in aynı yapısal hatası, gerçek `pnl_col` veren İLK adayda ilk kez göründü); (b) kapı **endeks-olay özelliklerine bakmıyor** (tutulan↔elenen `gun_since_*` farkı −0,09/+0,13), gördüğü şey takvim (`ceyrek_sonuna_gun` +24,3 fark; en büyük standardize katsayı 0,063); (c) **`exec_lag_1` PASS ama fazlalığın %72'si buhar** (+0,2766 → +0,0765, t 8,51→2,22) = ACI-7 dördüncü gözlem. Seçicilik sorunu YOK (%8–12, 18k–27k işlem) — yani büyük örneklemli PASS'ler de kenar değil. H1 doğrudan okuma: fiyat baskısının olması gereken 0–10 gün penceresi BOŞ (n=359/279, t=−1,16/−1,73). H2: satır düzeyi t=+5,75 → **tarih-kümelenmiş t=+0,52** (11× şişme; bağımsızlık birimi TARİH'tir). 101 ekleme / 103 çıkarma olayı (çıkarmaların yalnız 65'i satır üretebildi — satın alma → hisse işlem görmüyor). Detay: `SONUC_rebalance_pilot_2026-08-10.md` |
| 5 | **Form 4 açık-piyasa ALIM** (insider) | **SEC toplu Form 345** çeyreklik setleri + Alpaca günlük **düzeltilmiş** (`_1d_all`), PIT evren | Yüksek (Lakonishok-Lee 2001; Cohen-Malloy-Pomorski 2012) | **KOSULDU-FAIL** (2026-08-28, Tur-52) | **5/5 kapı FAIL** — ilk aday ki hiçbir kapıyı geçemedi. `purged_cv` **0,494** (yazı-tura) · `gate_time` t=**−0,76** ve `gate_symbol` t=**−0,38** = kapı base'ten DAHA KÖTÜ seçiyor · `dsr` **0,0077** · `exec_lag_1` FAIL. Veri: **1.815 satır / 456 PIT sembol / 2021-01-05→2026-03-31**, örtüşmesiz, feature-lag 1, filing tarihi (transaction DEĞİL), yalnız kod P + edinim A, `4/A` hariç. **Kaynak değişimi:** per-filing EDGAR ~140 sn/sembol (tam evren ~20 saat, ancak örneklenmiş pilot mümkündü) → SEC çeyreklik toplu set **22 istek / ~60 sn / TAM evren** (55.003 sembol-gün alımı, 5.711 sembol). **Koşulsuz iz de ölçüldü ve BETA çıktı:** ham 21g getiri +1,221% görünüyor ama AYNI pencerede eşit-ağırlık evren **+1,053%** → **fazlalık yalnız +0,168%, kazanan %48,9** (yarının ALTI), gün-t=0,93, blok-bootstrap P(fazlalık≤0)=**0,16**. Beta kontrolü artık veri setinin ŞEMASINDA (`mkt_ret`/`excess`) — çağırana bırakılırsa unutulur. |
| 8 | **Günlük açığa satış AKIŞI** (FINRA konsolide short volume) | FINRA `regsho/daily` (ücretsiz, anahtarsız, 12k sembol/gün) + Alpaca günlük **düzeltilmiş** (`_1d_all`), PIT evren | Yüksek (Boehmer-Jones-Zhang 2008 JF; Diether-Lee-Werner 2009 RFS) | **KOSULDU-FAIL** (2026-08-29, Tur-53) | **5/5 kapı FAIL** — insider'dan sonra hiçbir kapıyı geçemeyen ikinci aday. `purged_cv` **0,509** · `gate_time`/`gate_symbol` kapıyı geçen işlem **0** (gerekli 221/223) · `dsr` hesaplanamadı · `exec_lag_1` de 0. Veri: **147.318 satır / 593 sembol / 2021-03-31→2026-06-23**, 1.378 işlem gününün TAMAMI (eksik gün 0, 777.556 sembol-gün akış), örtüşmesiz, feature-lag 1 (FINRA D dosyasını D kapandıktan SONRA yayımlar → giriş D+1 kapanışı), ödünç maliyeti short bacakta, eşik/ufuk taraması YOK (ufuk 5 ön-kayıtlı). **"0 tutuldu" burada §26'nın kuyruk artefaktı DEĞİL:** test diliminde üretilen olasılıklar [0,4214·0,5559] aralığında (ort 0,4830 ↔ taban 0,4850), 0,55'i aşan 1 satır; standardize katsayıların hepsi ≤0,0225 ve akış özellikleri 6./7. sırada — model tabandan KIMILDAYAMIYOR. **BULGU-53-1:** koşulsuz izde literatürün işareti YOK (short bacak +0,0173%, long −0,0153% = ikisi de TERS ve gürültü) ve satır-t ile GÜN-t **işaret değiştiriyor** (+1,10→−0,55 · −0,92→+1,82) — Tur-27 birimin t'yi ŞİŞİRDİĞİNİ ölçmüştü, burada YÖNÜNÜ çeviriyor. SINIR: ölçülen şey short VOLUME'dur, short INTEREST değil. Detay: `docs/EDGE_FINDINGS.md` §29 |
| 9 | **Opsiyon YÜZEYİ yön sinyali** (put-call parite sapması + smirk) | DoltHub `option_chain` toplu CSV (10,2 GB, 116,5M satır) + Alpaca günlük **düzeltilmiş** (`_1d_all`), PIT evren | Yüksek (Cremers-Weinbaum 2010 JFQA; Xing-Zhang-Zhao 2010 JFQA) | **KOSULDU-FAIL** (2026-08-29, Tur-54) | **5/5 kapı FAIL.** `purged_cv` **0,513** · iki kapıda da tutulan işlem **3** (gerekli 65/66) · `dsr` kurulamadı · `exec_lag_1` **0**. Yüzey: **514.779 gün×sembol / 590 sembol / 2021-2026**, ort. 6,0 eşleşen çift (`data/edge_lab_optskew_yuzey.csv` önbelleğinde). **BULGU-54-1 — VERDİKT 1,7 YILI KAPSAR, 5,5 YILI DEĞİL:** sağlayıcı kapsaması 2021-23'te işlem günlerinin ~%55'i, 2024 ortasından sonra ~%96 → 60g/40-gözlem penceresi 2021-23'te **hiçbir sembolde** satır üretemiyor; veri seti 2024-10-14→2026-06-23, 43.140 satır / 551 sembol. BULGU-51-2'nin kardeşi (orada eksik olan KOLON, burada TAKVİM). **Koşulsuz iz §29'un TERSİ:** işaret literatürün yönünde (long +0,0110% · short −0,0312%) ama gün-t +0,96/−0,16 = maliyetin altında; smirk ise TERS (dik +0,0939% ↔ düz +0,0565%). **AÇIK KALEM KAPANDI (Tur-63):** `_stage_coverage` artık İSTENEN pencereyi (kurucunun `--since/--until` beyanı + `ISINMA_GUN`) DOĞAN pencereyle kıyaslar. Bu veri seti bugün yeniden koşulsa ÖN-KOŞULDAN DÜŞER (baş kesme 1.326 gün = %68,0 > tolerans %5) ve verdikt "kenar YOK" değil "HÜKÜM YOK" olur — kesilmiş pencere bir çürütme DEĞİLDİR. Detay: `docs/EDGE_FINDINGS.md` §30, `docs/TUR63_RAPORU_2026-08-31.md` |
| 10 | **Düşük beta (BAB) × makro rejim** | PIT `_1d_all` + **FRED** (`T10Y2Y`, `NFCI`; ANAHTAR GEREKMEZ — `nihai/data/fred_data.py`) | Yüksek (Frazzini-Pedersen 2014 JFE) | **KOSULDU-FAIL** (2026-08-29, Tur-55) | 5 kapıdan 4'ü FAIL (`gate_symbol` tek başına PASS). `purged_cv` 0,518 · `gate_time` kapı base'in **ALTINDA** (Welch t=**−3,75**) · `dsr` **0,0** · `exec_lag_1` etki −0,0268. 32.417 satır / 595 sembol / 2021-08→2026-05. **İşaret bu dönemde TERS:** beta-ayarlı fazlalık düşük-betada −0,0754%, yüksek-betada +0,0668%. ÖLÇÜT SEÇİMİ AÇIK: `excess` beta-ayarlı, `excess_duz` de şemada (düz ölçüt long bacağı yapısal cezalandırırdı: −0,4424% ↔ +0,4373%). Detay: §31.1 |
| 6 | 13F kurumsal kümelenme | SEC çeyreklik toplu 13F (VAR: `.../form-13f-data-sets/01mar2026-31may2026_form13f.zip`, ~100 MB/çeyrek) | Orta | **VERI_ONKOSULU_YOK** (2026-08-29, Tur-55) | **Ölçüm YAPILAMADI — "kenar yok" DEĞİL.** `INFOTABLE` şemasında TICKER yok (`NAMEOFISSUER · CUSIP · FIGI(boş) · VALUE · SSHPRNAMT`). Evrene bağlamanın tek ücretsiz yolu isim eşlemesi ve ÖLÇÜLDÜ (3.822.885 satır / 35.649 CUSIP): tam eşleşme **%30,7** · CUSIP'lerin **%42,2'si birden fazla isimle** geliyor · eşleşen ticker'ların **%69,4'ü birden fazla CUSIP** alıyor = eşleme bire-bir DEĞİL. İsim haritası bağımsız değişkenin KENDİSİNİ yanlış ölçerdi. Lisanslı CUSIP↔ticker haritası olmadan koşulamaz. Detay: §31.4 |
| 11 | **Kalıntı (residual) momentum** — Blitz-Huij-Martens 2011 | Mevcut fiyat verisi (PIT `_1d_all`) + Ken French FF5 önbelleği (`data/factor_cache`, ANAHTARSIZ) | Orta-yüksek (Blitz-Huij-Martens 2011 JEF; toplam momentumun faktör kaynaklı çöküşlerini yumuşattığı raporlanır) | **KOSULDU-FAIL** (2026-09-19, Tur-86) | **5/5 kapı FAIL.** `purged_cv` **0,517** · `gate_time` kapı base'in ALTINDA (E −0,0041 ↔ +0,0038, Welch t=**−0,92**, seçicilik %1,9) · `gate_symbol` t=0,39 (%0,9) · `dsr` **0,0006** (`n_trials=44`) · `exec_lag_1` FAIL. Veri: **18.194 satır / 545 PIT sembol / 37 karar ayı** (2023-07→2026-07), ay başına ~492 o-günkü üye. **Koşulsuz iz literatürün YÖNÜNDE ama anlamsız:** uzun bacak +0,231%, kısa −0,231% (maliyet öncesi, aylık), fark +0,46% · satır-t 3,66 ↔ **gün-t 1,63** (37 ay, %56,8 pozitif ay) · maliyet sonrası taban +0,089%/işlem. **BULGU-86-4 bu ailede ölçüldü:** PIT süzgeci olmasa satırların %11,2'si (2.298; 84 sonradan eklenen isim) o gün üye OLMAYAN isimden gelir, o satırların ileri getirisi **+3,21%/ay** → fark +0,69%, gün-t **2,23** = süzgeçsiz aile SAHTE biçimde "anlamlı" görünürdü. İlk koşu `veri_kapsami` ön-koşulundan düştü (son kesme %5,5 — kurucu etiket verisini `until`'de kesiyordu; düzeltildi, HÜKÜM YOK koşusu deftere yazılmadı). Detay: `docs/EDGE_FINDINGS.md` §33, `docs/TUR86_RAPORU_2026-09-19.md`. **ÖN-KAYIT (sonuç görülmeden yazılan metin):** Kural: her ay sonu D'de aylık aşırı getiri son **36 ayda** FF3 (Mkt-RF·SMB·HML, sabitli EKK) üzerine regres edilir; skor = **t−12…t−2** kalıntı toplamı / aynı 11 ayın kalıntı std'si (son ay HARİÇ). Kesitsel **MEDYAN** üstü LONG, altı SHORT · ufuk **21** · feature-lag 1 · maliyet %0,1 + ödünç 100 bps/yıl (Tur-55 ile aynı) · **TARAMA YOK**. **PIT ÜYELİK kilidi:** kesit her D'de `pit_universe.sp500_uyelik` ile O GÜNKÜ üyelere süzülür — panel pencerede SONRADAN eklenen isimleri de taşır ve endekse eklenme yükselişin sonucudur (momentumda evren seçimi look-ahead'i). Kıyas tabanı AYNI PIT kesitinin eşit-ağırlık ileri getirisi. Isınma: 36 ay → ilk karar ~2023-06 (veri 2020-06'da başlar). DSR çoklu-test sayacı: **11**. |
| 7 | **Mevsimsellik (Heston-Sadka kesitsel)** | Mevcut fiyat verisi (PIT `_1d_all`) | Düşük-orta (çoklu-test riski YÜKSEK) | **KOSULDU-FAIL** (2026-08-29, Tur-55) | **ÜÇ kapıyı geçen İLK aile**, sonra düştü. `purged_cv` **0,534 PASS** (bu kapıyı geçen ilk aday) · `gate_time` **PASS** (E −0,0030→+0,0068, Welch t **3,73**, seçicilik %21,0) · `gate_symbol` **PASS** (t=2,07) · `dsr` **0,8558 FAIL** (n_trials=36) · `exec_lag_1` **FAIL** — kapı-E +0,0068→−0,0014, t 3,73→**0,40** = **ACI-7 ALTINCI kez**. 22.386 satır / 577 sembol / 2023-02→2026-05. **BULGU-55-1: defter ilk kez bir KAPININ verdiktini ÇEVİRDİ** — aynı seride DSR n_trials=4'te 0,9845 (PASS), 8'de 0,9600 (PASS), 36'da 0,8558 (fail). Ailenin verdikti yine de exec_lag'den FAIL'di. Aile bilerek KESİTSEL kola daraltıldı: piyasa-geneli takvim etkileri piyasa-nötr fazlalıkta tanımı gereği ölçülemez. Detay: §31.2/31.3 |

## Denenen-aday sayacı (çoklu-test muhasebesi)

**Bu sayaç artık ELLE tutulmuyor — MAKİNEDE tutuluyor** (`research/edge_lab/ledger.py`,
defter: `research/edge_lab/kosu_defteri.jsonl`, sürüm kontrolünde):

```bash
python -m research.edge_lab           # koşulan aileler, verdiktler, sıradaki n_trials
```

Bu sayaç dekoratif değildir: her yeni aday, "şans eseri kenar" olasılığını artırır
(plan §WP-E risk 8). `run_candidate` DSR'yi eşik-taramasının ima ettiği `n_trials`'la
deflate ediyordu ama bu YALNIZ tek adayın içindeki taramayı sayıyordu; **aileler arası**
çoklu-test yalnız bu markdown satırında yaşıyordu — yani 1. adayla 10. adaya AYNI çıta
uygulanıyordu. Artık deflasyon **defterdeki ayrık aday sayısıyla ÇARPILIR**:

    n_trials = (adayın kendi eşik taraması: 4) × (defterdeki ayrık aday, bu koşu dahil)

Sayım birimi AİLEDİR: aynı adayı yeniden koşmak (hata düzeltme, kapı sıkılaştırma
yeniden-ölçümü — `*_TUR24GATE`/`*_bulgu25gate` gibi) yeni bir ARAMA değildir ve sayacı
şişirmez; defter yine de her koşuyu kaydeder (denetim izi). Boş defterde formül eski
sabitin ta kendisidir (1×4) → geçmişte koşulan üç ailenin verdikti DEĞİŞMEZ.

**Ölçüldü** (n=20.000 kapı-sonrası işlem, DSR=0,95 eşiği için gereken `SR_fazlalık`):

| n_trials | 4 (eski sabit) | 8 | 12 | **16** (3 aile + yeni aday) | 40 |
|---|---|---|---|---|---|
| gereken SR | 0,01907 | 0,02195 | 0,02341 | **0,02437** (+%27,8) | 0,02712 |

**Tur-86 itibarıyla defterde 11 ayrık aile var** (Tur-74 `vixts_v1`, Tur-86 `rmom_v1`) →
sıradaki adayın deflasyonu `12×4 = 48`'dir (`python -m research.edge_lab` güncel sayıyı
basar). VRP'nin hata-düzeltme sonrası yeniden koşusu sayacı ŞİŞİRMEDİ (aynı aile; defter
sözleşmesi). Ön-koşuldan (`veri_kapsami`) düşen koşu deftere YAZILMAZ — ölçülmemiş bir
aday bir arama değildir (Tur-86'nın ilk denemesi böyle düştü, bkz. #11).
**Tur-88 (2026-09-23):** BULGU-86-4 için yedi aile (vixts · insider · vrp · optskew · seasonality · shortflow ·
babmacro) mevcut satırları KARAR GÜNÜ S&P 500 üyeliğiyle süzülerek AYNI adlarla yeniden koşuldu (+7 defter satırı,
sayaç şişmedi). Hiçbir hüküm değişmedi; `optskew` kapsam ön-koşulundan HÜKÜM YOK, `rebalance` bilerek dışarıda.
Ayrıntı: `docs/TUR88_RAPORU_2026-09-22.md` §4c.

Çıta logaritmik büyür: arama sürdükçe sıkışır ama asla geçilemez hâle gelmez. Sayaç
büyüdükçe tek bir PASS'in kanıt değeri DÜŞER; 5-6 aileden sonra gelen ilk PASS'e,
birinciye gösterilenden çok daha fazla şüpheyle bakılmalıdır.

## Veri genişletme önceliği (harness'a beslenecek yeni kaynaklar)

| Kaynak | Ne getirir | Durum/maliyet |
|---|---|---|
| Sharadar PIT fundamentals | Survivorship-dertsiz temel veri → PEAD/değer/kalite ailelerinin GERÇEK testi | Ücretli; karar kullanıcıda |
| Opsiyon zinciri (IV) | VRP, put/call oranı, vade yapısı | Alpaca/ücretsiz kaynak araştırılacak |
| FRED makro | Rejim-koşullu aktivasyon değişkenleri | Ücretsiz; anahtar açık madde |
| Short interest + insider | Kalabalık/bilgili-taraf sinyalleri | Ücretsiz gecikmeli sürümler var |
| Kazanç çağrısı metinleri | LLM-tabanlı ton/belirsizlik skoru | Ücretsiz transkript kaynakları; `nihai/ai/text_intel.py` köprüsü hazır |

## Kullanım

```python
from research.edge_lab import run_candidate

verdikt = run_candidate(
    name="overnight_intraday_v1",
    dataset_csv="data/edge_lab_overnight_v1.csv",
    feature_cols=["overnight_ret_z", "gap_pct", "prev_range_pct"],
    label_col="label",
    ts_col="ts",
    symbol_col="symbol",
)
print(verdikt["verdict"])   # ASLA "kenar var" yazmaz
```

`run_candidate` koşuyu DEFTERE kendi yazar (çoklu-test sayacı için); `ledger_path=` ile
başka bir defter, `record_run=False` ile yalnız-okuma (keşif koşusu) seçilebilir.
Verdikt dict'i ayrıca `prior_runs` taşır: bu aile daha önce kaç kez koşuldu/FAIL etti.

Her koşu sonrası bu tablodaki `Durum` kolonu ELLE güncellenir (otomatik yazım YOK —
insan onayı gerektirir, WP-A `agent.plan()`'ın `edge_harness`/`verdikt_ozeti` görev-tipleri
bu güncellemeyi ÖNERİ olarak üretebilir, otomatik commit ETMEZ).
