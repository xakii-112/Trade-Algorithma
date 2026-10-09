# Dış "edge bulma" önerileri — bu projedeki karşılıkları (2026-08-29, Tur-53)

Kaynak: `Trading botta edge, piyasanın rastg.txt` (bir dış LLM ile yapılan sohbet).
Soru: **"bunlardan hangileri yapılabilir?"**

Bu belge her öneriyi projenin KENDİ ölçüm kaydına karşı işaretler. Üç durum vardır:

| işaret | anlamı |
|---|---|
| **ÖLÇÜLDÜ** | bu projede survivorship-temiz/PIT veriyle koşuldu; sonuç kayıtlı. Yeniden koşmak yeni bilgi ÜRETMEZ, yalnız çoklu-test sayacını şişirir. |
| **UYGULANAMAZ** | piyasa/altyapı bu sisteme uymuyor (kripto türev verisi, market making, L2 order book). |
| **YAPILABİLİR** | veri erişilebilir, aile önceki beşten yapısal olarak farklı, kanıt kapısından geçirilebilir. |

Önerinin YÖNTEM kısmı (hipotez → istatistiksel test → EV → walk-forward → overfitting
uyarısı) zaten bu projenin çekirdeğidir ve **önerilenden serttir**: `research/edge_lab/harness.py`
her adayı beş kapıdan geçirir (`veri_kapsami` → `purged_cv` → `gate_time`/`gate_symbol` →
`dsr` → `exec_lag_1`), DSR **program düzeyinde** deflate edilir (`kosu_defteri.jsonl`'deki
ayrık aile sayısıyla) ve verdikt metni yapısal olarak "kenar var" YAZAMAZ. Yani "önce
altyapı, sonra indikatör" tavsiyesi burada zaten kurulu.

---

## 1. Birinci mesaj — dört alan

| # | Öneri | Durum | Kanıt |
|---|---|---|---|
| 1 | Momentum / trend / breakout / volatilite genişlemesi | **ÖLÇÜLDÜ — kenar YOK** | §1-§9 (filtre/kip/çıkış turları), §13.3 kesitsel momentum (survivorship-şişkin), §23 (TÜM ufuklarda H* yok) |
| 2 | Mean reversion (aşırı hareket, VWAP sapması, likidite süpürmesi sonrası) | **ÖLÇÜLDÜ — kenar YOK** | §9.3 `trend_pullback` çıkış gridi (IS kazancı OOS'ta çürük), §22 pairs/kointegrasyon mean-reversion'ı (PF 2,08 → 1 bar gecikmede 1,03) |
| 3 | Mikroyapı — order-book imbalance, trade flow, funding, OI, likidasyon | **BÖLÜNÜR** — kripto kolonları UYGULANAMAZ (bu sistem ABD hissesi/opsiyonu tarar; funding/likidasyon/perp-OI yok, L2 defteri ücretsiz değil). Ölçüm ARAÇLARI zaten var: `nihai/ai/microstructure.py` (Roll · Corwin-Schultz · Kyle λ · Amihud · VPIN · TSRV · HAR-RV). **Trade flow'un ABD hissesindeki ücretsiz karşılığı VARDI ve BU TURDA KOŞULDU** → BACKLOG #8, aşağıda |
| 4 | Rejim tespiti ("özellikle ciddiye al") | **KISMEN ÖLÇÜLDÜ** — §12 VIX rejim kapısı: *marjinal ve kırılgan*. Altyapı hazır: `nihai/data/cboe_data.py` (VIX/VIX9D/VIX3M/VVIX/SKEW + vade yapısı, 1990→), `nihai/ai/regime_hmm.py` (NEDENSEL filtre), `observatory.py` gölge kolonları. Rejim bir sinyali kurtarmadı çünkü **kurtarılacak sinyal yok** (ACI DERS #5) |
| — | "AI'yi fiyat tahmini değil olasılık tahmini için kullan" | **ÖLÇÜLDÜ — zaten öyle** | meta-labeling P(win) mimarisi (§13.2). PIT'te OOS-AUC 0,51-0,52; 97k işlemlik en büyük ölçümde de kenar YOK (§21) |
| — | "Overfitting en büyük tuzak; train/val/OOS/paper" | **ZATEN KURULU ve DAHA SERT** | purged+embargo K-fold, iki bölme (zaman+sembol), DSR, PBO/CSCV (`research/pbo.py`), `exec_lag_1`, PIT evren, canlı `prediction_tracker` |

## 2. İkinci mesaj — sekiz alternatif

| # | Öneri | Durum | Kanıt |
|---|---|---|---|
| 1 | Anomali arama (büyük mum sonrası, hacim patlaması, OI uyumsuzluğu) | **ÖLÇÜLDÜ (hisse tarafı)** — hacim-teyitli kırılım §11 yan bulgusu + canlı `detect_vol_breakout` kipi; §23 tüm ufuklarda kenar bulamadı. OI/likidasyon kolonları kripto → UYGULANAMAZ |
| 2 | Cross-market / gecikmeli ilişki (BTC→ETH benzeri) | **ÖLÇÜLDÜ — kenar YOK** | §26 sektör-içi lead-lag: iki kapı PASS ama %0,087 seçicilik, `exec_lag_1`'de 29→3 işlem (ACI-7 üçüncü kez) |
| 3 | Statistical arbitrage / pairs / kointegrasyon | **ÖLÇÜLDÜ — kenar YOK** | §22: PF 2,08 · maliyet-dayanıklı · PBO 0,004 → 1 bar gecikmede PF 1,03. Kenarın %80-95'i ilk bardaydı |
| 4 | Market making | **UYGULANAMAZ** | sistem **sinyal-only**: emir göndermiyor, maker/taker yapısı yok, kotasyon-koyma altyapısı yok. Perakende hesapta adverse selection + envanter riski + latency bu yolu kapatır |
| 5 | Rejim değiştiren strateji (TREND/RANGE/HIGH-VOL) | **KISMEN ÖLÇÜLDÜ** | §12 + `regime_pretest` (`docs/REGIME_PRETEST_2026-07-18.md`: pencere TEK REJİM). Rejim yönlendirici ancak yönlendirilecek iki iyi strateji varken anlamlıdır |
| 6 | Event-driven (kazanç, akış olayları, ekonomik veri) | **ÜÇ AİLE ÖLÇÜLDÜ — üçü de kenar YOK** | PEAD §13.5/§19 (survivorship artefaktı) · endeks rebalans §27 (`purged_cv` 0,5066) · Form 4 insider Tur-52 (**5/5 kapı FAIL**). Token unlock / ETF haberi / borsa listeleme = kripto veya haber-akışı → bu evrende UYGULANAMAZ |
| 7 | Meta-strateji (üst modelin hangi strateji avantajlı dediği) | **ÖLÇÜLDÜ — kenar YOK** | meta-labeling §13.2/§16/§20/§21 + `research/agreement.py` mutabakat harness'ı. Canlı iç AI da aynı yeri gösteriyor: 18.099 çözülmüş tahminde her yönün %95 CI tabanı 0,5'in ALTINDA → motor %100 ABSTAIN (Tur-38) |
| 8 | **Execution edge** ("en sık gözden kaçan") | **DOĞRULANDI — bu projedeki TEK ölçülmüş gerçek kazanç buradan çıktı** | §23 giriş kayması keşfi (+2,83 isabet puanı) ve **Tur-39**: sinyal fiyatı bayattı → işlemlerin %46'sında MFE=0,00, `GAP_STOP` sınıfı **68/68 aleyhte** / −299.438 $ (sistemin TÜM net zararından büyük). `reanchor_to_live()` + dakika-tabanlı zaman stopu + seans-sonu düzleşme. Ayrıca `exec_lag_1` kapısı bu içgörünün kanıt-kapısı hâlidir: **beş ailede** kenarı öldüren şey tam da execution zamanlamasıydı |

---

## 3. Karar: bu turda ne YAPILDI

Listenin ölçülmemiş tek maddesi **"trade flow / order-flow imbalance"**in ABD hissesindeki
karşılığıydı. Kripto order-book'u yok ama **FINRA günlük konsolide açığa satış hacmi** var:
ücretsiz, anahtarsız, gün başına ~12.000 sembol, 2009'dan bugüne.

* Literatür: Boehmer-Jones-Zhang 2008 (JF) · Diether-Lee-Werner 2009 (RFS) — *"açığa satanlar
  ortalamada bilgilidir"*, kesitsel akış sinyallerinin en güçlü desteklenenlerinden.
* Önceki beş aileden YAPISAL olarak farklı: olay (kazanç/rebalans/insider) veya fiyat türevi
  (lead-lag/VRP) değil, **KONUMLANMA AKIŞI**.
* Kod: `research/edge_lab/build_shortflow_dataset.py` · kilit: `tests/test_tur53_shortflow_veri_seti.py`

**SONUÇ: 5/5 kapı FAIL — kenar YOK** (147.318 satır / 593 sembol / 2021-03-31→2026-06-23,
1.378 işlem gününün tamamı, eksik gün 0). `purged_cv` **0,509** · her iki kapı simülasyonunda
tutulan işlem **0** · `dsr` hesaplanamadı · `exec_lag_1` de 0. Sebep uç-kuyruk örneklemesi
DEĞİL: model olasılıkları [0,4214·0,5559] aralığında sıkışıyor (taban 0,4850) ve akış
özelliklerinin standardize katsayıları 9 özelliğin 6./7. sırasında, hepsi ≤0,0225.
Koşulsuz izde **literatürün işareti yok** (short bacak +0,0173% / long −0,0153% = ikisi de
beklenenin TERSİ ve gürültü). Detay + BULGU-53-1: `docs/EDGE_FINDINGS.md` §29 ·
`docs/TUR53_RAPORU_2026-08-29.md`.

## 4. Kuyruğun TAMAMI koşuldu (Tur-54 · Tur-55)

| aday | sonuç |
|---|---|
| **Opsiyon çarpıklığı + parite sapması** | **5/5 kapı FAIL** (Tur-54). 10,2 GB indirildi, 116,5M satır tarandı. Verdikt 2024-10→2026-06'yı kapsar (BULGU-54-1: sağlayıcı kapsaması 2024'te sıçrıyor). §30 |
| **FRED makro rejim** | Anahtarsız köprü kalıcı modüle alındı (`nihai/data/fred_data.py`) ve **BAB × makro rejim** ailesinde kullanıldı: **4/5 kapı FAIL**, işaret bu dönemde TERS. §31.1 |
| **Mevsimsellik** | **ÜÇ kapıyı geçen İLK aile**, sonra `dsr` (0,8558) ve `exec_lag_1`'de düştü = **ACI-7 altıncı kez**. §31.2 |
| **13F kurumsal kümelenme** | **ÖLÇÜM YAPILAMADI** — şemada ticker yok; isim eşlemesi ölçüldü ve bire-bir değil (tam eşleşme %30,7; ticker'ların %69,4'ü birden fazla CUSIP alıyor). "Kenar yok" DEĞİL, "ölçüm yok". §31.4 |

**Yan bulgu (BULGU-55-1):** kenar-keşif defteri ilk kez bir KAPININ verdiktini çevirdi.
Mevsimsellik serisinde DSR eski sabitle (`n_trials=4`) **0,9845 = PASS**, program-düzeyi
sayaçla (36) **0,8558 = fail**. Ailenin verdikti yine de `exec_lag_1`'den FAIL'di, ama
sayacın dekoratif olmadığı ilk kez ölçüldü.

**Sayaç:** defterde artık **9 ayrık aile** var → sıradaki adayın deflasyonu `10×4 = 40`.
Arama sürdükçe çıta sıkışır; bundan sonra gelecek ilk PASS'e çok daha fazla şüpheyle
bakılmalıdır.

## 5. Dürüst özet

Dış öneri listesinin 12 maddesinden **7'si bu projede zaten PIT veriyle ölçülmüş ve kenar
bulunamamış**, 3'ü bu piyasa/altyapı için uygulanamaz, 1'i (execution) projenin tek doğrulanmış
kazancını zaten üretmiş, 1'i (trade flow) ölçülmemişti ve ölçüldü. Ardından sıradaki üç aday
da koşuldu: ikisi düştü, biri (13F) veri ön-koşulu yüzünden ölçülemedi. **Toplam dört yeni
aile, dört FAIL.** Yani liste yanlış
değil — **bu projeden geride**. Darboğaz fikir sayısı değil, hâlâ ACI DERS #5'tir: *OOS-kenarlı
baz sinyal yokluğu*. İleriye giden iki gerçek yol değişmedi: (a) daha iyi VERİ (Sharadar PIT
fundamentals · opsiyon zinciri) ve (b) canlı paper kanıtı biriktirmek.
