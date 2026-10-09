# SONUÇ — Endeks rebalans / akış olayları pilotu (edge_lab BACKLOG #4) · 2026-08-10

**Verdikt: `KAPI(LAR) GEÇİLEMEDİ — kenar YOK / dayanıklı değil`** (5 kapıdan 1'i FAIL).
Ham çıktı: `data/verdict_rebalance_v1.json` · Veri: `data/edge_lab_rebalance_v1.csv`
(728.978 satır) · Üretici: `research/edge_lab/build_rebalance_dataset.py` ·
Testler: `tests/test_edge_lab_rebalance.py`.

---

## 0. Tek cümlelik özet

Aday, `purged_cv`'de yazı-turaya çakılıyor (**OOS-AUC 0,507**) ama diğer **dört kapıyı
geçiyor**; tanı koşusu bu çelişkinin kaynağını gösterdi: kapı endeks-olay özelliklerine
**hiç bakmıyor** (tutulan↔elenen olay mesafeleri birebir aynı), gördüğü şey bir **takvim
filtresi**, ve geçtiği `dsr` kapısı bu veri türünde **ayırt edici değil** — aynı DSR=1,0'ı
kapının ELEDİĞİ satırlar ve RASTGELE bir alt küme de üretiyor.

---

## 1. Koşudan ÖNCE yazılmış beklenti (kayıt için)

Kaynak: `build_rebalance_dataset.py` modül docstring'i, koşudan önce yazıldı.

> **Dürüst ön-beklenti: DÜŞÜK.** Petajisto (2011) ve sonrası literatür 2000'lerden
> itibaren ekleme etkisinin büyük ölçüde söndüğünü ölçüyor (arbitrajcıların olayı önceden
> fiyatlaması + endeks fonlarının uygulama esnekliği). H2 için de takvim etkileri kamuya
> en açık, dolayısıyla en çok arbitrajlanmış anomali sınıfıdır.

**Beklenti tuttu.** (Lead-lag turunda "ACI-7 riski yüksek" ön-notunun doğru çıkması gibi;
koşudan önce yazılan beklenti, sonradan üretilen rasyonalizasyondan daha değerlidir.)

---

## 2. Kurulum ve veri

| | |
|---|---|
| Fiyat verisi | Alpaca SIP **günlük**, `ALPACA_HIST_ADJUSTMENT=all` (bölünme/temettü DÜZELTİLMİŞ) |
| Pencere (istenen) | 2021-01-01 → 2026-07-31 |
| Pencere (ÖLÇÜLEN, ısınma+ufuk kırpması sonrası) | **2021-02-02 → 2026-07-23** |
| Evren | **607 sembol** = `sp500_as_of('2021-01-01')` (506) ∪ pencerede eklenen 101 sembol |
| Survivorship | 506 temel sembolün **90'ı bugün endekste YOK**; ayrıca panelde 65 çıkarılmış isim satır üretti → **survivorship-temiz** (ACI DERS #1) |
| Veri alınamayan sembol | **0** |
| Yetersiz geçmiş (düşen sembol) | **2** (CXO, TIF — 2021 başında delist) |
| Üretilen sembol | **605** |
| **Satır** | **728.978** (günde ort. 531 sembol × 1.374 işlem günü) |
| Taban oran (label=1) | **0,5268** |
| Ortalama `pnl` | **+0,1980 %** (5 barlık, maliyet düşülmüş LONG getirisi) |
| Ufuk | **5 işlem barı** (K5: süpürme YAPILMADI) |
| Özellik | **8** (K6) |

**Düşürülen satırların NEDENİ (sayılarak):** panel-öncesi (K2b) **59.659** · ısınma
(20-bar yuvarlanan pencere) **11.623** · ufuk kuyruğu (etiket yok) **3.025**.

**Panel kesiti:** endeks üyesi **691.237** satır / çıkarılmış (üye-dışı) **37.741** satır.

### Gerçek olay sayısı (§26 okuma kuralı — "kaç GERÇEK gözlem üstünde?")

| | Sayı |
|---|---|
| Pencerede EKLEME olayı | **101** (101 sembol) |
| Pencerede ÇIKARMA olayı | **103** (103 sembol) |
| Ekleme penceresi satırı (`gun_since_eklendi < 60`) | **3.900** = panelin **%0,535**'i (101 sembol) |
| Çıkarma penceresi satırı (`gun_since_cikarildi < 60`) | **2.617** = panelin **%0,359**'u (65 sembol) |

**Not (ölçülmüş sınırlama):** 103 çıkarma olayının yalnız **65**'i panelde çıkarma-sonrası
satır üretebildi. Nedeni yapısal: çıkarmaların büyük bölümü **satın alma** kaynaklıdır ve
hisse yürürlük tarihinde işlem görmeyi bırakır → "çıkarma sonrası fiyat baskısı" ölçülecek
fiyat serisi FİZİKSEL OLARAK yoktur. Bu, H1'in çıkarma ayağının bu veriyle ölçülebilir
kısmının baştan sınırlı olduğu anlamına gelir ve rapora bir sınırlama olarak yazılmıştır.

---

## 3. Tasarım kilitleri ve NEDENLERİ

| Kilit | Ne | Neden |
|---|---|---|
| **K0** | `ALPACA_HIST_ADJUSTMENT=all` env'de AÇIKÇA kilitlendi | `raw` dosyalar (`SYM_1d.csv`) düzeltmesizdir; bölünme günlerinde sahte ~%50 getiri üretir (BULGU-19: GE 1:8 ters bölünme, PIT evrenin %8,5'i). Varsayıma bırakılmadı. `test_cache_yolu_all_dosyasini_gosterir` gerçek etkiyi doğrular |
| **K1** | **İLERİ-bakan olay özelliği YASAK** | Wikipedia tablosu **yürürlük (effective)** tarihi verir, **ANONS** tarihini değil; anons günler öncedir ve elimizde YOKTUR. "Kaç gün SONRA değişiklik olacak" özelliği, henüz kamuya açık olmayan bilgiyi sızdırırdı (lead-lag turundaki "lider hindsight'sız" kilidinin aynısı). Yalnız GERİYE-bakan mesafe, 60 gün üst sınırlı |
| **K1-muafiyet** | Takvim özellikleri (`ay_sonuna_gun`, `ceyrek_sonuna_gun`) **ileri-bakandır ve meşrudur** | Takvim kamuya ve önceden bilinir; ay/çeyrek sonuna kalan gün karar anında hesaplanabilir. Bu ayrım kodda yorumla, burada açıkça yazılıdır |
| **K2** | Üyelik PIT | `sp500_as_of(start)` ile TEK temel anlık görüntü + değişiklik tablosunun İLERİ sarılması (~1400 ağ çağrısı yerine 2). `test_ileri_sarma_sp500_as_of_ile_ayni_uyelik` 8 farklı tarihte eşdeğerliği kilitler — önbellek semantiği BOZMUYOR |
| **K2b** | **Evren de PIT** — sembol panele ancak endekse GİRDİĞİ tarihten itibaren alınır | "Bu hisse ileride S&P 500'e alınacak" bilgisi karar anında YOKTUR; giriş öncesi barları eklemek evren seçimine gizli look-ahead sokar (ekleme öncesi ralli literatürün en bilinen etkisi = uydurulmuş kenar üretecek tam yer). Çıkarılan isimler panelde KALIR (çıkarma karar anında bilinir + survivorship düzeltmesinin ta kendisi) |
| **K3** | Look-ahead yok | Tüm özellikler `t` kapanışına kadarki veriden; etiket/PnL yalnız `shift(-5)`. `test_gelecek_barlar_bozulunca_ozellikler_degismez` gelecek barları BOZUP hiçbir özelliğin değişmediğini, YALNIZ label/pnl'in değiştiğini kanıtlar |
| **K4** | **GERÇEK `pnl_col`** | `pnl = 100×(fwd_5b − 5bps)` yüzde cinsinden gerçek getiri; `run_candidate`'a `pnl_col="pnl"` ile verildi. Bu, gerçek `pnl_col` veren İLK adaydır → `exec_lag_1` kapısı ilk kez tam anlamıyla ısırdı (Tur-24'te düzeltilen `_shift_label_1bar` PnL kaydırması burada devreye girdi) |
| **K5** | Tek ufuk (5 bar), tek eşik | Tur-19 zaten ufuk süpürdü (H* YOK); süpürme yalnız çoklu-test yükü ekler. Eşik taraması yapılmadı — `harness` DSR'yi `n_trials=4` ile deflate ediyor |
| **K6** | 8 özellik | ~729k satırlık panelde bile özellik enflasyonu overfit üretir |
| **K7** | Tam panel (olay-pencereli alt küme DEĞİL) | ~200 olayla DSR kapısı zaten geçilemezdi; olay sayısı ayrıca ölçülüp yukarıda raporlandı |

### Şartnameden bilinçli SAPMA (koşudan ÖNCE alındı, kayıt için)

Şartname "bir sembol endekse girmeden önceki barlarda `endeks_uyesi=0`" testini istemişti.
**K2b gereği o barlar hiç üretilmez**, dolayısıyla o test yazılamazdı; yerine daha SIKI iki
kilit konuldu: `test_endekse_girmeden_onceki_barlar_panele_alinmaz` (giriş öncesi SATIR YOK)
ve `test_uyelik_pit_cikarmadan_sonra_sifirlanir` (çıkarmadan sonra `endeks_uyesi=0`). Karar
koşudan ÖNCE alındı ve sonuca göre değiştirilmedi.

---

## 4. Beş kapının sonucu (ham sayılarla)

| # | Kapı | Sonuç | Ham sayılar |
|---|---|---|---|
| 1 | `purged_cv` | **FAIL** | ortalama **OOS-AUC 0,5066** (5 fold; eşik >0,53) |
| 2 | `gate_time` | GEÇTİ | base E **+0,2514** → kapı E **+0,5280** · `n_kept` **27.003/218.694** (`keep_frac` **%12,347**; gerekli taban 1.094) · Welch **t=8,51** |
| 3 | `gate_symbol` | GEÇTİ | base E **+0,2079** → kapı E **+0,6853** · `n_kept` **18.188/223.163** (`keep_frac` **%8,150**; gerekli taban 1.116) · Welch **t=11,51** |
| 4 | `dsr` | GEÇTİ | **DSR 1,0000** (n=27.003, `n_trials`=4; eşik >0,95) |
| 5 | `exec_lag_1` | GEÇTİ | base E **+0,2506** → kapı E **+0,3271** · `n_kept` **24.220/218.512** (%11,084) · Welch **t=2,22** · **`effect_vs_lag0` = −0,2009** |

**Verdikt: FAIL** (bir kapı yeter).

`n_kept`/`keep_frac` okuması (§26 kuralı): lead-lag turunun aksine bu adayda seçicilik
patolojik DEĞİL — %8–12 tutma oranı ve 18k–27k işlem, oransal tabanın 16–24 katı. Yani
kapı simülasyonunun "PASS"i bu kez örneklem yetersizliğinden gelmiyor. Sorun başka yerde
(§5–§6).

---

## 5. `dsr` kapısının bu veri türünde AYIRT EDİCİ OLMAMASI (ölçülmüş)

ACI DERS #8: *"bir kanıt kapısı raporun geri kalanıyla ÇELİŞİYORSA (AUC~0,50 + DSR~1,0)
kapı bozuktur, sonuç değil."* Bu koşu tam o imzayı verdi, o yüzden ölçüldü. Aynı zaman
bölmesinde, AYNI DSR hesabı dört seriye uygulandı:

| Seri | n | Sharpe | **DSR** |
|---|---|---|---|
| Kapı-sonrası (harness'ın ölçtüğü) | 27.003 | +0,0913 | **1,0000** |
| **KAPISIZ base** (tüm test dilimi) | 218.694 | +0,0483 | **1,0000** |
| **Kapının ELEDİĞİ** satırlar | 191.691 | +0,0415 | **1,0000** |
| **RASTGELE** aynı büyüklükte alt küme | 27.003 | +0,0502 | **1,0000** |

**Kapının attığı satırlar da, rastgele bir avuç satır da aynı DSR=1,0'ı veriyor.** Yani bu
adayda `dsr` kapısının "GEÇTİ"si adayla ilgili **sıfır bilgi taşıyor**; ölçtüğü şey
"ABD hisseleri 2021–2026'da 5 günlük LONG tutuşta pozitif sürüklenmeye sahipti" —
zaten bildiğimiz piyasa betası.

**Mekanizma:** `_stage_dsr` yalnız kapı-sonrası seriyi alıyor ve onu **sıfıra** kıyaslıyor.
Sözde-pnl'li (±1) ilk iki adayda bu makuldü (taban ≈ 0). GERÇEK, uzun-yönlü bir hisse
`pnl_col`'unda ise taban sıfır değil, piyasa sürüklenmesidir; n=27.000 gözlemde SR=0,05
bile DSR'yi doyurur. Bu, ACI DERS #8'in **birebir aynı yapısal hatasıdır** (orada da düzeltme
"sözde-getiriyi TRİVİYAL kurala göre FAZLALIK olarak ölç" idi); burada trivial kural
"kapısız hepsini al"dır.

**Bu turda DEĞİŞTİRİLMEDİ.** Bir kanıt kapısını, adayın koşusundan SONRA elden geçirmek
sonuca göre kural değiştirmektir (Tur-23/24'te açıkça yasaklandı) — verdikt zaten FAIL
olduğu için değişiklik lehte de aleyhte de yorumlanabilirdi. §8'de öneri olarak bırakıldı.

---

## 6. Kapı NEYİ seçiyor? — endeks olaylarını DEĞİL, takvimi

Time bölmesinde kapının TUTTUĞU ve ELEDİĞİ satırların ortalama özellikleri:

| Özellik | tutulan | elenen | fark |
|---|---|---|---|
| `gun_since_eklendi` | 59,747 | 59,836 | **−0,089** |
| `gun_since_cikarildi` | 59,988 | 59,860 | **+0,128** |
| `endeks_uyesi` | 0,9785 | 0,9034 | +0,075 |
| `ay_sonuna_gun` | 7,99 | 15,56 | **−7,57** |
| `ceyrek_sonuna_gun` | 66,84 | 42,57 | **+24,27** |
| `dolar_hacim_z` | 0,166 | 0,001 | +0,164 |
| `ret_5b` | −0,037 | +0,008 | −0,046 |
| `vol_20b` | 0,0228 | 0,0206 | +0,0023 |
| *(pnl std)* | *5,783* | *5,122* | *+0,661* |

İki okuma:

1. **Olay özellikleri tutulan ve elenen tarafta pratik olarak AYNI** (60 gün sınırının
   hemen dibinde). Kapı, adayın ASIL hipotezi olan H1'i (ekleme/çıkarma akışı) **hiç
   kullanmıyor**. Standardize lojistik katsayılar da bunu söylüyor:
   `ceyrek_sonuna_gun **+0,0629**` · `ay_sonuna_gun **−0,0521**` · `ret_5b −0,0319` ·
   `vol_20b +0,0144` · `endeks_uyesi +0,0130` · `gun_since_cikarildi **+0,0059**` ·
   `gun_since_eklendi **+0,0027**` · `dolar_hacim_z −0,0017`. En büyük katsayı 0,063 —
   model neredeyse hiçbir şey öğrenmemiş, öğrendiği azıcık şey de **takvim**.
2. **Kapı daha az riskli değil, DAHA riskli seçiyor** (pnl std 5,78 vs 5,12). Beklenti
   farkı "kötüyü eleme"den değil, oynaklığı yüksek bir alt kümeyi yukarı yönlü bir dönemde
   tutmaktan geliyor.

---

## 7. Hipotezlerin DOĞRUDAN okuması (kapıdan bağımsız betimsel ölçüm)

### H1 — ekleme/çıkarma akışı (tam panel, satır düzeyi)

| Olay | Pencere | n | ort. `pnl` | taban farkı | Welch t |
|---|---|---|---|---|---|
| EKLENME (taban +0,1966%) | 0–5 gün | 359 | −0,155% | −0,352 | **−1,16** |
| | 5–10 gün | 279 | +0,542% | +0,345 | +0,81 |
| | 10–20 gün | 559 | +0,194% | −0,003 | −0,01 |
| | 20–60 gün | 2.703 | +0,592% | +0,396 | +2,98 |
| ÇIKARILMA (taban +0,1979%) | 0–5 gün | 279 | −0,419% | −0,617 | −1,73 |
| | 5–10 gün | 208 | −0,926% | −1,124 | −2,17 |
| | 10–20 gün | 406 | +0,049% | −0,149 | −0,45 |
| | 20–60 gün | 1.724 | +0,523% | +0,326 | +2,15 |

**Fiyat baskısının olması gereken yer (0–10 gün) BOŞ.** Anlamlı görünen iki hücre
(20–60 gün, t≈2–3) fiyat-baskısı penceresi DEĞİL, üstelik 8 hücre denendiği için
çoklu-test düzeltmesi altında ayakta kalmaz; ayrıca 5 barlık ÖRTÜŞEN ufuk yüzünden
satırlar bağımsız değil (aşağıya bak). Çıkarma tarafında 5–10 günde görünen −0,93%
literatürün işaret ettiği yönde ama **n=208 satır ≈ birkaç düzine olay**, ve §2'de
belirtildiği gibi çıkarmaların çoğu satın alma olduğu için ölçülebilir örneklem baştan
sakat. **Kenar olarak okunamaz.**

### H2 — takvim akışı: satır düzeyi t'ler KÜMELENME yüzünden 11× ŞİŞKİN

Satır düzeyi (naif) test "çeyrek sonuna ≤5 gün" için **t=+5,75** veriyordu
(+0,3155% vs +0,1895%, n=48.875). Ama aynı gün 531 sembolün getirisi **tek bir ortak
şoktur** — bağımsızlık birimi satır değil TARİH'tir. Doğru (tarih-kümelenmiş) test:

| Pencere | gün sayısı | ort. günlük kesit `pnl` | taban | fark | **Welch t** |
|---|---|---|---|---|---|
| çeyrek sonuna ≤5 gün | 92 | +0,3180% | +0,1877% | +0,130 | **+0,52** |
| ay sonuna ≤1 gün | 91 | +0,2880% | +0,1899% | +0,098 | **+0,41** |
| ay sonuna 1–3 gün | 89 | +0,1249% | +0,2013% | −0,077 | **−0,39** |

**t=+5,75 → t=+0,52.** Şişme çarpanı ≈ 11× ve bu tam olarak kesit-kümelenmesinin
beklenen büyüklüğü. **H2 de ölü.**

### Üyelik kesiti (survivorship kontrolü)

| | n | ort. `pnl` | kazanma oranı |
|---|---|---|---|
| Endeks ÜYESİ | 691.237 | +0,1975% | 0,5276 |
| Endeks DIŞI (çıkarılmış) | 37.741 | +0,2065% | 0,5109 |

Çıkarılmış isimler üyelerden **ayırt edilemez** durumda. Bu, PEAD §13.5'teki
("removed-since isimlerde PF 0,80 = KAYIP") tablodan farklı: burada survivorship
düzeltmesi sonucu YÖNLENDİRMİYOR, çünkü ölçülen zaten bir kenar değil.

---

## 8. `exec_lag_1` — kapı GEÇTİ ama fazlalığın %72'si buhar oldu

| | base E | kapı E | **kapı fazlalığı** |
|---|---|---|---|
| lag 0 (gecikmesiz) | +0,2514 | +0,5280 | **+0,2766** |
| lag 1 (1 bar gecikme) | +0,2506 | +0,3271 | **+0,0765** |

**Hayatta kalan fazlalık: %27,7.** Welch t de 8,51 → 2,22'ye düşüyor. Kapı teknik olarak
PASS ediyor (t>1,645) ama ACI DERS #7'nin imzası burada da var: bilginin çoğu
sinyal-sonrası İLK barda. Bu, §26'nın doğurduğu okuma kuralının genelleştirilmiş hali —
**kapı PASS dese bile sayılar okunur.** (pairs §22 · giriş kayması §23 · lead-lag §26 ·
şimdi burada = dördüncü gözlem.)

Not: bu, `harness._shift_label_1bar`'ın Tur-24'te düzeltilen PnL kaydırmasının **gerçek
`pnl_col` ile ilk sınavıdır**. Düzeltme olmasaydı bu kapı kaydırılmamış PnL'i ölçer ve
`effect_vs_lag0` ≈ 0 çıkardı — yani sahte bir "gecikmeye dayanıklı" okuması. Düzeltme
çalıştı.

---

## 9. Ölçülmüş sınırlamalar (dürüstlük kaydı)

1. **Anons tarihi yok.** K1 gereği yalnız geriye-bakan olay mesafesi kullanıldı. Gerçek
   fiyat baskısı literatürde **anons ile yürürlük ARASINDA** yoğunlaşır; bu pencere bu
   veriyle GÖZLENEMEZ. Yani H1 burada "eksiksiz test edildi" DEĞİL, "elimizdeki veriyle
   ölçülebilen kısmı ölçüldü ve boş çıktı" denebilir. Anons tarihli bir kaynak (S&P
   duyuruları / haber arşivi) gelirse aile yeniden açılabilir — ama ön-beklenti hâlâ düşük.
2. **Çıkarmaların çoğu satın alma** → yürürlük sonrası fiyat serisi yok (103 olaydan
   yalnız 65'i satır üretti).
3. **Örtüşen ufuk.** 5 barlık etiket her satır için örtüşür; satır düzeyi t'ler bu yüzden
   şişkindir (§7'de kümelenmiş test ile gösterildi). `purged_cv`'nin purge+embargosu fold
   sınırlarını korur, fold İÇİ örtüşme standart olarak kalır.
4. **`_time_split` sınır barı.** Kesim noktası bir takvim gününü ikiye böler (o gün ~531
   satır); 5 barlık ufukla sınırda ~5 günlük örtüşme kalır. 218k'lık test diliminde
   ihmal edilebilir; `harness` DEĞİŞTİRİLMEDİ.
5. **Isınma kırpması pencere içindedir.** Barlar `--start`'tan çekildiği için ilk 20 bar
   tüm semboller için düşer (11.623 satır); tarihsel kapsam 2021-02-02'de başlar.
6. **Pencere sonu 2026-07-23** (istenen 2026-07-31 değil): `--end` gün başlangıcı olarak
   yorumlanır (bar damgaları 04:00/05:00 UTC) + 5 barlık ufuk kuyruğu düşer.
7. **`dsr` kapısı bu veri türünde ayırt edici değil** (§5) — bu adayın verdiktini
   değiştirmiyor (zaten FAIL) ama gelecekte gerçek `pnl_col` veren adaylarda **yanlış
   pozitif üretebilir.**

---

## 10. Öneri (UYGULANMADI — ayrı, adayı olmayan bir turda karar verilmeli)

`harness._stage_dsr` gerçek `pnl_col`'lu adaylarda **kapı-sonrası seriyi sıfıra değil,
KAPISIZ base'e göre FAZLALIK olarak** ölçmeli (ACI DERS #8'in `excess_pseudo_returns`
düzeltmesinin birebir muadili): `dsr(kept − base_mean)` veya kapı-sonrası ile
eşit-büyüklükte rastgele alt küme arasındaki eşleştirilmiş fark. Bu turun ölçümü
(kapının ELEDİĞİ satırların da DSR=1,0 vermesi) düzeltmenin gerekli olduğunu **kanıtlıyor.**

**Bu turda YAPILMADI**, çünkü bir kanıt kapısını adayın koşusundan SONRA değiştirmek
sonuca göre kural değiştirmektir (§26'nın önerisi de aynı gerekçeyle bir sonraki tura
bırakılmış, Tur-24'te uygulanmıştı — aynı disiplin izlendi).

---

## 11. Ne öğrenildi (kalıcı)

1. **Endeks rebalans/akış ailesi bu veriyle KAPANDI.** Ekleme/çıkarma çevresinde
   ölçülebilir kısa vadeli getiri asimetrisi bulunamadı; olayın olması gereken 0–10 günlük
   penceresi boş. Literatürün "etki 2000 sonrası söndü" bulgusuyla tutarlı.
2. **Takvim (ay/çeyrek sonu) etkisi kesit-kümelenmesi düzeltilince yok oluyor**
   (t 5,75 → 0,52). **YENİ okuma kuralı adayı:** kesitsel panellerde satır düzeyi t
   ~√(gün başına sembol) kadar şişer; bağımsızlık birimi TARİH'tir. Bu kural bu projede
   ilk kez ölçülerek gösterildi.
3. **`dsr` kapısı gerçek uzun-yönlü `pnl_col` ile DOYUYOR** (§5) — kapının ELEDİĞİ
   satırlar bile geçiyor. ACI DERS #8'in aynı yapısal hatası, farklı yerde. Gerçek
   `pnl_col` veren ilk aday bu olduğu için ilk kez görüldü.
4. **ACI DERS #7 dördüncü kez görüldü** (pairs §22 · giriş kayması §23 · lead-lag §26 ·
   burada): kapı fazlalığının %72'si 1-bar gecikmede kayboluyor — kapı PASS dese bile.
5. **Kapı PASS'lerinin bu turda seçicilik sorunu YOKTU** (%8–12, 18k–27k işlem). Yani
   §26'daki "uç-kuyruk artefaktı" açıklaması burada geçerli değil; PASS'ler gerçek ve
   büyük örneklem üstünde — ama ölçtükleri şey piyasa sürüklenmesi + takvim gürültüsü.
   **Büyük örneklemli bir PASS de kenar değildir.**
