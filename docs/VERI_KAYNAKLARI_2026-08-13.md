# Dış veri kaynakları entegrasyonu — 2026-08-13

Değerlendirilen kaynaklar: optionsdx.com · dolthub.com · cboe.com · kaggle.com

## Neden bu eksen

Projenin test ettiği ALTI sinyal ailesi de (PEAD, meta-labeling, momentum, pairs,
lead-lag, endeks rebalans) **hisse kesitiydi** ve ACI DERS #1 (survivorship) ile #7
(exec-lag) hepsini eledi. Eklenen kaynakların ortak özelliği, o iki tuzağın yapısı
gereği bulunmadığı bir eksene açılmaları: **endeks ve opsiyon/volatilite verisi.**
Endeks delist olmaz, kurucusu geriye dönük değişmez.

Bu eksen daha önce ölçülemiyordu çünkü projede **tarihsel opsiyon verisi hiç yoktu**:

| Yer | Durum (önce) |
|---|---|
| `data_provider.get_option_chain` | yalnız CANLI yfinance zinciri |
| `iv_history.py` | günde 1 satır biriktirip ~252 günde IV Rank kurmaya çalışıyor |
| `backtest.py:850` | opsiyonu `gamma=0.02` sabiti + `theta=-prem/dte` ile fiyatlıyor |
| `config.iv_rank_high/low` (50/30) | hiç sınanmamış eşikler |

## Eklenenler

| Dosya | Ne yapar |
|---|---|
| `nihai/data/dolt_options.py` | DoltHub `post-no-preference/options` istemcisi: SQL API + toplu CSV + PIT (`AS OF`) |
| `nihai/data/cboe_data.py` | CBOE VIX/VIX9D/VIX3M/VIX6M/VVIX/SKEW + vade yapısı rejim özellikleri |
| `research/backfill_iv_history.py` | `data/iv_history/<SYM>.csv`'yi geriye doldurur → Madde 16 çalışır hale gelir |
| `tests/test_dolt_options.py`, `tests/test_cboe_data.py` | 25 test, tamamı çevrimdışı |

Yeni bağımlılık **yok** (`requests` + `pandas` zaten kurulu). dolt CLI **gerekmiyor**.

## Ölçülmüş sonuç

**IV Rank (Madde 16) dirildi.** `--from-existing` ile **510/536 sembol** dolduruldu:

```
        önce                    sonra
AAPL    18 satır  NO_HISTORY    1211 satır  2019-02-09→2026-08-12  rank=38.9  güven=OK
ABBV    21 satır  NO_HISTORY    1212 satır                          rank=30.3  güven=OK
ABNB    25 satır  NO_HISTORY     584 satır  2023-11-08→             rank=15.1  güven=OK
```

Mevcut kodun **tek satırı değişmedi**: `get_iv_rank` zaten `date,iv` CSV'sini okuyordu,
sadece o dosya beslendi. Çalışan bir okuyucuyu yeniden yazmak yerine beslemek en kısa yol.

**CBOE**: 6 seri de canlı, VIX 1990-01-02→2026-08-12 (9249 satır), vade yapısı 9253 satır.

**PIT doğrulandı**: AAPL `HEAD` = 1254 satır, `AS OF 'HEAD~50'` = 1229 satır.

## Yol boyunca ölçülen operasyonel kısıtlar

Bunlar pahalı öğrenildi, kodda da yorum olarak duruyor:

1. **Branch `master`, `main` DEĞİL.** `main` → "branch not found".
2. **PK öneki dışında sunucu-tarafı sıralama sorguyu öldürür.** PK `(date, act_symbol, …)`:
   ```
   ORDER BY act_symbol, date  → 51.522 ms   (aynı 53 satır)
   ORDER BY date              →     418 ms
   ORDER BY yok               →     323 ms
   IN ('AAPL','MSFT','SPY','QQQ')  → ZAMAN AŞIMI
   ```
   Kod artık tek sembol sorar, sıralamayı istemcide yapar. `test_sunucu_siralamasi_PK_onegi_disina_cikmaz` bunu kilitler.
3. **SQL API 1000 satırda sessizce kırpar** (`status="RowLimit"`), sayfalama token'ı YOK.
   Kendi LIMIT/OFFSET sayfalamamız var; sayfalanamıyorsa SUSMUYORUZ.
4. **Deadline'a takılan sorgu KISMİ satır döndürür** (Error + 28 satır ölçüldü).
   Bu satırlar KULLANILMIYOR — sessizce eksik veri, survivorship'ten beter bir hata sınıfı.
5. **SQL API toplu çekim için yanlış araç.** 403 hız sınırı görüldü; `WHERE act_symbol=X`
   PK önekini kullanamadığı için her sorgu 4,4M satırlık tam tarama.
   **Doğru yol: toplu CSV** — `dolthub.com/csv/<repo>/<branch>/<tablo>`, tek istekte
   **217 MB / 1.898.469 satır / 67 sn**, hız sınırı tetiklenmiyor. Üretim yolu budur;
   SQL API spot sorgu için.
6. **CSV ucu GÖRELİ ref kabul etmez, commit hash'i eder.** Ölçüldü:
   `HEAD~50` → **HTTP 500**, commit hash → **200**, uydurma ref → **404**.
   404 vermesi kritik: 200 dönmesi ref'in gerçekten UYGULANDIĞININ kanıtıdır (sessizce
   HEAD'e düşmüyor). SQL `AS OF` ise göreli ref'i kabul eder. Bu fark sessiz bırakılsaydı
   `--as-of HEAD~50` bir kipte PIT, diğerinde hata olurdu → `_resolve_ref()` göreli ref'i
   `dolt_log`'dan hash'e çevirir. Canlı doğrulandı: `HEAD~50 → fr8lgjgl…`, CSV 200.

## Kapsam dışı bırakılanlar (gerekçeli)

- **OptionsDX: gerek kalmadı.** Ücretsiz katmanı EOD zincir ve elle indirme ister.
  DoltHub `option_chain` aynı alanları (bid/ask + tam greeks) 2321 sembol için
  **otomatik** veriyor. İkinci bir yükleyici yazmak YAGNI olurdu. Tek üstünlüğü
  intraday veri — o da ücretli ve canlı 1m hattına köprü değil.
- **Kaggle: kullanılmadı, kullanılmamalı.** Popüler "tüm S&P 500 hisseleri" setleri
  BUGÜNKÜ üyelikten kurulur; bu tam olarak ACI DERS #1'in artefakt üreticisidir.
- **CBOE canlı sinyale bağlanmadı.** `strategy._load_vix()` VIX'i yfinance'ten çekiyor.
  Canlı sinyal yolundaki bir girdiyi sessizce başka kaynağa taşımak ölçülmüş sonuçları
  geçersiz kılardı ("mevcut disipline saygı"). Taşınacaksa A/B ölçümüyle taşınmalı.
- **`backtest.py`'nin sentetik opsiyon fiyatlaması değiştirilmedi.** Artık gerçek
  zincir MEVCUT (`dolt_options.option_chain`), ama fiyatlamayı değiştirmek tüm geçmiş
  opsiyon backtest sonuçlarını geçersiz kılar — ayrı ve bilinçli bir tur olmalı.

## IV Rank eşiklerinin ÖLÇÜMÜ (2026-08-13, aynı gün)

Araç: `research/iv_rank_threshold_test.py` · testler: `tests/test_iv_rank_threshold.py`

**Ölçülen iddia kodun GERÇEKTE yaptığıdır.** `config.py` yorumları "IV yüksek → prim sat /
IV düşük → prim al" der, ama `strategy.signal_quality_score:910-915` eşikleri bir **orta
bant** olarak kullanır: `30 <= iv_rank <= 50` ise skora **+0.5** ("ne çok düşük ne çok
yüksek = uzun opsiyon için ideal"). Sınanan önerme budur.

Vekil ölçüt: uzun opsiyon pozitif gamma taşır → `kenar = hv_ileri − iv_current`.

### BULGU 1 — bonus bugüne kadar SABİTTİ (modelsiz, kesin)
`get_iv_rank` veri yokken `IVRankResult(50.0, "NO_HISTORY", 0)` döner ve **50.0 bandın
İÇİNDEDİR** (`30 <= 50.0 <= 50`). IV geçmişi boş olduğu için **her sinyal +0.5 bonusu
koşulsuz aldı**; bileşen sıfır bilgi taşıdı. Backfill sonrası seçicilik %26 → bonus ilk
kez ayırt edici hâle geliyor, yani bu bir CANLI DAVRANIŞ DEĞİŞİKLİĞİdir.

### BULGU 2 — bant yanlış yönde, her koşulda
Üç şişme kanalı da kapatıldı: `--non-overlap` (örtüşen ileri pencereler),
`--exec-lag 1` (ACI DERS #7), gün-bazlı eşleştirilmiş t (kesitsel korelasyon).

| W | taban | bant içi | seçicilik | gün-bazlı t | verdikt |
|---|---|---|---|---|---|
| 10 | −0.02862 | −0.03182 | %26.1 | — | KAPI GEÇİLEMEDİ |
| 21 | −0.02579 | −0.03096 | %26.3 | **−7.35** (n=916 gün) | KAPI GEÇİLEMEDİ |
| 30 | −0.02611 | −0.03155 | %26.1 | — | KAPI GEÇİLEMEDİ |
| 42 | −0.02432 | −0.03034 | %25.6 | **−3.10** (n=862 gün) | KAPI GEÇİLEMEDİ |
| 63 | −0.02299 | −0.03919 | %25.6 | — | KAPI GEÇİLEMEDİ |

Bant **her W'de tabandan KÖTÜ** ve fark anlamlı. "Orta bant uzun opsiyon için ideal"
önermesi veriyle çelişiyor.

### BULGU 3 — ilişki tümsek değil, MONOTON (W=21, gün-bazlı t)
```
[  0, 10)  −0.00940   +5.75   <- uzun opsiyon için en iyi bölge
[ 10, 20)  −0.02128   +0.78
[ 20, 30)  −0.02737   −1.61
[ 30, 40)  −0.03091   −4.78   } kodun BONUS verdiği bant
[ 40, 50)  −0.03103   −5.09   }
[ 50, 60)  −0.03062   −3.45
[ 60, 70)  −0.03446   −4.26
[ 70, 80)  −0.03222   −1.14
[ 80, 90)  −0.03914   −2.39
[ 90,100)  −0.05860   −4.63   <- prim SATIŞI için en iyi bölge
```
Yani `config.py`'nin YORUMLARI doğru, `strategy.py`'nin UYGULAMASI ters. Taban −0.026
(varyans risk primi: opsiyonlar ortalamada gerçekleşen oynaklıktan pahalı) ve W≥30'da
en düşük dilim POZİTİF kenara geçiyor — uzun opsiyonun lehine tek bölge orası.

### BULGU 4 — kaynak karışımı riski ÖLÇÜLDÜ, küçük
Backfill sonrası canlı `get_iv_rank`, yfinance ATM IV'sini Dolt `iv_current` dağılımına
karşı sıralıyor (ve `append_iv_today` aynı dosyaya yfinance satırı ekliyor). 8 sembolde
ölçüldü: oran medyan **0.98** (0.90–1.02), rank farkı birkaç puan. İki sağlayıcı ~aynı
büyüklüğü ölçüyor → karışım rank'i bozacak boyutta DEĞİL. (Backfill docstring'indeki
uyarı yerindeydi; ölçüm onu kapattı.)

### NE YAPILMADI ve neden
`strategy.py:912` **DEĞİŞTİRİLMEDİ.** Kullanıcı "ölç" dedi, "değiştir" demedi; ayrıca
ölçülen şey IV−RV farkıdır, **P&L değildir** (bid/ask, delta hedge, gamma scalping
maliyeti yok) — ACI DERS #3'ün tam tarifi: metrik iyileşmesi kenar demek değildir.
Canlı skorlamayı bir vekil ölçüte dayanarak değiştirmek bu projenin disiplinine aykırı.
Önerilen (kullanıcı kararı): bonusu orta banttan çıkarıp DÜŞÜK rank'e vermek ya da
tamamen kaldırmak; ayrıca `confidence == "NO_HISTORY"` iken bonus verilmemeli
(ölçüm yokluğu ödüllendirilmemeli — bu ayrı ve tartışmasız bir kusur).

## Açık maddeler

- ~~**`ALPACA_API_KEY` .env'de YOK**~~ → **KAPANDI (2026-08-14 doğrulandı).** Anahtar `.env`'de
  mevcut (26 krkt) ve sağlayıcı canlı bağlanıyor: `[alpaca] bağlandı (feed=iex)`, AAPL 1h
  5 bar döndü. Sessiz yfinance düşüşü artık geçerli değil. Gözlemlenebilirlik zaten vardı:
  `FailoverProvider.source_stats()` hub `/health` ucunda yayınlanıyor.
- ~~IV Rank eşikleri henüz sınanmadı~~ → **SINANDI** (yukarıdaki bölüm) ve ayrıca ölçüm
  yokluğunun bonus alması **kapatıldı** (Tur-31, `IVRankResult.measured`). Bandı ÇEVİRMEK
  hâlâ kullanıcı kararıdır (ölçülen IV−RV, P&L değil — ACI DERS #3).
- Toplu CSV önbelleği (217 MB) OneDrive altında; `DOLT_CACHE_DIR` ile dışarı alınabilir.
- **CBOE rejim özellikleri henüz hiçbir yerde TÜKETİLMİYOR.** `term_structure()` çıktısını
  `meta_features.FEATURE_NAMES`'e eklemek canlı özellik vektörünü değiştirir → kullanıcı
  kararı. **AMA `_load_vix()` göçünün önündeki engel ÖLÇÜLDÜ ve büyük ölçüde kalktı**
  (2026-08-14, aşağıdaki bölüm).
- ~~`option_chain` tablosu indirilmedi (boyutu ölçülmedi)~~ → **BOYUT ÖLÇÜLDÜ (2026-08-14).**
  `SELECT COUNT(*)` = **115.181.550 satır** (SQL API'den, tek ucuz sorgu). Akış örneklemesi
  ortalama satırı ~91 bayt veriyor → **~10,5 GB CSV**. Kıyas için aynı yöntem
  `volatility_history`'yi 1,90 M satır × ~127 B ≈ 241 MB tahmin ediyor (gerçek 217 MB →
  tahmin yöntemi güvenilir). Yani `option_chain` `volatility_history`'nin **~60 katı**.
  Ölçülen akış hızıyla indirme saatler sürer ve OneDrive altına konulmamalıdır
  (`DOLT_CACHE_DIR` şart). Karar kullanıcıda ama artık sayıyla.

## VIX kaynağı A/B — CBOE ↔ yfinance (2026-08-14 ölçüldü)

Açık madde şuydu: *"`strategy._load_vix()` hâlâ yfinance kullanıyor; kaynağı sessizce
değiştirmek ölçülmüş sonuçları geçersiz kılardı (A/B ile taşınmalı)."* A/B koşuldu —
üretimin GERÇEKTEN kullandığı yol (`strategy._load_vix()`) ile `cboe_data.get_index("VIX")`
aynı günlerde karşılaştırıldı.

| ölçüt | değer |
|---|---|
| ortak gün | **2.514** (2016-08-15 → 2026-08-13) |
| korelasyon | **0,99997448** |
| \|fark\| medyan | **0,0000** puan |
| \|fark\| < 0,01 olan gün | **2.511 / 2.514 (%99,88)** |
| rejim eşiği kararı ayrışan gün (15/20/25/30) | **0 / 1 / 0 / 0** |

0,01 puandan fazla ayrışan **tüm** günler (3 tane):

| tarih | yfinance | CBOE | fark |
|---|---|---|---|
| 2026-02-06 | 20,37 | 17,76 | **+2,61** |
| 2021-08-11 | 16,17 | 16,06 | +0,11 |
| 2021-10-01 | 21,10 | 21,15 | −0,05 |

**Okuma:** iki kaynak aynı endeksi ölçüyor; göç ölçülmüş sonuçları geçersiz kılmaz
(10 yılda tek bir eşik kararı değişiyor, o da 2026-02-06 aykırısı yüzünden). Ek olarak
CBOE **1990'a** kadar gider (9.250 gün) — yfinance yolu yalnız 2016'ya (2.515 gün), yani
göç rejim çalışmalarına **26 yıl** ek geçmiş kazandırır. Göç hâlâ yapılMADI (ayrı ve
bilinçli bir adım), ama gerekçesi artık ölçülmüş durumda.
