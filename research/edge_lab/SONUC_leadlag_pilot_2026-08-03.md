# SONUÇ — Sektör-içi lead-lag pilotu (edge_lab BACKLOG #3) · 2026-08-03

**Verdikt: `KAPI(LAR) GEÇİLEMEDİ — kenar YOK / dayanıklı değil`** (5 kapıdan 2'si FAIL).
Ham çıktı: `data/verdict_leadlag_v1.json` · Veri: `data/edge_leadlag_pilot.csv`.

---

## 0. Tek cümlelik özet

Kapı simülasyonu ilk bakışta "çalışıyor" gibi görünüyor (base E −0,1095 → kapı E +0,1034),
ama bu **33.212 işlemden yalnız 29'u** üstünde duruyor ve **1 bar gecikme eklenince tamamen
yok oluyor (29 → 3 işlem)** — yani BACKLOG'un bu aile için önceden işaretlediği
**ACI DERS #7 zamanlama artefaktı** riski tam olarak gerçekleşti.

---

## 1. Kurulum

| | |
|---|---|
| Veri | Alpaca SIP saatlik, 2 yıl, 28 sembol (`.env STREAM_SYMBOLS`, ETF'ler hariç) |
| Satır | **110.704** (28 takipçi sembol, 7 sektör) |
| Taban oran (label=1) | **0,4414** |
| Etiket | takipçinin `t+1` barındaki getirisi > 5 bps |
| Özellikler | 8 adet, tümü `t` barının kapanışına kadarki veriden (bkz. modül docstring) |

**Lider seçimi hindsight'sız:** lider elle sabitlenmedi; her barda sektörün son 20 barlık
**medyan dolar-hacmi** en yüksek üyesi lider sayıldı (geriye-bakan, `t` kapanışında bilinen
bilgi). Bu yüzden liderlik zaman içinde el değiştirir — 8 üyeli yarıiletken sektöründe
**8 sembolün hepsi** bir noktada takipçi satırı üretti. "Bugün NVDA lider" demek 2 yıl önceki
veriye bugünün sıralamasını taşımak olurdu (sinsi look-ahead); test bunu kilitliyor
(`test_lider_zamanla_degisebilir_ve_geriye_bakar`).

**Aynı-gün kısıtı:** `t` ve `t+1` farklı takvim günündeyse satır düşürüldü. Aksi halde etiket
gece boşluğunu içerirdi ve BACKLOG #1'de (overnight, 5/5 kapı FAIL) ayrıca ölçülmüş BAŞKA bir
hipotezle karışırdı.

---

## 2. Kapı kapı sonuçlar

| # | Kapı | Sonuç | Ayrıntı |
|---|---|---|---|
| 1 | `purged_cv` | **GEÇTİ (kıl payı)** | ortalama OOS-AUC **0,532** (5 fold; eşik >0,53) |
| 2 | `gate_time` | GEÇTİ | base E −0,1095 → kapı E **+0,1034**, alınan **29/33.212** |
| 3 | `gate_symbol` | GEÇTİ | base E −0,1223 → kapı E **+0,0667**, alınan **30/29.617** |
| 4 | `dsr` | **FAIL** | **DSR 0,3058** (n=29, n_trials=4; eşik >0,95) |
| 5 | `exec_lag_1` | **FAIL** | 1-bar kaydırma sonrası kapıyı geçen işlem **3 < 5** → ölçülemedi |

---

## 3. Neden bu bir kenar değil — üç bağımsız gerekçe

**(a) Seçicilik patolojik.** Kapı 33.212 işlemden 29'unu tutuyor: **%0,087**. Bir "kenar"ın
işlem evreninin on-binde-birine sığması, kenar değil **uç-kuyruk örneklemi** demektir.
29 gözlemle hesaplanan +0,1034 expectancy'nin standart hatası, farkın kendisiyle aynı
mertebededir. Kapı "geçti" diyor ama söylediği şey istatistiksel olarak bir şey söylemiyor.

**(b) DSR bunu zaten yakalıyor.** Deflated Sharpe = 0,3058, kapı ≈0,95. Bu tam olarak DSR'nin
var oluş sebebidir: çoklu-test ve küçük örneklem düzeltmesinden sonra sinyal buharlaşıyor.
(ACI DERS #8'in tersi durum: orada kapı bozuktu ve rapora RAĞMEN geçiyordu; burada kapı
raporun geri kalanıyla TUTARLI davranıyor — AUC 0,53 + n=29 → DSR düşük. Sistem doğru çalışıyor.)

**(c) exec-lag-1 belirleyici darbe.** Etiketler sembol-içi 1 bar kaydırıldığında kapıyı geçen
işlem sayısı **29'dan 3'e** düşüyor — expectancy karşılaştırması yapılamayacak kadar az.
Sinyalin taşıdığı bilgi **tamamen sinyal-sonrası İLK barda**. Bu, `f_leader_ret1`/`f_gap`'in
gerçek bir "sektör-içi gecikmeli yayılım" ölçtüğü anlamına gelmez; saatlik barlarda
**bid-ask sıçraması / mikroyapı geri-dönüşü** aynı imzayı üretir.

Bu, EDGE_FINDINGS §22'deki pairs/kointegrasyon vakasının **birebir tekrarıdır**: orada da
PF 2,08 + maliyet-dayanıklı + PBO 0,004 vardı ve 1 bar gecikme PF'yi 1,03'e çökertmişti.
BACKLOG.md bu aile için "exec-lag-1 testi burada ÖZELLİKLE kritik (ACI-7 riski yüksek)"
notunu koşudan ÖNCE düşmüştü — **not doğru çıktı.**

---

## 4. `purged_cv`'nin "geçmesi" hakkında dürüst not

OOS-AUC 0,532, eşiğin (0,53) 0,002 üstünde. Bunu "model bir şey öğrendi" diye okumak
ACI DERS #3'ün ihlali olurdu: AUC bir kapı DEĞİL, ön-elemedir; karar veren kapı
simülasyonu + DSR + exec-lag'dir ve üçünün ikisi FAIL. Ayrıca 110k satırda 0,532 ile
0,500 arasındaki fark, tek bir sektörün (veya tek bir oynak dönemin) katkısıyla açıklanabilir
büyüklüktedir. **Tek başına AUC hiçbir şey ilan etmez.**

---

## 5. Ne öğrenildi (kalıcı)

1. **Sektör-içi lead-lag ailesi, saatlik barlarda ve bu evrende kapanmıştır.** PIT evrende
   tekrar denemenin önceliği DÜŞÜK: burada evren survivorship-şişkin OLMASINA RAĞMEN
   (yani kenar lehine yanlı) sonuç negatif. Şişkin evrende bile geçemeyen aday, temiz
   evrende geçmez.
2. **Kapı simülasyonunun "geçmesi" yeterli değil — `n_kept` okunmalı.** 29/33.212 gibi bir
   seçicilik, kapı PASS dese bile sonucun anlamsız olduğunu söyler. Bu, harness'ın bugünkü
   `_MIN_KEPT=5` eşiğinin bu ölçekte fazla gevşek kaldığını gösteriyor (aşağıdaki öneri).
3. **ACI-7 üçüncü kez doğrulandı** (pairs §22 · giriş kayması §23 · şimdi lead-lag).
   Saatlik/bar-düzeyi adaylarda "ilk bar" etkisi varsayılan şüphedir, istisna değil.

---

## 6. Öneri (uygulanmadı — ayrı karar)

`harness.run_candidate` kapı sonuçlarına **oransal** bir yeterlilik ölçütü eklenebilir
(örn. `n_kept / n_test` tabanı, ya da kapı-E farkı için örneklem-boyu koşullu güven
aralığı). Bugün `_MIN_KEPT=5` mutlak bir taban ve 33k işlemlik bir test dilimi için
anlamını yitiriyor. **Bu turda DEĞİŞTİRİLMEDİ** — kanıt kapısının eşiğini bir aday FAIL
ettikten SONRA sıkılaştırmak, sonuca göre kural değiştirmek olur; ayrı ve adayı olmayan
bir turda yapılmalı.

> **UYGULANDI — Tur-24 (2026-08-10), adayı olmayan turda.** `_min_kept_required(n_test) =
> max(5, ceil(0,005·n_test))` oransal tabanı + tutulan↔elenen Welch t (eşik 1,645) eklendi.
> Bu aday YENİ kapıdan tekrar geçirildi: `gate_time`/`gate_symbol` artık PASS DEĞİL
> (29 < gerekli 167 · 30 < gerekli 149). **Verdikt DEĞİŞMEDİ** (zaten FAIL'di) — kural
> yalnız istatistiksel olarak boş iki PASS'i kaldırdı. Ham çıktı:
> `data/verdict_leadlag_v1_tur24gate.json`.
