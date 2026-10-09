# Pairs / kointegrasyon PIT deneyi (2026-07-18)

**RESEARCH-ONLY.** Üretim koduna dokunulmadı, commit yapılmadı, "kenar bulundu" ilan
edilmedi. Verdikt dili yalnız *kapıyı geçti/geçmedi*. Kod: `research/pairs_test.py`,
testler: `tests/test_pairs_math.py`. Koşu: `python -m research.pairs_test --grid`.

## TL;DR — verdikt

**Pairs/kointegrasyon ailesi kanıt kapısını GEÇMEDİ. Forward-shadow adayı DEĞİL.**

- **Varyant A (statik OLS beta):** açıkça başarısız. Gerçekçi maliyette (5 bps/bacak-yön)
  PF 0.85, Sharpe −0.59, expectancy −7.9 bps/işlem. Üstelik önceki üç ailedeki (PEAD,
  meta-labeling, momentum) **aynı survivorship deseni** var: removed-since PF 0.60 «
  survivor PF 0.89.
- **Varyant B (Kalman dinamik beta):** ön-kayıtlı lag-1 konvansiyonunda **çarpıcı ham
  sayılar** üretti — PF 2.08, Sharpe 1.03, maliyet-dayanıklı, **removed-since PF 2.13 ≈
  survivor 2.07** (survivorship deseni YOK!), PBO 0.004. Bu, projede survivorship'e
  takılmayan İLK aileydi. AMA:
  1. **Deflated Sharpe (DSR) zaten zayıf:** 0.05–0.12 (< 0.5). 12 deneme için beklenen
     şans-maksimumunu aşamıyor.
  2. **BELİRLEYİCİ:** kenar tamamen **sinyal-sonrası İLK bara** bağlı. Uygulamaya **1 bar
     daha gecikme** eklenince PF **2.08 → 1.03**'e (başabaş), expectancy +58.3 → +1.8 bps'e
     çöküyor. Bu, dayanıklı kointegrasyon dönüşü değil, **1-günlük kısa-vade reversal /
     bid-ask sıçraması artefaktı** imzasıdır.

**Yeni ACI DERS:** ham PF + Sharpe + maliyet-dayanıklılık + survivorship-paritesi + düşük
PBO'nun HEPSİ olumlu görünüp yine de bir **uygulama-zamanlaması artefaktı** olabilir.
Ayırt eden test: **uygulama-gecikmesi (exec_lag) duyarlılığı**. Bu turda script'e kalıcı
eklendi (`--exec-lag`, `--grid` çıktısında "ZAMANLAMA DAYANIKLILIĞI" paneli).

---

## 1. Yöntem

Fikir: iki kointegre hissenin fiyat-spread'i ortalamaya döner (istatistiksel arbitraj).
Bu, fiyat-gösterge zamanlamasından (EDGE_FINDINGS §6) ve kesit-momentumdan (§13.3) FARKLI
bir kaynak — projedeki dördüncü bağımsız aile adayı.

**Walk-forward (OOS yapısı):** formation = 252 işlem günü → takip eden trade = 63 gün;
63'er gün kaydır (örtüşmeyen trade pencereleri). Örtüşmeyen trade pencereleri + her
pencerede formation-only seçim = OOS; bu tasarımda **purged-CV'nin karşılığı budur**
(formation ve trade zaman-ayrık; seçim asla trade verisini görmez). 16 pencere.

**Evren (survivorship-temiz PIT):** her formation başında S&P 500 üyeliği
`nihai/data/pit_universe` ile o TARİHTEKİ haliyle kuruldu (sonradan çıkarılan/satın-alınan/
iflas eden isimler dahil). `_current_and_changes` tek Wikipedia taramasıyla önbelleğe alınıp
`members_as_of` (o modülün rewind mantığının birebir aynası) ile pencere-başı as-of üyelik
hesaplandı. as-of = **formation başlangıç tarihi** (look-ahead yok).

**Aday çift seçimi (yalnız formation verisi):**
1. PIT üye ∩ formation penceresi TAM dolu (252 non-NaN kapanış) semboller.
2. Log-fiyat GETİRİ |korelasyon| en yüksek ~500 çift (ön-eleme, runtime tavanı).
3. Her aday: Engle-Granger `statsmodels.tsa.stattools.coint` (log fiyatlar) → p-değeri.
4. p < eşik olanlardan p'ye göre en iyi 20 çift/pencere.

**Hedge oranı — iki varyant, AYRI:**
- **(A) statik:** formation-OLS `log_a ~ β·log_b + α`; β işlem boyu sabit.
- **(B) Kalman:** durum [β, α] random-walk (saf numpy); gözlem `log_a_t = β_t·log_b_t + α_t
  + gürültü`. β0/α0 formation-OLS'ten; formation+trade warmup, trade z'si nedensel.

**Sinyal:** statikte z = (spread − μ_form)/σ_form (**trade'de yeniden ölçekleme YOK** =
look-ahead yok); Kalman'da z = innovation/√S. Giriş |z|≥2.0, çıkış |z|≤0.5, stop |z|≥3.5,
time-stop pencere sonu. Dolar-nötr 1:|β| bacaklar (β girişte sabitlenir). Sinyal t
kapanışında, uygulama t+1 kapanış-to-kapanış (same-bar fill YOK).

**Maliyet:** bacak-yön başına 5 bps (çift round-trip = 4 bacak-yön ≈ 20 bps); 0 ve 10 bps
duyarlılığı da raporlandı.

**Veri:** `data/alpaca_hist/{SYM}_1d.csv` (Alpaca SIP feed, delist edilenler DAHİL),
doğrudan okundu (ağsız, cache). **Günlük** barlar. Panel: **507 sembol × 1261 bar,
2021-07-01 → 2026-07-10** (~5 yıl). NOT: görev "≈2016→2026" öngördü ama Alpaca ücretsiz
SIP'te günlük geçmiş 2021-07'de başlıyor → gerçek pencere ~5 yıl / 16 pencere (görevin
tahmini ~40 değil). Dürüstçe belirtilir: örneklem KISA ve büyük ölçüde tek makro-rejim.

### Look-ahead önlemleri (sayılı)

1. Çift seçimi (corr + coint + β/μ/σ) **yalnız formation** [t0−252 : t0] verisinden.
2. as-of üyelik = formation başlangıç tarihi; gelecekteki endeks değişimleri geri-sarılır.
3. Statik z **formation μ/σ** ile ölçeklenir; trade penceresinde yeniden merkezleme/ölçekleme
   YOK (birim testte `test_statik_z_formation_parametreleri_kullanir_lookahead_yok` ile
   assert edildi: trade spread'ine sabit kayma → z = kayma/σ_form, 0 DEĞİL).
4. Kalman nedensel: z_t = predict-innovation (β_{t−1} + t'de gözlenen log_b_t, log_a_t);
   gelecekteki bar kullanılmaz. β_hold girişte dondurulur (girişteki bar verisi).
5. Sinyal t → pozisyon t→t+1 barında (lag-1); giriş barının kendi hareketi kazanılmaz.
6. Delist: çift trade penceresi iki bacağın ortak-dolu barlarına kırpılır (dropna); delist
   olan bacak time-stop ile son ortak barda kapanır (kayıp gerçekçi yakalanır).

---

## 2. Kapsama

| | değer |
|---|---|
| Panel | 507 sembol × 1261 bar (2021-07-01 → 2026-07-10) |
| Pencere sayısı | 16 (örtüşmeyen, formation 252 / trade 63) |
| Ort. evren / pencere | **465.8** sembol (aralık 436–491) |
| Aday-çiftlerde görünen removed-since sembol | 33 benzersiz |
| removed-since fiyat kapsaması (2022 as-of) | **62/69** isimde günlük CSV (~%90) |

Kritik: removed-since **fiyat** kapsaması ~%90 — PEAD'in §13.5'teki **~%60 earnings**
kapsamasından çok daha iyi (fiyat verisi Alpaca SIP'te delist sonrası korunur). Yani bu
deney survivorship'i PEAD'den daha adil test edebiliyor.

---

## 3. Izgara sonuçları — 12 konfig (2 varyant × z-giriş{1.5,2.0,2.5} × p{0.01,0.05})

`--grid` çıktısı (5 bps/bacak-yön, exit 0.5, stop 3.5, kalman-delta 1e-6):

| konfig | işlem | WR | PF | E (bps) | Sharpe(yıl) | boyutsuz-SR | **DSR** |
|---|---|---|---|---|---|---|---|
| a\|z1.5\|p0.01 | 1140 | 37.4 | 0.84 | −8.7 | −1.14 | −0.070 | 0.000 |
| a\|z1.5\|p0.05 | 1835 | 36.0 | 0.82 | −10.1 | −0.93 | −0.058 | 0.000 |
| a\|z2.0\|p0.01 | 1076 | 35.5 | 0.89 | −5.7 | −0.51 | −0.031 | 0.000 |
| a\|z2.0\|p0.05 | 1731 | 33.7 | 0.85 | −7.9 | −0.59 | −0.037 | 0.000 |
| a\|z2.5\|p0.01 | 1012 | 33.2 | 0.85 | −7.3 | −0.61 | −0.036 | 0.000 |
| a\|z2.5\|p0.05 | 1634 | 31.5 | 0.79 | −9.9 | −0.71 | −0.044 | 0.000 |
| b\|z1.5\|p0.01 | 199 | 63.8 | 1.94 | +51.3 | 1.39 | 0.065 | **0.102** |
| b\|z1.5\|p0.05 | 314 | 59.2 | 1.50 | +29.6 | 0.72 | 0.038 | 0.017 |
| b\|z2.0\|p0.01 | 98 | 66.3 | 2.92 | +103.0 | 1.32 | 0.049 | 0.000 |
| b\|z2.0\|p0.05 | 167 | 58.7 | 2.08 | +58.3 | 1.03 | 0.046 | 0.000 |
| b\|z2.5\|p0.01 | 50 | 54.0 | 1.27 | +19.8 | 0.89 | 0.026 | 0.013 |
| b\|z2.5\|p0.05 | 100 | 50.0 | 1.31 | +18.4 | 0.98 | 0.034 | 0.024 |

Varyant A'nın hepsi PF<1 (negatif Sharpe). Varyant B'nin hepsi PF>1 + pozitif Sharpe —
ama **DSR hepsinde < 0.11** (deflasyondan sonra ikna edici değil).

---

## 4. Kanıt kapısı — DSR + PBO

**DSR (deflated.py):** n_trials = 12 (denenen tüm konfig; dürüstçe sayıldı — maliyet
duyarlılığı arama boyutu değil, tekrar-ölçüm sayılmadı). var_trials = konfigler-arası
boyutsuz-Sharpe varyansı = **0.0034**, T ≈ 986 gün. Yukarıdaki tabloda: en yüksek DSR
0.102 (b\|z1.5\|p0.01). **Hiçbir konfig 0.5'i geçmiyor** → gözlenen günlük Sharpe, 12
denemenin şans-maksimumunu (E[max SR]) aşmıyor.

**PBO (pbo.py, CSCV):** T×N = günlük getiri (T=980) × 12 konfig, s=10 (252 kombinasyon):

```
PBO = 0.004   (OOS-zarar olasılığı 0.008; degradation eğim −0.46, kesim +0.089)
```

PBO çok düşük (seçim OOS'ta çökmüyor) — İLK bakışta olumlu. AMA PBO/DSR **zamanlama
artefaktını göremez** (tüm konfigler aynı 1-bar reversal'ı biner → hepsi "tutarlı"). Bu
yüzden §6 belirleyicidir.

---

## 5. Ana tablo — varyant × maliyet (z_entry=2.0, p<0.05)

| varyant | maliyet | PF | E (bps) | Sharpe(yıl) | DSR | **rem-PF** | **surv-PF** |
|---|---|---|---|---|---|---|---|
| A | 0 bps | 1.31 | +12.1 | 1.14 | 0.168 | 0.86 | 1.39 |
| A | 5 bps | 0.85 | −7.9 | −0.59 | 0.000 | 0.60 | 0.89 |
| A | 10 bps | 0.57 | −27.9 | −2.28 | 0.000 | 0.43 | 0.59 |
| B | 0 bps | 2.69 | +78.3 | 1.25 | 0.123 | 2.66 | 2.69 |
| B | 5 bps | **2.08** | +58.3 | 1.03 | 0.049 | **2.13** | 2.07 |
| B | 10 bps | 1.60 | +38.3 | 0.80 | 0.028 | 1.70 | 1.60 |

**Varyant A** klasik survivorship: rem-PF her maliyette « surv-PF (0.60 vs 0.89 @ 5bps).
Önceki üç ailenin başarısızlık modu — kenar hayatta-kalanlarda, çıkarılanlarda kayıp.

**Varyant B** survivorship desenini KIRDI: rem-PF ≈ surv-PF (2.13 vs 2.07), üstelik maliyet
10 bps/bacak-yön'e (40 bps round-trip) çıksa da PF>1. Bu, B'yi diğer üçünden ayıran ve
başlangıçta "atılım" gibi görünen bulguydu. (removed-since işlem örneklemi küçük, n=15;
ayrıca §13.5 nüansı: veri bulunan removed-since isimler çoğunlukla SATIN ALINANLAR — spread
anlaşma fiyatına yakınsar, iflaslar değil.)

---

## 6. ZAMANLAMA DAYANIKLILIĞI (BELİRLEYİCİ) — exec_lag duyarlılığı

Ön-kayıtlı konvansiyon (exec_lag=0) sinyali t kapanışında hesaplayıp t→t+1 barında uygular
= sinyal-bar kapanışında işlem girebildiğini varsayar (gerçekte imkânsız). **1 bar ek
gecikme (exec_lag=1) gerçekçi uygulamadır.** `--grid` çıktısındaki panel (z2.0, p<0.05,
5 bps):

| varyant | exec_lag | PF | E (bps) | Sharpe(yıl) | n |
|---|---|---|---|---|---|
| A | 0 | 0.85 | −7.9 | −0.59 | 1731 |
| A | 1 | 0.71 | −15.2 | −1.17 | 1687 |
| A | 2 | 0.85 | −7.5 | −0.61 | 1643 |
| **B** | **0** | **2.08** | **+58.3** | **1.03** | 167 |
| **B** | **1** | **1.03** | **+1.8** | **−0.05** | 165 |
| **B** | **2** | **1.00** | **−0.2** | **0.38** | 162 |

Diğer güçlü B konfiglerinde de aynı (ek diagnostik koşusu):
`b|z1.5|p0.01`: PF 1.98→1.19→1.06 (E +53→+11→+4 bps); `b|z2.0|p0.01`: PF 2.91→1.35→1.16
(E +103→+18→+9 bps).

**Yorum:** kenarın ~%80–95'i sinyal-sonrası İLK barda yaşıyor ve bir bar gecikince yok
oluyor. Bu, dayanıklı çok-günlük kointegrasyon dönüşünün değil, **1-günlük kısa-vade
reversal'ın** (büyük ölçüde bid-ask sıçraması + gün-sonu mikroyapı) imzasıdır. Kalman
innovation z'si tam da bir bacak o gün sıçradığında tetikleniyor ve ertesi gün kısmi
geri-dönüşü kazanıyor; gerçekçi gecikme/sürtünme altında realize edilemez.

---

## 7. Yıl-bazlı kırılım (dayanıksızlık teyidi)

Headline konfiglerin yıl-PF'leri (exec_lag=0, en iyimser hal):

| yıl | A z2.0/p0.05 (PF, n) | B z2.0/p0.05 (PF, n) |
|---|---|---|
| 2022 | 0.63 (241) | 1.57 (23) |
| 2023 | 1.18 (421) | **0.64 (52)** |
| 2024 | 0.63 (444) | 0.97 (29) |
| 2025 | 0.75 (348) | 1.85 (32) |
| 2026 | 1.10 (277) | 4.59 (31) |

B bile yıl-yıl tutarsız (2023 PF 0.64, 2024 ~başabaş; yıl-örneklemi n=23–52 çok küçük).
2026'nın PF 4.59'u kısmi-yıl + ince örneklem. exec_lag=0 zaten en iyimser haldir; §6 bunu
başabaşa indirir.

---

## 8. Kalman delta kalibrasyonu (şeffaflık)

İlk varsayılan delta=1e-4'te Kalman DEJENEREYDİ: süreç gürültüsü Q≈1e-4, gözlem gürültüsü
R≈σ²_resid≈4e-4 ile kıyaslanabilir → β aşırı uyarlanıp spread sapmalarını kendi içine
soğuruyor, innovation z_std ~0.15'e çöküyor, ±2 eşiği asla tetiklenmiyor (5 yılda 2–17
işlem). Bir tanısal koşuyla (gerçek çiftlerde z_std ölçümü) mekanizma doğrulandı ve delta,
**Q ≪ R → yavaş-değişen β** ilkesine göre **1e-6**'ya çekildi (z boyutsuz-standart ölçekte,
ön-kayıtlı ±2 eşiği anlamlı). Seçim **z-ölçeğine göre (filtre özelliği), PnL'e göre DEĞİL**
— p-hacking değil. Yine de Kalman sonuçları delta'ya duyarlıdır (bir sınırlılık); bu da
B'nin bir "kenar" değil hassas bir yapı olduğunun ek işareti.

---

## 9. Sınırlılıklar

- **Örneklem kısa:** ~5 yıl / 16 pencere, büyük ölçüde tek makro-rejim (2021 sonrası). Görev
  ~40 pencere öngördü; Alpaca ücretsiz SIP günlük geçmişi 2021-07'de başladığından mümkün değil.
- **removed-since işlem örneklemi küçük** (headline'da n=15); ayrıca veri bulunan çıkarılan
  isimler yanlı (satın alınanlar korunur, iflaslar çoğu kez eksik).
- **Toplam-getiri metriği** her zaman-yeniden-yatırılan aktif-çift-ortalaması varsayar
  (nakit sürüklemesi modellenmez) → gerçek fon getirisini abartır; Sharpe daha anlamlı.
- **Slipaj sabit-bps** modeli; gerçek spread/etki removed-since ve orta-kap isimlerde daha
  yüksek olurdu → B'nin zaten kırılgan kenarını daha da aşındırır.
- Coint eşiği çoklu-test için düzeltilmedi (Engle-Granger p'leri; Bonferroni yok) — bilinçli,
  çünkü seçim OOS trade'i etkilemiyor ve nihai kapı DSR/PBO/exec_lag.

## 10. Sonuç ve öneri

Pairs/kointegrasyon ailesi, PIT + survivorship + maliyet + **zamanlama** kapısını GEÇMEDİ.
Varyant B başlangıçta umut vericiydi (survivorship'i kıran ilk aile) ama kenarı gerçekçi
uygulama gecikmesinde çöken bir 1-bar mikroyapı reversal'ıdır. **Forward-shadow adayı
DEĞİL** (canlı gölge izlemeye değmez; exec_lag=1'de başabaş).

Bu, projenin beşinci "kenar YOK" ailesidir (PEAD, meta-labeling, momentum, PEAD-kesit-ML'in
ardından) ama YENİ bir başarısızlık modu belgeledi: **ham metrikler + PBO + survivorship-
paritesi hepsi olumlu olsa bile zamanlama artefaktı olabilir; ayırt eden test exec_lag
duyarlılığıdır.** Bu test artık `research/pairs_test.py`'de kalıcı (`--exec-lag`).

Daha ileri (opsiyonel, düşük öncelik): (a) daha uzun geçmiş (Sharadar/CRSP PIT) ile intraday
mikroyapıdan arınmış test; (b) t+1 AÇILIŞ'ta uygulama + gerçekçi etki modeli — ama exec_lag
kanıtı verildiğinde beklenti düşük.

### Tekrar üretmek için
```bash
.venv/Scripts/python.exe -m research.pairs_test --grid --max-cand-pairs 500 --pbo-s 10
.venv/Scripts/python.exe -m research.pairs_test --grid --exec-lag 1   # kenarın çöküşü
.venv/Scripts/python.exe -m pytest tests/test_pairs_math.py -q
.venv/Scripts/python.exe -m ruff check research/pairs_test.py tests/test_pairs_math.py
```
