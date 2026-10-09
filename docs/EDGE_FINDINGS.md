# Karar/Sinyal Mantığı — Ampirik Kenar (Edge) Bulguları

**Tarih:** 2026-06-04
**Yöntem:** 16-sembollü sabit sepet (`bench.py BASKET` — AAPL, MSFT, NVDA, AMD, TSLA,
AMZN, GOOGL, META, SPY, QQQ, JPM, XOM, GE, AVGO, NFLX, COST), 1h bar.
İki periyotta (1y ve 2y) kıyas → dayanıklılık (overfit) kontrolü.
Ölçüm araçları: `bench.py`, `ablation.py`, `modes_compare.py`.

> **Metodolojik ilke:** Önce ölç → değiştir → tekrar ölç. Her "iyileştirme" iddiası
> iki bağımsız periyotta doğrulanmadan ship edilmez.

---

## 1. "Daha çok filtre = daha iyi" hipotezi → YANLIŞ

Eklenen filtreler tek tek (ablation, 1h/1y) ölçüldü. `weighted_E` = işlem-ağırlıklı
ortalama PnL ($/işlem):

| Config | trades | weighted_E | kârlı sembol % |
|---|---|---|---|
| legacy (filtreler kapalı) | 128 | **−1.77** | 31.2 |
| + quality eşiği 2.5 | 128 | −1.77 (etkisiz) | 31.2 |
| + ADX taban kapısı | 128 | −35.96 | 18.8 |
| + confluence oylaması | 117 | −43.51 | 25.0 |
| + her iki kapı | 109 | −63.33 | 12.5 |
| + HTF trend-hizalama kapısı | 95 | **−95.09** | 12.5 |
| HEPSİ birden | 103 | −60.72 | 6.2 |

**Sonuç:** Eklenen her filtre performansı düşürdü. En zararlısı HTF trend-hizalama
kapısı — yani "trendin yönünde işlem yap" kuralı bu sepette TERS tepti. Sebep: intraday
mega-cap hisselerde momentum zayıf/negatiftir; EMA50 üstündeki "hizalı" kırılımlar geç/
yorgun girişlerdir (tepeyi alır). Bu yüzden tüm bu kapılar **varsayılan KAPALI** bırakıldı
(env ile opt-in: `GATE_TREND_ALIGN`, `GATE_ADX_FLOOR`, `MIN_QUALITY`, `ENSEMBLE_MODE`).

## 2. Strateji ailesi karşılaştırması (1h/1y, tüm kapılar kapalı)

| mod | trades | weighted_E | kârlı sembol % | WR |
|---|---|---|---|---|
| **squeeze** | 78 | **+38.60** | 50.0 | 39.7 |
| event_breakout | 137 | −29.86 | 18.8 | 38.0 |
| ensemble | 107 | −43.49 | 25.0 | 37.4 |
| donchian | 119 | −43.13 | 12.5 | 38.7 |
| mean_reversion | 99 | −94.98 | 12.5 | 29.3 |
| macd | 79 | −110.71 | 6.2 | 26.6 |
| rsi_divergence | 62 | −156.91 | 0.0 | 19.4 |

## 3. Dayanıklılık (overfit) kontrolü — 1y vs 2y

| mod | 1y weighted_E | 2y weighted_E | yorum |
|---|---|---|---|
| squeeze | +38.60 | **−82.60** | 1y kârı dayanıksız (rejime özgü) → kullanma |
| **event_breakout** | −29.86 | **−26.74** | en kararlı; iki periyotta da ensemble'dan iyi |
| ensemble | −43.49 | −56.40 | kazananı kaybedenlerle seyreltiyor |

**Ana çıkarım:** squeeze'in 1y'deki +39'u tek-periyot gürültüsüydü, 2y'de −83'e döndü.
Hiçbir config iki periyotta da pozitif değil → **bu gösterge ailesinin likit mega-cap'lerde
dayanıklı bir kenarı yok.** Parametre/filtre ayarıyla bu değişmiyor.

## 4. Ship edilen tek veri-doğrulanmış karar

Ensemble her iki periyotta da kayıp ediyor; tekil **event_breakout** tutarlı şekilde
daha az kaybediyor (1y −43→−30, 2y −56→−27). Bu, iki bağımsız periyotta doğrulanan
**tek dayanıklı iyileştirme** olduğu için:

- `ACTIVE_STRATEGY` varsayılanı `ensemble` → **`event_breakout`**.
- Zararlı filtre/oylama eklentileri kodda kaldı ama **varsayılan kapalı** (opt-in env).

Bu hâlâ net-negatif. Gerçek kârlılık için gereken (gelecek iş):
1. **Farklı evren/zaman dilimi** — günlük (1d) bar veya likit-olmayan/farklı sektörler.
2. **Maliyet düşürme** — slippage/komisyon, işlem sıklığını azaltmak.
3. **Rejim-koşullu strateji seçimi** — ama rejim sınıflandırması da out-of-sample
   doğrulanmalı (bu çalışmadaki ADX kapısı gibi naïf rejim kuralları işe yaramadı).
4. Bunlar başarısızsa: sistemi **eğitim/altyapı** projesi olarak konumlandır, alfa iddiası etme.

---

## 5. İkinci tur (2026-06-05): zaman dilimi + yön + çıkış

İlk turda entry-filtre tweak'leri başarısız olunca üç yeni kaldıraç denendi.
Hepsi 16-sembol sepetinde, en az iki periyotta.

### (a) Günlük (1d) bar — momentum günlükte daha iyi mi?
| mod | 1d/2y weighted_E | 1d/5y weighted_E |
|---|---|---|
| macd | −63.90 | **+25.35** |
| event_breakout | −119.27 | −43.65 |
| donchian | −62.47 | −62.94 |

macd 5y'de +25 ama 2y'de −64 → yine periyoda bağlı. Günlük kurtarmadı.

### (b) Long-only (SHORT bastır) — sürekli-yükselen evren tezi
| mod | 1h/1y noLO → LO | 1h/2y noLO → LO |
|---|---|---|
| event_breakout | −29.86 → **−7.34** | −26.74 → **−59.31** (KÖTÜ) |
| donchian | −43.13 → **+3.28** | ? → −46.32 |
| squeeze | +38.60 → +30.01 | −82.60 → −50.72 |

Long-only 1y'de kaybı yarıladı/pozitife çevirdi AMA event_breakout'u 2y'de KÖTÜLEŞTİRDİ
(2y'de short'lar düşüşleri yakalayıp para kazanmış). Dayanıklı değil.

### (c) Çıkış ızgarası (stop/target ATR çarpanı), long-only, event_breakout
| stop/target | 1h/1y weighted_E | 1h/2y weighted_E |
|---|---|---|
| 1.5/3.0 | −7.34 | −59.31 |
| 2.0/4.0 | +13.77 | −19.04 |
| 1.5/4.5 | +28.18 | −43.39 |
| 1.5/6.0 | **+98.85** | −67.63 |

Geniş hedef 1y'yi muhteşem gösterdi (+99) ama 2y'de daha kötü. Klasik overfit imzası.

## 6. KESİN SONUÇ (iki tur sonrası)

Denenen uzay: 7 strateji × {1h, 1d} × {1y, 2y, 5y} × {filtreler, long-only, 8 çıkış combo}.
**Hiçbir konfigürasyon iki bağımsız periyotta da pozitif değil.** Her pozitif sonuç tek
pencereye özgü (overfit) çıktı. Özellikle:

- **1y (2025-26) sonuçları yanıltıcı:** o pencere güçlü boğaydı; long-only + geniş hedef
  "kârı" büyük ölçüde **piyasa betası** (al-tut da pozitif olurdu), sinyal kenarı DEĞİL.
- **2y penceresi her config'de negatif** — gerçek (rejim-bağımsız) kenar olmadığının kanıtı.

→ Bu gösterge ailesinin (EMA/MACD/RSI/Bollinger/Donchian) bu evrende (likit mega-cap)
dayanıklı alfası **yoktur**. Parametre/filtre/yön/çıkış ayarı bunu değiştirmiyor; sadece
hangi rejime fit ettiğini değiştiriyor. Tek-periyot pozitifine güvenip ship etmek = overfit
tuzağı.

**Ship edilen (yalnızca dayanıklı olan):** `ACTIVE_STRATEGY=event_breakout` (ensemble'ı İKİ
periyotta da yendi). Diğer tüm kaldıraçlar (long_only, kapılar, çıkış çarpanları, confluence)
**env-ayarlanabilir ama varsayılan standart/kapalı** — hiçbiri dayanıklı üstünlük göstermedi.

**Gerçek kârlılık için gereken (bu repo'nun ötesinde):** alternatif veri/sinyal kaynağı
(akış, order-book, alternatif veri), maliyet-aware seçicilik, veya farklı varlık sınıfı —
saf fiyat-göstergesi mega-cap'te çalışmıyor. Aksi halde: eğitim/altyapı projesi olarak konumla.

## 7. Başarı oranı (WR) yükseltme — üçüncü tur (2026-06-05)

Soru: "başarı oranını nasıl artırabiliriz?" WR mekanik olarak kontrol edilebilir ama
yanlış hedef olduğu ölçüldü.

### (a) Dar hedef / geniş stop — WR mekanik yükselir
event_breakout, 1h, her stop/target için WR + PF (wr_grid.py):
| stop/target | 1y WR | 1y PF | 2y WR | 2y PF |
|---|---|---|---|---|
| 1.5/3.0 (baseline) | 38.5 | 0.85 | 35.8 | 0.82 |
| 2.0/1.5 | 57.7 | 0.77 | 55.7 | 0.77 |
| 3.0/1.5 | 61.4 | 0.74 | 62.0 | 0.73 |
| 3.0/1.0 | **69.5** | 0.60 | **68.5** | 0.64 |

WR %38→%69.5'e çıkarılabiliyor (dayanıklı!) AMA PF 0.85→0.60'a düşüyor → daha çok kayıp.
En iyi PF en DÜŞÜK WR'lı baseline'da. **WR yükseltmek kârı bozuyor.**

### (b) Breakeven stop — WR'ı yükseltmeden PF iyileştirme denemesi
position_manager._apply_breakeven (look-ahead-güvenli; env BREAKEVEN/BE_TRIGGER_R):
| config | 1y WR | 1y PF | 2y WR | 2y PF |
|---|---|---|---|---|
| baseline | 38.2 | 0.85 | 35.8 | 0.82 |
| BE @ 0.5R | 32.3 | 0.90 | 25.0 | 0.74 |
| BE @ 1.0R | 34.6 | 0.87 | 31.1 | 0.82 |
| BE @ 1.5R | 37.5 | 0.85 | 33.3 | 0.70 |

Breakeven WR'ı YÜKSELTMEDİ (başabaş çıkış PnL≈0 → "kazanç" sayılmaz, WR düştü). 1y PF'i
hafif iyileştirdi (0.85→0.90) ama 2y'de bozdu. Dayanıklı değil → varsayılan kapalı.

### Sonuç (WR)
WR mekanik olarak ~%70'e çıkarılabilir ama **her yükseltme PF'i düşürür** — WR yanlış hedef.
Doğru hedef PF×expectancy; onu dayanıklı şekilde >1 yapan ayar bu gösterge ailesinde yok.

## 8. WR sabit + kâr oranı (PF) yükseltme — trailing stop (2026-06-05)

Soru: "WR'ı sabit tutup kâr oranını artır." Matematik: sabit WR'da PF ∝ payoff
(ort.kazanç/ort.kayıp). → kazananları büyüt (kaybı sabit tut). Mekanizma: sabit hedefi
kaldır, **trailing stop** ile kazananı koştur (position_manager._apply_trailing; env TRAIL).

event_breakout, 1h (trail mesafesi = trail_r × başlangıç-risk):
| config | 1y WR | 1y PF | 1y E | 2y WR | 2y PF | 2y E |
|---|---|---|---|---|---|---|
| baseline (sabit tgt) | 40.3 | 0.89 | −19.7 | 36.4 | 0.82 | −34.7 |
| trail 2.0R | 39.2 | 0.91 | −17.6 | 35.8 | 0.45 | −107.5 |
| trail 2.5R | 37.9 | **0.95** | **−8.5** | 32.6 | 0.65 | −73.3 |
| trail 3.0R | 37.2 | 0.93 | −13.9 | 31.2 | 0.78 | −49.3 |

**WR sabit kaldı (~38%) ve trailing PF'i 1y'de yükseltti** (0.89→0.95, E −19.7→−8.5,
neredeyse başabaş). Tam istenen davranış. AMA **2y'de PF düştü** (0.82→0.65) — yine boğa-yılına
özgü. Trailing varsayılan KAPALI; opt-in: `TRAIL=true TRAIL_R=2.5 TRAIL_DROP_TARGET=true`.

→ Genel desen değişmiyor: bir kaldıraç (entry/zaman/yön/çıkış-oranı/breakeven/trailing) 1y'yi
iyileştiriyor, 2y'yi bozuyor. Dayanıklı (iki-periyot) kâr yok. 1y "iyileşmeleri" büyük ölçüde
o pencerenin boğa rejimini yakalamak.

## Tekrar üretmek için

```bash
.venv/Scripts/python.exe -m research.ablation                                       # filtre ablasyonu (1h/1y)
.venv/Scripts/python.exe -m research.modes_compare --interval 1h --period 1y        # tüm modlar
.venv/Scripts/python.exe -m research.modes_compare --interval 1d --period 5y        # günlük
.venv/Scripts/python.exe -m research.modes_compare --period 1y --long-only          # long-only
.venv/Scripts/python.exe -m research.exit_grid --mode event_breakout --periods 1y,2y  # çıkış ızgarası
.venv/Scripts/python.exe -m research.wr_grid --mode event_breakout --periods 1y,2y     # WR↔payoff ödünleşimi
.venv/Scripts/python.exe -m research.be_test --mode event_breakout --periods 1y,2y     # breakeven etkisi
.venv/Scripts/python.exe -m research.trail_test --mode event_breakout --periods 1y,2y  # trailing (WR sabit, PF↑)
.venv/Scripts/python.exe -m research.stats --mode event_breakout --periods 1y,2y       # WR/payoff/PF/expectancy
.venv/Scripts/python.exe -m research.bench --tag X --interval 1h --period 1y        # toplulaştırılmış kıyas
```

---

## 9. Yeni GİRİŞ mantıkları + hibrit + pullback çıkış-OOS (2026-06-06)

§5-8 mevcut göstergelerin çıkış/yön/WR ayarını taradı. Bu tur **yeni giriş tetikleyicileri**
yazıp ölçtü (16-sembol sepeti, {1h,1d}×{1y,2y}, kapılar kapalı).

### 9.1 Üç yeni aday
- **trend_pullback** — yükseliş trendinde düşüşü al (breakout kovalamaz; §1 kök-nedenini hedefler).
- **tsmom** — time-series momentum (uzun SMA cross + getiri işareti).
- **ker_breakout** — Kaufman Efficiency Ratio kapılı kırılım (ADX yerine).

| mod | 1h/1y | 1h/2y | 1d/1y | 1d/2y | WR(1h) |
|---|---|---|---|---|---|
| event_breakout (ref) | −27 | −35 | −185 | −119 | ~38 |
| **trend_pullback** | −27 | −51 | **−2** | −76 | **~44** |
| tsmom | −65 | −63 | −52 | −82 | ~36 |
| ker_breakout | −43 | −61 | −216 | −127 | ~38 |

trend_pullback en iyi aday (en yüksek WR, 1d/1y ≈ başabaş) ama iki periyotta dayanıklı pozitif değil.

### 9.2 Hibrit (tsmom yön + KER kalite + event/pullback tetik) — parçalardan KÖTÜ
4-hücre ort: hybrid −83, event_breakout −97, trend_pullback −42. Katmanlı AND kapıları
WR'ı %44→%25'e düşürdü (en uzamış girişe sokuyor). **"Daha çok birleştirme = daha kötü"yü
3. kez doğruladı** (ensemble → filtreler → hibrit).

### 9.3 trend_pullback çıkış gridi — in-sample kazanç, OOS çürük
`optimize_pullback.py` (stop×target ATR, iki-periyot worst-case):
- 1h: hiçbir combo kurtarmadı (en iyi worst-case −68).
- **1d: dar stop (1.0 ATR)+uzak hedef köşesi iki periyotta da pozitif.** En iyi
  **stop=1.0/target=5.0 (RR 5:1)**: 1y +34, 2y +26 → worst-case **+26**. Bitişik bölge (tutarlı).

**Walk-forward (gerçek OOS, `wf_pullback.py`, 1d/2y, IS180/OOS60) bunu ÇÜRÜTTÜ:**
sembol-ort. OOS E **−119**, pozitif-OOS %33 (5/15), WF verimliliği **0.10**. In-sample +26
**overfit'ti** (squeeze/mean_reversion ile aynı). Uyarı: 1d OOS örneklemi ince (2-6 işlem/sembol).

### 9.4 Ship
- trend_pullback kodda kaldı (opt-in `ACTIVE_STRATEGY=trend_pullback`).
- **tsmom/ker_breakout/hybrid silindi** (dayanıklı değil; tekrar için git geçmişi).
- Varsayılanlar değişmedi (`event_breakout`, çıkış 1.5/3.0).

**§6'yı pekiştirir:** yeni giriş mantığı da dayanıklı kenar üretmedi. Kaldıraç artık mantıkta
değil — evren/enstrüman değişimi ya da eğitim/altyapı konumlaması.

### Tekrar üretmek için (§9)
```bash
.venv/Scripts/python.exe -m research.modes_compare --interval 1d --period 2y --modes trend_pullback,event_breakout
.venv/Scripts/python.exe -m research.optimize_pullback --interval 1d --periods 1y,2y   # çıkış gridi
.venv/Scripts/python.exe -m research.wf_pullback --interval 1d --period 2y             # OOS doğrulama
```

---

## 10. ALTERNATİF sinyal tipi — PEAD (2026-06-06): ilk umut verici, ama kanıtlanmadı

§1-9 hep fiyat-göstergesi (EMA/RSI/MACD/Bollinger/Donchian) ailesindeydi ve hepsi çürüdü.
Bu tur **fiyat-göstergesi DEĞİL** bir sinyal denendi: **PEAD (post-earnings drift)** —
bilanço sürprizi (yfinance `earnings_dates` Surprise%) yönünde, earnings'in ertesi barında
gir, drift'i yakala. Kural a-priori sabit (sürpriz ≥ %2); veriye optimize EDİLMEDİ → exit-grid
gibi in-sample overfit riski yok. Yalnızca günlük (1d) bar. Look-ahead güvenli (önceki barın
bildirilmiş sürprizi kullanılır). ETF'ler (SPY/QQQ) earnings yok → doğru şekilde 0 işlem.

### 10.1 Mod karşılaştırması (1d, weighted_E)
| periyot | pead | event_breakout | trend_pullback |
|---|---|---|---|
| 1y | **+38.9** | −158.5 | +4.5 |
| 2y | **+130.3** | −114.2 | −69.1 |
| 5y | **−22.9** | −8.3 | −21.2 |

1y+2y POZİTİF (önceki tüm "kazananlar" periyotlar arası işaret değiştiriyordu — bu değiştirmedi).
AMA 5y NEGATİF → 2020-21 (COVID + dev sürprizler) PEAD'i bozuyor. Rejim-hassas.

### 10.2 Walk-forward (OOS, 1d/5y, IS=252/OOS=126)
| metrik | değer |
|---|---|
| sembol-ort. OOS expectancy | **+99.86** |
| pozitif-OOS sembol | **%78.6 (11/14)** |
| OOS işlem (toplam) | ~151 |

OOS GÜÇLÜ pozitif — düz 5y backtest'in tersi. Çelişkinin mekaniği:
1. **2020-21:** WF'in ilk IS penceresi o volatil dönemi yutar, OOS sakin ~2021-26'yı test eder.
2. **Kill-switch artefaktı:** düz backtest'te "30-işlem rolling avg < 0 → giriş durdur" emniyeti,
   düşük-frekanslı PEAD'de erken kayıplarda tetiklenip sonraki iyi işlemleri kilitliyor (düz=78
   işlem, WF=~151). "5y negatif"in bir kısmı sinyal değil, emniyet etkileşimi.
3. **WF verimliliği +2.31 YANILTICI** — IS-paydası ~0 olan semboller (META eff 33.86) oranı
   şişiriyor. Güvenilir olan: %79 sembolün OOS pozitifliği.

### 10.3 Karar
**PEAD = tüm araştırmadaki ilk gerçekten umut verici sinyal** ama KANITLANMADI:
- ✅ 1y/2y + walk-forward OOS pozitif; kural optimize edilmedi (overfit riski düşük); fiyat-only
  ailesini öldüren çıtayı geçen TEK aday.
- ⚠️ 5y düz backtest negatif (rejim-hassas); örneklem ince (çeyreklik); **survivorship bias**
  yukarı-yanlı (sepet = hayatta kalanlar, PEAD'de kritik); tek-işlem varyansı yüksek (+400 ↔ −137).

**Ship edilmedi.** Kodda kaldı (opt-in `ACTIVE_STRATEGY=pead`), varsayılan değişmedi.
Doğrulama için sonraki iş: (a) daha geniş + survivor-olmayan evren (daha çok olay, bias↓),
(b) COVID/outlier dönemi ayrı ele alma, (c) varyansa göre pozisyon boyutlama, (d) kill-switch'in
düşük-frekanslı sinyalle etkileşimini düzeltme.

**Ana çıkarım:** Kaldıraç fiyat-mantığında değil, **sinyal TİPİNDE** — olay/temel-bazlı sinyal
(PEAD) ilk kez çıtayı zorladı. Yön doğru: alternatif veri/sinyal tipleri (cross-sectional
momentum, vol-risk-primi, vb.) bu sistemin gerçek bir sonraki sınırı.

### Tekrar üretmek için (§10)
```bash
.venv/Scripts/python.exe -m research.modes_compare --interval 1d --period 2y --modes pead,event_breakout
.venv/Scripts/python.exe -m research.modes_compare --interval 1d --period 5y --modes pead
.venv/Scripts/python.exe -m research.wf_pullback --mode pead --interval 1d --period 5y --is-bars 252 --oos-bars 126
```

---

## 11. Maliyet atfı — slippage'in saf etkisi + hacim teyidi (2026-06-06)

Soru: "maliyet/kâr oranı nedir; kök-neden maliyet mi kenar yokluğu mu?" Aynı sepet + aynı
sinyaller, slippage AÇIK (hisse %0.1/yön) vs KAPALI (0). İşlem kararları maliyetten bağımsız
→ fark = saf maliyet sürüklenmesi. event_breakout, hisse modu (`cost_attribution.py`).

| dilim | işlem | PnL açık | PnL kapalı | maliyet $ | maliyet/işlem |
|---|---|---|---|---|---|
| **1h/1y** | 339 | −11.006 | **+1.980** | 12.985 | $38.30 |
| 1h/2y | 192 | −11.408 | −4.670 | 6.738 | $35.09 |
| 1d/1y | 69 | −10.938 | −10.264 | 674 | $9.77 |
| 1d/2y | 123 | −14.045 | −12.653 | 1.392 | $11.31 |

**İki ayrı teşhis (önceki §6 "kenar yok" ifadesini inceltiyor):**
- **1h'te maliyet BİRİNCİL katil.** 1h/1y maliyetsiz brüt POZİTİF (+1.980) ama 339 işlem ×
  %0.1 = 12.985$ maliyet bunu −11.006'ya gömüyor → **maliyet brüt kârın ~6.6 katı.** Aşırı
  devir hızı. AMA brüt kenar dayanıklı değil (1h/2y maliyetsiz de negatif −4.670) → maliyet
  sıfırlansa bile iki periyotta kalıcı kâr yok.
- **1d'de maliyet ihmal edilebilir** ($10/işlem) ama maliyetsiz bile derin negatif (−10.264)
  → saf kenar yokluğu.

**Çıkarım:** Profit Factor ~0.85 ile uyumlu (brüt ≈ başabaş, maliyet negatife itiyor).
1h'te **az-ve-seçici işlem + hisse modu** (opsiyon %3 slippage felaket) maliyeti ciddi düşürür;
ama dayanıklı kâr için yine gerçek kenar (yeni veri/sinyal) şart. Devir hızı = 1h'in düşmanı.

### Hacim-teyitli kırılım (vol_breakout) — yan bulgu
Kırılımı yalnızca RVOL (hacim/ortalama) yüksekse al. 1h/1y −63 (event −32, KÖTÜ), 1h/2y −55
(~başabaş), **1d/2y −8.8 (event −114'e karşı çok daha iyi, %31 sembol kârlı)** — günlükte
sahte kırılımları eliyor ama hâlâ negatif. Yine "daha az-kötü, kenar değil". Kodda opt-in kaldı.

### Tekrar üretmek için (§11)
```bash
.venv/Scripts/python.exe -m nihai.execution.cost_attribution --mode event_breakout --interval 1h --periods 1y,2y
.venv/Scripts/python.exe -m research.modes_compare --interval 1d --period 2y --modes vol_breakout,event_breakout
```

---

## 12. Çapraz-varlık VIX rejim kapısı (2026-06-06) — marjinal, kırılgan

Alternatif veri: ^VIX günlük kapanışı (look-ahead-güvenli, karar gününden STRICT önce).
Tez: yüksek-korku (VIX > eşik) rejiminde yeni giriş bastır → kırılım sinyali iyileşir mi?
event_breakout üzerinde eşik taraması, gate OFF baseline vs ON (`vix_test.py`).

| dilim | baseline (OFF) | en iyi eşik | iyileşme | not |
|---|---|---|---|---|
| 1h | −59.4 | VIX>25 → −57.0 | **+2.4** | ihmal edilebilir |
| 1d | −158.5 | VIX>22 → −132.4 | **+26.2** | kayıp kırptı, hâlâ derin negatif |

**Kenar değil:**
1. İşaret oluşturmuyor, sadece kayıp kırpıyor — strateji negatif kalıyor.
2. **Eşiğe aşırı duyarlı:** VIX>20 her iki dilimde de KÖTÜLEŞTİRDİ (1d −204; bazı yüksek-VIX
   girişleri kazançlıydı, onları da eledi). "En iyi eşik" 1h=25, 1d=22 → tutarlı değer yok =
   overfit-by-threshold.
3. **Düşük-güçlü test:** yfinance 1h ≤730g; test penceresi (~2024-26) sakindi, kapı az tetiklendi
   (VIX>28'de işlem sayısı değişmiyor). Gerçek panik (2020/2022) dönemi daha adil test ederdi.

**Karar:** dayanıklı iyileştirici değil → **opt-in KAPALI** (`VIX_GATE` env; varsayılan kapalı).
§6/§11 desenini pekiştirir: tek bir kaldıraç (filtre/yön/çıkış/rejim) bazen kaybı kırpar ama
hiçbiri kenar yaratmaz; her "iyileşme" tek-pencere/tek-eşik şansı.

### Tekrar üretmek için (§12)
```bash
.venv/Scripts/python.exe -m research.vix_test --mode event_breakout --interval 1h --periods 1y,2y
.venv/Scripts/python.exe -m research.vix_test --mode event_breakout --interval 1d --periods 1y,2y
```

---

## 13. Sinyal TİPİ değişimi (2026-06-10): üç bağımsız dayanıklı kenar

§10 şunu öngördü: *"Kaldıraç fiyat-mantığında değil, sinyal TİPİNDE… alternatif veri/sinyal
tipleri (cross-sectional momentum, …) gerçek bir sonraki sınır."* Bu tur o öngörüyü test etti:
üç ayrı sinyal kaynağı kuruldu/ölçüldü; **üçü de fiyat-göstergesi-zamanlamasından bağımsız ve
üçü de §1-9'u öldüren iki-dönem/çok-parametre çıtasını geçti.**

### 13.1 PEAD yeniden-doğrulama (`pead_test.py`) — §10'u güçlendiriyor
§10 PEAD'i 1d/`modes_compare` ile ölçüp "umut verici" demişti. Bu tur izole bir harness ile
BASKET üzerinde, **al-tut (beta) kontrolüyle** yeniden ölçüldü:

| | PEAD 2y | PEAD 5y | event_breakout 2y | event_breakout 5y |
|---|---|---|---|---|
| profit_factor | **1.57** | **1.18** | 0.91 | 1.11 |
| expectancy | **+4203** | **+1553** | −912 | +1036 |
| ort. al-tut | +72% | +263% | — | — |

İki periyotta da PF>1 ve expectancy>0; event_breakout 2y'de zarar ederken PEAD kâr ediyor →
salt beta değil. §10'un 5y düz-backtest negatifi (kill-switch artefaktı) izole harness'ta
tekrarlanmadı. **Sınır:** earnings verisi yfinance-sınırlı; survivorship yukarı-yanlı; mega-cap
PEAD'in en zayıf olduğu yer (literatürde küçük-cap'te daha güçlü) → bu sonuç muhtemelen alt-sınır.

### 13.2 Meta-labeling — birincil yönü filtreleyen ikincil model (#1)
Fikir (López de Prado): birincil model (event_breakout) YÖN üretir; ikincil model "bu sinyal
kâr eder mi?" olasılığını tahmin edip düşük-olasılıkları eler. 19 look-ahead-güvenli özellik
(`meta_features.py`), nasdaq100 1h/2y = **2727 işlem**. İki bağımsız holdout + kapı simülasyonu:

| model | bölme | OOS-AUC | base PF | kapı (eşik 0.50) |
|---|---|---|---|---|
| **lojistik** | zaman | 0.56 | 0.78 | **PF 1.07, E +837** |
| **lojistik** | sembol (görülmemiş hisseler) | 0.60 | 0.82 | **PF 1.37, E +3361** |
| gbm | zaman | 0.54 | 0.78 | ≈şans, düşük eşiklerde zarar |
| gbm | sembol | 0.58 | 0.82 | PF 2.18 ama… (aşağı) |

**Lojistik DAYANIKLI:** her iki holdout'ta da event_breakout'u zarardan kâra çeviriyor; eğitim/
test AUC farkı küçük (overfit yok). **GBM OVERFIT ve REDDEDİLDİ:** eğitim AUC 0.90 vs test 0.54-
0.58; sembol-holdout'ta gösterişli PF 2.18 ama zaman-holdout'ta şans seviyesi — bir dilimde
harika, diğerinde çuvallıyor = §3/§6 overfit imzası. **GBM'in parlak sayısına kanmadık.**
Doğrulama metodolojisi bir kez daha overfit'i yakaladı.

Yorumlanabilirlik bonusu: lojistik en güçlü özellik olarak `ret_5 −0.79` öğrendi = "son 5 barda
zaten çok yükselmiş girişin kazanma olasılığı düşük" = **§1'in "yorgun breakout kovalama"
kök-nedeninin veriden bağımsız nicel keşfi.** Final model tüm veride eğitilip `data/meta_model.json`
(eşik 0.50). Kapı `generate_signal`'a bağlı, env `META_GATE` ile **varsayılan KAPALI** (opt-in
disiplini); model yoksa sessizce pasif.

### 13.3 Cross-sectional momentum (`xsect.py`) — §10'un işaret ettiği sınır (#2)
Per-symbol zamanlama yerine PORTFÖY seçimi: her rebalance evreni geçmiş getiriye (lookback-skip)
göre sırala, eşit-ağırlık top-K tut, periyodik yenile (Jegadeesh-Titman). nasdaq100 1d. Kontrol:
aynı evrenin eşit-ağırlık al-tut'u (beta) — momentum bunu aşmalı (§6 betası tuzağı).

**Parametre dayanıklılığı: edge_vs_buyhold 12/12 kombinasyonda POZİTİF** (6 param × 2 periyot):

| lookback/skip/hold/top | 2y kenar% | 5y kenar% | posWR | PF |
|---|---|---|---|---|
| 126/21/21/10 (default) | +188 | +316 | 56-60 | 2.0-3.3 |
| 252/21/21/10 | +112 | +626 | 61-66 | 2.7-5.5 |
| 126/21/21/15 | +108 | +132 | 54-57 | 1.7-2.8 |
| 63/5/21/10 | +289 | +245 | 53-61 | 1.9-3.7 |

Squeeze (§3) gibi tek-pencere artefaktı DEĞİL — her parametrede, her periyotta beta-üstü.
**AMA KRİTİK CAVEAT — survivorship bias (Madde 51):** `nasdaq100_tickers()` BUGÜNKÜ üyelik =
hayatta kalıp büyüyen kazananlar. Mutlak getiriler (+248%/+434%) ileriye-dönük gerçekçi DEĞİL,
ağır şişkin. `edge_vs_buyhold` survivorship'i kısmen izole eder (her ikisi de aynı şişkin evren)
ama büyüklük yine güvenilmez. **Güvenilir ifade: "momentum seçimi naif eşit-ağırlığı dayanıklı
şekilde geçiyor" — büyüklük değil.** Gerçek test point-in-time üyelik (Sharadar/CRSP) ister.

### 13.4 Sentez
§1-12: saf fiyat-göstergesi zamanlaması mega-cap'te kenarsız (12 tur, hepsi çürüdü). §13: sinyal
TİPİ değişince üç bağımsız dayanıklı kenar çıktı — PEAD (olay-bazlı), meta-labeling (kalite-öğrenen
filtre), cross-sectional momentum (kesitsel seçim). **§10'un tezi doğrulandı: kaldıraç fiyat-
mantığında değil, sinyal tipinde.** Üçü de bağımsız kaynaklar → birleştirilebilirler (örn. momentum
seçimi + meta-labeling filtresi + PEAD olay katmanı).

**Ortak kusur (üçü de):** survivorship bias + sınırlı/ücretsiz veri (yfinance). Üç sinyalin
GERÇEK büyüklüğünü maskeleyen tek ortak engel bu. **Sonraki kritik iş: survivorship-temiz,
point-in-time evren (Sharadar/Polygon) ile yeniden ölçüm.** İkincil: canlı paper doğrulama.
Hiçbiri canlı para öncesi kanıtlanmış sayılmaz; ama §1-12'nin "dayanıklı kenar yok" sonucundan
sonra ilk kez üç ayrı aday çıtayı geçti.

### Tekrar üretmek için (§13)
```bash
.venv/Scripts/python.exe -m research.pead_test --periods 2y,5y                       # PEAD izole + al-tut kontrolü
.venv/Scripts/python.exe -m research.build_meta_dataset --universe nasdaq100 --interval 1h --period 2y
.venv/Scripts/python.exe -m research.train_meta --model both --split both            # lojistik vs gbm, zaman+sembol holdout
.venv/Scripts/python.exe -m research.xsect --universe nasdaq100 --periods 2y,5y --top 10   # cross-sectional momentum
```

### 13.5 PEAD point-in-time (survivorship-temiz) doğrulama (2026-06-10) — kenar BÜYÜK ÖLÇÜDE survivorship'ti

§13.1/§13.4 PEAD'i "ilk dayanıklı kenar" ilan etti AMA test evreni (BASKET 16 mega-cap +
güncel nasdaq100) survivorship-yanlıydı. Bu alt-tur o yanlılığı ÖLÇTÜ: evreni BUGÜNKÜ üyelikle
değil, **test başındaki (2022-01-01) S&P 500 üyeliğiyle** kurdu (`pit_universe.sp500_as_of` —
Wikipedia değişiklik tablosunu geri-sarar; sonradan çıkarılan/satın-alınan/iflas eden 73 ismi
DE içerir). Test 2022-01-01→bugün, 1d. (`pead_pit_test.py`)

| evren | işlem | WR | PF | expectancy |
|---|---|---|---|---|
| TÜM PIT (505 sembol, çıkarılanlar dahil) | 5920 | 38.1 | **1.05** | +403 |
| hayatta-kalanlar | 5379 | 38.7 | 1.08 | +628 |
| **removed-since (bugün artık yok)** | 541 | 32.9 | **0.80** | **−1839** |

Coverage: 476/505 sembolde veri; **44/73 çıkarılan isimde earnings verisi bulundu** (541 işlem
— sağlam örneklem, "ölü ticker boş" endişesini kısmen giderdi). Ort. al-tut +46.5%.

**SONUÇ — §13.1 düzeltilmeli:** PEAD'in BASKET'teki PF 1.57'si survivorship-şişkindi.
Survivorship-temiz evrende **PF 1.05'e (başabaşa yakın) çöküyor** ve bu marjinal pozitif TAMAMEN
hayatta-kalanlardan geliyor; sonradan çıkarılan isimlerde PEAD **kaybediyor** (PF 0.80). Yani
"bugünün kazananları"nı test etmek PEAD'i ~PF 1.5 gibi gösterirken, o tarihte işlem yapsaydın
(çıkarılacak isimler dahil) gerçek sonuç ~başabaş olurdu.

**Nüans (mekanik):** removed-since negatifinin bir kısmı YAPISAL — satın-alınan isimler anlaşma
fiyatına kilitlenir (drift oynayamaz), distress isimler çöker. Saf "PEAD çalışmıyor" değil; ama
ileriye-dönük işlemde de bu isimlere maruz kalırsın → kayıp gerçek maruziyet.

**Daha geniş çıkarım (§13.4 revize):** Aynı survivorship düzeltmesi muhtemelen #1 (meta-labeling)
ve #2 (momentum) sayılarını da deflate eder — ikisi de güncel nasdaq100'de ölçüldü. PEAD'de
düşüş PF 1.57→1.05 oldu; momentum'un +316% "kenar"ı da büyük ölçüde aynı artefakt. **Net durum:
üç sinyalin de GERÇEK büyüklüğü ilk göründüğünden epey düşük.** PEAD hâlâ marjinal pozitif (ölü
değil) ama "güçlü dayanıklı kenar" iddiası geri çekilmeli. Gerçek değerlendirme için Sharadar/CRSP
(ücretli, point-in-time + earnings) şart; ücretsiz yaklaşım (Wikipedia PIT + yfinance) çıkarılan
isimlerin yalnızca %60'ında earnings verisi bulabildi.

### Tekrar üretmek için (§13.5)
```bash
.venv/Scripts/python.exe -m nihai.data.pit_universe --as-of 2022-01-01                 # PIT üyelik (çıkarılanlar dahil)
.venv/Scripts/python.exe -m research.pead_pit_test --as-of 2022-01-01 --limit 0      # survivorship-temiz PEAD
```

---

## 14. AI_trade'den çıkış/risk overlay'leri (2026-06-17): kuruldu, ölçüldü, OOS'ta kenar-katkısı YOK

Yan proje `AI_trade` (Binance botu) incelendi; varlık-bağımsız risk/çıkış fikirleri ana projeye
**uyarlandı** (kripto birimlerinden hisse R/ATR-uzayına taşınarak). Hepsi **env-kapılı, default-KAPALI**,
ek olarak `bench.py`/`exit_grid.py`/`wf_pullback.py` ile ölçülebilir. Eklenenler:
- **A1** kademeli trailing + kâr-tabanı kilidi (`TRAIL_TIERED`, `TRAIL_LOCK_*`) — kâr (R) arttıkça trail'i sıkılaştır.
- **A2** MFE give-back (`MFE_GIVEBACK_*`) — tepe lehte hareketten R-bazlı geri-veriş çıkışı (volatiliteye uyarlı).
- **A4** piyasa rejimi → risk çarpanı (`REGIME_RISK`) — benchmark GÜNLÜK trendi (SMA200) ± VIX → ×0.5/0.8/1.0.
  Backtest'te de uygulanır (`market_regime.regime_multiplier_series`, 1-bar gecikmeli, look-ahead-güvenli).

### 14.1 In-sample bench (16-sembol sepeti, 1d/5y, weighted_E / Sharpe / %kârlı sembol)
| konfig | ensemble | event_breakout |
|---|---|---|
| baseline | E −6009, Sharpe −0.68, %0 | E **+1139**, Sharpe **0.03**, %37.5 |
| +A4 rejim | E −3761, Sharpe −0.54 | E +801, Sharpe 0.03 (getiri DÜŞER) |
| +A1+A2 (hedef-kaldır) | E −2590, Sharpe −0.29, %12.5 | E +1211, Sharpe **−0.06**, %50 |
| +A1+A2 (hedef-KORU) | — | E +875, Sharpe **+0.04**, %50, WR %42 |
| +hepsi | E −2068, Sharpe −0.33, %18.8 | E +724, Sharpe −0.07 |

**Ters etki (kritik):** Overlay'ler **alfa değil, kayıp-azaltıcı**. Zarar eden temelde (ensemble)
HER metriği iyileştiriyor (Sharpe −0.68→−0.29); kazanan trend temelinde (event_breakout, boğa-ağırlıklı
5y) rejim **zarar veriyor** (toparlanacak düşüşlerde maruziyeti kesiyor), çıkışlar Sharpe'ı bozuyor.
`trail_drop_target=True` (hedef kaldır) Sharpe düşüşünün suçlusu — hedef KORUNUNCA Sharpe baseline'a döndü
(0.03→0.04) + breadth/WR yükseldi (%37.5→%50, WR %42).

### 14.2 Walk-forward (GERÇEK OOS, `wf_pullback.py`, event_breakout, 1d/5y, IS180/OOS60)
| metrik | baseline | +A1+A2 (hedef-koru) |
|---|---|---|
| sembol-ort. OOS expectancy | −432 | **−570** (daha kötü) |
| pozitif-OOS sembol | %37.5 (6/16) | **%37.5 (6/16)** (aynı) |
| ort. WF verimliliği | −1.20 | −7.28 |

**In-sample "tutarlılık kazancı" (%50 kârlı, Sharpe 0.04) OOS'ta ÇÜRÜDÜ** → %37.5'e geri düştü
(baseline ile aynı), ortalama beklenti daha da negatif. Klasik overfit imzası. Dahası **baseline
event_breakout'un kendisi OOS'ta net negatif** (−432, %37.5) → in-sample +1139/+%3.86 büyük ölçüde
artefakt. (§6/§9'u pekiştirir: fiyat-göstergesi ailesinde dayanıklı kenar yok.)

### 14.3 Ship
- **Hiçbiri alfa olarak ship EDİLMEDİ.** Hepsi default-KAPALI; kodda opt-in kaldı (env knob'ları `.env.example`).
- Overlay'ler bu basket/stratejide OOS'ta kenar katmıyor; rejim kazanan trend stratejisinde zararlı bile.
- **Doğrulanmış tek rol:** zarar eden temelde kayıp-azaltma (ama temel zaten kaybediyor → pratik değer yok).
- Asıl darboğaz overlay değil: **OOS kenarı olan baz sinyal yok.** Overlay'ler ancak gerçek edge üstüne oturursa anlamlı.

### Tekrar üretmek için (§14)
```bash
# in-sample A/B (overlay env'leri ile)
.venv/Scripts/python.exe -m research.bench --interval 1d --period 5y --strategy event_breakout --tag before
REGIME_RISK=1 .venv/Scripts/python.exe -m research.bench --interval 1d --period 5y --strategy event_breakout --tag regime
TRAIL=1 TRAIL_TIERED=1 TRAIL_DROP_TARGET=0 MFE_GIVEBACK=1 .venv/Scripts/python.exe -m research.bench --interval 1d --period 5y --strategy event_breakout --tag exits_keeptarget
# OOS doğrulama (asıl karar)
.venv/Scripts/python.exe -m research.wf_pullback --mode event_breakout --interval 1d --period 5y
TRAIL=1 TRAIL_TIERED=1 TRAIL_DROP_TARGET=0 MFE_GIVEBACK=1 .venv/Scripts/python.exe -m research.wf_pullback --mode event_breakout --interval 1d --period 5y
```

---

## 15. Canlı forecast kalibrasyonu + kapalı-döngü doğrulama (2026-06-20)

§1-14 backtest/walk-forward (tarihsel, survivorship-yanlı) ölçtü. Bu tur **canlı forecast
heuristiğinin** (panodaki sürekli yön/güven; `_bias_and_confidence` = EMA20/50 + RSI + ADX +
HTF + rejim) kendi öngörü gücünü ölçtü ve **ileriye-dönük (paper) doğrulama altyapısı** kuruldu.
Bu, fiyat-göstergesi-zamanlamasının kenarsızlığı sonucunu (§6) backtest'ten BAĞIMSIZ bir kanıt
hattıyla pekiştirir.

### 15.1 Güven kalibrasyonu — öngörü gücü YOK (yeni, doğrudan kanıt)
`calibrate_forecast.py`: ham güveni 19 kovaya böl, her kovada `horizon`-bar-sonrası gerçekleşen
yön isabetini ölç. BASKET+stream (30 sembol), 1m, lookback 3000, **40.240 örnek**.

| güven kovası | gerçekleşen isabet |
|---|---|
| ~%9 | %55.9 |
| ~%22 | %50.6 |
| ~%48 | %50.6 |
| ~%72 | %51.2 |
| ~%98 | %47.6 |

**Genel isabet %49.7, Brier 0.323, kovalar düz (%47.6-55.9 bandında).** Güven ne olursa olsun
gerçekleşen isabet ~%50 — yani **forecast güveni geleceği öngörmüyor (yazı-tura).** Yüksek-güven
kovası düşük-güven kovasından daha isabetli DEĞİL. Bu §6/§13.5'in "dayanıklı kenar yok"unun
canlı-heuristik karşılığıdır; tek-zaman-dilimli base güven için kesin ve örneklem-zengin (40k).

**Ürün etkisi (kod):** pano başlık metriği ham "Yön Güveni"den **kalibre olasılığa** çevrildi
(`avg_prob_cal`); ~%50'de gri gösterilir = "öngörü gücü yok" (uydurma kenar görüntüsü yok). Bu,
EDGE_FINDINGS metodolojik ilkesinin ("uydurma kenar ship etme") UI'a taşınmış hâli.

### 15.2 Kapalı-döngü (paper) doğrulama altyapısı — `prediction_tracker.py`
Backtest survivorship'ten muzdarip; bu modül **gerçek-zamanlı, ileriye-bakmayan** kanıt üretir:
`record` (her bar forecast'i diske yaz, bar başına tek/dedup) → `resolve` (horizon bar sonra
gerçekleşen yönle etiketle, look-ahead'siz) → `report` (canlı isabet + güven-kovası kalibrasyon +
verdikt). Emir göndermez (paper-trade eşdeğeri). Servise `PREDICT_TRACK=true` ile opt-in bağlı.

Bu, §13.4/§13.5'in işaret ettiği "gerçek değerlendirme canlı/PIT ister" sonraki-işinin canlı-yarısı:
PIT veri (§13.5, ücretli) backtest-yanlılığını çözerken, prediction_tracker hiç para harcamadan
ileriye-dönük OOS kanıtı biriktirir. İkisi tamamlayıcı.

### 15.3 Karar
- **Yeni dayanıklı kenar İDDİASI YOK** — aksine, §6 "fiyat-göstergesi kenarsız" sonucu canlı
  heuristikte de doğrudan doğrulandı (kalibrasyon düz ~%50).
- Ship edilen: dürüst UI (kalibre olasılık) + doğrulama altyapısı. Forecast'in kendisi sinyal
  kaynağı olarak konumlanmadı; pano "öngörü gücü yok"u şeffaf gösteriyor.
- **Sonraki iş değişmedi (§13.5):** gerçek kenar için survivorship-temiz PIT veri (Sharadar/CRSP)
  + biriken canlı paper kanıtı. prediction_tracker o kanıtı toplamaya hazır.

> Not: Bu oturumun kalan değişiklikleri (log retention, paket reorganizasyonu nihai/+research/)
> kenar bulgusu DEĞİL, altyapı/bakım — ayrıntı `CHANGELOG.md`'de.

### Tekrar üretmek için (§15)
```bash
.venv/Scripts/python.exe -m research.calibrate_forecast --interval 1m --horizon 5   # güven kalibrasyonu (öngörü gücü testi)
# canlı paper doğrulama (PREDICT_TRACK=true ile servis çalışırken birikir):
.venv/Scripts/python.exe -m nihai.execution.prediction_tracker --resolve --report
```

---

## 16. "İsabeti aynı kâr oranıyla artır" — precision filtresi + mutabakat + survivorship testi (2026-06-20)

Soru (kullanıcı): *"botun seçtiği hisselerin doğru çıkma olasılığını aynı/yakın kâr oranıyla
nasıl artırırız?"* Üç deney: (A) meta-labeling'i precision↔return ödünleşim eğrisi olarak ölç,
(B) survivorship-temiz PIT evrende yeniden ölç (§13.5'in öngördüğü deflasyon testi), (C) bağımsız
sinyal-tipi mutabakatının precision'ı keskinleştirip keskinleştirmediğini ölç.

**Metodolojik ön-tespit (§7'yi pekiştirir):** WR'ı mekanik yükseltmek (dar hedef/geniş stop)
PF'i bozar; canlı forecast güveni kalibre değil (§15.1, ~%50). Doğru kaldıraç mekaniği oynamak
değil, **kötü sinyali girmeden elemek** (precision). Bu turun konusu o.

### 16.1 (A) Meta precision-retention taraması — nasdaq100 (survivorship-YANLI)
`train_meta.py`'ye seçicilik↔kâr eğrisi eklendi (sinyalleri kazanma-olasılığına göre sırala,
top-%X tut; WR/PF/E + toplam-$). nasdaq100 1h/2y, lojistik, iki holdout:

| holdout | base | meta-seçici (eşik) |
|---|---|---|
| zaman | WR 38.2, PF 0.78, E −2513 | 0.60 → PF **1.07**, E **+837** |
| görülmemiş-sembol | WR 38.8, PF 0.82, E −1944 | 0.45 → PF **1.38**, E **+3482** |

Taramanın netleştirdiği nüans: **base zaten zarar ediyor** → meta "kârı korumuyor", işlemlerin
~%85'ini eleyip kaybeden tabanı marjinal kâra çeviriyor. İlk bakışta §13.2'yi tekrarlıyor (umut verici).

### 16.2 (C) Bağımsız sinyal mutabakatı (`agreement.py`, yeni) — nasdaq100
META olasılığı (zamanlama kalitesi) × kesitsel-momentum percentile (portföy seçimi; §13.3),
event_breakout işlemleri üzerinde çaprazlandı. Look-ahead-güvenli: momentum yalnızca entry gününe
kadarki fiyatla.

| grup | işlem | WR% | PF |
|---|---|---|---|
| BASE | 2727 | 38.4 | 0.79 |
| yalnız META (p≥0.50) | 132 | 53.8 | 1.68 |
| yalnız MOM (üst-yarı) | 1165 | 39.4 | 0.79 |
| **İKİSİ (mutabakat)** | 56 | **66.1** | **2.75** |

**Mekanik içgörü:** momentum TEK BAŞINA ölü (0.79 = base) — beklendiği gibi, o bir portföy-seçim
sinyali, per-trade timing filtresi değil. AMA **bağımsız onay** olarak META'nın üstüne binince
precision'ı 53.8→66.1, PF 1.68→2.75 yapıyor. İki bağımsız "evet" tek kaynaktan isabetli. Caveat:
META bacağı burada in-sample (model bu veriyle eğitildi → iyimser); mutabakat hücresi 56 işlem
(yüksek varyans). İşaret, kanıt değil.

### 16.3 (B) Survivorship-temiz tekrar-ölçüm — BELİRLEYİCİ TEST
`build_meta_dataset.py`'ye PIT modu eklendi (`--as-of`, `pit_universe.sp500_as_of`, çıkarılanlar
dahil, 1d, start=as-of, `removed_since` etiketi). **S&P500@2022-01-01** (499 sembol, 73 bugün-yok),
**11.874 işlem**. Meta model bu PIT veride sıfırdan eğitilip OOS test edildi:

| metrik | nasdaq100 (yanlı) | **PIT-temiz** |
|---|---|---|
| OOS-AUC zaman | 0.562 | **0.515** |
| OOS-AUC sembol | 0.600 | **0.524** |
| kapı kenarı | PF 1.07–1.38 | **KENAR YOK** |

PIT-temiz evrende **hiçbir eşik** kaybeden tabanı kâra çeviremedi: her retention seviyesinde PF<1,
E<0, toplam-$<0 (en iyi top-%20 sembol-split: PF 0.96). AUC ~0.52 = neredeyse yazı-tura. Genel
kazanma %36.2.

### 16.4 Sentez — §13.5'in öngördüğü deflasyon DOĞRULANDI
§13.5 yazmıştı: *"Aynı survivorship düzeltmesi muhtemelen meta-labeling ve momentum sayılarını da
deflate eder."* Bu tur onu ölçtü: **(A)/(C)'nin parlak sayıları (PF 1.38, 2.75) büyük ölçüde
survivorship artefaktı.** Bugünün kazananları üzerinde kenar var görünüyor; o tarihte gerçekten
işlem yapsaydın (çıkarılacak isimler dahil) kenar yok.

**Confound (dürüstlük):** PIT testi aynı anda evreni (nasdaq100→sp500) VE zaman dilimini (1h→1d)
değiştirdi → çöküşün bir kısmı 1d event_breakout'un zaten daha kötü olmasından gelebilir. AMA
§13.5'te PEAD **sabit 1d'de** aynı deflasyonu (PF 1.57→1.05) gösterdi → baskın sürücü survivorship.

**Karar:**
- Meta-labeling kapısı (`META_GATE`) survivorship-temiz veride OOS kenar göstermedi → **default-KAPALI
  kalmalı** (açma için kanıt yok). nasdaq100'deki "kenar" ship edilmedi, doğru karar.
- Mutabakat fikri (C) **doğru bir mekanizma** ama besleyecek baz kenar PIT'te yok → tek başına çözmez.
- **Darboğaz değişmedi (§13.5/§14.3):** OOS kenarı olan baz sinyal yok. Filtre/mutabakat ancak gerçek
  kenarın üstüne oturursa anlamlı.
- **Tek dürüst sonraki adım:** Sharadar/Polygon PIT + earnings (ücretsiz yaklaşım çıkarılan isimlerin
  ~%60'ında earnings buluyor → hâlâ yukarı-yanlı) **veya** `prediction_tracker` ile canlı paper kanıtı.

### Tekrar üretmek için (§16)
```bash
.venv/Scripts/python.exe -m research.train_meta --model logistic --split both   # (A) precision-retention taraması
.venv/Scripts/python.exe -m research.agreement --meta-thr 0.50 --mom-frac 0.50  # (C) bağımsız mutabakat
.venv/Scripts/python.exe -m research.build_meta_dataset --universe pit_sp500 --as-of 2022-01-01  # (B) PIT dataset
.venv/Scripts/python.exe -m research.train_meta --csv data/meta_dataset_pit_sp500_event_breakout_1d_2022-01-01.csv --model logistic --split both  # (B) PIT OOS
```

---

## 17. "10 ek gösterge + öğrenen/ML algoritma" fikri canlı veride ölçüldü (2026-06-24) — kenar YOK, optimizasyon overfit

**Soru (kullanıcı):** Heiken Ashi, SMA, STDDEV, EMA, ATR, SuperTrend, MACD, BBand, RSI, Ichimoku
gibi göstergeler eklesek + her şeyi "öğrenen/ML tarzı" optimize etsek nasıl olur?

**Önce envanter:** İstenen 10 göstergenin 9'u zaten kodda (SMA `indicators_ext.py`, EMA/RSI/ATR/
Bollinger/MACD `strategy.py`, SuperTrend+Ichimoku+Keltner `indicators.py`). `indicators.py` ayrıca
VWAP, Stochastic, StochRSI, OBV, CCI, Williams %R, MFI, ROC, ParabolicSAR, CMF, Pivot içeriyor ve
hepsini tek konsensüs paneline (`indicator_panel`, -1..+1) bağlıyor. **Tek gerçek eksik: Heiken Ashi.**
Yani gösterge tarafı zaten doygun.

**Ölçüm A — ek göstergeler OOS kenar katıyor mu? (`ext_indicator_eval`, 1h/horizon=5, 10 sembol
yarı-iletken ağırlıklı evren, 5579 örnek):**

| varyant | IS-IC | OOS-IC | OOS-isabet |
|---|---|---|---|
| baseline (ext kapalı) | −0.007 | 0.007 | %49.8 |
| +ext (tüm ek göstergeler açık, el-ayarı) | 0.003 | **0.006** | %49.8 |
| aggressive (ağırlıklar IS'te en-küçük-kareyle fit) | **0.047** | **−0.008** | %51.0 |

- `+ext` ≈ `baseline` (OOS-IC 0.006 vs 0.007) → **ek göstergeler sıfır OOS kenar katıyor.**
- `aggressive` = kullanıcının "öğrenen/ML algoritma" fikrinin minyatürü: IS'te en iyi (+0.047),
  OOS'ta **negatife döndü (−0.008)** → ders-kitabı overfit. §16'daki GBM/meta-labeling deflasyonunun
  aynısı, taze veride yeniden doğrulandı.

**Ölçüm B — temel stratejinin kendisi (`bench`, event_breakout 1h/1y, 16-sembol BASKET):**
`weighted_E = −4858 $/işlem · kârlı sembol %18.8 · mean Sharpe −1.21` → taban negatif beklentili.

**Ölçüm C — canlı paper, ileriye-dönük, survivorship'siz kanıt (`prediction_tracker --report`,
5928 çözülmüş tahmin, 30-sembol STREAM_SYMBOLS, 1m):** Bu backtest DEĞİL — birikmiş gerçek-zamanlı
tahmin-vs-gerçekleşen muhasebesi, yapısı gereği survivorship'siz.
- Yön isabet oranı: **%47.2** (yazı-turanın *altında*) · ort. yön-ayarlı getiri **−0.020%/tahmin**.
- **Kalibrasyon TERS:** yüksek-güven kovaları (0.6–1.1) %40–44 isabet; düşük-güven 0.2–0.3 kovası %56.
  Model güveni anti-kalibre — güvendiği yerde daha çok yanılıyor. Filtreyi "yüksek güvene gir"
  yönünde kurmak performansı *düşürür*.
- (Yan düzeltme: `prediction_tracker.py` Windows cp1254 stdout'ta "→" basarken çöküyordu; modül
  başına utf-8 guard eklendi — rapor artık tam kalibrasyon tablosuyla çıkıyor.)

**Sonuç:** Darboğaz §13.5/§14.3/§16'dakiyle aynı — **gösterge sayısı veya model sofistikasyonu
DEĞİL; tabanda OOS kenar yok.** Negatif-kenarlı tabanın üstüne gösterge/ML eklemek overfit yüzeyini
büyütür, istatistiksel olarak durumu kötüleştirir. Heiken Ashi eklemek tamlık için yapılabilir ama
kenar yaratmaz. **Tek dürüst yol değişmedi:** survivorship-temiz PIT veri (Sharadar/CRSP) veya
`prediction_tracker` canlı paper kanıtı.

**Ölçüm D — anti-kalibrasyon STABİL ama SÖMÜRÜLEMEZ (`calibration_fade_test`, 2026-06-25,
5936 tahmin, kronolojik 70/30 train/test):** §17-C'deki "yüksek güven → düşük isabet"
gürültü mü, gerçek mi? Kronolojik bölünmede ölçüldü.
- **Anti-kalibrasyon GERÇEK/STABİL:** güven 0.6-1.1 kovaları HEM train HEM test yarısında
  <%50 isabet (çakışmasız alt-örneklemde de aynı). Model gerçekten ters-kalibre.
- **AMA fade etmek PARA KAYBETTİRİR:** OOS'ta `conf≥τ fade` isabeti **%54.6** (>%50!) ama
  ort. getiri **−0.071%** (NEGATİF). Yön doğru ama kazanç küçük/kayıp büyük → isabet ve
  beklenti KOPUK. Her iki örneklemde de dayanıklı fade kenarı YOK.
- **Bu, "win-rate artır" sorusunun canlı kanıtı:** %54.6 isabet elde edip yine kaybedebilirsin.
  Güven-filtresiyle win-rate artırma yolu ÖLÇÜLEREK KAPATILDI (neden kapalı = sayıyla görüldü).

**Sonuç:** Win-rate'i (güven-filtresi/fade ile) artırmak beklentiyi artırmaz; OOS isabeti
yalnızca yeni/dik PIT-doğrulanmış sinyal kaynağıyla değişir (small-cap momentum / olay sinyali).

### Tekrar üretmek için (§17)
```bash
.venv/Scripts/python.exe -m research.ext_indicator_eval --interval 1h --horizon 5 --lookback 1200 --stride 2
.venv/Scripts/python.exe -m research.bench --interval 1h --period 1y --strategy event_breakout
.venv/Scripts/python.exe -m research.calibration_fade_test   # §17-D anti-kalibrasyon fade testi
```

---

## 18. "İşlemlerin WR'ını benzer kâr payıyla artır" — üç taze ölçüm + bataryanın turnkey'leşmesi (2026-06-29)

**Soru (kullanıcı):** *"botun seçtiği işlemlerin başarı oranını (WR) benzer kâr payı kalacak
şekilde artırır mısın?"* §7/§16'nın tekrarı gibi görünse de bu tur ÜÇ bağımsız taze ölçümle
sınandı ve sonunda kullanıcı "gerçek çözüm = alternatif sinyal" yolunu seçti → batarya turnkey'lendi.

### 18.1 (A) Giriş-seçiciliği — "breakout tamponu" (survivorship-temiz PIT, 11.874 işlem)
PIT S&P500 meta-dataset'inde (`meta_dataset_pit_sp500_event_breakout_1d_2022-01-01.csv`) her
giriş-özelliği için eşik tarandı; profit sizing-bağımsız R-cinsinden (1R = medyan |kayıp|, winsor
[-8,12]). Baz: WR %36.2, sumR −897 (zaten net zarar — §16.3 ile tutarlı).

| filtre | n (kalan) | WR | sumR | H1/H2 tutarlı? |
|---|---|---|---|---|
| BAZ (tümü) | 11874 | 36.2 | −897 | — |
| `dist_hi20 ≥ 0.006` (20-bar zirveyi %0.6 AŞ) | 3516 | **39.6** | **−11** | evet (40.5/38.7) |
| `vix ≤ 16` | 4284 | 40.0 | +120 | **HAYIR** (H1 50.3 / H2 35.5 — rejim artefaktı) |

`dist_hi20` (marjinal kırılım yerine BELİRGİN kırılım iste) WR'ı ~3 puan artırıp net zararı sıfıra
çekiyor, iki zaman-yarısında tutarlı. **AMA tek-dataset** → §6 disiplini: ship etmeden ÖNCE gerçek
motorla 1y/2y doğrulama şart; vix filtresi gibi tek-pencere parlaklığı OOS'ta söner. Ship EDİLMEDİ.

### 18.2 (B) Çıkış-retarget — yaklaşıklamanın YANILTTIĞI metodolojik ders
mfe_pct/mae_pct ile alternatif stop/target re-simüle edildi; ilk sonuç "2.0/2.0 hem WR hem kârı
artırır" dedi. **Yanlıştı:** re-sim "neither-touched→0" ve "both-touched→stop" varsayımları
idealize ediyor. **Gerçek motor (§7a, `wr_grid.py`) belirleyici:** stop/target ile WR %38→%69
çıkar ama PF 0.85→0.60 DÜŞER. → Ders: excursion-yaklaşıklaması çıkış kararı için güvenilmez;
otorite event-driven backtest. §7'nin "WR mekanik yükseltmek kârı bozar" sonucu DEĞİŞMEDİ.

### 18.3 (C) Canlı paper log madenciliği — survivorship'SİZ, bağımsız doğrulama (YENİ)
`predictions_log.csv` (11.871 çözülmüş, 1m/5-bar, ileriye-dönük → yapısı gereği survivorship'siz)
kronolojik 60/40 train/test ile sembol × güven × saat × yön kesitlerinde tarandı:
- **Genel isabet %47.1** (yazı-turanın altında), ort. yön-ayarlı getiri −0.028%/tahmin.
- **Dayanıklı kenar cebi YOK:** TEST'te iyi görünen semboller (IBM %58.7, GOOGL %55.7) TRAIN'de
  KÖTÜYDÜ (%48, %42) → gürültü, sinyal değil. Hiçbir kesit iki-yarıda da >%52 + pozitif getiri değil.
- **Güven anti-kalibre (resmi rapor doğruladı):** güven 0.6–1.1 kovaları %40–46 isabet; en düşük
  güven en yüksek isabet. "Yüksek güvene gir" filtresi WR'ı DÜŞÜRÜR. §17C/D ile birebir tutarlı.

### 18.4 Sentez — darboğaz TEKNİK değil VERİ
Üç taze ölçüm de §6/§16/§17'yi pekiştirdi: bu sinyalde WR↔kâr katı ödünleşim; survivorship-temiz
ve canlı her ölçümde (bugünküler dahil) kenar yok. **WR'ı kârı koruyarak artırmanın tek yolu daha
iyi bir SİNYAL** — parametre/filtre/güven-eşiği değil. Akademik olarak güvenilir iki aday
(**PEAD PIT-temiz**, **small-cap momentum**) point-in-time veri ister; harness'lar hazır, veri yok.

**Ship edilen (bu tur):** `research/run_pit_validation.py` — survivorship-temiz doğrulama bataryası
TEK komutta, preflight-kontrollü: Adım 2 (smallcap-mom, Sharadar PIT) anahtar yoksa net yönergeyle
ATLANIR; Adım 3 (PEAD PIT, ücretsiz S&P500 as-of evren) ve Adım 4 (canlı paper) ücretsiz koşar.
Anahtar `.env`'e girince Adım 2 otomatik açılır. Üretim seçim-mantığına DOKUNULMADI (kenar kanıtı yok).

### 18.5 Yol haritası (öncelik)
1. **Sharadar (Nasdaq Data Link) SEP+TICKERS+SF1** aboneliği (~$30–150/ay) → `.env` NASDAQ_DATA_LINK_API_KEY
2. `run_pit_validation` → Adım 2 small-cap momentum (edge_vs_buyhold>0 İKİ pencerede ŞART)
3. Adım 3 PEAD'i cap-tier × surprise-büyüklüğü kesitiyle derinleştir (removed-since PF'i izle, §13.5)
4. (Bedava, paralel) `prediction_tracker` canlı paper'ı biriktir — tek survivorship'siz kanıt akışı

### Tekrar üretmek için (§18)
```bash
.venv/Scripts/python.exe -m research.run_pit_validation                 # üç adımlı batarya (preflight'lı)
.venv/Scripts/python.exe -m research.run_pit_validation --only 3,4      # yalnız ücretsiz adımlar
.venv/Scripts/python.exe -m nihai.execution.prediction_tracker --report # canlı isabet/kalibrasyon (§18.3)
```

---

## 19. PEAD KESİT-ML — "sıfırdan AI ile WR artır" ücretsiz yolu ölçüldü: KENAR YOK (2026-07-01)

**Soru (kullanıcı):** *"trade/analiz/hesaplama için sıfırdan AI/ML inşa edip WR/doğruluğu
artırır mısın?"* Mevcut (kenarsız) event_breakout tabanına yeni sınıflandırıcı = §16'da
ölçülerek kapatıldı; onun yerine ML **akademik desteği olan PEAD sinyaline + survivorship-temiz
ücretsiz veriye** uygulandı (kenar YARATMA potansiyeli olan yol; §18.5 #3).

### 19.1 Kurulan altyapı (yeniden kullanılabilir, PIT veri gelince tekrar koşulur)
- `research/build_pead_dataset.py` — PEAD'in fire ettiği HER earnings olayına look-ahead-güvenli
  KESİT özellikleri ekler: `surprise` (yönlü %), `abs_surprise`, `dollar_vol_med` → cap-tier
  (dolar-hacim yüzdeliği; PIT market-cap ücretsiz yok → likidite vekili). Etiket = GERÇEK
  backtest çıkışı; `removed_since` işaretli (survivorship kapısı).
- `research/train_pead_ml.py` — `train_meta`/`purged_cv` model sınıflarını yeniden kullanır;
  3 disiplin eklentisi: (1) genişletilmiş 22-özellik seti, (2) **SURVIVORSHIP-bölmeli kapı
  değerlendirmesi** (removed-since vs hayatta-kalan AYRI — train_meta'da yoktu), (3) purged+
  embargolu 5-fold CV genişletilmiş özelliklerle.

### 19.2 Veri
PIT S&P500 @2022-01-01: **473/506 sembolde veri, 3228 PEAD olayı, 354 removed-since (%11)**,
genel WR %36.5 (§16.3 ile tutarlı, zaten net-zarar taban). removed-since kapsamı yeterli →
survivorship kapısı ÖLÇÜLEBİLİR (en kötü senaryo—ölü ticker earnings'i tamamen boş—olmadı).

### 19.3 Purged 5-fold CV (KARAR MERCİİ — zaman-serisi-dürüst OOS)
| model | ort OOS-AUC | eğitim-AUC | kapı kazanan fold | removed-since kapı pozitif | verdikt |
|---|---|---|---|---|---|
| logistic | **0.496** ± 0.026 | 0.598 | 2/3 | **0/5** | kenar YOK |
| gbm | **0.515** ± 0.056 | 0.899 (ağır overfit) | 2/5 | **1/5** | kenar YOK |

Her iki modelde ortalama OOS-AUC ~yazı-tura; kapı yalnızca %3-7 işlem tutuyor (cherry-pick);
removed-since'te kapı neredeyse hiç kazanmıyor. → **META_GATE kapalı kalmalı.**

### 19.4 Tek-bölme tuzağı canlı yakalandı (§16.3 disiplininin değeri)
- **time-split** kapısı muhteşem görünüyor: logistic PF **4.40** / E +12946. AMA survivorship
  kırınca **removed✗** (removed-since BASE PF 0.75, kapı kaybettiriyor). Kenar tümüyle hayatta
  kalanlarda = artefakt. Naif tek-bölme bunu SHIP ederdi.
- **symbol-split** ise removed✓ gösterdi (GBM PF 2.11) — ama bu time-split VE purged CV ile
  ÇELİŞİYOR. Purged CV (5 pencere ortalaması) removed-since'i 0/5 ve 1/5 pozitif buluyor →
  symbol-split "removed✓" şanslı tek bölmeydi. **Çelişki = dayanıksızlık.**

### 19.5 Neden çalışmadı — özellik önemi teşhisi
Permütasyon önemi: model kararı `vix`, `ret_5`, `ret_20`, `cap_tier`, `ema_gap`'e dayanıyor —
yani **rejim/momentum gürültüsü**. PEAD'in tanımlayıcı değişkeni `surprise`/`abs_surprise`
her iki modelde de top-8'de YOK. ML, sürpriz büyüklüğünde sömürülebilir kesit-yapı bulamadı;
bulduğu şey OOS'ta genellemeyen rejim gürültüsü. Bu, kenar-yokluğunun mekanik açıklaması.

### 19.6 Sentez — §16/§18 pekişti, darboğaz değişmedi
"Sıfırdan AI ile WR artır" ücretsiz yolu (PEAD kesit-ML) titizlikle ölçüldü: **kenar YOK.**
Görünen tek-bölme kapıları survivorship/şans artefaktı; purged CV + removed-since kapısı
bunları eledi. **Ship edilen alfa = HİÇBİR ŞEY** (dürüst sonuç). Ship edilen = yeniden
kullanılabilir survivorship-bölmeli PEAD-ML harness'ı (PIT veri gelince tekrar koşulur).
Darboğaz §16/§18'deki gibi: **model değil, VERİ.** İleri yol değişmedi → Sharadar PIT
small-cap momentum (§18.5 #1-2) veya `prediction_tracker` canlı paper kanıtı.

### Tekrar üretmek için (§19)
```bash
.venv/Scripts/python.exe -m research.build_pead_dataset --as-of 2022-01-01   # PIT kesit dataset
.venv/Scripts/python.exe -m research.train_pead_ml --model both --split both --folds 5
# (xgboost kuruluysa --model all; ana verdikt purged CV satırlarında)
```

### 19.7 ADİL-ŞANS deneyi (2018 örneklem + SUE özelliği) — yine KENAR YOK (2026-07-01)
§19 "eğitime hakkını ver" itirazına yanıt: iki ücretsiz kaldıraç eklendi — (a) **2x örneklem**
(PIT S&P500 @2018, 3347 olay, 501 removed-since=%15, survivorship kapısı çok daha güçlü),
(b) **SUE** (`sue`,`abs_sue`: sürprizi hissenin geçmiş sürpriz-oynaklığına standardize — ham
surprise'ın daha temiz/öngörücü hali). Triple-barrier zaten backtest etiketinde (stop/hedef/zaman).

**Purged 5-fold CV (karar mercii):**
| model | ort OOS-AUC | eğitim-AUC | removed-since kapı pozitif | verdikt |
|---|---|---|---|---|
| logistic | 0.521 ± 0.040 | 0.603 | **0/5** | kenar YOK |
| gbm | 0.519 ± 0.019 | 0.903 (hâlâ ağır overfit) | **1/5** | kenar YOK |

**SUE mekaniği doğrulandı ama tavan aşılmadı:** GBM permütasyon öneminde `abs_sue_w`/`sue_w`
artık görünüyor (ham `surprise_w` logistic'te ~0.000 idi) → SUE gerçekten daha bilgilendirici.
AMA yeterli değil: OOS-AUC 0.515→~0.52 kıpırdadı, kapı yine %2-6 tutuyor (cherry-pick), removed-
since'te 0-1/5 fold pozitif. Tek-bölme kapıları hâlâ parlak (PF 3.9-5.9) ama survivorship/şans.

**Sonuç:** 2x veri + daha iyi özellik = overfit biraz azaldı, kenar YOK. **Büyük-cap bilgi
tavanı (arbitrajlanmış PEAD) doğrulandı.** Kalan tek gerçek kaldıraç: small-cap PIT (Sharadar,
paralı) veya canlı paper. Ship = HİÇBİR ŞEY; META_GATE kapalı. Harness SUE'li olarak güncellendi.

### 19.8 YIL-YIL walk-forward (2022→2026) — kenar HİÇBİR yılda dayanıklı değil (2026-07-02)
Kullanıcı "her yıl (2019-2026) için ölç" istedi. İki kısıt/bulgu: (a) **yfinance ücretsiz
earnings ~4 yıl** → 2019-2020 PEAD olayı ücretsiz ALINAMAZ (2018 as-of build'i bile ancak
2021'den olay verdi); Sharadar gibi paralı PIT gerekir. (b) Ayrı as-of build yerine `research/
pead_walkforward.py`: 2018 dataset'inde (2021-2026, 3347 olay) her yıl için eğitim=önceki yıllar,
test=o yıl (purged; SUE dahil).

**Sonuç (kapı eşik 0.55):**
| test-yıl | logistic OOS-AUC | gbm OOS-AUC | removed-since kapı |
|---|---|---|---|
| 2022 | 0.464 | 0.497 | ✗ / ✗ |
| 2023 | 0.534 | 0.496 | az-işlem / az-işlem |
| 2024 | 0.442 | 0.400 | az-işlem / ✓ |
| 2025 | 0.509 | 0.542 | ✗ / ✗ (E -12523) |
| 2026 | 0.453 | 0.512 | az-işlem / az-işlem |

OOS-AUC her yıl ~0.44-0.54 (yazı-tura); tutarlı >0.53 YOK. "Genel kapı" 3-4/5 yıl kazanıyor
gibi ama: (i) AUC yazı-tura → şans, (ii) kapı yılda 5-37 işlem tutuyor (cherry-pick), (iii)
**removed-since kapısı logistic 0/5, gbm 1/5 yıl pozitif** — dürüstlük alt-kümesinde ölçülebildiği
yıllarda kaybediyor (2022 E -7292, 2025 E -12523). Yıllar arası da tutarsız (2024 gbm ✓ ama 2025 ✗).

**Sonuç:** Kenar HİÇBİR yılda survivorship-temiz olarak dayanmıyor. §19/§19.7'yi 5 bağımsız yıl-
penceresinde bir kez daha doğrular. Ücretsiz büyük-cap yol tümüyle kapandı; tek kaldıraç Sharadar
PIT (small-cap) veya canlı paper. Ship = HİÇBİR ŞEY.

## 20. Meta-labeling SIP-kalitesi PIT veride yeniden ölçüldü (2026-07-03): kenar YOK — "veri kalitesi" hipotezi de elendi

§16'nın "kenar yok" hükmüne karşı kalan tek makul itiraz şuydu: "yfinance verisi eksik/ayarlı
(delist kopukluğu, split-ayar farkları, IEX-benzeri hacim sorunları) — belki kenar veri
kalitesinde kayboluyor." Bu tur o hipotez test edildi.

**Veri:** `alpaca_history.py` (SIP feed = konsolide tüm-borsa hacmi; delist edilmiş semboller
dahil) ile `pit_sp500@2021-07-01` evreninden 1d event_breakout dataset'i yeniden kuruldu:
**12.268 işlem, 505 sembol, 80'i removed-since (1.494 satır), NaN yok** —
`meta_dataset_pit_sp500_event_breakout_1d_2021-07-01_alpaca_hist.csv`.

**Ölçüm 1 — train_meta (time+symbol tek bölme):** logistic OOS-AUC 0.477 (time) / 0.532
(symbol) = yazı-tura. GBM görünürde "kenar" gösteriyor (symbol 0.584, eşik 0.60'ta PF 9.15)
ama **7-49 işlemlik** örneklemlerde — klasik GBM-overfit deseni (ACI DERS'teki nasdaq100
hikâyesinin aynısı).

**Ölçüm 2 — purged 5-fold CV (sağlam hüküm):**
- logistic: ORT OOS-AUC **0.487 ± 0.020** → eşiği geçen işlem yok → kenar YOK
- gbm: ORT OOS-AUC **0.503 ± 0.014**, kapı kazandığı fold 2/3, tutma %1 → kenar YOK / dayanıklı değil

**Sonuç:** §16 (yfinance PIT, 11.874 işlem, AUC 0.51-0.52) bağımsız ve daha kaliteli veri
kaynağında birebir doğrulandı. Kenarsızlık veri artefaktı DEĞİL — sinyalin kendisinde OOS
bilgi yok. META_GATE varsayılan-KAPALI kararı iki bağımsız veri kaynağıyla desteklenmiş oldu.
Darboğaz değişmedi: OOS-kenarlı baz sinyal yokluğu (ACI DERS #5); ileri yol Sharadar PIT
(small-cap) veya prediction_tracker canlı paper birikimi.

---

## 21 — Alpaca 10y saatlik PIT (97k işlem): meta-labeling kenarı YOK (en güçlü ölçüm)

**Veri (bugüne dek en iyisi):** Alpaca SIP saatlik barlar, S&P500 @ 2016-01-01 üyeliği
(504 sembol, 162'si bugün endekste YOK), 2016-01→2026-07, **97.143 işlem**. Delist edilmiş
isimlerin %100 kapsaması (Alpaca inaktif-varlık geçmişi) → §16'nın yfinance-günlük setinden
hem 8× daha büyük hem intraday hem TAM survivorship-temiz. removed-since payı %27.

**Kritik metodoloji düzeltmesi (bu veriye özel):** Madde 20 kill-switch + bileşik-kasa
iflası, çok-yıllık tek koşuda örneklemi kayıp rejimlerinde SANSÜRLÜYORDU (ilk üretim:
satırların %95'i 2016'ya sıkıştı, 10.204 satır). Çözüm: `build_meta_dataset --yearly-windows`
(her sembol bağımsız 1'er yıllık taze kasa) + `disable_kill_switch`. Sonuç dengeli 10y
dağılım (yıl başına 5.8k-11.3k). Canlı backtest davranışı DEĞİŞMEDİ (bayrak varsayılan kapalı).

**Ölçüm 1 — purged 5-fold CV (sağlam hüküm):**
- logistic: ORT OOS-AUC **0.511 ± 0.006**, kapı kazandığı fold 0/1, tutma %0 → kenar YOK
- gbm: ORT OOS-AUC **0.509 ± 0.003**, kapı kazandığı fold 1/1, tutma %0 → kenar YOK

**Ölçüm 2 — train_meta time+symbol:** logistic time-split 38.858 işlemden yalnız **6**
tuttu (PF 3.68) — saf gürültü; symbol-split GBM/XGB AUC 0.54 ama time-split'te 0.51 +
kenar yok → tek-bölme, dayanıklı değil.

**BULGU (araç hatası, düzeltildi):** `train_meta --save` bu 6-işlemlik "çift-bölme kenarını"
DAYANIKLI sayıp modeli kaydetti. `--min-gate-keep` (varsayılan 50) eklendi: bir bölmede
tutulan-işlem < 50 ise dayanıklılık sayılmaz. Yanlış kayıt geri alındı; regresyon guard
testi eklendi (`test_train_meta_save_discipline.py`).

**Sonuç:** §16 → §19 → §21: ÜÇ bağımsız veri kaynağı/ölçekte aynı hüküm. event_breakout'a
uygulanan meta-labeling kapısının OOS kenarı YOK; kenarsızlık veri kalitesi/derinliği/
survivorship artefaktı DEĞİL — sinyalde OOS bilgi yok. Darboğaz doğrulandı: OOS-kenarlı
BAZ sinyal yokluğu (ACI DERS #5), model mimarisi/veri değil. META_GATE varsayılan-KAPALI
üç ölçümle sağlam. İleri yol: yeni BAZ sinyal (haber/olay özellikli) VEYA prediction_tracker
canlı paper — daha çok fiyat-türevi veri değil.

---

## 22 — Pairs/kointegrasyon PIT deneyi (2026-07-18): kenar YOK — YENİ başarısızlık modu: zamanlama artefaktı

Beşinci aile (PEAD → meta → momentum → PEAD-kesit-ML → pairs). Tam rapor: `docs/PAIRS_TEST_2026-07-18.md`;
harness: `research/pairs_test.py` (CLI: `--formation 252 --trade 63 --max-pairs 20 --cost-bps 5
--variant a|b|both --grid --exec-lag N`); birim testleri `tests/test_pairs_math.py`.

**Tasarım (ön-kayıtlı):** 252g formation / 63g trade walk-forward, PIT S&P500 ∩ alpaca_hist (507 sembol,
2021-07→2026-07 ≈ 5 yıl — Alpaca ücretsiz SIP günlük geçmişi 2021-07'de başlıyor, "10 yıl" hedefi veri
gerçeğine küçüldü), 16 örtüşmeyen pencere, ort. evren 465.8. |corr| ön-eleme → Engle-Granger p<0.05 →
pencere başına en iyi 20 çift. Varyant A = statik formation-OLS beta; B = Kalman dinamik beta (numpy).
Giriş |z|≥2, çıkış |z|≤0.5, stop |z|≥3.5, time-stop; 5 bps/bacak-yön (+0/10 duyarlılık); t sinyal →
t+1 kapanış uygulaması (lag-1 konvansiyonu).

**Sonuç (z=2.0, p<0.05):**

| varyant | maliyet | PF | E(bps) | Sharpe | DSR | rem-PF | surv-PF |
|---|---|---|---|---|---|---|---|
| A | 5 bps | 0.85 | −7.9 | −0.59 | 0.000 | 0.60 | 0.89 |
| B | 5 bps | 2.08 | +58.3 | 1.03 | 0.049 | 2.13 | 2.07 |
| B | 10 bps | 1.60 | +38.3 | 0.80 | 0.028 | 1.70 | 1.60 |

PBO = **0.004** (CSCV s=10, 252 kombinasyon, `research/pbo.py` — İLK kullanım); grid genelinde en yüksek
DSR **0.102** (hiçbiri 0.5'i geçmedi). removed-since kapsaması 62/69 (~%90).

**Kritik bulgu — varyant B survivorship desenini KIRDI ama kenar yine de sahte:** rem-PF 2.13 ≈ surv-PF
2.07 (projede ilk kez removed-since paritesi). Ancak uygulamaya **1 bar ek gecikme** (`--exec-lag 1`)
eklenince B: PF 2.08 → **1.03**, expectancy +58.3 → **+1.8 bps** (güçlü diğer B konfiglerinde +103→+18).
Kenarın ~%80-95'i sinyal-sonrası İLK barda = 1-günlük kısa-vade reversal / bid-ask sıçraması artefaktı;
dayanıklı kointegrasyon dönüşü değil. DSR'nin zaten zayıf olması (≤0.102) bununla tutarlı.

**Hüküm:** kanıt kapısı GEÇİLMEDİ; forward-shadow adayı DEĞİL. **ACI DERS #7** (CLAUDE.md): ham PF +
maliyet-dayanıklılık + survivorship-paritesi + düşük PBO'nun TÜMÜ olumluyken bile zamanlama artefaktı
mümkün; ayırt eden test exec-lag duyarlılığıdır ve bundan sonra her aday-sinyal kapısının parçasıdır.

**Dürüst tasarım sapmaları:** (1) Kalman delta=1e-4 dejenere (z_std~0.15, 5 yılda 2-17 işlem) → z-ölçeği
ilkesiyle (PnL'e bakmadan) 1e-6'ya çekildi ve raporlandı; (2) veri 10y değil ~5y (SIP kısıtı).

### Tekrar üretmek için (§22)
```powershell
.\.venv\Scripts\python.exe -m research.pairs_test --variant both --grid            # tam grid (~137 sn)
.\.venv\Scripts\python.exe -m research.pairs_test --variant b --exec-lag 1         # zamanlama testi
```

---

## 23 — Canlı sinyal ailesi TÜM ufuklarda ölçüldü (2026-07-27): H* YOK + giriş kayması keşfi

**Soru:** canlı `predictions_log.csv`'deki 50.015 çözülmüş tahminin tamamı `horizon_bars=5`
(1m barda 5 dakika) ile puanlanmış ve isabet %46,8 çıkmıştı. Bu sinyal ailesinin gerçek
kalitesi mi, yoksa YANLIŞ ÖLÇEKTE ölçülmüş olmasının artefaktı mı? Sistemin en büyük
ölçülmemiş serbestlik derecesi bu tek parametreydi.

**Harness:** `research/horizon_sweep.py` — 52.257 tahmin, 36 sembol, H ∈ {1,5,15,30,60,120,390}
× exec-lag ∈ {0,1,2,5} (28 kombinasyon), Corwin–Schultz spread (medyan 2,11 bps), kesintisiz
RTH dakika ızgarası, sembol×gün kümelenmiş t, Šidák kritik t = 3,116, zaman-bölmeli doğrulama.

| Hücre (bar girişi) | isabet | net getiri % | t_küme | ilk yarı | ikinci yarı |
|---|---|---|---|---|---|
| H1_lag1  | 0.4858 | −0.0249 | −3.83 | − | − |
| H5_lag1  | 0.4848 | −0.0285 | −4.27 | − | − |
| H30_lag1 | 0.4848 | −0.0382 | −2.56 | − | − |
| H390_lag1| 0.5043 | +0.0052 | +0.23 | − | + |

**Hüküm: H\* YOK.** 1 dakikadan tam seansa (390 bar) kadar HİÇBİR ufuk/gecikme kombinasyonu
"maliyet sonrası pozitif + kümelenmiş t > Šidák + her iki zaman yarısında pozitif" kapılarını
birlikte geçmedi (`KABUL_ADAYLARI` = []). Tek pozitif net getiri H390'da ve t_küme ≤ 0,58 —
gürültü. **%46,8 bir ölçek artefaktı DEĞİLDİR; bu sinyal ailesi ölçülen tüm ufuklarda ölüdür.**

### Yan bulgu (asıl değerli olan): GİRİŞ KAYMASI
Parite kapısı kurulurken çıktı: logdaki `price` (canlı giriş) karar barının kapanışıyla
eşleşmiyor, `fwd_price` ise +5 barın SIP kapanışıyla birebir eşleşiyor. Yani giriş ve çıkış
**farklı kaynaklardan** geliyordu ve fark **sistematik olarak aleyhte**:

* ortalama kayma **+1,85 bps** sinyal yönünde (t = 21,2; n = 52.210) — medyan hareket ~13,5 bps
  olduğu için hareketin %7–22'si,
* aynı tahminler bar kapanışında girilseydi isabet 0,4654 → **0,4937** (+2,83 puan),
* parite: canlı-giriş varyantı üretimi %95,1 yeniden üretiyor (isabet 0,4654 vs logda 0,4682)
  → ölçüm hizalı, bulgu gerçek.

Mekanizma: yön canlı fiyattan hesaplanıyor (`_bias_and_confidence(price, …)`), giriş de o fiyata
yazılıyor, çıkış ise bar kapanışı. Canlı `hit` kolonu bu yüzden **yön becerisi ile kovalama
maliyetini karıştırıyordu**. Üretimde ayrıştırıldı: `bar_close` + `hit_bar` (saf yön becerisi)
+ `slip_bps` (uygulama maliyeti) kolonları eklendi; `hit` geriye uyum için DEĞİŞMEDİ.

### LONG bacağı: bug değil, kayma (F4)
`research/long_leg_diagnostic.py` dört hipotezi ayırdı:

* **H1 bug ELENDİ** — logdaki `hit`, `işaret(fwd−price)×yön` ile %99,94 (LONG) / %99,95 (SHORT)
  yeniden üretiliyor; taraf ayrışması 0,0001. Tur-15 tipi normalizasyon hatası YOK.
* **H2 KAYMA doğrulandı** — giriş kayması LONG'ta 2,69 bps, SHORT'ta 1,07 bps. Ham isabet açığı
  (SHORT−LONG) canlı girişte **0,0129** iken bar-kapanışı girişinde **0,0002**'ye düşüyor;
  gün-eşleştirilmiş getiri farkı t = −2,78 → **−1,23** (anlamsız).
* **H3/H4** — gün eşleştirmesi (sembol×gün kümesinde LONG−SHORT farkı, 469 küme) hem yükseliş
  hem düşüş günlerinde anlamsız; yapısal LONG zayıflığı iddiası desteklenmiyor.

**Uyarı:** "açık yok" ≠ "kenar var". Bar girişinde bile LONG 0,494 / SHORT 0,494 isabet ve her
iki tarafta net getiri maliyet sonrası NEGATİF.

### λ (karar eşiği) süpürmesi — %100 ABSTAIN kaza değil
`research/shadow_decision_backtest.py --lambda-sweep`, λ ∈ {0,5 … 0}, zaman-bölmeli (ilk %60
seçim / son %40 doğrulama, 1800 bağlam): λ=0,5/0,3/0,2'de motor hiç konuşmuyor; konuştuğu her
λ'da doğrulama fazlalığı ≤ 0 (λ=0,02'de +0,0008 puan = gürültü, PF 0,98). **λ=0,5'te KAL.**
Üretimdeki %100 ABSTAIN artık ölçülmüş doğru davranıştır. (Prior `ai_learning_state.json`'dan
geldiği için SEÇİM bölmesi in-sample; doğrulama da prior kirliliğinden tam arınmış değil.)

### Tekrar üretmek için (§23)
```powershell
.\.venv\Scripts\python.exe -m research.horizon_sweep --spread-model corwin --json data/horizon_sweep.json
.\.venv\Scripts\python.exe -m research.long_leg_diagnostic --json data/long_leg.json
.\.venv\Scripts\python.exe -m research.shadow_decision_backtest --lambda-sweep --json data/lambda_sweep.json
```
Artefaktlar: `data/horizon_sweep_2026-07-27.json`, `data/long_leg_2026-07-27.json`,
`data/lambda_sweep_2026-07-27.json`, `data/uretim_kpi.json`.

---

## 24 — Nöral model CANLI 1m veride eğitildi (2026-07-30): kenar YOK, ama ÖLÇEK hizalandı

**Bağlam:** denetim B-2, canlı gölge köprünün sustuğunu ölçmüştü — kayıtlı tüm nöral
modeller **1d/1h** barlarda eğitilmişti, canlı tahminlerin **%100'ü 1m**. `max|z| = 30,16`
→ OOD guard 600/600 susturuyordu. Guard'ın davranışı DOĞRUYDU; eksik olan **dağılımı
eşleşen bir modeldi**. Bu, "model kenarsız" ile "boru hattı uyumsuz" ayrımının canlı örneği.

**Veri:** `research/build_live_dataset` → `data/meta_dataset_live_1m_2026-07-30.csv`.
**5.996 satır · 30 sembol · taban oranı %46,8.** Yapısı gereği **survivorship-temiz**
(tahmin anında evrende olan semboller) ve **leak-free** (özellikler karar barında yazılır,
etiket horizon bar sonra çözülür). Bugüne kadarki en dağılım-uyumlu eğitim seti.

**Ölçüm (deep_mlp, 186.702 param, purged+embargolu 5-fold):**

| Fold | 0 | 1 | 2 | 3 | 4 | Ortalama |
|---|---|---|---|---|---|---|
| OOS-AUC | 0,522 | 0,507 | 0,542 | 0,537 | 0,518 | **0,525 ± 0,013** |

Sharpe(sözde-getiri, çoğunluk kuralına göre FAZLALIK — ACI DERS #8 düzeltmesiyle)
**−0,0265** · PSR 0,02 · **DSR 0,0006** (n_trials=5, n=5.996).

**HÜKÜM: KENAR YOK.** AUC 0,50'nin hafif üstünde görünse de çoklu-test düzeltmesinden
sonra beceri kanıtı sıfır (DSR ≈ 0). Bu, §2/§6/§19/§20/§21'in canlı-veri karşılığıdır:
**survivorship ve veri kalitesi hipotezleri elendikten sonra bile kenar yok.**

**Buna rağmen DEĞERLİ olan:** köprü artık **konuşuyor** ve kalibre.

| | Önce (`1f072998`, 1d) | Sonra (`3473ecf3`, 1m) |
|---|---|---|
| üretilen / denenen | **0 / 600** | **598 / 600** |
| OOD susturulan | 600 | 2 |
| eğitim taban oranı | 0,3569 | 0,4681 |
| p_win ortalaması | — | 0,4582 |
| **kalibrasyon sapması** | — | **0,0099** |

Kalan 2/600 susturma guard'ın asıl işidir (ısınmamış gösterge vb. uç vektörler).
`sources.neural` Brier'i artık ÖLÇÜLEBİLİR — öğrenme döngüsünün 4. kolu bilgi taşıyor.
Değer **tahmin gücünde değil, ölçüm hijyeninde**.

**DERS:** bir gölge katmanın SUSMASI iki apayrı şey olabilir — (a) model kenarsız,
(b) girdi dağılımı uyumsuz. Guard nedenini söylemeseydi (a) sanılacak ve YANLIŞ teşhis
konacaktı. Susan bir bileşen her zaman **nedenini** raporlamalıdır (Tur-12 "sessiz ölüm
yasak" dersinin ölçüm katmanındaki karşılığı).

### Tekrar üretmek için (§24)
```powershell
.\.venv\Scripts\python.exe -m research.build_live_dataset --min-rows 300 --out "data/meta_dataset_live_1m_<tarih>.csv"
.\.venv\Scripts\python.exe -m research.train_neural --tier deep_mlp --folds 5 --dataset "data/meta_dataset_live_1m_<tarih>.csv" --max-epochs 60 --save "data/neural_deep_mlp_live1m_<tarih>" --register
.\.venv\Scripts\python.exe -m nihai.execution.neural_shadow
```
Not: dosya adında `_1m_` bulunması ŞART — kart `data.source`'undan eğitim aralığını
regex ile okur (`_(1m|5m|15m|1h|1d)_`); adlandırma bozulursa provenance kaybolur.

---

## 25 — P2-11 ÖĞRENME EĞRİSİ (2026-07-31): 20× veri DSR'yi sıfırdan kımıldatmıyor

**Soru (denetim önerisi 11):** *"Deep model'i 97k/234k satırlık dataset'lerde yeniden eğit;
kapasite merdiveninin gerçekten VERİYLE mi yoksa GÜRÜLTÜYLE mi sınırlı olduğunu ayır."*
Ek bağlam (denetim "eksikler" §7): 234k ve 97k satırlık setler diskte dururken `deep_mlp`
**12k** satırlık sette eğitilmişti.

**Neden tek bir büyük eğitim yetmez:** büyük sette bir kez eğitip "AUC yine 0,50" demek iki
hipotezi AYIRMAZ. Ayrım için aynı kapasitenin ARTAN n'de OOS eğrisi gerekir
(`research/capacity_data_sweep.py`; alt-örnekleme KRONOLOJİK — rastgele örnekleme
purge/embargo'yu anlamsızlaştırırdı).

**Ölçüm — `deep_mlp` (186.702 param), purged+embargolu 3-fold, max_epochs=40:**

| n | PIT S&P500 1h (97k) AUC / DSR | Geniş evren 1h (234k) AUC / DSR |
|---|---|---|
| 12.000 | 0,5350 ± 0,0253 / **0,0** | 0,4939 ± 0,0092 / **0,0** |
| ~25.000 | 0,5167 ± 0,0117 / **0,0** | 0,5387 ± 0,0077 / 0,0249 |
| ~53.000 | 0,5025 ± 0,0045 / **0,0** | 0,5296 ± 0,0062 / 0,0032 |
| ~111.000 | 0,5099 ± 0,0084 / **0,0** *(57k)* | 0,5344 ± 0,0014 / **0,0** |
| en büyük | **0,5131 ± 0,0068 / 0,0** *(97k)* | **0,5246 ± 0,0070 / 0,0** *(234k)* |

**HÜKÜM (iki sette de aynı): `KENAR_YOK_HER_N`.** Deflasyondan sonra **hiçbir veri
boyutunda** beceri kanıtı yok. AUC 0,494–0,539 bandında salınıyor ama DSR sıfırda kalıyor.
**Sınırlayan ne veri MİKTARI ne model KAPASİTESİ — sinyalin kendisi** (ACI DERS #5, artık
20× veriyle sınanmış). Bu sinyal ailesinde veri toplamaya devam etmenin ölçülmüş gerekçesi
YOK; kapasiteyi büyütmenin (5,4M'lik `quant_transformer`) de yok.

### ⚠️ YENİ ACI DERS ADAYI — kendi kanıt kapım bozuktu (ACI DERS #8'in TEKRARI)

`_verdikt`'in İLK sürümü hükmü YALNIZ log-doğrusal eğime dayandırıyordu. Sonuç:

| Set | Eğim (tüm noktalar) | İlk hüküm | **Eğim (ilk nokta hariç)** |
|---|---|---|---|
| 97k PIT | −0,0223 | "küçük n şanstı" | **−0,0015** (DÜZ) |
| 234k | **+0,0177** | **"VERİ SINIRLI — veri toplamaya devam"** | **−0,0116** (İŞARET DÖNDÜ) |

5 noktalı bir eğride EN KÜÇÜK n'nin hata payı en büyüktür ve doğruyu **tek başına
döndürür**. 234k'da "daha çok veri işe yarıyor" iddiasının tamamı tek bir gürültülü
noktadan geliyordu — üstelik DSR o sette de sıfırdı, yani **etiket raporun geri kalanıyla
çelişiyordu**. Bu, ACI DERS #8'in birebir aynısıdır (AUC~0,50 iken DSR~1,0 çelişkisi).

**Düzeltme:** (1) ÖNCE kanıt kapısı — hiçbir n'de DSR > 0,05 yoksa hüküm `KENAR_YOK_HER_N`,
eğim tartışması açılmaz; (2) DSR geçerse eğim İKİ KEZ hesaplanır (tüm noktalar + ilk nokta
hariç) ve uyuşmazlarsa `EGRI_KARARSIZ` — uydurma hüküm verilmez.

**Ders:** *bir öğrenme eğrisinin eğimi, uç noktalarının hata payından küçükse hüküm
değildir.* Ve genel biçim yine aynı: **bir kanıt kapısı raporun geri kalanıyla çelişiyorsa
kapı bozuktur, sonuç değil.**

### Tekrar üretmek için (§25)
```powershell
.\.venv\Scripts\python.exe -m research.capacity_data_sweep --dataset "data/meta_dataset_pit_sp500_event_breakout_1h_2016-01-01_alpaca_hist.csv" --folds 3 --max-epochs 40 --json data/capacity_sweep_pit97k_<tarih>.json
.\.venv\Scripts\python.exe -m research.capacity_data_sweep --dataset "data/meta_dataset_extra_us_event_breakout_1h_2016-01-01.csv" --folds 3 --max-epochs 40 --json data/capacity_sweep_extra234k_<tarih>.json
```
Artefaktlar: `data/capacity_sweep_pit97k_2026-07-31.json`,
`data/capacity_sweep_extra234k_2026-07-31.json`.
NOT: bu koşulardan model KARTI kaydedilmez — 1h'te eğitilmiş bir araştırma kartı canlı 1m
köprüsünü çalardı (bkz. `neural_shadow` ARALIK TERCİHİ kilidi, §24 devamı).

---

## 26 — Sektör-içi LEAD-LAG ailesi ölçüldü (2026-08-03): kenar YOK — ACI-7 üçüncü kez

**edge_lab BACKLOG #3.** Aile: aynı sektördeki likit ("büyük") bir hissenin `t` barındaki
hareketi, daha küçük üyelerin `t+1` barına gecikmeyle yayılır mı? (Lo & MacKinlay 1990;
Hou 2007.) Veri: Alpaca SIP **saatlik, 2 yıl, 28 sembol, 7 sektör → 110.704 satır**,
taban oran 0,4414. Etiket: takipçinin t+1 getirisi > 5 bps.

### Tasarımda iki kilit (sonucun okunabilmesi için)

- **Lider hindsight'sız.** "NVDA yarıiletkenlerin lideridir" demek bugünün sıralamasını
  2 yıl öncesine taşımak olurdu. Lider her barda yeniden, **son 20 barın medyan
  dolar-hacmiyle** (geriye-bakan, t kapanışında bilinen) belirlendi. Liderlik gerçekten
  el değiştirdi: 8 üyeli yarıiletken sektöründe 8 sembolün hepsi bir noktada takipçi oldu.
- **Aynı-gün kısıtı.** t ve t+1 farklı gündeyse satır düşürüldü; yoksa etiket gece
  boşluğunu içerir ve §-BACKLOG#1'de zaten FAIL etmiş overnight hipoteziyle karışırdı.

### Sonuç: 5 kapıdan 2'si FAIL

| kapı | sonuç | sayı |
|---|---|---|
| purged_cv | geçti (kıl payı) | OOS-AUC **0,532** (eşik >0,53) |
| gate_time | geçti | base E −0,1095 → kapı E +0,1034, **alınan 29/33.212** |
| gate_symbol | geçti | base E −0,1223 → kapı E +0,0667, **alınan 30/29.617** |
| **dsr** | **FAIL** | **0,3058** (n=29; eşik >0,95) |
| **exec_lag_1** | **FAIL** | 1-bar gecikmede kapıyı geçen işlem **29 → 3**, ölçülemez |

### Neden bu bir kenar değil

1. **Seçicilik %0,087.** Kapı 33.212 işlemden 29'unu tutuyor. 29 gözlemle hesaplanan
   +0,10 expectancy'nin standart hatası farkın kendisi mertebesinde. "Kapı geçti" demek
   burada istatistiksel olarak bir şey söylemiyor.
2. **DSR bunu yakalıyor** (0,31 ≪ 0,95). ACI DERS #8'in AYNASI: orada kapı bozuktu ve
   rapora RAĞMEN geçiyordu; burada kapı raporun geri kalanıyla TUTARLI — sistem çalışıyor.
3. **exec-lag-1 belirleyici.** Bilgi tamamen sinyal-sonrası İLK barda. Saatlik barlarda
   bid-ask sıçraması / mikroyapı geri-dönüşü bu imzayı üretir; "gecikmeli yayılım" değil.

### Kalıcı ders

**ACI-7 üçüncü kez doğrulandı** (pairs §22 · giriş kayması §23 · lead-lag §26).
Bar-düzeyi adaylarda "ilk bar etkisi" artık **varsayılan şüphe**, istisna değil.

Ayrıca **YENİ okuma kuralı: kapı PASS dese bile `n_kept` okunmalı.** 29/33.212 gibi bir
seçicilik, PASS'i geçersiz kılar. Harness'ın `_MIN_KEPT=5` mutlak tabanı bu ölçekte
anlamını yitiriyor — oransal bir ölçüt eklenebilir AMA bu turda **kasıtlı yapılmadı**:
bir aday FAIL ettikten SONRA kapı eşiğini sıkılaştırmak, sonuca göre kural değiştirmektir.

**Öncelik notu:** evren survivorship-ŞİŞKİN (mevcut-likit 28 sembol), yani kenar LEHİNE
yanlı. Şişkin evrende bile geçemeyen aday temiz evrende geçmez → PIT tekrarının önceliği
DÜŞÜK. Aile kapandı.

### Tekrar üretmek için (§26)
```powershell
.\.venv\Scripts\python.exe -m research.edge_lab.build_leadlag_dataset --years 2 --interval 1h
.\.venv\Scripts\python.exe -c "import json; from research.edge_lab import run_candidate; from research.edge_lab.build_leadlag_dataset import FEATURE_COLS; print(json.dumps(run_candidate('leadlag_sector_v1','data/edge_leadlag_pilot.csv',FEATURE_COLS,'label',ts_col='ts',symbol_col='symbol'), ensure_ascii=False, indent=2))"
```
Artefaktlar: `data/edge_leadlag_pilot.csv`, `data/verdict_leadlag_v1.json`,
`research/edge_lab/SONUC_leadlag_pilot_2026-08-03.md`.

---

## 28 — VRP (varyans risk primi) ailesi ölçüldü (2026-08-27, Tur-51): kenar YOK — ve §"IV Rank bandı" bulgusu ÇÜRÜTÜLDÜ

BACKLOG #2 (`VRP`) iki haftadır *"veri kaynağı henüz belirlenmedi"* notuyla `BEKLIYOR`du.
Not ESKİMİŞTİ: 2026-08-13'te inen DoltHub `volatility_history` toplu CSV'si (2321 sembol ×
1,9M satır, her satırda `iv_current` + `hv_current`) tam da o yakıttı. Yakıt VARDI, defter
bilmiyordu — bu turun ilk bulgusu bir ölçüm değil bir MUHASEBE kusurudur.

### 28.1 Veri seti (`research/edge_lab/build_vrp_dataset.py`)

PIT/survivorship-TEMİZ: evren `data/alpaca_hist/*_1d_all.csv` (endeks-rebalans pilotunun
`sp500_as_of` evreni). **Doğrudan kanıt:** 609 sembolün **45'inin** fiyat serisi evren
sonundan >90 gün önce biter (TIF/LVMH, CXO/ConocoPhillips, XLNX/AMD, ALXN/AstraZeneca,
MXIM/ADI, KSU/CP, FLIR, WLTW, HFC…) — survivorship-yanlı evrende bunlardan HİÇBİRİ olmaz.

Etiket **fiyattan** hesaplanır (vekil kaydırma YOK): `rv_ileri` = karar barından SONRAKİ 21
barın yıllıklandırılmış RV'si → örtüşme yapısı gereği sıfır. `pnl = iv − rv_ileri − maliyet`
(1,0 oynaklık puanı), yön PRİM SATAN taraf. Örtüşmesiz (sembol başına her 21. gözlem),
`feature-lag 1`. Sonuç: **20.910 satır / 581 sembol / 2021-06-01→2026-07-09**.

**Primin KENDİSİ var:** taban **+0,00867** oynaklık puanı/işlem, kazanan oran **%60,5** —
1,0 puanlık maliyet DÜŞÜLDÜKTEN SONRA. Soru primin varlığı değil, SEÇİLEBİLİRLİĞİdir.

### 28.2 Verdikt: 5 kapıdan 3'ü FAIL

| kapı | sonuç | okuma |
|---|---|---|
| `purged_cv` | **FAIL** — OOS-AUC **0,514** (eşik >0,53) | yazı-turaya yakın |
| `gate_time` | GEÇTİ — base +0,0129 → kapı +0,0135, t=1,77, %91,0 tutuldu | eşiğin kılı |
| `gate_symbol` | GEÇTİ — base +0,0085 → kapı +0,0122, t=**7,32**, %85,3 tutuldu | görünüşte güçlü |
| `dsr` | **FAIL** — **0,083** (eşik >0,95), SR_fazlalık +0,0055, n_trials=16 | fazlalık gürültü |
| `exec_lag_1` | **FAIL** — t=**−2,61** | **fazlalık İŞARET DEĞİŞTİRİYOR** |

**ACI DERS #7 BEŞİNCİ KEZ.** Önceki dördünde (pairs §22 · lead-lag §26 · rebalans §27 ·
canlı §23) 1-bar gecikme fazlalığı ERİTİYORDU; burada **TERSİNE ÇEVİRİYOR**: sembol
bölmesinde t=+7,32 olan tutulan↔elenen farkı, bir bar gecikmede t=−2,61 oluyor. Kapının
seçtiği şey primin kalıcı bir bileşeni değil, tam o barın etrafındaki bir zamanlama
etkisidir. Kapı %85-91 tutuyor (seçicilik sorunu YOK, §26'nın tersi) — yani bu, büyük
örneklemli ve "sağlıklı görünen" bir PASS'in bile kenar olmadığının ikinci gözlemi (§27).

**Dürüst ifade:** varyans risk primi bu veride VARDIR ve pozitiftir; ama bir RİSK PRİMİdir,
zamanlanabilir bir anomali değil. Buradaki özelliklerle "ne zaman sat" sorusuna cevap YOK.

### 28.3 BULGU-51-1 — `hv_current` = geriye-bakan 30 işlem günü (ÖLÇÜLDÜ, artık tahmin değil)

`iv_rank_threshold_test.py` pencereyi *"sağlayıcı belgelemiyor"* diyerek TARIYORDU.
Fiyattan geriye-bakan RV ile eşleştirildi — **568.195 satır / 603 sembol**:

| W | 5 | 10 | 20 | 21 | 25 | **30** | 44 | 63 |
|---|---|---|---|---|---|---|---|---|
| korelasyon | 0,671 | 0,806 | 0,936 | 0,945 | 0,974 | **0,99962** | 0,947 | 0,900 |
| ort \|fark\| | 0,087 | 0,059 | 0,029 | 0,027 | 0,017 | **0,00143** | 0,025 | 0,036 |

Tüm evrende kor. **0,99504**, sembol ortancası 0,9998, <0,95 olan yalnız **8/600** ve
likidite çeyreklerinde fark YOK (hepsi ≈0,9997) → yapısal, likidite artefaktı değil.
Sonuç: `hv_current(t+W)` `(t+W−30, t+W]` ölçer; **W<30'da etiketin (30−W)/30'u karar
barından ÖNCEDİR**. Eski varsayılan W=21'de bu oran **%30**.

### 28.4 BULGU-51-3 — DÜZENSİZ seride `shift(N)` bir ZAMAN kaydırması DEĞİLDİR

Asıl kusur kirlenmeden büyük çıktı. Dolt serisi sembol başına fiyat barlarının ~%93,4'ünü
taşır ama boşluklar ÖBEKLİ (evren genelinde 1713 günün **623'ünde** kapsama %50 altı):

| `groupby(symbol).shift(-30)` | hedef | %25 | **ortanca** | %95 | ortalama | **std** | >60 gün |
|---|---|---|---|---|---|---|---|
| gerçekleşen takvim günü | 42-44 | 44 | **70** | 86 | 65,2 | **34,0** | **%59,6** |

Geri yönde `shift(21)` (hedef 29-31 gün) ortanca **49**, std 26,7. Yani ufuk sembole ve
döneme göre DEĞİŞİYOR — sabit-ufuklu bir ölçüm hiç yapılmamıştı. **Ders kendi kodumuza da
uygulandı:** bu turun ilk veri seti aynı kusuru taşıyordu; `_to_price_calendar` dolt'u TAM
fiyat takvimine taşıyıp sınırlı (5 işlem günü) nedensel ffill uygular → `shift(N)` TAM N
işlem günü olur. Kilit: `tests/test_tur51_vrp_veri_seti.py::test_izgarada_N_satir_TAM_N_ISLEM_GUNUDUR`.

### 28.5 BULGU-51-4 — "IV Rank bandı TERS + ilişki MONOTON" bulgusu VEKİLİN ARTEFAKTIYDI

CLAUDE.md 2026-08-13'ten beri şunu yazıyordu: bant `[30,50]` HER W'de tabandan KÖTÜ
(W=21'de gün-bazlı **t=−7,35**), ilişki MONOTON, en iyi `[0,10)` en kötü `[90,100)` →
*"`strategy.py` UYGULAMASI ters, bonusu çevirmek kullanıcı kararıdır"*.

**İZOLE KONTROL** — AYNI satırlar, aynı evren (584 sembol), aynı dönem, aynı **W=30** (her
iki etiket de kirlenmemiş), aynı rank, aynı exec-lag=1, aynı örtüşmesizlik. Tek değişken
ETİKET TAHMİNCİSİ (n=15.943):

| dilim | VEKİL `hv_current(t+30)−iv` | | TEMİZ `rv_ileri(30)−iv` | |
|---|---|---|---|---|
| | ort | gün-t | ort | gün-t |
| `[0,10)` | +0,00597 | **+4,43** | −0,01578 | +1,28 |
| `[40,50)` | −0,02265 | −2,51 | −0,01829 | −0,55 |
| `[80,90)` | −0,05007 | −2,81 | −0,02111 | −2,16 |
| `[90,100)` | −0,05787 | −2,61 | −0,02103 | −1,37 |
| **bant [30,50]** | −0,01748 | **−2,29** | −0,01544 | **+0,61** |

İki etiketin korelasyonu yalnız **0,7177**. Nihai (düzenli ızgara, W=21, n=20.910):
bant gün-bazlı **t=−0,17**, dilimlerde monoton desen YOK.

**Sonuç: bant hiçbir yönde bilgi TAŞIMIYOR — ne doğrudur ne terstir.** "Bonusu çevir"
önerisi GERİ ÇEKİLMİŞTİR: çevirmek, çürütülmüş bir ölçüme dayanarak yeni bir iddia kurmak
olurdu. Bonusu KALDIRMAK ayrı ve meşru bir sorudur (kanıtsız bir bileşen skoru kirletir),
ve **KARAR VERİLDİ: varsayılan `IV_REGIME_BONUS=off`** (Tur-51). Kanıtsız bir bileşen
skordan çıkarıldı — kenar iddiası değil, iddiasızlık. Üç kip de testte kilitli; `band`
eski davranışa döndürür. Tur-31'in "ölçüm yokluğu ödüllendirilmez" testi `off` altında
gözlemlenemez hâle geleceği için kipi kendi içinde açıkça `band`a sabitler (güvence
gevşetilmedi, yeniden çapalandı).

### 28.6 BULGU-51-2 — sağlayıcı `*_month_ago` kolonlarını 2024'ten beri DOLDURMUYOR

NaN oranı yıl-yıl: 2019-2022 ≈%0 · 2023 %10 · **2024/2025/2026 %100**. Bu kolonlara bağlanan
hat 2023-11'de SESSİZCE biter — hata YOK, yalnız daha kısa bir tarih aralığı. Bu turun ilk
veri seti tam bunu yaptı: satırların %55'i düştü, 2,7 yıl kayboldu, hiçbir uyarı çıkmadı.
Doldurulduğu yerde bile tanımı belirsiz: `iv_month_ago ≈ iv_current(t−k)` eşleştirmesi
k∈[18,23]'te MONOTON artan hata verir (en iyi k=18 = ARAMA ARALIĞININ SINIRI = uydurma).
Değişim özellikleri artık `iv_current`/`hv_current` serisinden hesaplanıyor.

### Tekrar üretmek için (§28)

```bash
python -m research.edge_lab.build_vrp_dataset          # veri seti (PIT, örtüşmesiz)
python -m research.edge_lab                            # defter: 4 aile, sıradaki n_trials
pytest tests/test_tur51_vrp_veri_seti.py -q            # kilitler
```


---

## 29 — Açığa satış AKIŞI (FINRA günlük) ailesi ölçüldü (2026-08-29, Tur-53): kenar YOK — ve literatürün İŞARETİ bu evrende YOK

### 29.1 Neden bu aile

Dış bir LLM sohbetinden gelen 12 maddelik "edge nasıl bulunur" listesi projenin kendi kaydına
karşı triyaj edildi (`docs/DIS_ONERI_TRIAJI_2026-08-29.md`): 7 madde ZATEN PIT veriyle ölçülmüş
ve kenar bulunamamış, 3'ü bu piyasa/altyapı için uygulanamaz (market making · kripto funding/
likidasyon/perp-OI · ücretsiz L2 defteri), 1'i (execution) projenin tek doğrulanmış kazancını
zaten üretmiş (§23 · Tur-39). Geriye ölçülmemiş **TEK** madde kaldı: *"trade flow /
order-flow imbalance"*.

Kripto order-book'u bu sistemde yok; ama ABD hissesinde **ücretsiz, anahtarsız ve GÜNLÜK**
bir akış ölçüsü var: FINRA konsolide açığa satış hacmi
(`cdn.finra.org/equity/regsho/daily/CNMSshvol<YYYYMMDD>.txt`, gün başına ~12.000 sembol).
Literatür desteği yüksek — Boehmer-Jones-Zhang 2008 (JF, *"Which Shorts Are Informed?"*),
Diether-Lee-Werner 2009 (RFS): **açığa satanlar ortalamada bilgilidir**, ağır açığa satılan
isimler sonraki günlerde düşük getiri verir.

Aile önceki beşten YAPISAL olarak farklıdır: olay (kazanç/rebalans/insider) veya fiyat türevi
(lead-lag/VRP) değil, **KONUMLANMA AKIŞI**. Defterde ayrık aile sayılır (5 → 6).

### 29.2 Veri seti (`research/edge_lab/build_shortflow_dataset.py`)

PIT/survivorship-TEMİZ evren (`data/alpaca_hist/*_1d_all.csv`, bugün endekste olmayan isimler
DAHİL). **1.378 işlem gününün TAMAMI indirildi, eksik gün 0** → 777.556 sembol-gün akış /
593 sembol. Sinyal: SVR = short/total hacim oranının sembol-içi 60 günlük **z-skoru** (ham
seviye ~%45-55'tir ve sembole göre sabit bir taban taşır — piyasa yapıcı hedge'i; z-skor
zorunludur). Yön literatürün işaretiyle bağlandı: z>0 → SHORT, z<0 → LONG.

Kilitler: **feature-lag 1** (FINRA D dosyasını D KAPANDIKTAN sonra yayımlar → giriş D+1
kapanışı, yayımla giriş arasında tam bir seans) · **örtüşmesizlik** (sembol içinde ≥ ufuk) ·
**beta kontrolü ŞEMADA** (`mkt_ret`/`excess`; puanlama piyasa-nötr FAZLALIK üstünden) ·
**ödünç maliyeti** short bacağa (100 bps/yıl — ağır açığa satılan isimler tam da pahalı ödünç
olanlardır, D'Avolio 2002, yani bu varsayılan İYİMSER) · **eşik/ufuk taraması YOK** (ufuk
5 bar ön-kayıtlı; tarama `n_trials`'ı ayrıca çarpardı, §23).

**147.318 satır / 593 sembol / 2021-03-31 → 2026-06-23**, short bacak %51,7.
Taban ("her satırı al" triviyal kuralı): ort. pnl **−0,1266%**, kazanan **%48,4**.

### 29.3 Verdikt: **5/5 kapı FAIL** (insider'dan sonra bunu yapan ikinci aday)

| kapı | sonuç |
|---|---|
| `purged_cv` | **FAIL** — OOS-AUC **0,509** (5 fold; eşik >0,53) |
| `gate_time` | **FAIL** — kapıyı geçen işlem **0** (gerekli 221 = %0,5 × 44.196) |
| `gate_symbol` | **FAIL** — kapıyı geçen işlem **0** (gerekli 223 = %0,5 × 44.546) |
| `dsr` | **FAIL** — kapı-sonrası seri YOK, hesaplanamadı |
| `exec_lag_1` | **FAIL** — 1-bar kaydırmada da 0 işlem; lag0 kapı-E yok → etki ölçülemez |

### 29.4 "0 satır tutuldu" burada bir KUYRUK ARTEFAKTI DEĞİL — model tabandan kımıldayamıyor

§26'da 0'a yakın seçicilik uç-kuyruk örneklemesiydi; burada sebep başka ve daha nettir.
Zaman bölmesinde lojistik yeniden fit edildi (%70/%30, standardize): test diliminde
üretilen olasılıkların tamamı **[0,4214 · 0,5559]** aralığında, ortalama **0,4830**
(taban 0,4850) ve 0,55'i aşan **1 satır** var (44.196 içinde). Yani kapı "uç kuyruğu
örnekliyor" değil, **hiçbir satıra 0,55 diyemiyor**.

Standardize katsayıların hepsi |c| ≤ **0,0225** (anlamlı bir lojistik katsayı O(0,1-1)'dir)
ve akış özellikleri listenin ALTINDA: `f_svr_z` +0,0089 (6/9), `f_svr_absz` +0,0036 (7/9).
En büyük katsayı `f_is_short` (+0,0225) — o bir bilgi değil bacak KUKLASIDIR (ödünç
maliyeti + sürüklenme farkı). Model akışa bakmıyor çünkü bakacak bir şey yok.

### 29.5 BULGU-53-1 — satır-t ile GÜN-t **İŞARET DEĞİŞTİRİYOR** (Tur-27'nin bir adım ötesi)

Koşulsuz iz (kapı yok, yalnız yön kuralı), bağımsızlık birimi TARİH alınarak:

| bacak | n | fazlalık | satır-t | **GÜN-t** |
|---|---|---|---|---|
| short (z>0) — literatür NEGATİF bekler | 76.109 | **+0,0173%** | +1,10 | **−0,55** |
| long (z<0) — literatür POZİTİF bekler | 71.209 | **−0,0153%** | −0,92 | **+1,82** |
| \|z\|>2 dilimi (yönlü) | 7.636 | −0,2250%/gün-ort | — | −1,12 |

İki okuma birden var: (a) **literatürün işareti bu evrende/ufukta YOK** — satır düzeyinde
her iki bacak da BEKLENENİN TERSİ yönde (ikisi de |t|<1,2 = gürültü); (b) satır-t ile
gün-t **işaret değiştiriyor**. Tur-27 bağımsızlık biriminin t'yi 11× ŞİŞİRDİĞİNİ ölçmüştü;
burada birim değişimi t'nin BÜYÜKLÜĞÜNÜ değil YÖNÜNÜ çeviriyor: satır ortalaması, çok
satırlı günlerin (kesitsel olarak korele) baskın olduğu bir ortalamadır. **Kural genişledi:
bir olay ailesinde satır düzeyi t, tarih düzeyi t ile birlikte raporlanmadan OKUNAMAZ —
yalnız şişebilir değil, TERS de işaret edebilir.**

### 29.6 İddianın SINIRI (bu bir "BJZ yanlış" iddiası DEĞİLDİR)

Ölçülen şey **açığa satış HACMİdir, açığa satış POZİSYONU (short interest) değil**; ve
günlük konsolide hacmin büyük bir kısmı piyasa yapıcı hedge'idir (seviyenin ~%50'de
oturmasının nedeni budur). BJZ order düzeyinde, kurumsal akışı ayırt edebilen veriyle
çalışır. Dürüst ifade: **ücretsiz günlük vekil, bu evrende ve 5 barlık ufukta etkiyi
TAŞIMIYOR.** Dosyadaki `ShortExemptVolume` kolonu okundu ama KULLANILMADI ve bu tur ayrı
bir varyant olarak KOŞULMADI — koşulsaydı gizli bir çoklu-test olurdu; ayrı bir aday
olarak kaydedilmesi gerekir.

### Tekrar üretmek için (§29)

```bash
python -m research.edge_lab.build_shortflow_dataset --kapi
python -m research.edge_lab          # defter: 6 ayrık aile → sıradaki n_trials = 7×4 = 28
```


---

## 30 — Opsiyon YÜZEYİNİN yön sinyali ölçüldü (2026-08-29, Tur-54): kenar YOK — ve verdikt yalnız 1,7 YILI kapsıyor

### 30.1 Neden bu aile, ve VRP'den farkı

Tur-53 triyajının bıraktığı en yüksek beklenen değerli aday buydu: literatür desteği yüksek,
veri kaynağı zaten kodda tanımlı ama HİÇ indirilmemişti. İki klasik bulgu tek ailede:
**Cremers-Weinbaum 2010** (put-call parite sapması `VS = IV_call − IV_put`; yüksekken pozitif,
düşükken negatif ileri getiri) **YÖN KURALI**, **Xing-Zhang-Zhao 2010** ("smirk"
`SKEW = IV(OTM put) − IV(ATM call)`; dik gülümseme düşük getiri) ÖZELLİK.

**§28'in (VRP) tekrarı DEĞİLDİR:** orada etiket ileri GERÇEKLEŞEN OYNAKLIKTI (prim satıcısının
sorusu), burada etiket HİSSENİN ileri getirisidir (yön sorusu). Aynı kaynak, iki ayrı soru.

### 30.2 Veri: 10,7 GB tarandı, yüzey diske alındı

`option_chain` toplu CSV **10.193,5 MB / 2.707 sn** ile indirildi (**OneDrive DIŞINA** —
`DOLT_CACHE_DIR`; 10 GB'lık bir araştırma önbelleği senkronlanacak bir şey değildir) ve
**116.487.570 satır** ~5,5 dakikada akıtıldı. Ürün: **514.779 gün×sembol yüzey ölçüsü /
590 sembol / 2021-01-01 → 2026-06-30**, ortalama 6,0 eşleşen call-put çifti. Yüzey serisi
artık `data/edge_lab_optskew_yuzey.csv` olarak SAKLANIYOR — 10,7 GB'lık taramanın ürünü her
teşhis sorusunda yeniden ödenmemeli (Tur-35: *"ölçüm bütçesini OKUYAN da öder"*).

Ön-kayıtlı seçim (HİÇBİRİ taranmadı): vade DTE ∈ [20,60] içinde 30'a en yakın · ATM call
|Δ−0,50| ≤ 0,15 · OTM put |Δ+0,20| ≤ 0,10 · VS için call Δ ∈ [0,20 · 0,80] ve ≥2 eşleşen
çift · ufuk 5 bar · `bid>0 ∧ ask>bid` kotasyon kalitesi.

**Vektörleştirme ZORUNLUYDU ve ölçüldü:** grup-döngülü referans gövde 31.911 grubu 378 sn'de
işliyordu (~12 ms/grup) → tam dosyada ~2,3 saat. `transform("min")` + tek `merge` ile aynı
girdide **1,9 sn = 204×**, çıktı satır sayısı birebir aynı. Referans gövde (`_yuzey_olc`)
SİLİNMEDİ: vektör yolunun ORAKLIDIR ve denklik testle kilitlidir.

### 30.3 BULGU-54-1 — sağlayıcı kapsaması 2024'te SIÇRIYOR; verdikt 5,5 yılı DEĞİL 1,7 yılı kapsıyor

Veri seti yüzeyin başladığı yerde değil, **2024-10-14**'te başlıyor. Sebep ölçüldü:

| yıl | işlem günü | yüzey satırı | aktif sembol | KAPSAMA | 60g pencerede ≥40 gözlem eşiğini (%66,7) geçen sembol payı |
|---|---|---|---|---|---|
| 2021 | 252 | 74.210 | 525 | %56,1 | **%0,0** |
| 2022 | 251 | 74.461 | 538 | %55,1 | **%0,0** |
| 2023 | 250 | 73.533 | 543 | %54,2 | **%0,0** |
| 2024 | 252 | 92.230 | 553 | %66,2 | %79,2 |
| 2025 | 250 | 134.261 | 558 | %96,2 | %92,1 |
| 2026 | 123 | 66.084 | 555 | %96,8 | %94,6 |

DoltHub zinciri 2021-2023'te işlem günlerinin ancak **~%55'ini** taşıyor; 60 günlük z-skor
penceresi 40 gözlem istediği için o üç yılda **hiçbir sembol tek satır bile üretemiyor**.
2024 ortasından sonra kapsama ~%96'ya çıkıyor ve veri seti orada başlıyor.

**Bu, BULGU-51-2'nin kardeşidir** (*"boş bir sağlayıcı kolonu HATA vermez, yalnız TARİH
ARALIĞINI kısaltır"*): orada eksik olan bir KOLONDU, burada eksik olan TAKVİMDİR; imza aynı —
istisna yok, uyarı yok, yalnız 3,7 yıl daha kısa bir veri seti. **Verdikt bu yüzden
2024-10-14 → 2026-06-23 dönemini kapsar, 2021-2026'yı DEĞİL.**

**AÇIK KALEM (adayı olmayan bir tura):** `harness._stage_coverage` veri setinin KENDİ
[min,max] aralığında boş ay arar; "istenen aralığın 3,7 yılı hiç doğmadı" durumunu göremez.
Kapı bu turda BİLEREK değiştirilmedi — bir aday FAIL ettikten sonra kapı sıkıştırmak,
sonuca göre kural değiştirmektir (Tur-23/Tur-25 kuralı).

### 30.4 Verdikt: **5/5 kapı FAIL**

| kapı | sonuç |
|---|---|
| `purged_cv` | **FAIL** — OOS-AUC **0,513** (eşik >0,53) |
| `gate_time` | **FAIL** — kapıyı geçen işlem **3** (gerekli 65) |
| `gate_symbol` | **FAIL** — kapıyı geçen işlem **3** (gerekli 66) |
| `dsr` | **FAIL** — kapı-sonrası seri kurulamadı |
| `exec_lag_1` | **FAIL** — 1-bar kaydırmada **0** işlem |

Veri seti **43.140 satır / 551 sembol**, short bacak %55,4; taban ort. pnl **−0,0888%**,
kazanan **%48,9**. §29'daki teşhis burada da aynen geçerli: zaman bölmesinde yeniden fit
edilen lojistiğin ürettiği olasılıklar **[0,4291 · 0,5769]** aralığında sıkışıyor
(ortalama 0,4891 ↔ taban 0,4890) ve 0,55'i aşan yalnız **5** satır var. Standardize
katsayıların hepsi ≤ **0,0529** ve en büyüğü yine `f_is_short` — bilgi değil bacak
KUKLASI. Yön özelliği `f_vs_z` **+0,0063 ile 10 özelliğin 9.'su**.

### 30.5 Koşulsuz iz: bu kez İŞARET DOĞRU ama BÜYÜKLÜK yok (§29'un tersi)

| bacak | n | fazlalık | satır-t | GÜN-t | literatür |
|---|---|---|---|---|---|
| LONG (vs_z>0) | 19.244 | **+0,0110%** | +0,31 | +0,96 | pozitif ✓ |
| SHORT (vs_z<0) | 23.896 | **−0,0312%** | −1,03 | −0,16 | negatif ✓ |
| \|z\|>2 dilimi (yönlü) | 2.559 | −0,0117%/gün-ort | — | −0,11 | — |

**§29'dan (açığa satış akışı) YAPISAL olarak farklı bir başarısızlık:** orada işaret TERSTİ;
burada **işaret literatürün dediği yönde** ama büyüklük 1-3 baz puan / 5 gün — 10 bp'lik
round-trip maliyetin altında ve gün-kümelenmiş t'de sıfırdan ayırt edilemiyor. Doğru okuma
*"parite sapması bilgi taşımıyor"* değil, *"taşıdığı bilgi bu evrende ve bu ufukta işlem
maliyetinden küçük"*tür.

**Smirk (Xing-Zhang-Zhao) AYRI okundu ve TERS çıktı:** dik gülümseme (`skew_z>1`, n=5.936)
fazlalığı **+0,0939%**, düz gülümseme (`skew_z<−1`, n=4.517) **+0,0565%** — literatür dikin
DAHA DÜŞÜK olmasını bekler. Fark 3,7 bp ve yine gürültü mertebesinde.

### Tekrar üretmek için (§30)

```bash
DOLT_CACHE_DIR='C:\Users\Kagan\dolt_cache' python -c "from nihai.data import dolt_options as D; D.bulk_csv('option_chain')"
python -m research.edge_lab.build_optskew_dataset --chain-csv 'C:\Users\Kagan\dolt_cache\option_chain__master.csv' --kapi
python -m research.edge_lab          # defter: 7 ayrık aile → sıradaki n_trials = 8×4 = 32
```


---

## 31 — Üç aile daha (2026-08-29, Tur-55): BAB×makro ve mevsimsellik ölçüldü, 13F ÖLÇÜLEMEDİ

Tur-53 triyajının bıraktığı kuyruğun tamamı koşuldu. Üçünün de sonucu farklı bir TÜRDE.

### 31.1 BAB × makro rejim (BACKLOG #10) — 4/5 kapı FAIL, ve literatürün İŞARETİ TERS

Yön kuralı Frazzini-Pedersen 2014 (düşük beta risk-ayarlı olarak yüksek betayı geçer);
makro rejim (FRED `T10Y2Y` + `NFCI`) yön ÜRETMEZ, özellik olarak girer. **FRED artık
anahtarsız çalışıyor** — `nihai/data/fred_data.py` (Tur-53'te doğrulanan uç nokta kalıcı
modüle alındı). Sınır dürüstçe taşınıyor: uç nokta her seride `cosd`'u dinlemiyor
(`BAMLH0A0HYM2` yalnız son ~3 yılı verdi → `attrs["kirpildi"]` bayrağı; o seri
KULLANILMADI).

**ÖLÇÜT SEÇİMİ AÇIK:** diğer ailelerde `excess = fwd_ret − mkt_ret`tir; BAB'da bu ölçüt
long bacağı yapısal olarak cezalandırırdı (betası 0,7 olan hisse yükselen piyasada tanımı
gereği geride kalır). Literatürün iddiası ALFA olduğu için burada
`excess = fwd_ret − beta × mkt_ret`; düz sürüm `excess_duz` olarak ŞEMADA durur.

**32.417 satır / 595 sembol / 2021-08-23 → 2026-05-29.** Verdikt:

| kapı | sonuç |
|---|---|
| `purged_cv` | FAIL — OOS-AUC 0,518 |
| `gate_time` | FAIL — kapı base'in ALTINDA (E −0,0050 → −0,0180), Welch t=**−3,75** |
| `gate_symbol` | PASS — E −0,0010 → +0,0073, t=1,85, %7,6 seçicilik |
| `dsr` | FAIL — **0,0** (SR_fazlalık −0,1286) |
| `exec_lag_1` | FAIL — etki **−0,0268**, seçicilik %8,0 → %0,5 |

**İşaret bu dönemde TERS:** beta-ayarlı fazlalık düşük-beta bacakta **−0,0754%**,
yüksek-beta bacakta **+0,0668%** (literatür tersini bekler). Düz fazlalıkta fark
−0,4424% ↔ +0,4373% — yani beta ayarı işini yaptı (88 bp'lik yapısal farkı 14 bp'ye
indirdi) ve işaret ayardan SONRA da literatüre aykırı. 2021-2026 ABD büyük-sermaye
evreninde yüksek beta kazandı; bu bir kenar değil, dönemin kendisidir.

### 31.2 Mevsimsellik (BACKLOG #7) — ÜÇ kapıyı geçen İLK aile, sonra ACI-7 ALTINCI kez

Yön kuralı Heston-Sadka 2008 (bir hissenin geçmiş yıllarda AYNI takvim ayında verdiği
ortalama getiri, bu yılın aynı ayını kesitsel olarak öngörür). **Aile bilerek daraltıldı:**
takvim etkilerinin çoğu (ay dönümü, Ocak, tatil-öncesi) PİYASA GENELİdir ve piyasa-nötr
fazlalık üstünde tanımı gereği sıfıra gider — bu harness onları ÖLÇEMEZ. Ölçülen şey
*"takvim ailesi"* değil *"takvimin KESİTSEL kolu"*dur.

**22.386 satır / 577 sembol / 2023-02-01 → 2026-05-01** (en az 2 yıl geçmiş şartı serinin
başını yiyor). Verdikt:

| kapı | sonuç |
|---|---|
| `purged_cv` | **PASS** — OOS-AUC **0,534** (eşik >0,53) — *bu kapıyı geçen İLK aile* |
| `gate_time` | **PASS** — E −0,0030 → **+0,0068**, Welch t=**3,73**, seçicilik %21,0 |
| `gate_symbol` | **PASS** — E −0,0022 → **+0,0175**, t=2,07, seçicilik %3,1 |
| `dsr` | FAIL — **0,8558** (n=1412, SR_fazlalık +0,0882, n_trials=**36**) |
| `exec_lag_1` | FAIL — kapı-E **+0,0068 → −0,0014**, Welch t **3,73 → 0,40** |

**ACI DERS #7 ALTINCI KEZ** (pairs §22 · giriş kayması §23 · lead-lag §26 · rebalans §27 ·
VRP §28 · mevsimsellik §31). Desen tanıdık: üç kapıda sağlıklı görünen, %21 seçicilikle
BÜYÜK örneklemli bir fazlalık, bir bar gecikmede tamamen buharlaşıyor.

### 31.3 BULGU-55-1 — kenar-keşif DEFTERİ ilk kez bir KAPININ verdiktini ÇEVİRDİ

Tur-46 DSR deflasyonunu defterdeki ayrık aile sayısıyla çarpmaya başlamıştı ve o gün
*"geçmişte koşulan üç ailenin verdikti DEĞİŞMEZ"* diye kayda geçmişti — mekanizma bugüne
kadar hiçbir şeyi çevirmemişti. Mevsimsellik ailesinde ÇEVİRDİ (aynı seri, n=1412,
SR_fazlalık +0,0882):

| n_trials | 4 (eski sabit) | 8 | 16 | 24 | 32 | **36 (bugün)** | 40 |
|---|---|---|---|---|---|---|---|
| DSR | **0,9845 PASS** | **0,9600 PASS** | 0,9206 fail | 0,8906 fail | 0,8664 fail | **0,8558 fail** | 0,8461 fail |

Eski sabitle bu kapı GEÇİLİRDİ. **İddianın sınırı:** defter bir KAPININ verdiktini çevirdi,
AİLENİN verdiktini değil — `exec_lag_1` zaten düşürüyordu. Yine de bu, "arama sürdükçe
çıta sıkışır" ilkesinin ilk somut ölçümüdür ve sayaç dekoratif olmadığının kanıtıdır.

### 31.4 13F kurumsal kümelenme (BACKLOG #6) — ÖLÇÜM YAPILAMADI, "kenar yok" DEĞİL

SEC çeyreklik toplu 13F setleri VAR (`.../form-13f-data-sets/01mar2026-31may2026_form13f.zip`,
~100 MB/çeyrek; adlandırma çeyrek etiketi değil TARİH ARALIĞIDIR — `2026q2_form13f.zip`
404 verir). Ama `INFOTABLE` şeması `NAMEOFISSUER · CUSIP · FIGI · VALUE · SSHPRNAMT`
taşır ve **TICKER YOKTUR**; `FIGI` kolonu örneklenen çeyrekte BOŞ.

Evrene bağlamanın tek ücretsiz yolu isim eşlemesidir (SEC `company_tickers.json`).
ÖLÇÜLDÜ (2026 Mart-Mayıs seti, 3.822.885 satır / 35.649 ayrık CUSIP):

* normalize isimle TAM eşleşen CUSIP: **%30,7**
* **%42,2** CUSIP birden fazla FARKLI isimle geliyor (dosyalayanlar arası yazım farkı)
* eşleşen ticker'ların **%69,4'ü birden fazla CUSIP alıyor** → eşleme BİRE-BİR DEĞİL

Yani isim tabanlı harita, BAĞIMSIZ DEĞİŞKENİN KENDİSİNİ yanlış ölçerdi (farklı menkul
kıymetlerin pozisyonlarını aynı tickerda toplayarak). Bu gürültülü bir sinyal değil,
YANLIŞ ÖLÇÜLMÜŞ bir sinyaldir. **Verdikt yok — ölçüm yok.** BACKLOG #6 `BEKLIYOR`dan
`VERI_ONKOSULU_YOK`a alındı: lisanslı bir CUSIP↔ticker haritası olmadan bu aile
koşulamaz. (Tur-31 ilkesi: ölçüm yokluğu ölçüm gibi raporlanamaz.)

### Tekrar üretmek için (§31)

```bash
python -m research.edge_lab.build_babmacro_dataset --kapi
python -m research.edge_lab.build_seasonality_dataset --kapi
python -m research.edge_lab          # defter: 9 ayrık aile → sıradaki n_trials = 10×4 = 40
```

---

## 32 — VIX VADE YAPISI × swing kırılması ölçüldü (2026-09-06, Tur-74): kenar YOK — ve dış listenin iddiası GÜN düzeyinde İŞARET DEĞİŞTİRİYOR

Dış geliştirme listesinin **Madde 9**'u somut bir kural öneriyordu:

> *"Contango (VIX < VIX3M) boğa swinglerine izin verirken, Backwardation (VIX > VIX3M)
> TÜM swing alımlarını kilitler."*

Tur-74 vade yapısını KARAR-DIŞI rapor olarak bağladı; bu bölüm iddianın KENDİSİNİ kanıt
kapısına sokar. **Aile: `vixts_v1`** (defterin 10. ayrık adayı).

### Kurgu (ön-kayıtlı, tarama YOK)

`research/edge_lab/build_vixts_dataset.py` — PIT/survivorship-temiz evren (595 sembol),
taban kural **20 barlık en yüksek kırılması = swing ALIMI** (Madde 9'un kilitlemek
istediği şey), ufuk **5 bar**, maliyet %0,1, feature-lag 1, sembol içi örtüşmezlik.
Rejim YÖN ÜRETMEZ, ÖZELLİK olarak girer (`babmacro` disiplini). Doğan veri:
**43.066 satır / 595 sembol / 1277 GÜN** (2021-03-03 → 2026-06-22).

**Ölçüt HAM getiridir, piyasaya göre fazlalık DEĞİL** — ve bu ailenin şart koştuğu
seçimdir: Madde 9 bir PİYASA ZAMANLAMASI iddiasıdır, piyasaya göre fazlalık almak
ölçülmek istenen etkiyi tanım gereği görünmez kılardı. Boğa artefaktı farklılaştırmayla
değil KIYAS TABANIYLA nötrlenir (harness "kapıyı hiç uygulama, her kırılmayı al"
triviyal kuralına karşı ölçer — sürüklenme iki kolda da vardır).

### BULGU: satır düzeyi iddiayı DOĞRULUYOR, gün düzeyi TERSİNE ÇEVİRİYOR

Taban (her kırılmayı al): ort. pnl **+0,1276%**, kazanan %51,7.

| birim | contango | backwardation | n |
|---|---:|---:|---|
| **SATIR** | +0,1294% | **+0,0289%** | 42.269 / 797 |
| **GÜN** | +0,1384% | **+0,1730%** | 1217 / 60 |

Satıra bakan biri Madde 9'u doğrulanmış sayardı (backwardation'da getiri 4,5 kat düşük).
Güne bakınca **işaret dönüyor**: backwardation günleri ORTALAMA OLARAK DAHA İYİ.
Sebep BULGU-53-1'in ta kendisidir — vade yapısı özellikleri PİYASA GENELİDİR, aynı gün
kırılan onlarca sembol AYNI değeri taşır, dolayısıyla satır ortalaması çok-satırlı
günlerce baskılanır. Etkin örneklem 43.066 değil **60 backwardation GÜNÜdür**.

### Kapılar

| kapı | sonuç |
|---|---|
| `veri_kapsami` | **PASS** — 64 ayda 0 boş ay; istenen pencere kapsandı (son kesme %0,4) |
| `purged_cv` | **FAIL** — ortalama OOS-AUC **0,505** (yazı-tura; eşik >0,53) |
| `gate_time` | **FAIL** — kapı-E +0,0009 ↔ base +0,0017, Welch **t=−1,39** (kapı ZARAR VERİYOR) |
| `gate_symbol` | PASS ama **seçicilik %2,9** (385/13.117) → Tur-23 kuralı: `n_kept` okunmadan PASS okunamaz |
| `dsr` | **FAIL** — DSR **0,0005**, SR_fazlalık −0,0159, `n_trials=40` |
| `exec_lag_1` | **FAIL** — t=−0,77 |

**VERDİKT: KAPI(LAR) GEÇİLEMEDİ — kenar YOK.** `gate_symbol`ün tek PASS'i uç-kuyruk
örneklemidir ve zaten `gate_time` ile TERS yöndedir; iki bölmede birden geçmeyen bir
kapı ACI DERS #3 gereği dayanıklı sayılmaz.

### Sonuç: Madde 9'un kapısı KURULMADI

Öneri ölçüldü ve karşılığı çıkmadı. `nihai/signals`, `nihai/risk`, `nihai/execution`
altında `cboe_data` importu **testle YASAK** (`tests/test_tur74_vade_yapisi.py`);
vade yapısı yalnız `/api/quant/regime` üzerinden RAPORLANIR.

**Bu, "VIX vade yapısı hiçbir yerde işe yaramaz" demek DEĞİLDİR** — ölçülen şey BU
evrende, BU taban kuralla (20-bar kırılma), BU ufukta (5 bar) ve BU dönemde
(2021-2026, yalnız 60 backwardation günü) bir kapı olarak değeridir. 60 gün, bir rejim
iddiasını çürütmek için küçük bir örneklemdir; dürüst ifade *"bu kurguda ayırt
edilemedi"*dir.

### Tekrar üretmek için (§32)

```bash
python -m nihai.data.cboe_data                              # önbelleği doldur
python -m research.edge_lab.build_vixts_dataset --kapi
python -m research.edge_lab          # defter: 10 ayrık aile
```

## 33 — KALINTI MOMENTUMU ölçüldü (2026-09-19, Tur-86): 5/5 kapı FAIL — ve panel EVRENİNDE bir look-ahead bulundu

Aday, bir dış (Gemini) benchmark sohbetinin triyajından çıktı: sohbet 5,4M→2B transformer'ların
HIZINI ölçmüş ve daha büyük model önermişti; projenin kendi kapasite süpürmesi (2026-07-31)
ise "sınır veri de kapasite de değil, sinyalin kendisi" diyordu. Doğru eksen yeni SİNYALdi ve
defterdeki on ailenin hiçbiri momentum değildi (BACKLOG #11).

### Kurgu (ön-kayıtlı, tarama YOK — BACKLOG #11 satırı sonuçtan ÖNCE yazıldı)

Blitz-Huij-Martens 2011 (JEF): her ay sonu D'de aylık aşırı getiri son **36 ayda** FF3
(Mkt-RF·SMB·HML, sabitli EKK) üzerine regres edilir; skor = **t−12…t−2** kalıntı toplamı /
aynı 11 ayın kalıntı std'si. Kesitsel **medyan** üstü LONG, altı SHORT · ufuk **21** ·
feature-lag 1 · maliyet %0,1 + ödünç 100 bps/yıl. **Karar günü üyeliği** `sp500_uyelik` ile
süzülür; kıyas AYNI PIT kesitinin eşit-ağırlık ileri getirisidir. Veri: PIT `_1d_all` +
Ken French FF5 önbelleği → **18.194 satır / 545 sembol / 37 karar ayı (2023-07→2026-07)**.

### Koşulsuz iz — literatürün yönünde, ama gün düzeyinde anlamsız

| evren | uzun bacak | kısa bacak | fark | satır-t | **gün-t** | pozitif ay |
|---|---|---|---|---|---|---|
| **karar günü üyeleri (PIT)** | +0,231% | −0,231% | +0,46% | 3,66 | **1,63** | %56,8 |
| süzgeçsiz (panel evreni) | +0,343% | −0,344% | +0,69% | 5,20 | **2,23** | %64,9 |

(Maliyet öncesi, aylık, kesite göre fazlalık.) Satır-t gün-t'nin 2,2 katı: kesitsel olarak
korele satırlar bağımsız sayılınca t şişer (BULGU-53-1'in üçüncü tekrarı). Maliyet sonrası
taban (kuralın kendisi) +0,089%/işlem, kazanan %49,8.

### Kapılar

| kapı | sonuç |
|---|---|
| `veri_kapsami` | PASS (37/37 ay dolu; baş/son kesme 0 g) — ilk denemede **FAIL** (son kesme %5,5): kurucu etiket verisini `until`'de kesiyordu ve son iki ayın 21 barlık etiketi kurulamıyordu. Düzeltme etiket verisini pencerenin SONRASINA taşıdı (etiket tanımı gereği gelecektir); karar penceresi ve ön-kayıtlı parametreler değişmedi. HÜKÜM YOK koşusu deftere yazılmadı. |
| `purged_cv` | **FAIL** — OOS-AUC **0,517** (eşik >0,53) |
| `gate_time` | **FAIL** — kapı-E −0,0041 ↔ base +0,0038, Welch **t=−0,92**, seçicilik %1,9 (102/5.459) |
| `gate_symbol` | **FAIL** — t=0,39, seçicilik %0,9 (46/5.351) |
| `dsr` | **FAIL** — DSR **0,0006**, SR_fazlalık −0,0914, `n_trials=44` |
| `exec_lag_1` | **FAIL** |

**VERDİKT: KAPI(LAR) GEÇİLEMEDİ — kenar YOK.** Kuralın kendisi (medyan bölmesi) literatürün
yönünde küçük bir fark üretiyor ama 37 ayda ayırt edilemiyor; model özelliklerden ek seçicilik
kuramıyor (AUC 0,517). Sınır: 37 ay küçük bir örneklemdir ve dönem (2023-07→2026-07) tek bir
piyasa rejimidir; dürüst ifade *"bu evrende, bu dönemde ayırt edilemedi"*dir.

### BULGU-86-4 — panel evreni KARAR GÜNÜ üyeliği değildir

`research/edge_lab/panel.load_panel` (ve rebalance ailesinden, §27, beri PIT evreni kuran kurucular) evreni
**"pencerede herhangi bir an üye"** olarak kurar: 2021-01 üyeleri ∪ pencerede EKLENENLER. Karar
gününün üyeliği hiç süzülmedi → sonradan eklenen isim geçmiş kesite girer, ve endekse eklenme
çoğu zaman yükselişin SONUCUDUR. Yukarıdaki tablonun ikinci satırı bunun momentumdaki
bedelidir: gün-t 1,63 → **2,23** (süzgeçsiz aile sahte biçimde anlamlı görünürdü); dışarıda
kalan 2.298 satırın ileri getirisi **+3,21%/ay**. Aynı ölçüm mevcut sekiz veri setinde
(üyelik `ts`'in ayına göre):

| veri seti | satır | üye-olmayan pay | pnl üye ↔ üye-olmayan | t |
|---|---|---|---|---|
| vixts_v1 | 43.066 | %12,5 | +0,078% ↔ **+0,472%** | **+4,23** |
| insider_v1 | 1.815 | %15,9 | +0,849% ↔ +2,557% | +1,86 |
| vrp_v1 | 34.808 | %7,9 | +0,575% ↔ +1,077% | +1,56 |
| optskew_v1 | 43.140 | %8,5 | −0,094% ↔ −0,032% | +0,54 |
| seasonality_v1 | 22.386 | %11,6 | −0,108% ↔ −0,121% | −0,04 |
| rebalance_v1 | 728.978 | %5,6 | (fark yok) | −0,28 |
| shortflow_v1 | 147.318 | %12,1 | −0,120% ↔ −0,174% | −1,10 |
| babmacro_v1 | 32.417 | %12,3 | −0,154% ↔ **−0,633%** | **−2,18** |

**Okuma:** kirlenme yöne bağlıdır (kırılma/momentum benzeri vixts'te yukarı, düşük-beta
babmacro'da aşağı). Sekiz ailenin hepsi FAIL olduğu için sahte bir PASS YAYIMLANMADI; ama
hükümlerin süzgeçle değişip değişmeyeceği yeniden koşmadan bilinemez → **açık kalem** (eski
kurucular yayımlanmış sayılarla donmuştur, panel.py docstring'i). Yeni aile karar günü
üyeliğini `pit_universe.sp500_uyelik` ile süzmelidir (`build_rmom_dataset` örneği).

### BULGU-86-3 — PIT üyelik kurulumu 2026-08-11'den beri KIRIKTI

Wikipedia "List of S&P 500 companies" maddesindeki değişiklik tablosu 2026-08-11'de ayrı maddeye
taşındı (revizyon notu: *"move to [[Historical components of the S&P 500]]"*). `pit_universe`
tabloyu KONUMLA (`tabs[1]`) okuyordu; orası bir navbox oldu ve `sp500_as_of` o günden beri
`KeyError('date')` ile düşüyordu → PIT evren kuran her kurucu kırıktı, beş hafta kimse
koşmadığı için fark edilmedi. Artık tablo kolon İMZASIYLA aranır (önce ana madde, sonra yeni
madde); bulunamazsa ya da `date`/`added`/`removed` kolonlarından biri kurulamazsa `RuntimeError`.
Güncel kümeyle sessizce devam etmek survivorship'i geri getirirdi; ikinci dal (alt başlık
"Ticker" dışına dönerse geri-sarma HATASIZ boş kalıyordu) Gemini incelemesinden çıktı. **Doğrulama:** `sp500_as_of('2021-01-01')` = **506 sembol, 90'ı bugün yok** —
rebalance ailesinin (§27) yayımlanmış sayısıyla birebir.

### Tekrar üretmek için (§33)

```bash
python -m nihai.data.factors                                # FF5 önbelleği (anahtarsız)
python -m research.edge_lab.build_rmom_dataset --kapi
python -m research.edge_lab          # defter: 11 ayrık aile, sıradaki n_trials 48
```
