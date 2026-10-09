# Ön-kayıt: kripto ileri yarışı — `ma10_50` ↔ `event_breakout` (2026-09-25)

**Bu belge kilitlidir.** sha256'sı `research/ileri_kripto_yaris_kilit.json`'da; tek bayt değişirse
`python -m research.ileri_kripto_yaris guncelle` deftere yazmayı durdurur. Değişiklik gerekirse bu dosya
DÜZENLENMEZ, tarihli bir ek belge (`ONKAYIT_KRIPTO_YARIS_EK<N>_<tarih>.md`) yazılır.

**Bağlam.** Tur-96 §9 madde 1 (kripto event_breakout ön-kayıtlı ileri sınama) + 2026-09-25 kafa kafaya ölçüm:
aynı 8 coin, aynı yfinance verisi, 2018-01-02 → 2026-09-24, %0,1/taraf, eşit ağırlıklı (EW-8) portföy:

| strateji | proje | Sharpe (tam) | alfa t (tam) | Sharpe 2022-26 | alfa t 2022-26 |
|---|---|---|---|---|---|
| al-tut | · | 0,81 | · | 0,37 | · |
| `ma10_50` | Quant AI (`alphalab.py`) | 1,21 | 2,70 | 0,64 | 1,19 |
| `ma10_50` +1 gün gecikme | Quant AI | 1,14 | 2,38 | 0,64 | 1,18 |
| `event_breakout` | Nihai (Tur-96) | 1,00 | 2,90 | 0,48 | 1,01 |

Geçmişte fark `ma10_50 − event_breakout` = +0,22 Sharpe, %90 aralık [−0,45, +0,91], P(≤0) = 0,29 → ayırt
edilemedi. İki kural da kendi projesinin en iyisi olarak SEÇİLDİ (29 ve 10 aday arasından) ve evren bugünün
büyük coin'leri: geçmiş rakamlar iyimser yönde yanlıdır. İleri dönemde bu iki yanlılık yoktur; sınamanın amacı budur.

## §1 Dondurulan tanımlar

- **Evren (sabit, 8):** BTC-USD, ETH-USD, XRP-USD, LTC-USD, BNB-USD, ADA-USD, DOGE-USD, SOL-USD.
- **Veri:** yfinance günlük bar (UTC günü), `research.piyasalar.indir` (auto_adjust), sınamaya özel önbellek
  `data/ileri_kripto_yaris/veri/` (Tur-96 önbelleği kullanılmaz). Yalnız TAMAMLANMIŞ barlar: tarih < bugün (UTC).
- **`ma10_50`:** kapanışın 10 günlük basit ortalaması 50 günlüğünden büyükse uzun (1), değilse nakit (0);
  t kapanışındaki sinyal t → t+1 kapanış getirisini kazanır (Quant AI `alphalab.py`: `net_t = pos_{t−1}·r_t − cost·|Δpos|`).
- **`ma10_50_g1` (ikincil):** aynı kural, bir gün daha geç uygulanır (t sinyali t+1 → t+2).
- **`event_breakout`:** Nihai motoru, `research.coklu_piyasa._algoritma_kos(sym, df, t0, "event_breakout")`
  birebir Tur-96 ayarıyla: meta kapı KAPALI, kill switch kapalı, VIX sınamanın önbelleğinden, `df = hazirla(sym)`
  (2008 sonrası, ilk kapanış = 100 ölçeği), `t0 = max(200, 2018-01-01'in indeksi)`. Getiri = motorun özsermaye
  eğrisinin günlük değişimi (kriptoda yfinance açılışı önceki kapanışa medyan 1 bp yakın → yürütme `ma10_50` ile eşdeğer).
- **Al-tut (kıyas):** kapanış → kapanış, maliyetsiz.
- **Maliyet:** %0,1/taraf (`ma10_50`: pozisyon değişimi başına; `event_breakout`: motorun kayması).
- **Portföy:** EW-8 = 8 coinin günlük getirilerinin ortalaması (günlük yeniden dengeleme maliyeti yok sayılır).
- **İleri dönem:** 2026-09-26 ve sonrasında gerçekleşen getiriler (ilk karar 2026-09-25 kapanışı; o gün kilit
  oluşturulurken henüz tamamlanmamıştı). Önceki hiçbir gün ileri sonuca girmez.

## §2 Hipotezler ve karar kuralları

**Birincil karar günü: 2029-09-25 (tek bakış).** Öncesindeki tüm raporlar yalnız betimseldir; erken durdurma,
erken ilan ya da karar gününü değiştirme YOKTUR.

- **H1 (yarış):** EW-8 ileri getirilerinde Sharpe(`ma10_50`) − Sharpe(`event_breakout`), eşleştirilmiş dairesel blok
  bootstrap (blok 20 gün, 2000 örnek, tohum 0). P(fark ≤ 0) < 0,05 → **`ma10_50` kazandı**; > 0,95 → **`event_breakout`
  kazandı**; arada → **ayırt edilemedi**.
- **H2 (her kural ayrı, al-tuta karşı):** ileri EW-8 getirisi ~ al-tut EW-8 regresyonunun Newey-West alfa t'si:
  t > 2 → **kenar teyit**; 0 < t ≤ 2 → **tutarlı, kanıtlanmadı**; t ≤ 0 → **ileri dönemde çöktü**.
- **Betimsel (karar değil):** `ma10_50_g1`, yıllık getiri, en büyük düşüş, nokta tahmini olarak "önde olan".

## §3 Güç analizi (sonuç görülmeden yazıldı)

Sharpe farkının standart hatası ≈ √(2(1−ρ)/T), ρ = iki kuralın günlük korelasyonu (geçmişte 0,22; 2022-26'da 0,25).
Geçmiş blok bootstrap bununla uyumlu (8,7 yılda SE ≈ 0,41).

| | 1 yıl | 3 yıl | 5 yıl |
|---|---|---|---|
| Sharpe farkının SE'si | 1,25 | 0,72 | 0,56 |
| %80 güçle saptanabilir en küçük fark (tek yönlü %5) | 3,1 | 1,8 | 1,4 |
| `ma10_50` beklenen alfa t (IR 0,91 tam / 0,55 2022-26) | 0,9 / 0,6 | 1,6 / 1,0 | 2,0 / 1,2 |
| `event_breakout` beklenen alfa t (IR 0,98 / 0,46) | 1,0 / 0,5 | 1,7 / 0,8 | 2,2 / 1,0 |

**Dürüst beklenti:** gerçek fark geçmişteki gibi ~0,2 Sharpe ise H1'in 3 yılda "ayırt edilemedi" çıkması neredeyse
kesindir (%80 güç için ~200 yıl gerekir). H1 ancak kurallardan biri ileri dönemde belirgin biçimde çökerse karar
verir. Sınamanın asıl bilgisi H2'dir: gerçek IR 2022-26 düzeyindeyse (≈0,5) 3 yılda alfa t > 2 olasılığı ≈ %12-15,
t ≤ 0 olasılığı ≈ %19; gerçek IR 0 ise t ≤ 0 olasılığı %50. Yani sınama "kenarı kanıtlamaktan" çok "kenarın
çöktüğünü yakalamaya" uygundur.

## §4 Bütünlük (kilit ve defter)

- `kilitle` (yalnız 2026-09-25 UTC içinde, bir kez): bu belgenin sha256'sı + 8 coin × 4 stratejinin
  2018-01-01 → 2026-09-24 getirilerinin parmak izi (tarih + 10 haneli getiri sha256'sı, yıllık toplamlar).
- `guncelle`: veriyi yeniden indirir, TÜM geçmişi yeniden hesaplar ve (a) belge sha256'sını, (b) geçmiş parmak izini,
  (c) deftere daha önce yazılmış her satırı karşılaştırır. Biri tutmazsa deftere yazmaz, farkı (coin/strateji/yıl) basar.
  Kilit davranışsaldır: motor kodu değişse de event_breakout'un geçmiş getirileri aynıysa sınama sürer.
- Defter (`data/ileri_kripto_yaris/defter.csv`) yalnız sona eklenir; yalnız 8 coinin hepsi için tamamlanmış günler yazılır.
- Kilit bozulursa: sınama DURUR; neden bulunur. Kod değişikliğiyse kod geri alınır (ya da motorun o sürümü sınama
  için ayrı tutulur); veri revizyonuysa ek belgede gerekçesiyle kayda geçer. Kilit sessizce yeniden oluşturulmaz.
- Coinlerden birinin verisi kesilirse defter bekler; 14 gün içinde dönmezse ek belgeyle karar verilir.

## §5 Yasaklar

Parametre (10/50, maliyet, blok, eşikler), evren, karar günü ya da karar kuralı değişikliği; H1'e yeni strateji
eklemek; ara sonuçlara bakıp kural "iyileştirmek"; kilidi yeniden oluşturmak. Yeni fikirler ayrı bir ön-kayıtla sınanır.

## §6 Çalıştırma

`guncelle` deterministiktir; kaçırılan günler sonraki koşuda aynı sonuçla doldurulur, bu yüzden haftalık koşu yeter:

```
python -m research.ileri_kripto_yaris guncelle
```
