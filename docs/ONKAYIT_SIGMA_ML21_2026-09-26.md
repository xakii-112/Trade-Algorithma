# ÖN-KAYIT — σ_ML21 ileri sınaması (2026-09-26)

Bu belge kilitlendikten sonra DÜZENLENMEZ (sha256'sı `research/ileri_sigma_ml21_kilit.json`'da). Değişiklik
gerekirse `ONKAYIT_SIGMA_ML21_EK<N>_<tarih>.md` yazılır; kural, evren, eşik ve karar günleri değişmez.
Kod: `research/ileri_sigma_ml21.py` · Test: `tests/test_ileri_sigma_ml21.py`.

## 1. Soru

Tur-98/99: ML oynaklık modeli σ_ML21 (21 günlük ortalama günlük varyans öngörüsü) EWMA(0,94)'ü öngörüde iki
pencerede de yendi (canlı evren DM t −7,45 / −8,06), ama `quant_trend` motorundaki Sharpe katkısı eşiğin
altında kaldı (Tur-99 teşhis, sonradan: birleşik z 1,33 / 2,41; tahmini etki +0,04 / +0,06 Sharpe). İki soru
yalnız KİLİT GÜNÜNDEN SONRA gerçekleşen veriyle sorulur:

- **H1 (öngörü):** σ_ML21'in 21 günlük öngörü üstünlüğü geleceğe taşınıyor mu?
- **H2 (karar):** σ_ML21'i canlı mekanizmada (σ yalnız GİRİŞTE: stop 3σ, adet risk $/stop) kullanmak EWMA'ya
  göre Sharpe'ı artırıyor mu?

## 2. Kollar (tek değişken: σ kaynağı)

- `ewma` — canlı `quant_trend` (σ = EWMA 0,94).
- `ml21` — aynı motor; σ = √(σ_ML21 öngörüsü) (öngörü sonlu ve pozitif değilse o gün EWMA; Tur-98 yama kuralı).
- Motor: `backtest_symbol` 1d, `quant_trend`, kill-switch ve AI kapısı kapalı, maliyet `stock_slippage_pct`
  (%0,1/taraf), `.env` risk ayarları kilit günündeki gibi. Motor 2026-01-01'den başlar (ısınma geçmişi 2008'den);
  iki kol aynı başlangıçla koşar, yalnız ileri günlerin getirisi sayılır.

## 3. Model tarifi (Tur-98 H1, dondurulmuş)

- Özellikler (`quant_ml.B_OZ`): ln EWMA v_t, ln ort. r² 5/22/63 gün, ln r_t², 1{r_t<0}, ln (ln H/L)²_t,
  ln ort((ln H/L)²) 5 gün. Hedef ln RV21 = ln ort(r²_{t+1..t+21}).
- `HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
  l2_regularization=1.0, random_state=42)`; yanlılık düzeltmesi c = ort(RV21)/ort(exp ŷ) EĞİTİM kümesinde.
- Eğitim yalnız 14 piyasa havuzunda (ABD_izleme satırları eğitime girmez).
- Yıllık model: Y yılının tarihleri için model Y, etiketi Y başından ÖNCE bilinen satırlarla
  (`quant_ml.egitim_maskesi(…, Y, 22)`) eğitilir. 2026 modeli kilitte; Y ≥ 2027 modeli Y yılındaki ilk
  `guncelle` koşusunda BİR KEZ eğitilir, dosyası ve sha256'sı dondurulur, sonra hiç yeniden eğitilmez.

## 4. Evren ve veri

- Evren kilit günü dondurulur: Tur-96'nın 14 piyasa grubu (kusur kuralından geçenler) + ABD_izleme (`.env`
  STREAM_SYMBOLS). Liste kilitte; sonradan `.env` değişse de evren değişmez.
- Veri: yfinance günlük (period=max, auto_adjust), sınamaya ÖZEL önbellek (Tur-96/97 önbelleği değişmez,
  BULGU-98-6). `coklu_piyasa.hazirla` (bayat bar süzgeci, tek-bar aralık onarımı, ilk kapanış = 100).
- İleri dönem: tarih > **2026-09-27**. Defterlere (getiri + öngörü) yalnız tamamlanmış günler (tarih < bugün
  UTC) sona eklenir; yazılmış satır asla yeniden yazılmaz.

## 5. H1 — öngörü (karar günü 2027-09-30)

- Örneklem: tarihi (2026-09-27, 2027-08-31] olan öngörüler; gerçekleşen RV21 ile QLIKE = ln f + RV21/f.
- Piyasa: günlük kesit ortalama kaybı; fark d = QLIKE(ml21) − QLIKE(ewma).
- **KAPI:** ort(d) < 0 olan piyasa ≥ 10 (15 piyasada) **VE** ABD_izleme DM t'si (Newey-West, gecikme ≥ 21) < −1,645.
- Güç (koşudan önce, Tur-98 izleme t'sinden √gün ölçeklemesi): ~231 çözülmüş günde beklenen t ≈ −2,66 → %85.

## 6. H2 — karar (karar günü 2031-09-30)

- Örneklem: (2026-09-27, 2031-09-30] günlük getiriler; piyasa = enstrümanların eşit ağırlıklı ortalaması.
- Piyasa başına yıllık ΔSharpe (ml21 − ewma) ve Memmel (2003) z'si. BİRLEŞİK z = Σ z_k / √(Σ_jk R_jk); R, ileri
  dönemin günlük fark serilerinin korelasyonu (ortak günü 60'tan az olan çiftte R = 0).
- **KAPI:** birleşik z > 1,645 **VE** ΔSharpe > 0 olan piyasa ≥ 10 **VE** ABD_izleme ΔSharpe > 0 (Tur-99 kapısı).
- Güç (koşudan önce; Tur-99 MDE'si √yıl ile ölçeklendi): gerçek etki +0,05 Sharpe ise 5 yılda ≈ %43
  (etki +0,04 → %30, +0,06 → %57). %80 güç ≈ 14 yıl ister. Yani H2'nin en olası sonucu "ayırt edilemedi"dir;
  bu belge bunu ÖNCEDEN söyler. Etki gerçek olsa bile küçüktür (+0,05 Sharpe).

## 7. Hüküm

| H1 | H2 | Sonuç |
|---|---|---|
| geçmedi | — | σ_ML21 adaylıktan DÜŞER (öngörü üstünlüğü geleceğe taşınmadı); H2 yalnız bilgi olarak raporlanır |
| geçti | geçmedi | öngörü sürüyor, karar değeri kanıtlanmadı; canlıya bağlanmaz |
| geçti | geçti | σ_ML21 canlı quant çekirdeğine ADAYDIR (entegrasyon ayrı tur, ayrı test) |

Karar günlerinden önceki her rapor ARA rapordur: betimseldir, hüküm yoktur, kapı ölçütü değişmez.

## 8. Kilit ve bozulma

- Kilit: bu belgenin sha256'sı; dondurulmuş evren; 2026 modelinin sha256'sı; KANARYA — her piyasanın ilk
  sembolünün kilit günü verisinin kopyası ve o veriyle iki kolun 2026-01-01 → 2026-09-27 getirilerinin ve σ_ML21
  serisinin parmak izi.
- Her `guncelle`: belge sha'sı, model sha'ları ve kanarya yeniden hesaplanır. Biri tutmazsa (kod, ayar, kütüphane
  ya da belge değişti) deftere YAZILMAZ ve fark basılır.
- Taze veride deftere yazılmış satırın yeniden hesabı farklıysa (yfinance revizyonu) satır değişmez; uyuşmayan
  satır sayısı raporlanır.

## 9. Bilinen sınırlar

Bugünkü sembol listesi (survivorship); yfinance revizyonları; kısa ileri dönemde korelasyon matrisi gürültülüdür;
güç H2 için düşüktür (§6).
