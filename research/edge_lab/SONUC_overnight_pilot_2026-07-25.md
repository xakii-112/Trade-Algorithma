# Sonuç — Overnight/Intraday Getiri Ayrışması (PİLOT), 2026-07-25

edge_lab programının İLK GERÇEK ADAY ölçümü (BACKLOG.md #1). `research.edge_lab.harness.
run_candidate` ile beş sert kapıdan geçirildi. **Sonuç: 5/5 kapı FAIL — kenar YOK.**

## Veri

- Kaynak: `nihai.data.alpaca_history.get_history` (SIP feed, ücretsiz plan), fallback yfinance
  (bu koşuda gerekmedi — 30/30 sembol Alpaca'dan geldi).
- Evren: `.env` `STREAM_SYMBOLS` — 30 sembol (AAPL, MSFT, SPY, NVDA, TSLA, QQQ, HPQ, LMT, AMD,
  IBM, TSM, MRVL, MU, AVGO, NOW, NBIS, ORCL, ONDS, SNDK, CSCO, AMZN, CVX, XOM, GE, DELL, GOOGL,
  JPM, V, QCOM, CRWD).
- **⚠️ SURVIVORSHIP-ŞİŞKİN PİLOT**: bu 30 sembol BUGÜN likit/işlem-gören isimler — hepsi
  hayatta kaldı, hiçbiri delist/removed değil (ACI DERS #1 doğrudan geçerli). Point-in-time
  (PIT) evren İNŞA EDİLMEDİ; bu, kenar bulunsaydı bile "artefakt olabilir" şüphesini otomatik
  taşırdı. Kenar YOK çıktığı için bu koşuda PIT'in kenarı GİZLEMİŞ olma ihtimali yok — ama
  tersi (survivorship'in olmayan bir kenarı YARATMASI) burada zaten gerçekleşmedi, çünkü
  sonuç zaten FAIL.
- Dönem: 2021-08-23 → 2026-07-23 (~5 yıl, günlük bar).
- Satır: 35.481 (hedef ≥15.000 — aşıldı). Sembol: 30/30 (hedef ≥20 — aşıldı).
- Kolon: `ts,symbol,f_overnight_mean5,f_overnight_mean20,f_intraday_mean5,f_intraday_mean20,
  f_vol20,f_volume_z,f_dow,label` — NaN yok (builder rolling-ısınma + t+1-yok satırlarını
  düşürüyor).
- Etiket dengesi: label=1 %49.8 / label=0 %50.2 (dengeli — sınıf dengesizliği AUC'yi
  şişirmiyor).
- Etiket kuralı: t+1 gecesi overnight getirisi (open_{t+1} − close_t)/close_t > 5 bps net ise 1.

## Beş kapının sonucu (harness.run_candidate, tam çıktı)

| # | Kapı | Sonuç | Detay |
|---|---|---|---|
| 1 | `purged_cv` | **FAIL** | Ortalama OOS-AUC **0.502** (5 fold, eşik>0.53) — tam anlamıyla yazı-tura (ACI DERS #2/#3'teki 0.51-0.52 aralığıyla aynı seviye) |
| 2 | `gate_time` | **FAIL** | Zaman-bölmesi test setinde (n=10.645) eşiği (≥0.55) geçen HİÇ işlem yok (0/10.645) — kapı simülasyonu değerlendirilemedi |
| 3 | `gate_symbol` | **FAIL** | Hiç-görülmemiş-sembol bölmesinde (n=10.460) yalnız 2 işlem eşiği geçti (<5 minimum) — istatistiksel olarak anlamsız |
| 4 | `dsr` | **FAIL** | Kapı-sonrası (time bölmesi) yeterli işlem yok → Deflated Sharpe hesaplanamadı |
| 5 | `exec_lag_1` | **FAIL** | 1-bar kaydırma sonrası da kapıyı geçen işlem yok (0/10.636) — zaten kapı-öncesi FAIL olduğundan artefakt sorusu bile gündeme gelmiyor |

**Verdikt metni (aynen)**:
```
KAPI(LAR) GEÇİLEMEDİ — kenar YOK / dayanıklı değil (CLAUDE.md ACI DERSLER)
```

## Dürüst yorum

Model, karar barına kadarki overnight/intraday getiri ortalamaları + oynaklık + hacim-z +
gün-of-hafta özelliklerinden **t+1 gecesi overnight getirisinin işaretini öngöremedi**
(OOS-AUC 0.502, tam şans seviyesi — ACI DERS #2'deki meta-labeling PIT ölçümüyle [0.51-0.52]
aynı sınıfta, hatta biraz daha zayıf). Bu, literatürdeki "bilinen overnight-drift anomalisi"
iddiasının BU evrende ve BU özellik setinde YOK olduğu anlamına gelir — anomalinin kendisi
muhtemelen (varsa) sembol-SEÇİMİNDEN bağımsız bir SEVİYE etkisidir (ör. "overnight ortalaması
her zaman pozitif" gibi sabit bir kayma), YÖN TAHMİNİ (bugünkü özelliklerden yarınki overnight
getiri işaretini öngörme) değil — bu iki şey ayrı sorular ve bu pilot yalnız ikincisini test
etti.

ACI DERS #2 ve #3'ün tam tekrarı burada da doğrulandı: AUC~0.50 kapı simülasyonunu asla
geçirmiyor (kapıyı geçen işlem sayısı 0-2 arası — istatistiksel gürültü). ACI DERS #5'in
söylediği gibi darboğaz muhtemelen özellik seti/veri, model mimarisi değil; ama bu pilotta
mimari zaten sorgulanacak noktaya (kapı sonrası örneklem) hiç ulaşmadı.

**Bu sonuç "overnight/intraday ailesinde asla kenar yoktur" demez** — yalnız "bu 30-sembol
survivorship-şişkin pilot evreninde, bu 7 leak-free özellik setiyle, bu 5-bps maliyet
eşiğiyle YOK" der. Farklı özellik seti (ör. VIX-koşullu overnight, sektör-göreli overnight,
opening-gap büyüklüğü [yön değil] gibi SEVİYE-tabanlı testler), farklı maliyet eşiği veya PIT
evren farklı sonuç verebilir — ama bu iddia burada TEST EDİLMEDİ, yalnız olası bir sonraki adım
olarak not ediliyor.

## Sonraki adım

1. **PIT evren önceliği düşük**: kenar zaten bu (daha kolay geçmesi beklenen) survivorship-
   şişkin evrende YOK çıktığı için, PIT evrende tekrar test etmek muhtemelen aynı sonucu
   (ya da daha kötüsünü) verir — kaynak BACKLOG.md'deki #2 (VRP) veya #3 (lead-lag) gibi
   diğer adaylara aktarılabilir.
2. Eğer overnight/intraday ailesi tekrar denenecekse: YÖN tahmini yerine SEVİYE testi (ör.
   "overnight getiri ortalaması sıfırdan anlamlı farklı mı" — DSR/PBO ile, ML olmadan) veya
   rejim-koşullu (VIX/gün-of-ay) alt-küme testi denenebilir; bu farklı bir aday tasarımı olur.
3. `research.edge_lab.build_overnight_dataset` script'i BAŞKA aday sinyaller için de şablon
   olarak kullanılabilir (leak-free rolling özellik + t+1 etiket deseni; `--symbols`/`--years`
   ile genelleştirilebilir).

**Tekrar canlı gölge doğrulaması ÖNERİLMEZ** — kapılardan hiçbiri geçmedi, `PASS_VERDICT`
yolu tetiklenmedi.
