# Tur-17 — Kanıt Döngüsü Devreye Alma Raporu (2026-07-25)

**Görev:** `docs/AI_V3_DERIN_BEYIN_PLAN_2026-07-19.md` uygulamasının Tur-16 sonrası kalan
kalemleri (bkz. `AI_V3_DERIN_BEYIN_RAPORU_2026-07-25.md` §5 "bilinçli sınırlar" + §6 "sıradaki").
Anayasa değişmedi: **KARAR-DIŞI/GÖLGE** — sinyal/eşik/META_GATE/execution'a sıfır bağlantı.

**Sonuç:** Tur-17'nin beş kaleminden **dördü kapandı**, biri (predictions_log `neural_pwin`
kolonları) gerekçesiyle açık bırakıldı. Yol boyunca **bir kanıt-kapısı kusuru bulundu ve
düzeltildi** (aşağıda §4 — bu turun en önemli çıktısı).

---

## 1) Kapanan kalemler

| # | Kalem | Durum | Nerede |
|---|---|---|---|
| 1 | Overnight/intraday GERÇEK edge_lab koşusu | **TAMAM — 5/5 kapı FAIL, kenar YOK** | `research/edge_lab/build_overnight_dataset.py` (+14 test), `SONUC_overnight_pilot_2026-07-25.md`, BACKLOG #1 → KOSULDU-FAIL |
| 2 | Portföy-bakışı köprüsü (brain/shadow döngüsüne) | **TAMAM** | `shadow_runner._shadow_portfolio_view` (+9 test) |
| 3 | Lab köprüsü (öz-eleştiriden deney önerisi) | **TAMAM** | `brain._self_critique` (+3 test) |
| 4 | Karne satırı → günlük brif **ve** `daily_summary --brain` | **TAMAM** | `selfcheck.last_card_line` (+5 test), `brain.daily_brief`, `daily_summary._brain_block` (+7 test) |
| 5 | registry'ye ilk model kartı (`train_neural --save`) | **TAMAM (2 makine-üretimi kart)** | `train_neural --register/--registry-dir` + `build_model_card` (+8 test) |

### 1.1 Overnight pilot — ilk gerçek edge_lab adayı
30 sembol × ~5 yıl (35.481 satır, Alpaca SIP), leak-free 7 özellik, etiket = t+1 gecesi
overnight getirisi > 5 bps. **Beş kapının beşi FAIL**: purged-CV OOS-AUC **0.502**, gate_time
0/10.645 işlem eşiği geçti, gate_symbol 2/10.460, DSR hesaplanamadı, exec-lag-1 de 0/10.636.
Evren survivorship-şişkin (bugünkü likit isimler) — kenar zaten bu *kolay* evrende çıkmadığı
için PIT tekrarının önceliği düşük. Ayrıntı: `research/edge_lab/SONUC_overnight_pilot_2026-07-25.md`.

### 1.2 Portföy köprüsü — hizalama düzeltmesiyle
Gölge kipte tur sonunda karantinasız sembollerin kapanış paneliyle `portfolio.propose`
(HRP/BL/CVaR/ERC/min-var/max-Sharpe/Kelly yan yana) çağrılır; yazdığı tek şey kendi GÖLGE
öneri-JSONL'idir. **Düzeltme:** panel önce *pozisyonel kuyrukla* hizalanıyordu — bir sembolde
eksik/fazla bar (tatil, halt, provider farkı) varsa farklı günler aynı satıra düşüp kovaryansı
sessizce bozardı. Artık `pd.Series(closes, index=df.index)` + `concat(join="inner")` ile
**zaman damgasıyla** hizalanır; yan fayda: ortak indeks `DatetimeIndex` kaldığı için
`portfolio._panel_dates` tarihsel stres penceresini artık hesaplayabiliyor (önce sessizce
atlanıyordu). Kapı şartları: yalnız `shadow` kipte, ≥3 sembol, ≥60 **ortak** bar; her hata → None.

### 1.3 Lab köprüsü — 5-madde tavanı kusuru
`_self_critique` deney önerisini listenin sonuna ekliyordu, ama `weak[:5]` tavanı önerinin
**tetiklendiği** durumda (özellik ve/veya rejim zemini yok) üstteki beş tanı tarafından zaten
doluyordu → köprü pratikte **hiç görünmüyordu**. Artık tavan yalnız tanılara uygulanır
(`weak[:4] + [öneri]`). Öneri METİNDİR: `brain` hiçbir koşulda `lab.run_experiment` çağırmaz
(test bunu doğruluyor — otomatik yürütme yapısal olarak yok).

### 1.4 Karne satırı — sahiplik selfcheck'e taşındı
Ayrıştırma `brain`'de özel bir fonksiyondaydı; dosyanın ve şemanın sahibi `selfcheck` olduğu
için `selfcheck.last_card_line()` olarak oraya taşındı (mevcut `_read_all` bozuk-satır
toleransı yeniden kullanıldı). İki raporlayıcı da onu çağırıyor. **SALT-OKUNUR sözleşmesi
testli:** ne yeni sınav koşar ne JSONL'e satır ekler (üç sınav pahalıdır ve günlük özet
her koşuşunda karne geçmişini şişirirdi).

---

## 2) Model kartları (registry)

`train_neural --save --register [--registry-dir]` eklendi. `--register` **tek başına yazmaz**
(artefaktı olmayan kart yanıltıcıdır). Kart içeriğinde zaman damgası bilerek yok: registry
hash'i içerikten türetir, ts eklemek aynı eğitimi her seferinde farklı hash'e sokardı.

| Kart | Veri | OOS-AUC (purged 5-fold) | DSR (düzeltilmiş) | Okuma |
|---|---|---|---|---|
| `4e74a336…` | sentetik müfredat, 4.000 satır (taban %49,8) | **0.533 ± 0.037** | 0.9987 | **Kontrol koşusu:** yapısı gereği öğrenilebilir sinyal ekilmiş veride hat çalışıyor |
| `1f072998…` | PIT S&P500 (alpaca_hist), 12.268 satır (taban %35,7) | **0.500 ± 0.019** | **0.0** | **Gerçek veride hiçbir şey yok** |

İkisi yan yana okununca sonuç ACI DERS #5'in bağımsız teyididir: **hat sağlam, veri boş.**
186.702 parametreli ağ, purged CV'de tam şans seviyesinde. Mimari eklemek bunu çözmüyor.

> Not: `data/ai_models/ac505353…/card.json` (25.07 09:09, elle yazılmış) aynı modelin
> ölçümünü taşıyor ve verdikti dürüst ("KENAR YOK"), ancak `dsr: 1.0 / psr: 1.0` alanları
> §4'teki **düzeltme öncesi** metrikten geliyor — o alanlar geçersizdir, `1f072998…` kartı
> onun yerine okunmalıdır. Kart silinmedi (registry append-only).

---

## 3) `neural_pwin` kolonları — bilinçli olarak AÇIK bırakıldı

Rapor §6'daki son kalem. Ön koşulu (şema-uyumlu, kayıtlı bir neural model) bu turda karşılandı
(`1f072998…`, `FEATURE_NAMES` ile birebir 25 özellik). **Yapılmadı, çünkü** kalan iş üretim
canlı döngüsüne dokunuyor: `prediction_tracker.FIELDS` şema genişletmesi + `live_predict_service`
içinde her aday için torch çıkarımı (gecikme, model yükleme, fail-soft) + eski-şema CSV göçü.
Bu, gölge de olsa sistemin en riskli yüzeyidir ve bu oturumda hak ettiği testle yapılamazdı.

Yapıldığında hazır olan zemin: `learning.py` zaten `neural_pwin` + `neural_pwin_side`
kolonlarını arıyor ve varsa `sources.neural` Brier'ini **otomatik** dolduruyor (kablolama
gerekmez). Sıradaki turun ilk maddesi budur.

---

## 4) BULGU — DSR kanıt kapısı sınıf dengesizliğini beceri sanıyordu

Bu turun en önemli çıktısı, planlanmış bir kalem değil; PIT koşusunun **kendi içinde çelişmesi**
sayesinde yakalandı: `OOS-AUC 0.500` (sıfır ayırt etme) ile `DSR 1.0 / PSR 1.0` aynı raporda.

**Kök neden:** `train_neural.run_cv` sözde-getiriyi ham isabetten üretiyordu (`2·hit − 1`).
Taban oranı %50'den uzak veri setlerinde "hep çoğunluk sınıfını söyle" gibi **hiçbir bilgi
içermeyen** bir kural bile pozitif ortalama üretir; PIT setinde (taban %35,7) bu Sharpe 0,29'a,
n=12.268'de ise PSR/DSR'yi 1,0'a taşıyordu. DSR bu projede bir **kanıt kapısıdır** — yani kapı,
elemesi gereken şeyi onaylıyordu.

**Düzeltme:** sözde-getiri artık *çoğunluk-kuralına göre fazlalık* (`excess_pseudo_returns`):
modelin isabeti − triviyal kuralın isabeti. Çoğunluk sınıfı **yalnız eğitim fold'undan**
hesaplanır (sızıntı yok). Triviyal kuralla birebir aynı davranan model sıfır varyanslı seri
üretir → rapor dürüstçe "ölçülemedi" der (sıfır bölme NaN'ı deflated.py'ye taşınmaz).

**Etki (aynı model, aynı veri, yalnız metrik düzeldi):**

| | Sharpe | PSR | DSR |
|---|---|---|---|
| Düzeltme öncesi | +0.2908 | 1.0 | **1.0** |
| Düzeltme sonrası | −0.0308 | 0.0002 | **0.0** |

Sentetik koşuda DSR yüksek kaldı (0.9987) — orada taban ~%50 olduğu için dengesizlik artefaktı
zaten yoktu ve veride *gerçekten* ekili yapı var; yani düzeltme sinyali körleştirmiyor, yalnız
sahte olanı söndürüyor. Regresyon testleri: dengesiz veride sabit tahminci → seri tam sıfır ve
DSR `None`; kusursuz tahminci → pozitif ortalama; çoğunluk eğitimden hesaplanıyor.

**edge_lab'daki DSR kapısı bu kusurdan etkilenmiyor** (farklı kurgu: kapı-sonrası işlem
pnl'i üstünde çalışıyor ve gate_time/gate_symbol kapıları zaten BASE'i geçmeyi şart koşuyor),
ama aynı ailenin riski olduğu için not düşülmüştür.

---

## 5) Doğrulama

- **pytest: 1678 yeşil** (bu oturumun başı 1645 → **+33 test**), ruff temiz,
  mypy(nihai) 83 dosya temiz.
- Dosya bazında (toplam): `test_ai_shadow_runner` 25 (+9) · `test_ai_brain` 23 (+10:
  7 karne/lab + 3 daily_summary köprüsü) · `test_ai_selfcheck` 23 (+5) ·
  `test_train_neural` 31 (+9: 4 DSR fazlalık + 3 model kartı + 2 CLI register) ·
  `test_edge_overnight_dataset` 14 (oturum başında yazılmıştı, 1645 tabanına dahil).
- Gerçek koşular: iki eğitim + kart kaydı (yukarıda), overnight pilot (5/5 FAIL).

## 6) Sıradaki (kanıt-kapılı, öncelik sırasıyla)

1. **`neural_pwin` kolonları** (§3) — şema + live_predict_service gölge çıkarımı + göç testi.
2. `tur17-shadow-gorev.ps1` **kaydı hâlâ kullanıcıda** (Yönetici PowerShell). Kayıt olmadan
   gölge birikim saatlik değil, yalnız elle koşulduğunda oluyor.
3. Web panosunda AI/Beyin bölümü **yok** — karne satırı `daily_summary`'ye bağlandı, web için
   önce bir uç nokta + sekme gerekir (ayrı iş; bu turda kapsam dışı bırakıldı).
4. edge_lab BACKLOG #2 (VRP) veya #3 (lead-lag) — overnight ailesi FAIL kapandı.
