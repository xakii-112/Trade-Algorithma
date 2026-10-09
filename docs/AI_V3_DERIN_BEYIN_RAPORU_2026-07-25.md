# AI v3 "Derin Quant Beyin" — Teslimat Raporu (Tur-16, 2026-07-25)

**Görev:** `docs/AI_V3_DERIN_BEYIN_PLAN_2026-07-19.md` (rev.3) uygulaması — 13 iş paketi,
×1000 parametre sözleşmesi, KARAR-DIŞI/GÖLGE anayasası korunarak.

**Sonuç:** İKİ DALGADA TAMAM. **1615 test yeşil** (Tur-15 tabanı 938 → +677), ruff temiz,
mypy(nihai) 82 dosya temiz, **torch'suz koşu 1548 yeşil + 67 skip / 0 kırmızı** (fail-soft
sözleşmesi kanıtlı). Dalga-1 commit `34f3f52`; Dalga-2 + köprüler bu commit.

---

## 1) Dalga-1 (çekirdek, `34f3f52` — önceki oturum)

`neural.py` (**QuantTransformer 5.438.739 param** — ≥3.777.000 ×1000 sözleşmesi TESTLİ;
DeepMLP; MC-dropout; çoklu-ölçek h_/d_/w_ önekleri), `quant_math` +16 fonksiyon (GARCH-MLE,
EVT-GPD, kopula, FFD, Welch/Haar, Almgren-Chriss[rapor-only], entropiler, MI, coskew/cokurt),
`knowledge` 103 kavram + curriculum DAG, `decision` v3 (Brier-ağırlıklı hiyerarşik birleşim +
fayda kapısı; v2-parite testli), `episodic` v3 + `consolidate.py`, `learning` v3 (kaynak-Brier +
yalnız-OOS kalibratör + retrain bayrağı), `research/train_neural.py` (purged CV + DSR),
`models.py` 6-katman merdiven (opt-in; v2 varsayılanlar bit-denk), `brain` v3 döngüsü.

## 2) Dalga-2 (bu commit) — 9 yeni modül + köprüler

| Modül | WP | Teslim |
|---|---|---|
| `advanced_math.py` | M2 | 21 fonksiyon / 8 grup: Merton MLE + bipower sıçrama, Heston charfn + Carr-Madan FFT + kalibrasyon, Amerikan opsiyon (Brennan-Schwartz CN + LSMC), Hawkes MLE/yoğunluk, BOCPD, 2-rejim Kalman, parçacık filtresi, rough-vol/Hurst, seviye-2/3 yol imzaları, matrix profile, qp_solve/quad/logsumexp/fft_convolve. Analitik-limit çapaları (Heston→BS %1, Amerikan≥Avrupa, LSMC≈CN) + sentetik geri-bulma. **101 test** |
| `portfolio.py` | P | GÖLGE öneri: min-var, max-Sharpe, **HRP**, ERC, **Black-Litterman** (decision posterior=görüş), **CVaR-LP**, çok-varlık Kelly — yedi yöntem YAN YANA + kondisyon sayısı + MC/tarihsel stres + atribüsyon; kısıtlar (max_weight/short-yasak/net-cap/ciro cezası); her öneri atomik gölge-JSONL. **47 test** |
| `data_health.py` | O | 6 denetim (NaN/OHLC/eksik-bar/donmuş/sıçrama-vs-split/hacim) → skor + KARANTİNA; çok-kategori bileşik ceza (orkestratör düzeltmesi: ≥3 cezalı kategori → +10/kategori). **26 test** |
| `explain.py` | X | Permütasyon önemi, integrated gradients (tamlık aksiyomu testli), attention rollout, **split-conformal** kalibrasyon+aralık (kapsam simülasyonu testli), karar açıklama kartı, küresel model raporu. **30 test** |
| `text_intel.py` | T | Loughran-McDonald ton (poz/neg/belirsizlik/dava), başlık-yoğunluğu z, yenilik (kosinüs), LLM özet (fail-soft, deterministik skorla YAN YANA), makro/kazanç/OPEX takvim yakınlığı. **35 test** |
| `registry.py` | O | sha256 model kartı + sürümlü saklama, `latest/list/get_card` savunmacı, gölge A/B JSONL + `ab_report` (Wilson CI; örtüşme → FARKSIZ). **25 test** |
| `lab.py` | C | Sandbox `run_sandbox` (beyaz-liste import/builtins, süre limiti, stdout yakalama, atomik log) + 5 deterministik deney şablonu + `ask_llm_analysis` fail-soft; her çıktıda "tanımlayıcı analiz; kenar iddiası değildir". **34 test** |
| `agent.py` | A | `ask()` kaynaklı Türkçe cevap (knowledge+lessons+docs taraması; LLM akıcılaştırma dürüstlük-guard'lı) + hipotez kuyruğu (yalnız-analiz görev tipleri, derinlik 12 / gün 50 tavanı, `research.*` import ETMEZ — otomatik yürütme yapısal olarak yok). **28 test** |
| `selfcheck.py` | A | Karne: bilgi sınavı (kavram/DAG sayısal doğrulama) + kalibrasyon (learning state) + sentetik rejim-replay; geçmiş karne JSONL + trend. **18 test** |
| `research/edge_lab/` | E | `run_candidate`: purged CV → time+symbol çift kapı → DSR → **exec-lag-1** → verdikt; verdikt "kenar var" İÇEREMEZ (kodda+testte). `BACKLOG.md` aday sıralaması (ilk aday: overnight/intraday). **14 test** |

**Köprüler (orkestratör):** brain ALGILA başına **veri-sağlık kapısı** (KARANTİNA → karar
motoru çağrılmadan zorunlu ABSTAIN, gerekçesi şeffaf) · **conformal kapı-c** kablolaması
(`think(conformal_calib=)` → decision'ın hazır kapısına) · `brain --ask` CLI · anlatıya
veri-sağlığı/conformal satırları · CLI UTF-8 (cp1254 emoji çökmesi düzeltildi) ·
advanced_math-yokluğu varsayan 2 eski test monkeypatch-simülasyona çevrildi.

## 3) Dürüstlük koşuları (Faz 3 — gerçek çıktılar)

- `brain --demo` (yazmasız): veri-sağlık skor=100 geçti → hiyerarşik birleşim
  posterior 0.506 → kapı-a kenar 0.0105 → **ABSTAIN** (kaynak ağırlıkları satır satır).
- `selfcheck.report_card()`: bilgi 100/100 · kalibrasyon **isabet 0.458 (CI 0.444–0.472),
  pwin Brier 0.262, trend "BOZULUYOR (drift)"** — bilinen NO_EDGE tablosu bağımsız teyit ·
  replay: 3/3 rejim doğru, ABSTAIN oranı 1.0 (disiplin).
- Portföy demosu (6 sentetik varlık): 7 yöntem yan yana ağırlık + kondisyon 3.53 + MC stres;
  çıktı başında GÖLGE/KARAR-DIŞI şerhi.
- edge_lab smoke (bilgisiz sentetik, 3000 satır): **5 kapının 5'i FAIL** → verdikt
  "KAPI(LAR) GEÇİLEMEDİ — kenar YOK" (yanlış-pozitif üretmiyor).

## 4) Anayasa denetimi

Sinyal/eşik/META_GATE/execution yönünde SIFIR yeni import; üretim `data/meta_model.*`'a
sıfır yazım; tüm durum yazımları atomic_io + UTC; `enforce` hiçbir modülde yok; tüm rapor
kapanışları reasoner CLOSING; edge_lab/lab çıktılarında kenar-iddiası yasağı testli.

## 5) Bilinçli sınırlar / sapmalar

1. **Lab sandbox = disiplin duvarı, düşman-dayanıklı değil:** pandas'ın kendi dosya-IO'su
   beyaz-listede kalır (analiz için zorunlu); attribute-zinciri kaçışları savunulmadı (belgeli).
2. `kelly_multi_asset` İŞARETLİ öneri vektörüdür (kısa bacak içerebilir) — tahsis yöntemi
   değil, Kelly-teorik referans satırı; `allow_short` kısıtı optimizasyon yöntemlerine uygulanır.
3. `agent.ask` LLM akıcılaştırmasında model önsözü ("İşte akıcılaştırılmış metin:") sızabiliyor —
   kozmetik; dürüstlük-guard (kaynak/kimlik korunumu) testli, önsöz temizliği backlog'da.
4. **Portföy-bakışı ve lab-köprüsü brain döngüsüne bağlanMADI** (plan Faz 2b'nin iki kalemi):
   canlı fiyat-paneli kaynağı gerektirir; `portfolio.propose`/`lab.run_experiment` bağımsız
   çağrılabilir durumda — kablolama Tur-17 adayı.
5. Overnight/intraday GERÇEK ölçümü yapılmadı (saatlik OHLC veri hazırlığı ister) —
   `edge_lab/BACKLOG.md` 1. sıra, harness hazır.
6. BOCPD `changepoint_prob` ham R_t(0) değil (sabit-hazard altında cebirsel sabit çıkar —
   ajan tespiti) → kısa-koşu posterior kütlesi kullanıldı (belgeli).

## 6) Sıradaki (kanıt-kapılı)

`AI_DECISION_MODE=shadow` + predictions_log'a `neural_pwin` kolonları (learning.sources
otomatik dolar) → conformal kalibrasyonun canlı beslenmesi → overnight/intraday veri
hazırlığı + ilk gerçek edge_lab koşusu → portföy/lab köprüleri → registry'ye ilk model
kartı (`train_neural --save` çıktısı).
