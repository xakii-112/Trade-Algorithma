# AI v2 "Quant Beyin" — Teslimat Raporu (2026-07-19, Tur-15)

**Görev:** Tur-13'te kurulan `nihai/ai` katmanını (router+memory+reasoner) borsa/ekonomi/
finans/quant'a **özelleşmiş, tam teşekküllü bir AI katmanına** geliştirmek: karar verme,
düşünme, hafıza, öğrenme, verileri matematiksel formüllerle işleme, derin matematik ve
parametre-kapasitesi artışı.

**Sonuç:** 7 yeni modül + 1 köprü; **938 test yeşil** (736→938; AI katmanı 17→219),
ruff temiz, mypy(nihai) 71 dosya temiz. Anayasa korunuyor: KARAR-DIŞI/GÖLGE.

---

## 1) Mimari — düşünme döngüsü

```
                    ┌────────────────────────────────────────────┐
                    │              brain.think()                 │
                    │  ALGILA → HATIRLA → TOPRAKLA → HESAPLA →   │
                    │        KARAR (gölge) → KAYDET              │
                    └────────────────────────────────────────────┘
                        │        │         │        │       │
   features/closes ─────┘        │         │        │       └─→ memory / episodic
   episodic.similar_episodes ────┘         │        │           (yalnız shadow kipinde)
   + LESSONS (ACI dersleri)                │        │
   knowledge (45 kavram) ──────────────────┘        │
   quant_math (rejim, vol, VaR, Kelly, MC...) ──────┘
                                                    │
   learning (Beta posterior, Brier, drift) ──→ decision.decide()
```

| Yetenek (hedef) | Modül | Ne verdi |
|---|---|---|
| Derin/zor matematik | `quant_math` | 40+ fonksiyon: 4 OHLC vol tahmincisi, VaR/CVaR/Cornish-Fisher, OU/Hurst/varyans-oranı, Engle-Granger, Kalman dinamik beta, Ledoit-Wolf, PCA riski, Monte Carlo (GBM+blok bootstrap), Black-Scholes+greekler, rejim sınıflandırıcı |
| Parametre artışı | `models` | Kapasite merdiveni (gerçek `param_count()` ile): aşağıdaki tablo |
| Hafıza | `episodic` (+mevcut `memory`) | Benzer-durum geri çağırma (z-normalize RMS), çözüm-birleştirmeli epizot günlüğü, 13 kayıtlı kurumsal ders kataloğu (`LESSONS`) |
| Öğrenme | `learning` | Çözülen canlı tahminlerden Beta-Binomial posterior + pwin kalibrasyonu (Brier/kovalar) + Page-Hinkley drift |
| Alan bilgisi | `knowledge` | 45 kavram (tanım + `quant_math` formül köprüsü + tuzaklar + ders bağları), `explain/search/glossary` |
| Karar verme | `decision` | GÖLGE Bayes motoru: prior×p_win×rejim → LONG_BIAS/SHORT_BIAS/ABSTAIN, şeffaf gerekçe zinciri, fractional-Kelly önerisi |
| Düşünme | `brain` | Uçtan uca düşünce zinciri + `--demo`/`--brief` CLI |
| Entegrasyon | `daily_summary --brain` | Günlük özete opsiyonel Beyin bölümü (fail-soft, bayraksız davranış aynı) |

## 2) Parametre kapasitesi artışı (istenen "AI modeli gibi büyüme")

25 özellik için **gerçek** sayımlar (`models.param_ladder_report(25)`):

| Katman | Parametre | Mevcut lojistik üretim modeline oran |
|---|---:|---:|
| logistic (bugünkü üretim) | 26 | 1× |
| elasticnet (etkileşimli) | 326 | ~13× |
| gbm_small (100 ağaç, derinlik 4) | ~2.660 düğüm | ~102× |
| mlp (64,32 gizli katman) | 3.777 | ~145× |

- `recommend_tier(n, f)` örnek/parametre oranı disiplini uygular: 11.874 örneklem → `gbm_small`;
  97.000 örneklem (Tur-5 Alpaca) → `mlp`. **Kapasite HAZIR; promotion kanıt-kapılıdır** —
  `research/purged_cv` + `live_retrain --promote` yolundan geçmeden hiçbir katman üretime
  giremez (ACI DERS #2/#5: az veriyle yüksek kapasite = overfit; kapasite eksikliği hiçbir
  zaman darboğaz değildi).
- Bilişsel "parametreler" ayrıca büyüdü: 45 kavram, 13 ders, GLOBAL/LONG/SHORT öğrenme
  posterior'ları, epizodik hafıza (sınırsız büyüyen ama 5MB/2000-satır korumalı).

## 3) İlk gerçek-veri koşusu — dürüstlük kanıtı

`learning.update_from_predictions(5000)` gerçek `predictions_log.csv` üstünde (OKUMA-YALNIZ):

```
GLOBAL: n=5000  isabet-posterior 0.458  (%95 CI 0.444–0.472)
LONG:   n=2688  isabet-posterior 0.458  (%95 CI 0.439–0.477)
SHORT:  n=2312  isabet-posterior 0.459  (%95 CI 0.438–0.479)
pwin kalibrasyonu: n=732  Brier=0.262  (0.25=bilgisiz → model bilgisizden KÖTÜ kalibre)
DRIFT UYARISI: yuvarlanan isabet serisinde kayma (pencere 50)
```

Yorum: forward'ın bilinen **NO_EDGE** durumu (isabet ~%47) bağımsız bir yoldan yeniden
doğrulandı; pwin Brier'i meta modelin kenarsızlığını (§16/§20) teyit ediyor. Gölge karar
motoru bu kanıtla **ABSTAIN** üretiyor — yön kararı için **prior CI tabanı > 0.5** şartı var
(belirsiz VEYA aleyhte kanıtta çekimser; aleyhte kanıttan ters-yön kenarı TÜRETİLMEZ —
bu yeni bir kenar iddiası olurdu, ACI DERS #3/#7).

`python -m nihai.ai.brain --demo` çıktısı (sentetik fiyat, yazmasız):

```
🧠 Düşünce [DEMO · event_breakout] ...
Rejim: down_high (trend=down, vol=high)
Risk: vol_yillik=0.2306 · cvar95_gunluk=0.03306 · max_dd=0.4709
Gölge karar: ABSTAIN (posterior 0.51, mod=off)
  · Prior (kaynak=GLOBAL): ortalama=0.458, %95 CI=[0.444, 0.472] ...
  · Karar: ABSTAIN (prior CI tabanı 0.5'in üstünde değil (belirsiz veya aleyhte kanıt)).
Bu değerlendirme karar vermez ve kenar iddiası içermez (ACI DERS #3/#4).
```

## 4) Anayasa (değişmedi, güçlendi)

1. `nihai/ai` sinyal/eşik/META_GATE/execution'a **bağlanmaz** — v2'de de hiçbir import yönü
   bunu ihlal etmiyor (decision/brain çıktıları yalnız rapor+JSONL+epizot).
2. `AI_DECISION_MODE`: `off` (varsayılan; **hiçbir dosyaya yazmaz**) | `shadow` (log-only).
   `enforce` **kodda yok**; env'e yanlışlıkla yazılırsa `off` muamelesi görür.
3. `models.save/load` yalnız çağıranın verdiği yola çalışır; üretim `data/meta_model.*`
   dosyalarına bu modülden yazılamaz (modülde DATA_DIR referansı bile yok).
4. Tüm durum yazımları `atomic_io` (OP-2 OneDrive dersi); tüm ts'ler UTC tz-aware (OP-1).
5. Rapor dilleri "kenar bulundu" iması taşıyamaz; her brif reasoner kapanışıyla biter.

## 5) Kullanım

```bash
python -m nihai.ai.brain --demo               # yazmasız uçtan uca gösterim
python -m nihai.ai.brain --brief              # günlük beyin brifi (salt-okunur)
python -m nihai.notifier.daily_summary --brain  # günlük özete Beyin bölümü ekle
python - <<'PY'                               # öğrenme durumunu güncelle (atomik)
from nihai.ai import learning
print(learning.learning_report(learning.update_from_predictions(5000)))
PY
# Gölge kipini açmak (kayıt biriktirme): ortamda AI_DECISION_MODE=shadow
```

## 6) Spec'ten gerekçeli sapmalar

1. **OP-6:** `predictions_log`'da `strategy` kolonu yok → öğrenme prior anahtarları
   GLOBAL/LONG/SHORT; `decision` strateji-adı bulamazsa GLOBAL'e düşer (gerekçe zincirinde
   `kaynak=` alanıyla şeffaf).
2. **Yön normalizasyonu:** canlı `direction` değerleri `+`/`-` — `_norm_dir` kanonikleştirir
   (ilk koşuda yakalandı; kalibrasyon etiketleri düzeltildi).
3. **CI kuralı sıkılaştı:** spec "CI 0.5'i kapsıyorsa ABSTAIN" idi; gerçek veri CI'sının
   tamamen 0.5 altında kalabildiği görülünce kural "CI tabanı > 0.5 değilse ABSTAIN"e
   çekildi (bkz. §3).

## 7) Süreç notu

Planlama/entegrasyon Fable 5 (orkestratör); kodlama 3 paralel Sonnet iş paketi (WP-A
quant_math+models, WP-C knowledge+decision tam teslim; WP-B episodic+learning oturum
limitine takıldı → orkestratör tamamladı — Tur-9 geri-düşüş deseni). Entegrasyon incelemesinde
ajan kodunda anayasa ihlali bulunmadı; 2 mypy düzeltmesi + sklearn deprecation + GLOBAL
geri-düşüşü + CI sıkılaştırması orkestratörce eklendi.

## 8) Sonraki adımlar (kanıt-kapılı)

- `AI_DECISION_MODE=shadow` açılıp gölge-karar + epizot birikimi başlatılabilir (yalnız log).
- Canlı dataset yeterince büyüyünce (`build_live_dataset`) kapasite merdiveni
  `research/purged_cv` düzeneğiyle katman-katman ölçülebilir; DSR/PBO + exec-lag kapıları geçmeden promotion yok.
- `daily_summary` zamanlanmış görevine `--brain` bayrağı eklenebilir (opsiyonel).
- ANTHROPIC_API_KEY girilirse `router` üstünden LLM akıcılaştırma (dürüstlük guardlı) canlanır.
