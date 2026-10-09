# AI v3 "Derin Quant Beyin" — UYGULAMA SPEC'İ (ajan brifleri)

**Tarih:** 2026-07-23 · **Tur:** 16 · **Orkestratör:** Fable 5
**Tek gerçek kaynak:** `docs/AI_V3_DERIN_BEYIN_PLAN_2026-07-19.md` (bu SPEC yalnız sözleşmeleri sabitler;
WP içerik detayı planın BÖLÜM E'sindedir). **Baz:** commit `2fa0988` (938 test yeşil, ruff+mypy temiz).
**Ortam:** venv `.venv\Scripts\python.exe` (Python 3.12, **torch 2.13.0+cpu KURULU**, sklearn/scipy mevcut).

---

## 0. DEĞİŞMEZLER (anayasa — pazarlıksız)

1. **KARAR-DIŞI/GÖLGE:** `nihai/ai` hiçbir üretim sinyal/eşik/META_GATE/execution koduna bağlanmaz.
   `nihai/ai` içinden `nihai.signals`/`nihai.strategy`/`nihai.main`/execution yönüne import **YOK**;
   üretim koduna `nihai.ai` importu eklemek de **YASAK** (köprüler yalnız orkestratörün fail-soft noktaları).
2. **Üretim `data/meta_model.*` dosyalarına hiçbir modülden yazım yok.** save/load YALNIZ çağıranın
   verdiği `path_base`'e (models.py sözleşmesi). Modül-düzeyi DATA_DIR sabiti yalnız kendi `ai_*` dosyaların için.
3. **Rapor dili:** "kenar bulundu" iması SIFIR; brif-tipi çıktılar `reasoner.CLOSING` ile biter.
4. **Look-ahead yasak:** her özellik yalnız karar barına kadarki veriden (`df.iloc[:i+1]` kuralı).
5. **Yazımlar atomik:** `nihai.atomic_io.atomic_write_text` / `atomic_replace` (OneDrive dersi).
   Tüm ts'ler UTC: `datetime.now(UTC).isoformat(timespec="seconds")`.
6. **Determinizm:** stokastik her yol `seed` parametreli; aynı seed+veri → aynı çıktı (test edilir).
7. **Savunmacı-girdi:** bozuk/kısa/NaN **VERİ** → `None` dönüşü (quant_math deseni; çökme yok).
   API ihlali (**programcı hatası**) → `ValueError`/`RuntimeError` (models.py deseni).
8. **fail-soft opsiyonel bağımlılık:** torch `try: import torch ... except ImportError: TORCH_AVAILABLE = False`
   deseniyle; torch yokken modül import edilebilir kalır, torch'lu testler `skipif` ile atlanır,
   v2 davranışı hiçbir yerde değişmez.

## 1. KOD SÖZLEŞMELERİ

- **Türkçe docstring/yorum**, mevcut yoğunluk/üslupta (docstring "neden"i anlatır, satır yorumu kısıt belirtir).
- ruff: E,F,I,B,UP; line-length 110. mypy temiz: `from __future__ import annotations` + tam tip işaretleri.
- logger: `from loguru import logger`, `logger.debug(f"[ai.<modül>] ...")`; kütüphane katmanında print yok
  (CLI `main()` içinde print serbest).
- Test dosyaları `tests/test_ai_<modül>.py`; MEVCUT test dosyasını genişlet (yeniden adlandırma yok).
- Mevcut koda **cerrahi** dokun: bölümleri yeniden biçimlendirme, mevcut fonksiyon imzalarını kırma yok.

## 2. SAHİPLİK MATRİSİ — Dalga-1 (başka ajanın dosyasına DOKUNMA)

| Ajan | Yeni dosya | Genişletir | Test dosyaları |
|---|---|---|---|
| **Sonnet-1** | `nihai/ai/neural.py`, `research/train_neural.py` | `nihai/ai/learning.py` | `tests/test_ai_neural.py` (yeni), `tests/test_ai_learning.py`, `tests/test_train_neural.py` (yeni) |
| **Sonnet-2** | — | `nihai/ai/quant_math.py`, `nihai/ai/knowledge.py` | `tests/test_ai_quant_math.py`, `tests/test_ai_knowledge.py` |
| **Sonnet-3** | `nihai/ai/consolidate.py` | `nihai/ai/decision.py`, `nihai/ai/episodic.py` | `tests/test_ai_decision.py`, `tests/test_ai_episodic.py`, `tests/test_ai_consolidate.py` (yeni) |

**ORKESTRATÖR SAHASI (ajanlar dokunmaz):** `nihai/ai/models.py`, `nihai/ai/brain.py`,
`nihai/ai/__init__.py`, `requirements.txt`, `pyproject.toml`, `conftest.py`, `docs/*`, git işlemleri.

**Dalga-2 (çekirdek entegrasyonundan sonra):** WP-M2 `advanced_math.py` (Sonnet-2) ·
WP-C `lab.py` + WP-P `portfolio.py` + WP-E `research/edge_lab/` + WP-O `data_health.py`/`registry.py`
(Sonnet-4) · WP-T `text_intel.py` + WP-X `explain.py` + WP-A `agent.py`/`selfcheck.py` (Sonnet-5).

## 3. ARAYÜZ SÖZLEŞMELERİ (Dalga-1)

### 3.1 `nihai/ai/neural.py` (Sonnet-1, WP-N)

- Modül seviyesi `TORCH_AVAILABLE: bool`; torch yokken sınıf kurucuları
  `RuntimeError("torch kurulu değil — neural katman devre dışı (fail-soft)")` fırlatır; modül importu asla patlamaz.
- `NEURAL_TIERS: list[str] = ["deep_mlp", "quant_transformer"]`.
- `class NeuralCapacity:` kurucu `(tier: str, feature_names: list[str], seed: int = 42)` —
  **CapacityModel API paritesi** (bkz. `nihai/ai/models.py`):
  - `fit(X, y, aux: dict[str, np.ndarray] | None = None) -> NeuralCapacity` — `fit(X, y)` tek başına çalışmalı;
    `aux` opsiyonel yardımcı hedefler: `{"magnitude": (n,) int kova [0..4], "vol": (n,) float, "regime": (n,) int [0..5]}` —
    verilenler çok-görevli kayba ağırlıklı eklenir, verilmeyen başlıklar eğitimsiz kalır (parametreleri yine sayılır).
  - `predict_proba(X) -> np.ndarray` şekil `(n,)`, p(y=1).
  - `predict_with_uncertainty(X, T: int = 20) -> tuple[np.ndarray, np.ndarray]` — MC-dropout (ortalama p, std).
  - `param_count() -> int` — gerçek `sum(p.numel())`.
  - `meta() -> dict` — models.py ile aynı anahtarlar (`tier/n_features/param_count/seed/trained_at`).
  - `save(path_base)` — `<base>.json` (meta+feature_names, `atomic_write_text`) + `<base>.pt`
    (state_dict, tmp'ye yaz → `atomic_replace`). YALNIZ verilen path_base.
  - `classmethod load(path_base) -> NeuralCapacity | None` — savunmacı (dosya yok/bozuk → None).
  - `is_fitted` property; fit-öncesi predict/param_count/save → `RuntimeError`.
- **Eğitim reçetesi:** AdamW, dropout 0.2, weight_decay 1e-4, grad-clip 1.0, val = kronolojik SON %15
  (**shuffle YOK — zaman serisi**), early-stop patience 10, `max_epochs` parametreli (varsayılan 100),
  `torch.manual_seed(seed)` + numpy seed → determinizm.
- **Mimari:** plan BÖLÜM D tablosu. `deep_mlp` = (512,256,128,64) MLP ≈380k. `quant_transformer`:
  per-feature gömme d=256, [CLS]+[SCALE] token, 6 blok × (4 başlık, FFN 1024, pre-LN), çok-ölçekli füzyon
  başlığı (3 ölçek özeti 768→512→256), çok-görevli başlıklar (yön/büyüklük/vol/rejim) + sıcaklık + 2 probe head.
  **TEST SÖZLEŞMESİ: `param_count() >= 3_777_000`** (gerçek sayı ~5.5M raporlanır; ±%10 serbest).
- **Çoklu zaman-ölçeği:** feature_names önek konvansiyonu — `h_*` saatlik, `d_*` günlük, `w_*` haftalık,
  öneksiz = ölçeksiz; [SCALE] tip gömmesi bu öneklerden. Önek hiç yoksa tek ölçek (savunmacı, yine çalışır).
- neural.py `models.py`'ı **import etmez** (orkestratör models.py'dan lazy import edecek).

### 3.2 `nihai/ai/learning.py` v3 (Sonnet-1, WP-L)

- Mevcut API korunur. YENİ state alanı (geriye-uyumlu):
  `state["sources"] = {"prior": {"brier": float|None, "n": int}, "pwin": {...}, "neural": {...}}` —
  `update_from_predictions` doldurur (neural gölge logu yoksa `None`); okuyanlar alan yoksa degrade eder.
- YENİ `fit_calibrator(pairs: list[tuple[float, int]], method: str = "platt") -> dict | None` —
  YALNIZ verilen (OOS) çiftlerde fit; dönüş katsayılar + önce/sonra Brier. `apply_calibration(p, calib) -> float`.
- YENİ: `update_from_predictions` dönüşüne `retrain_candidate: bool` (Page-Hinkley tetiklendiyse) —
  otomatik eğitim YOK, yalnız bayrak + rapor satırı.
- learning.py `research/`'e import **veremez** (üretim→research bağımlılığı yasak); replay join'i train_neural'da.

### 3.3 `research/train_neural.py` (Sonnet-1, WP-L)

- CLI: `python -m research.train_neural --tier quant_transformer --folds 5 [--dataset yol.csv]
  [--synthetic N] [--max-epochs N] [--save path_base]`.
- purged+embargolu K-fold bölme mantığını `research/purged_cv.py`'dan **yeniden kullan** (import; kopyalama yok);
  DSR için `research/deflated.py` mevcut — kullan. Fold-başı AUC + özet verdikt; sonuç NE ÇIKARSA o raporlanır.
- `--synthetic N`: BÖLÜM F sentetik müfredat (GBM + blok-bootstrap + OU + rejim karışımı; seed'li; etiket leak-free).
- `--save` verilmedikçe HİÇBİR dosya yazılmaz; verilirse `NeuralCapacity.save(path_base)` (data/meta_model* yasak).

### 3.4 `nihai/ai/quant_math.py` v3 (Sonnet-2, WP-M)

- Yeni fonksiyonlar (plan BÖLÜM E tablosu; imza deseni mevcutla aynı — ArrayLike girdi,
  `float|None` / `dict|None` / `np.ndarray|None` dönüş, savunmacı):
  `garch11_fit`, `garch11_forecast`, `evt_hill_tail_index`, `evt_gpd_var_es`,
  `copula_fit_gaussian`, `copula_fit_t`, `tail_dependence`, `frac_diff_ffd`,
  `welch_psd`, `haar_dwt_vol`, `almgren_chriss_cost`, `permutation_entropy`,
  `sample_entropy`, `mutual_info`, `coskewness`, `cokurtosis`.
  **NOT:** `omega_ratio` EKLENMEZ — mevcut `omega()` zaten var (çift tanım yasak).
- Yalnız numpy/scipy (statsmodels'e runtime bağımlılığı ekleme). Mevcut kodu yeniden düzenleme;
  yeni fonksiyonları dosya sonuna, mevcut bölüm-yorumu düzeninde gruplar halinde ekle.
- GARCH(1,1): scipy MLE (kısıtlı: ω>0, α≥0, β≥0, α+β<1); yakınsamazsa None. Testte sentetik geri-bulma.

### 3.5 `nihai/ai/knowledge.py` v3 (Sonnet-2, WP-K)

- `Concept` dataclass'ına yalnız **default'lu** alan eklenebilir (öneri: `prereqs: list[str] = field(default_factory=list)`);
  mevcut 45 kavramın id'leri/alanları AYNEN korunur (mevcut testler kırılmaz).
- +55 yeni kavram (plan gruplarına göre) → **toplam ≥100**; her kavram mevcut girdilerin üslup/derinliğinde
  (tanım + formül köprüsü + tuzaklar + ders bağları). Formül köprüleri GERÇEK fonksiyon adlarına çözünmeli
  (quant_math v3 yenileri dahil — Sonnet-2 ikisini de yazdığı için senkron).
- YENİ `curriculum() -> list[str]`: `prereqs` DAG'ının deterministik topolojik sırası; çevrim → test hatası.

### 3.6 `nihai/ai/decision.py` v3 (Sonnet-3, WP-D)

- `decide` imzası genişler (yeni parametreler SONA, hepsi default'lu — geriye uyumlu):
  `decide(symbol, strategy, features, p_win=None, closes=None, side_hint=None, cost_bps=5.0,
  neural_pwin: tuple[float, float] | None = None, conformal_interval: tuple[float, float] | None = None)`.
- **Hiyerarşik birleşim:** kaynaklar = prior (Beta) ⊕ p_win ⊕ `neural_pwin[0]` (std büyükçe katkı kısılır) ⊕ rejim ayarı.
  Kaynak ağırlıkları `learning.load_state()["sources"][k]["brier"]`'den monoton (ör. `w ∝ max(0.01, 0.25 − brier)`,
  belgele + monotonluk testi); **sources yoksa/boşsa mevcut sabit ağırlık davranışına SAYISAL DENK degrade**.
- **Fayda katmanı:** `E = post·kazanç − (1−post)·kayıp − λ·|CVaR95|`; kazanç/kayıp closes'tan savunmacı
  tahmin (yoksa 1:1); λ = `AI_UTILITY_LAMBDA` env (varsayılan 0.5, çağrı anında okunur). ABSTAIN ⟺ E ≤ 0.
- **KAPILAR (sıkılaşma):** yön çıktısı için (a) mevcut "prior CI tabanı > 0.5" AYNEN **ve** (b) E>0
  **ve** (c) `conformal_interval` verilmişse aralık 0.5'i KAPSAMAMALI (kapsıyorsa ABSTAIN; None → kapı pasif,
  Dalga-2 bağlayacak).
- `ShadowDecision`'a yeni alanlar yalnız SONA ve default'lu (ör. `utility: float | None = None`);
  mevcut alan sırası korunur (JSONL uyumu).
- **PARİTE TESTİ ŞART:** `neural_pwin=None` + sources alanı yokken v2 ile aynı girdi → aynı action/posterior (±1e-9).
- decision.py `neural.py`'ı import **etmez** (brain bağlar).

### 3.7 `nihai/ai/episodic.py` v3 + `nihai/ai/consolidate.py` (Sonnet-3, WP-H)

- `similar_episodes(..., metric: str = "rms")` — `"cosine"` seçeneği (z-norm özellik ⊕ rejim one-hot ⊕ sonuç);
  varsayılan `"rms"` (davranış değişmez); bilinmeyen metric → ValueError.
- **Unutma eğrisi:** `_maybe_trim` artık düşük-skorludan kırpar; skor = f(yaş, erişim sıklığı, bilgi değeri) —
  deterministik, belgeli; 5MB/2000-satır koruması aynen.
- **Ders-adayı:** `propose_lesson_candidate(claim, evidence, source) -> dict` →
  `data/ai_lesson_candidates.jsonl` (atomic append; `status="candidate"`). `LESSONS` kataloğuna
  OTOMATİK terfi YOK (test: katalog uzunluğu değişmez).
- `consolidate.py`: `consolidate(days: int = 90, min_cluster: int = 5) -> dict` — benzer epizotları
  deterministik kümeler, `data/ai_semantic_memory.jsonl`'e atomic append
  (`{"pattern": ..., "n": ..., "isabet": ..., "ts": ...}`); **idempotent** (aynı girdiyle 2. koşu satır eklemez).
  `semantic_recall(features: dict, k: int = 3) -> list[dict]` — brain HATIRLA adımı için.

## 3B. ARAYÜZ SÖZLEŞMELERİ — Dalga-2 (çekirdek DONDURULDU: commit `34f3f52`, 1207 test yeşil)

**Dalga-2 sahiplik matrisi** (çekirdek dosyalara — models/brain/__init__/decision/episodic/
learning/neural/quant_math/knowledge/consolidate — YAZMAK YASAK; okumak serbest):

| Ajan | Yeni dosyalar | Test dosyaları (hepsi YENİ) |
|---|---|---|
| **Sonnet-2b** | `nihai/ai/advanced_math.py` | `tests/test_ai_advanced_math.py` |
| **Sonnet-4** | `nihai/ai/lab.py`, `nihai/ai/portfolio.py`, `nihai/ai/data_health.py`, `nihai/ai/registry.py`, `research/edge_lab/__init__.py`, `research/edge_lab/harness.py`, `research/edge_lab/BACKLOG.md` | `tests/test_ai_lab.py`, `tests/test_ai_portfolio.py`, `tests/test_ai_data_health.py`, `tests/test_ai_registry.py`, `tests/test_edge_lab_harness.py` |
| **Sonnet-5** | `nihai/ai/text_intel.py`, `nihai/ai/explain.py`, `nihai/ai/agent.py`, `nihai/ai/selfcheck.py` | `tests/test_ai_text_intel.py`, `tests/test_ai_explain.py`, `tests/test_ai_agent.py`, `tests/test_ai_selfcheck.py` |

### 3.8 `advanced_math.py` (Sonnet-2b, WP-M2)
- quant_math ile AYNI savunmacı sözleşme (bozuk VERİ→None; ArrayLike imzalar; determinizm
  seed'li). Fonksiyon aileleri plan WP-M2 tablosu: `merton_jump_fit`, `jump_detect_bipower`,
  `heston_charfn`, `carr_madan_fft_price`, `heston_calibrate`, `american_crank_nicolson`,
  `longstaff_schwartz_lsmc`, `hawkes_fit`, `hawkes_intensity`, `bocpd`,
  `kalman_regime_switching`, `particle_filter_light` (≤1000 parçacık), `realized_vol_hurst`,
  `rough_vol_forecast`, `signature_level2/3`, `matrix_profile_light`,
  `qp_solve(P, q, a_eq=None, b_eq=None, lb=None, ub=None) -> np.ndarray | None` (SLSQP sarımı),
  `adaptive_quad`, `logsumexp_stable`, `fft_convolve`.
- Analitik-limit testleri ZORUNLU: Heston→BS limiti, Merton λ→0 → BS, CN-PDE vs BS
  (Amerikan call temettüsüz ≈ Avrupa), LSMC vs CN paritesi (tolerans belgeli).

### 3.9 `lab.py` (Sonnet-4, WP-C)
- `run_code(code: str, inputs: dict | None = None, timeout_s: int = 30) -> dict` —
  KISITLI sandbox: beyaz-liste namespace (numpy/scipy/pandas + quant_math/advanced_math),
  `open`/`__import__`(beyaz-liste dışı)/ağ/dosya-yazımı ENGELLİ; süre limiti; sonuç
  {"ok", "output", "error", "elapsed_s"}. Her koşu `data/ai_lab_runs.jsonl`'e atomik loglanır.
- `run_experiment(name, **kwargs) -> dict` şablonları (LLM'siz deterministik):
  `vol_model_karsilastirma`, `korelasyon_rejimi`, `olay_etudu`, `dagilim_testi`,
  `ozellik_tarama` (plan WP-C listesi).
- LLM analiz döngüsü (`analyze(question)`) router üstünden FAIL-SOFT (anahtar yoksa şablon moduna düşer).
- KAÇIŞ TESTLERİ ZORUNLU: dosya yazma / os,sys,subprocess importu / open / ağ denemesi →
  hepsi engellenir; süre aşımı kesilir. `native/` iskeleti YAZILMAZ (profil kanıtı yok — plan kararı).

### 3.10 `portfolio.py` (Sonnet-4, WP-P — GÖLGE)
- `propose(candidates: list[dict], closes_panel, constraints: dict | None = None) -> dict` —
  ağırlık vektörü + YÖNTEM KARŞILAŞTIRMA tablosu (min-var, max-Sharpe, HRP, ERC, BL, CVaR-LP,
  çok-varlık Kelly aynı raporda) + gerekçe; çıktı başlığında sabit GÖLGE ibaresi + reasoner.CLOSING.
- Kovaryans: quant_math.ledoit_wolf_cov / ewma_cov + tek-faktör; kondisyon sayısı raporlanır.
- QP: `advanced_math.qp_solve`'u FONKSİYON İÇİNDE lazy dene; import/None durumunda kendi
  scipy SLSQP yedeğinle devam (Dalga-2 içi bağımlılık gevşek — paralel ajan teslimi bloklamaz).
- CVaR: Rockafellar-Uryasev LP (`scipy.optimize.linprog`). Kısıtlar: pozisyon tavanı,
  net-directional cap, ciro cezası (bps parametrik). Stres: 2008/2020/2022 pencere replay
  (panelden; yoksa None) + MC (quant_math.mc_gbm_paths). Atribüsyon: sembol/strateji katkı.
- Rapor `data/ai_portfolio_proposals.jsonl` (atomik, gölge). Execution'a köprü YOK.

### 3.11 `research/edge_lab/` (Sonnet-4, WP-E)
- `harness.py`: `run_candidate(name, df_builder, signal_fn, cfg) -> dict` — purged CV →
  çift-bölme kapı → DSR (`research/deflated.py`) → PBO (`research/pbo.py`) → **exec-lag 1
  duyarlılığı (ACI DERS #7)** → PIT bayrağı → tek-sayfa Türkçe verdikt (EDGE_FINDINGS formatı;
  "kenar" ilanı YOK — yalnız kapı geçti/geçmedi tablosu). CLI: `python -m research.edge_lab.harness --demo`
  (sentetik veriyle kapıların uçtan uca koştuğunu gösterir).
- `BACKLOG.md`: plan WP-E aday listesi öncelik sırasıyla + durum sütunu + denenen-aday sayacı
  (DSR çoklu-test girdisi). İlk aday `overnight_intraday` işaretlenir (Faz 3'te orkestratör koşar).

### 3.12 `data_health.py` + `registry.py` (Sonnet-4, WP-O)
- `data_health.health_check(ohlcv_df, symbol=None) -> dict`: {"score":0..100, "issues":[...],
  "quarantine": bool} — eksik-bar (takvim varsa; yoksa bar-aralığı sezgisi), donmuş-fiyat,
  sıçrama-outlier (advanced_math.jump_detect_bipower LAZY; yoksa kendi bipower'ı) + split-faktör
  ayrımı, hacim anomalisi, NaN/tip. Deterministik, savunmacı.
- `registry.register(path_base, metrics: dict, notes: str = "") -> dict`: model kartı
  (JSON: hash, veri dönemi, hiperparam, OOS metrikler, DSR, ortam) → `data/ai_models/`
  altında sürümlü atomik kayıt; `list_models()`, `get_card(model_id)`.
  Gölge A/B: `ab_pick(model_ids, stream_key) -> str` deterministik atama + iki modelin
  izolasyon garantisi (birinin durumu diğerini bozamaz — test).
- Zamanlanmış görev TANIMLARI yalnız docstring/rapor (kayıt kullanıcı onayıyla — SYSTEM-zombi dersi).

### 3.13 `text_intel.py` (Sonnet-5, WP-T)
- Deterministik taban HER ZAMAN: `tone_score(text) -> dict` (Loughran-McDonald tarzı finans
  sözlüğü — modüle gömülü çekirdek alt-küme, kaynağı docstring'de; pozitif/negatif/belirsizlik/
  dava sınıf sayımları + net skor), `novelty_score(text, prior_texts) -> float` (BoW kosinüs),
  `headline_density(counts_by_symbol) -> dict` (z-skor).
- LLM katmanı router üstünden FAIL-SOFT; LLM çıktısı ASLA tek başına özellik olmaz —
  deterministik skorla YAN YANA yazılır (dürüstlük guard'ı; test).
- `market_calendar` alt-API'si: `next_macro_events(date, n=3)` (statik 2026 FOMC/CPI/NFP
  tablosu + elle güncelleme notu), `opex_week(date) -> bool`, `quarter_end(date) -> bool`,
  `earnings_proximity(symbol, date) -> dict | None` (mevcut nasdaq-earnings köprüsünü ARA —
  Tur-14 P0'da eklendi; bulunamazsa fail-soft None). Hepsi tz-güvenli (UTC).
- Skor kayıtları `data/ai_text_scores.jsonl` (atomik, gölge).

### 3.14 `explain.py` (Sonnet-5, WP-X)
- Model-bağımsız: `permutation_importance(model, X, y, seed=42) -> list[dict]` (predict_proba
  duck-tipli). Torch'lu (KENDİ try-import'u; nihai.ai.neural'a import bağı YOK — model objesi
  parametre olarak gelir): `integrated_gradients(neural_model, x_row, baseline=None) -> dict | None`,
  `attention_rollout(neural_model, x_row) -> dict | None` (yalnız transformer; değilse None).
- **Conformal:** `conformal_calibrate(p_cal: array, y_cal: array, alpha=0.1) -> dict | None`
  (split-conformal, YALNIZ verilen kalibrasyon dilimi) + `conformal_interval(p, calib) ->
  tuple[float, float] | None`. Bu çıktı decision.decide(conformal_interval=...) girişinin
  BESLEYİCİSİDİR (parametre çekirdekte HAZIR; brain bağlaması orkestratörde).
- `decision_card(decision_dict, attributions, conformal, extras=None) -> dict` + Türkçe metin
  hali; `global_model_report(model, X, y) -> str` (önem sıralaması + ilk-10 PDP özeti).
- Test ZORUNLU: sentetik conformal kapsam simülasyonu nominal %90'ı tutar (tolerans belgeli);
  IG tamlık aksiyomu (atıf toplamı ≈ çıktı farkı, torch varsa).

### 3.15 `agent.py` + `selfcheck.py` (Sonnet-5, WP-A)
- `agent.ask(question: str) -> dict` {"answer", "sources": [...]} — RAG: knowledge (100+ kavram
  + curriculum) + episodic/semantic + docs/EDGE_FINDINGS* metin araması; kaynak atfı zorunlu;
  LLM akıcılaştırma fail-soft; cevap reasoner.CLOSING ile biter.
- `agent.plan(goal: str) -> dict` → alt-görev kuyruğu `data/ai_agenda.jsonl` (atomik; kaldığı
  yerden devam); `agent.run_next() -> dict | None`. SINIR (testle kanıtlı): görev tipleri YALNIZ
  analiz/rapor — üretim dosyası yazan görev tipi YOK; kuyruk derinlik limiti + günlük tavan.
- `selfcheck.knowledge_exam(n=50, seed=42)` (bilinen-cevaplı; formül köprüleri sayısal doğrulanır),
  `selfcheck.calibration_exam()` (son N gölge karar Brier/isabet trendi),
  `selfcheck.replay_exam(closes_by_symbol, period_label)` (bilinen dönem replay — ABSTAIN oranı,
  rejim tespiti), `selfcheck.report_card() -> dict` + Türkçe karne metni (CLOSING'li).
- Karşı-olgusal: `agent.counterfactual_grid(resolved_rows, lambdas, thresholds) -> dict` —
  decision-motoru ızgara taraması, SONUÇ SADECE RAPOR (eşik otomatik değişmez; test).

## 4. TEST + KOŞU (ajan doğrulaması)

```powershell
# repo kökünden (yol boşluklu — tırnakla):
& ".venv\Scripts\python.exe" -m pytest tests/test_ai_neural.py -q     # yalnız KENDİ test dosyaların
& ".venv\Scripts\python.exe" -m ruff check nihai/ai/neural.py tests/test_ai_neural.py
```

- Değiştirdiğin modülün MEVCUT testleri de yeşil kalmalı (ör. Sonnet-3 → `tests/test_ai_decision.py` tamamı).
- Torch'lu testler: `@pytest.mark.skipif(not neural.TORCH_AVAILABLE, reason="torch yok")`
  (importorskip DEĞİL — modül fail-soft).
- Ağır eğitim testleri küçük tutulur (küçük n, `max_epochs<=5`, quant_transformer testinde d/blok küçültme YOK —
  param sözleşmesi gerçek mimaride ölçülür; ama fit çağrısı 1-2 epoch'la sınırlanabilir).
- Gerçek `data/`'ya yazma yok — kök `conftest.py` `NIHAI_DATA_DIR`'ı tmp'ye yönlendirir, buna güven.
- Tam suite + mypy + entegrasyon ORKESTRATÖRDE (sen koşma).

## 5. AJAN DÖNÜŞ RAPORU (zorunlu format)

1. Yazılan/değişen dosyalar (+ yaklaşık satır).
2. Test sayısı: eklenen adet + koşu sonucu (komut + özet satırı).
3. Sözleşmeden sapmalar (varsa) + NEDENİ.
4. Orkestratöre entegrasyon notları (models.py/brain.py bağlanırken bilinmesi gerekenler,
   API'de plan-dışı her detay).
