# AI v2 "Quant Beyin" — Mimari Spec (2026-07-19)

Tur-13'te kurulan `nihai/ai` (router+memory+reasoner) katmanının **tam teşekküllü, finans/quant-özelleşmiş
bilişsel katmana** yükseltilmesi: karar verme (GÖLGE), düşünme, epizodik hafıza, öğrenme, derin finans
matematiği ve **kanıtlanabilir parametre-kapasite artışı**.

## Değişmez anayasa (tüm modüller için)

1. **KARAR-DIŞI/GÖLGE sözleşmesi korunur.** Hiçbir yeni modül `nihai/signals`, `nihai/execution`,
   `META_GATE` veya eşiklere BAĞLANMAZ; üretim sinyali üretmez/bastırmaz/boyutlandırmaz. Karar motoru
   yalnız `off`/`shadow` modlarına sahiptir — `enforce` modu kasıtlı olarak YOKTUR (ACI DERS #3/#4/#5).
2. **Mevcut dosyalara dokunulmaz.** `router.py`, `memory.py`, `reasoner.py`, `__init__.py` ve nihai/ai
   dışındaki her şey OKUMA-YALNIZ. Her iş paketi yalnız kendi YENİ dosyalarını yazar.
   (`__init__.py` + `brain.py` entegrasyonu orkestratörde.)
3. **Yeni pip bağımlılığı YOK.** İzinli: numpy, scipy, pandas, scikit-learn; opsiyonel-import-guard'la:
   statsmodels, py_vollib, xgboost. Torch vb. EKLENMEZ.
4. **Proje konvansiyonları:** Türkçe docstring/yorum; `from __future__ import annotations`; py311+
   tipleme (`X | None`); ruff E,F,I,B,UP + line-length 110; loguru `logger`; modül-düzeyi veri yolları
   (`DATA_DIR / "..."`) testlerin monkeypatch'leyebilmesi için sabit isimli; tüm durum yazımları
   `nihai.atomic_io.atomic_write_text` ile (OneDrive/AV kilidi); ts'ler UTC tz-aware ISO
   (`datetime.now(UTC).isoformat(timespec="seconds")`); savunmacı okuma (yok/bozuk dosya → None/boş,
   çökme yok; bozuk satır sayısı `logger.warning`).
5. **Testler:** her modüle `tests/test_ai_<modül>.py`; deterministik (seed sabit), bilinen-değer
   assert'leri; veri yolları `monkeypatch.setattr(modül, "X_FILE", tmp_path/...)` deseniyle
   (bkz. `tests/test_ai_memory.py`). Üretim `data/`'sına asla yazma.
6. **Rastgelelik:** her Monte Carlo/model eğitimi `seed` parametresi alır; varsayılan sabit.

## Modüller ve halka-açık API sözleşmeleri

### 1) `nihai/ai/quant_math.py` — derin finans matematiği (WP-A)

Saf fonksiyonlar; numpy/scipy. Girdi `np.ndarray`/float; kısa/NaN'lı girdi → `None` (belgelenmiş).

```
simple_returns(prices) / log_returns(prices) -> np.ndarray
annualize_return(mean_ret, ppy) / annualize_vol(vol, ppy) -> float
zscore(x, window=None) -> np.ndarray
vol_close_to_close(close) | vol_parkinson(high, low) | vol_garman_klass(o,h,l,c)
  | vol_yang_zhang(o,h,l,c) | ewma_vol(returns, lam=0.94) -> float | None   # per-bar vol
var_historical(returns, alpha=0.05) / cvar_historical / var_cornish_fisher -> float | None  # kayıp büyüklüğü ≥0
max_drawdown(equity) -> tuple[float,int] | None    # (dd_frac, süre_bar)
downside_deviation(returns, mar=0.0) / ulcer_index(equity) -> float | None
sharpe(returns, rf=0.0, ppy=252) / sortino / calmar / omega(returns, threshold=0.0)
  / profit_factor(pnls) -> float | None
kelly_binary(p_win, win_loss_ratio) -> float       # negatif kenar → 0.0
kelly_continuous(mean_ret, var_ret) -> float
fractional_kelly(k, fraction=0.25, cap=0.02) -> float
ou_half_life(series) -> float | None               # AR(1) katsayısından
hurst_exponent(series, max_lag=20) -> float | None
variance_ratio(series, q) -> float | None
adf_test(series) -> tuple[float,float] | None      # statsmodels varsa; yoksa None (guard)
engle_granger(y, x) -> dict | None                 # {'beta','alpha','adf_stat','pvalue','half_life'}
kalman_beta(y, x, q=1e-5, r=1e-2) -> np.ndarray | None   # dinamik beta yolu
ledoit_wolf_cov(ret_matrix) -> np.ndarray | None   # sklearn.covariance.LedoitWolf
ewma_cov(ret_matrix, lam=0.94) -> np.ndarray | None
pca_risk(ret_matrix, n=3) -> dict | None           # {'explained': [...], 'loadings': [[...]]}
mc_gbm_paths(s0, mu, sigma, days, n_paths, seed) -> np.ndarray        # (n_paths, days+1)
mc_bootstrap_terminal(returns, horizon, n_paths, seed, block=5) -> dict  # {'p5','p50','p95','prob_loss'}
bs_price(s,k,t,r,sigma,kind='c') / bs_greeks(...) -> float|dict|None  # kendi kapalı-form (scipy.stats.norm)
classify_regime(close, high=None, low=None) -> dict
  # {"trend":"up"|"down"|"flat", "vol_bucket":"low"|"mid"|"high", "mean_reversion":bool, "label":str}
  # kural-tabanlı: SMA50 eğim işareti; gerçekleşen 20-bar vol'un 252-bar yüzdeliği (<%33 low, >%67 high);
  # hurst<0.45 → mean_reversion. Veri kısa → mevcut olanla düşür, en az 60 bar ister, azı → {"label":"unknown",...}
```

Test çapaları: BS fiyatı py_vollib ile çapraz-doğrula (varsa); Parkinson/GK bilinen sabit seri;
kelly elle hesap; max_drawdown bilinen eğri; mc_gbm seed determinizmi; engle_granger kointegre
sentetik çift (y=2x+ou-gürültü) → beta≈2.

### 2) `nihai/ai/models.py` — parametre kapasite merdiveni (WP-A)

```
PARAM_TIERS = ["logistic", "elasticnet", "gbm_small", "mlp"]
class CapacityModel:
    __init__(tier, feature_names, seed=42)
    fit(X, y) -> CapacityModel          # X:(n,f) float64, y:(n,) {0,1}
    predict_proba(X) -> np.ndarray      # p(y=1), (n,)
    param_count() -> int                # GERÇEK öğrenilmiş parametre sayısı
    meta() -> dict                      # {'tier','n_features','param_count','seed','trained_at'}
    save(path_base: Path)               # <base>.json (atomic, meta) + <base>.pkl (model)
    @classmethod load(path_base) -> CapacityModel | None
recommend_tier(n_samples, n_features) -> str
param_ladder_report(n_features=25) -> list[dict]   # katman başına kuramsal param sayısı
```

- logistic: `LogisticRegression(max_iter=1000)` → param = f+1
- elasticnet: `PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)` +
  `LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=3000)` → param = C(f,2)+f+1
- gbm_small: `HistGradientBoostingClassifier(max_iter=100, max_depth=4, early_stopping=True, random_state=seed)`
  → param_count = tüm ağaçlardaki düğüm sayısı (`sum(p.nodes.shape[0] ...)` — uygulanabilirse; değilse
  yaprak+bölünme sayımı; docstring'de yöntem)
- mlp: `MLPClassifier(hidden_layer_sizes=(64,32), early_stopping=True, max_iter=500, random_state=seed)`
  → param = Σ coefs + Σ intercepts
- `recommend_tier`: örnek/param oranı disiplini — n < 50f → logistic; n < 400f → elasticnet;
  n < 2000f → gbm_small; üstü → mlp (gerekçe docstring'de; ACI DERS #2/#5 bağlamı).
- Kayıt yolu SÖZLEŞMESİ: yalnız verilen `path_base` (test tmp'de). ÜRETİM `data/meta_model.*`
  dosyalarına HİÇBİR koşulda yazılmaz. Değerlendirme/promotion bu modülün işi DEĞİL
  (research/purged_cv + live_retrain --promote yolu).

Test çapaları: ayrılabilir sentetik veride 4 katman da fit+predict_proba (AUC>0.9); param_count
monotonik artar (logistic < elasticnet < mlp); save/load gidiş-dönüş; recommend_tier sınır değerleri.

### 3) `nihai/ai/episodic.py` — epizodik + semantik hafıza (WP-B)

```
EPISODES_FILE = DATA_DIR / "ai_episodes.jsonl"     # modül-düzeyi, monkeypatchable
@dataclass Episode: id(uuid4-hex-12), ts(UTC iso), kind, symbol|None,
                    features dict[str,float], summary, tags list[str], outcome dict|None
append_episode(kind, symbol=None, features=None, summary="", tags=None, outcome=None) -> Episode
resolve_episode(episode_id, outcome: dict) -> bool
  # JSONL append-only → çözüm AYRI satır: {"resolution_of": id, "ts":..., "outcome": {...}};
  # okuyucular birleştirir (en son resolution kazanır)
recent_episodes(kind=None, days=30, limit=500) -> list[Episode]
similar_episodes(features, k=5, kind=None, days=365) -> list[tuple[Episode, float]]
  # sayısal ortak anahtarlar üstünde z-normalize Öklid (istatistik epizot popülasyonundan);
  # ortak-anahtar kesişimi <3 → o epizot atlanır; hiç yoksa []
stats_by_tag(tag, days=365) -> dict   # {'n','win','loss','win_rate'|None} (outcome{'win':bool} sayımı)
@dataclass Lesson: id, claim, guard, source
LESSONS: list[Lesson]   # ACI DERSLER #1-#7 (CLAUDE.md) + operasyonel dersler (tz-canonik,
                        # atomik yazım/OneDrive, SYSTEM-watchdog, exec-lag, purge-simetri) ≥ 10 kayıt
check_against_lessons(text_or_tags) -> list[Lesson]  # küçük-harf anahtar-kelime eşleşmesi
```
Kilit/trim/bozuk-sayaç davranışı `memory.py` ile aynı desen (MAX ~5MB / son 2000 satır).

Test çapaları: append+recent+last gidiş-dönüş; resolve birleşimi; similar_episodes bilinen
en-yakın komşu; check_against_lessons("tek bölme PF") → DERS#3'ü içerir.

### 4) `nihai/ai/learning.py` — öğrenme döngüsü (WP-B)

```
LEARNING_STATE = DATA_DIR / "ai_learning_state.json"
PRED_LOG = DATA_DIR / "predictions_log.csv"        # OKUMA-YALNIZ
brier_score(pairs: list[tuple[float,int]]) -> float | None
reliability_bins(pairs, n_bins=10) -> list[dict]   # {'lo','hi','n','p_mean','y_rate'}
beta_posterior(wins, losses, prior_a=1.0, prior_b=1.0) -> tuple[float,float,float]  # (mean, lo95, hi95)
page_hinkley(values, delta=0.005, lam=0.05) -> dict  # {'drift': bool, 'index': int|None}
update_from_predictions(tail_n=5000) -> dict
  # predictions_log'un SON tail_n satırındaki ÇÖZÜLMÜŞ kayıtlardan strateji-başına
  # {n, win, loss, post_mean, lo95, hi95}; pwin kolonu doluysa kalibrasyon (brier + bins);
  # hit dizisinde page_hinkley drift bayrağı. Sonucu LEARNING_STATE'e atomik yaz + döndür.
  # Kolon adlarını nihai/execution/prediction_tracker.py'den DOĞRULA (savunmacı map;
  # dosya yok/boş/kolon eksik → {'strategies': {}, 'updated_at': ..., 'note': ...}).
  # Ucuz okuma: reasoner.py'deki _tail_lines/_read_header deseni (2M+ satır tam parse edilmez).
load_state() -> dict                                # yok/bozuk → boş varsayılan
learning_report(state=None) -> str                  # Türkçe, reasoner üslubu; kenar İMASI YOK
```
Rapor kapanışı reasoner ile uyumlu ton taşır: kalibrasyon/isabet İZLEME amaçlıdır, kenar kanıtı değildir.

Test çapaları: brier bilinen değer (ör. [(1.0,1)]→0, [(0.5,1)]→0.25); beta_posterior elle
(2 win/0 loss, uniform → mean=0.75); page_hinkley yapay kırılmada drift=True, sabit seride False;
update_from_predictions sentetik CSV'den (kolonlar gerçek şemayla) doğru sayım.

### 5) `nihai/ai/knowledge.py` — quant/finans bilgi çekirdeği (WP-C)

```
@dataclass Concept: id, ad, tanim, formul_ref list[str], tuzaklar list[str],
                    iliskili list[str], ders_ref list[str]
CONCEPTS: dict[str, Concept]   # ≥ 40 kavram (aşağıdaki zorunlu liste + serbest ekler)
explain(concept_id) -> str | None      # Türkçe blok: tanım + formül köprüsü + tuzaklar + ders bağları
formula_for(concept_id) -> list[str]   # quant_math fonksiyon adları (VAR OLAN adlar; uydurma yok)
search(query) -> list[Concept]         # ad/tanım küçük-harf substring
glossary() -> list[str]                # id listesi (alfabetik)
```
Zorunlu kavramlar: rejim, momentum, mean_reversion, pead, earnings_drift, survivorship_bias,
look_ahead_bias, exec_lag, bid_ask_bounce, kelly, var, cvar, sharpe, sortino, deflated_sharpe,
pbo, purged_cv, embargo, meta_labeling, kointegrasyon, hurst, ou_process, half_life,
parkinson_vol, garman_klass_vol, yang_zhang_vol, ewma_vol, kalman_filter, ledoit_wolf,
pca_risk, monte_carlo, block_bootstrap, black_scholes, delta, gamma, vega, theta, implied_vol,
put_call_parity, max_drawdown, position_sizing, drift_detection, brier, beta_binomial, carry.
`ders_ref` alanları episodic.LESSONS id'lerine işaret eder (ör. survivorship_bias → ACI-1/2).

Test çapaları: tüm formul_ref'ler gerçekten `nihai.ai.quant_math`'te var (getattr ile doğrula);
≥40 kavram; explain/search/glossary davranışı.

### 6) `nihai/ai/decision.py` — GÖLGE Bayes karar motoru (WP-C)

```
AI_DECISION_MODE: env "AI_DECISION_MODE" → "off" (varsayılan) | "shadow"; başka değer = off
DECISIONS_FILE = DATA_DIR / "ai_decisions.jsonl"
@dataclass ShadowDecision: ts, symbol, strategy, action("LONG_BIAS"|"SHORT_BIAS"|"ABSTAIN"),
    conviction float 0..1, prior_mean, posterior, evidence list[str] (Türkçe gerekçe zinciri),
    kelly_suggest, regime dict|None, lesson_warnings list[str], mode
decide(symbol, strategy, features, p_win=None, closes=None, side_hint=None, cost_bps=5.0) -> ShadowDecision
  # 1) prior: learning.load_state()['strategies'].get(strategy) → Beta posterior (yoksa mean=0.5, geniş CI)
  # 2) p_win doluysa sabit harman: post = 0.7*prior_mean + 0.3*clip(p_win, 0.01, 0.99)  (belgeli sabit)
  # 3) closes verilirse quant_math.classify_regime → statik strateji-rejim uyum tablosu (küçük ±0.02 ayar)
  # 4) episodic.check_against_lessons(strategy + tags) → lesson_warnings (ceza değil, şeffaflık)
  # 5) karar: CI TABANI (lo95) 0.5'in üstünde değilse (belirsiz VEYA aleyhte kanıt —
  #    ters-yön kenarı TÜRETİLMEZ) VEYA |2*post-1| - 2*cost_bps/1e4 <= 0 → ABSTAIN
  #    değilse side_hint/momentum işaretine göre LONG_BIAS/SHORT_BIAS; conviction = |2*post-1|
  # 6) kelly_suggest = fractional_kelly(kelly_binary(post, 1.0)) (ABSTAIN → 0.0)
  # 7) mode=="shadow" → DECISIONS_FILE'a append (memory.py kilit deseni) + episodic'e epizot;
  #    mode=="off" → HİÇBİR dosyaya yazmaz (saf hesap)
decision_report(days=7) -> str    # gölge kararların özeti: n, abstain oranı, ort. conviction, uyarılar
```
Docstring'e AYNEN: "KARAR-DIŞI/GÖLGE: bu modülün çıktısı hiçbir üretim sinyalini üretmez, bastırmaz
veya boyutlandırmaz; strategy/execution/META_GATE'e bağlanamaz. enforce modu kasıtlı olarak yoktur."

Test çapaları: boş prior → ABSTAIN; güçlü prior (ör. 40 win/10 loss) → yön + conviction>0;
mode=off dosya yazmaz, shadow yazar (tmp'ye monkeypatch); kelly ABSTAIN'de 0; evidence zinciri dolu.

### 7) `nihai/ai/brain.py` — düşünme döngüsü orkestratörü (ORKESTRATÖRDE — ajanlar YAZMAZ)

`think()` (algıla→hatırla→topraklama→hesapla→karar→kaydet), `daily_brief()`, `--demo`/`--brief` CLI.

## İş paketleri

| WP | Dosyalar | Ajan |
|----|----------|------|
| A | quant_math.py, models.py + testleri | Sonnet |
| B | episodic.py, learning.py + testleri | Sonnet |
| C | knowledge.py, decision.py + testleri | Sonnet |
| entegrasyon | brain.py, __init__.py, daily_summary köprüsü, tam suite, commit | Fable (orkestratör) |

Çapraz bağımlılıklar YALNIZ bu spec'teki imzalar üstünden: WP-C, quant_math/learning/episodic'i
import eder ama yalnız burada yazılı adları kullanır (modül henüz yoksa testlerde monkeypatch/duck-type
DEĞİL — import normal yapılır; entegrasyon sırası: A+B bitmeden C'nin testleri koşmayabilir, sorun değil;
C ajanı kendi testlerini "import edilebilirlik varsayımıyla" yazar, orkestratör tam suite'te doğrular).

## Parametre artışı raporu (entegrasyonda üretilecek)

Mevcut üretim meta modeli: lojistik ~26 parametre (25 özellik + bias). Merdiven (25 özellik):
elasticnet ~326, gbm_small ~10-30k düğüm, mlp(64,32) ~3.8k ağırlık — `param_ladder_report()` gerçek
sayıları verir. Kapasite HAZIR; promotion yalnız kanıt-kapılı manuel yol (live_retrain --promote).
