# Geliştirme Planı — Tier 1 + Tier 2 (2026-06-22)

Onaylanan kapsam: kenar (alpha) İDDİASI taşımayan, projenin overfit disiplinini
(docs/EDGE_FINDINGS.md) bozmayan kalemler. **Veri bütçesi kararı: SADECE ÜCRETSİZ**
(yfinance/Alpaca-IEX/TradingView/Stooq + SEC EDGAR + Reddit). Ücretli PIT veri YOK.

> Bu kararın iki sonucu:
> 1. Smart Router gerçek point-in-time veri değil → **sağlık-tabanlı failover** olur.
> 2. Survivorship doğrulama için mevcut ücretsiz yol (`pit_universe` = Wikipedia + yfinance,
>    ~%60 earnings kapsama) genişletilir; veri satın alınmaz. Bu yüzden **canlı paper
>    kanıt (prediction_tracker, FAZ D)** kenar sorusuna tek dürüst cevap yolu → önemi arttı.

---

## Mimari ön-karar (ortak temel)
`trade_log.txt` insan-okur blok formatında; parse etmek kırılgan. Kapanışta yapılandırılmış
**`data/trades.csv`** yazılır → post-mortem (A1), dashboard (A2), VaR (A3) hep bu temiz
kaynaktan beslenir. İlk iş bu (A0).

---

## FAZ A — Sıfır bağımlılık
- **A0** `data/trades.csv` çıktısı — `_close`'da `log_trade_postmortem` yanına `log_trade_csv`.
- **A1** `research/postmortem.py` — trades.csv + equity_curve.csv → Sharpe/maxDD/WR/PF/E → Markdown.
- **A2** Dashboard grafikleri — Lightweight Charts + `/api/equity` `/api/trades` `/api/bars`.
- **A3** VaR/Monte Carlo — `safety.py` portföy-VaR (%95/%99).

## FAZ B — Ücretsiz API anahtarı
- **B1** FRED makro risk-off — `macro_calendar.py` + `risk_manager._current_risk_pct`'e
  çarpımsal makro-çarpan (A4 rejim deseni). `MACRO_GATE=false` default. Gerekli: ücretsiz FRED key.
- **B2** Telegram `/grafik AAPL` — `telegram_bot.py` komut handler + matplotlib render.

## FAZ C — Yeni sinyal ADAYLARI (ölç, ship etme — §13.5 disiplini)
- **C1** Insider/13F (SEC EDGAR ücretsiz) — `insider.py` + `research/insider_test.py`
  (2 periyot + al-tut beta + pit_universe survivorship testi).
- **C2** Reddit sentiment — `sentiment.py` (PRAW+VADER) + `research/sentiment_test.py`.
  (X/Twitter atlandı: API ~$100+/ay.) Gerekli: ücretsiz Reddit app credential.

## FAZ D — Üstüne oturan katman
- **D1** SHAP/Explainable AI kartı — `meta_features.py`/`train_meta.py` + dashboard paneli.
- **D2** Docker + cloud deploy — `Dockerfile`/`docker-compose.yml`; `PREDICT_TRACK=true` ile
  prediction_tracker 7/24 → ücretsiz canlı paper kanıt.

## Sıra
```
A0 → A1 → A2 → A3  |  B1 → B2  |  D2 (erken deploy)  |  C1 → C2  |  D1
```

## Senden gereken ücretsiz hesaplar
- FRED API key (B1) — fred.stlouisfed.org
- Reddit app credential (C2)
- Diğer her şey hesapsız.

---
**Durum:** A0 + A1 + A2 tamam (2026-06-22). A2 notu: mum grafiği + stop/hedef/TD
marker overlay'leri (`/api/bars`,`/api/analyze`) zaten mevcuttu; A2'de eklenen tek
boşluk = temiz bakiye kaynağı `/api/equity` (data/equity_curve.csv) + grafiğin bunu
tercih edip trade_log parse'ına fallback yapması.

A3 tamam: `safety.py`'ye portföy VaR/CVaR eklendi — `historical_var` (tarihsel simülasyon),
`monte_carlo_var` (çok-değişkenli normal, korelasyon korunur), `compute_portfolio_var`
(açık pozisyonların dayanak getirilerinden, delta-doğrusal opsiyon yaklaşığı). 8 test yeşil
(`tests/test_safety_var.py`). Salt RAPORLAMA — boyut/emir kısmaz, alfa iddiası yok.
NOT: hesap katmanı hazır ama henüz UI'ya bağlı değil (opsiyonel `/api/var` + dashboard kartı
takip işi).

B1 iskeleti tamam (FRED makro risk-off): `config.MacroConfig` (`MACRO_GATE` default false,
`FRED_API_KEY`), `nihai/data/macro_calendar.py` (`compute_macro_multiplier` saf eşik mantığı +
`get_macro_snapshot` önbellekli FRED çekimi; VIXCLS/T10Y2Y/BAMLH0A0HYM2 → risk-off oyu),
`risk_manager._current_risk_pct`'e çarpımsal makro-çarpan + `refresh_macro()`. 6 test yeşil
(`tests/test_macro_calendar.py`). KAPALI/keysiz/ağ-hatası → çarpan 1.0 (sistem değişmez).
**Senden gereken:** ücretsiz FRED key (fred.stlouisfed.org) → `.env`'e `FRED_API_KEY=...`,
sonra `MACRO_GATE=true`. KALAN HOOKUP: canlı döngüde periyodik `rmgr.refresh_macro()` çağrısı
(main loop'a 1 satır — key gelince eklenecek).

B2 tamam (Telegram /grafik): `requirements.txt`'e matplotlib eklendi (3.11 kuruldu),
`nihai/notifier/chart_render.py` (headless Agg → kapanış+EMA20/50+hacim PNG), `telegram_bot.py`'ye
`send_photo` (sendPhoto multipart, fire-and-forget), `main.py` tg_handlers'a `/grafik` + `/chart`
+ help güncellendi. 6 test yeşil (`tests/test_chart_render.py`). Token yoksa no-op, veri yoksa
kullanıcıya metin uyarısı.

D2 tamam (Docker + 7/24): `Dockerfile` (python:3.12-slim, headless Agg, healthcheck `/api/status`,
`CMD --serve-web`), `.dockerignore` (venv/data/logs/.env/cache hariç), `docker-compose.yml`
(ana servis `PREDICT_TRACK=true` + `WEB_HOST=0.0.0.0` + kalıcı `./data`+`./logs` volume +
restart=unless-stopped; profile arkasında saatlik `resolver` servisi = prediction_tracker
--resolve --report döngüsü). `docker compose config` → VALID. KALAN: gerçek `docker build`
(kullanıcı çalıştırır) + `.env.example`'a PREDICT_TRACK/FRED_API_KEY/MACRO_GATE dokümante etme.
Kullanım: `docker compose up -d` → pano http://localhost:8765.

C1 tamam (SEC EDGAR insider/Form 4 ADAY sinyali — ÖLÇ, SHIP ETME): `nihai/data/insider.py`
(ücretsiz EDGAR: ticker→CIK `company_tickers.json`, CIK→tüm Form 4 `submissions` + arşiv
parçaları, her Form 4 ham XML'i ayrıştırıp açık-piyasa P/S işlemlerini FİLİNG tarihine göre
toplar; `data/insider_cache/<SYM>.json` disk cache + `insider_signal` saf küme-alım kararı,
varsayılan LONG-only). `nihai/signals/strategy.py`'ye `detect_insider` + dispatcher + STRATEGY_MODES
'insider'. `research/insider_test.py` (BASKET 2-periyot + al-tut beta + kıyas, `--pit` ile
pit_universe survivorship: TÜM/hayatta-kalan/removed-since). Look-ahead güvenli (giriş = filing
ertesi barı; bilgi ancak dosyalanınca halka açık). Canlı EDGAR doğrulandı: AAPL CIK=320193,
1337 Form 4, ham XML parse → P/S hisse/yön doğru çıkıyor (AAPL mega-cap → yalnız satış, beklenen).
11 strateji testi yeşil (regresyon yok). Yeni bağımlılık YOK (requests=yfinance transitive + stdlib).
**Senden gereken:** ilk koşu yavaş (EDGAR'dan sembol başına tüm Form 4 XML; cache'lenir) +
.env'e isteğe bağlı `SEC_USER_AGENT="Ad eposta"`. KALAN: gerçek ölçüm koşusu (`python -m research.insider_test`)
→ PEAD §16 dersi: survivorship deflasyonu beklenir, kenar çıksa bile `--pit` ile teyit et.

C2 tamam (Reddit sentiment ADAY sinyali — ÖLÇ, SHIP ETME): `nihai/data/sentiment.py`
(PRAW + VADER: finans subredditlerinde sembolü anan son gönderiler → VADER compound
upvote-ağırlıklı ort. = SentimentSnapshot; `update_sentiment` İLERİYE-BİRİKİMLİ olarak
`data/sentiment_history/<SYM>.csv`'ye yazar (iv_history deseni); `sentiment_signal` saf karar:
boğa-zirvesi+hacim→LONG, ayı→SHORT, `allow_contrarian` ile fade). `nihai/signals/strategy.py`'ye
`detect_sentiment` + dispatcher + STRATEGY_MODES 'sentiment'. `research/sentiment_test.py`
(BASKET 2-periyot + al-tut beta + `--pit` survivorship + `--coverage`). requirements.txt'e
praw>=7.7 + vaderSentiment>=3.3 (kuruldu). 22 test yeşil; offline mantık + dispatcher + soğuk
başlangıç coverage (0/16, beklenen) doğrulandı; credential yoksa REDDIT_DISABLED (sessiz nötr).
**KRİTİK SINIR:** Reddit'in ücretsiz TARİHSEL API'si YOK (Pushshift kapalı) → gerçek tarihsel
backtest YAPILAMAZ; geçmiş yalnız ileriye birikir. Anlamlı ölçüm ancak D2 (7/24 deploy) +
periyodik `update_sentiment()` ile haftalar/aylar geçmiş biriktikten sonra. **Senden gereken:**
ücretsiz Reddit 'script' app (reddit.com/prefs/apps) → .env'e REDDIT_CLIENT_ID/SECRET.
Kalan iş: D1 (SHAP kartı) — FAZ C bitti. NOT: sentiment'i canlı biriktirmek için D2 döngüsüne
periyodik `update_sentiment(sym)` çağrısı eklenmeli (geçmiş üretiminin tek yolu).

D1 tamam (SHAP/Explainable AI kartı): `nihai/signals/meta_gate.py`'ye `_MetaGate.explain()` —
lojistik meta-model için TAM özellik-katkısı (standardize özellik × ağırlık = log-odds katkısı;
lineer modelde bu tam SHAP değeridir, baz = sabit terim). Additivite kanıtlandı: logit(prob) =
baz + Σ(tüm katkılar) (eşleşme <1e-2). GBM yolu: shap kuruluysa TreeExplainer, değilse global
feature_importances_ (bayraklı) — ama aktif model lojistik. `web_dashboard.py /api/analyze`
`signal.meta_explain` alanı (yeni, additive) + `web_dashboard.html`'e "P(win) Açıklaması" paneli
(yeşil/kırmızı yatay katkı barları, baz→tahmin, ⚠️ OOS-AUC ~0.5-0.6 uyarısı). 18 test yeşil
(web + strateji). Yeni bağımlılık YOK (shap opsiyonel; lojistik yol saf numpy). DÜRÜSTLÜK: kart
'modelin ne düşündüğünü' açıklar, GELECEĞİ değil — kenar iddiası taşımaz (§16).

═══ PLAN TAMAMLANDI: FAZ A (A0-A3) + B (B1-B2) + C (C1-C2) + D (D1-D2) hepsi bitti. ═══
Kalan opsiyonel hookup'lar (kod hazır, tek-satır canlı bağlama): (1) main loop'a periyodik
`rmgr.refresh_macro()` [B1], (2) D2 döngüsüne periyodik `update_sentiment(sym)` [C2 geçmiş üretimi],
(3) `/api/var` + dashboard VaR kartı [A3 UI]. Hepsi ücretsiz hesap/anahtar bekliyor (FRED, Reddit).
