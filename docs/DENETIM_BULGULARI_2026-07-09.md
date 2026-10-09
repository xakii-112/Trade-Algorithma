# Denetim Bulguları — 2026-07-09 (4 paralel Opus 4.8 agent turu)

Baseline: nihai 455+1skip / MainHub 27 yeşil, ruff+mypy temiz. 4 salt-okunur agent
(correctness / ML / execution / performans) örtüşmeyen alanları denetledi. Durum: BEKLİYOR = uygulanacak.

> DURUM (2026-07-10): **TAMAMLANDI.** 3 paralel Opus Fix agent (data-infra / ML / execution) tüm Tier-1/2 +
> seçili Tier-3'ü uyguladı; entegrasyon tek-thread **YEŞİL** (nihai tam suite exit=0 ~520 test, mypy 56 dosya
> 0 hata, ruff iki projede temiz, MainHub 27). Kalıntı/kapsam-dışı: B5 (train_meta bug değildi), B7
> (`_enrich_row` finite-guard = takip), D4/D9 web-perf ertelendi. Detay: `OPTIMIZASYON_LOG.md → Ek Tur 9`.
>
> TUR-10 (2026-07-10 akşam, Fable 5): kalan perf kalemleri **D2/D3/D4/D8/D9/D10 UYGULANDI** + **B7**
> `_enrich_row` kaynak finite-guard eklendi — hepsi bit-exact, kalıcı parite testleriyle
> (`tests/test_perf_parity_round10.py`, +8 test). Suite ~528 yeşil · mypy 56/0 · ruff temiz.
> Gövdedeki madde-başı DURUM etiketleri Tur-9 ÖNCESİ anlık görüntüdür; geçerli durum = bu not +
> `OPTIMIZASYON_LOG → Ek Tur 9/10`. Açık kalan: yalnız "DÜRÜST İYİLEŞTİRMELER" backlog'u (kısmen:
> Wilson-CI Tur-9'da girdi) — bkz. `GELISTIRME_RAPORU_2026-07-10.md` §3.
>
> TUR-11 (2026-07-13, Fable 5): GELİŞTİRME RAPORU'nun 4 maddesi bir arada işlendi — 2 YENİ paralel
> salt-okunur denetim (ML/veri katmanı 7 bulgu + ops/web/MainHub 8 bulgu, `research/`, `prediction_tracker`,
> tüm ps1'ler, MainHub parite dahil) + 2 paralel uygulama agent'ı (Fix-CORE: META_GATE **shadow** kipi +
> `daily_summary` günlük özet + VaR dashboard kartı; Fix-ML: **DÜRÜST İYİLEŞTİRMELER (P3/FAZ AI-3) TAMAMEN
> KAPANDI** — eşik seçimi TEST'ten TRAIN'e taşındı, purged_cv null-AUC tabanı + sızıntı bayrağı,
> save_model provenance, `_purged_gate` artık terfi eşiğinde CV'yi sınıyor, `suggest_purge` yerel-yoğunluk
> düzeltmesi, `entry_time` datetime-doğru sıralama). Kalan bulguları (alpaca_history boş-cache kapsamı,
> prediction_tracker dedup kanonikleştirme, stop-hub/hub-watchdog SYSTEM-görev+asılı-süreç düzeltmeleri,
> MainHub all_expiries paritesi, update-config atomikliği, forward_harness 3×→1× okuma, BOM eksikleri) ben
> doğrudan uyguladım. **CANLI OLAY:** entegrasyon sırasında 2026-07-11 02:04 forward-harness çöküşü tespit
> edildi (OneDrive dosya-kilidi → `os.replace` `PermissionError` → tüm günün resolve/snapshot'ı kayboldu,
> AI-0 "3 gün üst üste" kabulü bu yüzden hâlâ karşılanmadı) → `_atomic_replace` (kısa üstel geri-çekilmeli
> yeniden deneme) + `daily_summary`'nin harness başarısız olsa BİLE çalışması (forward-harness.ps1 sıralama
> düzeltmesi) eklendi. İki LLM sinyal-adayı ölçümü (`llm_event_test --pit`) denendi: Alpaca SIP fiyat
> tarafı `DATA_PROVIDER=alpaca_hist` ile çalıştı ama Gemini ÜCRETSİZ katman (günde ~20 istek) PIT taramasını
> tüketemeden 429'a düştü → ölçüm TAMAMLANAMADI (kod sorunu değil, kota; `llm_event_test.py` docstring'ine
> saha notu eklendi). Entegrasyon: nihai tam suite **582 test, exit=0** · mypy **57 dosya 0 hata** ·
> ruff nihai+research+tests **temiz** · MainHub **27 test yeşil** + ruff temiz. Detay:
> `OPTIMIZASYON_LOG.md → Ek Tur 11`, `GELISTIRME_RAPORU_2026-07-10.md` §Ek Tur-11.

> TUR-13 (2026-07-17, Fable 5): 3 paralel Opus agent (AI-katmanı kod + dürüst-ML kod +
> Tur-12 değişiklik denetimi) OTURUM LİMİTİNE takılıp yarım kaldı; F5 kalan işi tamamladı
> (nihai/ai/ + deflated.py + daily_summary/forward_harness bağlama + 22 test). Denetim
> agent'ının tamamlanan kısmı: strategy.py shadow-kipi TEMİZ (erken return yok, sinyal
> bastırılmıyor). Tur-12 değişikliklerinin tam denetimi SONRAKİ tura devredildi. Ops
> bulguları (yeni): görev OLUŞTURMA yükseltisiz reddediliyor (0x80070005; güncelleme
> çalışıyor) → watchdog kayıtları için `tur13-servis-devri.ps1` (**aynı gün 18:24'te UAC ile
> koşuldu; devir TAMAM** — eski SYSTEM görevleri gerçekten VARMIŞ ve yükseltisiz görünmüyormuş,
> Tur-12 körlük teşhisi kesinleşti); forward-harness 30dk limiti 07-16 koşusunu sessiz öldürmüştü
> (4sa + 12:00 tetik + yarım-koşu tespiti girdi). Detay: `OPTIMIZASYON_LOG.md → Ek Tur 13`.

## TIER 1 — YÜKSEK değer, CONFIRMED (önce bunlar)

- **A1** [YÜKSEK] `nihai/data/alpaca_stream_provider.py:294` + `MainHub/alpaca_stream_provider.py:296`
  — oluşan-bar atma kontrolü naive-LOCAL `datetime.now()`'ı naive-UTC ts ile kıyaslıyor →
  UTC+3'te yarı-dolu bar ATILMAZ = look-ahead (Madde 3). Fix: `datetime.now(UTC).replace(tzinfo=None)`.
  İki kopyaya BİREBİR (parity). **Durum: UYGULANIYOR**
- **B1** [YÜKSEK] `research/live_retrain.py:150-155` `assess()` durable kontrolünde `min-gate-keep` YOK
  → canlı `--promote` avuç-dolusu işlemle sahte kenar terfi edebilir (üretim modelini YAZAN yol).
  Fix: `min_gate_keep=50` param + guard (`best_gate[4]>=min_gate_keep` iki bölmede) + `--min-gate-keep` CLI.
  **Durum: UYGULANIYOR**
- **B2** [YÜKSEK] `research/purged_cv.py:69-71` `purged_folds` yalnız SAĞ embargo; SOL PURGE yok →
  etiket sızıntısı OOS-AUC'yi şişirir (gelecek terfileri sahte-kenardan korur). Fix: sol-taraf purge
  (entry_time+bars_held) veya simetrik embargo. **Durum: BEKLİYOR (daha karmaşık — plumbing gerekir)**
- **C1** [YÜKSEK/PLAUSIBLE] `nihai/execution/prediction_tracker.py:182-195,226` `_forward_price`/`resolve`
  ham `str(ts)` eşleşmesi; tz-aware kayıt vs tz-naive çözüm → kalıcı `pending`, SESSİZ (forward-kanıt
  döngüsü ölür, ACI DERS #5). Fix: `bar_ts` kanonik (UTC isoformat/epoch) yazımda+eşlemede; lookback param.
  **Durum: BEKLİYOR**

## TIER 2 — ORTA, CONFIRMED

- **A2** [ORTA] `nihai/data/hub_server.py:100` + `MainHub/hub_server.py:107` (/options) + `data_provider.py:436`
  (`HubProvider.get_option_chain`) — `all_expiries` hub sınırında düşürülüyor → `DATA_PROVIDER=hub`'ta
  opsiyon seçimi tamamen kırık (sessiz). Fix: `all_expiries`'i iki tarafta da geçir. **BEKLİYOR**
- **B3** [ORTA] `nihai/signals/meta_gate.py:37-45` + `research/train_meta.py:258-264` — logistic `save_model`
  bayat `.pkl`'i arşivlemiyor; `_load` .pkl'i .json'dan önce okuyor → yeni logistic model eski GBM'le
  gölgeleniyor. Fix: logistic kaydında `.pkl`'i `.pkl.bak-<t>`'ye arşivle (simetrik). **BEKLİYOR**
- **B4** [ORTA] `research/forward_harness.py:100-101` — `pwin` metrikleri YÖN-isabeti (`hit`) etiketiyle
  hesaplanıyor, oysa `pwin_side` sonucu gerek; SHORT adaylarda etiket TERS → dürüstlük panosu bozuk.
  Fix: etiket = `_side_return(row) > 0`. **BEKLİYOR**
- **B5** [ORTA] `research/live_retrain.py:201-207` + `train_meta.py:359-364` — terfi edilen tüm-veri modeli,
  alt-modelin eşiğiyle kaydediliyor (model↔eşik uyuşmazlığı). Fix: tek aday eşik, iki bölme+purged CV o
  eşikte doğrulansın; nihai model üstünde eşik yeniden türet. **BEKLİYOR**
- **B6** [ORTA] `research/postmortem.py:112-115` — `sharpe_trade = mean/std*sqrt(n)` bir Sharpe değil t-istat;
  n ile şişer, yanıltıcı. Fix: `sharpe=mean/std`; ayrı `t_stat` alanı. **UYGULANIYOR**
- **C2** [ORTA] `nihai/execution/main.py:280-290,403` — cooldown yalnız başarılı açılışta; risk-reject/dolu
  portföyde actionable sinyal her 60s Telegram spam. Fix: erken-return yollarında (symbol,bar_ts) alert-dedup. **BEKLİYOR**
- **C3** [ORTA] `nihai/execution/backtest.py:439-467` — giriş barı çıkış-kontrolünden atlanıyor → iyimser
  sapma, backtest≠canlı. Fix: giriş barını değerlendir VEYA konvansiyonu dokümante+parite-testiyle kilitle. **BEKLİYOR**
- **C4** [ORTA] `nihai/execution/main.py:410-434` `flatten_all_action` `rehydrate_trackers()` yok + `_on_close`
  defterini atlıyor (unsettled proceeds/postmortem). Fix: rehydrate + ortak kapanış-yolu. **BEKLİYOR**
- **D1** [YÜKSEK-perf] `nihai/execution/prediction_tracker.py:65-76` `_existing_keys` → `record()` her ~20s
  döngüde TÜM `predictions_log.csv`'yi yeniden okuyor (sınırsız büyür, _WRITE_LOCK altında). Fix: modül
  `_seen_keys` cache (bir kez seed + append; mtime fallback). **BEKLİYOR**
- **D2** [YÜKSEK-perf] `research/train_meta.py:45-53` + `forward_harness.py:56-70` `_auc` saf-Python O(N)
  döngü × 26/fold × 5 fold × 234k → ~6M iter. Fix: cumsum ile vektörize (bit-exact). **UYGULANDI (Tur-10)**
- **D3** [ORTA-perf] `research/purged_cv.py:82-84` permütasyon-önem her özellik için tam `X.copy()`. Fix:
  tek sütun yerinde permüte+geri-yükle (bit-exact). **UYGULANDI (Tur-10)**
- **D4** [ORTA-perf] `nihai/web/web_dashboard.py:394-432` — aynı CSV istek başına 2-3× okunuyor. Fix: bir
  kez oku, `rows=` param geçir. **UYGULANDI (Tur-10)**
- **D5** [ORTA-perf] `prediction_tracker.py:182-195` `_forward_price` satır başına str-index listesi+lineer
  arama. Fix: cache'li df başına `pos_map` bir kez (bit-exact). **BEKLİYOR**
- **D6** [ORTA/DÜŞÜK-perf] `backtest.py:195` — sembol başına provider (httpx client/session) yeniden kurulur
  (hub/alpaca/tv). Fix: modül provider cache. **BEKLİYOR**

## TIER 3 — DÜŞÜK / savunmacı

- **A3** [ORTA→pratik-DÜŞÜK] `nihai/data/pit_universe.py:71` `sp500_as_of` değişiklikleri geri-sararken
  azalan-tarih sıralamasını enforce etmiyor (scrape sırasına bağlı). Fix: `sort_values(date, desc)`. **UYGULANIYOR**
- **A4** [DÜŞÜK] `nihai/notifier/chart_render.py:55-91` hata yolunda `plt.close(fig)` yok → figür sızıntısı.
  Fix: try/finally close. **BEKLİYOR**
- **A5** [DÜŞÜK] `MainHub/config.py:557` `validate_config` ölü `from strategy import` (hep ImportError). Fix:
  kaldır/bilinçli-işaretle. **BEKLİYOR**
- **B7** [DÜŞÜK] `research/train_pead_ml.py:60-77` `prepare()` `ensure_feature_columns` yok + clip NaN korur →
  NaN model'e sızabilir / eski-şema KeyError. Fix: finite-guard + ensure_feature_columns + fillna.
  **UYGULANDI (Tur-9 prepare() + Tur-10 `_enrich_row` kaynak finite-guard)**
- **B8** [DÜŞÜK] `research/train_meta.py:182` `XGBClassifier(n_jobs=0)` → terfi belirlenimsiz. Fix: `n_jobs=1`. **BEKLİYOR**
- **B9** [DÜŞÜK] `research/build_live_dataset.py:60,65-66` — feats anahtar-kolon tipi uyuşmazlığı → sessiz
  satır kaybı. Fix: `dtype={"symbol":str,"bar_ts":str}`. **BEKLİYOR**
- **C5** [DÜŞÜK] `prediction_tracker.py:85-102,129-133` — boş CSV → başlıksız append (şema bozulması). Fix:
  boş/başlıksızı "fresh" say. **BEKLİYOR**
- **C6** [DÜŞÜK] `main.py:103-112,780-781` — `equity_curve.csv` sınırsız + dakikada özdeş nokta. Fix: değişince
  ekle / retention. **BEKLİYOR**
- **D7** [DÜŞÜK-perf] `auto_screener.py:348` — `build_universe` iki kez (4 wiki fetch/tur). Fix: memoize. **BEKLİYOR**
- **D8** [DÜŞÜK-perf] `research/agreement.py:158-165` iterrows asof-join → vektörize searchsorted
  (`_map_mom_pct`; merge_asof yerine — NaN kuralları birebir kanıtlanabilir). **UYGULANDI (Tur-10)**
- **D9** [DÜŞÜK-perf] `web_dashboard.py:554,572,588` iterrows payload → vektörize. **UYGULANDI (Tur-10)**
- **D10** [DÜŞÜK-perf] `train_meta.py:316` çift `to_numpy`. **UYGULANDI (Tur-10)**

## DÜRÜST İYİLEŞTİRMELER (kenar UYDURMADAN — gelecek turlar)
- Kapı uplift'ine binom/bootstrap %95 CI; `WEAK_POSITIVE`/`GATE_HELPS` yalnız CI alt-sınırı base üstündeyse.
- Eşik seçimini TEST'ten TRAIN/val'e taşı; denenen konfig sayısını raporla; deflated bar (çoklu-test).
- purged_cv'ye leakage-bayrağı (beklenmedik özellik ilk-2'de) + label-shuffle null-AUC tabanı.
- `save_model` provenance: code_sha, dataset_sha256, n_rows, seed, embargo, min_gate_keep, ECE.

## TEMİZ ALANLAR (denetlendi, sağlam)
extract_features leak-free · train/serve hizası · BULGU-11/12 düzeltmeleri yerinde · META_GATE default-KAPALI
degrade-safe · gate.py/watchlist/dual_tier/state/concurrency testleri kapsamlı · CachingProvider/alpaca_history
cache disiplini · rounds 6-7 O(n²)→O(n) kapatılmış (yeni mimari O(n²) yok).
