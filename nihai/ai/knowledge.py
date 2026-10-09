"""
knowledge.py — Quant/finans BİLGİ ÇEKİRDEĞİ (FAZ AI v2 "Quant Beyin", WP-C). KARAR-DIŞI.

Ne yapar: ≥40 quant/finans kavramını (rejim, momentum, kelly, VaR/CVaR, kointegrasyon,
Black-Scholes Greek'leri, meta-labeling, survivorship/look-ahead/exec-lag yanlılıkları vb.)
yapılandırılmış `Concept` kayıtları olarak tutar; her kavram TANIM + quant_math'teki GERÇEK
fonksiyon adlarına köprü (`formul_ref`) + bilinen TUZAKLAR + İLİŞKİLİ kavramlar + (varsa)
CLAUDE.md'deki ACI DERS bağlantısı (`ders_ref`, ör. "ACI-1") taşır.

Ne yapmaz: hiçbir hesap yapmaz, hiçbir dosyaya yazmaz, hiçbir sinyale/eşiğe/META_GATE'e
bağlanmaz — saf, statik bir bilgi tablosu + sorgu API'sidir (KARAR-DIŞI; anayasa md.1).

API:
  - explain(concept_id)     -> Türkçe açıklama bloğu (tanım+formül+tuzak+ders) veya None
  - formula_for(concept_id) -> quant_math fonksiyon adları (list[str]; VAR OLAN adlar)
  - search(query)           -> ad/tanım/id içinde küçük-harf alt-dizge eşleşen Concept'ler
  - glossary()               -> tüm kavram id'leri, alfabetik

`formul_ref` alanlarındaki HER isim FORMÜL MODÜLLERİNDEN birinde gerçekten var olmalıdır
(`quant_math` · `options_math` · `microstructure`; test bunu getattr ile doğrular) — uydurma
fonksiyon adı YOK. Tur-32'ye kadar tek adres `quant_math`'ti; opsiyon/gülümseme ve
gerçekleşen-oynaklık aileleri kendi modüllerine çıkınca köprü de genişletildi (kavramın
"uygulaması yok" demesi ile gerçekte olması arasındaki sapma bu yüzden mümkün değildir).
Metodolojik/yanlılık kavramlarının
(survivorship_bias, look_ahead_bias, exec_lag, pbo, purged_cv, meta_labeling, embargo, carry,
drift_detection, brier, beta_binomial, pead, earnings_drift, deflated_sharpe, bid_ask_bounce,
kointegrasyonun bazı ölçütleri, implied_vol/put_call_parity gibi TÜRETİLMİŞ kavramlar dahil)
bir kısmının doğrudan sayısal karşılığı yoktur; bu durumda `formul_ref=[]` (veya en yakın
temel formüle işaret) bilinçli bir seçimdir — kavram bir fonksiyon değil, bir DİSİPLİN/anomali
olabilir. `ders_ref` yalnızca CLAUDE.md'nin "ACI DERSLER" numaralı listesindeki 9 kayda
("ACI-1".."ACI-9") işaret eder; bu modül episodic.LESSONS'a bağımlı DEĞİLDİR (gevşek çapraz
referans — id string'i, import değil).

⚠️ Tur-48: bu satır "ACI-1..ACI-7" diyordu ve BAYATTI — belgede #8/#9 aylardır vardı ve
`knowledge.py` içinde `ders_ref=["ACI-9"]` ZATEN kullanılıyordu, yani kod kendi
docstring'iyle çelişiyordu (Tur-42: "bir yorumdaki anahtar adı da API'dir"). "Gevşek
çapraz referans" IMPORT bağımsızlığıdır, DENETİMSİZLİK değil: bağ artık
`tests/test_tur48_ders_katalogu.py` ile kilitli — çalışma zamanında import YOK, testte
VARLIK kontrolü VAR.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Concept:
    """Tek bir quant/finans kavramı.

    id           : kısa, snake_case tekil anahtar (CONCEPTS sözlüğünün anahtarıyla aynı).
    ad           : insan-okunur Türkçe başlık.
    tanim        : 1-2 cümlelik Türkçe tanım.
    formul_ref   : formül modüllerindeki (quant_math/options_math/microstructure)
                   İLGİLİ fonksiyon adları (yoksa boş liste).
    tuzaklar     : bilinen yanlış-kullanım/yanlış-yorumlama tuzakları (Türkçe, kısa madde).
    iliskili     : diğer CONCEPTS anahtarlarına işaret eden ilişkili-kavram id'leri.
    ders_ref     : CLAUDE.md ACI DERSLER numaralı-liste id'leri (ör. "ACI-1"); yoksa boş liste.
    prereqs      : (v3, WP-K) `curriculum()`'ın topolojik sıralaması için ÖNKOŞUL kavram
                   id'leri (CONCEPTS'te olmalı); yoksa boş liste = kök/temel kavram. Mevcut
                   (v2) 45 kavram bu alanı hiç KULLANMAZ (varsayılan boş liste ile geriye-
                   uyumlu) — yalnız v3'te eklenen kavramlar prereqs belirtir.
    """

    id: str
    ad: str
    tanim: str
    formul_ref: list[str] = field(default_factory=list)
    tuzaklar: list[str] = field(default_factory=list)
    iliskili: list[str] = field(default_factory=list)
    ders_ref: list[str] = field(default_factory=list)
    prereqs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Kavram tablosu — spec §5 zorunlu liste (45 kavram, ≥40 şartını rahatça karşılar).
# formul_ref'ler formül modüllerindeki (quant_math/options_math/microstructure) AYNEN
# adlardır (uydurma yok).
# ---------------------------------------------------------------------------
_CONCEPT_LIST: list[Concept] = [
    Concept(
        id="rejim",
        ad="Piyasa Rejimi",
        tanim=(
            "Fiyat serisinin trend/yatay, düşük/yüksek oynaklık ve ortalamaya-dönüş eğilimi "
            "gibi durumsal sınıflandırması; strateji seçimini bağlama oturtur."
        ),
        formul_ref=["classify_regime"],
        tuzaklar=[
            "Rejim etiketleri kural-tabanlıdır, geleceği garanti etmez — rejim değişimleri gecikmeli fark edilir.",
            "Az bar (<60) ile 'unknown' döner; kısa pencerede rejim iddiası gürültü olabilir.",
        ],
        iliskili=["momentum", "mean_reversion", "hurst"],
    ),
    Concept(
        id="momentum",
        ad="Momentum (Fiyat Momentumu)",
        tanim=(
            "Geçmiş getirinin yakın vadede aynı yönde sürme eğilimi; trend-takip "
            "stratejilerinin dayanağı."
        ),
        formul_ref=["simple_returns", "zscore"],
        tuzaklar=[
            "Momentum rejime duyarlıdır — yatay/mean-reverting piyasada ters teper.",
            "Survivorship-şişkin evrende sahte-pozitif kenar görülebilir (cross-sectional momentum bulgusu).",
        ],
        iliskili=["rejim", "mean_reversion"],
    ),
    Concept(
        id="mean_reversion",
        ad="Ortalamaya Dönüş (Mean Reversion)",
        tanim=(
            "Fiyat/spread'in uzun-vadeli ortalamasından saptığında geri dönme eğilimi; "
            "Hurst<0.5 ve OU süreciyle karakterize edilir."
        ),
        formul_ref=["hurst_exponent", "ou_half_life", "variance_ratio"],
        tuzaklar=[
            "Trend rejiminde mean-reversion stratejisi sürekli zarar eder (rejim filtresi olmadan kullanma).",
            "Half-life çok kısa/uzun tahmin edilirse pozisyon tutma süresi yanlış kalibre olur.",
        ],
        iliskili=["ou_process", "half_life", "hurst", "kointegrasyon"],
    ),
    Concept(
        id="pead",
        ad="PEAD (Post-Earnings Announcement Drift)",
        tanim=(
            "Kazanç açıklamasından sonra fiyatın sürpriz yönünde haftalarca sürmesi "
            "beklenen akademik anomali."
        ),
        formul_ref=[],
        tuzaklar=[
            "nasdaq100'de PF 1.57 iken point-in-time S&P500'de PF 1.05'e (başabaş) düştü — survivorship artefaktı riski yüksek.",
            "removed-since (endeksten düşmüş) isimlerde PF 0.80 — hayatta-kalan evrende ölçüm kenarı abartır.",
        ],
        iliskili=["earnings_drift", "survivorship_bias", "look_ahead_bias"],
        ders_ref=["ACI-1", "ACI-6"],
    ),
    Concept(
        id="earnings_drift",
        ad="Kazanç Sürprizi Sürüklenmesi",
        tanim=(
            "PEAD ile eşanlamlı; kazanç sürprizinin büyüklüğü ile sonraki getiri arasındaki "
            "ilişkiyi ifade eder."
        ),
        formul_ref=[],
        tuzaklar=[
            "Kesit-seçim ML modeli surprise yerine vix/ret gibi vekil değişkenlere kilitlenebilir (sızıntı değil ama kenar de değil).",
            "Tek-bölme PF yüksek görünse de purged 5-fold CV OOS-AUC 0.50-0.52 ile kenarsız çıkabilir.",
        ],
        iliskili=["pead", "meta_labeling"],
        ders_ref=["ACI-1", "ACI-6"],
    ),
    Concept(
        id="survivorship_bias",
        ad="Hayatta Kalma Yanlılığı",
        tanim=(
            "Bugünkü endeks üyeliğiyle geçmişi test etmenin, endeksten düşmüş/iflas etmiş "
            "sembolleri dışlayarak sonucu olduğundan iyi göstermesi."
        ),
        formul_ref=[],
        tuzaklar=[
            "nasdaq100 sabit-listesiyle test etmek point-in-time (PIT) S&P500'e göre kenarı şişirir.",
            "'removed-since' alt-kümesinde performans ayrı ölçülmezse yanlılık gizli kalır.",
        ],
        iliskili=["look_ahead_bias", "pead"],
        ders_ref=["ACI-1"],
    ),
    Concept(
        id="look_ahead_bias",
        ad="İleriye Bakma Yanlılığı",
        tanim=(
            "Karar anında henüz bilinmeyen (gelecekteki) verinin özellik/sinyal hesabına "
            "sızması."
        ),
        formul_ref=[],
        tuzaklar=[
            "df.iloc[:i+1] dışına taşan herhangi bir istatistik (örn. tüm-seri normalize) look-ahead üretir.",
            "Kurumsal aksiyon (split/temettü) düzeltmeleri geriye dönük uygulanırsa test-anında bilinmeyen bilgi sızabilir.",
        ],
        iliskili=["survivorship_bias", "exec_lag"],
    ),
    Concept(
        id="exec_lag",
        ad="Uygulama Gecikmesi (Execution Lag)",
        tanim=(
            "Sinyal barından fiili emre kadar geçen bar sayısı; kenarın gerçek mi yoksa "
            "sinyal-sonrası ilk-bar sıçraması mı olduğunu ayırt eden duyarlılık testi."
        ),
        formul_ref=[],
        tuzaklar=[
            "Tüm metrikler (PF, DSR, PBO, survivorship pariteli) yeşilken bile 1 bar gecikme eklenince kenar çökebilir (pairs örneği: PF 2.08 → 1.03).",
            "Exec-lag testi atlanırsa bid-ask sıçraması/1-günlük reversal, kointegrasyon dönüşüyle karıştırılabilir.",
        ],
        iliskili=["bid_ask_bounce", "kointegrasyon", "look_ahead_bias"],
        ders_ref=["ACI-7"],
    ),
    Concept(
        id="bid_ask_bounce",
        ad="Bid-Ask Sıçraması",
        tanim=(
            "Ardışık işlemlerin alış/satış tarafları arasında sekmesinden doğan yapay, "
            "gerçek bilgi taşımayan fiyat oynaklığı."
        ),
        formul_ref=[],
        tuzaklar=[
            "Kısa vadeli 'ters dönüş' kenarı sanılan hareket çoğu zaman mikro-yapısal sıçramadır, gerçek mean-reversion değildir.",
            "Exec-lag testiyle ayrıştırılmazsa kalıcı sanılıp canlıya alınabilir.",
        ],
        iliskili=["exec_lag", "mean_reversion"],
        ders_ref=["ACI-7"],
    ),
    Concept(
        id="kelly",
        ad="Kelly Kriteri",
        tanim=(
            "Uzun-vadeli logaritmik servet büyümesini maksimize eden optimal bahis/pozisyon "
            "oranı."
        ),
        formul_ref=["kelly_binary", "kelly_continuous", "fractional_kelly"],
        tuzaklar=[
            "Tam Kelly gerçek dünyada aşırı oynaktır — kesirli Kelly (fraction≈0.25) ve tavan (cap) olmadan kullanılmamalı.",
            "p_win/win_loss_ratio tahmini yanlışsa (overfit model) Kelly boyutu da yanlış büyür — girdi kalitesi kritik.",
        ],
        iliskili=["position_sizing", "var"],
    ),
    Concept(
        id="var",
        ad="VaR (Value at Risk)",
        tanim=(
            "Belirli bir güven düzeyinde (örn. %95) belirli bir ufukta beklenen en kötü "
            "kayıp büyüklüğü."
        ),
        formul_ref=["var_historical", "var_cornish_fisher"],
        tuzaklar=[
            "VaR, eşiğin ÖTESİNDEKİ kuyruk kaybının büyüklüğü hakkında hiçbir şey söylemez (bkz. CVaR).",
            "Tarihsel VaR, örneklem dönemi düşük-vol rejimindeyse kriz riskini hafife alır.",
        ],
        iliskili=["cvar", "max_drawdown"],
    ),
    Concept(
        id="cvar",
        ad="CVaR (Conditional VaR / Expected Shortfall)",
        tanim=(
            "VaR eşiğini aşan senaryolardaki ORTALAMA kayıp; kuyruk riskini VaR'dan daha "
            "dürüst yakalar."
        ),
        formul_ref=["cvar_historical"],
        tuzaklar=[
            "Az örneklemde (ince kuyruk verisi) CVaR tahmini yüksek varyanslıdır.",
            "Parametrik olmayan (tarihsel) CVaR, örneklem dışı bir kuyruk olayını öngöremez.",
        ],
        iliskili=["var", "max_drawdown"],
    ),
    Concept(
        id="sharpe",
        ad="Sharpe Oranı",
        tanim="Risksiz orana göre fazla getirinin oynaklık başına ölçüsü (yıllıklandırılmış).",
        formul_ref=["sharpe", "annualize_return", "annualize_vol"],
        tuzaklar=[
            "Getiri dağılımının normal/simetrik olduğunu zımnen varsayar — kalın kuyruklu/çarpık stratejilerde yanıltıcıdır (bkz. Sortino).",
            "Çok sayıda strateji denenip en iyi Sharpe seçilirse (seçim yanlılığı) gerçek OOS Sharpe'tan yüksek çıkar (bkz. deflated_sharpe, pbo).",
        ],
        iliskili=["sortino", "deflated_sharpe", "pbo"],
    ),
    Concept(
        id="sortino",
        ad="Sortino Oranı",
        tanim="Sharpe'ın yalnızca AŞAĞI yönlü oynaklığı (downside deviation) cezalandıran versiyonu.",
        formul_ref=["sortino", "downside_deviation"],
        tuzaklar=[
            "mar (minimum acceptable return) seçimi keyfi olabilir — farklı mar'lar sıralamayı değiştirebilir.",
            "Az işlemde downside deviation tahmini gürültülüdür.",
        ],
        iliskili=["sharpe", "max_drawdown"],
    ),
    Concept(
        id="deflated_sharpe",
        ad="Deflated Sharpe Ratio (DSR)",
        tanim=(
            "Çoklu deneme/seçim yanlılığını düzelterek gözlenen Sharpe'ın şans eseri olma "
            "olasılığını hesaba katan düzeltilmiş metrik."
        ),
        formul_ref=["sharpe"],
        tuzaklar=[
            "quant_math'te ayrı bir DSR fonksiyonu yoktur; sharpe çıktısı üzerine deneme-sayısı/varyans düzeltmesi UYGULANARAK hesaplanır (bkz. research/pairs_test.py PBO/DSR kullanımı).",
            "Deneme sayısı (kaç strateji/parametre denendiği) doğru sayılmazsa DSR yanlış-iyimser çıkar.",
            "SINIF DENGESİZLİĞİ DSR'yi ŞİŞİRİR (ACI DERS #8): sözde-getiri SIFIRA karşı "
            "değil TRİVİYAL kurala ('hep çoğunluğu söyle' / 'kapıyı hiç uygulama') karşı "
            "FAZLALIK olarak ölçülmelidir; aksi halde OOS-AUC 0.500 iken DSR 1.0 çıkar.",
        ],
        # Tur-48: bu kavram ACI DERS #8'in TA KENDİSİ hakkında ama `ders_ref` BOŞTU —
        # ders kataloğuna #8/#9'un hiç girmemiş olmasının ikinci katmanı.
        ders_ref=["ACI-8"],
        iliskili=["sharpe", "pbo"],
    ),
    Concept(
        id="pbo",
        ad="PBO (Probability of Backtest Overfitting)",
        tanim=(
            "Kombinatoryal saf-dışı (CSCV) yeniden örnekleme ile bir stratejinin OOS'ta en "
            "iyi olma sırasını kaybetme olasılığı."
        ),
        formul_ref=[],
        tuzaklar=[
            "Düşük PBO tek başına kenar KANITI değildir — survivorship/exec-lag gibi başka artefaktları YAKALAMAZ (PBO 0.004 iken exec-lag testinde kenar PF 2.08→1.03'e çöktü).",
            "Az sayıda dönemle (kısa CSCV bölme sayısı) PBO tahmini gürültülü olur.",
        ],
        iliskili=["deflated_sharpe", "purged_cv", "exec_lag"],
        ders_ref=["ACI-7"],
    ),
    Concept(
        id="purged_cv",
        ad="Purged Cross-Validation",
        tanim=(
            "Eğitim ve test pencereleri arasında örtüşen etiket ufkunu (label horizon) "
            "budayarak sızıntıyı önleyen K-fold çapraz doğrulama (López de Prado)."
        ),
        formul_ref=[],
        tuzaklar=[
            "Budama (purge) uygulanmadan yapılan standart K-fold, komşu foldlar arası bilgi sızıntısıyla OOS-AUC'u yapay şişirir.",
            "Tek bölmeyle (60/40) yetinmek şanslı-bölme kırılganlığına açıktır — K bağımsız pencere gerekir.",
        ],
        iliskili=["embargo", "meta_labeling"],
        ders_ref=["ACI-3"],
    ),
    Concept(
        id="embargo",
        ad="Embargo (Purged CV)",
        tanim=(
            "Purged CV'de test penceresinden hemen sonraki ek bir tampon bölgeyi de "
            "eğitimden çıkarma; otokorelasyon nedeniyle kalan sızıntıyı keser."
        ),
        formul_ref=[],
        tuzaklar=[
            "Embargo simetrik uygulanmazsa (yalnız sol taraf budanır, sağ taraf değil) sızıntı yarı-kapatılmış olur.",
            "Embargo penceresi çok küçükse otokorelasyonlu etiketlerde sızıntı kalıntısı sürer.",
        ],
        iliskili=["purged_cv"],
        ders_ref=["ACI-3"],
    ),
    Concept(
        id="meta_labeling",
        ad="Meta-Labeling",
        tanim=(
            "Birincil modelin ürettiği YÖN sinyaline ikincil bir modelin 'bu işlem kazanır "
            "mı' olasılığını ekleyip düşük-olasılıklıları elediği iki-aşamalı mimari "
            "(López de Prado)."
        ),
        formul_ref=[],
        tuzaklar=[
            "Survivorship-temiz PIT S&P500'de OOS-AUC 0.51-0.52 (yazı-tura) ölçüldü — nasdaq100'deki görünen kenar (PF 1.07-1.38) artefakttı.",
            "AUC iyi görünse bile KAPI SİMÜLASYONU (hem time hem symbol bölmesinde expectancy/PF artışı) doğrulanmadan üretime alınmamalı.",
        ],
        iliskili=["purged_cv", "pead", "kelly"],
        ders_ref=["ACI-2", "ACI-3", "ACI-4"],
    ),
    Concept(
        id="kointegrasyon",
        ad="Kointegrasyon",
        tanim=(
            "İki (veya daha çok) fiyat serisinin ayrı ayrı durağan olmayıp doğrusal "
            "bileşiminin durağan (mean-reverting) olması; pairs-trading'in matematiksel "
            "temeli."
        ),
        formul_ref=["engle_granger", "kalman_beta"],
        tuzaklar=[
            "Statik beta (Engle-Granger) rejim değişiminde kayabilir — Kalman ile dinamik beta izlemek gerekir.",
            "Kointegre GÖRÜNEN spread'in mean-reversion'ı aslında exec-lag artefaktı (bid-ask sıçraması) olabilir — 1 bar gecikme testi ZORUNLU.",
        ],
        iliskili=["mean_reversion", "ou_process", "exec_lag"],
        ders_ref=["ACI-7"],
    ),
    Concept(
        id="hurst",
        ad="Hurst Üsteli",
        tanim=(
            "Bir serinin trend-sürekliliği (H>0.5), rastgele yürüyüş (H≈0.5) veya "
            "ortalamaya-dönüş (H<0.5) eğilimini ölçen kendine-benzerlik istatistiği."
        ),
        formul_ref=["hurst_exponent"],
        tuzaklar=[
            "Kısa seride (az max_lag/az bar) Hurst tahmini yüksek varyanslıdır, güvenilmez.",
            "H<0.5 bulmak tek başına ticaret edilebilir kenar DEĞİLDİR — maliyet/exec-lag sonrası test edilmeli.",
        ],
        iliskili=["mean_reversion", "ou_process"],
    ),
    Concept(
        id="ou_process",
        ad="Ornstein-Uhlenbeck (OU) Süreci",
        tanim=(
            "Ortalamaya-dönüş hızı sabit bir stokastik süreç; spread/mean-reversion "
            "modellemesinde half-life tahmininin temeli."
        ),
        formul_ref=["ou_half_life"],
        tuzaklar=[
            "OU parametreleri AR(1) regresyonundan tahmin edilir — rejim değişince (half-life kayarsa) model yanlış kalibre kalabilir.",
            "Gerçek fiyat serileri saf OU değildir (sıçrama/kalın kuyruk); model basitleştirmesi risklidir.",
        ],
        iliskili=["mean_reversion", "half_life", "kointegrasyon"],
    ),
    Concept(
        id="half_life",
        ad="Yarı-Ömür (Half-Life)",
        tanim=(
            "Bir sapmanın ortalamaya-dönüş yoluyla başlangıç büyüklüğünün yarısına inmesi "
            "için beklenen bar sayısı; OU AR(1) katsayısından türetilir."
        ),
        formul_ref=["ou_half_life"],
        tuzaklar=[
            "Negatif/pozitif-sonsuz half-life (AR katsayısı ≥1) mean-reversion YOK anlamına gelir — bu durumda strateji uygulanmamalı.",
            "Half-life tutma-süresi olarak kullanılırken exec-lag/maliyet payı düşülmeden ham alınırsa optimist görünür.",
        ],
        iliskili=["ou_process", "mean_reversion"],
    ),
    Concept(
        id="parkinson_vol",
        ad="Parkinson Oynaklığı",
        tanim=(
            "Yalnızca günlük yüksek/düşük (high/low) aralığını kullanan, kapanış-kapanış'tan "
            "daha verimli bir gerçekleşen oynaklık tahmincisi."
        ),
        formul_ref=["vol_parkinson"],
        tuzaklar=[
            "Gün-içi sıçrama/halt gibi anormal high-low aralıkları tahmini şişirir.",
            "Trend (drift) içeren serilerde sistematik olarak oynaklığı bir miktar hafife alır (drift varsayımı sıfırdır).",
        ],
        iliskili=["garman_klass_vol", "yang_zhang_vol"],
    ),
    Concept(
        id="garman_klass_vol",
        ad="Garman-Klass Oynaklığı",
        tanim=(
            "OHLC (açılış-yüksek-düşük-kapanış) verisinin tamamını kullanan, Parkinson'dan "
            "daha verimli gerçekleşen oynaklık tahmincisi."
        ),
        formul_ref=["vol_garman_klass"],
        tuzaklar=[
            "Gece açılış sıçramalarını (overnight jump) hesaba katmaz — Yang-Zhang bu eksiği kapatır.",
            "Sentetik/eksik OHLC verisinde (örn. ayarlanmamış split) tahmin bozulabilir.",
        ],
        iliskili=["parkinson_vol", "yang_zhang_vol"],
    ),
    Concept(
        id="yang_zhang_vol",
        ad="Yang-Zhang Oynaklığı",
        tanim=(
            "Gece (overnight) ve gün-içi bileşenleri ayrı modelleyip birleştiren, drift'ten "
            "bağımsız en verimli OHLC oynaklık tahmincilerinden biri."
        ),
        formul_ref=["vol_yang_zhang"],
        tuzaklar=[
            "Diğer OHLC tahmincilerine göre daha karmaşık — bileşen ağırlıklandırması hatalı uygulanırsa tutarsız sonuç verebilir.",
            "Kısa pencerede (az bar) bileşenler arası kovaryans tahmini gürültülü olur.",
        ],
        iliskili=["parkinson_vol", "garman_klass_vol", "ewma_vol"],
    ),
    Concept(
        id="ewma_vol",
        ad="EWMA Oynaklık",
        tanim=(
            "Üstel ağırlıklı hareketli ortalama ile güncel bara daha çok ağırlık veren, "
            "rejim değişimine hızlı tepki veren oynaklık tahmincisi (RiskMetrics λ=0.94 "
            "konvansiyonu)."
        ),
        formul_ref=["ewma_vol"],
        tuzaklar=[
            "λ (lam) küçüldükçe tahmin daha gürültülü/hızlı tepkili olur — sabit varsayılan her rejimde optimal olmayabilir.",
            "Ani sıçramadan (tek aykırı bar) sonra tahmin bir süre yapay yüksek kalır (üstel sönüm gecikmesi).",
        ],
        iliskili=["parkinson_vol", "kalman_filter"],
    ),
    Concept(
        id="kalman_filter",
        ad="Kalman Filtresi",
        tanim=(
            "Gözlem gürültüsü altında zamanla değişen (dinamik) bir durumu (örn. "
            "pairs-trading beta'sı) rekürsif olarak tahmin eden optimal doğrusal filtre."
        ),
        formul_ref=["kalman_beta"],
        tuzaklar=[
            "Süreç/gözlem gürültü parametreleri (q, r) yanlış kalibre edilirse filtre ya çok yavaş ya da çok gürültülü izler.",
            "Dinamik beta 'kointegrasyon' izlenimi verse de altta yatan ilişki rejimle birlikte tamamen kırılabilir.",
        ],
        iliskili=["kointegrasyon", "ewma_vol"],
    ),
    Concept(
        id="ledoit_wolf",
        ad="Ledoit-Wolf Küçültme (Shrinkage)",
        tanim=(
            "Örneklem kovaryans matrisini yapısal bir hedefe doğru küçülterek (shrinkage) "
            "az-örneklemde kararlılığını artıran tahmin yöntemi."
        ),
        formul_ref=["ledoit_wolf_cov"],
        tuzaklar=[
            "Varlık sayısı örneklem uzunluğuna göre büyükse (n<<p) ham örneklem kovaryansı tekil/kararsızdır — shrinkage bunu düzeltir ama sinyali de bir miktar bulanıklaştırır.",
            "Shrinkage yoğunluğu veri-güdümlü seçilir; rejim değişiminde geçmiş dönem ağırlığı yanlış kalabilir.",
        ],
        iliskili=["pca_risk"],
    ),
    Concept(
        id="pca_risk",
        ad="PCA Risk Ayrıştırması",
        tanim=(
            "Getiri kovaryans matrisini temel bileşenlere ayırarak portföy riskinin kaç "
            "'bağımsız' faktörden geldiğini gösteren boyut-indirgeme tekniği."
        ),
        formul_ref=["pca_risk", "ledoit_wolf_cov"],
        tuzaklar=[
            "İlk bileşen(ler) çoğu zaman 'piyasa faktörü'dür — düşük açıklanan-varyans sanıp gerçek çeşitlendirmeyi abartmak riskli.",
            "Kovaryans tahmini gürültülüyse (Ledoit-Wolf uygulanmadan ham örneklem) PCA yükleri de gürültülü çıkar.",
        ],
        iliskili=["ledoit_wolf"],
    ),
    Concept(
        id="monte_carlo",
        ad="Monte Carlo Simülasyonu",
        tanim=(
            "Rastgele örneklenmiş çok sayıda yol (GBM veya blok-bootstrap) üreterek "
            "getiri/ruin dağılımını ampirik tahmin etme yöntemi."
        ),
        formul_ref=["mc_gbm_paths", "mc_bootstrap_terminal"],
        tuzaklar=[
            "GBM varsayımı (log-normal, sabit drift/vol) kalın-kuyruk ve rejim değişimini yakalamaz — blok-bootstrap gerçek veriye daha sadıktır.",
            "seed sabitlenmezse sonuçlar koşudan koşuya değişir, karşılaştırma/regresyon testi imkansızlaşır.",
        ],
        iliskili=["block_bootstrap"],
    ),
    Concept(
        id="block_bootstrap",
        ad="Blok Bootstrap",
        tanim=(
            "Getiri serisinden ARDIŞIK bloklar halinde (tek tek bar değil) yeniden "
            "örnekleyerek otokorelasyonu koruyan bootstrap varyantı."
        ),
        formul_ref=["mc_bootstrap_terminal"],
        tuzaklar=[
            "Blok uzunluğu çok kısaysa otokorelasyon yapısı bozulur (adi bootstrap'e yaklaşır); çok uzunsa etkin örneklem küçülür.",
            "Getiri serisi rejim-değişkense (örn. vol şoku dönemi) blok örnekleme o rejimi olduğundan fazla/az temsil edebilir.",
        ],
        iliskili=["monte_carlo"],
    ),
    Concept(
        id="black_scholes",
        ad="Black-Scholes Modeli",
        tanim=(
            "Avrupa tipi opsiyonlar için kapalı-form fiyatlama modeli; sabit oynaklık/faiz "
            "ve log-normal fiyat varsayar."
        ),
        formul_ref=["bs_price", "bs_greeks"],
        tuzaklar=[
            "Sabit IV varsayımı gerçek piyasada geçerli değildir (volatilite gülümsemesi/çarpıklığı) — ATM dışı strike'larda fiyat sapabilir.",
            "Amerikan tipi (erken kullanım) opsiyonlarda model sistematik olarak hafife alır.",
        ],
        iliskili=["delta", "gamma", "vega", "theta", "implied_vol", "put_call_parity"],
    ),
    Concept(
        id="delta",
        ad="Delta (Opsiyon Greek)",
        tanim=(
            "Opsiyon fiyatının dayanak fiyatındaki birim değişime duyarlılığı; kabaca "
            "'para-içi kalma olasılığı' olarak da yorumlanır."
        ),
        formul_ref=["bs_greeks"],
        tuzaklar=[
            "Delta zamanla ve dayanak hareket ettikçe değişir (gamma etkisi) — statik hedge yeterli olmayabilir.",
            "Yüksek gamma dönemlerinde (vade yaklaşırken ATM) delta çok hızlı kayar; nadir yeniden dengeleme riskli.",
        ],
        iliskili=["black_scholes", "gamma"],
    ),
    Concept(
        id="gamma",
        ad="Gamma (Opsiyon Greek)",
        tanim="Delta'nın dayanak fiyatına göre ikinci türevi; delta'nın ne kadar hızlı değiştiğini ölçer.",
        formul_ref=["bs_greeks"],
        tuzaklar=[
            "Vadeye yakın ATM opsiyonlarda gamma patlar — küçük fiyat hareketleri PnL'i büyük oynatır (gamma riski).",
            "Portföy net gamma limiti (Madde 27) izlenmezse hedge maliyeti kontrolsüz büyüyebilir.",
        ],
        iliskili=["delta", "theta"],
    ),
    Concept(
        id="vega",
        ad="Vega (Opsiyon Greek)",
        tanim="Opsiyon fiyatının zımni oynaklıktaki (IV) birim değişime duyarlılığı.",
        formul_ref=["bs_greeks"],
        tuzaklar=[
            "Uzun opsiyon pozisyonları IV crush'a (olay sonrası ani IV düşüşü) karşı savunmasızdır — dayanak yönü doğru olsa bile vega kaybı domine edebilir.",
            "Portföy net vega limiti (Madde 27) aşılırsa tek bir IV şokunda toplu kayıp riski oluşur.",
        ],
        iliskili=["black_scholes", "implied_vol"],
    ),
    Concept(
        id="theta",
        ad="Theta (Opsiyon Greek)",
        tanim="Opsiyon fiyatının zamanın geçişiyle (sabit diğer koşullarda) kaybettiği değer; 'zaman aşınması'.",
        formul_ref=["bs_greeks"],
        tuzaklar=[
            "Theta vade yaklaştıkça (özellikle ATM) hızlanır — uzun opsiyon tutmak sabit bir maliyet değildir.",
            "Net portföy theta limiti (Madde 27) günlük aşınmayı sınırlamak için izlenmeli.",
        ],
        iliskili=["black_scholes", "gamma"],
    ),
    Concept(
        id="implied_vol",
        ad="Zımni Oynaklık (Implied Volatility)",
        tanim=(
            "Piyasa fiyatını Black-Scholes'e geri-çözerek elde edilen, piyasanın gelecek "
            "oynaklık beklentisini yansıtan oynaklık değeri."
        ),
        formul_ref=["implied_vol"],
        tuzaklar=[
            "Çözücü `options_math.implied_vol`'dur (Tur-32; öncesinde YOKTU): bs_price σ'da kesin artan olduğu için Brent parantezlemesi güvenlidir — Newton derin OTM'de vega≈0 yüzünden ıraksar.",
            "Arbitrajsızlık bandının DIŞINDAKİ bir kotasyon için IV TANIMSIZDIR; implied_vol orada None döner — sıfır/uydurma bir σ ile devam etmek sessiz veri hatasıdır.",
            "IV rank/percentile geçmiş IV dağılımına göredir — mutlak seviye tek başına 'ucuz/pahalı' anlamına gelmez.",
        ],
        iliskili=["black_scholes", "vega"],
    ),
    Concept(
        id="put_call_parity",
        ad="Put-Call Parite",
        tanim=(
            "Aynı strike/vadeli Avrupa call ve put fiyatları ile dayanak/nakit arasındaki "
            "arbitraj-serbestliği bağıntısı (C - P = S - K·e^(-rT))."
        ),
        formul_ref=["bs_price"],
        tuzaklar=[
            "Amerikan opsiyonlarda (erken kullanım opsiyonu) parite tam sağlanmaz — yalnızca yaklaşık kontrol için kullanılmalı.",
            "Temettü ödeyen dayanaklarda parite formülü temettü terimi eklenmeden yanlış sonuç verir.",
        ],
        iliskili=["black_scholes", "delta"],
    ),
    Concept(
        id="max_drawdown",
        ad="Maksimum Düşüş (Max Drawdown)",
        tanim="Equity eğrisinin geçmiş zirvesinden gördüğü en büyük yüzde geri çekilmesi (büyüklük + süre).",
        formul_ref=["max_drawdown", "ulcer_index"],
        tuzaklar=[
            "Tek bir maksimum değer, düşüşün SIKLIĞINI/SÜRESİNİ göstermez — Ulcer Index bu boyutu tamamlar.",
            "Kısa backtest penceresinde gerçek kuyruk-riski (henüz görülmemiş kriz) düşük tahmin edilebilir.",
        ],
        iliskili=["var", "cvar"],
    ),
    Concept(
        id="position_sizing",
        ad="Pozisyon Boyutlama",
        tanim=(
            "Sermayenin ne kadarının tek bir işleme risk edileceğini belirleyen kural "
            "(sabit yüzde, Kelly, volatilite-hedefleme vb.)."
        ),
        formul_ref=["kelly_binary", "kelly_continuous", "fractional_kelly"],
        tuzaklar=[
            "Kelly-türevi boyutlama, girdi olasılıklarının (p_win) DOĞRU kalibre olduğunu varsayar — kalibrasyonsuz model boyutu yanlış büyütür.",
            "Boyutlama kuralı risk yapılandırmasından (RiskConfig) bağımsız uygulanırsa portföy tavanlarını (Madde 14) aşabilir.",
        ],
        iliskili=["kelly", "max_drawdown"],
    ),
    Concept(
        id="drift_detection",
        ad="Model Sapma Tespiti (Drift Detection)",
        tanim=(
            "Bir modelin/stratejinin isabet oranının zamanla istatistiksel olarak anlamlı "
            "şekilde bozulduğunu erken yakalayan izleme tekniği (örn. Page-Hinkley testi)."
        ),
        formul_ref=[],
        tuzaklar=[
            "page_hinkley quant_math'te değil nihai.ai.learning'dedir — burada yalnız kavramsal köprü verilir (formül_ref boş).",
            "Drift alarmı TEK BAŞINA nedeni açıklamaz (rejim değişimi mi, gerçek model bozulması mı) — ek teşhis gerektirir.",
        ],
        iliskili=["brier", "beta_binomial"],
    ),
    Concept(
        id="brier",
        ad="Brier Skoru",
        tanim=(
            "Olasılıksal tahminlerin kalibrasyon kalitesini ölçen ortalama-kare-hata "
            "benzeri metrik (0=mükemmel, 1=en kötü)."
        ),
        formul_ref=[],
        tuzaklar=[
            "brier_score quant_math'te değil nihai.ai.learning'dedir (kalibrasyon ölçümü, kenar kanıtı değildir).",
            "Dengesiz sınıf dağılımında düşük Brier tek başına 'iyi model' anlamına gelmeyebilir — reliability_bins ile birlikte okunmalı.",
            "BRIER İYİLEŞMESİ 'bilgi bulundu' DEMEK DEĞİLDİR (ACI DERS #9): SABİT "
            "taban-oran tahmincisiyle EŞLEŞTİRİLMİŞ fark (learning.paired_brier_t, "
            "t<-1.645) alınmadan raporlanamaz — kazancın tamamı taban orana BÜZÜLMEDEN "
            "geliyor olabilir.",
        ],
        ders_ref=["ACI-9"],       # Tur-48: bu kavram #9'un konusu ama ders_ref BOŞTU
        iliskili=["beta_binomial", "drift_detection"],
    ),
    Concept(
        id="beta_binomial",
        ad="Beta-Binom Posterior",
        tanim=(
            "Kazanç/kayıp sayımından (Bernoulli) Bayesçi güncellemeyle kazanma "
            "olasılığının posterior dağılımını (Beta) türeten yöntem."
        ),
        formul_ref=[],
        tuzaklar=[
            "beta_posterior quant_math'te değil nihai.ai.learning'dedir; az örneklemde (az işlem) posterior geniş CI ile belirsiz kalır — bu KASITLI (ABSTAIN tetikleyicisi).",
            "Prior seçimi (prior_a/prior_b) keyfiyse az veriyle posterior'ı yanlış yöne çekebilir; varsayılan uniform (1,1) en az varsayımlıdır.",
        ],
        iliskili=["kelly", "brier"],
    ),
    Concept(
        id="carry",
        ad="Carry (Taşıma Getirisi)",
        tanim=(
            "Bir pozisyonu zaman içinde TUTMANIN (fiyat değişimi hariç) getirisi/maliyeti "
            "— opsiyonda theta/temettü, vadeli işlemde faiz farkı gibi."
        ),
        formul_ref=[],
        tuzaklar=[
            "Sistemde şu an doğrudan bir carry-trade stratejisi yok — kavram opsiyon theta'sı ve fonlama maliyeti bağlamında referans amaçlıdır.",
            "Yüksek carry görünümü çoğu zaman yüksek kuyruk-riskiyle birlikte gelir (carry-crash korelasyonu) — tek başına 'ücretsiz getiri' değildir.",
        ],
        iliskili=["theta", "kelly"],
    ),

    # -----------------------------------------------------------------------
    # v3 (Tur-16 WP-K) EKLENEN KAVRAMLAR (+58, toplam >=100). Gruplar: volatilite
    # ailesi, kuyruk riski, bağımlılık, mikroyapı, makro/ekonomi, opsiyon yüzeyi,
    # yürütme, istatistiksel-öğrenme tuzakları, derin-matematik/bilgi-kuramı.
    # formul_ref'ler quant_math v3 YENİ fonksiyonlarına da köprü kurar. `prereqs`
    # yalnız bu grupta doldurulur (v2 kavramları prereqs=[] varsayılanında kalır);
    # eski kavramlar da geçerli önkoşul hedefi olabilir (curriculum() DAG'i buna göre).
    # -----------------------------------------------------------------------

    # --- Volatilite ailesi -------------------------------------------------
    Concept(
        id="realized_vol",
        ad="Gerçekleşen Oynaklık (Realized Volatility)",
        tanim=(
            "Geçmiş fiyat hareketinden ÖLÇÜLEN (tahmin edilen değil) oynaklık; kapanış-"
            "kapanış veya OHLC-menzil tabanlı tahmincilerin ortak şemsiye kavramı."
        ),
        formul_ref=["vol_close_to_close", "vol_parkinson", "vol_garman_klass", "vol_yang_zhang"],
        tuzaklar=[
            "Tahminci seçimi sonucu değiştirir — Parkinson/GK/YZ aynı seride farklı sayı verir; tek tahminciye körü körüne güvenme.",
            "Geçmiş gerçekleşen vol, GELECEK vol'ün garantili öngörücüsü değildir (bkz. garch, variance_risk_premium).",
        ],
        iliskili=["parkinson_vol", "garman_klass_vol", "yang_zhang_vol", "ewma_vol"],
    ),
    Concept(
        id="volatility_clustering",
        ad="Oynaklık Kümelenmesi",
        tanim=(
            "Büyük fiyat hareketlerinin büyük hareketleri, küçüklerin küçükleri izleme "
            "eğilimi (Mandelbrot 1963) — GARCH/EWMA ailesinin varlık nedeni."
        ),
        formul_ref=["ewma_vol", "garch11_fit"],
        tuzaklar=[
            "Kümelenme GÖRÜLMESİ tek başına ticaret edilebilir kenar değildir — yalnız risk-tahmini kalitesini artırır.",
            "Rejim değişiminde (bkz. rejim) kümelenme yapısı aniden kırılabilir; sabit-parametreli model geriden gelir.",
        ],
        iliskili=["realized_vol", "garch", "rejim"],
        prereqs=["realized_vol"],
    ),
    Concept(
        id="garch",
        ad="GARCH(1,1) Modeli",
        tanim=(
            "Koşullu varyansı geçmiş şok VE geçmiş varyansın ağırlıklı toplamıyla "
            "modelleyen, oynaklık kümelenmesini yakalayan durağan (mean-reverting) süreç."
        ),
        formul_ref=["garch11_fit", "garch11_forecast"],
        tuzaklar=[
            "MLE yüzeyi düz bölgeler/yerel-optimum içerebilir — tek başlangıç noktası erken yakınsayabilir (garch11_fit çoklu-başlangıç kullanır, bilinçli tasarım).",
            "alpha+beta>=1 (IGARCH) durağan DEĞİLDİR — koşulsuz varyans tanımsız; garch11_fit bu bölgeyi kısıtla eler.",
        ],
        iliskili=["ewma_vol", "volatility_clustering", "variance_risk_premium"],
        prereqs=["volatility_clustering", "ewma_vol"],
    ),
    Concept(
        id="variance_risk_premium",
        ad="Volatilite Risk Primi (VRP)",
        tanim=(
            "Zımni oynaklık (piyasanın fiyatladığı beklenen vol) ile sonradan GERÇEKLEŞEN "
            "oynaklık arasındaki fark; opsiyon satıcısının aldığı sistematik risk primi."
        ),
        formul_ref=["vol_close_to_close"],
        tuzaklar=[
            "IV ayağı `options_math.implied_vol`, gerçekleşen ayağı `microstructure` (RV/BV) veya `vol_*` tahmincileridir — ikisi AYNI ufka hizalanmazsa VRP ölçümü anlamsızdır.",
            "VRP çoğunlukla pozitiftir (IV>gerçekleşen) ama kriz dönemlerinde işaret değiştirebilir — 'her zaman sat' stratejisi kuyruk-riskine açıktır (bkz. tail_risk_hedging).",
        ],
        iliskili=["implied_vol", "garch", "realized_vol"],
        prereqs=["garch", "implied_vol"],
    ),

    # --- Kuyruk riski (EVT/GPD) ---------------------------------------------
    Concept(
        id="fat_tails",
        ad="Kalın Kuyruklar (Fat Tails)",
        tanim=(
            "Finansal getiri dağılımlarının normal dağılıma göre aşırı-uç olayları çok "
            "daha sık üretme özelliği; VaR/CVaR'ın normal-varsayımlı versiyonlarını "
            "hafife-alıcı kılar."
        ),
        formul_ref=["var_cornish_fisher"],
        tuzaklar=[
            "Normal-dağılım varsayımlı VaR (yalnız mean/std) kalın kuyrukta sistematik olarak DÜŞÜK risk gösterir — Cornish-Fisher veya EVT (evt_genel) tercih edilmeli.",
            "'Kalın kuyruk var' bulgusu tek başına strateji değildir — kuyruk riskini NASIL yöneteceğin (hedge/pozisyon-boyutu) ayrı bir karardır.",
        ],
        iliskili=["var", "cvar", "evt_genel"],
        prereqs=["var"],
    ),
    Concept(
        id="evt_genel",
        ad="Aşırı Değer Teorisi (EVT)",
        tanim=(
            "Dağılımın YALNIZ kuyruk bölgesini modelleyen istatistik dalı; Hill "
            "tahmincisi (kuyruk üsteli) ve POT/GPD (kuyruk-ötesi VaR/ES) iki ana araçtır."
        ),
        formul_ref=["evt_hill_tail_index", "evt_gpd_var_es"],
        tuzaklar=[
            "EVT sonuçları eşik/k seçimine DUYARLIDIR — tek bir eşikle 'kesin' kuyruk riski iddia etmek yanıltıcıdır (hassasiyet taraması gerekir).",
            "Az aşım-sayısı (Nu<10-20) ile GPD/Hill fit'i gürültülü/dejenere olur — evt_gpd_var_es bunu Nu<10'da None ile eler.",
        ],
        iliskili=["fat_tails", "hill_estimator", "gpd_pot"],
        prereqs=["fat_tails"],
    ),
    Concept(
        id="hill_estimator",
        ad="Hill Tahmincisi",
        tanim=(
            "En-büyük k gözlemin log-oranlarından kuyruk üsteli (alfa) tahmin eden, "
            "parametrik-olmayan EVT yöntemi (Hill 1975)."
        ),
        formul_ref=["evt_hill_tail_index"],
        tuzaklar=[
            "k çok küçükse (az gözlem) tahmin yüksek-varyanslı, k çok büyükse (merkez-kütle karışır) yanlı olur — klasik 'Hill ufku' ödünleşimi.",
            "alfa küçüldükçe (ör. <4) daha yüksek momentler (kurtosis, hatta varyans alfa<2'de) SONSUZ/tanımsız olabilir — portföy istatistiklerinde dikkat.",
        ],
        iliskili=["evt_genel", "fat_tails"],
        prereqs=["evt_genel"],
    ),
    Concept(
        id="gpd_pot",
        ad="GPD / Peaks-Over-Threshold (POT)",
        tanim=(
            "Bir eşiğin ÜSTÜNDEKİ aşımların Genelleştirilmiş Pareto Dağılımı'na (GPD) "
            "uydurulmasıyla eşik-ötesi VaR ve Expected Shortfall üreten EVT yöntemi "
            "(McNeil & Frey 2000)."
        ),
        formul_ref=["evt_gpd_var_es"],
        tuzaklar=[
            "alpha (istenen kuyruk olasılığı), empirik aşım-oranından BÜYÜK/EŞİT olursa ekstrapolasyon yönü ters döner — evt_gpd_var_es bu durumda None döner (amaç-dışı kullanım guard'ı).",
            "Şekil parametresi xi>=1 ise Expected Shortfall SONSUZDUR (ortalama tanımsız) — fonksiyon bu durumda None döner, 'çok büyük ama sayısal' bir ES beklenmemeli.",
        ],
        iliskili=["evt_genel", "cvar", "kuyruk_riski_hedge"],
        prereqs=["evt_genel", "cvar"],
    ),

    # --- Bağımlılık (kopula) -------------------------------------------------
    Concept(
        id="copula_genel",
        ad="Kopula",
        tanim=(
            "İki (veya çok) değişkenin BAĞIMLILIK yapısını, marjinal dağılımlarından "
            "AYRI olarak tanımlayan fonksiyon (Sklar teoremi); korelasyonun ötesinde "
            "doğrusal-olmayan/kuyruk-bağımlılığını da yakalar."
        ),
        formul_ref=["copula_fit_gaussian", "copula_fit_t"],
        tuzaklar=[
            "Pearson korelasyonu YALNIZ doğrusal bağımlılığı ölçer — kopula, kriz-dönemi 'birlikte çöküş' gibi doğrusal-olmayan yapıyı da yakalar (bkz. tail_dependence).",
            "Yanlış kopula ailesi seçimi (ör. Gauss, gerçekte t-kopula gerekirken) kuyruk riskini SİSTEMATİK olarak hafife alır (2008 krizinin ünlü 'Gauss kopula hatası').",
        ],
        iliskili=["gaussian_copula", "student_t_copula", "kuyruk_bagimliligi"],
    ),
    Concept(
        id="gaussian_copula",
        ad="Gauss Kopulası",
        tanim=(
            "Normal-skorlara dönüştürülmüş pseudo-gözlemlerin Pearson korelasyonuyla "
            "(kapalı-form MLE) tanımlanan kopula ailesi; kuyruk-bağımlılığı YOKTUR "
            "(asimptotik olarak sıfır)."
        ),
        formul_ref=["copula_fit_gaussian"],
        tuzaklar=[
            "Kuyruk-bağımlılığı sıfır varsayımı, finansal ortak-çöküş riskini SİSTEMATİK olarak hafife alır — kredi türevlerinde tarihsel bir başarısızlık kaynağı.",
            "rho tek-parametreli özet — asimetrik (alt≠üst kuyruk) bağımlılığı yakalayamaz.",
        ],
        iliskili=["copula_genel", "student_t_copula"],
        prereqs=["copula_genel"],
    ),
    Concept(
        id="student_t_copula",
        ad="Student-t Kopulası",
        tanim=(
            "Gauss kopulasının serbestlik-derecesi (nu) parametreli genellemesi; nu "
            "küçüldükçe simetrik (alt=üst) ama SIFIR-OLMAYAN kuyruk-bağımlılığı taşır."
        ),
        formul_ref=["copula_fit_t"],
        tuzaklar=[
            "nu büyüdükçe (ör. >30) Gauss kopulasına pratik olarak yakınsar — 'gerçekten t mi gerekiyor' testi (nu tahmini) atlanmamalı.",
            "Simetrik kuyruk varsayımı (alt=üst) gerçek piyasa verisinde çoğu zaman ihlal edilir (düşüş-kuyruğu genelde daha kalın) — asimetrik kopulalar Tur-34'te EKLENDİ (bkz. arsimedyen_kopula: copula_fit_clayton/copula_fit_gumbel); hangisinin geçerli olduğu artık copula_select ile AIC üzerinden VERİYE sorulur.",
        ],
        iliskili=["copula_genel", "kuyruk_bagimliligi", "fat_tails"],
        prereqs=["copula_genel", "fat_tails"],
    ),
    Concept(
        id="kuyruk_bagimliligi",
        ad="Kuyruk Bağımlılığı (Tail Dependence)",
        tanim=(
            "İki değişkenin AYNI ANDA aşırı-uç değer alma koşullu olasılığı (alt "
            "lambda_L, üst lambda_U); çeşitlendirmenin TAM DE İHTİYAÇ DUYULDUĞU ANDA "
            "(kriz) ne kadar işe yaradığının ölçüsü."
        ),
        formul_ref=["tail_dependence"],
        tuzaklar=[
            "Örneklem-dışı (ampirik) tahmin, kuyruğa YAKIN ama içine GİRMEYEN q eşiği kullanır (q->0 teorik limite yaklaşır) — küçük q, az gözlem, yüksek varyans ödünleşimi.",
            "Sıradan (tüm-örneklem) korelasyon DÜŞÜK olsa bile kuyruk-bağımlılığı YÜKSEK olabilir — portföy çeşitlendirmesi 'normal günlerde' iyi, 'kriz gününde' kötü çalışabilir.",
        ],
        iliskili=["copula_genel", "evt_genel"],
        prereqs=["copula_genel", "evt_genel"],
    ),

    # --- Mikroyapı -----------------------------------------------------------
    Concept(
        id="market_microstructure",
        ad="Piyasa Mikroyapısı",
        tanim=(
            "Emir defteri, alış-satış farkı, işlem mekanizması gibi FİYAT-OLUŞUM "
            "sürecinin ayrıntılarını inceleyen alan; kısa-vadeli sinyallerin çoğu "
            "kaynağı VE tuzağıdır (bkz. bid_ask_bounce)."
        ),
        formul_ref=[],
        tuzaklar=[
            "Günlük/saatlik bar verisi mikroyapıyı DOĞRUDAN gözlemleyemez — tick/emir-defteri verisi olmadan mikroyapı iddiaları spekülatiftir.",
            "Mikroyapısal gürültü (bid_ask_bounce), gerçek mean-reversion sanılabilir — exec_lag testiyle ayrıştırılmalı (bkz. ACI-7).",
        ],
        iliskili=["bid_ask_spread", "market_impact", "bid_ask_bounce"],
    ),
    Concept(
        id="bid_ask_spread",
        ad="Alış-Satış Farkı (Bid-Ask Spread)",
        tanim=(
            "Piyasa yapıcının alış (bid) ve satış (ask) fiyatları arasındaki fark; "
            "likidite-sağlama maliyetinin ve işlem maliyetinin temel bileşeni."
        ),
        formul_ref=[],
        tuzaklar=[
            "Dar spread her zaman 'ucuz işlem' anlamına gelmez — büyük emirler spread'in ÖTESİNDE fiyat etkisi (market_impact) yaratır.",
            "Spread, oynaklık arttıkça (haber, açılış/kapanış) genişler — backtest'te sabit-bps maliyet varsayımı bu dinamiği kaçırır.",
        ],
        iliskili=["market_microstructure", "bid_ask_bounce", "market_impact"],
        prereqs=["market_microstructure"],
    ),
    Concept(
        id="market_impact",
        ad="Piyasa Etkisi (Market Impact)",
        tanim=(
            "Bir emrin KENDİSİNİN fiyatı hareket ettirme derecesi; geçici (temporary, "
            "emir bitince sönümlenir) ve kalıcı (permanent, bilgi-taşır) bileşenlere "
            "ayrılır."
        ),
        formul_ref=["almgren_chriss_cost"],
        tuzaklar=[
            "Büyük emri TEK SEFERDE göndermek, aynı miktarı zamana YAYMAKtan sistematik olarak daha maliyetlidir (bkz. almgren_chriss) — sinyal-only sistemlerde bile raporlama için önemli.",
            "Geçmiş fiyat hareketini 'kenar' sanıp emrin KENDİ etkisini kenara karıştırmak backtest'te görünmeyen, canlıda ortaya çıkan bir maliyettir.",
        ],
        iliskili=["market_microstructure", "almgren_chriss", "bid_ask_spread"],
        prereqs=["market_microstructure"],
    ),
    Concept(
        id="pin_informed_trading",
        ad="PIN (Probability of Informed Trading)",
        tanim=(
            "Emir akışının 'bilgili' (informed, özel bilgiye dayalı) taraftan mı yoksa "
            "'gürültü' (noise) taraftan mı geldiğini kestiren mikroyapı modeli."
        ),
        formul_ref=[],
        tuzaklar=[
            "Yüksek PIN tahmini geriye-dönük (ex-post) örneklemle hesaplanır — canlı/gerçek-zamanlı kullanımı ek varsayım/güncelleme gerektirir.",
            "PIN tahmincisi Tur-34'te EKLENDİ (microstructure.pin_ekop; bkz. pin_bilgili_islem). VPIN onun HACIM-SENKRONİZE VEKİLİdir — yapısal EKOP modelinin kendisi değildir; ikisi aynı sayıyı vermez ve vermesi beklenmez.",
        ],
        iliskili=["market_microstructure", "bid_ask_spread"],
        prereqs=["market_microstructure", "bid_ask_spread"],
    ),

    # --- Makro/ekonomi ---------------------------------------------------------
    Concept(
        id="yield_curve",
        ad="Getiri Eğrisi (Yield Curve)",
        tanim=(
            "Farklı vadelerdeki devlet tahvili getirilerinin vadeye göre grafiği; eğim "
            "(uzun-kısa fark) ekonomik döngü/resesyon beklentisinin klasik göstergesidir."
        ),
        formul_ref=[],
        tuzaklar=[
            "Ters (inverted) eğri tarihsel olarak resesyon-öncüsü sayılır ama ZAMANLAMA belirsizdir (aylar-yıllar gecikme) — doğrudan işlem sinyali değildir.",
            "Bu modülde doğrudan bir getiri-eğrisi veri kaynağı/hesaplayıcısı YOKTUR — kavramsal/makro-bağlam referansıdır (bkz. WP-T takvim/makro farkındalığı, gelecek dalga).",
        ],
        iliskili=["inflation_regime", "liquidity_cycle", "rate_equity_channel"],
    ),
    Concept(
        id="inflation_regime",
        ad="Enflasyon Rejimi",
        tanim=(
            "Fiyatlar genel seviyesinin yükseliş hızının (ve TREND'inin) sınıflandığı "
            "makro durum; varlık-sınıfı korelasyonlarını ve faktör getirisini rejime "
            "göre değiştirir."
        ),
        formul_ref=["classify_regime"],
        tuzaklar=[
            "`classify_regime` FİYAT-seviyesinde çalışır, doğrudan makro enflasyon verisi İŞLEMEZ — burada yalnız KAVRAMSAL bir köprü (rejim sınıflandırma metodolojisinin analojisi).",
            "Enflasyon rejimi geçişleri (ör. 2021-2022) hisse-tahvil korelasyonunun İŞARETİNİ değiştirebilir — sabit-korelasyon varsayımlı portföy modelleri (ledoit_wolf_cov) bu geçişte yanılır.",
        ],
        iliskili=["yield_curve", "rejim"],
        prereqs=["yield_curve"],
    ),
    Concept(
        id="liquidity_cycle",
        ad="Likidite Döngüsü",
        tanim=(
            "Merkez bankası politikası ve bankacılık sistemi kredi genişlemesinin "
            "birlikte yarattığı, varlık fiyatlarını sistematik etkileyen çok-yıllı "
            "döngü."
        ),
        formul_ref=[],
        tuzaklar=[
            "Likidite daralması (QT, faiz artışı) genelde YÜKSEK-değerlemeli/spekülatif varlıkları orantısız vurur — tek-faktör (piyasa) beta modeli bu farkı yakalamaz.",
            "Döngü FAZI (genişleme/daralma) gerçek-zamanlı kesin belirlenemez — yalnız geriye-dönük netleşir (look-ahead riski yüksek bir makro değişken).",
        ],
        iliskili=["yield_curve", "fonlama_likiditesi"],
        prereqs=["yield_curve"],
    ),
    Concept(
        id="rate_equity_channel",
        ad="Faiz-Hisse Kanalı",
        tanim=(
            "Faiz oranlarındaki değişimin, iskonto oranı (değerleme) VE ekonomik "
            "büyüme beklentisi (kazanç) kanallarıyla hisse fiyatlarına aktarım "
            "mekanizması."
        ),
        formul_ref=[],
        tuzaklar=[
            "'Faiz artışı = hisse düşer' basit kuralı her zaman geçerli değildir — büyüme beklentisi kanalı bazen iskonto etkisini geçersiz kılar (kanalın YÖNÜ rejime bağlıdır).",
            "Uzun-vadeli (10y) ve kısa-vadeli faiz farklı kanallardan etkiler — tek bir 'faiz' değişkeniyle özetlemek yanıltıcı olabilir.",
        ],
        iliskili=["yield_curve", "black_scholes"],
        prereqs=["yield_curve", "black_scholes"],
    ),

    # --- Opsiyon yüzeyi --------------------------------------------------------
    Concept(
        id="vol_smile",
        ad="Volatilite Gülümsemesi (Vol Smile)",
        tanim=(
            "Aynı vadeli opsiyonlarda zımni oynaklığın (IV) strike'a göre U-şekilli "
            "(veya çarpık) değişimi; Black-Scholes'in 'sabit sigma' varsayımının "
            "ampirik ihlalidir."
        ),
        formul_ref=["svi_fit"],
        tuzaklar=[
            "Smile'ın VARLIĞI, tek bir Black-Scholes fiyatlamasının (tüm strike'larda AYNI sigma ile) sistematik yanlı olduğu anlamına gelir — ATM-dışı strike'larda dikkat.",
            "Fit `options_math.svi_fit`'tir (raw SVI, Tur-32; öncesinde YOKTU) ve fit tek başına yetmez: `svi_no_arbitrage` ile butterfly (Durrleman g(k) ≥ 0) kontrolü yapılmalıdır — iyi fit eden bir yüzey pekâlâ arbitrajlı olabilir.",
            "Log-moneyness FORWARD'a göre alınır (spot'a göre DEĞİL); forward `put_call_parity`'nin örtük forward'ıdır — yanlış çapa tüm gülümsemeyi kaydırır.",
        ],
        iliskili=["implied_vol", "vol_skew", "black_scholes"],
        prereqs=["implied_vol"],
    ),
    Concept(
        id="vol_skew",
        ad="Volatilite Çarpıklığı (Vol Skew)",
        tanim=(
            "Smile'ın SİMETRİK olmayan hali — özellikle hisse endekslerinde düşük-"
            "strike (put) IV'nin yüksek-strike (call) IV'den sistematik daha yüksek "
            "olması ('düşüş korkusu' fiyatlaması)."
        ),
        formul_ref=["bs_price"],
        tuzaklar=[
            "Skew, piyasanın kuyruk-riski ALGISINI yansıtır — GERÇEKLEŞEN kuyruk riskinin (evt_genel) doğru tahmincisi olmayabilir (algı≠gerçekleşme).",
            "Skew'in DİK olması tek başına 'opsiyon pahalı/ucuz' sinyali değildir — tarihsel skew-percentile bağlamı olmadan yorumlanmamalı.",
        ],
        iliskili=["vol_smile", "implied_vol"],
        prereqs=["vol_smile"],
    ),
    Concept(
        id="vol_term_structure",
        ad="Zımni Oynaklık Vade Yapısı",
        tanim=(
            "Aynı strike'lı (genelde ATM) opsiyonların IV'sinin VADEye göre değişimi; "
            "normal (contango, uzun-vade>kısa-vade) veya ters (backwardation, olay-"
            "öncesi tipik) olabilir."
        ),
        formul_ref=["bs_price"],
        tuzaklar=[
            "Kazanç/FOMC gibi bilinen-olay öncesi kısa-vade IV'nin sıçraması NORMAL bir vade-yapısı deseni olabilir — 'anomali' sanıp yanlış yorumlama riski.",
            "Vade-yapısının TERS DÖNMESİ (backwardation) tarihsel olarak stres göstergesidir ama zamanlama sinyali olarak tek başına test edilmemiştir (bkz. WP-E aday listesi).",
        ],
        iliskili=["vol_smile", "vol_skew"],
        prereqs=["vol_smile"],
    ),

    # --- Yürütme (Almgren-Chriss ailesi) ---------------------------------------
    Concept(
        id="almgren_chriss",
        ad="Almgren-Chriss Optimal Yürütme Modeli",
        tanim=(
            "Bir pozisyonu belirli bir ufukta tasfiye ederken GEÇİCİ piyasa-etkisi "
            "maliyeti ile fiyat-riski (elde-tutma varyansı) arasındaki ödünleşimi "
            "çözen, kapalı-forma yakın optimal-yörünge modeli (2001)."
        ),
        formul_ref=["almgren_chriss_cost"],
        tuzaklar=[
            "Bu sistem SİNYAL-ONLY'dir — almgren_chriss_cost YALNIZ rapor amaçlıdır, hiçbir execution/emir sistemine bağlanmaz (anayasa madde 1).",
            "Risk-kaçınma (gamma_risk) parametresi SUBJEKTİFTİR — 'doğru' değeri yoktur, senaryo-karşılaştırması (düşük vs yüksek risk-kaçınma) tek-sayı yorumundan daha güvenilirdir.",
        ],
        iliskili=["market_impact", "participation_rate", "implementation_shortfall"],
        prereqs=["market_impact"],
    ),
    Concept(
        id="participation_rate",
        ad="Katılım Oranı (Participation Rate)",
        tanim=(
            "Bir yürütme algoritmasının, o an piyasada işlem gören TOPLAM hacmin ne "
            "kadarını oluşturduğu oranı; büyük emirlerin piyasa-etkisini sınırlamak "
            "için tipik bir kısıt."
        ),
        formul_ref=["almgren_chriss_cost"],
        tuzaklar=[
            "Sabit katılım-oranı (ör. hacmin %10'u) stratejisi, Almgren-Chriss'in ORTALAMA-VARYANS optimal yörüngesinden farklıdır — ikisi aynı problemi FARKLI basitleştirmelerle çözer.",
            "Düşük-likit sembollerde küçük mutlak emir bile yüksek katılım-oranına denk gelebilir — 'küçük emir=düşük etki' varsayımı likiditeye göre kalibre edilmeli.",
        ],
        iliskili=["almgren_chriss", "market_impact"],
        prereqs=["almgren_chriss"],
    ),
    Concept(
        id="implementation_shortfall",
        ad="Uygulama Açığı (Implementation Shortfall)",
        tanim=(
            "Karar-anı fiyatı ile fiili ortalama yürütme fiyatı arasındaki fark; "
            "gerçek bir stratejinin TAM maliyetini (piyasa-etkisi + gecikme + fırsat-"
            "maliyeti) özetleyen standart ölçüt (Perold 1988)."
        ),
        formul_ref=["almgren_chriss_cost"],
        tuzaklar=[
            "Backtest'te sıfır/sabit-bps varsayımıyla hesaplanan 'kâr', gerçek implementation shortfall'ı SİSTEMATİK olarak hafife alabilir — bkz. exec_lag (ACI-7) ile aynı aile tuzağı.",
            "Şortfall'ın büyük kısmı bazen SİNYAL-SONRASI ilk barda oluşur (fiyat zaten hareket etmiş) — bu, kenarın kendisinin artefakt olup olmadığı sorusuyla iç içe geçer.",
        ],
        iliskili=["almgren_chriss", "exec_lag"],
        prereqs=["almgren_chriss", "exec_lag"],
        ders_ref=["ACI-7"],
    ),

    # --- İstatistiksel öğrenme tuzakları -----------------------------------
    Concept(
        id="overfitting",
        ad="Aşırı Uyum (Overfitting)",
        tanim=(
            "Bir modelin/stratejinin eğitim (in-sample) verisindeki GÜRÜLTÜYE uyum "
            "sağlayıp örneklem-dışında (OOS) genellemeyi başaramaması."
        ),
        formul_ref=[],
        tuzaklar=[
            "Tek bölmede (60/40) yüksek Sharpe/PF görmek overfitting'i EKARTE ETMEZ — purged K-fold + çift-bölme (time+symbol) zorunludur (bkz. purged_cv, ACI-3).",
            "Çok sayıda parametre/eşik denenip en iyisi seçilmesi (deneme-sayısı raporlanmazsa) 'şanslı' bir sonucu 'kenar' gibi gösterir — bkz. deflated_sharpe, pbo.",
        ],
        iliskili=["purged_cv", "pbo", "deflated_sharpe", "backtest_tuzaklari"],
        prereqs=["sharpe"],
        ders_ref=["ACI-3", "ACI-5"],
    ),
    Concept(
        id="data_leakage",
        ad="Veri Sızıntısı (Data Leakage)",
        tanim=(
            "Karar anında GERÇEKTE bilinemeyecek bilginin (gelecekteki fiyat, tüm-"
            "örneklem istatistiği, sonradan düzeltilmiş veri) eğitim/özellik hesabına "
            "sızması; look_ahead_bias'ın genel istatistiksel-öğrenme adı."
        ),
        formul_ref=[],
        tuzaklar=[
            "Yalnız açık 'gelecek fiyat' kullanımı değil — TÜM-ÖRNEKLEM normalizasyon (ör. zscore(window=None) tüm seriye bakar) da bir sızıntı biçimidir; window'lu versiyon tercih edilmeli.",
            "Purge/embargo UYGULANMADAN yapılan K-fold, komşu foldlar arası otokorelasyon üzerinden dolaylı sızıntı yaratır (bkz. purged_cv, embargo).",
        ],
        iliskili=["look_ahead_bias", "purged_cv", "embargo"],
        prereqs=["look_ahead_bias"],
        ders_ref=["ACI-3"],
    ),
    Concept(
        id="cscv",
        ad="CSCV (Combinatorially Symmetric Cross-Validation)",
        tanim=(
            "PBO hesaplamak için kullanılan, veriyi kombinatoryal olarak çok sayıda "
            "eğitim/test alt-kümesine bölüp her bölmede 'en iyi' stratejinin OOS "
            "sırasını izleyen yeniden-örnekleme yöntemi (Bailey et al.)."
        ),
        formul_ref=[],
        tuzaklar=[
            "Düşük PBO (CSCV çıktısı) tek başına kenar KANITI değildir — survivorship/exec-lag gibi başka artefaktları YAKALAMAZ (pairs örneği: PBO 0.004 iken exec-lag'de kenar çöktü).",
            "Az bölme-sayısıyla (kısa CSCV) PBO tahmini gürültülü olur — literatür genelde >=10 bölme önerir.",
        ],
        iliskili=["purged_cv", "pbo"],
        prereqs=["purged_cv", "pbo"],
        ders_ref=["ACI-7"],
    ),
    Concept(
        id="triple_barrier",
        ad="Üçlü-Bariyer Etiketleme (Triple-Barrier Labeling)",
        tanim=(
            "Bir işlemin sonucunu kâr-hedefi, zarar-durdur ve zaman-aşımı (üç bariyer) "
            "hangisine ÖNCE çarptığına göre etiketleyen, López de Prado'nun sabit-"
            "ufuk etiketlemesine göre daha gerçekçi alternatifi."
        ),
        formul_ref=[],
        tuzaklar=[
            "Sistemin MEVCUT etiketleme yöntemi (horizon-bazlı taraf-getirisi) triple-barrier DEĞİLDİR — bu kavram gelecekteki bir genelleme adayı olarak referans amaçlıdır (formul_ref boş).",
            "Bariyer genişliği (kâr/zarar eşiği) volatiliteye göre ÖLÇEKLENMEZSE yüksek-vol dönemde aşırı erken, düşük-vol dönemde aşırı geç tetiklenir.",
        ],
        iliskili=["meta_labeling"],
        prereqs=["meta_labeling"],
    ),
    Concept(
        id="permutasyon_onemi",
        ad="Permütasyon Özellik Önemi",
        tanim=(
            "Bir özelliği RASTGELE karıştırıp modelin OOS performansındaki düşüşü "
            "ölçerek özelliğin gerçek katkısını tahmin eden, model-bağımsız önem "
            "ölçütü (purged_cv modülünde kullanılır)."
        ),
        formul_ref=["mutual_info"],
        tuzaklar=[
            "Yüksek korelasyonlu özellik çiftlerinde (ör. iki farklı vol tahmincisi) her ikisi de düşük permütasyon-önemi gösterebilir (biri karıştırılınca diğeri bilgiyi TAŞIMAYA devam eder) — 'önemsiz' yanlış-yorumu.",
            "mutual_info doğrusal-olmayan bağımlılığı da yakalar ama permütasyon-önemi YERİNE geçmez — ikisi TAMAMLAYICI ölçülerdir, aynı şey değildir.",
        ],
        iliskili=["purged_cv"],
        prereqs=["purged_cv"],
    ),

    # --- Derin matematik / bilgi-kuramı (WP-M v3 köprüleri) --------------------
    Concept(
        id="fractional_differentiation",
        ad="Kesirli Fark Alma (Fractional Differentiation / FFD)",
        tanim=(
            "Tamsayı fark almanın (d=1) GEREĞİNDEN FAZLA hafıza sildiği durumlarda, "
            "kesirli d değeriyle durağanlık İLE hafıza/öngörülebilirlik arasında "
            "ayarlanabilir bir denge kuran López de Prado yöntemi."
        ),
        formul_ref=["frac_diff_ffd"],
        tuzaklar=[
            "Küçük threshold (uzun ağırlık-penceresi) daha çok hafıza korur ama pencere seriden UZUN olursa None döner — pratikte threshold/veri-uzunluğu ödünleşimi kaçınılmazdır.",
            "d'nin 'doğru' değeri ADF testiyle (bkz. duraganlik) TARANARAK bulunur — sabit d=0.5 gibi keyfi seçim optimal olmayabilir.",
        ],
        iliskili=["duraganlik", "hafiza_duraganlik_dengesi"],
    ),
    Concept(
        id="spektral_analiz",
        ad="Spektral Analiz (Welch PSD)",
        tanim=(
            "Bir zaman serisini frekans bileşenlerine ayrıştırıp güç spektral "
            "yoğunluğunu (PSD) tahmin eden yöntem (Welch 1967, ortalamalı-periodogram); "
            "döngüsel/periyodik desenleri açığa çıkarır."
        ),
        formul_ref=["welch_psd"],
        tuzaklar=[
            "Finansal seriler genelde durağan-periyodik DEĞİLDİR (rejim değişir) — PSD, TÜM örneklem üzerinden TEK bir 'ortalama' spektrum verir, zamanla-değişen periyodikliği (bkz. dalgacik_analizi) kaçırır.",
            "Baskın frekans BULUNMASI, o frekansta ticaret edilebilir bir döngü OLDUĞU anlamına gelmez — istatistiksel anlamlılık testi olmadan yorumlanmamalı.",
        ],
        iliskili=["dalgacik_analizi"],
        prereqs=["sinyal_isleme"],
    ),
    Concept(
        id="dalgacik_analizi",
        ad="Dalgacık Analizi (Haar DWT)",
        tanim=(
            "Bir seriyi ÇOK-ÖLÇEKLİ (multi-resolution) yaklaşıklık ve detay "
            "bileşenlerine ayrıştıran dönüşüm; spektral analizin aksine ZAMANA göre "
            "yerelleştirilmiş frekans bilgisi verir."
        ),
        formul_ref=["haar_dwt_vol"],
        tuzaklar=[
            "Haar, EN BASİT dalgacık ailesidir (keskin/kutu-şekilli) — düzgün (smooth) sinyallerde daha gelişmiş aileler (Daubechies vb.) daha az 'sızıntı' verir; bu modül bilinçli olarak yalnız Haar (saf-numpy) uygular.",
            "Ayrıştırma seviyesi (levels) arttıkça pencere KIRPILIR (2^levels'in katına) — çok yüksek levels, kullanılabilir veri miktarını ciddi azaltabilir.",
        ],
        iliskili=["spektral_analiz"],
        prereqs=["spektral_analiz"],
    ),
    Concept(
        id="sinyal_isleme",
        ad="Zaman-Serisi Sinyal İşleme",
        tanim=(
            "Fiyat/getiri serisini bir 'sinyal' olarak ele alıp frekans-alanı "
            "(spektral) veya zaman-frekans (dalgacık) araçlarla ayrıştıran, "
            "mühendislik kökenli analiz şemsiyesi."
        ),
        formul_ref=["welch_psd", "haar_dwt_vol"],
        tuzaklar=[
            "Sinyal-işleme araçları GÜRÜLTÜ-AYRIŞTIRMADA güçlüdür ama nedensellik/öngörü İDDİA ETMEZ — bir desenin varlığı, gelecekte SÜRECEĞİ anlamına gelmez.",
            "Finansal seriler mühendislik sinyallerinden farklı olarak durağan-OLMAYAN, rejim-değişken ve DÜŞÜK sinyal/gürültü oranlıdır — doğrudan aktarım riskli.",
        ],
        iliskili=["spektral_analiz", "dalgacik_analizi"],
    ),
    Concept(
        id="entropi_genel",
        ad="Bilgi Kuramı / Shannon Entropisi",
        tanim=(
            "Bir dağılımın/dizinin BELİRSİZLİĞİNİ (öngörülemezliğini) ölçen, Shannon "
            "(1948) kökenli matematiksel çerçeve; finans'ta desen-düzenliliği ve "
            "değişkenler-arası bilgi paylaşımı ölçümünün temeli."
        ),
        formul_ref=["permutation_entropy", "sample_entropy", "mutual_info"],
        tuzaklar=[
            "Yüksek entropi ('düzensiz') tek başına 'ticaret edilemez' anlamına gelmez, düşük entropi ('düzenli') de tek başına 'kenar var' anlamına gelmez — betimsel bir ÖZELLİK, doğrudan sinyal değildir.",
            "Farklı entropi ölçüleri (permütasyon vs örnek-tabanlı) FARKLI şeyleri yakalar — biri sıralama yapısına, diğeri genlik-benzerliğine bakar; birbirinin yerine geçmez.",
        ],
        iliskili=["permutasyon_entropisi", "ornek_entropisi", "karsilikli_bilgi"],
    ),
    Concept(
        id="permutasyon_entropisi",
        ad="Permütasyon Entropisi",
        tanim=(
            "Bir serinin m-boyutlu gecikme-gömme vektörlerinin SIRALAMA (ordinal) "
            "örüntü dağılımından hesaplanan, ölçek/monoton-dönüşüme dayanıklı "
            "düzensizlik ölçüsü (Bandt-Pompe 2002)."
        ),
        formul_ref=["permutation_entropy"],
        tuzaklar=[
            "m (gömme boyutu) çok küçükse (m=2) az örüntü-çeşidi ayırt eder, çok büyükse (m>7-8) m! örüntü için yetersiz veri kalır — tipik pratik aralık 3-7.",
            "DEĞERE değil SIRALAMAYA baktığı için büyüklük/genlik bilgisini kaybeder — genlik önemliyse sample_entropy tamamlayıcı olmalı.",
        ],
        iliskili=["entropi_genel", "ornek_entropisi"],
        prereqs=["entropi_genel"],
    ),
    Concept(
        id="ornek_entropisi",
        ad="Örnek Entropisi (Sample Entropy)",
        tanim=(
            "m ve m+1 boyutlu şablonlar arasındaki GENLİK-tabanlı (Chebyshev mesafe) "
            "eşleşme oranlarının log-oranından hesaplanan düzensizlik ölçüsü (Richman "
            "& Moorman 2000); permütasyon entropisinin genlik-tabanlı eşi."
        ),
        formul_ref=["sample_entropy"],
        tuzaklar=[
            "Tolerans r, seri std'sinin ORANI olarak yorumlanır — r çok küçükse hemen hiç eşleşme bulunmaz (None/aşırı-yüksek entropi), çok büyükse her şey 'eşleşir' (yapay-düşük entropi).",
            "O(n²) karmaşıklık — büyük n'de (binlerce+) yavaşlar; büyük seri için alt-örnekleme/pencereleme düşünülmeli.",
        ],
        iliskili=["entropi_genel", "permutasyon_entropisi"],
        prereqs=["permutasyon_entropisi"],
    ),
    Concept(
        id="karsilikli_bilgi",
        ad="Karşılıklı Bilgi (Mutual Information)",
        tanim=(
            "İki değişken arasında PAYLAŞILAN bilgi miktarı; korelasyonun aksine "
            "DOĞRUSAL-OLMAYAN bağımlılığı da (ör. y=x²) yakalayan, histogram-tabanlı "
            "deterministik tahmin."
        ),
        formul_ref=["mutual_info"],
        tuzaklar=[
            "bins (histogram hücre sayısı) seçimi sonucu etkiler — az bin bilgiyi kaybeder, çok bin (az veriyle) sonlu-örneklem yanlılığı büyütür.",
            "MI>0 bağımlılığın YÖNÜNÜ veya işaretini SÖYLEMEZ (simetrik bir ölçüdür) — hangi değişkenin diğerini 'sürdüğü' ayrı bir nedensellik sorusudur.",
        ],
        iliskili=["entropi_genel", "permutasyon_onemi"],
        prereqs=["entropi_genel"],
    ),
    Concept(
        id="yuksek_momentler",
        ad="Yüksek Momentler (Çarpıklık/Basıklık Ötesi)",
        tanim=(
            "Getiri dağılımının ortalama/varyans ÖTESİNDEKİ şekil özellikleri "
            "(çarpıklık=3., basıklık=4. moment); portföy düzeyinde COSKEWNESS/"
            "COKURTOSIS ile çok-varlık genellemesine taşınır."
        ),
        formul_ref=["coskewness", "cokurtosis"],
        tuzaklar=[
            "Yalnız varyans/kovaryansa (ör. ledoit_wolf_cov) dayalı portföy optimizasyonu, YÜKSEK-momentleri (eşzamanlı çöküş eğilimi) GÖRMEZDEN gelir — negatif coskewness riski gizli kalabilir.",
            "Yüksek momentlerin ÖRNEKLEM tahmini (özellikle 4. moment) az veri ile ÇOK gürültülüdür — coskewness/cokurtosis minimum satır-sayısı şartları (10/15) bu yüzden konservatiftir.",
        ],
        iliskili=["coskewness", "cokurtosis", "sharpe"],
        prereqs=["sharpe"],
    ),
    Concept(
        id="coskewness",
        ad="Ortak-Çarpıklık (Co-Skewness)",
        tanim=(
            "Çok-varlıklı standardize 3-indisli tensör; köşegeni tek-varlık "
            "çarpıklığına eşittir, köşegen-dışı terimler varlıklar-arası ÇARPIK "
            "ortak-hareketi (ör. bir varlık düşerken diğerinin de aşırı düşme "
            "eğilimi) ölçer."
        ),
        formul_ref=["coskewness"],
        tuzaklar=[
            "Negatif coskewness, çeşitlendirmenin TAM İHTİYAÇ DUYULAN ANDA (piyasa çöküşü) beklenenden az koruma sağlayabileceğinin işaretidir — Black-Litterman/HRP gibi yalnız-kovaryans yöntemleri bunu kaçırır (gelecek WP-P portföy köprüsü).",
            "n_obs küçükse (10'a yakın) tensör tahmini son derece gürültülüdür — yalnız yapısal/keşif amaçlı kullanılmalı, tek-sayı kesin karara dayanak yapılmamalı.",
        ],
        iliskili=["yuksek_momentler", "cokurtosis"],
        prereqs=["yuksek_momentler"],
    ),
    Concept(
        id="cokurtosis",
        ad="Ortak-Basıklık (Co-Kurtosis)",
        tanim=(
            "Çok-varlıklı standardize 4-indisli tensör; köşegeni tek-varlık HAM "
            "(Pearson) basıklığına eşittir, köşegen-dışı terimler varlıkların EŞ "
            "ZAMANLI aşırı-hareket etme eğilimini ölçer."
        ),
        formul_ref=["cokurtosis"],
        tuzaklar=[
            "4. moment tahmini 3. momentten (coskewness) DAHA FAZLA veri ister — küçük örneklemde aşırı-duyarlı, dış-değer (outlier) birkaç gözlem tüm tensörü domine edebilir.",
            "Yüksek cokurtosis = 'çeşitlendirme kuyruk-riskini az azaltır' demektir ama TEK BAŞINA hangi varlık-çiftinin sorumlu olduğunu göstermez — köşegen-dışı terimler tek tek incelenmeli.",
        ],
        iliskili=["yuksek_momentler", "coskewness"],
        prereqs=["coskewness"],
    ),
    Concept(
        id="duraganlik",
        ad="Durağanlık (Stationarity)",
        tanim=(
            "Bir serinin istatistiksel özelliklerinin (ortalama, varyans, "
            "otokorelasyon) ZAMANLA DEĞİŞMEMESİ; çoğu istatistiksel öğrenme/regresyon "
            "yönteminin zımni varsayımı."
        ),
        formul_ref=["adf_test", "frac_diff_ffd"],
        tuzaklar=[
            "Ham fiyat serisi (rastgele-yürüyüşe yakın) genelde durağan DEĞİLDİR — getiri serisine geçmek (log_returns) genelde gerekli ama HER ZAMAN yeterli değildir (bkz. hurst).",
            "Durağanlığı sağlamak için AŞIRI fark almak (d=1) hafızayı/öngörülebilirliği gereğinden fazla siler — kesirli fark alma (frac_diff_ffd) bu ödünleşimi yumuşatır.",
        ],
        iliskili=["hurst", "hafiza_duraganlik_dengesi"],
        prereqs=["hurst"],
    ),
    Concept(
        id="hafiza_duraganlik_dengesi",
        ad="Hafıza-Durağanlık Dengesi",
        tanim=(
            "İstatistiksel öğrenmede durağanlık (ADF anlamında) İLE serinin geçmiş "
            "bilgiyi (hafıza, öngörülebilirlik) ne kadar koruduğu arasındaki temel "
            "ödünleşim; López de Prado'nun FFD çerçevesinin ana motivasyonu."
        ),
        formul_ref=["frac_diff_ffd"],
        tuzaklar=[
            "Tamsayı fark alma (d=1) durağanlığı GARANTİLER ama hafızayı GEREĞİNDEN FAZLA siler — modelin geçmiş-seviye bilgisine erişimini kısıtlar.",
            "Ham seviyeyle (d=0) çalışmak hafızayı TAM korur ama durağan-olmayan seride birçok istatistiksel yöntem (regresyon, korelasyon) yanıltıcı ('sahte regresyon') sonuç verir.",
        ],
        iliskili=["duraganlik", "fractional_differentiation"],
        prereqs=["duraganlik", "fractional_differentiation"],
    ),

    # --- Rejim / olay farkındalığı (makro-bağlam köprüleri) --------------------
    Concept(
        id="rejim_gecisi",
        ad="Rejim Geçiş Modelleri",
        tanim=(
            "Piyasanın ayrık durumlar (ör. trend/yatay, düşük/yüksek-vol) arasında "
            "zamanla GEÇİŞ yaptığını varsayan modelleme yaklaşımı; `classify_regime` "
            "kural-tabanlı, olasılıksal genellemeleri (ör. Markov-switching, BOCPD) "
            "gelecek dalgada (WP-M2) planlanmıştır."
        ),
        formul_ref=["classify_regime"],
        tuzaklar=[
            "Kural-tabanlı sınıflandırma (mevcut `rejim`) GECİKMELİ tepki verir (ör. SMA50 eğimi) — geçişin TAM ANINI değil, geçiş SONRASINI yakalar.",
            "Rejim etiketi KATEGORİKTİR (ör. 'up_high') — geçiş OLASILIĞI Tur-34'te ayrı bir modülle geldi (nihai/ai/regime_hmm.py; bkz. rejim_gecis_matrisi). classify_regime hala KESKİN etiket verir; belirsizlik istendiğinde HMM'in filtreli dağılımı okunmalıdır.",
        ],
        iliskili=["rejim"],
        prereqs=["rejim"],
    ),
    Concept(
        id="sicrama_riski",
        ad="Sıçrama Riski (Jump Risk)",
        tanim=(
            "Fiyatın SÜREKLİ (difüzyon) hareketten ayrı, ANİ ve BÜYÜK (haber-kaynaklı) "
            "sıçramalar yapma eğilimi; kalın-kuyruğun bir kaynağı, GARCH'ın "
            "modellemediği bir bileşendir."
        ),
        formul_ref=[],
        tuzaklar=[
            "GARCH(1,1) SÜREKLİ volatilite kümelenmesini modeller — ANİ sıçramayı AYRI bir bileşen olarak YAKALAMAZ (sıçrama-difüzyon modelleri, ör. Merton, gelecek WP-M2 kapsamında).",
            "Sıçrama İLE veri-hatası (yanlış fiyat/split-düzeltmesi) ayırt edilmezse yanlış 'kenar' veya yanlış risk-tahmini üretilebilir (bkz. WP-O veri-sağlığı kapısı, gelecek dalga).",
        ],
        iliskili=["fat_tails", "evt_genel"],
        prereqs=["fat_tails"],
    ),
    Concept(
        id="kredi_spreadi",
        ad="Kredi Spreadi",
        tanim=(
            "Riskli (kurumsal) tahvil getirisi ile risksiz (devlet) tahvil getirisi "
            "arasındaki fark; piyasanın algıladığı temerrüt/likidite riskinin makro "
            "göstergesi."
        ),
        formul_ref=[],
        tuzaklar=[
            "Kredi spreadi GENİŞLEMESİ hisse senedi stresiyle YÜKSEK korelelidir ama nedensellik yönü döneme göre değişir — tek-yönlü öncü-gösterge varsayımı riskli.",
            "Bu modülde doğrudan bir kredi-spreadi veri kaynağı/hesaplayıcısı YOKTUR — makro-bağlam referansıdır (WP-E veri-genişletme adayları arasında, FRED kaynağı).",
        ],
        iliskili=["yield_curve"],
        prereqs=["yield_curve"],
    ),
    Concept(
        id="fonlama_likiditesi",
        ad="Fonlama Likiditesi",
        tanim=(
            "Piyasa katılımcılarının (özellikle kaldıraçlı) pozisyon FONLAMA "
            "kolaylığı; fonlama likiditesi daralması, piyasa likiditesini (dar "
            "spread) de ANİDEN kurutabilir (Brunnermeier & Pedersen 2009)."
        ),
        formul_ref=[],
        tuzaklar=[
            "Fonlama-likiditesi ve piyasa-likiditesi (bid_ask_spread) NORMAL zamanlarda ayrı görünür ama KRİZDE birbirini besleyerek sarmal oluşturabilir (likidite sarmalı) — bu etkileşim mevcut modülde formalize edilmemiştir.",
            "Kavramsal referans amaçlıdır (formul_ref boş) — doğrudan bir fonlama-likiditesi ölçütü bu modülde hesaplanmaz.",
        ],
        iliskili=["liquidity_cycle", "bid_ask_spread"],
        prereqs=["liquidity_cycle"],
    ),
    Concept(
        id="olay_riski",
        ad="Olay Riski / Takvim Farkındalığı",
        tanim=(
            "Bilinen-zamanlı makro/şirket olaylarının (FOMC, CPI, kazanç açıklaması, "
            "OPEX) yarattığı, NORMAL günlerden sistematik farklı oynaklık/likidite "
            "rejimi."
        ),
        formul_ref=[],
        tuzaklar=[
            "Olay-öncesi IV sıçraması (bkz. vol_term_structure) genelde RASYONELDİR (beklenen belirsizliği fiyatlar) — 'anomali' sanıp ticaret etmek yanıltıcı olabilir.",
            "Bu modülde doğrudan bir takvim/tarih hesaplayıcısı YOKTUR — WP-T (text_intel, gelecek dalga) takvim-farkındalığı alt-modülüne kavramsal köprüdür.",
        ],
        iliskili=["rejim", "kazanc_takvimi"],
        prereqs=["rejim"],
    ),
    Concept(
        id="kazanc_takvimi",
        ad="Kazanç Takvimi Farkındalığı",
        tanim=(
            "Bir sembolün kazanç-açıklama tarihine YAKINLIĞININ, o sembol için "
            "oynaklık/PEAD-benzeri sürüklenme beklentisini değiştirdiği takvim-tabanlı "
            "bağlam bilgisi."
        ),
        formul_ref=[],
        tuzaklar=[
            "Kazanç YAKLAŞIRKEN artan IV, kazanç SONRASI PEAD sürüklenmesiyle (bkz. pead, earnings_drift) KARIŞTIRILMAMALIDIR — biri olay-öncesi belirsizlik fiyatlaması, diğeri olay-sonrası sürüklenme iddiasıdır.",
            "PEAD/earnings_drift'in survivorship-temiz PIT testte KENAR YOK çıktığı (ACI-1/ACI-6) unutulmamalı — takvim-yakınlığı tek başına ticaret gerekçesi DEĞİLDİR.",
        ],
        iliskili=["olay_riski", "pead", "earnings_drift"],
        prereqs=["olay_riski"],
    ),

    # --- Faktör / portföy köprüleri (mevcut kovaryans/momentum altyapısına) ----
    Concept(
        id="faktor_yatirimciligi",
        ad="Faktör Yatırımcılığı",
        tanim=(
            "Getirileri ortak, sistematik 'faktörlere' (momentum, değer, kalite, "
            "düşük-oynaklık vb.) ayrıştırıp bu faktörlere BİLİNÇLİ maruziyet almayı "
            "amaçlayan yatırım yaklaşımı."
        ),
        formul_ref=[],
        tuzaklar=[
            "Akademik faktör bulguları çoğu zaman SURVIVORSHIP-şişkin evrende raporlanır — bu sistemde momentum/PEAD'in PIT testte kenarının önemli kısmı ARTEFAKT çıktı (bkz. ACI-1, kesitsel_momentum).",
            "Faktörler ARASINDA korelasyon zamanla DEĞİŞİR (faktör-krizleri, ör. 2009 momentum crash) — sabit faktör-ağırlığı varsayımı riskli.",
        ],
        iliskili=["momentum", "kesitsel_momentum"],
        prereqs=["momentum"],
    ),
    Concept(
        id="kesitsel_momentum",
        ad="Kesitsel Momentum (Cross-Sectional Momentum)",
        tanim=(
            "Sembolleri BİRBİRLERİNE göre (kesit içinde) geçmiş getirilerine göre "
            "sıralayıp en-iyi/en-kötü performansı ayrıştıran momentum varyantı — "
            "zaman-serisi momentumdan (bkz. momentum) farklıdır."
        ),
        formul_ref=["zscore"],
        tuzaklar=[
            "12/12 kombinasyonda beta-üstü pozitif görünmüştü AMA evren survivorship-şişkindi (bkz. proje hafızası 2026-06-10) — point-in-time (PIT) evren ile yeniden ölçülmeden kenar iddiası YAPILAMAZ.",
            "Kesitsel sıralama, evrenin BÜYÜKLÜĞÜNE ve YENİDEN-DENGELEME sıklığına duyarlıdır — küçük evren/sık dengeleme ciro-maliyetini büyütür.",
        ],
        iliskili=["faktor_yatirimciligi", "survivorship_bias"],
        prereqs=["faktor_yatirimciligi", "survivorship_bias"],
        ders_ref=["ACI-1"],
    ),
    Concept(
        id="risk_paritesi",
        ad="Risk Paritesi (Risk Parity)",
        tanim=(
            "Portföy ağırlıklarını SERMAYE değil RİSK KATKISI eşit olacak şekilde "
            "belirleyen portföy-inşa yaklaşımı; düşük-vol varlıklara kaldıraçla daha "
            "fazla ağırlık verir."
        ),
        formul_ref=["ledoit_wolf_cov"],
        tuzaklar=[
            "Risk paritesi genelde KALDIRAÇ gerektirir (düşük-vol varlığa 'eşit risk' için fazla ağırlık) — kaldıraç maliyeti/kısıtları göz ardı edilirse pratikte uygulanamaz.",
            "Kovaryans tahmini GÜRÜLTÜLÜYSE (Ledoit-Wolf uygulanmadan ham örneklem) risk-katkı hesabı da gürültülü çıkar — bkz. ledoit_wolf.",
        ],
        iliskili=["pca_risk", "ledoit_wolf", "cesitlendirme_orani"],
        prereqs=["pca_risk"],
    ),
    Concept(
        id="cesitlendirme_orani",
        ad="Çeşitlendirme Oranı (Diversification Ratio)",
        tanim=(
            "Portföydeki varlıkların AYRI AYRI oynaklıklarının ağırlıklı toplamı ile "
            "portföyün GERÇEK (kovaryans dikkate alınmış) oynaklığı arasındaki oran; "
            "1'e yakın=çeşitlendirme yok, büyük=güçlü çeşitlendirme."
        ),
        formul_ref=["ledoit_wolf_cov", "pca_risk"],
        tuzaklar=[
            "Yüksek çeşitlendirme oranı, KUYRUK riskinde de aynı oranda koruma sağlar VARSAYIMI yanlıştır — kuyruk-bağımlılığı (bkz. kuyruk_bagimliligi) NORMAL-dönem korelasyonundan yüksek olabilir.",
            "PCA'nın İLK bileşeni genelde 'piyasa faktörü'dür — düşük ilk-bileşen-payı sanıp gerçek çeşitlendirmeyi ABARTMAK riskli (bkz. pca_risk tuzağı).",
        ],
        iliskili=["risk_paritesi", "pca_risk"],
        prereqs=["risk_paritesi"],
    ),
    Concept(
        id="kuyruk_riski_hedge",
        ad="Kuyruk Riski Hedge'i (Tail Risk Hedging)",
        tanim=(
            "Nadir ama BÜYÜK-kayıplı olaylara karşı (genelde opsiyon-tabanlı) "
            "sigorta-benzeri koruma; maliyeti (theta/prim) normal zamanlarda getiriyi "
            "aşındırır, kriz döneminde büyük fayda sağlar."
        ),
        formul_ref=["evt_gpd_var_es", "bs_price"],
        tuzaklar=[
            "Sürekli tail-hedge TUTMAK (ör. daima OTM put) uzun-vadede theta-aşınmasıyla ÖNEMLİ maliyet biriktirir — 'ücretsiz sigorta' yoktur (bkz. carry tuzağı, aynı aile).",
            "GPD-tabanlı VaR/ES (evt_gpd_var_es) tarihsel örneklemin YAKALAMADIĞI (henüz görülmemiş) bir kuyruk olayını GARANTİ ÖNGÖREMEZ — model, veri kadar iyidir.",
        ],
        iliskili=["gpd_pot", "black_scholes"],
        prereqs=["gpd_pot", "black_scholes"],
    ),
    Concept(
        id="stres_testi",
        ad="Stres Testi / Senaryo Analizi",
        tanim=(
            "Portföyü GEÇMİŞ kriz pencereleri (ör. 2008, 2020, 2022) veya sentetik "
            "Monte Carlo şoklarıyla yeniden-fiyatlayarak olası kayıp büyüklüğünü "
            "raporlayan risk-yönetim tekniği."
        ),
        formul_ref=["cvar_historical", "mc_bootstrap_terminal"],
        tuzaklar=[
            "TARİHSEL senaryo replay'i, 'bir sonraki kriz TAM OLARAK böyle olacak' varsaymaz — yalnız BENZER büyüklükte bir şokun etkisini gösterir (ölçek referansı, kehanet değil).",
            "Monte Carlo stres testinde GBM varsayımı (mc_gbm_paths) kalın-kuyruğu/sıçramayı yakalamaz — blok-bootstrap (mc_bootstrap_terminal) gerçek tarihsel dağılıma daha sadıktır.",
        ],
        iliskili=["cvar", "monte_carlo"],
        prereqs=["cvar", "monte_carlo"],
    ),
    Concept(
        id="backtest_tuzaklari",
        ad="Backtest Tuzakları (Özet)",
        tanim=(
            "Bir backtest sonucunu YANLIŞ-İYİMSER gösteren bilinen hata ailesinin "
            "şemsiye kavramı: survivorship, look-ahead/sızıntı, aşırı-uyum, exec-lag "
            "— bu projenin ACI DERSLER'inin çoğu bu şemsiyeye girer."
        ),
        formul_ref=[],
        tuzaklar=[
            "Tuzaklardan BİRİNİ kontrol edip (ör. purged CV kullandım) DİĞERLERİNİ (survivorship, exec-lag) atlamak yanlış güven verir — TÜM kapılar (Bölüm G) sırayla geçilmelidir.",
            "'Tüm metrikler yeşil' bile YETERLİ değildir — pairs örneğinde PBO/DSR/survivorship hepsi iyiydi, exec-lag testi kenarı çökertti (ACI-7); son kapı ATLANMAMALIDIR.",
        ],
        iliskili=["overfitting", "data_leakage", "survivorship_bias"],
        prereqs=["overfitting", "data_leakage", "survivorship_bias"],
        ders_ref=["ACI-1", "ACI-3", "ACI-5"],
    ),
    Concept(
        id="ileri_yuruyuslu_analiz",
        ad="İleri-Yürüyüşlü Analiz (Walk-Forward Analysis)",
        tanim=(
            "Modeli/stratejiyi KRONOLOJİK olarak ardışık pencerelerde eğitip bir "
            "sonraki pencerede test ederek, gerçek canlı-kullanım sırasını taklit "
            "eden doğrulama yöntemi; purged K-fold'un zaman-sıralı özel hali."
        ),
        formul_ref=[],
        tuzaklar=[
            "Walk-forward, purged_cv'nin YERİNE geçmez — ikisi FARKLI soruları yanıtlar (walk-forward: 'zamanla nasıl performans gösterirdi', purged CV: 'OOS-AUC ortalaması ne').",
            "Pencere sayısı AZSA (ör. yalnız 2-3 ileri-adım) sonuç şanslı-bölme kırılganlığına purged K-fold kadar dayanıklı DEĞİLDİR.",
        ],
        iliskili=["purged_cv"],
        prereqs=["purged_cv"],
    ),
    # -----------------------------------------------------------------------
    # v6 (Tur-34) — kataloğun KENDİ işaret ettiği boşlukların kapanışı
    # -----------------------------------------------------------------------
    Concept(
        id="arsimedyen_kopula",
        ad="Arşimedyen Kopulalar (Clayton / Gumbel)",
        tanim=(
            "Tek parametreli (theta) ASİMETRİK bağımlılık aileleri: Clayton yalnız ALT "
            "kuyrukta (lambda_L = 2^(-1/theta)), Gumbel yalnız ÜST kuyrukta "
            "(lambda_U = 2 - 2^(1/theta)) bağımlılık taşır. Gauss (iki kuyrukta sıfır) ve "
            "t (simetrik) kopulalarının kapatamadığı boşluk budur."
        ),
        formul_ref=["copula_fit_clayton", "copula_fit_gumbel", "copula_tail_dependence",
                    "copula_select"],
        tuzaklar=[
            "İki aile de YALNIZ POZİTİF bağımlılık modeller (Clayton theta>0, Gumbel theta>=1 <=> tau>=0) — ampirik tau <= 0 iken uydurmak ZORLAMADIR; copula_fit_* bu durumda None döner.",
            "Aile seçimi bir MODEL KARARIDIR: 'düşüş kuyruğu daha kalın' sezgisiyle Clayton seçmek varsayımı veriye SORMADAN dayatır — copula_select AIC ile sorar.",
            "AIC farkı 2'nin altındaysa aileler AYIRT EDİLEMEZ (decisive=False); kazananı 'bulundu' diye okumak Tur-23'ün n_kept kuralının kopula karşılığıdır.",
            "Kuyruk katsayısı ASİMPTOTİK bir limittir; sonlu örneklemde ampirik tahmin (tail_dependence) çok gürültülüdür ve teorik lambda ile ayrışması normaldir.",
        ],
        iliskili=["copula_genel", "gaussian_copula", "student_t_copula", "kuyruk_bagimliligi",
                  "fat_tails"],
        prereqs=["copula_genel", "kuyruk_bagimliligi"],
    ),
    Concept(
        id="rejim_gecis_matrisi",
        ad="Rejim Geçiş Matrisi (Saklı Markov Modeli)",
        tanim=(
            "Rejimi GÖZLENEN bir etiket değil SAKLI bir durum olarak modelleyip her bar "
            "için OLASILIK DAĞILIMI, rejimler arası GEÇİŞ matrisi ve rejim başına beklenen "
            "SÜRE üreten yapı (Gauss HMM, Baum-Welch ile uydurulur)."
        ),
        formul_ref=["gaussian_hmm_fit", "hmm_filter", "hmm_next_state_probs",
                    "hmm_transition_summary", "hmm_select_states"],
        tuzaklar=[
            "hmm_smooth ve hmm_viterbi GELECEĞİ GÖRÜR (causal=False) — geçmişi etiketlemek için doğrudur, sinyal/özellik hesabında look-ahead'dir. Canlı yolda yalnız hmm_filter kullanılır.",
            "gaussian_hmm_fit parametreleri TÜM örneklemden tahmin eder = iç-örneklem; canlı kullanımda eğitim penceresinde fit edip ileri uygulamak gerekir (purged_cv disiplininin HMM karşılığı).",
            "Durum kimliği permütasyona kadar tanımlıdır; bu modül durumları ORTALAMAYA göre sıralar — yine de 'durum 0' evrensel bir anlam taşımaz.",
            "mode_prob okunmadan mode okunamaz: 0,34'lük bir tepe rejim ilan etmez (Tur-33 'bir oran paydası okunmadan okunamaz' kuralı).",
            "Rejim SAYISI bir model kararıdır; elle 2 seçmek varsayımı gizler — hmm_select_states BIC ile sorar ve deltaBIC<2'de decisive=False der.",
        ],
        iliskili=["rejim", "rejim_gecisi", "volatility_clustering"],
        prereqs=["rejim"],
        ders_ref=["ACI-3"],
    ),
    Concept(
        id="ssvi_arbitrajsiz_yuzey",
        ad="SSVI — Arbitrajsız Volatilite Yüzeyi",
        tanim=(
            "Gatheral-Jacquier (2014) yüzey parametrizasyonu: dilimler ortak (rho, eta, "
            "gamma) ve dilim başına ATM toplam varyans theta_t ile bağlanır. gamma <= 1/2 "
            "ve eta*(1+|rho|) <= 2 iken KELEBEK arbitrajı YAPISI GEREĞİ yoktur; theta_t "
            "artan iken takvim arbitrajı da yoktur."
        ),
        formul_ref=["ssvi_total_variance", "ssvi_fit", "ssvi_no_arbitrage", "ssvi_diagnose"],
        tuzaklar=[
            "SSVI raw SVI'den DAHA KÖTÜ uyar — bu bir kusur değil, arbitrajsızlığın FİYATIDIR (serbestlik derecesi azaltılarak elde edilir). 'RMSE daha yüksek' tek başına SSVI aleyhine kanıt değildir.",
            "enforce_arb_free=False ile fit iyileşir ama arbitrajsızlık İDDİA EDİLEMEZ; no_arb raporu okunmadan sonuç kullanılamaz.",
            "ssvi_diagnose 'VERIDE_ARBITRAJ' derse kaynağı SÖYLEMEZ: asenkron EOD kotasyonu, Amerikan erken-kullanım primi ve gerçek arbitraj bu testle BİRBİRİNDEN ayrılmaz — hepsi 'fit hatası' şıkkından ayrılır.",
            "Verdikt bir KENAR iddiası değildir; arbitraj bulmak onu işlenebilir kılmaz (maliyet/erişim/eşzamanlılık ayrı sorulardır).",
        ],
        iliskili=["vol_smile", "vol_skew", "vol_term_structure", "implied_vol"],
        prereqs=["vol_smile"],
    ),
    Concept(
        id="pin_bilgili_islem",
        ad="PIN — Bilgiye Dayalı İşlem Olasılığı",
        tanim=(
            "Easley-Kiefer-O'Hara-Paperman (1996) yapısal modeli: alpha olasılıkla bilgi "
            "olayı olur (delta olasılıkla kötü haber), bilgili işlemciler haberin yönünde "
            "Poisson(mu) ekler, bilgisiz akış Poisson(eps_b)/Poisson(eps_s)'dir. "
            "PIN = alpha*mu / (alpha*mu + eps_b + eps_s) = bir işlemin bilgili olma olasılığı."
        ),
        formul_ref=["pin_ekop", "pin_from_bars"],
        tuzaklar=[
            "Poisson varsayımı İŞLEM SAYISI üzerinedir; hacim aşırı-dağılımlıdır. pin_from_bars bu yüzden proxy=True bayrağıyla döner ve gerçek tick PIN'i ile AYNI ŞEY DEĞİLDİR.",
            "Olabilirlik ÇOK-TEPELİDİR; tek başlangıçlı optimizasyon yerel optimuma kilitlenir (bu modül sabit bir başlangıç ızgarası tarar).",
            "PIN ölçek-değişmezdir (tüm sayımlar c ile çarpılınca değişmez) — bu bir kolaylıktır ama alpha/mu ayrışmasının anlamlı olduğu ANLAMINA GELMEZ.",
            "Yüksek PIN 'kenar' değil TERS SEÇİM riskidir: bilgili akışın karşı tarafında kalma olasılığı yüksektir (ACI DERS #7'nin makas teşhisiyle birlikte okunur).",
        ],
        iliskili=["bid_ask_spread", "market_impact", "fonlama_likiditesi"],
        prereqs=["bid_ask_spread"],
        ders_ref=["ACI-7"],
    ),
    # --- Tur-35 (AI v7) ---
    Concept(
        id="maskeli_on_egitim",
        ad="Maskeli Özellik Ön-Eğitimi (öz-denetimli)",
        tanim=(
            "Her satırda özelliklerin bir kısmının TOKEN'ı ogrenilebilir bir [MASK] gommesiyle "
            "DEGISTIRILIR ve ag maskelenen ozelligin gercek (normalize) degerini geri kurar; "
            "kayip YALNIZ maskeli konumlardan alinir. Etiket GEREKTIRMEZ — ozellikler arasi "
            "kosullu yapiyi etiketsiz veriden ogrenir (BERT'in maskeleme fikrinin tablosal "
            "karsiligi: VIME/SAINT ailesi). Ogrenilen govde, etiketli egitime warm start olur. "
            "Uygulama: nihai.ai.neural.NeuralCapacity.pretrain (formul koprusune EKLENMEZ — "
            "koprü torch importunu tetiklemez, bkz. selfcheck._FORMULA_MODULES)."
        ),
        formul_ref=[],
        tuzaklar=[
            "Rekonstruksiyon kaybinin dusmesi YON TAHMININ iyilesecegi anlamina GELMEZ — iki gorev ayri olculur (ACI DERS #5: yeni bir siniflandirici darbogazi cozmez; bu yontemin degeri ETIKETSIZ veriyi kullanabilmesindedir).",
            "On-egitilmis govde EGITILMIS MODEL DEGILDIR: is_fitted acilmaz, predict_proba reddeder. Etiket gormemis govdeden p_win okumak olcum yoklugunu olcum gibi sunmaktir.",
            "Maskelenen ozelligin token'i SIFIRLANMAZ, DEGISTIRILIR — sifir da gecerli bir normalize degerdir; sifirlamak aga 'maskelendi'yi 'degeri 0' diye ogretir.",
            "Val bolmesi KRONOLOJIK olmalidir: look-ahead yasagi etiketsiz veride de gecerlidir.",
            "Ozellikler zaten birbirinden bagimsizsa (iyi muhendislik edilmis vektor) rekonstruksiyon ogrenecek bir sey bulamaz — Tur-35'te canli 1m setinde tam olarak bu gozlendi.",
        ],
        iliskili=["rejim", "drift_detection"],
        ders_ref=["ACI-5"],
    ),
    Concept(
        id="quantile_regresyon",
        ad="Quantile (Pinball) Regresyonu — Dağılımsal Tahmin",
        tanim=(
            "Pinball kaybi L_tau(e) = max(tau*e, (tau-1)*e), e = y - q_tau, minimize edildiginde "
            "q_tau kosullu dagilimin tau-quantile'ina yakinsar (Koenker & Bassett 1978) — MSE'nin "
            "kosullu ORTALAMAyi vermesi gibi. Nokta tahmini dagilimi gizler; quantile demeti "
            "araligi soyler. Caprazlama (q_0.9 < q_0.1) pinball kaybiyla CEZALANDIRILMAZ cunku "
            "her quantile AYRI olculur — kisit kayba degil MIMARIYE yazilmalidir (kumulatif "
            "softplus). Uygulama: nihai.ai.neural (`vol_q` basligi, `predict_vol_quantiles`); "
            "formul koprusune EKLENMEZ cunku kopru torch importunu tetiklemez."
        ),
        formul_ref=[],
        tuzaklar=[
            "Kumulatif kurulumun garantisi AZALMAYAN'dir, 'kesin artan' DEGIL: softplus float32'de 0.0'a alt-tasinca komsu quantile'lar esitlenebilir (sira bozulmaz).",
            "Kalibre quantile bir KENAR iddiasi degildir: sentetik veride bilinen dagilimin geri kazanilmasi canli oynakligin tahmin edilebildigi anlamina gelmez.",
            "Ampirik kapsama olculmeden 'quantile dogru' denemez — mimari caprazlamayi engeller, DOGRULUGU garanti etmez.",
            "Quantile hedefi HAM olcektedir (vol hedefi normalize edilmez); cagiran hangi buyuklugu verdiyse onu geri okur.",
        ],
        iliskili=["realized_vol", "ewma_vol", "vol_skew"],
        prereqs=["realized_vol"],
    ),
    Concept(
        id="cevrimici_uzman_birlestirme",
        ad="Çevrimiçi Uzman Birleştirme (Hedge + fixed-share)",
        tanim=(
            "K olasilik kaynagi her adimda agirlikli harmanlanir; agirliklar o adima KADARKI "
            "kayiptan gelir (prequential, sizinti yok): w_i <- w_i * exp(-eta * kayip_i). "
            "fixed-share her adimda agirligin alpha'sini yeniden dagitir → rejim degisiminde "
            "olu uzman dirilebilir. Teorik garanti EN IYI UZMANA YAKIN KALMAKTIR, onu gecmek degil."
        ),
        formul_ref=["hedge_combine", "running_base_rate", "source_fusion"],
        tuzaklar=[
            "Uzmanlar AYNI olayi yanitlamalidir; kaynak karsi tarafi soyluyorsa olasilik 1-p ile cevrilmelidir — hizalama yapilmadan birlesim anlamsizdir.",
            "'Birlesim en iyi tek uzmandan iyi' beklentisi YANLISTIR: pismanlik (regret) pozitif olabilir ve bu Hedge'in kusuru degil TANIMIDIR.",
            "Kosan taban orani UCUNCU uzman olarak konmadan sonuc okunamaz (ACI DERS #9); ayrica farkin GURULTU olup olmadigi eslestirilmis t ile sinanmalidir.",
            "Son agirliklar bir KENAR sinyali degildir: en iyi uzmanin agirliginin buyumesi o uzmanin bilgi tasidigini degil, digerlerinden AZ KOTU oldugunu gosterir.",
        ],
        iliskili=["rejim", "drift_detection", "rejim_gecis_matrisi"],
        ders_ref=["ACI-9"],
    ),
    Concept(
        id="kanit_temelli_belirsizlik",
        ad="Kanıt-Temelli Belirsizlik (Dirichlet / EDL)",
        tanim=(
            "Siniflandirici softmax OLASILIK yerine Dirichlet KANITI uretir: alpha = softplus(raw)+1 "
            "(kanit e = alpha-1 >= 0). Beklenen olasilik p_k = alpha_k / S (S = toplam alpha) ve "
            "BOSLUK (vacuity) u = K / S; u->1 'hic kanit yok', u->0 'bol kanit'. Kayip Bayes-riski "
            "capraz-entropi (Sensoy ve ark. 2018) + YANILTICI kaniti duz Dirichlet'e ceken KL "
            "cezasidir; KL katsayisi TAVLANMALIDIR (kanit olusmadan cezalandirilirsa model hic kanit "
            "uretmemeyi ogrenir). Ayirt edici ozellik: p ile u BAGIMSIZ olabilir — p=0,5 + bol kanit "
            "'gercekten kararsiz', p=0,5 + kanit yok 'bilmiyorum' demektir; MC-dropout bu ikisini "
            "ayirmaz. Uygulama: nihai.ai.neural (`dir_evid` basligi, `predict_evidential`)."
        ),
        formul_ref=[],
        tuzaklar=[
            "u ile MC-dropout std'si AYNI SAYI DEGILDIR: canli olcumde korelasyon −0,386 cikti; biri kanit MIKTARINI, digeri orneklem SACILIMINI olcer.",
            "Bir belirsizlik olcusunun YONU varsayilamaz: 'yuksek u = daha cok yanilma' iddiasi DILIM ISABETIYLE sinanmalidir (olculdu: u 5/5 seed dogru, MC-dropout std 0/5 TERS).",
            "Kanit basligi ana gorevi ELE GECIRMEZ: p_win yine `direction` basligindan ve sicaklik kalibrasyonuyla okunur; sicaklik alpha'ya UYGULANMAZ (alpha'yi bolmek belirsizligi bozar).",
            "Belirsizligin dogru siralanmasi bir KENAR degildir: ayirt etme gucu (AUC) degismeden 'hangi tahmine guvenilmez' sorusu ucuzlar, o kadar.",
        ],
        iliskili=["brier", "quantile_regresyon", "maskeli_on_egitim"],
        ders_ref=["ACI-3"],
    ),
]

CONCEPTS: dict[str, Concept] = {c.id: c for c in _CONCEPT_LIST}


# ---------------------------------------------------------------------------
# Sorgu API'si
# ---------------------------------------------------------------------------
def explain(concept_id: str) -> str | None:
    """Kavramı Türkçe insan-okunur bloğa çevir: tanım + formül köprüsü + tuzaklar + ders
    bağları + ilişkili kavramlar. Bilinmeyen id → None (savunmacı; çökme yok)."""
    c = CONCEPTS.get(concept_id)
    if c is None:
        return None
    lines = [f"{c.ad} ({c.id})", "", f"Tanım: {c.tanim}"]
    formul_txt = ", ".join(c.formul_ref) if c.formul_ref else "(yok — kavramsal/metodolojik kavram)"
    lines.append(f"Formül köprüsü (nihai.ai.quant_math): {formul_txt}")
    if c.tuzaklar:
        lines.append("Tuzaklar:")
        lines.extend(f"  - {t}" for t in c.tuzaklar)
    ders_txt = ", ".join(c.ders_ref) if c.ders_ref else "(yok)"
    lines.append(f"Ders bağları: {ders_txt}")
    iliskili_txt = ", ".join(c.iliskili) if c.iliskili else "(yok)"
    lines.append(f"İlişkili kavramlar: {iliskili_txt}")
    return "\n".join(lines)


def formula_for(concept_id: str) -> list[str]:
    """concept_id → nihai.ai.quant_math fonksiyon adları (kopya liste — çağıran değiştirse
    kaynak tablo etkilenmez). Bilinmeyen id → [] (savunmacı; çökme yok)."""
    c = CONCEPTS.get(concept_id)
    return list(c.formul_ref) if c is not None else []


def search(query: str) -> list[Concept]:
    """ad/tanım/id alanlarında küçük-harf alt-dizge (substring) araması. Boş/boşluk sorgu
    → [] (bilinçli: 'her şeyi döndür' sürprizi yaratmaz)."""
    q = (query or "").strip().lower()
    if not q:
        return []
    return [
        c for c in _CONCEPT_LIST
        if q in c.ad.lower() or q in c.tanim.lower() or q in c.id.lower()
    ]


def glossary() -> list[str]:
    """Tüm kavram id'leri, alfabetik sırayla."""
    return sorted(CONCEPTS.keys())


def curriculum() -> list[str]:
    """`prereqs` alanına göre DETERMİNİSTİK topolojik öğrenme sırası (v3, WP-K).

    Kahn-benzeri katmanlı algoritma: her turda TÜM önkoşulları zaten çözülmüş kavramlar
    alfabetik sırayla eklenir (sabit tohum/rastgelelik YOK — aynı CONCEPTS girdisi her
    zaman AYNI sırayı üretir). Bilinmeyen (CONCEPTS'te olmayan) prereqs id'si sessizce yok
    sayılır (savunmacı — kırık bir referans curriculum'u çökertmez, yalnız o önkoşulu
    'zaten karşılanmış' sayar). CONCEPTS bir DAG DEĞİLSE (çevrim varsa) — hiçbir kavram
    TÜM önkoşulları çözülmüş duruma gelemeyeceği için ilerleme durur — ValueError.
    """
    graph: dict[str, list[str]] = {
        cid: [p for p in c.prereqs if p in CONCEPTS] for cid, c in CONCEPTS.items()
    }
    resolved: list[str] = []
    resolved_set: set[str] = set()
    remaining: set[str] = set(graph)
    while remaining:
        ready = sorted(
            cid for cid in remaining if all(p in resolved_set for p in graph[cid])
        )
        if not ready:
            raise ValueError(
                "knowledge.curriculum: prereqs grafiğinde ÇEVRİM tespit edildi (DAG değil)"
            )
        for cid in ready:
            resolved.append(cid)
            resolved_set.add(cid)
            remaining.discard(cid)
    return resolved
