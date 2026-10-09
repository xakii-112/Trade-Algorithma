# Web arayüzü eksikliklerinin tamamlanması — 8 Eylül 2026

Kaynak: `proje arayüz eksiklikleri.txt`. Var olan hesaplama motorları kullanıldı; eksik çalıştırma, durum ve raporlama bağlantıları eklendi. Mevcut Quant araçları ve operasyon kartları korundu.

| Rapordaki madde | Arayüzde karşılığı |
| --- | --- |
| 2 · Backtest | **Laboratuvar:** semboller, tarih/dönem/aralık, strateji, ATR, sermaye, tek sembol/ortak portföy/sentetik opsiyon/walk-forward. Bakiye, Monte Carlo bantları ve IS/OOS grafikleri; iş günlüğü ve JSON indirme. |
| 3 · Araştırma | **Kanıt & Doğrulama:** Edge Lab defteri ve tarih kapsamı beyanlı hipotez testi. **Laboratuvar:** DSR/PBO, Purged CV/özellik önemi, PIT, PEAD, ablasyon, mutabakat, model uzlaşması, stop/hedef ve isabet ızgaraları, ufuk/kapasite, veri A/B, çiftler, kesitsel ve küçük şirket momentumu. |
| 4 · Model fabrikası | **Laboratuvar:** meta/nöral eğitim; nöral epoch, son minibatch ve doğrulama kaybı; CV raporu ve ayrı aday artefaktları. Canlı yeniden eğitimde mevcut örneklem/time/symbol/purged kapıları; isteğe bağlı, onaylı model kaydı. Gölge terfi kanıtı incelemesi. |
| 5 · MainHub | **Sistem:** abonelikler, mesaj sayaçları, son veri zamanı, tampon doluluğu, bağlantı girişimleri; başlat/durdur/yeniden başlat/eşitle. |
| 6 · Watchdog/deadman | **Sistem:** `NihaiAlgo-*` zamanlanmış görevleri, son/sonraki koşu, çıkış kodu, log kuyrukları; kanıt tazeliği doğrulanırsa manuel dış ping. |
| 7 · Denetim/NTP | **Sistem:** zincir doğrulaması, ilk bozuk halka, son hash/Merkle kökü; düğmeyle NTP ölçümü. Dosya/ağ yokluğu ölçüm gibi sunulmaz. |
| 8 · Vergi | **Günlük Defteri:** yıl filtresi, kısa/uzun vade ve §1256 toplamları; sanal işlemler için ayrı 8949/6781 CSV şablonları. |
| 9 · Log bakımı | **Sistem:** silinecek/sıkıştırılacak dosyaların önizlemesi; aynı önizleme hâlâ geçerliyse onaylı uygulama. |
| 10 · Telegram | **Sistem:** yerel gönderim/polling geçmişi, son komutlar, şablonlar; isteğe bağlı bağlantı sorgusu ve onaylı test bildirimi. Tokenlar gösterilmez. |
| 11 · Oto-tarama | **İzleme Listesi:** evren/Top-N seçimi, arka plan taraması, son sıralama tablosu. İki katmanda kalan evren korunur; seçilen odak sınırı canlı süreçlerce okunur. |
| 12 · Dinamik strateji | **Operatör:** sonraki sinyaller için strateji/oy/ATR seçimi, katmanların sembol ve süreleri, SSR uyarıları ve HALT/LULD engelleri. |
| 13 · Dış veri | **Laboratuvar / Veri kaynakları:** DoltHub, OPRA arşivi, float ısıtma, EDGAR, Fama–French/momentum ve FRED. Dolt opsiyon zinciri OneDrive dışına alınır; başarısız yenileme eski önbelleği başarılı indirme gibi sunmaz. |

## Rapordaki iddiaların düzeltilmesi

- SSR mevcut kodda **uyarıdır**; engel olarak etiketlenmedi. HALT/LULD engel kayıtları ayrı gösterilir.
- `shadow_promote` kanıt değerlendirmesidir; canlı kuralları kendiliğinden etkinleştirmez.
- PBO motoru en az **beş** hizalı strateji sütunu ister. Daha az sütunda yalnız DSR hesaplanır. Mevcut kanıt eşikleri düşürülmedi.
- Sentetik opsiyon testi tarihsel opsiyon zinciri kullanmaz. Vergi dosyaları sanal işlemlerin şablon dökümüdür; wash-sale adayları otomatik vergi düzeltmesi değildir.

## Çalıştırma ve sınırlar

Değişen Python kodunu almak için mevcut başlatıcı üzerinden panel/canlı tarayıcı ve MainHub yeniden başlatılmalıdır. Eski hub sürümü süreç kimliği vermiyorsa panel onu durdurmayı reddeder; ilk açılış yerel başlatıcıyla yapılır.

Güncel canlı süreçler meta model dosyası değişince yeniden yükler. Eğitim kaydı tek başına kapıyı enforce yapmaz. Web terfisi modelin SHA-256 kimliğine, stratejiye ve bar aralığına bağlıdır; model değişirse eski terfi izni yeni modele geçmez. Nöral aday dosyaları ilgili işin altında tutulur.

## Ayrıntılı listedeki kalan yedi madde

| Madde | Tamamlanan bağlantı |
| --- | --- |
| 12 · Gölge model terfisi | Laboratuvar: gölge meta modeli için kanıt inceleme, uygun sonuçtan canlıya terfi ve gölgeye dönüş. Sunucu terfi anında kanıtı tekrar hesaplar. Aynı model kimliği/strateji/bar aralığı; mevcut gölge kapıları, sabit eşik, OOS ve ortak fiyat referansı aranır. Canlı süreçlerin uyguladığı kip ve zaman ayrıca gösterilir. |
| 14 · Ham tick gecikmesi | Sistem: quote/trade olay damgasından callback alımına milisaniye farkı; son değer, p50/p95/azami, örneklem ve eksik damga sayısı. Son 512 tick; bar damgaları bu ölçüme katılmaz. Negatif değerler saat kaymasını gizlememek için korunur. |
| 15 · Deadman | Saniyelik kalan süre, 15 saniyelik sağlık yenilemesi ve tüm sekmelerde görünen alarm. Kanıt gecikmesi alarmı seans/proje kapalıyken bastırılır; sağlık bağlantısının kaybı ayrıca bildirilir. Eksik dosya sıfır yaş sayılmaz. |
| 22 · Telegram grafiği | Sistem: sembol/aralık/bar sayısı seçimiyle gerçek PNG üretimi; iş sonucunda önizleme ve indirme. Ayrı onayla tam önizlenen görsel gönderilir; Telegram başarısızlığı başarı olarak gösterilmez. |
| 23 · Katman yükü | Operatör: odak/geniş katmanların son tamamlanan turdaki tarama ve bekleme süreleri grafiği, sembol sayısı ve kayıt zamanı. Sekme açıkken yenilenir; bu ölçü CPU yüzdesi değildir. |
| 25 · Nakit ve wash-sale | Risk: state dosyasındaki nakit çözülme tarihleri/tutarları ve sembol bazında son zarar, serbest gün ve kalan gün. Motorun mevcut kapsayıcı gün sınırı ve muafiyetleri korunur; wash-sale mevcut sistemde uyarıdır. |
| 26 · Trailing ve bütçe | Açık Pozisyonlar: hisse seçimiyle fiyat/aktif stop çizgisi, yeni durum geldikçe güncelleme ve son 120 gözlem. Tarihsel stop yolu uydurulmaz. Mevcut RateBudget kartı Sistem sekmesi açıkken düzenli yenilenir. |

Yeni terfi defteri `data/meta_activation.json`, süreç uygulama kayıtları `data/meta_applied_<pid>.json` altındadır. Önceki terfi kaydı ayrıca saklanır. **Eski canlı loglarda model kimliği yoktur**; yeni kodla yeniden başlayan kayıtçı kimlikli kanıt biriktirene ve kapılar geçilene kadar terfi düğmesi etkinleşmez. Kontrolün var olması, mevcut modelin terfiye uygun olduğu anlamına gelmez.

İşler `data/web_jobs/<kimlik>` altında ayrı süreçte çalışır; aynı panelde en çok iki iş, aynı türden en çok bir iş ve dört saat süre sınırı vardır. Sonuçlar panel yeniden açıldığında okunabilir. Önceki panelin henüz bitmemiş süreçleri için canlılık iddia edilmez; bu süreçlerin iptali yeni panelden yapılmaz. İptal yarım çıktı bırakabilir; tamamlanmış sonuç ve mevcut veri dosyası birbirine karıştırılmaz.

CSV seçimi mevcut veri diziniyle ve 250 MB ile sınırlıdır. Harici araçlar mevcut anahtar, abonelik ve veri önkoşullarını gerektirir; araç günlükleri başarısızlıkları ve ölçülemeyen sonuçları gösterir. Ayrı JSON raporu olmayan araştırma araçlarının ölçümleri iş günlüğündedir.

`MainHub` ortak sağlayıcı dosyaları `python -m tools.sync_mainhub --apply` ile eşitlendi. Yan projedeki `hub_server.py`, `start-hub.ps1` ve hub HTTP testi ayrıca güncellendi.

## Doğrulama

Son koşu: **4.077 test geçti, 1 test atlandı** (ana proje); **27 test geçti** (MainHub). `ruff check .` temiz, `mypy .` 204 kaynak dosyasında hatasız; JavaScript sözdizimi ve sözleşme kontrolü geçti. FastAPI/Starlette ve seaborn bağımlılıklarından kullanım dışı kalma uyarıları var; test hatası yok.

Testler geçici veri/log dizinleri ve sahte sağlayıcılarla çalışır. Yeni kontroller gerçek arka plan sürecini, dört backtest türünü, MC bantlarını, eğitim çıktısını, CSV ayırımını, girdi/erişim sınırlarını, bakım önizlemesini ve süreçler arası watchlist güncellemesini kapsar. Nöral ilerleme takibinin aynı seed ile eğitim sonucunu değiştirmediği kontrol edildi. Kalan yedi madde için ayrıca olay-zamanı gecikmesi, eksik/negatif damgalar, sonlu tampon, terfi/geri-alma/model değişimi, nakit/wash-sale sınır günü, aynı PNG'nin onaylı gönderimi ve JS sayaç/stop çizgisi kontrol edildi.

Çalıştırılabilir kontroller:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --tb=short
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider ..\MainHub\tests --rootdir=..\MainHub
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy .
node --check nihai/web/lab_dashboard.js
node tests/check_lab_dashboard.cjs
```

Yalıtılmış yerel HTTP sunucusunda sentetik CSV ile DSR/PBO başlatma → durum → sonuç akışı doğrulandı. Bu oturumda kullanılabilir tarayıcı bulunmadığından görsel yerleşim kontrolü yapılamadı; JavaScript kimlik/olay, grafik veri sözleşmeleri ve HTML kaçışı ayrı kontrol edildi.

Üretim servislerini durdurma/başlatma, canlı model terfisi, gerçek Telegram bildirimi, dış deadman ping'i ve büyük veri indirmeleri bu geliştirme doğrulamasında çalıştırılmadı.
