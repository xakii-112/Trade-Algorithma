# 📊 Nihai Algoritma — Premium Kontrol Paneli Rehberi

**8 Eylül 2026 güncellemesi:** Laboratuvar sekmesi, kanıt değerlendirme, eğitim, servis durumu, bakım ve veri güncelleme kontrolleri eklendi. Hangi özelliğin nerede olduğu, yeniden başlatma gereksinimleri ve doğrulama sonuçları için [arayüz tamamlama kaydına](docs/WEB_ARAYUZ_TAMAMLAMA.md) bakın. Sistem sinyal ve sanal portföy üretir; broker'a emir göndermez.

Nihai Algoritma trading sistemi için tasarlanan ve geliştirilen **Premium Web Arayüzü** tamamlanmıştır. Bu arayüz, standart bir veri listeleme panelinin ötesine geçerek en son modern web tasarım trendlerini (glassmorphic derinlik, neon durum ışıkları, dinamik SVG fiyat skalaları, interaktif log analizörleri) bünyesinde barındıran, tam yetkili bir **Operatör Komuta Merkezi** olarak tasarlanmıştır.

Aşağıda, yeni arayüzün mimari detayları, görsel bileşenleri ve operatör yönetişim araçları listelenmiştir.

---

## 🎨 Tasarım Estetiği ve Görsel Mimari
Panel, yüksek performanslı ve modern bir kullanıcı deneyimi (UX) sunmak üzere sıfırdan tasarlanmıştır:
- **Glassmorphism (Cam Estetiği):** Kartlar ve tablolar, `backdrop-filter: blur` ve sınır geçişli yarı saydam gradyan sınırları kullanılarak şeffaf, derinlik hissi yüksek yüzeyler olarak kurgulanmıştır.
- **Dinamik Neon Göstergeler:** Canlı bağlantı, seans durumu (RTH) ve Kill-Switch gibi kritik sistem durumları, neon parlamalı ve ritmik nefes alma (`breathe`) animasyonlu renk kodlarıyla canlı olarak yansıtılır.
- **Futuristik Tipografi:** Başlıklar ve yapısal metinler için premium **Outfit** fontu tercih edilmişken; sayısal veriler, finansal değerler ve log akışları için yüksek kontrastlı **JetBrains Mono** monospace fontu entegre edilmiştir.

---

## 🚀 Ana Panel Sekmeleri ve Gelişmiş Özellikler

Arayüz 5 ana fonksiyona bölünmüş sekme yapısıyla kolayca yönetilebilir:

### 1. 📊 Genel Bakış (Overview Dashboard)
* **Kasa Değeri (NLV) Kartı:** Güncel kasa bakiyesi, nakit rezervi ve tüm zamanların zirve (ATH) değeri yan yana gösterilir.
* **Günlük Kar/Zarar (PnL) Kartı:** Kar durumunda neon yeşil, zarar durumunda ise neon kırmızı renk şemasına otomatik olarak bürünerek anlık durumu dramatik şekilde yansıtır.
* **Kapasite ve Pozisyon Limiti:** Portföydeki aktif pozisyon sayısının maksimum sınıra oranını dinamik bir doluluk barı üzerinden gösterir.
* **Kill-Switch Emniyet Kilidi:** Rolling expectancy negatif değere ulaştığında tetiklenen kill-switch durumunu neon kırmızı uyarıyla raporlar.
* **İkincil İstatistikler:** Gün-içi al-sat limit takibi (PDT), üst üste stop limiti takibi ve T+1 bloke nakit verileri.

### 2. 🛡️ Açık Pozisyonlar ve Fiyat Skalası (Visual Position Tracker)
* Sıradan tablolar yerine, her pozisyon için **Giriş - Mevcut Fiyat - Stop Loss - Kar Hedefi** dengesini interaktif bir sürgülü gösterge üzerinden görüntüler.
* Pozisyonun hedefe veya stop noktasına yakınlığına göre gösterge çizgileri anlık renk değiştirerek risk durumunu operatöre sezgisel olarak aktarır.
* **Münferit Kapatma Yetkisi:** Her pozisyonun yanındaki **"Kapat"** butonu, onay sonrasında sanal pozisyonu mevcut fiyatla kapatır ve portföy kaydını günceller.

### 3. 🔍 İzleme Listesi & Sinyal Akışı
* **Dinamik Sinyal Timeline'ı:** Algoritmanın tarama esnasında ürettiği sinyalleri, block/cooldown kararlarını ve sistem durumlarını renk kodlu (SIGNAL, SKIP, CLOSE, BLOCK) özel akış kartlarıyla sunar.
* **Sinyal Arama Barı:** Operatör, yüzlerce tarama kaydı arasından sadece belirli bir sembolü (örn. "AAPL") veya kararı ("SKIP") süzmek için gerçek zamanlı arama yapabilir.
* **Sektörel Watchlist Görünümü:** Watchlist'teki hisselerin durumunu (Pozisyonda / Boşta) gösterir.

### 4. 📈 Yunan Parametreleri ve Sektör Kümelenmesi (Greeks & Risk)
* **Greek Kapasite Kartları:** Portföyün toplam Delta (Δ), Theta (Θ), Gamma (Γ) ve Vega (𝒱) risklerini, `config.py` limitleriyle karşılaştırarak progress bar ve doluluk oranıyla görselleştirir. Limit aşımında kartlar kırmızı uyarı moduna geçer.
* **Sektör Kümelenme Grafiği (Madde 12, 14):** Korelasyon riski oluşturmaması adına aynı sektörden maksimum 3 pozisyon açılması kuralını izler ve sektörel dağılımı görselleştirir.

### 5. 📝 Günlük Defteri (Trade Post-Mortem & Logs)
* **İşlem Otopsisi Analizörü (Trade Post-Mortem):** Kapanan işlemlerin yer aldığı `trade_log.txt` dosyasını otomatik olarak parse eder. Operatöre toplam kapanan işlem sayısı, Win Rate (Başarı Oranı), Kazanç/Kayıp oranı ve toplam net kar gibi kritik istatistikleri ve şık işlem kartlarını sunar.
* **Hatalar ve Konsol Akışı (system.log):** `system.log` dosyasının son 100 satırını tarayarak sistem kararlılığını tek panelden izleme imkanı sunar.

### 6. ⚙️ Operatör Kontrol Paneli & Acil Müdahale (Flatten All)
* Sistem yapılandırma parametrelerini tek ekranda özetler.
* **BÜYÜK KIRMIZI BUTON (FLATTEN ALL):** Madde 47 gereğince, tüm açık pozisyonları anında piyasa fiyatından kapatan, emniyet loglarını yazan ve sistemi korumaya alan acil durum mekanizmasını tetikler. Süreç, çift aşamalı bir doğrulama modalı ile korunmaktadır.

---

## 🛠️ Nasıl Çalıştırılır?

Geliştirilen bu arayüz, projenizin mevcut web servis mekanizmasıyla tam uyumludur. Çalıştırmak için:

```powershell
# Web arayüzü ile birlikte canlı tarama döngüsünü başlatmak için:
python -m nihai.execution.main --serve-web
```

FastAPI sunucusu arka planda bağımsız bir thread olarak başlatılacak ve tarayıcınızdan **`http://127.0.0.1:8765`** adresinden panele erişebileceksiniz.

> [!IMPORTANT]
> **Güvenlik Notu:** Kapatma ve Flatten All komutları doğrudan sanal portföy durumunu günceller ve kalıcı diske (`state.json`) işler. Bu sebeple canlı entegrasyonda yetkisiz kişilerin erişimine karşı dikkatli olunmalıdır.
