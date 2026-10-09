"""observability.py — sistem metrikleri + API hız BÜTÇESİ (Tur-58, Madde 37/38/48).

Dış rapor §H üç boşluk sayıyor; ikisi bu modülde kapanır:

* **"API Rate-Limit Monitörü Yok."** Yarısı yanlıştı: `execution/main.py` çağrı/dk'yı
  ZATEN loglıyordu (`[rate] alpaca … | limit 200/dk`). Eksik olan yarı, o sayının
  MAKİNE tarafından okunması ve limite yaklaşıldığında tarama hızının KISILMASIYDI —
  bu depoda tanıdık kusur sınıfı: *yazılıp okunmayan ölçüm* (Tur-38).
* **"Sistem Metrikleri Yok."** Döngü gecikmesi, RSS/CPU ve sağlayıcı sayaçları hiçbir
  makine-okunur yüzeyde değildi. `prometheus_text()` bunları standart biçimde basar;
  panoya `/metrics` olarak bağlanır. YENİ BAĞIMLILIK YOK — biçim düz metindir.

TASARIM KARARLARI

* **Kısma yalnız YUMUŞAK TAVANIN ÜSTÜNDE çalışır.** `throttle_sleep()` kullanım oranı
  `ALPACA_RATE_SOFT_PCT` (vars. %80) altındayken **tam 0.0** döner → bugünkü rejimde
  davranış bit-denktir ve bu testle kilitlidir. Bir hız sınırlayıcının varsayılan
  KAPALI olması, onu hiç yazmamakla aynı şeydir; bu yüzden açıktır ama yalnız kenarda
  konuşur.
* **Ölçüm penceresi MONOTONİKTİR** (`time.monotonic`): duvar saati DST'de/NTP
  düzeltmesinde geriye gidebilir ve negatif bir pencere sonsuz bir "çağrı/dk" üretirdi.
* **Ölçüm yokluğu sıfır DEĞİLDİR.** İlk çağrıda pencere yoktur → `rate_per_min=None`,
  `usage_pct=None` (0.0 değil, yoksa pano "hiç çağrı yok" sanırdı — Tur-31/33).
* **psutil OPSİYONELDİR.** Yoksa süreç metrikleri None döner, geri kalanı çalışır.
* **WebSocket sağlığı KAYIT ile gelir.** Sağlayıcıyı KURAN modül `set_provider()` ile
  bildirir; observability kendi sağlayıcısını KURMAZ (kurarsa ikinci bir Alpaca bağlantısı
  açardı — hub'ın tek-upstream amacının tersi). Kayıt yoksa akış metrikleri BASILMAZ.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

try:                                              # opsiyonel — yoksa süreç metrikleri None
    import psutil
    _PROC: psutil.Process | None = psutil.Process()
except Exception:                                 # noqa: BLE001
    psutil = None
    _PROC = None

_T0 = time.monotonic()


def _envf(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# API hız bütçesi
# ---------------------------------------------------------------------------
@dataclass
class RateBudget:
    """Kayan pencereli çağrı/dk ölçümü + yumuşak tavana göre kısma önerisi.

    Sayaç kaynağı `AlpacaProvider._calls` (sınıf düzeyi, thread-safe): tüm örneklerin
    GERÇEK REST çağrılarını toplar. Burada ikinci bir sayaç TUTULMAZ — iki sayaç iki
    farklı gerçeklik demek olurdu."""

    limit_per_min: float = field(default_factory=lambda: _envf("ALPACA_RATE_LIMIT_PER_MIN", 200.0))
    soft_pct: float = field(default_factory=lambda: _envf("ALPACA_RATE_SOFT_PCT", 80.0))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last_total: int | None = None
    _last_mono: float = 0.0
    _rate: float | None = None

    def _counts(self) -> dict:
        try:
            from nihai.data.alpaca_provider import AlpacaProvider
            return AlpacaProvider.call_counts()
        except Exception:                          # noqa: BLE001 — SDK yoksa bütçe sessiz
            return {"total": 0}

    def observe(self) -> float | None:
        """Pencereyi ilerlet, ölçülen çağrı/dk'yı döndür (ilk çağrıda None)."""
        total = int(self._counts().get("total", 0))
        now = time.monotonic()
        with self._lock:
            prev_total, prev_mono = self._last_total, self._last_mono
            self._last_total, self._last_mono = total, now
            if prev_total is None:
                return None
            dt = now - prev_mono
            if dt <= 0:                            # aynı an içinde iki çağrı → ölçüm yok
                return self._rate
            self._rate = max(total - prev_total, 0) / dt * 60.0
            return self._rate

    def usage(self) -> dict:
        """{'rate_per_min', 'limit_per_min', 'usage_pct', 'soft_pct', 'cumulative'}."""
        c = self._counts()
        rate = self._rate
        pct = (rate / self.limit_per_min * 100.0) if (rate is not None and self.limit_per_min > 0) else None
        return {"rate_per_min": rate, "limit_per_min": self.limit_per_min,
                "usage_pct": pct, "soft_pct": self.soft_pct, "cumulative": c}

    def throttle_sleep(self, cycle_sec: float) -> float:
        """Bu döngüye EKLENECEK saniye. Yumuşak tavanın altında **tam 0.0**.

        Üstündeyse, aynı çağrı sayısını yumuşak tavanda taşıyacak döngü süresine kadar
        uzatır: `ek = cycle · (oran/tavan − 1)`. Üst sınır bir tam döngüdür — kısma bir
        FREN'dir, durdurma değil (tarama tamamen durursa stop/target yönetimi de durur).
        """
        rate = self._rate
        if rate is None or self.limit_per_min <= 0 or cycle_sec <= 0:
            return 0.0
        soft = self.limit_per_min * max(self.soft_pct, 0.0) / 100.0
        if soft <= 0 or rate <= soft:
            return 0.0
        return min(cycle_sec * (rate / soft - 1.0), cycle_sec)


BUDGET = RateBudget()


# ---------------------------------------------------------------------------
# Süreç / döngü metrikleri
# ---------------------------------------------------------------------------
_LOOP: dict[str, float] = {}
_LOOP_LOCK = threading.Lock()
_PROVIDER = None            # akış karnesini okumak için (bkz. set_provider)


def set_provider(provider) -> None:
    """Akış sağlığını okuyacağı sağlayıcıyı bildir (sağlayıcıyı KURAN modül çağırır).

    Rapor §H'nin üçüncü kalemi ("WebSocket sağlığı Prometheus formatında sunulmuyor")
    ancak bu kayıtla kapanır: karne `alpaca_stream_provider.stream_status()`tedir ve
    sarmalayıcı zincirinin arkasındadır; observability onu KENDİ kurduğu bir sağlayıcıdan
    okusaydı ikinci bir WebSocket bağlantısı açardı."""
    global _PROVIDER
    _PROVIDER = provider


def stream_health() -> dict | None:
    """Kayıtlı sağlayıcının akış karnesi; kayıt/akış yoksa **None** (0 DEĞİL)."""
    if _PROVIDER is None:
        return None
    try:
        from nihai.data.data_provider import stream_status
        return stream_status(_PROVIDER)
    except Exception:                              # noqa: BLE001 — teşhis katmanı
        return None


def record_loop(name: str, seconds: float) -> None:
    """Bir döngü turunun süresini kaydet (son değer; histogram TUTULMAZ — YAGNI)."""
    with _LOOP_LOCK:
        _LOOP[name] = float(seconds)


# Saat kayması ÖNBELLEĞİ (Madde 242). `clock_drift()` bir UDP turu ödeyen ağ
# çağrısıdır; `/metrics` her kazımada onu ödeyemez (Tur-59 BULGU-59-8: ölçüm
# bütçesini OKUYAN da öder). Bu yüzden okuyucu ÖNBELLEĞİ okur, doldurmak AYRI
# ve bilinçli bir adımdır (`refresh_clock_drift()`), tıpkı `float_shares(fetch=False)`.
_CLOCK: dict | None = None
_CLOCK_TS: float = 0.0
CLOCK_TTL_SEC = float(_envf("CLOCK_DRIFT_TTL_SEC", 3600.0))


def refresh_clock_drift(force: bool = False) -> dict | None:
    """NTP saat kaymasını ÖLÇ ve önbelleğe yaz (TTL dolmadıysa yeniden ölçmez)."""
    global _CLOCK, _CLOCK_TS
    if not force and _CLOCK is not None and (time.monotonic() - _CLOCK_TS) < CLOCK_TTL_SEC:
        return _CLOCK
    try:
        from nihai.audit_chain import clock_drift
        _CLOCK = clock_drift()
    except Exception:                              # noqa: BLE001 - fail-soft
        # loguru BİLEREK import EDİLMEZ: bu modül `/metrics` yolundadır ve
        # `clock_drift` zaten kendi `measured=False` sözleşmesini taşır.
        return _CLOCK
    _CLOCK_TS = time.monotonic()
    return _CLOCK


def clock_drift_cached() -> dict | None:
    """Önbellekteki saat kayması; HİÇ ölçülmediyse None (0,0 UYDURULMAZ)."""
    return _CLOCK


def snapshot() -> dict:
    """Makine-okunur sistem karnesi. Hiçbir alan uydurulmaz; ölçülemeyen None'dır."""
    rss = cpu = None
    if _PROC is not None:
        try:
            rss = float(_PROC.memory_info().rss)
            cpu = float(_PROC.cpu_percent(interval=None))
        except Exception:                          # noqa: BLE001
            pass
    with _LOOP_LOCK:
        loops = dict(_LOOP)
    return {"uptime_sec": time.monotonic() - _T0, "rss_bytes": rss, "cpu_pct": cpu,
            "loop_seconds": loops, "rate": BUDGET.usage(), "stream": stream_health(),
            "clock": clock_drift_cached()}


def _line(out: list[str], name: str, help_: str, typ: str, value, labels: str = "") -> None:
    """Ölçülemeyen metrik BASILMAZ — Prometheus'ta 0, 'ölçüm yok' DEĞİL 'sıfır'dır."""
    if value is None:
        return
    # Tur-106 (BULGU-106-14): HELP/TYPE metrik adı başına BİR kez (metin biçimi kuralı) — etiketli her örnekte
    # tekrarlanınca katı ayrıştırıcılar (promtool, expfmt) sergiyi "second TYPE line" ile reddeder.
    if f"# TYPE {name} {typ}" not in out:
        out.append(f"# HELP {name} {help_}")
        out.append(f"# TYPE {name} {typ}")
    out.append(f"{name}{labels} {float(value):g}")


def prometheus_text() -> str:
    """Prometheus/OpenTelemetry uyumlu düz-metin sergisi (`text/plain; version=0.0.4`)."""
    s = snapshot()
    out: list[str] = []
    _line(out, "nihai_uptime_seconds", "sürecin ayakta kalma süresi", "gauge", s["uptime_sec"])
    _line(out, "nihai_process_resident_bytes", "süreç RSS", "gauge", s["rss_bytes"])
    _line(out, "nihai_process_cpu_percent", "süreç CPU %", "gauge", s["cpu_pct"])
    r = s["rate"]
    _line(out, "nihai_api_calls_per_minute", "Alpaca REST çağrı hızı", "gauge", r["rate_per_min"])
    _line(out, "nihai_api_rate_limit_per_minute", "yapılandırılmış limit", "gauge", r["limit_per_min"])
    _line(out, "nihai_api_rate_usage_percent", "limitin yüzde kaçı", "gauge", r["usage_pct"])
    for k, v in (r["cumulative"] or {}).items():
        if k == "total":          # toplam `sum()` ile alınır; etiket olarak eklenirse aile toplamı iki katı çıkar
            continue
        _line(out, "nihai_api_calls_total", "kümülatif REST çağrısı", "counter", v,
              labels=f'{{method="{k}"}}')
    for k, v in s["loop_seconds"].items():
        _line(out, "nihai_loop_seconds", "son döngü turu süresi", "gauge", v,
              labels=f'{{loop="{k}"}}')
    st = s["stream"]
    if st is not None:
        # Akış YOKSA bu satırlar hiç basılmaz: "0 bağlı" ile "akış sağlayıcısı yok"
        # aynı şey değildir (Tur-31/33/56 sözleşmesi).
        _line(out, "nihai_stream_connected", "WebSocket bağlı mı (1/0)", "gauge",
              1 if st.get("connected") else 0)
        for alan, ad in (("subscribed", "abone sembol"),
                         ("symbols_with_minutes", "canlı 1m barı gelen sembol"),
                         ("symbols_with_quotes", "canlı kotasyonu gelen sembol"),
                         ("symbols_with_trades", "canlı işlemi gelen sembol")):
            v2 = st.get(alan)
            _line(out, f"nihai_stream_{alan}", ad, "gauge",
                  len(v2) if isinstance(v2, (list, tuple, set)) else None)
    ck = s.get("clock")
    if ck is not None and ck.get("measured"):
        # HİÇ ölçülmemişse satır BASILMAZ: "0 ms kayma" ile "ölçmedim" aynı şey değil
        # (Tur-31/33/56 sözleşmesi; akış satırlarıyla aynı kural).
        _line(out, "nihai_clock_offset_ms", "NTP saat kayması (ms)", "gauge",
              ck.get("offset_ms"))
        _line(out, "nihai_clock_roundtrip_ms", "NTP gidiş-dönüş süresi (ms)", "gauge",
              ck.get("roundtrip_ms"))
        _line(out, "nihai_clock_drift_warn", "kayma eşiği aşıldı mı (1/0)", "gauge",
              1 if ck.get("warn") else 0)
    return "\n".join(out) + ("\n" if out else "")
