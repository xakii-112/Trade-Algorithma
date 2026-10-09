"""
neural_shadow.py — Nöral kapasite katmanının CANLI GÖLGE p_win kaydı (Tur-18).

Tur-17 raporu §3'ün açık kalemi: `nihai/ai/learning.py` zaten `neural_pwin` +
`neural_pwin_side` kolonlarını arıyor ve varsa `sources.neural` Brier'ini otomatik
dolduruyordu — ama bu kolonları YAZAN taraf yoktu (ölü köprü). Bu modül o tarafı kurar:
canlı aday sinyalin karar-barı özellik vektörünü kayıtlı bir `NeuralCapacity` modeline
verir, MC-dropout ile (p_win, std) döner.

ANAYASA (nihai/ai KARAR-DIŞI/GÖLGE) BOZULMAZ:
- Çıktı YALNIZ `predictions_log.csv`'ye yazılır. Sinyale, `pwin`'e, META_GATE'e,
  boyutlandırmaya, emre HİÇBİR yolla dokunmaz. `live_predict_service` değeri `_neural`
  altında taşır ve kayıttan sonra satırdan DÜŞÜRÜR (`_feats` deseni) → web JSON'una
  ve pano tablosuna sızmaz. Operatörün gördüğü tek yer, dürüstlük şeridi olan
  Beyin/öğrenme karnesidir (`sources.neural` Brier'i).
- Neden panoya konmuyor: kayıtlı modelin ÖLÇÜLEN OOS-AUC'si 0.500 (kart
  `1f072998…`, ACI DERS #5). İşlem tablosunda ikinci bir "P(win)" sütunu göstermek
  onu sinyal sanma riski taşır. Amaç kenar değil, kalibrasyonun CANLI ölçümüdür.

FAIL-SOFT: torch yok / kart yok / artefakt diskte yok / şema uyuşmuyor / çıkarım
hata verdi → `None` (kayıt boş kolonla geçer, tahmin satırı sağlam kalır). Pasiflik
nedeni `status()["reason"]`'da görünür (sessiz ölüm yasak — Tur-12 dersi).

MODEL SEÇİMİ (sessiz yanlış-model riskine karşı üç kilit):
  1. `NEURAL_SHADOW_MODEL` env verilmişse YALNIZ o base-path (operatör kontrolü).
  2. Yoksa registry kartları: `kind=="neural"` + artefaktı olan + `feature_names`
     güncel `FEATURE_NAMES` ile BİREBİR eşleşen + `data.source` "synthetic" ile
     BAŞLAMAYAN kartlar arasından en yeni `registered_at`.
     (Sentetik müfredat modeli canlı bara sokulmaz — kontrol koşusuydu, kenar değil.)
  3. Yükleme sonrası model nesnesinin kendi `feature_names`'i de FEATURE_NAMES ile
     karşılaştırılır (train/serve hizası; yeniden-sıralama = sessiz yanlış olasılık).

DAĞILIM-DIŞI (OOD) GUARD'ı — 2026-07-26 denetimi B-2:
Eski doygunluk guard'ı yalnız `std == 0 ve p uçta` vakasını yakalıyordu. Canlı ölçüm
(600/600 üretildi, 0 susturuldu, **p_win ortalaması 0.0585** — eğitim taban oranı 0.357)
guard'ın ASIL vakayı kaçırdığını gösterdi: model GÜNLÜK (1d) barlarda eğitilmiş, canlı
tahminler ise `1m` barlarda üretiliyor. Sonuç, ölçekleri uyuşmayan bir girdi vektörü:
`hour` özelliğinde |z| ≈ 26 (eğitim σ=0.485), `rvol` |z| ≈ 13; `atr_pct`/`bb_width`/
`range_pct` canlıda 20–36 kat küçük. Ağ doyuyor ama MC-dropout std'si 0 DEĞİL (≈0.034),
bu yüzden eski guard geçiriyordu. Loglansaydı `sources.neural` Brier ≈ 0.42 (bilgisiz
taban 0.25) çıkacak ve "model kenarsız" diye YANLIŞ teşhis edilecekti.

Yeni guard modelin KENDİ eğitim istatistiklerini (`feat_mu`/`feat_sigma`) kullanır:
`max|z| > NEURAL_SHADOW_MAX_Z` (varsayılan 6.0) → kaydedilmez + sayaç artar. Sıfır-varyanslı
özellikler (σ<1e-9; ör. `rr_ratio`) z hesabından DIŞLANIR — orada z tanımsızdır. Bu guard
zaman-dilimi uyumsuzluğunu KENDİLİĞİNDEN yakalar ve nedeni `status()` üzerinden görünür
kalır (sessiz ölüm yasak).

ARALIK TERCİHİ — 4. kilit (Tur-20, 2026-07-30):
Guard yeterli sanılmıştı ("ayrı bir interval kilidine gerek yok") — çünkü o sırada canlı
1m'de eğitilmiş HİÇBİR model yoktu ve susmak zaten en doğru sonuçtu. 1m kartı kaydedilip
köprü canlıya geçtikten sonra durum değişti: `_pick()` "en yeni uygun kart"ı seçtiği için
bir ARAŞTIRMA koşusu (ör. 1h/234k sette eğitilen P2-11 modeli) kaydedildiği anda en yeni
kart olur ve köprüyü sessizce yeniden dağılım-dışına düşürürdü — O4 düzeltmesi geri
alınırdı. Bu yüzden seçim artık ÖNCE canlı aralıkla (`PREDICT_INTERVAL`/`HTF_INTERVAL`)
eşleşen kartları süzer. Eşleşen yoksa eski davranış korunur ve guard nedenini söyleyerek
susar. İlke: "hiç seçilmesin" > "seçilip sussun".

.env:
    NEURAL_SHADOW=off            # gölge çıkarımı tamamen kapat (varsayılan: açık)
    NEURAL_SHADOW_MODEL=<base>   # uzantısız artefakt yolu (.json/.pt yanyana)
    NEURAL_SHADOW_MC_PASSES=20   # MC-dropout geçiş sayısı (1..200)
    NEURAL_SHADOW_DEVICE=auto    # auto: deep_mlp dışı basamak CUDA varsa GPU'da (Tur-93) | cpu | cuda
    NEURAL_SHADOW_MAX_Z=6.0      # OOD eşiği; 0/negatif → guard kapalı (önerilmez)
                                 # "auto" → eğitim |z| %99 kuyruğunun 1,5 katı (en az 4,0);
                                 #   eşik böylece modelin GÖRDÜĞÜ dağılıma ait olur
                                 #   (artefaktta q99 yoksa sessizce 6,0'a düşer)

TOPLULUK (v4/Tur-22): artefakt bir `NeuralEnsemble` manifestosuysa `load_any` onu yükler;
p_win üye ORTALAMASI, std ise toplam varyans yasasıyla (üye-içi MC + üye-arası yayılım)
gelir. Kod yolu değişmez — `status()["ensemble"]` üye sayısını gösterir. Topluluk da
KARAR-DIŞIDIR: "K üye hemfikir" bir kenar kanıtı değildir (hepsi aynı veriyi gördü).
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from nihai.config import DATA_DIR, ROOT
from nihai.console import ensure_utf8_stdout
from nihai.signals.meta_features import FEATURE_NAMES, features_to_vector

_OFF_VALUES = ("0", "off", "false", "no")


def _resolve_base(raw: str) -> Path:
    """Kart/env'deki artefakt yolunu diskteki gerçek yola çevir.

    Mutlak yol aynen kullanılır. `data/...` ile başlayan göreli yol DATA_DIR'e
    yeniden köklenir (testlerde `NIHAI_DATA_DIR` geçici dizine bakar — kök
    conftest.py sözleşmesi); diğer göreli yollar repo köküne (ROOT) göredir."""
    path = Path(raw)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "data":
        return DATA_DIR.joinpath(*parts[1:])
    return ROOT / path


def _card_interval(card: dict[str, Any] | None) -> str | None:
    """Kartın eğitim bar aralığı (`data.source` dosya adından). Okunamazsa None."""
    src = str(((card or {}).get("data") or {}).get("source") or "")
    m = re.search(r"_(1m|5m|15m|1h|1d)_", src)
    return m.group(1) if m else None


def _live_interval() -> str:
    """Canlı tahminlerin bar aralığı — `live_predict_service` ile AYNI kaynak sırası."""
    from nihai.config import HTF_INTERVAL
    return (os.getenv("PREDICT_INTERVAL") or HTF_INTERVAL or "").strip()


def _eligible(card: dict[str, Any]) -> bool:
    """Kart canlı gölge çıkarımına uygun mu? (yukarıdaki 2. kilit)"""
    if card.get("kind") != "neural":
        return False
    if str((card.get("data") or {}).get("source") or "").startswith("synthetic"):
        return False
    if list(card.get("feature_names") or []) != list(FEATURE_NAMES):
        return False
    return bool((card.get("artifacts") or {}).get("base_path"))


class _NeuralShadow:
    """Kayıtlı nöral modelin tek-örnek (singleton) gölge çıkarıcısı."""

    # SINIF-DÜZEYİ VARSAYILANLAR: testler örneği `__init__`'i atlayarak (stub fixture)
    # kurabildiğinden guard alanları burada da tanımlıdır — eksik öznitelik yüzünden
    # çıkarımın sessizce None dönmesi (guard'ın kendisinin bug'a dönüşmesi) engellenir.
    _max_z: float = 6.0
    _max_z_mode: str = "fixed"
    _mu: np.ndarray | None = None
    _sigma: np.ndarray | None = None
    train_interval: str | None = None
    train_pos_rate: float | None = None
    train_zmax_q99: float | None = None
    n_members: int = 1
    device: str = "cpu"
    _batched: bool = False
    n_ok: int = 0
    n_ood: int = 0
    n_saturated: int = 0
    max_z_seen: float = 0.0

    def __init__(self) -> None:
        self.tier: str | None = None        # yüklü modelin katmanı ("deep_mlp" vb.)
        self.n_members = 1                  # derin topluluk üye sayısı (tekil model = 1)
        self.train_zmax_q99 = None          # eğitim |z| %99 kuyruğu (auto eşiğin kaynağı)
        self.card_hash: str | None = None   # hangi model kartından geldiği (izlenebilirlik)
        self.reason: str | None = None      # pasifse NEDEN (operatör görünürlüğü)
        self.train_interval: str | None = None   # kartın eğitim bar aralığı (kaynak adından)
        self.train_pos_rate: float | None = None  # eğitim taban oranı (kalibrasyon referansı)
        self._model: Any = None
        self._features: list[str] = list(FEATURE_NAMES)
        # OOD guard sayaçları (status() üzerinden görünür — sessiz ölüm yasak)
        self.n_ok = 0
        self.n_ood = 0
        self.n_saturated = 0
        self.max_z_seen = 0.0
        try:
            self._passes = int(os.getenv("NEURAL_SHADOW_MC_PASSES", "20"))
        except ValueError:
            self._passes = 20
        self._passes = max(1, min(200, self._passes))
        # OOD eşiği: sayı → sabit eşik; "auto" → modelin KENDİ eğitim dağılımından türetilir
        # (bkz. `_load`). Sabit 6.0 bir tahmindi; farklı özellik setlerinde eğitim verisinin
        # kendi kuyruğu 6'nın altında da üstünde de olabilir — eşiği veriye bağlamak
        # "yanlış susma"yı da "yanlış geçirme"yi de azaltır.
        raw_max_z = os.getenv("NEURAL_SHADOW_MAX_Z", "6.0").strip().lower()
        self._max_z_mode = "auto" if raw_max_z == "auto" else "fixed"
        try:
            self._max_z = 6.0 if self._max_z_mode == "auto" else float(raw_max_z)
        except ValueError:
            self._max_z = 6.0
            self._max_z_mode = "fixed"
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self._load()

    # ------------------------------------------------------------------ OOD
    def _abs_z_max(self, vec: list[float]) -> float | None:
        """Girdi vektörünün eğitim dağılımına göre en büyük |z|'si. Normalizasyon
        istatistikleri yoksa None (guard sessizce PASİF kalır, akış kırılmaz).
        σ<1e-9 olan özellikler ATLANIR: eğitimde sabit oldukları için z tanımsızdır
        (`rr_ratio` σ=3.2e-15 örneği — bölme 10¹⁵ mertebesinde sahte z üretirdi)."""
        if self._mu is None or self._sigma is None:
            return None
        x = np.asarray(vec, dtype=float)
        if x.shape != self._mu.shape:
            return None
        live = self._sigma > 1e-9
        if not bool(live.any()):
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (x[live] - self._mu[live]) / self._sigma[live]
        z = z[np.isfinite(z)]
        return float(np.abs(z).max()) if z.size else None

    # ------------------------------------------------------------------ yükleme
    def _pick(self) -> tuple[Path | None, dict[str, Any] | None]:
        """(artefakt base-path, kart) — bulunamazsa (None, None) + self.reason."""
        explicit = os.getenv("NEURAL_SHADOW_MODEL", "").strip()
        if explicit:
            base = _resolve_base(explicit)
            if not Path(f"{base}.json").exists():
                self.reason = f"NEURAL_SHADOW_MODEL artefaktı yok: {base}.json"
                return None, None
            return base, None
        try:
            from nihai.ai import registry
            cards = [c for c in registry.list_models() if _eligible(c)]
        except Exception as e:
            self.reason = f"registry okunamadı: {e}"
            return None, None
        if not cards:
            self.reason = "uygun nöral model kartı yok (kind/şema/artefakt kilidi)"
            return None, None
        # ARALIK TERCİHİ (4. kilit, Tur-20): "en yeni uygun kart" tek başına yetmiyor.
        # Bir ARAŞTIRMA koşusu (ör. 1h/234k sette eğitilmiş bir model, P2-11) kaydedildiği
        # anda en yeni kart olur ve canlı 1m köprüsünü yeniden dağılım-dışına düşürürdü —
        # yani O4 düzeltmesi sessizce geri alınırdı. OOD guard bunu çıkarım anında yakalar
        # ama sonuç yine SESSİZLİKTİR; "hiç seçilmesin" > "seçilip sussun".
        # Eşleşen aralıkta kart YOKSA eski davranış korunur (en yeni uygun kart) — o zaman
        # guard nedenini söyleyerek susar; sessiz ölüm yine yok.
        live_iv = _live_interval()
        same = [c for c in cards if live_iv and _card_interval(c) == live_iv]
        if same:
            cards = same
        elif live_iv:
            logger.warning(f"[neural-shadow] canlı aralık={live_iv} ile eşleşen nöral kart YOK "
                           f"({len(cards)} aday) — dağılım-dışı kalma olasılığı yüksek")
        card = max(cards, key=lambda c: str(c.get("registered_at", "")))
        base = _resolve_base(str(card["artifacts"]["base_path"]))
        if not Path(f"{base}.json").exists():
            self.reason = f"kart artefaktı diskte yok: {base}.json"
            return None, None
        return base, card

    def _load(self) -> None:
        if os.getenv("NEURAL_SHADOW", "").strip().lower() in _OFF_VALUES:
            self.reason = "NEURAL_SHADOW=off"
            return
        try:
            from nihai.ai.neural import TORCH_AVAILABLE, load_any
        except Exception as e:   # pragma: no cover - import hatası ortama özgü
            self.reason = f"nihai.ai.neural içe aktarılamadı: {e}"
            return
        if not TORCH_AVAILABLE:
            self.reason = "torch kurulu değil"
            return
        base, card = self._pick()
        if base is None:
            return
        # `load_any`: tekil model VE derin topluluk (v4/Tur-22) artefaktlarını ayırt eder.
        # İkisi de aynı API'yi sunduğundan buradan sonraki hiçbir kod yolu değişmez —
        # topluluk yüklenirse p_win üye ortalaması, std ise toplam varyans yasasıyla gelir.
        model = load_any(base)
        if model is None:
            self.reason = f"model yüklenemedi (bozuk/eksik artefakt): {base}"
            return
        if list(model.feature_names) != list(FEATURE_NAMES):
            self.reason = "model feature şeması FEATURE_NAMES ile uyuşmuyor (train/serve hizası)"
            return
        self._model = model
        self._features = list(model.feature_names)
        self.tier = model.tier
        # Tur-93: 282M MoE-XL CPU'da tek aday MC=20 2,9 sn sürüyordu; döngü dakikada ~30 aday
        # sorar ve çıkarım kilit altında sıralıdır → deep_mlp dışı basamak CUDA varsa GPU'ya
        # taşınır ve MC kopyaları tek ileri geçişte örneklenir (102 ms). deep_mlp CPU'da,
        # döngüyle kalır (11 ms, geçmiş kayıtla bit-denk).
        self._batched = self.tier != "deep_mlp"
        istek = os.getenv("NEURAL_SHADOW_DEVICE", "auto").strip().lower() or "auto"
        if istek == "auto":
            istek = "cuda" if self._batched else "cpu"
        tasi = getattr(model, "to_device", None)
        if callable(tasi):
            try:
                self.device = str(tasi(istek))
            except Exception as e:   # noqa: BLE001 — aygıt hatası (VRAM) gölgeyi düşürmesin
                logger.warning(f"[neural-shadow] {istek} aygıtına taşınamadı ({e}) — CPU'da kalıyor")
                try:
                    self.device = str(tasi("cpu"))
                except Exception:   # noqa: BLE001
                    self.device = "cpu"
        if self._batched and self.device == "cpu":
            logger.warning(f"[neural-shadow] {self.tier} CPU'da: aday başına ~1 sn çıkarım "
                           "(Tur-93 ölçümü) — tahmin döngüsü yavaşlayabilir")
        self.n_members = int(getattr(model, "n_members", 1) or 1)
        self.card_hash = str(card.get("hash")) if card else None
        # VERİ-TÜRETİLMİŞ OOD eşiği (NEURAL_SHADOW_MAX_Z=auto): eğitim setinin |z| %99
        # kuyruğunun 1,5 katı, en az 4,0. Gerekçe: eşik modelin gördüğü dağılıma ait
        # olmalı; sabit 6,0 farklı özellik setlerinde ya çok gevşek ya çok sıkı kalıyordu.
        # Eski artefaktlarda (q99 alanı yok) sessizce sabit eşiğe düşer — geriye uyum.
        zq = getattr(model, "train_zmax_q99", None)
        if self._max_z_mode == "auto":
            if zq:
                self._max_z = max(4.0, round(1.5 * float(zq), 2))
            else:
                logger.warning("[neural-shadow] MAX_Z=auto istendi ama artefaktta "
                               "train_zmax_q99 YOK — sabit 6.0'a düşülüyor")
        self.train_zmax_q99 = (float(zq) if zq else None)
        # OOD guard için modelin KENDİ eğitim normalizasyon istatistikleri
        mu, sigma = getattr(model, "_feat_mu", None), getattr(model, "_feat_sigma", None)
        if mu is not None and sigma is not None:
            self._mu = np.asarray(mu, dtype=float).reshape(-1)
            self._sigma = np.asarray(sigma, dtype=float).reshape(-1)
        else:
            logger.warning("[neural-shadow] modelde feat_mu/feat_sigma yok — OOD guard PASİF")
        # Eğitim bar aralığı + taban oranı (kalibrasyon referansı; yalnız raporlama)
        if card:
            data = card.get("data") or {}
            self.train_interval = _card_interval(card)
            try:
                self.train_pos_rate = float(data["positive_rate"])
            except (KeyError, TypeError, ValueError):
                self.train_pos_rate = None
        logger.info(f"[neural-shadow] gölge çıkarım aktif: tier={self.tier} "
                    f"kart={(self.card_hash or 'env')[:8]} MC={self._passes} "
                    f"eğitim-aralığı={self.train_interval or '?'} taban={self.train_pos_rate} "
                    f"maxZ={self._max_z} (KARAR-DIŞI: yalnız predictions_log'a yazılır)")

    # ------------------------------------------------------------------ çıkarım
    @property
    def active(self) -> bool:
        return self._model is not None

    def pwin(self, feats: dict[str, float], side: str) -> tuple[float, float] | None:
        """Aday işlem TARAFI için (p_win, std) — MC-dropout. Pasif/hatalıysa None.

        `side` ("LONG"/"SHORT") özellik vektöründeki `is_long`'u BELİRLER: model bu
        özelliği eğitimde gördü, dolayısıyla olasılık sorulan tarafa göre kurulmalıdır
        (aday LONG iken is_long=0 vermek sessizce SHORT'un olasılığını döndürür)."""
        if not self.active or not feats:
            return None
        s = (side or "").strip().upper()
        if s not in ("LONG", "SHORT"):
            return None
        try:
            row = dict(feats)
            row["is_long"] = 1.0 if s == "LONG" else 0.0
            vec = features_to_vector(row, self._features)
            # DAĞILIM-DIŞI (OOD) GUARD'ı — modül docstring'i B-2. Doygunluk guard'ından
            # ÖNCE çalışır: girdi eğitim dağılımının dışındaysa çıkarımın kendisi anlamsızdır.
            if self._max_z > 0:
                zmax = self._abs_z_max(vec)
                if zmax is not None:
                    self.max_z_seen = max(self.max_z_seen, zmax)
                    if zmax > self._max_z:
                        self.n_ood += 1
                        logger.debug(f"[neural-shadow] dağılım-dışı girdi (max|z|={zmax:.1f} > "
                                     f"{self._max_z}) — kaydedilmiyor")
                        return None
            x = [vec]
            # Tur-80: tek satırlık MC-dropout TEK iç thread'le ve çıkarım kilidi altında.
            # Ölçüldü (sistem Python, torch 2.11 CPU, deep_mlp MC=20): varsayılan 8 OpenMP
            # thread'i çağrı başına 90,5 ms CPU yakıyordu (duvar 11,6 ms), tek thread 16,3 ms CPU
            # (duvar 13,0 ms) ve 480 çağrıda sonuç BİT-DENK. Kilit BULGU-80-3 yarışını kapatır.
            # Büyük basamak (Tur-93) `_load`da GPU'ya ve toplu MC'ye alınır.
            from nihai.ai.neural import inference_threads
            with inference_threads(1):
                kw = {"batched": True} if self._batched else {}
                p, sd = self._model.predict_with_uncertainty(x, T=self._passes, **kw)
            p_f, sd_f = float(p[0]), float(sd[0])
            # DOYGUNLUK GUARD'ı (canlı duman testinde yakalandı): dağılım-DIŞI bir vektör
            # (ör. göstergeler ısınmamış, çoğu özellik 0) ağı doyurur → p tam 0.0/1.0 ve
            # MC-dropout varyansı TAM 0. Bu "kesin tahmin" değil, ölçüm dışı bir artefakttır;
            # loglanırsa learning.py'nin Brier'ini sahte-kesin değerlerle kirletir. Dropout
            # aktifken gerçek bir tahminin std'si asla tam 0 olmaz → güvenli ayrım.
            if sd_f == 0.0 and (p_f <= 1e-3 or p_f >= 1.0 - 1e-3):
                self.n_saturated += 1
                logger.debug("[neural-shadow] doygun çıktı (dağılım-dışı girdi) — kaydedilmiyor")
                return None
            self.n_ok += 1
            return p_f, sd_f
        except Exception as e:
            logger.debug(f"[neural-shadow] çıkarım atlandı: {e}")
            return None

    def status(self) -> dict[str, Any]:
        """Pano/günlük özet için tek bakışta durum (kenar iddiası taşımaz)."""
        return {"active": self.active, "tier": self.tier, "card": self.card_hash,
                "mc_passes": self._passes, "reason": self.reason, "device": self.device,
                "train_interval": self.train_interval, "train_pos_rate": self.train_pos_rate,
                "max_z": self._max_z, "max_z_seen": round(self.max_z_seen, 2),
                "max_z_mode": self._max_z_mode, "train_zmax_q99": self.train_zmax_q99,
                "ensemble": self.n_members,
                "n_ok": self.n_ok, "n_ood": self.n_ood, "n_saturated": self.n_saturated}


_SHADOW: _NeuralShadow | None = None
_SHADOW_LOCK = threading.Lock()


def get_shadow() -> _NeuralShadow:
    """Tekil gölge çıkarıcı. Kilitli: tahmin havuzu 8 worker'la aynı anda çağırır,
    kilitsiz desende torch modeli birden fazla kez yüklenirdi."""
    global _SHADOW
    if _SHADOW is None:
        with _SHADOW_LOCK:
            if _SHADOW is None:
                _SHADOW = _NeuralShadow()
    return _SHADOW


def reload_shadow() -> None:
    """Yeni model kaydedildiğinde/env değiştiğinde canlı süreçte tazele."""
    global _SHADOW
    with _SHADOW_LOCK:
        _SHADOW = None


# ---------------------------------------------------------------------------
# B-8 öz-denetim — köprü GERÇEKTEN yazacak mı? (piyasa açılışını beklemeden)
# ---------------------------------------------------------------------------
def selfcheck(sample: int = 300) -> dict:
    """Gölge köprünün canlıda çıktı üretip üretmeyeceğini ÖLÇ (salt-okunur).

    Neden gerekli: `predictions_log.csv`'nin neural_* kolonları boş görünüyordu ve
    bunun iki apayrı açıklaması vardı — (a) piyasa kapalı, henüz aday sinyal yok
    (zararsız), (b) model hiç çıktı vermiyor (köprü ÖLÜ). Canlı çıkarım yolundaki
    her hata bilinçli olarak sessizce yutulduğundan (`_neural_shadow_pwin` → None)
    ikisini logdan ayırt etmek imkânsızdı. Bu fonksiyon GERÇEK canlı özellik
    vektörlerini (data/live_features_log.csv) modele vererek farkı ölçer.

    Dönen: {'ok', 'model', 'denenen', 'uretildi', 'susturuldu', 'oran', 'pwin_ort'}.
    'oran' 0.0 ise köprü canlıda da sessiz kalacak demektir → araştır."""
    from nihai.config import DATA_DIR
    from nihai.signals.meta_features import FEATURE_NAMES

    sh = get_shadow()
    out: dict = {"ok": False, "model": getattr(sh, "card_hash", None),
                 "denenen": 0, "uretildi": 0, "susturuldu": 0, "oran": 0.0, "pwin_ort": None}

    log = DATA_DIR / "live_features_log.csv"
    if not log.exists():
        out["hata"] = f"canlı özellik logu yok: {log}"
        return out
    try:
        import pandas as pd
        df = pd.read_csv(log).tail(sample)
    except Exception as e:  # noqa: BLE001
        out["hata"] = f"log okunamadı: {e}"
        return out
    if df.empty:
        out["hata"] = "canlı özellik logu boş"
        return out

    pwins: list[float] = []
    for _, r in df.iterrows():
        feats = {k: float(r[k]) for k in FEATURE_NAMES if k in r and pd.notna(r[k])}
        if not feats:
            continue
        for side in ("LONG", "SHORT"):
            feats["is_long"] = 1.0 if side == "LONG" else 0.0
            out["denenen"] += 1
            res = sh.pwin(dict(feats), side)
            if res is None:
                out["susturuldu"] += 1
            else:
                out["uretildi"] += 1
                pwins.append(float(res[0]))

    if out["denenen"]:
        out["oran"] = round(out["uretildi"] / out["denenen"], 4)
    if pwins:
        out["pwin_ort"] = round(sum(pwins) / len(pwins), 4)
    out["ok"] = out["uretildi"] > 0
    # --- OOD/kalibrasyon tanılaması (B-2): "sessiz mi, ölü mü, yoksa dağılım-dışı mı?" ---
    # `status()` savunmacı okunur: testler `sh`'i minimal bir duck-type ile değiştirebilir.
    try:
        st = sh.status()
    except (AttributeError, Exception):  # noqa: BLE001 — tanılama asla selfcheck'i düşürmemeli
        st = {}
    out["ood_susturulan"] = st.get("n_ood")
    out["doygun_susturulan"] = st.get("n_saturated")
    out["max_z_gorulen"] = st.get("max_z_seen")
    out["egitim_araligi"] = st.get("train_interval")
    out["egitim_taban_orani"] = st.get("train_pos_rate")
    base = st.get("train_pos_rate")
    sapma = (abs(out["pwin_ort"] - float(base))
             if (out["pwin_ort"] is not None and isinstance(base, (int, float))) else None)
    out["kalibrasyon_sapmasi"] = round(sapma, 4) if sapma is not None else None
    if not out["ok"] and (st.get("n_ood") or 0) > 0:
        out["verdikt"] = ("DAĞILIM-DIŞI: canlı özellikler modelin eğitim dağılımının dışında "
                          f"(max|z|={st.get('max_z_seen')}, eşik={st.get('max_z')}); "
                          f"eğitim aralığı={st.get('train_interval')}. Köprü bilinçli SUSUYOR.")
    elif not out["ok"]:
        out["verdikt"] = "KÖPRÜ SESSİZ — neural_* kolonları dolmayacak; nedeni 'reason'da."
    elif sapma is not None and sapma > 0.20:
        out["verdikt"] = (f"KALİBRASYON ŞÜPHELİ: p_win ortalaması {out['pwin_ort']} vs eğitim "
                          f"taban oranı {base} (sapma {sapma:.3f} > 0.20).")
    else:
        out["verdikt"] = "KÖPRÜ CANLI ve kalibrasyon taban oranıyla tutarlı."
    return out


if __name__ == "__main__":   # python -m nihai.execution.neural_shadow
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    import json as _json

    _r = selfcheck()
    print(_json.dumps(_r, ensure_ascii=False, indent=2))
    print("KÖPRÜ CANLI" if _r["ok"] else "KÖPRÜ SESSİZ — neural_* kolonları dolmayacak")
