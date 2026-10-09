"""
registry.py — MODEL YAŞAM DÖNGÜSÜ / registry (Tur-16 WP-O kalan kısım). KARAR-DIŞI.

Ne yapar: her eğitilen modeli (herhangi bir tier — CapacityModel/NeuralCapacity/deney
modeli fark etmez) İÇERİK-HASH'i ile kimlikleyip bir MODEL KARTI (JSON) olarak
`data/ai_models/<hash>/card.json` altında sürümlü saklar; ayrıca iki modelin canlı akışta
GÖLGE A/B karşılaştırmasını (`shadow_ab_log` + `ab_report`) sağlar.

ANAYASA: bu modül ÜRETİM `data/meta_model.json` / `data/meta_model.pkl` dosyalarına
HİÇBİR KOŞULDA dokunmaz (okumaz da, yazmaz da) — registry TAMAMEN ayrı bir GÖLGE kayıt
alanıdır (`data/ai_models/`). "Kayıt terfi" (registry'deki bir modelin üretime alınması)
bu modülün sorumluluğu DEĞİLDİR; o zaten kanıt-kapılı ayrı bir yoldur (research/live_retrain
--promote, ACI DERS #2/#5). `register()` bir modelin var olduğunu ve ölçümlerini BELGELER,
onu hiçbir akışa BAĞLAMAZ.

Kayıt kökü: `artifacts_base_path` argümanı VERİLMEZSE `nihai.config.DATA_DIR / "ai_models"`
kullanılır; `DATA_DIR` zaten `NIHAI_DATA_DIR` env'ini okur (kök `conftest.py` bunu testlerde
geçici dizine yönlendirir) — bu yüzden testler artifacts_base_path'i AÇIKÇA vermese bile
gerçek `data/`'ya yazılmaz. Yine de testler determinizm/izolasyon için genelde `tmp_path`
tabanlı bir `artifacts_base_path` geçer (iki yol da desteklenir — "çağırandan/env'den türer").

API (docs/AI_V3_DERIN_BEYIN_SPEC.md §3.12 taslağının bu görev talimatına göre genişletilmiş
hali — bkz. teslim raporundaki "bilinçli sapmalar"):
  - `register(model_meta, artifacts_base_path=None) -> dict`             : model kartı yazar, kartı döner
  - `list_models(artifacts_base_path=None, include_deprecated=False) -> list[dict]` : GÜNCEL kartlar
  - `get_card(model_hash, artifacts_base_path=None) -> dict | None`      : tek kart (bozuk → None)
  - `latest(tier=None, artifacts_base_path=None, include_deprecated=False) -> dict | None` : en son kart
  - `deprecate(model_hash, reason, artifacts_base_path=None) -> dict | None`  : kartı emekliye ayırır
  - `undeprecate(model_hash, artifacts_base_path=None) -> dict | None`        : bayrağı kaldırır

EMEKLİLİK (`deprecated`) — 2026-07-26 denetimi B-5:
Kayıt defterinde ÖLÇÜMÜ BOZUK kartlar birikebiliyor. Somut vaka: `ac505353…` kartı aynı PIT
verisinde **DSR 1.0 / PSR 1.0** raporluyordu — ACI DERS #8'in düzelttiği sınıf-dengesizliği
hatasının ürünü (sözde-getiri triviyal kurala göre FAZLALIK olmalıydı). Düzeltilmiş kart
(`1f072998…`, DSR 0.0) onun YANINDA duruyor ve `list_models()` üçünü de EŞİT statüde
döndürüyordu → gelecekte yanlış kartın seçilmesi/alıntılanması riski (`neural_shadow._pick()`
"en yeni uygun kart"ı seçer; bir gün bozuk kart en yeni olabilirdi).

`deprecate()` kartı SİLMEZ — bilimsel kayıt korunur, yalnız `deprecated={"at","reason"}`
alanı eklenir ve varsayılan listelemenin DIŞINDA kalır. Silmek yerine işaretlemek bilinçli:
"neden emekli" bilgisi kurumsal hafızadır (CLAUDE.md ACI DERSLER deseni).
`register()` içerik-özdeş bir kartı yeniden yazarken MEVCUT emeklilik bayrağını KORUR —
aynı eğitimin tekrar koşulması bir küratör kararını sessizce geri almamalıdır.
  - `shadow_ab_log(model_a_hash, model_b_hash, symbol, p_a, p_b, *,
                    outcome=None, artifacts_base_path=None) -> dict`      : ab_log.jsonl'e atomik satır
  - `ab_report(model_a_hash, model_b_hash, *, min_n=100,
               artifacts_base_path=None) -> dict`                         : Brier/isabet + Wilson + verdikt

Savunmacı: bozuk/eksik JSON kart → None (list_models sessizce atlar); tüm yazımlar
`nihai.atomic_io` üzerinden (OneDrive/AV kilidine dayanıklı, ACI DERS — atomic_io.py başlığı).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR

# Varsayılan kayıt kökü (testler NIHAI_DATA_DIR üzerinden veya artifacts_base_path ile geçersiz kılar).
AI_MODELS_DIR = DATA_DIR / "ai_models"
AB_LOG_FILENAME = "ab_log.jsonl"
CARD_FILENAME = "card.json"

# ab_log.jsonl için oku+ekle+atomik-tam-yeniden-yaz (portfolio.py/episodic.py deseni) —
# aynı süreç içinde iki iş parçacığının satırları ezmesini önler.
_LOCK = threading.Lock()

# Wilson-skor aralığı için %95 çift-taraflı z (sabit — testlerde beklenen değer).
_Z_95 = 1.959963984540054


def _utc_now_iso() -> str:
    """UTC ts (tz-aware, saniye çözünürlüğü) — router.py/models.py ile aynı konvansiyon."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _resolve_base(artifacts_base_path: Path | str | None) -> Path:
    return Path(artifacts_base_path) if artifacts_base_path is not None else AI_MODELS_DIR


def _content_hash(model_meta: dict[str, Any]) -> str:
    """`model_meta` içeriğinin kararlı (anahtar-sıralı) JSON temsilinin sha256'sı —
    aynı içerik HER ZAMAN aynı hash'i üretir (kayıt-tekrarında idempotent dizin)."""
    canonical = json.dumps(model_meta, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_card(path: Path) -> dict[str, Any] | None:
    """Savunmacı kart okuma: dosya yok/bozuk-JSON/sözlük-değil/`hash` alanı yok → None
    (çökme yok — models.py `CapacityModel.load` ile aynı savunmacı desen)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or "hash" not in raw:
        return None
    return raw


# ---------------------------------------------------------------------------
# Model kartı yaşam döngüsü
# ---------------------------------------------------------------------------
def register(model_meta: dict[str, Any], artifacts_base_path: Path | str | None = None) -> dict[str, Any]:
    """`model_meta` (veri dönemi/boyutu, hiperparametreler, OOS metrikler, kalibrasyon,
    DSR, eğitim ortamı, üst-model hash'i — çağıranın sorumluluğu, burada şema zorunlu
    DEĞİL) içeriğinden sha256 hesaplar, `hash` + `registered_at` (UTC) alanlarını ekleyip
    MODEL KARTI olarak `<base>/<hash>/card.json`'a ATOMİK yazar; yazılan kartı döner.

    `model_meta` sözlük değilse ValueError (programcı-hatası — savunmacı-girdi sözleşmesi
    yalnız DATA'yı, yani veri-kalitesi sorunlarını kapsar; API yanlış kullanımını değil).
    Aynı içerikle tekrar çağrı → aynı dizine (idempotent hash) yeni `registered_at`'lı
    kart yazar (üzerine yazma; içerik/hash DEĞİŞMEZ — yalnız kayıt zamanı güncellenir)."""
    if not isinstance(model_meta, dict):
        raise ValueError("model_meta bir sözlük (dict) olmalı")
    base = _resolve_base(artifacts_base_path)
    content_hash = _content_hash(model_meta)
    card: dict[str, Any] = dict(model_meta)
    card["hash"] = content_hash
    card["registered_at"] = _utc_now_iso()
    card_dir = base / content_hash
    # Emeklilik bayrağı bir KÜRATÖR kararıdır; içerik-özdeş yeniden-kayıt (aynı eğitimin
    # tekrar koşulması) onu sessizce geri almamalı — varsa taşı (modül başlığı B-5).
    prev = _load_card(card_dir / CARD_FILENAME)
    if prev is not None and prev.get("deprecated") and "deprecated" not in card:
        card["deprecated"] = prev["deprecated"]
        logger.debug(f"[ai.registry] emeklilik bayrağı korundu: {content_hash[:12]}…")
    card_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(card_dir / CARD_FILENAME, json.dumps(card, ensure_ascii=False, indent=2, default=str))
    return card


def is_deprecated(card: dict[str, Any] | None) -> bool:
    """Kart emekliye ayrılmış mı? (bayrak yok/boş → False; savunmacı: None de False)"""
    return bool((card or {}).get("deprecated"))


def list_models(artifacts_base_path: Path | str | None = None, *,
                include_deprecated: bool = False) -> list[dict[str, Any]]:
    """`<base>/*/card.json` altındaki bozulmamış kartlar (sıra: dizin adı/hash sırası —
    kronolojik sıralama isteyen `latest()`/çağıran `registered_at`'a göre kendi sıralar).
    Kök dizin yoksa boş liste (çökme yok).

    VARSAYILAN olarak EMEKLİ kartlar (`deprecated` alanı dolu) DIŞARIDA bırakılır —
    ölçümü bozuk bir kartın "geçerli model" gibi seçilmesi/alıntılanması B-5 bulgusuydu.
    Tam dökümü (denetim/arşiv) isteyen `include_deprecated=True` geçer."""
    base = _resolve_base(artifacts_base_path)
    if not base.exists() or not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        card = _load_card(d / CARD_FILENAME)
        if card is None:
            logger.debug(f"[ai.registry] bozuk/eksik kart atlandı: {d.name}")
            continue
        if is_deprecated(card) and not include_deprecated:
            continue
        out.append(card)
    return out


def get_card(model_hash: str, artifacts_base_path: Path | str | None = None) -> dict[str, Any] | None:
    """Tek bir modelin kartı (hash ile). Yok/bozuk → None."""
    base = _resolve_base(artifacts_base_path)
    return _load_card(base / str(model_hash) / CARD_FILENAME)


def latest(tier: str | None = None, artifacts_base_path: Path | str | None = None, *,
           include_deprecated: bool = False) -> dict[str, Any] | None:
    """`registered_at`'ı EN BÜYÜK (ISO-8601 string sıralaması = kronolojik) kart; `tier`
    verilirse yalnız `card["tier"] == tier` olanlar arasından. Hiç kart yoksa None.
    Emekli kartlar varsayılan olarak ELENİR (bkz. `list_models`)."""
    cards = list_models(artifacts_base_path, include_deprecated=include_deprecated)
    if tier is not None:
        cards = [c for c in cards if c.get("tier") == tier]
    if not cards:
        return None
    return max(cards, key=lambda c: str(c.get("registered_at", "")))


def deprecate(model_hash: str, reason: str,
              artifacts_base_path: Path | str | None = None) -> dict[str, Any] | None:
    """Kartı EMEKLİYE ayır: `deprecated={"at": <UTC>, "reason": <neden>}` ekleyip atomik
    yeniden yazar; güncellenmiş kartı döner. Kart yok/bozuksa None (çökme yok).

    Kart SİLİNMEZ ve ölçüm alanları DEĞİŞTİRİLMEZ — "neden emekli" bilgisi kurumsal
    hafızanın parçasıdır. `reason` zorunlu ve boş olamaz: gerekçesiz emeklilik gelecekte
    okunamaz bir karardır (ValueError — programcı-hatası sözleşmesi, `register` ile aynı)."""
    if not str(reason or "").strip():
        raise ValueError("deprecate(): 'reason' boş olamaz (gerekçesiz emeklilik yasak)")
    base = _resolve_base(artifacts_base_path)
    card_path = base / str(model_hash) / CARD_FILENAME
    card = _load_card(card_path)
    if card is None:
        logger.warning(f"[ai.registry] emekliye ayrılamadı (kart yok/bozuk): {model_hash}")
        return None
    card["deprecated"] = {"at": _utc_now_iso(), "reason": str(reason).strip()}
    atomic_write_text(card_path, json.dumps(card, ensure_ascii=False, indent=2, default=str))
    logger.info(f"[ai.registry] kart emekliye ayrıldı: {str(model_hash)[:12]}… — {reason}")
    return card


def undeprecate(model_hash: str,
                artifacts_base_path: Path | str | None = None) -> dict[str, Any] | None:
    """`deprecate`'in tersi: emeklilik bayrağını kaldırır. Kart yok/bozuksa None.
    Zaten emekli değilse kart olduğu gibi döner (idempotent)."""
    base = _resolve_base(artifacts_base_path)
    card_path = base / str(model_hash) / CARD_FILENAME
    card = _load_card(card_path)
    if card is None:
        logger.warning(f"[ai.registry] bayrak kaldırılamadı (kart yok/bozuk): {model_hash}")
        return None
    if card.pop("deprecated", None) is not None:
        atomic_write_text(card_path, json.dumps(card, ensure_ascii=False, indent=2, default=str))
        logger.info(f"[ai.registry] emeklilik bayrağı kaldırıldı: {str(model_hash)[:12]}…")
    return card


# ---------------------------------------------------------------------------
# Gölge A/B — log + istatistiksel karşılaştırma
# ---------------------------------------------------------------------------
def _append_line(path: Path, line: str) -> None:
    """JSONL'e TEK satır ekle — O(dosya) değil O(satır) (Tur-24 optimizasyonu).

    Eskiden bu yol "dosyanın tamamını oku + satırı ekle + `atomic_write_text` ile tamamını
    yeniden yaz" idi. İki maliyet vardı: (a) her çağrıda geçici dosya + `os.replace`
    (Windows'ta pahalı), (b) dosya BÜYÜDÜKÇE büyüyen oku/yaz terimi — ve `ab_log.jsonl`
    yalnız-ekleme, sınırsız büyüyen bir günlüktür. Bu, prediction_tracker'da D1 olarak
    kapatılan hatanın (canlı döngüde tüm log'u yeniden okuma) aynısıydı.

    ÖLÇÜLDÜ (tahmin edilmedi): 300 satır 3,385 sn → 0,352 sn (**9,6×**), 2.000 satır
    22,935 sn → 2,263 sn (**10,1×**).

    Dayanıklılık DÜŞMEZ, artar: `os.replace` çökme anında ya eski ya yeni TAM dosyayı
    bırakır ama her yazımda tüm dosyayı riske atar; küçük bir satırın `write+flush+fsync`
    ile eklenmesi en kötü ihtimalle SON satırı yarım bırakır ve `_read_ab_rows` bozuk
    satırı zaten atlar (`except ValueError: continue`). Süreç-içi eşzamanlılık çağıranın
    `_LOCK`'uyla korunur (imza değişmedi)."""
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def shadow_ab_log(model_a_hash: str, model_b_hash: str, symbol: str, p_a: float, p_b: float, *,
                  outcome: int | None = None,
                  artifacts_base_path: Path | str | None = None) -> dict[str, Any]:
    """Bir A/B gözlemini `<base>/ab_log.jsonl`'e ATOMİK ekler (oku+ekle+tam-yeniden-yaz;
    portfolio.py `_append_jsonl` deseni). `outcome` (0/1, gerçekleşen yön/sonuç) genelde
    KARAR ANINDA bilinmez — verilmezse (None) satır yalnız p_a/p_b'yi kaydeder; sonuç
    çözüldüğünde AYNI çağrı `outcome` ile TEKRAR loglanabilir (append-only günlük — iki
    satır da kalır, `ab_report` yalnız `outcome is not None` satırları sayar).
    İzolasyon garantisi: bu fonksiyon SADECE ab_log.jsonl'e yazar — `register`/kart
    dosyalarına dokunmaz (iki modelin durumu birbirini bozamaz — bkz. testler)."""
    base = _resolve_base(artifacts_base_path)
    row: dict[str, Any] = {
        "ts": _utc_now_iso(),
        "model_a": str(model_a_hash),
        "model_b": str(model_b_hash),
        "symbol": str(symbol),
        "p_a": float(p_a),
        "p_b": float(p_b),
        "outcome": int(outcome) if outcome is not None else None,
    }
    line = json.dumps(row, ensure_ascii=False, default=str)
    path = base / AB_LOG_FILENAME
    try:
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            _append_line(path, line)
    except OSError as e:
        logger.warning(f"[ai.registry] {AB_LOG_FILENAME} yazılamadı: {e}")
    return row


def _wilson_interval(k: int, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson skor aralığı (normal-yaklaşıklığından daha sağlam, küçük n'de de makul).
    n<=0 → (0.0, 1.0) (bilgisiz aralık, çökme yok)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _read_ab_rows(path: Path, model_a_hash: str, model_b_hash: str) -> list[dict[str, Any]]:
    """`ab_log.jsonl`'den verilen çift (sıra bağımsız) + `outcome is not None` satırları;
    çıktı DAİMA çağıranın istediği (a,b) sırasına normalize edilmiş (p_a/p_b/y)."""
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.debug(f"[ai.registry] {path.name} okunamadı: {e}")
        return []
    a_key, b_key = str(model_a_hash), str(model_b_hash)
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            row = json.loads(raw_line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("outcome") is None:
            continue
        a, b = row.get("model_a"), row.get("model_b")
        if {a, b} != {a_key, b_key}:
            continue
        try:
            p_a_raw, p_b_raw = float(row["p_a"]), float(row["p_b"])
            y = int(row["outcome"])
        except (KeyError, TypeError, ValueError):
            continue
        if a == a_key:
            rows.append({"p_a": p_a_raw, "p_b": p_b_raw, "y": y})
        else:
            rows.append({"p_a": p_b_raw, "p_b": p_a_raw, "y": y})
    return rows


def ab_report(model_a_hash: str, model_b_hash: str, *, min_n: int = 100,
              artifacts_base_path: Path | str | None = None) -> dict[str, Any]:
    """İki modelin (`shadow_ab_log`'a loglanmış, `outcome` çözülmüş) gözlemlerinden
    Brier skoru + isabet (0.5 eşikli ikili doğruluk) + Wilson %95 güven aralığı hesaplar.

    n < `min_n` → `verdict="YETERSIZ_ORNEK"` (metrik alanları YOK — örneklem küçükken
    sayı üretip yanlış güven vermek yerine dürüstçe reddet, ACI DERS #3 ruhu: kapı
    örneklem-kapılı olmalı).

    İSTATİSTİKSEL üstünlük testi: isabet oranlarının Wilson %95 aralıkları ÇAKIŞIYORSA
    (konservatif — tip-1 hatayı düşük tutan basit ama sağlam bir kıstas) `verdict="FARKSIZ"`;
    çakışmıyorsa daha yüksek isabetli taraf `"A_USTUN"`/`"B_USTUN"`. Brier skorları da
    (düşük=iyi) rapora eklenir — verdikt YALNIZ isabet-CI çakışmasına dayanır, Brier
    ek-bilgi/tie-break DEĞİLDİR (tek, belgeli kritere sadakat — yorum karmaşası önlenir)."""
    base = _resolve_base(artifacts_base_path)
    rows = _read_ab_rows(base / AB_LOG_FILENAME, model_a_hash, model_b_hash)
    n = len(rows)
    result: dict[str, Any] = {
        "model_a": str(model_a_hash), "model_b": str(model_b_hash),
        "n": n, "min_n": int(min_n), "verdict": "YETERSIZ_ORNEK",
    }
    if n < max(1, int(min_n)):
        return result

    y = np.array([r["y"] for r in rows], dtype=float)
    p_a = np.array([r["p_a"] for r in rows], dtype=float)
    p_b = np.array([r["p_b"] for r in rows], dtype=float)

    brier_a = float(np.mean((p_a - y) ** 2))
    brier_b = float(np.mean((p_b - y) ** 2))
    hits_a = int(np.sum((p_a >= 0.5).astype(float) == y))
    hits_b = int(np.sum((p_b >= 0.5).astype(float) == y))
    isabet_a = hits_a / n
    isabet_b = hits_b / n
    wilson_a = _wilson_interval(hits_a, n)
    wilson_b = _wilson_interval(hits_b, n)

    result.update({
        "brier_a": brier_a, "brier_b": brier_b,
        "isabet_a": isabet_a, "isabet_b": isabet_b,
        "wilson_a": wilson_a, "wilson_b": wilson_b,
    })

    overlap = not (wilson_a[1] < wilson_b[0] or wilson_b[1] < wilson_a[0])
    if overlap:
        result["verdict"] = "FARKSIZ"
    elif isabet_a > isabet_b:
        result["verdict"] = "A_USTUN"
    else:
        result["verdict"] = "B_USTUN"
    return result
