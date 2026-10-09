"""
audit_chain.py — SHA-256 KARAR ZİNCİRİ ve NTP SAAT KAYMASI (Tur-75). KARAR-DIŞI.

Dış geliştirme listesinin BÖLÜM 12'sinden iki madde:

| Madde | Fonksiyon |
|---|---|
| 242 NTP / saat kayması koruyucusu | `clock_drift` |
| 244 Kriptografik değiştirilemez karar zinciri | `append` · `verify` · `merkle_root` |

ANAYASA: KARAR-DIŞI. Zincir hiçbir kararı DEĞİŞTİRMEZ; yalnız kaydı MÜHÜRLER. Saat
kayması da bir kapı değil bir ÖLÇÜMDÜR — `observability` üzerinden `/metrics`e girer.

## Zincirin ne İDDİA ETTİĞİ ve ETMEDİĞİ (abartılmıyor)

Her satır bir öncekinin SHA-256'sını taşır; ortadaki bir satırı değiştiren, ondan
sonraki TÜM özetleri de yeniden hesaplamak zorundadır. Bu, **kazara bozulmayı ve
seçici düzenlemeyi** yakalar.

**Kötü niyetli bir yazara karşı kanıt DEĞİLDİR.** Dosyaya yazma yetkisi olan biri
zinciri baştan sona yeniden üretebilir. Gerçek değiştirilemezlik dış bir çapa ister
(zaman damgası otoritesi, ayrı makinede tutulan kök, blok zinciri). `merkle_root`
tam da bunun için vardır: gün sonunda TEK bir 64 karakterlik kök dışarı taşınırsa
(Telegram mesajı, e-posta, dış depo) o günün kaydı fiilen mühürlenmiş olur.
Kök dışarı taşınmadıkça zincir yalnız BÜTÜNLÜK kontrolüdür.

Sözleşme: fail-soft. Ağ/dosya hatasında istisna FIRLATMAZ; `clock_drift` ölçemezse
`measured=False` döner (ölçüm yokluğu ölçüm gibi sunulmaz — Tur-31/33).
"""
from __future__ import annotations

import hashlib
import json
import socket
import struct
import time
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from nihai.atomic_io import tail_lines
from nihai.config import DATA_DIR

CHAIN_PATH = Path(DATA_DIR) / "audit_chain.jsonl"
# Zincirin ilk halkası — sabit ve belgeli (rastgele bir kök, doğrulamayı
# yeniden üretilemez kılardı).
GENESIS = "0" * 64

NTP_SERVER = "pool.ntp.org"
NTP_PORT = 123
NTP_TIMEOUT = 3.0
# 1900-01-01 (NTP dönemi) ile 1970-01-01 (Unix dönemi) arası saniye.
NTP_EPOCH_DELTA = 2_208_988_800
# Listenin önerdiği eşik: 200 ms üzerinde uyarı.
DRIFT_WARN_MS = 200.0


# ---------------------------------------------------------------------------
# Madde 244 — SHA-256 karar zinciri
# ---------------------------------------------------------------------------
def _digest(prev_hash: str, payload: str) -> str:
    """SHA-256(prev || payload). AYIRAÇ ZORUNLUDUR: onsuz ("ab","c") ile ("a","bc")
    AYNI özeti verir ve zincir çakıştırılabilir hâle gelir (uzunluk-uzatma sınıfı)."""
    return hashlib.sha256(f"{prev_hash}\x1f{payload}".encode()).hexdigest()


def _canonical(record: dict) -> str:
    """Kaydın KANONİK metni: sıralı anahtar, ayıraçsız boşluk, ASCII-kaçışsız.

    Kanoniklik olmadan aynı sözlük iki farklı JSON metnine ve dolayısıyla iki farklı
    özete gidebilir (anahtar sırası Python sürümüne/eklenme sırasına bağlıdır) →
    doğrulama SAHTE olarak başarısız olurdu.
    """
    return json.dumps(record, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def son_halka(path: Path | str | None = None) -> tuple[str, int]:
    """Zincirin son (hash, seq) ikilisi; dosya yok/boş/bozuksa `(GENESIS, 0)`.

    **SON SATIR okunur, dosya TARANMAZ.** İlk sürüm `append` başına dosyayı İKİ KEZ
    baştan sona geziyordu (bir kez son hash için, bir kez `seq` saymak için) → zincir
    büyüdükçe ekleme maliyeti O(n), zincirin ömrü boyunca O(n²) oluyordu. Bu, Tur-35'in
    `_tail_lines` dersinin ta kendisidir ve doğru desen zaten depoda: `atomic_io.
    tail_lines` (ölçüldü 9,7-86×). `seq` KAYDIN İÇİNDEDİR, sayılması gerekmez.
    """
    p = Path(path) if path is not None else CHAIN_PATH
    try:
        if not p.exists() or p.stat().st_size == 0:
            return GENESIS, 0
        satirlar = tail_lines(p, 1)
        if not satirlar:
            return GENESIS, 0
        son = json.loads(satirlar[-1])
        return str(son["hash"]), int(son["seq"]) + 1
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("[zincir] son halka okunamadı — GENESIS'ten devam")
        return GENESIS, 0


def last_hash(path: Path | str | None = None) -> str:
    """Zincirin son halkası; dosya yok/boş/bozuksa GENESIS."""
    return son_halka(path)[0]


def append(record: dict, path: Path | str | None = None) -> dict | None:
    """Kaydı zincire ekle: `{ts, seq, prev, payload, hash}` satırı yazılır.

    `ts` UTC ISO-8601'dir ve ÖZETE DAHİLDİR (zaman damgasını sonradan değiştirmek de
    zinciri kırmalıdır). Yazma APPEND-ONLY'dur ve `atomic_io` yolu BİLEREK
    kullanılmaz: atomik-değiştirme dosyanın TAMAMINI yeniden yazar, oysa bu defterin
    tüm amacı yeniden-yazılmamaktır. Ek satır tek bir `write` çağrısıyla gider.
    """
    p = Path(path) if path is not None else CHAIN_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        onceki, seq = son_halka(p)
        govde = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "seq": seq,
            "prev": onceki,
            "payload": record,
        }
        govde["hash"] = _digest(onceki, _canonical(govde))
        with open(p, "a", encoding="utf-8") as f:
            f.write(_canonical(govde) + "\n")
        return govde
    except (OSError, ValueError, TypeError) as e:
        logger.warning(f"[zincir] eklenemedi: {type(e).__name__}: {e}")
        return None


def verify(path: Path | str | None = None) -> dict:
    """Zinciri baştan sona doğrula. İLK bozuk halkanın indeksi raporlanır.

    Dönen `ok=False` iken `broken_at` HANGİ satırın kırıldığını söyler — "zincir
    bozuk" demek yeterli değildir, operatörün nereye bakacağını bilmesi gerekir.
    """
    p = Path(path) if path is not None else CHAIN_PATH
    if not p.exists():
        return {"ok": True, "n": 0, "reason": "DOSYA_YOK", "broken_at": None}
    onceki = GENESIS
    n = 0
    try:
        with open(p, encoding="utf-8") as f:
            for i, satir in enumerate(f):
                if not satir.strip():
                    continue
                kayit = json.loads(satir)
                if not isinstance(kayit, dict):
                    raise ValueError("Zincir kaydı nesne olmalı")
                govde = {k: kayit[k] for k in ("ts", "seq", "prev", "payload")}
                beklenen = _digest(onceki, _canonical(govde))
                if kayit.get("prev") != onceki:
                    return {"ok": False, "n": n, "broken_at": i,
                            "reason": "PREV_UYUSMUYOR", "broken_seq": kayit.get("seq")}
                if kayit.get("hash") != beklenen:
                    return {"ok": False, "n": n, "broken_at": i,
                            "reason": "HASH_UYUSMUYOR", "broken_seq": kayit.get("seq")}
                onceki = str(kayit["hash"])
                n += 1
    except (OSError, ValueError, KeyError) as e:
        return {"ok": False, "n": n, "broken_at": n, "reason": f"OKUNAMADI:{type(e).__name__}"}
    return {"ok": True, "n": n, "head": onceki, "broken_at": None, "reason": None}


def merkle_root(path: Path | str | None = None) -> str | None:
    """Tüm halkaların Merkle kökü — dışarı taşınacak TEK mühür.

    Tek sayıda düğüm kalan seviyede SON düğüm KENDİSİYLE eşlenir (yaygın konvansiyon).
    Bu, bilinen bir CVE sınıfının (Bitcoin CVE-2012-2459, aynı kökü veren iki farklı
    ağaç) kaynağıdır; burada zincirin sırası ZATEN `prev` ile kilitli olduğu için
    saldırı yüzeyi yoktur — yine de sözleşme yazılıdır ki bir sonraki tur bu kökü
    başka bir bağlamda yeniden kullanmasın.
    """
    p = Path(path) if path is not None else CHAIN_PATH
    try:
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            yapraklar = [str(json.loads(s)["hash"]) for s in f if s.strip()]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not yapraklar:
        return None
    seviye = yapraklar
    while len(seviye) > 1:
        if len(seviye) % 2:
            seviye = [*seviye, seviye[-1]]
        seviye = [hashlib.sha256((seviye[i] + seviye[i + 1]).encode()).hexdigest()
                  for i in range(0, len(seviye), 2)]
    return seviye[0]


def seal(path: Path | str | None = None) -> dict:
    """Gün sonu mührü: doğrulama + Merkle kökü tek çağrıda (operatöre taşınacak özet)."""
    d = verify(path)
    d["merkle_root"] = merkle_root(path) if d.get("ok") else None
    return d


# ---------------------------------------------------------------------------
# Madde 242 — NTP saat kayması
# ---------------------------------------------------------------------------
def clock_drift(server: str = NTP_SERVER, timeout: float = NTP_TIMEOUT) -> dict:
    """Yerel saat ile NTP sunucusu arasındaki kayma (ms). ÖLÇÜMDÜR, kapı DEĞİLDİR.

    SNTP (RFC 4330) gecikme-düzeltmeli tahmin:

        offset = ((T2 − T1) + (T3 − T4)) / 2      ·      roundtrip = (T4 − T1) − (T3 − T2)

    Tek yönlü farkı offset saymak (yaygın kısayol) ağ gecikmesinin YARISINI saat
    kayması diye raporlar; 60 ms'lik bir turda bu 30 ms'lik hayalet kayma demektir ve
    200 ms eşiğinin altında bile ölçümü sistematik olarak kaydırır.

    Ağ yoksa/zaman aşımında `measured=False` + `reason` döner — 0,0 ms DÖNMEZ
    ("sıfır kayma" ile "ölçemedim" aynı şey değildir; Tur-31 kuralı).
    """
    paket = b"\x1b" + 47 * b"\0"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.settimeout(float(timeout))
        t1 = time.time()
        s.sendto(paket, (server, NTP_PORT))
        veri, _ = s.recvfrom(48)
        t4 = time.time()
    except (OSError, ValueError) as e:   # socket.timeout ⊂ OSError (UP041)
        return {"measured": False, "reason": f"{type(e).__name__}", "server": server,
                "offset_ms": None, "roundtrip_ms": None, "warn": False}
    finally:
        s.close()
    if len(veri) < 48:
        return {"measured": False, "reason": "KISA_PAKET", "server": server,
                "offset_ms": None, "roundtrip_ms": None, "warn": False}
    # RFC 4330 paket düzeni — OFSETLER KRİTİKTİR: 16-23 referans, 24-31 ORİJİN
    # (bizim gönderdiğimizin yankısı), **32-39 ALINMA (T2)**, 40-47 GÖNDERME (T3).
    # T2'yi 24'ten okuyan bir uygulama, gönderdiğimiz SIFIR damgayı okur ve offset'i
    # NTP dönem farkı kadar (−2.208.988.800 s) kaydırır — ölçüldü: −1,99e12 ms.
    rx_s, rx_f, tx_s, tx_f = struct.unpack("!IIII", veri[32:48])
    t2 = (rx_s - NTP_EPOCH_DELTA) + rx_f / 2 ** 32
    t3 = (tx_s - NTP_EPOCH_DELTA) + tx_f / 2 ** 32
    offset = ((t2 - t1) + (t3 - t4)) / 2.0
    gidis_donus = (t4 - t1) - (t3 - t2)
    ms = offset * 1000.0
    return {
        "measured": True,
        "reason": None,
        "server": server,
        "offset_ms": ms,
        "roundtrip_ms": gidis_donus * 1000.0,
        "warn": abs(ms) > DRIFT_WARN_MS,
        "threshold_ms": DRIFT_WARN_MS,
    }


def _cli() -> None:                              # pragma: no cover - operatör aracı
    import argparse

    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Karar zinciri doğrulaması + saat kayması")
    ap.add_argument("--verify", action="store_true", help="zinciri doğrula")
    ap.add_argument("--seal", action="store_true", help="doğrula + Merkle kökü bas")
    ap.add_argument("--clock", action="store_true", help="NTP saat kaymasını ölç")
    a = ap.parse_args()
    if a.clock or not (a.verify or a.seal):
        d = clock_drift()
        if d["measured"]:
            print(f"Saat kayması: {d['offset_ms']:+.1f} ms "
                  f"(tur {d['roundtrip_ms']:.1f} ms) — "
                  f"{'UYARI' if d['warn'] else 'normal'}")
        else:
            print(f"Saat kayması ÖLÇÜLEMEDİ: {d['reason']}")
    if a.verify or a.seal:
        d = seal(CHAIN_PATH) if a.seal else verify(CHAIN_PATH)
        print(f"Zincir: {'SAĞLAM' if d['ok'] else 'BOZUK'} · {d['n']} halka"
              + (f" · ilk bozuk satır {d['broken_at']} ({d['reason']})"
                 if not d["ok"] else ""))
        if a.seal and d.get("merkle_root"):
            print(f"Merkle kökü: {d['merkle_root']}")


if __name__ == "__main__":                       # pragma: no cover
    _cli()


__all__ = ["CHAIN_PATH", "GENESIS", "append", "clock_drift", "last_hash",
           "merkle_root", "seal", "verify"]
