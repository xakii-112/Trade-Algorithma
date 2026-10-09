"""atomic_io.py — OneDrive/AV kilidine dayanıklı atomik dosya yazımı (TEK doğruluk kaynağı).

Neden: 2026-07-11 02:04 olayı — forward-harness `os.replace(tmp, predictions_log.csv)`
OneDrive senkron sürecinin anlık dosya kilidiyle `PermissionError: [WinError 5]` alıp TÜM
harness'i (resolve+snapshot+günlük özet) çökertti; o gün hiçbir tahmin çözülmedi (O5'in
belgelenmiş riski gerçekleşti). Tur-11 bunu YALNIZ prediction_tracker'da (`_atomic_replace`)
düzeltmişti; Denetim Tur-12 aynı ham `os.replace`'in state.json (portföy + açık pozisyonlar,
~60sn'de bir yazılır — çok daha kritik), watchlist ve .env yazımlarında da açık olduğunu buldu.
Bu modül o düzeltmeyi TEK yerde toplar; kritik yazımlar HEP buradan geçer.

Windows'ta AV/senkron kilitleri tipik olarak alt-saniye ömürlü → kısa üstel geri-çekilmeli
yeniden deneme (0.1·2^k; 6 deneme → ~3.1sn toplam bekleme) çoğu kilidi bekler. Son denemede de
başarısızsa orijinal istisna YÜKSELİR (sessizce yutulmaz — çağıran/log hatayı görür).

TMP ADI YAZANA ÖZELDİR (Tur-28). Eskiden ara dosya SABİT `<dosya>.tmp` idi: aynı hedefe yazan
iki YAZAN (canlı kayıtçı ↔ `research/forward_harness`, watchdog yeniden başlatırken üst üste
binen iki süreç, ya da tek süreçte iki thread) AYNI tmp'yi paylaşıyordu. Sıra
`A: tmp'ye yaz → B: tmp'yi KIRPIP yazmaya başla → A: replace` olduğunda hedefe B'nin YARIM
içeriği atomik olarak yayınlanır — atomiklik korunur ama içerik BOZUKTUR. Etki alanı bu
modülün koruduğu en kritik dosyalar: `state.json` (portföy + açık pozisyonlar),
`predictions_log.csv`, `ai_learning_state.json`, model kartları. `tmp_path()` ara dosyayı
pid+thread ile ayırır → yazanlar birbirinin tmp'sine dokunamaz. Hata yolunda tmp SİLİNİR
(benzersiz ad artık kendini üzerine yazmaz; Tur-26'daki tmp sızıntısı tekrarlanmasın).

KUYRUK OKUMA (Tur-35). Aynı append-only devasa logları YAZAN modüller onları OKUMAK da
zorunda (`predictions_log.csv` bugün 8,4 MB / 62k satır ve yalnız büyür). `tail_lines`
o okumanın TEK doğruluk kaynağıdır — üç ayrı modülde (`ai/learning`, `ai/calibration_loop`,
`ai/reasoner`) BİREBİR kopya hâlinde ve üçünde de KUADRATİK olarak yaşıyordu: her 8 KB'lik
adımda biriken tamponun TAMAMI yeniden `count(b"\\n")` ediliyor ve `f.read(step) + data`
ile baştan kopyalanıyordu. Ölçüldü: 8,4 MB dosyada 1020 yineleme = 4,3 GB tarama, 2,10 s.
Doğru desen zaten `reasoner._count_data_lines`'ta duruyordu (yalnız YENİ parçayı say) —
ders MODÜLE değil DESENE aitti. Yeni hâl parçaları listede biriktirir, yalnız yeni parçayı
sayar ve sonda bir kez birleştirir: O(okunan bayt). Çıktı BİREBİR aynıdır.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

_RETRIES = 6

# Kuyruk okuma bloğu GEOMETRİK büyür: küçük n (reasoner 500 satır ister) gereksiz yere 256 KB
# çözmesin, büyük n (learning 5000+) 8 KB'lik adımlarla 85 syscall'a girmesin. Çıktı blok
# boyutundan BAĞIMSIZDIR: döngü "en az n+1 satır sonu biriktir" der, daha büyük blok yalnız
# DAHA FAZLA satır getirebilir ve fazlası sondaki `[-n:]` dilimiyle atılır.
_TAIL_CHUNK_MIN = 1 << 14      # 16 KB
_TAIL_CHUNK_MAX = 1 << 20      # 1 MB


def tek_yazar_kilidi(yol: Path):
    """`yol`da süreç ömürlü, ENGELSİZ özel kilit (Windows `msvcrt` / POSIX `flock`). Alınırsa açık dosya nesnesi
    döner (tutulduğu sürece kilitli; süreç ölünce işletim sistemi bırakır), başka süreç tutuyorsa None. Tur-102
    gölge kasaların `motor.lock`u ve Tur-106 ana süreç kilidi aynı yardımcıyı kullanır."""
    yol.parent.mkdir(parents=True, exist_ok=True)
    f = yol.open("a+b")
    try:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def tmp_path(dst: Path) -> Path:
    """`dst` ile AYNI dizinde, bu yazana (pid+thread) özel ara dosya yolu.

    Aynı volüm şart: `os.replace` yalnız aynı dosya sisteminde atomiktir. Ad `dst`'nin
    yanında durur (`state.json.12345.6789.tmp`) → `.gitignore`'daki `*.tmp` deseni tutar."""
    return dst.with_suffix(f"{dst.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")


def tail_lines(path: Path, n: int) -> list[str]:
    """Dosyanın SON n boş-olmayan satırı (sondan bloklar okuyarak; devasa logda ucuz).

    Dosya sondan geriye doğru bloklar hâlinde okunur; en az n+1 satır sonu biriktiğinde
    durulur — böylece blok sınırında kalan YARIM ilk satır her zaman `[-n:]` diliminin
    dışında kalır. Okunamayan dosya → `[]` (çökme yok; çağıranlar "log henüz yok" hâlini
    boş liste olarak bekler).

    Karmaşıklık O(okunan bayt): her blok YALNIZ BİR KEZ sayılır ve birleştirme sonda tek
    `b"".join` ile yapılır (bkz. modül docstring'i — eski kuadratik hâl 8,4 MB'de 2,10 s
    harcıyordu).

    `n <= 0` → `[]`. Eski kopyalarda bu dal SESSİZ KUSURLUYDU: `lst[-0:]` Python'da `lst[0:]`
    demektir, yani "sıfır satır iste" TÜM okunan bloğu geri veriyordu ve satır sayısı BLOK
    BOYUTUNA bağlıydı (8 KB'de 70, 256 KB'de 1693) — tanımsız davranış. `AI_REASONER_PRED_TAIL=0`
    ile tetiklenebilirdi."""
    if n <= 0:
        return []
    chunks: list[bytes] = []
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            pos = f.tell()
            newlines = 0
            step_size = _TAIL_CHUNK_MIN
            while pos > 0 and newlines <= n:
                step = min(step_size, pos)
                pos -= step
                f.seek(pos)
                chunk = f.read(step)
                newlines += chunk.count(b"\n")
                chunks.append(chunk)
                step_size = min(step_size * 2, _TAIL_CHUNK_MAX)
    except OSError:
        return []
    data = b"".join(reversed(chunks))
    return [p.decode("utf-8", "replace") for p in data.split(b"\n") if p.strip()][-n:]


def atomic_replace(tmp: Path, dst: Path, *, retries: int = _RETRIES) -> None:
    """`os.replace(tmp, dst)` + kısa üstel geri-çekilmeli yeniden deneme (OneDrive/AV kilidi).

    tmp aynı dizinde önceden yazılmış olmalı (os.replace aynı volümde atomiktir); yolu
    `tmp_path(dst)` ile üret ki başka bir yazanla çakışmasın. Son denemede de OSError alınırsa
    istisna YÜKSELİR — veri sessizce kaybolmaz. Kalıcı hatada tmp'yi çağıran temizlemeli
    (`atomic_write_text` bunu yapar)."""
    delay = 0.1
    for attempt in range(retries):
        try:
            os.replace(tmp, dst)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8",
                      retries: int = _RETRIES, newline: str | None = None) -> None:
    """`text`'i `path`'e atomik yaz: aynı dizinde `.tmp`'ye yaz → `atomic_replace`.

    Yazma sırasında çökme/SIGKILL hedef dosyayı YARIM bırakmaz (ya eski ya yeni tam içerik);
    OneDrive kilidi replace'i düşürmez (yeniden dener). Kalıcı hatada istisna yükselir ve
    yarım tmp SİLİNİR (ad yazana özel olduğundan kimse onu üzerine yazmayacak)."""
    tmp = tmp_path(path)
    try:
        tmp.write_text(text, encoding=encoding, newline=newline)
        atomic_replace(tmp, path, retries=retries)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass       # temizlik en iyi çaba — asıl hatayı gölgeleme
        raise
