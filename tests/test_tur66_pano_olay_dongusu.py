"""Tur-66 — panonun uçları OLAY DÖNGÜSÜNÜ bloke etmesin.

**Bulgu.** `nihai/web/web_dashboard.py`'deki **23 uç `async def`** ilan edilmişti ve
hiçbirinde **tek bir `await` yoktu** (AST ile sayıldı: 23/23). FastAPI `async def` bir
ucu thread havuzuna GÖNDERMEZ — doğrudan olay döngüsü thread'inde koşturur. Dolayısıyla
bu 23 ucun yaptığı her bloklayan iş (dosya okuma, `provider.get_quote()` ağ turu,
`PORTFOLIO_LOCK` beklemesi) tek döngü thread'ini kilitliyordu: sıradaki İSTEKLER DE
donuyordu (head-of-line blocking).

**Neden bu depoda ağırdı.** `main --serve-web` panoyu TARAYICIYLA AYNI SÜREÇTE kaldırır
(`main.py:1048` → `start_web_server()` uvicorn'u daemon thread'te başlatır, sonra
`run_loop()` `LIVE_SCAN_WORKERS=8` thread ile tarar). Üçü de aynı `PORTFOLIO_LOCK`
(`threading.RLock`) üzerinde buluşur. 23 ucun **14'ü** `with PORTFOLIO_LOCK` alıyordu →
bir tarama worker'ı kilidi tutup ağ beklerken, panonun döngü thread'i o kilitte
BLOKLANIYOR ve pano tümden yanıt veremez hale geliyordu.

**Onarım.** 23 ucun ilanı `async def` → `def`. Tek kelime; gövde, sözleşme, dönüş tipi
ve kilit sırası AYNI. FastAPI sync bir ucu kendi thread havuzuna gönderir → uç bloke
olsa bile döngü serbest kalır. `await` KULLANAN tek fonksiyon (`_web_token_guard`
middleware'i, `await call_next(request)`) bilinçli olarak `async def` KALIR.

**ÖLÇÜLDÜ** (gerçek pano + gerçek `PORTFOLIO_LOCK` + tarayıcı deseni: kilidi tutup
250 ms ağ bekle; pano `/api/status`u tazelerken kilit-ALMAYAN uçlar 8 eşzamanlı
istemciyle ölçüldü; iki yineleme):

| | ÖNCE (async def) | SONRA (def) | oran |
|---|---|---|---|
| verim | 27,8 · 28,0 istek/sn | 69,3 · 68,8 istek/sn | **2,47×** |
| p50 | 277,7 · 274,3 ms | 111,7 · 113,8 ms | **2,45×** |
| p95 | 318,7 · 343,8 ms | 167,2 · 182,8 ms | 1,94× |
| en kötü | 496,2 · 507,6 ms | 244,5 · 252,8 ms | 2,03× |

Mekanizma yalıtılmış olarak da ölçüldü (250 ms bloklayan uç eşzamanlı koşarken hafif
ucun verimi): `async def` **28,2 istek/sn** ↔ `def` **542,3 istek/sn** = **19,2×**;
en kötü gecikme 889,0 → 23,2 ms. Depodaki oran (2,47×) daha küçüktür çünkü gerçek
uçların işi GIL'e bağlı Python'dur — kazanç saf bloklama payı kadardır.

**Katalog bağlamı.** Dış katalog #11 "winloop/uvloop kur, 2-3× asenkron hız" diyor.
ÖLÇÜLDÜ ve bu depo için REDDEDİLDİ: winloop `async def` + işsiz uçta gerçekten kazanıyor
(64 eşzamanlıda 259,1 → 553,8 istek/sn = 2,14×) ama deponun uçları `def`tir ve iş
YAPARLAR; gerçek hub şeklinde winloop 0,92–0,99× ile **daha yavaştı**. Yani kataloğun
öncülü doğru, ilacı yanlış: darboğaz olay döngüsünün İMPLEMENTASYONU değil, döngüye
bloklayan iş KOYULMASIYDI. Bağımlılık eklemeden, ilan değiştirilerek çözüldü.
"""
from __future__ import annotations

import ast
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

KOK = pathlib.Path(__file__).resolve().parents[1]
PANO = KOK / "nihai" / "web" / "web_dashboard.py"

# Uç sayılan dekoratörler. `websocket` BİLEREK dışarıda: websocket handler'ı
# protokol gereği `async def` OLMALIDIR ve bu kuralın istisnasıdır.
_UC_DEKORATORLERI = (".get(", ".post(", ".put(", ".patch(", ".delete(")


def _uc_dekoratoru(node: ast.AST) -> str | None:
    for d in getattr(node, "decorator_list", []):
        s = ast.unparse(d)
        if "websocket" in s:
            return None
        if any(k in s for k in _UC_DEKORATORLERI):
            return s
    return None


def _await_var(node: ast.AST) -> bool:
    return any(isinstance(a, ast.Await) for a in ast.walk(node))


# Taranan ÜRETİM paketleri. `tests/` BİLEREK dışarıda: bu dosyanın kendi
# `test_bloklayan_uc_komsusunu_dondurmaz` testi, mekanizmayı göstermek için KASITLI
# bir await'siz `async def` uç kurar. Kapsam daraltılmasaydı tarayıcı kendi karşı-
# örneğini suçlu sayardı (Tur-62'nin "ölü kapı testi kendini sayıyordu" dersi).
URETIM_PAKETLERI = ("nihai", "research", "tools")


def _py_dosyalari() -> list[pathlib.Path]:
    yok = {".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".git", ".pytest_cache"}
    return [p for paket in URETIM_PAKETLERI for p in (KOK / paket).rglob("*.py")
            if not (yok & set(p.parts))]


# --- 1. Kök neden: await'siz async uç TÜM ÜRETİM KODUNDA kalmamalı ----------------
def test_hicbir_ucta_awaitsiz_async_def_kalmadi():
    """Kural TEK DOSYAYA değil ÜRETİM PAKETLERİNE konur — kardeş app'ler (`hub_server`,
    `api_quant`, `api_extra`) aynı tuzağa düşmesin.

    `async def` + sıfır `await` = FastAPI'nin thread havuzunu ATLAYIP olay döngüsünde
    koşan bir uç. Yeni bir uç bu şekilde eklenirse bu test onu yakalar."""
    dosyalar = _py_dosyalari()
    # KAPSAM KİLİDİ: tarayıcı bir gün boşa düşerse (paket taşındı, glob bozuldu) test
    # sessizce YEŞİL kalmasın — Tur-50/Tur-26'nın "kalite kapısı kördü" dersi.
    assert PANO in dosyalar, f"tarama panoyu KAPSAMIYOR — kapsam körleşmiş ({len(dosyalar)} dosya)"
    assert len(dosyalar) > 100, f"üretim taraması beklenenden dar: {len(dosyalar)} dosya"

    suclular = []
    for p in dosyalar:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.AsyncFunctionDef) and _uc_dekoratoru(n) and not _await_var(n):
                suclular.append(f"{p.relative_to(KOK)}:{n.lineno} {n.name}")
    assert not suclular, (
        "await KULLANMAYAN `async def` uç bulundu — bunlar olay döngüsünde koşar ve "
        "bloklayan işleri (dosya/ağ/kilit) tüm panoyu dondurur. `def` yapın:\n  "
        + "\n  ".join(suclular)
    )


# --- 2. Panonun uçları sync, middleware async KALIR --------------------------------
def test_panonun_tum_uclari_sync_def():
    tree = ast.parse(PANO.read_text(encoding="utf-8"))
    async_uc = [n.name for n in ast.walk(tree)
                if isinstance(n, ast.AsyncFunctionDef) and _uc_dekoratoru(n)]
    assert async_uc == [], f"pano ucu async kalmış: {async_uc}"


def test_middleware_async_kalir():
    """`_web_token_guard` `await call_next(request)` yapar — sync'e ÇEVRİLEMEZ.

    Bu testin varlığı bilinçlidir: yukarıdaki "hepsini def yap" kuralı bir gün
    middleware'e de uygulanırsa yetki kapısı çöker (Starlette middleware'i awaitable
    bekler). Kural uçlar içindir, middleware için DEĞİL."""
    tree = ast.parse(PANO.read_text(encoding="utf-8"))
    mw = [n for n in ast.walk(tree)
          if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
          and n.name == "_web_token_guard"]
    assert len(mw) == 1, "_web_token_guard bulunamadı"
    assert isinstance(mw[0], ast.AsyncFunctionDef), "middleware sync'e çevrilmiş — await'i var"
    assert _await_var(mw[0]), "middleware'in await'i kaybolmuş"


# --- 3. PORTFOLIO_LOCK alan uçlar döngü thread'inde beklememelidir -----------------
def test_kilit_alan_uclar_dongude_kosmuyor():
    """14 uç `with PORTFOLIO_LOCK` alır; kilidi TARAMA THREAD'LERİ de tutar.

    Kilidi olay döngüsü thread'inde beklemek = pano tümden donar. Hepsi `def` olmalı."""
    tree = ast.parse(PANO.read_text(encoding="utf-8"))
    kilitli_async = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.AsyncFunctionDef) and _uc_dekoratoru(n)):
            continue
        if any(isinstance(w, ast.With)
               and any("PORTFOLIO_LOCK" in ast.unparse(i.context_expr) for i in w.items)
               for w in ast.walk(n)):
            kilitli_async.append(n.name)
    assert kilitli_async == [], f"kilit alan uç olay döngüsünde: {kilitli_async}"


def _sure(c: TestClient) -> float:
    a = time.perf_counter()
    c.get("/hafif")
    return time.perf_counter() - a


# --- 4. Davranış: bloklayan bir uç komşusunu DONDURMAMALI --------------------------
@pytest.mark.parametrize("sync_uc", [True, False])
def test_bloklayan_uc_komsusunu_dondurmaz(sync_uc: bool):
    """Mekanizmanın kendisi — iddia değil, KOŞAN kanıt.

    Aynı iş yükü iki kez kurulur; tek fark ilanın `def` mi `async def` mi olduğudur.
    `async def` dalında hafif uç, ağır ucun ARKASINDA sıraya girer (gecikme ≈ blok
    süresi); `def` dalında girmez. Eşik gevşek tutuldu (yavaş makinede de geçsin):
    ölçülen fark 19,2× idi, test yalnız 3× ister."""
    BLOK = 0.20
    app = FastAPI()
    if sync_uc:
        @app.get("/agir")
        def agir():
            time.sleep(BLOK)
            return {"ok": True}

        @app.get("/hafif")
        def hafif():
            return {"ok": True}
    else:
        @app.get("/agir")
        async def agir_a():
            time.sleep(BLOK)      # await YOK → döngüyü kilitler (deponun eski hali)
            return {"ok": True}

        @app.get("/hafif")
        async def hafif_a():
            return {"ok": True}

    with TestClient(app) as c:
        c.get("/hafif")
        dur = threading.Event()

        def agir_yuk():
            while not dur.is_set():
                try:
                    c.get("/agir")
                except Exception:
                    return
        t = threading.Thread(target=agir_yuk, daemon=True)
        t.start()
        time.sleep(BLOK / 2)
        try:
            with ThreadPoolExecutor(max_workers=4) as ex:
                gec = sorted(ex.map(lambda _: _sure(c), range(12)))
        finally:
            dur.set()
            t.join(timeout=5)
    p50 = gec[len(gec) // 2]
    if sync_uc:
        assert p50 < BLOK / 3, f"sync uçta hafif istek yine de bekledi: p50={p50*1000:.0f} ms"
    else:
        assert p50 > BLOK / 3, (
            f"async uçta bloklama görülmedi (p50={p50*1000:.0f} ms) — testin öncülü "
            "geçersizleşmiş olabilir (FastAPI davranışı değiştiyse kuralı gözden geçir)"
        )
