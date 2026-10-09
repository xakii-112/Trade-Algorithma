"""Tur-61 kilidi — KARAR KÖPRÜLERİNİN YÜZEYİ + BELGE↔KOD sözleşmesi.

Bu turun üç bulgusu da aynı sınıftandır ve projenin kendi kuralının (Tur-38:
*yazılıp okunmayan kolon kusurdur*) üç ayrı yüzeydeki tekrarıdır:

  * **BULGU-61-2** — `ai_advisor.status()` (Tur-38) ve `shadow_bridge.status()`
    (Tur-60) docstring'lerinde "pano/günlük özet için" yazıyordu ama ikisinin de tek
    çağıranı TESTTİ. Üçüncü kardeş `neural_shadow.status()` `/api/ai/overview`da
    zaten yayındaydı — desen vardı, tam da CANLI KARAR yoluna bağlı iki köprü açıkta
    kalmıştı. Önemi `last_error`dadır: köprüler fail-soft'tur ve ilk hatayı BİR KEZ
    uyarı, gerisini debug olarak loglar; 7/24 koşan bir tarayıcının system.log'unda o
    satır görünmez. `status()` kalıcı yüzeydi ve okunmadığı için "sessiz ölüm yok"
    koruması KENDİSİ sessizdi.
  * **BULGU-61-3** — üretim kodunun okuduğu 23 env adı hiçbir belgede yoktu.
  * **BULGU-61-4** — CLAUDE.md `python -m nihai.main` diyordu; öyle bir modül YOK
    (doğrusu `nihai.execution.main`). Her oturum bu dosyayı okuyarak başlar.

Kilit BİR KEZLİK DÜZELTMEYİ DEĞİL KURALI korur: yeni bir env okuması belgesiz kalırsa,
belgeye var olmayan bir modül yazılırsa ya da panonun okuduğu bir alan uçtan düşerse
test KIRMIZI olur.
"""
from __future__ import annotations

import ast
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nihai.web.web_dashboard import app

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


# --------------------------------------------------------------- BULGU-61-2: köprüler
def test_health_ucu_kopru_durumunu_tasiyor():
    """`/api/system/health` iki köprünün de durumunu YAYINLAR (uç, ok, alanlar)."""
    body = client.get("/api/system/health").json()
    br = body.get("bridges")
    assert isinstance(br, dict) and br.get("ok") is not False, f"bridges okunamadi: {br}"
    assert set(br) == {"ai_advisor", "shadow_bridge"}
    # Sözleşme ALAN ALAN kilitlenir: Tur-30'da pano var OLMAYAN anahtarları okuyordu ve
    # kart sessizce "durdu · 0 · —" gösteriyordu. Bir alanın adı değişirse burası kırılır.
    assert {"mode", "n_calls", "n_veto", "last_error"} <= set(br["ai_advisor"])
    assert {"mode", "calls", "with_marks", "promoted", "neural", "last_error",
            "rules"} <= set(br["shadow_bridge"])


def test_kopru_durumunun_URETIM_okuyucusu_var():
    """`status()` en az bir ÜRETİM dosyasından çağrılmalı — testten değil.

    BULGU-61-2 tam olarak buydu: iki `status()` de yalnız testlerden çağrılıyordu.
    Bir sonraki tur bu okuyucuyu silerse (ya da yeni bir köprü aynı hatayla doğarsa)
    kilit kırmızıya döner."""
    kaynak = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in (ROOT / "nihai").rglob("*.py"))
    for modul in ("ai_advisor", "shadow_bridge"):
        assert re.search(rf"\b{modul}\.status\(\)", kaynak), (
            f"{modul}.status() uretimde HIC okunmuyor — yazilip okunmayan yuzey kusurdur")


def test_pano_yalnizca_UCUN_verdigi_alanlari_okuyor():
    """Pano ↔ uç sözleşmesi: HTML'in `aa.`/`sb.` ile okuduğu her alan uçta VAR olmalı.

    Tur-29/30 dersinin makine hâli: pano var olmayan anahtar okursa hata YOK, yalnız
    daima yanlış bir kart olur — yani en sinsi kusur türü."""
    html = (ROOT / "nihai" / "web" / "web_dashboard.html").read_text(encoding="utf-8")
    blok = html.split("const br = d.bridges || {};", 1)
    assert len(blok) == 2, "pano kopru karti kayboldu"
    govde = blok[1].split("const env = d.env", 1)[0]

    body = client.get("/api/system/health").json()["bridges"]
    for takma, gercek in (("aa", "ai_advisor"), ("sb", "shadow_bridge")):
        okunan = set(re.findall(rf"\b{takma}\.([a-z_]+)\b", govde))
        eksik = okunan - set(body[gercek])
        assert not eksik, f"pano {gercek} icin OLMAYAN alan okuyor: {sorted(eksik)}"
        assert okunan, f"pano {gercek} alanlarini hic okumuyor"


def test_advise_kipi_karta_ENFORCE_gibi_yansimaz():
    """Kart kipin kenar iddiası taşımadığını göstermeli: `promoted` boşken 'YOK' yazar."""
    html = (ROOT / "nihai" / "web" / "web_dashboard.html").read_text(encoding="utf-8")
    assert "YOK (hepsi gölgede)" in html
    body = client.get("/api/system/health").json()["bridges"]
    assert body["shadow_bridge"]["promoted"] == [], (
        "terfi eden bir kural varsa kart metni ve shadow_bridge._enforce_hukmu birlikte "
        "guncellenmeli (Tur-60 defter sozlesmesi)")


def test_health_ucu_koprulere_AST_ile_bagli():
    """`/api/system/health` gövdesi iki köprüyü de ADIYLA çağırmalı.

    Metin araması bir yorumla da tatmin olurdu; bu yüzden çağrı AST'ten okunur."""
    src = (ROOT / "nihai" / "web" / "api_extra.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "system_health")
    cagrilar = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert "ai_advisor.status" in cagrilar
    assert "shadow_bridge.status" in cagrilar


# ------------------------------------------------------- BULGU-61-4: belgelenen komutlar
_STDLIB_M = {"venv", "pip", "pytest", "ruff", "mypy", "http.server"}


@pytest.mark.parametrize("belge", ["CLAUDE.md", "README.md"])
def test_belgelenen_python_m_modulleri_GERCEKTEN_var(belge):
    """Belgedeki her `python -m X` çözümlenebilir bir modüle işaret etmeli.

    BULGU-61-4: CLAUDE.md'nin "Komutlar" bölümü `python -m nihai.main` diyordu ve o
    modül YOKTU (doğrusu `nihai.execution.main`) — yani her oturumun okuduğu
    sözleşmenin İLK çalıştırma komutu yanlıştı."""
    metin = (ROOT / belge).read_text(encoding="utf-8", errors="replace")
    eksik = []
    for mod in sorted(set(re.findall(r"python -m ([a-zA-Z0-9_.]+)", metin))):
        if mod in _STDLIB_M:
            continue
        yol = ROOT / Path(mod.replace(".", "/"))
        if not (yol.with_suffix(".py").exists() or (yol / "__init__.py").exists()
                or (yol / "__main__.py").exists()):
            eksik.append(mod)
    assert not eksik, f"{belge} olmayan modulu belgeliyor: {eksik}"


# ------------------------------------------------------------ BULGU-61-3: env beyanı
_ENV_OKUMA = re.compile(
    r"""(?:os\.environ\.get|os\.getenv|os\.environ\[|environ\.get|environ\["""
    r"""|_envb|_envf|_envi)\s*\(?\s*["']([A-Z][A-Z0-9_]{2,})["']""")


def _kodun_okudugu_env() -> dict[str, str]:
    bulunan: dict[str, str] = {}
    for paket in ("nihai", "research", "tools", "trade"):
        for p in (ROOT / paket).rglob("*.py"):
            for ad in _ENV_OKUMA.findall(p.read_text(encoding="utf-8", errors="replace")):
                bulunan.setdefault(ad, str(p.relative_to(ROOT)))
    return bulunan


def _beyan_edilen_env() -> set[str]:
    metin = (ROOT / ".env.example").read_text(encoding="utf-8", errors="replace")
    # Yorumlu satır da BEYANDIR: `# X=` "bu düğme var, varsayılanı kullan" demektir.
    return set(re.findall(r"^\s*#?\s*([A-Z][A-Z0-9_]{2,})\s*=", metin, flags=re.M))


def test_uretimin_okudugu_her_env_belgede_beyanli():
    """Belgesiz bir düğme çevrilmeyen düğmedir (BULGU-61-3).

    Somut kanıt: Tur-60 raporu operatöre "NEURAL_SHADOW açık kalsın" diyordu ve o ad
    `.env.example`te YOKTU. Denetimin ters yönü zaten temizdi — beyan edilen 187 adın
    hepsi kodda okunuyordu; eksik olan yön buydu."""
    beyan = _beyan_edilen_env()
    eksik = {k: v for k, v in _kodun_okudugu_env().items() if k not in beyan}
    assert not eksik, (
        "kodun okudugu ama .env.example'te beyani olmayan ayarlar: "
        + ", ".join(f"{k} ({v})" for k, v in sorted(eksik.items())))


def test_env_beyan_kilidi_gercekten_eliyor():
    """Kapı ÖLÜ olmasın: uydurma bir ad eklenirse denetim onu YAKALAMALI.

    (Tur-60 dersi: yalnız reddeden bir kapı kapı değil bir `return False`tir — burada
    tersi geçerli: yalnız kabul eden bir kapı da kapı değildir.)"""
    beyan = _beyan_edilen_env()
    assert "ASLA_TANIMLANMAYAN_AYAR" not in beyan
    sahte = {"ASLA_TANIMLANMAYAN_AYAR": "sahte.py"}
    assert {k: v for k, v in sahte.items() if k not in beyan} == sahte


# ------------------------------------------- BULGU-61-7: yazilip KOSULMAYAN izleyici
def _ps1_metinleri() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8", errors="replace")
            for p in ROOT.glob("*.ps1")}


def test_her_register_scripti_VAR_OLAN_bir_sarmalayiciya_isaret_ediyor():
    """`register-X.ps1` olmayan bir `X.ps1`i kaydederse görev ilk tetikte ölür."""
    metinler = _ps1_metinleri()
    eksik = []
    for ad, metin in metinler.items():
        if not ad.startswith("register-"):
            continue
        for hedef in set(re.findall(r'Join-Path \$scriptDir "([A-Za-z0-9_.-]+\.ps1)"', metin)):
            if not (ROOT / hedef).exists():
                eksik.append(f"{ad} -> {hedef}")
    assert not eksik, f"kayit scripti olmayan sarmalayiciya isaret ediyor: {eksik}"


def test_ps1_sarmalayicilarinin_cagirdigi_python_modulu_VAR():
    """Ops sarmalayıcıları da bir SÖZLEŞMEDİR: `-m X` çözümlenebilmeli.

    CLAUDE.md kilidinin (BULGU-61-4) işletim tarafı — orada belge, burada zamanlanmış
    görevin fiilen çalıştırdığı komut yanlış olabilirdi."""
    eksik = []
    for ad, metin in _ps1_metinleri().items():
        for mod in set(re.findall(r'"-m",\s*"([a-zA-Z0-9_.]+)"', metin)
                       + re.findall(r'-m\s+((?:nihai|research|tools)[a-zA-Z0-9_.]*)', metin)):
            yol = ROOT / Path(mod.replace(".", "/"))
            if not (yol.with_suffix(".py").exists() or (yol / "__init__.py").exists()):
                eksik.append(f"{ad} -> {mod}")
    assert not eksik, f"ps1 olmayan modulu calistiriyor: {eksik}"


def test_olu_adam_anahtarinin_KOSUCUSU_ve_KAYDI_var():
    """BULGU-61-7: `deadman.py` (Tur-59) tam da "servisler sessizce öldü" hâlini
    yakalamak için yazılmıştı ama onu ÇALIŞTIRAN sarmalayıcı ve görev YOKTU — modül
    üretimde hiç koşmadı. Ölçüldü (2026-08-31): üç servis portu da kapalı,
    `predictions_log` 35 sa / `live_features_log` 87 sa bayat, tek uyarı çıkmadı.

    Kayıt ADIMI operatörün bilinçli tercihidir (modülün kendi docstring'i böyle diyor);
    kilit yalnız ARACIN var ve doğru modelde olduğunu garanti eder."""
    metinler = _ps1_metinleri()
    assert "deadman.ps1" in metinler, "olu adam anahtarinin kosucusu YOK"
    assert "nihai.notifier.deadman" in metinler["deadman.ps1"]
    kayit = metinler.get("register-deadman.ps1")
    assert kayit, "olu adam anahtarinin gorev kaydi YOK"
    # Tur-12 dersi: SYSTEM hesabi OneDrive profiline erisemez -> interactive kullanici.
    # Denetim KODA bakar, ANLATIYA degil (Tur-49): script'in kendi yorumu "SYSTEM" kelimesini
    # zaten geciyor — ilk surum tam olarak ona takildi (bu turda ucuncu tekrar).
    assert "LogonType Interactive" in kayit
    assert "WindowsIdentity" in kayit, "principal mevcut kullanicidan turetilmiyor"
    for tehlike in ('-UserId "SYSTEM"', "-UserId 'SYSTEM'", "S-1-5-18", "RunLevel Highest"):
        assert tehlike not in kayit, f"gorev yukseltilmis/SYSTEM olarak kaydediliyor: {tehlike}"


# --------------------------------- BULGU-61-8: bilincli kapatmaya BAGIRAN izleyici
def test_deadman_PROJE_KAPALIYKEN_alarm_vermez(monkeypatch, tmp_path):
    """`proje-kapat.ps1` bayragi varsa bayatlik BEKLENEN durumdur, alarm degil.

    Kusur bu turda `deadman`i 15 dk'lik bir goreve baglarken ortaya cikti: operator
    2026-08-30'da projeyi bilerek kapatmisti (`logs/project.stopped`) ve watchdog'lar
    bayragi GORUP yeniden baslatmayi birakmisti — `deadman` ise bayragi bilmiyordu, yani
    seans boyunca her 15 dakikada bir yanlis alarm uretecekti. Bilincli bir duruma
    bagiran izleyicinin sonu SUSTURULMAKTIR; asil ariza gunu susturulmus bir alarm hic
    olmayandan kotudur."""
    import os as _os

    from nihai.notifier import deadman as DM
    monkeypatch.setattr(DM, "_market_open", lambda: True)
    veri, loglar = tmp_path / "data", tmp_path / "logs"
    veri.mkdir(); loglar.mkdir()
    bayat = veri / "predictions_log.csv"
    bayat.write_text("x", encoding="utf-8")
    eski = time.time() - 3600 * 5
    _os.utime(bayat, (eski, eski))

    # Bayrak YOKken: gercek bir alarm (kilit once kapinin CALISTIGINI gosterir)
    once = DM.check(data_dir=veri, log_dir=loglar)
    assert once["verdict"] == "BAYAT" and once["stale"] == ["predictions_log"]

    # Bayrak VARken: ayni dosya, ayni yas -> alarm YOK ama OLCUM gizlenmiyor
    (loglar / "project.stopped").write_text("KAPATILDI", encoding="utf-8")
    sonra = DM.check(data_dir=veri, log_dir=loglar)
    assert sonra["verdict"] == "PROJE_KAPALI"
    assert sonra["stale"] == []
    assert sonra["project_stopped"] is True
    assert sonra["files"]["predictions_log"]["stale"] is True     # olcum RAPORLANIYOR
    assert "project.stopped" in DM.format_report(sonra)           # nedeni GORUNUR
