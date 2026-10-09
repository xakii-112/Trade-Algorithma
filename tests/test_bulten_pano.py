"""Bülten — mevcut panodan BAĞIMSIZ ikinci arayüz (nihai/web/bulten.html + bulten_sunucu.py).

Dört soru:
  * BAĞLANDI MI — /bulten sayfası sunuluyor, token same-origin enjekte ediliyor, klasik pano
    (/) aynen duruyor; betiğin dokunduğu her öğe sayfada var, fihrist bölümlerle aynı sırada.
  * UÇLAR VAR MI — sayfanın çağırdığı her /api/* yolu uygulamada doğru METOTLA kayıtlı
    (bir uç yeniden adlandırılırsa bülten sessizce "—" basardı).
  * GÜVENLİ Mİ — API/LLM metni hiçbir yerde HTML olarak yorumlanmıyor; token `</script>`
    ile etiketi kapatamıyor; yazan uçlar yalnız operatör eylemleri.
  * HESAP UÇLARI YAZMAZ MI — POST olsa da yalnız hesaplayan iki uç (pozisyon boyutu, Beyne
    soru) durum dosyalarına dokunmuyor.
"""
import re
from pathlib import Path

from fastapi.testclient import TestClient

import nihai.web.bulten_sunucu as bs
from nihai.config import DATA_DIR
from nihai.web.web_dashboard import app

client = TestClient(app)
HTML = (Path(bs.__file__).with_name("bulten.html")).read_text(encoding="utf-8")
_m = re.search(r"<script>(.*)</script>", HTML, re.S)
assert _m, "bulten.html satır-içi betiği bulunamadı"
SCRIPT = _m.group(1)

# Bültenin yazdığı uçlar — bilinçli olarak üç operatör eylemiyle sınırlı.
YAZAN_UCLAR = {"close-position", "flatten", "reset-killswitch"}
# POST ama yalnız HESAPLAR: gövde parametre taşır, durum yazmaz (bkz. test_hesap_ucu_durum_yazmaz).
HESAP_UCLARI = {"tools/position-size", "ai/ask"}


def _kayitli() -> dict[str, set[str]]:
    rotalar: dict[str, set[str]] = {}
    for r in app.routes:
        yol, metotlar = getattr(r, "path", None), getattr(r, "methods", None)
        if yol and metotlar:
            rotalar.setdefault(yol, set()).update(metotlar)
    return rotalar


def _api_yollari() -> set[str]:
    # api('ai/brief') · api('analyze?symbol=' + …) · api('flatten', { method: 'POST' })
    return set(re.findall(r"\bapi\('([a-z0-9/_-]+)", SCRIPT))


def test_bulten_sunuluyor_ve_token_enjekte_ediliyor():
    r = client.get("/bulten")
    assert r.status_code == 200
    assert "<title>Nihai Bülten</title>" in r.text
    assert r.headers["cache-control"].startswith("no-store")
    # Enjeksiyon </head>'den ÖNCE ve tek kez (sayfanın kendi okuması ayrı sayılır).
    bas = r.text.split("</head>", 1)[0]
    assert bas.count("window.__WEB_TOKEN__ =") == 1


def test_token_script_etiketini_kapatamaz(monkeypatch):
    monkeypatch.setattr(bs, "WEB_TOKEN", "x</script><img src=y onerror=alert(1)>")
    r = client.get("/bulten")
    assert "</script><img" not in r.text
    assert "\\u003c/script>" in r.text


def test_klasik_pano_yerinde_ve_bultenden_bagimsiz():
    r = client.get("/")
    assert r.status_code == 200
    assert "Nihai Bülten" not in r.text          # klasik pano bültenin içeriğini taşımaz
    assert "switchTab(" in r.text                # mevcut sekmeli pano aynen sunuluyor


def test_sayfanin_cagirdigi_her_uc_dogru_metotla_kayitli():
    rotalar = _kayitli()
    yollar = _api_yollari()
    assert len(yollar) >= 45, yollar             # tarayıcı regex'i kör kalmasın (kendi kontrolü)
    assert YAZAN_UCLAR | HESAP_UCLARI <= yollar
    eksik = []
    for y in sorted(yollar):
        beklenen = "POST" if y in YAZAN_UCLAR | HESAP_UCLARI else "GET"
        if beklenen not in rotalar.get(f"/api/{y}", set()):
            eksik.append(f"{beklenen} /api/{y}")
    assert not eksik, f"bültenin çağırdığı ama uygulamada olmayan uçlar: {eksik}"
    # Bağlantı olarak verilen tearsheet de gerçek bir GET ucu.
    assert "GET" in rotalar.get("/api/tearsheet", set())


def test_yalniz_uc_operator_eylemi_yazar():
    # POST uçları = üç operatör eylemi + iki hesap ucu; yeni bir POST bu listeye bilinçli girmeli.
    post = set(re.findall(r"\bapi\('([a-z0-9/_-]+)'[^;]*?method:\s*'POST'", SCRIPT))
    assert post == YAZAN_UCLAR | HESAP_UCLARI


def test_hesap_ucu_durum_yazmaz(monkeypatch):
    # Pozisyon boyutu hesabı canlı portföyü OKUR (RiskManager.size_stock) ama hiçbir dosyaya yazmaz.
    # Veri dizini kök conftest ile geçici dizine yönlendirilir; oradaki her dosyanın imzası sabit kalmalı.
    # Rejim çarpanı SPY barı ister; testte ağ yasak → bar yok (uç bunu fail-soft karşılar).
    import nihai.web.api_extra as ae
    monkeypatch.setattr(ae, "_bars", lambda *a, **k: None)

    def imza():
        return {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in Path(DATA_DIR).rglob("*") if p.is_file()}

    once = imza()
    r = client.post("/api/tools/position-size", json={"symbol": "AAPL", "entry": 100.0, "stop": 95.0, "side": "LONG"},
                    headers={"X-Auth-Token": bs.WEB_TOKEN or ""})
    assert r.status_code == 200, r.text
    govde = r.json()
    assert govde["ok"] is True and "approved" in govde and "SİNYAL-ONLY" in govde["note"]
    assert imza() == once


def test_betikteki_her_kimlik_sayfada_var():
    # doldur('x') / yaz('x') / byId('x') hedefi yoksa bölüm sessizce boş kalır; yeni bölümler bunu çoğaltır.
    kimlikler = set(re.findall(r'\bid="([^"]+)"', HTML))
    cagrilan = set(re.findall(r"\b(?:byId|doldur|yaz|istatistikler)\('([A-Za-z0-9_-]+)'", SCRIPT))
    assert len(cagrilan) >= 150, len(cagrilan)   # desen kör kalmasın (kendi kontrolü)
    assert not cagrilan - kimlikler, sorted(cagrilan - kimlikler)


def test_fihrist_bolumler_ve_kisayollar_ayni_duzende():
    fihrist = re.findall(r'<li><a href="#([a-z]+)"><b>([A-Z])</b>', HTML)
    bolumler = re.findall(r'<section id="([a-z]+)" class="bolum"', HTML)
    harfler = re.findall(r'<span class="bolum-harf" aria-hidden="true">([A-Z])</span>', HTML)
    assert [f[0] for f in fihrist] == bolumler
    beklenen = [chr(ord("A") + i) for i in range(len(bolumler))]
    assert [f[1] for f in fihrist] == harfler == beklenen
    # "g + harf" kısayolu tam olarak bölüm harflerini kapsar.
    assert f"/^[a-{beklenen[-1].lower()}]$/i" in SCRIPT


def test_her_bolum_bir_gorevle_yuklenir():
    # Operatör bölümü durum göreviyle beslenir; geri kalan her bölümün kendi zamanlayıcı görevi olmalı.
    gorevli = set(re.findall(r"gorev\('[a-z-]+',\s*'([a-z]+)'", SCRIPT))
    bolumler = set(re.findall(r'<section id="([a-z]+)" class="bolum"', HTML)) - {"operator"}
    assert bolumler <= gorevli, sorted(bolumler - gorevli)


def test_api_metni_html_olarak_yorumlanmaz():
    # Tüm dinamik metin textContent / metin düğümüyle girer (h() yardımcısı); HTML ayrıştıran
    # bir API kullanılırsa LLM gerekçesi ya da log satırı sayfaya betik sokabilirdi.
    for yasak in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert yasak not in SCRIPT, yasak


def test_tehlikeli_eylemler_yazili_onay_ister():
    # Toplu kapatma ve kilit sıfırlama "yazarak onay" ister; tek pozisyon kapatma diyalog ister.
    assert re.search(r"kelime:\s*'KAPAT'", SCRIPT)
    assert re.search(r"kelime:\s*'SIFIRLA'", SCRIPT)
    assert "pozisyonKapat" in SCRIPT and "onayIste(" in SCRIPT
