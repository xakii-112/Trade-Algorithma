"""Tur-69 — "kendini tanıt" User-Agent'ının TEK KAYNAĞI (`nihai.net.contact_ua`).

Neden bu dosya var: dört çağıran (`data/insider.py`, `execution/screen.py`, edge_lab'in
iki veri kurucusu) aynı UA dizesini KOPYALAMIŞTI ve dizede geliştiricinin kişisel
e-postası vardı. Merkezileştirme sırasında üç dosyada `net` import'u UNUTULDU; suite
bunun yalnız BİRİNİ yakaladı. Buradaki kapılar üçünü de yakalar.
"""
from __future__ import annotations

import ast
import pathlib
import re
import subprocess

import pytest
from loguru import logger

from nihai import net

KOK = pathlib.Path(__file__).resolve().parents[1]
CAGIRANLAR = ["nihai/data/insider.py", "nihai/execution/screen.py",
              "research/edge_lab/build_insider_dataset.py",
              "research/edge_lab/build_shortflow_dataset.py"]


@pytest.fixture(autouse=True)
def _uyari_sifirla():
    net._ua_uyarildi = False
    yield
    net._ua_uyarildi = False


def test_env_doluysa_iletisim_bilgisi_gomulur(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Ad Soyad ornek@ornek.com")
    assert net.contact_ua("Urun/1.0") == "Urun/1.0 (Ad Soyad ornek@ornek.com)"


@pytest.mark.parametrize("deger", ["", "   "])
def test_env_bossa_istek_sessizce_bozulmaz_ama_uyarir(monkeypatch, deger):
    """Boş env = SEC 403 riski. İstek YİNE yapılır (dize döner) ama BİR KEZ uyarılır.

    `caplog` KULLANILMAZ: loguru stdlib `logging`'e propagate ETMEZ (ölçüldü — uyarı
    gerçekten basılırken `caplog.records` 0 kayıt taşıyor). caplog'la yazılmış bir sürüm
    yalnız `_ua_uyarildi` bayrağını sınar, yani `logger.warning(...)` satırı SİLİNSE bile
    yeşil kalırdı: uyarıyı denetlediğini sanan test, uyarının YOKLUĞUNU da onaylar.
    Sözleşme ("sessizce bozulmaz") uyarının KENDİSİDİR, o yüzden loguru sink'iyle sayılır.
    """
    monkeypatch.setenv("SEC_USER_AGENT", deger)
    uyarilar: list[str] = []
    sink = logger.add(uyarilar.append, level="WARNING", format="{message}")
    try:
        ilk = net.contact_ua("Urun/1.0")
        ikinci = net.contact_ua("Baska/2.0")
    finally:
        logger.remove(sink)
    assert "ornek.com" not in ilk and ilk.startswith("Urun/1.0")
    assert ikinci.startswith("Baska/2.0")
    assert len(uyarilar) == 1, f"süreçte TAM BİR uyarı bekleniyordu, {len(uyarilar)} geldi"
    assert "SEC_USER_AGENT" in uyarilar[0]
    assert net._ua_uyarildi is True, "uyarı bayrağı kalkmadı"


@pytest.mark.parametrize("deger", ["NihaiBot", "NihaiBot@", "@", "Ad Soyad @nihaibot"])
def test_adressiz_deger_de_uyarir_ama_KULLANILIR(monkeypatch, deger):
    """"Dolu" yetmez: SEC/Wikimedia UA'da ulasilabilir bir ADRES arar.

    `SEC_USER_AGENT=NihaiBot` gibi bir deger eski surumde sessizce kabul ediliyordu; 403
    riski bos degerinkiyle ayniydi ama tek uyari bile cikmiyordu — operatorun yapmaya en
    yatkin oldugu yanlis yapilandirma tam olarak budur. Deger yine de KULLANILIR (operator
    bilerek set etmistir), yalniz sessiz kalmaz.

    `"@" in contact` testi YETMEZ, o yuzden burada `@` TASIYAN ama adres OLMAYAN degerler
    de var: `"@"`, `"NihaiBot@"` ve `"Ad Soyad @nihaibot"` o testten sessizce geciyordu
    (olculdu) — yani kapinin yakalamak icin yazildigi hatanin en sik bicimi acikta kaldi.
    """
    monkeypatch.setenv("SEC_USER_AGENT", deger)
    uyarilar: list[str] = []
    sink = logger.add(uyarilar.append, level="WARNING", format="{message}")
    try:
        ua = net.contact_ua("Urun/1.0")
    finally:
        logger.remove(sink)
    assert ua == f"Urun/1.0 ({deger})", "adressiz deger yok sayilmamali, kullanilmali"
    assert len(uyarilar) == 1, f"adressiz deger icin 1 uyari bekleniyordu, {len(uyarilar)} geldi"
    assert deger in uyarilar[0], "uyari sorunlu degeri gostermeli"


def test_iletisim_URLsi_ADRES_sayilir_ve_uyarmaz(monkeypatch):
    """Karsi kutup: Wikimedia UA politikasi e-posta VEYA iletisim URL'si kabul eder.

    Aksi halde gecerli bir UA (`"NihaiAlgo/1.0 (+https://github.com/x/y)"`) her surecte
    yanlis alarm uretir ve gercek uyariyi degersizlestirir — uyari-yorgunlugu sessizligin
    kardesi. Kapi bu yonde de sinanmali, yoksa "adres var mi" testi keyfi siki olabilir."""
    monkeypatch.setenv("SEC_USER_AGENT", "NihaiAlgo/1.0 (+https://github.com/x/y)")
    uyarilar: list[str] = []
    sink = logger.add(uyarilar.append, level="WARNING", format="{message}")
    try:
        ua = net.contact_ua("Urun/1.0")
    finally:
        logger.remove(sink)
    assert ua == "Urun/1.0 (NihaiAlgo/1.0 (+https://github.com/x/y))"
    assert uyarilar == [], f"iletisim URL'si adrestir, uyari beklenmiyordu: {uyarilar}"


def _contact_ua_cagiranlari() -> list[tuple[str, ast.Module]]:
    """`net.contact_ua` çağıran üretim/araştırma dosyaları — liste ELLE TUTULMAZ.

    Sabit bir dosya listesi bu dosyanın var oluş nedenini tekrar ederdi ("tek dosyayı
    düzeltmek diğerlerini açıkta bırakır"): BEŞİNCİ çağıran eklendiğinde kapı onu hiç
    görmez ve aynı regresyon sessizce geri gelir. Çağıran kümesi KODDAN türetilir.

    Kapsam `nihai/` + `research/` DEĞİL: beşinci çağıran `tools/`a ya da kökteki bir
    script'e de düşebilir (`tools/sync_mainhub.py`, `ib_smoke.py`, `conftest.py` hepsi
    depoda) — aşağıdaki sızıntı kapısı tam bu gerekçeyle `git ls-files`e genişletilmişti,
    bu kapı ise dar kalmıştı. Depodaki her üretim/araç `*.py` taranır."""
    out: list[tuple[str, ast.Module]] = []
    adaylar = [*(KOK / "nihai").rglob("*.py"), *(KOK / "research").rglob("*.py"),
               *(KOK / "tools").rglob("*.py"), *KOK.glob("*.py")]
    for p in sorted(set(adaylar)):
        if "__pycache__" in p.parts:
            continue
        try:
            metin = p.read_text(encoding="utf-8")
            if "contact_ua" not in metin:
                continue
            tree = ast.parse(metin)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue  # koşamayan dosya çağıran da olamaz; kapı ALAKASIZ bir hatayla
            # düşmemeli — düşerse gerçek bulguyu maskeler (kardeş kapıyla aynı desen)
        if any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
               and n.value.id == "net" and n.attr == "contact_ua"
               for n in ast.walk(tree)):
            out.append((p.relative_to(KOK).as_posix(), tree))
    return out


def test_contact_ua_cagiranlarinin_HEPSI_net_i_IMPORT_ediyor():
    """Asıl regresyon: `net.contact_ua(...)` yazıp `from nihai import net` yazmamak.

    ruff bunu yakalamaz (F821 yalnız --preview'da), suite ise dördün yalnız birini
    yakalamıştı. Bilinen dört çağıran ayrıca TABANDIR: biri contact_ua'yı çağırmayı
    bırakırsa kapı körleşmiş demektir (Tur-62: kapı kendi kendini tatmin edemez)."""
    cagiranlar = _contact_ua_cagiranlari()
    bulunan = {rel for rel, _ in cagiranlar}
    assert set(CAGIRANLAR) <= bulunan, (
        "contact_ua ARTIK ÇAĞRILMIYOR — kapı körleşmiş: "
        + ", ".join(sorted(set(CAGIRANLAR) - bulunan)))
    eksik = []
    for rel, tree in cagiranlar:
        adlar: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                adlar |= {a.asname or a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                adlar |= {a.asname or a.name for a in n.names}
        if "net" not in adlar:
            eksik.append(rel)
    assert not eksik, "net import'u eksik:\n  " + "\n  ".join(eksik)


EPOSTA = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Belgelerdeki yer tutucular — gercek kimlik degil, kasten yazili.
YER_TUTUCU = {"eposta@ornek.com", "ornek@ornek.com"}


def test_yayinlanan_dosyalarda_kisisel_eposta_yok():
    """Depo herkese acildiginda kisisel adres YAYINLANMASIN.

    Kapsam GIT'IN TAKIP ETTIGI dosyalardir, `nihai/` + `research/` degil: yayinlanan sey
    tam olarak budur ve adres `tools/`, `docs/`, bir `*.ps1` ya da conftest'e de dusebilir
    (eski surum bunlarin hicbirine bakmiyordu). Desen de tek bir sabit adres degil jenerik
    e-postadir; baska bir gelistiricinin adresi de yakalanir. `.env` gitignore'lu oldugu
    icin dogal olarak kapsam disidir — kimligin YASAMASI gereken tek yer orasi.

    SINIR — BU KAPI GECMISI TEMIZLEMEZ: adres `git log -p`de hala duruyor (Tur-69 oncesi
    dort dosyada gomuluydu). Depoyu bugun public yapmak o gecmisi de yayinlar; bunu
    kapatmak gecmis yeniden yazmayi (filter-repo) gerektirir ve BILEREK yapilmadi.
    """
    try:
        # encoding ŞART: `text=True` tek başına stdout'u YEREL kod sayfasıyla çözer
        # (bu makinede cp1254). git yolları UTF-8 bayt olarak basar → ASCII dışı adı olan
        # bir dosya mojibake'e döner, `read_text` FileNotFoundError verir ve aşağıdaki
        # `continue` onu SESSİZCE atlar: kapı adlandıramadığı dosyayı hiç denetlemez.
        cikti = subprocess.run(["git", "ls-files", "-z"], cwd=KOK, capture_output=True,
                               text=True, encoding="utf-8", timeout=60)
    except (OSError, subprocess.SubprocessError) as e:  # pragma: no cover - git yoksa
        pytest.skip(f"git calistirilamadi: {e}")
    if cikti.returncode != 0:  # pragma: no cover - depo degilse
        pytest.skip("git deposu degil")
    suclular = []
    for rel in cikti.stdout.split("\0"):
        if not rel:
            continue
        try:
            metin = (KOK / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # ikili dosya / silinmis girdi
        for i, satir in enumerate(metin.splitlines(), 1):
            for adres in EPOSTA.findall(satir):
                if adres.lower() not in YER_TUTUCU:
                    suclular.append(f"{rel}:{i} -> {adres}")
    assert not suclular, (
        "yayinlanan dosyalarda kisisel e-posta:\n  " + ("\n  ".join(suclular))
        + "\n  — `.env`deki SEC_USER_AGENT + net.contact_ua kullanin")
