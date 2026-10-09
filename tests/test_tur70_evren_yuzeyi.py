"""Tur-70 kilidi — EVREN DEGRADE'İNİN OKUNAN YÜZEYİ (BULGU-69-1'in kapanışı).

Tur-69 boş S&P 500 evrenine bir `logger.error` ekledi ve `.env.example`a "sessiz
değildir" yazdı. Denetimde çıkan kusur düzeltmenin kendisinde değil, KAPSAMINDAYDI:
CLAUDE.md'nin kendi kuralı (*Tur-61 / BULGU-61-2*) bir durum yüzeyi yazan turun onu
OKUYAN yüzeyi de AYNI turda bağlamasını ve okuyucunun varlığının testle kilitlenmesini
ister. Gerekçe ölçülmüş bir gerçektir: 7/24 koşan bir tarayıcının `system.log`'unda tek
bir ERROR satırı pratikte görünmez — yani "sessiz evren çökmesi yok" koruması KENDİSİ
sessiz kalır. `bridges` (Tur-61) ile bire bir aynı sınıf.

Bu tur eksik yarıyı tamamlar:
  * `screen.evren_durum()` — kalıcı süreç-içi durum (guard ile AYNI fonksiyonda yazılır,
    yani dört çağıranın hepsi kapsanır).
  * `/api/system/health` → `universe` — üretim okuyucusu.
  * pano "Evren Kaynakları" kartı — operatörün gördüğü yüzey.

Ayrıca Tur-69'un ERROR'la kapatmadığı İKİNCİ sessiz yol da buraya bağlanır: `nasdaq100`
bayat yedek listesine düşerken yalnız WARNING üretir, `AUTO_SCREEN_UNIVERSE=nasdaq100`
kipinde başka hiçbir işaret yoktur. `canli=False` her iki yolu da işaretler.

Kilit BİR KEZLİK DÜZELTMEYİ DEĞİL KURALI korur: uç alanı düşerse, pano olmayan bir alan
okumaya başlarsa ya da degrade işareti yazılmaz olursa test KIRMIZI olur.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import nihai.execution.screen as s
from nihai.web.web_dashboard import app

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)

# Uç fail-soft'tur (`_safe`): istisna hâlinde gövde {"ok": False, "error": ...} olur.
# Pano bu iki alanı okur, başarı gövdesinde ikisi de BULUNMAZ.
_FAILSOFT = {"ok", "error"}


@pytest.fixture
def temiz_durum(monkeypatch):
    """`_EVREN_DURUM` modül-düzeyi durumdur; testler birbirini kirletmesin."""
    monkeypatch.setattr(s, "_EVREN_DURUM", {})
    return s


def _tablo(semboller: list[str]) -> list[pd.DataFrame]:
    return [pd.DataFrame({"Symbol": semboller})]


# ------------------------------------------------- 1) DEGRADE gerçekten işaretleniyor mu
def test_bos_sp500_DEGRADE_olarak_isaretlenir(temiz_durum, monkeypatch):
    """Boş S&P 500 yalnız loglanmaz, KALICI yüzeye de yazılır (log satırı görünmez)."""
    monkeypatch.setattr(s, "_read_html_tables", lambda url: _tablo([]))
    assert s.sp500_tickers() == []
    durum = s.evren_durum()["sp500"]
    assert durum["canli"] is False, "boş S&P 500 degrade olarak İŞARETLENMEDİ"
    assert durum["n"] == 0
    assert durum["at"] is not None


def test_dolu_sp500_CANLI_isaretlenir(temiz_durum, monkeypatch):
    """Karşı kutup: normal tur degrade görünmemeli (işaret kendini tatmin etmesin)."""
    semboller = [f"S{i:03d}" for i in range(55)]   # _extract_ticker_col ≥50 satır ister
    monkeypatch.setattr(s, "_read_html_tables", lambda url: _tablo(semboller))
    assert s.sp500_tickers() == semboller
    durum = s.evren_durum()["sp500"]
    assert durum["canli"] is True
    assert durum["n"] == 55


def test_nasdaq_BAYAT_yedek_de_DEGRADE_isaretlenir(temiz_durum, monkeypatch):
    """Tur-69'un ERROR'u yalnız sp500'ü kapsıyordu; nasdaq yedeği WARNING'de kalıyordu.

    `AUTO_SCREEN_UNIVERSE=nasdaq100` kipinde ERROR dalına HİÇ girilmez — o kipte bu
    işaret operatörün sahip olduğu TEK göstergedir."""
    monkeypatch.setattr(s, "_read_html_tables", lambda url: _tablo([]))
    semboller = s.nasdaq100_tickers()
    assert semboller, "bayat yedek liste yine de dönmeli (fail-soft)"
    durum = s.evren_durum()["nasdaq100"]
    assert durum["canli"] is False, "bayat yedek listeye düşüş İŞARETLENMEDİ"
    assert durum["n"] == len(semboller)


def test_evren_kurulmadan_durum_BOSTUR(temiz_durum):
    """Boş sözlük 'henüz kurulmadı' demektir; 'çekim başarısız' DEĞİL. Pano ikisini
    ayırt edebilsin diye ayrı hâller."""
    assert s.evren_durum() == {}


# --------------------------------------------------------- 2) üretim okuyucusu var mı
def test_evren_durumunun_URETIM_okuyucusu_var():
    """`evren_durum()` en az bir ÜRETİM dosyasından çağrılmalı — testten değil.

    BULGU-61-2 tam olarak buydu: `status()` yazılmış ama yalnız testlerden çağrılıyordu.
    Bir sonraki tur bu okuyucuyu silerse kilit kırmızıya döner."""
    kaynak = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in (ROOT / "nihai").rglob("*.py"))
    assert re.search(r"\bevren_durum\(\)", kaynak.replace("def evren_durum()", "")), (
        "screen.evren_durum() uretimde HIC okunmuyor — yazilip okunmayan yuzey kusurdur")


def test_health_ucu_evren_durumuna_AST_ile_bagli():
    """`/api/system/health` gövdesi `screen.evren_durum`u ADIYLA çağırmalı.

    Metin araması bir yorumla da tatmin olurdu; bu yüzden çağrı AST'ten okunur."""
    src = (ROOT / "nihai" / "web" / "api_extra.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "system_health")
    cagrilar = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert "screen.evren_durum" in cagrilar


# ------------------------------------------------------------- 3) uç ↔ pano sözleşmesi
def test_health_ucu_evren_durumunu_YAYINLAR():
    """Uç `universe` taşır ve alan adları ALAN ALAN kilitlenir (Tur-30 dersi: pano var
    OLMAYAN anahtar okursa hata çıkmaz, yalnız daima yanlış bir kart olur)."""
    body = client.get("/api/system/health").json()
    uni = body.get("universe")
    assert isinstance(uni, dict) and uni.get("ok") is not False, f"universe okunamadi: {uni}"
    assert {"auto_screen", "sources"} <= set(uni)
    assert isinstance(uni["sources"], dict)
    for ad, kayit in uni["sources"].items():
        assert {"n", "canli", "at"} <= set(kayit), f"{ad} kaydinin alanlari eksik: {kayit}"


def test_pano_yalnizca_UCUN_verdigi_alanlari_okuyor():
    """Pano ↔ uç sözleşmesi: HTML'in `uni.`/`src.` ile okuduğu her alan uçta VAR olmalı."""
    html = (ROOT / "nihai" / "web" / "web_dashboard.html").read_text(encoding="utf-8")
    blok = html.split("const uni = d.universe || {};", 1)
    assert len(blok) == 2, "pano evren karti kayboldu"
    govde = blok[1].split("const env = d.env", 1)[0]

    uni = client.get("/api/system/health").json()["universe"]
    okunan = set(re.findall(r"\buni\.([a-z_]+)\b", govde))
    assert okunan, "pano universe alanlarini hic okumuyor"
    eksik = okunan - set(uni) - _FAILSOFT
    assert not eksik, f"pano universe icin OLMAYAN alan okuyor: {sorted(eksik)}"

    # Kaynak kaydının alanları uçtan DEĞİL sözleşmeden doğrulanır: `sources` taze bir
    # süreçte hakli olarak BOŞ olabilir, ama pano yine de bu adları okur.
    okunan_src = set(re.findall(r"\bsrc\.([a-z_]+)\b", govde))
    assert okunan_src, "pano kaynak alanlarini hic okumuyor"
    assert okunan_src <= {"n", "canli", "at"}, (
        f"pano kaynak kaydinda OLMAYAN alan okuyor: {sorted(okunan_src - {'n', 'canli', 'at'})}")


def test_pano_karti_DEGRADE_rozetini_tasiyor():
    """Kart degrade'i GÖRÜNÜR kılmalı — sessiz daralmanın tüm amacı budur.

    Rozet, kartı dolduran JS bloğunun İÇİNDE aranır: HTML'in başka bir yerindeki
    'DEGRADE' kelimesi (ör. açıklama metni) bu kilidi tatmin etmemeli."""
    html = (ROOT / "nihai" / "web" / "web_dashboard.html").read_text(encoding="utf-8")
    assert 'id="sys-universe"' in html, "pano evren kartinin kabi yok"
    govde = html.split("const uni = d.universe || {};", 1)[1].split("const env = d.env", 1)[0]
    assert "DEGRADE" in govde, "kart degrade halini isaretlemiyor"
    assert "src.canli" in govde, "rozet gercek durum alanina bagli degil"
