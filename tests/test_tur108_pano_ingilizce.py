"""Tur-108 — panonun İngilizce arayüzü: çalışma anı sözlük katmanı + TR | EN düğmesi.

Kaynak HTML/JS Türkçe KALIR (onlarca test kaynak metnine bakar); İngilizce kipte `pano_i18n.js`
ekrana giden metni `pano_en.json` sözlüğünden çevirir. Bu dosya üç soruyu sorar:

1. BAĞLANDI MI — `/i18n.js` ucu sözlük + motoru veriyor, pano onu <head>'de yüklüyor.
2. SÖZLÜK TAM MI — HTML'deki her Türkçe metin düğümü/özniteliği sözlükte; yer tutucular tutarlı.
3. MOTOR DOĞRU MU (node) — tam eşleşme, kalıp, bileşik metin; Türkçe kipte HİÇBİR şeye dokunmuyor;
   sözlükte olmayan İngilizce metni bozmuyor.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tools import pano_i18n

KOK = Path(__file__).resolve().parents[1]
WEB = KOK / "nihai" / "web"
YT = re.compile(r"\{\d+\}")


@pytest.fixture(scope="module")
def sozluk() -> dict[str, str]:
    return pano_i18n.sozluk()


# ---------------------------------------------------------------- 1. bağlandı mı
def test_uc_sozluk_ve_motoru_tek_yanitta_verir():
    from nihai.web import web_dashboard as web
    r = TestClient(web.app).get("/i18n.js")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]
    assert r.headers["cache-control"] == "no-store"          # sözlük düzenlenince eski kopya takılmasın
    govde = r.text
    assert govde.startswith("window.__PANO_EN={")             # sözlük motorun ÖNÜNDE (motor onu okur)
    assert govde.index("window.__PANO_EN=") < govde.index("window.panoCevir")
    assert '"Genel Bakış": "Overview"' in govde


def test_pano_motoru_head_icinde_eszamanli_yukler():
    kaynak = (WEB / "web_dashboard.html").read_text(encoding="utf-8")
    bas = kaynak[: kaynak.index("</head>")]
    assert '<script src="/i18n.js"></script>' in bas          # defer/async DEĞİL: ilk boyamadan önce
    assert 'lang="tr"' in kaynak[:200]                        # kaynak dili Türkçe kalır


# ---------------------------------------------------------------- 2. sözlük tam mı
def test_sozluk_bicimi(sozluk):
    assert len(sozluk) > 2500
    for tr, en in sozluk.items():
        assert tr == pano_i18n.norm(tr) and en == pano_i18n.norm(en) and tr != en, tr
        # Motor grup sırasını numaradan okur: TR'de 0..k ardışık, EN yalnız TR'dekileri kullanır.
        sira = list(dict.fromkeys(YT.findall(tr)))
        assert sira == [f"{{{i}}}" for i in range(len(sira))], tr
        assert set(YT.findall(en)) <= set(sira), tr


def test_htmldeki_her_turkce_metin_sozlukte():
    """Yeni bir pano metni ekleyen tur çevirisini de ekler: `python -m tools.pano_i18n` eksikleri basar."""
    eksik = {k: v["yer"][0] for k, v in pano_i18n.eksikler().items() if "html" in v["tur"]}
    assert not eksik, f"pano_en.json'da karşılığı yok: {eksik}"


def test_js_sabitlerinde_kapsama(sozluk):
    aday = {k for k, v in pano_i18n.topla().items() if pano_i18n.TR_OZEL.search(k)}
    oran = len(aday & sozluk.keys()) / len(aday)
    assert oran >= 0.99, f"JS/HTML Türkçe metin kapsaması {oran:.3f} — `python -m tools.pano_i18n`"


def test_cikarici_sablon_ve_birlestirmeyi_kaliba_cevirir(tmp_path):
    (tmp_path / "web_dashboard.html").write_text(
        '<p title="Açıklama">Kasa <b>değeri</b></p><script>x.innerText = `${n} sn önce`;'
        "y = 'Toplam: ' + n + ' işlem'; z = \"<td>Açık</td><td class='c'>\" + v; // yorum 'şaka'\n"
        "var re = /'ığ'/;</script>", encoding="utf-8")
    (tmp_path / "lab_dashboard.html").write_text("", encoding="utf-8")
    (tmp_path / "lab_dashboard.js").write_text("", encoding="utf-8")
    b = set(pano_i18n.topla(tmp_path))
    assert {"Açıklama", "Kasa", "değeri", "{0} sn önce", "{0} işlem", "Açık"} <= b
    assert "Toplam: {0}" in b
    assert not any("şaka" in k or "ığ" in k for k in b)       # yorum ve regex metin değildir


# ---------------------------------------------------------------- 3. motor doğru mu
_SURUCU = r"""
const fs = require('fs');
const [sozlukYol, motorYol, dil] = process.argv.slice(1);   // `node -e`: betik yolu yok, argümanlar 1'den başlar
const depo = { 'pano.dil': dil };
const w = { __PANO_EN: JSON.parse(fs.readFileSync(sozlukYol, 'utf8')), alert() {}, confirm() {}, prompt() {} };
const html = { lang: 'tr', nodeType: 1, nodeName: 'HTML', getAttribute: () => null };
const document = { addEventListener() {}, documentElement: html, createTreeWalker: () => ({ nextNode: () => null }) };
const yazilan = [];
const C2D = function () {};
C2D.prototype.fillText = function (t) { yazilan.push(t); };
C2D.prototype.strokeText = C2D.prototype.measureText = function () {};
w.CanvasRenderingContext2D = C2D;
const g = { window: w, document, location: { search: '', href: 'http://x/' }, URLSearchParams, URL, NodeFilter: {},
  localStorage: { getItem: (k) => depo[k] || null, setItem: (k, v) => { depo[k] = v; } },
  MutationObserver: class { observe() {} }, CanvasRenderingContext2D: C2D };
new Function(...Object.keys(g), fs.readFileSync(motorYol, 'utf8'))(...Object.values(g));
new C2D().fillText('Kasa');
const girdi = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify({ dil: w.panoDil, lang: html.lang, canvas: yazilan[0], sayi: (1234.5).toLocaleString('tr-TR'),
  cevir: w.panoCevir ? girdi.map((s) => w.panoCevir(s)) : null }));
"""

_ORNEK = {
    "Genel Bakış": "Overview",                                           # tam eşleşme
    "  Açık Pozisyonlar\n": "  Open Positions\n",                        # uç boşluk korunur
    "7 sn önce": "7 s ago",                                              # kalıp
    "2026-06-26 → 2026-10-09 · 1000 kayıt": "2026-06-26 → 2026-10-09 · 1000 records",   # boş uç yer tutucu
    "Durum: AKTİF": "Status: ACTIVE",                                    # bileşik: etiket + değer
    "2026-10-09 16:31 · çekimser 30": "2026-10-09 16:31 · abstain 30",   # kenar ayraçlı anahtar
    "Risk reddi 8, Quant: aynı gün 3": "Risk reject 8, Quant: same day 3",   # liste + sondaki sayı
    "Hub (nihai.data.hub_server) işlemi: başlat. Onaylıyor musunuz?":
        "Hub (nihai.data.hub_server) action: start. Do you confirm?",    # iç içe kalıp (confirm metni)
    "AAPL": "AAPL", "1,234.56": "1,234.56",                              # veri: dokunulmaz
    "Father and Son: Not So Fast On Monday": "Father and Son: Not So Fast On Monday",   # İngilizce metin
}


def _motor(dil: str) -> dict:
    if shutil.which("node") is None:
        pytest.skip("node yok")
    r = subprocess.run(["node", "-e", _SURUCU, str(pano_i18n.SOZLUK), str(WEB / "pano_i18n.js"), dil],
                       input=json.dumps(list(_ORNEK), ensure_ascii=False), capture_output=True,
                       encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_motor_ingilizce_kipte_cevirir():
    c = _motor("en")
    assert c["dil"] == "en" and c["lang"] == "en"
    assert dict(zip(_ORNEK, c["cevir"], strict=True)) == _ORNEK
    assert c["canvas"] == "Equity"                 # canvas yazısı (Chart.js etiketleri) da çevrilir
    assert c["sayi"] == "1,234.5"                  # 'tr-TR' biçimi İngilizce kipte en-GB


def test_motor_turkce_kipte_hicbir_seye_dokunmaz():
    c = _motor("tr")
    assert c == {"dil": "tr", "lang": "tr", "canvas": "Kasa", "sayi": "1.234,5", "cevir": None}
