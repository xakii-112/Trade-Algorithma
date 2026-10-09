"""Klasik pano rötuşu (2026-09-25) — üç davranış geri gelmesin.

  * Geniş ekranda sekme çubuğu SARAR: tek satır 1280 px'te 7 sekmeyi kaydırmanın ardına saklıyordu.
  * Bakiye grafiği veri yokken sahte "$1.000.000" noktası BASMAZ (ölçülmemiş bakiye ölçülmüş gibi görünürdü).
  * İstihbarat, apiCall'un hata biçimini ({success:false}) tanır: eskiden "undefined bar" diye çiziyordu.
"""
import re
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "nihai" / "web" / "web_dashboard.html").read_text(encoding="utf-8")


def test_genis_ekranda_sekmeler_sarar():
    m = re.search(r"@media \(min-width: 901px\)\s*\{\s*\.tabs\s*\{([^}]*)\}", HTML)
    assert m and "flex-wrap: wrap" in m.group(1) and "overflow-x: visible" in m.group(1)


def test_bakiye_grafigi_veri_uydurmaz():
    govde = HTML.split("function renderEquityChart", 1)[1].split("async function loadEquityChart", 1)[0]
    assert "1000000" not in govde
    assert "data.length === 0" in govde and "clearRect" in govde


def test_istihbarat_apicall_hatasini_tanir():
    govde = HTML.split("async function loadIntel()", 1)[1].split("\n        }\n", 1)[0]
    assert "d.success === false" in govde
    assert "istek !== loadIntel._no" in govde   # geç gelen eski yanıt yenisini ezmez
