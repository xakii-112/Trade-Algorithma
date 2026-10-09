"""Tur-61 / BULGU-61-6 — İÇ-AI DEFTERLERİNİN YÜZEYİ.

`nihai/ai` yedi kalıcı defter yazıyordu ve **hiçbirinin üretim okuyucusu yoktu**: yazan
modül kendi dosyasını okur, ama pano/günlük özet/CLI hiçbirini göstermiyordu. Tur-38'in
kuralı ("yazılıp okunmayan kolon kusurdur") burada DOSYA ölçeğinde ihlal ediliyordu.

Bedeli ölçüldü ve teşhisin kendisi bulgunun kanıtıdır: gerçek `data/` dizininde
**yedinin BEŞİ hiç yazılmamıştı** (`ai_agenda` · `ai_lab_log` · `ai_lesson_candidates` ·
`ai_semantic_memory` · `ai_text_scores`), kalan ikisi 19-20 gün bayattı. Yani yazan yollar
üretimde hiç koşmamıştı ve bunu görmenin BİR YOLU YOKTU.

Kilit tek seferlik düzeltmeyi değil KURALI korur: `nihai/ai` altında yeni bir kalıcı
defter doğar da `AI_STORES` kaydına (ya da "başka uçta" muafiyetine) yazılmazsa test
KIRMIZI olur.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from nihai.web.api_extra import AI_STORES, AI_STORES_BASKA_UCTA
from nihai.web.web_dashboard import app

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def _yazilan_defterler() -> dict[str, str]:
    """`nihai/ai/*.py` içinde `DATA_DIR / "x"` ile kurulan KALICI dosya adları (ast ile).

    Metin araması bir yorumla da tatmin olurdu (Tur-49/50 dersi) → ayrıştırılmış ağaç."""
    bulunan: dict[str, str] = {}
    for p in sorted((ROOT / "nihai" / "ai").glob("*.py")):
        agac = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        for n in ast.walk(agac):
            if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
                    and isinstance(n.left, ast.Name) and n.left.id == "DATA_DIR"
                    and isinstance(n.right, ast.Constant) and isinstance(n.right.value, str)):
                ad = n.right.value
                if ad.endswith((".json", ".jsonl", ".csv")):
                    bulunan.setdefault(ad, p.name)
    return bulunan


def test_yazilan_her_defterin_bir_YUZEYI_var():
    """Kayıt eksiksiz mi — yeni bir defter sessizce okunmadan kalabilir mi?"""
    kayitli = {dosya for dosya, _ in AI_STORES.values()}
    acikta = {ad: sahip for ad, sahip in _yazilan_defterler().items()
              if ad not in kayitli and ad not in AI_STORES_BASKA_UCTA}
    assert not acikta, (
        "nihai/ai yeni bir defter yaziyor ama hicbir yuzey okumuyor: "
        + ", ".join(f"{a} ({s})" for a, s in sorted(acikta.items()))
        + " — AI_STORES'a ekle ya da AI_STORES_BASKA_UCTA'ya (zaten bir ucu varsa).")


def test_muafiyet_listesi_UYDURMA_ad_tasimiyor():
    """Muafiyet bir KAÇIŞ deliği olmamalı: listedeki her ad gerçekten yazılıyor olmalı.

    (Tur-60 dersinin tersi: yalnız kabul eden bir kapı da kapı değildir — muafiyet
    listesi zamanla ölü adlarla dolarsa denetim sessizce gevşer.)"""
    yazilan = set(_yazilan_defterler())
    olu = {a for a in AI_STORES_BASKA_UCTA if a not in yazilan}
    assert not olu, f"muafiyet listesinde artik yazilmayan ad var: {sorted(olu)}"


def test_uc_yedi_defteri_de_raporluyor():
    body = client.get("/api/ai/stores").json()
    assert body.get("ok") is True
    st = body.get("stores") or {}
    assert set(st) == set(AI_STORES)
    for ad, kayit in st.items():
        assert "file" in kayit and "exists" in kayit, ad
        # Var olan defterde boyut+yas ZORUNLU: "var" demek yetmez, canlilik SAYIDIR.
        if kayit["exists"] and "error" not in kayit:
            assert "size_kb" in kayit and "age_min" in kayit, ad


def test_yok_olan_defter_HATA_degil_OLCUM(tmp_path, monkeypatch):
    """Hiç yazılmamış defter uçta 500 üretmez ve `exists=False` ile SAYILIR.

    Bu turun teşhisi tam olarak buydu: yedinin beşi hiç yazılmamıştı. `n_missing`
    olmasaydı kart yalnızca boş satırlar gösterir, kusur görünmezdi."""
    from nihai.web import api_extra
    monkeypatch.setattr(api_extra, "DATA_DIR", tmp_path)
    body = client.get("/api/ai/stores").json()
    assert body["ok"] is True
    assert body["n_missing"] == len(AI_STORES)
    assert all(v["exists"] is False for v in body["stores"].values())


def test_gercek_kayit_KUYRUKTAN_okunuyor_ve_kucultuluyor(tmp_path, monkeypatch):
    """Son kayıt okunuyor, büyük alanlar kartı boğmuyor, bozuk satır çökertmiyor."""
    from nihai.web import api_extra
    monkeypatch.setattr(api_extra, "DATA_DIR", tmp_path)
    dosya = tmp_path / AI_STORES["agenda"][0]
    dosya.write_text(
        "\n".join([json.dumps({"ts": f"2026-08-{i:02d}", "soru": "x" * 300,
                               "kaynaklar": list(range(50))}) for i in range(1, 9)]
                  + ["{bozuk json"]),
        encoding="utf-8")
    kayit = client.get("/api/ai/stores?tail=2").json()["stores"]["agenda"]
    assert kayit["exists"] is True and kayit["size_kb"] > 0
    son = kayit["last"]
    assert len(son) == 2 and son[-1]["ts"] == "2026-08-08"          # bozuk satır atlandı
    assert len(son[-1]["soru"]) <= 90                                # metin kırpıldı
    assert son[-1]["kaynaklar"] == "<list n=50>"                     # koleksiyon özetlendi


def test_pano_yalnizca_UCUN_verdigi_alanlari_okuyor():
    """Pano ↔ uç sözleşmesi (Tur-29/30 dersi): olmayan anahtar hata vermez, DAİMA yanlış
    bir kart üretir — bu yüzden alan kümesi testte kilitlenir."""
    html = (ROOT / "nihai" / "web" / "web_dashboard.html").read_text(encoding="utf-8")
    blok = html.split("apiCall('ai/stores')", 1)
    assert len(blok) == 2, "pano defter karti kayboldu"
    govde = blok[1].split("apiCall('ai/selfcheck')", 1)[0]
    okunan = set(re.findall(r"\bs\.([a-z_]+)\b", govde))
    izinli = {"file", "desc", "exists", "size_kb", "age_min", "last", "error"}
    assert okunan and okunan <= izinli, f"pano OLMAYAN alan okuyor: {sorted(okunan - izinli)}"


def test_kuyruk_okuyucu_TEK_kaynaktan(tmp_path):
    """BULGU-61-5: `_tail_jsonl` tüm dosyayı okuyordu; artık `atomic_io.tail_lines`.

    Tur-35 bu deseni üç modülde birden tek kaynağa indirmişti — dördüncü kopya burada
    yeniden doğmuştu. Kilit hem SONUCU hem KAYNAĞI denetler."""
    # Denetim AST iledir: bu testin KENDİ docstring'i "read_text" kelimesini içeriyor ve
    # ilk sürümdeki metin araması ona takıldı — Tur-49'un "kaynak metnini tarayan kapı
    # KODA bakmalı, ANLATIYA değil" dersinin bu turdaki (ikinci) tekrarı.
    src = (ROOT / "nihai" / "web" / "api_extra.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_tail_jsonl")
    cagrilar = {ast.unparse(n.func).rsplit(".", 1)[-1]
                for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert "tail_lines" in cagrilar
    assert "read_text" not in cagrilar, "kuyruk okuyucu yine TÜM dosyayı okuyor"

    from nihai.web.api_extra import _tail_jsonl
    f = tmp_path / "x.jsonl"
    f.write_text("\n".join(json.dumps({"i": i}) for i in range(500)), encoding="utf-8")
    assert [r["i"] for r in _tail_jsonl(f, 3)] == [497, 498, 499]
    assert _tail_jsonl(tmp_path / "yok.jsonl", 3) == []
