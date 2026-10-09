"""test_api_extra.py — Tur-18: panoya açılan yeni uçların sözleşmesi.

Kritik davranış: uçlar FAIL-SOFT olmalı. Alt sistem yoksa (FRED anahtarı boş, torch
kurulu değil, log henüz birikmemiş) uç 500 ATMAMALI — 200 + `{"ok": ...}` dönmeli ki
pano tek bir eksik parça yüzünden komple kararmasın.

Ağ gerektiren uçlar (intel/symbol, options/chain, indicators/extended, portfolio-propose)
burada BİLİNÇLİ olarak dışarıda: sağlayıcıya bağımlı olmaları testi kırılgan yapar.
Onların sözleşmesi `_bars`/`_provider` üzerinden zaten aynı fail-soft desenini kullanır.
"""

import pytest
from fastapi.testclient import TestClient

from nihai.web.web_dashboard import app

client = TestClient(app)

# Ağ/sağlayıcı gerektirmeyen, yalnız yerel dosya + saf hesap okuyan uçlar.
READONLY_ENDPOINTS = [
    "/api/ai/overview",
    "/api/ai/learning",
    "/api/ai/decisions?days=7",
    "/api/ai/episodes?days=7",
    "/api/ai/knowledge",
    "/api/ai/registry",
    "/api/ai/capacity",
    "/api/ai/selfcheck",
    "/api/ai/proposals",
    "/api/system/health",
    "/api/system/daily-summary",
    "/api/system/evidence-notes",
]


@pytest.mark.parametrize("url", READONLY_ENDPOINTS)
def test_readonly_endpoint_never_500(url):
    """Hiçbiri istisna sızdırmamalı; hepsi JSON sözlük + `ok` alanı döndürmeli."""
    r = client.get(url)
    assert r.status_code == 200, f"{url} → {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert isinstance(body, dict)
    assert "ok" in body


def test_ai_overview_alanlari():
    """Pano kartlarının okuduğu alanlar sözleşmenin parçası — ad değişirse test kırılsın."""
    d = client.get("/api/ai/overview").json()
    for key in ("decision_mode", "meta_gate_mode", "predict_track", "llm",
                "selfcheck", "agenda", "memory", "shadow_runner", "neural_shadow",
                "capacity_ladder"):
        assert key in d, f"eksik alan: {key}"


def test_neural_shadow_durumu_gorunur():
    """Tur-18: gölge köprüsü pasifleşirse NEDENİ panoda görünmeli — sessiz ölüm yasak.

    Testte DATA_DIR geçici dizindir → kayıtlı model kartı yoktur → uç 'pasif + neden'
    döndürmeli, ASLA 500 atmamalı (fail-soft sözleşmesi)."""
    d = client.get("/api/ai/overview").json()
    ns = d["neural_shadow"]
    assert ns.get("ok") is not False or "error" in ns   # _safe sarmalı bozulmamış
    if ns.get("active") is False:
        assert ns.get("reason"), "pasiflik nedeni boş bırakılamaz"


def test_capacity_ladder_parametre_sayilari_artan():
    """Kapasite merdiveni ADIM ADIM büyümeli (az veriyle yüksek kapasite uyarısının temeli)."""
    d = client.get("/api/ai/capacity?n_features=25&include_neural=false").json()
    ladder = d["ladder"]
    assert isinstance(ladder, list) and ladder
    counts = [row["param_count"] for row in ladder]
    assert counts == sorted(counts), f"merdiven artan değil: {counts}"


def test_knowledge_arama_ve_kavram():
    """Sözlük her zaman dolu; arama sonuçları sözlükteki id'lere işaret etmeli."""
    d = client.get("/api/ai/knowledge?q=sharpe").json()
    assert d["glossary"], "bilgi çekirdeği boş olmamalı"
    results = d.get("results", [])
    assert isinstance(results, list)
    for row in results:
        assert row["id"] in d["glossary"]


def test_evidence_notes_dersleri_okur():
    """CLAUDE.md ACI DERSLER bloğu panoya taşınabiliyor mu (kurumsal hafıza görünürlüğü)."""
    d = client.get("/api/system/evidence-notes").json()
    assert isinstance(d.get("lessons"), list)
    assert d["lessons"], "ACI DERSLER okunamadı — CLAUDE.md formatı değişmiş olabilir"


def test_position_size_gecersiz_girdi_400_yerine_dogru_reddeder():
    """Fat-finger koruması: entry==stop sonsuz boyut üretmemeli, dürüstçe reddetmeli."""
    r = client.post("/api/tools/position-size",
                    json={"symbol": "TEST", "entry": 100.0, "stop": 100.0, "side": "LONG"})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is False and "reason" in d


def test_position_size_gercek_hesap(monkeypatch):
    """Onaylanan hesapta adet pozitif ve risk $ = |entry-stop| × adet ile tutarlı olmalı.

    `_bars` sıfırlanır (Tur-26): uç, rejim çarpanı için `_bars("SPY","1d",220)` çekiyordu,
    yani bu test dosyanın kendi "ağ gerektiren uçlar burada DEĞİL" sözleşmesini farkında
    olmadan ihlal ediyordu — curl_cffi deliği (BULGU-26) yüzünden guard görmüyordu, ölçülen
    maliyet 9,80 sn idi. Rejim `_safe` sarmalında; None → nötr çarpan, testin ASIL iddiası
    (risk $ = |entry−stop| × adet) etkilenmez."""
    from nihai.web import api_extra
    monkeypatch.setattr(api_extra, "_bars", lambda *a, **k: None)
    r = client.post("/api/tools/position-size",
                    json={"symbol": "TEST", "entry": 100.0, "stop": 95.0, "side": "LONG"})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert "approved" in d and "sizing" in d
    if d["approved"]:
        shares = d["sizing"]["shares"]
        assert shares > 0
        assert d["sizing"]["risk_dollars"] == pytest.approx(5.0 * shares, rel=1e-6)


def test_ai_ask_bos_soruda_cokmez():
    """Savunmacı: anlamsız/boş girdi 500 değil, kaynaklı-boş cevap döndürmeli."""
    r = client.post("/api/ai/ask", json={"question": "   "})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_yeni_uclar_uygulamaya_bagli():
    """Router gerçekten include edilmiş mi (regresyon: import sırası değişirse kaçabilir)."""
    paths = {r.path for r in app.routes}
    for p in ("/api/ai/overview", "/api/intel/symbol", "/api/options/chain",
              "/api/tools/position-size", "/api/indicators/extended", "/api/system/health"):
        assert p in paths, f"{p} uygulamaya bağlanmamış"


# ------------------------------------------------------------------------- Tur-30
def test_overview_merdiveni_noral_katmanlari_da_icerir():
    """Pano merdiveni model-kartı tablosuyla AYNI evreni göstermeli.

    `param_ladder_report()` varsayılanı include_neural=False; overview onu argümansız
    çağırdığı için pano v2'nin 4 satırında duruyordu — hemen altındaki registry tablosu
    deep_mlp/quant_transformer kartlarını listelerken merdiven onları hiç bilmiyordu."""
    ladder = client.get("/api/ai/overview").json()["capacity_ladder"]
    assert isinstance(ladder, list), f"merdiven liste değil: {ladder}"
    tiers = {row.get("tier") for row in ladder}
    assert {"deep_mlp", "quant_transformer", "quant_transformer_xl"} <= tiers, tiers
    # torch yoksa satır param_count=None + note ile gelir (fail-soft) — yine de VAR olmalı.
    for row in ladder:
        assert row.get("param_count") is not None or row.get("note")


def test_system_health_kapsamayi_onbellekli_ozetten_okur(monkeypatch):
    """Sağlık ucu ham `coverage()`i çağırmamalı (her istekte 53k satır yeniden ayrıştırma).

    Ham yolu PATLATIYORUZ: uç hâlâ gerçek kapsama döndürüyorsa imza-önbellekli
    `summary()["coverage"]` yolundan okuyor demektir (Tur-28 optimizasyonu api_extra'da
    atlanmıştı; ÖLÇÜLDÜ: ~400 ms/istek)."""
    from nihai.execution import prediction_tracker as pt

    pt.summary()          # önbelleği ısıt (summary kendi içinde coverage(rows) çağırır)

    def _patlat(*a, **k):
        raise AssertionError("ham coverage() çağrıldı — önbellekli özet kullanılmalı")

    monkeypatch.setattr(pt, "coverage", _patlat)
    body = client.get("/api/system/health").json()
    assert "total" in body["coverage"], body["coverage"]


def test_system_health_tahmin_servisi_gercek_anahtarlari_okur():
    """"Tahmin Servisi" kartının üç alanı da OLMAYAN anahtarları okuyordu.

    snapshot() sözleşmesi {"meta", "predictions", "running"}; uç ise "running"/"rows"/
    "updated_at" arıyordu → kart servis AYAKTAYKEN bile daima 'durdu · 0 sembol · —'.
    Servislerin sessizce ölmesi bu projenin bilinen arızası olduğundan, hep-yanlış bir
    sağlık kartı en zararlı hatadır: operatörü kartı görmezden gelmeye alıştırır."""
    from nihai.execution.live_predict_service import get_service

    svc = get_service()
    snap = svc.snapshot()
    assert {"meta", "predictions", "running"} <= set(snap), sorted(snap)

    ps = client.get("/api/system/health").json()["predict_service"]
    assert ps["running"] is snap["running"]
    assert ps["n_symbols"] == len(snap["predictions"])
    assert ps["updated_at"] == snap["meta"].get("updated")

    # AYAKTA hâli de doğrulanmalı: iki taraf da False iken eşitlik hiçbir şey kanıtlamaz.
    # Gerçek döngüyü başlatmadan (ağ + canlı log yazımı) sahte-canlı thread enjekte et.
    class _CanliThread:
        @staticmethod
        def is_alive() -> bool:
            return True

    svc._thread = _CanliThread()          # type: ignore[assignment]
    svc._rows = [{"symbol": "AAPL"}, {"symbol": "MSFT"}]
    svc._meta = dict(svc._meta, updated="2026-08-14T09:00:00")
    try:
        ps = client.get("/api/system/health").json()["predict_service"]
        assert ps["running"] is True, "servis AYAKTA iken kart hâlâ 'durdu' diyor"
        assert ps["n_symbols"] == 2, ps
        assert ps["updated_at"] == "2026-08-14T09:00:00", ps
    finally:
        svc._thread = None
        svc._rows = []
