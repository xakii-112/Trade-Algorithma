"""Tur-77 · Madde 6.4 — pano JS modülerleştirmesi + Karar Hunisi / K-NN anı / U-eğrisi bağlantıları.

BULGU-76-6: pano JS'i TEK `<script>` bloğuydu; tek sözdizimi hatası BÜTÜN panoyu öldürüyordu.
Bu dosya dört şeyi kilitler:
  1. Her satır-içi blok TEK BAŞINA ayrışır (bir bloğun hatası diğerini bozamaz).
  2. Modül kaydı tutarlı: başlatma bloğunun beklediği liste == modüllerin kendini yazdığı adlar.
  3. SABOTAJ benzetimi (node `vm`, tarayıcıdaki gibi ortak global kapsam + sahte DOM): bir modüle
     sözdizimi hatası sokulunca YALNIZ o modül düşer, diğerleri tanımlı kalır ve şerit ADIYLA basılır.
  4. Yeni kartlar bağlı: sekme düğmeleri yükleyiciyi çağırır, huni çizicisi gerçek biçimdeki JSON'u çizer.
`node` yoksa JS testleri ATLANIR (ölçüm yokluğu ölçüm gibi raporlanmaz — Tur-76 kapısıyla aynı).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[1]
HTML = (KOK / "nihai" / "web" / "web_dashboard.html").read_text(encoding="utf-8")
BLOKLAR = re.findall(r"<script(?![^>]*src=)[^>]*>(.*?)</script>", HTML, re.S)
NODE = shutil.which("node")
JS = pytest.mark.skipif(NODE is None, reason="node kurulu değil — JS ölçülemedi")


def test_modul_kaydi_tutarli():
    beklenen = json.loads(re.search(r"var beklenen = (\[.*?\]);", HTML).group(1))
    yazilan = re.findall(r'window\.__PANO_MODULLERI\.push\("([^"]+)"\);', HTML)
    assert yazilan == beklenen and len(beklenen) >= 9
    modul_bloklari = [b for b in BLOKLAR if "[pano modülü " in b]
    assert len(modul_bloklari) == len(beklenen)
    for blok, ad in zip(modul_bloklari, beklenen, strict=True):
        assert f"[pano modülü {ad}]" in blok
        assert blok.rstrip().endswith(f'window.__PANO_MODULLERI.push("{ad}");'), ad   # kayıt SON deyim
    assert "window.__PANO_MODULLERI = []" in BLOKLAR[0]                                 # kayıt ilk blok
    baslatma = BLOKLAR[-1]
    assert "panoModulDenetimi" in baslatma and baslatma.count("try {") >= 5
    assert "loadAllData();" in baslatma


@JS
def test_her_blok_tek_basina_ayrisir(tmp_path):
    for i, blok in enumerate(BLOKLAR):
        hedef = tmp_path / f"blok{i}.js"
        hedef.write_text(blok, encoding="utf-8")
        r = subprocess.run([NODE, "--check", str(hedef)], capture_output=True, text=True)
        assert r.returncode == 0, f"blok{i} ayrışmıyor:\n{r.stderr[:600]}"


_HARNESS = r"""
const fs = require("fs"); const vm = require("vm");
const [html, sabote, ornek] = [fs.readFileSync(process.argv[2], "utf8"), process.argv[3], process.argv[4]];
const re = /<script(?![^>]*src=)[^>]*>([\s\S]*?)<\/script>/g; const bloklar = []; let m;
while ((m = re.exec(html))) bloklar.push(m[1]);
const kayit = { eklenen: [], hatalar: [], dinleyici: {} };
const p = new Proxy(function () {}, { get(t, k) {
  if (k === Symbol.toPrimitive) return () => 0; if (k === Symbol.iterator) return function* () {};
  if (k === "then") return undefined; if (k === "length") return 0;
  if (k === "appendChild") return el => { kayit.eklenen.push(el); return el; };
  if (k === "createElement") return () => ({ style: {}, setAttribute() {}, textContent: "" });
  return p; }, set() { return true; }, apply() { return p; }, construct() { return p; } });
const ctx = { console: { log() {}, warn() {}, error: (...a) => kayit.hatalar.push("console " + a.join(" ").slice(0, 120)) },
  document: p, location: { search: "", hash: "", href: "http://x/" }, navigator: p, localStorage: p, sessionStorage: p,
  setInterval() { return 1; }, setTimeout() { return 1; }, clearInterval() {}, clearTimeout() {},
  fetch: () => new Promise(() => {}), requestAnimationFrame() {}, Chart: p, LightweightCharts: p,
  URLSearchParams, Intl, Date, Math, JSON, Promise, Array, Object, Number, String, Set, Map,
  getComputedStyle: () => p, matchMedia: () => p, performance: { now: () => 0 } };
ctx.window = ctx; ctx.addEventListener = (ad, fn) => { (kayit.dinleyici[ad] = kayit.dinleyici[ad] || []).push(fn); };
vm.createContext(ctx);
bloklar.forEach((kod, i) => {
  if (sabote && kod.includes("[pano modülü " + sabote + "]")) kod = kod.replace("// ===== [pano", "const = ;\n// ===== [pano");
  let s; try { s = new vm.Script(kod); } catch (e) {
    kayit.hatalar.push("SOZDIZIMI " + i); (kayit.dinleyici.error || []).forEach(fn => fn({ message: e.message, lineno: 1 })); return; }
  try { s.runInContext(ctx, { timeout: 5000 }); } catch (e) { kayit.hatalar.push("CALISMA " + i + " " + (e && e.message)); }
});
let huni = null, huniYok = null;
if (ornek && typeof ctx.renderDecisionFunnel === "function") {
  const c1 = { innerHTML: "" }; ctx.renderDecisionFunnel(c1, JSON.parse(fs.readFileSync(ornek, "utf8"))); huni = c1.innerHTML;
  const c2 = { innerHTML: "" }; ctx.renderDecisionFunnel(c2, { ok: false, reason: "LOG_YOK" }); huniYok = c2.innerHTML;
}
const fonk = ["loadAllData", "switchTab", "loadPredictions", "loadDecisionFunnel", "loadSimilarEpisodes", "qmRenderMicro"];
console.log(JSON.stringify({ hatalar: kayit.hatalar, moduller: ctx.__PANO_MODULLERI,
  tipler: Object.fromEntries(fonk.map(f => [f, typeof ctx[f]])), serit: kayit.eklenen.map(e => e.textContent), huni, huniYok }));
"""


def _kos(tmp_path, sabote: str = "", ornek: Path | None = None) -> dict:
    harness = tmp_path / "harness.cjs"
    harness.write_text(_HARNESS, encoding="utf-8")
    html = tmp_path / "pano.html"
    html.write_text(HTML, encoding="utf-8")
    args = [NODE, str(harness), str(html), sabote, str(ornek) if ornek else ""]
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[:800]
    return json.loads(r.stdout.strip().splitlines()[-1])


@JS
def test_temiz_yukleme_hatasiz_ve_tum_moduller_kayitli(tmp_path):
    r = _kos(tmp_path)
    assert r["hatalar"] == []
    assert r["moduller"] == json.loads(re.search(r"var beklenen = (\[.*?\]);", HTML).group(1))
    assert all(t == "function" for t in r["tipler"].values()), r["tipler"]
    assert r["serit"] == []


@JS
def test_sabotaj_yalniz_hedef_modulu_dusurur_ve_seridi_basar(tmp_path):
    r = _kos(tmp_path, sabote="m03-loadPredictions")
    assert r["hatalar"] == ["SOZDIZIMI 3"]
    assert "m03-loadPredictions" not in r["moduller"] and len(r["moduller"]) >= 8
    assert r["tipler"]["loadPredictions"] == "undefined"
    for ad in ("loadAllData", "switchTab", "loadDecisionFunnel", "loadSimilarEpisodes", "qmRenderMicro"):
        assert r["tipler"][ad] == "function", ad
    assert len(r["serit"]) == 1 and "m03-loadPredictions" in r["serit"][0]


@JS
def test_huni_cizicisi_gercek_bicimde_json_cizer(tmp_path):
    ornek = {"ok": True, "dosya": "decision_log_2026_08_31.txt", "toplam_deneme": 118404, "kabul": 65,
             "uyari": 398, "pozisyon_acilis": 65, "n_sembol": 520, "taninmayan": {},
             "sembol_en_ileri": {"KABUL": 62, "RISK_REJECT": 186},
             "asamalar": [{"asama": "SESSION_GUARD", "elenen": 100690, "kalan": 17714,
                           "alt_neden": {"MARKET_CLOSED": 92597, "<script>": 1}},
                          {"asama": "COOLDOWN", "elenen": 0, "kalan": 17714, "alt_neden": {}},
                          {"asama": "RISK_REJECT", "elenen": 17649, "kalan": 65, "alt_neden": {"DAILY_MAX_LOSS_HIT": 624}}]}
    yol = tmp_path / "ornek.json"
    yol.write_text(json.dumps(ornek), encoding="utf-8")
    r = _kos(tmp_path, ornek=yol)
    assert "Kabul (sanal pozisyon)" in r["huni"] and "Seans" in r["huni"] and "Risk reddi" in r["huni"]
    assert "Soğuma</text>" not in r["huni"]                  # elenen 0 satırı SVG'de çizilmez (tabloda var)
    assert "<script>" not in r["huni"] and "&lt;script&gt;" in r["huni"]   # veri kaçışlı
    assert 'role="img"' in r["huni"] and "log ölçek" in r["huni"]
    assert "karar logu yok" in r["huniYok"]


def _u_barlari(tz_aware: bool):
    import numpy as np
    import pandas as pd
    gunler = pd.bdate_range("2024-06-03", periods=8)
    idx = []
    for g in gunler:
        bas = pd.Timestamp(g.date()).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        idx.extend(bas + pd.Timedelta(minutes=15 * i) for i in range(26))
    idx = pd.DatetimeIndex(idx)
    if not tz_aware:
        idx = idx.tz_convert("UTC").tz_localize(None)             # Alpaca kuralı: naive-UTC
    m = np.tile(np.arange(26), len(gunler))
    vol = 1000.0 * (1 + 3 * ((m - 12.5) / 12.5) ** 2)
    close = np.full(vol.size, 100.0)
    return pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
                         "volume": vol}, index=idx)


@pytest.mark.parametrize(("tz_aware", "saglayici", "ok"), [
    (True, "yfinance+cache45s", True),        # tz-farkında indeks: sağlayıcıdan bağımsız ölçülür
    (False, "alpaca+cache45s", True),         # naive + doğrulanmış naive-UTC sağlayıcı
    (False, "hub+cache45s", True),
    (False, "tradingview", False),            # naive + doğrulanmamış sağlayıcı: saat dilimi UYDURULMAZ
])
def test_ucurve_ucu_saat_dilimini_tahmin_etmez(monkeypatch, tz_aware, saglayici, ok):
    from types import SimpleNamespace

    from nihai.web import api_quant
    monkeypatch.setattr(api_quant, "_bars_raw", lambda s, i, lb: _u_barlari(tz_aware))
    monkeypatch.setattr(api_quant, "_provider", lambda: SimpleNamespace(name=saglayici))
    r = api_quant._ucurve_block("SPY")
    assert r["ok"] is ok
    if ok:
        assert r["hacim"]["slots"][0] == "09:30" and r["hacim"]["n_sessions"] == 7
        med = r["hacim"]["median"]
        assert med[0] > 3 * med[6]                                 # U: açılış > öğle
        assert r["cekilme"] is None and "MAKAS_VERISI_YOK" in r["cekilme_reason"]
        assert r["index_tz"] == ("America/New_York" if tz_aware else "naive→UTC")   # kaynağı söyler
    else:
        assert r["reason"] == "SAAT_DILIMI_DOGRULANAMADI"


def test_ucurve_ucu_bar_yoksa_neden_doner(monkeypatch):
    from nihai.web import api_quant
    monkeypatch.setattr(api_quant, "_bars_raw", lambda s, i, lb: None)
    assert api_quant._ucurve_block("SPY") == {"ok": False, "reason": "BAR_YOK"}


def test_kartlar_sekme_dugmelerine_ve_uclara_bagli():
    assert "switchTab('tab-signals', this);loadDecisionFunnel()" in HTML
    assert "switchTab('tab-brain', this);loadSimilarEpisodes()" in HTML
    assert 'id="funnel-body"' in HTML and 'id="episode-similar-body"' in HTML
    assert "decision/funnel" in HTML and "ai/episodes/similar" in HTML
    assert '<canvas id="qmc-mi-ucurve">' in HTML and "qmChart('qmc-mi-ucurve'" in HTML
