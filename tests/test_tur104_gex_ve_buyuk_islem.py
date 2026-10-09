"""Tur-104 — Emir Akışı'na net gamma pozlaması (GEX), footprint Delta/Hacim kipi ve büyük işlemler.

Dört soru kilitlenir:
  1. GEX HESABI DOĞRU MU — çok vadeli toplam vade başına `dealer_gamma_exposure`ın TOPLAMIDIR (strike, ızgara,
     net); flip toplam eğrinin spota en yakın kökü; IV/OI'siz satır atılır ve SAYILIR; tek taraflı vade atlanır.
  2. VERİ YOLU DOĞRU MU — Bybit/OKX sembol ayrıştırma (ay, 08:00 UTC, USDT/USDC son eki), OKX OI ↔ IV eşlemesi,
     süresi dolmuş vade, borsa başına fail-soft; hisse zinciri yalnız ≤45 gün / en çok 8 vade ister.
  3. UÇ — fail-soft, `def`, başarılı sonuç önbellekte (başarısız DEĞİL), `min_size` → `en_az`.
  4. PANO — denetimler + istek; çizici GEX'i, üç footprint kipini, büyük işlemler tablosunu hatasız çizer ve
     yükleme sürerken değişen sembolü düşürmez (node varsa).
Ağ YOK: her dış çağrı enjekte edilen sahtelerle.
"""
from __future__ import annotations

import inspect
import json
import shutil
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nihai.ai.option_flow import dealer_gamma_exposure
from nihai.data import orderflow as O

KOK = Path(__file__).resolve().parents[1]
HTML = (KOK / "nihai" / "web" / "web_dashboard.html").read_text(encoding="utf-8")
NODE = shutil.which("node")
SIMDI = datetime(2026, 10, 1, 8, tzinfo=UTC)


def _zincir(kaydir=0.0, oi_c=1000.0, oi_p=1500.0):
    """Spot 100 çevresi: put OI aşağıda, call OI yukarıda yoğun → toplam eğri ızgarada işaret değiştirir."""
    k = np.arange(80.0, 121.0, 2.5)
    calls = {"strike": k, "openInterest": np.where(k >= 100, oi_c, 100.0), "impliedVolatility": 0.30 + kaydir + 0 * k}
    puts = {"strike": k, "openInterest": np.where(k <= 100, oi_p, 100.0), "impliedVolatility": 0.35 + kaydir + 0 * k}
    return calls, puts


# --------------------------------------------------------------------------- 1. GEX hesabı
def test_gex_temizle_iv_oi_strike_eksigi_atilir_ve_sayilir():
    df = pd.DataFrame({"strike": [90.0, 95.0, 100.0, 105.0, -5.0, 110.0, 115.0],
                       "openInterest": [10.0, np.nan, 5.0, 7.0, 3.0, 2.0, np.inf],      # NaN ve +inf OI
                       "impliedVolatility": [0.3, 0.3, 0.0, np.inf, 0.3, 0.25, 0.3]})
    t, n = O.gex_temizle(df)
    assert n == 5 and list(t["strike"]) == [90.0, 110.0] and list(t["openInterest"]) == [10.0, 2.0]
    assert O.gex_temizle({"strike": [1.0], "openInterest": [1.0], "impliedVolatility": [np.nan]}) == (None, 1)
    assert O.gex_temizle(None) == (None, 0)
    t, n = O.gex_temizle({"strike": [100.0, 105.0], "openInterest": [3.0, 4.0]})        # IV kolonu YOK
    assert t is None and n == 2


def test_gex_flip_dogrusal_kok_spota_en_yakin_ve_dejenere_girdi():
    assert O.gex_flip([0, 1, 2, 3], [-2, -1, 1, 3], 1.0) == pytest.approx(1.5)
    x, y = np.arange(7.0), [1, -1, -1, 1, 1, -1, -1]                               # kökler 0,5 · 2,5 · 4,5
    assert O.gex_flip(x, y, 4.2) == pytest.approx(4.5) and O.gex_flip(x, y, 1.4) == pytest.approx(0.5)
    assert O.gex_flip([0, 1, 2], [1, 2, 3], 1.0) is None
    assert O.gex_flip([0, 1, 2], [0.0, -0.0, 0.0], 1.0) is None                     # ±0: bölme yok, kök yok
    assert O.gex_flip([5, 6], [0.0, -1.0], 5.0) == 5.0


def test_gex_toplu_vade_basina_dealer_gex_toplamidir():
    v1, v2 = _zincir(), _zincir(kaydir=0.1, oi_c=400.0, oi_p=300.0)
    t1, t2, spot = 0.02, 0.15, 100.0
    g = O.gex_toplu([("V1", t1, *v1), ("V2", t2, *v2)], spot, r=0.04, kontrat=100)
    a = dealer_gamma_exposure(*v1, spot, t1, 0.04, 100, O.GEX_IZGARA, O.GEX_SPAN)
    b = dealer_gamma_exposure(*v2, spot, t2, 0.04, 100, O.GEX_IZGARA, O.GEX_SPAN)
    beklenen_net = a["net_gex"] + b["net_gex"]
    assert g["net"] == pytest.approx(beklenen_net, rel=1e-7)
    ks = np.array([s[0] for s in g["strikes"]])
    assert np.array_equal(ks, a["strikes"])                                         # aynı strike kümesi
    assert np.allclose([s[1] for s in g["strikes"]], a["call_gex"] + b["call_gex"], rtol=1e-7)
    assert np.allclose([s[2] for s in g["strikes"]], a["put_gex"] + b["put_gex"], rtol=1e-7)
    izg = np.array(g["izgara"])
    assert np.allclose(izg[:, 1], a["grid_net_gex"] + b["grid_net_gex"], rtol=1e-7)
    assert izg[len(izg) // 2, 0] == pytest.approx(spot) and izg[len(izg) // 2, 1] == pytest.approx(beklenen_net, rel=1e-7)
    assert g["flip"] == O.gex_flip(a["grid_spot"], a["grid_net_gex"] + b["grid_net_gex"], spot) is not None
    toplam = np.array([s[1] + s[2] for s in g["strikes"]])
    assert g["call_wall"] == ks[np.argmax([s[1] for s in g["strikes"]])]
    assert g["put_wall"] == ks[np.argmin([s[2] for s in g["strikes"]])]
    assert g["rejim"] == ("pozitif" if toplam.sum() > 0 else "negatif")
    assert [v["vade"] for v in g["vadeler"]] == ["V1", "V2"] and "VARSAYIM" in g["konvansiyon"]


def test_gex_toplu_tek_vadede_flip_brentq_kokune_izgara_adimi_icinde():
    c, p = _zincir()
    d = dealer_gamma_exposure(c, p, 100.0, 0.05, 0.04, 100, O.GEX_IZGARA, O.GEX_SPAN)
    g = O.gex_toplu([("V", 0.05, c, p)], 100.0, r=0.04)
    adim = 2 * O.GEX_SPAN * 100.0 / (O.GEX_IZGARA - 1)
    assert d["gamma_flip"] is not None and abs(g["flip"] - d["gamma_flip"]) < adim
    assert g["net"] == pytest.approx(d["net_gex"], rel=1e-7)


def test_gex_toplu_tek_tarafli_ve_suresi_dolmus_vade_atlanir_atilan_sayilir():
    c, p = _zincir()
    bozuk_put = dict(p, impliedVolatility=np.full(p["strike"].size, np.nan))      # put tarafı tamamen IV'siz
    kirli = dict(c, openInterest=np.where(c["strike"] == 110.0, np.nan, c["openInterest"]))
    g = O.gex_toplu([("TEK", 0.05, c, bozuk_put), ("BITTI", 0.0, c, p), ("IYI", 0.05, kirli, p)], 100.0)
    assert [v.get("atlandi", False) for v in g["vadeler"]] == [True, True, False]
    assert g["n_atilan"] == p["strike"].size + 1
    tek = O.gex_toplu([("IYI", 0.05, kirli, p)], 100.0)
    assert g["net"] == tek["net"] and g["strikes"] == tek["strikes"]               # atlanan vade nete GİRMEZ
    assert O.gex_toplu([("TEK", 0.05, c, bozuk_put)], 100.0) is None


# --------------------------------------------------------------------------- 2. veri yolu
def _sahte_borsa(bybit=True, okx=True, cagri=None):
    def get(url, params):
        if cagri is not None:
            cagri.append((url, dict(params)))
        if "bybit" in url:
            if not bybit:
                raise ConnectionError("bybit kapalı")
            assert params == {"category": "option", "baseCoin": "BTC"}
            return {"result": {"list": [
                {"symbol": "BTC-3OCT26-80000-C-USDT", "openInterest": "10", "markIv": "0.5", "indexPrice": "84000"},
                {"symbol": "BTC-3OCT26-80000-P", "openInterest": "5", "markIv": "0.55", "indexPrice": "84010"},
                {"symbol": "BTC-3OCT26-90000-C-USDT", "openInterest": "8", "markIv": "0.45", "indexPrice": "84020"},
                {"symbol": "BTC-3OCT26-90000-P-USDT", "openInterest": "x", "markIv": "0.6", "indexPrice": "?"},
                {"symbol": "ETH-3OCT26-2000-C-USDT", "openInterest": "99", "markIv": "0.5", "indexPrice": "2700"},
                {"symbol": "BTC-30SEP26-80000-C-USDT", "openInterest": "1", "markIv": "0.5", "indexPrice": "84000"},
                {"symbol": "BTC-3XXX26-80000-C-USDT", "openInterest": "1", "markIv": "0.5"},
                {"symbol": "çöp"}]}}
        if not okx:
            raise TimeoutError("okx yavaş")
        if url.endswith("/open-interest"):
            assert params == {"instType": "OPTION", "instFamily": "BTC-USD"}
            return {"data": [{"instId": "BTC-USD-261009-85000-C", "oiCcy": "3"},
                             {"instId": "BTC-USD-261009-85000-P", "oiCcy": "4"},
                             {"instId": "BTC-USD-260917-70000-C", "oiCcy": "0"}]}
        if url.endswith("/opt-summary"):
            return {"data": [{"instId": "BTC-USD-261009-85000-C", "markVol": "0.40"},
                             {"instId": "BTC-USD-261009-85000-P", "markVol": "0.45"},
                             {"instId": "BTC-USD-261009-86000-C", "markVol": "0.41"}]}       # OI'siz
        if url.endswith("/index-tickers"):
            return {"data": [{"instId": "BTC-USD", "idxPx": "84500"}]}
        raise AssertionError(url)
    return get


def test_kripto_zinciri_bybit_okx_ayristirma_ve_vade_saati():
    spot, vadeler, kaynak, uyari = O.kripto_opsiyon_vadeleri("BTC", SIMDI, _sahte_borsa())
    assert spot == pytest.approx(84005.0) and uyari == []                           # Bybit endeks medyanı (4 fiyat)
    assert kaynak == ["Bybit (5 kontrat)", "OKX (3 kontrat)"]                       # ETH/çöp/ay hatası sayılmaz
    assert [v[0] for v in vadeler] == ["Bybit 2026-10-03", "OKX 2026-10-09"]        # 30 Eylül vadesi düştü
    by, ok = vadeler
    assert by[1] == pytest.approx(2 / 365) and ok[1] == pytest.approx(8 / 365)       # 08:00 UTC teslim
    assert list(by[2]["strike"]) == [80000.0, 90000.0] and list(by[2]["openInterest"]) == [10.0, 8.0]
    assert list(by[3]["strike"]) == [80000.0, 90000.0] and np.isnan(by[3]["openInterest"][1])
    assert list(ok[2]["strike"]) == [85000.0, 86000.0] and np.isnan(ok[2]["openInterest"][1])
    assert list(ok[3]["openInterest"]) == [4.0] and list(ok[3]["impliedVolatility"]) == [0.45]


def test_kripto_zinciri_borsa_basina_fail_soft_ve_okx_endeks_yedegi():
    cagri = []
    spot, vadeler, kaynak, uyari = O.kripto_opsiyon_vadeleri("BTC", SIMDI, _sahte_borsa(bybit=False, cagri=cagri))
    assert spot == 84500.0 and kaynak == ["OKX (3 kontrat)"] and [v[0] for v in vadeler] == ["OKX 2026-10-09"]
    assert uyari == ["Bybit okunamadı: ConnectionError"]
    assert any(u.endswith("/index-tickers") for u, _ in cagri)
    cagri.clear()
    spot, _, kaynak, uyari = O.kripto_opsiyon_vadeleri("BTC", SIMDI, _sahte_borsa(okx=False, cagri=cagri))
    assert spot == pytest.approx(84005.0) and uyari == ["OKX okunamadı: TimeoutError"] and len(kaynak) == 1
    assert not any(u.endswith("/index-tickers") for u, _ in cagri)                   # Bybit endeksi varken sorulmaz


def test_kripto_gex_paketi_ve_zincirsiz_coin():
    g = O.kripto_gex("BTC", simdi=SIMDI, get=_sahte_borsa())
    assert g["ok"] is True and g["piyasa"] == "kripto" and g["kaynak"] == "Bybit (5 kontrat) + OKX (3 kontrat)"
    assert g["n_atilan"] == 2 and len(g["vadeler"]) == 2                            # OI'si okunamayan iki satır
    spot, vadeler, _, _ = O.kripto_opsiyon_vadeleri("BTC", SIMDI, _sahte_borsa())
    assert g["net"] == O.gex_toplu(vadeler, spot, r=0.0, kontrat=1)["net"]           # OI coin cinsinden → çarpan 1
    bos = O.kripto_gex("XYZ", simdi=SIMDI, get=lambda u, p: {"result": {"list": []}} if "bybit" in u else {"data": []})
    assert bos["ok"] is False and "XYZ" in bos["reason"]


def test_kripto_tabani():
    assert O.kripto_tabani("btcusdt") == "BTC" and O.kripto_tabani("ETH-USD") == "ETH"
    assert O.kripto_tabani("SOLUSDC") == "SOL" and O.kripto_tabani("AAPL") is None


def test_baloncuk_en_az_suzgeci():
    t = np.arange(10, dtype=np.int64) * 10
    p = np.full(10, 100.0)
    q = np.array([1, 2, 3, 40, 5, 60, 7, 80, 9, 100], dtype=float)
    s = np.where(np.arange(10) % 2, 1, -1)
    hepsi = O.baloncuklar(t, p, q, s, en_cok=10)
    sec = O.baloncuklar(t, p, q, s, en_cok=10, en_az=40)
    assert len(hepsi["items"]) == 10 and [i[2] for i in sec["items"]] == [40.0, 60.0, 80.0, 100.0]
    assert sec["esik"] == 40.0 and sec["n_emir"] == 10
    assert O.baloncuklar(t, p, q, s, en_cok=10, en_az=1e9) == {"items": [], "esik": None, "n_emir": 10}


# --------------------------------------------------------------------------- 3. uç
@pytest.fixture
def istemci(monkeypatch):
    from fastapi.testclient import TestClient

    from nihai.web import api_orderflow as A
    from nihai.web.web_dashboard import app
    monkeypatch.setattr(A, "_GEX_ON", {})
    return TestClient(app), A


def test_gex_ucu_kripto_onbellek_basarisiz_onbelleklenmez_fail_soft(istemci, monkeypatch):
    c, A = istemci
    assert c.get("/api/orderflow/gex", params={"symbol": "BT$C"}).json()["ok"] is False
    cagri = []
    monkeypatch.setattr(A.OF, "kripto_gex", lambda coin: cagri.append(coin) or {"ok": True, "net": 1.0})
    r1 = c.get("/api/orderflow/gex", params={"symbol": "btcusdt"}).json()
    r2 = c.get("/api/orderflow/gex", params={"symbol": "BTCUSDT"}).json()
    assert cagri == ["BTC"] and r1 == r2 and r1["symbol"] == "BTCUSDT" and r1["zaman"] > 0
    monkeypatch.setattr(A.OF, "kripto_gex", lambda coin: cagri.append(coin) or {"ok": False, "reason": "yok"})
    for _ in range(2):
        assert c.get("/api/orderflow/gex", params={"symbol": "ETH-USD"}).json()["ok"] is False
    assert cagri == ["BTC", "ETH", "ETH"]

    def patla(coin):
        raise RuntimeError("ağ yok")
    monkeypatch.setattr(A.OF, "kripto_gex", patla)
    assert c.get("/api/orderflow/gex", params={"symbol": "SOLUSDT"}).json() == {"ok": False,
                                                                              "reason": "RuntimeError: ağ yok"}
    assert not inspect.iscoroutinefunction(A.orderflow_gex)


class _SahteSaglayici:
    def __init__(self):
        self.istek = []

    def get_option_chain(self, sym, expiry):
        self.istek.append(expiry)
        if expiry is None:
            return {"underlying_price": 100.0, "all_expiries": [f"E{i:02d}" for i in range(14)]}
        c, p = _zincir(kaydir=int(expiry[1:]) / 100)
        return {"calls": pd.DataFrame(c), "puts": pd.DataFrame(p), "expiry": expiry, "option_source": "yfinance"}


def test_hisse_gex_vade_ufku_ve_toplam(istemci, monkeypatch):
    c, A = istemci
    from nihai.web import api_quant
    gun = {"E00": 0, "E01": 1, "E02": 3, "E03": 50, "E04": 5, "E05": 8, "E06": 10, "E07": 15, "E08": 20,
           "E09": 30, "E10": 40, "E11": 44, "E12": 45, "E13": 60}
    s = _SahteSaglayici()
    monkeypatch.setattr(api_quant, "_provider", lambda: s)
    monkeypatch.setattr(api_quant, "_expiry_years", lambda e: gun[e] / 365)
    g = c.get("/api/orderflow/gex", params={"symbol": "spy"}).json()
    secilen = ["E01", "E02", "E04", "E05", "E06", "E07", "E08", "E09"]               # >0, ≤45 gün, ilk 8
    assert s.istek[0] is None and sorted(s.istek[1:]) == secilen
    assert g["ok"] is True and g["piyasa"] == "hisse" and [v["vade"] for v in g["vadeler"]] == secilen
    beklenen = O.gex_toplu([(e, gun[e] / 365, *_zincir(kaydir=int(e[1:]) / 100)) for e in secilen], 100.0,
                           r=0.04, kontrat=100)
    assert g["net"] == beklenen["net"] and g["flip"] == pytest.approx(beklenen["flip"])
    assert "yfinance" in g["kaynak"] and g["yenile_sn"] == 600


def test_orderflow_min_size_dogrulama_ve_iletim(istemci, monkeypatch):
    c, A = istemci
    alinan = {}
    monkeypatch.setattr(A.OF, "emir_akisi", lambda sym, **k: alinan.update(k) or {"ok": True})
    for kotu in ("-1", "nan", "inf"):                                               # sahte "ok" dönerken bile red
        assert c.get("/api/orderflow", params={"symbol": "AAPL", "min_size": kotu}).json()["ok"] is False
    assert alinan == {}
    c.get("/api/orderflow", params={"symbol": "AAPL", "min_size": "25.5"})
    assert alinan["en_az"] == 25.5


def test_en_az_emir_akisindan_baloncuga_iki_piyasada_ulasir(monkeypatch):
    from tests.test_tur103_emir_akisi import _rastgele
    t, p, q, s = _rastgele(n=2000)
    hepsi = O._paketle(t, p, q, s, 60, 0.0, 0.01, 150)["baloncuk"]
    sec = O._paketle(t, p, q, s, 60, 0.0, 0.01, 150, en_az=490.0)["baloncuk"]
    assert hepsi["esik"] < 490 <= sec["esik"] and 0 < len(sec["items"]) < len(hepsi["items"])   # kurgu AYIRT EDİCİ
    alinan = []
    gercek = O._paketle
    monkeypatch.setattr(O, "_paketle", lambda *a: alinan.append(a[8]) or gercek(*a))

    def cek(sym, bas, son, feed, limit):
        ms = np.arange(int(bas.timestamp() * 1000) + 1, int(son.timestamp() * 1000), 1000, dtype=np.int64)
        return O.Islemler(ms, 10 + (ms % 7) / 100, np.ones(ms.size))

    O._HISSE_DEPO.clear()
    assert O.emir_akisi("ENAZ", dakika=10, feed="iex", en_az=7.0, simdi=SIMDI, cek=cek)["ok"]
    O._HISSE_DEPO.clear()

    class Kaydedici:
        hazir, baslangic_ms, aralik_sn, hata, isitma_dk = threading.Event(), int(t[0]), 3, None, 60

        def islemler(self):
            return t, p, q, s

        def goruntuler(self):
            return []
    Kaydedici.hazir.set()
    assert O.emir_akisi("BTCUSDT", dakika=30, en_az=9.0, kaydedici_al=lambda sym: Kaydedici())["ok"]
    assert alinan == [7.0, 9.0]


# --------------------------------------------------------------------------- 4. pano
def test_pano_denetimleri_ve_istekler_bagli():
    for id_ in ("of-k-gex", "of-fp-mod", "of-min", "of-gex-bilgi", "of-buyuk"):
        assert f'id="{id_}"' in HTML, id_
    assert "apiCall('orderflow/gex?' + new URLSearchParams({ symbol: sym }).toString())" in HTML
    assert "min_size: String(enAz > 0 ? enAz : 0)" in HTML
    for kip in ('value="sa"', 'value="delta"', 'value="hacim"'):
        assert kip in HTML


_CIZ = r"""
const fs = require("fs"); const vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const veri = JSON.parse(fs.readFileSync(process.argv[3], "utf8")), gex = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
const blok = /<script>\s*\/\/ ===== \[pano modülü m11-emir-akisi\][\s\S]*?<\/script>/.exec(html)[0].replace(/<\/?script>/g, "");
let yazi = [];
const ctx = new Proxy({}, { get(t, k) { if (k in t) return t[k]; return (...a) => { if (k === "fillText") yazi.push(String(a[0])); }; },
                            set(t, k, v) { t[k] = v; return true; } });
const el = () => ({ checked: true, value: "50", style: {}, clientWidth: 1400, clientHeight: 660, width: 0, height: 0,
  offsetWidth: 100, offsetHeight: 40, textContent: "", innerHTML: "", classList: { contains: () => true },
  getContext: () => ctx, addEventListener() {}, getBoundingClientRect: () => ({ left: 0, top: 0 }) });
const dom = {};
const document = { getElementById: (id) => (dom[id] = dom[id] || el()), createElement: () => el(), querySelector: () => null };
const istek = [];
async function apiCall(yol) {
  istek.push(yol); await Promise.resolve(); await Promise.resolve();
  if (yol.startsWith("orderflow/gex")) return gex;
  return Object.assign({}, veri, { symbol: new URLSearchParams(yol.split("?")[1]).get("symbol") });
}
const sb = { document, window: {}, console, Math, JSON, Number, String, Date, Array, Object, isFinite, URLSearchParams,
             Promise, apiCall, setTimeout: () => 1, clearTimeout() {} };
sb.window = sb; sb.window.__PANO_MODULLERI = []; sb.window.devicePixelRatio = 1; sb.addEventListener = () => {};
vm.createContext(sb); vm.runInContext(blok, sb);
sb.dom = dom; sb.istek = istek; sb.yaziAl = () => { const y = yazi; yazi = []; return y; };
vm.runInContext(`(async () => {
  const o = {};
  document.getElementById('of-sym').value = 'AAA'; const ilk = ofYukle();
  document.getElementById('of-sym').value = 'BBB'; ofYukle();                       // AAA sürerken → düşmemeli
  await ilk;
  for (let i = 0; i < 30; i++) await Promise.resolve();
  o.sym = OF.veri.symbol; o.gexSym = OF.gexSym;
  o.gexBilgi = dom['of-gex-bilgi'].innerHTML; o.buyuk = dom['of-buyuk'].innerHTML;
  o.mod = {};
  for (const m of ['sa', 'delta', 'hacim']) {
    document.getElementById('of-fp-mod').value = m; OF.n = 4; OF.kaydir = 0; yaziAl(); ofCiz();
    const y = yaziAl();
    o.mod[m] = { carpi: y.filter((s) => s === '×').length, isaretli: y.filter((s) => /^[+-][0-9]/.test(s)).length, n: y.length };
    o.GW = OF.geo.GW;
  }
  document.getElementById('of-fp-mod').value = 'delta'; ofCiz(); o.eksen = yaziAl().filter((s) => /FLIP|CW|PW|Net GEX/.test(s));
  OF.fare = { x: 300, y: 200 }; ofImlec(); o.ipucu = dom['of-ipucu'].innerHTML;
  OF.gex = { ok: false, reason: 'zincir yok' }; ofGexBilgi(); ofCiz(); o.bozukGW = OF.geo.GW; o.bozuk = dom['of-gex-bilgi'].innerHTML;
  o.istek = istek.slice();
  const gp = ofGexYukle('CCC'); OF.veri.symbol = 'DDD'; ofGexYukle('DDD');     // GEX isteği sürerken sembol değişti
  await gp; for (let i = 0; i < 30; i++) await Promise.resolve();
  o.gexSon = OF.gexSym; const n0 = istek.length; await ofGexYukle('DDD'); o.tazeAtlandi = istek.length === n0;
  o.gexIstek = istek.slice(o.istek.length);
  console.log(JSON.stringify(o));
})().catch((e) => { console.error(e && e.stack || e); process.exitCode = 3; });`, Object.assign(sb, { process }));
"""


@pytest.mark.skipif(NODE is None, reason="node kurulu değil — JS ölçülemedi")
def test_cizici_gex_footprint_kipleri_buyuk_islem_ve_yaris(tmp_path):
    from tests.test_tur103_emir_akisi import _rastgele
    t, p, q, s = _rastgele(n=3000)
    v = O._paketle(t, p, q, s, 60, 0.0, 0.01, 50)
    v.update({"ok": True, "symbol": "X", "feed": "sip", "piyasa": "hisse", "kaynak": "test", "taraf_yontemi": "tick",
              "bas": int(t[0]), "son": int(t[-1]), "uyari": [], "yenile_sn": 30, "isi": None, "isi_neden": "yok"})
    lo, hi = float(p.min()), float(p.max())
    c, pt = _zincir()
    c["strike"] = pt["strike"] = np.round(np.linspace(lo, hi, c["strike"].size), 2)
    g = O.gex_toplu([("V", 0.05, c, pt)], (lo + hi) / 2, r=0.04)
    g.update({"ok": True, "flip": hi + 5.0, "call_wall": v["bars"][-1]["c"], "put_wall": lo - 5.0, "kaynak": "test",
              "vade_ufku": "test", "uyari": [], "yenile_sn": 600})
    for ad, icerik in (("v.json", v), ("g.json", g)):
        (tmp_path / ad).write_text(json.dumps(icerik), encoding="utf-8")
    (tmp_path / "c.js").write_text(_CIZ, encoding="utf-8")
    (tmp_path / "p.html").write_text(HTML, encoding="utf-8")
    r = subprocess.run([NODE, str(tmp_path / "c.js"), str(tmp_path / "p.html"), str(tmp_path / "v.json"),
                        str(tmp_path / "g.json")], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[:800]
    o = json.loads(r.stdout)
    assert o["sym"] == "BBB" and o["gexSym"] == "BBB"                                 # bayat AAA yanıtı atıldı
    assert [i.split("?")[0] for i in o["istek"]] == ["orderflow", "orderflow", "orderflow/gex"]
    assert "symbol=BBB" in o["istek"][1] and "min_size=50" in o["istek"][1] and "symbol=BBB" in o["istek"][2]
    assert "Net GEX" in o["gexBilgi"] and "VARSAYIM" in o["gexBilgi"]
    assert "<table>" in o["buyuk"] and ("ALIŞ" in o["buyuk"] or "SATIŞ" in o["buyuk"])
    m = o["mod"]
    assert m["sa"]["carpi"] > 0 and m["delta"]["carpi"] == 0 and m["hacim"]["carpi"] == 0
    assert m["delta"]["isaretli"] > m["hacim"]["isaretli"] and o["GW"] > 0
    assert "Net GEX" in o["eksen"] and any(x.startswith("▲FLIP") for x in o["eksen"])
    assert any(x.startswith("▼PW") for x in o["eksen"]) and any(x.startswith("CW ") for x in o["eksen"])
    assert "net GEX" in o["ipucu"]
    assert o["bozukGW"] == 0 and "YOK" in o["bozuk"] and "zincir yok" in o["bozuk"]
    assert o["gexSon"] == "DDD" and o["tazeAtlandi"] is True
    assert o["gexIstek"] == ["orderflow/gex?symbol=CCC", "orderflow/gex?symbol=DDD"]
