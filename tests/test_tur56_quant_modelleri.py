"""test_tur56_quant_modelleri.py — /api/quant/* sözleşmesi + pano bağlantısı.

İki soruyu birden sorar (Tur-37/38 test deseni):

  1) **Uç FAIL-SOFT mu?** Sağlayıcı yoksa/veri kısaysa uç 500 ATMAMALI;
     `{"ok": false, "reason": ...}` dönmeli ki tek bir eksik parça sekmeyi karartmasın.
  2) **Bağlandı mı?** Bir uç yazılıp panodan ÇAĞRILMIYORSA bu depoda kusurdur
     (Tur-37 "yazılıp okunmayan kolon" kuralının web tarafı). Katalogdaki her model
     için panoda panel + açıklama kancası VAR olmalı ve her `formul` referansı
     GERÇEK bir fonksiyona işaret etmeli — uydurma fonksiyon adı YOK.

Ağ gerektiren uçlar (`/api/quant/smile` canlı zincir ister) sahte sağlayıcıyla
koşulur; matematiğin kendisi zaten `test_ai_quant_math`/`test_ai_options_math`/
`test_ai_microstructure` altında kilitlidir — burada ölçülen ARAYÜZDÜR.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import nihai.web.api_quant as aq
from nihai.web.web_dashboard import app

client = TestClient(app)

DASHBOARD_HTML = Path(__file__).resolve().parents[1] / "nihai" / "web" / "web_dashboard.html"

# Ağ/sağlayıcı gerektirmeyen SAF hesaplayıcılar — her ortamda ok=True dönmeli.
PURE_ENDPOINTS = [
    "/api/quant/catalog",
    "/api/quant/options?s=100&k=100&t=0.25&r=0.04&sigma=0.25&kind=c",
    "/api/quant/options?s=90&k=100&t=0.08&r=0.04&sigma=0.6&kind=p&premium=11.5",
    "/api/quant/kelly?p_win=0.55&win_loss=1.5",
    "/api/quant/execution?q=10000&sigma=0.02&eta=1e-6&gamma_risk=1e-6&t=1&n=20",
]

# Veri gerektirenler — sağlayıcı yoksa ok=False DÖNER ama 500 ATMAZ.
DATA_ENDPOINTS = [
    "/api/quant/risk?symbol=SPY&interval=1d&lookback=300",
    "/api/quant/vol?symbol=SPY&interval=1d&lookback=300",
    "/api/quant/regime?symbol=SPY&interval=1d&lookback=300",
    "/api/quant/mc?symbol=SPY&interval=1d&lookback=300&paths=500",
    "/api/quant/micro?symbol=SPY&interval=1h&lookback=300",
    "/api/quant/smile?symbol=SPY",
    "/api/quant/portfolio?symbols=AAA,BBB,CCC&interval=1d&lookback=200",
    "/api/quant/flow?symbol=SPY&interval=1d&lookback=300",
    "/api/quant/statarb?symbols=AAA,BBB,CCC,DDD&interval=1d&lookback=300",
]


# ---------------------------------------------------------------------------
# Sahte sağlayıcı — deterministik, ağ YOK
# ---------------------------------------------------------------------------
class _FakeProvider:
    """Bar + opsiyon zinciri üretir. Zincir GERÇEK Black-Scholes fiyatlarından
    kurulur; aksi halde SVI/RND uyumu veriye değil gürültüye bakardı."""

    def get_bars(self, symbol, interval, lookback=200):
        n = min(int(lookback), 600)
        seed = abs(hash(symbol)) % 9973
        rng = np.random.default_rng(seed)
        r = rng.normal(0.0004, 0.012, n)
        r[n // 3 : n // 3 + 12] *= 3.0  # rejim/sıçrama ayrışması görünür olsun
        close = 100.0 * np.exp(np.cumsum(r))
        op = close * (1 + rng.normal(0, 0.002, n))
        hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
        lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
        vol = rng.integers(100_000, 5_000_000, n).astype(float)
        idx = pd.date_range("2023-01-02", periods=n, freq="B")
        return pd.DataFrame(
            {"open": op, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx
        )

    def get_option_chain(self, symbol, expiry=None):
        today = datetime.date.today()
        exps = [(today + datetime.timedelta(days=k)).isoformat() for k in (30, 60, 95)]
        if expiry is None:
            return {"all_expiries": exps}
        from nihai.ai.quant_math import bs_price

        under = 100.0
        t = max((datetime.date.fromisoformat(expiry) - today).days, 1) / 365.0
        strikes = np.arange(70.0, 131.0, 2.5)
        lm = np.log(strikes / under)
        iv = 0.22 + 0.35 * lm**2 - 0.10 * lm            # eğimli + kavisli gülümseme
        calls = np.array([bs_price(under, float(k), t, 0.04, float(s), "c") or 0.0
                          for k, s in zip(strikes, iv, strict=True)])
        puts = np.array([bs_price(under, float(k), t, 0.04, float(s), "p") or 0.0
                         for k, s in zip(strikes, iv, strict=True)])

        def _df(mid):
            return pd.DataFrame({"strike": strikes, "bid": mid * 0.98, "ask": mid * 1.02,
                                 "lastPrice": mid, "impliedVolatility": iv,
                                 "openInterest": 500, "volume": 100})

        return {"calls": _df(calls), "puts": _df(puts), "underlying_price": under,
                "expiry": expiry, "all_expiries": exps}


class _DeadProvider:
    """Her çağrıda patlar — fail-soft sözleşmesinin sınav ortamı."""

    def get_bars(self, *a, **k):
        raise RuntimeError("sağlayıcı yok")

    def get_option_chain(self, *a, **k):
        raise RuntimeError("sağlayıcı yok")


@pytest.fixture
def fake_provider(monkeypatch):
    monkeypatch.setattr(aq, "_provider", lambda: _FakeProvider())
    monkeypatch.setattr("nihai.execution.watchlist.current", lambda: ["AAA", "BBB", "CCC"])
    # Akış ucu broker bayraklarını da sorar; sahte sağlayıcıda o yol AĞA çıkmamalı
    # (bayraklar veri sağlayıcısından DEĞİL trading API'sinden gelir — Tur-57).
    monkeypatch.setattr(aq, "_asset_flags",
                        lambda s: {"symbol": s, "shortable": True, "easy_to_borrow": True,
                                   "marginable": True, "tradable": True, "as_of": "2026-08-29"})
    return _FakeProvider()


# ---------------------------------------------------------------------------
# 1) FAIL-SOFT sözleşmesi
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("url", PURE_ENDPOINTS + DATA_ENDPOINTS)
def test_uc_asla_500_atmaz(url, monkeypatch):
    """Sağlayıcı TAMAMEN ölüyken bile uç JSON + `ok` döndürmeli."""
    monkeypatch.setattr(aq, "_provider", lambda: _DeadProvider())
    monkeypatch.setattr(aq, "_asset_flags", lambda s: None)   # ağ guard'ı: broker sorgusu yok
    r = client.get(url)
    assert r.status_code == 200, f"{url} → {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert isinstance(body, dict) and "ok" in body


@pytest.mark.parametrize("url", DATA_ENDPOINTS)
def test_veri_yoksa_neden_soylenir(url, monkeypatch):
    """`ok=False` bir hata DEĞİL bir CEVAPTIR — ama NEDENİ boş bırakılamaz."""
    monkeypatch.setattr(aq, "_provider", lambda: _DeadProvider())
    monkeypatch.setattr(aq, "_asset_flags", lambda s: None)
    body = client.get(url).json()
    assert body["ok"] is False
    assert body.get("reason"), f"{url}: ok=False ama reason boş"


@pytest.mark.parametrize("url", PURE_ENDPOINTS)
def test_saf_hesaplayicilar_sagliyicisiz_calisir(url, monkeypatch):
    """Black-Scholes/Kelly/Almgren-Chriss ağ İSTEMEZ; sağlayıcı ölüyken de ok=True."""
    monkeypatch.setattr(aq, "_provider", lambda: _DeadProvider())
    assert client.get(url).json()["ok"] is True


@pytest.mark.parametrize("url", DATA_ENDPOINTS)
def test_veri_varken_hepsi_olcer(url, fake_provider):
    assert client.get(url).json()["ok"] is True, url


# ---------------------------------------------------------------------------
# 2) Grafik sözleşmesi — pano çizeceği seriyi BULABİLMELİ
# ---------------------------------------------------------------------------
def test_risk_dagilim_ve_kuyruk_serileri(fake_provider):
    d = client.get("/api/quant/risk?symbol=AAA&interval=1d&lookback=400").json()
    assert len(d["hist"]["centers"]) == len(d["hist"]["density"]) > 10
    assert len(d["underwater"]) == len(d["equity"]) == d["n_bars"]
    assert all(u <= 1e-9 for u in d["underwater"])          # su-altı eğrisi ≤ 0
    # VaR/CVaR KAYIP BÜYÜKLÜĞÜDÜR (≥0) ve CVaR kuyrukta VaR'dan küçük olamaz.
    t = d["tail"]
    assert t["hist_var"] >= 0 and t["hist_cvar"] >= t["hist_var"]
    assert t["hist_var_99"] >= t["hist_var"]                # daha uç kuyruk = daha büyük kayıp


def test_vol_koni_ve_garch_yolu_hizali(fake_provider):
    d = client.get("/api/quant/vol?symbol=AAA&interval=1d&lookback=400&horizon=15").json()
    assert len(d["realized_vol"]) == len(d["dates"])
    c = d["cone"]
    for k in ("p5", "p25", "p50", "p75", "p95", "current"):
        assert len(c[k]) == len(c["windows"])
    for i in range(len(c["windows"])):                      # yüzdelikler SIRALI olmalı
        assert c["p5"][i] <= c["p50"][i] <= c["p95"][i]
    g = d["garch"]
    if g is not None:
        assert len(g["cond_vol"]) == len(d["dates"])        # yol tarihlerle hizalı
        assert len(g["forecast"]) == 15
        assert 0.0 <= g["persistence"] < 1.0
        # IGARCH köşesinde koşulsuz vol SAYI BASILMAZ (Tur-49 dersi).
        assert (g["uncond_vol"] is None) == g["igarch_boundary"]


def test_regime_olasilik_seridi_toplami_bir(fake_provider):
    d = client.get("/api/quant/regime?symbol=AAA&interval=1d&lookback=400").json()
    h = d["hmm"]
    assert h is not None and h["state_probs"]
    n = len(d["dates"])
    assert all(len(p) == n for p in h["state_probs"])
    for i in (0, n // 2, n - 1):                            # her barda dağılım toplamı 1
        assert abs(sum(p[i] for p in h["state_probs"]) - 1.0) < 1e-3
    for row in h["matrix"]:                                 # geçiş matrisi satırları stokastik
        assert abs(sum(row) - 1.0) < 1e-3
    # Aynı grafikte çizilen HER seri AYNI tarih eksenine oturmalı. Uzunluk kayması
    # Chart.js'te istisna ÜRETMEZ — sessizce kaymış bir eğri çizer ve operatör kırılma
    # noktasını yanlış bara okur (bu depoda "sessiz görsel yalan" sınıfı bir kusurdur).
    b = d["bocpd"]
    assert len(d["close"]) == n
    assert len(b["changepoint_prob"]) == len(b["map_run_length"]) == n


def test_mc_yelpazesi_sirali_ve_terminal_tutarli(fake_provider):
    d = client.get("/api/quant/mc?symbol=AAA&interval=1d&lookback=300"
                   "&horizon=10&paths=800&seed=7").json()
    f = d["fan"]
    assert all(len(f[k]) == d["horizon"] + 1 for k in f)
    for i in range(d["horizon"] + 1):                       # p5 ≤ p50 ≤ p95, her adımda
        assert f["p5"][i] <= f["p50"][i] <= f["p95"][i]
    assert 0.0 <= d["terminal"]["prob_loss"] <= 1.0


def test_mc_seed_deterministik(fake_provider):
    """Aynı seed → aynı yelpaze (anayasa madde 6: determinizm)."""
    u = "/api/quant/mc?symbol=AAA&interval=1d&lookback=300&horizon=8&paths=600&seed=11"
    assert client.get(u).json()["fan"] == client.get(u).json()["fan"]


def test_options_egrileri_ve_ariteler():
    d = client.get("/api/quant/options?s=100&k=100&t=0.25&r=0.04&sigma=0.25&kind=c").json()
    n = len(d["spot_grid"])
    assert n == len(d["price_curve"]) == len(d["payoff"])
    for g in ("delta", "gamma", "theta", "vega", "rho"):
        assert len(d["greeks_curve"][g]) == n
    for g in ("vanna", "volga", "charm", "speed", "zomma", "color"):
        assert len(d["greeks_ext_curve"][g]) == n
    # Değer eğrisi vade-sonu ödemesinin ALTINA düşemez (erken kullanım sınırı).
    assert all(p >= po - 1e-6 for p, po in zip(d["price_curve"], d["payoff"], strict=True))
    # Call delta (0,1) aralığında ve spotla ARTAR; gamma ≥ 0.
    dl = d["greeks_curve"]["delta"]
    assert all(0.0 <= x <= 1.0 for x in dl) and dl[0] < dl[-1]
    assert all(x >= -1e-9 for x in d["greeks_curve"]["gamma"])
    b = d["now"]["no_arb_bounds"]
    assert b["lower"] <= d["now"]["price"] <= b["upper"]


def test_options_implied_vol_gidis_donus():
    """Primden çözülen σ, o primi üreten σ'ya dönmeli (arayüz doğru bağlanmış mı)."""
    p = client.get("/api/quant/options?s=100&k=100&t=0.5&r=0.03&sigma=0.3&kind=c").json()
    prem = p["now"]["price"]
    back = client.get("/api/quant/options?s=100&k=100&t=0.5&r=0.03&sigma=0.9"
                      f"&kind=c&premium={prem}").json()
    assert abs(back["now"]["implied_vol_from_premium"] - 0.30) < 1e-3


def test_smile_svi_ve_rnd(fake_provider):
    d = client.get("/api/quant/smile?symbol=AAA").json()
    assert len(d["market"]["k"]) == len(d["market"]["iv"]) >= 5
    assert d["svi"] is not None and d["svi_curve"] is not None
    assert len(d["svi_curve"]["k"]) == len(d["svi_curve"]["iv"])
    # Temiz (arbitrajsız üretilmiş) bir yüzeyde Durrleman g NEGATİFE düşmemeli.
    assert d["durrleman"]["butterfly_free"] is True
    assert d["rnd"]["n_negative"] == 0
    assert d["metrics"]["atm_vol"] > 0


def test_portfolio_agirlik_ve_risk_katkisi(fake_provider):
    d = client.get("/api/quant/portfolio?symbols=AAA,BBB,CCC&interval=1d&lookback=250").json()
    n = len(d["symbols"])
    assert n == 3 and d["methods"]
    for name, m in d["methods"].items():
        assert len(m["weights"]) == len(m["risk_contrib"]) == n, name
        assert abs(sum(m["risk_contrib"]) - 1.0) < 1e-3, f"{name}: risk katkısı toplamı 1 değil"
    assert len(d["correlation"]) == n
    for i in range(n):
        assert abs(d["correlation"][i][i] - 1.0) < 1e-6      # köşegen tam 1
    # Tolerans yuvarlamayı hesaba katar: uç `_f(v, 4)` ile basar, bileşen başına en çok
    # 5e-5 YUKARI yuvarlanabilir → 3 bileşende 1,5e-4. 1e-6'lık eşik matematiği değil
    # serileştirme hassasiyetini sınıyordu (ölçülen sapma: 1,0001).
    assert d["pca"] is not None
    assert sum(d["pca"]["explained"]) <= 1.0 + 5e-5 * len(d["pca"]["explained"])
    fr = d["frontier"]
    assert len(fr["vol"]) == len(fr["ret"]) > 0


def test_micro_olcum_yoklugu_uydurmaz(fake_provider):
    """Makas tahmincisi yansız değerde pozitif değilse `spread` None KALMALI —
    kırpılmış pozitif ortalama basmak ölçüm YOKLUĞUNU ölçüm gibi sunardı
    (ACI DERSLER · BULGU-33-4). Anahtar DÜŞÜRÜLMEZ, None taşınır."""
    d = client.get("/api/quant/micro?symbol=AAA&interval=1h&lookback=500").json()
    for key in ("roll_spread", "corwin_schultz"):
        o = d[key]
        if o is None:
            continue
        assert "spread" in o, f"{key}: 'spread' anahtarı düşürülmüş"
        if o.get("spread_raw") is not None and o["spread_raw"] <= 0:
            assert o["spread"] is None, f"{key}: ölçülemeyen makas sayı olarak basıldı"
    sig = d["signature"]
    assert sig is not None and len(sig["scales"]) == len(sig["rv_per_bar"])


def test_execution_yorunge_azalan_ve_twap_referansli():
    d = client.get("/api/quant/execution?q=10000&sigma=0.02&eta=1e-6"
                   "&gamma_risk=1e-5&t=1&n=20").json()
    tr = d["trajectory"]
    assert len(tr) == 21 and tr[0] == pytest.approx(10000, rel=1e-6)
    assert tr[-1] == pytest.approx(0.0, abs=1e-6)
    assert all(tr[i] >= tr[i + 1] - 1e-6 for i in range(len(tr) - 1))   # monoton azalan
    # Risk kaçınması VARSA yörünge TWAP'ın ALTINDA (öne yüklü) olmalı.
    assert any(a < b - 1e-9 for a, b in zip(tr, d["twap"], strict=True))
    assert len(d["sweep"]["gamma"]) == len(d["sweep"]["impact"]) > 5


def test_kelly_egrisi_tepe_noktasi_tam_kellyde():
    d = client.get("/api/quant/kelly?p_win=0.6&win_loss=1.0&fraction=0.25").json()
    assert d["kelly_full"] == pytest.approx(0.2, abs=1e-6)      # 2p−1
    assert d["kelly_fractional"] <= d["kelly_full"]
    g, f = d["growth"], d["f_grid"]
    peak = f[max(range(len(g)), key=lambda i: (g[i] if g[i] is not None else -9e9))]
    assert abs(peak - d["kelly_full"]) < 0.02                   # tepe = tam Kelly
    assert d["breakeven_p"] == pytest.approx(0.5, abs=1e-6)


def test_kelly_kenar_yoksa_boyut_sifir_ama_isaret_gorunur():
    """p başabaşın altındaysa boyut SIFIRDIR — `kelly_binary` bilinçli olarak negatif
    pozisyon ÖNERMEZ (sinyal-only sistemde ters bahis bir yön iddiasıdır). Ama sıfır
    tek başına "kenar yok" ile "tam başabaş"ı ayırmaz: `edge` işareti TAŞINMALI ki
    pano ikisini ayrı gösterebilsin (ölçüm yokluğu ≠ nötr ölçüm)."""
    d = client.get("/api/quant/kelly?p_win=0.30&win_loss=1.0").json()
    assert d["kelly_full"] == 0.0 and d["kelly_fractional"] == 0.0
    assert d["edge"] < 0, "negatif kenar işareti kayboldu"
    assert d["inputs"]["p_win"] < d["breakeven_p"]

    pos = client.get("/api/quant/kelly?p_win=0.70&win_loss=1.0").json()
    assert pos["kelly_full"] > 0 and pos["edge"] > 0


# ---------------------------------------------------------------------------
# 3) PANO BAĞLANTISI — yazılıp çağrılmayan uç bu depoda kusurdur
# ---------------------------------------------------------------------------
def test_katalog_her_model_icin_panoda_panel_var():
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    models = client.get("/api/quant/catalog").json()["models"]
    assert len(models) >= 10
    for m in models:
        mid = m["id"]
        assert f'id="qm-p-{mid}"' in html, f"{mid}: panel yok"
        assert f'id="qm-head-{mid}"' in html, f"{mid}: başlık kancası yok"
        assert f'id="qm-s-{mid}"' in html, f"{mid}: ölçüm şeridi yok"
        assert f'id="qm-f-{mid}"' in html, f"{mid}: formül referansı kancası yok"
        # "Hesapla" düğmesi katalogdan ÜRETİLİR (qmRun('${id}') şablonu); bağlantının
        # kanıtı düz metin değil, modelin gönderici + çizici tablolarında OLMASIDIR.
        assert f"'{mid}'" in html, f"{mid}: pano JS'inde hiç anılmıyor"

    catalog_ids = {m["id"] for m in models}
    listed = set(re.search(r"const QM_MODELS = \[(.*?)\];", html, re.S).group(1).split("'")[1::2])
    rendered = set(re.findall(r"(\w+): qmRender\w+", re.search(
        r"const QM_RENDER = \{(.*?)\};", html, re.S).group(1)))
    assert catalog_ids == listed, f"gönderici tablosu ayrık: {catalog_ids ^ listed}"
    assert catalog_ids == rendered, f"çizici tablosu ayrık: {catalog_ids ^ rendered}"
    assert "qmRun('${id}')" in html, "katalogdan üretilen Hesapla düğmesi kayıp"


def test_pano_sekmesi_ve_tembel_yukleme_bagli():
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    assert "switchTab('tab-quant'" in html          # sekme düğmesi
    assert 'id="tab-quant"' in html                 # sekme içeriği
    assert "if (tabId === 'tab-quant') loadQuantCatalog();" in html


def test_katalog_metinleri_operatore_konusuyor():
    """Her model NE anlattığını ve NASIL OKUNDUĞUNU söylemeli — grafik tek başına
    'anlaşılır' değildir; kartın yanındaki cümle modelin parçasıdır."""
    for m in client.get("/api/quant/catalog").json()["models"]:
        assert len(m["soru"]) > 10, m["id"]
        assert len(m["anlatim"]) > 150, m["id"]
        assert len(m.get("oku", "")) > 40, f"{m['id']}: 'nasıl okunur' notu yok"
        assert m["formul"], m["id"]


def test_formul_referanslari_GERCEK_fonksiyonlara_isaret_ediyor():
    """`knowledge.py` disiplininin buradaki karşılığı: uydurma fonksiyon adı YOK.
    Katalogdaki her `modul.fonksiyon` referansı gerçekten içe aktarılabilmeli."""
    import importlib

    prefixes = {"quant_math": "nihai.ai.quant_math", "options_math": "nihai.ai.options_math",
                "advanced_math": "nihai.ai.advanced_math", "microstructure": "nihai.ai.microstructure",
                "portfolio": "nihai.ai.portfolio", "regime_hmm": "nihai.ai.regime_hmm",
                "option_flow": "nihai.ai.option_flow",
                "statarb": "nihai.ai.statarb",
                "fred_data": "nihai.data.fred_data",   # Tur-73: eğri modeli VERİ modülünü de anar
                "cboe_data": "nihai.data.cboe_data",   # Tur-74: rejim modeli vade yapısını anar
                "macro_regime": "nihai.data.macro_regime",   # Tur-76: makro paneli
                "edgar_facts": "nihai.data.edgar_facts",     # Tur-76: SEC XBRL köprüsü
                "fundamentals": "nihai.ai.fundamentals",     # Tur-76: temel skorlar
                "factors": "nihai.data.factors",             # Tur-76: Ken French faktörleri
                "drift_monitor": "nihai.ai.drift_monitor",   # Tur-79: SR 11-7 sürüklenme
                "econometrics": "nihai.ai.econometrics",     # Tur-79: DCC/FIGARCH/MODWT
                "tda": "nihai.ai.tda",                       # Tur-79: kalıcı homoloji
                "etf_holdings": "nihai.data.etf_holdings",   # Tur-79: SPDR sepeti
                "risk.prob_sizing": "nihai.risk.prob_sizing",
                "risk.options_analyzer": "nihai.risk.options_analyzer"}
    checked = 0
    for m in client.get("/api/quant/catalog").json()["models"]:
        for ref in m["formul"]:
            mod_name, _, fn = ref.rpartition(".")
            assert mod_name in prefixes, f"{ref}: bilinmeyen modül öneki"
            mod = importlib.import_module(prefixes[mod_name])
            assert hasattr(mod, fn), f"{ref}: {prefixes[mod_name]} içinde YOK"
            checked += 1
    assert checked >= 60, f"yalnız {checked} referans denetlendi"


def test_pano_grafik_kanvaslari_render_fonksiyonlariyla_eslesiyor():
    """HTML'de tanımlı her `qmc-*` kanvası JS'te ÇİZİLİYOR olmalı ve tersi de:
    boş kalan kanvas panoda sessiz bir delik bırakır."""
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    declared = set(re.findall(r'<canvas id="(qmc-[\w-]+)"', html))
    drawn = set(re.findall(r"qmChart\('(qmc-[\w-]+)'", html))
    assert declared, "hiç quant kanvası tanımlanmamış"
    assert declared == drawn, (f"çizilmeyen: {sorted(declared - drawn)} · "
                              f"kanvasi olmayan: {sorted(drawn - declared)}")


def test_api_quant_router_kayitli():
    """Router include edilmemişse uçların hepsi 404 olurdu — sessiz kopuş."""
    paths = {r.path for r in app.routes}
    for p in ("/api/quant/catalog", "/api/quant/risk", "/api/quant/vol",
              "/api/quant/regime", "/api/quant/mc", "/api/quant/options",
              "/api/quant/smile", "/api/quant/portfolio", "/api/quant/micro",
              "/api/quant/execution", "/api/quant/kelly", "/api/quant/statarb"):
        assert p in paths, f"{p} kayıtlı değil"
