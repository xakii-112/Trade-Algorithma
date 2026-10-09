"""Tur-79 — yetim kalan dört ölçüm modülünün PANO KÖPRÜSÜ.

`drift_monitor` · `econometrics` · `tda` · `etf_holdings` yazılmış, test edilmiş ve
ÜRETİM ÇAĞIRANI OLMADAN bırakılmıştı (Tur-38/61 kuralı: yazılıp OKUNMAYAN yüzey bu
depoda kusurdur). Bu dosya köprüyü üç soruyla kilitler:

  1. FAIL-SOFT mu — sağlayıcı/veri tamamen yokken uç 500 atmıyor, `reason` taşıyor.
  2. SÖZLEŞME tutuyor mu — ölçüm yokluğu `None` olarak KORUNUYOR, nedensellik bayrağı
     taşınıyor, popülasyon karnesi süzgeçten BAĞIMSIZ raporlanıyor.
  3. BAĞLANDI mı — katalog · pano paneli · gönderici · çizici dört tabloda da var.

Ağ YOK: uçların hepsi önbellek-yalnız (`fetch=False`) çağrılır.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import nihai.web.api_quant as aq
from nihai.web.api_quant import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)

DASHBOARD_HTML = Path(__file__).resolve().parents[1] / "nihai" / "web" / "web_dashboard.html"
YENI = ("drift", "econometrics", "tda", "dispersion")


# ---------------------------------------------------------------------------
# Sentetik sağlayıcı — deterministik, AĞ YOK (conftest ağ guard'ı zorunlu kılar).
# Semboller ORTAK bir faktörü paylaşır: DCC'nin ölçtüğü şey korelasyondur, bağımsız
# serilerde R≈I çıkar ve uç "çalıştı" görünürken hiçbir şey ölçmemiş olurdu.
# ---------------------------------------------------------------------------
class _FakeProvider:
    def get_bars(self, symbol, interval, lookback=200):
        n = min(int(lookback), 1000)
        rng = np.random.default_rng(abs(hash(symbol)) % 9973)
        ortak = np.random.default_rng(11).normal(0.0002, 0.010, n)   # TÜM sembollerde AYNI
        r = 0.7 * ortak + 0.3 * rng.normal(0.0003, 0.012, n)
        close = 100.0 * np.exp(np.cumsum(r))
        op = close * (1 + rng.normal(0, 0.002, n))
        return pd.DataFrame(
            {"open": op,
             "high": np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.004, n))),
             "low": np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.004, n))),
             "close": close,
             "volume": rng.integers(1e5, 5e6, n).astype(float)},
            index=pd.date_range("2022-01-03", periods=n, freq="B"))


@pytest.fixture
def fake_provider(monkeypatch):
    monkeypatch.setattr(aq, "_provider", lambda: _FakeProvider())
    monkeypatch.setattr("nihai.execution.watchlist.current",
                        lambda: ["AAA", "BBB", "CCC", "DDD", "EEE"])


@pytest.fixture
def olu_provider(monkeypatch):
    """Sağlayıcı TAMAMEN ölü: fail-soft yolun gerçek sınavı."""
    def _patla():
        raise RuntimeError("saglayici yok")
    monkeypatch.setattr(aq, "_provider", _patla)
    monkeypatch.setattr("nihai.execution.watchlist.current", lambda: [])


# ---------------------------------------------------------------------------
# 1) BAĞLANDI MI — dört tablo birden
# ---------------------------------------------------------------------------
def test_dort_model_KATALOGDA_ve_PANODA():
    from nihai.web.api_quant import _CATALOG

    idler = {m["id"] for m in _CATALOG}
    assert set(YENI) <= idler, f"katalogda eksik: {set(YENI) - idler}"

    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    for mid in YENI:
        for kanca in (f'id="qm-p-{mid}"', f'id="qm-head-{mid}"',
                      f'id="qm-s-{mid}"', f'id="qm-f-{mid}"'):
            assert kanca in html, f"{mid}: {kanca} yok"
    listed = set(re.search(r"const QM_MODELS = \[(.*?)\];", html, re.S).group(1).split("'")[1::2])
    rendered = set(re.findall(r"(\w+): qmRender\w+", re.search(
        r"const QM_RENDER = \{(.*?)\};", html, re.S).group(1)))
    assert set(YENI) <= listed and set(YENI) <= rendered
    # Gönderici dalı: her model kendi URL'ini kurmalı (aksi halde panel BOŞ açılır)
    for mid in YENI:
        assert f"quant/{mid}?" in html, f"{mid}: gönderici dalı yok"


def test_KARAR_YOLU_bu_modulleri_IMPORT_ETMIYOR():
    """ANAYASA: ölçüm yüzeyleri `signals`/`risk`/`execution`'a sızamaz (ACI DERS #5)."""
    kok = Path(__file__).resolve().parents[1] / "nihai"
    for alt in ("signals", "risk", "execution"):
        for p in (kok / alt).rglob("*.py"):
            metin = p.read_text(encoding="utf-8")
            for ad in ("drift_monitor", "ai.econometrics", "ai import econometrics",
                       "ai.tda", "ai import tda", "etf_holdings"):
                assert ad not in metin, f"{p}: {ad} karar yoluna sızmış"


# ---------------------------------------------------------------------------
# 2) FAIL-SOFT — veri yokken 500 YOK
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "/api/quant/drift?interval=YOK_BOYLE_ARALIK",
    "/api/quant/econometrics?symbols=AAA,BBB&symbol=AAA&interval=1d&lookback=300",
    "/api/quant/tda?symbols=AAA,BBB&interval=1d&lookback=200",
    "/api/quant/dispersion?etf=__YOKETF__",
])
def test_veri_yokken_500_ATMIYOR(url: str, olu_provider):
    r = client.get(url)
    assert r.status_code == 200
    d = r.json()
    if d.get("ok") is False:
        assert d.get("reason"), "fail-soft yanıtı NEDEN taşımalı"


def test_drift_log_yokken_YONERGE_veriyor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import nihai.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    d = client.get("/api/quant/drift").json()
    assert d["ok"] is False and d["reason"] == "CANLI_OZELLIK_LOGU_YOK"
    assert "live_predict_service" in d["ipucu"]


# ---------------------------------------------------------------------------
# 3) SÖZLEŞME — sürüklenme
# ---------------------------------------------------------------------------
def _sentetik_log(path: Path, *, n: int = 900, kaydir: float = 0.0,
                  aday_pay_ref: float = 1.0, aday_pay_cur: float = 1.0) -> None:
    """Referans ↔ cari penceresi ayrık, tz'si KARIŞIK bir canlı özellik logu."""
    rng = np.random.default_rng(7)
    son = pd.Timestamp("2026-09-15 16:00:00")
    ref_ts = pd.date_range(son - pd.Timedelta(days=80), periods=n, freq="min")
    cur_ts = pd.date_range(son - pd.Timedelta(days=20), periods=n, freq="min")
    satir = []
    for ts_dizi, kay, pay in ((ref_ts, 0.0, aday_pay_ref), (cur_ts, kaydir, aday_pay_cur)):
        for i, ts in enumerate(ts_dizi):
            aday = i < int(pay * len(ts_dizi))
            # Satırların YARISI tz-aware yazılır: üretimdeki gerçek durum (%37,2)
            bar = (ts.isoformat() if i % 2 else
                   ts.tz_localize("UTC").tz_convert("America/New_York").isoformat())
            satir.append({"logged_at": ts.isoformat(), "symbol": "AAA", "interval": "1m",
                          "bar_ts": bar, "rsi14": 50.0 + kay + rng.normal(0, 5),
                          "atr_pct": 1.0 + rng.normal(0, 0.1),
                          "quality_score": 3.0 if aday else 0.0,
                          "rr_ratio": 2.0 if aday else 0.0,
                          "cand_side": ("LONG" if aday else "")})
    pd.DataFrame(satir).to_csv(path, index=False)


def test_karisik_tz_LOGU_OKUNUYOR_ve_pencereler_AYRIK(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch):
    """Üretim logunun %37'si tz-aware'dir; reddetmek logun TAMAMINI reddetmekti."""
    import nihai.config as cfg

    _sentetik_log(tmp_path / "live_features_log.csv")
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    d = client.get("/api/quant/drift?interval=1m&min_n=100&populasyon=hepsi").json()
    assert d["ok"] is True, d.get("reason")
    p = d["pencere"]
    assert p["n_ref"] > 0 and p["n_cur"] > 0
    assert p["ref_son"] <= p["cur_bas"], "referans cari penceresiyle ÇAKIŞAMAZ"


def test_kaymayan_ozellik_STABIL_kayan_ozellik_BUYUK(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch):
    import nihai.config as cfg

    _sentetik_log(tmp_path / "live_features_log.csv", kaydir=25.0)
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    d = client.get("/api/quant/drift?interval=1m&min_n=100&populasyon=hepsi").json()
    oz = d["ozellikler"]
    assert oz["rsi14"]["durum"] == "BUYUK", "25 puanlık kayma yakalanmadı"
    assert oz["atr_pct"]["durum"] == "STABIL", "kaymayan özellik sürüklenmiş sayıldı"
    assert d["siralama"][0] == "rsi14", "sıralama etki büyüklüğüne göre değil"
    # Wasserstein KAYMANIN kendisini ölçer (std'ye oranlı) → pozitif ve büyük
    assert oz["rsi14"]["w1_std"] > oz["atr_pct"]["w1_std"]


def test_populasyon_KARNESI_suzgecten_BAGIMSIZ(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch):
    """Karne HER seçimde AYNI olmalı: artefaktı gizleyen bir panel kusurdur."""
    import nihai.config as cfg

    _sentetik_log(tmp_path / "live_features_log.csv", aday_pay_ref=1.0, aday_pay_cur=0.3)
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    karneler = []
    for pop in ("hepsi", "aday"):
        d = client.get(f"/api/quant/drift?interval=1m&min_n=100&populasyon={pop}").json()
        assert d["ok"] is True, (pop, d.get("reason"))
        karneler.append(d["populasyon_karnesi"])
        assert d["populasyon"] == pop
    assert karneler[0] == karneler[1], "karne süzgece göre değişiyor"
    assert karneler[0]["ayrac"] == "quality_score>0"
    # Kurgu gereği: referansın tamamı aday, carinin %30'u
    assert karneler[0]["ref_aday"] == karneler[0]["ref_n"]
    assert karneler[0]["cur_aday"] < karneler[0]["cur_n"]


def test_KARISIK_populasyon_SAHTE_surukle_uretir(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch):
    """Bu turun bulgusu: aday oranı değişince `quality_score` kaymamışken BÜYÜK çıkar.

    Özelliğin KENDİSİ iki pencerede de aynı dağılımdadır; değişen tek şey aday-dışı
    satırların payıdır. `populasyon=aday` bunu elemeli.
    """
    import nihai.config as cfg

    _sentetik_log(tmp_path / "live_features_log.csv", aday_pay_ref=1.0, aday_pay_cur=0.3)
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    karisik = client.get("/api/quant/drift?interval=1m&min_n=100&populasyon=hepsi").json()
    homojen = client.get("/api/quant/drift?interval=1m&min_n=100&populasyon=aday").json()
    assert karisik["ozellikler"]["quality_score"]["durum"] == "BUYUK"
    q = homojen["ozellikler"]["quality_score"]
    assert q["yetersiz"] or q["durum"] == "STABIL", "homojen popülasyonda da sürüklenmiş"
    assert homojen["ozet"]["n_buyuk"] <= karisik["ozet"]["n_buyuk"]


# ---------------------------------------------------------------------------
# 4) SÖZLEŞME — ekonometri · TDA · dispersiyon
# ---------------------------------------------------------------------------
def test_econometrics_NEDENSELLIK_bayragi_TASINIYOR(fake_provider):
    """MODWT döngüseldir → `causal: false` OLMADAN canlı özellik sanılabilir."""
    d = client.get("/api/quant/econometrics?symbols=AAA,BBB,CCC&symbol=AAA"
                   "&interval=1d&lookback=900").json()
    assert d["ok"] is True
    if d.get("dalgacik"):
        assert d["dalgacik"]["causal"] is False
        paylar = [v for v in d["dalgacik"]["enerji_payi"] if v is not None]
        assert paylar and abs(sum(paylar) - 1.0) < 0.02, "varyans payları 1'e toplanmıyor"
        assert d["dalgacik"]["olcek"] == sorted(d["dalgacik"]["olcek"])
    if d.get("dcc"):
        c = d["dcc"]
        assert 0.0 <= c["persistence"] <= 1.0 + 1e-9
        n = len(c["names"])
        for i in range(n):                       # köşegen TAM 1
            assert abs(c["R_son"][i][i] - 1.0) < 1e-8
        assert len(c["ort_korelasyon"]) == len(c["dates"])


def test_tda_KAYAN_pencere_NEDENSEL_ve_omur_POZITIF(fake_provider):
    d = client.get("/api/quant/tda?symbols=AAA,BBB,CCC&interval=1d"
                   "&lookback=300&window=30&step=5").json()
    assert d["ok"] is True, d.get("reason")
    if d.get("kayan"):
        assert d["kayan"]["causal"] is True
        assert len(d["kayan"]["dates"]) == len(d["kayan"]["l1"])
    for cubuk in (d["H0"] + d["H1"]):            # ölüm ≥ doğum (ömür negatif olamaz)
        if cubuk["olum"] is not None:
            assert cubuk["olum"] >= cubuk["dogum"] - 1e-12
            assert cubuk["omur"] is not None and cubuk["omur"] >= -1e-12
    assert d["H1"] == sorted(d["H1"], key=lambda b: -(b["omur"] if b["omur"] is not None
                                                      else float("inf")))
    assert "ÖNGÖRMEZ" in d["uyari"], "çöküş öngörüsü iddiası reddedilmiyor"


def test_tda_manzara_ve_normlar_SERILESIYOR(fake_provider):
    """numpy dizisi JSON'a düşerse uç 500 atar (pydantic) — `_nums_derin` kilidi."""
    d = client.get("/api/quant/tda?symbols=AAA,BBB,CCC,DDD&interval=1d"
                   "&lookback=250&window=25&step=5").json()
    assert d["ok"] is True
    if d.get("normlar"):
        for ad in ("l1_seviye", "l2_seviye"):
            assert isinstance(d["normlar"][ad], list)
    if d.get("manzara"):
        assert isinstance(d["manzara"]["lambda"], list)
        assert all(isinstance(r, list) for r in d["manzara"]["lambda"])


def test_dispersion_ONBELLEK_YALNIZ_ve_PIT_acigi_BEYAN_EDILIYOR():
    d = client.get("/api/quant/dispersion?etf=SPY&n_top=50&fetch=false").json()
    if d.get("ok") is False:                     # önbellek yoksa yönergeli reddet
        assert d["reason"] in ("SEPET_ONBELLEKTE_YOK", "ORTUK_KORELASYON_SERISI_BOS")
        assert d.get("ipucu")
        return
    assert d["agirlik_pit_acigi"] is True, "bugünün sepeti geçmişe uygulanıyor, beyan ŞART"
    assert len(d["dates"]) == len(d["rho_imp"])
    gecerli = [v for v in d["rho_imp"] if v is not None]
    assert gecerli, "tek geçerli ρ yok"
    assert all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in gecerli), "ρ [-1,1] dışında"
    if d.get("cor1m"):                           # dış referansla kıyas kurulabiliyorsa
        assert d["cor1m"]["n_ortak"] >= 30
        assert -1.0 <= d["cor1m"]["korelasyon_seviye"] <= 1.0


def test_dispersion_ucu_AGA_CIKMIYOR(monkeypatch: pytest.MonkeyPatch):
    """`fetch=False` sözleşmesi (BULGU-59-8): okuyan da ödemez.

    Fail-soft yolda İSTİSNA ile problamak hiçbir şey kanıtlamaz (Tur-74 dersi) →
    gözlem bir BAYRAKLA yapılır.
    """
    import nihai.net as net

    denendi: list[str] = []

    def _izle():
        denendi.append("shared_client")
        raise AssertionError("bu satıra gelinmemeli")

    monkeypatch.setattr(net, "shared_client", _izle)
    client.get("/api/quant/dispersion?etf=SPY&fetch=false")
    assert not denendi, "fetch=False iken ağ istemcisi kuruldu"


# ---------------------------------------------------------------------------
# 5) §4.1 — Black-Litterman'ın piyasa önseli: ÖNCÜL BAYATLADI
# ---------------------------------------------------------------------------
def _panel_ve_adaylar(adlar: list[str], n: int = 300):
    rng = np.random.default_rng(3)
    ortak = rng.normal(0.0004, 0.010, n)
    closes = {s: 100 * np.exp(np.cumsum(0.8 * ortak + rng.normal(0.0002, 0.006 + 0.004 * i, n)))
              for i, s in enumerate(adlar)}
    panel = pd.DataFrame(closes, index=pd.date_range("2024-01-02", periods=n, freq="B"))
    return panel, [{"symbol": s, "strategy": "panel"} for s in adlar]


def test_market_weights_VARSAYILANI_BIT_DENK():
    """`market_weights=None` eski davranıştır — tek bit değişirse geçmiş ölçümler ölür."""
    from nihai.ai import portfolio as pf

    adlar = ["AAA", "BBB", "CCC", "DDD"]
    panel, cands = _panel_ve_adaylar(adlar)
    a = pf.propose(cands, panel, {"max_weight": 0.35})
    b = pf.propose(cands, panel, {"max_weight": 0.35, "market_weights": None})
    wa = np.array([a["methods"]["black_litterman"]["weights"][s] for s in adlar])
    wb = np.array([b["methods"]["black_litterman"]["weights"][s] for s in adlar])
    assert np.max(np.abs(wa - wb)) == 0.0
    assert a["methods"]["black_litterman"]["market_weights_source"] == "ters_vol_proxy"


def test_gercek_agirlik_SONUCU_DEGISTIRIYOR_ve_KAYNAK_beyan_ediliyor():
    """Proxy bir yaklaşıklık değil FARKLI bir önseldir; fark ölçülebilir olmalı."""
    from nihai.ai import portfolio as pf

    adlar = ["AAA", "BBB", "CCC", "DDD"]
    panel, cands = _panel_ve_adaylar(adlar)
    gercek = [0.50, 0.25, 0.15, 0.10]          # kapitalizasyon: oynaklıkla İLGİSİZ
    a = pf.propose(cands, panel, {"max_weight": 0.60})
    b = pf.propose(cands, panel, {"max_weight": 0.60, "market_weights": gercek})
    mb = b["methods"]["black_litterman"]
    assert mb["market_weights_source"] == "gercek"
    wa = np.array([a["methods"]["black_litterman"]["weights"][s] for s in adlar])
    wb = np.array([mb["weights"][s] for s in adlar])
    assert np.max(np.abs(wa - wb)) > 0.01, "gerçek ağırlık sonucu değiştirmiyor"
    # Önsel 1'e normalize edilir (sepet ALT KÜMESİ 1'e toplanmaz; π ölçeğe duyarlıdır)
    assert abs(sum(mb["market_weights"].values()) - 1.0) < 1e-9


def test_gecersiz_market_weights_SESSIZCE_proxyye_DUSMUYOR():
    """Yanlış bir piyasa önselini 'piyasa dengesi' diye raporlamak proxy'den KÖTÜDÜR."""
    from nihai.ai import portfolio as pf

    adlar = ["AAA", "BBB", "CCC"]
    panel, cands = _panel_ve_adaylar(adlar)
    cov = np.cov(np.diff(np.log(panel.to_numpy()), axis=0), rowvar=False)
    mu = np.zeros(3)
    bounds = (np.zeros(3), np.full(3, 0.6))
    for bozuk in ([0.5, 0.5], [0.5, -0.2, 0.7], [0.0, 0.0, 0.0], [0.5, np.nan, 0.5]):
        with pytest.raises(ValueError):
            pf.compare_methods(adlar, np.diff(np.log(panel.to_numpy()), axis=0), cov, mu,
                               cands, bounds, 1.0, 0.05, 0.25, 0.6, market_weights=bozuk)
    # `propose` seviyesinde ise NOT yazılır ve proxy'ye düşülür (fail-soft rapor katmanı)
    r = pf.propose(cands, panel, {"max_weight": 0.6, "market_weights": [0.5, 0.5]})
    assert any("market_weights" in x for x in r["notes"])
    assert r["methods"]["black_litterman"]["market_weights_source"] == "ters_vol_proxy"


def test_eksik_sembolde_proxyye_dusup_NEDENI_yaziyor():
    from nihai.ai import portfolio as pf

    adlar = ["AAA", "BBB", "CCC"]
    panel, cands = _panel_ve_adaylar(adlar)
    r = pf.propose(cands, panel, {"max_weight": 0.6,
                                  "market_weights": {"AAA": 0.6, "BBB": 0.4}})  # CCC YOK
    assert r["methods"]["black_litterman"]["market_weights_source"] == "ters_vol_proxy"
    assert any("sepette YOK" in x for x in r["notes"])


def test_portfoy_ucu_market_prior_KAYNAGINI_TASIYOR(fake_provider):
    """Ölçüm taşınmıyorsa yok sayılır (Tur-63): 'Black-Litterman' satırının anlamı
    hangi önselle kurulduğuna bağlıdır."""
    d = client.get("/api/quant/portfolio?symbols=AAA,BBB,CCC,DDD&interval=1d"
                   "&lookback=300").json()
    assert d.get("ok") is not False, d.get("reason")
    mp = d["market_prior"]
    assert mp["kaynak"] in ("ters_vol_proxy", "SPY_sepeti")
    assert mp["etf"] == "SPY"
    if mp["kaynak"] == "ters_vol_proxy" and mp["sepet_okundu"]:
        assert mp["n_eksik"] > 0, "proxy'ye düşüldüyse NEDENİ sayılmalı"
    if not mp["sepet_okundu"]:
        assert "etf_holdings --warm" in (mp["ipucu"] or "")
