"""test_tur57_akis_ve_veri.py — opsiyon akışı ailesi + veri katmanı köprüleri.

Üç soruyu birden sorar (Tur-37/38/56 test deseni):

  1) **Matematik doğru mu?** GEX/max pain/PCR/vanna analitik olarak kurulabilir
     durumlarda ÖLÇÜLEBİLİR bir cevaba sahiptir: tek-strike zincirde max pain o
     strike'tadır, yalnız call'lu zincirde net GEX POZİTİFTİR, `bs_gamma_vec`
     `quant_math.bs_greeks` ile BİT-DENKTİR (formül iki yerde yaşıyor).
  2) **Ölçüm yokluğu ölçüm gibi raporlanıyor mu?** OI kolonu olmayan zincirde GEX
     `None` döner (0 DEĞİL), hacim kolonu olmayan zincirde UOA `None` döner,
     `trade_count` vermeyen sağlayıcıda işlem büyüklüğü `None` kalır. Bu sözleşme
     Tur-31/33/56'nın devamıdır: uydurulmuş varsayılan basmaktansa boş bırakılır.
  3) **Bağlandı mı?** Yeni veri yolları ÜRETİMDE çağrılıyor mu — `_normalize_alpaca_bars`
     `trade_count`/`vwap`'i KORUYOR mu, `main.py` SHORT dalında borç bayrağı okunuyor
     mu, bayrak okuyucusunun TEK kaynağı var mı (iki kopya = iki farklı zincir gezme).

Ağ YOK: her şey saf fonksiyon, sahte istemci veya kaynak metni üstünde ölçülür.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nihai.ai import microstructure as ms
from nihai.ai import option_flow as of
from nihai.ai.quant_math import bs_greeks

ROOT = Path(__file__).resolve().parents[1]
MAIN_SRC = (ROOT / "nihai" / "execution" / "main.py").read_text(encoding="utf-8")

S, T, R = 100.0, 0.25, 0.02
STRIKES = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
IV = np.array([0.35, 0.32, 0.30, 0.29, 0.30])


def _side(oi=None, vol=None, iv=IV, strikes=STRIKES, gamma=None) -> dict:
    d: dict = {"strike": strikes}
    if oi is not None:
        d["openInterest"] = oi
    if vol is not None:
        d["volume"] = vol
    if iv is not None:
        d["impliedVolatility"] = iv
    if gamma is not None:
        d["gamma"] = gamma
    return d


CALLS = _side(oi=[100, 200, 500, 300, 100], vol=[10, 50, 900, 400, 20])
PUTS = _side(oi=[400, 300, 500, 120, 60], vol=[80, 120, 600, 50, 5],
             iv=np.array([0.38, 0.34, 0.31, 0.30, 0.31]))


# ---------------------------------------------------------------------------
# 1) MATEMATİK — dış referansa karşı
# ---------------------------------------------------------------------------
def test_bs_gamma_vec_skaler_greeklerle_BIT_DENK():
    """Aynı formül `quant_math.bs_greeks` (skaler) ve burada (dizi) yaşıyor. Parite
    testi olmadan biri sessizce kayar — Tur-32'nin greek parite deseni."""
    g = of.bs_gamma_vec(S, STRIKES, T, R, IV)
    ref = np.array([bs_greeks(S, float(k), T, R, float(s))["gamma"]
                    for k, s in zip(STRIKES, IV, strict=True)])
    assert g is not None
    assert np.max(np.abs(g - ref)) == 0.0


def test_gex_isareti_dealer_konvansiyonunu_izler():
    """Yalnız call'lu zincir POZİTİF, yalnız put'lu zincir NEGATİF net GEX verir
    (dealer call'da uzun / put'ta kısa varsayımı). İşaret bir yön iddiası değil bir
    KONVANSİYONDUR ve çıktıda adıyla taşınır."""
    only_c = of.dealer_gamma_exposure(CALLS, None, S, T, R)
    only_p = of.dealer_gamma_exposure(None, PUTS, S, T, R)
    assert only_c is not None and only_c["net_gex"] > 0
    assert only_p is not None and only_p["net_gex"] < 0
    assert only_c["dealer_convention"] == "call_long_put_short"


def test_gex_dolar_olcegi_formulle_uyusuyor():
    """$GEX(K) = Γ·OI·100·S²·0.01 — tek strike'lı zincirde elle hesapla eşit olmalı."""
    one = {"strike": np.array([100.0]), "openInterest": np.array([10.0]),
           "impliedVolatility": np.array([0.30])}
    res = of.dealer_gamma_exposure(one, None, S, T, R)
    gam = bs_greeks(S, 100.0, T, R, 0.30)["gamma"]
    assert res is not None
    assert res["net_gex"] == pytest.approx(gam * 10.0 * 100 * S * S * 0.01, rel=1e-12)


def test_gamma_flip_isaret_degisiminde_ve_net_gex_orada_sifira_yakin():
    """Flip bir eşik değil KÖKTÜR: net GEX o spot'ta ~0 olmalı ve ızgara içinde kalmalı."""
    res = of.dealer_gamma_exposure(CALLS, PUTS, S, T, R, grid=201, span=0.30)
    assert res is not None and res["gamma_flip"] is not None
    flip = res["gamma_flip"]
    assert float(np.min(res["grid_spot"])) <= flip <= float(np.max(res["grid_spot"]))
    at_flip = of.dealer_gamma_exposure(CALLS, PUTS, flip, T, R, grid=11)
    assert at_flip is not None
    olcek = max(abs(res["net_gex"]), 1.0)
    assert abs(at_flip["net_gex"]) < 0.05 * olcek


def test_max_pain_tek_yogunlukta_o_strikeda():
    """Tüm açık pozisyon tek strike'ta ise yazar ödemesi orada TAM SIFIRDIR."""
    c = {"strike": np.array([95.0, 100.0, 105.0]), "openInterest": np.array([0.0, 1000.0, 0.0])}
    p = {"strike": np.array([95.0, 100.0, 105.0]), "openInterest": np.array([0.0, 1000.0, 0.0])}
    mp = of.max_pain(c, p)
    assert mp is not None
    assert mp["max_pain"] == 100.0 and mp["pain_min"] == 0.0


def test_max_pain_odeme_egrisi_disbukey_ve_minimumda():
    mp = of.max_pain(CALLS, PUTS)
    assert mp is not None
    j = int(np.argmin(mp["pain"]))
    assert mp["max_pain"] == mp["strikes"][j]
    assert mp["pain"][0] >= mp["pain"][j] and mp["pain"][-1] >= mp["pain"][j]


def test_put_call_orani_toplamlardan_hesaplanir():
    pcr = of.put_call_ratio(CALLS, PUTS)
    assert pcr is not None
    assert pcr["oi_pcr"] == pytest.approx(1380 / 1200)
    assert pcr["volume_pcr"] == pytest.approx(855 / 1380)
    assert pcr["measured"] == {"oi": True, "volume": True}


def test_uoa_hacim_oi_oranina_gore_sirali():
    u = of.unusual_activity(CALLS, PUTS, min_volume=10, ratio=1.0)
    assert u is not None and u["rows"]
    oranlar = [row["vol_oi"] for row in u["rows"]]
    assert oranlar == sorted(oranlar, reverse=True)
    assert all(row["vol_oi"] >= 1.0 for row in u["rows"])


def test_vanna_charm_options_math_ile_ayni_formul():
    """Dizi hâli, `options_math.bs_greeks_extended` skaler hâliyle eşleşmeli."""
    from nihai.ai.options_math import bs_greeks_extended
    one = {"strike": np.array([105.0]), "openInterest": np.array([1.0]),
           "impliedVolatility": np.array([0.28])}
    res = of.dealer_vanna_charm(one, None, S, T, R, contract_size=1)
    ext = bs_greeks_extended(S, 105.0, T, R, 0.28, "c")
    assert res is not None and ext is not None
    assert res["vanna_exposure"] == pytest.approx(ext["vanna"], rel=1e-12)
    assert res["charm_exposure"] == pytest.approx(ext["charm"], rel=1e-12)


# ---------------------------------------------------------------------------
# 2) ÖLÇÜM YOKLUĞU ÖLÇÜM DEĞİLDİR
# ---------------------------------------------------------------------------
def test_oi_kolonu_yoksa_gex_ve_max_pain_None():
    """DoltHub önbelleğinde OI kolonu YOKTUR (Tur-54). Sıfır varsaymak, olmayan bir
    ölçümü 'açık pozisyon yok' diye raporlamak olurdu."""
    yalin = {"strike": STRIKES, "impliedVolatility": IV}
    assert of.dealer_gamma_exposure(yalin, yalin, S, T, R) is None
    assert of.max_pain(yalin, yalin) is None


def test_hacim_kolonu_yoksa_uoa_None_ve_pcr_kismi():
    """Alpaca OPRA anlık zincirinde HACİM yoktur; OI vardır. UOA ölçülemez ama
    OI-PCR ölçülebilir — kısmi ölçüm kısmi raporlanır."""
    c, p = _side(oi=[1, 2, 3, 4, 5]), _side(oi=[5, 4, 3, 2, 1])
    assert of.unusual_activity(c, p) is None
    pcr = of.put_call_ratio(c, p)
    assert pcr is not None and pcr["oi_pcr"] is not None
    assert pcr["volume_pcr"] is None and pcr["measured"]["volume"] is False


def test_gamma_zincirden_gelirse_kaynak_adiyla_taşinir():
    """IV yoksa hazır gamma kolonu kullanılır ve KAYNAK çıktıda yazar (Tur-34:
    ölçüm koşulu ölçümle birlikte taşınır)."""
    c = {"strike": STRIKES, "openInterest": np.full(5, 100.0), "gamma": np.full(5, 0.03)}
    res = of.dealer_gamma_exposure(c, None, S, T, R)
    assert res is not None and res["gamma_source"] == "zincir"
    res2 = of.dealer_gamma_exposure(CALLS, None, S, T, R)
    assert res2 is not None and res2["gamma_source"] == "iv_turetildi"


@pytest.mark.parametrize("bozuk", [None, {}, {"strike": []}, {"strike": [0.0, -1.0]},
                                   {"strike": [np.nan, np.nan]}])
def test_akis_fonksiyonlari_istisna_atmaz(bozuk):
    """Savunmacı sözleşme `quant_math`/`options_math` ile aynı: hiçbir girdi patlamaz."""
    for fn in (of.dealer_gamma_exposure, of.put_call_ratio, of.max_pain,
               of.unusual_activity):
        if fn in (of.dealer_gamma_exposure,):
            assert fn(bozuk, bozuk, S, T, R) is None
        elif fn is of.unusual_activity:
            assert fn(bozuk, bozuk) is None
        else:
            assert fn(bozuk, bozuk) is None
    assert of.dealer_vanna_charm(bozuk, bozuk, S, T, R) is None
    assert of.dealer_gamma_exposure(CALLS, PUTS, -1.0, T, R) is None
    assert of.dealer_gamma_exposure(CALLS, PUTS, S, 0.0, R) is None


# ---------------------------------------------------------------------------
# 3) BAR AKIŞI ÖLÇÜLERİ
# ---------------------------------------------------------------------------
def test_rvol_yalniz_AYNI_dilimle_kiyaslar():
    """Açılış barı öğle barıyla kıyaslanamaz — kıyas tabanı aynı gün-içi dilimdir."""
    slots = np.tile(np.arange(4), 6)
    vol = np.tile(np.array([1000.0, 100.0, 100.0, 100.0]), 6)
    vol[-4] = 3000.0                      # son 'dilim 0' barı 3 katı
    v = np.roll(vol, 3)                   # son bar dilim 0 olsun
    s = np.roll(slots, 3)
    r = ms.relative_volume(v, s)
    assert r is not None and r["slot"] == 0
    # Kıyas TABANI aynı dilimin medyanıdır (1000), düz ortalama (325) DEĞİL: düz
    # ortalamaya bölünseydi RVOL ~9,2 çıkar ve gün-içi mevsimsellik "patlama" sanılırdı.
    assert r["rvol"] == pytest.approx(3.0)
    assert r["slot_median"] == pytest.approx(1000.0)
    assert r["n_history"] == 5


def test_rvol_yeterli_gecmis_yoksa_None():
    assert ms.relative_volume([100.0, 200.0], [0, 1]) is None
    assert ms.relative_volume([1, 2, 3], [0, 0, 0], min_history=5) is None


def test_islem_buyuklugu_trade_count_olmadan_olculemez():
    """yfinance `trade_count` VERMEZ → ölçü None. Hacmi işlem sayısı sanmak yanlış
    bir sayı üretirdi."""
    vol = np.full(40, 1000.0)
    assert ms.trade_size_profile(vol, np.zeros(40)) is None      # trade_count YOK
    rng = np.random.default_rng(7)
    tc = 10.0 + rng.normal(0, 0.4, 40)              # geçmişte KÜÇÜK oynama
    tc[-1] = 1.0                                    # son barda dev blok
    prof = ms.trade_size_profile(vol, tc)
    assert prof is not None and prof["avg_trade_size"] == pytest.approx(1000.0)
    assert prof["z"] is not None and prof["z"] > 2.0 and prof["whale"] is True


def test_turnover_float_olmadan_None():
    assert ms.turnover_rate(1e6, 5e7) == pytest.approx(0.02)
    assert ms.turnover_rate(1e6, 0) is None
    assert ms.turnover_rate("x", 1) is None
    assert ms.turnover_rate(float("nan"), 1) is None


def test_vwap_sapmasi_bps_ve_None_sozlesmesi():
    close = np.full(30, 101.0)
    vwap = np.full(30, 100.0)
    d = ms.vwap_deviation(close, vwap)
    assert d is not None and d["dev_bps"] == pytest.approx(100.0)
    assert ms.vwap_deviation(close, np.full(30, np.nan)) is None


# ---------------------------------------------------------------------------
# 4) VERİ KATMANI — Alpaca köprüleri
# ---------------------------------------------------------------------------
def _raw_bars(with_extra: bool) -> pd.DataFrame:
    idx = pd.date_range("2026-08-20", periods=6, freq="D", tz="UTC")
    d = {"open": np.arange(6.0) + 100, "high": np.arange(6.0) + 101,
         "low": np.arange(6.0) + 99, "close": np.arange(6.0) + 100.5,
         "volume": np.full(6, 1e6)}
    if with_extra:
        d["trade_count"] = np.full(6, 5000.0)
        d["vwap"] = np.arange(6.0) + 100.4
    return pd.DataFrame(d, index=idx)


def test_normalize_bars_trade_count_ve_vwap_KORUR():
    """Tur-57'nin veri tarafındaki kök bulgusu: kolonlar geliyordu ve normalize
    adımında DÜŞÜRÜLÜYORDU — üç ölçü bu yüzden 'ölçülemez' sanılıyordu."""
    from nihai.data.alpaca_provider import _normalize_alpaca_bars
    out = _normalize_alpaca_bars(_raw_bars(True), "AAPL", "1d", 10)
    assert {"trade_count", "vwap"} <= set(out.columns)
    assert list(out.columns)[:5] == ["open", "high", "low", "close", "volume"]


def test_normalize_bars_ek_kolon_yoksa_da_calisir():
    from nihai.data.alpaca_provider import _normalize_alpaca_bars
    out = _normalize_alpaca_bars(_raw_bars(False), "AAPL", "1d", 10)
    assert not out.empty and "trade_count" not in out.columns


def test_zincir_zenginlestirme_yalniz_GECERLI_degerleri_yazar():
    """Zenginleştirme veriyi yalnız İYİLEŞTİREBİLİR: sıfır/None/negatif Alpaca değeri
    yfinance satırını BOZAMAZ."""
    from nihai.data.alpaca_provider import enrich_chain_side
    df = pd.DataFrame({"contractSymbol": ["A1", "A2"], "strike": [100.0, 105.0],
                       "bid": [1.0, 2.0], "ask": [1.2, 2.4],
                       "impliedVolatility": [0.30, 0.31],
                       "openInterest": [100, 200], "volume": [10, 20]})
    snaps = {"A1": {"latest_quote": {"bid_price": 1.05, "ask_price": 1.10},
                    "implied_volatility": 0.285,
                    "greeks": {"delta": 0.55, "gamma": 0.04, "theta": -0.02, "vega": 0.11}},
             "A2": {"latest_quote": {"bid_price": 0.0, "ask_price": 0.0},
                    "implied_volatility": None}}
    out, n = enrich_chain_side(df, snaps)
    assert n == 1
    assert out.loc[0, "bid"] == 1.05 and out.loc[0, "impliedVolatility"] == 0.285
    assert out.loc[0, "gamma"] == 0.04
    assert out.loc[1, "bid"] == 2.0 and out.loc[1, "impliedVolatility"] == 0.31


def test_zincir_zenginlestirme_eslesme_yoksa_dokunmaz():
    from nihai.data.alpaca_provider import enrich_chain_side
    df = pd.DataFrame({"strike": [100.0], "bid": [1.0]})       # contractSymbol YOK
    out, n = enrich_chain_side(df, {"X": {"implied_volatility": 0.9}})
    assert n == 0 and out.equals(df)
    df2 = pd.DataFrame({"contractSymbol": ["Z"], "bid": [1.0]})
    out2, n2 = enrich_chain_side(df2, {})
    assert n2 == 0 and out2 is df2


def test_opra_zenginlestirme_varsayilan_KAPALI(monkeypatch):
    """Ek ağ çağrısı tarama döngüsüne gecikme ekler → varsayılan davranış DEĞİŞMEZ."""
    from nihai.data.alpaca_provider import _alpaca_options_enabled
    monkeypatch.delenv("ALPACA_OPTIONS", raising=False)
    assert _alpaca_options_enabled() is False
    for v in ("0", "off", "false", "hayır", ""):
        monkeypatch.setenv("ALPACA_OPTIONS", v)
        assert _alpaca_options_enabled() is False
    monkeypatch.setenv("ALPACA_OPTIONS", "1")
    assert _alpaca_options_enabled() is True


def test_asset_flags_gun_bazinda_onbelleklenir(monkeypatch):
    """Bayraklar gün içinde değişebilir → önbellek GÜN anahtarlı; aynı gün ikinci
    çağrı ağa ÇIKMAZ."""
    from nihai.data.alpaca_provider import AlpacaProvider

    class _Asset:
        shortable, easy_to_borrow, marginable, tradable = True, False, True, True

    class _TC:
        def __init__(self):
            self.n = 0

        def get_asset(self, sym):
            self.n += 1
            return _Asset()

    p = AlpacaProvider(api_key="k", secret_key="s")
    tc = _TC()
    monkeypatch.setattr(p, "_get_trading_client", lambda: tc)
    a = p.asset_flags("aapl")
    b = p.asset_flags("AAPL")
    assert a is not None and a["symbol"] == "AAPL" and a["easy_to_borrow"] is False
    assert b == a and tc.n == 1


def test_asset_flags_anahtar_yoksa_None(monkeypatch):
    """Boş argüman env'e DÜŞER (`api_key or os.getenv(...)`) — anahtarı olan bir
    geliştirme makinesinde test sessizce ağa çıkardı; env de temizlenmeli."""
    from nihai.data.alpaca_provider import AlpacaProvider
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    assert AlpacaProvider(api_key="", secret_key="").asset_flags("AAPL") is None


# ---------------------------------------------------------------------------
# 5) BAYRAK OKUYUCU — TEK kaynak, zinciri gezer
# ---------------------------------------------------------------------------
class _Bayrakli:
    name = "sahte"

    def asset_flags(self, symbol):
        return {"symbol": symbol, "shortable": False, "easy_to_borrow": False}


class _Sarmal:
    name = "sarmal"

    def __init__(self, inner):
        self.inner = inner


def test_flag_okuyucu_sarmalayici_zincirini_geziyor():
    from nihai.data.data_provider import asset_flags
    zincir = _Sarmal(_Sarmal(_Bayrakli()))
    f = asset_flags("AAPL", zincir)
    assert f is not None and f["shortable"] is False


def test_flag_okuyucu_sagliyici_patlarsa_None():
    from nihai.data.data_provider import asset_flags

    class _Patlar:
        def asset_flags(self, symbol):
            raise RuntimeError("kopuk")

    assert asset_flags("AAPL", _Patlar()) is None


def test_flag_okuyucu_TEK_kopya():
    """İki çağıran (main.py + api_quant) AYNI okuyucuyu kullanmalı; ikinci bir kopya
    aynı zinciri iki farklı biçimde gezme riskini doğurur (Tur-35 dersi)."""
    api = (ROOT / "nihai" / "web" / "api_quant.py").read_text(encoding="utf-8")
    assert "from nihai.data.data_provider import asset_flags" in api
    assert "from nihai.data.data_provider import asset_flags" in MAIN_SRC
    assert MAIN_SRC.count("def asset_flags") == 0


# ---------------------------------------------------------------------------
# 6) SHORT yolu — Madde 30'un ikinci yarısı
# ---------------------------------------------------------------------------
def _short_blok() -> str:
    i = MAIN_SRC.index("Madde 30 — SSR farkındalığı")
    j = MAIN_SRC.index("Madde 53 — Wash-sale farkındalığı")
    return MAIN_SRC[i:j]


def test_borc_bayragi_SHORT_dalinda_okunuyor():
    blok = _short_blok()
    assert "asset_flags" in blok, "SHORT dalında borç bayrağı hiç okunmuyor"
    assert "NOT_SHORTABLE" in blok and "HARD_TO_BORROW" in blok


def test_borc_bayragi_varsayilan_BLOKE_ETMEZ():
    """SSR ile aynı sözleşme: işaretler, bloke etmez. Bloke etmek yalnız operatörün
    açık tercihiyle (`SHORT_BORROW_GUARD=block`) olur."""
    blok = _short_blok()
    assert 'SHORT_BORROW_GUARD' in blok and '"block"' in blok
    ret_idx = blok.index("return", blok.index("SHORT_BORROW_GUARD"))
    guard_idx = blok.index("SHORT_BORROW_GUARD")
    assert guard_idx < ret_idx, "return, kip kontrolünden ÖNCE geliyor"


def test_bilinmeyen_bayrak_RED_gerekcesi_degil():
    """`flags is None` = 'okunamadı'. Onu 'satılamaz' saymak, ölçüm yokluğunu ölçüm
    gibi kullanmak olurdu (Tur-31)."""
    blok = _short_blok()
    assert "flags is not None" in blok


# ---------------------------------------------------------------------------
# 7) VERİ YETENEK ÖLÇÜMÜ
# ---------------------------------------------------------------------------
def test_yetenek_olcumu_agsiz_calisir():
    """`--offline` yolu ağ çağrısı YAPMAZ (test guard'ı zaten engellerdi) ama yine de
    anahtar envanterini ve yürürlükteki ayarı döndürür."""
    from nihai.data.data_capabilities import probe, rapor
    d = probe(network=False)
    assert set(d) >= {"olcum_zamani", "anahtarlar", "ayar", "yetenekler"}
    assert "ALPACA_FEED" in d["ayar"] and "ALPACA_BARS_FEED" in d["ayar"]
    assert all(isinstance(v, bool) for v in d["anahtarlar"].values())
    metin = rapor(d)
    assert "VERİ YETENEK ÖLÇÜMÜ" in metin and "KENAR iddiası değildir" in metin


def test_yetenek_raporu_anahtar_DEGERI_basmaz(monkeypatch):
    """Rapor operatöre gösterilir ve loglanabilir — sır değeri asla yazılmamalı."""
    from nihai.data.data_capabilities import probe, rapor
    monkeypatch.setenv("ALPACA_API_KEY", "GIZLI_ANAHTAR_123")
    metin = rapor(probe(network=False))
    assert "GIZLI_ANAHTAR_123" not in metin


def test_yetenek_modulu_utf8_stdout_kuruyor():
    """Tur-24 kuralı: her yeni `__main__` giriş noktası cp1254 çökmesine karşı korunur."""
    src = (ROOT / "nihai" / "data" / "data_capabilities.py").read_text(encoding="utf-8")
    assert "ensure_utf8_stdout()" in src and '__name__ == "__main__"' in src


# ---------------------------------------------------------------------------
# 8) PANO KÖPRÜSÜ
# ---------------------------------------------------------------------------
def test_flow_ucu_katalogda_ve_panoda():
    from fastapi.testclient import TestClient

    from nihai.web.web_dashboard import app
    c = TestClient(app)
    ids = {m["id"] for m in c.get("/api/quant/catalog").json()["models"]}
    assert "flow" in ids
    assert "/api/quant/flow" in {r.path for r in app.routes}
    html = (ROOT / "nihai" / "web" / "web_dashboard.html").read_text(encoding="utf-8")
    assert 'id="qm-p-flow"' in html and "flow: qmRenderFlow" in html
    kanvas = set(re.findall(r'<canvas id="(qmc-flow-[\w-]+)"', html))
    cizilen = set(re.findall(r"qmChart\('(qmc-flow-[\w-]+)'", html))
    assert kanvas and kanvas == cizilen


def test_flow_ucu_saglayici_olunce_500_ATMAZ(monkeypatch):
    from fastapi.testclient import TestClient

    import nihai.web.api_quant as aq
    from nihai.web.web_dashboard import app

    class _Olu:
        def get_bars(self, *a, **k):
            raise RuntimeError("yok")

        def get_option_chain(self, *a, **k):
            raise RuntimeError("yok")

    monkeypatch.setattr(aq, "_provider", lambda: _Olu())
    monkeypatch.setattr(aq, "_asset_flags", lambda s: None)
    r = TestClient(app).get("/api/quant/flow?symbol=SPY&interval=1d&lookback=300")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["reason"]
