"""Tur-92 — Hisse Detay tahmin fiyatı + grafik, Quant Beyin'in canlı yola alınması.

Kilitlenen bulgular (ölçümler: docs/TUR92_RAPORU_2026-09-23.md):
  BULGU-92-1  Hisse Detay'ın tek tahmini defterden kalibre 1dk aralığıydı; seçicide 1dk YOKTU → hiçbir
              aralıkta tahmin fiyatı basılmıyordu. → her aralıkta ATR ölçekli ampirik aralık + merkez,
              yürüyen sınavı (yalnız ÇÖZÜLMÜŞ geçmiş) ile; merkez medyan ancak "fiyat değişmez"i yenerse.
  BULGU-92-2  Quant karar motorunun kapı-b'si (fayda) kuyruk cezasını KOŞULSUZ düşüyordu → post=1'de
              bile E<0 (30 sembol×aralıkta gereken post 1,05–1,48): motor hiçbir kanıtla yön üretemezdi.
  BULGU-92-3  Gölge koşucu GÜNLÜK özellikleri 1dk'da eğitilmiş nöral modele veriyordu (max|z| 21–71).
  BULGU-92-4  Grafik: genişlik oluşturma anında bir kez okunuyordu (gizli sekmede 0 px → boş grafik);
              alt grafikler mantıksal indeksle senkronlanırken ısınma boşluğu yüzünden KAYIKTI.
"""
from __future__ import annotations

import io
import tokenize
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nihai.ai import brain, decision, learning

KOK = Path(__file__).resolve().parents[1]


def _kod(yol: str) -> str:
    """Kaynağın YORUM ve DİZGİ içermeyen hâli — regresyon kapısı anlatıya takılmasın (Tur-49)."""
    kaynak = (KOK / yol).read_text(encoding="utf-8")
    return " ".join(t.string for t in tokenize.generate_tokens(io.StringIO(kaynak).readline)
                    if t.type not in (tokenize.COMMENT, tokenize.STRING))


def _utc(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp())


# ============================================================ 1. kapı-b (BULGU-92-2)
def _durum(mean, lo, hi):
    return {"strategies": {"event_breakout": {"n": 500, "post_mean": mean, "lo95": lo, "hi95": hi}}}


def _kapanislar(tohum=7, n=400):
    rng = np.random.default_rng(tohum)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))


@pytest.fixture
def izole_karar(tmp_path, monkeypatch):
    monkeypatch.delenv("AI_DECISION_MODE", raising=False)
    monkeypatch.delenv("AI_UTILITY_LAMBDA", raising=False)
    monkeypatch.setattr(decision, "DECISIONS_FILE", tmp_path / "ai_decisions.jsonl")
    monkeypatch.setattr(decision, "_sources_from_state", lambda: None)
    return tmp_path


def test_kapi_b_guclu_kanitla_gecilebilir(izole_karar, monkeypatch):
    kap = _kapanislar()
    r = decision._returns_from_closes(kap)
    kazanc, _kayip = decision._estimate_gain_loss(r)
    # Eski formülde E(post=1) = kazanç − λ·|CVaR| — gerçekçi getiride NEGATİF: kapı kapalıydı.
    assert kazanc - 0.5 * abs(decision._cvar_penalty(r)) < 0
    monkeypatch.setattr(learning, "load_state", lambda: _durum(0.9, 0.86, 0.94))
    d = decision.decide("AAPL", "event_breakout", {}, closes=kap, side_hint="LONG")
    assert d.utility is not None and d.utility > 0
    assert d.action == "LONG_BIAS"


def test_kapi_b_hala_gercek_bir_kapi(izole_karar, monkeypatch):
    """Kapı-a geçse de (lo95 > 0,5) kuyruk riski ılımlı bir posterioru susturur."""
    monkeypatch.setattr(learning, "load_state", lambda: _durum(0.6, 0.55, 0.65))
    d = decision.decide("AAPL", "event_breakout", {}, closes=_kapanislar(), side_hint="LONG")
    assert d.action == "ABSTAIN"
    assert d.utility is not None and d.utility <= 0
    assert "kapı-b fayda E ≤ 0" in d.evidence[-1]


def test_kapi_b_esiği_kayip_dalinda():
    kod = _kod("nihai/ai/decision.py")
    assert "- ( 1.0 - post ) * loss - util_lambda * abs ( cvar_pen )" not in kod
    assert "- ( 1.0 - post ) * ( loss + util_lambda * abs ( cvar_pen ) )" in kod


def test_persist_false_deftere_yazmaz(izole_karar, monkeypatch):
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    monkeypatch.setattr(learning, "load_state", lambda: _durum(0.6, 0.55, 0.65))
    decision.decide("AAPL", "event_breakout", {}, record_episode=False, persist=False)
    assert not decision.DECISIONS_FILE.exists()
    decision.decide("AAPL", "event_breakout", {}, record_episode=False)
    assert decision.DECISIONS_FILE.exists()


# ============================================================ 2. nöral OOD kapısı (BULGU-92-3)
class _Model:
    is_fitted = True
    feature_names = ["a", "b"]
    train_zmax_q99 = 3.0

    def __init__(self, z):
        self.z = z

    def ood_zmax(self, x):
        return np.array([self.z])

    def predict_with_uncertainty(self, x):
        return np.array([0.3]), np.array([0.05])


def test_brain_noral_dagilim_disi_girdiyi_atlar(monkeypatch):
    f = {"a": 1.0, "b": 2.0}
    monkeypatch.delenv("NEURAL_SHADOW_MAX_Z", raising=False)                 # canlı köprüyle aynı: 6,0
    assert brain._neural_forecast(_Model(21.0), f) is None
    assert brain._neural_forecast(_Model(5.9), f) == (0.3, 0.05)
    monkeypatch.setenv("NEURAL_SHADOW_MAX_Z", "auto")                        # max(4, 1,5·q99) = 4,5
    assert brain._neural_forecast(_Model(4.6), f) is None
    assert brain._neural_forecast(_Model(4.4), f) == (0.3, 0.05)
    monkeypatch.setenv("NEURAL_SHADOW_MAX_Z", "0")                           # kapı kapalı
    assert brain._neural_forecast(_Model(21.0), f) == (0.3, 0.05)


# ============================================================ 3. tahmin bandı (BULGU-92-1)
def _tb():
    from nihai.web import web_dashboard as wd
    return wd


def test_yuruyen_sinav_yalniz_cozulmus_gecmisi_kullanir():
    """t anında kullanılabilen z'ler s+h ≤ t olanlardır → sınav sayısı n − 2h − min_n + 1."""
    wd = _tb()
    rng = np.random.default_rng(1)
    n, h, m = 300, 5, 30
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    r = wd._tahmin_bandi(c, np.full(n, 1.0), h, min_n=m)
    assert r["sinav"]["n"] == n - 2 * h - m + 1
    assert r["n"] == n - h


def test_kapsama_nominale_yakin_ve_fiyatlar_sirali():
    wd = _tb()
    rng = np.random.default_rng(2)
    n = 3000
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    r = wd._tahmin_bandi(c, np.full(n, 1.0), 5)
    assert 75.0 <= r["sinav"]["kapsama"] <= 85.0
    assert r["alt"] <= r["medyan"] <= r["ust"]
    assert r["capa"] == round(c[-1], 2)
    if r["merkez_kaynak"] == "son_kapanis":
        assert r["merkez"] == r["capa"]


def test_merkez_medyana_yalniz_isabet_katarsa_gecer():
    wd = _tb()
    rng = np.random.default_rng(3)
    n = 600
    c = 100.0 * np.exp(np.cumsum(0.004 + rng.normal(0, 0.002, n)))           # güçlü, kalıcı sürüklenme
    r = wd._tahmin_bandi(c, np.full(n, 0.5), 5)
    assert r["merkez_kaynak"] == "medyan"
    assert r["merkez"] == r["medyan"] > r["capa"]
    assert r["sinav"]["mae_medyan_bps"] < r["sinav"]["mae_sabit_bps"]


def test_gecmis_kayma_yanilttiginda_merkez_son_kapanis():
    """300 bar yukarı + 300 bar aşağı: aşağı fazın tamamında tarihsel medyan hâlâ YUKARI der → yürüyen
    sınavda medyan "değişmez"den kötü (214,5 ↔ 201,6 bps) → merkez son kapanış (BULGU-84-2 dersi)."""
    wd = _tb()
    rng = np.random.default_rng(4)
    r = np.r_[np.full(300, 0.004), np.full(300, -0.004)] + rng.normal(0, 0.0005, 600)
    c = 100.0 * np.exp(np.cumsum(r))
    t = wd._tahmin_bandi(c, np.full(600, 0.5), 5)
    assert t["sinav"]["mae_medyan_bps"] > t["sinav"]["mae_sabit_bps"]
    assert t["merkez_kaynak"] == "son_kapanis"
    assert t["merkez"] == t["capa"] != t["medyan"]


def test_yetersiz_ya_da_bozuk_girdide_tahmin_yok():
    wd = _tb()
    c = np.linspace(100, 110, 30)
    assert wd._tahmin_bandi(c, np.full(30, 1.0), 5) is None                  # 25 z < 30
    c = np.linspace(100, 130, 200)
    a = np.full(200, 1.0)
    a[-1] = np.nan
    assert wd._tahmin_bandi(c, a, 5) is None                                  # son ATR yok → ölçek yok


# ============================================================ 4. gelecek seans barları
def test_saatlik_gelecek_hafta_sonunu_atlar():
    wd = _tb()
    g = wd._gelecek_barlar(pd.Timestamp("2026-09-18 19:30"), "1h", 9)        # Cuma son saat barı
    assert g[0] == _utc("2026-09-21 13:30")                                   # Pazartesi 09:30 ET
    assert g[6] == _utc("2026-09-21 19:30") and g[7] == _utc("2026-09-22 13:30")


def test_gunluk_gelecek_tatili_atlar_ve_etiketi_korur():
    wd = _tb()
    g = wd._gelecek_barlar(pd.Timestamp("2026-11-25 05:00"), "1d", 3)        # ET gece yarısı (EST)
    assert g == [_utc("2026-11-27 05:00"), _utc("2026-11-30 05:00"), _utc("2026-12-01 05:00")]


def test_yarim_gunde_saatlik_barlar_ve_bar_kapanisi():
    wd = _tb()
    g = wd._gelecek_barlar(pd.Timestamp("2026-11-25 20:30"), "1h", 5)
    assert g[:4] == [_utc(f"2026-11-27 {s}") for s in ("14:30", "15:30", "16:30", "17:30")]
    assert g[4] == _utc("2026-11-30 14:30")
    assert wd._bar_bitisi(g[3], "1h") == _utc("2026-11-27 18:00")           # 13:00 ET yarım gün
    assert wd._bar_bitisi(_utc("2026-09-18 19:30"), "1h") == _utc("2026-09-18 20:00")
    assert wd._bar_bitisi(_utc("2026-09-18 14:30"), "1h") == _utc("2026-09-18 15:30")
    assert wd._bar_bitisi(_utc("2026-11-27 05:00"), "1d") == _utc("2026-11-27 18:00")
    assert wd._gelecek_barlar(pd.Timestamp("2026-09-18 19:30"), "3h", 5) == []


# ============================================================ 5. /api/analyze
class _Saglayici:
    name = "sahte"

    def __init__(self, df):
        self.df = df

    def get_bars(self, sym, interval, lookback_bars=200):
        return self.df.tail(lookback_bars)

    def get_quote(self, sym):
        return None


def test_analiz_ucu_tahmin_gelecek_ve_quant(tmp_path, monkeypatch):
    from nihai.execution import live_predict_service as lps
    from nihai.web import web_dashboard as wd
    rng = np.random.default_rng(5)
    g = wd._gelecek_barlar(pd.Timestamp("2026-06-01 13:30"), "1h", 1000)
    idx = pd.DatetimeIndex([pd.Timestamp(t, unit="s") for t in g])
    kap = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
    df = pd.DataFrame({"open": kap, "high": kap * 1.002, "low": kap * 0.998, "close": kap,
                       "volume": 1e5}, index=idx)
    monkeypatch.setattr(wd, "_analysis_provider", lambda: _Saglayici(df))
    monkeypatch.setattr(wd, "is_market_open", lambda: False)
    monkeypatch.setattr(lps, "_seans_acilisi", lambda *a, **k: None)
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    monkeypatch.setattr(decision, "DECISIONS_FILE", tmp_path / "ai_decisions.jsonl")
    d = wd.analyze_symbol("AAPL", "1h")
    assert d["ok"]
    t = d["tahmin"]
    assert t["capa"] == round(float(kap[-1]), 2)
    assert t["n"] == 1000 - 5 - 13            # 1000 barlık pencere (220 değil); ATR14 ısınması 13 bar
    assert t["hedef_ts"] == d["gelecek"][4] and t["hedef_bitis_ts"] > t["hedef_ts"]
    assert d["gelecek"][0] > int(idx[-1].tz_localize("UTC").timestamp())
    ileri = [p["time"] for p in d["overlays"]["span_a_ileri"]]
    assert ileri == d["gelecek"][:26]
    q = d["quant"]
    assert q["action"] in ("LONG_BIAS", "SHORT_BIAS", "ABSTAIN") and q["taraf"] in ("LONG", "SHORT")
    assert q["kip"] in ("off", "advise", "enforce") and q["karar"].startswith("Karar:")
    assert not decision.DECISIONS_FILE.exists()                               # pano sorgusu yazmaz


def test_canli_danisma_ayni_bari_deftere_bir_kez_yazar(tmp_path, monkeypatch):
    """advise açılınca tarama turu aynı barı ~6 sn'de bir yeniden sordu (QCOM: tek barda onlarca satır).
    Karar her turda hesaplanır; `ai_decisions.jsonl` (kanal, sembol, taraf) başına bar başına BİR satır."""
    from types import SimpleNamespace

    from nihai.execution import ai_advisor
    monkeypatch.setenv("AI_LIVE_MODE", "advise")
    monkeypatch.setenv("AI_DECISION_MODE", "shadow")
    monkeypatch.setattr(decision, "DECISIONS_FILE", tmp_path / "ai_decisions.jsonl")
    monkeypatch.setattr(ai_advisor, "extract_features", lambda df, sig: {})
    monkeypatch.setattr(ai_advisor, "_son_kayit", {})
    monkeypatch.setattr(learning, "load_state", lambda: _durum(0.6, 0.55, 0.65))
    idx = pd.date_range("2026-09-01 13:30", periods=80, freq="h")
    df = pd.DataFrame({"close": np.linspace(100.0, 110.0, 80)}, index=idx)
    sig = SimpleNamespace(action=SimpleNamespace(value="LONG"), metadata={})
    satir = lambda: decision.DECISIONS_FILE.read_text(encoding="utf-8").strip().splitlines()  # noqa: E731
    for _ in range(3):
        assert ai_advisor.advise("ZZZ", df, sig) is not None                  # her tur yine görüş verir
    assert len(satir()) == 1
    assert ai_advisor.shadow_verdict("ZZZ", df, sig) is not None               # ayrı kanal (1dk ölçüm)
    assert len(satir()) == 2
    ai_advisor.advise("ZZZ", df.iloc[:-1], sig)                                # başka bar → yeni kayıt
    assert len(satir()) == 3


def test_ai_genel_bakis_canli_kipi_tasir(monkeypatch):
    from nihai.web import api_extra
    monkeypatch.setenv("AI_LIVE_MODE", "advise")
    assert api_extra.ai_overview()["live_mode"] == "advise"
    monkeypatch.setenv("AI_LIVE_MODE", "yanlis")
    assert api_extra.ai_overview()["live_mode"] == "off"


# ============================================================ 6. pano sözleşmesi (BULGU-92-4)
def test_pano_grafik_ve_kart_sozlesmesi():
    html = (KOK / "nihai/web/web_dashboard.html").read_text(encoding="utf-8")
    assert "width: el.clientWidth, height: 360" not in html                   # tek seferlik genişlik
    assert "autoSize: true" in html
    assert 'id="az-chart" style="height:360px;"' in html
    assert '<option value="1m">1dk</option>' in html
    for s in ("azHizala(sc.macd.hist)", "azHizala(sc.macd.macd)", "azHizala(sc.macd.signal)",
              "azHizala(sc.rsi.rsi)", "subscribeSizeChange(azEksenEsitle)", "tickMarkFormatter: azTick",
              "d.tahmin", "t.hedef_bitis_ts", "ov.span_a_ileri", 'id="az-quant"', "ov.live_mode"):
        assert s in html, s
    assert "addLine(d.band_lo" not in html                                     # koni fiyat çizgisinin yerinde


def test_env_ornegi_ve_kip_belgeli():
    ornek = (KOK / ".env.example").read_text(encoding="utf-8")
    assert "AI_LIVE_MODE=" in ornek and "BULGU-92-2" in ornek
