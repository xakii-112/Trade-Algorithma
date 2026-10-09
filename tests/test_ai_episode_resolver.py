"""nihai/ai/episode_resolver.py — gölge karar epizotlarının SONUÇ kolu (B-3, 2026-07-26).

Kapsam: taraf çözümü, ufuk beklemesi (look-ahead disiplini), kazanç/kayıp etiketi,
ABSTAIN'in nötr kalması, çift-çözüm olmaması ve rapor toplamı.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nihai.ai import episode_resolver as er
from nihai.ai import episodic


@pytest.fixture(autouse=True)
def _tmp_episodes(tmp_path, monkeypatch):
    """Epizot dosyasını geçici dizine yönlendir (üretim dosyalarına dokunma — BULGU-8/9)."""
    monkeypatch.setattr(episodic, "EPISODES_FILE", tmp_path / "ai_episodes.jsonl")


class _Provider:
    """Deterministik günlük bar sağlayıcı: kapanışlar `closes`, index günlük UTC."""

    def __init__(self, closes: list[float], son_tarih: str = "2026-07-24"):
        self.closes = closes
        self.son_tarih = son_tarih
        self.cagrilar: list[str] = []

    def get_bars(self, symbol, interval, lookback_bars=None):   # noqa: ARG002
        self.cagrilar.append(symbol)
        idx = pd.date_range(end=pd.Timestamp(self.son_tarih, tz="UTC"),
                            periods=len(self.closes), freq="D")
        return pd.DataFrame({"close": self.closes}, index=idx)


def _ekle(action: str, symbol: str = "AAA", ts_gun: str = "2026-07-14") -> episodic.Episode:
    ep = episodic.append_episode(kind="ai_decision", symbol=symbol, features={"ret_5": 0.01},
                                 summary=f"event_breakout: {action}",
                                 tags=["event_breakout", action.lower()])
    # ts'i geriye al (append_episode "şimdi" yazar) — ham satırı yeniden yazmak yerine
    # dosyaya ts'i düzeltilmiş kopyayı eklemek yerine, testte doğrudan dosyayı güncelliyoruz.
    p = episodic.EPISODES_FILE
    lines = p.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1].replace(ep.ts, f"{ts_gun}T00:00:00+00:00")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ep.ts = f"{ts_gun}T00:00:00+00:00"
    return ep


def test_long_bias_yukselen_seride_kazanir():
    _ekle("LONG_BIAS")
    prov = _Provider([100.0 + i for i in range(30)])       # düz yükseliş
    out = er.resolve_pending(horizon_bars=5, provider=prov)
    assert out["cozulen"] == 1 and out["kazanan"] == 1 and out["kaybeden"] == 0
    eps = episodic.recent_episodes(kind="ai_decision", days=90)
    assert eps[0].outcome["win"] is True
    assert eps[0].outcome["fwd_ret_pct"] > 0
    assert eps[0].outcome["horizon_bars"] == 5


def test_short_bias_yukselen_seride_kaybeder():
    _ekle("SHORT_BIAS")
    prov = _Provider([100.0 + i for i in range(30)])
    out = er.resolve_pending(horizon_bars=5, provider=prov)
    assert out["cozulen"] == 1 and out["kaybeden"] == 1
    assert episodic.recent_episodes(kind="ai_decision", days=90)[0].outcome["win"] is False


def test_abstain_notr_kalir_ama_ileri_getiri_kaydedilir():
    """ABSTAIN'de taraf yok → uydurma 'kazandı' etiketi YAZILMAZ (ACI DERS #3)."""
    _ekle("ABSTAIN")
    prov = _Provider([100.0 + i for i in range(30)])
    out = er.resolve_pending(horizon_bars=5, provider=prov)
    assert out["cozulen"] == 1 and out["notr"] == 1
    oc = episodic.recent_episodes(kind="ai_decision", days=90)[0].outcome
    assert oc["win"] is None and oc["fwd_ret_pct"] > 0 and oc["side"] == 0


def test_ufuk_dolmadiysa_cozulmez():
    """Karar barından sonra `horizon` bar YOKSA epizot BEKLER — geleceği uydurmaz."""
    _ekle("LONG_BIAS", ts_gun="2026-07-23")
    prov = _Provider([100.0 + i for i in range(30)], son_tarih="2026-07-24")
    out = er.resolve_pending(horizon_bars=5, provider=prov)
    assert out["bekleyen"] == 1 and out["cozulen"] == 0
    assert episodic.recent_episodes(kind="ai_decision", days=90)[0].outcome is None


def test_karar_bari_ts_ten_sonraki_bar_olamaz():
    """Karar barı ts'ten SONRAKİ bir bar seçilirse ileri getiri sızıntı olurdu."""
    ep = _ekle("LONG_BIAS", ts_gun="2026-07-14")
    prov = _Provider([100.0 + i for i in range(30)])
    er.resolve_pending(horizon_bars=5, provider=prov)
    oc = episodic.recent_episodes(kind="ai_decision", days=90)[0].outcome
    assert pd.Timestamp(oc["decision_bar"]) <= pd.Timestamp(ep.ts)


def test_cozulmus_epizot_tekrar_cozulmez():
    _ekle("LONG_BIAS")
    prov = _Provider([100.0 + i for i in range(30)])
    assert er.resolve_pending(horizon_bars=5, provider=prov)["cozulen"] == 1
    ikinci = er.resolve_pending(horizon_bars=5, provider=prov)
    assert ikinci["aday"] == 0 and ikinci["cozulen"] == 0


def test_veri_yoksa_sayilir_ve_cokmez():
    class _Bos:
        def get_bars(self, *a, **k):
            raise RuntimeError("sağlayıcı yok")
    _ekle("LONG_BIAS")
    out = er.resolve_pending(horizon_bars=5, provider=_Bos())
    assert out["veri_yok"] == 1 and out["cozulen"] == 0


def test_min_abs_ret_gurultu_bandi_notrler(monkeypatch):
    monkeypatch.setattr(er, "MIN_ABS_RET", 0.50)   # %50 — hiçbir hareket geçemez
    _ekle("LONG_BIAS")
    prov = _Provider([100.0 + i for i in range(30)])
    out = er.resolve_pending(horizon_bars=5, provider=prov)
    assert out["notr"] == 1 and out["kazanan"] == 0
    assert episodic.recent_episodes(kind="ai_decision", days=90)[0].outcome["win"] is None


def test_outcome_report_toplar():
    _ekle("LONG_BIAS", symbol="AAA")
    _ekle("SHORT_BIAS", symbol="BBB")
    prov = _Provider([100.0 + i for i in range(30)])
    er.resolve_pending(horizon_bars=5, provider=prov)
    rep = er.outcome_report()
    assert rep["n_epizot"] == 2 and rep["n_cozulmus"] == 2 and rep["n_tarafli"] == 2
    assert rep["isabet"] == 0.5 and rep["cozulme_orani"] == 1.0


def test_bos_hafizada_cokmez():
    out = er.resolve_pending(horizon_bars=5, provider=_Provider([100.0]))
    assert out["aday"] == 0 and out["cozulen"] == 0
    assert er.outcome_report()["n_epizot"] == 0


def test_forward_return_sonlu_olmayan_fiyati_reddeder():
    idx = pd.date_range(end=pd.Timestamp("2026-07-24", tz="UTC"), periods=10, freq="D")
    df = pd.DataFrame({"close": [100.0] * 4 + [np.nan] + [100.0] * 5}, index=idx)
    assert er._forward_return(df, pd.Timestamp("2026-07-19", tz="UTC").to_pydatetime(), 1) is None
