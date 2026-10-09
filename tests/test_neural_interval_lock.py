"""neural_shadow ARALIK TERCİHİ (4. kilit) testleri — Tur-20 / P2-11 önkoşulu.

Neden: `_pick()` "en yeni uygun kart"ı seçiyordu. 1m köprüsü canlıya alındıktan sonra bir
ARAŞTIRMA koşusunun (P2-11: 1h/234k sette eğitilen model) kaydedilmesi, o kartı en yeni
yapıp köprüyü SESSİZCE yeniden dağılım-dışına düşürürdü — O4 düzeltmesi geri alınırdı.
OOD guard bunu çıkarım anında yakalar ama sonuç yine sessizliktir.
İlke: "hiç seçilmesin" > "seçilip sussun".
"""
from __future__ import annotations

import pytest

from nihai.execution import neural_shadow as ns
from nihai.signals.meta_features import FEATURE_NAMES


def _kart(hash_, source, registered_at, base="data/model_x"):
    return {"hash": hash_, "kind": "neural", "tier": "deep_mlp",
            "feature_names": list(FEATURE_NAMES),
            "data": {"source": source, "positive_rate": 0.45},
            "artifacts": {"base_path": base},
            "registered_at": registered_at}


# ---------------------------------------------------------------------------
# _card_interval — dosya adından aralık çıkarımı
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("src", "bekl"), [
    ("data/meta_dataset_live_1m_2026-07-30.csv", "1m"),
    ("data/meta_dataset_pit_sp500_event_breakout_1h_2016-01-01_alpaca_hist.csv", "1h"),
    ("data/meta_dataset_pit_sp500_event_breakout_1d_2021-07-01_alpaca_hist.csv", "1d"),
    ("data/meta_dataset_x_15m_2026.csv", "15m"),
    ("synthetic:4000", None),
    ("", None),
])
def test_card_interval(src, bekl):
    assert ns._card_interval(_kart("h", src, "2026-01-01")) == bekl


def test_card_interval_savunmaci():
    assert ns._card_interval(None) is None
    assert ns._card_interval({}) is None
    assert ns._card_interval({"data": None}) is None


# ---------------------------------------------------------------------------
# _live_interval
# ---------------------------------------------------------------------------
def test_live_interval_predict_oncelikli(monkeypatch):
    monkeypatch.setenv("PREDICT_INTERVAL", "1m")
    assert ns._live_interval() == "1m"


def test_live_interval_htf_yedegi(monkeypatch):
    monkeypatch.delenv("PREDICT_INTERVAL", raising=False)
    assert ns._live_interval() == "1h"   # nihai.config varsayılanı


# ---------------------------------------------------------------------------
# _pick — asıl regresyon kapısı
# ---------------------------------------------------------------------------
def _pick_ile(monkeypatch, tmp_path, cards, live_iv="1m"):
    """Sahte registry + diskte var olan artefaktla _pick() çalıştır."""
    from nihai.ai import registry

    base = tmp_path / "model_x"
    (tmp_path / "model_x.json").write_text("{}", encoding="utf-8")
    for c in cards:
        c["artifacts"]["base_path"] = str(base)
    monkeypatch.setattr(registry, "list_models", lambda *_a, **_k: cards)
    monkeypatch.setenv("PREDICT_INTERVAL", live_iv)
    monkeypatch.delenv("NEURAL_SHADOW_MODEL", raising=False)

    sh = ns._NeuralShadow.__new__(ns._NeuralShadow)
    sh.reason = None
    return sh._pick()


def test_arastirma_karti_canli_1m_koprusunu_calmaz(monkeypatch, tmp_path):
    """P2-11 SENARYOSU: 1h araştırma kartı DAHA YENİ olsa bile seçilmemeli."""
    canli_1m = _kart("aaa", "data/meta_dataset_live_1m_2026-07-30.csv", "2026-07-30T19:49:09+00:00")
    arastirma_1h = _kart("bbb", "data/meta_dataset_extra_us_event_breakout_1h_2016-01-01.csv",
                         "2026-07-31T10:00:00+00:00")   # DAHA YENİ
    _base, card = _pick_ile(monkeypatch, tmp_path, [canli_1m, arastirma_1h], live_iv="1m")
    assert card is not None and card["hash"] == "aaa"


def test_eslesen_aralikta_en_yeni_secilir(monkeypatch, tmp_path):
    eski_1m = _kart("eski", "data/meta_dataset_live_1m_2026-07-01.csv", "2026-07-01T00:00:00+00:00")
    yeni_1m = _kart("yeni", "data/meta_dataset_live_1m_2026-07-30.csv", "2026-07-30T00:00:00+00:00")
    _base, card = _pick_ile(monkeypatch, tmp_path, [eski_1m, yeni_1m], live_iv="1m")
    assert card["hash"] == "yeni"


def test_eslesen_yoksa_eski_davranis_korunur(monkeypatch, tmp_path):
    """Eşleşen aralık yoksa köprü yine de bir kart seçer → guard NEDENİNİ söyleyerek susar
    (sessiz ölüm yasak). Hiç kart seçmemek 'reason'u da yok ederdi."""
    a = _kart("a", "data/meta_dataset_x_1d_2021.csv", "2026-07-01T00:00:00+00:00")
    b = _kart("b", "data/meta_dataset_x_1h_2016.csv", "2026-07-02T00:00:00+00:00")
    _base, card = _pick_ile(monkeypatch, tmp_path, [a, b], live_iv="1m")
    assert card is not None and card["hash"] == "b"      # en yeni (eski davranış)


def test_canli_aralik_bilinmiyorsa_eski_davranis(monkeypatch, tmp_path):
    a = _kart("a", "data/meta_dataset_x_1d_2021.csv", "2026-07-01T00:00:00+00:00")
    b = _kart("b", "data/meta_dataset_live_1m_2026.csv", "2026-07-02T00:00:00+00:00")
    _base, card = _pick_ile(monkeypatch, tmp_path, [a, b], live_iv="")
    assert card["hash"] == "b"
