"""
test_tur38_ai_karar_koprusu.py — Tur-38: iç AI ↔ CANLI KARAR/EMİR yolu köprüsü.

Üç soruyu birden kilitler:

  1. **BAĞLANDI MI?** `scan_symbol` (tek canlı karar yolu) köprüyü gerçekten çağırıyor
     mu, veto girişi durduruyor mu, boyut çarpanı `conf_mult`'a giriyor mu?
     (Tur-37'nin "yazıldı ama kimse çağırmıyor" kusuru tekrar etmesin.)
  2. **VARSAYILAN GÜVENLİ Mİ?** `AI_LIVE_MODE` verilmemişken köprü HİÇ çalışmıyor mu
     (davranış Tur-37 ile bit-denk), yanlış yazılmış bir kip enforce'a düşmüyor mu?
  3. **TEK YÖNLÜ MÜ?** enforce yalnız SUSTURUYOR ve KÜÇÜLTÜYOR mu — yön çeviremiyor,
     pozisyonu büyütemiyor, ÇIKIŞ sinyaline hiç dokunamıyor mu?

Ağ YOK, model YOK, disk yazımı YOK — sentetik df + monkeypatch'lenmiş `decide`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nihai.ai.decision import ShadowDecision
from nihai.execution import ai_advisor
from nihai.signals.strategy import Signal, SignalAction

_MAIN = Path(__file__).resolve().parents[1] / "nihai" / "execution" / "main.py"


def _bars(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.008, n)))
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": close, "high": close * 1.004, "low": close * 0.996,
                         "close": close, "volume": 1e6}, index=idx)


def _sig(action: SignalAction = SignalAction.LONG, **meta) -> Signal:
    return Signal(action=action, symbol="AAA", price=100.0, stop=98.0, target=104.0,
                  reasons=["test"], quality_score=3.0, rr_ratio=2.0, metadata=dict(meta))


def _karar(action: str = "LONG_BIAS", posterior: float = 0.55,
           conviction: float = 0.10) -> ShadowDecision:
    return ShadowDecision(
        ts="2026-08-21T00:00:00+00:00", symbol="AAA", strategy="event_breakout",
        action=action, conviction=conviction, prior_mean=0.5, posterior=posterior,
        evidence=[f"Karar: {action}."], kelly_suggest=0.0, regime=None,
        lesson_warnings=[], mode="off", utility=0.01)


@pytest.fixture
def sahte_karar(monkeypatch):
    """`decide` yerine kontrollü bir ShadowDecision koy; çağrı kwargs'ını da yakala."""
    yakalanan: dict = {}

    def _kur(karar: ShadowDecision) -> dict:
        def _fake(symbol, strategy, features, **kw):
            yakalanan.update({"symbol": symbol, "strategy": strategy,
                              "features": features, **kw})
            return karar
        monkeypatch.setattr("nihai.ai.decision.decide", _fake)
        return yakalanan
    return _kur


# ---------------------------------------------------------------------------
# 2) Varsayılan güvenli mi?
# ---------------------------------------------------------------------------
def test_varsayilan_off_karar_motorunu_hic_cagirmaz(monkeypatch):
    """AI_LIVE_MODE yokken köprü ÖLÜDÜR: decide() bir kez bile çağrılmaz."""
    monkeypatch.delenv("AI_LIVE_MODE", raising=False)

    def _patlat(*a, **k):
        raise AssertionError("off kipinde decide() çağrıldı — anayasa ihlali")

    monkeypatch.setattr("nihai.ai.decision.decide", _patlat)
    assert ai_advisor.advise("AAA", _bars(), _sig()) is None
    assert ai_advisor.resolve_mode() == "off"


def test_kopru_modulu_ai_katmanini_import_etmez():
    """Tur-26 "lazy nihai.ai" disiplini: köprünün İMPORTU `nihai.ai` zincirini (ve
    scipy'yi) çekmemeli — tarayıcı açılışına ~2,3 sn ekliyordu ve VARSAYILAN kip onu
    hiç kullanmıyor. `decide` importu `advise()` içinde, kip kontrolünden SONRA olmalı."""
    src = Path(ai_advisor.__file__).read_text(encoding="utf-8")
    tepe = src.split("def resolve_mode", 1)[0]
    assert "from nihai.ai import decision" not in tepe, "AI importu modül tepesine kaçmış"
    assert "from nihai.ai import decision" in src


@pytest.mark.parametrize("deger", ["", "  ", "shadow", "enfroce", "aç", "advise-plus"])
def test_taninmayan_kip_enforce_a_dusmez(monkeypatch, deger):
    """Yanlış yazım sessizce enforce olamaz (decision.resolve_mode ile aynı güvenlik)."""
    monkeypatch.setenv("AI_LIVE_MODE", deger)
    assert ai_advisor.resolve_mode() == "off"


# ---------------------------------------------------------------------------
# 3) Tek yönlü mü? (advise nötr · enforce yalnız susturur/küçültür)
# ---------------------------------------------------------------------------
def test_advise_kipi_sonucu_degistirmez(monkeypatch, sahte_karar):
    """advise: ABSTAIN bile olsa veto YOK, boyut çarpanı NÖTR — yalnız ölçüm/şeffaflık."""
    monkeypatch.setenv("AI_LIVE_MODE", "advise")
    sahte_karar(_karar("ABSTAIN", posterior=0.50, conviction=0.0))
    adv = ai_advisor.advise("AAA", _bars(), _sig())
    assert adv is not None
    assert (adv.veto, adv.size_mult) == (False, 1.0)
    assert "AI_ADVISE" in adv.reason


def test_enforce_abstain_veto_eder(monkeypatch, sahte_karar):
    monkeypatch.setenv("AI_LIVE_MODE", "enforce")
    sahte_karar(_karar("ABSTAIN", posterior=0.50, conviction=0.0))
    adv = ai_advisor.advise("AAA", _bars(), _sig())
    assert adv is not None and adv.veto is True
    assert "VETO" in adv.reason


def test_enforce_ters_yonde_veto_eder_yon_cevirmez(monkeypatch, sahte_karar):
    """LONG adayına SHORT_BIAS gelirse sonuç VETO'dur — "ters yöne gir" ASLA olmaz."""
    monkeypatch.setenv("AI_LIVE_MODE", "enforce")
    sahte_karar(_karar("SHORT_BIAS", posterior=0.20))
    sig = _sig(SignalAction.LONG)
    adv = ai_advisor.advise("AAA", _bars(), sig)
    assert adv is not None and adv.veto is True
    assert sig.action is SignalAction.LONG          # sinyalin yönü DEĞİŞMEDİ


@pytest.mark.parametrize("posterior", [0.0, 0.45, 0.50, 0.55, 0.60, 0.99, 1.0])
def test_enforce_boyutu_asla_buyutmez(monkeypatch, sahte_karar, posterior):
    """size_mult tavanı 1.0 (prob_sizing sözleşmesi) ve pozitif kalır."""
    monkeypatch.setenv("AI_LIVE_MODE", "enforce")
    sahte_karar(_karar("LONG_BIAS", posterior=posterior))
    adv = ai_advisor.advise("AAA", _bars(), _sig())
    assert adv is not None
    assert 0.0 < adv.size_mult <= 1.0


def test_enforce_zayif_posterioru_kisar(monkeypatch, sahte_karar):
    """Rampa gerçekten çalışıyor: 0.50 taban → floor, 0.60 üstü → 1.0."""
    monkeypatch.setenv("AI_LIVE_MODE", "enforce")
    sahte_karar(_karar("LONG_BIAS", posterior=0.50))
    zayif = ai_advisor.advise("AAA", _bars(), _sig())
    sahte_karar(_karar("LONG_BIAS", posterior=0.70))
    guclu = ai_advisor.advise("AAA", _bars(), _sig())
    assert zayif is not None and guclu is not None
    assert zayif.size_mult < guclu.size_mult == 1.0
    assert "boyut" in zayif.reason


@pytest.mark.parametrize("action", [SignalAction.EXIT, SignalAction.WAIT])
def test_cikis_sinyali_ai_a_sorulmaz(monkeypatch, action):
    """Zarar-kesme/çıkış yolu AI'a KAPALI: enforce'ta bile decide() çağrılmaz."""
    monkeypatch.setenv("AI_LIVE_MODE", "enforce")

    def _patlat(*a, **k):
        raise AssertionError("çıkış sinyali için decide() çağrıldı")

    monkeypatch.setattr("nihai.ai.decision.decide", _patlat)
    assert ai_advisor.advise("AAA", _bars(), _sig(action)) is None


# ---------------------------------------------------------------------------
# Köprünün kendisi: girdi hijyeni + fail-soft
# ---------------------------------------------------------------------------
def test_pwin_metadata_dan_okunur_ikinci_model_cagrisi_yok(monkeypatch, sahte_karar):
    """p_win, strategy.py'ın yazdığı `meta_p_win`'den gelir — yeni bir çıkarım DOĞMAZ."""
    monkeypatch.setenv("AI_LIVE_MODE", "advise")
    yakalanan = sahte_karar(_karar())
    ai_advisor.advise("AAA", _bars(), _sig(meta_p_win=0.63), strategy="squeeze")
    assert yakalanan["p_win"] == pytest.approx(0.63)
    assert yakalanan["side_hint"] == "LONG"
    assert yakalanan["strategy"] == "squeeze"
    assert yakalanan["closes"] is not None


def test_pwin_yoksa_none_gecilir(monkeypatch, sahte_karar):
    monkeypatch.setenv("AI_LIVE_MODE", "advise")
    yakalanan = sahte_karar(_karar())
    ai_advisor.advise("AAA", _bars(), _sig())
    assert yakalanan["p_win"] is None


def test_decide_patlarsa_fail_soft(monkeypatch):
    """AI hatası canlı taramayı DÜŞÜRMEZ ama sessizce de ölmez (last_error)."""
    monkeypatch.setenv("AI_LIVE_MODE", "enforce")

    def _patlat(*a, **k):
        raise RuntimeError("bilerek")

    monkeypatch.setattr("nihai.ai.decision.decide", _patlat)
    assert ai_advisor.advise("AAA", _bars(), _sig()) is None
    assert "bilerek" in (ai_advisor.status()["last_error"] or "")


@pytest.mark.parametrize("df", [None, pd.DataFrame(), pd.DataFrame({"close": [1.0]})])
def test_bozuk_df_istisna_firlatmaz(monkeypatch, sahte_karar, df):
    monkeypatch.setenv("AI_LIVE_MODE", "enforce")
    sahte_karar(_karar())
    ai_advisor.advise("AAA", df, _sig())   # patlamamalı (None veya AIAdvice)


# ---------------------------------------------------------------------------
# 1) Bağlandı mı? — üretim çağıranı gerçekten var mı?
# ---------------------------------------------------------------------------
def test_scan_symbol_kopruyu_cagiriyor():
    src = _MAIN.read_text(encoding="utf-8")
    assert "ai_advisor.advise(" in src, "scan_symbol köprüyü çağırmıyor — köprü ölü"
    assert 'log_block("AI_VETO"' in src, "veto girişi durdurmuyor"
    assert "conf_mult *= ai_adv.size_mult" in src, "boyut çarpanı sizing'e girmiyor"


def test_ai_kapisi_determinist_guardlardan_sonra():
    """AI, kuralların YERİNE değil ARDINA konur: kill-switch/seans/cooldown/olay
    guard'ları AI çağrısından ÖNCE gelmeli (AI onları atlatamaz)."""
    src = _MAIN.read_text(encoding="utf-8")
    ai_idx = src.index("ai_advisor.advise(")
    for guard in ('log_block("KILL_SWITCH_ACTIVE"', 'log_block("SESSION_GUARD"',
                  'log_block("COOLDOWN"', 'log_block("EVENT_GUARD"'):
        assert src.index(guard) < ai_idx, f"{guard} AI kapısından SONRA kalmış"


# ---------------------------------------------------------------------------
# 4) ÖLÇÜM yolu (Tur-38B) — verdikt makine-okunur kolona yazılıyor mu?
#     "Kapı açılmalı mı?" sorusu ancak kapı KAPALIYKEN biriken kayıtla yanıtlanır.
# ---------------------------------------------------------------------------
def test_shadow_verdict_off_kipinde_de_olcer(monkeypatch, sahte_karar):
    """Kanıt döngüsünün TAMAMI bu testte: `AI_LIVE_MODE=off` (varsayılan) iken bile
    verdikt hesaplanmalı. Aksi hâlde kapı kapalıyken kayıt birikmez ve 'enforce
    açılmalı mı?' sorusu bir daha ASLA yanıtlanamaz (META_GATE shadow dersi)."""
    monkeypatch.delenv("AI_LIVE_MODE", raising=False)
    monkeypatch.delenv("AI_VERDICT_LOG", raising=False)
    sahte_karar(_karar("ABSTAIN", posterior=0.5391))
    v = ai_advisor.shadow_verdict("AAA", _bars(), _sig())
    assert v == {"action": "ABSTAIN", "post": 0.5391, "veto": 1, "side": "LONG"}
    assert ai_advisor.advise("AAA", _bars(), _sig()) is None   # karar yolu hâlâ ÖLÜ


def test_shadow_verdict_uyumlu_yonde_veto_sifir(monkeypatch, sahte_karar):
    monkeypatch.delenv("AI_LIVE_MODE", raising=False)
    sahte_karar(_karar("LONG_BIAS", posterior=0.62))
    assert ai_advisor.shadow_verdict("AAA", _bars(), _sig())["veto"] == 0
    sahte_karar(_karar("SHORT_BIAS", posterior=0.62))
    assert ai_advisor.shadow_verdict("AAA", _bars(), _sig(SignalAction.LONG))["veto"] == 1


def test_verdict_log_kapatilabilir(monkeypatch, sahte_karar):
    monkeypatch.setenv("AI_VERDICT_LOG", "off")
    sahte_karar(_karar())
    assert ai_advisor.shadow_verdict("AAA", _bars(), _sig()) is None


def test_olculen_verdikt_uygulanacak_verdiktle_ayni(monkeypatch, sahte_karar):
    """ANTI-SAPMA KİLİDİ: log'a yazılan verdikt ile enforce'un uygulayacağı verdikt AYNI
    çekirdekten gelmeli. Ayrı hesaplansalardı biriken kanıt yanlış soruyu yanıtlardı."""
    monkeypatch.setenv("AI_LIVE_MODE", "enforce")
    sahte_karar(_karar("ABSTAIN", posterior=0.48))
    olcum = ai_advisor.shadow_verdict("AAA", _bars(), _sig())
    karar = ai_advisor.advise("AAA", _bars(), _sig())
    assert olcum["action"] == karar.action
    assert olcum["post"] == pytest.approx(karar.posterior, abs=1e-4)
    assert bool(olcum["veto"]) is karar.veto


def test_ai_kolonlari_semada_var():
    from nihai.execution.prediction_tracker import FIELDS
    for col in ("ai_action", "ai_post", "ai_veto", "ai_side"):
        assert col in FIELDS


def _satir(symbol="AAA", bar_ts="2026-08-21T14:00:00", **extra):
    row = {"symbol": symbol, "bar_ts": bar_ts, "direction": "+", "action": "LONG",
           "confidence": 0.61, "prob_cal": 0.55, "price": 101.5, "bar_close": 101.0,
           "pwin": None, "pwin_side": None}
    row.update(extra)
    return row


def _son_satir(path):
    import csv
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))[-1]


def test_record_ai_kolonlarini_diske_yazar():
    """UÇTAN UCA: `_ai` → `record()` → CSV. Zincirin bir halkası kopsa verdikt sessizce
    kaybolur ve kolonlar boş kaldığı için kimse fark etmez."""
    from nihai.execution import prediction_tracker as pt

    ai = {"action": "ABSTAIN", "post": 0.5391, "veto": 1, "side": "LONG"}
    assert pt.record([_satir(symbol="AIX1", _ai=ai)], "1m") == 1
    row = _son_satir(pt.PRED_LOG)
    assert (row["ai_action"], row["ai_post"], row["ai_veto"], row["ai_side"]) == (
        "ABSTAIN", "0.5391", "1", "LONG")
    # ANAYASA: karar sütunları DEĞİŞMEDİ — verdikt yalnız kendi kolonlarına yazılır.
    assert row["direction"] == "+" and row["confidence"] == "0.61"


@pytest.mark.parametrize("bozuk", [
    None, {},
    {"action": "HAYIR", "post": 0.5, "veto": 0, "side": "LONG"},
    {"action": "ABSTAIN", "post": "x", "veto": 0, "side": "LONG"},
    {"action": "ABSTAIN", "post": 0.5, "veto": 7, "side": "LONG"},
    {"action": "ABSTAIN", "post": 0.5, "veto": 1},               # taraf YOK → boş
])
def test_bozuk_verdikt_kaydi_engellemez(bozuk):
    """Gölge katman tahmin satırından daha az önemlidir: bozuk verdikt → boş hücre."""
    from nihai.execution import prediction_tracker as pt

    assert pt.record([_satir(symbol=f"AIX{abs(hash(str(bozuk))) % 9999}",
                             _ai=bozuk)], "1m") == 1
    row = _son_satir(pt.PRED_LOG)
    assert row["ai_action"] == "" and row["ai_veto"] == ""


def test_olcum_koprusunun_uretim_cagirani_var():
    """Tur-37'nin 'yazıldı ama kimse çağırmıyor' kusuru bu köprüde de tekrar etmesin."""
    lps = (_MAIN.parent / "live_predict_service.py").read_text(encoding="utf-8")
    pt = (_MAIN.parent / "prediction_tracker.py").read_text(encoding="utf-8")
    assert "shadow_verdict(" in lps and "_ai_verdict(" in lps
    assert "_ai_cells(" in pt
    assert 'r.pop("_ai", None)' in lps, "'_ai' satırdan düşürülmüyor (web JSON'una sızar)"


def test_meta_gate_enforce_adayi_olcumden_dusmez(monkeypatch, sahte_karar):
    """META_GATE=enforce aday sinyali WAIT'e çevirir; `sig.action`a bakan bir ölçüm o
    adayı sessizce atlar ve kayıt SEÇİLİM YANLI olur (canlı kanıt döngüsünün
    `_signal_pwin`'de kırdığı yanlılığın tıpkısı). `side` override bunu engeller."""
    monkeypatch.delenv("AI_LIVE_MODE", raising=False)
    sahte_karar(_karar("LONG_BIAS", posterior=0.61))
    engellenmis = _sig(SignalAction.WAIT, meta_p_win=0.31, meta_gated="LONG")
    assert ai_advisor.shadow_verdict("AAA", _bars(), engellenmis) is None       # side'sız
    v = ai_advisor.shadow_verdict("AAA", _bars(), engellenmis, side="LONG")     # side'lı
    assert v["side"] == "LONG" and v["veto"] == 0


# ---------------------------------------------------------------------------
# 5) Kolonu OKUYAN taraf: ai_gate_sim (yazılıp okunmayan kolon bu projede kusurdur)
# ---------------------------------------------------------------------------
def _cozulmus(veto: int, ret: float, side: str = "LONG") -> dict:
    """Çözülmüş bir predictions_log satırı (CSV'den okunmuş gibi: hepsi string)."""
    return {"ai_veto": str(veto), "ai_side": side, "direction": "+",
            "fwd_ret_pct": str(ret if side == "LONG" else -ret)}


def test_ai_gate_sim_yetersiz_orneklemde_hukum_vermez():
    from nihai.execution.prediction_tracker import ai_gate_sim
    out = ai_gate_sim([_cozulmus(1, -0.4) for _ in range(100)], min_side=30)
    assert out["label"] == "INSUFFICIENT"        # "geçen" kovası boş → hüküm YOK
    assert out["cut"]["n"] == 100 and out["kept"]["n"] == 0


def test_ai_gate_sim_ayiran_kapiyi_bulur():
    from nihai.execution.prediction_tracker import ai_gate_sim
    rows = [_cozulmus(0, +0.5) for _ in range(40)] + [_cozulmus(1, -0.5) for _ in range(40)]
    out = ai_gate_sim(rows, min_side=30)
    assert out["label"] == "AI_GATE_HELPS" and out["uplift_pp"] > 0


def test_ai_gate_sim_zarar_veren_kapiyi_da_soyler():
    from nihai.execution.prediction_tracker import ai_gate_sim
    rows = [_cozulmus(0, -0.5) for _ in range(40)] + [_cozulmus(1, +0.5) for _ in range(40)]
    assert ai_gate_sim(rows, min_side=30)["label"] == "AI_GATE_HURTS"


def test_ai_gate_sim_getiriyi_TARAFA_gore_isaretler():
    """SHORT adayda fiyat düşüşü KAZANÇTIR. Taraf okunmazsa işaret ters döner ve
    ayıran bir kapı zarar veriyormuş gibi görünür."""
    from nihai.execution.prediction_tracker import ai_gate_sim
    rows = ([_cozulmus(0, +0.5, "SHORT") for _ in range(40)]
            + [_cozulmus(1, -0.5, "SHORT") for _ in range(40)])
    assert ai_gate_sim(rows, min_side=30)["label"] == "AI_GATE_HELPS"


def test_ai_gate_sim_eski_sema_satirini_atlar():
    """`ai_veto` kolonu olmayan (Tur-38 öncesi) satır simülasyona GİRMEZ."""
    from nihai.execution.prediction_tracker import ai_gate_sim
    assert ai_gate_sim([{"direction": "+", "fwd_ret_pct": "0.5"}] * 50)["n_candidates"] == 0
