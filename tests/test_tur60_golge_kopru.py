"""Gölge köprüsü — Tur-60 kilidi (`nihai/execution/shadow_bridge.py`).

Dört soruyu birden sorar:
  1. BAĞLANDI MI? — `main.scan_symbol` köprüyü GERÇEKTEN çağırıyor mu, ölçüler
     gerekçeye/metadata'ya düşüyor mu, çarpan boyutlamaya giriyor mu.
  2. VARSAYILAN GÜVENLİ Mİ? — `advise` kipinde karar DEĞİŞMEZ; yanlış yazılmış kip
     enforce'a düşmez; `off` iken tek satır bile çalışmaz.
  3. TEK YÖNLÜ MÜ? — yön çevrilemez, boyut büyütülemez, ÇIKIŞ yoluna bağlanmaz.
  4. TERFİ DEFTERİ İŞLİYOR MU? — `PROMOTE` taşımayan bir kural enforce kipinde bile
     hüküm veremez (ölçüm ile canlı davranış ayrışamaz).
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pandas as pd
import pytest

from nihai.execution import shadow_bridge as sb
from nihai.signals.strategy import Signal, SignalAction

MAIN_PY = Path(__file__).resolve().parents[1] / "nihai" / "execution" / "main.py"


@pytest.fixture(autouse=True)
def _temiz(monkeypatch):
    """Her test kendi kipini kursun; sayaçlar/uyarı bayrağı sızmasın."""
    monkeypatch.delenv("SHADOW_BRIDGE", raising=False)
    monkeypatch.delenv("SHADOW_BRIDGE_NEURAL", raising=False)
    sb._n_calls = sb._n_marks = 0
    sb._last_error = None
    sb._enforce_uyarildi = False


def _df(n: int = 120) -> pd.DataFrame:
    """Gözlemevinin ölçebileceği kadar uzun, dejenere OLMAYAN bar serisi."""
    idx = pd.date_range("2026-08-01", periods=n, freq="1h")
    close = [100.0 + (i % 7) * 0.4 - (i % 3) * 0.25 for i in range(n)]
    return pd.DataFrame({"open": close, "high": [c + 0.5 for c in close],
                         "low": [c - 0.5 for c in close], "close": close,
                         "volume": [1_000 + i for i in range(n)]}, index=idx)


def _sig(action=SignalAction.LONG, price=100.0) -> Signal:
    return Signal(action=action, symbol="AAPL", price=price, stop=price - 1.0,
                  target=price + 2.0, reasons=["TEST"], quality_score=5.0, rr_ratio=2.0)


# --------------------------------------------------------- 2) VARSAYILAN GÜVENLİ Mİ
def test_varsayilan_kip_advise():
    assert sb.resolve_mode() == "advise"


@pytest.mark.parametrize("deger", ["", "  ", "ADVISE", "tavsiye", "enfore", "veto1", "şey"])
def test_yanlis_yazim_enforce_a_dusmez(monkeypatch, deger):
    """Yanlış yazılmış bir kip SESSİZCE enforce olamaz (ai_advisor ile aynı disiplin)."""
    monkeypatch.setenv("SHADOW_BRIDGE", deger)
    assert sb.resolve_mode() != "enforce"


@pytest.mark.parametrize("deger", ["off", "0", "false", "kapali", "kapalı", "no"])
def test_off_kipinde_tek_satir_calismaz(monkeypatch, deger):
    monkeypatch.setenv("SHADOW_BRIDGE", deger)
    cagrildi = []
    monkeypatch.setattr(sb, "_observatory", lambda *a, **k: cagrildi.append(1) or {})
    assert sb.evaluate("AAPL", _df(), _sig()) is None
    assert cagrildi == []


def test_advise_karari_degistirmez():
    v = sb.evaluate("AAPL", _df(), _sig(), live_price=100.2)
    assert v is not None
    assert v.mode == "advise"
    assert v.veto is False
    assert v.size_mult == 1.0


def test_advise_olcumleri_gerekceye_yazar():
    v = sb.evaluate("AAPL", _df(), _sig(), live_price=100.2)
    assert v.reason.startswith("SHADOW[advise]")
    # kovalama sapması canlı fiyattan türetilir ve LONG'da POZİTİF olmalı (yukarı kovaladık)
    assert v.marks["chase_drift_bps"] == pytest.approx(20.0, abs=0.2)


def test_short_tarafinda_sapma_isareti_dogru():
    """SHORT'ta fiyat DÜŞERSE kovalanmış olur → sapma POZİTİF (slip_bps tanımıyla aynı)."""
    v = sb.evaluate("AAPL", _df(), _sig(SignalAction.SHORT), live_price=99.8)
    assert v.marks["chase_drift_bps"] == pytest.approx(20.0, abs=0.2)


def test_canli_fiyat_yoksa_sapma_uydurulmaz():
    """Ölçüm YOKLUĞU ölçüm gibi raporlanmaz — anahtar HİÇ yazılmaz (0.0 değil)."""
    v = sb.evaluate("AAPL", _df(), _sig(), live_price=None)
    assert "chase_drift_bps" not in v.marks


# ------------------------------------------------------------------ 3) TEK YÖNLÜ MÜ
@pytest.mark.parametrize("action", [SignalAction.WAIT, SignalAction.EXIT])
def test_cikis_ve_bekleme_yoluna_baglanmaz(action, monkeypatch):
    """Bir gölge ölçünün zarar-kesmeyi geciktirmesi en pahalı hata olurdu."""
    cagrildi = []
    monkeypatch.setattr(sb, "_observatory", lambda *a, **k: cagrildi.append(1) or {})
    assert sb.evaluate("AAPL", _df(), _sig(action)) is None
    assert cagrildi == []


def test_size_mult_tavani_bir(monkeypatch):
    monkeypatch.setenv("SHADOW_BRIDGE", "enforce")
    v = sb.evaluate("AAPL", _df(), _sig(), live_price=100.1)
    assert v.size_mult <= 1.0


def test_verdikt_yon_alani_tasimaz():
    """Köprü YÖN ÜRETMEZ: verdiktte bir 'action'/'side' alanı olmamalı."""
    alanlar = set(sb.ShadowVerdict.__dataclass_fields__)
    assert not (alanlar & {"action", "side", "direction"})


# --------------------------------------------------------- 4) TERFİ DEFTERİ İŞLİYOR MU
def test_bugun_terfi_etmis_kural_yok():
    """Ölçüm 2026-08-31'de hiçbir yüzeyi geçirmedi; defter bunu TAŞIMALI."""
    assert sb.PROMOTED == ()
    assert set(sb.RULES) >= {"meta_pwin", "dir_bar_agree", "neural_pwin", "mkt_health"}
    assert sb.RULES["meta_pwin"][0] == "INVERTED"
    assert sb.RULES["dir_bar_agree"][0] == "SLIP_ARTIFACT"


def test_enforce_terfisiz_kuralla_hukum_vermez(monkeypatch, caplog):
    """PROMOTE taşımayan defterle enforce, advise gibi davranmalı VE nedenini söylemeli."""
    monkeypatch.setenv("SHADOW_BRIDGE", "enforce")
    v = sb.evaluate("AAPL", _df(), _sig(), live_price=100.1)
    assert v.mode == "enforce"
    assert v.veto is False and v.size_mult == 1.0


def test_defter_promote_derse_uygulama_zorunlu(monkeypatch):
    """Deftere PROMOTE yazıp uygulamayı UNUTMAK sessiz kalamaz.

    Bu, defterin dekoratif olmasına karşı kilittir: bir sonraki tur bir kuralı terfi
    ettirirse, onu KARARA bağlamadan yeşil geçemez."""
    monkeypatch.setattr(sb, "PROMOTED", ("mkt_health",))
    with pytest.raises(AssertionError, match="uygulaması yok"):
        sb._enforce_hukmu({"health": 90})


# ---------------------------------------------------------------- fail-soft + durum
def test_yuzey_patlarsa_tarama_dusmez(monkeypatch):
    def patla(*_a, **_k):
        raise RuntimeError("bilerek")
    monkeypatch.setattr(sb, "_observatory", patla)
    v = sb.evaluate("AAPL", _df(), _sig(), live_price=100.1)
    assert v is not None                       # verdikt YİNE döner
    assert v.veto is False


def test_sessiz_olum_yok(monkeypatch):
    """Hata yutulur ama İZİ kalır (Tur-44: zarif degrade evet, sessiz degrade hayır)."""
    def patla(*_a, **_k):
        raise RuntimeError("bilerek")
    monkeypatch.setattr(sb, "bar_shadow", patla, raising=False)
    monkeypatch.setattr("nihai.ai.observatory.bar_shadow", patla)
    sb.evaluate("AAPL", _df(), _sig(), live_price=100.1)
    assert sb.status()["last_error"] and "bilerek" in sb.status()["last_error"]


def test_neural_ayri_bayrak_ister(monkeypatch):
    """torch importu VARSAYILAN olarak tetiklenmemeli (Tur-26 import hijyeni)."""
    cagrildi = []
    monkeypatch.setattr(sb, "_envb", lambda n, d=False: cagrildi.append(n) or False)
    sb.evaluate("AAPL", _df(), _sig(), live_price=100.1)
    assert "SHADOW_BRIDGE_NEURAL" in cagrildi


def test_status_terfi_defterini_yayinlar():
    st = sb.status()
    assert st["mode"] == "advise"
    assert st["promoted"] == []
    assert st["rules"]["meta_pwin"] == "INVERTED"


# ------------------------------------------------------------------- 1) BAĞLANDI MI
def _scan_kaynagi() -> str:
    return MAIN_PY.read_text(encoding="utf-8")


def test_main_kopruyu_cagiriyor():
    """Üretim çağıranı OLMAYAN köprü, köprü değildir (Tur-37 kuralı)."""
    src = _scan_kaynagi()
    agac = ast.parse(src)
    cagrilar = [n for n in ast.walk(agac)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "evaluate"
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "shadow_bridge"]
    assert len(cagrilar) == 1, "köprü tam olarak BİR yerden çağrılmalı"


def test_kopru_determinist_guardlardan_SONRA():
    """AI köprüsüyle aynı duruş: gölge, kuralların YERİNE değil ARDINA konur.

    Çapa (`reanchor_to_live`) ve kill-switch/seans guard'ları köprüden ÖNCE olmalı;
    sinyal-only uyarı (`notify_signal`) köprüden SONRA — veto uyarıyı da susturabilsin."""
    src = _scan_kaynagi()
    i_capa = src.index("reanchor_to_live(")
    i_kopru = src.index("shadow_bridge.evaluate(")
    i_uyari = src.index("telegram_bot.notify_signal(")
    assert i_capa < i_kopru < i_uyari


def test_carpan_boyutlamaya_giriyor():
    """`size_mult` okunmadan taşınırsa terfi eden ilk kural sessizce etkisiz kalırdı."""
    assert "conf_mult *= sh_v.size_mult" in _scan_kaynagi()


def test_olculer_metadataya_ve_gerekceye_dusuyor():
    src = _scan_kaynagi()
    assert 'sig.metadata["shadow"] = sh_v.marks' in src
    assert "sig.reasons.append(sh_v.reason)" in src


def test_olculer_predictions_loga_sizmiyor():
    """Tarama yolu HTF barıyla çalışır; 1m gölge kolonlarıyla AYNI havuza dökülemez
    (yoksa `shadow_promote`ın ölçtüğü dağılım kirlenirdi).

    Kapı KODA bakar, ANLATIYA değil: modül `prediction_tracker`ı docstring'inde
    ALINTILIYOR (slip_bps tanımı oradan gelir) ve ham metin arayan bir kapı kendi
    açıklamasına takılırdı (Tur-49 dersi) → `ast` ile İMPORT'lar denetlenir."""
    agac = ast.parse(inspect.getsource(sb))
    moduller = set()
    for n in ast.walk(agac):
        if isinstance(n, ast.Import):
            moduller.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            moduller.add(n.module)
    assert not any("prediction_tracker" in m for m in moduller), moduller
