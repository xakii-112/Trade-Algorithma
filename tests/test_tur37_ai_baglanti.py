"""
test_tur37_ai_baglanti.py — Tur-37: `nihai/ai` ↔ üretim KÖPRÜLERİ + ANAYASA kilitleri.

İki soru sorar ve ikisini de kilitler:

  1. **BAĞLANDI MI?** Denetimde "tek satır `nihai.ai` importu yok" denen her yüzey artık
     köprüyü çağırıyor mu, ve yazılıp hiç çağrılmayan beş modülün ÜRETİM ÇAĞIRANI var mı?
     (Bir köprü "yazıldı ama kimse çağırmıyor" ise kusurun kendisini tekrar etmiş olurduk.)

  2. **ANAYASA BOZULDU MU?** Bağlanan hiçbir değer karar yüzeyine sızmıyor mu?
     `_obs` satırdan düşürülüyor mu, `compute_regime` gövdesinde AI importu var mı,
     `regime_multiplier` gölge okumadan etkileniyor mu?

Ağ YOK, torch YOK, disk yazımı YOK — hepsi sentetik df ve saf fonksiyonlar üstünde.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nihai.ai import observatory


def _bars(n: int = 260, seed: int = 11) -> pd.DataFrame:
    """İki rejimli sentetik OHLCV — HMM'in ayırt edecek bir şeyi olsun."""
    rng = np.random.default_rng(seed)
    vol = np.where(np.arange(n) < n // 2, 0.004, 0.012)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, vol)))
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": close, "high": close * 1.003, "low": close * 0.997,
                         "close": close, "volume": 1e6}, index=idx)


# ---------------------------------------------------------------------------
# 1) Köprünün kendisi
# ---------------------------------------------------------------------------
def test_bar_shadow_uc_aileyi_de_uretir():
    """Mikroyapı + rejim-HMM + veri sağlığı: üçü de tek çağrıda ölçülür."""
    out = observatory.bar_shadow(_bars(), "TEST")
    assert out is not None
    assert out["har_rv_fc"] > 0                      # microstructure (HAR-RV öngörüsü)
    assert 0.0 <= out["hmm_mode_prob"] <= 1.0        # regime_hmm (nedensel)
    assert 0.0 <= out["hmm_switch_prob"] <= 1.0
    assert 0.0 <= out["health"] <= 100.0             # ai/data_health (data_provider köprüsü)


def test_rv_ve_har_ayni_birimde():
    """`rv` ve `har_rv_fc` AYNI tanımda (Σr²) olmalı — biri ortalama olsaydı öngörü
    pencere boyu kadar (20×) düşük görünür ve 'model bozuk' diye YANLIŞ teşhis edilirdi."""
    out = observatory.bar_shadow(_bars(), "TEST")
    assert out is not None
    assert 0.1 < out["har_rv_fc"] / out["rv"] < 10.0


def test_determinizm_ve_hmm_cache_ciktiyi_bozmaz():
    """Anayasa madde 6: aynı girdi → aynı çıktı. HMM cache'i bunu bozmamalı."""
    df = _bars()
    assert observatory.bar_shadow(df, "DET") == observatory.bar_shadow(df, "DET")


@pytest.mark.parametrize("bad", [None, "df değil", 42])
def test_bozuk_girdi_istisna_firlatmaz(bad):
    assert observatory.bar_shadow(bad, "X") is None


def test_kisa_seri_none_doner():
    """Uydurma yok: 30 getiriden az veri → ölçüm YAPILMAZ."""
    assert observatory.bar_shadow(_bars(20), "X") is None


def test_kapali_env_olcumu_durdurur(monkeypatch):
    monkeypatch.setenv("AI_OBSERVATORY", "off")
    assert observatory.bar_shadow(_bars(), "X") is None


def test_olculemeyen_makas_bos_kalir():
    """Corwin-Schultz yansız tahmini pozitif değilse `spread_bps` YAZILMAZ.

    Tur-31/33 dersi: ölçüm YOKLUĞU bir ölçüm gibi raporlanamaz. Sabit yüzdeli high/low
    (dejenere) tam olarak bu vakayı üretir."""
    df = _bars()
    df["high"] = df["close"] * 1.002
    df["low"] = df["close"] * 0.998
    out = observatory.bar_shadow(df, "FLAT")
    assert out is not None
    assert "spread_bps" not in out


def test_calibration_line_artefaktsiz_da_durust():
    line = observatory.calibration_line()
    assert isinstance(line, str) and line


# ---------------------------------------------------------------------------
# 2) "Yazıldı ama çağrılmıyor" tekrar etmesin: HER köprünün üretim çağıranı var mı?
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1] / "nihai"

# (çağrılan ad, o adı ÇAĞIRMASI gereken üretim dosyaları)
_KOPRULER = [
    ("bar_shadow", ["execution/live_predict_service.py", "signals/market_regime.py"]),
    ("bars_health", ["ai/observatory.py"]),
    ("regime_shadow", ["web/api_extra.py"]),
    ("calibration_line", ["notifier/terminal_dashboard.py", "web/tearsheet.py"]),
    ("ai_status_text", ["notifier/telegram_bot.py"]),
    ("refresh_calibration", ["notifier/daily_summary.py"]),
    ("_greeks_without_vollib", ["risk/options_analyzer.py"]),
    ("_obs_cells", ["execution/prediction_tracker.py"]),
]


@pytest.mark.parametrize(("isim", "cagiranlar"), _KOPRULER)
def test_koprunun_uretim_cagirani_var(isim, cagiranlar):
    """Bir köprü fonksiyonu üretimde ÇAĞRILMIYORSA denetimin bulduğu kusuru tekrar etmiş
    oluruz — bu test tam olarak o tekrarı engeller."""
    for rel in cagiranlar:
        assert f"{isim}(" in (_ROOT / rel).read_text(encoding="utf-8"), \
            f"{rel} artık {isim}() çağırmıyor — köprü öldü"


def test_denetimdeki_yuzeyler_ai_katmanina_bagli():
    """Denetimde 'tek satır nihai.ai importu yok' denen her yüzeyde artık bağ VAR."""
    for rel in ("signals/market_regime.py", "risk/options_analyzer.py",
                "execution/live_predict_service.py", "data/data_provider.py",
                "notifier/telegram_bot.py", "notifier/terminal_dashboard.py",
                "web/tearsheet.py"):
        assert "nihai.ai" in (_ROOT / rel).read_text(encoding="utf-8"), f"{rel} hâlâ bağsız"


# ---------------------------------------------------------------------------
# 3) ANAYASA — bağlandı ama karara GİRMEDİ
# ---------------------------------------------------------------------------
def test_obs_satirdan_dusuruluyor():
    """`_obs` yalnız disk-loga gider; `_neural`/`_dirbar` ile AYNI sözleşme.

    Düşürülmezse gölge ölçüm web JSON'una ve pano tablosuna sızar — operatör onu ikinci
    bir sinyal sanabilir (neural_shadow'un panoya çıkmama gerekçesiyle aynı)."""
    src = (_ROOT / "execution/live_predict_service.py").read_text(encoding="utf-8")
    assert 'r.pop("_obs", None)' in src


def test_compute_regime_govdesinde_ai_yok():
    """Rejim ÇARPANINI üreten fonksiyon (karar yolu) AI katmanına dokunmamalı.

    Değerin kullanılmaması yetmez: karar fonksiyonunun gövdesindeki bir AI importu
    sonraki düzenleyici için 'buradan besleyebilirim' davetidir."""
    from nihai.signals.market_regime import compute_regime
    assert "nihai.ai" not in inspect.getsource(compute_regime)


def test_regime_shadow_carpani_degistirmez():
    """Gölge okuma `compute_regime`'in çıktısını hiçbir şekilde etkilemez."""
    from nihai.signals.market_regime import compute_regime, regime_shadow
    df = _bars(300)
    once = compute_regime(df)
    regime_shadow(df)
    sonra = compute_regime(df)
    assert (once.regime, once.multiplier) == (sonra.regime, sonra.multiplier)


def test_observatory_execution_ve_signals_import_etmez():
    """`nihai/ai` yukarı doğru (execution/signals/strategy) bağlanamaz — tek yön aşağıdır.

    `data_provider` istisnası BİLİNÇLİ ve tek yönlüdür: gölge ölçüm sağlayıcının bar
    sağlık karnesini OKUR, sağlayıcıya hiçbir şey yaptırmaz."""
    tree = ast.parse((_ROOT / "ai/observatory.py").read_text(encoding="utf-8"))
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    yasak = {m for m in modules
             if m.startswith(("nihai.execution", "nihai.signals", "nihai.risk"))}
    assert not yasak, f"anayasa ihlali: nihai.ai → {yasak}"


def test_hmm_label_mode_prob_olmadan_yazilmaz():
    """`prediction_tracker._obs_cells`: paydası okunmadan bir tepe rejim ilan edilemez
    (Tur-33'ün 'bir ORAN, paydası okunmadan okunamaz' kuralının buradaki karşılığı)."""
    from nihai.execution.prediction_tracker import _obs_cells
    assert _obs_cells({"_obs": {"hmm_label": "up_low"}}) == {}
    dolu = _obs_cells({"_obs": {"hmm_label": "up_low", "hmm_mode_prob": 0.81}})
    assert dolu == {"hmm_mode_prob": "0.81", "hmm_label": "up_low"}


@pytest.mark.parametrize("row", [{}, {"_obs": None}, {"_obs": "bozuk"},
                                 {"_obs": {"health": "sayı değil"}},
                                 {"_obs": {"har_rv_fc": float("nan")}}])
def test_obs_cells_bozuk_girdide_kaydi_engellemez(row):
    """Gölge katman tahmin satırından DAHA AZ önemlidir — bozuksa hücre boş kalır, kayıt sürer."""
    from nihai.execution.prediction_tracker import _obs_cells
    assert _obs_cells(row) == {}


def test_obs_kolonlari_semada_var():
    from nihai.execution.prediction_tracker import FIELDS
    for col in ("mkt_health", "har_rv_fc", "spread_bps",
                "hmm_mode_prob", "hmm_switch", "hmm_label"):
        assert col in FIELDS


def _satir(symbol="AAPL", bar_ts="2026-07-30T14:00:00", **extra):
    row = {"symbol": symbol, "bar_ts": bar_ts, "direction": "+", "action": "BUY",
           "confidence": 0.61, "prob_cal": 0.55, "price": 101.5, "bar_close": 101.0,
           "pwin": None, "pwin_side": None}
    row.update(extra)
    return row


def _son_satir(path):
    import csv
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))[-1]


def test_record_obs_kolonlarini_diske_yazar():
    """UÇTAN UCA: `_obs` → `record()` → CSV. Zincirin herhangi bir halkası kopsa gölge
    ölçüm sessizce kaybolur ve kimse fark etmez (kolonlar zaten boş kalabiliyor)."""
    from nihai.execution import prediction_tracker as pt

    obs = {"health": 93.4, "har_rv_fc": 0.00012345, "spread_bps": 4.2,
           "hmm_mode_prob": 0.77, "hmm_switch_prob": 0.23, "hmm_label": "up_low"}
    assert pt.record([_satir(symbol="OBSX", _obs=obs)], "1m") == 1
    row = _son_satir(pt.PRED_LOG)
    assert row["mkt_health"] == "93.4" and row["spread_bps"] == "4.2"
    assert row["hmm_mode_prob"] == "0.77" and row["hmm_label"] == "up_low"
    assert float(row["har_rv_fc"]) == pytest.approx(0.00012345)
    # ANAYASA: yürürlükteki yön/güven DEĞİŞMEDİ — gölge ölçüm karar sütunlarına dokunmaz.
    assert row["direction"] == "+" and row["confidence"] == "0.61"


def test_record_obs_yoksa_kayit_yine_yazilir():
    from nihai.execution import prediction_tracker as pt

    assert pt.record([_satir(symbol="NOOBS")], "1m") == 1
    row = _son_satir(pt.PRED_LOG)
    assert row["mkt_health"] == "" and row["hmm_label"] == ""


# ---------------------------------------------------------------------------
# 4) Greekler: uydurma sabitler gitti, gerçek Black-Scholes geldi
# ---------------------------------------------------------------------------
def test_vollibsiz_greekler_vollib_ile_birebir():
    """`_greeks_without_vollib` vollib dalıyla AYNI birimde ve aynı sayıyı vermeli.

    Dönüşüm (theta/365, vega/100) atlanırsa iki yol 365 ve 100 kat ayrışır — sessiz ve
    ölümcül bir tutarsızlık. Bu test o dönüşümü kilitler."""
    from nihai.risk import options_analyzer as oa
    if not oa.HAS_VOLLIB:
        pytest.skip("vollib yok — karşılaştırılacak referans dal mevcut değil")
    kw = {"option_type": "c", "S": 100.0, "K": 105.0, "T": 0.25, "r": 0.05}
    ref = oa.compute_greeks(**kw, sigma=0.30)
    ai = oa._greeks_without_vollib(**kw, market_price=None, sigma=0.30)
    assert ref is not None and ai is not None
    for alan in ("delta", "gamma", "theta", "vega", "iv"):
        assert getattr(ai, alan) == pytest.approx(getattr(ref, alan), abs=1e-9)


def test_vollibsiz_greekler_uydurma_sabit_dondurmez():
    """ESKİ davranış `delta=±0.5, gamma=0.02, theta=-0.05, vega=0.10` idi ve S/K/T/σ'ya
    HİÇ bakmıyordu. Derin OTM bir kontrat artık kendi Greeklerini almalı."""
    from nihai.risk.options_analyzer import _greeks_without_vollib
    otm = _greeks_without_vollib(option_type="c", S=100.0, K=200.0, T=0.05, r=0.05,
                                 market_price=None, sigma=0.30)
    assert otm is not None
    assert otm.delta < 0.05          # eski sabit 0.5 idi
    assert otm.gamma != 0.02


def test_cozulemeyen_iv_none_doner():
    """Yanlış sayı, sayı yokluğundan kötüdür: IV kökü yoksa uydurma sabite DÜŞÜLMEZ."""
    from nihai.risk.options_analyzer import _greeks_without_vollib
    assert _greeks_without_vollib(option_type="c", S=100.0, K=105.0, T=0.25, r=0.05,
                                  market_price=None, sigma=None) is None


# ---------------------------------------------------------------------------
# 5) Rapor yüzeyleri
# ---------------------------------------------------------------------------
def test_tearsheet_eki_html_kacisi_yapar():
    from nihai.web.tearsheet import _ai_appendix_html, _with_ai_appendix
    blok = _ai_appendix_html()
    assert "<script" not in blok
    sayfa = _with_ai_appendix("<html><body>x</body></html>")
    assert sayfa.index("KARAR-DIŞI") < sayfa.index("</body>")


def test_tearsheet_eki_body_yoksa_da_calisir():
    from nihai.web.tearsheet import _with_ai_appendix
    assert _with_ai_appendix("düz metin").startswith("düz metin")


def test_telegram_karnesi_sinira_kirpilir(monkeypatch):
    from nihai.notifier import telegram_bot
    monkeypatch.setattr(observatory, "ai_status_text", lambda: "x" * 99_999)
    out = telegram_bot.ai_status_text()
    assert len(out) < 4096 and out.endswith("(kırpıldı)")


def test_telegram_karnesi_ai_cokse_de_metin_doner(monkeypatch):
    from nihai.notifier import telegram_bot

    def _patla():
        raise RuntimeError("AI katmanı yok")

    monkeypatch.setattr(observatory, "ai_status_text", _patla)
    assert "okunamadı" in telegram_bot.ai_status_text()


def test_pano_seridi_ttl_ile_onbelleklenir(monkeypatch):
    """Pano saniyede 2 kez render eder; şerit her karede diskten okunmamalı."""
    from nihai.notifier import terminal_dashboard as td

    sayac = {"n": 0}

    def _say():
        sayac["n"] += 1
        return "şerit"

    monkeypatch.setattr(td, "_AI_STRIP", (0.0, ""))
    monkeypatch.setattr(observatory, "calibration_line", _say)
    assert td._ai_strip() == td._ai_strip() == "şerit"
    assert sayac["n"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
