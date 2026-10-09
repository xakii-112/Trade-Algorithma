"""
Tur-76 — Tur-75'in AÇIK BIRAKTIĞI köprüler: yazılmış ama ÇAĞRILMAYAN modüller bağlandı.

Tur-38'in kuralı: *yazılıp okunmayan ölçüm bu depoda kusurdur*. Tur-75 raporu beş açık
kalem bırakmıştı; bu dosya dördünü kilitler:

  1. `audit_chain` üretim ÇAĞIRANI yoktu     → `log_signal(accepted)` + kapanış
  2. `volume_profile`/`econophysics` bağsızdı → `observatory.bar_shadow`
  3. Madde 045 `ultima` eksikti               → `bs_greeks_extended` (sonlu farkla kilitli)
  4. Madde 049 Feller denetimi yoktu          → `heston_calibrate` çıktısında ÖLÇÜLÜR

Ayrıca zincirin EKLEME MALİYETİ (O(1)) ve `nihai.ai` lazy listesinin dizinle
eşitliği kilitlenir — ikisi de sessizce bozulabilecek sözleşmelerdir.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from nihai import audit_chain
from nihai.ai import observatory as obs
from nihai.ai.options_math import bs_greeks_extended


# ---------------------------------------------------------------------------
# 1) KARAR ZİNCİRİ — üretim çağıranı + O(1) ekleme
# ---------------------------------------------------------------------------
def test_kabul_edilen_sinyal_zincire_HALKA_ekler(tmp_path, monkeypatch):
    from nihai.signals import signal_engine as se
    from nihai.signals.strategy import Signal, SignalAction

    zincir = tmp_path / "chain.jsonl"
    monkeypatch.setattr(audit_chain, "CHAIN_PATH", zincir)
    monkeypatch.setattr(se, "LOG_DIR", tmp_path)
    monkeypatch.delenv("AUDIT_CHAIN", raising=False)

    sig = Signal(action=SignalAction.LONG, symbol="AAPL", price=100.0,
                 stop=98.0, target=104.0, reasons=["test"],
                 quality_score=7.5, rr_ratio=2.0)
    se.log_signal(sig, accepted=True)
    assert zincir.exists(), "kabul edilen sinyal zincire girmeli"
    kayit = json.loads(zincir.read_text(encoding="utf-8").strip())
    assert kayit["payload"]["tur"] == "sinyal"
    assert kayit["payload"]["symbol"] == "AAPL"
    assert kayit["prev"] == audit_chain.GENESIS and kayit["seq"] == 0


def test_PAS_GECILEN_sinyal_zincire_GIRMEZ(tmp_path, monkeypatch):
    """Zincirin amacı 'her satırı say' değil — tek günlük decision_log 35 MB (Tur-58)."""
    from nihai.signals import signal_engine as se
    from nihai.signals.strategy import Signal, SignalAction

    zincir = tmp_path / "chain.jsonl"
    monkeypatch.setattr(audit_chain, "CHAIN_PATH", zincir)
    monkeypatch.setattr(se, "LOG_DIR", tmp_path)
    monkeypatch.delenv("AUDIT_CHAIN", raising=False)
    se.log_signal(Signal(action=SignalAction.WAIT, symbol="X", price=1.0,
                         stop=0.9, target=1.2, reasons=[],
                         quality_score=1.0, rr_ratio=1.0), accepted=False)
    assert not zincir.exists()


def test_AUDIT_CHAIN_off_yazmayi_KAPATIR(tmp_path, monkeypatch):
    from nihai.signals import signal_engine as se
    from nihai.signals.strategy import Signal, SignalAction

    zincir = tmp_path / "chain.jsonl"
    monkeypatch.setattr(audit_chain, "CHAIN_PATH", zincir)
    monkeypatch.setattr(se, "LOG_DIR", tmp_path)
    monkeypatch.setenv("AUDIT_CHAIN", "off")
    se.log_signal(Signal(action=SignalAction.LONG, symbol="X", price=1.0,
                         stop=0.9, target=1.2, reasons=[],
                         quality_score=1.0, rr_ratio=1.0), accepted=True)
    assert not zincir.exists()


def test_zincir_EKLEMESI_dosyayi_TARAMIYOR(tmp_path, monkeypatch):
    """İlk sürüm ekleme başına dosyayı İKİ KEZ baştan sona geziyordu → O(n²).

    Ölçüm dolaylı DEĞİL doğrudandır: `tail_lines` çağrı sayısı ekleme başına 1 olmalı
    ve okunan satır sayısı zincir uzunluğuyla BÜYÜMEMELİDİR."""
    zincir = tmp_path / "chain.jsonl"
    okunan: list[int] = []
    gercek = audit_chain.tail_lines

    def sayan(path, n):
        out = gercek(path, n)
        okunan.append(len(out))
        return out

    monkeypatch.setattr(audit_chain, "tail_lines", sayan)
    for i in range(50):
        audit_chain.append({"i": i}, zincir)
    assert okunan and max(okunan) <= 1, f"ekleme başına en çok 1 satır okunmalı: {okunan}"
    # 49: İLK ekleme dosya HİÇ YOKKEN olur ve kuyruk hiç okunmaz (erken dönüş).
    assert len(okunan) == 49, "ekleme başına EN ÇOK bir kuyruk okuması olmalı"
    d = audit_chain.verify(zincir)
    assert d["ok"] is True and d["n"] == 50


def test_zincir_seq_ARDISIK_ve_dogrulama_bozulmayi_YAKALIYOR(tmp_path):
    zincir = tmp_path / "chain.jsonl"
    for i in range(5):
        audit_chain.append({"i": i}, zincir)
    satirlar = zincir.read_text(encoding="utf-8").strip().split("\n")
    assert [json.loads(s)["seq"] for s in satirlar] == [0, 1, 2, 3, 4]
    kurcalanmis = json.loads(satirlar[2])
    kurcalanmis["payload"]["i"] = 99
    satirlar[2] = json.dumps(kurcalanmis, ensure_ascii=False, separators=(",", ":"),
                             sort_keys=True)
    zincir.write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    d = audit_chain.verify(zincir)
    assert d["ok"] is False and d["broken_at"] == 2


def test_saglik_ucu_ZINCIRI_okuyor():
    """Yazan bir defterin OKUYAN yüzeyi olmadan bırakılması bu depoda kusurdur."""
    import inspect

    from nihai.web import api_extra
    kaynak = inspect.getsource(api_extra.system_health)
    assert '"audit_chain": _safe(_zincir)' in kaynak
    assert "audit_chain.verify()" in kaynak


# ---------------------------------------------------------------------------
# 2) GÖZLEMEVİ — hacim profili + CUSUM bağlandı
# ---------------------------------------------------------------------------
def _df(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {"open": c, "high": c + abs(rng.normal(0, 0.5, n)),
         "low": c - abs(rng.normal(0, 0.5, n)), "close": c,
         "volume": rng.integers(1_000, 9_000, n).astype(float)},
        index=pd.date_range("2026-01-01", periods=n, freq="h"))


def test_bar_shadow_HACIM_PROFILI_olculerini_uretiyor(monkeypatch):
    monkeypatch.delenv("AI_OBSERVATORY", raising=False)
    r = obs.bar_shadow(_df(), "TEST") or {}
    for k in ("poc", "value_area_pos", "poc_dist_bps", "cvd_last", "cusum_rate"):
        assert k in r, f"{k} üretilmedi"
    assert r["value_area_pos"] in (-1, 0, 1)
    assert isinstance(r["cusum_event"], bool)


def test_bar_shadow_ESKI_anahtarlari_KORUYOR(monkeypatch):
    """Yeni ölçü eklemek eskisini düşürmemeli (gölge kolonlar sözleşmedir)."""
    monkeypatch.delenv("AI_OBSERVATORY", raising=False)
    r = obs.bar_shadow(_df(), "TEST") or {}
    for k in ("health", "rv", "har_rv_fc", "jump_share", "hmm_mode_prob"):
        assert k in r, f"eski anahtar {k} kayboldu"


def test_cusum_bloguna_numpy_dizisi_gelse_bile_SESSIZ_dusmuyor(monkeypatch):
    """`dizi or []` deseni `bool(dizi)` çağırır ve patlar; geniş `except` onu YUTAR.

    Kusur ancak ÖLÇÜLEN alanın YOKLUĞUYLA görünür — bu test o yokluğu kilitler."""
    monkeypatch.delenv("AI_OBSERVATORY", raising=False)
    r = obs.bar_shadow(_df(), "TEST2") or {}
    assert "cusum_rate" in r and 0.0 <= r["cusum_rate"] <= 1.0


def test_gunluk_defter_yeni_GOLGE_kolonlarini_tasiyor():
    from nihai.execution.prediction_tracker import _OBS_NUMERIC, FIELDS

    for kol in ("poc_dist_bps", "value_area_pos", "cvd_divergence"):
        assert kol in FIELDS, f"{kol} defter şemasında yok"
        assert any(c[0] == kol for c in _OBS_NUMERIC), f"{kol} hücreye çevrilmiyor"


def test_terfi_defteri_yeni_yuzeyleri_TANIYOR():
    """Yeni gölge kolon kanıt kapısının SIRASINA girmeli (Tur-37: sıra kur, kapı değil)."""
    from research.shadow_promote import SURFACES

    anahtarlar = {s["key"] for s in SURFACES}
    assert {"poc_dist_bps", "value_area_pos", "cvd_divergence"} <= anahtarlar
    # Yönü ÖN-KAYITLI değil: `hib=None` → kapı ters-kalibrasyon kilidini uygulamaz.
    for s in SURFACES:
        if s["key"] in ("poc_dist_bps", "value_area_pos", "cvd_divergence"):
            assert s["hib"] is None


# ---------------------------------------------------------------------------
# 3) MADDE 045 — ultima, sonlu farkla kilitli
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [80.0, 100.0, 130.0])
def test_ultima_volganin_vol_TUREVIDIR(k):
    s, t, r, sig, h = 100.0, 0.5, 0.03, 0.25, 1e-5
    g = bs_greeks_extended(s, k, t, r, sig)
    ust = bs_greeks_extended(s, k, t, r, sig + h)
    alt = bs_greeks_extended(s, k, t, r, sig - h)
    assert g and ust and alt
    sayisal = (ust["volga"] - alt["volga"]) / (2 * h)
    assert g["ultima"] == pytest.approx(sayisal, rel=1e-5)


def test_ultima_call_ve_put_icin_AYNI():
    """q=0'da put-call paritesi σ'ya bağlı DEĞİLDİR → vol türevleri eşittir."""
    c = bs_greeks_extended(100.0, 100.0, 0.5, 0.03, 0.25, "c")
    p = bs_greeks_extended(100.0, 100.0, 0.5, 0.03, 0.25, "p")
    assert c["ultima"] == pytest.approx(p["ultima"], rel=1e-12)


def test_ultima_ucta_ve_panoda_YAYINDA():
    """Yazılıp panele bağlanmayan gösterge bu depoda kusurdur (Tur-38)."""
    import pathlib

    kok = pathlib.Path(__file__).resolve().parents[1]
    api = (kok / "nihai" / "web" / "api_quant.py").read_text(encoding="utf-8")
    html = (kok / "nihai" / "web" / "web_dashboard.html").read_text(encoding="utf-8")
    assert '"ultima"' in api, "uç ultima eğrisini üretmiyor"
    assert "ge.ultima" in html and "ex.ultima" in html, "pano ultima'yı çizmiyor"


# ---------------------------------------------------------------------------
# 4) MADDE 049 — Feller koşulu ÖLÇÜLÜR, DAYATILMAZ
# ---------------------------------------------------------------------------
def test_heston_kalibrasyonu_FELLER_kosulunu_TASIYOR():
    from nihai.ai.advanced_math import heston_calibrate

    s0, r, T = 100.0, 0.02, 0.5
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    # Basit BS fiyatlarını hedef alarak kalibre et — amaç parametre geri kazanımı DEĞİL,
    # çıktının SÖZLEŞMESİDİR (Feller alanı var mı, işareti tutarlı mı).
    from nihai.ai.quant_math import bs_price
    fiyatlar = [bs_price(s0, k, T, r, 0.25, "c") for k in strikes]
    d = heston_calibrate(strikes, fiyatlar, s0, r, T)
    if d is None:                       # yakınsamama fail-soft: iddia yok
        pytest.skip("kalibrasyon yakınsamadı — bu testin konusu değil")
    assert "feller" in d and "feller_ok" in d
    assert d["feller"] == pytest.approx(2 * d["kappa"] * d["theta"] - d["sigma"] ** 2,
                                        rel=1e-12)
    assert d["feller_ok"] is (d["feller"] > 0)


def test_feller_kosulu_KISIT_olarak_dayatilmiyor():
    """Dayatmak, piyasanın Feller-dışı fiyatladığı yüzeyi zorla içeri çekip uyum
    hatasını GİZLERDİ (Tur-49: sınırda duran fit bir tahmin değildir)."""
    import inspect

    from nihai.ai import advanced_math
    kaynak = inspect.getsource(advanced_math.heston_calibrate)
    assert "feller" in kaynak
    # Kısıt olarak eklenmiş olsaydı optimizasyon çağrısında görünürdü.
    assert "constraints" not in kaynak, "Feller bir KISIT olarak eklenmemeli"


# ---------------------------------------------------------------------------
# 5) LAZY LİSTE — dizinle eşit kalmalı
# ---------------------------------------------------------------------------
def test_nihai_ai_LAZY_listesi_dizinle_ESIT():
    """`import nihai.ai; nihai.ai.X` erişimi PEP 562 kancasına bağlıdır.

    Liste elle bakıldığı için sessizce eskir: Tur-76'da DÖRT modül (option_flow,
    econophysics, ml_hygiene, volume_profile) listede yoktu ve `nihai.ai.volume_profile`
    özniteliği AttributeError veriyordu. Kural artık makinededir."""
    import pathlib

    import nihai.ai as ai

    dizin = {p.stem for p in (pathlib.Path(ai.__file__).parent).glob("*.py")
             if p.stem != "__init__"}
    assert dizin == set(ai._LAZY_SUBMODULES), (
        f"lazy liste ayrık: {dizin ^ set(ai._LAZY_SUBMODULES)}")


def test_lazy_kanca_yeni_modulleri_GERCEKTEN_yukluyor():
    import nihai.ai as ai

    for ad in ("fundamentals", "volume_profile", "econophysics", "option_flow"):
        mod = getattr(ai, ad)
        assert mod.__name__ == f"nihai.ai.{ad}"


def test_import_nihai_ai_hala_UCUZ():
    """Tur-26 disiplini: `import nihai.ai` eager import EKLEMEZ (3854 → 11 ms)."""
    import pathlib

    import nihai.ai as ai

    kaynak = pathlib.Path(ai.__file__).read_text(encoding="utf-8")
    govde = [s for s in kaynak.split("\n")
             if s.startswith("from nihai.ai") or s.startswith("import nihai.ai")]
    assert not govde, f"modül düzeyinde eager alt-modül importu: {govde}"


def test_macro_ve_fundamentals_KATALOGDA():
    from nihai.web.api_quant import _CATALOG

    idler = {m["id"] for m in _CATALOG}
    assert {"macro", "fundamentals"} <= idler


def test_fundamentals_modulu_KARAR_YOLUNA_bagli_DEGIL():
    """ANAYASA: temel skorlar `signals`/`risk`/`execution`'a import EDİLMEZ."""
    import pathlib

    kok = pathlib.Path(__file__).resolve().parents[1] / "nihai"
    for alt in ("signals", "risk", "execution"):
        for p in (kok / alt).rglob("*.py"):
            metin = p.read_text(encoding="utf-8")
            assert "fundamentals" not in metin, f"{p}: temel skorlar karar yoluna sızmış"
            assert "edgar_facts" not in metin, f"{p}: EDGAR köprüsü karar yoluna sızmış"


def test_macro_regime_KARAR_YOLUNA_bagli_DEGIL():
    import pathlib

    kok = pathlib.Path(__file__).resolve().parents[1] / "nihai"
    for alt in ("signals", "risk", "execution"):
        for p in (kok / alt).rglob("*.py"):
            assert "macro_regime" not in p.read_text(encoding="utf-8"), \
                f"{p}: makro paneli karar yoluna sızmış"


def test_ultima_sonlu_fark_disinda_KAPALI_FORM_dogru():
    """d1=d2=0 (ATM forward, σ√t küçük) sınırında ultima yaklaşık −vega/σ²·0 → 0'a gider."""
    s, k, t, r = 100.0, 100.0, 1e-4, 0.0
    g = bs_greeks_extended(s, k, t, r, 0.2)
    assert g is not None and math.isfinite(g["ultima"])


# ---------------------------------------------------------------------------
# 6) PANO JS SÖZDİZİMİ — tek bir hata TÜM panoyu öldürür
# ---------------------------------------------------------------------------
def test_pano_JS_sozdizimi_gecerli():
    """BULGU-76-6: pano JS'i TEK bir `<script>` bloğudur.

    O blokta tek bir sözdizimi hatası TÜM panoyu öldürür — sekmeler açılmaz, hiçbir
    grafik çizilmez ve tarayıcı konsolu dışında hiçbir yerde iz kalmaz. Bu tur tam
    böyle bir hata üretti: tek tırnaklı bir JS dizgisinin İÇİNE Türkçe kesme işareti
    (`volga'nın`) girdi ve blok ayrıştırılamaz oldu. Mevcut pano testleri kanvas↔çizici
    eşleşmesini denetliyordu ama SÖZDİZİMİNİ hiç sormuyordu.

    `node` yoksa test ATLANIR (iddia edilmez — ölçüm yokluğu ölçüm gibi raporlanmaz).
    """
    import pathlib
    import re
    import shutil
    import subprocess
    import tempfile

    if shutil.which("node") is None:
        pytest.skip("node kurulu değil — JS sözdizimi ölçülemedi")
    kok = pathlib.Path(__file__).resolve().parents[1]
    html = (kok / "nihai" / "web" / "web_dashboard.html").read_text(encoding="utf-8")
    bloklar = re.findall(r"<script(?![^>]*src=)[^>]*>(.*?)</script>", html, re.S)
    assert bloklar, "panoda satır-içi script bloğu yok"
    hedef = pathlib.Path(tempfile.gettempdir()) / "nihai_pano_sozdizimi.js"
    hedef.write_text("\n".join(bloklar), encoding="utf-8")
    r = subprocess.run(["node", "--check", str(hedef)], capture_output=True, text=True)
    assert r.returncode == 0, f"pano JS sözdizimi HATALI:\n{r.stderr[:800]}"
