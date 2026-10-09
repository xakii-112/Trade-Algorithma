"""Tur-58 — dış rapor §E/§F/§G/§H maddelerinin kilidi.

Dört ayrı yüzey, tek soru: **eklenen şey gerçekten BAĞLI mı, ve varsayılanı GÜVENLİ mi?**

* Madde 25 — ex-temettü erken kullanım (`early_exercise_risk` + iki üretim çağıranı)
* Madde 29/53 — Section 1256 wash-sale muafiyeti + çapraz (hisse ↔ opsiyon) etiketi
* Madde 43 — ABD seansının YEREL penceresi (DST kayması ÖLÇÜLÜR)
* Madde 37/48 — hız bütçesi + `/metrics` (yumuşak tavan altında ek uyku TAM 0.0)
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nihai.data.market_calendar import ET, local_session_window
from nihai.observability import RateBudget, prometheus_text, record_loop, snapshot
from nihai.risk.options_analyzer import early_exercise_risk


# ===========================================================================
# Madde 25 — ex-temettü erken kullanım
# ===========================================================================
def _ee(**kw):
    base = dict(option_type="c", strike=100.0, underlying=110.0, option_mid=10.6,
                days_to_ex_div=1, dividend=0.50, symbol="AAPL")
    base.update(kw)
    return early_exercise_risk(**base)


def test_ekstrinsik_temettunun_altinda_risk_var():
    """Ekstrinsik 0.60 > D 0.50 → risk YOK; 0.20 < 0.50 → risk VAR."""
    yok = _ee(option_mid=10.60)          # içsel 10.0 → ekstrinsik 0.60 > 0.50
    assert yok["risk"] is False and yok["reason"] == "EXDIV_EXTRINSIC_COVERS_DIV"
    var = _ee(option_mid=10.20)          # ekstrinsik 0.20 < 0.50
    assert var["risk"] is True and var["measured"] is True
    assert var["reason"] == "EXDIV_EXTRINSIC_BELOW_DIV"
    assert var["extrinsic"] == pytest.approx(0.20)


def test_temettu_tutari_yoksa_olculmedi_bayragi():
    """Ölçüm YOKLUĞU ölçüm gibi raporlanmaz: risk işaretlenir ama measured=False."""
    r = _ee(dividend=None)
    assert r["risk"] is True and r["measured"] is False
    assert r["reason"] == "EXDIV_ITM_UNMEASURED"


def test_put_ve_otm_riskli_degil():
    assert _ee(option_type="p")["reason"] == "PUT_NO_EX_DIV_RISK"
    assert _ee(underlying=90.0)["reason"] == "OTM_NO_EARLY_EXERCISE"       # OTM call


def test_avrupa_nakit_uzlasili_yapisal_olarak_muaf():
    """SPX Avrupa tipi → erken kullanım YOKTUR; bu 'ölçemedik' DEĞİL 'olamaz'dır."""
    r = _ee(symbol="SPX")
    assert r["risk"] is False and r["measured"] is True
    assert r["reason"] == "EUROPEAN_NO_EARLY_EXERCISE"


def test_ex_div_tarihi_bilinmiyorsa_karar_yok():
    r = _ee(days_to_ex_div=None)
    assert r["risk"] is False and r["reason"] == "EX_DIV_DATE_UNKNOWN"


def test_pencere_disinda_risk_yok():
    assert _ee(days_to_ex_div=5)["risk"] is False


def test_uretim_cagiranlari_var():
    """Bu depoda yazılıp ÇAĞRILMAYAN fonksiyon kusurdur (Tur-37/38).

    `assignment_risk_short_call` yıllarca tam bu durumdaydı; yenisi iki üretim
    yolundan da çağrılmalı."""
    from pathlib import Path
    kok = Path(__file__).resolve().parents[1] / "nihai"
    sec = (kok / "risk" / "option_strategy.py").read_text(encoding="utf-8")
    pos = (kok / "risk" / "position_manager.py").read_text(encoding="utf-8")
    assert "early_exercise_risk(" in sec, "kontrat seçiminde çağrılmıyor"
    assert "early_exercise_risk(" in pos, "açık pozisyon yolunda çağrılmıyor"
    assert "_check_ex_dividend_exit" in pos and "update_all_with_quotes" in pos


def test_canli_yol_backtest_yolunu_kirletmiyor():
    """Ex-div kapısı YALNIZ canlı kotasyon yolundadır: takvim BUGÜNÜN tarihini bilir,
    backtest barına uygulamak look-ahead olurdu (Madde 3)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "nihai" / "risk"
           / "position_manager.py").read_text(encoding="utf-8")
    govde = src.split("def check_exit(")[1].split("\n    def ", 1)[0]
    assert "_check_ex_dividend_exit" not in govde
    assert "ex_dividend_context" not in govde


# ===========================================================================
# Madde 29/53 — Section 1256 + çapraz wash-sale
# ===========================================================================
def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("NIHAI_DATA_DIR", str(tmp_path))
    from nihai.execution.state import StateStore
    return StateStore()


def test_section_1256_wash_sale_muaf(tmp_path, monkeypatch):
    """SPX §1256 kapsamında piyasaya göre değerlenir → wash-sale UYGULANMAZ."""
    st = _store(tmp_path, monkeypatch)
    now = datetime(2026, 8, 20, 12, 0)
    st.register_close_for_wash_sale("SPX", -500.0, now, "option")
    hit, why = st.wash_sale_check("SPX", now + timedelta(days=3), "option")
    assert hit is False and why == "SECTION_1256_WASH_SALE_MUAF"


def test_capraz_wash_sale_etiketi(tmp_path, monkeypatch):
    """Hisse zararı + AYNI dayanakta opsiyon girişi = 'substantially identical' adayı."""
    st = _store(tmp_path, monkeypatch)
    now = datetime(2026, 8, 20, 12, 0)
    st.register_close_for_wash_sale("AAPL", -250.0, now, "stock")
    hit, why = st.wash_sale_check("AAPL", now + timedelta(days=3), "option")
    assert hit is True and why.startswith("WASH_SALE_CAPRAZ_AAPL")
    # aynı araçta ise düz etiket
    hit2, why2 = st.wash_sale_check("AAPL", now + timedelta(days=3), "stock")
    assert hit2 is True and why2.startswith("WASH_SALE_AAPL")


def test_arac_turu_bilinmiyorsa_capraz_iddia_edilmez(tmp_path, monkeypatch):
    """OPTION_MODE=both iken tür seçilemez → uydurma sınıflandırma YAPILMAZ."""
    st = _store(tmp_path, monkeypatch)
    now = datetime(2026, 8, 20, 12, 0)
    st.register_close_for_wash_sale("AAPL", -250.0, now, "stock")
    hit, why = st.wash_sale_check("AAPL", now + timedelta(days=3), "")
    assert hit is True and why.startswith("WASH_SALE_AAPL")


def test_eski_kayitlar_arac_turu_olmadan_okunabilir(tmp_path, monkeypatch):
    """Geriye uyum: `asset_type` alanı olmayan state.json kayıtları patlamamalı."""
    st = _store(tmp_path, monkeypatch)
    now = datetime(2026, 8, 20, 12, 0)
    st.loss_history = [{"symbol": "AAPL", "date": now.date().isoformat(), "loss": -100.0}]
    hit, why = st.wash_sale_check("AAPL", now + timedelta(days=1), "option")
    assert hit is True and "AAPL" in why


# ===========================================================================
# Madde 43 — DST / yerel seans penceresi
# ===========================================================================
def test_dst_kaymasi_gercekten_var():
    """Rapor §G'nin iddiası ÖLÇÜLÜR: yaz ile kış arasında 1 saatlik kayma."""
    yaz = local_session_window(datetime(2026, 8, 28, 12, tzinfo=ET))
    kis = local_session_window(datetime(2026, 1, 15, 12, tzinfo=ET))
    assert yaz and kis
    assert yaz["open_et"] == kis["open_et"] == "09:30"        # ET sabit
    assert yaz["open_local"] != kis["open_local"]             # YEREL kayıyor
    assert abs(yaz["offset_hours"] - kis["offset_hours"]) == pytest.approx(1.0)


def test_yarim_gun_seansi_kisa_olcusuyor():
    """Şükran Günü ertesi yarım gün: pencere 6.5 değil ~3.5 saat."""
    w = local_session_window(datetime(2026, 11, 27, 12, tzinfo=ET))
    assert w and w["duration_hours"] < 4.0


def test_kapali_gunde_pencere_uydurulmaz():
    assert local_session_window(datetime(2026, 8, 30, 12, tzinfo=ET)) is None   # Cumartesi


def test_gorev_scripti_saati_olcumden_turetiyor():
    from pathlib import Path
    ps1 = (Path(__file__).resolve().parents[1] / "tur17-shadow-gorev.ps1").read_text(
        encoding="utf-8", errors="replace")
    assert "market_calendar --at" in ps1
    assert '$At = "09:30"' not in ps1, "sabit yerel saat geri geldi (ABD seansı TR 16:30/17:30)"


# ===========================================================================
# Madde 37/48 — hız bütçesi + metrics
# ===========================================================================
def test_yumusak_tavan_altinda_ek_uyku_tam_sifir():
    """Bugünkü rejimde davranış BİT-DENK olmalı: kısma yalnız kenarda konuşur."""
    b = RateBudget(limit_per_min=200.0, soft_pct=80.0)
    b._rate = 10.0
    assert b.throttle_sleep(60.0) == 0.0
    b._rate = 160.0                       # tam yumuşak tavan
    assert b.throttle_sleep(60.0) == 0.0


def test_tavan_ustunde_fren_oransal_ve_sinirli():
    b = RateBudget(limit_per_min=200.0, soft_pct=80.0)
    b._rate = 240.0                       # tavanın 1.5 katı → +%50 döngü
    assert b.throttle_sleep(60.0) == pytest.approx(30.0)
    b._rate = 10_000.0                    # üst sınır: bir tam döngü (fren, durdurma değil)
    assert b.throttle_sleep(60.0) == 60.0


def test_olcum_yoklugu_sifir_degil():
    b = RateBudget()
    assert b.usage()["rate_per_min"] is None
    assert b.usage()["usage_pct"] is None
    assert b.throttle_sleep(60.0) == 0.0


def test_pencere_ilk_gozlemde_olcum_uretmez(monkeypatch):
    b = RateBudget()
    monkeypatch.setattr(b, "_counts", lambda: {"total": 100})
    assert b.observe() is None            # önceki nokta yok
    monkeypatch.setattr(b, "_counts", lambda: {"total": 160})
    b._last_mono -= 60.0                  # 60 sn geçmiş gibi
    assert b.observe() == pytest.approx(60.0, abs=1.0)


def test_prometheus_bicimi_ve_olculemeyen_alan():
    record_loop("test_loop", 1.25)
    txt = prometheus_text()
    assert "# TYPE nihai_loop_seconds gauge" in txt
    assert 'nihai_loop_seconds{loop="test_loop"} 1.25' in txt
    # Ölçülemeyen metrik BASILMAZ: 0 "sıfır"dır, "ölçüm yok" değil (Tur-31/33/56).
    b = RateBudget()
    assert b.usage()["usage_pct"] is None
    from nihai.observability import _line
    bos: list[str] = []
    _line(bos, "x_metric", "yardım", "gauge", None)
    assert bos == []
    assert snapshot()["uptime_sec"] >= 0


def test_metrics_ucu_bagli_ve_prometheus_donuyor():
    from fastapi.testclient import TestClient

    import nihai.web.web_dashboard as w
    r = TestClient(w.app).get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "nihai_uptime_seconds" in r.text


def test_ana_dongu_freni_cagiriyor():
    """Yazılıp ÇAĞRILMAYAN sınırlayıcı, hiç yazılmamış sınırlayıcıdır."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "nihai" / "execution"
           / "main.py").read_text(encoding="utf-8")
    assert "OBS.BUDGET.observe()" in src
    assert "OBS.BUDGET.throttle_sleep(60.0)" in src
    assert "time_mod.sleep(60 + _extra)" in src


# ===========================================================================
# Madde 44 — token karşılaştırması ve kaba kuvvet freni
# ===========================================================================
def test_token_sabit_zamanli_karsilastiriliyor():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "nihai" / "web"
           / "web_dashboard.py").read_text(encoding="utf-8")
    assert "hmac.compare_digest" in src
    assert 'request.headers.get("X-Auth-Token", "") != WEB_TOKEN' not in src


def test_kaba_kuvvet_kilidi(monkeypatch):
    import nihai.web.web_dashboard as w
    monkeypatch.setattr(w, "WEB_TOKEN", "dogru-token")
    w._auth_fails.clear()
    for _ in range(w._AUTH_FAIL_MAX):
        ok, why = w._auth_ok("yanlis", "1.2.3.4")
        assert ok is False
    assert why == "KILITLENDI"
    assert w._auth_ok("dogru-token", "1.2.3.4") == (False, "KILITLI")   # kilitliyken doğru token da geçmez
    assert w._auth_ok("dogru-token", "5.6.7.8")[0] is True              # başka IP etkilenmez
    w._auth_fails.clear()


def test_dogru_token_sayaci_sifirlar(monkeypatch):
    import nihai.web.web_dashboard as w
    monkeypatch.setattr(w, "WEB_TOKEN", "t")
    w._auth_fails.clear()
    w._auth_ok("x", "9.9.9.9")
    assert w._auth_ok("t", "9.9.9.9") == (True, "OK")
    assert "9.9.9.9" not in w._auth_fails


# ===========================================================================
# Tamamlama turu — akış sağlığı, döngü gecikmesi, pano yüzeyi, env belgesi
# ===========================================================================
class _Stream:
    def stream_status(self):
        return {"connected": True, "subscribed": ["A", "B"],
                "symbols_with_minutes": ["A"], "symbols_with_quotes": [],
                "symbols_with_trades": ["A", "B", "C"]}


class _Sarmal:
    def __init__(self, inner):
        self.inner = inner


def test_akis_karnesi_tek_kaynaktan_okunuyor():
    """Tur-35 dersi ÖNCEDEN uygulandı: gezinti `data_provider.stream_status`ta TEK yerde.

    `live_predict_service._find_stream_status` adı korunur ama kendi gezintisini YAZMAZ."""
    from pathlib import Path

    from nihai.data.data_provider import stream_status
    from nihai.execution.live_predict_service import _find_stream_status
    assert stream_status(_Sarmal(_Stream()))["connected"] is True
    assert _find_stream_status(_Sarmal(_Stream()))["connected"] is True
    src = (Path(__file__).resolve().parents[1] / "nihai" / "execution"
           / "live_predict_service.py").read_text(encoding="utf-8")
    govde = src.split("def _find_stream_status(provider)")[1].split("\n\n")[0]
    assert "while stack" not in govde, "ikinci bir zincir gezintisi kopyası doğdu"


def test_akis_kaydi_yoksa_metrik_BASILMAZ(monkeypatch):
    """'0 bağlı' ile 'akış sağlayıcısı kayıtlı değil' aynı şey DEĞİLDİR."""
    import nihai.observability as O
    monkeypatch.setattr(O, "_PROVIDER", None, raising=False)
    assert O.stream_health() is None
    assert "nihai_stream" not in O.prometheus_text()


def test_akis_kayitliysa_sayilar_uzunluktan_geliyor(monkeypatch):
    import nihai.observability as O
    monkeypatch.setattr(O, "_PROVIDER", _Sarmal(_Stream()), raising=False)
    satirlar = [x for x in O.prometheus_text().splitlines() if x.startswith("nihai_stream")]
    assert "nihai_stream_connected 1" in satirlar
    assert "nihai_stream_subscribed 2" in satirlar
    assert "nihai_stream_symbols_with_quotes 0" in satirlar       # boş liste = ÖLÇÜLDÜ, 0
    assert "nihai_stream_symbols_with_trades 3" in satirlar


def test_saglayici_kuran_iki_yer_kaydi_yapiyor():
    """`set_provider` çağrılmazsa akış metriği hiç doğmaz = yazılıp bağlanmamış kod."""
    from pathlib import Path
    kok = Path(__file__).resolve().parents[1] / "nihai"
    assert "OBS.set_provider(provider)" in (kok / "execution" / "main.py").read_text(encoding="utf-8")
    assert "set_provider(_FALLBACK_PROVIDER)" in (kok / "web" / "web_dashboard.py").read_text(
        encoding="utf-8")


def test_tahmin_dongusu_gecikmesi_de_olculuyor():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "nihai" / "execution"
           / "live_predict_service.py").read_text(encoding="utf-8")
    assert 'record_loop("predict", elapsed)' in src


def test_exdiv_pano_ucunda_ve_htmlde_var():
    """Backend'de var ama panoda yok = bu depoda kusurdur (Tur-18/29/30/56)."""
    from pathlib import Path
    kok = Path(__file__).resolve().parents[1] / "nihai" / "web"
    assert 'out["early_exercise"] = _safe(_exdiv)' in (kok / "api_extra.py").read_text(
        encoding="utf-8")
    html = (kok / "web_dashboard.html").read_text(encoding="utf-8")
    assert "exdivTxt(d.early_exercise)" in html and "function exdivTxt(" in html


def test_metrics_panoda_okunuyor():
    """`/metrics` yalnız Prometheus için değil, operatör kartları için de okunur."""
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "nihai" / "web"
            / "web_dashboard.html").read_text(encoding="utf-8")
    assert "function parseProm(" in html and "async function loadMetrics(" in html
    assert "loadMetrics();" in html                      # loadSystem içinden çağrılıyor
    for kart in ("sys-rate", "sys-stream", "sys-loop", "sys-proc"):
        assert f'id="{kart}"' in html, f"{kart} kartı yok"


def test_yeni_env_anahtarlari_belgelenmis():
    """Okunan ama belgelenmemiş bir env anahtarı, olmayan bir ayar kadar kötüdür."""
    from pathlib import Path
    env = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    for k in ("EXDIV_GUARD", "ALPACA_RATE_LIMIT_PER_MIN", "ALPACA_RATE_SOFT_PCT",
              "LOG_COMPRESS_DAYS"):
        assert k in env, f"{k} .env.example'da yok"
