"""Web dashboard testleri — parse_trade_log sözleşmesi, CORS kilidi ve salt-okunur uçlar.

NOT: Mutasyon uçları (close/flatten/open) gerçek StateStore'a dokunduğundan burada
kapsanmaz; bu dosya yan-etkisiz yüzeyleri (log/CONFIG okuma) ve sözleşmeleri doğrular.
"""
from fastapi.testclient import TestClient

import nihai.web.web_dashboard as wd

SEP = "=" * 70

client = TestClient(wd.app)


# ----------------------------------------------------- _bars_for_analysis (Madde 1)
class _FakeProvider:
    """get_bars çağrılarını kaydeder; lookback büyüdükçe daha çok bar döndürür
    (IEX'in dar pencerede az bar üretmesini taklit eder)."""

    def __init__(self, bars_by_lookback):
        self.bars_by_lookback = bars_by_lookback  # {lookback: bar_sayısı}
        self.calls = []

    def get_bars(self, sym, interval, lookback_bars=200):
        import pandas as pd
        self.calls.append(lookback_bars)
        n = self.bars_by_lookback.get(lookback_bars, 0)
        if n == 0:
            return pd.DataFrame()
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        return pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            index=idx,
        )


def test_bars_for_analysis_widens_window_when_intraday_short():
    # İlk dar pencere 40 bar (yetersiz) → 500'e genişleyince 80 bar (yeterli).
    prov = _FakeProvider({220: 40, 500: 80})
    df = wd._bars_for_analysis(prov, "AAPL", "5m", min_bars=60)
    assert len(df) >= 60
    assert prov.calls[0] == 220 and 500 in prov.calls  # genişletme denendi


def test_bars_for_analysis_no_retry_for_daily():
    # 1g için yeterli olmasa bile genişletme YAPMA (intraday'e özgü davranış).
    prov = _FakeProvider({220: 40})
    df = wd._bars_for_analysis(prov, "AAPL", "1d", min_bars=60)
    assert prov.calls == [220]
    assert len(df) == 40


# ----------------------------------------------------------- _order_plan (Madde 6)
def test_order_plan_long_has_bracket_and_levels():
    plan = wd._order_plan(price=100.0, atr_val=2.0, action="LONG", stop=95.0, target=110.0)
    assert plan["side"] == "BUY"
    assert plan["limit"]["price"] < 100.0          # long limit = pullback altta
    assert plan["stop_market"]["side"] == "SELL"   # çıkış ters yön
    assert plan["bracket"]["take_profit"]["price"] == 110.0
    assert plan["multi_leg_option"]["structure"] == "bull_call_spread"
    # sinyal-only güvencesi
    assert "SİNYAL-ONLY" in plan["disclaimer"]


def test_order_plan_short_inverts_sides():
    plan = wd._order_plan(price=50.0, atr_val=1.0, action="SHORT", stop=53.0, target=44.0)
    assert plan["side"] == "SELL"
    assert plan["limit"]["price"] > 50.0           # short limit = üstte
    assert plan["stop_market"]["side"] == "BUY"
    assert plan["multi_leg_option"]["structure"] == "bear_put_spread"


def test_order_plan_wait_returns_note_no_legs():
    plan = wd._order_plan(price=100.0, atr_val=2.0, action="WAIT", stop=None, target=None)
    assert "note" in plan and plan["legs"] == []


# ------------------------------------------------- /api/predictions/accuracy (Madde 3)
def test_prediction_accuracy_empty_log_is_honest(tmp_path, monkeypatch):
    # Log yokken uç çökmeden 'çözülmüş tahmin yok' sözleşmesi döndürmeli (resolve=false).
    import nihai.execution.prediction_tracker as pt
    monkeypatch.setattr(pt, "PRED_LOG", tmp_path / "nope.csv")
    r = client.get("/api/predictions/accuracy?resolve=false")
    assert r.status_code == 200
    body = r.json()
    assert body["resolved"] == 0
    assert "tracking_enabled" in body  # UI bu bayrakla 'takip kapalı' uyarısı gösterir


# --------------------------------------------------------------- parse_trade_log
_CSV_HEAD = (
    "exit_time,symbol,asset_type,side,strategy,entry_time,entry_price,exit_price,"
    "quantity,risk_dollars,planned_rr,realized_pnl,r_multiple,mae_pct,mfe_pct,"
    "exit_trigger,sector,nlv_after\n"
)


def test_parse_trade_log_missing_file_has_equity_curve(tmp_path, monkeypatch):
    # Dosya yoksa bile dönüş sözleşmesi equity_curve içermeli (frontend bunu okur).
    monkeypatch.setattr(wd, "DATA_DIR", tmp_path)
    out = wd.parse_trade_log()
    assert "equity_curve" in out and out["equity_curve"] == []
    assert out["stats"]["total_trades"] == 0


def test_parse_trade_log_parses_stats_and_equity(tmp_path, monkeypatch):
    csv = _CSV_HEAD + (
        "2026-01-02T10:00:00,AAPL,stock,LONG,S1,2026-01-01T10:00:00,10.0,11.5,100,"
        "50.00,2.00,150.00,3.000,-0.50,4.00,TARGET,TECH,1000150.00\n"
        "2026-01-03T10:00:00,TSLA,stock,SHORT,S1,2026-01-02T10:00:00,20.0,20.5,100,"
        "50.00,2.00,-50.00,-1.000,-2.00,0.00,STOP,AUTO,1000100.00\n"
    )
    (tmp_path / "trades.csv").write_text(csv, encoding="utf-8")
    monkeypatch.setattr(wd, "DATA_DIR", tmp_path)

    out = wd.parse_trade_log()
    s = out["stats"]
    assert s["total_trades"] == 2
    assert s["wins"] == 1 and s["losses"] == 1
    assert s["total_pnl"] == 100.0
    assert s["win_rate"] == 50.0
    assert s["profit_factor"] == 3.0          # 150 / 50
    # equity_curve kronolojik (reverse'ten önce kurulur): ilk işlem başta.
    assert [e["equity"] for e in out["equity_curve"]] == [1_000_150.0, 1_000_100.0]
    # Frontend'in bastığı zenginleştirilmiş alanlar (en yeni işlem başta).
    t = out["trades"][0]
    assert t["asset"] == "TSLA (SHORT Stock x 100)"
    assert t["duration"] == "1 Gun 0 Saat 0 Dakika"
    assert t["mae_mfe"] == "-2.0% / +0.0%"
    assert t["r_multiple"] == -1.0


def test_parse_trade_log_corrupt_row_falls_back_cleanly(tmp_path, monkeypatch):
    # realized_pnl'i olmayan (kapanmamış/bozuk) satır sayılmaz ama sözleşme bozulmaz.
    (tmp_path / "trades.csv").write_text(
        _CSV_HEAD + "2026-01-02T10:00:00,AAPL,stock,LONG,S1,,,,,,,,,,,,,\n", encoding="utf-8")
    monkeypatch.setattr(wd, "DATA_DIR", tmp_path)
    out = wd.parse_trade_log()
    assert out["stats"]["total_trades"] == 0
    assert out["trades"] == []
    assert out["equity_curve"] == []


def test_parse_trade_log_reads_what_the_real_writer_writes(tmp_path, monkeypatch):
    """Regresyon: defter sekmesi 330 kapanışta BOŞ görünüyordu çünkü ayrıştırıcı
    üretimdeki YAZANIN biçimini okuyamıyordu (fixture gerçek biçim değildi).
    Bu test yazanı (`log_trade_csv`) gerçekten çağırır → biçim ikisinde de değişirse
    ya da tek tarafta değişirse burası kırılır."""
    from datetime import datetime

    import nihai.signals.signal_engine as se

    monkeypatch.setattr(se, "TRADES_CSV", tmp_path / "trades.csv")
    monkeypatch.setattr(wd, "DATA_DIR", tmp_path)

    se.log_trade_csv(se.TradeRecord(
        asset_label="AAPL (LONG Stock x 10)", strategy="S1",
        entry_time=datetime(2026, 1, 1, 10, 0), entry_price=10.0,
        entry_greeks=None, entry_iv=None, entry_iv_rank=None, dte_at_entry=None,
        risk_dollars=100.0, planned_rr=2.0, contracts_or_shares=10,
        exit_time=datetime(2026, 1, 1, 12, 30), exit_price=12.0,
        exit_trigger="TARGET", mae_pct=-1.0, mfe_pct=25.0,
        realized_pnl=20.0, nlv_after=1_000_020.0,
    ))

    out = wd.parse_trade_log()
    assert out["stats"]["total_trades"] == 1, "yazanın yazdığını ayrıştırıcı okuyamadı"
    assert out["stats"]["total_pnl"] == 20.0
    assert out["trades"][0]["duration"] == "0 Gun 2 Saat 30 Dakika"


# ------------------------------------------------------------------------- CORS
def test_cors_locked_down_by_default():
    # '*' + allow_credentials geçersiz kombinasyon; varsayılan localhost'a kilitli olmalı.
    assert "*" not in wd.WEB_CORS_ORIGINS
    assert any("localhost" in o or "127.0.0.1" in o for o in wd.WEB_CORS_ORIGINS)
    assert wd._cors_wildcard is False

    cors = next(mw for mw in wd.app.user_middleware if mw.cls.__name__ == "CORSMiddleware")
    assert cors.kwargs["allow_origins"] == wd.WEB_CORS_ORIGINS
    # Kilitli origin listesiyle credentials açık kalabilir (geçerli kombinasyon).
    assert cors.kwargs["allow_credentials"] is True
    # Sözleşme: middleware credentials kararı wildcard bayrağının tersi olmalı.
    assert cors.kwargs["allow_credentials"] is (not wd._cors_wildcard)


# --------------------------------------------------------------- salt-okunur uçlar
def test_api_config_returns_risk_params():
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    for key in ("per_trade_risk_pct", "daily_max_loss_pct", "max_gross_risk_pct",
                "wash_sale_days", "settlement_days"):
        assert key in body


def test_api_trades_endpoint_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "LOG_DIR", tmp_path)
    r = client.get("/api/trades")
    assert r.status_code == 200
    body = r.json()
    assert "stats" in body and "trades" in body and "equity_curve" in body


def test_api_logs_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "LOG_DIR", tmp_path)   # system.log yok
    r = client.get("/api/logs")
    assert r.status_code == 200
    assert r.json() == []


# ------------------------------------------------------- WEB_TOKEN guard (Madde 44)
# Guard, routing'den önce çalışır → sahte bir /api/ yoluna POST ile yan-etkisiz test
# edilir (gerçek mutasyon uçlarına dokunmaya gerek yok).
def test_web_token_disabled_means_no_enforcement(monkeypatch):
    # Token boşsa guard hiçbir şeyi engellemez → istek routing'e düşer (404), 401 DEĞİL.
    monkeypatch.setattr(wd, "WEB_TOKEN", "")
    r = client.post("/api/__guard_probe__")
    assert r.status_code != 401


def test_web_token_blocks_unauthenticated_mutation(monkeypatch):
    monkeypatch.setattr(wd, "WEB_TOKEN", "secret123")
    # Token'sız mutasyon → 401
    assert client.post("/api/__guard_probe__").status_code == 401
    # Yanlış token → 401
    assert client.post("/api/__guard_probe__",
                       headers={"X-Auth-Token": "wrong"}).status_code == 401
    # Doğru token → guard geçer (routing'e düşer; 401 OLMAZ)
    assert client.post("/api/__guard_probe__",
                       headers={"X-Auth-Token": "secret123"}).status_code != 401


def test_web_token_does_not_block_reads(monkeypatch):
    # GET (salt-okunur) token ayarlı olsa bile her zaman serbest.
    monkeypatch.setattr(wd, "WEB_TOKEN", "secret123")
    assert client.get("/api/config").status_code == 200


def test_dashboard_injects_token_for_same_origin(monkeypatch):
    monkeypatch.setattr(wd, "WEB_TOKEN", "secret123")
    body = client.get("/").text
    assert 'window.__WEB_TOKEN__ = "secret123"' in body


# ----------------------------------------------- /api/forward-test (canlı doğrulama #3)
def _write_pred_log(path, rows):
    import csv

    import nihai.execution.prediction_tracker as pt
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pt.FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_forward_test_coverage_and_verdict(tmp_path, monkeypatch):
    import nihai.execution.prediction_tracker as pt
    log = tmp_path / "predictions_log.csv"
    # 8 çözülmüş (6 isabet → %75 → WEAK_POSITIVE) + 2 bekleyen (hit boş).
    base = {"interval": "1m", "direction": "+", "confidence": "0.6", "prob_cal": "0.6",
            "price": "100", "horizon_bars": "5", "resolved_at": "2026-06-24T10:05:00",
            "fwd_price": "101", "fwd_ret_pct": "1.0"}
    rows = []
    for i in range(8):
        r = dict(base, logged_at=f"2026-06-2{1 + i % 3}T10:00:0{i}", symbol=f"S{i}",
                 bar_ts=f"2026-06-24 10:0{i}:00", hit="1" if i < 6 else "0")
        rows.append(r)
    for i in range(2):  # bekleyen (henüz çözülmemiş)
        rows.append({**base, "logged_at": "2026-06-24T11:00:00", "symbol": "PEND",
                     "bar_ts": f"2026-06-24 11:0{i}:00", "resolved_at": "",
                     "fwd_price": "", "fwd_ret_pct": "", "hit": ""})
    _write_pred_log(log, rows)
    monkeypatch.setattr(pt, "PRED_LOG", log)

    r = client.get("/api/forward-test")          # resolve=false → ağ yok
    assert r.status_code == 200
    body = r.json()
    cov = body["coverage"]
    assert cov["total"] == 10 and cov["resolved"] == 8 and cov["pending"] == 2
    assert cov["days_span"] >= 0.0 and cov["symbols"] >= 1
    assert body["verdict"]["label"] == "WEAK_POSITIVE"   # %75 isabet
    assert body["report"]["resolved"] == 8
    assert "target_days" in body


def test_forward_test_no_data_is_honest(tmp_path, monkeypatch):
    import nihai.execution.prediction_tracker as pt
    monkeypatch.setattr(pt, "PRED_LOG", tmp_path / "yok.csv")
    body = client.get("/api/forward-test").json()
    assert body["coverage"]["tracking_file"] is False
    assert body["verdict"]["label"] == "NO_DATA"


# ------------------------------------------------ /api/var + genel-bakış VaR kartı (A3)
def test_api_var_returns_200_with_honest_contract():
    # conftest DATA_DIR'i tmp'ye yönlendirir → taze state; pozisyon yoksa ok=False + reason
    # (dürüst boş durum), pozisyon varsa historical/monte_carlo blokları. Her iki yüzde de
    # equity alanı bulunur (UI kartı % hesabını buna dayar).
    r = client.get("/api/var")
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body and "equity" in body
    if body["ok"]:
        assert "historical" in body and "monte_carlo" in body
    else:
        assert body.get("reason") or body.get("note")


def test_dashboard_html_has_var_card_and_loader():
    body = client.get("/").text
    assert 'id="var-card-val"' in body          # genel bakış "Portföy VaR" kartı
    assert 'id="var-card-cvar"' in body
    assert "loadVarCard" in body                # kart, diğerleriyle aynı yenileme desenine bağlı


# ------------------------------------------------------------------------- Tur-30
def test_html_karsilastirmadan_once_degilleme_yok():
    """`!x === false` deseni panoda BİR paneli tamamen öldürmüştü — bir daha girmesin.

    `!` `===` den önce bağlar: `!d.success === false` → `(!undefined) === false` →
    `true === false` → HER ZAMAN false. `loadEvidence` bu yüzden forward-test panelini
    (projenin #1 önceliği: verdikt/isabet/kapsama/meta-kapı) hiç doldurmuyordu; uç 200
    dönüyor, konsola hata düşmüyor, panel sessizce boş kalıyordu. Doğrusu `x !== false`.
    Tarama bütün pano JS'ini kapsar — aynı hatanın ikinci kopyası da yakalanır."""
    import re

    body = client.get("/").text
    hits = re.findall(r"![A-Za-z_$][\w$.]*\s*(?:===|!==|==|!=)\s*(?:false|true|null|undefined)",
                      body)
    assert not hits, f"karşılaştırmadan önce değilleme (parantez eksik): {hits}"


def test_analysis_provider_yedek_dali_paylasilir(monkeypatch):
    """Yedek provider BİR KEZ kurulmalı; her istekte yenisi = boş önbellek = ağdan tekrar.

    `CachingProvider` paylaşılmak üzere yazılmış (thread-safe, TTL bar 45 sn/quote 1,5 sn)
    ama `_analysis_provider` her çağrıda `get_provider()` ile yenisini kuruyordu. ÖLÇÜLDÜ:
    /api/var sıcak istek 2845 ms → 12 ms."""
    calls = []

    def _sahte_get_provider(*a, **k):
        calls.append(1)
        return object()

    monkeypatch.setattr(wd, "get_provider", _sahte_get_provider)
    monkeypatch.setattr(wd, "_FALLBACK_PROVIDER", None)

    import nihai.execution.live_predict_service as lps

    def _servis_kapali():
        raise RuntimeError("tahmin servisi kapalı")

    monkeypatch.setattr(lps, "get_service", _servis_kapali)

    first = wd._analysis_provider()
    assert wd._analysis_provider() is first, "her istekte yeni provider kuruluyor"
    assert len(calls) == 1, f"get_provider {len(calls)} kez çağrıldı"


# ----------------------------------------------------- Tur-31: resolve gaz kelebeği
def test_resolve_tarayici_yoklamasinda_her_seferinde_kosmaz(monkeypatch):
    """Pano yoklaması tam-dosya `resolve()` TETİKLEMEMELİ.

    `pt.resolve()` predictions_log'un tamamını ayrıştırır (7,8 MB / 53k satır) ve maliyeti
    çözülecek satırla değil TOPLAM satırla ölçeklenir; ÖLÇÜLDÜ 286 ms/çağrı, çözülecek
    satır 0 iken bile. İki uç bunu saniyeler arayla çağırıyordu."""
    from nihai.execution import prediction_tracker as pt

    cagrilar = []
    monkeypatch.setattr(pt, "resolve", lambda *a, **k: cagrilar.append(1) or {"resolved": 0})
    monkeypatch.setattr(wd, "_analysis_provider", lambda: object())
    monkeypatch.setattr(pt, "_file_sig", lambda: (111, 222))     # dosya DEĞİŞMİYOR
    monkeypatch.setattr(wd, "_resolve_cache", None)

    ilk = wd._resolve_throttled()
    for _ in range(9):
        wd._resolve_throttled()
    assert len(cagrilar) == 1, f"TTL içinde {len(cagrilar)} kez resolve koştu"
    assert ilk == {"resolved": 0}


def test_resolve_dosya_degisince_TTL_beklemeden_kosar(monkeypatch):
    """Yeni tahmin yazıldıysa (imza değişti) gaz kelebeği devreye GİRMEMELİ —
    yoksa pano bayat kalırdı."""
    from nihai.execution import prediction_tracker as pt

    cagrilar = []
    monkeypatch.setattr(pt, "resolve", lambda *a, **k: cagrilar.append(1) or {"resolved": 0})
    monkeypatch.setattr(wd, "_analysis_provider", lambda: object())
    monkeypatch.setattr(wd, "_resolve_cache", None)

    imza = [(1, 1)]
    monkeypatch.setattr(pt, "_file_sig", lambda: imza[0])
    wd._resolve_throttled()
    imza[0] = (2, 2)                       # başka bir süreç/record yazdı
    wd._resolve_throttled()
    assert len(cagrilar) == 2, "dosya değiştiği hâlde önbellekten dönüldü"


def test_port_yoklamasi_onbellekli_ama_TTL_sonrasi_tazelenir(monkeypatch):
    """Kapalı port zaman aşımının tamamını yer; arka arkaya yoklamalar soket açmamalı."""
    import socket

    import nihai.web.api_extra as ax

    acilan = []

    def _sahte_connect(adres, timeout=None):
        acilan.append(adres)
        raise OSError("baglanti reddedildi")

    monkeypatch.setattr(socket, "create_connection", _sahte_connect)
    monkeypatch.setattr(ax, "_PORT_CACHE", {})
    c = TestClient(wd.app)
    c.get("/api/system/health")
    ilk_sayi = len(acilan)
    assert ilk_sayi >= 1
    c.get("/api/system/health")
    assert len(acilan) == ilk_sayi, "TTL içinde port yeniden yoklandı"

    monkeypatch.setattr(ax, "_PORT_TTL_SEC", 0.0)      # TTL doldu
    c.get("/api/system/health")
    assert len(acilan) > ilk_sayi, "TTL dolduğu hâlde önbellek tazelenmedi"
