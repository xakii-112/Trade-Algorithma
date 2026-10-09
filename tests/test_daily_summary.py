"""daily_summary testleri — saf build_summary sözleşmesi + collect IO + CLI smoke.

Ağ/LLM çağrısı YOK: servis sağlığı testte yerel dinleyici soketle, collect CLI smoke'ta
monkeypatch'le sürülür. Dosya yolları modül-attr monkeypatch (prediction_tracker deseni)."""
from __future__ import annotations

import socket
from datetime import date, datetime

import nihai.notifier.daily_summary as ds

_LAST = {"run_at": "2026-07-10T21:23:23", "resolved": "20170", "days_span": "16.2",
         "symbols": "36", "hit_rate": "0.4746", "auc_prob_cal": "0.4874",
         "auc_pwin": "0.4818", "brier_pwin": "0.2664", "ece_pwin": "0.1",
         "avg_signed_ret_pct": "-0.0232", "verdict": "NO_EDGE"}
_PREV = {**_LAST, "run_at": "2026-07-10T06:00:55", "resolved": "20010"}


def _data(**over):
    d = {"today": "2026-07-11", "history_last": dict(_LAST), "history_prev": dict(_PREV),
         "gate_sim": {"label": "GATE_HURTS", "uplift_pp": -0.021},
         "coverage": None, "services": {"hub": True, "kayitci": False},
         "live_features_today": 42}
    d.update(over)
    return d


# --------------------------------------------------------------- build_summary (saf)
def test_build_summary_quotes_verdict_delta_and_services():
    text = ds.build_summary(_data())
    assert '"NO_EDGE"' in text                      # verdikt AYNEN alıntılanır
    assert "(+160)" in text                         # çözülen delta: 20170-20010
    assert "hub AÇIK" in text and "kayıtçı KAPALI" in text
    assert '"GATE_HURTS"' in text                   # gate_sim verdikti aynen
    assert "+42 satır" in text                      # bugünkü canlı özellik satırı
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert 5 <= len(lines) <= 8                     # sözleşme: 5-8 satır
    # kenar İDDİASI yok (dürüstlük sözleşmesi)
    assert "kenar bulundu" not in text.lower()
    assert "kenar var" not in text.lower()


def test_build_summary_graceful_when_all_missing():
    text = ds.build_summary({"today": "2026-07-11", "history_last": None, "history_prev": None,
                             "gate_sim": None, "coverage": None,
                             "services": {"hub": False, "kayitci": False},
                             "live_features_today": None})
    assert "veri yok" in text                       # eksikler dürüstçe söylenir
    assert "hub KAPALI" in text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert 5 <= len(lines) <= 8


def test_build_summary_no_prev_row_no_delta():
    text = ds.build_summary(_data(history_prev=None))
    assert "çözülen 20170" in text and "(+" not in text   # önceki satır yoksa delta yok


# ----------------------------------------------------------------------- collect (IO)
def test_collect_reads_history_ports_and_features(tmp_path, monkeypatch):
    hist = tmp_path / "forward_eval" / "history.csv"
    hist.parent.mkdir(parents=True)
    cols = list(_LAST)
    rows = [",".join(_PREV[c] for c in cols), ",".join(_LAST[c] for c in cols)]
    hist.write_text(",".join(cols) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(ds, "HISTORY_CSV", hist)
    monkeypatch.setattr(ds, "GATE_SIM_JSON", tmp_path / "gate_sim.json")     # yok → None
    monkeypatch.setattr(ds, "COVERAGE_JSON", tmp_path / "coverage.json")     # yok → None

    feats = tmp_path / "live_features_log.csv"
    simdi = datetime.now().isoformat(timespec="seconds")   # Tur-106 (BULGU-106-36): sayım son seans açılışından beri
    feats.write_text("logged_at,symbol\n"
                     f"{simdi},AAPL\n"
                     f"{simdi},MSFT\n"
                     "2020-01-01T09:31:00,OLD\n", encoding="utf-8")
    monkeypatch.setattr(ds, "FEATS_LOG", feats)

    # Servis sağlığı: gerçek dinleyici soket (AÇIK) + kapalı port (KAPALI). Ağ dışarı çıkmaz.
    srv = socket.socket()
    try:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        open_port = srv.getsockname()[1]
        closed = socket.socket()
        closed.bind(("127.0.0.1", 0))
        closed_port = closed.getsockname()[1]
        closed.close()                                # bind'lenip kapatıldı → reddedilir
        data = ds.collect(ports={"hub": open_port, "kayitci": closed_port})
    finally:
        srv.close()

    assert data["services"] == {"hub": True, "kayitci": False}
    assert data["history_last"]["verdict"] == "NO_EDGE"
    assert data["history_prev"]["resolved"] == "20010"
    assert data["gate_sim"] is None and data["coverage"] is None
    assert data["live_features_today"] == 2           # yalnız son seans açılışından sonrakiler
    assert len(data["live_features_day"]) == 10       # sayımın ait olduğu seans günü taşınır


def test_collect_graceful_when_files_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "HISTORY_CSV", tmp_path / "yok.csv")
    monkeypatch.setattr(ds, "GATE_SIM_JSON", tmp_path / "yok.json")
    monkeypatch.setattr(ds, "COVERAGE_JSON", tmp_path / "yok2.json")
    monkeypatch.setattr(ds, "FEATS_LOG", tmp_path / "yok3.csv")
    data = ds.collect(ports={})
    assert data["history_last"] is None and data["history_prev"] is None
    assert data["live_features_today"] is None
    assert ds.build_summary(data)                     # uçtan uca çökmez


# ------------------------------------------------------------------------- CLI smoke
def _fake_collect(**_kw):
    return _data(today=datetime.now().strftime("%Y-%m-%d"))


def test_main_writes_file_and_prints(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ds, "SUMMARY_TXT", tmp_path / "daily_summary_last.txt")
    monkeypatch.setattr(ds, "collect", _fake_collect)
    rc = ds.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NO_EDGE" in out
    saved = (tmp_path / "daily_summary_last.txt").read_text(encoding="utf-8")
    assert "NO_EDGE" in saved


def test_main_send_without_token_notes_gracefully(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ds, "SUMMARY_TXT", tmp_path / "s.txt")
    monkeypatch.setattr(ds, "collect", _fake_collect)
    import nihai.notifier.telegram_bot as tb
    monkeypatch.setattr(tb, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(tb, "TELEGRAM_CHAT_ID", "")
    rc = ds.main(["--send"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "telegram yapılandırılmamış" in out         # çökme yok, dürüst not


def _llm_ortami(tmp_path, monkeypatch):
    """--llm testleri için ortak kurulum: dosya yolu + sahte collect + SIFIR sağlayıcı anahtarı.

    BULGU-19 (2026-07-31): eskiden yalnız ANTHROPIC_API_KEY siliniyordu; router zinciri
    GEMINI'ye düşüp .env'deki gerçek anahtarla CANLI istek atıyordu (docstring "ağ yok"
    derken). Anahtarların HEPSİ silinir → `_provider_chain()` boş → sağlayıcı denenmez."""
    monkeypatch.setattr(ds, "SUMMARY_TXT", tmp_path / "s.txt")
    monkeypatch.setattr(ds, "collect", _fake_collect)
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_main_llm_falls_back_deterministic(tmp_path, monkeypatch, capsys):
    # --llm: hiçbir sağlayıcı anahtarı yok → _llm_polish deterministik metni AYNEN döndürür.
    _llm_ortami(tmp_path, monkeypatch)
    monkeypatch.setenv("LLM_EVENT_PROVIDER", "anthropic")
    rc = ds.main(["--llm"])
    out = capsys.readouterr().out
    assert rc == 0 and "NO_EDGE" in out


def test_main_llm_provider_hatasi_deterministige_duser(tmp_path, monkeypatch, capsys):
    """Sağlayıcı İSTİSNA fırlatırsa özet düşmez (router.complete sözleşmesi)."""
    _llm_ortami(tmp_path, monkeypatch)
    import nihai.ai.router as router

    def _patlar(task, prompt, **kw):
        raise RuntimeError("sağlayıcı çöktü")

    monkeypatch.setattr(router, "complete", _patlar)
    rc = ds.main(["--llm"])
    out = capsys.readouterr().out
    assert rc == 0 and "NO_EDGE" in out


def test_main_llm_sayilari_bozan_cikti_atilir(tmp_path, monkeypatch, capsys):
    """DÜRÜSTLÜK GUARDI: LLM verdikti/sayıyı korumazsa çıktısı ATILIR (uydurma koruması)."""
    _llm_ortami(tmp_path, monkeypatch)
    import nihai.ai.router as router
    monkeypatch.setattr(router, "complete",
                        lambda task, prompt, **kw: "Sistem bugün harika, kenar bulundu!")
    rc = ds.main(["--llm"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NO_EDGE" in out                      # deterministik metne dönüldü
    assert "kenar bulundu" not in out            # LLM uydurması basılmadı


def test_main_llm_sayilari_koruyan_cikti_kullanilir(tmp_path, monkeypatch, capsys):
    """Guard'ın KABUL dalı: verdikt/sayı/kapanış korunmuşsa LLM metni kullanılır."""
    _llm_ortami(tmp_path, monkeypatch)
    import nihai.ai.router as router
    # prompt = başlık + "\n\n" + deterministik metin → metni aynen koruyup başlık ekle
    monkeypatch.setattr(router, "complete",
                        lambda task, prompt, **kw: "AKICI-BASLIK\n" + prompt.split("\n\n", 1)[1])
    rc = ds.main(["--llm"])
    out = capsys.readouterr().out
    assert rc == 0 and "AKICI-BASLIK" in out and "NO_EDGE" in out
