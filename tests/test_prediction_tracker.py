"""test_prediction_tracker.py — #3 kapalı-döngü doğrulama (record/resolve/report)."""
from __future__ import annotations

import pandas as pd
import pytest

import nihai.execution.prediction_tracker as pt
from nihai import atomic_io  # Tur-12: _atomic_replace buraya taşındı (os/time patch hedefi)


def test_atomic_replace_gecici_kilit_yeniden_dener(tmp_path, monkeypatch):
    """Tur-11 olayı: 2026-07-11 OneDrive senkron kilidi os.replace'i PermissionError'la
    düşürüp TÜM harness'i çökertti. _atomic_replace geçici OSError'da yeniden denemeli,
    kalıcı hatada ise (tüm denemeler tükenince) istisnayı YÜKSELTMELİ (sessiz yutma yok)."""
    src = tmp_path / "a.tmp"
    dst = tmp_path / "a.csv"
    src.write_text("x", encoding="utf-8")
    calls = {"n": 0}
    real_replace = atomic_io.os.replace

    def flaky_replace(s, d):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("gecici kilit (OneDrive benzeri)")
        return real_replace(s, d)

    monkeypatch.setattr(atomic_io.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_io.time, "sleep", lambda *_a, **_k: None)  # testte bekleme yok
    pt._atomic_replace(src, dst, retries=6)
    assert calls["n"] == 3
    assert dst.read_text(encoding="utf-8") == "x"


def test_atomic_replace_kalici_hata_yukselir(tmp_path, monkeypatch):
    src = tmp_path / "b.tmp"
    dst = tmp_path / "b.csv"
    src.write_text("x", encoding="utf-8")

    def always_fail(s, d):
        raise PermissionError("kalıcı kilit")

    monkeypatch.setattr(atomic_io.os, "replace", always_fail)
    monkeypatch.setattr(atomic_io.time, "sleep", lambda *_a, **_k: None)
    with pytest.raises(PermissionError):
        pt._atomic_replace(src, dst, retries=3)


@pytest.fixture()
def tmp_log(tmp_path, monkeypatch):
    """PRED_LOG'u geçici dosyaya yönlendir (gerçek data/ dokunulmaz)."""
    monkeypatch.setattr(pt, "PRED_LOG", tmp_path / "predictions_log.csv")
    monkeypatch.setattr(pt, "FEATS_LOG", tmp_path / "live_features_log.csv")
    return pt.PRED_LOG


def _df(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-06-19 10:00:00", periods=len(prices), freq="h")
    return pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices}, index=idx)


class _FakeProvider:
    def __init__(self, df):
        self._df = df

    def get_bars(self, symbol, interval, lookback_bars=400):
        return self._df


def _row(symbol, bar_ts, direction="+", price=100.0, action="WAIT"):
    return {"symbol": symbol, "action": action, "direction": direction,
            "bar_ts": bar_ts, "confidence": 0.6, "prob_cal": 0.52, "price": price}


def test_record_dedup(tmp_log):
    df = _df([100, 101, 102, 103, 104, 105])
    bts = str(df.index[0])
    rows = [_row("AAPL", bts), _row("MSFT", bts, direction="-")]
    assert pt.record(rows, "1h", horizon=3) == 2
    # Aynı bar tekrar → 0 (bar başına tek kayıt).
    assert pt.record(rows, "1h", horizon=3) == 0
    # Yeni bar → kaydedilir.
    assert pt.record([_row("AAPL", str(df.index[1]))], "1h", horizon=3) == 1


def test_record_dedup_kanonik_ts_format_karisik(tmp_log):
    """Denetim Tur-11: dedup artık _canon_ts kullanır — aynı bar farklı gösterimle
    (naive-UTC vs tz-aware-eşdeğer) gelirse (örn. sağlayıcı failover hub→yfinance) İKİ
    satır DEĞİL, TEK satır yazılmalı (isabet/AUC istatistiklerini şişirmesin)."""
    df = _df([100, 101, 102])
    naive = df.index[0]                              # ör. 2026-06-19 10:00:00
    aware_equiv = naive.tz_localize("UTC")            # AYNI an, tz-aware gösterim
    assert pt.record([_row("AAPL", str(naive))], "1h", horizon=1) == 1
    # Aynı bar, FARKLI string biçimi → dedup canonik eşitlikle yakalamalı.
    assert pt.record([_row("AAPL", str(aware_equiv))], "1h", horizon=1) == 0
    with pt.PRED_LOG.open(encoding="utf-8", newline="") as f:
        import csv as _csv
        rows = list(_csv.DictReader(f))
    assert len(rows) == 1


def test_nondirectional_skipped(tmp_log):
    rows = [{"symbol": "X", "action": "NODATA", "direction": "•", "bar_ts": "t", "price": 0}]
    assert pt.record(rows, "1h") == 0


def test_resolve_and_hit(tmp_log):
    # Fiyat 100 → +3 yön bar sonra 103: LONG tahmini isabet (hit=1).
    df = _df([100, 101, 102, 103, 104, 105])
    bts = str(df.index[0])
    pt.record([_row("AAPL", bts, direction="+", price=100.0)], "1h", horizon=3)
    res = pt.resolve(provider=_FakeProvider(df))
    assert res["resolved"] == 1
    rep = pt.report()
    assert rep["resolved"] == 1
    assert rep["hit_rate"] == 1.0          # 103 > 100, yön + → isabet
    assert rep["avg_signed_ret_pct"] == pytest.approx(3.0, abs=1e-6)


def test_resolve_miss_short(tmp_log):
    # Fiyat yükseldi ama tahmin SHORT → isabetsiz (hit=0), yön-ayarlı getiri negatif.
    df = _df([100, 101, 102, 103])
    bts = str(df.index[0])
    pt.record([_row("Z", bts, direction="-", price=100.0)], "1h", horizon=2)
    pt.resolve(provider=_FakeProvider(df))
    rep = pt.report()
    assert rep["hit_rate"] == 0.0
    assert rep["avg_signed_ret_pct"] < 0


def test_resolve_pending_when_horizon_not_reached(tmp_log):
    # Yalnız 2 bar var ama horizon 5 → henüz çözülemez (pending).
    df = _df([100, 101])
    bts = str(df.index[0])
    pt.record([_row("AAPL", bts, price=100.0)], "1h", horizon=5)
    res = pt.resolve(provider=_FakeProvider(df))
    assert res["resolved"] == 0
    assert res["pending"] == 1


# ---------------------------------------------------------------------------
# Meta-kapı canlı kanıt döngüsü: action/pwin_side kolonları + kapı simülasyonu
# ---------------------------------------------------------------------------
def test_record_writes_action_and_pwin_side(tmp_log):
    """Kapı-engelli aday (action=WAIT + pwin + pwin_side) şemaya yazılır ve geri okunur."""
    import csv
    df = _df([100, 101, 102, 103])
    bts = str(df.index[0])
    r = _row("AAPL", bts, action="WAIT")
    r["pwin"] = 0.31
    r["pwin_side"] = "LONG"
    assert pt.record([r], "1h", horizon=2) == 1
    with pt.PRED_LOG.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["action"] == "WAIT"
    assert rows[0]["pwin"] == "0.31"
    assert rows[0]["pwin_side"] == "LONG"


def test_schema_upgrade_from_old_header(tmp_log):
    """Eski (action/pwin_side'sız) dosyaya yeni kayıt → başlık atomik yükseltilir,
    eski satırlar korunur (yeni kolonlar boş), kayma olmaz."""
    import csv
    old_fields = ["logged_at", "symbol", "interval", "bar_ts", "direction", "confidence",
                  "prob_cal", "pwin", "price", "horizon_bars",
                  "resolved_at", "fwd_price", "hit", "fwd_ret_pct"]
    with pt.PRED_LOG.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=old_fields)
        w.writeheader()
        w.writerow({"logged_at": "t0", "symbol": "OLD", "interval": "1h", "bar_ts": "b0",
                    "direction": "+", "confidence": "0.5", "prob_cal": "", "pwin": "",
                    "price": "10", "horizon_bars": "5",
                    "resolved_at": "", "fwd_price": "", "hit": "", "fwd_ret_pct": ""})
    df = _df([100, 101, 102])
    assert pt.record([_row("NEW", str(df.index[0]))], "1h", horizon=2) == 1
    with pt.PRED_LOG.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == pt.FIELDS
        rows = list(reader)
    old = next(r for r in rows if r["symbol"] == "OLD")
    assert old["action"] == "" and old["pwin_side"] == ""   # eski satır kaymadan taşındı
    assert old["price"] == "10"


def test_side_return_sign_recovery():
    """fwd_ret_pct yön-işaretlidir; _side_return ham getiriyi çözüp pwin tarafına işaretler."""
    # Fiyat +2% hareket etti; direction '-' kaydettiği için fwd_ret_pct=-2.0 yazılmıştı.
    row = {"direction": "-", "fwd_ret_pct": "-2.0", "pwin_side": "LONG"}
    assert pt._side_return(row) == pytest.approx(2.0)       # LONG taraf +2% kazanır
    row["pwin_side"] = "SHORT"
    assert pt._side_return(row) == pytest.approx(-2.0)      # SHORT taraf -2% kaybeder
    assert pt._side_return({"direction": "+", "fwd_ret_pct": "1.0"}) is None  # eski şema → hariç


def test_gate_sim_separation_and_labels():
    """Yüksek-pwin adaylar kazanıp düşük-pwin kaybederse GATE_HELPS; ters ise GATE_HURTS."""
    def mk(pwin, ret, side="LONG", direction="+"):
        fwd = ret if direction == "+" else -ret
        return {"pwin": str(pwin), "pwin_side": side, "direction": direction,
                "fwd_ret_pct": str(fwd)}
    good = [mk(0.60, +1.0) for _ in range(5)]
    bad = [mk(0.30, -1.0) for _ in range(5)]
    gs = pt.gate_sim(good + bad, threshold=0.45, min_side=3)
    assert gs["label"] == "GATE_HELPS"
    assert gs["kept"]["n"] == 5 and gs["cut"]["n"] == 5
    assert gs["kept"]["avg_ret_pct"] > gs["base"]["avg_ret_pct"]
    # Ters dünya: model yüksek dediklerinde kaybediyor → kapı zarar veriyor.
    inv = pt.gate_sim([mk(0.60, -1.0) for _ in range(5)] + [mk(0.30, +1.0) for _ in range(5)],
                      threshold=0.45, min_side=3)
    assert inv["label"] == "GATE_HURTS"


def test_gate_sim_threshold_sweep_sample_gated():
    """sweep: her eşik kendi örneklem kapısından geçer — iki taraf >= min_side ise
    uplift_pp yazılır, geçemeyen eşikte None (hüküm yok, kenar uydurulmaz)."""
    def mk(pwin, ret):
        return {"pwin": str(pwin), "pwin_side": "LONG", "direction": "+",
                "fwd_ret_pct": str(ret)}
    rows = ([mk(0.60, +1.0) for _ in range(5)] + [mk(0.42, +0.2) for _ in range(5)]
            + [mk(0.30, -1.0) for _ in range(5)])
    gs = pt.gate_sim(rows, threshold=0.45, min_side=3)
    sweep = {s["thr"]: s for s in gs["sweep"]}
    assert set(sweep) == set(pt.SWEEP_THRESHOLDS)
    # thr=0.35: cut=5 (0.30'lar), kept=10 → iki taraf >=3 → uplift yazılır.
    assert sweep[0.35]["uplift_pp"] is not None
    assert sweep[0.35]["kept"]["n"] == 10 and sweep[0.35]["cut"]["n"] == 5
    # thr=0.60: kept yalnız 5 ama cut 10 → yine >=3; thr'lerin tutarlılığı:
    assert sweep[0.55]["kept"]["n"] == 5 and sweep[0.55]["cut"]["n"] == 10
    # Yüksek pwin gerçekten iyi → 0.55 eşiğinin uplifti pozitif olmalı.
    assert sweep[0.55]["uplift_pp"] > 0
    # min_side'ı yükselt: hiçbir eşik iki tarafı dolduramaz → hepsi None.
    gs2 = pt.gate_sim(rows, threshold=0.45, min_side=11)
    assert all(s["uplift_pp"] is None for s in gs2["sweep"])


def test_report_pwin_calibration_buckets(tmp_log):
    """report() pwin kovası → gerçek taraf-kazanma eğrisi üretir (canlı model
    kalibrasyonu; 'buckets' yön-güvenidir, bu META modelin ayrı eğrisidir)."""
    df = _df([100, 101, 102, 103, 104, 105])
    r1 = _row("AAPL", str(df.index[0]))
    r1["pwin"], r1["pwin_side"] = 0.62, "LONG"
    r2 = _row("MSFT", str(df.index[1]))
    r2["pwin"], r2["pwin_side"] = 0.31, "SHORT"
    pt.record([r1, r2], "1h", horizon=2)
    pt.resolve(provider=_FakeProvider(df))
    rep = pt.report()
    pc = rep["pwin_calibration"]
    assert "0.6-0.7" in pc and pc["0.6-0.7"]["n"] == 1
    assert pc["0.6-0.7"]["win_rate"] == 1.0    # fiyat yükseldi, LONG taraf kazandı
    assert "0.3-0.4" in pc and pc["0.3-0.4"]["win_rate"] == 0.0  # SHORT taraf kaybetti


def test_gate_sim_insufficient_is_honest():
    """Örneklem kapısının altında kapı hakkında hüküm verilmez (INSUFFICIENT)."""
    rows = [{"pwin": "0.6", "pwin_side": "LONG", "direction": "+", "fwd_ret_pct": "1.0"}]
    gs = pt.gate_sim(rows, threshold=0.45, min_side=30)
    assert gs["label"] == "INSUFFICIENT"
    assert gs["n_candidates"] == 1
    assert "uplift_pp" not in gs


def test_report_includes_meta_gate_section(tmp_log):
    """report() 'meta_gate' anahtarını taşır (web /api/forward-test otomatik alır)."""
    df = _df([100, 101, 102, 103, 104, 105])
    r = _row("AAPL", str(df.index[0]))
    r["pwin"] = 0.6
    r["pwin_side"] = "LONG"
    pt.record([r], "1h", horizon=3)
    pt.resolve(provider=_FakeProvider(df))
    rep = pt.report()
    assert "meta_gate" in rep
    assert rep["meta_gate"]["n_candidates"] == 1
    assert rep["meta_gate"]["label"] == "INSUFFICIENT"   # tek aday < min_side → dürüst


def test_signal_pwin_helper_covers_gated_candidates():
    """_signal_pwin: aktif sinyal → metadata/gate; kapı-engelli WAIT → metadata+taraf;
    sıradan WAIT → (None, None)."""
    from nihai.execution.live_predict_service import _signal_pwin
    from nihai.signals.strategy import Signal, SignalAction

    class _GateBoom:
        def probability(self, df, sig):   # metadata varken ÇAĞRILMAMALI
            raise AssertionError("gate.probability çağrılmamalıydı")

    class _Gate04:
        def probability(self, df, sig):
            return 0.4

    s_long = Signal(SignalAction.LONG, "X", 100, 99, 102, metadata={"meta_p_win": 0.61})
    assert _signal_pwin(s_long, _GateBoom(), None) == (0.61, "LONG")

    s_long2 = Signal(SignalAction.SHORT, "X", 100, 101, 98)   # metadata yok → gate'e sor
    assert _signal_pwin(s_long2, _Gate04(), None) == (0.4, "SHORT")

    s_gated = Signal(SignalAction.WAIT, "X", 100, 99, 102,
                     metadata={"meta_p_win": 0.31, "meta_gated": "LONG"})
    assert _signal_pwin(s_gated, _GateBoom(), None) == (0.31, "LONG")

    s_wait = Signal(SignalAction.WAIT, "X", 100, 0, 0)
    assert _signal_pwin(s_wait, _GateBoom(), None) == (None, None)


# ---------------------------------------------------------------- canli ozellik logu

def _feats(**over):
    """FEATURE_NAMES uzunlugunda ornek ozellik sozlugu."""
    from nihai.signals.meta_features import FEATURE_NAMES
    d = {n: 0.5 for n in FEATURE_NAMES}
    d.update(over)
    return d


def test_record_writes_features_log(tmp_log):
    """'_feats' tasiyan yeni satir → FEATS_LOG'a FEATURE_NAMES kolonlariyla yazilir;
    dedup predictions_log'dan miras (ayni bar ikinci kez → ozellik de yazilmaz)."""
    import csv

    from nihai.signals.meta_features import FEATURE_NAMES
    df = _df([100, 101, 102])
    bts = str(df.index[0])
    r = _row("AAPL", bts, action="LONG")
    r["pwin"], r["pwin_side"], r["_feats"] = 0.61, "LONG", _feats(adx14=33.0)
    assert pt.record([r], "1h", horizon=3) == 1
    assert pt.FEATS_LOG.exists()
    with pt.FEATS_LOG.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL" and rows[0]["bar_ts"] == bts
    assert float(rows[0]["adx14"]) == 33.0
    assert all(n in rows[0] for n in FEATURE_NAMES)
    # Ayni bar tekrar → ne tahmin ne ozellik satiri eklenir.
    assert pt.record([r], "1h", horizon=3) == 0
    with pt.FEATS_LOG.open(encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == 1


def test_record_without_feats_skips_features_log(tmp_log):
    """_feats'siz satir (track kapali / aday degil) → FEATS_LOG olusmasin."""
    df = _df([100, 101])
    assert pt.record([_row("MSFT", str(df.index[0]))], "1h") == 1
    assert not pt.FEATS_LOG.exists()


def test_features_log_header_upgrade(tmp_log):
    """Eski (eksik-kolon) baslikli FEATS_LOG → yeni yazimda atomik yukseltilir;
    eski satirlarin eksik kolonlari bos kalir, veri kaybolmaz."""
    import csv
    pt.FEATS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with pt.FEATS_LOG.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["logged_at", "symbol", "interval", "bar_ts", "adx14"])
        w.writeheader()
        w.writerow({"logged_at": "t0", "symbol": "OLD", "interval": "1h",
                    "bar_ts": "b0", "adx14": "11.0"})
    df = _df([100, 101])
    r = _row("NEW", str(df.index[0]), action="LONG")
    r["pwin"], r["pwin_side"], r["_feats"] = 0.55, "LONG", _feats()
    assert pt.record([r], "1h") == 1
    with pt.FEATS_LOG.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == pt.FEAT_FIELDS
        rows = list(reader)
    assert [r0["symbol"] for r0 in rows] == ["OLD", "NEW"]
    assert rows[0]["adx14"] == "11.0" and rows[0]["rsi14"] == ""


# --------------------------------------------------------------------------- D4b (Tur-28)
# summary(): report+coverage+verdict TEK okumadan, PRED_LOG imzasıyla önbellekli.
# Ölçüm (gerçek 6,7 MB / 53k satır log): tekrar istek 385 ms → 0,16 ms.
def _summary_cache_sifirla():
    pt._SUMMARY = None


def test_summary_manuel_bilesimle_birebir_ayni(tmp_log):
    """Önbellek DAVRANIŞI değiştirmez: summary() eski üç-çağrı bileşimiyle aynı sonucu verir."""
    _summary_cache_sifirla()
    df = _df([100, 101, 102, 103])
    r = _row("AAA", str(df.index[0]), action="LONG")
    assert pt.record([r], "1h") == 1
    pt.resolve(provider=_FakeProvider(df))

    rows = pt._read_log_rows()
    beklenen = {"report": pt.report(rows), "coverage": pt.coverage(rows),
                "verdict": pt.verdict(pt.report(rows))}
    _summary_cache_sifirla()
    assert pt.summary() == beklenen


def test_summary_dosya_degismediyse_yeniden_ayristirmaz(tmp_log, monkeypatch):
    """İkinci çağrı diski AYRIŞTIRMAZ: _read_log_rows çağrılırsa test patlar."""
    _summary_cache_sifirla()
    r = _row("AAA", "2026-06-19 10:00:00", action="LONG")
    pt.record([r], "1h")
    ilk = pt.summary()

    def _yasak():   # pragma: no cover - çağrılırsa zaten AssertionError
        raise AssertionError("dosya değişmediği hâlde yeniden okundu")

    monkeypatch.setattr(pt, "_read_log_rows", _yasak)
    assert pt.summary() == ilk


def test_summary_dosya_degisince_kendiliginden_tazelenir(tmp_log):
    """Yeni kayıt sonrası özet BAYAT kalmamalı (imza = mtime_ns + size)."""
    _summary_cache_sifirla()
    pt.record([_row("AAA", "2026-06-19 10:00:00", action="LONG")], "1h")
    once = pt.summary()["coverage"]["total"]
    pt.record([_row("BBB", "2026-06-19 11:00:00", action="SHORT")], "1h")
    sonra = pt.summary()["coverage"]["total"]
    assert sonra == once + 1, "önbellek bayat özet döndürdü"
    _summary_cache_sifirla()


def test_summary_log_yokken_cokmez(tmp_log):
    _summary_cache_sifirla()
    s = pt.summary()
    assert s["coverage"]["total"] == 0 and s["verdict"]["label"] == "NO_DATA"
    _summary_cache_sifirla()
