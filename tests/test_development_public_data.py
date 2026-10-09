"""FDA/Wikimedia: günlük nedensellik, eksik ölçüm, atomik önbellek ve pano tüketimi."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nihai.data import fda_calendar as fda
from nihai.data import macro_regime as mr
from nihai.data import wiki_trends as wiki
from nihai.web import api_lab as lab


def fda_payload():
    def sub(day, status="AP", number="001"):
        return dict(submission_status_date=day, submission_status=status,
                    submission_type="ORIG", submission_number=number)
    return {"meta": {"results": {"total": 2}}, "results": [dict(application_number="BLA123456",
        sponsor_name="TEST BIO", products=[dict(brand_name="Test Product")], submissions=[
            sub("20250303"), sub("20250303"), sub("20250304", "TA"), sub("20270101"),
            sub("invalid"), {"submission_status": "AP"}])]}


def wiki_items():
    days = pd.date_range("2025-01-01", periods=40)
    return [dict(timestamp=day.strftime("%Y%m%d00"), views=i + 1) for i, day in enumerate(days)]


def test_fda_submissions_are_filtered_individually_and_no_pdufa_inferred():
    payload = fda_payload()
    events = fda.approval_events(payload, "2025-01-01", "2025-12-31")
    assert len(events) == 1 and events[0]["date"] == "2025-03-03"
    assert events[0]["submission_type"] == "ORIG" and events[0]["status"] == "AP"
    payload["results"][0]["products"] = None
    assert fda.approval_events(payload, "2025-01-01", "2025-12-31")[0]["products"] == []
    for bad in (None, {}, {"results": None}):
        assert fda.approval_events(bad, "2025-01-01", "2025-12-31") is None
    assert fda.approval_events({"results": []}, "2025-01-01", "2025-12-31") == []


def test_attention_has_prior_only_baseline_and_preserves_missing_days():
    items = wiki_items()
    data = wiki.attention_series(items)
    history = np.arange(1, 29)
    row = data.iloc[28]
    assert row.baseline_mean == pytest.approx(history.mean())
    assert row.zscore == pytest.approx((29 - history.mean()) / history.std(ddof=1))
    assert data.iloc[:28].ratio.isna().all()
    changed = [dict(item) for item in items]
    changed[-1]["views"] = 100000
    pd.testing.assert_frame_equal(wiki.attention_series(changed).iloc[:-1], data.iloc[:-1])
    gap = wiki.attention_series([item for i, item in enumerate(items) if i != 15])
    assert np.isnan(gap.iloc[15].views) and np.isnan(gap.iloc[28].ratio)
    flat = wiki.attention_series([{**item, "views": 100} for item in items])
    assert flat.iloc[-1].ratio == 1 and np.isnan(flat.iloc[-1].zscore)
    for bad in ([], None, [None], [dict(timestamp="2025010100", views=np.nan)],
                [dict(timestamp="2025010100", views=np.inf)], [dict(timestamp="bad", views=1)]):
        assert wiki.attention_series(bad) is None


@pytest.mark.parametrize("module,kwargs,payload", [
    (fda, dict(sponsor="TEST BIO", since="2025-01-01", until="2025-12-31"), fda_payload()),
    (wiki, dict(article="Apple Inc.", since="2025-01-01", until="2025-02-09"), {"items": wiki_items()}),
])
def test_cache_only_force_never_fetches_and_refresh_failure_preserves_cache(module, kwargs, payload, monkeypatch, tmp_path):
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path / module.__name__.split(".")[-1])
    monkeypatch.setattr(module, "contact_ua", lambda _: "Test/1.0 (eposta@ornek.com)")
    calls = []
    def get(url, **options):
        calls.append((url, options))
        return SimpleNamespace(status_code=200, json=lambda: payload, raise_for_status=lambda: None)
    monkeypatch.setattr(module, "shared_client", lambda: SimpleNamespace(get=get))
    reader = fda.calendar if module is fda else wiki.report
    assert reader(**kwargs, fetch=False, force=True) is None
    assert not calls and not module.CACHE_DIR.exists()
    first = reader(**kwargs, fetch=True)
    assert first["cache_state"] == "downloaded"
    if module is fda:
        assert first["upcoming_pdufa"] is None and first["truncated"]
        assert 'sponsor_name:"TEST BIO"' in calls[0][1]["params"]["search"]
    else:
        assert "/all-access/user/Apple_Inc./daily/" in calls[0][0]
        assert first["rows"][0]["zscore"] is None
        assert first["rows"][-1]["date"] == "2025-02-09"
    saved = next(module.CACHE_DIR.glob("*.json")).read_bytes()
    assert reader(**kwargs, fetch=False, force=True)["cache_state"] == "fresh"
    assert len(calls) == 1
    def fail(*args, **options):
        raise TimeoutError("offline")
    monkeypatch.setattr(module, "shared_client", lambda: SimpleNamespace(get=fail))
    stale = reader(**kwargs, fetch=True, force=True)
    assert stale["cache_state"] == "stale" and stale["refresh_failed"]
    assert next(module.CACHE_DIR.glob("*.json")).read_bytes() == saved


def test_public_data_inputs_and_lab_runner(monkeypatch, tmp_path):
    assert fda.calendar('x" OR *', fetch=False) is None
    assert wiki.report("Title", project="evil.example.org", fetch=False) is None
    assert wiki.report("Title", since="2010-01-01", fetch=False) is None
    assert wiki.report("Title", since="2025-03-01", until="2025-01-01", fetch=False) is None
    for tool, reader_name, module, key, value in (
        ("fda_calendar", "calendar", fda, "sponsor", "TEST BIO"),
        ("wiki_trends", "report", wiki, "article", "Apple Inc."),
    ):
        calls = []
        def collect(_tool=tool, _calls=calls, **kwargs):
            _calls.append(kwargs)
            return {"ok": True, "report_type": _tool}
        monkeypatch.setattr(module, reader_name, collect)
        report = lab.execute_tool("tool", {"tool": tool, "parameters": {key: value}}, tmp_path)
        assert report["report_type"] == tool and calls[0]["fetch"] is True
        assert calls[0][key] == value
        with pytest.raises(ValueError):
            lab.ToolInput(tool=tool, parameters={key: []})
        with pytest.raises(ValueError):
            lab.ToolInput(tool=tool, parameters={"since": "2025-12-01", "until": "2025-01-01"})


def test_etf_ratios_return_window_and_completed_sessions(monkeypatch):
    from nihai.data import alpaca_history as ah
    from nihai.data.market_calendar import NYSE

    sessions = NYSE.sessions.tz_localize(None)
    days = sessions[(sessions >= "2025-01-02") & (sessions <= "2025-03-31")][:25]
    prices = pd.DataFrame({"GLD": 200., "USO": 50., "CPER": 25., "HYG": 80., "LQD": 100.,
                           "BIL": 100 * 1.0001 ** np.arange(len(days))}, index=days)
    measured = mr.etf_indicators(prices)
    assert measured.iloc[-1].copper_gold_etf == .125
    assert measured.iloc[-1].gold_oil_etf == 4
    assert measured.iloc[-1].hy_ig_etf == .8
    assert measured.bil_return20_ann.iloc[:20].isna().all()
    assert measured.bil_return20_ann.iloc[20] == pytest.approx(1.0001 ** 252 - 1)
    changed = prices.copy()
    changed.iloc[-1] *= 10
    pd.testing.assert_frame_equal(mr.etf_indicators(changed).iloc[:-1], measured.iloc[:-1])
    invalid = prices.copy()
    invalid.loc[days[-1], ["GLD", "BIL"]] = [0, np.inf]
    assert mr.etf_indicators(invalid).iloc[-1][["copper_gold_etf", "gold_oil_etf", "bil_return20_ann"]].isna().all()
    assert mr.etf_indicators(None).empty
    assert mr.etf_indicators(prices.iloc[::-1]).empty

    calls = []
    def history(symbol, interval, **kwargs):
        calls.append((symbol, kwargs))
        return prices[[symbol]].rename(columns={symbol: "close"})
    monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", "all")
    monkeypatch.setattr(ah, "get_history", history)
    raw = mr.etf_proxies(since="2025-01-01", today=days[-1], fetch=False, lag=False)
    pd.testing.assert_frame_equal(raw, measured.iloc[:-1])
    lagged = mr.etf_proxies(since="2025-01-01", today=days[-1], fetch=False)
    pd.testing.assert_index_equal(lagged.index, raw.index + pd.Timedelta(days=1))
    assert all(not opts["fetch"] for _, opts in calls)
    monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", "raw")
    before = len(calls)
    assert mr.etf_proxies(fetch=True).empty and len(calls) == before


def test_macro_cache_contract_expired_values_and_available_dates(monkeypatch, tmp_path):
    from nihai.data import alpaca_history as ah
    from nihai.web import api_quant as aq

    monkeypatch.setattr(ah, "HIST_DIR", tmp_path / "history")
    monkeypatch.setattr(ah, "_fetch", lambda *a, **k: pytest.fail("cache-only opened network"))
    monkeypatch.setenv("ALPACA_HIST_ADJUSTMENT", "all")
    assert ah.get_history("BIL", start="2025-01-01", fetch=False).empty
    assert not ah.HIST_DIR.exists()
    ah.HIST_DIR.mkdir()
    path = ah._cache_path("BIL", "1d")
    pd.DataFrame({"close": [100., 101., 102.]}, index=pd.date_range("2025-01-02", periods=3)).to_csv(path)
    before = path.read_bytes()
    bars = ah.get_history("BIL", start="2025-01-03", end="2025-01-03", fetch=False, use_cache=False)
    assert len(bars) == 1 and bars.close.iloc[0] == 101
    assert path.read_bytes() == before
    calls = []
    def fred(series, **kwargs):
        calls.append(kwargs)
        return pd.Series([1.], index=pd.DatetimeIndex(["2025-02-10"]))
    monkeypatch.setattr(mr.fred_data, "get", fred)
    panel = mr.panel(since="2025-01-01", today=pd.Timestamp("2025-02-11"), fetch=False)
    assert len(calls) > 10 and all(not opts["fetch"] for opts in calls)
    assert panel.index.max() <= pd.Timestamp("2025-02-11")

    def sparse(**kwargs):
        return pd.DataFrame({"a": [2., np.nan], "b": [3., 4.]},
                            index=pd.DatetimeIndex(["2025-01-02", "2025-01-15"]))
    monkeypatch.setattr(mr, "PANEL_BUILDERS", {"a": (sparse, "a", 5), "b": (sparse, "b", 5)})
    summary = mr.summary(fetch=False)
    assert summary["degerler"]["a"] is None and summary["degerler"]["b"] == 4
    assert summary["source_dates"]["a"] == "2025-01-02"
    report = aq.quant_macro()
    assert report["gostergeler"]["a"]["son"] is None
    assert report["gostergeler"]["a"]["tarih"] == "2025-01-02"


# --- Tur-78: SEC 8-K/6-K bültenlerinden PDUFA takvimi (ağsız) ---------------------------------
PDUFA_METIN = (
    "<p>Acme Bio, Inc. (ACME) announced the FDA accepted the BLA and assigned a PDUFA target action "
    "date of November&nbsp;30, 2026.</p> The company set an updated PDUFA target action date from "
    "September 27, 2026 to December 27, 2026. A PDUFA date in the first half of 2027 is expected for X. "
    "Target PDUFA Date in 2028 Conference Call on Tuesday, 8:30 a.m. ET Malvern, Pa., August 4, 2026")


def test_pdufa_mentions_hassasiyet_revizyon_ve_olculmus_yanlis_pozitif():
    rows = [(r["precision"], r["period"], r["pdufa_date"], r["revised"]) for r in fda.pdufa_mentions(PDUFA_METIN)]
    assert rows == [("day", "2026-11-30", "2026-11-30", False),     # HTML/&nbsp; temizlenir
                    ("day", "2026-12-27", "2026-12-27", True),      # "from X to Y" → YENİ tarih
                    ("half", "2027-H1", None, False),
                    ("year", "2028", None, False)]                  # bülten tarih satırı (Aug 4) alınmaz
    assert [r["period"] for r in fda.pdufa_mentions("PDUFA date expected in the fourth quarter of 2026; Q1 2027")] == ["2026-Q4"]
    for bos in ("PDUFA goal of November 14, 2026",                   # bağlaçta "date" yok
                "No PDUFA. The meeting is on May 1, 2026",           # cümle sonu
                "PDUFA date of February 30, 2027",                   # takvimde olmayan gün
                None, 123, ""):
        assert fda.pdufa_mentions(bos) == []


def test_pdufa_upcoming_yalniz_bilinen_ve_gelecek_gun_tarihleri():
    from datetime import date
    m = [dict(ticker="AAA", precision="day", pdufa_date="2026-10-01", file_date="2026-06-01", revised=False, context="a"),
         dict(ticker="AAA", precision="day", pdufa_date="2026-10-01", file_date="2026-07-01", revised=False, context="b"),
         dict(ticker="AAA", precision="day", pdufa_date="2026-12-01", file_date="2026-08-01", revised=True, context="c"),
         dict(ticker="BBB", precision="day", pdufa_date="2026-05-01", file_date="2026-04-01", revised=False),
         dict(ticker="CCC", precision="day", pdufa_date="2026-11-01", file_date="2026-09-20", revised=False),
         dict(ticker="DDD", precision="quarter", period="2026-Q4", pdufa_date=None, file_date="2026-06-01")]
    u = fda._upcoming(m, date(2026, 9, 15))
    assert [(r["ticker"], r["pdufa_date"]) for r in u["upcoming"]] == [("AAA", "2026-10-01"), ("AAA", "2026-12-01")]
    ekim = u["upcoming"][0]
    assert (ekim["n_mentions"], ekim["first_seen"], ekim["last_seen"], ekim["context"]) == (2, "2026-06-01", "2026-07-01", "b")
    assert ekim["possibly_superseded"] and not u["upcoming"][1]["possibly_superseded"]
    once = fda._upcoming(m, date(2026, 7, 15))                        # revizyon henüz dosyalanmamış
    assert [r["pdufa_date"] for r in once["upcoming"]] == ["2026-10-01"] and not once["upcoming"][0]["possibly_superseded"]


def test_pdufa_calendar_onbellek_ve_yenileme_hatasi(monkeypatch, tmp_path):
    from nihai.data import insider
    monkeypatch.setattr(fda, "CACHE_DIR", tmp_path / "fda")
    calls = []
    hit = {"_id": "0001234567-26-000001:acme8k.htm",
           "_source": {"ciks": ["0000000001"], "file_date": "2026-08-06", "form": "8-K",
                       "display_names": ["Acme Bio, Inc. (ACME) (CIK 0000000001)"]}}

    def fake_get(url, *, as_json=True, timeout=15.0):
        calls.append(url)
        return {"hits": {"total": {"value": 1}, "hits": [hit]}} if url.startswith(fda.EFTS_URL) else PDUFA_METIN

    kw = dict(since="2026-06-01", until="2026-09-14", as_of="2026-09-14")
    assert fda.pdufa_calendar(**kw) is None and not calls             # fetch=False ağ açmaz
    monkeypatch.setattr(insider, "_get", fake_get)
    r = fda.pdufa_calendar(**kw, fetch=True)
    assert r["cache_state"] == "downloaded" and (r["n_filings_read"], r["truncated"]) == (1, False)
    assert calls[1].endswith("/edgar/data/1/000123456726000001/acme8k.htm")
    assert {(x["ticker"], x["company"]) for x in r["mentions"]} == {("ACME", "Acme Bio, Inc.")}
    assert [u["pdufa_date"] for u in r["upcoming"]] == ["2026-11-30", "2026-12-27"]
    saved = next(fda.CACHE_DIR.glob("pdufa_*.json")).read_bytes()
    sonra = fda.pdufa_calendar(**{**kw, "as_of": "2026-12-01"})       # önbellekten, as_of'a göre yeniden süzülür
    assert sonra["cache_state"] == "fresh" and [u["pdufa_date"] for u in sonra["upcoming"]] == ["2026-12-27"]
    monkeypatch.setattr(insider, "_get", lambda *a, **k: None)
    stale = fda.pdufa_calendar(**kw, fetch=True, force=True)
    assert stale["cache_state"] == "stale" and stale["refresh_failed"]
    assert next(fda.CACHE_DIR.glob("pdufa_*.json")).read_bytes() == saved
    assert fda.pdufa_calendar(since="2026-09-01", until="2026-06-01") is None


def test_pdufa_lab_araci(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(fda, "pdufa_calendar", lambda **kw: calls.append(kw) or {"ok": True, "report_type": "pdufa_calendar"})
    report = lab.execute_tool("tool", {"tool": "pdufa_calendar",
                                       "parameters": {"since": "2026-06-01", "until": "2026-09-14"}}, tmp_path)
    assert report["report_type"] == "pdufa_calendar"
    assert calls == [{"since": "2026-06-01", "until": "2026-09-14", "force": False, "fetch": True}]
    with pytest.raises(ValueError):
        lab.ToolInput(tool="pdufa_calendar", parameters={"since": "2026-12-01", "until": "2026-01-01"})

# --- PDUFA sözleşmesi: 24 düşman vaka; #4, #6 ve #13 beklentileri düzeltildi. ---
@pytest.mark.parametrize("metin,beklenen", [
    pytest.param(
        'The PDUFA target action date is November 30, 2026.',
        [('day', '2026-11-30', '2026-11-30', False)], id="vaka-01"),
    pytest.param(
        'The pdufa target action date is November 30, 2026.',
        [], id="vaka-02"),
    pytest.param(
        'The PDUFA goal is November 30, 2026.',
        [], id="vaka-03"),
    pytest.param(
        'The PDUFA update is scheduled for November 30, 2026.',
        [], id="vaka-04"),
    pytest.param(
        'PDUFA target date is Nov.&nbsp;30, 2026',
        [('day', '2026-11-30', '2026-11-30', False)], id="vaka-05"),
    pytest.param(
        'The PDUFA action date for the drug developed by Genmab Inc. (Nasdaq: GMAB) is November 30, '
        '2026.',
        [('day', '2026-11-30', '2026-11-30', False)], id="vaka-06"),
    pytest.param(
        'The PDUFA action date was extended from November 30, 2026 to December 15, 2026.',
        [('day', '2026-12-15', '2026-12-15', True)], id="vaka-07"),
    pytest.param(
        'The PDUFA action date was changed from the original November 30, 2026 to December 15, 2026.',
        [('day', '2026-11-30', '2026-11-30', False)], id="vaka-08"),
    pytest.param(
        'The PDUFA target action date is revised to November 30, 2026.',
        [('day', '2026-11-30', '2026-11-30', True)], id="vaka-09"),
    pytest.param(
        'The PDUFA action date for Drug A is November 30, 2026 and for Drug B is December 15, 2026.',
        [('day', '2026-11-30', '2026-11-30', False)], id="vaka-10"),
    pytest.param(
        'PDUFA date: Sept. 3, 2026',
        [('day', '2026-09-03', '2026-09-03', False)], id="vaka-11"),
    pytest.param(
        'At this time, no PDUFA date has been assigned by the FDA, but we expect one in 2026.',
        [('year', '2026', None, False)], id="vaka-12"),
    pytest.param(
        'The PDUFA target action date will be discussed at the conference on November 30, 2026 at '
        '9:00 a.m. EST.',
        [], id="vaka-13"),
    pytest.param(
        'The PDUFA meeting at 9:00 a.m. will decide the date of November 30, 2026.',
        [], id="vaka-14"),
    pytest.param(
        'The PDUFA target date is in 1H 2027.',
        [], id="vaka-15"),
    pytest.param(
        'The PDUFA target date is mid-2027.',
        [], id="vaka-16"),
    pytest.param(
        'The PDUFA target date is year-end 2026.',
        [], id="vaka-17"),
    pytest.param(
        'The PDUFA target action date was October 15, 2020.',
        [('day', '2020-10-15', '2020-10-15', False)], id="vaka-18"),
    pytest.param(
        'While the EMA CHMP opinion is expected in Q3 2026, the FDA PDUFA action date is Q4 2026.',
        [('quarter', '2026-Q4', None, False)], id="vaka-19"),
    pytest.param(
        'The PDUFA target action date is February 30, 2026.',
        [], id="vaka-20"),
    pytest.param(
        'The PDUFA target action date is the first quarter of 2026.',
        [('quarter', '2026-Q1', None, False)], id="vaka-21"),
    pytest.param(
        'The PDUFA target action date is the second half of 2026.',
        [('half', '2026-H2', None, False)], id="vaka-22"),
    pytest.param(
        'The PDUFA target action date is by 2026.',
        [('year', '2026', None, False)], id="vaka-23"),
    pytest.param(
        'The PDUFA target date. is November 30, 2026.',
        [('day', '2026-11-30', '2026-11-30', False)], id="vaka-24"),
])
def test_pdufa_sozlesme_24_vaka(metin, beklenen):
    assert [(r["precision"], r["period"], r["pdufa_date"], r["revised"])
            for r in fda.pdufa_mentions(metin)] == beklenen


# Gerçek EDGAR bağlamları: sıra numaraları kaynak JSON içinde 1 tabanlıdır.
@pytest.mark.parametrize("metin,beklenen", [
    pytest.param(
        'jor safety or efficacy concerns to date and does not plan to request an advisory committee '
        'meeting; PDUFA date of January 29, 2027 Mid-cycle meeting completed f',
        [('day', '2027-01-29', '2027-01-29', False)], id="mentions-12"),
    pytest.param(
        ' expected - Corresponding look at OS with PFS HARMONi: - Updated conference presentation '
        '(OS) - BLA PDUFA Date November 14, 2026 New Clinical Trials 1H27 HARM',
        [('day', '2026-11-14', '2026-11-14', False)], id="mentions-55"),
    pytest.param(
        'rowth factor receptor 2 (HER2)-positive metastatic gastroesophageal adenocarcinoma, and '
        'granted Priority Review, with a PDUFA date of August 25, 2026. Financial Results Call '
        'Royalty Pharma will host a conference call and simultaneous webcast to discuss its second '
        'quarter of 2026 results today at 8:00 a.m., Eastern Time. Please visit the “Inves',
        [('day', '2026-08-25', '2026-08-25', False)], id="proto-86"),
])
def test_pdufa_gercek_baglam_regresyon(metin, beklenen):
    assert [(r["precision"], r["period"], r["pdufa_date"], r["revised"])
            for r in fda.pdufa_mentions(metin)] == beklenen


def test_pdufa_sirket_kisaltmalari_ve_sozcuk_sinirlari():
    for kisaltma in ("Inc.", "Corp.", "Ltd.", "Co.", "plc.", "plc", "N.V.", "S.A.", "INC."):
        metin = f"PDUFA action dates for Acme {kisaltma} (ACME) are November 30, 2026."
        assert [r["pdufa_date"] for r in fda.pdufa_mentions(metin)] == ["2026-11-30"]
    for olay in ("conference", "call", "webcast", "meeting", "presentation", "event", "discussed"):
        assert fda.pdufa_mentions(f"PDUFA date {olay.upper()} on November 30, 2026") == []
    for sinir in (". A", ". (", ";", "!", "?", " a.m.", " p.m.", " Zinc. A"):
        assert fda.pdufa_mentions(f"PDUFA date{sinir} November 30, 2026") == []
    # Alt-dizgiler etkinlik değildir; gerçek date sözcüğü korunur.
    assert len(fda.pdufa_mentions("PDUFA updated date for recall prevention is November 30, 2026")) == 1


def test_pdufa_etkinlik_sozcugu_yalniz_tarihe_bitisikse_reddeder():
    """Tur-78 inceleme: etkinlik sözcüğünü arada metnin HERHANGİ bir yerinde reddetmek gerçek PDUFA
    tarihlerini düşürüyordu (FDA yazışmasında "Type A meeting", "in the event of a major amendment")."""
    gercek = {
        "The PDUFA target action date, in the event of a major amendment, may be extended to December 1, 2026.":
            [("day", "2026-12-01", True)],
        "The PDUFA target action date, following a Type A meeting with the FDA, is December 1, 2026.":
            [("day", "2026-12-01", False)],
        "The PDUFA date, as discussed on our earnings call, is December 1, 2026.": [("day", "2026-12-01", False)],
    }
    for metin, beklenen in gercek.items():
        assert [(r["precision"], r["period"], r["revised"]) for r in fda.pdufa_mentions(metin)] == beklenen
    for etkinlik in ("The PDUFA date will be discussed during a call on May 5, 2026.",
                     "The PDUFA date will be discussed at the conference on November 30, 2026 at 9:00 a.m."):
        assert fda.pdufa_mentions(etkinlik) == []
