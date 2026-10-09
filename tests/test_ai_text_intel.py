"""nihai.ai.text_intel testleri — LM ton skoru, yenilik (BoW kosinüs), başlık-yoğunluğu
z-skoru, LLM fail-soft dürüstlük guard'ı, market_calendar (FOMC/CPI/NFP/OPEX/çeyrek-sonu/
bilanço-yakınlığı) tz-güvenli hesapları.

Ağ YOK: router.complete monkeypatch'lenir; nihai.data.event_calendar.get_company_events
monkeypatch'lenir (gerçek yfinance/Nasdaq çağrısı YAPILMAZ). TEXT_SCORES_FILE tmp_path'e
yönlendirilir (episodic.py/decision.py test deseniyle aynı).
"""
from __future__ import annotations

import calendar as _calendar_mod
import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

from nihai.ai import router, text_intel


@pytest.fixture(autouse=True)
def izole(tmp_path, monkeypatch):
    monkeypatch.setattr(text_intel, "TEXT_SCORES_FILE", tmp_path / "ai_text_scores.jsonl")
    yield


# ---------------------------------------------------------------------------
# tone_score — Loughran-McDonald çekirdek alt-küme
# ---------------------------------------------------------------------------
def test_tone_score_bilinen_pozitif_cumle():
    out = text_intel.tone_score("The company reported excellent growth and strong profitable results")
    assert out["positive"] >= 4
    assert out["negative"] == 0
    assert out["tone_label"] == "pozitif"
    assert out["net_tone"] > 0


def test_tone_score_bilinen_negatif_cumle():
    out = text_intel.tone_score("The firm faces a severe crisis with declining revenue and bankruptcy risk")
    assert out["negative"] >= 4
    assert out["tone_label"] == "negatif"
    assert out["net_tone"] < 0


def test_tone_score_belirsizlik_ve_litigious_sayilir():
    out = text_intel.tone_score(
        "There is uncertainty about the pending litigation and the lawsuit filed by the plaintiff")
    assert out["uncertainty"] >= 1
    assert out["litigious"] >= 3


def test_tone_score_bos_metin_notr_cokmez():
    out = text_intel.tone_score("")
    assert out == {"positive": 0, "negative": 0, "uncertainty": 0, "litigious": 0,
                    "total_words": 0, "net_tone": 0.0, "tone_label": "nötr"}


def test_tone_score_str_olmayan_girdi_cokmez():
    out = text_intel.tone_score(None)  # type: ignore[arg-type]
    assert out["total_words"] == 0
    assert out["tone_label"] == "nötr"


def test_tone_score_dengeli_karisik_notr():
    out = text_intel.tone_score("good weak")  # pozitif+negatif dengede, net~0
    assert out["positive"] == 1 and out["negative"] == 1
    assert out["tone_label"] == "nötr"


def test_tone_score_buyuk_kucuk_harf_duyarsiz():
    a = text_intel.tone_score("EXCELLENT GROWTH")
    b = text_intel.tone_score("excellent growth")
    assert a == b
    assert a["positive"] == 2


def test_lm_kategorileri_ayrik():
    """Dört kelime kümesi KESİŞMEZ (gerçek LM kategorileri de birbirini dışlar)."""
    sets = [text_intel._LM_POSITIVE, text_intel._LM_NEGATIVE,
            text_intel._LM_UNCERTAINTY, text_intel._LM_LITIGIOUS]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j]), f"kategori {i} ve {j} kesişiyor"


# ---------------------------------------------------------------------------
# novelty_score — BoW kosinüs
# ---------------------------------------------------------------------------
def test_novelty_score_onceki_metin_yoksa_tam_yeni():
    assert text_intel.novelty_score("Apple reports record earnings", []) == 1.0


def test_novelty_score_ayni_metin_tekrari_dusuk():
    metin = "Apple reports record quarterly earnings beating analyst expectations widely"
    out = text_intel.novelty_score(metin, [metin])
    assert out < 0.05   # neredeyse aynı → yenilik ~0


def test_novelty_score_tamamen_farkli_metin_yuksek():
    out = text_intel.novelty_score("Apple reports record earnings this quarter",
                                    ["Weather forecast predicts rain tomorrow in Chicago"])
    assert out > 0.8


def test_novelty_score_bos_metin_sifir():
    assert text_intel.novelty_score("", ["some prior text here"]) == 0.0


def test_novelty_score_en_yakina_gore_hesaplanir():
    metin = "the company announced a new product launch today"
    priors = ["completely unrelated content about sports and weather",
              "the company announced a new product launch today with fanfare"]
    out = text_intel.novelty_score(metin, priors)
    # en yakın (2. prior, neredeyse aynı) benzerliği kullanır → düşük yenilik
    assert out < 0.3


def test_novelty_score_deterministik():
    metin = "quarterly revenue increased significantly this year"
    priors = ["some other unrelated news article about politics"]
    assert text_intel.novelty_score(metin, priors) == text_intel.novelty_score(metin, priors)


def test_novelty_score_gecersiz_prior_elemanlari_atlanir():
    out = text_intel.novelty_score("hello world", [None, 123, "", "   "])  # type: ignore[list-item]
    assert out == 1.0   # geçerli hiçbir prior yok → tam-yeni varsayımı


# ---------------------------------------------------------------------------
# headline_density — z-skor
# ---------------------------------------------------------------------------
def test_headline_density_bos_dict():
    assert text_intel.headline_density({}) == {}


def test_headline_density_esit_sayida_hepsi_sifir():
    out = text_intel.headline_density({"AAPL": 5, "MSFT": 5, "NVDA": 5})
    assert out == {"AAPL": 0.0, "MSFT": 0.0, "NVDA": 0.0}


def test_headline_density_bilinen_ornek_manuel_dogrulama():
    counts = {"AAPL": 10, "MSFT": 2, "NVDA": 2}
    out = text_intel.headline_density(counts)
    m = (10 + 2 + 2) / 3
    var = ((10 - m) ** 2 + (2 - m) ** 2 + (2 - m) ** 2) / 3
    sd = var ** 0.5
    assert out["AAPL"] == pytest.approx(round((10 - m) / sd, 6))
    assert out["AAPL"] > out["MSFT"] == out["NVDA"]


def test_headline_density_sayisal_olmayan_deger_o_sembol_sifir():
    out = text_intel.headline_density({"AAPL": 10, "BAD": "yok", "MSFT": 4})
    assert out["BAD"] == 0.0
    assert out["AAPL"] != 0.0


# ---------------------------------------------------------------------------
# analyze_text — LLM fail-soft dürüstlük guard'ı + kalıcılık
# ---------------------------------------------------------------------------
def test_analyze_text_llm_yokken_deterministic_dolu_llm_none(monkeypatch):
    monkeypatch.setattr(router, "complete", lambda *a, **k: None)
    rec = text_intel.analyze_text("The company reported strong growth", symbol="aapl")
    assert rec["llm"] is None
    assert rec["deterministic"]["tone"]["positive"] >= 1
    assert rec["symbol"] == "AAPL"


def test_analyze_text_llm_varken_yan_yana_raporlanir(monkeypatch):
    yapay_cikti = ("TON: pozitif\nYÖN: olumlu\nBELIRSIZLIK: düşük\n"
                    "GEREKCE: Güçlü büyüme rakamları.")
    monkeypatch.setattr(router, "complete", lambda *a, **k: yapay_cikti)
    rec = text_intel.analyze_text("The company reported strong growth")
    assert rec["llm"] is not None
    assert rec["llm"]["tone"] == "pozitif"
    assert rec["llm"]["direction"] == "olumlu"
    # dürüstlük guard'ı: LLM'in varlığı deterministik alanı DEĞİŞTİRMEZ
    assert rec["deterministic"]["tone"] == text_intel.tone_score("The company reported strong growth")


def test_analyze_text_llm_ayristirilamaz_cikti_none(monkeypatch):
    monkeypatch.setattr(router, "complete", lambda *a, **k: "alakasız serbest metin cevap")
    rec = text_intel.analyze_text("some headline text here")
    assert rec["llm"] is None
    assert rec["deterministic"] is not None


def test_analyze_text_router_patlarsa_fail_soft(monkeypatch):
    def patlayan(*a, **k):
        raise RuntimeError("ağ hatası")
    monkeypatch.setattr(router, "complete", patlayan)
    rec = text_intel.analyze_text("company news headline")
    assert rec["llm"] is None
    assert rec["deterministic"]["tone"]["total_words"] > 0


def test_analyze_text_bos_metin_llm_hic_cagrilmaz(monkeypatch):
    cagrildi = []
    monkeypatch.setattr(router, "complete", lambda *a, **k: cagrildi.append(1) or "TON: pozitif")
    rec = text_intel.analyze_text("")
    assert cagrildi == []
    assert rec["llm"] is None


def test_analyze_text_persist_true_dosyaya_yazar(monkeypatch):
    monkeypatch.setattr(router, "complete", lambda *a, **k: None)
    text_intel.analyze_text("haber metni", symbol="tsla", persist=True)
    assert text_intel.TEXT_SCORES_FILE.exists()
    satirlar = text_intel.TEXT_SCORES_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(satirlar) == 1
    kayit = json.loads(satirlar[0])
    assert kayit["symbol"] == "TSLA"


def test_analyze_text_persist_false_yazmaz(monkeypatch):
    monkeypatch.setattr(router, "complete", lambda *a, **k: None)
    text_intel.analyze_text("haber metni", persist=False)
    assert not text_intel.TEXT_SCORES_FILE.exists()


def test_analyze_text_ts_utc_formatinda(monkeypatch):
    monkeypatch.setattr(router, "complete", lambda *a, **k: None)
    rec = text_intel.analyze_text("x", persist=False)
    assert datetime.fromisoformat(rec["ts"]).tzinfo is not None


# ---------------------------------------------------------------------------
# market_calendar — next_macro_events / opex_week / quarter_end (tz-güvenli, bağımsız
# referans hesabıyla doğrulanır — text_intel'in kendi mantığını KOPYALAMAZ)
# ---------------------------------------------------------------------------
def _ref_first_friday(year: int, month: int) -> date:
    cal = _calendar_mod.monthcalendar(year, month)
    week0 = cal[0]
    if week0[_calendar_mod.FRIDAY] != 0:
        return date(year, month, week0[_calendar_mod.FRIDAY])
    return date(year, month, cal[1][_calendar_mod.FRIDAY])


def _ref_third_friday(year: int, month: int) -> date:
    cal = _calendar_mod.monthcalendar(year, month)
    fridays = [week[_calendar_mod.FRIDAY] for week in cal if week[_calendar_mod.FRIDAY] != 0]
    return date(year, month, fridays[2])


def test_next_macro_events_artan_sirali_ve_days_away_negatif_degil():
    out = text_intel.next_macro_events(date(2026, 1, 1), n=5)
    assert len(out) == 5
    tarihler = [datetime.fromisoformat(e["date"]).date() for e in out]
    assert tarihler == sorted(tarihler)
    assert all(e["days_away"] >= 0 for e in out)


def test_next_macro_events_n_sifir_bos():
    assert text_intel.next_macro_events(date(2026, 1, 1), n=0) == []


def test_next_macro_events_nfp_ilk_cuma_kurali():
    """NFP olayları bağımsız referans (`calendar` modülü) ile hesaplanan 'ayın ilk Cuma
    günü' ile birebir eşleşmeli."""
    out = text_intel.next_macro_events(date(2026, 8, 1), n=40)
    nfp_dates = {datetime.fromisoformat(e["date"]).date() for e in out if e["event"] == "NFP"}
    assert _ref_first_friday(2026, 8) in nfp_dates
    assert _ref_first_friday(2026, 9) in nfp_dates


def test_next_macro_events_fomc_2026_tablosunda():
    out = text_intel.next_macro_events(date(2026, 1, 1), n=50)
    fomc_dates = {datetime.fromisoformat(e["date"]).date() for e in out if e["event"] == "FOMC"}
    assert len(fomc_dates) == 8   # 2026 için 8 toplantı (statik tablo)


@pytest.mark.parametrize("year,month", [(2026, 1), (2026, 7), (2026, 12), (2027, 2)])
def test_opex_week_referansla_tutarli(year, month):
    friday3 = _ref_third_friday(year, month)
    assert text_intel.opex_week(friday3) is True
    monday = friday3 - timedelta(days=friday3.weekday())
    for offset in range(7):
        assert text_intel.opex_week(monday + timedelta(days=offset)) is True
    assert text_intel.opex_week(monday - timedelta(days=1)) is False
    assert text_intel.opex_week(monday + timedelta(days=7)) is False


def test_opex_week_datetime_girdi_kabul_eder():
    friday3 = _ref_third_friday(2026, 7)
    dt = datetime(friday3.year, friday3.month, friday3.day, 14, 30, tzinfo=UTC)
    assert text_intel.opex_week(dt) is True


def test_quarter_end_pencere_icinde_ve_disinda():
    assert text_intel.quarter_end(date(2026, 3, 31)) is True
    assert text_intel.quarter_end(date(2026, 3, 29)) is True
    assert text_intel.quarter_end(date(2026, 3, 28)) is False
    assert text_intel.quarter_end(date(2026, 3, 15)) is False


def test_quarter_end_ceyrek_disi_ay_her_zaman_false():
    assert text_intel.quarter_end(date(2026, 1, 31)) is False
    assert text_intel.quarter_end(date(2026, 7, 31)) is False


def test_quarter_end_aralik_yil_sonu():
    assert text_intel.quarter_end(date(2026, 12, 31)) is True
    assert text_intel.quarter_end(date(2026, 12, 20)) is False


def test_quarter_end_ozel_pencere():
    assert text_intel.quarter_end(date(2026, 6, 25), window_days=10) is True
    assert text_intel.quarter_end(date(2026, 6, 25), window_days=1) is False


# ---------------------------------------------------------------------------
# earnings_proximity — nasdaq-earnings köprüsü (mock; ağ YOK)
# ---------------------------------------------------------------------------
def test_earnings_proximity_basarili_gunu_dogru_hesaplar(monkeypatch):
    from nihai.data import event_calendar
    fake = SimpleNamespace(next_earnings=date(2026, 8, 15), timing="BMO", source="yfinance")
    monkeypatch.setattr(event_calendar, "get_company_events", lambda sym: fake)
    out = text_intel.earnings_proximity("AAPL", date(2026, 8, 10))
    assert out is not None
    assert out["days_to_earnings"] == 5
    assert out["timing"] == "BMO"
    assert out["next_earnings"] == "2026-08-15"


def test_earnings_proximity_gecmis_tarihe_gore_hesaplar_bugune_gore_degil(monkeypatch):
    """days_to_earnings VERİLEN `d`'ye göre hesaplanmalı, date.today()'e göre DEĞİL —
    look-ahead'siz replay/test edilebilirlik şartı."""
    from nihai.data import event_calendar
    fake = SimpleNamespace(next_earnings=date(2020, 1, 20), timing=None, source="yfinance")
    monkeypatch.setattr(event_calendar, "get_company_events", lambda sym: fake)
    out = text_intel.earnings_proximity("AAPL", date(2020, 1, 10))
    assert out["days_to_earnings"] == 10


def test_earnings_proximity_next_earnings_none(monkeypatch):
    from nihai.data import event_calendar
    fake = SimpleNamespace(next_earnings=None, timing=None, source="yfinance")
    monkeypatch.setattr(event_calendar, "get_company_events", lambda sym: fake)
    out = text_intel.earnings_proximity("AAPL", date(2026, 1, 1))
    assert out == {"symbol": "AAPL", "date": "2026-01-01", "timing": None,
                    "source": "yfinance", "next_earnings": None, "days_to_earnings": None}


def test_earnings_proximity_kopru_hata_verirse_none(monkeypatch):
    from nihai.data import event_calendar

    def patlayan(sym):
        raise RuntimeError("ağ hatası / yfinance kullanılamıyor")
    monkeypatch.setattr(event_calendar, "get_company_events", patlayan)
    assert text_intel.earnings_proximity("AAPL", date(2026, 1, 1)) is None


@pytest.mark.parametrize("exc", [ImportError("modül yok"), RuntimeError("ağ"), AttributeError("şema")])
def test_earnings_proximity_kopru_herhangi_bir_hatada_none(monkeypatch, exc):
    """Köprü ('bulunamaz' dahil — ImportError de bu genel try/except'in içindedir) HANGİ
    türden hata verirse versin fail-soft None (spec §3.13: 'bulamazsan fail-soft None')."""
    from nihai.data import event_calendar

    def patlayan(sym):
        raise exc
    monkeypatch.setattr(event_calendar, "get_company_events", patlayan)
    assert text_intel.earnings_proximity("AAPL", date(2026, 1, 1)) is None


# ---------------------------------------------------------------------------
# _to_utc_date + market_calendar namespace
# ---------------------------------------------------------------------------
def test_to_utc_date_naive_datetime_utc_varsayilir():
    naive = datetime(2026, 3, 20, 10, 0, 0)
    assert text_intel._to_utc_date(naive) == date(2026, 3, 20)


def test_to_utc_date_tz_aware_farkli_tz_donusturulur():
    from datetime import timezone
    tz_minus5 = timezone(timedelta(hours=-5))
    # 2026-03-20 23:30 UTC-5 → 2026-03-21 04:30 UTC (gün DEĞİŞİR)
    dt = datetime(2026, 3, 20, 23, 30, tzinfo=tz_minus5)
    assert text_intel._to_utc_date(dt) == date(2026, 3, 21)


def test_to_utc_date_plain_date_degismez():
    d = date(2026, 5, 5)
    assert text_intel._to_utc_date(d) is d


def test_market_calendar_namespace_ayni_fonksiyonlara_esdeger():
    d = date(2026, 7, 1)
    assert text_intel.market_calendar.opex_week(d) == text_intel.opex_week(d)
    assert text_intel.market_calendar.quarter_end(d) == text_intel.quarter_end(d)
    assert (text_intel.market_calendar.next_macro_events(d, 2)
            == text_intel.next_macro_events(d, 2))
